"""The rollout worker: M live fights, honest encoding on the CPU, actions from the server.

Protocol on the worker's duplex pipe (spawn context; worker code lives here,
never in `python -c`):

* worker -> server: `("obs", wid, keys, packed, ends)`. `keys` holds one
  `(slot, episode)` per decision in `packed`. `ends` holds a dict per episode
  that finished since the last message (`slot`, `ep`, `won`, `reward` terms,
  `room`, `encounter`, `steps`, `decisions`, `capped`, `error`).
* server -> worker: `("act", [action index per decision])` or `("stop",)`.

A fight is built with `rollout_starts.build_fight` (as `play.py` does) from a
fresh build seed per episode. Every step calls `Encoder.observe` (the intent log
needs single-option steps too). A single legal action is applied without asking
the server. A fight ends when there is no legal action or at `max_steps`
(capped = lost). An engine exception aborts only that episode: `error` is
recorded and the server drops its decisions.
"""

from __future__ import annotations

import os
import traceback
from typing import Any


def _won(cs: Any) -> bool:
    term, val = cs.terminal_value()
    return bool(term) and float(val) > 0


class Env:
    def __init__(self, slot: int) -> None:
        self.slot = slot
        self.ep = -1
        self.cs: Any = None
        self.row: dict[str, Any] = {}
        self.ilog: Any = None
        self.steps = 0
        self.decisions = 0


def worker_main(wid: int, conn: Any, cfg: dict[str, Any]) -> None:
    import faulthandler

    faulthandler.enable()  # a native crash (Rust abort, segfault) still leaves a trace
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sts2_rs

    from ppo import rollout_starts

    from .encode import Encoder, pack
    from .pool import StartSampler, is_own, own_pool, train_pool

    if cfg.get("fixed_rows"):
        pool = {"boss": [], "elite": [], "monster": []}
        for r in cfg["fixed_rows"]:
            pool[r["room_type"]].append(r)
    else:
        pool, _ = train_pool(cfg["hbank"], cfg["boss800"], cfg.get("extra_holdouts"))
    own = own_pool(cfg["own_pool"]) if cfg.get("own_pool") else None
    sampler = StartSampler(
        pool,
        cfg["weights"],
        seed=cfg["seed"] * 1_000_003 + wid,
        own=own,
        own_share=float(cfg.get("own_share", 0.0)),
    )
    enc = Encoder(cfg["arm"], cfg["rare"], cfg["priv_k"])
    max_steps = int(cfg["max_steps"])

    n_run = int(cfg.get("run_slots") or 0)
    envs = [Env(i) for i in range(int(cfg["n_envs"]) - n_run)]
    ends: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    runs: list[Any] = []
    cur: dict[str, Any] = {"ver": int(cfg.get("run_ver0") or 0), "pol": None, "loaded": -1}
    if n_run:
        import itertools

        import torch

        from ppo.net_policy import NetPolicy

        from .runslot import RunSlot

        torch.set_num_threads(1)

        def run_policy() -> Any:
            if cur["loaded"] != cur["ver"]:
                cur["pol"] = NetPolicy(os.path.join(cfg["run_acting_dir"], f"v{cur['ver']:05d}.pt"))
                cur["loaded"] = cur["ver"]
            return cur["pol"]

        rcfg = dict(cfg, _run_ver=lambda: cur["ver"])
        for k in range(n_run):
            slot = len(envs) + k
            base = int(cfg["run_seed_base"]) + (wid * 64 + k) * 100_000
            runs.append(RunSlot(slot, itertools.count(base), rcfg, enc, run_policy, ends, run_rows))

    def reset(e: Env) -> None:
        while True:
            row, seed = sampler.draw()
            try:
                e.cs = rollout_starts.build_fight(row, seed).combat_state()
            except ValueError:
                continue
            break
        e.row, e.ep = row, e.ep + 1
        e.ilog = enc.new_log()
        e.steps = e.decisions = 0

    def finish(e: Env, capped: bool, error: str | None) -> None:
        won = False if (capped or error) else _won(e.cs)
        hp_frac = removed = 0.0
        if not error:
            try:
                st = sts2_rs.rollout_features(e.cs, version=1)["state"]
                hp_frac, removed = float(st[0]), float(st[15])
            except Exception:  # a terminal state without features
                hp_frac, removed = 0.0, 0.0
        ends.append(
            {
                "slot": e.slot,
                "ep": e.ep,
                "won": won,
                "hp_frac": max(0.0, hp_frac) if won else 0.0,
                "removed": removed,
                "room": e.row["room_type"],
                "own": is_own(e.row),
                "encounter": e.row["encounter"],
                "steps": e.steps,
                "decisions": e.decisions,
                "capped": capped,
                "error": error,
            }
        )

    for e in envs:
        reset(e)
    while True:
        keys, ds, pending = [], [], []
        for e in envs:
            while True:
                try:
                    acts = e.cs.legal_actions()
                    if not acts or e.steps >= max_steps:
                        finish(e, capped=bool(acts), error=None)
                        reset(e)
                        continue
                    snap, bv = enc.observe(e.ilog, e.cs, e.row["room_type"])
                    if len(acts) == 1:
                        e.cs.apply(acts[0])
                        e.steps += 1
                        continue
                    d = enc.encode(snap, bv, e.cs, acts)
                except BaseException as ex:  # a Rust panic is a BaseException
                    if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                        raise
                    finish(e, capped=False, error=traceback.format_exc(limit=3))
                    reset(e)
                    continue
                keys.append((e.slot, e.ep))
                ds.append(d)
                pending.append((e, acts))
                break
        run_pending = []
        for r in runs:
            d = r.advance()
            keys.append((r.slot, r.ep))
            ds.append(d)
            run_pending.append(r)
        conn.send(("obs", wid, keys, pack(ds), ends, run_rows))
        # `ends`/`run_rows` are shared with the run slots: clear in place.
        ends.clear()
        run_rows.clear()
        msg = conn.recv()
        if msg[0] == "stop":
            conn.close()
            return
        if len(msg) > 2 and msg[2] is not None:
            cur["ver"] = int(msg[2])
        n_fight = len(pending)
        for r, a in zip(run_pending, msg[1][n_fight:], strict=True):
            r.apply(a)
        msg = (msg[0], msg[1][:n_fight])
        for (e, acts), a in zip(pending, msg[1], strict=True):
            try:
                e.cs.apply(acts[int(a)])
            except BaseException as ex:  # a Rust panic too
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                finish(e, capped=False, error=traceback.format_exc(limit=3))
                reset(e)
                continue
            e.steps += 1
            e.decisions += 1
