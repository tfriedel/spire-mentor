"""The PPO loop: inference server for the workers, the update ranks, logs.

The learner process owns the inference device. It collects decisions from every
worker that is ready, runs the acting policy (honest tokens) in one batched
no-grad forward, samples at T = 1, and sends the actions back. Each decision is
stored under its `(worker, slot, episode)` with its behaviour log-prob. When the
worker reports the episode's end, its trajectory gets the terminal reward and
becomes trainable. An iteration's batch is complete once `iter_decisions`
decisions of finished episodes are in hand. Episodes still open carry over.

The update runs in `ranks.py`: one process per update device, data-parallel.
The learner hands a finished batch to the ranks and goes straight back to
collecting with the policy it has. When the ranks finish, the new policy
replaces the acting one between two inference batches. At most one batch is in
flight: a batch completed while the previous update still runs waits for it.
So a batch is sampled by a policy at most one update old, which the ranks'
decoupled PPO objective accounts for.

Reward (rl2's shaped terminal reward, `rollout_learn.shaped_reward`): a win is
`1 + w_hp·hp_frac`, a loss (including capped fights) is `w_removed·removed`. All
intermediate rewards are 0, with γ = 1 and GAE λ.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from multiprocessing.connection import wait
from typing import Any

import numpy as np
import torch

from ppo.combat_ctx import tokens as T

from . import nets
from .actors import worker_main
from .encode import Decision, Packed, concat_packed, unpack
from .ranks import entropy_coef, gae, rank_main, to_shm

__all__ = ["Cfg", "Learner", "Step", "entropy_coef", "gae"]


@dataclass
class Cfg:
    init: str
    out: str
    hbank: str = ""
    boss800: str = ""
    extra_holdouts: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(
        default_factory=lambda: {"boss": 0.5, "elite": 0.3, "monster": 0.2}
    )
    fixed_rows: list[dict[str, Any]] = field(default_factory=list)  # tests: a fixed pool
    # Harvested own-deck fights (`combat_bench.own_decks harvest`) and their share.
    own_pool: list[str] = field(default_factory=list)
    own_share: float = 0.0
    workers: int = 40
    n_envs: int = 32
    priv_k: int = 10
    max_steps: int = 600
    seed: int = 0
    infer_device: str = "cuda:0"
    # One update rank per entry (data-parallel; the same device may repeat on CPU).
    update_devices: str = "cuda:1,cuda:2,cuda:3"
    budget_decisions: int = 1_000_000_000
    iter_decisions: int = 200_000
    epochs: int = 3
    minibatch: int = 4096
    clip: float = 0.2
    target_kl: float = 0.02
    gae_lambda: float = 0.95
    # Truncation of the behaviour -> proximal importance weight (decoupled PPO).
    is_trunc: float = 2.0
    lr: float = 1e-4
    lr_final_frac: float = 0.0
    critic_lr: float = 3e-4
    ent0: float = 0.01
    ent1: float = 0.001
    grad_clip: float = 1.0
    warmup_iters: int = 10
    w_hp: float = 0.3
    w_removed: float = 0.2
    bf16: bool = True
    ckpt_every: int = 25_000_000
    eval_every: int = 50_000_000
    eval_fights: str = ""
    eval_reps: int = 4
    eval_procs: int = 12
    max_iters: int = 0  # 0 = until the budget
    # A branch from another run's checkpoint starts its counters (and so its lr
    # and entropy schedules) at that run's decision count.
    init_decisions: int = 0
    # Rows per forward/backward chunk in the update (0 = a rank's whole share).
    micro_rows: int = 0
    # ── hybrid: full runs inside the workers (`runslot.py`, `runlearn.py`) ──
    run_slots: int = 0  # per worker, out of n_envs
    run_init: str = ""  # the run net's warm start (a `ppo.trainer` checkpoint)
    run_device: str = "cuda:0"
    run_games_per_iter: int = 960
    run_policy_lr: float = 1.5e-5
    run_value_lr: float = 1e-4
    run_lr_final: float = 0.3
    run_entropy: float = 0.0125
    run_warmup: int = 3
    run_ckpt_every: int = 25
    run_ascension: int = 10
    run_act1: str = "roll"
    run_character: str = "Ironclad"
    run_seed_base: int = 300_000_000
    # In-run fight reward: "fight" = the curriculum's (win 1 + w_hp·hp_frac, loss
    # w_removed·removed); "value" = run_reward_scale · (m_after − m_before), the
    # change of the run critic's expected terminal base value across the fight
    # (`runslot.RunSlot`), valued by the critic's own `value_head_run`.
    run_reward: str = "fight"
    run_reward_scale: float = 1.0
    # Value-only updates until this iteration (a branch's fresh run head).
    policy_frozen_until: int = 0
    # Full-run eval at every eval point: greedy run net + greedy combat net.
    fullrun_eval_seeds: int = 0
    fullrun_eval_seed_start: int = 154_000_000
    fullrun_eval_procs: int = 12


@dataclass
class Step:
    d: Decision
    action: int
    logp: float


def _release_queues(qs: list[Any]) -> None:
    """Let the process exit with data still buffered in `qs`.

    A `multiprocessing.Queue` joins its feeder thread at exit. When the reader
    died (an OOM-killed update rank) with a large batch still unread, that
    thread blocks on the full pipe forever, so the learner hung instead of
    exiting with the rank's error. Called after the reader is stopped or gone.
    """
    for q in qs:
        if q is not None:
            q.cancel_join_thread()


class Learner:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        os.makedirs(os.path.join(cfg.out, "ckpt"), exist_ok=True)
        self.dev_i = torch.device(cfg.infer_device)
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator(device=self.dev_i)
        self.gen.manual_seed(cfg.seed)
        policy, critic, self.meta = nets.load_init(cfg.init)
        self.it = 0
        self.decisions = 0
        self.episodes = 0
        self.last_ckpt = 0
        self.last_eval = -1
        p = self._state_path()
        if os.path.exists(p):
            st = torch.load(p, map_location="cpu", weights_only=True)
            policy.load_state_dict(st["policy"])
            self.it, self.decisions, self.episodes = st["it"], st["decisions"], st["episodes"]
            self.last_ckpt, self.last_eval = st["last_ckpt"], st["last_eval"]
        elif cfg.init_decisions:
            self.decisions = self.last_ckpt = self.last_eval = int(cfg.init_decisions)
            self.gen.manual_seed(cfg.seed + 7919 * self.it)
            print(f"[combat_ppo] resumed at it {self.it}, {self.decisions:,} decisions", flush=True)
        self._init_nets: tuple[Any, Any] | None = (policy, critic) if self.it == 0 else None
        # The acting policy; the update ranks own the trained replicas.
        self.pol_inf = (copy.deepcopy(policy) if self.it == 0 else policy).to(self.dev_i)
        self.pol_inf.eval()
        self.evals: list[subprocess.Popen] = []
        self.open: dict[tuple[int, int, int], list[Step]] = defaultdict(list)
        self.conns: list[Any] = []
        self.procs: list[Any] = []
        self.ranks: list[Any] = []
        self.rank_inq: list[Any] = []
        self.rank_outq: Any = None
        self.inflight: dict[str, Any] | None = None
        self._store_dir = ""
        self._t_row = time.time()
        self.run_ver = 0
        self.run_proc: Any = None
        self.run_inq: Any = None
        self.run_outq: Any = None
        self.run_rows_sent = 0

    # ── state ────────────────────────────────────────────────────────────

    def _state_path(self) -> str:
        return os.path.join(self.cfg.out, "state.pt")

    def _counters(self) -> dict[str, int]:
        return {
            "it": self.it,
            "decisions": self.decisions,
            "episodes": self.episodes,
            "last_ckpt": self.last_ckpt,
            "last_eval": self.last_eval,
        }

    # ── workers ──────────────────────────────────────────────────────────

    def _run_shaping(self) -> dict[str, float]:
        """The run net's `ShapingConfig` coefficients (its potential Φ)."""
        if not (self.cfg.run_slots and self.cfg.run_init):
            return {}
        c = torch.load(self.cfg.run_init, map_location="cpu", weights_only=True)["cfg"]
        return {
            "hp_coef": c["shaping_hp_coef"],
            "upg_coef": c["shaping_upg_coef"],
            "relic_coef": c["shaping_relic_coef"],
            "offset": c["shaping_offset"],
        }

    def start_workers(self) -> None:
        cfg = self.cfg
        wcfg = {
            "hbank": cfg.hbank,
            "boss800": cfg.boss800,
            "extra_holdouts": cfg.extra_holdouts,
            "weights": cfg.weights,
            "fixed_rows": cfg.fixed_rows,
            "own_pool": cfg.own_pool,
            "own_share": cfg.own_share,
            "n_envs": cfg.n_envs,
            "priv_k": cfg.priv_k,
            "max_steps": cfg.max_steps,
            # A resumed run draws fresh starts, not the first iterations' again.
            "seed": cfg.seed * 1009 + self.it,
            "arm": self.meta["arm"],
            "rare": self.meta["rare"],
            "run_slots": cfg.run_slots,
            "run_ascension": cfg.run_ascension,
            "run_act1": cfg.run_act1,
            "run_character": cfg.run_character,
            "run_seed_base": cfg.run_seed_base + 7 * self.it,
            "run_acting_dir": os.path.join(cfg.out, "run", "acting"),
            "run_ver0": self.run_ver,
            "run_reward": cfg.run_reward,
            "run_reward_scale": cfg.run_reward_scale,
            "run_shaping": self._run_shaping(),
        }
        ctx = get_context("spawn")
        for wid in range(cfg.workers):
            a, b = ctx.Pipe(duplex=True)
            p = ctx.Process(target=worker_main, args=(wid, b, wcfg), daemon=True)
            p.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p)

    def stop_workers(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        deadline = time.time() + 30
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.terminate()
        self.conns, self.procs = [], []

    # ── the run learner (hybrid) ─────────────────────────────────────────

    def start_run_learner(self) -> None:
        from .runlearn import init_acting, run_learner_main

        cfg = self.cfg
        if not os.path.exists(os.path.join(cfg.out, "run", "state.pt")):
            init_acting(cfg.out, cfg.run_init)
        ctx = get_context("spawn")
        self.run_inq, self.run_outq = ctx.Queue(), ctx.Queue()
        rcfg = {
            "out": cfg.out,
            "init": cfg.run_init,
            "device": cfg.run_device,
            "warmup": cfg.run_warmup,
            "ckpt_every": cfg.run_ckpt_every,
            "over": {
                "games_per_iter": cfg.run_games_per_iter,
                "policy_lr": cfg.run_policy_lr,
                "value_lr": cfg.run_value_lr,
                "lr_anneal_final": cfg.run_lr_final,
                "entropy_coef_final": cfg.run_entropy,
                "device": cfg.run_device,
            },
        }
        self.run_proc = ctx.Process(
            target=run_learner_main, args=(rcfg, self.run_inq, self.run_outq), daemon=True
        )
        self.run_proc.start()
        kind, ver, _ = self._run_get(block=True)
        if kind != "run_ready":
            raise RuntimeError(f"run learner: unexpected {kind!r} at start-up")
        self.run_ver = int(ver)
        print(f"[combat_ppo] run learner ready at version {self.run_ver}", flush=True)

    def _run_get(self, block: bool) -> tuple[str, int, Any]:
        while True:
            try:
                kind, ver, payload = self.run_outq.get(timeout=10.0 if block else 0.0)
            except queue.Empty:
                if not self.run_proc.is_alive():
                    raise RuntimeError(
                        f"the run learner died (exitcode {self.run_proc.exitcode})"
                    ) from None
                if not block:
                    return ("none", -1, None)
                continue
            if kind == "run_error":
                raise RuntimeError(f"the run learner failed:\n{payload}")
            return kind, ver, payload

    def _run_poll(self) -> None:
        while True:
            kind, ver, row = self._run_get(block=False)
            if kind == "none":
                return
            self.run_ver = int(ver)
            print(
                f"[run_ppo] v{ver} runs {row['runs']} win {row['win']:.3f} "
                f"floor {row['floor_mean']:.1f} 2nd-boss {row['reached_second_boss']:.3f} "
                f"err {row['errors']} trans {row['n_trans']} ev "
                f"{row.get('u_explained_variance', float('nan')):.3f} kl {row.get('u_kl', 0):.4f} "
                f"upd {row['update_s']}s{' (value warm-up)' if row['warmup'] else ''}",
                flush=True,
            )

    def stop_run_learner(self) -> None:
        if self.run_proc is None:
            return
        self.run_inq.put(("stop",))
        self.run_proc.join(timeout=60)
        if self.run_proc.is_alive():
            self.run_proc.terminate()
        _release_queues([self.run_inq, self.run_outq])
        self.run_proc = None

    # ── update ranks ─────────────────────────────────────────────────────

    def start_ranks(self) -> None:
        devs = self.cfg.update_devices.split(",")
        ctx = get_context("spawn")
        self._store_dir = tempfile.mkdtemp(prefix="cppo_dist_")
        store = os.path.join(self._store_dir, "store")
        self.rank_outq = ctx.Queue()
        for r, d in enumerate(devs):
            q = ctx.Queue()
            p = ctx.Process(
                target=rank_main,
                args=(r, len(devs), d, store, self.cfg, q, self.rank_outq),
                daemon=True,
            )
            p.start()
            self.ranks.append(p)
            self.rank_inq.append(q)
        ready = 0
        while ready < len(devs):
            kind, r, payload = self._rank_get()
            if kind != "ready":
                raise RuntimeError(f"update rank {r}: unexpected {kind!r} at start-up")
            ready += 1

    def stop_ranks(self) -> None:
        for q in self.rank_inq:
            q.put(("stop",))
        deadline = time.time() + 60
        for p in self.ranks:
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.terminate()
        _release_queues([*self.rank_inq, self.rank_outq])
        self.ranks, self.rank_inq = [], []
        if self._store_dir:
            shutil.rmtree(self._store_dir, ignore_errors=True)
            self._store_dir = ""

    def _rank_get(self, block: bool = True) -> tuple[str, int, Any]:
        """The next rank message; raises if a rank failed or died."""
        while True:
            try:
                kind, r, payload = self.rank_outq.get(timeout=10.0 if block else 0.0)
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.ranks) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"an update rank died: (rank, exitcode) {dead}") from None
                if not block:
                    return ("none", -1, None)
                continue
            if kind == "error":
                raise RuntimeError(f"update rank {r} failed:\n{payload}")
            return kind, r, payload

    # ── collection ───────────────────────────────────────────────────────

    def _autocast(self, dev: torch.device) -> Any:
        return torch.autocast("cuda", torch.bfloat16, enabled=self.cfg.bf16 and dev.type == "cuda")

    @torch.no_grad()
    def _act(self, p: Packed) -> tuple[np.ndarray, np.ndarray]:
        """Sample T = 1 actions from the acting policy (honest tokens only)."""
        g = nets.Padded.from_flat(p, None, priv=False).to(self.dev_i)
        xp = nets.batch_inputs(g, None, critic=False)
        with self._autocast(self.dev_i):
            logits, _ = nets.run(self.pol_inf, xp)
        logp_all = torch.log_softmax(logits.float(), -1)
        a = torch.multinomial(logp_all.exp(), 1, generator=self.gen).squeeze(-1)
        logp = logp_all.gather(1, a[:, None]).squeeze(-1)
        out = torch.stack([a.float(), logp]).cpu().numpy()
        return out[0].astype(np.int64), out[1]

    def collect(self) -> tuple[list[tuple[list[Step], float]], dict[str, Any]]:
        """Serve the workers until `iter_decisions` decisions of finished episodes.

        Between inference batches, a finished update is taken in (`_poll`).
        """
        cfg = self.cfg
        done: list[tuple[list[Step], float, bool]] = []
        n_done = 0
        st: dict[str, Any] = {
            "won": Counter(),
            "fights": Counter(),
            "won_own": Counter(),
            "fights_own": Counter(),
            "errors": 0,
            "capped": 0,
            "ep_decisions": [],
            "reward": [],
            "batches": 0,
            "batch_size": [],
            "fwd_s": 0.0,
            "wait_s": 0.0,
            "errors_sample": [],
            "policy_swaps": 0,
            "won_run": Counter(),
            "fights_run": Counter(),
            "run_decisions": 0,
            "runs_done": 0,
            "reward_run": [],
        }
        t0 = time.time()
        while n_done < cfg.iter_decisions:
            if self.inflight is not None and self._poll(block=False):
                st["policy_swaps"] += 1
            if self.run_proc is not None:
                self._run_poll()
            tw = time.time()
            ready = wait(self.conns)
            st["wait_s"] += time.time() - tw
            msgs = []
            for c in ready:
                try:
                    msgs.append((c, c.recv()))
                except EOFError:
                    dead = [
                        (i, p.pid, p.exitcode)
                        for i, p in enumerate(self.procs)
                        if p.exitcode is not None
                    ]
                    raise RuntimeError(
                        f"a rollout worker died: (wid, pid, exitcode) {dead}"
                    ) from None
            ds: list[Decision] = []
            packs: list[Packed] = []
            keys: list[tuple[int, int, int]] = []
            per_conn: list[tuple[Any, int]] = []
            run_rows: list[dict[str, Any]] = []
            for c, (_kind, wid, wkeys, packed, ends, *rest) in msgs:
                if rest and rest[0]:
                    run_rows.extend(rest[0])
                for e in ends:
                    key = (wid, e["slot"], e["ep"])
                    traj = self.open.pop(key, None)
                    if e.get("run"):
                        st["fights_run"][e["room"]] += 1
                        if not e["error"]:
                            st["won_run"][e["room"]] += int(e["won"])
                    else:
                        st["fights"][e["room"]] += 1
                    if e.get("own"):
                        st["fights_own"][e["room"]] += 1
                    if e["error"]:
                        st["errors"] += 1
                        if len(st["errors_sample"]) < 3:
                            st["errors_sample"].append(e["error"])
                        continue
                    if not e.get("run"):
                        st["won"][e["room"]] += int(e["won"])
                    else:
                        st["run_decisions"] += len(traj or [])
                    if e.get("own"):
                        st["won_own"][e["room"]] += int(e["won"])
                    st["capped"] += int(e["capped"])
                    if e.get("r_run") is not None:
                        r = float(e["r_run"])
                        st["reward_run"].append(r)
                    else:
                        r = (
                            1.0 + cfg.w_hp * e["hp_frac"]
                            if e["won"]
                            else cfg.w_removed * e["removed"]
                        )
                        st["reward"].append(r)
                    if traj:
                        done.append((traj, r, e.get("r_run") is not None))
                        n_done += len(traj)
                        st["ep_decisions"].append(len(traj))
                dd = unpack(packed)
                if dd:
                    packs.append(packed)
                ds.extend(dd)
                keys.extend((wid, s, ep) for s, ep in wkeys)
                per_conn.append((c, len(dd)))
            if ds:
                tf = time.time()
                acts, logps = self._act(concat_packed(packs))
                st["fwd_s"] += time.time() - tf
                st["batches"] += 1
                st["batch_size"].append(len(ds))
                for k, d, a, lp in zip(keys, ds, acts, logps, strict=True):
                    self.open[k].append(Step(d, int(a), float(lp)))
            if run_rows:
                st["runs_done"] += len(run_rows)
                frac = min(1.0, self.decisions / max(1, cfg.budget_decisions))
                self.run_inq.put(("rows", run_rows, frac))
            ver = self.run_ver if self.run_proc is not None else None
            i = 0
            for c, n in per_conn:
                c.send(("act", [int(x) for x in acts[i : i + n]] if n else [], ver))
                i += n
        st["collect_s"] = time.time() - t0
        st["n_done"] = n_done
        return done, st

    # ── hand-off to the ranks ────────────────────────────────────────────

    def _send(self, done: list[tuple[list[Step], float, bool]], cst: dict[str, Any]) -> None:
        self._dispatch(self._pack(done, cst))

    def _pack(
        self, done: list[tuple[list[Step], float, bool]], cst: dict[str, Any]
    ) -> dict[str, Any]:
        """A finished batch in shared memory (built while the previous update runs)."""
        t0 = time.time()
        steps = [s for traj, *_ in done for s in traj]
        n = len(steps)
        ds = [s.d for s in steps]
        n_tok = np.fromiter((len(d.sec) for d in ds), np.int32, n)
        n_priv = np.fromiter((len(d.pcat) for d in ds), np.int32, n)
        nt, npv = int(n_tok.sum()), int(n_priv.sum())
        shm, layout = to_shm(
            {
                "cat": ((nt, T.N_CAT), np.int16, [d.cat for d in ds]),
                "cont": ((nt, T.N_CONT), np.float16, [d.cont for d in ds]),
                "sec": ((nt,), np.int8, [d.sec for d in ds]),
                "n_tok": ((n,), np.int32, n_tok),
                "n_act": ((n,), np.int32, np.fromiter((d.n_act for d in ds), np.int32, n)),
                "pcat": ((npv, T.N_CAT), np.int16, [d.pcat for d in ds]),
                "pcont": ((npv, T.N_CONT), np.float16, [d.pcont for d in ds]),
                "n_priv": ((n,), np.int32, n_priv),
                "action": ((n,), np.int64, np.fromiter((s.action for s in steps), np.int64, n)),
                "logp": ((n,), np.float32, np.fromiter((s.logp for s in steps), np.float32, n)),
                "ep_len": ((len(done),), np.int64, np.asarray([len(x[0]) for x in done], np.int64)),
                "reward": ((len(done),), np.float64, np.asarray([x[1] for x in done], np.float64)),
                "ep_run": (
                    (len(done),),
                    np.int8,
                    np.asarray([len(x) > 2 and x[2] for x in done], np.int8),
                ),
            }
        )
        return {
            "shm": shm,
            "layout": layout,
            "done": len(done),
            "cst": cst,
            "pack_s": time.time() - t0,
        }

    def _dispatch(self, b: dict[str, Any]) -> None:
        """Hand a packed batch to the ranks; advance the counters."""
        cfg = self.cfg
        cst, shm, layout = b["cst"], b["shm"], b["layout"]
        job: dict[str, Any] = {
            "shm": shm.name,
            "layout": layout,
            "it": self.it,
            "frac": min(1.0, self.decisions / max(1, cfg.budget_decisions)),
            "train_policy": self.it >= max(cfg.warmup_iters, cfg.policy_frozen_until),
        }
        self.it += 1
        self.decisions += cst["n_done"]
        self.episodes += b["done"]
        tag = f"it{self.it:05d}"
        eval_tag = None
        if self.decisions - self.last_ckpt >= cfg.ckpt_every:
            job["ckpt"] = os.path.join(cfg.out, "ckpt", f"{tag}.pt")
            job["ckpt_extra"] = {
                "it": self.it,
                "decisions": self.decisions,
                "cfg": asdict(cfg) | {"fixed_rows": []},
            }
            self.last_ckpt = self.decisions
            if self.decisions - self.last_eval >= cfg.eval_every:
                eval_tag = tag
                self.last_eval = self.decisions
        job["state"] = self._counters()
        for q in self.rank_inq:
            q.put(("update", job))
        self.inflight = {
            "job": job,
            "cst": cst,
            "shm": shm,
            "eval_tag": eval_tag,
            "it": self.it,
            "decisions": self.decisions,
            "episodes": self.episodes,
            "send_s": b["pack_s"],
            "open_episodes": len(self.open),
        }

    def _poll(self, block: bool) -> bool:
        """Take in the in-flight update if it is done (or wait for it)."""
        kind, _r, res = self._rank_get(block=block)
        if kind == "none":
            return False
        if kind != "done":
            raise RuntimeError(f"unexpected rank message {kind!r}")
        inf, self.inflight = self.inflight, None
        assert inf is not None
        inf["shm"].unlink()
        inf["shm"].close()
        sd = {k: torch.from_numpy(v) for k, v in res.pop("policy").items()}
        self.pol_inf.load_state_dict(sd)
        self._log(inf, res)
        if inf["eval_tag"]:
            self.launch_eval(inf["job"]["ckpt"], inf["eval_tag"])
        return True

    def _log(self, inf: dict[str, Any], ust: dict[str, Any]) -> None:
        cst = inf["cst"]
        now = time.time()
        fights = cst["fights"]
        row = {
            "it": inf["it"],
            "decisions": inf["decisions"],
            "episodes": inf["episodes"],
            "time": now,
            "collect_s": cst["collect_s"],
            "dec_per_s": cst["n_done"] / max(1e-9, now - self._t_row),
            "fwd_s": cst["fwd_s"],
            "wait_s": cst["wait_s"],
            "send_s": inf["send_s"],
            "policy_swaps": cst["policy_swaps"],
            "batch_mean": float(np.mean(cst["batch_size"])) if cst["batch_size"] else 0,
            "win": {k: cst["won"][k] / max(1, fights[k]) for k in fights},
            "fights": dict(fights),
            "win_own": {k: cst["won_own"][k] / max(1, v) for k, v in cst["fights_own"].items()},
            "fights_own": dict(cst["fights_own"]),
            "win_run": {k: cst["won_run"][k] / max(1, v) for k, v in cst["fights_run"].items()},
            "fights_run": dict(cst["fights_run"]),
            "run_fight_decisions": cst["run_decisions"],
            "runs_done": cst["runs_done"],
            "run_ver": self.run_ver,
            "errors": cst["errors"],
            "errors_sample": cst["errors_sample"],
            "capped": cst["capped"],
            "reward_mean": float(np.mean(cst["reward"])) if cst["reward"] else math.nan,
            "reward_std": float(np.std(cst["reward"])) if cst["reward"] else math.nan,
            "reward_run_mean": float(np.mean(cst["reward_run"])) if cst["reward_run"] else math.nan,
            "reward_run_std": float(np.std(cst["reward_run"])) if cst["reward_run"] else math.nan,
            "ep_decisions_mean": float(np.mean(cst["ep_decisions"]))
            if cst["ep_decisions"]
            else math.nan,
            "open_episodes": inf["open_episodes"],
            **ust,
        }
        self._t_row = now
        with open(os.path.join(self.cfg.out, "stats.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[combat_ppo] it {row['it']} dec {row['decisions']:,} "
            f"{row['dec_per_s']:.0f}/s win {json.dumps({k: round(v, 3) for k, v in row['win'].items()})} "
            f"kl {ust.get('kl', 0):.4f} lag {ust.get('lag_kl', 0):.4f} ev {ust['value_ev']:.3f} "
            f"ent {ust.get('entropy', 0):.3f} coll {cst['collect_s']:.1f}s upd {ust['update_s']:.1f}s",
            flush=True,
        )

    # ── checkpoints / evals ──────────────────────────────────────────────

    def launch_eval(self, path: str, tag: str) -> None:
        """One greedy eval per `eval_fights` file (comma-separated).

        The first file's result is `evals/<tag>.json`; each further file's is
        `evals/<tag>.<file stem>.json`.
        """
        cfg = self.cfg
        files = [f for f in cfg.eval_fights.split(",") if f]
        for i, fights in enumerate(files):
            stem = pathlib.Path(fights).stem
            name = f"{tag}.json" if i == 0 else f"{tag}.{stem}.json"
            out = os.path.join(cfg.out, "evals", name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "ppo.combat_ppo",
                "eval",
                "--ckpt",
                path,
                "--fights",
                fights,
                "--build-reps",
                str(cfg.eval_reps),
                "--procs",
                str(cfg.eval_procs),
                "--greedy",
                "--out",
                out,
            ]
            log = open(out + ".log", "w", encoding="utf-8")  # owned by the child
            self.evals.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        if cfg.fullrun_eval_seeds:
            self.launch_fullrun_eval(path, tag)

    def launch_fullrun_eval(self, path: str, tag: str) -> None:
        """Full IC runs: the current run net and the combat net at `path`, both greedy."""
        cfg = self.cfg
        if cfg.run_slots and self.run_proc is not None:
            from .runlearn import acting_path

            run_ck = os.path.join(cfg.out, "run", "ckpt", f"at_{tag}.pt")
            shutil.copyfile(acting_path(cfg.out, self.run_ver), run_ck)
        else:
            run_ck = cfg.run_init
        out = os.path.join(cfg.out, "evals", f"{tag}.fullrun.jsonl")
        cmd = [
            sys.executable, "-m", "ppo.probe",
            "--policy", "net_greedy", "--ckpt", run_ck,
            "--ascension", str(cfg.run_ascension), "--act1", cfg.run_act1,
            "--character", cfg.run_character,
            "--seeds", str(cfg.fullrun_eval_seeds),
            "--seed-start", str(cfg.fullrun_eval_seed_start),
            "--procs", str(cfg.fullrun_eval_procs),
            "--combat", "policy", "--combat-ckpt", path,
            "--out", out,
        ]  # fmt: skip
        log = open(out + ".log", "w", encoding="utf-8")  # owned by the child
        self.evals.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))

    # ── main loop ────────────────────────────────────────────────────────

    def _more(self) -> bool:
        cfg = self.cfg
        return self.decisions < cfg.budget_decisions and (
            cfg.max_iters <= 0 or self.it < cfg.max_iters
        )

    def train(self) -> None:
        cfg = self.cfg
        with open(os.path.join(cfg.out, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg) | {"fixed_rows": len(cfg.fixed_rows)}, f, indent=1)
        if self._init_nets is not None:
            pol, crit = self._init_nets
            path = os.path.join(cfg.out, "ckpt", "it00000.pt")
            nets.save_policy(path, pol, crit, self.meta, {"it": 0, "decisions": 0})
            self.launch_eval(path, "it00000")
            self.last_eval = 0
            self._init_nets = None
        if cfg.run_slots:
            self.start_run_learner()
        self.start_ranks()
        self.start_workers()
        self._t_row = time.time()
        try:
            while self._more():
                done, cst = self.collect()
                batch = self._pack(done, cst)
                del done
                try:
                    if self.inflight is not None:
                        self._poll(block=True)
                except BaseException:
                    batch["shm"].unlink()
                    batch["shm"].close()
                    raise
                self._dispatch(batch)
            if self.inflight is not None:
                self._poll(block=True)
            # The final checkpoint, from rank 0's replica.
            self.rank_inq[0].put(
                (
                    "save",
                    {
                        "ckpt": os.path.join(cfg.out, "ckpt", "final.pt"),
                        "ckpt_extra": {
                            "it": self.it,
                            "decisions": self.decisions,
                            "cfg": asdict(cfg) | {"fixed_rows": []},
                        },
                        "state": self._counters(),
                    },
                )
            )
            kind, _r, _ = self._rank_get()
            if kind != "saved":
                raise RuntimeError(f"unexpected rank message {kind!r}")
        finally:
            if self.inflight is not None:
                self.inflight["shm"].unlink()
                self.inflight["shm"].close()
                self.inflight = None
            self.stop_workers()
            self.stop_ranks()
            self.stop_run_learner()
        for p in self.evals:
            p.wait()
