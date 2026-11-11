"""Lockstep: `env.E2EEnv` vs its Rust twin (`rsenv.RsE2EEnv`'s `sts2_rs.E2eTrainEnv`).

Both envs play one segment from the same run with the same choices. Compared at
every stored row: the env fields (`is_combat`, `dtype`, `n_classes`, `classes`,
`members`, `unit`, `cell`, `act`, `floor`, `bootstrap`; the Rust env runs with
`detail=True` to emit the merge and the act/floor), the reward (bit for bit) and
the tokenized row (`tokens.encode(d, rare=rare, priv=priv)`, every array bit for
bit); at the end: the terminal (`TERM_FIELDS`, every nonzero counter, `f9/*`
included, whether it errored, the run fingerprint), the action log, the practice
captures (cell, Move, seed, steps, rows, action log, snapshot fingerprint), the
F9-sampled states (by fingerprint, in order; the Rust env's F9 bound is lifted
so none is dropped) and the F9 violations.

    python -m ppo.e2e.rsenv_lockstep --seeds 1300000000 50 [--pinned] [--practice 2]
    python -m ppo.e2e.rsenv_lockstep --log <actions-NNN.e2el> --n 100 [--practice 2]
    (--rare <RareMap json> --f9-p 0.05 --out <json>)

`--log` replays the newest records of a lane's action log (`actionlog`): the
student's own choices, so the runs reach every act and the double boss. Exit 1
on any mismatch.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from . import env as E
from . import tokens as TK

TOK_FIELDS = (
    "cat",
    "cont",
    "sec",
    "tok_flags",
    "mem_tok",
    "mem_class",
    "cls_tok",
    "cls_kind",
    "pool_tok",
    "pool_vid",
    "pool_w",
)
PRIV_FIELDS = ("pcat", "pcont", "psec", "ppool_tok", "ppool_vid", "ppool_w")
TERM_FIELDS = (
    "outcome",
    "won",
    "cap",
    "reward",
    "total_reward",
    "unassigned_reward",
    "rows",
    "steps",
    "floor",
    "act",
    "bosses",
    "reached_boss2",
)


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.uint16) if a.dtype == np.float16 else a


def tok_diff(a: TK.Tok, b: TK.Tok, *, priv: bool) -> list[str]:
    """The fields in which two rows differ (float16 compared by bits)."""
    out = []
    for f in TOK_FIELDS + (PRIV_FIELDS if priv else ()):
        x, y = getattr(a, f), getattr(b, f)
        if x.shape != y.shape or not np.array_equal(_bits(x), _bits(y)):
            out.append(f)
    if (a.sctx_lo, a.n_sctx, a.is_combat, a.dtype) != (b.sctx_lo, b.n_sctx, b.is_combat, b.dtype):
        out.append("scalars")
    return out


class Lockstep:
    """Runs segments both ways; collects mismatches (`bad`) and counts (`stats`)."""

    def __init__(
        self, *, f9_p: float = 0.01, rare_json: Any = None, priv: bool = True, max_ex: int = 40
    ) -> None:
        import sts2_rs

        from .rsenv import RsE2EEnv

        self.RsE2EEnv = RsE2EEnv
        self.f9_p = float(f9_p)
        self.priv = bool(priv)
        self.rare_py = self.rare_rs = None
        if rare_json is not None:
            from ppo.combat_ctx.data import RareMap

            self.rare_py = RareMap.from_json(rare_json)
            self.rare_rs = sts2_rs.E2eRareMap(rare_json)
        self.stats: collections.Counter = collections.Counter()
        self.examples: list[dict[str, Any]] = []
        self.max_ex = max_ex

    @property
    def mismatches(self) -> int:
        return int(self.stats["mismatch"])

    def bad(self, where: str, **kw: Any) -> None:
        self.stats["mismatch"] += 1
        self.stats[f"mismatch/{where}"] += 1
        if len(self.examples) < self.max_ex:
            self.examples.append({"where": where, **{k: str(v)[:400] for k, v in kw.items()}})

    def segment(
        self,
        seed: int,
        rm: Any,
        segment: E.Segment,
        choose: Callable[[Any, int], int],
        *,
        capture: bool = True,
        by_action: bool = False,
        fingerprint: str | None = None,
        tag: str = "",
    ) -> list[E.Capture]:
        """One segment both ways from `rm` (each env gets a clone). `choose(d, k)` picks
        row `k`'s merged class (with `by_action`, its member index). Returns the Python
        env's captures."""
        caps_py: list[E.Capture] = []
        caps_rs: list[E.Capture] = []
        hits_py: list[str] = []
        cfg_py = E.EnvConfig(f9_p=self.f9_p, on_capture=caps_py.append if capture else None)
        cfg_rs = E.EnvConfig(f9_p=self.f9_p, on_capture=caps_rs.append if capture else None)
        pe = E.E2EEnv(seed, rm=rm.clone(), cfg=cfg_py, segment=segment)
        chk = pe.f9
        check = chk.check

        def spy(r: Any, cs: Any = None, *, run_key: Any = None) -> Any:
            hits_py.append(E.run_fingerprint(r))
            return check(r, cs, run_key=run_key)

        chk.check = spy
        re_ = self.RsE2EEnv(
            seed,
            rm=rm.clone(),
            cfg=cfg_rs,
            segment=segment,
            rare=self.rare_rs,
            detail=True,
            f9_queue_max=1 << 30,
        )
        hits_rs: list[str] = []
        rcheck = re_.f9.check

        def rspy(r: Any, cs: Any = None, *, run_key: Any = None) -> Any:
            hits_rs.append(E.run_fingerprint(r))
            return rcheck(r, cs, run_key=run_key)

        re_.f9.check = rspy
        d, r = pe.reset(), re_.reset()
        k = 0
        while True:
            isd = isinstance(d, E.E2EDecision)
            if isd != (not isinstance(r, E.E2ETerminal)):
                self.bad(
                    "kind",
                    tag=tag,
                    row=k,
                    py=type(d).__name__,
                    rs=type(r).__name__,
                    py_err=getattr(d, "error", None),
                    rs_err=getattr(r, "error", None),
                )
                return caps_py
            if not isd:
                break
            self.stats["rows"] += 1
            self.stats["combat" if d.is_combat else "ooc"] += 1
            for f, pv, rv in (
                ("is_combat", bool(d.is_combat), r.is_combat),
                ("dtype", int(d.dtype), r.dtype),
                ("n_classes", int(d.n_classes), r.n_classes),
                ("bootstrap", bool(d.bootstrap), r.bootstrap),
                ("unit", tuple(d.unit), r.unit),
                ("cell", int(d.cell), r.cell),
                ("classes", [int(x) for x in d.classes], r.classes),
                ("members", [[int(x) for x in m] for m in d.members], r.members),
                ("act", int(d.act), r.act),
                ("floor", int(d.floor), r.floor),
            ):
                if pv != rv:
                    self.bad(f"row/{f}", tag=tag, row=k, py=pv, rs=rv)
                    return caps_py
            if float(d.reward).hex() != float(r.reward).hex():  # bit-exact
                self.bad("row/reward", tag=tag, row=k, py=repr(d.reward), rs=repr(r.reward))
            try:
                tp = TK.encode(d, rare=self.rare_py, priv=self.priv)
            except Exception as ex:
                tp = ex
            try:
                tr = r.tok
            except Exception as ex:
                tr = ex
            if isinstance(tp, Exception) or isinstance(tr, Exception):
                if type(tp) is not type(tr):
                    self.bad("tok/error", tag=tag, row=k, py=repr(tp), rs=repr(tr))
                else:
                    self.stats["tok_errors"] += 1
            else:
                diff = tok_diff(tp, tr, priv=self.priv)
                if diff:
                    self.bad("tok", tag=tag, row=k, fields=diff, combat=d.is_combat, dtype=d.dtype)
            if d.bootstrap:
                d, r = pe.terminal, re_.terminal
                break
            if isinstance(tp, Exception):
                return caps_py  # the worker would end this segment here
            c = int(choose(d, k))
            k += 1
            if by_action:
                d, r = pe.step_action(c), re_.step_action(c)
            else:
                d, r = pe.step(c), re_.step(c)
        self.stats["segments"] += 1
        self.stats[f"outcome/{d.outcome}"] += 1
        for f in TERM_FIELDS:
            if getattr(d, f) != getattr(r, f):
                self.bad(
                    f"term/{f}",
                    tag=tag,
                    py=getattr(d, f),
                    rs=getattr(r, f),
                    py_err=d.error,
                    rs_err=r.error,
                )
        pc = {k_: v for k_, v in d.counts.items() if v}
        rc = {k_: v for k_, v in r.counts.items() if v}
        if pc != rc:
            diff = {x: (pc.get(x), rc.get(x)) for x in set(pc) | set(rc) if pc.get(x) != rc.get(x)}
            self.bad("term/counts", tag=tag, diff=diff)
        if (d.error is None) != (r.error is None):
            self.bad("term/error", tag=tag, py=d.error, rs=r.error)
        if d.fingerprint != r.fingerprint:
            self.bad("term/fingerprint", tag=tag)
        if fingerprint is not None:
            self.stats["log_fp_match" if r.fingerprint == fingerprint else "log_fp_differ"] += 1
        if list(d.log_idx) != list(r.log_idx):
            self.bad("term/log_idx", tag=tag)
        if hits_py != hits_rs:
            self.bad("f9", tag=tag, py=len(hits_py), rs=len(hits_rs))
        if pe.f9_violations != re_.f9_violations:
            self.bad("f9/violations", tag=tag)
        self.stats["f9_hits"] += len(hits_py)
        if len(caps_py) != len(caps_rs):
            self.bad("captures/n", tag=tag, py=len(caps_py), rs=len(caps_rs))
        else:
            for cp, cr in zip(caps_py, caps_rs, strict=True):
                self.stats["captures"] += 1
                for f in ("cell", "move", "seed", "steps", "rows", "idx"):
                    if getattr(cp, f) != getattr(cr, f):
                        self.bad(f"capture/{f}", tag=tag, py=getattr(cp, f), rs=getattr(cr, f))
                if E.run_fingerprint(cp.snap) != E.run_fingerprint(cr.snap):
                    self.bad("capture/snap", tag=tag, cell=cp.cell)
        return caps_py

    def practice(self, caps: list[E.Capture], n: int, rng: np.random.Generator, tag: str) -> None:
        """Every double-boss capture and `n` random others as practice segments, random play."""
        from .actionlog import start_from_capture
        from .practice import DOUBLE_BOSS, ROOM_OF, segment_for

        pick = [j for j, c in enumerate(caps) if c.cell == DOUBLE_BOSS]
        rest = [j for j in range(len(caps)) if j not in pick]
        rng.shuffle(rest)
        for j in pick + rest[:n]:
            cap = caps[j]
            reseed = 600_000_000 + int(rng.integers(40_000_000))
            rm = start_from_capture(cap, reseed)
            if (rm.current_room() or {}).get("room_type") != ROOM_OF[cap.cell]:
                self.stats["practice_bad_start"] += 1
                continue
            self.stats[f"practice/{cap.cell}"] += 1
            self.segment(
                cap.seed,
                rm,
                segment_for(cap, reseed),
                lambda d, _k, rng=rng: int(rng.integers(d.n_classes)),
                capture=False,
                tag=f"{tag}:practice:{cap.cell}:{j}",
            )


def run_seeds(ls: Lockstep, seed0: int, n: int, *, pinned: bool, practice: int) -> None:
    for seed in range(seed0, seed0 + n):
        rng = np.random.default_rng(seed)
        rm = E.make_run_manager(seed)
        if pinned:
            rm.set_player(hp=100_000, max_hp=100_000)
        caps = ls.segment(
            seed,
            rm,
            E.RUN,
            lambda d, _k, rng=rng: int(rng.integers(d.n_classes)),
            tag=f"run:{seed}",
        )
        if practice:
            ls.practice(caps, practice, rng, f"run:{seed}")


def run_log(ls: Lockstep, path: str, n: int, *, practice: int, window: int) -> None:
    from .actionlog import WORKER_ERROR, read_records, replay_capture, start_from_capture

    recs = list(read_records(path))
    k = 0
    for rec in recs[-window:]:
        h = rec.header
        seg = h["segment"]
        if seg["kind"] not in ("run", "practice") or h.get("outcome") == WORKER_ERROR:
            continue
        seed = int(h["seed"])
        rng = np.random.default_rng(seed)
        idx = rec.idx
        if seg["kind"] == "run":
            caps = ls.segment(
                seed,
                E.make_run_manager(seed),
                E.RUN,
                lambda _d, j, idx=idx: idx[j],
                by_action=True,
                fingerprint=h.get("fingerprint"),
                tag=f"log:{seed}",
            )
            if practice:
                ls.practice(caps, practice, rng, f"log:{seed}")
        else:
            cap = replay_capture(
                int(seg["parent_seed"]), list(seg["parent_idx"]), int(seg["capture_steps"])
            )
            reseed = int(seg["reseed"])
            segment = E.Segment(
                kind=seg["kind"],
                cell=seg["cell"],
                end=seg["end"],
                parent_seed=int(seg["parent_seed"]),
                parent_idx=tuple(seg["parent_idx"]),
                capture_steps=int(seg["capture_steps"]),
                move=tuple(sorted(seg["move"].items())),
                reseed=reseed,
                uid=seg.get("uid"),
            )
            ls.segment(
                seed,
                start_from_capture(cap, reseed),
                segment,
                lambda _d, j, idx=idx: idx[j],
                capture=False,
                by_action=True,
                fingerprint=h.get("fingerprint"),
                tag=f"log:{seed}:{seg['cell']}",
            )
        k += 1
        if k >= n:
            break


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", nargs=2, type=int, metavar=("SEED0", "N"))
    ap.add_argument("--pinned", action="store_true", help="player HP 100,000: deep random runs")
    ap.add_argument("--log", help="a lane action log (actions-NNN.e2el)")
    ap.add_argument("--n", type=int, default=50, help="--log: records to replay")
    ap.add_argument("--window", type=int, default=0, help="--log: newest records read (3n)")
    ap.add_argument("--practice", type=int, default=2, help="practice segments per run")
    ap.add_argument("--f9-p", type=float, default=0.01)
    ap.add_argument("--rare", help="a RareMap JSON (RareMap.to_json())")
    ap.add_argument("--no-priv", action="store_true")
    ap.add_argument("--out", help="write the stats and examples here (JSON)")
    a = ap.parse_args(argv)
    rare = None
    if a.rare:
        with open(a.rare, encoding="utf-8") as fh:
            rare = json.load(fh)
    ls = Lockstep(f9_p=a.f9_p, rare_json=rare, priv=not a.no_priv)
    t0 = time.time()
    if a.log:
        run_log(ls, a.log, a.n, practice=a.practice, window=a.window or 3 * a.n)
    elif a.seeds:
        run_seeds(ls, a.seeds[0], a.seeds[1], pinned=a.pinned, practice=a.practice)
    else:
        ap.error("give --seeds or --log")
    res = {"seconds": round(time.time() - t0, 1), "stats": dict(ls.stats), "examples": ls.examples}
    print(json.dumps(res, indent=1)[:8000])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 1 if ls.mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
