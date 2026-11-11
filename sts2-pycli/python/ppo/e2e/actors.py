"""The e2e rollout worker: `n_envs` env slots of `ppo.e2e.env.E2EEnv`, tokenized on the CPU.

Each slot plays segments back to back. `ppo.e2e.practice.SegmentSource` picks
the next one: a practice segment (plan §3.3) when the slot is a practice slot
at the learner's decision clock and a cell is active, else a full campaign
run (`E2EEnv(seed)`, which builds its run with `ppo.e2e.factory.new_run`).
Full runs feed the worker's practice queues. Every slot has its own
`SegmentSource` over the worker's one `PracticeQueues`, so each slot keeps its
own run-seed sequence.

Protocol on the worker's duplex pipe (spawn context; worker code lives here,
never in `python -c`):

* worker -> learner: `("obs", wid, meta, r_prev, rows, extra, ends)`. One
  stored decision per env slot, every round. `meta[i]` is `META_COLS` (int64);
  `r_prev[i]` is the reward the env just assigned to the same segment's row
  `t - 1` (`E2EDecision.reward`; ignored at `t == 0`). `rows` is the round's
  `contracts` row batch (SCHEMA.md section 9), built by WP3b's tokenizer
  (`tokens.encode`, `tokens.row_batch`): `run_uid` is the segment uid below,
  `t` the row's index in its segment, `cell` the practice cell, `bootstrap`
  a practice segment's value-only end row; the acting-time fields keep their
  defaults (the learner fills them). The combat S_ctx block carries the
  combat teacher's RareMap (`cfg["teacher_rare"]`), so the teacher reads its
  input straight from the rows (X-C). `extra` holds the worker's `stamps`
  (wheel and py_src; the learner refuses the batch of a worker whose stamps
  differ from its own), every 64th round the worker's counters (`stats`:
  practice queues and segment sources) and, when the learner asks for them,
  `feats` (`run_teacher_features` of OOC rows), None elsewhere. `ends` holds
  a dict per segment that ended since the last message (`end_record`); a
  practice segment's value-only bootstrap row (`meta.boot`) and its end ride
  in the same message.
* learner -> worker: `("act", [merged class per row], info)` or `("stop",)`.
  `info["decisions"]` is the learner's decision clock (practice schedule and
  queue ages); `info["feats"]` asks for the run teacher's inputs.

Run seeds follow #1081's slot layout (plan §3.1 "Other seeds"):
`run_seed_base + 7 * start_it + (wid * 64 + slot) * 100,000 + run count`.
Segment uids (the row batch's `run_uid`) are the learner's, unique across
sessions: `(session << 48) | (wid << 40) | (slot << 24) | segment count`,
where `session` is the iteration the learner started at. A resumed learner
keeps its held rows, so a new run must never reuse a held run's uid; the
env's own `run_uid` (the run seed) does not have that property.

What a failure outside the env does follows the env's own rule (`env.ends_segment`,
GH #1097 A2): an exception or a Rust panic (a `BaseException`) while a segment starts
(the env's and run's construction, a practice start, the env's first `run.reset()`),
while the tokenizer reads a decision (its Rust getters) or while the env steps ends the
segment as an error record (`error/<type>`); a start fails at most `MAX_START_FAILS`
times in a row. A `sts2_client.GameVersionMismatch` (the process or the engine is on
the wrong build, so every segment would fail alike), `KeyboardInterrupt`, `SystemExit`
and every other `BaseException` leave the worker, which the learner reports as a dead
worker.

Rows come from `tokens.encode(d)` on each `env.E2EDecision`: the env's own
intent log and combat view, and the open selection's picks (`d.sel`).
WP3b's test checks that it equals the `tokens.Tokenizer` path the env can
drive (`EnvConfig.tokenizer`). A slot's new segment drops its last run's entry from
the tokenizer's run-static cache, which would otherwise keep finished runs'
RunManagers alive; the other slots' entries stay.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

META_COLS = (
    "slot",
    "uid",
    "t",
    "is_combat",
    "dtype",
    "cell",
    "unit_kind",
    "unit_idx",
    "n_classes",
    "boot",
    "practice",
)
M = {k: i for i, k in enumerate(META_COLS)}
UNIT_KIND = {"fight": 0, "act": 1}
#: Consecutive failed segment starts before the worker gives up (a broken engine).
MAX_START_FAILS = 20
#: Bit layout of a segment uid.
UID_SESSION_SHIFT, UID_WID_SHIFT, UID_SLOT_SHIFT = 48, 40, 24
UID_SESSION_MASK = (1 << 14) - 1


def run_seed(cfg: dict[str, Any], wid: int, slot: int, count: int) -> int:
    return (
        int(cfg["run_seed_base"])
        + 7 * int(cfg["start_it"])
        + (wid * 64 + slot) * 100_000
        + int(count)
    )


def segment_uid(session: int, wid: int, slot: int, count: int) -> int:
    if not (0 <= wid < 256 and 0 <= slot < 1 << 16 and 0 <= count < 1 << 24):
        raise ValueError(f"segment uid out of range: wid {wid} slot {slot} count {count}")
    return (
        ((int(session) & UID_SESSION_MASK) << UID_SESSION_SHIFT)
        | (wid << UID_WID_SHIFT)
        | (slot << UID_SLOT_SHIFT)
        | count
    )


F9_SAMPLES = 8  # F9 violation texts an end record carries (`E2EEnv.f9_violations` keeps as many)
F9_SAMPLE_CHARS = 600


def f9_samples(env: Any) -> list[str]:
    """The first F9 violation texts of a segment's env (`E2EEnv.f9_violations`), cut to
    what an end record carries: with the seed they say which invariant broke and where."""
    return [str(v)[:F9_SAMPLE_CHARS] for v in (getattr(env, "f9_violations", None) or [])][
        :F9_SAMPLES
    ]


def end_record(
    uid: int,
    slot: int,
    t_last: int,
    term: Any,
    kind: str,
    cell: str | None,
    stamps: dict[str, Any],
    *,
    f9_texts: list[str] | None = None,
) -> dict[str, Any]:
    """What the learner needs from an `E2ETerminal` (plain types only). `f9_texts` are
    the segment's first F9 violation texts (`f9_samples`)."""
    counts = dict(term.counts) if term.counts is not None else {}
    return {
        "uid": uid,
        "slot": slot,
        "t_last": int(t_last),
        "r_last": float(term.reward),
        "outcome": str(term.outcome),
        "won": bool(term.won),
        "cap": term.cap,
        "floor": int(term.floor),
        "act": int(term.act),
        "bosses": int(term.bosses),
        "reached_boss2": bool(term.reached_boss2),
        "rows": int(term.rows),
        "steps": int(term.steps),
        "seed": int(term.seed),
        "kind": kind,
        "cell": cell,
        "error": term.error,
        "total_reward": float(term.total_reward),
        "unassigned_reward": float(term.unassigned_reward),
        "f9_checks": int(counts.get("f9/checks", 0)),
        "f9_violations": int(counts.get("f9/violations", 0)),
        "f9_samples": list(f9_texts or []),
        "unknown": int(counts.get("unknown", 0)),
        "fights": int(sum(v for k, v in counts.items() if str(k).startswith("fights/"))),
        "counts": {str(k): int(v) for k, v in counts.items()},
        **stamps,
    }


_RUN_FEATURES: list[Any] = []


def run_teacher_features(d: Any, *, honest: bool = False) -> Any:
    """The run teacher's input for an OOC decision, worker side: `featurize_row` of the
    raw `run_obs` (what R2 and R3 were measured on), or with `honest` of
    `contracts.honest_obs(run_obs)` (SCHEMA.md open question 15's fallback, the
    learner's `run_teacher_honest`). `teachers.RunTeacher.featurize_decision` with the
    same `honest_previews` gives the same features."""
    from ppo.encoder import featurize_row

    from . import contracts as C

    if not _RUN_FEATURES:
        from ppo import schema

        from .teachers import RUN_FEATURES

        _RUN_FEATURES.append(schema.parse_features(RUN_FEATURES))
    obs = C.honest_obs(d.run_obs) if honest else d.run_obs
    return featurize_row(obs, d.actions, d.metrics, features=_RUN_FEATURES[0])


def error_record(
    uid: int,
    slot: int,
    t_last: int,
    kind: str,
    cell: str | None,
    err: str,
    stamps: dict[str, Any],
    etype: str | None = None,
    *,
    env_counts: Mapping[str, int] | None = None,
    seed: int = -1,
    f9_texts: list[str] | None = None,
) -> dict[str, Any]:
    """A segment the worker ended on an exception of its own (outside the env, which
    ends a segment as an `error` terminal itself); `etype` is counted as
    `error/<etype>`, the env's name for an error segment's reason. `env_counts` and
    `seed` are the segment's env's so far (`E2EEnv.counts`, `E2EEnv.seed`), so an F9
    violation it sampled before the error still reaches the learner, and the seed
    replays it."""
    counts: Counter = Counter({str(k): int(v) for k, v in (env_counts or {}).items()})
    if etype:
        counts[f"error/{etype}"] += 1
    return {
        "uid": uid,
        "slot": slot,
        "t_last": int(t_last),
        "r_last": 0.0,
        "outcome": "error",
        "won": False,
        "cap": None,
        "floor": 0,
        "act": 0,
        "bosses": 0,
        "reached_boss2": False,
        "rows": int(t_last + 1),
        "steps": 0,
        "seed": int(seed),
        "kind": kind,
        "cell": cell,
        "error": err,
        "total_reward": 0.0,
        "unassigned_reward": 0.0,
        "f9_checks": int(counts.get("f9/checks", 0)),
        "f9_violations": int(counts.get("f9/violations", 0)),
        "f9_samples": list(f9_texts or []),
        "unknown": int(counts.get("unknown", 0)),
        "fights": int(sum(v for k, v in counts.items() if k.startswith("fights/"))),
        "counts": dict(counts),
        **stamps,
    }


class Slot:
    """One env slot: its segment source, the pending decision and the segment's bookkeeping."""

    def __init__(self, wid: int, k: int, source: Any) -> None:
        self.wid, self.k = wid, k
        self.source = source
        self.count = -1  # segments started (the uid counter)
        self.env: Any = None
        self.d: Any = None  # the pending E2EDecision or E2ETerminal; None = start anew
        self.uid = -1
        self.t = 0
        self.kind = "run"
        self.cell: str | None = None
        self.t0 = time.time()


def _seeds(cfg: dict[str, Any], wid: int, slot: int) -> Iterator[int]:
    n = 0
    while True:
        yield run_seed(cfg, wid, slot, n)
        n += 1


def worker_main(wid: int, conn: Any, cfg: dict[str, Any]) -> None:
    import faulthandler

    faulthandler.enable()  # a native crash still leaves a trace
    # The learner handles SIGINT/SIGTERM and stops its workers itself (`("stop",)`);
    # a worker whose learner died sees its pipe close.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ppo.wheel_guard import python_source_digest, wheel_identity

    from . import contracts as C
    from . import tokens as TK
    from .env import E2EDecision, EnvConfig, ends_segment
    from .practice import PracticeQueues, SegmentSource
    from .rsenv import RsDecision

    stamps = {"wheel": wheel_identity(), "py_src": python_source_digest()}
    # Action logs (plan §3.2, review finding ENVR-1): every segment this worker ends is
    # offered; `actionlog.LogSink` keeps boss-2, capped and error segments and 2% of runs.
    sink = None
    if cfg.get("action_log_dir"):
        from .actionlog import LogSink

        sink = LogSink(
            cfg["action_log_dir"],
            worker=wid,
            seed=(int(cfg["seed"]) * 1_000_003 + int(cfg["start_it"])) & 0xFFFFFFFF,
        )
    rare = None
    if cfg.get("teacher_rare"):
        from ppo.combat_ctx.data import RareMap

        rare = RareMap.from_json(cfg["teacher_rare"])
    session = int(cfg["start_it"])
    clock = {"decisions": int(cfg["start_decisions"])}
    want = {"feats": False}
    queues = PracticeQueues(
        worker=wid,
        seed=(int(cfg["seed"]) * 1_000_003 + int(cfg["start_it"])) & 0xFFFFFFFF,
        enabled=bool(cfg["practice"]),
    )
    honest = bool(cfg.get("run_teacher_honest"))

    def feats_of(d: Any) -> Any:
        return run_teacher_features(d, honest=honest)

    n_envs = int(cfg["n_envs"])
    rust = bool(cfg.get("rust_env"))
    rust_rare = None
    if rust and cfg.get("teacher_rare"):
        import sts2_rs

        rust_rare = sts2_rs.E2eRareMap(cfg["teacher_rare"])
    env_cfg = EnvConfig(
        f9_tolerate=tuple(cfg.get("f9_tolerate") or ()), rust=rust, rust_rare=rust_rare
    )

    def now() -> int:
        return clock["decisions"]

    slots = [
        Slot(
            wid,
            k,
            SegmentSource(queues, _seeds(cfg, wid, k), n_slots=n_envs, cfg=env_cfg, clock=now),
        )
        for k in range(n_envs)
    ]
    fails = {"start": 0}
    counters: Counter = Counter()
    rounds = 0

    def start(s: Slot) -> None:
        s.count += 1
        s.uid = segment_uid(session, wid, s.k, s.count)
        s.t = 0
        s.t0 = time.time()
        # The tokenizer's run-static cache keys an env decision by its RunManager
        # (`id(d.rm)`: E2EDecision has no `run_key`) and holds it alive. Drop this
        # slot's last run only: the other slots' entries stay warm, and the cache
        # holds at most one run per slot.
        old = s.env
        s.env, s.kind, s.cell = None, "run", None
        if old is not None and not getattr(old, "rust", False):
            TK.clear_cache(id(old.rm))
        s.env = s.source.next(s.k, clock["decisions"])
        seg = s.env.segment
        s.kind, s.cell = seg.kind, seg.cell
        s.d = s.env.reset()

    # Segment ends not yet sent; a step error lands here and goes out with the next round.
    ends: list[dict[str, Any]] = []

    def log_end(s: Slot, term: Any) -> None:
        """Offer an ended segment to the action log. A log that fails (a full disk) is
        counted (`log_errors`), never a segment error."""
        if sink is None:
            return
        try:
            sink.offer(term, extra={"uid": s.uid, "decisions": clock["decisions"]})
        except BaseException as ex:
            if not ends_segment(ex):
                raise
            counters["log_errors"] += 1

    def fail(s: Slot, t_last: int) -> None:
        counters["errors"] += 1
        err, ex = traceback.format_exc(limit=8), sys.exc_info()[1]
        etype = type(ex).__name__ if ex is not None else None
        env = s.env  # None when the segment failed before its env existed
        if sink is not None and hasattr(env, "log_idx") and not getattr(env, "_done", False):
            # The choices so far replay to the decision that failed (#1097): keep them.
            try:
                from .actionlog import worker_error_record

                sink.write(worker_error_record(env, err, extra={"uid": s.uid, "etype": etype}))
            except BaseException as lex:
                if not ends_segment(lex):
                    raise
                counters["log_errors"] += 1
        ends.append(
            error_record(
                s.uid,
                s.k,
                t_last,
                s.kind,
                s.cell,
                err,
                stamps,
                etype,
                env_counts=getattr(env, "counts", None),
                seed=int(getattr(env, "seed", -1)),
                f9_texts=f9_samples(env),
            )
        )
        s.d = None

    while True:
        toks: list[Any] = []
        uids: list[int] = []
        ts: list[int] = []
        cells: list[int] = []
        boots: list[bool] = []
        metas: list[tuple[int, ...]] = []
        r_prev: list[float] = []
        feats: list[Any] = []
        for s in slots:
            while True:
                if s.d is None:
                    try:
                        start(s)
                        fails["start"] = 0
                    except BaseException as ex:  # a Rust panic is a BaseException
                        if not ends_segment(ex):  # a build mismatch, an interrupt
                            raise
                        fails["start"] += 1
                        if fails["start"] > MAX_START_FAILS:
                            raise
                        fail(s, -1)
                        continue
                d = s.d
                if not isinstance(d, (E2EDecision, RsDecision)):  # an E2ETerminal
                    ends.append(
                        end_record(
                            s.uid,
                            s.k,
                            s.t - 1,
                            d,
                            s.kind,
                            s.cell,
                            stamps,
                            f9_texts=f9_samples(s.env),
                        )
                    )
                    log_end(s, d)
                    s.d = None
                    continue
                try:
                    tk = d.tok if rust else TK.encode(d, rare=rare, priv=True)
                    if tk.n_classes != d.n_classes:
                        raise C.SchemaError(f"{tk.n_classes} token classes for {d.n_classes}")
                    f = feats_of(d) if (want["feats"] and not d.is_combat) else None
                except BaseException as ex:  # the tokenizer's getters are Rust calls
                    if not ends_segment(ex):
                        raise
                    fail(s, s.t - 1)
                    continue
                unit_kind, unit_idx = d.unit
                metas.append(
                    (
                        s.k,
                        s.uid,
                        s.t,
                        int(d.is_combat),
                        int(d.dtype),
                        int(d.cell),
                        UNIT_KIND[unit_kind],
                        int(unit_idx),
                        int(d.n_classes),
                        int(bool(d.bootstrap)),
                        int(s.kind == "practice"),
                    )
                )
                r_prev.append(float(d.reward))
                toks.append(tk)
                uids.append(s.uid)
                ts.append(s.t)
                cells.append(int(d.cell))
                boots.append(bool(d.bootstrap))
                feats.append(f)
                s.t += 1
                if d.bootstrap:  # a practice segment's value-only end row
                    ends.append(
                        end_record(
                            s.uid,
                            s.k,
                            s.t - 1,
                            s.env.terminal,
                            s.kind,
                            s.cell,
                            stamps,
                            f9_texts=f9_samples(s.env),
                        )
                    )
                    log_end(s, s.env.terminal)
                break
        stats = None
        if rounds % 64 == 0:
            src: Counter = Counter()
            for s in slots:
                src.update(s.source.stats)
            stats = {
                "practice": dict(queues.stats),
                "sources": dict(src),
                "worker": dict(counters),
                "cells": {c: queues.size(c) for c in C.CELLS},
                "actions": None
                if sink is None
                else {
                    "records": sink.records,
                    "bytes": sink.written,
                    "dropped": sink.dropped,
                    "full": bool(sink.full),
                },
            }
        extra = {"feats": feats if want["feats"] else None, "stats": stats, "stamps": stamps}
        rb = TK.row_batch(toks, run_uid=uids, t=ts, cell=cells, bootstrap=boots)
        rounds += 1
        sent, ends = ends, []
        conn.send(
            (
                "obs",
                wid,
                np.asarray(metas, np.int64).reshape(-1, len(META_COLS)),
                np.asarray(r_prev, np.float64),
                rb,
                extra,
                sent,
            )
        )
        msg = conn.recv()
        if msg[0] == "stop":
            conn.close()
            return
        _, acts, info = msg
        clock["decisions"] = int(info["decisions"])
        want["feats"] = bool(info.get("feats"))
        if rust and want["feats"]:
            raise RuntimeError(
                "rust_env workers cannot make the run teacher's features (rsenv rows carry "
                "no run_obs): run with --rust-env false while the run teacher is on"
            )
        for s, a in zip(slots, acts, strict=True):
            d = s.d
            if d.bootstrap:
                s.d = None  # the segment ended at this row; its end is already sent
                continue
            try:
                s.d = s.env.step(int(a))
            except BaseException as ex:
                if not ends_segment(ex):
                    raise
                fail(s, s.t - 1)  # its end goes out with the next round
