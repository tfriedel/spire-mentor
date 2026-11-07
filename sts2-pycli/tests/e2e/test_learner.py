"""The e2e learner (WP4c, plan §6-§8): cuts, the update loop, states and resume, stops.

Most tests drive `ppo.e2e.learner.Learner` with `FakeSource`: scripted
segments of `net.synthetic_row` rows (valid `contracts` rows) behind the
rollout source interface (`ready`, `recv`, `send`, `stop`), with rewards that
depend on the actions the learner sends back. The update runs in-process
(`updater="local"`, one CPU rank) on a tiny net. Names follow the plan's
unit tests: T3 (bootstrap rows at the cut), T9 (the KL stop), T10 (one
optimizer step per minibatch) and T11 (a kill at iteration 7, 150 or a random
one reproduces the next 3 iterations), the learner halves; WP4a's
`test_algo.py` has the algorithm halves. The last test runs the real worker
processes on the real v0.111.0 engine (G-skeleton).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sts2_client
import torch
from ppo.e2e import actors, ckpt
from ppo.e2e import algo as A
from ppo.e2e import contracts as C
from ppo.e2e import fillcheck as FC
from ppo.e2e import learner as L
from ppo.e2e import ops_contract as OC
from ppo.e2e import schedules as S
from ppo.e2e.net import load_checkpoint, synthetic_row
from ppo.e2e.rowstore import ACTING, concat
from ppo.e2e.update import export_variant
from ppo.wheel_guard import python_source_digest, wheel_identity

V111 = "v0.111.0"
TINY = {"dim": 32, "n_layers": 1, "n_heads": 2, "ffn": 64, "head_hidden": 32, "num_hidden": 64}
#: 64 decisions per iteration: U0 ends at iteration 15, λ_KS reaches 0 at 130.
ITER = 64
SCHED = {
    "u0_end": 15 * ITER,
    "ks_end": 130 * ITER,
    "ent_end": 400 * ITER,
    "budget_decisions": 10**12,
    "anneal_start": S.ANNEAL_START,
    "kl_freeze_at": S.ANNEAL_START,
}


@pytest.fixture(autouse=True)
def _v111(monkeypatch: pytest.MonkeyPatch) -> None:
    """The learner's stamps need the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)


# ── a scripted rollout source ────────────────────────────────────────────


class _Slot:
    def __init__(self) -> None:
        self.count = -1
        self.active = False
        self.uid = -1
        self.t = 0
        self.length = 0
        self.kind = "run"
        self.cell = C.CELL_NONE
        self.err_at: int | None = None
        self.rng: np.random.Generator | None = None
        self.r_prev = 0.0
        self.row: dict[str, np.ndarray] | None = None
        self.boot = False
        self.total = 0.0


def _term(**kw: Any) -> SimpleNamespace:
    base = {
        "counts": Counter(),
        "reward": 0.0,
        "outcome": "lost",
        "won": False,
        "cap": None,
        "floor": 1,
        "act": 1,
        "bosses": 0,
        "reached_boss2": False,
        "rows": 0,
        "steps": 0,
        "seed": 0,
        "error": None,
        "total_reward": 0.0,
        "unassigned_reward": 0.0,
    }
    return SimpleNamespace(**(base | kw))


def _hash01(*xs: int) -> float:
    return L.mix64(*xs) / 2.0**64


class FakeSource:
    """`ProcessSource`'s interface over scripted segments (no engine, no processes).

    Every worker has `slots` env slots that play segments back to back, as
    `actors.worker_main` does: one row per slot per round, segment uids
    `actors.segment_uid(start_it, ...)`, a run's end record with the next
    round, a practice segment's bootstrap row and its end in one message, and
    a step error as an error record with the next round. Segment contents
    depend only on (`seed`, `start_it`, worker, slot, segment count); rewards
    also depend on the actions.
    """

    def __init__(
        self,
        wcfg: dict[str, Any],
        *,
        workers: int = 2,
        slots: int = 4,
        length: tuple[int, int] = (4, 24),
        practice_p: float = 0.2,
        error_p: float = 0.04,
        start_error_p: float = 0.02,
        seed: int = 7,
        stamps: dict[str, Any] | None = None,
        on_round: Any = None,
        f9_hit_round: int = -1,
        floor_fn: Any = None,
        total_bias: float = 0.0,
    ) -> None:
        self.session = int(wcfg["start_it"])
        self.wcfg = wcfg
        self.seed = seed
        self.length = length
        self.practice_p = practice_p
        self.error_p = error_p
        self.start_error_p = start_error_p
        self.stamps = stamps or {"wheel": wheel_identity(), "py_src": python_source_digest()}
        self.slots = [[_Slot() for _ in range(slots)] for _ in range(workers)]
        self.ends: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
        self.want_feats = False
        self.rounds = 0
        self.on_round = on_round
        self.f9_hit_round = f9_hit_round
        self.floor_fn = floor_fn  # run end count -> the run's floor (None: from its length)
        self.total_bias = total_bias  # added to a finished run's env total (a T4 break)
        self.run_ends = 0
        self.stopped = False
        self.sent_rows = 0

    # ── the source interface ──

    def ready(self) -> list[int]:
        return list(range(len(self.slots)))

    def recv(self, w: int) -> Any:
        if w == 0:
            if self.on_round is not None:
                self.on_round(self)
            self.rounds += 1
        metas, r_prev, rows, feats = [], [], [], []
        for k, s in enumerate(self.slots[w]):
            while not s.active:
                self._start(w, k, s)
            boot = s.kind == "practice" and s.t == s.length - 1
            # A practice segment ends at its first out-of-combat decision (SCHEMA.md 9.1).
            comb = bool(s.rng.random() < 0.7) and not boot
            row = synthetic_row(s.rng, combat=comb, t=s.t, body=(2, 10), priv=(0, 4))
            row["run_uid"][0] = s.uid
            row["cell"][0] = s.cell
            row["bootstrap"][0] = boot
            comb = bool(row["is_combat"][0])
            unit = (0, s.t // 6) if comb else (1, s.t // 15)
            metas.append(
                (
                    k,
                    s.uid,
                    s.t,
                    int(comb),
                    int(row["dtype"][0]),
                    s.cell,
                    unit[0],
                    unit[1],
                    int(row["n_classes"][0]),
                    int(boot),
                    int(s.kind == "practice"),
                )
            )
            r_prev.append(s.r_prev if s.t else 0.0)
            rows.append(row)
            feats.append(("feats", s.uid, s.t) if (self.want_feats and not comb) else None)
            s.row, s.boot = row, boot
            if boot:
                self.ends[w].append(self._end(s, k, "bootstrap", 0.0))
        sent, self.ends[w] = self.ends[w], []
        extra = {"feats": feats if self.want_feats else None, "stats": None, "stamps": self.stamps}
        self.sent_rows += len(rows)
        return (
            "obs",
            w,
            np.asarray(metas, np.int64).reshape(-1, len(actors.META_COLS)),
            np.asarray(r_prev, np.float64),
            concat(rows),
            extra,
            sent,
        )

    def send(self, w: int, msg: Any) -> None:
        _, acts, info = msg
        self.want_feats = bool(info.get("feats"))
        for k, (s, a) in enumerate(zip(self.slots[w], acts, strict=True)):
            if s.boot:
                s.active = False
                continue
            n = int(s.row["n_classes"][0])
            good = int(L.mix64(self.seed, s.uid, s.t) % n)
            r = 0.01 if int(a) == good else -0.002
            s.total += r
            if s.err_at == s.t:
                rec = actors.error_record(
                    s.uid, k, s.t, s.kind, None, "scripted step error", self.stamps, "StepError"
                )
                self.ends[w].append(rec)
                s.active = False
            elif s.t == s.length - 1:
                u = float(s.rng.random())
                won = u < 0.2
                outcome = "won" if won else ("capped" if u > 0.8 else "lost")
                last = r + (1.0 if won else 0.0)
                s.total += last - r + self.total_bias  # the env's total counts the win too
                self.ends[w].append(self._end(s, k, outcome, last))
                s.active = False
            else:
                s.r_prev = r
                s.t += 1

    def stop(self) -> None:
        self.stopped = True

    # ── segments ──

    def _start(self, w: int, k: int, s: _Slot) -> None:
        s.count += 1
        s.rng = np.random.default_rng([self.seed, self.session, w, k, s.count])
        s.uid = actors.segment_uid(self.session, w, k, s.count)
        s.t, s.r_prev, s.total, s.boot = 0, 0.0, 0.0, False
        if s.rng.random() < self.start_error_p:  # the segment fails before its first row
            self.ends[w].append(
                actors.error_record(
                    s.uid, k, -1, "run", None, "scripted start error", self.stamps, "StartError"
                )
            )
            return
        s.length = int(s.rng.integers(self.length[0], self.length[1] + 1))
        s.kind = "practice" if s.rng.random() < self.practice_p else "run"
        s.cell = int(s.rng.integers(0, len(C.CELLS))) if s.kind == "practice" else C.CELL_NONE
        s.err_at = int(s.rng.integers(0, s.length)) if s.rng.random() < self.error_p else None
        s.active = True

    def _end(self, s: _Slot, k: int, outcome: str, last: float) -> dict[str, Any]:
        counts = Counter({"fights/monster": 1 + s.length // 8, "f9/checks": 3})
        if self.f9_hit_round >= 0 and self.rounds >= self.f9_hit_round:
            counts["f9/violations"] = 1
        cap = "turns" if outcome == "capped" else None
        floor = 1 + s.length // 3
        if s.kind == "run" and outcome != "bootstrap":
            if self.floor_fn is not None:
                floor = int(self.floor_fn(self.run_ends))
            self.run_ends += 1
        term = _term(
            counts=counts,
            reward=last,
            outcome=outcome,
            won=outcome == "won",
            cap=cap,
            floor=floor,
            bosses=int(outcome == "won"),
            rows=s.t + 1,
            total_reward=s.total,
        )
        cell = C.CELLS[s.cell] if s.kind == "practice" else None
        return actors.end_record(s.uid, k, s.t, term, s.kind, cell, self.stamps)


def spy_shipped(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, np.ndarray]]:
    """Copies of every row block the learner ships (`learner.pack_rows`), as sent."""
    from ppo.e2e import rowstore

    out: list[dict[str, np.ndarray]] = []
    real = L.pack_rows

    def spy(src: Any, table: Any, threads: int = 1) -> Any:
        shm, layout, secs = real(src, table, threads)
        rb = rowstore.batch_from_shm(shm, layout)
        out.append({k: np.array(v, copy=True) for k, v in rb.items()})
        del rb
        return shm, layout, secs

    monkeypatch.setattr(L, "pack_rows", spy)
    return out


# ── fake teachers (deterministic class probabilities) ────────────────────


def _fake_logp(n: int, *key: int) -> np.ndarray:
    z = np.asarray([_hash01(*key, j) for j in range(n)]) * 3.0
    return z - np.log(np.exp(z).sum())


class FakeCombatTeacher:
    rare_json = None

    def logp_rows(self, rb: dict[str, np.ndarray], rows: list[int]) -> list[np.ndarray]:
        out = []
        for i in rows:
            assert bool(rb["is_combat"][i])
            out.append(_fake_logp(int(rb["n_classes"][i]), int(rb["run_uid"][i]), int(rb["t"][i])))
        return out


class FakeRunTeacher:
    def logp(self, feats: list[Any], classes: list[np.ndarray], n_classes: list[int]) -> list:
        out = []
        for f, cls, n in zip(feats, classes, n_classes, strict=True):
            assert f[0] == "feats" and int(np.max(cls)) == n - 1
            out.append(_fake_logp(int(n), int(f[1]), int(f[2]), 1))
        return out


class TeacherLearner(L.Learner):
    """A learner with the fake teacher pair (combat and run)."""

    def _teacher_flags(self) -> tuple[bool, bool]:
        return True, True

    def _load_teachers(self) -> Any:
        return SimpleNamespace(combat=FakeCombatTeacher(), run=FakeRunTeacher())


class Clock:
    """Wall time as a function of the learner's decision clock (both runs of T11 agree).

    One iteration (`ITER` decisions) is 15 minutes; the lr anneal starts at
    iteration `anneal_at`.
    """

    def __init__(self, anneal_at: int = 96) -> None:
        self.t0 = S.ANNEAL_START - anneal_at * 900.0
        self.learner: L.Learner | None = None

    def __call__(self) -> float:
        d = self.learner.decisions if self.learner is not None else 0
        return self.t0 + d * (900.0 / ITER)


def make_cfg(save_dir: Path, **kw: Any) -> L.Cfg:
    base: dict[str, Any] = {
        "save_dir": str(save_dir),
        "workers": 2,
        "n_envs": 4,
        "infer_device": "cpu",
        "update_devices": "cpu",
        "bf16": False,
        "rank_threads": 2,
        "sync_rounds": True,
        "sync_updates": True,
        "net": TINY,
        "iter_decisions": ITER,
        "epochs": 2,
        "minibatch": 32,
        "strata_per_mb": 2,
        "probe_strata": 2,
        "fwd_chunk": 64,
        "max_iters": 4,
        "heartbeat_s": 0.0,
        "throughput_s": 0.05,
        "state_every": 10**12,
        "full_every": 10**12,
        "pin_every": 10**12,
        "slim_every": 10**12,
        "k_gates": [],
        "sched": dict(SCHED),
        # Scripted rows are not real rows: the fill check (FILL_MISMATCH) would refuse
        # them. The real-worker tests and the fill-check tests turn it on.
        "fill_check": False,
    }
    return L.Cfg(**(base | kw))


def make_learner(
    cfg: L.Cfg,
    *,
    teachers: bool = True,
    clock: Clock | None = None,
    cls: type[L.Learner] | None = None,
    on_round: Any = None,
    updater: str = "local",
    **src: Any,
) -> L.Learner:
    """A learner on `FakeSource`s; `on_round(learner, source)` runs before every round."""
    clock = clock or Clock()
    klass = cls or (TeacherLearner if teachers else L.Learner)
    box: dict[str, L.Learner] = {}
    hook = None if on_round is None else (lambda src_: on_round(box["lr"], src_))
    lr = klass(
        cfg,
        source=lambda wcfg: FakeSource(wcfg, on_round=hook, **src),
        updater=updater,
        clock=clock,
    )
    box["lr"] = lr
    clock.learner = lr
    return lr


def _write_now(lr: L.Learner, path: Path) -> None:
    """Write a control file and let the learner see it at its next round (it polls the
    save dir at most once a second)."""
    path.write_text("operator stop\n" if path.name == OC.DONE else "", encoding="utf-8")
    lr._t_done = 0.0


# ── serving: length groups (S1-lean) ─────────────────────────────────────


def test_length_groups_split_sorted_rows_into_equal_groups() -> None:
    from ppo.e2e.teachers import MIN_GROUP_ROWS, length_groups

    lens = np.random.default_rng(3).integers(1, 200, 5 * MIN_GROUP_ROWS)
    parts = length_groups(lens, 4)
    assert len(parts) == 4
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(lens)))
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    for a, b in itertools.pairwise(parts):
        assert lens[a].max() <= lens[b].min()  # sorted: each group pads to its own length
    # Too few rows for two groups of MIN_GROUP_ROWS, or one group asked: the rows in order.
    for n, g in ((2 * MIN_GROUP_ROWS - 1, 4), (len(lens), 1), (len(lens), 0)):
        (one,) = length_groups(lens[:n], g)
        assert one.tolist() == list(range(n))
    assert len(length_groups(lens[: 2 * MIN_GROUP_ROWS], 4)) == 2


def _serve_round(
    tmp_path: Path, groups: int, **kw: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One serving round of 2 x 48 scripted rows at β_act 0.5 with the fake teachers."""
    lr = make_learner(make_cfg(tmp_path, serve_groups=groups, **kw))
    lr.decisions = lr.algo.cfg.u0_end // 2
    assert 0 < lr._beta_act() < 1 and lr._teacher_on()
    src = FakeSource(lr.worker_cfg(), workers=2, slots=48, error_p=0.0, start_error_p=0.0)
    src.want_feats = True
    msgs = [src.recv(w) for w in range(2)]
    from ppo.e2e.rowstore import Chunk

    ch = Chunk(concat([m[4] for m in msgs]))
    metas = np.concatenate([m[2] for m in msgs])
    feats = [f for m in msgs for f in m[5]["feats"]]
    rows = [lr._store(ch, i, m, 0.0, 0) for i, m in enumerate(metas)]
    st = lr._new_stats()
    lr.gen.manual_seed(11)
    acts = lr._serve(rows, ch, metas, feats, st)
    assert np.array_equal(acts, ch.rb["action"])
    lr.stop()
    return {k: ch.rb[k].copy() for k in (*ACTING, "tprob")}, st


def test_length_grouped_serving_matches_one_batch(tmp_path: Path) -> None:
    """`serve_groups` changes only the student's float rounding: every teacher field,
    actor, action and agreement count is the one-batch round's, log-probs and values
    agree to fp32 rounding."""
    one, st1 = _serve_round(tmp_path / "one", 1)
    grp, st4 = _serve_round(tmp_path / "grp", 3)
    assert (one["actor"] != A.ACTOR_STUDENT).any() and (one["actor"] == A.ACTOR_STUDENT).any()
    assert (one["t_src"] == 1).any() and (one["t_src"] == 2).any()
    for k in ("actor", "t_src", "tprob", "action", "iter"):
        assert np.array_equal(one[k], grp[k], equal_nan=k == "tprob"), k
    np.testing.assert_allclose(grp["logp_behav"], one["logp_behav"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(grp["v_behav"], one["v_behav"], rtol=0, atol=1e-5)
    for k in ("agree", "agree_n", "teacher_rows"):
        assert st1[k] == st4[k], k


def test_packed_serving_matches_the_padded_round(tmp_path: Path) -> None:
    """`packed_serve` changes only the student's float rounding: every teacher field,
    actor, action and agreement count is the padded round's, log-probs and values
    agree to fp32 rounding."""
    one, st1 = _serve_round(tmp_path / "pad", 1)
    pk, st2 = _serve_round(tmp_path / "packed", 1, packed_serve=True)
    assert (one["actor"] != A.ACTOR_STUDENT).any() and (one["t_src"] == 1).any()
    for k in ("actor", "t_src", "tprob", "action", "iter"):
        assert np.array_equal(one[k], pk[k], equal_nan=k == "tprob"), k
    np.testing.assert_allclose(pk["logp_behav"], one["logp_behav"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(pk["v_behav"], one["v_behav"], rtol=0, atol=1e-5)
    for k in ("agree", "agree_n", "teacher_rows"):
        assert st1[k] == st2[k], k


def test_the_packed_serve_flag_parses_and_defaults_off() -> None:
    p = L.build_parser()
    base = ["train", "--save-dir", "x"]
    assert not L.cfg_from_args(p.parse_args(base)).packed_serve
    assert L.cfg_from_args(p.parse_args([*base, "--packed-serve", "true"])).packed_serve


def _meta(uid: int, t: int, *, boot: bool = False, practice: bool = False) -> np.ndarray:
    m = np.zeros(len(actors.META_COLS), np.int64)
    m[actors.M["uid"]], m[actors.M["t"]] = uid, t
    m[actors.M["dtype"]] = C.DT["MAP"]
    m[actors.M["n_classes"]] = 3
    m[actors.M["cell"]] = 0 if practice else C.CELL_NONE
    m[actors.M["boot"]] = int(boot)
    m[actors.M["practice"]] = int(practice)
    return m


def _chunk(n: int, seed: int) -> Any:
    from ppo.e2e.rowstore import Chunk

    rng = np.random.default_rng(seed)
    return Chunk(concat([synthetic_row(rng, combat=False, t=i, body=(2, 6)) for i in range(n)]))


def test_cut_ships_ended_open_and_practice_rows_and_drops_errors(tmp_path: Path) -> None:
    lr = make_learner(make_cfg(tmp_path), teachers=False)
    try:
        ch = _chunk(16, 1)
        for j in range(ch.n):  # the learner checks (run_uid, t) against the meta
            ch.rb["run_uid"][j] = 0
        st = lr._new_stats()

        def store(uid: int, t: int, r_prev: float, j: int, **kw: Any) -> L.Row:
            ch.rb["run_uid"][j], ch.rb["t"][j] = uid, t
            return lr._store(ch, j, _meta(uid, t, **kw), r_prev, lr.n_dispatched)

        def end(uid: int, t_last: int, outcome: str, r_last: float = 0.0, **kw: Any) -> None:
            rec = {"uid": uid, "t_last": t_last, "outcome": outcome, "r_last": r_last}
            rec |= {"floor": 3, "won": outcome == "won", "bosses": 0, "reached_boss2": False}
            lr._ingest_end(rec | kw, st)

        # Run 1 ends this iteration; run 2 stays open; practice segment 3 ends in its
        # bootstrap row; run 4 errors.
        store(1, 0, 0.0, 0)
        store(1, 1, 0.1, 1)
        store(2, 0, 0.0, 2)
        store(1, 2, 0.2, 3)
        store(2, 1, 0.3, 4)
        store(3, 0, 0.0, 5, practice=True)
        store(3, 1, 0.4, 6, practice=True, boot=True)
        store(4, 0, 0.0, 7)
        end(1, 2, "won", 1.5)
        end(3, 1, "bootstrap", cell="ELITE_ACT1")
        end(4, 0, "error", error="boom")
        with pytest.raises(RuntimeError, match="out of order"):
            store(2, 5, 0.0, 8)
        cols, _rows, drop, ast = lr.cut()
        by = {(int(u), int(t)): i for i, (u, t) in enumerate(zip(cols["run_uid"], cols["t"]))}
        assert drop == [4] and ast["dropped_error_rows"] == 1
        assert set(by) == {(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)}

        def flags(u: int, t: int) -> tuple[bool, bool, float]:
            i = by[u, t]
            return bool(cols["done"][i]), bool(cols["bootstrap"][i]), float(cols["reward"][i])

        assert flags(1, 0) == (False, False, pytest.approx(0.1))
        assert flags(1, 2) == (True, False, 1.5)  # the terminal row
        assert flags(2, 0) == (False, False, pytest.approx(0.3))
        assert flags(2, 1) == (False, True, 0.0)  # the cut's bootstrap row: value only
        assert flags(3, 0)[:2] == (False, False)
        assert flags(3, 1) == (True, True, 0.0)  # a practice end: bootstrap with done set
        assert list(lr.segs) == [2] and [r.t for r in lr.segs[2].pending] == [1]

        # Next iteration: the open run's bootstrap row ships again as an ordinary row
        # once its reward arrives; a segment with no new row stays dormant.
        lr.n_dispatched += 1
        store(5, 0, 0.0, 9)
        cols, _rows, _drop, ast = lr.cut()
        assert ast["fresh_rows"] == 1 and set(cols["run_uid"].tolist()) == {5}
        assert ast["dormant_segments"] == 1 and 2 in lr.segs
        lr.n_dispatched += 1
        store(2, 2, 0.7, 10)
        cols, _rows, _drop, _ast = lr.cut()
        got = {
            (int(u), int(t)): (bool(b), float(r))
            for u, t, b, r in zip(
                cols["run_uid"], cols["t"], cols["bootstrap"], cols["reward"], strict=True
            )
        }
        assert got[2, 1] == (False, pytest.approx(0.7)) and got[2, 2] == (True, 0.0)
        # A reward that never arrived is an error, not a zero.
        lr.n_dispatched += 1
        store(6, 0, 0.0, 11)
        store(6, 1, 0.1, 12)
        lr.segs[6].pending[0].reward = None
        with pytest.raises(RuntimeError, match="no reward"):
            lr.cut()
    finally:
        lr.stop()


# ── the loop: phases, steps, stats rows, files (T10, ops contract) ────────


def _check_rows(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        assert OC.check_stats_row(r) == [], r["it"]
        assert r["reward_mismatch"] == 0, r["reward_mismatch_sample"]  # T4 at run time
        assert sum(r["error_types"].values()) == r["errors"], r["error_types"]


def test_rows_whose_rewards_miss_the_env_total_are_counted(tmp_path: Path) -> None:
    """T4, the learner half: a finished segment whose stored rewards do not add up
    to the env's `total_reward` is counted in the stats row (and sampled)."""
    lr = make_learner(make_cfg(tmp_path, max_iters=3), total_bias=0.5, error_p=0.0)
    lr.train()
    bad = sum(r["reward_mismatch"] for r in lr.stats_rows)
    runs = sum(r["runs"] for r in lr.stats_rows)
    assert 0 < bad == runs  # every finished run; practice ends keep their totals
    sample = lr.stats_rows[-1]["reward_mismatch_sample"][0]
    assert abs(sample["env"] - sample["rows"] - 0.5) < 1e-9


def test_iterations_follow_the_warm_start_and_the_ops_contract(tmp_path: Path) -> None:
    cfg = make_cfg(
        tmp_path,
        max_iters=6,
        state_every=2 * ITER,
        full_every=3 * ITER,
        slim_every=3 * ITER,
        pin_every=4 * ITER,
        k_gates=[["k2", 5 * ITER]],
    )
    lr = make_learner(cfg)
    assert lr.train() == "max_iters"
    rows = lr.stats_rows
    assert [r["it"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r["update"]["phase"] for r in rows] == ["V-warm"] * 3 + ["U0"] * 3
    _check_rows(rows)
    for r in rows:
        u = r["update"]
        assert math.isfinite(r["losses"]["value"]) and r["nonfinite"] == 0
        # T10: one optimizer step per minibatch per epoch, every epoch run (U0: no KL stop).
        n_mb = max(1, round(u["rows_trained"] / cfg.minibatch))
        assert u["steps"] == n_mb * u["epochs_run"] and u["epochs_run"] == cfg.epochs
        assert not u["kl_stopped"]
        assert r["sampler_seed"] == L.sampler_seed(cfg.seed, 0, r["it"] - 1)
    assert rows[0]["losses"]["policy"] is None  # V-warm trains the value only
    assert all(r["losses"]["policy"] is not None for r in rows[3:])
    assert sum(r["teacher_rows"] for r in rows) > 0  # U0: the teachers act
    assert any(r["agreement"] for r in rows)
    assert sum(r["practice_segments"] for r in rows) > 0
    assert sum(r["errors"] for r in rows) > 0
    assert sum(r["caps"]["fights_capped"] for r in rows) > 0
    # The files of the ops contract.
    sd = ckpt.SaveDir(tmp_path)
    dec = int(lr.algo.clocks.decisions)
    assert dec == rows[-1]["decisions"] >= 6 * ITER
    assert sd.state.is_file() and ckpt.state_decisions(sd) == dec
    assert len(list(sd.full_dir.glob("full_*.pt"))) == 2
    reasons = sorted(r.reason for r in ckpt.pinned_states(sd))
    assert reasons == ["final", "k2", "milestone"]
    assert OC.read_json(tmp_path / OC.DONE)["reason"] == "max_iters"
    assert OC.read_json(tmp_path / OC.HEARTBEAT)["decisions"] == dec
    tp = [
        json.loads(x)
        for x in (tmp_path / OC.THROUGHPUT).read_text(encoding="utf-8").split("\n")
        if x
    ]
    assert tp and all(OC.check_throughput_row(x) == [] for x in tp)
    on_disk = [
        json.loads(x) for x in (tmp_path / OC.STATS).read_text(encoding="utf-8").split("\n") if x
    ]
    assert [x["it"] for x in on_disk] == [1, 2, 3, 4, 5, 6]
    assert json.loads((tmp_path / "cfg.json").read_text(encoding="utf-8"))["max_iters"] == 6
    assert not (tmp_path / OC.CKPT_DIR / "rank_init.pt").exists()
    # The state has a hold-back queue but not its rows' inputs (a resume drops them).
    st = torch.load(sd.state, map_location="cpu", weights_only=True)
    assert st["format"] == L.STATE_FORMAT and st["learner"]["decisions"] == dec
    assert st["held"] is None and len(A.AlgoState.from_dict(st["algo"]).hold) > 0
    assert st["stamps"]["schema_sha256"] == C.schema_sha256()


def test_shipped_table_is_a_valid_row_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every update's table passes `contracts.validate_row_batch` with the
    algorithm's flags (SCHEMA.md 9.1, revision 2): cut rows `bootstrap` alone,
    practice ends `bootstrap` and `done` on an out-of-combat row with a cell,
    terminals `done` alone, and reward 0 on every bootstrap row."""
    seen = spy_shipped(monkeypatch)
    lr = make_learner(make_cfg(tmp_path, max_iters=4))
    lr.train()
    assert len(seen) >= 4
    for rb in seen:
        C.validate_row_batch(rb)
    done = np.concatenate([rb["done"] for rb in seen])
    boot = np.concatenate([rb["bootstrap"] for rb in seen])
    cell = np.concatenate([rb["cell"] for rb in seen])
    comb = np.concatenate([rb["is_combat"] for rb in seen])
    reward = np.concatenate([rb["reward"] for rb in seen])
    end = done & boot
    assert end.any() and (boot & ~done).any() and (done & ~boot).any()
    assert (cell[end] != C.CELL_NONE).all() and not comb[end].any()
    assert (reward[boot] == 0).all() and (reward[~boot] != 0).any()
    tsrc = np.concatenate([rb["t_src"] for rb in seen])
    assert set(np.unique(tsrc).tolist()) <= {0, 1, 2} and (tsrc > 0).any()


# ── T9, the update half: the KL stop acts in U1 and U2 only ──────────────


def test_kl_stop_skips_epochs_only_after_u0(tmp_path: Path) -> None:
    """With a threshold every probe KL exceeds, U1 and U2 stop after one epoch
    and V-warm and U0 run them all. (A tiny positive threshold is not enough:
    the tiny net's exact KL after one epoch is float32 noise, ±5e-8.)"""
    sched = dict(SCHED, u0_end=6 * ITER, ks_end=8 * ITER, kl_stop=-1.0, value_only_iters=1)
    cfg = make_cfg(tmp_path, max_iters=10, epochs=3, sched=sched)
    lr = make_learner(cfg)
    lr.train()
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for r in lr.stats_rows:
        by_phase.setdefault(r["update"]["phase"], []).append(r["update"])
    assert {"U0", "U1"} <= set(by_phase)
    for u in by_phase["U0"] + by_phase.get("V-warm", []):
        assert u["epochs_run"] == 3 and not u["kl_stopped"]
    stopped = [u for p in ("U1", "U2") for u in by_phase.get(p, [])]
    assert stopped and all(u["kl_stopped"] and u["epochs_run"] == 1 for u in stopped)


# ── held rows: one block of their own ────────────────────────────────────


@pytest.mark.parametrize("idx", [[4, 0, 5, 2], [], [3]])
def test_take_to_shm_is_batch_to_shm_of_take(idx: list[int]) -> None:
    from ppo.e2e.rowstore import batch_from_shm, batch_to_shm, take, take_to_shm

    src = _chunk(6, 4)
    want = take(src, idx)
    shm, layout = take_to_shm(src, idx)
    ref_shm, ref_layout = batch_to_shm(want)
    try:
        assert layout == ref_layout  # `ranks.to_shm`'s layout, so the ranks' reader fits
        assert shm.size == ref_shm.size
        got = batch_from_shm(shm, layout)
        for f in C.ROW_FIELDS:  # `tprob` is NaN where a row has no teacher
            nan = np.issubdtype(want[f.name].dtype, np.floating)
            assert got[f.name].dtype == want[f.name].dtype, f.name
            assert np.array_equal(got[f.name], want[f.name], equal_nan=nan), f.name
        del got
    finally:
        for b in (shm, ref_shm):
            b.close()
            b.unlink()


def _refs(case: str) -> list[tuple[Any, int]]:
    """Rows of several chunks in a joined table's shape: a held block's rows in
    order first, then rows of other chunks interleaved (a segment's rows sit in
    different rounds)."""
    held, r1, r2 = _chunk(5, 11), _chunk(7, 12), _chunk(6, 13)
    if case == "empty":
        return []
    if case == "one":
        return [(r1, 3)]
    if case == "in_order":
        return [(r1, i) for i in range(r1.n)]
    if case == "held_gap":  # a drop took held row 2: the held block copies by index
        return [(held, i) for i in (0, 1, 3, 4)] + [(r1, 5), (r2, 2)]
    if case == "reversed":  # rows that land on one span from scattered places
        return [(r1, i) for i in reversed(range(r1.n))]
    fresh = [(r1, 2), (r2, 0), (r1, 6), (r2, 5), (r1, 0), (r2, 3), (r1, 4), (r2, 1)]
    return [(held, i) for i in range(held.n)] + fresh + [(r1, 2)]  # a row twice too


@pytest.mark.parametrize("block", [None, 1])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("case", ["table", "empty", "one", "in_order", "held_gap", "reversed"])
def test_gather_to_shm_is_batch_to_shm_of_gather(
    case: str, threads: int, block: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cut's one-copy block holds the bytes the gathered row batch held, in the
    layout the ranks read (S1-cut). `block=1`: every scattered row copies in a
    block of its own (the row blocks that keep temporaries small)."""
    from ppo.e2e import rowstore
    from ppo.e2e.rowstore import batch_from_shm, batch_to_shm, gather, gather_to_shm

    if block is not None:
        monkeypatch.setattr(rowstore, "COPY_BLOCK_BYTES", block)
        real = rowstore._blocks
        monkeypatch.setattr(  # past the small-chunk shortcut too
            rowstore,
            "_blocks",
            lambda ch, idx, pos, like: (
                [(idx[i : i + 1], pos[i : i + 1]) for i in range(len(idx))]
                if len(idx) <= 64
                else real(ch, idx, pos, like)
            ),
        )
    refs = _refs(case)
    shm, layout = gather_to_shm(refs, threads=threads)
    ref_shm, ref_layout = batch_to_shm(gather(refs))
    try:
        assert layout == ref_layout and shm.size == ref_shm.size
        got, want = batch_from_shm(shm, layout), batch_from_shm(ref_shm, ref_layout)
        for f in C.ROW_FIELDS:  # bytes: `tprob` is NaN where a row has no teacher
            assert got[f.name].dtype == want[f.name].dtype, f.name
            assert np.array_equal(got[f.name].view(np.uint8), want[f.name].view(np.uint8)), f.name
        del got, want
    finally:
        for b in (shm, ref_shm):
            b.close()
            b.unlink()


def test_gather_to_shm_allocates_no_large_array() -> None:
    """A large chunk whose rows copy by index (a held block after a drop) copies in
    row blocks: no temporary reaches numpy's 4 MiB huge-page threshold (on a
    fragmented host each such allocation can stall in direct compaction)."""
    import tracemalloc

    from ppo.e2e.rowstore import COPY_BLOCK_BYTES, Chunk, gather_to_shm

    rng = np.random.default_rng(7)
    big = Chunk(concat([synthetic_row(rng, combat=False, t=i, body=(40, 60)) for i in range(400)]))
    big = Chunk(concat([big.rb] * 25))  # 10,000 rows
    refs = [(big, i) for i in range(big.n) if i != 17]  # the gap: copies by index
    data = sum(x.nbytes for x in big.rb.values())
    assert data > 16 * COPY_BLOCK_BYTES  # one temporary per field would be large
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        shm, _layout = gather_to_shm(refs, threads=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    shm.close()
    shm.unlink()
    assert peak - before < 4 * COPY_BLOCK_BYTES, (peak - before, data)


def test_a_shm_chunk_reads_its_block_and_closes_it_when_it_goes() -> None:
    from ppo.e2e.rowstore import ShmChunk, batch_to_shm, gather

    src = _chunk(6, 4)
    shm, layout = batch_to_shm(src.rb)
    ch = ShmChunk(shm, layout)
    refs = [4, 0, 5, 2]
    want, got = gather([(src, i) for i in refs]), gather([(ch, i) for i in refs])
    for f in C.ROW_FIELDS:
        nan = np.issubdtype(want[f.name].dtype, np.floating)
        assert np.array_equal(got[f.name], want[f.name], equal_nan=nan), f.name
    ch.unlink()
    ch.unlink()  # a second unlink is a no-op
    assert shm.buf is not None  # the name is gone; the mapping stays with the chunk
    del ch
    assert shm.buf is None


class ChunkProbe(TeacherLearner):
    """Records, whenever an update returns, how many held rows it left and the
    rows of the chunks they point into."""

    seen: list[tuple[int, int]]

    def _poll(self, block: bool) -> bool:
        moved = super()._poll(block)
        if moved:
            chunks = {id(r.chunk): r.chunk.n for r in self.held.values()}
            self.seen.append((len(self.held), sum(chunks.values())))
        return moved


@pytest.mark.parametrize("sync_updates", [True, False])
def test_held_rows_keep_only_their_own_rows_alive(tmp_path: Path, sync_updates: bool) -> None:
    lr = make_learner(make_cfg(tmp_path, max_iters=7, sync_updates=sync_updates), cls=ChunkProbe)
    assert isinstance(lr, ChunkProbe)
    lr.seen = []
    lr.train()
    assert len(lr.seen) == 7 and all(held > 0 for held, _ in lr.seen)
    # Before rank 0's copy the held rows point into the update's row block (the
    # whole table); without any copy, into the rounds of the last HOLD_ITERS
    # iterations, about three times their own rows.
    assert all(alive == held for held, alive in lr.seen), lr.seen


class CutProbe(TeacherLearner):
    """Records, after every cut, where the rows the learner keeps point."""

    seen: list[dict[str, Any]]

    def _pack_done(self, *, block: bool) -> bool:
        if not super()._pack_done(block=block):
            return False
        assert self.inflight is not None
        self._probe(self.inflight["block"], self.inflight["rows"])
        return True

    def _probe(self, block: Any, shipped: list[L.Row]) -> None:
        pending = [r for s in self.segs.values() for r in s.pending]
        pend = {id(r) for r in pending}
        self.seen.append(
            {
                "shipped": len(shipped),
                "in_block": sum(r.chunk is block for r in shipped),
                "pending_shipped": sum(id(r) in pend for r in shipped),
                "held": len(self.held),
                "held_in_block": sum(r.chunk is block for r in self.held.values()),
                "pending_in_block": sum(r.chunk is block for r in pending),
                "pending_chunks": len({id(r.chunk) for r in pending}),
            }
        )


def test_the_cut_points_the_shipped_rows_into_the_row_block(tmp_path: Path) -> None:
    """After a cut every shipped row reads the update's row block, so no serving
    round and no older held chunk stays alive through them; only an open
    segment's newest row (shipped as the bootstrap row, stored again next
    iteration) stays in its round, the last one."""
    lr = make_learner(make_cfg(tmp_path, max_iters=5), cls=CutProbe)
    assert isinstance(lr, CutProbe)
    lr.seen = []
    lr.train()
    assert len(lr.seen) == 5
    for s in lr.seen:
        assert s["in_block"] + s["pending_shipped"] == s["shipped"], s
        assert s["pending_shipped"] > 0 and s["pending_in_block"] == 0, s
        assert s["held_in_block"] == s["held"], s
        assert s["pending_chunks"] == 1, s
    assert all(s["held"] > 0 for s in lr.seen[1:])


#: Fields the learner writes itself: at acting time (`rowstore.ACTING`, `tprob`)
#: and from the joined table (`learner.table_to_rows`).
_LEARNER_FIELDS = {"tprob", "reward", "done", "bootstrap", "run_uid", "t", "cell"}


class SentProbe(TeacherLearner):
    """Keeps every stored row's inputs as its worker sent them."""

    sent: dict[tuple[int, int], dict[str, np.ndarray]]

    def _store(self, ch: Any, i: int, m: np.ndarray, r_prev: float, c: int) -> L.Row:
        from ppo.e2e.rowstore import take

        row = super()._store(ch, i, m, r_prev, c)
        self.sent[row.uid, row.t] = take(ch, [i])
        return row


def test_every_shipped_row_carries_the_inputs_its_worker_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held rows ship in several tables: from their round, then from the last
    update's row block, then from their own chunk. Every shipment of every row
    carries the inputs its worker sent."""
    from ppo.e2e import rowstore

    shipped = spy_shipped(monkeypatch)
    lr = make_learner(make_cfg(tmp_path, max_iters=7), cls=SentProbe)
    assert isinstance(lr, SentProbe)
    lr.sent = {}
    lr.train()
    skip = set(rowstore.ACTING) | _LEARNER_FIELDS
    times: Counter = Counter()
    for rb in shipped:
        ch = rowstore.Chunk(rb)
        for k in range(ch.n):
            key = (int(rb["run_uid"][k]), int(rb["t"][k]))
            want, got = lr.sent[key], rowstore.take(ch, [k])
            for f in C.ROW_FIELDS:
                if f.name not in skip:
                    assert np.array_equal(got[f.name], want[f.name]), (key, f.name)
            times[key] += 1
    assert len(shipped) == 7 and max(times.values()) >= 3


@pytest.mark.parametrize("sync_updates", [True, False])
def test_row_blocks_are_unlinked_after_the_update_and_closed_with_their_last_row(
    tmp_path: Path, sync_updates: bool
) -> None:
    """The ranks' row block loses its name when the update returns, and its
    mapping then too: the rows rank 0's hold-back keeps move to the block rank 0
    copied them into (mapped by the learner, its name gone at once), and the
    other shipped rows go. So after every update the held rows read one block,
    and no row block is left."""
    from multiprocessing import shared_memory

    row_blocks: list[Any] = []
    held_blocks: list[Any] = []
    after: list[tuple[int, int, int]] = []

    class Probe(TeacherLearner):
        def _pack_done(self, *, block: bool) -> bool:
            if not super()._pack_done(block=block):
                return False
            assert self.inflight is not None
            row_blocks.append(self.inflight["block"].shm)
            return True

        def _poll(self, block: bool) -> bool:
            moved = super()._poll(block)
            if moved:
                chunks = {id(r.chunk): r.chunk for r in self.held.values()}
                assert all(isinstance(c, L.ShmChunk) for c in chunks.values())
                held_blocks.extend(c.shm for c in chunks.values())
                mapped_rows = sum(s.buf is not None for s in row_blocks)
                mapped_held = sum(s.buf is not None for s in held_blocks)
                after.append((mapped_rows, len(chunks), mapped_held))
            return moved

    lr = make_learner(make_cfg(tmp_path, max_iters=5, sync_updates=sync_updates), cls=Probe)
    lr.train()
    assert len(row_blocks) == 5 and after == [(0, 1, 1)] * 5
    assert len({s.name for s in held_blocks}) == 5
    for s in row_blocks + held_blocks:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=s.name, track=False)
    for r in lr.stats_rows:
        assert r["update"]["held_pack_s"] > 0 and r["assembly"]["held_land_s"] > 0


def test_a_held_block_no_one_maps_goes_at_the_stop(tmp_path: Path) -> None:
    """An update still in flight at a stop (an error, a guard) answered with its
    held rows' block: the updater's stop removes it."""
    from multiprocessing import shared_memory

    names: list[str] = []

    class Probe(TeacherLearner):
        def _pack_done(self, *, block: bool) -> bool:
            if not super()._pack_done(block=block):
                return False
            res = self.updater.res  # the local updater ran the update at submit
            names.append(res["held"]["shm"])
            raise RuntimeError("stop with an update in flight")

    lr = make_learner(make_cfg(tmp_path, max_iters=5), cls=Probe)
    with pytest.raises(RuntimeError, match="update in flight"):
        lr.train()
    assert len(names) == 1
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0], track=False)


# ── S1-cut: the row block is copied while serving goes on ────────────────


def _cut_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_pack: bool) -> dict[str, Any]:
    """Six iterations, state.pt, a resume for five more (its held rows dropped):
    the row blocks and jobs shipped to the update, and the stats rows."""
    shipped = spy_shipped(monkeypatch)
    jobs: list[dict[str, Any]] = []
    real = L.LocalUpdater.submit

    def submit(self: Any, job: dict[str, Any]) -> None:
        keep = ("it", "n_collected", "t_wall", "refs", "drop_uids", "pack_held")
        jobs.append({k: job[k] for k in keep} | {"fresh": dict(job["fresh"])})
        real(self, job)

    monkeypatch.setattr(L.LocalUpdater, "submit", submit)
    src = {"workers": 3, "slots": 5, "length": (4, 40), "error_p": 0.08}
    sd = tmp_path / ("async" if async_pack else "inline")
    kw = {"async_pack": async_pack, "pack_threads": 3 if async_pack else 1}
    a = make_learner(make_cfg(sd, max_iters=6, **kw), **src)
    a.train()
    (sd / OC.DONE).unlink()  # the budget's stop marker
    b = make_learner(make_cfg(sd, max_iters=11, resume=str(sd / "state.pt"), **kw), **src)
    lost = b.held_lost
    b.train()
    return {
        "shipped": shipped,
        "jobs": jobs,
        "stats": [_stable(r) for r in a.stats_rows + b.stats_rows],
        "held_lost": lost,
        "net": b.net_inf.state_dict(),
    }


def test_async_and_inline_cuts_ship_the_same_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `sync_updates` and fixed seeds, the threaded copy (`async_pack`, the
    default, here on 3 threads) and the inline one ship the same bytes in every row block, the same
    jobs (tables, refs, drops) and the same stats, across segment ends, errors,
    practice segments, held rows and a resume."""
    with monkeypatch.context() as mp:
        x = _cut_run(tmp_path, mp, True)
    with monkeypatch.context() as mp:
        y = _cut_run(tmp_path, mp, False)
    assert len(x["shipped"]) == len(y["shipped"]) == 11 == len(x["jobs"]) == len(y["jobs"])
    for bx, by in zip(x["shipped"], y["shipped"], strict=True):
        for f in C.ROW_FIELDS:
            assert np.array_equal(bx[f.name].view(np.uint8), by[f.name].view(np.uint8)), f.name
    for jx, jy in zip(x["jobs"], y["jobs"], strict=True):
        assert set(jx) == set(jy)
        for k in jx:
            if k == "fresh":
                assert all(np.array_equal(jx[k][c], jy[k][c]) for c in jx[k])
            else:
                assert np.array_equal(jx[k], jy[k]), k
    assert x["stats"] == y["stats"] and x["held_lost"] == y["held_lost"] > 0
    assert all(torch.equal(v, y["net"][n]) for n, v in x["net"].items())
    st = x["stats"]
    assert sum(r["errors"] for r in st) > 0 and sum(r["practice_segments"] for r in st) > 0
    assert sum(r["assembly"]["held_rows"] for r in st) > 0


class PackProbe(TeacherLearner):
    """Holds each cut's copy until `hold` rounds were served after it, and keeps
    every stored row's iteration and whether a copy was running."""

    hold = 3
    stored: dict[tuple[int, int], tuple[int, bool]]
    during: list[int]

    def _store(self, ch: Any, i: int, m: np.ndarray, r_prev: float, c: int) -> L.Row:
        row = super()._store(ch, i, m, r_prev, c)
        assert not (self.packing is not None and self.inflight is not None)
        self.stored[row.uid, row.t] = (row.it, self.packing is not None)
        return row


def test_serving_goes_on_while_the_cut_is_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `sync_updates` the learner serves rounds while the cut's block is
    copied, submits the update when the copy is done (never two in flight), and
    the rows it stored meanwhile belong to the next iteration."""
    import threading

    gate = threading.Event()
    real = L.pack_rows
    box: dict[str, L.Learner] = {}

    def slow(src: Any, table: Any, threads: int = 1) -> Any:
        if box["lr"].n_dispatched < 6:  # the last cut has no collection after it
            assert gate.wait(30)
        gate.clear()
        return real(src, table, threads)

    monkeypatch.setattr(L, "pack_rows", slow)
    shipped = spy_shipped(monkeypatch)  # wraps `slow`

    def on_round(lr: L.Learner, _src: FakeSource) -> None:
        assert isinstance(lr, PackProbe)
        if lr.packing is not None:
            lr.during[-1] += 1
            if lr.during[-1] >= lr.hold:
                gate.set()
        elif not lr.during or lr.during[-1]:
            lr.during.append(0)

    lr = make_learner(
        make_cfg(tmp_path, max_iters=6, sync_updates=False), cls=PackProbe, on_round=on_round
    )
    assert isinstance(lr, PackProbe)
    lr.stored, lr.during = {}, []
    box["lr"] = lr
    # The first table's copy runs inline (the fill check's start rule); this test's
    # learner has the check off, so every copy is threaded.
    lr.train()
    assert len(lr.stats_rows) == 6 and len(shipped) == 6
    assert sum(d >= lr.hold for d in lr.during) >= 5, lr.during
    during = {k for k, (_it, p) in lr.stored.items() if p}
    assert during
    for k, rb in enumerate(shipped):
        for j in range(len(rb["n_tok"])):
            key = (int(rb["run_uid"][j]), int(rb["t"][j]))
            it, _p = lr.stored[key]
            assert int(rb["iter"][j]) == it  # the iteration it was stored in
            if key in during:
                assert it >= 1  # stored after a cut: the next iteration's
    for r in lr.stats_rows:
        assert r["assembly"]["pack_s"] > 0 and r["assembly"]["cut_s"] >= 0


def test_a_cut_still_copying_at_a_stop_leaves_no_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop (here an error) while a cut's block is being copied removes the block."""
    from multiprocessing import shared_memory

    names: list[str] = []
    real = L.pack_rows

    def spy(src: Any, table: Any, threads: int = 1) -> Any:
        shm, layout, secs = real(src, table, threads)
        names.append(shm.name)
        return shm, layout, secs

    monkeypatch.setattr(L, "pack_rows", spy)

    def on_round(lr: L.Learner, _src: FakeSource) -> None:
        if lr.packing is not None:
            lr.packing["fut"].result()  # the copy is done, the update not yet submitted
            raise RuntimeError("stop with a cut being copied")

    lr = make_learner(make_cfg(tmp_path, max_iters=5, sync_updates=False), on_round=on_round)
    with pytest.raises(RuntimeError, match="cut being copied"):
        lr.train()
    assert len(names) == 1 and lr.packing is None and lr.inflight is None
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=names[0], track=False)


# ── T11, the learner half: a kill reproduces the next 3 iterations ───────

#: Stats fields that hold wall time, process ids or sizes of this session.
_VOLATILE = {
    "time",
    "pid",
    "dec_per_s",
    "collect_s",
    "fwd_s",
    "wait_s",
    "policy_swaps",
    "stamps",
    "workers",
}
_VOLATILE_UPDATE = {"prepass_s", "update_s", "rank_s", "held_pack_s"}
_VOLATILE_ASSEMBLY = {"cut_s", "held_land_s", "pack_s", "pack_wait_s"}


def _stable(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in _VOLATILE}
    out["update"] = {k: v for k, v in row["update"].items() if k not in _VOLATILE_UPDATE}
    out["assembly"] = {k: v for k, v in row["assembly"].items() if k not in _VOLATILE_ASSEMBLY}
    return out


class Uninterrupted(TeacherLearner):
    """Run A: at each kill point it writes its state, then drops its open segments and
    starts new workers (a resume loses both, and with them the held rows' runs), and it
    snapshots its net 3 iterations on."""

    kills: tuple[int, ...] = ()
    snap_dir: Path = Path()

    def collect(self) -> dict[str, Any] | None:
        k = self.n_dispatched
        if k in self.kills and not (self.snap_dir / f"state_{k}.pt").exists():
            self.save(state=True)
            shutil.copy(self.sd.state, self.snap_dir / f"state_{k}.pt")
            self.segs.clear()
            self.lost_uids = L.lost_runs(self.algo)
            wcfg = self.worker_cfg()
            self.source.stop()
            self.source = FakeSource(wcfg)
        if k - 3 in self.kills:
            net = {n: v.detach().clone() for n, v in self.net_inf.state_dict().items()}
            torch.save(
                {"net": net, "algo": self.algo.to_dict()}, self.snap_dir / f"after_{k - 3}.pt"
            )
        return super().collect()


def _t11(tmp_path: Path, kills: tuple[int, ...]) -> None:
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    last = max(kills) + 4  # A collects once more after k + 3: the net snapshot
    Uninterrupted.kills, Uninterrupted.snap_dir = kills, snaps
    a = make_learner(make_cfg(tmp_path / "A", max_iters=last), cls=Uninterrupted)
    a.train()
    rows_a = {r["it"]: r for r in a.stats_rows}
    phases = {r["update"]["phase"] for r in a.stats_rows}
    for k in kills:
        b = make_learner(
            make_cfg(tmp_path / f"B{k}", max_iters=k + 3, resume=str(snaps / f"state_{k}.pt"))
        )
        # The resume dropped every held row (`HoldBack.resume`); A drops the same
        # rows at its next update (`lost_uids`), on the learner and the ranks.
        n_held = b.held_lost
        assert n_held > 0 and not b.held and not len(b.algo.hold) and b.n_dispatched == k
        b.train()
        assert [r["it"] for r in b.stats_rows] == [k + 1, k + 2, k + 3]
        assert b.stats_rows[0]["assembly"]["held_lost"] == n_held
        assert [r["assembly"]["held_lost"] for r in b.stats_rows[1:]] == [0, 0]
        # A's ranks drop those rows in its next update; B's start without them.
        ua, ub = rows_a[k + 1]["update"], b.stats_rows[0]["update"]
        assert ua.pop("held_dropped") == n_held + ub.pop("held_dropped")
        for r in b.stats_rows:
            assert _stable(r) == _stable(rows_a[r["it"]]), (k, r["it"])
        after = torch.load(snaps / f"after_{k}.pt", weights_only=True)
        got = b.net_inf.state_dict()
        assert all(torch.equal(got[n], v) for n, v in after["net"].items()), k
        assert b.algo.to_dict()["clocks"] == after["algo"]["clocks"]
    assert phases  # the phases the kills sit in are asserted by the callers


def test_t11_kill_at_7_reproduces_the_next_three_iterations(tmp_path: Path) -> None:
    _t11(tmp_path, (7,))


@pytest.mark.slow
def test_t11_kill_at_random_and_150_reproduce_the_next_three_iterations(tmp_path: Path) -> None:
    r = int(np.random.default_rng(1097).integers(8, 147))
    _t11(tmp_path, (r, 150))


# ── stops, refusals, guards ──────────────────────────────────────────────


def test_done_mid_collection_saves_the_state_and_a_resume_continues(tmp_path: Path) -> None:
    seen = {"rounds": 0}

    def done_at(lr: L.Learner, src: FakeSource) -> None:
        if lr.n_dispatched == 2:
            seen["rounds"] += 1
            if seen["rounds"] == 3:
                _write_now(lr, tmp_path / OC.DONE)

    lr = make_learner(make_cfg(tmp_path, max_iters=10), on_round=done_at)
    assert lr.train() == "DONE"
    assert [r["it"] for r in lr.stats_rows] == [1, 2]
    assert lr.decisions > lr.algo.clocks.decisions  # the cut-off collection is lost
    assert ckpt.state_decisions(tmp_path) == lr.stats_rows[-1]["decisions"]
    assert (tmp_path / OC.DONE).read_text(encoding="utf-8") == "operator stop\n"
    assert not ckpt.pinned_states(tmp_path)  # a stop is not a final state
    # A second learner on the same save dir needs --resume.
    with pytest.raises(L.UsageError, match="--resume"):
        make_learner(make_cfg(tmp_path, max_iters=4))
    (tmp_path / OC.DONE).unlink()
    lr2 = make_learner(make_cfg(tmp_path, max_iters=4, resume=str(tmp_path / "state.pt")))
    assert lr2.train() == "max_iters"
    assert [r["it"] for r in lr2.stats_rows] == [3, 4]
    assert lr2.stats_rows[0]["runs_done"] > lr.stats_rows[-1]["runs_done"]


def _its(path: Path) -> list[int]:
    return [json.loads(x)["it"] for x in path.read_text(encoding="utf-8").splitlines()]


def test_park_orphan_stats_moves_the_rows_past_the_state(tmp_path: Path) -> None:
    """G0 review S1: a crash replays every iteration since the last state.pt (up to
    ~38 at launch), so a resume moves their rows, and a torn last line, aside."""
    stats, orphan = tmp_path / OC.STATS, tmp_path / OC.STATS_ORPHAN
    assert L.park_orphan_stats(stats, orphan, 3) == 0  # no file yet
    rows = [json.dumps({"it": i, "time": float(i)}) for i in range(1, 7)]
    stats.write_text("\n".join(rows) + "\n" + '{"it": 7, "ti', encoding="utf-8")
    assert L.park_orphan_stats(stats, orphan, 3) == 4
    assert _its(stats) == [1, 2, 3]
    lines = orphan.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == rows[3:] and lines[3] == '{"it": 7, "ti'
    assert L.park_orphan_stats(stats, orphan, 3) == 0  # idempotent
    assert stats.read_text(encoding="utf-8") == "".join(r + "\n" for r in rows[:3])


def test_a_resume_from_an_older_state_parks_the_replayed_rows(tmp_path: Path) -> None:
    """A resume from a state k iterations back (a kill between two state.pt, or a
    rollback) leaves `it` unique and increasing in stats.jsonl; the discarded
    trajectory's rows go to stats.orphan.jsonl, and its NaN still fails the
    learning check."""
    lr = make_learner(make_cfg(tmp_path, max_iters=3))
    assert lr.train() == "max_iters"
    old = tmp_path / "state_3.pt"
    shutil.copy(lr.sd.state, old)
    (tmp_path / OC.DONE).unlink()  # a max_iters stop writes DONE
    lr2 = make_learner(make_cfg(tmp_path, max_iters=7, resume=str(lr.sd.state)))
    assert lr2.train() == "max_iters"
    stats, orphan = tmp_path / OC.STATS, tmp_path / OC.STATS_ORPHAN
    assert _its(stats) == list(range(1, 8)) and not orphan.exists()
    # A NaN on the discarded trajectory (iteration 7's row, rewritten as one).
    lines = stats.read_text(encoding="utf-8").splitlines()
    lines[-1] = json.dumps({**json.loads(lines[-1]), "nonfinite": True})
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / OC.DONE).unlink()
    lr3 = make_learner(make_cfg(tmp_path, max_iters=5, resume=str(old)))
    assert _its(stats) == [1, 2, 3] and _its(orphan) == [4, 5, 6, 7]
    assert lr3.train() == "max_iters"
    assert _its(stats) == [1, 2, 3, 4, 5]
    assert lr3.learning_check()["nan"]


def test_stats_rows_split_the_fresh_rows_staleness_from_the_held_rows(tmp_path: Path) -> None:
    """Review finding LRN-3: `staleness.max` covers every trained row, so it reaches the
    held-back run rows' age (up to 5) by design; `staleness.fresh_max` covers this
    iteration's rows, at most 1 with one update in flight (what B-T0 reads)."""
    lr = make_learner(
        make_cfg(tmp_path, max_iters=9, sync_updates=False), length=(60, 120), practice_p=0.0
    )
    assert lr.train() == "max_iters"
    st = [r["staleness"] for r in lr.stats_rows]
    assert {"mean", "max", "fresh_mean", "fresh_max"} <= set(st[-1])
    assert max(s["max"] for s in st) >= 2  # released held rows
    assert all(s["fresh_max"] <= 1 for s in st)
    assert any(s["max"] > s["fresh_max"] + 1 for s in st)


def test_a_resume_takes_the_new_lr_and_weight_decay(tmp_path: Path) -> None:
    """Review finding LRN-4: the optimizer state a resume loads keeps its moments, but
    the groups' base lr and weight decay follow the resumed cfg. They used to be the
    saved ones, so a changed lr stopped the lane on LR_CLOSED_FORM after training at
    the old lr, and a changed weight decay was silently ignored."""
    lr = make_learner(make_cfg(tmp_path, max_iters=2))
    assert lr.train() == "max_iters"
    (tmp_path / OC.DONE).unlink()
    sched = dict(SCHED, lr_trunk=S.DEFAULT.lr_trunk / 2, lr_value=S.DEFAULT.lr_value / 3)
    cfg = make_cfg(
        tmp_path, max_iters=4, resume=str(tmp_path / "state.pt"), sched=sched, weight_decay=0.0
    )
    lr2 = make_learner(cfg)
    assert lr2.train() == "max_iters"
    groups = {g["name"]: g for g in lr2.updater.rank.opt.param_groups}
    assert groups["trunk"]["base_lr"] == sched["lr_trunk"]
    assert groups["value"]["base_lr"] == sched["lr_value"]
    assert {g["weight_decay"] for g in groups.values()} == {0.0}
    assert [r["it"] for r in lr2.stats_rows] == [3, 4]


def test_refusals_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = ["train", OC.FLAG_SAVE_DIR, str(tmp_path), OC.FLAG_SEED_OFFSET, "0"]
    base += ["--infer-device", "cpu", "--update-devices", "cpu", "--net", json.dumps(TINY)]
    # A campaign must carry a budget of at least 1e10 decisions (plan §6).
    sched = json.dumps({"budget_decisions": 10**9})
    assert L.main([*base, "--sched", sched]) == OC.EXIT_USAGE
    # The game build must be v0.111.0.
    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.107.1")
    assert L.main([*base, "--max-iters", "1"]) == OC.EXIT_USAGE
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)
    # state.pt without --resume.
    (tmp_path / "state.pt").write_bytes(b"x")
    assert L.main([*base, "--max-iters", "1"]) == OC.EXIT_USAGE
    (tmp_path / "state.pt").unlink()
    # Another learner holds the lock.
    held = make_learner(make_cfg(tmp_path))
    try:
        assert L.main([*base, "--max-iters", "1"]) == OC.EXIT_USAGE
    finally:
        held.stop()
    # A resume from something that is not a learner state.
    bad = tmp_path / "bad.pt"
    torch.save({"format": "nope"}, bad)
    assert L.main([*base, "--max-iters", "1", OC.FLAG_RESUME, str(bad)]) == OC.EXIT_USAGE
    # An F9 tolerance that is not a known engine issue (FIDELITY.md).
    assert L.main([*base, "--max-iters", "1", "--f9-tolerate", '["nope"]']) == OC.EXIT_USAGE
    # A fill event the schema does not know (contracts.FILL_EVENTS).
    assert L.main([*base, "--max-iters", "1", "--filled", '["nope"]']) == OC.EXIT_USAGE


def test_a_lane_draws_its_run_seeds_from_its_prereg_block(tmp_path: Path) -> None:
    """PREREG P1 (review findings LRN-1, ENVR-6): a campaign lane (mode train, no
    max_iters) plays trainer runs from `panels.STUDENT_TRAINER_SEEDS`, every other lane
    from `THROWAWAY_TRAINER_SEEDS`; a base whose seeds leave the lane's block is refused."""
    from ppo.e2e import panels as P

    student, throwaway = P.STUDENT_TRAINER_SEEDS, P.THROWAWAY_TRAINER_SEEDS
    # The production iteration size: the span covers 7 seeds per iteration up to 2x the
    # budget's iterations (a campaign of 40 workers fits, plan's B-T0).
    campaign = {
        "max_iters": 0,
        "iter_decisions": 131_072,
        "sched": dict(SCHED, budget_decisions=10**10),
    }
    for sub, kw, want in (
        ("campaign", campaign, student.start),
        ("smoke", {}, throwaway.start),
        ("explicit", dict(campaign, run_seed_base=student.start + 5), student.start + 5),
    ):
        lr = make_learner(make_cfg(tmp_path / sub, **kw))
        try:
            assert lr.cfg.run_seed_base == want and lr.worker_cfg()["run_seed_base"] == want
        finally:
            lr.stop()
    for sub, kw in (
        ("throwaway", dict(campaign, run_seed_base=throwaway.start)),  # the default before
        ("student", {"run_seed_base": student.start}),
        ("span", dict(campaign, run_seed_base=student.stop - 1_000_000)),
        ("slots", {"n_envs": 65}),
        ("workers", dict(campaign, workers=41)),
    ):
        with pytest.raises(L.UsageError, match=r"run seeds|n-envs"):
            make_learner(make_cfg(tmp_path / sub, **kw))


def _stamped(stamp: dict[str, str]) -> type[L.Learner]:
    class Stamped(TeacherLearner):
        def _load_teachers(self) -> Any:
            t = super()._load_teachers()
            t.combat.teacher_stamp = dict(stamp)
            return t

    return Stamped


def test_stamps_carry_the_teacher_and_a_resume_keeps_them(tmp_path: Path) -> None:
    """SCHEMA.md 10: the stamps name the combat teacher whose RareMap the rows use;
    a resume refuses another teacher and a dropped fill event, and may add one."""
    t1 = {"ckpt_sha256": "c1", "rare_sha256": "r1", "sctx_vocab_sha256": "v1"}
    t2 = dict(t1, ckpt_sha256="c2")
    lr = make_learner(make_cfg(tmp_path, max_iters=2), cls=_stamped(t1))
    assert lr.train() == "max_iters"
    assert lr.stamps["teacher"] == t1 and lr.stats_rows[-1]["stamps"]["teacher"] == t1
    rec = json.loads((tmp_path / "cfg.json").read_text(encoding="utf-8"))
    assert rec["stamps"]["teacher"] == t1
    assert rec["teachers"] == {"combat": None, "run": None}  # the fakes have no stamps()
    st = torch.load(tmp_path / "state.pt", weights_only=True)
    assert st["stamps"]["teacher"] == t1 and st["net"]["cfg"]["stamps"]["teacher"] == t1
    (tmp_path / OC.DONE).unlink()
    state = str(tmp_path / "state.pt")
    with pytest.raises(L.UsageError, match="teacher"):
        make_learner(make_cfg(tmp_path, max_iters=3, resume=state), cls=_stamped(t2))
    filled = make_cfg(tmp_path, max_iters=3, resume=state, filled=["WP-ENG-1"])
    lr2 = make_learner(filled, cls=_stamped(t1))  # a restart may add a fill event
    assert lr2.train() == "max_iters" and lr2.stamps["filled"] == ["WP-ENG-1"]
    (tmp_path / OC.DONE).unlink()
    with pytest.raises(L.UsageError, match="filled"):
        make_learner(make_cfg(tmp_path, max_iters=4, resume=state), cls=_stamped(t1))


def test_a_fresh_learner_inits_its_fill_events_live(tmp_path: Path) -> None:
    """GH #1097 A4: `--filled '["WP-ENG-1"]'` on a fresh launch builds the net with
    `E2ENet(filled=...)`, so the honest X_ID_AUX gain and the encounter and move ID
    rows start live; without it they start at zero. The checkpoint the ranks load
    carries the same weights."""
    from ppo.e2e import net as N

    aux = N.layout().id_cols.index(C.XC["X_ID_AUX"])
    rows = list(N.zero_id_rows())
    assert rows and N.zero_id_rows(["WP-ENG-1"]) == ()
    for sub, filled in (("base", []), ("wp", ["WP-ENG-1"])):
        lr = make_learner(make_cfg(tmp_path / sub, filled=filled), teachers=False)
        try:
            live = bool(filled)
            assert lr.stamps["filled"] == filled
            ck = torch.load(lr.init_path, map_location="cpu", weights_only=True)["net"]
            for net in (lr.net_inf, N.net_from_state(ck, schema_sha256=L.SCHEMA_SHA256)):
                assert bool(net.id_gain[aux].eq(1.0).all()) is live
                assert bool(net.id_gain[aux].any()) is live
                assert bool(net.ids.weight[rows].abs().sum(1).gt(0).all()) is live
                assert bool(net.ids.weight[rows].any()) is live
        finally:
            lr.stop()


def test_the_content_prior_fill_reads_the_committed_prior(tmp_path: Path) -> None:
    """Review findings NT-1 and NT-2: `--filled` with "content-prior" fills the net's
    `prior` buffer from `content_prior.table()` and stamps its sha256, on a fresh launch
    and on a restart that adds the event. The EMA shadows copy the buffer instead of
    averaging it, so an EMA export after that restart serves the whole prior under its
    stamp."""
    from ppo.e2e import content_prior as CP
    from ppo.e2e import net as N

    sha, table = CP.sha256(), torch.from_numpy(CP.table().copy())
    lr = make_learner(make_cfg(tmp_path / "fresh", filled=["content-prior"]), teachers=False)
    try:
        assert lr.stamps["prior_sha256"] == sha and lr.stamps["filled"] == ["content-prior"]
        ck = torch.load(lr.init_path, map_location="cpu", weights_only=True)["net"]
        for net in (lr.net_inf, N.net_from_state(ck, schema_sha256=L.SCHEMA_SHA256)):
            assert net.prior_sha256 == sha and torch.equal(net.prior.cpu(), table)
    finally:
        lr.stop()
    d = tmp_path / "restart"
    lr = make_learner(make_cfg(d, max_iters=2))
    assert lr.train() == "max_iters" and lr.stamps["prior_sha256"] is None
    (d / OC.DONE).unlink()
    state = str(d / "state.pt")
    lr2 = make_learner(make_cfg(d, max_iters=4, resume=state, filled=["content-prior"]))
    assert lr2.train() == "max_iters"
    st = torch.load(state, weights_only=True)
    assert st["net"]["cfg"]["prior_sha256"] == sha
    assert torch.equal(st["net"]["state_dict"]["prior"], table)
    for v in ("raw", "ema0996", "ema0999"):
        out = d / f"{v}.pt"
        export_variant(state, v, str(out))
        net, cfg = load_checkpoint(out)
        assert net.prior_sha256 == sha and torch.equal(net.prior, table), v
        assert cfg["stamps"]["filled"] == ["content-prior"]
        assert cfg["stamps"]["prior_sha256"] == sha


def test_an_init_checkpoint_follows_the_restart_rule(tmp_path: Path) -> None:
    """Review finding LRN-2: `--init` restamps its checkpoint with the lane's stamps,
    so it first passes `check_stamps(restart=True)`: a lane that drops a fill event the
    net read is refused, one that adds an event (here the content prior) is not."""
    from ppo.e2e import content_prior as CP

    src = make_learner(make_cfg(tmp_path / "src", max_iters=1, filled=["WP-ENG-1"]))
    assert src.train() == "max_iters"
    init = tmp_path / "init.pt"
    export_variant(str(tmp_path / "src" / "state.pt"), "raw", str(init))
    with pytest.raises(L.UsageError, match=r"--init .*filled"):
        make_learner(make_cfg(tmp_path / "drop", init=str(init)))
    lr = make_learner(
        make_cfg(tmp_path / "add", init=str(init), filled=["WP-ENG-1", "content-prior"])
    )
    try:
        assert lr.net_inf.prior_sha256 == CP.sha256()
    finally:
        lr.stop()


def test_the_learners_stats_rows_feed_the_preflight(tmp_path: Path) -> None:
    """GH #1097 A7: the G0 pre-flight reads a learner run's `stats.jsonl` (the learner
    writes no action logs): every field `preflight.counts_from_stats` needs is in the
    rows, and its counters add up to the rows' own totals."""
    from ppo.e2e import preflight as PF

    lr = make_learner(make_cfg(tmp_path, max_iters=2))
    assert lr.train() == "max_iters"
    rows = PF.read_stats(tmp_path)
    assert len(rows) == len(lr.stats_rows) == 2
    c = PF.counts_from_stats(rows)
    assert sum(v for k, v in c.items() if k.startswith("rows/")) == sum(r["rows"] for r in rows)
    assert c["segments"] == sum(sum(r["outcomes"].values()) for r in rows) > 0
    assert c["segments/run"] == sum(r["runs"] for r in rows) > 0
    assert c["f9/checks"] == sum(r["f9"]["checked"] for r in rows)
    # The ended segments' summed counters (the end records' `counts`) ride along, so
    # the pre-flight checks the fine keys, asserts and forced share too (A7's open
    # issue): nothing is left unchecked.
    seg = Counter()
    for r in rows:
        seg.update(r["segment_counts"])
    assert seg["f9/checks"] == c["f9/checks"] > 0
    assert seg["fights/monster"] == sum(r["caps"]["fights"] for r in rows) > 0
    assert seg["error/StepError"] + seg["error/StartError"] == c["outcome/error"] > 0
    pf = PF.preflight_stats(rows)
    assert pf.segments == c["segments"] and pf.unchecked == []
    assert set(pf.missing) >= {f"fine/{k}" for k in PF.REQUIRED_FINE}  # FakeSource has none
    assert rows[-1]["stamps"]["filled"] == []


def test_run_teacher_features_read_the_honest_obs_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCHEMA.md open question 15's fallback reaches the workers: with
    `run_teacher_honest` the run teacher's features read `contracts.honest_obs`."""
    import ppo.encoder

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(ppo.encoder, "featurize_row", lambda obs, *a, **k: seen.append(obs))
    d = SimpleNamespace(run_obs={"seed": 5, "hp": 70}, actions=[], metrics={})
    actors.run_teacher_features(d)
    actors.run_teacher_features(d, honest=True)
    assert seen == [{"seed": 5, "hp": 70}, C.honest_obs(d.run_obs)] and "seed" not in seen[1]
    for sub, honest in (("raw", False), ("honest", True)):
        lr = make_learner(make_cfg(tmp_path / sub, run_teacher_honest=honest))
        assert lr.worker_cfg()["run_teacher_honest"] is honest
        lr.stop()


def _learning_check(save_dir: Path, floor_fn: Any) -> dict[str, Any]:
    cfg = make_cfg(
        save_dir,
        mode="learning_check",
        teacher_pair="R2",  # the mode drops it: no teacher, λ_KS = 0 (plan §7)
        lc_window=6,
        lc_ev=-math.inf,  # synthetic rows: the floor rule alone decides here
        max_iters=8,
    )
    lr = make_learner(
        cfg, teachers=False, floor_fn=floor_fn, practice_p=0.0, error_p=0.0, start_error_p=0.0
    )
    assert lr.train() == "max_iters"
    assert lr.teachers is None and cfg.teacher_pair == "" and cfg.init == ""
    assert not any(r["agreement"] for r in lr.stats_rows)
    return OC.read_json(save_dir / "learning_check.json")


def test_learning_check_writes_its_verdict_and_a_campaign_applies_it(tmp_path: Path) -> None:
    """Plan §7: pass iff the last window's mean floor beats the first window's by
    lc_margin (combat EV and NaN aside). The verdict file then starts the λ_KS ramp
    (pass) or holds λ_KS at 1 (fail) in the campaign that reads it."""
    good = _learning_check(tmp_path / "good", lambda n: 1 if n < 6 else 4)
    assert good["complete"] and good["pass"]
    assert (good["first_mean"], good["last_mean"], good["gain"]) == (1.0, 4.0, 3.0)
    flat = _learning_check(tmp_path / "flat", lambda n: 2)
    assert flat["complete"] and not flat["pass"] and abs(flat["gain"]) < 1e-12
    for name, verdict, hold in (("pass", good, False), ("fail", flat, True)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(verdict), encoding="utf-8")
        lr = make_learner(make_cfg(tmp_path / f"camp_{name}", lc_verdict=str(path), max_iters=1))
        try:
            assert lr.lc_applied is (not hold) and lr.algo.clocks.ks_hold is hold
            lr.train()
            assert lr.stats_rows[-1]["lc_verdict_pass"] is (not hold)
        finally:
            lr.stop()


def test_a_failed_recheck_after_a_pass_keeps_the_ramp_and_alarms(tmp_path: Path) -> None:
    good, bad = tmp_path / "pass.json", tmp_path / "fail.json"
    good.write_text(json.dumps({"pass": True}), encoding="utf-8")
    bad.write_text(json.dumps({"pass": False}), encoding="utf-8")
    a = make_learner(make_cfg(tmp_path / "A", lc_verdict=str(good), max_iters=2))
    a.train()
    assert a.algo.clocks.ks_passed and not a.algo.clocks.ks_hold
    clocks = a.algo.to_dict()["clocks"]
    b = make_learner(make_cfg(tmp_path / "B", lc_verdict=str(bad), resume=str(a.sd.state)))
    try:
        assert b.lc_applied is False and b.algo.to_dict()["clocks"] == clocks
        rec = OC.read_json(OC.alarm_path(tmp_path / "B", L.LC_RECHECK_FAIL))
        assert rec["severity"] == OC.OPERATOR and rec["writer"] == "learner"
        # The alarm table's row (B2) says who acts; the learner no longer overrides it.
        spec = OC.ALARMS_BY_NAME[L.LC_RECHECK_FAIL]
        assert rec["who_acts"] == spec.who_acts and rec["severity"] == spec.severity
    finally:
        b.stop()
    c = make_learner(make_cfg(tmp_path / "C", lc_verdict=str(bad), max_iters=1))
    try:  # no pass yet: a failed check holds λ_KS and raises nothing
        assert c.algo.clocks.ks_hold
        assert not OC.alarm_path(tmp_path / "C", L.LC_RECHECK_FAIL).exists()
    finally:
        c.stop()


def test_pin_request_pins_after_the_next_update(tmp_path: Path) -> None:
    def ask(lr: L.Learner, src: FakeSource) -> None:
        if lr.n_dispatched == 1 and not (tmp_path / "asked").exists():
            (tmp_path / "asked").write_text("", encoding="utf-8")
            (tmp_path / OC.PIN_REQUEST).write_text("", encoding="utf-8")

    lr = make_learner(make_cfg(tmp_path, max_iters=3), on_round=ask)
    lr.train()
    pins = {r.reason: r.decisions for r in ckpt.pinned_states(tmp_path)}
    dec = [r["decisions"] for r in lr.stats_rows]
    assert pins == {"request": dec[1], "final": dec[2]}
    assert not (tmp_path / OC.PIN_REQUEST).exists()


def test_nonfinite_update_writes_nan_record_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lr = make_learner(make_cfg(tmp_path, max_iters=5, state_every=ITER))
    real = L.LocalUpdater.poll

    def poll(self: Any, block: bool) -> Any:
        res = real(self, block)
        if res is not None and res["algo"]["clocks"]["it"] == 3:
            res["nonfinite"] = ["values (it 2)"]
        return res

    monkeypatch.setattr(L.LocalUpdater, "poll", poll)
    with pytest.raises(L.NanStop):
        lr.train()
    rec = OC.read_json(tmp_path / OC.NAN_RECORD)
    assert rec["it"] == 2 and rec["what"] == ["values (it 2)"]
    assert [r["it"] for r in lr.stats_rows] == [1, 2]
    # The state is the one before the bad update.
    assert ckpt.state_decisions(tmp_path) == lr.stats_rows[-1]["decisions"]
    assert not (tmp_path / OC.DONE).exists()


@pytest.mark.parametrize("bad", ["wheel", "py_src"])
def test_a_worker_stamp_mismatch_stops_before_its_batch_trains(tmp_path: Path, bad: str) -> None:
    """SCHEMA.md 10: the learner rejects a batch whose stamps differ from its own
    (here every worker message's), with the ops contract's alarm, before any update."""
    stamps = {"wheel": wheel_identity(), "py_src": python_source_digest()}
    stamps[bad] = {"sha256": "other"} if bad == "wheel" else "other"
    lr = make_learner(make_cfg(tmp_path, max_iters=4), stamps=stamps)
    with pytest.raises(L.GuardStop) as e:
        lr.train()
    name = "WHEEL_MISMATCH" if bad == "wheel" else "SOURCE_MISMATCH"
    assert e.value.name == name and OC.alarm_path(tmp_path, name).exists()
    assert lr.stats_rows == [] and not (tmp_path / "state.pt").exists()


def test_an_f9_hit_stops_with_its_alarm(tmp_path: Path) -> None:
    lr = make_learner(make_cfg(tmp_path, max_iters=4), f9_hit_round=3)
    with pytest.raises(L.GuardStop) as e:
        lr.train()
    assert e.value.name == "F9_INVARIANT"


def test_a_start_whose_rows_fail_the_fill_check_stops_before_it_trains(tmp_path: Path) -> None:
    """FILL_MISMATCH (A4's open issue): every learner start, fresh or resumed, runs the
    fill check (`fillcheck.check`) on its first table against `--filled`, and stops
    with exit 87 before that batch trains. The scripted rows here have draw piles
    without PRIV_DRAW, which no declaration makes right."""
    lr = make_learner(make_cfg(tmp_path, fill_check=True, max_iters=3))
    with pytest.raises(L.GuardStop) as e:
        lr.train()
    assert e.value.name == "FILL_MISMATCH"
    rec = OC.read_json(OC.alarm_path(tmp_path, "FILL_MISMATCH"))
    assert rec["severity"] == OC.HUMAN and rec["writer"] == "learner"
    assert "no PRIV_DRAW" in rec["reason"] and rec["filled"] == []
    assert lr.stats_rows == [] and not lr.sd.state.exists()


def _fill(ok: bool, inconclusive: bool, filled: Any) -> Any:
    probs = ("inconclusive: 40 combat rows < 100",) if inconclusive else ()
    n = dict.fromkeys(
        ("rows", "combat_rows", "aux_rows", "enemies", "tele", "forecast", "priv_enemy"), 0
    )
    n |= dict.fromkeys(("dead", "dead_filled", "draw_rows", "draw_missing"), 0)
    return FC.FillCheck(ok=ok, filled=tuple(filled), problems=probs, inconclusive=inconclusive, **n)


def test_the_fill_check_reads_each_starts_first_conclusive_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inconclusive check (too few combat rows or live enemies) is repeated on the
    next table; the first conclusive one ends it for that start. A resume checks
    again (it may add a fill event). The result rides in that iteration's stats row."""
    calls: list[tuple[int, list[str]]] = []

    def fake(rb: dict[str, np.ndarray], filled: Any) -> Any:
        calls.append((len(rb["n_tok"]), list(filled)))
        return _fill(len(calls) != 1, len(calls) == 1, filled)

    monkeypatch.setattr(FC, "check", fake)
    wp = ["WP-ENG-1"]
    lr = make_learner(make_cfg(tmp_path, fill_check=True, max_iters=3, filled=wp))
    assert lr.train() == "max_iters"
    assert [f for _n, f in calls] == [wp, wp] and all(n > 0 for n, _f in calls)
    got = [r.get("fill_check") for r in lr.stats_rows]
    assert got[0]["inconclusive"] and got[1]["ok"] and got[2] is None
    (tmp_path / OC.DONE).unlink()
    cfg = make_cfg(tmp_path, fill_check=True, max_iters=4, filled=wp, resume=str(lr.sd.state))
    lr2 = make_learner(cfg)
    assert lr2.train() == "max_iters" and len(calls) == 3
    assert lr2.stats_rows[0]["fill_check"]["ok"]


# ── exports and the eval queue ───────────────────────────────────────────


def test_export_variants_load_as_net_checkpoints(tmp_path: Path) -> None:
    lr = make_learner(make_cfg(tmp_path, max_iters=2))
    lr.train()
    state = tmp_path / "state.pt"
    outs = {}
    for v in ("raw", "ema0996", "ema0999"):
        out = tmp_path / f"{v}.pt"
        assert len(export_variant(str(state), v, str(out))) == 64
        net, cfg = load_checkpoint(out)
        outs[v] = net
        assert cfg["stamps"]["variant"] == v and cfg["schema_sha256"] == C.schema_sha256()
    raw = outs["raw"].state_dict()
    assert any(not torch.equal(raw[k], outs["ema0996"].state_dict()[k]) for k in raw)
    with pytest.raises(ValueError):
        export_variant(str(state), "ema0997", str(tmp_path / "x.pt"))
    assert L.main(["export", "--state", str(state), "--out", str(tmp_path / "cli.pt")]) == 0


def test_eval_requests_are_claimed_run_and_filed(tmp_path: Path) -> None:
    script = "import sys; open(sys.argv[1], 'w', encoding='utf-8').write(sys.argv[2])"
    cmd = f'{sys.executable} -c "{script}" {{out}} {{variant}}'
    lr = make_learner(make_cfg(tmp_path, eval_cmd=cmd))
    try:
        q = tmp_path / OC.EVAL_QUEUE
        q.mkdir(parents=True)
        (q / "r1.json").write_text(json.dumps({"id": "r1", "variant": "ema0996"}), encoding="utf-8")
        lr._evals_tick()
        assert not (q / "r1.json").exists() and (tmp_path / OC.EVAL_RUNNING / "r1.json").exists()
        p, _path = lr.evals["r1"]
        assert p.wait(timeout=60) == 0
        lr._evals_tick()
        done = OC.read_json(tmp_path / OC.EVAL_DONE / "r1.json")
        assert done["rc"] == 0
        assert Path(done["out"]).read_text(encoding="utf-8") == "ema0996"
        # EVALS_PAUSE stops running evals and puts them back in the queue.
        (q / "r2.json").write_text(json.dumps({"id": "r2"}), encoding="utf-8")
        lr.cfg.eval_cmd = f'{sys.executable} -c "import time; time.sleep(60)"'
        lr._evals_tick()
        assert "r2" in lr.evals
        (tmp_path / OC.EVALS_PAUSE).write_text("", encoding="utf-8")
        lr._evals_tick()
        assert not lr.evals and (q / "r2.json").exists()
    finally:
        lr.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX command line")
def test_eval_commands_split_posix_style_under_nice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(L.shutil, "which", lambda name: "/usr/bin/" + name)
    got = L.eval_argv(["py", "-m", "x", "/a b/req.json"], 10, "--procs 2 --tag 'a b'")
    assert got == [
        "nice",
        "-n",
        "10",
        "py",
        "-m",
        "x",
        "/a b/req.json",
        "--procs",
        "2",
        "--tag",
        "a b",
    ]
    assert L.eval_argv("py -c 'print(1)' /out", 5) == [
        "nice",
        "-n",
        "5",
        "py",
        "-c",
        "print(1)",
        "/out",
    ]
    monkeypatch.setattr(L.shutil, "which", lambda name: None)  # no `nice`: run as is
    assert L.eval_argv("py /out", 5) == ["py", "/out"]


def test_eval_done_carries_the_runner_result(tmp_path: Path) -> None:
    """A command that writes {out}/result.json has its fields (ops contract item 8:
    `wins`, `runs`) added to evals/done/<id>.json; the request's own fields win."""
    script = tmp_path / "runner.py"
    res = {"wins": 7, "runs": 1000, "id": "not-the-request", "arm": "a"}
    script.write_text(
        "import json, os, sys\n"
        "os.makedirs(sys.argv[1], exist_ok=True)\n"
        "with open(os.path.join(sys.argv[1], 'result.json'), 'w', encoding='utf-8') as f:\n"
        f"    f.write({json.dumps(json.dumps(res))})\n",
        encoding="utf-8",
    )
    lr = make_learner(make_cfg(tmp_path, eval_cmd=f"{sys.executable} {script} {{out}}"))
    try:
        q = tmp_path / OC.EVAL_QUEUE
        q.mkdir(parents=True)
        req = {"id": "r1", "n": 1000, "panel": "DEV-1000", "variant": "ema0996"}
        (q / "r1.json").write_text(json.dumps(req), encoding="utf-8")
        lr._evals_tick()
        assert lr.evals["r1"][0].wait(timeout=60) == 0
        lr._evals_tick()
        done = OC.read_json(tmp_path / OC.EVAL_DONE / "r1.json")
        assert {k: done[k] for k in OC.EVAL_RESULT_FIELDS} == {"wins": 7, "runs": 1000}
        assert done["id"] == "r1" and done["arm"] == "a" and done["rc"] == 0
    finally:
        lr.stop()


def _no_caps(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The learner's environment without the worker thread caps, but with its own
    OMP_NUM_THREADS=8; returns it."""
    for k in OC.WORKER_THREAD_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    return dict(os.environ)


CAPS_PROBE = '''
import os


def worker_main(wid, conn, cfg):
    """A stand-in `actors.worker_main`: the caps it started with and torch's threads."""
    import torch

    conn.send(({k: os.environ.get(k) for k in cfg["keys"]}, torch.get_num_threads()))
    conn.recv()  # ("stop",)
'''


def test_rollout_workers_start_under_the_thread_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawned rollout worker has `ops_contract.WORKER_THREAD_ENV` from exec (B3), so
    the numpy and torch it imports before its own code runs start one thread; the
    learner's environment is as it was."""
    (tmp_path / "e2e_caps_probe.py").write_text(CAPS_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))  # a spawned child gets this sys.path
    import e2e_caps_probe

    before = _no_caps(monkeypatch)
    monkeypatch.setattr(actors, "worker_main", e2e_caps_probe.worker_main)
    src = L.ProcessSource(2, {"keys": list(OC.WORKER_THREAD_ENV)}, sync=True)
    try:
        assert dict(os.environ) == before
        got = [src.recv(i) for i in src.ready()]
    finally:
        src.stop()
    assert got == [(OC.WORKER_THREAD_ENV, 1)] * 2


def test_eval_commands_run_under_the_thread_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eval command is a worker too: it starts with the caps (B3), whatever the
    learner's own environment says."""
    before = _no_caps(monkeypatch)
    script = tmp_path / "caps.py"
    script.write_text(
        "import json, os, sys\n"
        f"keys = {list(OC.WORKER_THREAD_ENV)!r}\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
        "    json.dump({k: os.environ.get(k) for k in keys}, f)\n",
        encoding="utf-8",
    )
    lr = make_learner(make_cfg(tmp_path, eval_cmd=f"{sys.executable} {script} {{out}}"))
    try:
        q = tmp_path / OC.EVAL_QUEUE
        q.mkdir(parents=True)
        (q / "r1.json").write_text(json.dumps({"id": "r1"}), encoding="utf-8")
        lr._evals_tick()
        assert lr.evals["r1"][0].wait(timeout=60) == 0
        lr._evals_tick()
        done = OC.read_json(tmp_path / OC.EVAL_DONE / "r1.json")
        assert OC.read_json(done["out"]) == OC.WORKER_THREAD_ENV
        assert dict(os.environ) == before
    finally:
        lr.stop()


def _tiny_state(path: Path) -> str:
    """A slim state of a fresh tiny net (what `export_variant` reads); its sha256."""
    from ppo.e2e.net import E2ENet, NetConfig, net_state
    from ppo.e2e.update import SLIM_FORMAT

    net = E2ENet(NetConfig.from_dict(TINY), seed=0)
    stamps = {"torch": str(torch.__version__)}  # `default_stamps` keeps a TorchVersion
    st = {"format": SLIM_FORMAT, "net": net_state(net, stamps=stamps), "ema": None}
    torch.save(st, path)
    return ckpt.sha256_file(path)


def _request(tmp_path: Path, **kw: Any) -> Path:
    state = tmp_path / "pin.pt"
    sha = _tiny_state(state)
    req = {
        "id": "lane_d0000000000768_DEV-1000_raw_milestone",
        "state": str(state),
        "sha256": sha,
        "decisions": 768,
        "panel": "DEV-1000",
        "n": 1000,
        "seed_start": OC.DEV_START,
        "variant": "raw",
        "max_procs": OC.EVAL_MAX_PROCS,
    } | kw
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    return path


def test_eval_runner_plays_the_exported_variant_on_its_dev_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ppo.e2e import eval_runner as R
    from ppo.e2e.evaluate import CLEAN_MARKER

    monkeypatch.delenv("E2E_PIN", raising=False)
    monkeypatch.setenv("STS2_WHEEL_PIN", "/the/pin")
    monkeypatch.setenv(CLEAN_MARKER, "1")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(argv: list[str], env: dict[str, str]) -> SimpleNamespace:
        calls.append((argv, env))
        return SimpleNamespace(returncode=0)

    def report(evals: Path, panel: Any, arm: str) -> dict[str, Any]:
        assert panel.name == "DEV"
        return {"wins": 230, "scored": 1000, "wr": 0.23}

    req = _request(tmp_path)
    sha = OC.read_json(req)["sha256"]
    out, evals = tmp_path / "out", tmp_path / "evals"
    kw = {"evals": evals, "pin": R.default_pin(), "run": run, "report": report}
    res = R.run_request(req, out, **kw)
    stack = R.stack_id("/the/pin", "cpu", R.DEFAULT_POLICY)
    arm = f"lane_d0000000000768_DEV-1000_raw_milestone_{sha[:8]}_{stack}"
    assert len(stack) == 8 and stack != R.stack_id("/another/pin", "cpu", R.DEFAULT_POLICY)
    student = out / "raw.pt"
    argv, env = calls[0]
    assert argv[1:] == [
        "-m", "ppo.e2e.evaluate", "run", arm, "DEV", "1000", str(OC.DEV_START),
        "--policy", "ppo.e2e.net:make_student", "--ckpt", str(student),
        "--procs", str(OC.EVAL_MAX_PROCS), "--device", "cpu", "--evals", str(evals),
        "--pin", "/the/pin",
    ]  # fmt: skip
    assert CLEAN_MARKER not in env  # evaluate re-executes itself in its clean env
    _net, cfg = load_checkpoint(student)
    assert cfg["stamps"]["variant"] == "raw" and cfg["schema_sha256"] == C.schema_sha256()
    assert res == OC.read_json(out / "result.json")
    assert (res["wins"], res["runs"], res["complete"], res["arm"]) == (230, 1000, True, arm)
    # A rerun (the request came back from EVALS_PAUSE) keeps the student file, so
    # evaluate's arm config (its ckpt sha256) still matches and its shards stay.
    mtime = student.stat().st_mtime_ns
    R.run_request(req, out, **kw)
    assert student.stat().st_mtime_ns == mtime and calls[1][0] == argv


def test_eval_runner_refusals(tmp_path: Path) -> None:
    from ppo.e2e import eval_runner as R

    out = str(tmp_path / "out")
    for kw in (
        {"panel": "TEST", "seed_start": 161_000_000},  # sealed (P4)
        {"seed_start": OC.DEV_START + 1500},  # leaves DEV
        {"sha256": "0" * 64},  # not the pinned state
        {"state": str(tmp_path / "missing.pt")},
    ):
        assert R.main([str(_request(tmp_path, **kw)), "--out", out]) == 2, kw
    assert not (tmp_path / "out" / "result.json").exists()


@pytest.mark.slow
def test_builtin_eval_runner_on_the_real_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The learner claims a SMOKE request and its builtin runner plays two real runs
    with `ppo.e2e.evaluate` and `make_student`; evals/done/<id>.json gets wins and runs."""
    pytest.importorskip("sts2_rs")
    pycli = Path(__file__).resolve().parents[2]
    if not (pycli.parent / "release").is_dir():  # an export (`release/` is never exported)
        from ppo.wheel_guard import default_rs_root, wheel_staleness

        # As test_e2e_evaluate's `needs_tree_wheel`: the export rewrites Rust doc
        # comments, so the runner's wheel guard refuses a wheel built from the tree.
        if why := wheel_staleness(default_rs_root()):
            pytest.skip(f"the wheel was not built from this export's Rust source ({why})")
    monkeypatch.setenv("STS2_GAME_VERSION", V111)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("E2E_PIN", raising=False)
    monkeypatch.delenv("STS2_WHEEL_PIN", raising=False)
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # sts2-pycli: -m finds ppo
    monkeypatch.setenv("PYTHONPATH", "python")
    args = f"--evals {tmp_path / 'evals'} --allow-unpinned --allow-dirty --allow-plain-wheel"
    lr = make_learner(make_cfg(tmp_path / "lane", eval_args=args))
    try:
        req = _request(tmp_path, panel="SMOKE", n=2, seed_start=162_000_000, max_procs=2)
        q = tmp_path / "lane" / OC.EVAL_QUEUE
        q.mkdir(parents=True)
        shutil.copy(req, q / "r1.json")
        lr._evals_tick()
        rid = next(iter(lr.evals))
        rc = lr.evals[rid][0].wait(timeout=600)
        log = tmp_path / "lane" / "evals" / "out" / f"{rid}.log"
        assert rc == 0, log.read_text(encoding="utf-8")[-4000:]
        lr._evals_tick()
        done = OC.read_json(tmp_path / "lane" / OC.EVAL_DONE / f"{rid}.json")
        assert done["rc"] == 0 and done["runs"] == done["n"] == 2 and 0 <= done["wins"] <= 2
        assert Path(done["arm_dir"]).is_dir() and done["complete"] is True
    finally:
        lr.stop()


# ── G-skeleton: the real workers on the real engine ──────────────────────


@pytest.mark.slow
def test_g_skeleton_real_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spawned `actors.worker_main` processes on v0.111.0 A10 runs (practice on),
    the WP3b tokenizer, the local update: three iterations, then a resume. Every
    shipped table of real rows passes the schema's validator."""
    pytest.importorskip("sts2_rs")
    monkeypatch.setenv("STS2_GAME_VERSION", V111)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    shipped = spy_shipped(monkeypatch)
    # Real rows: the fill check runs, against what this wheel fills.
    fill = {"fill_check": True, "filled": list(FC.wheel_filled())}
    cfg = make_cfg(tmp_path, workers=2, n_envs=3, iter_decisions=96, max_iters=3, **fill)
    lr = L.Learner(cfg, updater="local", clock=lambda: S.ANNEAL_START - 86400.0)
    assert lr.train() == "max_iters"
    rows = lr.stats_rows
    assert [r["it"] for r in rows] == [1, 2, 3]
    _check_rows(rows)
    assert sum(r["runs"] for r in rows) > 0 and rows[-1]["errors"] == 0
    assert len(shipped) >= 3
    for rb in shipped:
        C.validate_row_batch(rb)
    boot = np.concatenate([rb["bootstrap"] for rb in shipped])
    done = np.concatenate([rb["done"] for rb in shipped])
    assert (boot & ~done).any() and (done & ~boot).any()  # cut rows and terminals
    st = torch.load(tmp_path / "state.pt", weights_only=True)
    assert st["held"] is None
    (tmp_path / OC.DONE).unlink()
    lr2 = L.Learner(
        make_cfg(
            tmp_path,
            workers=2,
            n_envs=3,
            iter_decisions=96,
            max_iters=4,
            resume=str(tmp_path / "state.pt"),
            f9_tolerate=["crossbow_hand_cap"],  # FIDELITY.md option 2 reaches the env
            **fill,
        ),
        updater="local",
        clock=lambda: S.ANNEAL_START - 86400.0,
    )
    assert lr2.train() == "max_iters" and [r["it"] for r in lr2.stats_rows] == [4]
    assert set(lr2.stats_rows[0]["f9"]) == {"checked", "hits", "tolerated"}
    assert os.path.exists(tmp_path / OC.DONE)


def test_workers_on_another_build_stop_the_learner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8 at the lane level (plan section 3.1): real spawned workers that select
    v0.107.1 under a learner that started on v0.111.0. Each link has its own test
    (the env raises `GameVersionMismatch`, a worker leaves with it, the learner
    refuses another build at startup); this drives the chain end to end. The
    learner must stop with an error, with no stats row and no state.pt: it must not
    train on, book error segments or hang."""
    import contextlib

    pytest.importorskip("sts2_rs")
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)
    monkeypatch.setenv("STS2_GAME_VERSION", V111)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    fill = {"fill_check": True, "filled": list(FC.wheel_filled())}
    cfg = make_cfg(tmp_path, workers=2, n_envs=3, iter_decisions=96, max_iters=3, **fill)
    lr = L.Learner(cfg, updater="local", clock=lambda: S.ANNEAL_START - 86400.0)
    monkeypatch.setenv("STS2_GAME_VERSION", "v0.107.1")  # what the spawned workers select
    try:
        with pytest.raises(RuntimeError, match="rollout worker died"):
            lr.train()
    finally:
        with contextlib.suppress(Exception):
            lr.stop()
    assert lr.stats_rows == [] and not (tmp_path / "state.pt").exists()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and signals")
def test_process_updater_stops_cleanly_on_a_group_sigterm(tmp_path: Path) -> None:
    """The CLI with two spawned CPU update ranks (gloo) and real workers: a SIGTERM
    to the whole process group (a Ctrl-C or `timeout` does that) still finishes
    the update in flight, writes state.pt and exits 0 without DONE (ops contract
    item 5), because the ranks and workers leave the signal to the learner."""
    import signal
    import subprocess
    import time

    pytest.importorskip("sts2_rs")
    sd = tmp_path / "lane"
    argv = [sys.executable, "-u", "-m", "ppo.e2e.learner", "train", OC.FLAG_SAVE_DIR, str(sd)]
    argv += [OC.FLAG_SEED_OFFSET, "0", "--updater", "process", "--workers", "2", "--n-envs", "3"]
    argv += ["--infer-device", "cpu", "--update-devices", "cpu,cpu", "--bf16", "false"]
    argv += ["--iter-decisions", "96", "--minibatch", "64", "--strata-per-mb", "2"]
    argv += ["--probe-strata", "2", "--fwd-chunk", "64", "--net", json.dumps(TINY)]
    argv += ["--state-every", "96", "--k-gates", "[]", "--max-iters", "200"]
    argv += ["--filled", json.dumps(list(FC.wheel_filled()))]  # real rows: the fill check
    env = {**os.environ, "STS2_GAME_VERSION": V111, "OMP_NUM_THREADS": "1", "PYTHONPATH": "python"}
    with open(tmp_path / "learner.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(
            argv,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    stats = sd / OC.STATS
    try:
        t0 = time.time()
        while time.time() - t0 < 300 and p.poll() is None:
            if stats.exists() and len(stats.read_text(encoding="utf-8").splitlines()) >= 2:
                break
            time.sleep(0.5)
        assert p.poll() is None, (tmp_path / "learner.log").read_text(encoding="utf-8")[-3000:]
        os.killpg(p.pid, signal.SIGTERM)
        rc = p.wait(timeout=300)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    text = (tmp_path / "learner.log").read_text(encoding="utf-8")
    assert rc == 0, text[-3000:]
    assert "stopped: SIGTERM" in text and not (sd / OC.DONE).exists()
    st = torch.load(sd / "state.pt", weights_only=True)
    assert st["format"] == L.STATE_FORMAT


# ── the process updater's stop (#1151) ──────────────────────────────────


def _psm() -> set[str]:
    return {p.name for p in Path("/dev/shm").glob("psm_*")} if Path("/dev/shm").exists() else set()


def _returns_within(fn: Any, seconds: float) -> bool:
    import threading

    done = threading.Event()

    def run() -> None:
        fn()
        done.set()

    threading.Thread(target=run, daemon=True).start()
    return done.wait(seconds)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes")
def test_a_torn_rank_message_does_not_hang_the_stop(tmp_path: Path) -> None:
    """A rank killed mid-write leaves a half message in the result pipe: a length
    header and too few bytes. `stop` used to read it and block forever (the parent
    holds the pipe's write end, so no EOF comes); now only the reader thread waits."""
    import struct

    lr = make_learner(make_cfg(tmp_path, update_devices="cpu"), updater="process")
    lr.start()
    up = lr.updater
    os.write(up.outq._writer.fileno(), struct.pack("!i", 1 << 20) + b"x" * 16)
    assert _returns_within(lr.stop, 90), "stop hung on a torn rank message"
    assert not any(p.is_alive() for p in up.procs)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes and /dev/shm")
def test_a_stop_with_an_unread_update_result_is_quick_and_frees_its_held_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The learner fails with an update's result arrived but unread. The ranks exit
    at once (their messages are drained, not left in a full pipe until a kill tears
    one), and `stop` frees the result's held rows' block."""
    import time

    lr = make_learner(make_cfg(tmp_path, update_devices="cpu", max_iters=4), updater="process")
    drained: list[Any] = []
    real_drop = L._drop_held

    def drop(res: Any) -> None:
        drained.append(res)
        real_drop(res)

    monkeypatch.setattr(L, "_drop_held", drop)

    def gone(block: bool) -> bool:  # once the result is there, unread
        deadline = time.time() + 60
        while not lr.updater._inbox.qsize() and time.time() < deadline:
            time.sleep(0.01)
        raise RuntimeError("the learner fails before it reads the result")

    lr._poll = gone  # type: ignore[method-assign]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="fails before"):
        lr.train()
    assert time.time() - t0 < 50  # not the 60 s join deadline and a kill
    assert not any(p.is_alive() for p in lr.updater.procs)
    assert drained, "the unread result was not drained at the stop"
    held = {r["held"]["shm"] for r in drained if isinstance(r, dict) and r.get("held")}
    assert not (held & _psm())  # its held rows' block, if it had one, is gone


# The e2e lane is Linux-only (systemd units, /dev/shm); the public CI also runs Windows.
pytestmark = pytest.mark.skipif(
    'sys.platform == "win32"',
    reason="the learner ships rows through POSIX shared memory that outlives its creator; Windows unlinks a block when its last handle closes. The lane runs on Linux only.",
)
