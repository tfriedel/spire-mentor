"""Tests of the e2e algorithm core (plan §6 and §7; `ppo.e2e.algo`).

The tests carry the plan's names (§7, "Unit tests before G0"): T1 cut vs uncut
GAE, T2 the double boss, T3 bootstrap rows, T4 reward conservation, T7
entropy, T9 KL on student rows, T10 value loss and optimizer steps, T12
strata, and the algorithm half of T11 (a restored state reproduces the next 3
iterations bit for bit). The references port the plan's ``review/scripts/``:
gae_cut.py, gae_two_chain.py, cut_defer.py and minibatch_pad.py. Everything
runs on the CPU in seconds.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import io
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from ppo.e2e import algo
from ppo.e2e import schedules as S

LC, LR = algo.LAM_C, algo.LAM_R
M = 1_000_000
NAN = math.nan


def _exact(a: float, b: float) -> bool:
    return abs(a - b) <= 0.0


# ── references (ports of review/scripts/) ────────────────────────────────


def ref_gae(values: np.ndarray, rewards: np.ndarray, lam: float, boot: float) -> np.ndarray:
    """gae_cut.py: one GAE chain (γ = 1); ``boot`` is the value after the last row."""
    n = len(values)
    nxt = np.append(values[1:], boot)
    delta = rewards + nxt - values
    adv = np.zeros(n)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv


def ref_two_chain(kind, v, rew, lam_c, lam_r, end) -> np.ndarray:
    """gae_two_chain.py: ``kind[t]`` is "C" or "R"; ``end`` is ("term", None) or ("cut", (vb, kb))."""
    n = len(v)
    vb, kb = (0.0, None) if end[0] == "term" else end[1]
    vnext = np.append(v[1:], vb)
    knext = [*kind[1:], kb]
    adv = np.zeros(n)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        if kind[t] != "C":
            continue
        d = rew[t] + vnext[t] - v[t]
        cont = acc if (knext[t] == "C" and t + 1 < n) else 0.0
        acc = d + lam_c * cont
        adv[t] = acc
    acc_r, nxt = 0.0, None
    for i in reversed([i for i in range(n) if kind[i] == "R"]):
        j = nxt if nxt is not None else n
        vj = v[j] if j < n else vb
        d = rew[i:j].sum() + vj - v[i]
        acc_r = d + lam_r * (acc_r if j < n else 0.0)
        adv[i] = acc_r
        nxt = i
    return adv


def cut_terms(kind, v, rew, full, c, lam_c, lam_r, v_end=0.0) -> tuple[float, float]:
    """B per chain for a cut whose bootstrap row is row c (gae_two_chain.py's identity).

    For every row i before the cut, A_full(i) = A_cut(i) + boot_w(i) · B, where
    boot_w(i) = λ^(m - i) counts the chain's steps from i to its last row m
    before the cut.
    """
    b_c = lam_c * full[c] if kind[c - 1] == "C" and kind[c] == "C" else 0.0
    later = [j for j in range(c, len(kind)) if kind[j] == "R"]
    if later:
        j = later[0]
        b_r = rew[c:j].sum() + v[j] - v[c] + lam_r * full[j]
    else:
        b_r = rew[c:].sum() + v_end - v[c]
    return b_c, b_r


def ref_boot_w(kind, c, lam_c, lam_r) -> np.ndarray:
    """The weight of V(row c) in each earlier row's cut target."""
    w = np.zeros(c)
    run_after = 0
    block = kind[c - 1] == "C"
    for i in range(c - 1, -1, -1):
        if kind[i] == "R":
            w[i] = lam_r**run_after
            run_after += 1
            block = False
        elif block:
            w[i] = lam_c ** (c - 1 - i)
    return w


class RefMoments:
    """`ppo.trainer.RunningMoments`, ported (importing ppo.trainer pulls in the engine)."""

    def __init__(self, decay: float = 5e-4) -> None:
        self.decay, self.mean, self.sq, self.inited = decay, 0.0, 1.0, False

    def update(self, xs: np.ndarray) -> None:
        if xs.size == 0:
            return
        if not self.inited:
            self.mean, self.sq, self.inited = float(xs.mean()), float((xs * xs).mean()), True
            return
        keep = (1.0 - self.decay) ** xs.size
        self.mean = keep * self.mean + (1.0 - keep) * float(xs.mean())
        self.sq = keep * self.sq + (1.0 - keep) * float((xs * xs).mean())

    @property
    def std(self) -> float:
        return math.sqrt(max(1e-6, self.sq - self.mean * self.mean))


# ── tables ───────────────────────────────────────────────────────────────


def _pad(x, m: int, dt) -> np.ndarray:
    a = np.asarray(x, dt)
    return np.concatenate([a, np.zeros(m - len(a), dt)]) if len(a) < m else a


def run_table(kind, rew, *, end="term", uid=7, t0=0, actor=None, dtype=None, boot_reward=NAN):
    """One run's rows. ``kind`` is a string of "C" and "R".

    ``end``: "term" (the last row is done), "C" / "R" (a cut: a bootstrap row
    of that kind follows), "pC" / "pR" (a practice segment's done bootstrap
    row follows) or None (no end marker).
    """
    comb = [k == "C" for k in kind]
    rewards = [float(x) for x in rew]
    n = len(comb)
    boot, done = [False] * n, [False] * n
    if end == "term":
        done[-1] = True
    elif end is not None:
        comb.append(end[-1] == "C")
        boot.append(True)
        done.append(end.startswith("p"))
        rewards.append(boot_reward)
    m = len(comb)
    return {
        "run_uid": np.full(m, uid, np.int64),
        "t": np.arange(t0, t0 + m, dtype=np.int32),
        "is_combat": np.array(comb, bool),
        "dtype": _pad([] if dtype is None else dtype, m, np.uint8),
        "reward": np.array(rewards, np.float64),
        "done": np.array(done, bool),
        "bootstrap": np.array(boot, bool),
        "actor": _pad([] if actor is None else actor, m, np.uint8),
    }


def cat(*tables: dict) -> dict:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


#: Every row column of the stream: the schema's fields plus ``ref`` (the row's value id).
COLS = {
    "run_uid": np.int64,
    "t": np.int32,
    "is_combat": bool,
    "dtype": np.uint8,
    "reward": np.float32,
    "done": bool,
    "bootstrap": bool,
    "actor": np.uint8,
    "logp_behav": np.float32,
    "n_classes": np.uint16,
    "action": np.uint16,
    "iter": np.int32,
    "v_behav": np.float32,
    "cell": np.uint8,
    "ref": np.int64,
}


@dataclass(eq=False)
class SimRun:
    uid: int
    cols: dict  # per ordinary row
    practice: bool
    end_ref: int  # a practice segment's final bootstrap row (else -1)
    end_comb: bool

    @property
    def n(self) -> int:
        return len(self.cols["ref"])

    def kind(self) -> str:
        return "".join("C" if c else "R" for c in self.cols["is_combat"])


class Stream:
    """A synthetic collector: envs play back-to-back runs and ship ``per_iter`` rows each per iteration.

    A run still going at the cut gets a bootstrap row: a copy of its next row
    with garbage in every column but ``ref``. The env stores that row again,
    as an ordinary row, in the next iteration it ships. A practice segment ends
    in a bootstrap row with ``done`` set. With ``stall_p`` an env ships nothing
    in an iteration, so its held rows go dormant. ``shape`` "fights" follows
    gae_two_chain.py (fights of 15-60 combat rows between 1-6 run rows, then a
    double boss); "defer" follows cut_defer.py (800 rows, 59 run rows at random
    positions). Values are fixed per row (``values()[ref]``), as if the net
    did not change, so cut and uncut returns are comparable.
    """

    def __init__(
        self,
        n_envs,
        per_iter,
        *,
        seed,
        shape="fights",
        run_len=(100, 300),
        practice_p=0.0,
        stall_p=0.0,
        first_budget=None,
        teacher_p=0.3,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_envs, self.per_iter, self.shape, self.run_len = n_envs, per_iter, shape, run_len
        self.practice_p, self.stall_p, self.teacher_p = practice_p, stall_p, teacher_p
        self.first_budget = first_budget
        self.runs: dict[int, SimRun] = {}
        self.cur: list[SimRun | None] = [None] * n_envs
        self.pos = [0] * n_envs
        self.k = 0
        self._uid = 1000
        self._vchunks: list[np.ndarray] = []
        self._nv = 0
        self._vcache = np.zeros(0)
        self._full: dict[int, np.ndarray] = {}
        self._terms: dict[tuple[int, int], tuple[float, float]] = {}

    def values(self) -> np.ndarray:
        if len(self._vcache) != self._nv:
            self._vcache = np.concatenate(self._vchunks)
        return self._vcache

    def _refs(self, n):
        vals = self.rng.uniform(0.0, 1.0, n)
        self._vchunks.append(vals)
        start = self._nv
        self._nv += n
        return np.arange(start, start + n, dtype=np.int64), vals

    def _new_run(self) -> SimRun:
        rng = self.rng
        practice = self.practice_p > 0 and rng.random() < self.practice_p
        if self.shape == "defer":
            comb = np.ones(800, bool)
            comb[rng.choice(800, 59, replace=False)] = False
        else:
            lo, hi = self.run_len
            want = int(rng.integers(lo, hi)) // (3 if practice else 1)
            kinds: list[bool] = []
            while len(kinds) < want:
                kinds += [False] * int(rng.integers(1, 7)) + [True] * int(rng.integers(15, 61))
            kinds = kinds[:want]
            if not practice:
                kinds += [True] * 116  # boss 1 then boss 2, back to back
            comb = np.array(kinds, bool)
        n = len(comb)
        rew = np.zeros(n, np.float32)
        hit = rng.random(n) < 0.05
        rew[hit] = rng.choice(np.array([0.002, 0.05], np.float32), int(hit.sum()))
        if not practice:
            rew[-1] += np.float32(rng.integers(0, 2))
        nc = rng.integers(2, 9, n)
        teach = rng.random(n) < self.teacher_p
        refs, vals = self._refs(n)
        cell = int(rng.integers(0, 10)) if practice else algo.CELL_NONE
        cols = {
            "run_uid": np.full(n, self._uid, np.int64),
            "t": np.arange(n, dtype=np.int32),
            "is_combat": comb,
            "dtype": np.where(comb, rng.integers(0, 5, n), rng.integers(10, 20, n)).astype(
                np.uint8
            ),
            "reward": rew,
            "actor": np.where(teach, np.where(comb, 1, 2), 0).astype(np.uint8),
            "logp_behav": (-np.log(nc) + rng.normal(0, 0.1, n)).astype(np.float32),
            "n_classes": nc.astype(np.uint16),
            "action": rng.integers(0, nc).astype(np.uint16),
            "v_behav": (vals + rng.normal(0, 0.1, n)).astype(np.float32),
            "cell": np.full(n, cell, np.uint8),
            "ref": refs,
        }
        end_ref, end_comb = -1, False
        if practice:
            end_ref, end_comb = int(self._refs(1)[0][0]), bool(rng.random() < 0.5)
        run = SimRun(self._uid, cols, practice, end_ref, end_comb)
        self.runs[self._uid] = run
        self._uid += 1
        return run

    def _rows(self, run: SimRun, p: int, q: int, k: int) -> dict:
        out = {c: v[p:q] for c, v in run.cols.items()}
        out["done"] = (np.arange(p, q) == run.n - 1) & (not run.practice)
        out["bootstrap"] = np.zeros(q - p, bool)
        out["iter"] = np.full(q - p, k, np.int32)
        return out

    def _garbage(self, run: SimRun, ref: int, t: int, comb: bool, done: bool, k: int) -> dict:
        """A bootstrap row: only run_uid, t, is_combat, done, bootstrap and ref are meaningful."""
        return {
            "run_uid": np.array([run.uid], np.int64),
            "t": np.array([t], np.int32),
            "is_combat": np.array([comb]),
            "dtype": np.array([7], np.uint8),
            "reward": np.array([NAN], np.float32),
            "done": np.array([done]),
            "bootstrap": np.array([True]),
            "actor": np.array([algo.ACTOR_STUDENT], np.uint8),
            "logp_behav": np.array([NAN], np.float32),
            "n_classes": np.array([0], np.uint16),
            "action": np.array([999], np.uint16),
            "iter": np.array([k], np.int32),
            "v_behav": np.array([NAN], np.float32),
            "cell": np.array([algo.CELL_NONE], np.uint8),
            "ref": np.array([ref], np.int64),
        }

    def next_table(self) -> dict[str, np.ndarray]:
        """The fresh rows of the next iteration: stored rows plus bootstrap rows."""
        k = self.k
        self.k += 1
        parts = []
        for e in range(self.n_envs):
            if self.stall_p and self.rng.random() < self.stall_p:
                continue
            first = k == 0 and self.first_budget is not None
            budget = int(self.first_budget[e]) if first else self.per_iter
            while budget > 0:
                if self.cur[e] is None:
                    self.cur[e], self.pos[e] = self._new_run(), 0
                run, p = self.cur[e], self.pos[e]
                q = min(run.n, p + budget)
                parts.append(self._rows(run, p, q, k))
                budget -= q - p
                self.pos[e] = q
                if q == run.n:
                    if run.practice:
                        parts.append(self._garbage(run, run.end_ref, run.n, run.end_comb, True, k))
                    self.cur[e] = None
            run = self.cur[e]
            if run is not None:
                p = self.pos[e]
                ref, comb = int(run.cols["ref"][p]), bool(run.cols["is_combat"][p])
                parts.append(self._garbage(run, ref, p, comb, False, k))
        if not parts:
            return {c: np.zeros(0, dt) for c, dt in COLS.items()}
        return {c: np.concatenate([x[c] for x in parts]).astype(dt) for c, dt in COLS.items()}

    def full_adv(self, run: SimRun) -> np.ndarray:
        """The run's uncut two-chain advantages (gae_two_chain.py)."""
        if run.uid not in self._full:
            vv = self.values()
            end = ("cut", (vv[run.end_ref], "C" if run.end_comb else "R"))
            self._full[run.uid] = ref_two_chain(
                run.kind(),
                vv[run.cols["ref"]],
                run.cols["reward"].astype(np.float64),
                LC,
                LR,
                end if run.practice else ("term", None),
            )
        return self._full[run.uid]

    def terms(self, run: SimRun, c: int) -> tuple[float, float]:
        """`cut_terms` of a cut at row c of the run."""
        key = (run.uid, c)
        if key not in self._terms:
            vv = self.values()
            v_end = vv[run.end_ref] if run.practice else 0.0
            self._terms[key] = cut_terms(
                run.kind(),
                vv[run.cols["ref"]],
                run.cols["reward"].astype(np.float64),
                self.full_adv(run),
                c,
                LC,
                LR,
                v_end,
            )
        return self._terms[key]


# ── T1: cut vs uncut GAE ─────────────────────────────────────────────────


@pytest.mark.parametrize("lam", [0.95, 0.97, 0.9])
def test_t1_single_chain_cut_identity(lam: float) -> None:
    """gae_cut.py: A_uncut(k) = A_cut(k) + λ^(c - k) · A_uncut(c), for either chain alone."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        n = int(rng.integers(5, 400))
        v = rng.uniform(0, 1.2, n)
        r = np.zeros(n)
        r[-1] = float(rng.integers(0, 2))
        r[rng.integers(0, n, 3)] += 0.05
        c = int(rng.integers(1, n))
        k = np.arange(c)
        for ch in "CR":
            lam_c, lam_r = (lam, 0.5) if ch == "C" else (0.5, lam)
            kw = {"lam_c": lam_c, "lam_r": lam_r, "hold_iters": 0}
            full = algo.compute_returns(run_table(ch * n, r), v, **kw)
            np.testing.assert_allclose(full.adv, ref_gae(v, r, lam, 0.0), rtol=0, atol=1e-12)
            assert full.trained.all() and np.all(full.boot_w == 0)
            tab = run_table(ch * c, r[:c], end=ch)
            cut = algo.compute_returns(tab, np.append(v[:c], v[c]), **kw)
            np.testing.assert_allclose(cut.adv[:c], ref_gae(v[:c], r[:c], lam, v[c]), atol=1e-12)
            np.testing.assert_allclose(cut.boot_w[:c], lam ** (c - 1 - k), rtol=1e-12)
            assert not cut.trained[c] and np.isnan(cut.adv[c])
            ident = full.adv[:c] - (cut.adv[:c] + lam ** (c - k) * full.adv[c])
            worst = max(worst, float(np.abs(ident).max()))
    assert worst <= 1e-9


def _rand_kind(rng, n: int) -> str:
    """gae_two_chain.py's rand_run: fights between run rows, then a double boss."""
    kind: list[str] = []
    while len(kind) < n:
        kind += ["R"] * int(rng.integers(1, 7)) + ["C"] * int(rng.integers(15, 61))
    return "".join(kind[:n]) + "C" * 116


def test_t1_two_chain_matches_the_reference_and_its_cut_identity() -> None:
    """gae_two_chain.py: per chain, A_uncut(i) = A_cut(i) + λ^(m - i) · B."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(150):
        kind = _rand_kind(rng, int(rng.integers(100, 400)))
        n = len(kind)
        v = rng.uniform(0.0, 1.0, n)
        rew = np.zeros(n)
        hit = rng.random(n) < 0.05
        rew[hit] = rng.choice([0.002, 0.05], int(hit.sum()))
        rew[-1] += float(rng.integers(0, 2))
        full_ref = ref_two_chain(kind, v, rew, LC, LR, ("term", None))
        full = algo.compute_returns(run_table(kind, rew), v, hold_iters=0)
        np.testing.assert_allclose(full.adv, full_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(full.target, full_ref + v, rtol=0, atol=1e-12)
        assert np.all(full.boot_w == 0)  # nothing bootstraps without a cut
        c = int(rng.integers(1, n))
        cut_ref = ref_two_chain(kind[:c], v[:c], rew[:c], LC, LR, ("cut", (v[c], kind[c])))
        tab = run_table(kind[:c], rew[:c], end=kind[c])
        cut = algo.compute_returns(tab, np.append(v[:c], v[c]), hold_iters=0)
        np.testing.assert_allclose(cut.adv[:c], cut_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cut.boot_w[:c], ref_boot_w(kind, c, LC, LR), rtol=1e-12)
        b_c, b_r = cut_terms(kind, v, rew, full_ref, c, LC, LR)
        corr = cut.boot_w[:c] * np.where(tab["is_combat"][:c], b_c, b_r)
        worst = max(worst, float(np.abs(full_ref[:c] - (cut.adv[:c] + corr)).max()))
    assert worst <= 1e-9


def test_t1_t3_stream_cut_identity_and_each_row_trains_once() -> None:
    """Many envs, back-to-back runs, practice segments, stalls and the 4-iteration hold-back.

    Every trained row obeys the cut identity against its run's uncut returns
    (exactly when its chain reaches the run's end); every stored row trains
    exactly once; every mid-run bootstrap row is stored again later.
    """
    st = Stream(12, 64, seed=3, run_len=(80, 260), practice_p=0.2, stall_p=0.08)
    hold = algo.HoldBack(algo.HOLD_ITERS)
    n_iter = 70
    trained = collections.Counter()
    stored: dict[int, int] = {}
    boots: list[tuple[int, int]] = []
    worst_cut = worst_end = 0.0
    n_cut = n_end = n_released = n_dormant = 0
    for k in range(n_iter):
        fresh = st.next_table()
        b_f = fresh["bootstrap"]
        for r in fresh["ref"][~b_f].tolist():
            assert r not in stored
            stored[r] = k
        boots += [(k, r) for r in fresh["ref"][b_f & ~fresh["done"]].tolist()]
        table = hold.join(fresh)
        ret = algo.compute_returns(table, st.values()[table["ref"]])
        boot, comb, age = table["bootstrap"], table["is_combat"], table["age"]
        assert not (ret.trained & boot).any() and not (ret.held & boot).any()
        assert np.isnan(ret.adv[boot]).all() and np.isnan(ret.target[boot]).all()
        assert not (ret.held & comb).any() and not (ret.trained & ret.held).any()
        assert (ret.trained | ret.held | boot).all()
        np.testing.assert_array_equal(ret.pi, ret.trained & (table["actor"] == 0))
        assert (age[ret.held & ~ret.dormant] < algo.HOLD_ITERS).all()
        assert (age[ret.trained] <= algo.HOLD_ITERS).all()
        n_dormant += int(ret.dormant.sum())
        n_released += int((ret.trained & (age == algo.HOLD_ITERS)).sum())
        cut_at = {
            int(u): int(t)
            for u, t in zip(
                table["run_uid"][boot & ~table["done"]], table["t"][boot & ~table["done"]]
            )
        }
        for j in np.flatnonzero(ret.trained).tolist():
            run = st.runs[int(table["run_uid"][j])]
            i = int(table["t"][j])
            want = st.full_adv(run)[i]
            if run.uid in cut_at:
                b_c, b_r = st.terms(run, cut_at[run.uid])
                corr = ret.boot_w[j] * (b_c if comb[j] else b_r)
                worst_cut = max(worst_cut, abs(want - (ret.adv[j] + corr)))
                n_cut += 1
            else:
                worst_end = max(worst_end, abs(want - ret.adv[j]))
                n_end += 1
            trained[int(table["ref"][j])] += 1
        hold.carry(table, ret)
    assert worst_cut <= 1e-9 and worst_end <= 1e-12
    assert n_cut > 1000 and n_end > 1000 and n_released > 50 and n_dormant > 0
    assert max(trained.values()) == 1
    # Every stored row of every finished run trained, exactly once.
    finished = [r for r in st.runs.values() if all(x in stored for x in r.cols["ref"].tolist())]
    finished = [r for r in finished if r.uid not in hold.cols.get("run_uid", [])]
    assert len(finished) > 100
    for run in finished:
        assert all(trained[x] == 1 for x in run.cols["ref"].tolist()), run.uid
    # Every mid-run bootstrap row came back as an ordinary row in a later iteration.
    late = [(k, r) for k, r in boots if k < n_iter - 3]
    assert late and all(stored.get(r, -1) > k for k, r in late)


@pytest.mark.parametrize(
    ("hold_iters", "want_w", "want_stale"), [(0, 0.783, 1.0), (4, 0.078, 3.72)]
)
def test_hold_back_bootstrap_weight_matches_cut_defer(
    hold_iters: int, want_w: float, want_stale: float
) -> None:
    """cut_defer.py: 800-row runs with 59 run rows, 128 rows per env per iteration.

    The reference's weight is λ^m with m the run rows from the row up to the
    cut, i.e. ``boot_w · λ_r``; its staleness is ``age + 1`` iterations.
    """
    rng = np.random.default_rng(2)
    n_envs, per = 64, 128
    st = Stream(n_envs, per, seed=4, shape="defer", first_budget=rng.integers(1, per + 1, n_envs))
    hold = algo.HoldBack(hold_iters)
    uids, ws, ages, ended = [], [], [], set()
    for _ in range(34):
        table = hold.join(st.next_table())
        ret = algo.compute_returns(table, st.values()[table["ref"]], hold_iters=hold_iters)
        m = ret.trained & ~table["is_combat"]
        uids.append(table["run_uid"][m])
        ws.append(ret.boot_w[m] * LR)
        ages.append(table["age"][m])
        ended.update(table["run_uid"][table["done"]].tolist())
        hold.carry(table, ret)
    uid, w, age = np.concatenate(uids), np.concatenate(ws), np.concatenate(ages)
    keep = np.isin(uid, np.fromiter(ended, np.int64))
    assert len(ended) >= 250 and keep.sum() == 59 * len(ended)
    w, stale = w[keep], age[keep] + 1
    assert abs(float(w.mean()) - want_w) <= 0.02, float(w.mean())
    assert abs(float(stale.mean()) - want_stale) <= 0.08, float(stale.mean())
    assert int(stale.max()) == hold_iters + 1


# ── T2: the double boss ──────────────────────────────────────────────────


def test_t2_boss_one_chains_into_boss_two_and_the_bonus_counts_once() -> None:
    rng = np.random.default_rng(2)
    kind = "RR" + "C" * 5 + "C" * 4  # two run rows, boss 1 (rows 2-6), boss 2 (rows 7-10)
    n = len(kind)
    rew = np.zeros(n)
    rew[6] = 0.05  # boss-1 bonus, on the transition into boss 2
    rew[-1] = 1.0  # the run is won
    v = rng.uniform(0, 1, n)
    ret = algo.compute_returns(run_table(kind, rew), v, hold_iters=0)
    assert ret.succ[6] == 7
    delta = rew[6] + v[7] - v[6]
    assert ret.adv[6] == pytest.approx(delta + LC * ret.adv[7], abs=1e-12)
    np.testing.assert_allclose(ret.adv, ref_two_chain(kind, v, rew, LC, LR, ("term", None)))
    # λ = 1: every chain telescopes; the bonus is counted once.
    ret1 = algo.compute_returns(run_table(kind, rew), v, lam_c=1.0, lam_r=1.0, hold_iters=0)
    np.testing.assert_allclose(ret1.target[:7], 1.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ret1.target[7:], 1.0, rtol=0, atol=1e-12)
    # Exact values: every δ is 0.
    v_exact = np.r_[np.full(7, 1.05), np.full(4, 1.0)]
    ret2 = algo.compute_returns(run_table(kind, rew), v_exact, hold_iters=0)
    np.testing.assert_allclose(ret2.adv, 0.0, rtol=0, atol=1e-12)
    # Boss 1 lost: the run ends at its last row, nothing follows.
    ret3 = algo.compute_returns(run_table(kind[:7], np.zeros(7)), v[:7], hold_iters=0)
    assert ret3.succ[6] == -1 and ret3.adv[6] == pytest.approx(-v[6], abs=1e-12)
    # A cut between the bosses: boss 1 bootstraps from V_combat of boss 2's first row.
    tab = run_table(kind[:7], rew[:7], end="C")
    cut = algo.compute_returns(tab, np.r_[v[:7], v[7]], hold_iters=0)
    assert cut.adv[6] == pytest.approx(delta, abs=1e-12) and _exact(cut.boot_w[6], 1.0)
    b_c, b_r = cut_terms(kind, v, rew, ret.adv, 7, LC, LR)
    assert b_c == pytest.approx(LC * ret.adv[7], abs=1e-15)  # boss 1's chain continues into boss 2
    corr = cut.boot_w[:7] * np.where(tab["is_combat"][:7], b_c, b_r)
    np.testing.assert_allclose(cut.adv[:7] + corr, ret.adv[:7], rtol=0, atol=1e-12)


# ── T3: bootstrap rows are inert ─────────────────────────────────────────


@pytest.mark.parametrize("hold_iters", [0, 4])
def test_t3_bootstrap_row_is_inert_in_the_returns(hold_iters: int) -> None:
    rng = np.random.default_rng(3)
    kind = "RRCCCCRCCC"
    b = len(kind)
    base = run_table(kind, rng.uniform(0, 0.1, b), end="C")
    v = rng.uniform(0, 1, b + 1)
    ret = algo.compute_returns(base, v, hold_iters=hold_iters)
    assert not ret.trained[b] and not ret.pi[b] and not ret.held[b]
    assert np.isnan([ret.adv[b], ret.target[b], ret.boot_w[b]]).all()
    assert ret.trained[~base["is_combat"]].all() == (hold_iters == 0)
    for col, junk in [("reward", math.inf), ("reward", -1e30), ("actor", 1), ("dtype", 200)]:
        tab = {k: x.copy() for k, x in base.items()}
        tab[col][b] = junk
        r2 = algo.compute_returns(tab, v, hold_iters=hold_iters)
        for f in ("adv", "target", "boot_w", "trained", "pi", "held"):
            np.testing.assert_array_equal(getattr(r2, f), getattr(ret, f))
    # Only its value counts: the targets move by boot_w · ΔV.
    v2 = v.copy()
    v2[b] += 0.37
    r3 = algo.compute_returns(base, v2, hold_iters=hold_iters)
    m = ret.trained
    np.testing.assert_allclose(r3.target[m] - ret.target[m], 0.37 * ret.boot_w[m], atol=1e-12)
    assert (ret.boot_w[m] > 0).sum() >= 3


def _loss_rows(b, a, *, seed, pi_frac=0.6, teacher=True, ks=0.5, beta=0.01) -> algo.LossRows:
    g = torch.Generator().manual_seed(seed)
    nc = torch.randint(2, a + 1, (b,), generator=g)
    action = (torch.rand(b, generator=g) * nc).long()
    lp_prox = -torch.log(nc.float()) + 0.3 * torch.randn(b, generator=g)
    tp = None
    if teacher:
        cm = algo.class_mask(nc, a)
        raw = torch.where(cm, torch.rand(b, a, generator=g) + 0.01, torch.zeros(b, a))
        tp = torch.where(cm, raw / raw.sum(-1, keepdim=True), torch.full((b, a), NAN))
    return algo.LossRows(
        n_classes=nc,
        action=action,
        lp_prox=lp_prox,
        lp_behav=lp_prox + 0.5 * torch.randn(b, generator=g),
        adv=torch.randn(b, generator=g),
        target=torch.randn(b, generator=g),
        pi=torch.rand(b, generator=g) < pi_frac,
        v=torch.ones(b, dtype=torch.bool),
        beta=torch.full((b,), beta),
        ks=torch.full((b,), ks),
        dtype=torch.randint(0, 8, (b,), generator=g),
        teacher=tp,
    )


def _with_garbage_row(rows: algo.LossRows) -> algo.LossRows:
    """``rows`` plus a bootstrap-like row: outside v, NaN or nonsense in every column."""

    def add(x: torch.Tensor, val) -> torch.Tensor:
        return torch.cat([x, torch.tensor([val], dtype=x.dtype)])

    teacher = rows.teacher
    if teacher is not None:
        teacher = torch.cat([teacher, torch.full((1, teacher.shape[1]), NAN)])
    return algo.LossRows(
        n_classes=add(rows.n_classes, 0),
        action=add(rows.action, 999),
        lp_prox=add(rows.lp_prox, NAN),
        lp_behav=add(rows.lp_behav, NAN),
        adv=add(rows.adv, NAN),
        target=add(rows.target, NAN),
        pi=add(rows.pi, True),  # pi without v is still inert
        v=add(rows.v, False),
        beta=add(rows.beta, NAN),
        ks=add(rows.ks, NAN),
        dtype=add(rows.dtype, 300),
        teacher=teacher,
    )


def test_t3_bootstrap_rows_are_inert_in_every_loss_and_statistic() -> None:
    b, a = 16, 6
    rows = _loss_rows(b, a, seed=3, ks=0.6, beta=0.02)
    g = torch.Generator().manual_seed(33)
    logits = torch.randn(b, a, generator=g).requires_grad_()
    value = torch.randn(b, generator=g).requires_grad_()
    out = algo.ppo_losses(logits, value, rows)
    out.loss.backward()
    rows_g = _with_garbage_row(rows)
    lg = torch.cat([logits.detach(), torch.full((1, a), NAN)]).requires_grad_()
    vg = torch.cat([value.detach(), torch.tensor([NAN])]).requires_grad_()
    out_g = algo.ppo_losses(lg, vg, rows_g)
    assert torch.isfinite(out_g.loss)
    torch.testing.assert_close(out_g.loss, out.loss)
    assert out_g.denom == out.denom == b
    assert out_g.stats.keys() == out.stats.keys()
    for k in out.stats:
        torch.testing.assert_close(out_g.stats[k], out.stats[k], msg=k)
    out_g.loss.backward()
    assert (lg.grad[-1] == 0).all() and vg.grad[-1] == 0
    torch.testing.assert_close(lg.grad[:-1], logits.grad)
    torch.testing.assert_close(vg.grad[:-1], value.grad)
    # The pre-pass helpers mask the same way.
    mask = torch.cat([rows.pi, torch.tensor([False])])
    s, n = algo.entropy_by_type(lg.detach(), rows_g.n_classes, rows_g.dtype, mask)
    s0, n0 = algo.entropy_by_type(logits.detach(), rows.n_classes, rows.dtype, rows.pi)
    torch.testing.assert_close(s, s0)
    torch.testing.assert_close(n, n0)
    k, kn = algo.probe_kl(lg.detach(), lg.detach() * 0.5, rows_g.n_classes, mask)
    k0, kn0 = algo.probe_kl(logits.detach(), logits.detach() * 0.5, rows.n_classes, rows.pi)
    torch.testing.assert_close(k, k0)
    assert _exact(float(kn), float(kn0))


def test_t3_bootstrap_rows_stay_out_of_the_moments() -> None:
    st = algo.AlgoState(hold_iters=0)
    st.begin(10, S.ANNEAL_START - 1e6)
    t1 = run_table("RCCRCC", [0.1, 0, 0.2, 0, 0.3, 0], end="R", uid=1, dtype=[10, 1, 1, 11, 2, 2])
    t2 = run_table("CCCC", [0, 0, 0, 1], uid=2, actor=[0, 1, 0, 0], dtype=[1, 1, 2, 3])
    table = st.join(cat(t1, t2))
    table["dtype"][6] = 4  # the bootstrap row's type: no row of type 4 trains
    v = np.linspace(0.1, 0.9, len(table["run_uid"]))
    ret = st.returns(table, v)
    norm = st.normalize(table, ret)
    assert not ret.pi[6] and np.isnan(norm[6])
    assert np.isnan(norm[~ret.pi]).all() and np.isfinite(norm[ret.pi]).all()
    assert not st.adv.inited[4]
    for d in (1, 2, 3, 10, 11):
        ref = RefMoments()
        ref.update(ret.adv[ret.pi & (table["dtype"] == d)])
        assert st.adv.mean[d] == pytest.approx(ref.mean, abs=1e-12), d
        assert st.adv.std()[d] == pytest.approx(ref.std, abs=1e-12), d


# ── T4: Σr over stored rows equals the env total ─────────────────────────


def test_t4_row_rewards_conserve_the_env_total() -> None:
    rng = np.random.default_rng(4)
    for _ in range(300):
        n = int(rng.integers(1, 300))
        step_r = rng.normal(0, 1, n) * (rng.random(n) < 0.3)
        stored = rng.random(n) < 0.6  # auto-advanced steps are not stored
        stored[int(rng.integers(0, n))] = True
        r = algo.row_rewards(step_r, stored)
        idx = np.flatnonzero(stored)
        assert len(r) == len(idx)
        assert r.sum() == pytest.approx(step_r.sum(), abs=1e-9)
        ends = [*idx[1:].tolist(), n]
        want = [step_r[s:e].sum() for s, e in zip(idx.tolist(), ends)]
        want[0] += step_r[: idx[0]].sum()
        np.testing.assert_allclose(r, want, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        algo.row_rewards([1.0, 2.0], [False, False])


def test_t4_lambda_one_target_of_a_runs_first_row_is_its_total_reward() -> None:
    """The cut and the hold-back fold rewards without losing any (λ = 1, run rows wait to the end)."""
    st = Stream(6, 48, seed=5, run_len=(100, 250))
    hold = algo.HoldBack(10_000)
    checked = 0
    for _ in range(45):
        table = hold.join(st.next_table())
        kw = {"lam_c": 1.0, "lam_r": 1.0, "hold_iters": 10_000}
        ret = algo.compute_returns(table, st.values()[table["ref"]], **kw)
        for j in np.flatnonzero(ret.trained & (table["t"] == 0)).tolist():
            run = st.runs[int(table["run_uid"][j])]
            total = float(run.cols["reward"].astype(np.float64).sum())
            assert ret.target[j] == pytest.approx(total, abs=1e-9)
            checked += 1
        hold.carry(table, ret)
    assert checked >= 20


# ── hold-back bookkeeping ────────────────────────────────────────────────


def test_hold_back_folds_rewards_ages_rows_and_waits_while_dormant() -> None:
    hb = algo.HoldBack(4)
    t1 = hb.join(run_table("RCCRC", [1, 2, 4, 8, 16], end="C", uid=1))
    assert t1["age"].tolist() == [0] * 6 and not t1["carried"].any()
    r1 = algo.compute_returns(t1, np.full(6, 0.5))
    hb.carry(t1, r1)
    assert len(hb) == 2 and hb.cols["t"].tolist() == [0, 3]
    np.testing.assert_array_equal(hb.cols["reward"], [7.0, 24.0])  # rewards of rows that left
    assert hb.cols["age"].tolist() == [1, 1] and hb.cols["carried"].all()
    # The env stalls: run 1 sends nothing. Its rows are dormant: no values needed, no ageing.
    other = run_table("CC", [0, 1], uid=2)
    t2 = hb.join(other)
    r2 = algo.compute_returns(t2, np.r_[NAN, NAN, 0.5, 0.5])
    assert r2.dormant.tolist() == [True, True, False, False] and r2.held[:2].all()
    hb.carry(t2, r2)
    assert hb.cols["age"].tolist() == [1, 1]
    assert hb.cols["waited"].tolist() == [2, 2]  # waited counts the dormant iteration
    # Run 1 resumes and ends: its held rows train with the chain to the terminal.
    t3 = hb.join(run_table("CRC", [32, 64, 128], uid=1, t0=5))
    v3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r3 = algo.compute_returns(t3, v3)
    assert r3.trained.all() and np.all(r3.boot_w == 0)
    full = ref_two_chain(
        "RCCRCCRC",
        np.r_[0.1, 0.5, 0.5, 0.2, 0.5, 0.3, 0.4, 0.5],
        np.array([1, 2, 4, 8, 16, 32, 64, 128.0]),
        LC,
        LR,
        ("term", None),
    )
    np.testing.assert_allclose(r3.adv[:2], full[[0, 3]], rtol=0, atol=1e-12)
    hb.carry(t3, r3)
    assert len(hb) == 0


def test_hold_back_releases_at_age_four_and_a_practice_segment_at_its_end() -> None:
    hb = algo.HoldBack(4)
    ages = []
    for k in range(5):
        tab = run_table("RC", [0.0, 0.1], end="C", uid=3, t0=2 * k)
        table = hb.join(tab)
        ret = algo.compute_returns(table, np.full(len(table["t"]), 0.5))
        ages.append(table["age"][ret.trained & ~table["is_combat"]].tolist())
        hb.carry(table, ret)
    assert ages == [[], [], [], [], [4]]  # the first run row trains in its fifth iteration
    # A practice segment: its bootstrap row is done, so every held row trains, bootstrapping from it.
    hb = algo.HoldBack(4)
    seg = run_table("RCCR", [0.1, 0.0, 0.2, 0.0], end="C", uid=9)
    t1 = hb.join(seg)
    v = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    hb.carry(t1, algo.compute_returns(t1, v))
    t2 = hb.join(run_table("CCR", [0.0, 0.05, 0.0], end="pC", uid=9, t0=4))
    v2 = np.array([0.3, 0.6, 0.7, 0.8, 0.2, 0.9])
    r2 = algo.compute_returns(t2, v2)
    assert r2.trained[:5].all() and not r2.trained[5] and not r2.held.any()
    full = ref_two_chain(
        "RCCRCCR",
        np.r_[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2],
        np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.05, 0.0]),
        LC,
        LR,
        ("cut", (0.9, "C")),
    )
    np.testing.assert_allclose(r2.adv[[0, 1, 4]], full[[0, 3, 6]], rtol=0, atol=1e-12)
    assert (r2.boot_w[[0, 1, 4]] > 0).all()
    hb.carry(t2, r2)
    assert len(hb) == 0


def test_hold_back_drop_round_trip_and_guards() -> None:
    hb = algo.HoldBack(4)
    tab = cat(
        run_table("RCR", [0, 0, 0], end="C", uid=1),
        run_table("RRC", [0, 0, 0], end="R", uid=2),
    )
    tab["n_classes"] = np.arange(8, dtype=np.uint16) + 300
    # Fixed-width 2-D columns ride along too: padded tokens, padded teacher probabilities.
    tab["tok"] = np.arange(32, dtype=np.int32).reshape(8, 4)
    tab["teacher"] = algo.pad_ragged(np.full(16, 0.5, np.float32), np.full(8, 2), width=3, fill=NAN)
    table = hb.join(tab)
    hb.carry(table, algo.compute_returns(table, np.full(8, 0.5)))
    assert sorted(hb.cols["run_uid"].tolist()) == [1, 1, 2, 2]
    np.testing.assert_array_equal(hb.cols["tok"][:, 0], [0, 8, 16, 20])  # rows 0, 2, 4, 5
    buf = io.BytesIO()
    torch.save(hb.to_dict(), buf)
    buf.seek(0)
    hb2 = algo.HoldBack.from_dict(torch.load(buf, weights_only=True))
    assert hb2.hold_iters == 4 and hb2.cols.keys() == hb.cols.keys()
    for k, x in hb.cols.items():
        assert hb2.cols[k].dtype == x.dtype and x.tobytes() == hb2.cols[k].tobytes(), k
    nxt = hb2.join({k: x[:1] for k, x in tab.items()})
    assert nxt["tok"].shape == (5, 4) and nxt["teacher"].shape == (5, 3)
    assert hb.drop([1, 99]) == 2 and hb.cols["run_uid"].tolist() == [2, 2]
    assert hb.drop([]) == 0 and algo.HoldBack().drop([1]) == 0
    with pytest.raises(ValueError, match="age"):
        hb.join({**run_table("C", [0]), "age": np.zeros(1, np.int64)})
    with pytest.raises(ValueError, match="differ"):
        hb.join(run_table("C", [0], uid=5))  # lacks the n_classes column of the held rows
    # Guard: a released run row may never follow a held one.
    bad = run_table("RRC", [0, 0, 0], end="C", uid=4)
    bad["age"] = np.array([0, 4, 0, 0])
    bad["carried"] = np.array([True, True, False, False])
    with pytest.raises(ValueError, match="released run row follows a held one"):
        algo.HoldBack().carry(bad, algo.compute_returns(bad, np.full(4, 0.5)))
    with pytest.raises(ValueError, match="waited"):
        hb.join({**run_table("C", [0]), "waited": np.zeros(1, np.int64)})


def _algo_step(
    st: algo.AlgoState, fresh: dict, values: np.ndarray
) -> tuple[dict, algo.Returns, np.ndarray]:
    """One iteration of the `AlgoState` protocol, minus the net: table, returns, advantages."""
    st.begin(int((~fresh["bootstrap"]).sum()), S.ANNEAL_START - 1e6)
    table = st.join(fresh)
    ret = st.returns(table, values[table["ref"]])
    norm = st.normalize(table, ret)
    st.end(table, ret)
    return table, ret, norm


def test_resume_drops_the_lost_runs_held_rows_and_the_overdue_stat_catches_a_leak() -> None:
    """A restart loses every open run, and a lost run never sends another row (segment uids
    carry the session). After `HoldBack.resume`, none of its held rows trains or lingers.
    Without it (the control), they stay dormant with a frozen age, and ``rows_overdue``
    counts them once they have waited more than hold_iters + 2 iterations."""
    s1 = Stream(16, 96, seed=21, run_len=(300, 700))
    st = algo.AlgoState()
    for _ in range(12):
        table, ret, _ = _algo_step(st, s1.next_table(), s1.values())
        assert ret.stats(table)["rows_overdue"] == 0
    blob = _snapshot(st)
    held = st.hold.cols
    old_uids = np.unique(held["run_uid"])
    n_held = len(held["run_uid"])
    assert n_held > 50 and len(old_uids) >= 8
    np.testing.assert_array_equal(held["waited"], held["age"])  # nothing was dormant
    assert set(held["waited"].tolist()) == {1, 2, 3, 4}
    resumed, ctl = _restore(blob), _restore(blob)
    assert len(resumed.hold) == n_held
    assert resumed.hold.resume() == n_held and len(resumed.hold) == 0  # all dropped, counted
    snap = resumed.to_dict()
    assert resumed.hold.resume() == 0 and _same(resumed.to_dict(), snap)  # idempotent
    # Dropping every held row's run instead drops the same rows and trains bit-identically.
    dropped = _restore(blob)
    assert dropped.hold.drop(old_uids.tolist()) == n_held and len(dropped.hold) == 0
    # The new session: new workers play new runs, whose uids are the new session's.
    s2 = Stream(16, 96, seed=22, run_len=(300, 700))
    s2._uid = 10_000_000
    off = len(s1.values())
    over = []
    for k in range(12):
        fresh = s2.next_table()
        fresh["ref"] = fresh["ref"] + off
        vals = np.concatenate([s1.values(), s2.values()])
        out_r = _algo_step(resumed, {c: x.copy() for c, x in fresh.items()}, vals)
        out_d = _algo_step(dropped, {c: x.copy() for c, x in fresh.items()}, vals)
        assert _same(out_d[0], out_r[0]) and _same(out_d[2], out_r[2])
        assert _same(dropped.to_dict(), resumed.to_dict())
        t_r, r_r, _ = out_r
        assert not np.isin(t_r["run_uid"], old_uids).any()  # none trains, none lingers
        assert r_r.stats(t_r)["rows_overdue"] == 0 and r_r.stats(t_r)["rows_dormant"] == 0
        t_c, r_c, _ = _algo_step(ctl, fresh, vals)
        old = np.isin(t_c["run_uid"], old_uids)
        assert int(old.sum()) == n_held and r_c.dormant[old].all() and not r_c.trained[old].any()
        assert sorted(t_c["age"][old].tolist()) == sorted(held["age"].tolist())  # frozen
        assert sorted(t_c["waited"][old].tolist()) == sorted((held["waited"] + k).tolist())
        assert not r_c.overdue[~old].any()
        assert r_c.stats(t_c)["rows_overdue"] == int((held["waited"] + k > 6).sum())
        over.append(r_c.stats(t_c)["rows_overdue"])
    assert over[0] == 0 and over[-1] == n_held
    assert not np.isin(resumed.hold.cols["run_uid"], old_uids).any()
    assert int(np.isin(ctl.hold.cols["run_uid"], old_uids).sum()) == n_held


def _planted(dormant: set[int], n_iter: int = 12) -> list[tuple[int, int, bool, bool]]:
    """The first row (a run row) of run 1, stored in iteration 0. In every other iteration
    its run sends a combat row and a cut, except in ``dormant`` ones; a live run ends in
    each. Per iteration the row is in the table: (age, waited, trained, overdue)."""
    hb = algo.HoldBack(algo.HOLD_ITERS)
    out = []
    t_next = 2  # the t of run 1's pending (bootstrap) row
    for k in range(n_iter):
        parts = [run_table("CC", [0.0, 1.0], uid=100 + k)]
        if k == 0:
            parts.append(run_table("RC", [0.0, 0.0], end="C", uid=1))
        elif k not in dormant:
            parts.append(run_table("C", [0.0], end="C", uid=1, t0=t_next))
            t_next += 1
        table = hb.join(cat(*parts))
        ret = algo.compute_returns(table, np.full(len(table["t"]), 0.5))
        assert ret.stats(table)["rows_overdue"] == int(ret.overdue.sum())
        j = np.flatnonzero((table["run_uid"] == 1) & (table["t"] == 0))
        if j.size:
            i = int(j[0])
            row = (int(table["age"][i]), int(table["waited"][i]), bool(ret.trained[i]))
            out.append((*row, bool(ret.overdue[i])))
        assert int(ret.overdue.sum()) == (out[-1][3] if j.size else 0)
        hb.carry(table, ret)
    return out


def test_overdue_is_zero_in_normal_operation_and_counts_a_planted_dormant_row() -> None:
    # Normal operation: the row trains at age 4, having waited 4 iterations.
    assert _planted(set()) == [(0, 0, False, False), (1, 1, False, False), (2, 2, False, False),
                               (3, 3, False, False), (4, 4, True, False)]  # fmt: skip
    # Dormant for 2 iterations: it trains at waited 6 = hold_iters + 2, not overdue yet.
    two = _planted({1, 2})
    assert [a for a, *_ in two] == [0, 1, 1, 1, 2, 3, 4] and two[-1] == (4, 6, True, False)
    # Dormant for 3: it trains at waited 7, counted in that iteration.
    assert _planted({1, 2, 3})[-1] == (4, 7, True, True)
    # A planted row whose run never sends another row: its age freezes at 1, it never trains,
    # and it is overdue from the iteration it has waited 7.
    planted = _planted(set(range(1, 12)))
    assert planted == [(min(k, 1), k, False, k > 6) for k in range(12)]
    # A stream without stalls never holds a row past hold_iters, practice segments included.
    st = Stream(12, 64, seed=23, run_len=(80, 260), practice_p=0.2)
    hold = algo.HoldBack(algo.HOLD_ITERS)
    for _ in range(40):
        table = hold.join(st.next_table())
        ret = algo.compute_returns(table, st.values()[table["ref"]])
        assert ret.stats(table)["rows_overdue"] == 0 and not ret.overdue.any()
        np.testing.assert_array_equal(table["waited"], table["age"])
        assert int(table["waited"].max()) <= algo.HOLD_ITERS
        hold.carry(table, ret)


@pytest.mark.parametrize(
    "case",
    ["dup", "boot_mid", "done_mid", "carried_combat", "open", "nan_value", "nan_reward", "shape"],
)
def test_compute_returns_rejects_malformed_tables(case: str) -> None:
    tab = run_table("RCCRC", [0.1] * 5, end="C")
    v = np.full(6, 0.5)
    if case == "dup":
        tab["t"][2] = 1
    elif case == "boot_mid":
        tab["bootstrap"][2] = True
    elif case == "done_mid":
        tab["done"][1] = True
    elif case == "carried_combat":
        tab["carried"] = np.zeros(6, bool)
        tab["carried"][1] = True
    elif case == "open":
        tab = run_table("RCC", [0.1] * 3, end=None)
        v = np.full(3, 0.5)
    elif case == "nan_value":
        v[3] = NAN
    elif case == "nan_reward":
        tab["reward"][2] = NAN
    elif case == "shape":
        v = v[:-1]
    with pytest.raises(ValueError):
        algo.compute_returns(tab, v)


def test_returns_stats_explained_variance_and_empty_tables() -> None:
    tab = cat(run_table("RCC", [0, 0, 1], uid=1), run_table("RC", [0, 0], end="C", uid=2))
    table = algo.HoldBack(4).join(tab)
    ret = algo.compute_returns(table, np.full(6, 0.5))
    s = ret.stats(table)
    assert s["rows_trained"] == 4 and s["rows_trained_combat"] == 3 and s["rows_trained_run"] == 1
    assert s["rows_held"] == 1 and s["rows_pi"] == 4 and s["rows_dormant"] == 0
    assert s["boot_w_combat"] == pytest.approx(1 / 3)  # only row 4 bootstraps (from row 5)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert algo.explained_variance(y, y, np.ones(4, bool)) == pytest.approx(1.0)
    assert algo.explained_variance(y, np.full(4, 2.5), np.ones(4, bool)) == pytest.approx(0.0)
    assert math.isnan(algo.explained_variance(y, y, np.array([True, False, False, False])))
    empty = algo.compute_returns(run_table("", [], end=None), np.zeros(0))
    assert empty.adv.shape == (0,) and empty.trained.dtype == bool


def test_table_from_and_pad_ragged(monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(algo.COLUMNS) == {
        "run_uid", "t", "is_combat", "dtype", "reward", "done", "bootstrap", "actor",
        "logp_behav", "n_classes", "action", "iter", "v_behav", "cell",
    }  # fmt: skip
    out = algo.table_from({"run_uid": [1], "t": [0], "extra": [5]})
    assert set(out) == {"run_uid", "t", "extra"} and isinstance(out["t"], np.ndarray)
    monkeypatch.setitem(algo.COLUMNS, "dtype", "decision_type")  # a schema rename: one line
    assert list(algo.table_from({"decision_type": np.array([3], np.uint8)})) == ["dtype"]
    flat = np.array([0.25, 0.75, 0.1, 0.2, 0.7, NAN, NAN], np.float32)
    lengths = np.array([2, 0, 3, 2])
    got = algo.pad_ragged(flat, lengths, width=4, fill=-1.0)
    want = np.array(
        [[0.25, 0.75, -1, -1], [-1, -1, -1, -1], [0.1, 0.2, 0.7, -1], [NAN, NAN, -1, -1]],
        np.float32,
    )
    np.testing.assert_array_equal(got, want)
    assert algo.pad_ragged(flat, lengths).shape == (4, 3)
    assert algo.pad_ragged(np.zeros(0), np.zeros(0, np.int64)).shape == (0, 0)
    with pytest.raises(ValueError):
        algo.pad_ragged(flat, [2, 2], width=4)
    with pytest.raises(ValueError):
        algo.pad_ragged(flat, lengths, width=2)


# ── T7: n_classes ≥ 2, entropy in [0, 1] ─────────────────────────────────


def test_t7_entropy_is_normalised_over_merged_classes() -> None:
    b, a = 40, 9
    rows = _loss_rows(b, a, seed=7, pi_frac=1.0, teacher=False, ks=0.0, beta=1.0)
    g = torch.Generator().manual_seed(7)
    cm = algo.class_mask(rows.n_classes, a)
    logits = torch.where(cm, 4 * torch.randn(b, a, generator=g), torch.full((b, a), 1e6))
    for i in range(b):
        st = algo.ppo_losses(logits[i : i + 1], torch.zeros(1), rows.index(slice(i, i + 1))).stats
        n = int(rows.n_classes[i])
        p = torch.softmax(logits[i, :n].double(), 0)
        h = float(-(p * p.log()).sum())
        assert 0.0 <= float(st["ent_n"]) <= 1.0
        assert float(st["ent_n"]) == pytest.approx(h / math.log(n), abs=1e-5)
        assert float(st["ent"]) == pytest.approx(h, abs=1e-5)
        assert float(st["ent_bonus"]) == pytest.approx(h / math.log(n), abs=1e-5)  # β = 1
    rows3 = _loss_rows(3, a, seed=8, pi_frac=1.0, teacher=False, ks=0.0, beta=1.0)
    rows3.n_classes[:] = torch.tensor([2, 5, 9])
    rows3.action[:] = 0
    flat = algo.ppo_losses(torch.zeros(3, a), torch.zeros(3), rows3).stats
    assert float(flat["ent_n"]) == pytest.approx(3.0, abs=1e-6)  # uniform: 1 per row
    peaked = torch.zeros(3, a)
    peaked[:, 0] = 50.0
    assert float(algo.ppo_losses(peaked, torch.zeros(3), rows3).stats["ent_n"]) < 1e-6


@pytest.mark.parametrize(("nc", "act"), [(1, 0), (0, 0), (3, 3), (3, -1), (12, 0)])
def test_t7_used_rows_need_two_classes_and_a_valid_action(nc: int, act: int) -> None:
    rows = _loss_rows(4, 9, seed=1, teacher=False)
    rows.n_classes[2] = nc
    rows.action[2] = act
    with pytest.raises(ValueError, match="row 2"):
        algo.ppo_losses(torch.zeros(4, 9), torch.zeros(4), rows)
    rows.v[2] = False  # the same row outside the trained rows is fine
    out = algo.ppo_losses(torch.zeros(4, 9), torch.zeros(4), rows)
    assert torch.isfinite(out.loss)


def _entropy_sums(logits: torch.Tensor, nc: np.ndarray, dt: np.ndarray, mask: np.ndarray, k: int):
    sums, cnt = np.zeros(k), np.zeros(k)
    for i in np.flatnonzero(mask):
        n = int(nc[i])
        p = torch.softmax(logits[i, :n].double(), 0)
        sums[dt[i]] += float(-(p * p.log()).sum()) / math.log(n)
        cnt[dt[i]] += 1
    return sums, cnt


def test_t7_stats_entropy_is_the_per_type_normalised_entropy() -> None:
    """T7 on the stats path: stats.jsonl's per-type `entropy` comes from
    `update.normalized_entropy_by_type`, which (unlike `ppo_losses`) does not clamp.
    Per type it sums H / log(n_classes) over the masked rows' merged classes only; the
    per-type mean is in [0, 1], and rows outside the mask are inert, NaN ones too."""
    from ppo.e2e import update as U

    g = torch.Generator().manual_seed(17)
    b, a, k = 64, 9, 6
    nc = torch.randint(2, a + 1, (b,), generator=g)
    cm = torch.arange(a)[None, :] < nc[:, None]
    # Padded classes get a huge logit: if they leaked in, the entropy would be wrong.
    logits = torch.where(cm, 3 * torch.randn(b, a, generator=g), torch.full((b, a), 1e6))
    dt = torch.randint(0, k - 1, (b,), generator=g).numpy()  # type k - 1 unused
    mask = (torch.rand(b, generator=g) < 0.7).numpy()
    got = U.normalized_entropy_by_type(logits, nc, dt, mask, k).double().numpy()
    want, cnt = _entropy_sums(logits, nc.numpy(), dt, mask, k)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)
    mean = np.where(cnt > 0, got / np.maximum(cnt, 1), np.nan)
    assert np.isnan(mean[k - 1])
    seen = mean[cnt > 0]
    assert ((seen >= 0.0) & (seen <= 1.0)).all(), seen
    lg2 = logits.clone()
    lg2[~torch.from_numpy(mask)] = math.nan
    got2 = U.normalized_entropy_by_type(lg2, nc, dt, mask, k).double().numpy()
    np.testing.assert_array_equal(got2, got)


@pytest.mark.parametrize("n", [2, 3, 7, 9, 64])
def test_t7_stats_entropy_reaches_both_ends(n: int) -> None:
    from ppo.e2e import update as U

    nc = torch.tensor([n, n])
    dt = np.array([0, 1])
    mask = np.array([True, True])
    flat = torch.zeros(2, 64)
    peaked = flat.clone()
    peaked[:, 0] = 50.0
    uni = U.normalized_entropy_by_type(flat, nc, dt, mask, 2).double().numpy()
    pk = U.normalized_entropy_by_type(peaked, nc, dt, mask, 2).double().numpy()
    # Uniform over n merged classes is 1 (up to fp32), never meaningfully above it.
    assert np.all(np.abs(uni - 1.0) <= 1e-6), uni
    assert np.all(pk >= 0.0) and np.all(pk < 1e-6), pk


# ── T9: KL on student rows, no stop in U0 ────────────────────────────────


def test_t9_kl_and_statistics_use_student_rows_only() -> None:
    b, a = 24, 7
    rows = _loss_rows(b, a, seed=9, pi_frac=0.5, ks=0.4)
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(b, a, generator=g)
    out = algo.ppo_losses(logits, torch.zeros(b), rows)
    assert int(out.stats["n_pi"]) == int(rows.pi.sum()) > 0
    assert int(out.stats["n_v"]) == b and int(out.stats["n_ks"]) == b
    # New logits on the teacher-acted rows: every PPO statistic is unchanged, the teacher term moves.
    l2 = logits.clone()
    l2[~rows.pi] = torch.randn(int((~rows.pi).sum()), a, generator=g)
    out2 = algo.ppo_losses(l2, torch.zeros(b), rows)
    for k in ("n_pi", "ploss", "ent", "ent_n", "ent_bonus", "kl", "clipfrac", "dualfrac", "isw"):
        assert torch.equal(out2.stats[k], out.stats[k]), k
    for k in ("is_clamped", "n_pi_t", "ent_t", "kl_t"):
        assert torch.equal(out2.stats[k], out.stats[k]), k
    assert not torch.equal(out2.stats["ks_loss"], out.stats["ks_loss"])
    # Without the teacher term, no gradient reaches the teacher-acted rows' logits.
    lg = logits.clone().requires_grad_()
    rows0 = dataclasses.replace(rows, ks=torch.zeros(b))
    algo.ppo_losses(lg, torch.zeros(b), rows0).loss.backward()
    assert (lg.grad[~rows.pi] == 0).all() and (lg.grad[rows.pi] != 0).any()
    # The probe KL: exact KL(π_prox ‖ π_θ) over merged classes, on the probe's student rows.
    prox = torch.randn(b, a, generator=g)
    s, n = algo.probe_kl(logits, prox, rows.n_classes, rows.pi)
    s2, n2 = algo.probe_kl(l2, prox, rows.n_classes, rows.pi)
    assert torch.equal(s, s2) and torch.equal(n, n2) and int(n) == int(rows.pi.sum())
    want = 0.0
    for i in torch.nonzero(rows.pi).flatten().tolist():
        k = int(rows.n_classes[i])
        lo = torch.log_softmax(prox[i, :k].double(), 0)
        ln = torch.log_softmax(logits[i, :k].double(), 0)
        want += float((lo.exp() * (lo - ln)).sum())
    assert float(s) == pytest.approx(want, rel=1e-5)
    z, _ = algo.probe_kl(prox, prox, rows.n_classes, rows.pi)
    assert abs(float(z)) < 1e-6


def test_t9_no_kl_stop_and_no_factor_change_in_u0() -> None:
    st = algo.AlgoState()
    st.clocks.it = 5
    t = S.ANNEAL_START - 10 * 3600.0

    def one(uid: int, probe: float) -> dict:
        table = st.join(run_table("CCC", [0, 0, 1.0], uid=uid))
        ret = st.returns(table, np.full(3, 0.5))
        st.normalize(table, ret)
        return st.end(table, ret, probe_kl=probe)

    s = st.begin(1_000, t)
    assert s.phase == "U0" and s.policy_on and not st.kl.stop(5.0, s, st.cfg)
    out = one(1, 5.0)
    assert _exact(out["kl_m"], 1.0) and _exact(out["kl_m_before"], 1.0)
    st.clocks.decisions = 200 * M
    s = st.begin(1_000, t)
    assert s.phase == "U1" and st.kl.stop(5.0, s, st.cfg)
    assert one(2, 5.0)["kl_m"] == pytest.approx(1 / 1.5)


def test_t9_advantage_moments_per_type_from_masked_rows_only() -> None:
    """The EW moments port `RunningMoments` (``min_rows = 1``: the first batch sets them)."""
    rng = np.random.default_rng(9)
    an = algo.AdvNorm(n_types=6, decay=0.01, min_rows=1)
    refs = [RefMoments(0.01) for _ in range(6)]
    for it in range(6):
        n = 200
        adv = rng.normal(it, 1 + it, n)
        dt = rng.integers(0, 5, n)  # type 5 never appears
        pi = rng.random(n) < 0.6
        adv[~pi] = NAN  # rows outside the mask may hold anything
        out = an.update_normalize(adv, dt, pi)
        for d in range(6):
            refs[d].update(adv[pi & (dt == d)])
        np.testing.assert_allclose(an.mean, [r.mean for r in refs], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(an.std(), [r.std for r in refs], rtol=1e-12)
        want = (adv[pi] - an.mean[dt[pi]]) / an.std()[dt[pi]]
        np.testing.assert_allclose(out[pi], want, rtol=1e-12)
        assert np.isnan(out[~pi]).all()
    assert not an.inited[5] and _exact(an.mean[5], 0.0)
    an2 = algo.AdvNorm.from_dict(an.to_dict())
    assert an2.min_rows == 1
    for f in ("mean", "sq", "count", "inited"):
        np.testing.assert_array_equal(getattr(an2, f), getattr(an, f))
    with pytest.raises(ValueError):
        an.update(np.array([NAN]), np.array([0]))
    with pytest.raises(ValueError):
        an.update(np.array([1.0]), np.array([6]))


class RefWarm:
    """§6 Advantages for one type, from scratch: the plain means of its first ``min_rows``
    rows (the batch that reaches them counts whole), then `RefMoments`' EW update."""

    def __init__(self, decay: float = 5e-4, min_rows: int = 64) -> None:
        self.ew, self.first, self.min_rows = RefMoments(decay), [], min_rows

    @property
    def warm(self) -> bool:
        return len(self.first) >= self.min_rows

    def update(self, xs: np.ndarray) -> None:
        if xs.size and not self.warm:
            self.first += xs.tolist()
            r = np.array(self.first)
            self.ew.mean, self.ew.sq, self.ew.inited = float(r.mean()), float((r * r).mean()), True
        else:
            self.ew.update(xs)


def _pool(x: np.ndarray) -> tuple[float, float]:
    m = float(x.mean())
    return m, math.sqrt(max(1e-6, float((x * x).mean()) - m * m))


def ref_normalize(refs: list[RefWarm], a, d, comb, min_rows: int = 64):
    """One batch: update every type, then normalise each row by its type's own moments if
    warm, else its family's pool if that has min_rows rows, else the whole batch's pool.
    Also returns each row's source: 0 own, 1 family, 2 whole batch."""
    for t in np.unique(d).tolist():
        refs[t].update(a[d == t])
    out, src = np.empty(len(a)), np.empty(len(a), np.int64)
    for i in range(len(a)):
        fam = comb == comb[i]
        if refs[d[i]].warm:
            (m, s), src[i] = (refs[d[i]].ew.mean, refs[d[i]].ew.std), 0
        elif fam.sum() >= min_rows:
            (m, s), src[i] = _pool(a[fam]), 1
        else:
            (m, s), src[i] = _pool(a), 2
        out[i] = (a[i] - m) / s
    return out, src


def _check(out: np.ndarray, a: np.ndarray, rows: np.ndarray, m_s: tuple[float, float]) -> None:
    np.testing.assert_allclose(out[rows], (a[rows] - m_s[0]) / m_s[1], rtol=1e-12, atol=0)


def test_advnorm_a_rare_type_borrows_its_familys_moments_until_it_has_64_rows() -> None:
    """Both regimes and the hand-over, by hand. Combat types 0 (100 rows a batch) and 1
    (10); out-of-combat types 5 (100, but 40 in batch 2) and 6 (3, but 5 in batch 2)."""
    rng = np.random.default_rng(31)
    an = algo.AdvNorm(n_types=8)
    assert an.min_rows == algo.ADV_MIN_ROWS == 64 and _exact(an.decay, 5e-4)
    seen1: list[float] = []
    for b in range(1, 9):
        sizes = {0: 100, 1: 10, 5: 40 if b == 2 else 100, 6: 5 if b == 2 else 3}
        d = np.concatenate([np.full(n, t) for t, n in sizes.items()])
        comb = d < 5
        a = np.where(comb, rng.normal(0.2, 1.0, len(d)), rng.normal(-0.5, 3.0, len(d)))
        a[d == 1] = rng.normal(0.0, 0.1, sizes[1])  # type 1's own scale differs from its family's
        before = (an.mean[1], an.sq[1])
        out = an.update_normalize(a, d, np.ones(len(d), bool), comb)
        seen1 += a[d == 1].tolist()
        # Types 0 and 5 have 64 rows from batch 1 on: their own moments (batch 1 sets them).
        _check(out, a, d == 0, (an.mean[0], an.std()[0]))
        _check(out, a, d == 5, (an.mean[5], an.std()[5]))
        # Type 6 never reaches 64 rows: its family's pool, or in batch 2 (40 + 5
        # out-of-combat rows) the whole batch's.
        _check(out, a, d == 6, _pool(a) if b == 2 else _pool(a[~comb]))
        if b <= 6:  # type 1 has seen 10b < 64 rows: the combat pool
            assert an.count[1] == 10 * b
            _check(out, a, d == 1, _pool(a[comb]))
        else:  # the hand-over at 70 rows: its own moments
            _check(out, a, d == 1, (an.mean[1], an.std()[1]))
        if b == 7:  # its first 70 rows set them, plain means (batch 7 counts whole)
            r = np.array(seen1)
            assert an.mean[1] == pytest.approx(float(r.mean()), rel=1e-12, abs=1e-15)
            assert an.sq[1] == pytest.approx(float((r * r).mean()), rel=1e-12)
            assert an.std()[1] == pytest.approx(float(r.std()), rel=1e-9)
        if b == 8:  # then the EW update: (1 - decay)^10 of the old moments stays
            keep, x = (1 - 5e-4) ** 10, a[d == 1]
            assert an.mean[1] == pytest.approx(keep * before[0] + (1 - keep) * x.mean(), rel=1e-12)
            assert an.sq[1] == pytest.approx(keep * before[1] + (1 - keep) * (x * x).mean())
    assert an.count.tolist() == [800, 80, 0, 0, 0, 740, 26, 0]
    # The review's case: a type's first batch is one row (0.3), its next one row (-0.2).
    # The old rule (min_rows 1) set its moments from that row: -0.2 became -44.7.
    old = algo.AdvNorm(n_types=4, min_rows=1)
    old.update(np.array([0.3]), np.array([2]))
    assert old.update_normalize(np.array([-0.2]), np.array([2]), np.array([True]))[0] < -40
    new = algo.AdvNorm(n_types=4)
    base = rng.normal(0.0, 0.3, 200)  # its family: 200 rows of a common type
    for x in (0.3, -0.2):
        a, d = np.r_[base, x], np.r_[np.zeros(200, np.int64), 2]
        out = new.update_normalize(a, d, np.ones(201, bool), np.ones(201, bool))
        assert out[-1] == pytest.approx((x - _pool(a)[0]) / _pool(a)[1], rel=1e-12)
        assert abs(out[-1]) < 3
    # The 1e-3 floor holds in every pool: one row, or equal rows, normalise to 0.
    one = algo.AdvNorm(n_types=2)
    assert _exact(one.update_normalize(np.array([0.3]), np.array([0]), np.array([True]))[0], 0.0)
    flat = one.normalize(np.full(5, 0.75), np.ones(5, np.int64), np.ones(5, bool))
    assert np.array_equal(flat, np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        one.normalize(np.zeros(3), np.zeros(3, np.int64), np.zeros(2, bool))
    with pytest.raises(ValueError, match="min_rows"):
        algo.AdvNorm(min_rows=0)


def test_advnorm_follows_the_reference_through_algo_state_and_resumes_bit_exactly() -> None:
    """`AlgoState.normalize` against `ref_normalize` over a stream, each row's family its
    ``is_combat``. Combat types are warm at once, run types after several iterations, and
    all three sources occur. A state restored mid warm-up continues bit for bit."""
    st = Stream(12, 128, seed=32, run_len=(100, 300))
    tables = [st.next_table() for _ in range(24)]
    vals = st.values()
    a1 = algo.AlgoState()
    refs = [RefWarm() for _ in range(algo.N_TYPES)]
    srcs: collections.Counter = collections.Counter()
    first: dict[tuple[int, bool], int] = {}  # (type, own moments) -> first iteration
    norms, blob = [], b""
    for k, fresh in enumerate(tables):
        table, ret, norm = _algo_step(a1, fresh, vals)
        m = ret.pi
        d = table["dtype"][m].astype(np.int64)
        want, src = ref_normalize(refs, ret.adv[m], d, table["is_combat"][m])
        np.testing.assert_allclose(norm[m], want, rtol=1e-9, atol=1e-12)
        assert np.isnan(norm[~m]).all()
        srcs.update(src.tolist())
        for t, s in zip(d.tolist(), src.tolist()):
            first.setdefault((t, s == 0), k)
        norms.append(norm)
        if k == 4:
            blob = _snapshot(a1)
    assert min(srcs[0], srcs[1], srcs[2]) > 0, srcs
    handed = [t for (t, own), k in first.items() if own and first.get((t, False), k) < k]
    assert len(handed) >= 5 and all(a1.adv.count[t] >= 64 for t in handed), handed
    # A resume after iteration 4, mid warm-up: the next iterations, bit for bit.
    a2 = _restore(blob)
    assert ((a2.adv.count > 0) & (a2.adv.count < 64)).sum() >= 5
    for k in range(5, 9):
        assert _same(_algo_step(a2, tables[k], vals)[2], norms[k]), k


# ── the loss terms ───────────────────────────────────────────────────────


def test_decoupled_ppo_with_is_clamp_and_dual_clip() -> None:
    b, a = 40, 5
    rows = _loss_rows(b, a, seed=6, pi_frac=1.0, teacher=False, ks=0.0, beta=0.0)
    g = torch.Generator().manual_seed(6)
    logits = 2 * torch.randn(b, a, generator=g)
    lp = algo.logp_of(logits, rows.n_classes, rows.action)
    rows.lp_prox = lp - torch.linspace(-1.5, 1.5, b)  # ratios from e^-1.5 to e^1.5
    rows.lp_behav = rows.lp_prox - torch.linspace(-2.0, 2.0, b).flip(0)  # IS weights e^±2
    rows.adv = torch.linspace(1, 2, b) * torch.tensor([1.0, -1.0]).repeat(b // 2)
    out = algo.ppo_losses(logits, torch.zeros(b), rows)
    ratio = (lp - rows.lp_prox).exp()
    w = (rows.lp_prox - rows.lp_behav).exp().clamp(max=2.0)
    adv = rows.adv
    surr = torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
    surr = torch.where(adv < 0, torch.maximum(surr, 3.0 * adv), surr)
    want = float(-(w * surr).sum())
    assert float(out.stats["ploss"]) == pytest.approx(want, rel=1e-5)
    vloss = 0.5 * float((rows.target**2).sum())
    assert float(out.loss) == pytest.approx((vloss + want) / b, rel=1e-5)
    assert float(out.stats["dualfrac"]) > 0 and float(out.stats["is_clamped"]) > 0
    clipped = ((ratio - 1).abs() > 0.2).sum()
    assert int(out.stats["clipfrac"]) == int(clipped) > 0
    assert float(out.stats["isw"]) == pytest.approx(float(w.sum()), rel=1e-5)


def test_decoupled_ppo_matches_hand_derived_constants() -> None:
    """Four 2-class rows, derived by hand at clip 0.2, IS clamp 2 and dual clip 3. The
    taken class has π_θ = 0.8, 0.6, 0.3, 0.6; r = π_θ / π_prox; w = min(w_raw, 2):

    ===  ===  =====  ==  =================================  ======  ===================
    row  r    w_raw  A   surrogate                          -w·sur  d loss / d z_taken
    ===  ===  =====  ==  =================================  ======  ===================
    1    4    3      -1  min(-4, -1.2) = -4 < 3A: dual, -3  6       0 (the dual is flat)
    2    2    0.5    -1  min(-2, -1.2) = -2 > 3A            1       ¼·(0.5·2)·0.4 = 0.1
    3    0.5  1      2   min(0.5·2, 0.8·2) = 1              -1      ¼·(-2·0.5)·0.7 = -0.175
    4    1.5  1.5    1   min(1.5, 1.2) = 1.2: clipped       -1.8    0 (the clip is flat)
    ===  ===  =====  ==  =================================  ======  ===================

    ploss 6 + 1 - 1 - 1.8 = 4.2; loss 4.2 / 4 = 1.05 (no value, entropy or teacher term).
    The dual clip and the IS clamp fire on row 1 only. A gradient row is
    ¼ · d(-w·surr)/d log π · (1 - π_θ): the unclipped surrogate's d/d log π is r·A.
    """
    p = torch.tensor([0.8, 0.6, 0.3, 0.6], dtype=torch.float64)
    r = torch.tensor([4.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    w_raw = torch.tensor([3.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    lp_prox = torch.log(p / r)
    rows = algo.LossRows(
        n_classes=torch.full((4,), 2),
        action=torch.zeros(4, dtype=torch.int64),
        lp_prox=lp_prox.float(),
        lp_behav=(lp_prox - torch.log(w_raw)).float(),
        adv=torch.tensor([-1.0, -1.0, 2.0, 1.0]),
        target=torch.zeros(4),
        pi=torch.ones(4, dtype=torch.bool),
        v=torch.ones(4, dtype=torch.bool),
        beta=torch.zeros(4),
        ks=torch.zeros(4),
        dtype=torch.zeros(4, dtype=torch.int64),
    )
    logits = torch.log(torch.stack([p, 1 - p], 1)).float().requires_grad_()
    out = algo.ppo_losses(logits, torch.zeros(4), rows)
    assert float(out.stats["ploss"]) == pytest.approx(4.2, abs=1e-5)
    assert out.loss.item() == pytest.approx(1.05, abs=1e-6)
    assert _exact(float(out.stats["dualfrac"]), 1.0) and _exact(float(out.stats["is_clamped"]), 1.0)
    assert _exact(float(out.stats["clipfrac"]), 4.0)
    assert float(out.stats["isw"]) == pytest.approx(5.0, abs=1e-5)
    out.loss.backward()
    want = torch.tensor([0.0, 0.1, -0.175, 0.0])
    torch.testing.assert_close(logits.grad[:, 0], want, rtol=0, atol=1e-6)
    torch.testing.assert_close(logits.grad[:, 1], -want, rtol=0, atol=1e-6)
    # Row by row: the loss of each row, and which one fires the dual clip and the IS clamp.
    for i, pl in enumerate([6.0, 1.0, -1.0, -1.8]):
        one = rows.index(slice(i, i + 1))
        st = algo.ppo_losses(logits[i : i + 1].detach(), torch.zeros(1), one).stats
        assert float(st["ploss"]) == pytest.approx(pl, abs=1e-5), i
        fired = 1.0 if i == 0 else 0.0
        assert _exact(float(st["dualfrac"]), fired) and _exact(float(st["is_clamped"]), fired), i


def test_kickstarting_term_is_lambda_kl_with_the_cross_entropy_gradient() -> None:
    """λ_KS · KL(π_T ‖ π_S) on trained rows with a teacher; its gradient is the U0 cross-entropy's."""
    b, a = 12, 6
    rows = _loss_rows(b, a, seed=4, pi_frac=0.5, ks=0.7, beta=0.0)
    rows.ks[0] = 0.0  # a teacherless type (`IterSchedule.ks_rows`)
    rows.teacher[1] = NAN  # no teacher output on this row
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(b, a, generator=g).requires_grad_()
    out = algo.ppo_losses(logits, torch.zeros(b), rows)
    off = algo.ppo_losses(logits, torch.zeros(b), dataclasses.replace(rows, ks=torch.zeros(b)))
    lp = algo.masked_log_softmax(logits, rows.n_classes)
    ok = torch.arange(b) >= 2
    pt = torch.nan_to_num(rows.teacher)
    kl = (torch.xlogy(pt, pt) - pt * lp).sum(-1)
    ce = -(pt * lp).sum(-1)
    want_kl = (0.7 * kl * ok).sum() / b
    torch.testing.assert_close(out.loss - off.loss, want_kl)
    assert int(out.stats["n_ks"]) == b - 2 and int(out.stats["n_ks_missing"]) == 1
    g_kl = torch.autograd.grad(out.loss - off.loss, logits, retain_graph=True)[0]
    g_ce = torch.autograd.grad((0.7 * ce * ok).sum() / b, logits)[0]
    torch.testing.assert_close(g_kl, g_ce)
    assert float(out.stats["ks_ce"] - out.stats["ks_kl"]) > 0  # the teacher's own entropy


def test_loss_rows_from_numpy_columns() -> None:
    """The learner's path: table columns, `Returns` masks and schedule weights to `LossRows`."""
    table = algo.HoldBack().join(run_table("RCCC", [0, 0, 0, 1.0], actor=[2, 0, 1, 0]))
    table["n_classes"] = np.array([3, 2, 4, 2], np.uint16)
    table["action"] = np.array([2, 1, 0, 1], np.uint16)
    table["logp_behav"] = np.log(np.array([0.2, 0.5, 0.3, 0.9], np.float32))
    ret = algo.compute_returns(table, np.full(4, 0.5), hold_iters=0)
    st = algo.AlgoState(hold_iters=0)
    s = st.begin(4, S.ANNEAL_START - 1e6)
    adv = st.normalize(table, ret)
    teacher = algo.pad_ragged(np.full(11, 0.25, np.float32), table["n_classes"], width=4)
    rows = algo.LossRows.from_numpy(
        n_classes=table["n_classes"],
        action=table["action"],
        lp_prox=table["logp_behav"],  # stands in for the pre-pass
        lp_behav=table["logp_behav"],
        adv=adv,
        target=ret.target,
        pi=ret.pi,
        v=ret.trained,
        beta=s.beta_rows(table["dtype"], table["is_combat"]),
        ks=s.ks_rows(table["dtype"], np.ones(4, bool)),
        dtype=table["dtype"],
        teacher=teacher,
    )
    assert rows.n_classes.dtype == torch.int64 and rows.pi.dtype == torch.bool
    assert rows.adv.dtype == torch.float32 and rows.teacher.shape == (4, 4)
    assert rows.to("cpu").index(torch.tensor([1, 3])).action.tolist() == [1, 1]
    out = algo.ppo_losses(torch.zeros(4, 4), torch.zeros(4), rows)
    assert int(out.stats["n_pi"]) == 2 and int(out.stats["n_v"]) == 4
    assert algo.LossRows.from_numpy(**{**rows.__dict__, "teacher": None}).teacher is None
    with pytest.raises(KeyError, match="missing loss column"):
        algo.LossRows.from_numpy(n_classes=table["n_classes"])


def test_stats_means_divide_by_their_row_counts() -> None:
    rows = _loss_rows(30, 6, seed=12, pi_frac=0.5)
    logits = torch.randn(30, 6, generator=torch.Generator().manual_seed(12))
    full = algo.ppo_losses(logits, torch.zeros(30), rows, n_types=8)
    parts = [
        algo.ppo_losses(logits[i], torch.zeros(len(i)), rows.index(i), n_types=8)
        for i in torch.arange(30).chunk(3)
    ]
    summed = {k: sum(p.stats[k] for p in parts) for k in full.stats}
    m = algo.stats_means(summed)
    mf = algo.stats_means(full.stats)
    for k in ("vloss", "ploss", "ent_n", "kl", "ks_kl", "isw"):
        assert m[k] == pytest.approx(mf[k], rel=1e-5), k
    assert m["n_pi"] == int(rows.pi.sum()) and m["n_v"] == 30
    assert m["vloss"] == pytest.approx(float(full.stats["vloss"]) / 30, rel=1e-6)
    seen = m["n_pi_t"] > 0
    assert np.isnan(m["ent_t"][~seen]).all() and np.isfinite(m["ent_t"][seen]).all()


def test_entropy_by_type_sums_raw_entropy() -> None:
    rows = _loss_rows(20, 5, seed=13)
    logits = torch.randn(20, 5, generator=torch.Generator().manual_seed(13))
    s, n = algo.entropy_by_type(logits, rows.n_classes, rows.dtype, rows.pi, n_types=8)
    for d in range(8):
        idx = [i for i in range(20) if bool(rows.pi[i]) and int(rows.dtype[i]) == d]
        h = 0.0
        for i in idx:
            p = torch.softmax(logits[i, : int(rows.n_classes[i])].double(), 0)
            h -= float((p * p.log()).sum())
        assert int(n[d]) == len(idx) and float(s[d]) == pytest.approx(h, abs=1e-5)


# ── T10: no value clip, one optimizer step per minibatch ─────────────────


def test_t10_value_loss_is_not_clipped() -> None:
    rows = _loss_rows(6, 4, seed=2, pi_frac=0.0, teacher=False, ks=0.0, beta=0.0)
    rows.target = torch.tensor([0.0, 1.0, -50.0, 100.0, 0.5, 0.0])
    value = torch.tensor([0.0, 0.0, 50.0, -100.0, 0.5, 3.0], requires_grad=True)
    out = algo.ppo_losses(None, value, rows, policy_on=False)
    want = 0.5 * float(((value.detach() - rows.target) ** 2).sum()) / 6
    assert out.loss.item() == pytest.approx(want, rel=1e-6)
    out.loss.backward()
    torch.testing.assert_close(value.grad, (value - rows.target).detach() / 6)
    on = algo.ppo_losses(torch.zeros(6, 4), value, rows)  # policy terms on, no student rows
    assert on.loss.item() == pytest.approx(want, rel=1e-6)
    with pytest.raises(ValueError, match="logits"):
        algo.ppo_losses(None, value, rows)


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 6)
        self.trunk = torch.nn.Linear(6, 8)
        self.policy = torch.nn.Linear(8, 5)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
        )
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)

    def forward(self, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.trunk(self.emb(tok).mean(1)))
        return self.policy(h), self.value(h).squeeze(-1)


def test_t10_param_groups_partition_the_trainable_parameters() -> None:
    net = _Net()
    net.value.add_module("tied", torch.nn.Linear(6, 10, bias=False))
    net.value.tied.weight = net.emb.weight  # shared with the trunk's ID table
    groups = algo.param_groups(net)
    assert [g["name"] for g in groups] == ["trunk", "value"]
    assert [g["base_lr"] for g in groups] == [1e-4, 3e-4]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in net.parameters() if p.requires_grad}
    name = {id(p): n for n, p in net.named_parameters()}
    for g in groups:
        for p in g["params"]:
            assert name[id(p)].startswith("value.") == (g["name"] == "value"), name[id(p)]
    assert id(net.emb.weight) in {id(p) for p in groups[0]["params"]}
    with pytest.raises(ValueError):
        algo.param_groups(net, ("nothing.",))
    opt = algo.make_optimizer(net)
    assert all(g["weight_decay"] == pytest.approx(1e-4) for g in opt.param_groups)
    applied = algo.set_lr(opt, 0.25)
    assert applied == {"trunk": pytest.approx(2.5e-5), "value": pytest.approx(7.5e-5)}
    assert [g["lr"] for g in opt.param_groups] == [applied["trunk"], applied["value"]]


def _adamw_first_step(p, g, lr, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
    p = p.double() * (1 - lr * wd)
    m_hat = (1 - b1) * g.double() / (1 - b1)
    v_hat = (1 - b2) * g.double() ** 2 / (1 - b2)
    return (p - lr * m_hat / (v_hat.sqrt() + eps)).float()


def test_t10_one_clipped_adamw_step_per_minibatch() -> None:
    """3 ranks x micro-chunks of one minibatch: gradients sum, then one global clip and one step."""
    torch.manual_seed(10)
    net = _Net()
    ref = copy.deepcopy(net)
    opt = algo.make_optimizer(net)
    frac = 0.37
    algo.set_lr(opt, frac)
    b = 48
    tok = torch.randint(0, 10, (b, 3))
    rows = _loss_rows(b, 5, seed=10, pi_frac=0.6, ks=0.5)
    rows.v[::7] = False  # untrained rows ride along, inert
    denom = float(rows.v.sum())
    ranks = torch.randperm(b).chunk(3)
    for rank in ranks:
        for chunk in rank.chunk(2):
            logits, value = net(tok[chunk])
            algo.ppo_losses(logits, value, rows.index(chunk), denom=denom).loss.backward()
    got_grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    gn = algo.clip_and_step(opt, max_norm=0.05)
    logits, value = ref(tok)
    algo.ppo_losses(logits, value, rows).loss.backward()
    ref_grads = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
    assert got_grads.keys() == ref_grads.keys() and "frozen.weight" not in ref_grads
    for n, g in ref_grads.items():
        torch.testing.assert_close(got_grads[n], g, rtol=1e-5, atol=1e-7, msg=n)
    total = math.sqrt(sum(float((g.double() ** 2).sum()) for g in ref_grads.values()))
    assert float(gn) == pytest.approx(total, rel=1e-5) and total > 0.05  # the clip is active
    scale = 0.05 / (total + 1e-6)
    params = dict(net.named_parameters())
    for n, p in ref.named_parameters():
        if p.grad is None:
            continue
        lr = (3e-4 if n.startswith("value.") else 1e-4) * frac
        want = _adamw_first_step(p.detach(), p.grad * scale, lr)
        torch.testing.assert_close(params[n].detach(), want, rtol=1e-6, atol=1e-9, msg=n)
    assert {int(opt.state[p]["step"]) for p in ref_grads_params(net)} == {1}
    opt.zero_grad()
    logits, value = net(tok)
    algo.ppo_losses(logits, value, rows).loss.backward()
    algo.clip_and_step(opt)
    assert {int(opt.state[p]["step"]) for p in ref_grads_params(net)} == {2}


def ref_grads_params(net: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


# ── T12: strata ──────────────────────────────────────────────────────────


def _length_model(n: int = 131_072, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """minibatch_pad.py's synthetic token lengths: 0 combat, 1 run non-map, 2 map."""
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, n, p=[0.86, 0.11, 0.03])
    floor = rng.integers(1, 51, n)
    base = np.array([42.0, 40.0, 75.0])[kind]
    noise = rng.normal(0, 9, n) * np.array([1, 8 / 9, 10 / 9])[kind]
    return kind, np.clip(np.round(base + 0.35 * floor + noise), 20, 160).astype(int)


def test_t12_strata_cover_each_row_once_and_pad_at_most_5_percent() -> None:
    kind, n_tok = _length_model()
    n = len(n_tok)
    strata = algo.length_strata(n_tok)
    assert len(strata) == 384 and {len(s) for s in strata} == {341, 342}
    assert np.array_equal(np.sort(np.concatenate(strata)), np.arange(n))
    assert all(n_tok[a].max() <= n_tok[b].min() for a, b in itertools.pairwise(strata))
    share = float(np.mean(kind != 0))
    dev = []
    first = None
    for epoch in range(50):
        mbs = algo.epoch_minibatches(strata, seed=11, it=5, epoch=epoch)
        assert len(mbs) == 16 and all(len(mb) == 3 and all(len(r) == 8 for r in mb) for mb in mbs)
        rows = [np.concatenate([s for r in mb for s in r]) for mb in mbs]
        assert all(8184 <= len(x) <= 8208 for x in rows)
        assert np.array_equal(np.sort(np.concatenate(rows)), np.arange(n))  # once per epoch
        chunks = [s for mb in mbs for r in mb for s in r]
        assert algo.padding_ratio(n_tok, chunks) <= 1.05
        dev += [float(np.mean(kind[x] != 0)) - share for x in rows]
        if first is None:
            first = rows
    again = algo.epoch_minibatches(strata, seed=11, it=5, epoch=0)
    assert all(
        np.array_equal(np.concatenate([s for r in mb for s in r]), x) for mb, x in zip(again, first)
    )  # every rank draws the same
    other = algo.epoch_minibatches(strata, seed=11, it=5, epoch=1)
    assert not np.array_equal(np.concatenate([s for r in other[0] for s in r]), first[0])
    # Each step mixes the batch: #1081's contiguous length-sorted slices give 9-46% run rows.
    order = np.argsort(n_tok, kind="stable")
    contiguous = [float(np.mean(kind[order[i : i + 8192]] != 0)) - share for i in range(0, n, 8192)]
    dev_a = np.abs(np.array(dev))
    assert float(np.mean(dev_a <= 0.05)) >= 0.9
    assert dev_a.max() < 0.5 * max(abs(x) for x in contiguous)
    # The KL probe: 24 strata (8,192 rows), fixed per iteration.
    p = algo.probe_strata(384, seed=11, it=5)
    assert len(p) == 24 == len(set(p.tolist())) and np.array_equal(p, np.sort(p))
    assert np.array_equal(p, algo.probe_strata(384, seed=11, it=5))
    assert not np.array_equal(p, algo.probe_strata(384, seed=11, it=6))
    assert 8184 <= sum(len(strata[i]) for i in p) <= 8208
    with pytest.raises(ValueError):
        algo.epoch_minibatches(strata, seed=1, it=0, epoch=0, per_mb=24, world=5)
    with pytest.raises(ValueError):
        algo.epoch_minibatches(strata[:100], seed=1, it=0, epoch=0)


def _real_lengths(n_runs: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    """Stored `n_tok` and student-visible tokens (TF_HIDDEN members dropped) of the
    rows of `n_runs` random-actor v0.111.0 A10 runs (WP3b's tokenizer)."""
    import random

    from ppo.e2e import contracts as C
    from ppo.e2e import tokens as TK
    from ppo.e2e.env import E2EDecision, E2EEnv

    n_tok, n_vis = [], []
    for s in range(n_runs):
        rng = random.Random(s)
        env = E2EEnv(seed0 + s)
        d = env.reset()
        while isinstance(d, E2EDecision):
            row = TK.encode(d, priv=False)
            n_tok.append(row.n_tok)
            n_vis.append(int(((row.tok_flags & C.TF_HIDDEN) == 0).sum()))
            if d.bootstrap:
                break
            ends = {c for c, a in zip(d.classes, d.actions) if a.get("action") == "end_turn"}
            plays = [c for c in range(d.n_classes) if c not in ends]
            c = rng.choice(plays) if (plays and rng.random() < 0.85) else rng.randrange(d.n_classes)
            d = env.step(c)
    return np.asarray(n_tok), np.asarray(n_vis)


def test_t12_padding_on_real_rows_with_the_pad_key_to_batch_uses(monkeypatch) -> None:
    """T12 on real rows. update.py cuts the strata by the stored `n_tok`, but
    `net.to_batch(..., NO_BUCKETS)` pads each micro-chunk to its longest
    student-visible row, and the stats row's `padding` reports
    `padding_ratio(n_tok, strata)`. On a bootstrapped 131,072-row iteration of real
    random-actor rows (the early-training mix), the padding the forward really does
    stays within the 5% bound."""
    pytest.importorskip("sts2_rs")
    import sts2_client
    from ppo.e2e import factory

    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    nt, nv = _real_lengths(12, 1_300_000_300)
    assert len(nt) > 500 and (nt > nv).any()  # some rows carry hidden members
    idx = np.random.default_rng(0).integers(0, len(nt), 131_072)
    t_b, v_b = nt[idx], nv[idx]
    strata = algo.length_strata(t_b)
    assert np.array_equal(np.sort(np.concatenate(strata)), np.arange(len(t_b)))  # each row once
    reported = algo.padding_ratio(t_b, strata)  # the stats row's `padding`
    actual = algo.padding_ratio(v_b, strata)  # what to_batch pads: max visible per chunk
    assert reported <= 1.05 and actual <= 1.05, (reported, actual)


# ── EMA ──────────────────────────────────────────────────────────────────


def test_ema_shadows_follow_the_closed_form_and_round_trip() -> None:
    net = torch.nn.Linear(3, 2)
    net.register_buffer("count", torch.tensor(0))
    half = torch.ones(2, dtype=torch.bfloat16)

    def state() -> dict[str, torch.Tensor]:
        return {**net.state_dict(), "half": half}

    ema = algo.Ema(state())
    assert set(ema.shadows) == {"ema0.996", "ema0.999"}
    w0 = net.weight.detach().clone()
    seen = []
    for i in range(5):
        with torch.no_grad():
            net.weight.add_(0.1 * (i + 1))
            net.count += 1
        half.add_(1.0)
        ema.update(state())
        seen.append(net.weight.detach().clone())
    for d in (0.996, 0.999):
        sh = ema.shadow(d)
        want = w0.clone()
        for w in seen:
            want = d * want + (1 - d) * w
        torch.testing.assert_close(sh["weight"], want, rtol=1e-6, atol=1e-7)
        assert int(sh["count"]) == 5 and sh["half"].dtype == torch.float32
    buf = io.BytesIO()
    torch.save(ema.state_dict(), buf)
    buf.seek(0)
    e2 = algo.Ema(state())
    e2.load_state_dict(torch.load(buf, weights_only=True))
    for k, sh in ema.shadows.items():
        assert all(torch.equal(sh[n], e2.shadows[k][n]) for n in sh)
    with pytest.raises(KeyError):
        ema.update({"weight": net.weight})


def test_ema_copies_its_copy_keys_instead_of_averaging_them() -> None:
    """A frozen buffer (E2ENet's `prior`) is copied into every shadow, so a restart that
    fills it does not export a prior averaged toward zero (review finding NT-2);
    `copy_from` syncs the loaded shadows at once."""
    net = torch.nn.Linear(3, 2)
    net.register_buffer("prior", torch.zeros(4))

    def state() -> dict[str, torch.Tensor]:
        return net.state_dict()

    old = algo.Ema(state(), copy=["prior"]).state_dict()
    with pytest.raises(KeyError):
        algo.Ema(state(), copy=["nope"])
    ema = algo.Ema(state(), copy=["prior"])
    ema.load_state_dict(old)
    net.prior.copy_(torch.arange(4.0))  # the restart fills the prior
    ema.copy_from(state())
    assert all(torch.equal(sh["prior"], net.prior) for sh in ema.shadows.values())
    for _ in range(3):
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(state())
    for d in (0.996, 0.999):
        assert torch.equal(ema.shadow(d)["prior"], net.prior)
        assert not torch.equal(ema.shadow(d)["weight"], net.weight)  # still averaged
    plain = algo.Ema(state())
    net.prior.mul_(2.0)
    plain.update(state())
    assert not torch.equal(plain.shadow(0.996)["prior"], net.prior)  # the old default


# ── T11, the algorithm half: a restored state reproduces the next 3 iterations ──


def _same(a, b) -> bool:
    """Bit-for-bit equality of nested outputs (NaN equals NaN)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    if torch.is_tensor(a):
        a = a.numpy()
        b = b.numpy() if torch.is_tensor(b) else None
    if isinstance(a, np.ndarray):
        return (
            isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.tobytes() == b.tobytes()
        )
    if isinstance(a, float):
        return isinstance(b, float) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


T11_ITERS = 154
T11_SCALE = 12_500  # ~4M decisions per iteration: U0 ends near it 21, λ_KS reaches 0 near it 140


def _t11_step(st: algo.AlgoState, k: int, fresh: dict, values: np.ndarray) -> dict:
    """One learner iteration minus the net: what the update would consume and what it records."""
    t_wall = S.ANNEAL_START + (k - 96) * 900.0  # 15 min per iteration; Sun 06:00 at it 96
    sched = st.begin(int((~fresh["bootstrap"]).sum()) * T11_SCALE, t_wall)
    table = st.join(fresh)
    ret = st.returns(table, values[table["ref"]])
    norm = st.normalize(table, ret)
    kl = (0.003, 0.003, 0.01, 0.03, 0.004, 0.025)[k % 6]
    ent = np.full(algo.N_TYPES, NAN)
    ent[3] = 0.05  # collapses every 20 policy iterations
    ent[4] = 0.05 if k % 25 else 0.5  # a high reading resets its streak
    ent[12] = 0.3
    stop = st.kl.stop(kl, sched, st.cfg)
    end = st.end(table, ret, probe_kl=kl, entropy_per_type=ent)
    if k == 29:
        st.clocks.pass_learning_check(st.cfg)  # the λ_KS hold ends
    if k == 60:
        assert st.clocks.hold_ks() is False  # a failed re-check after the pass: the ramp stands
    fields = (
        "adv", "target", "trained", "pi", "held", "dormant", "overdue", "boot_w", "succ", "order",
    )  # fmt: skip
    return {
        "sched": sched.log_fields(),
        "ks_types": sched.ks_types.copy(),
        "ent_gain": sched.ent_gain.copy(),
        "beta": sched.beta_rows(table["dtype"], table["is_combat"]),
        "ks": sched.ks_rows(table["dtype"], ~table["bootstrap"]),
        "ret": {f: getattr(ret, f).copy() for f in fields},
        "stats": ret.stats(table),
        "norm": norm,
        "stop": stop,
        "end": end,
        "table": {c: x.copy() for c, x in table.items()},
    }


def _snapshot(st: algo.AlgoState) -> bytes:
    buf = io.BytesIO()
    torch.save(st.to_dict(), buf)
    return buf.getvalue()


def _restore(blob: bytes) -> algo.AlgoState:
    return algo.AlgoState.from_dict(torch.load(io.BytesIO(blob), weights_only=True))


def test_t11_restored_state_reproduces_the_next_three_iterations() -> None:
    stream = Stream(8, 40, seed=11, run_len=(60, 160), practice_p=0.15, stall_p=0.05)
    tables = [stream.next_table() for _ in range(T11_ITERS)]
    values = stream.values()
    teach = np.ones(algo.N_TYPES, bool)
    teach[15] = False  # a teacherless decision type
    st = algo.AlgoState(teacher_types=teach)
    st.clocks.hold_ks()  # the learning check holds λ_KS at 1 until it 29
    r = int(np.random.default_rng(1097).integers(8, 147))
    kills = (7, r, 150)
    outs, blobs, after = [], {}, {}
    for k in range(T11_ITERS):
        outs.append(_t11_step(st, k, tables[k], values))
        if k in kills:
            blobs[k] = _snapshot(st)
        if k - 3 in kills:
            after[k - 3] = st.to_dict()
    # The run reached every regime the state has to carry.
    phases = [o["sched"]["phase"] for o in outs]
    assert {"V-warm", "U0", "U1", "U2"} <= set(phases)
    assert outs[150]["sched"]["phase"] == "U2" and outs[150]["sched"]["ks_offset"] > 0
    assert outs[7]["sched"]["phase"] == "U0" and outs[7]["stats"]["rows_held"] > 0
    assert not outs[7]["sched"]["ks_passed"] and outs[150]["sched"]["ks_passed"]
    assert outs[61]["sched"]["ks_offset"] == outs[30]["sched"]["ks_offset"] > 0
    assert outs[95]["end"]["kl_m"] != outs[95]["end"]["kl_m_before"] or any(
        o["end"]["kl_m"] != o["end"]["kl_m_before"] for o in outs[21:96]
    )
    assert all(_exact(o["end"]["kl_m"], o["end"]["kl_m_before"]) for o in outs[96:])  # frozen
    assert st.collapse.boosts[3] == 2 and outs[150]["ent_gain"][3] > 1.0
    assert sum(o["stats"]["rows_dormant"] for o in outs) > 0
    for k in kills:
        st2 = _restore(blobs[k])
        for j in range(k + 1, k + 4):
            got = _t11_step(st2, j, tables[j], values)
            for key in got:
                assert _same(got[key], outs[j][key]), (k, j, key)
        assert _same(st2.to_dict(), after[k]), k


def test_algo_state_protocol_guards() -> None:
    st = algo.AlgoState()
    with pytest.raises(RuntimeError):
        st.end({}, algo.compute_returns(run_table("", [], end=None), np.zeros(0)))
    st.begin(5, 0.0)
    with pytest.raises(RuntimeError):
        st.begin(5, 0.0)
    with pytest.raises(RuntimeError):
        st.to_dict()
    table = st.join(run_table("C", [1.0]))
    ret = st.returns(table, np.zeros(1))
    st.normalize(table, ret)
    with pytest.raises(RuntimeError):
        st.normalize(table, ret)
    st.end(table, ret)
    d = st.to_dict()
    assert d["clocks"]["it"] == 1 and d["clocks"]["decisions"] == 5
    with pytest.raises(ValueError):
        algo.AlgoState.from_dict({**d, "version": 99})
    with pytest.raises(ValueError):
        algo.AlgoState(n_types=4, teacher_types=np.ones(5, bool))
