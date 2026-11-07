"""Algorithm core of the e2e learner (plan §6 and §7), free of the net and the tokens.

Everything here is a pure, unit-tested function or a small state object with
``to_dict`` / ``from_dict``. The learner wires them together; the schedules
live in `ppo.e2e.schedules`.

**Rows.** An iteration hands over a *table*: a dict of equal-length numpy
columns, one row per stored decision, under the names in `COLUMNS` (the
schema's row fields; `table_from` renames them). A run (or a practice segment)
is the set of rows with one ``run_uid``, ordered by ``t``. The successor of a
row is the next stored row of the same run, of any type, or the terminal.
``reward`` is all reward from the row up to its successor, or to the terminal.

**Cuts and bootstrap rows.** Every iteration is cut. For each run still in
progress, the table must end with a *bootstrap row*: a copy of the run's next
stored decision, flagged ``bootstrap``. Only its value is used: as the
successor value of the row before it. It enters no loss, normaliser or
statistic, and the collector stores it again, as an ordinary row, in the next
iteration's table. A practice segment ends in a bootstrap row with ``done`` set
(the segment is over). A run that ended has a non-bootstrap ``done`` row last.

**Returns (§6 Returns, γ = 1).** Two chains run over every run's rows:

- *combat chain*, by row adjacency: ``δ_t = r_t + V(succ) - V(t)`` and
  ``A_t = δ_t + λ_c · 1[succ is a non-bootstrap combat row] · A_succ``. So
  act-3 boss 1 chains into boss 2, and a fight hands its value over to the next
  run row (``V(succ)`` is then ``V_run``).
- *run chain*, over the run rows of a run: ``δ_i = R(i→j) + V(j) - V(i)`` and
  ``A_i = δ_i + λ_r · A_j``, where j is the next run row and R(i→j) sums the
  reward of every row from i up to j. Without a next run row, j is the end:
  the bootstrap row (cut) or the terminal (value 0).

Combat rows train in the iteration that stored them; their chain is cut there.
Run rows are *held back* (`HoldBack`): they wait until their run ends or until
they are ``hold_iters`` (4) iterations old, then train with the chain cut at the
current bootstrap row. While they wait, the rewards of the combat rows that
trained and left the table are folded into them, so R stays exact. A restart
loses every open run, so a resumed session drops the rows held at that point
(`HoldBack.resume`), and `Returns.overdue` counts rows that have waited too
long, so a leak shows in the stats.

**Values** are the caller's update-start pre-pass, one per table row:
``V_combat`` for combat rows, ``V_run`` for run rows, including held and
bootstrap rows. Values must be finite on every row of a run that is not
dormant (see `Returns.dormant`).

**Per-iteration protocol** (`AlgoState`; ``sched`` fixes every lr, β and λ):

.. code-block:: text

    sched = state.begin(n_collected, t_wall)   # decisions += n_collected
    table = state.join(fresh)                  # held rows + this iteration's rows
    values = prepass(table)                    # learner: V and π_prox of every row
    ret = state.returns(table, values)         # advantages, targets, masks
    adv = state.normalize(table, ret)          # per-type moments of student rows
    strata = length_strata(n_tok[ret.trained]) # minibatches of trained rows only
    for epoch ...: for mb in epoch_minibatches(...): ppo_losses(...); clip_and_step(...)
        if state.kl.stop(probe_kl, sched, state.cfg): break
    state.end(table, ret, probe_kl=..., entropy_per_type=...)

**Resume.** A new learner session restores the state with
``AlgoState.from_dict`` and calls ``state.hold.resume()`` once, before its
first ``join`` and before it hands the state to its ranks. Dropping every
held row's run with ``hold.drop`` on every replica before its first ``join``
drops the same rows.

Every step is deterministic given its inputs, so each data-parallel rank can
keep an identical replica of the state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .schedules import (
    DEFAULT,
    Clocks,
    EntropyCollapse,
    IterSchedule,
    KLControl,
    ScheduleCfg,
    schedule_for,
)

__all__ = [
    "ACTOR_COMBAT_TEACHER",
    "ACTOR_RUN_TEACHER",
    "ACTOR_STUDENT",
    "ADV_MIN_ROWS",
    "CELL_NONE",
    "COLUMNS",
    "HOLD_ITERS",
    "LAM_C",
    "LAM_R",
    "N_TYPES",
    "AdvNorm",
    "AlgoState",
    "Ema",
    "HoldBack",
    "LossCfg",
    "LossOut",
    "LossRows",
    "Returns",
    "clip_and_step",
    "compute_returns",
    "entropy_by_type",
    "epoch_minibatches",
    "explained_variance",
    "length_strata",
    "logp_of",
    "make_optimizer",
    "masked_log_softmax",
    "pad_ragged",
    "padding_ratio",
    "param_groups",
    "ppo_losses",
    "probe_kl",
    "probe_strata",
    "row_rewards",
    "set_lr",
    "stats_means",
    "table_from",
]

# ── row columns ──────────────────────────────────────────────────────────

#: Our column name -> the schema's row field (``ppo.e2e.contracts``, frozen on
#: the schema branch). If the schema renames a field, change the right-hand
#: side here and nothing else.
COLUMNS: dict[str, str] = {
    "run_uid": "run_uid",  # int64: unique per run or practice segment
    "t": "t",  # int32: order of the stored row within its run
    "is_combat": "is_combat",  # bool
    "dtype": "dtype",  # uint8: decision-type id
    "reward": "reward",  # float32: reward up to the next stored row (or terminal)
    "done": "done",  # bool: the run or segment ended after this row
    "bootstrap": "bootstrap",  # bool: value-only row
    "actor": "actor",  # uint8: 0 student, 1 combat teacher, 2 run teacher
    "logp_behav": "logp_behav",  # float32: log-prob of the taken class, acting policy
    "n_classes": "n_classes",  # uint16: merged options
    "action": "action",  # uint16: taken merged class
    "iter": "iter",  # int32: collection iteration
    "v_behav": "v_behav",  # float32: value at acting time
    "cell": "cell",  # uint8: practice cell, 255 = none
}
ACTOR_STUDENT, ACTOR_COMBAT_TEACHER, ACTOR_RUN_TEACHER = 0, 1, 2
CELL_NONE = 255
N_TYPES = 256  # dtype is uint8
LAM_C, LAM_R, HOLD_ITERS = 0.95, 0.97, 4
#: Rows a decision type must have seen before it normalises with its own moments (`AdvNorm`).
ADV_MIN_ROWS = 64


def table_from(cols: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The schema's columns under our names (`COLUMNS`); other columns pass through."""
    back = {v: k for k, v in COLUMNS.items()}
    return {back.get(k, k): np.asarray(v) for k, v in cols.items()}


def _col(table: Mapping[str, Any], name: str, dt: Any, default: Any = None) -> np.ndarray:
    if name in table:
        return np.asarray(table[name]).astype(dt, copy=False)
    if default is None:
        raise KeyError(f"table has no column {name!r}")
    return np.full(len(np.asarray(table["run_uid"])), default, dt)


def _nrows(table: Mapping[str, Any]) -> int:
    lens = {k: len(np.asarray(v)) for k, v in table.items()}
    if len(set(lens.values())) > 1:
        raise ValueError(f"columns differ in length: {lens}")
    return next(iter(lens.values()), 0)


def pad_ragged(
    flat: np.ndarray, lengths: np.ndarray, width: int | None = None, fill: float = 0.0
) -> np.ndarray:
    """Variable-length rows (e.g. teacher probabilities) as a [B, width] float32 matrix.

    Row i is ``flat[o_i : o_i + lengths[i]]`` with ``o_i = sum(lengths[:i])``;
    positions at or past ``lengths[i]`` hold ``fill``. NaN rows (no teacher)
    stay NaN within their length.
    """
    flat = np.asarray(flat, np.float32)
    lengths = np.asarray(lengths, np.int64)
    if int(lengths.sum()) != len(flat):
        raise ValueError(f"lengths sum to {int(lengths.sum())}, flat has {len(flat)}")
    w = int(lengths.max(initial=0)) if width is None else int(width)
    if lengths.size and int(lengths.max()) > w:
        raise ValueError(f"a row is longer ({int(lengths.max())}) than width {w}")
    out = np.full((len(lengths), w), fill, np.float32)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if lengths.size else lengths
    cols = np.arange(len(flat)) - np.repeat(offs, lengths)
    out[rows, cols] = flat
    return out


# ── rewards per stored row ───────────────────────────────────────────────


def row_rewards(step_reward: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """One run's step rewards summed onto its stored rows (the ``reward`` column).

    ``step_reward[s]`` is the reward of the transition after env step s, for
    every step including auto-advanced ones; ``stored[s]`` marks the steps that
    became stored rows. Each stored row gets all reward from its step up to the
    next stored step; the last one gets everything to the terminal. Reward
    before the first stored step goes onto the first stored row, so the stored
    rows sum to the run's total reward exactly (T4).
    """
    r = np.asarray(step_reward, np.float64)
    idx = np.flatnonzero(np.asarray(stored, bool))
    if idx.size == 0:
        raise ValueError("a run needs at least one stored row")
    out = np.add.reduceat(r, idx)
    out[0] += r[: idx[0]].sum()
    return out


# ── returns ──────────────────────────────────────────────────────────────


def _scan(delta: np.ndarray, nxt: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """``A[i] = delta[i] + mult[i] · A[nxt[i]]`` over a forest of chains (``nxt = -1`` ends one).

    Pointer jumping: each pass doubles how far every node has summed along its
    chain, so a chain of length L takes about log2(L) vectorised passes. The
    sums are associated differently from the sequential recursion, which
    changes results only at float rounding (about 1e-16).
    """
    s = np.asarray(delta, np.float64).copy()
    j = np.asarray(nxt, np.int64).copy()
    p = np.where(j >= 0, np.asarray(mult, np.float64), 0.0)
    act = np.flatnonzero(j >= 0)
    while act.size:
        jj = j[act]
        s[act] = s[act] + p[act] * s[jj]
        p[act] = p[act] * p[jj]
        j[act] = j[jj]
        act = act[j[act] >= 0]
    return s


@dataclass(eq=False)
class Returns:
    """`compute_returns` output, aligned with the input table's rows.

    - ``adv``, ``target`` (float64): raw advantage and value target (adv + V).
      NaN on every row that trains nothing this iteration.
    - ``trained``: the row's losses apply now. The value loss and the teacher
      term apply on all of them.
    - ``pi``: trained and student-acted. PPO, advantage moments and statistics.
    - ``held``: run rows that wait; `HoldBack.carry` keeps exactly these.
    - ``dormant``: held rows of runs that sent no new row this iteration
      (neither a terminal nor a bootstrap row). They keep their age.
    - ``overdue``: rows that have waited more than ``hold_iters + 2``
      iterations (``waited``, which counts dormant iterations too). It is empty
      in normal operation, and a row that never leaves (a leak) stays in it
      (stat ``rows_overdue``).
    - ``boot_w``: weight of the cut (bootstrap) value in the row's target; 0 when
      the chain reached the terminal or a run row, NaN when not trained.
    - ``succ``: table index of the row's successor (the next row of its run in
      the table, possibly its bootstrap row); -1 for a run's last row.
    - ``order``: the table's (run_uid, t) sort order.
    """

    adv: np.ndarray
    target: np.ndarray
    trained: np.ndarray
    pi: np.ndarray
    held: np.ndarray
    dormant: np.ndarray
    overdue: np.ndarray
    boot_w: np.ndarray
    succ: np.ndarray
    order: np.ndarray

    def stats(self, table: Mapping[str, Any]) -> dict[str, float]:
        comb = _col(table, "is_combat", bool)
        age = _col(table, "age", np.int64, default=0)
        tc, tr = self.trained & comb, self.trained & ~comb
        return {
            "rows_trained": int(self.trained.sum()),
            "rows_trained_combat": int(tc.sum()),
            "rows_trained_run": int(tr.sum()),
            "rows_pi": int(self.pi.sum()),
            "rows_held": int(self.held.sum()),
            "rows_dormant": int(self.dormant.sum()),
            "rows_overdue": int(self.overdue.sum()),
            "boot_w_combat": float(self.boot_w[tc].mean()) if tc.any() else math.nan,
            "boot_w_run": float(self.boot_w[tr].mean()) if tr.any() else math.nan,
            "age_run_mean": float(age[tr].mean()) if tr.any() else math.nan,
            "age_run_max": int(age[tr].max()) if tr.any() else 0,
        }


def compute_returns(
    table: Mapping[str, Any],
    values: np.ndarray,
    *,
    lam_c: float = LAM_C,
    lam_r: float = LAM_R,
    hold_iters: int = HOLD_ITERS,
) -> Returns:
    """Two-chain GAE (γ = 1) over one iteration's table (see the module docstring).

    Columns read: run_uid, t, is_combat, reward, done, bootstrap, actor, and
    ``age`` / ``waited`` / ``carried`` if present (`HoldBack.join` adds them;
    fresh rows are age 0). A run row trains iff its run ended in this table or
    its age is at least ``hold_iters`` (0 = no hold-back: run rows are cut like
    combat rows). ``waited`` only feeds ``overdue``.
    Raises ValueError on a malformed table: duplicate (run_uid, t), a bootstrap
    or done row that is not last in its run, a carried combat or bootstrap row,
    a run with new rows but neither a terminal nor a bootstrap row, or
    non-finite values or rewards where they are used.
    """
    run = _col(table, "run_uid", np.int64)
    n = len(run)
    t = _col(table, "t", np.int64)
    comb = _col(table, "is_combat", bool)
    rew = _col(table, "reward", np.float64)
    done = _col(table, "done", bool)
    boot = _col(table, "bootstrap", bool)
    actor = _col(table, "actor", np.int64)
    age = _col(table, "age", np.int64, default=0)
    carried = _col(table, "carried", bool, default=False)
    overdue = _col(table, "waited", np.int64, default=0) > hold_iters + 2
    v_in = np.asarray(values, np.float64)
    if v_in.shape != (n,):
        raise ValueError(f"values have shape {v_in.shape}, want ({n},)")
    if n == 0:
        e = np.zeros(0)
        b = np.zeros(0, bool)
        i = np.zeros(0, np.int64)
        return Returns(
            adv=e,
            target=e.copy(),
            trained=b,
            pi=b.copy(),
            held=b.copy(),
            dormant=b.copy(),
            overdue=b.copy(),
            boot_w=e.copy(),
            succ=i,
            order=i.copy(),
        )

    order = np.lexsort((t, run))
    r_, t_, c_, w_, d_, b_, a_, g_, k_, v_ = (
        x[order] for x in (run, t, comb, rew, done, boot, actor, age, carried, v_in)
    )
    pos = np.arange(n)
    same = np.zeros(n, bool)  # same[p]: row p + 1 belongs to the same run
    same[:-1] = r_[1:] == r_[:-1]
    dup = same[:-1] & (t_[1:] <= t_[:-1])
    if dup.any():
        p = int(np.flatnonzero(dup)[0])
        raise ValueError(f"run {r_[p]}: t {t_[p]} appears twice")
    last = ~same
    first = np.ones(n, bool)
    first[1:] = ~same[:-1]
    starts, ends = np.flatnonzero(first), np.flatnonzero(last)
    rid = np.cumsum(first) - 1  # run index per sorted row

    def _bad(mask: np.ndarray, what: str) -> None:
        if mask.any():
            p = int(np.flatnonzero(mask)[0])
            raise ValueError(f"run {r_[p]}, t {t_[p]}: {what}")

    _bad(b_ & ~last, "a bootstrap row must be the last row of its run")
    _bad(d_ & ~b_ & ~last, "a done row must be the last row of its run")
    _bad(k_ & (c_ | b_), "carried rows must be run rows, never combat or bootstrap rows")
    end_boot = b_[ends]
    end_term = d_[ends] & ~b_[ends]
    open_run = ~end_boot & ~end_term
    has_fresh = np.logical_or.reduceat(~k_, starts)
    _bad((open_run & has_fresh)[rid] & last, "new rows but neither a terminal nor a bootstrap row")
    dorm = open_run[rid]  # only carried rows and no end: nothing to compute
    ended = (end_term | (end_boot & d_[ends]))[rid]

    cn = c_ & ~b_  # combat-chain nodes
    rn = ~c_ & ~b_ & ~dorm  # run-chain nodes
    _bad(~dorm & ~np.isfinite(v_), "value is not finite")
    _bad((cn | rn) & ~np.isfinite(w_), "reward is not finite")

    succ = np.where(same, pos + 1, -1)
    has_s = succ >= 0
    vnext = np.zeros(n)
    vnext[has_s] = v_[succ[has_s]]
    delta = np.zeros(n)
    nxt = np.full(n, -1, np.int64)
    mult = np.zeros(n)

    # Combat chain: continue only into a non-bootstrap combat successor.
    delta[cn] = w_[cn] + vnext[cn] - v_[cn]
    ci = np.flatnonzero(cn & has_s)
    cs = succ[ci]
    ok = c_[cs] & ~b_[cs]
    nxt[ci[ok]] = cs[ok]
    mult[cn] = lam_c

    # Run chain over each run's run rows; R(i→j) by segment sums (local, so exact).
    rp = np.flatnonzero(rn)
    nq = np.full(rp.size, -1, np.int64)
    if rp.size > 1:
        same_r = rid[rp[1:]] == rid[rp[:-1]]
        nq[:-1][same_r] = rp[1:][same_r]
    r_eff = np.where(b_, 0.0, w_)
    seg = np.unique(np.concatenate([rp, starts, np.flatnonzero(b_)]))
    seg_sum = np.add.reduceat(r_eff, seg)
    big_r = seg_sum[np.searchsorted(seg, rp)]
    hq = nq >= 0
    vend = np.zeros(rp.size)
    vend[hq] = v_[nq[hq]]
    e_end = ends[rid[rp[~hq]]]  # the run's last row: bootstrap (cut) or terminal
    vend[~hq] = np.where(b_[e_end], v_[e_end], 0.0)
    delta[rp] = big_r + vend - v_[rp]
    nxt[rp] = nq
    mult[rp] = lam_r

    adv_s = _scan(delta, nxt, mult)

    # The weight of the cut value: 1 at a chain end that bootstraps from a cut.
    cut_end = np.zeros(n)
    ce = np.flatnonzero(cn & (nxt < 0) & has_s)
    cut_end[ce] = b_[succ[ce]]
    cut_end[rp[~hq]] = b_[e_end]
    bw_s = _scan(cut_end, nxt, mult)

    release = ended | (g_ >= hold_iters)
    trained = cn | (rn & release)
    held = (rn & ~release) | dorm
    adv_s = np.where(trained, adv_s, np.nan)
    tgt_s = adv_s + v_
    bw_s = np.where(trained, bw_s, np.nan)
    pi_s = trained & (a_ == ACTOR_STUDENT)
    succ_s = np.where(has_s, order[np.maximum(succ, 0)], -1)

    def back(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[order] = x
        return out

    return Returns(
        adv=back(adv_s),
        target=back(tgt_s),
        trained=back(trained),
        pi=back(pi_s),
        held=back(held),
        dormant=back(dorm),
        overdue=overdue,
        boot_w=back(bw_s),
        succ=back(succ_s),
        order=order,
    )


def explained_variance(target: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    """1 - Var(target - V) / Var(target) over ``mask`` rows (NaN for fewer than 2)."""
    m = np.asarray(mask, bool)
    if m.sum() < 2:
        return math.nan
    y = np.asarray(target, np.float64)[m]
    d = y - np.asarray(values, np.float64)[m]
    return 1.0 - float(np.var(d)) / max(1e-12, float(np.var(y)))


# ── hold-back ────────────────────────────────────────────────────────────


class HoldBack:
    """The run rows that wait for their run's end or their release age (§6 Returns).

    `HoldBack` carries every column of the table for the rows it keeps,
    including fixed-width 2-D numeric columns (padded tokens, `pad_ragged`
    teacher probabilities), and `to_dict` saves them. Alternatively the learner
    keeps held rows' inputs itself, keyed by a column it adds (e.g. ``ref``).
    `join` adds three columns. ``age`` counts the iterations a row has waited
    while its run sent rows: it freezes while the run is dormant, and the row
    trains at ``hold_iters``. ``waited`` counts every iteration it has waited,
    dormant ones included. ``carried`` marks rows that came from here.
    A kept row's ``reward`` becomes the sum of its own reward and the reward of
    every row that left the table after it, up to the next kept row or the cut.

    **Restarts.** A restart loses every open run with its workers, and a lost
    run never sends another row (segment uids carry the session). Its held
    rows would stay dormant for good: they would never train, never leave, and
    ride along in every later table, row batch and state. So a resumed session
    calls `resume` before its first `join` (a `drop` of every held row's run
    drops the same rows). That drops every held row (the ~39K held at a
    restart, under a third of one iteration's 131,072), so the held rows'
    inputs need not survive a restart. `Returns.overdue` (stat
    ``rows_overdue``) counts rows that have waited more than ``hold_iters + 2``
    iterations. Normal operation never has one, so a leak shows in the stats.
    """

    def __init__(self, hold_iters: int = HOLD_ITERS) -> None:
        self.hold_iters = int(hold_iters)
        self.cols: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.cols["run_uid"]) if self.cols else 0

    def join(self, fresh: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """This iteration's table: the held rows, then the fresh rows (age 0).

        ``fresh`` holds the rows stored this iteration plus the bootstrap rows.
        Its columns must match the held rows' columns (minus age, waited and
        carried). ``reward`` becomes float64 so folded sums stay exact.
        """
        f = {k: np.asarray(v) for k, v in fresh.items()}
        for k in ("age", "waited", "carried"):
            if k in f:
                raise ValueError(f"fresh rows must not have a {k!r} column")
        n = _nrows(f)
        f["reward"] = f["reward"].astype(np.float64)
        f["age"] = np.zeros(n, np.int64)
        f["waited"] = np.zeros(n, np.int64)
        f["carried"] = np.zeros(n, bool)
        if not self.cols:
            return f
        if set(self.cols) != set(f):
            raise ValueError(
                f"fresh columns {sorted(f)} differ from the held rows' {sorted(self.cols)}"
            )
        return {k: np.concatenate([self.cols[k], f[k]]) for k in f}

    def carry(self, table: Mapping[str, Any], ret: Returns) -> None:
        """Keep ``ret.held`` rows for the next iteration: fold rewards, age them by 1.

        ``age`` stays put on dormant rows; ``waited`` grows on every kept row.
        """
        keep = np.asarray(ret.held, bool)
        if len(keep) != _nrows(table):
            raise ValueError("ret does not belong to this table")
        if not keep.any():
            self.cols = {}
            return
        order = ret.order
        run_s = _col(table, "run_uid", np.int64)[order]
        boot_s = _col(table, "bootstrap", bool)[order]
        comb_s = _col(table, "is_combat", bool)[order]
        rew_s = _col(table, "reward", np.float64)[order]
        keep_s = keep[order]
        n = len(order)
        first = np.ones(n, bool)
        first[1:] = run_s[1:] != run_s[:-1]
        # Kept rows are the newest run rows of their run: no released run row after one.
        rp = np.flatnonzero(~comb_s & ~boot_s)
        if rp.size > 1:
            same = run_s[rp[1:]] == run_s[rp[:-1]]
            if np.any(same & keep_s[rp[:-1]] & ~keep_s[rp[1:]]):
                raise ValueError("a released run row follows a held one in the same run")
        kp = np.flatnonzero(keep_s)
        seg = np.unique(np.concatenate([kp, np.flatnonzero(first), np.flatnonzero(boot_s)]))
        sums = np.add.reduceat(np.where(boot_s, 0.0, rew_s), seg)
        folded = sums[np.searchsorted(seg, kp)]
        idx = order[kp]
        new = {k: np.asarray(v)[idx] for k, v in table.items()}
        new["reward"] = folded
        age = _col(table, "age", np.int64, default=0)[idx]
        new["age"] = age + np.where(np.asarray(ret.dormant, bool)[idx], 0, 1)
        new["waited"] = _col(table, "waited", np.int64, default=0)[idx] + 1
        new["carried"] = np.ones(len(idx), bool)
        self.cols = new

    def resume(self) -> int:
        """Start a resumed session: drop every held row and return the count.

        Every row held at a restart belongs to a run the restart lost, so none
        of them would ever train (see the class docstring). Call this once per
        resumed session, before its first `join`, on the state the session and
        its ranks start from. Calling it again does nothing and returns 0.
        Dropping costs the dropped rows' training: under 2 iterations' worth
        of rows over the week's 4-6 restarts, out of ~70,000 iterations.
        """
        n = len(self)
        self.cols = {}
        return n

    def drop(self, run_uids: Iterable[int]) -> int:
        """Forget the held rows of runs that will never send another row (errors); return the count."""
        if not self.cols:
            return 0
        gone = np.isin(self.cols["run_uid"], np.asarray(list(run_uids), np.int64))
        if gone.any():
            self.cols = {k: v[~gone] for k, v in self.cols.items()}
        return int(gone.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hold_iters": self.hold_iters,
            "cols": {k: _pack(v) for k, v in self.cols.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> HoldBack:
        h = cls(int(d["hold_iters"]))
        h.cols = {k: _unpack(v) for k, v in d["cols"].items()}
        return h


# ── advantage normalisation ──────────────────────────────────────────────


def _pooled(a: np.ndarray) -> tuple[float, float]:
    """Plain mean and std (same floor as `AdvNorm.std`) of one pool of advantages."""
    m = float(np.mean(a))
    return m, math.sqrt(max(1e-6, float(np.mean(a * a)) - m * m))


class AdvNorm:
    """Per decision type, exponentially weighted advantage moments (§6 Advantages).

    **Moments.** A type's first ``min_rows`` (64) rows set its moments to their
    plain means of a and a² (the batch that reaches ``min_rows`` counts
    whole). After that each batch of k rows keeps ``(1 - decay)^k`` of the old
    ones. That is `ppo.trainer.RunningMoments` per type, whose first batch, of
    any size, sets the moments: ``min_rows = 1``. std = sqrt(max(1e-6,
    E[a²] - E[a]²)), so at least 1e-3. Update it with student-acted rows only.

    **Normalising.** A type that has seen ``min_rows`` rows, this batch's
    included, uses its own moments. Until then its rows use the pooled moments
    of the batch's rows of their family (combat or out of combat), or of the
    whole batch when that family has fewer than ``min_rows`` rows too. A
    once-per-run type in early U0 gets a few student rows per iteration. Its
    own moments from so few rows would inflate its normalised advantages up to
    ~10x for tens of iterations.
    """

    def __init__(
        self, n_types: int = N_TYPES, decay: float = 5e-4, min_rows: int = ADV_MIN_ROWS
    ) -> None:
        if int(min_rows) < 1:
            raise ValueError(f"min_rows must be at least 1, not {min_rows}")
        self.decay = float(decay)
        self.min_rows = int(min_rows)
        self.mean = np.zeros(n_types)
        self.sq = np.ones(n_types)
        self.count = np.zeros(n_types, np.int64)  # rows seen per type

    @property
    def inited(self) -> np.ndarray:
        """The types that have seen a row."""
        return self.count > 0

    def update(self, adv: np.ndarray, dtype: np.ndarray) -> None:
        a = np.asarray(adv, np.float64)
        d = np.asarray(dtype, np.int64)
        n_t = len(self.mean)
        if a.shape != d.shape:
            raise ValueError("adv and dtype differ in shape")
        if d.size and (d.min() < 0 or d.max() >= n_t):
            raise ValueError(f"dtype outside [0, {n_t})")
        if not np.all(np.isfinite(a)):
            raise ValueError("advantages must be finite")
        cnt = np.bincount(d, minlength=n_t)
        s1 = np.bincount(d, weights=a, minlength=n_t)
        s2 = np.bincount(d, weights=a * a, minlength=n_t)
        has = cnt > 0
        bm = np.where(has, s1 / np.maximum(cnt, 1), 0.0)
        bs = np.where(has, s2 / np.maximum(cnt, 1), 0.0)
        warm = has & (self.count >= self.min_rows)
        cold = has & ~warm
        # Warming up: the plain mean of every row so far (exactly the batch's for a new type).
        w = self.count[cold] / (self.count[cold] + cnt[cold])
        self.mean[cold] = w * self.mean[cold] + (1.0 - w) * bm[cold]
        self.sq[cold] = w * self.sq[cold] + (1.0 - w) * bs[cold]
        keep = (1.0 - self.decay) ** cnt[warm]
        self.mean[warm] = keep * self.mean[warm] + (1.0 - keep) * bm[warm]
        self.sq[warm] = keep * self.sq[warm] + (1.0 - keep) * bs[warm]
        self.count += cnt

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(1e-6, self.sq - self.mean * self.mean))

    def normalize(
        self, adv: np.ndarray, dtype: np.ndarray, combat: np.ndarray | None = None
    ) -> np.ndarray:
        """One batch's advantages, normalised (see the class docstring).

        ``combat`` (per row) splits the batch into its two families; None puts
        every row in one.
        """
        a = np.asarray(adv, np.float64)
        d = np.asarray(dtype, np.int64)
        fam = np.zeros(a.shape, bool) if combat is None else np.asarray(combat, bool)
        if d.shape != a.shape or fam.shape != a.shape:
            raise ValueError("adv, dtype and combat differ in shape")
        mean, std = self.mean[d], self.std()[d]
        cold = self.count[d] < self.min_rows
        if cold.any():
            whole = _pooled(a)
            for f in (False, True):
                rows = cold & (fam == f)
                if rows.any():
                    in_f = fam == f
                    m, s = _pooled(a[in_f]) if int(in_f.sum()) >= self.min_rows else whole
                    mean[rows], std[rows] = m, s
        return (a - mean) / std

    def update_normalize(
        self,
        adv: np.ndarray,
        dtype: np.ndarray,
        mask: np.ndarray,
        combat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update with the ``mask`` rows, then normalise them as one batch; NaN elsewhere."""
        m = np.asarray(mask, bool)
        a = np.asarray(adv, np.float64)
        d = np.asarray(dtype, np.int64)
        self.update(a[m], d[m])
        out = np.full(len(a), np.nan)
        out[m] = self.normalize(a[m], d[m], None if combat is None else np.asarray(combat, bool)[m])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "min_rows": self.min_rows,
            "mean": self.mean.tolist(),
            "sq": self.sq.tolist(),
            "count": self.count.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AdvNorm:
        a = cls(len(d["mean"]), float(d["decay"]), int(d["min_rows"]))
        a.mean[:] = d["mean"]
        a.sq[:] = d["sq"]
        a.count[:] = d["count"]
        return a


# ── losses (torch) ───────────────────────────────────────────────────────

#: Fill for logits of padded classes. Finite, so 0 · log p never becomes NaN in
#: the forward or backward pass.
NEG = -1e9


def class_mask(n_classes: torch.Tensor, width: int) -> torch.Tensor:
    """[B, width] bool: True for the row's merged classes 0 .. n_classes - 1."""
    return torch.arange(width, device=n_classes.device)[None, :] < n_classes[:, None]


def masked_log_softmax(logits: torch.Tensor, n_classes: torch.Tensor) -> torch.Tensor:
    """fp32 log-softmax over each row's first ``n_classes`` logits; padded classes get ~-1e9.

    Use it in the pre-pass too, so π_prox and π_θ mask alike.
    """
    m = class_mask(n_classes, logits.shape[-1])
    return torch.log_softmax(logits.float().masked_fill(~m, NEG), dim=-1)


def logp_of(logits: torch.Tensor, n_classes: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """log π(action) per row (the pre-pass's π_prox)."""
    lp = masked_log_softmax(logits, n_classes)
    return lp.gather(1, action.long()[:, None]).squeeze(1)


@dataclass(frozen=True)
class LossCfg:
    clip: float = 0.2  # ratio clip around π_prox
    is_clamp: float = 2.0  # max of the IS weight π_prox / π_behav
    dual_clip: float = 3.0  # dual-clip c for negative advantages
    vf_coef: float = 0.5  # value loss = vf_coef · (V - target)²


@dataclass(eq=False)
class LossRows:
    """Per-row loss inputs of one micro-chunk (all length-B tensors on one device).

    ``adv`` is the normalised advantage (`AlgoState.normalize`) and ``target``
    the value target; both may be NaN outside ``pi`` and ``v``. ``beta`` and
    ``ks`` are `IterSchedule.beta_rows` / `ks_rows`. ``teacher`` is [B, A]
    (`pad_ragged`), NaN on rows without a teacher; None when no row has one.
    """

    n_classes: torch.Tensor  # int64
    action: torch.Tensor  # int64
    lp_prox: torch.Tensor  # float32, from the update-start pre-pass
    lp_behav: torch.Tensor  # float32, the acting policy's
    adv: torch.Tensor  # float32
    target: torch.Tensor  # float32
    pi: torch.Tensor  # bool: PPO, entropy bonus and statistics rows
    v: torch.Tensor  # bool: value-loss and teacher-term rows (trained rows)
    beta: torch.Tensor  # float32: entropy coefficient
    ks: torch.Tensor  # float32: teacher-term weight λ_KS
    dtype: torch.Tensor  # int64: decision type
    teacher: torch.Tensor | None = None  # float32 [B, A]

    @classmethod
    def from_numpy(cls, device: Any = "cpu", **cols: Any) -> LossRows:
        """Build from numpy columns (same names as the fields)."""
        dt = {
            "n_classes": torch.int64,
            "action": torch.int64,
            "dtype": torch.int64,
            "pi": torch.bool,
            "v": torch.bool,
        }
        out: dict[str, Any] = {}
        for k in cls.__dataclass_fields__:
            x = cols.get(k)
            if x is None:
                if k != "teacher":
                    raise KeyError(f"missing loss column {k!r}")
                out[k] = None
                continue
            out[k] = torch.as_tensor(np.asarray(x), dtype=dt.get(k, torch.float32), device=device)
        return cls(**out)

    def index(self, idx: Any) -> LossRows:
        return LossRows(
            **{
                k: (None if getattr(self, k) is None else getattr(self, k)[idx])
                for k in self.__dataclass_fields__
            }
        )

    def to(self, device: Any) -> LossRows:
        return LossRows(
            **{
                k: (None if getattr(self, k) is None else getattr(self, k).to(device))
                for k in self.__dataclass_fields__
            }
        )


@dataclass(eq=False)
class LossOut:
    loss: torch.Tensor  # scalar, for backward
    stats: dict[str, torch.Tensor]  # detached sums (see `stats_means`)
    denom: float


def ppo_losses(
    logits: torch.Tensor | None,
    value: torch.Tensor,
    rows: LossRows,
    *,
    policy_on: bool = True,
    denom: float | None = None,
    cfg: LossCfg = LossCfg(),
    n_types: int = N_TYPES,
) -> LossOut:
    """The update's loss on one micro-chunk (§6 Objective, Values, Entropy; §7 Loss).

    Every term is a sum over its rows divided by ``denom``: the number of
    trained rows in the whole minibatch, over all ranks (default: this chunk's
    ``rows.v`` count). The summed gradients of all chunks and ranks are then
    the gradient of the minibatch mean.

    - value: ``vf_coef · (V - target)²`` on ``v`` rows, no value clipping;
    - PPO (``pi`` rows): ratio = π_θ/π_prox clipped at ``clip``, times the IS
      weight π_prox/π_behav clamped at ``is_clamp``, dual-clipped at
      ``dual_clip · A`` for A < 0;
    - entropy bonus (``pi`` rows): β · H / log(n_classes) over merged classes;
    - teacher term (``v`` rows with ks > 0 and a teacher):
      λ_KS · KL(π_T ‖ π_S). Its gradient equals the U0 cross-entropy H(π_T, π_S).

    ``policy_on=False`` (V-warm) returns the value loss only; ``logits`` may
    then be None. Rows outside ``v`` are inert: their inputs (logits, value and
    every column) are replaced before use, so even NaN cannot reach the loss,
    the stats or the gradient with respect to ``logits`` and ``value``. Keep
    such rows out of minibatches anyway: a NaN inside the net still poisons
    its weight gradients. Raises ValueError if a used row has n_classes < 2
    or an action outside it.
    """
    v_m = rows.v.bool()
    pi_m = rows.pi.bool() & v_m
    if denom is None:
        denom = float(max(1, int(v_m.sum())))
    zero = torch.zeros_like(value, dtype=torch.float32)
    vt = torch.where(v_m, value.float(), zero)
    tgt = torch.where(v_m, rows.target.float(), zero)
    vmf = v_m.float()
    vl_rows = cfg.vf_coef * (vt - tgt) ** 2 * vmf
    vloss = vl_rows.sum() / denom
    st: dict[str, torch.Tensor] = {
        "n_v": vmf.sum().detach(),
        "vloss": vl_rows.sum().detach(),
    }
    if not policy_on:
        return LossOut(vloss, st, denom)
    if logits is None:
        raise ValueError("the policy terms need logits")
    b, width = logits.shape
    nc_raw = rows.n_classes.long()
    act_raw = rows.action.long()
    bad = v_m & ((nc_raw < 2) | (act_raw < 0) | (act_raw >= nc_raw) | (nc_raw > width))
    if bool(bad.any()):
        i = int(torch.nonzero(bad)[0])
        raise ValueError(
            f"row {i}: n_classes {int(nc_raw[i])}, action {int(act_raw[i])}, width {width}"
            " (a stored row needs 2 or more classes and an action among them)"
        )
    nc = torch.where(v_m, nc_raw, torch.full_like(nc_raw, 2))
    act = torch.where(v_m, act_raw, torch.zeros_like(act_raw))
    lg = torch.where(v_m[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    cmask = class_mask(nc, width)
    lp_all = torch.log_softmax(lg.masked_fill(~cmask, NEG), dim=-1)
    lp = lp_all.gather(1, act[:, None]).squeeze(1)
    pim = pi_m.float()

    # PPO, decoupled (Hilton et al. 2021) with dual clip (Ye et al. 2020).
    lpp = torch.where(pi_m, rows.lp_prox.float(), zero)
    lpb = torch.where(pi_m, rows.lp_behav.float(), zero)
    a = torch.where(pi_m, rows.adv.float(), zero)
    d = torch.where(pi_m, lp - lpp, zero)
    ratio = d.exp()
    w_raw = (lpp - lpb).exp()
    w = w_raw.clamp(max=cfg.is_clamp)
    s1 = ratio * a
    s2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * a
    surr = torch.min(s1, s2)
    dual = cfg.dual_clip * a
    use_dual = (a < 0) & (surr < dual)
    surr = torch.where(use_dual, dual, surr)
    pl_rows = -(w * surr) * pim

    # Entropy over merged classes, normalised by log(n_classes) into [0, 1].
    p_all = lp_all.exp()
    h = -(torch.where(cmask, p_all * lp_all, torch.zeros_like(lp_all))).sum(-1)
    hn = (h / torch.log(nc.float())).clamp(0.0, 1.0)
    beta = torch.where(pi_m, rows.beta.float(), zero)
    eb_rows = beta * hn * pim

    # Teacher term: λ_KS · KL(π_T ‖ π_S) on trained rows with a teacher.
    ks_w = torch.where(v_m, rows.ks.float(), zero)
    if rows.teacher is not None:
        if rows.teacher.shape != logits.shape:
            raise ValueError(f"teacher {tuple(rows.teacher.shape)} vs logits {(b, width)}")
        tp = rows.teacher.float()
        t_fin = torch.where(cmask, torch.isfinite(tp), torch.ones_like(cmask)).all(-1)
        t_ok = (ks_w > 0) & t_fin
        p_t = torch.where(cmask & t_ok[:, None], tp, torch.zeros_like(tp))
        ce_rows = -(p_t * lp_all).sum(-1)
        kl_rows = (torch.xlogy(p_t, p_t) - p_t * lp_all).sum(-1)
        missing = (ks_w > 0) & ~t_fin
    else:
        t_ok = torch.zeros_like(v_m)
        ce_rows = kl_rows = zero
        missing = ks_w > 0
    tkf = t_ok.float()
    ks_rows = ks_w * kl_rows * tkf

    loss = vloss + (pl_rows.sum() - eb_rows.sum() + ks_rows.sum()) / denom

    with torch.no_grad():
        k3 = (ratio - 1.0) - d  # KL(π_prox ‖ π_θ) estimate at the taken action
        dt = torch.where(pi_m, rows.dtype.long(), torch.zeros_like(rows.dtype.long()))
        st.update(
            {
                "n_pi": pim.sum(),
                "ploss": pl_rows.sum(),
                "ent": (h * pim).sum(),
                "ent_n": (hn * pim).sum(),
                "ent_bonus": eb_rows.sum(),
                "kl": (k3 * pim).sum(),
                "clipfrac": (((ratio - 1.0).abs() > cfg.clip).float() * pim).sum(),
                "dualfrac": (use_dual.float() * pim).sum(),
                "isw": (w * pim).sum(),
                "is_clamped": ((w_raw > cfg.is_clamp).float() * pim).sum(),
                "n_ks": tkf.sum(),
                "n_ks_missing": missing.float().sum(),
                "ks_kl": (kl_rows * tkf).sum(),
                "ks_ce": (ce_rows * tkf).sum(),
                "ks_loss": ks_rows.sum(),
                "n_pi_t": torch.zeros(n_types, device=vt.device).index_add_(0, dt, pim),
                "ent_t": torch.zeros(n_types, device=vt.device).index_add_(0, dt, h * pim),
                "kl_t": torch.zeros(n_types, device=vt.device).index_add_(0, dt, k3 * pim),
            }
        )
    return LossOut(loss, {k: v.detach() for k, v in st.items()}, denom)


def stats_means(sums: Mapping[str, Any]) -> dict[str, Any]:
    """Means of `ppo_losses` stats summed over chunks, minibatches and ranks.

    Scalar sums are divided by their row counts (n_v, n_pi or n_ks); the
    per-type vectors (``*_t``) by n_pi_t, NaN where a type had no rows.
    """

    def f(x: Any) -> np.ndarray:
        return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, np.float64)

    out: dict[str, Any] = {}
    n_v, n_pi = f(sums.get("n_v", 0.0)), f(sums.get("n_pi", 0.0))
    n_ks = f(sums.get("n_ks", 0.0))
    for k in ("vloss",):
        if k in sums:
            out[k] = float(f(sums[k]) / n_v) if n_v > 0 else math.nan
    for k in ("ploss", "ent", "ent_n", "ent_bonus", "kl", "clipfrac", "dualfrac", "isw"):
        if k in sums:
            out[k] = float(f(sums[k]) / n_pi) if n_pi > 0 else math.nan
    if "is_clamped" in sums:
        out["is_clamped"] = float(f(sums["is_clamped"]) / n_pi) if n_pi > 0 else math.nan
    for k in ("ks_kl", "ks_ce", "ks_loss"):
        if k in sums:
            out[k] = float(f(sums[k]) / n_ks) if n_ks > 0 else math.nan
    for k in ("n_v", "n_pi", "n_ks", "n_ks_missing"):
        if k in sums:
            out[k] = int(f(sums[k]))
    if "n_pi_t" in sums:
        cnt = f(sums["n_pi_t"])
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in ("ent_t", "kl_t"):
                if k in sums:
                    out[k] = np.where(cnt > 0, f(sums[k]) / cnt, np.nan)
        out["n_pi_t"] = cnt
    return out


def entropy_by_type(
    logits: torch.Tensor,
    n_classes: torch.Tensor,
    dtype: torch.Tensor,
    mask: torch.Tensor,
    n_types: int = N_TYPES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw entropy (nats) summed per decision type over ``mask`` rows, and the row counts.

    Feed it the pre-pass logits of student-acted trained rows; sum both over
    ranks; their ratio is the per-type mean for `EntropyCollapse.update`.
    """
    m = mask.bool()
    nc = torch.where(m, n_classes.long(), torch.full_like(n_classes.long(), 2))
    lg = torch.where(m[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    cm = class_mask(nc, logits.shape[-1])
    lp = torch.log_softmax(lg.masked_fill(~cm, NEG), dim=-1)
    h = -(torch.where(cm, lp.exp() * lp, torch.zeros_like(lp))).sum(-1)
    mf = m.float()
    dt = torch.where(m, dtype.long(), torch.zeros_like(dtype.long()))
    zs = torch.zeros(n_types, device=logits.device)
    return zs.index_add(0, dt, h * mf), zs.index_add(0, dt, mf)


def probe_kl(
    logits_new: torch.Tensor,
    logits_prox: torch.Tensor,
    n_classes: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact KL(π_prox ‖ π_θ) over merged classes, summed over ``mask`` rows, and the count.

    ``mask`` = the probe's student-acted rows. The mean (sum / count over all
    ranks) feeds `KLControl.stop` after each epoch and `KLControl.update`.
    """
    m = mask.bool()
    nc = torch.where(m, n_classes.long(), torch.full_like(n_classes.long(), 2))
    cm = class_mask(nc, logits_new.shape[-1])

    def lsm(x: torch.Tensor) -> torch.Tensor:
        x = torch.where(m[:, None], x.float(), torch.zeros_like(x, dtype=torch.float32))
        return torch.log_softmax(x.masked_fill(~cm, NEG), dim=-1)

    lo, ln = lsm(logits_prox), lsm(logits_new)
    kl = torch.where(cm, lo.exp() * (lo - ln), torch.zeros_like(lo)).sum(-1)
    mf = m.float()
    return (kl * mf).sum(), mf.sum()


# ── minibatch strata (§6 Batches) ────────────────────────────────────────


def length_strata(n_tok: np.ndarray, n_strata: int = 384) -> list[np.ndarray]:
    """The rows (indices into ``n_tok``) sorted by token length and cut into ``n_strata``.

    For 131,072 rows: 384 strata of 341 or 342 rows. Build them over the
    iteration's trained rows only.
    """
    order = np.argsort(np.asarray(n_tok), kind="stable")
    return list(np.array_split(order, n_strata))


def epoch_minibatches(
    strata: Sequence[np.ndarray],
    *,
    seed: int,
    it: int,
    epoch: int,
    per_mb: int = 24,
    world: int = 3,
) -> list[list[list[np.ndarray]]]:
    """One epoch's minibatches: ``mbs[i][rank]`` is that rank's list of strata.

    Each minibatch is ``per_mb`` random strata (``per_mb / world`` per rank),
    redrawn every epoch from (seed, it, epoch), so every rank draws the same.
    Every row appears exactly once per epoch. Pad each stratum on its own
    (it is one micro-chunk): `padding_ratio` of the strata is then the cost.
    """
    if per_mb % world:
        raise ValueError(f"per_mb {per_mb} is not a multiple of world {world}")
    if len(strata) % per_mb:
        raise ValueError(f"{len(strata)} strata do not split into minibatches of {per_mb}")
    rng = np.random.default_rng([seed, it, 0, epoch])
    perm = rng.permutation(len(strata))
    k = per_mb // world
    return [
        [[strata[j] for j in perm[i + r * k : i + (r + 1) * k]] for r in range(world)]
        for i in range(0, len(perm), per_mb)
    ]


def probe_strata(n_strata: int, *, seed: int, it: int, k: int = 24) -> np.ndarray:
    """The KL probe of iteration ``it``: ``k`` strata (8,192 rows at 131,072), fixed for its epochs."""
    rng = np.random.default_rng([seed, it, 1])
    return np.sort(rng.choice(n_strata, size=min(k, n_strata), replace=False))


def padding_ratio(n_tok: np.ndarray, groups: Iterable[np.ndarray]) -> float:
    """Padded tokens / real tokens when each group is padded to its own longest row."""
    nt = np.asarray(n_tok, np.int64)
    real = padded = 0
    for g in groups:
        if len(g):
            x = nt[g]
            real += int(x.sum())
            padded += int(x.max()) * len(g)
    return padded / real if real else 1.0


# ── EMA of the weights ───────────────────────────────────────────────────


class Ema:
    """Shadow copies of a state dict at each decay, updated once per iteration (§6 EMA).

    ``shadow ← d · shadow + (1 - d) · weights`` for floating tensors; other
    tensors (counters) are copied, and so are the `copy` keys: buffers such as
    E2ENet's frozen `prior`, which an average would shrink toward the value a
    restart replaced (review finding NT-2). 0.996 and 0.999 average over ~33M
    and ~131M decisions at 131,072 per iteration. Shadows of float tensors are
    at least fp32 and live on the device of the first state dict. Save
    `state_dict` with the run.
    """

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        decays: Sequence[float] = (0.996, 0.999),
        *,
        copy: Iterable[str] = (),
    ) -> None:
        self.decays = tuple(float(d) for d in decays)
        self.copy_keys = frozenset(copy)
        if unknown := sorted(self.copy_keys - set(state)):
            raise KeyError(f"copy keys {unknown} are not in the state dict")

        def clone(v: torch.Tensor) -> torch.Tensor:
            v = v.detach()
            if v.is_floating_point() and torch.finfo(v.dtype).bits < 32:
                return v.float()
            return v.clone()

        self.shadows: dict[str, dict[str, torch.Tensor]] = {
            self.key(d): {k: clone(v) for k, v in state.items()} for d in self.decays
        }

    @staticmethod
    def key(decay: float) -> str:
        return f"ema{decay:g}"

    @torch.no_grad()
    def update(self, state: Mapping[str, torch.Tensor]) -> None:
        for d in self.decays:
            sh = self.shadows[self.key(d)]
            if set(sh) != set(state):
                raise KeyError("the state dict's keys changed")
            for k, v in state.items():
                s = sh[k]
                x = v.detach().to(device=s.device, dtype=s.dtype)
                if s.is_floating_point() and k not in self.copy_keys:
                    s.lerp_(x, 1.0 - d)
                else:
                    s.copy_(x)

    @torch.no_grad()
    def copy_from(self, state: Mapping[str, torch.Tensor]) -> None:
        """Copy the `copy` keys of `state` into every shadow (after a load: a restart
        that filled the prior must not export the old one before the next update)."""
        for sh in self.shadows.values():
            for k in self.copy_keys:
                sh[k].copy_(state[k].detach().to(device=sh[k].device, dtype=sh[k].dtype))

    def shadow(self, decay: float) -> dict[str, torch.Tensor]:
        return self.shadows[self.key(decay)]

    def state_dict(self) -> dict[str, Any]:
        return {"decays": list(self.decays), "shadows": self.shadows}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.decays = tuple(float(x) for x in d["decays"])
        self.shadows = {k: dict(v) for k, v in d["shadows"].items()}


# ── optimizer (§6 Optimizer) ─────────────────────────────────────────────


def param_groups(
    model: torch.nn.Module,
    value_prefixes: Sequence[str] = ("value.",),
    *,
    lr_trunk: float = DEFAULT.lr_trunk,
    lr_value: float = DEFAULT.lr_value,
) -> list[dict[str, Any]]:
    """Two AdamW groups by parameter-name prefix: "trunk" (everything else) and "value".

    The trunk side is the ID table, stems, trunk and policy heads; the value
    module's parameters are those whose name starts with a ``value_prefixes``
    entry. Every trainable parameter lands in exactly one group (shared tensors
    once, under their first name). Each group keeps its ``base_lr`` for `set_lr`.
    """
    pre = tuple(value_prefixes)
    trunk: list[torch.nn.Parameter] = []
    value: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (value if name.startswith(pre) else trunk).append(p)
    if not value:
        raise ValueError(f"no parameter name starts with {pre}")
    if not trunk:
        raise ValueError("every parameter is in the value module")
    return [
        {"params": trunk, "lr": lr_trunk, "base_lr": lr_trunk, "name": "trunk"},
        {"params": value, "lr": lr_value, "base_lr": lr_value, "name": "value"},
    ]


def make_optimizer(
    model: torch.nn.Module,
    value_prefixes: Sequence[str] = ("value.",),
    *,
    cfg: ScheduleCfg = DEFAULT,
    weight_decay: float = 1e-4,
    **adamw: Any,
) -> torch.optim.AdamW:
    """One AdamW (weight decay 1e-4) over `param_groups`, at the cfg's base lrs."""
    groups = param_groups(model, value_prefixes, lr_trunk=cfg.lr_trunk, lr_value=cfg.lr_value)
    return torch.optim.AdamW(groups, lr=cfg.lr_trunk, weight_decay=weight_decay, **adamw)


def set_lr(opt: torch.optim.Optimizer, frac: float) -> dict[str, float]:
    """Set every group's lr to ``base_lr · frac`` (``sched.lr_frac``); return them, read back.

    Log the returned values (`IterSchedule.log_fields(applied=...)`).
    """
    out: dict[str, float] = {}
    for i, g in enumerate(opt.param_groups):
        g["lr"] = float(g["base_lr"]) * float(frac)
        out[str(g.get("name", i))] = float(g["lr"])
    return out


def clip_and_step(opt: torch.optim.Optimizer, max_norm: float = 1.0) -> torch.Tensor:
    """One clip_grad_norm over every parameter with a gradient, then one step.

    Call it once per minibatch, after the gradients of all chunks are
    accumulated and all-reduced. Returns the pre-clip norm (a tensor, no sync).
    """
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()
    return gn


# ── state for state.pt ───────────────────────────────────────────────────


def _pack(a: np.ndarray) -> dict[str, Any]:
    """A numpy array as types ``torch.load(weights_only=True)`` accepts."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"cannot pack dtype {a.dtype}")
    safe = a.astype(np.int64) if a.dtype.kind == "u" and a.dtype.itemsize > 1 else a
    return {"dtype": a.dtype.str, "data": torch.from_numpy(safe.copy())}


def _unpack(d: Mapping[str, Any]) -> np.ndarray:
    return d["data"].numpy().astype(np.dtype(d["dtype"]), copy=True)


class AlgoState:
    """Every piece of learner algorithm state that must survive a restart (``state.pt``).

    Holds the clocks (iteration, decisions, λ_KS hold and pass), the KL
    factor, the entropy-collapse counters, the per-type advantage moments and
    the hold-back queue. `to_dict` returns only types that
    ``torch.load(weights_only=True)`` reads back; save it between iterations
    (after `end`). EMA shadows, weights and the optimizer are saved by the
    learner (`Ema.state_dict`). A resumed session calls ``hold.resume()``
    (`HoldBack`).

    VERSION 2 added the advantage moments' per-type row counts, the clocks'
    ``ks_passed`` and the held rows' ``waited``; a version-1 state is refused.
    """

    VERSION = 2

    def __init__(
        self,
        cfg: ScheduleCfg | None = None,
        *,
        n_types: int = N_TYPES,
        teacher_types: np.ndarray | None = None,
        hold_iters: int = HOLD_ITERS,
        adv_decay: float = 5e-4,
        adv_min_rows: int = ADV_MIN_ROWS,
        lam_c: float = LAM_C,
        lam_r: float = LAM_R,
    ) -> None:
        self.cfg = cfg or ScheduleCfg()
        self.n_types = int(n_types)
        self.teacher_types = (
            np.ones(n_types, bool) if teacher_types is None else np.asarray(teacher_types, bool)
        )
        if self.teacher_types.shape != (n_types,):
            raise ValueError("teacher_types must have one entry per decision type")
        self.lam_c, self.lam_r = float(lam_c), float(lam_r)
        self.clocks = Clocks()
        self.kl = KLControl()
        self.collapse = EntropyCollapse(n_types)
        self.adv = AdvNorm(n_types, adv_decay, adv_min_rows)
        self.hold = HoldBack(hold_iters)
        self.sched: IterSchedule | None = None
        self._normalized = False

    # The iteration protocol (see the module docstring).

    def begin(self, n_collected: int, t_wall: float) -> IterSchedule:
        """Count the batch's stored rows, then fix this update's schedule."""
        if self.sched is not None:
            raise RuntimeError("end() the previous iteration first")
        self.clocks.decisions += int(n_collected)
        self.sched = schedule_for(
            self.clocks, t_wall, self.kl.m, self.collapse, self.cfg, self.teacher_types
        )
        self._normalized = False
        return self.sched

    def join(self, fresh: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return self.hold.join(fresh)

    def returns(self, table: Mapping[str, Any], values: np.ndarray) -> Returns:
        return compute_returns(
            table, values, lam_c=self.lam_c, lam_r=self.lam_r, hold_iters=self.hold.hold_iters
        )

    def normalize(self, table: Mapping[str, Any], ret: Returns) -> np.ndarray:
        """Normalised advantages of ``ret.pi`` rows (NaN elsewhere); updates the moments once.

        The rows' ``is_combat`` picks the family pool of a type still warming up (`AdvNorm`).
        """
        if self._normalized:
            raise RuntimeError("normalize() runs once per iteration")
        self._normalized = True
        return self.adv.update_normalize(
            ret.adv, _col(table, "dtype", np.int64), ret.pi, _col(table, "is_combat", bool)
        )

    def end(
        self,
        table: Mapping[str, Any],
        ret: Returns,
        *,
        probe_kl: float | None = None,
        entropy_per_type: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Close the iteration: carry held rows, adapt m, count collapses, advance ``it``."""
        s = self.sched
        if s is None:
            raise RuntimeError("begin() first")
        self.hold.carry(table, ret)
        m0 = self.kl.m
        m1 = self.kl.update(probe_kl, s, self.cfg)
        fired = self.collapse.update(entropy_per_type, s, self.cfg)
        self.clocks.it += 1
        self.sched = None
        return {"kl_m_before": m0, "kl_m": m1, "collapsed_types": fired.tolist()}

    # Serialisation.

    def to_dict(self) -> dict[str, Any]:
        if self.sched is not None:
            raise RuntimeError("save between iterations (after end())")
        return {
            "version": self.VERSION,
            "cfg": self.cfg.to_dict(),
            "n_types": self.n_types,
            "teacher_types": self.teacher_types.tolist(),
            "lam_c": self.lam_c,
            "lam_r": self.lam_r,
            "clocks": self.clocks.to_dict(),
            "kl": self.kl.to_dict(),
            "collapse": self.collapse.to_dict(),
            "adv": self.adv.to_dict(),
            "hold": self.hold.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], cfg: ScheduleCfg | None = None) -> AlgoState:
        """Restore a saved state. ``cfg`` overrides the saved schedule constants (a deliberate change)."""
        if int(d["version"]) != cls.VERSION:
            raise ValueError(f"state version {d['version']}, this code reads {cls.VERSION}")
        s = cls(
            cfg or ScheduleCfg.from_dict(d["cfg"]),
            n_types=int(d["n_types"]),
            teacher_types=np.asarray(d["teacher_types"], bool),
            lam_c=float(d["lam_c"]),
            lam_r=float(d["lam_r"]),
        )
        s.clocks = Clocks.from_dict(d["clocks"])
        s.kl = KLControl.from_dict(d["kl"])
        s.collapse = EntropyCollapse.from_dict(d["collapse"])
        s.adv = AdvNorm.from_dict(d["adv"])
        s.hold = HoldBack.from_dict(d["hold"])
        return s
