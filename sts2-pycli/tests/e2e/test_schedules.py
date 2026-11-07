"""Closed-form tests of the e2e schedules (plan §6 to §8; `ppo.e2e.schedules`).

Every schedule value is checked at chosen clocks against the plan's numbers to
1e-9: the lr anneal and warmup, λ_KS with its hold and offset, the entropy
coefficients with their gate and collapse boosts, β_act, the KL-adaptive factor
and the watchdog's closed-form lr check.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
import torch
from ppo.e2e import algo
from ppo.e2e import schedules as S

H = 3600.0
M = 1_000_000
CFG = S.ScheduleCfg()


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _exact(a: float, b: float) -> bool:
    """Bitwise-exact float equality (the schedules return these values unrounded)."""
    return abs(a - b) <= 0.0


def _sched(
    it: int = 50,
    decisions: int = 0,
    t_wall: float = S.ANNEAL_START - 48 * H,
    m: float = 1.0,
    hold: bool = False,
    offset: int = 0,
    collapse: S.EntropyCollapse | None = None,
    teacher_types: np.ndarray | None = None,
) -> S.IterSchedule:
    c = S.Clocks(it=it, decisions=decisions, ks_hold=hold, ks_offset=offset)
    return S.schedule_for(c, t_wall, m, collapse or S.EntropyCollapse(), CFG, teacher_types)


# ── wall clock: anneal, KL freeze ────────────────────────────────────────


def test_anneal_start_is_sunday_nov_8_0600_cet() -> None:
    assert _exact(S.ANNEAL_START, 1_794_114_000.0)
    assert datetime.fromtimestamp(S.ANNEAL_START, UTC) == datetime(2026, 11, 8, 5, 0, tzinfo=UTC)
    assert datetime.fromtimestamp(S.ANNEAL_START, S.CET).strftime("%a %d %b %H:%M") == (
        "Sun 08 Nov 06:00"
    )
    assert CFG.kl_freeze_at == S.ANNEAL_START


@pytest.mark.parametrize(
    ("dt_hours", "want"),
    [
        (-100, 1.0),
        (-1e-6, 1.0),
        (0, 1.0),
        (6, 0.825),
        (12, 0.65),
        (18, 0.475),
        (24, 0.3),
        (99, 0.3),
    ],
)
def test_anneal_closed_form(dt_hours: float, want: float) -> None:
    # anneal(t) = 1 - 0.7 · clamp((t - Sun 06:00) / 24 h, 0, 1)
    got = S.anneal(S.ANNEAL_START + dt_hours * H)
    assert _close(got, want), (dt_hours, got, want)
    assert _close(got, 1 - 0.7 * min(1.0, max(0.0, dt_hours / 24.0)))


# ── iteration clock: warmup, V-warm ──────────────────────────────────────


def test_warmup_and_value_only_iterations() -> None:
    assert CFG.lr_warmup_iters == 10
    assert CFG.value_only_iters == 3
    for it, want in [(0, 0.1), (1, 0.2), (4, 0.5), (8, 0.9), (9, 1.0), (10, 1.0), (10_000, 1.0)]:
        assert _close(S.warmup(it), want), (it, S.warmup(it))
    assert [S.policy_on(i) for i in range(5)] == [False, False, False, True, True]
    assert [S.phase(i, 0) for i in range(4)] == ["V-warm"] * 3 + ["U0"]


@pytest.mark.parametrize(
    ("it", "dt_hours", "m", "frac"),
    [
        (0, -10, 1.0, 0.1),
        (4, 6, 1.5, 1.5 * 0.825 * 0.5),
        (9, 12, 0.3, 0.3 * 0.65),
        (500, 30, 2.0, 2.0 * 0.3),
        (500, -1, 1.0, 1.0),
    ],
)
def test_lr_closed_form(it: int, dt_hours: float, m: float, frac: float) -> None:
    # lr(t) = base · m · anneal(t) · warmup(it), for both AdamW groups.
    t = S.ANNEAL_START + dt_hours * H
    assert _close(S.lr_frac(t, it, m), frac)
    assert _close(S.lr(1e-4, t, it, m), 1e-4 * frac, 1e-15)
    assert _close(S.lr(3e-4, t, it, m), 3e-4 * frac, 1e-15)
    s = _sched(it=it, t_wall=t, m=m, decisions=200 * M)
    assert _close(s.lr_frac, frac)
    assert _close(s.lr_trunk, 1e-4 * frac, 1e-15)
    assert _close(s.lr_value, 3e-4 * frac, 1e-15)


# ── decision clocks: β_act, λ_KS, entropy ────────────────────────────────


@pytest.mark.parametrize(
    ("d", "want"), [(0, 1.0), (20 * M, 0.75), (40 * M, 0.5), (80 * M, 0.0), (10**9, 0.0)]
)
def test_beta_act_closed_form(d: int, want: float) -> None:
    assert _close(S.beta_act(d), want)


@pytest.mark.parametrize(
    ("d", "want"),
    [
        (0, 1.0),
        (80 * M, 1.0),
        (185 * M, 0.75),
        (290 * M, 0.5),
        (374 * M, 0.3),
        (458 * M, 0.1),
        (500 * M, 0.0),
        (2 * 10**9, 0.0),
    ],
)
def test_lam_ks_closed_form(d: int, want: float) -> None:
    # λ_KS: 1 until 80M, then linear to 0 at 500M (0.3 at 374M).
    assert _close(S.lam_ks(d), want)
    assert _close(S.lam_ks(d, hold=True), 1.0)


def test_lam_ks_hold_then_late_pass_shifts_the_ramp() -> None:
    c = S.Clocks()
    c.hold_ks()  # the learning check failed at launch
    c.decisions = 150 * M
    assert _close(S.lam_ks(c.decisions, hold=c.ks_hold, offset=c.ks_offset), 1.0)
    assert S.phase(10, c.decisions, hold=c.ks_hold) == "U1"
    c.pass_learning_check()  # passes at 150M: the decay starts now
    assert not c.ks_hold
    assert c.ks_offset == 70 * M
    for d, want in [(150 * M, 1.0), (360 * M, 0.5), (444 * M, 0.3), (570 * M, 0.0)]:
        assert _close(S.lam_ks(d, offset=c.ks_offset), want), d
    # The entropy schedule moves with it: gate at 374M + 70M, end at 4B + 70M.
    mid = (374 * M + 4 * 10**9) // 2
    assert _close(S.entropy_beta(444 * M, True, offset=c.ks_offset), 0.01)
    assert _close(S.entropy_beta(mid + 70 * M, True, offset=c.ks_offset), math.sqrt(0.01 * 0.002))
    assert _close(S.entropy_beta(4070 * M, False, offset=c.ks_offset), 0.006)
    assert S.entropy_beta(mid, True, offset=c.ks_offset) > S.entropy_beta(mid, True)
    # The schedule of an update sees the shifted clock.
    c.decisions, c.it = 444 * M, 40
    s = S.schedule_for(c, S.ANNEAL_START - 48 * H, 1.0, S.EntropyCollapse(), CFG)
    assert s.ks_decisions == 374 * M and _close(s.lam_ks, 0.3) and _exact(s.ent_gain[0], 1.0)
    assert _close(s.beta_act, 0.0) and s.phase == "U1"
    # A check that passes before 80M leaves the plan's ramp.
    c2 = S.Clocks(decisions=30 * M, ks_hold=True)
    c2.pass_learning_check()
    assert c2.ks_offset == 0
    assert _close(S.lam_ks(374 * M, offset=c2.ks_offset), 0.3)


def test_a_failed_recheck_after_a_pass_keeps_the_ramp() -> None:
    """The check re-runs at K2 and at each restart (§7). Once one has passed, a failure
    must not reset λ_KS to 1 and shift the ramp: hold_ks leaves the clocks alone and
    returns False, for the learner's OPERATOR alarm. Before any pass, it holds as before."""
    c = S.Clocks()
    assert c.hold_ks() is True and c.ks_hold and not c.ks_passed  # fails at launch: held
    assert c.hold_ks() is True and c.ks_hold  # fails again (a restart): still held
    c.decisions = 150 * M
    c.pass_learning_check()  # passes at 150M: the ramp starts there
    assert c.ks_passed and not c.ks_hold and c.ks_offset == 70 * M
    c.it, c.decisions = 40, 360 * M  # a restart re-runs the check, and it fails
    before = c.to_dict()
    assert c.hold_ks() is False and c.to_dict() == before
    assert _close(S.lam_ks(c.decisions, hold=c.ks_hold, offset=c.ks_offset), 0.5)
    s = S.schedule_for(c, S.ANNEAL_START - 48 * H, 1.0, S.EntropyCollapse(), CFG)
    assert _close(s.lam_ks, 0.5) and not s.ks_hold and s.ks_passed and s.phase == "U1"
    assert s.log_fields()["ks_passed"] is True and s.log_fields()["ks_offset"] == 70 * M
    c.pass_learning_check()  # a later pass leaves the ramp where the first one put it
    assert c.to_dict() == before
    # The pass persists: restored clocks refuse the hold too.
    c2 = S.Clocks.from_dict(c.to_dict())
    assert c2 == c and c2.ks_passed and c2.hold_ks() is False and c2 == c
    # A pass at launch (before 80M) counts too: K2's failed re-check keeps the plan's ramp.
    c3 = S.Clocks()
    c3.pass_learning_check()
    assert c3.ks_passed and c3.ks_offset == 0 and not c3.ks_hold
    c3.decisions = 100 * M
    assert c3.hold_ks() is False and not c3.ks_hold and c3.ks_offset == 0
    for d, want in [(100 * M, 1.0 - 20 / 420), (185 * M, 0.75), (500 * M, 0.0)]:
        assert _close(S.lam_ks(d, hold=c3.ks_hold, offset=c3.ks_offset), want), d
    # Never passed: the schedule logs it.
    assert _sched(hold=True).log_fields()["ks_passed"] is False


def test_entropy_beta_family_schedules() -> None:
    assert S.ent_gate_decisions() == 374 * M
    g, end = 374 * M, 4 * 10**9
    mid = (g + end) // 2
    for combat, (b0, b1) in ((True, (0.01, 0.002)), (False, (0.0125, 0.006))):
        assert _close(S.entropy_beta(0, combat), b0)
        assert _close(S.entropy_beta(g, combat), b0)
        assert _close(S.entropy_beta(mid, combat), math.sqrt(b0 * b1))
        assert _close(S.entropy_beta(end, combat), b1)
        assert _close(S.entropy_beta(10 * end, combat), b1)
        # geometric: equal ratios over equal spans
        q = (end - g) // 4
        r = [S.entropy_beta(g + i * q, combat) for i in range(5)]
        assert all(_close(r[i + 1] / r[i], (b1 / b0) ** 0.25) for i in range(4))


def test_entropy_gate_per_type() -> None:
    teach = np.ones(256, bool)
    teach[15] = False  # a teacherless type: λ_KS = 0, gate open from the start
    dt = np.array([0, 15, 0, 15])
    comb = np.array([True, True, False, False])
    for d in (0, 100 * M, 373_999_999):
        s = _sched(decisions=d, teacher_types=teach)
        assert _exact(s.ent_gain[0], 0.0) and _exact(s.ent_gain[15], 1.0), d
        assert _close(s.ks_types[0], S.lam_ks(d)) and _exact(s.ks_types[15], 0.0)
        beta = s.beta_rows(dt, comb)
        assert _exact(beta[0], 0.0) and _exact(beta[2], 0.0)
        assert _close(beta[1], s.beta_combat) and _close(beta[3], s.beta_ooc)
    s = _sched(decisions=374 * M, teacher_types=teach)  # λ_KS = 0.3: the gate opens
    assert _exact(s.ent_gain[0], 1.0)
    assert np.allclose(s.beta_rows(dt, comb), [0.01, 0.01, 0.0125, 0.0125], rtol=0, atol=1e-12)
    # A held λ_KS keeps the gate shut whatever the decision count.
    s = _sched(decisions=900 * M, hold=True, teacher_types=teach)
    assert _exact(s.ent_gain[0], 0.0) and _exact(s.lam_ks, 1.0) and s.phase == "U1"
    # Teacher rows only get λ_KS where the row has teacher output.
    s = _sched(decisions=290 * M, teacher_types=teach)
    ks = s.ks_rows(np.array([0, 0, 15]), np.array([True, False, True]))
    assert np.allclose(ks, [0.5, 0.0, 0.0], rtol=0, atol=1e-12)


def test_entropy_collapse_lifts_gate_and_boosts_at_most_twice() -> None:
    cfg, col = CFG, S.EntropyCollapse()
    s_on = _sched(decisions=100 * M)  # gated (λ_KS 0.95), policy training
    s_vwarm = _sched(it=0, decisions=0)
    ent = np.full(256, np.nan)
    ent[3], ent[4] = 0.05, 0.5
    # V-warm does not count.
    for _ in range(30):
        assert col.update(ent, s_vwarm, cfg).size == 0
    assert col.low[3] == 0
    fired_at = []
    for i in range(70):
        if i == 10:  # an iteration without type-3 rows leaves the streak alone
            e = ent.copy()
            e[3] = np.nan
            col.update(e, s_on, cfg)
            continue
        if col.update(ent, s_on, cfg).tolist() == [3]:
            fired_at.append(i)
    assert fired_at == [20, 40, 60]
    assert col.boosts[3] == 2 and col.lifted[3]  # at most twice
    assert col.boosts[4] == 0 and not col.lifted[4]
    s = S.schedule_for(S.Clocks(it=50, decisions=100 * M), S.ANNEAL_START - 48 * H, 1.0, col, cfg)
    assert _close(s.ent_gain[3], 2.25)  # lifted while λ_KS > 0.3, x1.5 twice
    assert _exact(s.ent_gain[4], 0.0)
    # A reading at or above the threshold resets the streak.
    c2 = S.EntropyCollapse()
    e = np.full(256, np.nan)
    for i in range(25):
        e[7] = 0.05 if i != 12 else 0.2
        c2.update(e, s_on, cfg)
    assert c2.low[7] == 12 and c2.boosts[7] == 0


def test_kl_factor_rules() -> None:
    cfg = CFG
    t0 = S.ANNEAL_START - 10 * H
    vwarm = _sched(it=1, decisions=0, t_wall=t0)
    u0 = _sched(decisions=50 * M, t_wall=t0)
    u1 = _sched(decisions=200 * M, t_wall=t0)
    u2 = _sched(decisions=600 * M, t_wall=t0)
    frozen = _sched(decisions=600 * M, t_wall=S.ANNEAL_START)
    assert (vwarm.phase, u0.phase, u1.phase, u2.phase) == ("V-warm", "U0", "U1", "U2")
    kl = S.KLControl()
    for s in (vwarm, u0):  # no KL control in V-warm and U0
        assert not s.kl_control
        assert not kl.stop(10.0, s, cfg)
        assert _exact(kl.update(1e-6, s, cfg), 1.0) and _exact(kl.update(10.0, s, cfg), 1.0)
    for s in (u1, u2):
        assert s.kl_control
        assert kl.stop(0.0601, s, cfg) and not kl.stop(0.06, s, cfg)
    assert _close(kl.update(0.004, u1, cfg), 1.5)
    assert _close(kl.update(0.01, u1, cfg), 1.5)  # inside [0.005, 0.02]: unchanged
    assert _close(kl.update(0.0049, u2, cfg), 2.0)  # 2.25 clamped to 2
    assert _close(kl.update(0.03, u1, cfg), 2.0 / 1.5)
    for _ in range(10):
        kl.update(0.5, u1, cfg)
    assert _close(kl.m, 0.3)  # floor
    assert kl.update(0.001, frozen, cfg) == kl.m  # frozen from Sun 06:00
    assert kl.update(None, u1, cfg) == kl.m and kl.update(math.nan, u1, cfg) == kl.m


def test_phase_boundaries_and_budget() -> None:
    assert S.phase(3, 80 * M - 1) == "U0"
    assert S.phase(3, 80 * M) == "U1"
    assert S.phase(3, 500 * M - 1) == "U1"
    assert S.phase(3, 500 * M) == "U2"
    assert S.phase(3, 900 * M, hold=True) == "U1"
    assert S.phase(3, 550 * M, offset=60 * M) == "U1"
    assert CFG.budget_decisions >= 10**10
    assert S.Clocks(decisions=10**10 - 1).more() and not S.Clocks(decisions=10**10).more()


# ── the watchdog's lr check ──────────────────────────────────────────────


def _logged_row(it: int = 20, dt_hours: float = 7.5, m: float = 1.5) -> dict:
    """A stats row as the learner writes it: the lrs read back from a real AdamW."""
    net = torch.nn.ModuleDict({"trunk": torch.nn.Linear(2, 2), "value": torch.nn.Linear(2, 1)})
    opt = algo.make_optimizer(net, ("value.",))
    s = _sched(it=it, decisions=300 * M, t_wall=S.ANNEAL_START + dt_hours * H, m=m)
    return s.log_fields(applied=algo.set_lr(opt, s.lr_frac))


def test_check_lr_row_accepts_the_closed_form() -> None:
    for it, dt, m in [(0, -5, 1.0), (20, 7.5, 1.5), (3000, 40, 0.3), (5, 23.9, 2.0)]:
        row = _logged_row(it, dt, m)
        assert S.check_lr_row(row) == [], (it, dt, m)


@pytest.mark.parametrize(
    ("key", "delta", "bad"),
    [
        ("lr_frac", 2e-6, True),
        ("lr_frac", 5e-7, False),
        ("lr_trunk", 2e-6 * 1e-4, True),
        ("lr_value", 2e-6 * 3e-4, True),
        ("lr_value", 5e-7 * 3e-4, False),
        ("anneal", 1e-5, True),
        ("t_wall", 3600.0, True),  # an hour off: anneal no longer matches
        ("it", -15, True),  # warmup no longer matches
        ("kl_m", 0.01, True),
        ("lr_base_trunk", 1e-5, True),
    ],
)
def test_check_lr_row_flags_drift(key: str, delta: float, bad: bool) -> None:
    row = _logged_row()
    row[key] = row[key] + delta
    assert bool(S.check_lr_row(row)) is bad, (key, S.check_lr_row(row))


def test_check_lr_row_flags_missing_nan_and_bounds() -> None:
    row = _logged_row()
    del row["lr_value"]
    assert "missing" in S.check_lr_row(row)[0]
    row = _logged_row()
    row["lr_trunk"] = math.nan
    assert S.check_lr_row(row)
    row = _logged_row(m=2.5)  # m outside [0.3, 2] (a broken clamp)
    assert any("outside" in e for e in S.check_lr_row(row))


def test_lr_watch_flags_a_kl_factor_change_after_the_freeze() -> None:
    w = S.LrWatch()
    rows = [_logged_row(it=i, dt_hours=h, m=m) for i, h, m in [(7, -2, 1.0), (8, -1, 1.5)]]
    assert w.check_all(rows) == []  # m may change before the freeze
    w = S.LrWatch()
    rows = [_logged_row(it=i, dt_hours=h, m=m) for i, h, m in [(7, 0.1, 1.5), (8, 0.2, 1.5)]]
    assert w.check_all(rows) == []
    w = S.LrWatch()
    rows = [_logged_row(it=i, dt_hours=h, m=m) for i, h, m in [(7, 0.1, 1.5), (8, 0.2, 1.0)]]
    bad = w.check_all(rows)
    assert len(bad) == 1 and bad[0][0] == 1 and "freeze" in bad[0][1][0]


def test_schedule_state_round_trips() -> None:
    c = S.Clocks(it=12, decisions=123_456_789, ks_hold=True, ks_offset=5)
    assert S.Clocks.from_dict(c.to_dict()) == c
    p = S.Clocks(it=3, decisions=9, ks_offset=7, ks_passed=True)
    assert S.Clocks.from_dict(p.to_dict()) == p and S.Clocks.from_dict(p.to_dict()).ks_passed
    assert S.ScheduleCfg.from_dict(CFG.to_dict()) == CFG
    k = S.KLControl(0.75)
    assert S.KLControl.from_dict(k.to_dict()) == k
    col = S.EntropyCollapse()
    col.low[3], col.boosts[9], col.lifted[9] = 4, 1, True
    col2 = S.EntropyCollapse.from_dict(col.to_dict())
    assert np.array_equal(col2.low, col.low) and np.array_equal(col2.boosts, col.boosts)
    assert np.array_equal(col2.lifted, col.lifted)
