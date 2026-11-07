"""PREREG-0 P1 panels (ppo.e2e.panels): ranges, validator, and agreement with e2e_arm.sh."""

import itertools
import re
from pathlib import Path

import pytest
from ppo.e2e import panels as P

PYCLI = Path(__file__).resolve().parents[2]
ARM_SH = PYCLI / "scripts" / "e2e_arm.sh"


def test_prereg_p1_ranges():
    assert (P.DEV_1000.start, P.DEV_1000.stop) == (160_000_000, 160_001_000)
    assert (P.DEV_2000.start, P.DEV_2000.stop) == (160_000_000, 160_002_000)
    assert (P.TEST.start, P.TEST.stop) == (161_000_000, 161_004_000)
    assert (P.SMOKE.start, P.SMOKE.stop) == (162_000_000, 162_001_000)
    assert P.TEST.sealed and P.RUNNER_PANELS["SMOKE_SEALED"].sealed
    assert not P.DEV_2000.sealed and not P.SMOKE.sealed
    assert P.RUNNER_PANELS["DEV"].seeds() == P.DEV_2000.seeds()


def test_devboss_reseeds():
    assert P.DEVBOSS_N_STATES == 600
    assert P.devboss_reseed(0, 0) == 175_500_000
    assert P.devboss_reseed(1, 0) == 175_500_004
    assert P.devboss_reseed(599, 3) == 175_502_399
    every = {P.devboss_reseed(i, j) for i in range(600) for j in range(4)}
    assert len(every) == 2400
    for bad in ((600, 0), (-1, 0), (0, 4), (0, -1)):
        with pytest.raises(ValueError):
            P.devboss_reseed(*bad)


def test_ranges_are_disjoint_and_below_2_31():
    ranges = {
        "DEV": P.DEV_2000.seeds(),
        "TEST": P.TEST.seeds(),
        "SMOKE": P.SMOKE.seeds(),
        "DEVBOSS runs": range(P.DEVBOSS_RUN_SEED_START, P.DEVBOSS_RUN_SEED_STOP),
        "DEVBOSS reseeds": range(
            P.DEVBOSS_RESEED_START,
            P.DEVBOSS_RESEED_START + P.DEVBOSS_N_STATES * P.DEVBOSS_RESEEDS_PER_STATE,
        ),
        "practice": P.PRACTICE_RESEEDS,
        "student": P.STUDENT_TRAINER_SEEDS,
        "throwaway": P.THROWAWAY_TRAINER_SEEDS,
    }
    items = sorted(ranges.items(), key=lambda kv: kv[1].start)
    for (na, a), (nb, b) in itertools.pairwise(items):
        assert a.stop <= b.start, f"{na} overlaps {nb}"
    assert all(r.stop <= P.SEED_LIMIT for r in ranges.values())


def test_max_bad_is_twenty_on_test():
    assert P.max_bad_seeds(4000) == 20
    assert P.max_bad_seeds(2000) == 10
    assert P.max_bad_seeds(199) == 0


def test_validator_accepts_exact_panel():
    check = P.validate_seeds(list(P.TEST.seeds())[::-1], P.TEST)
    assert check.ok and check.rows == 4000


def test_validator_flags_each_failure():
    seeds = list(P.TEST.seeds())
    short = P.check_seeds(seeds[:-1], P.TEST)
    assert not short.ok and short.missing == (161_003_999,)
    dup = P.check_seeds([*seeds[:-1], seeds[0]], P.TEST)
    assert not dup.ok and dup.duplicates == (161_000_000,) and dup.rows == 4000
    off = P.check_seeds([*seeds[:-1], 160_000_000], P.TEST)
    assert not off.ok and off.off_panel == (160_000_000,)
    junk = P.check_seeds([*seeds[:-1], "161003999"], P.TEST)
    assert not junk.ok and junk.non_int == 1
    with pytest.raises(P.PanelError, match="off-panel"):
        P.validate_seeds([*seeds[:-1], 999], P.TEST)


def test_validator_on_a_slice():
    dev = P.RUNNER_PANELS["DEV"]
    ok = P.check_seeds(range(160_000_000, 160_001_000), dev, n=1000)
    assert ok.ok
    extra = P.check_seeds(range(160_000_000, 160_001_001), dev, n=1000)
    assert not extra.ok and extra.out_of_slice == (160_001_000,) and not extra.off_panel
    with pytest.raises(P.PanelError, match="leave the DEV panel"):
        P.check_slice(dev, 160_001_500, 1000)
    with pytest.raises(P.PanelError):
        P.check_slice(dev, 160_000_000, 0)


def test_panel_membership_rejects_bools_and_strings():
    assert 161_000_000 in P.TEST
    assert "161000000" not in P.TEST
    assert True not in P.Panel("tiny", 0, 2)


@pytest.mark.skipif(not ARM_SH.is_file(), reason="scripts/e2e_arm.sh not present")
def test_runner_script_agrees_with_panels():
    text = ARM_SH.read_text(encoding="utf-8")
    found = {}
    for m in re.finditer(r"^\s*([A-Z_ |]+)\)\s*LO=(\d+)\s+HI=(\d+)", text, re.M):
        for name in m.group(1).split("|"):
            found[name.strip()] = (int(m.group(2)), int(m.group(3)))
    assert found == {k: (p.start, p.stop) for k, p in P.RUNNER_PANELS.items()}
    sealed_line = re.search(r"^\[\[ (.*) \]\] && SEALED=1$", text, re.M)
    assert sealed_line
    sealed = set(re.findall(r"\$PANEL == (\w+)", sealed_line.group(1)))
    assert sealed == {k for k, p in P.RUNNER_PANELS.items() if p.sealed}
    assert f"lo += {P.SHARD_SIZE}" in text
