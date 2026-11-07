"""PREREG-0 P1: the campaign's evaluation panels and seed ranges, and the row validator.

Every range here is copied from P1 of
``docs/experiments/2026-11-03-ironclad-e2e-v111/PREREG.md``. ``scripts/e2e_arm.sh``
keeps its own copy of the runner panels (its ``case $PANEL``); a test reads the
script and checks that the two agree, so neither can drift alone.

Runner panels (:data:`RUNNER_PANELS`) are what the baseline runner and
``ppo.e2e.evaluate`` take as ``PANEL`` and use as directory names under
``~/runs/e2e/evals/``. ``DEV`` is the DEV-2000 range; DEV-1000 is its first
1,000 seeds. ``SMOKE_SEALED`` is SMOKE's seeds with TEST's sealed output, so the
sealing can be tested without touching a TEST seed.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

#: P1: every seed in the campaign is below 2**31.
SEED_LIMIT = 2**31


@dataclass(frozen=True)
class Panel:
    """A contiguous seed range [start, start + n)."""

    name: str
    start: int
    n: int
    sealed: bool = False

    @property
    def stop(self) -> int:
        return self.start + self.n

    def seeds(self) -> range:
        return range(self.start, self.stop)

    def __contains__(self, seed: object) -> bool:
        return (
            isinstance(seed, int) and not isinstance(seed, bool) and self.start <= seed < self.stop
        )


DEV_1000 = Panel("DEV-1000", 160_000_000, 1_000)
DEV_2000 = Panel("DEV-2000", 160_000_000, 2_000)
TEST = Panel("TEST", 161_000_000, 4_000, sealed=True)
SMOKE = Panel("SMOKE", 162_000_000, 1_000)

#: The PANEL names of `scripts/e2e_arm.sh` and `ppo.e2e.evaluate`.
RUNNER_PANELS: dict[str, Panel] = {
    "DEV": Panel("DEV", DEV_2000.start, DEV_2000.n),
    "TEST": TEST,
    "SMOKE": SMOKE,
    "SMOKE_SEALED": Panel("SMOKE_SEALED", SMOKE.start, SMOKE.n, sealed=True),
}

#: Rows per runner shard (`e2e_arm.sh`: 250-seed shards that skip when complete).
SHARD_SIZE = 250

# DEVBOSS (plan §9, §11): 600 boss-entry states from teacher-acted runs on seeds
# 175,000,000+, 200 per cell, each played from 4 shared reseeds.
DEVBOSS_RUN_SEED_START = 175_000_000
#: Capture runs stay below the reseed block.
DEVBOSS_RUN_SEED_STOP = 175_500_000
DEVBOSS_RESEED_START = 175_500_000
DEVBOSS_CELLS: tuple[str, ...] = ("act1_boss", "act2_boss", "act3_double")
DEVBOSS_STATES_PER_CELL = 200
DEVBOSS_N_STATES = DEVBOSS_STATES_PER_CELL * len(DEVBOSS_CELLS)
DEVBOSS_RESEEDS_PER_STATE = 4

#: P1's other ranges (no panel reads them; listed so the overlap test covers them).
PRACTICE_RESEEDS = range(600_000_000, 640_000_000)
STUDENT_TRAINER_SEEDS = range(1_000_000_000, 1_260_000_000)
THROWAWAY_TRAINER_SEEDS = range(1_300_000_000, 1_560_000_000)

#: P4: nobody reads `evals/TEST/` before Thu Nov 12, 09:00 CET (08:00 UTC); was Mon Nov 9 until
#: PREREG Amendment 5 moved the end of training by 72 h.
UNSEAL_AT = dt.datetime(2026, 11, 12, 8, 0, tzinfo=dt.UTC)

#: P2: more than 0.5% such seeds (20 of TEST's 4,000) make a contrast INVALID.
MAX_BAD_FRACTION = 0.005


def runner_panel(name: str) -> Panel:
    """The runner panel called `name` (DEV, TEST, SMOKE or SMOKE_SEALED)."""
    try:
        return RUNNER_PANELS[name]
    except KeyError:
        raise PanelError(
            f"unknown panel {name!r}; expected one of {', '.join(RUNNER_PANELS)}"
        ) from None


def max_bad_seeds(n_seeds: int) -> int:
    """P2's INVALID threshold for a contrast over `n_seeds` paired seeds: 20 at 4,000."""
    return int(n_seeds * MAX_BAD_FRACTION)


def devboss_reseed(i: int, j: int) -> int:
    """The shared reseed of DEVBOSS state `i`, replay `j`: 175,500,000 + 4i + j."""
    if not 0 <= i < DEVBOSS_N_STATES:
        raise ValueError(f"DEVBOSS state index {i} outside [0, {DEVBOSS_N_STATES})")
    if not 0 <= j < DEVBOSS_RESEEDS_PER_STATE:
        raise ValueError(f"DEVBOSS reseed index {j} outside [0, {DEVBOSS_RESEEDS_PER_STATE})")
    return DEVBOSS_RESEED_START + DEVBOSS_RESEEDS_PER_STATE * i + j


class PanelError(ValueError):
    """Seeds or rows that break a panel's rules."""


def check_slice(panel: Panel, start: int, n: int) -> None:
    """Raise unless [start, start + n) is a non-empty slice of `panel` (as e2e_arm.sh checks)."""
    if n < 1:
        raise PanelError(f"n must be positive, got {n}")
    if not (panel.start <= start and start + n <= panel.stop):
        raise PanelError(
            f"seeds {start}..{start + n - 1} leave the {panel.name} panel "
            f"[{panel.start}, {panel.stop})"
        )


@dataclass(frozen=True)
class SeedCheck:
    """What `check_seeds` found. `ok` iff the rows are exactly the expected seeds, once each."""

    panel: str
    start: int
    expected: int
    rows: int
    non_int: int
    off_panel: tuple[int, ...]
    out_of_slice: tuple[int, ...]
    duplicates: tuple[int, ...]
    missing: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return (
            self.rows == self.expected
            and not self.non_int
            and not self.off_panel
            and not self.out_of_slice
            and not self.duplicates
            and not self.missing
        )

    def describe(self) -> str:
        if self.ok:
            return (
                f"{self.rows} rows, seeds {self.start}..{self.start + self.expected - 1} once each"
            )

        def few(xs: tuple[int, ...]) -> str:
            head = ", ".join(str(x) for x in xs[:5])
            return head + (f", ... ({len(xs)} in all)" if len(xs) > 5 else "")

        parts = [f"{self.rows} rows, expected {self.expected}"]
        if self.non_int:
            parts.append(f"{self.non_int} row(s) without an integer seed")
        if self.off_panel:
            parts.append(f"off-panel seed(s) {few(self.off_panel)}")
        if self.out_of_slice:
            parts.append(f"seed(s) outside the run's slice {few(self.out_of_slice)}")
        if self.duplicates:
            parts.append(f"duplicate seed(s) {few(self.duplicates)}")
        if self.missing:
            parts.append(f"missing seed(s) {few(self.missing)}")
        return f"{self.panel}: " + "; ".join(parts)


def check_seeds(
    seeds: Iterable[object], panel: Panel, *, start: int | None = None, n: int | None = None
) -> SeedCheck:
    """Check one row's seed per item against the slice [start, start + n) of `panel`.

    The default slice is the whole panel. Off-panel seeds (outside the panel)
    and in-panel seeds outside the slice are reported separately.
    """
    start = panel.start if start is None else start
    n = panel.n if n is None else n
    check_slice(panel, start, n)
    stop = start + n
    counts: Counter[int] = Counter()
    rows = non_int = 0
    for s in seeds:
        rows += 1
        if not isinstance(s, int) or isinstance(s, bool):
            non_int += 1
            continue
        counts[s] += 1
    off_panel = tuple(sorted(s for s in counts if s not in panel))
    out_of_slice = tuple(sorted(s for s in counts if s in panel and not start <= s < stop))
    duplicates = tuple(sorted(s for s, c in counts.items() if c > 1))
    missing = tuple(s for s in range(start, stop) if s not in counts)
    return SeedCheck(
        panel=panel.name,
        start=start,
        expected=n,
        rows=rows,
        non_int=non_int,
        off_panel=off_panel,
        out_of_slice=out_of_slice,
        duplicates=duplicates,
        missing=missing,
    )


def validate_seeds(
    seeds: Iterable[object], panel: Panel, *, start: int | None = None, n: int | None = None
) -> SeedCheck:
    """`check_seeds`, raising `PanelError` unless the rows are exactly the slice."""
    check = check_seeds(seeds, panel, start=start, n=n)
    if not check.ok:
        raise PanelError(check.describe())
    return check
