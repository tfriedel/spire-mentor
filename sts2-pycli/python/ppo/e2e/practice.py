"""Practice on the student's own runs (plan §3.3, WP2). Switchable: practice is first in
the cut order and may enter at a restart, so a worker with `enabled=False` plays full runs
only and nothing else changes.

**Capture.** A full run hands `PracticeQueues.add` an `env.Capture` before every Move into
an elite or boss room: `rm.clone()` taken just before the Move, plus the Move. Six cells,
named as in `contracts.CELLS`: `ELITE_ACT1..3`, `BOSS_ACT1`, `BOSS_ACT2` and
`BOSS_ACT3_DOUBLE` (captured before boss 1; boss 2 has no cell, a double-boss start that
wins boss 1 trains it). Each worker keeps a first-in, first-out queue of 32 per cell; an
entry expires 50M decisions after its capture or after 4 starts.

**Start.** `clone`, `reseed_run(s)` (run RNG only, run_manager.rs:1955-1958), then the
Move. `s` comes from 600,000,000-639,999,999 through a per-worker PCG64. The encounter is
the same (rooms are generated up front); shuffles, AI and rewards change.

**End.** A double-boss segment plays to the run's true outcome. Every other segment ends
at its first OOC decision, which is yielded once as a value-only bootstrap row
(`E2EDecision.bootstrap`).

**Schedule** (linear over 0-2B decisions since T0): practice runs on 40% -> 25% of a
worker's env slots; of practice starts, 40% -> 75% are bosses; bosses are 65% double, 20%
act 2, 15% act 1; elites are 20/40/40 by act. A cell is active at >= 8 entries; an
inactive cell falls back to the next lower cell of its kind (act-3 -> act-2 -> act-1),
then to a full run. Every fallback is counted.

Interpretations, recorded: "40% of slots" is the share of env slots that run practice
(AlphaDefect's "full runs 50% -> 75% of engines"), applied when a slot picks its next
segment; "<= 4 reuses" is at most 4 starts per entry; an active cell's entry is drawn
uniformly at random (the FIFO bounds the queue, it does not order draws).

**Ids.** A practice segment's `run_uid` (row batch) is `practice_uid(worker, seed, n)`
for the worker's n-th start: bit 62 set, so it never equals a run seed (< 2**31), and
unique per worker and start. A start made outside the queues (`start_env` without a
`uid`) gets `start_uid(capture, reseed)`, in the same space.

**Per slot.** `SegmentSource.next(slot, t)` picks a slot's next segment: practice on the
scheduled share of slots when a cell is active, else a full run whose captures feed the
queues.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import contracts as C
from .env import Capture, E2EEnv, EnvConfig, Segment, new_env

CELLS = C.CELLS  # ELITE_ACT1..3, BOSS_ACT1, BOSS_ACT2, BOSS_ACT3_DOUBLE
DOUBLE_BOSS = "BOSS_ACT3_DOUBLE"
BOSS_MIX = {"BOSS_ACT3_DOUBLE": 0.65, "BOSS_ACT2": 0.20, "BOSS_ACT1": 0.15}
ELITE_MIX = {"ELITE_ACT1": 0.20, "ELITE_ACT2": 0.40, "ELITE_ACT3": 0.40}
FALLBACK = {
    "BOSS_ACT3_DOUBLE": ("BOSS_ACT2", "BOSS_ACT1"),
    "BOSS_ACT2": ("BOSS_ACT1",),
    "BOSS_ACT1": (),
    "ELITE_ACT3": ("ELITE_ACT2", "ELITE_ACT1"),
    "ELITE_ACT2": ("ELITE_ACT1",),
    "ELITE_ACT1": (),
}
ROOM_OF = {c: ("boss" if c.startswith("BOSS") else "elite") for c in CELLS}

QUEUE_LEN = 32
MAX_AGE = 50_000_000
MAX_USES = 4
MIN_ACTIVE = 8

SPAN = 2_000_000_000
PRACTICE_SHARE = (0.40, 0.25)
BOSS_SHARE = (0.40, 0.75)

RESEED_LO = 600_000_000
RESEED_HI = 639_999_999


def practice_uid(worker: int, seed: int, n: int) -> int:
    """The `run_uid` of a worker's n-th practice start: 62 hash bits plus bit 62."""
    return _uid(f"{int(worker)}:{int(seed)}:{int(n)}")


def start_uid(cap: Capture, reseed: int) -> int:
    """The `run_uid` of a practice start made outside the queues (`start_env` without a
    `uid`): one per capture and reseed, in `practice_uid`'s space, never a run seed."""
    return _uid(f"start:{int(cap.seed)}:{int(cap.steps)}:{int(reseed)}")


def _uid(key: str) -> int:
    h = hashlib.blake2b(key.encode(), digest_size=8)
    return (1 << 62) | (int.from_bytes(h.digest(), "little") & ((1 << 62) - 1))


def _lerp(ab: tuple[float, float], t: float) -> float:
    x = min(1.0, max(0.0, float(t) / SPAN))
    return ab[0] + (ab[1] - ab[0]) * x


def practice_share(t: float) -> float:
    """Share of env slots that run practice at `t` decisions since T0."""
    return _lerp(PRACTICE_SHARE, t)


def boss_share(t: float) -> float:
    """Share of practice starts that are bosses at `t` decisions since T0."""
    return _lerp(BOSS_SHARE, t)


def cell_probs(t: float) -> dict[str, float]:
    """P(cell) for one practice start at `t`, before fallbacks (sums to 1)."""
    b = boss_share(t)
    out = {c: b * p for c, p in BOSS_MIX.items()}
    out.update({c: (1.0 - b) * p for c, p in ELITE_MIX.items()})
    return out


def practice_slots(n_slots: int, t: float) -> int:
    """How many of a worker's `n_slots` env slots run practice at `t`."""
    return round(practice_share(t) * n_slots)


def is_practice_slot(slot: int, n_slots: int, t: float) -> bool:
    """Whether env slot `slot` (0-based) picks practice when it next starts a segment."""
    return int(slot) < practice_slots(n_slots, t)


@dataclass
class Entry:
    cap: Capture
    born: int  # the decision clock at capture
    uses: int = 0


@dataclass
class Start:
    """A practice start: the env to walk, or None when every cell fell back."""

    env: E2EEnv | None
    wanted: str
    cell: str | None
    reseed: int | None


class PracticeQueues:
    """One worker's six cells. `now` is the decision clock the caller keeps (decisions
    since T0, as the learner broadcasts it); ages and the schedule both read it."""

    def __init__(
        self,
        *,
        worker: int,
        seed: int = 0,
        enabled: bool = True,
        queue_len: int = QUEUE_LEN,
        max_age: int = MAX_AGE,
        max_uses: int = MAX_USES,
        min_active: int = MIN_ACTIVE,
    ) -> None:
        self.worker = int(worker)
        self.seed = int(seed)
        self.starts = 0  # practice starts so far (the uid counter)
        self.enabled = bool(enabled)
        self.queue_len = int(queue_len)
        self.max_age = int(max_age)
        self.max_uses = int(max_uses)
        self.min_active = int(min_active)
        self.q: dict[str, collections.deque[Entry]] = {
            c: collections.deque(maxlen=self.queue_len) for c in CELLS
        }
        ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, self.worker, 0x9AC7])
        self.rng = np.random.Generator(np.random.PCG64(ss))
        self.stats: collections.Counter = collections.Counter()

    # ── capture ─────────────────────────────────────────────────────────────

    def add(self, cap: Capture, now: int) -> None:
        if not self.enabled:
            return
        if cap.cell not in self.q:
            self.stats["capture/unknown_cell"] += 1
            return
        q = self.q[cap.cell]
        if len(q) == q.maxlen:
            self.stats[f"evict_full/{cap.cell}"] += 1
        q.append(Entry(cap=cap, born=int(now)))
        self.stats[f"captured/{cap.cell}"] += 1

    def sink(self, clock: Callable[[], int]) -> Callable[[Capture], None] | None:
        """An `EnvConfig.on_capture` that files captures at `clock()`; None (no capture,
        no clones) when practice is off."""
        if not self.enabled:
            return None
        return lambda cap: self.add(cap, clock())

    def wants_practice(self, slot: int, n_slots: int, t: float) -> bool:
        """Whether env slot `slot` starts a practice segment next (False when off)."""
        return self.enabled and is_practice_slot(slot, n_slots, t)

    def expire(self, now: int) -> None:
        for cell, q in self.q.items():
            while q and int(now) - q[0].born > self.max_age:
                q.popleft()
                self.stats[f"expired/{cell}"] += 1

    def size(self, cell: str) -> int:
        return len(self.q[cell])

    def active(self, cell: str) -> bool:
        return len(self.q[cell]) >= self.min_active

    # ── starts ──────────────────────────────────────────────────────────────

    def draw_cell(self, t: float) -> str:
        probs = cell_probs(t)
        cells = list(probs)
        p = np.asarray([probs[c] for c in cells], dtype=float)
        return cells[int(self.rng.choice(len(cells), p=p / p.sum()))]

    def reseed(self) -> int:
        return int(self.rng.integers(RESEED_LO, RESEED_HI + 1))

    def take(self, cell: str) -> Entry:
        q = self.q[cell]
        k = int(self.rng.integers(len(q)))
        e = q[k]
        e.uses += 1
        if e.uses >= self.max_uses:
            del q[k]
            self.stats[f"used_up/{cell}"] += 1
        return e

    def start(self, now: int, t: float, cfg: EnvConfig | None = None) -> Start:
        """Pick a cell for one practice start (with fallbacks) and build its env.

        `now` ages the queues; `t` (decisions since T0) sets the schedule. Returns
        `Start(env=None, ...)` when the wanted cell and its fallbacks are all inactive:
        the slot then plays a full run.
        """
        if not self.enabled:
            self.stats["disabled"] += 1
            return Start(env=None, wanted="", cell=None, reseed=None)
        self.expire(now)
        wanted = self.draw_cell(t)
        self.stats[f"wanted/{wanted}"] += 1
        for cell in (wanted, *FALLBACK[wanted]):
            if not self.active(cell):
                continue
            if cell != wanted:
                self.stats[f"fallback/{wanted}->{cell}"] += 1
            for _ in range(4):
                if not self.active(cell):
                    break
                e = self.take(cell)
                s = self.reseed()
                uid = practice_uid(self.worker, self.seed, self.starts)
                self.starts += 1
                env = start_env(e.cap, s, cfg=cfg, uid=uid)
                if env is not None:
                    self.stats[f"started/{cell}"] += 1
                    return Start(env=env, wanted=wanted, cell=cell, reseed=s)
                self.stats[f"bad_start/{cell}"] += 1
        self.stats[f"fallback/{wanted}->run"] += 1
        return Start(env=None, wanted=wanted, cell=None, reseed=None)

    def fallback_rates(self) -> dict[str, float]:
        """Per wanted cell: the share of its draws that fell back (to any cell or a run)."""
        out = {}
        for c in CELLS:
            n = self.stats[f"wanted/{c}"]
            fb = sum(v for k, v in self.stats.items() if k.startswith(f"fallback/{c}->"))
            out[c] = fb / n if n else 0.0
        return out


class SegmentSource:
    """A worker's next segment for each env slot: practice on the scheduled share of
    slots (when a cell is active), else a full run whose captures feed the queues::

        src = SegmentSource(queues, seeds=iter(run_seeds), n_slots=64, cfg=EnvConfig())
        env = src.next(slot, t)  # t: decisions since T0, the learner's clock

    Captures are filed at `clock()`, the actor's live decision clock, when given (review
    finding ENVR-7: a slot's own `t` is its run's start, a whole run of every slot's
    decisions stale by capture time), else at the latest `t` this source was given.
    """

    def __init__(
        self,
        queues: PracticeQueues,
        seeds: Iterator[int],
        n_slots: int,
        cfg: EnvConfig | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.queues = queues
        self.seeds = seeds
        self.n_slots = int(n_slots)
        self.cfg = cfg or EnvConfig()
        self.t = 0
        self.clock = clock or (lambda: self.t)
        self.stats: collections.Counter = collections.Counter()

    def next(self, slot: int, t: int) -> E2EEnv:
        self.t = int(t)
        if self.queues.wants_practice(slot, self.n_slots, t):
            st = self.queues.start(now=self.t, t=self.t, cfg=self.cfg)
            if st.env is not None:
                self.stats[f"practice/{st.cell}"] += 1
                return st.env
            self.stats["practice/run_fallback"] += 1
        cfg = dataclasses.replace(self.cfg, on_capture=self.queues.sink(self.clock))
        self.stats["run"] += 1
        return new_env(int(next(self.seeds)), cfg=cfg)


def segment_for(cap: Capture, reseed: int, uid: int | None = None) -> Segment:
    return Segment(
        kind="practice",
        cell=cap.cell,
        end="terminal" if cap.cell == DOUBLE_BOSS else "first_ooc",
        parent_seed=int(cap.seed),
        parent_idx=tuple(int(i) for i in cap.idx),
        capture_steps=int(cap.steps),
        move=tuple(sorted(cap.move.items())),
        reseed=int(reseed),
        uid=start_uid(cap, reseed) if uid is None else int(uid),
    )


def start_env(
    cap: Capture, reseed: int, *, cfg: EnvConfig | None = None, uid: int | None = None
) -> E2EEnv | None:
    """The practice env for `cap` at `reseed`, or None if the Move does not enter the
    cell's room (a guard; captures read the room after the Move, so expect none)."""
    from .actionlog import start_from_capture

    rm = start_from_capture(cap, reseed)
    room = (rm.current_room() or {}).get("room_type")
    if room != ROOM_OF[cap.cell]:
        return None
    # Every setting carries over (caps, F9, the tokenizer) except capture: practice
    # segments feed no queue.
    pcfg = dataclasses.replace(cfg or EnvConfig(), on_capture=None)
    return new_env(cap.seed, rm=rm, cfg=pcfg, segment=segment_for(cap, reseed, uid))


def opening_key(env: E2EEnv) -> tuple[Any, ...]:
    """The opening hand of a practice start (T13): its card instances in hand order.

    Instances, not content: a deck of five Strikes and four Defends has few distinct
    5-card contents, so content-only keys collide by chance even when every deal differs.
    """
    import sts2_rs

    cs = env.rm.combat_state()
    hand = sts2_rs.state_to_dict(cs)["deck"].get("hand", [])
    return tuple((c["id"], bool(c.get("upgraded")), int(c.get("stable_idx", -1))) for c in hand)
