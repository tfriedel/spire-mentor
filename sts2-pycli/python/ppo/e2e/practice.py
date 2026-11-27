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
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import contracts as C
from . import factory
from .env import Capture, E2EEnv, EnvConfig, Segment, new_env
from .forced import ForcedCfg, draw_forced
from .human import HumanCfg, draw_start
from .mix import MixCfg, draw_character

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


def cell_probs(
    t: float,
    boss: float | None = None,
    boss_mix: dict[str, float] | None = None,
    elite_mix: dict[str, float] | None = None,
) -> dict[str, float]:
    """P(cell) for one practice start at `t`, before fallbacks (sums to 1). `boss`
    replaces the scheduled boss share, `boss_mix` / `elite_mix` (normalized, every cell
    of their kind) the default splits (a lane's `practice_mix`)."""
    b = boss_share(t) if boss is None else float(boss)
    out = {c: b * p for c, p in (BOSS_MIX if boss_mix is None else boss_mix).items()}
    out.update(
        {c: (1.0 - b) * p for c, p in (ELITE_MIX if elite_mix is None else elite_mix).items()}
    )
    return out


def practice_slots(n_slots: int, t: float, share: float | None = None) -> int:
    """How many of a worker's `n_slots` env slots run practice at `t`. `share`
    replaces the scheduled share (a lane's `practice_mix`)."""
    return round((practice_share(t) if share is None else float(share)) * n_slots)


def is_practice_slot(slot: int, n_slots: int, t: float, share: float | None = None) -> bool:
    """Whether env slot `slot` (0-based) picks practice when it next starts a segment."""
    return int(slot) < practice_slots(n_slots, t, share)


PRACTICE_MIX_KEYS = ("share", "boss", "boss_mix", "elite_mix", "characters")


def _check_split(key: str, split: Any, cells: dict[str, float]) -> dict[str, float]:
    """A `boss_mix` / `elite_mix`: weights over `cells`' names (a cell left out weighs
    0), each >= 0 and not all 0, normalized to sum 1, in `cells`' order."""
    if not isinstance(split, dict):
        raise ValueError(f"{key} {split!r} is not an object of cell weights")
    bad = sorted(set(split) - set(cells))
    if bad:
        raise ValueError(f"{key}: unknown cells {bad} (known: {', '.join(cells)})")
    w = {c: float(split.get(c, 0.0)) for c in cells}
    if any(not math.isfinite(v) or v < 0.0 for v in w.values()):
        raise ValueError(f"{key} {split}: a weight is negative or not finite")
    tot = sum(w.values())
    if tot <= 0.0:
        raise ValueError(f"{key} {split}: the weights sum to 0")
    return {c: v / tot for c, v in w.items()}


def check_practice_mix(mix: dict[str, Any]) -> dict[str, Any]:
    """A lane's `practice_mix`, checked and normalized; {} keeps the schedule.

    * "share" (of env slots) and "boss" (of practice starts): each in [0, 1], replacing
      the schedule's value.
    * "boss_mix" / "elite_mix": weights over the `BOSS_MIX` / `ELITE_MIX` cells, which
      replace that default split (normalized; a cell left out is never wanted, though
      it still serves as a fallback).
    * "characters": a non-empty list of `contracts.CHARACTERS`; the queues keep only
      captures of those characters' runs, so practice starts only from them.
    """
    bad = sorted(set(mix) - set(PRACTICE_MIX_KEYS))
    if bad:
        raise ValueError(f"unknown keys {bad} (known: {', '.join(PRACTICE_MIX_KEYS)})")
    out: dict[str, Any] = {}
    for k in ("share", "boss"):
        if k in mix:
            v = float(mix[k])
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{k} {v} is not in [0, 1]")
            out[k] = v
    for k, cells in (("boss_mix", BOSS_MIX), ("elite_mix", ELITE_MIX)):
        if k in mix:
            out[k] = _check_split(k, mix[k], cells)
    if "characters" in mix:
        chars = mix["characters"]
        if isinstance(chars, str) or not isinstance(chars, (list, tuple)) or not chars:
            raise ValueError(f"characters {chars!r} is not a non-empty list")
        unknown = sorted({str(c) for c in chars} - set(C.CHARACTERS))
        if unknown:
            raise ValueError(f"characters: unknown {unknown} (known: {', '.join(C.CHARACTERS)})")
        out["characters"] = sorted({str(c) for c in chars}, key=C.CHARACTERS.index)
    return out


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
        mix: dict[str, Any] | None = None,
        queue_len: int = QUEUE_LEN,
        max_age: int = MAX_AGE,
        max_uses: int = MAX_USES,
        min_active: int = MIN_ACTIVE,
    ) -> None:
        self.worker = int(worker)
        self.seed = int(seed)
        self.starts = 0  # practice starts so far (the uid counter)
        self.enabled = bool(enabled)
        m = check_practice_mix(dict(mix or {}))
        self.share, self.boss = m.get("share"), m.get("boss")  # None: the schedule
        self.boss_mix, self.elite_mix = m.get("boss_mix"), m.get("elite_mix")  # None: default
        # The characters whose runs feed the queues; None: every character.
        self.characters: frozenset[str] | None = (
            frozenset(m["characters"]) if "characters" in m else None
        )
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

    def sink(
        self, clock: Callable[[], int], character: str | None = None
    ) -> Callable[[Capture], None] | None:
        """An `EnvConfig.on_capture` that files captures at `clock()`; None (no capture,
        no clones) when practice is off, or when the run's `character` is not one the
        lane's `practice_mix` "characters" keeps."""
        if not self.enabled:
            return None
        if self.characters is not None and character not in self.characters:
            self.stats[f"capture/skipped_character/{character}"] += 1
            return None
        return lambda cap: self.add(cap, clock())

    def wants_practice(self, slot: int, n_slots: int, t: float) -> bool:
        """Whether env slot `slot` starts a practice segment next (False when off)."""
        return self.enabled and is_practice_slot(slot, n_slots, t, self.share)

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
        probs = cell_probs(t, self.boss, self.boss_mix, self.elite_mix)
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

    With a `mix` (`ppo.e2e.mix.MixCfg`, an all-character lane) each full run plays
    `draw_character(seed, t, mix)`, and every env's run check (practice starts included:
    a capture keeps its run's character) accepts the mix's characters. Without one,
    every run is the process's character, as before.

    With `forced` (`ppo.e2e.forced.ForcedCfg`, GH #1541) a full run whose seed draws a
    card (`draw_forced`) starts with it in its deck and captures nothing: a practice
    start replays its capture from the seed and the action log, which lack the card.

    With `human` (`ppo.e2e.human.HumanCfg`) a full run whose seed draws a human start
    (`draw_start`) plays from it instead (no forced card then) and captures nothing,
    for the same reason.
    """

    def __init__(
        self,
        queues: PracticeQueues,
        seeds: Iterator[int],
        n_slots: int,
        cfg: EnvConfig | None = None,
        clock: Callable[[], int] | None = None,
        mix: MixCfg | None = None,
        forced: ForcedCfg | None = None,
        human: HumanCfg | None = None,
    ) -> None:
        self.queues = queues
        self.seeds = seeds
        self.n_slots = int(n_slots)
        self.mix = mix
        self.forced = forced
        self.human = human
        self.cfg = cfg or EnvConfig()
        if mix is not None:
            self.cfg = dataclasses.replace(self.cfg, characters=tuple(mix.characters))
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
        cfg = self.cfg
        self.stats["run"] += 1
        seed = int(next(self.seeds))
        char = cfg.character or factory.CHARACTER
        if self.mix is not None:
            char = draw_character(seed, self.t, self.mix)
            cfg = dataclasses.replace(cfg, character=char)
            self.stats[f"run/{char}"] += 1
        cfg = dataclasses.replace(cfg, on_capture=self.queues.sink(self.clock, char))
        start = draw_start(seed, char, self.human)
        if start is not None:
            cfg = dataclasses.replace(cfg, human_start=start, on_capture=None)
            self.stats["run/human"] += 1
            return new_env(seed, cfg=cfg)
        card = draw_forced(seed, char, self.forced)
        if card is not None:
            cfg = dataclasses.replace(cfg, forced_card=card, on_capture=None)
            self.stats["run/forced"] += 1
        return new_env(seed, cfg=cfg)


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
