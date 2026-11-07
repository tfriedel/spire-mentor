"""F9: cheap engine invariants for the e2e Ironclad campaign (plan §3.5).

The e2e env calls :meth:`InvariantChecker.maybe_check` on every step, which
runs :meth:`InvariantChecker.check` on about 1% of them (:data:`SAMPLE_RATE`),
and counts what it returns. The pass rule is 0
violations per 1e6 steps, live all week; a violation stops the lane and calls
HUMAN (plan §8). So every check here must hold on every reachable state. Each
one is either proved from the engine code or the C# decomp (the notes below
say where), and the tests replay real v0.111.0 A10 runs through all of them.

The checker never raises. A check that throws reports ``checker.error``.

Checks, by code:

``run.hp_range``
    ``0 <= rm.current_hp <= rm.max_hp`` and ``rm.max_hp >= 1``.
``run.gold_negative``
    ``rm.gold >= 0``.
``run.potion_overflow``
    The belt holds at most ``rm.max_potion_slots()`` potions.
``run.act_range``
    ``1 <= rm.act_index <= 4``.
``run.floor_regressed``
    ``rm.total_floor()`` never decreases within one run. Only checked when the
    caller passes ``run_key`` (a practice start from a snapshot is a new run).
``run.game_version`` / ``combat.game_version``
    The run and its combat say ``expected_game_version`` (F8's stamp, cheap).
``combat.engine``
    The engine's own structural suite, ``CombatState.check_invariants``
    (``sts2-core/src/invariants.rs::check_combat_invariants``), on the
    unclamped state: player and every creature at ``0 <= hp <= max_hp`` with
    block ``>= 0``, energy ``>= 0``, every card in exactly one of hand, draw,
    discard, exhaust and play and present in the combat's card arena (which
    holds every deck card and every generated card), hand ``<= 10``, the
    player/creature mirrors, termination, power sanity. The wire dicts clamp
    hp, block and energy on emit, so only this check can see those overflow.
    Two of its failures are legal states and only count in ``stats()``'s
    ``tolerated``: a second instance of a power that C# instances
    (``instanced_power``) and a stale ``pinned_top`` / ``pinned_bottom`` /
    ``retain_this_turn`` entry (``stale_*``). The F9 soak found both. See
    :func:`tolerated_engine_message`.
``combat.pile_duplicate``
    The ledger tier: a card uid sits twice in hand, draw, discard and exhaust
    (an independent Python re-check of the partition).
``combat.deck_card_missing``
    The ledger tier: deck cards that left every pile without a legal cause.
    See :func:`deck_ledger`.

Cost, measured by :mod:`ppo.e2e.invariant_soak` (4000 random-action A10 runs,
2.57M checks, a loaded 32-core host): the fast tier (everything but the
ledger) is 10 us median and 45 us mean per in-combat call, 3 us out of combat.
The ledger tier brings an in-combat call to 82 us median and 207 us mean
(deep act-3 states are the tail), so it runs on every ``ledger_every``-th
combat call (default 4). At the 1% sample rate that is about 1 us per env
step.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import random
import re
from collections.abc import Hashable, Iterable
from typing import Any

#: Share of env steps the e2e env checks (plan §3.5, F9).
SAMPLE_RATE = 0.01
#: The build every e2e run must be on.
EXPECTED_GAME_VERSION = "v0.111.0"
#: Ledger tier cadence: one combat call in this many also runs the ledger.
LEDGER_EVERY = 4

RUN_HP = "run.hp_range"
RUN_GOLD = "run.gold_negative"
RUN_POTIONS = "run.potion_overflow"
RUN_ACT = "run.act_range"
RUN_FLOOR = "run.floor_regressed"
RUN_VERSION = "run.game_version"
COMBAT_VERSION = "combat.game_version"
COMBAT_ENGINE = "combat.engine"
COMBAT_DUPLICATE = "combat.pile_duplicate"
COMBAT_DECK_MISSING = "combat.deck_card_missing"
CHECKER_ERROR = "checker.error"

#: Every violation code :class:`InvariantChecker` can return.
CODES: tuple[str, ...] = (
    RUN_HP,
    RUN_GOLD,
    RUN_POTIONS,
    RUN_ACT,
    RUN_FLOOR,
    RUN_VERSION,
    COMBAT_VERSION,
    COMBAT_ENGINE,
    COMBAT_DUPLICATE,
    COMBAT_DECK_MISSING,
    CHECKER_ERROR,
)

# The four piles the partition checks cover. ``state_to_dict`` also emits ``play_pile``:
# a card in it has left the others legally (`deck_ledger`), so it only counts as present.
_PILES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Cards that an in-combat transform creates (v0.111.0 decomp):
# PrimalForce -> GiantRock, Begone -> MinionStrike, Charge -> MinionDiveBomb,
# Guards -> MinionSacrifice, Compact -> Fuel, Seance -> Soul. None is a Power,
# so a transformed deck card always leaves one of these in a pile (or, if it
# is transformed again, another one).
_TRANSFORM_RESULTS = frozenset(
    {"GIANTROCK", "MINIONSTRIKE", "MINIONDIVEBOMB", "MINIONSACRIFICE", "FUEL", "SOUL"}
)
# EntropyPower.AfterPlayerTurnStart transforms hand cards into random cards,
# which can be Powers that then leave combat: no ledger while it is up.
_RANDOM_TRANSFORM_POWERS = frozenset({"ENTROPY"})
# ThievingHopper's THIEVERY move removes deck cards from combat
# (ThievingHopper.cs:224, CardPileCmd.RemoveFromCombat) and keeps them if it
# escapes. The engine drops the stolen card from the master deck at once, so
# the uid map shifts too. The creature stays on the wire with is_escaped, so
# this skip holds for the whole fight.
_THIEF_KINDS = frozenset({"THIEVINGHOPPER"})
_THIEF_POWERS = frozenset({"SWIPE"})
# MadScience's type is the one its TinkerTime event picked (MadScience.cs:80,
# ``Type => TinkerTimeType``), so a Power-type copy leaves every pile when
# played. ``card_kinds()`` gives it one fixed type, so the ledger skips it.
_DYNAMIC_TYPE_CARDS = frozenset({"MADSCIENCE"})
# Powers whose ``InstanceType`` is ``Instanced`` or ``InstancedPerApplier`` in
# the v0.111.0 decomp: each application adds a new instance
# (PowerInstanceType.cs), so one owner can hold two. The engine's dedup check
# (invariants.rs, check 9) flags those; the checker tolerates them. A test
# rebuilds this set from the decomp.
_INSTANCED_POWERS = frozenset(
    {
        "AUTOMATION",
        "CACOPHONY",
        "COVERED",
        "FLANKING",
        "GUARDED",
        "HEIST",
        "IMITATIONLEARNING",
        "KNOCKDOWN",
        "MAGICBOMB",
        "MONOLOGUE",
        "NIGHTMARE",
        "OBLIVION",
        "ORBIT",
        "PANACHE",
        "ROLLINGBOULDER",
        "SANDPIT",
        "STRANGLE",
        "SWIPE",
        "TAGTEAM",
        "THEBOMB",
        "THIEVERY",
        "TORICTOUGHNESS",
        "WITHERINGPRESENCE",
    }
)
_DUPLICATE_POWER = re.compile(r"^duplicate PowerKind (\w+) on ")
# ``pinned_top`` / ``pinned_bottom`` / ``retain_this_turn`` are annotations on
# cards in a pile. A card can leave the pile and keep its annotation (an
# auto-play of a Headbutt-pinned card, a played Expertise draw), and every
# reader filters them first: ``cards::shuffle`` and ``pop_next_card`` sweep the
# pins (cards.rs, "Staleness guard"), the reshuffle clears them, and the
# end-of-turn retain keeps only cards still in hand (combat_loop.rs). So a
# stale entry changes nothing, and the checker tolerates it.
_STALE_ANNOTATION = re.compile(
    r"^card CardUid\(\d+\) in (pinned_top|pinned_bottom|retain_this_turn) "
    r"but not in the pile it annotates"
)
# The hand cap (invariants.rs ``HAND_CAP``; C# ``CardPile.MaxCardsInHand``).
# C# never exceeds it: an add to a full hand goes to the discard pile
# (CardPileCmd.cs:485-489) and a draw stops at it (CardPileCmd.cs:1030-1066).
_HAND_CAP = 10
_HAND_OVER_CAP = re.compile(r"^hand size \d+ exceeds cap ")

#: Engine faults the F9 soak found and explained, by the key that
#: ``InvariantChecker(tolerate_known=...)`` takes. Off by default: each one is a
#: real difference from C#, so the strict checker reports it.
KNOWN_ISSUES: dict[str, str] = {
    "crossbow_hand_cap": (
        "Crossbow pushes its free Attack into a full hand "
        "(card_effects.rs generate_free_attack_in_hand); C# sends it to discard"
    ),
}
# Powers whose amount is never negative (invariants.rs ``NON_NEGATIVE_POWERS``).
_NON_NEGATIVE_POWERS = frozenset({"VULNERABLE", "WEAK", "FRAIL", "POISON", "INTANGIBLE"})


@functools.lru_cache(maxsize=8192)
def _canon(s: object) -> str:
    """``"StrikeIronclad"`` / ``"STRIKE_IRONCLAD"`` -> ``"STRIKEIRONCLAD"``."""
    return _NON_ALNUM.sub("", str(s).upper())


@functools.cache
def _card_types() -> dict[str, str]:
    """Canonical card id -> the engine's spec type (``"POWER"``, ``"ATTACK"``, ...).

    ``card_kinds()`` is one list for every build. It lacks, or types as a
    Power, 19 cards that are not Powers in v0.111.0 (mostly the build's new
    cards: Blaze, Outrage, TheBall, ...; Outbreak became a Skill), so the
    ledger exempts them. That only loses coverage. The reverse, a v0.111.0
    Power typed as a non-Power, would be a false violation; a test checks
    the decomp for it.
    """
    import sts2_rs

    return {_canon(name): kind for name, kind, _rarity in sts2_rs.card_kinds()}


@dataclasses.dataclass(frozen=True, slots=True)
class Violation:
    """One failed invariant: a stable ``code`` (see :data:`CODES`) and a detail."""

    code: str
    detail: str


def deck_ledger(
    master_deck: list[str], state: dict[str, Any]
) -> tuple[list[Violation], str | None]:
    """Check that every deck card of a combat is still accounted for.

    ``master_deck`` is ``rm.master_deck()``; ``state`` is
    ``sts2_rs.state_to_dict(cs)`` of the same combat at a player decision with
    no open choice or pending play. A card can still sit in ``play_pile``: an
    engine turn stuck inside an auto-play leaves it there at every step (seed
    1130202794, Hellraiser auto-playing Seeker Strike, the F9 stop of 2026-11-04),
    and it counts as present. Returns ``(violations, skip_reason)``; ``skip_reason`` is
    set when the deck check could not run (the duplicate check always runs).

    Why it holds. A combat's card arena only grows, a card's uid is its arena
    row + 1 (``CombatState::add_card_to_draw_pile`` and ``transform_card``),
    and the combat deck is built first, so the deck cards are uids
    ``1..len(master_deck)``, with uid ``u`` the copy of
    ``master_deck[len - u]`` (checked on every present uid; a mismatch skips
    the check). A card leaves every pile only through
    ``CardModel.GetResultLocationForCardPlay`` (a played Power or a dupe goes
    to ``PileType.None``, CardModel.cs:2086), ``CardCmd.Transform`` (the old
    uid is orphaned, a new one takes its slot), ThievingHopper's steal, or the
    end of combat. So a missing deck card that is not a Power needs a
    transform result in a pile (:data:`_TRANSFORM_RESULTS`) to explain it,
    unless a thief or a random-transform power is in the fight.
    """
    out: list[Violation] = []
    uid_to_id: dict[int, str] = {}
    for pile in _PILES:
        for card in state["deck"][pile]:
            uid = int(card["stable_idx"])
            if uid in uid_to_id:
                out.append(
                    Violation(COMBAT_DUPLICATE, f"card uid {uid} ({card['id']}) twice in the piles")
                )
            uid_to_id[uid] = str(card["id"])
    in_play = {int(card["stable_idx"]) for card in state["deck"].get("play_pile", ())}
    n = len(master_deck)
    if n == 0:
        return out, "empty_deck"
    enemies = state.get("encounter") or []
    if any(_canon(e.get("kind")) in _THIEF_KINDS for e in enemies) or any(
        _canon(p.get("name") or p.get("id")) in _THIEF_POWERS
        for e in enemies
        for p in e.get("powers") or []
    ):
        return out, "thief"
    for uid, cid in uid_to_id.items():
        if uid <= n and _canon(cid) != _canon(master_deck[n - uid]):
            return out, "deck_order"
    player = state.get("player") or {}
    if any(
        _canon(p.get("name") or p.get("id")) in _RANDOM_TRANSFORM_POWERS
        for p in player.get("powers") or []
    ):
        return out, "random_transform"
    types = _card_types()
    missing = [
        master_deck[n - uid]
        for uid in range(1, n + 1)
        if uid not in uid_to_id
        and uid not in in_play
        and _canon(master_deck[n - uid]) not in _DYNAMIC_TYPE_CARDS
        and types.get(_canon(master_deck[n - uid]), "POWER") != "POWER"
    ]
    if missing:
        transformed = sum(
            1 for uid, cid in uid_to_id.items() if uid > n and _canon(cid) in _TRANSFORM_RESULTS
        )
        if len(missing) > transformed:
            out.append(
                Violation(
                    COMBAT_DECK_MISSING,
                    f"{len(missing)} non-Power deck card(s) in no pile, {transformed} "
                    f"transform result(s) to explain them: {missing[:8]}",
                )
            )
    return out, None


def tolerated_engine_message(msg: str) -> str | None:
    """The reason the checker lets an engine-suite failure pass, or ``None``.

    Two failures of ``check_invariants`` are legal states (see
    :data:`_INSTANCED_POWERS` and :data:`_STALE_ANNOTATION`): a duplicate
    instanced power and a stale pile annotation.
    """
    m = _DUPLICATE_POWER.match(msg)
    if m is not None and _canon(m.group(1)) in _INSTANCED_POWERS:
        return "instanced_power"
    m = _STALE_ANNOTATION.match(msg)
    if m is not None:
        return f"stale_{m.group(1)}"
    return None


def _owns_relic(state: dict[str, Any], relic: str) -> bool:
    relics = (state.get("player") or {}).get("relics") or []
    return any(_canon(r.get("id") if isinstance(r, dict) else r) == relic for r in relics)


def after_tolerated_checks(
    state: dict[str, Any], *, crossbow_overflow_ok: bool = False
) -> list[Violation]:
    """The engine checks that a tolerated failure skipped, on the wire state.

    ``check_invariants`` returns its first failure only. After a stale
    annotation it has not checked the hand cap, the mirrors, termination or
    the powers; after an instanced duplicate it has not checked the rest of
    the powers. This re-checks what ``state_to_dict`` shows: the hand cap, a
    duplicate of a power that is not instanced, and a negative amount of a
    power that cannot go negative. The mirrors and termination stay unchecked
    for that state. ``crossbow_overflow_ok`` lets a hand over the cap pass
    when the player owns Crossbow (the ``crossbow_hand_cap`` known issue).
    """
    out: list[Violation] = []
    hand = len(state["deck"]["hand"])
    if hand > _HAND_CAP and not (crossbow_overflow_ok and _owns_relic(state, "CROSSBOW")):
        out.append(Violation(COMBAT_ENGINE, f"hand size {hand} exceeds cap {_HAND_CAP} (checker)"))
    owners = [("player", (state.get("player") or {}).get("powers") or [])]
    owners += [
        (f"creature {e.get('uid')}", e.get("powers") or []) for e in state.get("encounter") or []
    ]
    for owner, powers in owners:
        seen: set[str] = set()
        for p in powers:
            name = _canon(p.get("name") or p.get("id"))
            if name in seen and name not in _INSTANCED_POWERS:
                out.append(Violation(COMBAT_ENGINE, f"duplicate power {name} on {owner} (checker)"))
            seen.add(name)
            if name in _NON_NEGATIVE_POWERS and (p.get("amount") or 0) < 0:
                out.append(
                    Violation(
                        COMBAT_ENGINE, f"{owner} has negative {name}: {p.get('amount')} (checker)"
                    )
                )
    return out


class InvariantChecker:
    """Per-env F9 checker. Holds the floor watermark per run and the counters.

    ``check`` returns the violations and also tallies them in ``counts`` (by
    code) so the env can log ``stats()`` per iteration. ``maybe_check`` is
    ``check`` on a ``sample_rate`` share of calls, drawn from the checker's own
    RNG so that turning the checks on leaves the env's random stream alone.
    ``tolerate_known`` takes keys of :data:`KNOWN_ISSUES` to count in
    ``stats()["tolerated"]`` instead of reporting; the default is strict.
    """

    def __init__(
        self,
        *,
        expected_game_version: str | None = EXPECTED_GAME_VERSION,
        ledger_every: int = LEDGER_EVERY,
        sample_rate: float = SAMPLE_RATE,
        seed: int = 0,
        max_runs_tracked: int = 4096,
        tolerate_known: Iterable[str] = (),
    ) -> None:
        unknown = sorted(set(tolerate_known) - set(KNOWN_ISSUES))
        if unknown:
            raise ValueError(f"unknown known-issue keys {unknown}; see KNOWN_ISSUES")
        self._crossbow_ok = "crossbow_hand_cap" in set(tolerate_known)
        self.expected_game_version = expected_game_version
        self.ledger_every = max(0, int(ledger_every))
        self.sample_rate = float(sample_rate)
        self._rng = random.Random(seed)
        self._max_runs = max_runs_tracked
        self._floor: collections.OrderedDict[Hashable, int] = collections.OrderedDict()
        self._combat_calls = 0
        self.steps_seen = 0
        self.calls = 0
        self.ledger_calls = 0
        self.counts: collections.Counter[str] = collections.Counter()
        self.ledger_skips: collections.Counter[str] = collections.Counter()
        self.tolerated: collections.Counter[str] = collections.Counter()

    # -- public ---------------------------------------------------------------

    def maybe_check(
        self, rm: Any, cs: Any = None, *, run_key: Hashable | None = None
    ) -> list[Violation]:
        """Call once per env step: runs :meth:`check` on ``sample_rate`` of them."""
        self.steps_seen += 1
        if self._rng.random() >= self.sample_rate:
            return []
        return self.check(rm, cs, run_key=run_key)

    def check(self, rm: Any, cs: Any = None, *, run_key: Hashable | None = None) -> list[Violation]:
        """Check one env state. ``rm`` is the run's ``sts2_rs.RunManager``.

        ``cs`` is the live ``CombatState`` if the caller already holds one
        (else, in combat, ``rm.combat_state()`` is read). ``run_key`` (any
        hashable, unique per run) turns on the floor-monotonicity check.
        """
        self.calls += 1
        out: list[Violation] = []
        _guard(out, self._run_checks, rm, run_key, out)
        if _guard(out, _in_combat, rm):
            if cs is None:
                cs = _guard(out, rm.combat_state)
            if cs is not None:
                self._combat_calls += 1
                _guard(out, self._combat_checks, cs, out)
                if self.ledger_every and self._combat_calls % self.ledger_every == 0:
                    _guard(out, self._ledger, rm, cs, out)
        for v in out:
            self.counts[v.code] += 1
        return out

    def check_combat(self, cs: Any) -> list[Violation]:
        """The fast combat checks alone, for a ``CombatState`` with no run."""
        self.calls += 1
        out: list[Violation] = []
        _guard(out, self._combat_checks, cs, out)
        for v in out:
            self.counts[v.code] += 1
        return out

    def stats(self) -> dict[str, Any]:
        """Counters for the env's per-iteration stats row."""
        return {
            "steps_seen": self.steps_seen,
            "calls": self.calls,
            "ledger_calls": self.ledger_calls,
            "violations": sum(self.counts.values()),
            "by_code": dict(self.counts),
            "ledger_skips": dict(self.ledger_skips),
            "tolerated": dict(self.tolerated),
        }

    # -- tiers ----------------------------------------------------------------

    def _run_checks(self, rm: Any, run_key: Hashable | None, out: list[Violation]) -> None:
        hp, max_hp = rm.current_hp, rm.max_hp
        if not (0 <= hp <= max_hp) or max_hp < 1:
            out.append(Violation(RUN_HP, f"run hp={hp} max_hp={max_hp}"))
        gold = rm.gold
        if gold < 0:
            out.append(Violation(RUN_GOLD, f"gold={gold}"))
        held = len(rm.potions())
        slots = rm.max_potion_slots()
        if held > slots:
            out.append(Violation(RUN_POTIONS, f"{held} potions held, {slots} slots"))
        act = rm.act_index
        if not 1 <= act <= 4:
            out.append(Violation(RUN_ACT, f"act_index={act}"))
        if self.expected_game_version is not None:
            ver = rm.game_version
            if ver != self.expected_game_version:
                out.append(
                    Violation(RUN_VERSION, f"run on {ver}, expected {self.expected_game_version}")
                )
        if run_key is not None:
            floor = rm.total_floor()
            last = self._floor.get(run_key)
            if last is not None and floor < last:
                out.append(Violation(RUN_FLOOR, f"total_floor {last} -> {floor}"))
            self._floor[run_key] = floor if last is None else max(floor, last)
            self._floor.move_to_end(run_key)
            while len(self._floor) > self._max_runs:
                self._floor.popitem(last=False)

    def _combat_checks(self, cs: Any, out: list[Violation]) -> None:
        msg = cs.check_invariants("e2e")
        if msg is not None:
            import sts2_rs

            msg = str(msg)
            reason = tolerated_engine_message(msg)
            state = None
            if reason is None and self._crossbow_ok and _HAND_OVER_CAP.match(msg):
                state = sts2_rs.state_to_dict(cs)
                if _owns_relic(state, "CROSSBOW"):
                    reason = "known_crossbow_hand_cap"
            if reason is None:
                out.append(Violation(COMBAT_ENGINE, msg))
            else:
                # The engine suite stops at its first failure, so the checks
                # after a tolerated one did not run. Re-run the ones the wire
                # can show.
                self.tolerated[reason] += 1
                if state is None:
                    state = sts2_rs.state_to_dict(cs)
                out.extend(after_tolerated_checks(state, crossbow_overflow_ok=self._crossbow_ok))
        if self.expected_game_version is not None:
            ver = cs.game_version
            if ver != self.expected_game_version:
                out.append(
                    Violation(
                        COMBAT_VERSION, f"combat on {ver}, expected {self.expected_game_version}"
                    )
                )

    def _ledger(self, rm: Any, cs: Any, out: list[Violation]) -> None:
        if cs.pending_choice() is not None or cs.has_pending_play():
            self.ledger_skips["pending"] += 1
            return
        import sts2_rs

        self.ledger_calls += 1
        found, skip = deck_ledger(rm.master_deck(), sts2_rs.state_to_dict(cs))
        out.extend(found)
        if skip is not None:
            self.ledger_skips[skip] += 1


def _guard(out: list[Violation], fn: Any, *args: Any) -> Any:
    """Call ``fn(*args)``; a raise becomes a ``checker.error`` violation in ``out``."""
    try:
        return fn(*args)
    except Exception as exc:  # the checker must never raise
        out.append(Violation(CHECKER_ERROR, f"{getattr(fn, '__name__', fn)}: {exc!r}"))
        return None


def _in_combat(rm: Any) -> bool:
    return rm.phase() == "combat" and not rm.is_over()


def check(rm: Any, cs: Any = None) -> list[Violation]:
    """One-off check with the ledger on (no floor watermark). For tests and audits."""
    return InvariantChecker(ledger_every=1).check(rm, cs)
