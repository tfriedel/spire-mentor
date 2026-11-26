"""Persistent per-fight engine mirror for the DLL / real-game drivers.

Plan: ``2026-07-24 persistent combat mirror`` (Phase 1).

Why this exists
===============

When the C# headless DLL (rung 3.5) or the live game (SpireMentor) is the
*world*, the Rust engine has no state of its own: historically every combat
decision rebuilt a throwaway ``CombatState`` from the decision message,
searched it, and threw it away. That works for ``combat_play``, whose message
carries the whole board. It fails for ``card_select``, whose message carries
only the offered cards — no enemies, no hand, no energy, no piles — so the
driver had to *replay* the previous snapshot forward and hope the engine
re-derived an identical option list. Measured on the realdll_v4 sweep (60
runs, 635 mid-combat prompts): 251 searched, 263 replays failed the match
gate, 121 were never attempted, **175 were answered at random**.

A :class:`CombatMirror` holds ONE engine state alive for a whole fight and
advances it by the actions actually committed. A ``card_select`` then needs no
reconstruction at all: the mirror is already parked on the choice the card
opened.

The one rule
============

**Ground truth replays what already happened; the information set decides what
happens next.**

* The mirror holds concrete truth — the true pile order the world reported,
  the RNG cursors carried across decisions. It is never searched directly.
* :meth:`CombatMirror.search` runs on a *clone*, blindfolded at the root by
  the engine's ``hidden_info`` machinery. Clairvoyant profiles are refused
  outright (:meth:`CombatMirror.search`), because a mirror is exactly the
  object that would make clairvoyance work — and therefore exactly the object
  that must not be allowed to.

Transplant reconcile
====================

:meth:`CombatMirror.reconcile` is correct-by-construction: drift cannot
accumulate, because reality wins on every observable, every decision.

1. Rebuild the whole state from the wire (``sts2_rs.state_from_dict``).
2. Carry over from the previous mirror only what the wire cannot express —
   RNG cursors, per-enemy ``intent_history``, draw-pile pins, the belief flag
   (``CombatState.transplant_hidden``).
3. Diff the carried mirror's *prediction* against the fresh reconstruction on
   deterministic observables. That list is the DESYNC report.

Discipline
==========

:data:`MirrorPolicy.STRICT` (the default, and the only mode whose numbers may
be quoted) raises :class:`MirrorDivergence`. :data:`MirrorPolicy.OBSERVE`
(``STS2_MIRROR_OBSERVE=1``) dumps forensics, drops the mirror, and **tags the
run** via :attr:`CombatMirror.tainted` so its results can't be mistaken for
clean. Either way the divergence is appended to
``logs/diag/mirror_divergences.jsonl``.

There is exactly one principled non-crash: an advance that crossed a shuffle
produces a pile order the world rolled independently, so the next decision's
observables are *unverifiable* rather than wrong. Those are tagged and skipped
(:attr:`MirrorStats.reshuffle_skips`), and the reconcile after them
self-corrects.

Known Phase-2 divergence
========================

The engine keeps dead enemies in ``encounter`` (corpses drive revive
machinery: Decimillipede segments, Fogmog's Eye with Teeth); the real game
removes ordinary corpses from its enemy list. When the two lists disagree on
length, ``transplant_hidden`` raises rather than truncating — a kill in a
multi-enemy fight is expected to surface here first when the DLL driver lands.
That is the intended failure mode, not something to paper over: the crash
names the exact mismatch.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple

from sts2_hint.recommender import RecommendConfig

#: Env flag that downgrades STRICT to OBSERVE for a whole process.
OBSERVE_ENV = "STS2_MIRROR_OBSERVE"

#: Env override for the forensic JSONL path.
LOG_ENV = "STS2_MIRROR_LOG"

#: Env flag that arms the engine's damage-source trace and attaches the rows
#: for the diverging step to every divergence record. GH #368.1 — an HP desync
#: names the amount and never the cause, and the mirror re-imposes reality one
#: decision later, so the evidence is gone before anyone can ask. Off by
#: default: it makes the records substantially larger, and the sweeps that do
#: NOT need it should not pay for it.
DAMAGE_TRACE_ENV = "STS2_MIRROR_DAMAGE_TRACE"


class MirrorPolicy(StrEnum):
    """What a divergence does.

    ``STRICT`` raises. ``OBSERVE`` records, drops the mirror, and taints the
    run. There is deliberately no "repair" mode: a mirror that silently
    patches over a mismatch is worse than no mirror, because its search then
    plans against a board that never existed.

    Co-op (:attr:`CombatMirror.teammates`) is not a repair in that sense. A
    teammate's moves land on the enemies between ours, so an enemy field the
    engine could not have predicted is not a misprediction; the reconcile
    already ADOPTS the world's board, and only the carried hidden state is
    kept. That is counted (:attr:`MirrorStats.teammate_adoptions`), never
    silent, and scoped to ``$.encounter``: our own side still diverges.
    """

    STRICT = "strict"
    OBSERVE = "observe"


def policy_from_env(default: MirrorPolicy = MirrorPolicy.STRICT) -> MirrorPolicy:
    """``MirrorPolicy.OBSERVE`` iff ``STS2_MIRROR_OBSERVE=1``, else ``default``."""
    return MirrorPolicy.OBSERVE if os.environ.get(OBSERVE_ENV) == "1" else default


@dataclass(frozen=True)
class Desync:
    """One deterministic observable the mirror predicted wrong.

    ``path`` is a JSON-path-ish label into the wire dict (``$.player.hp``,
    ``$.encounter[1].intents``) so a report line points straight at the field.
    """

    path: str
    mirror: Any
    world: Any

    def __str__(self) -> str:
        return f"{self.path}: mirror={self.mirror!r} world={self.world!r}"


@dataclass
class MirrorStats:
    """Per-mirror counters. The fidelity metric this project reports.

    ``desync_rate_per_1000`` replaces the old "fallback share": under a
    correct mirror there is no fallback to take a share of, so the question
    stops being "how often did we guess?" and becomes "how often did the
    engine mispredict reality?".
    """

    decisions: int = 0
    reconciles: int = 0
    #: Reconciles whose observable diff was non-empty. Numerator of
    #: :attr:`desync_rate_per_1000`.
    desyncs: int = 0
    desync_fields: int = 0
    #: Injected choices the engine's own derivation graded wrong. Counted
    #: SEPARATELY from :attr:`desyncs`: they are a different event over a
    #: different denominator (:attr:`injections`), and folding them into the
    #: reconcile rate inflated a numerator its denominator never counted.
    choice_mismatches: int = 0
    reshuffle_skips: int = 0
    #: Observable paths excluded from a diff because the world's producer
    #: does not emit them (:func:`unreported_paths`). Non-zero means the
    #: fidelity number is measured over an incomplete observable set — the
    #: DLL and live arms report no ``play_pile``, so this is expected to be
    #: exactly ``reconciles`` there and ``0`` against the native oracle.
    unreported_fields: int = 0
    injections: int = 0
    #: Injections where the engine had NOT parked a choice of its own, so the
    #: driver had to fall back to the ``card_select`` catch-all kind. That
    #: kind's resolver is a deliberate no-op, so the pick changes nothing in
    #: the mirror while it changes the world — every such injection is a
    #: guaranteed one-advance misprediction of whatever the screen granted.
    #: Some are legitimate (a combat-START selection the engine never had a
    #: chance to open); the rest are engine modelling gaps. Counted, not
    #: raised, so the size of the gap is measurable before it is closed.
    unparked_injections: int = 0
    #: Per-stream RNG read-count differences seen at reconcile, once the
    #: producer carries the run's dice. A drift means the two engines
    #: consumed different amounts of randomness for the same step — the
    #: one divergence no field diff can see coming, because the board
    #: still looks right this turn and is wrong for the rest of the fight.
    #: Recorded rather than raised: the cursor is hidden state, not one of
    #: the deterministic observables STRICT is defined over.
    cursor_drifts: int = 0
    #: Reconciles where a card was SUSPENDED mid-play, so the mirror's
    #: observables were verified against the wire without rebuilding it.
    #: Rebuilding would delete the suspended tail, which no wire can carry
    #: (oracle Silent-16).
    parked_verifies: int = 0
    #: Reconciles SKIPPED because the WORLD's wire showed a card still in its
    #: ``play_pile`` while the mirror had nothing in flight — the live
    #: producer snapshotted mid-card-resolution (effects half-applied), so no
    #: observable on that board is gradable. The diff is deferred to the next
    #: stable wire; see the world-in-flight comment in :meth:`reconcile`.
    in_flight_skips: int = 0
    #: Reconciles SKIPPED because the world's wire was snapshotted between a
    #: card-select closing and the effect that screen *granted* landing —
    #: the mirror already holds the picked card (added to hand, or its cost
    #: zeroed) and the world does not yet. Same deferral contract as
    #: :attr:`in_flight_skips`, for the choice channel; see
    #: :func:`_choice_effect_in_flight`.
    choice_in_flight_skips: int = 0
    #: Co-op reconciles whose whole diff was enemy-side, so the world's board
    #: was adopted instead of reported (see :class:`MirrorPolicy`). Kept out
    #: of :attr:`desyncs`: a teammate's hit is not an engine misprediction.
    teammate_adoptions: int = 0
    searches: int = 0
    seeds: int = 0
    drops: int = 0

    @property
    def desync_rate_per_1000(self) -> float:
        """DESYNC-ing reconciles per 1000 reconciles. 0.0 when none ran."""
        return 1000.0 * self.desyncs / self.reconciles if self.reconciles else 0.0

    @property
    def choice_mismatch_rate_per_1000(self) -> float:
        """Mis-derived choices per 1000 injections. 0.0 when none ran."""
        return 1000.0 * self.choice_mismatches / self.injections if self.injections else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "reconciles": self.reconciles,
            "desyncs": self.desyncs,
            "desync_fields": self.desync_fields,
            "choice_mismatches": self.choice_mismatches,
            "reshuffle_skips": self.reshuffle_skips,
            "unreported_fields": self.unreported_fields,
            "injections": self.injections,
            "unparked_injections": self.unparked_injections,
            "cursor_drifts": self.cursor_drifts,
            "parked_verifies": self.parked_verifies,
            "in_flight_skips": self.in_flight_skips,
            "choice_in_flight_skips": self.choice_in_flight_skips,
            "teammate_adoptions": self.teammate_adoptions,
            "searches": self.searches,
            "seeds": self.seeds,
            "drops": self.drops,
            "desync_rate_per_1000": round(self.desync_rate_per_1000, 3),
            "choice_mismatch_rate_per_1000": round(self.choice_mismatch_rate_per_1000, 3),
        }


class MirrorDivergence(RuntimeError):
    """The mirror and the world disagree about a board that must match.

    Carries the forensic ``report`` that was written to the divergence log, so
    a caller that catches this (a sweep driver tallying failures) has the same
    detail the file does without re-reading it.
    """

    def __init__(self, message: str, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report: dict[str, Any] = report or {}


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------
#
# Both sides of a diff pass through `sts2_rs.state_to_dict`, so normalization
# is identical by construction and a difference is a real difference — never
# an emitter artifact.
#
# The comparisons themselves run in Rust (`sts2-rs/crates/sts2-advisor`,
# `mirror.rs`, behind `sts2_rs.mirror_*`; GH #1426 step 1), which the native
# advisor will link. The functions below keep their names and contracts, and
# their docstrings say why each rule is what it is; the tests in
# `tests/test_combat_mirror.py` hold the Rust side to them. Values that were
# tuples (a card key, a power pair) are lists now.


def enemy_slot_key(enemy: dict[str, Any], index: int, encounter: list[dict[str, Any]]) -> str:
    """An enemy's identity across the two worlds: ``"<kind>#<nth of kind>"``.

    NOT the slot index. The two worlds disagree about slots by construction:
    the game removes an ordinary corpse from its enemy list the moment it
    dies (keeping only revivers, which it ships separately and the adapter
    APPENDS), while the engine keeps every creature in its original spawn
    position so its revive machinery can find it. So one kill shifts every
    later slot, and a slot-wise diff then reports six fields wrong on an
    otherwise perfectly synchronized board — 515 of the 2992 divergences on
    the 60-run realdll_v5 sweep were exactly that.

    Same-kind enemies are numbered in list order, and the LIVING and the DEAD
    are numbered separately. Counting corpses too would defeat the whole
    point: the engine's surviving Nibbit sits behind its own corpse and would
    be numbered ``Nibbit#1`` while the game — which reaped that corpse —
    numbers the same creature ``Nibbit#0``. The keys would then miss each
    other, the corpse would be diffed against the survivor, and the reaping
    rule in :func:`diff_observables` could never fire, because the key it
    computes for the mirror-only enemy names the LIVE one.

    **What this key is not.** Numbering within each namespace is stable under
    reaping only while BOTH worlds reap the same creature. On a mispredicted
    kill — the class that dominates the residual — one side reaps and the other
    does not, and every later same-kind enemy shifts by one: a mirror holding
    ``[NibbitA alive, NibbitB alive]`` against a world holding ``[NibbitB
    alive]`` numbers B as ``Nibbit#1`` while the world calls it ``Nibbit#0``, so
    the keys cross and B is diffed against A. Nothing local can fix that — two
    same-kind enemies are indistinguishable on the wire, so the numbering IS the
    identity and a shift destroys it. What the roster comparison in
    :func:`diff_observables` reports on such a board is correct and is the real
    finding; the per-field differences reported alongside it are noise.
    :func:`_enemy_alignment` refuses to carry hidden state across a shift for
    exactly this reason — see :func:`_unalignable_kinds`.
    """
    import sts2_rs

    return sts2_rs.mirror_enemy_slot_key(enemy, index, encounter)


def observables(wire: dict[str, Any]) -> dict[str, Any]:
    """The deterministic half of a wire dict, flattened to ``{path: value}``.

    Deterministic means: given the same board and the same committed action,
    both engines must produce the same value. Anything the engines are allowed
    to roll independently (the RNG cursors themselves, ``combat_id``, the
    unresolved seed) is deliberately absent.
    """
    import sts2_rs

    return sts2_rs.mirror_observables(wire)


def unreported_paths(raw_world_wire: dict[str, Any]) -> frozenset[str]:
    """Observable paths the WORLD's producer does not emit at all.

    Both sides of :func:`diff_observables` are ``state_to_dict`` output, which
    always emits every key — so by diff time "the producer never sent this"
    is indistinguishable from "the producer sent empty". That distinction has
    to be read off the raw wire, before the round-trip normalizes it.

    Only ``play_pile`` needs this today: the Rust wire round-trips it, but
    both C# producers OMIT it unless a card is actually in flight
    (``RunSimulator.CombatPlayState`` sends null when empty, and the
    live-arm translation drops an empty pile — ``sts2_hint.adapter``). A
    producer that predates the field also means "no card is mid-play" by
    its silence, so diffing an absent pile against the mirror's would
    report a divergence whose fault is the producer's, not the mirror's.
    Absent means UNKNOWN-or-empty, not comparable.
    """
    import sts2_rs

    return frozenset(sts2_rs.mirror_unreported_paths(raw_world_wire))


def _choice_effect_in_flight(desyncs: list[Desync], picked: list[dict[str, Any]]) -> bool:
    """Is this hand diff exactly the just-picked card's effect not-yet-landed?

    The live producer answers a ``card_select`` the instant the pick arrives
    (``SimCardSelector.GetSelectedCards`` returns; the ``CardSelectCmd`` that
    awaited it then finishes on the action queue), and nothing downstream
    waits for the grant to land the way ``PlayCardAndWaitAsync`` waits for the
    play pile to drain. So the next serialized combat board can be a
    MID-RESOLUTION snapshot of the *choice* channel: the screen is gone, the
    pick is committed, and the card it granted is still on its way into the
    hand. The mirror resolved the same pick atomically, so at that instant the
    mirror is AHEAD and the hand is not gradable — 33 of the 49 pure
    ``$.deck.hand`` records in the ``mirror_live3`` corpus, the other 16
    being the post-reseed echo each one manufactured one decision later.

    Two shapes, one per grant flavour, both proven by the corpus:

    * ``add_generated[_free]_to_hand`` — the mirror's hand carries the picked
      card and the world's does not carry it *anywhere* (27 records);
    * ``make_free_in_hand`` — both hands carry it, the mirror's at cost 0 and
      the world's still at the printed cost (6 records).

    Deliberately narrow, so this defers a snapshot and never grants amnesty:

    * the ONLY diverging path may be ``$.deck.hand``;
    * every card in the symmetric difference must be one the pick just named;
    * the mirror must be the ADVANCED side (it holds a card the world lacks,
      or the same card at a strictly lower cost). A mirror that is BEHIND the
      world looks identical to the post-reseed echo this fixes and to a real
      misprediction, so it is never deferred.

    Args:
        desyncs: The reconcile's observable diff.
        picked: The option dicts (``{"id", "upgraded", …}``) the world's
            selection resolved to.

    Returns:
        True when the diff is fully explained by the pending grant.
    """
    import sts2_rs

    return sts2_rs.mirror_choice_effect_in_flight(
        [(d.path, d.mirror, d.world) for d in desyncs], picked
    )


def diff_observables(
    mirror_wire: dict[str, Any],
    world_wire: dict[str, Any],
    *,
    skip_prefix: str | None = None,
    skip_paths: frozenset[str] = frozenset(),
) -> list[Desync]:
    """Every deterministic observable on which the two dicts disagree.

    Args:
        mirror_wire: What the carried mirror predicts.
        world_wire: What reality reports.
        skip_prefix: Drop paths starting with this. Used for the
            crossed-a-reshuffle case, where the pile paths (and only the
            pile paths) are genuinely unverifiable — see
            :meth:`CombatMirror.reconcile`.
        skip_paths: Drop these exact paths. See :func:`unreported_paths`.
    """
    import sts2_rs

    return [
        Desync(path, mine, theirs)
        for path, mine, theirs in sts2_rs.mirror_diff_observables(
            mirror_wire, world_wire, skip_prefix, sorted(skip_paths)
        )
    ]


def _cursor_drift(mirror_wire: dict[str, Any], world_wire: dict[str, Any]) -> list[Desync]:
    """How many rolls the mirror took that the world didn't (and vice versa).

    This is the one observable that says whether the two engines *consumed*
    the same randomness, as opposed to landing on the same values. A monster
    whose AI reads one extra roll produces a correct-looking board this turn
    and a wrong one for the rest of the fight, and no field diff can see it
    coming — the counters can.

    Only meaningful once the producer carries the run's dice: before that the
    mirror's streams are seeded from something unrelated to the game's, so the
    read counts are not comparable at all.
    """
    import sts2_rs

    return [Desync(*d) for d in sts2_rs.mirror_cursor_drift(mirror_wire, world_wire)]


def _turn_history_disclosed(wire: dict[str, Any]) -> bool:
    """Whether the wire carries our turn's exhausts and card block gains
    (newer mods, GH #989). Then reality wins on them over the carried
    mirror's, as on every other observable."""
    from sts2_hint.adapter import TURN_HISTORY_KEYS

    player = wire.get("player") or {}
    return all(key in player for key in TURN_HISTORY_KEYS)


def _undisclosed_relic_counters(wire: dict[str, Any]) -> list[str]:
    """Relics the wire named but told us no ``counter`` for.

    A relic's counter is not only a displayed tally — it is also where a
    one-shot per-combat latch lives (Vambrace: "the first card-granted block
    this combat is doubled"). Both producers only emit a counter for relics
    that SHOW one, so those latches arrive as ``None`` and every
    reconstruction re-arms them. Naming them here is what lets the transplant
    keep the carried mirror's value; the ones reality DID report keep
    reality's.
    """
    import sts2_rs

    return sts2_rs.mirror_undisclosed_relic_counters(wire)


def _undisclosed_power_counters(wire: dict[str, Any]) -> list[str]:
    """Powers the wire named but told us no ``counter`` for — either side.

    The power twin of :func:`_undisclosed_relic_counters`: a power whose
    accumulator lives in C# ``InternalData`` (Juggling's attacks-this-turn,
    Possess*'s stolen-stat totals) arrives as ``{name, amount}`` and every
    reconstruction re-zeroed it — the Juggling 3rd-attack clone never fired
    (26 v28wr hand records). Naming them lets the transplant keep the carried
    mirror's accumulation; a counter reality DID report keeps reality's.
    (Juggling's is now derived from the wire instead, see ``derived``.)
    """
    import sts2_rs

    return sts2_rs.mirror_undisclosed_power_counters(wire)


def _relocate_parked(parked: dict[str, Any], seeded: dict[str, Any]) -> dict[str, Any] | None:
    """Re-locate a stale parked choice's options against the reseeded piles.

    A parked choice that survived a reseed carries ``pile``/``index``
    locators that indexed the DROPPED state, so each option is matched
    one-to-one on ``(id, upgraded)`` in option order against the piles the
    mirror now holds, consuming positions so duplicates resolve to distinct
    cards. An option not found anywhere is dropped rather than guessed —
    a later ``_resolve_locators`` then reports the concrete
    ``unlocatable_option`` instead of a blanket refusal. Returns ``None``
    when no option survives (the carried derivation was entirely stale).

    ``seeded`` must be ``state_to_dict`` output, NOT the raw wire, for the
    same reason ``reconcile``'s parked-verify round-trips before diffing:
    the wire is the producer's dialect and a parked option is the engine's,
    so ``curse:Decay`` never equals ``DECAY`` and the option is silently
    dropped. Only prefixed ids diverge, so the failure is invisible on an
    ordinary board and total on a cursed one. In ``logs/f7-mirror-check16``
    it is every one of the six ``unlocatable_option`` records: the world
    offered a whole hand of 8, the carried derivation came back as 7 —
    short by exactly its unplayable cards (DECAY, DAZED, CLUMSY) — and six
    advices across four Survivor / Dagger Throw screens fell to the rarity
    sort. The engine
    itself offers unplayable cards for discard, exactly as C# does; nothing
    was ever filtering them.
    """
    import sts2_rs

    return sts2_rs.mirror_relocate_parked(parked, seeded)


def _unalignable_kinds(world_wire: dict[str, Any], mirror_wire: dict[str, Any]) -> set[str]:
    """Kinds whose ``#nth`` numbering the two worlds cannot be trusted to share.

    :func:`enemy_slot_key`'s numbering is only an identity while the two worlds
    hold the same *number* of living enemies of that kind. When they do not —
    a mispredicted kill, or a summon one side has and the other does not — the
    numbering shifts on one side only and the keys start naming different
    creatures. There is nothing local left to disambiguate with: two Nibbits at
    the same HP are genuinely indistinguishable on the wire, so the numbering IS
    the identity, and once it shifts the identity is gone.

    Note this is per KIND, not global: numbering counts only same-kind enemies,
    so a Nibbit the world reaped cannot shift the Wurm beside it.

    A same-kind summon lands here too, which is stricter than it strictly has to
    be — a spawn appends, so the enemies before it keep their numbers. But
    "world has one more" is also what a kill the MIRROR predicted and the world
    did not looks like (the mirror parks it in the dead namespace, shortening
    its own living run), and those two are not distinguishable either. The cost
    of refusing is one enemy's carried ``intent_history``, i.e. one turn of
    telegraph prediction; the cost of guessing wrong is that history landing on
    a different creature, which is worse and silent.
    """
    import sts2_rs

    return set(sts2_rs.mirror_unalignable_kinds(world_wire, mirror_wire))


def _enemy_alignment(world_wire: dict[str, Any], mirror_wire: dict[str, Any]) -> list[int | None]:
    """For each of the world's enemies, the carried mirror's slot for it.

    ``None`` where the mirror has no counterpart (a newly-spawned enemy has no
    history to carry) — or where the two worlds' rosters have drifted far enough
    that the counterpart cannot be identified, see :func:`_unalignable_kinds`.
    Matching is otherwise by :func:`enemy_slot_key`, because the two worlds do
    not agree on slot numbering the moment one of them reaps a corpse.
    """
    import sts2_rs

    return sts2_rs.mirror_enemy_alignment(world_wire, mirror_wire)


def wire_enemy_index(wire: dict[str, Any]) -> dict[str, int]:
    """``{identity key -> slot}`` for a wire dict's encounter."""
    import sts2_rs

    return sts2_rs.mirror_wire_enemy_index(wire)


def _default_log_path() -> Path:
    override = os.environ.get(LOG_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "logs" / "diag" / "mirror_divergences.jsonl"


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------


#: Search profiles that would let the mirror leak its own ground truth into
#: the search root. Refused by :meth:`CombatMirror.search`.
class CombatMirror:
    """One live engine state per fight, re-synchronized from reality.

    Typical driver loop for a ``combat_play`` decision::

        mirror.reconcile(wire)                 # or .seed(wire) for a new fight
        result = mirror.search()
        action = best_recommend_action(result)
        world.take_action(action)              # the world commits it
        mirror.advance(action)                 # the mirror follows

    and for a ``card_select``::

        mirror.reconcile(wire)                 # the choice's own combat payload
        mirror.inject_choice(kind, options, min_select, max_select)
        result = mirror.search()
        mirror.advance_through_choice(picks)

    The per-fight state machine (seed, reconcile, advance, the choice path,
    the forensic report) runs in Rust (``sts2_rs.CombatMirrorCore``,
    ``sts2-advisor/src/combat_mirror.rs``, GH #1426), which documents each
    rule where it is applied. This class keeps the API, the search, the
    divergence log and :attr:`on_divergence`.

    Args:
        cfg: The search profile :meth:`search` uses by default. Must be
            honest — see :meth:`search`.
        policy: Divergence policy. Defaults to the env-resolved value
            (:func:`policy_from_env`).
        label: Free-form tag written into every forensic record (a seed, a
            run id) so a sweep's divergences can be attributed.
        log_path: Forensic JSONL destination. Defaults to
            ``logs/diag/mirror_divergences.jsonl``.
    """

    def __init__(
        self,
        cfg: RecommendConfig | None = None,
        *,
        policy: MirrorPolicy | None = None,
        label: str = "",
        log_path: str | Path | None = None,
        _core: Any | None = None,
    ) -> None:
        import sts2_rs

        self.cfg = cfg or RecommendConfig(hidden_info="beliefdraw", c_puct=2.5, draw_widening_c=2.0)
        self.policy = policy if policy is not None else policy_from_env()
        self.log_path = Path(log_path) if log_path is not None else _default_log_path()
        #: Optional ``(paths: list[str]) -> None`` hook, called once per
        #: divergence with the field paths that diverged, BEFORE the policy
        #: raises. Lets a sweep classify divergences live instead of
        #: re-parsing the forensic log. Never swallows the divergence.
        self.on_divergence: Callable[[list[str]], None] | None = None
        # GH #368.1: the damage trace is latched from the env at construction
        # so a sweep cannot half-arm itself mid-run.
        self._core = _core or sts2_rs.CombatMirrorCore(
            self.policy.value, label, os.environ.get(DAMAGE_TRACE_ENV) == "1"
        )
        # The world's own wire object, for :attr:`wire`; the core keeps a
        # JSON copy for its reports. `_wire_generation` says which call's
        # wire the core last adopted.
        self._last_wire: dict[str, Any] | None = None
        self._wire_generation = self._core.wire_generation

    # -- core plumbing -----------------------------------------------------

    def _call(self, method: str, *args: Any, wire: dict[str, Any] | None = None) -> Any:
        """Run one core method: log and report its divergences, track the wire.

        Every divergence the call recorded is appended to the forensic log and
        handed to :attr:`on_divergence`, oldest first; then a STRICT one
        raises as :class:`MirrorDivergence`.
        """
        import sts2_rs

        try:
            out = getattr(self._core, method)(*args)
        except sts2_rs.MirrorDivergedError as exc:
            self._flush_divergences()
            message, report = exc.args
            raise MirrorDivergence(f"[mirror {self.label}] {message}", report) from None
        finally:
            if wire is not None and self._core.wire_generation != self._wire_generation:
                self._last_wire = wire
            self._wire_generation = self._core.wire_generation
        self._flush_divergences()
        return out

    def _flush_divergences(self) -> None:
        for paths, report in self._core.take_divergences():
            _append_jsonl(self.log_path, {"ts": time.time(), **report})
            if self.on_divergence is not None:
                self.on_divergence(paths)

    def take_enemy_remap(self) -> list[tuple[int, int]] | None:
        """The enemy uid renames the rebuilds made since the last call
        (``[(uid then, uid now)]``), or None (GH #1651). A rebuild numbers
        enemies by wire position and the wire drops corpses, so a kill shifts
        the survivors' uids; a log keyed by uid follows them through this."""
        return self._core.take_enemy_remap()

    @property
    def label(self) -> str:
        return self._core.label

    @property
    def stats(self) -> MirrorStats:
        """A snapshot of the counters."""
        counts = self._core.stats
        return MirrorStats(
            **{f: counts[f] for f in MirrorStats.__dataclass_fields__}  # type: ignore[arg-type]
        )

    @property
    def tainted(self) -> bool:
        """Set once the mirror recorded a divergence under OBSERVE. A run with
        this set must never be reported as a clean measurement."""
        return self._core.tainted

    @tainted.setter
    def tainted(self, value: bool) -> None:
        self._core.tainted = value

    @property
    def drop_reason(self) -> str:
        """Why the mirror was last dropped (``""`` for a clean end-of-combat drop)."""
        return self._core.drop_reason

    @property
    def teammates(self) -> int:
        """How many other players share this fight (co-op). Non-zero lets a
        reconcile adopt an enemy-side diff instead of reporting it; see
        :class:`MirrorPolicy`. The driver sets it from the board."""
        return self._core.teammates

    @teammates.setter
    def teammates(self, value: int) -> None:
        self._core.teammates = value

    @property
    def last_adopted(self) -> list[Desync]:
        """The diff the last reconcile adopted under :attr:`teammates`."""
        return [Desync(p, m, w) for p, m, w in self._core.last_adopted]

    @property
    def _crossed_shuffle(self) -> bool:
        return self._core.crossed_shuffle

    @_crossed_shuffle.setter
    def _crossed_shuffle(self, value: bool) -> None:
        self._core.crossed_shuffle = value

    @property
    def _choice_in_flight(self) -> list[dict[str, Any]] | None:
        return self._core.choice_in_flight

    # -- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Is a state currently mirrored? False before :meth:`seed` / after :meth:`drop`."""
        return self._core.alive

    @property
    def state(self) -> Any:
        """The live ``sts2_rs.CombatState``. Read it; do not advance it directly.

        A copy of the mirror's state, re-made only when the state changed, so
        two reads of an unchanged mirror are the same object.

        Raises:
            MirrorDivergence: When no fight is mirrored — asking for the state
                of a dropped mirror is a driver bug, not a game state.
        """
        state = self._core.state
        if state is None:
            raise MirrorDivergence("mirror has no state: seed() a fight first")
        return state

    @property
    def wire(self) -> dict[str, Any] | None:
        """The last wire dict reality handed us, or ``None``.

        Drivers need it to map a search result's ``card_index`` back to the
        world's own card handle: the engine indexes the hand it was built from,
        and that hand is ``wire["deck"]["hand"]`` with its ``stable_idx``
        values intact.
        """
        return self._last_wire

    def seed(self, wire: dict[str, Any]) -> None:
        """Start mirroring a fight from a full combat state.

        Args:
            wire: A wire-schema dict (``sts2_rs.state_to_dict`` shape) — from
                ``sts2_hint.adapter.translate`` on the DLL / live path, or
                straight from the engine in the oracle test.
        """
        self._call("seed", wire, wire=wire)

    def fork(self, label: str = "") -> CombatMirror | None:
        """A DETACHED copy of this fight, for work that must not touch it.

        Answering a ``card_select`` is destructive by design: the caller
        reconciles, replays the opening play, and injects the world's option
        list, leaving the mirror parked on a choice only *that* caller can
        resolve. A driver executes its own picks, so it resolves it. An
        ADVISOR does not execute anything, so it never can.

        Doing that work on a fork makes the answer path side-effect free.
        Without one it is not: an injection the engine cannot locate drops the
        mirror outright, and it does so on screens the advisor was always
        going to decline. Live run 001-ironclad-a8-0824-2239 shows the whole
        chain in three lines -- ``DROPLET_OF_PRECOGNITION is a potion, not a
        card`` (an honest decline), then ``injection dropped the mirror``,
        then ``no live mirror at the screen`` when the same screen was
        re-asked. A screen nobody could answer cost the fight its mirror.

        Carries the engine state and the small amount of bookkeeping an answer
        reads: the engine's own derived choice, the reshuffle blind spot, and
        the last wire. Stats are NOT shared -- the fork's work is not the
        fight's history, and counting a throwaway's reconciles as the live
        mirror's would corrupt every rate computed from them.

        Returns:
            The fork, or ``None`` when there is nothing to fork. Never raises:
            a fork is an optimisation over declining.
        """
        core = self._core.fork(label)
        if core is None:
            return None
        twin = CombatMirror(self.cfg, policy=self.policy, log_path=self.log_path, _core=core)
        twin._last_wire = self._last_wire
        return twin

    def drop(self, reason: str = "") -> None:
        """Forget the current fight. Idempotent.

        Called at combat end, and by OBSERVE after a divergence (a mirror that
        has already been proven wrong must not keep answering). The dropped
        state and its parked choice are kept for the reseed after an OBSERVE
        divergence, which salvages the UNDISCLOSED hidden state (relic
        latches, history counters, per-card vars) the wire structurally cannot
        restore — the divergence proved the state's *prediction* wrong, not
        those.
        """
        self._core.drop(reason)

    # -- reconcile ---------------------------------------------------------

    def reconcile(self, wire: dict[str, Any]) -> list[Desync]:
        """Re-impose every observable from reality; return the DESYNC report.

        Seeds the fight when nothing is mirrored yet, so a driver can call
        this unconditionally at every combat decision.

        A state suspended where no wire can describe it (a card mid-play, an
        enemy turn parked on a curse screen, a turn-transition walk, a drink
        on its own card-select) is verified and KEPT rather than rebuilt. A
        world snapshotted mid-resolution (a card still in its ``play_pile``,
        a choice's grant still in transit) is skipped and graded at the next
        wire. Otherwise the state is rebuilt from the wire, the hidden state
        is carried over, and the carried prediction is diffed against it;
        under OBSERVE a divergence re-seeds from the wire, salvaging what the
        wire cannot say.

        Args:
            wire: The world's current combat state, wire-schema.

        Returns:
            The deterministic observables the carried mirror predicted wrong,
            so an OBSERVE-mode caller sees them without re-reading the
            forensic log. Empty on a healthy mirror; never non-empty under
            STRICT, which raises instead of returning.

        Raises:
            MirrorDivergence: Under STRICT, on any DESYNC or on a
                transplant the shapes refuse (draw pile or enemy roster
                disagreeing on length).
        """
        return [Desync(p, m, w) for p, m, w in self._call("reconcile", wire, wire=wire)]

    # -- advance -----------------------------------------------------------

    def advance(self, action: dict[str, Any]) -> str:
        """Commit one action the world already took.

        Args:
            action: A bridge-shape action dict (``{"action": "play",
                "card_index": …}``) — the same shape ``legal_actions``
                emits.

        Returns:
            The engine's ``StepResult`` as a lowercase string: ``advanced`` |
            ``pending_choice`` | ``combat_won`` | ``combat_lost`` |
            ``invalid_target``.

        Raises:
            MirrorDivergence: Under STRICT, when the engine reports
                ``invalid_target`` — the world and the engine disagreeing
                about what is legal is a divergence, not a recoverable
                condition.
        """
        return self._call("advance", action)

    # -- choices -----------------------------------------------------------

    def adopt_unseen_play(self, wire: dict[str, Any]) -> bool:
        """Seed from ``wire`` when it shows a card in play the mirror never
        played (``sts2_advisor::combat_mirror::CombatMirror::adopt_unseen_play``):
        Imbued's turn-1 autoplay opening a screen after a pre-draw seed.
        Call after :meth:`reconcile`, before reading :meth:`parked_choice`.
        """
        return bool(self._call("adopt_unseen_play", wire, wire=wire))

    def parked_choice(self) -> dict[str, Any] | None:
        """The choice the engine derived on its own, or ``None``.

        Captured when an :meth:`advance` parked on a pending choice, and kept
        across the following :meth:`reconcile` (which rebuilds the state and
        therefore wipes the live choice). Cleared by :meth:`inject_choice`,
        which is what grades it.
        """
        return self._core.parked_choice()

    def inject_choice(
        self,
        kind: str,
        options: Sequence[dict[str, Any]],
        min_select: int,
        max_select: int,
        *,
        can_skip: bool = False,
        origin: dict[str, Any] | None = None,
    ) -> list[Desync]:
        """Open the choice the WORLD is showing, and grade the engine's own.

        This is the single path for every mid-combat selection — there is no
        "trust the engine's derivation" branch. Removing option-matching from
        the critical path is what retires the three-tier fallback chain: the
        engine's derivation becomes an assertion instead of a data source.

        Grading is by kind. A *deterministic* kind's options are concrete
        cards out of the player's own piles, so engine and world must derive
        the same list and a mismatch is a divergence. A *generative* kind
        (``add_generated_*``) rolled its own content, so a mismatch says
        nothing about fidelity and is not graded
        (``sts2_rs.choice_kind_is_generative``).

        Args:
            kind: ``ChoiceKind`` python name (``put_back``, ``card_select``, …).
            options: ``{"id", "upgraded", "pile", "index"}`` dicts. ``pile`` +
                ``index`` locate a concrete card and must be given together;
                omitting both makes a synthetic option, which is only
                meaningful for generative kinds. This is exactly the shape
                ``CombatState.pending_choice()`` emits.

                A real driver rarely has locators: a ``card_select`` message
                names cards, not board positions. Options without one are
                RESOLVED against the engine's own parked choice, one-to-one on
                ``(id, upgraded)`` in the world's order — which is also where
                a live list the engine cannot account for becomes a
                divergence. Without a parked choice a deterministic kind
                cannot be answered at all: injecting unlocated options would
                resolve to a silent no-op.
            min_select / max_select: Selection arity from the world.
            can_skip: Whether the world offers a skip.
            origin: The world's own description of the screen — the
                ``prompt_key`` / ``source`` the C# ``CardSelectCmd`` context
                patch captured. Purely forensic: it is what names the culprit
                when the engine parked no choice of its own, since an option
                list of three generated cards says what was offered but not
                which effect offered it.

        Returns:
            The graded differences against the engine's own derivation, so an
            OBSERVE-mode caller sees them without re-reading the forensic log.
            Empty for a generative kind and for a matching derivation; never
            non-empty under STRICT, which raises instead.

        Raises:
            MirrorDivergence: Under STRICT, when a deterministic kind's
                engine-derived option list differs from the world's, or when
                the engine refuses the injected options.
        """
        graded = self._call(
            "inject_choice",
            kind,
            [dict(o) for o in options],
            min_select,
            max_select,
            can_skip,
            origin,
        )
        return [Desync(p, m, w) for p, m, w in graded]

    def arm_turn_start_resume(self, after_relic: str) -> bool:
        """Park the rest of turn setup behind the OPEN turn-start screen.

        C# ``Hook.AfterPlayerTurnStart`` *awaits* the ``CardSelectCmd`` a relic
        opens from it, so the DLL's board at such a decision is a mid-setup
        snapshot: the hand is drawn and energy is reset, but every later
        listener on that pass, the whole ``AfterPlayerTurnStartLate`` pass and
        the outer loop's ``Hook.AfterSideTurnStart`` still have to run. A mirror
        seeded from that wire and handed a free-standing injected choice never
        ran any of it.

        Only Gambling Chip's round-1 mulligan reaches this today; the engine
        parks the identical marker when its own copy of the relic raises the
        screen.

        Args:
            after_relic: Relic id (``"GAMBLING_CHIP"``) whose hook the world is
                blocked inside.

        Returns:
            True when the marker was armed. False when the engine declined —
            no choice open, or the relic is not in the mirror's own inventory —
            in which case nothing is parked and the resolve behaves as before.
        """
        return self._call("arm_turn_start_resume", after_relic)

    def arm_turn_start_resume_after_power(self, after_power: str) -> bool:
        """Park the rest of turn setup behind a POWER's turn-start screen.

        The any-turn twin of :meth:`arm_turn_start_resume` (GH #355). C#
        ``Hook.AfterPlayerTurnStart`` awaits the ``CardSelectCmd`` that
        ``TyrannyPower`` / ``ToolsOfTheTradePower`` open, and unlike the two
        relics with the same shape neither carries a turn gate — so the
        mid-setup snapshot this repairs is reachable on ANY turn, which is
        what made the relic-only fix a partial one.

        Args:
            after_power: Power id (``"TYRANNY"``) whose hook the world is
                blocked inside.

        Returns:
            True when the marker was armed. False when the engine declined —
            no choice open, or the mirror's player does not carry the power.
        """
        return self._call("arm_turn_start_resume_after_power", after_power)

    def arm_turn_start_resume_for(self, block: TurnStartBlock) -> bool:
        """Arm whichever resume :class:`TurnStartBlock` names.

        The two surfaces are not interchangeable — see
        :class:`TurnStartBlock` — so the dispatch lives here rather than at
        each driver, which is what kept the relic case from drifting between
        ``headless_eval`` and ``realgame_mcts`` in the first place.
        """
        if block.source == "power":
            return self.arm_turn_start_resume_after_power(block.id)
        return self.arm_turn_start_resume(block.id)

    def advance_through_choice(self, selection: Sequence[int]) -> str:
        """Answer the open choice and confirm it closed.

        A world that picks fewer than ``max_select`` is finalized here rather
        than left mid-accumulation (the engine re-opens a partial multi-pick
        and waits; the world resolved in one shot). Anything still open after
        that is a genuine arity disagreement. The picked options are armed for
        one reconcile, whose board may show the screen closed before its grant
        landed.

        Args:
            selection: Option indices the world picked. Empty means skip.

        Returns:
            The engine's ``StepResult`` string from the final step.

        Raises:
            MirrorDivergence: Under STRICT, when no choice is open (the world
                showed a selection the engine never opened), or when the
                choice survives both the picks and the finalize.
        """
        return self._call("advance_through_choice", [int(i) for i in selection])

    # -- search ------------------------------------------------------------

    def search(
        self, cfg: RecommendConfig | None = None, *, seed: int | None = None
    ) -> dict[str, Any]:
        """Run one honest MCTS search on a CLONE of the mirror.

        The only place this module launches a search. Three invariants live
        here and nowhere else:

        * it searches a clone, so no search can ever mutate ground truth;
        * it refuses a clairvoyant profile. The mirror is the one object that
          would make clairvoyance actually work — the true draw order and the
          true RNG cursors are sitting right there — which makes it the one
          object that must not be allowed to use them;
        * it refuses to search a mirror that is itself already on the BELIEF
          representation. ``deferred_draw_order`` is a claim that the pile is
          a canonical belief rather than a materialized order; the mirror
          holds concrete truth, so only the private clone the search builds
          may ever carry it. A live mirror with the flag set means a belief
          state leaked back into ground truth, and searching it would quietly
          change how the pile is consumed.

        Args:
            cfg: Overrides the mirror's default profile for this call.
            seed: Overrides ``cfg.seed`` (drivers vary it per decision).

        Returns:
            The raw ``sts2_rs.mcts_recommend`` result dict.

        Raises:
            ValueError: On a clairvoyant ``hidden_info``.
            MirrorDivergence: When the live mirror is already on the belief
                representation.
        """
        cfg = cfg or self.cfg
        if seed is not None:
            cfg = replace(cfg, seed=seed)
        # The guards and the call: `CombatMirrorCore.search`
        # (`sts2_advisor::combat_mirror`, GH #1426).
        return self._call("search", asdict(cfg))

    def search_choice_picks(
        self,
        cfg: RecommendConfig | None = None,
        *,
        hand: list[dict[str, Any]],
        n_options: int,
        min_select: int,
        max_select: int,
    ) -> list[int]:
        """Answer the open choice — ONE SEARCH PER PICK on a multi-pick screen.

        Shared by both mirrored ``card_select`` drivers
        (``headless_eval.pick_card_select_mirrored`` and
        ``realgame_mcts.pick_card_select_mirrored``).

        A single-pick screen (``max_select <= 1``) is one search, unchanged.

        A MULTI-pick screen used to be answered by taking the top
        ``max_select`` rows off that same single search. But the engine
        decomposes a multi-pick SEQUENTIALLY (bead ``sts2-rs-sopgf``): each
        incremental ``ResolveChoice`` appends to ``pc.picked``, fires no
        effect, and RE-OPENS the same choice, so the root the search scored
        only ever answered "which card first". Its second row is the runner-up
        to that question, not the best follow-up to the pick actually made —
        and for a discard-2 or an exhaust-2 the two picks interact by
        construction. This walks the engine's own decomposition instead: pick,
        apply, re-search, repeat.

        The walk runs on a PRIVATE clone. The mirror is not advanced here —
        the caller still applies the picks only after the world accepts them
        (:meth:`advance_through_choice`), so a rejected selection cannot leave
        ground truth half-answered.

        Args:
            cfg: Overrides the mirror's default profile for this call.
            hand: The wire's ``deck.hand`` — the stable-idx frame the tags are
                read in. Incremental picks fire no effects, so one hand stays
                valid across the whole walk.
            n_options: How many options the world's screen shows.
            min_select / max_select: The screen's (already clamped) arity.

        Returns:
            Distinct option indices, in pick order. Empty means skip.
        """
        return self._call(
            "search_choice_picks",
            asdict(cfg or self.cfg),
            hand,
            n_options,
            min_select,
            max_select,
        )

    def prepare_card_select(
        self, state: dict[str, Any], wire: dict[str, Any], headless: dict[str, Any]
    ) -> tuple[int, int, int]:
        """Put the mirror on the world's ``card_select``: reconcile against the
        screen's ``wire``, recognize the screen (:func:`recognize_card_select`),
        inject the world's options and arm a blocked turn start behind them.

        The first half of ``realgame_mcts.pick_card_select_mirrored``, in Rust
        (``sts2_advisor::card_select``, GH #1426).

        Args:
            state: The bridge ``card_select`` message.
            wire: ``realgame_adapter.combat_wire`` of its embedded combat.
            headless: ``to_headless_combat_play`` of the same board.

        Returns:
            ``(n_options, min_select, max_select)``, the arity clamped to the
            options on screen.

        Raises:
            MirrorDivergence: Under STRICT, on a divergence.
            sts2_rs.UnsupportedStateError: A card the registry does not know.
        """
        return tuple(self._call("prepare_card_select", state, wire, headless, wire=wire))

    # -- divergence --------------------------------------------------------

    def _diverge(
        self,
        message: str,
        *,
        kind: str,
        paths: list[str],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a divergence the search found, then raise (STRICT) or drop
        the mirror (OBSERVE)."""
        self._call("diverge", message, kind, paths, extra)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one forensic record. Best-effort: never masks the divergence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        # A full disk must not turn a legible divergence into an IOError
        # traceback that hides it.
        pass


class TurnStartBlock(NamedTuple):
    """What the world's turn-start hook batch is currently blocked inside.

    ``source`` is ``"relic"`` or ``"power"``; ``id`` is that model's python id
    (``"GAMBLING_CHIP"`` / ``"TYRANNY"``). Fed to
    :meth:`CombatMirror.arm_turn_start_resume_for`, which picks the matching
    engine surface — the two are NOT interchangeable: a relic block resumes
    the relic ``AfterPlayerTurnStart`` bus, a power block resumes the power
    cluster walk that runs BEFORE it.
    """

    source: str
    id: str


def skip_arity_from_world(
    *, cancelable: Any, source: str | None, min_select: int, max_select: int
) -> tuple[bool, int]:
    """Reconcile a screen's skippability with its reported minimum arity.

    Shared by both world drivers so the two cannot disagree.

    The legacy rule was ``can_skip = min_select <= 0``, which is sound for
    every ``CardSelectorPrefs``-carrying entry point — there ``MinSelect`` is
    the real arity the screen enforces — but NOT for
    ``CardSelectCmd.FromChooseACardScreen``. That overload hard-codes its call
    into the selector as ``GetSelectedCards(cards, 0, 1)``
    (``CardSelectCmd.cs:230``): the 0 is a constant, not the screen's minimum,
    and the actual skippability lives in the separate ``canSkip`` parameter
    which defaults to **false** (``CardSelectCmd.cs:216``). Ten of its twelve
    v0.107.1 callers pass ``canSkip: true`` explicitly; the two that take the
    default — ``Toolbox`` and ``KnowledgeDemon`` — are MANDATORY, and Toolbox
    is the single largest unparked-injection family in the corpora. Injected
    as skippable, the search was free to decline a card the real screen has no
    skip button for.

    ``cancelable`` is NOT a general substitute for the arity test, which is
    why this is keyed on ``source`` rather than applied everywhere.
    ``CardSelectorPrefs.Cancelable`` is an ``init``-only field defaulting to
    false that the two public constructors never set, so it reads false on
    screens that are plainly declinable — Gambling Chip's mulligan is
    ``CardSelectorPrefs(prompt, 0, 999999999)``, answerable with nothing via
    confirmation rather than via a Cancel button. Trusting it there would
    force a mandatory discard and break the mulligan recognition below.

    Clearing ``can_skip`` alone would not close the hole either: both drivers
    gate their "answer with nothing" branch on ``min_select <= 0``, and
    :func:`choice_picks_from_tags` treats a zero minimum as permission to
    return an empty selection. So a mandatory screen's minimum is re-floored
    to 1 — which is what the C# screen means anyway (exactly one card, no skip
    button).

    Args:
        cancelable: The world's captured flag (``state["cancelable"]``) — for
            ``FromChooseACardScreen`` this is the ``canSkip`` argument
            verbatim. None when no capture patch fired, in which case the
            legacy inference is kept, so a world that does not send it is
            unchanged.
        source: The ``CardSelectCmd`` entry-point name the world captured
            (``state["source"]``), or None.
        min_select: The already-clamped minimum selection arity.
        max_select: The already-clamped maximum selection arity.

    Returns:
        ``(can_skip, min_select)``, both normalized.
    """
    import sts2_rs

    return sts2_rs.card_select_skip_arity(cancelable, source, min_select, max_select)


def recognize_card_select(
    *,
    parked: dict[str, Any] | None,
    options: list[dict[str, Any]],
    board: dict[str, Any],
    wire: dict[str, Any],
    min_select: int,
    source_model: str | None = None,
    prompt_key: str | None = None,
    source: str | None = None,
) -> tuple[str, str | None]:
    """Recognize a mid-combat ``card_select`` screen the engine never opened.

    Shared by BOTH world drivers — ``headless_eval.pick_card_select_mirrored``
    (DLL world) and ``realgame_mcts.pick_card_select_mirrored`` (live
    SpireMentor world) — so the recognition logic cannot fork and drift again.
    It operates on dialect-neutral inputs: a headless-dialect ``board`` (the
    DLL's own ``combat_play`` dict, or ``realgame_adapter.
    to_headless_combat_play``'s translation of the bridge snapshot), the
    ``sts2_hint.adapter.translate`` wire built from it, and the option list
    the caller is about to hand to :meth:`CombatMirror.inject_choice`.

    The engine's own kind is used when it derived one (``parked``). A screen
    the engine never opened (a combat-START selection, Gambling-Chip-style)
    has no parked choice — and ``card_select`` is NOT a usable catch-all for
    that shape: with nothing to resolve the world's card names against, the
    options carry no locator, and ``ChoiceKind::CardSelect`` is an explicit
    no-op placeholder arm in the engine. ``inject_choice`` refuses the
    injection rather than resolving it to nothing; the recognition below maps
    the known unparked screens to their real engine kinds instead:

    * Generative screens (Toolbox's combat-start colorless offer, the
      Attack/Skill/Power/Colorless potions, Discovery) — by ``source_model``
      when the world named the effect, else by shape (no option present in
      any pile).
    * KnowledgeDemon's curse pick — ``resolve_monster_curse_select`` maps
      card_id → (PowerKind, amount), no locators needed either.
    * Gambling Chip's turn-1 mulligan — round 1, skippable, and the option
      multiset is exactly the mirror's whole hand. Mapped to the engine's
      real ``discard_from_hand_and_draw`` (locators filled in from the hand
      itself, mutating ``options`` in place) so resolution actually
      discards+draws in the mirror and the search ranks the picks.

    Args:
        parked: :meth:`CombatMirror.parked_choice` — the engine's own derived
            choice, or ``None`` when it never opened one.
        options: The world's option list (``{"id", "upgraded"}`` dicts, ids
            already resolved via ``sts2_hint.adapter.registry_card_id``).
            MUTATED in place when the Gambling-Chip recognition fires: each
            option gains its ``pile``/``index`` hand locator.
        board: Headless-dialect combat board (``hand``, ``round``,
            ``player.relics``).
        wire: ``sts2_hint.adapter.translate(board, ...)`` — the same wire the
            caller just reconciled the mirror against.
        min_select: The already-clamped minimum selection arity.
        source_model: The world's name for the effect that opened the screen
            (``state["source_model"]``, RunSimulator.cs CardSelectCmd context
            patch), when it captured one.
        prompt_key: The world's localization prompt key, when captured.
        source: The world's selector (``state["source"]``). A
            ``FromChooseACardScreen`` offers generated cards, also when one
            matches a card in a pile (GH #1746).

    Returns:
        ``(kind, turn_start_blocked_on)``. ``kind`` is what the caller passes
        to :meth:`CombatMirror.inject_choice`. ``turn_start_blocked_on`` is
        the :class:`TurnStartBlock` naming the relic OR power whose turn-start
        hook the world is blocked inside, or ``None``; when set, the caller
        MUST call :meth:`CombatMirror.arm_turn_start_resume_for` with it after
        ``inject_choice`` and BEFORE the search, so the rollouts see the
        deferred turn-setup continuation.
    """
    import sts2_rs

    kind, block, located = sts2_rs.card_select_recognize(
        parked, options, board, wire, min_select, source_model, prompt_key, source
    )
    for option, stamped in zip(options, located, strict=True):
        option.update(stamped)
    return kind, (TurnStartBlock(*block) if block is not None else None)


__all__ = [
    "CombatMirror",
    "Desync",
    "MirrorDivergence",
    "MirrorPolicy",
    "MirrorStats",
    "TurnStartBlock",
    "diff_observables",
    "observables",
    "policy_from_env",
    "recognize_card_select",
]
