"""A persistent per-fight engine mirror for the live ADVISOR lane.

Why
---
The advisor is *stateless*: every decision it throws the fight away and
rebuilds the board from the snapshot the mod just sent. The autopilot is not
-- it keeps a :class:`combat_mirror.CombatMirror` running alongside the fight.
Three things live only in that mirror, and the advisor therefore never had
them:

1. **The open choice and its suspended tail.** Burning Pact is "exhaust a card
   THEN draw 2"; the engine holds that tail as the parked
   ``PendingChoice.post_effect``. A choice injected onto a freshly rebuilt
   board has ``post_effect: None``, so a search over it ranks the exhaust
   while believing no draw follows. That is a confident wrong answer, which is
   worse than the leftmost-card default it would replace -- amputating
   ``post_effect`` was 60 of 188 power divergences on the v27wr sweep.
2. **Hidden per-turn counters** (GH #370). ``card_plays_started_this_turn``,
   ``shivs_played_this_turn``, ``generated_cards_this_combat``,
   ``stars_gained_this_turn``, ``vigor_latch_card``: none is emitted by
   SpireMentor, so a rebuilt board reads them as 0 forever. Some are legality
   gates (``RingingPower.ShouldPlay`` / ``SlothPower.ShouldPlay``), so a stuck
   value changes which actions the search believes exist.
   ``transplant_hidden`` carries them; only a carried mirror has them to carry.
3. The true RNG cursors and undisclosed relic/power counters, same mechanism.

What this does NOT buy
----------------------
Clairvoyance. The mirror holds the true draw order, which makes it the one
object forbidden from searching with it -- :meth:`CombatMirror.search` refuses
a clairvoyant profile outright. Both lanes still search under ``beliefdraw``.

OBSERVE, not STRICT
-------------------
The autopilot's mirror watches a policy; this one watches a *human*, who may
play a card nobody recommended, rewind, or walk away. OBSERVE is the mode for
that: a divergence tags the mirror and re-seeds from reality instead of
raising. A dropped mirror costs advice quality for the rest of the fight --
which is exactly where the advisor already sits permanently -- so the floor
does not move and the ceiling does.

Lifecycle
---------
Seeded/reconciled on every combat ``advice_request``; dropped on a fight
change, on ``snapshot_restored`` / ``combat_undone`` (the mod broadcasts both
for *any* rewind, including ones the human triggers from the in-game toolbar),
and at ``game_over`` / ``run_complete``.

In Rust since GH #1426 (``sts2_advisor::advisor_mirror``, through
``sts2_rs.AdvisorMirrorCore``). This class keeps the API and the reasons
(the method docstrings below), writes the log lines and the divergence
records the core queues, and hands out :attr:`AdvisorMirror.live` and its
answer fork as :class:`_MirrorView` objects that read like a
:class:`combat_mirror.CombatMirror`.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from combat_mirror import (
    DAMAGE_TRACE_ENV,
    Desync,
    MirrorDivergence,
    _append_jsonl,
    _default_log_path,
)
from realgame_adapter import combat_wire
from sts2_hint.recommender import RecommendConfig

logger = logging.getLogger(__name__)


def fight_key(state: dict[str, Any]) -> tuple[Any, Any]:
    """Identity of the fight a bridge message belongs to.

    One room is one fight, so ``(act, floor)`` separates them. A key that
    changes mid-fight only costs a re-seed (the next reconcile rebuilds from
    the wire), which is the same thing the mirror does after any divergence --
    so an imperfect key degrades quality, never correctness.

    """
    return (state.get("act"), state.get("floor"))


def _enemy_hp(wire: dict[str, Any]) -> list[Any]:
    """The LIVING enemy HP vector — the cheapest fingerprint a target moves.

    Living-only, and that is the whole point. The engine keeps a corpse in
    `encounter` (`{"hp": 0, "is_dead": true}`); the live producer drops the
    dead enemy from its embedded combat state entirely. Comparing the two
    positionally means a play that KILLS can never fingerprint-match — the
    mirror's vector is one entry longer than the world's no matter which
    target is tried — so `replay_opening_play` declines every killing play
    that then opens a screen.

    That is the three `no board matched replay of NEOWS_FURY` declines in
    `logs/f7-mirror-check16`, and Neow's Fury is only the card that happened
    to show it: 10-14 damage, and it opens a discard-pile select immediately
    after. Nothing about the class is Neow-specific.

    Filtering both sides is safe whichever way the producer behaves: if it
    ever does report a corpse, dropping it on both sides still compares like
    with like. Order is still positional over the survivors, so it stays
    discriminating — killing a different enemy yields a different vector.

    """
    import sts2_rs

    return sts2_rs.advisor_enemy_hp(wire)


def _wire_for(board: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """The answer path's board import: :func:`realgame_adapter.combat_wire`,
    looked up here at call time so a test can watch what it is handed."""
    return combat_wire(board, **kwargs)


class _MirrorView:
    """The live mirror, or the fork an answer is worked out on, as the code
    above it reads a :class:`combat_mirror.CombatMirror`: ``alive``,
    ``state``, ``advance``, ``reconcile``, ``search``, ``fork``, ``drop``.

    Both live inside the Rust core; a view only names which one.
    """

    def __init__(self, owner: AdvisorMirror, which: str, label: str) -> None:
        self._owner = owner
        self._which = which
        self.label = label
        self._state: tuple[int, Any] | None = None

    def _call(self, method: str, *args: Any) -> Any:
        import sts2_rs

        try:
            return getattr(self._owner._core, method)(self._which, *args)
        except sts2_rs.MirrorDivergedError as exc:
            message, report = exc.args
            raise MirrorDivergence(f"[mirror {self.label}] {message}", report) from None
        finally:
            self._owner._flush()

    @property
    def alive(self) -> bool:
        return self._owner._core.alive(self._which)

    @property
    def state(self) -> Any:
        """The engine state, a copy re-made only when it changed.

        Raises:
            MirrorDivergence: When no fight is mirrored.
        """
        generation = self._owner._core.state_generation(self._which)
        if self._state is not None and self._state[0] == generation:
            return self._state[1]
        state = self._owner._core.state(self._which)
        if state is None:
            raise MirrorDivergence("mirror has no state: seed() a fight first")
        self._state = (generation, state)
        return state

    def advance(self, action: dict[str, Any]) -> str:
        return self._call("advance", action)

    @property
    def tainted(self) -> bool:
        return self._owner._core.tainted(self._which)

    def reconcile(self, wire: dict[str, Any]) -> list[Desync]:
        return [Desync(p, m, w) for p, m, w in self._call("reconcile", wire)]

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

    def advance_through_choice(self, selection: Sequence[int]) -> str:
        return self._call("advance_through_choice", [int(i) for i in selection])

    def search(
        self, cfg: RecommendConfig | None = None, *, seed: int | None = None
    ) -> dict[str, Any]:
        """One honest search on a clone (``CombatMirror.search``)."""
        cfg = cfg or self._owner.cfg
        if seed is not None:
            cfg = replace(cfg, seed=seed)
        return self._call("search", asdict(cfg))

    def fork(self, label: str = "") -> _MirrorView | None:
        """The fight's throwaway fork (``CombatMirror.fork``); one at a time."""
        if self._which != "live" or not self._owner._core.fork_live(label):
            return None
        return _MirrorView(self._owner, "fork", label)

    def drop(self, reason: str = "") -> None:
        self._owner._core.drop_mirror(self._which, reason)
        self._owner._flush()


@dataclass(frozen=True)
class _StateView:
    """What a picker reads off the parked fork: its ``state``."""

    state: Any


class AdvisorMirror:
    """Owns the advisor's per-fight :class:`CombatMirror`, or nothing.

    Every method is total: on any internal failure the mirror is dropped and
    the caller falls back to the stateless path it used before. Nothing here
    may raise into the advisor's serve loop.

    """

    def __init__(
        self,
        cfg: RecommendConfig,
        *,
        enabled: bool = True,
        log_path: Any = None,
    ) -> None:
        import sts2_rs

        self.cfg = cfg
        self._log_path = Path(log_path) if log_path is not None else _default_log_path()
        self._core = sts2_rs.AdvisorMirrorCore(enabled, os.environ.get(DAMAGE_TRACE_ENV) == "1")
        self._live = _MirrorView(self, "live", "advisor")
        self._live_generation = self._core.generation

    @property
    def enabled(self) -> bool:
        return self._core.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._core.enabled = value

    @property
    def stats(self) -> dict[str, int]:
        """Counters for the run log: how often the carried state was there."""
        return self._core.stats

    @property
    def declines(self) -> dict[str, int]:
        """Why the searched rung declined, by reason."""
        return self._core.declines

    @property
    def desync_paths(self) -> dict[str, int]:
        """Desyncs by wire path, list keys folded (``$.encounter[].hp``)."""
        return self._core.desync_paths

    @property
    def adopted_paths(self) -> dict[str, int]:
        """Co-op enemy-side diffs adopted instead of reported, by path."""
        return self._core.adopted_paths

    def _flush(self) -> None:
        """Write what the core queued: its log lines and the divergence
        records of every mirror it ran (the fork's included)."""
        for level, message in self._core.take_log():
            getattr(logger, level)("%s", message)
        for _paths, report in self._core.take_divergences():
            _append_jsonl(self._log_path, {"ts": time.time(), **report})

    def _which(self, on: Any) -> str:
        return "fork" if isinstance(on, _MirrorView) and on._which == "fork" else "live"

    def decline(self, reason: str) -> None:
        """Record (and, once per reason, announce) a decline to search."""
        self._core.decline(reason)
        self._flush()

    @property
    def live(self) -> _MirrorView | None:
        """The mirror, iff one is seeded and has not been dropped."""
        if not self._core.alive("live"):
            return None
        if self._live_generation != self._core.generation:
            # A new fight is a new mirror.
            self._live = _MirrorView(self, "live", "advisor")
            self._live_generation = self._core.generation
        return self._live

    def drop(self, reason: str) -> None:
        """Forget the fight. Safe to call at any time, including when empty."""
        self._core.drop(reason)
        self._flush()

    def on_rewind(self, msg: dict[str, Any]) -> None:
        """Drop on ``snapshot_restored`` / ``combat_undone``.

        Both carry a fresh ``state_id`` precisely because everything older is
        stale (PROTOCOL.md section 7); a carried combat state is exactly that.
        A snapshot restore lands at room entry so the fight ends anyway, but a
        turn undo keeps the fight and moves the board BACKWARDS, which is the
        case that would silently corrupt a carried mirror.

        """
        self._core.on_rewind(msg.get("type"))
        self._flush()

    def on_combat_board(self, wire: dict[str, Any], state: dict[str, Any]) -> _MirrorView | None:
        """Seed or reconcile for one combat decision; return the live mirror.

        Args:
            wire: ``sts2_hint.adapter.translate`` output for this board.
            state: The bridge message, for :func:`fight_key`.

        Returns:
            The mirror when it is carrying this fight, else ``None`` (the
            caller then behaves exactly as it did before this class existed).

        """
        live = self._core.on_combat_board(wire, state)
        self._flush()
        return self.live if live else None

    def set_advice_target(self, target: Any) -> None:
        """Remember the last combat advice's board target (``best.target_index``,
        None for an untargeted one): a play that kills its target reports a
        null target, which the mirror recovers from it (GH #1918)."""
        self._core.set_advice_target(target)

    def on_world_action(self, msg: dict[str, Any]) -> None:
        """Advance the mirror through an action the world just committed.

        THE thing that makes a mirror a mirror. The autopilot advances its
        own after every accepted action, so its mirror PREDICTS the next
        board and a desync means a real disagreement. Until the mod carried
        the machine form of a play, the advisor could only `reconcile` — it
        predicted nothing, so every ordinary change read as a divergence
        (2232 desyncs over 540 decisions on live run
        001-ironclad-a8-0824-1939, 42x the autopilot's rate) and the mirror
        re-seeded constantly instead of carrying anything.

        `human_action` now carries `card` (`id` + `upgraded`) and
        `target_index`. The card is located in the mirror's own hand by
        identity rather than by a wire index, so the two never have to agree
        on hand ORDER — only on contents.

        A potion use arrives the same way, as ``potion`` (``id``), and is
        followed through the engine's own drink action. It used to be the
        headline example of an action with no machine form, and so DROPPED the
        mirror every time: a fight in which the player drank anything lost its
        mirror at that moment, and with it every searched decision afterwards.

        Any action that still cannot be followed DROPS the mirror instead of
        letting it fall silently out of step: a card the mirror does not hold,
        a potion not in its belt, or an engine rejection. A dropped mirror
        re-seeds on the next board and costs advice quality; a wrong one costs
        correctness.

        """
        self._core.on_world_action(msg)
        self._flush()

    def source_is_replayable(self, source_model: str | None, source_kind: str | None) -> bool:
        """Can a replay possibly work for this source? Decided from the wire alone.

        Cheap and board-free, so a screen that was never answerable is
        declined BEFORE building a board and forking a mirror for it. Shared
        with :meth:`replay_opening_play` so the two can never disagree about
        what counts as replayable.

        """
        out = self._core.source_is_replayable(source_model, source_kind)
        self._flush()
        return out

    def replay_opening_drink(
        self,
        source_model: str | None,
        world_wire: dict[str, Any],
        *,
        target_index: Any = None,
        on: _MirrorView | None = None,
    ) -> bool:
        """Re-drink the potion that opened this screen, to reach its choice.

        The potion sibling of :meth:`replay_opening_play`, and simpler for one
        reason: ``source_model`` names the potion outright, so there is no
        "which of these was it" to settle. A card play has to be disambiguated
        against the board because several plays in hand could have opened the
        same screen; a potion is identified by the wire.

        The board proof therefore does a different job here. It is not
        choosing between candidates -- it is checking that this mirror's
        picture of the fight still matches the world's before a search runs on
        it. A mismatch DECLINES rather than dropping: the identity was never
        in doubt, so a mismatch says the mirror drifted, and
        :meth:`CombatMirror.reconcile` already re-seeds that on the next
        board. Dropping here would destroy a mirror on the strength of an
        enemy-HP check that an untargeted potion cannot move.

        Args:
            source_model: The potion's ``Id.Entry``, from the screen's wire.
            world_wire: The world's own board, as the match gate.
            target_index: The drink's target, when the wire named one.
            on: The mirror to replay into -- a fork, in the advisor lane.

        Returns:
            True when the mirror is now parked on the potion's own choice.

        """
        out = self._core.replay_opening_drink(
            self._which(on), source_model, world_wire, target_index
        )
        self._flush()
        return out

    def remember_picks(
        self, source_model: Any, offered: Sequence[Any], picks: Sequence[int]
    ) -> None:
        """Keep the picks a MULTI-pick search made, for the screens still to come.

        A card that discards two opens ONE engine choice with `max_select`
        two, and the search answers all of it at once -- the note says so:
        `combat MCTS - 2 pick(s) searched`. The mod then presents one advice
        request per pick, with the offered list shrinking by the card just
        taken. Only the first of those was ever searched; the rest replayed
        the opening play from scratch, so the engine parked a choice with the
        FULL option list against the world's shortened one, `inject_choice`
        could not reconcile them, and the fork was dropped. Run
        003-silent-a8-0825-0125: `70/70 replays` landed alongside 33 injection
        drops, and 21 decisions fell to the rarity sort with an answer already
        computed and thrown away.

        Stored by identity rather than index, because the indices are exactly
        what the next screen renumbers.

        """
        self._core.remember_picks(source_model, list(offered), [int(i) for i in picks])

    def next_remembered_pick(self, source_model: Any, offered: Sequence[Any]) -> int | None:
        """The next searched pick for this selection, as an index into `offered`.

        Returns None -- and forgets the selection -- unless the world's offer
        is EXACTLY the original minus the plan's own first N picks. That is
        the whole safety argument: the advisor recommends but never executes,
        so the world may have picked something else entirely, and the multiset
        check is what tells the two apart. A shrinking list is not proof on its
        own; the same list minus the same cards is.

        IDEMPOTENT, and that is the point. How far through the plan we are is
        re-derived from the offer every call (``len(offer) - len(now)``), so
        being asked twice for the same pick answers it twice with the same
        card. Consuming the plan on SERVE instead was wrong, because a serve
        is not an execution: the mod drops advice whose ``request_id`` has
        gone stale (131 such drops in run 007 alone) and then re-requests,
        which burned the cached pick and dropped the re-ask to the rarity
        sort. All 7 Hidden Daggers second picks in `logs/f7-mirror-check16`
        went that way -- every one of them with the answer already computed,
        the multiset matching, and `the second pick of the same search`
        logged one line above.

        """
        return self._core.next_remembered_pick(source_model, list(offered))

    def replay_end_turn(
        self,
        world_wire: dict[str, Any],
        *,
        on: _MirrorView | None = None,
    ) -> bool:
        """End the fork's turn to reach a screen the END OF TURN opened.

        The third replay kind, and the only one with no name to match. A card
        and a potion are replayed by re-issuing the action that opened the
        screen; here that action is ``end_turn``, which is unconditionally
        legal in the play phase -- so "which one was it" has no candidates and
        the identity proof the other two need has nothing to prove.

        What replaces it is stricter. Ending a turn normally runs the flush,
        the enemy turn and the next turn-setup and comes back to rest, so
        PARKING is not the common case: it happens only where the engine's own
        end-of-turn pipeline surfaced a player choice. "It parked, and on a
        kind this boundary is known to raise" is therefore the engine's own
        evidence about what the screen is -- not a name match against
        ``WELL_LAID_PLANS_POWER``.

        Why this class was invisible: the retain screen opens INSIDE the
        end-of-turn flush, before the round increments, and
        :meth:`_catch_up_turns` only ends the mirror's turn once the world's
        ``round`` has moved. So at the screen the mirror is still in the play
        phase and never parked itself. Live run 001-silent-a8-0825-0104: 146
        declines, every one of them this power, against 122 ranked decisions
        in the whole run -- 77% of everything the advisor had to rank.

        ``end_turn`` runs past the boundary and into the NEXT turn's setup, so
        a turn-START screen is this path's too. Tools of the Trade is the one
        that matters: its power draws +1 and then opens a ``FromHandForDiscard``
        inside ``AfterPlayerTurnStart`` (``ToolsOfTheTradePower.cs:27-37``), and
        the engine parks ``discard_from_hand`` there. Accepting only the retain
        kind sent 9 of them to the heuristic in the ``logs/f7-mirror-check15``
        corpus.

        ``TO_DISCARD`` in those logs is the PROMPT KEY, not the source, and
        ``FromHandForDiscard`` has two unrelated callers. The corpus's other 4
        are cards opening the same screen from their own ``OnPlay``
        (HiddenDaggers, Survivor, DaggerThrow, Prepared); those are
        :meth:`replay_opening_play`'s and are NOT fixed by this path.

        Args:
            world_wire: The world's own board, as the match gate. Comparable
                at either park point, because the probe stops exactly where
                the engine does: at a retain prompt neither side has run the
                enemy turn, at a turn-start discard both have. At the retain
                prompt it is also the REPAIR: a mismatch there is drift the
                turn end cannot have caused, so the fork is reconciled against
                this wire and re-probed (GH #593).
            on: The mirror to replay into -- a fork, in the advisor lane.

        Returns:
            True when the mirror is now parked on the end-of-turn choice.

        """
        out = self._core.replay_end_turn(self._which(on), world_wire)
        self._flush()
        return out

    def replay_opening_play(
        self,
        source_model: str | None,
        world_wire: dict[str, Any],
        *,
        source_kind: str | None = None,
        target_index: Any = None,
        upgraded: Any = None,
        on: _MirrorView | None = None,
    ) -> bool:
        """Play the card that opened a select screen, so the engine parks it.

        The autopilot's mirror is parked on the choice because it *played the
        card itself*; the advisor only learns a screen is up, so the play has
        to be reconstructed -- and then PROVEN, not assumed.

        How much has to be reconstructed depends on what the mod tells us.

        **Told (current builds).** ``source_kind`` says what the source IS by
        runtime type, and for a card ``target_index`` / ``upgraded`` say which
        play it was. Then there is exactly one candidate, the board check
        VERIFIES it, and a mismatch means the mirror is wrong -- which is a
        drop, not a decline.

        **Not told (older builds).** Fall back to proving it from the board:
        replay every legal play of that card on a clone and keep the one whose
        enemy-HP vector matches the world's. That settles which copy and which
        enemy at once, but it is inference: a damage prediction off by any
        amount matches nothing, and the screen goes unanswered
        (``no board matched replay of NEOWS_FURY``, live run
        001-ironclad-a8-0824-2150). Kept only for compatibility.

        A source that is not a card -- a potion (``TouchOfInsanity`` opens a
        ``FromHand`` select with ``source: this``), a power
        (``WELL_LAID_PLANS_POWER``), a relic -- has nothing to replay and is
        left to the unparked recognition in
        ``combat_mirror.recognize_card_select``, which already handles it.

        Args:
            source_model: The world's name for the effect that opened the
                screen (``CardSelectContextPatch``), when it captured one.
            world_wire: The wire built from the screen's embedded combat --
                the board AFTER the play, used as the match target.
            source_kind: ``"card"`` / ``"potion"`` / ``"power"`` / ... from the
                model's runtime type. ``None`` on an older mod build, which
                falls back to inferring the kind from the piles.
            target_index: The play's target as an index into the full enemy
                list, or ``None`` for an untargeted card.
            upgraded: Whether the copy played was upgraded.

        Returns:
            True when the mirror is now parked on the engine's own choice,
            with the suspended tail intact.

        """
        out = self._core.replay_opening_play(
            self._which(on), source_model, world_wire, source_kind, target_index, upgraded
        )
        self._flush()
        return out

    def answer_card_select(
        self,
        state: dict[str, Any],
        cfg: RecommendConfig | None,
        sims: int | None,
        picker: Any = None,
        picker_note: str | None = None,
    ) -> dict[str, Any] | None:
        """Answer an IN-COMBAT ``card_select`` with the combat MCTS, or give up.

        The autopilot answers this exact screen by search
        (``realgame_mcts.pick_card_select_mirrored``); what it has is a mirror
        already parked on the choice with the rest of the card suspended behind
        it. This method puts the advisor in the same position: it picks the
        replay that re-parks a FORK of the carried fight on the world's choice
        (already parked / end of turn / a potion / the opening play), searches
        it, drops the fork, and remembers the picks for a multi-pick follow-up.
        One code path with the driver, no second dialect to drift.

        ``picker`` / ``picker_note``: who picks on the parked fork instead of
        the combat MCTS, and the note that says so (Instinct,
        :mod:`realgame_instinct`); see ``realgame_mcts.pick_card_select_mirrored``.

        Returns ``None`` for every reason not to answer -- no embedded combat
        (a deck screen), no live mirror, an unreplayable source, a fork that
        could not be taken, or a search that failed -- each one counted as a
        decline so the stat line tells the whole story. The caller then falls to
        its heuristic rung.

        """
        adapted = None
        if picker is not None:

            def adapted(fork_state: Any, n: int, lo: int, hi: int) -> list[int]:
                picked = picker(_StateView(fork_state), n_options=n, min_select=lo, max_select=hi)
                return [int(i) for i in picked]

        try:
            return self._core.answer_card_select(
                state,
                asdict(cfg) if cfg is not None else None,
                sims,
                _wire_for,
                adapted,
                picker_note,
            )
        finally:
            self._flush()

    def note(self) -> str:
        """One-line state summary for the advice note / run log."""
        return self._core.note()
