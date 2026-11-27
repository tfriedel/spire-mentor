"""Instinct in the live game: the e2e student as the mod's advisor.

The mod lets the player pick the advisor's brain (PROTOCOL.md §5, v5.2,
``advice_request.brain``): **Search** (the MCTS for fights, the PPO run
policy for screens -- :mod:`realgame_advisor`) or **Instinct**, the e2e
student (``sts2-rs/crates/sts2-e2e``): one transformer that makes every
decision of a run, fights included, with no search. The browser workbench has
had it since #1156 (``docs/browser-workbench.md``, "Instinct as advisor");
this is the same student for the real game.

How it answers
--------------
The student reads an engine ``RunManager`` that stands AT the decision: its
tokenizer reads the run (deck, relics, map, the fight) and its legal moves
are the engine's. The workbench owns such a run; the advisor does not -- it
only sees the mod's messages. So every request rebuilds one from the
envelope the mod stamps on it (seed, act, floor, ``run_state``, the act map
with the visited coords, every RNG stream's cursor):

* the RUN, exactly as the workbench's state import does
  (:func:`graft_run`, the twin of ``RustGame.start_imported``): the run is
  rebuilt from its seed, so the maps, the boss and the encounter queue are
  the live run's; position, deck, relics, potions, HP, gold and the RNG
  cursors are grafted on;
* the FIGHT, exactly as the search builds it (the advisor's own combat
  adapter, and the fight mirror's carried state when it has one), dropped
  onto that run (``debug_enter_combat``).

The student's own per-run state -- the env driver's fight tracking and the
fight's intent log, a policy input -- is kept here across requests
(:class:`InstinctAdvisor`), fed every combat position the mod shows us.

A position Instinct cannot read (a surface not modelled yet, a graft the
engine refuses, options that do not match the screen) raises
:class:`InstinctUnavailable` with a player-facing reason; the advisor then
answers with the search and says why (``advice.instinct_unavailable``), so
autoplay never stalls on it.

What it answers with
--------------------
The usual ``advice`` shape, so the HUD, F6/F7, Autobattle, the coach's
choice chip and the AI partners need nothing new: one option per merged
class (the student's unit of choice -- two copies of Strike are one class),
its softmax probability on ``visit_share`` (the field the HUD draws as a bar)
and ``prob``, no ``p_win`` / ``q`` (a policy, not a value), and an
executable ``best``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import runtime_paths
import sts2_rs
from realgame_adapter import character_display_name, envelope_game_version

logger = logging.getLogger(__name__)

#: The student's weights, relative to the data root (dev: ``sts2-pycli/``;
#: frozen: the bundle root, where the spec replicates the layout).
WEIGHTS_RELPATH = Path("data/e2e/allchar_e2e_d4500_ema0999.e2ew")

#: The Fool's own student (GH #1535), relative to the data root: a v0.113.0
#: net, so it answers only a v0.113.0 game.
FOOL_WEIGHTS_RELPATH = Path("data/e2e/fool_e2e_d850_ema0996.e2ew")
#: The build the Fool's student plays (the Fool exists from v0.113.0 on).
FOOL_GAME_VERSION = "v0.113.0"
#: The live builds the Fool's student answers for, each on its own rules: its
#: own, and v0.114.0 until a net is trained there (GH #1924).
FOOL_GAME_VERSIONS: tuple[str, ...] = ("v0.113.0", "v0.114.0")

#: The characters the all-character student plays (GH #1197: one net for all
#: five). The Fool has its own net (:data:`FOOL_WEIGHTS_RELPATH`).
INSTINCT_CHARACTERS: tuple[str, ...] = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
#: The character a request that names none is rebuilt as.
INSTINCT_CHARACTER = INSTINCT_CHARACTERS[0]
#: The build the student was trained on.
INSTINCT_GAME_VERSION = "v0.111.0"
#: The live builds Instinct answers for. The student plays each on that
#: build's own rules (:func:`engine_game_version`), as it already did in a
#: fight: the fight mirror always ran the live build.
INSTINCT_GAME_VERSIONS: tuple[str, ...] = ("v0.111.0", "v0.113.0", "v0.114.0")


def engine_game_version(state: dict[str, Any]) -> str:
    """The build a live run is rebuilt on: the one the game is playing.

    Not the training build. v0.113.0 generates its maps differently (the
    rest-site row, the room-type queue, the path pruning; ported in #1319),
    so a v0.113.0 run rebuilt on v0.111.0 has another map: the player's
    room was missing from it ("the map does not match this run"), and every
    screen that rebuilds the run -- the map, the rewards, an in-fight card
    offer -- fell back to the search.
    """
    try:
        version = envelope_game_version(state)
    except Exception:
        return INSTINCT_GAME_VERSION
    return version if version in INSTINCT_GAME_VERSIONS else INSTINCT_GAME_VERSION


class InstinctUnavailable(Exception):
    """Instinct cannot answer this position; the message says why (player-facing)."""


@dataclass(frozen=True)
class InstinctPick:
    """The student's answer at one position, before it is shaped for the wire."""

    decision: dict[str, Any]
    """The peeked decision (``E2ePeek.decision``)."""
    probs: list[float]
    """Softmax per merged class."""
    best: int
    """The greedy class."""
    ms: float
    estimate: dict[str, Any] | None = None
    """The win probe's read of the position (GH #1222), when the weights carry one."""


def default_weights_path() -> Path:
    return runtime_paths.data_root() / WEIGHTS_RELPATH


def default_fool_weights_path() -> Path:
    return runtime_paths.data_root() / FOOL_WEIGHTS_RELPATH


def _norm_char(raw: Any) -> str:
    return (character_display_name(raw) or str(raw or "")).strip().lower()


# ---------------------------------------------------------------------------
# The run graft (twin of workbench.backends.rust_game.RustGame.start_imported)
# ---------------------------------------------------------------------------


#: ``graft_run``'s note when a co-op run stands on the nearest node of a map
#: that is not the screen's.
COOP_MAP_APPROX: str = sts2_rs.GRAFT_COOP_MAP_APPROX


def graft_run(state: dict[str, Any]) -> tuple[Any, list[str]]:
    """Rebuild the live run behind ``state`` as an engine ``RunManager``.

    The same steps, in the same load-bearing order, as the workbench's state
    import (place on the map, empty the deck, relics, deck, potions, HP/gold,
    RNG cursors last), on a run built the way the student's campaign builds
    its own (``mode="agent"``, interactive potion swap on), on the live
    build (:func:`engine_game_version`). Runs in Rust
    (``sts2_advisor::graft``, GH #1426): the Spoils Map act, co-op maps and
    the map-less fight placement are documented there.

    Returns:
        ``(rm, notes)``: the run, parked in ``map_choice`` at the player's
        node, and what could not be restored.

    Raises:
        InstinctUnavailable: no run to rebuild, or the engine refuses it.
    """
    import sts2_rs

    try:
        return sts2_rs.graft_run(state, engine_game_version(state))
    except sts2_rs.GraftUnavailableError as exc:
        raise InstinctUnavailable(str(exc)) from exc


def is_coop(state: dict[str, Any]) -> bool:
    """A multiplayer run (``player_count`` > 1): AI partners, Steam co-op."""
    try:
        return int(state.get("player_count") or 1) > 1
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------


def _room_kind(state: dict[str, Any]) -> str:
    raw = str(state.get("room_type") or "").lower()
    return raw if raw in ("monster", "elite", "boss") else "monster"


def enter_live_combat(rm: Any, cs: Any, state: dict[str, Any], room: str) -> None:
    """Install the live fight ``cs`` on the grafted run ``rm``, naming its
    encounter when the mod sent one (``encounter_id``, on the board or on an
    in-combat screen's embedded ``combat`` board): the student reads it
    (RUNCTX ``X_ID_AUX``, GH #1197), and that cell alone can move a pick. An
    event fight's id stages that event's fight (GH #1477). An id the engine
    has no fight for leaves the run without one, as before. Rust:
    ``sts2_advisor::instinct_combat::enter_live_combat``."""
    note = sts2_rs.instinct_enter_live_combat(rm, cs, state, room)
    if note is not None:
        logger.info("%s", note)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def note_for(pick: InstinctPick, p_best: float) -> str:
    """``"Instinct (e2e net, no search) · <decision> · <p>% · <ms> ms"``."""
    return sts2_rs.instinct_note_for(pick.decision, pick.ms, p_best)


# ---------------------------------------------------------------------------
# The advisor
# ---------------------------------------------------------------------------


class InstinctAdvisor:
    """The student, loaded once, plus its per-run state.

    Rust: ``sts2_advisor::instinct_advisor`` (GH #1426 step 4) loads the net,
    keeps the run's driver and screen trail, and dispatches each request;
    this class keeps the Python name over ``sts2_rs.InstinctAdvisorCore``.
    Thread-safe: requests arrive on the advisor's socket loop, but the
    control plane may ask from its HTTP threads.
    """

    def __init__(self, weights: Path | None = None, fool_weights: Path | None = None) -> None:
        self._weights = weights or default_weights_path()
        self._fool_weights = fool_weights or default_fool_weights_path()
        self._core = sts2_rs.InstinctAdvisorCore(self._weights, self._fool_weights)

    @staticmethod
    def _emit(log: list[tuple[str, str]]) -> None:
        for level, line in log:
            getattr(logger, level)("%s", line)

    # -- per-run state (read by the screen handlers and the tests) ------

    @property
    def last_room(self) -> str | None:
        """The last fight's room type (``monster`` / ``elite`` / ``boss``): the
        card reward after it is that room's (a ``?`` fight included)."""
        return self._core.last_room

    @last_room.setter
    def last_room(self, value: str | None) -> None:
        self._core.last_room = value

    @property
    def last_screen(self) -> str | None:
        """The ``advice_for`` of the previous request: what opened a card screen
        (a smith after ``rest_site``, a removal after ``shop``)."""
        return self._core.last_screen

    @last_screen.setter
    def last_screen(self, value: str | None) -> None:
        self._core.last_screen = value

    @property
    def shop_removal(self) -> tuple[str, bool, str] | None:
        """The deck card Instinct removes in the shop (the engine picks it
        inside the shop decision; the game asks on the next screen)."""
        return self._core.shop_removal

    @shop_removal.setter
    def shop_removal(self, value: tuple[str, bool, str] | None) -> None:
        self._core.shop_removal = value

    @property
    def event_track(self) -> dict[str, Any] | None:
        """The current event's trail of taken options."""
        return self._core.event_track

    @event_track.setter
    def event_track(self, value: dict[str, Any] | None) -> None:
        self._core.event_track = value

    @property
    def last_pick(self) -> InstinctPick | None:
        """The last position a screen handler asked the student about."""
        pick = self._core.last_pick
        if pick is None:
            return None
        return InstinctPick(
            decision=pick["decision"],
            probs=list(pick["probs"]),
            best=int(pick["best"]),
            ms=float(pick["ms"]),
        )

    # -- loading -------------------------------------------------------

    @property
    def available(self) -> bool:
        ok, log = self._core.available()
        self._emit(log)
        return ok

    def capabilities(self) -> list[dict[str, str]]:
        """``capabilities.instinct``: what this advisor's Instinct plays."""
        caps, log = self._core.capabilities()
        self._emit(log)
        return caps

    # -- per-run state -------------------------------------------------

    def reset(self, reason: str) -> None:
        """Forget the student's per-run state (run over, rewind, new build)."""
        line = self._core.reset(reason)
        if line is not None:
            logger.info("%s", line)

    def remap_enemies(self, pairs: list[tuple[int, int]]) -> None:
        """Follow the live fight's enemies through renamed uids
        (``CombatMirror.take_enemy_remap``, GH #1651)."""
        self._core.remap_enemies(pairs)

    def on_human_action(self, msg: dict[str, Any]) -> None:
        """Follow a move the player made (``human_action``): an event page's
        choice, so the next page of that event is replayed right."""
        self._core.on_human_action(msg)

    def rust_screen(self, screen: str, state: dict[str, Any]) -> dict[str, Any]:
        """A screen whose handler runs in Rust (``sts2_advisor::instinct_screens``)."""
        from display_names import _names

        kind, payload, log = self._core.screen(screen, state, _names())
        self._emit(log)
        return _outcome(kind, payload)

    # -- answering ----------------------------------------------------

    def eligibility(self, state: dict[str, Any]) -> str | None:
        """Why Instinct cannot play this run, or ``None`` when it can."""
        return sts2_rs.InstinctAdvisorCore.eligibility(state)

    def advise(
        self,
        state: dict[str, Any],
        advice_for: str,
        *,
        cfg: Any = None,
        mirror: Any = None,
        session: dict[str, Any] | None = None,
        sims: int | None = None,
    ) -> dict[str, Any]:
        """Instinct's ``advice`` for one ``advice_request`` (``state``: its payload).

        Raises:
            InstinctUnavailable: Instinct cannot answer here (the reason is
                player-facing); the caller answers with the search.
        """
        return self._call(self._core.advise, state, advice_for, cfg, mirror, session, sims)

    def _dispatch(
        self,
        state: dict[str, Any],
        advice_for: str,
        cfg: Any,
        mirror: Any,
        session: dict[str, Any] | None,
        sims: int | None,
    ) -> dict[str, Any]:
        """:meth:`advise` without its eligibility check and net load: what
        answers without a net (a remembered pick, a confirm) still answers."""
        return self._call(self._core.dispatch, state, advice_for, cfg, mirror, session, sims)

    def _advise_combat_select(
        self, state: dict[str, Any], cfg: Any, mirror: Any, sims: int | None
    ) -> dict[str, Any]:
        """An in-combat ``card_select`` (Burning Pact, a Skill Potion's offer):
        Instinct picks on the fight mirror's fork, one pick at a time (Rust:
        ``sts2_advisor::instinct_combat::advise_choice``)."""
        return self._call(self._core.combat_select, state, None, cfg, mirror, None, sims)

    def _call(
        self,
        method: Any,
        state: dict[str, Any],
        advice_for: str | None,
        cfg: Any,
        mirror: Any,
        session: dict[str, Any] | None,
        sims: int | None,
    ) -> dict[str, Any]:
        from display_names import _names

        lead = (state,) if advice_for is None else (state, advice_for)
        try:
            kind, payload, log = method(
                *lead,
                _names(),
                dataclasses.asdict(cfg) if cfg is not None else None,
                mirror._core if mirror is not None else None,
                session,
                sims,
            )
        finally:
            if mirror is not None:
                mirror._flush()
        self._emit(log)
        return _outcome(kind, payload)

    def evaluate(self, state: dict[str, Any], rm: Any, expect: tuple[str, ...]) -> InstinctPick:
        """Peek ``rm`` with this run's driver state and run the student
        (``sts2_advisor::instinct::evaluate``, GH #1426).

        ``expect``: the decision types this screen can be; anything else means
        the rebuilt run is not where the screen is.
        """
        out, why = self._core.evaluate(state, rm, list(expect))
        if out is None:
            raise InstinctUnavailable(why)
        return InstinctPick(
            decision=out["decision"],
            probs=[float(p) for p in out["probs"]],
            best=int(out["best"]),
            ms=float(out["ms"]),
            estimate=out.get("estimate"),
        )

    def choice_picks(self, state: dict[str, Any], rm: Any, max_select: int) -> list[int]:
        """The student's picks on the in-fight choice ``rm`` stands on: the
        native run's COMBAT_CHOICE decisions, one pick at a time (GH #1667)."""
        picks, why = self._core.choice_picks(state, rm, int(max_select))
        if picks is None:
            raise InstinctUnavailable(why)
        return [int(p) for p in picks]


def _outcome(kind: str, payload: Any) -> dict[str, Any]:
    """The core's ``(kind, payload)`` as the advice, or the exception the
    Python advisor raised."""
    if kind == "advice":
        return payload
    if kind == "unavailable":
        raise InstinctUnavailable(payload)
    exc_type, message = payload
    if exc_type == "OverflowError":
        raise OverflowError(message)
    raise ValueError(message)
