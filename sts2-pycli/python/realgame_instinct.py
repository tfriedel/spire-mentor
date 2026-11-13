"""Instinct in the live game: the e2e Ironclad student as the mod's advisor.

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

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import runtime_paths
from realgame_adapter import character_display_name, envelope_game_version

logger = logging.getLogger(__name__)

#: The student's weights, relative to the data root (dev: ``sts2-pycli/``;
#: frozen: the bundle root, where the spec replicates the layout).
WEIGHTS_RELPATH = Path("data/e2e/ironclad_e2e_d2500_avg5_ema0999.e2ew")

#: What this student plays. A tuple per character as more students land.
INSTINCT_CHARACTER = "Ironclad"
#: The build the student was trained on. The run graft and the screens' own
#: runs are built on it, whichever live build asks.
INSTINCT_GAME_VERSION = "v0.111.0"
#: The live builds Instinct answers for. The v0.113.0 beta seeds like
#: v0.111.0 and the engine runs it on v0.111.0's rules wherever its content is
#: not ported yet (#1303), so the student plays it on its training build; the
#: advisor labels every beta answer approximate (``mark_beta_build``).
INSTINCT_GAME_VERSIONS: tuple[str, ...] = ("v0.111.0", "v0.113.0")


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


def default_weights_path() -> Path:
    return runtime_paths.data_root() / WEIGHTS_RELPATH


def _norm_char(raw: Any) -> str:
    return (character_display_name(raw) or str(raw or "")).strip().lower()


# ---------------------------------------------------------------------------
# The run graft (twin of workbench.backends.rust_game.RustGame.start_imported)
# ---------------------------------------------------------------------------


#: The act whose map the Spoils Map quest card replaces (``SpoilsActIndex = 1``,
#: 0-based, ``SpoilsMap.cs:47-50``).
SPOILS_MAP_ACT = 2

#: ``graft_run``'s note when a co-op run stands on the nearest node of a map
#: that is not the screen's.
COOP_MAP_APPROX = "co-op: placed on the nearest node of a map that differs"


def graft_run(state: dict[str, Any]) -> tuple[Any, list[str]]:
    """Rebuild the live run behind ``state`` as an engine ``RunManager``.

    The same steps, in the same load-bearing order, as the workbench's state
    import (``RustGame.start_imported``: place on the map, empty the deck,
    relics, deck, upgrades, potions, HP/gold, RNG cursors last), on a run built
    the way the student's campaign builds its own (``mode="agent"``,
    interactive potion swap on: ``sts2_e2e::env::new_run_at``), so the engine
    offers exactly the decisions the student was trained on.

    Returns:
        ``(rm, notes)``: the run, parked in ``map_choice`` at the player's
        node, and what could not be restored.

    Raises:
        InstinctUnavailable: no run to rebuild, or the engine refuses it.
    """
    import sts2_rs
    from floors import engine_total_floor
    from workbench.backends.rust_game import _map_disagreements
    from workbench.state_import import StateImportError, _run_snapshot

    notes: list[str] = []
    try:
        snap = _run_snapshot(state, notes)
    except StateImportError as exc:
        raise InstinctUnavailable(f"no run to rebuild ({exc})") from exc
    deck = (state.get("run_state") or {}).get("master_deck") or []
    spoils_quest = any(
        isinstance(c, dict) and str(c.get("id") or "").upper().endswith("SPOILS_MAP") for c in deck
    )

    def new_run(spoils: bool) -> Any:
        try:
            rm = sts2_rs.RunManager(
                game_version=INSTINCT_GAME_VERSION,
                string_seed=snap.seed,
                character=snap.character,
                ascension_level=snap.ascension,
                neow_start=False,
                mode="agent",
                interactive_potion_swap=True,
                act1="roll",
            )
        except Exception as exc:
            raise InstinctUnavailable(f"the engine cannot rebuild this run ({exc})") from exc
        if is_coop(state):
            # The game builds co-op act maps one room shorter, from the same
            # map RNG (`ActModel.GetNumberOfRooms(isMultiplayer)`): the engine
            # then builds exactly the screen's map (19 of 19 recorded co-op
            # acts).
            rm.debug_use_coop_maps()
        if spoils:
            # The Spoils Map quest card swaps act 2's map for the hourglass
            # one as the act is built (`SpoilsMap.ModifyGeneratedMap`), and
            # the deck is restored only after placement (below): put the
            # card in first, or act 2 is built as a standard map.
            rm.debug_set_master_deck([{"id": "SPOILS_MAP", "upgraded": 0}])
        return rm

    def place(rm: Any, col: int, row: int) -> None:
        rm.debug_place_on_map(
            snap.act,
            col,
            row,
            list(snap.visited),
            row + 1,
            engine_total_floor(snap.floor, snap.act),
        )

    rm = new_run(spoils_quest)
    coord = snap.coord
    if coord is not None:
        col, row = coord
        try:
            try:
                place(rm, col, row)
                if not spoils_quest and snap.act == SPOILS_MAP_ACT and _map_disagreements(rm, snap):
                    raise ValueError("act 2's standard map is not the screen's")
            except ValueError:
                if spoils_quest or snap.act != SPOILS_MAP_ACT:
                    raise
                # The quest is done (the card left the deck) but its act keeps
                # the hourglass map: rebuild with the card for the map, and
                # keep that only if it is the screen's.
                spoils_rm = new_run(True)
                place(spoils_rm, col, row)
                if _map_disagreements(spoils_rm, snap):
                    raise
                rm = spoils_rm
        except ValueError as exc:
            if not is_coop(state):
                raise InstinctUnavailable(f"the map does not match this run ({exc})") from exc
            # Co-op, and the co-op map still differs (not seen in 19
            # recorded acts). A fight still has its answer: stand on the
            # nearest node of the engine's map in the same act, so act, floor
            # and the room's place in the act stay right. The map handler
            # refuses to route on it (COOP_MAP_APPROX).
            _place_nearest(rm, snap.act, col, row, engine_total_floor(snap.floor, snap.act))
            notes.append(COOP_MAP_APPROX)
        else:
            notes.extend(_map_disagreements(rm, snap))
    else:
        # Fight requests carry no map graph (GH #1212). Left unplaced, the run
        # sat at act 1 / floor 0, and the student's combat row read every
        # fight as the first room of the run with every boss ahead. Place by
        # the screen's act and floor instead.
        row = _place_by_floor(rm, snap.act, snap.floor, _graft_room(state))
        notes.append(f"no map position: placed by floor on row {row}")

    # Deck emptied BEFORE the relics (on-obtain upgrades must not chew on it),
    # then set whole -- ids, upgrades AND enchantments, which the tokenizer
    # reads and the workbench's per-card graft drops -- with no hook firing.
    for name, _upgraded in list(rm.master_deck_detailed()):
        rm.force_remove_card(name)
    for relic_id in snap.relics:
        rm.force_add_relic(relic_id)
    # The pickup hooks queue the choices the live run already made (Lead
    # Paperweight's choose screen): left queued, the first one surfaces in
    # place of the next real screen (GH #1183).
    rm.debug_clear_pending_choices()
    import realgame_instinct_screens as screens

    deck = (state.get("run_state") or {}).get("master_deck") or []
    missing = rm.debug_set_master_deck([screens.card_spec(c) for c in deck if isinstance(c, dict)])
    if missing:
        notes.append("deck cards not built: " + ", ".join(missing))
    rm.debug_set_potion_state(list(snap.potions))
    rm.set_player(hp=snap.hp, max_hp=snap.max_hp, gold=snap.gold)
    if snap.rng is not None:
        try:
            ignored = rm.debug_set_rng_counters(snap.rng["run_seed"], snap.rng["counters"])
        except ValueError as exc:
            notes.append(f"RNG cursors not restored ({exc})")
        else:
            run_level = [n for n in ignored if not n.startswith("combat_")]
            if run_level:
                notes.append("RNG streams not carried: " + ", ".join(run_level))
    return rm, notes


def is_coop(state: dict[str, Any]) -> bool:
    """A multiplayer run (``player_count`` > 1): AI partners, Steam co-op."""
    try:
        return int(state.get("player_count") or 1) > 1
    except (TypeError, ValueError):
        return False


def _place_nearest(rm: Any, act: int, col: int, row: int, total_floor: int) -> None:
    """Place ``rm`` on the engine-map node of ``act`` nearest ``(col, row)``.

    Raises:
        InstinctUnavailable: no node of that act takes the placement.
    """
    # The failed placement already advanced the run to ``act`` and generated
    # its map (`debug_place_on_map` walks the acts before checking the node),
    # so the engine's own act map is readable here.
    full = rm.full_map() or {}
    nodes = [(n["col"], n["row"]) for r in full.get("rows") or [] for n in r]
    for c, r in sorted(nodes, key=lambda cr: (abs(cr[1] - row), abs(cr[0] - col))):
        try:
            rm.debug_place_on_map(act, c, r, [], r + 1, total_floor)
            return
        except ValueError:
            continue
    raise InstinctUnavailable("no node of the engine's map takes this co-op position")


#: Node types a map-less graft prefers for a room kind, best first. A ``?``
#: room can hold a fight; any node of the row keeps the position right.
_ROOM_NODE_TYPES = {
    "monster": ("Monster", "Unknown"),
    "elite": ("Elite", "Unknown"),
}


def _graft_room(state: dict[str, Any]) -> str | None:
    """The fight's room kind (``monster`` / ``elite`` / ``boss``), if the request has one."""
    for src in (state, state.get("combat")):
        if isinstance(src, dict) and src.get("room_type"):
            return _room_kind(src)
    return None


def _enter_act(rm: Any, act: int) -> dict[str, Any]:
    """Advance ``rm`` to ``act`` and return the engine's map of it.

    ``debug_place_on_map`` walks the acts (generating each map off its own
    map RNG) before it checks the node, so an off-map coordinate advances the
    run and is then refused.
    """
    try:
        rm.debug_place_on_map(act, -1, -1, [], 0, 0)
    except ValueError:
        pass
    return rm.full_map() or {}


def _place_by_floor(rm: Any, act: int, floor: int, room: str | None) -> int:
    """Place ``rm`` on a node of ``act`` in the row the game ``floor`` stands on.

    The engine's total floor counts each earlier act's rows up to its boss
    (``act 1 floor 17`` = its boss row 16, ``act 2 floor 18`` = row 0), and the
    run is rebuilt from the live seed, so the engine's act maps are the
    screen's. A fight's tokens read the act, the floors, the current row and
    the boss rows -- not the column -- so any node of that row, of the room's
    kind when the row has one, gives the position the screen is in.

    Returns:
        The act-local row placed on.

    Raises:
        InstinctUnavailable: no node of the act takes the placement.
    """
    from floors import engine_total_floor

    total = engine_total_floor(floor, act)
    before = 0
    fm: dict[str, Any] = {}
    for a in range(1, act + 1):
        fm = _enter_act(rm, a)
        if a < act:
            before += int((fm.get("boss") or {}).get("row") or len(fm.get("rows") or []))
    row = max(0, total - before)
    boss = fm.get("boss") or {}
    boss_row = int(boss.get("row") or len(fm.get("rows") or []))
    second = fm.get("second_boss") or {}
    if room == "boss" or row >= boss_row:
        pt = second if row > boss_row and second.get("row") is not None else boss
        if pt.get("row") is not None:
            r = int(pt["row"])
            rm.debug_place_on_map(act, int(pt["col"]), r, [], r + 1, total)
            return r
        row = boss_row - 1
    nodes = [n for r in fm.get("rows") or [] for n in r if int(n.get("row", -1)) == row]
    prefer = _ROOM_NODE_TYPES.get(room or "", ())
    nodes.sort(key=lambda n: prefer.index(n["type"]) if n.get("type") in prefer else len(prefer))
    for n in nodes:
        try:
            rm.debug_place_on_map(act, int(n["col"]), row, [], row + 1, total)
            return row
        except ValueError:
            continue
    raise InstinctUnavailable(f"no node of act {act}'s map stands on floor {floor}")


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------


def _room_kind(state: dict[str, Any]) -> str:
    raw = str(state.get("room_type") or "").lower()
    return raw if raw in ("monster", "elite", "boss") else "monster"


def _combat_form(run_action: dict[str, Any]) -> dict[str, Any]:
    """A combat action's run form (``sts2_e2e::env::combat_to_rm``) back in
    the legal-action form the advisor's mapping reads."""
    name = run_action.get("action")
    if name == "play_card":
        return {
            "action": "play",
            "card_index": run_action.get("hand_idx"),
            "target_index": run_action.get("target_index"),
        }
    if name == "drink_potion":
        return {
            "action": "drink_potion",
            "slot": run_action.get("potion_slot"),
            "target_index": run_action.get("target_index"),
        }
    if name == "combat_resolve_choice":
        option = run_action.get("option")
        return {"action": "choose", "selection": [] if option is None else [option]}
    if name == "combat_resolve_choice_selection":
        return {"action": "choose", "selection": list(run_action.get("selection") or [])}
    return dict(run_action)


def forced_pick(forced: dict[str, Any], ms: float) -> InstinctPick:
    """The pick for a position with one move: that move, at probability 1."""
    action = forced.get("action") or {}
    combat = forced.get("kind") == "combat"
    decision = {
        "decision_type": "FORCED",
        "is_combat": combat,
        "classes": [{"class": 0, "actions": [_combat_form(action) if combat else action]}],
    }
    return InstinctPick(decision=decision, probs=[1.0], best=0, ms=ms)


def combat_state_for(state: dict[str, Any], cfg: Any, mirror: Any) -> tuple[Any, dict[str, Any]]:
    """The engine fight behind a ``combat_action`` board, as the search builds it.

    The advisor's own path (``realgame_advisor.compute_combat_advice``): the
    mod's board through ``to_headless_combat_play`` + ``adapter.translate``,
    and the fight mirror's CARRIED state when it is following this fight
    (hidden per-turn counters no wire carries). Feeding the mirror the board
    here also keeps it in step while Instinct, not the search, answers.

    Returns:
        ``(cs, wire)``: the native combat state and the engine wire dict
        (whose ``deck.hand`` maps engine card indices to the mod's).
    """
    import sts2_rs
    from realgame_adapter import run_ascension, to_headless_combat_play
    from sts2_hint import adapter

    run_state = state.get("run_state") or {}
    headless = to_headless_combat_play(state)
    wire = adapter.translate(
        headless,
        character=character_display_name(run_state.get("character")) or INSTINCT_CHARACTER,
        combat_id=getattr(cfg, "combat_id", None),
        seed=getattr(cfg, "seed", 0),
        ascension=run_ascension(run_state, getattr(cfg, "ascension", 0)),
    )
    live = mirror.on_combat_board(wire, state) if mirror is not None else None
    if live is not None and live.state.pending_choice() is None:
        try:
            return live.state.clone(), wire
        except Exception as exc:  # the rebuild below is always available
            logger.debug("instinct: cannot clone the mirror: %s", exc)
    return sts2_rs.state_from_dict(wire), wire


def combat_advice(
    pick: InstinctPick, state: dict[str, Any], wire: dict[str, Any]
) -> dict[str, Any]:
    """The wire ``advice`` for a fight decision, in the search's own option shape.

    Every merged class becomes one row named by its representative (the
    student plays the class's first member, as the env does); its
    ``stable_idx`` / ``target_index`` / ``card_index`` are mapped through the
    SAME helper the search's rows go through, so the mod reads them
    identically. A class the game would refuse (``veto_unplayable``) hands
    the crown to the student's next choice, never to an arbitrary row.
    """
    import realgame_advisor as ra
    from sts2_hint.recommender import HintTag, UnmappableAction, _stable_idx_for_action

    hand = (wire.get("deck") or {}).get("hand") or []
    order = sorted(range(len(pick.probs)), key=lambda c: -pick.probs[c])
    rank = {c: r for r, c in enumerate(order)}
    tags: list[HintTag] = []
    for cls in pick.decision["classes"]:
        ci = int(cls["class"])
        action = (cls.get("actions") or [None])[0]
        if not isinstance(action, dict):
            continue
        try:
            stable_idx = _stable_idx_for_action(action, hand)
        except (UnmappableAction, KeyError) as exc:
            logger.warning("instinct: class %d has no mod form (%s)", ci, exc)
            continue
        target = action.get("target_index")
        ally = action.get("ally_index")
        p = float(pick.probs[ci])
        tags.append(
            HintTag(
                stable_idx=stable_idx,
                win_prob=p,
                bucket_label="",
                is_best=ci == pick.best,
                visits=0,
                q_saturated=False,
                low_confidence=False,
                target_index=int(target) if target is not None else None,
                q=0.0,
                ally_index=int(ally) if ally is not None else None,
                pick_rank=rank[ci],
            )
        )
    if not tags:
        raise InstinctUnavailable("none of Instinct's moves map onto this board")
    tags, vetoed = ra.veto_unplayable(tags, state)
    if not tags:
        raise InstinctUnavailable("the game refuses every move Instinct ranks")
    tags.sort(key=lambda t: t.pick_rank if t.pick_rank is not None else 1 << 30)
    best = next((t for t in tags if t.is_best), tags[0])
    options = []
    for t in tags:
        options.append(
            {
                "stable_idx": t.stable_idx,
                "card_index": t.stable_idx if t.stable_idx >= 0 else None,
                "target_index": t.target_index,
                **({"ally_index": t.ally_index} if t.ally_index is not None else {}),
                **ra.option_move(t, state),
                "label": ra._tag_label(t, state),
                "visit_share": round(t.win_prob, 4),
                "prob": round(t.win_prob, 4),
                "is_best": t is best,
                "pick_rank": t.pick_rank,
            }
        )
    advice: dict[str, Any] = {
        "note": note_for(pick, best.win_prob),
        "options": options,
    }
    best_action = ra._best_to_action(best, state)
    if best_action is not None:
        best_action["label"] = ra._tag_label(best, state)
        advice["best"] = best_action
    else:
        raise InstinctUnavailable("Instinct's pick has no executable form here")
    if vetoed:
        advice["note"] += f" · vetoed {len(vetoed)} the game calls unplayable"
    return advice


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def map_advice(pick: InstinctPick, state: dict[str, Any]) -> dict[str, Any]:
    """The wire ``advice`` for ``map_select``: each class is a ``move`` to a column.

    The engine's move names the child by its column; the mod's ``nodes[]``
    name the same children with their ``index`` (what the takeover clicks)
    and their coordinate (where the HUD pins the score).
    """
    nodes = [n for n in (state.get("nodes") or []) if isinstance(n, dict)]
    by_col = {int(n["col"]): n for n in nodes if isinstance(n.get("col"), int)}
    options = []
    for cls in pick.decision["classes"]:
        ci = int(cls["class"])
        action = (cls.get("actions") or [None])[0] or {}
        if action.get("action") != "move":
            raise InstinctUnavailable(f"unexpected map move {action}")
        node = by_col.get(int(action.get("col", -1)))
        if node is None:
            raise InstinctUnavailable("Instinct's map moves do not match the screen's nodes")
        p = float(pick.probs[ci])
        options.append(
            {
                "label": _node_label(node),
                "visit_share": round(p, 4),
                "prob": round(p, 4),
                "is_best": ci == pick.best,
                "index": int(node.get("index", nodes.index(node))),
                "action": "choose",
                "col": int(node["col"]),
                "row": int(node["row"]),
            }
        )
    if len(options) != len(nodes):
        raise InstinctUnavailable(f"Instinct sees {len(options)} paths, the map shows {len(nodes)}")
    best = next(o for o in options if o["is_best"])
    return {
        "note": note_for(pick, best["prob"]),
        "options": sorted(options, key=lambda o: -o["prob"]),
        "best": {"action": "choose", "index": best["index"], "label": best["label"]},
    }


def _node_label(node: dict[str, Any]) -> str:
    import realgame_ppo

    return realgame_ppo._map_labels([node])[0]


def note_for(pick: InstinctPick, p_best: float) -> str:
    dtype = str(pick.decision.get("decision_type") or "").lower().replace("_", " ")
    return f"Instinct (e2e net, no search) · {dtype} · {p_best:.0%} · {pick.ms:.0f} ms"


# ---------------------------------------------------------------------------
# The advisor
# ---------------------------------------------------------------------------


class InstinctAdvisor:
    """The student, loaded once, plus its per-run state.

    Thread-safe: requests arrive on the advisor's socket loop, but the
    control plane may ask from its HTTP threads.
    """

    def __init__(self, weights: Path | None = None) -> None:
        self._weights = weights or default_weights_path()
        self._lock = threading.RLock()
        self._student: Any = None
        self._error: str | None = None
        self._loaded = False
        self._run_key: tuple[Any, ...] | None = None
        self._driver: Any = None
        #: The last fight's room type (``monster`` / ``elite`` / ``boss``): the
        #: card reward after it is that room's (a ``?`` fight included).
        self.last_room: str | None = None
        #: The ``advice_for`` of the previous request: what opened a card screen
        #: (a smith after ``rest_site``, a removal after ``shop``).
        self.last_screen: str | None = None
        #: The deck card Instinct removes in the shop (the engine picks it
        #: inside the shop decision; the game asks on the next screen).
        self.shop_removal: tuple[str, bool, str] | None = None
        #: The current event's trail of taken options
        #: (``realgame_instinct_screens._event_track``).
        self.event_track: dict[str, Any] | None = None

    # -- loading -------------------------------------------------------

    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import sts2_rs

            if not self._weights.is_file():
                self._error = f"no Instinct net at {self._weights}"
                return
            self._student = sts2_rs.E2eStudent(str(self._weights))
            logger.info("instinct: loaded %s", self._weights.name)
        except Exception as exc:
            self._error = f"Instinct net failed to load: {exc}"
            logger.warning("instinct: %s", self._error)

    @property
    def available(self) -> bool:
        with self._lock:
            self._load_locked()
            return self._student is not None

    def capabilities(self) -> list[dict[str, str]]:
        """``capabilities.instinct``: what this advisor's Instinct plays."""
        if not self.available:
            return []
        return [
            {"character": INSTINCT_CHARACTER, "game_version": v} for v in INSTINCT_GAME_VERSIONS
        ]

    # -- per-run state -------------------------------------------------

    def reset(self, reason: str) -> None:
        """Forget the student's per-run state (run over, rewind, new build)."""
        with self._lock:
            if self._driver is not None:
                logger.info("instinct: state dropped (%s)", reason)
            self._run_key = None
            self._driver = None
            self.last_room = None
            self.last_screen = None
            self.shop_removal = None
            self.event_track = None

    def _driver_for(self, state: dict[str, Any], rm: Any) -> Any:
        import sts2_rs

        key = (state.get("seed"), _norm_char((state.get("run_state") or {}).get("character")))
        if key != self._run_key or self._driver is None:
            self._run_key = key
            self._driver = sts2_rs.E2eDriver(int(rm.seed))
        return self._driver

    def on_human_action(self, msg: dict[str, Any]) -> None:
        """Follow a move the player made (``human_action``): an event page's
        choice, so the next page of that event is replayed right."""
        if msg.get("decision_type") != "event":
            return
        import realgame_instinct_screens as screens

        with self._lock:
            screens.note_event_choice(self, msg)

    # -- answering ----------------------------------------------------

    def eligibility(self, state: dict[str, Any]) -> str | None:
        """Why Instinct cannot play this run, or ``None`` when it can."""
        character = _norm_char((state.get("run_state") or {}).get("character"))
        if character != INSTINCT_CHARACTER.lower():
            return f"Instinct plays {INSTINCT_CHARACTER} only"
        try:
            version = envelope_game_version(state)
        except Exception:
            version = None
        if version not in INSTINCT_GAME_VERSIONS:
            builds = " and ".join(INSTINCT_GAME_VERSIONS)
            return f"Instinct plays {builds} only (this game is {version})"
        return None

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
        why = self.eligibility(state)
        if why is not None:
            raise InstinctUnavailable(why)
        with self._lock:
            self._load_locked()
            if self._student is None:
                raise InstinctUnavailable(self._error or "no Instinct net loaded")
            try:
                advice = self._dispatch(state, advice_for, cfg, mirror, session, sims)
            finally:
                self.last_screen = advice_for
        advice["brain"] = "instinct"
        if is_coop(state):
            advice["note"] = f"{advice.get('note', '')} · co-op: Instinct trained single-player"
        return advice

    def _dispatch(
        self,
        state: dict[str, Any],
        advice_for: str,
        cfg: Any,
        mirror: Any,
        session: dict[str, Any] | None,
        sims: int | None,
    ) -> dict[str, Any]:
        if advice_for == "combat_action":
            return self._advise_combat(state, cfg, mirror)
        if advice_for == "map_select":
            return self._advise_map(state, session)
        if advice_for == "card_select" and state.get("combat") is not None:
            return self._advise_combat_select(state, cfg, mirror, sims)
        if advice_for == "card_reward":
            import realgame_advisor as ra

            if ra.is_fight_offer(state):
                # A card generated into the fight (Discovery, a Skill
                # Potion's offer): the in-combat selection it is, on the
                # fight mirror's park.
                return self._advise_combat_select(
                    ra.fight_offer_as_select(state), cfg, mirror, sims
                )
        if advice_for == "treasure":
            offered = [
                o
                for o in state.get("options") or []
                if isinstance(o, dict) and o.get("enabled", True)
            ]
            if len(offered) == 1:
                # Single player's chest: one relic, taken, not chosen -- in
                # Instinct's game too. Answered here rather than handed to the
                # search, which only ever had the one answer as well. The
                # reply says "choose", like every screen reply: the mod
                # clicks a screen only for "choose" / "skip", and read the
                # option's own "collect" as a fight move -- out of a fight,
                # nothing happened and AI play sat at the chest.
                import realgame_instinct_screens as screens

                advice = screens.lone_option(offered[0])
                advice["note"] = "Instinct (e2e net, no search) · the chest's one relic"
                return advice
            raise InstinctUnavailable("the chest relic is taken, not chosen, in Instinct's game")
        handler = self._screen_handler(advice_for)
        if handler is None:
            raise InstinctUnavailable(f"Instinct does not read {advice_for.replace('_', ' ')} yet")
        return handler(self, state, session)

    # -- the student at one position ------------------------------------

    def evaluate(self, state: dict[str, Any], rm: Any, expect: tuple[str, ...]) -> InstinctPick:
        """Peek ``rm`` with this run's driver state and run the student.

        ``expect``: the decision types this screen can be; anything else means
        the rebuilt run is not where the screen is.
        """
        t0 = time.perf_counter()
        driver = self._driver_for(state, rm)
        try:
            driver.observe_combat_position(rm)
        except Exception as exc:
            logger.debug("instinct: observe failed: %s", exc)
        probe = driver.copy()
        peek = probe.peek(rm)
        if peek.kind == "terminal":
            term = peek.terminal or {}
            raise InstinctUnavailable(
                f"Instinct cannot read this position ({term.get('error') or term.get('outcome')})"
            )
        if peek.kind == "forced":
            # One move (a lone legal action, or one merged class): the env
            # takes it without asking; Instinct answers it with certainty.
            return forced_pick(peek.forced or {}, (time.perf_counter() - t0) * 1000.0)
        decision = peek.decision
        if expect and decision.get("decision_type") not in expect:
            raise InstinctUnavailable(
                f"the rebuilt run stands at {decision.get('decision_type')}, not this screen"
            )
        try:
            out = self._student.decide(probe, peek, rm)
        except Exception as exc:
            raise InstinctUnavailable(f"Instinct could not evaluate this position ({exc})") from exc
        return InstinctPick(
            decision=decision,
            probs=[float(p) for p in out["probs"]],
            best=int(out["class"]),
            ms=(time.perf_counter() - t0) * 1000.0,
        )

    # -- fights -------------------------------------------------------

    def _advise_combat(self, state: dict[str, Any], cfg: Any, mirror: Any) -> dict[str, Any]:
        rm, _notes = graft_run(state)
        cs, wire = combat_state_for(state, cfg, mirror)
        self.last_room = _room_kind(state)
        rm.debug_enter_combat(cs, self.last_room)
        pick = self.evaluate(state, rm, ("COMBAT_TURN", "COMBAT_CHOICE"))
        advice = combat_advice(pick, state, wire)
        try:
            plan = self._combat_plan(state, rm, wire, pick, advice.get("best"))
        except Exception as exc:  # a drawing must never cost the advice
            logger.warning("instinct: turn plan failed: %s", exc)
            plan = []
        if plan:
            advice["plan"] = plan
        return advice

    def _combat_plan(
        self,
        state: dict[str, Any],
        rm: Any,
        wire: dict[str, Any],
        pick: InstinctPick,
        best: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """``advice.plan``: the student's greedy rollout through this turn.

        Played on a clone of the grafted run with a copy of this run's driver,
        the student deciding every later move exactly as it would live; mapped
        onto the current hand and cut by :class:`realgame_turn_plan.PlanWalk`'s
        rules (a draw, a choice, an unmappable move end it). Its first step
        is ``best``; when ``best`` is not the student's own first move (the
        mod's veto took it), the plan is ``best`` alone.
        """
        from realgame_turn_plan import MAX_PLAN_STEPS, PlanWalk, first_step, same_move

        if not isinstance(best, dict):
            return []
        lone = first_step(best, state)
        if pick.decision.get("decision_type") not in ("COMBAT_TURN", "FORCED"):
            return [lone] if lone else []
        walk = PlanWalk(wire, state)
        sim = rm.clone()
        probe = self._driver_for(state, rm).copy()
        for ply in range(MAX_PLAN_STEPS):
            peek = probe.peek(sim)
            if peek.kind == "forced":
                forced = peek.forced or {}
                if forced.get("kind") != "combat":
                    break
                run_action, kind = forced["action"], forced["kind"]
            elif peek.kind == "decision":
                if peek.decision.get("decision_type") != "COMBAT_TURN":
                    break
                ci = pick.best if ply == 0 else int(self._student.decide(probe, peek, sim)["class"])
                ca = peek.class_action(ci)
                run_action, kind = ca["action"], ca["kind"]
            else:
                break
            action = _combat_form(run_action)
            step = walk.to_step(action)
            if step is None or (ply == 0 and not same_move(step, best)):
                break
            walk.push(step)
            if walk.done:
                break
            cs_before = sim.combat_state()
            before = cs_before.get_state() if cs_before is not None else {}
            try:
                probe.apply(sim, run_action, kind)
            except Exception as exc:
                logger.debug("instinct: plan stopped at ply %d: %s", ply, exc)
                walk.steps.pop()
                break
            cs_after = sim.combat_state()
            if cs_after is None:
                break  # the fight ended on this move
            result = "pending_choice" if cs_after.pending_choice() is not None else None
            walk.advance(action, before, cs_after.get_state(), result)
            if walk.done:
                break
        if not walk.steps:
            return [lone] if lone else []
        return walk.steps

    def _advise_combat_select(
        self, state: dict[str, Any], cfg: Any, mirror: Any, sims: int | None
    ) -> dict[str, Any]:
        """An in-combat ``card_select`` (Burning Pact, a Skill Potion's offer).

        The fight mirror re-parks a fork of the carried fight on the world's
        choice -- the replay and injection the search uses
        (``AdvisorMirror.answer_card_select``) -- and Instinct picks on it, one
        pick at a time, as the engine decomposes a multi-pick.
        """
        if mirror is None or cfg is None:
            raise InstinctUnavailable("no fight mirror to stand on this choice")
        room = _room_kind(state.get("combat") or {})
        first: dict[str, Any] = {}

        def picker(fork: Any, *, n_options: int, min_select: int, max_select: int) -> list[int]:
            rm, _notes = graft_run(state)
            rm.debug_enter_combat(fork.state.clone(), room)
            probe = self._driver_for(state, rm).copy()
            picks: list[int] = []
            for _ in range(max(1, max_select) + 1):
                peek = probe.peek(rm)
                if peek.kind == "forced":
                    forced = peek.forced or {}
                    option = (forced.get("action") or {}).get("option")
                    if isinstance(option, int) and len(picks) < max_select:
                        picks.append(option)
                    probe.apply(rm, forced["action"], forced["kind"], strict=False)
                    continue
                if peek.kind != "decision":
                    break
                dec = peek.decision
                if dec.get("decision_type") != "COMBAT_CHOICE":
                    break
                out = self._student.decide(probe, peek, rm)
                ci = int(out["class"])
                if not first:
                    first.update(decision=dec, probs=[float(p) for p in out["probs"]])
                selection = dec["classes"][ci]["actions"][0].get("selection") or []
                if not selection:
                    break  # done / skip
                picks.extend(int(i) for i in selection)
                ca = peek.class_action(ci)
                probe.apply(rm, ca["action"], ca["kind"])
                if len(picks) >= max_select:
                    break
            return picks[:max_select] if max_select > 0 else picks

        advice = mirror.answer_card_select(
            state, cfg, sims, picker=picker, picker_note="Instinct (e2e net, no search) · choice"
        )
        if advice is None:
            raise InstinctUnavailable("the fight mirror cannot stand on this choice")
        if first:
            _annotate_choice_probs(advice, first)
        return advice

    # -- the map ------------------------------------------------------

    def _advise_map(self, state: dict[str, Any], session: dict[str, Any] | None) -> dict[str, Any]:
        import realgame_ppo

        state = realgame_ppo.apply_session_context(state, session)
        nodes = [n for n in state.get("nodes") or [] if isinstance(n, dict)]
        try:
            rm, notes = graft_run(state)
        except InstinctUnavailable:
            if len(nodes) != 1:
                raise
            rm, notes = None, []  # a corridor still has its one answer, just no route
        if COOP_MAP_APPROX in notes:
            # Co-op runs are rebuilt on co-op sized maps (graft_run); this
            # one still differs from the screen, so its paths are not real.
            if len(nodes) != 1:
                raise InstinctUnavailable("the rebuilt co-op map differs from the screen's")
            rm = None
        if len(nodes) == 1:
            # One way on (an act's entrance, a corridor): no decision, but
            # still the route from it. These are most of an act's map
            # screens, and without coordinates and a path they drew no star
            # and no line (live, 2026-11-08).
            import realgame_instinct_screens as screens

            node = nodes[0]
            advice = screens.lone_option({**node, "label": _node_label(node)})
            for o in advice["options"]:
                o["col"], o["row"] = int(node["col"]), int(node["row"])
        else:
            pick = self.evaluate(state, rm, ("MAP",))
            advice = map_advice(pick, state)
        best = next((o for o in advice["options"] if o.get("is_best")), None)
        if best is not None and rm is not None:
            try:
                path = self._map_route(state, rm, (int(best["col"]), int(best["row"])))
            except Exception as exc:  # a drawing must never cost the advice
                logger.warning("instinct: route projection failed: %s", exc)
                path = []
            if path:
                import realgame_advisor as ra

                advice["path"] = path
                advice["note"] += ra._route_note(path)
        return advice

    def _map_route(
        self, state: dict[str, Any], rm: Any, chosen: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """``advice.path`` for Instinct: the student's own MAP choice, hop by hop.

        The twin of the search's ``realgame_advisor._project_map_path``, same
        schema and HP projection: from the chosen node, follow single-child
        chains, and at every fork park a clone of the grafted run ON the hop's
        node (``debug_place_on_map``, the projected HP set) and ask the student
        for its MAP decision there. Deck, gold, relics and potions stay frozen,
        as in the search's projection; the act's last row ends the walk.
        """
        import realgame_advisor as ra
        from floors import engine_total_floor

        m = state.get("map") or {}
        by_coord = ra.route_graph(state)  # the act's start and boss included
        if not by_coord:
            return []
        table = ra._projection_hp_table()
        run_state = state.get("run_state") or {}
        hp = float(run_state.get("hp") or 0)
        max_hp = float(run_state.get("max_hp") or 0)
        gold = int(run_state.get("gold") or 0)
        act = int(state.get("act") or 1)
        base_total = engine_total_floor(state.get("floor"), act)
        visited = [
            (int(v["col"]), int(v["row"]))
            for v in m.get("visited") or []
            if isinstance(v, dict)
            and isinstance(v.get("col"), int)
            and isinstance(v.get("row"), int)
        ]
        path: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        cur = chosen
        for hop in range(ra.MAX_PROJECTED_HOPS):
            node = by_coord.get(cur)
            if node is None or cur in seen:
                break
            seen.add(cur)
            room = node.get("type", "?")
            step: dict[str, Any] = {"col": cur[0], "row": cur[1], "type": room, "hop": hop}
            if table:
                step["hp"] = round(hp)
            path.append(step)
            hp = ra._project_hp(hp, max_hp, room, act, table)
            visited.append(cur)
            children = []
            for c in node.get("children") or []:
                try:
                    coord = (int(c["col"]), int(c["row"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if coord in by_coord:
                    children.append(coord)
            if not children:
                break  # the act's last row
            if len(children) == 1:
                cur = children[0]
                continue
            hop_rm = rm.clone()
            try:
                hop_rm.debug_place_on_map(
                    act, cur[0], cur[1], list(visited), cur[1] + 1, base_total + hop + 1
                )
            except ValueError as exc:
                logger.debug("instinct: route stops at hop %d (%s)", hop, exc)
                break
            hop_rm.set_player(hp=max(1, round(hp)), max_hp=round(max_hp), gold=gold)
            try:
                pick = self.evaluate(state, hop_rm, ("MAP",))
            except InstinctUnavailable as exc:
                logger.debug("instinct: route stops at hop %d (%s)", hop, exc)
                break
            action = (pick.decision["classes"][pick.best].get("actions") or [{}])[0] or {}
            nxt = next((c for c in children if c[0] == action.get("col")), None)
            if nxt is None:
                break
            cur = nxt
        return path

    # -- screens ------------------------------------------------------

    def _screen_handler(self, advice_for: str) -> Any:
        """The handler for an out-of-combat screen, or ``None`` (not read yet)."""
        import realgame_instinct_screens

        return realgame_instinct_screens.HANDLERS.get(advice_for)


def _annotate_choice_probs(advice: dict[str, Any], first: dict[str, Any]) -> None:
    """Put the first pick's class probabilities on the choice's option rows.

    A class's members are interchangeable options (two identical cards); each
    option row carries its class's probability.
    """
    dec, probs = first["decision"], first["probs"]
    by_option: dict[int, float] = {}
    for cls in dec.get("classes") or []:
        ci = int(cls["class"])
        p = probs[ci] if ci < len(probs) else 0.0
        for a in cls.get("actions") or []:
            for i in a.get("selection") or []:
                by_option.setdefault(int(i), p)
    for pos, opt in enumerate(advice.get("options") or []):
        if pos in by_option:
            opt["visit_share"] = round(by_option[pos], 4)
            opt["prob"] = round(by_option[pos], 4)
