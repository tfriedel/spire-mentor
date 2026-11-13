"""``RustGame`` — a :class:`~workbench.backends.GameBackend` over ``sts2_rs``.

Wraps the Rust port's :class:`sts2_rs.RunManager` (method-based: ``phase()`` +
``available_actions()`` + ``take_action()`` + ``combat_state()`` + ``full_map()``)
behind the C# ``sts2_client.Game`` surface, translating in both directions:

* **actions** — the C# verbs (``select_map_node``, ``play_card`` …) become the
  Rust action dicts (``{"action": "move", "col": …}`` …) that ``take_action``
  accepts;
* **state** — :meth:`_serialize` dispatches on ``phase()`` and rebuilds the C#
  state-dict shape (``decision``/``player``/``enemies``/``hand``/``options`` …)
  the workbench's server/render/diff layers already consume.

Card / relic / power names are emitted as localization keys or display strings
that ``server._enrich`` resolves, exactly as the C# backend's leaked keys are.

MVP scope: Ironclad full run loop. Combat potion use (``use_potion`` →
``drink_potion``) and shop potion purchases are mapped; potion *discard* and
Neow card-bundle picks are not yet (they have no run-action surface), so
calling those verbs returns an error dict rather than raising.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from floors import engine_total_floor, game_floor
from sts2_client import DEFAULT_GAME_VERSION, rs_engine_call
from sts2_hint import display_maps

from workbench import localization
from workbench.backends.base import GameBackend
from workbench.seed import json_exact_seed

_log = logging.getLogger(__name__)

# ``LostWisp`` (event Python class name) -> ``LOST_WISP`` (loc-key stem used by
# ``localization_eng/events.json`` for ``<STEM>.title`` / ``.pages.INITIAL.``).
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _event_loc_stem(python_name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", python_name).upper()


# ``target_convention="csharp"`` files loaded to resolve event/boss-relic/
# relic-reward loc keys to plain display text (see ``_csharp_resolve_event_fields``).
# ``ancients.json`` (Neow + the per-act boss-relic "Ancient" node titles, e.g.
# ``NEOW.title``/``DARV.title``) is NOT merged by ``workbench/server.py``'s own
# loc bundle -- that bundle only serves the workbench UI's own resolver
# (``server._enrich``), which never sees boss-relic/Neow titles raw in the
# first place because it prettifies any miss. This module needs the real
# values (matching the C# DLL's ``_loc.Bilingual("ancients", ...)`` calls),
# not a prettified guess.
_CSHARP_LOC_FILES = ("ancients.json", "events.json", "relics.json", "potions.json")
_csharp_loc_cache: dict[str, str] | None = None


def _csharp_loc() -> dict[str, str]:
    """Merged ``localization_eng/{ancients,events,relics,potions}.json``.

    Cached at module scope (read-only, process-lifetime static data) -- same
    pattern as ``workbench/server.py``'s ``card_loc`` build, but scoped to
    just the tables this module's event/boss-relic/relic-reward resolution
    needs. Later files don't overwrite earlier ones (``setdefault``): a
    handful of event ids collide with the relic they grant (e.g.
    ``LOST_WISP.title``), and — mirroring ``server.py``'s own comment — the
    earlier (more specific) table should win.
    """
    global _csharp_loc_cache
    if _csharp_loc_cache is None:
        merged: dict[str, str] = {}
        repo_root = Path(__file__).resolve().parents[3]
        for fname in _CSHARP_LOC_FILES:
            path = repo_root / "localization_eng" / fname
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for key, val in data.items():
                    merged.setdefault(key, val)
        _csharp_loc_cache = merged
    return _csharp_loc_cache


def _csharp_resolve_event_fields(state: dict[str, Any]) -> None:
    """Resolve ``event_name``/option ``title``/option ``description`` from raw
    loc keys (``"WINGED_BOOTS.title"``) to plain display text
    (``"Winged Boots"``), matching what the real Sts2Headless DLL actually
    emits for these screens.

    The native/workbench-UI path (``target_convention="native"``) leaves
    these as loc keys deliberately — ``workbench/server.py``'s ``_enrich``
    resolves them client-side. But the C#-dialect emulation
    (``target_convention="csharp"``) feeds this state DIRECTLY into the
    deployed PPO dispatch (``headless_eval.to_bridge_event`` ->
    ``realgame_ppo.pick_noncombat_action_ppo``), which has no such
    enrichment step — it is exactly the code path the real DLL's ALREADY
    RESOLVED ``event_name``/option ``title`` (``RunSimulator.cs``'s
    ``_loc.Bilingual``/``_loc.Relic`` calls) feed. Leaving keys raw here
    corrupts the net's input (meta_differ_report.md BUG #1/#2): an
    unresolved ``"NEOW.title"`` breaks ``_event_option_label``'s Neow gate
    (`` != "NEOW"`` after upper-casing never matches), and unresolved relic
    titles for boss-relic / free-relic offers give the net no relic
    identity to condition on at all.
    """
    loc = _csharp_loc()
    state["event_name"] = localization.resolve_name(state.get("event_name"), loc)
    for opt in state.get("options") or []:
        title = opt.get("title")
        if title:
            opt["title"] = localization.resolve_name(title, loc)
        desc = opt.get("description")
        if desc:
            resolved = localization.resolve(desc, loc)
            opt["description"] = (
                "" if localization.looks_like_unresolved_key(resolved) else resolved
            )


# C# `decision` ← Rust `phase()`.
_PHASE_TO_DECISION = {
    "map_choice": "map_select",
    "combat": "combat_play",
    "card_reward": "card_reward",
    "shop": "shop",
    "rest_site": "rest_site",
    "event": "event_choice",
    "treasure": "event_choice",
    "boss_relic": "event_choice",
    "run_over": "game_over",
}


def _err(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


# Display fallback for a rest-site option whose engine ``option_id`` the
# ``rest_site_options()`` accessor could not supply (an older wheel). The base
# order is always [Rest, Smith] (run_manager.rs), but the list is DYNAMIC past
# index 1 — Shovel appends Dig, Girya Lift, PumpkinCandle Kindle, MeatCleaver
# Cook, a ByrdonisEgg Hatch — so an index-keyed table can only name the base
# two. Relying on it shipped a bare "Option 2" plaque for Kindle
# (bug report ``wb-500a91f0d29256f6``); the ids are the real fix.
_REST_LABELS = {0: "Rest", 1: "Smith"}

# Boss encounters the map should label by their common encounter name rather
# than by the first monster in the fight. The Rust serializer picks the boss
# node's ``kind`` from the encounter's lead monster, which for the Queen fight
# is the Torch Head Amalgam it spawns alongside the Queen (QueenBoss.cs slots
# ["amalgam", "queen"]); players know that node as "Queen". Keyed by monster
# ``kind`` → encounter display name. Only the map node is renamed — both
# monsters keep their own names as combatants. The Kin's lead is a follower
# (the roster mirrors C#'s spawn order, priest last); its node is "The Kin".
_BOSS_KIND_ENCOUNTER_NAME = {"TorchHeadAmalgam": "Queen", "KinFollower": "The Kin"}


def _boss_encounter_title(encounter_id: str) -> str:
    """Display title for an ``UPPER_SNAKE`` boss encounter id.

    ``QUEEN_BOSS`` -> ``Queen``, ``LAGAVULIN_MATRIARCH_BOSS`` -> ``Lagavulin
    Matriarch``, ``THE_KIN_BOSS`` -> ``The Kin``. The A10 second boss is known
    to the engine only by this id (it has no pinned lead-monster ``kind`` the
    way the act boss does), and the C# side labels it the same way — the
    encounter's own name, which is exactly the id minus its ``_BOSS`` suffix.
    """
    stem = encounter_id.removesuffix("_BOSS")
    return " ".join(word.capitalize() for word in stem.split("_") if word)


# Act display names, by 1-based ``act_index``. Act 1 has TWO models and the
# name has to say WHICH (GH #544): `Overgrowth` (default) and `Underdocks`.
# Acts 2 and 3 are alone at their index (`act_selection.rs`).
#
# These are the strings the C# oracle reports through `act_name`
# (`RunSimulator.cs:3318`), and — because the frontend keys every painted
# region off `act_name` (combat backdrop, card-reward scene, rest-site
# clearing, map parchment) — they are also the asset-directory names,
# case-insensitively. Keep them spelled like the act models.
_ACT_NAMES = {2: "Hive", 3: "Glory"}


def _act_name(act_index: int, act1: str) -> str | None:
    """Display name of the act ``act_index`` (1-based), or ``None`` if unknown.

    ``act1`` is the resolved Act-1 model (``rm.act1``), which decides the name
    only at index 1. An unknown index returns ``None`` rather than a guess: the
    frontend falls back to ``"Act N"`` for the label and to the act-1 region for
    the art, and a wrong region is louder than a missing one.
    """
    if act_index == 1:
        return "Underdocks" if act1 == "underdocks" else "Overgrowth"
    return _ACT_NAMES.get(act_index)


def _map_disagreements(rm: Any, snapshot: Any) -> list[str]:
    """Where the engine's generated act map differs from the pasted one.

    The map is a pure function of ``(seed, act)``, so on a correct import these
    agree node for node. A disagreement means the engine and the live build
    generate different maps for this seed — worth saying out loud, because
    every path the user then explores is a path their real run does not have.

    Best-effort and non-fatal: a shape this cannot read produces no note rather
    than a false alarm.
    """
    world_nodes = (snapshot.map or {}).get("nodes")
    if not isinstance(world_nodes, list) or not world_nodes:
        return []
    try:
        full = rm.full_map()
    except Exception:  # pragma: no cover — defensive around the engine surface
        return []
    if not isinstance(full, dict):
        return []
    ours: dict[tuple[int, int], str] = {}
    for row in full.get("rows") or []:
        for node in row or []:
            if isinstance(node, dict):
                ours[node.get("col"), node.get("row")] = str(node.get("type") or "")
    boss = full.get("boss")
    if isinstance(boss, dict):
        ours[boss.get("col"), boss.get("row")] = str(boss.get("type") or "Boss")

    mismatched = 0
    missing = 0
    for node in world_nodes:
        if not isinstance(node, dict):
            continue
        key = (node.get("col"), node.get("row"))
        theirs = str(node.get("type") or "")
        if key not in ours:
            missing += 1
        elif ours[key] != theirs:
            mismatched += 1
    if not mismatched and not missing:
        return []
    return [
        f"the act map this engine generates differs from the live one "
        f"({mismatched} node(s) of a different type, {missing} it does not "
        f"have) — paths explored from here are not the ones in your run"
    ]


class RustGame(GameBackend):
    """Drives an ``sts2_rs.RunManager`` through the :class:`GameBackend` interface."""

    def __init__(
        self, *, target_convention: str = "native", game_version: str | None = None
    ) -> None:
        self.rm: Any = None
        # The game build every run this backend constructs simulates (GH #801):
        # ``"v0.107.1"`` / ``"v0.111.0"``. ``None`` defers to the process-wide
        # selection (``sts2_client.GAME_VERSION``) — the parity harnesses rely
        # on that; the workbench always passes its session's build explicitly.
        self.game_version = game_version
        self.character = ""
        # The Act-1 model this run was built with (GH #544). Recorded so a
        # save can be restored under the act it was played on, and so the
        # UI can show which of the two acts a `roll` landed on.
        self.act1 = "overgrowth"
        # How an incoming ``target_index`` is interpreted (see
        # ``_full_enemy_index``):
        #   "native"  — already the FULL encounter-array slot (corpses
        #               included). This is the rust engine's own convention and
        #               what the workbench frontend echoes back, since
        #               ``build_combat_view`` labels each enemy with its
        #               full-array ``index``. Passed straight through.
        #   "csharp"  — over the ALIVE subset (the C# headless convention).
        #               Only the cross-engine parity replay drives the rust
        #               backend off C#-oracle choices, so only it opts in. The
        #               index is remapped alive→full before the engine sees it.
        if target_convention not in ("native", "csharp"):
            raise ValueError(
                f"target_convention must be 'native' or 'csharp', got {target_convention!r}"
            )
        self._target_convention = target_convention
        # Whether to record the engine card play-log (workbench event log).
        # Remembered here so ``set_record_card_plays`` works even when called
        # before ``start`` creates the RunManager — ``start`` re-applies it once
        # ``self.rm`` exists.
        self._record_card_plays = False
        # Same for the full game-style combat log (the log panel). Separate
        # flag: a caller can want the event-log labels without paying for every
        # draw and discard.
        self._record_combat_log = False
        # Rust action dicts backing the option indices of the *current* state,
        # for phases rendered via the generic per-action option fallback
        # (treasure / boss relic / combat-relic reward). ``choose_option`` and
        # ``proceed`` index into this when the phase has no nicer mapping.
        self._option_actions: list[dict[str, Any]] = []
        # True while the shop's card-removal picker is open: ``remove_card``
        # presented a synthetic card_select over the engine's own
        # ``shop_remove_card`` offers (the engine has no pending-select state
        # for the shop service), and the next ``select_cards``/``skip_select``
        # resolves or cancels it.
        self._shop_removal_pick = False
        # Gold / relics / room as they stood BEFORE the action currently being
        # applied, so ``_serialize`` can tell what a just-entered treasure room
        # paid out (see ``_treasure_reveal``). ``None`` until the first action.
        self._pre_action: dict[str, Any] | None = None
        # What a live-state import could NOT reproduce (see ``start_imported``).
        # Empty on every other start path, so a caller can read it unguarded.
        self._import_notes: list[str] = []

    def import_notes(self) -> list[str]:
        """What the last :meth:`start_imported` could not reproduce."""
        return list(self._import_notes)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(
        self,
        *,
        character: str,
        seed: str,
        ascension: int,
        act1: str = "overgrowth",
        interactive_potion_swap: bool = False,
    ) -> dict[str, Any]:
        import sts2_rs

        self.character = character
        self.act1 = act1
        self.rm = rs_engine_call(
            sts2_rs.RunManager,
            version=self.game_version,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            # GH #981: agent mode — every deck mutation the real game asks
            # about (on-obtain removal: Precise Scissors; upgrade: Pomander;
            # transform: New Leaf, which also gates enchant) is a player
            # card_select matching C#, and the chosen Neow relic is granted
            # directly with no "Take relic / Skip" step. One definition:
            # `RunState::set_agent_decisions`, shared with the arena, the PPO
            # env and the browser workbench.
            mode="agent",
            # GH #544: STS2 ships TWO acts at index 0 and the real lobby rolls
            # between them off the `act_selection` stream. `"roll"` reproduces
            # that pick, which is what makes a workbench run match the player's
            # game for a given seed. It is uniform 50/50 once Underdocks is
            # unlocked and discovered (`ActModel.GetRandomList`, ActModel.cs:558
            # — `rng.NextItem` over the unlocked candidates), so leaving this at
            # `"overgrowth"` silently shows the WRONG ACT for half of all seeds.
            act1=act1,
            # GH #1069: full-belt potion rewards surface a `potion_swap`
            # decision instead of being dropped (the oracle's matching flag).
            interactive_potion_swap=interactive_potion_swap,
        )
        # Apply the remembered play-log recording preference now that the
        # RunManager exists (set_record_card_plays no-ops while rm is None).
        if self._record_card_plays and hasattr(self.rm, "set_record_card_plays"):
            self.rm.set_record_card_plays(True)
        if self._record_combat_log and hasattr(self.rm, "set_record_combat_log"):
            self.rm.set_record_combat_log(True)
        return self._serialize()

    def force_event(self, name: str) -> dict[str, Any]:
        """Bind a SPECIFIC named event onto the current room and serialize it.

        Mirrors the raw ``RunManager.force_event`` used by ``event_parity`` for
        the option-structure sweep, but returns a full C#-shape decision view
        (with ``player`` + ``options``) so the event-EFFECT parity harness can
        resolve an option and diff the resulting state. ``name`` is the event's
        Python class name (e.g. ``"LostWisp"``). Returns an ``_err`` dict if no
        run is live or the name is unknown.
        """
        if self.rm is None:
            return _err("no live run")
        if self.rm.force_event(name) is None:
            return _err(f"unknown event {name!r}")
        return self._serialize()

    def start_faithful(self, *, character: str, seed: int, ascension: int = 0) -> dict[str, Any]:
        """Start a run constructed IDENTICALLY to the PPO training env.

        This is the *third* mirror of the one parity-pinned recipe — the other
        two being ``ppo.env.PpoRunEnv.__post_init__`` and the run-AI arena's
        ``new_arena_run_manager`` (``sts2-runai/src/arena.rs``). It exists so
        the workbench's PPO option serves the deployed net over runs that are
        in-distribution *by construction*;
        ``tests/test_ppo_workbench_integration.py::test_faithful_construction_obs_parity``
        fails the moment the three drift apart.

        Integer seed + ``neow_start`` (matching the arena's ``with_neow_start``
        — a run that skips Neow forfeits the opening blessing AND shifts the
        act-1 event cursor). GH #399 turned the four deck-mutation flags on in
        training, so they are on here too; leaving them off would have served
        the net removal / upgrade / transform / enchant floors it never saw
        during training. ``interactive_potion_swap`` stays off on all three
        (no encoder surface — GH #379). Powers
        :meth:`BranchingLiveSession.start_faithful`.
        """
        import sts2_rs

        self.character = character
        # NOT act1-parameterised, on purpose. This constructor's contract is to
        # mirror `PpoRunEnv.__post_init__` statement for statement, and that env
        # does not pass `act1` — it trains on Overgrowth. Serving the deployed
        # net an Underdocks run here would put it out of distribution, which is
        # the exact failure this mirror exists to prevent. When training moves
        # to the rolled mixture, this and `ppo/env.py` change together.
        self.act1 = "overgrowth"
        self.rm = rs_engine_call(
            sts2_rs.RunManager,
            version=self.game_version,
            seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            mode="agent",  # GH #981: see `start` above
        )
        if self._record_card_plays and hasattr(self.rm, "set_record_card_plays"):
            self.rm.set_record_card_plays(True)
        if self._record_combat_log and hasattr(self.rm, "set_record_combat_log"):
            self.rm.set_record_combat_log(True)
        return self._serialize()

    def start_imported(self, snapshot: Any) -> dict[str, Any]:
        """Rebuild the run behind a state copied out of the live game.

        The run identity (seed, character, ascension) is REAL — this is the
        same run, constructed the same way — so the act maps, the boss and the
        pre-rolled encounter queue are the live game's, not a look-alike. What
        the engine cannot derive from a seed is then grafted on from the wire:
        position, deck, relics, potions, gold, HP, and the RNG cursors.

        Order is load-bearing:

        1. **Place on the map** before anything else, so the run is in
           ``map_choice`` at the node the player was standing on.
        2. **Empty the deck, THEN grant relics.** ``force_add_relic`` fires
           ``after_obtained`` eagerly, and several relics upgrade cards when
           they land (Whetstone, War Paint). Granting them over a starter deck
           would upgrade cards the imported deck never had — the same trap
           ``repro_wb27d55ac0.py`` documents.
        3. **Deck, then upgrades**, so a multiset like "5 Strikes, 2 of them
           upgraded" lands as exactly that.
        4. **HP/gold last**: the world's numbers are authoritative, and a
           relic grant can move ``max_hp`` on its way in.
        5. **RNG cursors last of all**, so nothing above can advance a stream
           after it has been positioned.

        ``act1`` is not guessed: the run is built with ``act1="roll"``, which
        reproduces the game's own seed-driven pick between Overgrowth and
        Underdocks (GH #544), and the resolved value is read back off the run.

        Args:
            snapshot: A :class:`workbench.state_import.RunSnapshot`.

        Returns:
            The opening state dict, as every other ``start*`` returns.

        Raises:
            ValueError: When the engine rejects the placement (a coord that is
                not on this seed's map — i.e. the paste and the engine
                disagree about the run) or the character has no port.
        """
        import sts2_rs

        self.character = snapshot.character
        self.rm = rs_engine_call(
            sts2_rs.RunManager,
            version=self.game_version,
            string_seed=snapshot.seed,
            character=snapshot.character,
            ascension_level=snapshot.ascension,
            # Not a fresh run: the blessing was taken floors ago and the deck
            # arriving below already reflects it.
            neow_start=False,
            mode="agent",  # GH #981: see `start` above
            # The game rolled this for the player; `roll` makes the same roll.
            act1="roll",
        )
        self.act1 = self.rm.act1
        self._import_notes: list[str] = []

        coord = snapshot.coord
        if coord is not None:
            col, row = coord
            # Raises ValueError when the coord is not on this seed's map, which
            # means the paste and the engine disagree about what run this is —
            # a louder failure than quietly starting somewhere else.
            self.rm.debug_place_on_map(
                snapshot.act,
                col,
                row,
                list(snapshot.visited),
                row + 1,
                # the paste carries the game's floor (see ``_base_state``)
                engine_total_floor(snapshot.floor, snapshot.act),
            )
            self._import_notes.extend(_map_disagreements(self.rm, snapshot))
        else:
            self._import_notes.append(
                "No map position in the paste — the run starts at the act entrance."
            )

        # 2. Deck emptied first so relic on-obtain upgrades have nothing to
        #    chew on; see the docstring.
        for name, _upgraded in list(self.rm.master_deck_detailed()):
            self.rm.force_remove_card(name)
        for relic_id in snapshot.relics:
            self.rm.force_add_relic(relic_id)
        # The live run already made the pickups' choices (Lead Paperweight's
        # choose screen); left queued, they surface as the run advances.
        self.rm.debug_clear_pending_choices()
        for card_id, _upgraded in snapshot.deck:
            self.rm.force_add_card(card_id)
        for card_id, upgraded in snapshot.deck:
            if upgraded and not self.rm.force_upgrade_card(card_id):
                self._import_notes.append(f"could not upgrade {card_id} (no copy left to upgrade)")

        self.rm.debug_set_potion_state(list(snapshot.potions))
        if snapshot.potion_slots and self.rm.max_potion_slots() != snapshot.potion_slots:
            self._import_notes.append(
                f"belt is {self.rm.max_potion_slots()} slots here, "
                f"{snapshot.potion_slots} in the live run"
            )
        self.rm.set_player(hp=snapshot.hp, max_hp=snapshot.max_hp, gold=snapshot.gold)

        if snapshot.rng is not None:
            try:
                ignored = self.rm.debug_set_rng_counters(
                    snapshot.rng["run_seed"], snapshot.rng["counters"]
                )
            except ValueError as exc:
                # A seed mismatch or a backwards cursor: the lineage is not
                # restorable, and pretending otherwise would make every
                # downstream roll a confident lie.
                self._import_notes.append(f"RNG cursors not restored — {exc}")
            else:
                # Combat streams live on a CombatState, not the run; they are
                # restored by the combat import, not here.
                run_level = [n for n in ignored if not n.startswith("combat_")]
                if run_level:
                    self._import_notes.append(
                        "RNG streams this engine does not carry: " + ", ".join(run_level)
                    )

        if self._record_card_plays and hasattr(self.rm, "set_record_card_plays"):
            self.rm.set_record_card_plays(True)
        if self._record_combat_log and hasattr(self.rm, "set_record_combat_log"):
            self.rm.set_record_combat_log(True)
        return self._serialize()

    def close(self) -> None:
        self.rm = None

    def set_record_card_plays(self, on: bool) -> None:
        # Remember the preference so it survives being set before ``start``
        # creates the RunManager (``start`` re-applies it).
        self._record_card_plays = on
        if self.rm is not None and hasattr(self.rm, "set_record_card_plays"):
            self.rm.set_record_card_plays(on)

    def card_play_log(self) -> list[dict[str, Any]]:
        if self.rm is not None and hasattr(self.rm, "card_play_log"):
            return list(self.rm.card_play_log())
        return []

    def set_record_combat_log(self, on: bool) -> None:
        # Remembered like the play-log flag above, for the same reason: the
        # session opts in before ``start`` has built the RunManager.
        self._record_combat_log = on
        if self.rm is not None and hasattr(self.rm, "set_record_combat_log"):
            self.rm.set_record_combat_log(on)

    def combat_log(self) -> list[dict[str, Any]]:
        """This combat's event rows, oldest first (empty outside combat).

        Raw engine rows, unflattened — callers (`server.py`, `branching.py`)
        run these through `combatlog.format_combat_log`, the single place
        `source` gets flattened to display + id fields.
        """
        if self.rm is not None and hasattr(self.rm, "combat_log"):
            return list(self.rm.combat_log())
        return []

    def set_player(self, **kwargs: Any) -> dict[str, Any]:
        """Debug/coverage override, mirroring the C# ``set_player`` command so
        both engines can be pinned identically in lockstep (see
        ``act23_state_parity.py``). Accepts ``hp`` / ``max_hp`` / ``gold`` and
        ``deck`` / ``relics`` / ``potions``, which replace the whole list and
        fire no on-obtain hook, as in C#."""
        if self.rm is None:
            return {"type": "error", "message": "no run in progress"}
        self.rm.debug_set_loadout(kwargs.get("deck"), kwargs.get("relics"))
        if kwargs.get("potions") is not None:
            self.rm.debug_set_potion_state(list(kwargs["potions"]))
        self.rm.set_player(
            hp=kwargs.get("hp"),
            max_hp=kwargs.get("max_hp"),
            gold=kwargs.get("gold"),
        )
        return self._serialize()

    def get_state(self) -> dict[str, Any]:
        return self._serialize()

    def get_map(self, first_col: int | None = None) -> dict[str, Any]:
        # The full graph (``rows``/``boss``/``current_coord``) already matches
        # the frontend ``FullMap`` schema and — since the start-row navigation
        # fix — is consistent with the offered moves, so surfacing it lets the
        # SVG map picker render and highlight the reachable next nodes (parity
        # with the C# backend), instead of the bare button-list fallback.
        full_map = self.rm.full_map() or {}
        # Resolve the boss display name from its monster ``kind`` (the Rust
        # serializer only sets ``kind``/``id`` for deterministic single-boss
        # acts), matching the C# backend's ``boss.name`` shape.
        boss = full_map.get("boss")
        if isinstance(boss, dict) and boss.get("kind") and not boss.get("name"):
            kind = boss["kind"]
            boss["name"] = _BOSS_KIND_ENCOUNTER_NAME.get(kind) or display_maps.monster_title(kind)
        # A10 DoubleBoss: the second boss node is a DIFFERENT encounter, and
        # the engine hands it over by id only (no lead-monster ``kind``) — name
        # it off the id so the map can label and illustrate it as its own boss
        # instead of repeating the act boss (C# `NBossMapPoint` draws
        # `Act.SecondBossEncounter` on that node).
        second = full_map.get("second_boss")
        if isinstance(second, dict) and second.get("id") and not second.get("name"):
            second["name"] = _boss_encounter_title(second["id"])
        # Optionality-aware path plan overlay (run-AI pathing surface): the
        # planned route to the boss + per-first-column scores, so the SVG map
        # can highlight the plan and the elite-skip options it keeps open.
        # Best-effort — the C#/replay backends omit it and the UI degrades to
        # the plain map.
        plan = self.get_path_plan(first_col)
        if plan is not None:
            full_map["path_plan"] = plan
        return full_map

    def get_path_plan(self, first_col: int | None = None) -> dict[str, Any] | None:
        """Optionality-aware path plan for the current map (run-AI pathing).

        Returns the dict described by ``PyRunManager::path_plan`` — the planned
        route to the boss, the optionality vs. hard-max first-move columns, and
        per-column flexibility stats (paths kept open, elite-skip option) — or
        ``None`` when no map is loaded or the binding predates the method.

        *first_col* seeds the route's first hop with the column the run policy
        that is actually DRIVING would take (see
        :meth:`BranchingSession._driving_first_col`). Without it the painted
        route is this surface's own opinion, which under the PPO brain can start
        at a node the run will never enter — the map then contradicts the policy
        panel beside it. Silently ignored by an older binding that has no such
        parameter, which degrades to the unseeded route rather than no route.
        """
        path_plan = getattr(self.rm, "path_plan", None)
        if path_plan is None:
            return None
        try:
            plan = path_plan(first_col) if first_col is not None else path_plan()
        except TypeError:  # pragma: no cover - binding predates `first_col`
            try:
                plan = path_plan()
            except Exception:
                _log.warning("path_plan failed; overlay omitted", exc_info=True)
                return None
        except Exception:  # pragma: no cover - defensive, keeps the map alive
            _log.warning("path_plan failed; overlay omitted", exc_info=True)
            return None
        return plan if isinstance(plan, dict) else None

    def combat_state(self) -> Any:
        """The native ``sts2_rs`` combat state, or ``None`` outside combat.

        Lets the session route MCTS hints natively (see
        :func:`workbench.reasoning.score_native`) instead of round-tripping
        the C# dict through the display-name adapter.
        """
        if self.rm is None or self.rm.phase() != "combat":
            return None
        return self.rm.combat_state()

    def native_run(self) -> Any | None:
        """The in-process ``RunManager`` (``None`` before :meth:`start`)."""
        return self.rm

    def recommended_action(self) -> tuple[str, dict[str, Any]] | None:
        """The ``(verb, args)`` the out-of-combat heuristic policy would take at
        the current decision, or ``None`` in combat / at a finished run.

        Powers the workbench's "play best (G)" / "play to end (F)" autoplay on
        non-combat screens (card reward, shop, event, map, rest, …): it is the
        same deployed self-play policy (:func:`run_policy.recommend_run_action`),
        mapped to the C# verb the live action endpoint applies. Combat is left to
        the frontend's MCTS hint, so this returns ``None`` there.

        The pick seed is derived from the run seed + floor so it is stable for a
        given decision; the returned action is concrete, so a branch-tree
        seek/replay just re-applies it (no re-rolling needed)."""
        import random

        from run_policy import recommend_run_action

        if self.rm is None:
            return None
        seed = (self.rm.seed & 0x7FFF_FFFF) ^ (self.rm.total_floor() & 0x7FFF_FFFF)
        return recommend_run_action(self.rm, random.Random(seed))

    # ── shop card-removal picker (synthetic card_select) ─────────────────
    def _open_shop_removal_pick(self) -> dict[str, Any]:
        """Present the shop's card-removal service as a ``card_select``.

        Rows come from the engine's own ``shop_remove_card`` offers (absent
        once the service is used or unaffordable), with ``index`` = the
        action's raw ``deck_idx`` so the follow-up ``select_cards`` maps 1:1.
        """
        offers = [a for a in self.rm.available_actions() if a.get("action") == "shop_remove_card"]
        if not offers:
            return _err("card removal is not available here")
        deck = self.rm.master_deck_detailed()
        cards = []
        for a in offers:
            i = int(a.get("deck_idx", -1))
            if not 0 <= i < len(deck):
                continue
            cid, upgraded = deck[i]
            loc = display_maps.loc_id(cid)
            cards.append(
                {
                    "index": i,
                    "id": loc,
                    "name": f"{loc}.title",
                    "description": f"{loc}.description",
                    "upgraded": upgraded,
                    "type": "card",
                }
            )
        self._shop_removal_pick = True
        state = self._serialize()
        state["decision"] = "card_select"
        state["cards"] = cards
        state["min_select"] = 0  # min 0 → the CardSelect UI offers Skip (= cancel)
        state["max_select"] = 1
        state["can_skip"] = True
        state["context"] = "remove"
        return state

    def _resolve_shop_removal_pick(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve or cancel the synthetic shop-removal ``card_select``."""
        self._shop_removal_pick = False
        raw = str(args.get("indices", "")) if action == "select_cards" else ""
        first = raw.split(",")[0].strip()
        if action == "skip_select" or not first:
            return self._serialize()  # cancel → back to the shop
        result = self.rm.take_action({"action": "shop_remove_card", "deck_idx": int(first)})
        if result not in ("advanced", "run_over"):
            return _err(f"remove_card rejected by engine: {result}")
        return self._serialize()

    # ── action translation ────────────────────────────────────────────────
    def act(self, action: str, **args: Any) -> dict[str, Any]:
        if self.rm is None:
            return _err("no live run")
        # What this action is about to change, for the treasure-reveal panel.
        # Taken here (the single funnel every action passes through) rather
        # than per-branch, so it is correct no matter which return fires.
        self._pre_action = self._room_snapshot()
        # Shop card-removal service: ``remove_card`` opens a synthetic
        # card_select over the engine's own ``shop_remove_card`` offers, and
        # the follow-up ``select_cards`` / ``skip_select`` resolves or cancels
        # it (the engine models the service as one-shot actions keyed by raw
        # deck position — there is no pending-select state to lean on).
        if action == "remove_card":
            return self._open_shop_removal_pick()
        if self._shop_removal_pick and action in ("select_cards", "skip_select"):
            return self._resolve_shop_removal_pick(action, args)
        # Interactive deck removal is presented as a single pick-N card_select
        # (see ``_fill_card_removal``), matching C#'s ``FromDeckForRemoval(N)``.
        # Resolve EVERY chosen index in this one ``act`` so the decision
        # granularity matches C# (the engine models it as N sequential
        # ``remove_deck_card`` picks). Remove in descending index order so an
        # earlier removal doesn't shift the master-deck position of a later one.
        if action == "select_cards" and getattr(self.rm, "pending_deck_removals", 0) > 0:
            raw = args.get("indices", "")
            idxs = sorted(
                {int(s) for s in str(raw).split(",") if s.strip() != ""},
                reverse=True,
            )
            if not idxs:
                return _err("select_cards requires an index")
            for i in idxs:
                result = self.rm.take_action({"action": "remove_deck_card", "idx": i})
                if result not in ("advanced", "run_over"):
                    return _err(f"{action} rejected by engine: {result}")
            return self._serialize()
        # Interactive deck transform (New Leaf): likewise a single pick-N
        # card_select. C# applies transforms in ASCENDING original-index
        # order (`CardCmd.Transform`'s `PileIndexSort`), and each resolve
        # removes its original (appending the replacement at the tail), so
        # walk ascending and shift each later index down by the number of
        # earlier removals.
        if action == "select_cards" and getattr(self.rm, "pending_deck_transforms", 0) > 0:
            raw = args.get("indices", "")
            idxs = sorted({int(s) for s in str(raw).split(",") if s.strip() != ""})
            if not idxs:
                return _err("select_cards requires an index")
            for n, i in enumerate(idxs):
                result = self.rm.take_action({"action": "transform_deck_card", "idx": i - n})
                if result not in ("advanced", "run_over"):
                    return _err(f"{action} rejected by engine: {result}")
            return self._serialize()
        # Interactive multi-card deck upgrade (YummyCookie: upgrade 4): C#
        # resolves the whole ``FromDeckForUpgrade`` pick-N in one
        # ``select_cards``. Each index is a position in the engine's upgradable
        # subset, and an upgraded card drops out of it, so apply descending to
        # keep the lower indices valid. A single index keeps the one-pick path
        # below (Smith).
        if action == "select_cards" and getattr(self.rm, "pending_deck_upgrades", 0) > 0:
            raw = args.get("indices", "")
            idxs = sorted({int(s) for s in str(raw).split(",") if s.strip() != ""}, reverse=True)
            if len(idxs) > 1:
                for i in idxs:
                    result = self.rm.take_action({"action": "upgrade_deck_card", "idx": i})
                    if result not in ("advanced", "run_over"):
                        return _err(f"{action} rejected by engine: {result}")
                return self._serialize()
        # Interactive deck enchant (Wood Carvings' Snake → Slither): same
        # pick-N card_select shape. Enchanting mutates the card in place (no
        # removal), but an enchanted card drops OUT of the eligible subset
        # the engine recomputes per resolve — apply descending so earlier
        # (lower) original indices stay valid.
        if action == "select_cards" and getattr(self.rm, "pending_deck_enchants", 0) > 0:
            raw = args.get("indices", "")
            idxs = sorted({int(s) for s in str(raw).split(",") if s.strip() != ""}, reverse=True)
            # An "up to N" screen (Kifuda / Gnarled Hammer, C#
            # `CardSelectorPrefs(prompt, 0, N)`) may be confirmed with fewer
            # cards, or none; the rest of its batch is then dropped.
            optional = bool(getattr(self.rm, "pending_enchant_optional", False))
            if not idxs and not optional:
                return _err("select_cards requires an index")
            for i in idxs:
                result = self.rm.take_action({"action": "enchant_deck_card", "idx": i})
                if result not in ("advanced", "run_over"):
                    return _err(f"{action} rejected by engine: {result}")
            if optional and getattr(self.rm, "pending_enchant_optional", False):
                self.rm.decline_optional_deck_enchants()
            return self._serialize()
        # Skipping an "up to N" enchant screen is confirming it with no card:
        # C# `CardSelectorPrefs(prompt, 0, N)` with `Cancelable = false`
        # (Kifuda.cs:24-26) still accepts an empty confirm, and the oracle
        # offers `skip_select` for it (min_select 0).
        if (
            action == "skip_select"
            and getattr(self.rm, "pending_deck_enchants", 0) > 0
            and getattr(self.rm, "pending_enchant_optional", False)
        ):
            self.rm.decline_optional_deck_enchants()
            return self._serialize()
        # Skipping the optional grant grid (Orobas SeaGlass,
        # FromSimpleGridForRewards MinSelect 0) is confirming it with no pick:
        # the engine's decline is ``grant_deck_card`` at idx == option count,
        # the same confirm the ``select_cards`` branch below appends.
        if (
            action == "skip_select"
            and getattr(self.rm, "pending_card_grants", 0) > 0
            and int(getattr(self.rm, "pending_card_grant_min", 1)) == 0
        ):
            total = len(self.rm.card_grant_options())
            result = self.rm.take_action({"action": "grant_deck_card", "idx": total})
            if result not in ("advanced", "run_over"):
                return _err(f"{action} rejected by engine: {result}")
            return self._serialize()
        # Interactive multi-pick card grant (RoomFullOfCheese Gorge): a single
        # pick-N card_select over the generated options. C# adds the picks to
        # the deck in selection (ascending-index) order; each grant removes its
        # option, so walk ascending and shift each later index down by the
        # number of earlier picks (mirrors the transform fan-out).
        if action == "select_cards" and getattr(self.rm, "pending_card_grants", 0) > 0:
            raw = args.get("indices", "")
            idxs = sorted({int(s) for s in str(raw).split(",") if s.strip() != ""})
            # Optional grid (SeaGlass, MinSelect 0): any 0..=N picks, then a
            # confirm at idx == remaining option count.
            optional = int(getattr(self.rm, "pending_card_grant_min", 1)) == 0
            if not idxs and not optional:
                return _err("select_cards requires an index")
            total = len(self.rm.card_grant_options())
            steps = [i - n for n, i in enumerate(idxs)]
            if optional and len(idxs) < total:
                steps.append(total - len(idxs))
            for idx in steps:
                result = self.rm.take_action({"action": "grant_deck_card", "idx": idx})
                if result not in ("advanced", "run_over"):
                    return _err(f"{action} rejected by engine: {result}")
            return self._serialize()
        # Multi-pick in-combat pending choice (e.g. Necrobinder's Dredge:
        # "choose up to 3 discard cards", min==max==3). When MULTIPLE indices are
        # supplied at once (C#'s exact fixed-count multi-select, e.g. discard
        # ``1,2,4``), route the whole list through the engine's explicit-selection
        # action so it resolves in one step on both engines.
        #
        # A SINGLE index is always a single pick — even for an "up to N" choice
        # (max_select>1, min_select 0): it exhausts/selects ONE card and leaves
        # the choice open, matching ``run_parity``'s own min_select<=1 branch
        # (single-index ``select_cards`` per pick) and the faithful ``RunStepper``
        # (which emits one ``combat_resolve_choice{option}`` per step). Gating the
        # collapse on ``max_select`` instead of the index count confirmed-and-
        # resolved a single pick prematurely, so a workbench faithful run diverged
        # from ``play_run_mcts`` at any such choice. Gate on the index count.
        if action == "select_cards":
            cs = self.rm.combat_state()
            pc = cs.pending_choice() if cs is not None else None
            raw = args.get("indices", "")
            idxs = [int(s) for s in str(raw).split(",") if s.strip() != ""]
            # csharp dialect: the real Sts2Headless resolves the WHOLE pending
            # choice with whatever indices one ``select_cards`` submits — a
            # single index on a "choose up to 2" closes the choice (verified
            # against the recorded DLL flow, 2026-07-18 shadow census seed
            # 1007: ``indices: "1"`` on min 0/max 2 returned straight to
            # combat_play). The native dialect keeps the verified run-parity
            # convention (single index = one pick, choice stays open).
            resolve_whole = pc is not None and (
                len(idxs) > 1
                or (
                    self._target_convention == "csharp"
                    and bool(idxs)
                    and pc.get("max_select", 1) > 1
                )
            )
            if resolve_whole:
                result = self.rm.take_action(
                    {"action": "combat_resolve_choice_selection", "selection": idxs}
                )
                if result not in ("advanced", "run_over"):
                    return _err(f"{action} rejected by engine: {result}")
                return self._serialize()
        try:
            rust_action = self._translate(action, args)
        except _Unsupported as exc:
            return _err(str(exc))
        result = self.rm.take_action(rust_action)
        if result not in ("advanced", "run_over"):
            return _err(f"{action} rejected by engine: {result}")
        return self._serialize()

    def _full_enemy_index(self, target_index: Any) -> Any:
        """Resolve an incoming ``target_index`` to the Rust engine's
        encounter-array index, honoring ``self._target_convention``.

        The Rust engine's ``play_card``/``drink_potion`` index the FULL
        encounter array (dead slots included — see the enemy ``index`` comment
        in ``build_combat_view``).

        * ``"native"`` (workbench / MCTS / standalone rust): the index is
          ALREADY a full-array slot — ``build_combat_view`` labels each board
          enemy with its full-array position and the frontend echoes it back.
          Pass it straight through. Remapping it would double-count: once an
          earlier enemy is a corpse the play would shift one slot later (the
          slime mis-target bug — Strike on Twig Slime (M) hit Leaf Slime (S)).

        * ``"csharp"`` (cross-engine parity replay only): the C# headless
          resolves ``target_index`` over the ALIVE subset
          (``state.Enemies.Where(e => e.IsAlive)`` in both DoPlayCard and
          DoUsePotion). Remap alive→full so the same C#-oracle choice hits the
          same enemy in both engines (run-parity enemies_hp bucket, e.g.
          Ironclad RP7 step 9: C# hits the 2nd alive enemy; without the remap
          Rust hit the corpse's successor instead).
        """
        if target_index is None:
            return None
        if self._target_convention == "native":
            return target_index
        cs = self.rm.combat_state()
        if cs is None:
            return target_index
        import sts2_rs

        encounter = sts2_rs.state_to_dict(cs).get("encounter") or []
        alive = [pos for pos, c in enumerate(encounter) if not c.get("is_dead")]
        idx = int(target_index)
        return alive[idx] if 0 <= idx < len(alive) else target_index

    def _translate(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        """C# verb + args → a Rust ``take_action`` dict (raises _Unsupported)."""
        phase = self.rm.phase()
        if action == "select_map_node":
            return {"action": "move", "col": args["col"]}
        if action == "play_card":
            return {
                "action": "play_card",
                "hand_idx": args["card_index"],
                "target_index": self._full_enemy_index(args.get("target_index")),
            }
        if action == "end_turn":
            return {"action": "end_turn"}
        if action == "use_potion":
            # ``potion_index`` is the belt slot; in combat the rust backend's
            # ``potions()`` reflects the combat-side belt, so the slot lines up
            # 1:1 with ``DrinkPotion``'s ``potion_slot``. ``target_index`` is the
            # enemy index for AnyEnemy potions (self/no-target potions ignore it).
            return {
                "action": "drink_potion",
                "potion_slot": args["potion_index"],
                "target_index": self._full_enemy_index(args.get("target_index")),
            }
        if action == "select_cards":
            # ``indices`` is a single chosen card index. The rust backend emits
            # a card_select for three cases: an interactive deck removal (Precise
            # Scissors etc., preempts the phase), the Smith upgrade sub-state at a
            # rest site, and an in-combat pending choice (Headbutt etc.). Route to
            # the matching action.
            raw = args.get("indices", "")
            first = str(raw).split(",")[0].strip()
            if not first:
                raise _Unsupported("select_cards requires an index")
            idx = int(first)
            if getattr(self.rm, "pending_deck_removals", 0) > 0:
                return {"action": "remove_deck_card", "idx": idx}
            # `idx` is the position in the upgradeable (non-upgraded) subset,
            # matching C#'s FromDeckForUpgrade PendingOptions index. Bead
            # sts2-rs-1hiht.
            if getattr(self.rm, "pending_deck_upgrades", 0) > 0:
                return {"action": "upgrade_deck_card", "idx": idx}
            if getattr(self.rm, "pending_deck_transforms", 0) > 0:
                return {"action": "transform_deck_card", "idx": idx}
            if getattr(self.rm, "pending_deck_enchants", 0) > 0:
                return {"action": "enchant_deck_card", "idx": idx}
            # A choose screen (HeftyTablet's 3 Rares / Lead Paperweight's 2
            # Colorless) renders as a card_select but resolves through the
            # card-reward pick. GH #34.
            if phase == "card_reward" and getattr(self.rm, "card_offer_choose_screen", False):
                return {"action": "pick_card_reward", "idx": idx}
            if phase == "rest_site" and self.rm.smith_options():
                return {"action": "smith_card", "idx": idx}
            if phase == "shop":
                # The shop's card-removal service: a one-shot engine action
                # keyed by raw master-deck position (no pending select, no
                # combat choice possible at a shop). This is the path
                # run-policy recommendations (heuristic/PPO
                # ``resolve_workbench_verb``) dispatch through.
                return {"action": "shop_remove_card", "deck_idx": idx}
            if self._combat_choice_open():
                return {"action": "combat_resolve_choice", "option": idx}
            return {"action": "remove_deck_card", "idx": idx}
        if action == "skip_select":
            # Skippable card_selects: an in-combat pending choice that allows
            # skipping, or a run-level choose screen (canSkip: true).
            if self._combat_choice_open():
                return {"action": "combat_resolve_choice", "option": None}
            if phase == "card_reward" and getattr(self.rm, "card_offer_choose_screen", False):
                return {"action": "pick_card_reward", "idx": None}
            raise _Unsupported("skip_select not valid here")
        if action == "select_card_reward":
            return {"action": "pick_card_reward", "idx": args["card_index"]}
        if action == "skip_card_reward":
            return {"action": "pick_card_reward", "idx": None}
        if action == "potion_swap":
            # GH #1069: `discard_slot` None declines the parked reward potion.
            slot = args.get("discard_slot")
            return {"action": "potion_swap", "discard_slot": None if slot is None else int(slot)}
        if action == "select_bundle":
            # Scroll Boxes: pick one pack. The engine models packs as a
            # no-skip CardReward fan-out, so the pack index IS the
            # ``pick_card_reward`` idx (which adds the whole pack to the deck).
            return {"action": "pick_card_reward", "idx": args["bundle_index"]}
        if action == "buy_card":
            return {"action": "shop_buy_card", "idx": args["card_index"]}
        if action == "buy_relic":
            return {"action": "shop_buy_relic", "idx": args["relic_index"]}
        if action == "buy_potion":
            return {"action": "shop_buy_potion", "idx": args["potion_index"]}
        if action == "choose_option":
            idx = args["option_index"]
            if phase == "event":
                return {"action": "event_choice", "idx": idx}
            if phase == "boss_relic":
                return {"action": "pick_boss_relic", "idx": idx}
            # rest_site (rest/smith + smith card-pick sub-state), treasure, and
            # combat-relic rewards all index into the per-screen action list the
            # matching ``_fill_*`` stored, so option order == action order.
            if 0 <= idx < len(self._option_actions):
                return self._option_actions[idx]
            raise _Unsupported(f"choose_option {idx} not valid in phase {phase}")
        if action == "leave_room":
            if phase == "shop":
                return {"action": "leave_shop"}
            # A MiniatureTent rest site with options still open: leaving is
            # the C# `BeforeLocalRestSiteExited` skip, which the engine offers
            # as `leave_shop` once one option was taken.
            if phase == "rest_site":
                leave = next(
                    (a for a in self.rm.available_actions() if a.get("action") == "leave_shop"),
                    None,
                )
                if leave is not None:
                    return leave
            return self._first_action(phase)
        if action == "proceed":
            return self._first_action(phase)
        raise _Unsupported(f"action {action!r} not supported by the rust backend")

    def _combat_choice_open(self) -> bool:
        """True when an in-combat ``pending_choice`` (Headbutt etc.) is open."""
        cs = self.rm.combat_state()
        return cs is not None and cs.pending_choice() is not None

    def _first_action(self, phase: str) -> dict[str, Any]:
        """The single advancing action for a continue-style screen."""
        acts = self.rm.available_actions()
        if not acts:
            raise _Unsupported(f"no available action in phase {phase}")
        return acts[0]

    # ── state serialization ───────────────────────────────────────────────
    def _room_snapshot(self) -> dict[str, Any]:
        """Gold / owned relics / current room, for the treasure-reveal diff."""
        rm = self.rm
        current_room = getattr(rm, "current_room", None)
        return {
            "gold": rm.gold,
            "relics": list(rm.relics()),
            "room": current_room() if callable(current_room) else None,
        }

    def _serialize(self) -> dict[str, Any]:
        """The C#-shaped state dict, plus any room-reveal panel it carries.

        Thin wrapper over :meth:`_serialize_state` (which has many early
        returns) so the reveal is attached exactly once, whatever decision the
        engine landed on.
        """
        state = self._serialize_state()
        reveal = _treasure_reveal(self._pre_action, self._room_snapshot(), self.rm.act_index)
        if reveal is not None:
            state["room_reveal"] = reveal
        # GH #544: which Act 1 this run walks. The C# oracle reports the act
        # through `act_name` (`RunSimulator.cs:3318`); the Rust backend has no
        # equivalent, so surface it here. It matters most under `act1="roll"`,
        # where the act came from the seed and the caller cannot know it —
        # without this the two acts are indistinguishable in the UI, and a bug
        # report against "act 1" does not say WHICH act 1.
        #
        # RESOLVED, from the run itself: `self.act1` may be `"roll"`, which
        # tells a reader nothing. `rm.act1` reports which of the two the seed
        # actually landed on.
        state["act1"] = self.rm.act1
        # ...and the act's DISPLAY name, the same field the C# oracle reports.
        # This is not cosmetic: the frontend picks every painted region from
        # `act_name` (Board's `arenaRegion`, CardReward's `rewardRegion`,
        # RestSite's `restScene`, MapView's `regionFor`), each falling back to
        # overgrowth when it is missing. The Rust backend never emitted it, so
        # the fallback was the ONLY branch ever taken: acts 2 and 3 were painted
        # in act 1's jungle ruin, and an Underdocks run showed Overgrowth art.
        state["act_name"] = _act_name(self.rm.act_index, self.rm.act1)
        # GH #801: the build this run simulates, so the SPA (and a bug report
        # built off this state) can tell a v0.111.0 run from a default one.
        # Emitted only off the default, which keeps every v0.107.1 state —
        # and the golden bytes recorded from them — exactly as it was. A
        # reader treats the missing key as the default.
        version = getattr(self.rm, "game_version", None)
        if isinstance(version, str) and version != DEFAULT_GAME_VERSION:
            state["game_version"] = version
        # The run's master RNG seed and the map node this room sits on. The game
        # rolls a combat room's layer variants from
        # `Rng(runSeed + row + col * 747)`
        # (`NCombatRoom.GenerateBackgroundRngForCurrentPoint`), so these three
        # numbers are what let a fight show the barrel, netting and window the
        # real one had rather than a different variant of the same room. Both
        # come straight off the run — `run_seed` IS `RunState.Rng.Seed` as a
        # uint, and `map_coord` is the CURRENT room's node.
        #
        # Read through `getattr`: these two accessors are newer than some of the
        # run objects that reach here (the test fakes, an older installed wheel),
        # and a missing node must leave the room on its fallback roll rather than
        # take the whole state serialization down.
        # ``run_seed`` is a PyO3 ``#[getter]`` (an int on the bound object,
        # not a method), so it is read as a value; ``map_coord`` is a method.
        # A ``callable`` check here silently dropped ``bg_seed`` for every
        # live run until the browser-workbench parity replay caught it.
        seed_of = getattr(self.rm, "run_seed", None)
        coord_of = getattr(self.rm, "map_coord", None)
        if callable(seed_of):
            seed_of = seed_of()
        if seed_of is not None:
            state["bg_seed"] = json_exact_seed(int(seed_of))
        coord = coord_of() if callable(coord_of) else None
        if coord is not None:
            state["bg_col"], state["bg_row"] = int(coord[0]), int(coord[1])
        return state

    def _serialize_state(self) -> dict[str, Any]:
        rm = self.rm
        phase = rm.phase()

        # csharp dialect ONLY: the real Sts2Headless DLL auto-collects combat
        # reward relics/gold/potions (RunSimulator.cs `DetectPostCombatState`,
        # `reward.SelectUnsynchronized()` on Gold/Relic/PotionReward) and never
        # presents a take/skip decision for them — only CardReward entries
        # ever reach the player. Run this BEFORE `decision`/preempts are
        # computed so a relic's `AfterObtained` side effect (e.g. a follow-on
        # deck upgrade/removal) is caught by the normal preempt checks below,
        # and so the state dict built further down reflects post-collection
        # reality (gold/relics/phase already advanced).
        if self._target_convention == "csharp" and phase == "card_reward":
            self._csharp_auto_collect_combat_relic_rewards()
            phase = rm.phase()

        decision = _PHASE_TO_DECISION.get(phase, "unknown")
        self._option_actions = []

        # An interactive deck mutation the engine is waiting on preempts the
        # phase: it offers the pick regardless of ``phase()``, so render that
        # pick as a deck ``card_select`` before anything the phase would show.
        for counter, fill in _PREEMPTS:
            if getattr(rm, counter, 0) > 0:
                state = self._base_state("card_select")
                fill(self, state)
                return state
        # GH #1069: a full-belt potion reward parked for a swap preempts the
        # phase next, in the engine's own legal-action order.
        incoming = getattr(rm, "pending_potion_swap", None)
        if incoming:
            state = self._base_state("potion_swap")
            self._fill_potion_swap(state, incoming)
            return state

        state = self._base_state(decision)

        if decision == "combat_play":
            self._fill_combat(state)
        elif phase == "map_choice":
            self._fill_map(state)
        elif phase == "card_reward":
            self._fill_card_reward(state)
        elif phase == "shop":
            self._fill_shop(state)
        elif phase == "rest_site":
            self._fill_rest(state)
        elif phase == "event":
            self._fill_event(state)
        elif decision == "game_over":
            state["victory"] = bool(rm.player_won)
        elif phase == "boss_relic":
            self._fill_boss_relic(state)
        else:
            # treasure / combat-relic reward → generic option list.
            self._fill_generic(state, phase)
        if self._target_convention == "csharp" and state.get("decision") == "event_choice":
            # Matches the real DLL's already-resolved event_name/option-title
            # emissions (see ``_csharp_resolve_event_fields``); native mode
            # (workbench UI) leaves keys raw for ``server._enrich`` to resolve
            # client-side and must NOT be touched here.
            _csharp_resolve_event_fields(state)
        return state

    # ── per-decision builders ─────────────────────────────────────────────
    def _fill_potion_swap(self, state: dict[str, Any], incoming: str) -> None:
        """The oracle's ``potion_swap`` view: option 0 declines, option
        ``i + 1`` discards held belt potion ``i`` (``RunSimulator.PotionSwapState``)."""
        state["potion"] = incoming
        options: list[dict[str, Any]] = [{"index": 0, "discard_slot": None, "label": "decline"}]
        for i, kind in enumerate(self.rm.potions()):
            options.append({"index": i + 1, "discard_slot": i, "label": f"discard {kind}"})
        state["options"] = options

    def _base_state(self, decision: str) -> dict[str, Any]:
        """The run-wide part of every decision view; a ``_fill_*`` adds the rest."""
        rm = self.rm
        return {
            "decision": decision,
            "player": self._player(),
            # The game's floor (`RunState.TotalFloor`) counts each act's start
            # room (Neow, the Ancients); the engine's counter does not.
            "floor": game_floor(rm.total_floor(), rm.act_index),
            "act": rm.act_index,
            "ascension": rm.ascension_level,
        }

    def _player(self) -> dict[str, Any]:
        rm = self.rm
        relics = [_owned_entry(rid, None) for rid in rm.relics()]
        potions = [_owned_entry(pid, None) for pid in rm.potions()]
        deck = [
            self._deck_card(i, cid, upgraded)
            for i, (cid, upgraded) in enumerate(rm.master_deck_detailed())
        ]
        return {
            "name": self.character,
            "hp": rm.current_hp,
            "max_hp": rm.max_hp,
            "block": 0,
            "gold": rm.gold,
            "relics": relics,
            "potions": potions,
            # Belt capacity, matching the real DLL's PlayerSummary
            # (RunSimulator.cs `potion_slots`). Without it the AI bridge's
            # belt-full heuristic sees a beltful after ONE potion and the
            # net can never buy potions (2026-07-16 shop audit).
            "potion_slots": int(rm.max_potion_slots()),
            "deck": deck,
        }

    def _deck_card(self, index: int, raw_id: str, upgraded: bool) -> dict[str, Any]:
        cid = display_maps.loc_id(raw_id)
        return {
            "index": index,
            "id": cid,
            "name": f"{cid}.title",
            "description": f"{cid}.description",
            "upgraded": upgraded,
            "type": "card",
        }

    def _combat_card(
        self, index: int, wire: dict[str, Any], can_play: bool, target_type: str
    ) -> dict[str, Any]:
        return _combat_card_dict(index, wire, can_play, target_type)

    def _fill_combat(self, state: dict[str, Any]) -> None:
        rm = self.rm
        cs = rm.combat_state()

        # Build the canonical combat view (the single source of truth shared
        # with the MCTS recorder) FIRST, and copy the LIVE combat HP/block/
        # max_hp onto the player up front — this must happen even when a
        # pending choice is open. ``_player()`` seeded the player with the
        # run-manager ``rm.current_hp`` (the combat-ENTRY HP); without this
        # overwrite, an in-combat ``card_select`` (Headbutt / Survivor's
        # discard / …) reports the entry HP instead of the live combat HP,
        # which the whole-run lockstep flags as a phantom divergence that
        # reconverges the moment the choice resolves (run-parity Silent RP2
        # @step35 cs=37 rs=64 → step36 both 37). Bead ``sts2-rs-oac``.
        # ``loc=None`` keeps card names as loc keys (``STRIKE_IRONCLAD.title``)
        # for the server's ``_enrich`` to resolve.
        view = build_combat_view(cs, self.character, legal=rm.available_actions())
        player = state["player"]
        for k in ("hp", "max_hp", "block"):
            player[k] = view["player"][k]

        # A card that opens an in-combat choice (e.g. Headbutt — put a discard
        # card on top of the draw pile; Survivor — discard 1) leaves the engine
        # with a pending choice whose only legal actions are
        # ``combat_resolve_choice``. Surface it as a card_select so the player
        # can resolve it (otherwise the board is a dead end: unplayable hand +
        # an End-turn button that can't fire).
        # An in-combat pending choice (Headbutt, Touch of Insanity, Dredge, …) is
        # still a combat state: carry the live enemies and player powers so the
        # event-log diff across the choice boundary doesn't read every enemy as
        # dying (the card_select otherwise drops the enemy list) and respawning
        # when the choice resolves — nor wobble player powers (phantom Dexterity
        # −/+). Bead: in-combat card_select must stay enemy-/power-bearing.
        state["enemies"] = view.get("enemies", [])
        state["player_powers"] = view.get("player_powers", [])
        # Defect orb belt: carry it (and its slot capacity) like enemies/powers
        # so it stays rendered across an in-combat choice boundary and doesn't
        # flicker. ``build_combat_view`` only emits these keys for a belt with
        # capacity, so this is a no-op for non-Defect characters. Without this
        # copy the whole belt is dropped from the live/replay view (the fixed
        # merge loop below never listed it), so the Defect board showed no orbs.
        if "orbs" in view:
            state["orbs"] = view["orbs"]
            state["orb_slots"] = view.get("orb_slots", 0)
        pc = cs.pending_choice() if cs is not None else None
        if pc:
            fill_combat_choice(state, pc)
            return

        # Merge the remaining combat fields onto the existing state, preserving
        # the player's run-wide relics / potions / deck / gold from ``_player()``.
        if "osty" in view:
            state["osty"] = view["osty"]
        for k in (
            "round",
            "energy",
            "max_energy",
            "hand",
            "enemies",
            "player_powers",
            "draw_pile_count",
            "discard_pile_count",
            "exhaust_pile_count",
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
        ):
            state[k] = view[k]

    def _fill_map(self, state: dict[str, Any]) -> None:
        rm = self.rm
        # Rust moves by ``col`` alone (unique within a row), so ``move_cols`` is
        # the authoritative set of legal moves; ``row``/``type`` are display-only.
        move_cols = [a["col"] for a in rm.available_actions() if a.get("action") == "move"]

        # The offered nodes are the graph children of the player's current node,
        # so derive each choice's row + type from that connectivity rather than
        # from ``coord[1] + 1`` arithmetic: the latter assumes all choices share
        # one row and breaks whenever ``map_coord`` is absent. The engine omits
        # the coord for some rooms (notably events — see sts2-rs map_coord
        # handling), where it then surfaces the *start* node's children; mirror
        # that by falling back to row 0 so the displayed nodes match the moves
        # the engine actually offers, instead of mislabeling them as row 1.
        rows = (rm.full_map() or {}).get("rows") or []
        node_at = {(n["col"], n["row"]): n for row in rows for n in row}
        coord = rm.map_coord()
        current = node_at.get(tuple(coord)) if coord is not None else None
        if current is None:
            current = rows[0][0] if rows and rows[0] else None

        # col -> child node, from the current node's outgoing edges.
        children = (current.get("children") or []) if current is not None else []
        child_by_col = {c["col"]: node_at.get((c["col"], c["row"]), c) for c in children}
        state["choices"] = [
            {
                "col": c,
                "row": child_by_col[c]["row"] if c in child_by_col else None,
                "type": child_by_col[c].get("type") if c in child_by_col else None,
            }
            for c in move_cols
        ]

    def _fill_card_removal(self, state: dict[str, Any]) -> None:
        """Deck card_select for a pending interactive removal.

        Cards are the master deck (index = deck position). C# presents the
        whole removal as a SINGLE pick-N card_select (``CardSelectCmd.
        FromDeckForRemoval(N)`` → ``min_select == max_select == N``), so mirror
        that granularity: ``min_select``/``max_select`` are the full pending
        count, and ``act`` resolves every chosen index in one step (the engine
        still models it as N sequential ``remove_deck_card`` picks internally).
        Mandatory (no skip). Bead ``sts2-rs-1hiht``.
        """
        # C# ``FromDeckForRemoval`` offers REMOVABLE cards with Curse-type
        # sorted to the front; ``index`` is a position in that order (the
        # engine's ``remove_deck_card`` maps it back to the raw master-deck
        # position). Using raw ``master_deck_detailed`` order removed the wrong
        # card whenever a curse sat mid-deck (RP94 Necrobinder).
        cards = []
        for i, (cid, upgraded) in enumerate(self.rm.card_removal_options()):
            loc = display_maps.loc_id(cid)
            cards.append(
                {
                    "index": i,
                    "id": loc,
                    "name": f"{loc}.title",
                    "description": f"{loc}.description",
                    "upgraded": upgraded,
                    "type": "card",
                }
            )
        count = int(getattr(self.rm, "pending_deck_removals", 1) or 1)
        # Can't remove more cards than the deck holds.
        count = max(1, min(count, len(cards)))
        state["cards"] = cards
        state["min_select"] = count
        state["max_select"] = count
        state["can_skip"] = False
        state["context"] = "remove"

    def _fill_card_transform(self, state: dict[str, Any]) -> None:
        """Deck card_select for a pending interactive transform.

        Cards are the engine's eligible subset
        (``transform_select_deck_indices()`` — the whole deck for the
        unfiltered NewLeaf shape, the Basic-rarity subset for Wood Carvings'
        Bird/Torus), with ``index`` = position in that subset, matching the
        C# headless card_select index space (``FromDeckForTransformation`` /
        ``FromDeckGeneric``). Mandatory pick-N
        (``CardSelectorPrefs(TransformSelectionPrompt, N)`` →
        ``min_select == max_select == N``, no skip). The replacement is
        rolled (``Rng.Niche``) or fixed engine-side only after the pick.
        GH #34.
        """
        cards = self._deck_subset_cards(self.rm.transform_select_deck_indices())
        count = int(getattr(self.rm, "pending_deck_transforms", 1) or 1)
        count = max(1, min(count, len(cards)))
        state["cards"] = cards
        state["min_select"] = count
        state["max_select"] = count
        state["can_skip"] = False
        state["context"] = "transform"

    def _fill_card_enchant(self, state: dict[str, Any]) -> None:
        """Deck card_select for a pending interactive enchant (Wood
        Carvings' Snake → Slither). Cards are the engine's CanEnchant
        subset (``enchant_select_deck_indices()``), ``index`` = subset
        position — C# ``FromDeckForEnchantment``'s pool. Mandatory pick-N.
        """
        cards = self._deck_subset_cards(self.rm.enchant_select_deck_indices())
        count = int(getattr(self.rm, "pending_deck_enchants", 1) or 1)
        count = max(1, min(count, len(cards)))
        state["cards"] = cards
        # "Up to N" (Kifuda / Gnarled Hammer) vs exactly N.
        optional = bool(getattr(self.rm, "pending_enchant_optional", False))
        state["min_select"] = 0 if optional else count
        state["max_select"] = count
        state["can_skip"] = False
        state["context"] = "enchant"

    def _fill_card_grant(self, state: dict[str, Any]) -> None:
        """Card_select for a pending multi-pick card grant (RoomFullOfCheese
        Gorge). Cards are the engine's generated option list
        (``card_grant_options()`` → ``(PascalCase id, upgraded)``), ``index`` =
        position in that list (the ``grant_deck_card`` ``idx`` space, matching
        C# ``FromSimpleGridForRewards``). Pick-N (``max ==
        pending_card_grants``); ``min`` is N too, or 0 for an optional grid
        (SeaGlass, ``pending_card_grant_min == 0``)."""
        options: list[tuple[str, bool]] = self.rm.card_grant_options()
        cards = []
        for i, (cid, upgraded) in enumerate(options):
            loc = display_maps.loc_id(cid)
            cards.append(
                {
                    "index": i,
                    "id": loc,
                    "name": f"{loc}.title",
                    "description": f"{loc}.description",
                    "upgraded": upgraded,
                    "type": "card",
                }
            )
        count = int(getattr(self.rm, "pending_card_grants", 1) or 1)
        count = max(1, min(count, len(cards)))
        state["cards"] = cards
        # Optional grid (SeaGlass, MinSelect 0): any 0..=count.
        optional = int(getattr(self.rm, "pending_card_grant_min", 1)) == 0
        state["min_select"] = 0 if optional else count
        state["max_select"] = count
        state["can_skip"] = False
        state["context"] = "grant"

    def _deck_subset_cards(self, deck_indices: list[int]) -> list[dict[str, Any]]:
        """Render the master-deck cards at ``deck_indices`` as card_select
        options whose ``index`` is the SUBSET position (the engine action's
        ``idx`` space), not the raw deck position."""
        deck = self.rm.master_deck_detailed()
        cards = []
        for subset_idx, deck_idx in enumerate(deck_indices):
            cid, upgraded = deck[deck_idx]
            loc = display_maps.loc_id(cid)
            cards.append(
                {
                    "index": subset_idx,
                    "id": loc,
                    "name": f"{loc}.title",
                    "description": f"{loc}.description",
                    "upgraded": upgraded,
                    "type": "card",
                }
            )
        return cards

    def _fill_card_upgrade(self, state: dict[str, Any]) -> None:
        """Deck card_select for a pending interactive upgrade (Pomander).

        Only the UPGRADABLE deck cards are offered (C# ``FromDeckForUpgrade``
        = ``Where(IsUpgradable)``: curses/statuses/quest cards excluded, not
        merely non-upgraded — GUILTY in the list shifted every later index,
        census seed 1010), indexed by their position in that subset
        (``index`` 0..) — matching the engine's ``upgrade_deck_card``
        resolution. Mandatory (``min_select`` 1, no skip) — the engine keeps
        offering ``upgrade_deck_card`` until the queue drains. Bead
        sts2-rs-1hiht.
        """
        cards = []
        subset_idx = 0
        deck = self.rm.master_deck_detailed()
        eligible = set(self.rm.upgrade_select_deck_indices())
        for deck_idx, (cid, _upgraded) in enumerate(deck):
            if deck_idx not in eligible:
                continue
            loc = display_maps.loc_id(cid)
            cards.append(
                {
                    "index": subset_idx,
                    "id": loc,
                    "name": f"{loc}.title",
                    "description": f"{loc}.description",
                    "upgraded": False,
                    "type": "card",
                }
            )
            subset_idx += 1
        state["cards"] = cards
        state["min_select"] = 1
        state["max_select"] = 1
        state["can_skip"] = False
        state["context"] = "upgrade"

    def _csharp_auto_collect_combat_relic_rewards(self) -> None:
        """csharp mode ONLY: auto-take every pending relic in the current
        ``Phase::CardReward`` fan-out, mirroring the real Sts2Headless DLL's
        combat-reward auto-collect (see ``_serialize``'s call site for the
        ground-truth citation). The C# side never offers "skip" for these —
        it unconditionally calls ``SelectUnsynchronized()`` — so this always
        takes, never skips.

        Stops (leaving the decision to surface normally, unmodified) the
        instant the fan-out is anything OTHER than a plain relic pick: real
        card choices, a Scroll Boxes bundle pick, or any other action kind
        (e.g. ``sacrifice_card_reward``) — those are genuine player
        decisions the real DLL does present. Bounded: each iteration
        consumes exactly one relic or exits, so this always terminates.
        Native/workbench-UI mode is untouched by this method — callers only
        invoke it under ``target_convention == "csharp"``.
        """
        rm = self.rm
        while rm.phase() == "card_reward":
            if rm.card_reward_bundles():
                return
            actions = rm.available_actions()
            if any(a.get("action") == "pick_card_reward" for a in actions):
                return
            pick = next((a for a in actions if a.get("action") == "pick_relic_reward"), None)
            if pick is None:
                return
            result = rm.take_action(pick)
            if result not in ("advanced", "run_over"):
                return

    def _fill_card_reward(self, state: dict[str, Any]) -> None:
        rm = self.rm
        # Scroll Boxes: the engine surfaces a bundle pick through the
        # CardReward phase (no-skip fan-out). Render it as the C#-shaped
        # ``bundle_select`` decision (pick 1 of N packs, whole pack added).
        bundles_raw = rm.card_reward_bundles()
        if bundles_raw:
            self._fill_bundle_select(state, bundles_raw)
            return
        actions = rm.available_actions()
        has_cards = any(a.get("action") == "pick_card_reward" for a in actions)
        if not has_cards:
            # Relic / sacrifice reward — no card list; render as options.
            state["decision"] = "event_choice"
            self._fill_relic_reward(state)
            return
        ids = rm.card_reward_option_ids()
        cards = []
        for i, raw in enumerate(ids):
            cid = display_maps.loc_id(raw)
            cards.append(
                {
                    "index": i,
                    "id": cid,
                    "name": f"{cid}.title",
                    "description": f"{cid}.description",
                    "type": "card",
                }
            )
        state["cards"] = cards
        state["can_skip"] = any(
            a.get("action") == "pick_card_reward" and a.get("idx") is None for a in actions
        )
        # A choose screen (HeftyTablet's 3 Rares / Lead Paperweight's 2
        # Colorless — C# ``CardSelectCmd.FromChooseACardScreen(canSkip:
        # true)``) is a ``card_select`` in the C# headless decision
        # vocabulary, not a ``card_reward``; picks still resolve through
        # ``pick_card_reward``. GH #34.
        if getattr(rm, "card_offer_choose_screen", False):
            state["decision"] = "card_select"
            state["min_select"] = 0
            state["max_select"] = 1

    def _fill_bundle_select(self, state: dict[str, Any], bundles_raw: list[list[str]]) -> None:
        """Render a Scroll Boxes bundle pick as the C#-shaped ``bundle_select``
        decision: ``bundles`` is a list of ``{index, cards: [...]}`` packs; the
        player picks one pack (via ``select_bundle``) and the whole pack is
        added to the deck (no skip)."""
        state["decision"] = "bundle_select"
        bundles: list[dict[str, Any]] = []
        for i, pack in enumerate(bundles_raw):
            cards = []
            for raw in pack:
                cid = display_maps.loc_id(raw)
                cards.append(
                    {
                        "id": cid,
                        "name": f"{cid}.title",
                        "description": f"{cid}.description",
                        "type": "card",
                    }
                )
            bundles.append({"index": i, "cards": cards})
        state["bundles"] = bundles

    def _fill_relic_reward(self, state: dict[str, Any]) -> None:
        """Render a relic reward (Lost Wisp event, combat relic, …) naming the
        relic, not a bare "Take relic".

        The ``pick_relic_reward`` option's ``title`` is set to the relic's loc
        key (server enrichment resolves it to the display name) and ``text_key``
        to the relic id (so the server attaches the relic art + description).
        """
        rm = self.rm
        actions = rm.available_actions()
        self._option_actions = list(actions)
        ids = rm.relic_reward_ids() if hasattr(rm, "relic_reward_ids") else []
        head = display_maps.loc_id(ids[0]) if ids else None
        options = []
        for i, a in enumerate(actions):
            kind = a.get("action")
            if kind == "pick_relic_reward" and head:
                options.append(
                    {"index": i, "title": f"{head}.title", "text_key": head, "is_locked": False}
                )
            elif kind == "skip_relic":
                options.append({"index": i, "title": "Skip", "is_locked": False})
            else:
                options.append({"index": i, "title": _action_label(a), "is_locked": False})
        state["options"] = options

    def _fill_shop(self, state: dict[str, Any]) -> None:
        """Render the merchant inventory from ``rm.shop_inventory()``.

        Shows every slot with its cost. Two different facts travel here and
        must not be conflated:

        * ``is_stocked`` — this slot is BUYABLE RIGHT NOW: in stock *and*
          affordable. It gates the buy option, matching the engine's own
          buy-action filter, so every offered buy is accepted.
        * ``is_available`` — the slot still holds an item at all. A bought
          slot has no card/relic/potion left in it.

        The frontend used to read ``is_stocked`` as availability and stamp
        "Sold" across anything you merely could not afford. The real game
        leaves those wares untouched and just paints the price red.
        """
        rm = self.rm
        inv = rm.shop_inventory() or {}
        gold = state["player"]["gold"]

        def affordable(entry: dict[str, Any]) -> bool:
            price = entry.get("price")
            return bool(entry.get("available")) and price is not None and price <= gold

        cards = []
        for e in inv.get("cards", []):
            cid = display_maps.loc_id(e["id"]) if e.get("id") else f"card_{e['idx']}"
            cards.append(
                {
                    "index": e["idx"],
                    "id": cid,
                    "name": f"{cid}.title",
                    "description": f"{cid}.description",
                    "type": "card",
                    "cost": e.get("price"),
                    # The merchant discounts exactly one card per shop
                    # (`MerchantInventory` calls `SetOnSale`, which halves the
                    # rolled price). The game marks it twice — a "%" sales tag
                    # over the card's top-right corner and a GREEN cost label
                    # (`NMerchantCard.UpdateVisual`) — so the price alone does
                    # not say why it is low. Cards only; relics and potions
                    # never go on sale.
                    "on_sale": bool(e.get("on_sale")),
                    "is_stocked": affordable(e),
                    "is_available": bool(e.get("available")),
                }
            )
        relics = []
        for e in inv.get("relics", []):
            rid = display_maps.loc_id(e["id"]) if e.get("id") else None
            relics.append(
                {
                    "index": e["idx"],
                    # `id` matches the real DLL's RelicOfferDict shape -- the
                    # AI bridge reads it for the net's relic-identity token
                    # (missing id == every shop relic OOV, 2026-07-16 audit).
                    "id": rid,
                    "name": f"{rid}.title" if rid else "Relic",
                    "cost": e.get("price"),
                    "is_stocked": affordable(e),
                    "is_available": bool(e.get("available")),
                }
            )
        potions = []
        for e in inv.get("potions", []):
            pid = display_maps.loc_id(e["id"]) if e.get("id") else None
            potions.append(
                {
                    "index": e["idx"],
                    # `id` matches the real DLL's PotionOfferDict shape (see
                    # the relic comment above).
                    "id": pid,
                    "name": f"{pid}.title" if pid else "Potion",
                    "cost": e.get("price"),
                    "is_stocked": affordable(e),
                    "is_available": bool(e.get("available")),
                }
            )
        state["cards"] = cards
        state["relics"] = relics
        state["potions"] = potions
        # Card-removal service: advertise the price while the engine still
        # offers ``shop_remove_card`` actions (they vanish once the service is
        # used or unaffordable). The C# backend sets the same field; the
        # frontend's "Remove a card · Ng" button is gated on it.
        if any(a.get("action") == "shop_remove_card" for a in rm.available_actions()):
            state["card_removal_cost"] = int(inv.get("removal_cost") or 0)

    def _fill_rest(self, state: dict[str, Any]) -> None:
        rm = self.rm
        acts = rm.available_actions()
        rest_acts = [a for a in acts if a.get("action") == "rest_option"]
        # Smith opens a sub-state (still phase RestSite) whose actions are
        # ``smith_card`` card picks over the deck's upgradeable cards. Render
        # it as a card_select (names / art / text) — matching the C# backend —
        # rather than bare "Upgrade card N" buttons. ``smith_card`` idx maps
        # 1:1 to ``smith_options()``.
        smith_acts = [a for a in acts if a.get("action") == "smith_card"]
        smith_ids = rm.smith_options()
        if smith_acts and smith_ids:
            self._fill_smith(state, smith_acts, smith_ids)
            return
        # ``(option_id, enabled)`` per offered option, in ``rest_option`` idx
        # order. This is what lets the UI name the option and find its plaque
        # art: the frontend keys ``ui/rest_site/option_<option_id>.png`` and
        # the loc row ``rest_site.OPTION_<ID>.name`` off ``option_id``.
        try:
            engine_opts = list(rm.rest_site_options())
        except AttributeError:  # wheel predating the accessor
            engine_opts = []
        # choose_option indexes into _option_actions. The engine does not offer
        # a DISABLED option (Smith with nothing to upgrade) as an action, but
        # C# still shows it greyed, so render every engine option at its idx
        # (``is_enabled`` False; picking it is rejected). With no enabled
        # option left (MiniatureTent) the remaining actions — leave — follow.
        # Without the accessor, drive off whichever set is live so the screen
        # is never empty.
        if engine_opts:
            self._option_actions = [
                {"action": "rest_option", "idx": i} for i in range(len(engine_opts))
            ]
            if not rest_acts:
                self._option_actions += list(acts)
        else:
            self._option_actions = rest_acts or list(acts)
        options = []
        for i, a in enumerate(self._option_actions):
            if a.get("action") == "rest_option":
                idx = a.get("idx")
                pair = engine_opts[idx] if isinstance(idx, int) and idx < len(engine_opts) else None
                if pair is None:
                    option_id = _REST_LABELS.get(idx, f"Option {idx}")
                    name, enabled = option_id, True
                else:
                    option_id, enabled = str(pair[0]), bool(pair[1])
                    name = f"rest_site.OPTION_{option_id}.name"
            else:
                option_id = _action_label(a)
                name, enabled = option_id, True
            options.append(
                {"index": i, "option_id": option_id, "name": name, "is_enabled": enabled}
            )
        state["options"] = options

    def _fill_smith(
        self, state: dict[str, Any], smith_acts: list[dict[str, Any]], smith_ids: list[str]
    ) -> None:
        """Render the Smith upgrade sub-state as a deck card_select.

        Cards are the upgradeable deck candidates; the player picks one to
        upgrade. ``select_cards`` with the chosen card's ``index`` resolves to
        the matching ``smith_card`` action (see :meth:`_translate`).
        """
        cards = []
        for a in smith_acts:
            idx = a.get("idx")
            if idx is None or not (0 <= idx < len(smith_ids)):
                continue
            cid = display_maps.loc_id(smith_ids[idx])
            cards.append(
                {
                    "index": idx,
                    "id": cid,
                    "name": f"{cid}.title",
                    "description": f"{cid}.description",
                    "upgraded": False,
                    "type": "card",
                }
            )
        state["decision"] = "card_select"
        state["cards"] = cards
        state["min_select"] = 1
        state["max_select"] = 1
        state["can_skip"] = False
        state["context"] = "upgrade"

    def _fill_event(self, state: dict[str, Any]) -> None:
        rm = self.rm
        # The Rust port doesn't bind the room's random event until the first
        # choice resolves (only Neow is pre-bound), so on arrival there are no
        # options to show — just a lone "Continue". Materialize it up front
        # (deterministic, RNG-free) so the real name + choices appear, matching
        # how C# presents an event on entry. No-op once an event is bound.
        event_pyname = rm.materialize_event() or rm.current_event_name()
        # Rich per-option info ({option_id,label,description,enabled}) keyed by
        # the EventChoice idx; None when no event is materialized.
        infos = rm.event_option_infos() or []
        options = []
        for a in rm.available_actions():
            if a.get("action") != "event_choice":
                continue
            idx = a["idx"]
            info = infos[idx] if 0 <= idx < len(infos) else None
            option_id = info["option_id"] if info else ""
            label = (info["label"] if info else "") or "Continue"
            # Neow relic offers come back as relic ids (all-caps); expose them
            # via a ``text_key`` tail so server enrichment finds the art/desc.
            looks_like_id = label.isupper() and "_" in label
            if looks_like_id:
                title = f"{display_maps.loc_id(label)}.title"
                text_key = label
            else:
                title = label
                text_key = ""
            # The engine's own `label` / `description` are DEVELOPER SHORTHAND
            # written for run logs — "Gain a random potion." where the game
            # says "Procure 1 random Potion.". When the option carries a
            # `loc_key`, hand the server the game's own loc keys instead and
            # let it render them against `localization_eng/events.json` with
            # `vars`. Bug report: Wellspring's Bottle.
            #
            # NATIVE ONLY, deliberately. The `csharp` dialect feeds straight
            # into the deployed PPO dispatch, which conditions on this text
            # (`realgame_ppo._event_option_label` plus a description parse).
            # Swapping the strings under it changes a shipped net's input,
            # which is a measured change, not a display fix — so that path
            # keeps the shorthand it has always had.
            loc_key = (info or {}).get("loc_key") or ""
            if loc_key and not looks_like_id and self._target_convention == "native":
                title = f"{loc_key}.title"
                description = f"{loc_key}.description"
            else:
                # No loc row (Ancient / Neow relic offers, per-run-chosen
                # rows): the relic's own text via `text_key`, else shorthand.
                description = "" if looks_like_id else (info["description"] if info else "")
            options.append(
                {
                    "index": idx,
                    "title": title or f"Option {idx}",
                    "description": description,
                    "option_id": option_id,
                    "text_key": text_key,
                    "vars": dict((info or {}).get("vars") or {}),
                    # Title-only overrides; empty for all but The Future of
                    # Potions, whose `{Rarity}` means two different things
                    # across its title and its description.
                    "title_vars": dict((info or {}).get("title_vars") or {}),
                    # What to show if the loc template needs a var the engine
                    # did not supply — the server falls back rather than
                    # printing a bare "{Gold}".
                    "fallback_description": (info["description"] if info else ""),
                    "fallback_title": label,
                    "is_locked": bool(info and not info["enabled"]),
                }
            )
        state["options"] = options
        # Event title + body as loc keys; the server resolves them (and blanks
        # any our bundle lacks, e.g. NEOW). resolve_name prettifies a missing
        # title key (NEOW.title -> "Neow").
        if event_pyname:
            stem = _event_loc_stem(event_pyname)
            state["event_name"] = f"{stem}.title"
            state["description"] = f"{stem}.pages.INITIAL.description"
        else:
            state["event_name"] = ""

    def _fill_boss_relic(self, state: dict[str, Any]) -> None:
        """Render the post-boss boss-relic choice as 3 named relics.

        The rust backend offers `pick_boss_relic{idx=0..2}` actions on
        the BossRelic phase; here we resolve each ``idx`` to the actual
        relic's loc key so the server enriches it with art + name +
        description (same path the relic-reward and shop screens take),
        instead of showing "Relic 0/1/2" buttons.
        """
        rm = self.rm
        actions = rm.available_actions()
        self._option_actions = list(actions)
        ids = rm.boss_relic_options() if hasattr(rm, "boss_relic_options") else []
        options = []
        for i, a in enumerate(actions):
            kind = a.get("action")
            if kind == "pick_boss_relic":
                idx = a.get("idx")
                rid = ids[idx] if isinstance(idx, int) and 0 <= idx < len(ids) else None
                if rid:
                    loc = display_maps.loc_id(rid)
                    options.append(
                        {"index": i, "title": f"{loc}.title", "text_key": loc, "is_locked": False}
                    )
                else:
                    options.append({"index": i, "title": _action_label(a), "is_locked": False})
            else:
                options.append({"index": i, "title": _action_label(a), "is_locked": False})
        state["decision"] = "event_choice"
        state["options"] = options
        state["event_name"] = ""

    def _fill_generic(self, state: dict[str, Any], phase: str | None = None) -> None:
        """One option per available action — never-stuck fallback.

        Used for treasure / boss-relic / combat-relic reward screens. The
        emitted ``options`` index maps 1:1 to :attr:`_option_actions`, which
        :meth:`_translate` resolves on ``choose_option``.

        A treasure room is named. It reaches the UI as an ``event_choice`` like
        everything else here, and with a blank ``event_name`` the frontend had
        no way to tell "you are standing in front of a chest" from any other
        one-button screen — so the room the game gives a whole scene to showed
        up as a bare Open bar.
        """
        rm = self.rm
        self._option_actions = list(rm.available_actions())
        options = []
        for i, a in enumerate(self._option_actions):
            options.append({"index": i, "title": _action_label(a), "is_locked": False})
        state["decision"] = "event_choice"
        state["options"] = options
        state["event_name"] = "Treasure" if phase == "treasure" else ""


class _Unsupported(Exception):
    """An action the rust backend can't express (surfaced as an error dict)."""


# Pending interactive deck mutations, in the order they preempt the phase:
# the engine's counter attribute, and the renderer for its pick. Removal
# (Precise Scissors from Neow), upgrade (Pomander; bead sts2-rs-1hiht),
# transform (New Leaf, Wood Carvings' Bird/Torus; GH #34), enchant (Wood
# Carvings' Snake → Slither), and the multi-pick card grant (RoomFullOfCheese
# Gorge, rendered as a mandatory pick-N matching C# ``FromSimpleGridForRewards``).
_PREEMPTS = (
    ("pending_deck_removals", RustGame._fill_card_removal),
    ("pending_deck_upgrades", RustGame._fill_card_upgrade),
    ("pending_deck_transforms", RustGame._fill_card_transform),
    ("pending_deck_enchants", RustGame._fill_card_enchant),
    ("pending_card_grants", RustGame._fill_card_grant),
)


def fill_combat_choice(state: dict[str, Any], pc: dict[str, Any]) -> None:
    """Render an in-combat pending choice (Headbutt, Warcry, …) as a
    ``card_select`` on *state*.

    ``pc`` is ``combat_state().pending_choice()``; its option order maps 1:1
    to the ``combat_resolve_choice`` actions, so the card ``index`` is the
    option index the verb translation resolves on. Shared by the live backend
    and the combat-replay backend so the two can't drift.
    """
    cards = []
    for i, opt in enumerate(pc.get("cards") or []):
        cid = display_maps.loc_id(opt.get("id", ""))
        cards.append(
            {
                "index": i,
                "id": cid,
                "name": f"{cid}.title",
                "description": f"{cid}.description",
                "upgraded": bool(opt.get("upgraded")),
                "type": "card",
            }
        )
    state["decision"] = "card_select"
    state["cards"] = cards
    state["min_select"] = pc.get("min_select", 1)
    state["max_select"] = pc.get("max_select", 1)
    state["can_skip"] = bool(pc.get("can_skip"))
    # ``kind`` (e.g. PutBack) lets the UI label the prompt; default to a
    # generic in-combat selection context.
    state["context"] = str(pc.get("kind") or "combat_choice").lower()


def _combat_card_dict(
    index: int,
    wire: dict[str, Any],
    can_play: bool,
    target_type: str,
    *,
    loc: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Frontend hand/pile card dict from a Rust wire card.

    ``name`` defaults to a loc key (``STRIKE_IRONCLAD.title``) for the server's
    ``_enrich`` to resolve at serve time. When ``loc`` is supplied (the offline
    MCTS recorder), it is resolved eagerly so the persisted log is directly
    readable (``Strike``) without a server round-trip.
    """
    cid = display_maps.loc_id(wire["id"])
    stats: dict[str, int] = {}
    if wire.get("base_damage"):
        stats["damage"] = wire["base_damage"]
    if wire.get("base_block"):
        stats["block"] = wire["base_block"]
    name = f"{cid}.title"
    if loc is not None:
        name = localization.resolve_name(name, loc)
    out = {
        "index": index,
        "id": cid,
        "name": name,
        "description": f"{cid}.description",
        "stats": stats,
        "cost": wire.get("cost_override", 0),
        # The card's PRINTED cost, next to the effective one above. The frame
        # colours the numeral off the difference, exactly as
        # `CardCostHelper.GetEnergyCostColor` does: chartreuse when a hook or a
        # local modifier pushed the cost DOWN (Unrelenting's FreeAttack zeroes
        # the next Attack in hand), energy-blue when it pushed it up. Without
        # the base the frame can only print "0" and never say why.
        "base_cost": wire.get("base_cost"),
        "type": wire.get("kind", "card"),
        "upgraded": wire.get("upgraded", False),
        "can_play": can_play,
        "target_type": target_type,
    }
    # A value the card only knows on a live board — Ashen Strike's "Deal 6
    # damage." is really 6 + 3 × <Exhaust Pile>. The engine computes it from
    # the same function the card's play handler uses. Carried ONLY when the
    # engine sent one: this dict is an allow-list, so an omitted key here is
    # how the ~530 cards without such a value say "I have no such number",
    # which `server.enrich_card` must not read as zero.
    if wire.get("calculated_value") is not None:
        out["calculated_value"] = wire["calculated_value"]
    # Per-card affliction (Bound / Smog / Entangled / ...) — the game's own
    # restriction marker, and the only thing on the board that says a card is
    # chained BEFORE you try to play it: `NCard.ReloadOverlay` lays the
    # affliction's overlay over the card and `CardModel.GetDescriptionForPile`
    # appends its extra card text in purple. Without it the workbench's Queen
    # fight showed no trace of ChainsOfBinding at all — the legality gate was
    # right (the engine drops every further Bound card from `legal_actions`
    # once one has been played) but the board never said why, so the fight read
    # as "Chains does nothing here".
    #
    # Field names mirror the C# backend (`RunSimulator.cs:3799-3802`) so both
    # lanes hand `server.enrich_card` the same shape. The Rust wire's value is
    # the affliction TYPE name ("Bound"), which is also its English title, so
    # the two lanes agree on the string as well.
    if wire.get("affliction"):
        out["affliction"] = wire["affliction"]
        if wire.get("affliction_stacks"):
            out["affliction_amount"] = wire["affliction_stacks"]
    return out


def _stolen_card_dict(wire: dict[str, Any], loc: dict[str, str] | None) -> dict[str, Any]:
    """Frontend card dict for a card a `SwipePower` thief is holding hostage.

    The engine ships only the identity (``{id, upgraded}``) — a stolen card has
    been removed from every combat pile, so there is no live wire card to copy
    stats off. That is enough: ``server.enrich_card`` resolves art, type,
    rarity, cost and the at-rest description from the manifest, exactly as it
    does for master-deck cards outside combat.

    ``name`` / ``description`` stay loc keys for ``server._enrich`` unless
    ``loc`` is given (the offline recorder), matching :func:`_combat_card_dict`.
    """
    cid = display_maps.loc_id(wire["id"])
    name = f"{cid}.title"
    if loc is not None:
        name = localization.resolve_name(name, loc)
    return {
        "index": -1,
        "id": cid,
        "name": name,
        "description": f"{cid}.description",
        "upgraded": bool(wire.get("upgraded")),
        "type": "card",
    }


def _owned_entry(wire: dict[str, Any] | str, loc: dict[str, str] | None) -> dict[str, Any]:
    """Owned relic/potion dict from a wire entry (``{id}`` or a bare id string).

    ``name`` / ``description`` default to loc keys (``ICE_CREAM.title`` /
    ``ICE_CREAM.description``) for ``server._enrich`` to resolve; when ``loc`` is
    supplied (the recorder) they resolve eagerly so the log is directly
    readable and the relic-tray hover text is present. An unresolved description
    key is blanked rather than shown raw.
    """
    raw_id = wire.get("id", "") if isinstance(wire, dict) else wire
    cid = display_maps.loc_id(raw_id)
    name = f"{cid}.title"
    description = f"{cid}.description"
    if loc is not None:
        name = localization.resolve_name(name, loc)
        resolved = localization.resolve(description, loc)
        description = "" if localization.looks_like_unresolved_key(resolved) else resolved
    return {"id": cid, "name": name, "description": description}


def build_combat_view(
    cs: Any,
    character: str,
    *,
    legal: list[dict[str, Any]],
    loc: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Canonical ``combat_play`` state dict from a native ``sts2_rs`` combat state.

    This is the single source of truth for the frontend combat shape, shared by
    the interactive/replay backend (:meth:`RustGame._fill_combat`) and the
    offline MCTS recorder (:mod:`workbench.record`).

    Args:
        cs: A native ``sts2_rs`` combat state (anything ``state_to_dict`` accepts).
        character: Display character name for ``player.name`` (e.g. ``Ironclad``).
        legal: The legal-action list used for hand legality + targeting. Both
            shapes are accepted: the run backend's ``{action: 'play_card',
            hand_idx, target_index}`` and the native ``{action: 'play',
            card_index, target_index}``.
        loc: Optional card loc map; when given, card names resolve eagerly.

    Returns:
        A combat-play state dict (``player`` / ``player_powers`` / ``enemies`` /
        ``hand`` / pile counts + contents), with enemy intents in the
        ``{type, damage, hits}`` shape the recompute/search path requires.
    """
    import sts2_rs

    wire = sts2_rs.state_to_dict(cs)

    # Legality + targeting per hand position from the legal-action set. Accept
    # both the run backend's ``play_card``/``hand_idx`` shape and the native
    # ``play``/``card_index`` shape so either caller can pass its own.
    targeted: dict[int, bool] = {}
    playable: set[int] = set()
    for a in legal:
        if a.get("action") in ("play", "play_card"):
            raw_ci = a.get("card_index", a.get("hand_idx"))
            if raw_ci is None:
                continue
            ci = int(raw_ci)
            playable.add(ci)
            targeted[ci] = targeted.get(ci, False) or (a.get("target_index") is not None)

    # Per-enemy modified-damage preview, aligned with the hand: each entry is a
    # ``{full_encounter_index: modified_per_hit_damage}`` dict with Strength /
    # Vulnerable / Weak / relic-additive folded in (the same ``compute_modified
    # _damage`` stage ``play_card`` runs). Lets the UI show the live "9" on a
    # Strike aimed at a Vulnerable target instead of the base "6". Best-effort:
    # an older wheel without the binding simply yields no previews.
    damage_preview: list[dict[int, int]] = []
    if hasattr(cs, "card_damage_preview"):
        try:
            damage_preview = cs.card_damage_preview()
        except Exception:  # pragma: no cover - defensive; degrade to no preview
            damage_preview = []

    # The block half of the same idea (GH #520): the block a card would really
    # grant, Dexterity / Frail / Fasten folded in. Without it a Defend under
    # Dexterity 5 rendered its printed "5 Block" while granting 10, and a human
    # comparing it against Necrobinder's "summon 7" / "summon 9" read the
    # mitigation ranking backwards. Hand-aligned, ``None`` for non-block cards.
    block_preview: list[int | None] = []
    if hasattr(cs, "card_block_preview"):
        try:
            block_preview = cs.card_block_preview()
        except Exception:  # pragma: no cover - defensive; degrade to no preview
            block_preview = []

    # The live energy, for the unplayable-reason split below.
    energy = int((wire.get("player") or {}).get("energy") or 0)

    hand = []
    for i, c in enumerate(wire["deck"]["hand"]):
        tt = "AnyEnemy" if targeted.get(i) else "None"
        card = _combat_card_dict(i, c, i in playable, tt, loc=loc)
        if i not in playable:
            # WHY the card cannot be played, which the game paints as two
            # different things: `NCard.UpdateEnergyCostVisuals` reddens the cost
            # numeral for a resource shortfall, and shows the red `unplayable`
            # slash over the orb for everything else
            # (`unplayableEnergyIcon.Visible = !reason.HasResourceCostReason()`,
            # NCard.cs:1019-1026). A Bound card under ChainsOfBinding is the
            # second kind: reddening its cost claims you are short on energy
            # when the real answer is "you already played a Bound card".
            #
            # The engine's own `UnplayableReason` is not on the wire, so this
            # reconstructs the one bit the frontend needs from the effective
            # cost and the live energy. Known gap: the Rust wire carries no
            # per-card STAR cost (Regent), so a star-short card lands in
            # "blocked" rather than "cost" — the same lane already renders no
            # star gem at all, so this narrows nothing further.
            cost = card.get("cost")
            card["unplayable_reason"] = (
                "cost" if isinstance(cost, int) and cost > energy else "blocked"
            )
        if i < len(damage_preview) and damage_preview[i]:
            # JSON object keys are strings; normalize here so the frontend can
            # look up ``damage_preview[String(enemyIndex)]`` consistently.
            card["damage_preview"] = {str(k): int(v) for k, v in damage_preview[i].items()}
        if i < len(block_preview) and block_preview[i] is not None:
            # Only when it actually differs from the printed value — an
            # unmodified card must not render a redundant "5 → 5".
            blk = int(block_preview[i])  # ty: ignore[invalid-argument-type]
            if blk != (card.get("stats") or {}).get("block"):
                card["block_preview"] = blk
        hand.append(card)

    enemies = []
    for pos, c in enumerate(wire["encounter"]):
        if c.get("is_dead"):
            continue
        enemies.append(
            {
                # ``index`` must be the position in the *full* encounter array,
                # because that is the ``target_index`` the engine's ``play_card``
                # expects (and the frontend echoes ``index`` back as the target).
                # Re-counting only living enemies shifts the index whenever an
                # earlier enemy dies, so a play would target a corpse and no-op.
                "index": pos,
                # The engine's ``CreatureUid``. ``index`` is a board slot; the
                # combat log keys its per-hit rows by uid
                # (``LogCreature::Monster.uid``), and only this field lets the
                # client join one to the other. ``.get`` (not ``[...]``): the
                # real serializer (``state_dict.rs``) always emits this, but
                # synthetic/older encounter fixtures may not — degrade to
                # ``None`` (the client falls back to snapshot-diff
                # reconciliation) rather than crash a caller that omits it.
                "uid": c.get("uid"),
                # Raw MonsterKind (PascalCase) — the server's ``_enrich`` needs
                # it to kind-address rigs for monsters whose segments share one
                # display name (the three Decimillipede segments all localize to
                # "Decimillipede") and to pick state-dependent idle clips
                # (SpinyToad / Tunneler / TestSubject). Without it that whole
                # path degrades every such enemy to a static sprite.
                "kind": c["kind"],
                "name": display_maps.monster_title(c["kind"]),
                "hp": c["hp"],
                "max_hp": c["max_hp"],
                "block": c.get("block", 0),
                "intents": [_intent(it) for it in c.get("intents", [])],
                "powers": [_power(p) for p in c.get("powers", [])],
                # AI move-state id — carried so the recompute/"search" path can
                # rebuild the enemy AI faithfully (sts2-rs-3f3rz); without it the
                # reconstructed enemy goes passive and every line looks winning.
                "move_state": c.get("move_state"),
            }
        )
        # Cards this enemy is currently holding hostage (ThievingHopper's
        # SwipePower). The game pins the card ON the thief; without it the UI
        # shows a bare "Swipe 1" and the card just disappears from the deck
        # view — while whether you kill the thief before it escapes decides if
        # you ever get it back. Key omitted when there is nothing stolen, so
        # every other enemy's dict keeps its historical shape.
        stolen = [_stolen_card_dict(sc, loc) for sc in c.get("stolen_cards", [])]
        if stolen:
            enemies[-1]["stolen_cards"] = stolen

    wp = wire["player"]
    deck = wire["deck"]
    state: dict[str, Any] = {
        "decision": "combat_play",
        "round": wire.get("round", 0),
        "energy": wp.get("energy", 0),
        "max_energy": wp.get("max_energy", 0),
        "player": {
            "name": character,
            "hp": wp["hp"],
            "max_hp": wp["max_hp"],
            "block": wp.get("block", 0),
            # The owned relics / potions live in the wire combat state, so the
            # recorder surfaces them too (they explain otherwise-mysterious
            # effects, e.g. Ice Cream carrying energy across turns).
            "relics": [_owned_entry(r, loc) for r in wp.get("relics", [])],
            "potions": [_owned_entry(p, loc) for p in wp.get("potions", []) if p],
            "deck": [],
        },
        "player_powers": [_power(p) for p in wp.get("powers", [])],
        "enemies": enemies,
        "hand": hand,
        "draw_pile_count": len(deck.get("draw_pile", [])),
        "discard_pile_count": len(deck.get("discard_pile", [])),
        "exhaust_pile_count": len(deck.get("exhaust_pile", [])),
        "draw_pile": [
            _combat_card_dict(-1, c, False, "None", loc=loc) for c in deck.get("draw_pile", [])
        ],
        "discard_pile": [
            _combat_card_dict(-1, c, False, "None", loc=loc) for c in deck.get("discard_pile", [])
        ],
        "exhaust_pile": [
            _combat_card_dict(-1, c, False, "None", loc=loc) for c in deck.get("exhaust_pile", [])
        ],
    }

    # Necrobinder's Osty pet: the Rust wire nests it under ``player.osty``; lift
    # it to the top-level ``osty`` key the C# backend + art-enrich step expect.
    osty = _top_level_osty(wp.get("osty"))
    if osty is not None:
        state["osty"] = osty

    # Defect orbs: the Rust wire nests them under ``player.orbs`` /
    # ``player.orb_slots``; lift to the top-level ``orbs`` / ``orb_slots`` keys
    # the C# backend emits and the art-enrich step (server.py) + Board read.
    if "orbs" in wp:
        state["orbs"] = wp["orbs"]
        state["orb_slots"] = wp.get("orb_slots", 0)

    # Regent stars: the Rust wire emits ``player.stars`` only when non-zero
    # (``state_dict.rs:2184``); the C# combat view surfaces ``stars`` at the
    # TOP level. Lift it so the parity snapshot can compare the Regent star
    # economy (and a UI host can render the star counter).
    state["stars"] = wp.get("stars", 0)
    return state


def _intent(wire: dict[str, Any]) -> dict[str, Any]:
    """Wire intent ``{kind, amount, times}`` → C# ``{type, damage, hits}``.

    ``damage`` is the EFFECTIVE telegraph — Strength / Weak / Vulnerable folded
    in — which is what the live game shows and what the C# backend already
    emits (``sts2_hint.schema.IntentAmounts``). ``state_to_dict`` stamps
    ``intent_amounts: "base"`` and ships the engine's own base amount, so this
    is the seam that has to convert; forwarding ``amount`` verbatim published
    the base number under an "effective" name and diverged the two backends.

    Passing the base through cost more than a wrong label: the intent-atlas
    dagger tier is picked from this value (``assets.intent``), so a Bygone
    Effigy telegraphing 13-base / 23-real drew the <20 dagger instead of the
    <40 one — and a human reading "13" against a base-5 Defend concluded the
    search was mis-ranking Necrobinder's summons (GH #520, bug report
    ``wb-b5c453e74f1b80e3``).

    ``base_damage`` is kept alongside for callers that want the unmodified
    number; it is omitted when the two coincide.

    ``modifiers`` names the listeners that produced the gap between the two —
    the attacker's Strength, its Weak, the player's Vulnerable — so the board
    can say WHY a telegraph reads what it reads. Bug report
    ``wb-4eae3bad6ebd57f8`` is what that is for: a player applied Weak to an
    enemy that already had it, watched a 12x3 telegraph stay 12x3, and
    reported the engine as broken. Weak is a duration counter, so the x0.75
    was already in the 12 and nothing on the screen said so. Forwarded
    verbatim from the engine (``hooks::damage_modifier_breakdown``) — never
    re-derived here, which is the same discipline ``damage`` itself follows.
    """
    out: dict[str, Any] = {"type": wire.get("kind", "Unknown")}
    base = wire.get("amount")
    if base is not None:
        # `effective_amount` is absent on producers predating it; fall back to
        # the base rather than dropping the telegraph entirely.
        effective = wire.get("effective_amount", base)
        out["damage"] = effective
        if effective != base:
            out["base_damage"] = base
    if wire.get("times") is not None:
        out["hits"] = wire["times"]
    if wire.get("modifiers"):
        out["modifiers"] = [dict(m) for m in wire["modifiers"]]
    return out


def _top_level_osty(wire: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lift the Rust wire ``player.osty`` to the C# top-level ``osty`` shape.

    The Rust serializer emits ``{alive, hp, max_hp, block}`` (no name); the C#
    backend and the art-enrich step expect a top-level ``osty`` carrying a
    ``name`` (used to resolve the ``monsters/osty.png`` sprite). Returns the
    enriched dict, or ``None`` when no pet exists. A dead Osty keeps its
    ``{alive: false}`` payload unchanged (no name → no sprite), matching C#.
    """
    if not wire:
        return None
    if wire.get("alive"):
        return {**wire, "name": "Osty"}
    return wire


def _power(wire: dict[str, Any]) -> dict[str, Any]:
    # ``display_amount`` is the player-facing value for Counter powers (e.g.
    # Slow shows SlowAmount*10 while ``amount`` stays at the stack count).
    # Fall back to ``amount`` when the engine doesn't emit a separate value.
    amount = wire.get("amount", 0)
    raw_name = wire.get("name", "")
    return {
        "name": display_maps.power_title(raw_name),
        # Carry the engine registry id (the power's python_name, e.g.
        # ``MANGLE_POWER``) alongside the display title. The hint adapter
        # (sts2_hint.adapter._translate_power) reconstructs the power from this
        # id unambiguously even when its display title has no localization
        # entry — ``state_from_dict`` accepts the id verbatim. Without it the
        # title round-trip silently drops TemporaryStrength-class powers
        # (Mangle, …), stranding their temporary stat change as permanent and
        # defanging the enemy in the dict-reconstructed search.
        "id": raw_name,
        "amount": amount,
        "display_amount": wire.get("display_amount", amount),
    }


def _treasure_reveal(
    before: dict[str, Any] | None,
    after: dict[str, Any],
    act: int,
) -> dict[str, Any] | None:
    """The "what's inside" payload for a chest just walked into, else ``None``.

    A treasure room is **fully auto-resolved by the engine** — `resolve_treasure_
    room` (sts2-rs `run_manager.rs`) rolls the relic, pays the gold and obtains
    the relic on room entry, then jumps straight to the map, because that is
    what the real headless game does (AutoSlay's `TreasureRoomHandler` always
    grabs it). So no decision ever reaches the UI and a chest used to register
    only as log lines. This is the display payload that lets the frontend show
    the room: it reports what already happened, it does not gate it.

    Fires ONLY on the action that *entered* the room — a later action taken
    while still standing there (an on-obtain card pick, say) compares equal
    rooms and yields ``None``, so the panel doesn't reappear empty-handed.

    An empty chest (Silver Crucible's veto) is a real outcome, not a miss: it
    returns a reveal with no relic and no gold.

    Args:
        before: :meth:`RustGame._room_snapshot` from before the action, or
            ``None`` when no action has been taken yet (run start).
        after: the same snapshot taken after it.
        act: the current act index, selecting the act's own chest art.
    """
    if before is None:
        return None
    room = after.get("room") or {}
    if room.get("room_type") != "treasure":
        return None
    was = before.get("room") or {}
    if was.get("room_type") == "treasure" and was.get("map_coord") == room.get("map_coord"):
        return None  # already in this chest room — not a fresh entry
    had = list(before.get("relics") or [])
    gained = [r for r in (after.get("relics") or []) if r not in had]
    # The chest's own relic is the FIRST new one: `obtain_relic` appends it
    # before any `AfterObtained` listener can queue further relics.
    relic = _owned_entry(gained[0], None) if gained else None
    return {
        "kind": "treasure",
        "act": act,
        "gold": int(after.get("gold", 0)) - int(before.get("gold", 0)),
        "relic": relic,
    }


def _action_label(a: dict[str, Any]) -> str:
    """Human label for a raw Rust action (generic option screens)."""
    kind = a.get("action", "?")
    if kind == "pick_relic_reward":
        return "Take relic"
    if kind == "skip_relic":
        return "Skip"
    if kind == "collect_treasure":
        return "Open"
    if kind == "pick_boss_relic":
        return f"Relic {a.get('idx')}"
    if kind == "sacrifice_card_reward":
        return "Sacrifice"
    if kind == "smith_card":
        return f"Upgrade card {a.get('idx')}"
    return kind.replace("_", " ").title()
