"""Branching, time-travel live session for the workbench.

A :class:`BranchingLiveSession` owns one game subprocess and a **branch tree**
of decision points. It lets the user rewind to any past node and continue from
there on a new branch, keeping the original line intact.

Restore is **replay-based** (spec v1): the game engine is deterministic given
``(character, seed, ascension)`` plus an action sequence, so to position the
game at any node we restart the subprocess and re-apply the recorded actions
along the path from the root. No mid-combat save files are used.

Each node is one recorded step::

    Node(id, parent_id, idx, decision, state, action, args, label, details, error)

where ``action``/``args`` is the action that *produced this node* from its
parent (``None`` for the root). The **active path** (root → active leaf) is what
the live log renders; acting while positioned on a non-leaf node forks a new
child, preserving the original subtree.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import sts2_client
from game_log import GameLogger
from run_policy import to_workbench_verb
from sts2_client import DEFAULT_GAME_VERSION, LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS, Game
from sts2_hint import display_maps

from workbench import (
    autobattle,
    canonicalize_seed,
    localization,
    reasoning,
    regret_review,
    render,
    wpa_review,
)
from workbench.backends import GameBackend, adapt_backend, make_backend
from workbench.combatlog import LogNames, format_combat_log
from workbench.session_log import SessionLog, snapshot
from workbench.statediff import diff_states

_log = logging.getLogger(__name__)

#: Combat-log kinds the game's own log hides but the WORKBENCH's timeline
#: slices must carry anyway, because the board's VFX layer reads its effect
#: stream off the timeline node (``combatlog.ts``'s ``activeNode``).
#:
#: Exactly one kind, and it is not a display row: ``power_flash`` is a view
#: signal (C# ``NCreature.OnPowerFlashed`` spawning ``power_flash_vfx``). The
#: client re-hides it from every reader-facing panel (``readableRows``).
#: Deliberately NOT ``verbose=True``, which would also let the six
#: card-movement / energy kinds through — nothing here wants those, and they
#: would ride into every saved run's JSONL.
_VFX_ONLY_LOG_KINDS = frozenset({"power_flash"})

# Actions the UI is allowed to send (mirrors the sts2_client.Game / MCP verbs).
ALLOWED_ACTIONS = frozenset(
    {
        "play_card",
        "end_turn",
        "choose_option",
        "select_map_node",
        "select_card_reward",
        "skip_card_reward",
        "buy_card",
        "buy_relic",
        "buy_potion",
        "remove_card",
        "select_bundle",
        "select_cards",
        "skip_select",
        "use_potion",
        "discard_potion",
        "leave_room",
        "proceed",
    }
)

# Sentinel verb for a node produced by the NATIVE run agent (run-mode autoplay).
# Its ``args`` are a raw Rust ``RunAction`` dict, not workbench-verb args, so it
# is replayed through ``rm.take_action`` rather than the verb layer. It is
# deliberately NOT in ``ALLOWED_ACTIONS``: the UI may never send one.
#
# Defined in ``workbench.render`` (which this module imports, not the other way
# round) so the recorder and the label renderer agree on one spelling.
NATIVE_ACTION = render.NATIVE_ACTION


def _attach_full_map(
    game: GameBackend | None, state: dict[str, Any], first_col: int | None = None
) -> dict[str, Any]:
    """At a ``map_select`` decision, attach the connected graph under
    ``state['full_map']`` (best-effort).

    *first_col* seeds the attached ``path_plan``'s route with the move the
    driving run policy will actually make (see
    :meth:`BranchingSession._driving_first_col`)."""
    if game is None or not isinstance(state, dict):
        return state
    if state.get("decision") != "map_select":
        return state
    try:
        full_map = game.get_map(first_col)
    except Exception:
        _log.warning("get_map failed; full_map omitted from state", exc_info=True)
        return state
    if isinstance(full_map, dict) and full_map:
        state["full_map"] = full_map
    return state


@dataclass
class _Node:
    id: int
    parent_id: int | None
    idx: int  # depth from root (root = 0)
    decision: str
    state: dict[str, Any]
    action: str | None = None  # action that produced this node from its parent
    args: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    details: list[str] = field(default_factory=list)
    # This action's slice of the engine's game-style combat log, already
    # rendered for display (``workbench.combatlog``). Empty outside combat and
    # on backends without the log. The UI concatenates these along the active
    # path, so seeking back naturally truncates the panel to that line.
    combat_log: list[dict[str, Any]] = field(default_factory=list)
    error: bool = False
    child_ids: list[int] = field(default_factory=list)
    # PPO critic V(s) of this node's state (shaped-return Φ, NOT a win
    # probability), recorded at node creation while the live RunManager is in
    # hand — the rendered state dict can't rebuild the net's observation later.
    # None in combat / at game over / when the net or rust rm is unavailable.
    value: float | None = None

    def meta(self) -> dict[str, Any]:
        """Lightweight metadata for the timeline (no full state)."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "idx": self.idx,
            "decision": self.decision,
            "label": self.label,
            "details": self.details,
            "combat_log": self.combat_log,
            "error": self.error,
            "child_ids": list(self.child_ids),
            "value": self.value,
        }


def _json_safe(obj: Any) -> Any:
    """Recursively make a PyO3 payload ``json.dumps``-able (bytes → int list).

    The rust agent's ``combat_search`` embeds card-select ``selection`` masks
    as Rust ``Vec<u8>`` → Python ``bytes``. FastAPI's encoder tolerates bytes,
    but the ``quit()`` log writer uses plain ``json.dumps`` and would crash on
    the first such node — sanitize once where the payload enters the state.
    """
    if isinstance(obj, bytes):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _native_action(verb: str | None, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize a live branch node's ``(verb, args)`` into the native engine
    action shape that :func:`workbench.regret_review.action_stable_key` expects
    (the same shape the recorder logs as ``chosen_action.args``).

    A faithful (native-agent) step is stored under the sentinel verb
    ``"__agent_native__"`` with its real RunAction inside ``args`` — see
    :meth:`BranchingLiveSession.faithful_step`. Unwrap that, or the RunAction
    reaches ``action_stable_key`` verbatim: it reads ``action == "play_card"``
    (not ``"play"``) and ``hand_idx`` (not ``card_index"``), resolves to
    ``stable_idx=None``, and every card play fails to match a score row. The
    regret review then returned ZERO decisions for every faithful run — which is
    exactly the "clicking decisions doesn't jump" surface the review exists for.

    Both index spellings are accepted on each verb, so a node recorded under
    either vocabulary normalizes: ``card_index``/``hand_idx`` for a play,
    ``slot``/``potion_slot`` for a potion (``to_workbench_verb`` in
    ``run_policy`` emits the latter of each pair).
    """
    a = dict(args or {})
    kind = a.get("action") if verb == NATIVE_ACTION else verb

    def _first(*keys: str) -> Any:
        return next((a[k] for k in keys if a.get(k) is not None), None)

    if kind in ("play_card", "play"):
        return {
            "action": "play",
            "card_index": _first("card_index", "hand_idx"),
            "target_index": a.get("target_index"),
        }
    if kind == "end_turn":
        return {"action": "end_turn"}
    if kind in ("drink_potion", "potion", "use_potion"):
        return {
            "action": "drink_potion",
            "slot": _first("slot", "potion_slot"),
            "target_index": a.get("target_index"),
        }
    return a  # already native, or a verb regret-review doesn't surface


def _needs_rust(what: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """The advisor error payload for a backend without a native run."""
    return {
        "error": f"{what} requires the Rust engine backend (start the run with backend = rust)",
        "candidates": [],
        "config": config or {},
    }


def _needs_newer_wheel(what: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """The advisor error payload for an ``sts2_rs`` build that cannot fork a run.

    Both advisors clone the live ``RunManager`` and re-seed each rollout, so a
    wheel predating :meth:`RunManager.reseed_run` would raise ``AttributeError``
    deep inside the advisor (a 500 with an opaque message). Say what to do
    instead.
    """
    return {
        "error": f"engine build too old for {what} (rebuild the sts2_rs wheel)",
        "candidates": [],
        "config": config or {},
    }


class BranchingLiveSession:
    """A single in-progress live game with a rewindable branch tree."""

    def __init__(
        self,
        logs_dir: str,
        backend: str = "csharp",
        loc: dict[str, str] | None = None,
        game_version: str | None = None,
    ) -> None:
        self.logs_dir = logs_dir
        self.backend = backend
        # The game build this session's run simulates (GH #801). Fixed for the
        # session's lifetime: ``seek`` rebuilds the run from scratch, and a
        # rebuild under another build would replay the line against different
        # rules (and, since v0.111.0 hashes seeds differently, a different
        # run). On the Rust engine ``None`` is the default build, NOT the
        # process-wide ``STS2_GAME_VERSION`` — a workbench run's build is part
        # of its identity, recorded in its save and log. The C# backend has
        # exactly one build, the one this process's oracle was selected for.
        process_build = sts2_client.GAME_VERSION
        version = game_version or (DEFAULT_GAME_VERSION if backend == "rust" else process_build)
        if version not in SUPPORTED_GAME_VERSIONS:
            raise ValueError(
                f"unsupported game_version {version!r} "
                f"(supported: {', '.join(SUPPORTED_GAME_VERSIONS)})"
            )
        if backend != "rust" and version != process_build:
            # The C# oracle binary is selected per PROCESS (``STS2_GAME_VERSION``
            # at import); a per-session build switch only exists on the Rust
            # engine.
            raise ValueError(
                f"game_version {version!r} requires the rust backend (got {backend!r}, "
                f"whose oracle in this process is {process_build})"
            )
        self.game_version = version
        # Loc map for display-time name resolution of the log labels/diffs. The
        # Rust backend stores loc keys (``STRIKE_IRONCLAD.title``) in state; the
        # C# backend already ships display names, so this is a no-op there.
        self._loc = loc or {}
        # The default C# backend resolves through the module-level ``Game``
        # symbol (kept patchable so tests can inject a FakeGame); other
        # backends go through the factory. ``make_backend`` validates the name.
        self._backend_cls = (
            Game if backend == "csharp" else make_backend(backend, game_version=version)
        )
        # The driver for the current run. Always a ``GameBackend`` — whatever
        # ``_backend_cls`` produces is adapted in :meth:`_spawn_game`, so the
        # code below never has to probe the engine for what it can do.
        self.game: GameBackend | None = None
        self.character = ""
        self.seed = ""
        self.ascension = 0
        # Which Act-1 model this run walks (GH #544). `seek` rebuilds the run
        # from scratch, so this must be remembered, not re-derived.
        self.act1 = "overgrowth"
        self._nodes: dict[int, _Node] = {}
        self._next_id = 0
        self._root_id: int | None = None
        self._active_id: int | None = None
        # Live debug trace (actions + seek/branch navigation); opened in start().
        self._slog: SessionLog | None = None
        # Native combat-MCTS agent (heuristic meta), set by start_run_mode()
        # (start_faithful() is its heuristic-brain alias); drives
        # run_mode_step() and play_to_end(). None when no run-mode run is active.
        self._faithful_agent: Any = None  # sts2_rs.HeuristicRunAgent | None
        # True iff this run was created by start_run_mode() (or its start_faithful
        # alias) — i.e. a Neow-start, numeric-seed, run_arena-shaped run that
        # seek() must rebuild via start_faithful(). Tracked explicitly rather than
        # inferred from `_faithful_agent is not None`: run_mode_step() and
        # wpa_verify() lazily build a combat agent on *interactive* rust runs too,
        # and those must still seek() via the interactive start() path (matching
        # how their actions were recorded, and avoiding int()-ing a text seed).
        self._run_mode: bool = False
        # Cumulative action count for the current faithful run (for play_to_end
        # step counting and seek() depth alignment).
        self._faithful_steps: int = 0
        # Two-axis run mode for autoplay ("play best" / "play to end"):
        #   off_combat_brain — who resolves off-combat multi-option decisions:
        #     "heuristic" (the native HeuristicRunAgent meta, ==
        #     recommend_run_action) / "ppo" (PPO net).
        #   combat_* — combat is ALWAYS the combat MCTS agent
        #     (HeuristicRunAgent); these are the only knobs. Two presets:
        #     Benchmark 1000/4000 (== training and the offline win-rate
        #     sweep) and Quick 200/800 (fast iteration).
        #
        # The fallback here is BENCHMARK, matching the SPA's default and
        # `HeuristicRunAgent`'s own pyo3 default. It used to be Quick, so any
        # path that reached the agent without the client having posted a
        # budget — notably mid-run takeover, where `_agent_step` builds the
        # agent from these fields — silently played the run out at a fifth of
        # the benchmark budget while the UI still displayed 1000/4000.
        self.off_combat_brain: str = "ppo"
        self.ppo_impl: str = "rust"
        self.combat_sims: int = 1000
        self.combat_boss_sims: int = 4000
        self.combat_threads: int = 1
        self.reward_sims: int | None = None
        # -- autobattle mode (run-level drill) --
        # State at the PREVIOUS autobattle stop, so each stop's change-diff spans
        # stop→stop rather than node→node (the autoplayed combats in between are
        # exactly what the diff is meant to summarize). None before the first stop.
        self._autobattle_prev_state: dict[str, Any] | None = None
        # The champion's pick at the CURRENT stop, captured when the autobattle
        # pauses and withheld until the human commits — revealing it alongside
        # the options would destroy the training signal. Consumed by
        # :meth:`autobattle_verdict` after the action lands. See `workbench.autobattle`.
        self._autobattle_pending: dict[str, Any] | None = None
        # Decisions auto-played since the last stop. Accumulates across calls so
        # the change-diff's window is the whole leg even when the leg was walked
        # one visible step at a time (see :meth:`autobattle_step`) — or half
        # watched and half skipped. Reset when a stop is reached.
        self._autobattle_autoplayed = 0
        # Pre-recorded node values consumed by :meth:`_add_node` instead of
        # calling the PPO critic, set only for the duration of a
        # :meth:`load_line` restore, and the read cursor into them. See
        # :meth:`active_node_values`.
        self._restore_values: list[float | None] | None = None
        self._restore_idx = 0
        # The live-game snapshot this run was imported from, or ``None`` for
        # every other start path. ``seek`` rebuilds through it: an imported run
        # is neither a text-seed interactive run nor a numeric-seed faithful
        # one, and restarting it as either would replay the recorded line
        # against a run with a different deck standing somewhere else.
        self._imported: Any = None

    @classmethod
    def from_replay(
        cls,
        logs_dir: str,
        loc: dict[str, str] | None,
        fixture_path: str,
        prefix_actions: list[dict[str, Any]],
        character: str,
        game_version: str | None = None,
    ) -> BranchingLiveSession:
        """A combat-only session that takes over a *recorded* fight from an
        exact decision ("play from here").

        Binds the backend factory to a :class:`CombatReplayBackend` over
        ``(fixture_path, prefix_actions, character)`` so the bound state survives
        ``seek``'s zero-arg backend reconstruction. The caller must still call
        :meth:`start` (its run-param args are ignored by the combat backend).
        Branch tree, hints, and the Board UI are inherited unchanged.
        """
        from functools import partial

        from workbench.backends.combat_replay import CombatReplayBackend

        sess = cls(logs_dir=logs_dir, backend="rust", loc=loc, game_version=game_version)
        sess._backend_cls = partial(CombatReplayBackend, fixture_path, prefix_actions, character)  # type: ignore[assignment]
        return sess

    @classmethod
    def from_imported_combat(
        cls,
        logs_dir: str,
        loc: dict[str, str] | None,
        wire: dict[str, Any],
        character: str,
        game_version: str | None = None,
    ) -> BranchingLiveSession:
        """A combat-only session over a board copied out of the LIVE game.

        The sibling of :meth:`from_replay`: same binding trick, different
        source. ``wire`` is the engine wire dict
        :func:`workbench.state_import.parse_pasted_state` produced, and
        ``state_from_dict`` is a pure function of it, so the bound state
        survives ``seek``'s zero-arg backend reconstruction. The caller must
        still call :meth:`start` (its run-param args are ignored).

        Branch tree, hints, search tree and the Board UI are inherited
        unchanged — which is the point: the position is real, so every
        counterfactual played on it is too.
        """
        from functools import partial

        from workbench.backends.imported_combat import ImportedCombatBackend

        sess = cls(logs_dir=logs_dir, backend="rust", loc=loc, game_version=game_version)
        if sess.game_version != LEGACY_GAME_VERSION:
            # The board is rebuilt by ``sts2_rs.state_from_dict``, which reads
            # the build off ``$.game_version`` (absent = the default). The
            # session's build wins over whatever the wire names, as in the
            # Rust ``Game::start_imported``.
            wire = {**wire, "game_version": sess.game_version}
        sess._backend_cls = partial(ImportedCombatBackend, wire, character)  # type: ignore[assignment]
        return sess

    def _spawn_game(self) -> GameBackend:
        """Construct a fresh driver for this session and opt it into the
        engine's display logs (the card play-log for event-log labels and the
        full game-style combat log for the log panel). Backends without those
        logs accept the opt-in as a no-op and report empty logs."""
        game = adapt_backend(self._backend_cls())
        game.set_record_card_plays(True)
        game.set_record_combat_log(True)
        self.game = game
        return game

    def _native(self) -> Any | None:
        """The in-process ``RunManager`` behind the run, or ``None``."""
        return self.game.native_run() if self.game is not None else None

    def _require_native(self, what: str) -> Any:
        """The in-process ``RunManager``, or ``RuntimeError`` naming *what*
        needed it (the server maps that to a 400 the UI can show)."""
        rm = self._native()
        if rm is None:
            raise RuntimeError(f"{what} requires the rust backend")
        return rm

    def _mark_display_logs(self) -> tuple[int, int]:
        """Lengths of the two combat-scoped display logs, taken BEFORE an action.

        Both logs (the card play-log and the full game-style combat log) are
        cumulative for the whole combat, so the only way to attribute rows to
        the action that produced them is to mark their length first and slice
        the tail afterwards (:meth:`_slice_display_logs`). A backend without a
        log reports it empty, so its mark is 0 and it contributes nothing.
        """
        if self.game is None:
            return 0, 0
        return len(self.game.card_play_log()), len(self.game.combat_log())

    def _slice_display_logs(self, marks: tuple[int, int]) -> tuple[list[str], list[dict[str, Any]]]:
        """One action's tail of the two display logs, rendered for the UI.

        *marks* is what :meth:`_mark_display_logs` returned before the action.
        A log SHORTER than its mark means the action started a new fight (both
        logs reset at combat start), so the whole of the new log is this
        action's — take it all.

        Every path that appends a node must run this pair, not just the
        hand-play one: an autoplayed step whose node carries no tape leaves the
        history sidebar with nothing to draw under it.

        The combat log is rendered with ``keep=`` :data:`_VFX_ONLY_LOG_KINDS`.
        That is not a display choice: this is the ONLY formatter call whose
        output reaches the board's per-hit effect stream (the client reads it
        off the timeline node, via ``combatlog.ts``'s ``activeNode``), and the
        VFX layer needs the ``power_flash`` rows that ``_VERBOSE_ONLY`` hides.
        Those rows still must not be READ, so the client re-hides them
        (``combatlog.ts``'s ``readableRows``) and the log panel and history
        ledger look exactly as they did.
        """
        if self.game is None:
            return [], []
        play_before, clog_before = marks
        play_lines: list[str] = []
        full = self.game.card_play_log()
        if full:
            raw_tail = full[play_before:] if len(full) >= play_before else full
            from workbench.playlog import format_play_log

            play_lines = format_play_log(
                raw_tail,
                name_of=lambda cid: localization.resolve_name(f"{cid}.title", self._loc),
            )
        combat_rows: list[dict[str, Any]] = []
        full_log = self.game.combat_log()
        if full_log:
            tail = full_log[clog_before:] if len(full_log) >= clog_before else full_log
            combat_rows = format_combat_log(
                tail,
                self._log_names(),
                keep=_VFX_ONLY_LOG_KINDS,
                player_id=self._player_loc_id(),
            )
        return play_lines, combat_rows

    def _player_loc_id(self) -> str:
        """The run character's loc id (``"IRONCLAD"``) for combat-log rows.

        ``self.character`` is the workbench's display-ish name ("Ironclad");
        the loc keys are SCREAMING_SNAKE. Rows that name a creature carry the
        id the engine stamped; this covers the ones that are implicitly about
        the player (draws, generated cards, potions, orbs, energy).
        """
        return (self.character or "PLAYER").upper().replace(" ", "_")

    def _log_names(self) -> LogNames:
        """Display-name resolvers for the combat log, off this session's loc.

        Cheap to rebuild per action (closures over two dicts), and rebuilt
        rather than cached because a session's ``_loc`` is swapped when the
        localization data reloads.
        """
        loc = self._loc

        def titled(raw: str) -> str:
            return localization.resolve_name(f"{raw}.title", loc)

        return LogNames(
            card=titled,
            monster=display_maps.monster_title,
            power=display_maps.power_title,
            potion=titled,
            orb=lambda raw: titled(f"{raw}_ORB"),
            character=titled,
        )

    # -- tree helpers ------------------------------------------------------
    @property
    def run_mode(self) -> bool:
        """True iff this run was started run-mode (integer seed, ``start_faithful``).

        A save must record this: it selects which constructor
        :meth:`load_line` restores through, and the two hash the seed
        differently (see :meth:`load_line`).
        """
        return self._run_mode

    def _add_node(self, parent_id: int | None, state: dict[str, Any], **kw: Any) -> _Node:
        node = _Node(
            id=self._next_id,
            parent_id=parent_id,
            idx=0 if parent_id is None else self._nodes[parent_id].idx + 1,
            decision=str(state.get("decision", "")),
            state=state,
            value=self._value_for_new_node(state),
            **kw,
        )
        self._nodes[node.id] = node
        self._next_id += 1
        if parent_id is not None:
            self._nodes[parent_id].child_ids.append(node.id)
        return node

    def _value_for_new_node(self, state: dict[str, Any]) -> float | None:
        """The value to stamp on a node being created: recorded, or computed.

        During a :meth:`load_line` restore the values were already computed in
        the session that wrote the save, so they are replayed off
        ``_restore_values`` rather than recomputed. That is the difference
        between a restore that costs the engine's own replay (~3 ms/step) and
        one that pays a 320-token transformer forward at every non-combat
        decision — on a 780-step act-3 line the critic was ~85% of the load and
        pushed it past two minutes, with no UI feedback while it ran.

        Nodes are created in path order (root first, then one per line entry),
        which is the order :meth:`active_node_values` recorded them in, so a
        positional queue is enough to re-pair them. A short queue means the two
        went out of step: stamp ``None`` (a valueless node is simply skipped by
        the WPA review) rather than silently mis-pairing the rest of the line.
        """
        recorded = self._restore_values
        if recorded is None:
            return self._node_value(state)
        i = self._restore_idx
        self._restore_idx += 1
        if i == len(recorded):
            _log.warning("restore ran out of recorded node values at node %d", i)
        return recorded[i] if i < len(recorded) else None

    def _node_value(self, state: dict[str, Any]) -> float | None:
        """PPO critic V(s) of a freshly created node's state, or ``None``.

        The WPA run review's data source (issue #201): one net forward per
        non-combat decision, recorded live because the rendered state dict
        cannot rebuild the net's observation post-hoc. Deliberately
        failure-proof — combat/game-over states, a backend without an
        in-process RunManager (C#), a missing checkpoint, or any inference
        error all yield ``None`` and never disturb play. Negligible next to
        the combat MCTS the session already runs.
        """
        if state.get("decision") in ("combat_play", "game_over"):
            return None
        rm = self._native()
        if rm is None:
            return None
        try:
            from workbench import ppo_policy

            # `character=` matters: without it every run was scored by the
            # DEFAULT character's champion (Ironclad's net valued Silent's
            # decisions), which the browser port's golden parity caught.
            ev = ppo_policy.evaluate(
                rm, character=self.character, ascension=self.ascension, impl=self.ppo_impl
            )
        except (KeyboardInterrupt, SystemExit):
            # Never swallow interpreter-shutdown / Ctrl-C signals — only inference
            # failures. (These subclass BaseException, so the broad catch below
            # would otherwise trap them.)
            raise
        except BaseException:  # incl. pyo3 panics — value capture must never break play
            _log.debug("WPA value capture failed", exc_info=True)
            return None
        return None if ev is None else ev.value

    def _path_to(self, node_id: int) -> list[_Node]:
        """Nodes from root → ``node_id`` inclusive."""
        chain: list[_Node] = []
        cur: int | None = node_id
        while cur is not None:
            node = self._nodes[cur]
            chain.append(node)
            cur = node.parent_id
        chain.reverse()
        return chain

    def _active(self) -> _Node:
        if self._active_id is None:
            raise RuntimeError("no live run")
        return self._nodes[self._active_id]

    # -- lifecycle ---------------------------------------------------------
    def start(
        self, character: str, seed: str, ascension: int, act1: str = "overgrowth"
    ) -> dict[str, Any]:
        """Spawn the game, start the run, create the root node, return its state.

        ``act1`` selects which of the two Act-1 models the run walks
        (GH #544): ``"overgrowth"`` | ``"underdocks"`` | ``"roll"``. ``"roll"``
        reproduces the real lobby's seed-driven pick, which is uniform 50/50
        once Underdocks is unlocked and discovered.

        The choice is remembered on the session because :meth:`seek` rebuilds
        the run from scratch and replays the line into it — restarting on the
        other act would replay the line against a completely different game,
        the same failure mode the ``run_mode`` field exists to prevent.
        """
        game = self._spawn_game()
        seed = canonicalize_seed(seed)
        self._run_mode = False
        self.character, self.seed, self.ascension = character, seed, ascension
        self.act1 = act1
        state = self._map_state(
            game.start(character=character, seed=seed, ascension=ascension, act1=act1)
        )
        root = self._add_node(
            None, state, label=f"Started {character} (seed {seed}, A{ascension}, act1 {act1})"
        )
        self._root_id = root.id
        self._active_id = root.id
        self._autobattle_prev_state = None
        self._autobattle_pending = None
        self._autobattle_autoplayed = 0
        self._slog = SessionLog(self.logs_dir, character, seed, ascension, self.backend)
        self._slog.line("ROOT", f"node {root.id}  {snapshot(state)}")
        return state

    def start_imported(self, imported: Any) -> dict[str, Any]:
        """Start a run rebuilt from a state copied out of the live game.

        The run-level twin of :meth:`from_imported_combat`: same run identity
        (seed / character / ascension), the same act maps, with position, deck,
        relics, potions, gold, HP and the RNG cursors grafted on — see
        :meth:`workbench.backends.rust_game.RustGame.start_imported`.

        Requires the rust backend: the graft is built out of ``RunManager``
        affordances the C# oracle has no analogue for.

        Args:
            imported: A :class:`workbench.state_import.RunSnapshot`.

        Returns:
            The root state, as the other ``start*`` methods return.

        Raises:
            UnsupportedOperation: When the backend has no native run to graft
                onto (the C# oracle, the test fakes).
        """
        game = self._spawn_game()
        self._run_mode = False
        self._imported = imported
        self.character = imported.character
        self.seed = imported.seed
        self.ascension = imported.ascension
        state = self._map_state(game.start_imported(imported))
        # `act1` is resolved by the run itself (built with `act1="roll"`), not
        # chosen here — `seek` must remember which one it landed on.
        self.act1 = getattr(game, "act1", "overgrowth")
        root = self._add_node(
            None,
            state,
            label=(
                f"Imported {imported.character} (seed {imported.seed}, "
                f"A{imported.ascension}, act {imported.act}, floor {imported.floor})"
            ),
        )
        self._root_id = root.id
        self._active_id = root.id
        self._autobattle_prev_state = None
        self._autobattle_pending = None
        self._autobattle_autoplayed = 0
        self._slog = SessionLog(
            self.logs_dir, imported.character, imported.seed, imported.ascension, self.backend
        )
        self._slog.line("ROOT", f"node {root.id}  imported  {snapshot(state)}")
        return state

    # -- run-mode autoplay ---------------------------------------------------
    def start_run_mode(
        self,
        character: str,
        seed: str | int,
        *,
        ascension: int = 0,
        off_combat_brain: str = "ppo",
        ppo_impl: str = "rust",
        combat_sims: int = 1000,
        combat_boss_sims: int = 4000,
        threads: int = 1,
        reward_sims: int | None = None,
    ) -> dict[str, Any]:
        """Start a parity-pinned run for run-mode autoplay (rust backend).

        The run is constructed the SAME way `run_arena` / `PpoRunEnv` do (Neow
        start, shared setup, and since GH #399 the deck-mutation flags ON:
        ``interactive_card_removal`` / ``_upgrade`` / ``_transform`` — which
        also gates enchant — plus ``direct_event_relic_grant``), so
        ``heuristic`` reproduces the benchmark and ``ppo`` reproduces training. Combat is always the
        HeuristicRunAgent MCTS at (``combat_sims`` / ``combat_boss_sims``)."""
        if self.backend != "rust":
            raise ValueError("run-mode play-to-end requires the rust backend")
        self.set_run_mode(
            off_combat_brain=off_combat_brain,
            ppo_impl=ppo_impl,
            combat_sims=combat_sims,
            combat_boss_sims=combat_boss_sims,
            combat_threads=threads,
            reward_sims=reward_sims,
        )
        from run_stepper import hash_seed

        seed_int = seed if isinstance(seed, int) else hash_seed(seed)
        game = self._spawn_game()
        self._run_mode = True
        self.character, self.seed, self.ascension = character, str(seed_int), ascension
        # Run-mode goes through `start_faithful`, which is pinned to Overgrowth
        # because it mirrors the PPO training env exactly (see its docstring).
        self.act1 = "overgrowth"
        state = self._map_state(
            game.start_faithful(character=character, seed=seed_int, ascension=ascension)
        )
        root = self._add_node(
            None,
            state,
            label=f"Started {character} (seed {seed_int}, A{ascension}, {off_combat_brain})",
        )
        self._root_id = root.id
        self._active_id = root.id
        self._autobattle_prev_state = None
        self._autobattle_pending = None
        self._autobattle_autoplayed = 0
        self._slog = SessionLog(self.logs_dir, character, str(seed_int), ascension, self.backend)
        self._slog.line("ROOT", f"node {root.id}  {off_combat_brain}  {snapshot(state)}")
        # Build the HeuristicRunAgent ONCE; it carries the MCTS budget across
        # all step() calls so the policy is stateful across pause / resume.
        # Constructed for EVERY brain since combat is always driven by the
        # native agent's MCTS regardless of the off-combat brain.
        import sts2_rs

        self._faithful_agent = sts2_rs.HeuristicRunAgent(
            seed=seed_int, sims=combat_sims, boss_sims=combat_boss_sims, threads=threads
        )
        self._faithful_steps = 0
        return state

    def start_faithful(
        self,
        character: str,
        seed: str | int,
        *,
        ascension: int = 0,
        sims: int = 1000,
        boss_sims: int = 4000,
        threads: int = 1,
    ) -> dict[str, Any]:
        """Back-compat alias: the ``heuristic`` brain at the given budget (== the
        old faithful / `run_arena` mode)."""
        return self.start_run_mode(
            character,
            seed,
            ascension=ascension,
            off_combat_brain="heuristic",
            combat_sims=sims,
            combat_boss_sims=boss_sims,
            threads=threads,
        )

    def _agent_step(self) -> dict[str, Any]:
        """Apply ONE native-agent decision to the persisted faithful run.

        Calls :meth:`sts2_rs.HeuristicRunAgent.step` once (the agent applies the
        action to ``rm`` internally), records one branch node, and returns a
        dict compatible with the server's ``/api/live/faithful-step`` response
        shape: ``{"state", "steps", "victory", "won", "over", "error", "done",
        "debug"}``.

        ``done`` is True when the agent returned ``None`` (no legal action) or
        the run ended; ``won`` / ``victory`` are identical aliases so both
        the server and the new test contract can key on them.
        """
        if self.game is None or self._faithful_agent is None:
            raise RuntimeError("no faithful run; call start_faithful() first")
        rm = self._require_native("run-mode autoplay")
        # Snapshot the pre-action state for the node label.
        parent_named = localization.resolve_state_names(self._active().state, self._loc)
        # Mark the display logs BEFORE the agent acts, exactly as apply() does:
        # an autoplayed step has to record the same per-action tape a
        # hand-played one does or the history sidebar renders it bare.
        marks = self._mark_display_logs()
        rec = self._faithful_agent.step(rm)
        if rec is None:
            # Policy sees no legal actions — run is finished.
            state = self._map_state(self.game.get_state())
            over = state.get("decision") == "game_over" or rm.is_over()
            won = bool(rm.player_won) if over else False
            return {
                "state": state,
                "steps": self._faithful_steps,
                "victory": won,
                "won": won,
                "over": over,
                "error": None,
                "done": True,
                "debug": None,
                "search": None,
            }
        # Fetch state AFTER the agent applied the action. Resolve names on the
        # raw dict BEFORE _map_state attaches ``full_map``, so the diff sees
        # the same shape apply()'s does.
        raw = self.game.get_state()
        play_lines, combat_rows = self._slice_display_logs(marks)
        details = play_lines + diff_states(
            parent_named, localization.resolve_state_names(raw, self._loc)
        )
        state = self._map_state(raw)
        # Embed the MCTS root distribution into the state if available. Kept
        # for log/bug-report consumers; the live UI reads the ``search``
        # payload built below instead.
        if rec.get("combat_search"):
            rec["combat_search"] = _json_safe(rec["combat_search"])
            state["mcts"] = rec["combat_search"]
        # Build a human-readable label for the timeline node.
        # Translate the native RunAction dict to workbench (verb, args) first so
        # render.describe_action receives the right keys (e.g. card_index not
        # hand_idx) and can resolve card names against the pre-action state.
        verb = rec["action"].get("action", "?")
        try:
            wb_verb, wb_args = to_workbench_verb(rm, rec["action"])
            act_label = render.describe_action(
                parent_named, {"cmd": "action", "action": wb_verb, "args": wb_args}
            )
        except Exception:
            act_label = verb
        # The root distribution of the search that produced THIS move, in the
        # same row shape ``/api/live/hint`` returns — so the live UI's
        # reasoning panel can render an autoplayed move's branches with the
        # component it already uses for a hand-requested hint.
        #
        # Scored against ``parent_named``, the PRE-action state: the rows are
        # about the position the search rooted at, and ``rows_from_mcts``
        # resolves card names out of ``state["hand"]`` by index. By the time
        # this state is rendered the card has been played and the hand has
        # shifted, so labelling against ``state`` would rename every row.
        # That asymmetry is also why this rides in its OWN field instead of
        # the step's ``scores``: those are index-matched onto the live hand by
        # ``stable_idx``, and these no longer line up with it.
        search: dict[str, Any] | None = None
        if rec.get("combat_search"):
            search = {
                "scores": reasoning.rows_from_mcts(rec["combat_search"], parent_named),
                "action": act_label,
                # Also the PRE-action board: a row's ``target_index`` is a slot
                # in the encounter the search saw. Resolving it against the
                # post-action one renames every target after anything dies.
                "enemies": parent_named.get("enemies") or [],
            }
        # Store the native Rust action dict in the node under the sentinel key
        # ``"__agent_native__"`` so seek() can replay it via rm.take_action()
        # directly rather than going through the workbench verb translation layer.
        node = self._add_node(
            self._active_id,
            state,
            action=NATIVE_ACTION,
            args=dict(rec["action"]),
            label=act_label,
            details=details,
            combat_log=combat_rows,
        )
        self._active_id = node.id
        self._faithful_steps += 1
        # Build a structured debug record for the debug log panel.
        debug: dict[str, Any] = {
            "phase": rec["phase"],
            "is_combat": rec["is_combat"],
            "action": act_label,
            "result": rec["result"],
        }
        if rec.get("combat_search"):
            debug["combat_search"] = rec["combat_search"]
            debug["summary"] = f"combat MCTS → {act_label}"
        else:
            debug["summary"] = f"{rec['phase']} → {act_label}"
        if self._slog:
            self._slog.line("STEP", f"node {node.id}  {verb}  {snapshot(state)}")
        over = rec["over"]
        won = rec["won"]
        return {
            "state": state,
            "steps": self._faithful_steps,
            "victory": won,
            "won": won,
            "over": over,
            "error": None,
            "done": False,
            "debug": debug,
            "search": search,
        }

    def _brain_recommend(self) -> dict[str, Any] | None:
        """Off-combat pick for the ``ppo`` brain, or None to defer to the
        native agent (single-option / unknown surface / unavailable). ``ppo``
        degrades to the heuristic when the net can't decide — never crashes."""
        rec = self._ppo_recommend()
        if rec is not None:
            return rec
        return self._heuristic_recommend()  # degrade

    def run_mode_step(self) -> dict[str, Any]:
        """Apply ONE decision under the current run mode and record a node.

        combat and the ``heuristic`` brain go through the native
        HeuristicRunAgent (:meth:`_agent_step`) — its meta IS the heuristic.
        For ``ppo`` an off-combat multi-option decision is resolved by the
        brain and applied via :meth:`apply`; anything the brain declines
        (single-option, unknown surface, combat) defers to the agent so the
        run still advances."""
        if self.game is None:
            raise RuntimeError("no live run")
        rm = self._require_native("run-mode autoplay")
        if self._faithful_agent is None:
            # A run opened via plain start() (hand-play) has no combat agent yet.
            # Build it lazily from the session's current combat budget so autoplay
            # works on ANY live rust run, not just start_run_mode() ones. The agent
            # re-searches combat from rm's live state, so mid-run takeover is safe.
            import sts2_rs
            from run_stepper import hash_seed

            # Hash the seed like start_run_mode() / the engine do — never
            # int() a digit-string: an >i32 numeric seed would overflow the
            # agent's i32 seed param (OverflowError → 500). The agent seed
            # only drives its own MCTS RNG, so it need not match rm's stream.
            seed_int = hash_seed(str(self.seed))
            self._faithful_agent = sts2_rs.HeuristicRunAgent(
                seed=seed_int,
                sims=self.combat_sims,
                boss_sims=self.combat_boss_sims,
                threads=self.combat_threads,
            )
        if self.off_combat_brain == "heuristic" or rm.phase() == "combat":
            return self._agent_step()
        rec = self._brain_recommend()
        if rec is None:
            return self._agent_step()
        result = self.apply(rec["action"], rec["args"])
        state = result["state"]
        over = state.get("decision") == "game_over" or rm.is_over()
        won = bool(rm.player_won) if over else False
        if result["error"] is None:
            self._faithful_steps += 1
        return {
            "state": state,
            "steps": self._faithful_steps,
            "victory": won,
            "won": won,
            "over": over,
            "error": result["error"],
            "done": bool(over) or result["error"] is not None,
            "debug": {"summary": result["label"], "is_combat": False},
            # Off-combat brain pick — no combat tree was searched.
            "search": None,
        }

    def faithful_step(self) -> dict[str, Any]:
        """Back-compat alias for the heuristic brain; see :meth:`run_mode_step`."""
        return self.run_mode_step()

    # -- autobattle mode (run-level drill) ---------------------------------------
    def autobattle_advance(
        self,
        *,
        max_steps: int = 500,
        stop_decisions: frozenset[str] = autobattle.STOP_DECISIONS,
    ) -> dict[str, Any]:
        """Auto-play to the next run-level decision in ONE call, then hand over.

        The autobattle's whole premise (see :mod:`workbench.autobattle`): combats
        are played by the native MCTS agent so the human only makes run-level
        decisions, with context accruing across the run the way it does in real
        play. Returns the stop state plus the change-diff since the *previous*
        stop — deliberately spanning the auto-played steps, because summarizing
        them is what makes the stop cheap to understand.

        This is the **skip** path: the fights resolve invisibly and the caller
        sees only the stop. :meth:`autobattle_step` is the **watch** path, which
        walks the same leg one paintable decision at a time; both share the
        stop→stop diff window, so a leg can be half watched and half skipped and
        still report one honest diff.

        The champion's pick at the stop is captured here but NOT returned;
        :meth:`autobattle_verdict` releases it after the human commits.

        ``max_steps`` bounds the auto-play so a policy that somehow fails to
        advance can't spin the request forever; hitting it returns
        ``stopped: False`` with the state reached.
        """
        if self.game is None:
            raise RuntimeError("no live run")
        played = 0
        over = won = False
        error: str | None = None

        while played < max_steps:
            state = self._active().state
            if state.get("decision") == "game_over":
                over = True
                break
            if autobattle.is_stop_point(state, stop_decisions):
                break
            out = self.run_mode_step()
            played += 1
            self._autobattle_autoplayed += 1
            error = out.get("error")
            if error is not None:
                break
            if out.get("over") or out.get("done"):
                over = bool(out.get("over"))
                won = bool(out.get("victory"))
                break

        settled = self._autobattle_settle(
            over=over, won=won, error=error, stop_decisions=stop_decisions, always_diff=True
        )
        # ``autoplayed`` here is what THIS call played, which is what a caller
        # driving the skip path reports; the diff carries the window total.
        return {**settled, "autoplayed": played}

    def autobattle_step(
        self,
        *,
        stop_decisions: frozenset[str] = autobattle.STOP_DECISIONS,
    ) -> dict[str, Any]:
        """Advance the autobattle by AT MOST one decision, so it can be watched.

        The skip path (:meth:`autobattle_advance`) resolves a whole leg inside
        one request, which is why the fights never appear: the caller only ever
        sees the state at the far end. This is the same leg, one paintable
        decision per call — the client loops it and renders each step, so the
        combats the agent plays are visible instead of inferred from an HP
        number in the diff.

        Returns the run-mode step's payload (``state`` / ``search`` / ``debug``,
        as :meth:`run_mode_step` gives it) plus the autobattle's own framing:
        ``stopped`` when the state reached is a run-level decision, and ``diff``
        — the stop→stop change-diff, present only at a stop (or at the run's
        end), ``None`` mid-fight where there is no window to summarize yet.

        Already at a stop point, this plays nothing and reports the stop, so a
        caller can loop it unconditionally without special-casing the first call.
        """
        if self.game is None:
            raise RuntimeError("no live run")
        state = self._active().state
        over = won = played = False
        error: str | None = None
        step: dict[str, Any] = {}

        if state.get("decision") == "game_over":
            over = True
        elif not autobattle.is_stop_point(state, stop_decisions):
            step = self.run_mode_step()
            played = True
            self._autobattle_autoplayed += 1
            error = step.get("error")
            if step.get("over") or step.get("done"):
                over = bool(step.get("over"))
                won = bool(step.get("victory"))

        settled = self._autobattle_settle(
            over=over,
            won=won,
            error=error,
            stop_decisions=stop_decisions,
            # Mid-fight there is no stop to diff against yet, and resolving both
            # states' names every combat move to render a card nobody sees is
            # pure cost. The stop (and the run's end) still get theirs.
            always_diff=False,
        )
        return {
            **settled,
            "played": played,
            "search": step.get("search"),
            "debug": step.get("debug"),
        }

    def _autobattle_settle(
        self,
        *,
        over: bool,
        won: bool,
        error: str | None,
        stop_decisions: frozenset[str],
        always_diff: bool,
    ) -> dict[str, Any]:
        """Shared tail of both advance paths: is this a stop, and what changed?

        Capturing the champion's pick and closing the diff window happen here so
        the watch and skip paths cannot drift apart on the two things a verdict
        depends on.
        """
        state = self._active().state
        stopped = error is None and not over and autobattle.is_stop_point(state, stop_decisions)
        diff = None
        if always_diff or stopped or over:
            # Diff against name-resolved states so the card reads "Inflame", not
            # "INFLAME.title" — the same resolution apply() does before rendering.
            prev = self._autobattle_prev_state
            diff = autobattle.context_diff(
                localization.resolve_state_names(prev, self._loc) if prev else None,
                localization.resolve_state_names(state, self._loc),
                autoplayed=self._autobattle_autoplayed,
            ).to_dict()
        self._autobattle_pending = self._capture_champion_pick() if stopped else None
        if stopped:
            self._autobattle_prev_state = state
            self._autobattle_autoplayed = 0
        return {
            "state": state,
            "diff": diff,
            "stopped": stopped,
            "over": over,
            "won": won,
            "error": error,
        }

    def _capture_champion_pick(self) -> dict[str, Any] | None:
        """The off-combat brain's pick at the active node, with a display label.

        Best-effort by design: a brain that declines the surface (or fails
        outright) leaves the stop ungraded rather than breaking the autobattle —
        the human still makes the decision, they just get no verdict.
        """
        node = self._active()
        try:
            rec = self._brain_recommend()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # incl. pyo3 panics — a verdict must never break play
            _log.debug("autobattle champion pick failed", exc_info=True)
            return None
        if rec is None:
            return None
        named = localization.resolve_state_names(node.state, self._loc)
        label = render.describe_action(
            named, {"cmd": "action", "action": rec["action"], "args": rec["args"]}
        )
        return {
            "node_id": node.id,
            "action": rec["action"],
            "args": dict(rec["args"]),
            "label": label,
            "options": self._capture_option_probs(),
        }

    def _capture_option_probs(self) -> list[dict[str, Any]]:
        """Per-option policy probabilities at the active node, or ``[]``.

        Feeds the verdict's near-tie band: without them a disagreement can only
        be reported as a miss, which over-punishes a coin flip. Empty in combat
        or when the net can't load — the verdict then degrades to two bands
        rather than inventing a margin.

        Gated on the PPO brain on purpose. ``ppo_policy_panel`` evaluates the
        net regardless of which brain is driving, so under the heuristic brain
        its probabilities would describe a DIFFERENT policy than the one whose
        pick is being graded — a margin computed from them would be measuring
        the wrong thing.
        """
        if self.off_combat_brain != "ppo":
            return []
        try:
            panel = self.ppo_policy_panel()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # a missing margin must never break the autobattle
            _log.debug("autobattle option probabilities failed", exc_info=True)
            return []
        if not panel or not panel.get("ppo_available"):
            return []
        return [
            {"action": o["action"], "args": o["args"], "label": o["label"], "prob": o["prob"]}
            for o in panel.get("options") or []
            if o.get("prob") is not None
        ]

    def autobattle_verdict(self) -> dict[str, Any]:
        """Reveal the champion's pick for the stop the human just committed.

        Called *after* the action lands, never before: seeing the recommendation
        while choosing would destroy the training signal, which is the whole
        point of the drill.

        ``available: False`` (with a reason) when there is nothing to grade —
        no pending pick, the brain declined the surface, or the active node is
        not a child of the stop the pick was captured at (the user seeked away).

        The verdict is deliberately **agreement only**: it says whether you
        matched the champion, never what the choice was worth. The critic's
        value is shaped-return Φ, not a win probability, and win-probability
        grading needs the counterfactual-finish batch that autobattle mode does not
        run. Same honesty rule as :mod:`workbench.wpa_review`.

        Disagreements are split into a near-tie band and a real miss by the
        policy-probability margin (see :func:`workbench.autobattle.verdict_band`),
        because reporting a coin flip as a miss trains the player to match
        noise. ``champion_prob`` / ``your_prob`` ride along so the UI can show
        the actual spread instead of asking the reader to trust the band.

        Load-bearing caveat for anything rendering this: a flat policy means
        the net was nearly *indifferent*, which is NOT evidence the options are
        equally good — an uncertain or undertrained surface looks identical
        from the probabilities alone.
        """
        pending = self._autobattle_pending
        if pending is None:
            return {"available": False, "reason": "no graded decision pending"}
        node = self._active()
        if node.parent_id != pending["node_id"] or node.action is None:
            return {"available": False, "reason": "no action taken at the graded decision"}
        agreed = node.action == pending["action"] and _native_action(
            node.action, node.args
        ) == _native_action(pending["action"], pending["args"])
        options = pending.get("options") or []
        champion_prob = self._prob_of(options, pending["action"], pending["args"])
        your_prob = self._prob_of(options, node.action, node.args)
        margin = (
            None
            if champion_prob is None or your_prob is None
            else max(0.0, champion_prob - your_prob)
        )
        self._autobattle_pending = None
        return {
            "available": True,
            "agreed": agreed,
            "band": autobattle.verdict_band(agreed, margin),
            "margin": margin,
            "champion_prob": champion_prob,
            "your_prob": your_prob,
            "champion_label": pending["label"],
            "your_label": node.label,
            "brain": self.off_combat_brain,
        }

    @staticmethod
    def _prob_of(options: list[dict[str, Any]], action: str, args: dict[str, Any]) -> float | None:
        """Policy probability the panel gave *action*, or ``None`` if absent.

        Matched through :func:`_native_action` so the panel's workbench verb and
        the recorded node's verb compare on the same normalized shape.
        """
        want = _native_action(action, args)
        for option in options:
            if (
                option["action"] == action
                and _native_action(option["action"], option["args"]) == want
            ):
                return option["prob"]
        return None

    def play_to_end(self, *, max_steps: int = 20_000) -> dict[str, Any]:
        """Play a :meth:`start_faithful` run to completion with the
        :class:`sts2_rs.HeuristicRunAgent`, recording every step as a branch node.

        A thin loop over :meth:`run_mode_step`, kept so the parity test and
        the live UI share exactly one code path.  Returns
        ``{"state", "steps", "victory", "won", "over", "error"}``; ``steps``
        counts the actions applied by *this* call (not cumulative).  ``won``
        and ``victory`` are identical aliases.
        """
        if self.game is None or self._faithful_agent is None:
            raise RuntimeError("no faithful run; call start_faithful() first")
        applied_before = self._faithful_steps
        out: dict[str, Any] = {
            "state": self._active().state,
            "victory": False,
            "won": False,
            "over": False,
            "error": None,
        }
        while self._faithful_steps - applied_before < max_steps:
            out = self.run_mode_step()
            if out["done"] or out["over"]:
                break
        return {
            "state": out["state"],
            "steps": self._faithful_steps - applied_before,
            "victory": out["victory"],
            "won": out["won"],
            "over": out["over"],
            "error": out["error"],
        }

    def apply(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        """Forward *action* to the game, append a child of the active node.

        If the active node is not a leaf (the user seeked back), the new child
        forks a branch; the prior subtree is preserved. Returns the same shape
        as :meth:`LiveSession.apply`.
        """
        if self.game is None:
            raise RuntimeError("no live run")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"action not allowed: {action}")

        parent = self._active()
        # Acting on a node that already has children forks a new branch (the user
        # seeked back); record which existing line we diverged from for the trace.
        forked_from = list(parent.child_ids)
        # Resolve loc-key display names before rendering the log label/diff so
        # the history reads "Play Strike", not "Play STRIKE_IRONCLAD.title".
        parent_named = localization.resolve_state_names(parent.state, self._loc)
        label = render.describe_action(
            parent_named, {"cmd": "action", "action": action, "args": args}
        )
        marks = self._mark_display_logs()
        resp = self.game.act(action, **args)
        if isinstance(resp, dict) and resp.get("type") == "error":
            # Illegal move: leave the tree untouched, re-sync the real state.
            state = self._map_state(self.game.get_state())
            if self._slog:
                self._slog.line(
                    "REJECT", f"from node {parent.id}  {label}  ({resp.get('message')})"
                )
            return {
                "state": state,
                "error": resp.get("message", "illegal action"),
                "label": label,
                "details": [],
            }
        play_lines, combat_rows = self._slice_display_logs(marks)
        resp_named = localization.resolve_state_names(resp, self._loc)
        details = play_lines + diff_states(parent_named, resp_named)
        state = self._map_state(resp)
        node = self._add_node(
            parent.id,
            state,
            action=action,
            args=dict(args),
            label=label,
            details=details,
            combat_log=combat_rows,
        )
        self._active_id = node.id
        if self._slog:
            kind = "BRANCH" if forked_from else "ACTION"
            head = f"node {parent.id} -> {node.id}  {action} {dict(args)}  | {label}"
            if forked_from:
                head += f"  [forked from line {forked_from}]"
            self._slog.line(
                kind,
                head,
                details=[
                    *details,
                    f"pre:  {snapshot(parent_named)}",
                    f"post: {snapshot(resp_named)}",
                ],
            )
        return {"state": state, "error": None, "label": label, "details": details}

    def seek(self, node_id: int) -> dict[str, Any]:
        """Replay to ``node_id`` and make it the active node; return its state.

        Restarts the subprocess and re-applies the recorded actions along the
        path from the root. Read-only: no new node is created.

        Raises:
            RuntimeError: if no run is active.
            KeyError: if ``node_id`` is unknown.
        """
        if self.game is None or self._root_id is None:
            raise RuntimeError("no live run")
        if node_id not in self._nodes:
            raise KeyError(node_id)

        path = self._path_to(node_id)
        self.game.close()
        game = self._spawn_game()
        # Rebuild the run the SAME way it was originally constructed. Since
        # GH #399 both constructors set the four deck-mutation flags
        # (`interactive_card_*` / `direct_event_relic_grant`), so what still
        # separates them is the SEED TYPE: `start_faithful` takes an integer
        # seed, the interactive `start()` a text seed (`string_seed`), and the
        # two hash to different runs. Restarting a faithful run through
        # `start()` would therefore replay recorded actions against a
        # completely different run and get rejected with `invalid_for_phase`,
        # so match the original constructor.
        # Rebuild via start_faithful ONLY for genuine run-mode runs (numeric
        # seed, Neow-shaped). NOT keyed off `_faithful_agent is not None`: an
        # interactive run that was later autoplayed (run_mode_step / wpa_verify
        # lazily build a combat agent) still recorded its actions through the
        # interactive start() path and carries a text seed, so it must replay
        # that way — int()-ing a text seed here would otherwise raise.
        faithful = self._run_mode
        if self._imported is not None:
            # A third constructor: an imported run is neither a text-seed
            # interactive run nor a numeric-seed faithful one. Rebuilding it as
            # either would replay the line against a run holding a different
            # deck and standing somewhere else. The graft is deterministic in
            # the snapshot, so this reproduces the same root every time.
            game.start_imported(self._imported)
        elif faithful:
            game.start_faithful(
                character=self.character, seed=int(self.seed), ascension=self.ascension
            )
        else:
            game.start(
                character=self.character,
                seed=self.seed,
                ascension=self.ascension,
                act1=self.act1,
            )
        # Re-apply each node's producing action (skip the root).  Two cases:
        # • ``action == "__agent_native__"``: a native-agent step stored the
        #   raw Rust RunAction dict in ``node.args``; replay via take_action
        #   directly to avoid the workbench verb translation layer.
        # • any other action: a user-interactive or legacy node; replay via
        #   game.act() as before.
        last_resp: dict[str, Any] | None = None
        for node in path[1:]:
            if node.action == NATIVE_ACTION:
                result = self._require_native("replaying a native agent action").take_action(
                    node.args
                )
                if result not in ("advanced", "run_over"):
                    _log.error(
                        "replay diverged at node %d (native=%s): %s",
                        node.id,
                        node.args,
                        result,
                    )
                    raise RuntimeError(
                        f"replay diverged at node {node.id}: take_action returned {result!r}"
                    )
                last_resp = self.game.get_state()
            else:
                assert node.action is not None
                resp = self.game.act(node.action, **node.args)
                if isinstance(resp, dict) and resp.get("type") == "error":
                    _log.error(
                        "replay diverged at node %d (action=%s): %s",
                        node.id,
                        node.action,
                        resp.get("message"),
                    )
                    raise RuntimeError(f"replay diverged at node {node.id}: {resp.get('message')}")
                last_resp = resp
        prev_active = self._active_id
        self._active_id = node_id
        # For agent-driven faithful runs: the agent receives rm per step() call
        # so there is no stale reference to re-point.  Align the step counter
        # to the rewound depth so a subsequent resume doesn't double-count.
        # (The out-of-combat RNG is not re-synced, so a resume after seek is
        # for exploration, not benchmark-exact reproduction.)
        if faithful:
            self._faithful_steps = self._nodes[node_id].idx
        # Use the last replayed action's response (root has none → its recorded
        # start state). Keep the stored state authoritative and return it.
        state = self._map_state(last_resp) if last_resp is not None else path[0].state
        self._nodes[node_id].state = state
        if self._slog:
            target = self._nodes[node_id]
            self._slog.line(
                "SEEK",
                f"node {prev_active} -> {node_id} ({target.decision}) "
                f'"{target.label}"  [rewound, depth {target.idx}]',
            )
        return state

    def state(self) -> dict[str, Any]:
        """Return the active node's current state (re-synced from the game)."""
        if self.game is None:
            raise RuntimeError("no live run")
        node = self._active()
        node.state = self._map_state(self.game.get_state())
        return node.state

    def _driving_first_col(self, state: dict[str, Any] | None = None) -> int | None:
        """Column of the next map room the ACTIVE run policy would take, or None.

        The map overlay paints a route from the Rust pathing surface, which is
        the HEURISTIC brain's plan. With ``off_combat_brain == "ppo"`` a
        different net is the one that will actually move, so the pencil line
        could set off toward a node the run was never going to enter while the
        policy panel under it named another — the same screen giving two
        answers. Feeding this column back into ``path_plan`` makes the drawn
        route the continuation of the move that will happen.

        ``None`` (leave the planner alone) unless PPO is driving, the decision
        really is a map choice, and the net's pick resolves to a map node.
        """
        if self.off_combat_brain != "ppo":
            return None
        if state is None:
            state = self._active().state if self._active_id is not None else None
        if not isinstance(state, dict) or state.get("decision") != "map_select":
            return None
        try:
            rec = self._ppo_recommend()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # a seeded route must never break the map
            _log.debug("PPO map pick failed; route left unseeded", exc_info=True)
            return None
        if not rec or rec.get("action") != "select_map_node":
            return None
        col = (rec.get("args") or {}).get("col")
        return int(col) if isinstance(col, int) else None

    def _map_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """:func:`_attach_full_map` with the route seeded on the driving brain."""
        return _attach_full_map(self.game, state, self._driving_first_col(state))

    def map(self) -> dict[str, Any] | None:
        """Return the current act's full connected map graph, on demand.

        Unlike :func:`_attach_full_map` — which only fires at a ``map_select``
        decision so every action doesn't pay the ``get_map`` cost — this is a
        best-effort fetch the UI can call at *any* decision (combat, card
        reward, shop, …) to show the map. Returns the ``full_map`` dict, or
        ``None`` when no map is available (e.g. the boss room) or on failure.

        Raises:
            RuntimeError: if no run is active.
        """
        if self.game is None:
            raise RuntimeError("no live run")
        try:
            full_map = self.game.get_map(self._driving_first_col())
        except Exception:
            _log.warning("get_map failed; live map unavailable", exc_info=True)
            return None
        if not isinstance(full_map, dict) or not full_map:
            return None
        return full_map

    def timeline(self) -> dict[str, Any]:
        """Return the branch tree metadata plus the active path (root → leaf).

        Each node's ``meta()`` is enriched with the ``floor``/``act`` it sits on,
        carried forward from its ancestor chain (a node's own state may lack them
        — e.g. mid-combat card-play steps before the next ``map_select``). The
        live log uses these to draw act/floor separators. Nodes are stored in
        creation order, which is also ancestor-before-descendant order (a parent
        is always created first), so a single forward pass resolves every node.
        """
        if self._active_id is None:
            return {"nodes": [], "active_path": [], "active_leaf": None}
        resolved: dict[int, tuple[int | None, int | None]] = {}
        nodes: list[dict[str, Any]] = []
        for node in self._nodes.values():
            pf, pa = (
                resolved.get(node.parent_id, (None, None))
                if node.parent_id is not None
                else (None, None)
            )
            floor = node.state.get("floor")
            act = node.state.get("act")
            cur = (floor if floor is not None else pf, act if act is not None else pa)
            resolved[node.id] = cur
            meta = node.meta()
            meta["floor"], meta["act"] = cur
            # Combat round, for the history log's turn grouping. Deliberately
            # NOT carried forward: off-combat states have no round and the UI
            # closes the turn group there.
            meta["turn"] = node.state.get("round")
            nodes.append(meta)
        return {
            "nodes": nodes,
            "active_path": [n.id for n in self._path_to(self._active_id)],
            "active_leaf": self._active_id,
        }

    def hint(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """MCTS scores for the active combat state, or ``None`` outside combat.

        The Rust backend scores natively off its engine ``CombatState`` (no
        lossy C#-dict round-trip); the C# backend goes through the adapter.
        """
        node = self._active()
        if node.state.get("decision") != "combat_play":
            return None
        # Resolve loc-key card names so the per-action labels read "Strike",
        # not "STRIKE_IRONCLAD.title" (Rust backend).
        named = localization.resolve_state_names(node.state, self._loc)
        cs = self._native_combat_state()
        if cs is not None:
            return reasoning.score_native(cs, named, config)
        return reasoning.score(named, config)

    def _agent_scores(self, parent: Any, child: Any) -> list[dict[str, Any]]:
        """The root distribution of the search that actually chose ``child``.

        ``_agent_step`` stashes the native agent's ``combat_search`` on the
        node the move PRODUCED (``child.state["mcts"]``), so the search for the
        decision at ``parent`` lives one node downstream. Rows are labelled and
        targeted against ``parent`` — the position the search rooted at — since
        ``rows_from_mcts`` resolves both out of the state's hand/encounter by
        index, and by ``child`` the card is gone and the slots have shifted.

        Empty when the move was not an agent-searched combat play (a hand-played
        move, an off-combat pick, a run loaded from before this was recorded).
        """
        if parent.state.get("decision") != "combat_play":
            return []
        search = child.state.get("mcts")
        if not isinstance(search, dict) or not search.get("actions"):
            return []
        named = localization.resolve_state_names(parent.state, self._loc)
        return reasoning.rows_from_mcts(search, named)

    def review_run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Full-run regret review over the active line — the live twin of the
        replay ``/api/regret-review``.

        Diffs every combat decision on the active path (root → active leaf)
        against the MCTS line and surfaces the costliest plays. Live nodes carry
        no recorded MCTS stats, so each combat position is re-scored on demand
        (mirroring the replay endpoint's ``recompute`` path). ``floor``/``act``
        are carried forward from the last screen that reported them, since combat
        states don't include them. Raises ``RuntimeError`` when no run is live.
        """
        if self._active_id is None:
            raise RuntimeError("no live run")
        cfg = dict(config or {})
        blunder_pct = float(cfg.pop("blunder_pct", regret_review.DEFAULT_BLUNDER_DROP_PCT))
        top_n = int(cfg.pop("top_n", 2))
        path = self._path_to(self._active_id)
        steps: list[dict[str, Any]] = []
        floor = act = None
        for parent, node in itertools.pairwise(path):
            if parent.state.get("floor") is not None:
                floor = parent.state.get("floor")
            if parent.state.get("act") is not None:
                act = parent.state.get("act")
            if parent.state.get("decision") != "combat_play":
                continue
            named = localization.resolve_state_names(parent.state, self._loc)
            steps.append(
                {
                    "idx": parent.idx,
                    "node_id": parent.id,
                    "state": named,
                    # The AGENT'S OWN root distribution when the move came from
                    # a searched autoplay step. This is what makes the review
                    # agree with the run: it grades the move against the very
                    # numbers that picked it, so an autoplayed line reports its
                    # real regret (zero, wherever the agent took its own best)
                    # instead of a disagreement between two different searches.
                    "scores": self._agent_scores(parent, node),
                    "chosen_action": {"args": _native_action(node.action, node.args)},
                    "meta": {"turn": named.get("round"), "floor": floor, "act": act},
                }
            )

        def _rescore(step: dict[str, Any]) -> list[dict[str, Any]]:
            # Only reached for a decision with no recorded search — a
            # hand-played move, or a run older than the recording. A fresh
            # search here is a DIFFERENT referee than the agent (see
            # `RecommendConfig.hidden_info`: it is at least honest now, but the
            # budget and seed still differ), so these rows are advisory.
            return reasoning.score(step["state"], cfg).get("scores", [])

        return regret_review.review_steps(
            steps, blunder_pct=blunder_pct, top_n=top_n, rescore=_rescore
        )

    def _wpa_rows(self) -> list[dict[str, Any]]:
        """Scored decision rows along the active path for the WPA review.

        One row per node with a recorded value: the node's coordinates plus
        ``taken_label`` — the label of the on-path child, i.e. the action the
        player took AT this decision (the swing to the next scored decision is
        attributed to it). ``floor``/``act`` are carried forward from the last
        screen that reported them, like :meth:`review_run` does.
        """
        path = self._path_to(self._active_id) if self._active_id is not None else []
        rows: list[dict[str, Any]] = []
        floor = act = None
        for i, node in enumerate(path):
            if node.state.get("floor") is not None:
                floor = node.state.get("floor")
            if node.state.get("act") is not None:
                act = node.state.get("act")
            if node.value is None:
                continue
            taken = path[i + 1].label if i + 1 < len(path) else None
            rows.append(
                {
                    "node_id": node.id,
                    "idx": node.idx,
                    "decision": node.decision,
                    "floor": floor,
                    "act": act,
                    "value": node.value,
                    "taken_label": taken,
                }
            )
        return rows

    def wpa_review(self, config: dict[str, Any]) -> dict[str, Any]:
        """WPA run review over the active line — the run-level twin of
        :meth:`review_run` (issue #201).

        Turns the per-decision PPO values recorded at node creation into a
        value curve + ranked turning points. ``available: False`` (with a
        reason) when fewer than two decisions carry a value — e.g. a C#
        backend run or a missing PPO checkpoint. Raises ``RuntimeError`` when
        no run is live.
        """
        if self._active_id is None:
            raise RuntimeError("no live run")
        cfg = dict(config or {})
        top_k = int(cfg.pop("top_k", wpa_review.DEFAULT_TOP_K))
        min_swing = float(cfg.pop("min_swing", wpa_review.DEFAULT_MIN_SWING))
        rm = self._native()
        won: bool | None = None
        if rm is not None and self._active().state.get("decision") == "game_over":
            won = bool(rm.player_won)
        return wpa_review.review(
            self._wpa_rows(),
            top_k=top_k,
            min_swing=min_swing,
            won=won,
            reason_if_empty=(
                "no run-level value data — value recording needs the rust backend "
                "and a loadable PPO checkpoint"
            ),
        )

    def wpa_verify(
        self, node_id: int, *, max_steps: int = 2_000, max_seconds: float = 120.0
    ) -> dict[str, Any]:
        """Counterfactually check a WPA turning point on the live run.

        Agent-finishes the run twice from the flagged decision — once
        continuing from the line the player actually took (seek to the on-path
        child), once taking the PPO policy's preferred *other* option (a new
        branch) — and compares the real outcomes. Both continuations run under
        the session's current run mode, each with a freshly re-seeded combat
        agent so the two arms start from the same MCTS RNG (a paired spot
        check, one deterministic continuation per arm — not a probability
        estimate). The original active leaf is restored afterwards; the two
        continuation branches stay in the tree for inspection.

        Each arm is bounded by BOTH ``max_steps`` and a ``max_seconds``
        wall-clock deadline (checked between decisions), so a pathological run
        that never terminates can't wedge the request forever; a timed-out arm
        returns with ``over=False`` and an ``error`` note, and the verdict
        falls back to "same" rather than lying. (A single decision that hangs
        inside the engine is not interruptible from here — the deadline bounds
        the step *count*, not one runaway C-level call.)

        Raises:
            RuntimeError: no live run, or the backend can't autoplay.
            KeyError: unknown ``node_id``.
            ValueError: node has no recorded follow-up move (a leaf) / no
                alternative option to try.
        """
        if self.game is None or self._active_id is None:
            raise RuntimeError("no live run")
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._require_native("counterfactual verify")
        # Grade the RECORDED line, not the transient active leaf. Clicking a
        # turning point in the Run Review seeks the active leaf onto it (to jump
        # the viewer there), so by verify time the flagged decision is usually no
        # longer a strict ancestor of the active leaf — keying off _active_id
        # here 409'd every such click ("node N is not on the active line").
        # Resolve the played move from the node's OWN recorded subtree instead:
        # its first-created child is the move the run actually made at this
        # decision (verify / manual-branch forks are appended after it). The
        # seeks below rebuild engine state from the root regardless of where the
        # active leaf currently sits, and it is restored in the `finally`.
        decision_node = self._nodes[node_id]
        if not decision_node.child_ids:
            raise ValueError(f"node {node_id} has no follow-up move to compare against")
        played_child = self._nodes[decision_node.child_ids[0]]
        original_active = self._active_id

        import sts2_rs
        from run_stepper import hash_seed

        agent_seed = hash_seed(str(self.seed))

        def _fresh_agent() -> None:
            # Same-seeded agent per arm → both continuations start from the
            # same combat-MCTS RNG state (paired, CRN-style).
            self._faithful_agent = sts2_rs.HeuristicRunAgent(
                seed=agent_seed,
                sims=self.combat_sims,
                boss_sims=self.combat_boss_sims,
                threads=self.combat_threads,
            )

        def _finish() -> dict[str, Any]:
            steps = 0
            deadline = time.monotonic() + max_seconds
            out: dict[str, Any] = {"won": False, "over": False, "error": None}
            timed_out = False
            while steps < max_steps:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                out = self.run_mode_step()
                steps += 1
                if out["done"] or out["over"]:
                    break
            error = out.get("error")
            if timed_out:
                error = f"verify arm timed out after {max_seconds:g}s ({steps} steps)"
            elif not out.get("over") and steps >= max_steps:
                error = error or f"verify arm hit the {max_steps}-step cap without ending"
            return {
                "won": bool(out.get("won")),
                "over": bool(out.get("over")),
                "steps": steps,
                "error": error,
            }

        try:
            # Arm 1 — the played line: continue from the state AFTER the move
            # the player actually made at this decision.
            _fresh_agent()
            self.seek(played_child.id)
            played = {"label": played_child.label, **_finish()}

            # Arm 2 — the alternative: back to the decision itself, ask the
            # PPO panel for its preferred other option, take it (forks a
            # sibling branch), finish.
            self.seek(node_id)
            panel = self.ppo_policy_panel()
            options = (panel or {}).get("options") or []
            # Exclude only the move actually played — an option is a valid
            # alternative if it differs in EITHER the native action shape OR the
            # rendered label (i.e. it is not the exact same move on both axes).
            # Both are checked because agent-made moves store native args
            # (``__agent_native__``) whose shape differs from the panel's
            # workbench-verb args, so matching on one axis alone is unreliable;
            # an AND here would wrongly reject a genuine alternative that merely
            # happens to share the played move's label (or native shape).
            played_native = _native_action(played_child.action, played_child.args)
            alt = next(
                (
                    o
                    for o in sorted(
                        options, key=lambda o: (o.get("prob") is None, -(o.get("prob") or 0.0))
                    )
                    if _native_action(o["action"], o["args"]) != played_native
                    or o["label"] != played_child.label
                ),
                None,
            )
            if alt is None:
                raise ValueError("no alternative option available at this decision")
            _fresh_agent()
            result = self.apply(alt["action"], alt["args"])
            if result["error"] is not None:
                raise RuntimeError(f"alternative rejected by the engine: {result['error']}")
            alternative = {"label": alt["label"], **_finish()}
        finally:
            try:
                self.seek(original_active)
            except Exception:  # leave the tree usable even if the restore replay fails
                _log.exception("wpa_verify: failed to restore the active node")

        note = (
            "one agent-played continuation per line (same-seeded combat search) — "
            "a paired spot check, not a probability"
        )
        if not (played["over"] and alternative["over"]):
            # An arm that didn't reach a terminal state (timed out / hit the
            # step cap) can't be honestly compared on its win flag — don't
            # invent a winner.
            verdict = "same"
            note = "inconclusive — a continuation didn't finish; " + note
        elif played["won"] == alternative["won"]:
            verdict = "same"
        elif alternative["won"]:
            verdict = "alternative_better"
        else:
            verdict = "played_better"
        return {
            "node_id": node_id,
            "decision": decision_node.decision,
            "played": played,
            "alternative": alternative,
            "verdict": verdict,
            "note": note,
        }

    def advise_card_reward(
        self,
        config: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any] | None:
        """Monte-Carlo card-pick advice for the active card-reward decision.

        Returns ``None`` when the active node is not a card-reward decision
        (the caller maps that to HTTP 409). Requires the Rust backend, which
        holds an in-process ``RunManager`` the advisor can fork; on the C#
        backend (a subprocess, not forkable) this returns an ``error`` payload
        the UI surfaces instead of running a meaningless analysis. An
        ``sts2_rs`` build predating ``RunManager.reseed_run`` gets the same
        treatment (see :func:`_needs_newer_wheel`).

        *progress* is an optional per-playout callback forwarded to the backend
        so the streaming endpoint can report live progress.
        """
        node = self._active()
        if node.state.get("decision") != "card_reward":
            return None
        rm = self._native()
        if rm is None:
            return _needs_rust("the card advisor", config)
        if not hasattr(rm, "reseed_run"):
            return _needs_newer_wheel("the card advisor", config)
        from workbench import card_advisor

        return card_advisor.advise(rm, config, progress=progress)

    def rank_choice(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Fast per-option combat-MCTS ranking for the active in-combat choice.

        The cheap counterpart to :meth:`advise_decision` for an in-combat
        single-pick card-select (Headbutt's put-back, an upgrade-in-hand pick,
        …): a single combat MCTS search ranks every option (see
        :func:`workbench.choice_ranker.rank`), instead of the slow advisor's
        fork-and-play-out-the-act-per-option rollouts. Returns ``None`` when the
        active node is not an in-combat ``card_select`` (the caller maps that to
        HTTP 409). Requires the Rust backend, which holds the in-process
        ``RunManager`` whose native combat state the search runs over; the C#
        backend returns an ``error`` payload the UI surfaces instead.
        """
        node = self._active()
        if node.state.get("decision") != "card_select":
            return None
        rm = self._native()
        if rm is None:
            return _needs_rust("the choice ranker", config)
        from workbench import choice_ranker

        return choice_ranker.rank(rm, config)

    def advise_decision(
        self,
        config: dict[str, Any],
        progress: Any | None = None,
    ) -> dict[str, Any] | None:
        """Monte-Carlo option advice for the active shop / event / in-combat choice.

        The generic sibling of :meth:`advise_card_reward`: covers shops, events
        (incl. treasure / boss relic), and single-pick in-combat card selects.
        Returns ``None`` when the active node is not one of those decisions (the
        caller maps that to HTTP 409). Requires the Rust backend, which holds an
        in-process ``RunManager`` the advisor can fork; on the C# backend (a
        subprocess) — or an ``sts2_rs`` build predating
        ``RunManager.reseed_run`` — this returns an ``error`` payload the UI
        surfaces instead.

        *progress* is an optional per-playout callback forwarded to the backend
        so the streaming endpoint can report live progress.
        """
        node = self._active()
        if node.state.get("decision") not in ("card_select", "shop", "event_choice"):
            return None
        rm = self._native()
        if rm is None:
            return _needs_rust("the decision advisor", config)
        if not hasattr(rm, "reseed_run"):
            return _needs_newer_wheel("the decision advisor", config)
        from workbench import decision_advisor

        return decision_advisor.advise(rm, config, progress=progress)

    def set_run_mode(
        self,
        *,
        off_combat_brain: str | None = None,
        ppo_impl: str | None = None,
        combat_sims: int | None = None,
        combat_boss_sims: int | None = None,
        combat_threads: int | None = None,
        reward_sims: int | None = None,
    ) -> None:
        """Update any subset of the two-axis run mode. Validates enums and
        positivity; raises ValueError on a bad value (mapped to HTTP 400 by the
        endpoint).

        A changed combat budget is pushed straight through to a LIVE agent, so
        "stop the run, raise the sims, resume" takes effect from the next
        decision. It has to happen here: the stepper only ever constructs an
        agent when there is none, so before this the new budget sat on the
        session and the run kept searching at the old one with nothing saying
        so. ``set_budget`` mutates the agent in place rather than rebuilding
        it — a fresh agent would rewind the per-run decision counter that
        seeds each search, replaying seeds the run already spent.

        ``combat_threads`` and ``reward_sims`` are stored and echoed but drive
        nothing: ``HeuristicRunAgent`` ignores ``threads`` (its combat search
        is single-threaded), and the boss-sim card-reward chooser
        ``reward_sims`` budgeted was replaced by the brain's own pick. Both
        are kept only so an older SPA can still POST them.
        """
        if off_combat_brain is not None:
            if off_combat_brain not in ("heuristic", "ppo"):
                raise ValueError(
                    f"off_combat_brain must be heuristic|ppo, got {off_combat_brain!r}"
                )
            self.off_combat_brain = off_combat_brain
        if ppo_impl is not None:
            if ppo_impl not in ("python", "rust"):
                raise ValueError(f"ppo_impl must be python|rust, got {ppo_impl!r}")
            self.ppo_impl = ppo_impl
        for name, val in (
            ("combat_sims", combat_sims),
            ("combat_boss_sims", combat_boss_sims),
            ("combat_threads", combat_threads),
        ):
            if val is not None:
                if val <= 0:
                    raise ValueError(f"{name} must be positive, got {val}")
                setattr(self, name, val)
        if reward_sims is not None:
            if reward_sims < 0:
                raise ValueError(f"reward_sims must be >= 0, got {reward_sims}")
            self.reward_sims = reward_sims
        # Push a changed budget onto the live agent. `set_budget` landed with
        # the mid-run knob; guarded so an older wheel degrades to the previous
        # behaviour (budget applies to the NEXT run) instead of 500-ing.
        agent = self._faithful_agent
        if agent is not None and (combat_sims is not None or combat_boss_sims is not None):
            setter = getattr(agent, "set_budget", None)
            if setter is not None:
                setter(sims=combat_sims, boss_sims=combat_boss_sims)

    def recommended_action(self) -> dict[str, Any] | None:
        """The policy's ``{"action", "args"}`` for the active decision, or
        ``None`` at a finished run / in plain combat / when the backend can't
        recommend (C# subprocess — no in-process RunManager).

        Drives the "play best" / "play to end" autoplay: the frontend dispatches
        the returned verb through the normal action endpoint, so the branch tree
        and recorded log stay faithfully replayable. Plain combat (playing cards)
        is left to the frontend's MCTS hint; an open in-combat *choice* is
        answered by :meth:`_combat_choice_recommend`.

        Off combat, ``off_combat_brain == "ppo"`` has the greedy PPO snapshot
        answer; when PPO can't decide (net unavailable, unknown surface,
        non-rust backend) the heuristic answer is returned flagged
        ``fallback: True``."""
        if self._active_id is None:
            return None
        rec = self._combat_choice_recommend()
        if rec is not None:
            return rec
        if self.off_combat_brain == "ppo":
            rec = self._ppo_recommend()
            if rec is not None:
                return rec
            out = self._heuristic_recommend()
            if out is not None:
                out["policy"] = "ppo"
                out["fallback"] = True
            return out
        return self._heuristic_recommend()

    def _combat_choice_recommend(self) -> dict[str, Any] | None:
        """The combat MCTS's pick for an open in-combat single-pick choice.

        Neither run-level brain answers in combat — PPO abstains by
        construction (combat was resolved with MCTS during training) and
        ``run_policy.recommend_run_action`` bails on ``phase() == "combat"`` —
        so before this existed "play best" fell through to the frontend's
        first-enabled-option fallback and took card 0. Report
        ``wb-31c2d79f995fe1a6``: a Colorless Potion offered Volley / Salvo /
        Bolas, the panel starred Bolas, and G played Volley.

        The choice is a searched decision everywhere else: ``legal_actions``
        fans an open ``PendingChoice`` into one ``ResolveChoice`` per option, so
        the combat MCTS that plays training and the native ``HeuristicRunAgent``
        autoplay (:meth:`run_mode_step`) expand every branch. That agent's
        ``step`` applies its action to ``rm``, which a *recommendation* must not
        do, so this runs the same search read-only through
        :func:`workbench.choice_ranker.best_action` — at the ranker's default
        config, the one the on-screen ``search %`` chips use, so "play best"
        plays the option the panel stars rather than a second opinion.

        A multi-pick choice (Dredge's "choose up to 3") answers one pick at a
        time: the engine accumulates each index and re-opens the same screen, so
        this returns the best NEXT pick until the search picks FINALIZE
        (``skip_select``). The caller must run it to completion — leaving a
        choice mid-accumulation desyncs the screen's own checkboxes from the
        engine's ``picked`` set.

        ``None`` (the caller falls through to the run-level brains) out of
        combat, on a non-rust backend, when fewer than two options are legal, or
        if the search fails.
        """
        rm = self._native()
        if rm is None:
            return None
        # Imported here, not at module scope: choice_ranker pulls in the
        # advisor + native search, which the C#-backend paths never need.
        from workbench import choice_ranker

        action = choice_ranker.best_action(rm, {})
        if action is None:
            return None
        verb, args = to_workbench_verb(rm, action)
        return {"action": verb, "args": args, "policy": "combat_mcts"}

    def _heuristic_recommend(self) -> dict[str, Any] | None:
        out = self.game.recommended_action() if self.game is not None else None
        if out is None:
            return None
        verb, args = out
        return {"action": verb, "args": args}

    def _ppo_recommend(self) -> dict[str, Any] | None:
        rm = self._native()
        if rm is None:
            return None
        from workbench import ppo_policy

        rec = ppo_policy.recommend(
            rm, character=self.character, ascension=self.ascension, impl=self.ppo_impl
        )
        if rec is None:
            return None
        verb, args = rec
        return {"action": verb, "args": args, "policy": "ppo"}

    def ppo_policy_panel(self) -> dict[str, Any] | None:
        """Masked PPO softmax + V(s) for the active non-combat decision,
        labeled for the UI. ``None`` when the panel doesn't apply (combat /
        game over / no rust RunManager / no active node); an
        ``ppo_available: False`` payload when the decision applies but the
        net can't load."""
        if self._active_id is None:
            return None
        rm = self._native()
        if rm is None:
            return None
        from run_policy import resolve_workbench_verb

        from workbench import ppo_policy

        ev = ppo_policy.evaluate(
            rm, character=self.character, ascension=self.ascension, impl=self.ppo_impl
        )
        if ev is None:
            return None
        parent_named = localization.resolve_state_names(self._active().state, self._loc)
        # Shop card-removal actions need the MASTER deck for naming — the shop
        # state's ``cards`` is the stock, so describe_action would degrade to
        # "Select card 16". Fetch the deck once for the whole option list.
        deck = rm.master_deck_detailed() if ev.surface == "shop" else []
        options = []
        for i, native in enumerate(ev.actions):
            resolved = resolve_workbench_verb(rm, native, ev.full_actions)
            if resolved is None:
                # The panel endpoint runs unlocked against the live rm; an action can
                # interleave between evaluate()'s available_actions() read and this
                # resolve, making the action stale — skip it rather than TypeError.
                continue
            verb, args = resolved
            label = None
            if native.get("action") == "shop_remove_card":
                label = ppo_policy.shop_remove_label(deck, int(native.get("deck_idx", -1)))
            if label is None:
                try:
                    label = render.describe_action(
                        parent_named, {"cmd": "action", "action": verb, "args": args}
                    )
                except Exception:
                    label = verb
            options.append(
                {
                    "action": verb,
                    "args": args,
                    "label": label,
                    "prob": None if ev.probs is None else ev.probs[i],
                    "is_best": ev.chosen == i,
                }
            )
        return {
            "ppo_available": ev.probs is not None,
            "impl": self.ppo_impl,
            "surface": ev.surface,
            "value": ev.value,
            "options": options,
            "error": ev.error,
        }

    def search_tree(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Full MCTS search tree for the active combat state (None off-combat).

        Uses the active node's authoritative recorded state (like
        :meth:`hint`), not a fresh ``get_state()`` query.
        """
        node = self._active()
        if node.state.get("decision") != "combat_play":
            return None
        cs = self._native_combat_state()
        if cs is not None:
            return reasoning.search_tree_native(cs, config)
        named = localization.resolve_state_names(node.state, self._loc)
        return reasoning.search_tree(named, config)

    def combat_boards(
        self, node_ids: Sequence[int] | None = None
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
        """Engine wire dicts for the named branch nodes, plus what was skipped.

        The input to a SimLab sweep launched from the workbench: naming several
        nodes turns "which line was better" from two single playouts into two
        distributions.

        Collecting a node's board means standing on it, so this seeks to each in
        turn and **restores the active node before returning** — launching a
        sweep must not move the user. Nodes that hold no fight (a shop, a map
        choice, a finished combat) are reported in the second return value
        rather than dropped: a silently shorter sweep reads as a complete
        comparison.

        Args:
            node_ids: Branch nodes to collect. ``None`` or empty means the
                active node alone.

        Returns:
            ``(boards, skipped)`` where ``boards`` is ``(label, wire)`` pairs,
            labelled by node so two branches with the same move text stay
            distinct.

        Raises:
            RuntimeError: if no run is active.
        """
        import sts2_rs

        if self.game is None or self._root_id is None:
            raise RuntimeError("no live run")
        active = self._active_id
        targets = list(node_ids or ([active] if active is not None else []))
        boards: list[tuple[str, dict[str, Any]]] = []
        skipped: list[str] = []
        try:
            for node_id in targets:
                label = self.node_label(node_id)
                if node_id != self._active_id:
                    try:
                        self.seek(node_id)
                    except (KeyError, RuntimeError) as exc:
                        skipped.append(f"{label}: {exc}")
                        continue
                cs = self._native_combat_state()
                if cs is None:
                    skipped.append(f"{label}: not a combat position")
                    continue
                boards.append((label, sts2_rs.state_to_dict(cs)))
        finally:
            if active is not None and self._active_id != active:
                try:
                    self.seek(active)
                except (KeyError, RuntimeError):
                    # Best-effort: raising from `finally` would REPLACE whatever
                    # sent us here, and the original failure is the one worth
                    # reading. The user is left on the last node collected.
                    _log.exception("could not restore node %s after collecting boards", active)
        return boards, skipped

    def node_label(self, node_id: int) -> str:
        """A stable, collision-free display label for one branch node.

        Prefixed with the node id because callers use it as a bucket key and
        two branches can easily produce the same move text — merging those into
        one bucket is exactly what a counterfactual comparison must not do.
        """
        node = self._nodes.get(node_id)
        text = (getattr(node, "label", "") or "start").replace("/", "-").strip()
        return f"{node_id}: {text}"[:80]

    def active_node_id(self) -> int | None:
        """The node the session is standing on, or ``None`` before ``start``."""
        return self._active_id

    def import_notes(self) -> list[str]:
        """What an imported position could not reproduce; empty otherwise."""
        return self.game.import_notes() if self.game is not None else []

    def _native_combat_state(self) -> Any:
        """The backend's native combat state if it has one, else ``None``
        (hints then fall back to the display-name adapter)."""
        return self.game.combat_state() if self.game is not None else None

    # -- named saves (active line only) ------------------------------------
    def _progress(self, path: list[_Node]) -> dict[str, Any]:
        """act/floor/hp/max_hp for the leaf, carrying floor/act forward.

        ``floor``/``act`` appear only on some states (map_select, game_over), so
        we carry the last-seen values along *path*; ``hp``/``max_hp`` come from
        the leaf's player block (``None`` when unavailable, e.g. pre-combat)."""
        floor = act = None
        for node in path:
            if node.state.get("floor") is not None:
                floor = node.state.get("floor")
            if node.state.get("act") is not None:
                act = node.state.get("act")
        player = path[-1].state.get("player") or {}
        return {
            "act": act,
            "floor": floor,
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
        }

    def active_line(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return ``(line, progress)`` for the active path (root → active leaf).

        ``line`` is the ordered ``{"action", "args"}`` pairs producing the leaf
        (root excluded); ``progress`` is leaf context for save-list display.
        """
        if self._active_id is None:
            raise RuntimeError("no live run")
        path = self._path_to(self._active_id)
        line = [{"action": n.action, "args": dict(n.args)} for n in path[1:]]
        return line, self._progress(path)

    def active_node_values(self) -> list[float | None]:
        """PPO critic values along the active path, root first.

        Aligned to :meth:`active_line`: index 0 is the root and index ``i + 1``
        the node ``line[i]`` produced. A save carries these so a restore can
        stamp them back on (:meth:`_value_for_new_node`) instead of re-running the
        net once per non-combat decision — the values are identical either way,
        since replay reproduces the same states, but recomputing them is what
        made restoring a long run take minutes.
        """
        if self._active_id is None:
            raise RuntimeError("no live run")
        return [n.value for n in self._path_to(self._active_id)]

    def _apply_native(self, args: dict[str, Any]) -> dict[str, Any]:
        """Replay one recorded NATIVE agent action, appending a branch node.

        The ``apply()`` counterpart for :data:`NATIVE_ACTION` nodes: their
        ``args`` are a raw Rust ``RunAction`` dict that the workbench verb layer
        cannot express, so they go straight to ``rm.take_action`` (the same
        escape hatch :meth:`seek` uses). Returns ``apply()``'s result shape.
        """
        rm = self._require_native("replaying a native agent action")
        parent = self._active()
        parent_named = localization.resolve_state_names(parent.state, self._loc)
        try:
            wb_verb, wb_args = to_workbench_verb(rm, args)
            label = render.describe_action(
                parent_named, {"cmd": "action", "action": wb_verb, "args": wb_args}
            )
        except Exception:
            label = str(args.get("action", "?"))
        marks = self._mark_display_logs()
        result = rm.take_action(dict(args))
        if result not in ("advanced", "run_over"):
            return {
                "state": self._map_state(self.game.get_state()),
                "error": f"take_action returned {result!r}",
                "label": label,
                "details": [],
            }
        raw = self.game.get_state()
        play_lines, combat_rows = self._slice_display_logs(marks)
        details = play_lines + diff_states(
            parent_named, localization.resolve_state_names(raw, self._loc)
        )
        state = self._map_state(raw)
        node = self._add_node(
            parent.id,
            state,
            action=NATIVE_ACTION,
            args=dict(args),
            label=label,
            details=details,
            combat_log=combat_rows,
        )
        self._active_id = node.id
        return {"state": state, "error": None, "label": label, "details": details}

    def load_line(
        self,
        character: str,
        seed: str,
        ascension: int,
        line: list[dict[str, Any]],
        *,
        run_mode: bool = False,
        act1: str = "overgrowth",
        node_values: list[float | None] | None = None,
    ) -> dict[str, Any]:
        """Start a fresh run and replay *line*, rebuilding the branch tree.

        ``run_mode`` MUST match how the saved run was originally started, for
        the same reason :meth:`seek` re-derives it: run-mode runs are built from
        an INTEGER seed (``start_faithful``) and interactive ones from a TEXT
        seed (``string_seed``), and the two hash to different runs. Restoring a
        run-mode line through the interactive constructor therefore replays it
        against a completely different game and the engine rejects the first
        action whose phase no longer matches — historically surfacing as a
        mystery ``invalid_for_phase`` several actions in (GH: workbench save
        of a "play to end" run could never be loaded).

        ``act1`` MUST likewise match how the run was recorded (GH #544). The
        two Act-1 models are entirely different acts — different boss, encounter
        pool and event pool — so restoring under the wrong one diverges exactly
        as a wrong ``run_mode`` does. Saves written before the field existed
        default to ``"overgrowth"``, which is what they were played on.

        ``node_values`` is the saving session's :meth:`active_node_values` —
        the PPO critic values it already computed for this exact line. Passing
        them replays them onto the rebuilt nodes instead of re-running the net
        at every non-combat decision, which is the bulk of a long restore's
        cost. Saves written before the field existed pass ``None`` and the
        values are recomputed as before.

        Returns the active leaf state. Raises ``RuntimeError`` if any replayed
        action is rejected by the engine (the line diverged from a valid run).
        """
        self._restore_values = node_values
        self._restore_idx = 0
        try:
            return self._load_line(character, seed, ascension, line, run_mode=run_mode, act1=act1)
        finally:
            # Only the restore replays values; everything after it — including
            # a fork off a restored node — must compute its own.
            self._restore_values = None

    def _load_line(
        self,
        character: str,
        seed: str,
        ascension: int,
        line: list[dict[str, Any]],
        *,
        run_mode: bool,
        act1: str,
    ) -> dict[str, Any]:
        """:meth:`load_line`'s replay loop, minus the recorded-value bookkeeping."""
        if run_mode:
            try:
                seed_int = int(seed)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"run-mode save carries a non-numeric seed {seed!r}; cannot restore it"
                ) from exc
            state = self.start_run_mode(character, seed_int, ascension=ascension)
        else:
            state = self.start(character, seed, ascension, act1)
        for i, entry in enumerate(line):
            action = entry["action"]
            args = entry.get("args") or {}
            if action == NATIVE_ACTION:
                result = self._apply_native(args)
            else:
                result = self.apply(action, args)
            if result["error"]:
                # Name the position AND what the engine was actually doing —
                # "rejected at step 3 of 454, engine is at card_reward" is
                # debuggable; the bare engine message is not.
                raise RuntimeError(
                    f"replay diverged at step {i + 1}/{len(line)} "
                    f"(action {action!r}): {result['error']}; "
                    f"engine is at {result['state'].get('decision')!r}"
                )
            state = result["state"]
        return state

    def discard(self) -> None:
        """Tear the session down WITHOUT recording a run log.

        :meth:`quit` is *the player ending a run*: it writes the JSONL the
        replay viewer reads. A restore that diverged half-way was never a run,
        so its partial line must not land in the logs dir looking like one.
        """
        if self._slog:
            self._slog.close()
            self._slog = None
        if self.game is not None:
            self.game.close()
        self.game = None
        self._active_id = None

    def quit(self) -> dict[str, Any]:
        """Close the game and write the **active path** to a fresh JSONL log.

        Branch metadata is workbench-only and not persisted (spec v1); the log
        records the final active line so the finished run re-opens in the replay
        viewer.
        """
        log_path: str | None = None
        if self.game is not None and self._active_id is not None:
            logger = GameLogger(
                self.character,
                self.seed,
                enabled=True,
                log_dir=self.logs_dir,
                ascension=self.ascension,
                # Run identity the replay viewer needs to REBUILD this run
                # ("play from here" on a recorded run, /api/live/from_replay).
                # A line only replays through the same constructor, act model
                # and engine that recorded it — exactly the fields a named
                # save carries (``workbench.saves``); without them a recorded
                # run can be watched but never resumed.
                extra={
                    "run_mode": bool(self._run_mode),
                    "act1": self.act1,
                    "backend": self.backend,
                    # GH #801: the build is part of the identity (a line only
                    # replays under the rules that recorded it). Always
                    # written, so a log names its build; a reader treats the
                    # missing key (an older v0.107.1 log) as v0.107.1.
                    "game_version": self.game_version,
                },
            )
            path = self._path_to(self._active_id)

            def _log_wpa(i: int) -> None:
                # Persist the per-decision PPO value (+ the action taken here,
                # for review labels) so the replay viewer's WPA review works
                # without a net — the rendered state can't rebuild the
                # observation post-hoc.
                if path[i].value is not None:
                    taken = path[i + 1].label if i + 1 < len(path) else None
                    logger.log_wpa({"value": path[i].value, "taken_label": taken})

            logger.log_state(path[0].state)
            _log_wpa(0)
            for j, node in enumerate(path[1:], start=1):
                # Persist the action's args in the NATIVE engine shape (the same
                # form ``workbench.record`` logs), not the raw workbench-verb args
                # this session applies. The replay pipeline — notably the regret
                # review's ``action_stable_key`` — keys off ``args["action"]``; the
                # verb args lack it, so an ended run would otherwise replay with
                # every combat decision unresolved ("0 decisions · clean run").
                # The display label and the replayable (verb, args) ride along
                # beside it — see the two comments below.
                logger.log_action(
                    {
                        "cmd": "action",
                        "action": node.action,
                        "args": _native_action(node.action, node.args),
                        # The label the live log showed. ``node.action`` is the
                        # sentinel ``__agent_native__`` for every autoplayed
                        # step, which ``render.describe_action`` cannot name
                        # — so without this the replay's history rendered a
                        # wall of "__agent_native__" where the live one had
                        # read "Play Strike → Twig Slime".
                        "label": node.label,
                        # The action in its REPLAYABLE form: the verb + args
                        # exactly as ``load_line`` re-applies them. ``args``
                        # above is lossily normalized to the native shape the
                        # regret review keys off (verb args lose ``action``;
                        # a native RunAction loses ``hand_idx``), so it cannot
                        # drive a restore. Mirrors :meth:`active_line`.
                        "replay": {
                            "action": node.action,
                            "args": _json_safe(dict(node.args)),
                        },
                    }
                )
                # The root distribution of the search that CHOSE this action,
                # attached to the decision it answered (`log_mcts` shares the
                # step number of the preceding `log_state`, which is the
                # parent's). Without it a replay of this run had no recorded
                # reasoning at all, so the regret review re-searched every
                # position with a different referee and reported the agent
                # blundering against itself. Rows are already labelled and
                # win-prob'd — `TimelineStore.load_jsonl` reads them straight
                # through.
                agent_scores = self._agent_scores(path[j - 1], node)
                if agent_scores:
                    logger.log_mcts(
                        {
                            "source": "agent",
                            "total_visits": sum(r["visits"] for r in agent_scores),
                            # `action_label` is `workbench.report`'s key and it
                            # indexes it unguarded; `action` is the review's.
                            # Same string, written under both so the Markdown
                            # report of a workbench-played run doesn't KeyError.
                            "root_outcomes": [
                                {**r, "action_label": r["action"]} for r in agent_scores
                            ],
                        }
                    )
                # This action's combat-log slice, so the replay timeline keeps
                # the per-event granularity the live log had (display rows —
                # localization is already baked into the rest of the record).
                if node.combat_log:
                    logger.log_combat_log(node.combat_log)
                logger.log_state(node.state)
                _log_wpa(j)
            # Terminal summary row. Without it `workbench.aggregate._parse_run`
            # rejects the file, so a hand-played run never appeared on the
            # "MCTS Runs" page or in its aggregate — the page read as "you have
            # recorded nothing" while the log sat right beside the recorder's.
            leaf = path[-1].state
            progress = self._progress(path)
            logger.log_summary(
                {
                    "final_won": bool(leaf.get("victory")),
                    # `round` is combat-scoped, so it is None for a run
                    # abandoned off-combat; the floor is the run-level analogue.
                    "final_turn": leaf.get("round"),
                    "final_floor": progress["floor"],
                    "final_act": progress["act"],
                    "final_player_hp": progress["hp"],
                    "final_player_max_hp": progress["max_hp"],
                }
            )
            log_path = logger.path
            logger.close()
        if self._slog:
            self._slog.line("END", f"active node {self._active_id}, game log -> {log_path}")
            self._slog.close()
            self._slog = None
        if self.game is not None:
            self.game.close()
        self.game = None
        self._active_id = None
        return {"ok": True, "log_path": log_path}
