"""``RealGame``: play the live Slay the Spire 2 process through the exact
headless ``Sts2Headless`` decision-point schema.

Real-game bridge phase 2 (see
``../docs/plans/2026-07-10-real-game-bridge.md``). :class:`RealGame` speaks
the SpireMentor wire protocol (``spire-mentor-mod/PROTOCOL.md``) via
:class:`realgame_bridge.RealGameBridgeClient` and translates every message
into *exactly* the dict shape ``src/Sts2Headless/RunSimulator.cs`` emits for
each ``decision`` (``game_over``, ``bundle_select``, ``card_reward``,
``card_select``, ``map_select``, ``combat_play``, ``event_choice``,
``rest_site``, ``shop``) -- so every existing driver that talks to
``sts2_client.Game`` (``sts2_mcp.py``, the ``sts2-play`` skill, batch
runners, loggers, ``sts2_hint``'s MCTS recommender) plays the real game
**unchanged**.

Headless conventions reproduced here in Python (none of this exists on the
wire -- the mod stays deliberately dumb, see the plan's "Design rule"):

* **Treasure auto-resolve.** The bridge's ``treasure`` message still asks
  which relic to take (mirroring ``NTreasureRoomHandler``), even though a
  treasure room only ever offers exactly one. Headless's own
  ``TreasureState``/``CollectTreasureRelic`` always takes ``relics[0]`` and
  never surfaces a decision at all -- :meth:`RealGame._auto_treasure` does
  the same: send ``{"action":"choose","index":0}`` and keep going.
* **Reward auto-collect.** ``reward_screen`` lists every reward button
  (gold/potion/relic/card) plus an optional ``proceed``. Headless
  auto-collects ``GoldReward``/``PotionReward``/``RelicReward`` immediately
  and only THEN surfaces a ``CardReward`` (if any) as its own
  ``card_reward`` decision (``DetectPostCombatState``: rewards are
  partitioned and every non-card reward is ``SelectUnsynchronized()``'d
  before ``CardRewardState`` is ever returned). :meth:`RealGame._auto_reward_screen`
  mirrors that ordering: it clicks any collectible gold/potion/relic
  button first, and only clicks ``CardReward`` (which reliably opens the
  bridge's *separate* ``card_reward`` message, ``SimCardRewardScreenHandler``)
  once no other collectible option remains, else ``proceed`` -- looping via
  the outer :meth:`RealGame._next_decision` dispatcher until a
  non-``reward_screen`` message arrives.
* **Boss relic -> event_choice.** Headless auto-enters the act-start
  Ancient node and surfaces the boss-relic pick through the ordinary
  ``EventChoiceState`` path (see ``DetectPostCombatState``'s "Auto-entering
  act-start Ancient node" comment). The bridge instead has a dedicated
  ``boss_relic`` message type (shape-identical to ``treasure``, but a REAL
  3-way choice, never auto-resolved) -- :meth:`RealGame._translate_boss_relic`
  reshapes it into the same ``event_choice`` dict headless would emit.
* **Alive-enemy re-indexing.** Combat translation is fully delegated to
  :mod:`realgame_adapter` (``to_headless_combat_play``, NOT modified here
  per the phase-2 task split) -- the bridge already only lists alive
  enemies, so no re-indexing work is needed on this side; enemy ``index``
  is positional exactly like headless's own ``Where(IsAlive).Select((e,i)``.

Action translation (headless verb -> bridge verb) is the mirror image: the
driver keeps calling the same ``sts2_client.Game``-shaped methods
(``play_card``, ``choose_option``, ``select_cards``, ...); each one is
translated to the matching bridge action (``play``/``potion``/``choose``/
..., see ``spire-mentor-mod/PROTOCOL.md`` section 4) and sent. ``shop`` and
``map_select`` need extra bookkeeping because their headless action
addressing (``buy_card(card_index)`` into a *filtered* per-kind list;
``select_map_node(col, row)``) does not match the bridge's own flat
option-index / node-index addressing -- :attr:`RealGame._shop_map` and
:attr:`RealGame._map_nodes` cache the mapping from the most recently
translated decision.

Error handling: a bridge ``error`` notification (PROTOCOL.md section 3a) is
always immediately followed by a fresh re-send of the decision it rejected.
:meth:`RealGame._next_decision` treats ``error`` as transparent: it logs the
rejection and loops to receive (and translate) that guaranteed re-send --
this keeps the client's ``request_id``/``state_id`` bookkeeping correct for
the *next* action without ever leaving the socket in a state that would
desync the blocking protocol.

Parity residuals (bridge has no equivalent field, so the headless key is
included but ``None``/best-effort -- see the phase-2 report for the full
exhaustive list): ``context.act_name``/``context.boss`` (no localized act
name or boss identity on the wire outside the wiki), per-card/relic/potion
``stats``/``after_upgrade`` (dynamic-var previews are a headless-only
synthetic computation), ``card_reward.gold_earned`` (best-effort via a
gold snapshot taken at the last ``combat_action`` before the reward
screen), event/boss-relic ``event_name`` (raw ``event_id``/a synthetic
constant, never the localized title), and arbitrary string run seeds
(the bridge's ``start_run`` only accepts an integer seed via
``DebugSeedOverride``).

Wire-shape residuals still open after the 2026-07-11 parity sweep
(``gate_seed12345_v3.json``; everything else in that report's
``player.*``/``player_powers`` mismatch set -- gold, name, potion shape,
relic id/name/description presence, deck id/cost/type/upgraded, and
combat_play's ``player``/``player_powers`` field presence -- is now fixed):

* ``player.relics[].description`` -- always the bridge's *rendered*
  description (with the live value substituted in, still wrapped in the
  game's BBCode-ish color tags, e.g. ``"heal [green]6[/green] HP."``),
  never headless's raw ``{Var}`` *template* string (``"heal {Heal} HP."``).
  The bridge only ever sends rendered text -- there is no client-side way
  to recover the original template.
* ``player.relics[].vars`` -- best-effort only: derived from the bridge
  relic's ``counter``/``stack_count`` fields the same way
  :mod:`realgame_adapter`'s own relic translator does. A relic whose
  dynamic var isn't exposed as ``counter`` (e.g. Burning Blood's ``Heal``)
  keeps ``vars: None`` here even though headless resolves a real
  ``{Heal: 6}`` dict for it -- the bridge simply doesn't expose the named
  var, only the rendered description it produced from it. Field
  *presence* still matches headless's own null-when-empty convention.
* ``player.deck[].stats``/``player.deck[].after_upgrade`` -- always
  ``None`` (see the ``stats``/``after_upgrade`` residual above; deck cards
  are no exception).
* ``player.deck[].tinker_time_rider`` -- always ``None``; no bridge field
  carries the Necrobinder Mad Science "will this rider fire" preview.
* Deck-content itself (which cards are in the deck, and ``deck_size``) is
  NOT a wire-shape residual -- when it differs from headless it reflects
  the two engines' *run states* having actually diverged (e.g. a
  Neow's-Fury pickup difference), which this module cannot and should not
  paper over.
"""

from __future__ import annotations

import logging
from typing import Any

from realgame_adapter import character_display_name, to_headless_combat_play
from realgame_bridge import DEFAULT_HOST, DEFAULT_PORT, RealGameBridgeClient

logger = logging.getLogger(__name__)

__all__ = ["RealGame", "potion_swap_group", "reward_screen_choice"]


def reward_screen_choice(options: list[dict[str, Any]]) -> int | None:
    """Pick the button to click on a bridge ``reward_screen``.

    Canonical claim order, matching headless's own ``DetectPostCombatState``:
    auto-collect every non-card reward (gold/potion/relic) first, then open the
    ``CardReward`` (which the bridge re-surfaces as a real ``card_reward``
    decision -- so the *card* choice stays with the run policy, not this
    helper), and only ``proceed`` once nothing is left to claim. Returns the
    option ``index`` to choose, or ``None`` when no option is actionable.

    Shared by :meth:`RealGame._auto_reward_screen` and the realgame drivers so
    every backend collects post-combat rewards through identical logic.

    A potion reward that finds a full belt (GH #1069) comes as a
    :func:`potion_swap_group` instead of a ``pick_reward``. It is answered
    after the other rewards and before the card, the order the engines raise
    their ``potion_swap`` decision in, and this default answer is the group's
    decline: leave the potion, as the engines do with the swap off. A caller
    that lets a policy decide checks whether the returned option's ``action``
    is ``potion_swap`` and asks it instead.
    """
    collect_idx = next(
        (
            o.get("index")
            for o in options
            if o.get("action") == "pick_reward" and o.get("id") != "CardReward"
        ),
        None,
    )
    if collect_idx is not None:
        return collect_idx
    decline_idx = next(
        (o.get("index") for o in potion_swap_group(options) if o.get("discard_slot") is None),
        None,
    )
    if decline_idx is not None:
        return decline_idx
    card_idx = next((o.get("index") for o in options if o.get("id") == "CardReward"), None)
    if card_idx is not None:
        return card_idx
    return next((o.get("index") for o in options if o.get("action") == "proceed"), None)


def potion_swap_group(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first full-belt potion reward's ``potion_swap`` options, decline first.

    The mod offers every potion reward that only a full belt blocks as a group
    of ``reward_screen`` options sharing one ``reward_button``: a decline
    (``discard_slot`` null) and one option per belt slot (``discard_slot`` k:
    discard that potion, then claim the reward). The group mirrors the
    engines' ``potion_swap`` decision. Empty when nothing is blocked.
    """
    swaps = [o for o in options if o.get("action") == "potion_swap"]
    if not swaps:
        return []
    first = min(int(o.get("reward_button") or 0) for o in swaps)
    group = [o for o in swaps if int(o.get("reward_button") or 0) == first]
    return sorted(group, key=lambda o: -1 if o.get("discard_slot") is None else o["discard_slot"])


class RealGame:
    """Live-game backend implementing the ``sts2_client.Game`` surface.

    Owns one :class:`~realgame_bridge.RealGameBridgeClient` session and
    translates every message it receives into the exact headless
    decision-point schema (see module docstring). Construct, then call
    :meth:`connect` (or use as a context manager) before :meth:`start`.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        character: str = "Ironclad",
        seed: str | int | None = "test",
        ascension: int = 0,
        timeout: float = 60.0,
        reconnect_attempts: int = 30,
        reconnect_delay: float = 2.0,
    ) -> None:
        """Configure a RealGame session (does not connect yet).

        Args:
            host: SpireMentor mod TCP host.
            port: SpireMentor mod TCP port.
            character: Default character for :meth:`start` when not
                overridden per-call, e.g. ``"Ironclad"``.
            seed: Default run seed for :meth:`start`. Only an int (or a
                numeric string) survives translation to the bridge's
                ``start_run`` (int-or-null); a non-numeric string is
                logged and dropped (fresh random seed) -- see the module
                docstring's parity residuals.
            ascension: Default ascension level (0-10) for :meth:`start`.
            timeout: Socket receive timeout in seconds (connection-health
                guard only -- the mod itself waits indefinitely for a
                decision reply).
            reconnect_attempts: TCP connect retry budget.
            reconnect_delay: Seconds between connect retries.
        """
        self.character = character
        self.seed = seed
        self.ascension = ascension
        self.client = RealGameBridgeClient(
            host=host,
            port=port,
            timeout=timeout,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
        )
        self.step = 0
        self.last_raw: dict[str, Any] | None = None

        self._connected = False
        self._terminal_decision: dict[str, Any] | None = None
        self._last_decision_type: str | None = None

        # Bookkeeping needed to translate a headless-shaped action back to
        # the bridge's own addressing for the currently pending decision.
        self._map_nodes: list[dict[str, Any]] = []
        self._shop_map: dict[str, Any] = {
            "cards": [],
            "relics": [],
            "potions": [],
            "leave": 0,
            "remove_card": None,
        }
        self._gold_before_reward: int | None = None
        # GH #1069: surface full-belt potion rewards as `potion_swap`
        # decisions (set by :meth:`start`), and the pending group's bridge
        # option index per `discard_slot` (None = decline).
        self._interactive_potion_swap = False
        self._swap_options: dict[int | None, int] = {}
        # Belt slot of each held potion, in host numbering (GH #1287).
        self._potion_slots: list[int | None] = []
        # Card rewards skipped on the current rewards screen (GH #1287).
        self._skipped_card_rewards = 0

    # ── Connection lifecycle ────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the TCP connection to the SpireMentor mod. Idempotent."""
        if not self._connected:
            self.client.connect()
            self._connected = True

    def close(self) -> None:
        """Close the connection, if open. Safe to call multiple times."""
        if self._connected:
            self.client.disconnect()
            self._connected = False

    def __enter__(self) -> RealGame:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # ── Lifecycle: start ─────────────────────────────────────────────────────

    def start(
        self,
        character: str | None = None,
        seed: str | int | None = None,
        ascension: int | None = None,
        lang: str = "en",
        interactive_potion_swap: bool = False,
        require_new_run: bool = False,
    ) -> dict[str, Any]:
        """Start (or attach to) a run and return the first decision.

        Mirrors ``sts2_client.Game.start()``'s signature exactly so drivers
        (``sts2_mcp.start_run``) work unchanged.

        If the mod's ``client_run_config`` flag is enabled, performs the
        ``run_config`` -> ``start_run`` handshake (PROTOCOL.md section 3/4)
        with the given (or constructor-default) character/seed/ascension.
        If it is disabled, the run has already auto-started from the mod's
        own ``SpireMentor.conf`` defaults by the time we connect -- any
        requested character/seed/ascension is logged and ignored (the mod
        offers no way to override it after the fact).

        Args:
            character: Character id/display name, e.g. ``"Ironclad"``.
            seed: Run seed. Only an int or numeric string round-trips to
                the bridge (``DebugSeedOverride`` is int-only); anything
                else falls back to a fresh random seed.
            ascension: Ascension level (0-10).
            lang: Accepted for interface parity with ``sts2_client.Game``;
                the bridge has no language setting (its Card/Relic/Potion
                text is always whatever the live game's active locale is).
            interactive_potion_swap: GH #1069, as in ``sts2_client.Game``. A
                potion reward that finds a full belt becomes a
                ``potion_swap`` decision (answer with the ``potion_swap``
                action: ``discard_slot`` None declines, k discards that held
                potion and claims the reward). Off, it is left on the reward
                screen, which is what the engines do with the swap off.
            require_new_run: Raise instead of attaching when the mod is not
                waiting for a ``start_run`` (a run is already in progress, or
                ``client_run_config`` is off). A caller that compares the run
                with another world needs the seed it asked for (GH #1287).
        """
        self._interactive_potion_swap = interactive_potion_swap
        if lang != "en":
            logger.warning(
                "RealGame.start(): lang=%r ignored (bridge has no language switch)", lang
            )
        character = character if character is not None else self.character
        ascension = ascension if ascension is not None else self.ascension
        seed = seed if seed is not None else self.seed

        self.connect()
        msg = self.client.receive_state()
        while msg.get("type") in _NOTIFICATION_TYPES:
            msg = self.client.receive_state()

        if require_new_run and msg.get("type") != "run_config":
            raise RuntimeError(
                f"RealGame.start(): the mod is not waiting for a new run (got {msg.get('type')!r})"
            )
        if msg.get("type") == "run_config":
            seed_int = _coerce_int_seed(seed)
            char_id = character.upper()
            self.client.start_run(char_id, ascension=ascension, seed=seed_int)
            msg = self.client.receive_state()
            # A rejected start_run gets a structured error then a fresh
            # run_config re-send (PROTOCOL.md). Retry a bounded number of
            # times rather than looping forever on a persistently-invalid
            # request.
            for _ in range(5):
                if msg.get("type") != "error":
                    break
                logger.warning("start_run rejected: %s: %s", msg.get("error"), msg.get("detail"))
                msg = self.client.receive_state()
                if msg.get("type") == "run_config":
                    self.client.start_run(char_id, ascension=ascension, seed=seed_int)
                    msg = self.client.receive_state()
        else:
            logger.warning(
                "RealGame.start(): mod's client_run_config is off; the run already "
                "auto-started from SpireMentor.conf defaults -- ignoring requested "
                "character=%r seed=%r ascension=%r",
                character,
                seed,
                ascension,
            )

        return self._next_decision(msg)

    # ── Low-level dispatch (sts2_client.Game.send/act parity) ──────────────

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a raw headless-style command dict.

        Supports the subset of ``sts2_client.Game.send`` commands that make
        sense against a live game: ``start_run``, ``action``, ``quit``.
        """
        cmd_type = cmd.get("cmd")
        if cmd_type == "start_run":
            return self.start(
                character=cmd.get("character"),
                seed=cmd.get("seed"),
                ascension=cmd.get("ascension"),
                lang=cmd.get("lang", "en"),
                interactive_potion_swap=bool(cmd.get("interactive_potion_swap")),
            )
        if cmd_type == "action":
            return self.act(cmd.get("action"), **(cmd.get("args") or {}))
        if cmd_type == "quit":
            self.close()
            return {"type": "ok"}
        return {"type": "error", "message": f"RealGame.send: unsupported cmd {cmd_type!r}"}

    def act(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch one of the 17 headless action verbs by name."""
        handler = _ACTION_DISPATCH.get(action)
        if handler is None:
            return {"type": "error", "message": f"Unknown action: {action}"}
        return handler(self, kwargs)

    # ── Typed action helpers (sts2_client.Game parity, one per ACTIONS verb) ─

    def select_map_node(self, col: int, row: int) -> dict[str, Any]:
        idx = next(
            (
                n.get("index")
                for n in self._map_nodes
                if n.get("col") == col and n.get("row") == row
            ),
            None,
        )
        if idx is None:
            return {
                "type": "error",
                "message": f"select_map_node: ({col},{row}) is not a currently reachable node",
            }
        return self._send({"action": "choose", "index": idx})

    def play_card(self, card_index: int, target_index: int | None = None) -> dict[str, Any]:
        action: dict[str, Any] = {"action": "play", "card_index": card_index}
        if target_index is not None:
            action["target_index"] = target_index
        return self._send(action)

    def end_turn(self) -> dict[str, Any]:
        return self._send({"action": "end_turn"})

    def choose_option(self, option_index: int) -> dict[str, Any]:
        """Reply to a ``rest_site``/``event_choice`` decision.

        The bridge's generic ``choose`` verb handles both room types
        identically -- no dispatch needed here (unlike headless, which
        routes ``choose_option`` server-side by ``CurrentRoom`` type).
        """
        return self._send({"action": "choose", "index": option_index})

    def select_card_reward(self, card_index: int) -> dict[str, Any]:
        return self._send({"action": "choose", "index": card_index})

    def skip_card_reward(self) -> dict[str, Any]:
        # The skipped reward stays on the rewards screen, and the mod asks
        # about that screen again: remember it, or it is reopened forever.
        self._skipped_card_rewards += 1
        return self._send({"action": "skip"})

    def buy_card(self, card_index: int) -> dict[str, Any]:
        return self._shop_buy("cards", card_index)

    def buy_relic(self, relic_index: int) -> dict[str, Any]:
        return self._shop_buy("relics", relic_index)

    def buy_potion(self, potion_index: int) -> dict[str, Any]:
        return self._shop_buy("potions", potion_index)

    def remove_card(self) -> dict[str, Any]:
        idx = self._shop_map.get("remove_card")
        if idx is None:
            return {"type": "error", "message": "remove_card: no card removal stocked in this shop"}
        return self._send({"action": "choose", "index": idx})

    def select_bundle(self, bundle_index: int) -> dict[str, Any]:
        return self._send({"action": "choose", "index": bundle_index})

    def select_cards(self, indices: str) -> dict[str, Any]:
        idx_list = [
            int(s)
            for s in (part.strip() for part in str(indices).split(","))
            if s.lstrip("-").isdigit()
        ]
        return self._send({"action": "choose", "indices": idx_list})

    def skip_select(self) -> dict[str, Any]:
        return self._send({"action": "skip"})

    def _potion_slot(self, potion_index: int) -> int:
        """Belt slot of the host-numbered potion (position among held potions)."""
        if 0 <= potion_index < len(self._potion_slots):
            slot = self._potion_slots[potion_index]
            if slot is not None:
                return int(slot)
        return potion_index

    def use_potion(self, potion_index: int, target_index: int | None = None) -> dict[str, Any]:
        action: dict[str, Any] = {"action": "potion", "slot": self._potion_slot(potion_index)}
        if target_index is not None:
            action["target_index"] = target_index
        return self._send(action)

    def discard_potion(self, potion_index: int) -> dict[str, Any]:
        return self._send(
            {"action": "potion", "slot": self._potion_slot(potion_index), "discard": True}
        )

    def leave_room(self) -> dict[str, Any]:
        if self._last_decision_type == "shop":
            return self._send({"action": "choose", "index": self._shop_map.get("leave", 0)})
        return {
            "type": "error",
            "message": (
                "leave_room: only meaningful for shop under RealGame "
                f"(current decision: {self._last_decision_type!r})"
            ),
        }

    def proceed(self) -> dict[str, Any]:
        """Generic 'continue'.

        Headless surfaces ``proceed`` after terminal reward/treasure
        screens; RealGame auto-drives those internally (see
        :meth:`_auto_treasure`/:meth:`_auto_reward_screen`), so a live
        decision that still needs an explicit ``proceed`` should not arise
        in practice. Kept for interface parity; always an error.
        """
        return {
            "type": "error",
            "message": "proceed: not applicable under RealGame (reward/treasure screens auto-advance)",
        }

    # ── Internal: action send + decision-loop ───────────────────────────────

    def _shop_buy(self, kind: str, index: int) -> dict[str, Any]:
        bridge_indices: list[int] = self._shop_map.get(kind) or []
        if index < 0 or index >= len(bridge_indices):
            return {"type": "error", "message": f"invalid {kind[:-1]}_index {index}"}
        if bridge_indices[index] is None:
            # A sold-out or unaffordable slot from the full `stock` list.
            return {"type": "error", "message": f"{kind[:-1]} {index} cannot be bought now"}
        return self._send({"action": "choose", "index": bridge_indices[index]})

    def _send(self, bridge_action: dict[str, Any]) -> dict[str, Any]:
        """Send one bridge action and translate the resulting decision."""
        if self._terminal_decision is not None:
            return self._terminal_decision
        self.step += 1
        self.client.send_action(bridge_action)
        return self._next_decision(self.client.receive_state())

    def _next_decision(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Given an already-received bridge message, resolve/translate it
        into a headless decision dict, auto-handling every convention
        (pong/error/treasure/reward_screen) transparently -- see module
        docstring."""
        while True:
            mtype = msg.get("type")
            #: The bridge message behind the latest decision, for diagnostics.
            self.last_raw = msg

            if mtype == "pong" or mtype in _NOTIFICATION_TYPES:
                msg = self.client.receive_state()
                continue

            if mtype == "error":
                logger.warning(
                    "bridge rejected the last action: %s: %s", msg.get("error"), msg.get("detail")
                )
                # PROTOCOL.md section 3a: an error is always immediately
                # followed by a fresh re-send of the decision it rejected.
                # Draining it here keeps request_id/state_id fresh for
                # whatever the caller sends next.
                msg = self.client.receive_state()
                continue

            if mtype in ("game_over", "run_complete"):
                return self._handle_terminal(msg)

            if mtype == "treasure":
                msg = self._auto_treasure(msg)
                continue

            if mtype == "event" and _only_proceed(msg.get("options") or []):
                # A finished event: headless walks to the map by itself
                # (RunSimulator.EventChoiceState, `localEvent.IsFinished`),
                # the game waits for its Proceed button. Click it.
                self.client.send_action({"action": "choose", "index": msg["options"][0]["index"]})
                msg = self.client.receive_state()
                continue

            if mtype == "reward_screen":
                swap = self._translate_potion_swap(msg) if self._interactive_potion_swap else None
                if swap is not None:
                    self._last_decision_type = "potion_swap"
                    return swap
                msg = self._auto_reward_screen(msg)
                continue

            if mtype == "run_config":
                logger.error("unexpected mid-session run_config: %s", msg)
                return {"type": "error", "message": "unexpected run_config mid-session"}

            method_name = _TRANSLATE_METHOD.get(mtype)
            if method_name is None:
                logger.error("RealGame: unhandled bridge message type %r: %s", mtype, msg)
                return {"type": "error", "message": f"unhandled bridge message type: {mtype!r}"}

            decision = getattr(self, method_name)(msg)
            if _only_ancient_node(decision):
                # Headless enters the act-start Ancient by itself
                # (RunSimulator.DetectPostCombatState, "Auto-entering
                # act-start Ancient node"); the game puts it on the map as
                # the only node. Take it (GH #1287).
                node = decision["choices"][0]
                return self.select_map_node(node.get("col"), node.get("row"))
            if decision.get("type") == "decision":
                self._last_decision_type = decision.get("decision")
                if decision.get("decision") != "card_reward":
                    self._skipped_card_rewards = 0
            return decision

    def _handle_terminal(self, msg: dict[str, Any]) -> dict[str, Any]:
        mtype = msg.get("type")
        if mtype == "game_over":
            decision = self._translate_game_over(msg)
            self._terminal_decision = decision
            # game_over is always followed by run_complete a moment later
            # (both fire-and-forget, no reply expected/accepted for
            # either) -- drain it so the connection is left in a clean
            # state and doesn't desync a caller that keeps polling.
            try:
                follow = self.client.receive_state()
                if follow.get("type") != "run_complete":
                    logger.warning(
                        "expected run_complete after game_over, got %r", follow.get("type")
                    )
            except (ConnectionError, TimeoutError) as e:
                logger.warning("connection lost draining run_complete after game_over: %s", e)
            return decision

        # A bare run_complete with no preceding game_over shouldn't happen
        # per PROTOCOL.md, but build a defeat-shaped terminal decision
        # rather than raising if it ever does (e.g. a session resumed
        # mid-stream after another client already consumed game_over).
        run_state = msg.get("run_state") or {}
        decision = {
            "type": "decision",
            "decision": "game_over",
            "context": self._context(msg, room_type=None),
            "victory": msg.get("result") == "victory",
            "player": self._player_summary(run_state),
            "act": msg.get("act", 1),
            "floor": msg.get("floor", 0),
        }
        self._terminal_decision = decision
        return decision

    # ── Auto-resolved screens (never surfaced to the caller) ───────────────

    def _auto_treasure(self, _msg: dict[str, Any]) -> dict[str, Any]:
        """Auto-collect the treasure room's (sole) relic -- headless parity."""
        self.client.send_action({"action": "choose", "index": 0})
        return self.client.receive_state()

    def _auto_reward_screen(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Auto-collect gold/potion/relic rewards; open the card reward if any.

        One round trip per call -- the outer :meth:`_next_decision` loop
        re-invokes this for every ``reward_screen`` re-send (the screen
        shrinks by one button per click) until a ``card_reward`` (real
        decision) or the post-``proceed`` next room arrives. Delegates the
        button choice to :func:`reward_screen_choice` (shared with the
        realgame drivers so every backend collects rewards identically).
        """
        options = msg.get("options") or []
        chosen = reward_screen_choice(options)
        cards = [o.get("index") for o in options if o.get("id") == "CardReward"]
        if chosen in cards and self._skipped_card_rewards:
            # A skipped card reward is still listed; open the next one, or
            # leave the screen as a player who skipped does.
            rest = cards[self._skipped_card_rewards :]
            chosen = (
                rest[0]
                if rest
                else next((o.get("index") for o in options if o.get("action") == "proceed"), None)
            )
        if chosen is None:
            logger.error("reward_screen had no actionable option: %s", msg)
            chosen = 0
        self.client.send_action({"action": "choose", "index": chosen})
        return self.client.receive_state()

    def potion_swap(self, discard_slot: int | None = None) -> dict[str, Any]:
        """Answer a ``potion_swap`` decision: None declines, k discards slot k."""
        idx = self._swap_options.get(discard_slot)
        if idx is None:
            return {
                "type": "error",
                "message": f"potion_swap: discard_slot {discard_slot!r} is not offered",
            }
        self._swap_options = {}
        return self._send({"action": "choose", "index": idx})

    # ── Decision translators: bridge message -> headless decision dict ─────

    def _translate_potion_swap(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Headless ``PotionSwapState()``-shaped decision, or None when the
        reward screen's next step is not a swap (see :func:`reward_screen_choice`).
        """
        options = msg.get("options") or []
        chosen = reward_screen_choice(options)
        picked = next((o for o in options if o.get("index") == chosen), None)
        if picked is None or picked.get("action") != "potion_swap":
            return None
        group = potion_swap_group(options)
        self._swap_options = {o.get("discard_slot"): int(o["index"]) for o in group}
        potion = (group[0].get("reward") or {}).get("id")
        return {
            "type": "decision",
            "decision": "potion_swap",
            "context": self._context(msg, room_type=None),
            "potion": potion,
            "options": [
                {"index": i, "discard_slot": o.get("discard_slot"), "label": o.get("label")}
                for i, o in enumerate(group)
            ],
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_combat(self, msg: dict[str, Any]) -> dict[str, Any]:
        run_state = msg.get("run_state") or {}
        gold = run_state.get("gold")
        if isinstance(gold, int):
            # Snapshot the most recent in-combat gold total so a following
            # card_reward can report gold_earned (see its translator).
            self._gold_before_reward = gold
        result = to_headless_combat_play(msg)

        # `to_headless_combat_play` (realgame_adapter.py -- phase-2 combat
        # translation, deliberately not modified here per the module
        # docstring) builds a reduced in-combat ``player`` dict
        # (hp/max_hp/block/relics/potions) tailored to what
        # sts2_hint.adapter needs. Headless's own combat_play reuses the
        # exact same ``PlayerSummary()`` as every other decision
        # (``RunSimulator.cs``'s combat_play dict assigns
        # ``player = PlayerSummary(player)`` just like map_select/shop/etc),
        # which additionally carries gold/name/deck/deck_size and the
        # richer relic/potion shapes -- layer those on top here (patching
        # the adapter's return value, not the adapter itself) so
        # combat_play matches the wire-shape gate the same way every other
        # decision already does via :meth:`_player_summary`. hp/max_hp/
        # block are left exactly as the adapter emitted them (live
        # per-turn combat state, not run_state -- unrelated to this
        # residual sweep). Swapping in `_player_summary`'s relics
        # ({id, name, description, vars, counter, is_wax, is_melted}) and
        # potions ({index, name, description, vars, target_type}, filtered
        # to non-empty) is safe for sts2_hint.adapter: it reads ``name``,
        # ``counter`` and the melt flags off each relic and ``name`` off
        # each potion, so the extra/reshaped fields are ignored, not
        # mismatched.
        full_summary = self._player_summary(run_state)
        player = result.get("player") or {}
        player["gold"] = full_summary["gold"]
        player["name"] = full_summary["name"]
        player["relics"] = full_summary["relics"]
        player["potions"] = full_summary["potions"]
        player["deck_size"] = full_summary["deck_size"]
        player["deck"] = full_summary["deck"]
        result["player"] = player

        # RunSimulator.cs: ``player_powers`` is ``null`` (not ``[]``) when
        # the player has no active powers; the adapter always returns a
        # (possibly empty) list -- normalize to match headless's
        # field-presence convention.
        result["player_powers"] = result.get("player_powers") or None

        return result

    def _translate_map_select(self, msg: dict[str, Any]) -> dict[str, Any]:
        nodes = msg.get("nodes") or []
        self._map_nodes = nodes
        return {
            "type": "decision",
            "decision": "map_select",
            "context": self._context(msg, room_type="Map"),
            "choices": [
                {"col": n.get("col"), "row": n.get("row"), "type": n.get("type")} for n in nodes
            ],
            "player": self._player_summary(msg.get("run_state") or {}),
            "act": msg.get("act", 1),
            "act_name": None,
            "floor": msg.get("floor", 0),
        }

    def _translate_rest_site(self, msg: dict[str, Any]) -> dict[str, Any]:
        options = msg.get("options") or []
        return {
            "type": "decision",
            "decision": "rest_site",
            "context": self._context(msg, room_type="RestSite"),
            "options": [
                {
                    "index": o.get("index"),
                    "option_id": o.get("id"),
                    "name": o.get("label"),
                    "is_enabled": o.get("enabled", True),
                }
                for o in options
            ],
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_shop(self, msg: dict[str, Any]) -> dict[str, Any]:
        if msg.get("stock"):
            return self._translate_shop_stock(msg)
        options = msg.get("options") or []
        cards: list[dict[str, Any]] = []
        relics: list[dict[str, Any]] = []
        potions: list[dict[str, Any]] = []
        shop_map: dict[str, Any] = {
            "cards": [],
            "relics": [],
            "potions": [],
            "leave": 0,
            "remove_card": None,
        }
        for o in options:
            action = o.get("action")
            item = o.get("item") or {}
            if action == "leave_shop":
                shop_map["leave"] = o.get("index", 0)
            elif action == "buy_card":
                shop_map["cards"].append(o.get("index"))
                cards.append(
                    {
                        "index": len(cards),
                        "name": item.get("name") or o.get("label"),
                        "type": item.get("type") or "?",
                        "card_cost": _card_cost(item, 0),
                        "description": item.get("description"),
                        "stats": None,
                        "after_upgrade": None,
                        "cost": o.get("price"),
                        "is_stocked": o.get("enabled", True),
                        # Sent since mod 2026-11-12 (GH #1287); older mods: False.
                        "on_sale": bool(o.get("on_sale", False)),
                    }
                )
            elif action == "buy_relic":
                shop_map["relics"].append(o.get("index"))
                relic_dict = _relic_offer_dict(item, len(relics))
                relic_dict["cost"] = o.get("price")
                relic_dict["is_stocked"] = o.get("enabled", True)
                relics.append(relic_dict)
            elif action == "buy_potion":
                shop_map["potions"].append(o.get("index"))
                potion_dict = _potion_offer_dict(item, len(potions))
                potion_dict["cost"] = o.get("price")
                potion_dict["is_stocked"] = o.get("enabled", True)
                potions.append(potion_dict)
            elif action == "remove_card":
                shop_map["remove_card"] = o.get("index")
        self._shop_map = shop_map
        return {
            "type": "decision",
            "decision": "shop",
            "context": self._context(msg, room_type="Shop"),
            "cards": cards,
            "relics": relics,
            "potions": potions,
            "card_removal_cost": msg.get("purge_cost"),
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_shop_stock(self, msg: dict[str, Any]) -> dict[str, Any]:
        """The shop from the mod's full ``stock`` rows (mod 2026-11-12, GH #1287).

        The host lists every slot, sold out or not, and its buy actions index
        those lists. ``options`` only holds what can be bought now, so it
        cannot be indexed the same way. Each ``stock`` row names the option
        that buys it (``option``); a slot without one cannot be bought, and
        buying it is refused here as the host refuses it.
        """
        cards: list[dict[str, Any]] = []
        relics: list[dict[str, Any]] = []
        potions: list[dict[str, Any]] = []
        shop_map: dict[str, Any] = {
            "cards": [],
            "relics": [],
            "potions": [],
            "leave": 0,
            "remove_card": None,
        }
        removal_cost: int | None = None
        for row in msg.get("stock") or []:
            kind = row.get("kind")
            item = row.get("item") or {}
            option = row.get("option")
            stocked = bool(row.get("stocked", True))
            if kind == "buy_card":
                shop_map["cards"].append(option)
                cards.append(
                    {
                        "index": len(cards),
                        "id": row.get("label"),
                        "name": item.get("name") or row.get("label"),
                        "type": item.get("type") or "?",
                        "card_cost": _card_cost(item, 0),
                        "description": item.get("description"),
                        "stats": None,
                        "after_upgrade": None,
                        "cost": row.get("price"),
                        "is_stocked": stocked,
                        "on_sale": bool(row.get("on_sale", False)),
                    }
                )
            elif kind == "buy_relic":
                shop_map["relics"].append(option)
                relic_dict = _relic_offer_dict(item, len(relics))
                relic_dict["cost"] = row.get("price")
                relic_dict["is_stocked"] = stocked
                relics.append(relic_dict)
            elif kind == "buy_potion":
                shop_map["potions"].append(option)
                potion_dict = _potion_offer_dict(item, len(potions))
                potion_dict["cost"] = row.get("price")
                potion_dict["is_stocked"] = stocked
                potions.append(potion_dict)
            elif kind == "remove_card":
                shop_map["remove_card"] = option
                # Used up this visit: the host then reports no removal.
                removal_cost = row.get("price") if stocked else None
        self._shop_map = shop_map
        return {
            "type": "decision",
            "decision": "shop",
            "context": self._context(msg, room_type="Shop"),
            "cards": cards,
            "relics": relics,
            "potions": potions,
            "card_removal_cost": removal_cost,
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_event(self, msg: dict[str, Any]) -> dict[str, Any]:
        options = msg.get("options") or []
        event_id = next((o.get("event_id") for o in options if o.get("event_id")), None)
        return self._event_choice_dict(
            msg,
            event_name=event_id,
            description=None,
            options=[
                {
                    "index": o.get("index"),
                    "title": o.get("label"),
                    "description": o.get("description"),
                    "is_locked": not o.get("enabled", True),
                    "vars": None,
                }
                for o in options
            ],
        )

    def _translate_boss_relic(self, msg: dict[str, Any]) -> dict[str, Any]:
        """``boss_relic`` -> ``event_choice`` -- see module docstring."""
        options = msg.get("options") or []
        return self._event_choice_dict(
            msg,
            event_name="Boss Relic",
            description=None,
            options=[
                {
                    "index": o.get("index"),
                    "title": o.get("label"),
                    "description": o.get("description"),
                    "is_locked": not o.get("enabled", True),
                    "vars": None,
                }
                for o in options
            ],
        )

    def _translate_crystal_sphere(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Best-effort ``crystal_sphere`` -> ``event_choice``.

        No headless decision type corresponds to this screen at all (it is
        a Necrobinder-adjacent mechanic never covered by the 9-item
        decision vocabulary the phase-2 plan enumerates), and it was never
        exercised in the ~3.5k-decision live corpus this module's fixtures
        are drawn from -- treat this translation as untested / a parity
        residual, not a verified mapping.
        """
        options = msg.get("options") or []

        def _title(o: dict[str, Any]) -> str:
            if o.get("action") == "proceed":
                return "Proceed"
            return f"Cell ({o.get('x')},{o.get('y')})"

        return self._event_choice_dict(
            msg,
            event_name="Crystal Sphere",
            description=None,
            options=[
                {
                    "index": o.get("index"),
                    "title": _title(o),
                    "description": None,
                    "is_locked": not o.get("enabled", True),
                    "vars": None,
                }
                for o in options
            ],
        )

    def _event_choice_dict(
        self,
        msg: dict[str, Any],
        *,
        event_name: str | None,
        description: str | None,
        options: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "type": "decision",
            "decision": "event_choice",
            "context": self._context(msg, room_type="Event"),
            "event_name": event_name,
            "description": description,
            "options": options,
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_card_reward(self, msg: dict[str, Any]) -> dict[str, Any]:
        cards = msg.get("cards") or []
        gold_earned = None
        if self._gold_before_reward is not None:
            run_gold = (msg.get("run_state") or {}).get("gold")
            if isinstance(run_gold, int):
                gold_earned = run_gold - self._gold_before_reward
        return {
            "type": "decision",
            "decision": "card_reward",
            "context": self._context(msg, room_type=None),
            "cards": [
                {
                    "index": c.get("index"),
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "cost": _card_cost(c),
                    "type": c.get("type"),
                    "rarity": c.get("rarity"),
                    "description": c.get("description"),
                    "stats": None,
                    "after_upgrade": None,
                }
                for c in cards
            ],
            "can_skip": bool(msg.get("can_skip", True)),
            "gold_earned": gold_earned,
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_card_select(self, msg: dict[str, Any]) -> dict[str, Any]:
        cards = msg.get("cards") or []
        min_select = msg.get("min_select", msg.get("min_count", 0))
        max_select = msg.get("max_select", msg.get("max_count", len(cards)))
        return {
            "type": "decision",
            "decision": "card_select",
            "context": self._context(msg, room_type=None),
            "cards": [
                {
                    "index": c.get("index"),
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "cost": _card_cost(c),
                    "type": c.get("type"),
                    "upgraded": bool(c.get("upgrade_level")) or bool(c.get("upgraded")),
                    "stats": None,
                    "description": c.get("description"),
                    "after_upgrade": None,
                }
                for c in cards
            ],
            "min_select": min_select,
            "max_select": max_select,
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_bundle_select(self, msg: dict[str, Any]) -> dict[str, Any]:
        bundles = msg.get("bundles") or []
        return {
            "type": "decision",
            "decision": "bundle_select",
            "context": self._context(msg, room_type=None),
            "bundles": [
                {
                    "index": b.get("index"),
                    "cards": [
                        {
                            "name": c.get("name"),
                            "cost": _card_cost(c),
                            "type": c.get("type"),
                            "description": c.get("description"),
                            "stats": None,
                        }
                        for c in (b.get("cards") or [])
                    ],
                }
                for b in bundles
            ],
            "player": self._player_summary(msg.get("run_state") or {}),
        }

    def _translate_game_over(self, msg: dict[str, Any]) -> dict[str, Any]:
        run_state = msg.get("run_state") or {}
        return {
            "type": "decision",
            "decision": "game_over",
            "context": self._context(msg, room_type=None),
            "victory": msg.get("result") == "victory",
            "player": self._player_summary(run_state),
            "act": msg.get("act", 1),
            "floor": msg.get("floor", 0),
        }

    # ── Shared field builders ────────────────────────────────────────────────

    def _context(self, msg: dict[str, Any], *, room_type: str | None) -> dict[str, Any]:
        """Headless ``RunContext()``-shaped dict.

        ``act_name`` and ``boss`` are parity residuals: the bridge never
        sends a localized act title or the current act boss's identity
        (only its map coordinates, inside ``map_select.map.boss``), so
        both are always omitted/``None`` here.
        """
        return {
            "act": msg.get("act", 1),
            "act_name": None,
            "floor": msg.get("floor", 0),
            "room_type": room_type,
        }

    def _player_summary(self, run_state: dict[str, Any]) -> dict[str, Any]:
        """Headless ``PlayerSummary()``-shaped dict built from ``run_state``.

        ``block`` is always 0 -- ``run_state`` has no block field (block
        only exists mid-combat, where the fuller ``combat_play.player``
        already carries it via :mod:`realgame_adapter`, untouched here).
        """
        character_id = run_state.get("character")
        name = _character_title(character_id)
        master_deck = run_state.get("master_deck") or []
        held = [p for p in (run_state.get("potions") or []) if not p.get("empty")]
        # The host numbers potions by position among the filled slots
        # (`Player.Potions` skips empty ones); the bridge addresses belt
        # slots. Remember which slot each position is, for use/discard.
        self._potion_slots = [p.get("slot") for p in held]
        return {
            "name": name,
            "hp": run_state.get("hp", 0),
            "max_hp": run_state.get("max_hp", 0),
            "block": 0,
            "gold": run_state.get("gold", 0),
            "relics": [_relic_summary_dict(r) for r in (run_state.get("relics") or [])],
            "potions": [
                {
                    "index": i,
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "vars": None,
                    "target_type": p.get("target_type"),
                }
                for i, p in enumerate(held)
            ],
            "deck_size": len(master_deck),
            "deck": [_deck_card_dict(c) for c in master_deck],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (no per-instance state needed).
# ──────────────────────────────────────────────────────────────────────────────


def _character_title(character_id: str | None) -> str:
    """Headless ``PlayerSummary()``'s localized ``name`` (e.g. ``"The Ironclad"``).

    ``RunSimulator.cs`` resolves this via ``_loc.Bilingual("characters",
    <ID>.title")``; every one of the 5 playable characters'
    ``<ID>.title`` entry in ``localization_eng/characters.json`` is simply
    ``"The " + <Title Case name>`` (verified against all 5:
    IRONCLAD/SILENT/DEFECT/NECROBINDER/REGENT), so this reuses
    :data:`realgame_adapter.CHARACTER_ID_TO_DISPLAY` (already the
    authoritative id -> Title Case map) rather than hand-rolling a second
    table that could drift out of sync with it.
    """
    if not character_id:
        return "?"
    display = character_display_name(character_id)
    if display is None:
        return character_id
    return f"The {display}"


def _coerce_int_seed(seed: str | int | None) -> int | None:
    """Coerce a headless-style seed to the int the bridge's ``start_run`` wants.

    ``DebugSeedOverride`` (the only mechanism the mod has for pinning a
    seed) is integer-only; a non-numeric string (e.g. the headless
    default ``"test"``) cannot be reproduced against the live game and
    falls back to a fresh random seed (logged).
    """
    if seed is None:
        return None
    if isinstance(seed, int):
        return seed
    try:
        return int(seed)
    except (TypeError, ValueError):
        logger.warning(
            "RealGame: seed %r is not an integer; the bridge only supports int seeds "
            "(DebugSeedOverride) -- using a fresh random seed instead",
            seed,
        )
        return None


def _relic_vars(relic: dict[str, Any]) -> dict[str, int] | None:
    vars_dict: dict[str, int] = {}
    counter = relic.get("counter")
    if isinstance(counter, int):
        vars_dict["counter"] = counter
    stack_count = relic.get("stack_count")
    if isinstance(stack_count, int) and stack_count > 1:
        vars_dict["stack_count"] = stack_count
    return vars_dict or None


def _relic_summary_dict(relic: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": relic.get("id"),
        "name": relic.get("name"),
        "description": relic.get("description"),
        "vars": _relic_vars(relic),
    }
    # Headless-dialect reconstruction fields: sts2_hint.adapter reads the
    # live progress from a TOP-LEVEL ``counter`` (never from ``vars``) plus
    # the ToyBox melt flags. `_translate_combat` swaps these summaries in
    # over realgame_adapter's relics, so omitting them here re-broke the
    # counter/melt passthrough on the RealGame-class lane (realgame_parity /
    # sts2_mcp) after it was fixed in realgame_adapter._translate_relic.
    counter = relic.get("counter")
    if isinstance(counter, int) and not isinstance(counter, bool):
        out["counter"] = counter
    if relic.get("is_wax"):
        out["is_wax"] = True
    if relic.get("is_melted"):
        out["is_melted"] = True
    return out


def _relic_offer_dict(relic: dict[str, Any], index: int) -> dict[str, Any]:
    """Headless ``RelicOfferDict()``-shaped dict (used by ``shop``)."""
    d = _relic_summary_dict(relic)
    d["index"] = index
    return d


def _potion_offer_dict(potion: dict[str, Any], index: int) -> dict[str, Any]:
    """Headless ``PotionOfferDict()``-shaped dict (used by ``shop``).

    ``vars`` is always ``None`` -- the bridge's shop-slot Potion object
    carries no dynamic-var breakdown (unlike relics' ``counter``/
    ``stack_count``), so there is nothing to derive it from.
    """
    return {
        "id": potion.get("id"),
        "name": potion.get("name"),
        "description": potion.get("description"),
        "vars": None,
        "index": index,
    }


_KEYWORD_FLAGS: tuple[str, ...] = ("exhaust", "innate", "retain", "ethereal")


def _deck_card_dict(card: dict[str, Any]) -> dict[str, Any]:
    """Headless ``PlayerSummary()``'s per-card deck dict.

    ``id`` is prefixed ``"CARD."`` -- ``PlayerSummary``'s deck list emits
    ``c.Id.ToString()`` (``ModelId.ToString()`` == ``Category + "." +
    Entry``, ``MegaCrit.Sts2.Core.Models/ModelId.cs``), not the bare
    ``c.Id.Entry`` other surfaces (shop/card_reward/card_select) use. The
    bridge's card id is already confirmed bare-Entry-equivalent
    (:mod:`realgame_adapter`'s module docstring), so prepending the fixed
    ``"CARD"`` category reproduces the headless string exactly.
    """
    keywords = [flag.capitalize() for flag in _KEYWORD_FLAGS if card.get(flag)]
    raw_id = card.get("id")
    return {
        "id": f"CARD.{raw_id}" if raw_id else None,
        "name": card.get("name"),
        "cost": _card_cost(card),
        "type": card.get("type"),
        "upgraded": bool(card.get("upgrade_level")) or bool(card.get("upgraded")),
        "description": card.get("description"),
        "stats": None,
        "keywords": keywords or None,
        "tinker_time_rider": None,
        "after_upgrade": None,
    }


# Bridge message "type" -> RealGame translator method name. Kept as a plain
# dict (not a decorator registry) so _next_decision's dispatch is a single
# dict lookup; combat_action is handled inline (it needs the gold-snapshot
# side effect), so it is intentionally absent here and checked separately
# were this table ever consulted outside _next_decision.
_TRANSLATE_METHOD: dict[str, str] = {
    "combat_action": "_translate_combat",
    "map_select": "_translate_map_select",
    "rest_site": "_translate_rest_site",
    "shop": "_translate_shop",
    "event": "_translate_event",
    "boss_relic": "_translate_boss_relic",
    "crystal_sphere": "_translate_crystal_sphere",
    "card_reward": "_translate_card_reward",
    "card_select": "_translate_card_select",
    "card_bundle": "_translate_bundle_select",
}


def _card_cost(card: dict[str, Any], default: Any = None) -> Any:
    """A card's energy cost as the host states it.

    ``effective_cost`` is what playing the card costs now, which for an X-cost
    card is the player's energy (Multi-Cast offered at 3, Heavenly Drill at
    6); the host, like the card itself, says 0 (``cost``) for those.
    """
    if card.get("costs_x"):
        return card.get("cost", default)
    return card.get("effective_cost", card.get("cost", default))


def _only_ancient_node(decision: dict[str, Any]) -> bool:
    """A map choice whose only node is the act-start Ancient.

    The game can list that node more than once (act 3 of Ironclad seed 1000
    offered ``(3, 0)`` twice), so nodes count by coordinates.
    """
    choices = decision.get("choices") or []
    nodes = {(c.get("col"), c.get("row")) for c in choices}
    return (
        decision.get("decision") == "map_select"
        and len(nodes) == 1
        and str(choices[0].get("type")).lower() == "ancient"
    )


def _only_proceed(options: list[dict[str, Any]]) -> bool:
    """True when every option leaves a finished event (``is_proceed``, mod 2026-11-12)."""
    return bool(options) and all(o.get("is_proceed") for o in options)


#: Fire-and-forget mod messages that carry no decision: the snapshot lane
#: (PROTOCOL.md "Run snapshots"), the turn-undo broadcast and the coach's
#: fight recap. The mod sends them to every client, so a driver that stops on
#: an unknown type stops at the first save (GH #1287).
_NOTIFICATION_TYPES = frozenset(
    {"snapshot_saved", "snapshots", "snapshot_restored", "combat_undone", "combat_ended"}
)


# Headless action verb -> RealGame method, for RealGame.act()/send(). Every
# entry mirrors the corresponding sts2_client.ACTIONS verb's argument names.
_ACTION_DISPATCH: dict[str, Any] = {
    "select_map_node": lambda self, a: self.select_map_node(a.get("col"), a.get("row")),
    "play_card": lambda self, a: self.play_card(a.get("card_index"), a.get("target_index")),
    "end_turn": lambda self, a: self.end_turn(),
    "choose_option": lambda self, a: self.choose_option(a.get("option_index")),
    "select_card_reward": lambda self, a: self.select_card_reward(a.get("card_index")),
    "skip_card_reward": lambda self, a: self.skip_card_reward(),
    "buy_card": lambda self, a: self.buy_card(a.get("card_index")),
    "buy_relic": lambda self, a: self.buy_relic(a.get("relic_index")),
    "buy_potion": lambda self, a: self.buy_potion(a.get("potion_index")),
    "remove_card": lambda self, a: self.remove_card(),
    "select_bundle": lambda self, a: self.select_bundle(a.get("bundle_index")),
    "select_cards": lambda self, a: self.select_cards(a.get("indices")),
    "skip_select": lambda self, a: self.skip_select(),
    "use_potion": lambda self, a: self.use_potion(a.get("potion_index"), a.get("target_index")),
    "discard_potion": lambda self, a: self.discard_potion(a.get("potion_index")),
    "leave_room": lambda self, a: self.leave_room(),
    "proceed": lambda self, a: self.proceed(),
    "potion_swap": lambda self, a: self.potion_swap(a.get("discard_slot")),
}
