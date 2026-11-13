"""Tests for the Rust workbench backend (``workbench.backends.rust_game``).

These exercise the ``sts2_rs``-backed engine only — no C# headless process —
so they run on any box where ``maturin develop`` has installed ``sts2_rs``.
The whole module skips if that extension isn't importable.
"""

from __future__ import annotations

import pytest
import sts2_client

pytest.importorskip("sts2_rs", reason="rust backend requires the sts2_rs extension")

from workbench.backends import DuckTypedBackend, GameBackend, make_backend
from workbench.backends.rust_game import RustGame
from workbench.branching import BranchingLiveSession
from workbench.options import normalize_options

SEED = "WORKBENCH1"
_KNOWN_DECISIONS = {
    "map_select",
    "combat_play",
    "card_reward",
    "shop",
    "rest_site",
    "event_choice",
    "game_over",
}


def _new() -> RustGame:
    g = RustGame()
    g.start(character="Ironclad", seed=SEED, ascension=0)
    return g


def _drive_to_combat(g: RustGame, max_steps: int = 30) -> dict:
    """Advance through Neow/map until the first combat, return that state."""
    state = g.get_state()
    for _ in range(max_steps):
        dec = state["decision"]
        if dec == "combat_play":
            return state
        if dec == "map_select":
            ch = state["choices"][0]
            state = g.act("select_map_node", col=ch["col"], row=ch["row"])
        else:
            opts = normalize_options(state)
            o = opts[0]
            state = g.act(o["action"], **o["args"])
        if state.get("type") == "error":
            pytest.fail(f"unexpected engine error: {state}")
    pytest.fail("did not reach combat within step budget")


def test_make_backend_selects_classes() -> None:
    # The C# client is a subprocess speaking the five verbs; the factory wraps
    # it in the duck-typed adapter (constructing one would spawn dotnet).
    assert callable(make_backend("csharp"))
    assert issubclass(DuckTypedBackend, GameBackend)
    assert make_backend("rust") is RustGame
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("bogus")


def test_rust_game_satisfies_protocol() -> None:
    assert isinstance(RustGame(), GameBackend)


def test_start_returns_known_decision() -> None:
    g = _new()
    state = g.get_state()
    assert state["decision"] in _KNOWN_DECISIONS
    p = state["player"]
    assert p["name"] == "Ironclad"
    assert p["hp"] > 0 and p["max_hp"] >= p["hp"]
    assert isinstance(p["deck"], list) and p["deck"]


def test_combat_state_shape_and_play() -> None:
    g = _new()
    state = _drive_to_combat(g)

    # Hand cards carry the fields the frontend / renderer read.
    assert state["hand"], "combat should deal a hand"
    for c in state["hand"]:
        assert {"index", "id", "name", "cost", "can_play", "target_type"} <= c.keys()
    assert any(c["can_play"] for c in state["hand"])

    # Enemies carry intents now (full-parity serializer).
    assert state["enemies"]
    enemy = state["enemies"][0]
    assert {"index", "name", "hp", "max_hp", "intents"} <= enemy.keys()
    assert enemy["intents"], "enemy should telegraph an intent"
    assert "type" in enemy["intents"][0]

    # Playing a legal card advances the state without error.
    playable = next(c for c in state["hand"] if c["can_play"])
    target = 0 if playable["target_type"] == "AnyEnemy" else None
    after = g.act("play_card", card_index=playable["index"], target_index=target)
    assert after.get("type") != "error"


def test_required_keys_per_decision() -> None:
    """Every decision dict carries the keys server enrichment reads."""
    g = _new()
    state = g.get_state()
    seen: set[str] = set()
    for _ in range(40):
        dec = state["decision"]
        seen.add(dec)
        assert "player" in state
        if dec == "map_select":
            # No full_map: the Rust map graph is inconsistent with the move
            # space, so we render the reachable choices as a button list.
            assert "choices" in state and "full_map" not in state
            ch = state["choices"][0]
            state = g.act("select_map_node", col=ch["col"], row=ch["row"])
        elif dec == "combat_play":
            assert {"hand", "enemies", "energy", "round"} <= state.keys()
            pc = [c for c in state["hand"] if c["can_play"]]
            if pc:
                # Native convention: the FULL encounter slot of a live enemy
                # (slot 0 may be a corpse; GH #1119 made that a rejection).
                live = state["enemies"][0]["index"]
                t = live if pc[0]["target_type"] == "AnyEnemy" else None
                state = g.act("play_card", card_index=pc[0]["index"], target_index=t)
            else:
                state = g.act("end_turn")
        elif dec == "card_reward":
            assert "cards" in state
            opts = normalize_options(state)
            state = g.act(opts[0]["action"], **opts[0]["args"])
        elif dec == "game_over":
            assert "victory" in state
            break
        else:
            opts = normalize_options(state)
            state = g.act(opts[0]["action"], **opts[0]["args"])
        if state.get("type") == "error":
            pytest.fail(f"engine error at {dec}: {state}")
    assert {"map_select", "combat_play"} <= seen


def test_map_select_is_selectable() -> None:
    """Regression: every reachable move surfaces as a clickable choice and the
    SVG-picker-only ``full_map`` is omitted (its graph misses some moves)."""
    g = _new()
    state = g.get_state()
    for _ in range(8):
        if state["decision"] == "map_select":
            break
        opts = normalize_options(state)
        state = g.act(opts[0]["action"], **opts[0]["args"])
    assert state["decision"] == "map_select"
    assert "full_map" not in state
    moves = [a["col"] for a in g.rm.available_actions() if a.get("action") == "move"]
    choice_cols = [c["col"] for c in state["choices"]]
    assert sorted(choice_cols) == sorted(moves), "every reachable move must be a choice"
    # Picking each offered choice is accepted by the engine.
    for col in choice_cols:
        gg = _new()
        s = gg.get_state()
        while s["decision"] != "map_select":
            o = normalize_options(s)[0]
            s = gg.act(o["action"], **o["args"])
        res = gg.act("select_map_node", col=col, row=0)
        assert res.get("type") != "error", f"move col={col} rejected"


def test_event_with_no_labels_still_offers_option() -> None:
    """Regression: an event exposing an ``event_choice`` action but no
    ``event_options()`` label must still yield a clickable option (it rendered
    an empty, unescapable screen before)."""

    class _FakeRM:
        # The Act-1 model the run walks (GH #544). `_serialize` reports it,
        # so a RunManager stand-in has to carry it like the real one does.
        act1 = "overgrowth"

        def available_actions(self):
            return [{"action": "event_choice", "idx": 0}]

        def materialize_event(self):
            return None  # no event could be picked / bound

        def current_event_name(self):
            return None

        def event_option_infos(self):
            return None

    g = RustGame()
    g.rm = _FakeRM()
    state: dict = {}
    g._fill_event(state)
    assert len(state["options"]) == 1
    assert state["options"][0]["index"] == 0
    assert state["options"][0]["title"] == "Continue"
    assert state["event_name"] == ""
    assert normalize_options({"decision": "event_choice", **state})


def test_materialized_event_renders_name_and_choices() -> None:
    """A normal ``?`` event materializes its name + choices on arrival: the
    title becomes the loc-key stem, options carry the spec's English label and
    description, and the body points at the event's INITIAL page."""

    class _FakeRM:
        # The Act-1 model the run walks (GH #544). `_serialize` reports it,
        # so a RunManager stand-in has to carry it like the real one does.
        act1 = "overgrowth"

        def available_actions(self):
            return [{"action": "event_choice", "idx": 0}, {"action": "event_choice", "idx": 1}]

        def materialize_event(self):
            return "LostWisp"

        def current_event_name(self):
            return "LostWisp"

        def event_option_infos(self):
            return [
                {
                    "option_id": "claim",
                    "label": "Claim",
                    "description": "Gain Lost Wisp relic + Decay curse.",
                    "enabled": True,
                },
                {
                    "option_id": "search",
                    "label": "Search",
                    "description": "Gain 60 gold.",
                    "enabled": False,
                },
            ]

    g = RustGame()
    g.rm = _FakeRM()
    state: dict = {}
    g._fill_event(state)
    assert state["event_name"] == "LOST_WISP.title"
    assert state["description"] == "LOST_WISP.pages.INITIAL.description"
    opts = state["options"]
    assert [o["title"] for o in opts] == ["Claim", "Search"]
    assert opts[0]["description"] == "Gain Lost Wisp relic + Decay curse."
    assert opts[0]["is_locked"] is False
    assert opts[1]["is_locked"] is True  # enabled=False -> locked
    # Plain-text choices don't carry a relic art text_key.
    assert all(not o["text_key"] for o in opts)


def test_neow_relic_options_keep_art_text_key() -> None:
    """Neow offers come back with relic-id labels (all-caps): they must keep
    the ``text_key`` so server enrichment attaches the relic art/description,
    and not leak the generic blurb as the option description."""

    class _FakeRM:
        # The Act-1 model the run walks (GH #544). `_serialize` reports it,
        # so a RunManager stand-in has to carry it like the real one does.
        act1 = "overgrowth"

        def available_actions(self):
            return [{"action": "event_choice", "idx": 0}]

        def materialize_event(self):
            return "Neow"

        def current_event_name(self):
            return "Neow"

        def event_option_infos(self):
            return [
                {
                    "option_id": "positive_1",
                    "label": "WINGED_BOOTS",
                    "description": "Gain a positive relic.",
                    "enabled": True,
                }
            ]

    g = RustGame()
    g.rm = _FakeRM()
    state: dict = {}
    g._fill_event(state)
    o = state["options"][0]
    assert o["text_key"] == "WINGED_BOOTS"
    assert o["title"] == "WINGED_BOOTS.title"
    assert o["description"] == ""  # let the relic's own description fill in
    assert state["event_name"] == "NEOW.title"


# ── target_convention="csharp" loc-key resolution ──────────────────────────
#
# meta_differ_report.md BUG #1/#2: the deployed PPO dispatch
# (``headless_eval.to_bridge_event`` -> ``realgame_ppo``) reads
# ``event_name``/option ``title`` DIRECTLY, with no server-side enrichment
# step in between (unlike the workbench UI, which resolves these client-side
# via ``server._enrich``). The real Sts2Headless DLL always sends already
# -resolved display text for these fields (``RunSimulator.cs``'s
# ``_loc.Bilingual``/``_loc.Relic`` calls); ``target_convention="csharp"``
# must match that, while ``target_convention="native"`` (asserted raw above)
# must NOT change.


def test_csharp_resolve_event_fields_resolves_neow_gate_value() -> None:
    """The exact bug: an unresolved ``"NEOW.title"`` never equals ``"NEOW"``
    after upper-casing, so ``realgame_ppo._event_option_label``'s Neow gate
    (``event_id.upper().replace(".", "_") != "NEOW"``) never fires and the
    enum-tail relic-id substitution it guards is permanently skipped."""
    from workbench.backends.rust_game import _csharp_resolve_event_fields

    state = {
        "event_name": "NEOW.title",
        "options": [
            {"index": 0, "title": "WINGED_BOOTS.title", "text_key": "WINGED_BOOTS"},
        ],
    }
    _csharp_resolve_event_fields(state)
    assert state["event_name"] == "Neow"
    assert state["event_name"].upper().replace(".", "_") == "NEOW"
    assert state["options"][0]["title"] == "Winged Boots"


def test_csharp_resolve_event_fields_resolves_boss_relic_and_regular_event() -> None:
    """Boss-relic option titles (the 59%-disagreement / slot-1-collapse
    surface) and a regular event's title both resolve to plain prose."""
    from workbench.backends.rust_game import _csharp_resolve_event_fields

    state = {
        "event_name": "",
        "options": [
            {"index": 0, "title": "DRIFTWOOD.title", "text_key": "DRIFTWOOD"},
            {"index": 1, "title": "BIIIG_HUG.title", "text_key": "BIIIG_HUG"},
        ],
    }
    _csharp_resolve_event_fields(state)
    assert [o["title"] for o in state["options"]] == ["Driftwood", "Biiig Hug"]

    event_state = {"event_name": "LOST_WISP.title", "options": []}
    _csharp_resolve_event_fields(event_state)
    assert event_state["event_name"] == "The Lost Wisp"


def test_csharp_resolve_event_fields_prettify_fallback_and_prose_passthrough() -> None:
    """A loc key our bundle has no entry for falls back to prettified text
    (never leaks the raw key); already-resolved prose passes through
    unchanged (idempotent — safe to call on native-shaped state too)."""
    from workbench.backends.rust_game import _csharp_resolve_event_fields

    state = {
        "event_name": "SOME_UNKNOWN_EVENT.title",
        "options": [
            {"index": 0, "title": "Claim", "description": "Gain 60 gold."},
            {"index": 1, "title": "", "description": ""},
        ],
    }
    _csharp_resolve_event_fields(state)
    assert state["event_name"] == "Some Unknown Event"
    assert state["options"][0]["title"] == "Claim"
    assert state["options"][0]["description"] == "Gain 60 gold."
    assert state["options"][1]["title"] == ""


def test_csharp_relic_reward_title_resolves() -> None:
    """``card_reward_relic`` (BUG #1's highest-volume finding — deployed
    ``Skip``s a free relic 86% of the time when driven blind): the take
    /skip option must carry the relic's real name, not a raw loc key."""

    class _FakeRM:
        # The Act-1 model the run walks (GH #544). `_serialize` reports it,
        # so a RunManager stand-in has to carry it like the real one does.
        act1 = "overgrowth"

        def card_reward_bundles(self):
            return []

        def available_actions(self):
            return [
                {"action": "pick_relic_reward", "idx": 0},
                {"action": "skip_relic", "idx": 1},
            ]

        def relic_reward_ids(self):
            return ["DRIFTWOOD"]

    g = RustGame(target_convention="csharp")
    g.rm = _FakeRM()
    state: dict = {"decision": "card_reward"}
    g._fill_card_reward(state)
    from workbench.backends.rust_game import _csharp_resolve_event_fields

    assert state["decision"] == "event_choice"
    _csharp_resolve_event_fields(state)
    titles = [o["title"] for o in state["options"]]
    assert "Driftwood" in titles
    assert "Skip" in titles


class _RelicAutoCollectRM:
    # The Act-1 model the run walks (GH #544). `_serialize` reports it,
    # so a RunManager stand-in has to carry it like the real one does.
    act1 = "overgrowth"

    """Minimal fake RunManager: a pure combat relic reward (no card choices)
    that, once auto-taken, advances straight to ``run_over``."""

    def __init__(self) -> None:
        self._phase = "card_reward"
        self.taken_actions: list[dict] = []
        self.gold = 0
        self.current_hp = 50
        self.max_hp = 80
        self.act_index = 1
        self.ascension_level = 0
        self.player_won = True

    def phase(self):
        return self._phase

    def total_floor(self):
        return 16

    def relics(self):
        return ["DRIFTWOOD"] if self.taken_actions else []

    def potions(self):
        return []

    def max_potion_slots(self):
        return 3

    def master_deck_detailed(self):
        return []

    def card_reward_bundles(self):
        return []

    def available_actions(self):
        if self._phase != "card_reward":
            return []
        return [
            {"action": "pick_relic_reward", "idx": 0},
            {"action": "skip_relic", "idx": 1},
        ]

    def relic_reward_ids(self):
        return ["DRIFTWOOD"]

    def take_action(self, action):
        self.taken_actions.append(action)
        assert action["action"] == "pick_relic_reward", (
            "csharp auto-collect must TAKE, never skip — matches "
            "RunSimulator.cs's unconditional SelectUnsynchronized()"
        )
        self._phase = "run_over"
        return "advanced"


def test_csharp_auto_collects_combat_relic_reward_no_decision() -> None:
    """The real Sts2Headless DLL auto-collects combat relic/gold/potion
    rewards (RunSimulator.cs ``DetectPostCombatState``,
    ``reward.SelectUnsynchronized()`` for Gold/Relic/PotionReward) and NEVER
    presents a take/skip decision for them — only CardReward entries ever
    surface. csharp-mode emulation must match: auto-take the Rust engine's
    CardReward-phase relic fan-out (``pick_relic_reward``) instead of
    rendering the old phantom ``event_choice`` "Take X / Skip" decision that
    contaminated the deployed net's obs (meta_differ_report.md BUG #1)."""
    g = RustGame(target_convention="csharp")
    g.character = "Ironclad"
    rm = _RelicAutoCollectRM()
    g.rm = rm

    state = g._serialize()

    assert rm.taken_actions == [{"action": "pick_relic_reward", "idx": 0}]
    assert state["decision"] == "game_over"
    assert state["victory"] is True


def test_native_still_surfaces_combat_relic_reward_decision() -> None:
    """``target_convention="native"`` (the workbench UI) is UNCHANGED: the
    take/skip choice still surfaces for a human player to click through."""
    g = RustGame()
    g.character = "Ironclad"
    rm = _RelicAutoCollectRM()
    g.rm = rm

    state = g._serialize()

    assert rm.taken_actions == []
    assert state["decision"] == "event_choice"


def test_serialize_boss_relic_wiring_native_vs_csharp() -> None:
    """End-to-end through ``_serialize`` (not the ``_fill_boss_relic`` unit
    directly): ``target_convention="csharp"`` resolves the boss-relic option
    titles; ``"native"`` (the workbench UI's own convention, unchanged by
    this fix) must still leak the raw loc key for ``server._enrich``."""

    class _FakeRM:
        # The Act-1 model the run walks (GH #544). `_serialize` reports it,
        # so a RunManager stand-in has to carry it like the real one does.
        act1 = "overgrowth"

        def phase(self):
            return "boss_relic"

        def total_floor(self):
            return 16

        act_index = 2
        ascension_level = 0
        current_hp = 50
        max_hp = 80
        gold = 0

        def relics(self):
            return []

        def potions(self):
            return []

        def max_potion_slots(self):
            return 3

        def master_deck_detailed(self):
            return []

        def available_actions(self):
            return [
                {"action": "pick_boss_relic", "idx": 0},
                {"action": "pick_boss_relic", "idx": 1},
            ]

        def boss_relic_options(self):
            return ["DRIFTWOOD", "BIIIG_HUG"]

    g_native = RustGame()
    g_native.character = "Ironclad"
    g_native.rm = _FakeRM()
    native_state = g_native._serialize()
    assert native_state["decision"] == "event_choice"
    assert [o["title"] for o in native_state["options"]] == ["DRIFTWOOD.title", "BIIIG_HUG.title"]

    g_csharp = RustGame(target_convention="csharp")
    g_csharp.character = "Ironclad"
    g_csharp.rm = _FakeRM()
    csharp_state = g_csharp._serialize()
    assert csharp_state["decision"] == "event_choice"
    assert [o["title"] for o in csharp_state["options"]] == ["Driftwood", "Biiig Hug"]


def test_rest_site_labels_and_smith_substate() -> None:
    """Rest options render as Rest/Smith (not Option 0/1), and the Smith
    card-pick sub-state still yields clickable options (never stuck)."""

    class _RestRM:
        def __init__(self, acts, smith_ids=None):
            self._acts = acts
            self._smith_ids = smith_ids or []
            self.pending_deck_removals = 0
            self.current_hp = 50
            self.max_hp = 80
            self.gold = 0
            self.act_index = 0

        def available_actions(self):
            return self._acts

        def smith_options(self):
            return self._smith_ids

        def phase(self):
            return "rest_site"

        def relics(self):
            return []

        def master_deck_detailed(self):
            return []

        def total_floor(self):
            return 1

    g = RustGame()
    # Base rest list: [Rest, Smith].
    g.rm = _RestRM([{"action": "rest_option", "idx": 0}, {"action": "rest_option", "idx": 1}])
    state: dict = {}
    g._fill_rest(state)
    labels = [o["option_id"] for o in state["options"]]
    assert labels == ["Rest", "Smith"]
    # choose_option(1) must map back to the Smith rest_option action.
    assert g._translate("choose_option", {"option_index": 1}) == {
        "action": "rest_option",
        "idx": 1,
    }

    # Smith sub-state: actions are smith_card picks → render as a card_select
    # with real card ids (not "Upgrade card N"); select_cards routes to the
    # smith_card action (not a deck removal).
    g.rm = _RestRM(
        [{"action": "smith_card", "idx": 0}, {"action": "smith_card", "idx": 3}],
        smith_ids=["StrikeIronclad", "DefendIronclad", "Bash", "Anger"],
    )
    state = {}
    g._fill_rest(state)
    assert state["decision"] == "card_select"
    assert state["context"] == "upgrade"
    assert [c["index"] for c in state["cards"]] == [0, 3]
    assert state["cards"][1]["id"] == "ANGER"
    assert g._translate("select_cards", {"indices": "3"}) == {
        "action": "smith_card",
        "idx": 3,
    }


def test_closed_loop_legality() -> None:
    """Every option ``normalize_options`` offers is accepted by ``act``."""
    sess = BranchingLiveSession(logs_dir="/tmp", backend="rust")
    state = sess.start("Ironclad", SEED, 0)
    for _ in range(25):
        dec = state["decision"]
        if dec == "game_over":
            break
        if dec == "combat_play":
            pc = [c for c in state["hand"] if c["can_play"]]
            if pc:
                # Native convention: the FULL encounter slot of a live enemy
                # (slot 0 may be a corpse; GH #1119 made that a rejection).
                live = state["enemies"][0]["index"]
                t = live if pc[0]["target_type"] == "AnyEnemy" else None
                res = sess.apply("play_card", {"card_index": pc[0]["index"], "target_index": t})
            else:
                res = sess.apply("end_turn", {})
        else:
            opts = normalize_options(state)
            o = opts[0]
            res = sess.apply(o["action"], dict(o["args"]))
        assert res["error"] is None, f"option rejected in {dec}: {res['error']}"
        state = res["state"]
    sess.quit()


def test_replay_determinism() -> None:
    """A recorded verb line replays to an identical leaf (seed-deterministic)."""
    sess = BranchingLiveSession(logs_dir="/tmp", backend="rust")
    state = sess.start("Ironclad", SEED, 0)
    for _ in range(6):
        if state["decision"] == "game_over":
            break
        if state["decision"] == "combat_play":
            pc = [c for c in state["hand"] if c["can_play"]]
            if pc:
                # Native convention: the FULL encounter slot of a live enemy
                # (slot 0 may be a corpse; GH #1119 made that a rejection).
                live = state["enemies"][0]["index"]
                t = live if pc[0]["target_type"] == "AnyEnemy" else None
                res = sess.apply("play_card", {"card_index": pc[0]["index"], "target_index": t})
            else:
                res = sess.apply("end_turn", {})
        else:
            o = normalize_options(state)[0]
            res = sess.apply(o["action"], dict(o["args"]))
        state = res["state"]
    line, _ = sess.active_line()
    leaf_hp = state["player"]["hp"]
    leaf_deck = [c["id"] for c in state["player"]["deck"]]
    sess.quit()

    replay = BranchingLiveSession(logs_dir="/tmp", backend="rust")
    rstate = replay.load_line("Ironclad", SEED, 0, line)
    assert rstate["player"]["hp"] == leaf_hp
    assert [c["id"] for c in rstate["player"]["deck"]] == leaf_deck
    replay.quit()


def test_interactive_card_removal_renders_card_select() -> None:
    """An on-obtain card-removal relic (Precise Scissors from Neow) presents a
    deck ``card_select``; resolving it removes the picked card (matching C#)."""
    import sts2_rs

    # Seed 3's Neow offers Precise Scissors (a remove-1-card relic) under the
    # v0.107.1 build. Drive the adapter through it via a direct int-seed
    # RunManager. (The exact option index is resolved dynamically below.)
    g = RustGame()
    g.character = "Ironclad"
    g.rm = sts2_rs.RunManager(
        seed=3,
        character="Ironclad",
        ascension_level=0,
        neow_start=True,
        interactive_card_removal=True,
    )
    state = g.get_state()
    deck_before = len(state["player"]["deck"])

    # Neow blessing → pick Precise Scissors.
    assert state["decision"] == "event_choice"
    idx = next(
        (o["index"] for o in state["options"] if o.get("text_key") == "PRECISE_SCISSORS"),
        None,
    )
    assert idx is not None, "seed 9 Neow should offer Precise Scissors"
    state = g.act("choose_option", option_index=idx)

    # Collect the relic (Take relic), reaching the forced removal card_select.
    for _ in range(3):
        if state["decision"] == "card_select":
            break
        opts = normalize_options(state)
        state = g.act(opts[0]["action"], **opts[0]["args"])
    assert state["decision"] == "card_select"
    assert state.get("context") == "remove"
    assert state["min_select"] == 1 and not state.get("can_skip")
    assert len(state["cards"]) == deck_before

    # Resolving the pick removes exactly one card from the deck.
    opt = normalize_options(state)[0]
    state = g.act(opt["action"], **opt["args"])
    assert state.get("type") != "error"
    assert len(state["player"]["deck"]) == deck_before - 1


def test_native_hint_in_combat() -> None:
    """``hint`` returns MCTS scores keyed to hand indices during combat, with
    card loc keys resolved in the per-action labels (not 'STRIKE_IRONCLAD.title')."""
    loc = {"STRIKE_IRONCLAD.title": "Strike", "DEFEND_IRONCLAD.title": "Defend"}
    sess = BranchingLiveSession(logs_dir="/tmp", backend="rust", loc=loc)
    state = sess.start("Ironclad", SEED, 0)
    for _ in range(30):
        if state["decision"] == "combat_play":
            break
        if state["decision"] == "map_select":
            ch = state["choices"][0]
            res = sess.apply("select_map_node", {"col": ch["col"], "row": ch["row"]})
        else:
            o = normalize_options(state)[0]
            res = sess.apply(o["action"], dict(o["args"]))
        state = res["state"]
    assert state["decision"] == "combat_play"
    result = sess.hint({"sims": 64})
    assert result is not None and "scores" in result
    assert result["scores"], "MCTS should score at least one action"
    assert any(r["is_best"] for r in result["scores"])
    # Labels are resolved display names, never raw loc keys.
    assert all(".title" not in r["action"] for r in result["scores"]), result["scores"]
    sess.quit()


def test_added_card_carries_spec_cost_not_pod_default() -> None:
    """A card added to the deck (reward / force-add) is built through the
    card factory, so its spec cost is preserved. Regression: cards were
    created from ``Card::pod_defaults()`` (cost 1), mis-showing a 0-cost
    card like Anger as cost 1 in combat."""
    import sts2_rs

    g = _new()
    g.rm.force_add_card("Anger")  # Anger is a 0-cost attack
    state = _drive_to_combat(g)
    assert state["decision"] == "combat_play"
    wire = sts2_rs.state_to_dict(g.rm.combat_state())
    cards = wire["deck"]["hand"] + wire["deck"].get("draw_pile", [])
    anger = next((c for c in cards if "ANGER" in (c.get("id") or "").upper()), None)
    assert anger is not None, "Anger should be in the combat deck"
    assert anger["cost_override"] == 0, f"Anger must cost 0, got {anger['cost_override']}"
    assert anger["base_damage"] == 6, "Anger should carry its spec damage, not POD default"


def test_full_map_boss_carries_pre_rolled_name() -> None:
    """The content side pre-rolls the act's boss at act-map generation
    time (`pre_roll_act_boss`), so ``get_map`` surfaces the encounter
    the player will actually fight from floor 1 — matching the C# game
    where the boss is visible on the map widget at act start. Act 1's
    live pool is multi-entry (Vantom, Ceremonial Beast, The Kin); the
    seed-dependent pick is one of those three, and the Python side
    resolves a display ``name`` from ``kind``."""
    g = _new()
    boss = g.get_map().get("boss")
    assert isinstance(boss, dict)
    assert boss.get("type") == "Boss"
    assert boss.get("kind") in {"Vantom", "CeremonialBeast", "KinPriest"}
    assert boss.get("name") in {"Vantom", "Ceremonial Beast", "Kin Priest"}
    g.close()


def test_map_select_surfaces_selectable_full_map() -> None:
    """At a map node, the state carries the full graph (attached by the
    session's ``_attach_full_map``) and every offered choice (col, row) is a
    node in it, so the SVG picker can highlight it."""
    sess = BranchingLiveSession(logs_dir="/tmp", backend="rust")
    state = sess.start("Ironclad", SEED, 0)
    for _ in range(8):
        if state["decision"] == "map_select":
            break
        o = normalize_options(state)[0]
        state = sess.apply(o["action"], dict(o["args"]))["state"]
    assert state["decision"] == "map_select"
    fm = state.get("full_map")
    assert fm and fm.get("rows"), "map_select must carry the full graph"
    keys = {f"{n['col']},{n['row']}" for row in fm["rows"] for n in row}
    choices = state.get("choices") or []
    assert choices, "expected at least one reachable node"
    for c in choices:
        assert f"{c['col']},{c['row']}" in keys, f"choice {c} missing from graph"
    sess.quit()


def test_live_log_label_resolves_loc_keys() -> None:
    """The branch log label/diff resolve loc keys (``Play Strike``, not
    ``Play STRIKE_IRONCLAD.title``) when the session is given a loc map."""
    loc = {
        "STRIKE_IRONCLAD.title": "Strike",
        "DEFEND_IRONCLAD.title": "Defend",
    }
    sess = BranchingLiveSession(logs_dir="/tmp", backend="rust", loc=loc)
    state = sess.start("Ironclad", SEED, 0)
    for _ in range(30):
        if state["decision"] == "combat_play":
            break
        if state["decision"] == "map_select":
            ch = state["choices"][0]
            res = sess.apply("select_map_node", {"col": ch["col"], "row": ch["row"]})
        else:
            o = normalize_options(state)[0]
            res = sess.apply(o["action"], dict(o["args"]))
        state = res["state"]
    assert state["decision"] == "combat_play"
    play = next(c for c in state["hand"] if c.get("can_play"))
    args = {"card_index": play["index"]}
    if play.get("target_type") not in (None, "None"):
        args["target_index"] = state["enemies"][0]["index"]
    res = sess.apply("play_card", args)
    assert ".title" not in res["label"], res["label"]
    assert res["label"].startswith("Play "), res["label"]
    sess.quit()


def test_enemy_index_is_encounter_position_skipping_dead(monkeypatch) -> None:
    """A dead enemy earlier in the encounter must not shift the survivor's
    index. ``enemies[].index`` is the engine encounter position (which is the
    ``play_card`` target_index the frontend echoes back), so a play still hits
    the right creature after a kill. Regression for the silent no-op where a
    play targeted a corpse — energy was not spent and the card was replayable.
    """
    import sts2_rs

    wire = {
        "deck": {
            "hand": [{"id": "FIGHT_ME", "base_damage": 5, "kind": "card"}],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        },
        # Dead TwigSlime at encounter position 0, living Flyconid at position 1.
        "encounter": [
            {"kind": "TwigSlimeM", "hp": 0, "max_hp": 48, "is_dead": True},
            {"kind": "Flyconid", "hp": 36, "max_hp": 36, "is_dead": False},
        ],
        "player": {"hp": 62, "max_hp": 80, "block": 3, "energy": 2, "max_energy": 3, "powers": []},
        "round": 2,
    }
    monkeypatch.setattr(sts2_rs, "state_to_dict", lambda cs: wire)

    class _Cs:
        def pending_choice(self):
            return None

    class _Rm:
        def combat_state(self):
            return _Cs()

        def available_actions(self):
            return [{"action": "play_card", "hand_idx": 0, "target_index": 1}]

    g = RustGame()
    g.rm = _Rm()  # type: ignore[assignment]
    state: dict = {"player": {}}
    g._fill_combat(state)

    assert [e["index"] for e in state["enemies"]] == [1], (
        "the living enemy keeps its encounter position (1), not a re-counted 0"
    )
    assert state["enemies"][0]["name"]  # rendered, sanity


def test_top_level_osty_lift_shape() -> None:
    """The pure wire→display mapping: a live pet gains a name, a dead one
    keeps its bare payload, and absence yields ``None``."""
    from workbench.backends.rust_game import _top_level_osty

    alive = _top_level_osty({"alive": True, "hp": 25, "max_hp": 40, "block": 5})
    assert alive == {"alive": True, "hp": 25, "max_hp": 40, "block": 5, "name": "Osty"}
    assert _top_level_osty({"alive": False}) == {"alive": False}  # no name → no sprite
    assert _top_level_osty(None) is None


def test_fill_combat_lifts_osty_to_top_level(monkeypatch) -> None:
    """Regression: the Rust backend must surface Necrobinder's Osty the same
    way C# does — at top-level ``state['osty']`` — so the frontend renders it.
    The Rust wire nests it under ``player.osty`` (no name); ``_fill_combat``
    lifts it and injects the name the sprite lookup needs."""
    import sts2_rs

    wire = {
        "deck": {"hand": [], "draw_pile": [], "discard_pile": [], "exhaust_pile": []},
        "encounter": [{"kind": "Flyconid", "hp": 36, "max_hp": 36, "is_dead": False}],
        "player": {
            "hp": 50,
            "max_hp": 60,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "osty": {"alive": True, "hp": 25, "max_hp": 40, "block": 5},
        },
        "round": 1,
    }
    monkeypatch.setattr(sts2_rs, "state_to_dict", lambda cs: wire)

    class _Cs:
        def pending_choice(self):
            return None

    class _Rm:
        def combat_state(self):
            return _Cs()

        def available_actions(self):
            return []

    g = RustGame()
    g.rm = _Rm()  # type: ignore[assignment]
    state: dict = {"player": {}}
    g._fill_combat(state)

    assert state.get("osty") == {
        "alive": True,
        "hp": 25,
        "max_hp": 40,
        "block": 5,
        "name": "Osty",
    }


#: Per game build: a Necrobinder seed whose Neow offers Scroll Boxes, and the
#: two packs it rolls. v0.111.0 rehashed every seed (XxHash64), so ECHO rolls a
#: different Neow there; ECHO1's packs were checked against the v0.111.0 C#
#: headless (same ids, same order).
_SCROLL_BOXES = {
    "v0.107.1": (
        "ECHO",
        [["POKE", "REAP", "HIGH_FIVE"], ["REAVE", "DRAIN_POWER", "DEATH_MARCH"]],
    ),
    "v0.111.0": (
        "ECHO1",
        [["REAVE", "SCOURGE", "LETHALITY"], ["DEFILE", "NEGATIVE_PULSE", "NO_ESCAPE"]],
    ),
    # Captured from the v0.113.0 C# headless (ECHO1 rolls no Scroll Boxes
    # there). The engine still rolls v0.111.0's Neow offer on v0.113.0, so the
    # test is a strict xfail there (`pending_run_level`, tests/conftest.py).
    "v0.113.0": (
        "ECHO2",
        [["GRAVE_WARDEN", "PULL_AGGRO", "DEATHBRINGER"], ["DEFILE", "GRAVEBLAST", "BURY"]],
    ),
}


@pytest.mark.pending_run_level
def test_scroll_boxes_renders_bundle_select_and_adds_whole_pack() -> None:
    """Scroll Boxes (Necrobinder Neow, `_SCROLL_BOXES` seed) must surface as a
    ``bundle_select`` (pick 1 of 2 packs), not two single-card rewards.
    Picking a pack adds all 3 of its cards to the deck and advances — matching
    the real game's ``ScrollBoxes.AfterObtained``. (v0.107.1 dropped the
    LoseGold call, so the pick no longer costs gold; pre-v0.107.1 it did.)

    The pack contents are a golden snapshot byte-faithful to the C# headless
    (`GenerateRandomBundles`), captured against the v0.107.1 build (the
    v0.103.2 RNG/pool produced a different pair).
    """
    seed, expected_packs = _SCROLL_BOXES[sts2_client.GAME_VERSION]
    g = RustGame()
    g.start(character="Necrobinder", seed=seed, ascension=0)
    state = g.get_state()
    opts = normalize_options(state)
    sb = [i for i, o in enumerate(opts) if "SCROLL_BOX" in (o.get("label") or "").upper()]
    assert sb, f"{seed}/Necrobinder Neow should offer Scroll Boxes"
    deck_before = len((state.get("player") or {}).get("deck") or [])
    gold_before = (state.get("player") or {}).get("gold")

    state = g.act("choose_option", option_index=sb[0])
    assert state["decision"] == "bundle_select"
    packs = [[c["id"] for c in b["cards"]] for b in state["bundles"]]
    assert packs == expected_packs
    assert (state.get("player") or {}).get("gold") == gold_before, (
        "Scroll Boxes does not cost gold in v0.107.1 (LoseGold removed from AfterObtained)"
    )

    # Picking pack 0 adds its whole 3-card contents to the deck and advances
    # (no skip, no second screen).
    state = g.act("select_bundle", bundle_index=0)
    assert state["decision"] != "bundle_select"
    deck_after = len((state.get("player") or {}).get("deck") or [])
    assert deck_after == deck_before + 3, "the whole chosen pack is added to the deck"


# ── treasure rooms: the reveal payload ──────────────────────────────────────
# The engine resolves a chest ENTIRELY on room entry (`resolve_treasure_room`:
# roll relic, pay gold, obtain, jump to the map), for C# parity with AutoSlay's
# always-grabbing `TreasureRoomHandler`. No decision reaches the UI, so the
# workbench showed nothing at all for a treasure room until `room_reveal`.

#: Ironclad A0 seed whose greedy walk reaches an act-1 chest, per game build.
#: On v0.113.0 the SEEDTEST walk dies first, so that build walks CHEST1.
TREASURE_SEED = {"v0.113.0": "CHEST1"}.get(sts2_client.GAME_VERSION, "SEEDTEST")

#: What that first chest holds, per game build: (gold, relic). v0.111.0
#: rehashed every seed (XxHash64), so the same seed walks a different act 1.
#: The v0.113.0 pin is the C# headless driven through the same walk.
_TREASURE_CHEST = {
    "v0.107.1": (43, "HORN_CLEAT.title"),
    "v0.111.0": (47, "MANGO.title"),
    "v0.113.0": (42, "AMETHYST_AUBERGINE.title"),
}


def _walk_to_chest(g: RustGame, max_steps: int = 400) -> tuple[dict, list[dict]]:
    """Play greedily, preferring Treasure nodes; return (chest state, all states).

    Combat is "play the first playable card, else end turn" — enough to clear
    early Ironclad fights on this pinned seed.

    Everywhere else it takes option 0, with ONE exception: on a repeat page of
    the same event it takes the LAST option, which is the walk-away. A
    multi-page event escalates its price per page, so "always option 0" is not
    greedy there, it is a commitment to the worst branch — Tablet of Truth
    (GH #487) charges 3/6/12/24/(MaxHp-1) and leaves the run at 1 Max HP, which
    killed this fixture two floors later. The rule is deliberately structural
    rather than a check for that one event: the next faithfully-ported
    escalation must not silently break the chest tests either.
    """
    state = g.get_state()
    seen = [state]
    for _ in range(max_steps):
        was_event = (seen[-2].get("decision") if len(seen) > 1 else None) == "event_choice"
        if state.get("room_reveal"):
            return state, seen
        dec = state.get("decision")
        assert dec != "game_over", "run ended before reaching a chest"
        if dec == "map_select":
            choices = state.get("choices") or []
            assert choices, "no map moves offered"
            pick = next((c for c in choices if c.get("type") == "Treasure"), choices[0])
            state = g.act("select_map_node", col=pick["col"], row=pick["row"])
        elif dec == "combat_play":
            playable = [c for c in state["hand"] if c["can_play"]]
            if playable:
                # Aim at a LIVE enemy: once enemy 0 dies (a split slime's
                # parent), a play aimed at index 0 is a silent no-op, and the
                # walk spins on it until the step budget runs out.
                live = [e["index"] for e in state.get("enemies") or [] if e.get("hp", 0) > 0]
                target = live[0] if playable[0]["target_type"] == "AnyEnemy" else None
                state = g.act("play_card", card_index=playable[0]["index"], target_index=target)
            else:
                state = g.act("end_turn")
        else:
            opts = normalize_options(state)
            assert opts, f"no options at {dec}"
            pick = opts[-1] if (dec == "event_choice" and was_event) else opts[0]
            state = g.act(pick["action"], **pick["args"])
        assert state.get("type") != "error", f"engine error at {dec}: {state}"
        seen.append(state)
    pytest.fail("did not reach a treasure room within the step budget")


def test_walking_into_a_chest_reveals_what_it_held(run_level_pending: bool) -> None:
    # A build whose run level is still pending (tests/conftest.py
    # `PENDING_BUILDS`) has no pin: on v0.113.0 the C# run of this seed already
    # differs at Neow, so the chest the engine's walk reaches does not exist in
    # the real game. Every other build must be pinned.
    version = sts2_client.GAME_VERSION
    assert (version in _TREASURE_CHEST) != run_level_pending, (
        f"{version}: pin its chest in _TREASURE_CHEST (or its run level is pending)"
    )
    g = RustGame()
    g.start(character="Ironclad", seed=TREASURE_SEED, ascension=0)
    state, seen = _walk_to_chest(g)
    reveal = state["room_reveal"]
    assert reveal["kind"] == "treasure"
    assert reveal["act"] == 1
    # The reveal reports what the room really did: the gold the player gained
    # entering it, and a relic they now own (the panel reports, never gates).
    gold_before = (seen[-2].get("player") or {}).get("gold")
    assert (state.get("player") or {}).get("gold") - gold_before == reveal["gold"]
    owned = [r["name"] for r in (state.get("player") or {}).get("relics") or []]
    assert reveal["relic"]["name"] in owned
    # Pinned to the seed: at v0.107.1 this chest pays 43 gold and holds Horn Cleat.
    if version in _TREASURE_CHEST:
        assert (reveal["gold"], reveal["relic"]["name"]) == _TREASURE_CHEST[version]


def test_only_the_chest_room_reveals() -> None:
    g = RustGame()
    g.start(character="Ironclad", seed=TREASURE_SEED, ascension=0)
    _, seen = _walk_to_chest(g)
    # Every state up to the chest — Neow, combats, card rewards, map picks —
    # carries no reveal. Otherwise the panel would interrupt the whole run.
    assert [s for s in seen[:-1] if s.get("room_reveal")] == []


def test_leaving_the_chest_room_clears_the_reveal() -> None:
    g = RustGame()
    g.start(character="Ironclad", seed=TREASURE_SEED, ascension=0)
    state, _ = _walk_to_chest(g)
    # Polling the same position keeps it (the frontend dismisses by key, and a
    # re-render must not lose the panel mid-look).
    assert g.get_state().get("room_reveal") is not None
    # Moving on drops it — a second panel on the next room would be a lie.
    while state.get("decision") != "map_select":
        opts = normalize_options(state)
        state = g.act(opts[0]["action"], **opts[0]["args"])
    choices = state["choices"]
    state = g.act("select_map_node", col=choices[0]["col"], row=choices[0]["row"])
    assert state.get("room_reveal") is None


def test_skip_select_declines_an_up_to_n_enchant() -> None:
    """Kifuda's enchant screen is "up to 3" (``CardSelectorPrefs(prompt, 0, 3)``,
    not cancelable): the oracle offers ``skip_select`` for it, which confirms
    with no card. It must drop the whole optional batch, not be rejected
    (act-2 checkpoint branch ``Defect-1001-a0-act2#b0``, v0.111.0)."""
    g = _new()
    g.start(character="Defect", seed="1001", ascension=0)
    deck_before = repr(g.rm.master_deck())
    assert g.rm.force_add_relic("KIFUDA")
    assert g.rm.pending_deck_enchants > 0 and g.rm.pending_enchant_optional
    view = g.act("skip_select")
    assert not view.get("error"), view
    assert g.rm.pending_deck_enchants == 0
    assert repr(g.rm.master_deck()) == deck_before


def test_skip_select_declines_the_optional_sea_glass_grid() -> None:
    """Orobas' Sea Glass grid is ``FromSimpleGridForRewards`` with MinSelect 0:
    the oracle offers ``skip_select``, which leaves the deck unchanged. The
    adapter must route it to the engine's grant decline (``grant_deck_card`` at
    idx == option count) instead of rejecting it (Ironclad:4023, act 2)."""
    g = RustGame(target_convention="csharp")
    g.start(character="Ironclad", seed="4023", ascension=0)
    assert g.rm.force_add_relic("SEA_GLASS")
    assert g.rm.pending_card_grants > 0 and g.rm.pending_card_grant_min == 0
    deck_before = repr(g.rm.master_deck())
    view = g.act("skip_select")
    assert not view.get("error"), view
    assert g.rm.pending_card_grants == 0
    assert view.get("decision") != "card_select"
    assert repr(g.rm.master_deck()) == deck_before


def test_set_player_grafts_a_loadout_without_on_obtain_hooks() -> None:
    g = _new()
    view = g.set_player(
        deck=["BASH+1", "ANGER"],
        relics=["BURNING_BLOOD", "WHETSTONE"],
        potions=["FIRE_POTION"],
    )
    player = view["player"]
    deck = sorted((c["id"].removeprefix("CARD."), bool(c.get("upgraded"))) for c in player["deck"])
    assert deck == [("ANGER", False), ("BASH", True)]
    # Whetstone's on-obtain upgrade would have upgraded Anger.
    assert len(player["relics"]) == 2
    assert len(player["potions"]) == 1


def test_set_player_rejects_a_bad_id_without_a_partial_loadout() -> None:
    g = _new()
    before = g.get_state()["player"]["deck"]
    with pytest.raises(ValueError, match="unknown relic"):
        g.set_player(deck=["ANGER"], relics=["NOT_A_RELIC"])
    assert g.get_state()["player"]["deck"] == before
