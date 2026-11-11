"""Tests for events."""

import pytest
import sts2_client
from sts2_client import Game


class TestNeowEvent:
    def test_neow_is_first_event(self, game):
        state = game.start(seed="ne1")
        assert state["decision"] == "event_choice"
        assert "Neow" in str(state.get("event_name", ""))

    def test_neow_options(self, game):
        state = game.start(seed="ne2")
        for opt in state["options"]:
            assert "title" in opt
            assert isinstance(opt["title"], str)
            assert "is_locked" in opt

    def test_neow_option_vars(self, game):
        state = game.start(seed="ne3")
        for opt in state["options"]:
            if opt.get("vars"):
                for k, v in opt["vars"].items():
                    assert isinstance(v, (int, float))

    def test_choose_neow(self, game):
        state = game.start(seed="ne4")
        opts = [o for o in state["options"] if not o.get("is_locked")]
        state = game.act("choose_option", option_index=opts[0]["index"])
        assert state.get("decision") is not None


class TestEventDescriptions:
    def test_no_ismultiplayer_tag(self, game):
        state = game.start(seed="ed1")
        for opt in state.get("options", []):
            d = opt.get("description") or ""
            assert "IsMultiplayer" not in d


class TestEventCombatBridge:
    """Bead sts2-rs-ck9z6 (GH #166): event options that fire
    ``EnterCombatWithoutExitingEvent`` now run a REAL nested combat instead of
    short-circuiting to an instant-victory reward."""

    def test_battleworn_dummy_enters_and_resolves_real_combat(self, game):
        game.start(character="Ironclad", seed="bwd1")
        # Big HP pool so the fight can't kill us before it resolves.
        game.set_player(current_hp=200, max_hp=200)
        state = game.enter_room("event", event="BATTLEWORN_DUMMY")
        assert state.get("decision") == "event_choice"
        # Setting 1 (BattleFriendV1) — choosing it must ENTER combat, not
        # instantly resolve to a reward / map.
        state = game.choose_option(0)
        assert state.get("decision") == "combat_play", (
            f"event option must enter a real combat, got {state.get('decision')}"
        )
        # The existing combat loop drives it to completion and returns to the run.
        state = game.auto_play_combat(state)
        assert state.get("decision") != "combat_play", "combat must resolve"

    def test_the_lantern_key_fight_enters_real_combat(self, game):
        game.start(character="Ironclad", seed="tlk1")
        game.set_player(current_hp=200, max_hp=200)
        state = game.enter_room("event", event="THE_LANTERN_KEY")
        assert state.get("decision") == "event_choice"
        # Page 0 "Keep the Key" opens the page-1 Fight option (multi-page).
        state = game.choose_option(1)
        assert state.get("decision") == "event_choice"
        # Page 1 "Fight" enters the MysteriousKnight combat.
        state = game.choose_option(0)
        assert state.get("decision") == "combat_play", (
            f"the Fight option must enter combat, got {state.get('decision')}"
        )
        state = game.auto_play_combat(state)
        assert state.get("decision") != "combat_play"


@pytest.mark.oracle
def test_the_hunt_reward_after_the_battleworn_dummy_surfaces(monkeypatch):
    """On v0.111.0, BattlewornDummy.Resume offers the fight's rewards together
    with the combat room's ExtraRewards: The Hunt's Fatal kill adds a card
    reward. The headless resumed the event on the game thread, where the
    blocking card-reward selector parked the thread that has to answer it: no
    reply to the winning play_card (OracleHang, loadout Silent seed 8529), then
    a silent skip after the selector's 300 s timeout. The real game shows the
    card reward screen; so must we."""
    if not (sts2_client.oracle_available("v0.111.0") and sts2_client.oracle_built("v0.111.0")):
        pytest.skip("C# oracle for v0.111.0 unavailable")
    monkeypatch.setenv(sts2_client.REPLY_TIMEOUT_ENV, "60")
    prev = sts2_client.GAME_VERSION
    sts2_client.set_game_version("v0.111.0")
    g = Game()
    try:
        g.start(character="Silent", seed="bwh1")
        g.set_player(hp=200, max_hp=200, deck=["THE_HUNT+1"] * 15)
        state = g.enter_room("event", event="BATTLEWORN_DUMMY")
        assert state.get("decision") == "event_choice"
        state = g.choose_option(0)
        for _ in range(40):
            if state.get("decision") != "combat_play":
                break
            played = g.play_card(0, target_index=0)
            state = g.end_turn() if played.get("type") == "error" else played
        assert state.get("decision") == "card_reward", state.get("decision")
        deck = (state.get("player") or {}).get("deck_size")
        state = g.select_card_reward(0)
        assert state.get("decision") not in (None, "card_reward"), state.get("decision")
        assert (state.get("player") or {}).get("deck_size") == deck + 1
    finally:
        g.close()
        sts2_client.set_game_version(prev)
