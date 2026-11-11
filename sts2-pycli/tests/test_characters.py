"""Tests for all 5 characters."""

import pytest
import sts2_client
from sts2_client import Game

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]


def _neow_offer_names(state: dict) -> list[str]:
    opts = state.get("options") or state.get("event_options") or []
    return [(o.get("name") or o.get("title") or "") for o in opts]


class TestSeedCanonicalization:
    """The headless must canonicalize seeds like the real game
    (`SeedHelper.CanonicalizeSeed`: upper-case, O->0, I->1, trim) so a
    case/typo-variant seed yields the same run — and matches the sts2_rs
    engine, which already canonicalizes. Regression for the bug where the
    headless fed the raw string straight to RunState, so "seed2" and "SEED2"
    diverged here while the game treated them identically.
    """

    def test_case_variants_yield_same_neow_offers(self, game):
        base = _neow_offer_names(game.start(character="Ironclad", seed="SEED2"))
        assert base, "Neow offers should be present at run start"
        for variant in ("seed2", "sEeD2"):
            other = Game()
            try:
                got = _neow_offer_names(other.start(character="Ironclad", seed=variant))
            finally:
                other.close()
            assert got == base, f"seed {variant!r} should canonicalize to SEED2's run"

    def test_o_and_i_fold_to_digits(self, game):
        # 'O'->'0', 'I'->'1' folding (the game's seed alphabet excludes O/I).
        base = _neow_offer_names(game.start(character="Ironclad", seed="0FF1CE"))
        other = Game()
        try:
            got = _neow_offer_names(other.start(character="Ironclad", seed="OffIce"))
        finally:
            other.close()
        assert got == base


#: A Necrobinder seed whose Neow offers Scroll Boxes, per game build: v0.111.0
#: rehashed every seed (XxHash64), so ECHO rolls a different Neow there.
SCROLL_BOXES_SEED = {"v0.107.1": "ECHO", "v0.111.0": "ECHO1"}


class TestScrollBoxes:
    """The Scroll Boxes Neow relic (Necrobinder, `SCROLL_BOXES_SEED`) presents a
    bundle_select and, on pick, adds the whole chosen pack to the deck.
    Regression for the headless not draining the `ScrollBoxes.AfterObtained`
    deck-Adds (cards ended up in the RunState but in no pile, so the deck never
    grew). v0.107.1 removed the LoseGold call, so the pick no longer costs
    gold (pre-v0.107.1 it drained all of it)."""

    def test_picking_a_pack_adds_three_cards_to_deck(self, game):
        state = game.start(
            character="Necrobinder", seed=SCROLL_BOXES_SEED[sts2_client.GAME_VERSION]
        )
        opts = state.get("options") or state.get("event_options") or []
        sb = [
            i for i, o in enumerate(opts) if "Scroll Box" in (o.get("name") or o.get("title") or "")
        ]
        assert sb, "the Necrobinder Neow should offer Scroll Boxes on this seed"
        deck_before = (state.get("player") or {}).get("deck_size") or 0
        gold_before = (state.get("player") or {}).get("gold")

        state = game.act("choose_option", option_index=sb[0])
        assert state["decision"] == "bundle_select"
        assert len(state["bundles"]) == 2
        pack0 = [c.get("name") for c in state["bundles"][0]["cards"]]
        assert len(pack0) == 3

        state = game.act("select_bundle", bundle_index=0)
        p = state.get("player") or {}
        assert p.get("gold") == gold_before, "Scroll Boxes does not cost gold in v0.107.1"
        assert p.get("deck_size") == deck_before + 3, "the whole pack is added to the deck"
        deck_names = [c.get("name") for c in (p.get("deck") or [])]
        for name in pack0:
            assert name in deck_names, f"{name} should be in the deck after picking its pack"


@pytest.mark.oracle
def test_a_card_reward_after_the_scroll_boxes_pack_surfaces(monkeypatch):
    """On v0.111.0 seed 7709, Neow's Bones (option 2) offers Scroll Boxes and
    Lost Coffer. After the pack, the same RewardsSet.Offer obtains Lost Coffer,
    whose AfterObtained offers a card reward. The headless released the pack's
    continuation on the game thread, where that blocking card reward parked the
    thread that has to answer it: no reply to select_bundle (OracleHang, every
    character). The real game shows the card reward screen; so must we."""
    if not (sts2_client.oracle_available("v0.111.0") and sts2_client.oracle_built("v0.111.0")):
        pytest.skip("C# oracle for v0.111.0 unavailable")
    monkeypatch.setenv(sts2_client.REPLY_TIMEOUT_ENV, "60")
    prev = sts2_client.GAME_VERSION
    sts2_client.set_game_version("v0.111.0")
    g = Game()
    try:
        state = g.start(character="Ironclad", seed="7709")
        names = _neow_offer_names(state)
        assert names[2] == "Neow's Bones", names
        state = g.act("choose_option", option_index=2)
        assert state["decision"] == "bundle_select", state.get("decision")
        state = g.act("select_bundle", bundle_index=0)
        assert state["decision"] == "card_reward", state.get("decision")
        deck = (state.get("player") or {}).get("deck_size")
        state = g.act("select_card_reward", card_index=0)
        # The picked card, then the curse Neow's Bones adds once its Offer returns.
        assert (state.get("player") or {}).get("deck_size") == deck + 2
    finally:
        g.close()
        sts2_client.set_game_version(prev)


class TestCharacterMechanics:
    def test_defect_has_orbs(self, game):
        state = game.start(character="Defect", seed="dm1")
        game.skip_neow(state)
        state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
        assert "orbs" in state or "orb_slots" in state

    def test_regent_has_stars(self, game):
        state = game.start(character="Regent", seed="dm2")
        game.skip_neow(state)
        state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
        assert "stars" in state

    def test_necrobinder_has_osty(self, game):
        state = game.start(character="Necrobinder", seed="dm3")
        game.skip_neow(state)
        state = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
        assert "osty" in state
        assert "hp" in state["osty"]
        assert "alive" in state["osty"]


class TestFullRun:
    @pytest.mark.parametrize("character", CHARACTERS)
    @pytest.mark.slow
    def test_full_run(self, game, character):
        state = game.start(character=character, seed=f"full_{character.lower()}")
        steps = 0
        while steps < 2000:
            dec = state.get("decision", "")
            if dec == "game_over":
                assert "victory" in state
                return
            if state.get("type") == "error":
                state = game.act("proceed")
            elif dec == "combat_play":
                state = game.auto_combat(state)
            elif dec == "map_select":
                state = game.act(
                    "select_map_node",
                    col=state["choices"][0]["col"],
                    row=state["choices"][0]["row"],
                )
            elif dec == "event_choice":
                opts = [o for o in state["options"] if not o.get("is_locked")]
                state = (
                    game.act("choose_option", option_index=opts[0]["index"])
                    if opts
                    else game.act("leave_room")
                )
            elif dec == "card_reward":
                state = game.act("skip_card_reward")
            elif dec == "bundle_select":
                state = game.act("select_bundle", bundle_index=0)
            elif dec == "card_select":
                if state.get("min_select", 0) == 0:
                    state = game.act("skip_select")
                else:
                    state = game.act("select_cards", indices="0")
            elif dec == "rest_site":
                opts = [o for o in state["options"] if o.get("is_enabled")]
                state = game.act("choose_option", option_index=opts[0]["index"])
            elif dec == "shop":
                state = game.act("leave_room")
            else:
                state = game.act("proceed")
            steps += 1
        pytest.fail(f"{character} did not finish in {steps} steps")
