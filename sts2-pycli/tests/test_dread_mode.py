"""v0.113.0 Dread mode on the C# headless host (GH #1346).

``GameMode.Dread`` exists from v0.113.0. A Dread run is an ascension-10 run
(the game hard-wires it, ``NDreadScreen.cs:584``) with a ``DreadRunModel``
that sets a per-act dominion from the act's boss.
"""

import pytest
import sts2_client

HAS_DREAD = sts2_client.GAME_VERSION in ("v0.113.0", "v0.114.0")
needs_dread = pytest.mark.skipif(not HAS_DREAD, reason="GameMode.Dread exists from v0.113.0")


@needs_dread
def test_a_dread_run_reports_its_mode_and_the_acts_dominion(game):
    state = game.start(character="Ironclad", seed="DREAD1", game_mode="dread")
    assert state.get("type") != "error", state
    ctx = state["context"]
    assert ctx["game_mode"] == "dread"
    assert ctx["dominion"].endswith("_DOMINION"), ctx["dominion"]
    # The game starts every Dread run at ascension 10, whatever was asked.
    assert ctx["ascension"] == 10


def test_a_standard_run_reports_no_game_mode(game):
    ctx = game.start(character="Ironclad", seed="DREAD1")["context"]
    assert "game_mode" not in ctx
    assert "dominion" not in ctx


def test_an_unknown_game_mode_is_refused(game):
    state = game.start(character="Ironclad", seed="DREAD1", game_mode="bogus")
    assert state.get("type") == "error"
    assert "game_mode" in state.get("message", "")


@pytest.mark.skipif(HAS_DREAD, reason="this build has Dread")
def test_dread_is_refused_on_a_build_without_it(game):
    state = game.start(character="Ironclad", seed="DREAD1", game_mode="dread")
    assert state.get("type") == "error"


# ----- get_state: the Dread context of the combat (GH #1346) -----------------

# Encounter ids with their `IsWeak` and what `WaterfallGiantDominion.
# GetExplosiveAmount` (WaterfallGiantDominion.cs:44-59) gives each: 3 with more
# than one monster, else 4 for a weak encounter, else 5 in a Monster room.
WATERFALL_CASES = [
    ("SHRINKER_BEETLE_WEAK", True, 4),
    ("TOADPOLES_WEAK", True, 3),
    ("NIBBITS_NORMAL", False, 3),
]


# A seed whose act-1 boss is the Waterfall Giant (DREAD1's is the Kin).
WATERFALL_SEED = "DREAD3"


@needs_dread
@pytest.mark.parametrize(("encounter", "weak", "explosive"), WATERFALL_CASES)
def test_get_state_reports_the_waterfall_giant_notes(game, encounter, weak, explosive):
    state = game.start(character="Ironclad", seed=WATERFALL_SEED, game_mode="dread")
    assert state["context"]["dominion"] == "WATERFALL_GIANT_DOMINION"
    game.skip_neow(state)
    game.enter_room("combat", encounter=encounter)
    st = game.get_state()
    assert st["game_mode"] == "dread"
    assert st["dominion"] == "WATERFALL_GIANT_DOMINION"
    assert st["dread_weak_encounter"] is weak
    assert st["dread_explosive"] == explosive
    # The amount is the one the dominion gave the roster.
    for m in st["monsters"]:
        amounts = [p["amount"] for p in m["powers"] if p["kind"] == "EXPLOSIVE_POWER"]
        assert amounts == [explosive], m


@needs_dread
def test_get_state_reports_no_explosive_under_another_dominion(game):
    state = game.start(character="Ironclad", seed="DREAD1", game_mode="dread")
    dominion = state["context"]["dominion"]
    assert dominion != "WATERFALL_GIANT_DOMINION", "pick another seed"
    game.skip_neow(state)
    game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    st = game.get_state()
    assert (st["game_mode"], st["dominion"]) == ("dread", dominion)
    assert st["dread_weak_encounter"] is True
    assert st["dread_explosive"] == 0


def test_a_standard_get_state_has_no_dread_context(game):
    state = game.start(character="Ironclad", seed="DREAD1")
    game.skip_neow(state)
    game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    st = game.get_state()
    for key in ("game_mode", "dominion", "dread_explosive", "dread_weak_encounter"):
        assert key not in st
