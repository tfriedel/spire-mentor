"""GH #1286: Entropy's turn-start pick is the player's.

``EntropyPower.AfterPlayerTurnStart`` asks the player which card in hand to
transform (``CardSelectCmd.FromHand``). The host's default stand-in transforms
the first card in hand instead (the Rust engine mirrors it for parity). With
``STS2_INTERACTIVE_TURN_START_PICKS=1`` (always on in referee mode) the game's
own hook runs and the pick is a ``card_select`` after ``end_turn``. A replay
census run (seed 4V8GTTF2LRE6, Regent A10) transformed another Strike than
the player did and stopped two plays later.
"""

from __future__ import annotations

from typing import Any

import pytest
from sts2_client import Game

DECK = ["ENTROPY", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD", "BASH"]  # all drawn


def _entropy_turn(game: Any) -> dict[str, Any]:
    state = game.start(seed="ENTRO")
    game.skip_neow(state)
    game.set_player(deck=DECK, hp=80, max_hp=80)
    state = game.enter_room("monster")
    assert state["decision"] == "combat_play"
    hand = state["hand"]
    entropy = next(c["index"] for c in hand if c["id"] == "CARD.ENTROPY")
    state = game.act("play_card", card_index=entropy)
    return game.act("end_turn")


@pytest.fixture
def interactive_game(monkeypatch):
    monkeypatch.setenv("STS2_INTERACTIVE_TURN_START_PICKS", "1")
    g = Game()
    yield g
    g.close()


def test_entropy_pick_is_a_card_select(interactive_game):
    state = _entropy_turn(interactive_game)
    assert state["decision"] == "card_select"
    assert state["min_select"] == 1 and state["max_select"] == 1
    offered = [c["id"] for c in state["cards"]]
    assert len(offered) >= 2
    assert offered.count("CARD.BASH") == 1
    bash = offered.index("CARD.BASH")
    state = interactive_game.act("select_cards", indices=str(bash))
    assert state["decision"] == "combat_play"
    hand = [c["id"] for c in state["hand"]]
    assert len(hand) == len(offered)  # one card swapped, none lost
    assert "CARD.BASH" not in hand  # the card the player chose, not the first


def test_default_host_keeps_the_stand_in(game):
    state = _entropy_turn(game)
    assert state["decision"] == "combat_play"
