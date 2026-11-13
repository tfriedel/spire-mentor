"""Card-reward alternatives in the headless host (GH #1296, #1315).

The game's card reward screen has extra buttons beside the cards: Reroll
(Driftwood) and Sacrifice (Pael's Wing), built by
``CardRewardAlternative.Generate``. The host lists them as ``alternatives`` and
runs the pressed one through the game's own ``OnSelect``.
"""

from __future__ import annotations

import random

import pytest
import referee_spike
from test_referee_rng import Host, _project

pytestmark = pytest.mark.oracle


def _to_card_reward(host: Host, relic: str, seed: str) -> dict:
    """Start a run, add ``relic``, and play on to the first card reward."""
    state = host.send(cmd="start_run", character="Ironclad", seed=seed, ascension=0)
    held = [r["id"] for r in state["player"]["relics"]]
    host.send(cmd="set_player", relics=[*held, relic])
    rnd = random.Random(seed)
    for _ in range(600):
        if state.get("decision") == "card_reward" and state.get("cards"):
            return state
        assert state.get("decision") != "game_over", "run ended before a card reward"
        state = host.send(**referee_spike.choose(state, rnd))
    raise AssertionError("no card reward reached")


def _alternative(state: dict, option: str) -> int:
    ids = {a["id"]: a["index"] for a in state.get("alternatives", [])}
    assert option in ids, state.get("alternatives")
    return ids[option]


def test_driftwood_rerolls_a_card_reward_once():
    with Host(referee=False) as host:
        state = _to_card_reward(host, "DRIFTWOOD", "REROLL1")
        before = [c["id"] for c in state["cards"]]
        after = host.send(
            cmd="action",
            action="card_reward_alternative",
            args={"alternative_index": _alternative(state, "REROLL")},
        )
        assert after["decision"] == "card_reward"
        assert [c["id"] for c in after["cards"]] != before
        # A reward rerolls once (CardReward.Reroll sets CanReroll false).
        assert "REROLL" not in {a["id"] for a in after.get("alternatives", [])}
        deck = after["player"]["deck_size"]
        picked = host.send(cmd="action", action="select_card_reward", args={"card_index": 0})
        assert picked["player"]["deck_size"] == deck + 1


def test_without_driftwood_there_is_no_reroll():
    with Host(referee=False) as host:
        state = _to_card_reward(host, "ANCHOR", "REROLL1")
        assert "REROLL" not in {a["id"] for a in state.get("alternatives", [])}


def test_paels_wing_sacrifices_rewards_and_every_second_one_gives_a_relic():
    with Host(referee=False) as host:
        state = _to_card_reward(host, "PAELS_WING", "SACRIF1")
        relics = len(state["player"]["relics"])
        deck = state["player"]["deck_size"]
        rnd = random.Random("SACRIF1")
        sacrificed = 0
        for _ in range(800):
            if sacrificed == 2 or state.get("decision") == "game_over":
                break
            if state.get("decision") == "card_reward" and state.get("cards"):
                state = host.send(
                    cmd="action",
                    action="card_reward_alternative",
                    args={"alternative_index": _alternative(state, "SACRIFICE")},
                )
                sacrificed += 1
                assert state["player"]["deck_size"] == deck
                continue
            state = host.send(**referee_spike.choose(state, rnd))
            deck = state.get("player", {}).get("deck_size", deck)
        assert sacrificed == 2
        assert len(state["player"]["relics"]) == relics + 1


def test_the_view_lists_each_alternative_as_an_action():
    payload = {
        "type": "decision",
        "decision": "card_reward",
        "cards": [{"index": 0, "id": "CardModel.INFLAME", "name": "Inflame"}],
        "can_skip": True,
        "alternatives": [{"index": 0, "id": "REROLL", "name": "Reroll"}],
    }
    view = _project(payload)
    assert view["_unclassified"] == []
    assert view["legal_actions"] == [
        {"action": "select_card_reward", "args": {"card_index": 0}},
        {"action": "card_reward_alternative", "args": {"alternative_index": 0}},
        {"action": "skip_card_reward"},
    ]
