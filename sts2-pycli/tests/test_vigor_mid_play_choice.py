"""A card that attacks and then asks has spent its Vigor before the pick.

C# runs `AfterAttack` when the card's `AttackCommand` ends, before its
`CardSelectCmd` (Neow's Fury, Headbutt, Dagger Throw, ...): Vigor is gone while
the player picks. The engine spent it only when the card finished, so the pick
showed Vigor the real game had already spent (Spire Bench corpus, ~17 runs).
"""

from __future__ import annotations

import pytest
import sts2_client
import sts2_rs
from ppo.e2e import factory


@pytest.fixture(autouse=True)
def _campaign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the e2e lane's build, as a campaign process does: the factory
    refuses any other selection, so `--game-version v0.113.0` failed here
    (GH #1781)."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)


def _vigor(cs) -> int:
    p = sts2_rs.state_to_dict(cs)["player"]
    return sum(int(x.get("amount", 0)) for x in p.get("powers", []) if x.get("name") == "VIGOR")


def _play(cs, card_id: str) -> None:
    hand = sts2_rs.state_to_dict(cs)["deck"]["hand"]
    a = next(
        x
        for x in cs.legal_actions()
        if x["action"] == "play" and hand[x["card_index"]]["id"] == card_id
    )
    cs.apply(
        a | ({"target_index": 0} if "target_index" in a or card_id != "DEFEND_IRONCLAD" else {})
    )


def test_neows_fury_spends_vigor_before_its_discard_pick() -> None:
    cs = factory.scenario_state(
        seed=5,
        character="Ironclad",
        deck=["NEOWS_FURY", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
        hp=80,
        max_hp=80,
        relics=["AKABEKO"],
        encounter="NIBBITS_WEAK",
        ascension=10,
    )
    assert _vigor(cs) == 8
    _play(cs, "DEFEND_IRONCLAD")
    _play(cs, "NEOWS_FURY")
    assert cs.pending_choice(), "Neow's Fury opens its discard pick"
    assert _vigor(cs) == 0
