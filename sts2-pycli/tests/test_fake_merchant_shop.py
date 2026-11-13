"""GH #1294: the Fake Merchant's relics are a shop, and a Foul Potion fights it.

The event has no options (``FakeMerchant.GenerateInitialOptions`` is empty, so
the game marks it finished on entry); its page is a merchant inventory of six
fake relics, and Proceed opens the map. A Foul Potion thrown at it starts the
fight whose rewards are the Rug and the relics still stocked
(``FakeMerchant.FoulPotionThrown``).
"""

from __future__ import annotations

from typing import Any


def _fake_merchant(game: Any, **player: Any) -> dict[str, Any]:
    state = game.start(seed="FAKEM")
    game.skip_neow(state)
    game.set_player(gold=500, **player)
    return game.enter_room("event", event="FAKE_MERCHANT")


def _relic_ids(state: dict[str, Any]) -> list[str]:
    return [str(r.get("id")) for r in state["player"]["relics"]]


def test_fake_merchant_offers_its_relics(game):
    state = _fake_merchant(game)
    assert state["decision"] == "shop"
    assert state["event_name"] == "FAKE_MERCHANT"
    assert len(state["relics"]) == 6
    assert all("FAKE" in str(r["id"]) for r in state["relics"])
    assert state["cards"] == [] and state["potions"] == []

    pick = state["relics"][0]
    gold = state["player"]["gold"]
    state = game.act("buy_relic", relic_index=pick["index"])
    assert state["decision"] == "shop"
    assert state["player"]["gold"] == gold - pick["cost"]
    assert pick["id"] in _relic_ids(state)
    assert not state["relics"][0]["is_stocked"]

    state = game.act("leave_room")
    assert state["decision"] == "map_select"


def test_foul_potion_starts_the_fake_merchant_fight(game):
    state = _fake_merchant(game, potions=["FOUL_POTION"])
    assert state["decision"] == "shop"
    state = game.act("use_potion", potion_index=0)
    assert state["decision"] == "combat_play"
    assert [e["name"] for e in state["enemies"]]


def test_foul_potion_at_the_merchant_pays_gold(game):
    """The same potion in a merchant room is drunk for gold (FoulPotion.OnUse)."""
    state = game.start(seed="FAKEM")
    game.skip_neow(state)
    game.set_player(gold=100, potions=["FOUL_POTION"])
    state = game.enter_room("shop")
    assert state["decision"] == "shop"
    state = game.act("use_potion", potion_index=0)
    assert state["player"]["gold"] > 100
    assert not [p for p in state["player"]["potions"] if p]
