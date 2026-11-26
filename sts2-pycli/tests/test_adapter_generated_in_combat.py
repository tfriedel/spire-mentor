"""A pile card the master deck does not hold is stamped generated_in_combat.

Thieving Hopper steals only cards with a DeckVersion (ThievingHopper.cs:203);
the wire carries none, so without the stamp the mirror counted a Vexing
Puzzlebox card as a steal candidate and stole a different card than the game
(seed 180000462).
"""

from sts2_hint import adapter


def _raw(i: int, cid: str) -> dict:
    return {"index": i, "id": f"CARD.{cid}", "cost": 1, "type": "Attack", "upgraded": False}


def _cards(ids: list[str]) -> list[dict]:
    return [adapter._translate_card(_raw(i, c), f"$.t[{i}]") for i, c in enumerate(ids)]


def test_a_card_missing_from_the_deck_is_stamped_generated() -> None:
    hand = _cards(["STRIKE_IRONCLAD"])
    draw = _cards(["BASH"])
    discard = _cards(["FLAME_BARRIER", "STRIKE_IRONCLAD"])
    deck = [_raw(0, "STRIKE_IRONCLAD"), _raw(1, "STRIKE_IRONCLAD"), _raw(2, "BASH")]
    adapter._stamp_generated_in_combat([hand, draw, discard, [], []], deck)
    assert discard[0]["combat_vars"]["generated_in_combat"] == 1
    for c in (*hand, *draw, discard[1]):
        assert "generated_in_combat" not in c.get("combat_vars", {})


def test_no_deck_stamps_nothing() -> None:
    discard = _cards(["FLAME_BARRIER"])
    adapter._stamp_generated_in_combat([[], [], discard, [], []], None)
    assert "generated_in_combat" not in discard[0].get("combat_vars", {})
