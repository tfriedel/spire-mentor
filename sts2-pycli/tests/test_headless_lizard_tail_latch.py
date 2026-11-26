"""Lizard Tail's spent bit on the wire.

Lizard Tail prevents one death per run and is then used up (C#
``LizardTail._wasUsed`` / ``IsUsedUp``). The host sent no counter for it, so a
fight rebuilt after the tail fired re-armed it, and the mirror read a lethal
hit as a 50% heal (Spire Bench seed 180001448).
"""

from __future__ import annotations


def _tail(state: dict) -> dict:
    relics = (state.get("player") or {}).get("relics") or []
    return next(r for r in relics if r.get("id") == "LIZARD_TAIL")


def test_lizard_tail_reads_spent_after_it_prevents_a_death(game):
    state = game.start(character="Ironclad", seed="TAIL1")
    game.skip_neow(state)
    r = game.set_player(
        hp=2, max_hp=80, deck=["BLOODLETTING"] * 6, relics=["LIZARD_TAIL"], potions=[]
    )
    assert r.get("type") != "error", r
    before = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    assert _tail(before).get("counter") == 0, _tail(before)
    after = game.play_card(0, None)  # Bloodletting: lose 3 HP at 2 HP
    assert (after.get("player") or {}).get("hp", 0) > 0, after.get("player")
    assert _tail(after).get("counter") == 1, _tail(after)
