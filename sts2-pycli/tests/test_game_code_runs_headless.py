"""The game's own code runs in the host where it used to need hand-written copies.

The host used to replace seven game methods with copies written from the decomp,
because the originals crashed on a missing Godot node (GH #1285):
``DecimillipedeSegment.ReattachMove``, ``Vantom.DismemberMove``,
``SlumberingBeetle.AfterAddedToRoom`` and ``OnPlay`` of Whirlwind, Flash of
Steel, Slice and Neutralize. On v0.107.1 and v0.111.0 every scene-node access
in them is null-guarded or behind ``TestMode.IsOff``, so the copies are gone
and the originals run. Scripted fights with and without the copies gave the
same state at every step on both builds
(``docs/experiments/2026-11-12-host-patch-inventory.md``).

These tests pin each state effect, so a game update that brings a crash back
fails here instead of silently dropping the effect.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.oracle

_CARDS = ["WHIRLWIND", "FLASH_OF_STEEL", "SLICE", "NEUTRALIZE"]


def _fight(game, character: str, deck: list[str], encounter: str) -> dict[str, Any]:
    state = game.start(character=character, seed="ORIGINALS1")
    game.skip_neow(state)
    r = game.set_player(hp=999, max_hp=999, deck=deck, relics=[], potions=[])
    assert r.get("type") != "error", r
    state = game.enter_room("combat", encounter=encounter)
    assert state["decision"] == "combat_play", state
    return state


def _powers(enemy: dict[str, Any]) -> set[str]:
    return {p["id"] for p in enemy.get("powers") or []}


def _card(state: dict[str, Any], card_id: str) -> dict[str, Any] | None:
    return next((c for c in state["hand"] if c["id"] == f"CARD.{card_id}"), None)


def test_slumbering_beetle_enters_with_plating_and_slumber(game) -> None:
    state = _fight(game, "Ironclad", ["DEFEND_IRONCLAD"] * 5, "SLUMBERING_BEETLE_NORMAL")
    beetle = next(e for e in state["enemies"] if "BEETLE" in e["name"].upper().replace(" ", "_"))
    assert {"PLATING_POWER", "SLUMBER_POWER"} <= _powers(beetle), beetle


def test_the_four_attack_cards_resolve(game) -> None:
    # Every card is 0 cost except Whirlwind (X), so one hand plays them all.
    state = _fight(game, "Ironclad", _CARDS * 2, "SLUMBERING_BEETLE_NORMAL")
    seen: set[str] = set()
    for _ in range(4):
        for card_id in ["NEUTRALIZE", "SLICE", "FLASH_OF_STEEL", "WHIRLWIND"]:
            card = _card(state, card_id)
            if card is None or card_id in seen:
                continue
            before = state
            target = 0 if card.get("target_type") == "AnyEnemy" else None
            hp = [e["hp"] + e["block"] for e in before["enemies"]]
            hand = len(before["hand"])
            state = game.play_card(card["index"], target)
            seen.add(card_id)
            after = {e["index"]: e for e in state["enemies"]}
            if card_id == "WHIRLWIND":
                # X = all 3 energy, every enemy hit.
                assert state["energy"] == 0
                for i, e in enumerate(before["enemies"]):
                    if e["index"] in after:
                        assert after[e["index"]]["hp"] + after[e["index"]]["block"] < hp[i], card_id
                continue
            hit = after.get(before["enemies"][0]["index"])
            assert hit is None or hit["hp"] + hit["block"] < hp[0], (
                card_id,
                before["enemies"][0],
                hit,
            )
            if card_id == "NEUTRALIZE" and hit is not None:
                assert "WEAK_POWER" in _powers(hit)
            if card_id == "FLASH_OF_STEEL":
                assert len(state["hand"]) == hand  # played one, drew one
        if len(seen) == 4 or state.get("decision") != "combat_play":
            break
        state = game.end_turn()
    assert seen == set(_CARDS)


def test_vantom_dismember_adds_three_wounds(game) -> None:
    state = _fight(game, "Ironclad", ["DEFEND_IRONCLAD"] * 10, "VANTOM_BOSS")
    for _ in range(10):
        piles = state["hand"] + state["draw_pile"] + state["discard_pile"]
        if sum(c["id"] == "CARD.WOUND" for c in piles) >= 3:
            assert state["player"]["hp"] < 999  # its attack landed too
            return
        state = game.end_turn()
        assert state["decision"] == "combat_play", state
    pytest.fail("Vantom never used Dismember")


def test_decimillipede_segments_reattach(game) -> None:
    state = _fight(game, "Ironclad", _CARDS * 2 + ["STRIKE_IRONCLAD"] * 4, "DECIMILLIPEDE_ELITE")
    fewest = len(state["enemies"])
    for _ in range(300):
        if state.get("decision") != "combat_play":
            break
        alive = len(state["enemies"])
        if alive > fewest:
            return  # a felled segment came back
        fewest = min(fewest, alive)
        energy = state["energy"]
        playable = [c for c in state["hand"] if c.get("can_play") and c.get("cost", 0) <= energy]
        if playable:
            card = playable[0]
            target = 0 if card.get("target_type") == "AnyEnemy" else None
            state = game.play_card(card["index"], target)
        else:
            state = game.end_turn()
    pytest.fail(f"no segment reattached (fewest alive {fewest})")
