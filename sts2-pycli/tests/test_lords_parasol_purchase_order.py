"""Lord's Parasol buys the shop out after the other room-entry hooks have run.

``LordsParasol.AfterRoomEntered`` starts ``PurchaseEverything`` fire-and-forget.
In the real game (TestMode off) that body first awaits a process frame and two
``Cmd.Wait`` calls (LordsParasol.cs:40-52), so ``Hook.AfterRoomEntered`` runs
every later listener before anything is bought. Under TestMode nothing yields,
so the host used to buy the shop out inside the hook loop.

The probe: Maw Bank sits after Lord's Parasol, and seed ``parasol4``'s shop
stocks Dragon Fruit (+1 max HP whenever gold is gained). In the real game Maw
Bank's 12 gold arrives before Dragon Fruit is bought, so max HP stays 80. The
unpatched host bought Dragon Fruit first and ended at 81. See
``docs/experiments/2026-11-12-testmode-audit.md``.
"""

from __future__ import annotations

import pytest
import sts2_client


@pytest.mark.oracle
def test_parasol_buys_after_the_room_entry_hooks(game) -> None:
    if sts2_client.GAME_VERSION != "v0.111.0":
        pytest.skip("seed parasol4's shop stock was found on v0.111.0")
    state = game.start(character="Ironclad", seed="parasol4")
    game.skip_neow(state)
    r = game.set_player(
        hp=50, max_hp=80, gold=100, relics=["LORDS_PARASOL", "MAW_BANK"], potions=[]
    )
    assert r.get("type") != "error", r

    state = game.enter_room("shop")
    for _ in range(5):  # the bought card removal asks which card to remove
        if state.get("decision") != "card_select":
            break
        state = game.select_cards("0")

    player = state["player"]
    relics = [r["id"] for r in player["relics"]]
    assert "DRAGON_FRUIT" in relics, f"the shop no longer stocks Dragon Fruit: {relics}"
    assert player["gold"] == 112, "Maw Bank's 12 gold should still arrive"
    assert (player["hp"], player["max_hp"]) == (50, 80), (
        "Dragon Fruit was bought before Maw Bank's gold: the purchase ran inside "
        "the room-entry hooks"
    )
