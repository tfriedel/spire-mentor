"""GH #1069: a potion reward that finds a full belt is a real decision.

The real reward screen leaves a TooFull ``PotionReward`` claimable, so the
player can discard a held potion and take it, or leave it. With
``interactive_potion_swap`` both engines surface that as a ``potion_swap``
decision (option 0 declines, option ``i + 1`` discards belt slot ``i``).
"""

from __future__ import annotations

from typing import Any

import pytest
import sts2_client

FULL_BELT = ["FIRE_POTION", "BLOCK_POTION", "ENERGY_POTION"]

#: The potion SWAPW's Wellspring Bottle rolls, per game build: (id, belt name).
#: v0.111.0 rehashed every seed (XxHash64), so the same seed rolls another one.
_BOTTLE_POTION = {
    "v0.107.1": ("STABLE_SERUM", "Stable Serum"),
    "v0.111.0": ("CLARITY", "Clarity Extract"),
    "v0.113.0": ("ATTACK_POTION", "Attack Potion"),  # C# headless, v0.113.0
}


def _belt(state: dict[str, Any]) -> list[str]:
    return [str(p.get("name")) for p in (state.get("player") or {}).get("potions") or [] if p]


def _wellspring_bottle(game: Any, *, swap: bool) -> dict[str, Any]:
    """Full belt, then Wellspring's Bottle: one ``OfferCustom(PotionReward)``."""
    state = game.start(seed="SWAPW", interactive_potion_swap=swap)
    game.skip_neow(state)
    game.set_player(potions=FULL_BELT)
    state = game.enter_room("event", event="WELLSPRING")
    assert state["decision"] == "event_choice"
    bottle = next(o["index"] for o in state["options"] if o.get("title") == "Bottle")
    return game.act("choose_option", option_index=bottle)


def test_csharp_full_belt_reward_surfaces_swap(game):
    state = _wellspring_bottle(game, swap=True)
    potion_id, potion_name = _BOTTLE_POTION[sts2_client.GAME_VERSION]
    assert state["decision"] == "potion_swap"
    assert state["potion"] == potion_id
    assert [o["discard_slot"] for o in state["options"]] == [None, 0, 1, 2]

    state = game.act("potion_swap", discard_slot=1)
    assert state["decision"] == "map_select"
    # Slot 1 (Block Potion) is discarded and the reward fills its slot.
    assert _belt(state) == ["Fire Potion", potion_name, "Energy Potion"]


def test_csharp_full_belt_reward_decline_keeps_belt(game):
    state = _wellspring_bottle(game, swap=True)
    assert state["decision"] == "potion_swap"
    state = game.act("potion_swap", discard_slot=None)
    assert state["decision"] == "map_select"
    assert _belt(state) == ["Fire Potion", "Block Potion", "Energy Potion"]


def test_csharp_swap_off_drops_silently(game):
    state = _wellspring_bottle(game, swap=False)
    assert state["decision"] == "map_select"


def test_rust_backend_renders_and_resolves_swap():
    sts2_rs = pytest.importorskip("sts2_rs")
    del sts2_rs
    from workbench.backends.rust_game import RustGame

    r = RustGame(target_convention="csharp")
    r.start(character="Ironclad", seed="SWAPW", ascension=0, interactive_potion_swap=True)
    r.rm.debug_set_potion_state(FULL_BELT, ["STABLE_SERUM"])
    state = r.get_state()
    assert state["decision"] == "potion_swap"
    assert [o["discard_slot"] for o in state["options"]] == [None, 0, 1, 2]

    r.act("potion_swap", discard_slot=1)
    assert r.rm.potions() == ["FirePotion", "StableSerum", "EnergyPotion"]
    assert r.rm.pending_potion_swap is None
