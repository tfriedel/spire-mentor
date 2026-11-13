"""The Blitz merchant (v0.113.0): free wares, a fixed number of picks.

No net was trained on Blitz, so the answer is an admitted rule. These pin the
two things the rule exists for: it does not walk out on free picks (what the
generic shop builder did -- every ware encodes as the same "none" token), and
it does not take a ware that adds a Curse.
"""

from __future__ import annotations

from typing import Any

import realgame_advisor as ra
import realgame_ppo as rp


def _ware(index: int, kind: str, *, cursed: bool = False, enabled: bool = True) -> dict[str, Any]:
    return {
        "index": index,
        "id": "blitz_choice",
        "action": "blitz_choice",
        "label": f"blitz:{kind}",
        "enabled": enabled,
        "price": 0,
        "item": {"blitz_kind": kind, "cursed": cursed, "rare": False, "title": kind},
    }


def _shop(*wares: dict[str, Any], remaining: int | None = 3) -> dict[str, Any]:
    state: dict[str, Any] = {
        "type": "shop",
        "options": [
            {"index": 0, "id": "leave_shop", "action": "leave_shop", "label": "Leave shop"},
            *wares,
        ],
        "player_gold": 0,
        "run_state": {"character": "Ironclad"},
    }
    if remaining is not None:
        state["blitz_choices_remaining"] = remaining
    return state


def test_a_standard_shop_is_not_blitz() -> None:
    state = {"type": "shop", "options": [{"index": 0, "action": "leave_shop"}, {"index": 1, "action": "buy_card"}]}
    assert not rp.is_blitz_shop(state)


def test_takes_a_free_ware_instead_of_leaving() -> None:
    state = _shop(_ware(1, "Glam", cursed=True), _ware(2, "CardRewards"), _ware(3, "Relic"))
    assert rp.is_blitz_shop(state)
    assert rp.blitz_shop_choice(state)[0] == 2


def test_leaves_when_only_cursed_wares_remain() -> None:
    state = _shop(_ware(1, "TwoRelics", cursed=True))
    assert rp.blitz_shop_choice(state) == (0, "only cursed wares left")


def test_leaves_when_no_picks_remain() -> None:
    state = _shop(_ware(1, "CardRewards"), remaining=0)
    assert rp.blitz_shop_choice(state)[0] == 0


def test_the_autopilot_lane_answers_with_the_same_rule() -> None:
    import random

    state = _shop(_ware(1, "CardRewards"), _ware(2, "Relic"))
    # No policy cache: the Blitz rule answers before any net is consulted.
    action, extra = rp.pick_noncombat_action_ppo(state, random.Random(0), None)  # type: ignore[arg-type]
    assert action == {"action": "choose", "index": 1}
    assert extra is None


def test_the_advisor_says_it_is_not_a_ranking() -> None:
    answer = ra._blitz_shop_answer(_shop(_ware(1, "Glam", cursed=True), _ware(2, "Potions")))
    assert answer["best"]["index"] == 2
    assert "NOT a recommendation" in answer["note"]
    assert answer["hand_back"]
    assert [o["index"] for o in answer["options"] if o["is_best"]] == [2]
