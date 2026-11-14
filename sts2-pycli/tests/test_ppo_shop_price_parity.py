"""Rust PPO shop prices match the Python training observations (pep299 fork).

Uses one real RunManager for both producers, including discounted prices,
sold-out slots and the legal-action/gold boundary. No game installation or
private checkpoints are needed.
"""

from __future__ import annotations

import json

import pytest
import sts2_rs

SEED = 984001164
DISCOUNTS = [
    ((), 1, 1),
    (("MEMBERSHIP_CARD",), 1, 2),
    (("THE_COURIER",), 4, 5),
    (("MEMBERSHIP_CARD", "THE_COURIER"), 4, 10),
]


def _enter_shop():
    """Debug-position at an actual parent node, then enter a real shop.

    The shop inventory is rolled by the normal movement handler, not mocked.
    This test fixture is intentionally artificial, not a natural-run sample.
    """
    rm = sts2_rs.RunManager(
        seed=SEED,
        character="Ironclad",
        ascension_level=10,
        game_version="v0.111.0",
    )
    rows = rm.full_map()["rows"]
    nodes = {(node["col"], node["row"]): node for row in rows for node in row}
    for row in rows:
        for parent in row:
            for child in parent["children"]:
                dest = nodes.get((child["col"], child["row"]))
                if dest is None or dest["type"] != "Shop":
                    continue
                rm.debug_place_on_map(
                    1, parent["col"], parent["row"], [], parent["row"] + 1, parent["row"] + 1
                )
                move = next(
                    (
                        a
                        for a in rm.available_actions()
                        if a["action"] == "move" and a["col"] == child["col"]
                    ),
                    None,
                )
                if move is not None:
                    rm.take_action(move)
                    assert rm.phase() == "shop"
                    assert rm.shop_inventory() is not None
                    return rm
    pytest.fail(f"seed {SEED}: no reachable shop edge in the generated act-1 map")


def _rust_obs(rm):
    return json.loads(sts2_rs.ppo_observe_json(rm, "shop", "shop", SEED, 0))


def _shop_rows(shop):
    for category in ("cards", "relics", "potions"):
        yield category, shop[category]


def _assert_shop_observations_agree(rm):
    """Byte-shape-equivalent inventory, including indices and availability."""
    python_shop = json.loads(json.dumps(rm.shop_inventory()))
    rust_shop = _rust_obs(rm)["shop"]
    assert rust_shop == python_shop
    # In particular, a sold-out slot must not become available after discount.
    for _, rows in _shop_rows(rust_shop):
        for entry in rows:
            if entry["price"] == 999_999:
                assert not entry["available"]
    return python_shop


@pytest.mark.parametrize("relic_ids,numerator,denominator", DISCOUNTS)
def test_discounted_shop_inventory_matches_python(relic_ids, numerator, denominator):
    rm = _enter_shop()
    baseline = rm.shop_inventory()
    # Change only owned relics AFTER inventory generation: each variant sees
    # the identical item ids, indices, on-sale flags and rolled base prices.
    rm.debug_set_loadout(relics=[*rm.relics(), *relic_ids])
    discounted = _assert_shop_observations_agree(rm)

    for category, items in _shop_rows(baseline):
        actual = discounted[category]
        assert len(items) == len(actual)
        for source, target in zip(items, actual, strict=True):
            assert source.get("id") == target.get("id")
            assert source["idx"] == target["idx"]
            assert source["available"] == target["available"]
            base = source["price"]
            expected = base if base <= 0 or base >= 999_999 else base * numerator // denominator
            assert target["price"] == expected, (category, source, target)

    base_removal = baseline["removal_cost"]
    expected_removal = base_removal * numerator // denominator
    assert discounted["removal_cost"] == expected_removal
    assert discounted["removal_used"] == baseline["removal_used"]

    # Gold changes legal action indices, never the list positions of the stock
    # or the meaning of a quoted price.
    rm.set_player(gold=0)
    zero_gold_actions = rm.available_actions()
    assert all(a["action"] == "leave_shop" for a in zero_gold_actions)
    assert _assert_shop_observations_agree(rm) == discounted


def test_sold_out_slots_stay_unavailable_and_price_parity_holds():
    rm = _enter_shop()
    rm.debug_set_loadout(relics=[*rm.relics(), "MEMBERSHIP_CARD"])
    rm.set_player(gold=9999)
    buy = next(a for a in rm.available_actions() if a["action"] == "shop_buy_card")
    bought_idx = buy["idx"]
    rm.take_action(buy)
    assert rm.phase() == "shop"
    shop = _assert_shop_observations_agree(rm)
    assert shop["cards"][bought_idx]["available"] is False
    assert shop["cards"][bought_idx]["price"] == 999_999
    assert all(
        not (a["action"] == "shop_buy_card" and a["idx"] == bought_idx)
        for a in rm.available_actions()
    )
