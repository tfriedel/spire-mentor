"""WP3b: tactical columns (`ppo.e2e.tactical`): knapsack damage, killable, HP loss."""

from __future__ import annotations

import time

import pytest
import sts2_client
from ppo.e2e import factory
from ppo.e2e import tactical as TC


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _snap(hand, energy, enemies, block=0):
    return {
        "player": {"energy": energy, "block": block, "hp": 50},
        "hand": hand,
        "enemies": enemies,
    }


def _card(cid, cost, dmg, base_cost=None):
    return {
        "id": cid,
        "cost": cost,
        "base_cost": cost if base_cost is None else base_cost,
        "dmg": dmg,
    }


def _enemy(hp, block=0, intents=(), dead=False):
    return {"hp": hp, "block": block, "intents": list(intents), "is_dead": dead}


def test_knapsack_picks_the_best_cards_under_energy() -> None:
    hand = [
        _card("STRIKE_IRONCLAD", 1, {"0": 6, "1": 6}),
        _card("BASH", 2, {"0": 8, "1": 8}),
        _card("STRIKE_IRONCLAD", 1, {"0": 6, "1": 6}),
        _card("DEFEND_IRONCLAD", 1, {}),
    ]
    s = _snap(hand, 3, [_enemy(12), _enemy(30)])
    md, kill, loss = TC.tactical(s, [{"action": "play", "card_index": i} for i in range(4)])
    assert md == [14.0, 14.0]  # Bash + a Strike beats two Strikes (12)
    assert kill == [1.0, 0.0]
    assert loss == pytest.approx(0.0)
    # Two energy: Bash alone (8) loses to two Strikes (12).
    assert TC.max_damage(_snap(hand, 2, [_enemy(12)]), {0, 1, 2}) == [12.0]


def test_hit_counts_x_cost_zero_cost_and_unplayable() -> None:
    hand = [
        _card("TWIN_STRIKE", 1, {"0": 5}),  # 2 hits
        _card("WHIRLWIND", 0, {"0": 5}, base_cost=-1),  # X: hits = energy left
        _card("ANGER", 0, {"0": 6}),  # free: always played
        _card("STRIKE_IRONCLAD", 1, {"0": 6}),  # not playable below
    ]
    s = _snap(hand, 3, [_enemy(100)])
    # Twin Strike (10) + Whirlwind x2 (10) + Anger (6) = 26 > Whirlwind x3 (15) + 6.
    assert TC.max_damage(s, {0, 1, 2}) == [26.0]
    assert TC.hits_of("TWIN_STRIKE") == 2 and TC.hits_of("STRIKE_IRONCLAD") == 1


def test_block_and_dead_enemies_and_hp_loss() -> None:
    atk = {"kind": "Attack", "amount": 5, "effective_amount": 7, "times": 2}
    s = _snap(
        [_card("STRIKE_IRONCLAD", 1, {"0": 6})],
        1,
        [_enemy(5, block=2, intents=[atk]), _enemy(0, dead=True, intents=[atk])],
        block=4,
    )
    md, kill, loss = TC.tactical(s, [{"action": "play", "card_index": 0, "target_index": 0}])
    assert md == [6.0, 0.0]
    assert kill == [0.0, 0.0]  # 6 < 5 HP + 2 block; the dead enemy is never killable
    assert loss == pytest.approx(10.0)  # the dead enemy's intent does not count
    # A full telegraph overrides the displayed head intent.
    two = [[atk, atk], []]
    assert TC.hp_loss(s, two) == pytest.approx(24.0)


def test_fast_enough_for_the_worker() -> None:
    hand = [_card("STRIKE_IRONCLAD", 1, {"0": 6, "1": 6, "2": 6}) for _ in range(10)]
    s = _snap(hand, 5, [_enemy(40), _enemy(40), _enemy(40)])
    acts = [{"action": "play", "card_index": i} for i in range(10)]
    TC.tactical(s, acts)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        TC.tactical(s, acts)
    us = (time.perf_counter() - t0) / n * 1e6
    # Plan target 20-50 us; a 10-card, 5-energy, 3-enemy hand is a heavy case
    # and the laptop is shared, so the bound is loose.
    assert us < 400, us


@pytest.mark.parametrize("seed", [5, 6])
def test_on_a_real_fight(seed: int) -> None:
    from ppo.combat_ctx.snapshot import honest_snapshot

    cs = factory.scenario_state(
        seed=seed,
        character="Ironclad",
        deck=["TWIN_STRIKE"] * 3 + ["BASH"] * 2 + ["STRIKE_IRONCLAD"] * 5,
        hp=70,
        max_hp=80,
        relics=["BURNING_BLOOD"],
        encounter="SLIMES_WEAK",
        ascension=10,
    )
    snap = honest_snapshot(cs.belief_view(0), "monster")
    md, kill, loss = TC.tactical(snap, cs.legal_actions())
    assert len(md) == len(kill) == len(snap["enemies"])
    assert all(x >= 0 for x in md) and loss >= 0
    assert max(md) > 0
