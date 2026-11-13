"""Unit tests for the lockstep live-game diff (GH #1287). No game process needed."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from live_game_diff import LockstepDivergence, LockstepGame, compare


def _map(choices: list[tuple[int, int]], **extra: Any) -> dict[str, Any]:
    return {
        "type": "decision",
        "decision": "map_select",
        "choices": [{"col": c, "row": r, "type": "Monster"} for c, r in choices],
        **extra,
    }


def test_compare_equal_decisions_have_no_difference() -> None:
    assert compare(_map([(0, 1)]), _map([(0, 1)])) == ([], [])


def test_compare_kind_mismatch_is_content() -> None:
    content, shape = compare(_map([(0, 1)]), {"type": "decision", "decision": "event_choice"})
    assert content == [{"path": "<kind>", "host": "map_select", "real": "event_choice"}]
    assert shape == []


def test_compare_value_on_both_sides_is_content_one_side_is_shape() -> None:
    host = _map([(0, 1)], gold=99, host_only=1)
    real = _map([(0, 2)], gold=99)
    content, shape = compare(host, real)
    assert content == [{"path": "choices[0].row", "host": 1, "real": 2}]
    assert shape == ["host-only:host_only"]


def test_compare_ignores_listed_paths() -> None:
    host = {**_map([(0, 1)]), "context": {"act_name": "Overgrowth", "floor": 1}}
    real = {**_map([(0, 1)]), "context": {"act_name": "?", "floor": 1}}
    assert compare(host, real) == ([], [])


def test_compare_undoes_the_wires_display_conventions() -> None:
    host = {
        "type": "decision",
        "decision": "combat_play",
        "hand": [{"id": "CARD.BASH", "name": "Strike", "upgraded": True}],
        "draw_pile": [{"id": "CARD.STRIKE_IRONCLAD", "index": 0}],
        "play_pile": None,
    }
    real = {
        "type": "decision",
        "decision": "combat_play",
        "hand": [{"id": "BASH", "name": "Strike+", "upgraded": True}],
        "draw_pile": [{"id": "STRIKE_IRONCLAD", "index": 5}],
        "play_pile": [],
    }
    assert compare(host, real) == ([], [])


def test_compare_still_sees_a_different_card() -> None:
    host = {"type": "decision", "decision": "combat_play", "hand": [{"id": "CARD.BASH"}]}
    real = {"type": "decision", "decision": "combat_play", "hand": [{"id": "ANGER"}]}
    content, _ = compare(host, real)
    assert content == [{"path": "hand[0].id", "host": "CARD.BASH", "real": "ANGER"}]


class _Fake:
    """Replays a scripted list of replies and records what it was sent."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sent: list[dict[str, Any]] = []
        self.proc = None

    def start(self, *_: Any, **__: Any) -> dict[str, Any]:
        self.sent.append({"cmd": "start_run"})
        return self.replies.pop(0)

    def send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(cmd)
        return self.replies.pop(0)


def test_lockstep_raises_on_first_difference_and_writes_the_step() -> None:
    host = _Fake([_map([(0, 1)]), _map([(1, 2)])])
    real = _Fake([_map([(0, 1)]), _map([(2, 2)])])
    trail = io.StringIO()
    game = LockstepGame(host, real, trail)  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    with pytest.raises(LockstepDivergence):
        game.act("select_map_node", col=0, row=1)
    rows = [json.loads(line) for line in trail.getvalue().splitlines()]
    assert len(rows) == 2
    assert rows[1]["content"] == [{"path": "choices[0].col", "host": 1, "real": 2}]
    assert game.divergence is not None
    assert game.divergence["kind"] == "content"


def test_lockstep_host_rejection_is_not_sent_to_the_game() -> None:
    first = _map([(0, 1)])
    host = _Fake([first, {"type": "error", "message": "bad"}, first])
    real = _Fake([first])
    game = LockstepGame(host, real, io.StringIO())  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    assert game.act("play_card", card_index=9)["type"] == "error"
    # The proceed after a rejection is answered by the host alone and compared
    # with the game's last decision, which never moved.
    assert game.act("proceed") == first
    assert real.sent == [{"cmd": "start_run"}]


def _shop(cards: list[dict[str, Any]], gold: int = 99, removal: int | None = None) -> dict:
    return {
        "type": "decision",
        "decision": "shop",
        "cards": cards,
        "relics": [],
        "potions": [],
        "card_removal_cost": removal,
        "player": {"gold": gold},
    }


def test_compare_reduces_sold_out_slots_to_the_stock_flag() -> None:
    host = _shop([{"name": "?.title", "type": "?", "cost": 83, "is_stocked": False}])
    real = _shop([{"name": "Gold Axe", "type": "Attack", "cost": 175, "is_stocked": False}])
    assert compare(host, real)[0] == []


def test_compare_sees_a_slot_sold_out_on_one_side_only() -> None:
    host = _shop([{"name": "Anger", "cost": 50, "is_stocked": True}])
    real = _shop([{"name": "Anger", "cost": 50, "is_stocked": False}])
    assert compare(host, real)[0] == [{"path": "cards[0].is_stocked", "host": True, "real": False}]


def test_lockstep_waits_out_a_shop_the_game_skipped() -> None:
    poor_shop = _shop([{"name": "Anger", "cost": 50, "is_stocked": True}], gold=10)
    after = _map([(0, 5)])
    host = _Fake([_map([(0, 4)]), poor_shop, after])
    real = _Fake([_map([(0, 4)]), after])
    game = LockstepGame(host, real, io.StringIO())  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    assert game.act("select_map_node", col=0, row=4) == poor_shop
    # The game already left; the host's leave_room is answered by that reply.
    assert game.act("leave_room") == after
    assert [c.get("action") for c in real.sent] == [None, "select_map_node"]


def test_lockstep_buying_in_a_shop_the_game_skipped_diverges() -> None:
    poor_shop = _shop([{"name": "Anger", "cost": 50, "is_stocked": True}], gold=10)
    host = _Fake([_map([(0, 4)]), poor_shop, poor_shop])
    real = _Fake([_map([(0, 4)]), _map([(0, 5)])])
    game = LockstepGame(host, real, io.StringIO())  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    game.act("select_map_node", col=0, row=4)
    with pytest.raises(LockstepDivergence):
        game.act("buy_card", card_index=0)


def test_compare_ignores_the_order_of_map_choices() -> None:
    assert compare(_map([(3, 4), (2, 4)]), _map([(2, 4), (3, 4)])) == ([], [])


def test_compare_cuts_keywords_to_what_the_bridge_carries() -> None:
    host = {
        "type": "decision",
        "decision": "combat_play",
        "hand": [{"id": "GREED", "keywords": ["Eternal", "Unplayable"], "enchantment": "Swift"}],
    }
    real = {
        "type": "decision",
        "decision": "combat_play",
        "hand": [{"id": "GREED", "keywords": ["unplayable"], "enchantment": "SWIFT"}],
    }
    assert compare(host, real)[0] == []


def test_compare_cuts_deck_keywords_to_the_deck_flags() -> None:
    host = _map(
        [(0, 1)], player={"deck": [{"id": "CARD.CLUMSY", "keywords": ["Unplayable", "Ethereal"]}]}
    )
    real = _map([(0, 1)], player={"deck": [{"id": "CARD.CLUMSY", "keywords": ["Ethereal"]}]})
    assert compare(host, real)[0] == []


def test_compare_counts_floors_over_the_run() -> None:
    host = _map([(3, 0)], context={"act": 2, "floor": 1, "total_floor": 18})
    real = _map([(3, 0)], context={"act": 2, "floor": 18})
    assert compare(host, real)[0] == []


def test_compare_notes_an_unlocalized_host_name() -> None:
    host = {
        "type": "decision",
        "decision": "combat_play",
        "player_powers": [{"name": "SPEED_POTION_POWER.title"}],
    }
    real = {
        "type": "decision",
        "decision": "combat_play",
        "player_powers": [{"name": "Speed Potion"}],
    }
    content, shape = compare(host, real)
    assert content == []
    assert shape == ["host-unlocalized:player_powers[].name=SPEED_POTION_POWER.title"]


def _fight(enemies: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"type": "decision", "decision": "combat_play", "enemies": enemies, **extra}


def test_compare_moves_the_games_corpses_out_of_the_enemy_list() -> None:
    host = _fight(
        [{"index": 0, "name": "The Obscura", "hp": 48}],
        dead_enemies=[{"name": "Parafright", "hp": 0, "is_dead": True}],
    )
    real = _fight(
        [
            {"index": 0, "name": "Parafright", "hp": 0, "is_dead": True},
            {"index": 1, "name": "The Obscura", "hp": 48, "is_dead": False},
        ]
    )
    content, shape = compare(host, real)
    assert content == []
    assert shape == ["real-only:enemies[].is_dead"]


def test_lockstep_sends_the_game_its_own_number_for_the_target() -> None:
    corpse_first = [
        {"index": 0, "name": "Parafright", "hp": 0, "is_dead": True},
        {"index": 1, "name": "The Obscura", "hp": 48, "is_dead": False},
    ]
    host_view = _fight(
        [{"index": 0, "name": "The Obscura", "hp": 48}],
        dead_enemies=[{"name": "Parafright", "hp": 0, "is_dead": True}],
    )
    host = _Fake([host_view, host_view])
    real = _Fake([_fight(corpse_first), _fight(corpse_first)])
    game = LockstepGame(host, real, io.StringIO())  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    game.act("play_card", card_index=0, target_index=0)
    assert host.sent[-1]["args"] == {"card_index": 0, "target_index": 0}
    assert real.sent[-1]["args"] == {"card_index": 0, "target_index": 1}


def test_compare_notes_an_unlocalized_monster_name() -> None:
    host = _fight([{"index": 0, "name": "DECIMILLIPEDE_SEGMENT_FRONT.name"}])
    real = _fight([{"index": 0, "name": "Decimillipede", "is_dead": False}])
    content, shape = compare(host, real)
    assert content == []
    assert "host-unlocalized:enemies[].name=DECIMILLIPEDE_SEGMENT_FRONT.name" in shape


def test_compare_skips_what_the_bridge_cannot_see_on_a_curse_choice() -> None:
    host = {
        "type": "decision",
        "decision": "card_select",
        "source": "FromChooseACardScreen",
        "min_select": 1,
        "max_select": 1,
        "player": {"hp": 80, "block": 12},
    }
    real = {
        "type": "decision",
        "decision": "card_select",
        "min_select": 0,
        "max_select": 1,
        "player": {"hp": 80, "block": 0},
    }
    assert compare(host, real)[0] == []


def test_compare_still_sees_block_in_combat() -> None:
    host = _fight([], player={"block": 12})
    real = _fight([], player={"block": 0})
    assert compare(host, real)[0] == [{"path": "player.block", "host": 12, "real": 0}]


def _sphere(cells: list[tuple[int, int]], proceed: bool = False) -> dict[str, Any]:
    options = [{"index": i, "title": f"Cell ({x},{y})"} for i, (x, y) in enumerate(cells)]
    if proceed:
        options.append({"index": len(options), "title": "Proceed"})
    return {"type": "decision", "decision": "event_choice", "options": options}


def test_lockstep_digs_the_games_crystal_sphere_the_hosts_way() -> None:
    grid = [(0, 5), (1, 3), (4, 3), (7, 3), (1, 6)]
    reward = {"type": "decision", "decision": "card_reward", "cards": [{"id": "ANGER"}]}
    host = _Fake([_map([(0, 1)]), reward])
    real = _Fake(
        [
            _map([(0, 1)]),
            _sphere(grid),
            _sphere(grid[2:]),
            _sphere(grid[3:]),
            _sphere(grid[4:], proceed=True),
            reward,
        ]
    )
    game = LockstepGame(host, real, io.StringIO())  # type: ignore[arg-type]
    game.start("Ironclad", seed="5")
    assert game.act("choose_option", option_index=1) == reward
    picks = [c["args"]["option_index"] for c in real.sent[2:]]
    # (1,3), (4,3), (7,3) by their index on each screen, then proceed.
    assert picks == [1, 0, 0, 1]


def test_compare_notes_an_unformatted_host_title() -> None:
    def event(title: str) -> dict[str, Any]:
        return {"type": "decision", "decision": "event_choice", "options": [{"title": title}]}

    content, shape = compare(event("Insert {Rarity} Potion"), event("Insert Common Potion"))
    assert content == []
    assert shape == ["host-unformatted:options[].title=Insert {Rarity} Potion"]
    content, _ = compare(event("Insert {Rarity} Potion"), event("Take the Relic"))
    assert content != []


def test_compare_costs_an_x_card_in_hand_as_the_host_does() -> None:
    def hand(cost: int) -> dict[str, Any]:
        card = {"id": "HEAVENLY_DRILL", "base_cost": -1, "cost": cost}
        return {"type": "decision", "decision": "combat_play", "hand": [card]}

    assert compare(hand(0), hand(6))[0] == []


def test_compare_ignores_what_a_played_x_card_cost() -> None:
    def discard(cost: int) -> dict[str, Any]:
        card = {"id": "HEAVENLY_DRILL", "base_cost": -1, "cost": cost}
        return {"type": "decision", "decision": "combat_play", "discard_pile": [card]}

    assert compare(discard(4), discard(0))[0] == []


def test_compare_matches_an_enchantment_name_to_its_id() -> None:
    def pile(enchantment: str) -> dict[str, Any]:
        card = {"id": "STRIKE_IRONCLAD", "enchantment": enchantment}
        return {"type": "decision", "decision": "combat_play", "draw_pile": [card]}

    assert compare(pile("Perfect Fit"), pile("PERFECT_FIT"))[0] == []
    assert compare(pile("Perfect Fit"), pile("SWIFT"))[0] != []
