"""GH #1286: the parts of the replay harness that need no game.

Which of several identical cards a log's pick means, and how the profile
search tells that one replay got further than another.
"""

from __future__ import annotations

from typing import Any

import host_replay as hr
import host_replay_profile as hp


def _tracker(rows: list[dict[str, Any]]) -> hr.Tracker:
    t = hr.Tracker({"character": "DEFECT", "starting_deck": []})
    for r in rows:
        t._move_card(r)
    return t


def test_a_card_picked_from_the_discard_pile_is_ranked_among_its_copies():
    # Hologram takes a Claw back from the discard pile. The copies differ by
    # their enchantment only, and where each lies decides the next shuffle.
    t = _tracker(
        [
            {"t": "combat_start"},
            {"t": "play", "c": 1, "id": "CLAW", "result_pile": "discard"},
            {"t": "play", "c": 2, "id": "ZAP", "result_pile": "discard"},
            {"t": "play", "c": 3, "id": "CLAW", "result_pile": "discard"},
            {"t": "flush", "flushed_c": [4]},
            {"t": "draw", "c": 4, "id": "CLAW"},
            {"t": "discard", "c": 4, "id": "CLAW"},
        ]
    )
    assert [t.hand_rank(c, 3) for c in (1, 3, 4)] == [0, 1, 2]
    assert t.hand_rank(2, 1) == 0


def test_a_shuffle_empties_the_discard_pile():
    t = _tracker(
        [
            {"t": "combat_start"},
            {"t": "play", "c": 1, "id": "CLAW", "result_pile": "discard"},
            {"t": "play", "c": 3, "id": "CLAW", "result_pile": "discard"},
            {"t": "shuffle", "order_c": [3, 1]},
        ]
    )
    assert t.discard == []
    assert [t.hand_rank(c, 2) for c in (3, 1)] == [0, 1]  # the draw pile's order


def test_a_played_power_does_not_reach_the_discard_pile():
    t = _tracker(
        [{"t": "combat_start"}, {"t": "play", "c": 1, "id": "LOOP", "result_pile": "none"}]
    )
    assert t.discard == []


def test_a_run_that_does_not_start_is_behind_one_that_does():
    stuck = {"desync": "rng_start", "progress": [1, 0]}
    started = {"desync": "event_options", "progress": [1, 0], "mismatch": 2}
    assert hp._score(started) > hp._score(stuck)


def test_less_amiss_at_the_same_stop_is_further():
    two = {"desync": "floor_state", "progress": [26, 4], "mismatch": 2}
    one = {"desync": "floor_state", "progress": [26, 4], "mismatch": 1}
    later = {"desync": "card_reward_offer", "progress": [124, 22], "mismatch": 3}
    assert hp._score(one) > hp._score(two)
    assert hp._score(later) > hp._score(one)
    assert hp._score({"end": "log_end", "progress": [900, 150]}) > hp._score(later)


def test_a_card_generation_count_off_the_logs_marks_a_wrong_pool():
    # A Colorless Potion offers the picked card from a pool of another size
    # too; only the stream's count at the next fight shows it.
    off = {
        "desync": "opening_hand",
        "progress": [1520, 221],
        "notes": [
            "foe_hits_lost f17 VANTOM_BOSS",
            "rng@BOWLBUGS_WEAK f19 host/log {'combat_card_generation': (46, 44)}",
        ],
    }
    even = {**off, "notes": ["rng@start host/log {'up_front': (415, 375)}"]}
    assert hp._drift(off) == ["combat_card_generation"]
    assert hp._drift(even) == []
    assert hp._score(even) > hp._score(off)


def test_a_reward_screen_logged_without_resolve_rows_gets_them():
    rows = [
        {"t": "combat_end", "result": "victory"},
        {"t": "decision", "decision_id": 23, "decision_type": "card_reward", "source": "reward"},
        {"t": "gold", "gold": 98},
        {"t": "decision", "decision_id": 24, "decision_type": "relic_reward", "source": "reward"},
        {"t": "relic", "id": "HAPPY_FLOWER"},
        {"t": "acquire", "decision_id": 23, "source": "reward", "c": 168, "id": "DARKNESS"},
        {"t": "room_exit", "depth": 1},
    ]
    out = hr._with_missing_resolves(rows)
    assert [r.get("reward_kind") for r in out if r["t"] == "resolve"] == [
        "GoldReward",
        "RelicReward",
        "CardReward",
    ]
    assert [r for r in out if not r.get("added")] == rows
    # A screen that has its rows is left as it is.
    assert hr._with_missing_resolves(out) == out


def test_a_card_reward_taken_with_no_acquire_row_gets_one_from_the_deck():
    dec = {
        "t": "decision",
        "decision_id": 3,
        "decision_type": "card_reward",
        "source": "reward",
        "options": [
            {"option_id": "HOTFIX", "instance_id": 22},
            {"option_id": "HOLOGRAM", "instance_id": 23},
        ],
    }
    rows = [
        dec,
        {"t": "resolve", "decision_id": 3, "reward_kind": "GoldReward"},
        {"t": "resolve", "decision_id": 3, "reward_kind": "CardReward"},
        {"t": "deck", "cards": [{"c": 1, "id": "ZAP"}, {"c": 23, "id": "HOLOGRAM"}]},
        {"t": "room_exit", "depth": 1},
    ]
    out = hr._with_missing_acquires(rows)
    (added,) = [r for r in out if r["t"] == "acquire"]
    assert (added["id"], added["c"], added["option_index"]) == ("HOLOGRAM", 23, 1)
    assert out.index(added) == 2  # before the card reward's resolve row
    assert hr._with_missing_acquires(out) == out


def test_a_skipped_card_reward_gets_no_acquire_row():
    rows = [
        {
            "t": "decision",
            "decision_id": 3,
            "decision_type": "card_reward",
            "source": "reward",
            "options": [{"option_id": "HOTFIX", "instance_id": 22}],
        },
        {"t": "outcome", "decision_id": 3, "outcome": "skip"},
        {"t": "deck", "cards": [{"c": 22, "id": "HOTFIX"}]},
        {"t": "room_exit", "depth": 1},
    ]
    assert hr._with_missing_acquires(rows) == rows


def test_an_orb_evoked_by_no_card_means_the_plays_were_not_logged():
    turn = {"t": "turn", "side": "player"}
    evoke = {"t": "orb_evoke", "id": "LIGHTNING_ORB", "kept": True}
    assert hr._plays_lost([turn, evoke])
    assert not hr._plays_lost([turn, {"t": "play", "id": "DUALCAST"}, evoke])
    # So does an enemy an attack killed in a turn with no play.
    kill = {"t": "death", "tgt": "TOADPOLE", "killed_by": "ATTACK_COMMAND"}
    assert hr._plays_lost([turn, kill])
    assert not hr._plays_lost([turn, {"t": "play", "id": "STRIKE"}, kill])
    assert not hr._plays_lost([turn, {**kill, "killed_by": "POISON_POWER"}])
    # An orb pushed out by a full row of slots is not kept.
    assert not hr._plays_lost([turn, {**evoke, "kept": False}])


def test_a_players_attack_with_no_hit_row_blinds_the_enemy_hp():
    fight = [
        {"t": "play", "id": "STRIKE"},
        {"t": "attack", "src": "player", "hits": 1},
        {"t": "hit", "src": "player", "dst_cid": 1},
        {"t": "play", "id": "EXTERMINATE"},
        {"t": "attack", "src": "player", "hits": 4},
        {"t": "potion_start", "id": "EXPLOSIVE_AMPOULE"},
    ]
    assert hr._foe_hits_lost_from(fight) == 4
    assert hr._foe_hits_lost_from(fight[:4]) is None


def test_a_deck_pick_with_nothing_to_choose_is_not_awaited():
    # Amalgamator removes two Strikes; with exactly two in the deck the game
    # logs the decision and picks at once, and the host never asks.
    forced = {
        "decision_type": "deck_select_remove",
        "min_select": 2,
        "max_select": 2,
        "n_selectable": 2,
        "options": [{"option_id": "STRIKE_SILENT"}] * 2 + [{"option_id": "DEFEND_SILENT"}] * 18,
    }
    assert hr._forced_pick(forced)
    assert not hr._forced_pick({**forced, "n_selectable": 3})
    assert not hr._forced_pick({**forced, "decision_type": "card_reward"})
    # An upgrade one may decline is a choice even with one card to take.
    assert not hr._forced_pick({**forced, "min_select": 0, "max_select": 1, "n_selectable": 1})


def test_the_copy_picked_from_the_draw_pile_is_the_one_the_next_draws_miss():
    # Cleanse exhausts one of three Dazed in the draw pile. The log then draws
    # a Soul where the pile had a Dazed first: that Dazed is the one gone.
    pile = ["DEFEND", "DAZED", "SOUL", "TOUCH", "DAZED", "ASH", "DAZED"]
    coming = ["DEFEND", "SOUL", "TOUCH", "DAZED", "ASH"]
    assert hr._pile_picks(pile, [1, 4, 6], 1, coming) == (1,)
    # A pick that turns the copies into another card leaves them in place.
    coming = ["DEFEND", "BOMB", "SOUL", "TOUCH", "BOMB", "ASH"]
    assert hr._pile_picks(pile, [1, 4, 6], 2, coming, stays=True) == (1, 4)
    # A card that leaves is not drawn as another: with the Strike on top
    # drawn next, the one exhausted lay further down.
    pile = ["SLIMED", "STRIKE", "DEFEND", "UNLEASH", "STRIKE", "SLIMED"]
    assert hr._pile_picks(pile, [1, 4], 1, ["SLIMED", "STRIKE", "DEFEND", "UNLEASH"]) == (4,)
    # Draws that fit two choices alike decide nothing.
    assert hr._pile_picks(pile, [1, 4, 6], 1, ["DEFEND"]) is None


def _replayer(rows: list[dict[str, Any]]) -> hr.Replayer:
    r = object.__new__(hr.Replayer)
    r.rows, r.ci, r.combat_end, r.next_choice = rows, 1, len(rows), None
    r.track = _tracker(rows)
    return r


def test_only_stampede_asks_for_a_pick_as_the_turn_ends():
    auto = {"t": "play", "c": 9, "id": "TRUE_GRIT", "auto": True}
    tail = [auto, {"t": "pick", "cards": []}, {"t": "end_turn", "side": "player"}]
    # Stampede plays an attack once the turn is ended: the host asks then.
    stampede = {"t": "play", "c": 1, "id": "STAMPEDE", "auto": False}
    rows = [{"t": "combat_start"}, stampede, *tail]
    assert _replayer(rows).plays_itself_at_turn_end(3)
    # Cascade plays its cards at once: the pick was asked, or settled, then.
    cascade = {"t": "play", "c": 1, "id": "CASCADE", "auto": False}
    rows = [{"t": "combat_start"}, cascade, *tail]
    assert not _replayer(rows).plays_itself_at_turn_end(3)


def test_a_card_picked_from_the_discard_pile_is_not_ranked_by_the_draws_to_come():
    # Cosmic Indifference takes the second Defend discarded; it is drawn
    # first next turn, which says nothing of its place in the discard pile.
    rows = [
        {"t": "combat_start"},
        {"t": "play", "c": 1, "id": "DEFEND_REGENT", "result_pile": "discard"},
        {"t": "play", "c": 2, "id": "DEFEND_REGENT", "result_pile": "discard"},
        {"t": "pick", "cards": [{"c": 2, "id": "DEFEND_REGENT"}]},
        {"t": "draw", "c": 2, "id": "DEFEND_REGENT"},
        {"t": "draw", "c": 1, "id": "DEFEND_REGENT"},
    ]
    r = _replayer(rows[:3])
    r.rows, r.combat_end, r.next_choice = rows, len(rows), 3
    assert r.drawn_rank(2, "DEFEND_REGENT") is None
    assert r.track.hand_rank(2, 2) == 1


def test_a_copy_put_back_on_the_pile_is_not_ranked_among_the_piles():
    # Charge turns two of the pile's Defends into another card; Photon Cut
    # then puts the Defend in hand on the pile, and it is drawn before them.
    rows = [
        {"t": "combat_start"},
        {"t": "shuffle", "order_c": [1, 2, 3]},
        {"t": "draw", "c": 9, "id": "DEFEND_REGENT"},
        {"t": "pick", "cards": [{"c": 3, "id": "DEFEND_REGENT"}]},
        {"t": "draw", "c": 9, "id": "DEFEND_REGENT"},
        {"t": "draw", "c": 1, "id": "DEFEND_REGENT"},
        {"t": "draw", "c": 2, "id": "DEFEND_REGENT"},
        {"t": "draw", "c": 3, "id": "DEFEND_REGENT"},
    ]
    r = _replayer(rows[:3])
    r.track.ids.update({1: "DEFEND_REGENT", 2: "DEFEND_REGENT", 3: "DEFEND_REGENT"})
    r.rows, r.combat_end, r.next_choice = rows, len(rows), 3
    assert r.drawn_rank(3, "DEFEND_REGENT") == 2


def test_no_character_is_unlocked_behind_a_locked_one():
    assert hp._in_unlock_order(frozenset({"RELIC1_EPOCH"}))
    assert hp._in_unlock_order(frozenset({"NECROBINDER1_EPOCH", "DEFECT1_EPOCH"}))
    assert not hp._in_unlock_order(frozenset({"REGENT1_EPOCH"}))
    search = object.__new__(hp.Search)
    search.character = "SILENT"
    sets = search.foreign_sets(set())
    assert sets[0] == frozenset()
    # A character still locked is the likeliest state after all unlocked.
    assert sets[1] == frozenset(f"DEFECT{n}_EPOCH" for n in "1257")
    assert all(hp._in_unlock_order(s) for s in sets)
