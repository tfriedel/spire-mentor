"""``coop_replay``: a recorded co-op decision pair replayed seat by seat."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import coop_replay
from realgame_adapter import to_headless_combat_play

pytest.importorskip("sts2_rs")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
)


def _mod_board() -> dict[str, Any]:
    """The fixture's Ironclad board (one Shrinker Beetle, 40 HP), with net
    ids and one Silent teammate holding a Strike, their whole seat sent."""
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    strike = copy.deepcopy(board["hand"][2])
    strike.update(id="STRIKE_SILENT")
    board["player"]["net_id"] = "1"
    board["allies"] = [
        {
            "character": "SILENT",
            "net_id": "2",
            "hp": 50,
            "max_hp": 70,
            "block": 0,
            "is_dead": False,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "hand": [strike],
            "draw_pile": {"type": "Draw", "count": 0, "cards": []},
            "discard_pile": {"type": "Discard", "count": 0, "cards": []},
            "exhaust_pile": {"type": "Exhaust", "count": 0, "cards": []},
            "play_pile": {"type": "Play", "count": 0, "cards": []},
            "relics": [],
            "stars": 0,
            "ended_turn": False,
        }
    ]
    return board


def _teammate_strike_row() -> dict[str, Any]:
    return {
        "kind": "card_play_started",
        "round": 1,
        "side": "player",
        "card": "STRIKE_SILENT",
        "upgraded": False,
        "auto_played": False,
        "owner": "2",
        "target": {"kind": "monster", "id": "SHRINKER_BEETLE", "uid": None},
        "target_index": 0,
        "play_index": 0,
    }


def _pair(enemy_hp_after: int) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _mod_board()
    after = _mod_board()
    after["enemies"][0]["hp"] = enemy_hp_after
    after["allies"][0]["hand"] = []
    after["allies"][0]["energy"] = 2
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = [_teammate_strike_row()]
    return a, b


def test_a_teammates_recorded_play_replays_in_sync() -> None:
    a, b = _pair(enemy_hp_after=40 - 6)
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["played"] == 1
    assert res["diffs"] == {}


def test_a_divergence_is_reported_by_field() -> None:
    a, b = _pair(enemy_hp_after=40 - 9)
    res = coop_replay.replay_pair(a, b)
    assert res["diffs"] == {"encounter[0].hp": [34, 31]}


def test_an_older_board_is_skipped_not_guessed() -> None:
    a, b = _pair(enemy_hp_after=34)
    for ally in a["allies"]:
        del ally["draw_pile"]
    assert coop_replay.replay_pair(a, b) == {"skipped": "teammate piles missing (older mod)"}


def test_a_cards_combat_vars_reach_the_engine() -> None:
    """A Thrumming Hatchet played this turn (``combat_vars.return_if_played_round``,
    from the mod's History read) survives both adapters into the engine."""
    import sts2_rs
    from sts2_hint.adapter import translate

    board = _mod_board()
    hatchet = copy.deepcopy(board["hand"][2])
    hatchet.update(id="THRUMMING_HATCHET", combat_vars={"return_if_played_round": 1})
    board["discard_pile"] = {"type": "Discard", "count": 1, "cards": [hatchet]}
    wire = translate(to_headless_combat_play(board))
    card = wire["deck"]["discard_pile"][0]
    assert card["combat_vars"] == {"return_if_played_round": 1}
    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert again["deck"]["discard_pile"][0]["combat_vars"]["return_if_played_round"] == 1


def test_our_place_in_the_player_order_reaches_the_engine() -> None:
    """A board from the second player's seat loads with us behind the
    teammate (``coop_seat_index``), who then sets up and draws first."""
    import sts2_rs

    a, _ = _pair(enemy_hp_after=34)
    assert "coop_seat_index" not in coop_replay.full_fight_wire(a)
    a["seat_index"] = 1
    wire = coop_replay.full_fight_wire(a)
    assert wire["coop_seat_index"] == 1
    again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert again["coop_seat_index"] == 1


def _survivor_pair(choice_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The teammate plays Survivor (8 block, discard 1) from Survivor, Strike,
    Defend and discards the Defend."""
    before = _mod_board()
    strike = before["allies"][0]["hand"][0]
    survivor = copy.deepcopy(strike)
    survivor.update(id="SURVIVOR", type="Skill", target="Self")
    defend = copy.deepcopy(strike)
    defend.update(id="DEFEND_SILENT", type="Skill", target="Self")
    before["allies"][0]["hand"] = [survivor, strike, defend]
    after = copy.deepcopy(before)
    after["allies"][0].update(hand=[strike], block=8, energy=2)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    play = _teammate_strike_row()
    play.update(card="SURVIVOR", target=None, target_index=None)
    b["combat_log_raw"] = [play, *choice_rows]
    return a, b


def _choice_row(owner: str, cards: list[str]) -> dict[str, Any]:
    return {
        "kind": "player_choice",
        "round": 1,
        "side": "player",
        "owner": owner,
        "choice_id": 0,
        "choice_type": "CombatCard",
        "cards": [{"id": c, "upgraded": False} for c in cards],
        "indexes": None,
        "player": None,
    }


def test_a_teammates_logged_pick_is_the_one_replayed() -> None:
    a, b = _survivor_pair([_choice_row("2", ["DEFEND_SILENT"])])
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["choices"] == 1
    assert res["diffs"] == {}


def test_without_a_logged_pick_the_engine_picks_and_diverges() -> None:
    a, b = _survivor_pair([])
    res = coop_replay.replay_pair(a, b)
    assert "allies[0].hand" in res["diffs"]


def test_a_pick_matches_cards_by_id_once_each() -> None:
    options = [
        {"id": "StrikeSilent"},
        {"id": "DefendSilent"},
        {"id": "DefendSilent", "upgraded": True},
    ]
    row = _choice_row("2", ["DEFEND_SILENT", "DEFEND_SILENT"])
    assert coop_replay._choice_selection(row, options) == [1, 2]
    row["cards"][0]["upgraded"] = True
    assert coop_replay._choice_selection(row, options) == [1, 2]
    assert coop_replay._choice_selection(_choice_row("2", ["BASH"]), options) is None
    index = {**_choice_row("2", []), "choice_type": "Index", "indexes": [2]}
    assert coop_replay._choice_selection(index, options) == [2]


def test_our_own_logged_pick_is_answered_not_skipped() -> None:
    """We play True Grit+ (9 block, exhaust a card of our choice) and exhaust
    the Bash: the replay answers our screen from the log."""
    before = _mod_board()
    grit = copy.deepcopy(before["hand"][0])
    grit.update(id="TRUE_GRIT", upgraded=True)
    before["hand"][0] = grit
    after = copy.deepcopy(before)
    after["hand"] = before["hand"][1:4]
    after["player"].update(block=9, energy=2)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    play = _teammate_strike_row()
    play.update(card="TRUE_GRIT", upgraded=True, owner="1", target=None, target_index=None)
    b["combat_log_raw"] = [play, _choice_row("1", ["BASH"])]
    b["interrupted"] = True
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["choices"] == 1
    assert res["diffs"] == {}


def test_an_index_skip_selects_nothing() -> None:
    skip = {**_choice_row("2", []), "choice_type": "Index", "indexes": [-1]}
    assert coop_replay._choice_selection(skip, [{"id": "A"}, {"id": "B"}]) == []


def _block_potion_pair(belt: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """The teammate drinks a Block Potion (12 block); their belt is sent
    with the board when ``belt``, as newer mods do."""
    before = _mod_board()
    if belt:
        before["allies"][0]["potions"] = [
            {"slot": 0, "empty": False, "id": "BLOCK_POTION", "name": "Block Potion"},
            {"slot": 1, "empty": True},
        ]
    after = copy.deepcopy(before)
    after["allies"][0]["block"] = 12
    after["allies"][0].pop("potions", None)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = [
        {
            "kind": "potion",
            "round": 1,
            "side": "player",
            "potion": "BLOCK_POTION",
            "owner": "2",
            "target_index": None,
        }
    ]
    return a, b


@pytest.mark.parametrize("belt", [True, False])
def test_a_teammates_potion_is_drunk_from_their_belt(belt: bool) -> None:
    a, b = _block_potion_pair(belt)
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["potions"] == 1
    assert res["diffs"] == {}


@pytest.mark.parametrize("discarded", [1, 2])
def test_a_logged_multi_pick_is_taken_whole(discarded: int) -> None:
    """The teammate drinks Gambler's Brew and discards one or both Defends in
    one pick (the engine takes a multi-pick a card at a time, its legal
    actions single picks only), drawing a Strike for each."""
    before = _mod_board()
    mate = before["allies"][0]
    strike = mate["hand"][0]
    defend = copy.deepcopy(strike)
    defend.update(id="DEFEND_SILENT", type="Skill", target="Self")
    mate["hand"] = [strike, defend, copy.deepcopy(defend)]
    mate["draw_pile"] = {
        "type": "Draw",
        "count": 2,
        "cards": [copy.deepcopy(strike), copy.deepcopy(strike)],
    }
    mate["potions"] = [{"slot": 0, "empty": False, "id": "GAMBLERS_BREW", "name": "Gambler's Brew"}]
    after = copy.deepcopy(before)
    after["allies"][0].update(
        hand=[copy.deepcopy(c) for c in [strike] * (1 + discarded) + [defend] * (2 - discarded)],
        draw_pile={"type": "Draw", "count": 2 - discarded, "cards": [strike] * (2 - discarded)},
        discard_pile={"type": "Discard", "count": discarded, "cards": [defend] * discarded},
    )
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    drink = {
        "kind": "potion",
        "round": 1,
        "side": "player",
        "potion": "GAMBLERS_BREW",
        "owner": "2",
        "target_index": None,
    }
    b["combat_log_raw"] = [_choice_row("2", ["DEFEND_SILENT"] * discarded), drink]
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert (res["choices"], res["unanswered"]) == (1, 0)
    assert res["diffs"] == {}


def test_a_cards_extra_plays_reach_the_engine() -> None:
    """A Strike that Soldier's Stew made play twice (``base_replay_count``)
    keeps it through both adapters and hits twice."""
    import sts2_rs
    from sts2_hint.adapter import translate

    board = _mod_board()
    strike = next(c for c in board["hand"] if c["id"] == "STRIKE_IRONCLAD")
    strike["base_replay_count"] = 1
    wire = translate(to_headless_combat_play(board))
    eng = sts2_rs.state_from_dict(wire)
    hand = sts2_rs.state_to_dict(eng)["deck"]["hand"]
    idx = next(i for i, c in enumerate(hand) if c.get("base_replay_count") == 1)
    hp = sts2_rs.state_to_dict(eng)["encounter"][0]["hp"]
    eng.apply({"action": "play", "card_index": idx, "target_index": 0})
    assert hp - sts2_rs.state_to_dict(eng)["encounter"][0]["hp"] == 12


@pytest.mark.parametrize("seat", ["us", "teammate"])
def test_lost_hp_this_turn_reaches_the_engine(seat: str) -> None:
    """A player who lost HP this turn (``hp_lost_this_turn``, a History scan
    the engine cannot redo) hits twice with Spite."""
    import sts2_rs
    from sts2_hint.adapter import translate

    board = _mod_board()
    spite = copy.deepcopy(board["hand"][2])
    spite.update(id="SPITE", cost=0, stats={"damage": 5})
    if seat == "us":
        board["hand"][2] = spite
        board["player"]["hp_lost_this_turn"] = True
    else:
        board["allies"][0]["hand"] = [spite]
        board["allies"][0]["hp_lost_this_turn"] = True
    wire = translate(to_headless_combat_play(board))
    wire["coop_full"] = True
    eng = sts2_rs.state_from_dict(wire)
    now = sts2_rs.state_to_dict(eng)
    hp = now["encounter"][0]["hp"]
    play = {"action": "play", "target_index": 0}
    if seat == "us":
        hand = [c["id"] for c in now["deck"]["hand"]]
        eng.apply({**play, "card_index": hand.index("SPITE")})
    else:
        eng.apply_as_ally(0, {**play, "card_index": 0})
    assert hp - sts2_rs.state_to_dict(eng)["encounter"][0]["hp"] == 10


def test_a_v0111_only_power_stays_on_the_board() -> None:
    """One For All (v0.111.0 only) is kept, not dropped as unknown: every
    0-cost Attack its owner plays deals its amount more."""
    from sts2_hint.adapter import translate

    board = _mod_board()
    board["player"]["powers"] = [
        {"id": "ONE_FOR_ALL_POWER", "name": "One For All", "amount": 4, "type": "Buff"}
    ]
    wire = translate(to_headless_combat_play(board))
    assert {"name": "ONE_FOR_ALL", "amount": 4} in [
        {k: p[k] for k in ("name", "amount")} for p in wire["player"]["powers"]
    ]


@pytest.mark.parametrize(
    ("engine", "game"),
    [("STATUS:WOUND", "WOUND"), ("CURSE:DOUBT", "DOUBT"), ("STAMPEDE_CARD", "STAMPEDE")],
)
def test_the_adapters_card_prefixes_are_not_a_divergence(engine: str, game: str) -> None:
    assert coop_replay._norm(engine) == coop_replay._norm(game)


def test_a_teammates_tutor_takes_our_logged_pick() -> None:
    """The teammate plays Tutor on us and we fetch the Bash from our draw
    pile: the engine asks us (chooser ``-1``) and the log answers."""
    before = _mod_board()
    tutor = copy.deepcopy(before["allies"][0]["hand"][0])
    tutor.update(id="TUTOR", type="Skill", target="AnyAlly", cost=1)
    before["allies"][0]["hand"] = [tutor]
    bash = copy.deepcopy(before["hand"][4])
    before["draw_pile"]["cards"] = [*before["draw_pile"]["cards"][:4], bash]
    after = copy.deepcopy(before)
    after["hand"] = [*before["hand"], bash]
    after["draw_pile"]["cards"] = before["draw_pile"]["cards"][:4]
    after["allies"][0].update(hand=[], energy=2)
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    play = _teammate_strike_row()
    play.update(card="TUTOR", target=None, target_index=None)
    b["combat_log_raw"] = [play, _choice_row("1", ["BASH"])]
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["choices"] == 1
    assert res["diffs"] == {}


def _mittens_boards() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Our turn ends with the Silent teammate holding Toasty Mittens (v0.111.0,
    where it asks) and a Strike and a Defend to draw. At their next turn start
    they exhaust a card from hand (the Strike) -- after the game sent us our
    board ``b``, so ``b`` shows the Strike still in hand and the pick is
    logged with the board after, ``c`` (GH #989)."""
    before = _mod_board()
    ally = before["allies"][0]
    strike = ally["hand"].pop()
    defend = {**strike, "id": "DEFEND_SILENT", "type": "Skill", "target": "Self"}
    ally["draw_pile"] = {"type": "Draw", "count": 2, "cards": [strike, defend]}
    ally["relics"] = [{"id": "TOASTY_MITTENS", "name": "Toasty Mittens", "vars": None}]
    before.update(floor=3, game_version="v0.111.0")
    turn_start = copy.deepcopy(before)
    turn_start["allies"][0].update(
        hand=[strike, defend], draw_pile={"type": "Draw", "count": 0, "cards": []}
    )
    picked = copy.deepcopy(turn_start)
    picked["allies"][0]["hand"] = [defend]
    a, b, c = (to_headless_combat_play(s) for s in (before, turn_start, picked))
    b["combat_log_raw"] = [
        {"kind": "monster_move", "round": 1, "side": "enemy", "text": "", "move_id": "BITE"}
    ]
    c["combat_log_raw"] = [{**_choice_row("2", ["STRIKE_SILENT"]), "round": 2}]
    return a, b, c


def test_a_teammates_late_turn_start_pick_is_replayed_through_to_its_board() -> None:
    a, b, c = _mittens_boards()
    alone = coop_replay.replay_pair(a, b)
    assert alone["late"] == ["2"]
    assert alone["unanswered"] == 1
    res = coop_replay.replay_pair(a, b, c)
    assert res.get("skipped") is None, res
    assert (res["choices"], res["unanswered"], res["late"]) == (1, 0, [])
    assert "allies[0].hand" not in res["diffs"]


def _brew_boards(discarded_on: int) -> list[dict[str, Any]]:
    """Three boards. The teammate starts Gambler's Brew after the first: the
    game takes it off their belt at once, and logs the drink with the board
    after their pick, the ``discarded_on``-th, which shows the Defend
    discarded and a Strike drawn."""
    first = _mod_board()
    mate = first["allies"][0]
    strike = mate["hand"][0]
    defend = copy.deepcopy(strike)
    defend.update(id="DEFEND_SILENT", type="Skill", target="Self")
    mate["hand"] = [strike, defend]
    mate["draw_pile"] = {"type": "Draw", "count": 1, "cards": [copy.deepcopy(strike)]}
    mate["potions"] = [
        {"slot": 0, "empty": True},
        {"slot": 1, "empty": False, "id": "GAMBLERS_BREW", "name": "Gambler's Brew"},
    ]
    picking = copy.deepcopy(first)
    picking["allies"][0]["potions"][1] = {"slot": 1, "empty": True}
    done = copy.deepcopy(picking)
    done["allies"][0].update(
        hand=[copy.deepcopy(strike), copy.deepcopy(strike)],
        draw_pile={"type": "Draw", "count": 0, "cards": []},
        discard_pile={"type": "Discard", "count": 1, "cards": [defend]},
    )
    boards = [first] + [picking] * (discarded_on - 1) + [done]
    boards = [to_headless_combat_play(copy.deepcopy(b)) for b in boards]
    drink = {
        "kind": "potion",
        "round": 1,
        "side": "player",
        "potion": "GAMBLERS_BREW",
        "owner": "2",
        "target_index": None,
    }
    boards[-1]["combat_log_raw"] = [_choice_row("2", ["DEFEND_SILENT"]), drink]
    return boards


@pytest.mark.parametrize("discarded_on", [2, 3])
def test_a_potion_still_being_drunk_goes_back_on_the_belt(
    tmp_path: Path, discarded_on: int
) -> None:
    """GH #1017: the drink is replayed from the slot an earlier board showed it
    in, instead of skipped as "not in 2's belt"."""
    boards = _brew_boards(discarded_on)
    res = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", boards))[-1]
    assert res.get("skipped") is None, res
    assert (res["potions"], res["choices"]) == (1, 1)
    assert res["diffs"] == {}


def test_a_potion_already_drunk_is_not_put_back() -> None:
    """A board whose belt shows no Brew and no earlier board of the fight held
    one: nothing to put back, the row is still skipped."""
    *_, picking, done = _brew_boards(2)
    res = coop_replay.replay_pair(picking, done)
    assert res["skipped"] == "GAMBLERS_BREW not in 2's belt"


def _write_log(path: Path, boards: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for board in boards:
            f.write(json.dumps({"type": "state", "data": board}) + "\n")
    return path


def test_the_log_replay_spans_the_boards_to_a_late_pick(tmp_path: Path) -> None:
    a, b, c = _mittens_boards()
    (res,) = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c]))
    assert res["span"] == 2
    assert res["choices"] == 1
    assert "allies[0].hand" not in res["diffs"]


def test_a_fight_won_before_the_teammate_picked_is_skipped(tmp_path: Path) -> None:
    a, b, _ = _mittens_boards()
    (res,) = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b]))
    assert res["skipped"] == "a teammate's turn-start pick still open when the fight ended"


def test_a_board_from_before_a_teammates_turn_start_pick_is_skipped() -> None:
    _, b, c = _mittens_boards()
    res = coop_replay.replay_pair(b, c)
    assert res == {"skipped": "a teammate's turn-start pick made after board a"}


def _mayhem_boards(
    first: dict[str, Any] | None = None, **auto: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Our turn ends; at the next turn start the Silent teammate's Mayhem
    auto-plays their top card -- queued on their own action queue, so it
    resolves after the game sent us our board ``b`` (GH #1017, a 3-player
    recording) and is logged with the board after, ``c``."""
    a, b = _pair(enemy_hp_after=40)
    c = copy.deepcopy(b)
    b["combat_log_raw"] = [
        {"kind": "monster_move", "round": 1, "side": "enemy", "text": "attacked"}
    ]
    b["round"] = c["round"] = 2
    row = {**_teammate_strike_row(), "round": 2, "auto_played": True, **auto}
    c["combat_log_raw"] = [*([first] if first else []), row]
    return a, b, c


def test_a_board_from_before_a_teammates_turn_start_auto_play_is_skipped() -> None:
    _, b, c = _mayhem_boards()
    res = coop_replay.replay_pair(b, c)
    assert res == {"skipped": "a teammate's turn-start auto-play resolved after board a"}


@pytest.mark.parametrize(
    ("first", "auto"),
    [
        # A play of ours in between: the auto-play came from that play.
        ({**_teammate_strike_row(), "owner": "1", "round": 2}, {}),
        # Our own auto-play: the game sends our board after it.
        (None, {"owner": "1"}),
        # Last round's: not a turn start this board missed.
        (None, {"round": 1}),
    ],
)
def test_other_auto_plays_are_not_a_late_turn_start(
    first: dict[str, Any] | None, auto: dict[str, Any]
) -> None:
    _, b, c = _mayhem_boards(first, **auto)
    assert not coop_replay._late_turn_start_auto_play(b, c)


def test_the_log_replay_spans_a_board_sent_before_a_teammates_auto_play(
    tmp_path: Path,
) -> None:
    a, b, c = _mayhem_boards()
    c["combat_log_raw"].append({"kind": "card_play_finished", "round": 2, "card": "STRIKE_SILENT"})
    (res,) = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c]))
    assert res.get("span") == 2, res


def test_an_auto_play_after_a_potion_is_not_a_late_turn_start() -> None:
    """Distilled Chaos auto-plays cards: the potion row comes first."""
    potion = {"kind": "potion", "round": 2, "owner": "2", "potion": "DISTILLED_CHAOS"}
    _, b, c = _mayhem_boards(potion)
    assert not coop_replay._late_turn_start_auto_play(b, c)


def test_an_auto_play_whose_parent_play_is_still_open_is_not_a_late_turn_start() -> None:
    """A teammate's play logged with ``a`` (not yet finished) whose
    auto-played follow-ups are logged with ``b``."""
    _, b, c = _mayhem_boards()
    b["combat_log_raw"] = [
        *b["combat_log_raw"],
        {**_teammate_strike_row(), "card": "HAVOC", "round": 2},
    ]
    assert not coop_replay._late_turn_start_auto_play(b, c)


def test_an_auto_play_that_never_finishes_in_the_log_is_skipped(tmp_path: Path) -> None:
    a, b, c = _mayhem_boards()
    results = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c]))
    assert results[0]["skipped"] == "a teammate's turn-start auto-play never finished in the log"


def test_a_teammates_ordinary_play_alongside_does_not_stretch_the_span(tmp_path: Path) -> None:
    """The auto-play finishes in ``c``; a teammate's own play starting there
    and finishing later does not hold the span open."""
    a, b, c = _mayhem_boards()
    c["combat_log_raw"] = [
        *c["combat_log_raw"],
        {"kind": "card_play_finished", "round": 2, "card": "STRIKE_SILENT"},
        {**_teammate_strike_row(), "card": "DEFEND_SILENT", "round": 2},
    ]
    d = copy.deepcopy(c)
    d["combat_log_raw"] = [{"kind": "card_play_finished", "round": 2, "card": "DEFEND_SILENT"}]
    results = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c, d]))
    assert results[0].get("span") == 2, results


def test_the_span_runs_on_to_the_board_the_auto_play_finished_by(tmp_path: Path) -> None:
    """The game sent another board while the auto-play was still resolving;
    its damage is logged with the board after (GH #1017)."""
    a, b, c = _mayhem_boards()
    d = copy.deepcopy(c)
    d["combat_log_raw"] = [{"kind": "card_play_finished", "round": 2, "card": "STRIKE_SILENT"}]
    results = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c, d]))
    assert results[0].get("span") == 3, results


@pytest.mark.parametrize(
    ("text", "owner"),
    [
        # Older mods named The Ball's new owner (GH #1017); the text names
        # the player who played it.
        ("IRONCLAD started playing THE_BALL targeting SHRINKER_BEETLE", "1"),
        ("SILENT started playing STRIKE_SILENT targeting SHRINKER_BEETLE", "2"),
        # No text, or a character no seat has: the row's owner stands.
        ("", "2"),
        ("DEFECT started playing ZAP", "2"),
    ],
)
def test_a_card_plays_owner_follows_its_text(text: str, owner: str) -> None:
    a, _ = _pair(enemy_hp_after=40)
    a["character_id"] = "IRONCLAD"
    row = {**_teammate_strike_row(), "text": text}
    assert coop_replay._row_owner(row, a) == owner


def _damage_row(dealer: dict[str, Any], receiver: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {
        "kind": "damage",
        "round": 1,
        "side": "player",
        "dealer": dealer,
        "receiver": receiver,
        "source": {"kind": "card", "id": "STRIKE_SILENT", "upgraded": False},
        "unblocked": 6,
        "blocked": 0,
        **kw,
    }


def test_a_seats_combat_history_is_recounted_from_the_fight_rows() -> None:
    """GH #989: the hits that took a seat's HP this combat (Tear Asunder) and
    the seat's card hits on each enemy this turn (a teammate's Gang Up) are
    recounted from the fight's rows, by creature ``combat_id`` (ours is our
    ``seat_index``)."""
    a, _ = _pair(enemy_hp_after=34)
    a["seat_index"] = 1
    beetle = {"kind": "monster", "id": "SHRINKER_BEETLE", "combat_id": 2}
    a["enemies"][0]["combat_id"] = 2
    teammate = {"kind": "player", "id": "SILENT", "combat_id": 0}
    us = {"kind": "player", "id": "IRONCLAD", "combat_id": 1}
    a["fight_rows"] = [
        _damage_row(teammate, beetle),
        _damage_row(teammate, beetle),
        _damage_row(teammate, beetle, round=0),  # an earlier turn
        _damage_row(teammate, beetle, source={"kind": "power"}),  # not a card hit
        _damage_row(beetle, us, side="enemy"),
        _damage_row(beetle, teammate, side="enemy"),
        _damage_row(beetle, teammate, side="enemy", unblocked=0),  # all blocked
        _damage_row(beetle, teammate, side="enemy"),
    ]
    wire = coop_replay.full_fight_wire(a)
    assert wire["player"]["unblocked_hits_this_combat"] == 1
    assert "powered_hits_this_turn" not in wire["player"]
    ally = wire["allies"][0]
    assert ally["unblocked_hits_this_combat"] == 2
    assert ally["powered_hits_this_turn"] == [{"enemy": 0, "hits": 2}]


def test_a_seats_combat_history_from_the_mod_is_kept() -> None:
    """GH #989: newer mods send each seat's History counters themselves; the
    board's own wins over a recount from the rows."""
    board = _mod_board()
    board["player"].update(
        unblocked_hits_this_combat=3, powered_hits_this_turn=[], cards_exhausted_this_combat=1
    )
    board["allies"][0].update(
        unblocked_hits_this_combat=0,
        powered_hits_this_turn=[{"enemy": 0, "hits": 2}],
        cards_exhausted_this_combat=4,
    )
    a = to_headless_combat_play(board)
    a["enemies"][0]["combat_id"] = 2
    beetle = {"kind": "monster", "id": "SHRINKER_BEETLE", "combat_id": 2}
    teammate = {"kind": "player", "id": "SILENT", "combat_id": 1}
    # Rows a recount would read otherwise: one teammate hit, one on us.
    a["fight_rows"] = [
        _damage_row(teammate, beetle),
        _damage_row(beetle, {"kind": "player", "id": "IRONCLAD", "combat_id": 0}, side="enemy"),
    ]
    wire = coop_replay.full_fight_wire(a)
    assert wire["player"]["unblocked_hits_this_combat"] == 3
    assert "powered_hits_this_turn" not in wire["player"]
    ally = wire["allies"][0]
    assert "unblocked_hits_this_combat" not in ally
    assert ally["powered_hits_this_turn"] == [{"enemy": 0, "hits": 2}]
    import sts2_rs

    back = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert back["player"]["unblocked_hits_this_combat"] == 3
    assert back["allies"][0]["powered_hits_this_turn"] == [{"enemy": 0, "hits": 2}]
    # Midnight's count of every player's exhausts.
    assert back["player"]["cards_exhausted_this_combat"] == 1
    assert back["allies"][0]["cards_exhausted_this_combat"] == 4


def test_a_seats_extra_turn_count_from_the_mod_reaches_the_engine() -> None:
    """GH #812: newer mods send each seat's extra turns this combat
    (PaelsEye, Ambergris: TurnNumber - RoundNumber). The turn-number gates
    read round + this, so it must survive adapter, wire and engine; a zero
    stays off the wire, its default."""
    board = _mod_board()
    board["player"]["extra_turns_taken_this_combat"] = 1
    board["allies"][0]["extra_turns_taken_this_combat"] = 2
    wire = coop_replay.full_fight_wire(to_headless_combat_play(board))
    assert wire["player"]["extra_turns_taken_this_combat"] == 1
    assert wire["allies"][0]["extra_turns_taken_this_combat"] == 2
    import sts2_rs

    back = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert back["player"]["extra_turns_taken_this_combat"] == 1
    assert back["allies"][0]["extra_turns_taken_this_combat"] == 2

    board = _mod_board()
    board["player"]["extra_turns_taken_this_combat"] = 0
    wire = coop_replay.full_fight_wire(to_headless_combat_play(board))
    assert "extra_turns_taken_this_combat" not in wire["player"]


def _mod_board_with_orbs(orbs: list[dict[str, Any]] | None) -> dict[str, Any]:
    board = _mod_board()
    if orbs is not None:
        board["allies"][0].update(orbs=orbs, orb_capacity=3)
    return board


def test_a_teammates_frost_orb_blocks_at_their_turn_end() -> None:
    """GH #989: the mod sends a teammate's orb belt as ours (Chaos channels
    orbs for any character). Their Frost orb blocks 2 at their turn end, so
    the beetle's first attack (round 2) takes 2 less of their HP."""
    import sts2_rs

    frost = {"index": 0, "id": "FROST_ORB", "name": "Frost", "passive_value": 2, "evoke_value": 5}

    def ally_hp_after_two_rounds(orbs: list[dict[str, Any]] | None) -> int:
        wire = coop_replay.full_fight_wire(to_headless_combat_play(_mod_board_with_orbs(orbs)))
        eng = sts2_rs.state_from_dict(wire)
        for _ in range(2):
            eng.apply({"action": "end_turn"})
        return sts2_rs.state_to_dict(eng)["allies"][0]["hp"]

    assert ally_hp_after_two_rounds([frost]) - ally_hp_after_two_rounds(None) == 2
    wire = coop_replay.full_fight_wire(to_headless_combat_play(_mod_board_with_orbs([frost])))
    assert wire["allies"][0]["orb_slots"] == 3
    assert [o["type"] for o in wire["allies"][0]["orbs"]] == ["FROST"]


@pytest.mark.parametrize(
    ("offered", "reported"),
    [(["STRIKE_SILENT", "DEFEND_SILENT"], False), (["BASH", "DEFEND_SILENT"], True)],
)
def test_an_offer_unlike_the_games_is_reported(offered: list[str], reported: bool) -> None:
    """GH #989: the mod logs the cards a screen offered, so a pick from a
    random offer the engine rolled differently is named, not just a hand
    diff."""
    row = {**_choice_row("2", ["DEFEND_SILENT"]), "offered": [{"id": c} for c in offered]}
    a, b = _survivor_pair([row])
    res = coop_replay.replay_pair(a, b)
    assert ("offers" in res) == reported
    if reported:
        assert res["offers"][0]["game"] == ["BASH", "DEFENDSILENT"]


def test_a_target_is_found_in_the_order_the_engine_was_built_from() -> None:
    """GH #989: the engine keeps the enemies of the board it was built from,
    the dead included, while a later board lists only the living. A play on
    the second of three enemies, after the first died, is still index 1."""
    origin = {"enemies": [{"combat_id": 4}, {"combat_id": 5}, {"combat_id": 6}]}
    later = {"enemies": [{"combat_id": 5}, {"combat_id": 6}, {"combat_id": 9}]}

    def target(combat_id: int) -> int | None:
        row = {"target": {"combat_id": combat_id}}
        return coop_replay._target({}, row, later, origin)

    assert target(5) == 1
    assert target(9) == 2, "a spawn is placed by the later board"


def test_a_teammates_play_in_the_next_round_is_replayed() -> None:
    """GH #989: the teammate can play in the new round before the game sends
    us ``b``; that play comes after the enemy turn in ``b``'s rows. Their
    Strike reshuffles back into their hand at the turn start."""
    a, b = _pair(enemy_hp_after=40)
    enemy_turn = {"kind": "monster_move", "round": 1, "side": "enemy", "text": "attacked"}
    b["combat_log_raw"] = [enemy_turn, {**_teammate_strike_row(), "round": 2}]
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["ended_turn"]
    assert res["played"] == 1


def _brand_boards() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The teammate plays Survivor (8 block, discard 1). The game sends us a
    board while the card waits on its pick, ``b``, and logs the pick with the
    board after, ``c`` (GH #989: a teammate's Brand, floors 10-29 of the
    2026-10-28 run)."""
    a, done = _survivor_pair([])
    play = done["combat_log_raw"][0]
    mid = copy.deepcopy(done)
    mid["allies"][0]["hand"] = done["allies"][0]["hand"] + [
        {**done["allies"][0]["hand"][0], "id": "DEFEND_SILENT"}
    ]
    b = {**mid, "combat_log_raw": [play]}
    c = {**copy.deepcopy(done), "combat_log_raw": [_choice_row("2", ["DEFEND_SILENT"])]}
    return a, b, c


def test_a_teammates_pick_mid_play_is_replayed_through_to_its_board(tmp_path: Path) -> None:
    a, b, c = _brand_boards()
    alone = coop_replay.replay_pair(a, b)
    assert (alone["unanswered"], alone["late"]) == (1, ["2"])
    (res,) = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, c]))
    assert res["span"] == 2
    assert (res["choices"], res["unanswered"]) == (1, 0)
    assert res["diffs"] == {}


def test_a_pick_logged_only_with_the_next_fight_is_skipped(tmp_path: Path) -> None:
    """The fight ended while the teammate was still picking: the pick's row
    comes with the next fight's first board."""
    a, b, c = _brand_boards()
    still = copy.deepcopy(b)
    still["combat_log_raw"] = []
    c["floor"] = (a.get("floor") or 0) + 1
    res = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", [a, b, still, c]))
    assert res[0]["skipped"] == "a teammate's turn-start pick still open when the fight ended"


@pytest.mark.parametrize("next_fight", [True, False])
def test_a_mid_play_pick_the_game_never_logged_is_reported(
    tmp_path: Path, next_fight: bool
) -> None:
    """A pick only the engine raised has no row anywhere: the pair is
    reported with its diffs, not excused as a late pick (PR #1023 review)."""
    a, b, c = _brand_boards()
    c["combat_log_raw"] = []
    c["floor"] = (a.get("floor") or 0) + 1
    boards = [a, b, c] if next_fight else [a, b]
    res = coop_replay.replay_log(_write_log(tmp_path / "run.jsonl", boards))
    assert "skipped" not in res[0], res[0]
    assert res[0]["late_in_play"] == ["2"]
    assert res[0]["unanswered"] == 1
    assert res[0]["diffs"]


def test_a_pair_across_an_advisor_reconnect_is_skipped() -> None:
    a, b = _pair(enemy_hp_after=34)
    b["after_reconnect"] = True
    assert coop_replay.replay_pair(a, b) == {"skipped": "the advisor reconnected: rows may be lost"}


def _row(kind: str, **kw: Any) -> dict[str, Any]:
    return {"kind": kind, "round": 1, "side": "player", **kw}


def test_a_seats_turn_so_far_is_recounted_from_the_fight_rows() -> None:
    """GH #989: an exhaust or a card's block goes to the seat whose play or
    pick it follows (the rows name no owner); this turn's only."""
    a, _ = _pair(enemy_hp_after=34)
    card = {"kind": "card", "id": "DEFEND_IRONCLAD", "upgraded": False}
    a["fight_rows"] = [
        _row("card_exhausted", card="STRIKE"),  # no play or pick yet: nobody's
        _row("player_choice", owner="1"),  # our Toasty Mittens pick
        _row("card_exhausted", card="STRIKE"),
        _row("card_play_started", owner="1", card="DEFEND_IRONCLAD"),
        _row("block", source=card, amount=10),
        _row("card_play_started", owner="2", card="SURVIVOR"),
        _row("block", source=card, amount=8),
        _row("block", source={"kind": "move"}, amount=4),  # not a card's
        _row("card_exhausted", card="STRIKE"),
        _row("card_exhausted", card="STRIKE", round=0),  # last turn
        _row("potion", owner="1", potion="X"),  # our potion makes the teammate
        _row("card_exhausted", card="STRIKE", owner="2"),  # exhaust: theirs, by the row
        _row("block", source=card, amount=3),  # still the teammate's Survivor play
    ]
    wire = coop_replay.full_fight_wire(a)
    assert wire["player"]["cards_exhausted_this_turn"] == 1
    assert wire["player"]["card_block_gains_this_turn"] == 1
    ally = wire["allies"][0]
    assert ally["cards_exhausted_this_turn"] == 2
    assert ally["card_block_gains_this_turn"] == 2
    # v0.114.0 Unmovable counts plays: the teammate's one Survivor play gained
    # block twice, and counts once.
    a["game_version"] = "v0.114.0"
    assert coop_replay.full_fight_wire(a)["allies"][0]["card_block_gains_this_turn"] == 1


def test_a_seats_turn_so_far_from_the_mod_is_kept() -> None:
    board = _mod_board()
    board["player"].update(cards_exhausted_this_turn=0, card_block_gains_this_turn=2)
    board["allies"][0].update(cards_exhausted_this_turn=3)
    a = to_headless_combat_play(board)
    a["fight_rows"] = [
        _row("player_choice", owner="1"),
        _row("card_exhausted", card="STRIKE"),
    ]
    wire = coop_replay.full_fight_wire(a)
    # Sent as 0: kept, so the live mirror knows reality disclosed it.
    assert wire["player"]["cards_exhausted_this_turn"] == 0
    assert wire["player"]["card_block_gains_this_turn"] == 2
    assert wire["allies"][0]["cards_exhausted_this_turn"] == 3
    import sts2_rs

    back = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    assert back["player"]["card_block_gains_this_turn"] == 2
    assert back["allies"][0]["cards_exhausted_this_turn"] == 3


def _crab_board(facing: int | None = None) -> dict[str, Any]:
    """The fixture board against Crusher (Back Attack Left, combat id 2) and
    Rocket (Back Attack Right, 3), every player Surrounded."""
    board = _mod_board()
    beetle = board["enemies"][0]
    arms = []
    for cid, kind, back in ((2, "CRUSHER", "LEFT"), (3, "ROCKET", "RIGHT")):
        arm = copy.deepcopy(beetle)
        arm.update(combat_id=cid, id=kind, name=kind.title(), hp=300, max_hp=300)
        arm["powers"] = [{"id": f"BACK_ATTACK_{back}_POWER", "name": "Back Attack", "amount": 1}]
        arms.append(arm)
    board["enemies"] = arms
    surrounded = {"id": "SURROUNDED_POWER", "name": "Surrounded", "amount": 1}
    if facing is not None:
        surrounded["vars"] = {"Facing": facing}
    board["allies"][0]["powers"] = [surrounded]
    return board


@pytest.mark.parametrize(
    ("targets", "facing"),
    [([], 0), ([2], 1), ([2, 3], 0), ([3], 0), ([2, 2], 1), (["potion"], 1)],
)
def test_a_teammates_facing_is_retraced_from_their_plays(targets: list[Any], facing: int) -> None:
    """GH #989: a teammate turns toward the crab arm behind them when they
    play a card at it; facing Left, Rocket's hits on them take 1.5x."""
    a = to_headless_combat_play(_crab_board())
    # A potion row from an older mod: only the index, of Crusher (2).
    potion = {"kind": "potion", "round": 1, "side": "player", "owner": "2", "target_index": 0}
    a["fight_rows"] = [
        potion
        if cid == "potion"
        else {**_teammate_strike_row(), "target": {"kind": "monster", "combat_id": cid}}
        for cid in targets
    ]
    (surrounded,) = coop_replay.full_fight_wire(a)["allies"][0]["powers"]
    assert surrounded["counter"] == facing


def test_a_teammates_facing_from_the_mod_is_kept() -> None:
    a = to_headless_combat_play(_crab_board(facing=1))
    a["fight_rows"] = []
    (surrounded,) = coop_replay.full_fight_wire(a)["allies"][0]["powers"]
    assert surrounded["counter"] == 1


def test_a_reconnect_mark_on_a_screen_moves_to_the_next_combat_board(tmp_path: Path) -> None:
    """The advisor marks whatever it logs next after a reconnect, a reward
    screen too; the mark belongs to the next combat board (PR #1023 review)."""
    a, b = _pair(enemy_hp_after=34)
    screen = {"decision": "card_reward", "after_reconnect": True, "combat_log_raw": []}
    path = _write_log(tmp_path / "run.jsonl", [a, screen, b])
    states = coop_replay.load_states(path)
    assert [bool(s.get("after_reconnect")) for s in states] == [False, True]


def _three_seat_board() -> dict[str, Any]:
    """The fixture board with a second Silent teammate (net id ``3``) and a
    Lift in our hand and in teammate ``2``'s."""
    board = _mod_board()
    mate = copy.deepcopy(board["allies"][0])
    mate.update(net_id="3", hand=[])
    board["allies"].append(mate)
    lift = copy.deepcopy(board["hand"][2])
    lift.update(id="LIFT", type="Skill", target="AnyAlly", cost=1)
    board["hand"] = [lift]
    board["allies"][0]["hand"] = [copy.deepcopy(lift)]
    return board


def _lift_row(owner: str, target_net: str | None, combat_id: int | None = None) -> dict[str, Any]:
    row = _teammate_strike_row()
    target = {"kind": "player", "id": "SILENT", "uid": None, "combat_id": combat_id}
    if target_net is not None:
        target["net_id"] = target_net
    row.update(card="LIFT", owner=owner, target=target, target_index=None)
    return row


def test_an_any_ally_play_targets_the_logged_teammate() -> None:
    """GH #1017: with two teammates the replay aims our Lift at the one the
    log names, and a teammate's at us, in that teammate's own frame."""
    import sts2_rs

    a = to_headless_combat_play(_three_seat_board())
    rows = [_lift_row("1", "3"), _lift_row("2", "1")]
    # Mapped in each actor's frame: ours to allies[1]; teammate 2 (allies[0])
    # sees us at their allies[0].
    assert coop_replay._ally_target(rows[0], "1", "1", a) == 1
    assert coop_replay._ally_target(rows[1], "2", "1", a) == 0
    assert coop_replay._ally_target(_lift_row("2", "3"), "2", "1", a) == 1
    # Older rows: the player's combat id is their place in the run order.
    assert coop_replay._ally_target(_lift_row("1", None, combat_id=2), "1", "1", a) == 1
    assert coop_replay._ally_target(_lift_row("2", None, combat_id=0), "2", "1", a) == 0
    assert coop_replay._ally_target(_teammate_strike_row(), "2", "1", a) is None

    eng = sts2_rs.state_from_dict(coop_replay.full_fight_wire(a))
    lift_block = sts2_rs.state_to_dict(eng)["deck"]["hand"][0]["base_block"]
    after = _three_seat_board()
    after["hand"] = []
    after["allies"][0]["hand"] = []
    after["player"]["block"] = lift_block
    after["allies"][1]["block"] = lift_block
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = rows
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["played"] == 2
    assert {k: v for k, v in res["diffs"].items() if k != "player.energy"} == {}
