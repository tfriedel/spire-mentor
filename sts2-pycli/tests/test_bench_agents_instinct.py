"""The Instinct referee driver's pieces that need no net (GH #1636)."""

from __future__ import annotations

from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

from bench_agents.instinct_policy import (  # noqa: E402
    InstinctPolicy,
    _event_option,
    event_encounter,
    infer_encounter,
)
from bench_agents.referee_game import host_answer  # noqa: E402

_ROOM = {"Weak": "monster", "Normal": "monster", "Elite": "elite", "Boss": "boss"}


def test_every_encounter_is_named_by_its_act_room_and_roster():
    # The referee withholds the encounter id; the opening roster names it.
    for enc in sts2_rs.list_encounters():
        got = infer_encounter(enc["act_name"], _ROOM[enc["tier"]], list(enc["monsters"]))
        assert got == enc["id"], enc


def test_a_roster_rolled_at_spawn_names_its_encounter():
    # GH #1659: the slime fights pick their members when they spawn, so the
    # screen rarely shows the listed roster. This one came from the real
    # game (seed 170000012, floor 2): the canonical roster is
    # LeafSlimeS + TwigSlimeS + LeafSlimeM.
    rolled = ["TwigSlimeS", "TwigSlimeM", "LeafSlimeS"]
    assert infer_encounter("Overgrowth", "monster", rolled) == "SLIMES_WEAK"
    for enc in sts2_rs.list_encounters():
        for roster in enc["rosters"]:
            assert infer_encounter(enc["act_name"], _ROOM[enc["tier"]], roster) == enc["id"], (
                enc["id"],
                roster,
            )


def test_a_roster_the_tables_do_not_hold_names_nothing():
    assert (
        infer_encounter("Overgrowth", "monster", ["Nibbit", "Nibbit", "Nibbit", "Axebot"]) is None
    )
    assert infer_encounter("Overgrowth", "event", ["Nibbit"]) is None


def test_an_event_button_names_its_event_and_its_relic_by_text_key():
    neow = _event_option({"index": 2, "text_key": "NEOW.pages.INITIAL.options.LARGE_CAPSULE"})
    assert neow["relic_id"] == "LARGE_CAPSULE" and neow["event_id"] == "NEOW"
    # The display name of Reflections is mirrored; the key is not.
    mirror = _event_option(
        {
            "event_id": "Reflections snoitcelfeR",
            "text_key": "REFLECTIONS.pages.INITIAL.options.SHATTER",
        }
    )
    assert mirror["event_id"] == "REFLECTIONS" and mirror["relic_id"] == ""
    # The Doll Room's buttons are the relics, keyed by their display names.
    doll = _event_option({"event_id": "Doll Room", "text_key": "Mr. Struggles"})
    assert doll["relic_id"] == "MR_STRUGGLES" and doll["event_id"] == "Doll Room"


class _Scripted:
    """An ``InstinctAdvisor`` stand-in answering with scripted ``best``s."""

    def __init__(self, picks: list[dict[str, Any]]) -> None:
        self.picks = picks
        self.asked: list[dict[str, Any]] = []

    def advise(self, msg: dict[str, Any], advice_for: str) -> dict[str, Any]:
        self.asked.append(msg)
        return {"best": self.picks[len(self.asked) - 1]}


def _policy(picks: list[dict[str, Any]]) -> InstinctPolicy:
    pol = InstinctPolicy.__new__(InstinctPolicy)
    pol.character, pol.ascension, pol.game_version = "Ironclad", 10, "v0.111.0"
    pol.adv = _Scripted(picks)  # type: ignore[assignment]
    pol.encounter = None
    pol.fight_hp = None
    pol.room_hp = None
    pol.fight_floor = None
    pol.last_room_type = None
    pol.last_room_floor = None
    pol.event_button = None
    pol.last_view = None
    pol.act_end_view = None
    pol.event_fight = False
    from collections import Counter

    pol.unavailable = Counter()
    return pol


def _select(n: int, lo: int, hi: int) -> dict[str, Any]:
    cards = [{"index": i, "id": f"C{i}"} for i in range(n)]
    return {"type": "card_select", "cards": cards, "min_count": lo, "max_count": hi}


def test_the_envelope_names_the_room_the_player_stands_in():
    # GH #1664: Neow's room at the first map, which has no current node.
    pol = _policy([])
    assert "current_room" not in pol.envelope({"context": {"act": 1}}, None)
    state = {"context": {"act": 1, "total_floor": 1, "room_type": "Event"}}
    assert pol.envelope(state, None)["current_room"] == "Event"
    # Once the map opens the game names `Map`; the run is still in Neow's room.
    state = {"context": {"act": 1, "total_floor": 1, "room_type": "Map"}}
    assert pol.envelope(state, None)["current_room"] == "Event"


def test_royal_poisons_first_turn_hit_is_not_in_the_fights_run_hp():
    # GH #1831: Royal Poison hits at the first turn's start, before Fake Blood
    # Vial's heal and before the first decision; the fight began at 90.
    pol = _policy([])
    relics = [{"id": "RELIC.ROYAL_POISON"}, {"id": "RELIC.FAKE_BLOOD_VIAL"}]
    pol.saw(
        {
            "decision": "map_select",
            "player": {"hp": 90, "max_hp": 90},
            "context": {"room_type": "Map"},
        }
    )
    boss = {"decision": "combat_play", "context": {"act": 3, "room_type": "Boss"}}
    msg = pol._fight_msg(boss, {"hp": 87, "max_hp": 90, "relics": relics}, None)
    assert msg["run_state"]["hp"] == 90
    pol.fight_started()
    alone = [{"id": "RELIC.ROYAL_POISON"}]
    msg = pol._fight_msg(boss, {"hp": 85, "max_hp": 93, "relics": alone}, None)
    assert msg["run_state"]["hp"] == 89
    pol.fight_started()
    rod = [*alone, {"id": "RELIC.TUNGSTEN_ROD"}]
    msg = pol._fight_msg(boss, {"hp": 94, "max_hp": 99, "relics": rod}, None)
    assert msg["run_state"]["hp"] == 97  # Tungsten Rod: the hit cost 3
    pol.fight_started()
    pick = {"decision": "card_select", "context": {"act": 3, "room_type": "Monster"}}
    msg = pol._fight_msg(pick, {"hp": 60, "max_hp": 93, "relics": alone}, None)
    assert msg["run_state"]["hp"] == 60  # before the first turn: not hit yet


def test_the_fights_run_hp_is_the_hp_before_its_first_turn():
    # GH #1705: Blood Vial heals at the first turn's start, before the first
    # fight decision; a boss room's Pantograph heals before the fight.
    pol = _policy([])
    vial = [{"id": "RELIC.BLOOD_VIAL"}, {"id": "RELIC.PANTOGRAPH"}]
    pol.saw(
        {
            "decision": "map_select",
            "player": {"hp": 50, "max_hp": 80},
            "context": {"room_type": "Map"},
        }
    )
    msg = pol._fight_msg(
        {"decision": "combat_play", "context": {"act": 1, "room_type": "Boss"}},
        {"hp": 77, "max_hp": 80, "relics": vial},
        None,
    )
    assert (msg["run_state"]["hp"], msg["run_state"]["max_hp"]) == (75, 80)
    pol.room_hp = 79
    pol.fight_started()
    msg = pol._fight_msg(
        {"decision": "combat_play", "context": {"act": 1, "room_type": "Monster"}},
        {"hp": 80, "max_hp": 80, "relics": vial},
        None,
    )
    assert msg["run_state"]["hp"] == 79  # capped heal: the map read 79
    pol.fight_started()
    pick = {"decision": "card_select", "context": {"act": 1, "room_type": "Monster"}}
    msg = pol._fight_msg(pick, {"hp": 60, "max_hp": 80, "relics": vial}, None)
    assert msg["run_state"]["hp"] == 60  # a pick before the first turn: not healed yet


def test_a_heal_before_the_first_turn_counts_when_blood_vials_was_capped():
    # Pantograph (a boss room) and Planisphere (a `?` room) heal before the
    # fight starts. When they fill the HP bar, the fight started at full HP,
    # and Blood Vial's heal at the first turn had nothing to heal: the screen
    # before the fight is no bound then (Spire Bench seeds 170000000 and
    # 170000052).
    pol = _policy([])
    map_screen = {"decision": "map_select", "context": {"room_type": "Map"}}
    boss = {"decision": "combat_play", "context": {"act": 2, "room_type": "Boss"}}
    pol.saw({**map_screen, "player": {"hp": 70, "max_hp": 80}})
    relics = [{"id": "RELIC.FAKE_BLOOD_VIAL"}, {"id": "RELIC.PANTOGRAPH"}]
    msg = pol._fight_msg(boss, {"hp": 80, "max_hp": 80, "relics": relics}, None)
    assert msg["run_state"]["hp"] == 80
    pol.saw({**map_screen, "player": {"hp": 94, "max_hp": 98}})
    pol.fight_started()
    fight = {"decision": "combat_play", "context": {"act": 3, "room_type": "Monster"}}
    relics = [{"id": "RELIC.BLOOD_VIAL"}, {"id": "RELIC.PLANISPHERE"}]
    msg = pol._fight_msg(fight, {"hp": 98, "max_hp": 98, "relics": relics}, None, node="unknown")
    assert msg["run_state"]["hp"] == 98
    # A plain fight node: Planisphere does not heal, the screen bounds it.
    pol.fight_started()
    msg = pol._fight_msg(fight, {"hp": 98, "max_hp": 98, "relics": relics}, None, node="monster")
    assert msg["run_state"]["hp"] == 96


def test_whispering_earrings_first_turn_does_not_move_the_fights_run_hp():
    # Whispering Earring plays the first turn by itself, before the first
    # fight decision. The fight's run HP is the HP the fight started on (the
    # map's), not the HP after those plays (Spire Bench seed 170000121: an
    # auto-played card cost 2 HP).
    pol = _policy([])
    ring = [{"id": "RELIC.WHISPERING_EARRING"}, {"id": "RELIC.PANTOGRAPH"}]
    map_screen = {
        "decision": "map_select",
        "player": {"hp": 78, "max_hp": 78},
        "context": {"room_type": "Map"},
    }
    pol.saw(map_screen)
    fight = {"decision": "combat_play", "context": {"act": 3, "room_type": "Monster"}}
    msg = pol._fight_msg(fight, {"hp": 76, "max_hp": 78, "relics": ring}, None)
    assert msg["run_state"]["hp"] == 78
    # A boss room's Pantograph heals 25 before the fight starts.
    pol.saw({**map_screen, "player": {"hp": 40, "max_hp": 78}})
    pol.fight_started()
    boss = {"decision": "combat_play", "context": {"act": 3, "room_type": "Boss"}}
    msg = pol._fight_msg(boss, {"hp": 61, "max_hp": 78, "relics": ring}, None)
    assert msg["run_state"]["hp"] == 65
    # A fight an event button started reads its first decision, as before;
    # Blood Vial's capped heal is still bounded by the screen before it.
    pol.saw({**map_screen, "player": {"hp": 79, "max_hp": 80}})
    pol.fight_started()
    pol.event_button = "FIGHT"
    worn = [*ring, {"id": "RELIC.BLOOD_VIAL"}]
    msg = pol._fight_msg(fight, {"hp": 80, "max_hp": 80, "relics": worn}, None)
    assert msg["run_state"]["hp"] == 79


def test_a_melted_blood_vial_heals_nothing_before_the_first_turn():
    # Toy Box's wax Blood Vial melts; the game shows it, it heals nothing
    # (Spire Bench seed 180000158: the map read 86, so did the first turn).
    pol = _policy([])
    pol.saw(
        {
            "decision": "map_select",
            "player": {"hp": 86, "max_hp": 107},
            "context": {"room_type": "Map"},
        }
    )
    melted = [{"id": "RELIC.TOY_BOX"}, {"id": "RELIC.BLOOD_VIAL", "melted": True}]
    fight = {"decision": "combat_play", "context": {"act": 3, "room_type": "Monster"}}
    msg = pol._fight_msg(fight, {"hp": 86, "max_hp": 107, "relics": melted}, None)
    assert msg["run_state"]["hp"] == 86


def test_an_imbued_cards_first_turn_play_does_not_move_the_fights_run_hp():
    # Imbued plays its card at the first turn's start, before the first
    # decision; an Imbued Offering cost 6 HP there (seed 180000474: map 71,
    # first turn 65). The fight started on the map's HP.
    pol = _policy([])
    pol.saw(
        {
            "decision": "map_select",
            "player": {"hp": 71, "max_hp": 80},
            "context": {"room_type": "Map"},
        }
    )
    deck = [{"id": "OFFERING", "enchantment": "Imbued", "enchantment_amount": 1}]
    fight = {"decision": "combat_play", "context": {"act": 2, "room_type": "Monster"}}
    run = {"hp": 65, "max_hp": 80, "relics": [], "master_deck": deck}
    assert pol._fight_msg(fight, run, None)["run_state"]["hp"] == 71


def test_an_event_fights_first_decision_still_knows_its_button():
    # The fight's first decision names the encounter before it reads the
    # starting HP; naming it used to forget the event button, so an event
    # fight took the map's path: Whispering Earring added Planisphere's heal,
    # already on the event's own screen, a second time (seed 180001585: 81
    # read as 86).
    pol = _policy([])
    relics = [{"id": "RELIC.WHISPERING_EARRING"}, {"id": "RELIC.PLANISPHERE"}]
    pol.saw(
        {
            "decision": "event_choice",
            "player": {"hp": 81, "max_hp": 97},
            "context": {"room_type": "Event"},
        }
    )
    pol.event_button = "BATTLEWORN_DUMMY.pages.INITIAL.options.SETTING_2"
    fight = {"decision": "combat_play", "context": {"act": 3, "room_type": "Monster"}}
    pol._name_encounter(fight, ["BattleFriendV2"], "unknown")
    run = {"hp": 81, "max_hp": 97, "relics": relics}
    msg = pol._fight_msg(fight, run, None, node="unknown")
    assert msg["run_state"]["hp"] == 81
    assert pol.event_button is None


def test_whispering_earrings_first_turn_in_an_event_fight_does_not_move_its_run_hp():
    # Battleworn Dummy's fight starts on the event page's HP; Whispering
    # Earring then plays the first turn before the first decision, and that
    # turn cost HP (seed 180001585: page 97, first decision 96; seed
    # 180001606: page 87, first decision 81). The engine's fight starts on
    # the page's HP. Planisphere healed entering the `?` room, before the
    # page, so it is not added again.
    pol = _policy([])
    relics = [{"id": "RELIC.WHISPERING_EARRING"}, {"id": "RELIC.PLANISPHERE"}]
    pol.saw(
        {
            "decision": "event_choice",
            "player": {"hp": 97, "max_hp": 99},
            "context": {"room_type": "Event"},
        }
    )
    pol.event_button = "BATTLEWORN_DUMMY.pages.INITIAL.options.SETTING_2"
    fight = {"decision": "combat_play", "context": {"act": 3, "room_type": "Monster"}}
    pol._name_encounter(fight, ["BattleFriendV2"], "unknown")
    run = {"hp": 96, "max_hp": 99, "relics": relics}
    msg = pol._fight_msg(fight, run, None, node="unknown")
    assert msg["run_state"]["hp"] == 97


def test_a_fight_does_not_start_again_when_its_mirror_was_dropped():
    # The driver calls `fight_started` whenever it meets a fight screen with
    # no live mirror. A mirror dropped in the middle of a fight (a desync)
    # must not restart the fight: a turn-6 Toasty Mittens screen re-read the
    # run's HP there (Spire Bench seed 170000068).
    pol = _policy([])
    turn = {
        "decision": "combat_play",
        "player": {"hp": 80, "max_hp": 80},
        "context": {"act": 3, "total_floor": 48, "room_type": "Boss"},
    }
    pol.saw(turn)
    pol.fight_started(turn)
    pol.encounter = "X"
    assert pol._fight_msg(turn, {"hp": 80, "max_hp": 80}, None)["run_state"]["hp"] == 80
    pick = {**turn, "decision": "card_select", "combat": {}, "player": {"hp": 68, "max_hp": 80}}
    pol.saw(pick)
    pol.fight_started(pick)
    assert pol.encounter == "X"
    assert pol._fight_msg(pick, {"hp": 68, "max_hp": 80}, None)["run_state"]["hp"] == 80
    # The next floor's fight is a new one, also with no screen between (the
    # second act-3 boss).
    nxt = {**turn, "context": {"act": 3, "total_floor": 49, "room_type": "Boss"}}
    nxt["player"] = {"hp": 51, "max_hp": 80}
    pol.saw(nxt)
    pol.fight_started(nxt)
    assert pol.encounter is None
    assert pol._fight_msg(nxt, {"hp": 51, "max_hp": 80}, None)["run_state"]["hp"] == 51
    # And so is a later fight on the same floor, once a screen outside a
    # fight came between.
    pol.saw({"decision": "rewards", "player": {"hp": 57, "max_hp": 80}, "context": nxt["context"]})
    pol.saw(nxt)
    pol.fight_started(nxt)
    assert pol._fight_msg(nxt, {"hp": 57, "max_hp": 80}, None)["run_state"]["hp"] == 57


def test_a_room_the_host_answered_still_names_the_next_map():
    # GH #1704: a `?` that rolled a treasure chest shows only the chest's
    # reward screen, which the bench answers; the game wrapper shows it to
    # the student, and the run stands in that room at the next map.
    pol = _policy([])
    elite = {"context": {"act": 1, "total_floor": 12, "room_type": "Elite"}}
    assert pol.envelope(elite, None)["current_room"] == "Elite"
    pol.saw(
        {"decision": "rewards", "context": {"act": 1, "total_floor": 13, "room_type": "Treasure"}}
    )
    at_map = {"context": {"act": 1, "total_floor": 13, "room_type": "Map"}}
    assert pol.envelope(at_map, None)["current_room"] == "Treasure"


def test_a_room_that_showed_no_screen_is_an_empty_chest():
    # Spire Bench seed 180000315: under Silver Crucible the first chest holds
    # nothing, so a `?` that rolled it shows no screen at all; the next map
    # is a floor later, and the run stands in that treasure room, not the
    # fight before it.
    pol = _policy([])
    fight = {"context": {"act": 1, "total_floor": 2, "room_type": "Monster"}}
    assert pol.envelope(fight, None)["current_room"] == "Monster"
    same_floor = {"context": {"act": 1, "total_floor": 2, "room_type": "Map"}}
    assert pol.envelope(same_floor, None)["current_room"] == "Monster"
    next_floor = {"context": {"act": 1, "total_floor": 3, "room_type": "Map"}}
    assert pol.envelope(next_floor, None)["current_room"] == "Treasure"
    assert pol.envelope(next_floor, None)["current_room"] == "Treasure"


def test_an_in_fight_choice_is_asked_of_the_student(monkeypatch):
    # GH #1667: natively the student answers it (COMBAT_CHOICE), not the search.
    import bench_agents.instinct_policy as ip

    monkeypatch.setattr(ip.sts2_rs, "graft_run", lambda msg, v: (msg, []))
    monkeypatch.setattr(ip.sts2_rs, "instinct_enter_live_combat", lambda *a: None)
    pol = _policy([])
    asked: list[tuple[Any, int]] = []

    def picks(msg: dict[str, Any], rm: Any, n: int) -> list[int]:
        asked.append((msg["run_state"]["hp"], n))
        return [2]

    pol.adv.choice_picks = picks  # type: ignore[attr-defined]
    kw = {"room_type": "monster", "bridge_map": None}
    got = pol.combat_choice(
        {"context": {"act": 1}}, None, 1, bridge_run_state={"hp": 50, "max_hp": 80}, **kw
    )
    assert got == [2]
    assert asked == [(50, 1)]


def test_a_fight_that_opens_on_a_screen_is_named_there(monkeypatch):
    # GH #1697: Toasty Mittens' turn-1 exhaust is a fight's first decision;
    # the student reads this fight's encounter on it, not the last one's.
    import bench_agents.instinct_policy as ip

    entered: list[str] = []
    monkeypatch.setattr(ip.sts2_rs, "graft_run", lambda msg, v: (msg, []))
    monkeypatch.setattr(
        ip.sts2_rs,
        "instinct_enter_live_combat",
        lambda rm, cs, b, room: entered.append(b["encounter_id"]),
    )
    monkeypatch.setattr(ip.sts2_rs, "state_to_dict", lambda cs: {"encounter": [{"kind": "Nibbit"}]})
    pol = _policy([])
    pol.adv.choice_picks = lambda msg, rm, n: [0]  # type: ignore[attr-defined]
    state = {"context": {"act": 1, "act_name": "Overgrowth", "room_type": "Monster"}}
    pol.combat_choice(
        state, object(), 1, room_type="monster", bridge_run_state={"hp": 50}, bridge_map=None
    )
    assert entered == [ip.infer_encounter("Overgrowth", "monster", ["Nibbit"])]
    assert entered[0]


def test_an_out_of_fight_choose_a_card_screen_is_asked_as_the_mods_card_reward():
    # GH #1665: Lead Paperweight. The host sends a card_select; the mod (and
    # the student's training) a card_reward on the choose_a_card surface.
    pol = _policy([{"action": "choose", "index": 1}])
    msg = {**_select(2, 0, 1), "cancelable": True}
    got = pol.screen(msg, {"context": {"act": 1}, "source": "FromChooseACardScreen"}, None)
    assert got == {"action": "choose", "index": 1}
    asked = pol.adv.asked[0]
    assert (asked["type"], asked["surface"], asked["can_skip"]) == (
        "card_reward",
        "choose_a_card",
        True,
    )
    assert "max_count" not in asked


def test_the_run_hp_stays_at_the_fights_start_until_the_fight_ends(monkeypatch):
    # GH #1666: the engine's run HP (RUNCTX R_HP) is the fight's start HP.
    import bench_agents.instinct_policy as ip

    seen: list[dict[str, Any]] = []

    def graft(msg: dict[str, Any], _v: str) -> Any:
        seen.append(msg["run_state"])
        raise ValueError("stop")

    monkeypatch.setattr(ip.sts2_rs, "graft_run", graft)
    pol = _policy([])
    pol.encounter = "X"
    pol.fight_started()
    pol.encounter = "X"
    state = {"context": {"act": 1}}
    for hp in (64, 51):
        pol.combat(state, {}, None, "monster", {"hp": hp, "max_hp": 80}, None)
    assert [r["hp"] for r in seen] == [64, 64]
    pol.fight_started()
    pol.encounter = "X"
    pol.combat(state, {}, None, "monster", {"hp": 51, "max_hp": 80}, None)
    assert seen[-1]["hp"] == 51


def test_an_enemy_dead_before_the_first_decision_still_names_the_encounter(monkeypatch):
    # GH #1716: Festive Popper killed a Twig Slime before the first decision.
    # The board lists three slimes; the encounter is the four that opened it.
    import bench_agents.instinct_policy as ip

    seen: list[list[str]] = []

    def infer(_act: str, _tier: str, kinds: list[str]) -> str:
        seen.append(kinds)
        return "X"

    monkeypatch.setattr(ip, "infer_encounter", infer)
    monkeypatch.setattr(
        ip.sts2_rs, "graft_run", lambda *_a: (_ for _ in ()).throw(ValueError("stop"))
    )
    pol = _policy([])
    board = [{"kind": "TwigSlimeM"}, {"kind": "LeafSlimeM"}, {"kind": "LeafSlimeS"}]
    roster = [*board[:2], {"kind": "TwigSlimeS", "is_dead": True}, board[2]]
    state = {"context": {"act": 1, "act_name": "Underdocks"}}
    wire = {"encounter": board, "removed_roster": roster}
    pol.combat(state, wire, None, "monster", {"hp": 60, "max_hp": 80}, None)
    assert seen == [["TwigSlimeM", "LeafSlimeM", "TwigSlimeS", "LeafSlimeS"]]


def test_a_question_mark_fight_is_tiered_by_the_games_room():
    from bench_agents.instinct_policy import _fight_tier

    assert _fight_tier({"context": {"room_type": "Monster"}}, "unknown") == "monster"
    assert _fight_tier({"context": {"room_type": "Elite"}}, "unknown") == "elite"
    assert _fight_tier({"context": {}}, "boss") == "boss"


def test_a_pick_two_screen_is_answered_with_two_picks_in_one_action():
    pol = _policy([{"action": "choose", "index": 3}, {"action": "choose", "index": 0}])
    got = pol._multi_select(_select(5, 2, 2))
    assert got == {"action": "choose", "indices": [3, 0]}
    # Every ask holds the whole screen and the picks so far (positions):
    # the advisor replays them in the engine (GH #1673).
    assert [len(a["cards"]) for a in pol.adv.asked] == [5, 5]
    assert [a["picked_indices"] for a in pol.adv.asked] == [[], [3]]
    assert [(a["min_count"], a["max_count"]) for a in pol.adv.asked] == [(2, 2), (2, 2)]


def test_a_pick_screen_stops_when_instinct_confirms_after_the_minimum():
    pol = _policy([{"action": "choose", "index": 1}, {"action": "skip"}])
    assert pol._multi_select(_select(5, 1, 3)) == {"action": "choose", "indices": [1]}


def test_a_confirm_before_the_minimum_falls_back():
    pol = _policy([{"action": "skip"}])
    assert pol._multi_select(_select(5, 2, 2)) is None


def test_the_crystal_sphere_is_dug_by_the_digger_on_a_fogged_cell():
    cells = [{"x": x, "y": y, "hidden": True} for x in range(11) for y in range(11)]
    obs = {
        "decision": "crystal_sphere",
        "cells": cells,
        "divinations_left": 3,
        "divinations_total": 3,
    }
    got = host_answer(obs, 1, set())
    assert got is not None and got["action"] == "crystal_sphere_click"
    assert got["args"]["tool"] == "big"
    assert 0 <= got["args"]["x"] < 11 and 0 <= got["args"]["y"] < 11


def test_rewards_are_taken_in_order_and_a_refused_one_is_skipped():
    obs = {"decision": "rewards", "rewards": [{"index": 0}, {"index": 1}]}
    assert host_answer(obs, 7, set()) == {"action": "take_reward", "args": {"reward_index": 0}}
    refused = {(7, '{"action": "take_reward", "args": {"reward_index": 0}}')}
    assert host_answer(obs, 7, refused) == {"action": "skip_reward", "args": {"reward_index": 0}}
    assert host_answer({"decision": "rewards", "rewards": []}, 8, set()) == {"action": "proceed"}


def test_an_event_fights_extra_potion_waits_for_the_card_on_a_full_belt():
    # GH #1832: on a full belt the engine asks PunchOff's extra potion's swap
    # after the card pick (Spire Bench seed 180000054); the room's own potion
    # still comes first.
    belt = [{"slot": 0, "entry": "SKILL_POTION"}]
    obs = {
        "decision": "rewards",
        "player": {"potion_slots": 2, "potions": belt},
        "rewards": [
            {"index": 1, "type": "potion", "id": "SWIFT_POTION"},
            {"index": 2, "type": "potion", "id": "COLORLESS_POTION", "extra": True},
            {"index": 4, "type": "card"},
        ],
    }
    assert host_answer(obs, 3, set()) == {"action": "take_reward", "args": {"reward_index": 1}}
    obs["rewards"] = obs["rewards"][1:]
    obs["player"]["potions"] = [*belt, {"slot": 1, "entry": "SWIFT_POTION"}]
    assert host_answer(obs, 4, set()) == {"action": "take_reward", "args": {"reward_index": 4}}
    obs["player"]["potions"] = belt  # room on the belt: the potion first (GH #1803)
    assert host_answer(obs, 4, set()) == {"action": "take_reward", "args": {"reward_index": 2}}


def test_a_full_belt_potion_reward_is_asked_as_the_mods_swap_group():
    # GH #1670: the engine's POTION_SWAP; the host parks it as `potion_swap`.
    pol = _policy([{"action": "choose", "index": 2}])
    state = {
        "context": {"act": 1},
        "potion": "FIRE_POTION",
        "options": [
            {"index": 0, "discard_slot": None, "label": "decline"},
            {"index": 1, "discard_slot": 0, "label": "discard BLOCK_POTION"},
            {"index": 2, "discard_slot": 1, "label": "discard SWIFT_POTION"},
        ],
    }
    assert pol.potion_swap({"type": "reward_screen"}, state, None) == 1
    asked = pol.adv.asked[0]
    assert asked["type"] == "reward_screen"
    assert [o["discard_slot"] for o in asked["options"]] == [None, 0, 1]
    assert {o["action"] for o in asked["options"]} == {"potion_swap"}
    assert asked["options"][0]["reward"] == {"id": "FIRE_POTION"}


def test_an_event_button_names_the_fight_it_starts():
    # GH #1676: no map table holds an event's fight; the engine names it
    # after the event (the Battleworn Dummy per setting from v0.111.0).
    assert event_encounter("PUNCH_OFF.pages.INITIAL.options.FIGHT", "v0.111.0") == (
        "PUNCH_OFF_EVENT_ENCOUNTER"
    )
    dummy = "BATTLEWORN_DUMMY.pages.INITIAL.options.SETTING_2"
    assert event_encounter(dummy, "v0.111.0") == "BATTLEWORN_DUMMY_EVENT_V2_ENCOUNTER"
    assert event_encounter(dummy, "v0.107.1") == "BATTLEWORN_DUMMY_EVENT_ENCOUNTER"
    assert event_encounter("NEOW.pages.INITIAL.options.LARGE_CAPSULE", "v0.111.0") is None
    # The Lantern Key's fight is the Mysterious Knight's (Spire Bench seed
    # 170000032: the encounter id was 0 on the real side).
    assert (
        event_encounter("THE_LANTERN_KEY.pages.KEEP_THE_KEY.options.FIGHT", "v0.111.0")
        == "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER"
    )


def test_a_fight_after_an_event_button_is_that_events(monkeypatch):
    import bench_agents.instinct_policy as ip

    def no_graft(msg, v):
        raise ValueError("stop here")

    monkeypatch.setattr(ip.sts2_rs, "graft_run", no_graft)
    keys = [
        "DENSE_VEGETATION.pages.INITIAL.options.REST",
        "DENSE_VEGETATION.pages.INITIAL.options.FIGHT",
    ]
    pol = _policy([{"action": "choose", "index": 1}])
    screen = {"type": "event", "options": [{"index": i, "text_key": k} for i, k in enumerate(keys)]}
    pol.screen(screen, {"context": {"act": 1}}, None)
    fight = {"context": {"act": 1, "act_name": "Overgrowth"}}
    wire = {"encounter": [{"kind": "Wriggler"}]}
    pol.combat(fight, wire, None, "event", bridge_run_state={}, bridge_map=None)
    assert pol.encounter == "DENSE_VEGETATION_EVENT_ENCOUNTER"
    # The next fight is not the event's.
    pol.fight_started()
    pol.combat(fight, wire, None, "monster", bridge_run_state={}, bridge_map=None)
    assert pol.encounter == ""


def test_an_event_fights_reward_stands_in_the_events_room(monkeypatch):
    # GH #1688: the game names the event fight's room `Monster`; the engine
    # stays in the event's, until the map opens.
    import bench_agents.instinct_policy as ip

    monkeypatch.setattr(ip.sts2_rs, "graft_run", lambda *_: (_ for _ in ()).throw(ValueError()))
    pol = _policy([{"action": "choose", "index": 0}])
    key = "DENSE_VEGETATION.pages.INITIAL.options.FIGHT"
    pol.screen(
        {"type": "event", "options": [{"index": 0, "text_key": key}]}, {"context": {"act": 1}}, None
    )
    fight = {"context": {"act": 1, "act_name": "Overgrowth", "room_type": "Monster"}}
    pol.combat(
        fight,
        {"encounter": [{"kind": "Wriggler"}]},
        None,
        "event",
        bridge_run_state={},
        bridge_map=None,
    )
    assert pol.envelope(fight, None)["current_room"] == "Event"
    pol.envelope({"context": {"act": 1, "room_type": "Map"}}, None)
    assert pol.envelope(fight, None)["current_room"] == "Monster"


def test_the_boss_relic_is_asked_at_the_acts_end():
    # GH #1678: the game offers it at the next act's Ancient, the engine in
    # the boss's room before the next act opens.
    pol = _policy([{"index": 0}] * 4)
    boss = {"context": {"act": 1, "room_type": "Boss", "total_floor": 16}}
    ancient = {"context": {"act": 2, "room_type": "Event", "total_floor": 16}}
    act1_map, act2_map = {"act": 1}, {"act": 2}
    pol.screen({"type": "card_reward", "options": []}, boss, act1_map)
    # The Ancient's dialogue page first, then its relic offer.
    pol.screen({"type": "event", "options": [{"index": 0}, {"index": 1}]}, ancient, act2_map)
    pol.screen({"type": "boss_relic", "options": []}, ancient, act2_map)
    pol.screen({"type": "map_select", "options": []}, {"context": {"act": 2}}, act2_map)
    _reward, _dialogue, relic, after = pol.adv.asked
    assert (relic["act"], relic["current_room"], relic["map"]) == (1, "Boss", act1_map)
    # The host may leave it on the event surface: the Ancient's own page.
    pol = _policy([{"index": 0}] * 2)
    pol.screen({"type": "card_reward", "options": []}, boss, act1_map)
    page = [{"index": i, "text_key": f"PAEL.pages.INITIAL.options.{r}"} for i, r in enumerate("AB")]
    pol.screen({"type": "event", "options": page}, ancient, act2_map)
    assert (pol.adv.asked[1]["act"], pol.adv.asked[1]["map"]) == (1, act1_map)
    assert (after["act"], after["current_room"], after["map"]) == (2, "Event", act2_map)


def test_an_event_fights_extra_potion_is_taken_before_its_card():
    # GH #1698: C# lists a potion before a card; the engine takes the relics
    # first. Since GH #1801 it rolls an event fight's extra potion with the
    # set and takes it after the room's own, before the card pick (Spire
    # Bench seed 170000019: Blessing of the Forge stood on the native belt at
    # the card pick, the bench took it after both cards).
    from bench_agents.referee_game import host_answer

    obs = {
        "decision": "rewards",
        "rewards": [
            {"index": 0, "type": "potion"},
            {"index": 1, "type": "potion", "extra": True},
            {"index": 2, "type": "relic", "extra": True},
            {"index": 3, "type": "card"},
        ],
    }
    order = []
    while obs["rewards"]:
        pick = host_answer(obs, 0, set())["args"]["reward_index"]
        order.append(pick)
        obs["rewards"] = [r for r in obs["rewards"] if r["index"] != pick]
    assert order == [2, 0, 1, 3]


def test_an_event_fight_the_host_names_is_still_an_event_fight():
    # GH #1937: once the host sent `encounter_id` (#1925), the event button
    # was no longer read, so an event-started fight lost `event_fight` and the
    # student read its room as `monster` (Spire Bench cmp_v10: 306 runs, first
    # difference RUNCTX X_ROOM event -> monster).
    pol = _policy([])
    pol.event_button = "BATTLEWORN_DUMMY.pages.INITIAL.options.SETTING_2"
    fight = {
        "decision": "combat_play",
        "encounter_id": "BATTLEWORN_DUMMY_EVENT_ENCOUNTER",
        "context": {"act": 3, "room_type": "Monster"},
    }
    pol._name_encounter(fight, ["BattleFriendV2"], "unknown")
    assert pol.encounter == "BATTLEWORN_DUMMY_EVENT_ENCOUNTER"
    assert pol.event_fight
