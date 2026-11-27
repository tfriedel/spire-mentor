"""v0.113.0 Blitz mode on the C# headless host (GH #1347).

``GameMode.Blitz`` exists from v0.113.0. A Blitz run is three acts of a fixed
eight-node line (Ancient, Monster, Monster, Shop, Elite, Treasure, RestSite,
Boss: ``Map/BlitzActMap.cs``) with no map choice, a ``BlitzRunModel`` that
replaces the room rewards, a merchant of six free wares, and a two-relic chest.
The host keeps its decision shapes; these tests pin where Blitz differs.
"""

from typing import Any

import pytest
import sts2_client

HAS_BLITZ = sts2_client.GAME_VERSION in ("v0.113.0", "v0.114.0")
needs_blitz = pytest.mark.skipif(not HAS_BLITZ, reason="GameMode.Blitz exists from v0.113.0")

LINE = ["Monster", "Monster", "Shop", "Elite", "Treasure", "RestSite"]
CURSED_KINDS = {
    "Glam",
    "TwoMoreChoices",
    "CardRewardOfEachRarity",
    "UpgradedRareCardReward",
    "TwoRelics",
    "FullHealAndMaxHp",
}


def _start(game, seed: str = "BLITZ1", character: str = "Ironclad", **kw: Any) -> dict[str, Any]:
    state = game.start(character=character, seed=seed, game_mode="blitz", **kw)
    assert state.get("type") != "error", state
    return state


def _step(game, state: dict[str, Any]) -> dict[str, Any]:
    """One move of a fixed policy that never buys and never dies."""
    dec = state.get("decision")
    if dec == "combat_play":
        # The tests are about the rooms, not the fights: keep the player alive.
        if state["player"]["hp"] < 500:
            game.set_player(hp=999, max_hp=999)
        return game.auto_combat(state)
    if dec == "map_select":
        (choice,) = state["choices"]
        return game.select_map_node(choice["col"], choice["row"])
    if dec == "event_choice":
        option = next(o for o in state["options"] if not o.get("is_locked"))
        return game.choose_option(option["index"])
    if dec == "card_select":
        want = int(state.get("min_select") or 0) or 1
        return game.select_cards(",".join(str(c["index"]) for c in state["cards"][:want]))
    if dec == "card_reward":
        return game.select_card_reward(0)
    if dec == "rewards":
        return game.act("take_reward", reward_index=state["rewards"][0]["index"])
    if dec == "rest_site":
        option = next(o for o in state["options"] if o.get("is_enabled", True))
        return game.choose_option(option["index"])
    if dec == "shop":
        return game.leave_room()
    raise AssertionError(f"unexpected decision {dec!r}: {state}")


def _play_until(game, state: dict[str, Any], stop, max_steps: int = 4000) -> dict[str, Any]:
    for _ in range(max_steps):
        assert state.get("type") != "error", state
        if stop(state):
            return state
        assert state.get("decision") != "game_over", state
        state = _step(game, state)
    raise AssertionError("the run did not reach the wanted decision")


def _in_room(room_type: str, decision: str):
    return lambda s: s.get("decision") == decision and s["context"]["room_type"] == room_type


def _relic_ids(state: dict[str, Any]) -> list[str]:
    return [r["id"] for r in state["player"]["relics"]]


@needs_blitz
def test_a_blitz_run_starts_on_neows_single_option(game):
    state = _start(game)
    ctx = state["context"]
    assert ctx["game_mode"] == "blitz"
    assert ctx["ascension"] == 0
    assert state["decision"] == "event_choice"
    (option,) = state["options"]
    assert option["text_key"] == "BLITZ"


@needs_blitz
def test_the_map_is_the_eight_node_line_with_one_choice_per_step(game):
    state = _start(game)
    full = game.get_map()
    nodes = [node for row in full["rows"] for node in row]
    assert [n["type"] for n in nodes] == LINE
    assert [(n["col"], n["row"]) for n in nodes] == [(3, r) for r in range(1, 7)]
    assert (full["boss"]["col"], full["boss"]["row"], full["boss"]["type"]) == (3, 7, "Boss")
    # Node 0 of the line is the Ancient the act opens on.
    assert state["context"]["room_type"] == "Event"

    # Every act is the same line (no RNG builds it).
    dumped = game.send({"cmd": "dump_maps"})
    assert dumped.get("type") == "maps", dumped
    assert len(dumped["acts"]) == 3
    for act in dumped["acts"]:
        assert [n["type"] for row in act["rows"] for n in row] == LINE

    seen = []
    for _ in range(3):
        state = _play_until(game, state, lambda s: s.get("decision") == "map_select")
        (choice,) = state["choices"]
        seen.append((choice["col"], choice["row"], choice["type"]))
        state = _step(game, state)
    assert seen == [(3, 1, "Monster"), (3, 2, "Monster"), (3, 3, "Shop")]


@needs_blitz
def test_neow_gives_a_pick_of_three_out_of_ten_cards(game):
    state = _start(game)
    deck_before = state["player"]["deck_size"]
    state = game.choose_option(0)
    assert state["decision"] == "card_select", state
    assert len(state["cards"]) == 10
    assert (state["min_select"], state["max_select"]) == (3, 3)
    picked = [state["cards"][i]["id"] for i in (0, 4, 9)]
    state = game.select_cards("0,4,9")
    assert state["decision"] == "map_select", state
    assert state["player"]["deck_size"] == deck_before + 3
    deck_ids = [c["id"] for c in state["player"]["deck"]]
    for card_id in picked:
        assert card_id in deck_ids


@needs_blitz
def test_a_monster_gives_no_gold_and_an_elite_gives_two_relics(game):
    state = _start(game)
    gold = state["player"]["gold"]

    state = _play_until(game, state, _in_room("Monster", "card_reward"))
    assert state["gold_earned"] == 0
    assert state["player"]["gold"] == gold

    state = _play_until(game, state, _in_room("Elite", "combat_play"))
    relics_before = _relic_ids(state)
    state = _play_until(game, state, _in_room("Elite", "card_reward"))
    assert state["gold_earned"] == 0
    assert state["player"]["gold"] == gold
    assert len(_relic_ids(state)) == len(relics_before) + 2
    assert all(c["rarity"] == "Uncommon" for c in state["cards"])


@needs_blitz
def test_the_chest_gives_two_relics_over_two_rounds_and_no_gold(game):
    state = _start(game)
    state = _play_until(game, state, _in_room("Treasure", "rewards"))
    gold = state["player"]["gold"]
    relics_before = _relic_ids(state)
    assert state["chest_picks_left"] == 2
    assert [r["type"] for r in state["rewards"]] == ["relic", "relic"]
    assert [r["index"] for r in state["rewards"]] == [0, 1]
    first, second = state["rewards"]

    # Round 1: take the SECOND relic; the first stays for round 2.
    state = game.act("take_reward", reward_index=1)
    assert state["decision"] == "rewards", state
    assert state["chest_picks_left"] == 1
    assert [(r["index"], r["id"]) for r in state["rewards"]] == [(0, first["id"])]
    assert _relic_ids(state) == [*relics_before, second["id"]]

    state = game.act("take_reward", reward_index=0)
    assert state["decision"] == "map_select", state
    assert _relic_ids(state) == [*relics_before, second["id"], first["id"]]
    assert state["player"]["gold"] == gold
    (choice,) = state["choices"]
    assert choice["type"] == "RestSite"


@needs_blitz
def test_skipping_the_chest_leaves_both_relics(game):
    state = _start(game)
    state = _play_until(game, state, _in_room("Treasure", "rewards"))
    relics_before = _relic_ids(state)
    state = game.act("skip_reward", reward_index=0)
    assert state["decision"] == "map_select", state
    assert _relic_ids(state) == relics_before


def _follow_up(game, state: dict[str, Any]) -> dict[str, Any]:
    """Answer a ware's follow-up screens until the shop is back."""
    for _ in range(40):
        assert state.get("type") != "error", state
        if state.get("decision") == "shop":
            return state
        state = _step(game, state)
    raise AssertionError(f"the ware's follow-up never returned to the shop: {state}")


@needs_blitz
def test_the_merchant_offers_six_wares_and_three_picks(game):
    state = _start(game)
    state = _play_until(game, state, lambda s: s.get("decision") == "shop")
    assert state["blitz_choices_remaining"] == 3
    assert state["cards"] == [] and state["relics"] == [] and state["potions"] == []
    assert state["card_removal_cost"] is None

    leave, *wares = state["options"]
    assert (leave["index"], leave["id"], leave["action"]) == (0, "leave_shop", "leave_shop")
    assert len(wares) == 6
    for i, ware in enumerate(wares, start=1):
        assert ware["index"] == i
        assert ware["id"] == ware["action"] == "blitz_choice"
        assert ware["label"] == "blitz:" + ware["item"]["blitz_kind"]
        assert ware["price"] == 0 and ware["enabled"] is True
        assert set(ware["item"]) == {"blitz_kind", "cursed", "rare", "title", "description"}
        assert ware["item"]["title"] and "{" not in ware["item"]["title"]
    # Five plain wares, all different, then the one cursed ware.
    kinds = [w["item"]["blitz_kind"] for w in wares]
    assert len(set(kinds)) == 6
    assert [w["item"]["cursed"] for w in wares] == [False] * 5 + [True]
    assert kinds[5] in CURSED_KINDS and not CURSED_KINDS & set(kinds[:5])


@needs_blitz
def test_a_cursed_ware_adds_a_curse_and_the_picks_run_out(game):
    state = _start(game)
    state = _play_until(game, state, lambda s: s.get("decision") == "shop")
    cursed = state["options"][6]
    kind = cursed["item"]["blitz_kind"]
    curses_before = sum(c["type"] == "Curse" for c in state["player"]["deck"])

    state = _follow_up(game, game.blitz_choice(6))
    assert sum(c["type"] == "Curse" for c in state["player"]["deck"]) == curses_before + 1
    assert state["options"][6]["enabled"] is False
    assert state["options"][6]["status"] == "Used"
    # TwoMoreChoices hands back two picks for the one it costs.
    left = 4 if kind == "TwoMoreChoices" else 2
    assert state["blitz_choices_remaining"] == left

    # A used ware cannot be taken again.
    again = game.blitz_choice(6)
    assert again.get("type") == "error" and "Used" in again["message"], again

    index = 1
    while state["blitz_choices_remaining"] > 0 and index <= 5:
        before = state["blitz_choices_remaining"]
        state = _follow_up(game, game.blitz_choice(index))
        # RandomUpgrades with nothing to upgrade is refused and costs nothing.
        assert state["blitz_choices_remaining"] in (before - 1, before)
        index += 1
    assert state["blitz_choices_remaining"] == 0
    assert not any(o["enabled"] for o in state["options"][1:])
    refused = game.blitz_choice(5)
    assert refused.get("type") == "error", refused

    state = game.act("leave_shop")
    assert state["decision"] == "map_select", state
    (choice,) = state["choices"]
    assert choice["type"] == "Elite"


@needs_blitz
def test_the_merchant_can_be_left_with_picks_unspent(game):
    state = _start(game)
    state = _play_until(game, state, lambda s: s.get("decision") == "shop")
    state = game.leave_room()
    assert state["decision"] == "map_select", state


@needs_blitz
def test_dump_shop_rolls_a_blitz_inventory(game):
    _start(game)
    dumped = game.send({"cmd": "dump_shop"})
    assert dumped.get("type") == "shop", dumped
    assert dumped["game_mode"] == "blitz"
    assert len(dumped["blitz_entries"]) == 6
    assert dumped["blitz_choices_remaining"] == 3
    assert dumped["shops_reads"] >= 6


@needs_blitz
def test_the_blitz_level_runs_from_zero_to_five(game):
    state = game.start(character="Ironclad", seed="BLITZ1", game_mode="blitz", ascension=6)
    assert state.get("type") == "error"
    assert "0..5" in state["message"], state

    state = _start(game, ascension=5)
    assert state["context"]["ascension"] == 5
    # Blitz level 5 is Ascender's Bane; level 1 is Weary Traveler, not the
    # standard ladder's Swarming Elites (AscensionHelper.GetLevelForBlitz).
    assert "ASCENDERS_BANE" in " ".join(c["id"] for c in state["player"]["deck"])


@needs_blitz
def test_every_act_opens_on_its_ancient(game):
    state = _start(game)

    def next_act(s: dict[str, Any]) -> bool:
        return s["context"]["act"] == 2

    state = _play_until(game, state, next_act, max_steps=20000)
    # No map decision between the boss and the next act's Ancient.
    assert state["decision"] == "event_choice", state
    assert state["context"]["room_type"] == "Event"
    assert state["context"]["floor"] == 1


def test_a_standard_run_is_unchanged(game):
    state = game.start(character="Ironclad", seed="BLITZ1")
    assert "game_mode" not in state["context"]
    assert "ascension" not in state["context"]
    # Neow offers its usual three blessings, not the Blitz option.
    assert len(state["options"]) == 3
    assert all(o["text_key"] != "BLITZ" for o in state["options"])


@pytest.mark.skipif(HAS_BLITZ, reason="this build has Blitz")
def test_blitz_is_refused_on_a_build_without_it(game):
    state = game.start(character="Ironclad", seed="BLITZ1", game_mode="blitz")
    assert state.get("type") == "error"
    assert "v0.113.0" in state["message"]


def test_the_lockstep_choice_list_covers_the_blitz_shop_and_chest():
    from run_parity import enumerate_choices

    leave = {"index": 0, "id": "leave_shop", "action": "leave_shop", "enabled": True}
    wares = [
        {"index": 1, "id": "blitz_choice", "action": "blitz_choice", "enabled": True},
        {"index": 2, "id": "blitz_choice", "action": "blitz_choice", "enabled": False},
    ]
    shop = {"decision": "shop", "blitz_choices_remaining": 1, "options": [leave, *wares]}
    assert enumerate_choices(None, shop) == [("blitz_choice", {"index": 1}), ("leave_room", {})]
    assert enumerate_choices(None, {**shop, "blitz_choices_remaining": 0}) == [("leave_room", {})]

    chest = {"decision": "rewards", "chest_picks_left": 2, "rewards": [{"index": 0}, {"index": 1}]}
    assert enumerate_choices(None, chest) == [
        ("take_reward", {"reward_index": 0}),
        ("skip_reward", {"reward_index": 0}),
        ("take_reward", {"reward_index": 1}),
        ("skip_reward", {"reward_index": 1}),
        ("proceed", {}),
    ]


def test_the_lockstep_compares_what_a_blitz_decision_offers(monkeypatch):
    from run_parity import diff_blitz_offers

    ware = {"action": "blitz_choice", "label": "blitz:Relic", "status": "Available"}
    shop = {"decision": "shop", "blitz_choices_remaining": 3, "options": [ware]}
    other = {**shop, "options": [{**ware, "label": "blitz:Potions"}]}
    chest = {"decision": "rewards", "rewards": [{"index": 0, "id": "BRONZE_SCALES"}]}
    rs_chest = {"decision": "rewards", "rewards": [{"index": 0, "id": "RELIC.BRONZE_SCALES"}]}
    cards = {"decision": "card_reward", "cards": [{"id": "CARD.HEADBUTT"}]}

    # Standard and Dread sweeps are untouched.
    monkeypatch.delenv("RUN_PARITY_GAME_MODE", raising=False)
    assert diff_blitz_offers(shop, other) == []

    monkeypatch.setenv("RUN_PARITY_GAME_MODE", "blitz")
    assert diff_blitz_offers(shop, shop) == []
    assert diff_blitz_offers(chest, rs_chest) == []
    assert (
        diff_blitz_offers(cards, {"decision": "card_reward", "cards": [{"id": "HEADBUTT"}]}) == []
    )
    (wares,) = diff_blitz_offers(shop, other)
    assert wares.startswith("blitz.wares:")
    (picks,) = diff_blitz_offers(shop, {**shop, "blitz_choices_remaining": 2})
    assert picks.startswith("blitz.choices_remaining:")
    (relic,) = diff_blitz_offers(chest, {**rs_chest, "rewards": [{"index": 1, "id": "ANCHOR"}]})
    assert relic.startswith("blitz.chest:")
    (card,) = diff_blitz_offers(cards, {"decision": "card_reward", "cards": [{"id": "ANGER"}]})
    assert card.startswith("blitz.card_reward:")
    # Different decisions are the phase diff's to report.
    assert diff_blitz_offers(shop, chest) == []
