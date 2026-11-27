"""GH #1286: the host starts a run on a stated player profile.

A run depends on the player's progress save: the timeline epochs revealed (the
card, relic, potion and event pools, the ancients, Underdocks), the bosses met
(an act's boss is the first unseen one in its discovery order), the acts
entered, and whether it is the first run ever. The host has no save and starts
with everything unlocked, so a recorded run of a player who has not unlocked
everything parted from it within a few rooms. `start_run.profile` states those
facts; `epochs` lists the ids and `probe_profiles` runs the start-of-run draws
for many profiles in one process.
"""

from __future__ import annotations

from typing import Any

import pytest
import sts2_client
from sts2_client import Game

SEED = "PROFILE1"

#: v0.113 shuffles a list off ONE draw from the caller's RNG: `UnstableShuffle`
#: seeds a child `Rng` with `rng.NextUnsignedLong()` (ListExtensions.cs:45-52).
#: Earlier builds draw `NextInt` once per item, so there the relics a profile
#: locks move the `up_front` counter, and on v0.113 they do not (GH #1781).
ONE_DRAW_SHUFFLE = sts2_client.GAME_VERSION in ("v0.113.0", "v0.114.0")


def _rooms(game: Any) -> list[dict[str, Any]]:
    return game.send({"cmd": "dump_rooms"})["acts"]


def _up_front(game: Any) -> int:
    return int(game.get_state()["rng_counters"]["up_front"])


@pytest.fixture
def second_game(game):
    g = Game()
    yield g
    g.close()


def test_epochs_lists_what_each_epoch_unlocks(game):
    out = game.epochs()
    epochs = out["epochs"]
    assert len(epochs["RELIC1_EPOCH"]["relics"]) == 3
    assert len(epochs["SILENT2_EPOCH"]["cards"]) == 3
    assert len(epochs["POTION1_EPOCH"]["potions"]) == 3
    assert epochs["EVENT1_EPOCH"]["events"]
    assert epochs["NEOW_EPOCH"] == {}
    overgrowth = out["acts"]["OVERGROWTH"]
    assert overgrowth["boss_discovery_order"][0] == "VANTOM_BOSS"
    assert sorted(overgrowth["boss_discovery_order"]) == sorted(overgrowth["bosses"])


def test_a_locked_relic_epoch_changes_the_start_of_run_draws(game, second_game):
    game.start(seed=SEED, act1="overgrowth", profile={})
    second_game.start(seed=SEED, act1="overgrowth", profile={"locked_epochs": ["RELIC1_EPOCH"]})
    # The relic bags are shuffled off `up_front`: one draw a relic, or one a
    # bag on v0.113, whatever the bag holds.
    if ONE_DRAW_SHUFFLE:
        assert _up_front(game) == _up_front(second_game)
    else:
        assert _up_front(game) != _up_front(second_game)


def test_an_empty_profile_is_the_default_run(game, second_game):
    first = game.start(seed=SEED)
    second = second_game.start(seed=SEED, profile={})
    assert first["decision"] == second["decision"] == "event_choice"
    assert _rooms(game) == _rooms(second_game)
    assert _up_front(game) == _up_front(second_game)


def test_without_neow_revealed_the_run_opens_on_the_map(game):
    state = game.start(seed=SEED, profile={"locked_epochs": ["NEOW_EPOCH"]})
    assert state["decision"] == "map_select"


def test_the_first_unseen_boss_is_the_acts_boss(game, second_game):
    order = game.epochs()["acts"]["OVERGROWTH"]["boss_discovery_order"]
    game.start(seed=SEED, act1="overgrowth", profile={"unseen_encounters": order[1:]})
    assert _rooms(game)[0]["boss"] == order[1]
    # With every boss met the seed decides, as it does without a profile.
    second_game.start(seed=SEED, act1="overgrowth", profile={})
    rolled = _rooms(second_game)[0]["boss"]
    assert rolled in order


def test_the_first_run_ever_has_its_rooms_in_a_set_order(game):
    game.start(seed=SEED, act1="overgrowth", profile={"number_of_runs": 0})
    act = _rooms(game)[0]
    # Overgrowth.ApplyActDiscoveryOrderModifications.
    assert act["normal"][:3] == ["NIBBITS_WEAK", "SLIMES_WEAK", "SHRINKER_BEETLE_WEAK"]
    assert act["events"][:2] == ["BYRDONIS_NEST", "SAPPHIRE_SEED"]
    assert act["elite"][:2] == ["BYRDONIS_ELITE", "PHROG_PARASITE_ELITE"]


def test_an_act_never_entered_is_forced(game, second_game):
    state = game.start(seed=SEED, profile={"undiscovered_acts": ["UNDERDOCKS"]})
    assert state["context"]["act_name"].upper() == "UNDERDOCKS"
    state = second_game.start(seed=SEED, profile={"locked_epochs": ["UNDERDOCKS_EPOCH"]})
    assert state["context"]["act_name"].upper() == "OVERGROWTH"


def test_unknown_ids_are_refused(game):
    out = game.start(seed=SEED, profile={"locked_epochs": ["NO_SUCH_EPOCH"]})
    assert out["type"] == "error"
    assert "NO_SUCH_EPOCH" in out["message"]


@pytest.mark.parametrize(
    ("ascension", "profile"),
    [
        (0, {}),
        (0, {"locked_epochs": ["RELIC2_EPOCH", "EVENT1_EPOCH", "DARV_EPOCH", "IRONCLAD3_EPOCH"]}),
        (10, {"locked_epochs": ["OROBAS_EPOCH", "RELIC5_EPOCH"]}),
        (0, {"locked_epochs": ["NEOW_EPOCH", "UNDERDOCKS_EPOCH"], "number_of_runs": 0}),
        (0, {"unseen_encounters": ["THE_KIN_BOSS", "KAISER_CRAB_BOSS"]}),
    ],
)
def test_the_probe_draws_what_a_start_draws(game, second_game, ascension, profile):
    # `probe_profiles` runs the game's start-of-run steps on a RunState no
    # RunManager owns. It must end where a real start ends, or the replay
    # harness would search with a wrong map of the profiles.
    game.start(seed=SEED, ascension=ascension, profile=profile)
    started = _rooms(game)
    probed = second_game.send(
        {
            "cmd": "probe_profiles",
            "character": "Ironclad",
            "ascension": ascension,
            "seed": SEED,
            "profiles": [{"locked_epochs": ["RELIC1_EPOCH"]}, profile],
        }
    )
    assert probed["type"] == "profiles" and probed["tried"] == 2
    mine = probed["results"][1]
    assert mine["up_front"] == _up_front(game)
    for act, room_set in zip(mine["acts"], started, strict=True):
        for key in ("boss", "ancient", "events", "normal", "elite"):
            assert act[key] == room_set[key], key


def test_the_probe_keeps_only_profiles_on_the_counter(game):
    profiles = [{}, {"locked_epochs": ["RELIC1_EPOCH"]}]
    every = game.send({"cmd": "probe_profiles", "seed": SEED, "profiles": profiles})["results"]
    assert [r["index"] for r in every] == [0, 1]
    kept = game.send(
        {
            "cmd": "probe_profiles",
            "seed": SEED,
            "profiles": profiles,
            "up_front": every[1]["up_front"],
        }
    )["results"]
    # On v0.113 the counter cannot tell a locked relic epoch apart; the bag
    # order can (`test_the_probe_lists_the_relic_bag_in_its_order`).
    assert [r["index"] for r in kept] == ([0, 1] if ONE_DRAW_SHUFFLE else [1])


def test_the_probe_lists_the_relic_bag_in_its_order(game):
    # Two sets locking as many relics end on one counter with the bag in
    # another order; the harness tells them apart by where the log's relics lie.
    epochs = game.epochs()["epochs"]
    locked = epochs["RELIC1_EPOCH"]["relics"]
    profiles = [{}, {"locked_epochs": ["RELIC1_EPOCH"]}]
    full, less = (
        r["relics"]
        for r in game.send({"cmd": "probe_profiles", "seed": SEED, "profiles": profiles})["results"]
    )
    assert set(full) == {"common", "uncommon", "rare", "shop"}
    all_full = [x for xs in full.values() for x in xs]
    all_less = [x for xs in less.values() for x in xs]
    assert len(all_full) == len(set(all_full))
    assert set(all_full) - set(all_less) == set(locked)
