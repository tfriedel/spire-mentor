"""The host sends the text the game shows, not a key or an unfilled template (GH #1324).

The live same-seed diff (#1287) found the host sending ``Insert {Rarity} Potion``
for both *The Future of Potions?* options: the agent could not tell which potion
each one costs. The cause was the host's ``LocTable.GetRawText`` patch, which
returned the key for every entry, so the game's own ``GetFormattedText`` never
saw a template. Names had a second cause: the host built ``<id>.title`` keys
instead of using the model's own ``Title`` (potion powers are named after their
potion, Decimillipede segments after ``DECIMILLIPEDE_SEGMENT``).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.oracle

UNLOCALIZED = re.compile(r"^[A-Z0-9_]+\.(?:title|name)$")
PLACEHOLDER = re.compile(r"\{[A-Za-z]\w*\}")


def test_the_future_of_potions_names_each_options_potion(game):
    game.send({"cmd": "start_run", "character": "Ironclad", "seed": "FUTURE1", "ascension": 0})
    game.send({"cmd": "set_player", "potions": ["STRENGTH_POTION", "ENERGY_POTION"]})
    state = game.send({"cmd": "enter_room", "type": "event", "event": "THE_FUTURE_OF_POTIONS"})

    assert state["decision"] == "event_choice"
    options = state["options"]
    assert [o["title"] for o in options] == ["Insert Common Potion"] * 2
    descriptions = [o["description"] for o in options]
    assert descriptions[0].startswith("Lose Strength Potion. Obtain an Upgraded Common ")
    assert descriptions[1].startswith("Lose Energy Potion. Obtain an Upgraded Common ")
    for text in [state["event_name"], *(o["title"] for o in options), *descriptions]:
        assert not PLACEHOLDER.search(text), text


def test_decimillipede_segments_and_a_potion_power_have_display_names(game):
    game.send({"cmd": "start_run", "character": "Ironclad", "seed": "NAMES1", "ascension": 0})
    game.send({"cmd": "set_player", "potions": ["SPEED_POTION"]})
    state = game.send({"cmd": "enter_room", "type": "monster", "encounter": "DECIMILLIPEDE_ELITE"})
    assert state["decision"] == "combat_play"
    assert {e["name"] for e in state["enemies"]} == {"Decimillipede"}

    state = game.send({"cmd": "action", "action": "use_potion", "args": {"potion_index": 0}})
    names = {p["id"]: p["name"] for p in state["player_powers"]}
    assert names.get("SPEED_POTION_POWER") == "Speed Potion", names
    for name in [*names.values(), *(e["name"] for e in state["enemies"])]:
        assert not UNLOCALIZED.match(name), name
