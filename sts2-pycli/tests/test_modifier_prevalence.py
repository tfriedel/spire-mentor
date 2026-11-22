"""`ppo.modifier_prevalence.option_enchantment` on card-grant grids."""

from __future__ import annotations

from ppo.modifier_prevalence import option_enchantment

OBS = {
    "card_grant_options": [["STRIKE_IRONCLAD", False], ["BASH", False]],
    "card_grant_option_enchantments": [None, ["SHARP", 3]],
}


def test_grant_options_are_card_options():
    assert option_enchantment(OBS, {"action": "grant_deck_card", "idx": 0}) == (True, None)
    assert option_enchantment(OBS, {"action": "grant_deck_card", "idx": 1}) == (
        True,
        ["SHARP", 3],
    )


def test_optional_grant_confirm_is_not_a_card_option():
    # Sea Glass: `idx == len(options)` confirms the grid with no (more) picks.
    assert option_enchantment(OBS, {"action": "grant_deck_card", "idx": 2}) == (False, None)
