"""Headless star-cost fields for a card made free this turn.

C# ``SetToFreeThisTurn`` (Splash, Discovery, Infernal Blade, the generative
potions) zeroes the STAR cost along with the energy cost
(``CardModel.cs:1267-1271``). The headless ``SerializeCombatCard`` used to ship
the printed ``BaseStarCost`` as ``star_cost`` and veto ``can_play`` against it
(BUG-007), so a Silent's Splash'd Regent attack -- free in the game -- read as
costing stars the Silent never has, and as unplayable. Same fields as the
mod's ``SimStateExtractor`` (``PROTOCOL.md`` "Card object").

Seed ``STARPROBE14`` (v0.107.1) / ``STARPROBE31`` (v0.111.0) /
``STARPROBE69`` (v0.113.0): the Splash offer holds Astral Pulse (printed 3
stars at every build, ``AstralPulse.cs``).
"""

from __future__ import annotations

import sts2_client

STAR_CARD = "CARD.ASTRAL_PULSE"

#: A Silent seed whose first Splash offers `STAR_CARD`, per game build: v0.111.0
#: rehashed every seed (XxHash64), so STARPROBE14 offers other cards there;
#: STARPROBE69 was found on the v0.113.0 C# headless (STARPROBE1..68 offer none).
_STAR_SEED = {"v0.107.1": "STARPROBE14", "v0.111.0": "STARPROBE31", "v0.113.0": "STARPROBE69"}


def test_splashed_star_card_is_free_in_stars_and_playable(game):
    state = game.start(character="Silent", seed=_STAR_SEED[sts2_client.GAME_VERSION])
    game.skip_neow(state)
    r = game.set_player(hp=70, max_hp=80, deck=["SPLASH"] * 5, relics=[], potions=[])
    assert r.get("type") != "error", r
    game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    offer = game.play_card(0, None)
    assert offer.get("decision") == "card_select", offer
    ids = [c.get("id") for c in offer.get("cards", [])]
    assert STAR_CARD in ids, f"seed no longer offers {STAR_CARD}: {ids}"

    after = game.select_cards(str(ids.index(STAR_CARD)))
    picked = [c for c in after.get("hand") or [] if c.get("id") == STAR_CARD]
    assert len(picked) == 1, after.get("hand")
    card = picked[0]
    assert card["star_cost"] == 0, "free this turn zeroes the star cost"
    assert card["persistent_star_cost"] == 3, "and it springs back to the printed 3"
    assert card["can_play"] is True, "a star-less Silent can play the free card"
    legal = game.legal_actions()["actions"]
    assert {"action": "play_card", "card_index": card["index"], "target_index": None} in legal
