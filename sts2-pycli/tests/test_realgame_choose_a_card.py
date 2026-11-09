"""Choose-a-card screens (``surface: "choose_a_card"``), both brains.

The game opens one screen for two different things: a card generated INTO A
FIGHT (Discovery, the Attack / Skill / Power Potions) and, out of a fight, a
card for the DECK (Lead Paperweight). In the first live beta.6 run
(2026-11-08, ``logs/2026-11-08-beta6-live``) 79 of 289 answers fell from
Instinct to the search on these screens, and the search answered the fight
ones with its deck-pick head ("is this a good deck card"):

- in a fight, the mod sent no ``source_model``, so the fight mirror could not
  replay what opened the screen (it now does, 0.10.0-beta.7);
- out of a fight, Instinct declined as if the card were offered into a fight.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

import realgame_advisor as ra  # noqa: E402
from test_advisor_mirror import HONEST, _potion_screen  # noqa: E402
from test_display_names import _StubPolicies, _StubPolicy  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "realgame_instinct"


def _attack_potion_offer(with_source: bool) -> tuple[Any, dict[str, Any]]:
    """An Attack Potion's offer, shaped as the mod sends it (DescribeChooseACard)."""
    advisor, select = _potion_screen("ATTACK_POTION")
    state: dict[str, Any] = {
        "type": "card_reward",
        "advice_for": "card_reward",
        "surface": "choose_a_card",
        "cards": select["cards"],
        "can_skip": True,
        "combat": select["combat"],
        "run_state": select["run_state"],
    }
    if with_source:
        state.update(
            source="FromChooseACardScreen",
            prompt_key="TO_OBTAIN",
            source_model="ATTACK_POTION",
            source_kind="potion",
        )
    return advisor, state


def test_search_answers_a_fight_offer_with_the_combat_search() -> None:
    advisor, state = _attack_potion_offer(with_source=True)
    advice = ra.compute_screen_advice(
        state, "card_reward", _StubPolicies(_StubPolicy(0)), mirror=advisor, cfg=HONEST, sims=8
    )
    assert advice["note"].startswith("combat MCTS"), advice["note"]
    assert advice["best"]["index"] in {0, 1, 2}
    assert not advisor.declines, advisor.declines


def test_a_fight_offer_without_its_source_still_gets_the_deck_pick_answer() -> None:
    """An older mod build (no ``source_model``): the deck-pick head, as before."""
    advisor, state = _attack_potion_offer(with_source=False)
    advice = ra.compute_screen_advice(
        state, "card_reward", _StubPolicies(_StubPolicy(0)), mirror=advisor, cfg=HONEST, sims=8
    )
    assert "this copy only lasts the fight" in advice["note"]
    assert advisor.declines == {"wire named no source_model": 1}


def test_a_fight_offer_is_the_in_combat_select_it_is() -> None:
    _advisor, state = _attack_potion_offer(with_source=True)
    select = ra.fight_offer_as_select(state)
    assert select["advice_for"] == "card_select"
    assert (select["min_select"], select["max_select"], select["cancelable"]) == (1, 1, True)
    assert select["source_model"] == "ATTACK_POTION"
    assert ra.is_fight_offer(state)
    assert not ra.is_fight_offer({**state, "combat": None})


def _lead_paperweight() -> dict[str, Any]:
    """The live Lead Paperweight screen at Neow (floor 1, no fight)."""
    msg = json.loads((FIXTURES / "lead_paperweight_choose_a_card.json").read_text(encoding="utf-8"))
    return {**msg, "type": msg["advice_for"]}


def test_search_does_not_call_a_deck_card_a_fight_copy() -> None:
    state = _lead_paperweight()
    advice = ra.compute_screen_advice(state, "card_reward", _StubPolicies(_StubPolicy(0)))
    assert "only lasts the fight" not in advice["note"]
    assert advice["best"]["label"] == "Discovery"


def test_instinct_answers_the_out_of_fight_offer() -> None:
    import realgame_instinct as ri

    if not ri.default_weights_path().is_file():
        pytest.skip("no Instinct weights (data/e2e)")
    state = _lead_paperweight()
    advice = ri.InstinctAdvisor().advise(copy.deepcopy(state), "card_reward", session={})
    assert advice["brain"] == "instinct"
    labels = {o["label"] for o in advice["options"]}
    assert {"Discovery", "Production"} <= labels
    assert advice["best"]["label"] in labels
