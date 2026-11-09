"""Knowledge Demon's curse pick cannot be skipped (#1181 H4).

``KnowledgeDemon.CurseOfKnowledgeMove`` offers its curses through
``CardSelectCmd.FromChooseACardScreen(..., canSkip: false)``
(``KnowledgeDemon.cs:187``). Under a test ``Selector`` that method asks for
0-1 cards whatever ``canSkip`` says (``CardSelectCmd.cs:265-267``), so the
headless used to accept a skip the real screen does not offer, and the run
split from Rust (F4 seed 1022). The headless now reports ``min_select=1`` for
it and refuses ``skip_select`` and an empty ``select_cards``.
"""

from __future__ import annotations

from typing import Any


def _to_curse_pick(game: Any) -> dict[str, Any]:
    state = game.start(seed="kd1181")
    game.skip_neow(state)
    state = game.enter_room("combat", encounter="KNOWLEDGE_DEMON_BOSS")
    for _ in range(8):
        if state.get("decision") == "card_select":
            return state
        state = game.end_turn()
    raise AssertionError(f"no curse pick: {state.get('decision')}")


def test_the_curse_pick_reports_a_forced_pick(game: Any) -> None:
    state = _to_curse_pick(game)
    assert state["source"] == "FromChooseACardScreen", state["source"]
    assert state["cancelable"] is False
    assert state["min_select"] == 1
    assert state["max_select"] == 1


def test_the_curse_pick_refuses_a_skip_and_takes_a_pick(game: Any) -> None:
    state = _to_curse_pick(game)
    refused = game.act("skip_select")
    assert refused["type"] == "error", refused
    assert "cannot be skipped" in refused["message"]
    # Still pending: the pick goes through.
    picked = game.act("select_cards", indices="0")
    assert picked.get("type") != "error", picked
    assert picked.get("decision") != "card_select" or picked.get("source") != state["source"]
