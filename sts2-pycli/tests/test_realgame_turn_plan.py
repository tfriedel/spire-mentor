"""``advice.plan`` and the per-option move fields (PROTOCOL.md §5, v5.3).

The plan numbers this turn's cards for the HUD. Every step after the first
indexes a hand that exists only once the earlier ones are played, so these
pin the mapping back to the hand the request sent, and the cuts: a card
that changes the hand beyond leaving it ends the plan after itself, a move
that cannot be mapped ends it before itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="the plan walks a native fight")

import realgame_advisor as ra  # noqa: E402
import realgame_turn_plan as tp  # noqa: E402
from realgame_adapter import to_headless_combat_play  # noqa: E402
from sts2_hint import adapter  # noqa: E402
from sts2_hint.recommender import (  # noqa: E402
    END_TURN_STABLE_IDX,
    POTION_STABLE_IDX_BASE,
    HintTag,
)

FIXTURES = Path(__file__).parent / "fixtures" / "realgame_adapter"


def _board(name: str = "multi_enemy_ironclad.json") -> dict[str, Any]:
    """Hand: Defend, Defend, Defend, Strike, Strike; three enemies."""
    state = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["state"]
    state["type"] = "combat_action"
    return state


def _fight(state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    wire = adapter.translate(
        to_headless_combat_play(state), character="Ironclad", combat_id=0, seed=0, ascension=0
    )
    return sts2_rs.state_from_dict(wire), wire


def _play(i: int, target: int | None = None) -> dict[str, Any]:
    return {"action": "play", "card_index": i, "target_index": target}


END = {"action": "end_turn"}


def test_later_steps_index_the_hand_the_request_sent() -> None:
    state = _board()
    cs, wire = _fight(state)
    # Defend (0), then the engine's Strike at index 3 of the 4-card hand left
    # -- which is the request's card 4 -- then the other Strike, then end.
    pv = [_play(0), _play(3, 0), _play(2, 0), END]
    best = {"action": "play", "card_index": 0, "label": "Defend"}
    plan = tp.search_plan(cs, pv, wire, state, best)
    assert [(s["action"], s.get("card_index"), s.get("target_index")) for s in plan] == [
        ("play", 0, None),
        ("play", 4, 0),
        ("play", 3, 0),
        ("end_turn", None, None),
    ]
    assert plan[0]["label"] == "Defend"
    assert plan[1]["label"].startswith(state["hand"][4]["name"])
    assert plan[-1]["label"] == "End turn"


def test_the_fight_the_search_ran_on_is_not_advanced() -> None:
    state = _board()
    cs, wire = _fight(state)
    before = cs.get_state()
    tp.search_plan(cs, [_play(0), END], wire, state, {"action": "play", "card_index": 0})
    assert cs.get_state() == before


def test_a_line_that_does_not_open_with_best_is_best_alone() -> None:
    state = _board()
    cs, wire = _fight(state)
    best = {"action": "play", "card_index": 3, "target_index": 0}
    plan = tp.search_plan(cs, [_play(0), _play(2), END], wire, state, best)
    assert len(plan) == 1
    assert plan[0]["card_index"] == 3 and plan[0]["target_index"] == 0


def test_a_card_that_draws_ends_the_plan_after_itself() -> None:
    state = _board()
    state["hand"][0] = {**state["hand"][0], "id": "SHRUG_IT_OFF", "name": "Shrug It Off"}
    cs, wire = _fight(state)
    plan = tp.search_plan(
        cs, [_play(0), _play(2, 0), END], wire, state, {"action": "play", "card_index": 0}
    )
    assert [s.get("card_index") for s in plan] == [0]


def test_an_unmappable_move_ends_the_plan_before_itself() -> None:
    state = _board()
    cs, wire = _fight(state)
    pv = [_play(0), {"action": "choose", "selection": [1]}, END]
    plan = tp.search_plan(cs, pv, wire, state, {"action": "play", "card_index": 0})
    assert [s["action"] for s in plan] == ["play"]


def test_a_short_pv_is_extended_by_re_searching() -> None:
    state = _board()
    cs, wire = _fight(state)
    seen: list[int] = []

    def research(fight: Any) -> list[Any]:
        seen.append(len(fight.get_state()["hand"]))
        return [_play(0), END] if len(seen) == 1 else [END]

    plan = tp.search_plan(
        cs, [_play(0)], wire, state, {"action": "play", "card_index": 0}, research
    )
    assert seen == [4], "the re-search sees the fight after step 1"
    assert [(s["action"], s.get("card_index")) for s in plan] == [
        ("play", 0),
        ("play", 1),
        ("end_turn", None),
    ]


def test_an_exhausted_budget_ends_the_plan_where_it_got_to() -> None:
    state = _board()
    cs, wire = _fight(state)
    plan = tp.search_plan(
        cs,
        [_play(0)],
        wire,
        state,
        {"action": "play", "card_index": 0},
        lambda fight: [_play(0)],
        budget_s=0.0,
    )
    assert len(plan) == 1


def test_end_turn_alone_is_a_plan() -> None:
    state = _board()
    cs, wire = _fight(state)
    plan = tp.search_plan(cs, [END], wire, state, {"action": "end_turn"})
    assert plan == [{"action": "end_turn", "label": "End turn"}]


def test_compute_combat_advice_sends_a_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end with a canned search: step 1 is ``best``; the rest comes
    from the re-search the advisor wires in."""
    state = _board()
    calls: list[int] = []

    def fake(cs: Any, **kw: Any) -> dict[str, Any]:
        calls.append(kw["sims"])
        if len(calls) == 1:
            return {
                "actions": [
                    _row(_play(3, 1), 0.9, 500, best=True),
                    _row(_play(0), 0.8, 300),
                    _row(END, 0.5, 100),
                ],
                "pv": [_play(3, 1)],
            }
        return {"actions": [_row(END, 0.5, 10, best=True)], "pv": [END]}

    monkeypatch.setattr(sts2_rs, "mcts_recommend", fake)
    advice, *_ = ra.compute_combat_advice(state, ra.RecommendConfig(threads=1), 4000)
    assert advice["best"]["card_index"] == 3
    assert [(s["action"], s.get("card_index")) for s in advice["plan"]] == [
        ("play", 3),
        ("end_turn", None),
    ]
    assert calls == [4000, ra.PLAN_MIN_SIMS]
    by_kind = {o["action"] for o in advice["options"]}
    assert by_kind == {"play", "end_turn"}
    best = next(o for o in advice["options"] if o["is_best"])
    assert best["hp_mean"] == pytest.approx(59.0)


def _row(action: dict[str, Any], p_win: float, visits: int, *, best: bool = False) -> dict:
    wins = round(p_win * visits)
    return {
        "action": action,
        "p_win": p_win,
        "q": p_win,
        "visits": visits,
        "wins": wins,
        "losses": visits - wins,
        "ties": 0,
        "is_best": best,
        "mean_final_hp": 50.0 + 10 * p_win,
        "hp_n": visits,
    }


def _tag(stable_idx: int, target: int | None = None) -> HintTag:
    return HintTag(
        stable_idx=stable_idx,
        win_prob=0.5,
        bucket_label="",
        is_best=False,
        visits=1,
        q_saturated=False,
        low_confidence=False,
        target_index=target,
    )


def test_every_option_row_says_what_it_does() -> None:
    state = {
        "hand": [{"name": "Strike"}],
        "potions": [
            {"slot": 0, "empty": True},
            {"slot": 1, "id": "FIRE_POTION", "name": "Fire Potion"},
        ],
    }
    assert ra.option_move(_tag(0, 0), state) == {"action": "play"}
    assert ra.option_move(_tag(END_TURN_STABLE_IDX), state) == {"action": "end_turn"}
    # The engine's potion 0 is the first HELD potion: belt slot 1.
    assert ra.option_move(_tag(POTION_STABLE_IDX_BASE - 0, 0), state) == {
        "action": "potion",
        "slot": 1,
    }
    assert ra.option_move(_tag(POTION_STABLE_IDX_BASE - 1), state) == {"action": "potion"}
    assert ra.option_move(_tag(ra.CHOOSE_STABLE_IDX_BASE - 2), state) == {"action": "choose"}
