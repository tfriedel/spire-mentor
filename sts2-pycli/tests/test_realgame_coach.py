"""The live coach (``realgame_coach``): per-turn and per-choice grades.

Canned search rows and a stub engine -- no ``sts2_rs`` needed. A "position" is
a plain :class:`_Pos`; the stub engine reads hands from it, and records which
searches the grader asked for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import realgame_coach as rc
from display_names import display_name


@dataclass
class _Pos:
    hand: list[tuple[str, bool]] = field(default_factory=list)
    potions: list[str | None] = field(default_factory=list)
    round: int = 3
    board: dict[str, Any] | None = None


class _Engine:
    """Stub :class:`realgame_coach.CoachEngine`."""

    def __init__(self) -> None:
        self.top_up_rows: list[dict[str, Any] | None] = []
        self.catch_up_rows: list[list[dict[str, Any]]] = []
        self.catch_up_calls: list[Any] = []
        self.top_up_calls: list[tuple[Any, dict[str, Any]]] = []
        self.line = ["Bash", "Defend", "end turn"]
        #: The board the AI line ends its turn from (``None``: stopped short).
        self.ai_end: dict[str, Any] | None = None

    def position_info(self, cs: Any, msg: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"hand": cs.hand, "potions": cs.potions, "round": cs.round}

    def top_up(self, cs: Any, native: dict[str, Any], cfg: Any, sims: int) -> Any:
        self.top_up_calls.append((cs, native))
        return self.top_up_rows.pop(0) if self.top_up_rows else None

    def catch_up(self, cs: Any, cfg: Any, sims: int) -> list[dict[str, Any]]:
        self.catch_up_calls.append(cs)
        return self.catch_up_rows.pop(0)

    def ai_line(self, cs: Any, first: Any, cfg: Any, label: Any) -> list[str]:
        return list(self.line)

    def ai_turn(self, cs: Any, first: Any, cfg: Any, label: Any) -> tuple[list[str], Any]:
        return list(self.line), self.ai_end

    def board(self, cs: Any) -> dict[str, Any] | None:
        return cs.board if cs is not None else None


def _row(
    stable_idx: int,
    win: float,
    hp: float,
    *,
    visits: int = 400,
    best: bool = False,
    target: int | None = None,
    label: str = "?",
) -> dict[str, Any]:
    return {
        "stable_idx": stable_idx,
        "action": label,
        "win_prob": win,
        "q": win,
        "visits": visits,
        "is_best": best,
        "target_index": target,
        "mean_final_hp": hp,
        "hp_n": visits,
    }


HAND = [("BASH", False), ("DEFEND_IRONCLAD", False), ("STRIKE_IRONCLAD", False)]


def _grader(**kw: Any) -> tuple[rc.CoachGrader, list[dict], list[dict], _Engine]:
    sent: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    engine = _Engine()
    g = rc.CoachGrader(send=sent.append, log=logged.append, engine=engine, **kw)
    g.on_advice_request({"capabilities": {"coach": 1}, "floor": 5})
    return g, sent, logged, engine


def _combat_entry(g: rc.CoachGrader, rid: str, sid: int, rows: list[dict], pos: _Pos) -> None:
    entry = g.cache.put(
        rid, sid, kind="combat", rows=rows, cs=pos, msg={"round": 3, "floor": 5, "hand": []}
    )
    g.on_advice(entry)


def _play(card: str, rid: str | None, pos_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "play",
        "detail": card.title(),
        "card": {"id": card, "upgraded": False},
        "target_index": None,
        "advice_request_id": rid,
        "position_id": pos_id,
        **extra,
    }


def _end(rid: str | None, pos_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "end_turn",
        "detail": "End turn",
        "advice_request_id": rid,
        "position_id": pos_id,
        **extra,
    }


def _turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "turn"]


def _on_line_turn(g: rc.CoachGrader, *, end_rows: list[dict] | None = None, **end_kw: Any) -> None:
    pos = _Pos(hand=list(HAND))
    _combat_entry(g, "a1", 1, [_row(0, 0.95, 50, best=True), _row(1, 0.94, 45)], pos)
    g.on_human_action(_play("BASH", "a1", 1), node_id=10)
    pos2 = _Pos(hand=HAND[1:])
    _combat_entry(g, "a2", 2, [_row(0, 0.95, 50, best=True), _row(-1, 0.94, 44)], pos2)
    g.on_human_action(_play("DEFEND_IRONCLAD", "a2", 2), node_id=11)
    pos3 = _Pos(hand=HAND[2:])
    rows = end_rows or [_row(0, 0.95, 50, best=True), _row(-1, 0.95, 50)]
    _combat_entry(g, "a3", 3, rows, pos3)
    g.on_human_action(_end("a3", 3, **end_kw), node_id=12)


def _lagavulin(*, asleep: bool, dead: bool = False, uid: int = 7) -> dict[str, Any]:
    powers = [{"name": "ASLEEP", "amount": 3}] if asleep else []
    return {"kind": "LAGAVULIN_MATRIARCH", "uid": uid, "is_dead": dead, "powers": powers}


def _board(*enemies: dict[str, Any], phase: str = "PlayerPhase") -> dict[str, Any]:
    return {"phase": phase, "encounter": list(enemies)}


class TestWhy:
    """``board_events`` / ``why_text``: the recap's reason line (GH #888)."""

    def test_wake_kill_and_win_events(self) -> None:
        start = _board(_lagavulin(asleep=True), _lagavulin(asleep=False, uid=8))
        end = _board(_lagavulin(asleep=False), _lagavulin(asleep=False, dead=True, uid=8))
        assert rc.board_events(start, end) == [("wakes", 0), ("kills", 1)]
        assert rc.board_events(start, _board(phase="Victory")) == [("wins", -1)]
        assert rc.board_events(start, start) == []

    def test_slumber_counts_as_sleep(self) -> None:
        sleeping = {**_lagavulin(asleep=False), "powers": [{"name": "SLUMBER"}]}
        assert rc.board_events(_board(sleeping), _board(_lagavulin(asleep=False))) == [("wakes", 0)]

    def test_why_names_each_sides_extra_events(self) -> None:
        names = ["Lagavulin Matriarch", "Cultist"]
        assert (
            rc.why_text([("wakes", 0)], [], names, 1)
            == "Your line wakes Lagavulin Matriarch on turn 1"
        )
        assert (
            rc.why_text([("wakes", 0)], [("wakes", 0), ("kills", 1)], names, 2)
            == "The AI's line kills Cultist on turn 2"
        )
        assert rc.why_text([("wakes", 0)], [("wins", -1)], names, 4) == (
            "Your line wakes Lagavulin Matriarch; the AI's line wins the fight on turn 4"
        )
        assert rc.why_text([("kills", 1)], [("kills", 1)], names, 1) is None


class TestCombatTurns:
    def test_wasted_turn_emits_blunder_turn_record(self) -> None:
        g, sent, logged, _ = _grader()
        # Two plays on the line, then an end turn that leaves 12 HP on the table.
        _on_line_turn(
            g,
            end_rows=[_row(0, 0.95, 50, best=True), _row(-1, 0.94, 38, visits=200)],
        )
        (turn,) = _turns(sent)
        assert turn["verdict"] == "blunder"
        assert turn["hp_drop"] == 12
        assert turn["you"][-1] == "end turn"
        # An id the fight never named is shown by its display name
        # (`Defend` from the localization tables; prettified without them).
        assert turn["you"][:2] == ["Bash", display_name("DEFEND_IRONCLAD")]
        assert turn["ai"] == ["Bash", "Defend", "end turn"]
        assert turn["id"] == "turn:10" and turn["node_id"] == 10
        assert turn["turn"] == 3 and turn["floor"] == 5
        assert turn["win_before"] == pytest.approx(0.95)
        assert turn["win_after"] == pytest.approx(0.95)
        assert turn["peeked"] is False
        assert _turns(logged) == [turn]

    def test_turn_record_says_why_the_line_is_worse(self) -> None:
        """GH #888: the player's line wakes Lagavulin, the AI's does not --
        the record says so, or a correct call reads as noise."""
        g, sent, _logged, engine = _grader()
        start = _board(_lagavulin(asleep=True))
        engine.ai_end = _board(_lagavulin(asleep=True))
        pos = _Pos(hand=list(HAND), board=start)
        _combat_entry(g, "a1", 1, [_row(0, 0.95, 50, best=True), _row(1, 0.94, 45)], pos)
        g.on_human_action(_play("BASH", "a1", 1), node_id=10)
        end_pos = _Pos(hand=HAND[1:], board=_board(_lagavulin(asleep=False)))
        end_rows = [_row(0, 0.95, 50, best=True), _row(-1, 0.94, 38, visits=200)]
        _combat_entry(g, "a3", 3, end_rows, end_pos)
        g.on_human_action(_end("a3", 3), node_id=12)
        (turn,) = _turns(sent)
        assert turn["verdict"] == "blunder"
        assert turn["why"] == "Your line wakes Lagavulin Matriarch on turn 3"

    def test_no_why_when_the_ai_line_stops_short(self) -> None:
        g, sent, _logged, _engine = _grader()  # ai_end stays None
        _on_line_turn(
            g,
            end_rows=[_row(0, 0.95, 50, best=True), _row(-1, 0.94, 38, visits=200)],
        )
        (turn,) = _turns(sent)
        assert turn["why"] is None

    def test_on_line_turn_is_silent_by_default(self) -> None:
        g, sent, _logged, _ = _grader()
        _on_line_turn(g)
        assert not [t for t in _turns(sent) if t["verdict"] in ("slip", "blunder")]
        (turn,) = _turns(sent)
        # Graded, so the log (and a mod with "show good turns" on) can use it,
        # but nothing worth a chip -- and no AI line was searched for it.
        assert turn["verdict"] == "best"
        assert turn["ai"] == []

    def test_uncertain_action_silences_the_turn(self) -> None:
        g, sent, logged, engine = _grader()
        engine.top_up_rows = [{"p_win": 0.5, "q": 0.5, "visits": 30, "wins": 15, "hp_n": 30}]
        _on_line_turn(
            g,
            end_rows=[_row(0, 0.95, 50, best=True), _row(-1, 0.6, 20, visits=8)],
        )
        assert engine.top_up_calls, "a thin played row is topped up first"
        assert _turns(sent) == []
        (ungraded,) = _turns(logged)
        assert ungraded["verdict"] == "ungraded"
        assert "uncertain" in ungraded["reason"]

    def test_top_up_overwrites_the_played_row(self) -> None:
        g, sent, _, engine = _grader()
        engine.top_up_rows = [
            {"p_win": 0.85, "q": 0.8, "visits": 400, "wins": 340, "mean_final_hp": 30, "hp_n": 400}
        ]
        _on_line_turn(
            g,
            end_rows=[_row(0, 0.95, 50, best=True), _row(-1, 0.99, 60, visits=8)],
        )
        (turn,) = _turns(sent)
        assert turn["verdict"] == "blunder"
        assert turn["hp_drop"] == 20

    def test_stale_advice_is_not_used(self) -> None:
        g, sent, _, engine = _grader()
        pos = _Pos(hand=list(HAND))
        # Cached advice says Bash is a 30-HP blunder -- but it was for state 1,
        # and the play happened at position 2 (a fast player).
        _combat_entry(g, "a1", 1, [_row(1, 0.95, 50, best=True), _row(0, 0.6, 20)], pos)
        pre = _Pos(hand=list(HAND))
        engine.catch_up_rows = [
            [_row(0, 0.95, 50, best=True), _row(1, 0.9, 45)],
            [_row(-1, 0.95, 50, best=True), _row(0, 0.9, 45)],
        ]
        g.on_human_action(_play("BASH", "a1", 2), pre_state=pre, node_id=20)
        assert engine.catch_up_calls == [], "catch-up waits for the enemy turn"
        g.on_human_action(_end(None, 3), pre_state=_Pos(hand=HAND[1:]), node_id=21)
        assert engine.catch_up_calls[0] is pre
        (turn,) = _turns(sent)
        assert turn["verdict"] == "best"

    def test_a_card_missing_from_its_position_silences_the_turn(self) -> None:
        g, sent, logged, _ = _grader()
        pos = _Pos(hand=list(HAND))
        _combat_entry(g, "a1", 1, [_row(0, 0.95, 50, best=True), _row(1, 0.9, 45)], pos)
        g.on_human_action(_play("WHIRLWIND", "a1", 1), node_id=30)
        _combat_entry(g, "a2", 2, [_row(-1, 0.95, 50, best=True), _row(0, 0.9, 45)], pos)
        g.on_human_action(_end("a2", 2), node_id=31)
        assert _turns(sent) == []
        assert _turns(logged)[0]["verdict"] == "ungraded"

    def test_peeked_flag_propagates(self) -> None:
        g, sent, _, _ = _grader()
        _on_line_turn(g, peeked=True)
        (turn,) = _turns(sent)
        assert turn["peeked"] is True

    def test_an_in_combat_card_select_is_counted_not_graded(self) -> None:
        g, sent, _, _ = _grader()
        pos = _Pos(hand=list(HAND))
        _combat_entry(g, "a1", 1, [_row(0, 0.95, 50, best=True), _row(1, 0.9, 45)], pos)
        g.on_human_action(_play("BASH", "a1", 1), node_id=40)
        g.on_human_action(
            {"decision_type": "card_select", "action": "choose", "choice": {"index": 0}},
            node_id=41,
        )
        _combat_entry(g, "a2", 2, [_row(-1, 0.95, 50, best=True), _row(1, 0.9, 45)], pos)
        g.on_human_action(_end("a2", 2), node_id=42)
        (turn,) = _turns(sent)
        assert turn["verdict"] == "best"
        assert turn["ungraded_actions"] == 1


def _screen_entry(g: rc.CoachGrader, rid: str, sid: int, options: list[dict], **kw: Any) -> None:
    entry = g.cache.put(rid, sid, kind="screen", options=options, **kw)
    g.on_advice(entry)


REWARD_OPTIONS = [
    {"label": "Inflame", "visit_share": 0.72, "is_best": True, "index": 0, "action": "choose"},
    {
        "label": "Pommel Strike",
        "visit_share": 0.14,
        "is_best": False,
        "index": 1,
        "action": "choose",
    },
    {"label": "skip", "visit_share": 0.09, "is_best": False, "index": 3, "action": "skip"},
    {"label": "Clash", "visit_share": 0.05, "is_best": False, "index": 2, "action": "choose"},
]


def _choice(decision: str, index: int | None, rid: str, pos_id: int, **extra: Any) -> dict:
    return {
        "type": "human_action",
        "decision_type": decision,
        "action": "skip" if index is None else "choose",
        "choice": {"index": index, "label": "Skip" if index is None else "?"},
        "advice_request_id": rid,
        "position_id": pos_id,
        **extra,
    }


class TestChoices:
    def test_choice_flag_record_has_agreement_wording(self) -> None:
        g, sent, _, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=50)
        (rec,) = sent
        assert rec["kind"] == "choice" and rec["id"] == "choice:50"
        assert rec["verdict"] == "flag"
        assert rec["strength"] == "strong preference"
        assert rec["label_kind"] == "Card reward"
        assert rec["played"] == {"label": "skip", "prob": 0.09, "rank": 3}
        assert rec["top"] == {"label": "Inflame", "prob": 0.72}
        assert [a["label"] for a in rec["alternatives"]] == ["Inflame", "Pommel Strike", "skip"]
        flat = repr(rec).lower()
        assert "win" not in flat and "cost" not in flat

    def test_taking_the_top_pick_agrees(self) -> None:
        g, sent, _, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", 0, "r1", 5), node_id=51)
        assert sent[0]["verdict"] == "agree"

    def test_a_choice_against_stale_advice_is_not_graded(self) -> None:
        g, sent, logged, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 6), node_id=52)
        assert sent == [] and logged == []

    def test_peeked_flag_propagates(self) -> None:
        g, sent, _, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5, peeked=True), node_id=53)
        assert sent[0]["peeked"] is True

    def test_a_map_choice_carries_both_routes(self) -> None:
        you_path = [{"col": 1, "row": 2, "hp": 40}]
        ai_path = [{"col": 3, "row": 2, "hp": 55}]
        g, sent, _, _ = _grader(map_path=lambda msg, i: you_path if i == 1 else None)
        options = [
            {"label": "(3,2)", "visit_share": 0.8, "is_best": True, "index": 0, "action": "choose"},
            {
                "label": "(1,2)",
                "visit_share": 0.2,
                "is_best": False,
                "index": 1,
                "action": "choose",
            },
        ]
        _screen_entry(g, "m1", 9, options, advice={"path": ai_path})
        g.on_human_action(_choice("map_select", 1, "m1", 9), node_id=54)
        assert sent[0]["path"] == {"ai": ai_path, "you": you_path}
        assert sent[0]["label_kind"] == "Map"


class TestDelivery:
    def test_no_send_to_a_mod_without_the_capability(self) -> None:
        g, sent, logged, _ = _grader()
        g.on_advice_request({})  # an older mod: no capabilities
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=60)
        assert sent == []
        assert logged and logged[0]["verdict"] == "flag"

    def test_turn_and_choice_records_carry_log_position(self) -> None:
        where = {10: 7, 50: 12}
        g, sent, _, _ = _grader(locate=lambda node: ("run.jsonl", where.get(node)))
        _on_line_turn(g, end_rows=[_row(0, 0.95, 50, best=True), _row(-1, 0.94, 38)])
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=50)
        turn, choice = sent
        assert (turn["log_name"], turn["log_idx"]) == ("run.jsonl", 7)
        assert (choice["log_name"], choice["log_idx"]) == ("run.jsonl", 12)

    def test_records_without_a_locator_have_null_positions(self) -> None:
        g, sent, _, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=50)
        assert sent[0]["log_name"] is None and sent[0]["log_idx"] is None


@pytest.mark.parametrize(
    ("rows", "native", "needs"),
    [
        (
            [_row(0, 0.9, 50, best=True), _row(1, 0.8, 40, visits=10)],
            {"action": "play", "card_index": 1},
            True,
        ),
        (
            [_row(0, 0.9, 50, best=True), _row(1, 0.8, 40, visits=60)],
            {"action": "play", "card_index": 1},
            False,
        ),
        ([_row(0, 0.9, 50, best=True, visits=5)], {"action": "play", "card_index": 0}, False),
        ([_row(0, 0.9, 50, best=True)], {"action": "end_turn"}, False),
    ],
)
def test_played_needs_top_up(rows: list[dict], native: dict, needs: bool) -> None:
    assert rc.played_needs_top_up(rows, native) is needs


class _SlowEngine(_Engine):
    """Catch-up searches that take ``delay`` seconds and rank the move best."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def catch_up(self, cs: Any, cfg: Any, sims: int) -> list[dict[str, Any]]:
        time.sleep(self.delay)
        self.catch_up_calls.append(cs)
        return [_row(-1, 0.95, 50, best=True), _row(0, 0.9, 45)]


def _threaded(delay: float, budget_s: float = 5.0) -> tuple[rc.CoachGrader, list, list]:
    sent: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    g = rc.CoachGrader(
        send=sent.append,
        log=logged.append,
        engine=_SlowEngine(delay),
        executor=rc.GraderThread(),
        budget_s=budget_s,
    )
    g.on_advice_request({"capabilities": {"coach": 1}, "floor": 5})
    return g, sent, logged


def _skipped_end_turn(g: rc.CoachGrader, node_id: int = 70) -> None:
    """An end turn the advisor never searched: graded by catch-up only."""
    g.on_human_action(_end(None, 9), pre_state=_Pos(hand=list(HAND)), node_id=node_id)


class TestGraderThread:
    def test_catchup_completes_turn_within_budget(self) -> None:
        g, sent, _ = _threaded(delay=0.1)
        _skipped_end_turn(g)
        assert _turns(sent) == [], "the catch-up runs in the background"
        assert g.executor.wait_idle(5.0)
        (turn,) = _turns(sent)
        assert turn["verdict"] == "best"
        g.executor.stop()

    def test_preempt_drops_turn_silently(self) -> None:
        g, sent, logged = _threaded(delay=0.5)
        _skipped_end_turn(g)
        # The next player turn asks for advice before the catch-up finished.
        g.on_advice_request(
            {"advice_for": "combat_action", "round": 4, "capabilities": {"coach": 1}}
        )
        assert g.executor.wait_idle(5.0)
        assert _turns(sent) == []
        (ungraded,) = _turns(logged)
        assert ungraded["verdict"] == "ungraded"
        assert "preempted" in ungraded["reason"]
        g.executor.stop()

    def test_a_request_for_the_same_round_does_not_preempt(self) -> None:
        g, sent, _ = _threaded(delay=0.2)
        _skipped_end_turn(g)
        g.on_advice_request({"advice_for": "combat_action", "round": 3})
        g.on_advice_request({"advice_for": "map_select", "capabilities": {"coach": 1}})
        assert g.executor.wait_idle(5.0)
        assert [t["verdict"] for t in _turns(sent)] == ["best"]
        g.executor.stop()

    def test_deadline_drops_turn(self) -> None:
        g, sent, logged = _threaded(delay=0.3, budget_s=0.05)
        _skipped_end_turn(g)
        assert g.executor.wait_idle(5.0)
        assert _turns(sent) == []
        (ungraded,) = _turns(logged)
        assert "budget" in ungraded["reason"]
        g.executor.stop()

    def test_rewind_withdraws_records_off_the_active_path(self) -> None:
        g, sent, logged, _ = _grader()
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=50)
        _on_line_turn(g)  # turn record keyed by node 10
        out = g.on_rewind({"type": "combat_undone", "turn": 3}, active_path=[0, 50])
        assert out == [{"kind": "withdraw", "ids": ["turn:10"]}]
        assert sent[-1] == out[0] and logged[-1] == out[0]
        # Withdrawn once: a second rewind has nothing left to take back.
        assert g.on_rewind({"type": "combat_undone"}, active_path=[0, 50]) == []

    def test_rewind_drops_the_open_turn(self) -> None:
        g, sent, logged, _ = _grader()
        pos = _Pos(hand=list(HAND))
        _combat_entry(g, "a1", 1, [_row(0, 0.95, 50, best=True), _row(1, 0.9, 45)], pos)
        g.on_human_action(_play("BASH", "a1", 1), node_id=80)
        g.on_rewind({"type": "combat_undone", "turn": 3}, active_path=[0])
        _combat_entry(g, "a2", 2, [_row(-1, 0.95, 50, best=True), _row(0, 0.9, 45)], pos)
        g.on_human_action(_end("a2", 2), node_id=81)
        (turn,) = _turns(sent)
        assert turn["you"] == ["end turn"], "nothing from the abandoned branch"
        assert any(t.get("reason", "").startswith("undone") for t in _turns(logged))


# ---------------------------------------------------------------------------
# Fight recap
# ---------------------------------------------------------------------------


def _fight_msg(rnd: int, hp: int = 80) -> dict[str, Any]:
    """A combat ``advice_request`` in the mod's wire dialect."""
    return {
        "round": rnd,
        "floor": 5,
        "room_type": "Monster",
        "player": {"hp": hp, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3},
        "hand": [{"id": "BASH", "name": "Bash"}, {"id": "STRIKE_IRONCLAD", "name": "Strike+"}],
        "enemies": [
            {
                "name": "Jaw Worm",
                "hp": 40,
                "is_alive": True,
                "intents": [{"intent_type": "Attack", "damage": 11, "hits": 1}],
            },
            {"name": "Louse", "hp": 0, "is_alive": False, "intents": []},
        ],
    }


def _grader2(**kw: Any) -> tuple[rc.CoachGrader, list[dict], list[dict], _Engine]:
    g, sent, logged, engine = _grader(**kw)
    g.on_advice_request({"capabilities": {"coach": 2}, "floor": 5})
    return g, sent, logged, engine


def _fight_turn(
    g: rc.CoachGrader, rnd: int, drop: float, node: int, *, end_visits: int = 400
) -> None:
    """One turn of round ``rnd``: an end turn ``drop`` HP worse than the best."""
    rows = [_row(0, 0.95, 50, best=True), _row(-1, 0.95, 50 - drop, visits=end_visits)]
    entry = g.cache.put(
        f"t{rnd}", rnd * 10, kind="combat", rows=rows, cs=_Pos(round=rnd), msg=_fight_msg(rnd)
    )
    g.on_advice(entry)
    g.on_human_action(_end(f"t{rnd}", rnd * 10), node_id=node)


def _recaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "recap"]


WON = {"type": "combat_ended", "victory": True, "hp": 76, "round": 3, "room_type": "Monster"}


class TestHelpers:
    def test_intent_text_attack_multi_and_buff(self) -> None:
        intents = [
            {"intent_type": "Attack", "damage": 6, "hits": 2},
            {"intent_type": "Debuff"},
        ]
        assert rc.intent_text(intents) == "attack 6×2 + debuff"
        assert rc.intent_text([{"intent_type": "Attack", "damage": 11, "hits": 1}]) == "attack 11"
        assert rc.intent_text([{"intent_type": "Buff"}, {"intent_type": "Buff"}]) == "buff"
        assert rc.intent_text([]) == "unknown"

    def test_turn_context_alive_enemies_only(self) -> None:
        ctx = rc.turn_context(_fight_msg(2, hp=61))
        assert ctx == {
            "hand": ["Bash", "Strike+"],
            "energy": 3,
            "max_energy": 3,
            "hp": 61,
            "max_hp": 80,
            "block": 0,
            "enemies": [{"name": "Jaw Worm", "hp": 40, "intent": "attack 11"}],
        }
        assert rc.turn_context(None) is None

    def test_turn_context_numbers_same_named_enemies(self) -> None:
        msg = _fight_msg(2)
        rat = {"name": "Two-Tailed Rat", "hp": 20, "is_alive": True, "intents": []}
        msg["enemies"] = [rat, {**rat, "is_alive": False}, rat]
        names = [e["name"] for e in rc.turn_context(msg)["enemies"]]
        # The dying rat takes no number (as in the option labels and the
        # HUD tags), so the live rats are #1 and #2, as on the workbench.
        assert names == ["Two-Tailed Rat #1", "Two-Tailed Rat #2"]


class TestRecap:
    def test_clean_fight_recap_has_no_costliest(self) -> None:
        g, sent, _, _ = _grader2(locate=lambda node: ("run.jsonl", 3 if node == 10 else None))
        _fight_turn(g, 1, 0, 10)
        _fight_turn(g, 2, 0, 11)
        g.on_combat_ended(WON)
        (recap,) = _recaps(sent)
        assert recap["id"] == "recap:10" and recap["node_id"] == 10
        assert (recap["log_name"], recap["log_idx"]) == ("run.jsonl", 3)
        assert recap["hp_start"] == 80 and recap["hp_end"] == 76 and recap["hp_lost"] == 4
        # Reserved: the opening search's mean final HP is not the AI's own
        # play (it overstated the AI's loss live), so it is never reported.
        assert recap["ai_hp_lost"] is None
        assert recap["costliest"] == []
        assert (recap["turns"], recap["turns_graded"], recap["turns_ungraded"]) == (2, 2, 0)
        assert recap["unfinished_turns"] == 0
        assert recap["floor"] == 5 and recap["room_type"] == "Monster"

    def test_recap_lists_two_costliest_by_severity(self) -> None:
        g, sent, _, _ = _grader2()
        _fight_turn(g, 1, 4, 10)
        _fight_turn(g, 2, 12, 11)
        _fight_turn(g, 3, 9, 12)
        g.on_combat_ended(WON)
        (recap,) = _recaps(sent)
        assert [t["severity"] for t in recap["costliest"]] == [12, 9]
        assert [t["turn"] for t in recap["costliest"]] == [2, 3]
        assert all(t["context"]["hand"] == ["Bash", "Strike+"] for t in recap["costliest"])
        assert recap["costliest"][0]["context"]["enemies"][0]["intent"] == "attack 11"

    def test_recap_ai_line_stays_null_even_with_hp_samples(self) -> None:
        g, sent, _, _ = _grader2()
        rows = [_row(0, 0.95, 50, best=True), _row(-1, 0.95, 50)]
        rows[0]["hp_n"] = 5000
        entry = g.cache.put("t1", 10, kind="combat", rows=rows, cs=_Pos(round=1), msg=_fight_msg(1))
        g.on_advice(entry)
        g.on_human_action(_end("t1", 10), node_id=10)
        g.on_combat_ended(WON)
        (recap,) = _recaps(sent)
        assert recap["ai_hp_lost"] is None and recap["hp_lost"] == 4

    def test_killing_blow_turn_is_unfinished_not_ungraded(self) -> None:
        g, sent, _, _ = _grader2()
        _fight_turn(g, 1, 0, 10)
        rows = [_row(0, 0.95, 50, best=True), _row(1, 0.9, 45)]
        entry = g.cache.put(
            "t2", 20, kind="combat", rows=rows, cs=_Pos(hand=list(HAND), round=2),
            msg=_fight_msg(2),
        )  # fmt: skip
        g.on_advice(entry)
        g.on_human_action(_play("BASH", "t2", 20), node_id=11)  # the killing blow
        g.on_combat_ended(WON)
        (recap,) = _recaps(sent)
        assert recap["unfinished_turns"] == 1
        assert recap["turns_ungraded"] == 0
        assert recap["turns"] == 1

    def test_defeat_sends_no_recap(self) -> None:
        g, sent, logged, _ = _grader2()
        _fight_turn(g, 1, 12, 10)
        g.on_combat_ended({**WON, "victory": False, "hp": 0})
        assert _recaps(sent) == [] and _recaps(logged) == []

    def test_no_recap_to_a_coach_1_mod(self) -> None:
        g, sent, logged, _ = _grader()  # the mod advertised coach 1
        _fight_turn(g, 1, 0, 10)
        g.on_combat_ended(WON)
        assert _recaps(sent) == []
        assert len(_recaps(logged)) == 1, "the run log still gets the recap"

    def test_recap_withdrawn_on_restore_before_fight(self) -> None:
        g, _, _, _ = _grader2()
        _fight_turn(g, 1, 0, 10)
        g.on_combat_ended(WON)
        out = g.on_rewind({"type": "snapshot_restored"}, active_path=[0, 5])
        assert out and "recap:10" in out[0]["ids"] and "turn:10" in out[0]["ids"]

    def test_turn_without_position_never_costliest(self) -> None:
        g, sent, _, engine = _grader2()
        _fight_turn(g, 1, 0, 10)
        # Round 2 was never advised: graded from the mirror's position only.
        engine.catch_up_rows = [[_row(0, 0.95, 50, best=True), _row(-1, 0.95, 38)]]
        g.on_human_action(_end(None, 99), pre_state=_Pos(round=2), node_id=11)
        (slip,) = [t for t in _turns(sent) if t["turn"] == 2]
        assert slip["verdict"] == "blunder" and slip["context"] is None
        g.on_combat_ended(WON)
        (recap,) = _recaps(sent)
        assert recap["costliest"] == [] and recap["turns_graded"] == 2

    def test_a_new_fight_starts_after_a_run_level_screen(self) -> None:
        g, sent, _, _ = _grader2()
        _fight_turn(g, 1, 12, 10)
        g.on_combat_ended(WON)
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)  # the card reward
        _fight_turn(g, 1, 0, 20)
        g.on_combat_ended(WON)
        first, second = _recaps(sent)
        assert first["id"] == "recap:10" and len(first["costliest"]) == 1
        assert second["id"] == "recap:20" and second["costliest"] == []


class _TopUpEngine(_Engine):
    """A played-row top-up that takes ``delay`` seconds and finds 12 HP lost."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def top_up(self, cs: Any, native: dict[str, Any], cfg: Any, sims: int) -> Any:
        time.sleep(self.delay)
        return {"p_win": 0.95, "q": 0.95, "visits": 400, "wins": 380,
                "mean_final_hp": 38, "hp_n": 400}  # fmt: skip


def _threaded_recap(delay: float, budget_s: float) -> tuple[rc.CoachGrader, list, list]:
    sent: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    g = rc.CoachGrader(
        send=sent.append,
        log=logged.append,
        engine=_TopUpEngine(delay),
        executor=rc.GraderThread(),
        budget_s=budget_s,
    )
    g.on_advice_request({"capabilities": {"coach": 2}, "floor": 5})
    return g, sent, logged


class TestRecapTiming:
    def test_recap_waits_for_turn_still_grading(self) -> None:
        g, sent, _ = _threaded_recap(delay=0.5, budget_s=5.0)
        _fight_turn(g, 1, 0, 10, end_visits=8)  # thin played row -> background top-up
        g.on_combat_ended(WON)
        assert g.executor.wait_idle(5.0)
        (recap,) = _recaps(sent)
        (turn,) = recap["costliest"]
        assert turn["verdict"] == "blunder" and turn["ai"] == ["Bash", "Defend", "end turn"]
        assert recap["turns_ungraded"] == 0
        g.executor.stop()

    def test_recap_counts_overdue_turn_as_ungraded(self) -> None:
        g, sent, _ = _threaded_recap(delay=0.5, budget_s=0.1)
        _fight_turn(g, 1, 0, 10, end_visits=8)
        g.on_combat_ended(WON)
        assert g.executor.wait_idle(5.0)
        (recap,) = _recaps(sent)
        assert recap["costliest"] == [] and recap["turns_ungraded"] == 1
        g.executor.stop()


# ---------------------------------------------------------------------------
# Death autopsy
# ---------------------------------------------------------------------------


def _t(
    severity: float, hp: float | None = None, *, ctx: bool = True, verdict: str = "slip"
) -> dict:
    return {
        "kind": "turn",
        "id": f"turn:{severity}",
        "verdict": verdict,
        "severity": severity,
        "hp_drop": hp,
        "context": {"hand": ["Bash"]} if ctx else None,
    }


def _c(top: float, played: float, *, verdict: str = "flag") -> dict:
    return {
        "kind": "choice",
        "id": f"choice:{top}-{played}",
        "verdict": verdict,
        "top": {"label": "Inflame", "prob": top},
        "played": {"label": "Skip", "prob": played, "rank": 3},
    }


DIED = {"type": "game_over", "result": "defeat", "seed": "ABC", "floor": 9, "act": 1}


def _autopsies(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == "autopsy"]


class TestRankAutopsy:
    def test_rank_autopsy_interleaves_kinds(self) -> None:
        turns = [_t(4, 4), _t(12, 12), _t(9, None)]
        choices = [_c(0.5, 0.2), _c(0.9, 0.05)]
        out = rc.rank_autopsy(turns, choices)
        assert [i["kind"] for i in out] == ["turn", "choice", "turn"]
        assert out[0]["record"]["severity"] == 12
        assert out[1]["record"]["id"] == "choice:0.9-0.05"
        assert out[2]["record"]["severity"] == 9, "no HP figure: ranked by severity"

    def test_rank_autopsy_fills_from_other_kind(self) -> None:
        out = rc.rank_autopsy([_t(3), _t(5), _t(8), _t(4)], [])
        assert [i["record"]["severity"] for i in out] == [8, 5, 4]
        only_choices = rc.rank_autopsy([], [_c(0.5, 0.2), _c(0.6, 0.1)])
        assert [i["kind"] for i in only_choices] == ["choice", "choice"]

    def test_rank_autopsy_skips_contextless_and_good_records(self) -> None:
        turns = [_t(20, 20, ctx=False), _t(2, verdict="tie"), _t(5)]
        choices = [_c(0.5, 0.4, verdict="agree")]
        out = rc.rank_autopsy(turns, choices)
        assert [i["record"]["severity"] for i in out] == [5]


class TestAutopsy:
    def test_autopsy_ranks_the_runs_decisions(self) -> None:
        g, sent, _, _ = _grader2(locate=lambda node: ("run.jsonl", None))
        _fight_turn(g, 1, 12, 10)
        g.on_combat_ended(WON)
        _screen_entry(g, "r1", 5, REWARD_OPTIONS)
        g.on_human_action(_choice("card_reward", None, "r1", 5), node_id=50)
        _fight_turn(g, 1, 4, 60)
        g.on_game_over(DIED)
        (autopsy,) = _autopsies(sent)
        assert autopsy["id"] == "autopsy:ABC" and autopsy["log_name"] == "run.jsonl"
        assert (autopsy["floor"], autopsy["act"]) == (9, 1)
        assert [i["kind"] for i in autopsy["items"]] == ["turn", "choice", "turn"]
        assert autopsy["items"][0]["record"]["id"] == "turn:10"
        assert autopsy["decisions_graded"] == 3 and autopsy["turns_ungraded"] == 0

    def test_autopsy_excludes_withdrawn_records(self) -> None:
        g, sent, _, _ = _grader2()
        _fight_turn(g, 1, 12, 10)
        _fight_turn(g, 2, 9, 11)
        g.on_rewind({"type": "combat_undone", "turn": 2}, active_path=[0, 10])
        g.on_game_over(DIED)
        (autopsy,) = _autopsies(sent)
        assert [i["record"]["id"] for i in autopsy["items"]] == ["turn:10"]

    def test_empty_autopsy_still_goes_out(self) -> None:
        g, sent, _, _ = _grader2()
        _fight_turn(g, 1, 0, 10)
        g.on_game_over(DIED)
        (autopsy,) = _autopsies(sent)
        assert autopsy["items"] == [] and autopsy["decisions_graded"] == 1

    def test_reset_runs_after_autopsy(self) -> None:
        g, sent, _ = _threaded_recap(delay=0.3, budget_s=5.0)
        _fight_turn(g, 1, 0, 10, end_visits=8)  # graded in the background
        g.on_game_over(DIED)
        assert g._emitted_records or g._grading, "nothing is reset before the job runs"
        assert g.executor.wait_idle(5.0)
        (autopsy,) = _autopsies(sent)
        assert [i["record"]["id"] for i in autopsy["items"]] == ["turn:10"]
        assert g._emitted_records == {} and g._emitted == {} and len(g.cache) == 0
        g.executor.stop()

    def test_autopsy_waits_for_dying_turn_grade(self) -> None:
        g, sent, _ = _threaded_recap(delay=0.5, budget_s=5.0)
        _fight_turn(g, 1, 0, 10, end_visits=8)
        g.on_game_over(DIED)  # while the top-up is still running
        assert g.executor.wait_idle(5.0)
        (autopsy,) = _autopsies(sent)
        (item,) = autopsy["items"]
        assert item["record"]["verdict"] == "blunder"
        g.executor.stop()

    def test_no_autopsy_to_a_coach_1_mod(self) -> None:
        g, sent, logged, _ = _grader()
        _fight_turn(g, 1, 12, 10)
        g.on_game_over(DIED)
        assert _autopsies(sent) == [] and len(_autopsies(logged)) == 1

    def test_victory_sends_no_autopsy(self) -> None:
        import realgame_advisor as ra

        assert ra.is_death(DIED)
        assert not ra.is_death({**DIED, "result": "victory"})
        assert not ra.is_death({"type": "run_complete", "result": "defeat"})
        g, sent, logged, _ = _grader2()
        _fight_turn(g, 1, 12, 10)
        g.on_run_end()
        assert _autopsies(sent) == [] and _autopsies(logged) == []


def test_main_loop_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import realgame_advisor as ra

    def _fake(state: dict[str, Any], cfg: Any, sims: int, mirror: Any) -> tuple:
        return {"note": "searched", "options": []}, object(), cfg, []

    monkeypatch.setattr(ra, "compute_combat_advice", _fake)
    g, _, _ = _threaded(delay=0.0)
    g.executor.submit(lambda: time.sleep(2.0))  # a long catch-up in flight
    sent: list[dict[str, Any]] = []
    client = SimpleNamespace(send_action=sent.append)
    wlog = SimpleNamespace(log_decision=lambda *a: None, log_advice=lambda *a, **k: None)
    request = {
        "type": "advice_request",
        "advice_for": "combat_action",
        "request_id": "a9",
        "state_id": 9,
        "round": 2,
    }
    t0 = time.monotonic()
    ra._handle_advice_request(
        client,
        ra.RecommendConfig(),
        SimpleNamespace(sims=10, boss_sims=10, log_search_tree=0),
        SimpleNamespace(write=lambda *a: None),
        wlog,
        request,
        None,
        session={},
        coach=g,
    )
    assert time.monotonic() - t0 < 1.0
    assert sent and sent[0]["action"] == "advice"
    g.executor.stop()


def test_advice_cache_is_a_bounded_lru() -> None:
    cache = rc.AdviceCache(size=2)
    cache.put("a", 1, kind="combat")
    cache.put("b", 2, kind="combat")
    assert cache.get("a") is not None  # touch: "b" is now the oldest
    cache.put("c", 3, kind="combat")
    assert cache.get("b") is None
    assert cache.get("a") is not None and cache.get("c") is not None
    assert cache.put(None, 4, kind="combat") is None
    cache.clear()
    assert len(cache) == 0


class TestRealEngine:
    """:class:`realgame_coach.CoachEngine` against the real ``sts2_rs`` -- the
    stubs above pin the grading; this pins the engine calls they stand for."""

    @pytest.fixture
    def cs(self) -> Any:
        sts2_rs = pytest.importorskip("sts2_rs")
        return sts2_rs.build_scenario_state(
            3,
            "Ironclad",
            ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 5,
            80,
            80,
            [],
            "SHRINKER_BEETLE_WEAK",
        )

    def test_position_info_and_searches(self, cs: Any) -> None:
        from sts2_hint.recommender import RecommendConfig

        engine = rc.CoachEngine()
        info = engine.position_info(cs)
        assert len(info["hand"]) == 5 and isinstance(info["round"], int)
        card_id = info["hand"][0][0]
        native = rc.native_action(
            {"action": "play", "card": {"id": card_id, "upgraded": False}}, info
        )
        assert native is not None and native["card_index"] == 0

        cfg = RecommendConfig(threads=1)
        rows = engine.catch_up(cs, cfg, 200)
        assert any(r["is_best"] for r in rows)
        fresh = engine.top_up(cs, {"action": "end_turn"}, cfg, 100)
        assert fresh is not None and fresh["visits"] >= 90

        g = rc.CoachGrader(send=lambda r: None, log=lambda r: None, engine=engine)
        line = engine.ai_line(cs, None, cfg, g._label_engine)
        assert 1 <= len(line) <= rc.AI_LINE_MAX_STEPS
        assert all(isinstance(s, str) and s for s in line)
        line2, end = engine.ai_turn(cs, None, cfg, g._label_engine)
        if line2[-1] == "end turn":
            assert end is not None and end.get("encounter"), end
            assert rc.board_events(engine.board(cs), end) is not None


def test_human_label_uses_card_names_not_raw_detail() -> None:
    """The mod's ``detail`` can be raw history text; the label is rebuilt
    from ``card`` / ``target_index``, with the target in parentheses (an arrow
    would read as a second move in a turn line)."""
    g, _sent, _logged, _engine = _grader()
    g.on_advice(
        g.cache.put(
            "a0",
            1,
            kind="combat",
            rows=[],
            cs=None,
            msg={
                "round": 1,
                "hand": [{"id": "STRIKE_IRONCLAD", "name": "Strike"}],
                "enemies": [{"name": "Toadpole"}, {"name": "Cultist"}],
            },
        )
    )
    raw = _play("STRIKE_IRONCLAD", None, 1, target_index=1)
    raw["detail"] = "IRONCLAD played STRIKE_IRONCLAD targeting CULTIST"
    assert g._label_human(raw) == "Strike (Cultist)"
    up = _play("STRIKE_IRONCLAD", None, 1, target_index=0)
    up["card"]["upgraded"] = True
    assert g._label_human(up) == "Strike+ (Toadpole)"
    assert g._label_human({"potion": {"id": "FIRE_POTION"}}) == "drink Fire Potion"


def test_label_human_numbers_same_named_targets() -> None:
    g, _sent, _logged, _engine = _grader()
    g.on_advice(
        g.cache.put(
            "a0",
            1,
            kind="combat",
            rows=[],
            cs=None,
            msg={
                "round": 1,
                "hand": [{"id": "STRIKE_IRONCLAD", "name": "Strike"}],
                "enemies": [{"name": "Louse"}, {"name": "Cultist"}, {"name": "Louse"}],
            },
        )
    )
    assert g._label_human(_play("STRIKE_IRONCLAD", None, 1, target_index=2)) == "Strike (Louse #2)"
    assert g._label_human(_play("STRIKE_IRONCLAD", None, 1, target_index=1)) == "Strike (Cultist)"


_INTENT_CASES = Path(__file__).resolve().parent / "fixtures" / "coach" / "intent_text_cases.json"


@pytest.mark.parametrize(
    "case",
    json.loads(_INTENT_CASES.read_text(encoding="utf-8"))["cases"],
    ids=lambda c: c["name"],
)
def test_intent_text_shared_cases(case: dict[str, Any]) -> None:
    """Twin of advisors::intent_text (the browser workbench's turn context)."""
    intents = [
        {**{k: v for k, v in i.items() if k != "kind"}, "intent_type": i["kind"]}
        for i in case["intents"]
    ]
    assert rc.intent_text(intents) == case["expect"]


_RANK_CASES = Path(__file__).resolve().parent / "fixtures" / "coach" / "rank_autopsy_cases.json"


@pytest.mark.parametrize(
    "case",
    json.loads(_RANK_CASES.read_text(encoding="utf-8"))["cases"],
    ids=lambda c: c["name"],
)
def test_rank_autopsy_shared_cases(case: dict[str, Any]) -> None:
    """Twin of web/src/coach/coachRank.ts (the browser's live-play autopsy)."""
    items = rc.rank_autopsy(case["turns"], case["choices"], case["k"])
    assert [[i["kind"], i["record"]["id"]] for i in items] == case["expect"]
