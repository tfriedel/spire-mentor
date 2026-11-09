"""Instinct, the e2e student, as the live advisor (``realgame_instinct``).

Two things are pinned here:

* the ``sts2_rs`` binding plays the student exactly as the native env does
  (``e2e_selfplay`` vs a Python loop over ``E2eDriver`` / ``E2eStudent``);
* the advisor's RUN GRAFT is faithful: a native campaign run is played to a
  decision, the mod's envelope for that position is synthesized from it (its
  ``map`` / ``rng`` / ``run_state`` from the very same engine run, as
  ``workbench/test_run_import.py`` does), and Instinct answering that envelope
  must see what the native student saw there -- the same options and the same
  probabilities.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_advisor as ra  # noqa: E402
import realgame_instinct as ri  # noqa: E402
from floors import game_floor  # noqa: E402

SEED = "INSTINCTSEED7"
WEIGHTS = ri.default_weights_path()


@pytest.fixture(scope="module")
def student() -> Any:
    return sts2_rs.E2eStudent(str(WEIGHTS))


def _campaign_run(seed: str) -> Any:
    """The student's campaign run, at a STRING seed (as live runs have)."""
    return sts2_rs.RunManager(
        game_version=ri.INSTINCT_GAME_VERSION,
        string_seed=seed,
        character="Ironclad",
        ascension_level=10,
        neow_start=True,
        mode="agent",
        interactive_potion_swap=True,
        act1="roll",
    )


def _play_to(student: Any, rm: Any, driver: Any, stop: Any, limit: int = 2000) -> Any:
    """Play the student greedily until ``stop(peek)`` holds; return that peek."""
    for _ in range(limit):
        peek = driver.peek(rm)
        if peek.kind == "terminal":
            pytest.fail(f"the run ended before the position: {peek.terminal}")
        if peek.kind == "forced":
            f = peek.forced
            driver.apply(rm, f["action"], f["kind"], strict=False)
            continue
        if stop(peek):
            return peek
        out = student.decide(driver, peek, rm)
        ca = peek.class_action(out["class"])
        driver.apply(rm, ca["action"], ca["kind"])
    pytest.fail("position not reached")


def _wire_id(engine_name: str) -> str:
    """Engine Debug spelling (``StrikeIronclad``, ``RELIC.BURNING_BLOOD``) -> the mod's id."""
    tail = engine_name.split(".")[-1]
    if "_" in tail or tail.isupper():
        return tail.upper()
    return re.sub(r"(?<!^)(?=[A-Z])", "_", tail).upper()


def _envelope(rm: Any, seed: str, advice_for: str) -> dict[str, Any]:
    """The mod's ``advice_request`` envelope for ``rm``'s position."""
    full = rm.full_map()
    nodes, visited = [], []
    for row in full["rows"]:
        for n in row:
            nodes.append(
                {
                    "col": n["col"],
                    "row": n["row"],
                    "type": n["type"],
                    "children": n.get("children") or [],
                }
            )
            if n.get("visited"):
                visited.append({"col": n["col"], "row": n["row"]})
    cur = full.get("current_coord")
    slots = rm.max_potion_slots()
    held = [_wire_id(p) for p in rm.potions()]
    potions = [{"slot": i, "id": pid} for i, pid in enumerate(held)]
    potions += [{"slot": i, "empty": True} for i in range(len(held), slots)]
    act = rm.act_index
    return {
        "type": "advice_request",
        "advice_for": advice_for,
        "brain": "instinct",
        "game_version": ri.INSTINCT_GAME_VERSION,
        "seed": seed,
        "act": act,
        "floor": game_floor(rm.total_floor(), act),
        "map": {"nodes": nodes, "current": cur, "visited": visited},
        "rng": {"run_seed": rm.run_seed, "counters": dict(rm.rng_counters())},
        "run_state": {
            "character": "IRONCLAD",
            "ascension": rm.ascension_level,
            "hp": rm.current_hp,
            "max_hp": rm.max_hp,
            "gold": rm.gold,
            "potion_slot_count": slots,
            "master_deck": [
                {"id": _wire_id(name), "upgrade_level": int(up)}
                for name, up in rm.master_deck_detailed()
            ],
            "relics": [{"id": _wire_id(r)} for r in rm.relics()],
            "potions": potions,
        },
    }


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="no Instinct weights (data/e2e)")
class TestInstinct:
    def test_binding_plays_like_the_native_env(self, student: Any) -> None:
        """The Python driver loop over the binding == the native env loop."""
        n = 80
        native = sts2_rs.e2e_selfplay(student, 170000000, max_decisions=n)
        rm = sts2_rs.e2e_new_run(170000000)
        driver = sts2_rs.E2eDriver(170000000)
        chosen: list[int] = []
        while len(chosen) < n:
            peek = driver.peek(rm)
            assert peek.kind != "terminal"
            if peek.kind == "forced":
                f = peek.forced
                driver.apply(rm, f["action"], f["kind"], strict=False)
                continue
            out = student.decide(driver, peek, rm)
            chosen.append(out["class"])
            ca = peek.class_action(out["class"])
            driver.apply(rm, ca["action"], ca["kind"])
        assert chosen == native["chosen"]

    def test_map_advice_from_an_envelope_sees_what_the_native_student_sees(
        self, student: Any
    ) -> None:
        rm = _campaign_run(SEED)
        driver = sts2_rs.E2eDriver(rm.seed)
        maps = 0

        def third_map(peek: Any) -> bool:
            nonlocal maps
            if peek.decision["decision_type"] != "MAP":
                return False
            maps += 1
            return maps == 3

        peek = _play_to(student, rm, driver, third_map)
        native = student.decide(driver.copy(), peek, rm)
        cols = [c["actions"][0]["col"] for c in peek.decision["classes"]]

        msg = _envelope(rm, SEED, "map_select")
        children = {(c["col"], c["row"]) for c in _children_of(msg)}
        msg["nodes"] = [
            {"index": i, "col": col, "row": row, "type": "Monster"}
            for i, (col, row) in enumerate(sorted(children))
        ]
        advisor = ri.InstinctAdvisor(WEIGHTS)
        advice = advisor.advise(msg, "map_select", session={})

        assert advice["brain"] == "instinct"
        by_col = {o["col"]: o["prob"] for o in advice["options"]}
        assert sorted(by_col) == sorted(cols)
        for ci, col in enumerate(cols):
            assert by_col[col] == pytest.approx(native["probs"][ci], abs=2e-3)
        best = next(o for o in advice["options"] if o["is_best"])
        assert best["col"] == cols[native["class"]]

        # The route (PROTOCOL.md v5.3): the search's `path` schema, from the
        # chosen node, one row per hop along the map's edges, to the act's end.
        path = advice["path"]
        assert (path[0]["col"], path[0]["row"], path[0]["hop"]) == (best["col"], best["row"], 0)
        graph = {(n["col"], n["row"]): n for n in msg["map"]["nodes"]}
        for a, b in itertools.pairwise(path):
            assert b["hop"] == a["hop"] + 1
            assert {"col": b["col"], "row": b["row"]} in graph[a["col"], a["row"]]["children"]
        last = graph[path[-1]["col"], path[-1]["row"]]["children"]
        assert not [c for c in last if (c["col"], c["row"]) in graph], "walks to the act's end"
        assert all(set(s) == {"col", "row", "type", "hop", "hp"} for s in path)
        assert "route" in advice["note"]

    def test_a_fight_without_a_map_sees_what_the_native_student_sees(self, student: Any) -> None:
        """GH #1212: fight requests carry no map graph, and the graft left the
        run at act 1 / floor 0, so the student read every live fight as the
        run's first room. A native act-2 fight, grafted from an envelope with
        no ``map``, must stand where the native run stands and give the same
        probabilities."""
        rm = _campaign_run(SEED)
        driver = sts2_rs.E2eDriver(rm.seed)

        def act2_fight(peek: Any) -> bool:
            return rm.act_index >= 2 and peek.decision["decision_type"] == "COMBAT_TURN"

        peek = _play_to(student, rm, driver, act2_fight)
        native = student.decide(driver.copy(), peek, rm)
        room = str(rm.current_room()["room_type"]).lower()

        msg = _envelope(rm, SEED, "combat_action")
        del msg["map"]
        msg["room_type"] = room.capitalize()
        grafted, notes = ri.graft_run(msg)
        assert any(n.startswith("no map position") for n in notes)
        assert (grafted.act_index, grafted.total_floor(), grafted.act_floor()) == (
            rm.act_index,
            rm.total_floor(),
            rm.act_floor(),
        )
        assert grafted.full_map()["current_coord"]["row"] == rm.full_map()["current_coord"]["row"]

        grafted.debug_enter_combat(rm.combat_state(), room)
        probe = driver.copy()
        out = student.decide(probe, probe.peek(grafted), grafted)
        assert out["probs"] == pytest.approx(native["probs"], abs=2e-3)

    def test_other_characters_fall_back_to_the_search_with_a_reason(self) -> None:
        advisor = ri.InstinctAdvisor(WEIGHTS)
        msg = {"run_state": {"character": "SILENT"}, "game_version": "v0.111.0"}
        with pytest.raises(ri.InstinctUnavailable, match="Ironclad"):
            advisor.advise(msg, "map_select")
        msg = {"run_state": {"character": "IRONCLAD"}, "game_version": "v0.107.1"}
        with pytest.raises(ri.InstinctUnavailable, match=r"v0.111.0"):
            advisor.advise(msg, "map_select")

    def test_capabilities_name_what_instinct_plays(self) -> None:
        assert ri.InstinctAdvisor(WEIGHTS).capabilities() == [
            {"character": "Ironclad", "game_version": "v0.111.0"}
        ]


def _children_of(msg: dict[str, Any]) -> list[dict[str, Any]]:
    cur = msg["map"]["current"]
    for n in msg["map"]["nodes"]:
        if n["col"] == cur["col"] and n["row"] == cur["row"]:
            return n["children"]
    return []


def test_a_missing_net_is_a_reason_not_a_crash(tmp_path: Any) -> None:
    advisor = ri.InstinctAdvisor(tmp_path / "nope.e2ew")
    assert advisor.capabilities() == []
    msg = {"run_state": {"character": "IRONCLAD"}, "game_version": "v0.111.0"}
    with pytest.raises(ri.InstinctUnavailable, match="no Instinct net"):
        advisor.advise(msg, "map_select")


def test_unavailable_instinct_is_marked_on_the_search_answer() -> None:
    advice = ra.mark_instinct_unavailable({"note": "MCTS 500 sims"}, "Instinct plays Ironclad only")
    assert advice["brain"] == "search"
    assert advice["instinct_unavailable"] == "Instinct plays Ironclad only"
    assert advice["note"].startswith("Search (Instinct: Instinct plays Ironclad only)")
