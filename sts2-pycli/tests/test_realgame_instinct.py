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
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_advisor as ra  # noqa: E402
import realgame_instinct as ri  # noqa: E402
from floors import game_floor  # noqa: E402

SEED = "INSTINCTSEED7"
WEIGHTS = ri.default_weights_path()
# Silent and Necrobinder die in act 1 of SEED (Defect too since the d4500 net); this one
# reaches act 2 (re-picked when each act's first MAP row began flagging its moves,
# GH #1616: the greedy runs moved).
_ACT2_SEED = {
    "Silent": "INSTINCTSEED1",
    "Necrobinder": "INSTINCTSEED1",
    "Regent": "INSTINCTSEED1",
    "Defect": "INSTINCTSEED1",
}
# The run token (row 0) cell holding the fight's encounter id.


@pytest.fixture(scope="module")
def student() -> Any:
    return sts2_rs.E2eStudent(str(WEIGHTS))


def _campaign_run(
    seed: str, build: str = ri.INSTINCT_GAME_VERSION, character: str = "Ironclad"
) -> Any:
    """The student's campaign run, at a STRING seed (as live runs have)."""
    return sts2_rs.RunManager(
        game_version=build,
        string_seed=seed,
        character=character,
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


def _deck_card(name: str, upgraded: bool, ench: tuple[str, int] | None) -> dict[str, Any]:
    """A master-deck card as the mod sends it, its enchantment included."""
    card: dict[str, Any] = {"id": _wire_id(name), "upgrade_level": int(upgraded)}
    if ench is not None:
        card["enchantment"], card["enchantment_amount"] = ench
    return card


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
    msg: dict[str, Any] = {
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
            "character": str(rm.character).upper(),
            "ascension": rm.ascension_level,
            "hp": rm.current_hp,
            "max_hp": rm.max_hp,
            "gold": rm.gold,
            "potion_slot_count": slots,
            "master_deck": [
                _deck_card(name, up, ench)
                for (name, up), ench in zip(
                    rm.master_deck_detailed(), rm.master_deck_enchantments(), strict=True
                )
            ],
            "relics": [{"id": _wire_id(r)} for r in rm.relics()],
            "potions": potions,
        },
    }
    # The room the player stands in, as the mod names it (the game's
    # `RoomType`): what a `?` node turned out to be (GH #1646).
    room = _GAME_ROOM.get((rm.current_room() or {}).get("room_type"))
    if room is not None:
        msg["current_room"] = room
    return msg


#: Engine room type -> the game's `RoomType` name the mod sends as ``current_room``.
_GAME_ROOM = {
    "monster": "Monster",
    "elite": "Elite",
    "boss": "Boss",
    "event": "Event",
    "shop": "Shop",
    "rest_site": "RestSite",
    "treasure": "Treasure",
}


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="no Instinct weights (data/e2e)")
class TestInstinct:
    def test_the_graft_holds_exactly_the_live_relics(self, student: Any) -> None:
        """The fresh run's starter relic does not survive the graft. Left in
        place, Touch of Orobas upgraded it again on pickup, so a Regent who had
        taken Orobas held two Divine Destiny."""
        rm = _campaign_run(SEED, character="Regent")
        driver = sts2_rs.E2eDriver(rm.seed)
        _play_to(student, rm, driver, lambda p: p.decision["decision_type"] == "MAP")
        msg = _envelope(rm, SEED, "map_select")
        live = ["DIVINE_DESTINY", "TOUCH_OF_OROBAS"]
        msg["run_state"]["relics"] = [{"id": r} for r in live]
        grafted, _notes = ri.graft_run(msg)
        assert list(grafted.relics()) == [f"RELIC.{r}" for r in live]

    def test_the_smoke_check_plays_the_shipped_net(self) -> None:
        """`realgame_advisor --smoke` (the packaged exe's build gate) checks
        Instinct with 20 self-play decisions. It read `INSTINCT._student`,
        which went into Rust with the advisor core (GH #1426 step 4), and so
        failed on every build that shipped the net."""
        assert ra._smoke_instinct() == f"instinct OK ({WEIGHTS.name}, 20 decisions)"

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

    @pytest.mark.parametrize("character", ri.INSTINCT_CHARACTERS)
    def test_map_advice_from_an_envelope_sees_what_the_native_student_sees(
        self, student: Any, character: str
    ) -> None:
        seed = _ACT2_SEED.get(character, SEED)
        rm = _campaign_run(seed, character=character)
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

        msg = _envelope(rm, seed, "map_select")
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

    @pytest.mark.parametrize("character", ri.INSTINCT_CHARACTERS)
    def test_a_fight_without_a_map_sees_what_the_native_student_sees(
        self, student: Any, character: str
    ) -> None:
        """GH #1212: fight requests carry no map graph, and the graft left the
        run at act 1 / floor 0, so the student read every live fight as the
        run's first room. A native act-2 fight, grafted from an envelope with
        no ``map``, must stand where the native run stands and give the same
        probabilities. Every character (GH #1197): orbs, Osty, stars and Doom
        ride the fight's own state."""
        seed = _ACT2_SEED.get(character, SEED)
        rm = _campaign_run(seed, character=character)
        driver = sts2_rs.E2eDriver(rm.seed)

        def act2_fight(peek: Any) -> bool:
            return rm.act_index >= 2 and peek.decision["decision_type"] == "COMBAT_TURN"

        _play_to(student, rm, driver, act2_fight)
        room = str(rm.current_room()["room_type"]).lower()

        msg = _envelope(rm, seed, "combat_action")
        del msg["map"]
        msg["room_type"] = room.capitalize()
        # The mod names the fight's encounter (CombatState.Encounter.Id.Entry).
        msg["encounter_id"] = rm.combat_state().encounter_id
        assert msg["encounter_id"]
        grafted, notes = ri.graft_run(msg)
        assert any(n.startswith("no map position") for n in notes)
        assert (grafted.act_index, grafted.total_floor(), grafted.act_floor()) == (
            rm.act_index,
            rm.total_floor(),
            rm.act_floor(),
        )
        assert grafted.full_map()["current_coord"]["row"] == rm.full_map()["current_coord"]["row"]

        ri.enter_live_combat(grafted, rm.combat_state(), msg, room)
        assert grafted.combat_state().encounter_id == msg["encounter_id"]
        a, b = driver.copy(), driver.copy()
        want = student.encode(a, a.peek(rm), rm)
        got = student.encode(b, b.peek(grafted), grafted)
        assert got["n_tok"] == want["n_tok"]
        for key, dt in (
            ("cat", np.int16),
            ("cont", np.uint16),
            ("sec", np.int8),
            ("tok_flags", np.uint8),
        ):
            x = np.frombuffer(want[key], dt)
            y = np.frombuffer(got[key], dt)
            diff = np.nonzero(x != y)[0]
            assert diff.size == 0, (key, diff[:10], x[diff[:10]], y[diff[:10]])
        native = student.decide(driver.copy(), driver.copy().peek(rm), rm)
        probe = driver.copy()
        out = student.decide(probe, probe.peek(grafted), grafted)
        assert out["probs"] == pytest.approx(native["probs"], abs=2e-3)

    @pytest.mark.parametrize(
        "msg",
        [
            {"encounter_id": "FAKE_MERCHANT_EVENT_ENCOUNTER"},
            {"encounter_id": "PUNCH_OFF_EVENT_ENCOUNTER"},
            # An in-combat screen (card_select) embeds the board.
            {"combat": {"encounter_id": "PUNCH_OFF_EVENT_ENCOUNTER"}},
        ],
    )
    def test_an_event_fight_names_its_event_encounter(
        self, student: Any, msg: dict[str, Any]
    ) -> None:
        """GH #1477: an event fight's id (the sim's event encounter, as the
        native env reports it) names the grafted fight too."""
        rm = _campaign_run(SEED)
        driver = sts2_rs.E2eDriver(rm.seed)
        _play_to(student, rm, driver, lambda p: p.decision["decision_type"] == "COMBAT_TURN")
        grafted = _campaign_run(SEED)
        ri.enter_live_combat(grafted, rm.combat_state(), msg, "monster")
        want = msg.get("encounter_id") or msg["combat"]["encounter_id"]
        assert grafted.combat_state().encounter_id == want

    def test_an_unknown_fight_id_leaves_the_run_without_an_encounter(self, student: Any) -> None:
        """An id the sim has no fight for must not cost the advice: the run is
        entered without one."""
        rm = _campaign_run(SEED)
        driver = sts2_rs.E2eDriver(rm.seed)
        _play_to(student, rm, driver, lambda p: p.decision["decision_type"] == "COMBAT_TURN")
        grafted = _campaign_run(SEED)
        msg = {"encounter_id": "NOT_AN_ENCOUNTER"}
        ri.enter_live_combat(grafted, rm.combat_state(), msg, "monster")
        assert grafted.combat_state().encounter_id is None

    def test_other_characters_fall_back_to_the_search_with_a_reason(self, tmp_path: Path) -> None:
        advisor = ri.InstinctAdvisor(WEIGHTS, tmp_path / "no_fool.e2ew")
        msg = {"run_state": {"character": "WATCHER"}, "game_version": "v0.113.0"}
        with pytest.raises(ri.InstinctUnavailable, match="Ironclad, Silent"):
            advisor.advise(msg, "map_select")
        # The Fool has its own net (GH #1535), on v0.113.0 and v0.114.0 only.
        msg = {"run_state": {"character": "FOOL"}, "game_version": "v0.111.0"}
        with pytest.raises(ri.InstinctUnavailable, match=r"the Fool on v0.113.0 and v0.114.0 only"):
            advisor.advise(msg, "map_select")
        msg = {"run_state": {"character": "FOOL"}, "game_version": "v0.113.0"}
        with pytest.raises(ri.InstinctUnavailable, match=r"no Instinct net at .*no_fool.e2ew"):
            advisor.advise(msg, "map_select")
        msg = {"run_state": {"character": "IRONCLAD"}, "game_version": "v0.107.1"}
        with pytest.raises(ri.InstinctUnavailable, match=r"v0.111.0, v0.113.0 and v0.114.0 only"):
            advisor.advise(msg, "map_select")

    def test_every_character_list_matches_the_engines(self) -> None:
        """The engine's ``sts2_e2e::env::CHARACTERS`` is the one source; the
        Python, web-advisor and seed-checker lists are copies a new character
        must reach (GH #1197)."""
        engine = list(sts2_rs.E2E_CHARACTERS)
        assert list(ri.INSTINCT_CHARACTERS) == engine
        web = Path(__file__).resolve().parents[1] / "web" / "src"
        advisor = web / "advisor.ts"
        if not advisor.exists():
            pytest.skip("the web workbench is not in this checkout")
        m = re.search(r"E2E_CHARACTERS = \[([^\]]*)\]", advisor.read_text(encoding="utf-8"))
        assert m, "advisor.ts: E2E_CHARACTERS not found"
        # The web list also names the Fool, whose own net plays it (GH #1535).
        assert re.findall(r'"(\w+)"', m.group(1)) == [*engine, "Fool"]
        page = (web / "seedcheck" / "SeedCheckPage.tsx").read_text(encoding="utf-8")
        enabled = re.findall(r'\{ id: "(\w+)", enabled: true \}', page.split("VERSIONS")[0])
        assert sorted(enabled) == sorted(engine)

    @pytest.mark.parametrize("build", ["v0.111.0", "v0.113.0"])
    @pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"])
    def test_its_characters_are_eligible_on_the_builds_it_plays(
        self, build: str, character: str
    ) -> None:
        msg = {"run_state": {"character": character}, "game_version": build}
        assert ri.InstinctAdvisor(WEIGHTS).eligibility(msg) is None

    def test_the_shipped_fool_net_plays_the_fool_on_v0113(self) -> None:
        advisor = ri.InstinctAdvisor(WEIGHTS)
        assert (
            advisor.eligibility({"run_state": {"character": "FOOL"}, "game_version": "v0.113.0"})
            is None
        )

    def test_capabilities_name_what_instinct_plays(self) -> None:
        assert ri.InstinctAdvisor(WEIGHTS).capabilities() == [
            {"character": c, "game_version": v}
            for c in ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
            for v in ("v0.111.0", "v0.113.0", "v0.114.0")
        ] + [{"character": "Fool", "game_version": v} for v in ("v0.113.0", "v0.114.0")]

    @pytest.mark.parametrize("character", ri.INSTINCT_CHARACTERS)
    @pytest.mark.parametrize("build", ["v0.111.0", "v0.113.0"])
    def test_a_run_is_rebuilt_on_the_build_it_is_played_on(
        self, student: Any, build: str, character: str
    ) -> None:
        """A live run is grafted onto its own build's engine, map and all.

        v0.113.0 generates maps differently from v0.111.0, the student's
        training build. Grafted onto v0.111.0, a v0.113.0 run got another
        map, the player's room was missing from it ("the map does not match
        this run"), and every Instinct screen that rebuilds the run -- an
        in-fight Skill Potion offer included -- fell back to the search.
        """
        rm = _campaign_run(SEED, build, character)
        driver = sts2_rs.E2eDriver(rm.seed)

        maps = 0

        def second_map(peek: Any) -> bool:
            nonlocal maps
            if peek.decision["decision_type"] != "MAP":
                return False
            maps += 1
            return maps == 2

        peek = _play_to(student, rm, driver, second_map)
        native = student.decide(driver.copy(), peek, rm)
        cols = [c["actions"][0]["col"] for c in peek.decision["classes"]]

        msg = _envelope(rm, SEED, "map_select")
        msg["game_version"] = build
        grafted, _notes = ri.graft_run(msg)
        assert _map_shape(grafted) == _map_shape(rm)

        children = {(c["col"], c["row"]) for c in _children_of(msg)}
        msg["nodes"] = [
            {"index": i, "col": col, "row": row, "type": "Monster"}
            for i, (col, row) in enumerate(sorted(children))
        ]
        advice = ri.InstinctAdvisor(WEIGHTS).advise(msg, "map_select", session={})
        by_col = {o["col"]: o["prob"] for o in advice["options"]}
        assert sorted(by_col) == sorted(cols)
        for ci, col in enumerate(cols):
            assert by_col[col] == pytest.approx(native["probs"][ci], abs=2e-3)

    def test_the_two_builds_draw_different_maps(self) -> None:
        """Why the build matters: the same seed is another map on v0.113.0."""
        assert _map_shape(_campaign_run(SEED, "v0.111.0")) != _map_shape(
            _campaign_run(SEED, "v0.113.0")
        )


def _map_shape(rm: Any) -> list[tuple[int, int, str]]:
    return sorted(
        (n["col"], n["row"], str(n["type"])) for row in rm.full_map()["rows"] for n in row
    )


def _children_of(msg: dict[str, Any]) -> list[dict[str, Any]]:
    cur = msg["map"]["current"]
    if cur is None:
        # An act's first move (GH #1616): no node is current yet; the player
        # stands on the start node, the one node of row 0.
        (cur,) = [n for n in msg["map"]["nodes"] if n["row"] == 0]
    for n in msg["map"]["nodes"]:
        if n["col"] == cur["col"] and n["row"] == cur["row"]:
            return n["children"]
    return []


def test_a_missing_net_is_a_reason_not_a_crash(tmp_path: Any) -> None:
    advisor = ri.InstinctAdvisor(tmp_path / "nope.e2ew", tmp_path / "no_fool.e2ew")
    assert advisor.capabilities() == []
    msg = {"run_state": {"character": "IRONCLAD"}, "game_version": "v0.111.0"}
    with pytest.raises(ri.InstinctUnavailable, match="no Instinct net"):
        advisor.advise(msg, "map_select")


def test_unavailable_instinct_is_marked_on_the_search_answer() -> None:
    advice = ra.mark_instinct_unavailable({"note": "MCTS 500 sims"}, "Instinct plays Ironclad only")
    assert advice["brain"] == "search"
    assert advice["instinct_unavailable"] == "Instinct plays Ironclad only"
    assert advice["note"].startswith("Search (Instinct: Instinct plays Ironclad only)")


def test_a_mirror_that_declines_on_its_own_keeps_the_old_reason(tmp_path: Path) -> None:
    """No fight to stand on: the mirror declines, and the reason is its own.

    When Instinct's own pick fails, its reason wins instead (a Skill Potion's
    offer used to read "cannot stand on this choice" while the search then
    answered the same screen): ``sts2_advisor::instinct_combat``'s
    ``a_pick_instinct_cannot_make_says_why``.
    """
    from realgame_advisor_mirror import AdvisorMirror
    from sts2_hint.recommender import RecommendConfig

    cfg = RecommendConfig(threads=1)
    mirror = AdvisorMirror(cfg, log_path=tmp_path / "mirror.jsonl")
    with pytest.raises(ri.InstinctUnavailable, match="cannot stand on this choice"):
        ri.InstinctAdvisor()._advise_combat_select({"combat": {}}, cfg, mirror, 8)
