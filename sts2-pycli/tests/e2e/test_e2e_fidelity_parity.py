"""The F5/F6 lockstep driver's engine-side pieces (``ppo.e2e.fidelity_parity``).

The C#-oracle half runs as the gates themselves; these tests need only the
wheel. The first is the regression that retired the #918 script: the trial
core passes ``encounter_floor``, and the init must take it.
"""

from __future__ import annotations

import collections
import inspect

import parity_audit
import pytest
import sts2_client
import sts2_rs
from ppo.e2e import fidelity_parity as fp


@pytest.fixture
def v0111(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.111.0")
    monkeypatch.setattr(fp, "_ASC", 10)


def test_rust_init_takes_every_argument_the_trial_core_passes(v0111) -> None:
    core = inspect.getsource(parity_audit._run_trial_core)
    assert "encounter_floor=floor" in core
    sc = fp.encounter_scenario("DECIMILLIPEDE_ELITE", 0, "v1")
    state = fp._rust_init(sc, monsters=None, move_states=None, encounter_floor=7)
    d = sts2_rs.state_to_dict(state.inner)
    assert d["game_version"] == "v0.111.0"
    assert len(d["encounter"]) >= 2


def test_encounter_scenarios_are_deterministic(v0111) -> None:
    for pool in ("v1", "v2"):
        a = fp.encounter_scenario("KNIGHTS_ELITE", 5, pool)
        b = fp.encounter_scenario("KNIGHTS_ELITE", 5, pool)
        assert a == b
        assert a.encounter == "KNIGHTS_ELITE"
    forced = fp.encounter_scenario("KNIGHTS_ELITE", 5, "v1", ("BRONZE_SCALES",))
    assert forced.relics == ("BRONZE_SCALES",)


def test_multi_enemy_encounters_have_two_or_more_enemies(v0111) -> None:
    multi = fp.multi_enemy_encounters()
    assert len(multi) >= 20
    assert "NIBBITS_NORMAL" in multi
    assert "VANTOM_BOSS" not in multi


def test_random_target_scenarios_build_and_cover_every_source(v0111) -> None:
    multi = fp.multi_enemy_encounters()
    seen: collections.Counter[str] = collections.Counter()
    for seed in range(60):
        sc = fp.random_target_scenario(seed, multi)
        assert sc == fp.random_target_scenario(seed, multi)
        assert sc.encounter in multi
        sources = [c for c in sc.deck if c.split("+")[0] in fp.RANDOM_TARGET_CARDS]
        assert 3 <= len(sources) <= 6
        seen.update(c.split("+")[0] for c in sources)
        seen.update(sc.relics)
        fp._rust_init(sc)  # Rust takes every id
    missing = set(fp.RANDOM_TARGET_CARDS + fp.RANDOM_TARGET_RELICS) - set(seen)
    assert not missing


def test_x_cost_tokens() -> None:
    assert fp._x_token("CARD.HEAVENLY_DRILL") == "HEAVENLYDRILL"
    assert fp._x_token("VOLLEY+1") in fp._X_COST
    assert fp._x_token("CARD.CASCADE") in fp._X_COST
    assert fp._x_token("CARD.STRIKE_IRONCLAD") not in fp._X_COST


def test_setup_deaths_do_not_shift_the_forced_move_states(v0111, monkeypatch) -> None:
    # F6 seed 569: Whispering Earring auto-plays turn 1 and kills the first
    # rat before C#'s first snapshot, which then lists two living rats with
    # moves SCRATCH and DISEASE_BITE (observed on the v0.111.0 oracle).
    sc = fp.random_target_scenario(569, fp.multi_enemy_encounters())
    assert "WHISPERING_EARRING" in sc.relics
    assert sc.encounter == "TWO_TAILED_RATS_NORMAL"
    forced = ["SCRATCH_MOVE", "DISEASE_BITE_MOVE"]

    def living_moves() -> list[str]:
        d = sts2_rs.state_to_dict(fp._rust_init(sc, move_states=forced, encounter_floor=1).inner)
        return [m["move_state"] for m in d["encounter"] if not m["removed_after_death"]]

    monkeypatch.setattr(fp, "_ALIGN_SETUP_DEATHS", False)
    assert living_moves() == ["DISEASE_BITE", "DISEASE_BITE"]  # shifted by one
    monkeypatch.setattr(fp, "_ALIGN_SETUP_DEATHS", True)
    assert living_moves() == ["SCRATCH", "DISEASE_BITE"]


def test_the_a10_map_compares_the_second_boss_as_a_meta_node() -> None:
    from ppo.e2e.act23_a10 import norm_map_graph_a10

    second = {"col": 3, "row": 15, "type": "Boss", "id": "AEONGLASS_BOSS"}
    boss = {"col": 3, "row": 14, "type": "Boss"}
    elite = {"col": 3, "row": 13, "type": "Elite", "children": []}
    rust = {"rows": [[elite], [{"col": 3, "row": 15, "type": "Boss"}]], "boss": boss}
    csharp = {"rows": [[elite]], "boss": boss, "second_boss": second}
    rust["second_boss"] = second
    assert norm_map_graph_a10(rust) == norm_map_graph_a10(csharp)
    assert norm_map_graph_a10(csharp)["BOSS2"] == ("BOSS", (3, 15))
    below_a10 = {"rows": [[elite]], "boss": boss, "second_boss": None}
    assert "BOSS2" not in norm_map_graph_a10(below_a10)
