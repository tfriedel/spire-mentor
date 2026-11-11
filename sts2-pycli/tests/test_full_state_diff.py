"""Tests for full_state_diff — the act23 full-state combat diff (GH #739 oracle half)."""

from __future__ import annotations

import copy

from full_state_diff import (
    canon_card_id,
    combat_split_lines,
    diff_full,
    path_class,
    project_csharp,
    project_rust_dict,
)


def _csharp() -> dict:
    """Minimal C# ``get_state`` response (``RunSimulator.GetStateImpl``)."""
    return {
        "type": "ok",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
        "monsters": [
            {
                "uid": 0,
                "kind": "NIBBIT",
                "hp": 8,
                "max_hp": 8,
                "block": 0,
                "alive": True,
                "powers": [],
                "move_state": "BUTT_MOVE",
            }
        ],
        "hand_count": 2,
        "hand_ids": ["CARD.STRIKE_IRONCLAD", "CARD.CASCADE"],
        "hand_costs": [1, 3],
        "draw_count": 2,
        "draw_ids": ["CARD.RUPTURE", "CARD.DEFEND_IRONCLAD"],
        "discard_count": 0,
        "discard_ids": [],
        "exhaust_count": 0,
        "exhaust_ids": [],
        "turn": 1,
        "relics": ["BURNING_BLOOD", "NEOWS_SACRIFICE"],
        "combat_terminal": False,
        "player_won": None,
        "rng_counters": {"shuffle": 9, "combat_targets": 2, "rewards": 0},
    }


def _rust() -> dict:
    """The matching ``sts2_rs.state_to_dict`` shape."""
    return {
        "turn": 1,
        "phase": "PlayerPlayPhase",
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "powers": [],
            "relics": [{"id": "BURNING_BLOOD"}],
        },
        "deck": {
            "hand": [
                {"id": "STRIKE_IRONCLAD", "base_cost": 1, "cost_override": 1},
                # X-cost: Rust `base_cost` -1, C# reports the current energy.
                {"id": "CASCADE", "base_cost": -1, "cost_override": 0},
            ],
            "draw_pile": [{"id": "RUPTURE_CARD"}, {"id": "DEFEND_IRONCLAD"}],
            "discard_pile": [],
            "exhaust_pile": [],
        },
        "encounter": [
            {
                "uid": 7,
                "kind": "Nibbit",
                "hp": 8,
                "max_hp": 8,
                "block": 0,
                "is_dead": False,
                "powers": [],
                "move_state": "BUTT",
            }
        ],
        # `run_manager` is Rust-only: not a desync.
        "rng": {"counters": {"shuffle": 9, "combat_targets": 2, "run_manager": 4}},
    }


def test_matching_states_have_no_split():
    assert diff_full(project_csharp(_csharp()), project_rust_dict(_rust())) == []


def test_card_ids_canonicalize_across_engines():
    assert canon_card_id("CARD.RUPTURE") == canon_card_id("RUPTURE_CARD") == "RUPTURE"
    assert canon_card_id(None) is None


def test_draw_order_split_is_reported():
    rs = _rust()
    rs["deck"]["draw_pile"].reverse()
    paths = [d["path"] for d in diff_full(project_csharp(_csharp()), project_rust_dict(rs))]
    assert paths == ["piles.draw[0]", "piles.draw[1]"]


def test_rng_cursor_split_is_reported_on_shared_streams_only():
    rs = _rust()
    rs["rng"]["counters"]["combat_targets"] = 3
    divs = diff_full(project_csharp(_csharp()), project_rust_dict(rs))
    assert divs == [{"path": "rng.combat_targets", "csharp": 2, "rust": 3}]


def test_x_cost_card_skips_the_cost_check_but_others_do_not():
    rs = _rust()
    rs["deck"]["hand"][0]["cost_override"] = 0
    divs = diff_full(project_csharp(_csharp()), project_rust_dict(rs))
    assert divs == [{"path": "hand_costs[0]", "csharp": 1, "rust": 0}]


def test_relics_are_left_to_the_coarse_diff():
    cs = copy.deepcopy(_csharp())
    cs["relics"].append("ANCHOR")
    assert diff_full(project_csharp(cs), project_rust_dict(_rust())) == []


def test_rust_only_monster_variant_matches_its_csharp_kind():
    cs = _csharp()
    cs["monsters"][0]["kind"] = "WRIGGLER"
    rs = _rust()
    rs["encounter"][0]["kind"] = "DenseVegetationWriggler"
    assert diff_full(project_csharp(cs), project_rust_dict(rs)) == []


def test_path_class_strips_indices_and_len():
    assert path_class("piles.hand[3]") == "piles.hand"
    assert path_class("monsters[0].powers[1].amount") == "monsters.powers.amount"
    assert path_class("piles.draw.len") == "piles.draw"


def _with_afflictions(cs_affs: list, cs_amounts: list) -> dict:
    cs = copy.deepcopy(_csharp())
    cs["hand_afflictions"] = cs_affs
    cs["hand_affliction_amounts"] = cs_amounts
    return cs


def test_bound_affliction_split_is_reported():
    """GH #1150: a Bound split shows on the step it happens."""
    cs = _with_afflictions(["BOUND", None], [2, 0])
    assert diff_full(project_csharp(cs), project_rust_dict(_rust())) == [
        {"path": "hand_afflictions[0]", "csharp": {"id": "BOUND", "amount": 2}, "rust": None}
    ]
    rs = _rust()
    rs["deck"]["hand"][0]["affliction"] = "Bound"
    rs["deck"]["hand"][0]["affliction_stacks"] = 2
    assert diff_full(project_csharp(cs), project_rust_dict(rs)) == []


def test_affliction_amount_split_is_reported():
    cs = _with_afflictions(["BOUND", None], [2, 0])
    rs = _rust()
    rs["deck"]["hand"][0]["affliction"] = "Bound"
    rs["deck"]["hand"][0]["affliction_stacks"] = 1
    paths = [d["path"] for d in diff_full(project_csharp(cs), project_rust_dict(rs))]
    assert paths == ["hand_afflictions[0].amount"]


def test_oracle_without_afflictions_skips_the_check():
    rs = _rust()
    rs["deck"]["hand"][0]["affliction"] = "Bound"
    assert diff_full(project_csharp(_csharp()), project_rust_dict(rs)) == []


def test_combat_split_reports_a_one_sided_combat_exit():
    """Regent v0.107.1 seed 6488: C# still in combat, Rust at the card reward.
    The full-state projection never ran there, so the run printed
    ``full=clean`` beside a phase divergence."""
    phase = "phase: cs='combat' rs='card_reward'"
    assert combat_split_lines("combat", "card_reward", [], [phase]) == [phase]


def test_combat_split_prefers_the_full_state_lines():
    full = ["piles.hand[0]: cs='STRIKE' rs='DEFEND'"]
    assert combat_split_lines("combat", "combat", full, ["hp: cs=5 rs=6"]) == full


def test_combat_split_falls_back_to_the_coarse_fields():
    """A coarse mismatch the projection does not cover is still a split."""
    assert combat_split_lines("combat", "combat", [], ["gold: cs=5 rs=6"]) == ["gold: cs=5 rs=6"]
    assert combat_split_lines("combat", "combat", [], []) == []
