"""RL2 S04 (max energy / 3) reads the live value for the Instinct tokenizer.

`player.max_energy` holds what the last turn start computed, so a Pyre gained
this turn shows natively only from the next turn, while the game screen (and a
fight rebuilt from it) reads `PlayerCombatState.MaxEnergy` live
(`PyrePower.ModifyMaxEnergy`). `ppo/e2e/tokens.py` asks for the live value; the
rollout nets keep the turn-start one they were trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from realgame_adapter import to_headless_combat_play
from sts2_hint.adapter import translate

sts2_rs = pytest.importorskip("sts2_rs")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
)


def test_rl2_max_energy_counts_a_pyre_gained_this_turn() -> None:
    wire = translate(
        to_headless_combat_play(json.loads(FIXTURE.read_text(encoding="utf-8"))["state"])
    )
    assert wire["player"]["max_energy"] == 3
    wire["player"]["powers"] = [{"name": "PYRE", "amount": 1}]
    state = sts2_rs.state_from_dict(wire)
    turn_start = sts2_rs.rollout_features(state, version=1)["state"][4]
    live = sts2_rs.rollout_features(state, version=1, live_max_energy=True)["state"][4]
    assert turn_start == pytest.approx(3 / 3)
    assert live == pytest.approx(4 / 3)
