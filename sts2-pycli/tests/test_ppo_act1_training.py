"""GH #545: training must be able to reach the Underdocks Act 1.

`PpoRunEnv` deliberately omitted `act1` (env.py:377), so every champion to
date trained on 0% Underdocks and the three Underdocks boss embedding rows
stayed at their OOV init.
"""

import pytest
from ppo.env import PpoRunEnv


def test_env_defaults_to_the_seed_roll():
    """No `act1` means the real game's seed-driven pick, not Overgrowth."""
    for s in range(1, 40):
        default = PpoRunEnv(seed=s, character="Ironclad").rm.act1
        assert default == PpoRunEnv(seed=s, character="Ironclad", act1="roll").rm.act1


def test_env_can_be_pointed_at_underdocks():
    env = PpoRunEnv(seed=7, character="Ironclad", act1="underdocks")
    assert env.rm.act1 == "underdocks"


def test_env_roll_resolves_to_one_of_the_two_act1_models():
    acts = {PpoRunEnv(seed=s, character="Ironclad", act1="roll").rm.act1 for s in range(1, 40)}
    # A seed-driven 50/50 over 39 seeds must produce both.
    assert acts == {"overgrowth", "underdocks"}


def test_env_rejects_an_unknown_act1():
    with pytest.raises(ValueError):
        PpoRunEnv(seed=7, character="Ironclad", act1="atlantis")


def test_play_one_records_the_resolved_act():
    from ppo import workers

    row = workers.play_one(seed=7, policy="random", act1="underdocks", max_steps=3)
    assert row["act1"] == "underdocks"
