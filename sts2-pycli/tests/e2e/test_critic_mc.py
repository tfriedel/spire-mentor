"""`ppo.e2e.critic_mc`'s report: the noise correction recovers a critic's real R²."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ppo.e2e import critic_mc as M


def _states(v_true: np.ndarray, v_critic: np.ndarray, k: int, sd: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [
        {"v_run": float(c), "g": t + rng.normal(0.0, sd, k), "act": 1, "char": "X", "dtype": 0}
        for t, c in zip(v_true, v_critic, strict=True)
    ]


def test_r2_vs_true_value_removes_the_monte_carlo_noise() -> None:
    rng = np.random.default_rng(0)
    v_true = rng.normal(0.8, 0.2, 20_000)
    err = rng.normal(0.0, 0.1, v_true.size)  # R² = 1 - 0.01 / 0.04 = 0.75
    states = _states(v_true, v_true + err, k=8, sd=0.4, seed=1)
    s = M.summarize(states)
    assert s.r2_true == pytest.approx(0.75, abs=0.02)
    # the naive R² against the noisy V_MC is far lower: the correction is what matters
    v = np.array([st["v_run"] for st in states])
    m = np.array([st["g"].mean() for st in states])
    assert 1 - np.mean((v - m) ** 2) / m.var() < 0.6
    # ceiling: var(true V) / var(G) = 0.04 / (0.04 + 0.16)
    assert s.ceiling_ev == pytest.approx(0.2, abs=0.01)
    assert s.bias == pytest.approx(0.0, abs=0.005)


def test_a_perfect_critic_reads_r2_one_and_a_constant_reads_zero() -> None:
    rng = np.random.default_rng(2)
    v_true = rng.normal(0.5, 0.3, 10_000)
    assert M.summarize(_states(v_true, v_true, 8, 0.5, 3)).r2_true == pytest.approx(1.0, abs=0.03)
    const = np.full_like(v_true, v_true.mean())
    assert M.summarize(_states(v_true, const, 8, 0.5, 4)).r2_true == pytest.approx(0.0, abs=0.03)


def test_read_states_skips_meta_and_states_with_too_few_returns(tmp_path) -> None:
    p = tmp_path / "mc.jsonl"
    rows = [
        {"meta": {"k": 2}},
        {"v_run": 0.5, "act": 1, "char": "X", "dtype": 0, "rollouts": [{"G": 1.0}, {"G": 0.0}]},
        {"v_run": 0.5, "act": 1, "char": "X", "dtype": 0, "rollouts": [{"G": 1.0}, {"error": "x"}]},
        {"file": "f", "error": "replay"},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = M.read_states([p])
    assert len(got) == 1
    assert got[0]["g"].tolist() == [1.0, 0.0]


def test_cli_parses_both_subcommands() -> None:
    with pytest.raises(SystemExit):
        M.main(["run"])  # missing required args
    with pytest.raises(SystemExit):
        M.main([])
