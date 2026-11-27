"""`ppo.e2e.archetype_probe`: the labels, the seed split and the ridge fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ppo.e2e import archetype_probe as AP


def test_poison_turns_counts_the_decaying_ticks() -> None:
    assert AP.poison_turns(5, 5) == 1  # 5
    assert AP.poison_turns(9, 5) == 2  # 5 + 4
    assert AP.poison_turns(15, 5) == 5  # 5 + 4 + 3 + 2 + 1
    assert AP.poison_turns(16, 5) == AP.CAP + 1  # Poison runs out first
    assert AP.poison_turns(1, 0) == AP.CAP + 1
    assert AP.poison_turns(100, 30, cap=3) == 4  # 30 + 29 + 28 < 100


def test_power_sums_a_named_power() -> None:
    m = {"powers": [{"name": "DOOM", "amount": 7}, {"name": "POISON", "amount": 3}]}
    assert (AP._power(m, "DOOM"), AP._power(m, "POISON"), AP._power(m, "WEAK")) == (7, 3, 0)
    assert AP._power({"powers": None}, "DOOM") == 0


def test_the_split_keeps_a_seed_on_one_side() -> None:
    seeds = np.repeat(np.arange(500), 3)
    te = AP._split(seeds)
    per_seed = te.reshape(500, 3)
    assert (per_seed.all(1) | ~per_seed.any(1)).all()
    assert 0.1 < te.mean() < 0.3


def test_fit_one_recovers_a_linear_label_and_not_noise() -> None:
    rng = np.random.default_rng(0)
    seeds = np.repeat(np.arange(400), 5)
    x = rng.normal(size=(len(seeds), 16))
    te = AP._split(seeds)
    y = x @ rng.normal(size=16)
    assert AP._fit_one(x, y, te, binary=False) > 0.99
    assert AP._fit_one(x, rng.normal(size=len(y)), te, binary=False) < 0.05
    yb = (x[:, 0] > 0).astype(float)
    assert AP._fit_one(x, yb, te, binary=True) > 0.9


def test_archetype_sets_cover_the_forced_targets() -> None:
    from ppo.e2e import forced_rollout as FR

    for char, t in FR.TARGETS.items():
        assert t.payoff <= AP.ARCHETYPE[char]
    assert "REAPER_FORM" in AP.ARCHETYPE["Necrobinder"]


@pytest.mark.parametrize("probe", ["doom", "poison", "deck", "option"])
def test_fit_reads_the_collected_shards(tmp_path: Path, probe: str) -> None:
    rng = np.random.default_rng(1)
    n = 600
    z = {
        "h": rng.normal(size=(n, 8)).astype(np.float16),
        "x0": rng.normal(size=(n, 8)).astype(np.float16),
        "seed": np.repeat(np.arange(n // 3), 3),
        "hp": rng.integers(1, 100, n),
        "max_hp": np.full(n, 100),
        "doom": rng.integers(0, 50, n),
        "poison": rng.integers(0, 20, n),
        "turns": rng.integers(1, AP.CAP + 2, n),
        "arch": rng.integers(0, 5, n),
    }
    z["max_hp"][0] = 999_999_999  # a placeholder enemy: dropped
    if probe in ("deck", "option"):
        del z["max_hp"]  # out-of-combat rows carry no enemy
    np.savez_compressed(tmp_path / f"Necrobinder_0000_{probe}.npz", **z)
    AP.main(["fit", str(tmp_path)])
    res = json.loads((tmp_path / "probe_fit.json").read_text(encoding="utf-8"))
    rows = [v for k, v in res.items() if k.startswith(probe + "/")]
    assert rows
    assert all(r["n"] == (n - 1 if probe in ("doom", "poison") else n) for r in rows)
