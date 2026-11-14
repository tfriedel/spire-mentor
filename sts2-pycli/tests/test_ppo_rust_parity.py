"""Phase-2 gates A2 + C: Rust obs builder and Rust PPO agent vs Python.

A2: sts2_rs.ppo_observe_json == ppo.env.observe on every yielded decision.
C:  Rust PpoRunAgent.evaluate matches Python NetPolicy.distribution
    (same argmax; |Δπ|, |ΔV| < 1e-3) along Python-greedy trajectories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import sts2_rs
from ppo.env import PpoRunEnv, Terminal
from ppo.net_policy import NetPolicy

CKPT = "data/ppo/ppo_fullrun_it350.pt"
ONNX = Path("data/ppo/ppo_fullrun_it350.onnx")
# `PARITY_SEEDS` caps both sweeps (the knob tests/test_ppo_env_parity.py reads
# too). Unset is the full sweep; the per-PR gate (`just test-slow-pr`) sets a
# smaller one, because every seed checks the same invariant.
_MAX_SEEDS = int(os.environ.get("PARITY_SEEDS", "20"))
A2_SEEDS = list(range(min(12, _MAX_SEEDS)))
C_SEEDS = list(range(min(20, _MAX_SEEDS)))

# Every test here walks a full PPO run (Python AND Rust) per seed, so the file
# costs ~2200 CPU-seconds across its 40 parametrizations — ~40% of the entire
# suite. Gated behind `slow` (deselected by the default addopts); run it with
# `just test-slow` / `pytest -m slow`.
pytestmark = [
    pytest.mark.skipif(not ONNX.exists(), reason="no ONNX snapshot"),
    pytest.mark.slow,
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Policy-mode `PpoRunEnv` drives combat with the combat-only
    # `MctsCombatAgent` (GH #220) — no net files or env vars needed.
    monkeypatch.delenv("STS2_RUNAI_VALUE", raising=False)
    # These gates assert Rust argmax == numpy argmax over the RAW
    # per-candidate probs. The Rust side's group-mass decode (A8 audit
    # §8, default ON) deliberately diverges from that on split-mass
    # screens — e.g. pooled ShopRemoveCard mass beating the single best
    # buy — so parity runs pin the plain per-candidate argmax. The
    # grouped decode itself is covered by Rust-side unit tests
    # (sts2-runai ppo::policy group_decode_tests).
    monkeypatch.setenv("STS2_RUNAI_PPO_GROUP_DECODE", "0")


@pytest.mark.parametrize("seed", A2_SEEDS)
def test_gate_a2_rust_obs_matches_python(seed):
    # Policy-mode walk (reset/step), NOT expert_run(): expert mode yields the
    # Decision AFTER the expert has already acted (env.py:287-301), so env.rm
    # would no longer be at the observed decision state.
    env = PpoRunEnv(seed=seed)
    pol = NetPolicy(CKPT)
    out = env.reset()
    while not isinstance(out, Terminal):
        rust = json.loads(sts2_rs.ppo_observe_json(env.rm, out.phase, out.surface, seed, 0))
        # `out.obs` is a native Python dict straight from the PyO3 getters
        # (e.g. `deck` is a list of *tuples*, not lists — JSON has no tuple
        # type), so a bare `==` against `rust` (which came out of
        # `json.loads`) would spuriously fail on every nested sequence even
        # when every value matches. Round-tripping `out.obs` through
        # `json.dumps`/`json.loads` normalizes it to the same tuple-free
        # shape `rust` already has, without relaxing any VALUE comparison —
        # `obs.rs`'s own `round_trips_every_fixture_obs_row` test uses the
        # same `serde_json::Value`-shaped equality for the identical reason.
        py_obs = json.loads(json.dumps(out.obs))
        assert rust == py_obs, f"obs drift @ seed {seed} step {out.step_index}"
        probs, _ = pol.distribution(out.obs, out.actions, out.metrics)
        out = env.step(int(np.argmax(probs)))


@pytest.mark.parametrize("seed", C_SEEDS)
def test_gate_c_rust_agent_matches_python_greedy(seed):
    os.environ.setdefault("STS2_RUNAI_PPO", str(ONNX.resolve()))
    env = PpoRunEnv(seed=seed)
    pol = NetPolicy(CKPT)
    agent = sts2_rs.PpoRunAgent(seed=seed)
    out = env.reset()
    checked = 0
    while not isinstance(out, Terminal):
        py_probs, py_value = pol.distribution(out.obs, out.actions, out.metrics)
        ev = agent.evaluate(env.rm)
        assert ev is not None, f"rust declined a multi-option decision @ seed {seed}"
        rs_probs, rs_value, rs_idx = ev
        py_idx = int(np.argmax(py_probs))
        if rs_idx != py_idx:
            # Candidate lists and per-option probabilities agree (asserted
            # below); the ONLY way the argmax itself can differ is a
            # genuine near-exact tie between two candidates (e.g. two
            # identical duplicate-card options at a smith/reward pick,
            # which the net scores equally). tract (Rust) and torch
            # (Python) resolve such ties differently because their
            # floating-point reduction order isn't bit-identical — this is
            # inherent to comparing two independent numerical runtimes at
            # an exact tie, not a masking/order bug (verified by hand for
            # seeds 4/6: candidate lists matched exactly, and the "wrong"
            # index's probability differs from the "right" one by ~1e-7).
            # Require the two candidates to be a real near-tie: 1e-5 is
            # ~100x the measured tie gap (~1e-7) yet 100x tighter than the
            # blanket |Δπ| < 1e-3 tolerance below, so a quiet ranking
            # regression in the 1e-4..1e-3 band still fails loudly instead
            # of hiding in the tie carve-out.
            assert abs(rs_probs[rs_idx] - rs_probs[py_idx]) < 1e-5, (
                f"argmax drift @ seed {seed} step {out.step_index}: rust picked "
                f"{rs_idx} ({rs_probs[rs_idx]!r}), python picked {py_idx} "
                f"({rs_probs[py_idx]!r}) — not a near-tie"
            )
        assert max(abs(a - b) for a, b in zip(rs_probs, py_probs)) < 1e-3
        assert abs(rs_value - py_value) < 1e-3
        checked += 1
        out = env.step(py_idx)
    assert checked > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_with_actions_aligns(seed):
    """PpoRunAgent.evaluate_with_actions returns a masked action list aligned
    with probs, and its (probs,value,argmax) equal the older evaluate()."""
    os.environ.setdefault("STS2_RUNAI_PPO", str(ONNX.resolve()))
    env = PpoRunEnv(seed=seed)
    agent = sts2_rs.PpoRunAgent(seed=seed)
    out = env.reset()
    checked = 0
    while not isinstance(out, Terminal):
        wa = agent.evaluate_with_actions(env.rm)
        base = agent.evaluate(env.rm)
        # Both decline (single-option) or both engage, in lockstep.
        assert (wa is None) == (base is None)
        if wa is not None:
            actions, probs, value, argmax = wa
            assert len(actions) == len(probs)
            assert 0 <= argmax < len(actions)
            b_probs, b_value, b_argmax = base
            assert probs == pytest.approx(b_probs, abs=0.0)
            assert value == pytest.approx(b_value, abs=0.0)
            assert argmax == b_argmax
            checked += 1
        # Drive with the base agent's argmax (or the env's first action when
        # single-option) to walk the same trajectory shape as Gate C.
        py_idx = base[2] if base is not None else 0
        out = env.step(py_idx)
    assert checked > 0


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_workbench_rust_backend_matches_python(seed):
    """workbench.ppo_policy.evaluate(impl='rust') agrees with impl='python'
    on argmax, per-option prob, V(s), and the resolved candidate labels."""
    # `python/` is already on sys.path via pyproject's `pythonpath = ["python"]`.
    os.environ.setdefault("STS2_RUNAI_PPO", str(ONNX.resolve()))
    from workbench import ppo_policy

    ppo_policy._reset_for_tests()
    env = PpoRunEnv(seed=seed)
    pol = NetPolicy(CKPT)
    out = env.reset()
    checked = 0
    while not isinstance(out, Terminal):
        py = ppo_policy.evaluate(env.rm, impl="python")
        rs = ppo_policy.evaluate(env.rm, impl="rust")
        assert (py is None) == (rs is None)
        if py is not None and py.probs is not None and rs is not None:
            assert rs.probs is not None
            assert rs.surface == py.surface
            assert [a["action"] for a in rs.actions] == [a["action"] for a in py.actions]
            assert max(abs(a - b) for a, b in zip(rs.probs, py.probs)) < 1e-3
            if rs.chosen != py.chosen:
                # Same near-exact-tie carve-out as test_gate_c_rust_agent_matches_python_greedy
                # above: per-option probs already agree above, so a chosen-index
                # mismatch can only be torch/tract resolving a genuine tie
                # differently — require the two candidates' probs to actually be
                # a near-tie (~1e-7 gap observed by hand), 100x tighter than the
                # blanket 1e-3 tolerance so a real ranking regression still fails.
                assert abs(rs.probs[rs.chosen] - rs.probs[py.chosen]) < 1e-5, (
                    f"argmax drift @ seed {seed} step {out.step_index}: rust picked "
                    f"{rs.chosen} ({rs.probs[rs.chosen]!r}), python picked {py.chosen} "
                    f"({rs.probs[py.chosen]!r}) — not a near-tie"
                )
            checked += 1
        # Walk with the python argmax so both backends see one trajectory.
        probs, _ = pol.distribution(out.obs, out.actions, out.metrics)
        out = env.step(int(np.argmax(probs)))
    ppo_policy._reset_for_tests()
    assert checked > 0


# Seed 3 is `soak`: its run is twice as long as seed 0's (240 s, the second
# longest slow test), and seed 0 already walks the benchmark path per PR.
@pytest.mark.parametrize("seed", [0, pytest.param(3, marks=pytest.mark.soak)])
def test_ppo_brain_benchmark_run_completes(seed, tmp_path):
    """The workbench run-mode stepper with brain=ppo + Benchmark budget runs a
    PPO 'as trained' session to completion without error.

    This does not assert per-decision trajectory equality against a
    PpoRunEnv rollout on the same seed; that per-decision PPO-net equality is
    gated separately by ``test_workbench_rust_backend_matches_python``."""
    os.environ.setdefault("STS2_RUNAI_PPO", str(ONNX.resolve()))
    from workbench import branching, ppo_policy

    ppo_policy._reset_for_tests()
    # Reference trajectory: PpoRunEnv greedy over the torch net.
    env = PpoRunEnv(seed=seed, sims=1000, boss_sims=4000)
    pol = NetPolicy(CKPT)
    ref_picks: list[str] = []
    out = env.reset()
    while not isinstance(out, Terminal):
        probs, _ = pol.distribution(out.obs, out.actions, out.metrics)
        idx = int(np.argmax(probs))
        ref_picks.append(out.actions[idx]["action"])
        out = env.step(idx)

    # Workbench stepper: brain=ppo (rust ONNX), Benchmark budget.
    sess = branching.BranchingLiveSession(str(tmp_path), backend="rust")
    sess.start_run_mode(
        "Ironclad",
        seed,
        off_combat_brain="ppo",
        ppo_impl="rust",
        combat_sims=1000,
        combat_boss_sims=4000,
    )
    for _ in range(20_000):
        step = sess.run_mode_step()
        assert step["error"] is None
        if step["done"] or step["over"]:
            break
    ppo_policy._reset_for_tests()
    # The run reaches a terminal state without error under the PPO brain.
    assert step["over"] or step["done"]
    # Sanity: PpoRunEnv produced off-combat decisions to compare against.
    assert len(ref_picks) > 0
