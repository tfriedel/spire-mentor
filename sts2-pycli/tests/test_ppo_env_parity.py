"""G1 parity gate: PpoRunEnv expert mode must match run_arena --policy heuristic.

The env's expert mode wraps the same HeuristicRunAgent
(MctsCombatPolicy<HeuristicPolicy>) that `run_arena --policy heuristic`
drives, but routes non-combat multi-option decisions through the env's
yield/observe path (including event materialization in `observe()`). Any
divergence from run_arena per-seed outcomes means the env's observation
path perturbed the run — a real bug; do not weaken this test to make it
pass.

Plan: sts2-rs docs/plans/2026-07-03-ppo-fullrun-plan.md M-P1, gate G1.
"""

import functools
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from sts2_client import LEGACY_GAME_VERSION

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARENA = os.path.join(ROOT, "sts2-rs", "target", "release", "run_arena")
PY = os.path.join(ROOT, "sts2-pycli", "python")
N = int(os.environ.get("PARITY_SEEDS", "20"))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("STS2_RUNAI_VALUE", raising=False)
    monkeypatch.syspath_prepend(PY)


def _env_expert_outcome(seed: int, act1: str = "overgrowth") -> dict:
    from ppo.env import PpoRunEnv, Terminal

    # `run_arena` has no build switch: it simulates the engine's own default,
    # which is the legacy build. Pin the env to it, or the selected build
    # (v0.111.0 by default) turns every cell into version skew.
    env = PpoRunEnv(seed=seed, expert=True, act1=act1, game_version=LEGACY_GAME_VERSION)
    terminal = None
    for out in env.expert_run():
        if isinstance(out, Terminal):
            terminal = out
    assert terminal is not None
    # Ownership audit (G1b, per-seed slice): every logged resolution class
    # must be one of the three expected ones, and every yielded decision
    # must have a known surface.
    for _, phase, surface, klass in env.resolution_log:
        assert klass in ("combat-agent", "single-option-auto", "expert", "potion-swap-rule"), (
            f"unexpected resolution class {klass} at phase {phase}"
        )
        if klass == "expert":
            # "potion-swap" joined this list with GH #379, which turned
            # `interactive_potion_swap` on for both arms — a full-belt potion
            # reward is now a yielded decision rather than a silent drop.
            assert surface in (
                "path",
                "card-pick",
                "event",
                "shop",
                "rest",
                "treasure",
                "potion-swap",
            ), f"yielded decision with unclassified surface: phase={phase}"
    return {
        "seed": seed,
        "won": bool(env.rm.player_won),
        "final_hp": int(env.rm.current_hp),
        "act_index": int(env.rm.act_index),
    }


# Cross-worker arena cache: the first xdist worker to take the flock runs
# the (deterministic) sweep; the rest block on the lock then read the cache.
# The cache path is keyed on a content hash of everything that determines the
# sweep output — the run_arena binary, the seed count and the act1 token
# (GH #545: overgrowth/underdocks/roll walk different content, so each token
# needs its own cache row) — so a rebuilt engine / different N / different
# act1 invalidates it automatically instead of a stale /tmp file producing
# false parity failures.
def _arena_cache_path(act1: str) -> str:
    h = hashlib.sha256()
    with open(ARENA, "rb") as fh:
        h.update(fh.read())
    h.update(str(N).encode())
    h.update(act1.encode())
    return f"/tmp/ppo-parity-arena.{h.hexdigest()[:16]}.jsonl"


#: What builds `run_arena`. `_ensure_current_arena` runs it before every
#: session's sweeps; it is also named in the failure messages below.
REBUILD_CMD = "cargo build --release --bin run_arena   # from sts2-rs/"

RS_ROOT = os.path.join(ROOT, "sts2-rs")
_BUILD_MUTEX = os.path.join(RS_ROOT, "scripts", "build-mutex.sh")


@functools.cache
def _arena_build_problem() -> tuple[str, str] | None:
    """Build `run_arena` so the sweep always answers for THIS tree.

    `run_arena` is a build artifact the parity gate shells out to, and a binary
    left over from an older tree answers with that tree's engine. That rotted
    twice: on 2026-08-26 a binary that predated `--act1` broke all 60 cases as
    a bare `CalledProcessError`, and a later mtime guard ("is any `.rs` newer
    than the binary?") then failed all 60 for a change to
    `sts2-workbench/src/assets.rs`, a crate `run_arena` does not even link.

    So the test builds it instead of judging it: cargo's fingerprints know the
    binary's real dependency graph, rebuild exactly when an input changed, and
    cost a second or two when nothing did. The build goes through
    `build-mutex.sh`, which serialises it with the wheel builds sharing
    `target/`; xdist workers each call this, and cargo's own target-dir lock
    makes all but the first a no-op. Cached per process.

    An ABSENT binary still skips, as it always has: building is opt-in (one
    `cargo build` here), and the public export's CI runs this file with
    ``-m "not oracle"``, which replaces the default ``-m 'not slow'``, on a
    runner that has cargo. Building there would add a release compile and 60
    full runs to every push. Once a binary exists it is kept current.

    Returns:
        None when the binary is current, else ("skip" | "fail", message):
        skip when there is no binary or no cargo; a failed build fails.
    """
    if not os.path.exists(ARENA):
        return ("skip", f"run_arena release binary not built (opt-in):\n    {REBUILD_CMD}")
    if shutil.which("cargo") is None:
        return (
            "skip",
            f"cargo not on PATH, so run_arena cannot be built/checked:\n    {REBUILD_CMD}",
        )
    argv = ["cargo", "build", "--release", "--bin", "run_arena"]
    if os.path.exists(_BUILD_MUTEX):
        argv = [_BUILD_MUTEX, *argv]
    proc = subprocess.run(argv, cwd=RS_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
        return ("fail", f"building run_arena failed (exit {proc.returncode}):\n{tail}")
    return None


def _require_current_arena() -> None:
    """Build `run_arena` if its inputs changed (see `_arena_build_problem`)."""
    problem = _arena_build_problem()
    if problem is None:
        return
    kind, message = problem
    if kind == "skip":
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


def _arena_rows(act1: str) -> dict:
    import fcntl

    cache = _arena_cache_path(act1)
    with open(f"{cache}.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(cache):
            tmp = f"{cache}.{os.getpid()}"
            argv = [
                ARENA,
                "sweep",
                "--seeds",
                str(N),
                "--seed-start",
                "0",
                "--policy",
                "heuristic",
                "--combat",
                "eval-cheap",
                "--jobs",
                "0",
                "--act1",
                act1,
                "--out",
                tmp,
            ]
            # CAPTURED, not inherited: `check=True` on an inheriting call
            # raises a `CalledProcessError` whose repr is just the argv and an
            # exit code, while the one line that explains it ("unexpected
            # argument '--act1' found") has already gone to the terminal —
            # and under xdist, into a worker's discarded stream. Surface it in
            # the failure itself, with the rebuild that fixes the likeliest
            # cause. `_require_current_arena` rebuilds the binary first, so
            # this should now only fire on a genuine engine/CLI problem.
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise AssertionError(
                    f"run_arena sweep failed (exit {proc.returncode}); the binary was "
                    f"just built from this tree, so this is an engine/CLI problem.\n"
                    f"argv: {' '.join(argv)}\n"
                    f"stderr:\n{proc.stderr.strip()}\n"
                    f"stdout:\n{proc.stdout.strip()}"
                )
            pathlib.Path(tmp).replace(cache)
    with open(cache, encoding="utf-8") as fh:
        return {r["seed"]: r for r in (json.loads(line) for line in fh)}


@pytest.mark.slow
@pytest.mark.timeout(0)  # full runs — disable the global cap
@pytest.mark.parametrize("act1", ["overgrowth", "underdocks", "roll"])
@pytest.mark.parametrize("seed", range(N))
def test_ppo_env_expert_matches_run_arena(seed, act1):
    """Per-seed so xdist parallelizes the 20 env runs (was one 27-min test).

    Parameterised over the GH #545 act1 token so G1 (env vs `arena::run_cell`
    parity) covers Underdocks and the seed-driven roll, not just the historical
    Overgrowth-only default — an arena arm and a training env that disagree
    about the act is the silent-divergence class this gate exists to catch.
    """
    _require_current_arena()
    ar = _arena_rows(act1)[seed]
    a = _env_expert_outcome(seed, act1)
    got = (a["won"], a["final_hp"], a["act_index"])
    want = (bool(ar["won"]), int(ar["final_hp"]), int(ar["act_index"]))
    assert got == want, f"PpoRunEnv expert != run_arena on seed {seed} act1={act1}: {got} vs {want}"


def test_classify_surface_unit():
    """Pure-python classifier unit checks (no engine)."""
    sys.path.insert(0, PY)
    from ppo.env import classify_surface

    assert classify_surface("map_choice", [{"action": "move"}] * 2) == "path"
    assert classify_surface("card_reward", [{"action": "pick_card_reward"}] * 3) == "card-pick"
    assert classify_surface("boss_relic", [{"action": "pick_boss_relic"}] * 3) == "card-pick"
    assert classify_surface("shop", [{"action": "shop_buy_card"}] * 4) == "shop"
    assert classify_surface("event", [{"action": "event_choice"}] * 2) == "event"
    assert classify_surface("rest_site", [{"action": "rest_option"}] * 2) == "rest"
    assert classify_surface("treasure", [{"action": "collect_treasure"}] * 2) == "treasure"
    # Smith target preempts the rest surface (ablation.rs contract).
    assert classify_surface("rest_site", [{"action": "smith_card"}] * 3) == "card-pick"
    # Phase-agnostic deck mutations are card picks.
    assert classify_surface("event", [{"action": "remove_deck_card"}] * 5) == "card-pick"
    # Mixed lists fall through to the phase.
    assert (
        classify_surface("event", [{"action": "remove_deck_card"}, {"action": "event_choice"}])
        == "event"
    )
    assert classify_surface("combat", [{"action": "end_turn"}]) is None


# ── potion-stack seams (plan 2026-08-20-ppo-potion-bootstrap P2) ──────────

_OC_TABLE = {
    "version": 1,
    "character": "ironclad",
    "units": "hp_equiv",
    "source": "manual",
    "default_oc": 8.0,
    "entries": [
        {"potion": "POTION.FIRE_POTION", "act": 1, "oc": 4.0},
        {"potion": "POTION.POTION_SHAPED_ROCK", "act": 1, "oc": 0.0},
    ],
}


def _set_potion_seams(monkeypatch, tmp_path):
    table = tmp_path / "oc.json"
    table.write_text(json.dumps(_OC_TABLE))
    monkeypatch.setenv("STS2_RUNAI_POTION_OC", str(table))
    monkeypatch.setenv("STS2_RUNAI_POTION_WEIGHT", "0")
    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")


def test_potion_swap_flag_has_one_source_of_truth(monkeypatch, tmp_path):
    """The env's RunManager takes `interactive_potion_swap` from the SAME
    resolver `arena.rs::new_arena_run_manager` reads — unset means off,
    seams-on means on, with no env.py-local reimplementation."""
    import sts2_rs
    from ppo.env import PpoRunEnv

    monkeypatch.delenv("STS2_RUNAI_POTION_SWAP", raising=False)
    monkeypatch.delenv("STS2_RUNAI_POTION_OC", raising=False)
    assert sts2_rs.resolve_potion_swap_flag() is False
    env = PpoRunEnv(seed=0)
    assert env.rm.interactive_potion_swap is False

    _set_potion_seams(monkeypatch, tmp_path)
    assert sts2_rs.resolve_potion_swap_flag() is True
    env = PpoRunEnv(seed=0)
    assert env.rm.interactive_potion_swap is True


def test_potion_swap_resolved_rule_side_never_yielded(monkeypatch, tmp_path):
    """A pending full-belt swap is resolved by the Rust OC rule inside the
    env loop — logged as `potion-swap-rule`, never yielded to the net and
    never counted as a decision (no encoder surface, GH #379)."""
    from ppo.env import PpoRunEnv, Terminal

    _set_potion_seams(monkeypatch, tmp_path)
    env = PpoRunEnv(seed=0)
    # Rock (OC 0) + Block (default 8) held; Fire (OC 4) offered — the
    # rule discards the worthless Rock (same fixture as the Rust test
    # `potion_swap_decision_is_resolved_by_the_oc_rule`).
    env.rm.debug_set_potion_state(["POTION_SHAPED_ROCK", "BLOCK_POTION"], pending=["FIRE_POTION"])
    out = env.reset()
    assert not isinstance(out, Terminal)
    assert env.rm.pending_potion_swap is None, "swap left pending"
    assert "FirePotion" in env.rm.potions(), "offer not claimed"
    assert "PotionShapedRock" not in env.rm.potions(), "argmin slot kept"
    classes = [k for *_, k in env.resolution_log]
    assert "potion-swap-rule" in classes
    # The swap never surfaced as a net decision: every yielded action set
    # so far is swap-free (the first Decision is the Neow event).
    assert all(a.get("action") != "potion_swap" for a in out.actions)
