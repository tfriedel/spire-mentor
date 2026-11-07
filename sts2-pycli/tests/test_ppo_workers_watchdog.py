"""Stall-watchdog gate for `ppo.workers.play_batch`.

A single hung run (native combat can loop in a way Python signals can't
interrupt) must NOT stall the whole collection batch: the watchdog abandons the
straggler as outcome='timeout' after `run_timeout_s` of no completion, keeps the
runs that finished, and preserves submission order for the learner.

Uses a fork context + a monkeypatched `play_one` so the hang is simulated
without the engine (see `play_batch(mp_context=...)`).
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo import workers


def _fake_play_one(seed, policy="random", **kwargs):
    """Fast for every seed except the poisoned one, which hangs forever."""
    if seed == 1:  # the "infinite combat loop" seed
        while True:
            time.sleep(0.05)
    return {
        "seed": seed,
        "policy": policy,
        "outcome": "won",
        "won": True,
        "steps": 1,
        "n_decisions": 0,
        "final": {},
        "wall_s": 0.0,
        "resolution_counts": {},
        "decisions": [],
        "transitions": [],
    }


class _FakePanic(BaseException):
    """Stand-in for `pyo3_runtime.PanicException` (also a `BaseException`)."""


def test_worker_books_engine_panic_as_error_row(monkeypatch):
    """GH #335: a Rust panic reaches Python as `pyo3_runtime.PanicException`,
    which derives from `BaseException` — an `except Exception` guard lets it
    through and the pool worker dies, leaving the `AsyncResult` never-ready
    until the stall watchdog fires ~10 min later. `_worker` must book it as an
    `outcome='error'` row instead."""

    def _boom(seed, **kwargs):
        raise _FakePanic("Shiv: Enemy-targeted")

    monkeypatch.setattr(workers, "play_one", _boom)
    # 15-tuple, positional order per `_worker`'s unpack: seed, policy,
    # ascension, character, record_decisions, ckpt_path, sims, boss_sims,
    # combat, surrogate_path, search, act1, sample_seed (PLR; None is the
    # default), curse_inject_p (GH #977; 0.0 is the default), combat_ckpt
    # (last; None is the default), so this test still only exercises
    # watchdog behaviour.
    row = workers._worker(
        (
            7,
            "random",
            0,
            "Silent",
            False,
            None,
            None,
            None,
            "mcts",
            None,
            None,
            "overgrowth",
            None,
            0.0,
            None,
        )
    )
    assert row["seed"] == 7
    assert row["outcome"] == "error"
    assert row["won"] is False
    assert "Shiv" in row["error"]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_worker_still_propagates_interrupts(monkeypatch, exc):
    """Ctrl-C / a pool shutdown must NOT be swallowed into an error row."""

    def _raise(seed, **kwargs):
        raise exc()

    monkeypatch.setattr(workers, "play_one", _raise)
    with pytest.raises(exc):
        workers._worker(
            (
                7,
                "random",
                0,
                "Silent",
                False,
                None,
                None,
                None,
                "mcts",
                None,
                None,
                "overgrowth",
                None,
                0.0,
                None,
            )
        )


@pytest.mark.skipif(sys.platform == "win32", reason="fork context unavailable on Windows")
def test_watchdog_abandons_hung_run_and_keeps_the_rest(monkeypatch):
    monkeypatch.setattr(workers, "play_one", _fake_play_one)
    seeds = [0, 1, 2, 3]  # seed 1 hangs
    t0 = time.time()
    rows = workers.play_batch(
        seeds,
        policy="random",
        procs=4,
        run_timeout_s=2.0,  # short stall window for the test
        game_timeout_s=1.0,  # and a short per-game budget
        mp_context="fork",
    )
    elapsed = time.time() - t0

    # Returns one row per seed, in submission order (load-bearing for the
    # trainer's row-by-row transition concatenation).
    assert [r["seed"] for r in rows] == seeds
    by_seed = {r["seed"]: r for r in rows}
    assert by_seed[1]["outcome"] == "timeout"
    assert by_seed[1]["won"] is False
    assert by_seed[1]["age_s"] >= 2.0
    for s in (0, 2, 3):
        assert by_seed[s]["outcome"] == "won"

    # Self-heals promptly: watchdog window (2s) + a little slack, NOT the old
    # ~run_timeout_s * batches formula.
    assert elapsed < 15.0, f"watchdog took {elapsed:.1f}s — too slow"


@pytest.mark.skipif(sys.platform == "win32", reason="fork context unavailable on Windows")
def test_healthy_batch_never_trips_watchdog(monkeypatch):
    """A batch where every run completes must return all real rows even with a
    tiny watchdog window — completions keep resetting the clock."""
    monkeypatch.setattr(workers, "play_one", _fake_play_one)
    seeds = [0, 2, 3, 4, 5]  # none hang
    rows = workers.play_batch(seeds, policy="random", procs=2, run_timeout_s=1.0, mp_context="fork")
    assert [r["seed"] for r in rows] == seeds
    assert all(r["outcome"] == "won" for r in rows)


def _slow_play_one(seed, policy="random", **kwargs):
    """Seed 1 is slow but finishes (a long boss fight inside one env.step)."""
    if seed == 1:
        time.sleep(4.0)
    return {**_fake_play_one(0, policy), "seed": seed}  # _fake_play_one hangs on seed 1


@pytest.mark.skipif(sys.platform == "win32", reason="fork context unavailable on Windows")
def test_slow_progressing_game_outliving_the_rest_is_not_dropped(monkeypatch):
    """GH #763: the old watchdog fired on 'no completion for run_timeout_s'
    alone, so a slow game that outlived every other one by that window was
    dropped. It must now run to its own `game_timeout_s` budget."""
    monkeypatch.setattr(workers, "play_one", _slow_play_one)
    seeds = [0, 1, 2, 3]
    rows = workers.play_batch(
        seeds, policy="random", procs=4, run_timeout_s=1.0, game_timeout_s=30.0, mp_context="fork"
    )
    assert [r["seed"] for r in rows] == seeds
    assert all(r["outcome"] == "won" for r in rows)
