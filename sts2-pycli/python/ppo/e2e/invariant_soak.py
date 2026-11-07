"""F9 soak: play random-action v0.111.0 A10 Ironclad runs and check every step.

The live F9 gate samples 1% of the e2e env's steps. This soak checks every
decision of many random-action runs before the week starts, so a check that
can fire on a legal state shows up here, not as a HUMAN call mid-campaign.
Every other run pins the player's HP out of combat (``set_player``) so that
the walk reaches acts 2 and 3. It also times the two check tiers.

Run it from ``sts2-pycli/``::

    PYTHONPATH=python python -m ppo.e2e.invariant_soak --runs 400 --workers 16 \\
        --out ~/runs/e2e/fidelity/f9_soak.json
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import random
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ppo.e2e.invariants import EXPECTED_GAME_VERSION, InvariantChecker

#: The pinned HP for the deep-walk runs.
PINNED_HP = 100_000


def random_run(
    seed: int,
    *,
    pin_hp: bool,
    max_steps: int = 6000,
    play_bias: float = 0.85,
    game_version: str = EXPECTED_GAME_VERSION,
) -> Iterator[Any]:
    """Yield the run's ``RunManager`` at every decision of a random-action run.

    The run is the e2e live configuration (A10, Neow start, rolled act 1,
    agent mode, interactive potion swap). At each decision it plays a random
    card with probability ``play_bias`` when one is legal, else a random legal
    action. The final (over or step-capped) state is yielded too.
    """
    import sts2_rs

    rm = sts2_rs.RunManager(
        seed,
        character="Ironclad",
        ascension_level=10,
        neow_start=True,
        act1="roll",
        game_version=game_version,
        mode="agent",
        interactive_potion_swap=True,
    )
    rng = random.Random(seed)
    for _ in range(max_steps):
        if rm.is_over():
            break
        phase = rm.phase()
        if phase == "event":
            rm.materialize_event()
        if pin_hp and phase != "combat":
            rm.set_player(hp=PINNED_HP, max_hp=PINNED_HP)
        yield rm
        actions = rm.available_actions()
        if not actions:
            break
        plays = [a for a in actions if a.get("action") == "play_card"]
        if plays and rng.random() < play_bias:
            rm.take_action(rng.choice(plays))
        else:
            rm.take_action(rng.choice(actions))
    yield rm


def _quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean_us": round(statistics.fmean(xs) * 1e6, 2),
        "p50_us": round(xs[len(xs) // 2] * 1e6, 2),
        "p99_us": round(xs[min(len(xs) - 1, int(len(xs) * 0.99))] * 1e6, 2),
    }


def _thin(xs: list[float], keep: int = 500) -> list[float]:
    """At most about ``keep`` evenly strided samples (the parent pools them)."""
    return xs[:: max(1, len(xs) // keep)]


def soak_one(job: tuple[int, bool, int]) -> dict[str, Any]:
    """Check every decision of one run. Returns its counters and timings."""
    seed, pin_hp, max_steps = job
    full = InvariantChecker(ledger_every=1)
    fast = InvariantChecker(ledger_every=0)
    t_fast_combat: list[float] = []
    t_fast_other: list[float] = []
    t_full_combat: list[float] = []
    examples: dict[str, str] = {}
    steps = combat_steps = 0
    max_act = max_floor = 0
    rm = None
    for rm in random_run(seed, pin_hp=pin_hp, max_steps=max_steps):
        steps += 1
        in_combat = rm.phase() == "combat" and not rm.is_over()
        combat_steps += in_combat
        t0 = time.perf_counter()
        fast.check(rm)
        t1 = time.perf_counter()
        found = full.check(rm, run_key=seed)
        t2 = time.perf_counter()
        (t_fast_combat if in_combat else t_fast_other).append(t1 - t0)
        if in_combat:
            t_full_combat.append(t2 - t1)
        for v in found:
            examples.setdefault(v.code, f"seed {seed} floor {rm.total_floor()}: {v.detail}")
        max_act = max(max_act, rm.act_index)
        max_floor = max(max_floor, rm.total_floor())
    return {
        "seed": seed,
        "pin_hp": pin_hp,
        "steps": steps,
        "combat_steps": combat_steps,
        "max_act": max_act,
        "max_floor": max_floor,
        "won": bool(rm is not None and rm.is_over() and rm.player_won),
        "stats": full.stats(),
        "examples": examples,
        "t_fast_combat": _thin(t_fast_combat),
        "t_fast_other": _thin(t_fast_other),
        "t_full_combat": _thin(t_full_combat),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=1_400_000_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--out", type=Path, required=True, help="summary JSON path")
    args = ap.parse_args(argv)

    jobs = [(args.seed_start + i, i % 2 == 0, args.max_steps) for i in range(args.runs)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_run = args.out.with_suffix(".runs.jsonl")
    totals: collections.Counter[str] = collections.Counter()
    by_code: collections.Counter[str] = collections.Counter()
    skips: collections.Counter[str] = collections.Counter()
    tolerated: collections.Counter[str] = collections.Counter()
    examples: dict[str, str] = {}
    acts: collections.Counter[int] = collections.Counter()
    timings: dict[str, list[float]] = {"fast_combat": [], "fast_other": [], "full_combat": []}
    t_start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool, per_run.open("w", encoding="utf-8") as fh:
        for done, row in enumerate(pool.imap_unordered(soak_one, jobs), 1):
            timings["fast_combat"] += row.pop("t_fast_combat")
            timings["fast_other"] += row.pop("t_fast_other")
            timings["full_combat"] += row.pop("t_full_combat")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            st = row["stats"]
            totals["runs"] += 1
            totals["steps"] += row["steps"]
            totals["combat_steps"] += row["combat_steps"]
            totals["checks"] += st["calls"]
            totals["ledger_checks"] += st["ledger_calls"]
            totals["violations"] += st["violations"]
            totals["wins"] += row["won"]
            by_code.update(st["by_code"])
            skips.update(st["ledger_skips"])
            tolerated.update(st.get("tolerated", {}))
            acts[row["max_act"]] += 1
            for code, ex in row["examples"].items():
                examples.setdefault(code, ex)
            if done % 20 == 0 or done == len(jobs):
                print(
                    f"[{time.time() - t_start:7.1f}s] {done}/{len(jobs)} runs, "
                    f"{totals['steps']} steps, {totals['violations']} violations",
                    flush=True,
                )
    summary = {
        "game_version": EXPECTED_GAME_VERSION,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "totals": dict(totals),
        "by_code": dict(by_code),
        "ledger_skips": dict(skips),
        "tolerated": dict(tolerated),
        "max_act_reached": {str(k): v for k, v in sorted(acts.items())},
        "examples": examples,
        "timing": {k: _quantiles(v) for k, v in timings.items()},
        "wall_s": round(time.time() - t_start, 1),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if totals["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
