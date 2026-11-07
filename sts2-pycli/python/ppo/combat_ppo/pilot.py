"""The deployed MCTS pilot on the PPO eval jobs: the reference line, paired by cell.

    python -m ppo.combat_ppo pilot --fights dev_boss.jsonl --build-reps 4 \
        --out RUN/evals/pilot.dev_boss.json

Same jobs as `eval` (`__main__.eval_jobs`: the build seed of every fight and
build rep, the job seed `1_000_003·i + 17`), so each result pairs with the PPO
eval's cell of the same `(fight, build_rep)`. The fight is built with
`rollout_starts.build_fight`, exactly as `play.py` builds it, and then played
by `sts2_rs.MctsCombatAgent(seed=job seed mod 2^31, sims, boss_sims)` to the end. The
workers run with every engine seam scrubbed (`combat_bench.agents.apply_env({})`),
so the agent is the search this wheel ships (on Ironclad since GH #830: the
forms + combo knobs and the BC move prior in elite and boss fights).

A fight that hits `max_steps` actions is a loss (`capped`), as in `play.py`.
"""

from __future__ import annotations

import time
from multiprocessing import get_context
from typing import Any

MAX_STEPS = 3000


def _init() -> None:
    from ppo.combat_bench import agents

    agents.apply_env({})


def play_job(job: dict[str, Any], sims: int, boss_sims: int) -> dict[str, Any]:
    import sts2_rs

    from ppo import rollout_starts

    t0 = time.process_time()
    rm = rollout_starts.build_fight(job["row"], job["build_seed"])
    # The agent takes an i32 seed; job seeds pass 2**31 beyond ~2,147 jobs.
    agent = sts2_rs.MctsCombatAgent(
        seed=int(job["seed"]) % (1 << 31), sims=sims, boss_sims=boss_sims
    )
    hp_before = int(job["row"]["hp_before"])
    steps = 0
    end = None
    try:
        while rm.phase() == "combat" and not rm.is_over():
            if steps >= MAX_STEPS:
                end = "capped"
                break
            if agent.step(rm) is None:
                end = "no_action"
                break
            steps += 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a PyO3 panic is a BaseException
        end = f"error: {e!r}"
    if end is None:
        end = "died" if rm.is_over() and not rm.player_won else "won"
    won = end == "won"
    return {
        "i": job["i"],
        "fight": job["fight"],
        "file": job["row"]["file"],
        "key": f"{job['row']['file']}:{job['row']['floor']}",
        "rep": job["rep"],
        "build_rep": job["rep"],
        "won": won,
        "end": end,
        "capped": end == "capped",
        "hp_before": hp_before,
        "hp_after": int(rm.current_hp) if won else 0,
        "steps": steps,
        "cpu_s": round(time.process_time() - t0, 3),
    }


def _run(args: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    return play_job(*args)


def play_many(
    jobs: list[dict[str, Any]], sims: int, boss_sims: int, procs: int
) -> list[dict[str, Any]]:
    ctx = get_context("spawn")
    with ctx.Pool(procs, initializer=_init) as pool:
        out = list(pool.imap_unordered(_run, [(j, sims, boss_sims) for j in jobs], chunksize=1))
    return sorted(out, key=lambda r: r["i"])
