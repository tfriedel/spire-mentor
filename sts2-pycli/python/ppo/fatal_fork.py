"""Fatal-fight fork diagnosis for PPO run-policy losses (GH #777 part 2).

Plays full runs with a PPO checkpoint (`net_greedy`, deployed eval-cheap combat)
and clones the `RunManager` at the entry of every fight. When a run is lost in
a fight, that fight's entry snapshot is re-fought `--reps` times under each arm:

  * `deployed` — honest BeliefDraw, the run's own budget (1000 / 4000 boss).
    Re-samples only the search's own randomness: the fraction survived here is
    how much of the loss was search-noise bad luck.
  * `strong`   — honest BeliefDraw at 5000 / 15000 boss sims (the eval-strong
    sims; the leaf stays TexelWide — `MctsCombatAgent` exposes no leaf knob).
  * `clair`    — clairvoyant search (`STS2_RUNAI_HONEST=0`, held for the whole
    fight because the policy reads it per search) at 1000 / 4000: the
    search sees the true draw order and enemy rolls. An upper bound on what
    better piloting of THIS deck from THIS entry state could do.
  * `heal` / `heal_clair` — the same two searches, but the fight is entered
    at full HP. Separates "this deck cannot beat this fight" from "HP lost in
    earlier fights (combat attrition, pathing) made it unwinnable".

The clone fixes the game's own RNG (shuffles, enemy rolls) at fight entry, so
reps differ only in the search seed; under `clair` every rep sees the same
true future. `RunManager` does not pickle, so forks run in the worker that
played the run, right after it ends.

A loss the clairvoyant arm cannot flip is a state problem; one only the heal
arms flip is HP attrition carried in; one even `heal_clair` cannot flip is
the deck (plus relics/potions) itself. One that `strong` flips is a
combat-budget problem.

    PYTHONPATH=python python -m ppo.fatal_fork --ckpt data/ppo/<champion>.pt \
        --ascension 10 --seeds 200 --act1 roll --reps 4 --out fatal.a10.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from typing import Any

FORK_STEP_CAP = 3000
ARMS: dict[str, tuple[int, int, bool, bool]] = {
    # name: (sims, boss_sims, clairvoyant, enter at full HP)
    "deployed": (1000, 4000, False, False),
    "strong": (5000, 15000, False, False),
    "clair": (1000, 4000, True, False),
    "heal": (1000, 4000, False, True),
    "heal_clair": (1000, 4000, True, True),
}


class _SnapshotAgent:
    """Delegating proxy over the env's combat agent that clones the run at
    the first combat step of every fight."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._key: tuple[int, int] | None = None
        self.snapshots: list[dict[str, Any]] = []

    def step(self, rm: Any) -> Any:
        if rm.phase() == "combat":
            key = (int(rm.act_index), int(rm.total_floor()))
            if key != self._key:
                self._key = key
                room = rm.current_room() or {}
                self.snapshots.append(
                    {
                        "rm": rm.clone(),
                        "act_index": key[0],
                        "total_floor": key[1],
                        "room_type": room.get("room_type"),
                        "encounter": room.get("encounter_id") or room.get("encounter"),
                        "entry_hp": int(rm.current_hp),
                        "max_hp": int(rm.max_hp),
                        "deck_size": len(rm.master_deck_detailed()),
                    }
                )
        else:
            self._key = None
        return self._inner.step(rm)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def fight(
    snapshot_rm: Any, sims: int, boss_sims: int, clair: bool, heal: bool, seed: int
) -> dict[str, Any]:
    """Re-fight one entry snapshot to the end of the combat."""
    import sts2_rs

    # `MctsCombatPolicy` reads STS2_RUNAI_HONEST on EVERY search (`policy/mod.rs`
    # `resolve_hidden_info`), not at construction, so the override has to stay
    # set for the whole fight. Workers are single-threaded, so this is safe.
    prev = os.environ.get("STS2_RUNAI_HONEST")
    if clair:
        os.environ["STS2_RUNAI_HONEST"] = "0"
    try:
        agent = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=boss_sims)
        rm = snapshot_rm.clone()
        if heal:
            # Patches run_state AND the live combat creature (`debug_set_player`).
            rm.set_player(hp=int(rm.max_hp))
        steps = 0
        while rm.phase() == "combat" and not rm.is_over() and steps < FORK_STEP_CAP:
            if agent.step(rm) is None:
                break
            steps += 1
    finally:
        if clair:
            if prev is None:
                os.environ.pop("STS2_RUNAI_HONEST", None)
            else:
                os.environ["STS2_RUNAI_HONEST"] = prev
    hp = int(rm.current_hp)
    survived = bool(rm.player_won) or (not rm.is_over() and hp > 0)
    return {"survived": survived, "end_hp": hp, "steps": steps}


def play_and_fork(job: tuple[Any, ...]) -> dict[str, Any]:
    seed, ckpt, ascension, character, act1, reps, arms = job
    import numpy as np

    from ppo.env import Decision, PpoRunEnv
    from ppo.workers import _net_policy

    t0 = time.time()
    try:
        kwargs: dict[str, Any] = {"act1": act1}
        env = PpoRunEnv(seed=seed, ascension=ascension, character=character, **kwargs)
        proxy = _SnapshotAgent(env.agent)
        env.agent = proxy
        pol = _net_policy(ckpt)
        nrng = np.random.default_rng(seed)
        out = env.reset()
        while isinstance(out, Decision):
            idx, _, _, _ = pol.sample(out.obs, out.actions, out.metrics, nrng, greedy=True)
            out = env.step(idx)
        terminal = out
        row: dict[str, Any] = {
            "seed": seed,
            "ascension": ascension,
            "act1": env.rm.act1,
            "outcome": terminal.outcome,
            "won": terminal.won,
            "final": terminal.metrics,
            "n_fights": len(proxy.snapshots),
        }
        fatal = proxy.snapshots[-1] if proxy.snapshots else None
        died_in_fight = (
            not terminal.won
            and terminal.outcome == "lost"
            and fatal is not None
            and int(env.rm.current_hp) <= 0
            and fatal["total_floor"] == int(env.rm.total_floor())
        )
        row["died_in_fight"] = died_in_fight
        if died_in_fight:
            assert fatal is not None
            row["fatal"] = {k: v for k, v in fatal.items() if k != "rm"}
            forks: dict[str, list[dict[str, Any]]] = {}
            for arm in arms:
                sims, boss_sims, clair, heal = ARMS[arm]
                forks[arm] = [
                    fight(fatal["rm"], sims, boss_sims, clair, heal, seed * 1000 + 17 * rep + 1)
                    for rep in range(reps)
                ]
            row["forks"] = forks
        row["wall_s"] = round(time.time() - t0, 1)
        return row
    except Exception as e:
        return {"seed": seed, "ascension": ascension, "outcome": "error", "error": repr(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--act1", default="roll", choices=("overgrowth", "underdocks", "roll"))
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    done: set[int] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {r["seed"] for r in map(json.loads, f) if r.get("outcome") != "error"}
    jobs = [
        (s, args.ckpt, args.ascension, args.character, args.act1, args.reps, args.arms)
        for s in range(args.seed_start, args.seed_start + args.seeds)
        if s not in done
    ]
    t0 = time.time()
    n = lost = 0
    flips = dict.fromkeys(args.arms, 0)
    with (
        get_context("spawn").Pool(args.procs) as pool,
        open(args.out, "a", encoding="utf-8") as sink,
    ):
        for row in pool.imap_unordered(play_and_fork, jobs):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            n += 1
            if row.get("died_in_fight"):
                lost += 1
                for arm in args.arms:
                    flips[arm] += any(f["survived"] for f in row["forks"][arm])
            print(
                f"[fatal_fork] A{args.ascension} {n}/{len(jobs)} runs, {lost} fight deaths, "
                f"any-rep flips {flips}, {time.time() - t0:.0f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
