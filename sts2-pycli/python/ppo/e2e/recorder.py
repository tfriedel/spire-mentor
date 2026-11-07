"""Record real v0.111.0 A10 decision states as seeds and action logs (WP3, X-C, X-T).

A `RunManager` cannot be pickled, and a `state_to_dict` round trip changes
every state hash, so a recorded state is its run's seed plus the index of
every action taken from the start (`rm.available_actions()` order), plus the
steps at which a multi-option decision was sampled. `replay()` rebuilds each
sampled state by applying the log again, and checks each one against a
fingerprint taken at record time.

The recording agent is the deployed search agent (`sts2_rs.PpoRunAgent`: the
Ironclad champion run net with MCTS combat), at a reduced search budget so a
run takes ~25 s; it reaches act 3 in most runs, so the states cover every act.
The champion resolves as the workbench's agent does (`champion()`): any
champion works, because a replay applies the action log and never runs the
agent. Each record names the net it was played with.

    PYTHONPATH=python python -m ppo.e2e.recorder record --out STATES.jsonl \\
        --seed-start 162000200 --runs 60 --p 0.4

Replay keeps each fight's `IntentLog` the way #1081's run slot does: it
observes the honest snapshot at EVERY combat step, forced ones included, so a
rebuilt state's intent history is the one the tokenizer would have seen live.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import factory

GAME_VERSION = factory.GAME_VERSION
# The campaign's construction (plan §3.1), as `factory.new_run` builds it;
# written into every recorded run so a log names the run it replays.
RUN_KWARGS: dict[str, Any] = {
    "character": factory.CHARACTER,
    "ascension_level": factory.ASCENSION,
    "neow_start": True,
    "act1": factory.ACT1,
    "game_version": factory.GAME_VERSION,
    "mode": factory.MODE,
    "interactive_potion_swap": True,
}
MAX_STEPS = 6000
ROOMS = ("monster", "elite", "boss")


def make_run(seed: int) -> Any:
    """A campaign run (`ppo.e2e.factory.new_run`, checked)."""
    return factory.new_run(int(seed))


def fingerprint(rm: Any) -> list[Any]:
    """A cheap identity of the decision point (checked on replay)."""
    fp: list[Any] = [
        rm.phase(),
        int(rm.total_floor()),
        int(rm.gold),
        int(rm.current_hp),
        len(rm.available_actions()),
    ]
    cs = rm.combat_state() if rm.phase() == "combat" else None
    if cs is not None:
        fp.append(int(cs.ordered_hash()) & 0xFFFF_FFFF_FFFF)
    return fp


def _pycli_root() -> Path:
    return Path(__file__).resolve().parents[3]


def champion() -> tuple[str, str | None]:
    """The recording agent's run net (an ONNX path) and the encoder features
    it was trained with, resolved as the workbench's agent resolves them
    (`workbench.ppo_policy`): the Ironclad champion `data/ppo/champions.toml`
    declares, with its declared features; or `STS2_WB_PPO_ONNX` when set, with
    features None (`PpoRunAgent` then reads `STS2_PPO_FEATURES`).

    Raises:
        FileNotFoundError: the manifest's net or the override's file is absent.
    """
    import champions
    from workbench import ppo_policy

    path = ppo_policy.onnx_path(factory.CHARACTER)
    if path is None:
        raise FileNotFoundError(
            f"no {factory.CHARACTER} run net: STS2_WB_PPO_ONNX names a missing file, "
            "or data/ppo/champions.toml declares no champion"
        )
    if os.environ.get("STS2_WB_PPO_ONNX"):
        return path, None
    champ = champions.get(factory.CHARACTER)
    return path, None if champ is None else champ.features


def _shown(path: str) -> str:
    """`path` relative to `sts2-pycli/` when it is inside it (a record's provenance)."""
    p = Path(path).resolve()
    root = _pycli_root()
    return p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)


def record_run(seed: int, p: float, sims: int, boss_sims: int) -> dict[str, Any]:
    """Play one run with the champion agent; sample decisions with probability `p`."""
    import sts2_rs

    rm = make_run(seed)
    onnx, features = champion()
    agent = sts2_rs.PpoRunAgent(
        seed=seed, sims=sims, boss_sims=boss_sims, model_path=onnx, features=features
    )
    rng = random.Random(f"e2e-recorder:{seed}")
    log: list[int] = []
    states: list[int] = []
    fps: list[list[Any]] = []
    t0 = time.time()
    end = "over"
    while not rm.is_over():
        if len(log) >= MAX_STEPS:
            end = "max_steps"
            break
        if rm.phase() == "event":
            rm.materialize_event()
        acts = rm.available_actions()
        if not acts:
            end = "no_actions"
            break
        if len(acts) > 1 and rng.random() < p:
            states.append(len(log))
            fps.append(fingerprint(rm))
        rec = agent.step(rm)
        if rec is None:
            end = "agent_halt"
            break
        log.append(acts.index(rec["action"]))
    return {
        "seed": seed,
        "kwargs": RUN_KWARGS,
        "agent": {
            "kind": "PpoRunAgent",
            "onnx": _shown(onnx),
            "features": features,
            "sims": sims,
            "boss_sims": boss_sims,
        },
        "log": log,
        "states": states,
        "fingerprints": fps,
        "end": end,
        "won": bool(rm.player_won),
        "act": int(rm.act_index),
        "floor": int(rm.total_floor()),
        "wall_s": round(time.time() - t0, 2),
    }


def _worker(args: tuple[list[int], float, int, int, str]) -> int:
    seeds, p, sims, boss_sims, out = args
    n = 0
    with gzip.open(out, "at", encoding="utf-8") as f:
        for s in seeds:
            row = record_run(s, p, sims, boss_sims)
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += len(row["states"])
            print(
                f"seed {s} states {len(row['states'])} act {row['act']} floor {row['floor']} "
                f"won {row['won']} {row['wall_s']}s",
                flush=True,
            )
    return n


def read_runs(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every recorded run of `path` (gzip JSONL; several shard files may be joined)."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


@dataclass
class State:
    """A rebuilt decision point. `rm` is live: use it before advancing replay."""

    seed: int
    step: int
    rm: Any
    phase: str
    actions: list[dict[str, Any]]
    # Combat only: the state, its fight's room, encounter and the honest
    # snapshot with the fight's intent history (`IntentLog`).
    cs: Any = None
    room: str | None = None
    snap: dict[str, Any] | None = None
    bv: Any = None
    fight: dict[str, Any] = field(default_factory=dict)


def replay(run: dict[str, Any], check: bool = True, tokenizer: Any = None) -> Iterator[State]:
    """Rebuild `run`'s sampled states in order (each valid until the next yield).

    A `tokenizer` (`contracts.Tokenizer`) is driven as the env drives it:
    `reset_fight()` at every fight start and `observe(policy)` at every combat
    step, forced ones included, before the action; a yielded combat state's
    `bv` is the one it observed."""
    from ppo.combat_ctx.intent_log import IntentLog
    from ppo.combat_ctx.snapshot import honest_snapshot, obs_actions

    from .env import PolicySource

    rm = make_run(int(run["seed"]))
    want = dict(
        zip(run["states"], run.get("fingerprints") or [None] * len(run["states"]), strict=True)
    )
    ilog: IntentLog | None = None
    fight: dict[str, Any] = {}
    last_round = 0
    in_combat = False
    for step, idx in enumerate([*run["log"], None]):
        if rm.is_over():
            break
        phase = rm.phase()
        if phase == "event":
            rm.materialize_event()
        snap = bv = cs = None
        if phase == "combat":
            cs = rm.combat_state()
            room = rm.current_room().get("room_type")
            room = room if room in ROOMS else "monster"
            if not in_combat or ilog is None:
                ilog, fight, last_round = IntentLog(), {"room": room, "n": fight.get("n", 0) + 1}, 0
                if tokenizer is not None:
                    tokenizer.reset_fight()
            bv = cs.belief_view(0)
            snap = honest_snapshot(bv, fight["room"])
            if int(snap["round"]) < last_round:
                # A new fight in the same combat phase (back-to-back bosses).
                ilog, fight = IntentLog(), {"room": room, "n": fight.get("n", 0) + 1}
                if tokenizer is not None:
                    tokenizer.reset_fight()
            last_round = int(snap["round"])
            snap = ilog.observe(snap)
            if tokenizer is not None:
                legal = cs.legal_actions()
                acts = obs_actions(legal, cs.pending_choice(), bv.pending_choice())
                tokenizer.observe(PolicySource(bv, acts, rm))
            in_combat = True
        else:
            in_combat = False
            ilog = None
        if step in want:
            if check and want[step] is not None and fingerprint(rm) != want[step]:
                raise RuntimeError(f"seed {run['seed']} step {step}: replay diverged")
            yield State(
                seed=int(run["seed"]),
                step=step,
                rm=rm,
                phase=phase,
                actions=rm.available_actions(),
                cs=cs,
                room=fight.get("room") if cs is not None else None,
                snap=snap,
                bv=bv,
                fight=dict(fight),
            )
        if idx is None:
            break
        rm.take_action(rm.available_actions()[idx])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--seed-start", type=int, default=162_000_200)
    r.add_argument("--runs", type=int, default=60)
    r.add_argument("--p", type=float, default=0.4)
    r.add_argument("--sims", type=int, default=300)
    r.add_argument("--boss-sims", type=int, default=1200)
    r.add_argument("--procs", type=int, default=1)
    c = sub.add_parser("check")
    c.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        seeds = list(range(a.seed_start, a.seed_start + a.runs))
        if a.procs <= 1:
            n = _worker((seeds, a.p, a.sims, a.boss_sims, a.out))
        else:
            from multiprocessing import get_context

            shards = [seeds[i :: a.procs] for i in range(a.procs)]
            outs = [f"{a.out}.{i}" for i in range(a.procs)]
            with get_context("spawn").Pool(a.procs) as pool:
                n = sum(
                    pool.map(
                        _worker,
                        [
                            (s, a.p, a.sims, a.boss_sims, o)
                            for s, o in zip(shards, outs, strict=True)
                        ],
                    )
                )
        print(f"recorded {n} states", flush=True)
        return 0
    runs = read_runs(a.path)
    n = 0
    t0 = time.time()
    for run in runs:
        for _st in replay(run):
            n += 1
    print(f"{len(runs)} runs, {n} states replayed and matched in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
