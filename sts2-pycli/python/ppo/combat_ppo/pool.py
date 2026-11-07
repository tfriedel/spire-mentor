"""The training start pool: human-bank fights minus every held-out run.

Held out, never trained on (and asserted by run id, not by fight):

* every boss800 run, every run with a Waterfall Giant fight, every combat-bench
  MANIFEST run (`combat_ctx.gen.holdout`, the step-0 / ctx-RL holdout);
* fold 2 (`combat_ctx.gen.fold_of`), the `dev_boss` monitoring set's fold.

A start is a bank row; each episode builds it from a fresh build seed, so a row
is replayed under many shuffles and enemy rolls.

An optional second source, the champion's OWN fights (`own_pool`), holds the
fight entries of full runs played by the deployed run agent
(`combat_bench.own_decks harvest`, bench schema, `cohort = "own"`). Those runs
come from their own seed range, so no human holdout can leak through them. The
sampler draws a room by its weight, then the own pool with probability
`own_share` (when it has fights of that room), else the human pool.
"""

from __future__ import annotations

import json
import random
from typing import Any

from ppo.combat_ctx import gen

ROOMS = ("boss", "elite", "monster")
DEV_FOLD = 2
TRAIN_SEED_LO = 2**30


def held_out_runs(rows: list[dict[str, Any]], boss800: str) -> set[str]:
    """Every run the training pool must not contain (holdout union + fold 2)."""
    held = gen.holdout(rows, boss800)["all"]
    return held | {r["file"] for r in rows if gen.fold_of(r["file"]) == DEV_FOLD}


def train_pool(
    hbank: str, boss800: str, extra_holdouts: list[str] | None = None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """`{room: rows}` of trainable fights, and a metadata dict.

    Args:
        hbank: The human bank (`pilot_human_decks extract` rows).
        boss800: The boss800 file (its runs are held out).
        extra_holdouts: More fight files whose runs are held out (elite800, ...).
    """
    rows = gen.read_bank(hbank)
    held = held_out_runs(rows, boss800)
    for path in extra_holdouts or []:
        held |= {r["file"] for r in gen.read_bank(path)}
    pool: dict[str, list[dict[str, Any]]] = {room: [] for room in ROOMS}
    for r in rows:
        if r["file"] in held or r["room_type"] not in pool:
            continue
        pool[r["room_type"]].append(r)
    assert not {r["file"] for rs in pool.values() for r in rs} & held
    meta = {
        "hbank_rows": len(rows),
        "held_out_runs": len(held),
        "fights": {k: len(v) for k, v in pool.items()},
        "runs": len({r["file"] for rs in pool.values() for r in rs}),
    }
    return pool, meta


# `build_scenario_state` caps the potion belt at 5 slots.
MAX_POTION_SLOTS = 5


def own_pool(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    """`{room: rows}` of harvested own-deck fights (JSONL files)."""
    pool: dict[str, list[dict[str, Any]]] = {room: [] for room in ROOMS}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("room_type") not in pool:
                    continue
                if int(r.get("potion_slots") or 0) > MAX_POTION_SLOTS:
                    continue
                pool[r["room_type"]].append(r)
    return pool


def is_own(row: dict[str, Any]) -> bool:
    return row.get("cohort") == "own"


class StartSampler:
    """Draws `(row, build_seed)` with fixed room weights (and an own-pool share)."""

    def __init__(
        self,
        pool: dict[str, list[dict[str, Any]]],
        weights: dict[str, float],
        seed: int,
        own: dict[str, list[dict[str, Any]]] | None = None,
        own_share: float = 0.0,
    ) -> None:
        own = own or {}
        self.rooms = [r for r in ROOMS if weights.get(r, 0) > 0 and (pool.get(r) or own.get(r))]
        if not self.rooms:
            raise ValueError("no room with a positive weight has fights")
        self.w = [weights[r] for r in self.rooms]
        self.pool = pool
        self.own = own
        self.own_share = own_share
        self.rng = random.Random(seed)

    def draw(self) -> tuple[dict[str, Any], int]:
        room = self.rng.choices(self.rooms, self.w)[0]
        use_own = bool(self.own.get(room)) and (
            not self.pool.get(room) or self.rng.random() < self.own_share
        )
        row = self.rng.choice((self.own if use_own else self.pool)[room])
        # `RunManager` takes an i32 seed. [2^30, 2^31) never meets an eval build
        # seed (`_fight_seed·100 + 77 (+1000·rep)` stays below 1.1e8).
        return row, self.rng.randrange(TRAIN_SEED_LO, 2**31)


def parse_weights(spec: str) -> dict[str, float]:
    """`boss=0.5,elite=0.3,monster=0.2` -> dict."""
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        if k not in ROOMS:
            raise ValueError(f"room {k!r} not in {ROOMS}")
        out[k] = float(v)
    return out
