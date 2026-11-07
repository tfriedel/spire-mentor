"""Decisions for tests, parity checks and benchmarks.

`game_decisions` plays hand-built v0.111.0 Ironclad fights at ascension 10
with random legal actions and encodes every decision with more than one
option through #1081's `Encoder` (identity rare map, `PRIV_DRAW` k = 10).
Late-game decks, relics, potions and elite and boss encounters make the
token counts realistic (mean about 48 honest tokens).

`synthetic_decisions` makes random decisions with the same shapes and about
the same sparsity, without the engine (fast; for unit tests and load tests).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

GAME_VERSION = "v0.111.0"

_DECK = [
    *["STRIKE_IRONCLAD"] * 4,
    *["DEFEND_IRONCLAD"] * 4,
    "BASH",
    "POMMEL_STRIKE",
    "INFLAME",
    "SHRUG_IT_OFF",
    "TWIN_STRIKE",
    "HEADBUTT",
    "IRON_WAVE",
    "ANGER",
    "THUNDERCLAP",
    "BATTLE_TRANCE",
    "OFFERING",
    "TRUE_GRIT",
    "UPPERCUT",
    "WHIRLWIND",
    "BODY_SLAM",
    "ARMAMENTS",
    "SECOND_WIND",
    "RAMPAGE",
    "MANGLE",
    "WOUND",
]
_RELICS = [
    "BURNING_BLOOD",
    "ANCHOR",
    "AKABEKO",
    "BAG_OF_MARBLES",
    "LANTERN",
    "VAJRA",
    "ORICHALCUM",
    "PEN_NIB",
    "KUNAI",
    "BRONZE_SCALES",
]
FIGHTS: tuple[dict[str, Any], ...] = (
    {
        "deck": _DECK,
        "hp": 60,
        "max_hp": 85,
        "relics": _RELICS,
        "encounter": "KNIGHTS_ELITE",
        "potions": ["FIRE_POTION", "BLOCK_POTION"],
        "room": "elite",
    },
    {
        "deck": _DECK[:22],
        "hp": 70,
        "max_hp": 80,
        "relics": _RELICS[:6],
        "encounter": "CEREMONIAL_BEAST_BOSS",
        "potions": ["ATTACK_POTION"],
        "room": "boss",
    },
    {
        "deck": _DECK,
        "hp": 55,
        "max_hp": 90,
        "relics": _RELICS,
        "encounter": "KAISER_CRAB_BOSS",
        "potions": ["FIRE_POTION", "DEXTERITY_POTION", "ENERGY_POTION"],
        "room": "boss",
    },
    {
        "deck": _DECK[:16],
        "hp": 50,
        "max_hp": 80,
        "relics": _RELICS[:4],
        "encounter": "SLIMES_NORMAL",
        "potions": [],
        "room": "monster",
    },
    {
        "deck": _DECK,
        "hp": 80,
        "max_hp": 95,
        "relics": _RELICS,
        "encounter": "QUEEN_BOSS",
        "potions": ["BLOCK_POTION", "FLEX_POTION"],
        "room": "boss",
    },
)


def identity_rare() -> dict[str, list[int]]:
    """A `RareMap` that maps every id to itself (as #1081's tests use)."""
    from ppo.combat_ctx import tokens as T

    v = T.vocab()
    return {s: list(range(v.tables[T.S[s]].size)) for s in T.CONTENT_SLOTS}


def game_decisions(
    n: int,
    seed: int = 1000,
    game_version: str = GAME_VERSION,
    priv_k: int = 10,
    rng_seed: int = 1,
    max_steps: int = 300,
) -> list[Any]:
    """`n` encoded decisions from random play of `FIGHTS` (round robin)."""
    import sts2_rs

    from ppo.combat_ctx.intent_log import IntentLog
    from ppo.combat_ppo.encode import Encoder

    enc = Encoder("S_ctx", identity_rare(), priv_k=priv_k)
    rng = random.Random(rng_seed)
    out: list[Any] = []
    s = seed
    while len(out) < n:
        f = FIGHTS[s % len(FIGHTS)]
        cs = sts2_rs.build_scenario_state(
            s,
            "Ironclad",
            list(f["deck"]),
            f["hp"],
            f["max_hp"],
            list(f["relics"]),
            f["encounter"],
            ascension=10,
            potions=list(f["potions"]),
            game_version=game_version,
        )
        s += 1
        ilog = IntentLog()
        for _ in range(max_steps):
            if len(out) >= n:
                break
            acts = cs.legal_actions()
            if not acts:
                break
            snap, bv = enc.observe(ilog, cs, f["room"])
            if len(acts) > 1:
                out.append(enc.encode(snap, bv, cs, acts))
            cs.apply(acts[rng.randrange(len(acts))])
    return out


def synthetic_decisions(
    n: int,
    seed: int = 0,
    mean_tok: int = 48,
    max_tok: int = 120,
    max_priv: int = 10,
    max_act: int = 12,
    cat_nnz: int = 3,
    cont_nnz: int = 5,
    edge_values: bool = False,
) -> list[Any]:
    """Random decisions with #1081's shapes and about its sparsity.

    With `edge_values`, some continuous cells are -0.0, NaN, inf or
    subnormal, to test that the codec keeps bit patterns.
    """
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ppo.encode import Decision

    rng = np.random.default_rng(seed)
    pads = T.vocab().pads.astype(np.int64)
    special = np.array([-0.0, np.nan, np.inf, -np.inf, 6e-8, 65504.0], np.float16)

    def toks(t: int) -> tuple[np.ndarray, np.ndarray]:
        cat = np.tile(pads.astype(np.int16), (t, 1))
        cont = np.zeros((t, T.N_CONT), np.float16)
        if t:
            for i in range(t):
                cs = rng.choice(T.N_CAT, size=min(cat_nnz, T.N_CAT), replace=False)
                cat[i, cs] = rng.integers(0, pads[cs])
                ks = rng.choice(T.N_CONT, size=min(cont_nnz, T.N_CONT), replace=False)
                cont[i, ks] = rng.normal(0, 2, len(ks)).astype(np.float16)
                if edge_values and rng.random() < 0.2:
                    cont[i, rng.integers(T.N_CONT)] = special[rng.integers(len(special))]
        return cat, cont

    out = []
    for _ in range(n):
        t = int(np.clip(rng.normal(mean_tok, 10), 4, max_tok))
        na = int(rng.integers(2, min(max_act, t) + 1))
        cat, cont = toks(t)
        sec = rng.integers(0, T.N_SECTION, t).astype(np.int8)
        p = int(rng.integers(0, max_priv + 1))
        pcat, pcont = toks(p)
        out.append(Decision(cat=cat, cont=cont, sec=sec, n_act=na, pcat=pcat, pcont=pcont))
    return out
