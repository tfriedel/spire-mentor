"""F5 and F6: turn-by-turn C#-oracle lockstep at A10 on v0.111.0 (plan §3.5).

Both modes reuse ``parity_audit._run_trial_core`` (the same state surface and
first-divergence classifier as every parity sweep) with the ascension threaded
into both inits, and play random legal lines.

``encounters`` (F5) is the #918 elite lockstep,
``docs/experiments/2026-10-23-918-elite-fidelity/encounter_parity.py``, with the
same scenario pools. That script cannot run any more: its ``_rust_init``
override predates the ``encounter_floor`` argument GH #1054 added to the
trial core, so every trial now ends in ``HARNESS_INIT`` (``TypeError``). This
copy takes the argument and forwards it, as ``parity_fuzz._rust_init`` does.

``random-target`` (F6) builds Ironclad decks around everything an Ironclad run
can own that draws a target from the ``CombatTargets`` stream: random-target
attacks (``AttackCommand.TargetingRandomOpponents``), auto-played cards
(``CardCmd.AutoPlay`` picks an ``AnyEnemy`` target there), Juggernaut, The
Ball, three relics and Distilled Chaos. It plays them against the multi-enemy
fights, where the target choice matters.

Run from ``sts2-pycli/``::

    PYTHONPATH=python python -m ppo.e2e.fidelity_parity encounters \\
        --encounters DECIMILLIPEDE_ELITE,KNIGHTS_ELITE --trials 60 --max-turns 15 \\
        --pool v1 -j 16 --out ~/runs/e2e/fidelity/f5_v1.json
    PYTHONPATH=python python -m ppo.e2e.fidelity_parity random-target \\
        --trials 400 --max-turns 12 -j 16 --out ~/runs/e2e/fidelity/f6.json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

#: The build both engines run.
GAME_VERSION = "v0.111.0"

#: Ironclad-reachable cards that draw from ``CombatTargets`` in v0.111.0, with
#: the decomp class: SwordBoomerang, Volley and RipAndTear attack random
#: enemies; Havoc, Cascade, Stampede, Hellraiser, Mayhem and BeatDown
#: auto-play cards (CardCmd.cs:78); Juggernaut's power hits a random enemy on
#: block gain; TheBall picks a random player (TheBall.cs:73).
RANDOM_TARGET_CARDS: tuple[str, ...] = (
    "SWORD_BOOMERANG",
    "VOLLEY",
    "RIP_AND_TEAR",
    "HAVOC",
    "CASCADE",
    "STAMPEDE_CARD",
    "HELLRAISER_CARD",
    "MAYHEM_CARD",
    "BEAT_DOWN",
    "JUGGERNAUT_CARD",
    "THE_BALL",
)
#: Block for Juggernaut and Parrying Shield, single-target attacks for the
#: auto-play paths, and exhausts for Forgotten Soul.
RANDOM_TARGET_FILLER: tuple[str, ...] = (
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "SHRUG_IT_OFF",
    "IRON_WAVE",
    "POMMEL_STRIKE",
    "TRUE_GRIT",
    "BASH",
    "FLAME_BARRIER_CARD",
)
#: Kusarigama.cs:115, ParryingShield.cs:28, ForgottenSoul, and
#: WhisperingEarring (auto-plays the first turn).
RANDOM_TARGET_RELICS: tuple[str, ...] = (
    "KUSARIGAMA",
    "PARRYING_SHIELD",
    "FORGOTTEN_SOUL",
    "WHISPERING_EARRING",
)
RANDOM_TARGET_POTION = "DISTILLED_CHAOS"
# ``parity_audit._canon_card`` drops the Rust ``_CARD`` suffix from hand ids.
_RT_TOKENS = frozenset(c.removesuffix("_CARD") for c in RANDOM_TARGET_CARDS)
#: The v0.111.0 cards with ``HasEnergyCostX => true``.
_X_COST = frozenset(
    {
        "CASCADE",
        "DIRGE",
        "HEAVENLYDRILL",
        "MULTICAST",
        "SKEWER",
        "VOLLEY",
        "ERADICATE",
        "MALAISE",
        "TEMPEST",
        "WHIRLWIND",
    }
)


def _x_token(card_id: object) -> str:
    """``"CARD.HEAVENLY_DRILL"`` -> ``"HEAVENLYDRILL"`` (upgrade marks dropped)."""
    cid = str(card_id).upper().removeprefix("CARD.").split("+")[0]
    return cid.removesuffix("_CARD").replace("_", "")


_ASC = 10
_POOL = "v1"
_FORCED_RELICS: tuple[str, ...] = ()
_MULTI: tuple[str, ...] = ()
_ALIGN_SETUP_DEATHS = False


def _csharp_init(g: Any, sc: Any) -> dict:
    import parity_fuzz

    state = g.start(character=sc.character, seed=parity_fuzz.SEED_STRING, ascension=_ASC)
    g.skip_neow(state)
    r = g.set_player(
        hp=sc.hp,
        max_hp=sc.max_hp,
        deck=[parity_fuzz._csharp_card_id(c) for c in sc.deck],
        relics=list(sc.relics),
        potions=list(sc.potions),
    )
    if r.get("type") == "error":
        raise RuntimeError(f"set_player failed: {r.get('message')!r}")
    r = g.enter_room("combat", encounter=sc.encounter)
    if r.get("type") == "error":
        raise RuntimeError(f"enter_room failed: {r.get('message')!r}")
    return g.get_state()


def _rust_init(
    sc: Any,
    monsters: list[str] | None = None,
    move_states: list[str | None] | None = None,
    encounter_floor: int | None = None,
) -> Any:
    """``parity_fuzz._rust_init`` plus the ascension.

    With ``_ALIGN_SETUP_DEATHS`` (the random-target mode) the forced move
    states skip the enemies that died during combat setup. The trial core
    reads them from C#'s first snapshot, which lists only the living enemies,
    and the builder applies them by roster position. When a combat-start
    effect kills an enemy first (Whispering Earring auto-plays turn 1), every
    later enemy got its left neighbour's move: GH #1054 fixed the same shift
    for ``monsters``.
    """
    import parity_fuzz
    import sts2_client
    import sts2_rs

    rust_seed = parity_fuzz.rust_master_seed(parity_fuzz.SEED_STRING_CANONICAL)
    floor_kwargs = (
        {}
        if encounter_floor is None
        else {"run_seed": rust_seed, "encounter_floor": encounter_floor}
    )

    def build(forced: list[str | None] | None) -> Any:
        return sts2_rs.build_scenario_state(
            seed=rust_seed,
            character=sc.character,
            deck=list(sc.deck),
            hp=sc.hp,
            max_hp=sc.max_hp,
            relics=list(sc.relics),
            encounter=sc.encounter,
            monsters=monsters,
            potions=list(sc.potions),
            move_states=forced,
            ascension=_ASC,
            game_version=sts2_client.GAME_VERSION,
            **floor_kwargs,
        )

    if _ALIGN_SETUP_DEATHS and move_states:
        roster = sts2_rs.state_to_dict(build(None))["encounter"]
        gone = [bool(m.get("removed_after_death")) for m in roster]
        if any(gone) and gone.count(False) == len(move_states):
            living = iter(move_states)
            move_states = [None if g else next(living) for g in gone]
    return parity_fuzz._RustStateAdapter(build(move_states))


def encounter_scenario(encounter: str, seed: int, pool: str, forced_relics=()) -> Any:
    """The #918 scenario for ``(encounter, seed, pool)`` (encounter_parity.py)."""
    from parity_fuzz_scenario import IC_CARDS, SAFE_POTIONS, SAFE_RELICS, Scenario, generate_v2

    rng = random.Random(seed * 7919 + sum(map(ord, encounter)))
    if pool == "v2":
        sc = generate_v2(seed)
        # A4 Tight Belt: C# holds 2 potions, while a listed 3rd stays in the
        # Rust belt by design (scenario.rs), a LEGAL_ACTION artifact.
        potions = sc.potions[:2] if _ASC >= 4 else sc.potions
        return dataclasses.replace(
            sc, encounter=encounter, hp=sc.max_hp + 40, max_hp=sc.max_hp + 40, potions=potions
        )
    deck = tuple(rng.choice(IC_CARDS) for _ in range(rng.randint(10, 16)))
    relics = tuple(forced_relics) or tuple(
        rng.sample(SAFE_RELICS, rng.randint(0, min(3, len(SAFE_RELICS))))
    )
    potions = tuple(rng.sample(SAFE_POTIONS, rng.randint(0, min(2, len(SAFE_POTIONS)))))
    max_hp = rng.randint(90, 140)
    return Scenario("Ironclad", deck, max_hp, max_hp, relics, encounter, potions)


def random_target_scenario(seed: int, encounters: tuple[str, ...]) -> Any:
    """An Ironclad deck with 3-6 random-target cards in a multi-enemy fight."""
    from parity_fuzz_scenario import Scenario

    rng = random.Random(seed * 104729 + 17)
    sources = [rng.choice(RANDOM_TARGET_CARDS) for _ in range(rng.randint(3, 6))]
    filler = [rng.choice(RANDOM_TARGET_FILLER) for _ in range(rng.randint(5, 9))]
    deck = tuple(c + ("+1" if rng.random() < 0.25 else "") for c in sources + filler)
    relics = tuple(rng.sample(RANDOM_TARGET_RELICS, rng.randint(0, 2)))
    potions = (RANDOM_TARGET_POTION,) if rng.random() < 0.3 else ()
    max_hp = rng.randint(120, 180)
    return Scenario("Ironclad", deck, max_hp, max_hp, relics, rng.choice(encounters), potions)


def multi_enemy_encounters() -> tuple[str, ...]:
    """The sweep's encounters whose native v0.111.0 roster has 2+ enemies."""
    import sts2_rs
    from parity_fuzz_scenario import ALL_ENCOUNTERS

    out = []
    for enc in ALL_ENCOUNTERS:
        cs = sts2_rs.build_scenario_state(
            seed=1,
            character="Ironclad",
            deck=["STRIKE_IRONCLAD"] * 10,
            hp=100,
            max_hp=100,
            relics=[],
            encounter=enc,
            potions=[],
            ascension=_ASC,
            game_version=GAME_VERSION,
        )
        if len(sts2_rs.state_to_dict(cs)["encounter"]) >= 2:
            out.append(enc)
    return tuple(out)


def _counting_oracle(sc: Any, *, x_cost_zero: bool = False) -> Any:
    """The live C# oracle, counting what the trial reached.

    ``x_cost_zero`` reports every X-cost card's hand cost as 0, as Rust does.
    The headless reads ``EnergyCost.GetResolved()``, which for an X-cost card
    is the X captured at its last play (CardEnergyCost.cs:155-159), so a
    played Volley or Cascade that comes back to the hand reads 1-3 in C#.
    ``_is_inflight_pile_move`` tolerates hand-cost diffs and so books that as
    an artifact, which ends the trial clean at the first reshuffle. X is
    captured again at every play, and the X-driven damage stays compared.
    """
    import parity_audit

    class CountingOracle(parity_audit._LiveCsOracle):
        """The live C# oracle, also recording enemy turns, the outcome and the
        (monster, move state) pairs it showed."""

        def __init__(self, sc: Any) -> None:
            super().__init__(sc)
            self.end_turns = 0
            self.last: dict = {}
            self.moves: set[tuple[str, str]] = set()

        def apply(self, action: dict) -> None:
            if action["action"] == "end_turn":
                self.end_turns += 1
            super().apply(action)

        def snapshot(self) -> dict:
            s = super().snapshot()
            costs = s.get("hand_costs")
            if x_cost_zero and costs:
                hand = s.get("hand_ids") or []
                s["hand_costs"] = [
                    0 if i < len(hand) and _x_token(hand[i]) in _X_COST else c
                    for i, c in enumerate(costs)
                ]
            self.last = s
            for m in s.get("monsters") or []:
                self.moves.add((m.get("kind"), m.get("move_state")))
            return s

    return CountingOracle(sc)


def _init_worker(
    asc: int,
    version: str,
    pool: str,
    relics: tuple[str, ...],
    multi: tuple[str, ...],
    align: bool = False,
) -> None:
    global _ASC, _POOL, _FORCED_RELICS, _MULTI, _ALIGN_SETUP_DEATHS
    import parity_audit
    import sts2_client

    _ASC, _POOL, _FORCED_RELICS, _MULTI = asc, pool, relics, multi
    _ALIGN_SETUP_DEATHS = align
    sts2_client.set_game_version(version)
    parity_audit._csharp_init = _csharp_init
    parity_audit._rust_init = _rust_init
    parity_audit._init_worker_cache(None, None, False)


def _one(item: tuple[str, str, int, int]) -> tuple[str, dict]:
    import parity_audit

    mode, enc, seed, max_turns = item
    if mode == "random-target":
        sc = random_target_scenario(seed, _MULTI)
    else:
        sc = encounter_scenario(enc, seed, _POOL, _FORCED_RELICS)
    oracle = _counting_oracle(sc, x_cost_zero=mode == "random-target")
    r = parity_audit._run_trial_core(seed, max_turns, sc, oracle)
    cov = r.pop("_coverage", None) or {}
    played = collections.Counter(str(c).upper() for c in cov.get("cards_played") or [])
    rt = {c: n for c, n in played.items() if c.split("+")[0] in _RT_TOKENS}
    r["random_target_plays"] = rt
    r["end_turns"] = oracle.end_turns
    r["outcome"] = (
        ("won" if oracle.last.get("player_won") else "lost")
        if oracle.last.get("combat_terminal")
        else "open"
    )
    r["moves"] = sorted(f"{k}:{m}" for k, m in oracle.moves if k)
    return sc.encounter, r


_KEEP = (
    "seed",
    "category",
    "first_path",
    "action",
    "turn",
    "divergences",
    "scenario",
    "end_turns",
    "outcome",
    "moves",
    "random_target_plays",
)


def main(argv: list[str] | None = None) -> int:
    import parity_audit
    import sts2_client

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["encounters", "random-target"])
    ap.add_argument("--encounters", default="", help="comma-separated ids (encounters mode)")
    ap.add_argument("--trials", type=int, default=20, help="per encounter, or in all")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--pool", choices=["v1", "v2"], default="v1")
    ap.add_argument("--relics", default="", help="relics every v1 scenario holds")
    ap.add_argument(
        "--align-setup-deaths",
        action="store_true",
        help="skip enemies killed during combat setup when forcing C#'s move states",
    )
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sts2_client.set_game_version(GAME_VERSION)
    parity_audit.assert_oracle_fresh()

    global _ASC
    _ASC = args.ascension
    seeds = range(args.seed_start, args.seed_start + args.trials)
    multi: tuple[str, ...] = ()
    if args.mode == "random-target":
        multi = multi_enemy_encounters()
        items = [(args.mode, "", s, args.max_turns) for s in seeds]
    else:
        encs = [e for e in args.encounters.split(",") if e]
        if not encs:
            ap.error("encounters mode needs --encounters")
        items = [(args.mode, e, s, args.max_turns) for e in encs for s in seeds]

    by_enc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    depth: dict[str, list[int]] = collections.defaultdict(list)
    outcomes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    moves: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rt_plays: collections.Counter[str] = collections.Counter()
    artifact_stops: collections.Counter[str] = collections.Counter()
    rows = []
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    progress = args.out.with_suffix(".rows.jsonl")
    initargs = (
        args.ascension,
        GAME_VERSION,
        args.pool,
        tuple(r for r in args.relics.split(",") if r),
        multi,
        args.align_setup_deaths,
    )
    with (
        ProcessPoolExecutor(args.jobs, initializer=_init_worker, initargs=initargs) as ex,
        progress.open("w", encoding="utf-8") as fh,
    ):
        for done, (enc, r) in enumerate(ex.map(_one, items), 1):
            cat = r["category"] or "clean"
            by_enc[enc][cat] += 1
            if cat == "clean":
                depth[enc].append(r["end_turns"])
                outcomes[enc][r["outcome"]] += 1
                moves[enc].update(r["moves"])
                if r.get("first_path"):
                    artifact_stops[r["first_path"]] += 1
            rt_plays.update(r["random_target_plays"])
            row = {"encounter": enc, **{k: r.get(k) for k in _KEEP}}
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if cat != "clean":
                print(
                    f"{enc} seed={r['seed']} {cat} {r['first_path']} turn={r['turn']}",
                    file=sys.stderr,
                    flush=True,
                )
            if done % 25 == 0 or done == len(items):
                print(f"[{time.time() - t0:7.1f}s] {done}/{len(items)} trials", flush=True)
    summary = {
        enc: {
            "categories": dict(c),
            "clean_enemy_turns_total": sum(depth[enc]),
            "clean_enemy_turns_mean": round(sum(depth[enc]) / max(len(depth[enc]), 1), 2),
            "clean_outcomes": dict(outcomes[enc]),
            "moves_seen": dict(sorted(moves[enc].items())),
        }
        for enc, c in sorted(by_enc.items())
    }
    total = collections.Counter()
    for c in by_enc.values():
        total.update(c)
    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "game_version": GAME_VERSION,
        "totals": dict(total),
        "clean": total["clean"],
        "trials": sum(total.values()),
        "clean_but_stopped_at_an_artifact": dict(artifact_stops),
        "clean_enemy_turns_total": sum(sum(v) for v in depth.values()),
        "random_target_plays": dict(rt_plays),
        "multi_enemy_encounters": list(multi),
        "wall_s": round(time.time() - t0, 1),
        "summary": summary,
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "summary")}, indent=1))
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
