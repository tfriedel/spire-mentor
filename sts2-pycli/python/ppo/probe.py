"""M-P1 gate driver: ownership audit (G1b), throughput (G2), random-WR (G3).

Usage (from sts2-pycli, venv with sts2_rs):

    PYTHONPATH=python uv run --frozen python -m ppo.probe \
        --policy random --seeds 256 --out /tmp/ppo-g3.jsonl

    --policy expert --seeds 200   → G1b ownership table over 200 seeds
    --policy random --seeds 64    → G2 throughput probe / G3 win rate

Prints a summary: outcome histogram, win rate, wall-clock stats, and the
aggregated (phase, surface, resolution-class) ownership table. The G1b
assertion: no resolution class outside {combat-agent, single-option-auto, potion-swap-rule,
expert|net} and no yielded decision with surface 'unknown'.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time


def main(argv: list[str] | None = None) -> int:
    import sts2_client

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy",
        choices=("random", "expert", "net", "net_greedy", "net_search"),
        default="random",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="RunPolicyNet checkpoint (--policy net)")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--character", type=str, default="Ironclad")
    ap.add_argument(
        "--act1",
        type=str,
        default="roll",
        choices=["overgrowth", "underdocks", "roll"],
        help=(
            "which Act-1 model to walk (GH #545). Default 'roll', the real "
            "game's seed-driven pick; probe numbers from before that default "
            "are Overgrowth-only, so compare them with 'overgrowth'. Pair 'overgrowth' "
            "against 'underdocks' for a paired act-difficulty delta; do NOT "
            "pair either against 'roll', where the act is a function of the "
            "seed and a per-seed diff mixes the act change with everything "
            "downstream of it."
        ),
    )
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="JSONL sink (streamed)")
    ap.add_argument("--record", action="store_true", help="record per-decision data")
    ap.add_argument(
        "--sims", type=int, default=None, help="combat MCTS sims (None = eval-cheap 1000)"
    )
    ap.add_argument(
        "--boss-sims", type=int, default=None, help="boss combat MCTS sims (None = eval-cheap 4000)"
    )
    ap.add_argument(
        "--combat",
        choices=("mcts", "surrogate", "policy"),
        default="mcts",
        help="combat backend: play every fight (mcts) or fast-forward it through "
        "the combat surrogate (surrogate; requires --surrogate-path)",
    )
    ap.add_argument("--surrogate-path", type=str, default=None, help="combat surrogate net JSON")
    ap.add_argument(
        "--combat-ckpt",
        type=str,
        default=None,
        help="combat-PPO checkpoint for --combat policy (greedy, no search)",
    )
    ap.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="per-run wall-clock timeout in seconds (raise above the 600s default for "
        "high-sim benchmarks, else long runs are abandoned as losses)",
    )
    # Gumbel depth-1 search teacher (GH #452; ppo.search_policy). Shaping
    # defaults match rewards.ShapingConfig; pass the campaign's refit
    # coefficients when gating a specific champion.
    ap.add_argument("--search-m", type=int, default=6, help="Gumbel shortlist size")
    ap.add_argument("--search-n", type=int, default=2, help="reseeded forks per candidate")
    ap.add_argument("--search-combat-sims", type=int, default=100)
    ap.add_argument("--search-boss-sims", type=int, default=400)
    ap.add_argument(
        "--search-gumbel-scale",
        type=float,
        default=1.0,
        help="0 = deterministic teacher (no Gumbel exploration noise)",
    )
    ap.add_argument("--shaping-hp-coef", type=float, default=0.1)
    ap.add_argument("--shaping-upg-coef", type=float, default=0.035)
    ap.add_argument("--shaping-relic-coef", type=float, default=0.0)
    ap.add_argument("--shaping-offset", type=float, default=0.307)
    sts2_client.add_game_version_arg(ap)
    args = ap.parse_args(argv)
    sts2_client.apply_game_version_arg(args)
    if args.combat == "surrogate" and not args.surrogate_path:
        ap.error("--combat surrogate requires --surrogate-path")
    if args.combat == "policy" and not args.combat_ckpt:
        ap.error("--combat policy requires --combat-ckpt")
    search = None
    if args.policy == "net_search":
        search = {
            "m": args.search_m,
            "n": args.search_n,
            "combat_sims": args.search_combat_sims,
            "combat_boss_sims": args.search_boss_sims,
            "gumbel_scale": args.search_gumbel_scale,
            "shaping": {
                "hp_coef": args.shaping_hp_coef,
                "upg_coef": args.shaping_upg_coef,
                "relic_coef": args.shaping_relic_coef,
                "offset": args.shaping_offset,
            },
        }

    from ppo.workers import DEFAULT_RUN_TIMEOUT_S, play_batch

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    t0 = time.time()
    rows = play_batch(
        seeds,
        policy=args.policy,
        ascension=args.ascension,
        character=args.character,
        procs=args.procs,
        record_decisions=args.record,
        out_path=args.out,
        ckpt_path=args.ckpt,
        sims=args.sims,
        boss_sims=args.boss_sims,
        combat=args.combat,
        surrogate_path=args.surrogate_path,
        combat_ckpt=args.combat_ckpt,
        run_timeout_s=(args.run_timeout if args.run_timeout is not None else DEFAULT_RUN_TIMEOUT_S),
        search=search,
        act1=args.act1,
    )
    wall = time.time() - t0

    outcomes = collections.Counter(r.get("outcome", "?") for r in rows)
    # A crashed seed is MISSING DATA, not a loss. It carries `won: false` like
    # any other unfinished run, so counting it in the denominator reads a
    # half-crashed cell as a win-rate collapse instead of as a broken probe.
    # That happened (GH #545, 2026-08-26): a `just sim-engine` rebuild swapped
    # the extension module out from under a running probe's workers, 91/100
    # seeds raised `ModuleNotFoundError: No module named 'sts2_rs'`, and the
    # cell reported 8.0% between neighbours at 75%. Pooled with its arm it
    # inverted the sign of the headline result.
    errors = [r for r in rows if r.get("outcome") == "error"]
    scored = [r for r in rows if r.get("outcome") != "error"]
    wins = sum(1 for r in scored if r.get("won"))
    walls = [r["wall_s"] for r in rows if "wall_s" in r]
    n_dec = [r["n_decisions"] for r in rows if "n_decisions" in r]

    print(
        f"\n=== ppo.probe --policy {args.policy} n={len(rows)} "
        f"asc={args.ascension} char={args.character} ==="
    )
    print(f"outcomes: {dict(outcomes)}")
    print(f"win rate: {wins}/{len(scored)} = {wins / max(1, len(scored)):.3f}")
    if errors:
        print(
            f"\n*** {len(errors)}/{len(rows)} SEEDS CRASHED — THIS CELL IS "
            f"INCOMPLETE ***\n"
            f"    The win rate above covers the {len(scored)} runs that "
            f"finished. Downstream pairing MUST drop errored seeds "
            f"(filter on the `error` key); scoring them as losses biases the "
            f"delta toward whichever arm crashed."
        )
    if walls:
        print(
            f"wall/run: mean {statistics.mean(walls):.1f}s  "
            f"median {statistics.median(walls):.1f}s  max {max(walls):.1f}s; "
            f"batch wall {wall:.0f}s → {len(rows) / (wall / 60):.1f} runs/min"
        )
    if n_dec:
        print(f"yielded decisions/run: mean {statistics.mean(n_dec):.1f}")
    if args.policy == "net_search":
        tot_s = sum(r.get("search_stats", {}).get("searched", 0) for r in rows)
        tot_o = sum(r.get("search_stats", {}).get("overrides", 0) for r in rows)
        by: collections.Counter[str] = collections.Counter()
        byn: collections.Counter[str] = collections.Counter()
        for r in rows:
            for s, (o, cnt) in (r.get("search_stats", {}).get("by_surface") or {}).items():
                by[s] += o
                byn[s] += cnt
        print(f"\nsearch: {tot_o}/{tot_s} decisions overridden ({tot_o / max(1, tot_s):.1%})")
        for s in sorted(byn):
            print(f"  {s:12s} {by[s]}/{byn[s]}")

    # Aggregated ownership table (G1b).
    table: collections.Counter[str] = collections.Counter()
    for r in rows:
        for key, n in (r.get("resolution_counts") or {}).items():
            table[key] += n
    print("\nownership (phase/surface/class):")
    bad = []
    for key, n in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {key:48s} {n}")
        phase, surface, klass = key.split("/", 2)
        if klass not in (
            "combat-agent",
            "combat-surrogate",
            "single-option-auto",
            "expert",
            "net",
            "potion-swap-rule",
        ):
            bad.append(key)
        if klass in ("expert", "net") and surface in ("None", "unknown"):
            bad.append(key)
    if errors:
        print(f"\nERROR rows ({len(errors)}):")
        for r in errors[:5]:
            print(f"  seed {r['seed']}: {r.get('error')}")
    if bad:
        print(f"\nG1b OWNERSHIP AUDIT FAIL: unexplained resolutions: {bad}")
        return 1
    print("\nG1b ownership audit: PASS (no unexplained resolution classes)")
    if errors:
        # Non-zero exit for the same reason the ownership audit uses one: the
        # instrument's contract is "n seeds measured", and it did not deliver.
        # Silence here is what let a clobbered cell reach a results table.
        print(f"PROBE INCOMPLETE: {len(errors)}/{len(rows)} seeds errored")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
