"""Python twin of the Rust ``run_arena`` binary (full-run-AI plan M0).

Bead ``sts2-rs-40ikw.4``. Wraps the Rust BINARY via subprocess (NOT the
PyO3 wheel — the binary is the product; it owns the process-per-seed
wall-clock-timeout layer). Adds the analysis surface:

  * ``sweep``   — freshness-guarded build + run of a panel sweep, then a
    completeness/bins/win-rate summary of the resulting ``report.jsonl``
    (plan §3.2: a win-rate number is only quotable alongside the
    completeness %).
  * ``summary`` — re-summarize an existing report.jsonl.
  * ``compare`` — paired-by-seed comparison of two report files:
    exact McNemar test on the discordant win pairs + a paired bootstrap
    CI on the win-rate delta (plan §3.1: all comparisons paired by seed).

Build-freshness guard (the stale-artifact trap has bitten twice — see
``parity_audit.assert_wheel_fresh``): before every sweep the binary's
mtime is checked against the newest ``crates/**/*.rs`` (+ Cargo.toml /
Cargo.lock) in the sibling sts2-rs checkout; a stale or missing binary is
rebuilt via ``cargo build --release -p sts2-cli --bin run_arena``
(``--no-build`` aborts with a clear message instead).

Run:

  python python/run_arena.py sweep --panel dev-200 --policy random --combat none
  python python/run_arena.py sweep --seeds 20 --combat eval-cheap --sims 64 --boss-sims 64
  python python/run_arena.py compare a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RS_ROOT = (_HERE / ".." / ".." / "sts2-rs").resolve()
_BINARY = _RS_ROOT / "target" / "release" / "run_arena"
_PANEL_DIR = _RS_ROOT / "data" / "seed-panels"


# ── build-freshness guard ────────────────────────────────────────────────────


def _newest_source_mtime(rs_root: Path) -> float:
    """Newest mtime over the Rust workspace inputs the binary is built from."""
    newest = 0.0
    for pattern in ("crates/**/*.rs", "crates/**/Cargo.toml"):
        for p in rs_root.glob(pattern):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
    for name in ("Cargo.toml", "Cargo.lock"):
        p = rs_root / name
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    return newest


def ensure_binary_fresh(no_build: bool = False) -> Path:
    """Return the run_arena binary path, rebuilding if stale/missing.

    Mtime-based (the binary has no embedded src-hash surface yet): stale
    means any crates/**/*.rs (or workspace Cargo.{toml,lock}) is newer
    than the binary. With ``no_build`` a stale binary ABORTS — silently
    sweeping a stale binary produces garbage attributed to fresh code.
    """
    stale = not _BINARY.exists() or _BINARY.stat().st_mtime < _newest_source_mtime(_RS_ROOT)
    if not stale:
        return _BINARY
    if no_build:
        sys.exit(
            f"run_arena binary is stale or missing ({_BINARY}).\n"
            "Rebuild with: cargo build --release -p sts2-cli --bin run_arena "
            f"(in {_RS_ROOT}) or drop --no-build."
        )
    print(f"[run_arena] rebuilding stale binary: {_BINARY}", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "sts2-cli", "--bin", "run_arena"],
        cwd=_RS_ROOT,
        check=True,
    )
    return _BINARY


# ── report loading + summary ─────────────────────────────────────────────────


def load_report(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Completeness % + non-completion bins + win rate (plan §3.2)."""
    total = len(rows)
    completed = sum(1 for r in rows if r["noncompletion_bin"] is None)
    won = sum(1 for r in rows if r["won"])
    bins = Counter(r["noncompletion_bin"] for r in rows if r["noncompletion_bin"] is not None)
    return {
        "total": total,
        "completed": completed,
        "completeness_pct": 100.0 * completed / total if total else 0.0,
        "won": won,
        "win_rate_pct": 100.0 * won / total if total else 0.0,
        "bins": sorted(bins.items(), key=lambda kv: (-kv[1], kv[0])),
    }


def print_summary(summary: dict) -> None:
    print("── arena summary (python) ──")
    print(f"  cells        : {summary['total']}")
    print(f"  completed    : {summary['completed']} ({summary['completeness_pct']:.1f}%)")
    print(f"  won          : {summary['won']} ({summary['win_rate_pct']:.1f}% of all cells)")
    if not summary["bins"]:
        print("  non-completion bins: none")
    else:
        print("  non-completion bins:")
        for bin_name, count in summary["bins"]:
            print(f"    {count:>6}  {bin_name}")
    if summary["completeness_pct"] < 90.0:
        print("  WARNING: completeness < 90% — win-rate numbers are PROVISIONAL (plan §3.2)")


# ── paired comparison (McNemar + bootstrap) ──────────────────────────────────


def _pair_rows(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    key = lambda r: (r["seed"], r["character"], r["ascension"])  # noqa: E731
    bmap = {key(r): r for r in b}
    pairs = []
    for ra in a:
        rb = bmap.get(key(ra))
        if rb is not None:
            pairs.append((ra, rb))
    return pairs


def _act1_composition(pairs: list[tuple[dict, dict]]) -> dict:
    """Classify how the two arms' Act-1 models line up across paired seeds.

    Three cases, and only two of them mean anything (GH #544):

    * ``same``  — every pair walks the same act. The act is a CONTROL and the
      delta is about whatever else differs between the arms.
    * ``split`` — every pair walks a different act. The act IS the treatment;
      this is the paired act-difficulty measurement.
    * ``mixed`` — some pairs match and some do not, which is what you get by
      comparing a fixed-act arm against a ``--act1 roll`` arm. The delta then
      blends "act B is harder" with "these particular seeds are harder", and
      no amount of n fixes it. Callers surface this as a warning.

    Reports written before ``act1`` existed carry no field; they were all
    Overgrowth, hence the default.
    """
    a_acts = [ra.get("act1", "overgrowth") for ra, _ in pairs]
    b_acts = [rb.get("act1", "overgrowth") for _, rb in pairs]
    matches = sum(1 for x, y in zip(a_acts, b_acts) if x == y)
    if matches == len(pairs):
        kind = "same"
    elif matches == 0:
        kind = "split"
    else:
        kind = "mixed"
    return {
        "kind": kind,
        "a": dict(sorted(Counter(a_acts).items())),
        "b": dict(sorted(Counter(b_acts).items())),
        "pairs_on_the_same_act": matches,
    }


def compare_reports(a: list[dict], b: list[dict], n_boot: int = 10_000) -> dict:
    """Paired-by-seed comparison of win outcomes (plan §3.1).

    McNemar: exact binomial test on the discordant pairs (b-won-only vs
    a-won-only). Bootstrap: resample pairs with replacement (fixed RNG
    seed → reproducible), 95% percentile CI on the win-rate delta (b−a).
    """
    import numpy as np
    from scipy import stats

    pairs = _pair_rows(a, b)
    if not pairs:
        sys.exit("no paired (seed, character, ascension) rows between the two reports")
    a_won = np.array([1 if ra["won"] else 0 for ra, _ in pairs])
    b_won = np.array([1 if rb["won"] else 0 for _, rb in pairs])

    n01 = int(((a_won == 0) & (b_won == 1)).sum())  # b wins where a lost
    n10 = int(((a_won == 1) & (b_won == 0)).sum())  # a wins where b lost
    discordant = n01 + n10
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        mcnemar_p = stats.binomtest(n01, discordant, 0.5).pvalue

    delta = float(b_won.mean() - a_won.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    boot_deltas = b_won[idx].mean(axis=1) - a_won[idx].mean(axis=1)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boot_deltas, [2.5, 97.5]))

    return {
        "pairs": len(pairs),
        "act1": _act1_composition(pairs),
        "a_win_rate_pct": 100.0 * float(a_won.mean()),
        "b_win_rate_pct": 100.0 * float(b_won.mean()),
        "delta_pp": 100.0 * delta,
        "discordant_b_only": n01,
        "discordant_a_only": n10,
        "mcnemar_p": mcnemar_p,
        "bootstrap_ci_pp": (100.0 * ci_lo, 100.0 * ci_hi),
    }


# ── CLI commands ─────────────────────────────────────────────────────────────


def _resolve_panel(panel: str) -> Path:
    """Accept a committed panel name (dev-200 / main-500 / release-2000) or a path."""
    p = Path(panel)
    if p.exists():
        return p
    named = _PANEL_DIR / f"{panel}.json"
    if named.exists():
        return named
    sys.exit(f"panel not found: '{panel}' (no such file, and no {named})")


def cmd_sweep(args: argparse.Namespace) -> None:
    binary = ensure_binary_fresh(no_build=args.no_build)
    cmd = [str(binary), "sweep", "--policy", args.policy, "--combat", args.combat]
    if args.panel:
        if args.character:
            sys.exit("--character applies to --seeds only; --panel rows carry their own")
        cmd += ["--panel", str(_resolve_panel(args.panel))]
    elif args.seeds is not None:
        cmd += ["--seeds", str(args.seeds), "--seed-start", str(args.seed_start)]
        if args.character:
            cmd += ["--character", args.character]
    else:
        sys.exit("pass --panel <name|file> or --seeds <N>")
    if args.ascension is not None:
        cmd += ["--ascension", str(args.ascension)]
    cmd += ["--act1", args.act1]
    if args.sims is not None:
        cmd += ["--sims", str(args.sims)]
    if args.boss_sims is not None:
        cmd += ["--boss-sims", str(args.boss_sims)]
    if args.jobs:
        cmd += ["--jobs", str(args.jobs)]
    if args.audit_log:
        cmd += ["--audit-log", str(args.audit_log)]
    if args.no_wall_ms:
        cmd += ["--no-wall-ms"]
    cmd += ["--timeout-ms", str(args.timeout_ms), "--out", str(args.out)]

    print(f"[run_arena] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    print_summary(summarize(load_report(Path(args.out))))


def cmd_summary(args: argparse.Namespace) -> None:
    print_summary(summarize(load_report(Path(args.report))))


def cmd_compare(args: argparse.Namespace) -> None:
    a = load_report(Path(args.report_a))
    b = load_report(Path(args.report_b))
    result = compare_reports(a, b, n_boot=args.bootstrap)
    print("── paired comparison (b vs a) ──")
    print(f"  paired cells : {result['pairs']}")
    act1 = result["act1"]
    print(f"  act 1        : a={act1['a']}  b={act1['b']}  ({act1['kind']})")
    if act1["kind"] == "mixed":
        print(
            "  WARNING: the two arms agree on Act 1 for "
            f"{act1['pairs_on_the_same_act']}/{result['pairs']} pairs and differ on the "
            "rest. This delta blends the act change with per-seed difficulty — it is "
            "NOT a clean paired act comparison. Run both arms on a fixed --act1."
        )
    print(f"  a win rate   : {result['a_win_rate_pct']:.1f}%  ({args.report_a})")
    print(f"  b win rate   : {result['b_win_rate_pct']:.1f}%  ({args.report_b})")
    print(f"  delta        : {result['delta_pp']:+.1f} pp")
    print(
        f"  discordant   : b-only {result['discordant_b_only']}, a-only {result['discordant_a_only']}"
    )
    print(f"  McNemar p    : {result['mcnemar_p']:.4g}")
    lo, hi = result["bootstrap_ci_pp"]
    print(f"  bootstrap CI : [{lo:+.1f}, {hi:+.1f}] pp (95%, paired)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser("sweep", help="run a panel sweep via the Rust binary")
    p_sweep.add_argument("--panel", help="panel name (dev-200|main-500|release-2000) or JSON path")
    p_sweep.add_argument("--seeds", type=int, help="ad-hoc seed range size (instead of --panel)")
    p_sweep.add_argument("--seed-start", type=int, default=0)
    # Must list every policy the Rust binary accepts. It used to stop at
    # `heuristic`, so asking for `ppo` was rejected by argparse and anyone who
    # dropped the flag silently got `random` instead -- a 0%-win-rate arm whose
    # runs die early and build tiny decks. That is the exact inverse of the
    # worst case the ONNX headroom blocker exists to find, and it reports a
    # comfortable PASS (GH #428).
    p_sweep.add_argument(
        "--policy",
        default="random",
        choices=["random", "first", "heuristic", "tiered", "value-net", "sampled-deck", "ppo"],
    )
    # Only meaningful with --seeds; --panel rows carry their own character.
    p_sweep.add_argument("--character", default=None)
    p_sweep.add_argument("--combat", default="none", choices=["none", "eval-cheap", "eval-strong"])
    p_sweep.add_argument("--ascension", type=int, default=None)
    # GH #544. Same token set as the Rust binary and the PyO3 `act1` kwarg.
    # `overgrowth` is the default so an unflagged sweep keeps measuring what
    # every historical report measured. To measure act difficulty, run two
    # fixed-act arms and `compare` them — do NOT compare a fixed arm against
    # `roll` (see `_act1_composition`).
    p_sweep.add_argument("--act1", default="roll", choices=["overgrowth", "underdocks", "roll"])
    p_sweep.add_argument("--sims", type=int, default=None, help="debug sims override")
    p_sweep.add_argument("--boss-sims", type=int, default=None)
    p_sweep.add_argument("--jobs", type=int, default=0)
    p_sweep.add_argument("--timeout-ms", type=int, default=600_000)
    p_sweep.add_argument("--out", default="report.jsonl")
    p_sweep.add_argument("--audit-log", default=None)
    p_sweep.add_argument("--no-wall-ms", action="store_true")
    p_sweep.add_argument(
        "--no-build", action="store_true", help="abort (instead of rebuild) on a stale binary"
    )
    p_sweep.set_defaults(func=cmd_sweep)

    p_summary = sub.add_parser("summary", help="summarize an existing report.jsonl")
    p_summary.add_argument("report")
    p_summary.set_defaults(func=cmd_summary)

    p_compare = sub.add_parser(
        "compare", help="paired McNemar + bootstrap CI between two report files"
    )
    p_compare.add_argument("report_a")
    p_compare.add_argument("report_b")
    p_compare.add_argument("--bootstrap", type=int, default=10_000)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
