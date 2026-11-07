"""Paired contrasts between e2e DEV arms: python -I sts2-pycli/scripts/e2e_paired.py NAME=ARM_DIR ...

Pairs on shared non-errored seeds; retry/ rows replace errored ones. Refuses evals/TEST (sealed).
"""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from ppo.e2e.stats import mcnemar_exact, newcombe_paired, wilson


def rows_in(folder: Path):
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def load(path: Path) -> dict[int, dict]:
    if "/evals/TEST" in str(path.resolve()):
        raise SystemExit(f"refusing sealed TEST path {path}")
    rows = {r["seed"]: r for r in rows_in(path / "shards")}
    for r in rows_in(path / "retry"):
        if rows.get(r["seed"], {}).get("outcome") == "error":
            rows[r["seed"]] = r
    return rows


def main() -> None:
    arms = {}
    for spec in sys.argv[1:]:
        name, _, p = spec.partition("=")
        arms[name] = load(Path(p).expanduser())
    for name, rows in arms.items():
        ok = [r for r in rows.values() if r.get("outcome") != "error"]
        k = sum(bool(r.get("won")) for r in ok)
        lo, hi = wilson(k, len(ok))
        print(
            f"{name:12s} {k:5d}/{len(ok):<5d} {100 * k / len(ok):6.2f}% "
            f"({100 * lo:.2f}-{100 * hi:.2f}) errors {len(rows) - len(ok)}"
        )
    for (n1, r1), (n2, r2) in combinations(arms.items(), 2):
        seeds = [
            s
            for s in sorted(set(r1) & set(r2))
            if r1[s].get("outcome") != "error" and r2[s].get("outcome") != "error"
        ]
        a = sum(bool(r1[s]["won"] and r2[s]["won"]) for s in seeds)
        b = sum(bool(r1[s]["won"] and not r2[s]["won"]) for s in seeds)
        c = sum(bool(r2[s]["won"] and not r1[s]["won"]) for s in seeds)
        d = len(seeds) - a - b - c
        lo, hi = newcombe_paired(a, b, c, d)
        print(
            f"{n1} - {n2}: n={len(seeds)} delta {100 * (b - c) / len(seeds):+.2f}pp "
            f"CI [{100 * lo:+.2f}, {100 * hi:+.2f}] discordant {b}/{c} "
            f"McNemar p={mcnemar_exact(b, c):.3g}"
        )


if __name__ == "__main__":
    main()
