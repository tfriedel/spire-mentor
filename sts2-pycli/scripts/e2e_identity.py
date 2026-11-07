"""Compare two e2e_arm.sh arms seed by seed (plan §8 identity rules).

Usage (no engine import needed):

    python scripts/e2e_identity.py ~/runs/e2e/evals/SMOKE/R2_W1 ~/runs/e2e/evals/SMOKE/R2_W2

Each argument is an arm directory written by ``scripts/e2e_arm.sh``
(``shards/*.jsonl`` plus ``retry/*.jsonl``). Rows are matched on ``seed`` and
compared on every field except the ones that describe the host or the clock
(wall time, the wheel record, the Python source digest). The W1 = W2 check and
the cross-host check at T0 both pass iff every shared seed matches and neither
arm has seeds the other lacks.

Refuses to read a TEST arm: the sealed panel is read only by score_test.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields that legitimately differ between two binaries or hosts that play
# identically: timings, and the record of which wheel and source ran.
IGNORED = frozenset({"wall_s", "age_s", "wheel", "py_src"})


def load_rows(arm_dir: Path) -> dict[int, dict]:
    if "TEST" in arm_dir.resolve().parts:
        raise SystemExit(f"refusing to read the sealed TEST panel: {arm_dir}")
    rows: dict[int, dict] = {}
    for sub in ("shards", "retry"):
        for path in sorted((arm_dir / sub).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                # A retry row replaces its errored shard row (e2e_arm.sh re-runs
                # an errored seed once), so later files win.
                if sub == "shards" and row["seed"] in rows:
                    raise SystemExit(f"duplicate seed {row['seed']} in {arm_dir}")
                rows[row["seed"]] = row
    if not rows:
        raise SystemExit(f"no rows under {arm_dir}/shards")
    return rows


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in IGNORED}


def compare(a: dict[int, dict], b: dict[int, dict]) -> tuple[list[int], list[int], list[int]]:
    """Return (mismatched seeds, seeds only in a, seeds only in b)."""
    shared = sorted(a.keys() & b.keys())
    mismatched = [s for s in shared if strip(a[s]) != strip(b[s])]
    return mismatched, sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())


def first_difference(ra: dict, rb: dict) -> str:
    sa, sb = strip(ra), strip(rb)
    for key in sorted(sa.keys() | sb.keys()):
        if sa.get(key) != sb.get(key):
            va, vb = json.dumps(sa.get(key))[:200], json.dumps(sb.get(key))[:200]
            return f"{key}: {va} != {vb}"
    return "(no difference)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two e2e_arm.sh arms seed by seed.")
    ap.add_argument("arm_a", type=Path)
    ap.add_argument("arm_b", type=Path)
    ap.add_argument("--show", type=int, default=5, help="mismatches to print")
    args = ap.parse_args(argv)
    a, b = load_rows(args.arm_a), load_rows(args.arm_b)
    mismatched, only_a, only_b = compare(a, b)
    shared = len(a.keys() & b.keys())
    print(
        f"shared seeds {shared}; mismatched {len(mismatched)}; only in A {len(only_a)}; only in B {len(only_b)}"
    )
    for seed in mismatched[: args.show]:
        print(f"  seed {seed}: {first_difference(a[seed], b[seed])}")
    ok = not mismatched and not only_a and not only_b
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
