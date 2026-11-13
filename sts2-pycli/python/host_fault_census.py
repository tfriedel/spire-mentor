"""Census of the errors the headless host swallows (GH #1283).

Plays random-policy runs (``referee_spike.choose``, the policy of
``play_full_run.py``) on the game's own RNG and reads the host's per-site
counts of swallowed exceptions, force-finished events and game-reported errors
(``host_faults``, ``src/Sts2Headless/HostFaults.cs``) at the end of each run.

Each run is appended to the JSONL as it finishes; ``--summary`` prints the
per-site table from a finished (or running) JSONL.

Usage::

    STS2_GAME_VERSION=v0.111.0 python python/host_fault_census.py \\
        --runs 50 --out /tmp/census.jsonl
    python python/host_fault_census.py --summary /tmp/census.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import referee_spike
import sts2_client


def summarize(path: str) -> str:
    rows = []
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    fired: Counter[str] = Counter()
    runs_hit: Counter[str] = Counter()
    sample: dict[str, str] = {}
    by_cell: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    errors = 0
    floors = [r.get("floor") or 0 for r in rows]
    for row in rows:
        counts = row.get("host_faults") or {}
        cell = by_cell[row["character"], row["ascension"]]
        cell[0] += 1
        cell[1] += bool(counts)
        errors += "error" in row
        for site, n in counts.items():
            fired[site] += n
            runs_hit[site] += 1
            sample.setdefault(site, (row.get("host_fault_samples") or {}).get(site, ""))
    clean = sum(1 for r in rows if not r.get("host_faults"))
    lines = [
        f"runs: {len(rows)}, without any fault: {clean}, driver errors: {errors}, "
        f"mean floor reached: {sum(floors) / max(len(floors), 1):.1f}, max: {max(floors, default=0)}",
        "",
        "| character | asc | runs | runs with a fault |",
        "|---|---|---|---|",
    ]
    for (ch, asc), (n, hit) in sorted(by_cell.items()):
        lines.append(f"| {ch} | {asc} | {n} | {hit} |")
    lines += ["", "| site | fires | runs | sample |", "|---|---|---|---|"]
    for site, n in fired.most_common():
        text = sample[site].replace("|", "\\|")[:120]
        lines.append(f"| `{site}` | {n} | {runs_hit[site]} | {text} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=50, help="runs per character and ascension")
    ap.add_argument("--ascensions", type=int, nargs="+", default=[0, 10])
    ap.add_argument("--characters", nargs="+", default=referee_spike.CHARACTERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--referee",
        action="store_true",
        help="referee mode with the random-legal policy; a fault ends the run (v0.107.1 only)",
    )
    ap.add_argument("--out")
    ap.add_argument("--summary", metavar="JSONL")
    args = ap.parse_args()
    if args.summary:
        print(summarize(args.summary))
        return 0
    if not args.out:
        ap.error("--out is required")

    jobs = [
        (ch, asc, f"CENSUS{asc}-{i}")
        for ch in args.characters
        for asc in args.ascensions
        for i in range(args.runs)
    ]

    def one(job: tuple[str, int, str]) -> dict[str, Any]:
        ch, asc, label = job
        key = (
            referee_spike.episode_key(b"host-fault-census", f"{ch}:{label}")
            if args.referee
            else None
        )
        run = referee_spike.play(character=ch, ascension=asc, label=label, key_hex=key, census=True)
        if run.get("fault", "").startswith("host fault: "):
            # Referee mode stops at the first fault; count that one.
            site = run["fault"].removeprefix("host fault: ").split(" (", 1)[0]
            run["host_faults"] = {site: 1}
            run["host_fault_samples"] = {site: run["fault"]}
        run.pop("actions", None)
        run["game_version"] = sts2_client.GAME_VERSION
        return run

    t0 = time.perf_counter()
    with open(args.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for done, fut in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()
            print(
                f"[{done}/{len(jobs)}] {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True
            )
    print(summarize(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
