"""Replay a recorded live run through Instinct, offline (``--record-requests``).

Feeds every ``advice_request`` of a recorded advisor session (the raw JSONL
``realgame_advisor --record-requests`` writes) through
:class:`realgame_instinct.InstinctAdvisor`, in order, and reports per surface
how many positions Instinct answered and why it declined the rest. The live
fight mirror is not replayed (no ``human_action`` following), so in-combat
card selections decline here even where the live advisor answers them.

Usage::

    uv run python python/realgame_instinct_replay.py logs/<run>/requests/requests-*.jsonl
        [--show REASON_SUBSTRING] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from realgame_instinct import InstinctAdvisor, InstinctUnavailable


def replay(path: Path, show: str | None = None, limit: int = 0) -> dict[str, Any]:
    from sts2_hint.recommender import RecommendConfig

    advisor = InstinctAdvisor()
    cfg = RecommendConfig()
    answered: Counter[str] = Counter()
    declined: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    n = 0
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            msg = json.loads(line)
            if msg.get("type") in ("game_over", "run_complete"):
                advisor.reset("run over")
            if msg.get("type") == "human_action":
                advisor.on_human_action(msg)
            if msg.get("type") != "advice_request":
                continue
            n += 1
            if limit and n > limit:
                break
            advice_for = msg.get("advice_for") or "combat_action"
            state = dict(msg)
            state["type"] = advice_for
            try:
                advisor.advise(state, advice_for, cfg=cfg, session={})
                answered[advice_for] += 1
            except InstinctUnavailable as exc:
                declined[advice_for][str(exc)] += 1
                if show and show in str(exc):
                    print(f"--- {advice_for}: {exc}\n{json.dumps(msg)[:1500]}\n")
            except Exception as exc:  # a bug, not a decline
                errors[f"{advice_for}: {type(exc).__name__}: {exc}"] += 1
    return {"requests": n, "answered": answered, "declined": declined, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--show", help="print the payload of declines whose reason contains this")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    r = replay(args.jsonl, args.show, args.limit)
    print(f"{r['requests']} requests")
    surfaces = sorted(set(r["answered"]) | set(r["declined"]))
    for s in surfaces:
        d = sum(r["declined"][s].values())
        print(f"  {s:15s} answered {r['answered'][s]:4d}  declined {d:4d}")
        for why, k in r["declined"][s].most_common():
            print(f"      {k:4d}  {why}")
    for e, k in r["errors"].most_common():
        print(f"  ERROR x{k}: {e}")
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
