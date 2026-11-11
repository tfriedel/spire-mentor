"""Mirror an e2e lane's logs to Weights & Biases, from outside the learner.

The learner writes JSONL only (``ops_contract``: stats, throughput, events, evals/done).
This sidecar tails those files and logs them to one wandb run per lane, so training never
imports wandb and a wandb outage cannot touch it. It only reads the lane dir, except for
its own cursor file ``ops/wandb_sync.json``.

    python -m ppo.e2e.wandb_sync <lane save dir> [--project sts2-ppo-ironclad-e2e] [--once]

Every series uses ``decisions`` (the campaign's clock) as its x axis. Nested maps flatten
to ``group/key`` (``kl/combat``); strings and the stamps object are skipped (the stamps go
to the run config). A resume rewrites ``stats.jsonl`` (rows past the resumed iteration go
to ``stats.orphan.jsonl``): the cursor resets, and rows with an ``it`` already logged are
skipped, so wandb shows each iteration once (the replayed ones keep their first values).
Finished evals log ``eval/<panel>/<variant>/win_rate`` (+ wins, runs, rc) and the
panel-independent ``eval/<variant>/win_rate`` (one curve per variant across DEV-1000 and
DEV-2000, which share their first seeds) at the evaluated state's decision count, parsed
from its file name (``..._d<decisions>...``). Records without a ``variant`` count as
``raw``. Cursor schema 1 keyed evals by panel only, so the three variants of a DEV-2000
pin landed on one series: a cursor below ``EVAL_SCHEMA`` re-logs every eval once.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from . import ops_contract as OC

CURSOR = "ops/wandb_sync.json"
DEFAULT_PROJECT = "sts2-ppo-ironclad-e2e"
_D_RE = re.compile(r"_d(\d+)")
#: The eval key layout of the cursor's ``evals_seen``; a lower one re-logs every eval.
EVAL_SCHEMA = 2


def flatten(row: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric leaves of ``row`` as ``prefix + a/b`` keys; strings, None and non-finite skipped."""
    out: dict[str, float] = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            if math.isfinite(v):
                out[key] = float(v)
        elif isinstance(v, Mapping) and k != "stamps":
            out.update(flatten(v, key + "/"))
    return out


# One call reads at most this much, and a file that shrank below the cursor (a resume
# truncates stats.jsonl to the checkpoint's row) is re-read from this far before its end,
# not from byte 0: a 2 GB stats.jsonl parsed in one go is the mirror's memory limit
# (2026-11-11). The rows before the window were logged already (`max_it` skips repeats).
MAX_READ = 64 << 20


def read_new_lines(
    path: Path, offset: int, *, max_read: int = MAX_READ
) -> tuple[list[dict[str, Any]], int, bool]:
    """Complete JSON lines after ``offset``, ``max_read`` bytes at most (the rest comes
    with the next call); returns (rows, new offset, file was rewritten)."""
    if not path.is_file():
        return [], offset, False
    size = path.stat().st_size
    rewritten = size < offset
    rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        if rewritten:
            offset = max(0, size - max_read)
            if offset:  # start on a line: skip the one the window cuts
                f.seek(offset)
                offset += len(f.readline())
        f.seek(offset)
        data = f.read(max_read)
    end = data.rfind(b"\n")
    if end < 0:
        if len(data) >= max_read:  # one line longer than the window: step over it
            return rows, offset + len(data), rewritten
        return rows, offset, rewritten
    for line in data[: end + 1].splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, offset + end + 1, rewritten


def eval_points(done_dir: Path, seen: set[str]) -> Iterator[tuple[str, dict[str, float]]]:
    """(record id, metrics) for each finished eval record not in ``seen``."""
    if not done_dir.is_dir():
        return
    for p in sorted(done_dir.glob("*.json")):
        if p.stem in seen:
            continue
        rec = OC.read_json(p) or {}
        panel = str(rec.get("panel") or "unknown")
        variant = str(rec.get("variant") or "raw")
        m = _D_RE.search(Path(str(rec.get("state", ""))).name)
        point: dict[str, float] = {}
        if m:
            point["decisions"] = float(m.group(1))
        wins, runs = rec.get("wins"), rec.get("runs")
        key = f"eval/{panel}/{variant}"
        if isinstance(wins, int) and isinstance(runs, int) and runs > 0:
            point[f"{key}/win_rate"] = wins / runs
            point[f"{key}/wins"] = float(wins)
            point[f"{key}/runs"] = float(runs)
            point[f"eval/{variant}/win_rate"] = wins / runs
        point[f"{key}/rc"] = float(rec.get("rc", -1) or 0)
        yield p.stem, point


class Sync:
    def __init__(self, sd: Path, run: Any) -> None:
        self.sd, self.run = sd, run
        self.cur_path = sd / CURSOR
        cur = OC.read_json(self.cur_path) or {}
        self.off: dict[str, int] = dict(cur.get("offsets", {}))
        self.max_it = int(cur.get("max_it", -1))
        schema = int(cur.get("eval_schema", 1))
        self.evals_seen: set[str] = (
            set(cur.get("evals_seen", [])) if schema >= EVAL_SCHEMA else set()
        )

    def save(self) -> None:
        self.cur_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "offsets": self.off,
            "max_it": self.max_it,
            "evals_seen": sorted(self.evals_seen),
            "eval_schema": EVAL_SCHEMA,
            "run_id": getattr(self.run, "id", None),
        }
        OC.atomic_write_text(self.cur_path, json.dumps(rec, indent=1) + "\n")

    def tick(self) -> int:
        n = 0
        rows, self.off[OC.STATS], _ = read_new_lines(self.sd / OC.STATS, self.off.get(OC.STATS, 0))
        for row in rows:
            it = row.get("it")
            if isinstance(it, int) and it <= self.max_it:
                continue
            if isinstance(it, int):
                self.max_it = it
            self.run.log(flatten(row))
            n += 1
        rows, self.off[OC.THROUGHPUT], _ = read_new_lines(
            self.sd / OC.THROUGHPUT, self.off.get(OC.THROUGHPUT, 0)
        )
        for row in rows:
            pt = flatten({k: v for k, v in row.items() if k != "decisions"}, "throughput/")
            if "decisions" in row:
                pt["decisions"] = float(row["decisions"])
            self.run.log(pt)
            n += 1
        rows, self.off[OC.EVENTS], _ = read_new_lines(
            self.sd / OC.EVENTS, self.off.get(OC.EVENTS, 0)
        )
        for row in rows:
            kind = str(row.get("event") or row.get("kind") or "event")
            self.run.log({f"events/{kind}": 1.0})
            n += 1
        for rid, point in eval_points(self.sd / OC.EVAL_DONE, self.evals_seen):
            self.run.log(point)
            self.evals_seen.add(rid)
            n += 1
        self.save()
        return n


def init_run(sd: Path, project: str, name: str, mode: Literal["online", "offline"]) -> Any:
    import wandb

    cur = OC.read_json(sd / CURSOR) or {}
    cfg = OC.read_json(sd / "cfg.json") or {}
    run = wandb.init(
        project=project,
        name=name,
        id=cur.get("run_id") or None,
        resume="allow",
        mode=mode,
        dir=str(sd / "ops"),
        config={"lane_dir": str(sd), **cfg},
    )
    run.define_metric("decisions")
    run.define_metric("*", step_metric="decisions")
    return run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mirror an e2e lane's logs to wandb.")
    ap.add_argument("save_dir", type=Path)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name", default="", help="run name (default: the lane dir's name)")
    ap.add_argument("--mode", default="online", choices=("online", "offline"))
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true", help="sync what is there and exit")
    a = ap.parse_args(argv)
    sd = a.save_dir.resolve()
    mode: Literal["online", "offline"] = "offline" if a.mode == "offline" else "online"
    run = init_run(sd, a.project, a.name or sd.name, mode)
    sync = Sync(sd, run)
    try:
        while True:
            try:
                n = sync.tick()
            except OSError as e:  # a file mid-rotation: retry next tick
                print(f"[wandb_sync] {e!r}", flush=True)
                n = 0
            if a.once:
                print(f"[wandb_sync] logged {n} points", flush=True)
                return 0
            if (sd / OC.DONE).exists() and n == 0:
                return 0
            time.sleep(a.interval)
    finally:
        run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
