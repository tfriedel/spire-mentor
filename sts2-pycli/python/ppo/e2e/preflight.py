"""The env's half of the pre-flight (plan §3.2; playbook §0.1), from `E2ETerminal.counts`.

At the G0 smoke and at 100M decisions, every decision surface must occur (the fine keys
card reward, remove, upgrade, transform, enchant, grant and smith, boss relic, relic
reward, potion swap, plus Neow), `unknown` must be 0, capped segments must stay at or
below 0.1%, and F9 must be clean. The embedding-drift half (playbook §0.2) reads
checkpoints, not the env, and is not here.

    tot = tally(terms)                 # the finished segments of a smoke
    pf = preflight(tot)
    assert pf.ok, pf.problems()

The deck-mutation surfaces are rare (playbook §0.1: about 0.1-0.5 per run), so the check
is "occurs at least once over the sample", not a per-run rate. `dec_per_run` gives the
rates for the stats row, keyed `dec_per_run/<TYPE>` for row types (upper case, the
contracts names) and `dec_per_run/<fine-key>` for the playbook's fine keys.

Deviation, recorded: the plan's list includes "relic reward", but on W1 the engine takes
elite and chest relic rewards and treasure without offering an action (the accepted gap
#1069, plan §3.2). In 72 random v0.111.0 A10 runs (66 of them HP-pinned, most into
act 3), no relic-reward or treasure row was stored, and none was forced in the 52 runs
whose forced steps were tallied. So the two are reported in `Preflight.gaps` when absent
and do not fail the check; the env still counts them (`rows/RELIC_REWARD`,
`forced/RELIC_REWARD`) if a flow ever offers one. SCHEMA.md §6.2 reports the same for the
schema's own walk, and its open question 10 recommends exactly this: drop relic reward from
the must-occur list and report its count.

A learner run (GH #1097 A7): random actors never leave act 1, so the G0 pre-flight reads
the 30-minute learner smoke, where teacher acting takes runs into acts 2 and 3. Its
`stats.jsonl` rows carry per iteration the stored rows per type, the segment outcomes,
the run and practice segment counts, `unknown`, F9's hits, the error types and
`segment_counts`, the summed `E2ETerminal.counts` of the segments that ended (the end
records' `counts`). `counts_from_stats` maps them onto the counters above: the row
fields as before, and from `segment_counts` the keys no other field carries
(`SEGMENT_KEYS`: the fine keys, `fine_unknown`, the cap asserts and the forced steps).
An OOC stall with no legal action is an error segment (`error/NoActions`, GH #1097
ENVR-2), so `outcome/error` and the error types count it. `preflight_stats` then checks every rule the counter path checks. Rows
written before the learner carried `segment_counts` cannot feed those rules: if any row
of the run lacks it, they are reported in `Preflight.unchecked`, not failed (each fine
key names the same surface as a required row type, `FINE_DTYPE`, so every surface is
still checked).

    python -m ppo.e2e.preflight --run-dir <the learner's --save-dir>

runs the stats half on the run's `stats.jsonl` and the fill check (`ppo.e2e.fillcheck`,
A4) on `--fill-runs` random A10 runs of this process's wheel, against the fill events
the run's stamps declare. The process must match the run's stamps (game version, wheel
build, Python source, vocab and schema), or its rows would not show the run's. Exit 0
on a pass, 1 on a fail, 2 on a usage error (RUNBOOK.md, "Run the pre-flight").
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Stored-row types (`contracts.DECISION_TYPES` names, `rows/<name>`) that must occur.
# CARD_BUNDLE (Scroll Boxes) is live but rare and not on the plan's list.
REQUIRED_DTYPES = (
    "COMBAT_TURN",
    "COMBAT_CHOICE",
    "NEOW",
    "EVENT",
    "MAP",
    "CARD_REWARD",
    "BOSS_RELIC",
    "SHOP",
    "REST",
    "SMITH",
    "CARD_REMOVE",
    "CARD_UPGRADE",
    "CARD_TRANSFORM",
    "CARD_ENCHANT",
    "CARD_GRANT",
    "POTION_SWAP",
)
# `encoder.fine_surface` keys that must occur (the playbook's fine family).
REQUIRED_FINE = (
    "card-reward",
    "card-remove",
    "card-upgrade",
    "card-transform",
    "card-enchant",
    "card-grant",
    "card-smith",
    "boss-relic",
    "potion-swap",
)
# Reported when absent, never required: the engine resolves them without a decision on W1.
GAP_KEYS = ("rows/RELIC_REWARD", "fine/relic-reward", "rows/TREASURE", "fine/treasure")
CAP_ALARM = 0.001  # plan §3.2: alarm above 0.1% capped segments
# The required row type that offers each fine key's surface (`encoder.fine_surface`'s
# action kind, `contracts.ooc_action_kind`, `contracts._DTYPE_BY_KIND`).
FINE_DTYPE = {
    "card-reward": "CARD_REWARD",
    "card-remove": "CARD_REMOVE",
    "card-upgrade": "CARD_UPGRADE",
    "card-transform": "CARD_TRANSFORM",
    "card-enchant": "CARD_ENCHANT",
    "card-grant": "CARD_GRANT",
    "card-smith": "SMITH",
    "boss-relic": "BOSS_RELIC",
    "potion-swap": "POTION_SWAP",
}
# What stats rows without `segment_counts` do not carry (module docstring).
STATS_UNCHECKED = (
    "fine/<key> (each surface is required as rows/<type>, FINE_DTYPE)",
    "fine_unknown",
    "assert/<cap> (counted in errors)",
    "combat_auto_share",
)
# The fill check's local runs (smoke seeds, plan §3.1: 162,000,000-162,000,999).
FILL_RUNS = 30
FILL_SEED = 162_000_900


def tally(terms: Iterable[Any]) -> collections.Counter:
    """Sum the counters of finished segments (`env.E2ETerminal`), plus `segments` and
    `segments/<kind>`."""
    tot: collections.Counter = collections.Counter()
    for t in terms:
        tot.update(t.counts)
        tot["segments"] += 1
        tot[f"segments/{t.segment.kind}"] += 1
    return tot


# The stats row fields `counts_from_stats` reads (`learner.Learner._stats_row`).
STATS_FIELDS = ("rows_type", "outcomes", "runs", "practice_segments", "unknown", "f9")
# The keys `counts_from_stats` takes from a row's `segment_counts` (the ended segments'
# summed `E2ETerminal.counts`): those no other stats field carries. Prefixes end in "/".
SEGMENT_KEYS = ("fine/", "fine_unknown", "assert/", "forced/")


def _segment_key(k: str) -> bool:
    return any(k.startswith(p) if p.endswith("/") else k == p for p in SEGMENT_KEYS)


def counts_from_stats(rows: Iterable[Mapping[str, Any]]) -> collections.Counter:
    """`tally`'s counters from a learner run's `stats.jsonl` rows, one per iteration:
    `rows/<TYPE>` (stored rows), `outcome/<o>`, `segments` (every ended segment),
    `segments/run`, `segments/practice`, `unknown`, `f9/checks`, `f9/violations`,
    `f9/tolerated` and `error/<type>`, plus the `SEGMENT_KEYS` of the rows'
    `segment_counts` where a row has them.

    Raises:
        ValueError: A row lacks one of `STATS_FIELDS` (not a learner stats row).
    """
    tot: collections.Counter = collections.Counter()
    for i, r in enumerate(rows):
        if miss := [k for k in STATS_FIELDS if k not in r]:
            raise ValueError(f"stats row {i} (it {r.get('it')}) has no {miss}")
        for k, v in r["rows_type"].items():
            tot[f"rows/{k}"] += int(v)
        for k, v in r["outcomes"].items():
            tot[f"outcome/{k}"] += int(v)
            tot["segments"] += int(v)
        tot["segments/run"] += int(r["runs"])
        tot["segments/practice"] += int(r["practice_segments"])
        tot["unknown"] += int(r["unknown"])
        f9 = r["f9"]
        tot["f9/checks"] += int(f9.get("checked", 0))
        tot["f9/violations"] += int(f9.get("hits", 0))
        tot["f9/tolerated"] += int(f9.get("tolerated", 0))
        for k, v in (r.get("error_types") or {}).items():
            tot[f"error/{k}"] += int(v)
        for k, v in (r.get("segment_counts") or {}).items():
            if _segment_key(str(k)):
                tot[str(k)] += int(v)
    return tot


@dataclass
class Preflight:
    ok: bool
    segments: int
    missing: list[str]  # "rows/<dtype>" or "fine/<key>" never seen
    gaps: list[str]  # GAP_KEYS never seen (informational)
    unknown: int  # stored rows no kind table claims (dtype or fine key)
    caps: int
    cap_rate: float
    errors: int
    f9_violations: int
    dec_per_run: dict[str, float] = field(default_factory=dict)
    # Share of combat steps auto-advanced (one legal action or one merged class; the
    # plan's 15.7% [CPPO]). Random play: 20-22%.
    combat_auto_share: float = 0.0
    # Decisions over the schema's member or class cap (`assert/<cap>`). Each also ended
    # its segment as an error, so it is in `errors` too.
    asserts: dict[str, int] = field(default_factory=dict)
    # Error segments by exception type (`error/<type>`, the env's counted reason).
    error_types: dict[str, int] = field(default_factory=dict)
    # Rules the counters' source cannot feed (`STATS_UNCHECKED` for stats rows).
    unchecked: list[str] = field(default_factory=list)

    def problems(self) -> list[str]:
        out = [f"missing {m}" for m in self.missing]
        if self.unknown:
            out.append(f"unknown = {self.unknown}")
        if self.cap_rate > CAP_ALARM:
            out.append(f"capped {self.caps}/{self.segments} = {self.cap_rate:.4%} > 0.1%")
        for cap, n in self.asserts.items():
            out.append(f"{n} decision(s) over the schema's {cap} cap")
        if self.errors:
            why = ", ".join(f"{k} {v}" for k, v in self.error_types.items())
            out.append(
                f"{self.errors} segment(s) ended in an error (engine or assert)"
                + (f": {why}" if why else "")
            )
        if self.f9_violations:
            out.append(f"{self.f9_violations} F9 violation(s)")
        return out

    def stats_row(self) -> dict[str, float]:
        """Flat keys for a stats row (`dec_per_run/<dtype>` as in the playbook)."""
        row = {f"dec_per_run/{k}": v for k, v in self.dec_per_run.items()}
        row.update(
            {
                "preflight/ok": float(self.ok),
                "preflight/missing": float(len(self.missing)),
                "preflight/unknown": float(self.unknown),
                "preflight/cap_rate": self.cap_rate,
                "preflight/errors": float(self.errors),
                "preflight/f9_violations": float(self.f9_violations),
                "preflight/combat_auto_share": self.combat_auto_share,
                "preflight/asserts": float(sum(self.asserts.values())),
            }
        )
        return row


def preflight(counts: Mapping[str, int], *, fine: bool = True) -> Preflight:
    """Check summed counters (`tally`) against the pre-flight rules. `fine=False`:
    the counters come from stats rows (`counts_from_stats`), which carry no fine keys;
    `STATS_UNCHECKED` goes to `Preflight.unchecked`."""
    n = int(counts.get("segments", 0))
    missing = [f"rows/{t}" for t in REQUIRED_DTYPES if counts.get(f"rows/{t}", 0) <= 0]
    if fine:
        missing += [f"fine/{k}" for k in REQUIRED_FINE if counts.get(f"fine/{k}", 0) <= 0]
    gaps = [k for k in GAP_KEYS if counts.get(k, 0) <= 0 and (fine or not k.startswith("fine/"))]
    unknown = int(counts.get("unknown", 0)) + int(counts.get("fine_unknown", 0))
    caps = int(counts.get("outcome/capped", 0))
    # An OOC stall (`no_actions`) is an error segment itself (`error/NoActions`).
    errors = int(counts.get("outcome/error", 0))
    f9 = int(counts.get("f9/violations", 0))
    error_types = {
        k.split("/", 1)[1]: int(v)
        for k, v in sorted(counts.items())
        if k.startswith("error/") and v
    }
    asserts = {
        k.split("/", 1)[1]: int(v)
        for k, v in sorted(counts.items())
        if k.startswith("assert/") and v
    }
    # Rates per full run (all segments' rows, practice included, over full runs).
    runs = max(1, int(counts.get("segments/run", 0)) or n)
    per_run = {
        k.split("/", 1)[1]: v / runs
        for k, v in sorted(counts.items())
        if k.startswith(("rows/", "fine/")) and v
    }
    cap_rate = caps / n if n else 0.0
    auto = int(counts.get("forced/combat", 0)) + int(counts.get("forced/combat_merged", 0))
    stored = int(counts.get("rows/COMBAT_TURN", 0)) + int(counts.get("rows/COMBAT_CHOICE", 0))
    ok = n > 0 and not missing and unknown == 0 and cap_rate <= CAP_ALARM and not errors and not f9
    return Preflight(
        ok=ok,
        segments=n,
        missing=missing,
        gaps=gaps,
        unknown=unknown,
        caps=caps,
        cap_rate=cap_rate,
        errors=errors,
        f9_violations=f9,
        dec_per_run=per_run,
        combat_auto_share=auto / (auto + stored) if auto + stored else 0.0,
        asserts=asserts,
        error_types=error_types,
        unchecked=[] if fine else list(STATS_UNCHECKED),
    )


def preflight_stats(rows: Iterable[Mapping[str, Any]]) -> Preflight:
    """`preflight` of a learner run's `stats.jsonl` rows (`counts_from_stats`): every
    rule when every row carries `segment_counts`, else `STATS_UNCHECKED` unchecked."""
    rows = list(rows)
    full = bool(rows) and all("segment_counts" in r for r in rows)
    return preflight(counts_from_stats(rows), fine=full)


def read_stats(run_dir: Path) -> list[dict[str, Any]]:
    """The rows of `<run_dir>/stats.jsonl` (`ops_contract.STATS`); a torn last line
    (a learner killed mid-write) is skipped."""
    from . import ops_contract as OC

    path = Path(run_dir) / OC.STATS
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def _stamp_problems(run: Mapping[str, Any]) -> list[str]:
    """How this process differs from a run's stamps where it would change its rows."""
    from . import contracts as C
    from . import stamps as ST
    from . import vocab as V

    mine = ST.stamps(vocab_sha256=V.load().sha256, schema_sha256=C.schema_sha256())
    moved = ST.stamp_change(dict(run), mine)
    if not moved:
        return []
    return [f"this process is not the run's ({moved}): run from its worktree, venv and wheel"]


def run_preflight(
    run_dir: Path,
    *,
    filled: Sequence[str] | None = None,
    fill_runs: int = FILL_RUNS,
    fill_seed: int = FILL_SEED,
) -> dict[str, Any]:
    """Both halves on a learner run (module docstring): `{"ok", "problems", "stats",
    "fill", "declared"}`. `filled` overrides the declared fill events of the run's
    stamps (its last stats row's, else `cfg.json`'s); a run that declares other
    events is a problem.

    Raises:
        ValueError: No stats rows, no declared fill events, or `fill_runs` < 1.
    """
    from . import contracts as C
    from . import fillcheck as FC

    if fill_runs < 1:
        raise ValueError("--fill-runs must be at least 1")
    rows = read_stats(run_dir)
    if not rows:
        raise ValueError(f"{run_dir}: no stats rows yet (the run finished no iteration)")
    pf = preflight_stats(rows)
    stamps = rows[-1].get("stamps") or {}
    if "filled" not in stamps:
        cfg = Path(run_dir) / "cfg.json"
        if cfg.exists():
            stamps = json.loads(cfg.read_text(encoding="utf-8")).get("stamps") or {}
    problems = [f"stats: {p}" for p in pf.problems()]
    have = stamps.get("filled")
    if filled is None:
        if have is None:
            raise ValueError(f"{run_dir}: the run's stamps declare no fill events; pass --filled")
        filled = have
    elif have is not None and sorted(have) != sorted(filled):
        problems.append(f"--filled {sorted(filled)} but the run declares {sorted(have)}")
    if bad := sorted(set(filled) - set(C.FILL_EVENTS)):
        raise ValueError(f"unknown fill events {bad} (contracts.FILL_EVENTS)")
    problems += _stamp_problems(stamps)
    rb, terms = FC.sample_rows(fill_runs, fill_seed)
    fc = FC.check(rb, filled)
    problems += [f"fill: {p}" for p in fc.problems]
    errs = [t for t in terms if t.outcome == "error"]
    if errs:
        problems.append(f"fill: {len(errs)} of {len(terms)} local runs ended in an error")
    return {
        "ok": not problems,
        "problems": problems,
        "declared": sorted(filled),
        "stats": dataclasses.asdict(pf),
        "fill": dataclasses.asdict(fc),
        "fill_summary": fc.summary(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.e2e.preflight",
        description="The pre-flight of a learner run: its stats rows and the fill check.",
    )
    ap.add_argument("--run-dir", required=True, type=Path, help="the learner's --save-dir")
    ap.add_argument("--filled", help="the declared fill events as JSON (default: the stamps')")
    ap.add_argument("--fill-runs", type=int, default=FILL_RUNS, help="local A10 runs")
    ap.add_argument("--fill-seed", type=int, default=FILL_SEED, help="their first seed")
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    import sts2_client

    try:
        filled = None if a.filled is None else json.loads(a.filled)
        if filled is not None and not (
            isinstance(filled, list) and all(isinstance(e, str) for e in filled)
        ):
            raise ValueError(f"--filled {a.filled!r} is not a JSON list of event names")
        rep = run_preflight(a.run_dir, filled=filled, fill_runs=a.fill_runs, fill_seed=a.fill_seed)
    except (OSError, ValueError, sts2_client.GameVersionMismatch) as e:
        # JSONDecodeError and contracts.SchemaError are ValueErrors.
        print(f"preflight: {e}", file=sys.stderr)
        return 2
    st = rep["stats"]
    print(
        f"stats: {st['segments']} segments, unknown {st['unknown']}, capped {st['caps']} "
        f"({st['cap_rate']:.4%}), errors {st['errors']}, F9 {st['f9_violations']}"
    )
    print("stats: rows per run " + ", ".join(f"{k} {v:.2f}" for k, v in st["dec_per_run"].items()))
    if st["gaps"]:
        print(f"stats: gaps (reported, not required): {st['gaps']}")
    print(f"stats: unchecked (not in stats rows): {st['unchecked']}")
    print(f"fill: {rep['fill_summary']}")
    if rep["fill"]["unchecked"]:
        print(f"fill: declared, no check here: {rep['fill']['unchecked']}")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    for p in rep["problems"]:
        print(f"problem: {p}")
    print(f"PREFLIGHT {'PASS' if rep['ok'] else 'FAIL'}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
