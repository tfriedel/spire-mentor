"""The sealed TEST read of the Ironclad e2e campaign (GH #1097): PREREG-0 P2, applied once.

    PYTHONPATH=python <W2 python> -m ppo.e2e.score_test \\
        --prereg docs/experiments/2026-11-03-ironclad-e2e-v111/PREREG.md \\
        --evals ~/runs/e2e/evals --final <ARM> --cstar <ARM> --r1 R1 --r2 R2 --r3 <ARM> \\
        [--devboss FINAL.jsonl REF.jsonl] [--json OUT.json]

This is the only program that reads ``evals/TEST/`` (P4). ``--panel DEV`` (or
``SMOKE``, ``SMOKE_SEALED``) runs the same code on a non-sealed panel, which is
how it is tested. Arms are directories ``<evals>/<PANEL>/<ARM>/`` as
``scripts/e2e_arm.sh`` and ``ppo.e2e.evaluate`` write them.

The read runs in this order. Every check in steps 1-3 stops the read (exit 2)
before any statistic is computed, so a stopped read prints no win rate.

1. TEST: not before Thu Nov 12, 09:00 CET (P4, Amendment 5). TEST and its rehearsal
   SMOKE_SEALED: PREREG.md committed and unmodified; a second read needs
   ``--reread-reason`` (P4: "read once"), and every read is logged to
   ``<PANEL>/score_test.reads.jsonl``.
2. Each arm: its rows are the shard files ``shards/s<lo>_<hi>.jsonl`` plus the
   once-only retries ``retry/s<seed>.jsonl``. Each shard holds exactly its seeds
   once; the arm holds exactly its slice (TEST: all 4,000 seeds); no seed is
   off the panel (P2: "a file with != 4,000 rows or an off-panel seed stops the
   read"). A retry row belongs to a seed whose shard row errored, and replaces
   it. Every file matches the runner's manifest ``<PANEL>/<ARM>.sha256``, and on
   TEST that manifest matches its copy committed under ``sha256/`` next to
   PREREG.md (P4). An arm ran one engine binary (each row's ``wheel`` stamp),
   and every scored row is Ironclad A10.
3. Arguments: C* is R2 or R3 (PREREG-0 Agents table). ``--devboss`` files, if
   given, load and pair: the same states, the same pairs, each pair once.
4. Contrasts, P2's pairing: Δ = WR(Final) − WR(baseline) over the seeds both
   arms hold. A row is missing data when its outcome is ``error`` or
   ``timeout`` after the retry. A seed where the baseline is missing is dropped;
   a seed where only Final is missing counts as a Final loss. More than 0.5%
   such seeds (20 of TEST's 4,000) make the contrast INVALID, and so do two
   engine binaries or an act-1 model that differs on a seed (with
   ``--act1 roll`` the act is a function of the seed).
5. Bars, in sequence, stopping at the first failure (an INVALID bar fails):
   Bar1 Final − C*: Δ ≥ +5pp and exact two-sided McNemar p < 0.01.
   Bar2 Final − R1: Newcombe method-10 95% paired CI low end ≥ −3pp.
   Bar3 Final − R1: Δ ≥ +3pp and p < 0.01.
   BarR: WR(Final) ≥ 40%, and ≥ 56%, reported with Wilson CIs; Final's missing
   rows count as losses.
6. The descriptive family, Holm at α = 0.05 over all four members (a member
   that is missing or INVALID keeps its place with p = 1): Final − R2,
   Final − R3, A9-est Final − R1 (exact McNemar), DEVBOSS Final − REF (cluster
   bootstrap over states, ``ppo.e2e.devboss``).

Every contrast reports the observed discordance d (P3), the Newcombe CI, the
paired bootstrap cross-check (20,000 resamples, ``default_rng(20261109)``) and
its A9-est twin. A9-est is the cont300 gate's estimate
(``docs/experiments/ironclad-a10/a10_score.py``): an A10 run is an A9-est win
iff it won or ``reached_second_boss`` is true.

Timeouts are missing data, like errors. ``e2e_arm.sh`` re-runs only
``outcome == "error"``, and ``ppo.probe``'s watchdog books every pending run as
``timeout``, including runs that never started; so a timeout says nothing
about the run, and scoring a baseline timeout as a loss would favour Final.
This is the tighter reading of P2 and matches how the cont300 gate's
``a10_score.py`` dropped them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ppo.e2e import panels as P
from ppo.e2e import stats as S

MISSING_OUTCOMES = frozenset({"error", "timeout"})
#: The `wheel` stamp fields that name an engine build (`wheel_guard.IDENTITY_KEYS`
#: minus `so_path`, which names a file location, not a build).
WHEEL_KEYS = ("pgo", "pgo_instrumented", "profdata_sha256", "target_cpu", "src_hash")
BAR1_MIN_DELTA = Fraction(5, 100)
BAR3_MIN_DELTA = Fraction(3, 100)
BAR2_MIN_LOW = -0.03
BAR_ALPHA = 0.01
HOLM_ALPHA = 0.05
BARR_LINES = (0.40, 0.56)
CHARACTER = "Ironclad"
ASCENSION = 10
#: Paired-bootstrap resamples per contrast (P2); tests lower it for speed.
BOOT_RESAMPLES = S.BOOT_RESAMPLES

_SHARD_RE = re.compile(r"^s(\d+)_(\d+)\.jsonl$")
_RETRY_RE = re.compile(r"^s(\d+)\.jsonl$")
_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
READ_LOG = "score_test.reads.jsonl"


class ReadStopped(Exception):
    """An integrity failure: the read stops before any statistic is computed."""


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def is_missing(row: dict[str, Any]) -> bool:
    return row.get("outcome") in MISSING_OUTCOMES


def a10_win(row: dict[str, Any]) -> bool:
    return bool(row.get("won"))


def a9est_win(row: dict[str, Any]) -> bool:
    """The cont300 gate's A9 estimate: won, or entered the A10 second-boss room."""
    return bool(row.get("won")) or row.get("reached_second_boss") is True


METRICS: dict[str, Callable[[dict[str, Any]], bool]] = {"A10": a10_win, "A9-est": a9est_win}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_config(text: str) -> dict[str, list[str]]:
    """`key=value` lines (e2e_arm.sh's meta/config.txt); repeated keys collect in order."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out.setdefault(k, []).append(v)
    return out


def parse_manifest(text: str, where: Path) -> dict[str, str]:
    """`sha256sum` output (`<hex>  <path>` per line) -> {path: hex}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _MANIFEST_RE.match(line)
        if not m:
            raise ReadStopped(f"{where}: not a sha256sum line: {line!r}")
        if m.group(2) in out:
            raise ReadStopped(f"{where}: {m.group(2)} listed twice")
        out[m.group(2)] = m.group(1)
    return out


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ReadStopped(f"{path}:{lineno}: not JSON ({e})") from None
            if not isinstance(row, dict):
                raise ReadStopped(f"{path}:{lineno}: not a JSON object")
            rows.append(row)
    return rows


def _wheel_identity(row: dict[str, Any]) -> tuple | None:
    w = row.get("wheel")
    if not isinstance(w, dict):
        return None
    return tuple(w.get(k) for k in WHEEL_KEYS)


@dataclass
class Arm:
    """One arm's effective rows on one panel, after the integrity checks."""

    name: str
    panel: P.Panel
    start: int
    n: int
    rows: dict[int, dict[str, Any]]
    retried: list[int]
    config: dict[str, list[str]]
    wheel: tuple | None
    manifest: Path
    manifest_sha256: str
    files: int
    counts: dict[str, int] = field(default_factory=dict)

    def wheel_dict(self) -> dict[str, Any] | None:
        return None if self.wheel is None else dict(zip(WHEEL_KEYS, self.wheel))


def load_arm(evals: Path, panel: P.Panel, name: str, *, sealed_rules: bool) -> Arm:
    """Load and check one arm (step 2 of the module docstring); raise ReadStopped."""
    d = evals / panel.name / name
    if not (d / "shards").is_dir():
        raise ReadStopped(f"{d}: no shards/ directory")
    cfg_path = d / "meta" / "config.txt"
    config = parse_config(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    if not config and sealed_rules:
        raise ReadStopped(f"{cfg_path}: missing")
    if config:
        if config.get("arm") != [name]:
            raise ReadStopped(f"{cfg_path}: arm={config.get('arm')} is not {name}")
        if config.get("panel") != [panel.name]:
            raise ReadStopped(f"{cfg_path}: panel={config.get('panel')} is not {panel.name}")

    # Shards: each holds exactly its own seed range.
    shard_rows: dict[int, dict[str, Any]] = {}
    shard_files: list[Path] = []
    all_seeds: list[object] = []
    for path in sorted((d / "shards").iterdir()):
        m = _SHARD_RE.match(path.name)
        if not m or not path.is_file():
            raise ReadStopped(f"{path}: not a shard file (an unfinished or foreign run?)")
        lo, hi = int(m.group(1)), int(m.group(2))
        if not (panel.start <= lo < hi <= panel.stop):
            raise ReadStopped(f"{path}: shard range leaves the {panel.name} panel")
        rows = _read_rows(path)
        seeds = [r.get("seed") for r in rows]
        check = P.check_seeds(seeds, panel, start=lo, n=hi - lo)
        if not check.ok:
            raise ReadStopped(f"{path}: {check.describe()}")
        all_seeds.extend(seeds)
        for r in rows:
            shard_rows[r["seed"]] = r
        shard_files.append(path)
    if not shard_files:
        raise ReadStopped(f"{d}/shards: no shard files")

    # The arm holds exactly its slice.
    if config.get("seed_start") and config.get("n"):
        try:
            start, n = int(config["seed_start"][0]), int(config["n"][0])
        except ValueError:
            raise ReadStopped(f"{cfg_path}: seed_start/n are not integers") from None
    else:
        start, n = min(shard_rows), len(shard_rows)
    if sealed_rules and (start, n) != (panel.start, panel.n):
        raise ReadStopped(
            f"{d}: holds seeds {start}..{start + n - 1}; the {panel.name} read needs all "
            f"{panel.n} seeds {panel.start}..{panel.stop - 1}"
        )
    try:
        check = P.check_seeds(all_seeds, panel, start=start, n=n)
    except P.PanelError as e:
        raise ReadStopped(f"{d}: {e}") from None
    if not check.ok:
        raise ReadStopped(f"{d}: {check.describe()}")

    # Retries: at most one row each, for a seed whose shard row errored; the row
    # replaces the shard's. An empty retry file is a retry that produced no row
    # (its probe died before writing): the shard's error stands.
    rows = dict(shard_rows)
    retried: list[int] = []
    empty_retries: list[int] = []
    retry_files: list[Path] = []
    retry_dir = d / "retry"
    for path in sorted(retry_dir.iterdir()) if retry_dir.is_dir() else []:
        m = _RETRY_RE.match(path.name)
        if not m or not path.is_file():
            raise ReadStopped(f"{path}: not a retry file")
        seed = int(m.group(1))
        got = _read_rows(path)
        if len(got) > 1 or (got and got[0].get("seed") != seed):
            raise ReadStopped(f"{path}: a retry file holds at most one row, for seed {seed}")
        if seed not in panel:
            raise ReadStopped(f"{path}: off-panel seed {seed}")
        if seed not in shard_rows:
            raise ReadStopped(f"{path}: seed {seed} is not in this arm's slice")
        if shard_rows[seed].get("outcome") != "error":
            raise ReadStopped(f"{path}: seed {seed} did not error, so it has no retry")
        retry_files.append(path)
        if got:
            rows[seed] = got[0]
            retried.append(seed)
        else:
            empty_retries.append(seed)

    # The runner's manifest covers exactly these files, with these hashes.
    manifest = evals / panel.name / f"{name}.sha256"
    if not manifest.is_file():
        raise ReadStopped(f"{manifest}: missing (the runner writes it when the arm completes)")
    listed = parse_manifest(manifest.read_text(encoding="utf-8"), manifest)
    present = {f"shards/{p.name}": p for p in shard_files} | {
        f"retry/{p.name}": p for p in retry_files
    }
    if set(listed) != set(present):
        extra = sorted(set(listed) - set(present))[:3]
        absent = sorted(set(present) - set(listed))[:3]
        raise ReadStopped(f"{manifest}: lists {extra} not on disk; misses {absent}")
    for rel, path in present.items():
        if sha256_file(path) != listed[rel]:
            raise ReadStopped(f"{path}: sha256 differs from {manifest}")

    # One engine binary; every scored row is Ironclad A10.
    idents = {_wheel_identity(r) for r in rows.values() if not is_missing(r)}
    stamped = {i for i in idents if i is not None}
    if len(stamped) > 1:
        raise ReadStopped(f"{d}: rows ran {len(stamped)} engine builds: {sorted(stamped)}")
    if None in idents and sealed_rules:
        raise ReadStopped(f"{d}: a scored row carries no `wheel` stamp")
    for seed, r in rows.items():
        if is_missing(r):
            continue
        if r.get("character") != CHARACTER or r.get("ascension") != ASCENSION:
            raise ReadStopped(
                f"{d}: seed {seed} is {r.get('character')} A{r.get('ascension')}, "
                f"not {CHARACTER} A{ASCENSION}"
            )
        if not isinstance(r.get("won"), bool):
            raise ReadStopped(f"{d}: seed {seed} has no boolean `won`")

    outcomes = Counter(str(r.get("outcome")) for r in rows.values())
    counts = {
        "rows": len(rows),
        "errored_in_shards": sum(1 for r in shard_rows.values() if r.get("outcome") == "error"),
        "retried": len(retried),
        "retry_without_row": len(empty_retries),
        "missing_after_retry": sum(1 for r in rows.values() if is_missing(r)),
        "error_after_retry": outcomes.get("error", 0),
        "timeout": outcomes.get("timeout", 0),
        "capped": outcomes.get("capped", 0),
        "outcomes": dict(sorted(outcomes.items())),
    }
    return Arm(
        name=name,
        panel=panel,
        start=start,
        n=n,
        rows=rows,
        retried=sorted(retried),
        config=config,
        wheel=next(iter(stamped)) if stamped else None,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest),
        files=len(present),
        counts=counts,
    )


def arm_summary(arm: Arm) -> dict[str, Any]:
    """Descriptive per-arm numbers: WR over scored rows and with missing rows as losses."""
    scored = [r for r in arm.rows.values() if not is_missing(r)]
    out: dict[str, Any] = {
        "arm": arm.name,
        "seeds": f"{arm.start}..{arm.start + arm.n - 1}",
        "counts": arm.counts,
        "retried_seeds": arm.retried,
        "wheel": arm.wheel_dict(),
        "manifest_sha256": arm.manifest_sha256,
    }
    for metric, win in METRICS.items():
        k = sum(1 for r in scored if win(r))
        out[metric] = {
            "wins": k,
            "scored": len(scored),
            "rate_scored": k / len(scored) if scored else None,
            "wilson_scored": S.wilson(k, len(scored)) if scored else None,
            "rate_missing_as_loss": k / arm.n,
            "wilson_missing_as_loss": S.wilson(k, arm.n),
        }
    dec = [r["n_decisions"] for r in scored if isinstance(r.get("n_decisions"), int)]
    out["decisions_per_run"] = sum(dec) / len(dec) if dec else None
    out["act1"] = dict(Counter(str(r.get("act1")) for r in scored))
    return out


@dataclass
class Contrast:
    """Final − baseline over P2's pairing, for one metric (A10 or A9-est)."""

    label: str
    final: str
    base: str
    metric: str
    seeds: int
    dropped_base_missing: int
    final_missing_as_loss: int
    max_bad: int
    act1_mismatch: int
    wheels: tuple[tuple | None, tuple | None]
    pairs: list[tuple[bool, bool]]
    invalid: list[str]
    bootstrap: dict[str, float] | None = None

    @property
    def valid(self) -> bool:
        return not self.invalid

    @property
    def table(self) -> S.PairedTable | None:
        return S.PairedTable.from_pairs(self.pairs) if self.pairs else None

    @property
    def bad(self) -> int:
        return self.dropped_base_missing + self.final_missing_as_loss

    @property
    def p(self) -> float | None:
        t = self.table
        return None if t is None else S.mcnemar_exact(t.b, t.c)

    @property
    def delta(self) -> Fraction | None:
        t = self.table
        return None if t is None else Fraction(t.b - t.c, t.n)

    def to_dict(self) -> dict[str, Any]:
        t = self.table
        out: dict[str, Any] = {
            "label": self.label,
            "final": self.final,
            "baseline": self.base,
            "metric": self.metric,
            "valid": self.valid,
            "invalid": self.invalid,
            "seeds_in_both": self.seeds,
            "dropped_baseline_missing": self.dropped_base_missing,
            "final_missing_counted_as_loss": self.final_missing_as_loss,
            "missing_data_seeds": self.bad,
            "max_missing_data_seeds": self.max_bad,
            "act1_mismatch": self.act1_mismatch,
        }
        if t is not None:
            out |= {
                "pairs": t.n,
                "both_win": t.a,
                "final_only": t.b,
                "baseline_only": t.c,
                "both_lose": t.d,
                "wr_final": t.p1,
                "wr_baseline": t.p2,
                "delta": t.diff,
                "discordance": t.discordance,
                "mcnemar_p": self.p,
                "newcombe95": list(S.newcombe_paired(t.a, t.b, t.c, t.d)),
                "wilson95_final": list(S.wilson(t.a + t.b, t.n)),
                "wilson95_baseline": list(S.wilson(t.a + t.c, t.n)),
                "bootstrap95": self.bootstrap,
            }
        return out


def pair(
    final: Arm, base: Arm, metric: str, *, label: str | None = None, bootstrap: bool = True
) -> Contrast:
    """P2's pairing and error rules for Final − `base` on `metric`."""
    win = METRICS[metric]
    seeds = sorted(final.rows.keys() & base.rows.keys())
    pairs: list[tuple[bool, bool]] = []
    dropped = final_losses = act1_mismatch = 0
    for s in seeds:
        fr, br = final.rows[s], base.rows[s]
        if is_missing(br):
            dropped += 1
            continue
        if is_missing(fr):
            final_losses += 1
            pairs.append((False, win(br)))
            continue
        if "act1" in fr and "act1" in br and fr["act1"] != br["act1"]:
            act1_mismatch += 1
        pairs.append((win(fr), win(br)))
    max_bad = P.max_bad_seeds(len(seeds))
    invalid = []
    if not seeds:
        invalid.append("no seed in both arms")
    if dropped + final_losses > max_bad:
        invalid.append(
            f"{dropped + final_losses} seeds with missing data (baseline {dropped}, "
            f"Final only {final_losses}) > {max_bad} (0.5% of {len(seeds)})"
        )
    if final.wheel != base.wheel:
        invalid.append(f"engine builds differ: Final {final.wheel} vs {base.name} {base.wheel}")
    if act1_mismatch:
        invalid.append(f"the act-1 model differs on {act1_mismatch} seed(s)")
    c = Contrast(
        label=label or f"Final - {base.name} ({metric})",
        final=final.name,
        base=base.name,
        metric=metric,
        seeds=len(seeds),
        dropped_base_missing=dropped,
        final_missing_as_loss=final_losses,
        max_bad=max_bad,
        act1_mismatch=act1_mismatch,
        wheels=(final.wheel, base.wheel),
        pairs=pairs,
        invalid=invalid,
    )
    if bootstrap and pairs:
        c.bootstrap = S.paired_bootstrap(pairs, resamples=BOOT_RESAMPLES)
    return c


def bar_sequence(bar1: Contrast, r1: Contrast) -> list[dict[str, Any]]:
    """Bar1 -> Bar2 -> Bar3, each at its own level, stopping at the first failure."""
    out: list[dict[str, Any]] = []

    def entry(name: str, rule: str, c: Contrast, passed: bool | None) -> dict[str, Any]:
        t = c.table
        status = (
            "NOT TESTED"
            if passed is None
            else ("INVALID" if not c.valid else ("PASS" if passed else "FAIL"))
        )
        return {
            "bar": name,
            "rule": rule,
            "contrast": c.label,
            "status": status,
            "delta": None if t is None else t.diff,
            "mcnemar_p": c.p,
            "newcombe95": None if t is None else list(S.newcombe_paired(t.a, t.b, t.c, t.d)),
        }

    def significant(c: Contrast) -> bool:
        p = c.p
        return p is not None and p < BAR_ALPHA

    t1 = bar1.table
    pass1 = (
        bar1.valid
        and t1 is not None
        and Fraction(t1.b - t1.c, t1.n) >= BAR1_MIN_DELTA
        and significant(bar1)
    )
    out.append(entry("Bar1 (must)", "Δ ≥ +5pp and exact McNemar p < 0.01", bar1, pass1))
    t2 = r1.table
    pass2 = None
    if pass1:
        pass2 = (
            r1.valid
            and t2 is not None
            and S.newcombe_paired(t2.a, t2.b, t2.c, t2.d)[0] >= BAR2_MIN_LOW
        )
    out.append(entry("Bar2 (should)", "Newcombe 95% CI low end ≥ −3pp", r1, pass2))
    pass3 = None
    if pass2:
        pass3 = (
            r1.valid
            and t2 is not None
            and Fraction(t2.b - t2.c, t2.n) >= BAR3_MIN_DELTA
            and significant(r1)
        )
    out.append(entry("Bar3 (stretch)", "Δ ≥ +3pp and exact McNemar p < 0.01", r1, pass3))
    return out


def bar_r(final: Arm) -> dict[str, Any]:
    """BarR: WR(Final) over all its seeds, missing rows as losses, with Wilson CIs."""
    out: dict[str, Any] = {"n": final.n}
    for metric, win in METRICS.items():
        k = sum(1 for r in final.rows.values() if not is_missing(r) and win(r))
        lo, hi = S.wilson(k, final.n)
        out[metric] = {"wins": k, "rate": k / final.n, "wilson95": [lo, hi]}
    rate = out["A10"]["rate"]
    out["lines"] = {f">= {line:.0%}": rate >= line for line in BARR_LINES}
    return out


@dataclass
class Read:
    """Everything the scorer computed; `to_dict` is the --json output."""

    panel: str
    prereg: dict[str, Any]
    arms: dict[str, dict[str, Any]]
    manifests: dict[str, dict[str, Any]]
    bars: list[dict[str, Any]]
    barr: dict[str, Any]
    contrasts: dict[str, Contrast]
    family: dict[str, dict[str, Any]]
    devboss: dict[str, Any] | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "prereg": self.prereg,
            "arms": self.arms,
            "manifests": self.manifests,
            "bars": self.bars,
            "barR": self.barr,
            "contrasts": {k: c.to_dict() for k, c in self.contrasts.items()},
            "descriptive_family_holm": self.family,
            "devboss": self.devboss,
            "notes": self.notes,
        }


def git_state(path: Path) -> dict[str, Any]:
    """Whether `path` is tracked and unmodified in its git checkout, and its last commit."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(path.parent), *args], capture_output=True, text=True, check=False
        )

    try:
        tracked = git("ls-files", "--error-unmatch", path.name).returncode == 0
        clean = tracked and git("diff", "--quiet", "HEAD", "--", path.name).returncode == 0
        commit = git("log", "-1", "--format=%H", "--", path.name).stdout.strip() or None
    except OSError:
        return {"tracked": False, "clean": False, "commit": None}
    return {"tracked": tracked, "clean": clean, "commit": commit}


def manifest_candidates(prereg: Path, panel: str, arm: str) -> list[Path]:
    """Where the arm's manifest may be committed, next to PREREG.md under sha256/ (P4).

    ``sha256/<PANEL>/<ARM>.sha256`` is the recommended spelling; the others are
    accepted because P4 names only the directory.
    """
    base = prereg.parent / "sha256"
    cands = (
        base / panel / f"{arm}.sha256",
        base / f"{panel}_{arm}.sha256",
        base / f"{panel}-{arm}.sha256",
        base / f"{arm}.sha256",
    )
    return [c for c in cands if c.is_file()]


def check_manifest_commit(prereg: Path, arm: Arm, *, required: bool) -> dict[str, Any]:
    """Find the committed copy of the arm's manifest; on TEST it must exist and match."""
    out: dict[str, Any] = {"runner_manifest": str(arm.manifest), "sha256": arm.manifest_sha256}
    cands = manifest_candidates(prereg, arm.panel.name, arm.name)
    body = arm.manifest.read_bytes()
    match = next((c for c in cands if c.read_bytes() == body), None)
    if match is None:
        if required:
            where = ", ".join(str(c) for c in cands) or f"nothing under {prereg.parent / 'sha256'}"
            raise ReadStopped(
                f"{arm.name}: no committed manifest equals {arm.manifest} (P4); found {where}"
            )
        out["committed"] = None
        out["candidates"] = [str(c) for c in cands]
        return out
    state = git_state(match)
    out |= {"committed": str(match), "git": state}
    if required and not (state["tracked"] and state["clean"]):
        raise ReadStopped(f"{match}: not committed, or modified since its commit")
    return out


def _read_log_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score(
    *,
    prereg: Path,
    evals: Path,
    panel: str | P.Panel,
    final: str,
    cstar: str,
    r1: str,
    r2: str | None,
    r3: str | None,
    devboss: tuple[Path, Path] | None = None,
    argv: list[str] | None = None,
    reread_reason: str | None = None,
) -> Read:
    """The read (module docstring steps 1-6). Raises ReadStopped on any integrity failure.

    `panel` is a runner panel name (DEV, TEST, SMOKE, SMOKE_SEALED) or, for
    tests on other recorded seeds, a `Panel`. The sealed rules (step 1's PREREG
    and read-once checks, the full panel, committed manifests, wheel stamps)
    apply to the sealed panels, TEST and its rehearsal SMOKE_SEALED; the time
    lock applies to TEST alone.
    """
    panel = P.runner_panel(panel) if isinstance(panel, str) else panel
    sealed = panel.sealed
    notes: list[str] = []
    if cstar not in {r2, r3}:
        raise ReadStopped(f"--cstar {cstar} must be --r2 ({r2}) or --r3 ({r3}): C* is R2 or R3")
    names = [n for n in (final, cstar, r1, r2, r3) if n]
    distinct = [n for n in (final, r1, r2, r3) if n]
    if len(set(distinct)) != len(distinct):
        raise ReadStopped(f"--final/--r1/--r2/--r3 must name distinct arms, got {distinct}")

    if panel.name == P.TEST.name and _now() < P.UNSEAL_AT:
        raise ReadStopped(
            f"TEST is sealed until {P.UNSEAL_AT.isoformat()} (P4); it is {_now().isoformat()}"
        )
    if not prereg.is_file():
        raise ReadStopped(f"{prereg}: no PREREG file")
    pre = {"path": str(prereg), "sha256": sha256_file(prereg), "git": git_state(prereg)}
    if sealed and not (pre["git"]["tracked"] and pre["git"]["clean"]):
        raise ReadStopped(
            f"{prereg}: {panel.name} is read only under a committed, unmodified PREREG"
        )
    if not (pre["git"]["tracked"] and pre["git"]["clean"]):
        notes.append("PREREG.md is not committed or is modified (allowed on unsealed panels)")

    arms = {n: load_arm(evals, panel, n, sealed_rules=sealed) for n in dict.fromkeys(names)}
    manifests = {n: check_manifest_commit(prereg, a, required=sealed) for n, a in arms.items()}
    devboss_out = None
    if devboss is not None:
        from ppo.e2e.devboss import DevbossError, paired_results

        try:
            devboss_out = paired_results(devboss[0], devboss[1])
        except (DevbossError, OSError, ValueError, KeyError) as e:
            raise ReadStopped(f"DEVBOSS: {e}") from None

    log = evals / panel.name / READ_LOG
    if sealed:
        before = _read_log_entries(log)
        if before and not reread_reason:
            raise ReadStopped(
                f"{panel.name} was read {len(before)} time(s) before ({log}); P4 allows one "
                "read. Pass --reread-reason only for a read that printed nothing."
            )
        entry = {
            "at": _now().isoformat(),
            "argv": argv or [],
            "prereg_sha256": pre["sha256"],
            "prereg_commit": pre["git"]["commit"],
            "manifests": {n: a.manifest_sha256 for n, a in arms.items()},
            "reread_reason": reread_reason,
        }
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    fa = arms[final]
    contrasts: dict[str, Contrast] = {}
    contrasts["bar1"] = pair(fa, arms[cstar], "A10", label=f"Final - C* ({cstar}) (A10)")
    contrasts["r1"] = pair(fa, arms[r1], "A10", label=f"Final - R1 ({r1}) (A10)")
    contrasts["r1_a9"] = pair(fa, arms[r1], "A9-est", label=f"A9-est Final - R1 ({r1})")
    contrasts["bar1_a9"] = pair(fa, arms[cstar], "A9-est", label=f"A9-est Final - C* ({cstar})")
    for key, nm in (("r2", r2), ("r3", r3)):
        if nm:
            contrasts[key] = pair(fa, arms[nm], "A10", label=f"Final - {key.upper()} ({nm}) (A10)")
            contrasts[f"{key}_a9"] = pair(
                fa, arms[nm], "A9-est", label=f"A9-est Final - {key.upper()} ({nm})"
            )

    bars = bar_sequence(contrasts["bar1"], contrasts["r1"])

    def p_of(key: str) -> float | None:
        c = contrasts.get(key)
        return c.p if c is not None and c.valid else None

    family_p = {
        "Final - R2": p_of("r2"),
        "Final - R3": p_of("r3"),
        "A9-est Final - R1": p_of("r1_a9"),
        "DEVBOSS Final - REF": (
            devboss_out["pooled"]["p"]
            if devboss_out and devboss_out["valid"] and math.isfinite(devboss_out["pooled"]["p"])
            else None
        ),
    }
    family = S.holm(family_p, alpha=HOLM_ALPHA)
    for k in ("r2", "r3"):
        if k not in contrasts:
            notes.append(f"no {k.upper()} arm: its family member counts as p = 1")
    if devboss is None:
        notes.append("no DEVBOSS files: that family member counts as p = 1")

    return Read(
        panel=panel.name,
        prereg=pre,
        arms={n: arm_summary(a) for n, a in arms.items()},
        manifests=manifests,
        bars=bars,
        barr=bar_r(fa),
        contrasts=contrasts,
        family=family,
        devboss=devboss_out,
        notes=notes,
    )


def _pct(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "-"
    return f"{100 * x:+.2f}pp" if signed else f"{100 * x:.2f}%"


def _ci(ci: Any) -> str:
    if not ci:
        return "-"
    lo, hi = (ci["lo"], ci["hi"]) if isinstance(ci, dict) else ci
    return f"[{100 * lo:+.2f}, {100 * hi:+.2f}]pp"


def _p(p: float | None) -> str:
    return "-" if p is None else f"{p:.3g}"


def render(read: Read) -> str:
    """The printed report."""
    out = [f"=== score_test: {read.panel} read ==="]
    g = read.prereg["git"]
    out.append(
        f"PREREG {read.prereg['path']} sha256 {read.prereg['sha256'][:16]} "
        f"commit {g['commit'] or '-'} {'clean' if g['clean'] else 'NOT CLEAN'}"
    )
    out.append("\n-- arms (integrity checks passed) --")
    for name, a in read.arms.items():
        c = a["counts"]
        w = a["wheel"] or {}
        a10 = a["A10"]
        out.append(
            f"{name:>10}: seeds {a['seeds']}  rows {c['rows']}  errored {c['errored_in_shards']}"
            f"  retried {c['retried']}  missing after retry {c['missing_after_retry']}"
            f" (timeouts {c['timeout']})  capped {c['capped']}  src_hash {w.get('src_hash')}"
        )
        out.append(
            f"{'':>10}  WR {a10['wins']}/{a10['scored']} = {_pct(a10['rate_scored'])}"
            f"  A9-est {_pct(a['A9-est']['rate_scored'])}  manifest {a['manifest_sha256'][:16]}"
            f"  committed: {read.manifests[name].get('committed') or 'no'}"
        )
    out.append("\n-- bars (Bar1 -> Bar2 -> Bar3, stop at the first failure) --")
    for b in read.bars:
        out.append(
            f"{b['bar']:<15} {b['status']:<10} {b['contrast']}: Δ {_pct(b['delta'], True)}"
            f"  McNemar p {_p(b['mcnemar_p'])}  Newcombe95 {_ci(b['newcombe95'])}"
            f"  rule: {b['rule']}"
        )
    br = read.barr
    out.append(
        f"{'BarR':<15} WR(Final) {br['A10']['wins']}/{br['n']} = {_pct(br['A10']['rate'])}"
        f" Wilson95 [{_pct(br['A10']['wilson95'][0])}, {_pct(br['A10']['wilson95'][1])}]"
        f"  A9-est {_pct(br['A9-est']['rate'])}  "
        + "  ".join(f"{k}: {'yes' if v else 'no'}" for k, v in br["lines"].items())
    )
    out.append("\n-- contrasts (P2 pairing) --")
    for c in read.contrasts.values():
        d = c.to_dict()
        if c.table is None:
            out.append(f"{c.label}: no pairs; INVALID: {'; '.join(c.invalid)}")
            continue
        out.append(
            f"{c.label}: n {d['pairs']}  {_pct(d['wr_final'])} vs {_pct(d['wr_baseline'])}"
            f"  Δ {_pct(d['delta'], True)}  disc {d['final_only']}/{d['baseline_only']}"
            f" (d {d['discordance']:.3f})  p {_p(d['mcnemar_p'])}"
            f"  Newcombe95 {_ci(d['newcombe95'])}  bootstrap95 {_ci(d['bootstrap95'])}"
            f"  missing {d['missing_data_seeds']}/{d['max_missing_data_seeds']}"
            + ("" if c.valid else f"  INVALID: {'; '.join(c.invalid)}")
        )
    out.append(f"\n-- descriptive family (Holm, alpha {HOLM_ALPHA}) --")
    for name, h in read.family.items():
        out.append(
            f"{name:<22} p {_p(h['p'])}  Holm p {_p(h['p_holm'])}  "
            f"{'reject' if h['reject'] else 'not rejected'}"
        )
    if read.devboss:
        db = read.devboss
        pooled = db["pooled"]
        out.append(
            f"\n-- DEVBOSS Final - REF: {db['pairs']} pairs over {db['states']} states"
            f"  Δ {_pct(pooled['mean'], True)}  cluster95 {_ci(pooled)}  p {_p(pooled['p'])}"
            + ("" if db["valid"] else f"  INVALID: {'; '.join(db['invalid'])}")
        )
        for cell, cs in db["cells"].items():
            out.append(f"   {cell:<12} Δ {_pct(cs['mean'], True)}  cluster95 {_ci(cs)}")
    for n in read.notes:
        out.append(f"note: {n}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--evals", type=Path, default=Path.home() / "runs" / "e2e" / "evals")
    ap.add_argument("--panel", default="TEST", choices=sorted(P.RUNNER_PANELS))
    ap.add_argument("--final", required=True)
    ap.add_argument("--cstar", required=True)
    ap.add_argument("--r1", default="R1")
    ap.add_argument("--r2", default="R2")
    ap.add_argument("--r3", default=None)
    ap.add_argument(
        "--devboss",
        nargs=2,
        type=Path,
        metavar=("FINAL", "REF"),
        help="DEVBOSS result files (ppo.e2e.devboss play) for Final and REF",
    )
    ap.add_argument("--json", type=Path, default=None, help="write the full read as JSON")
    ap.add_argument("--reread-reason", default=None)
    args = ap.parse_args(argv)
    try:
        read = score(
            prereg=args.prereg,
            evals=args.evals.expanduser(),
            panel=args.panel,
            final=args.final,
            cstar=args.cstar,
            r1=args.r1,
            r2=args.r2,
            r3=args.r3,
            devboss=tuple(args.devboss) if args.devboss else None,
            argv=list(sys.argv if argv is None else argv),
            reread_reason=args.reread_reason,
        )
    except ReadStopped as e:
        print(f"score_test: READ STOPPED, no statistic computed: {e}", file=sys.stderr)
        return 2
    print(render(read))
    if args.json:
        args.json.write_text(json.dumps(read.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
