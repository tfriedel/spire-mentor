#!/usr/bin/env python3
"""e2e_driver.py: the campaign driver of the Ironclad e2e campaign (#1097, plan §8).

A 5-minute systemd user timer (``e2e-driver.timer``) runs ``e2e_driver.py
run``. Each pass reads every lane under ``<campaign>/lanes/``, executes only
the pre-registered rules below, and writes ``status.json``,
``campaign_state.json``, ``ALARM_<name>`` files and ``CAMPAIGN_LOG.md``. The
driver never decides D1-D11 and never clears a HUMAN alarm: the user decides,
the operator records the decision with ``ack``.

Rules (plan §8 Runbook, PREREG-0 P6; names in ``ppo.e2e.ops_contract``):

* HUMAN stops (``DONE`` plus the alarm): ``LR_CLOSED_FORM`` (an lr off its
  closed form by more than 1e-6 relative), ``WHEEL_MISMATCH`` (a stats row's
  ``wheel_mismatch``, the wheel stamp changed, or the pin's ``.so`` files
  changed), ``SOURCE_MISMATCH`` (``source_mismatch``, the source, vocab or
  schema stamp changed, or the lane's worktree moved or got dirty),
  ``GAME_VERSION`` (a stamp other than v0.111.0), ``F9_INVARIANT`` (any hit).
* OPERATOR alarms (the lane runs on): ``CAPS`` (over 0.1% capped fights or
  runs in the last 50 iterations), ``CONTRACT``, ``HEARTBEAT_STALE``,
  ``WATCHDOG_DOWN`` (10 min without a live watchdog and no ``DONE``),
  ``PIN_MISSING`` (it also writes ``PIN_REQUEST``), ``K2_READ``, ``K3_READ``,
  ``K4_READ``, ``E1_REVERT`` (E=1 and some type's KL over 0.03 or clipfrac
  over 0.3 for 20 iterations in a row), ``HELD_ROW_LEAK`` (``update.rows_overdue``
  above 0 in each of the last ``HELD_LEAK_ITERS`` = 7 iterations), ``DEV_JUMP``
  and ``DRIVER_ERROR``. The learner raises ``LC_RECHECK_FAIL`` itself (a
  learning check that fails after one passed); it shows in the status.
* Eval requests in ``<lane>/evals/queue/``: DEV-1000 EMA0.996 on every 250M
  milestone pin; DEV-2000 raw, EMA0.996 and EMA0.999 on every 1B pin (instead
  of that milestone's DEV-1000); DEV-1000 EMA0.996 on the K2 and K3 pins.
* Eval results in ``<lane>/evals/done/``: ``STALL_SIGNAL`` (HUMAN, the lane
  runs on) when DEV-2000 EMA0.996 is below 11% at 2B or 14% at 3B;
  ``DEV_JUMP`` when a DEV-1000 read is more than 5pp above the previous
  DEV-1000 read of its variant.
* Backstops while a learner runs without a live watchdog: ``RAM_STOP``
  (``EVALS_PAUSE``, then ``DONE``) and ``DISK_STOP`` (prune, then ``DONE``).

The paired gate verdicts (K2, K3 and K4 against the teacher pair and C*) need
the baseline arms' per-seed rows, so the operator reads them and records a
fail with ``raise``: ``K2_FAIL`` stops the lane; ``K3_NOGO``, ``K4_NOGO``,
``STALL_SIGNAL`` and ``SIM_EXPLOIT`` are HUMAN, and the lane runs on.

Commands::

    e2e_driver.py [--campaign DIR] [--now T] run
    e2e_driver.py init-lane LANE --worktree W --python PY [--pin DIR] --args-file F
                  [--learner-env F] [--hot-swap] [--note TEXT]
    e2e_driver.py ack NAME [--lane LANE] --note TEXT [--decision TEXT] [--mute-hours H]
    e2e_driver.py raise NAME --lane LANE --reason TEXT
    e2e_driver.py stop LANE --reason TEXT [--note TEXT]
    e2e_driver.py clear-done LANE --note TEXT [--reset-starts]
    e2e_driver.py note LANE|campaign TEXT...
    e2e_driver.py status

Every command except ``status`` holds ``driver.lock`` while it reads and
rewrites ``campaign_state.json``: a timer pass skips when the lock is taken,
and an operator command waits for it.

``--campaign`` defaults to ``$E2E_CAMPAIGN_DIR`` or ``~/runs/e2e``; ``--now``
(unix seconds or ISO 8601) is for tests and replays.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from ppo.e2e import ckpt  # noqa: E402
from ppo.e2e import ops_contract as C  # noqa: E402
from ppo.e2e import watchdog as W  # noqa: E402

DRIVER_VERSION = 1
WATCHDOG_DOWN_AFTER_S = 600.0
DRIVER_LOCK = "driver.lock"
LOCK_WAIT_S = 120.0  # an operator command waits this long for a timer pass
LOG_KINDS = {
    "start": "learner start",
    "exit": "learner exit",
    "crash": "crash",
    "stall": "stall",
    "alarm": "alarm",
    "pinned": "pinned",
    "refused": "refused",
    "done_grace": "DONE grace over",
    "evals_paused": "evals paused",
    "evals_resumed": "evals resumed",
    "starts_reset": "crash-loop count reset",
}
LOG_HEADER = (
    "# Ironclad e2e campaign log\n\n"
    "Append-only. The driver adds a line for each watchdog event and driver action;\n"
    "the operator adds a line for each decision (plan §8 Runbook).\n\n"
)


def utc(t: float) -> str:
    return C.utc_now_iso(t)


def parse_now(raw: str | None) -> float:
    if not raw:
        return time.time()
    try:
        return float(raw)
    except ValueError:
        pass
    t = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return t.timestamp()


def _git(worktree: str | os.PathLike, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(worktree), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def wheel_digest(pin: str | os.PathLike | None) -> str | None:
    """sha256 over the pin dir's ``.so`` files (relative path and content)."""
    if not pin or not Path(pin).is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(Path(pin).rglob("*.so")):
        h.update(str(f.relative_to(pin)).encode("utf-8") + b"\0")
        h.update(ckpt.sha256_file(f).encode("ascii"))
    return h.hexdigest()


# ── the campaign ─────────────────────────────────────────────────────────────


class Campaign:
    def __init__(self, root: Path, now: float, lim: C.Limits | None = None) -> None:
        self.root = root
        self.now = now
        self.lim = lim or C.Limits.from_env()
        state = C.read_json(root / C.CAMPAIGN_STATE)
        self.state: dict[str, Any] = state if isinstance(state, dict) else {}
        self.state.setdefault("lanes", {})
        self.state.setdefault("mutes", {})
        self.log_lines: list[str] = []

    def lanes(self) -> list[Path]:
        d = self.root / C.LANES_DIR
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.is_dir() and (p / C.LANE_ENV).is_file())

    def log(self, lane: str, text: str, t: float | None = None) -> None:
        self.log_lines.append(f"- {utc(self.now if t is None else t)} {lane}: {text}\n")

    def muted(self, lane: str, name: str) -> bool:
        until = self.state["mutes"].get(f"{lane}:{name}")
        return isinstance(until, (int, float)) and self.now < until

    def alarm(
        self, lane_dir: Path, name: str, reason: str, *, writer: str = "driver", **details: Any
    ) -> bool:
        """Raise ``name`` in the lane dir unless muted. An alarm that stops the
        lane (``stops_lane``) also writes ``DONE``; a mute never skips a stop."""
        lane = lane_dir.name
        spec = C.ALARMS_BY_NAME[name]
        if spec.severity == C.OPERATOR and not spec.stops_lane and self.muted(lane, name):
            return False
        new = not C.alarm_path(lane_dir, name).exists()
        C.write_alarm(lane_dir, name, writer=writer, reason=reason, lane=lane, **details)
        if spec.stops_lane:
            C.write_done(lane_dir, f"{writer}:{name}", by=writer, detail=reason)
        if new:
            self.log(lane, f"ALARM_{name} ({spec.severity}, {writer}): {reason}")
        return True

    def save(self) -> None:
        self.state["updated"] = utc(self.now)
        self.state["driver_version"] = DRIVER_VERSION
        C.atomic_write_text(
            self.root / C.CAMPAIGN_STATE, json.dumps(self.state, indent=1, sort_keys=True) + "\n"
        )
        if self.log_lines:
            path = self.root / C.CAMPAIGN_LOG
            new = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new:
                    f.write(LOG_HEADER)
                f.writelines(self.log_lines)
            self.log_lines = []


# ── rules on stats rows ──────────────────────────────────────────────────────


def lr_problem(row: dict) -> str | None:
    lr, cf = row.get("lr"), row.get("lr_closed_form")
    if not isinstance(lr, dict) or not isinstance(cf, dict):
        return None  # a CONTRACT problem, reported there
    for k in sorted(set(lr) | set(cf)):
        v, c = lr.get(k), cf.get(k)
        if not (C.is_num(v) and C.is_num(c)):
            return f"group {k}: lr {v!r}, closed form {c!r}"
        off = abs(v / c - 1.0) if c else abs(v)
        if off > C.LR_TOLERANCE:
            return f"group {k}: lr {v:.10g} vs closed form {c:.10g} (off by {off:.3g})"
    return None


def stamp_problem(row: dict, base: dict | None) -> tuple[str, str] | None:
    stamps = row.get("stamps")
    if not isinstance(stamps, dict):
        return None
    gv = stamps.get("game_version")
    if gv is not None and gv != C.GAME_VERSION:
        return "GAME_VERSION", f"stamp game_version {gv!r}, expected {C.GAME_VERSION}"
    if not base:
        return None
    if stamps.get("wheel") != base.get("wheel"):
        return "WHEEL_MISMATCH", f"wheel stamp {stamps.get('wheel')!r} != {base.get('wheel')!r}"
    for k in ("py_src", "vocab_sha256", "schema_sha256"):
        if stamps.get(k) != base.get(k):
            return "SOURCE_MISMATCH", f"{k} stamp {stamps.get(k)!r} != {base.get(k)!r}"
    return None


def human_rules(st: dict, rows: list[dict]) -> list[tuple[str, str, dict]]:
    """HUMAN findings in the rows since the last pass (each alarm once per pass)."""
    out: dict[str, tuple[str, str, dict]] = {}
    deployed = float(st.get("deployed_unix", 0.0))
    for row in rows:
        where = {"it": row.get("it"), "decisions": row.get("decisions")}
        if row.get("wheel_mismatch"):
            out.setdefault("WHEEL_MISMATCH", ("WHEEL_MISMATCH", str(row["wheel_mismatch"]), where))
        if row.get("source_mismatch"):
            out.setdefault(
                "SOURCE_MISMATCH", ("SOURCE_MISMATCH", str(row["source_mismatch"]), where)
            )
        lr = lr_problem(row)
        if lr:
            out.setdefault("LR_CLOSED_FORM", ("LR_CLOSED_FORM", lr, where))
        f9 = row.get("f9")
        if isinstance(f9, dict) and C.is_num(f9.get("hits")) and f9["hits"] > C.F9_MAX_HITS:
            out.setdefault(
                "F9_INVARIANT", ("F9_INVARIANT", f"{f9['hits']} F9 hit(s) in one iteration", where)
            )
        if isinstance(row.get("stamps"), dict):
            key = (
                "stamps" if C.is_num(row.get("time")) and row["time"] >= deployed else "prev_stamps"
            )
            if key == "stamps" and not st.get("stamps"):
                st["stamps"] = {k: row["stamps"].get(k) for k in C.STAMP_FIELDS}
            hit = stamp_problem(row, st.get(key))
            if hit:
                out.setdefault(hit[0], (hit[0], hit[1], where))
    return list(out.values())


def deploy_rules(st: dict) -> list[tuple[str, str, dict]]:
    """The lane's worktree and pin against what ``init-lane`` recorded."""
    out = []
    dep = st.get("deploy")
    if not isinstance(dep, dict):
        return out
    wt = dep.get("worktree")
    if wt:
        head = _git(wt, "rev-parse", "HEAD")
        if head is not None and head != dep.get("sha"):
            out.append(
                ("SOURCE_MISMATCH", f"worktree {wt} moved to {head} from {dep.get('sha')}", {})
            )
        dirty = _git(wt, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            out.append(("SOURCE_MISMATCH", f"worktree {wt} has uncommitted changes", {}))
    if dep.get("pin") and dep.get("wheel_digest"):
        now = wheel_digest(dep["pin"])
        if now != dep["wheel_digest"]:
            out.append(("WHEEL_MISMATCH", f"the .so files under {dep['pin']} changed", {}))
    return out


def caps_problem(rows: list[dict]) -> str | None:
    tot = {"fights": 0, "fights_capped": 0, "runs": 0, "runs_capped": 0}
    for row in rows[-C.CAPS_WINDOW_ITERS :]:
        caps = row.get("caps")
        if isinstance(caps, dict):
            for k in tot:
                if C.is_num(caps.get(k)):
                    tot[k] += caps[k]
    out = []
    for kind, floor in (("fights", C.CAPS_MIN_FIGHTS), ("runs", C.CAPS_MIN_RUNS)):
        n, capped = tot[kind], tot[f"{kind}_capped"]
        if n >= floor and capped / n > C.CAPS_MAX_FRAC:
            out.append(f"{capped}/{n} {kind} capped ({capped / n:.2%})")
    return "; ".join(out) or None


def _worst(v: Any) -> float | None:
    """The largest number in a per-type map, or the number itself."""
    if C.is_num(v):
        return float(v)
    if isinstance(v, dict):
        nums = [float(x) for x in v.values() if C.is_num(x)]
        return max(nums) if nums else None
    return None


def e1_problem(rows: list[dict]) -> str | None:
    """Revert rule (a) of the E=1 switch: the last 20 rows all ran E=1 and each
    had some type's KL or clipfrac over its bound."""
    last = rows[-C.E1_WINDOW_ITERS :]
    if len(last) < C.E1_WINDOW_ITERS or any(r.get("epochs") != 1 for r in last):
        return None
    for r in last:
        kl, cf = _worst(r.get("kl")), _worst(r.get("clipfrac"))
        if not (
            (kl is not None and kl > C.E1_KL_MAX) or (cf is not None and cf > C.E1_CLIPFRAC_MAX)
        ):
            return None
    kl = max(_worst(r.get("kl")) or 0.0 for r in last)
    cf = max(_worst(r.get("clipfrac")) or 0.0 for r in last)
    return (
        f"E=1 and KL > {C.E1_KL_MAX} or clipfrac > {C.E1_CLIPFRAC_MAX} for {len(last)} "
        f"iterations (its {last[0].get('it')}-{last[-1].get('it')}; max KL {kl:.3g}, "
        f"max clipfrac {cf:.3g})"
    )


def held_leak_problem(rows: list[dict]) -> str | None:
    """The held-row leak rule: the last `HELD_LEAK_ITERS` rows are consecutive
    iterations and each has ``update.rows_overdue`` above 0. A row without the
    field, or a gap or a step back in ``it`` (a restart, a rollback), breaks the
    run."""
    last = rows[-C.HELD_LEAK_ITERS :]
    if len(last) < C.HELD_LEAK_ITERS:
        return None
    its, overdue = [], []
    for r in last:
        u, it = r.get("update"), r.get("it")
        n = u.get("rows_overdue") if isinstance(u, dict) else None
        if not (C.is_num(n) and n > 0 and isinstance(it, int)):
            return None
        its.append(it)
        overdue.append(n)
    if its != list(range(its[0], its[0] + len(its))):
        return None
    return (
        f"update.rows_overdue above 0 for {len(last)} iterations (its {its[0]}-{its[-1]}; "
        f"now {overdue[-1]:g}, max {max(overdue):g})"
    )


# ── one lane ─────────────────────────────────────────────────────────────────


def _alive(rec: Any) -> bool:
    return C.record_alive(rec)


def _watchdog_alive(sd: ckpt.SaveDir) -> tuple[bool, dict | None]:
    rec = C.read_json(sd.p(C.WATCHDOG_STATE))
    wd = rec.get("watchdog") if isinstance(rec, dict) else None
    return _alive(wd), rec if isinstance(rec, dict) else None


def _milestone_variants(m: int) -> tuple[str, tuple[str, ...]]:
    if m % C.DEV2000_EVERY == 0:
        return "DEV-2000", C.EVAL_VARIANTS_DEV2000
    return "DEV-1000", C.EVAL_VARIANTS_DEV1000


def queue_eval(
    camp: Campaign,
    sd: ckpt.SaveDir,
    st: dict,
    pin: ckpt.ManifestRow,
    *,
    panel: str,
    variant: str,
    reason: str,
) -> bool:
    """Write one eval request into ``<lane>/evals/queue/``, once per pinned state
    (by sha256), panel and variant: a gate read on a milestone pin reuses the
    milestone's request, and a new lineage's pin at the same milestone gets its own."""
    key = f"{pin.sha256}:{panel}:{variant}"
    queued = st.setdefault("queued", {})
    if key in queued:
        return False
    spec = C.EVAL_PANELS[panel]
    rid = f"{sd.root.name}_d{pin.decisions:013d}_{panel}_{variant}_{reason}"
    req = {
        "id": rid,
        "lane": sd.root.name,
        "state": str(pin.path(sd)),
        "sha256": pin.sha256,
        "decisions": pin.decisions,
        "panel": panel,
        "n": spec["n"],
        "seed_start": spec["seed_start"],
        "variant": variant,
        "nice": C.EVAL_NICE,
        "max_procs": C.EVAL_MAX_PROCS,
        "reason": reason,
        "requested_by": "driver",
        "time": utc(camp.now),
    }
    C.atomic_write_text(sd.p(C.EVAL_QUEUE) / f"{rid}.json", json.dumps(req, indent=1) + "\n")
    queued[key] = rid
    camp.log(sd.root.name, f"queued {panel} {variant} on {pin.name} ({reason})")
    return True


def _request_pin(camp: Campaign, sd: ckpt.SaveDir, reason: str, why: str) -> bool:
    path = sd.root / C.PIN_REQUEST
    if path.exists():
        return False
    rec = {"by": "driver", "reason": reason, "why": why, "time": utc(camp.now)}
    C.atomic_write_text(path, json.dumps(rec) + "\n")
    camp.log(sd.root.name, f"wrote PIN_REQUEST ({reason}: {why})")
    return True


def drop_abandoned_requests(camp: Campaign, sd: ckpt.SaveDir, pins: list) -> None:
    """Queued (not yet claimed) driver requests whose state a rollback abandoned."""
    live = {str(p.path(sd)) for p in pins}
    qdir = sd.p(C.EVAL_QUEUE)
    if not qdir.is_dir():
        return
    for f in sorted(qdir.glob("*.json")):
        req = C.read_json(f)
        stale = isinstance(req, dict) and req.get("requested_by") == "driver"
        if stale and req.get("state") not in live:
            f.unlink(missing_ok=True)
            camp.log(sd.root.name, f"dropped eval request {f.stem}: its state is not pinned")


def pin_and_eval_rules(camp: Campaign, sd: ckpt.SaveDir, st: dict, decisions: int) -> None:
    pins = ckpt.pinned_states(sd)
    drop_abandoned_requests(camp, sd, pins)
    # Milestone pins and their evals.
    missing = []
    start = int(st.get("pin_check_from", 0))
    m = C.PIN_EVERY
    while m + C.PIN_MISSING_AFTER <= decisions:
        if m >= start:
            hits = sorted(
                (p for p in pins if m <= p.decisions < m + C.PIN_EVERY),
                key=lambda p: (p.reason != "milestone", p.decisions),
            )
            if not hits:
                missing.append(f"{m:,} milestone")
            else:
                panel, variants = _milestone_variants(m)
                for v in variants:
                    queue_eval(camp, sd, st, hits[0], panel=panel, variant=v, reason="milestone")
        m += C.PIN_EVERY
    # K-gates: flag once, with the gate's pinned state.
    flags = st.setdefault("flags", {})
    requests = st.setdefault("gate_requests", {})
    names = {p.name for p in pins}
    for g in C.K_GATES:
        if g.name in flags and flags[g.name].get("pin") not in names:
            # A rollback abandoned the gate's pinned state: the gate re-arms.
            camp.log(sd.root.name, f"{g.name} re-armed: {flags[g.name].get('pin')} is not pinned")
            flags.pop(g.name)
            requests.pop(g.name, None)
        if g.name in flags:
            continue
        by_dec = decisions >= g.at_decisions
        by_time = g.backstop is not None and camp.now >= g.backstop.timestamp()
        if not (by_dec or by_time):
            continue
        tag = g.name.lower()
        if by_dec:
            cand = [p for p in pins if p.reason == tag] or [
                p for p in pins if g.at_decisions <= p.decisions
            ]
            if not cand:
                if decisions >= g.at_decisions + C.PIN_MISSING_AFTER:
                    missing.append(f"{g.name} gate")
                continue
            gate_pin = cand[0]
        else:
            assert g.backstop is not None  # by_time without by_dec implies a backstop
            req = requests.get(g.name)
            if not isinstance(req, dict):
                requests[g.name] = {"time": utc(camp.now), "decisions": decisions}
                _request_pin(camp, sd, tag, f"{g.name} backstop {g.backstop.isoformat()}")
                continue
            newer = [p for p in pins if p.decisions >= int(req["decisions"])]
            if not newer:
                continue
            gate_pin = newer[0]
        flags[g.name] = {
            "time": utc(camp.now),
            "by": "decisions" if by_dec else "backstop",
            "pin": gate_pin.name,
            "decisions": gate_pin.decisions,
        }
        if g.eval_panel:
            for v in C.EVAL_VARIANTS_DEV1000:
                queue_eval(camp, sd, st, gate_pin, panel=g.eval_panel, variant=v, reason=tag)
        camp.alarm(
            sd.root, f"{g.name}_READ",
            f"{g.name} due ({flags[g.name]['by']}); pinned state {gate_pin.name}; read: {g.read}",
            pin=str(gate_pin.path(sd)), decisions=gate_pin.decisions,
        )  # fmt: skip
    if missing:
        _request_pin(camp, sd, "request", "missing: " + ", ".join(missing))
        camp.alarm(sd.root, "PIN_MISSING", "no pinned state for " + ", ".join(missing))


def eval_results(sd: ckpt.SaveDir, pins: list) -> list[dict]:
    """Finished evals of states still pinned (a rollback's abandoned lineage
    drops out), each with ``wr``, sorted by decisions."""
    ddir = sd.p(C.EVAL_DONE)
    live = {p.sha256 for p in pins}
    out = []
    for f in sorted(ddir.glob("*.json")) if ddir.is_dir() else []:
        r = C.read_json(f)
        if not isinstance(r, dict) or r.get("sha256") not in live:
            continue
        wins, runs, n = r.get("wins"), r.get("runs"), r.get("n")
        if not all(C.is_num(x) for x in (wins, runs, n, r.get("decisions"))):
            continue
        if runs != n or runs <= 0 or not 0 <= wins <= runs:
            continue  # partial or broken: the rules read complete panels only
        out.append({**r, "id": r.get("id") or f.stem, "wr": wins / runs})
    return sorted(out, key=lambda r: (r["decisions"], str(r["id"])))


def eval_result_rules(camp: Campaign, sd: ckpt.SaveDir, st: dict) -> list[dict]:
    """The stall floors (plan §1.4) and the DEV jump (plan §12), once per result."""
    pins = ckpt.pinned_states(sd)
    results = eval_results(sd, pins)
    seen = st.setdefault("evals_read", [])
    prev: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (str(r.get("panel")), str(r.get("variant")))
        before = prev.get(key)
        prev[key] = r
        if r["id"] in seen:
            continue
        seen.append(r["id"])
        camp.log(
            sd.root.name,
            f"eval {r['panel']} {r['variant']} at {r['decisions']:,}: "
            f"{r['wins']}/{r['runs']} = {r['wr']:.1%}",
        )
        if key == (C.STALL_PANEL, C.STALL_VARIANT):
            for m, floor in C.STALL_FLOORS:
                if m <= r["decisions"] < m + C.PIN_EVERY and r["wr"] < floor:
                    camp.alarm(
                        sd.root, "STALL_SIGNAL",
                        f"{r['panel']} {r['variant']} {r['wr']:.1%} < {floor:.0%} at "
                        f"{r['decisions']:,} decisions: diagnose <= 6 h, training continues",
                        result=r["id"],
                    )  # fmt: skip
        jump = before is not None and r["wr"] - before["wr"] > C.DEV_JUMP
        if jump and key[0] == C.DEV_JUMP_PANEL:
            camp.alarm(
                sd.root, "DEV_JUMP",
                f"{r['panel']} {r['variant']} {before['wr']:.1%} -> {r['wr']:.1%} "
                f"({before['decisions']:,} -> {r['decisions']:,})",
                result=r["id"],
            )  # fmt: skip
    return [
        {k: r.get(k) for k in ("panel", "variant", "decisions", "wins", "runs", "wr")}
        for r in results[-6:]
    ]


def _by(rec: Any) -> str | None:
    return rec.get("by") if isinstance(rec, dict) else None


def backstop_rules(camp: Campaign, sd: ckpt.SaveDir, learner_alive: bool, wd_alive: bool) -> None:
    """RAM and disk while a learner runs without a live watchdog."""
    lim = camp.lim
    pause = C.read_json(sd.root / C.EVALS_PAUSE)
    ram = C.meminfo_available_gb(lim.meminfo)
    if wd_alive or not learner_alive:
        return
    if ram is not None and ram < lim.ram_stop_gb:
        if pause is None:
            rec = {"by": "driver", "time": utc(camp.now), "ram_gb": round(ram, 2)}
            C.atomic_write_text(sd.root / C.EVALS_PAUSE, json.dumps(rec) + "\n")
            camp.log(sd.root.name, f"wrote EVALS_PAUSE (MemAvailable {ram:.1f} GB, no watchdog)")
        elif isinstance(pause, dict) and C.utc_now_iso(camp.now) > pause.get("time", ""):
            camp.alarm(
                sd.root, "RAM_STOP",
                f"MemAvailable {ram:.1f} GB with the evals stopped and no live watchdog",
            )  # fmt: skip
    elif ram is not None and ram >= lim.ram_resume_gb and _by(pause) == "driver":
        (sd.root / C.EVALS_PAUSE).unlink(missing_ok=True)
        camp.log(sd.root.name, "removed the driver's EVALS_PAUSE")
    v = ckpt.disk_check(sd, prune_below_gb=lim.disk_prune_gb, stop_below_gb=lim.disk_stop_gb)
    if v.action == "stop":
        camp.alarm(sd.root, "DISK_STOP", f"{v.free_gb:.1f} GB free after pruning (no watchdog)")
    elif v.deleted:
        camp.log(sd.root.name, f"pruned {len(v.deleted)} file(s) ({v.free_gb:.1f} GB free)")


def tail_events(camp: Campaign, sd: ckpt.SaveDir, st: dict) -> None:
    through = float(st.get("events_through", 0.0))
    last = through
    for e in C.iter_jsonl(C.tail_lines(sd.p(C.EVENTS), 4 << 20)):
        t = e.get("time")
        if not C.is_num(t) or t <= through:
            continue
        last = max(last, t)
        kind = e.get("kind")
        if kind not in LOG_KINDS:
            continue
        rest = {k: v for k, v in e.items() if k not in ("time", "utc", "kind")}
        text = ", ".join(f"{k}={v}" for k, v in rest.items() if v not in (None, "", [], {}))
        camp.log(sd.root.name, f"watchdog {LOG_KINDS[kind]}: {text}"[:400], t=t)
    st["events_through"] = last


def lane_pass(camp: Campaign, lane_dir: Path) -> dict[str, Any]:
    sd = ckpt.SaveDir(lane_dir)
    lane = lane_dir.name
    st = camp.state["lanes"].setdefault(lane, {})
    through = float(st.get("rows_through", st.get("deployed_unix", 0.0)))
    rows = [r for r in C.rows_since(sd.p(C.STATS), through) if r["time"] > through]
    tail = list(C.iter_jsonl(C.tail_lines(sd.p(C.STATS), 1 << 20)))
    newest = tail[-1] if tail else {}
    if rows:
        st["rows_through"] = max(r["time"] for r in rows)
    newest_decisions = newest.get("decisions")
    decisions = newest_decisions if C.is_num(newest_decisions) else 0
    done = C.read_json(sd.root / C.DONE)
    # HUMAN stops.
    for name, reason, where in human_rules(st, rows) + deploy_rules(st):
        camp.alarm(lane_dir, name, reason, **{k: v for k, v in where.items() if v is not None})
    # OPERATOR alarms.
    problems = []
    for r in rows:
        problems += [f"stats it={r.get('it')}: {p}" for p in C.check_stats_row(r)]
    tp = list(C.iter_jsonl(C.tail_lines(sd.p(C.THROUGHPUT), 1 << 20)))
    if tp and rows:
        problems += [f"throughput: {p}" for p in C.check_throughput_row(tp[-1])]
    if problems:
        camp.alarm(lane_dir, "CONTRACT", "; ".join(problems[:5]), problems=len(problems))
    caps = caps_problem(tail)
    if caps and rows:
        camp.alarm(lane_dir, "CAPS", caps)
    e1 = e1_problem(tail)
    if e1 and rows:
        camp.alarm(lane_dir, "E1_REVERT", e1)
    leak = held_leak_problem(tail)
    if leak and rows:
        camp.alarm(lane_dir, "HELD_ROW_LEAK", leak)
    learner_rec = C.read_json(sd.root / C.LEARNER_PID)
    learner_alive = _alive(learner_rec)
    wd_alive, wd_rec = _watchdog_alive(sd)
    hb = C.read_json(sd.root / C.HEARTBEAT)
    hb_age = camp.now - hb["time"] if isinstance(hb, dict) and C.is_num(hb.get("time")) else None
    if learner_alive and (hb_age is None or hb_age > C.HEARTBEAT_STALE_S):
        age = "no heartbeat" if hb_age is None else f"heartbeat {hb_age / 60:.0f} min old"
        camp.alarm(lane_dir, "HEARTBEAT_STALE", f"{age} while learner {learner_rec['pid']} runs")
    if done is None and not wd_alive:
        since = st.setdefault("watchdog_down_since", camp.now)
        if camp.now - since >= WATCHDOG_DOWN_AFTER_S:
            camp.alarm(
                lane_dir, "WATCHDOG_DOWN",
                f"no live watchdog for {(camp.now - since) / 60:.0f} min and no DONE"
                + (f"; learner {learner_rec['pid']} runs" if learner_alive else ""),
            )  # fmt: skip
    else:
        st.pop("watchdog_down_since", None)
    pin_and_eval_rules(camp, sd, st, int(decisions))
    reads = eval_result_rules(camp, sd, st)
    backstop_rules(camp, sd, learner_alive, wd_alive)
    tail_events(camp, sd, st)
    # Status.
    last = ckpt.last_pinned(sd)
    tp_hour = [r["rate"] for r in tp if C.is_num(r.get("time")) and camp.now - r["time"] <= 3600
               and C.is_num(r.get("rate"))]  # fmt: skip
    usage = ckpt.usage(sd)
    nxt = next((g for g in C.K_GATES if g.name not in st.get("flags", {})), None)
    st["last_pinned"] = last.name if last else None
    st["next_gate"] = nxt.name if nxt else None
    starts = W.read_starts(sd)
    return {
        "lane": lane,
        "decisions": decisions,
        "it": newest.get("it"),
        "row_age_s": round(camp.now - newest["time"], 1) if C.is_num(newest.get("time")) else None,
        "dec_per_s": newest.get("dec_per_s"),
        "rate_1h": round(sum(tp_hour) / len(tp_hour), 1) if tp_hour else None,
        "learner_alive": learner_alive,
        "learner_pid": learner_rec.get("pid") if isinstance(learner_rec, dict) else None,
        "watchdog_alive": wd_alive,
        "watchdog_pid": ((wd_rec or {}).get("watchdog") or {}).get("pid"),
        "heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
        "done": done,
        "evals_pause": C.read_json(sd.root / C.EVALS_PAUSE),
        "pin_request": (sd.root / C.PIN_REQUEST).exists(),
        "last_pinned": {"name": last.name, "decisions": last.decisions, "time": last.time}
        if last
        else None,
        "next_gate": {
            "name": nxt.name,
            "at": nxt.at_decisions,
            "backstop": nxt.backstop.isoformat() if nxt.backstop else None,
        }
        if nxt
        else None,
        "flags": st.get("flags", {}),
        "seed_offset": W.seed_offset(sd),
        "starts_2h": sum(1 for t in starts if camp.now - t < camp.lim.starts_window_s),
        "disk_free_gb": round(ckpt.disk_free_bytes(sd) / 1e9, 2),
        "usage_gb": {k: round(v / 1e9, 3) for k, v in usage.items()},
        "evals": {
            k: len(list(sd.p(rel).glob("*.json"))) if sd.p(rel).is_dir() else 0
            for k, rel in (
                ("queue", C.EVAL_QUEUE),
                ("running", C.EVAL_RUNNING),
                ("done", C.EVAL_DONE),
            )
        },
        "eval_reads": reads,
        "deploy": st.get("deploy"),
        "stamps": st.get("stamps"),
        "alarms": [_alarm_summary(a) for a in C.list_alarms(lane_dir)],
    }


def _alarm_summary(a: dict) -> dict:
    return {k: a.get(k) for k in ("name", "severity", "lane", "reason", "time", "count", "path")}


def run_pass(camp: Campaign) -> dict[str, Any]:
    status: dict[str, Any] = {
        "time": camp.now,
        "utc": utc(camp.now),
        "host": socket.gethostname(),
        "driver_version": DRIVER_VERSION,
        "ram_gb": C.meminfo_available_gb(camp.lim.meminfo),
        "lanes": {},
        "errors": [],
    }
    for lane_dir in camp.lanes():
        try:
            status["lanes"][lane_dir.name] = lane_pass(camp, lane_dir)
        except Exception as exc:  # one broken lane must not blind the others
            tb = traceback.format_exc()
            status["errors"].append(f"{lane_dir.name}: {exc!r}")
            C.write_alarm(
                camp.root, "DRIVER_ERROR", writer="driver", reason=f"{lane_dir.name}: {exc!r}",
                traceback=tb[-4000:],
            )  # fmt: skip
    alarms = [_alarm_summary(a) for a in C.list_alarms(camp.root)]
    for ln in status["lanes"].values():
        alarms += ln["alarms"]
    status["alarms"] = alarms
    sev = {a["severity"] for a in alarms}
    status["worst"] = C.HUMAN if C.HUMAN in sev else C.OPERATOR if sev else "OK"
    status["summary"] = summary_line(status)
    C.atomic_write_text(camp.root / C.STATUS, json.dumps(status, indent=1, default=str) + "\n")
    camp.save()
    return status


def summary_line(status: dict) -> str:
    parts = [status["worst"]]
    for name, ln in sorted(status["lanes"].items()):
        rate = ln.get("rate_1h") or ln.get("dec_per_s")
        bits = [f"{name} {ln['decisions'] / 1e9:.3f}B"]
        if rate:
            bits.append(f"{rate / 1e3:.1f}k/s")
        if ln.get("done"):
            bits.append(f"DONE({ln['done'].get('reason')})")
        elif not ln.get("learner_alive"):
            bits.append("learner down")
        if ln.get("next_gate"):
            bits.append(f"next {ln['next_gate']['name']}")
        parts.append(" ".join(bits))
    names = sorted({f"{a['name']}({a.get('lane') or '-'})" for a in status["alarms"]})
    if names:
        parts.append("alarms: " + ", ".join(names))
    return " | ".join(parts)


# ── commands ─────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def driver_lock(root: Path, *, wait_s: float) -> Iterator[bool]:
    """Hold ``driver.lock`` while a command reads and rewrites
    ``campaign_state.json``, so a timer pass and an operator command never
    overwrite each other's update. Yields False if it is still held after ``wait_s``."""
    import fcntl  # the driver runs on Linux; the module imports anywhere

    root.mkdir(parents=True, exist_ok=True)
    with open(root / DRIVER_LOCK, "a", encoding="utf-8") as lock:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.5)
        yield True


def cmd_run(camp: Campaign) -> int:
    try:
        status = run_pass(camp)
    except Exception as exc:
        C.write_alarm(
            camp.root, "DRIVER_ERROR", writer="driver", reason=repr(exc),
            traceback=traceback.format_exc()[-4000:],
        )  # fmt: skip
        err = {"time": camp.now, "utc": utc(camp.now), "worst": C.OPERATOR, "error": repr(exc),
               "summary": f"OPERATOR | driver error: {exc!r}", "lanes": {}, "alarms": []}  # fmt: skip
        C.atomic_write_text(camp.root / C.STATUS, json.dumps(err, indent=1) + "\n")
        raise
    print(status["summary"])
    return 0


def cmd_init_lane(camp: Campaign, a: argparse.Namespace) -> int:
    lane_dir = camp.root / C.LANES_DIR / a.lane
    sd = ckpt.SaveDir(lane_dir)
    wt = Path(a.worktree).expanduser().resolve()
    head = _git(wt, "rev-parse", "HEAD")
    if head is None:
        return _fail(f"{wt} is not a git worktree")
    if _git(wt, "status", "--porcelain", "--untracked-files=no"):
        return _fail(f"{wt} has uncommitted changes; deploy only clean commits")
    if not (wt / "sts2-pycli" / "scripts" / "e2e_watchdog.sh").is_file():
        return _fail(f"{wt} has no sts2-pycli/scripts/e2e_watchdog.sh")
    py = Path(a.python).expanduser()
    py = py.parent.resolve() / py.name
    if not os.access(py, os.X_OK):
        return _fail(f"no python at {py}")
    pin = Path(a.pin).expanduser().resolve() if a.pin else None
    if pin is not None and not pin.is_dir():
        return _fail(f"no pin dir at {pin}")
    used = sd.state.exists() or sd.p(C.STATS).exists()
    if used and not a.hot_swap:
        return _fail(f"lane {a.lane} already ran; pass --hot-swap to redeploy it")
    if a.hot_swap:
        if C.record_alive(C.read_json(sd.root / C.LEARNER_PID)) or _watchdog_alive(sd)[0]:
            return _fail(f"lane {a.lane} still runs: systemctl --user stop e2e-lane@{a.lane}")
        if sd.state.exists():
            dec = ckpt.state_decisions(sd)
            if dec is None:
                return _fail("state.pt has no state.json sidecar; pin it by hand first")
            p = ckpt.pin(sd, decisions=dec, reason="hot_swap", src=sd.state, note=f"deploy {head}")
            camp.log(a.lane, f"pinned {p.name} for the hot-swap")
    lane_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "E2E_WORKTREE": str(wt),
        "E2E_PY": str(py),
        "E2E_REQUIRE_CLEAN": "1",
    }
    if pin is not None:
        env["E2E_PIN"] = str(pin)
    text = f"# written by e2e_driver.py init-lane at {utc(camp.now)}\n"
    text += "".join(f"{k}={v}\n" for k, v in env.items())
    C.atomic_write_text(lane_dir / C.LANE_ENV, text)
    C.atomic_write_text(lane_dir / C.LANE_ARGS, Path(a.args_file).read_text(encoding="utf-8"))
    if a.learner_env:
        C.atomic_write_text(
            lane_dir / C.LANE_LEARNER_ENV, Path(a.learner_env).read_text(encoding="utf-8")
        )
    st = camp.state["lanes"].setdefault(a.lane, {})
    if st.get("deploy"):
        st.setdefault("deploys", []).append(st["deploy"])
    st["deploy"] = {
        "sha": head,
        "worktree": str(wt),
        "python": str(py),
        "pin": str(pin) if pin else None,
        "wheel_digest": wheel_digest(pin),
        "time": utc(camp.now),
        "hot_swap": bool(a.hot_swap),
        "note": a.note,
    }
    st["deployed_unix"] = camp.now
    st["prev_stamps"] = st.get("stamps")
    st["stamps"] = None
    st.setdefault("rows_through", camp.now if not used else st.get("rows_through", 0.0))
    camp.log(
        a.lane,
        f"deployed {head[:12]} from {wt} (pin {pin}){' as a hot-swap' if a.hot_swap else ''}",
    )
    camp.save()
    print(f"lane {a.lane} ready: systemctl --user start e2e-lane@{a.lane}")
    return 0


def _fail(msg: str) -> int:
    print(f"e2e_driver: {msg}", file=sys.stderr)
    return 2


def cmd_ack(camp: Campaign, a: argparse.Namespace) -> int:
    where = camp.root / C.LANES_DIR / a.lane if a.lane else camp.root
    path = C.alarm_path(where, a.name)
    rec = C.read_json(path)
    if not path.exists():
        return _fail(f"no {path}")
    rec = rec if isinstance(rec, dict) else {"name": a.name}
    spec = C.ALARMS_BY_NAME.get(a.name)
    severity = rec.get("severity") or (spec.severity if spec else C.HUMAN)
    if severity == C.HUMAN and not a.decision:
        return _fail("a HUMAN alarm is acknowledged only with the user's decision: --decision TEXT")
    if a.mute_hours and (severity == C.HUMAN or (spec is not None and spec.stops_lane)):
        return _fail("HUMAN alarms and alarms that stop the lane cannot be muted")
    rec["ack"] = {
        "time": utc(camp.now),
        "by": os.environ.get("USER", ""),
        "note": a.note,
        "decision": a.decision,
    }
    acked = camp.root / C.ACKED_DIR
    acked.mkdir(parents=True, exist_ok=True)
    dst = (
        acked
        / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(camp.now))}_{a.lane or 'campaign'}_{a.name}.json"
    )
    C.atomic_write_text(dst, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    path.unlink()
    if a.mute_hours:
        camp.state["mutes"][f"{a.lane or 'campaign'}:{a.name}"] = camp.now + a.mute_hours * 3600
    text = f"acked ALARM_{a.name}: {a.note}"
    if a.decision:
        text += f" (decision: {a.decision})"
    if a.mute_hours:
        text += f" (muted {a.mute_hours:g} h)"
    camp.log(a.lane or "campaign", text)
    camp.save()
    print(f"acknowledged {path.name} -> {dst}")
    return 0


def cmd_raise(camp: Campaign, a: argparse.Namespace) -> int:
    """The operator's verdicts the driver cannot compute (K2 fail, K3 and K4
    NO-GO, a stall read by hand, a found sim exploit), as alarms."""
    spec = C.ALARMS_BY_NAME.get(a.name)
    allowed = sorted(x.name for x in C.ALARMS if "operator" in x.writer)
    if spec is None or a.name not in allowed:
        return _fail(f"the operator raises only {', '.join(allowed)}")
    lane_dir = _lane_dir(camp, a.lane)
    if lane_dir is None:
        return _fail(f"no lane {a.lane} under {camp.root / C.LANES_DIR}")
    by = os.environ.get("USER", "") or "operator"
    camp.alarm(lane_dir, a.name, a.reason, writer="operator", raised_by=by)
    camp.save()
    if spec.stops_lane:
        what = "DONE written: the lane stops at the learner's next iteration"
    else:
        what = "the lane runs on"
    print(f"raised ALARM_{a.name} [{spec.severity}] on {a.lane}: {what}")
    return 0


def _lane_dir(camp: Campaign, lane: str) -> Path | None:
    d = camp.root / C.LANES_DIR / lane
    return d if (d / C.LANE_ENV).is_file() else None


def cmd_stop(camp: Campaign, a: argparse.Namespace) -> int:
    """Write ``DONE``: the learner saves ``state.pt`` and exits at its next
    iteration (the watchdog stops it if it still runs after ``done_grace_s``),
    and the unit stays down, after a reboot too, until ``clear-done``."""
    lane_dir = _lane_dir(camp, a.lane)
    if lane_dir is None:
        return _fail(f"no lane {a.lane} under {camp.root / C.LANES_DIR}")
    old = C.read_json(lane_dir / C.DONE)
    if old is not None:
        print(f"DONE is already there: {old}")
        return 0
    C.write_done(lane_dir, a.reason, by="operator", note=a.note)
    camp.log(a.lane, f"operator wrote DONE ({a.reason}): {a.note}")
    camp.save()
    print(f"wrote {lane_dir / C.DONE}: the learner stops at its next iteration")
    return 0


def cmd_clear_done(camp: Campaign, a: argparse.Namespace) -> int:
    """Archive ``DONE`` so the next start runs. Refused while the learner runs
    or while an alarm that stops the lane (or an unknown one) is open: ack it
    first, a HUMAN one with the user's decision. HUMAN alarms under which
    training continues (K3_NOGO, K4_NOGO, STALL_SIGNAL, SIM_EXPLOIT) don't block."""
    lane_dir = _lane_dir(camp, a.lane)
    if lane_dir is None:
        return _fail(f"no lane {a.lane} under {camp.root / C.LANES_DIR}")
    done = lane_dir / C.DONE
    if not done.exists():
        return _fail(f"{done} does not exist")
    if C.record_alive(C.read_json(lane_dir / C.LEARNER_PID)):
        return _fail("the learner still runs; wait for it to exit")
    blocking = []
    for rec in C.list_alarms(lane_dir):
        spec = C.ALARMS_BY_NAME.get(rec["name"])
        if spec is None or spec.stops_lane:
            blocking.append(rec["name"])
    if blocking:
        return _fail("ack these alarms first: " + ", ".join(sorted(blocking)))
    rec = C.read_json(done)
    hist = lane_dir / C.OPS_DIR / "done_history"
    hist.mkdir(parents=True, exist_ok=True)
    dst = hist / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(camp.now))}_DONE.json"
    entry = {"done": rec, "by": os.environ.get("USER", ""), "note": a.note, "time": utc(camp.now)}
    C.atomic_write_text(dst, json.dumps(entry) + "\n")
    done.unlink()
    if a.reset_starts:
        W.reset_starts(ckpt.SaveDir(lane_dir), a.note)
    reason = rec.get("reason") if isinstance(rec, dict) else rec
    text = f"cleared DONE ({reason}): {a.note}"
    if a.reset_starts:
        text += "; crash-loop count reset"
    camp.log(a.lane, text)
    camp.save()
    print(f"cleared DONE ({reason}); start the lane: systemctl --user start e2e-lane@{a.lane}")
    return 0


def cmd_note(camp: Campaign, a: argparse.Namespace) -> int:
    """One operator line in CAMPAIGN_LOG.md (a decision, a parity or learning check)."""
    if a.lane != "campaign" and _lane_dir(camp, a.lane) is None:
        return _fail(f"no lane {a.lane} under {camp.root / C.LANES_DIR}")
    camp.log(a.lane, f"operator: {' '.join(a.text)}")
    camp.save()
    return 0


def cmd_status(camp: Campaign) -> int:
    status = C.read_json(camp.root / C.STATUS)
    if not isinstance(status, dict):
        return _fail(f"no {camp.root / C.STATUS}")
    age = camp.now - float(status.get("time", 0))
    print(f"{status.get('summary', '?')} (status {age / 60:.0f} min old)")
    for a in status.get("alarms", []):
        print(f"  ALARM_{a['name']} [{a['severity']}] {a.get('lane') or '-'}: {a.get('reason')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e2e_driver.py", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--campaign", default=None, help="default: $E2E_CAMPAIGN_DIR or ~/runs/e2e")
    ap.add_argument("--now", default=None, help="unix seconds or ISO 8601 (tests, replays)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run")
    sub.add_parser("status")
    i = sub.add_parser("init-lane", help="deploy a lane: lane.env, lane.args, campaign state")
    i.add_argument("lane")
    i.add_argument("--worktree", required=True)
    i.add_argument("--python", required=True)
    i.add_argument("--pin", default=None)
    i.add_argument("--args-file", required=True)
    i.add_argument("--learner-env", default=None)
    i.add_argument("--hot-swap", action="store_true")
    i.add_argument("--note", default="")
    k = sub.add_parser("ack", help="acknowledge an alarm (HUMAN ones need --decision)")
    k.add_argument("name")
    k.add_argument("--lane", default=None)
    k.add_argument("--note", required=True)
    k.add_argument("--decision", default=None)
    k.add_argument("--mute-hours", type=float, default=0.0)
    r = sub.add_parser("raise", help="raise an operator verdict (K2_FAIL, K3_NOGO, ...)")
    r.add_argument("name")
    r.add_argument("--lane", required=True)
    r.add_argument("--reason", required=True)
    st = sub.add_parser("stop", help="write DONE: stop a lane for good (final stop, a fix)")
    st.add_argument("lane")
    st.add_argument("--reason", required=True)
    st.add_argument("--note", default="")
    cd = sub.add_parser("clear-done", help="archive DONE so the next start runs")
    cd.add_argument("lane")
    cd.add_argument("--note", required=True)
    cd.add_argument("--reset-starts", action="store_true", help="also reset the crash-loop count")
    nt = sub.add_parser("note", help="append an operator line to CAMPAIGN_LOG.md")
    nt.add_argument("lane", help="a lane, or campaign")
    nt.add_argument("text", nargs="+")
    a = ap.parse_args(argv)
    root = Path(a.campaign).expanduser() if a.campaign else C.campaign_dir()
    now = parse_now(a.now)
    if a.cmd == "status":
        return cmd_status(Campaign(root, now))
    # A timer pass skips when another holds the lock; an operator command waits.
    with driver_lock(root, wait_s=0.0 if a.cmd in (None, "run") else LOCK_WAIT_S) as locked:
        if not locked:
            print("another driver command holds driver.lock", file=sys.stderr)
            return 0 if a.cmd in (None, "run") else 2
        return _dispatch(Campaign(root, now), a)


def _dispatch(camp: Campaign, a: argparse.Namespace) -> int:
    if a.cmd in (None, "run"):
        return cmd_run(camp)
    if a.cmd == "init-lane":
        return cmd_init_lane(camp, a)
    if a.cmd == "ack":
        return cmd_ack(camp, a)
    if a.cmd == "raise":
        return cmd_raise(camp, a)
    if a.cmd == "stop":
        return cmd_stop(camp, a)
    if a.cmd == "clear-done":
        return cmd_clear_done(camp, a)
    if a.cmd == "note":
        return cmd_note(camp, a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
