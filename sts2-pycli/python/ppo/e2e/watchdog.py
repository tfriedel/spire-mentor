"""The rules of ``scripts/e2e_watchdog.sh`` (plan §8 Watchdog, WP7).

The bash watchdog runs processes; this module decides. Each command reads the
lane's save dir, does the file-level work (pruning, ``EVALS_PAUSE``, alarms,
``DONE``, rollbacks, the start log) and prints one line that tells the script
what to do with the learner:

``prestart SAVE``
    Before every start. Prints ``done``, ``wait <reason>`` (poll and ask
    again), ``stop <code> <reason>`` or ``start <resume path or -> <sampler
    seed offset>``. It pins the resume state at every restart and records the
    start.
``tick SAVE --since T --pid P``
    Every poll while the learner runs. Prints ``ok``, ``stop <code> <reason>``,
    ``restart stall`` or ``kill_nan <decisions>``.
``exited SAVE --rc RC [--intent stall|stop|signal] [--nan D]``
    After the learner exits. Prints ``done``, ``stopped``, ``stop <code>
    <reason>`` or ``restart``.
``limits``, ``env``, ``started``, ``reset-starts``, ``event``
    Helpers: the thresholds as shell assignments, the lane environment as
    ``KEY=VALUE`` lines, ``ops/watchdog.json``, the crash-loop reset, a log
    line in ``ops/events.jsonl``.

The rules (plan §8; PREREG-0 P6 names the stops that call the user):

* ``DONE`` (written by the learner at its end, the driver, the operator or a
  stop rule below) stops the lane, and nothing restarts it.
* More than :attr:`~ppo.e2e.ops_contract.Limits.starts_max` starts within
  ``starts_window_s`` stop the lane (``CRASH_LOOP``).
* Under ``disk_prune_gb`` free, prune quarantined, slim and old rolling states
  (never pinned ones). Under ``disk_stop_gb`` after that, stop (``DISK_STOP``).
* Under ``ram_stop_gb`` MemAvailable, stop the evals first (``EVALS_PAUSE``).
  If RAM is still low ``ram_grace_s`` later, stop the lane (``RAM_STOP``).
  The pause lifts at ``ram_resume_gb``.
* The lane's memory (PSS summed over the learner's session) above
  ``rss_max_gb`` stops it (``RSS_STOP``), the backstop for ``MemoryMax=``.
* A NaN (learner exit :data:`~ppo.e2e.ops_contract.EXIT_NAN`, ``NAN.json``
  or a non-finite stats row) resumes from the second-newest full state with
  the sampler seed offset raised by 1e6 (``NAN_ROLLBACK``). A second NaN
  within 100M decisions stops the lane (``NAN_HUMAN``).
* A stall restarts the learner: from 30 min after its start, every
  throughput rate of the last 10 min is below half the process's own median
  (or no row came in that window).
* Learner exit :data:`~ppo.e2e.ops_contract.EXIT_HUMAN` stops the lane
  (``LEARNER_HUMAN``). Any other exit without ``DONE`` is a crash: resume
  from ``state.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import ckpt
from . import ops_contract as C

# ── records ──────────────────────────────────────────────────────────────────


def _event(sd: ckpt.SaveDir, kind: str, **data: Any) -> None:
    rec = {"time": time.time(), "utc": C.utc_now_iso(), "kind": kind, **data}
    path = sd.p(C.EVENTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _alarm(sd: ckpt.SaveDir, name: str, reason: str, **details: Any) -> None:
    C.write_alarm(sd.root, name, writer="watchdog", reason=reason, lane=sd.root.name, **details)
    _event(sd, "alarm", name=name, reason=reason)


def _stop(sd: ckpt.SaveDir, code: int, alarm: str, reason: str, **details: Any) -> str:
    """Raise the alarm, write ``DONE`` so nothing restarts the lane, answer ``stop``."""
    _alarm(sd, alarm, reason, **details)
    C.write_done(sd.root, f"watchdog:{alarm}", by="watchdog", detail=reason)
    return f"stop {code} {alarm.lower()}"


def _refuse(sd: ckpt.SaveDir, code: int, alarm: str, reason: str) -> str:
    """Raise the alarm and refuse to start, without ``DONE``: once the operator
    fixes the cause, starting the unit is enough."""
    _alarm(sd, alarm, reason)
    return f"stop {code} {alarm.lower()}"


# ── starts and the seed offset ──────────────────────────────────────────────


def read_starts(sd: ckpt.SaveDir) -> list[float]:
    """Start times since the last ``#reset`` line of ``ops/starts.tsv``."""
    out: list[float] = []
    try:
        text = sd.p(C.STARTS).read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        head = line.split("\t", 1)[0]
        if head == C.STARTS_RESET:
            out = []
            continue
        try:
            out.append(float(head))
        except ValueError:
            continue
    return out


def reset_starts(sd: ckpt.SaveDir, note: str = "") -> None:
    """Restart the crash-loop count (the operator, after fixing a crash loop)."""
    sd.p(C.STARTS).parent.mkdir(parents=True, exist_ok=True)
    with open(sd.p(C.STARTS), "a", encoding="utf-8") as f:
        f.write(f"{C.STARTS_RESET}\t{time.time():.3f}\t{note}\n")
    _event(sd, "starts_reset", note=note)


def seed_offset(sd: ckpt.SaveDir) -> int:
    try:
        return int(sd.p(C.SEED_OFFSET).read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


# ── RAM ──────────────────────────────────────────────────────────────────────


def _lift_pause(sd: ckpt.SaveDir) -> None:
    """Remove ``EVALS_PAUSE`` if the watchdog wrote it (the driver lifts its own)."""
    pause = C.read_json(sd.root / C.EVALS_PAUSE)
    if isinstance(pause, dict) and pause.get("by") == "watchdog":
        (sd.root / C.EVALS_PAUSE).unlink(missing_ok=True)
        _event(sd, "evals_resumed")


def ram_rule(sd: ckpt.SaveDir, lim: C.Limits, now: float) -> tuple[str, float | None]:
    """``ok``, ``low`` (evals paused, inside the grace) or ``stop``; and MemAvailable."""
    ram = C.meminfo_available_gb(lim.meminfo)
    low = sd.p(C.RAM_LOW_SINCE)
    if ram is None:
        return "ok", None
    if ram >= lim.ram_stop_gb:
        low.unlink(missing_ok=True)
        if ram >= lim.ram_resume_gb:
            _lift_pause(sd)
        return "ok", ram
    if not low.exists():
        C.atomic_write_text(low, f"{now}\n")
    if not (sd.root / C.EVALS_PAUSE).exists():
        rec = {"by": "watchdog", "time": C.utc_now_iso(now), "ram_gb": round(ram, 2)}
        C.atomic_write_text(sd.root / C.EVALS_PAUSE, json.dumps(rec) + "\n")
        _event(sd, "evals_paused", ram_gb=round(ram, 2))
    try:
        t0 = float(low.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        t0 = now
    return ("stop" if now - t0 >= lim.ram_grace_s else "low"), ram


# ── before a start ──────────────────────────────────────────────────────────


def _is_pinned_file(sd: ckpt.SaveDir, path: Path) -> bool:
    if path.parent.resolve() != sd.pinned_dir.resolve():
        return False
    return any(r.name == path.name for r in ckpt.pinned_states(sd))


def _resume_decisions(sd: ckpt.SaveDir, path: Path) -> int | None:
    c = ckpt.parse_name(path.name)
    if c is not None:
        return c.decisions
    if path.resolve() == sd.state.resolve():
        return ckpt.state_decisions(sd)
    return None


def _consume_resume_next(sd: ckpt.SaveDir) -> None:
    used = sd.p(C.RESUME_USED_DIR) / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    used.parent.mkdir(parents=True, exist_ok=True)
    sd.p(C.RESUME_NEXT).replace(used)


def _pick_resume(sd: ckpt.SaveDir) -> tuple[Path | None, str, dict | None] | str:
    """The resume state, the pin reason and the rollback record; or a ``stop`` verdict."""
    rn = C.read_json(sd.p(C.RESUME_NEXT))
    if isinstance(rn, dict) and rn.get("path"):
        meta = C.read_json(sd.state_meta)
        saved = float(meta.get("time", 0)) if isinstance(meta, dict) else 0.0
        if sd.state.is_file() and saved > float(rn.get("unix", 0)):
            # The learner saved state.pt after the rollback: the rollback is done.
            _consume_resume_next(sd)
            return sd.state, "restart", None
        path = Path(rn["path"])
        nan = rn.get("reason") == "nan"
        if not path.is_file():
            if nan:
                return _stop(
                    sd, C.WD_EXIT_NAN_HUMAN, "NAN_HUMAN",
                    f"the NaN rollback target {path} is missing",
                )  # fmt: skip
            return _refuse(
                sd, C.WD_EXIT_USAGE, "RESUME_MISSING", f"ops/resume_next.json names {path}"
            )
        return path, ("nan_rollback" if nan else "restart"), rn
    if sd.state.is_file():
        return sd.state, "restart", None
    if ckpt.full_states(sd):
        # A fresh start here would start a new lineage next to the old states.
        return _refuse(
            sd, C.WD_EXIT_USAGE, "RESUME_MISSING", "state.pt is missing but full states exist"
        )
    return None, "restart", None


def prestart(sd: ckpt.SaveDir, lim: C.Limits, now: float | None = None) -> str:
    now = time.time() if now is None else now
    if (sd.root / C.DONE).exists():
        return "done"
    other = C.read_json(sd.root / C.LEARNER_PID)
    if C.record_alive(other):
        _event(sd, "refused", reason=f"a learner still runs (pid {other['pid']})")
        return f"stop {C.WD_EXIT_LOCKED} learner_alive"
    recent = [t for t in read_starts(sd) if now - t < lim.starts_window_s]
    if len(recent) >= lim.starts_max:
        return _stop(
            sd, C.WD_EXIT_CRASH_LOOP, "CRASH_LOOP",
            f"{len(recent)} starts in the last {lim.starts_window_s:.0f} s; "
            f"start {len(recent) + 1} is refused",
        )  # fmt: skip
    v = ckpt.disk_check(sd, prune_below_gb=lim.disk_prune_gb, stop_below_gb=lim.disk_stop_gb)
    if v.action == "stop":
        return _stop(sd, C.WD_EXIT_DISK, "DISK_STOP", f"{v.free_gb:.1f} GB free after pruning")
    if v.action == "pruned":
        _alarm(sd, "DISK_PRUNED", f"{v.free_gb:.1f} GB free", deleted=v.deleted)
    verdict, ram = ram_rule(sd, lim, now)
    if verdict == "stop":
        return _stop(sd, C.WD_EXIT_RAM, "RAM_STOP", f"MemAvailable {ram:.1f} GB before a start")
    if verdict == "low":
        return "wait ram_low"
    picked = _pick_resume(sd)
    if isinstance(picked, str):
        return picked
    resume, reason, rn = picked
    if resume is not None and (sd.p(C.STARTS).exists() or rn is not None):
        if _is_pinned_file(sd, resume):
            _event(sd, "pin_skipped", path=str(resume), reason="already pinned")
        else:
            dec = _resume_decisions(sd, resume)
            if dec is None:
                _event(sd, "pin_skipped", path=str(resume), reason="unknown decision count")
            else:
                pinned = ckpt.pin(sd, decisions=dec, reason=reason, src=resume, note="watchdog")
                _event(sd, "pinned", path=str(pinned), decisions=dec, reason=reason)
                if rn is not None:
                    # Resume from the pinned copy: rotation may delete a rolling state.
                    rn = {**rn, "path": str(pinned), "orig_path": rn["path"]}
                    C.atomic_write_text(sd.p(C.RESUME_NEXT), json.dumps(rn, indent=1) + "\n")
                    resume = pinned
    offset = seed_offset(sd)
    sd.p(C.STARTS).parent.mkdir(parents=True, exist_ok=True)
    with open(sd.p(C.STARTS), "a", encoding="utf-8") as f:
        f.write(f"{now:.3f}\t{resume or '-'}\t{offset}\n")
    _event(sd, "start", resume=str(resume) if resume else None, seed_offset=offset)
    return f"start {resume or '-'} {offset}"


# ── while the learner runs ───────────────────────────────────────────────────


def _proc_mem_bytes(d: Path, page: int) -> int:
    """PSS of one process (shared pages split between their users), else RSS."""
    try:
        for line in (d / "smaps_rollup").read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int((d / "statm").read_text(encoding="utf-8").split()[1]) * page
    except (OSError, ValueError, IndexError):
        return 0


def lane_mem_bytes(session: int) -> int:
    """Memory of every process in session ``session``. The learner is its
    session leader, so this is the learner and everything it started."""
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_bytes().decode("utf-8", "replace")
            if int(stat[stat.rfind(")") + 2 :].split()[3]) != session:  # field 6
                continue
        except (OSError, ValueError, IndexError):
            continue
        total += _proc_mem_bytes(d, page)
    return total


def stall_reason(rows: list[dict], since: float, now: float, lim: C.Limits) -> str | None:
    """Why the learner counts as stalled, or None. ``rows`` are throughput rows."""
    if lim.stall_frac <= 0 or now < since + lim.stall_grace_s + lim.stall_window_s:
        return None
    own = [r for r in rows if r["time"] >= since and C.is_num(r.get("rate"))]
    window = [r for r in own if r["time"] > now - lim.stall_window_s]
    if not window:
        return f"no throughput row for {lim.stall_window_s:g} s"
    med = statistics.median(r["rate"] for r in own)
    top = max(r["rate"] for r in window)
    if med <= 0:
        return None if top > 0 else "no progress since the start"
    if top < lim.stall_frac * med:
        return (
            f"rate at most {top:.0f}/s for {lim.stall_window_s:g} s, under "
            f"{lim.stall_frac:g} x the process's median {med:.0f}/s"
        )
    return None


def tick(sd: ckpt.SaveDir, lim: C.Limits, since: float, pid: int, now: float | None = None) -> str:
    now = time.time() if now is None else now
    done = sd.root / C.DONE
    if done.exists() and now - done.stat().st_mtime > lim.done_grace_s:
        _event(sd, "done_grace", reason="DONE exists but the learner still runs")
        return "stop 0 done_grace"
    verdict, ram = ram_rule(sd, lim, now)
    if verdict == "stop":
        return _stop(
            sd, C.WD_EXIT_RAM, "RAM_STOP",
            f"MemAvailable {ram:.1f} GB for {lim.ram_grace_s:.0f} s after the evals were stopped",
        )  # fmt: skip
    v = ckpt.disk_check(sd, prune_below_gb=lim.disk_prune_gb, stop_below_gb=lim.disk_stop_gb)
    if v.action == "stop":
        return _stop(sd, C.WD_EXIT_DISK, "DISK_STOP", f"{v.free_gb:.1f} GB free after pruning")
    if v.deleted:
        _alarm(sd, "DISK_PRUNED", f"{v.free_gb:.1f} GB free", deleted=v.deleted)
    if lim.rss_max_gb > 0 and sys.platform.startswith("linux"):
        mem = lane_mem_bytes(pid) / 1e9
        if mem > lim.rss_max_gb:
            return _stop(sd, C.WD_EXIT_RSS, "RSS_STOP", f"lane memory {mem:.1f} GB (PSS)")
    # A NaN row is the backstop to the learner's own exit 86: kill it unsaved.
    for row in C.rows_since(sd.root / C.STATS, since, max_bytes=1 << 20):
        if C.row_is_nonfinite(row):
            dec = row.get("decisions")
            return f"kill_nan {dec if isinstance(dec, int) else -1}"
    why = stall_reason(C.rows_since(sd.root / C.THROUGHPUT, since), since, now, lim)
    if why:
        _event(sd, "stall", reason=why)
        return "restart stall"
    return "ok"


# ── after the learner exits ─────────────────────────────────────────────────


def _nan_decisions(sd: ckpt.SaveDir, rec: Any, given: int | None) -> int:
    if given is not None and given >= 0:
        return given
    if isinstance(rec, dict) and isinstance(rec.get("decisions"), int):
        return rec["decisions"]
    rows = list(C.iter_jsonl(C.tail_lines(sd.root / C.STATS, 1 << 20)))
    if rows and isinstance(rows[-1].get("decisions"), int):
        return rows[-1]["decisions"]
    return ckpt.state_decisions(sd) or 0


def handle_nan(sd: ckpt.SaveDir, lim: C.Limits, decisions: int, detail: Any = None) -> str:
    """The NaN rule: roll back once, stop (HUMAN) on a second within the window."""
    path = sd.p(C.NAN_EVENTS)
    prev = list(C.iter_jsonl(C.tail_lines(path)))
    near = [
        e for e in prev
        if isinstance(e.get("decisions"), int)
        and abs(decisions - e["decisions"]) < lim.nan_window_decisions
    ]  # fmt: skip
    rec = {"time": time.time(), "utc": C.utc_now_iso(), "decisions": decisions, "detail": detail}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    if near:
        return _stop(
            sd, C.WD_EXIT_NAN_HUMAN, "NAN_HUMAN",
            f"NaN at {decisions:,} decisions; the one at {near[-1]['decisions']:,} is within "
            f"{lim.nan_window_decisions:,}",
        )  # fmt: skip
    target = ckpt.second_newest_full_state(sd)
    if target is None:
        return _stop(
            sd, C.WD_EXIT_NAN_HUMAN, "NAN_HUMAN",
            f"NaN at {decisions:,} decisions and no second full state to roll back to",
        )  # fmt: skip
    rb = ckpt.rollback(sd, target.path, reason="nan", allow_unpinned=True)
    offset = seed_offset(sd) + lim.nan_seed_step
    C.atomic_write_text(sd.p(C.SEED_OFFSET), f"{offset}\n")
    _alarm(
        sd, "NAN_ROLLBACK",
        f"NaN at {decisions:,} decisions; resuming from {target.path.name} "
        f"({target.decisions:,}) with sampler seed offset {offset}",
        target=str(target.path), quarantined=rb["quarantined"], abandoned=rb["abandoned"],
    )  # fmt: skip
    return "restart"


def exited(
    sd: ckpt.SaveDir,
    lim: C.Limits,
    rc: int,
    *,
    intent: str | None = None,
    nan: int | None = None,
) -> str:
    _event(sd, "exit", rc=rc, intent=intent, nan=nan)
    nan_file = sd.root / C.NAN_RECORD
    was_nan = rc == C.EXIT_NAN or nan is not None or nan_file.exists()
    if was_nan:
        rec = C.read_json(nan_file)
        dec = _nan_decisions(sd, rec, nan)
        if nan_file.exists():
            kept = sd.p(C.OPS_DIR) / f"NAN_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
            nan_file.replace(kept)
        verdict = handle_nan(sd, lim, dec, rec)
        if verdict.startswith("stop"):
            return verdict
    if rc == C.EXIT_HUMAN:
        return _stop(sd, C.WD_EXIT_LEARNER_HUMAN, "LEARNER_HUMAN", "the learner exited 87")
    if (sd.root / C.DONE).exists():
        return "done"
    if intent in ("signal", "stop"):
        return "stopped"
    if not was_nan and intent != "stall":
        _event(sd, "crash", rc=rc)
    return "restart"


def started(sd: ckpt.SaveDir, watchdog_pid: int, pid: int = 0, since: float = 0.0) -> None:
    """Write ``ops/watchdog.json`` (the driver reads it to see the watchdog live)."""
    rec = {
        "watchdog": C.pid_record(watchdog_pid, argv=[]),
        "learner": (
            {"pid": pid, "starttime": C.proc_starttime(pid), "since": since} if pid else None
        ),
        "time": C.utc_now_iso(),
    }
    C.atomic_write_text(sd.p(C.WATCHDOG_STATE), json.dumps(rec, indent=1) + "\n")
    if pid:
        _event(sd, "started", pid=pid)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.e2e.watchdog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("limits")
    sub.add_parser("env")
    sub.add_parser("prestart").add_argument("save_dir")
    t = sub.add_parser("tick")
    t.add_argument("save_dir")
    t.add_argument("--since", type=float, required=True)
    t.add_argument("--pid", type=int, required=True)
    e = sub.add_parser("exited")
    e.add_argument("save_dir")
    e.add_argument("--rc", type=int, required=True)
    e.add_argument("--intent", choices=("stall", "stop", "signal"), default=None)
    e.add_argument("--nan", type=int, default=None)
    s = sub.add_parser("started")
    s.add_argument("save_dir")
    s.add_argument("--watchdog-pid", type=int, required=True)
    s.add_argument("--pid", type=int, default=0)
    s.add_argument("--since", type=float, default=0.0)
    r = sub.add_parser("reset-starts", help="restart the crash-loop count")
    r.add_argument("save_dir")
    r.add_argument("--note", default="")
    ev = sub.add_parser("event")
    ev.add_argument("save_dir")
    ev.add_argument("kind")
    ev.add_argument("--note", default="")
    a = ap.parse_args(argv)
    lim = C.Limits.from_env()
    if a.cmd == "limits":
        sys.stdout.write(lim.shell())
        return 0
    if a.cmd == "env":
        sys.stdout.write("".join(f"{k}={v}\n" for k, v in C.LANE_ENVIRONMENT.items()))
        return 0
    sd = ckpt.SaveDir(a.save_dir)
    sd.p(C.OPS_DIR).mkdir(parents=True, exist_ok=True)
    if a.cmd == "prestart":
        print(prestart(sd, lim))
    elif a.cmd == "tick":
        print(tick(sd, lim, a.since, a.pid))
    elif a.cmd == "exited":
        print(exited(sd, lim, a.rc, intent=a.intent, nan=a.nan))
    elif a.cmd == "started":
        started(sd, a.watchdog_pid, a.pid, a.since)
    elif a.cmd == "reset-starts":
        reset_starts(sd, a.note)
    elif a.cmd == "event":
        _event(sd, a.kind, note=a.note)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
