"""The ops contract of one e2e training lane (plan §8 Ops, WP7).

Three programs share a lane's save dir, and this module is the one place that
names what they share: files, cadences, CLI flags, exit codes, log fields and
alarms.

* The **learner** (WP4, ``python -m ppo.e2e.learner train``; not built yet)
  writes the states, logs and markers below. It honors ``DONE``,
  ``EVALS_PAUSE`` and ``PIN_REQUEST``, and it exits with the codes below.
* The **watchdog** (``scripts/e2e_watchdog.sh``, its rules in
  :mod:`ppo.e2e.watchdog`) starts the learner as its own child. It resumes the
  learner after a crash, restarts it on a stall, rolls it back after a NaN, and
  stops it on ``DONE``, a crash loop, low disk, low RAM or the RSS cap.
* The **driver** (``scripts/e2e_driver.py``, a 5-min systemd user timer)
  reads the logs, executes only the pre-registered rules, and writes
  ``status.json``, ``ALARM_<name>`` files and ``CAMPAIGN_LOG.md``.

The module imports only the standard library, so the watchdog's checks start
in milliseconds and the driver runs on any Python 3.11 or later.

Learner obligations (the contract WP4 implements)
-------------------------------------------------
1. CLI: ``python -u -m ppo.e2e.learner train --save-dir DIR [--resume PATH]
   --sampler-seed-offset N <lane args>``. ``--resume`` loads a full state
   (weights, optimizer, EMAs, advantage moments, counters and schedule clocks)
   from PATH. Without ``--resume``, the learner refuses to start when
   ``DIR/state.pt`` exists (exit :data:`EXIT_USAGE`), so a run is never
   overwritten by accident. The action-sampler RNG derives from
   ``(cfg seed + sampler seed offset, iteration)``. The watchdog passes the
   offset on every start, and the learner records it in every stats row.
2. Pid and lock: at startup the learner takes an exclusive ``flock`` on
   :data:`LEARNER_LOCK` (exit :data:`EXIT_USAGE` if another process holds it)
   and writes :data:`LEARNER_PID`, a JSON object with ``pid``, ``starttime``
   (field 22 of ``/proc/<pid>/stat``, so a reused pid is detectable),
   ``host``, ``started_at`` and ``argv``. The watchdog waits on the pid it
   started itself and never searches with ``pgrep -f``.
3. States (all through :mod:`ppo.e2e.ckpt`, which writes atomically):
   ``state.pt`` every :data:`STATE_EVERY` decisions on a save job (not after
   every update); a rolling full state every :data:`FULL_EVERY` (the newest
   :data:`FULL_KEEP` are kept); a pinned full state, never rotated, at every
   :data:`PIN_EVERY` milestone, at each K-gate in :data:`K_GATES`, and when a
   ``PIN_REQUEST`` file appears; a slim state (weights and EMAs) every
   :data:`SLIM_EVERY`. The watchdog pins the resume state at every restart.
   The operator pins at a hot-swap and at the E=1 switch.
4. Logs: one :data:`STATS` row per iteration with :data:`STATS_FIELDS` (a
   resume moves the rows of the iterations after its state, which it replays,
   to :data:`STATS_ORPHAN`, so ``it`` stays unique and increasing); one
   :data:`THROUGHPUT` row per minute with :data:`THROUGHPUT_FIELDS`; a
   :data:`HEARTBEAT` at least every :data:`HEARTBEAT_EVERY_S` seconds.
5. Stops: the learner checks for ``DONE`` at every iteration boundary. When
   it finds one, it finishes the update in flight, writes ``state.pt`` and
   exits :data:`EXIT_OK`. It does the same on SIGTERM or SIGINT. At its normal
   end (budget reached) it writes ``state.pt``, then ``DONE`` with reason
   ``budget``, and exits :data:`EXIT_OK`. Exit 0 without ``DONE`` counts as a
   crash.
6. NaN: on a non-finite loss, gradient norm or value, the learner skips the
   step, saves nothing, writes :data:`NAN_RECORD` and exits :data:`EXIT_NAN`.
7. Guards: a wheel or source mismatch (``worker_wheel_mismatch`` and
   ``worker_source_mismatch``, checked every batch), an lr off its closed form,
   or an F9 invariant hit make the learner write ``ALARM_<name>`` and exit
   :data:`EXIT_HUMAN`. The watchdog does not restart it.
8. Evals: the learner (or its eval runner) consumes :data:`EVAL_QUEUE`.
   Claim a request by renaming it into :data:`EVAL_RUNNING`, run it at nice
   10 with at most ``max_procs`` workers, and write the result to
   ``EVAL_DONE/<id>.json``: the request's fields plus
   :data:`EVAL_RESULT_FIELDS` (``wins`` and ``runs``, the finished runs; a
   result counts only when ``runs`` equals the request's ``n``) and any extra
   fields, such as the path of the per-seed rows. While :data:`EVALS_PAUSE`
   exists, stop every running eval (move its request back to the queue) and
   claim nothing.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import os
import re
import shlex
import socket
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeGuard

CONTRACT_VERSION = 1

# ── the lane's save dir ─────────────────────────────────────────────────────
# Paths are relative to the save dir (the learner's --save-dir).

STATE = "state.pt"
CKPT_DIR = "ckpt"
FULL_DIR = "ckpt/full"  # rolling full states, full_d<decisions>.pt
PINNED_DIR = "ckpt/pinned"  # pinned_d<decisions>_<reason>.pt, never rotated
SLIM_DIR = "ckpt/slim"  # slim_d<decisions>.pt
QUARANTINE_DIR = "ckpt/quarantine"  # states moved aside by a rollback
MANIFEST = "ckpt/MANIFEST.tsv"  # sha256 of every pinned state, append-only
ACTIONS_DIR = "actions"  # the workers' action logs, actions-<worker>.e2el (actionlog.LogSink)
STATS = "stats.jsonl"
# Stats rows a resume parked: the iterations after the resumed state (learner.park_orphan_stats).
STATS_ORPHAN = "stats.orphan.jsonl"
THROUGHPUT = "throughput.jsonl"
HEARTBEAT = "heartbeat.json"
LEARNER_PID = "learner.pid"
LEARNER_LOCK = "learner.lock"
DONE = "DONE"
NAN_RECORD = "NAN.json"
PIN_REQUEST = "PIN_REQUEST"
EVALS_PAUSE = "EVALS_PAUSE"
EVAL_QUEUE = "evals/queue"
EVAL_RUNNING = "evals/running"
EVAL_DONE = "evals/done"
LANE_ENV = "lane.env"  # KEY=VALUE lines: E2E_WORKTREE, E2E_PY, E2E_PIN, ...
LANE_ARGS = "lane.args"  # the learner's extra args, one per line
LANE_LEARNER_ENV = "lane.learner_env"  # optional extra KEY=VALUE for the learner
ALARM_PREFIX = "ALARM_"

# The watchdog's own files.
OPS_DIR = "ops"
WATCHDOG_LOG = "ops/watchdog.log"
LEARNER_LOG = "ops/learner.log"
WATCHDOG_STATE = "ops/watchdog.json"
WATCHDOG_LOCK = "ops/watchdog.lock"
STARTS = "ops/starts.tsv"
EVENTS = "ops/events.jsonl"
NAN_EVENTS = "ops/nan_events.jsonl"
# Every ended segment with an F9 hit, one line each: seed, cell, the first violation
# texts and `dropped` (true for a hung segment, which does not stop the lane).
F9_HITS = "ops/f9_hits.jsonl"
SEED_OFFSET = "ops/seed_offset"
RESUME_NEXT = "ops/resume_next.json"
RESUME_USED_DIR = "ops/resume_used"
RAM_LOW_SINCE = "ops/ram_low_since"
STARTS_RESET = "#reset"  # a starts.tsv line that restarts the crash-loop count

# ── the campaign dir (one per host) ─────────────────────────────────────────

CAMPAIGN_DIR_ENV = "E2E_CAMPAIGN_DIR"
CAMPAIGN_STATE = "campaign_state.json"
CAMPAIGN_LOG = "CAMPAIGN_LOG.md"
STATUS = "status.json"
LANES_DIR = "lanes"
ACKED_DIR = "alarms/acked"


def campaign_dir() -> Path:
    """``$E2E_CAMPAIGN_DIR``, else ``~/runs/e2e``."""
    env = os.environ.get(CAMPAIGN_DIR_ENV)
    return Path(env).expanduser() if env else Path.home() / "runs" / "e2e"


# ── cadences (decisions) ────────────────────────────────────────────────────

STATE_EVERY = 5_000_000
FULL_EVERY = 25_000_000
FULL_KEEP = 3
PIN_EVERY = 250_000_000
SLIM_EVERY = 100_000_000
THROUGHPUT_EVERY_S = 60.0
HEARTBEAT_EVERY_S = 60.0

# Pin reasons (file-name safe). The learner pins milestones, K-gates and
# requests; the watchdog pins restarts and NaN rollbacks; the operator pins
# hot-swaps, the E=1 switch and anything manual.
PIN_REASONS = (
    "milestone",
    "k2",
    "k3",
    "k4",
    "restart",
    "nan_rollback",
    "hot_swap",
    "e1_switch",
    "request",
    "manual",
    "final",
)

# ── the learner CLI the watchdog uses ───────────────────────────────────────

LEARNER_MODULE = "ppo.e2e.learner"
LEARNER_SUBCOMMAND = "train"
FLAG_SAVE_DIR = "--save-dir"
FLAG_RESUME = "--resume"
FLAG_SEED_OFFSET = "--sampler-seed-offset"

# Learner exit codes.
EXIT_OK = 0  # normal end or a DONE-file / SIGTERM stop; state.pt is written
EXIT_USAGE = 2  # refused to start (bad args, state.pt without --resume, lock held)
EXIT_NAN = 86  # non-finite update; nothing saved after it; NAN.json written
EXIT_HUMAN = 87  # a guard refused to continue; ALARM_<name> written

# Watchdog exit codes. Every code but 0 and 1 is a deliberate stop, listed in
# the unit's RestartPreventExitStatus= so systemd does not restart it.
WD_EXIT_DONE = 0
WD_EXIT_ERROR = 1
WD_EXIT_USAGE = 2
WD_EXIT_DISK = 3
WD_EXIT_CRASH_LOOP = 4
WD_EXIT_RAM = 5
WD_EXIT_NAN_HUMAN = 6
WD_EXIT_LEARNER_HUMAN = 7
WD_EXIT_RSS = 8
WD_EXIT_LOCKED = 9
WD_NO_RESTART_CODES = (2, 3, 4, 5, 6, 7, 8, 9)

# The environment every lane runs under (plan §8 Ops; malloc settings [LES]).
# The watchdog starts the learner under `env -i` with these, HOME, PATH, LANG,
# USER, PYTHONPATH (`python`, or `<pin>:python`) and lane.learner_env.
LANE_ENVIRONMENT: dict[str, str] = {
    "STS2_GAME_VERSION": "v0.111.0",
    "STS2_RUNAI_POTION_SWAP": "1",
    "STS2_PPO_FEATURES": "card_mods,reward_upgrade",
    "OMP_NUM_THREADS": "1",
    "MALLOC_MMAP_MAX_": "0",
    "MALLOC_TRIM_THRESHOLD_": "68719476736",
    "MALLOC_TOP_PAD_": "268435456",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
GAME_VERSION = "v0.111.0"

# The thread caps every actor and eval worker starts under, whatever the lane's
# environment. A spawned worker imports numpy (and torch, through the learner's
# main module or the student net) before its own code runs, so the caps must be in
# its environment at exec. Without them numpy's OpenBLAS starts a thread per core
# at import and those threads spin before they sleep (32 threads and 1.3-1.5 s
# more CPU per worker start on the 32-thread laptop), and torch runs an OpenMP
# thread per core. OpenBLAS also reads OMP_NUM_THREADS, so LANE_ENVIRONMENT alone
# caps a lane the watchdog starts, but not a learner started directly. The learner
# and the update ranks keep their own threads.
WORKER_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@contextlib.contextmanager
def worker_thread_env() -> Iterator[None]:
    """`WORKER_THREAD_ENV` in this process's environment while the block runs, then
    the old values back. Wrap the `start()` of `spawn` workers in it: a spawned
    child starts with the environment it had at exec. This process's own numpy and
    torch read the variables when they loaded, so they keep their threads."""
    old = {k: os.environ.get(k) for k in WORKER_THREAD_ENV}
    os.environ.update(WORKER_THREAD_ENV)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── gates and pre-registered constants (PREREG-0 P6, plan §7, §8, §10.4) ────

_CET = _dt.timezone(_dt.timedelta(hours=1), "CET")


@dataclass(frozen=True)
class Gate:
    name: str
    at_decisions: int
    backstop: _dt.datetime | None  # the read happens by then even below at_decisions
    read: str  # what the operator reads (PREREG-0 P6)
    eval_panel: str  # the eval the driver queues on the gate's pinned state ("" = none)


K_GATES: tuple[Gate, ...] = (
    Gate(
        "K2",
        100_000_000,
        None,
        "student DEV-1000 >= teacher pair - 8pp (paired); agreement >= 85% combat and "
        ">= 75% run; finite value loss; EV within 0.1 of its 50M value",
        "DEV-1000",
    ),
    Gate(
        "K3",
        1_400_000_000,
        _dt.datetime(2026, 11, 5, 18, 0, tzinfo=_CET),
        "NO-GO iff paired DEV-1000 student - C* <= -3pp -> HUMAN",
        "DEV-1000",
    ),
    Gate(
        "K4",
        3_000_000_000,
        _dt.datetime(2026, 11, 7, 8, 0, tzinfo=_CET),
        "NO-GO iff paired DEV-2000 student < C* -> HUMAN (reads the 3B DEV-2000 evals)",
        "",
    ),
)

# Eval panels the driver queues (PREREG-0 P1; plan §10.4 "every 250M; every 1B").
DEV_START = 160_000_000
EVAL_PANELS: dict[str, dict[str, int]] = {
    "DEV-1000": {"n": 1000, "seed_start": DEV_START},
    "DEV-2000": {"n": 2000, "seed_start": DEV_START},
}
DEV1000_EVERY = 250_000_000  # EMA0.996
DEV2000_EVERY = 1_000_000_000  # raw, EMA0.996, EMA0.999 (replaces that milestone's DEV-1000)
EVAL_VARIANTS_DEV1000 = ("ema0996",)
EVAL_VARIANTS_DEV2000 = ("raw", "ema0996", "ema0999")
EVAL_NICE = 10
EVAL_MAX_PROCS = 6
EVAL_RESULT_FIELDS = ("wins", "runs")  # added to the request's fields in EVAL_DONE/<id>.json

# Pre-registered reads of finished evals (plan §1.4 and §12), on the default
# model (P5: EMA0.996) of milestone pins.
STALL_PANEL = "DEV-2000"
STALL_VARIANT = "ema0996"
STALL_FLOORS = ((2_000_000_000, 0.11), (3_000_000_000, 0.14))  # DEV-2000 below this: stall
# A DEV-1000 read more than 5pp above the previous one (250M earlier, 500M
# across a 1B milestone): audit for a sim exploit. DEV-2000 reads are 1B apart,
# and +5pp per 1B is the forecast's median slope, so they don't count.
DEV_JUMP = 0.05
DEV_JUMP_PANEL = "DEV-1000"

LR_TOLERANCE = 1e-6  # relative: |lr / closed form - 1| (the lr factor off by > 1e-6)
CAPS_MAX_FRAC = 0.001  # capped fights or runs above 0.1% -> alarm
CAPS_WINDOW_ITERS = 50
CAPS_MIN_FIGHTS = 5_000
CAPS_MIN_RUNS = 1_000
# The E=1 revert rule (a) (plan §9): any type's KL or clipfrac over its bound
# for 20 iterations in a row while E=1.
E1_KL_MAX = 0.03
E1_CLIPFRAC_MAX = 0.3
E1_WINDOW_ITERS = 20
# A held run row is overdue once it waited more than HOLD_ITERS + 2 iterations
# (`algo.Returns.overdue`; HOLD_ITERS = 4). A worker stall of d >= 3 iterations
# leaves rows overdue for d + 1 iterations in a row (the test streams of
# test_algo.py: 4 at most over 2000 iterations), a leaked held row (never
# released, never dropped) for good: `HELD_ROW_LEAK` needs HOLD_ITERS + 3 in a
# row (a test ties it to `algo.HOLD_ITERS`).
HELD_LEAK_ITERS = 7
F9_MAX_HITS = 0  # any hit stops the lane (plan §3.5 F9: 0 per 1e6), a hung segment's aside
# The env cap that marks a hung segment (PREREG Amendment 9): F9 hits in a segment that
# ended `capped` at this cap are logged (`F9_HITS`), its rows dropped, the lane goes on.
F9_HANG_CAP = "steps"
HEARTBEAT_STALE_S = 15 * 60
PIN_MISSING_AFTER = 10_000_000  # decisions past a milestone or K-gate without its pin


# ── watchdog limits (environment overrides for tests: E2E_WD_<FIELD>) ───────


@dataclass(frozen=True)
class Limits:
    """The watchdog's thresholds. Each field reads ``E2E_WD_<NAME>`` if set."""

    poll_s: float = 15.0
    restart_delay_s: float = 30.0
    stop_grace_s: float = 600.0  # SIGTERM -> SIGKILL on a deliberate stop
    stall_stop_grace_s: float = 120.0  # SIGTERM -> SIGKILL on a stall restart
    done_grace_s: float = 900.0  # DONE appeared but the learner still runs -> SIGTERM
    starts_max: int = 4  # more than 4 starts in 2 h stops the lane
    starts_window_s: float = 7200.0
    stall_grace_s: float = 1800.0  # the stall clock starts 30 min after any start
    stall_window_s: float = 600.0  # below half the median for 10 min
    stall_frac: float = 0.5
    nan_window_decisions: int = 100_000_000
    nan_seed_step: int = 1_000_000
    disk_prune_gb: float = 45.0
    disk_stop_gb: float = 30.0
    ram_stop_gb: float = 20.0
    ram_resume_gb: float = 24.0  # hysteresis before EVALS_PAUSE is lifted
    ram_grace_s: float = 120.0  # evals stop first; the lane stops if RAM stays low
    rss_max_gb: float = 200.0  # backstop for MemoryMax=200G
    meminfo: str = "/proc/meminfo"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Limits:
        env = os.environ if env is None else env
        kw: dict[str, Any] = {}
        for f in fields(cls):
            raw = env.get("E2E_WD_" + f.name.upper(), "")
            if not raw:
                continue
            default = getattr(cls, f.name)
            if isinstance(default, int):
                kw[f.name] = int(float(raw))
            elif isinstance(default, float):
                kw[f.name] = float(raw)
            else:
                kw[f.name] = raw
        return cls(**kw)

    def shell(self) -> str:
        """``WD_<NAME>=<value>`` lines for the bash watchdog to read, plus an
        integer ``WD_<NAME>_MS`` for every ``*_s`` field (bash has no floats)."""
        out = []
        for f in fields(self):
            v = getattr(self, f.name)
            out.append(f"WD_{f.name.upper()}={shlex.quote(str(v))}\n")
            if f.name.endswith("_s"):
                out.append(f"WD_{f.name.upper()[:-2]}_MS={round(float(v) * 1000)}\n")
        return "".join(out)


DISK_QUOTA_ENV = "E2E_DISK_QUOTA_GB"  # optional soft quota for the save dir


# ── log rows ─────────────────────────────────────────────────────────────────

# stats.jsonl, one row per iteration. Each field: (kind, meaning). Kinds:
# "int", "num" (int or float), "map" (str -> num, e.g. per decision type),
# "num|map", "str", "str|null", "obj".
STATS_FIELDS: dict[str, tuple[str, str]] = {
    "it": ("int", "iteration count after this update"),
    "decisions": (
        "int",
        "the decision clock: stored policy rows collected so far, the schedules' and "
        "budget's clock; it counts the rows of segments the cut later drops for an env "
        "error too (assembly.dropped_error_rows; small at the error alarm's rate)",
    ),
    "time": ("num", "unix time of the row"),
    "pid": ("int", "the learner's pid"),
    "dec_per_s": ("num", "decisions per second over the iteration"),
    "epochs": ("int", "PPO epochs per batch: 2, or 1 after the E=1 switch (plan §9)"),
    "staleness": (
        "map",
        "policy versions between sampling and update: mean, max over every trained row "
        "(held-back run rows lag up to 5), fresh_mean, fresh_max over this iteration's "
        "rows (B-T0's 'staleness <= 1' reads fresh_max)",
    ),
    "kl": ("map", "KL per decision type (combat, ooc, ...)"),
    "clipfrac": ("num|map", "share of rows outside the clip band, per type"),
    "is_tails": ("map", "IS weight pi_prox/pi_behav tails: p01, p99, max, frac_clamped"),
    "entropy": ("map", "normalised entropy per decision type"),
    "value_ev": ("map", "explained variance per value head (combat, run)"),
    "agreement": ("map", "teacher agreement per type (empty once teachers are off)"),
    "lambda_ks": ("num", "kickstarting weight"),
    "beta": (
        "map",
        "the family entropy coefficients (combat, ooc) before the per-type gates; a type's "
        "applied coefficient is its gain times its family's, 0 while it is gated "
        "(update.ent_gated_types counts those)",
    ),
    "lr": ("map", "optimizer lr per param group"),
    "lr_closed_form": ("map", "the same lrs from the schedule's closed form"),
    "caps": ("map", "this iteration: fights, fights_capped, runs, runs_capped"),
    "f9": (
        "map",
        "this iteration: checked, hits (they stop the lane), tolerated, and the hits and "
        "segments of hung fights (step cap), which are logged and dropped: hung_hits, hung_segments",
    ),
    "losses": ("map", "policy, value, ks, total"),
    "grad_norm": ("num", "gradient norm before clipping"),
    "nonfinite": ("int", "1 if any loss, gradient or value was non-finite"),
    "sampler_seed_offset": ("int", "the --sampler-seed-offset this process runs with"),
    "wheel_mismatch": ("str|null", "worker_wheel_mismatch() result for this batch"),
    "source_mismatch": ("str|null", "worker_source_mismatch() result for this batch"),
    "stamps": ("obj", "py_src, wheel, game_version, vocab_sha256, schema_sha256"),
}
STAMP_FIELDS = ("py_src", "wheel", "game_version", "vocab_sha256", "schema_sha256")
# Values the watchdog checks for NaN (a backstop to the learner's EXIT_NAN).
NAN_FIELDS = ("losses", "grad_norm", "kl", "value_ev")

# throughput.jsonl, one row per minute.
THROUGHPUT_FIELDS: dict[str, tuple[str, str]] = {
    "time": ("num", "unix time of the row"),
    "pid": ("int", "the learner's pid"),
    "decisions": ("int", "decisions so far"),
    "rate": ("num", "decisions per second over the last minute"),
}
THROUGHPUT_OPTIONAL = (
    "it",
    "worker_cpu_ms",  # worker CPU per decision (B-T0, X-T)
    "update_wait_frac",  # share of the minute the learner waited on the update (E=1 rule)
    "infer_ms",
    "eval_share",  # share of wall clock spent on evals (<= 5%)
)


def is_num(v: Any) -> TypeGuard[int | float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _kind_ok(kind: str, v: Any) -> bool:
    if kind == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if kind == "num":
        return is_num(v)
    if kind == "map":
        return isinstance(v, dict) and all(is_num(x) or x is None for x in v.values())
    if kind == "num|map":
        return is_num(v) or _kind_ok("map", v)
    if kind == "str":
        return isinstance(v, str)
    if kind == "str|null":
        return v is None or isinstance(v, str)
    if kind == "obj":
        return isinstance(v, dict)
    raise ValueError(kind)


def _check(row: Mapping[str, Any], spec: Mapping[str, tuple[str, str]]) -> list[str]:
    out = []
    for key, (kind, _) in spec.items():
        if key not in row:
            out.append(f"missing {key}")
        elif not _kind_ok(kind, row[key]):
            out.append(f"{key}: expected {kind}, got {type(row[key]).__name__}")
    return out


def check_stats_row(row: Mapping[str, Any]) -> list[str]:
    """Contract problems of one stats.jsonl row (empty when it conforms)."""
    out = _check(row, STATS_FIELDS)
    stamps = row.get("stamps")
    if isinstance(stamps, dict):
        out += [f"stamps: missing {k}" for k in STAMP_FIELDS if k not in stamps]
    return out


def check_throughput_row(row: Mapping[str, Any]) -> list[str]:
    """Contract problems of one throughput.jsonl row."""
    return _check(row, THROUGHPUT_FIELDS)


def _nonfinite(v: Any) -> bool:
    if isinstance(v, float):
        return not math.isfinite(v)
    if isinstance(v, dict):
        return any(_nonfinite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_nonfinite(x) for x in v)
    return False


def row_is_nonfinite(row: Mapping[str, Any]) -> bool:
    """True when a stats row reports a NaN: ``nonfinite`` set, or a NaN/inf in
    one of :data:`NAN_FIELDS`."""
    return bool(row.get("nonfinite")) or any(_nonfinite(row.get(k)) for k in NAN_FIELDS)


def minimal_stats_row(it: int, decisions: int, **over: Any) -> dict[str, Any]:
    """A conforming stats row with neutral values (tests, and an example for WP4)."""
    row: dict[str, Any] = {
        "it": it,
        "decisions": decisions,
        "time": time.time(),
        "pid": os.getpid(),
        "dec_per_s": 0.0,
        "epochs": 2,
        "staleness": {"mean": 0.0, "max": 0.0, "fresh_mean": 0.0, "fresh_max": 0.0},
        "kl": {"combat": 0.0, "ooc": 0.0},
        "clipfrac": {"combat": 0.0, "ooc": 0.0},
        "is_tails": {"p01": 1.0, "p99": 1.0, "max": 1.0, "frac_clamped": 0.0},
        "entropy": {"combat": 0.5, "ooc": 0.5},
        "value_ev": {"combat": 0.0, "run": 0.0},
        "agreement": {},
        "lambda_ks": 0.0,
        "beta": {"combat": 0.0, "ooc": 0.0},
        "lr": {"trunk": 1e-4, "value": 3e-4},
        "lr_closed_form": {"trunk": 1e-4, "value": 3e-4},
        "caps": {"fights": 0, "fights_capped": 0, "runs": 0, "runs_capped": 0},
        "f9": {"checked": 0, "hits": 0},
        "losses": {"policy": 0.0, "value": 0.0, "ks": 0.0, "total": 0.0},
        "grad_norm": 0.0,
        "nonfinite": 0,
        "sampler_seed_offset": 0,
        "wheel_mismatch": None,
        "source_mismatch": None,
        "stamps": {
            "py_src": "0" * 16,
            "wheel": "0" * 16,
            "game_version": GAME_VERSION,
            "vocab_sha256": "0" * 64,
            "schema_sha256": "0" * 64,
        },
    }
    row.update(over)
    return row


# ── alarms ───────────────────────────────────────────────────────────────────

HUMAN = "HUMAN"  # only the user decides what happens next
OPERATOR = "OPERATOR"  # the campaign's laptop session acts or acknowledges

_ALARM_NAME = re.compile(r"^[A-Z0-9_]+$")


@dataclass(frozen=True)
class Alarm:
    """One row of the alarm table.

    ``stops_lane``: the writer stops the lane by writing ``DONE``. The one
    exception is ``RESUME_MISSING``: the watchdog refuses to start and exits
    without ``DONE``. Most HUMAN alarms stop the lane. ``K3_NOGO``,
    ``K4_NOGO`` and ``STALL_SIGNAL`` do not: PREREG-0 P6 lets training
    continue while the operator diagnoses for up to 6 h. ``writer``
    ``operator`` means the operator raises it with ``e2e_driver.py raise``.
    """

    name: str
    severity: str
    writer: str
    trigger: str
    action: str  # what the writer already did
    who_acts: str
    stops_lane: bool


ALARMS: tuple[Alarm, ...] = (
    Alarm(
        "DISK_PRUNED",
        OPERATOR,
        "watchdog",
        "free disk under 45 GB",
        "deleted quarantined, slim and old rolling states (never pinned ones); lane runs on",
        "operator: find what grew (ckpt usage in status.json) and archive pinned states",
        False,
    ),
    Alarm(
        "DISK_STOP",
        OPERATOR,
        "watchdog, driver",
        "free disk under 30 GB after pruning",
        "stopped the lane (DONE)",
        "operator: free disk, then restart per the runbook",
        True,
    ),
    Alarm(
        "CRASH_LOOP",
        OPERATOR,
        "watchdog",
        "a 5th learner start within 2 h",
        "stopped the lane (DONE)",
        "operator: read ops/learner.log, fix, restart; HUMAN if no fix is found",
        True,
    ),
    Alarm(
        "RESUME_MISSING",
        OPERATOR,
        "watchdog",
        "the resume state is missing: ops/resume_next.json names no file, or "
        "state.pt is gone while full states exist",
        "refused to start (no DONE; the unit stays down)",
        "operator: pick a pinned state (`ckpt rollback`), then start the unit",
        True,
    ),
    Alarm(
        "NAN_ROLLBACK",
        OPERATOR,
        "watchdog",
        "first NaN (learner exit 86, or a non-finite stats row)",
        "resumed from the second-newest full state with sampler seed offset +1e6",
        "operator: note it in CAMPAIGN_LOG.md; diagnose while training continues",
        False,
    ),
    Alarm(
        "NAN_HUMAN",
        HUMAN,
        "watchdog",
        "a second NaN within 100M decisions, or no full state to roll back to",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "RAM_STOP",
        OPERATOR,
        "watchdog, driver",
        "MemAvailable under 20 GB after the evals were stopped",
        "stopped evals (EVALS_PAUSE), then the lane (DONE)",
        "operator: check `free -g` and who holds the memory, then restart",
        True,
    ),
    Alarm(
        "RSS_STOP",
        OPERATOR,
        "watchdog",
        "the lane's resident memory above 200 GB (MemoryMax backstop)",
        "stopped the lane (DONE)",
        "operator: find the leak (ops/learner.log), restart with fewer workers or envs",
        True,
    ),
    Alarm(
        "LEARNER_HUMAN",
        HUMAN,
        "watchdog",
        "the learner exited 87 (a guard refused to continue)",
        "stopped the lane (DONE); the learner's own ALARM_<name> says why",
        "user",
        True,
    ),
    Alarm(
        "LR_CLOSED_FORM",
        HUMAN,
        "driver, learner",
        "an lr off its closed form by more than 1e-6 (relative)",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "WHEEL_MISMATCH",
        HUMAN,
        "driver, learner",
        "worker_wheel_mismatch, or the wheel stamp differs from campaign_state.json",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "SOURCE_MISMATCH",
        HUMAN,
        "driver, learner",
        "worker_source_mismatch, the py_src stamp changed, or the lane's worktree moved",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "GAME_VERSION",
        HUMAN,
        "driver",
        "a stats stamp names a game version other than v0.111.0",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "F9_INVARIANT",
        HUMAN,
        "driver, learner",
        "any F9 invariant hit (HP <= max, energy >= 0, pile conservation) in a segment "
        "that did not end at the step cap; ops/f9_hits.jsonl has seed and violation text",
        "stopped the lane (DONE)",
        "user",
        True,
    ),
    Alarm(
        "FILL_MISMATCH",
        HUMAN,
        "learner",
        "the first conclusive table of a learner start (fresh or resumed) fails the fill "
        "check against --filled: undeclared fill-event inputs, or a declared one's missing",
        "stopped the lane (DONE): the learner exits 87 before the batch trains",
        "user",
        True,
    ),
    Alarm(
        "LC_RECHECK_FAIL",
        OPERATOR,
        "learner",
        "a learning check failed after an earlier one passed",
        "kept the lambda_KS ramp; training continues",
        "operator: read the verdict, decide whether to investigate",
        False,
    ),
    Alarm(
        "CAPS",
        OPERATOR,
        "driver",
        "capped fights or runs above 0.1% over the last 50 iterations",
        "none",
        "operator: look for deadlocks or sim exploits in the capped runs' action logs",
        False,
    ),
    Alarm(
        "CONTRACT",
        OPERATOR,
        "driver",
        "a stats or throughput row misses contract fields",
        "none (rules that need the field are skipped)",
        "operator: fix the learner's rows at the next restart",
        False,
    ),
    Alarm(
        "HELD_ROW_LEAK",
        OPERATOR,
        "driver",
        "update.rows_overdue above 0 in each of the last HELD_LEAK_ITERS (7) iterations",
        "none; training continues",
        "operator: find the runs whose held rows never train (a lost run not dropped, "
        "a hung worker) in ops/learner.log; restart the learner if it leaks",
        False,
    ),
    Alarm(
        "PIN_MISSING",
        OPERATOR,
        "driver",
        "no pinned state 10M decisions after a 250M milestone or a K-gate",
        "wrote PIN_REQUEST",
        "operator: check the learner's pinning; pin by hand if needed",
        False,
    ),
    Alarm(
        "K2_READ",
        OPERATOR,
        "driver",
        "K2 due (~100M decisions)",
        "queued DEV-1000 on the K2 pinned state",
        "operator: read K2 per PREREG-0 P6; a fail relaunches from the last good pin",
        False,
    ),
    Alarm(
        "K2_FAIL",
        OPERATOR,
        "operator",
        "the operator read K2 as a fail (PREREG-0 P6)",
        "stopped the lane (DONE)",
        "operator: fix, then relaunch from the last good pinned state (`ckpt rollback`)",
        True,
    ),
    Alarm(
        "K3_READ",
        OPERATOR,
        "driver",
        "K3 due (1.4B decisions, backstop Thu Nov 5 18:00 CET)",
        "queued DEV-1000 on the K3 pinned state",
        "operator: read K3; on a NO-GO, raise K3_NOGO",
        False,
    ),
    Alarm(
        "K3_NOGO",
        HUMAN,
        "operator",
        "paired DEV-1000 student - C* <= -3pp at K3",
        "none; training continues",
        "user; the operator diagnoses for <= 6 h; relaunch only for a found bug",
        False,
    ),
    Alarm(
        "K4_READ",
        OPERATOR,
        "driver",
        "K4 due (3B decisions, backstop Sat Nov 7 08:00 CET)",
        "the 3B DEV-2000 evals are queued by the 1B rule",
        "operator: read K4; on a NO-GO, raise K4_NOGO",
        False,
    ),
    Alarm(
        "K4_NOGO",
        HUMAN,
        "operator",
        "paired DEV-2000 student < C* at K4",
        "none; training continues",
        "user; the operator diagnoses for <= 6 h; relaunch only for a found bug",
        False,
    ),
    Alarm(
        "STALL_SIGNAL",
        HUMAN,
        "driver, operator",
        "DEV-2000 (EMA0.996) below 11% at 2B or below 14% at 3B decisions (plan §1.4)",
        "none; training continues",
        "user; the operator diagnoses for <= 6 h; relaunch only for a found bug",
        False,
    ),
    Alarm(
        "DEV_JUMP",
        OPERATOR,
        "driver",
        "a DEV-1000 read more than 5pp above the previous DEV-1000 read (plan §12)",
        "none",
        "operator: audit the top runs against C# for a sim exploit; raise SIM_EXPLOIT "
        "if one is found",
        False,
    ),
    Alarm(
        "SIM_EXPLOIT",
        HUMAN,
        "operator",
        "the operator found a sim exploit (DEV jump, F9 hits, odd card or relic use)",
        "none; training continues until the user decides",
        "user: patch, W3, resume from the last clean pinned state (plan §12)",
        False,
    ),
    Alarm(
        "E1_REVERT",
        OPERATOR,
        "driver, operator",
        "E=1 and any type's KL > 0.03 or clipfrac > 0.3 for 20 iterations in a row; or, "
        "at +300M, paired DEV-1000 >= 3.0pp below pre-switch with McNemar p < 0.05",
        "none",
        "operator: revert to E=2 from the e1_switch pinned state (plan §9)",
        False,
    ),
    Alarm(
        "HEARTBEAT_STALE",
        OPERATOR,
        "driver",
        "no heartbeat for 15 min while the learner is alive",
        "none (the watchdog's stall rule should restart it)",
        "operator: read ops/watchdog.log; restart by hand if the watchdog is stuck",
        False,
    ),
    Alarm(
        "WATCHDOG_DOWN",
        OPERATOR,
        "driver",
        "no DONE, but the watchdog is not alive",
        "none",
        "operator: `systemctl --user status e2e-lane@<lane>`; restart the unit",
        False,
    ),
    Alarm(
        "DRIVER_ERROR",
        OPERATOR,
        "driver",
        "the driver raised an exception",
        "none; the traceback is in the alarm file",
        "operator: fix the driver; the lane itself is unaffected",
        False,
    ),
)
ALARMS_BY_NAME = {a.name: a for a in ALARMS}


def alarm_path(directory: str | os.PathLike, name: str) -> Path:
    return Path(directory) / f"{ALARM_PREFIX}{name}"


def write_alarm(
    directory: str | os.PathLike,
    name: str,
    *,
    writer: str,
    reason: str,
    lane: str = "",
    **details: Any,
) -> Path:
    """Raise ``ALARM_<name>`` in ``directory``. A repeat bumps ``count`` and
    ``last_*`` and keeps the first occurrence's fields."""
    if name not in ALARMS_BY_NAME and not _ALARM_NAME.match(name):
        raise ValueError(f"bad alarm name {name!r}")
    spec = ALARMS_BY_NAME.get(name)
    path = alarm_path(directory, name)
    now = utc_now_iso()
    old = read_json(path)
    if isinstance(old, dict):
        old["count"] = int(old.get("count", 1)) + 1
        old["last_time"] = now
        old["last_reason"] = reason
        rec = old
    else:
        rec = {
            "name": name,
            "severity": spec.severity if spec else HUMAN,
            "writer": writer,
            "lane": lane,
            "reason": reason,
            "time": now,
            "count": 1,
            "who_acts": spec.who_acts if spec else "user",
            **details,
        }
    atomic_write_text(path, json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return path


def list_alarms(*directories: str | os.PathLike) -> list[dict[str, Any]]:
    """Every ``ALARM_*`` file in ``directories``, with its severity."""
    out = []
    for d in directories:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob(ALARM_PREFIX + "*")):
            rec = read_json(f)
            name = f.name[len(ALARM_PREFIX) :]
            rec = rec if isinstance(rec, dict) else {"name": name}
            spec = ALARMS_BY_NAME.get(name)
            rec.setdefault("severity", spec.severity if spec else HUMAN)
            rec["path"] = str(f)
            rec["name"] = name
            out.append(rec)
    return out


def write_done(save_dir: str | os.PathLike, reason: str, *, by: str, **extra: Any) -> Path:
    """Write ``DONE`` unless it exists: the first reason wins."""
    path = Path(save_dir) / DONE
    if not path.exists():
        rec = {"reason": reason, "by": by, "time": utc_now_iso(), **extra}
        atomic_write_text(path, json.dumps(rec, sort_keys=True, default=str) + "\n")
    return path


# ── small file helpers (shared by the watchdog, the driver and the learner) ──


def utc_now_iso(t: float | None = None) -> str:
    ts = time.time() if t is None else t
    return _dt.datetime.fromtimestamp(ts, _dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    """Write ``text`` (utf-8) to a temp file next to ``path``, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise
    return path


def read_json(path: str | os.PathLike) -> Any:
    """The parsed JSON at ``path``, or None if it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_lines(path: str | os.PathLike, max_bytes: int = 4 << 20) -> list[str]:
    """The complete lines in the last ``max_bytes`` of ``path``."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]  # the first one is cut
    return [x.decode("utf-8", "replace") for x in lines if x.strip()]


def iter_jsonl(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def rows_since(path: str | os.PathLike, since: float, max_bytes: int = 16 << 20) -> list[dict]:
    """Rows of a jsonl log whose ``time`` is at or after ``since``."""
    return [
        r
        for r in iter_jsonl(tail_lines(path, max_bytes))
        if is_num(r.get("time")) and r["time"] >= since
    ]


def _proc_stat(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after ``comm`` (state is [0]), or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes().decode("utf-8", "replace")
    except OSError:
        return None
    # comm (field 2) may hold spaces and parentheses: split after the last ')'.
    return stat[stat.rfind(")") + 2 :].split()


def proc_starttime(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat`` (start time in clock ticks), or None."""
    rest = _proc_stat(pid)
    try:
        return int(rest[19]) if rest else None
    except (IndexError, ValueError):
        return None


def pid_alive(pid: int | None, starttime: int | None = None) -> bool:
    """True when ``pid`` runs (not a zombie) and, if given, started at ``starttime``."""
    if not pid or pid <= 0:
        return False
    if sys.platform.startswith("linux"):
        rest = _proc_stat(pid)
        if not rest or rest[0] == "Z":
            return False
        return starttime is None or (len(rest) > 19 and rest[19] == str(starttime))
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows.
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    import ctypes

    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        k32.CloseHandle(handle)


def pid_record(pid: int | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    """The JSON a process writes into a pid file (learner.pid, ops/watchdog.json)."""
    pid = os.getpid() if pid is None else pid
    return {
        "pid": pid,
        "starttime": proc_starttime(pid),
        "host": socket.gethostname(),
        "started_at": utc_now_iso(),
        "argv": list(sys.argv if argv is None else argv),
    }


def record_alive(rec: Any) -> bool:
    """True when a pid-file record names a live process."""
    if not isinstance(rec, dict) or not isinstance(rec.get("pid"), int):
        return False
    if rec.get("host") and rec["host"] != socket.gethostname():
        return False
    return pid_alive(rec["pid"], rec.get("starttime"))


def meminfo_available_gb(path: str = "/proc/meminfo") -> float | None:
    """MemAvailable in GB (1e9 bytes), or None if unreadable."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        return None
    return None
