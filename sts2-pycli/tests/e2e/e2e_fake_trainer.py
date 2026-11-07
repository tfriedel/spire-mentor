"""A stand-in learner for the e2e ops tests (WP7): it speaks the ops contract
of :mod:`ppo.e2e.ops_contract` with fake numbers and no torch.

The watchdog starts it the way it starts the real learner::

    python e2e_fake_trainer.py --save-dir DIR --sampler-seed-offset N [--resume PATH] ...

Each start reads ``DIR/fake_plan.json``: ``{"default": {...}, "launches":
{"1": {...}, "2": {...}}}``. The knobs of launch ``n`` are ``DEFAULTS``,
updated by ``default``, updated by ``launches[n]``. Fault knobs (decisions at
which they fire; ``None`` is off):

``crash_at``      exit 3 without saving
``human_at``      write ``ALARM_F9_INVARIANT`` and exit 87
``nan_at``        ``nan_mode`` ``exit``: write ``NAN.json``, exit 86; ``row``:
                  write a non-finite stats row and hang until killed
``stall_at``      stop making progress; ``stall_mode`` ``slow`` keeps writing
                  throughput rows at ``stall_rate``, ``silent`` writes none

Behavior knobs: ``lc_recheck_fail`` (raise ``ALARM_LC_RECHECK_FAIL`` at the
start, as the learner does when a learning check fails after one passed, and
train on), ``ignore_term`` (SIGTERM is ignored), ``done_ignore`` (``DONE``
is ignored), ``eval`` (keep a fake eval child that ``EVALS_PAUSE`` stops),
``spawn_child`` (leave a child process behind in the group), ``alloc_mb``
(hold that much resident memory), ``exit_at_budget`` (exit code at the
budget; default 0, with ``DONE``).

Everything it does goes to ``DIR/fake/log.jsonl``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C

DEFAULTS: dict = {
    "budget": 400,
    "iter_decisions": 10,
    "tick_s": 0.02,
    "state_every": 20,
    "full_every": 50,
    "slim_every": 100,
    "pin_every": 200,
    "state_bytes": 256,
    "throughput_every_s": 0.05,
    "rate": 1000.0,
    "stall_rate": 10.0,
    "crash_at": None,
    "human_at": None,
    "nan_at": None,
    "nan_mode": "exit",
    "stall_at": None,
    "stall_mode": "slow",
    "lc_recheck_fail": False,
    "ignore_term": False,
    "done_ignore": False,
    "eval": False,
    "spawn_child": False,
    "alloc_mb": 0,
    "exit_at_budget": 0,
}
ENV_KEYS = (
    "PYTHONPATH",
    "STS2_WHEEL_PIN",
    "HOME",
    "PATH",
    "LANG",
    "USER",
    *C.LANE_ENVIRONMENT,
)


class Fake:
    def __init__(self, args: argparse.Namespace, extra: list[str]) -> None:
        self.sd = ckpt.SaveDir(args.save_dir)
        self.dir = self.sd.p("fake")
        self.dir.mkdir(parents=True, exist_ok=True)
        launches = self.dir / "launches"
        try:
            self.n = int(launches.read_text(encoding="utf-8").strip()) + 1
        except (OSError, ValueError):
            self.n = 1
        launches.write_text(f"{self.n}\n", encoding="utf-8")
        plan = C.read_json(self.sd.p("fake_plan.json")) or {}
        self.cfg = {
            **DEFAULTS,
            **plan.get("default", {}),
            **plan.get("launches", {}).get(str(self.n), {}),
        }
        self.args = args
        self.extra = extra
        self.offset = args.sampler_seed_offset
        self.stop = False
        self.decisions = 0
        self.it = 0
        self.eval_child: subprocess.Popen | None = None

    def log(self, event: str, **data) -> None:
        rec = {"launch": self.n, "pid": os.getpid(), "time": time.time(), "event": event, **data}
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def state_bytes(self) -> bytes:
        head = json.dumps({"decisions": self.decisions, "it": self.it, "launch": self.n})
        return head.encode("utf-8") + b"\n" + b"x" * int(self.cfg["state_bytes"])

    def save_state(self) -> None:
        ckpt.save_state(self.sd, self.state_bytes(), decisions=self.decisions, it=self.it)

    def stats_row(self, **over) -> dict:
        return C.minimal_stats_row(
            self.it,
            self.decisions,
            pid=os.getpid(),
            dec_per_s=float(self.cfg["rate"]),
            sampler_seed_offset=self.offset,
            **over,
        )

    def append(self, rel: str, row: dict) -> None:
        with open(self.sd.p(rel), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def evals(self) -> None:
        paused = (self.sd.root / C.EVALS_PAUSE).exists()
        running = self.eval_child is not None and self.eval_child.poll() is None
        if paused and running:
            assert self.eval_child is not None
            self.eval_child.kill()
            self.eval_child.wait()
            self.log("eval_stopped", child=self.eval_child.pid)
            self.eval_child = None
        elif not paused and not running:
            self.eval_child = subprocess.Popen(["sleep", "600"])
            self.log("eval_started", child=self.eval_child.pid)

    def run(self) -> int:
        cfg, sd = self.cfg, self.sd
        env = {k: os.environ[k] for k in ENV_KEYS if k in os.environ}
        self.log(
            "start",
            argv=sys.argv[1:],
            resume=self.args.resume,
            seed_offset=self.offset,
            extra=self.extra,
            env=env,
            env_keys=sorted(os.environ),
            cwd=os.getcwd(),
            sid=os.getsid(0),
            pgid=os.getpgid(0),
        )
        lock = open(sd.p(C.LEARNER_LOCK), "a", encoding="utf-8")  # noqa: SIM115
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.log("refused", reason="learner.lock is held")
            return C.EXIT_USAGE
        if sd.state.exists() and not self.args.resume:
            self.log("refused", reason="state.pt exists and no --resume")
            return C.EXIT_USAGE
        C.atomic_write_text(sd.root / C.LEARNER_PID, json.dumps(C.pid_record()) + "\n")
        if self.args.resume:
            head = Path(self.args.resume).read_bytes().split(b"\n", 1)[0]
            st = json.loads(head)
            self.decisions, self.it = int(st["decisions"]), int(st["it"])
            self.log("resumed", decisions=self.decisions, it=self.it, src_launch=st.get("launch"))
        if cfg["lc_recheck_fail"]:  # learner.py's call, in `Learner.__init__`
            C.write_alarm(
                sd.root, "LC_RECHECK_FAIL", writer="learner", reason="fake re-check fail",
                severity=C.OPERATOR, who_acts="operator", verdict="fake_lc.json",
            )  # fmt: skip
            self.log("lc_recheck_fail")
        parent = os.getppid()

        def on_term(signum, _frame):
            self.stop = True
            self.log("signal", signum=signum)

        if cfg["ignore_term"]:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        else:
            signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)
        if cfg["spawn_child"]:
            child = subprocess.Popen(["sleep", "600"])
            (self.dir / f"child_{self.n}.pid").write_text(f"{child.pid}\n", encoding="utf-8")
            self.log("child", child=child.pid)
        hog = b"x" * (int(cfg["alloc_mb"]) << 20) if cfg["alloc_mb"] else b""
        if hog:
            self.log("alloc", mb=len(hog) >> 20)
        stalled = False
        hung = False
        last_tp = 0.0

        def hit(knob: str) -> bool:
            at = cfg[knob]
            return at is not None and self.decisions >= int(at)

        while True:
            if os.getppid() != parent:
                self.log("orphaned")
                return 5
            if self.stop:
                if not hung:
                    self.save_state()
                self.log("stopped", decisions=self.decisions)
                return C.EXIT_OK
            if (sd.root / C.DONE).exists() and not cfg["done_ignore"]:
                self.save_state()
                self.log("done_seen", decisions=self.decisions)
                return C.EXIT_OK
            if cfg["eval"]:
                self.evals()
            now = time.time()
            if stalled or hung:
                if (
                    stalled
                    and cfg["stall_mode"] == "slow"
                    and now - last_tp >= cfg["throughput_every_s"]
                ):
                    tp = {"time": now, "pid": os.getpid(), "decisions": self.decisions,
                          "rate": float(cfg["stall_rate"])}  # fmt: skip
                    self.append(C.THROUGHPUT, tp)
                    last_tp = now
                time.sleep(cfg["tick_s"])
                continue
            before = self.decisions
            self.decisions += int(cfg["iter_decisions"])
            self.it += 1
            if hit("crash_at"):
                self.log("crash", decisions=self.decisions)
                os._exit(3)
            if hit("human_at"):
                C.write_alarm(sd.root, "F9_INVARIANT", writer="learner", reason="fake hit")
                self.log("human", decisions=self.decisions)
                return C.EXIT_HUMAN
            if hit("nan_at"):
                if cfg["nan_mode"] == "exit":
                    rec = {"decisions": self.decisions, "it": self.it, "what": "loss"}
                    C.atomic_write_text(sd.root / C.NAN_RECORD, json.dumps(rec) + "\n")
                    self.log("nan_exit", decisions=self.decisions)
                    return C.EXIT_NAN
                nan = float("nan")
                row = self.stats_row(nonfinite=1, losses={"policy": nan, "total": nan})
                self.append(C.STATS, row)
                self.log("nan_row", decisions=self.decisions)
                hung = True
                continue
            if hit("stall_at"):
                stalled = True
                self.log("stall", decisions=self.decisions)
            self.append(C.STATS, self.stats_row())
            if ckpt.crossed(before, self.decisions, int(cfg["state_every"])):
                self.save_state()
            if ckpt.crossed(before, self.decisions, int(cfg["full_every"])):
                ckpt.save_full(sd, self.state_bytes(), decisions=self.decisions)
            if ckpt.crossed(before, self.decisions, int(cfg["slim_every"])):
                ckpt.save_slim(sd, self.state_bytes()[:64], decisions=self.decisions)
            if ckpt.crossed(before, self.decisions, int(cfg["pin_every"])):
                ckpt.pin(sd, decisions=self.decisions, reason="milestone", data=self.state_bytes())
            if (sd.root / C.PIN_REQUEST).exists():
                ckpt.pin(sd, decisions=self.decisions, reason="request", data=self.state_bytes())
                (sd.root / C.PIN_REQUEST).unlink(missing_ok=True)
                self.log("pin_request", decisions=self.decisions)
            if now - last_tp >= cfg["throughput_every_s"]:
                tp = {"time": now, "pid": os.getpid(), "decisions": self.decisions,
                      "rate": float(cfg["rate"]), "it": self.it}  # fmt: skip
                self.append(C.THROUGHPUT, tp)
                last_tp = now
            hb = {"time": now, "pid": os.getpid(), "decisions": self.decisions, "it": self.it}
            C.atomic_write_text(sd.root / C.HEARTBEAT, json.dumps(hb) + "\n")
            if self.decisions >= int(cfg["budget"]):
                self.save_state()
                C.write_done(sd.root, "budget", by="learner", decisions=self.decisions)
                self.log("budget", decisions=self.decisions)
                return int(cfg["exit_at_budget"])
            time.sleep(cfg["tick_s"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(C.FLAG_SAVE_DIR, dest="save_dir", required=True)
    ap.add_argument(C.FLAG_RESUME, dest="resume", default=None)
    ap.add_argument(C.FLAG_SEED_OFFSET, dest="sampler_seed_offset", type=int, required=True)
    args, extra = ap.parse_known_args()
    fake = Fake(args, extra)
    try:
        rc = fake.run()
    finally:
        if fake.eval_child is not None and fake.eval_child.poll() is None:
            with contextlib.suppress(OSError):
                fake.eval_child.kill()
    fake.log("exit", rc=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
