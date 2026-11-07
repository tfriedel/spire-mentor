"""Failure paths of the e2e watchdog, end to end (plan §8: "each failure path
tested before T0").

Each test runs the real ``scripts/e2e_watchdog.sh`` on a temp save dir with a
fake learner (``e2e_fake_trainer.py``), short time constants (``E2E_WD_*``),
a fake ``/proc/meminfo`` and, where a test needs it, a disk quota.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C
from ppo.e2e import watchdog as W

HERE = Path(__file__).resolve().parent
PYCLI = HERE.parents[1]
SCRIPT = PYCLI / "scripts" / "e2e_watchdog.sh"
FAKE = HERE / "e2e_fake_trainer.py"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not all(shutil.which(t) for t in ("bash", "setsid", "flock")),
    reason="the watchdog needs Linux, bash, setsid and flock",
)

FAST = {
    "E2E_WD_POLL_S": "0.1",
    "E2E_WD_RESTART_DELAY_S": "0.05",
    "E2E_WD_STOP_GRACE_S": "10",
    "E2E_WD_STALL_STOP_GRACE_S": "5",
    "E2E_WD_DONE_GRACE_S": "1",
    "E2E_WD_STALL_FRAC": "0",  # off unless a test turns it on
    "E2E_WD_DISK_PRUNE_GB": "0",
    "E2E_WD_DISK_STOP_GB": "0",
    "E2E_WD_RAM_GRACE_S": "1.5",
    "E2E_WD_RSS_MAX_GB": "0",  # off unless a test turns it on
}


class Lane:
    """One temp lane: save dir, lane files, fake meminfo, watchdog processes."""

    def __init__(
        self,
        root: Path,
        plan: dict | None = None,
        env: dict[str, str] | None = None,
        lane_env: dict[str, str] | None = None,
        args: list[str] | None = None,
        learner_env: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        (root / "lanes" / "main").mkdir(parents=True)
        self.save = (root / "lanes" / "main").resolve()
        self.meminfo = root / "meminfo"
        self.set_ram(100)
        lines = {"E2E_PY": sys.executable, "E2E_REQUIRE_CLEAN": "0", **(lane_env or {})}
        text = "# test lane\n" + "".join(f"{k}={v}\n" for k, v in lines.items())
        (self.save / C.LANE_ENV).write_text(text, encoding="utf-8")
        if args is not None:
            (self.save / C.LANE_ARGS).write_text("".join(a + "\n" for a in args), encoding="utf-8")
        if learner_env:
            text = "".join(f"{k}={v}\n" for k, v in learner_env.items())
            (self.save / C.LANE_LEARNER_ENV).write_text(text, encoding="utf-8")
        self.write_plan(plan or {})
        base = {k: v for k, v in os.environ.items() if not k.startswith("E2E_")}
        self.env = {
            **base,
            **FAST,
            "E2E_WD_MEMINFO": str(self.meminfo),
            "E2E_TEST_LEAK": "1",
            **(env or {}),
        }
        self.procs: list[subprocess.Popen] = []

    def write_plan(self, plan: dict) -> None:
        C.atomic_write_text(self.save / "fake_plan.json", json.dumps(plan))

    def set_ram(self, gb: float) -> None:
        kb = int(gb * 1e9 / 1024)
        C.atomic_write_text(self.meminfo, f"MemTotal: {kb * 2} kB\nMemAvailable: {kb} kB\n")

    def start(self) -> subprocess.Popen:
        out = open(self.root / f"watchdog_{len(self.procs)}.out", "w", encoding="utf-8")  # noqa: SIM115
        p = subprocess.Popen(
            ["bash", str(SCRIPT), str(self.save), "--", sys.executable, str(FAKE)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=self.env,
            cwd=self.root,
            start_new_session=True,
        )
        out.close()
        self.procs.append(p)
        return p

    def wait(self, p: subprocess.Popen, timeout: float = 90) -> int:
        try:
            return p.wait(timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"watchdog still runs after {timeout} s\n{self.dump()}")

    def run(self, timeout: float = 90) -> int:
        return self.wait(self.start(), timeout)

    def dump(self) -> str:
        parts = []
        for f in sorted(self.root.glob("watchdog_*.out")):
            parts.append(f"--- {f.name}\n{f.read_text(encoding='utf-8')[-4000:]}")
        for rel in (C.LEARNER_LOG, C.EVENTS, "fake/log.jsonl"):
            p = self.save / rel
            if p.exists():
                parts.append(f"--- {rel}\n{p.read_text(encoding='utf-8')[-4000:]}")
        return "\n".join(parts)

    def log(self) -> list[dict]:
        return list(C.iter_jsonl(C.tail_lines(self.save / "fake" / "log.jsonl")))

    def starts(self) -> list[dict]:
        return [r for r in self.log() if r["event"] == "start"]

    def events(self) -> list[dict]:
        return list(C.iter_jsonl(C.tail_lines(self.save / C.EVENTS)))

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.events()]

    def alarm(self, name: str) -> dict | None:
        return C.read_json(C.alarm_path(self.save, name))

    def done(self) -> dict | None:
        return C.read_json(self.save / C.DONE)

    def wait_for(self, pred, timeout: float = 30, what: str = "") -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pred():
                return
            time.sleep(0.05)
        pytest.fail(f"timed out waiting for {what or pred}\n{self.dump()}")

    def decisions(self) -> int:
        rows = list(C.iter_jsonl(C.tail_lines(self.save / C.STATS, 1 << 16)))
        return rows[-1]["decisions"] if rows else 0

    def cleanup(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                with contextlib.suppress(OSError):
                    os.killpg(p.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    p.wait(5)
        pids = set()
        for r in self.log():
            pids.add(r.get("pid"))
            pids.add(r.get("child"))
        for f in (self.save / "fake").glob("child_*.pid"):
            with contextlib.suppress(ValueError):
                pids.add(int(f.read_text(encoding="utf-8")))
        for pid in pids:
            if not isinstance(pid, int) or not C.pid_alive(pid):
                continue
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            except OSError:
                continue
            if b"e2e_fake_trainer" in cmd or cmd.startswith(b"sleep\x00600"):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


@pytest.fixture
def make_lane(tmp_path):
    lanes: list[Lane] = []

    def make(**kw) -> Lane:
        lane = Lane(tmp_path / f"lane{len(lanes)}", **kw)
        lanes.append(lane)
        return lane

    yield make
    for lane in lanes:
        lane.cleanup()


def _pinned(lane: Lane, reason: str | None = None) -> list[ckpt.ManifestRow]:
    return [r for r in ckpt.pinned_states(lane.save) if reason is None or r.reason == reason]


# ── crash, DONE, crash loop ─────────────────────────────────────────────────


def test_crash_resumes_from_state_pt(make_lane):
    lane = make_lane(plan={"launches": {"1": {"crash_at": 130}}})
    assert lane.run() == 0, lane.dump()
    starts = lane.starts()
    assert len(starts) == 2
    assert starts[0]["resume"] is None
    assert starts[1]["resume"] == str(lane.save / C.STATE)
    resumed = [r for r in lane.log() if r["event"] == "resumed"]
    assert resumed[0]["decisions"] == 120  # the last 20-multiple saved before the crash
    assert [r.decisions for r in _pinned(lane, "restart")] == [120]
    assert "crash" in lane.kinds()
    assert lane.done()["reason"] == "budget"
    assert ckpt.verify_pinned(lane.save) == []


def test_done_before_start_starts_nothing(make_lane):
    lane = make_lane()
    C.write_done(lane.save, "operator", by="test")
    assert lane.run() == 0
    assert lane.starts() == []


def test_done_mid_run_stops_the_lane(make_lane):
    lane = make_lane(plan={"default": {"budget": 10**9}})
    p = lane.start()
    lane.wait_for(lambda: lane.decisions() > 50, what="training")
    C.write_done(lane.save, "operator", by="test")
    assert lane.wait(p) == 0, lane.dump()
    assert [r["event"] for r in lane.log()][-2:] == ["done_seen", "exit"]
    assert len(lane.starts()) == 1
    assert (lane.save / C.STATE).is_file()


def test_done_ignored_is_terminated_after_the_grace(make_lane):
    lane = make_lane(plan={"default": {"budget": 10**9, "done_ignore": True}})
    p = lane.start()
    lane.wait_for(lambda: lane.decisions() > 50, what="training")
    C.write_done(lane.save, "operator", by="test")
    assert lane.wait(p) == 0, lane.dump()
    assert "done_grace" in lane.kinds()
    assert any(r["event"] == "signal" for r in lane.log())


def test_crash_loop_stops_at_the_fifth_start_and_reset_clears_it(make_lane):
    lane = make_lane(plan={"default": {"crash_at": 10}})
    assert lane.run() == C.WD_EXIT_CRASH_LOOP, lane.dump()
    assert len(lane.starts()) == 4
    assert lane.alarm("CRASH_LOOP")["severity"] == C.OPERATOR
    assert lane.done()["reason"] == "watchdog:CRASH_LOOP"
    assert lane.run() == 0  # DONE stops every later start too
    assert len(lane.starts()) == 4
    # The operator fixes the cause, clears DONE and resets the count.
    (lane.save / C.DONE).unlink()
    W.reset_starts(ckpt.SaveDir(lane.save), "fixed in the test")
    lane.write_plan({"default": {"crash_at": None, "budget": 100}})
    assert lane.run() == 0, lane.dump()
    assert len(lane.starts()) == 5
    assert lane.done()["reason"] == "budget"


def test_a_learner_alarm_that_stops_nothing_leaves_the_lane_running(make_lane):
    """``LC_RECHECK_FAIL`` (the learner raises it at a start whose learning check
    failed after one passed, and trains on) is OPERATOR: the watchdog neither
    stops nor restarts the learner, which runs to its budget."""
    lane = make_lane(plan={"default": {"lc_recheck_fail": True}})
    assert lane.run() == 0, lane.dump()
    assert len(lane.starts()) == 1
    rec = lane.alarm("LC_RECHECK_FAIL")
    assert (rec["severity"], rec["writer"], rec["count"]) == (C.OPERATOR, "learner", 1)
    assert not C.ALARMS_BY_NAME["LC_RECHECK_FAIL"].stops_lane
    assert lane.done()["reason"] == "budget"
    assert "alarm" not in lane.kinds() and "crash" not in lane.kinds()


def test_learner_human_exit_stops_the_lane(make_lane):
    lane = make_lane(plan={"default": {"human_at": 50}})
    assert lane.run() == C.WD_EXIT_LEARNER_HUMAN, lane.dump()
    assert lane.alarm("LEARNER_HUMAN")["severity"] == C.HUMAN
    assert lane.alarm("F9_INVARIANT")["writer"] == "learner"
    assert lane.done()["reason"] == "watchdog:LEARNER_HUMAN"
    assert len(lane.starts()) == 1


# ── disk ─────────────────────────────────────────────────────────────────────


def test_low_disk_prunes_then_stops_and_keeps_pinned_states(make_lane):
    plan = {
        "default": {
            "budget": 10**9,
            "state_bytes": 100_000,
            "full_every": 50,
            "pin_every": 100,
            "slim_every": 100,
        }
    }
    env = {
        "E2E_DISK_QUOTA_GB": "0.002",
        "E2E_WD_DISK_PRUNE_GB": "0.0008",
        "E2E_WD_DISK_STOP_GB": "0.0004",
    }
    lane = make_lane(plan=plan, env=env)
    assert lane.run() == C.WD_EXIT_DISK, lane.dump()
    assert lane.alarm("DISK_PRUNED") is not None
    assert lane.alarm("DISK_STOP") is not None
    assert lane.done()["reason"] == "watchdog:DISK_STOP"
    pinned = _pinned(lane, "milestone")
    assert len(pinned) >= 5
    assert ckpt.verify_pinned(lane.save) == []  # pruning never touches pinned states
    assert (lane.save / C.STATE).is_file()
    assert len(ckpt.SaveDir(lane.save).listing("full")) <= C.FULL_KEEP


# ── stall ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["slow", "silent"])
def test_stall_restarts_the_learner(make_lane, mode):
    # The healthy phase (90 iterations) must outlast the arming delay so the
    # process's median stays the healthy rate.
    plan = {"default": {"budget": 1000}, "launches": {"1": {"stall_at": 900, "stall_mode": mode}}}
    env = {
        "E2E_WD_STALL_FRAC": "0.5",
        "E2E_WD_STALL_GRACE_S": "1.5",
        "E2E_WD_STALL_WINDOW_S": "0.3",
    }
    lane = make_lane(plan=plan, env=env)
    assert lane.run() == 0, lane.dump()
    assert "stall" in lane.kinds()
    starts = lane.starts()
    assert len(starts) == 2
    assert starts[1]["resume"] == str(lane.save / C.STATE)
    resumed = [r for r in lane.log() if r["event"] == "resumed"]
    assert resumed[0]["decisions"] == 900  # SIGTERM made it save at the stall point
    assert lane.done()["reason"] == "budget"


# ── NaN ──────────────────────────────────────────────────────────────────────

# With full states every 50 (keep 3) and milestone pins every 200, a NaN at
# 330 leaves full states at 200 (pinned), 250 and 300: the rollback goes to 250.


def test_nan_exit_rolls_back_to_the_second_newest_full_state(make_lane):
    plan = {
        "default": {"budget": 500},
        "launches": {"1": {"nan_at": 330}, "2": {"crash_at": 300}},
    }
    lane = make_lane(plan=plan)
    assert lane.run() == 0, lane.dump()
    starts = lane.starts()
    assert len(starts) == 3
    assert starts[0]["seed_offset"] == 0
    # Launch 2 resumes from the pinned copy of full_d...250 with seed offset +1e6.
    assert starts[1]["seed_offset"] == 1_000_000
    assert Path(starts[1]["resume"]).name == ckpt.ckpt_name("pinned", 250, "nan_rollback")
    resumed = [r for r in lane.log() if r["event"] == "resumed"]
    assert [r["decisions"] for r in resumed] == [250, 280]
    # Launch 2 crashed after saving state.pt at 280: launch 3 resumes from it,
    # the rollback record is consumed, the offset stays.
    assert starts[2]["resume"] == str(lane.save / C.STATE)
    assert starts[2]["seed_offset"] == 1_000_000
    assert not (lane.save / C.RESUME_NEXT).exists()
    assert len(list((lane.save / C.RESUME_USED_DIR).iterdir())) == 1
    alarm = lane.alarm("NAN_ROLLBACK")
    assert alarm["severity"] == C.OPERATOR
    assert Path(alarm["target"]).name == ckpt.ckpt_name("full", 250)
    quarantined = {Path(p).name for p in alarm["quarantined"]}
    assert {C.STATE, ckpt.STATE_META, ckpt.ckpt_name("full", 300)} <= quarantined
    nan_events = list(C.iter_jsonl(C.tail_lines(lane.save / C.NAN_EVENTS)))
    assert [e["decisions"] for e in nan_events] == [330]
    assert list((lane.save / C.OPS_DIR).glob("NAN_*.json"))
    offsets = {
        r["sampler_seed_offset"]
        for r in C.iter_jsonl(C.tail_lines(lane.save / C.STATS))
        if r["pid"] == starts[1]["pid"]
    }
    assert offsets == {1_000_000}


def test_nan_row_is_killed_unsaved_and_rolled_back(make_lane):
    plan = {"default": {"budget": 500}, "launches": {"1": {"nan_at": 330, "nan_mode": "row"}}}
    lane = make_lane(plan=plan)
    assert lane.run() == 0, lane.dump()
    first = [r["event"] for r in lane.log() if r["launch"] == 1]
    assert "nan_row" in first
    assert "stopped" not in first  # SIGKILL, so nothing was saved after the NaN
    starts = lane.starts()
    assert len(starts) == 2
    assert starts[1]["seed_offset"] == 1_000_000
    assert lane.alarm("NAN_ROLLBACK") is not None


def test_second_nan_within_the_window_is_human(make_lane):
    plan = {"default": {"budget": 500}, "launches": {"1": {"nan_at": 330}, "2": {"nan_at": 300}}}
    lane = make_lane(plan=plan)
    assert lane.run() == C.WD_EXIT_NAN_HUMAN, lane.dump()
    assert lane.alarm("NAN_HUMAN")["severity"] == C.HUMAN
    assert lane.done()["reason"] == "watchdog:NAN_HUMAN"
    assert len(lane.starts()) == 2


def test_second_nan_outside_the_window_rolls_back_again(make_lane):
    plan = {"default": {"budget": 500}, "launches": {"1": {"nan_at": 330}, "2": {"nan_at": 380}}}
    lane = make_lane(plan=plan, env={"E2E_WD_NAN_WINDOW_DECISIONS": "40"})
    assert lane.run() == 0, lane.dump()
    starts = lane.starts()
    assert [s["seed_offset"] for s in starts] == [0, 1_000_000, 2_000_000]
    resumed = [r["decisions"] for r in lane.log() if r["event"] == "resumed"]
    assert resumed == [250, 300]
    assert lane.alarm("NAN_ROLLBACK")["count"] == 2


# ── RAM and memory ───────────────────────────────────────────────────────────


def test_low_ram_stops_evals_first_then_the_lane(make_lane):
    lane = make_lane(plan={"default": {"budget": 10**9, "eval": True}})
    p = lane.start()
    lane.wait_for(lambda: any(r["event"] == "eval_started" for r in lane.log()), what="eval")
    lane.set_ram(10)
    assert lane.wait(p) == C.WD_EXIT_RAM, lane.dump()
    log = lane.log()
    stopped = [r for r in log if r["event"] == "eval_stopped"]
    assert stopped, "the evals did not stop before the lane"
    alarm_time = C.alarm_path(lane.save, "RAM_STOP").stat().st_mtime
    assert stopped[0]["time"] < alarm_time
    assert lane.done()["reason"] == "watchdog:RAM_STOP"
    assert "evals_paused" in lane.kinds()


def test_ram_recovery_lifts_the_eval_pause_with_hysteresis(make_lane):
    lane = make_lane(
        plan={"default": {"budget": 10**9, "eval": True}}, env={"E2E_WD_RAM_GRACE_S": "60"}
    )
    p = lane.start()
    lane.wait_for(lambda: any(r["event"] == "eval_started" for r in lane.log()), what="eval")
    lane.set_ram(10)
    lane.wait_for(lambda: any(r["event"] == "eval_stopped" for r in lane.log()), what="pause")
    lane.set_ram(22)  # above the stop line, below the resume line: the pause stays
    time.sleep(0.6)
    assert (lane.save / C.EVALS_PAUSE).exists()
    lane.set_ram(30)
    lane.wait_for(lambda: not (lane.save / C.EVALS_PAUSE).exists(), what="lifted pause")
    lane.wait_for(
        lambda: sum(r["event"] == "eval_started" for r in lane.log()) >= 2, what="eval restart"
    )
    C.write_done(lane.save, "operator", by="test")
    assert lane.wait(p) == 0, lane.dump()
    assert lane.alarm("RAM_STOP") is None
    assert {"evals_paused", "evals_resumed"} <= set(lane.kinds())


def test_memory_cap_stops_the_lane(make_lane):
    lane = make_lane(
        plan={"default": {"budget": 10**9, "alloc_mb": 160}}, env={"E2E_WD_RSS_MAX_GB": "0.1"}
    )
    assert lane.run() == C.WD_EXIT_RSS, lane.dump()
    assert lane.alarm("RSS_STOP") is not None
    assert lane.done()["reason"] == "watchdog:RSS_STOP"


# ── signals, lock, environment ──────────────────────────────────────────────


def test_sigterm_stops_cleanly_and_the_next_start_resumes(make_lane):
    lane = make_lane(plan={"default": {"budget": 10**9}, "launches": {"2": {"budget": 0}}})
    p = lane.start()
    lane.wait_for(lambda: lane.decisions() > 100, what="training")
    p.send_signal(signal.SIGTERM)
    assert lane.wait(p) == 0, lane.dump()
    assert lane.done() is None  # a signal is not DONE: the unit resumes on its next start
    stopped = next(r for r in lane.log() if r["event"] == "stopped")
    assert lane.run() == 0, lane.dump()
    starts = lane.starts()
    assert starts[1]["resume"] == str(lane.save / C.STATE)
    resumed = next(r for r in lane.log() if r["event"] == "resumed")
    assert resumed["decisions"] == stopped["decisions"]


def test_sigterm_escalates_to_sigkill_of_the_whole_group(make_lane):
    plan = {"default": {"budget": 10**9, "ignore_term": True, "spawn_child": True}}
    lane = make_lane(plan=plan, env={"E2E_WD_STOP_GRACE_S": "1"})
    p = lane.start()
    lane.wait_for(lambda: (lane.save / "fake" / "child_1.pid").exists(), what="child")
    lane.wait_for(lambda: lane.decisions() > 50, what="training")
    child = int((lane.save / "fake" / "child_1.pid").read_text(encoding="utf-8"))
    learner = lane.starts()[0]["pid"]
    p.send_signal(signal.SIGTERM)
    assert lane.wait(p) == 0, lane.dump()
    lane.wait_for(lambda: not C.pid_alive(child) and not C.pid_alive(learner), timeout=10)
    assert "SIGKILL" in (lane.save / C.WATCHDOG_LOG).read_text(encoding="utf-8")


def test_second_watchdog_on_the_same_lane_is_refused(make_lane):
    lane = make_lane(plan={"default": {"budget": 10**9}})
    p = lane.start()
    lane.wait_for(lambda: lane.decisions() > 20, what="training")
    assert lane.run(timeout=30) == C.WD_EXIT_LOCKED
    assert len(lane.starts()) == 1
    p.send_signal(signal.SIGTERM)
    assert lane.wait(p) == 0


def test_learner_env_is_the_allowlist_and_runs_in_its_own_session(make_lane, tmp_path):
    pin = tmp_path / "pin"
    (pin / "sts2_rs").mkdir(parents=True)
    (pin / "sts2_rs" / "__init__.py").write_text("", encoding="utf-8")
    lane = make_lane(
        plan={"default": {"budget": 50}},
        lane_env={"E2E_PIN": str(pin)},
        args=["--note", "two words"],
        learner_env={"EXTRA_FOR_LEARNER": "1"},
    )
    assert lane.run() == 0, lane.dump()
    start = lane.starts()[0]
    env = start["env"]
    assert env["PYTHONPATH"] == f"{pin.resolve()}:python"
    assert env["STS2_WHEEL_PIN"] == str(pin.resolve())
    for k, v in C.LANE_ENVIRONMENT.items():
        assert env[k] == v
    assert "EXTRA_FOR_LEARNER" in start["env_keys"]
    assert "E2E_TEST_LEAK" not in start["env_keys"]
    assert not any(k.startswith("E2E_") for k in start["env_keys"])
    assert start["extra"] == ["--note", "two words"]
    assert start["cwd"] == str(PYCLI)
    assert start["sid"] == start["pid"] == start["pgid"]
    assert start["argv"][:2] == [C.FLAG_SAVE_DIR, str(lane.save)]


def test_a_pin_that_does_not_serve_sts2_rs_is_refused(make_lane, tmp_path):
    pin = tmp_path / "empty_pin"
    pin.mkdir()
    lane = make_lane(lane_env={"E2E_PIN": str(pin)})
    assert lane.run() == C.WD_EXIT_USAGE
    assert lane.starts() == []
    assert "refusing" in (lane.save / C.WATCHDOG_LOG).read_text(encoding="utf-8")
