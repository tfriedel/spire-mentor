"""The campaign driver (scripts/e2e_driver.py): each pre-registered rule, the
eval queue, gate flags, acks, status.json and CAMPAIGN_LOG.md."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C

PYCLI = Path(__file__).resolve().parents[2]
SCRIPT = PYCLI / "scripts" / "e2e_driver.py"

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the driver runs on the Linux box")


def _load_driver():
    spec = importlib.util.spec_from_file_location("e2e_driver", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
T0 = 1_762_000_000.0  # 2025-11-01, before every K-gate backstop


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    r = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
        check=True,
    )  # fmt: skip
    return r.stdout.strip()


class Camp:
    """A temp campaign dir with one lane deployed from a temp git worktree."""

    def __init__(self, root: Path, lane: str = "main") -> None:
        self.root = root / "campaign"
        self.wt = root / "wt"
        (self.wt / "sts2-pycli" / "scripts").mkdir(parents=True)
        (self.wt / "sts2-pycli" / "scripts" / "e2e_watchdog.sh").write_text(
            "#!/bin/sh\n", encoding="utf-8"
        )
        _git(self.wt, "init", "-q")
        _git(self.wt, "add", "-A")
        _git(self.wt, "commit", "-q", "-m", "init")
        self.pin = root / "pin" / "abc"
        (self.pin / "sts2_rs").mkdir(parents=True)
        (self.pin / "sts2_rs" / "sts2_rs.abi3.so").write_bytes(b"\x7fELF fake")
        self.args = root / "lane.args"
        self.args.write_text("--budget-decisions\n9000000000\n", encoding="utf-8")
        self.meminfo = root / "meminfo"
        self.set_ram(100)
        self.lane = lane
        self.save = self.root / C.LANES_DIR / lane
        self.now = T0
        self.lim = C.Limits(meminfo=str(self.meminfo), disk_prune_gb=0, disk_stop_gb=0)
        rc = self.cli("init-lane", lane, "--worktree", str(self.wt), "--python", sys.executable,
                      "--pin", str(self.pin), "--args-file", str(self.args))  # fmt: skip
        assert rc == 0
        self.alive_watchdog()

    def set_ram(self, gb: float) -> None:
        self.meminfo.write_text(f"MemAvailable: {int(gb * 1e9 / 1024)} kB\n", encoding="utf-8")

    def cli(self, *args: str) -> int:
        return D.main(["--campaign", str(self.root), "--now", str(self.now), *args])

    def campaign(self) -> D.Campaign:
        return D.Campaign(self.root, self.now, self.lim)

    def run(self, dt: float = 300.0) -> dict:
        self.now += dt
        return D.run_pass(self.campaign())

    def alive_watchdog(self) -> None:
        rec = {"watchdog": C.pid_record(), "learner": None, "time": C.utc_now_iso()}
        C.atomic_write_text(self.save / C.WATCHDOG_STATE, json.dumps(rec))

    def dead_watchdog(self) -> None:
        rec = {"watchdog": {**C.pid_record(), "pid": 2**22 + 12345}, "time": C.utc_now_iso()}
        C.atomic_write_text(self.save / C.WATCHDOG_STATE, json.dumps(rec))

    def learner_alive(self) -> None:
        C.atomic_write_text(self.save / C.LEARNER_PID, json.dumps(C.pid_record()))

    def row_time(self) -> float:
        """Row times increase like a learner's, and stay after the deploy time."""
        self.last_row = max(getattr(self, "last_row", 0.0), self.now) + 0.01
        return self.last_row

    def rows(self, n: int, *, start_it: int = 1, decisions: int | None = None, **over) -> None:
        with open(self.save / C.STATS, "a", encoding="utf-8") as f:
            for i in range(n):
                it = start_it + i
                dec = decisions if decisions is not None else it * 1_000_000
                row = C.minimal_stats_row(it, dec, time=self.row_time(), **over)
                f.write(json.dumps(row) + "\n")

    def heartbeat(self, age_s: float = 0.0) -> None:
        C.atomic_write_text(self.save / C.HEARTBEAT, json.dumps({"time": self.now - age_s}))

    def alarm(self, name: str) -> dict | None:
        return C.read_json(C.alarm_path(self.save, name))

    def done(self) -> dict | None:
        return C.read_json(self.save / C.DONE)

    def queue(self) -> list[dict]:
        q = self.save / C.EVAL_QUEUE
        return [C.read_json(f) for f in sorted(q.glob("*.json"))] if q.is_dir() else []


@pytest.fixture
def camp(tmp_path):
    return Camp(tmp_path)


def test_init_lane_writes_the_lane_files_and_the_deploy_record(camp):
    env = (camp.save / C.LANE_ENV).read_text(encoding="utf-8")
    assert f"E2E_WORKTREE={camp.wt.resolve()}" in env
    assert f"E2E_PIN={camp.pin.resolve()}" in env
    assert "E2E_REQUIRE_CLEAN=1" in env
    assert (camp.save / C.LANE_ARGS).read_text(encoding="utf-8") == camp.args.read_text(
        encoding="utf-8"
    )
    st = C.read_json(camp.root / C.CAMPAIGN_STATE)["lanes"]["main"]
    assert st["deploy"]["sha"] == _git(camp.wt, "rev-parse", "HEAD")
    assert st["deploy"]["wheel_digest"] == D.wheel_digest(camp.pin)
    # A lane that ran is redeployed only as a hot-swap.
    ckpt.save_state(camp.save, b"s", decisions=5_000_000, it=5)
    args = ["--worktree", str(camp.wt), "--python", sys.executable, "--args-file", str(camp.args)]
    assert camp.cli("init-lane", "main", *args) == 2
    camp.learner_alive()
    assert camp.cli("init-lane", "main", *args, "--hot-swap") == 2  # still runs
    (camp.save / C.LEARNER_PID).unlink()
    camp.dead_watchdog()
    assert camp.cli("init-lane", "main", *args, "--hot-swap") == 0
    assert [r.reason for r in ckpt.pinned_states(camp.save)] == ["hot_swap"]
    st = C.read_json(camp.root / C.CAMPAIGN_STATE)["lanes"]["main"]
    assert len(st["deploys"]) == 1 and st["deploy"]["hot_swap"] is True


def test_init_lane_refuses_a_dirty_worktree(tmp_path):
    c = Camp(tmp_path)
    (c.wt / "sts2-pycli" / "scripts" / "e2e_watchdog.sh").write_text("changed", encoding="utf-8")
    assert c.cli("init-lane", "other", "--worktree", str(c.wt), "--python", sys.executable,
                 "--args-file", str(c.args)) == 2  # fmt: skip


def test_healthy_lane_is_ok(camp):
    camp.learner_alive()
    camp.rows(10)
    camp.heartbeat()
    status = camp.run(dt=1)
    assert status["worst"] == "OK", status["alarms"]
    ln = status["lanes"]["main"]
    assert ln["decisions"] == 10_000_000
    assert ln["learner_alive"] and ln["watchdog_alive"]
    assert ln["next_gate"]["name"] == "K2"
    assert C.read_json(camp.root / C.STATUS)["summary"].startswith("OK | main 0.010B")
    assert camp.done() is None


@pytest.mark.parametrize(
    ("over", "name"),
    [
        ({"lr": {"trunk": 1.0000021e-4}, "lr_closed_form": {"trunk": 1e-4}}, "LR_CLOSED_FORM"),
        (
            {"lr": {"trunk": 1e-4, "value": 3e-4}, "lr_closed_form": {"trunk": 1e-4}},
            "LR_CLOSED_FORM",
        ),
        ({"wheel_mismatch": "worker 3 ran wheel b"}, "WHEEL_MISMATCH"),
        ({"source_mismatch": "worker 3 ran py_src x"}, "SOURCE_MISMATCH"),
        ({"f9": {"checked": 100, "hits": 1}}, "F9_INVARIANT"),
    ],
)
def test_human_rules_stop_the_lane(camp, over, name):
    camp.rows(3)
    camp.rows(1, start_it=4, **over)
    status = camp.run()
    assert camp.alarm(name)["severity"] == C.HUMAN
    assert camp.done()["reason"] == f"driver:{name}"
    assert status["worst"] == C.HUMAN
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert f"ALARM_{name} (HUMAN, driver)" in log


def test_lr_within_tolerance_and_zero_lr_pass(camp):
    camp.rows(1, lr={"trunk": 1.0000005e-4, "w": 0.0}, lr_closed_form={"trunk": 1e-4, "w": 0.0})
    assert camp.run()["worst"] == "OK"
    assert camp.done() is None


def test_stamp_drift_and_game_version(camp):
    camp.rows(2)
    camp.run()
    assert camp.done() is None
    stamps = C.minimal_stats_row(1, 1)["stamps"]
    camp.rows(1, start_it=3, stamps={**stamps, "wheel": "1" * 16})
    camp.run()
    assert camp.alarm("WHEEL_MISMATCH") is not None
    c2 = Camp(camp.root.parent / "two")
    c2.rows(1, stamps={**stamps, "game_version": "v0.107.1"})
    c2.run()
    assert c2.alarm("GAME_VERSION")["severity"] == C.HUMAN
    c3 = Camp(camp.root.parent / "three")
    c3.rows(1)
    c3.rows(1, start_it=2, stamps={**stamps, "py_src": "f" * 16})
    c3.run()
    assert c3.alarm("SOURCE_MISMATCH") is not None


def test_a_hot_swap_resets_the_stamp_baseline(camp):
    stamps = C.minimal_stats_row(1, 1)["stamps"]
    camp.rows(2)
    camp.run()
    camp.dead_watchdog()
    args = ["--worktree", str(camp.wt), "--python", sys.executable, "--args-file", str(camp.args)]
    camp.now += 60
    assert camp.cli("init-lane", "main", *args, "--hot-swap") == 0
    camp.now += 60
    camp.rows(2, start_it=3, stamps={**stamps, "py_src": "a" * 16, "schema_sha256": "b" * 64})
    camp.alive_watchdog()
    camp.run()
    assert camp.alarm("SOURCE_MISMATCH") is None
    assert camp.done() is None


def test_worktree_moves_and_pin_changes_are_human(camp):
    (camp.wt / "new.txt").write_text("x", encoding="utf-8")
    _git(camp.wt, "add", "new.txt")
    _git(camp.wt, "commit", "-q", "-m", "moved")
    camp.run()
    assert "moved to" in camp.alarm("SOURCE_MISMATCH")["reason"]
    c2 = Camp(camp.root.parent / "two")
    (c2.pin / "sts2_rs" / "sts2_rs.abi3.so").write_bytes(b"\x7fELF other")
    c2.run()
    assert c2.alarm("WHEEL_MISMATCH") is not None
    c3 = Camp(camp.root.parent / "three")
    (c3.wt / "sts2-pycli" / "scripts" / "e2e_watchdog.sh").write_text("edit", encoding="utf-8")
    c3.run()
    assert "uncommitted" in c3.alarm("SOURCE_MISMATCH")["reason"]


def test_caps_need_a_large_enough_window(camp):
    caps = {"fights": 200, "fights_capped": 1, "runs": 20, "runs_capped": 0}
    camp.rows(50, caps=caps)  # 50/10000 fights = 0.5%
    camp.run()
    assert "fights capped" in camp.alarm("CAPS")["reason"]
    assert camp.done() is None  # OPERATOR: the lane runs on
    c2 = Camp(camp.root.parent / "two")
    c2.rows(50, caps={"fights": 50, "fights_capped": 1, "runs": 5, "runs_capped": 1})
    c2.run()
    assert c2.alarm("CAPS") is None  # 2500 fights, 250 runs: under the floors


def test_contract_heartbeat_and_watchdog_down(camp):
    row = C.minimal_stats_row(1, 1, time=camp.row_time())
    del row["kl"]
    with open(camp.save / C.STATS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    camp.learner_alive()
    camp.heartbeat(age_s=10_000)
    camp.dead_watchdog()
    camp.run()
    assert "missing kl" in camp.alarm("CONTRACT")["reason"]
    assert "min old" in camp.alarm("HEARTBEAT_STALE")["reason"]
    assert camp.alarm("WATCHDOG_DOWN") is None  # debounced: down for less than 10 min
    camp.run(dt=D.WATCHDOG_DOWN_AFTER_S)
    assert "learner" in camp.alarm("WATCHDOG_DOWN")["reason"]
    assert camp.done() is None


def test_milestone_pins_queue_evals_once(camp):
    ckpt.pin(camp.save, decisions=250_000_128, reason="milestone", data=b"m250")
    ckpt.pin(camp.save, decisions=500_000_064, reason="milestone", data=b"m500")
    ckpt.pin(camp.save, decisions=1_000_000_064, reason="milestone", data=b"m1000")
    camp.rows(1, decisions=1_011_000_000)
    camp.run()
    q = camp.queue()
    got = sorted((r["decisions"], r["panel"], r["variant"]) for r in q)
    assert got == [
        (250_000_128, "DEV-1000", "ema0996"),
        (500_000_064, "DEV-1000", "ema0996"),
        (1_000_000_064, "DEV-2000", "ema0996"),
        (1_000_000_064, "DEV-2000", "ema0999"),
        (1_000_000_064, "DEV-2000", "raw"),
    ]
    assert all(r["seed_start"] == C.DEV_START and r["nice"] == C.EVAL_NICE for r in q)
    assert {r["n"] for r in q if r["panel"] == "DEV-2000"} == {2000}
    # 750M has no pin: PIN_REQUEST plus an alarm.
    assert "750,000,000 milestone" in camp.alarm("PIN_MISSING")["reason"]
    assert C.read_json(camp.save / C.PIN_REQUEST)["by"] == "driver"
    camp.rows(1, start_it=2, decisions=1_012_000_000)
    camp.run()
    assert len(camp.queue()) == 5  # nothing queued twice


def test_rollback_drops_queued_requests_and_rearms_the_gate(camp):
    p0 = ckpt.pin(camp.save, decisions=50_000_000, reason="restart", data=b"r50")
    ckpt.pin(camp.save, decisions=100_000_000, reason="k2", data=b"k2")
    camp.rows(1, decisions=101_000_000)
    camp.run()
    assert camp.alarm("K2_READ")["decisions"] == 100_000_000
    assert [r["reason"] for r in camp.queue()] == ["k2"]
    (camp.save / f"{C.ALARM_PREFIX}K2_READ").unlink()
    ckpt.rollback(camp.save, p0, reason="k2_relaunch")  # abandons the K2 pin
    camp.rows(1, start_it=2, decisions=60_000_000)
    camp.run()
    assert camp.queue() == []
    assert camp.alarm("K2_READ") is None
    st = C.read_json(camp.root / C.CAMPAIGN_STATE)["lanes"]["main"]
    assert "K2" not in st["flags"]
    ckpt.pin(camp.save, decisions=100_000_032, reason="k2", data=b"k2 again")
    camp.rows(1, start_it=3, decisions=101_000_000)
    camp.run()
    assert camp.alarm("K2_READ")["decisions"] == 100_000_032
    assert len(camp.queue()) == 1


def test_k3_backstop_requests_a_pin_then_flags(camp):
    ckpt.pin(camp.save, decisions=100_000_000, reason="k2", data=b"k2")
    for m in (250, 500, 750):
        ckpt.pin(camp.save, decisions=m * 1_000_000, reason="milestone", data=b"m%d" % m)
    camp.now = C.K_GATES[1].backstop.timestamp() + 60
    camp.rows(1, decisions=900_000_000)
    camp.run(dt=0)
    assert C.read_json(camp.save / C.PIN_REQUEST)["reason"] == "k3"
    assert camp.alarm("K3_READ") is None
    assert camp.alarm("K2_READ") is not None
    # The learner serves the request.
    (camp.save / C.PIN_REQUEST).unlink()
    ckpt.pin(camp.save, decisions=905_000_000, reason="k3", data=b"k3")
    camp.rows(1, start_it=2, decisions=905_000_000)
    camp.run()
    alarm = camp.alarm("K3_READ")
    assert alarm["decisions"] == 905_000_000
    assert "backstop" in alarm["reason"]
    k3 = [r for r in camp.queue() if r["reason"] == "k3"]
    assert [(r["decisions"], r["panel"], r["variant"]) for r in k3] == [
        (905_000_000, "DEV-1000", "ema0996")
    ]
    assert camp.alarm("PIN_MISSING") is None


def test_backstops_without_a_watchdog(camp):
    camp.learner_alive()
    camp.dead_watchdog()
    camp.set_ram(10)
    camp.rows(1)
    camp.heartbeat()
    camp.run()
    assert C.read_json(camp.save / C.EVALS_PAUSE)["by"] == "driver"
    assert camp.done() is None  # evals first
    camp.run()
    assert camp.alarm("RAM_STOP")["writer"] == "driver"
    assert camp.done()["reason"] == "driver:RAM_STOP"


def test_backstops_leave_a_live_watchdog_alone(camp):
    camp.learner_alive()
    camp.set_ram(10)
    camp.rows(1)
    camp.heartbeat()
    camp.run()
    camp.run()
    assert not (camp.save / C.EVALS_PAUSE).exists()
    assert camp.alarm("RAM_STOP") is None


def test_ack_rules(camp):
    camp.rows(1, f9={"checked": 1, "hits": 2})
    caps = {"fights": 200, "fights_capped": 2, "runs": 20, "runs_capped": 0}
    camp.rows(50, start_it=2, caps=caps)
    camp.run()
    assert camp.cli("ack", "F9_INVARIANT", "--lane", "main", "--note", "seen") == 2
    assert camp.alarm("F9_INVARIANT") is not None
    assert camp.cli("ack", "F9_INVARIANT", "--lane", "main", "--note", "seen",
                    "--decision", "user: fix the engine, relaunch from the K2 pin") == 0  # fmt: skip
    assert camp.alarm("F9_INVARIANT") is None
    assert camp.done() is not None  # an ack never restarts the lane
    acked = list((camp.root / C.ACKED_DIR).glob("*F9_INVARIANT.json"))
    assert C.read_json(acked[0])["ack"]["decision"].startswith("user:")
    assert camp.cli("ack", "CAPS", "--lane", "main", "--note", "known", "--mute-hours", "2") == 0
    camp.rows(1, start_it=60, caps=caps)
    camp.run()
    assert camp.alarm("CAPS") is None  # muted
    camp.now += 3 * 3600
    camp.rows(1, start_it=61, caps=caps)
    camp.run(dt=0)
    assert camp.alarm("CAPS") is not None
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert "acked ALARM_F9_INVARIANT" in log and "decision: user:" in log


def test_campaign_log_tails_watchdog_events(camp):
    ev = camp.save / C.EVENTS
    ev.parent.mkdir(parents=True, exist_ok=True)
    with open(ev, "a", encoding="utf-8") as f:
        for kind in ("start", "tick_noise", "crash"):
            f.write(json.dumps({"time": camp.now + 1, "kind": kind, "rc": 3}) + "\n")
    camp.run()
    camp.run()
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert log.count("watchdog learner start") == 1
    assert log.count("watchdog crash: rc=3") == 1
    assert "tick_noise" not in log


def test_a_broken_lane_raises_driver_error_and_the_rest_still_reports(camp, monkeypatch):
    other = camp.root / C.LANES_DIR / "broken"
    other.mkdir(parents=True)
    (other / C.LANE_ENV).write_text("E2E_PY=/bin/false\n", encoding="utf-8")
    real = D.lane_pass

    def flaky(c, lane_dir):
        if lane_dir.name == "broken":
            raise RuntimeError("boom")
        return real(c, lane_dir)

    monkeypatch.setattr(D, "lane_pass", flaky)
    status = camp.run()
    assert "main" in status["lanes"]
    assert "broken: RuntimeError('boom')" in status["errors"]
    assert C.read_json(C.alarm_path(camp.root, "DRIVER_ERROR"))["severity"] == C.OPERATOR


def test_cli_run_and_status(camp):
    camp.rows(2)
    env = {**os.environ, "E2E_WD_MEMINFO": str(camp.meminfo)}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--campaign", str(camp.root), "run"],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK | main")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--campaign", str(camp.root), "status"],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120, check=False,
    )  # fmt: skip
    assert r.returncode == 0 and "min old" in r.stdout


def _result(camp: Camp, pin, panel: str, variant: str, wins: int, runs: int | None = None) -> None:
    n = C.EVAL_PANELS[panel]["n"]
    rid = f"main_d{pin.decisions:013d}_{panel}_{variant}_milestone"
    rec = {"id": rid, "lane": "main", "state": str(pin.path(camp.save)), "sha256": pin.sha256,
           "decisions": pin.decisions, "panel": panel, "n": n, "variant": variant,
           "reason": "milestone", "wins": wins, "runs": n if runs is None else runs}  # fmt: skip
    C.atomic_write_text(camp.save / C.EVAL_DONE / f"{rid}.json", json.dumps(rec))


def test_eval_results_raise_the_stall_signal_and_the_dev_jump(camp):
    for m in (250, 500, 1000, 2000):
        ckpt.pin(camp.save, decisions=m * 1_000_000, reason="milestone", data=b"%d" % m)
    pins = {r.decisions // 1_000_000: r for r in ckpt.pinned_states(camp.save)}
    camp.rows(1, decisions=2_011_000_000)
    _result(camp, pins[250], "DEV-1000", "ema0996", 100)
    _result(camp, pins[500], "DEV-1000", "ema0996", 140, runs=900)  # partial: not read
    _result(camp, pins[1000], "DEV-2000", "ema0996", 190)  # 9.5%: no floor at 1B
    _result(camp, pins[2000], "DEV-2000", "raw", 150)  # 7.5%, but not the default model
    status = camp.run()
    assert camp.alarm("STALL_SIGNAL") is None and camp.alarm("DEV_JUMP") is None
    assert [r["decisions"] for r in status["lanes"]["main"]["eval_reads"]] == [
        250_000_000, 1_000_000_000, 2_000_000_000,
    ]  # fmt: skip
    _result(camp, pins[500], "DEV-1000", "ema0996", 160)  # complete now: 10% -> 16%
    _result(camp, pins[2000], "DEV-2000", "ema0996", 210)  # 10.5% < 11% at 2B
    _result(camp, pins[1000], "DEV-2000", "ema0999", 200)  # 10% at 1B ...
    _result(camp, pins[2000], "DEV-2000", "ema0999", 320)  # ... 16% at 2B: the median slope
    camp.run()
    stall = camp.alarm("STALL_SIGNAL")
    assert stall["severity"] == C.HUMAN and "10.5% < 11%" in stall["reason"]
    jump = camp.alarm("DEV_JUMP")
    assert "DEV-1000 ema0996 10.0% -> 16.0%" in jump["reason"]
    assert jump["count"] == 1  # DEV-2000 reads 1B apart don't count
    assert camp.done() is None  # training continues (PREREG-0 P6)
    camp.run()
    assert camp.alarm("STALL_SIGNAL")["count"] == 1  # once per result
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert "eval DEV-2000 ema0996 at 2,000,000,000: 210/2000 = 10.5%" in log
    # A rollback's abandoned lineage drops out of the reads.
    ckpt.rollback(camp.save, pins[1000].path(camp.save), reason="test")
    camp.rows(1, start_it=2, decisions=1_100_000_000)
    reads = camp.run()["lanes"]["main"]["eval_reads"]
    assert max(r["decisions"] for r in reads) == 1_000_000_000


def test_raise_records_operator_verdicts(camp):
    assert camp.cli("raise", "K3_NOGO", "--lane", "main", "--reason", "student - C* = -3.4pp") == 0
    rec = camp.alarm("K3_NOGO")
    assert (rec["severity"], rec["writer"]) == (C.HUMAN, "operator")
    assert camp.done() is None  # diagnose <= 6 h while training continues
    assert camp.run()["worst"] == C.HUMAN
    assert camp.cli("raise", "CAPS", "--lane", "main", "--reason", "x") == 2  # driver-only
    assert camp.cli("raise", "K2_FAIL", "--lane", "nope", "--reason", "x") == 2
    assert camp.cli("ack", "K3_NOGO", "--lane", "main", "--note", "read") == 2
    assert camp.cli("raise", "K2_FAIL", "--lane", "main", "--reason", "agreement 71%") == 0
    assert camp.done()["reason"] == "operator:K2_FAIL"
    assert camp.cli("ack", "K2_FAIL", "--lane", "main", "--note", "x", "--mute-hours", "1") == 2
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert "ALARM_K2_FAIL (OPERATOR, operator): agreement 71%" in log


def test_stop_and_clear_done(camp):
    assert camp.cli("clear-done", "main", "--note", "x") == 2  # no DONE yet
    assert camp.cli("stop", "main", "--reason", "final", "--note", "Mon 06:00") == 0
    assert camp.done()["by"] == "operator" and camp.done()["reason"] == "final"
    assert camp.cli("stop", "main", "--reason", "again") == 0
    assert camp.done()["reason"] == "final"  # the first reason wins
    camp.cli("raise", "K2_FAIL", "--lane", "main", "--reason", "EV off")
    assert camp.cli("clear-done", "main", "--note", "fixed") == 2  # the alarm is open
    assert camp.cli("ack", "K2_FAIL", "--lane", "main", "--note", "fixed in abc123") == 0
    camp.cli("raise", "K3_NOGO", "--lane", "main", "--reason", "diagnosing")  # doesn't block
    camp.learner_alive()
    assert camp.cli("clear-done", "main", "--note", "fixed") == 2  # the learner runs
    (camp.save / C.LEARNER_PID).unlink()
    assert camp.cli("clear-done", "main", "--note", "relaunch", "--reset-starts") == 0
    assert camp.done() is None
    hist = list((camp.save / C.OPS_DIR / "done_history").glob("*_DONE.json"))
    assert C.read_json(hist[0])["done"]["reason"] == "final"
    starts = (camp.save / C.STARTS).read_text(encoding="utf-8")
    assert starts.startswith(C.STARTS_RESET)
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert "operator wrote DONE (final)" in log and "cleared DONE (final): relaunch" in log


def test_e1_revert_rule(camp):
    hot = {"kl": {"combat": 0.01, "ooc": 0.04}, "clipfrac": {"combat": 0.1, "ooc": 0.1}}
    camp.rows(19, epochs=1, **hot)
    camp.rows(1, start_it=20, epochs=1, kl={"combat": 0.01}, clipfrac={"combat": 0.35})
    camp.run()
    assert "20 iterations" in camp.alarm("E1_REVERT")["reason"]
    assert camp.done() is None
    camp2 = Camp(camp.root.parent / "two")
    camp2.rows(19, epochs=1, **hot)
    camp2.rows(1, start_it=20, epochs=1)  # one quiet iteration breaks the run
    camp2.rows(19, start_it=21, epochs=1, **hot)
    camp2.run()
    assert camp2.alarm("E1_REVERT") is None
    camp3 = Camp(camp.root.parent / "three")
    camp3.rows(20, epochs=2, **hot)  # E=2: the rule does not apply
    camp3.run()
    assert camp3.alarm("E1_REVERT") is None


def test_commands_hold_the_driver_lock(camp, monkeypatch):
    import fcntl

    monkeypatch.setattr(D, "LOCK_WAIT_S", 0.3)
    camp.rows(1)
    with open(camp.root / D.DRIVER_LOCK, "a", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert camp.cli("run") == 0  # a timer pass skips
        assert not (camp.root / C.STATUS).exists()
        assert camp.cli("note", "main", "parity", "S1a", "PASS") == 2  # waited, then gave up
    assert camp.cli("note", "main", "parity", "S1a", "PASS") == 0
    assert camp.cli("note", "nope", "x") == 2
    log = (camp.root / C.CAMPAIGN_LOG).read_text(encoding="utf-8")
    assert "main: operator: parity S1a PASS" in log
    assert camp.cli("run") == 0
    assert (camp.root / C.STATUS).exists()


def test_lc_recheck_fail_is_operator_and_blocks_nothing(camp):
    """The learner raises ``LC_RECHECK_FAIL`` when a learning check fails after
    one passed, and trains on. The driver reports it as OPERATOR and writes no
    ``DONE``; it does not block ``clear-done``, and the operator acks it without
    a user decision (a mute is allowed)."""
    camp.rows(3)
    C.write_alarm(  # learner.py's call
        camp.save, "LC_RECHECK_FAIL", writer="learner",
        reason="lc_k2.json: the learning check failed after one passed",
        severity=C.OPERATOR, who_acts="operator", verdict="lc_k2.json",
    )  # fmt: skip
    status = camp.run()
    assert status["worst"] == C.OPERATOR
    assert [a["name"] for a in status["alarms"]] == ["LC_RECHECK_FAIL"]
    assert camp.done() is None
    C.write_done(camp.save, "budget", by="learner")
    assert camp.cli("clear-done", "main", "--note", "next leg") == 0
    assert camp.done() is None
    assert camp.alarm("LC_RECHECK_FAIL") is not None  # still open: clear-done skips it
    assert camp.cli("ack", "LC_RECHECK_FAIL", "--lane", "main", "--note", "read it",
                    "--mute-hours", "6") == 0  # fmt: skip
    assert camp.alarm("LC_RECHECK_FAIL") is None


def test_held_row_leak_rule(camp):
    """``update.rows_overdue`` above 0 in each of the last 7 consecutive
    iterations raises ``HELD_ROW_LEAK`` (OPERATOR, the lane runs on); 6 do not,
    and neither does a gap in ``it`` or a row without the field."""
    leak = {"update": {"rows_overdue": 12, "rows_held": 900}}
    camp.rows(3)
    camp.rows(C.HELD_LEAK_ITERS - 1, start_it=4, **leak)
    camp.run()
    assert camp.alarm("HELD_ROW_LEAK") is None
    camp.rows(1, start_it=3 + C.HELD_LEAK_ITERS, **leak)
    status = camp.run()
    rec = camp.alarm("HELD_ROW_LEAK")
    assert rec["severity"] == C.OPERATOR and f"for {C.HELD_LEAK_ITERS} iterations" in rec["reason"]
    assert "its 4-10" in rec["reason"] and camp.done() is None
    assert status["worst"] == C.OPERATOR
    c2 = Camp(camp.root.parent / "two")
    c2.rows(3, **leak)
    c2.rows(C.HELD_LEAK_ITERS - 3, start_it=5, **leak)  # it 4 is missing: a restart
    c2.run()
    assert c2.alarm("HELD_ROW_LEAK") is None
    c3 = Camp(camp.root.parent / "three")
    c3.rows(C.HELD_LEAK_ITERS - 1, **leak)
    c3.rows(1, start_it=C.HELD_LEAK_ITERS)  # no `update` field
    c3.run()
    assert c3.alarm("HELD_ROW_LEAK") is None


def _algo_streams() -> Any:
    """tests/e2e/test_algo.py as a module, for its `Stream` (registered first: it
    defines dataclasses)."""
    name = "e2e_test_algo_streams"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / "test_algo.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _overdue_rows(stream: Any, n_iter: int, lose_at: int | None = None) -> list[dict]:
    """One stats row per iteration of `stream` through the hold-back (`update`'s
    ``rows_overdue``). At ``lose_at`` an env loses its run (it starts a new one)
    and nothing drops the lost run's held rows: a planted leak."""
    from ppo.e2e import algo

    hold = algo.HoldBack(algo.HOLD_ITERS)
    rows = []
    for k in range(n_iter):
        if k == lose_at:
            uids = set(hold.cols["run_uid"].tolist())
            e = next(e for e, r in enumerate(stream.cur) if r is not None and r.uid in uids)
            stream.cur[e] = None
        table = hold.join(stream.next_table())
        ret = algo.compute_returns(table, stream.values()[table["ref"]])
        rows.append({"it": k + 1, "update": {"rows_overdue": ret.stats(table)["rows_overdue"]}})
        hold.carry(table, ret)
    return rows


def _first_alarm(rows: list[dict]) -> int | None:
    """The first iteration at which the rule fires on the stats tail so far."""
    for i in range(len(rows)):
        if D.held_leak_problem(rows[: i + 1]) is not None:
            return rows[i]["it"]
    return None


def test_held_row_leak_is_quiet_on_the_algo_streams_and_fires_on_a_leak():
    """On test_algo.py's streams with worker stalls (T1/T3's and T11's, as long
    as those tests run) rows go overdue in short bursts only: no alarm. A lost
    run nothing drops stays overdue, and the rule fires HELD_LEAK_ITERS
    iterations after its rows first go overdue."""
    m = _algo_streams()
    t1 = m.Stream(12, 64, seed=3, run_len=(80, 260), practice_p=0.2, stall_p=0.08)
    t11 = m.Stream(8, 40, seed=11, run_len=(60, 160), practice_p=0.15, stall_p=0.05)
    for stream, n in ((t1, 70), (t11, 154)):
        rows = _overdue_rows(stream, n)
        assert any(r["update"]["rows_overdue"] for r in rows)  # the bursts are there
        assert _first_alarm(rows) is None
    rows = _overdue_rows(
        m.Stream(12, 64, seed=3, run_len=(80, 260), practice_p=0.2, stall_p=0.08), 70, lose_at=30
    )
    first_overdue = next(r["it"] for r in rows[30:] if r["update"]["rows_overdue"])
    assert all(r["update"]["rows_overdue"] for r in rows if r["it"] >= first_overdue)
    assert _first_alarm(rows) == first_overdue + C.HELD_LEAK_ITERS - 1
