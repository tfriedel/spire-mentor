"""The watchdog's decision rules (ppo.e2e.watchdog) as plain functions, and the
ops contract (ppo.e2e.ops_contract) they share with the learner and driver."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C
from ppo.e2e import watchdog as W

PYCLI = Path(__file__).resolve().parents[2]
REPO = PYCLI.parent
EXP = REPO / "docs" / "experiments" / "2026-11-03-ironclad-e2e-v111"
FAKE = Path(__file__).resolve().parent / "e2e_fake_trainer.py"


@pytest.fixture
def lane(tmp_path):
    sd = ckpt.SaveDir(tmp_path / "lane")
    sd.p(C.OPS_DIR).mkdir(parents=True)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 97656250 kB\n", encoding="utf-8")  # 100 GB
    lim = C.Limits(meminfo=str(meminfo), disk_prune_gb=0, disk_stop_gb=0, rss_max_gb=0)
    return sd, lim, meminfo


def _set_ram(meminfo: Path, gb: float) -> None:
    meminfo.write_text(f"MemAvailable: {int(gb * 1e9 / 1024)} kB\n", encoding="utf-8")


# ── the contract ────────────────────────────────────────────────────────────


def test_minimal_stats_row_conforms_and_problems_are_named():
    row = C.minimal_stats_row(3, 30)
    assert C.check_stats_row(row) == []
    bad = dict(row)
    del bad["kl"]
    bad["it"] = 1.5
    bad["stamps"] = {"py_src": "x"}
    problems = C.check_stats_row(bad)
    assert "missing kl" in problems
    assert any(p.startswith("it:") for p in problems)
    assert "stamps: missing wheel" in problems
    assert C.check_throughput_row({"time": 1.0, "pid": 1, "decisions": 2, "rate": 3}) == []
    assert C.check_throughput_row({"time": 1.0}) == [
        "missing pid",
        "missing decisions",
        "missing rate",
    ]


def test_nonfinite_rows():
    assert not C.row_is_nonfinite(C.minimal_stats_row(1, 1))
    assert C.row_is_nonfinite(C.minimal_stats_row(1, 1, nonfinite=1))
    assert C.row_is_nonfinite(C.minimal_stats_row(1, 1, losses={"value": math.nan}))
    assert C.row_is_nonfinite(C.minimal_stats_row(1, 1, grad_norm=math.inf))
    # Parsed from JSON, as the watchdog reads it.
    text = json.dumps(C.minimal_stats_row(1, 1, kl={"combat": math.nan}))
    assert C.row_is_nonfinite(json.loads(text))


def test_alarm_table_is_consistent():
    names = [a.name for a in C.ALARMS]
    assert len(names) == len(set(names))
    for a in C.ALARMS:
        assert re.fullmatch(r"[A-Z0-9_]+", a.name)
        assert a.severity in (C.HUMAN, C.OPERATOR)
        assert a.who_acts
        assert set(a.writer.split(", ")) <= {"watchdog", "driver", "learner", "operator"}
        # A stop writes DONE, except the watchdog's refusal to start.
        assert ("DONE" in a.action) == a.stops_lane or a.name == "RESUME_MISSING", a.name
    # The stops PREREG-0 P6 and plan §8 send to the user are HUMAN.
    for name in (
        "NAN_HUMAN",
        "LR_CLOSED_FORM",
        "WHEEL_MISMATCH",
        "SOURCE_MISMATCH",
        "F9_INVARIANT",
    ):
        spec = C.ALARMS_BY_NAME[name]
        assert (spec.severity, spec.stops_lane) == (C.HUMAN, True), name
    # K3, K4 and the stall branch: HUMAN, but training continues (P6, §1.4).
    for name in ("K3_NOGO", "K4_NOGO", "STALL_SIGNAL"):
        spec = C.ALARMS_BY_NAME[name]
        assert (spec.severity, spec.stops_lane) == (C.HUMAN, False), name
        assert "operator" in spec.writer
    # A K2 fail relaunches from the last good pin, so it stops the lane.
    assert C.ALARMS_BY_NAME["K2_FAIL"].stops_lane


def test_the_held_row_leak_window_follows_the_hold_back():
    """A held row is overdue once it waited more than HOLD_ITERS + 2 iterations;
    `HELD_ROW_LEAK` waits HOLD_ITERS + 3 in a row (a worker stall of d >= 3
    iterations is d + 1 in a row; the driver tests run it on test_algo's streams)."""
    from ppo.e2e import algo

    assert C.HELD_LEAK_ITERS == algo.HOLD_ITERS + 3
    spec = C.ALARMS_BY_NAME["HELD_ROW_LEAK"]
    assert (spec.severity, spec.writer, spec.stops_lane) == (C.OPERATOR, "driver", False)
    assert f"({C.HELD_LEAK_ITERS})" in spec.trigger
    spec = C.ALARMS_BY_NAME["LC_RECHECK_FAIL"]
    assert (spec.severity, spec.writer, spec.stops_lane) == (C.OPERATOR, "learner", False)


def test_the_runbook_covers_every_alarm():
    runbook = EXP / "RUNBOOK.md"
    if not runbook.exists():
        pytest.skip("RUNBOOK.md is not in this tree")
    text = runbook.read_text(encoding="utf-8")
    missing = [a.name for a in C.ALARMS if f"`{a.name}`" not in text]
    assert missing == []


def _alarm_names_in(text: str) -> set[str]:
    # ALARM_BEGIN and ALARM_END frame the alarm files in the poller's ssh output.
    return {
        m for m in re.findall(r"ALARM_([A-Z0-9_]+)", text) if m not in ("PREFIX", "BEGIN", "END")
    }


def test_every_alarm_named_in_code_and_docs_is_in_the_table():
    sources = [
        PYCLI / "python" / "ppo" / "e2e" / "watchdog.py",
        PYCLI / "scripts" / "e2e_watchdog.sh",
        PYCLI / "scripts" / "e2e_driver.py",
        PYCLI / "scripts" / "e2e_poll.sh",
        EXP / "RUNBOOK.md",
    ]
    names: set[str] = set()
    for f in sources:
        if f.exists():
            names |= _alarm_names_in(f.read_text(encoding="utf-8"))
    for name in re.findall(r'_(?:stop|alarm|refuse)\([^"]*"([A-Z0-9_]+)"', _read(sources[0])):
        names.add(name)
    unknown = sorted(n for n in names if n not in C.ALARMS_BY_NAME and not n.endswith("_"))
    assert unknown == []


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_write_alarm_counts_repeats_and_done_keeps_the_first_reason(tmp_path):
    p = C.write_alarm(tmp_path, "CAPS", writer="driver", reason="first")
    C.write_alarm(tmp_path, "CAPS", writer="driver", reason="second")
    rec = C.read_json(p)
    assert (rec["count"], rec["reason"], rec["last_reason"]) == (2, "first", "second")
    assert rec["severity"] == C.OPERATOR
    with pytest.raises(ValueError):
        C.write_alarm(tmp_path, "bad name", writer="x", reason="y")
    C.write_done(tmp_path, "budget", by="learner")
    C.write_done(tmp_path, "operator", by="operator")
    assert C.read_json(tmp_path / C.DONE)["reason"] == "budget"
    listed = C.list_alarms(tmp_path)
    assert [a["name"] for a in listed] == ["CAPS"]


def test_limits_from_env_and_shell():
    lim = C.Limits.from_env({"E2E_WD_POLL_S": "0.5", "E2E_WD_STARTS_MAX": "7", "E2E_WD_X": "1"})
    assert lim.poll_s == pytest.approx(0.5)
    assert lim.starts_max == 7
    lines = dict(line.split("=", 1) for line in lim.shell().splitlines())
    assert lines["WD_POLL_S"] == "0.5"
    assert lines["WD_POLL_MS"] == "500"
    assert lines["WD_STOP_GRACE_MS"] == "600000"
    assert lines["WD_STARTS_MAX"] == "7"
    assert C.Limits() == C.Limits.from_env({})


def test_pre_registered_values():
    """The numbers plan §8 and PREREG-0 fix; a change here is a plan change."""
    lim = C.Limits()
    assert (lim.starts_max, lim.starts_window_s) == (4, 7200.0)
    assert (lim.disk_prune_gb, lim.disk_stop_gb, lim.ram_stop_gb) == (45.0, 30.0, 20.0)
    assert (lim.stall_grace_s, lim.stall_window_s, lim.stall_frac) == (1800.0, 600.0, 0.5)
    assert (lim.nan_window_decisions, lim.nan_seed_step) == (100_000_000, 1_000_000)
    assert lim.rss_max_gb == pytest.approx(200.0)
    assert (C.STATE_EVERY, C.FULL_EVERY, C.FULL_KEEP) == (5_000_000, 25_000_000, 3)
    assert (C.PIN_EVERY, C.SLIM_EVERY) == (250_000_000, 100_000_000)
    assert repr(C.LR_TOLERANCE) == "1e-06"
    assert C.LANE_ENVIRONMENT["STS2_GAME_VERSION"] == "v0.111.0"
    assert C.LANE_ENVIRONMENT["STS2_PPO_FEATURES"] == "card_mods,reward_upgrade"
    assert C.LANE_ENVIRONMENT["STS2_RUNAI_POTION_SWAP"] == "1"
    assert C.LANE_ENVIRONMENT["OMP_NUM_THREADS"] == "1"
    assert [g.name for g in C.K_GATES] == ["K2", "K3", "K4"]
    assert [g.at_decisions for g in C.K_GATES] == [100_000_000, 1_400_000_000, 3_000_000_000]


def test_worker_thread_env_holds_for_the_block_only(monkeypatch):
    """The worker thread caps (B3) are in the environment inside the block, for the
    workers it spawns; after it, raised or not, the old values are back."""
    caps = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    assert dict.fromkeys(caps, "1") == C.WORKER_THREAD_ENV
    for k in caps:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # the learner's own, say
    before = dict(os.environ)
    with C.worker_thread_env():
        assert {k: os.environ.get(k) for k in caps} == C.WORKER_THREAD_ENV
    assert dict(os.environ) == before
    with pytest.raises(OSError, match="no fork"), C.worker_thread_env():
        raise OSError("no fork")
    assert dict(os.environ) == before


def test_systemd_unit_matches_the_exit_codes():
    unit = EXP / "systemd" / "e2e-lane@.service"
    if not unit.exists():
        pytest.skip("the systemd templates are not in this checkout")
    text = _read(unit)
    m = re.search(r"^RestartPreventExitStatus=(.+)$", text, re.M)
    assert m and tuple(int(x) for x in m.group(1).split()) == C.WD_NO_RESTART_CODES
    assert "Restart=on-failure" in text
    assert "MemoryMax=200G" in text
    assert "/home/" not in text
    stop = re.search(r"^TimeoutStopSec=(\d+)$", text, re.M)
    assert stop and int(stop.group(1)) > C.Limits().stop_grace_s


def test_pid_helpers():
    me = C.pid_record()
    assert C.record_alive(me)
    if sys.platform.startswith("linux"):  # a reused pid shows a new start time
        assert not C.record_alive({**me, "starttime": (me["starttime"] or 0) + 1})
    assert not C.record_alive({**me, "host": "elsewhere"})
    assert not C.record_alive(None)
    assert not C.pid_alive(0)


# ── watchdog rules ──────────────────────────────────────────────────────────


def _tp(t: float, rate: float) -> dict:
    return {"time": t, "pid": 1, "decisions": 0, "rate": rate}


def test_stall_rule():
    lim = C.Limits()
    since = 1000.0
    healthy = [_tp(since + 60 * i, 5000.0) for i in range(60)]  # one hour
    now = since + 3600
    assert W.stall_reason(healthy, since, now, lim) is None
    # Not armed in the first 30 + 10 min after a start.
    assert W.stall_reason([], since, since + 1799 + 600, lim) is None
    assert "no throughput row" in W.stall_reason([], since, since + 1800 + 601, lim)
    slow = healthy + [_tp(now + 60 * i, 2000.0) for i in range(1, 11)]
    assert "under 0.5" in W.stall_reason(slow, since, now + 600, lim)
    # One sample at or above half the median in the window: not a stall.
    mixed = healthy + [_tp(now + 60 * i, 2000.0 if i != 5 else 2600.0) for i in range(1, 11)]
    assert W.stall_reason(mixed, since, now + 600, lim) is None
    # Rows of an earlier process do not count toward this one's median.
    assert "no throughput row" in W.stall_reason(healthy, now + 1, now + 2500, lim)
    assert W.stall_reason(slow, since, now + 600, replace(lim, stall_frac=0)) is None


def test_starts_and_reset(lane):
    sd, _lim, _ = lane
    sd.p(C.STARTS).write_text("100.0\t-\t0\n200.0\t-\t0\n", encoding="utf-8")
    assert W.read_starts(sd) == [100.0, 200.0]
    W.reset_starts(sd, "fixed")
    assert W.read_starts(sd) == []
    with open(sd.p(C.STARTS), "a", encoding="utf-8") as f:
        f.write("300.0\t-\t0\n")
    assert W.read_starts(sd) == [300.0]


def test_prestart_counts_starts_in_the_window(lane):
    sd, lim, _ = lane
    now = 1_000_000.0
    lines = "".join(f"{now - t:.3f}\t-\t0\n" for t in (8000, 7000, 3000, 2000, 1000))
    sd.p(C.STARTS).write_text(lines, encoding="utf-8")
    assert W.prestart(sd, lim, now=now) == f"stop {C.WD_EXIT_CRASH_LOOP} crash_loop"
    assert C.read_json(sd.root / C.DONE)["reason"] == "watchdog:CRASH_LOOP"
    sd2 = ckpt.SaveDir(sd.root.parent / "lane2")
    sd2.p(C.OPS_DIR).mkdir(parents=True)
    lines = "".join(f"{now - t:.3f}\t-\t0\n" for t in (9000, 8000, 3000, 2000, 1000))
    sd2.p(C.STARTS).write_text(lines, encoding="utf-8")
    assert W.prestart(sd2, lim, now=now) == "start - 0"  # 3 in the window: the 4th may start


def test_prestart_refusals(lane):
    sd, lim, meminfo = lane
    C.write_done(sd.root, "budget", by="learner")
    assert W.prestart(sd, lim) == "done"
    (sd.root / C.DONE).unlink()
    C.atomic_write_text(sd.root / C.LEARNER_PID, json.dumps(C.pid_record()))
    assert W.prestart(sd, lim) == f"stop {C.WD_EXIT_LOCKED} learner_alive"
    (sd.root / C.LEARNER_PID).unlink()
    ckpt.save_full(sd, b"f", decisions=25)
    assert W.prestart(sd, lim) == f"stop {C.WD_EXIT_USAGE} resume_missing"
    assert not (sd.root / C.DONE).exists()  # a refusal, not a stop: no DONE
    assert C.read_json(C.alarm_path(sd.root, "RESUME_MISSING"))["severity"] == C.OPERATOR
    ckpt.save_state(sd, b"s", decisions=30, it=3)
    _set_ram(meminfo, 10)
    assert W.prestart(sd, lim, now=100.0) == "wait ram_low"
    assert W.prestart(sd, lim, now=100.0 + lim.ram_grace_s) == f"stop {C.WD_EXIT_RAM} ram_stop"


def test_prestart_pins_the_resume_state_on_a_restart_only(lane):
    sd, lim, _ = lane
    assert W.prestart(sd, lim) == "start - 0"  # fresh: nothing to pin
    ckpt.save_state(sd, b"s", decisions=30, it=3)
    assert W.prestart(sd, lim) == f"start {sd.state} 0"
    assert [(r.decisions, r.reason) for r in ckpt.pinned_states(sd)] == [(30, "restart")]
    W.prestart(sd, lim)  # the same state again: deduped
    assert len(ckpt.pinned_states(sd)) == 1


def test_ram_rule_hysteresis_and_the_eval_pause(lane):
    sd, lim, meminfo = lane
    assert W.ram_rule(sd, lim, 0.0)[0] == "ok"
    _set_ram(meminfo, 19)
    assert W.ram_rule(sd, lim, 10.0)[0] == "low"
    pause = C.read_json(sd.root / C.EVALS_PAUSE)
    assert pause["by"] == "watchdog"
    _set_ram(meminfo, 22)
    assert W.ram_rule(sd, lim, 20.0)[0] == "ok"
    assert (sd.root / C.EVALS_PAUSE).exists()  # under 24 GB: stays paused
    _set_ram(meminfo, 19)
    assert W.ram_rule(sd, lim, 30.0)[0] == "low"  # the grace clock restarted at 30
    assert W.ram_rule(sd, lim, 30.0 + lim.ram_grace_s - 1)[0] == "low"
    assert W.ram_rule(sd, lim, 30.0 + lim.ram_grace_s)[0] == "stop"
    _set_ram(meminfo, 30)
    assert W.ram_rule(sd, lim, 500.0)[0] == "ok"
    assert not (sd.root / C.EVALS_PAUSE).exists()
    # A pause the driver wrote is the driver's to lift.
    C.atomic_write_text(sd.root / C.EVALS_PAUSE, json.dumps({"by": "driver"}))
    W.ram_rule(sd, lim, 600.0)
    assert (sd.root / C.EVALS_PAUSE).exists()


def test_tick_nan_row_and_done_grace(lane):
    sd, lim, _ = lane
    since = time.time() - 5
    with open(sd.root / C.STATS, "a", encoding="utf-8") as f:
        f.write(json.dumps(C.minimal_stats_row(1, 10, time=since - 10, nonfinite=1)) + "\n")
        f.write(json.dumps(C.minimal_stats_row(2, 20)) + "\n")
    assert W.tick(sd, lim, since, os.getpid()) == "ok"  # the NaN row is an old process's
    with open(sd.root / C.STATS, "a", encoding="utf-8") as f:
        f.write(json.dumps(C.minimal_stats_row(3, 30, losses={"total": math.nan})) + "\n")
    assert W.tick(sd, lim, since, os.getpid()) == "kill_nan 30"
    C.write_done(sd.root, "operator", by="test")
    assert W.tick(sd, lim, since, os.getpid()) == "kill_nan 30"  # inside the DONE grace
    later = time.time() + lim.done_grace_s + 1
    assert W.tick(sd, lim, since, os.getpid(), now=later) == "stop 0 done_grace"


def test_tick_memory_cap(lane):
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc")
    sd, lim, _ = lane
    assert W.lane_mem_bytes(os.getsid(0)) > 0
    lim = replace(lim, rss_max_gb=1e-6)
    out = W.tick(sd, lim, time.time(), os.getsid(0))
    assert out == f"stop {C.WD_EXIT_RSS} rss_stop"


def test_exited_classification(lane):
    sd, lim, _ = lane
    assert W.exited(sd, lim, 3) == "restart"
    assert "crash" in [e["kind"] for e in C.iter_jsonl(C.tail_lines(sd.p(C.EVENTS)))]
    assert W.exited(sd, lim, 0, intent="signal") == "stopped"
    assert W.exited(sd, lim, 143, intent="stall") == "restart"
    C.write_done(sd.root, "budget", by="learner")
    assert W.exited(sd, lim, 0) == "done"
    (sd.root / C.DONE).unlink()
    assert W.exited(sd, lim, C.EXIT_HUMAN) == f"stop {C.WD_EXIT_LEARNER_HUMAN} learner_human"


def test_nan_without_a_second_full_state_is_human(lane):
    sd, lim, _ = lane
    ckpt.save_full(sd, b"f", decisions=25)
    C.atomic_write_text(sd.root / C.NAN_RECORD, json.dumps({"decisions": 30}))
    assert W.exited(sd, lim, C.EXIT_NAN) == f"stop {C.WD_EXIT_NAN_HUMAN} nan_human"
    assert "no second full state" in C.read_json(C.alarm_path(sd.root, "NAN_HUMAN"))["reason"]


def test_nan_rollback_then_resume_next_is_used_until_state_pt_is_newer(lane):
    sd, lim, _ = lane
    ckpt.save_state(sd, b"s", decisions=60, it=6)
    for d in (25, 50):
        ckpt.save_full(sd, json.dumps({"decisions": d, "it": d // 10}), decisions=d)
    assert W.handle_nan(sd, lim, 60) == "restart"
    assert W.seed_offset(sd) == lim.nan_seed_step
    out = W.prestart(sd, lim)
    pinned = sd.pinned_dir / ckpt.ckpt_name("pinned", 25, "nan_rollback")
    assert out == f"start {pinned} {lim.nan_seed_step}"
    # Crash before the first save: the next start uses the rollback target again.
    assert W.prestart(sd, lim) == f"start {pinned} {lim.nan_seed_step}"
    ckpt.save_state(sd, b"s2", decisions=30, it=3)
    assert W.prestart(sd, lim) == f"start {sd.state} {lim.nan_seed_step}"
    assert not sd.p(C.RESUME_NEXT).exists()


# ── the fake learner speaks the contract ────────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="the fake learner uses fcntl")
def test_fake_learner_rows_conform_to_the_contract(tmp_path):
    save = tmp_path / "lane"
    save.mkdir()
    (save / "fake_plan.json").write_text(json.dumps({"default": {"budget": 100}}), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(FAKE), C.FLAG_SAVE_DIR, str(save), C.FLAG_SEED_OFFSET, "7"],
        cwd=PYCLI,
        env={**os.environ, "PYTHONPATH": str(PYCLI / "python")},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    stats = list(C.iter_jsonl(C.tail_lines(save / C.STATS)))
    assert len(stats) == 10
    for row in stats:
        assert C.check_stats_row(row) == []
        assert row["sampler_seed_offset"] == 7
    for row in C.iter_jsonl(C.tail_lines(save / C.THROUGHPUT)):
        assert C.check_throughput_row(row) == []
    assert C.read_json(save / C.DONE)["reason"] == "budget"
    assert ckpt.state_decisions(ckpt.SaveDir(save)) == 100
    # A second start without --resume refuses to overwrite the run.
    (save / C.DONE).unlink()
    r = subprocess.run(
        [sys.executable, str(FAKE), C.FLAG_SAVE_DIR, str(save), C.FLAG_SEED_OFFSET, "0"],
        cwd=PYCLI,
        env={**os.environ, "PYTHONPATH": str(PYCLI / "python")},
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == C.EXIT_USAGE
