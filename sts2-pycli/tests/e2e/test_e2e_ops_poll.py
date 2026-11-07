"""The laptop poller (scripts/e2e_poll.sh) against a local fake host: fake ssh
and tailscale commands, a temp campaign dir."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ppo.e2e import ops_contract as C

PYCLI = Path(__file__).resolve().parents[2]
SCRIPT = PYCLI / "scripts" / "e2e_poll.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None or shutil.which("timeout") is None,
    reason="the poller is a bash script",
)

FAKE_SSH = """#!/usr/bin/env bash
# ssh -o K=V ... HOST CMD...: run CMD here, with stdin.
while [[ $# -gt 0 && $1 == -o ]]; do shift 2; done
shift
[[ -n ${FAKE_SSH_FAIL:-} ]] && exit 255
[[ -n ${FAKE_SSH_BANNER:-} ]] && echo "Welcome to the box"
exec "$@"
"""
FAKE_TS = """#!/usr/bin/env bash
[[ -n ${FAKE_TS_FAIL:-} ]] && { echo "no reply"; exit 1; }
echo "pong from a4000 (100.64.0.7) ${FAKE_TS_VIA:-via 192.168.1.20:41641} in 3ms"
"""


@pytest.fixture
def host(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("ssh", FAKE_SSH), ("tailscale", FAKE_TS)):
        f = bin_dir / name
        f.write_text(text, encoding="utf-8")
        f.chmod(0o755)
    camp = tmp_path / "campaign"
    (camp / "lanes" / "main").mkdir(parents=True)
    env = {
        **os.environ,
        "E2E_POLL_SSH": str(bin_dir / "ssh"),
        "E2E_POLL_TAILSCALE": str(bin_dir / "tailscale"),
        "E2E_POLL_PY": sys.executable,
        "E2E_POLL_CAMPAIGN": str(camp),
        "E2E_POLL_LOG": str(tmp_path / "poll.log"),
    }
    return camp, env, tmp_path


def _status(camp: Path, age_s: float = 60.0, summary: str = "OK | main 1.234B 41.2k/s") -> None:
    rec = {"time": time.time() - age_s, "summary": summary, "lanes": {}, "alarms": []}
    C.atomic_write_text(camp / C.STATUS, json.dumps(rec))


def _poll(env: dict, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), "a4000"],
        env={**env, **extra},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def test_ok(host):
    camp, env, tmp = host
    _status(camp)
    r = _poll(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("e2e OK | OK | main 1.234B")
    assert "status 1 min old" in r.stdout
    assert "rc=0 e2e OK" in (tmp / "poll.log").read_text(encoding="utf-8")


def test_operator_and_human_alarms(host):
    camp, env, _ = host
    _status(camp)
    C.write_alarm(camp / "lanes" / "main", "CAPS", writer="driver", reason="x", lane="main")
    r = _poll(env)
    assert r.returncode == 1
    assert "OPERATOR: CAPS(main)" in r.stdout
    C.write_alarm(camp / "lanes" / "main", "NAN_HUMAN", writer="watchdog", reason="y", lane="main")
    C.write_alarm(camp, "DRIVER_ERROR", writer="driver", reason="z")
    r = _poll(env)
    assert r.returncode == 2
    assert "HUMAN: NAN_HUMAN(main)" in r.stdout
    assert "DRIVER_ERROR(campaign)" in r.stdout


def test_unreadable_alarm_counts_as_human(host):
    camp, env, _ = host
    _status(camp)
    (camp / "ALARM_SOMETHING").write_text("not json", encoding="utf-8")
    assert _poll(env).returncode == 2


def test_stale_or_missing_status(host):
    camp, env, _ = host
    r = _poll(env)
    assert r.returncode == 3
    assert "no status.json" in r.stdout
    _status(camp, age_s=3600)
    r = _poll(env)
    assert r.returncode == 3
    assert "status 60 min old" in r.stdout


def test_unreachable_and_derp(host):
    camp, env, _ = host
    _status(camp)
    r = _poll(env, FAKE_SSH_FAIL="1", FAKE_TS_FAIL="1")
    assert r.returncode == 4
    assert "UNREACHABLE" in r.stdout and "tailscale ping failed" in r.stdout
    r = _poll(env, FAKE_TS_VIA="via DERP(fra)", FAKE_SSH_BANNER="1")
    assert r.returncode == 0, r.stdout
    assert "via DERP (slow, not down)" in r.stdout
