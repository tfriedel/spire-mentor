"""The checkpoint store of an e2e lane (ppo.e2e.ckpt): atomic writes,
rotation, pinning and the manifest, rollbacks, pruning and the disk rule."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C

PYCLI = Path(__file__).resolve().parents[2]


def _sd(tmp_path: Path) -> ckpt.SaveDir:
    return ckpt.SaveDir(tmp_path / "lane")


def test_atomic_write_keeps_the_old_file_when_the_writer_fails(tmp_path):
    path = tmp_path / "state.pt"
    ckpt.atomic_write(path, b"old")

    def boom(p: Path) -> None:
        p.write_bytes(b"partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        ckpt.atomic_write(path, boom)
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["state.pt"]  # no temp file left
    ckpt.atomic_write(path, lambda p: p.write_bytes(b"new"))
    assert path.read_bytes() == b"new"


def test_names_round_trip_and_sort_by_decisions():
    name = ckpt.ckpt_name("pinned", 250_000_000, "k2")
    assert name == "pinned_d0000250000000_k2.pt"
    c = ckpt.parse_name(name)
    assert (c.kind, c.decisions, c.reason) == ("pinned", 250_000_000, "k2")
    assert ckpt.parse_name(ckpt.ckpt_name("pinned", 5, "restart", bump=2)).reason == "restart"
    assert ckpt.parse_name("full_d0000000000005.pt").kind == "full"
    assert ckpt.parse_name("state.pt") is None
    assert ckpt.parse_name("pinned_d0000000000005.pt") is None  # a pin needs a reason
    with pytest.raises(ValueError):
        ckpt.ckpt_name("full", 5, "k2")
    with pytest.raises(ValueError):
        ckpt.ckpt_name("pinned", 5, "Bad-Reason")
    names = [ckpt.ckpt_name("full", d) for d in (9, 10, 100)]
    assert sorted(names) == names


def test_state_and_sidecar(tmp_path):
    sd = _sd(tmp_path)
    ckpt.save_state(sd, b"s", decisions=1234, it=7)
    assert sd.state.read_bytes() == b"s"
    assert ckpt.state_decisions(sd) == 1234
    assert C.read_json(sd.state_meta)["it"] == 7


def test_rolling_full_states_keep_the_newest_three(tmp_path):
    sd = _sd(tmp_path)
    for d in range(1, 6):
        ckpt.save_full(sd, f"f{d}", decisions=d * 25)
    assert [c.decisions for c in sd.listing("full")] == [75, 100, 125]


def test_pin_records_sha256_dedupes_and_bumps(tmp_path):
    sd = _sd(tmp_path)
    a = ckpt.pin(sd, decisions=100, reason="milestone", data=b"A")
    assert a.name == ckpt.ckpt_name("pinned", 100, "milestone")
    rows = ckpt.read_manifest(sd)
    assert rows[0].sha256 == ckpt.sha256_file(a)
    assert rows[0].bytes == 1
    # The same bytes at the same decisions: the existing pin, no new row.
    src = tmp_path / "copy.pt"
    src.write_bytes(b"A")
    assert ckpt.pin(sd, decisions=100, reason="restart", src=src) == a
    assert len(ckpt.read_manifest(sd)) == 1
    # Other bytes at the same decisions and reason: a bumped name.
    b = ckpt.pin(sd, decisions=100, reason="milestone", data=b"B")
    assert b.name == ckpt.ckpt_name("pinned", 100, "milestone", bump=1)
    assert [r.name for r in ckpt.pinned_states(sd)] == [a.name, b.name]
    assert ckpt.last_pinned(sd).name == b.name
    assert ckpt.verify_pinned(sd) == []
    with pytest.raises(ValueError):
        ckpt.pin(sd, decisions=1, reason="manual")


def test_verify_pinned_finds_tampering_and_strays(tmp_path):
    sd = _sd(tmp_path)
    a = ckpt.pin(sd, decisions=100, reason="milestone", data=b"A")
    a.write_bytes(b"tampered")
    (sd.pinned_dir / ckpt.ckpt_name("pinned", 200, "manual")).write_bytes(b"stray")
    problems = ckpt.verify_pinned(sd)
    assert any("sha256 differs" in p for p in problems)
    assert any("not in the manifest" in p for p in problems)
    a.unlink()
    assert any("missing" in p for p in ckpt.verify_pinned(sd))


def test_full_states_merge_rolling_and_pinned(tmp_path):
    sd = _sd(tmp_path)
    for d in (50, 75, 100):
        ckpt.save_full(sd, f"f{d}", decisions=d)
    ckpt.pin(sd, decisions=75, reason="k2", data=b"pinned75")
    ckpt.pin(sd, decisions=10, reason="milestone", data=b"pinned10")
    states = ckpt.full_states(sd)
    assert [(s.decisions, s.kind) for s in states] == [
        (10, "pinned"),
        (50, "full"),
        (75, "pinned"),
        (100, "full"),
    ]
    assert ckpt.second_newest_full_state(sd).decisions == 75
    assert ckpt.second_newest_full_state(_sd(tmp_path / "empty")) is None


def test_rollback_quarantines_newer_states_and_abandons_newer_pins(tmp_path):
    sd = _sd(tmp_path)
    ckpt.save_state(sd, b"s", decisions=130, it=13)
    for d in (50, 75, 100, 125):
        ckpt.save_full(sd, f"f{d}", decisions=d)
    ckpt.save_slim(sd, b"slim", decisions=100)
    k2 = ckpt.pin(sd, decisions=75, reason="k2", data=b"k2")
    late = ckpt.pin(sd, decisions=110, reason="request", data=b"late")
    # A deliberate rollback needs a pinned state.
    with pytest.raises(ValueError):
        ckpt.rollback(sd, sd.full_dir / ckpt.ckpt_name("full", 100), reason="k2_relaunch")
    rec = ckpt.rollback(sd, k2.name, reason="k2_relaunch")
    assert rec["path"] == str(k2)
    assert rec["decisions"] == 75
    assert rec["abandoned"] == [late.name]
    assert not sd.state.exists()
    assert [c.decisions for c in sd.listing("full")] == [75]
    assert sd.listing("slim") == []
    moved = {Path(p).name for p in rec["quarantined"]}
    assert {"state.pt", "state.json", ckpt.ckpt_name("full", 100)} <= moved
    assert late.exists()  # abandoned pins stay on disk, out of pinned_states()
    assert [r.name for r in ckpt.pinned_states(sd)] == [k2.name]
    nxt = C.read_json(sd.p(C.RESUME_NEXT))
    assert nxt["path"] == str(k2) and isinstance(nxt["unix"], float)
    assert ckpt.verify_pinned(sd) == []


def test_rollback_refuses_while_the_learner_runs(tmp_path):
    sd = _sd(tmp_path)
    k2 = ckpt.pin(sd, decisions=75, reason="k2", data=b"k2")
    C.atomic_write_text(sd.root / C.LEARNER_PID, json.dumps(C.pid_record()))  # this process
    with pytest.raises(RuntimeError, match="still runs"):
        ckpt.rollback(sd, k2, reason="k2_relaunch")


def test_prune_order_spares_pinned_state_and_the_newest_two(tmp_path):
    sd = _sd(tmp_path)
    ckpt.save_state(sd, b"s", decisions=600_000_000, it=1)
    for d in (100, 200, 300, 400, 500):
        ckpt.save_slim(sd, b"slim", decisions=d * 1_000_000)
    for d in (450, 475, 500):
        ckpt.save_full(sd, b"full", decisions=d * 1_000_000)
    pinned = ckpt.pin(sd, decisions=250_000_000, reason="milestone", data=b"p")
    q = sd.quarantine_dir / "20261101T000000Z_nan"
    q.mkdir(parents=True)
    (q / "state.pt").write_bytes(b"q")
    order = [p.name for p in ckpt.prune_candidates(sd)]
    assert order[0] == "state.pt"  # the quarantined copy goes first
    slim_off = [ckpt.ckpt_name("slim", d * 1_000_000) for d in (100, 200, 300, 400)]
    assert order[1:5] == slim_off
    assert order[5] == ckpt.ckpt_name("slim", 500_000_000)  # on the 500M grid: last slim
    assert order[6:] == [ckpt.ckpt_name("full", 450_000_000)]
    deleted = ckpt.prune(sd, target_free_gb=1e12)  # cannot be reached: delete all candidates
    assert len(deleted) == len(order)
    assert sd.state.exists() and pinned.exists()
    assert [c.decisions for c in sd.listing("full")] == [475_000_000, 500_000_000]
    assert not sd.quarantine_dir.exists() or not any(sd.quarantine_dir.iterdir())


def test_disk_check_under_a_quota(tmp_path):
    sd = _sd(tmp_path)
    for d in (1, 2, 3):
        ckpt.save_full(sd, b"x" * 100_000, decisions=d)
    ckpt.pin(sd, decisions=1, reason="milestone", data=b"y" * 100_000)
    # 400 KB used of a 1 MB quota: 0.6 MB free.
    v = ckpt.disk_check(sd, prune_below_gb=0.0005, stop_below_gb=0.0001, quota_gb=0.001)
    assert v.action == "ok" and v.deleted == []
    # Prune line above the free space: delete the oldest rolling state, run on.
    v = ckpt.disk_check(sd, prune_below_gb=0.00065, stop_below_gb=0.0001, quota_gb=0.001)
    assert v.action == "pruned"
    assert [Path(p).name for p in v.deleted] == [ckpt.ckpt_name("full", 1)]
    # A stop line above what pruning can free: stop; the pin stays.
    v = ckpt.disk_check(sd, prune_below_gb=0.0009, stop_below_gb=0.0009, quota_gb=0.001)
    assert v.action == "stop"
    assert ckpt.verify_pinned(sd) == []
    assert len(sd.listing("full")) == ckpt.KEEP_FULL_ON_PRUNE


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppo.e2e.ckpt", *args],
        cwd=PYCLI,
        env={**os.environ, "PYTHONPATH": str(PYCLI / "python"), **(env or {})},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_cli_pin_verify_rollback_and_disk_check(tmp_path):
    sd = _sd(tmp_path)
    ckpt.save_state(sd, b"s", decisions=300, it=3)
    r = _cli("pin", str(sd.root), "--src", str(sd.state), "--reason", "hot_swap")
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.strip()).name == ckpt.ckpt_name("pinned", 300, "hot_swap")
    assert _cli("verify", str(sd.root)).returncode == 0
    r = _cli("last-pinned", str(sd.root))
    assert json.loads(r.stdout)["decisions"] == 300
    r = _cli("rollback", str(sd.root), "--to", ckpt.ckpt_name("pinned", 300, "hot_swap"),
             "--reason", "e1_revert")  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert C.read_json(sd.p(C.RESUME_NEXT))["reason"] == "e1_revert"
    env = {"E2E_DISK_QUOTA_GB": "0.000001", "E2E_WD_DISK_PRUNE_GB": "1", "E2E_WD_DISK_STOP_GB": "1"}
    r = _cli("disk-check", str(sd.root), env=env)
    assert r.returncode == C.WD_EXIT_DISK
    assert json.loads(r.stdout)["action"] == "stop"
