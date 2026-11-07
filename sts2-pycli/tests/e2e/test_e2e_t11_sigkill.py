"""T11 with a real kill (plan section 7, GH #1097 G0 audit).

`test_learner.py`'s T11 never kills a process: run A saves state.pt at iteration
k in-process, and a fresh `Learner` in the same process resumes from a copy.
Here the learner runs in its own OS process (`VICTIM`: the same scripted
learner on `FakeSource`, state.pt every iteration) and is SIGKILLed, process
group and all, once its stats file shows >= `target` iterations, wherever it
is (collecting, serving, updating, saving). Then a learner resumes in the
victim's own save dir (the kill leaves a stale pid file and lock) from
whatever state.pt the kill left, and its next 3 iterations must equal those of
the in-process run A of `_t11` that drops its open segments at the same k: the
stable stats rows, the net 3 iterations on, the clocks.

Still `FakeSource` and the local updater: no engine workers, no update ranks.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import pytest
import sts2_client
import test_learner as TL
import torch

HERE = Path(__file__).resolve().parent
PYCLI = HERE.parents[1]

# argv: <save dir> <shm log> <max iters> <here> <state every, in iterations>. Every SharedMemory block it creates is
# logged to <shm log>, so the parent can unlink exactly those after the kill.
VICTIM = """
import multiprocessing.shared_memory as shm_mod
import sys
from pathlib import Path

save, shm_log, max_iters, here = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
every = int(sys.argv[5])
_log = open(shm_log, "a", encoding="utf-8")
_orig = shm_mod.SharedMemory.__init__


def _init(self, name=None, create=False, size=0, *a, **kw):
    _orig(self, name, create, size, *a, **kw)
    if create:
        _log.write(self.name + "\\n")
        _log.flush()


shm_mod.SharedMemory.__init__ = _init

import sts2_client

sts2_client.GAME_VERSION = "v0.111.0"
sys.path.insert(0, here)
import test_learner as TL

lr = TL.make_learner(TL.make_cfg(save, max_iters=max_iters, state_every=every * TL.ITER))
lr.train()
"""


@pytest.fixture(autouse=True)
def _v111(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sts2_client, "GAME_VERSION", TL.V111)


def _kill_at(victim: Path, target: int, shm_log: Path, every: int = 1) -> None:
    env = {
        **os.environ,
        "STS2_GAME_VERSION": TL.V111,
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
    }
    log_path = victim.parent / "victim.log"
    argv = [sys.executable, "-c", VICTIM, str(victim), str(shm_log), "400", str(HERE), str(every)]
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(
            argv,
            cwd=PYCLI,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    stats = victim / "stats.jsonl"
    try:
        t0 = time.time()
        while time.time() - t0 < 300 and p.poll() is None:
            if stats.exists() and len(stats.read_text(encoding="utf-8").splitlines()) >= target:
                break
            time.sleep(0.01)
        assert p.poll() is None, log_path.read_text(encoding="utf-8")[-3000:]
        os.killpg(p.pid, signal.SIGKILL)
        assert p.wait(timeout=60) == -signal.SIGKILL
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        if shm_log.exists():  # the victim's leaked row blocks, and only those
            for name in set(shm_log.read_text(encoding="utf-8").split()):
                try:
                    shared_memory.SharedMemory(name=name, track=False).unlink()
                except FileNotFoundError:
                    pass


def _its(path: Path) -> list[int]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(int(json.loads(line)["it"]))
        except ValueError:  # a torn line
            continue
    return out


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and signals")
@pytest.mark.parametrize(
    ("target", "every"),
    # every > 1: state.pt every few iterations, as at launch (5M decisions = ~38
    # iterations), so the kill leaves rows past the state that the resume replays.
    [(7, 1), (19, 1), (151, 1), (23, 5)],
)
def test_sigkill_then_resume_reproduces_the_next_three_iterations(
    tmp_path: Path, target: int, every: int
) -> None:
    victim = tmp_path / "victim"
    _kill_at(victim, target, tmp_path / "shm.log", every)
    assert not (victim / "DONE").exists()
    st = torch.load(victim / "state.pt", map_location="cpu", weights_only=True)
    k = int(st["algo"]["clocks"]["it"])
    assert k >= target - 1 - every, k
    stats = victim / "stats.jsonl"
    killed = _its(stats)  # what the kill left, a torn last line aside
    # Run A (in-process, as `_t11`): drop the open segments at k, snapshot 3 iterations on.
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    TL.Uninterrupted.kills, TL.Uninterrupted.snap_dir = (k,), snaps
    a = TL.make_learner(TL.make_cfg(tmp_path / "A", max_iters=k + 4), cls=TL.Uninterrupted)
    a.train()
    rows_a = {r["it"]: r for r in a.stats_rows}
    # B: resume in the killed lane's own save dir (stale lock and pid file).
    b = TL.make_learner(TL.make_cfg(victim, max_iters=k + 3, resume=str(victim / "state.pt")))
    n_held = b.held_lost
    assert b.n_dispatched == k and n_held > 0
    b.train()
    assert [r["it"] for r in b.stats_rows] == [k + 1, k + 2, k + 3]
    ua, ub = rows_a[k + 1]["update"], b.stats_rows[0]["update"]
    assert ua.pop("held_dropped") == n_held + ub.pop("held_dropped")
    for r in b.stats_rows:
        assert TL._stable(r) == TL._stable(rows_a[r["it"]]), (k, r["it"])
    # The rows past k moved aside, so `it` is unique and increasing.
    assert _its(stats) == list(range(1, k + 4))
    orphan = victim / "stats.orphan.jsonl"
    # (Usually some with every > 1; a kill right after a state.pt leaves none.
    # test_learner's older-state resume covers the parking deterministically.)
    assert (_its(orphan) if orphan.exists() else []) == [i for i in killed if i > k]
    after = torch.load(snaps / f"after_{k}.pt", weights_only=True)
    got = b.net_inf.state_dict()
    assert all(torch.equal(got[n], v) for n, v in after["net"].items()), k
    assert b.algo.to_dict()["clocks"] == after["algo"]["clocks"]
    # And A's state at k is the killed process's state at k (the same trajectory).
    sa = torch.load(snaps / f"state_{k}.pt", map_location="cpu", weights_only=True)
    assert sa["algo"]["clocks"] == st["algo"]["clocks"]
