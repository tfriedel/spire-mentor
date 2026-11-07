"""ppo.e2e.evaluate: the runner's refusals, its pieces on fakes, and real runs on SMOKE seeds.

The real runs go through ``python -m ppo.e2e.evaluate`` (the command re-executes
itself in a clean environment, so it cannot run in-process) with the stand-in env
(``--env raw``) and stand-in policies, on SMOKE seeds, under tmp_path. They skip
when the engine wheel does not import. No test reads ``~/runs/e2e/evals/``.
"""

import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ppo.e2e import evaluate as E
from ppo.e2e import ops_contract as OC
from ppo.e2e import panels as P
from ppo.e2e import score_test as T
from ppo.e2e import stamps as ST

PYCLI = Path(__file__).resolve().parents[2]
PY_DIR = PYCLI / "python"
SMOKE0 = P.SMOKE.start

try:
    import sts2_rs  # noqa: F401

    HAVE_ENGINE = True
except ImportError:  # pragma: no cover - depends on the venv
    HAVE_ENGINE = False

needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="sts2_rs wheel not importable")


def _export_wheel_mismatch() -> str | None:
    """Why the runner's wheel guard must refuse this tree's wheel, or None.

    `just test-public-shape` runs the export with the private tree's wheel, and
    the export rewrites Rust doc comments, so the guard's src_hash check fails
    there and nowhere else (the public CI builds its wheel from the export).
    `release/` is never exported, so its absence marks an export; in the
    private tree this is always None and a stale wheel still fails loudly."""
    if not HAVE_ENGINE or (PYCLI.parent / "release").is_dir():
        return None
    from ppo.wheel_guard import default_rs_root, wheel_staleness

    return wheel_staleness(default_rs_root())


_MISMATCH = _export_wheel_mismatch()
# Real runs pass the runner's wheel guard: skip them only in an export whose
# wheel was built from other source.
needs_tree_wheel = pytest.mark.skipif(
    not HAVE_ENGINE or _MISMATCH is not None,
    reason=f"the wheel was not built from this export's Rust source ({_MISMATCH})",
)
# Off TEST, a test may run from a dirty tree, an unpinned wheel and a plain build.
LOOSE = ["--allow-unpinned", "--allow-dirty", "--allow-plain-wheel"]

HELPER = '''
import os
import time
from pathlib import Path

from ppo.e2e.evaluate import RawEnv


def flaky(seed):
    """Fails once for FLAKY_SEED (a marker file remembers), then plays."""
    marker = Path(os.environ["FLAKY_DIR"]) / str(seed)
    if seed == int(os.environ["FLAKY_SEED"]) and not marker.exists():
        marker.touch()
        raise RuntimeError("flaky on purpose")
    return RawEnv(seed)


def hang(seed):
    if seed == int(os.environ["FLAKY_SEED"]):
        time.sleep(3600)
    return RawEnv(seed)
'''


def evaluate(*args, cwd=PYCLI, timeout=600):
    """Run the CLI as the docs say: `python -m ppo.e2e.evaluate`, PYTHONPATH=python."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E2E_", "STS2_"))}
    env["PYTHONPATH"] = str(PY_DIR)
    return subprocess.run(
        [sys.executable, "-m", "ppo.e2e.evaluate", *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def arm_files(evals, panel, arm):
    d = evals / panel / arm
    return {
        p.relative_to(d).as_posix(): p.read_bytes()
        for sub in ("shards", "retry")
        for p in sorted((d / sub).glob("*.jsonl"))
    }


# The stamps a campaign process computes, with the pre-flight's vocab and schema
# sources: what every eval row must carry (plan section 3.1, F8).
STAMPS_OF_THIS_PROCESS = (
    "import json\n"
    "from ppo.e2e import contracts as C, stamps as ST, vocab as V\n"
    "print(json.dumps(ST.stamps(vocab_sha256=V.load().sha256, schema_sha256=C.schema_sha256())))\n"
)


@functools.cache
def cli_stamps() -> dict:
    """`stamps.stamps` of a fresh process on the sys.path the CLI runs below get
    (`--allow-unpinned`, no `--pin`: PYTHONPATH is python/ alone, so the wheel is the
    venv's, which need not be this process's)."""
    out = subprocess.run(
        [sys.executable, "-c", STAMPS_OF_THIS_PROCESS],
        cwd=PYCLI,
        env=E._clean_env(None, []),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def assert_stamped(row, want):
    """Every `stamps.STAMP_KEYS` value of `row` is `want`'s, and `want` is a campaign
    process's: v0.111.0, with a vocab and a schema sha256."""
    assert want["game_version"] == E.GAME_VERSION
    assert len(want["vocab_sha256"]) == 64 and len(want["schema_sha256"]) == 64
    got = {k: row.get(k, "<absent>") for k in ST.STAMP_KEYS}
    assert got == {k: want[k] for k in ST.STAMP_KEYS}


# ── refusals (no engine, in-process: they come before the re-exec) ────────


@pytest.mark.parametrize(
    ("argv", "msg"),
    [
        (["SMOKE_SEALED", "4", str(SMOKE0)], "is sealed: pass --sealed"),
        (["SMOKE", "4", str(SMOKE0), "--sealed"], "--sealed is for the sealed panels"),
        (["SMOKE_SEALED", "4", str(SMOKE0), "--sealed", "--allow-dirty"], "takes no --allow-dirty"),
        (["SMOKE_SEALED", "4", str(SMOKE0), "--sealed"], "pass --pin"),
        (["SMOKE", "4", str(SMOKE0 - 1)], "leave the SMOKE panel"),
        (["SMOKE", "4", str(SMOKE0), "--set-env", "NOEQUALS"], "--set-env needs K=V"),
        (["SMOKE", "4", str(SMOKE0), "--procs", "0"], "--procs must be positive"),
    ],
)
def test_refusals(argv, msg, capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("E2E_PIN", raising=False)
    panel, n, start, *rest = argv
    full = ["run", "A", panel, n, start, "--policy", "first", "--evals", str(tmp_path)]
    if "--procs" not in rest:
        full += ["--procs", "1"]
    assert E.main([*full, *rest]) == 2
    assert msg in capsys.readouterr().err
    assert not (tmp_path / panel).exists()


def test_report_refuses_a_sealed_panel(capsys, tmp_path):
    assert E.main(["report", "A", "SMOKE_SEALED", "--evals", str(tmp_path)]) == 2
    assert "never reads a sealed panel" in capsys.readouterr().err


def test_reexec_only_from_dash_m(monkeypatch):
    monkeypatch.setattr(sys, "orig_argv", ["python", "-c", "pass"])
    with pytest.raises(E.EvalError, match=r"python -m ppo\.e2e\.evaluate"):
        E.reexec_clean("ppo.e2e.evaluate", [], None, [])


def test_clean_env_drops_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("STS2_RUNAI_HONEST", "0")
    monkeypatch.setenv("STS2_MCTS_SIMS", "7")
    env = E._clean_env(tmp_path, ["STS2_RUNAI_X=1"])
    assert "STS2_RUNAI_HONEST" not in env and "STS2_MCTS_SIMS" not in env
    assert env["STS2_RUNAI_X"] == "1"
    assert env["STS2_GAME_VERSION"] == E.GAME_VERSION
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), str(PY_DIR)]
    assert env[E.CLEAN_MARKER] == "1"
    assert {k: env[k] for k in OC.WORKER_THREAD_ENV} == OC.WORKER_THREAD_ENV


PROBE = '''
import os


class Probe:
    """A stand-in handler that loads torch, as a net policy does, and sets its own threads."""

    def __init__(self, keys):
        import torch

        self.torch, self.keys = torch, keys
        torch.set_num_threads(2)

    def play(self, task):
        return ({k: os.environ.get(k) for k in self.keys}, self.torch.get_num_threads())
'''


def test_pool_workers_run_under_the_thread_caps(monkeypatch, tmp_path):
    """Whoever runs the pool (``run`` under its clean env, or devboss), a worker starts
    with ``ops_contract.WORKER_THREAD_ENV`` and runs torch on one thread (B3); the
    caller's environment is as it was."""
    (tmp_path / "e2e_eval_probe.py").write_text(PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))  # a spawned worker gets this sys.path
    for k in OC.WORKER_THREAD_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    before = dict(os.environ)
    rows = []
    cfg = {"handler": "e2e_eval_probe:Probe", "handler_args": {"keys": list(OC.WORKER_THREAD_ENV)}}
    pool = E.Pool(cfg, 2, 300.0, lambda task, row: rows.append(row), lambda task, why: why)
    pool.run([0, 1])
    assert dict(os.environ) == before
    assert rows == [(OC.WORKER_THREAD_ENV, 1)] * 2


def test_campaign_build_is_required(monkeypatch):
    import sts2_client

    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.107.1")
    with pytest.raises(E.EvalError, match=r"needs v0\.111\.0"):
        E.require_campaign_build()
    monkeypatch.setenv(sts2_client.GAME_VERSION_ENV, "v0.107.1")
    with pytest.raises(E.EvalError, match=r"the campaign runs v0\.111\.0"):
        E.select_campaign_build()


# ── pieces ────────────────────────────────────────────────────────────────


def test_manifest_is_sha256sum_shaped(tmp_path):
    arm = tmp_path / "A"
    for sub in ("shards", "retry"):
        (arm / sub).mkdir(parents=True)
    shard, retry = b'{"seed": 0}\n{"seed": 1}\n', b'{"seed": 1}\n'
    (arm / "shards" / "s0_2.jsonl").write_bytes(shard)
    (arm / "retry" / "s1.jsonl").write_bytes(retry)
    digest = E.write_manifest(arm, tmp_path / "A.sha256")
    lines = (tmp_path / "A.sha256").read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"{hashlib.sha256(shard).hexdigest()}  shards/s0_2.jsonl",
        f"{hashlib.sha256(retry).hexdigest()}  retry/s1.jsonl",
    ]
    assert digest == hashlib.sha256((tmp_path / "A.sha256").read_bytes()).hexdigest()
    assert E.shard_complete(arm / "shards" / "s0_2.jsonl", 0, 2)
    assert not E.shard_complete(arm / "shards" / "s0_2.jsonl", 0, 3)
    assert E.shard_ranges(100, 600) == [(100, 350), (350, 600), (600, 700)]


class FakeRun:
    """Just enough of a RunManager for BossTracker: rooms, phase, HP, potions, map."""

    def __init__(self):
        self.act_index = 1
        self.room = {"map_coord": (0, 0), "room_type": "monster"}
        self.ph = "map_choice"
        self.over = False
        self.player_won = False
        self.current_hp, self.max_hp = 50, 80
        self.boss = {"col": 3, "row": 16, "id": "THE_KIN_BOSS"}

    def is_over(self):
        return self.over

    def current_room(self):
        return dict(self.room)

    def phase(self):
        return self.ph

    def potions(self):
        return ["FIRE_POTION", None, None]

    def full_map(self):
        return {"boss": self.boss, "second_boss": None}


@needs_engine
def test_a_player_logs_every_run_it_plays(tmp_path, monkeypatch):
    """Review finding ENVR-1: `Player` with a `log_dir` (`evaluate` passes
    `<arm>/actions`) keeps every eval run's action log, which replays to the run's end;
    a terminal without a log (the raw stand-in) is skipped."""
    import sts2_client
    from ppo.e2e import actionlog as AL

    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.111.0")
    E._process_stamps.cache_clear()  # another test of this process may have filled it
    logs = tmp_path / "actions"
    pl = E.Player(E.DEFAULT_ENV, "first", None, "cpu", log_dir=str(logs))
    row = pl.play(SMOKE0)
    assert row["outcome"] != "error", row
    assert_stamped(row, _stamps_here())
    (path,) = logs.glob("actions-*.e2el")
    (rec,) = AL.read_records(path)
    assert rec.header["reason"] == "eval" and rec.header["seed"] == SMOKE0
    assert rec.header["extra"] == {"policy": "first"}
    term = AL.replay(rec)
    assert ("won" if term.won else term.outcome) == row["outcome"]
    assert len(rec.idx) == row["n_decisions"]
    raw = E.Player("raw", "first", None, "cpu", log_dir=str(tmp_path / "raw"))
    raw_row = raw.play(SMOKE0)
    assert raw_row["outcome"] != "error"
    assert_stamped(raw_row, _stamps_here())
    assert not list((tmp_path / "raw").glob("*.e2el"))


def _stamps_here() -> dict:
    """This process's stamps (`stamps.stamps`, the pre-flight's vocab and schema)."""
    from ppo.e2e import contracts as C
    from ppo.e2e import vocab as V

    return ST.stamps(vocab_sha256=V.load().sha256, schema_sha256=C.schema_sha256())


class _ErrorEnv:
    """An env whose run ends at once on an error terminal (`outcome == "error"`)."""

    def __init__(self, seed):
        self.rm = None

    def reset(self):
        return SimpleNamespace(outcome="error", won=False, error="the env booked an error")


def _no_env(seed):
    raise RuntimeError("no env on purpose")


@needs_engine
def test_every_eval_row_carries_the_process_stamps(monkeypatch):
    """Plan section 3.1 (F8): an eval row carries `stamps.STAMP_KEYS`, an error row too:
    `error_row` (the rows the pool books for a hung or dead worker), a run that raises
    (`Player.play`'s except path) and an env that ends on an error terminal. A played
    row keeps its `wheel` and `py_src` (the values `wheel_guard` gives)."""
    import sts2_client
    from ppo.wheel_guard import python_source_digest, wheel_identity

    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.111.0")
    E._process_stamps.cache_clear()  # another test of this process may have filled it
    want = _stamps_here()
    assert_stamped(E.error_row(SMOKE0, "first", "boom"), want)
    pl = E.Player("raw", "first", None, "cpu")
    played = pl.play(SMOKE0)
    assert played["outcome"] in ("won", "lost", "capped"), played
    assert_stamped(played, want)
    assert played["wheel"] == wheel_identity() and played["py_src"] == python_source_digest()
    pl.make_env = _no_env
    raised = pl.play(SMOKE0)
    assert raised["outcome"] == "error" and "no env on purpose" in raised["error"]
    assert_stamped(raised, want)
    pl.make_env = _ErrorEnv
    booked = pl.play(SMOKE0)
    assert booked["outcome"] == "error" and booked["error"] == "the env booked an error"
    assert_stamped(booked, want)
    # Each row owns its stamps: editing one row's leaves the next row's alone.
    booked["wheel"]["src_hash"] = "edited"
    assert E.error_row(SMOKE0, "first", "x")["wheel"] == want["wheel"]


def test_boss_tracker():
    rm, t = FakeRun(), E.BossTracker()
    t.observe(rm)
    rm.room, rm.ph = {"map_coord": (3, 16), "room_type": "boss"}, "combat"
    t.observe(rm)
    rm.current_hp = 20
    t.observe(rm)  # same fight: entry stays at 50
    rm.act_index, rm.room, rm.ph = 2, {"map_coord": (2, 0), "room_type": "monster"}, "map_choice"
    t.observe(rm)  # another room: the boss was beaten
    rm.room, rm.ph = {"map_coord": (3, 16), "room_type": "boss"}, "combat"
    rm.boss = {"col": 3, "row": 16, "id": "KAISER_CRAB_BOSS"}
    t.observe(rm)
    got = t.finish(False)
    assert [(b["act"], b["slot"], b["boss"], b["entry_hp"], b["won"]) for b in got] == [
        (1, "boss", "THE_KIN_BOSS", 50, True),
        (2, "boss", "KAISER_CRAB_BOSS", 20, False),
    ]
    assert got[0]["entry_potions"] == 1 and "_where" not in got[0]


def test_report_numbers():
    rows = [
        {"seed": 1, "outcome": "won", "won": True, "n_decisions": 10, "end": {"act": 3}},
        {"seed": 2, "outcome": "lost", "won": False, "n_decisions": 20, "end": {"act": 1}},
        {"seed": 3, "outcome": "capped", "won": False, "cap": "turns", "end": {"act": 2}},
        {"seed": 4, "outcome": "error", "won": False},
    ]
    rows[1]["bosses"] = [
        {"act": 1, "slot": "boss", "boss": "B", "entry_hp": 40, "entry_max_hp": 80, "won": False}
    ]
    rows[0]["bosses"] = [
        {"act": 1, "slot": "boss", "boss": "B", "entry_hp": 80, "entry_max_hp": 80, "won": True}
    ]
    out = E.report(rows)
    assert (out["rows"], out["scored"], out["missing"], out["wins"]) == (4, 3, 1, 1)
    assert out["deaths_by_act"] == {1: 1, 2: 1}
    assert out["caps"] == {"turns": 1}
    assert out["per_boss"]["a1 boss B"] == {
        "entered": 2,
        "won": 1,
        "conversion": 0.5,
        "entry_hp_frac": 0.75,
        "entry_potions": 0.0,
    }
    assert out["decisions_per_run"] == 15


# ── real runs ─────────────────────────────────────────────────────────────


@needs_tree_wheel
def test_runs_repeat_byte_for_byte_and_resume(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["SMOKE", 6, SMOKE0, "--policy", "random", "--env", "raw", *LOOSE]
    r1 = evaluate("run", "RND", *common, "--procs", 2, "--evals", a)
    assert r1.returncode == 0, r1.stderr
    assert "1 of 1 shard(s) to run" in r1.stdout and "errors_after_retry=0" in r1.stdout
    r2 = evaluate("run", "RND", *common, "--procs", 1, "--evals", b)
    assert r2.returncode == 0, r2.stderr
    files = arm_files(a, "SMOKE", "RND")
    assert list(files) == [f"shards/s{SMOKE0}_{SMOKE0 + 6}.jsonl"]
    assert files == arm_files(b, "SMOKE", "RND")
    assert (a / "SMOKE" / "RND.sha256").read_bytes() == (b / "SMOKE" / "RND.sha256").read_bytes()
    rows = [json.loads(x) for x in files[f"shards/s{SMOKE0}_{SMOKE0 + 6}.jsonl"].splitlines()]
    assert [r["seed"] for r in rows] == list(range(SMOKE0, SMOKE0 + 6))
    for r in rows:
        assert (r["ascension"], r["character"], r["policy"]) == (10, "Ironclad", "random")
        assert r["outcome"] in ("won", "lost", "capped") and isinstance(r["won"], bool)
        assert set(r["wheel"]) >= set(T.WHEEL_KEYS) and r["n_decisions"] > 0
        assert_stamped(r, cli_stamps())
    # A repeat skips complete shards; another config in the same arm is refused.
    r3 = evaluate("run", "RND", *common, "--procs", 1, "--evals", a)
    assert r3.returncode == 0 and "0 of 1 shard(s) to run" in r3.stdout
    r4 = evaluate("run", "RND", *common[:4], "first", *common[5:], "--procs", 1, "--evals", a)
    assert r4.returncode == 2 and "produced by a different config" in r4.stderr
    # The scorer and the report read what the runner wrote.
    arm = T.load_arm(a, P.SMOKE, "RND", sealed_rules=False)
    assert sorted(arm.rows) == list(range(SMOKE0, SMOKE0 + 6))
    rep = evaluate("report", "RND", "SMOKE", "--evals", a, "--json")
    assert rep.returncode == 0, rep.stderr
    assert json.loads(rep.stdout)["rows"] == 6
    config = (a / "SMOKE" / "RND" / "meta" / "config.txt").read_text(encoding="utf-8")
    assert "runner=ppo.e2e.evaluate\n" in config and "policy=random\n" in config


@needs_tree_wheel
def test_an_errored_seed_reruns_once(tmp_path):
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "flaky_env.py").write_text(HELPER, encoding="utf-8")
    (tmp_path / "marks").mkdir()
    bad = SMOKE0 + 2
    r = evaluate(
        "run", "FL", "SMOKE", 4, SMOKE0, "--policy", "first", "--env", "flaky_env:flaky",
        "--procs", 2, "--evals", tmp_path / "ev", *LOOSE,
        "--set-env", f"PYTHONPATH={tmp_path / 'mods'}{os.pathsep}{PY_DIR}",
        "--set-env", f"FLAKY_DIR={tmp_path / 'marks'}",
        "--set-env", f"FLAKY_SEED={bad}",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "re-running 1 errored seed(s) once" in r.stdout and "errors_after_retry=0" in r.stdout
    files = arm_files(tmp_path / "ev", "SMOKE", "FL")
    shard = [json.loads(x) for x in files[f"shards/s{SMOKE0}_{SMOKE0 + 4}.jsonl"].splitlines()]
    assert shard[2]["outcome"] == "error" and "flaky on purpose" in shard[2]["error"]
    (retry,) = [json.loads(x) for x in files[f"retry/s{bad}.jsonl"].splitlines()]
    assert retry["seed"] == bad and retry["outcome"] != "error"
    for row in (*shard, retry):  # the worker's error row, its played rows and the retry
        assert_stamped(row, cli_stamps())
    arm = T.load_arm(tmp_path / "ev", P.SMOKE, "FL", sealed_rules=False)
    assert arm.rows[bad]["outcome"] == retry["outcome"]


@needs_tree_wheel
def test_a_hung_run_is_killed_and_booked(tmp_path):
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "flaky_env.py").write_text(HELPER, encoding="utf-8")
    bad = SMOKE0 + 1
    r = evaluate(
        "run", "HG", "SMOKE", 3, SMOKE0, "--policy", "first", "--env", "flaky_env:hang",
        "--procs", 2, "--hang-timeout", 2, "--evals", tmp_path / "ev", *LOOSE,
        "--set-env", f"PYTHONPATH={tmp_path / 'mods'}{os.pathsep}{PY_DIR}",
        "--set-env", f"FLAKY_SEED={bad}",
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert "errors_after_retry=1" in r.stdout
    files = arm_files(tmp_path / "ev", "SMOKE", "HG")
    shard = [json.loads(x) for x in files[f"shards/s{SMOKE0}_{SMOKE0 + 3}.jsonl"].splitlines()]
    assert [x["outcome"] == "error" for x in shard] == [False, True, False]
    assert "killed: no result after 2 s" in shard[1]["error"]
    (retry,) = [json.loads(x) for x in files[f"retry/s{bad}.jsonl"].splitlines()]
    assert retry["outcome"] == "error"
    for row in (*shard, retry):  # the pool books the hung runs' rows itself
        assert_stamped(row, cli_stamps())


@needs_tree_wheel
def test_a_policy_that_does_not_load_stops_the_call(tmp_path):
    r = evaluate(
        "run", "BAD", "SMOKE", 3, SMOKE0, "--policy", "no_such_module:make", "--env", "raw",
        "--procs", 1, "--evals", tmp_path, *LOOSE,
    )  # fmt: skip
    assert r.returncode == 2
    assert "could not build its policy/env" in r.stderr and "no_such_module" in r.stderr
    assert not list((tmp_path / "SMOKE" / "BAD" / "shards").glob("*.jsonl"))
    assert not (tmp_path / "SMOKE" / "BAD.sha256").exists()


# The e2e lane is Linux-only (systemd units, /dev/shm); the public CI also runs Windows.
pytestmark = pytest.mark.skipif(
    'sys.platform == "win32"',
    reason="evaluate runs its shards under a stripped POSIX environment (no HOME/USERPROFILE). Linux only, like the lane.",
)
