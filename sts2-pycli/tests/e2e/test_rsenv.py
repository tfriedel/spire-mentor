"""The Rust training env (`ppo.e2e.rsenv`, `sts2_rs.E2eTrainEnv`) against `env.E2EEnv`.

`rsenv_lockstep` plays both envs with the same choices: every row (fields, reward,
tokens with PRIV), the end, the counters, the captures and the F9 states must agree.
Pinned runs (player HP 100,000) take random play through every act, the double boss
and practice starts. The worker test runs `actors.worker_main` on either env and
compares every message byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal

import numpy as np
import pytest
import sts2_client
from ppo.e2e import actors, factory
from ppo.e2e import contracts as C
from ppo.e2e import env as E
from ppo.e2e.rsenv import RsDecision, RsE2EEnv
from ppo.e2e.rsenv_lockstep import Lockstep, run_seeds

SEED = 1_300_000_000


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _report(ls: Lockstep) -> str:
    return json.dumps({"stats": dict(ls.stats), "examples": ls.examples}, indent=1)[:4000]


def test_random_runs_agree_row_for_row() -> None:
    ls = Lockstep(f9_p=0.2)
    run_seeds(ls, SEED, 6, pinned=False, practice=0)
    assert ls.stats["rows"] > 100
    assert ls.stats["f9_hits"] > 10
    assert ls.mismatches == 0, _report(ls)


def test_deep_runs_and_practice_agree_row_for_row() -> None:
    """Pinned random runs reach act 3; their captures start practice segments
    (`first_ooc` bootstrap ends, the double boss)."""
    ls = Lockstep(f9_p=0.02)
    run_seeds(ls, SEED + 100, 2, pinned=True, practice=3)
    assert ls.stats["captures"] > 0
    assert any(k.startswith("practice/") for k in ls.stats), _report(ls)
    assert ls.mismatches == 0, _report(ls)


def test_a_rare_map_reaches_sctx_and_priv_draw() -> None:
    """A RareMap that sends every card to row 1 changes the rows of both envs alike."""
    n = C.SCTX_TABLE_SIZES["card"]
    rare = {"card": [0] + [1] * (n - 1)}
    ls = Lockstep(f9_p=0.0, rare_json=rare)
    run_seeds(ls, SEED + 7, 2, pinned=False, practice=0)
    assert ls.stats["combat"] > 0
    assert ls.mismatches == 0, _report(ls)


def test_the_rust_row_raises_what_the_tokenizer_would() -> None:
    r = {
        "row": 0,
        "is_combat": True,
        "dtype": 0,
        "n_classes": 2,
        "reward": 0.0,
        "bootstrap": False,
        "unit": ("fight", 0),
        "tok_error": "SchemaError: members of one class tokenize differently: [[0, 1]]",
    }
    with pytest.raises(C.SchemaError):
        _ = RsDecision(r, C.CELL_NONE).tok
    with pytest.raises(RuntimeError):
        _ = RsDecision({**r, "tok_error": "S_ctx built 3 action tokens for 2 members"}, 0).tok


def test_a_shared_f9_checker_is_refused() -> None:
    from ppo.e2e.invariants import InvariantChecker

    with pytest.raises(ValueError, match="shared"):
        RsE2EEnv(SEED, cfg=E.EnvConfig(f9=InvariantChecker()))


class _Conn:
    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.r = 0
        self.h = hashlib.sha256()
        self.rng = np.random.default_rng(0)
        self.n: np.ndarray = np.zeros(0)
        self.ends = 0

    def send(self, msg) -> None:
        _, _wid, metas, r_prev, rb, _extra, ends = msg
        for k in sorted(rb):
            v = rb[k]
            if isinstance(v, np.ndarray):
                self.h.update(k.encode() + str(v.dtype).encode())
                self.h.update(np.ascontiguousarray(v).tobytes())
        self.h.update(metas.tobytes())
        self.h.update(np.asarray(r_prev, np.float64).tobytes())
        skip = {"wheel", "py_src", "error", "f9_samples"}
        for e in ends:
            self.ends += 1
            self.h.update(
                json.dumps({k: v for k, v in e.items() if k not in skip}, sort_keys=True).encode()
            )
        self.n = metas[:, 8]

    def recv(self):
        self.r += 1
        if self.r > self.rounds:
            return ("stop",)
        return (
            "act",
            [int(self.rng.integers(max(1, int(n)))) for n in self.n],
            {"decisions": 4_000_000_000 + self.r},
        )

    def close(self) -> None:
        pass


def _worker(rust: bool, monkeypatch: pytest.MonkeyPatch) -> tuple[str, int]:
    # `worker_main` sets SIGINT/SIGTERM to SIG_IGN and RAYON/OMP thread counts for
    # its process; in-process here, that would outlive the test in the pytest
    # (xdist) worker and every subprocess it spawns later (test_actors.py does
    # the same).
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(k, os.environ.get(k, "1"))
    conn = _Conn(60)
    cfg = {
        "n_envs": 8,
        "run_seed_base": SEED,
        "start_it": 0,
        "start_decisions": 4_000_000_000,
        "seed": 0,
        "practice": True,
        "f9_tolerate": [],
        "teacher_rare": None,
        "run_teacher_honest": False,
        "action_log_dir": None,
        "rust_env": rust,
    }
    actors.worker_main(0, conn, cfg)
    return conn.h.hexdigest(), conn.ends


def test_the_worker_sends_the_same_messages_on_either_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    py, n = _worker(False, monkeypatch)
    rs, m = _worker(True, monkeypatch)
    assert n > 0
    assert (rs, m) == (py, n)


def _play(env, rng: np.random.Generator):
    """Random merged classes to the segment's end; the terminal."""
    out = env.reset()
    while not isinstance(out, E.E2ETerminal):
        out = env.step(int(rng.integers(max(1, out.n_classes))))
    return out


def test_envconfig_caps_reach_the_rust_env() -> None:
    """`EnvConfig`'s caps are the Rust env's, not ignored: both envs cap the same run
    at the same step."""
    for cfg, cap in (
        (E.EnvConfig(max_run_steps=20, f9_p=0.0), "steps"),
        (E.EnvConfig(max_run_decisions=10, f9_p=0.0), "run_decisions"),
    ):
        py = _play(E.E2EEnv(SEED, cfg=cfg), np.random.default_rng(3))
        rs = _play(RsE2EEnv(SEED, cfg=cfg), np.random.default_rng(3))
        assert py.cap == cap
        assert (rs.outcome, rs.cap, rs.steps, rs.rows) == (py.outcome, py.cap, py.steps, py.rows)


def test_the_f9_queue_is_bounded_and_drops_are_counted() -> None:
    """At f9_rate 1 every applied action is a hit: each drain holds at most
    `f9_queue_max` runs, and kept plus dropped hits account for every step."""
    import sts2_rs

    env = sts2_rs.E2eTrainEnv(
        factory.new_run(SEED + 1), SEED + 1, f9_rate=1.0, encode=False, f9_queue_max=1
    )
    rng = np.random.default_rng(1)
    kept, biggest = 0, 0
    r = env.reset()
    while True:
        n = len(env.drain_f9())
        kept += n
        biggest = max(biggest, n)
        if r["kind"] == "terminal":
            break
        r = env.step(int(rng.integers(max(1, r["n_classes"]))))
    assert r["outcome"] in ("won", "lost", "capped"), r
    dropped = int(r["counts"].get("f9/dropped", 0))
    assert biggest <= 1
    assert dropped > 0  # some call applies more than one action
    assert kept + dropped == r["steps"]


def test_a_fight_on_another_build_raises_like_the_python_env() -> None:
    """`factory.check_fight`: a fight simulating another build propagates
    `GameVersionMismatch` (not an error terminal) from both envs."""
    import sts2_rs

    def run() -> object:
        return sts2_rs.RunManager(
            SEED,
            character=factory.CHARACTER,
            ascension_level=factory.ASCENSION,
            neow_start=True,
            act1=factory.ACT1,
            game_version="v0.107.1",
            mode=factory.MODE,
            interactive_potion_swap=True,
        )

    cfg = E.EnvConfig(allow_any_run=True, f9_p=0.0)
    for env in (E.E2EEnv(SEED, rm=run(), cfg=cfg), RsE2EEnv(SEED, rm=run(), cfg=cfg)):
        with pytest.raises(sts2_client.GameVersionMismatch):
            _play(env, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("etype", "cls"),
    [
        ("SchemaError", C.SchemaError),
        ("GameVersionMismatch", sts2_client.GameVersionMismatch),
        ("NoActions", E.NoActions),
        ("PanicException", RuntimeError),
        ("RuntimeError", RuntimeError),
    ],
)
def test_a_raised_rust_error_keeps_its_python_type(etype: str, cls: type) -> None:
    from ppo.e2e.rsenv import _python_error

    ex = _python_error(etype, f"{etype}: boom")
    assert isinstance(ex, cls)
    assert str(ex) == f"{etype}: boom"  # no doubled prefix
    assert E.ends_segment(ex) == (etype != "GameVersionMismatch")
