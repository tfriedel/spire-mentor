"""The e2e rollout worker (`ppo.e2e.actors.worker_main`), in-process on scripted envs.

`worker_main` runs here with a fake pipe, a scripted `practice.SegmentSource` and a
stubbed tokenizer, so the tests see exactly what the worker sends: its row metas and
its segment end records. What a segment's failure does follows `env.ends_segment`
(GH #1097 A2), outside the env too: an exception or a Rust panic ends the segment as a
counted error, a build mismatch or an interrupt leaves the worker.
"""

from __future__ import annotations

import os
import signal
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sts2_client
from ppo.e2e import actors, practice
from ppo.e2e import contracts as C
from ppo.e2e import env as E
from ppo.e2e import tokens as TK

#: A pyo3 panic as `env.is_rust_panic` sees one: a BaseException from `pyo3_runtime`.
PanicException = type("PanicException", (BaseException,), {"__module__": "pyo3_runtime"})


class FakeConn:
    """The worker's pipe: records what it sends, answers `rounds` rounds, then stops."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.sent: list[Any] = []
        self.closed = False

    def send(self, msg: Any) -> None:
        self.sent.append(msg)

    def recv(self) -> Any:
        if len(self.sent) >= self.rounds:
            return ("stop",)
        meta = self.sent[-1][2]
        return ("act", [0] * len(meta), {"decisions": 0})

    def close(self) -> None:
        self.closed = True

    def ends(self) -> list[dict[str, Any]]:
        return [e for msg in self.sent for e in msg[6]]


class FakeDecision:
    """An `env.E2EDecision` as the worker reads one (`E2EDecision` is patched to it)."""

    def __init__(self, t: int, *, bootstrap: bool = False) -> None:
        self.t = t
        self.is_combat = False
        self.dtype = C.DT["MAP"]
        self.cell = C.CELL_NONE
        self.unit = ("act", 0)
        self.n_classes = 2
        self.reward = 0.0
        self.bootstrap = bootstrap


def fake_term(env: FakeEnv) -> SimpleNamespace:
    return SimpleNamespace(
        counts=Counter(env.counts),
        reward=0.0,
        outcome=env.outcome,
        won=False,
        cap="run_decisions" if env.outcome == "capped" else None,
        floor=1,
        act=0,
        bosses=0,
        reached_boss2=False,
        rows=env.t,
        steps=env.t,
        seed=env.seed,
        error=None,
        total_reward=0.0,
        unassigned_reward=0.0,
        segment=env.segment,
        fingerprint=f"fp{env.seed}",
        log_idx=[0] * env.t,
    )


class FakeEnv:
    """A scripted segment of `length` rows. `on_reset` / `on_step(t)` may raise."""

    def __init__(
        self,
        seed: int,
        *,
        length: int = 3,
        on_reset: Callable[[], None] | None = None,
        on_step: Callable[[int], None] | None = None,
        outcome: str = "lost",
    ) -> None:
        self.seed = seed
        self.length = length
        self.on_reset = on_reset
        self.on_step = on_step
        self.outcome = outcome
        self.segment = SimpleNamespace(kind="run", cell=None, header=lambda: {"kind": "run"})
        self.counts: Counter = Counter({"f9/checks": 2, "fights/monster": 1})
        self.t = 0
        self.terminal: Any = None
        self.rm = object()

    def reset(self) -> Any:
        if self.on_reset is not None:
            self.on_reset()
        return FakeDecision(0)

    def step(self, a: int) -> Any:
        if self.on_step is not None:
            self.on_step(self.t)
        self.t += 1
        if self.t >= self.length:
            self.terminal = fake_term(self)
            return self.terminal
        return FakeDecision(self.t)


class Harness:
    """`worker_main` on one slot whose segments `make_env(n)` scripts (n = 0, 1, ...)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, make_env: Callable[[int], Any]) -> None:
        self.make_env = make_env
        self.started = 0
        self.envs: list[Any] = []
        self.encode: Callable[[Any], Any] = lambda d: SimpleNamespace(n_classes=d.n_classes)
        h = self

        class Source:
            def __init__(self, queues: Any, seeds: Any, n_slots: int, cfg: Any, **kw: Any) -> None:
                self.stats: Counter = Counter()

            def next(self, slot: int, t: int) -> Any:
                n = h.started
                h.started += 1
                env = h.make_env(n)
                h.envs.append(env)
                return env

        monkeypatch.setattr(practice, "SegmentSource", Source)
        monkeypatch.setattr(E, "E2EDecision", FakeDecision)
        monkeypatch.setattr(TK, "encode", lambda d, **kw: h.encode(d))
        monkeypatch.setattr(TK, "row_batch", lambda toks, **kw: {"n": len(toks)})
        monkeypatch.setattr(signal, "signal", lambda *a: None)  # keep pytest's handlers
        for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
            monkeypatch.setenv(k, os.environ.get(k, "1"))

    def run(self, rounds: int, n_envs: int = 1, **extra: Any) -> FakeConn:
        conn = FakeConn(rounds)
        cfg = {
            "n_envs": n_envs,
            "run_seed_base": 0,
            "start_it": 0,
            "start_decisions": 0,
            "seed": 0,
            "practice": False,
            "f9_tolerate": [],
            "teacher_rare": None,
            "run_teacher_honest": False,
            **extra,
        }
        actors.worker_main(0, conn, cfg)
        return conn


def _mismatch() -> None:
    raise sts2_client.GameVersionMismatch("the process selects v0.107.1")


def test_scripted_segments_end_as_their_env_says(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, lambda n: FakeEnv(100 + n, length=3))
    conn = h.run(7)
    assert conn.closed
    ends = conn.ends()
    assert [e["outcome"] for e in ends] == ["lost", "lost"]
    assert [e["seed"] for e in ends] == [100, 101]
    assert all(np.asarray(m[2]).shape == (1, len(actors.META_COLS)) for m in conn.sent)


@pytest.mark.parametrize("where", ["start", "step"])
def test_a_build_mismatch_leaves_the_worker(monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    """A `GameVersionMismatch` (an AssertionError) means the process or the engine is on
    the wrong build: every later segment would fail the same way, so the worker stops
    with it (`env.ends_segment`), instead of booking error segments or retrying the
    start up to `MAX_START_FAILS` times."""
    if where == "start":
        h = Harness(monkeypatch, lambda n: FakeEnv(n, on_reset=_mismatch))
    else:
        h = Harness(monkeypatch, lambda n: FakeEnv(n, on_step=lambda t: _mismatch()))
    with pytest.raises(sts2_client.GameVersionMismatch):
        h.run(5)
    assert h.started == 1


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_leaves_the_worker(monkeypatch: pytest.MonkeyPatch, exc: type) -> None:
    def boom(t: int) -> None:
        raise exc()

    h = Harness(monkeypatch, lambda n: FakeEnv(n, on_step=boom))
    with pytest.raises(exc):
        h.run(5)


def test_a_rust_panic_at_a_start_errors_the_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`E2EEnv`/`RunManager` construction, `practice.start_env` and the env's first
    `run.reset()` run outside the env's guards: a panic there is a failed start (counted,
    retried), not a dead worker."""

    def panic() -> None:
        raise PanicException("index out of bounds")

    h = Harness(monkeypatch, lambda n: FakeEnv(n, on_reset=panic if n == 0 else None))
    conn = h.run(6)
    ends = conn.ends()
    assert ends[0]["outcome"] == "error" and ends[0]["counts"]["error/PanicException"] == 1
    assert ends[0]["t_last"] == -1
    assert [e["outcome"] for e in ends[1:]] == ["lost"]


def test_a_rust_panic_in_the_tokenizer_errors_the_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tokenizer's getter reads (`bv.telegraph`, `cs.true_draw_top`, ...) are Rust
    calls outside the env's guards: a panic there ends the segment as an error."""
    h = Harness(monkeypatch, lambda n: FakeEnv(n, length=3))

    def encode(d: Any) -> Any:
        if h.started == 1 and d.t == 1:
            raise PanicException("a getter panicked")
        return SimpleNamespace(n_classes=d.n_classes)

    h.encode = encode
    conn = h.run(6)
    ends = conn.ends()
    assert ends[0]["outcome"] == "error" and ends[0]["counts"]["error/PanicException"] == 1
    assert ends[0]["t_last"] == 0  # the segment's row 0 went out before the panic
    assert "lost" in [e["outcome"] for e in ends[1:]]


def test_an_actor_side_error_keeps_the_envs_counts_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment the worker ends itself (a tokenizer error) still reports what its env
    counted so far: an F9 violation sampled earlier in the segment must reach the
    learner, which reads violations only from end records (plan §8: one is a lane
    stop), and the seed lets the error be replayed (review finding ENVR-4)."""

    def make(n: int) -> FakeEnv:
        env = FakeEnv(4242 + n, length=3)
        env.counts.update({"f9/violations": 1, "f9/hp_le_max": 1, "unknown": 2})
        return env

    h = Harness(monkeypatch, make)

    def encode(d: Any) -> Any:
        if h.started == 1 and d.t == 1:
            raise C.SchemaError("3 token classes for 2")
        return SimpleNamespace(n_classes=d.n_classes)

    h.encode = encode
    ends = h.run(3).ends()
    e = ends[0]
    assert e["outcome"] == "error" and e["seed"] == 4242
    assert (e["f9_checks"], e["f9_violations"], e["unknown"], e["fights"]) == (2, 1, 2, 1)
    assert e["counts"]["error/SchemaError"] == 1 and e["counts"]["f9/hp_le_max"] == 1


def test_a_segment_start_drops_only_its_slots_last_run_from_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tokenizer's run-static cache is keyed by the RunManager (`id(d.rm)` on the env
    path). A slot that starts a segment drops its own last run's entry, not the other
    slots' (review findings ENVR-5, TOK-2: a full clear made every slot's next row a
    cold encode). The cache stays at no more than `n_envs` entries."""
    cleared: list[Any] = []
    monkeypatch.setattr(TK, "clear_cache", lambda key=None: cleared.append(key))
    lengths = [2, 5, 3, 4, 2, 6, 3]
    h = Harness(monkeypatch, lambda n: FakeEnv(n, length=lengths[n % len(lengths)]))
    h.run(9, n_envs=2)
    assert None not in cleared
    # Each start after a slot's first drops that slot's previous env's run, once.
    ids = {id(env.rm) for env in h.envs}
    assert cleared and set(cleared) <= ids and len(cleared) == len(set(cleared))
    assert len(cleared) == h.started - 2


def test_the_worker_logs_the_segments_its_sink_keeps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review finding ENVR-1: the worker offers every segment it ends to an
    `actionlog.LogSink` in the learner's `<save_dir>/actions`. Every capped or error
    segment is kept (what the CAPS alarm sends the operator to), a plain loss only in the
    2% sample. The worker's stats report the sink."""
    from ppo.e2e import actionlog as AL

    outcomes = ["capped", "lost", "error", "lost"]
    h = Harness(monkeypatch, lambda n: FakeEnv(700 + n, length=2, outcome=outcomes[n % 4]))
    conn = h.run(70, action_log_dir=str(tmp_path / "actions"))
    recs = list(AL.read_records(tmp_path / "actions" / "actions-000.e2el"))
    kept = {(r.header["seed"], r.header["reason"]) for r in recs if r.header["reason"] != "sample"}
    want = {(e.seed, e.outcome) for e in h.envs if e.outcome != "lost" and e.terminal is not None}
    assert want and kept == want
    rec = next(r for r in recs if r.header["reason"] == "capped")
    uid = {e["seed"]: e["uid"] for e in conn.ends()}
    assert rec.header["cap"] == "run_decisions"
    assert rec.header["extra"]["uid"] == uid[rec.header["seed"]]
    assert rec.idx == [0, 0] and rec.header["fingerprint"] == f"fp{rec.header['seed']}"
    stats = [m[5]["stats"] for m in conn.sent if m[5]["stats"] is not None]
    assert 0 < stats[-1]["actions"]["records"] <= len(recs)  # sampled every 64 rounds
    assert stats[-1]["actions"]["dropped"] == 0 and not stats[-1]["actions"]["full"]
    # A failing log (a full disk) is counted, never a segment error.

    def full(self: Any, rec: Any) -> bool:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(AL.LogSink, "write", full)
    h = Harness(monkeypatch, lambda n: FakeEnv(800 + n, length=2, outcome="capped"))
    conn = h.run(70, action_log_dir=str(tmp_path / "full"))
    assert {e["outcome"] for e in conn.ends()} == {"capped"}
    stats = [m[5]["stats"] for m in conn.sent if m[5]["stats"] is not None]
    assert stats[-1]["worker"]["log_errors"] > 0


def test_a_worker_error_keeps_the_segment_in_the_action_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A segment the worker ends on its own exception goes to the action log as a
    `worker_error` record (its choices so far replay to the failed decision, #1097), and
    still reaches the learner as an `error` end. A record that cannot be built is counted
    in `log_errors`, never a second error."""
    from ppo.e2e import actionlog as AL

    def make(n: int) -> FakeEnv:
        env = FakeEnv(5150 + n, length=3)
        env.log_idx = [0] * 2
        return env

    h = Harness(monkeypatch, make)

    def encode(d: Any) -> Any:
        if h.started == 1 and d.t == 1:
            raise C.SchemaError("members of one class tokenize differently: [[0, 1]]")
        return SimpleNamespace(n_classes=d.n_classes)

    h.encode = encode
    seen: list[tuple[Any, str]] = []

    def rec(env: Any, err: str, *, extra: Any = None) -> Any:
        seen.append((env, err))
        head = {
            "seed": env.seed,
            "outcome": AL.WORKER_ERROR,
            "reason": AL.WORKER_ERROR,
            "extra": extra,
        }
        return AL.LogRecord(header=head, idx=list(env.log_idx))

    monkeypatch.setattr(AL, "worker_error_record", rec)
    conn = h.run(3, action_log_dir=str(tmp_path / "actions"))
    assert conn.ends()[0]["outcome"] == "error"
    recs = [
        r
        for r in AL.read_records(tmp_path / "actions" / "actions-000.e2el")
        if r.header["reason"] == AL.WORKER_ERROR
    ]
    assert len(recs) == 1 and recs[0].header["seed"] == 5150
    assert recs[0].header["extra"]["etype"] == "SchemaError"
    assert "tokenize differently" in seen[0][1]

    def broken(env: Any, err: str, *, extra: Any = None) -> Any:
        raise AttributeError("no rm")

    monkeypatch.setattr(AL, "worker_error_record", broken)
    h = Harness(monkeypatch, make)
    h.encode = encode
    conn = h.run(70, action_log_dir=str(tmp_path / "broken"))  # stats every 64 rounds
    assert [e["outcome"] for e in conn.ends()][:1] == ["error"]
    stats = [m[5]["stats"] for m in conn.sent if m[5]["stats"] is not None]
    assert stats[-1]["worker"]["log_errors"] == 1
