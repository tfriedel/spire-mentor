"""Action logs (`ppo.e2e.actionlog`, plan §3.2): the record format, the sink's keep rules
and byte cap, and replay of runs, captures and practice segments."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import sts2_client
from ppo.e2e import actionlog as L
from ppo.e2e import factory
from ppo.e2e import practice as P
from ppo.e2e.env import Capture, E2EDecision, E2EEnv, EnvConfig

SMOKE = 162_000_000


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _actor(seed: int):
    rng = random.Random(seed)

    def act(d: E2EDecision) -> int:
        if d.kind == "combat":
            ends = {c for c, a in zip(d.classes, d.actions) if a["action"] == "end_turn"}
            others = [c for c in range(d.n_classes) if c not in ends]
            if others and rng.random() < 0.85:
                return rng.choice(others)
        return rng.randrange(d.n_classes)

    return act


def _walk(env: E2EEnv, act) -> object:
    d = env.reset()
    while isinstance(d, E2EDecision) and not d.bootstrap:
        d = env.step(act(d))
    return env.terminal


# ── format ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("idx", "width"), [([0, 3, 255, 1], 1), ([0, 300, 65535], 2), ([], 1)])
def test_records_round_trip(idx: list[int], width: int) -> None:
    rec = L.LogRecord(header={"seed": 5, "segment": {"kind": "run"}}, idx=idx)
    blob = rec.encode()
    assert blob[:4] == L.MAGIC and blob[4:5] in (L.CODEC_ZSTD, L.CODEC_LZMA)
    got, end = L.decode(blob)
    assert end == len(blob)
    assert got.idx == idx and got.header["seed"] == 5
    assert got.header["width"] == width and got.header["n"] == len(idx)


def test_an_index_past_u16_is_refused() -> None:
    with pytest.raises(ValueError, match="u16"):
        L.LogRecord(header={}, idx=[70_000]).encode()


def test_a_torn_tail_loses_only_the_last_record(tmp_path) -> None:
    path = tmp_path / "a.e2el"
    recs = [L.LogRecord(header={"k": k}, idx=[k, k + 1]) for k in range(3)]
    blob = b"".join(r.encode() for r in recs)
    path.write_bytes(blob[:-3])
    got = list(L.read_records(path))
    assert [r.header["k"] for r in got] == [0, 1]
    with pytest.raises(ValueError):
        L.decode(b"XXXX" + blob[4:])


def test_stamps_fall_back_only_for_a_wheel_without_the_getter(monkeypatch) -> None:
    """A2 audit: a wheel without `build_git` stamps `unknown`; any other failure of the
    getter propagates rather than stamping the log as from an unknown build."""
    import sts2_rs

    assert L._stamps() == {"game_version": "v0.111.0", "wheel_build": str(sts2_rs.build_git())}
    monkeypatch.delattr(sts2_rs, "build_git")
    assert L._stamps() == {"game_version": "v0.111.0", "wheel_build": "unknown"}

    def broken() -> str:
        raise RuntimeError("stub getter failure")

    monkeypatch.setattr(sts2_rs, "build_git", broken, raising=False)
    with pytest.raises(RuntimeError, match="stub getter failure"):
        L._stamps()


# ── the sink ─────────────────────────────────────────────────────────────────


def _term(kind: str = "run", boss2: bool = False, outcome: str = "lost") -> SimpleNamespace:
    return SimpleNamespace(reached_boss2=boss2, segment=SimpleNamespace(kind=kind), outcome=outcome)


def test_the_sink_keeps_evals_boss2_and_a_sample(tmp_path) -> None:
    keep_all = L.LogSink(tmp_path / "a", worker=0, seed=1, train_p=1.0)
    keep_none = L.LogSink(tmp_path / "b", worker=0, seed=1, train_p=0.0)
    assert keep_none.reason(_term(), eval_run=True) == "eval"
    assert keep_none.reason(_term(boss2=True), eval_run=False) == "boss2"
    assert keep_none.reason(_term("practice", boss2=True), eval_run=False) == "boss2"
    assert keep_none.reason(_term(), eval_run=False) is None
    # Every capped or error segment (review finding ENVR-1: the CAPS alarm's evidence).
    for kind in ("run", "practice"):
        for out in ("capped", "error"):
            assert keep_none.reason(_term(kind, outcome=out), eval_run=False) == out
    assert keep_all.reason(_term(), eval_run=False) == "sample"
    assert keep_all.reason(_term("practice"), eval_run=False) is None  # runs only
    sink = L.LogSink(tmp_path / "c", worker=0, seed=3)
    n = sum(sink.reason(_term(), eval_run=False) == "sample" for _ in range(20_000))
    assert 300 < n < 500  # 2% of 20,000


def test_the_byte_cap_covers_every_worker(tmp_path) -> None:
    rec = L.LogRecord(header={"seed": 1}, idx=list(range(200)))
    size = len(rec.encode())
    a = L.LogSink(tmp_path, worker=0, seed=0, cap_bytes=3 * size, rescan=1)
    b = L.LogSink(tmp_path, worker=1, seed=0, cap_bytes=3 * size, rescan=1)
    assert a.write(rec) and b.write(rec) and a.write(rec)
    assert not b.write(rec)  # b rescans and sees a's bytes
    assert b.dropped == 1 and b.full
    assert not L.LogSink(tmp_path, worker=2, seed=0, cap_bytes=3 * size).write(rec)
    assert sum(1 for p in tmp_path.glob("*.e2el") for _ in L.read_records(p)) == 3


# ── replay ───────────────────────────────────────────────────────────────────


def test_a_logged_run_replays_to_its_fingerprint(tmp_path) -> None:
    sink = L.LogSink(tmp_path, worker=0, seed=0, train_p=1.0)
    terms = [_walk(E2EEnv(SMOKE + k), _actor(k)) for k in (10, 11)]
    assert all(sink.offer(t) for t in terms)
    recs = list(L.read_records(sink.path))
    assert len(recs) == 2
    for t, rec in zip(terms, recs):
        assert rec.header["game_version"] == "v0.111.0"
        assert rec.header["reason"] == "sample"
        back = L.replay(rec)
        assert back.fingerprint == t.fingerprint and back.outcome == t.outcome
        assert back.rewards == t.rewards


def test_a_spawned_worker_replays_a_logged_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env, the factory and the log replay work in a spawned process (spawn-safe):
    the worker inherits the version selection through the environment."""
    import multiprocessing

    monkeypatch.setenv(sts2_client.GAME_VERSION_ENV, factory.GAME_VERSION)
    t = _walk(E2EEnv(SMOKE + 13), _actor(13))
    rec = L.terminal_record(t, reason="eval")
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        back = pool.apply(L.replay, (rec,))
    assert back.fingerprint == t.fingerprint and back.rewards == t.rewards
    assert back.counts == t.counts


def test_a_tampered_log_does_not_replay() -> None:
    t = _walk(E2EEnv(SMOKE + 12), _actor(12))
    rec = L.terminal_record(t, reason="eval")
    rec.idx[0] = 1 if rec.idx[0] == 0 else 0  # another Neow option (OOC: no merged twins)
    with pytest.raises((L.ReplayMismatch, IndexError)):
        L.replay(rec)


def test_a_record_without_a_fingerprint_is_refused() -> None:
    rec = L.LogRecord(
        header={"seed": 1, "segment": {"kind": "run"}, "fingerprint": "unavailable"}, idx=[]
    )
    with pytest.raises(L.ReplayMismatch, match="no fingerprint"):
        L.replay(rec)


@pytest.fixture(scope="module")
def plain_capture() -> Capture:
    """An elite or boss capture from an unpinned run (replayable from seed and log)."""
    for seed in range(SMOKE + 20, SMOKE + 120):
        caps: list[Capture] = []
        _walk(E2EEnv(seed, cfg=EnvConfig(on_capture=caps.append)), _actor(seed))
        if caps:
            return caps[0]
    pytest.fail("no elite or boss room in 100 random runs")


def test_a_capture_replays_from_its_record(plain_capture: Capture) -> None:
    from ppo.e2e.env import run_fingerprint

    rec = L.capture_record(plain_capture)
    back, _ = L.decode(rec.encode())
    cap = L.replay(back)
    assert cap.cell == plain_capture.cell and cap.steps == plain_capture.steps
    assert run_fingerprint(cap.snap) == run_fingerprint(plain_capture.snap)


def test_a_practice_segment_replays_from_its_record(plain_capture: Capture) -> None:
    env = P.start_env(plain_capture, P.RESEED_LO + 3)
    assert env is not None
    t = _walk(env, _actor(3))
    rec = L.terminal_record(t, reason="eval")
    assert rec.header["segment"]["kind"] == "practice"
    assert rec.header["segment"]["reseed"] == P.RESEED_LO + 3
    back = L.replay(L.decode(rec.encode())[0])
    assert back.fingerprint == t.fingerprint and back.outcome == t.outcome


def test_a_worker_error_record_replays_to_the_failed_decision() -> None:
    """A segment a worker ended on its own exception (a tokenizer `SchemaError`) is kept
    with its choices so far, and `replay_open` rebuilds the decision it failed on (#1097:
    the member-class errors had no repro). `replay` refuses it: it has no terminal."""
    env = E2EEnv(SMOKE + 21)
    act = _actor(21)
    d = env.reset()
    for _ in range(40):
        d = env.step(act(d))
    assert isinstance(d, E2EDecision) and not d.bootstrap
    rec = L.worker_error_record(env, "Traceback ...\nSchemaError: x", extra={"uid": 7})
    back, _ = L.decode(rec.encode())
    assert back.header["outcome"] == back.header["reason"] == L.WORKER_ERROR
    assert back.header["error"].endswith("SchemaError: x") and back.header["extra"]["uid"] == 7
    assert back.idx == env.log_idx and back.header["steps"] == len(env.log_idx)
    env2, d2 = L.replay_open(back)
    assert (d2.kind, d2.n_classes, d2.actions) == (d.kind, d.n_classes, d.actions)
    assert env2.log_idx == env.log_idx
    with pytest.raises(L.ReplayMismatch, match="replay_open"):
        L.replay(back)
    bad = L.LogRecord(header=dict(back.header, fingerprint="0" * 24), idx=back.idx)
    with pytest.raises(L.ReplayMismatch, match="fingerprint"):
        L.replay_open(bad)
    term = _walk(E2EEnv(SMOKE + 22), _actor(22))
    with pytest.raises(L.ReplayMismatch, match="not a worker_error"):
        L.replay_open(L.terminal_record(term, reason="eval"))
