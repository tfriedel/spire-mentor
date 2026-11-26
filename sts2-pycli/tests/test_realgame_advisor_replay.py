"""The whole advisor over a recorded live run (GH #1426 step 0).

``realgame_advisor_replay`` drives the advisor's own message loop with the
recorded corpus in place of the bridge socket. These pin what the Rust port
of the advisor leans on: every recorded request gets exactly one reply, none
is an ``advisor error``, and a replay is reproducible (with one search tree
and the coach inline), so its replies can serve as golden fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sts2_rs", reason="the advisor needs the sts2_rs extension")

import realgame_advisor_replay as rr

CORPUS = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "2026-11-07-instinct-live"
    / "run1-requests.jsonl.gz"
)

#: The first fights, maps and card rewards of act 1.
LIMIT = 60
FAST = {"sims": 64, "boss_sims": 128, "threads": 1, "limit": LIMIT}

GOLDENS = sorted((Path(__file__).parent / "fixtures" / "advisor_replay").glob("*.jsonl.gz"))

pytestmark = pytest.mark.skipif(not CORPUS.is_file(), reason="no live corpus")


def _advice(replies: list[dict]) -> list[dict]:
    return [r for r in replies if r.get("action") == "advice"]


@pytest.mark.parametrize("brain", ["recorded", "search"])
def test_every_recorded_request_gets_one_reply(brain: str) -> None:
    replies = rr.replay(CORPUS, brain=brain, **FAST)
    advice = _advice(replies)
    assert len(advice) == LIMIT
    requests = [m for m in rr.read_corpus(CORPUS) if m.get("type") == "advice_request"][:LIMIT]
    assert [a["request_id"] for a in advice] == [m["request_id"] for m in requests]
    errors = [a for a in advice if "advisor error" in str(a["advice"].get("note") or "")]
    assert not errors, errors[:3]
    assert all(a["advice"].get("best") or a["advice"].get("hand_back") for a in advice)


def test_a_replay_is_reproducible() -> None:
    a = rr.replay(CORPUS, brain="search", **FAST)
    b = rr.replay(CORPUS, brain="search", **FAST)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_static_closure_finds_the_advisor_modules() -> None:
    closure = rr.static_closure()
    for name in ("realgame_advisor", "combat_mirror", "realgame_ppo", "sts2_hint.adapter"):
        assert name in closure, name
    # Parent packages run their __init__ on a submodule import.
    assert "sts2_hint" in closure


def test_compare_replies_tolerates_float_noise_only() -> None:
    want = [{"action": "advice", "advice": {"p": 0.6523042321205139, "best": {"index": 0}}}]
    noise = [{"action": "advice", "advice": {"p": 0.6523041725158691, "best": {"index": 0}}}]
    other = [{"action": "advice", "advice": {"p": 0.6523042321205139, "best": {"index": 1}}}]
    assert rr.compare_replies(noise, want) == []
    # A probability rounded to 4 decimals on the other side of a boundary
    # (a4000 vs an AVX-512 box, reply #468 of the live corpus).
    rounded = [{"action": "advice", "advice": {"prob": 0.2678}}]
    assert rr.compare_replies(rounded, [{"action": "advice", "advice": {"prob": 0.2679}}]) == []
    assert rr.compare_replies(other, want)
    assert rr.compare_replies([], want)


@pytest.mark.slow
@pytest.mark.parametrize("golden", GOLDENS, ids=lambda p: p.name)
def test_the_advisor_reproduces_its_golden_replies(golden: Path) -> None:
    """The recorded run's replies (``tests/fixtures/advisor_replay``), the
    reference the Rust advisor port is held to (GH #1426). A deliberate
    change to the advice re-records them::

        uv run python python/realgame_advisor_replay.py \\
            ../logs/2026-11-07-instinct-live/run1-requests.jsonl.gz \\
            --out tests/fixtures/advisor_replay/run1-instinct.jsonl.gz
        uv run python python/realgame_advisor_replay.py \\
            ../logs/2026-11-07-instinct-live/run1-requests.jsonl.gz --brain search \\
            --sims 400 --boss-sims 800 --out tests/fixtures/advisor_replay/run1-search-400.jsonl.gz
    """
    diffs = rr.check_golden(golden)
    assert not diffs, "\n".join(diffs)


STS2_RS = Path(__file__).resolve().parents[2] / "sts2-rs"


@pytest.mark.slow
@pytest.mark.parametrize("golden", GOLDENS, ids=lambda p: p.name)
def test_the_native_advisor_reproduces_the_golden_replies(golden: Path) -> None:
    """The Rust ``SimAdvisor`` binary (``sts2-rs/crates/sim-advisor``, GH #1426)
    replays the same corpus without Python and must answer as the Python
    advisor did (``--check`` uses :func:`realgame_advisor_replay.compare_replies`'
    tolerances). Built in the wheel's ``pyext`` profile, whose artifacts the
    gate already has."""
    import shutil

    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("no cargo to build SimAdvisor")
    proc = subprocess.run(
        [
            cargo,
            "run",
            "-q",
            "--profile",
            "pyext",
            "-p",
            "sim-advisor",
            "--bin",
            "SimAdvisor",
            "--",
            "--check",
            str(golden.resolve()),
            "--data-root",
            str(Path(__file__).resolve().parents[1]),
        ],
        cwd=STS2_RS,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "replies match" in proc.stdout


def _build_native_advisor() -> Path:
    import shutil

    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("no cargo to build SimAdvisor")
    subprocess.run(
        [cargo, "build", "-q", "--profile", "pyext", "-p", "sim-advisor", "--bin", "SimAdvisor"],
        cwd=STS2_RS,
        check=True,
        timeout=3600,
    )
    exe = "SimAdvisor.exe" if sys.platform == "win32" else "SimAdvisor"
    return STS2_RS / "target" / "pyext" / exe


@pytest.mark.slow
def test_the_native_advisor_serves_the_mods_socket(tmp_path: Path) -> None:
    """``SimAdvisor`` in serve mode, against a fake mod that plays the live
    corpus over TCP as the mod does (each advice request waits for its
    answer): every advice reply is the search golden's."""
    import gzip
    import socket

    golden = next(g for g in GOLDENS if "search" in g.name)
    meta, want = rr.read_replies(golden)
    opts = meta["options"]
    binary = _build_native_advisor()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    with (tmp_path / "serve.log").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [
                str(binary),
                "--port",
                str(port),
                "--threads",
                "1",
                "--sims",
                str(opts["sims"]),
                "--boss-sims",
                str(opts["boss_sims"]),
                # A plan the clock cuts depends on the load (GH #1784).
                "--plan-budget-s",
                "inf",
                "--log-dir",
                str(tmp_path / "alog"),
                "--workbench-logs",
                str(tmp_path / "wb"),
                "--data-root",
                str(Path(__file__).resolve().parents[1]),
            ],
            stderr=err,
        )
        try:
            srv.settimeout(120)
            conn, _ = srv.accept()
            conn.settimeout(600)
            buf = b""
            advice: list[dict] = []
            with gzip.open(rr.REPO_ROOT / meta["corpus"], "rt", encoding="utf-8") as corpus:
                for line in corpus:
                    msg = json.loads(line)
                    if msg.get("type") == "advice_request":
                        msg.pop("brain", None)
                    conn.sendall(json.dumps(msg).encode() + b"\n")
                    if msg.get("type") != "advice_request":
                        continue
                    while True:
                        while b"\n" not in buf:
                            chunk = conn.recv(65536)
                            assert chunk, "the advisor closed the connection"
                            buf += chunk
                        raw, buf = buf.split(b"\n", 1)
                        reply = json.loads(raw)
                        if reply.get("action") == "advice":
                            assert reply.get("request_id") == msg.get("request_id")
                            advice.append(rr._scrub(reply))
                            break
            conn.close()
        finally:
            proc.terminate()
            proc.wait(30)
            srv.close()
    golden_advice = [r for r in want if r.get("action") == "advice"]
    diffs = rr.compare_replies(json.loads(json.dumps(advice)), golden_advice)
    assert not diffs, "\n".join(diffs)
