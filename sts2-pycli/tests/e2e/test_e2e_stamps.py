"""e2e WP1: the stamps every campaign cfg, checkpoint and row carries."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
from pathlib import Path

import pytest
import sts2_client
from ppo import wheel_guard
from ppo.e2e import stamps as S
from sts2_client import GameVersionMismatch

V111 = "v0.111.0"
HEX16 = re.compile(r"[0-9a-f]{16}")


@pytest.fixture(autouse=True)
def _campaign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select v0.111.0, as a campaign process started with
    `STS2_GAME_VERSION=v0.111.0` does."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)


def test_stamps_name_the_code_engine_and_build() -> None:
    got = S.stamps(vocab_sha256="v" * 64, schema_sha256="s" * 64)
    assert tuple(got) == S.STAMP_KEYS
    assert got["game_version"] == V111
    assert got["py_src"] == wheel_guard.python_source_digest()
    assert HEX16.fullmatch(got["py_src"])
    assert got["py_client"] == S.file_digest(Path(sts2_client.__file__))
    assert HEX16.fullmatch(got["py_client"])
    assert got["wheel"] == wheel_guard.wheel_identity()
    assert set(got["wheel"]) == set(wheel_guard.IDENTITY_KEYS)
    assert (got["vocab_sha256"], got["schema_sha256"]) == ("v" * 64, "s" * 64)
    # A row or a cfg stores it as JSON.
    assert json.loads(json.dumps(got)) == got


def test_stamps_without_a_schema_yet() -> None:
    got = S.stamps()
    assert (got["vocab_sha256"], got["schema_sha256"]) == (None, None)


def test_stamps_refuse_a_process_on_another_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.107.1")
    with pytest.raises(GameVersionMismatch, match=re.escape("STS2_GAME_VERSION=v0.111.0")):
        S.stamps()


def test_a_spawned_worker_stamps_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    # A worker inherits the selection through the environment.
    monkeypatch.setenv(sts2_client.GAME_VERSION_ENV, V111)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        child = pool.apply(S.stamps, kwds={"schema_sha256": "s" * 64})
    assert child == S.stamps(schema_sha256="s" * 64)


def test_file_digest_reads_the_name_and_the_bytes(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "sub"
    c.mkdir()
    a.write_text("x = 1\n", encoding="utf-8")
    b.write_text("x = 1\n", encoding="utf-8")
    (c / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert S.file_digest(a) == S.file_digest(c / "a.py")
    assert S.file_digest(a) != S.file_digest(b)
    before = S.file_digest(a)
    a.write_text("x = 2\n", encoding="utf-8")
    assert S.file_digest(a) != before
    data = b"x = 2\n"
    want = hashlib.sha256(b"a.py\0" + len(data).to_bytes(8, "little") + data).hexdigest()[:16]
    assert S.file_digest(a) == want


def test_canonical_sha256_ignores_key_order_only() -> None:
    spec = {"b": 1, "a": [1, 2], "ü": "é"}
    same = {"ü": "é", "a": [1, 2], "b": 1}
    assert S.canonical_sha256(spec) == S.canonical_sha256(same)
    want = hashlib.sha256('{"a":[1,2],"b":1,"ü":"é"}'.encode()).hexdigest()
    assert S.canonical_sha256(spec) == want
    assert S.canonical_sha256({**spec, "a": [2, 1]}) != want
    with pytest.raises(TypeError):
        S.canonical_sha256({"a": {1, 2}})


def test_stamp_change() -> None:
    base = S.stamps(vocab_sha256="v" * 64, schema_sha256="s" * 64)
    assert S.stamp_change(base, dict(base)) is None
    assert (
        S.stamp_change(base, {**base, "py_src": "0" * 16})
        == f"py_src {base['py_src']} -> {'0' * 16}"
    )
    moved = {**base, "wheel": {**base["wheel"], "so_path": "/elsewhere/sts2_rs.so"}}
    assert S.stamp_change(base, moved) is None  # a resumed run pins anew
    rebuilt = {**base, "wheel": {**base["wheel"], "src_hash": "other"}}
    assert S.stamp_change(base, rebuilt) == f"wheel src_hash {base['wheel']['src_hash']} -> other"
    assert S.stamp_change(base, {**base, "game_version": "v0.107.1"}) == (
        "game_version v0.111.0 -> v0.107.1"
    )
    unschema = {k: v for k, v in base.items() if k != "schema_sha256"}
    assert S.stamp_change(base, unschema) == "schema_sha256 missing (before: True, after: False)"
    assert S.stamp_change(None, base) == "missing stamp (before: False, after: True)"
    assert S.stamp_change(base, {**base, "wheel": None}) == f"wheel {base['wheel']} -> None"
