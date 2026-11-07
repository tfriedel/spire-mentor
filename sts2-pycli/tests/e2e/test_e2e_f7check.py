"""F7's tool (`python -m ppo.e2e.f7check`, plan section 3.4) on CPU.

One short recorded v0.111.0 A10 run, replayed: the launch-sized student (6 x 256,
the launch fill, the content prior live) passes; a checkpoint passes through
`--ckpt`; a leak into the policy arrays and a no-op re-deal both fail (exit 1);
bad arguments exit 2. The GPU run (S0 on CUDA bf16) is the RUNBOOK's, not a test.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sts2_client
import torch
from ppo.e2e import f7check as F
from ppo.e2e import factory

pytest.importorskip("sts2_rs")

MAX_ROWS = 48


@pytest.fixture(autouse=True)
def _campaign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` selects v0.111.0 for the process: undo that after each test."""
    for name in ("GAME_VERSION", "HEADLESS_BIN", "LIB_DIR"):
        monkeypatch.setattr(sts2_client, name, getattr(sts2_client, name))
    monkeypatch.setenv(sts2_client.GAME_VERSION_ENV, factory.GAME_VERSION)
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@lru_cache(maxsize=1)
def _run() -> dict[str, Any]:
    from ppo.e2e import recorder as R

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        return R.record_run(7, 0.6, 8, 8)


def _shard(tmp_path: Path) -> Path:
    path = tmp_path / "runs.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(_run()) + "\n")
    return path


def _main(tmp_path: Path, *extra: str) -> tuple[int, dict[str, Any] | None]:
    out = tmp_path / "f7.json"
    argv = ["--shards", str(_shard(tmp_path)), "--max-rows", str(MAX_ROWS), "--out", str(out)]
    rc = F.main([*argv, "--threads", "2", *extra])
    return rc, json.loads(out.read_text(encoding="utf-8")) if out.exists() else None


def test_the_launch_student_passes(tmp_path: Path) -> None:
    rc, v = _main(tmp_path)
    assert rc == 0 and v is not None and v["ok"], v and v["problems"]
    assert (v["mismatches"], v["problems"], v["device"], v["dtype"]) == (0, [], "cpu", "fp32")
    assert v["runs"] == 1 and v["rows"] == MAX_ROWS and v["combat_rows"] > 10
    assert v["net"]["cfg"]["n_layers"] == 6 and v["net"]["cfg"]["dim"] == 256
    assert v["net"]["filled"] == list(F.LAUNCH_FILLED) and v["net"]["prior_live"]
    assert set(v["paths"]) == {"cpu_fp32"} and set(v["paths"]["cpu_fp32"]) == set(F.KINDS)
    for kind, r in v["paths"]["cpu_fp32"].items():
        assert r["rows"] == v["counts"][f"{kind}/rows"] > 0 and r["mismatches"] == 0, kind
    from ppo.e2e import stamps as ST

    assert set(ST.STAMP_KEYS) <= set(v["stamps"]) and v["stamps"]["game_version"] == "v0.111.0"
    assert v["control"]["moved"] == v["control"]["rows"] > 0


def test_the_packed_serving_path_passes_and_sees_a_leak(tmp_path: Path, monkeypatch) -> None:
    """`--packed` adds the learner's `--packed-serve` path; it passes on the launch
    student and catches a leak into the policy arrays like the others."""
    for sub in ("ok", "leak"):
        (tmp_path / sub).mkdir()
    rc, v = _main(tmp_path / "ok", "--packed")
    assert rc == 0 and v is not None and v["ok"], v and v["problems"]
    assert set(v["paths"]) == {"cpu_fp32", "packed"}
    for kind, r in v["paths"]["packed"].items():
        assert r["rows"] == v["counts"][f"{kind}/rows"] > 0 and r["mismatches"] == 0, kind
    real = F._enc

    def leaky(d: Any, **kw: Any) -> Any:
        t = real(d, **kw)
        if isinstance(d.rm, F.HiddenRooms):
            t.cont = (t.cont.astype(np.float32) * 1.5).astype(t.cont.dtype)
        return t

    monkeypatch.setattr(F, "_enc", leaky)
    rc, v = _main(tmp_path / "leak", "--packed")
    assert rc == 1 and v is not None and v["paths"]["packed"]["rooms"]["mismatches"] > 0


def test_a_checkpoint_passes_through_ckpt(tmp_path: Path) -> None:
    from ppo.e2e import net as N

    path = tmp_path / "student.pt"
    tiny = {"dim": 32, "n_layers": 1, "n_heads": 2, "ffn": 64, "head_hidden": 32, "num_hidden": 64}
    N.save_checkpoint(path, N.E2ENet(N.NetConfig.from_dict(tiny), seed=3).eval(), stamps={})
    rc, v = _main(tmp_path, "--ckpt", str(path))
    assert rc == 0 and v is not None and v["ok"], v and v["problems"]
    assert v["net"]["source"] == str(path.resolve()) and v["net"]["cfg"]["dim"] == 32
    assert len(v["net"]["ckpt_sha256"]) == 64


def test_a_leak_into_the_policy_arrays_fails(tmp_path: Path, monkeypatch) -> None:
    """A row encoder that lets the hidden room queues reach a policy array."""
    real = F._enc

    def leaky(d: Any, **kw: Any) -> Any:
        t = real(d, **kw)
        if isinstance(d.rm, F.HiddenRooms):
            t.cont = (t.cont.astype(np.float32) * 1.5).astype(t.cont.dtype)
        return t

    monkeypatch.setattr(F, "_enc", leaky)
    rc, v = _main(tmp_path)
    assert rc == 1 and v is not None and not v["ok"]
    assert v["row_mismatches"] == v["counts"]["rooms/row_mismatch"] > 0
    assert v["paths"]["cpu_fp32"]["rooms"]["mismatches"] > 0
    assert all(v["paths"]["cpu_fp32"][k]["mismatches"] == 0 for k in F.KINDS if k != "rooms")


def test_a_priv_read_by_the_policy_tokenizer_fails_with_a_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    """G0 review S2: the most direct leak, the policy part reading a PRIV call, is
    counted and fails the check with a verdict, not a traceback."""
    real = F._enc

    def reads_priv(d: Any, **kw: Any) -> Any:
        if kw.get("priv") is False:
            d.rm.generated_rooms()  # a PRIV call (contracts.PRIV_CALLS)
        return real(d, **kw)

    monkeypatch.setattr(F, "_enc", reads_priv)
    rc, v = _main(tmp_path)
    assert rc == 1 and v is not None and not v["ok"]
    assert v["counts"]["no_priv/priv_read"] == v["rows"] == MAX_ROWS
    assert "no_priv/rows" not in v["counts"]
    assert any("read PRIV" in p for p in v["problems"]), v["problems"]
    assert "rm.generated_rooms" in v["examples"]["no_priv/priv_read"][0][-1]


def test_a_crash_writes_a_verdict_and_exits_3(tmp_path: Path, monkeypatch) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("recorder broke")

    monkeypatch.setattr(F, "make_pairs", boom)
    rc, v = _main(tmp_path)
    assert rc == 3 and v is not None and v["ok"] is False
    assert v["error"] == "RuntimeError: recorder broke" and "boom" in v["traceback"]


def test_a_no_op_re_deal_fails(tmp_path: Path, monkeypatch) -> None:
    """A re-deal that moves nothing passes bit-identity trivially: it must fail."""
    monkeypatch.setattr(F, "world_decision", lambda st, d, seed: (st.cs, d))
    rc, v = _main(tmp_path)
    assert rc == 1 and v is not None and v["mismatches"] == 0
    assert any("no-op re-deal" in p for p in v["problems"]), v["problems"]


@pytest.mark.parametrize(
    "extra",
    [["--ckpt", "no/such.pt"], ["--net", "[1]"], ["--net", "{bad"], ["--runs", "0"]],
)
def test_usage_errors_exit_2(tmp_path: Path, extra: list[str]) -> None:
    out = tmp_path / "f7.json"
    assert F.main(["--out", str(out), *extra]) == 2
    assert not out.exists()


def test_a_learner_state_passes_through_ckpt(tmp_path: Path) -> None:
    """`--ckpt` takes a learner state (state.pt, a pinned state: the net under "net"),
    as at a hot-swap or a 1B pin."""
    import test_learner as TL

    lane = tmp_path / "lane"
    lr = TL.make_learner(TL.make_cfg(lane, max_iters=1))
    assert lr.train() == "max_iters"
    rc, v = _main(tmp_path, "--ckpt", str(lane / "state.pt"))
    assert rc == 0 and v is not None and v["ok"], v and v["problems"]
    assert v["net"]["cfg"]["dim"] == TL.TINY["dim"]


# The e2e lane is Linux-only (systemd units, /dev/shm); the public CI also runs Windows.
pytestmark = pytest.mark.skipif(
    'sys.platform == "win32"',
    reason="f7check drives the learner, which needs POSIX shared memory (see test_learner.py). Linux only.",
)
