"""WP3b: recorded decision states replay to the same points (`ppo.e2e.recorder`)."""

from __future__ import annotations

import gzip
import json

import pytest
import sts2_client
from ppo.e2e import factory
from ppo.e2e import recorder as R


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def test_the_recording_champion_resolves_like_the_workbench(tmp_path, monkeypatch) -> None:
    """No pinned champion filename (`test_champion_resolvers_agree.py`): the
    Ironclad champion `champions.toml` declares, with its features, or the
    `STS2_WB_PPO_ONNX` override, as `workbench.ppo_policy` serves it."""
    import champions
    from workbench import ppo_policy

    monkeypatch.delenv("STS2_WB_PPO_ONNX", raising=False)
    onnx, features = R.champion()
    assert onnx == ppo_policy.onnx_path("Ironclad") == champions.onnx_path("Ironclad")
    assert features == champions.get("Ironclad").features and features
    assert R._shown(onnx) == f"data/ppo/{champions.get('Ironclad').onnx}"
    f = tmp_path / "net.onnx"
    f.write_bytes(b"onnx")
    monkeypatch.setenv("STS2_WB_PPO_ONNX", str(f))
    assert R.champion() == (str(f), None) and R._shown(str(f)) == str(f.resolve())
    monkeypatch.setenv("STS2_WB_PPO_ONNX", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError, match="STS2_WB_PPO_ONNX"):
        R.champion()


def test_record_then_replay_matches_every_fingerprint(tmp_path) -> None:
    run = R.record_run(7, 0.5, 8, 8)
    assert run["kwargs"] == R.RUN_KWARGS
    assert run["kwargs"]["game_version"] == "v0.111.0"
    onnx, features = R.champion()
    assert run["agent"]["onnx"] == R._shown(onnx) and run["agent"]["features"] == features
    assert run["states"] and len(run["states"]) == len(run["fingerprints"])
    path = tmp_path / "s.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")
    (back,) = R.read_runs(path)
    states = list(R.replay(back))
    assert [s.step for s in states] == run["states"]
    combat = [s for s in states if s.phase == "combat"]
    assert combat, "a run reaches a fight"
    for s in combat:
        assert s.cs is not None and s.snap is not None
        assert s.room in R.ROOMS
        assert len(s.snap["ehist"]) == len(s.snap["enemies"])
        assert len(s.actions) > 1


def test_recording_is_deterministic() -> None:
    a = R.record_run(11, 0.3, 8, 8)
    b = R.record_run(11, 0.3, 8, 8)
    assert a["log"] == b["log"] and a["states"] == b["states"]
