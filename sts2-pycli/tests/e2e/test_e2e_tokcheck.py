"""WP3b: the X-C and X-T harness (`ppo.e2e.tokcheck`) on one short recorded run."""

from __future__ import annotations

import gzip
import json

import pytest
import sts2_client
from ppo.e2e import contracts as CT
from ppo.e2e import factory, tokcheck
from ppo.e2e import recorder as R


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def test_tokcheck_on_a_short_run(tmp_path) -> None:
    pytest.importorskip("torch")
    if not tokcheck.TEACHER.exists():
        pytest.skip(f"needs the combat teacher checkpoint {tokcheck.TEACHER}")
    path = tmp_path / "states.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(R.record_run(7, 0.6, 8, 8)) + "\n")

    out = tokcheck.run_checks([str(path)], tokcheck.TEACHER, 5, 1)
    assert out["n_mismatches"] == 0, out["mismatches"]
    assert out["xc_rows"] >= 25 and out["logit_checks"] >= 1
    assert out["rows"] == sum(out["by_type"].values())
    assert sum(out["rows_per_bucket"].values()) == out["rows"]
    assert list(out["rows_per_bucket"]) == list(CT.TOKEN_BUCKETS)
    assert out["tokens_over_cap"] == 0 and out["padded_per_token"] >= 1.0
    assert out["n_timed"] == 5 and out["ratio_mean"] > 0

    untimed = tokcheck.run_checks([str(path)], tokcheck.TEACHER, 0, 1)
    assert untimed["n_timed"] == 0 and "ratio_mean" not in untimed
    assert untimed["rows_per_bucket"] == out["rows_per_bucket"]
