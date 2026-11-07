"""ppo.e2e.wandb_sync: mirrors a lane's JSONL logs into a wandb-like run, resumably."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ppo.e2e import ops_contract as OC
from ppo.e2e import wandb_sync as W


class FakeRun:
    id = "fake-run"

    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []

    def log(self, d: dict[str, float]) -> None:
        self.logged.append(d)


def _append(p: Path, rows: list[dict[str, Any]], partial: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(partial)


def test_flatten_nested_skips_strings_stamps_and_nonfinite() -> None:
    row = {
        "it": 3,
        "decisions": 100,
        "kl": {"combat": 0.1, "ooc": float("nan")},
        "wheel_mismatch": None,
        "stamps": {"py_src": "abc"},
        "nonfinite": 0,
        "caps": {"runs": 4},
    }
    assert W.flatten(row) == {
        "it": 3.0,
        "decisions": 100.0,
        "kl/combat": 0.1,
        "nonfinite": 0.0,
        "caps/runs": 4.0,
    }


def test_sync_tails_stats_throughput_and_evals_once(tmp_path: Path) -> None:
    sd = tmp_path / "lane"
    _append(
        sd / OC.STATS, [{"it": 1, "decisions": 10, "losses": {"policy": 0.5}}], partial='{"it": 2'
    )
    _append(sd / OC.THROUGHPUT, [{"time": 1.0, "pid": 7, "decisions": 12, "rate": 3.0}])
    done = sd / OC.EVAL_DONE
    done.mkdir(parents=True)
    (done / "r1.json").write_text(
        json.dumps(
            {
                "panel": "DEV-1000",
                "variant": "ema0996",
                "state": "/x/pinned_d500000_k1.pt",
                "wins": 50,
                "runs": 1000,
                "rc": 0,
            }
        ),
        encoding="utf-8",
    )
    run = FakeRun()
    s = W.Sync(sd, run)
    assert s.tick() == 3
    assert {"it": 1.0, "decisions": 10.0, "losses/policy": 0.5} in run.logged
    assert {
        "throughput/time": 1.0,
        "throughput/pid": 7.0,
        "throughput/rate": 3.0,
        "decisions": 12.0,
    } in run.logged
    ev = next(d for d in run.logged if "eval/DEV-1000/ema0996/win_rate" in d)
    assert ev["decisions"] == pytest.approx(500000.0)
    assert ev["eval/DEV-1000/ema0996/win_rate"] == pytest.approx(0.05)
    assert ev["eval/ema0996/win_rate"] == pytest.approx(0.05)

    # The partial line completes; nothing is logged twice, and a fresh Sync resumes from the cursor.
    with (sd / OC.STATS).open("a", encoding="utf-8") as f:
        f.write(', "decisions": 20}\n')
    run2 = FakeRun()
    s2 = W.Sync(sd, run2)
    assert s2.tick() == 1
    assert run2.logged == [{"it": 2.0, "decisions": 20.0}]


def test_rewritten_stats_after_a_resume_logs_each_it_once(tmp_path: Path) -> None:
    sd = tmp_path / "lane"
    _append(sd / OC.STATS, [{"it": i, "decisions": i * 10} for i in range(1, 6)])
    run = FakeRun()
    s = W.Sync(sd, run)
    s.tick()
    # A resume from it 3 rewrites stats.jsonl to its/1..3, then the learner replays 4.. and goes on.
    (sd / OC.STATS).write_text("", encoding="utf-8")
    _append(sd / OC.STATS, [{"it": i, "decisions": i * 10} for i in (1, 2, 3, 4)])
    s.tick()
    _append(sd / OC.STATS, [{"it": 5, "decisions": 50}, {"it": 6, "decisions": 60}])
    s.tick()
    its = [d["it"] for d in run.logged if "it" in d]
    assert its == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_variants_get_their_own_series_and_an_old_cursor_relogs_evals(tmp_path: Path) -> None:
    sd = tmp_path / "lane"
    done = sd / OC.EVAL_DONE
    done.mkdir(parents=True)
    for v, w in (("raw", 695), ("ema0996", 862), ("ema0999", 989)):
        rec = {
            "panel": "DEV-2000",
            "variant": v,
            "state": "/x/pinned_d1000_m.pt",
            "wins": w,
            "runs": 2000,
        }
        (done / f"m_{v}.json").write_text(json.dumps(rec), encoding="utf-8")
    # A schema-1 cursor (no eval_schema) that had already logged these under the old keys.
    cur = {"offsets": {}, "max_it": -1, "evals_seen": ["m_raw", "m_ema0996", "m_ema0999"]}
    (sd / W.CURSOR).parent.mkdir(parents=True, exist_ok=True)
    (sd / W.CURSOR).write_text(json.dumps(cur), encoding="utf-8")
    run = FakeRun()
    assert W.Sync(sd, run).tick() == 3
    rates = {
        k: d[k] for d in run.logged for k in d if k.endswith("/win_rate") and k.count("/") == 3
    }
    assert rates == {
        "eval/DEV-2000/raw/win_rate": pytest.approx(695 / 2000),
        "eval/DEV-2000/ema0996/win_rate": pytest.approx(862 / 2000),
        "eval/DEV-2000/ema0999/win_rate": pytest.approx(989 / 2000),
    }
    assert not any("eval/DEV-2000/win_rate" in d for d in run.logged)
    # The new cursor is schema 2: a second Sync logs nothing again.
    run2 = FakeRun()
    assert W.Sync(sd, run2).tick() == 0
