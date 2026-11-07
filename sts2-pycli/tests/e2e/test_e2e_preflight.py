"""The pre-flight of a learner run (GH #1097 A7): its `stats.jsonl` rows, and the CLI
that adds the fill check (A4) against the run's declared fill events.

The stats rows here carry exactly the fields `learner.Learner._stats_row` writes that
the pre-flight reads; `test_learner.py::test_the_learners_stats_rows_feed_the_preflight`
runs the same reader on a real learner's rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sts2_client
from ppo.e2e import factory
from ppo.e2e import learner as L
from ppo.e2e import ops_contract as OC
from ppo.e2e import preflight as PF

WP = ["WP-ENG-1"]
FILL = ["--fill-runs", "6"]


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


@pytest.fixture(scope="module")
def stamps() -> dict[str, Any]:
    """This process's stamps as the learner writes them, with WP-ENG-1 declared."""
    return L.make_stamps(filled=WP)


def stats_row(it: int, stamps: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "it": it,
        "rows_type": {t: 2 for t in PF.REQUIRED_DTYPES},
        "outcomes": {"lost": 600, "won": 200, "bootstrap": 200},
        "runs": 800,
        "practice_segments": 200,
        "unknown": 0,
        "f9": {"checked": 40, "hits": 0, "tolerated": 0},
        "error_types": {},
        "stamps": stamps or {},
    }
    row.update(kw)
    return row


def write_run(d: Path, rows: list[dict[str, Any]]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (d / OC.STATS).write_text(text, encoding="utf-8")
    return d


def test_counts_from_stats_sums_the_learner_rows() -> None:
    rows = [
        stats_row(1),
        stats_row(2, outcomes={"lost": 3, "error": 1, "capped": 1}, runs=4, practice_segments=0)
        | {"error_types": {"PanicException": 1}, "unknown": 2, "f9": {"checked": 9, "hits": 1}},
    ]
    c = PF.counts_from_stats(rows)
    assert c["rows/SMITH"] == 4 and c["rows/COMBAT_TURN"] == 4
    assert c["segments"] == 1000 + 5 and c["outcome/capped"] == 1 and c["outcome/error"] == 1
    assert c["segments/run"] == 804 and c["segments/practice"] == 200
    assert c["unknown"] == 2 and c["f9/checks"] == 49 and c["f9/violations"] == 1
    assert c["error/PanicException"] == 1 and not any(k.startswith("fine/") for k in c)
    with pytest.raises(ValueError, match=r"stats row 0 \(it 7\) has no \['outcomes'\]"):
        PF.counts_from_stats([{k: v for k, v in stats_row(7).items() if k != "outcomes"}])


def test_preflight_stats_checks_what_the_rows_carry() -> None:
    pf = PF.preflight_stats([stats_row(1), stats_row(2)])
    assert pf.ok, pf.problems()
    assert pf.missing == [] and pf.unchecked == list(PF.STATS_UNCHECKED)
    assert pf.gaps == ["rows/RELIC_REWARD", "rows/TREASURE"]  # no fine keys in stats rows
    assert pf.dec_per_run["SMITH"] == pytest.approx(4 / 1600)
    assert set(PF.FINE_DTYPE) == set(PF.REQUIRED_FINE)
    assert set(PF.FINE_DTYPE.values()) <= set(PF.REQUIRED_DTYPES)
    no_smith = {t: 2 for t in PF.REQUIRED_DTYPES if t != "SMITH"}
    bad = PF.preflight_stats(
        [
            stats_row(1, rows_type=no_smith, unknown=1),
            stats_row(2, rows_type=no_smith, outcomes={"lost": 10, "capped": 2, "error": 1})
            | {"runs": 12, "error_types": {"SchemaError": 1}, "f9": {"checked": 3, "hits": 1}},
        ]
    )
    probs = bad.problems()
    assert not bad.ok and bad.missing == ["rows/SMITH"]
    assert "unknown = 1" in probs and "1 F9 violation(s)" in probs
    assert "capped 2/1013 = 0.1974% > 0.1%" in probs
    assert "1 segment(s) ended in an error (engine or assert): SchemaError 1" in probs
    # The segment-counter path keeps requiring the fine keys and checks nothing less.
    seg = PF.preflight(PF.counts_from_stats([stats_row(1)]))
    assert not seg.ok and seg.unchecked == []
    assert seg.missing == [f"fine/{k}" for k in PF.REQUIRED_FINE]


def _seg(**kw: int) -> dict[str, int]:
    """A row's `segment_counts` with every fine surface once, plus `kw`."""
    out = {f"fine/{k}": 1 for k in PF.REQUIRED_FINE} | {"forced/combat": 30}
    out.update({k.replace("__", "/"): v for k, v in kw.items()})
    return out


def test_segment_counts_feed_the_rules_the_stats_lacked() -> None:
    """A learner's stats row carries its ended segments' summed counters
    (`segment_counts`): the fine keys, `fine_unknown`, the cap asserts and the forced
    steps are then checked as on the counter path (A7's open issue).
    The row's other fields keep their sources (rows per type from `rows_type`)."""
    rows = [
        stats_row(1, segment_counts=_seg(rows__SMITH=99, outcome__lost=5)),
        stats_row(2, segment_counts=_seg(**{"fine__card-smith": 2})),
    ]
    c = PF.counts_from_stats(rows)
    assert c["fine/card-smith"] == 3 and c["forced/combat"] == 60
    assert c["rows/SMITH"] == 4 and c["outcome/lost"] == 1200  # not doubled
    pf = PF.preflight_stats(rows)
    assert pf.ok and pf.unchecked == [], pf.problems()
    assert pf.combat_auto_share == pytest.approx(60 / (60 + 8))
    bad = PF.preflight_stats(
        [
            stats_row(1, segment_counts=_seg(fine_unknown=1, assert__members=1)),
            stats_row(2, segment_counts={"forced/combat": 1}),
        ]
    )
    probs = bad.problems()
    assert not bad.ok and bad.unchecked == [] and bad.missing == []
    assert "unknown = 1" in probs and "1 decision(s) over the schema's members cap" in probs
    nofine = PF.preflight_stats([stats_row(1, segment_counts={"forced/combat": 1})])
    assert nofine.missing == [f"fine/{k}" for k in PF.REQUIRED_FINE]
    # A row from before (no `segment_counts`) leaves those rules unchecked for the run.
    old = PF.preflight_stats([stats_row(1, segment_counts=_seg()), stats_row(2)])
    assert old.unchecked == list(PF.STATS_UNCHECKED) and old.missing == []


def test_the_cli_passes_a_run_whose_rows_and_fill_match(tmp_path, stamps, capsys) -> None:
    d = write_run(tmp_path / "run", [stats_row(1, stamps), stats_row(2, stamps)])
    out = tmp_path / "report.json"
    assert PF.main(["--run-dir", str(d), *FILL, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.rstrip().endswith("PREFLIGHT PASS"), text
    assert "fill: filled ['WP-ENG-1']:" in text and "problem:" not in text
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["ok"] and rep["declared"] == WP and rep["problems"] == []
    assert rep["fill"]["combat_rows"] >= 100 and rep["stats"]["segments"] == 2000


def test_the_cli_fails_a_fill_the_run_does_not_declare(tmp_path, stamps, capsys) -> None:
    """The getters' rows of a run launched without `--filled '["WP-ENG-1"]'`, or a
    `--filled` that disagrees with the run's stamps."""
    none = dict(stamps, filled=[])
    d = write_run(tmp_path / "run", [stats_row(1, none)])
    assert PF.main(["--run-dir", str(d), *FILL]) == 1
    text = capsys.readouterr().out
    assert "problem: fill: WP-ENG-1 is not declared but" in text
    assert text.rstrip().endswith("PREFLIGHT FAIL")
    d = write_run(tmp_path / "wp", [stats_row(1, stamps)])
    assert PF.main(["--run-dir", str(d), *FILL, "--filled", "[]"]) == 1
    text = capsys.readouterr().out
    assert "problem: --filled [] but the run declares ['WP-ENG-1']" in text


def test_the_cli_refuses_rows_from_another_build_or_source(tmp_path, stamps, capsys) -> None:
    for key, val in (("py_src", "0" * 16), ("schema_sha256", "f" * 64)):
        d = write_run(tmp_path / key, [stats_row(1, dict(stamps, **{key: val}))])
        assert PF.main(["--run-dir", str(d), *FILL]) == 1
        text = capsys.readouterr().out
        assert f"problem: this process is not the run's ({key} {val} -> " in text, text


def test_the_cli_usage_errors(tmp_path, stamps, capsys) -> None:
    assert PF.main(["--run-dir", str(tmp_path / "nothing"), *FILL]) == 2
    d = write_run(tmp_path / "empty", [])
    assert PF.main(["--run-dir", str(d), *FILL]) == 2
    assert "no stats rows yet" in capsys.readouterr().err
    d = write_run(tmp_path / "undeclared", [stats_row(1, {})])
    assert PF.main(["--run-dir", str(d), *FILL]) == 2
    assert "declare no fill events; pass --filled" in capsys.readouterr().err
    for bad in ('"WP-ENG-1"', "[1]", "nope", '["WP-ENG-2"]'):
        d = write_run(tmp_path / "ok", [stats_row(1, stamps)])
        assert PF.main(["--run-dir", str(d), *FILL, "--filled", bad]) == 2, bad
    assert PF.main(["--run-dir", str(d), "--fill-runs", "0"]) == 2
    # A torn last line (a learner killed mid-write) is skipped, a torn earlier one is not.
    good = json.dumps(stats_row(1, stamps))
    (d / OC.STATS).write_text(good + "\n" + good[:40], encoding="utf-8")
    assert len(PF.read_stats(d)) == 1
    (d / OC.STATS).write_text(good[:40] + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        PF.read_stats(d)
