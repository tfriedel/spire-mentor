"""ppo.e2e.stats: the Newcombe port, McNemar, Wilson, the bootstraps and Holm."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from ppo.e2e import stats as S

MONO = Path(__file__).resolve().parents[3]
RELEASE_ANALYZE = (
    MONO / "docs/experiments/2026-10-26-ironclad-release-winrates/analysis/scripts/analyze.py"
)
IT700_ANALYZE = MONO / "docs/experiments/2026-10-26-ic-a10-it700-standard-eval/scripts/analyze.py"

# (a both win, b arm1 only, c arm2 only, d both lose): typical panels, the
# cont300 gate tables, and edge cases (empty margins, tiny n, all one cell).
TABLES = [
    (113, 141, 104, 642),  # cont300 A10 x 1000, cand - champ
    (331, 192, 166, 311),  # cont300 A9-est x 1000
    (143, 82, 60, 115),  # cont300 A9 x 400
    (48, 72, 60, 220),  # cont300 std4x A10 x 400
    (59, 48, 37, 256),  # it700 standard eval A10, D - P
    (154, 55, 45, 146),  # it700 standard eval A9-est
    (500, 300, 120, 3080),  # a TEST-sized Bar1 pass
    (700, 160, 180, 2960),  # a TEST-sized parity read
    (0, 0, 0, 10),
    (10, 0, 0, 0),
    (0, 5, 0, 5),
    (0, 0, 5, 5),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (3, 2, 1, 0),
    (0, 7, 3, 0),
]


def _load_reference(path: Path, names: tuple[str, ...]) -> dict:
    """Exec only the named functions and constants of a reference script.

    The analysis scripts run their CLI setup at import (sys.argv paths), so
    import them by AST, not by module.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    ns: dict = {"math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.skipif(not RELEASE_ANALYZE.is_file(), reason="release-winrates analyze.py missing")
@pytest.mark.parametrize("table", TABLES)
def test_newcombe_port_matches_release_analyze(table):
    ref = _load_reference(RELEASE_ANALYZE, ("Z", "wilson", "mcnemar_exact", "newcombe_paired"))
    assert ref["Z"] == S.Z95
    assert S.newcombe_paired(*table) == ref["newcombe_paired"](*table)
    a, b, c, d = table
    assert S.mcnemar_exact(b, c) == ref["mcnemar_exact"](b, c)
    n = a + b + c + d
    assert S.wilson(a + b, n) == ref["wilson"](a + b, n)


@pytest.mark.skipif(not IT700_ANALYZE.is_file(), reason="it700 standard-eval analyze.py missing")
@pytest.mark.parametrize("table", TABLES[:8])
def test_newcombe_port_matches_it700_analyze(table):
    ref = _load_reference(IT700_ANALYZE, ("Z", "wilson", "mcnemar_exact", "newcombe_paired"))
    assert S.newcombe_paired(*table) == pytest.approx(ref["newcombe_paired"](*table), abs=1e-15)


def test_newcombe_brackets_the_difference():
    for table in TABLES:
        if sum(table) == 0:
            continue
        t = S.PairedTable(*table)
        lo, hi = S.newcombe_paired(*table)
        assert lo <= t.diff + 1e-12 and t.diff - 1e-12 <= hi
        assert lo >= -1.0 - 1e-12 and hi <= 1.0 + 1e-12


def test_mcnemar_known_values():
    assert S.mcnemar_exact(0, 0) == pytest.approx(1.0)
    assert S.mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert S.mcnemar_exact(104, 141) == S.mcnemar_exact(141, 104)
    # Published by the cont300 and it700 gates (a10_score.py, same formula).
    assert f"{S.mcnemar_exact(104, 141):.3g}" == "0.0213"
    assert f"{S.mcnemar_exact(63, 132):.2g}" == "8.7e-07"
    assert f"{S.mcnemar_exact(37, 48):.3g}" == "0.278"
    from scipy.stats import binomtest

    for b, c in ((104, 141), (60, 82), (0, 9), (13, 2)):
        assert S.mcnemar_exact(b, c) == pytest.approx(binomtest(b, b + c, 0.5).pvalue, rel=1e-9)


def test_wilson_matches_published_it700_interval():
    # docs/experiments/2026-10-26-ic-a10-it700-standard-eval/analysis/primary.json, arm D A10.
    assert S.wilson(107, 400) == (0.22648238411831478, 0.31294083274713314)
    assert all(math.isnan(x) for x in S.wilson(0, 0))


def test_paired_table_from_pairs():
    pairs = [(True, True)] * 3 + [(True, False)] * 2 + [(False, True)] + [(False, False)] * 4
    t = S.PairedTable.from_pairs(pairs)
    assert (t.a, t.b, t.c, t.d) == (3, 2, 1, 4)
    assert t.n == 10
    assert (t.p1, t.p2) == pytest.approx((0.5, 0.4))
    assert t.diff == pytest.approx(0.1) and t.discordance == pytest.approx(0.3)


def _pairs(a, b, c, d):
    return [(True, True)] * a + [(True, False)] * b + [(False, True)] * c + [(False, False)] * d


def test_paired_bootstrap_is_deterministic_and_chunk_free(monkeypatch):
    pairs = _pairs(59, 48, 37, 256)
    one = S.paired_bootstrap(pairs, resamples=2000)
    assert one == S.paired_bootstrap(pairs, resamples=2000)
    monkeypatch.setattr(S, "_CHUNK", 7)
    assert S.paired_bootstrap(pairs, resamples=2000) == one
    # Brackets the point estimate; SE near the analytic paired SE.
    diff = (48 - 37) / 400
    assert one["lo"] < diff < one["hi"]
    se = math.sqrt(((48 + 37) / 400 - diff**2) / 400)
    assert one["se"] == pytest.approx(se, rel=0.1)


def test_paired_bootstrap_default_is_prereg():
    out = S.paired_bootstrap(_pairs(5, 3, 1, 11), resamples=100)
    assert out["seed"] == 20261109 == S.BOOT_SEED
    assert S.BOOT_RESAMPLES == 20_000


def test_paired_bootstrap_agrees_with_newcombe_on_a_test_sized_table():
    pairs = _pairs(500, 300, 120, 3080)
    boot = S.paired_bootstrap(pairs)
    lo, hi = S.newcombe_paired(500, 300, 120, 3080)
    assert boot["lo"] == pytest.approx(lo, abs=0.003)
    assert boot["hi"] == pytest.approx(hi, abs=0.003)


def test_cluster_bootstrap_basics():
    zero = S.cluster_bootstrap([0.0] * 50, [4] * 50, resamples=500)
    assert (zero["mean"], zero["p"]) == pytest.approx((0.0, 1.0))
    rng = np.random.default_rng(1)
    sums = rng.integers(1, 4, size=100).astype(float)  # every state gains
    strong = S.cluster_bootstrap(sums, [4] * 100, resamples=2000)
    assert strong["mean"] > 0 and strong["lo"] > 0 and strong["p"] < 0.01
    assert strong == S.cluster_bootstrap(sums, [4] * 100, resamples=2000)
    empty = S.cluster_bootstrap([], [])
    assert math.isnan(empty["mean"])


def test_cluster_bootstrap_weights_by_pairs():
    # Two clusters: one with 1 pair at +1, one with 3 pairs at 0 -> mean 0.25.
    out = S.cluster_bootstrap([1.0, 0.0], [1, 3], resamples=200)
    assert out["mean"] == pytest.approx(0.25)


def test_holm_step_down():
    out = S.holm({"w": 0.01, "x": 0.04, "y": 0.03, "z": 0.005})
    assert [out[k]["p_holm"] for k in "zwyx"] == pytest.approx([0.02, 0.03, 0.06, 0.06])
    assert [out[k]["reject"] for k in "wxyz"] == [True, False, False, True]
    assert list(out) == ["w", "x", "y", "z"]


def test_holm_counts_missing_members():
    out = S.holm({"a": 0.02, "b": None, "c": None, "d": None})
    assert out["a"]["p_holm"] == pytest.approx(0.08) and not out["a"]["reject"]
    assert out["b"]["p_holm"] == pytest.approx(1.0) and not out["b"]["reject"]
    alone = S.holm({"a": 0.02})
    assert alone["a"]["reject"]
