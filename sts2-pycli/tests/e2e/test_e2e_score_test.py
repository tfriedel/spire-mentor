"""ppo.e2e.score_test: the PREREG-0 P2 read on synthetic panels and on recorded runs.

The synthetic panels are written in `scripts/e2e_arm.sh`'s layout (250-seed
shards, retry files, meta/config.txt, the sha256sum manifest) under tmp_path;
no test reads `~/runs/e2e/evals/`.
"""

import datetime as dt
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from ppo.e2e import panels as P
from ppo.e2e import score_test as T
from ppo.e2e import stats as S

MONO = Path(__file__).resolve().parents[3]
IT700_STD = MONO / "docs/experiments/2026-10-26-ic-a10-it700-standard-eval"
IT700_GATE = Path.home() / "ppo-ironclad-a10" / "gate"
SMOKE = P.RUNNER_PANELS["SMOKE"]
WHEEL = {
    "pgo": True,
    "pgo_instrumented": False,
    "profdata_sha256": "ab" * 32,
    "target_cpu": None,
    "src_hash": "acc813aa12f20ad2",
    "so_path": "<pin>/sts2_rs/sts2_rs.abi3.so",
}


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    monkeypatch.setattr(T, "BOOT_RESAMPLES", 400)


def row(seed, won=False, *, outcome=None, a9=None, act1="overgrowth", wheel=WHEEL, **kw):
    """A probe-shaped row; error and timeout rows carry what play_batch writes."""
    if outcome in T.MISSING_OUTCOMES:
        return {"seed": seed, "policy": "net_greedy", "outcome": outcome, "won": False, **kw}
    r = {
        "seed": seed,
        "policy": "net_greedy",
        "ascension": 10,
        "character": "Ironclad",
        "act1": act1,
        "outcome": outcome or ("won" if won else "lost"),
        "won": bool(won),
        "n_decisions": 100,
        "reached_second_boss": a9,
        "wheel": dict(wheel),
    }
    r.update(kw)
    return r


def write_manifest(evals, panel_name, arm):
    d = evals / panel_name / arm
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {sub}/{p.name}\n"
        for sub in ("shards", "retry")
        for p in sorted((d / sub).glob("*.jsonl"))
    ]
    (evals / panel_name / f"{arm}.sha256").write_text("".join(lines), encoding="utf-8")


def write_arm(evals, panel, arm, rows, *, start=None, n=None, retries=(), manifest=True):
    """Write one row per seed of [start, start + n) in e2e_arm.sh's layout."""
    start = panel.start if start is None else start
    n = panel.n if n is None else n
    d = evals / panel.name / arm
    for sub in ("shards", "meta", "retry", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    by_seed = {r["seed"]: r for r in rows}
    for lo in range(start, start + n, P.SHARD_SIZE):
        hi = min(lo + P.SHARD_SIZE, start + n)
        with open(d / "shards" / f"s{lo}_{hi}.jsonl", "w", encoding="utf-8") as f:
            for s in range(lo, hi):
                if s in by_seed:
                    f.write(json.dumps(by_seed[s]) + "\n")
    for r in retries:
        (d / "retry" / f"s{r['seed']}.jsonl").write_text(json.dumps(r) + "\n", encoding="utf-8")
    config = f"arm={arm}\npanel={panel.name}\nn={n}\nseed_start={start}\nsha=0\nenv=\n"
    (d / "meta" / "config.txt").write_text(config, encoding="utf-8")
    if manifest:
        write_manifest(evals, panel.name, arm)
    return d


def realize(final_wins, a, b, c, d):
    """Baseline wins that make the paired table (a, b, c, d) against `final_wins`."""
    assert len(final_wins) == a + b + c + d and sum(final_wins) == a + b
    left = {"a": a, "b": b, "c": c, "d": d}
    out = []
    for w in final_wins:
        key = ("a" if left["a"] else "b") if w else ("c" if left["c"] else "d")
        left[key] -= 1
        out.append(key in ("a", "c"))
    return out


def spread(n, k, seed=0):
    """`k` wins among `n` seeds, scattered (so no shard holds them all)."""
    idx = set(random.Random(seed).sample(range(n), k))
    return [i in idx for i in range(n)]


def build(evals, panel, tables, final_wins, *, final="F"):
    """Write Final and one baseline per (name, table) entry of `tables`."""
    seeds = list(panel.seeds())
    write_arm(evals, panel, final, [row(s, w) for s, w in zip(seeds, final_wins)])
    for name, table in tables.items():
        base = realize(final_wins, *table)
        write_arm(evals, panel, name, [row(s, w) for s, w in zip(seeds, base)])


def prereg_file(tmp_path):
    p = tmp_path / "prereg" / "PREREG.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# PREREG (test)\n", encoding="utf-8")
    return p


def read(tmp_path, *, panel="SMOKE", final="F", cstar="R2", r1="R1", r2="R2", r3=None, **kw):
    return T.score(
        prereg=kw.pop("prereg", None) or prereg_file(tmp_path),
        evals=kw.pop("evals", None) or tmp_path / "evals",
        panel=panel,
        final=final,
        cstar=cstar,
        r1=r1,
        r2=r2,
        r3=r3,
        **kw,
    )


def table_of(c):
    t = c.table
    return (t.a, t.b, t.c, t.d)


# ── the bars ──────────────────────────────────────────────────────────────


def test_cont300_gate_table_through_the_scorer(tmp_path):
    # docs/experiments/2026-10-31-ironclad-a10-cont300.md: A10 x 1000, cand - champ
    # (113, 141, 104, 642) = +3.70pp, McNemar p .021. Its gate files are gone, so the
    # table is rebuilt from the published counts.
    fw = spread(1000, 254)
    build(tmp_path / "evals", SMOKE, {"R2": (113, 141, 104, 642), "R1": (200, 54, 50, 696)}, fw)
    out = read(tmp_path)
    bar1 = out.contrasts["bar1"]
    assert table_of(bar1) == (113, 141, 104, 642)
    assert bar1.delta * 100 == pytest.approx(3.70)
    assert f"{bar1.p:.3g}" == "0.0213"
    assert bar1.to_dict()["newcombe95"] == list(S.newcombe_paired(113, 141, 104, 642))
    assert bar1.to_dict()["discordance"] == pytest.approx(0.245)
    assert [b["status"] for b in out.bars] == ["FAIL", "NOT TESTED", "NOT TESTED"]
    assert out.barr["A10"]["wins"] == 254 and out.barr["A10"]["rate"] == pytest.approx(0.254)


def test_all_bars_pass(tmp_path):
    fw = spread(1000, 420)
    build(tmp_path / "evals", SMOKE, {"R2": (300, 120, 60, 520), "R1": (350, 70, 20, 560)}, fw)
    out = read(tmp_path)
    assert [b["status"] for b in out.bars] == ["PASS", "PASS", "PASS"]
    assert out.barr["lines"] == {">= 40%": True, ">= 56%": False}
    assert out.barr["A10"]["wilson95"] == list(S.wilson(420, 1000))


def test_bar1_needs_a_full_five_points(tmp_path):
    fw = spread(1000, 400)
    build(tmp_path / "evals", SMOKE, {"R2": (300, 100, 50, 550), "R1": (380, 20, 20, 580)}, fw)
    out = read(tmp_path)
    assert out.contrasts["bar1"].delta == T.BAR1_MIN_DELTA  # exactly +5pp, p ~ 5e-5
    assert out.bars[0]["status"] == "PASS"
    # Bar2 passes (Δ 0, CI low end above −3pp); Bar3 fails (Δ < +3pp).
    assert [b["status"] for b in out.bars[1:]] == ["PASS", "FAIL"]

    shutil.rmtree(tmp_path / "evals")
    build(tmp_path / "evals", SMOKE, {"R2": (300, 100, 51, 549), "R1": (380, 20, 20, 580)}, fw)
    out = read(tmp_path)
    assert out.contrasts["bar1"].delta * 1000 == 49
    assert [b["status"] for b in out.bars] == ["FAIL", "NOT TESTED", "NOT TESTED"]


def test_bar2_failure_stops_the_sequence(tmp_path):
    fw = spread(1000, 400)
    build(tmp_path / "evals", SMOKE, {"R2": (300, 100, 40, 560), "R1": (300, 100, 150, 450)}, fw)
    out = read(tmp_path)
    assert [b["status"] for b in out.bars] == ["PASS", "FAIL", "NOT TESTED"]


# ── P2's pairing and error rules ──────────────────────────────────────────


def _error_panel(evals, final_bad, base_bad, both_bad, *, kind="error"):
    """Final and R2 with Final-only, baseline-only and two-sided missing rows."""
    seeds = list(SMOKE.seeds())
    fw = [i % 4 == 0 for i in range(1000)]
    bw = [i % 5 == 0 for i in range(1000)]
    f_rows = [row(s, w) for s, w in zip(seeds, fw)]
    b_rows = [row(s, w) for s, w in zip(seeds, bw)]
    for i in final_bad + both_bad:
        f_rows[i] = row(seeds[i], outcome=kind)
    for i in base_bad + both_bad:
        b_rows[i] = row(seeds[i], outcome=kind)
    write_arm(evals, SMOKE, "F", f_rows)
    write_arm(evals, SMOKE, "R2", b_rows)
    write_arm(evals, SMOKE, "R1", b_rows)
    expect = []
    for i in range(1000):
        if i in base_bad or i in both_bad:
            continue
        expect.append((False if i in final_bad else fw[i], bw[i]))
    return S.PairedTable.from_pairs(expect)


@pytest.mark.parametrize("kind", ["error", "timeout"])
def test_missing_rows_follow_p2(tmp_path, kind):
    # SMOKE has 1,000 seeds, so 5 missing-data seeds are allowed and 6 are not.
    want = _error_panel(tmp_path / "evals", [0, 4], [1, 2], [3], kind=kind)
    out = read(tmp_path)
    c = out.contrasts["bar1"]
    assert c.valid and table_of(c) == (want.a, want.b, want.c, want.d)
    assert (c.dropped_base_missing, c.final_missing_as_loss, c.max_bad) == (3, 2, 5)
    assert c.seeds == 1000 and c.table.n == 997


def test_one_seed_too_many_is_invalid(tmp_path):
    _error_panel(tmp_path / "evals", [0, 4, 8], [1, 2], [3])
    out = read(tmp_path)
    c = out.contrasts["bar1"]
    assert not c.valid and "6 seeds with missing data" in c.invalid[0]
    assert out.bars[0]["status"] == "INVALID"
    assert [b["status"] for b in out.bars[1:]] == ["NOT TESTED", "NOT TESTED"]


def test_retry_replaces_the_errored_row(tmp_path):
    evals = tmp_path / "evals"
    seeds = list(SMOKE.seeds())
    f_rows = [row(s, False) for s in seeds]
    f_rows[10] = row(seeds[10], outcome="error")
    f_rows[11] = row(seeds[11], outcome="error")
    f_rows[12] = row(seeds[12], outcome="error")
    retries = [row(seeds[10], True), row(seeds[11], outcome="error")]
    write_arm(evals, SMOKE, "F", f_rows, retries=retries)
    # An empty retry file: the retry produced no row, so the error stands.
    (evals / "SMOKE" / "F" / "retry" / f"s{seeds[12]}.jsonl").write_text("", encoding="utf-8")
    write_manifest(evals, "SMOKE", "F")
    for name in ("R1", "R2"):
        write_arm(evals, SMOKE, name, [row(s, False) for s in seeds])
    out = read(tmp_path)
    c = out.contrasts["bar1"]
    assert table_of(c) == (0, 1, 0, 999)
    assert c.final_missing_as_loss == 2
    counts = out.arms["F"]["counts"]
    # Seed 11's retry ran and errored again; seed 12's wrote nothing.
    assert (counts["errored_in_shards"], counts["retried"], counts["retry_without_row"]) == (
        3,
        2,
        1,
    )
    assert counts["missing_after_retry"] == 2


def test_engine_build_and_act1_mismatch_make_a_contrast_invalid(tmp_path):
    evals = tmp_path / "evals"
    seeds = list(SMOKE.seeds())
    write_arm(evals, SMOKE, "F", [row(s, s % 3 == 0) for s in seeds])
    other = dict(WHEEL, src_hash="0000000000000000")
    write_arm(evals, SMOKE, "R2", [row(s, s % 4 == 0, wheel=other) for s in seeds])
    act1 = [row(s, s % 4 == 0, act1="underdocks" if s == seeds[7] else "overgrowth") for s in seeds]
    write_arm(evals, SMOKE, "R1", act1)
    out = read(tmp_path)
    assert "engine builds differ" in out.contrasts["bar1"].invalid[0]
    assert out.contrasts["r1"].invalid == ["the act-1 model differs on 1 seed(s)"]
    assert out.bars[0]["status"] == "INVALID"


def test_a9est_and_the_descriptive_family(tmp_path):
    evals = tmp_path / "evals"
    seeds = list(SMOKE.seeds())
    f_rows = [row(s, i % 4 == 0, a9=i % 2 == 0) for i, s in enumerate(seeds)]
    b_rows = [row(s, i % 5 == 0, a9=i % 3 == 0) for i, s in enumerate(seeds)]
    write_arm(evals, SMOKE, "F", f_rows)
    write_arm(evals, SMOKE, "R2", b_rows)
    write_arm(evals, SMOKE, "R1", b_rows)
    out = read(tmp_path)
    a9 = out.contrasts["r1_a9"]
    want = S.PairedTable.from_pairs(
        [(i % 4 == 0 or i % 2 == 0, i % 5 == 0 or i % 3 == 0) for i in range(1000)]
    )
    assert table_of(a9) == (want.a, want.b, want.c, want.d)
    fam = out.family
    assert list(fam) == ["Final - R2", "Final - R3", "A9-est Final - R1", "DEVBOSS Final - REF"]
    assert fam["A9-est Final - R1"]["p"] == a9.p
    assert fam["Final - R2"]["p"] == out.contrasts["r2"].p
    assert fam["Final - R3"]["p"] is None
    assert fam["DEVBOSS Final - REF"]["p_holm"] == pytest.approx(1.0)
    assert out.arms["F"]["A9-est"]["wins"] == 500
    text = T.render(out)
    assert "Bar1 (must)" in text and "A9-est Final - R1" in text and "Holm" in text


# ── integrity: every failure stops the read before any statistic ──────────


def _ok_layout(evals):
    seeds = list(SMOKE.seeds())
    f_rows = [row(s, s % 3 == 0) for s in seeds]
    f_rows[5] = row(seeds[5], outcome="error")
    for name, rows in (("F", f_rows), ("R2", [row(s, s % 4 == 0) for s in seeds])):
        write_arm(evals, SMOKE, name, rows)
    write_arm(evals, SMOKE, "R1", [row(s, s % 5 == 0) for s in seeds])
    return evals / "SMOKE" / "F"


def _rewrite(path, fn):
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("".join(x + "\n" for x in fn(lines)), encoding="utf-8")


SHARD0 = "s162000000_162000250.jsonl"


def _edit_row(i, **kw):
    def fn(lines):
        r = json.loads(lines[i])
        r.update(kw)
        lines[i] = json.dumps(r)
        return lines

    return fn


MUTATIONS = {
    "a missing row": (lambda d: _rewrite(d / "shards" / SHARD0, lambda x: x[1:]), "missing"),
    "an off-panel seed": (
        lambda d: _rewrite(d / "shards" / SHARD0, _edit_row(0, seed=999)),
        "off-panel",
    ),
    "a duplicate row": (
        lambda d: _rewrite(d / "shards" / SHARD0, lambda x: [*x[:-1], x[0]]),
        "duplicate",
    ),
    "an unparsable line": (
        lambda d: _rewrite(d / "shards" / SHARD0, lambda x: [*x, "{oops"]),
        "not JSON",
    ),
    "a stray file in shards/": (
        lambda d: (d / "shards" / f"{SHARD0}.part").write_text("", encoding="utf-8"),
        "not a shard file",
    ),
    "a retry for a seed that did not error": (
        lambda d: (d / "retry" / "s162000001.jsonl").write_text(
            json.dumps(row(162000001, True)) + "\n", encoding="utf-8"
        ),
        "did not error",
    ),
    "a retry with two rows": (
        lambda d: (d / "retry" / "s162000005.jsonl").write_text(
            (json.dumps(row(162000005, True)) + "\n") * 2, encoding="utf-8"
        ),
        "at most one row",
    ),
    "two engine builds in one arm": (
        lambda d: _rewrite(d / "shards" / SHARD0, _edit_row(0, wheel=dict(WHEEL, pgo=False))),
        "engine builds",
    ),
    "a Silent row": (
        lambda d: _rewrite(d / "shards" / SHARD0, _edit_row(0, character="Silent")),
        "not Ironclad A10",
    ),
    "an A9 row": (
        lambda d: _rewrite(d / "shards" / SHARD0, _edit_row(0, ascension=9)),
        "not Ironclad A10",
    ),
    "a config for another arm": (
        lambda d: _rewrite(d / "meta" / "config.txt", lambda x: ["arm=G", *x[1:]]),
        "is not F",
    ),
}


@pytest.mark.parametrize("what", list(MUTATIONS))
def test_integrity_failures_stop_the_read(tmp_path, what):
    d = _ok_layout(tmp_path / "evals")
    mutate, msg = MUTATIONS[what]
    mutate(d)
    write_manifest(tmp_path / "evals", "SMOKE", "F")  # a consistent manifest: the rows are wrong
    with pytest.raises(T.ReadStopped, match=msg):
        read(tmp_path)


def test_a_file_changed_after_its_manifest_stops_the_read(tmp_path):
    d = _ok_layout(tmp_path / "evals")
    _rewrite(d / "shards" / SHARD0, _edit_row(4, won=True, outcome="won"))
    with pytest.raises(T.ReadStopped, match="sha256 differs"):
        read(tmp_path)
    (tmp_path / "evals" / "SMOKE" / "F.sha256").unlink()
    with pytest.raises(T.ReadStopped, match="missing"):
        read(tmp_path)


def test_manifest_listing_a_missing_file_stops_the_read(tmp_path):
    _ok_layout(tmp_path / "evals")
    m = tmp_path / "evals" / "SMOKE" / "F.sha256"
    extra = f"{'0' * 64}  retry/s162000005.jsonl\n"
    m.write_text(m.read_text(encoding="utf-8") + extra, encoding="utf-8")
    with pytest.raises(T.ReadStopped, match="not on disk"):
        read(tmp_path)


def test_cli_prints_no_statistic_when_stopped(tmp_path, capsys):
    d = _ok_layout(tmp_path / "evals")
    _rewrite(d / "shards" / SHARD0, lambda x: x[1:])
    argv = ["--prereg", str(prereg_file(tmp_path)), "--evals", str(tmp_path / "evals")]
    argv += ["--panel", "SMOKE", "--final", "F", "--cstar", "R2", "--r1", "R1", "--r2", "R2"]
    assert T.main(argv) == 2
    out, err = capsys.readouterr()
    assert out == "" and "READ STOPPED" in err and "%" not in err


def test_cli_writes_json(tmp_path, capsys):
    _ok_layout(tmp_path / "evals")
    js = tmp_path / "read.json"
    argv = ["--prereg", str(prereg_file(tmp_path)), "--evals", str(tmp_path / "evals")]
    argv += ["--panel", "SMOKE", "--final", "F", "--cstar", "R2", "--r1", "R1", "--r2", "R2"]
    assert T.main([*argv, "--json", str(js)]) == 0
    got = json.loads(js.read_text(encoding="utf-8"))
    assert got["panel"] == "SMOKE" and got["bars"][0]["bar"] == "Bar1 (must)"
    assert got["contrasts"]["bar1"]["final_missing_counted_as_loss"] == 1
    assert "Bar1 (must)" in capsys.readouterr().out


def test_arguments(tmp_path):
    _ok_layout(tmp_path / "evals")
    with pytest.raises(T.ReadStopped, match="C\\* is R2 or R3"):
        read(tmp_path, cstar="R1")
    with pytest.raises(T.ReadStopped, match="distinct"):
        read(tmp_path, final="R1")
    with pytest.raises(T.ReadStopped, match="no shards"):
        read(tmp_path, r3="R3m", cstar="R3m")


# ── the sealed rules (TEST, rehearsed on SMOKE_SEALED) ────────────────────


def _git(repo, *args):
    cmd = ["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t"]
    cmd += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args]
    subprocess.run(cmd, check=True, capture_output=True)


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def committed_prereg(tmp_path):
    repo = tmp_path / "repo"
    prereg = repo / "exp" / "PREREG.md"
    prereg.parent.mkdir(parents=True)
    prereg.write_text("# PREREG (test)\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg")
    return repo, prereg


@needs_git
def test_sealed_rules(tmp_path):
    repo, prereg = committed_prereg(tmp_path)
    evals = tmp_path / "evals"
    panel = P.RUNNER_PANELS["SMOKE_SEALED"]
    fw = spread(1000, 300)
    build(evals, panel, {"R2": (250, 50, 30, 670), "R1": (280, 20, 20, 680)}, fw)
    kw = {"panel": "SMOKE_SEALED", "prereg": prereg, "evals": evals}

    with pytest.raises(T.ReadStopped, match="no committed manifest"):
        read(tmp_path, **kw)
    committed = prereg.parent / "sha256" / "SMOKE_SEALED"
    committed.mkdir(parents=True)
    for arm in ("F", "R1", "R2"):
        shutil.copy(evals / "SMOKE_SEALED" / f"{arm}.sha256", committed / f"{arm}.sha256")
    with pytest.raises(T.ReadStopped, match="not committed"):
        read(tmp_path, **kw)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "manifests")

    prereg.write_text("# PREREG (edited)\n", encoding="utf-8")
    with pytest.raises(T.ReadStopped, match="committed, unmodified PREREG"):
        read(tmp_path, **kw)
    _git(repo, "checkout", "--", "exp/PREREG.md")

    log = evals / "SMOKE_SEALED" / T.READ_LOG
    out = read(tmp_path, **kw)
    assert out.bars[0]["status"] == "FAIL"  # Δ +2pp
    assert out.prereg["git"]["clean"] and out.manifests["F"]["committed"].endswith("F.sha256")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(T.ReadStopped, match="read 1 time"):
        read(tmp_path, **kw)
    read(tmp_path, reread_reason="the first read crashed while printing", **kw)
    entries = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [e["reread_reason"] for e in entries] == [None, "the first read crashed while printing"]
    assert entries[0]["manifests"]["F"] == T.sha256_file(evals / "SMOKE_SEALED" / "F.sha256")

    # A committed manifest that no longer matches the files stops the read.
    _rewrite(evals / "SMOKE_SEALED" / "R1" / "shards" / SHARD0, _edit_row(0, steps=1))
    write_manifest(evals, "SMOKE_SEALED", "R1")
    with pytest.raises(T.ReadStopped, match="no committed manifest equals"):
        read(tmp_path, reread_reason="x", **kw)


@needs_git
def test_sealed_panel_needs_every_seed(tmp_path):
    _, prereg = committed_prereg(tmp_path)
    evals = tmp_path / "evals"
    panel = P.RUNNER_PANELS["SMOKE_SEALED"]
    seeds = list(panel.seeds())[:999]
    write_arm(evals, panel, "F", [row(s, True) for s in seeds], n=999)
    with pytest.raises(T.ReadStopped, match="needs all 1000 seeds"):
        read(tmp_path, panel="SMOKE_SEALED", evals=evals, prereg=prereg)


def test_test_panel_is_time_locked(tmp_path, monkeypatch):
    before = P.UNSEAL_AT - dt.timedelta(seconds=1)
    monkeypatch.setattr(T, "_now", lambda: before)
    # The lock fires before the scorer looks at any file.
    with pytest.raises(T.ReadStopped, match="sealed until 2026-11-12T08:00:00"):
        read(tmp_path, panel="TEST", evals=tmp_path / "does-not-exist")
    monkeypatch.setattr(T, "_now", lambda: P.UNSEAL_AT)
    with pytest.raises(T.ReadStopped, match="committed, unmodified PREREG"):
        read(tmp_path, panel="TEST", evals=tmp_path / "does-not-exist")


# ── recorded runs ─────────────────────────────────────────────────────────


def _layout_from_jsonl(evals, panel, arm, path):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    write_arm(evals, panel, arm, rows)


@pytest.mark.skipif(
    not (IT700_STD / "analysis" / "primary.json").is_file(), reason="it700 standard eval missing"
)
def test_it700_standard_eval_reproduces_primary_json(tmp_path):
    """D - P on the committed it700 standard-eval rows equals its published analysis."""
    evals = tmp_path / "evals"
    panel = P.Panel("IT700", 91_000_000, 400)
    for arm in ("D", "P"):
        _layout_from_jsonl(evals, panel, arm, IT700_STD / "runs" / f"{arm}.jsonl")
    d = T.load_arm(evals, panel, "D", sealed_rules=True)
    p = T.load_arm(evals, panel, "P", sealed_rules=True)
    pub = json.loads((IT700_STD / "analysis" / "primary.json").read_text(encoding="utf-8"))
    paired = pub["paired_D_minus_P"]
    for metric, key in (("A10", "A10"), ("A9-est", "A9est")):
        c = T.pair(d, p, metric)
        want = paired[key]
        assert table_of(c) == (
            want["both_win"],
            want["b_D_only"],
            want["c_P_only"],
            want["both_lose"],
        )
        assert c.valid and c.p == want["mcnemar_exact_p_two_sided"]
        assert c.table.diff * 100 == pytest.approx(want["diff_pp"])
        got = [100 * x for x in S.newcombe_paired(*table_of(c))]
        assert got == pytest.approx(want["ci95_pp_newcombe10"], abs=1e-12)
    for arm, a in (("D", d), ("P", p)):
        summary = T.arm_summary(a)
        assert summary["A10"]["wilson_scored"] == tuple(pub["arms"][arm]["A10"]["wilson95"])
        assert summary["A9-est"]["wins"] == pub["arms"][arm]["A9est"]["reached"]
    # And the playbook scorer's text (a10_score.py P -> D).
    text = (IT700_STD / "analysis" / "a10_score.txt").read_text(encoding="utf-8")
    m = re.search(
        r"A10 +champ ([\d.]+)% -> cand ([\d.]+)% +delta ([+\d.]+)pp +disc (\d+)/(\d+)", text
    )
    c = T.pair(d, p, "A10")
    assert m and (int(m.group(4)), int(m.group(5))) == (c.table.c, c.table.b)
    assert f"{100 * c.table.p2:.1f}" == m.group(1) and f"{100 * c.table.p1:.1f}" == m.group(2)
    assert f"{100 * c.table.diff:+.2f}" == m.group(3)


def _arm_from_rows(name, rows):
    rows = {r["seed"]: r for r in rows}
    wheels = {T._wheel_identity(r) for r in rows.values()}
    return T.Arm(
        name=name,
        panel=P.Panel(name, min(rows), len(rows)),
        start=min(rows),
        n=len(rows),
        rows=rows,
        retried=[],
        config={},
        wheel=next(iter(wheels)),
        manifest=Path(),
        manifest_sha256="",
        files=0,
    )


@pytest.mark.skipif(not (IT700_GATE / "gate.log").is_file(), reason="it700 gate files missing")
def test_it700_gate_files_reproduce_gate_log(tmp_path):
    """The it700 A10-only gate (2026-10-23) through the scorer's pairing equals gate.log."""
    evals = tmp_path / "evals"
    panel = P.Panel("GATE", 131_000_000, 1000)
    for arm, f in (("cand", "cand_A10.jsonl"), ("champ", "champ_A10.jsonl")):
        _layout_from_jsonl(evals, panel, arm, IT700_GATE / f)
    cand = T.load_arm(evals, panel, "cand", sealed_rules=True)
    champ = T.load_arm(evals, panel, "champ", sealed_rules=True)
    log = (IT700_GATE / "gate.log").read_text(encoding="utf-8")
    rx = r"{} +champ ([\d.]+)% -> cand ([\d.]+)% +delta ([+\d.]+)pp +disc (\d+)/(\d+) +McNemar p=(\S+)"
    a10, a9 = (re.search(rx.format(re.escape(k)), log) for k in ("A10", "A9 (from A10)"))
    rows = [json.loads(x) for x in (IT700_GATE / "cand_A9.jsonl").open(encoding="utf-8")]
    base = [json.loads(x) for x in (IT700_GATE / "champ_A9.jsonl").open(encoding="utf-8")]
    true_a9 = T.pair(_arm_from_rows("cand", rows), _arm_from_rows("champ", base), "A10")
    true_m = list(re.finditer(rx.format("A10"), log))[-1]
    for c, m, p_fmt in (
        (T.pair(cand, champ, "A10"), a10, ".2g"),
        (T.pair(cand, champ, "A9-est"), a9, ".3g"),
        (true_a9, true_m, ".2g"),
    ):
        t = c.table
        assert (t.c, t.b) == (int(m.group(4)), int(m.group(5)))
        assert f"{100 * t.p2:.1f}" == m.group(1) and f"{100 * t.p1:.1f}" == m.group(2)
        assert f"{100 * t.diff:+.2f}" == m.group(3)
        assert f"{c.p:{p_fmt}}" == f"{float(m.group(6)):{p_fmt}}"
