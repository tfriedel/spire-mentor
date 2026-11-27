"""ppo.e2e.stats: the Newcombe port, McNemar, Wilson, the bootstraps and Holm."""

import ast
import json
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


# ── per-character training stats, evals and their schedule (GH #1197) ──────


def _run_end(char, outcome, act, floor, boss=(0, 0), elite=(0, 0), bosses=0):
    """A run's end record (`actors.end_record`'s fields the per-character split reads):
    `boss` and `elite` are (fights, fights won)."""
    return {
        "kind": "run",
        "character": char,
        "outcome": outcome,
        "won": outcome == "won",
        "act": act,
        "floor": floor,
        "bosses": bosses,
        "counts": {
            "fights/boss": boss[0],
            "fights_won/boss": boss[1],
            "fights/elite": elite[0],
            "fights_won/elite": elite[1],
        },
    }


def test_stats_row_by_char():
    from ppo.e2e import learner as L

    ends = [
        _run_end("Ironclad", "won", 3, 50, boss=(4, 4), elite=(3, 3), bosses=3),
        _run_end("Ironclad", "lost", 2, 30, boss=(2, 1), elite=(2, 2), bosses=1),  # act-2 boss
        _run_end("Silent", "lost", 1, 7, elite=(1, 1)),
        _run_end("Regent", "lost", 3, 49, boss=(4, 3), elite=(4, 3), bosses=3),  # 2nd act-3 boss
        _run_end(None, "capped", 1, 3),  # no character: the lane's
    ]
    update = {
        "ent_norm_ct": {"Silent": {"combat": 0.5}, "Regent": {"card_reward": 0.25}},
        "kl_ct": {"Silent": {"combat": 0.01}},
        "vloss_c": {"Regent": 0.2},
    }
    rows = {"Ironclad": 60, "Silent": 20, "Regent": 20}
    out = L.by_char_row(ends, default="Ironclad", rows_char=rows, update=update)
    assert set(out) == {"Ironclad", "Silent", "Regent"}
    ic, si, re = out["Ironclad"], out["Silent"], out["Regent"]
    assert ic["runs"] == 3 and si["runs"] == 1 and re["runs"] == 1
    assert ic["win"] == pytest.approx(1 / 3) and si["win"] == pytest.approx(0.0)
    assert ic["floor_mean"] == pytest.approx((50 + 30 + 3) / 3)
    assert ic["bosses_per_run"] == pytest.approx(4 / 3)
    # Act-k boss fights won / fought: earlier acts' bosses were all beaten.
    assert (ic["boss_win_act1"], ic["boss_win_act2"], ic["boss_win_act3"]) == pytest.approx(
        (1.0, 0.5, 1.0)
    )
    assert (re["boss_win_act1"], re["boss_win_act2"], re["boss_win_act3"]) == pytest.approx(
        (1.0, 1.0, 0.5)
    )
    assert math.isnan(si["boss_win_act1"])  # no boss fought: no point, not 0%
    assert (ic["elite_win"], re["elite_win"]) == pytest.approx((1.0, 0.75))
    # Deaths by act, per run (a capped run is a loss but no death).
    assert (ic["deaths_act1"], ic["deaths_act2"], ic["deaths_act3"]) == pytest.approx(
        (0.0, 1 / 3, 0.0)
    )
    assert (si["deaths_act1"], re["deaths_act3"]) == pytest.approx((1.0, 1.0))
    assert (ic["share"], si["share"], re["share"]) == pytest.approx((0.6, 0.2, 0.2))
    assert si["entropy"] == {"combat": 0.5} and si["kl"] == {"combat": 0.01}
    assert re["entropy"] == {"card_reward": 0.25} and re["kl"] == {} and ic["entropy"] == {}
    assert re["vloss"] == pytest.approx(0.2) and math.isnan(si["vloss"])
    # A character with training rows but no finished run this iteration.
    out = L.by_char_row([], default="Ironclad", rows_char={"Defect": 5}, update={})
    assert out["Defect"]["runs"] == 0 and math.isnan(out["Defect"]["win"])
    assert out["Defect"]["share"] == pytest.approx(1.0)


def test_update_entropy_by_char():
    """The update bins its per-type sums by (character, type); `char_type_means` reads
    them per character, and the folded bins are the pooled per-type sums."""
    from ppo.e2e import contracts as C
    from ppo.e2e import rowstore
    from ppo.e2e import update as U
    from ppo.e2e.net import synthetic_row_batch

    rb = synthetic_row_batch(4, seed=0)
    first = rowstore.offsets(rb)["tok"][:-1]
    rb["cat"][first, C.XC["X_CHAR"]] = [0, 1, 4, 2]  # undeclared Ironclad, Ironclad, Regent, Silent
    assert rowstore.row_chars(rb).tolist() == [0, 0, 3, 1]
    dtypes = ("combat", "card_reward")
    nch = len(C.CHARACTERS)
    s = np.zeros(nch * 2)
    n = np.zeros(nch * 2)
    s[0 * 2 + 0], n[0 * 2 + 0] = 3.0, 6  # Ironclad combat: mean 0.5
    s[3 * 2 + 1], n[3 * 2 + 1] = 1.0, 4  # Regent card_reward: mean 0.25
    out = U.char_type_means(s, n, dtypes)
    assert out == {"Ironclad": {"combat": 0.5}, "Regent": {"card_reward": 0.25}}
    assert U.fold_chars(s, 2).tolist() == [3.0, 1.0]


def test_eval_refuses_chars_net_without_chars_env(tmp_path, monkeypatch):
    """A net that read the "chars" columns is served only where they are filled: an
    Ironclad eval needs STS2_E2E_CHARS=1 (another character always fills them)."""
    from ppo.e2e import contracts as C
    from ppo.e2e import factory
    from ppo.e2e import net as N

    vocab = "b" * 64
    cfg = N.NetConfig.from_dict(
        {"dim": 32, "n_layers": 1, "n_heads": 2, "ffn": 64, "head_hidden": 32, "num_hidden": 64}
    )
    path = tmp_path / "chars.pt"
    stamps = C.stamps(vocab_sha256=vocab, filled=["chars"])
    N.save_checkpoint(path, N.E2ENet(cfg, seed=0), vocab_sha256=vocab, stamps=stamps)
    monkeypatch.setattr(factory, "CHARACTER", "Ironclad")
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    with pytest.raises(N.SchemaMismatch, match="chars"):
        N.StudentPolicy.from_checkpoint(path, vocab_sha256=vocab)
    monkeypatch.setenv(factory.CHARS_ENV, "1")
    N.StudentPolicy.from_checkpoint(path, vocab_sha256=vocab)
    monkeypatch.delenv(factory.CHARS_ENV)
    monkeypatch.setattr(factory, "CHARACTER", "Regent")
    N.StudentPolicy.from_checkpoint(path, vocab_sha256=vocab)
    # A net without the event is served anywhere (the identity rule adds events).
    plain = tmp_path / "plain.pt"
    N.save_checkpoint(
        plain, N.E2ENet(cfg, seed=0), vocab_sha256=vocab, stamps=C.stamps(vocab_sha256=vocab)
    )
    N.StudentPolicy.from_checkpoint(plain, vocab_sha256=vocab)


def _evalq():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "e2e_allchar_evalq.py"
    spec = importlib.util.spec_from_file_location("e2e_allchar_evalq", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evalq_only_one_character(tmp_path):
    from ppo.e2e import ckpt
    from ppo.e2e import ops_contract as OC

    Q = _evalq()
    M = 1_000_000
    lane = tmp_path / "lane"
    lane.mkdir()
    sd = ckpt.SaveDir(lane)
    mix = {"weights": {"Regent": 0.8, "Ironclad": 0.2}}
    (lane / "cfg.json").write_text(json.dumps({"mix": mix}), encoding="utf-8")
    for d, why in ((50 * M + 5, "milestone"), (150 * M + 5, "milestone"), (150 * M + 9, "k2")):
        src = tmp_path / f"s{d}.pt"
        src.write_bytes(str(d).encode())
        ckpt.pin(sd, decisions=d, reason=why, src=src)
    assert Q.queue(sd, only="Regent") == 2
    reqs = [json.loads(p.read_text(encoding="utf-8")) for p in sd.p(OC.EVAL_QUEUE).glob("*.json")]
    got = sorted((r["decisions"], r["character"], r["panel"]) for r in reqs)
    assert got == [(50 * M + 5, "Regent", "DEV-500"), (150 * M + 5, "Regent", "DEV-500")]
    assert Q.queue(sd, only="Regent") == 0  # once per pin
    assert Q.queue(sd, only="Regent", variant="ema0996") == 2  # its own requests
    late = [
        json.loads(p.read_text(encoding="utf-8")) for p in sd.p(OC.EVAL_QUEUE).glob("*ema0996*")
    ]
    assert len(late) == 2 and {r["variant"] for r in late} == {"ema0996"}
    with pytest.raises(SystemExit):
        Q.main(["--lane", str(lane), "--only", "Silent"])


def test_evalq_schedule(tmp_path):
    from ppo.e2e import ckpt
    from ppo.e2e import ops_contract as OC

    Q = _evalq()
    M = 1_000_000
    chars = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
    others = [(c, "DEV-500") for c in chars[1:]]
    ic = [("Ironclad", "DEV-1000")]

    def sched(d):
        return Q.schedule(d, characters=chars, ramp_end=205 * M)

    assert sched(30 * M) == []  # before the first milestone
    assert sched(50 * M + 1234) == ic  # ramp: Ironclad every 50M
    assert sched(100 * M + 7) == ic + others  # every character every 100M
    assert sched(150 * M) == ic and sched(200 * M + 9) == ic + others
    assert sched(250 * M + 9) == []  # past the ramp: Ironclad every 100M only
    assert sched(300 * M + 9) == ic + others
    assert OC.EVAL_PANELS["DEV-500"] == {"n": 500, "seed_start": OC.DEV_START}

    lane = tmp_path / "lane"
    lane.mkdir()
    sd = ckpt.SaveDir(lane)
    mix = {"characters": ["Ironclad", "Regent"], "settle_end": 5 * M, "ramp_end": 205 * M}
    (lane / "cfg.json").write_text(json.dumps({"mix": mix}), encoding="utf-8")
    for d, why in ((50 * M + 5, "milestone"), (100 * M + 5, "milestone"), (100 * M + 99, "k2")):
        src = tmp_path / f"s{d}.pt"
        src.write_bytes(str(d).encode())
        ckpt.pin(sd, decisions=d, reason=why, src=src)
    assert Q.queue(sd) == 3
    reqs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(sd.p(OC.EVAL_QUEUE).glob("*.json"))
    ]
    got = sorted((r["decisions"], r["character"], r["panel"], r["n"]) for r in reqs)
    assert got == [
        (50 * M + 5, "Ironclad", "DEV-1000", 1000),
        (100 * M + 5, "Ironclad", "DEV-1000", 1000),
        (100 * M + 5, "Regent", "DEV-500", 500),
    ]
    assert {r["variant"] for r in reqs} == {"ema0999"} and len({r["id"] for r in reqs}) == 3
    assert all(r["seed_start"] == OC.DEV_START for r in reqs)
    assert Q.queue(sd) == 0  # once per pin and character
    # A lane reverted to its 50M pin redoes the ramp: the redone 100M pin is new.
    redo = tmp_path / "redo.pt"
    redo.write_bytes(b"redo")
    ckpt.pin(sd, decisions=100 * M + 77, reason="milestone", src=redo)
    assert Q.queue(sd) == 2


def test_evalq_request_id_carries_the_pin(tmp_path, monkeypatch):
    """A pin redone at the SAME decision count after a tripwire revert is a different
    net: its requests need their own id, or the runner reuses the old pin's exported
    `evals/out/<id>/ema0999.pt` and the done file overwrites the old result."""
    from ppo.e2e import ckpt
    from ppo.e2e import ops_contract as OC

    Q = _evalq()
    M = 1_000_000
    lane = tmp_path / "lane"
    lane.mkdir()
    sd = ckpt.SaveDir(lane)
    mix = {"characters": ["Ironclad", "Regent"], "settle_end": 5 * M, "ramp_end": 205 * M}
    (lane / "cfg.json").write_text(json.dumps({"mix": mix}), encoding="utf-8")
    for body in (b"first", b"redone"):
        src = tmp_path / f"{body.decode()}.pt"
        src.write_bytes(body)
        ckpt.pin(sd, decisions=50 * M + 5, reason="milestone", src=src)
    pins = ckpt.pinned_states(sd)
    assert len(pins) == 2 and pins[0].decisions == pins[1].decisions
    assert Q.queue(sd) == 2
    reqs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(sd.p(OC.EVAL_QUEUE).glob("*.json"))
    ]
    assert len({r["id"] for r in reqs}) == 2  # distinct ids -> distinct evals/out/<id>/
    for r in reqs:
        assert r["sha256"][:8] in r["id"]

    # The seen file is saved after each queued request: a crash mid-loop does not
    # re-queue what was already written.
    lane2 = tmp_path / "lane2"
    lane2.mkdir()
    sd2 = ckpt.SaveDir(lane2)
    (lane2 / "cfg.json").write_text(json.dumps({"mix": mix}), encoding="utf-8")
    for d in (50 * M, 150 * M):
        src = tmp_path / f"p{d}.pt"
        src.write_bytes(str(d).encode())
        ckpt.pin(sd2, decisions=d, reason="milestone", src=src)
    real = OC.atomic_write_text
    calls = {"n": 0}

    def boom(path, text):
        if Path(path).parent == sd2.p(OC.EVAL_QUEUE):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash")
        return real(path, text)

    monkeypatch.setattr(Q.C, "atomic_write_text", boom)
    with pytest.raises(RuntimeError):
        Q.queue(sd2)
    seen = json.loads(sd2.p(Q.SEEN).read_text(encoding="utf-8"))
    assert len(seen) == 1


def test_evalq_queues_held_out_human_evals_when_the_lane_trains_on_them(tmp_path):
    """A lane with `human` p > 0 also gets a DEV-500 `"env": "human"` eval per character
    at each 100M pin; a lane without it gets none."""
    from ppo.e2e import ckpt
    from ppo.e2e import ops_contract as OC

    Q = _evalq()
    M = 1_000_000
    chars = ("Ironclad", "Regent")
    assert Q.human_schedule(100 * M + 3, characters=chars, on=True) == list(chars)
    assert Q.human_schedule(150 * M, characters=chars, on=True) == []
    assert Q.human_schedule(100 * M, characters=chars, on=False) == []
    lane = tmp_path / "lane"
    lane.mkdir()
    sd = ckpt.SaveDir(lane)
    mix = {"characters": list(chars), "settle_end": 5 * M, "ramp_end": 205 * M}
    cfg = {"mix": mix, "human": {"p": 0.05, "bank": ""}}
    (lane / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    src = tmp_path / "s.pt"
    src.write_bytes(b"x")
    ckpt.pin(sd, decisions=300 * M + 5, reason="milestone", src=src)
    assert Q.queue(sd) == 4  # Ironclad DEV-1000, Regent DEV-500, and two human evals
    reqs = [json.loads(p.read_text()) for p in sorted(sd.p(OC.EVAL_QUEUE).glob("*.json"))]
    human = sorted((r["character"], r["panel"]) for r in reqs if r.get("env") == "human")
    assert human == [("Ironclad", "DEV-500"), ("Regent", "DEV-500")]
    assert len({r["id"] for r in reqs}) == 4
    assert Q.queue(sd) == 0
