"""ppo.e2e.devboss: the Final - REF read on synthetic result files, the pair loop on fakes,
and, when the engine and the teacher checkpoints are present, capture -> play -> read.

The real-engine test reads the teacher pair from the bench archive (not in the repo, so it
skips in the public export) and writes only under tmp_path.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ppo.e2e import devboss as D
from ppo.e2e import evaluate as E
from ppo.e2e import panels as P
from ppo.e2e import score_test as T
from ppo.e2e import stats as S

PYCLI = Path(__file__).resolve().parents[2]
INPUTS = Path.home() / "projects" / "sts2" / "bench-archive" / "2026-11-02-e2e-inputs" / "hybrid"
RUN_CKPT, COMBAT_CKPT = INPUTS / "run_ckpt_v00375.pt", INPUTS / "it02500.pt"
WHEEL = {
    "pgo": True,
    "pgo_instrumented": False,
    "profdata_sha256": "ab" * 32,
    "target_cpu": None,
    "src_hash": "acc813aa12f20ad2",
    "so_path": "<pin>/sts2_rs/sts2_rs.abi3.so",
}
FULL = P.DEVBOSS_N_STATES * P.DEVBOSS_RESEEDS_PER_STATE


# ── the read ──────────────────────────────────────────────────────────────


def results(wins, *, errors=(), n_states=P.DEVBOSS_N_STATES):
    """One row per pair; `wins(i, j)` says who won, pairs in `errors` errored."""
    rows = []
    for i in range(n_states):
        cell = P.DEVBOSS_CELLS[i * len(P.DEVBOSS_CELLS) // n_states]
        for j in range(P.DEVBOSS_RESEEDS_PER_STATE):
            r = {"state_id": i, "j": j, "reseed": P.devboss_reseed(i, j), "cell": cell}
            if (i, j) in errors:
                r.update(outcome="error", won=False, error="boom")
            else:
                w = bool(wins(i, j))
                r.update(outcome="won" if w else "lost", won=w, cap=None, fights=[])
            rows.append(r)
    return rows


def write(path, rows, **meta):
    m = {
        "agent": "student",
        "states_sha256sums": "ab" * 32,
        "pairs": len(rows),
        "limit_states": None,
        "wheel": WHEEL,
        **meta,
    }
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(json.dumps({"meta": m}, sort_keys=True) + "\n" + body, encoding="utf-8")
    return path


REF_META = {"agent": "mcts", "sims": D.REF_SIMS, "boss_sims": D.REF_BOSS_SIMS}


def final_wins(i, j):
    return (i + j) % 3 == 0


def ref_wins(i, j):
    return i % 2 == 0


def test_the_read_is_a_cluster_bootstrap_over_states(tmp_path):
    f = write(tmp_path / "f.jsonl", results(final_wins))
    r = write(tmp_path / "r.jsonl", results(ref_wins), **REF_META)
    out = D.paired_results(f, r)
    assert out["valid"] and out["invalid"] == []
    assert (out["pairs"], out["states"]) == (FULL, P.DEVBOSS_N_STATES)
    sums = [sum(final_wins(i, j) - ref_wins(i, j) for j in range(4)) for i in range(600)]
    assert out["pooled"] == S.cluster_bootstrap(sums, [4] * 600)
    want = (sum(final_wins(i, j) for i in range(600) for j in range(4)) - 1200) / FULL
    assert out["pooled"]["mean"] == pytest.approx(want)
    for k, cell in enumerate(P.DEVBOSS_CELLS):
        ids = range(200 * k, 200 * (k + 1))
        assert out["cells"][cell] == S.cluster_bootstrap([sums[i] for i in ids], [4] * 200)
    assert out["final"]["pooled"]["wins"] == sum(
        final_wins(i, j) for i in range(600) for j in range(4)
    )
    assert out["ref"]["act1_boss"]["wins"] == 400 and out["ref"]["act1_boss"]["pairs"] == 800


def test_missing_pairs_follow_p2(tmp_path):
    ref_err = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)}
    fin_err = {(10, 0), (11, 1), (12, 2), (4, 0)}  # (4, 0): both errored, REF drops it
    f = write(tmp_path / "f.jsonl", results(lambda i, j: True, errors=fin_err))
    r = write(tmp_path / "r.jsonl", results(lambda i, j: False, errors=ref_err), **REF_META)
    out = D.paired_results(f, r)
    assert (out["dropped_ref_missing"], out["final_missing_as_loss"]) == (5, 3)
    assert out["valid"]  # 8 <= 12, 0.5% of 2400
    # Final wins every pair but its three errors, which count as losses.
    assert out["pooled"]["mean"] == pytest.approx((FULL - 5 - 3) / (FULL - 5))
    worse = fin_err | {(20 + k, 0) for k in range(5)}
    f2 = write(tmp_path / "f2.jsonl", results(lambda i, j: True, errors=worse))
    out2 = D.paired_results(f2, r)
    assert not out2["valid"] and "13 pairs with missing data > 12" in out2["invalid"][0]


def test_what_makes_a_read_invalid_or_stops_it(tmp_path):
    f = write(tmp_path / "f.jsonl", results(final_wins))
    r16 = write(tmp_path / "r16.jsonl", results(ref_wins), agent="mcts", sims=16, boss_sims=16)
    assert any("not the shipped MCTS" in x for x in D.paired_results(f, r16)["invalid"])
    other = dict(WHEEL, src_hash="0000000000000000")
    rw = write(tmp_path / "rw.jsonl", results(ref_wins), wheel=other, **REF_META)
    assert D.paired_results(f, rw)["invalid"] == ["Final and REF ran on different engine builds"]
    # so_path names a file, not a build: it does not count.
    rp = write(
        tmp_path / "rp.jsonl", results(ref_wins), wheel=dict(WHEEL, so_path="/x"), **REF_META
    )
    assert D.paired_results(f, rp)["valid"]
    small_f = write(tmp_path / "sf.jsonl", results(final_wins, n_states=6), limit_states=2)
    small_r = write(tmp_path / "sr.jsonl", results(ref_wins, n_states=6), **REF_META)
    assert (
        "a partial panel: 24 pairs, 2400 expected" in D.paired_results(small_f, small_r)["invalid"]
    )
    rs = write(tmp_path / "rs.jsonl", results(ref_wins), states_sha256sums="cd" * 32, **REF_META)
    with pytest.raises(D.DevbossError, match="different state sets"):
        D.paired_results(f, rs)
    with pytest.raises(D.DevbossError, match="different pairs"):
        D.paired_results(small_f, write(tmp_path / "r.jsonl", results(ref_wins), **REF_META))
    lines = f.read_text(encoding="utf-8").splitlines()
    (tmp_path / "dup.jsonl").write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    with pytest.raises(D.DevbossError, match="twice"):
        D.load_results(tmp_path / "dup.jsonl")
    (tmp_path / "short.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(D.DevbossError, match="the header says"):
        D.load_results(tmp_path / "short.jsonl")


def _smoke_arm(evals, name, wins):
    d = evals / "SMOKE" / name
    for sub in ("shards", "meta", "retry"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    seeds = list(P.SMOKE.seeds())
    for lo in range(P.SMOKE.start, P.SMOKE.stop, P.SHARD_SIZE):
        rows = [
            {
                "seed": s,
                "ascension": 10,
                "character": "Ironclad",
                "act1": "roll",
                "outcome": "won" if wins(s) else "lost",
                "won": bool(wins(s)),
                "wheel": WHEEL,
            }
            for s in seeds
            if lo <= s < lo + P.SHARD_SIZE
        ]
        shard = d / "shards" / f"s{lo}_{lo + P.SHARD_SIZE}.jsonl"
        shard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = f"arm={name}\npanel=SMOKE\nn=1000\nseed_start={P.SMOKE.start}\nenv=\n"
    (d / "meta" / "config.txt").write_text(config, encoding="utf-8")
    E.write_manifest(d, evals / "SMOKE" / f"{name}.sha256")


def test_score_test_reads_devboss_as_the_fourth_family_member(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "BOOT_RESAMPLES", 400)
    evals = tmp_path / "evals"
    _smoke_arm(evals, "F", lambda s: s % 3 == 0)
    _smoke_arm(evals, "R2", lambda s: s % 4 == 0)
    _smoke_arm(evals, "R1", lambda s: s % 2 == 0)
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("# test\n", encoding="utf-8")
    f = write(tmp_path / "f.jsonl", results(final_wins))
    r = write(tmp_path / "r.jsonl", results(ref_wins), **REF_META)
    kw = {"prereg": prereg, "evals": evals, "panel": "SMOKE", "final": "F", "cstar": "R2"}
    out = T.score(**kw, r1="R1", r2="R2", r3=None, devboss=(f, r))
    assert out.family["DEVBOSS Final - REF"]["p"] == D.paired_results(f, r)["pooled"]["p"]
    assert "DEVBOSS Final - REF" in T.render(out)
    r16 = write(tmp_path / "r16.jsonl", results(ref_wins), agent="mcts", sims=16, boss_sims=16)
    out16 = T.score(**kw, r1="R1", r2="R2", r3=None, devboss=(f, r16))
    assert out16.family["DEVBOSS Final - REF"]["p"] is None  # INVALID: counts as p = 1
    rs = write(tmp_path / "rs.jsonl", results(ref_wins), states_sha256sums="cd" * 32, **REF_META)
    with pytest.raises(T.ReadStopped, match="DEVBOSS: Final and REF played different"):
        T.score(**kw, r1="R1", r2="R2", r3=None, devboss=(f, rs))


# ── the pair loop, on fakes ───────────────────────────────────────────────


class FakeRun:
    """A RunManager stand-in whose state a fake env or agent scripts."""

    def __init__(self, act=1):
        self.act_index = act
        self.room = {"map_coord": (3, 16), "room_type": "boss"}
        self.ph = "combat"
        self.round = 1
        self.over = False
        self.player_won = False
        self.current_hp, self.max_hp = 60, 80
        self.map = {"boss": {"col": 3, "row": 16, "id": "B1"}, "second_boss": None}
        if act == 3:
            self.map["second_boss"] = {"col": 3, "row": 17, "id": "B2"}

    def is_over(self):
        return self.over

    def current_room(self):
        return dict(self.room)

    def phase(self):
        return self.ph

    def available_actions(self):
        if self.ph == "combat":
            return [{"action": "play_card", "card_idx": 0}, {"action": "end_turn"}]
        return [{"action": "move", "col": 3}] if self.ph == "map_choice" else [{"action": "x"}] * 2

    def potions(self):
        return [None, "FIRE_POTION"]

    def full_map(self):
        return self.map

    def combat_state(self):
        rnd = self.round

        class CS:
            def get_state(self):
                return {"round": rnd}

        return CS()


class Step:
    """One scripted step: mutate the fake run, then yield a decision, bootstrap or end."""

    def __init__(self, yields="decision", **set_):
        self.yields, self.set = yields, set_


class FakeEnv:
    def __init__(self, rm, script):
        self.rm, self.script = rm, list(script)

    def _next(self):
        st = self.script.pop(0)
        for k, v in st.set.items():
            setattr(self.rm, k, v)
        if st.yields == "decision":
            return E.RawDecision("combat", "combat", [], [[0], [1]], self.rm, 0, 0)
        if st.yields == "bootstrap":
            return E.RawDecision("ooc", "rewards", [], [[0], [1]], self.rm, 0, 0, bootstrap=True)
        return E.RawTerminal(outcome=st.yields, won=st.yields == "won", cap=st.set.get("cap"),
                             steps=0, rows=0)  # fmt: skip

    def reset(self):
        return self._next()

    def step(self, choice):
        assert choice == 0
        return self._next()


def student(cell, rm, script):
    p = D.Player.__new__(D.Player)
    p.policy = E.FirstOption()
    p.make_env = lambda seed, rm_, cell_, *, reseed, move: FakeEnv(rm_, script)
    s = {"seed": 1, "cell": cell, "move": {"action": "move", "col": 3}}
    return p._student(s, rm, 7)


BOSS2 = {"map_coord": (3, 17), "room_type": "boss"}


@pytest.mark.parametrize(
    ("cell", "act", "script", "want"),
    [
        # Act 1: WP2's env ends at the first out-of-combat decision (a bootstrap row).
        ("act1_boss", 1, [Step(), Step(round=3, current_hp=41), Step("bootstrap", ph="rewards")],
         ("won", True, [(True, 19, 3)])),
        # The raw stand-in yields that decision instead; the loop stops before acting.
        ("act2_boss", 2, [Step(), Step("decision", ph="rewards", current_hp=10)],
         ("won", True, [(True, 50, 1)])),
        ("act1_boss", 1, [Step(), Step(round=2), Step("lost", over=True)],
         ("lost", False, [(False, 60, 2)])),
        ("act1_boss", 1, [Step(), Step("capped", cap="turns", round=101)],
         ("capped", False, [(False, 0, 101)])),
        # The double boss: both fights, to the run's end.
        ("act3_double", 3, [Step(), Step("decision", room=BOSS2, current_hp=30, round=1),
                            Step("won", over=True, player_won=True, current_hp=12)],
         ("won", True, [(True, 30, 1), (True, 18, 1)])),
        ("act3_double", 3, [Step(), Step("decision", room=BOSS2, current_hp=30),
                            Step("lost", over=True)],
         ("lost", False, [(True, 30, 1), (False, 30, 1)])),
    ],
)  # fmt: skip
def test_student_pairs(cell, act, script, want):
    out = student(cell, FakeRun(act), script)
    got = [(f["won"], f["hp_lost"], f["turns"]) for f in out["fights"]]
    assert (out["outcome"], out["won"], got) == want


class FakeAgent:
    """`step(rm)` applies the next scripted step, like an agent acting on the run."""

    def __init__(self, script):
        self.script = list(script)

    def step(self, rm):
        for k, v in self.script.pop(0).set.items():
            setattr(rm, k, v)
        return {}


def test_pair_rows_carry_the_six_stamps(monkeypatch):
    """G0 review S3 (F8): a DEVBOSS pair row, an error row too, carries the writing
    process's stamps by value, as every `ppo.e2e.evaluate` row does."""
    import sts2_client
    from ppo.e2e import stamps as ST

    monkeypatch.setattr(sts2_client, "GAME_VERSION", E.GAME_VERSION)
    want = E.eval_stamps()

    class Broken:
        states = ({"reseeds": [7], "cell": "act1_boss"},)

        def base(self, i):
            raise RuntimeError("no state")

    p = D.Player.__new__(D.Player)
    p.set, p._base, p.agent = Broken(), None, "mcts"
    row = p.play((0, 0))
    assert row["outcome"] == "error" and "no state" in row["error"]
    assert {k: row[k] for k in ST.STAMP_KEYS} == {k: want[k] for k in ST.STAMP_KEYS}
    p.set = SimpleNamespace(states=Broken.states, base=lambda i: None, start=lambda i, j, b: None)
    monkeypatch.setattr(D.Player, "_direct", lambda self, s, rm, seed: {"outcome": "won"})
    row = p.play((0, 0))
    assert row["outcome"] == "won" and row["reseed"] == 7
    assert {k: row[k] for k in ST.STAMP_KEYS} == {k: want[k] for k in ST.STAMP_KEYS}


def test_direct_agent_plays_both_act3_bosses_and_the_move_between():
    p = D.Player.__new__(D.Player)
    p.agent = "policy"
    p._policy_agent = FakeAgent(
        [
            Step(round=2),
            Step(ph="map_choice"),  # boss 1 beaten: the one Move to boss 2
            Step(room=BOSS2, ph="combat", round=1, current_hp=25),
            Step(over=True, player_won=True, current_hp=5),
        ]
    )
    out = p._direct({"cell": "act3_double"}, FakeRun(3), 7)
    got = [(f["slot"], f["won"], f["hp_lost"], f["turns"]) for f in out["fights"]]
    assert (out["outcome"], out["won"], out["steps"]) == ("won", True, 4)
    assert got == [("boss", True, 35, 2), ("second_boss", True, 20, 1)]
    assert not p._policy_agent.script


def test_a_double_boss_alive_after_one_boss_is_not_a_win():
    rm = FakeRun(3)
    out = D.Player._end("act3_double", rm, D._Fights(), None, 0)
    assert (out["outcome"], out["won"]) == ("lost", False)


def test_boss_move():
    rm = FakeRun(3)
    rm.map["current_coord"] = {"col": 3, "row": 15}
    assert D.boss_move(rm, {"action": "move", "col": 3}) == "boss"
    assert D.boss_move(rm, {"action": "move", "col": 2}) is None
    assert D.boss_move(rm, {"action": "end_turn"}) is None
    rm.map["current_coord"] = (3, 16)
    assert D.boss_move(rm, {"action": "move", "col": 3}) == "second_boss"
    rm.map["current_coord"] = None
    assert D.boss_move(rm, {"action": "move", "col": 3}) is None


# ── capture -> play -> read on the engine ─────────────────────────────────

try:
    import sts2_rs  # noqa: F401

    HAVE_ENGINE = True
except ImportError:  # pragma: no cover
    HAVE_ENGINE = False


def devboss(*args, timeout=900):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E2E_", "STS2_"))}
    env["PYTHONPATH"] = str(PYCLI / "python")
    return subprocess.run(
        [sys.executable, "-m", "ppo.e2e.devboss", *map(str, args)],
        cwd=PYCLI,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.skipif(not HAVE_ENGINE, reason="sts2_rs wheel not importable")
@pytest.mark.skipif(not COMBAT_CKPT.is_file(), reason="teacher checkpoints not present")
def test_capture_play_read_on_the_engine(tmp_path, monkeypatch):
    out = tmp_path / "states"
    cap = ["capture", "--out", out, "--run-ckpt", RUN_CKPT, "--combat-ckpt", COMBAT_CKPT]
    c = devboss(*cap, "--procs", 2, "--per-cell", 1, "--batch", 2)
    assert c.returncode == 0, c.stderr
    states = [json.loads(x) for x in (out / D.STATES).read_text(encoding="utf-8").splitlines()]
    assert [s["cell"] for s in states] == list(P.DEVBOSS_CELLS)
    assert [s["reseeds"] for s in states] == [
        [P.devboss_reseed(i, j) for j in range(4)] for i in range(3)
    ]
    assert all(
        s["seed"] >= P.DEVBOSS_RUN_SEED_START and s["move"]["action"] == "move" for s in states
    )
    assert states[2]["second_boss"] and [s["act"] for s in states] == [1, 2, 3]
    # A finished capture is never overwritten; an interrupted one resumes to the same states.
    assert devboss(*cap, "--procs", 1, "--per-cell", 1).returncode == 2
    keep = {f: (out / f).read_bytes() for f in (D.STATES, D.RUNS)}
    for f in (D.SUMS, D.STATES, D.RUNS, D.CAPTURE):
        (out / f).unlink()
    again = devboss(*cap, "--procs", 1, "--per-cell", 1, "--batch", 5)
    assert again.returncode == 0, again.stderr
    assert "seed(s) already done" in again.stdout and "0 seed(s) already done" not in again.stdout
    assert {f: (out / f).read_bytes() for f in (D.STATES, D.RUNS)} == keep

    # Every state replays to its capture fingerprint, in-process too.
    import sts2_client

    monkeypatch.setattr(sts2_client, "GAME_VERSION", E.GAME_VERSION)
    ss = D.StateSet.load(out)
    rm = ss.start(2, 0)
    assert (rm.current_room() or {}).get("room_type") == "boss" and rm.phase() == "combat"

    # REF-shaped plays are repeatable; the student stand-in plays the same pairs.
    play = ["play", "--states", out, "--procs", 2]
    for name in ("ref_a", "ref_b"):
        p = devboss(
            *play, "--agent", "mcts", "--sims", 8, "--boss-sims", 8, "--out", tmp_path / name
        )
        assert p.returncode == 0, p.stderr
    assert (tmp_path / "ref_a").read_bytes() == (tmp_path / "ref_b").read_bytes()
    # A stopped play resumes from its .part (same config only) to the same bytes.
    mcts8 = ["--agent", "mcts", "--sims", 8, "--boss-sims", 8]
    lines = (tmp_path / "ref_a").read_text(encoding="utf-8").splitlines(keepends=True)
    (tmp_path / "ref_d.part").write_text("".join(lines[:6]) + '{"torn', encoding="utf-8")
    p = devboss(*play, *mcts8, "--out", tmp_path / "ref_d")
    assert p.returncode == 0 and "(7 to play)" in p.stdout, p.stdout + p.stderr
    assert (tmp_path / "ref_d").read_bytes() == (tmp_path / "ref_a").read_bytes()
    assert not (tmp_path / "ref_d.part").exists()
    (tmp_path / "ref_e.part").write_text("".join(lines[:6]), encoding="utf-8")
    p = devboss(
        *play, "--agent", "mcts", "--sims", 9, "--boss-sims", 8, "--out", tmp_path / "ref_e"
    )
    assert p.returncode == 2 and "written under another config" in p.stderr
    s = devboss(*play, "--agent", "student", "--policy", "first", "--segment-env", "raw",
                "--out", tmp_path / "stu")  # fmt: skip
    assert s.returncode == 0, s.stderr
    meta, rows = D.load_results(tmp_path / "stu")
    from ppo.e2e import stamps as ST

    assert all(set(ST.STAMP_KEYS) <= set(r) for r in rows.values())
    assert {r["py_src"] for r in rows.values()} == {E.eval_stamps()["py_src"]}
    assert meta["pairs"] == 12 and meta["agent"] == "student" and not meta["limit_states"]
    assert all(r["outcome"] in ("won", "lost", "capped") for r in rows.values())
    assert all(r["fights"] and r["fights"][0]["slot"] == "boss" for r in rows.values())
    rep = devboss("report", tmp_path / "stu", tmp_path / "ref_a", "--json")
    assert rep.returncode == 0, rep.stderr
    read = json.loads(rep.stdout)
    assert read["pairs"] == 12 and not read["valid"]
    assert any("partial panel" in x for x in read["invalid"])
    assert any("not the shipped MCTS" in x for x in read["invalid"])
    # A changed state file stops every play.
    (out / D.STATES).write_text("", encoding="utf-8")
    bad = devboss(*play, "--agent", "mcts", "--out", tmp_path / "ref_c")
    assert bad.returncode == 2 and "sha256 differs" in bad.stderr
