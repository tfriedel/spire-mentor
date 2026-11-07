"""Teacher wrappers (plan §7): merged-class log-probabilities and T8.

The merge tests use fake teachers. The real-teacher tests record decisions
from one campaign run (`ppo.e2e.env.E2EEnv`: v0.111.0, A10 Ironclad) played
greedy over merged classes by teacher pair R2, with the env's own merged
classes. They need the campaign's teacher checkpoints and skip where those
are absent (the public export, other machines). The env merges equal OOC
options too, but rarely, and none of that run's recorded OOC decisions is
merged: so the run teacher's merge is also checked on a hand-made merge, and
on the final R3 pair's own greedy runs, whose merged OOC rows the env made
(`test_t8_on_the_final_r3_pair`).
"""

from __future__ import annotations

import dataclasses
import pathlib
import types

import numpy as np
import pytest
import sts2_client
import torch
from ppo.e2e import contracts as C
from ppo.e2e import factory
from ppo.e2e import net as N
from ppo.e2e import teachers as TE

GAME_VERSION = factory.GAME_VERSION
R2 = TE.PAIRS["R2"]
HAVE_R2 = R2["combat"].is_file() and R2["run"].is_file()
needs_r2 = pytest.mark.skipif(not HAVE_R2, reason="needs the R2 teacher checkpoints")
R3 = TE.PAIRS["R3"]
HAVE_R3 = R3["combat"].is_file() and R3["run"].is_file()
needs_r3 = pytest.mark.skipif(not HAVE_R3, reason="needs the final R3 teacher checkpoints")


# ── fake teachers: the merge itself ───────────────────────────────────────


def test_identity_map_returns_the_options_log_probs():
    lp = torch.log_softmax(torch.randn(3, 5, dtype=torch.float64), -1)
    members = torch.arange(5).expand(3, 5)
    torch.testing.assert_close(TE.merge_logp(lp, members, 5), lp)


def test_members_merge_by_log_sum_exp_and_sum_to_one():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4, 6, generator=g, dtype=torch.float64)
    z[1, 4:] = float("-inf")  # a decision with 4 options
    lp = torch.log_softmax(z, -1)
    members = torch.tensor(
        [[0, 1, 0, 2, 1, 3], [0, 0, 1, 1, -1, -1], [2, 1, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]]
    )
    out = TE.merge_logp(lp, members, 6)
    for i in range(4):
        for c in range(6):
            sel = members[i] == c
            want = torch.logsumexp(lp[i][sel], 0) if sel.any() else torch.tensor(float("-inf"))
            torch.testing.assert_close(out[i, c], want.to(out.dtype))
    real = torch.isfinite(out)
    sums = torch.where(real, out.exp(), torch.zeros_like(out)).sum(-1)
    torch.testing.assert_close(sums, torch.ones(4, dtype=torch.float64))


def test_option_map_must_be_total_and_onto():
    TE.check_option_map(np.array([0, 1, 1, 2]), 3)
    with pytest.raises(TE.OptionMapError):
        TE.check_option_map(np.array([0, 2, 2]), 3)  # class 1 has no option
    with pytest.raises(TE.OptionMapError):
        TE.check_option_map(np.array([0, 1, 3]), 3)  # option outside the classes
    with pytest.raises(TE.OptionMapError):
        TE.check_option_map(np.array([-1, 0, 1]), 2)
    with pytest.raises(TE.OptionMapError):
        TE.check_option_map(np.array([], np.int64), 1)
    lp = torch.log_softmax(torch.randn(1, 3), -1)
    with pytest.raises(TE.OptionMapError):  # wrong member count for the options
        TE._merge(lp, [3], [np.array([0, 1])], None)
    with pytest.raises(TE.OptionMapError):  # a class count that leaves a class empty
        TE._merge(lp, [3], [np.array([0, 1, 1])], [3])


def test_manifest_mismatch_is_refused(tmp_path):
    f = tmp_path / "x.pt"
    f.write_bytes(b"abc")
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  ./x.pt\n", encoding="utf-8")
    assert TE.manifest_sha256(f) == "0" * 64
    with pytest.raises(TE.TeacherMismatch):
        TE._verify(f)
    (tmp_path / "SHA256SUMS").write_text(TE.file_sha256(f) + "  x.pt\n", encoding="utf-8")
    assert TE._verify(f) == TE.file_sha256(f)


def test_load_pair_takes_a_name_or_both_paths():
    for kw in ({}, {"combat": "c.pt"}, {"run": "r.pt"}, {"name": "R2", "combat": "c.pt"}):
        with pytest.raises(ValueError):
            TE.load_pair(**kw)
    with pytest.raises(KeyError):
        TE.load_pair("R9")


# ── the pinned pairs (teachers.toml) ──────────────────────────────────────


def test_the_pins_load_from_the_data_file():
    """PREREG-0 pins the teachers, file and sha256, in `teachers.toml` (data,
    which `test_champion_resolvers_agree.py` does not scan): a champion
    promotion cannot move them. Where the files are present, they are the
    pinned bytes, and the archive's SHA256SUMS records the same sha256."""
    assert TE.PINS_FILE.name == "teachers.toml" and TE.PINS_FILE.is_file()
    assert {"R2", "R3m", "R3"} <= set(TE.PINS)
    for name, pair in TE.PINS.items():
        assert tuple(pair) == TE.ROLES
        assert TE.PAIRS[name] == {role: pin.path for role, pin in pair.items()}
    assert TE.PAIRS["R2"]["combat"] == TE.ARCHIVE / "run/it06000.pt"
    assert TE.PAIRS["R2"]["run"].parent == TE.repo_root() / "sts2-pycli/data/ppo"
    assert TE.PAIRS["R3m"] == {
        "combat": TE.ARCHIVE / "hybrid/it02500.pt",
        "run": TE.ARCHIVE / "hybrid/run_ckpt_v00375.pt",
    }
    # The final R3, the teacher (PREREG-0 amendment 2): H2's matched it04000 pair.
    assert TE.PAIRS["R3"] == {
        "combat": TE.H2H3_ARCHIVE / "hybrid/ckpt/it04000.pt",
        "run": TE.H2H3_ARCHIVE / "hybrid/run/ckpt/at_it04000.pt",
    }
    for pair in TE.PINS.values():
        for pin in pair.values():
            if not pin.path.is_file():
                continue
            listed = TE.manifest_sha256(pin.path)
            assert listed in (None, pin.sha256), pin
            if listed is None:  # a repo file: hash it
                assert TE.file_sha256(pin.path) == pin.sha256, pin


def test_a_file_that_differs_from_its_pin_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "x.pt"
    f.write_bytes(b"abc")
    sha = TE.file_sha256(f)
    assert TE._verify(f, sha) == sha
    with pytest.raises(TE.TeacherMismatch, match="!= pinned"):
        TE._verify(f, "0" * 64)
    for cls in (TE.CombatTeacher, TE.RunTeacher):  # before the file is read as a checkpoint
        with pytest.raises(TE.TeacherMismatch, match="!= pinned"):
            cls(f, expect_sha256="0" * 64)
    # `load_pair` hands each role its pin, and nothing for paths.
    seen = {}

    def fake(role):
        def make(path, device="cpu", *, expect_sha256=None):
            seen[role] = (pathlib.Path(path), expect_sha256)

        return make

    monkeypatch.setattr(TE, "CombatTeacher", fake("combat"))
    monkeypatch.setattr(TE, "RunTeacher", fake("run"))
    for name in TE.PINS:
        TE.load_pair(name)
        assert seen == {r: (p.path, p.sha256) for r, p in TE.PINS[name].items()}
    TE.load_pair(combat="c.pt", run="r.pt")
    assert seen == {"combat": (pathlib.Path("c.pt"), None), "run": (pathlib.Path("r.pt"), None)}


_A, _B = "a" * 64, "b" * 64
PINS_TOML = f"""
[P.combat]  # a comment
root = "archive"
path = "run/x.pt"
sha256 = "{_A}"

[P.run]
root = "repo"
path = "sts2-pycli/data/ppo/y.pt"
sha256 = "{_B}"
"""


def test_a_pins_file_loads_its_roots(tmp_path):
    f = tmp_path / "teachers.toml"
    f.write_text(PINS_TOML, encoding="utf-8")
    assert TE.load_pins(f) == {
        "P": {
            "combat": TE.Pin(TE.ARCHIVE / "run/x.pt", _A),
            "run": TE.Pin(TE.repo_root() / "sts2-pycli/data/ppo/y.pt", _B),
        }
    }
    f.write_text(PINS_TOML.split("[P.run]")[0], encoding="utf-8")  # no run teacher
    with pytest.raises(ValueError, match="pair 'P' needs the roles"):
        TE.load_pins(f)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('root = "repo"', 'root = "home"', "root 'home' is not one of"),
        ('[P.run]\nroot = "repo"', '[P.critic]\nroot = "repo"', "pair 'P' needs the roles"),
        ("[P.combat]  # a comment", "[Q]\nx = 1\n\n[P.combat]", "pair 'Q' needs the roles"),
        (f'sha256 = "{_A}"', f'sha256 = "{_A.upper()}"', "not 64 lowercase hex"),
        (f'sha256 = "{_A}"', 'sha256 = "abc"', "not 64 lowercase hex"),
        (f'sha256 = "{_B}"', "", r"\[P\.run\] needs the keys"),
        ('path = "run/x.pt"', 'path = "run/x.pt"\nnote = "x"', r"\[P\.combat\] needs the keys"),
        ('path = "run/x.pt"', 'path = "/etc/x.pt"', "must be relative"),
        ('path = "run/x.pt"', 'path = "../x.pt"', "must be relative"),
        ('path = "run/x.pt"', 'path = ""', "must be relative"),
    ],
)
def test_a_malformed_pins_file_is_refused(tmp_path, old, new, match):
    assert PINS_TOML.count(old) == 1
    f = tmp_path / "teachers.toml"
    f.write_text(PINS_TOML.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        TE.load_pins(f)


@needs_r2
def test_load_pair_by_paths_is_the_named_pair():
    """The final R3 (PREREG-0 amendment 2) loads by its paths: the same pair
    as by name, each file checked against its manifest."""
    by_name = _keep_threads(TE.load_pair, "R2")
    by_path = _keep_threads(lambda: TE.load_pair(combat=R2["combat"], run=R2["run"]))
    assert by_path.stamps() == by_name.stamps()
    assert by_path.stamps()["combat"]["sha256"] == TE.manifest_sha256(R2["combat"])


# ── real teachers on recorded decisions ───────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _campaign_process():
    """The env's factory needs the process to select v0.111.0
    (`STS2_GAME_VERSION`), as `test_env.py` does."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", GAME_VERSION)
        yield


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    """Small CPU forwards on a shared, loaded host: many torch threads only
    contend (24 threads made this module ~5x slower). #1081's policies also
    pin torch to one thread at construction; restore the caller's count."""
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def _keep_threads(fn, *a):
    """Construct a #1081 policy (which pins torch to one thread), then restore."""
    n = torch.get_num_threads()
    try:
        return fn(*a)
    finally:
        torch.set_num_threads(n)


def _all_unk_rare(rare):
    """A RareMap that sends every content id to its slot's UNK row."""
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.data import RareMap

    maps = {}
    for slot, m in rare.maps.items():
        unk = T.vocab().tables[T.S[slot]].unk
        a = np.arange(len(m), dtype=np.int64)
        maps[slot] = np.where(a < unk, unk, a)
    return RareMap(maps, {})


@pytest.fixture(scope="module")
def recorded():
    """Up to 40 combat and 12 OOC stored decisions of one campaign run, played
    greedy over merged classes by the R2 pair (run net cont300_it1150 and
    #1081's it06000), with #1081's own reference outputs at each decision."""
    from ppo.combat_ctx.play import CtxNet
    from ppo.e2e import tokens as TK
    from ppo.e2e.env import E2EDecision, E2EEnv
    from ppo.net_policy import NetPolicy

    combat = TE.CombatTeacher(R2["combat"])
    run = TE.RunTeacher(R2["run"])
    pair = TE.Teachers(combat, run)
    ref_combat = _keep_threads(CtxNet, str(R2["combat"]))
    ref_run = _keep_threads(NetPolicy, str(R2["run"]))
    all_unk = _all_unk_rare(combat.rare)
    env = E2EEnv(1_300_000_011)
    assert env.rm.game_version == GAME_VERSION
    d = env.reset()
    cbt, runs = [], []
    while isinstance(d, E2EDecision) and (len(cbt) < 40 or len(runs) < 12):
        # Live references die at the next step: copy what the tests read.
        rec = {"classes": list(d.classes), "n_classes": len(d.members), "kind": d.kind}
        # WP3b's tokenizer row, S_ctx content RareMap'd for this teacher (X-C);
        # without a RareMap (what `net.make_student` encodes at eval); and with
        # every content id sent to UNK (a RareMap that surely acts).
        rec["row"] = TK.encode(d, rare=combat.rare, priv=False)
        rec["row_plain"] = TK.encode(d, rare=None, priv=False)
        rec["row_unk"] = TK.encode(d, rare=all_unk, priv=False)
        if d.kind == "combat":
            rec["tok"] = combat.encode(d)
            rec["kinds"] = [C.combat_action_kind(a) for a in d.actions]
            rec["dtype"] = C.classify_combat(d.actions, d.cs.pending_choice())[0]
            job = {"snap": d.snap, "bv": d.bv, "row": {"room_type": d.room}}
            rec["ref_logits"] = ref_combat.logits([(job, d.cs, d.actions)])[0]
            lp = combat.logp([rec["tok"]], [d.classes], [len(d.members)])[0]
            cbt.append(rec)
        else:
            rec["feat"] = run.featurize_decision(d)
            rec["ref_logp"] = ref_run.logits_value(d.run_obs, d.actions, d.metrics)[0]
            lp = run.logp([rec["feat"]], [d.classes], [len(d.members)])[0]
            # The schema's env decision gives the run teacher the same input.
            (lp_env,) = run.logp_decisions([d.decision])
            assert np.array_equal(lp_env, lp)
            runs.append(rec)
        (both,) = pair.logp_decisions([d])
        assert np.array_equal(both, lp)
        d = env.step(int(np.argmax(lp)))
    assert len(cbt) >= 10 and len(runs) >= 3, (len(cbt), len(runs))
    return {"combat": combat, "run": run, "cbt": cbt, "runs": runs}


@needs_r2
def test_combat_teacher_reproduces_1081s_own_logits(recorded):
    """X-C's core: the wrapper fed #1081's tokens gives CtxNet's logits."""
    t = recorded["combat"]
    for r in recorded["cbt"]:  # one at a time: bitwise, same padding as CtxNet
        z = t.logits([r["tok"]])[0]
        assert np.array_equal(z, r["ref_logits"])
    zs = t.logits([r["tok"] for r in recorded["cbt"]])  # batched: padding moves sums
    for z, r in zip(zs, recorded["cbt"], strict=True):
        np.testing.assert_allclose(z, r["ref_logits"], rtol=0, atol=1e-4)


@needs_r2
def test_combat_teacher_length_groups_match_one_batch(recorded, monkeypatch):
    """`serve_groups`: each length group runs as its own forward, and the rows land
    back in order: a row's logits are its group's forward's (bitwise), within float
    rounding of the one-batch forward's, and -inf past its options."""
    t = recorded["combat"]
    toks = [r["tok"] for r in recorded["cbt"]]
    monkeypatch.setattr(TE, "MIN_GROUP_ROWS", 3)
    one = t.logits_tensor(toks)
    t.serve_groups = 3
    try:
        grp = t.logits_tensor(toks)
    finally:
        t.serve_groups = 1
    assert grp.shape == one.shape
    parts = TE.length_groups([len(d.sec) for d in toks], 3)
    assert len(parts) == 3
    for p in parts:
        alone = t.logits_tensor([toks[i] for i in p])
        assert torch.equal(grp[torch.from_numpy(p), : alone.shape[1]], alone)
    fin = torch.isfinite(one)
    assert torch.equal(fin, torch.isfinite(grp))
    torch.testing.assert_close(grp[fin], one[fin], rtol=0, atol=1e-4)


@needs_r2
def test_combat_teacher_packed_matches_padded(recorded):
    """`packed` (the learner's `--packed-serve`): the same logits as the padded
    forward up to float rounding, -inf at the same places, one decision alone or
    many of different lengths together, and the merged log-probs follow."""
    t = recorded["combat"]
    toks = [r["tok"] for r in recorded["cbt"]]
    assert len({len(d.sec) for d in toks}) > 1
    one = t.logits_tensor(toks)
    t.packed = True
    try:
        got = t.logits_tensor(toks)
        alone = [t.logits_tensor([d]) for d in toks[:3]]
        lp = t.logp(
            toks, [r["classes"] for r in recorded["cbt"]], [r["n_classes"] for r in recorded["cbt"]]
        )
    finally:
        t.packed = False
    assert got.shape == one.shape
    fin = torch.isfinite(one)
    assert torch.equal(fin, torch.isfinite(got))
    torch.testing.assert_close(got[fin], one[fin], rtol=0, atol=1e-4)
    for i, z in enumerate(alone):
        torch.testing.assert_close(z[0], got[i, : z.shape[1]], rtol=0, atol=1e-4)
    ref = t.logp(
        toks, [r["classes"] for r in recorded["cbt"]], [r["n_classes"] for r in recorded["cbt"]]
    )
    for a, b in zip(lp, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-4)


@needs_r2
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_combat_teacher_packed_on_cuda_matches_padded(recorded):
    """The fp32 teacher on CUDA (the lane's): packed vs padded within float rounding."""
    t = recorded["combat"]
    toks = [r["tok"] for r in recorded["cbt"]]
    t.net.to("cuda")
    t.device = torch.device("cuda")
    try:
        one = t.logits_tensor(toks)
        t.packed = True
        got = t.logits_tensor(toks)
    finally:
        t.packed = False
        t.net.to("cpu")
        t.device = torch.device("cpu")
    fin = torch.isfinite(one)
    assert torch.equal(fin, torch.isfinite(got))
    torch.testing.assert_close(got[fin], one[fin], rtol=0, atol=1e-4)


@needs_r2
def test_logp_rows_checks_its_rows_and_queues_before_it_waits(recorded):
    """The vectorized X-C path keeps `sctx_rows`'s and `check_option_map`'s
    refusals, and its two halves give `logp_rows`'s result."""
    t = recorded["combat"]
    recs = [r for r in recorded["cbt"] if r["n_classes"] >= 2]
    rb = N.concat_rows([_combat_row(r) for r in recs])
    done = t.logp_rows_start(rb, [2, 0])
    for a, b in zip(done(), t.logp_rows(rb, [2, 0]), strict=True):
        assert np.array_equal(a, b)
    assert t.logp_rows_start(rb, [])() == []
    with pytest.raises(ValueError, match="not a combat row"):
        t.logp_rows(dict(rb, is_combat=np.zeros_like(rb["is_combat"])), [0])
    short = dict(rb, n_sctx=np.where(np.arange(len(recs)) == 1, 1, rb["n_sctx"]))
    with pytest.raises(ValueError, match="no S_ctx block"):
        t.logp_rows(short, [0, 1])
    unhit = dict(rb, n_classes=rb["n_classes"] + 1)  # the last class has no option
    with pytest.raises(TE.OptionMapError):
        t.logp_rows(unhit, [1])
    oob = dict(rb, mem_class=rb["mem_class"].copy())
    oob["mem_class"][0] = 99
    with pytest.raises(TE.OptionMapError):
        t.logp_rows(oob, [0])


@needs_r2
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_run_teacher_pinned_batch_is_collates(recorded):
    from ppo.model import collate

    t = recorded["run"]
    feats = [r["feat"] for r in recorded["runs"]]
    t.device = torch.device("cuda")
    try:
        got, want = t._batch(feats), collate(feats, t.device)
    finally:
        t.device = torch.device("cpu")
    for f in dataclasses.fields(want):
        x, y = getattr(got, f.name), getattr(want, f.name)
        if isinstance(y, torch.Tensor):
            assert x.device == y.device and x.dtype == y.dtype and torch.equal(x, y), f.name
        else:
            assert x == y, f.name


@needs_r2
def test_combat_teacher_t8_over_the_envs_merged_classes(recorded):
    t = recorded["combat"]
    recs = recorded["cbt"]
    got = t.logp(
        [r["tok"] for r in recs], [r["classes"] for r in recs], [r["n_classes"] for r in recs]
    )
    merged_any = False
    for lp, r in zip(got, recs, strict=True):
        m = np.asarray(r["classes"])
        assert np.isfinite(lp).all() and len(lp) == r["n_classes"]
        assert abs(np.exp(lp).sum() - 1.0) < 1e-9
        p_opt = np.exp(r["ref_logits"] - np.logaddexp.reduce(r["ref_logits"]))
        for c in range(len(lp)):
            assert abs(np.exp(lp[c]) - p_opt[m == c].sum()) < 1e-5
        merged_any |= len(lp) < len(m)
    assert merged_any, "no recorded decision had two merged options (pick another seed)"
    print(f"\n{len(recs)} combat decisions, {sum(len(r['classes']) for r in recs)} options")


@needs_r2
def test_run_teacher_matches_net_policy_and_t8(recorded):
    t = recorded["run"]
    recs = recorded["runs"]
    feats = [r["feat"] for r in recs]
    got = t.logp(feats, [r["classes"] for r in recs], [r["n_classes"] for r in recs])
    for lp, r in zip(got, recs, strict=True):
        # This run's recorded OOC options happen to be unmerged (the env does merge
        # equal OOC options; the R3 T8 test covers those rows).
        assert r["classes"] == list(range(len(r["classes"])))
        # NetPolicy's log-softmax is fp32, the wrapper's float64: ~1e-6 relative.
        np.testing.assert_allclose(lp, r["ref_logp"], rtol=1e-5, atol=2e-5)
        assert abs(np.exp(lp).sum() - 1.0) < 1e-9
    # A merge: the first two options of every decision with >= 3 options.
    multi = [i for i, f in enumerate(feats) if f.n_actions >= 3]
    assert multi
    classes = [np.r_[0, np.arange(feats[i].n_actions - 1)] for i in multi]
    lp2 = t.logp([feats[i] for i in multi], classes, [feats[i].n_actions - 1 for i in multi])
    for lp, i in zip(lp2, multi, strict=True):
        ref = recs[i]["ref_logp"]
        assert abs(np.exp(lp).sum() - 1.0) < 1e-9
        np.testing.assert_allclose(lp[0], np.logaddexp(ref[0], ref[1]), rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(lp[1:], ref[2:], rtol=1e-5, atol=2e-5)


def test_run_teacher_honest_previews_featurize_the_masked_obs():
    """SCHEMA open question 15's fallback: with `honest_previews` the run
    teacher featurizes `contracts.honest_obs(obs)`; by default the raw obs."""
    opt = {"label": "Take the relic", "gold_delta": 50, "hp_delta": -5, "max_hp_delta": 0}
    opt |= {"relics_gained": 2, "relics_lost": 0, "potions_gained": 1, "potions_lost": 0}
    opt |= {"cards_gained": 0, "cards_lost": 0}
    obs = {"seed": 7, "gold": 99, "event_options": [opt]}
    d = types.SimpleNamespace(run_obs=obs, actions=[{"kind": "event_choice"}], metrics={})
    for honest in (False, True):
        t = object.__new__(TE.RunTeacher)  # the obs routing only: no checkpoint
        t.honest_previews = honest
        seen = []
        t.featurize = lambda o, a, m, seen=seen: seen.append(o)
        t.featurize_decision(d)
        (got,) = seen
        if honest:
            assert got == C.honest_obs(obs) and "seed" not in got
            (o,) = got["event_options"]
            assert o["relics_gained"] == 1 and o["gold_delta"] == o["potions_gained"] == 0
        else:
            assert got is obs


@needs_r2
def test_run_teacher_refuses_other_features(tmp_path):
    ck = torch.load(R2["run"], map_location="cpu", weights_only=True)
    ck["cfg"] = dict(ck["cfg"], features="card_mods")
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    with pytest.raises(TE.TeacherMismatch):
        TE.RunTeacher(bad)
    ck.pop("cfg")
    torch.save(ck, bad)
    with pytest.raises(TE.TeacherMismatch):
        TE.RunTeacher(bad)


@needs_r2
def test_teacher_stamps_carry_the_manifest_sha(recorded):
    s = TE.Teachers(recorded["combat"], recorded["run"]).stamps()
    assert s["combat"]["sha256"].startswith("cc925b03")  # PREREG-0's it06000
    assert s["run"]["features"] == TE.RUN_FEATURES
    assert len(s["run"]["sha256"]) == 64
    # The contract's teacher stamp: the same file and RareMap hashes.
    ts = s["combat"]["teacher"]
    assert set(ts) == {"ckpt_sha256", "rare_sha256", "sctx_vocab_sha256"}
    assert ts["ckpt_sha256"] == s["combat"]["sha256"]
    assert ts["rare_sha256"] == s["combat"]["rare_sha256"]
    assert C.stamps(vocab_sha256="v" * 64, teacher=ts)["teacher"] == ts


def _combat_row(rec):
    """A one-row `contracts` row batch whose S_ctx block is #1081's tokens of
    a recorded decision: the head (RUNCTX, the five SUM_* tokens with empty
    pools, one BOSS token), then the block; members are the block's action
    tokens, merged by the env's classes, non-representatives hidden. The
    trailing columns stay absent but for X_DTYPE and X_KIND."""
    tok, cls = rec["tok"], np.asarray(rec["classes"], np.int64)
    n, na, nc = len(tok.sec), int(tok.n_act), int(rec["n_classes"])
    head = [C.SEC["RUNCTX"], *C.SUMMARY_SECTIONS, C.SEC["BOSS"]]
    h = len(head)
    absent = np.asarray(list(C.SCTX_PAD) + [0] * len(C.X_CAT), np.int16)
    cat = np.tile(absent, (h + n, 1))
    cat[h:, : C.SCTX_N_CAT] = tok.cat
    cat[0, C.XC["X_DTYPE"]] = rec["dtype"]
    acts = np.arange(h + n - na, h + n)
    cat[acts, C.XC["X_KIND"]] = rec["kinds"]
    cont = np.zeros((h + n, C.N_CONT), np.float16)
    cont[h:, : C.SCTX_N_CONT] = tok.cont
    reps = np.asarray([int(np.flatnonzero(cls == c)[0]) for c in range(nc)])
    flags = np.zeros(h + n, np.uint8)
    flags[1 : 1 + C.N_SUMMARY_TOKENS] = C.TF_POOLED
    flags[h:] = C.TF_TEACHER
    flags[acts] |= C.TF_OPTION | C.TF_HIDDEN
    flags[acts[reps]] = C.TF_TEACHER | C.TF_OPTION | C.TF_REP
    z = np.zeros(0, np.uint16)
    return {
        "run_uid": np.asarray([1], np.int64),
        "t": np.asarray([0], np.int32),
        "is_combat": np.asarray([True]),
        "dtype": np.asarray([rec["dtype"]], np.uint8),
        "reward": np.zeros(1, np.float32),
        "done": np.zeros(1, bool),
        "bootstrap": np.zeros(1, bool),
        "actor": np.asarray([1], np.uint8),
        "logp_behav": np.zeros(1, np.float32),
        "n_classes": np.asarray([nc], np.uint16),
        "action": np.zeros(1, np.uint16),
        "iter": np.zeros(1, np.int32),
        "v_behav": np.zeros(1, np.float32),
        "cell": np.asarray([C.CELL_NONE], np.uint8),
        "n_tok": np.asarray([h + n], np.int32),
        "n_mem": np.asarray([na], np.uint16),
        "n_priv": np.zeros(1, np.int32),
        "n_pool": np.zeros(1, np.int32),
        "n_ppool": np.zeros(1, np.int32),
        "sctx_lo": np.asarray([h], np.uint16),
        "n_sctx": np.asarray([n], np.uint16),
        "t_src": np.zeros(1, np.uint8),
        "cat": cat,
        "cont": cont,
        "sec": np.r_[head, tok.sec].astype(np.int8),
        "tok_flags": flags,
        "mem_tok": acts.astype(np.uint16),
        "mem_class": cls.astype(np.uint16),
        "cls_tok": acts[reps].astype(np.uint16),
        "cls_kind": np.asarray(rec["kinds"], np.uint8)[reps],
        "tprob": np.full(nc, np.nan, np.float32),
        "pcat": np.zeros((0, C.N_CAT), np.int16),
        "pcont": np.zeros((0, C.N_CONT), np.float16),
        "psec": np.zeros(0, np.int8),
        "pool_tok": z,
        "pool_vid": z,
        "pool_w": np.zeros(0, np.float16),
        "ppool_tok": z,
        "ppool_vid": z,
        "ppool_w": np.zeros(0, np.float16),
    }


@needs_r2
def test_teachers_on_the_wp3b_tokenizers_rows(recorded):
    """X-C and T8 on WP3b's real rows (`ppo.e2e.tokens`): the combat teacher's
    log-probs through the row slice equal its log-probs on #1081's own
    encoding bit for bit, the rows' classes are the env's, and the student
    scores the same rows."""
    from ppo.e2e import tokens as TK

    t = recorded["combat"]
    cbt, runs = recorded["cbt"], recorded["runs"]
    rb = TK.row_batch([r["row"] for r in cbt + runs])
    C.validate_row_batch(rb)
    got = t.logp_rows(rb)  # the combat rows, in order
    want = t.logp(
        [r["tok"] for r in cbt], [r["classes"] for r in cbt], [r["n_classes"] for r in cbt]
    )
    assert len(got) == len(cbt)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(a, b)
    for p in t.tprob_rows(rb):
        assert abs(float(p.sum()) - 1.0) < 1e-6  # T8 on combat rows
    off = C.row_offsets(rb)["mem"]
    for i, r in enumerate(cbt + runs):
        assert rb["mem_class"][off[i] : off[i + 1]].tolist() == list(r["classes"])
    # T8 on run rows: the run teacher's classes are these rows' classes.
    lp = recorded["run"].logp(
        [r["feat"] for r in runs], [r["classes"] for r in runs], [r["n_classes"] for r in runs]
    )
    nc = rb["n_classes"].astype(np.int64)
    for x, n in zip(lp, nc[len(cbt) :], strict=True):
        assert len(x) == n and abs(float(np.exp(x).sum()) - 1.0) < 1e-9
    net = N.E2ENet(N.NetConfig(dim=64, n_layers=1, ffn=176, head_hidden=64), seed=0).eval()
    lg = net.policy(N.to_batch(rb))
    assert torch.equal(torch.isfinite(lg).sum(1), torch.as_tensor(nc))
    # A RareMap touches only the S_ctx content slots, which the student never
    # reads: rows encoded without one, or with one that sends every content id
    # to UNK, give bit-identical student logits.
    plain = TK.row_batch([r["row_plain"] for r in cbt + runs])
    unk = TK.row_batch([r["row_unk"] for r in cbt + runs])
    assert (unk["cat"][:, : C.SCTX_N_CAT] != plain["cat"][:, : C.SCTX_N_CAT]).any()
    assert np.array_equal(unk["cat"][:, C.SCTX_N_CAT :], plain["cat"][:, C.SCTX_N_CAT :])
    for other in (plain, unk):
        assert torch.equal(net.policy(N.to_batch(other)), lg)


@needs_r2
def test_xc_slice_of_contract_rows_reproduces_the_teacher(recorded):
    """X-C on `contracts` rows: the section mask and column slice give back
    #1081's tokens bit for bit, so the teacher's merged log-probs over a row
    batch equal its log-probs on its own encoding; the student reads the same
    rows."""
    t = recorded["combat"]
    recs = [r for r in recorded["cbt"] if r["n_classes"] >= 2]
    rb = N.concat_rows([_combat_row(r) for r in recs])
    C.validate_row_batch(rb)
    views = TE.sctx_rows(rb)
    assert [v.row for v in views] == list(range(len(recs)))
    for v, r in zip(views, recs, strict=True):
        ref = C.teacher_slice(rb, v.row)  # the contract's one definition of X-C
        for k in ("cat", "cont", "sec"):
            assert np.array_equal(getattr(v.tokens, k), getattr(r["tok"], k)), k
            assert np.array_equal(getattr(v.tokens, k), ref[k]), k
        assert v.tokens.n_act == r["tok"].n_act == ref["n_act"]
        assert v.n_classes == r["n_classes"]
    got = t.logp_rows(rb)
    want = t.logp(
        [r["tok"] for r in recs], [r["classes"] for r in recs], [r["n_classes"] for r in recs]
    )
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(a, b)
    for v, p in zip(views, t.tprob_rows(rb), strict=True):
        assert p.dtype == np.float32 and abs(float(p.sum()) - 1.0) < 1e-6
        # The contract's own merge (SCHEMA.md section 3.1) gives the same row.
        ref = C.merge_probs(t.logits([v.tokens])[0], v.classes, v.n_classes)
        np.testing.assert_allclose(p, ref, rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        TE.sctx_rows(dict(rb, is_combat=np.zeros_like(rb["is_combat"])), [0])
    # The student consumes the same rows (hidden members dropped).
    net = N.E2ENet(N.NetConfig(dim=64, n_layers=1, ffn=176, head_hidden=64), seed=0).eval()
    b = N.to_batch(rb)
    lg = net.policy(b)
    nc = torch.as_tensor(rb["n_classes"].astype(np.int64))
    assert torch.equal(torch.isfinite(lg).sum(1), nc)
    assert t.stamps()["rare_sha256"] == TE.rare_sha256(t.rare_json)


# ── T8 on the final R3 pair, the campaign's teacher ───────────────────────


@pytest.mark.slow
@needs_r3
def test_t8_on_the_final_r3_pair():
    """T8 on the teacher the campaign launches with (PREREG-0 amendment 2), over
    ~3,000 stored decisions of 8 campaign runs played greedy over merged classes:
    at every decision the teacher's option count is the env's action count and
    class map length (the identity option map), the merged class distribution sums
    to 1, each class is the sum of its members' probabilities, and the learner's
    row path (`logp_rows` on the tokenizer row) gives the combat teacher's
    distribution exactly. Both teachers see env-merged rows; a non-bijective option
    map is refused."""
    from ppo.e2e import tokens as TK
    from ppo.e2e.env import E2EDecision, E2EEnv

    pair = _keep_threads(TE.load_pair, "R3")
    comb_t, run_t = pair.combat, pair.run
    worst = {"sum_dev": 0.0, "member_dev": 0.0, "row_path_dev": 0.0}
    n = dict.fromkeys(("rows", "combat", "ooc", "ooc_merged", "combat_merged"), 0)
    n |= {"len_mismatch": 0, "row_path_unequal": 0}
    for s in range(8):
        env = E2EEnv(1_300_000_200 + s)
        assert env.rm.game_version == GAME_VERSION
        d = env.reset()
        while isinstance(d, E2EDecision) and n["rows"] < 3_000:
            m = np.asarray(d.classes, np.int64)
            ncls = len(d.members)
            n["rows"] += 1
            if d.kind == "combat":
                tok = comb_t.encode(d)
                n["len_mismatch"] += not (tok.n_act == len(d.actions) == len(m))
                z = comb_t.logits([tok])[0].astype(np.float64)
                lp = comb_t.logp([tok], [m], [ncls])[0]
                row = TK.encode(d, rare=comb_t.rare, priv=False)
                (lp_row,) = comb_t.logp_rows(TK.row_batch([row]), [0])
                worst["row_path_dev"] = max(worst["row_path_dev"], float(np.abs(lp_row - lp).max()))
                n["row_path_unequal"] += not np.array_equal(lp_row, lp)
                n["combat"] += 1
                n["combat_merged"] += ncls < len(m)
            else:
                f = run_t.featurize_decision(d)
                n["len_mismatch"] += not (f.n_actions == len(d.actions) == len(m))
                z = run_t.logits([f])[0].astype(np.float64)
                lp = run_t.logp([f], [m], [ncls])[0]
                n["ooc"] += 1
                n["ooc_merged"] += ncls < len(m)
            p_opt = np.exp(z - np.logaddexp.reduce(z))
            worst["sum_dev"] = max(worst["sum_dev"], abs(float(np.exp(lp).sum()) - 1.0))
            ref = np.bincount(m, weights=p_opt, minlength=ncls)
            worst["member_dev"] = max(worst["member_dev"], float(np.abs(np.exp(lp) - ref).max()))
            d = env.step(int(np.argmax(lp)))
        if n["rows"] >= 3_000:
            break
    assert n["len_mismatch"] == 0, n
    assert n["combat"] > 1_000 and n["ooc"] > 100, n
    assert n["combat_merged"] > 0 and n["ooc_merged"] > 0, n  # both merges exercised
    assert worst["sum_dev"] < 1e-9 and worst["member_dev"] < 1e-9, worst
    assert n["row_path_unequal"] == 0, (n, worst)
    for bad, nc in (([0, 0], 2), ([0, 1, 2], 2)):
        with pytest.raises(TE.OptionMapError):
            TE.check_option_map(np.asarray(bad), nc)
