"""Serving-time distillation (GH #1197, `ScheduleCfg.distill_source` "acting").

A teacher trained without the "chars" fill event reads chars-off rows: each worker runs
a chars-off shadow env beside the distill characters' segments (`rsenv.RsE2EEnv`), the
learner scores its rows at serving time (`distill.DistillScorer`) into the stored
rows' `tprob`, and the update reads them (no teacher in the rank).
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sts2_client
import test_e2e_distill as TD  # _base, _learner, _resume, _weights
import test_learner as TL
import torch
from ppo.e2e import contracts as C
from ppo.e2e import distill as DS
from ppo.e2e import env as E
from ppo.e2e import factory
from ppo.e2e import learner as L
from ppo.e2e import ops_contract as OC
from ppo.e2e import schedules as S
from ppo.e2e import tokens as TK
from ppo.e2e import update as U
from ppo.e2e.net import (
    E2ENet,
    NetConfig,
    StudentPolicy,
    TokenizerRows,
    synthetic_row,
    to_batch,
)
from ppo.e2e.rowstore import Chunk, take
from ppo.e2e.rsenv_lockstep import tok_diff

# Four of these drive the scripted learner lane, which is Linux-only (see test_learner.py).
pytestmark = pytest.mark.skipif(
    'sys.platform == "win32"',
    reason="the learner ships rows through POSIX shared memory; the lane runs on Linux only.",
)

_v111 = TL._v111  # the autouse fixture: the stamps need v0.111.0

#: The 791/1000 Ironclad net (chars off), where the laptop has it.
IC791 = Path.home() / "runs/e2e-allchar/init/ic_d6750152301_ema0999.pt"
IC791_SHA = "26789e57b1079a6524a6a71c7fc6afe0bd02ee2c43c04fbb2d7d4f628d97ffa3"
SEED = 1_300_000_000


# ── config and the input guard ───────────────────────────────────────────


def test_distill_source_validates_and_an_older_state_loads_update() -> None:
    assert S.DEFAULT.distill_source == "update" and not S.DEFAULT.distill_acting
    with pytest.raises(ValueError, match="distill_source"):
        S.ScheduleCfg(distill_source="worker")
    with pytest.raises(ValueError, match="distill_flush_rows"):
        S.ScheduleCfg(distill_flush_rows=0)
    on = dict(distill_ckpt="/t.pt", distill_sha256="ab", distill_lambda=0.1)
    assert S.ScheduleCfg(**on, distill_source="acting").distill_acting
    assert not S.ScheduleCfg(**on).distill_acting
    assert not S.ScheduleCfg(
        **(on | {"distill_lambda": 0.0}), distill_source="acting"
    ).distill_acting
    from ppo.e2e import algo as A

    st = A.AlgoState(S.ScheduleCfg(**on, distill_source="acting"), n_types=A.N_TYPES).to_dict()
    assert A.AlgoState.from_dict(st).cfg.distill_source == "acting"
    del st["cfg"]["distill_source"], st["cfg"]["distill_flush_rows"]  # an 8e8996ce3 state
    del st["cfg"]["distill_rare"], st["cfg"]["distill_rare_sha256"]
    assert A.AlgoState.from_dict(st).cfg.distill_source == "update"
    assert A.AlgoState.from_dict(st).cfg.distill_rare is None
    with pytest.raises(ValueError, match="together"):
        S.ScheduleCfg(**on, distill_source="acting", distill_rare="/r.pt")
    with pytest.raises(ValueError, match="acting"):
        S.ScheduleCfg(**on, distill_rare="/r.pt", distill_rare_sha256="11")


def test_distill_rare_is_pinned_by_its_map_sha(tmp_path: Path) -> None:
    from ppo.e2e.teachers import rare_sha256

    rare = {"monster": [0, 1, 1]}
    good = tmp_path / "r.pt"
    torch.save({"rare": rare, "arm": "S_ctx"}, good)
    torch.save({"arm": "S_ctx"}, tmp_path / "none.pt")
    on = dict(distill_ckpt="/t.pt", distill_sha256="ab", distill_lambda=0.1)
    sha = rare_sha256(rare)
    cfg = S.ScheduleCfg(
        **on, distill_source="acting", distill_rare=str(good), distill_rare_sha256=sha
    )
    assert U.load_distill_rare(cfg) == rare
    assert U.load_distill_rare(S.ScheduleCfg(**on, distill_source="acting")) is None
    bad = dataclasses.replace(cfg, distill_rare_sha256="0" * 64)
    with pytest.raises(ValueError, match="rare_sha256"):
        U.load_distill_rare(bad)
    with pytest.raises(ValueError, match="no RareMap"):
        U.load_distill_rare(dataclasses.replace(cfg, distill_rare=str(tmp_path / "none.pt")))


@pytest.mark.parametrize(
    ("teacher", "lane", "source", "ok"),
    [
        ((), (), "update", True),
        (("chars",), ("chars",), "update", True),
        ((), ("chars",), "update", False),  # the 791 -> 728 case
        (("chars",), (), "update", False),
        ((), ("chars",), "acting", True),
        ((), (), "acting", True),
        (("chars",), ("chars",), "acting", False),
    ],
)
def test_teacher_input_guard(teacher: tuple, lane: tuple, source: str, ok: bool) -> None:
    t = ("WP-ENG-1", "content-prior", *teacher)
    if ok:
        U.check_distill_teacher_inputs(t, lane, source)
    else:
        with pytest.raises(ValueError, match="chars"):
            U.check_distill_teacher_inputs(t, lane, source)


@pytest.mark.skipif(not IC791.is_file(), reason="needs the 791 Ironclad checkpoint")
def test_the_791_checkpoint_is_a_chars_off_teacher() -> None:
    assert "chars" not in U.checkpoint_filled(str(IC791))
    cfg = S.ScheduleCfg(distill_ckpt=str(IC791), distill_sha256=IC791_SHA, distill_lambda=0.1)
    with pytest.raises(ValueError, match="acting"):
        U.check_distill_cfg(cfg, lane_filled=["WP-ENG-1", "chars", "content-prior"])
    acting = S.ScheduleCfg(
        distill_ckpt=str(IC791),
        distill_sha256=IC791_SHA,
        distill_lambda=0.1,
        distill_source="acting",
    )
    U.check_distill_cfg(acting, lane_filled=["WP-ENG-1", "chars", "content-prior"])


# ── the shadow env (real engine) ─────────────────────────────────────────


@pytest.fixture
def chars_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process whose Ironclad rows carry "chars" (the all-character lane's)."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    monkeypatch.setenv(factory.CHARS_ENV, "1")


def _shadow_env(seed: int, *, rm: Any = None, segment: E.Segment = E.RUN, **kw: Any) -> Any:
    from ppo.e2e.rsenv import RsE2EEnv

    cfg = E.EnvConfig(rust=True, distill_shadow=("Ironclad",), **kw)
    return RsE2EEnv(seed, rm=rm, cfg=cfg, segment=segment)


def _play(env: Any, rng: np.random.Generator, rows: list | None = None, hook: Any = None) -> Any:
    """Random play to the end; the terminal. `hook(env)` runs after the reset."""
    d = env.reset()
    if hook is not None:
        hook(env)
    while hasattr(d, "n_classes"):
        if rows is not None:
            rows.append(d)
        if d.bootstrap:
            return env.terminal
        d = env.step(int(rng.integers(d.n_classes)))
    return d


def test_shadow_rows_ride_every_ironclad_row_and_practice_starts(chars_lane: None) -> None:
    """Full runs (F9 on, captures on, pinned so they go deep) and practice starts from
    their captures: every non-bootstrap row carries the shadow's row, none desyncs, the
    shadow's rows have no X_CHAR and differ from the chars rows only in combat."""
    from ppo.e2e.actionlog import start_from_capture
    from ppo.e2e.practice import segment_for

    caps: list[E.Capture] = []
    rows: list[Any] = []
    n_combat_diff = 0
    for seed in (SEED, SEED + 1):
        rm = E.make_run_manager(seed)
        rm.set_player(hp=100_000, max_hp=100_000)
        env = _shadow_env(seed, rm=rm, f9_p=0.2, on_capture=caps.append)
        mine: list[Any] = []
        term = _play(env, np.random.default_rng(seed), mine)
        assert term is not None and not str(term.outcome).startswith("error"), term
        c = term.counts
        assert c["distill/desync"] == 0 and c["distill/shadow_error"] == 0
        assert c["distill/shadow_rows"] == sum(not d.bootstrap for d in mine) > 100
        rows += mine
    assert caps, "the pinned runs reach elites and bosses"
    rng = np.random.default_rng(5)
    for j, cap in enumerate(caps[:4]):
        reseed = 600_000_000 + j
        env = _shadow_env(
            cap.seed, rm=start_from_capture(cap, reseed), segment=segment_for(cap, reseed)
        )
        prow: list[Any] = []
        term = _play(env, rng, prow)
        assert term.counts["distill/desync"] == 0
        rows += prow
    for d in rows:
        if d.bootstrap:
            assert d.tok_t is None
            continue
        t = d.tok_t
        assert t is not None and t.n_classes == d.n_classes
        assert not t.cat[:, C.XC["X_CHAR"]].any()
        if not d.is_combat:
            assert tok_diff(t, d.tok, priv=False) in ([], ["cat"])
        elif tok_diff(t, d.tok, priv=False):
            n_combat_diff += 1
    assert n_combat_diff > 0  # the chars rows really are different inputs


def test_no_shadow_for_other_characters(chars_lane: None) -> None:
    from ppo.e2e.rsenv import RsE2EEnv

    cfg = E.EnvConfig(rust=True, distill_shadow=("Silent",))
    env = RsE2EEnv(SEED, cfg=cfg)
    d = env.reset()
    assert not env.has_shadow and d.tok_t is None
    with pytest.raises(ValueError, match="Rust env"):
        E.new_env(SEED, cfg=E.EnvConfig(distill_shadow=("Ironclad",)))


class _Broken:
    """A shadow that raises (`mode` "raise") or desyncs ("desync") at its `at`-th step."""

    def __init__(self, inner: Any, mode: str, at: int) -> None:
        self.inner, self.mode, self.at, self.k = inner, mode, at, 0

    def reset(self) -> Any:
        return self.inner.reset()

    def step(self, c: int) -> Any:
        self.k += 1
        r = self.inner.step(c)
        if self.k == self.at:
            if self.mode == "raise":
                raise RuntimeError("shadow boom")
            r = dict(r, n_classes=int(r.get("n_classes", 0)) + 1)
        return r


@pytest.mark.parametrize(("mode", "count"), [("raise", "shadow_error"), ("desync", "desync")])
def test_a_broken_shadow_is_dropped_never_the_segment(
    chars_lane: None, mode: str, count: str
) -> None:
    """The segment plays on exactly as without a shadow; from the break on, its rows
    carry no teacher row; the break is counted."""
    ref = _shadow_env(SEED)
    ref_rows: list[Any] = []
    ref_term = _play(ref, np.random.default_rng(1), ref_rows)
    env = _shadow_env(SEED)
    rows: list[Any] = []

    def brk(e: Any) -> None:
        e._shadow = _Broken(e._shadow, mode, at=10)

    term = _play(env, np.random.default_rng(1), rows, hook=brk)
    assert (term.outcome, term.rows, term.reward) == (
        ref_term.outcome,
        ref_term.rows,
        ref_term.reward,
    )
    assert len(rows) == len(ref_rows) > 10
    assert all(r.tok_t is not None for r in rows[:10])
    assert all(r.tok_t is None for r in rows[10:]) and not env.has_shadow
    assert term.counts[f"distill/{count}"] == 1


# ── parity with the eval path (the requested check) ─────────────────────


def _teacher() -> E2ENet:
    if IC791.is_file() and os.environ.get("STS2_DISTILL_REAL_TEACHER", "1") == "1":
        from ppo.e2e.net import load_checkpoint

        net, _ = load_checkpoint(str(IC791))
    else:
        net = E2ENet(NetConfig.from_dict(TL.TINY), seed=3)
    net.eval()
    return net


#: The R3 combat teacher (#1081 S_ctx net) whose RareMap the 791 net's training rows
#: carried (its lane ran `--teacher-pair R3`), where this machine has it.
R3_RARE_SHA = "11014549dfb8e4e90b17c2cd6b9913eb9e3b79ed9db80e2018ceb215f3eaf5a9"


def _r3_rare() -> Any:
    from ppo.e2e.teachers import PINS, rare_sha256

    path = PINS["R3"]["combat"].path
    if not path.is_file():
        pytest.skip(f"needs the R3 combat checkpoint ({path})")
    rare = torch.load(path, map_location="cpu", weights_only=True)["rare"]
    assert rare_sha256(rare) == R3_RARE_SHA
    return rare


@pytest.mark.parametrize("with_rare", [False, True], ids=["universe", "r3_rare"])
def test_stored_teacher_probs_equal_the_eval_path_on_chars_off_rows(
    monkeypatch: pytest.MonkeyPatch, with_rare: bool
) -> None:
    """On real Ironclad decisions: the shadow's row is bit for bit the eval path's
    chars-off row (`TokenizerRows`: Python env, Python tokenizer, no "chars"; with
    `distill_rare`, through the same RareMap: the teacher's training rows), and the
    probs the serving-time scorer stores are the teacher's softmax on that row through
    `StudentPolicy`'s forward (fp32, CPU), its argmax `StudentPolicy`'s pick. The same
    teacher on the stored chars rows is not. With the R3 RareMap the rows differ (the
    S_ctx content slots), the probs do not: the net never reads those slots."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    rare_py = rare_rs = None
    if with_rare:
        import sts2_rs
        from ppo.combat_ctx.data import RareMap

        rare_json = _r3_rare()
        rare_py, rare_rs = RareMap.from_json(rare_json), sts2_rs.E2eRareMap(rare_json)
    net = _teacher()
    sp = StudentPolicy(net, device="cpu", bf16=False, encode_row=TokenizerRows(rare=rare_py))
    seed = SEED + 7
    rm = E.make_run_manager(seed)
    rm.set_player(hp=100_000, max_hp=100_000)  # random play goes deep
    monkeypatch.setenv(factory.CHARS_ENV, "1")
    # The main env reads the flag once, here.
    rs = _shadow_env(seed, rm=rm.clone(), distill_rare=rare_rs)
    monkeypatch.delenv(factory.CHARS_ENV)  # the Python tokenizer reads it at encode
    py = E.E2EEnv(seed, rm=rm.clone(), cfg=E.EnvConfig())
    rng = np.random.default_rng(11)
    dr, dp = rs.reset(), py.reset()
    shadow, main, ref_rb, picks, universe = [], [], [], [], []
    while hasattr(dr, "n_classes") and not dr.bootstrap and len(shadow) < 400:
        assert dp.n_classes == dr.n_classes and dp.is_combat == dr.is_combat
        ref = TK.encode(dp, rare=rare_py, priv=False)
        if with_rare:
            universe.append(TK.encode(dp, rare=None, priv=False))
        assert tok_diff(dr.tok_t, ref, priv=False) == [], len(shadow)
        shadow.append(dr.tok_t)
        main.append(dr.tok)
        ref_rb.append(sp.row(dp))
        picks.append(sp.act_decisions([dp])[0])
        c = int(rng.integers(dr.n_classes))
        dr, dp = rs.step(c), py.step(c)
    assert sum(t.is_combat for t in shadow) > 50 and sum(not t.is_combat for t in shadow) > 10
    got = DS.teacher_probs(net, TK.row_batch(shadow), "cpu")
    with torch.no_grad():
        from ppo.e2e.net import concat_rows

        lg = net.policy(to_batch(concat_rows(ref_rb), "cpu"), bf16=False).float()
    worst = 0.0
    for k, t in enumerate(shadow):
        n = t.n_classes
        want = torch.softmax(lg[k, :n], 0).numpy()
        np.testing.assert_allclose(got[k, :n], want, atol=1e-5)
        assert not got[k, n:].any()
        worst = max(worst, float(np.abs(got[k, :n] - want).max()))
        assert abs(float(got[k, :n].sum()) - 1.0) < 1e-5
        top = np.flatnonzero(got[k, :n] >= got[k, :n].max() - 1e-6)
        assert picks[k] in top
    # The negative control: the stored (chars) rows give other probs in combat.
    on = DS.teacher_probs(net, TK.row_batch(main), "cpu")
    diffs = [
        float(np.abs(on[k, : t.n_classes] - got[k, : t.n_classes]).max())
        for k, t in enumerate(shadow)
        if t.is_combat
    ]
    if IC791.is_file() and os.environ.get("STS2_DISTILL_REAL_TEACHER", "1") == "1":
        assert max(diffs) > 1e-3, max(diffs)
    if with_rare:
        # The map moves content-slot ids on some combat rows (the shadow applies it) ...
        urb, srb = TK.row_batch(universe), TK.row_batch(shadow)
        moved = np.flatnonzero((urb["cat"] != srb["cat"]).any(0))
        assert len(moved) and set(moved.tolist()) <= {
            i for i, s_ in enumerate(C.SCTX_SLOTS) if s_ in C.SCTX_CONTENT_SLOTS
        }, moved
        # ... and the net, which reads no content slot, gives the same probs.
        np.testing.assert_array_equal(DS.teacher_probs(net, urb, "cpu"), got)


# ── the learner (scripted rows) ──────────────────────────────────────────


class ShadowSource(TL.CharSource):
    """`CharSource` whose Ironclad rows come with a "shadow" row: the stored row itself
    (so "acting" must match "update"), every `drop`-th one missing."""

    drop = 0

    def recv(self, w: int) -> Any:
        msg = super().recv(w)
        rb, ex = msg[4], msg[5]
        ic = ~(rb["run_uid"] % 2).astype(bool) & ~rb["bootstrap"].astype(bool)
        idx = np.flatnonzero(ic)
        if self.drop:
            idx = idx[(rb["t"][idx] % self.drop) != 0]
        if len(idx):
            ex["distill"] = {"idx": idx, "rb": take(Chunk(rb), idx)}
        return msg


def _acting_learner(cfg: L.Cfg, drop: int = 0) -> L.Learner:
    clock = TL.Clock()

    def src(wcfg: dict[str, Any]) -> Any:
        s = ShadowSource(wcfg)
        s.drop = drop
        return s

    lr = L.Learner(cfg, source=src, updater="local", clock=clock)
    clock.learner = lr
    return lr


def _resume_acting(
    tmp_path: Path, src: Path, name: str, sched: dict[str, Any], *, drop: int = 0, **kw: Any
) -> L.Learner:
    d = tmp_path / name
    shutil.copytree(src, d)
    (d / OC.DONE).unlink(missing_ok=True)
    cfg = TL.make_cfg(d, max_iters=7, resume=str(d / "state.pt"), sched=sched, rust_env=True, **kw)
    lr = _acting_learner(cfg, drop)
    assert lr.train() == "max_iters"
    return lr


def test_acting_matches_update_when_the_shadow_rows_are_the_stored_rows(tmp_path: Path) -> None:
    """With the shadow rows equal to the stored ones, "acting" (probs scored at serving
    time, stored in the rows) trains as "update" (the rank's forward on the same rows),
    up to float rounding; the rank loads no teacher; the stats carry the scorer's."""
    src, teacher, sha = TD._base(tmp_path)
    d = {"distill_ckpt": str(teacher), "distill_sha256": sha, "distill_lambda": 0.5}
    upd = TD._resume(tmp_path, src, "upd", dict(TL.SCHED, **d))
    act = _resume_acting(
        tmp_path, src, "act", dict(TL.SCHED, **d, distill_source="acting", distill_flush_rows=16)
    )
    assert act.updater.rank.distill is None and act.distill_scorer is not None
    ua = [r["update"] for r in act.stats_rows if r["update"]["distill_rows"] > 0]
    uu = [r["update"] for r in upd.stats_rows if r["update"]["distill_rows"] > 0]
    assert ua and len(ua) == len(uu)
    for a, u in zip(ua, uu, strict=True):
        assert a["distill_source"] == "acting" and a["distill_rows"] == u["distill_rows"]
        assert a["distill_rows_missing"] == 0
        assert a["distill_kl"] == pytest.approx(u["distill_kl"], rel=1e-4, abs=1e-6)
    wa, wu = TD._weights(act), TD._weights(upd)
    for k in wu:
        torch.testing.assert_close(wa[k], wu[k], rtol=1e-3, atol=1e-4)
    for r in act.stats_rows:
        assert r["distill"]["rows_scored"] > 0 and r["distill_desyncs"] == 0
        assert r["distill"]["mismatch"] == 0 and r["distill"]["serve_s"] >= 0
    assert "distill" not in upd.stats_rows[-1]
    cj = (tmp_path / "act" / "cfg.json").read_text(encoding="utf-8")
    assert '"distill_source": "acting"' in cj


def test_acting_rows_without_a_shadow_train_without_a_teacher(tmp_path: Path) -> None:
    src, teacher, sha = TD._base(tmp_path)
    d = {"distill_ckpt": str(teacher), "distill_sha256": sha, "distill_lambda": 0.5}
    lr = _resume_acting(tmp_path, src, "drop", dict(TL.SCHED, **d, distill_source="acting"), drop=3)
    on = [r["update"] for r in lr.stats_rows if r["update"]["distill_rows"] > 0]
    assert on and all(u["distill_rows_missing"] > 0 for u in on)
    assert all(u["n_ks_missing"] == 0 for u in on)


def test_acting_on_a_character_without_rows_is_bit_identical_to_off(tmp_path: Path) -> None:
    """The scorer writes the Ironclad rows' tprob, but distilling only Silent touches no
    row: the weights end bit-identical to distillation off."""
    src, teacher, sha = TD._base(tmp_path)
    off = TD._weights(TD._resume(tmp_path, src, "off", dict(TL.SCHED)))
    d = {"distill_ckpt": str(teacher), "distill_sha256": sha, "distill_lambda": 0.5}
    lr = _resume_acting(
        tmp_path,
        src,
        "silent",
        dict(TL.SCHED, **d, distill_source="acting", distill_chars=["Silent"]),
    )
    w = TD._weights(lr)
    assert all(torch.equal(w[k], off[k]) for k in off)
    assert any(r["distill"]["rows_scored"] > 0 for r in lr.stats_rows)


def test_learner_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The lane build, so the run starts reach the refusals on every test build.
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    src, teacher, sha = TD._base(tmp_path)
    d = {"distill_ckpt": str(teacher), "distill_sha256": sha, "distill_lambda": 0.5}
    # "acting" needs the Rust env.
    x = tmp_path / "py"
    shutil.copytree(src, x)
    (x / OC.DONE).unlink(missing_ok=True)
    cfg = TL.make_cfg(
        x,
        max_iters=8,
        resume=str(x / "state.pt"),
        sched=dict(TL.SCHED, **d, distill_source="acting"),
    )
    with pytest.raises(L.UsageError, match="rust-env"):
        _acting_learner(cfg)
    # A chars-off teacher on a "chars" lane through the update: the 791 -> 728 case.
    y = tmp_path / "chars"
    cfg = TL.make_cfg(y, max_iters=2, filled=["chars"], sched=dict(TL.SCHED, **d))
    with pytest.raises(L.UsageError, match="distill_source"):
        _acting_learner(cfg)
    # ... and a "chars" teacher scored on chars-off shadow rows.
    monkeypatch.setattr(U, "checkpoint_filled", lambda _p: ("chars",))
    z = tmp_path / "z"
    cfg = TL.make_cfg(
        z,
        max_iters=2,
        filled=["chars"],
        rust_env=True,
        sched=dict(TL.SCHED, **d, distill_source="acting"),
    )
    with pytest.raises(L.UsageError, match="distill_source"):
        _acting_learner(cfg)
    monkeypatch.undo()
    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    # "acting" beside a combat or run teacher: they would share the rows' teacher probs.
    monkeypatch.setattr(L.Learner, "_teacher_flags", lambda self: (True, False))
    w = tmp_path / "w"
    cfg = TL.make_cfg(
        w,
        max_iters=2,
        filled=["chars"],
        rust_env=True,
        sched=dict(TL.SCHED, **d, distill_source="acting"),
    )
    with pytest.raises(L.UsageError, match="teacher-pair"):
        _acting_learner(cfg)


def _scorer_rows(n: int = 4) -> Chunk:
    rows = [
        synthetic_row(np.random.default_rng(i), combat=True, t=i, body=(2, 6), priv=(0, 0))
        for i in range(n)
    ]
    from ppo.e2e.rowstore import concat

    rb = concat(rows)
    rb["run_uid"][:] = 7
    return Chunk(rb)


def test_scorer_survives_a_failing_teacher_and_bad_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward that raises (a CUDA OOM, say) costs only its chunk's rows; a payload
    that is not the round's is dropped whole; `discard` drops the buffer and counters."""
    ch = _scorer_rows(6)
    net = E2ENet(NetConfig.from_dict(TL.TINY), seed=1).eval()
    sc = DS.DistillScorer(net, "cpu", flush_rows=2)
    real = DS.teacher_probs
    calls = {"n": 0}

    def flaky(net_: Any, rb: Any, dev: Any) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return real(net_, rb, dev)

    monkeypatch.setattr(DS, "teacher_probs", flaky)
    sc.add(ch, 0, 6, {"idx": np.arange(6), "rb": take(ch, np.arange(6))})
    assert sc.pending == 6
    assert sc.flush() == 4  # three chunks of 2; the second raised
    assert calls["n"] == 3
    assert (ch.rb["t_src"] != 0).tolist() == [True, True, False, False, True, True]
    assert np.isnan(ch.rb["tprob"][ch.span("cls", 2)]).all()
    st = sc.take_stats()
    assert st["score_errors"] == 1 and st["score_error_rows"] == 2 and st["rows_scored"] == 4
    # Bad payloads: out of range, duplicate, missing keys, wrong length, not a dict.
    ch2 = _scorer_rows(3)
    ok_rb = take(ch2, np.arange(2))
    short = dict(ok_rb)
    short["t"] = short["t"][:1]
    for payload in (
        {"idx": np.array([0, 3]), "rb": ok_rb},
        {"idx": np.array([1, 1]), "rb": ok_rb},
        {"idx": np.array([0, 1]), "rb": {k: v for k, v in ok_rb.items() if k != "cat"}},
        {"idx": np.array([0, 1]), "rb": short},
        {"idx": np.array([0.0, 1.0]), "rb": ok_rb},
        {"rb": ok_rb},
        None,
    ):
        sc.add(ch2, 0, 3, payload)
    assert sc.pending == 0
    assert sc.take_stats()["bad_payload"] == 7
    # The message's rows sit at `base` in the round: indices are the message's.
    sc.add(ch2, 1, 2, {"idx": np.array([0, 1]), "rb": take(ch2, np.array([1, 2]))})
    assert sc.pending == 2
    assert sc.discard() == 2 and sc.pending == 0 and sc.flush() == 0
    assert not (ch2.rb["t_src"] != 0).any()
    assert sc.take_stats()["rows_scored"] == 0


def test_scorer_drops_rows_that_are_not_their_decision() -> None:
    ch = _scorer_rows(4)
    net = E2ENet(NetConfig.from_dict(TL.TINY), seed=1).eval()
    sc = DS.DistillScorer(net, "cpu", flush_rows=100)
    bad = take(ch, [0, 1, 2])
    bad["t"][1] = 99  # not row 1's decision
    sc.add(ch, 0, 4, {"idx": np.array([0, 1, 2]), "rb": bad})
    assert sc.pending == 2
    assert sc.flush() == 2
    assert (ch.rb["t_src"] != 0).tolist() == [True, False, True, False]
    assert np.isnan(ch.rb["tprob"][ch.span("cls", 1)]).all()
    p = ch.rb["tprob"][ch.span("cls", 0)]
    assert abs(float(p.sum()) - 1) < 1e-5 and ch.rb["t_src"][0] == 1
    C.validate_row_batch(ch.rb)
    assert sc.take_stats()["mismatch"] == 1
    assert math.isfinite(float(p[0]))


# ── the worker ───────────────────────────────────────────────────────────


class _Conn:
    """A learner stand-in: random acts for `rounds` rounds; keeps every message."""

    def __init__(self, rounds: int) -> None:
        self.rounds, self.r = rounds, 0
        self.rng = np.random.default_rng(0)
        self.n: np.ndarray = np.zeros(0)
        self.msgs: list[Any] = []

    def send(self, msg: Any) -> None:
        self.msgs.append(msg)
        self.n = msg[2][:, 8]

    def recv(self) -> Any:
        self.r += 1
        if self.r > self.rounds:
            return ("stop",)
        acts = [int(self.rng.integers(max(1, int(n)))) for n in self.n]
        return ("act", acts, {"decisions": 4_000_000_000 + self.r})

    def close(self) -> None:
        pass


def _worker_msgs(
    monkeypatch: pytest.MonkeyPatch, shadow: list[str] | None, rare: Any = None
) -> list[Any]:
    import signal

    from ppo.e2e import actors

    monkeypatch.setattr(signal, "signal", lambda *a: None)
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(k, os.environ.get(k, "1"))
    conn = _Conn(40)
    cfg = {
        "n_envs": 8,
        "run_seed_base": SEED,
        "start_it": 0,
        "start_decisions": 4_000_000_000,
        "seed": 0,
        "practice": True,
        "f9_tolerate": [],
        "teacher_rare": None,
        "run_teacher_honest": False,
        "action_log_dir": None,
        "rust_env": True,
        "distill_shadow": shadow,
        "distill_rare": rare,
    }
    actors.worker_main(0, conn, cfg)
    return conn.msgs


def test_the_worker_sends_a_shadow_row_with_every_ironclad_row(
    chars_lane: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main rows are byte for byte the shadowless worker's; every non-bootstrap row
    has its chars-off shadow row in `extra["distill"]` (same run uid, t and classes)."""
    with_ = _worker_msgs(monkeypatch, ["Ironclad"])
    without = _worker_msgs(monkeypatch, None)
    assert len(with_) == len(without) > 10
    n_rows = 0
    for a, b in zip(with_, without, strict=True):
        ra, rb = a[4], b[4]
        assert all(np.array_equal(ra[k].view(np.uint8), rb[k].view(np.uint8)) for k in rb)
        assert "distill" not in b[5]
        nb = np.flatnonzero(~ra["bootstrap"].astype(bool))
        dx = a[5].get("distill")
        if not len(nb):
            continue
        assert dx is not None
        np.testing.assert_array_equal(dx["idx"], nb)
        srb = dx["rb"]
        for k in ("run_uid", "t", "n_classes", "is_combat", "dtype"):
            np.testing.assert_array_equal(srb[k], ra[k][nb])
        assert not srb["cat"][:, C.XC["X_CHAR"]].any()
        assert ra["cat"][:, C.XC["X_CHAR"]].any()
        C.validate_row_batch(srb)
        n_rows += len(nb)
    assert n_rows > 200


def test_the_worker_shadow_takes_the_distill_rare_map(
    chars_lane: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`distill_rare` reaches the shadow and nothing else: the main rows are unchanged,
    the shadow's combat rows are the universe shadow's with the map applied to the S_ctx
    content slots. The map here sends every content id to UNK, so it always moves."""
    from ppo.combat_ctx import tokens as CT
    from ppo.combat_ctx.data import RareMap

    v = CT.vocab()
    unk = {}
    for slot, m in _r3_rare().items():
        tab = v.tables[CT.S[slot]]
        a_ = np.asarray(m, np.int64)
        unk[slot] = np.where(a_ < tab.unk, tab.unk, a_).tolist()
    rmap = RareMap.from_json(unk)
    a = _worker_msgs(monkeypatch, ["Ironclad"], unk)
    b = _worker_msgs(monkeypatch, ["Ironclad"])
    content = {i for i, s_ in enumerate(C.SCTX_SLOTS) if s_ in C.SCTX_CONTENT_SLOTS}
    moved: set[int] = set()
    for x, y in zip(a, b, strict=True):
        assert all(np.array_equal(x[4][k].view(np.uint8), y[4][k].view(np.uint8)) for k in y[4])
        dx, dy = x[5].get("distill"), y[5].get("distill")
        assert (dx is None) == (dy is None)
        if dx is None:
            continue
        np.testing.assert_array_equal(dx["idx"], dy["idx"])
        for k in dy["rb"]:
            if k != "cat":
                assert np.array_equal(dx["rb"][k].view(np.uint8), dy["rb"][k].view(np.uint8)), k
        cx, cy = dx["rb"]["cat"], dy["rb"]["cat"]
        moved |= set(np.flatnonzero((cx != cy).any(0)).tolist())
        # Token by token: S_ctx tokens of combat rows are the universe ones mapped.
        sctx = (cy[:, : C.SCTX_N_CAT] != cx[:, : C.SCTX_N_CAT]).any(1)
        if sctx.any():
            want = rmap.apply(cy[sctx, : C.SCTX_N_CAT]).astype(cx.dtype)
            np.testing.assert_array_equal(cx[sctx, : C.SCTX_N_CAT], want)
    assert moved and moved <= content, moved


def test_the_scorer_validates_one_payload_in_validate_every(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`validate_row_batch` is a per-row Python loop: 63% of the learner at 80% Ironclad
    when it ran on every payload. It runs on the first payload and then one in
    `VALIDATE_EVERY`; the shape, index and decision checks still run on every one."""
    ch = _scorer_rows(2)
    net = E2ENet(NetConfig.from_dict(TL.TINY), seed=1).eval()
    sc = DS.DistillScorer(net, "cpu", flush_rows=8)
    calls = {"n": 0}
    real = DS.C.validate_row_batch

    def counted(rb: Any) -> None:
        calls["n"] += 1
        real(rb)

    monkeypatch.setattr(DS.C, "validate_row_batch", counted)
    payload = {"idx": np.arange(2), "rb": take(ch, np.arange(2))}
    for _ in range(2 * DS.DistillScorer.VALIDATE_EVERY + 2):
        sc.add(ch, 0, 2, payload)
    assert calls["n"] == 3
    # A bad payload is still dropped by the every-payload checks.
    sc.add(ch, 0, 2, {"idx": np.array([0, 5]), "rb": payload["rb"]})
    assert sc.take_stats()["bad_payload"] == 1


def test_scorer_drops_the_potion_popup_from_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """GH #1606: the teacher never trained on kinds 54/55, so the target zeroes them and
    renormalizes over the rest; a row of only popup options gets no target."""
    ch = _scorer_rows(3)
    nc = ch.rb["n_classes"].astype(np.int64)
    assert (nc >= 2).all()
    off = np.concatenate(([0], np.cumsum(nc)))
    shadow = take(ch, np.arange(3))
    shadow["cls_kind"] = shadow["cls_kind"].copy()
    shadow["cls_kind"][off[1] - 1] = 55  # row 0: its last class is a discard
    shadow["cls_kind"][off[2] : off[3]] = 54  # row 2: only popup options

    def uniform(net_: Any, rb: Any, dev: Any) -> np.ndarray:
        n = rb["n_classes"].astype(np.int64)
        p = np.zeros((len(n), int(n.max())), np.float32)
        for j, k in enumerate(n):
            p[j, :k] = 1.0 / k
        return p

    monkeypatch.setattr(DS, "teacher_probs", uniform)
    sc = DS.DistillScorer(None, "cpu", flush_rows=100)  # type: ignore[arg-type]
    sc._payloads = 1  # skip the schema check: these synthetic rows are combat rows
    sc.add(ch, 0, 3, {"idx": np.arange(3), "rb": shadow})
    assert sc.pending == 3, sc.stats
    assert sc.flush() == 2, sc.stats
    p0 = ch.rb["tprob"][ch.span("cls", 0)]
    assert not p0[-1]
    np.testing.assert_allclose(p0[:-1], 1.0 / (nc[0] - 1), rtol=1e-6)
    np.testing.assert_allclose(ch.rb["tprob"][ch.span("cls", 1)], 1.0 / nc[1], rtol=1e-6)
    assert ch.rb["t_src"][2] == 0
    st = sc.take_stats()
    assert st["masked_rows"] == 1 and st["masked_empty"] == 1
    assert round(st["ooc_rows"]) == 2
    assert abs(st["ooc_mass"] - (1.0 / nc[0] + 1.0)) < 1e-5
