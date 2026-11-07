"""E2ENet (plan §5) on `contracts` row batches: parameter counts, shapes,
PRIV isolation (T14 at the net level), hidden members, pools, inert reserved
inputs, precision (X-B tolerance), determinism, padding invariance, the valve
and checkpoint IO."""

from __future__ import annotations

import dataclasses
import os
import types

import numpy as np
import pytest
import torch
from ppo.e2e import contracts as C
from ppo.e2e import net as N
from ppo.e2e import vocab as V

VOCAB = "b" * 64


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    """Small CPU forwards on a shared, loaded host: many torch threads only
    contend. Restore the caller's count."""
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def _small_cfg(**kw):
    """A cheap net with the real input layout (for tests that don't need 6 x 256)."""
    base = dict(num_hidden=64, dim=64, n_layers=2, ffn=176, head_hidden=64)
    return N.NetConfig(**{**base, **kw})


def _rows(n=12, seed=0, **kw):
    return N.synthetic_row_batch(n, seed, **kw)


def _loud_heads(net, scale=3.0, seed=1):
    """Re-draw the policy and value output layers so logits and values are
    O(1) instead of ~0 (init) and the comparisons below mean something."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (net.head_combat, net.head_ooc):
            w = head.out.weight
            w.copy_(torch.randn(w.shape, generator=g) * scale / w.shape[1] ** 0.5)
        for head in (net.value.v_combat, net.value.v_run):
            head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.1)
            head.bias.fill_(0.05)
    return net


def _net(cfg=None, **kw):
    return _loud_heads(N.E2ENet(cfg or _small_cfg(), seed=0), **kw).eval()


def _row_slices(rb, i):
    """`{axis: slice}` of row `i`'s ragged ranges."""
    out = {}
    for axis, cnt in N.AXES.items():
        o = np.concatenate([[0], np.cumsum(rb[cnt].astype(np.int64))])
        out[axis] = slice(int(o[i]), int(o[i + 1]))
    return out


def _copy(rb):
    return {k: v.copy() for k, v in rb.items()}


def _pool_ranges(rb):
    return [_row_slices(rb, i)["pool"] for i in range(len(rb["n_tok"]))]


# ── parameters and the contract ───────────────────────────────────────────


def test_parameter_counts_match_plan_section_5():
    net = N.E2ENet(seed=0)
    parts = N.param_breakdown(net)
    print("\n" + N.format_breakdown(net))
    # 6 x (4·256² + 3·256·704 + 2·256) + 256: plan §5's 4.82M trunk.
    assert parts["trunk"] == 4_820_224
    assert parts["id_table"] == C.VOCAB_ROWS * 256
    assert parts["stems"] < 0.5e6 and parts["value_module"] < 0.9e6
    assert 7.2e6 < parts["total"] < 7.7e6
    # The B-T0 bench variant (4 x 256).
    assert N.param_breakdown(N.E2ENet(N.NetConfig(n_layers=4)))["trunk"] == 3_213_568


def test_layout_follows_the_contract():
    L = N.layout()
    assert (L.n_cat, L.n_cont, L.n_prior) == (C.N_CAT, C.N_CONT, C.N_PRIOR)
    assert N.AXES == C._AXIS_COUNT
    read = set(L.small_cols) | set(L.id_cols)
    content = {C.SCTX_SLOTS.index(s) for s in C.SCTX_CONTENT_SLOTS}
    assert not read & content  # the student never reads RareMap'd content slots
    from ppo.combat_ctx import tokens as T

    assert set(C.SCTX_CONTENT_SLOTS) == set(T.CONTENT_SLOTS)  # what RareMap.apply remaps
    assert read | content == set(range(C.N_CAT))  # and reads every other column
    reserved_cont = {C.XF[c.name] for c in C.X_CONT if c.status != "live"}
    assert set(L.zero_cont) == reserved_cont
    assert dict(zip((C.X_CAT[i - C.SCTX_N_CAT].name for i in L.id_cols), L.id_live)) == {
        "X_ID_MAIN": True,
        "X_ID_MOD": True,
        "X_ID_REL": True,
        "X_ID_AUX": False,
    }
    v = V.load()
    zero = set(N.zero_id_rows())
    assert zero == set(v.block("encounter")) | set(v.block("move")) and 0 not in zero
    assert max(zero) < C.VOCAB_ROWS and N.current_vocab_sha256() == v.sha256
    assert L.n_kind == C.N_ACTION_KIND and set(L.live_kinds) == set(C.LIVE_ACTION_KINDS)


def test_synthetic_rows_satisfy_the_contract():
    rb = _rows(48, seed=3, teacher=True, ppool=True)
    C.validate_row_batch(rb)
    C.check_compact_lossless(rb["cat"], rb["cont"], rb["sec"])
    # SCHEMA.md section 2.1's token order: RUNCTX, the summaries, BOSS.
    head = [C.SEC["RUNCTX"], *C.SUMMARY_SECTIONS, C.SEC["BOSS"]]
    for i in range(len(rb["n_tok"])):
        assert rb["sec"][_row_slices(rb, i)["tok"]][: len(head)].tolist() == head
    assert rb["is_combat"].any() and not rb["is_combat"].all()
    assert ((rb["tok_flags"] & C.TF_HIDDEN) != 0).any()
    sub = N.select_rows(rb, [5, 0, 7])
    C.validate_row_batch(sub)
    back = N.concat_rows([N.select_rows(rb, range(20)), N.select_rows(rb, range(20, 48))])
    for k, v in rb.items():
        np.testing.assert_array_equal(back[k], v, err_msg=k)
    assert N.concat_rows([])["n_tok"].shape == (0,)
    # The same as the contract's own slicing and joining (revision 2).
    for idx in ([5, 0, 7], [3, 3, 47], [], list(range(48))[::-1]):
        want = C.take_rows(rb, idx)
        got = N.select_rows(rb, idx)
        for k in want:
            assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), k
    parts = [N.select_rows(rb, range(20)), N.select_rows(rb, range(20, 48))]
    joined = C.concat_row_batches(parts)
    for k, v in N.concat_rows(parts).items():
        assert np.array_equal(joined[k], v), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pinned_uploads_build_the_same_batch():
    """`to_batch(pin=True)` stages its uploads in pinned memory (the learner's
    serving path, so an upload never waits on queued kernels): same tensors."""
    rb = _rows(24, seed=5, teacher=True, ppool=True)
    a, b = N.to_batch(rb, "cuda", pin=True), N.to_batch(rb, "cuda")
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, torch.Tensor):
            assert x.device == y.device and x.dtype == y.dtype and torch.equal(x, y), f.name
        else:
            assert x == y, f.name


# ── shapes, masks, values ─────────────────────────────────────────────────


def test_shapes_and_masks():
    rb = _rows(10, seed=3)
    b = N.to_batch(rb, n_rows=16)
    bb, t, a, p, j, jp = b.shape
    assert bb == 16 and t in N.LEN_BUCKETS and a in N.OPT_BUCKETS and p in N.PRIV_BUCKETS
    assert j == C.N_SUMMARY_TOKENS and jp == 1
    net = _net()
    out = net(b)
    assert out.logits.shape == (16, a) and out.value.shape == (16,)
    assert out.logits.dtype == out.logp.dtype == out.value.dtype == torch.float32
    for i, nc in enumerate(rb["n_classes"].astype(int)):
        row = out.logits[i]
        assert torch.isfinite(row[:nc]).all() and torch.isinf(row[nc:]).all()
        assert torch.allclose(out.logp[i, :nc].exp().sum(), torch.tensor(1.0), atol=1e-5)
    # Filler rows are inert, not NaN.
    assert torch.isfinite(out.logp[10:, 0]).all() and torch.isfinite(out.value).all()
    assert net.act_greedy(b).shape == (10,)
    # Visible token counts: hidden members are dropped.
    hidden = (rb["tok_flags"] & C.TF_HIDDEN) != 0
    n_hidden = np.add.reduceat(hidden, np.concatenate([[0], np.cumsum(rb["n_tok"])[:-1]]))
    assert torch.equal((~b.pad[:10]).sum(1), torch.as_tensor(rb["n_tok"] - n_hidden).long())


def test_value_module_is_zero_at_init():
    net = N.E2ENet(seed=0).eval()
    out = net(N.to_batch(_rows(8, seed=5)))
    assert torch.equal(out.v_combat, torch.zeros_like(out.v_combat))
    assert torch.equal(out.v_run, torch.zeros_like(out.v_run))


def test_heads_follow_is_combat():
    net = _net()
    b = N.to_batch(_rows(8, seed=8))
    flipped = dataclasses.replace(b, is_combat=~b.is_combat)
    o1, o2 = net(b), net(flipped)
    assert torch.equal(o1.value, torch.where(b.is_combat, o1.v_combat, o1.v_run))
    assert torch.equal(o2.value, torch.where(b.is_combat, o2.v_run, o2.v_combat))
    assert not torch.allclose(o1.logits[:, :2], o2.logits[:, :2])


# ── what the policy can and cannot see ────────────────────────────────────


def test_priv_never_changes_policy_logits_but_can_change_values():
    """T14 at the net level: PRIV is absent from the trunk input."""
    net = _net()
    rb = _rows(16, seed=7, ppool=True)
    other = _copy(rb)
    g = np.random.default_rng(11)
    other["pcont"] = (g.standard_normal(other["pcont"].shape) * 50).astype(np.float16)
    ids = N.layout().id_cols
    other["pcat"][:, ids] = g.integers(1, C.VOCAB_ROWS, (len(other["pcat"]), len(ids)))
    other["psec"] = g.permutation(other["psec"])
    other["ppool_vid"] = g.integers(1, C.VOCAB_ROWS, other["ppool_vid"].shape).astype(np.uint16)
    b, b2 = N.to_batch(rb), N.to_batch(other)
    base, moved = net(b), net(b2)
    assert torch.equal(base.logits, moved.logits) and torch.equal(base.logp, moved.logp)
    assert torch.equal(net.policy(b), net.policy(b2))
    assert not torch.allclose(base.value, moved.value)
    # No gradient path: d(logits)/d(PRIV numbers) is absent.
    pcont = b2.pcont.clone().requires_grad_(True)
    out = net(dataclasses.replace(b2, pcont=pcont))
    out.logits[torch.isfinite(out.logits)].sum().backward()
    assert pcont.grad is None or not pcont.grad.any()


def test_hidden_members_are_invisible_to_the_student():
    """Non-representative members (TF_HIDDEN) never reach the student: their
    content cannot move a logit, and removing them changes nothing."""
    net = _net()
    rb = _rows(12, seed=19, combat_share=1.0)
    hidden = (rb["tok_flags"] & C.TF_HIDDEN) != 0
    assert hidden.any()
    other = _copy(rb)
    g = np.random.default_rng(2)
    other["cont"][hidden] = (g.standard_normal(other["cont"][hidden].shape) * 9).astype(np.float16)
    other["cat"][hidden, C.XC["X_ID_MAIN"]] = g.integers(1, C.VOCAB_ROWS, int(hidden.sum()))
    base = net.policy(N.to_batch(rb))
    assert torch.equal(base, net.policy(N.to_batch(other)))
    # Drop the hidden tokens: members become the representatives alone.
    rows = []
    for i in range(len(rb["n_tok"])):
        s = _row_slices(rb, i)
        r = N.select_rows(rb, [i])
        keep = ~hidden[s["tok"]]
        newpos = np.cumsum(keep) - 1
        for f in C.ROW_FIELDS:
            if f.axis == "tok":
                r[f.name] = r[f.name][keep]
        r["n_tok"] = np.asarray([keep.sum()], np.int32)
        r["cls_tok"] = newpos[r["cls_tok"].astype(np.int64)].astype(np.uint16)
        r["mem_tok"] = r["cls_tok"].copy()
        r["mem_class"] = np.arange(len(r["cls_tok"]), dtype=np.uint16)
        r["n_mem"] = r["n_classes"].copy()
        r["n_sctx"] = (r["n_sctx"] - (~keep).sum()).astype(np.uint16)
        r["pool_tok"] = newpos[r["pool_tok"].astype(np.int64)].astype(np.uint16)
        rows.append(r)
    lean = N.concat_rows(rows)
    C.validate_row_batch(lean)
    assert torch.equal(base, net.policy(N.to_batch(lean)))


def test_pools_are_weighted_means_of_id_embeddings():
    net = _net()
    rb = _rows(6, seed=23)
    base = net.policy(N.to_batch(rb))
    # Permuting entries, splitting a weight and doubling every weight: same mean.
    perm = _copy(rb)
    g = np.random.default_rng(5)
    o = np.concatenate([s.start + g.permutation(s.stop - s.start) for s in _pool_ranges(rb)])
    for k in ("pool_tok", "pool_vid", "pool_w"):
        perm[k] = perm[k][o]
    assert torch.equal(base, net.policy(N.to_batch(perm)))
    dbl = _copy(rb)
    dbl["pool_w"] = (dbl["pool_w"].astype(np.float32) * 2).astype(np.float16)
    assert torch.equal(base, net.policy(N.to_batch(dbl)))
    split = _copy(rb)
    i0 = int(np.flatnonzero(split["pool_w"] >= 2)[0])
    row0 = int(np.searchsorted(np.cumsum(rb["n_pool"]), i0, side="right"))
    split["pool_w"][i0] -= 1
    for k, v in (("pool_tok", split["pool_tok"][i0]), ("pool_vid", split["pool_vid"][i0])):
        split[k] = np.insert(split[k], i0, v)
    split["pool_w"] = np.insert(split["pool_w"], i0, np.float16(1))
    split["n_pool"][row0] += 1
    assert torch.equal(base, net.policy(N.to_batch(split)))
    # The embedding moves by exactly gain * weighted mean of the ID rows.
    b = N.to_batch(rb)
    empty = dataclasses.replace(b, pool_cnt=torch.zeros_like(b.pool_cnt))
    with torch.no_grad():
        e1 = net.embed(b.cat, b.cont, b.sec, b.pool_t, b.pool_ok, b.pool_cnt)
        e0 = net.embed(empty.cat, empty.cont, empty.sec, b.pool_t, b.pool_ok, empty.pool_cnt)
        s = _row_slices(rb, 0)
        vid = rb["pool_vid"][s["pool"]].astype(np.int64)
        w = rb["pool_w"][s["pool"]].astype(np.float32)
        tok = rb["pool_tok"][s["pool"]].astype(np.int64)
        for t0 in np.unique(tok):
            sel = tok == t0
            want = (torch.as_tensor(w[sel])[:, None] * net.ids.weight[vid[sel]]).sum(0)
            want = net.pool_gain * want / float(w[sel].sum())
            pos = int(b.pool_t[0][b.pool_ok[0]][list(np.unique(tok)).index(t0)])
            torch.testing.assert_close(e1[0, pos] - e0[0, pos], want, rtol=0, atol=1e-5)


def test_reserved_inputs_are_inert_until_trained():
    """Reserved columns and the content prior start at zero weight: filling
    them (a fill event) changes no output, and they then train."""
    net = _net()
    rb = _rows(10, seed=29)
    base = net(N.to_batch(rb))
    L = N.layout()
    filled = _copy(rb)
    g = np.random.default_rng(3)
    n = len(filled["cat"])
    for col, size, live in zip(L.small_cols, L.small_sizes, L.small_live, strict=True):
        if not live:
            filled["cat"][:, col] = g.integers(1, size, n)
    filled["cat"][:, C.XC["X_ID_AUX"]] = g.integers(1, C.VOCAB_ROWS, n)
    zc = list(L.zero_cont)
    filled["cont"][:, zc] = (g.standard_normal((n, len(zc))) * 5).astype(np.float16)
    prior = g.standard_normal((C.VOCAB_ROWS, C.N_PRIOR)).astype(np.float32)
    prior[0] = 0.0
    net.set_prior(prior)
    b = N.to_batch(filled)
    out = net(b)
    assert torch.equal(base.logits, out.logits) and torch.equal(base.value, out.value)
    (out.logits[torch.isfinite(out.logits)].sum() + out.value.sum()).backward()
    assert net.num.inp.weight.grad[:, list(L.zero_stem_cols)].abs().sum(0).gt(0).all()
    assert net.id_gain.grad[L.id_cols.index(C.XC["X_ID_AUX"])].abs().sum() > 0
    off = dict(zip(L.small_cols, L.small_offsets, strict=True))
    col = C.XC["X_FC1"]
    rows = torch.as_tensor(filled["cat"][:, col].astype(np.int64) + off[col])
    assert net.small.weight.grad[rows].abs().sum() > 0
    with pytest.raises(ValueError):
        net.set_prior(np.ones((C.VOCAB_ROWS, C.N_PRIOR), np.float32), sha256=None)
    with pytest.raises(ValueError, match="not the table's"):
        net.set_prior(prior, sha256="p" * 64)
    assert net.prior_sha256 == N.prior_buffer_sha256(prior)


#: WP-ENG-1's inputs (contracts: status `reserved:WP-ENG-1`).
WP_CAT = [c.name for c in C.X_CAT if c.status == "reserved:WP-ENG-1"]
WP_CONT = [C.XF[c.name] for c in C.X_CONT if c.status == "reserved:WP-ENG-1"]


def _small_block(net, L, name):
    off = dict(zip(L.small_cols, L.small_offsets, strict=True))
    size = dict(zip(L.small_cols, L.small_sizes, strict=True))
    o = off[C.XC[name]]
    return net.small.weight[o + 1 : o + size[C.XC[name]]]  # row o is the absent value


def test_a_launch_that_fills_wp_eng1_inits_its_inputs_live():
    """A4: `E2ENet(filled=["WP-ENG-1"])` (the learner's `--filled`) inits WP-ENG-1's
    inputs like live ones: the telegraph and forecast stem weights and embedding rows,
    the honest X_ID_AUX gain at 1 and the encounter and move ID rows. Everything else
    is the default net's init, bit for bit, and the default net keeps them at zero."""
    assert set(WP_CAT) == {"X_ID_AUX", "X_INTENT_T2", "X_INTENT_T3", "X_FC1", "X_FC2", "X_FC3"}
    assert len(WP_CONT) == 17
    base, wp = N.E2ENet(_small_cfg(), seed=5), N.E2ENet(_small_cfg(), seed=5, filled=["WP-ENG-1"])
    Lb, Lw = base.layout, wp.layout
    assert Lb == N.layout() and Lw == N.layout(["WP-ENG-1"]) and Lw.filled == ("WP-ENG-1",)
    aux = Lb.id_cols.index(C.XC["X_ID_AUX"])
    assert not base.id_gain[aux].any() and torch.equal(
        wp.id_gain[aux], torch.ones_like(wp.id_gain[aux])
    )
    assert torch.equal(wp.id_gain, torch.ones_like(wp.id_gain))
    for name in WP_CAT[1:]:
        assert not _small_block(base, Lb, name).any(), name
        assert _small_block(wp, Lw, name).abs().sum(1).gt(0).all(), name
    assert not base.num.inp.weight[:, WP_CONT].any()
    assert wp.num.inp.weight[:, WP_CONT].abs().sum(0).gt(0).all()
    rows = list(N.zero_id_rows())
    assert N.zero_id_rows(["WP-ENG-1"]) == ()
    assert not base.ids.weight[rows].any() and wp.ids.weight[rows].abs().sum(1).gt(0).all()
    # The other reserved inputs stay inert: D9's flag, the spares, the prior columns.
    still = sorted(set(Lw.zero_stem_cols))
    assert set(Lb.zero_stem_cols) - set(WP_CONT) == set(still) and still
    assert not wp.num.inp.weight[:, still].any()
    assert C.XF["R_HIST_MISSING"] in still and Lw.zero_cont == tuple(
        c for c in Lb.zero_cont if c not in WP_CONT
    )
    # Same RNG draws: only the zeroing differs, so every other weight is equal.
    sb, sw = base.state_dict(), wp.state_dict()
    for k in sb:
        if k not in ("id_gain", "small.weight", "num.inp.weight", "ids.weight"):
            assert torch.equal(sb[k], sw[k]), k
    keep = torch.ones(Lb.vocab_rows, dtype=torch.bool)
    keep[rows] = False
    assert torch.equal(sb["ids.weight"][keep], sw["ids.weight"][keep])
    with pytest.raises(C.SchemaError, match="unknown fill events"):
        N.E2ENet(_small_cfg(), filled=["WP-ENG-2"])


def test_a_fill_after_a_checkpoint_keeps_the_identity_rule():
    """A checkpoint of a net built without WP-ENG-1 keeps its zero weights when a
    process that now declares WP-ENG-1 loads it: the init is overridden, so filling
    the columns changes no output until training moves them."""
    net = _net()
    ck = N.net_state(net)
    back = N.net_from_state(ck)
    aux = back.layout.id_cols.index(C.XC["X_ID_AUX"])
    assert not back.id_gain[aux].any() and not back.num.inp.weight[:, WP_CONT].any()
    assert not back.ids.weight[list(N.zero_id_rows())].any()


def test_content_prior_and_1069_fill_their_own_inputs():
    """The rule is per event: "content-prior" makes the prior stem columns live,
    "#1069" the reward kinds and the reward decision type; WP-ENG-1's stay zero."""
    L = N.layout(["content-prior", "#1069"])
    assert L.prior_live and L.zero_stem_cols == L.zero_cont
    assert set(WP_CONT) <= set(L.zero_cont)
    k1069 = {k.id for k in C.ACTION_KINDS if k.status == "reserved:#1069"}
    d1069 = {d.id for d in C.DECISION_TYPES if d.status == "reserved:#1069"}
    assert k1069 and k1069 <= set(L.live_kinds) and not k1069 & set(N.layout().live_kinds)
    assert d1069 and d1069 <= set(L.live_dtypes) and not d1069 & set(N.layout().live_dtypes)
    net = N.E2ENet(_small_cfg(), seed=1, filled=["content-prior"])
    prior = list(range(C.N_CONT, C.N_CONT + C.N_PRIOR))
    assert net.num.inp.weight[:, prior].abs().sum(0).gt(0).all()


def test_zero_started_tables_and_the_kind_embedding():
    """SCHEMA.md section 2.12's zero-started rows (encounter and move ids,
    reserved sections, reserved action kinds and decision types) start at
    zero and still train; `cls_kind` drives each option's kind embedding."""
    net = _net()
    L = N.layout()
    zero = list(N.zero_id_rows())
    assert not net.ids.weight[zero].any() and net.ids.weight[1:].abs().sum(1).gt(0).sum() > 3000
    dead_kinds = sorted(set(range(C.N_ACTION_KIND)) - set(L.live_kinds))
    assert dead_kinds and not net.kind.weight[dead_kinds].any()
    assert net.kind.weight[list(L.live_kinds)].abs().sum(1).gt(0).all()
    assert not net.sec.weight[list(L.reserved_sections)].any()
    off = dict(zip(L.small_cols, L.small_offsets, strict=True))
    for name, live, n in (
        ("X_KIND", L.live_kinds, C.N_ACTION_KIND),
        ("X_DTYPE", L.live_dtypes, C.N_DTYPE),
    ):
        dead = [off[C.XC[name]] + k for k in sorted(set(range(1, n)) - set(live))]
        assert dead and not net.small.weight[dead].any()
    # Two encounters in a BOSS token: both rows are zero, so no output moves.
    rb = _rows(6, seed=37)
    enc = V.load().block("encounter")
    boss = np.flatnonzero(rb["sec"] == C.SEC["BOSS"])
    a, b2 = _copy(rb), _copy(rb)
    a["cat"][boss, C.XC["X_ID_MAIN"]] = enc[0]
    b2["cat"][boss, C.XC["X_ID_MAIN"]] = enc[1]
    oa, ob = net(N.to_batch(a)), net(N.to_batch(b2))
    assert torch.equal(oa.logits, ob.logits) and torch.equal(oa.value, ob.value)
    oa.logits[torch.isfinite(oa.logits)].sum().backward()
    assert net.ids.weight.grad[enc[0]].abs().sum() > 0  # the rows still train
    # The option's action kind (cls_kind) feeds its logit.
    rb = _rows(8, seed=41, combat_share=1.0)
    turn = np.flatnonzero(rb["dtype"] == C.DT["COMBAT_TURN"])
    assert len(turn)
    i = int(turn[0])
    s = _row_slices(rb, i)["cls"]
    other = _copy(rb)
    k0 = int(other["cls_kind"][s.start])
    other["cls_kind"][s.start] = C.AK["END_TURN"] if k0 != C.AK["END_TURN"] else C.AK["PLAY"]
    x, y = net.policy(N.to_batch(rb)), net.policy(N.to_batch(other))
    assert x[i, 0] != y[i, 0]
    assert torch.equal(x[i, 1 : int(rb["n_classes"][i])], y[i, 1 : int(rb["n_classes"][i])])
    keep = [j for j in range(8) if j != i]
    assert torch.equal(x[keep], y[keep])


# ── precision (X-B) ───────────────────────────────────────────────────────


def _xb_check(device, n_rows):
    net = _loud_heads(N.E2ENet(N.NetConfig(), seed=0)).to(device).eval()  # the real 6 x 256
    b = N.to_batch(_rows(n_rows, seed=13), device)
    gap = N.precision_gap(net, b)
    print(f"\nX-B on {device}: {gap}")
    lp = net(b, bf16=False).logp[: b.n_real]
    spread = lp[torch.isfinite(lp)].std()
    assert spread > 0.3, f"logits too flat ({spread}) for a meaningful precision check"
    assert gap["p99_abs_dlogp"] < 0.05
    assert gap["mean_kl"] < 1e-3


def test_bf16_matches_fp32_within_xb_tolerance_cpu():
    _xb_check("cpu", 64)  # CPU bf16 matmuls are slow: fewer rows


def test_xb_over_a_row_batch_in_chunks():
    net = _net()
    rb = _rows(40, seed=13)
    whole = N.precision_gap(net, N.to_batch(rb))
    parts = N.precision_gap_rows(net, rb, chunk=16)
    assert parts["n_rows"] == whole["n_rows"] == 40
    assert parts["n_options"] == whole["n_options"] == int(rb["n_classes"].sum())
    assert parts["p99_abs_dlogp"] < 0.05 and parts["mean_kl"] < 1e-3
    # A separate bf16 path (here: the same net, as a compiled one would be).
    same = N.precision_gap(net, N.to_batch(rb), bf16_policy=net.policy)
    assert same == whole


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_matches_fp32_within_xb_tolerance_cuda():
    _xb_check("cuda", 256)


def test_weights_stay_fp32_under_bf16():
    net = N.E2ENet(_small_cfg(), seed=0)
    net.bf16 = True
    out = net(N.to_batch(_rows(4)))
    assert all(p.dtype == torch.float32 for p in net.parameters())
    assert out.logp.dtype == out.value.dtype == torch.float32


def _linear_dtypes(net, b, **kw):
    """`{module name: output dtype}` of every Linear in one `forward`."""
    seen, hooks = {}, []

    def record(name):
        def hook(mod, inp, out):  # returns None: the output stays as it is
            seen.setdefault(name, out.dtype)

        return hook

    for name, mod in net.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(record(name)))
    try:
        with torch.no_grad():
            net(b, **kw)
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_each_layer_runs_in_the_asked_precision():
    """`bf16=False` is fp32 end to end (X-B's reference and fp32 evaluation);
    `bf16=True` runs the trunk and the heads' hidden layers in bf16, except
    with `fp32_heads`; the output projections stay fp32 either way."""
    net = _net()
    b = N.to_batch(_rows(6, seed=8, ppool=True))
    fp32 = _linear_dtypes(net, b, bf16=False)
    assert fp32 and set(fp32.values()) == {torch.float32}, fp32
    for fp32_heads in (False, True):
        net.cfg = dataclasses.replace(net.cfg, fp32_heads=fp32_heads)
        got = _linear_dtypes(net, b, bf16=True)
        assert got.keys() == fp32.keys()
        for name, dt in got.items():
            out = name.endswith(".out") and name.startswith("head_")
            head = name.startswith("head_")
            fp32_layer = out or name in ("value.v_combat", "value.v_run") or (head and fp32_heads)
            assert dt == (torch.float32 if fp32_layer else torch.bfloat16), (name, fp32_heads)


def test_fp32_heads_narrow_the_bf16_gap():
    cfg = _small_cfg(dim=128, ffn=352, head_hidden=128)
    net = _net(cfg, scale=12.0)
    b = N.to_batch(_rows(48, seed=17))
    gap16 = N.precision_gap(net, b)
    net.cfg = dataclasses.replace(net.cfg, fp32_heads=True)
    gap32 = N.precision_gap(net, b)
    assert gap32["p99_abs_dlogp"] < gap16["p99_abs_dlogp"]


# ── determinism and invariances ───────────────────────────────────────────


def test_init_and_forward_are_deterministic():
    cfg = _small_cfg()
    a, b2 = N.E2ENet(cfg, seed=42), N.E2ENet(cfg, seed=42)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b2.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb), ka
    assert not torch.equal(N.E2ENet(cfg, seed=43).ids.weight, a.ids.weight)
    # Seeding the net does not disturb the caller's RNG stream.
    torch.manual_seed(0)
    x = torch.rand(3)
    torch.manual_seed(0)
    N.E2ENet(cfg, seed=5)
    assert torch.equal(torch.rand(3), x)
    a = _loud_heads(a).eval()
    b = N.to_batch(_rows(10, seed=2))
    o1, o2 = a(b), a(b)
    assert torch.equal(o1.logits, o2.logits) and torch.equal(o1.value, o2.value)


def test_padded_and_unpadded_rows_give_the_same_logits():
    net = _net()
    rb = _rows(9, seed=21, ppool=True)
    big = net(N.to_batch(rb, n_rows=32))
    for i in range(9):
        alone = net(
            N.to_batch(
                N.select_rows(rb, [i]),
                len_buckets=None,
                opt_buckets=None,
                priv_buckets=None,
                pool_buckets=None,
                ppool_buckets=None,
            )
        )
        a = int(rb["n_classes"][i])
        torch.testing.assert_close(alone.logits[0, :a], big.logits[i, :a], rtol=0, atol=1e-5)
        torch.testing.assert_close(alone.value[0], big.value[i], rtol=0, atol=1e-5)


def test_token_order_is_equivariant():
    """No positional encoding: permuting a row's tokens (RUNCTX stays first)
    permutes nothing else."""
    net = _net()
    rb = _rows(4, seed=4)
    g = np.random.default_rng(0)
    rows = []
    for i in range(4):
        r = N.select_rows(rb, [i])
        nt = int(r["n_tok"][0])
        perm = np.r_[0, 1 + g.permutation(nt - 1)]
        inv = np.argsort(perm)
        for f in C.ROW_FIELDS:
            if f.axis == "tok":
                r[f.name] = r[f.name][perm]
        for k in ("cls_tok", "mem_tok", "pool_tok"):
            r[k] = inv[r[k].astype(np.int64)].astype(np.uint16)
        rows.append(r)
    x, y = net(N.to_batch(rb)), net(N.to_batch(N.concat_rows(rows)))
    torch.testing.assert_close(x.logits, y.logits, rtol=0, atol=1e-5)
    torch.testing.assert_close(x.value, y.value, rtol=0, atol=1e-5)


# ── packed serving (to_packed, forward_packed) ────────────────────────────


def _one_token_row(seed):
    """A degenerate row: RUNCTX alone, which is also its one class's token; no
    pools, no PRIV tokens."""
    r = N.select_rows(_rows(1, seed=seed, priv=(0, 0)), [0])
    keep = {"tok": 1, "mem": 1, "cls": 1, "pool": 0, "priv": 0, "ppool": 0}
    out = {}
    for f in C.ROW_FIELDS:
        v = r[f.name]
        out[f.name] = v[: keep[f.axis]].copy() if f.axis in keep else v.copy()
    for axis, cnt in N.AXES.items():
        out[cnt][:] = keep[axis]
    out["tok_flags"][:] = 0
    for k in ("cls_tok", "mem_tok", "mem_class"):
        out[k][:] = 0
    return out


def _assert_packed_matches(net, rb, *, bf16=False, attn="auto", atol=1e-5, device="cpu"):
    """`forward_packed(to_packed(rb))` vs `forward(to_batch(rb))`: the same
    shapes and masks, every output within `atol`."""
    n = len(rb["n_tok"])
    ref = net(N.to_batch(rb, device), bf16=bf16)
    got = net.forward_packed(N.to_packed(rb, device), bf16=bf16, attn=attn)
    for name, x, y in zip(ref._fields, ref, got, strict=True):
        x = x[:n]
        assert x.shape == y.shape and y.dtype == torch.float32, name
        fin = torch.isfinite(x)
        assert torch.equal(fin, torch.isfinite(y)), name
        torch.testing.assert_close(y[fin], x[fin], rtol=0, atol=atol, msg=name)
    return ref, got


@pytest.mark.parametrize("fp32_heads", [False, True])
def test_packed_forward_matches_the_padded_forward(fp32_heads):
    net = _net(_small_cfg(fp32_heads=fp32_heads))
    rb = _rows(24, seed=5, ppool=True)
    _ref, got = _assert_packed_matches(net, rb)
    pb = N.to_packed(rb)
    assert pb.n_opt == N.to_batch(rb).shape[2]  # the same option bucket
    assert pb.cat.shape[0] == int(C.student_mask(rb).sum())  # no padded token
    assert torch.equal(net.policy_packed(pb), got.logits)
    # Every row alone gives its outputs in the batch (no cross-row leak).
    for i in (0, 7, 23):
        alone = net.forward_packed(N.to_packed(N.select_rows(rb, [i])))
        w = alone.logits.shape[1]
        torch.testing.assert_close(alone.logits[0], got.logits[i, :w], rtol=0, atol=1e-5)
        assert torch.isinf(got.logits[i, w:]).all()
        torch.testing.assert_close(alone.value[0], got.value[i], rtol=0, atol=1e-5)


def test_packed_forward_edge_cases():
    """One-token rows (no pools, no PRIV), rows at the longest bucket, a batch
    with no PRIV token at all, and a single row."""
    net = _net()
    long = _rows(3, seed=8, body=(200, 210))
    rb = N.concat_rows([_one_token_row(1), _rows(5, seed=9), long, _one_token_row(2)])
    pb = N.to_packed(rb)
    assert pb.max_len > 192 and N.to_batch(rb).shape[1] == 256
    _assert_packed_matches(net, rb)
    _assert_packed_matches(net, _rows(6, seed=10, priv=(0, 0)))
    _assert_packed_matches(net, _one_token_row(3))
    _assert_packed_matches(net, N.select_rows(long, [1]))
    bad = _copy(_rows(2, seed=3))
    bad["n_classes"][0] = 0
    with pytest.raises(ValueError, match="n_classes outside"):
        N.to_packed(bad)


def _row_of_exact_length(n):
    """A synthetic row with exactly `n` student-visible tokens."""
    for seed in range(400):
        r = _rows(1, seed=seed, body=(n - 40, n))
        if int(C.student_mask(r).sum()) == n:
            return r
    raise AssertionError(f"no synthetic row of {n} visible tokens")


def _row_of_64_options():
    """A row at the option cap: 64 classes on 64 visible tokens."""
    r = _copy(_rows(1, seed=4, body=(80, 90)))
    r["tok_flags"] = r["tok_flags"] & ~np.asarray(C.TF_HIDDEN, r["tok_flags"].dtype)
    r["cls_tok"] = np.arange(1, 65, dtype=r["cls_tok"].dtype)
    r["cls_kind"] = np.resize(r["cls_kind"], 64)
    r["n_classes"][:] = C.MAX_OPTIONS
    return r


def test_packed_forward_at_the_token_and_option_caps():
    """A row of exactly MAX_TOKENS visible tokens, and one of MAX_OPTIONS classes."""
    net = _net()
    full = _row_of_exact_length(C.MAX_TOKENS)
    pb = N.to_packed(full)
    assert pb.max_len == C.MAX_TOKENS == N.to_batch(full).shape[1]
    _assert_packed_matches(net, full)
    wide = _row_of_64_options()
    assert N.to_packed(wide).n_opt == C.MAX_OPTIONS == N.to_batch(wide).shape[2]
    _assert_packed_matches(net, wide)
    _assert_packed_matches(net, N.concat_rows([full, wide, _one_token_row(5)]))


def test_packed_priv_never_changes_policy_logits_but_can_change_values():
    """F7 / T14 on the packed path: PRIV reaches the value module only."""
    net = _net()
    rb = _rows(16, seed=7, ppool=True)
    other = _copy(rb)
    g = np.random.default_rng(11)
    other["pcont"] = (g.standard_normal(other["pcont"].shape) * 50).astype(np.float16)
    ids = N.layout().id_cols
    other["pcat"][:, ids] = g.integers(1, C.VOCAB_ROWS, (len(other["pcat"]), len(ids)))
    other["psec"] = g.permutation(other["psec"])
    other["ppool_vid"] = g.integers(1, C.VOCAB_ROWS, other["ppool_vid"].shape).astype(np.uint16)
    none = N.concat_rows([_drop_priv(N.select_rows(rb, [i])) for i in range(16)])
    base = net.forward_packed(N.to_packed(rb))
    for rb2 in (other, none):
        moved = net.forward_packed(N.to_packed(rb2))
        assert torch.equal(base.logits, moved.logits) and torch.equal(base.logp, moved.logp)
        assert not torch.allclose(base.value, moved.value)
    pb = N.to_packed(other)
    pcont = pb.pcont.clone().requires_grad_(True)
    out = net.forward_packed(dataclasses.replace(pb, pcont=pcont))
    out.logits[torch.isfinite(out.logits)].sum().backward()
    assert pcont.grad is None or not pcont.grad.any()


def _drop_priv(r):
    out = dict(r)
    for f in C.ROW_FIELDS:
        if f.axis in ("priv", "ppool"):
            out[f.name] = r[f.name][:0]
    out["n_priv"][:] = 0
    out["n_ppool"][:] = 0
    return out


def _sdpa_rows(q, k, v, cu_q, cu_k):
    """Reference: each row's attention on its own."""
    outs = []
    for i in range(len(cu_q) - 1):
        a, b, c, d = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        o = torch.nn.functional.scaled_dot_product_attention(
            q[a:b].transpose(0, 1), k[c:d].transpose(0, 1), v[c:d].transpose(0, 1)
        )
        outs.append(o.transpose(0, 1))
    return torch.cat(outs)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
        ),
    ],
)
def test_varlen_attention_matches_per_row_attention(device):
    """`Varlen` (pad mode; varlen on CUDA in bf16): rows of 1..max tokens, all
    queries or one query per row."""
    g = torch.Generator().manual_seed(0)
    lens = torch.tensor([1, 5, 1, 17, 64, 2])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).int()
    n = int(lens.sum())
    q, k, v = (torch.randn(n, 4, 16, generator=g).to(device) for _ in range(3))
    one = torch.arange(len(lens) + 1, dtype=torch.int32)
    cases = [(cu, q), (one, q[cu[:-1].long()])]
    for cu_q, qq in cases:
        cu_q, cu_k = cu_q.to(device), cu.to(device)
        ref = _sdpa_rows(qq, k, v, cu_q.cpu(), cu_k.cpu())
        att = N.Varlen(cu_q, cu_k, int(cu_q.diff().max()), int(lens.max()), len(qq), n, "pad")
        torch.testing.assert_close(att(qq, k, v), ref, rtol=0, atol=1e-5)
        if device == "cuda":
            b16 = [x.bfloat16() for x in (qq, k, v)]
            att.mode = "varlen"
            torch.testing.assert_close(att(*b16).float(), ref, rtol=0, atol=3e-2)
            att.mode = "auto"  # auto picks varlen for bf16 on CUDA
            assert torch.equal(
                att(*b16), N.Varlen(cu_q, cu_k, att.max_q, att.max_k, len(qq), n, "varlen")(*b16)
            )
    with pytest.raises(ValueError, match="attention mode"):
        N.Varlen(cu, cu, 1, 1, 1, 1, "flash")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_packed_bf16_on_cuda_is_within_xb_of_fp32():
    """The serving precision: packed bf16 (varlen) vs packed fp32 by X-B's rule,
    and close to the padded bf16 forward."""
    net = _net(N.NetConfig(n_layers=2)).cuda()
    rb = _rows(64, seed=12, ppool=True)
    pb = N.to_packed(rb, "cuda")
    lp32 = net.forward_packed(pb, bf16=False).logp.double()
    lp16 = net.forward_packed(pb, bf16=True).logp.double()
    m = torch.isfinite(lp32)
    d = (lp32 - lp16).abs()[m]
    kl = torch.where(m, lp32.exp() * (lp32 - lp16), torch.zeros_like(lp32)).sum(-1)
    assert float(torch.quantile(d, 0.99)) < 0.05 and float(kl.mean()) < 1e-3
    _assert_packed_matches(net, rb, bf16=False, atol=1e-4, device="cuda")


# ── training-side plumbing ────────────────────────────────────────────────


def _grad_norms(net, out):
    net.zero_grad(set_to_none=True)
    out.value.square().sum().backward()

    def gn(m):
        return sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)

    return gn(net.blocks), gn(net.ids), gn(net.value)


def test_valve_stops_value_gradients_into_the_trunk():
    net = _loud_heads(N.E2ENet(_small_cfg(), seed=0))
    b = N.to_batch(_rows(8, seed=3))
    trunk, ids, value = _grad_norms(net, net(b))
    assert trunk > 0 and ids > 0 and value > 0  # shared trunk: value trains it
    net.set_valve(True)
    trunk, ids, value = _grad_norms(net, net(b))
    assert trunk == 0 and ids == 0 and value > 0
    # The valve changes no forward value.
    net.set_valve(False)
    v0 = net(b).value
    net.set_valve(True)
    assert torch.equal(v0, net(b).value)


def test_param_groups_split_by_the_value_prefix():
    net = N.E2ENet(_small_cfg(), seed=0)
    g = net.param_groups()
    ids = [id(p) for p in g["trunk"] + g["value"]]
    assert len(ids) == len(set(ids)) == len(list(net.parameters()))
    by_name = {id(p) for n, p in net.named_parameters() if n.startswith("value.")}
    assert {id(p) for p in g["value"]} == by_name  # `ppo.e2e.algo.param_groups`'s rule
    assert id(net.ids.weight) in {id(p) for p in g["trunk"]}


def test_checkpoint_roundtrip_and_schema_refusal(tmp_path):
    cfg = _small_cfg()
    net = _net(cfg)
    prior = np.random.default_rng(5).random((C.VOCAB_ROWS, C.N_PRIOR), np.float32)
    prior[0] = 0.0
    net.set_prior(prior)
    path = tmp_path / "net.pt"
    sha = N.save_checkpoint(path, net, vocab_sha256=VOCAB, stamps={"x": 1})
    assert sha == N.file_sha256(path)
    got, ck_cfg = N.load_checkpoint(path, vocab_sha256=VOCAB)
    assert got.cfg == cfg and ck_cfg["stamps"] == {"x": 1}
    assert ck_cfg["schema_sha256"] == C.schema_sha256()
    assert got.prior_sha256 == N.prior_buffer_sha256(prior) and torch.equal(got.prior, net.prior)
    b = N.to_batch(_rows(5, seed=1))
    assert torch.equal(net(b).logits, got.eval()(b).logits)
    with pytest.raises(N.SchemaMismatch):
        N.load_checkpoint(path, schema_sha256="c" * 64)
    with pytest.raises(N.SchemaMismatch):
        N.load_checkpoint(path, vocab_sha256="d" * 64)
    with pytest.raises(ValueError):
        N.net_state(net, vocab_sha256="")
    # Default stamps: the contract's keys, whatever WP1's stamps say; a prior is a fill.
    st = N.default_stamps(vocab_sha256=VOCAB, prior_sha256="p" * 64)
    assert set(C.STAMP_KEYS) <= set(st) and st["schema_sha256"] == C.schema_sha256()
    assert st["teacher"] is None and st["filled"] == ["content-prior"]


def test_a_checkpoint_whose_prior_is_not_its_stamp_is_refused(tmp_path):
    """The `prior` buffer must be what `prior_sha256` says (review finding NT-2: an EMA
    shadow that averaged the prior in would serve a shrunken prior under the full
    prior's stamp). A prior-less checkpoint's buffer must be zero."""
    net = _net()
    prior = np.random.default_rng(6).random((C.VOCAB_ROWS, C.N_PRIOR), np.float32)
    prior[0] = 0.0
    net.set_prior(prior)
    ck = N.net_state(net, vocab_sha256=VOCAB, stamps={})
    N.net_from_state(ck, vocab_sha256=VOCAB)
    ck["state_dict"]["prior"] = ck["state_dict"]["prior"] * 0.5
    with pytest.raises(N.SchemaMismatch, match="prior"):
        N.net_from_state(ck, vocab_sha256=VOCAB)
    ck["cfg"]["prior_sha256"] = None
    with pytest.raises(N.SchemaMismatch, match="not zero"):
        N.net_from_state(ck, vocab_sha256=VOCAB)


def test_a_default_checkpoint_loads_weights_only(campaign_process, tmp_path):
    """`save_checkpoint` with the default stamps (WP1's included) writes a file
    that `torch.load(weights_only=True)` reads: no `TorchVersion` or other
    non-plain stamp value. Stamps a caller passes are made plain too."""
    net = _net()
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    N.save_checkpoint(a, net, vocab_sha256=VOCAB)
    got, cfg = N.load_checkpoint(a, vocab_sha256=VOCAB)
    b0 = N.to_batch(_rows(3, seed=2))
    assert torch.equal(got.eval().policy(b0), net.policy(b0))
    assert cfg["stamps"]["torch"] == str(torch.__version__)
    assert cfg["stamps"]["game_version"] == C.GAME_VERSION and "stamp_error" not in cfg["stamps"]
    N.save_checkpoint(b, net, vocab_sha256=VOCAB, stamps={"p": tmp_path, "n": np.int64(3)})
    _, cfg = N.load_checkpoint(b, vocab_sha256=VOCAB)
    assert cfg["stamps"] == {"p": str(tmp_path), "n": 3}


def test_checkpoint_stamps_are_checked_against_the_process(tmp_path):
    """`want_stamps`: `contracts.check_stamps` on load, so a checkpoint that
    read filled columns or another teacher's rows is refused."""
    net = _net()
    teacher = C.teacher_stamp(ckpt_sha256="t" * 64, rare=None, sctx_vocab={"card": 601})
    saved = N.default_stamps(vocab_sha256=VOCAB, filled=["D9"], teacher=teacher)
    assert saved["teacher"] == teacher
    path = tmp_path / "net.pt"
    N.save_checkpoint(path, net, vocab_sha256=VOCAB, stamps=saved)
    same = C.stamps(vocab_sha256=VOCAB, filled=["D9"], teacher=teacher)
    N.load_checkpoint(path, vocab_sha256=VOCAB, want_stamps=same)
    pol = N.StudentPolicy.from_checkpoint(path, vocab_sha256=VOCAB, want_stamps=same)
    assert pol.meta["cfg"]["stamps"]["teacher"] == teacher
    unfilled = C.stamps(vocab_sha256=VOCAB, teacher=teacher)  # serves the column at zero
    with pytest.raises(C.SchemaError):
        N.load_checkpoint(path, vocab_sha256=VOCAB, want_stamps=unfilled)
    with pytest.raises(C.SchemaError):
        N.StudentPolicy.from_checkpoint(path, vocab_sha256=VOCAB, want_stamps=unfilled)
    other = C.teacher_stamp(ckpt_sha256="u" * 64, rare=None, sctx_vocab={"card": 601})
    with pytest.raises(C.SchemaError):
        N.load_checkpoint(
            path,
            vocab_sha256=VOCAB,
            want_stamps=C.stamps(vocab_sha256=VOCAB, filled=["D9"], teacher=other),
        )
    # A restart may add fill events (SCHEMA.md section 2.12's identity rule).
    more = C.stamps(vocab_sha256=VOCAB, filled=["#1069", "D9"], teacher=teacher)
    N.load_checkpoint(path, vocab_sha256=VOCAB, want_stamps=more, restart=True)
    with pytest.raises(C.SchemaError):
        N.load_checkpoint(path, vocab_sha256=VOCAB, want_stamps=more)


def test_a_student_serves_only_what_its_wheel_fills(tmp_path, monkeypatch):
    """Review finding LRN-2: an evaluator loads the student without the lane's stamps,
    so `from_checkpoint` checks what it can serve: the game version, and every
    wheel-filled event the checkpoint read (a WP-ENG-1 net on W1 would read its
    telegraph and forecast inputs at zero)."""
    from ppo.e2e import fillcheck as FC

    net = _net()
    path = tmp_path / "wp.pt"
    wp = C.stamps(vocab_sha256=VOCAB, filled=["WP-ENG-1"])
    N.save_checkpoint(path, net, vocab_sha256=VOCAB, stamps=wp)
    monkeypatch.setattr(FC, "wheel_filled", lambda: ("WP-ENG-1",))
    N.StudentPolicy.from_checkpoint(path, vocab_sha256=VOCAB)
    monkeypatch.setattr(FC, "wheel_filled", lambda: ())  # W1: no getters
    with pytest.raises(N.SchemaMismatch, match="WP-ENG-1"):
        N.StudentPolicy.from_checkpoint(path, vocab_sha256=VOCAB)
    old = tmp_path / "old.pt"
    stale = dict(C.stamps(vocab_sha256=VOCAB), game_version="v0.107.1")
    N.save_checkpoint(old, net, vocab_sha256=VOCAB, stamps=stale)
    with pytest.raises(N.SchemaMismatch, match=r"v0\.107\.1"):
        N.StudentPolicy.from_checkpoint(old, vocab_sha256=VOCAB)


def test_to_batch_rejects_bad_rows():
    rb = _rows(3, seed=0, combat_share=1.0)
    s = _row_slices(rb, 0)
    bad = _copy(rb)
    rep = int(bad["cls_tok"][s["cls"]][0])
    bad["tok_flags"][s["tok"].start + rep] |= C.TF_HIDDEN
    with pytest.raises(ValueError, match="hidden"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["tok_flags"][s["tok"].start] |= C.TF_HIDDEN
    with pytest.raises(ValueError, match="RUNCTX"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["pool_vid"][0] = 0
    with pytest.raises(ValueError, match="NONE"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["tok_flags"][s["tok"].start + int(bad["pool_tok"][0])] &= ~np.uint8(C.TF_POOLED)
    with pytest.raises(ValueError, match="POOLED"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["cat"][3, C.XC["X_KIND"]] = C.N_ACTION_KIND  # one past the X_KIND table
    with pytest.raises(ValueError, match="outside their tables"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["cat"][2, C.XC["X_ID_MAIN"]] = -1
    with pytest.raises(ValueError, match="outside their tables"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["psec"][:1] = C.N_SECTION
    with pytest.raises(ValueError, match="section"):
        N.to_batch(bad)
    bad = _copy(rb)
    bad["cls_kind"][0] = C.N_ACTION_KIND
    with pytest.raises(ValueError, match="cls_kind"):
        N.to_batch(bad)
    with pytest.raises(C.SchemaError):
        N.to_batch(dict(rb, action=rb["n_classes"].copy()), check=True)
    assert N.bucket(33, N.LEN_BUCKETS) == 48 and N.bucket(300, None) == 300


_ENCODE_FIELDS = [f.name for f in C.ROW_FIELDS if f.axis in ("tok", "mem", "pool")]
_ENCODE_FIELDS += ["cls_tok", "cls_kind", "sctx_lo", "n_sctx"]
_PRIV_FIELDS = [f.name for f in C.ROW_FIELDS if f.axis in ("priv", "ppool")]


class _RowTokenizer:
    """A `contracts.Tokenizer` stand-in that replays a row batch: `encode(d)`
    returns row `d.t`'s arrays under the protocol's names."""

    def __init__(self, rb):
        self.rows = [N.select_rows(rb, [i]) for i in range(len(rb["n_tok"]))]

    def reset_fight(self):
        pass

    def observe(self, policy):
        pass

    def _fields(self, d, names):
        r = self.rows[d.t]
        return {k: int(r[k][0]) if k in ("sctx_lo", "n_sctx") else r[k] for k in names}

    def encode(self, d):
        return self._fields(d, _ENCODE_FIELDS)

    def encode_priv(self, d):
        return self._fields(d, _PRIV_FIELDS)


def _decisions(rb):
    """`contracts.EnvDecision`s matching the rows of a synthetic batch."""
    out = []
    for i in range(len(rb["n_tok"])):
        r = N.select_rows(rb, [i])
        cls = r["mem_class"]
        rep = np.asarray([int(np.flatnonzero(cls == c)[0]) for c in range(len(r["cls_tok"]))])
        out.append(
            C.EnvDecision(
                run_uid=int(r["run_uid"][0]),
                t=i,
                dtype=int(r["dtype"][0]),
                actions=[{}] * len(cls),
                kinds=r["cls_kind"][cls.astype(np.intp)],
                keys=[str(c) for c in cls],
                member_class=cls,
                rep=rep.astype(np.uint16),
                policy=None,
            )
        )
    return out


def test_decision_row_assembles_the_tokenizer_output():
    rb = _rows(9, seed=43, teacher=True, ppool=True)
    tok, ds = _RowTokenizer(rb), _decisions(rb)
    assert isinstance(tok, C.Tokenizer)
    rows = [N.decision_row(d, tok.encode(d), tok.encode_priv(d)) for d in ds]
    back = N.concat_rows(rows)
    C.validate_row_batch(back)
    acting = {"reward", "logp_behav", "action", "actor", "v_behav", "tprob", "t_src", "t"}
    for k, v in rb.items():
        if k not in acting:
            np.testing.assert_array_equal(back[k], v, err_msg=k)
    assert np.isnan(back["tprob"]).all() and not back["t_src"].any()
    # Without the value path the row carries no PRIV tokens.
    lean = N.decision_row(ds[0], tok.encode(ds[0]))
    assert lean["n_priv"][0] == 0 and lean["pcat"].shape == (0, C.N_CAT)


def test_student_policy_acts_greedily_from_a_checkpoint(tmp_path):
    net = _net()
    path = tmp_path / "student.pt"
    N.save_checkpoint(path, net, vocab_sha256=VOCAB, stamps={})
    rb = _rows(7, seed=6)
    pol = N.StudentPolicy.from_checkpoint(path, vocab_sha256=VOCAB, tokenizer=_RowTokenizer(rb))
    acts = pol.act(rb)
    want = net.policy(N.to_batch(rb))[:7].argmax(-1).tolist()
    assert acts == want and pol.act_decisions(_decisions(rb)) == want
    # An env.E2EDecision acts through its contracts.EnvDecision (`decision`).
    wrapped = [types.SimpleNamespace(decision=d) for d in _decisions(rb)]
    assert pol.act_decisions(wrapped) == want
    assert all(0 <= a < n for a, n in zip(acts, rb["n_classes"], strict=True))
    assert pol.meta["sha256"] == N.file_sha256(path)
    assert pol.act(N.select_rows(rb, [])) == []
    with pytest.raises(N.SchemaMismatch):
        N.StudentPolicy.from_checkpoint(path, vocab_sha256="e" * 64)


class _Counting:
    """WP3b's `tokens.Tokenizer`, counting the env's `reset_fight` and
    `observe` calls."""

    def __init__(self, tk):
        self.tk, self.fights, self.observed = tk, 0, 0

    def reset_fight(self):
        self.fights += 1
        self.tk.reset_fight()

    def observe(self, policy):
        self.observed += 1
        self.tk.observe(policy)

    def encode(self, d):
        return self.tk.encode(d)

    def encode_priv(self, d):
        return self.tk.encode_priv(d)


#: The row fields the student's policy reads (or that place its rows).
_POLICY_FIELDS = [*_ENCODE_FIELDS, "run_uid", "t", "is_combat", "dtype", "n_classes"]
_POLICY_FIELDS += ["n_tok", "n_mem", "n_pool"]


@pytest.fixture
def campaign_process(monkeypatch):
    """The env's factory needs the process to select v0.111.0."""
    import sts2_client
    from ppo.e2e import factory

    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)


def test_student_plays_real_env_decisions_through_the_tokenizer_protocol(campaign_process):
    """A real A10 v0.111 run, env -> WP3b's `contracts.Tokenizer` ->
    `decision_row` -> greedy `StudentPolicy` -> `E2EEnv.step`. Each row's
    policy arrays equal `TokenizerRows`' (what `make_student` reads at
    evaluation, without env hooks). Then the stored rows, PRIV included,
    through the learner path (`forward`, a loss, finite gradients)."""
    TK = pytest.importorskip("ppo.e2e.tokens")
    from ppo.e2e.env import E2EDecision, E2EEnv, EnvConfig

    tok = _Counting(TK.Tokenizer())
    assert isinstance(tok, C.Tokenizer)
    env = E2EEnv(1_300_000_021, cfg=EnvConfig(tokenizer=tok))
    assert env.rm.game_version == "v0.111.0"
    pol = N.StudentPolicy(_net(), tokenizer=tok)
    ref = N.StudentPolicy(pol.net, encode_row=N.TokenizerRows())
    ref.begin_run(1_300_000_021)
    rows, combat_steps = [], 0
    d = env.reset()
    while isinstance(d, E2EDecision) and not d.bootstrap and len(rows) < 160:
        C.validate_decision(d.decision)
        row = N.decision_row(d.decision, tok.encode(d.decision), tok.encode_priv(d.decision))
        want = ref.row(d)
        for k in _POLICY_FIELDS:
            np.testing.assert_array_equal(row[k], want[k], err_msg=f"{k}, row {d.row}")
        rows.append(row)
        combat_steps += d.is_combat
        (c,) = pol.act_decisions([d])
        assert 0 <= c < d.n_classes and ref.act(want) == [c]
        d = env.step(c)
    rb = N.concat_rows(rows)
    C.validate_row_batch(rb)
    assert rb["is_combat"].any() and not rb["is_combat"].all()
    assert len(set(rb["dtype"].tolist())) >= 4  # several decision types met
    assert rb["n_priv"].sum() > 0 and rb["n_pool"].sum() > 0
    assert tok.fights >= 1 and tok.observed >= combat_steps  # forced steps observed too
    net = pol.net.train()
    out = net(N.to_batch(rb))
    act = torch.zeros(len(rows), dtype=torch.long)
    loss = -out.logp[torch.arange(len(rows)), act].mean() + out.value.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)


def test_wp3b_tokenizer_rows_through_the_student(campaign_process):
    """WP3b's tokenizer (`ppo.e2e.tokens`) on a real A10 v0.111 run: the
    student acts on its rows through `StudentPolicy(encode_row=...)`, and the
    stored rows (PRIV included) validate, keep the policy blind to PRIV (T14
    on real rows) and train with finite gradients."""
    TK = pytest.importorskip("ppo.e2e.tokens")
    from ppo.e2e.env import E2EDecision, E2EEnv

    def encode_row(d):
        return TK.row_batch([TK.encode(d, priv=False)], run_uid=[d.run_uid], t=[d.row])

    pol = N.StudentPolicy(_net(), encode_row=encode_row)
    env = E2EEnv(1_300_000_023)
    rng = np.random.default_rng(0)
    toks, acted = [], 0
    d = env.reset()
    while isinstance(d, E2EDecision) and not d.bootstrap and len(toks) < 150:
        toks.append(TK.encode(d, priv=True))
        if len(toks) % 3 == 0:  # the student acts on every third decision
            (c,) = pol.act_decisions([d])
            acted += 1
        else:  # a seeded random actor that prefers playing to ending the turn
            ends = {c for c, k in enumerate(d.cls_kind) if k == C.AK["END_TURN"]}
            others = [c for c in range(d.n_classes) if c not in ends]
            c = int(rng.choice(others)) if others and rng.random() < 0.85 else 0
        assert 0 <= c < d.n_classes
        d = env.step(c)
    rb = TK.row_batch(toks)
    C.validate_row_batch(rb)
    assert acted >= 10 and rb["is_combat"].any() and not rb["is_combat"].all()
    assert rb["n_priv"].sum() > 0 and rb["n_pool"].sum() > 0
    net = pol.net
    b = N.to_batch(rb)
    other = _copy(rb)
    g = np.random.default_rng(1)
    other["pcont"] = (g.standard_normal(other["pcont"].shape) * 9).astype(np.float16)
    other["ppool_vid"] = g.integers(1, C.VOCAB_ROWS, other["ppool_vid"].shape).astype(np.uint16)
    with torch.no_grad():
        assert torch.equal(net.policy(b), net.policy(N.to_batch(other)))
    net.train()
    out = net(b)
    act = torch.as_tensor(rb["action"].astype(np.int64))
    loss = -out.logp[torch.arange(len(toks)), act].mean() + out.value.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)


def test_make_student_plays_evaluate_runs(campaign_process, tmp_path):
    """`ppo.e2e.evaluate`'s plug-in contract: `--policy
    ppo.e2e.net:make_student` loads a checkpoint and its `Player` plays full
    A10 v0.111 runs to an outcome row (not an error row)."""
    pytest.importorskip("ppo.e2e.tokens")
    from ppo.e2e import evaluate as EV

    path = tmp_path / "student.pt"
    N.save_checkpoint(path, _net(), stamps={})
    player = EV.Player("ppo.e2e.env:E2EEnv", "ppo.e2e.net:make_student", str(path), "cpu")
    assert isinstance(player.policy, N.StudentPolicy)
    rows = [player.play(seed) for seed in (1_300_000_031, 1_300_000_032)]
    for r in rows:
        assert r["outcome"] in ("won", "lost", "capped"), r.get("error")
        assert r["n_decisions"] > 0 and r["ascension"] == 10
    # Repeatable: the same seed gives the same row.
    again = player.play(1_300_000_031)
    assert {k: again[k] for k in ("outcome", "n_decisions", "steps", "end")} == {
        k: rows[0][k] for k in ("outcome", "n_decisions", "steps", "end")
    }


def test_recompile_limit_covers_every_static_shape():
    import torch._dynamo.config as dc

    n = N.static_shapes((64, 256))
    assert n == 2 * len(N.LEN_BUCKETS) * len(N.OPT_BUCKETS) * len(N.POOL_BUCKETS) > 8
    # Pools: only the 5 summary tokens own entries, so J is one bucket.
    assert N.to_batch(_rows(6, seed=3)).shape[4] == C.N_SUMMARY_TOKENS == N.POOL_BUCKETS[0]
    old = dc.recompile_limit
    try:
        N.raise_recompile_limit(n)
        assert dc.recompile_limit >= n and dc.accumulated_recompile_limit >= n
        N.raise_recompile_limit(1)  # never lowers it
        assert dc.recompile_limit >= n
    finally:
        dc.recompile_limit = old


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_policy_matches_eager():
    """`torch.compile(fullgraph=True)` traces the policy and the learner path
    without graph breaks, on two bucket shapes. In fp32 the compiled paths
    match eager to rounding. In bf16 they pass X-B's rule against eager fp32
    (plan §9: p99 |Δlogp| < 0.05, mean KL < 1e-3): two bf16 paths differ by
    rounding alone, so a direct bf16-vs-bf16 logit bound only flakes (one
    logit of 479 once differed by 0.051)."""
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    net = _loud_heads(N.E2ENet(seed=0)).cuda().eval()
    pol = N.compile_net(net, batch_sizes=(64,))
    fwd = N.compile_net(net, batch_sizes=(64,), entry="forward")

    def fwd_logits(b, bf16):
        return fwd(b, bf16=bf16).logits

    for seed, body in ((31, (8, 60)), (32, (2, 12))):
        b = N.to_batch(_rows(64, seed=seed, body=body), "cuda", n_rows=64)
        m = b.opt_mask
        with torch.no_grad():
            eager = net.policy(b, bf16=False)
            for path in (pol, fwd_logits):
                torch.testing.assert_close(path(b, bf16=False)[m], eager[m], rtol=0, atol=1e-4)
        for path in (pol, fwd_logits):
            xb = N.precision_gap(net, b, bf16_policy=path)
            assert xb["p99_abs_dlogp"] < 0.05 and xb["mean_kl"] < 1e-3, xb
