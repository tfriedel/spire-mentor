"""WP5b serving components: codec, transport, batcher, pipeline, parity.

CPU-only by default; the CUDA tests skip without a GPU. The shared-memory
transport relies on x86-64 store ordering, so its tests skip elsewhere.
One test spawns one load-generator process (spawn context).
"""

from __future__ import annotations

import dataclasses
import platform
from multiprocessing import get_context

import numpy as np
import pytest
import torch
from ppo.combat_ppo import nets
from ppo.combat_ppo.encode import Decision, pack
from ppo.e2e.serve import (
    DEFAULT_BUCKETS,
    RAW,
    SPARSE,
    WANT_PROBS,
    Batcher,
    Field,
    Pipeline,
    RowCodec,
    RowStore,
    ServeFlags,
    Server,
    ShmTransport,
    SlotSpec,
    SyncEngine,
    WorkerPort,
    cppo,
    cppo_engine,
    pad_ratio,
    parity,
)
from ppo.e2e.serve.loadgen import load_process
from ppo.e2e.serve.sample import synthetic_decisions

X86 = platform.machine().lower() in ("x86_64", "amd64", "x64", "i686", "i386", "x86")
needs_x86 = pytest.mark.skipif(not X86, reason="the shm transport relies on x86-64 store ordering")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
CAPS = {cppo.TOK: 256, cppo.PRIV: 64}


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).reshape(-1).view(np.uint8)


def _decisions(n: int = 96, seed: int = 0, edge: bool = False) -> list[Decision]:
    ds = synthetic_decisions(n, seed=seed, edge_values=edge)
    # An empty privileged stream and a decision with no tokens at all.
    ds[1] = dataclasses.replace(ds[1], pcat=ds[1].pcat[:0], pcont=ds[1].pcont[:0])
    ds[2] = Decision(cat=ds[2].cat[:0], cont=ds[2].cont[:0], sec=ds[2].sec[:0], n_act=1)
    return ds


def _encode_all(codec: RowCodec, ds: list[Decision]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blobs = [codec.encode(cppo.decision_row(d)) for d in ds]
    sizes = np.array([len(b) for b in blobs], np.int64)
    return np.concatenate(blobs), np.cumsum(sizes) - sizes, sizes


def _fake(codec: RowCodec):
    """A deterministic 'policy': action (n_tok * 7 + 3) % n_act, value n_tok."""

    def infer(batch):
        n_act = codec.row_values(batch.buf, batch.offsets)["n_act"].astype(np.int64)
        n_tok = batch.hdr["n_tok"].astype(np.int64)
        n = batch.n
        probs = np.tile(np.arange(1, 65, dtype=np.float32), (n, 1)) / 100
        return {
            "action": ((n_tok * 7 + 3) % n_act).astype(np.int32),
            "logp": np.full(n, -0.25, np.float32),
            "value": n_tok.astype(np.float32),
            "probs": probs,
        }

    return infer


# ── flags ────────────────────────────────────────────────────────────────


def test_flags_parse_and_validate():
    assert not ServeFlags.parse("").any
    assert not ServeFlags.parse("none").any
    assert str(ServeFlags.parse("none")) == "none"
    f = ServeFlags.parse("all")
    assert f.shm and f.buckets and f.packed and f.dbuf and f.compact
    assert ServeFlags.parse(" shm , buckets ") == ServeFlags(shm=True, buckets=True)
    assert ServeFlags.parse(str(f)) == f
    assert ServeFlags.parse("packed,compact") == ServeFlags(packed=True, compact=True)
    with pytest.raises(ValueError, match="unknown"):
        ServeFlags.parse("shm,turbo")
    with pytest.raises(ValueError, match="needs 'shm'"):
        ServeFlags.parse("buckets")
    with pytest.raises(ValueError, match="needs 'shm'"):
        ServeFlags(dbuf=True)
    assert f.stamp() == {
        "shm": True,
        "buckets": True,
        "packed": True,
        "dbuf": True,
        "compact": True,
    }


# ── codec ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("compact", [True, False])
def test_codec_round_trip_is_bitwise(compact):
    ds = _decisions(edge=True)
    codec = cppo.cppo_codec(compact)
    buf, offs, sizes = _encode_all(codec, ds)
    assert np.all(offs % 8 == 0)
    ref = pack(ds)
    assert parity.packed_equal(ref, cppo.to_packed(codec.decode(buf, offs))) == []
    dec = codec.decode_torch(torch.from_numpy(buf), offs)
    host = type(dec)({k: v.numpy() for k, v in dec.fields.items()}, dec.lens)
    assert parity.packed_equal(ref, cppo.to_packed(host)) == []
    assert np.array_equal(dec.row("n_act"), ref.n_act)
    # encode_many writes the same bytes as encode, into slot-sized buffers
    cap = codec.max_bytes(CAPS)
    outs = [np.full(cap, 0xAB, np.uint8) for _ in ds]
    got = codec.encode_many([cppo.decision_row(d) for d in ds], outs)
    assert got == sizes.tolist()
    for i in range(len(ds)):
        assert np.array_equal(outs[i][: got[i]], buf[offs[i] : offs[i] + sizes[i]])
    assert codec.sizes(codec.headers(buf, offs)).tolist() == sizes.tolist()


def test_compact_rows_are_small():
    ds = synthetic_decisions(200, seed=3)
    compact = _encode_all(cppo.cppo_codec(True), ds)[2].mean()
    raw = _encode_all(cppo.cppo_codec(False), ds)[2].mean()
    assert compact <= 3500  # plan §8 S1b: ~3.5 KB per row
    assert raw / compact > 5


def test_codec_is_table_driven():
    fields = (
        Field("k", "int16"),
        Field("v", "float32", 3),
        Field("a", "int8", 0, "s1"),
        Field("b", "uint8", 10, "s1", SPARSE, tuple(range(10))),
        Field("c", "float64", 2, "s1", RAW),
        Field("d", "int64", 9, "s2", SPARSE, -1),
    )
    codec = RowCodec(fields)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        n1, n2 = int(rng.integers(0, 6)), int(rng.integers(0, 4))
        b = np.tile(np.arange(10, dtype=np.uint8), (n1, 1))
        b[rng.random((n1, 10)) < 0.3] = 200
        d = np.full((n2, 9), -1, np.int64)
        d[rng.random((n2, 9)) < 0.5] = rng.integers(-(2**40), 2**40)
        rows.append(
            {
                "k": i - 10,
                "v": np.array([0.5, -0.0, np.nan], np.float32),
                "a": rng.integers(-128, 127, n1).astype(np.int8),
                "b": b,
                "c": rng.normal(size=(n1, 2)),
                "d": d,
            }
        )
    blobs = [codec.encode(r) for r in rows]
    sizes = np.array([len(x) for x in blobs])
    buf, offs = np.concatenate(blobs), np.cumsum(sizes) - sizes
    for dec in (codec.decode(buf, offs), codec.decode_torch(torch.from_numpy(buf), offs)):
        f = {k: np.asarray(v) for k, v in dec.fields.items()}
        assert f["k"].tolist() == [r["k"] for r in rows]
        assert np.array_equal(_bits(f["v"]), _bits(np.stack([r["v"] for r in rows])))
        for name in ("a", "b", "c", "d"):
            assert np.array_equal(_bits(f[name]), _bits(np.concatenate([r[name] for r in rows])))
        assert dec.lens["s1"].tolist() == [len(r["a"]) for r in rows]
        assert dec.lens["s2"].tolist() == [len(r["d"]) for r in rows]
    assert [d["name"] for d in codec.describe()] == list("kvabcd")
    bad = dict(rows[0])
    with pytest.raises(TypeError, match="expected a int8 array"):
        codec.encode({**bad, "a": bad["a"].astype(np.int16)})
    with pytest.raises(ValueError, match="stream 's1'"):
        codec.encode({**bad, "c": np.zeros((len(bad["a"]) + 1, 2))})
    with pytest.raises(ValueError, match="does not fit"):
        codec.encode({**bad, "k": 70000})
    with pytest.raises(ValueError, match="bytes"):
        codec.encode(bad, out=np.zeros(8, np.uint8))
    with pytest.raises(ValueError, match="sparse field needs"):
        Field("x", "int8", 0, "s", SPARSE)


def test_decode_torch_reads_uint16_fields():
    """Review finding NT-3: 12 e2e row fields are uint16 (`n_classes`, `action`,
    `mem_tok`, `pool_vid`, ...); `decode_torch`, the S1b path of both the server and the
    ranks, refused the dtype. It now decodes them bit-exactly as the numpy path does."""
    fields = (
        Field("n", "uint16"),
        Field("t", "uint16", 0, "s"),
        Field("p", "uint16", 4, "s", SPARSE, 0),
    )
    codec = RowCodec(fields)
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(12):
        k = int(rng.integers(0, 5))
        pool = np.zeros((k, 4), np.uint16)
        pool[rng.random((k, 4)) < 0.4] = 65_535
        rows.append(
            {
                "n": int(rng.integers(0, 65_536)),
                "t": rng.integers(0, 65_536, k).astype(np.uint16),
                "p": pool,
            }
        )
    blobs = [codec.encode(r) for r in rows]
    sizes = np.array([len(x) for x in blobs])
    buf, offs = np.concatenate(blobs), np.cumsum(sizes) - sizes
    want = codec.decode(buf, offs).fields
    got = codec.decode_torch(torch.from_numpy(buf), offs).fields
    for name in ("n", "t", "p"):
        g = got[name]
        assert g.dtype == torch.uint16, name
        assert np.array_equal(g.numpy(), np.asarray(want[name])), name
        assert int(g.long().max()) <= 65_535


def test_rowstore_append_select_decode():
    ds = _decisions(64, seed=4)
    codec = cppo.cppo_codec(True)
    buf, offs, sizes = _encode_all(codec, ds)
    cols = (Field("action", "int32"), Field("logp", "float32"))
    st = RowStore(codec, cols, capacity_bytes=1024, capacity_rows=4)  # forces growth
    a = np.arange(64, dtype=np.int32)
    ids1 = st.append(buf, offs[:40], sizes[:40], action=a[:40], logp=-a[:40].astype(np.float32))
    # not back to back: every other row of the rest
    sel = np.arange(40, 64)
    perm = np.concatenate([sel[::2], sel[1::2]])
    ids2 = st.append(buf, offs[perm], sizes[perm], action=a[perm], logp=-a[perm].astype(np.float32))
    assert ids1.tolist() == list(range(40)) and ids2.tolist() == list(range(40, 64))
    order = np.concatenate([np.arange(40), perm])
    assert parity.packed_equal(pack([ds[i] for i in order]), cppo.to_packed(st.decode())) == []
    rev = np.arange(len(st))[::-1].copy()
    sub = st.select(rev)
    assert sub.column("action").tolist() == a[order][rev].tolist()
    assert (
        parity.packed_equal(pack([ds[order[i]] for i in rev]), cppo.to_packed(sub.decode())) == []
    )
    dt = st.decode_torch(rev)
    host = type(dt)({k: v.numpy() for k, v in dt.fields.items()}, dt.lens)
    assert parity.packed_equal(pack([ds[order[i]] for i in rev]), cppo.to_packed(host)) == []
    with pytest.raises(ValueError, match="missing columns"):
        st.append(buf, offs[:1], sizes[:1], action=a[:1])


# ── transport ────────────────────────────────────────────────────────────


@needs_x86
def test_transport_routes_replies_and_probs():
    codec = cppo.cppo_codec(True)
    ds = _decisions(60, seed=5)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, n_workers=3, n_envs=5) as tr:
        ports = [WorkerPort(tr.handle(), w) for w in range(3)]
        b = Batcher(tr, buckets=(32, 48, 64), max_rows=7, pin=False)
        sent = {}
        i = 0
        for _round in range(4):
            for p in ports:
                envs = list(range(5))
                rows = [ds[i + e] for e in envs]
                fl = [WANT_PROBS if (i + e) % 3 == 0 else 0 for e in envs]
                p.submit_many(
                    envs,
                    codec,
                    [cppo.decision_row(d) for d in rows],
                    fl,
                    n_tok=[len(d.sec) for d in rows],
                )
                for e in envs:
                    sent[p.wid, e] = (ds[i + e], fl[e])
                i += 5
            got = 0
            while got < 15:
                b.serve_once(_fake(codec))
                for p in ports:
                    for e in p.poll().tolist():
                        d, fl = sent[p.wid, e]
                        r = p.reply(e)
                        assert r.action == (len(d.sec) * 7 + 3) % d.n_act
                        assert r.value == pytest.approx(len(d.sec)) and r.logp == pytest.approx(
                            -0.25
                        )
                        pr = p.reply_field(e, "probs")
                        if fl & WANT_PROBS:
                            assert np.allclose(pr, np.arange(1, 65) / 100)
                        got += 1
            assert all(p.in_flight().size == 0 for p in ports)
        assert b.stats["deferred"] >= 1  # 15 ready > max_rows 7
        for p in ports:
            p.close()


@needs_x86
def test_transport_double_buffer_contract():
    codec = cppo.cppo_codec(True)
    d0, d1, d2 = _decisions(8, seed=6)[3:6]
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, 1, 1) as tr, WorkerPort(tr.handle(), 0) as p:
        k1 = p.submit(0, codec, cppo.decision_row(d0), n_tok=len(d0.sec))
        with pytest.raises(RuntimeError, match="in flight"):
            p.submit(0, codec, cppo.decision_row(d1), n_tok=len(d1.sec))
        slots = tr.poll()
        seq, rows = tr.take(slots)
        assert seq.tolist() == [k1] and tr.poll().size == 0  # taken, not polled again
        blob_k1 = tr.blob[rows[0], : tr.n_bytes[rows[0]]].copy()
        out = {"action": np.array([1]), "logp": np.array([0.0]), "value": np.array([2.0])}
        tr.reply(slots, rows, seq, out)
        assert p.reply(0).action == 1
        with pytest.raises(RuntimeError, match="no unread reply"):
            p.reply(0)
        # request k + 1 goes to the other buffer: request k's bytes are intact
        k2 = p.submit(0, codec, cppo.decision_row(d1), n_tok=len(d1.sec))
        assert k2 == k1 + 1
        assert np.array_equal(tr.blob[rows[0], : len(blob_k1)], blob_k1)
        seq2, rows2 = tr.take(tr.poll())
        assert rows2[0] != rows[0]
        # reply k stays readable until request k + 2 is published
        assert tr.rep["action"][rows[0]] == 1
        tr.reply(np.array([0]), rows2, seq2, {**out, "action": np.array([0])})
        assert p.wait(timeout=1.0).tolist() == [0] and p.reply(0).action == 0
        p.submit(0, codec, cppo.decision_row(d2), n_tok=len(d2.sec))
        assert tr.poll().tolist() == [0]


@needs_x86
def test_transport_rejects_foreign_layout_and_stops():
    spec = SlotSpec(blob_bytes=256)
    with ShmTransport(spec, 2, 3) as tr:
        h = tr.handle()
        name, off, _shape, _dt = h.layout[-1]
        bad = dataclasses.replace(h, layout=(*h.layout[:-1], (name, off, (1, 1), "|u1")))
        with pytest.raises(RuntimeError, match="layout"):
            WorkerPort(bad, 0)
        with pytest.raises(ValueError, match="out of range"):
            WorkerPort(h, 2)
        p = WorkerPort(h, 1)
        assert p.wait(timeout=0.01).size == 0  # nothing in flight: returns at once
        p.publish(0, 0, n_tok=0)
        assert not p.stopped
        tr.stop()
        assert p.stopped and p.wait().size == 0  # stop wakes a waiting worker
        with pytest.raises(ValueError, match="unknown header"):
            p.publish(1, 0, ep=3)
        p.close()


@needs_x86
def test_spawned_load_process_is_served_correctly():
    codec = cppo.cppo_codec(True)
    ds = synthetic_decisions(48, seed=7)
    blobs = [codec.encode(cppo.decision_row(d)) for d in ds]
    sizes = np.array([len(x) for x in blobs], np.int64)
    n_tok = np.array([len(d.sec) for d in ds], np.int64)
    n_act = np.array([d.n_act for d in ds], np.int64)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, 2, 4) as tr:
        b = Batcher(tr, buckets=DEFAULT_BUCKETS, max_rows=64, pin=False)
        pipe = Pipeline(b, SyncEngine(_fake(codec)), dbuf=True)
        proc, recv = load_process(
            get_context("spawn"),
            tr.handle(),
            [0, 1],
            flat=np.concatenate(blobs),
            offsets=np.cumsum(sizes) - sizes,
            sizes=sizes,
            n_tok=n_tok,
            expect_action=((n_tok * 7 + 3) % n_act).astype(np.int32),
            n_requests=400,
        )
        try:
            while not recv.poll():
                pipe.step()
                assert proc.is_alive() or recv.poll(), f"load process died: {proc.exitcode}"
            st = recv.recv()
        finally:
            tr.stop()
            proc.join(30)
    assert st["served"] >= 400 and st["mismatch"] == 0
    assert proc.exitcode == 0


# ── batcher and pipeline ─────────────────────────────────────────────────


@needs_x86
def test_batcher_buckets_merge_and_defer():
    codec = cppo.cppo_codec(True)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, 1, 12) as tr, WorkerPort(tr.handle(), 0) as p:
        b = Batcher(tr, buckets=(32, 64), max_rows=100, min_rows=3, pin=False)
        assert b.bucket_of(np.array([1, 32, 33, 64, 65, 200])).tolist() == [
            32,
            32,
            64,
            64,
            128,
            256,
        ]
        lens = [10, 20, 40, 40, 40, 50, 70, 10]
        base = synthetic_decisions(1, seed=8)[0]
        for e, n in enumerate(lens):
            d = Decision(
                cat=np.resize(base.cat, (n, base.cat.shape[1])),
                cont=np.resize(base.cont, (n, base.cont.shape[1])),
                sec=np.resize(base.sec, n),
                n_act=2,
            )
            p.submit(e, codec, cppo.decision_row(d), n_tok=n)
        batches = b.gather()
        # 3 rows <= 32, 4 rows <= 64, 1 row of 70 (bucket 128) joins the 64s
        assert [(x.n, x.bucket) for x in batches] == [(3, 32), (5, 128)]
        assert sorted(np.concatenate([x.hdr["n_tok"] for x in batches]).tolist()) == sorted(lens)
        for x in batches:
            dec = codec.decode(x.buf, x.offsets)
            assert dec.lens[cppo.TOK].tolist() == x.hdr["n_tok"].tolist()
            assert pad_ratio(x.hdr["n_tok"], x.bucket) >= 1.0
        assert b.gather() == []  # all taken
    d = synthetic_decisions(1, seed=9)[0]
    size = len(codec.encode(cppo.decision_row(d)))
    small = SlotSpec(blob_bytes=3 * size)  # a staging buffer of one blob holds 3 rows
    with ShmTransport(small, 1, 12) as tr, WorkerPort(tr.handle(), 0) as p:
        b = Batcher(tr, buckets=None, max_rows=100, stage_bytes=small.blob_cap, pin=False)
        for e in range(12):
            p.submit(e, codec, cppo.decision_row(d), n_tok=len(d.sec))
        per_stage = small.blob_cap // size
        assert 3 <= per_stage < 12
        first = b.gather()
        assert first[0].n == per_stage and b.stats["deferred"] == 1
        assert b.ready() == 12 - per_stage  # the rest waits for the next gather
        assert sum(x.n for x in b.gather()) == per_stage


@needs_x86
def test_pipeline_dbuf_replies_everything_once():
    codec = cppo.cppo_codec(True)
    ds = synthetic_decisions(64, seed=10)
    done = []
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, 2, 8) as tr:
        ports = [WorkerPort(tr.handle(), w) for w in range(2)]
        pipe = Pipeline(
            Batcher(tr, pin=False),
            SyncEngine(_fake(codec)),
            dbuf=True,
            on_done=lambda batch, out: done.append((batch.n, out["action"].copy())),
        )
        for p in ports:
            p.submit_many(
                range(8),
                codec,
                [cppo.decision_row(d) for d in ds[:8]],
                n_tok=[len(d.sec) for d in ds[:8]],
            )
        assert pipe.step() == 0  # launched, not yet completed (one batch ahead)
        assert all(p.poll().size == 0 for p in ports)
        assert pipe.flush() == 16
        assert sum(n for n, _ in done) == 16
        for p in ports:
            assert sorted(p.poll().tolist()) == list(range(8))
            for e in range(8):
                assert p.reply(e).action == (len(ds[e].sec) * 7 + 3) % ds[e].n_act
            p.close()


# ── inference and parity with #1081 ──────────────────────────────────────


@pytest.mark.parametrize("t_pad", [None, 128])
def test_policy_inputs_match_batch_inputs_bitwise(t_pad):
    ds = [d for d in synthetic_decisions(48, seed=11) if len(d.sec) <= 120]
    assert parity.policy_input_parity(ds, "cpu", t_pad)["ok"]


def test_packed_and_bucketed_forward_match_1081_in_fp32():
    ds = synthetic_decisions(96, seed=12)
    rep = parity.inference_parity(parity.tiny_net(), ds, "cpu")
    assert rep["ok"], rep
    assert rep["fp32_mixed"]["max_abs_logit"] <= 0.0  # same shapes: bit-identical


def test_priv_permutation_leaves_logits_bit_identical():
    # plan test T14 (S1b part)
    assert parity.priv_permutation_parity(
        parity.tiny_net(), synthetic_decisions(48, seed=13), "cpu"
    )["ok"]


@needs_x86
@pytest.mark.parametrize(("compact", "buckets"), [(True, True), (False, False)])
def test_transport_parity_with_1081_messages(compact, buckets):
    rep = parity.transport_parity(
        synthetic_decisions(64, seed=14), compact, buckets, n_workers=2, n_envs=8
    )
    assert rep["ok"], rep


@pytest.mark.pending_e2e_tables
def test_codec_parity_on_game_decisions():
    from ppo.e2e.serve.sample import game_decisions

    ds = game_decisions(40, seed=1000, game_version="v0.111.0")
    assert len(ds) == 40
    for compact in (True, False):
        assert parity.codec_parity(ds, compact)["ok"]


@needs_x86
def test_server_from_flags_matches_1081_greedy():
    ds = synthetic_decisions(32, seed=15)
    net = parity.tiny_net()
    g = nets.Padded.from_flat(pack(ds), None, priv=False).to("cpu")
    with torch.no_grad():
        logits = nets.run(net, nets.batch_inputs(g, None, critic=False))[0]
    ref = logits.argmax(-1).numpy()
    top2 = logits.topk(2, -1).values
    clear = ((top2[:, 0] - top2[:, 1]) > 1e-3).numpy()  # skip near-ties (fp32 noise)
    assert clear.sum() >= 24
    for fl in ("shm", "all"):
        flags = ServeFlags.parse(fl)
        codec = cppo.cppo_codec(flags.compact)
        eng = cppo_engine(net, codec, flags, device="cpu", greedy=True)
        with Server(codec, 2, 16, eng, flags, caps=CAPS, pin=False) as srv:
            ports = [WorkerPort(srv.handle(), w) for w in range(2)]
            for w, p in enumerate(ports):
                part = ds[w * 16 : (w + 1) * 16]
                p.submit_many(
                    range(16),
                    codec,
                    [cppo.decision_row(d) for d in part],
                    n_tok=[len(d.sec) for d in part],
                )
            while srv.pending():
                srv.step()
            srv.flush()
            got = np.array([ports[i // 16].reply(i % 16).action for i in range(32)])
            assert np.array_equal(got[clear], ref[clear]), fl
            for p in ports:
                p.close()
    with pytest.raises(ValueError, match="'shm'"):
        Server(codec, 1, 1, eng, ServeFlags(), caps=CAPS)


@needs_x86
def test_probs_refuse_more_options_than_the_reply_holds():
    codec = cppo.cppo_codec(True)
    d = synthetic_decisions(1, seed=20)[0]
    inf = cppo.CppoInfer(parity.tiny_net(), codec, greedy=True, max_opts=1)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    with ShmTransport(spec, 1, 2) as tr, WorkerPort(tr.handle(), 0) as p:
        b = Batcher(tr, buckets=None, pin=False)
        p.submit(0, codec, cppo.decision_row(d), WANT_PROBS, n_tok=len(d.sec))
        (batch,) = b.gather()
        with pytest.raises(ValueError, match="max_opts"):
            inf(batch)
        p.submit(1, codec, cppo.decision_row(d), 0, n_tok=len(d.sec))  # no probs: fine
        (batch,) = b.gather()
        assert inf(batch)["action"].shape == (1,)


# ── CUDA ─────────────────────────────────────────────────────────────────


@needs_cuda
def test_cuda_decode_and_policy_inputs_bitwise():
    ds = _decisions(64, seed=16, edge=True)
    for compact in (True, False):
        rep = parity.codec_parity(ds, compact)
        assert rep["ok"] and rep["torch_cuda"] == []
    ds = synthetic_decisions(64, seed=17)
    assert parity.policy_input_parity(ds, "cuda")["ok"]
    assert parity.policy_input_parity(ds, "cuda", 256)["ok"]


@needs_cuda
@needs_x86
def test_cuda_pipeline_dbuf_matches_cpu():
    ds = synthetic_decisions(64, seed=18)
    net = parity.tiny_net()
    g = nets.Padded.from_flat(pack(ds), None, priv=False).to("cpu")
    with torch.no_grad():
        ref = nets.run(net, nets.batch_inputs(g, None, critic=False))[0].argmax(-1).numpy()
    flags = ServeFlags.parse("all")
    codec = cppo.cppo_codec(True)
    eng = cppo_engine(net, codec, flags, device="cuda", bf16=False, greedy=True)
    with Server(codec, 1, 64, eng, flags, caps=CAPS) as srv:
        assert srv.batcher.pinned
        p = WorkerPort(srv.handle(), 0)
        p.submit_many(
            range(64), codec, [cppo.decision_row(d) for d in ds], n_tok=[len(d.sec) for d in ds]
        )
        while srv.pending():
            srv.step()
        srv.flush()
        got = [p.reply(e).action for e in range(64)]
        p.close()
    assert np.mean(np.array(got) == ref) >= 0.95  # fp32 on two devices: near-ties may flip


@needs_cuda
def test_cuda_bf16_varlen_inference_parity():
    rep = parity.inference_parity(parity.tiny_net(), synthetic_decisions(96, seed=19), "cuda")
    assert rep["ok"], rep
    assert rep["bf16_packed_varlen"]["within_bf16_atol"]
