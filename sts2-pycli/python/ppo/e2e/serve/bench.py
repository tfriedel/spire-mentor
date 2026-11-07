"""Benchmarks of the S1b serving components (µs per decision).

    python -m ppo.e2e.serve.bench all --json out.json
    python -m ppo.e2e.serve.bench codec | transport | baseline | pipeline

* `codec`: bytes per row (compact, raw, #1081's pickled `Packed`), worker-side
  encode, and decode with numpy, torch on the CPU and torch on CUDA.
* `transport`: a server in this process and one load-generator process that
  simulates 32 workers x 32 envs (`loadgen.run_load`). The server runs the
  batcher with a fake, vectorized `infer`. Reports the server's transport +
  batching time per decision (the plan's S1b target: <= 15 µs), batch sizes,
  and the worker side.
* `baseline`: #1081's per-decision costs on the same decisions, in process:
  `pack` + pickle (worker), unpickle + `unpack` + `concat_packed` +
  `Padded.from_flat` + `batch_inputs` (server), and `Learner._pack` (the
  hand-off to the ranks), against `RowStore.append`.
* `pipeline` (CUDA): the whole server with a `CombatCtxNet` under bf16 and
  the same load process, for each flag set, plus #1081's `_act` computation
  over the same transport.

Two processes at most are busy at a time (this one and the load generator).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from . import cppo
from .batcher import DEFAULT_BUCKETS, Batch, Batcher
from .flags import ServeFlags
from .loadgen import load_process
from .packed import Pipeline, SyncEngine, TorchEngine, pad_ratio
from .rows import RowStore
from .transport import ShmTransport, SlotSpec

CAPS = {cppo.TOK: 256, cppo.PRIV: 64}
TARGET_SERVER_US = 15.0  # plan §8 S1b: lean server 46 -> <= 15 µs per decision
TARGET_ROW_BYTES = 3500  # plan §8 S1b: compact rows 15.7 -> ~3.5 KB


def _decisions(source: str, n: int, seed: int) -> list[Any]:
    from . import sample

    if source == "game":
        return sample.game_decisions(n, seed=seed, game_version=sample.GAME_VERSION)
    return sample.synthetic_decisions(n, seed=seed)


def _timeit(fn: Any, reps: int) -> float:
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


# ── codec ────────────────────────────────────────────────────────────────


def bench_codec(ds: list[Any]) -> dict[str, Any]:
    import torch

    from ppo.combat_ppo.encode import pack

    n = len(ds)
    out: dict[str, Any] = {"rows": n}
    out["pickled_packed_bytes_per_row"] = len(pickle.dumps(pack(ds), pickle.HIGHEST_PROTOCOL)) / n
    rows = [cppo.decision_row(d) for d in ds]
    for compact in (True, False):
        name = "compact" if compact else "raw"
        codec = cppo.cppo_codec(compact)
        cap = codec.max_bytes(CAPS)
        slot = np.zeros(cap, np.uint8)

        def enc(codec: Any = codec, slot: np.ndarray = slot) -> None:
            for r in rows:
                codec.encode(r, out=slot)

        t_enc = _timeit(enc, 2) / n
        outs = [np.zeros(cap, np.uint8) for _ in range(32)]

        def enc_many(codec: Any = codec, outs: list[np.ndarray] = outs) -> None:
            for j in range(0, n - 31, 32):
                codec.encode_many(rows[j : j + 32], outs)

        t_many = _timeit(enc_many, 2) / (n // 32 * 32)
        blobs = [codec.encode(r) for r in rows]
        sizes = np.array([len(b) for b in blobs])
        buf = np.concatenate(blobs)
        offs = np.cumsum(sizes) - sizes
        t_np = _timeit(lambda codec=codec, buf=buf, offs=offs: codec.decode(buf, offs), 2) / n
        bt = torch.from_numpy(buf)
        hdr = codec.headers(buf, offs)
        t_cpu = (
            _timeit(
                lambda codec=codec, bt=bt, offs=offs, hdr=hdr: codec.decode_torch(bt, offs, hdr), 3
            )
            / n
        )
        r = {
            "bytes_per_row": float(sizes.mean()),
            "bytes_p99": float(np.percentile(sizes, 99)),
            "bytes_max": int(sizes.max()),
            "slot_capacity": int(cap),
            "encode_us": t_enc * 1e6,
            "encode_many32_us": t_many * 1e6,
            "decode_numpy_us": t_np * 1e6,
            "decode_torch_cpu_us": t_cpu * 1e6,
        }
        if torch.cuda.is_available():
            g = bt.cuda()

            def dec_gpu(
                codec: Any = codec,
                g: Any = g,
                offs: np.ndarray = offs,
                hdr: Any = hdr,
                buf: np.ndarray = buf,
            ) -> Any:
                return codec.decode_torch(g, offs, hdr, host_buf=buf)

            dec_gpu()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                dec_gpu()
            t_host = (time.perf_counter() - t0) / 10
            torch.cuda.synchronize()
            t_all = (time.perf_counter() - t0) / 10
            r["decode_torch_cuda_host_us"] = t_host / n * 1e6
            r["decode_torch_cuda_total_us"] = t_all / n * 1e6
        out[name] = r
    out["target_row_bytes"] = TARGET_ROW_BYTES
    out["compact_vs_pickled"] = (
        out["pickled_packed_bytes_per_row"] / out["compact"]["bytes_per_row"]
    )
    return out


# ── transport (fake infer) ───────────────────────────────────────────────


def _encoded(codec: Any, ds: list[Any]) -> dict[str, np.ndarray]:
    blobs = [codec.encode(cppo.decision_row(d)) for d in ds]
    sizes = np.array([len(b) for b in blobs], np.int64)
    n_tok = np.array([len(d.sec) for d in ds], np.int64)
    n_act = np.array([d.n_act for d in ds], np.int64)
    return {
        "flat": np.concatenate(blobs),
        "offsets": np.cumsum(sizes) - sizes,
        "sizes": sizes,
        "n_tok": n_tok,
        "n_act": n_act,
    }


def _fake_infer(codec: Any) -> Any:
    def infer(batch: Batch) -> dict[str, np.ndarray]:
        n_act = codec.row_values(batch.buf, batch.offsets)["n_act"].astype(np.int64)
        n_tok = batch.hdr["n_tok"].astype(np.int64)
        n = batch.n
        return {
            "action": ((n_tok * 7 + 3) % n_act).astype(np.int32),
            "logp": np.zeros(n, np.float32),
            "value": np.zeros(n, np.float32),
        }

    return infer


def _serve_with_load(
    codec: Any,
    enc: dict[str, np.ndarray],
    engine_for: Any,
    flags: ServeFlags,
    workers: int,
    envs: int,
    seconds: float,
    max_rows: int,
    pin: bool,
    store: bool,
    check: bool,
    warmup: float = 1.0,
    min_rows: int = 1,
    hold_rows: int = 0,
    hold_s: float = 2e-3,
) -> dict[str, Any]:
    """One server run against one load process; times after `warmup`.

    With `check`, the engine must reply `(n_tok * 7 + 3) % n_act` (the fake),
    and the load process counts mismatches. With `hold_rows`, the server waits
    (not counted as busy) until that many requests are ready or `hold_s` has
    passed, as it would while a GPU forward runs: realistic batch sizes for a
    fake `infer`.
    """
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    expect = ((enc["n_tok"] * 7 + 3) % enc["n_act"]).astype(np.int32)
    ctx = get_context("spawn")
    with ShmTransport(spec, workers, envs) as tr:
        b = Batcher(
            tr,
            buckets=DEFAULT_BUCKETS if flags.buckets else None,
            max_rows=max_rows,
            min_rows=min_rows,
            pin=pin,
        )
        rs = RowStore(codec, capacity_bytes=1 << 26, capacity_rows=1 << 16) if store else None

        def on_done(batch: Batch, out: dict[str, np.ndarray]) -> None:
            if rs is not None:
                if rs.used > (1 << 26) - (1 << 22):
                    rs.n = rs.used = 0  # keep memory bounded: a learner would hand rows off here
                rs.append(batch.buf, batch.offsets, batch.sizes)

        engine = engine_for(codec)
        pipe = Pipeline(b, engine, dbuf=flags.dbuf, on_done=on_done)
        proc, recv = load_process(
            ctx,
            tr.handle(),
            list(range(workers)),
            flat=enc["flat"],
            offsets=enc["offsets"],
            sizes=enc["sizes"],
            n_tok=enc["n_tok"],
            expect_action=expect if check else None,
            seconds=0.0,
        )
        t_start = time.perf_counter()
        t_mark = t_start + warmup
        t_end = t_mark + seconds
        marked = False
        busy = 0.0
        served = 0
        base: dict[str, float] = {}
        rows_hist: list[int] = []
        try:
            while True:
                now = time.perf_counter()
                if not marked and now >= t_mark:
                    marked = True
                    base = {**b.stats, **{f"p_{k}": v for k, v in pipe.stats.items()}}
                    busy = 0.0
                    served = 0
                    rows_hist = []
                if now >= t_end:
                    break
                if hold_rows and not pipe.pending:
                    t_hold = now + hold_s
                    while tr.pending() < hold_rows and time.perf_counter() < t_hold:
                        pass
                t0 = time.perf_counter()
                n = pipe.step()
                dt = time.perf_counter() - t0
                if n or pipe.pending:
                    busy += dt
                    served += n
                    if n:
                        rows_hist.append(n)
                if not proc.is_alive():
                    raise RuntimeError(f"the load process died (exit code {proc.exitcode})")
            t_wall = time.perf_counter() - t_mark
            served += pipe.flush()
        finally:
            tr.stop()
            st = recv.recv() if recv.poll(30) else {}
            proc.join(30)
    now_stats = {**b.stats, **{f"p_{k}": v for k, v in pipe.stats.items()}}
    d = {k: now_stats[k] - base.get(k, 0) for k in now_stats}
    srv = max(1, served)
    res = {
        "flags": str(flags),
        "workers": workers,
        "envs": envs,
        "served": served,
        "decisions_per_s": served / t_wall,
        "server_busy_us": busy / srv * 1e6,
        "gather_us": d["p_t_gather"] / srv * 1e6,
        "copy_us": d["t_copy"] / srv * 1e6,
        "poll_group_us": (d["t_poll"] + d["t_group"]) / srv * 1e6,
        "launch_us": d["p_t_launch"] / srv * 1e6,
        "wait_us": d["p_t_wait"] / srv * 1e6,
        "reply_us": d["p_t_reply"] / srv * 1e6,
        "store_us": d["p_t_done"] / srv * 1e6,
        "transport_batching_us": (d["p_t_gather"] + d["p_t_reply"]) / srv * 1e6,
        "min_rows": min_rows,
        "hold_rows": hold_rows,
        "rows_per_step_mean": float(np.mean(rows_hist)) if rows_hist else 0.0,
        "batches_per_gather": d["batches"] / max(1, d["gathers"]),
        "worker": {
            "served": st.get("served"),
            "mismatch": st.get("mismatch"),
            "submit_us": st.get("t_submit", 0) / max(1, st.get("served", 0)) * 1e6,
            "reply_read_us": st.get("t_reply", 0) / max(1, st.get("served", 0)) * 1e6,
        },
    }
    return res


def bench_transport(ds: list[Any], workers: int, envs: int, seconds: float) -> dict[str, Any]:
    out: dict[str, Any] = {"target_us": TARGET_SERVER_US}
    for hold in (0, 512):
        for compact in (True, False):
            codec = cppo.cppo_codec(compact)
            enc = _encoded(codec, ds)
            for fl in ("shm", "shm,buckets"):
                flags = ServeFlags.parse(fl + (",compact" if compact else ""))
                res = _serve_with_load(
                    codec,
                    enc,
                    lambda c: SyncEngine(_fake_infer(c)),
                    flags,
                    workers,
                    envs,
                    seconds,
                    max_rows=1024,
                    pin=False,
                    store=True,
                    check=True,
                    hold_rows=hold,
                )
                out[f"{flags}" + (f" hold{hold}" if hold else "")] = res
    return out


# ── #1081 baseline ───────────────────────────────────────────────────────


def bench_baseline(ds: list[Any], workers: int, envs: int) -> dict[str, Any]:
    from ppo.combat_ppo import nets
    from ppo.combat_ppo.encode import concat_packed, pack, unpack
    from ppo.combat_ppo.learner import Learner, Step

    per = workers * envs
    ds = (ds * (per // len(ds) + 1))[:per]
    msgs_d = [ds[w * envs : (w + 1) * envs] for w in range(workers)]
    keys = [(e, 0) for e in range(envs)]

    def worker_side() -> list[bytes]:
        return [
            pickle.dumps(("obs", w, keys, pack(dw), [], []), pickle.HIGHEST_PROTOCOL)
            for w, dw in enumerate(msgs_d)
        ]

    msgs = worker_side()
    t_worker = _timeit(worker_side, 3) / per

    def unpickle_unpack() -> tuple[list[Any], list[Any]]:
        packs, out = [], []
        for m in msgs:
            # The bytes come from `worker_side` in this process: safe to unpickle.
            _k, _w, _keys, p, _e, _r = pickle.loads(m)
            dd = unpack(p)
            packs.append(p)
            out.extend(dd)
        return packs, out

    packs, dd = unpickle_unpack()
    t_unpickle = _timeit(unpickle_unpack, 3) / per
    t_concat = _timeit(lambda: concat_packed(packs), 5) / per
    cp = concat_packed(packs)

    def padded() -> Any:
        g = nets.Padded.from_flat(cp, None, priv=False).to("cpu")
        return nets.batch_inputs(g, None, critic=False)

    t_padded = _timeit(padded, 5) / per
    done = [([Step(d, 0, -0.5) for d in ds[i : i + 64]], 1.0, False) for i in range(0, per, 64)]

    def learner_pack() -> None:
        b = Learner._pack(None, done, {})  # type: ignore[arg-type]  # `_pack` reads no attribute of self
        b["shm"].close()
        b["shm"].unlink()

    t_pack = _timeit(learner_pack, 5) / per
    codec = cppo.cppo_codec(True)
    enc = _encoded(codec, ds)
    flat, offs, sizes = enc["flat"], enc["offsets"], enc["sizes"]
    store = RowStore(codec, capacity_bytes=1 << 24, capacity_rows=1 << 14)

    def append() -> None:
        store.n = store.used = 0
        store.append(flat, offs, sizes)

    t_append = _timeit(append, 20) / per
    act: dict[str, float] = {}
    import torch

    if torch.cuda.is_available():
        from ppo.combat_ctx import tokens as T
        from ppo.combat_ctx.model import CombatCtxNet

        torch.manual_seed(0)
        net = CombatCtxNet(T.vocab().sizes, T.N_CONT, T.N_SECTION, 192, 3, 4).cuda().eval()
        gen = torch.Generator(device="cuda")

        @torch.no_grad()
        def act_1081() -> Any:  # learner.py `_act`, line for line
            g = nets.Padded.from_flat(cp, None, priv=False).to("cuda")
            xp = nets.batch_inputs(g, None, critic=False)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = nets.run(net, xp)
            logp_all = torch.log_softmax(logits.float(), -1)
            a = torch.multinomial(logp_all.exp(), 1, generator=gen).squeeze(-1)
            logp = logp_all.gather(1, a[:, None]).squeeze(-1)
            return torch.stack([a.float(), logp]).cpu().numpy()

        act["act_1081_3x192_bf16_us"] = _timeit(act_1081, 10) / per * 1e6
    return {
        **act,
        "batch": per,
        "worker_pack_pickle_us": t_worker * 1e6,
        "server_unpickle_unpack_us": t_unpickle * 1e6,
        "server_concat_packed_us": t_concat * 1e6,
        "server_padded_from_flat_batch_inputs_us": t_padded * 1e6,
        "server_total_before_forward_us": (t_unpickle + t_concat + t_padded) * 1e6,
        "learner_pack_to_shm_us": t_pack * 1e6,
        "rowstore_append_us": t_append * 1e6,
        "pad_ratio_mixed": pad_ratio(cp.n_tok),
    }


# ── pipeline (CUDA) ──────────────────────────────────────────────────────


PIPELINE_CONFIGS = (
    ("shm,compact", 1),
    ("shm,compact,buckets", 1),
    ("shm,compact,buckets", 128),
    ("shm,compact,packed", 1),
    ("shm,compact,dbuf", 1),
    ("shm,compact,buckets,dbuf", 128),
    ("shm,compact,packed,dbuf", 1),
    ("all", 128),
)


def bench_pipeline(
    ds: list[Any],
    workers: int,
    envs: int,
    seconds: float,
    dim: int,
    n_layers: int,
    n_heads: int,
    repeats: int = 1,
    configs: tuple[tuple[str, int], ...] = PIPELINE_CONFIGS,
) -> dict[str, Any]:
    """Each config `repeats` times, interleaved (a busy host drifts); the
    reported numbers are the medians, `runs` holds every run's rate."""
    import torch

    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.model import CombatCtxNet

    if not torch.cuda.is_available():
        return {"skipped": "no CUDA device"}
    torch.manual_seed(0)
    net = CombatCtxNet(T.vocab().sizes, T.N_CONT, T.N_SECTION, dim, n_layers, n_heads).cuda().eval()
    out: dict[str, Any] = {"net": f"{n_layers}x{dim} heads {n_heads}", "bf16": True}
    codec_c = cppo.cppo_codec(True)
    enc_c = _encoded(codec_c, ds)

    runs: dict[str, list[dict[str, Any]]] = {}
    for _rep, (fl, min_rows) in [(r, c) for r in range(repeats) for c in configs]:
        flags = ServeFlags.parse(fl)
        infs: list[Any] = []

        def eng(c: Any, flags: ServeFlags = flags, infs: list[Any] = infs) -> Any:
            inf = cppo.CppoInfer(net, c, device="cuda", bf16=True, packed=flags.packed)
            infs.append(inf)
            return TorchEngine(inf.device_outputs, "cuda")

        r = _serve_with_load(
            codec_c,
            enc_c,
            eng,
            flags,
            workers,
            envs,
            seconds,
            1024,
            True,
            True,
            False,
            min_rows=min_rows,
        )
        st = infs[0].stats
        rows = max(1, st["rows"])
        r["decode_host_us"] = st["t_decode"] / rows * 1e6  # whole run, warm-up included
        r["forward_host_us"] = st["t_forward"] / rows * 1e6
        r["rows_per_batch"] = st["rows"] / max(1, st["calls"])
        runs.setdefault(f"{flags}" + (f" min{min_rows}" if min_rows > 1 else ""), []).append(r)
    for key, rs in runs.items():
        med = {
            k: float(np.median([x[k] for x in rs]))
            for k, v in rs[0].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        out[key] = {**rs[0], **med, "runs": [round(x["decisions_per_s"]) for x in rs]}
    out["repeats"] = repeats
    return out


# ── CLI ──────────────────────────────────────────────────────────────────


def _loadavg() -> str:
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            return " ".join(fh.read().split()[:3])
    except OSError:
        return ""


def _summary(rep: dict[str, Any]) -> list[str]:
    lines = []
    c = rep.get("codec")
    if c:
        lines.append(
            f"rows: compact {c['compact']['bytes_per_row']:.0f} B (target ~{TARGET_ROW_BYTES}), "
            f"raw {c['raw']['bytes_per_row']:.0f} B, #1081 pickled {c['pickled_packed_bytes_per_row']:.0f} B"
        )
        lines.append(
            f"encode (worker) compact {c['compact']['encode_us']:.1f} µs "
            f"({c['compact']['encode_many32_us']:.1f} in rounds of 32), "
            f"raw {c['raw']['encode_us']:.1f} µs ({c['raw']['encode_many32_us']:.1f}) per row"
        )
    b = rep.get("baseline")
    if b:
        lines.append(
            f"#1081 server before forward {b['server_total_before_forward_us']:.1f} µs, "
            f"_pack {b['learner_pack_to_shm_us']:.1f} µs, RowStore.append {b['rowstore_append_us']:.2f} µs per decision"
        )
        if "act_1081_3x192_bf16_us" in b:
            lines.append(
                f"#1081 _act (3x192 bf16, from_flat + forward + sample) {b['act_1081_3x192_bf16_us']:.1f} µs"
            )
    t = rep.get("transport")
    if t:
        for k, v in t.items():
            if isinstance(v, dict):
                lines.append(
                    f"transport [{k}] {v['transport_batching_us']:.2f} µs/decision "
                    f"(target <= {TARGET_SERVER_US}), busy {v['server_busy_us']:.2f}, "
                    f"{v['decisions_per_s']:.0f}/s, {v['rows_per_step_mean']:.0f} rows/step, "
                    f"mismatch {v['worker']['mismatch']}"
                )
    p = rep.get("pipeline")
    if p and "skipped" not in p:
        for k, v in p.items():
            if isinstance(v, dict):
                lines.append(
                    f"pipeline [{k}] {v['decisions_per_s']:.0f}/s, server busy {v['server_busy_us']:.2f} µs "
                    f"(transport+batching {v['transport_batching_us']:.2f}, decode host "
                    f"{v['decode_host_us']:.2f}, forward host {v['forward_host_us']:.2f}, wait {v['wait_us']:.2f}), "
                    f"{v['rows_per_step_mean']:.0f} rows/step, {v['rows_per_batch']:.0f} rows/batch, "
                    f"runs {v['runs']}"
                )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("codec", "transport", "baseline", "pipeline", "all"))
    ap.add_argument("--source", choices=("game", "synthetic"), default="game")
    ap.add_argument("--n", type=int, default=1024, help="distinct decisions to cycle through")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=5.0, help="measured time per server run")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1, help="pipeline: runs per config (medians)")
    ap.add_argument(
        "--configs",
        default="",
        help="pipeline: flag sets to run, ';'-separated, each 'flags' or 'flags@min_rows'",
    )
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="torch intra-op threads (default 1: a busy shared host makes the pool spin)",
    )
    a = ap.parse_args(argv)
    import torch

    torch.set_num_threads(a.torch_threads)
    ds = _decisions(a.source, a.n, a.seed)
    rep: dict[str, Any] = {
        "loadavg_start": _loadavg(),
        "torch_threads": a.torch_threads,
        "source": a.source,
        "n": len(ds),
        "mean_tok": float(np.mean([len(d.sec) for d in ds])),
        "mean_priv": float(np.mean([len(d.pcat) for d in ds])),
        "mean_act": float(np.mean([d.n_act for d in ds])),
    }
    w = a.what
    if w in ("codec", "all"):
        rep["codec"] = bench_codec(ds)
    if w in ("baseline", "all"):
        rep["baseline"] = bench_baseline(ds, a.workers, a.envs)
    if w in ("transport", "all"):
        rep["transport"] = bench_transport(ds, a.workers, a.envs, a.seconds)
    if w in ("pipeline", "all"):
        cfgs = PIPELINE_CONFIGS
        if a.configs:
            cfgs = tuple(
                (c.split("@")[0], int(c.split("@")[1]) if "@" in c else 1)
                for c in a.configs.split(";")
                if c.strip()
            )
        rep["pipeline"] = bench_pipeline(
            ds, a.workers, a.envs, a.seconds, a.dim, a.layers, a.heads, a.repeats, cfgs
        )
    rep["loadavg_end"] = _loadavg()
    text = json.dumps(rep, indent=2, default=float)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    print("\n".join(_summary(rep)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
