"""Parity of the S1b serving path with #1081's.

Each check runs the same decisions through #1081's functions and through the
new component, and compares the results:

* `codec_parity` (bitwise): encode, then decode with numpy, with torch on the
  CPU and, if present, on CUDA, and through a `RowStore`. The result must equal
  `combat_ppo.encode.pack(ds)` bit for bit, for the raw and the compact table.
* `transport_parity` (bitwise): #1081 pickles one `pack`ed message per worker,
  the learner unpickles, `unpack`s and `concat_packed`s them. The new path
  writes each decision into its slot; the batcher gathers (bucketed or not);
  the rows are decoded. Each request's arrays must be equal bit for bit, and
  each worker must read the reply computed for its own request.
* `policy_input_parity` (bitwise): `cppo.policy_inputs` against
  `nets.batch_inputs(nets.Padded.from_flat(p, priv=False).to(dev), None,
  critic=False)`; with a bucket length, the extra columns must be padding.
* `inference_parity` (fp32 tolerance): logits, log-probs and values of the
  bucketed and the packed forward against #1081's padded forward, and greedy
  agreement wherever the top two logits differ by more than `gap`. On CUDA
  it also reports bf16 (varlen) against #1081's bf16 forward, for information.
* `priv_permutation_parity` (bitwise, plan test T14): permuting a decision's
  `PRIV_DRAW` tokens leaves the served logits bit-identical.

Run all of them: `python -m ppo.e2e.serve.parity [--source game|synthetic]`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from typing import Any

import numpy as np

from . import cppo
from .batcher import DEFAULT_BUCKETS, Batcher
from .rows import RowStore
from .transport import ShmTransport, SlotSpec, worker_ports

PACKED_FIELDS = ("cat", "cont", "sec", "n_tok", "n_act", "pcat", "pcont", "n_priv")
CAPS = {cppo.TOK: 256, cppo.PRIV: 64}


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.reshape(-1).view(np.uint8)


def packed_equal(a: Any, b: Any) -> list[str]:
    """The `Packed` fields that differ in dtype, shape or bits (empty: equal)."""
    bad = []
    for k in PACKED_FIELDS:
        x, y = getattr(a, k), getattr(b, k)
        if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(_bits(x), _bits(y)):
            bad.append(k)
    return bad


def _encode_all(codec: Any, ds: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blobs = [codec.encode(cppo.decision_row(d)) for d in ds]
    sizes = np.array([len(b) for b in blobs], np.int64)
    offs = np.cumsum(sizes) - sizes
    return np.concatenate(blobs), offs, sizes


def _devices() -> list[str]:
    import torch

    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def codec_parity(ds: list[Any], compact: bool) -> dict[str, Any]:
    import torch

    from ppo.combat_ppo.encode import pack

    codec = cppo.cppo_codec(compact)
    ref = pack(ds)
    buf, offs, sizes = _encode_all(codec, ds)
    out: dict[str, Any] = {
        "rows": len(ds),
        "bytes_per_row": float(sizes.mean()),
        "max_bytes": int(sizes.max()),
    }
    out["numpy"] = packed_equal(ref, cppo.to_packed(codec.decode(buf, offs)))
    for dev in _devices():
        t = torch.from_numpy(buf).to(dev)
        dec = codec.decode_torch(t, offs, codec.headers(buf, offs), host_buf=buf)
        host = type(dec)({k: v.cpu().numpy() for k, v in dec.fields.items()}, dec.lens)
        out[f"torch_{dev}"] = packed_equal(ref, cppo.to_packed(host))
    store = RowStore(codec, capacity_bytes=1 << 16, capacity_rows=16)
    half = len(ds) // 2
    store.append(buf, offs[:half], sizes[:half])
    store.append(buf, offs[half:], sizes[half:])
    out["store"] = packed_equal(ref, cppo.to_packed(store.decode()))
    rev = np.arange(len(ds))[::-1].copy()
    out["store_select"] = packed_equal(
        pack([ds[i] for i in rev]), cppo.to_packed(store.select(rev).decode())
    )
    out["ok"] = not any(v for k, v in out.items() if isinstance(v, list))
    return out


def _fake_action(n_tok: np.ndarray, n_act: np.ndarray) -> np.ndarray:
    """A deterministic 'policy' both paths can compute from a row."""
    return ((np.asarray(n_tok, np.int64) * 7 + 3) % np.asarray(n_act, np.int64)).astype(np.int32)


def transport_parity(
    ds: list[Any],
    compact: bool,
    buckets: bool,
    n_workers: int = 4,
    n_envs: int = 8,
    max_rows: int = 1024,
) -> dict[str, Any]:
    from ppo.combat_ppo.encode import concat_packed, pack, unpack

    codec = cppo.cppo_codec(compact)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    per_round = n_workers * n_envs
    rounds = len(ds) // per_round
    bad_arrays = 0
    bad_replies = 0
    served = 0
    t_old = t_new = 0.0
    with ShmTransport(spec, n_workers, n_envs) as tr:
        ports = worker_ports(tr.handle(), range(n_workers))
        b = Batcher(
            tr,
            buckets=DEFAULT_BUCKETS if buckets else None,
            max_rows=max_rows,
            pin=False,
        )
        try:
            for r in range(rounds):
                chunk = ds[r * per_round : (r + 1) * per_round]
                # #1081: one pickled message per worker, unpickled and concatenated.
                # (The bytes are made two lines up in this process: safe to unpickle.)
                t0 = time.perf_counter()
                packs = []
                for w in range(n_workers):
                    dw = chunk[w * n_envs : (w + 1) * n_envs]
                    keys = [(e, 0) for e in range(n_envs)]
                    msg = pickle.dumps(("obs", w, keys, pack(dw), [], []), pickle.HIGHEST_PROTOCOL)
                    _k, _w, _keys, packed, _ends, _rows = pickle.loads(msg)
                    packs.append(pack(unpack(packed)))
                old = concat_packed(packs)
                t_old += time.perf_counter() - t0
                old_act = _fake_action(old.n_tok, old.n_act)
                # new: slots, gather, decode; the replies go back through the slots.
                t0 = time.perf_counter()
                for w, p in enumerate(ports):
                    for e in range(n_envs):
                        d = chunk[w * n_envs + e]
                        p.submit(e, codec, cppo.decision_row(d), n_tok=len(d.sec))
                got: dict[int, Any] = {}
                while len(got) < per_round:
                    for batch in b.gather():
                        dec = codec.decode(batch.buf, batch.offsets)
                        act = _fake_action(dec.lens[cppo.TOK], dec["n_act"])
                        n = batch.n
                        b.reply(
                            batch,
                            {
                                "action": act,
                                "logp": np.zeros(n, np.float32),
                                "value": np.zeros(n, np.float32),
                            },
                        )
                        wid, env = tr.worker_env(batch.slots)
                        pk = cppo.to_packed(dec)
                        for j, (w, e) in enumerate(zip(wid.tolist(), env.tolist(), strict=True)):
                            got[w * n_envs + e] = _row_of(pk, j)
                t_new += time.perf_counter() - t0
                new = concat_packed([got[i] for i in range(per_round)])
                if packed_equal(old, new):
                    bad_arrays += 1
                for w, p in enumerate(ports):
                    for e in p.poll().tolist():
                        if p.reply(e).action != int(old_act[w * n_envs + e]):
                            bad_replies += 1
                        served += 1
        finally:
            for p in ports:
                p.close()
    return {
        "rounds": rounds,
        "requests": rounds * per_round,
        "replies_read": served,
        "bad_rounds": bad_arrays,
        "bad_replies": bad_replies,
        "batches": int(b.stats["batches"]),
        "old_s": t_old,
        "new_s": t_new,
        "ok": bad_arrays == 0 and bad_replies == 0 and served == rounds * per_round,
    }


def _row_of(p: Any, j: int) -> Any:
    from ppo.combat_ppo.encode import pack, unpack

    return pack([unpack(p)[j]])


def policy_input_parity(ds: list[Any], device: str, t_pad: int | None = None) -> dict[str, Any]:
    import torch

    from ppo.combat_ppo import nets
    from ppo.combat_ppo.encode import pack

    codec = cppo.cppo_codec(True)
    buf, offs, _sizes = _encode_all(codec, ds)
    t = torch.from_numpy(buf).to(device)
    dec = codec.decode_torch(t, offs, codec.headers(buf, offs), host_buf=buf)
    new = cppo.policy_inputs(dec, t_pad)
    g = nets.Padded.from_flat(pack(ds), None, priv=False).to(device)
    ref = nets.batch_inputs(g, None, critic=False)
    bad = []
    tm = ref["cat"].shape[1]
    for k, v in ref.items():
        x = new[k]
        if k in ("cat", "cont", "sec", "pad") and t_pad:
            extra = x[:, tm:]
            x = x[:, :tm]
            if k == "pad" and not bool(extra.all()):
                bad.append("pad_tail")
            if k == "sec" and bool(extra.any()):
                bad.append("sec_tail")
            if k == "cont" and bool(extra.any()):
                bad.append("cont_tail")
        if x.dtype != v.dtype or x.shape != v.shape or not torch.equal(x, v):
            bad.append(k)
    return {"device": device, "t_pad": t_pad, "rows": len(ds), "bad": bad, "ok": not bad}


def _ref_forward(net: Any, ds: list[Any], device: str, bf16: bool) -> tuple[Any, Any]:
    import torch

    from ppo.combat_ppo import nets
    from ppo.combat_ppo.encode import pack

    g = nets.Padded.from_flat(pack(ds), None, priv=False).to(device)
    xp = nets.batch_inputs(g, None, critic=False)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=bf16):
        logits, value = nets.run(net, xp)
    return logits.float().cpu(), value.float().cpu()


def _served_forward(
    net: Any, ds: list[Any], device: str, bf16: bool, packed: bool, buckets: bool
) -> tuple[Any, Any]:
    """Logits and values for `ds` through transport, batcher and `CppoInfer`."""
    import torch

    codec = cppo.cppo_codec(True)
    inf = cppo.CppoInfer(net, codec, device=device, bf16=bf16, greedy=True, packed=packed)
    spec = SlotSpec(blob_bytes=codec.max_bytes(CAPS))
    n = len(ds)
    logits = torch.full((n, 64), float("-inf"))
    value = torch.zeros(n)
    with ShmTransport(spec, 1, n) as tr:
        (port,) = worker_ports(tr.handle(), [0])
        b = Batcher(tr, buckets=DEFAULT_BUCKETS if buckets else None, max_rows=n, pin=False)
        try:
            for e, d in enumerate(ds):
                port.submit(e, codec, cppo.decision_row(d), n_tok=len(d.sec))
            for batch in b.gather():
                out = inf.device_outputs(batch, logits=True)
                lg = out["logits"].float().cpu()
                _w, env = tr.worker_env(batch.slots)
                env_t = torch.from_numpy(env)
                logits[env_t, : lg.shape[1]] = lg
                value[env_t] = out["value"].float().cpu()
                b.reply(batch, {k: v.cpu().numpy() for k, v in out.items() if k != "logits"})
        finally:
            port.close()
    return logits, value


def _compare(ref: tuple[Any, Any], new: tuple[Any, Any], gap: float) -> dict[str, Any]:
    import torch

    lr, vr = ref
    ln, vn = new
    a = lr.shape[1]
    ln = ln[:, :a]
    fin = torch.isfinite(lr)
    mask_ok = bool(torch.equal(fin, torch.isfinite(ln)))
    d_logit = float((lr[fin] - ln[fin]).abs().max()) if fin.any() else 0.0
    lpr, lpn = torch.log_softmax(lr, -1), torch.log_softmax(ln, -1)
    d_logp = float((lpr[fin] - lpn[fin]).abs().max()) if fin.any() else 0.0
    d_value = float((vr - vn).abs().max())
    top2 = lr.masked_fill(~fin, float("-inf")).topk(min(2, a), -1).values
    clear = (top2[:, 0] - top2[:, -1]) > gap if a > 1 else torch.ones(len(lr), dtype=torch.bool)
    agree = lr.argmax(-1) == ln.argmax(-1)
    return {
        "mask_equal": mask_ok,
        "max_abs_logit": d_logit,
        "max_abs_logp": d_logp,
        "max_abs_value": d_value,
        "clear_rows": int(clear.sum()),
        "clear_disagree": int((~agree & clear).sum()),
        "clear_agree": float(agree[clear].float().mean()) if clear.any() else 1.0,
        "all_agree": float(agree.float().mean()),
    }


def inference_parity(
    net: Any,
    ds: list[Any],
    device: str,
    atol: float = 1e-4,
    gap: float = 0.05,
    bf16_atol: float = 0.1,
) -> dict[str, Any]:
    net = net.to(device).eval()
    out: dict[str, Any] = {"device": device, "rows": len(ds), "atol": atol}
    ref = _ref_forward(net, ds, device, bf16=False)
    ok = True
    for name, packed, buckets in (
        ("bucketed", False, True),
        ("mixed", False, False),
        ("packed", True, True),
    ):
        c = _compare(ref, _served_forward(net, ds, device, False, packed, buckets), gap)
        c["ok"] = (
            c["mask_equal"]
            and c["max_abs_logit"] <= atol
            and c["max_abs_value"] <= atol
            and c["clear_disagree"] == 0
        )
        ok = ok and c["ok"]
        out[f"fp32_{name}"] = c
    if device == "cuda":
        ref16 = _ref_forward(net, ds, device, bf16=True)
        for name, packed in (("bucketed", False), ("packed_varlen", True)):
            c = _compare(ref16, _served_forward(net, ds, device, True, packed, True), gap)
            c["info_only"] = True
            c["within_bf16_atol"] = c["max_abs_logit"] <= bf16_atol
            out[f"bf16_{name}"] = c
    out["ok"] = ok
    return out


def priv_permutation_parity(net: Any, ds: list[Any], device: str, seed: int = 0) -> dict[str, Any]:
    """Plan test T14 for S1b: permuted `PRIV_DRAW` tokens, bit-identical logits."""
    import dataclasses

    import torch

    rng = np.random.default_rng(seed)
    perm = []
    for d in ds:
        k = len(d.pcat)
        order = rng.permutation(k) if k else np.arange(0)
        perm.append(dataclasses.replace(d, pcat=d.pcat[order].copy(), pcont=d.pcont[order].copy()))
    net = net.to(device).eval()
    res = {}
    for packed in (False, True):
        a = _served_forward(net, ds, device, False, packed, True)
        b = _served_forward(net, perm, device, False, packed, True)
        res["packed" if packed else "bucketed"] = bool(
            torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        )
    return {"device": device, "rows": len(ds), **res, "ok": all(res.values())}


def tiny_net(seed: int = 0, dim: int = 32, n_layers: int = 1, n_heads: int = 2) -> Any:
    """A small random `CombatCtxNet` over #1081's tables (eval mode)."""
    import torch

    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.model import CombatCtxNet

    torch.manual_seed(seed)
    net = CombatCtxNet(T.vocab().sizes, T.N_CONT, T.N_SECTION, dim, n_layers, n_heads)
    # A random policy head init of +-0.01 makes every logit nearly equal;
    # widen it so greedy agreement is a real test.
    torch.nn.init.normal_(net.policy_head.weight, 0.0, 0.5)
    return net.eval()


def run_all(ds: list[Any], net: Any | None = None) -> dict[str, Any]:
    import torch

    net = tiny_net() if net is None else net
    rep: dict[str, Any] = {}
    for compact in (True, False):
        rep[f"codec_{'compact' if compact else 'raw'}"] = codec_parity(ds, compact)
        for buckets in (False, True):
            key = f"transport_{'compact' if compact else 'raw'}_{'buckets' if buckets else 'mixed'}"
            rep[key] = transport_parity(ds, compact, buckets)
    for dev in _devices():
        rep[f"policy_inputs_{dev}"] = policy_input_parity(ds[:256], dev)
        rep[f"policy_inputs_{dev}_t256"] = policy_input_parity(ds[:256], dev, 256)
        rep[f"inference_{dev}"] = inference_parity(net, ds[:512], dev)
        rep[f"priv_perm_{dev}"] = priv_permutation_parity(net, ds[:128], dev)
    rep["ok"] = all(v.get("ok", True) for v in rep.values() if isinstance(v, dict))
    rep["torch"] = torch.__version__
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=("game", "synthetic"), default="game")
    ap.add_argument("--n", type=int, default=1024, help="decisions (a multiple of 32)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--game-version", default="v0.111.0")
    ap.add_argument("--json", default="", help="also write the report here")
    a = ap.parse_args(argv)
    from . import sample

    t0 = time.perf_counter()
    if a.source == "game":
        ds = sample.game_decisions(a.n, seed=a.seed, game_version=a.game_version)
    else:
        ds = sample.synthetic_decisions(a.n, seed=a.seed)
    rep = run_all(ds)
    rep["source"] = a.source
    rep["game_version"] = a.game_version if a.source == "game" else None
    rep["n"] = len(ds)
    rep["seconds"] = time.perf_counter() - t0
    text = json.dumps(rep, indent=2, default=str)
    print(text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
