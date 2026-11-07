"""Export the e2e student for the Rust port (the workbench brain): `ppo.e2e.export_wb`.

    PYTHONPATH=<pin>:python <python> -m ppo.e2e.export_wb weights CKPT OUT.e2ew
    PYTHONPATH=<pin>:python <python> -m ppo.e2e.export_wb fixtures CKPT OUT_DIR SEED [SEED ...]

``weights`` writes the policy half of a `save_checkpoint` file (no value
module, no PRIV gain) as one little-endian f32 blob behind a JSON header:

    b"E2EW" | u32 version (1) | u32 header_len | header JSON (utf-8) | tensor data

The header carries the net config, the `Layout` the forward needs (ID and
small columns, their absent values and offsets, the main column), the stamps
(schema, vocab, prior sha256, game version) and one entry per tensor
(`name`, `shape`, `offset` in bytes from the start of the data). The Rust
loader (`sts2-e2e`) refuses a header whose stamps differ from its embedded
tables.

``fixtures`` plays each seed with the greedy student (CPU fp32, the
evaluation path: `make_student` + `E2EEnv`) and writes `OUT_DIR/<seed>.jsonl.gz`:
one header line, one line per decision, one terminal line. A decision line
holds the env's view (`dtype`, `kinds`, `keys`, `member_class`, `members`,
the raw `actions`), the tokenizer's policy row (`cat`, `cont` as float16 bit
patterns, `sec`, `tok_flags`, `cls_tok`, `cls_kind`, `pool_*`, the row
scalars), the net's `logits` (fp32) and the `chosen` class. The Rust gates
read these: G-net feeds `row` to the forward and compares `logits`; G-tok
and G-play replay `chosen` on the same seed and compare the env view and
the row decision by decision.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any

import numpy as np
import torch

MAGIC = b"E2EW"
VERSION = 1
#: Row-batch fields the policy forward reads (the rest are actor or PRIV fields).
POLICY_TOKEN_FIELDS = ("cat", "cont", "sec", "tok_flags")
POLICY_CLASS_FIELDS = ("cls_tok", "cls_kind")
POLICY_POOL_FIELDS = ("pool_tok", "pool_vid", "pool_w")
ENV_MEMBER_FIELDS = ("mem_tok", "mem_class")
ROW_SCALARS = ("is_combat", "dtype", "n_classes", "n_tok", "n_mem", "n_pool", "sctx_lo", "n_sctx")


def _policy_tensor(name: str) -> bool:
    return not (name.startswith("value.") or name == "priv_id_gain")


def layout_header(net: Any) -> dict[str, Any]:
    from . import contracts as C

    L = net.layout
    return {
        "n_cat": L.n_cat,
        "n_cont": L.n_cont,
        "n_prior": L.n_prior,
        "vocab_rows": L.vocab_rows,
        "n_section": L.n_section,
        "n_kind": L.n_kind,
        "cat_absent": list(L.cat_absent),
        "id_cols": list(L.id_cols),
        "main_col": L.main_col,
        "small_cols": list(L.small_cols),
        "small_sizes": list(L.small_sizes),
        "small_absent": list(L.small_absent),
        "small_offsets": list(L.small_offsets),
        "tf_hidden": int(C.TF_HIDDEN),
        "rmsnorm_eps": float(torch.finfo(torch.float32).eps),
    }


def export_weights(ckpt: str, out: str) -> dict[str, Any]:
    from . import net as N

    netm, cfg = N.load_checkpoint(ckpt, device="cpu")
    sd = netm.state_dict()
    tensors, blobs, off = [], [], 0
    for name, t in sd.items():
        if not _policy_tensor(name):
            continue
        a = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4"))
        tensors.append({"name": name, "shape": list(a.shape), "offset": off})
        blobs.append(a.tobytes())
        off += a.nbytes
    stamps = dict(cfg.get("stamps") or {})
    header = {
        "format": "e2ew/1",
        "source": pathlib.Path(ckpt).name,
        "source_sha256": N.file_sha256(ckpt),
        "net": cfg["net"],
        "schema_sha256": cfg.get("schema_sha256"),
        "vocab_sha256": cfg.get("vocab_sha256"),
        "prior_sha256": cfg.get("prior_sha256"),
        "game_version": stamps.get("game_version"),
        "variant": stamps.get("variant"),
        "from_state": stamps.get("from_state"),
        "filled": stamps.get("filled"),
        "layout": layout_header(netm),
        "tensors": tensors,
        "data_bytes": off,
    }
    hb = json.dumps(header, sort_keys=True).encode("utf-8")
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(hb)) + hb)
        for b in blobs:
            f.write(b)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"out": str(p), "bytes": p.stat().st_size, "sha256": sha, "tensors": len(tensors)}


def _arr(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    if a.dtype == np.float16:
        return {"dtype": "f16bits", "shape": list(a.shape), "data": a.view(np.uint16).ravel().tolist()}
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.ravel().tolist()}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def play_fixture(pol: Any, seed: int, out_dir: pathlib.Path) -> dict[str, Any]:
    from .env import E2EDecision, E2EEnv

    pol.begin_run(seed)
    env = E2EEnv(seed)
    d = env.reset()
    out = out_dir / f"{seed}.jsonl.gz"
    n = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"seed": seed, "format": "e2e-fixture/1"}) + "\n")
        while isinstance(d, E2EDecision):
            rb = pol.row(d)
            with torch.no_grad():
                from .net import to_batch

                b = to_batch(rb, pol.device)
                logits = pol.net.policy(b, bf16=False)[0, : d.n_classes].float().cpu().numpy()
            chosen = int(np.argmax(logits))
            rec = {
                "i": n,
                "row": d.row,
                "kind": d.kind,
                "dtype": d.dtype,
                "act": d.act,
                "floor": d.floor,
                "kinds": list(map(int, d.kinds)),
                "keys": list(d.keys),
                "member_class": list(map(int, d.classes)),
                "members": [list(map(int, m)) for m in d.members],
                "actions": _jsonable(list(d.actions)),
                "obs_actions": _jsonable(list(d.obs_actions)),
                "sel": _jsonable(dataclasses.asdict(d.sel)),
                "event_name": d.event_name,
                "turn": d.turn,
                "row_scalars": {k: int(np.asarray(rb[k])[0]) for k in ROW_SCALARS},
                "rowb": {
                    k: _arr(rb[k])
                    for k in POLICY_TOKEN_FIELDS + POLICY_CLASS_FIELDS + POLICY_POOL_FIELDS + ENV_MEMBER_FIELDS
                },
                "logits": [float(x) for x in logits],
                "chosen": chosen,
            }
            f.write(json.dumps(rec) + "\n")
            d = env.step(chosen)
            n += 1
        term = {"terminal": True, "won": bool(d.won), "outcome": str(d.outcome), "decisions": n}
        f.write(json.dumps(term) + "\n")
    return {"seed": seed, "decisions": n, "won": bool(d.won), "file": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppo.e2e.export_wb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("ckpt")
    w.add_argument("out")
    fx = sub.add_parser("fixtures")
    fx.add_argument("ckpt")
    fx.add_argument("out_dir")
    fx.add_argument("seeds", nargs="+", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "weights":
        print(json.dumps(export_weights(a.ckpt, a.out)))
        return 0
    from . import net as N

    torch.set_num_threads(1)
    pol = N.make_student(a.ckpt, device="cpu")
    od = pathlib.Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        print(json.dumps(play_fixture(pol, s, od)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
