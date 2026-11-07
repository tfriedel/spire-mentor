"""#1081's token layout through the serving stack.

`cppo_fields` is the codec table for `combat_ppo.encode.Decision`: the honest
tokens (`tok`), the critic-only `PRIV_DRAW` tokens (`priv`) and `n_act`. With
`compact=True` the two wide arrays per stream use the sparse codec. The
categorical slots leave out their vocabulary PAD ids, the continuous columns
leave out zeros. With `compact=False` every field is stored raw.

`decision_row` turns a `Decision` into a codec row. `to_packed` turns decoded
rows back into a `Packed`, the arrays #1081's learner hands to `nets.Padded`.
Both directions are bitwise exact (`parity.codec_parity` checks it).

`policy_inputs` builds #1081's policy inputs (`nets.batch_inputs(critic=False)`)
on the device from decoded rows, padded to a chosen length. `CppoInfer` is a
reference `infer(batch)` for a `CombatCtxNet`: decode, pad (or pack), forward,
sample. It is the "new path" of the inference parity check and the model in
the GPU pipeline benchmark. The end-to-end net (WP4) plugs in the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .packed import cu_seqlens
from .rows import RAW, SPARSE, Decoded, Field, RowCodec

TOK = "tok"
PRIV = "priv"


def cppo_fields(compact: bool = True) -> tuple[Field, ...]:
    """The codec table for #1081's `Decision` (honest + privileged tokens)."""
    from ppo.combat_ctx import tokens as T

    pads = tuple(int(x) for x in T.vocab().pads)
    kind = SPARSE if compact else RAW
    return (
        Field("n_act", "int32"),
        Field("sec", "int8", 0, TOK),
        Field("cat", "int16", T.N_CAT, TOK, kind, pads),
        Field("cont", "float16", T.N_CONT, TOK, kind, 0),
        Field("pcat", "int16", T.N_CAT, PRIV, kind, pads),
        Field("pcont", "float16", T.N_CONT, PRIV, kind, 0),
    )


def cppo_codec(compact: bool = True) -> RowCodec:
    return RowCodec(cppo_fields(compact))


def decision_row(d: Any) -> dict[str, Any]:
    """A `combat_ppo.encode.Decision` as a codec row (no copies)."""
    return {
        "n_act": d.n_act,
        "sec": d.sec,
        "cat": d.cat,
        "cont": d.cont,
        "pcat": d.pcat,
        "pcont": d.pcont,
    }


def to_packed(dec: Decoded) -> Any:
    """Decoded rows as `combat_ppo.encode.Packed` (numpy, storage dtypes)."""
    from ppo.combat_ppo.encode import Packed

    f = dec.fields
    return Packed(
        cat=np.asarray(f["cat"]),
        cont=np.asarray(f["cont"]),
        sec=np.asarray(f["sec"]),
        n_tok=np.asarray(dec.lens[TOK], np.int32),
        n_act=np.asarray(f["n_act"], np.int32),
        pcat=np.asarray(f["pcat"]),
        pcont=np.asarray(f["pcont"]),
        n_priv=np.asarray(dec.lens[PRIV], np.int32),
    )


def packed_from_decisions(ds: list[Any]) -> Any:
    from ppo.combat_ppo.encode import pack

    return pack(ds)


def _rows_cols(lens: Any, total: int) -> tuple[Any, Any]:
    """`(row, col)` of each token of rows with lengths `lens` (a torch int64
    vector summing to `total`, known on the host so nothing syncs)."""
    import torch

    dev = lens.device
    row = torch.repeat_interleave(torch.arange(len(lens), device=dev), lens, output_size=total)
    start = torch.cumsum(lens, 0) - lens
    col = torch.arange(total, device=dev) - torch.repeat_interleave(start, lens, output_size=total)
    return row, col


_PADS: dict[str, Any] = {}


def _pads(dev: Any) -> Any:
    """The vocabulary PAD ids as an int64 tensor on `dev` (cached)."""
    import torch

    from ppo.combat_ctx import tokens as T

    key = str(dev)
    if key not in _PADS:
        _PADS[key] = torch.from_numpy(T.vocab().pads.astype(np.int64)).to(dev)
    return _PADS[key]


def _up(dev: Any, up: Any) -> Any:
    if up is not None:
        return up
    import torch

    return lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def policy_inputs(dec: Decoded, t_pad: int | None = None, up: Any = None) -> dict[str, Any]:
    """#1081's policy inputs from decoded rows, built on their device.

    Equals `nets.batch_inputs(nets.Padded.from_flat(p, priv=False).to(dev),
    None, critic=False)` for `p = to_packed(dec)` when `t_pad` is None (pad to
    the longest row). A larger `t_pad` (a length bucket) only adds padding.
    `up` uploads the small host arrays (see `rows.RowCodec.decode_torch`).
    """
    import torch

    f = dec.fields
    cat, cont, sec = f["cat"], f["cont"], f["sec"]
    dev = cat.device
    up = _up(dev, up)
    n_tok = np.asarray(dec.lens[TOK], np.int64)
    n_act = dec.row("n_act").astype(np.int64)
    b = len(n_tok)
    tm = int(n_tok.max()) if t_pad is None else int(t_pad)
    if b and int(n_tok.max()) > tm:
        raise ValueError(f"a row has {int(n_tok.max())} tokens, more than the padded length {tm}")
    am = int(n_act.max()) if b else 0
    lens = up(np.stack([n_tok, n_act]))
    row, col = _rows_cols(lens[0], int(n_tok.sum()))
    pads = _pads(dev)
    cat_p = pads.expand(b, tm, len(pads)).clone()
    cat_p[row, col] = cat.long()
    cont_p = torch.zeros((b, tm, cont.shape[1]), dtype=torch.float32, device=dev)
    cont_p[row, col] = cont.float()
    sec_p = torch.zeros((b, tm), dtype=torch.long, device=dev)
    sec_p[row, col] = sec.long()
    ar = torch.arange(tm, device=dev)
    pad = ar[None, :] >= lens[0][:, None]
    sec_p = sec_p.masked_fill(pad, 0)
    aa = torch.arange(am, device=dev)
    act_mask = aa[None, :] < lens[1][:, None]
    act_pos = ((lens[0] - lens[1])[:, None] + aa[None, :]).masked_fill(~act_mask, 0)
    return {
        "cat": cat_p,
        "cont": cont_p,
        "sec": sec_p,
        "pad": pad,
        "act_pos": act_pos,
        "act_mask": act_mask,
    }


def packed_forward(net: Any, dec: Decoded, attn: str = "auto", up: Any = None) -> tuple[Any, Any]:
    """A `CombatCtxNet` forward over pad-free packed tokens.

    Embeddings, norms and MLPs run on the `[sum T, d]` token matrix with no
    padding. Attention runs per row: with `attn="varlen"` through
    `torch.nn.attention.varlen.varlen_attn` (CUDA, bf16/fp16), with
    `attn="pad"` by scattering q, k, v into a padded batch for
    `scaled_dot_product_attention` and back. `"auto"` picks varlen when it
    can run. Returns logits `[B, A]` (-inf off the actions) and value `[B]`,
    like `CombatCtxNet.forward` (the net must be in eval mode: no dropout).
    """
    import torch
    import torch.nn.functional as F

    f = dec.fields
    cat, cont, sec = f["cat"], f["cont"], f["sec"]
    dev = cat.device
    up = _up(dev, up)
    n_tok = np.asarray(dec.lens[TOK], np.int64)
    n_act = dec.row("n_act").astype(np.int64)
    b = len(n_tok)
    tm = int(n_tok.max())
    lens = up(np.stack([n_tok, n_act]))
    cu = up(cu_seqlens(n_tok))
    # SinusoidalProj flattens from dim 2: give it a batch of one row of all tokens.
    x = net.emb_sec(sec.long()) + net.cont_proj(cont.float()[None])[0]
    for i, e in enumerate(net.emb):
        x = x + e(cat[:, i].long())
    n = x.shape[0]
    d = x.shape[1]
    if attn == "auto":
        attn = "varlen" if dev.type == "cuda" and torch.is_autocast_enabled("cuda") else "pad"
    row = col = None
    if attn == "pad":
        row, col = _rows_cols(lens[0], int(n_tok.sum()))
    for blk in net.blocks:
        h = blk.n1(x)
        qkv = blk.qkv(h).view(n, 3, blk.heads, d // blk.heads)
        q, k, v = qkv.unbind(1)
        if attn == "varlen":
            from torch.nn.attention.varlen import varlen_attn

            dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else q.dtype
            a = varlen_attn(q.to(dt), k.to(dt), v.to(dt), cu, cu, tm, tm).to(q.dtype)
        else:
            hd = (blk.heads, d // blk.heads)
            qp = q.new_zeros((b, tm, *hd))
            kp = k.new_zeros((b, tm, *hd))
            vp = v.new_zeros((b, tm, *hd))
            qp[row, col], kp[row, col], vp[row, col] = q, k, v
            keep = torch.arange(tm, device=dev)[None, :] < lens[0][:, None]
            ap = F.scaled_dot_product_attention(
                qp.transpose(1, 2),
                kp.transpose(1, 2),
                vp.transpose(1, 2),
                attn_mask=keep[:, None, None, :],
            )
            a = ap.transpose(1, 2)[row, col]
        x = x + blk.out(a.reshape(n, d))
        h = blk.n2(x)
        x = x + blk.w2(F.silu(blk.w1(h)) * blk.v(h))
    x = net.norm(x)
    am = int(n_act.max())
    aa = torch.arange(am, device=dev)
    act_mask = aa[None, :] < lens[1][:, None]
    pos = (cu[1:].long() - lens[1])[:, None] + aa[None, :]
    pos = pos.masked_fill(~act_mask, 0)
    logits = net.policy_head(x[pos]).squeeze(-1).float()
    logits = logits.masked_fill(~act_mask, float("-inf"))
    h0 = net.value_norm(x[cu[:-1].long()])
    value = net.value_head(h0).squeeze(-1).float()
    return logits, value


class CppoInfer:
    """A reference `infer(batch)` for a #1081 `CombatCtxNet` policy.

    Decodes the batch's blobs on `device`, pads the honest tokens to the
    batch's bucket (or packs them), runs the net and samples (or takes the
    argmax). Returns numpy `action`, `logp`, `value` and, if any row asked,
    `probs` (`[B, max_opts]`, zero past each row's options).
    """

    def __init__(
        self,
        net: Any,
        codec: RowCodec,
        device: Any = "cpu",
        bf16: bool = False,
        greedy: bool = False,
        seed: int = 0,
        packed: bool = False,
        max_opts: int = 64,
    ) -> None:
        import torch

        self.net = net.to(device).eval()
        self.codec = codec
        self.device = torch.device(device)
        self.bf16 = bf16 and self.device.type == "cuda"
        self.greedy = greedy
        self.packed = packed
        self.max_opts = max_opts
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        # Host time spent enqueuing work (no device sync): decode vs forward + sampling.
        self.stats = {"calls": 0, "rows": 0, "t_decode": 0.0, "t_forward": 0.0}

    def decode(self, batch: Any) -> Decoded:
        """The batch's rows on the device (`batch.extra["up"]` uploads, if set)."""
        import torch

        if getattr(batch, "buf_t", None) is not None:
            t = batch.buf_t
        else:
            t = torch.from_numpy(batch.buf)
        if self.device.type != "cpu":
            t = t.to(self.device, non_blocking=True)
        hdr = self.codec.headers(batch.buf, batch.offsets)
        up = batch.extra.get("up") if hasattr(batch, "extra") else None
        return self.codec.decode_torch(t, batch.offsets, hdr, host_buf=batch.buf, up=up)

    def device_outputs(self, batch: Any, logits: bool = False) -> dict[str, Any]:
        """The outputs as device tensors (no host sync)."""
        import time

        import torch

        t0 = time.perf_counter()
        dec = self.decode(batch)
        if batch.want_probs and batch.n and int(dec.row("n_act").max()) > self.max_opts:
            # plan §8: <= 64 options, asserted; a wider distribution would be cut off
            raise ValueError(
                f"a decision has {int(dec.row('n_act').max())} options, more than max_opts {self.max_opts}"
            )
        t1 = time.perf_counter()
        up = batch.extra.get("up") if hasattr(batch, "extra") else None
        with (
            torch.no_grad(),
            torch.autocast("cuda", torch.bfloat16, enabled=self.bf16),
        ):
            if self.packed:
                lg, value = packed_forward(self.net, dec, up=up)
            else:
                from ppo.combat_ppo import nets

                t_pad = batch.bucket or None
                lg, value = nets.run(self.net, policy_inputs(dec, t_pad, up=up))
        with torch.no_grad():
            logp_all = torch.log_softmax(lg.float(), -1)
            if self.greedy:
                a = logp_all.argmax(-1)
            else:
                a = torch.multinomial(logp_all.exp(), 1, generator=self.gen).squeeze(-1)
            out = {
                "action": a.to(torch.int32),
                "logp": logp_all.gather(1, a[:, None]).squeeze(-1),
                "value": value.float(),
            }
            if logits:
                out["logits"] = lg
            if batch.want_probs:
                p = logp_all.exp()
                k = min(p.shape[1], self.max_opts)
                probs = torch.zeros(
                    (p.shape[0], self.max_opts), dtype=torch.float32, device=p.device
                )
                probs[:, :k] = p[:, :k]
                out["probs"] = probs
        st = self.stats
        st["calls"] += 1
        st["rows"] += batch.n
        st["t_decode"] += t1 - t0
        st["t_forward"] += time.perf_counter() - t1
        return out

    def __call__(self, batch: Any) -> dict[str, np.ndarray]:
        out = self.device_outputs(batch)
        return {k: v.cpu().numpy() for k, v in out.items()}
