"""Policy and critic nets, batch collation, and checkpoint IO.

Both nets are `CombatCtxNet`s initialised from the step-0 `S_ctx` checkpoint,
with dropout 0 (PPO's probability ratio must not depend on a dropout mask).
The critic has one extra section embedding, `PRIV_DRAW`, initialised as a
copy of `DRAW_TOP`'s row: a privileged true-order draw card starts out read like
a pinned top card. Only the critic's value head is trained for it. Its policy
head is unused.

A policy checkpoint is saved in `combat_ctx.train.save`'s format, so
`combat_ctx.play`'s `ctx:<file>` policy and every `play` / `play-contrast`
read load it unchanged.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import torch

from ppo.combat_ctx import tokens as T
from ppo.combat_ctx.model import CombatCtxNet

from .encode import N_SECTION_CRITIC, SEC_PRIV, Decision, pack


def load_init(path: str) -> tuple[CombatCtxNet, CombatCtxNet, dict[str, Any]]:
    """`(policy, critic, meta)` from an `S_ctx` (or combat-PPO) checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["vocab"] != T.vocab().digest() or ck["n_cont"] != T.N_CONT:
        raise SystemExit(f"{path}: token schema differs from this checkout's")
    cfg = dict(ck["config"], dropout=0.0)
    policy = CombatCtxNet.from_config(cfg)
    policy.load_state_dict(ck["state_dict"])
    critic = CombatCtxNet.from_config(dict(cfg, n_section=N_SECTION_CRITIC))
    sd = dict(ck.get("critic_state_dict") or ck["state_dict"])
    if sd["emb_sec.weight"].shape[0] == T.N_SECTION:
        w = sd["emb_sec.weight"]
        sd["emb_sec.weight"] = torch.cat([w, w[T.SEC["DRAW_TOP"]][None]], 0)
    if any(k.startswith("value_head_run.") for k in sd):
        add_run_head(critic)
    critic.load_state_dict(sd)
    meta = {"arm": ck["arm"], "rare": ck["rare"], "config": cfg, "source": os.path.abspath(path)}
    return policy, critic, meta


def add_run_head(critic: CombatCtxNet) -> CombatCtxNet:
    """Give the critic `value_head_run`: the value of in-run fights under the
    run-value reward (`Cfg.run_reward = "value"`), a different scale from the
    curriculum's fight reward. It reads the same `value_norm` features. Zero
    init: the first updates fit it value-only (`Cfg.policy_frozen_until`)."""
    if getattr(critic, "value_head_run", None) is None:
        head = torch.nn.Linear(critic.config["dim"], 1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        dev = next(critic.parameters()).device
        critic.value_head_run = head.to(dev)
    return critic


def save_policy(
    path: str,
    policy: CombatCtxNet,
    critic: CombatCtxNet | None,
    meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomic save in `combat_ctx.train.save`'s format (+ the critic)."""
    ck = {
        "arm": meta["arm"],
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "config": policy.config,
        "rare": meta["rare"],
        "best": {},
        "settings": dict(extra or {}),
        "vocab": T.vocab().digest(),
        "n_cont": T.N_CONT,
    }
    if critic is not None:
        ck["critic_state_dict"] = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
    tmp = f"{path}.tmp"
    torch.save(ck, tmp)
    pathlib.Path(tmp).replace(path)


def collate(ds: list[Decision], device: str | torch.device, priv: bool) -> dict[str, torch.Tensor]:
    """Pad a list of decisions into net inputs.

    With `priv`, each decision's `PRIV_DRAW` tokens are appended after its
    honest tokens (critic input); without, they are dropped (policy input).
    The action positions index the honest tokens, which come first in both.
    """
    b = len(ds)
    nt = np.asarray([len(d.sec) + (len(d.pcat) if priv else 0) for d in ds])
    na = np.asarray([d.n_act for d in ds])
    tm, am = int(nt.max()), int(na.max())
    cat = np.tile(T.vocab().pads.astype(np.int64), (b, tm, 1))
    cont = np.zeros((b, tm, T.N_CONT), np.float32)
    sec = np.zeros((b, tm), np.int64)
    pad = np.ones((b, tm), bool)
    act_pos = np.zeros((b, am), np.int64)
    amask = np.zeros((b, am), bool)
    for i, d in enumerate(ds):
        h = len(d.sec)
        cat[i, :h] = d.cat
        cont[i, :h] = d.cont
        sec[i, :h] = d.sec
        if priv and len(d.pcat):
            p = len(d.pcat)
            cat[i, h : h + p] = d.pcat
            cont[i, h : h + p] = d.pcont
            sec[i, h : h + p] = SEC_PRIV
        pad[i, : nt[i]] = False
        act_pos[i, : d.n_act] = np.arange(h - d.n_act, h)
        amask[i, : d.n_act] = True
    tt = lambda x: torch.from_numpy(x).to(device, non_blocking=True)  # noqa: E731
    return {
        "cat": tt(cat),
        "cont": tt(cont),
        "sec": tt(sec),
        "pad": tt(pad),
        "act_pos": tt(act_pos),
        "act_mask": tt(amask),
    }


def run(net: CombatCtxNet, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return net(
        x["cat"], x["cont"], x["sec"], x["pad"], x["act_pos"], x["act_mask"], x.get("run_mask")
    )


class Padded:
    """Decisions as one padded array set: honest tokens, then `PRIV_DRAW` tokens.

    The policy and the critic read the same arrays with different key-padding
    masks. The policy's mask covers everything past the honest tokens, so a
    privileged token is never a key for it. Its outputs at those positions
    are computed but never read (`act_pos` points into the honest tokens).
    `test_policy_ignores_masked_priv_tokens` pins that the policy logits
    equal those of `collate(priv=False)`.

    Built with vectorized numpy from flat arrays (`encode.Packed`'s layout),
    once per inference batch or per update. The update keeps the arrays on the
    GPU and slices minibatches by index there.
    """

    def __init__(self, ds: list[Decision], priv: bool = True) -> None:
        self._build(pack(ds), None, priv)

    @classmethod
    def from_flat(cls, p: Any, rows: np.ndarray | None = None, priv: bool = True) -> Padded:
        """Rows `rows` (all if None) of a `Packed` (or anything with its fields)."""
        self = cls.__new__(cls)
        self._build(p, rows, priv)
        return self

    def _build(self, p: Any, rows: np.ndarray | None, priv: bool) -> None:
        n_tok = np.asarray(p.n_tok, np.int64)
        n_priv = np.asarray(p.n_priv, np.int64)
        if rows is None:
            rows = np.arange(len(n_tok))
        rows = np.asarray(rows, np.int64)
        b = len(rows)
        nh = n_tok[rows]
        npv = n_priv[rows] if priv else np.zeros(b, np.int64)
        na = np.asarray(p.n_act, np.int64)[rows]
        tot = nh + npv
        tm, am = int(tot.max()), int(na.max())
        self.cat = np.tile(T.vocab().pads.astype(np.int16), (b, tm, 1))
        self.cont = np.zeros((b, tm, T.N_CONT), np.float16)
        self.sec = np.zeros((b, tm), np.int8)
        # Honest rows: row i's tokens go to columns 0 .. nh[i].
        drow, dcol = _ragged(nh)
        src = np.repeat((np.cumsum(n_tok) - n_tok)[rows], nh) + dcol
        self.cat[drow, dcol] = p.cat[src]
        self.cont[drow, dcol] = p.cont[src]
        self.sec[drow, dcol] = p.sec[src]
        # Priv rows follow the honest ones.
        if priv and int(npv.sum()):
            prow, pw = _ragged(npv)
            psrc = np.repeat((np.cumsum(n_priv) - n_priv)[rows], npv) + pw
            pcol = np.repeat(nh, npv) + pw
            self.cat[prow, pcol] = p.pcat[psrc]
            self.cont[prow, pcol] = p.pcont[psrc]
            self.sec[prow, pcol] = SEC_PRIV
        arow, acol = _ragged(na)
        self.act_pos = np.zeros((b, am), np.int64)
        self.act_pos[arow, acol] = np.repeat(nh - na, na) + acol
        self.n_honest, self.n_total, self.n_act = nh, tot, na

    def to(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        tt = lambda x: torch.from_numpy(x).to(device, non_blocking=True)  # noqa: E731
        return {
            "cat": tt(self.cat),
            "cont": tt(self.cont),
            "sec": tt(self.sec),
            "act_pos": tt(self.act_pos),
            "n_honest": tt(self.n_honest),
            "n_total": tt(self.n_total),
            "n_act": tt(self.n_act),
        }


def _ragged(lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(row, col)` of every cell of a ragged array with row lengths `lens`."""
    row = np.repeat(np.arange(len(lens)), lens)
    start = np.cumsum(lens) - lens
    col = np.arange(int(lens.sum())) - np.repeat(start, lens)
    return row, col


def batch_inputs(
    g: dict[str, torch.Tensor], idx: torch.Tensor | None, critic: bool
) -> dict[str, torch.Tensor]:
    """Net inputs for rows `idx` (all rows if None) of a `Padded.to` dict.

    The token axis is cut to the longest row in the selection.
    """
    sel = (lambda x: x) if idx is None else (lambda x: x[idx])
    n_tot = sel(g["n_total"])
    n_vis = n_tot if critic else sel(g["n_honest"])
    tm = int(n_vis.max())
    n_act = sel(g["n_act"])
    am = int(n_act.max())
    ar = torch.arange(tm, device=n_tot.device)
    pad = ar[None, :] >= n_vis[:, None]
    sec = sel(g["sec"])[:, :tm].long()
    if not critic:
        # The policy has no PRIV_DRAW embedding row: a masked priv position
        # is embedded as section 0 (its output is never read, and it is no key).
        sec = sec.masked_fill(pad, 0)
    return {
        "cat": sel(g["cat"])[:, :tm].long(),
        "cont": sel(g["cont"])[:, :tm].float(),
        "sec": sec,
        "pad": pad,
        "act_pos": sel(g["act_pos"])[:, :am],
        "act_mask": torch.arange(am, device=n_tot.device)[None, :] < n_act[:, None],
        **({"run_mask": sel(g["is_run"])} if critic and "is_run" in g else {}),
    }
