"""Policy tokens (honest) and critic-only tokens (privileged), and their packing.

The policy input is exactly `combat_ctx.play.CtxNet`'s: the honest snapshot of
`cs.belief_view(0)` plus the episode's own `IntentLog`, rl2's v1 features, and
`tokens.tokens(..., "S_ctx")`, with the checkpoint's `RareMap` applied and the
continuous columns saturated to float16 (`tokens.f16`).

The critic additionally reads `PRIV_DRAW` tokens: the true next `priv_k` cards of
the draw pile, in order (the pin rank rides in the count column, as for
`DRAW_TOP`). They come from `state_to_dict(cs)` of the TRUE state, so the policy
must never receive them; `pack` keeps them in separate arrays and the policy
forward never reads those arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ppo.combat_ctx import tokens as T
from ppo.combat_ctx.data import RareMap
from ppo.combat_ctx.intent_log import IntentLog
from ppo.combat_ctx.snapshot import _card, honest_snapshot, obs_actions

# The critic's extra section, appended after the token schema's sections.
SEC_PRIV = T.N_SECTION
N_SECTION_CRITIC = T.N_SECTION + 1


@dataclass
class Decision:
    """One encoded decision (numpy arrays, storage dtypes)."""

    cat: np.ndarray  # int16 [T, N_CAT], RareMap applied
    cont: np.ndarray  # float16 [T, N_CONT]
    sec: np.ndarray  # int8 [T]
    n_act: int
    pcat: np.ndarray = field(default_factory=lambda: np.zeros((0, T.N_CAT), np.int16))
    pcont: np.ndarray = field(default_factory=lambda: np.zeros((0, T.N_CONT), np.float16))


class Encoder:
    """Worker-side encoder for one checkpoint's arm and RareMap."""

    def __init__(self, arm: str, rare: dict[str, Any], priv_k: int) -> None:
        if arm != "S_ctx":
            raise ValueError(f"combat PPO trains the S_ctx arm only, not {arm!r}")
        self.arm = arm
        self.rare = RareMap.from_json(rare)
        self.priv_k = priv_k
        self.v = T.vocab()

    @staticmethod
    def new_log() -> IntentLog:
        return IntentLog()

    def observe(self, ilog: IntentLog, cs: Any, room: str) -> tuple[dict[str, Any], Any]:
        """Advance the episode's intent log; return `(snapshot, belief view)`.

        Called at EVERY step, single-option ones included, exactly like
        `play.CtxNet.observe`, so the intent history matches `play.py`'s.
        """
        bv = cs.belief_view(0)
        return ilog.observe(honest_snapshot(bv, room)), bv

    def encode(self, snap: dict[str, Any], bv: Any, cs: Any, acts: list) -> Decision:
        import sts2_rs

        rf = sts2_rs.rollout_features(cs, version=1)
        act_obs = obs_actions(acts, cs.pending_choice(), bv.pending_choice())
        tk = T.tokens(snap, act_obs, rf, self.arm)
        d = Decision(
            cat=self.rare.apply(tk["cat"]).astype(np.int16),
            cont=T.f16(tk["cont"]),
            sec=np.asarray(tk["sec"], np.int8),
            n_act=int(tk["n_act"]),
        )
        if self.priv_k > 0:
            d.pcat, d.pcont = self.priv(cs)
        return d

    def priv(self, cs: Any) -> tuple[np.ndarray, np.ndarray]:
        """`PRIV_DRAW` tokens: the true top `priv_k` draw-pile cards (critic only)."""
        import sts2_rs

        draw = sts2_rs.state_to_dict(cs)["deck"].get("draw_pile", [])[: self.priv_k]
        b = T._Builder(self.v)
        for r, c in enumerate(draw):
            cat, cont = b.new("DRAW_TOP")  # section id replaced below
            T.card_feats(b, cat, cont, _card(c), float(r + 1))
        if not b.cat:
            return np.zeros((0, T.N_CAT), np.int16), np.zeros((0, T.N_CONT), np.float16)
        cat = self.rare.apply(np.stack(b.cat).astype(np.int16)).astype(np.int16)
        return cat, T.f16(np.stack(b.cont))


@dataclass
class Packed:
    """A batch of decisions as flat arrays (what crosses the process boundary)."""

    cat: np.ndarray  # int16 [sum T, N_CAT]
    cont: np.ndarray  # float16 [sum T, N_CONT]
    sec: np.ndarray  # int8 [sum T]
    n_tok: np.ndarray  # int32 [B]
    n_act: np.ndarray  # int32 [B]
    pcat: np.ndarray  # int16 [sum P, N_CAT]
    pcont: np.ndarray  # float16 [sum P, N_CONT]
    n_priv: np.ndarray  # int32 [B]

    def __len__(self) -> int:
        return len(self.n_tok)


def pack(ds: list[Decision]) -> Packed:
    def cat0(xs: list[np.ndarray], shape: tuple[int, ...], dt: Any) -> np.ndarray:
        return np.concatenate(xs) if xs else np.zeros(shape, dt)

    return Packed(
        cat=cat0([d.cat for d in ds], (0, T.N_CAT), np.int16),
        cont=cat0([d.cont for d in ds], (0, T.N_CONT), np.float16),
        sec=cat0([d.sec for d in ds], (0,), np.int8),
        n_tok=np.asarray([len(d.sec) for d in ds], np.int32),
        n_act=np.asarray([d.n_act for d in ds], np.int32),
        pcat=cat0([d.pcat for d in ds], (0, T.N_CAT), np.int16),
        pcont=cat0([d.pcont for d in ds], (0, T.N_CONT), np.float16),
        n_priv=np.asarray([len(d.pcat) for d in ds], np.int32),
    )


def unpack(p: Packed) -> list[Decision]:
    out = []
    t0 = q0 = 0
    for nt, na, npv in zip(p.n_tok, p.n_act, p.n_priv, strict=True):
        out.append(
            Decision(
                cat=p.cat[t0 : t0 + nt],
                cont=p.cont[t0 : t0 + nt],
                sec=p.sec[t0 : t0 + nt],
                n_act=int(na),
                pcat=p.pcat[q0 : q0 + npv],
                pcont=p.pcont[q0 : q0 + npv],
            )
        )
        t0 += int(nt)
        q0 += int(npv)
    return out


def concat_packed(ps: list[Packed]) -> Packed:
    """Several `Packed` batches as one, in order."""
    if len(ps) == 1:
        return ps[0]
    return Packed(
        **{k: np.concatenate([getattr(p, k) for p in ps]) for k in Packed.__dataclass_fields__}
    )
