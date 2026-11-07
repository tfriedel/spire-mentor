"""`CombatCtxNet`: the context-aware combat student (GH #830 step 0, DESIGN §c).

A copy of `ppo.model.RunPolicyNet`'s trunk over the combat token schemas of
`tokens`: each token is the sum of a section embedding, one embedding per
categorical slot (PAD rows are zero) and `SinusoidalProj` of its continuous
columns; pre-norm blocks (RMSNorm -> multi-head attention, RMSNorm -> SwiGLU
x4); a final RMSNorm. It differs from `RunPolicyNet` in four ways: the combat
tables, dropout 0.1 on the attention and MLP outputs (the data is smaller than
PPO's), a key-padding mask (fused `scaled_dot_product_attention`), and its
heads: a policy head `Linear(dim, 1)` on the action tokens (uniform +-0.01
init, like PPO's) and a value head `RMSNorm -> Linear(dim, 1)` on the STATE
token (position 0). There are no positional encodings: a token set is
unordered.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ppo.model import SinusoidalProj


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, mult: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.n1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.n2 = nn.RMSNorm(dim)
        hidden = dim * mult
        self.w1 = nn.Linear(dim, hidden)
        self.v = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.p_attn = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.p_attn if self.training else 0.0
        )
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, t, d)))
        h = self.n2(x)
        return x + self.drop(self.w2(F.silu(self.w1(h)) * self.v(h)))


class CombatCtxNet(nn.Module):
    def __init__(
        self,
        cat_sizes: tuple[int, ...],
        n_cont: int,
        n_section: int,
        dim: int = 192,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "cat_sizes": list(cat_sizes),
            "n_cont": n_cont,
            "n_section": n_section,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dropout": dropout,
        }
        self.emb = nn.ModuleList(nn.Embedding(n, dim, padding_idx=n - 1) for n in cat_sizes)
        self.emb_sec = nn.Embedding(n_section, dim)
        self.cont_proj = SinusoidalProj(n_cont, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads, dropout) for _ in range(n_layers))
        self.norm = nn.RMSNorm(dim)
        self.policy_head = nn.Linear(dim, 1)
        self.value_norm = nn.RMSNorm(dim)
        self.value_head = nn.Linear(dim, 1)
        nn.init.uniform_(self.policy_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.policy_head.bias)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CombatCtxNet:
        return cls(
            tuple(cfg["cat_sizes"]),
            cfg["n_cont"],
            cfg["n_section"],
            cfg["dim"],
            cfg["n_layers"],
            cfg["n_heads"],
            cfg["dropout"],
        )

    def forward(
        self,
        cat: torch.Tensor,
        cont: torch.Tensor,
        sec: torch.Tensor,
        pad: torch.Tensor,
        act_pos: torch.Tensor,
        act_mask: torch.Tensor,
        run_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits `[B, A]` (-inf off `act_mask`) and value `[B]`.

        Args:
            cat: `[B, T, N_CAT]` long. cont: `[B, T, N_CONT]`. sec: `[B, T]` long.
            pad: `[B, T]` True = padding. act_pos: `[B, A]` token position of
            each action. act_mask: `[B, A]` True = a real action.
            run_mask: `[B]` True = read the value from `value_head_run`, a
            second head a combat-PPO critic gets for a second reward scale
            (`combat_ppo.nets.add_run_head`). Ignored by a net without it.
        """
        x = self.emb_sec(sec) + self.cont_proj(cont.float())
        for i, e in enumerate(self.emb):
            x = x + e(cat[..., i])
        attn_mask = (~pad)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)
        a = torch.gather(x, 1, act_pos.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        logits = self.policy_head(a).squeeze(-1).float()
        logits = logits.masked_fill(~act_mask, float("-inf"))
        h0 = self.value_norm(x[:, 0])
        value = self.value_head(h0).squeeze(-1).float()
        head_run = getattr(self, "value_head_run", None)
        if run_mask is not None and head_run is not None:
            value = torch.where(run_mask, head_run(h0).squeeze(-1).float(), value)
        return logits, value


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
