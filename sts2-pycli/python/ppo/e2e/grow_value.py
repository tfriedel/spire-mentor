"""Grow the value module of a learner state by blocks: `python -m ppo.e2e.grow_value`.

    PYTHONPATH=<pin>:python <python> -m ppo.e2e.grow_value STATE OUT --value-layers 2

A resume rebuilds the net from the state's own `NetConfig` (`--net` only shapes a fresh
lane), so a lane that wants a deeper critic gets a rewritten state. This writes one:

- **Net.** `value_layers` goes up; each new block is appended after the existing ones
  and starts as an exact identity: its attention output (`out`) and its MLP output
  (`w2`) are zero, so the residual stream passes through unchanged and `V_combat` and
  `V_run` are bit-for-bit the old ones. Every other tensor is copied. The last block is
  still the one that runs at RUNCTX only (`ValueModule.forward`); the old last block
  now runs over every token, which gives the same RUNCTX output.
- **Optimizer.** AdamW state is keyed by position in the flat parameter list, and the
  new parameters sit inside the "value" group, so the saved moments move by parameter
  name to their new positions. The new block starts without moments (AdamW creates
  them at its first step).
- **EMA.** Every shadow gets the new block's tensors (fp32), so the shadows start at
  the identity too.

The algorithm state, the held rows, the learner counters and the stamps are kept. The
module's tests check that a grown net gives the old values.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from typing import Any

import torch

from . import algo as A
from .net import E2ENet, NetConfig

#: As `update.VALUE_PREFIXES`; the learner's states have no "input" group.
VALUE_PREFIXES = ("value.",)


def _names_in_group_order(net: torch.nn.Module) -> dict[str, list[str]]:
    """Parameter names per `algo.param_groups` group, in the group's order."""
    name_of = {id(p): n for n, p in net.named_parameters()}
    groups = A.param_groups(net, VALUE_PREFIXES)
    return {g["name"]: [name_of[id(p)] for p in g["params"]] for g in groups}


def _identity_block(net: E2ENet, i: int) -> dict[str, torch.Tensor]:
    """Block `i` of `net.value` with its output projections zeroed, as state-dict entries."""
    blk = net.value.blocks[i]
    with torch.no_grad():
        blk.out.weight.zero_()
        blk.w2.weight.zero_()
    return {f"value.blocks.{i}.{k}": v.detach().clone() for k, v in blk.state_dict().items()}


def grow_state(st: Mapping[str, Any], value_layers: int, *, seed: int = 0) -> dict[str, Any]:
    """A copy of learner state `st` whose net has `value_layers` value blocks."""
    out = copy.copy(dict(st))
    net_ck = copy.copy(dict(st["net"]))
    cfg = copy.deepcopy(net_ck["cfg"])
    old_nc = NetConfig.from_dict(cfg["net"])
    if value_layers <= old_nc.value_layers:
        raise ValueError(f"value_layers {value_layers} <= the state's {old_nc.value_layers}")
    new_d = {**cfg["net"], "value_layers": int(value_layers)}
    new_nc = NetConfig.from_dict(new_d)
    old_net = E2ENet(old_nc, seed=seed)
    new_net = E2ENet(new_nc, seed=seed)

    sd = dict(net_ck["state_dict"])
    added: dict[str, torch.Tensor] = {}
    for i in range(old_nc.value_layers, new_nc.value_layers):
        added.update(_identity_block(new_net, i))
    if clash := sorted(set(added) & set(sd)):
        raise ValueError(f"the state already holds {clash[:3]}")
    ref = sd["value.blocks.0.qkv.weight"]
    sd.update({k: v.to(dtype=ref.dtype) for k, v in added.items()})
    want = set(new_net.state_dict())
    if set(sd) != want:
        raise ValueError(
            f"grown state dict differs from the net: missing {sorted(want - set(sd))[:3]},"
            f" extra {sorted(set(sd) - want)[:3]}"
        )
    cfg["net"] = new_d
    net_ck["cfg"], net_ck["state_dict"] = cfg, sd
    out["net"] = net_ck

    opt = st.get("opt")
    if opt is not None:
        old_names = _names_in_group_order(old_net)
        new_names = _names_in_group_order(new_net)
        by_name: dict[str, Any] = {}
        for g in opt["param_groups"]:
            names = old_names[g["name"]]
            if len(names) != len(g["params"]):
                raise ValueError(
                    f"saved group {g['name']!r} has {len(g['params'])} parameters,"
                    f" the net {len(names)}"
                )
            for n, j in zip(names, g["params"], strict=True):
                if j in opt["state"]:
                    by_name[n] = opt["state"][j]
        state: dict[int, Any] = {}
        groups = []
        i = 0
        for g in opt["param_groups"]:
            idx = []
            for n in new_names[g["name"]]:
                if n in by_name:
                    state[i] = by_name[n]
                idx.append(i)
                i += 1
            groups.append({k: v for k, v in g.items() if k != "params"} | {"params": idx})
        out["opt"] = {"state": state, "param_groups": groups}

    ema = st.get("ema")
    if ema is not None:
        shadows = {}
        for key, sh in ema["shadows"].items():
            s2 = dict(sh)
            s2.update({k: v.float().clone() for k, v in added.items()})
            shadows[key] = s2
        out["ema"] = {**ema, "shadows": shadows}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ppo.e2e.grow_value", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("state")
    p.add_argument("out")
    p.add_argument("--value-layers", type=int, required=True)
    a = p.parse_args(argv)
    st = torch.load(a.state, map_location="cpu", weights_only=True)
    grown = grow_state(st, a.value_layers)
    torch.save(grown, a.out)
    old = st["net"]["cfg"]["net"]["value_layers"]
    print(f"grow_value: {a.state} value_layers {old} -> {a.value_layers}: wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
