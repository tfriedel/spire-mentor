"""`ppo.e2e.grow_value`: a grown value module keeps the values, the moments and the EMA."""

from __future__ import annotations

import pytest
import torch
from ppo.e2e import algo as A
from ppo.e2e import grow_value as G
from ppo.e2e import net as N


def _state(seed: int = 0) -> tuple[dict, N.E2ENet]:
    """A learner-like state: a net with live value heads, AdamW moments, EMA shadows."""
    net = N.E2ENet(N.NetConfig(n_layers=2), seed=seed)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (net.value.v_combat, net.value.v_run):
            head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.1)
            head.bias.fill_(0.3)
    opt = A.make_optimizer(net, G.VALUE_PREFIXES)
    out = net(N.to_batch(N.synthetic_row_batch(6, seed)))
    (out.v_combat.sum() + out.v_run.sum() + out.logp[out.logp > -1e30].sum()).backward()
    opt.step()
    ema = A.Ema(net.state_dict(), (0.996, 0.999))
    st = {
        "net": N.net_state(net),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
        "algo": {"kept": 1},
    }
    return st, net


def test_grown_net_gives_the_same_values_and_logits() -> None:
    st, net = _state()
    grown = G.grow_state(st, 2)
    assert grown["net"]["cfg"]["net"]["value_layers"] == 2
    assert grown["algo"] == {"kept": 1}
    big = N.net_from_state(grown["net"])
    assert len(big.value.blocks) == 2
    rb = N.synthetic_row_batch(12, 3)
    net.eval()
    big.eval()
    with torch.no_grad():
        a = net(N.to_batch(rb))
        b = big(N.to_batch(rb))
        pa = net.forward_packed(N.to_packed(rb))
        pb = big.forward_packed(N.to_packed(rb))
    assert float(a.v_run.abs().max()) > 0.01  # the heads are live, so equality means something
    torch.testing.assert_close(b.v_combat, a.v_combat, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b.v_run, a.v_run, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b.logp, a.logp)
    torch.testing.assert_close(pb.v_run, pa.v_run, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pb.v_combat, pa.v_combat, rtol=1e-5, atol=1e-5)


def test_grown_optimizer_keeps_moments_by_name_and_loads() -> None:
    st, net = _state(1)
    grown = G.grow_state(st, 2)
    big = N.net_from_state(grown["net"])
    opt = A.make_optimizer(big, G.VALUE_PREFIXES)
    A.load_opt_state(opt, grown["opt"], big, G.VALUE_PREFIXES)
    old_opt = A.make_optimizer(net, G.VALUE_PREFIXES)
    old_opt.load_state_dict(st["opt"])
    old_m = {n: old_opt.state[p] for n, p in net.named_parameters() if p in old_opt.state}
    new_m = {n: opt.state[p] for n, p in big.named_parameters() if p in opt.state}
    assert set(new_m) == set(old_m)  # the new block has none yet
    for n, s in old_m.items():
        torch.testing.assert_close(new_m[n]["exp_avg"], s["exp_avg"])
        torch.testing.assert_close(new_m[n]["exp_avg_sq"], s["exp_avg_sq"])
    # it trains: one step moves the new block's output projection off zero
    out = big(N.to_batch(N.synthetic_row_batch(6, 9)))
    (out.v_run.sum() + out.v_combat.sum()).backward()
    opt.step()
    assert float(big.value.blocks[1].w2.weight.detach().abs().max()) > 0


def test_grown_ema_matches_the_state_dict_and_starts_at_identity() -> None:
    st, _ = _state(2)
    grown = G.grow_state(st, 2)
    big = N.net_from_state(grown["net"])
    keys = set(big.state_dict())
    for sh in grown["ema"]["shadows"].values():
        assert set(sh) == keys
        assert not sh["value.blocks.1.w2.weight"].any()
        assert not sh["value.blocks.1.out.weight"].any()
    ema = A.Ema(big.state_dict(), (0.996, 0.999))
    ema.load_state_dict(grown["ema"])
    ema.update(big.state_dict())  # same keys: no KeyError


def test_grow_refuses_to_shrink() -> None:
    st, _ = _state()
    with pytest.raises(ValueError, match="value_layers 1 <="):
        G.grow_state(st, 1)
