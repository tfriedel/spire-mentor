"""`update.LossScale`: dynamic loss scaling of an fp16 update (`Cfg.update_amp`)."""

from __future__ import annotations

import math

import pytest
import torch
from ppo.e2e.update import LossScale


def test_a_finite_gradient_is_divided_by_the_scale() -> None:
    ls = LossScale()
    g = torch.tensor([ls.scale * 0.5, -ls.scale * 2.0])
    out = ls.unscale(g)
    assert out is not None
    assert out.tolist() == [0.5, -2.0]
    assert (ls.clean, ls.skipped) == (1, 0)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_a_non_finite_gradient_skips_the_step_and_halves_the_scale(bad: float) -> None:
    ls = LossScale()
    ls.clean = 7
    assert ls.unscale(torch.tensor([1.0, bad])) is None
    assert ls.scale == pytest.approx(LossScale.INIT / 2)
    assert (ls.clean, ls.skipped) == (0, 1)


def test_the_scale_doubles_after_grow_every_clean_steps() -> None:
    ls = LossScale()
    for _ in range(LossScale.GROW_EVERY - 1):
        assert ls.unscale(torch.ones(2)) is not None
    assert ls.scale == pytest.approx(LossScale.INIT)
    ls.unscale(torch.ones(2))
    assert ls.scale == pytest.approx(LossScale.INIT * 2)
    assert ls.clean == 0


def test_the_scale_stays_within_its_bounds() -> None:
    lo = LossScale({"scale": LossScale.MIN, "clean": 0, "skipped": 0})
    lo.unscale(torch.tensor([math.inf]))
    assert lo.scale == pytest.approx(LossScale.MIN)
    hi = LossScale({"scale": LossScale.MAX, "clean": LossScale.GROW_EVERY - 1, "skipped": 0})
    hi.unscale(torch.ones(1))
    assert hi.scale == pytest.approx(LossScale.MAX)


def test_the_state_round_trips() -> None:
    ls = LossScale()
    ls.unscale(torch.tensor([math.inf]))
    ls.unscale(torch.ones(1))
    again = LossScale(ls.state())
    assert (
        again.state()
        == ls.state()
        == {
            "scale": LossScale.INIT / 2,
            "clean": 1,
            "skipped": 1,
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast on CUDA")
def test_a_query_mlp_past_fp16_range_stays_finite_under_fp16() -> None:
    """The value head's RUNCTX MLP runs in fp32 under fp16 autocast: a trained net's
    output there passed 65504 (8.1e4, GH #1535) and turned the value into inf."""
    from ppo.e2e.net import Block

    torch.manual_seed(0)
    blk = Block(32, 4, 64).cuda()
    with torch.no_grad():
        blk.w2.weight.mul_(400_000.0)
    x = torch.randn(3, 5, 32, device="cuda")
    keep = torch.ones(3, 1, 1, 5, dtype=torch.bool, device="cuda")
    ref = blk.forward_first(x, keep)
    assert ref.abs().max() > 65504
    with torch.autocast("cuda", dtype=torch.float16):
        got = blk.forward_first(x, keep)
    assert torch.isfinite(got).all()
    assert torch.allclose(got.float(), ref, rtol=1e-2, atol=1.0)
