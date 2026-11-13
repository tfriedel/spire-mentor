"""GH #1222: Instinct's display-only win probe (`ppo.e2e.probe`) and its export."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch
from ppo.e2e import export_wb as X
from ppo.e2e import net as N
from ppo.e2e import probe as P


def _fake_runs(out, n_runs=60, n=40, dim=16, seed=0):
    """Runs whose outcome is readable from `h[:, 0]`: won iff the run's
    `h[0]` is positive; a fight is lost only in a lost run's last fight."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for s in range(n_runs):
        sign = 1.0 if s % 2 == 0 else -1.0
        h = rng.normal(0, 1, (n, dim)).astype(np.float32)
        h[:, 0] = sign * (1.0 + rng.random(n))
        won = sign > 0
        fight = np.repeat(np.arange(n // 4), 4)
        combat = np.tile([True, True, True, False], n // 4)
        fight = np.where(combat, fight, -1)
        fight_won = np.ones(n, bool)
        if not won:
            fight_won[fight == fight.max()] = False
        hp_loss = np.where(combat, 5.0 + 3.0 * h[:, 1], 0.0).astype(np.float32)
        np.savez(
            out / f"{s}.npz",
            h=h.astype(np.float16),
            v_run=np.full(n, 0.9 if won else 0.1, np.float32),
            v_combat=np.zeros(n, np.float32),
            is_combat=combat,
            dtype=np.zeros(n, np.int64),
            act=np.repeat([0, 1], n // 2),
            floor=np.arange(n),
            fight=fight,
            room=np.zeros(n, np.int64),
            hp=np.full(n, 50),
            max_hp=np.full(n, 80),
            p_chosen=np.ones(n, np.float32),
            reward_in=np.zeros(n, np.float32),
            shaping_to_go=np.zeros(n, np.float32),
            fight_won=fight_won,
            hp_loss=hp_loss,
            won=np.asarray(won),
            outcome=np.asarray("won" if won else "lost"),
            seed=np.asarray(s),
        )


def test_quantiles_never_cross():
    m = P.Probe(8, 16)
    q = m(torch.randn(64, 8))["hp_q"]
    assert q.shape == (64, len(P.QUANTILES))
    assert bool((q[:, 1:] >= q[:, :-1]).all())


def test_calibration_metrics():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert P.ece(p, y) == pytest.approx(0.1)
    assert P._auc(p, y) == pytest.approx(1.0)
    assert P._auc(p, 1 - y) == pytest.approx(0.0)


def test_fit_learns_a_readable_outcome(tmp_path):
    _fake_runs(tmp_path / "d")
    out = tmp_path / "probe.pt"
    meta = P.fit([str(tmp_path / "d")], range(0, 40), range(40, 60), str(out), hidden=32, epochs=40)
    assert meta["dim"] == 16 and out.exists()
    rep = P.report(str(out), [str(tmp_path / "d")], range(40, 60))
    assert rep["all"]["run"]["auc"] > 0.95
    assert rep["all"]["fight"]["rows"] > 0
    cov = rep["all"]["hp"]["coverage"]
    assert cov["q0.1"] < cov["q0.5"] < cov["q0.9"]


def test_export_embeds_the_probe(tmp_path):
    cfg = N.NetConfig(num_hidden=64, dim=64, n_layers=2, ffn=176, head_hidden=64)
    net = N.E2ENet(cfg, seed=0)
    ck = tmp_path / "net.pt"
    N.save_checkpoint(ck, net, vocab_sha256=N.current_vocab_sha256())
    m = P.Probe(64, 32)
    pp = tmp_path / "probe.pt"
    meta = {
        "format": P.FORMAT,
        "dim": 64,
        "hidden": 32,
        "quantiles": list(P.QUANTILES),
        "hp_scale": P.Probe.HP_SCALE,
        "train_seeds": [0, 1],
    }
    torch.save({"meta": meta, "state_dict": m.state_dict()}, pp)
    out = tmp_path / "w.e2ew"
    X.export_weights(str(ck), str(out), str(pp))
    raw = out.read_bytes()
    assert raw[:4] == X.MAGIC
    hl = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + hl])
    assert header["probe"]["hidden"] == 32 and header["probe"]["meta"]["train_seeds"] == [0, 1]
    names = {t["name"] for t in header["tensors"]}
    assert {"probe.l1.weight", "probe.l2.bias", "probe.out.weight"} <= names
    assert not any(n.startswith("value.") for n in names)
    # Without a probe the header says so.
    X.export_weights(str(ck), str(tmp_path / "plain.e2ew"))
    raw = (tmp_path / "plain.e2ew").read_bytes()
    hl = struct.unpack("<I", raw[8:12])[0]
    assert json.loads(raw[12 : 12 + hl])["probe"] is None


def test_estimate_text_matches_the_workbench_wording():
    import realgame_instinct as RI

    assert RI.estimate_text(None) == ""
    assert RI.estimate_text({"p_run": 0.342, "p_fight": None, "hp_loss": None}) == (
        "If Instinct plays on: run 34%"
    )
    e = {
        "p_run": 0.34,
        "p_fight": 0.931,
        "hp_loss": {"q10": 2, "q25": 5, "q50": 9.4, "q75": 14, "q90": 20},
    }
    assert RI.estimate_text(e) == (
        "If Instinct plays on: fight 93% · −9 HP (−2 HP to −20 HP) · run 34%"
    )
    heal = {
        "p_run": 1.0,
        "p_fight": 1.0,
        "hp_loss": {"q10": -6, "q25": -6, "q50": -6, "q75": -6, "q90": -6},
    }
    assert RI.estimate_text(heal) == "If Instinct plays on: fight 100% · +6 HP · run 100%"
