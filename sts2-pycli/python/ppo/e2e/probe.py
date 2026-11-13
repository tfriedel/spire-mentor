"""Instinct's display-only win probe (GH #1222): `ppo.e2e.probe`.

    PYTHONPATH=<pin>:python <python> -m ppo.e2e.probe collect CKPT OUT_DIR SEED_START N --procs 8
    PYTHONPATH=python <python> -m ppo.e2e.probe fit OUT_DIR... --train-seeds A:B --val-seeds C:D --out PROBE.pt
    PYTHONPATH=python <python> -m ppo.e2e.probe report PROBE.pt OUT_DIR... --seeds C:D

The student's policy forward already computes `h_STATE` (the normed trunk
output at RUNCTX, token 0). The probe is a small MLP on that vector with three
heads, all meaning "if Instinct plays on from here":

- `run`: P(win the run). Trained on the outcome (BCE) plus a distillation term
  towards the privileged `V_run` with the realized shaping still to come
  subtracted (`V_run` predicts the shaped return: +1 win, +0.05 per boss,
  +0.002 per floor, so `V_run - shaping_to_go` estimates the win term alone).
- `fight`: P(win the current fight), combat rows only, outcome only.
- `hp`: quantiles of the HP the fight costs from here (`hp_now - hp_after`,
  `hp_after` 0 on a death), combat rows only, pinball loss.

`collect` plays greedy runs (CPU fp32, the evaluation path) and writes one
`<seed>.npz` per run: per decision `h` (float16 `[n, D]`), `v_run`,
`v_combat`, `is_combat`, `act`, `floor`, `fight`, `room`, `hp`, `max_hp`,
`p_chosen`, then the labels `won`, `fight_won`, `hp_loss`, `shaping_to_go`.
Resumable: a seed whose file exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOMS = ("monster", "elite", "boss")
#: HP-loss quantiles the `hp` head predicts.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
FORMAT = "e2e-probe/1"


# ── collect ──────────────────────────────────────────────────────────────────


def _hp(d: Any) -> tuple[int, int]:
    if d.is_combat and d.d_bv is not None:
        p = d.d_bv.get("player") or {}
        return int(p.get("hp", 0)), int(p.get("max_hp", 0))
    return int(d.rm.current_hp), int(d.rm.max_hp)


@torch.no_grad()
def _forward(net: Any, rb: Any) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Logits, h_STATE, V_combat, V_run of a one-row priv row batch (fp32)."""
    from .net import to_batch

    b = to_batch(rb, "cpu")
    x, h = net.trunk(b)
    logits = net._logits(h, b)
    e_priv = net.embed(b.pcat, b.pcont, b.psec, b.ppool_t, b.ppool_ok, b.ppool_cnt, priv=True)
    vc, vr = net.value(x, b.pad, e_priv, b.ppad)
    return (
        logits[0].float().numpy(),
        h[0, 0].float().numpy(),
        float(vc[0]),
        float(vr[0]),
    )


def play_run(pol: Any, seed: int) -> dict[str, np.ndarray]:
    """One greedy run with the per-decision probe inputs and labels."""
    from . import tokens
    from .env import E2EDecision, E2EEnv

    pol.begin_run(seed)
    env = E2EEnv(seed)
    d = env.reset()
    rec: dict[str, list[Any]] = {
        k: []
        for k in (
            "h",
            "v_run",
            "v_combat",
            "is_combat",
            "dtype",
            "act",
            "floor",
            "fight",
            "room",
            "hp",
            "max_hp",
            "p_chosen",
            "reward_in",
        )
    }
    while isinstance(d, E2EDecision):
        tk = tokens.encode(d, priv=True)
        rb = tokens.row_batch([tk], run_uid=[d.run_uid], t=[d.row])
        logits, h, vc, vr = _forward(pol.net, rb)
        lg = logits[: d.n_classes]
        chosen = int(np.argmax(lg))
        p = np.exp(lg - lg.max())
        hp, mhp = _hp(d)
        rec["h"].append(h.astype(np.float16))
        rec["v_run"].append(vr)
        rec["v_combat"].append(vc)
        rec["is_combat"].append(bool(d.is_combat))
        rec["dtype"].append(int(d.dtype))
        rec["act"].append(int(d.act))
        rec["floor"].append(int(d.floor))
        rec["fight"].append(-1 if d.fight is None else int(d.fight))
        rec["room"].append(ROOMS.index(d.room) if d.room in ROOMS else -1)
        rec["hp"].append(hp)
        rec["max_hp"].append(mhp)
        rec["p_chosen"].append(float(p[chosen] / p.sum()))
        rec["reward_in"].append(float(d.reward))
        d = env.step(chosen)
    term = d
    n = len(rec["v_run"])
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["h"] = np.stack(rec["h"]) if n else np.zeros((0, 0), np.float16)
    won = bool(term.won)
    # rewards[t]: the reward on the transition from row t (the next row's
    # `reward_in`, or the terminal's for the last row).
    r = np.append(out["reward_in"][1:], float(term.reward)) if n else np.zeros(0)
    togo = np.cumsum(r[::-1])[::-1]
    out["shaping_to_go"] = (togo - (1.0 if won else 0.0)).astype(np.float32)
    # fight labels: a fight is won if a row after its last row exists, or the run is won
    fight = out["fight"]
    fight_won = np.zeros(n, bool)
    hp_loss = np.zeros(n, np.float32)
    end_hp = int(env.rm.current_hp) if won else 0
    for f in np.unique(fight[fight >= 0]):
        idx = np.flatnonzero(fight == f)
        last = idx[-1]
        if last + 1 < n:
            ok, after = True, int(out["hp"][last + 1])
        else:
            ok, after = won, end_hp
        fight_won[idx] = ok
        hp_loss[idx] = out["hp"][idx] - (after if ok else 0)
    out["fight_won"] = fight_won
    out["hp_loss"] = hp_loss
    out["won"] = np.asarray(won)
    out["outcome"] = np.asarray(str(term.outcome))
    out["seed"] = np.asarray(seed)
    return out


_POL: Any = None


def _work(args: tuple[str, int, str]) -> dict[str, Any]:
    ckpt, seed, out_dir = args
    global _POL
    torch.set_num_threads(1)
    if _POL is None:
        from . import net as N

        _POL = N.make_student(ckpt, device="cpu")
    t = time.perf_counter()
    try:
        rec = play_run(_POL, seed)
    except Exception as ex:
        return {"seed": seed, "error": repr(ex)[:500]}
    path = pathlib.Path(out_dir) / f"{seed}.npz"
    tmp = path.with_name(f".{seed}.tmp.npz")
    arrays: dict[str, Any] = dict(rec)
    np.savez_compressed(tmp, **arrays)
    pathlib.Path(tmp).replace(path)
    return {
        "seed": seed,
        "won": bool(rec["won"]),
        "n": len(rec["v_run"]),
        "s": round(time.perf_counter() - t, 1),
    }


def collect(ckpt: str, out_dir: str, start: int, n: int, procs: int) -> None:
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    todo = [(ckpt, s, out_dir) for s in range(start, start + n) if not (od / f"{s}.npz").exists()]
    print(f"collect: {len(todo)} of {n} seeds to play", flush=True)
    log = (od / "collect.log").open("a", encoding="utf-8")
    wins = done = 0
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for r in pool.imap_unordered(_work, todo, chunksize=1):
            log.write(json.dumps(r) + "\n")
            log.flush()
            done += 1
            wins += int(r.get("won", False))
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(
                    f"{done}/{len(todo)} wins {wins} ({wins / done:.3f}) "
                    f"{el / 60:.1f} min, eta {el / done * (len(todo) - done) / 60:.0f} min",
                    flush=True,
                )


# ── data ─────────────────────────────────────────────────────────────────────


def _seed_range(s: str) -> range:
    a, b = s.split(":")
    return range(int(a), int(b))


def load(dirs: Sequence[str], seeds: range | None = None) -> dict[str, np.ndarray]:
    """Concatenate the per-run files of `dirs` (optionally only `seeds`)."""
    parts: dict[str, list[np.ndarray]] = {}
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("*.npz")):
            if p.name.startswith("."):
                continue
            seed = int(p.stem)
            if seeds is not None and seed not in seeds:
                continue
            z = np.load(p)
            n = len(z["v_run"])
            if n == 0:
                continue
            for k in (
                "h",
                "v_run",
                "v_combat",
                "is_combat",
                "act",
                "floor",
                "fight",
                "room",
                "hp",
                "max_hp",
                "fight_won",
                "hp_loss",
                "shaping_to_go",
            ):
                parts.setdefault(k, []).append(z[k])
            parts.setdefault("won", []).append(np.full(n, bool(z["won"])))
            parts.setdefault("seed", []).append(np.full(n, seed, np.int64))
    if not parts:
        raise SystemExit(f"no runs in {list(dirs)}")
    return {k: np.concatenate(v) for k, v in parts.items()}


# ── the probe ────────────────────────────────────────────────────────────────


class Probe(nn.Module):
    """`h_STATE -> SiLU MLP -> [run logit, fight logit, hp quantiles]`.

    `hp` quantiles are `base + cumsum(softplus(deltas))`, so they never cross.
    HP is in units of 10 inside the net (`HP_SCALE`).
    """

    HP_SCALE = 10.0

    def __init__(self, dim: int, hidden: int = 256, n_q: int = len(QUANTILES)) -> None:
        super().__init__()
        self.l1 = nn.Linear(dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2 + n_q)
        self.n_q = n_q

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z = F.silu(self.l1(h))
        z = F.silu(self.l2(z))
        o = self.out(z)
        q = o[:, 2:3] + torch.cat(
            [torch.zeros_like(o[:, 2:3]), torch.cumsum(F.softplus(o[:, 3:]), -1)], -1
        )
        return {"run": o[:, 0], "fight": o[:, 1], "hp_q": q * self.HP_SCALE}


def _targets(z: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    v_win = np.clip(z["v_run"] - z["shaping_to_go"], 0.0, 1.0)
    return {
        "h": torch.from_numpy(z["h"].astype(np.float32)),
        "won": torch.from_numpy(z["won"].astype(np.float32)),
        "v_win": torch.from_numpy(v_win.astype(np.float32)),
        "combat": torch.from_numpy(z["is_combat"].astype(bool)),
        "fight_won": torch.from_numpy(z["fight_won"].astype(np.float32)),
        "hp_loss": torch.from_numpy(z["hp_loss"].astype(np.float32)),
    }


def losses(
    out: dict[str, torch.Tensor], t: dict[str, torch.Tensor], w_distil: float
) -> dict[str, torch.Tensor]:
    p_run = torch.sigmoid(out["run"])
    l_run = F.binary_cross_entropy_with_logits(out["run"], t["won"])
    l_dist = F.mse_loss(p_run, t["v_win"])
    c = t["combat"]
    if c.any():
        l_fight = F.binary_cross_entropy_with_logits(out["fight"][c], t["fight_won"][c])
        q = torch.tensor(QUANTILES)
        err = t["hp_loss"][c, None] - out["hp_q"][c]
        l_hp = (torch.maximum(q * err, (q - 1) * err)).mean() / Probe.HP_SCALE
    else:
        l_fight = l_hp = out["run"].sum() * 0
    total = l_run + w_distil * l_dist + l_fight + l_hp
    return {"total": total, "run": l_run, "distil": l_dist, "fight": l_fight, "hp": l_hp}


def fit(
    dirs: Sequence[str],
    train: range,
    val: range,
    out: str,
    *,
    hidden: int = 256,
    epochs: int = 30,
    w_distil: float = 4.0,
    lr: float = 1e-3,
    wd: float = 1e-4,
    batch: int = 4096,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    tr, va = _targets(load(dirs, train)), _targets(load(dirs, val))
    dim = tr["h"].shape[1]
    m = Probe(dim, hidden)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(tr["won"])
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best, best_state, hist = math.inf, None, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ix = perm[i : i + batch]
            loss = losses(m(tr["h"][ix]), {k: v[ix] for k, v in tr.items()}, w_distil)
            opt.zero_grad()
            loss["total"].backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            lv = {k: float(v) for k, v in losses(m(va["h"]), va, w_distil).items()}
        hist.append({"epoch": ep, **lv})
        print(json.dumps(hist[-1]), flush=True)
        if lv["total"] < best:
            best = lv["total"]
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    assert best_state is not None
    m.load_state_dict(best_state)
    meta = {
        "format": FORMAT,
        "dim": dim,
        "hidden": hidden,
        "quantiles": list(QUANTILES),
        "hp_scale": Probe.HP_SCALE,
        "train_seeds": [train.start, train.stop],
        "val_seeds": [val.start, val.stop],
        "train_rows": n,
        "w_distil": w_distil,
        "epochs": epochs,
        "history": hist,
    }
    torch.save({"meta": meta, "state_dict": m.state_dict()}, out)
    return meta


def load_probe(path: str) -> tuple[Probe, dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    meta = ck["meta"]
    m = Probe(meta["dim"], meta["hidden"], len(meta["quantiles"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), meta


# ── report ───────────────────────────────────────────────────────────────────


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    if len(p) == 0:
        return math.nan
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    ix = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        s = ix == b
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else math.nan


def predict(m: Probe, h: np.ndarray) -> dict[str, np.ndarray]:
    with torch.no_grad():
        o = m(torch.from_numpy(h.astype(np.float32)))
    return {
        "run": torch.sigmoid(o["run"]).numpy(),
        "fight": torch.sigmoid(o["fight"]).numpy(),
        "hp_q": o["hp_q"].numpy(),
    }


def report(probe: str, dirs: Sequence[str], seeds: range | None) -> dict[str, Any]:
    m, meta = load_probe(probe)
    z = load(dirs, seeds)
    pr = predict(m, z["h"])
    won = z["won"].astype(float)
    v_win = np.clip(z["v_run"] - z["shaping_to_go"], 0, 1)
    rows: dict[str, Any] = {"runs": len(np.unique(z["seed"])), "rows": len(won)}
    # `act` is 1-based (the env's `d.act`).
    for a in (None, 1, 2, 3):
        s = np.ones(len(won), bool) if a is None else z["act"] == a
        key = "all" if a is None else f"act{a}"
        c = s & z["is_combat"]
        fw = z["fight_won"].astype(float)
        hl = z["hp_loss"]
        cover = {
            f"q{q}": float(np.mean(hl[c] <= pr["hp_q"][c, i])) if c.any() else math.nan
            for i, q in enumerate(meta["quantiles"])
        }
        rows[key] = {
            "rows": int(s.sum()),
            "win_rate": float(won[s].mean()) if s.any() else math.nan,
            "run": {
                "ece": ece(pr["run"][s], won[s]),
                "brier": _brier(pr["run"][s], won[s]),
                "auc": _auc(pr["run"][s], won[s]),
                "mean_p": float(pr["run"][s].mean()) if s.any() else math.nan,
            },
            "v_win_raw": {
                "ece": ece(v_win[s], won[s]),
                "brier": _brier(v_win[s], won[s]),
                "auc": _auc(v_win[s], won[s]),
            },
            "fight": {
                "rows": int(c.sum()),
                "win_rate": float(fw[c].mean()) if c.any() else math.nan,
                "ece": ece(pr["fight"][c], fw[c]),
                "brier": _brier(pr["fight"][c], fw[c]),
                "auc": _auc(pr["fight"][c], fw[c]),
            },
            "fight_by_room": {
                room: {
                    "rows": int((c & (z["room"] == ri)).sum()),
                    "win_rate": float(fw[c & (z["room"] == ri)].mean())
                    if (c & (z["room"] == ri)).any()
                    else math.nan,
                    "mean_p": float(pr["fight"][c & (z["room"] == ri)].mean())
                    if (c & (z["room"] == ri)).any()
                    else math.nan,
                    "ece": ece(pr["fight"][c & (z["room"] == ri)], fw[c & (z["room"] == ri)]),
                    "auc": _auc(pr["fight"][c & (z["room"] == ri)], fw[c & (z["room"] == ri)]),
                }
                for ri, room in enumerate(ROOMS)
            },
            "hp": {
                "mae_median": float(np.mean(np.abs(hl[c] - pr["hp_q"][c, 2])))
                if c.any()
                else math.nan,
                "coverage": cover,
            },
        }
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppo.e2e.probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("ckpt")
    c.add_argument("out_dir")
    c.add_argument("start", type=int)
    c.add_argument("n", type=int)
    c.add_argument("--procs", type=int, default=8)
    f = sub.add_parser("fit")
    f.add_argument("dirs", nargs="+")
    f.add_argument("--train-seeds", required=True)
    f.add_argument("--val-seeds", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--hidden", type=int, default=256)
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--w-distil", type=float, default=4.0)
    r = sub.add_parser("report")
    r.add_argument("probe")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--seeds")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.ckpt, a.out_dir, a.start, a.n, a.procs)
    elif a.cmd == "fit":
        meta = fit(
            a.dirs,
            _seed_range(a.train_seeds),
            _seed_range(a.val_seeds),
            a.out,
            hidden=a.hidden,
            epochs=a.epochs,
            w_distil=a.w_distil,
        )
        print(json.dumps({k: v for k, v in meta.items() if k != "history"}))
    else:
        print(
            json.dumps(report(a.probe, a.dirs, _seed_range(a.seeds) if a.seeds else None), indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
