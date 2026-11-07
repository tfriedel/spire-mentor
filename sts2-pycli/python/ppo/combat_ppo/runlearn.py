"""The run-net PPO learner of the hybrid trainer (its own process).

It takes finished run rows (`runslot.RunSlot`, `workers.play_one`'s format)
from the combat learner and, every `games_per_iter` runs, runs one
`ppo.trainer` update on them: `gae_from_row` with the champion's shaping,
`ppo_update`, the champion's optimizer state and advantage moments carried
over. The new weights go to `<out>/run/acting/v<ver>.pt`, and the version
goes back to the combat learner, which forwards it to the workers with the
next actions.

`warmup` iterations train the value head only: the champion's critic learnt
run values under search combat, and the policy-played fights change them.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import traceback
from typing import Any

import numpy as np


def acting_path(out: str, ver: int) -> str:
    return os.path.join(out, "run", "acting", f"v{ver:05d}.pt")


def _save(path: str, payload: dict[str, Any]) -> None:
    import torch

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    pathlib.Path(tmp).replace(path)


def init_acting(out: str, init: str) -> None:
    """Version 0: the warm-start net, as the workers' first acting checkpoint."""
    import torch

    os.makedirs(os.path.join(out, "run", "acting"), exist_ok=True)
    os.makedirs(os.path.join(out, "run", "ckpt"), exist_ok=True)
    ck = torch.load(init, map_location="cpu", weights_only=True)
    _save(acting_path(out, 0), {"state_dict": ck["state_dict"], "cfg": ck["cfg"]})


def run_cfg(init_cfg: dict[str, Any], over: dict[str, Any]) -> Any:
    from ppo.trainer import PPOConfig

    fields = PPOConfig.__dataclass_fields__
    kw = {k: v for k, v in init_cfg.items() if k in fields}
    kw.update({k: v for k, v in over.items() if k in fields})
    return PPOConfig(**kw)


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    floors = [int((r.get("final") or {}).get("total_floor") or 0) for r in ok]
    acts = [int((r.get("final") or {}).get("act_index") or 0) for r in ok]
    return {
        "runs": len(rows),
        "errors": len(rows) - len(ok),
        "win": float(np.mean([r["won"] for r in ok])) if ok else float("nan"),
        "floor_mean": float(np.mean(floors)) if floors else float("nan"),
        "act_mean": float(np.mean(acts)) if acts else float("nan"),
        "reached_second_boss": float(np.mean([bool(r.get("reached_second_boss")) for r in ok]))
        if ok
        else float("nan"),
        "fights_mean": float(np.mean([r.get("fights", 0) for r in ok])) if ok else float("nan"),
        "decisions_mean": float(np.mean([r.get("n_decisions", 0) for r in ok]))
        if ok
        else float("nan"),
        "wall_s_mean": float(np.mean([r.get("wall_s", 0) for r in ok])) if ok else float("nan"),
        "stale_ver_mean": float(np.mean([r.get("run_ver", 0) for r in ok])) if ok else float("nan"),
    }


def run_learner_main(rcfg: dict[str, Any], inq: Any, outq: Any) -> None:
    try:
        _main(rcfg, inq, outq)
    except BaseException:
        outq.put(("run_error", -1, traceback.format_exc()))
        raise


def _main(rcfg: dict[str, Any], inq: Any, outq: Any) -> None:
    import torch

    from ppo import schema
    from ppo.model import RunPolicyNet, load_compat
    from ppo.rewards import ShapingConfig
    from ppo.trainer import (
        RunningMoments,
        acting_payload,
        gae_from_row,
        make_optimizer,
        ppo_update,
    )

    out = rcfg["out"]
    dev = torch.device(rcfg["device"])
    torch.manual_seed(0)
    torch.set_num_threads(4)
    ck = torch.load(rcfg["init"], map_location="cpu", weights_only=True)
    cfg = run_cfg(ck["cfg"], rcfg["over"])
    if schema.parse_features(cfg.features) != schema.ACTIVE_FEATURES:
        raise SystemExit(
            f"run net features {cfg.features!r} but STS2_PPO_FEATURES gives "
            f"{sorted(schema.ACTIVE_FEATURES)}: export STS2_PPO_FEATURES='{cfg.features}'"
        )
    shaping = ShapingConfig(
        hp_coef=cfg.shaping_hp_coef,
        upg_coef=cfg.shaping_upg_coef,
        relic_coef=cfg.shaping_relic_coef,
        offset=cfg.shaping_offset,
    )
    net = RunPolicyNet(dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads).to(dev)
    opt = make_optimizer(net, cfg)
    adv_norm = RunningMoments(cfg.adv_decay)
    ver = 0
    state_p = os.path.join(out, "run", "state.pt")
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location="cpu", weights_only=True)
        net.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["opt"])
        adv_norm.mean, adv_norm.sq, adv_norm.inited = st["adv_mean"], st["adv_sq"], True
        ver = int(st["ver"])
    else:
        # `load_compat`: a champion predates embeddings added since (they start
        # at their init, as in `ppo.trainer --init-ckpt`).
        sd0 = ck["state_dict"]
        missing = [
            k for k, v in net.state_dict().items() if k not in sd0 or sd0[k].shape != v.shape
        ]
        load_compat(net, sd0)
        if "opt" in ck and not missing:
            opt.load_state_dict(ck["opt"])
        else:
            print(f"[run_ppo] fresh optimizer (new params: {missing})", flush=True)
        adv_norm.mean, adv_norm.sq, adv_norm.inited = ck["adv_mean"], ck["adv_sq"], True
    for g in opt.param_groups:  # the configured lrs, not the champion's saved ones
        g["base_lr"] = cfg.policy_lr if g is opt.param_groups[0] else cfg.value_lr
    update_rng = np.random.default_rng(ver)
    outq.put(("run_ready", ver, None))

    rows: list[dict[str, Any]] = []
    stats_p = os.path.join(out, "run", "stats.jsonl")
    while True:
        msg = inq.get()
        if msg[0] == "stop":
            return
        if msg[0] == "rows":
            rows.extend(msg[1])
        if len(rows) < cfg.games_per_iter:
            continue
        batch, rows = rows[: cfg.games_per_iter], rows[cfg.games_per_iter :]
        t0 = time.time()
        it = ver  # one update per version
        frac = min(1.0, float(msg[2]) if len(msg) > 2 else 0.0)  # the combat budget fraction
        warm = it < int(rcfg["warmup"])
        lr_frac = 1.0 - (1.0 - cfg.lr_anneal_final) * frac
        for i, g in enumerate(opt.param_groups):
            g["lr"] = 0.0 if (warm and i == 0) else g["base_lr"] * lr_frac
        trans = [t for r in batch if not r.get("error") for t in gae_from_row(r, cfg, shaping)]
        net.train()
        ust = ppo_update(net, opt, trans, cfg, adv_norm, cfg.entropy_coef_final, update_rng, dev)
        net.eval()
        ver += 1
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        _save(acting_path(out, ver), acting_payload(sd, cfg))
        old = acting_path(out, ver - 8)
        if os.path.exists(old) and (ver - 8) % int(rcfg["ckpt_every"]) != 0:
            os.remove(old)
        if ver % int(rcfg["ckpt_every"]) == 0:
            _save(
                os.path.join(out, "run", "ckpt", f"v{ver:05d}.pt"),
                {
                    **acting_payload(sd, cfg),
                    "opt": opt.state_dict(),
                    "adv_mean": adv_norm.mean,
                    "adv_sq": adv_norm.sq,
                },
            )
        _save(
            state_p,
            {
                "state_dict": sd,
                "opt": opt.state_dict(),
                "adv_mean": adv_norm.mean,
                "adv_sq": adv_norm.sq,
                "ver": ver,
            },
        )
        row = {
            "ver": ver,
            "time": time.time(),
            "warmup": warm,
            "lr_frac": lr_frac,
            "n_trans": len(trans),
            "update_s": round(time.time() - t0, 2),
            "queued_runs": len(rows),
            **row_stats(batch),
            **{f"u_{k}": v for k, v in ust.items()},
        }
        with open(stats_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        outq.put(("run_ver", ver, row))
