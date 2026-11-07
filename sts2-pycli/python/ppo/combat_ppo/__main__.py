"""CLI: `train`, `eval`, `throughput`.

    python -m ppo.combat_ppo train --init S_ctx.pt --out RUN --hbank hbank.jsonl \
        --boss800 boss800.jsonl --eval-fights dev_boss.jsonl
    python -m ppo.combat_ppo eval --ckpt RUN/ckpt/it00250.pt --fights dev_boss.jsonl \
        --greedy --out RUN/evals/it00250.json [--build-reps 4]
    python -m ppo.combat_ppo throughput --init S_ctx.pt --hbank ... --boss800 ... --iters 3

`eval` writes `combat_ctx.play run --out`'s format (`{summary, results}`), so
`python -m ppo.combat_ctx play-contrast` reads it. With `--build-reps 1` (the
default) the jobs are exactly `play.py`'s: build seed `_fight_seed(key)·100 +
77`. `--build-reps R` instead plays each fight from R build seeds
(`… + 77 + 1000·r`), which is more precise for monitoring. It is not the
sealed protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

from .pool import parse_weights


def _train_args(p: argparse.ArgumentParser) -> None:
    from .learner import Cfg

    d = Cfg(init="", out="")
    p.add_argument("--init", required=True)
    p.add_argument("--hbank", required=True)
    p.add_argument("--boss800", required=True)
    p.add_argument("--extra-holdouts", nargs="*", default=[])
    p.add_argument("--own-pool", nargs="*", default=[], help="harvested own-deck fight files")
    p.add_argument("--weights", default="boss=0.5,elite=0.3,monster=0.2")
    for name in (
        "workers",
        "n_envs",
        "priv_k",
        "max_steps",
        "seed",
        "budget_decisions",
        "iter_decisions",
        "epochs",
        "minibatch",
        "warmup_iters",
        "ckpt_every",
        "eval_every",
        "eval_reps",
        "eval_procs",
        "max_iters",
        "init_decisions",
        "micro_rows",
        "run_slots",
        "run_games_per_iter",
        "run_warmup",
        "run_ckpt_every",
        "run_ascension",
        "run_seed_base",
        "policy_frozen_until",
        "fullrun_eval_seeds",
        "fullrun_eval_seed_start",
        "fullrun_eval_procs",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(d, name))
    for name in (
        "clip",
        "target_kl",
        "gae_lambda",
        "lr",
        "lr_final_frac",
        "critic_lr",
        "ent0",
        "ent1",
        "grad_clip",
        "is_trunc",
        "own_share",
        "w_hp",
        "w_removed",
        "run_policy_lr",
        "run_value_lr",
        "run_lr_final",
        "run_entropy",
        "run_reward_scale",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(d, name))
    p.add_argument("--infer-device", default=d.infer_device)
    p.add_argument(
        "--update-devices", default=d.update_devices, help="one data-parallel update rank each"
    )
    for name in ("run_init", "run_device", "run_act1", "run_character", "run_reward"):
        p.add_argument("--" + name.replace("_", "-"), default=getattr(d, name))
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--eval-fights", default="", help="comma-separated fight files")


def _cfg(args: argparse.Namespace, out: str) -> Any:
    from .learner import Cfg

    kw = {k: getattr(args, k) for k in Cfg.__dataclass_fields__ if hasattr(args, k)}
    kw.update(init=args.init, out=out, weights=parse_weights(args.weights))
    kw["bf16"] = not args.no_bf16
    return Cfg(**kw)


def cmd_train(args: argparse.Namespace) -> int:
    from ppo.wheel_guard import prepare_training_wheel

    from .learner import Learner

    os.makedirs(args.out, exist_ok=True)
    wheel = prepare_training_wheel(
        pin_root=args.out, allow_plain=args.allow_plain_wheel, main_module="ppo.combat_ppo"
    )
    with open(os.path.join(args.out, "wheel.json"), "w", encoding="utf-8") as f:
        json.dump(wheel, f, indent=1, default=str)
    Learner(_cfg(args, args.out)).train()
    return 0


def cmd_throughput(args: argparse.Namespace) -> int:
    """A few real training iterations in a scratch dir; their `stats.jsonl` rows."""
    import shutil
    import tempfile

    from .learner import Learner

    out = tempfile.mkdtemp(prefix="cppo_tp_")
    cfg = _cfg(args, out)
    cfg.eval_fights = ""
    cfg.warmup_iters = 0
    cfg.max_iters = args.iters
    cfg.ckpt_every = 1 << 62
    t0 = time.time()
    try:
        Learner(cfg).train()
        with open(os.path.join(out, "stats.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(x) for x in f]
    finally:
        shutil.rmtree(out, ignore_errors=True)
    keys = (
        "it",
        "n_steps",
        "collect_s",
        "fwd_s",
        "send_s",
        "update_s",
        "upd_prep_s",
        "upd_fwd_s",
        "upd_epochs_s",
        "epochs_run",
        "dec_per_s",
        "lag_kl",
        "policy_swaps",
    )
    for r in rows:
        print(
            json.dumps(
                {k: round(r[k], 3) if isinstance(r[k], float) else r[k] for k in keys if k in r}
            )
        )
    steady = rows[1:] or rows
    summary = {
        "iters": len(rows),
        "wall_s": round(time.time() - t0, 1),
        "steady_dec_per_s": round(float(np.mean([r["dec_per_s"] for r in steady]))),
    }
    print(json.dumps(summary), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "summary": summary, "argv": sys.argv}, f, indent=1)
    return 0


def eval_jobs(rows: list[dict[str, Any]], build_reps: int) -> tuple[list[dict[str, Any]], int]:
    from ppo import rollout_starts
    from ppo.combat_ctx import play as P

    if build_reps == 1:
        return P.make_jobs(rows, 1)
    jobs, skipped, i = [], 0, 0
    for fi, row in enumerate(rows):
        base = P.build_seed(row)
        for rep in range(build_reps):
            s = base + 1000 * rep
            try:
                rollout_starts.build_fight(row, s)
            except ValueError:
                skipped += 1
                continue
            jobs.append(
                {
                    "i": i,
                    "fight": fi,
                    "rep": rep,
                    "row": row,
                    "build_seed": s,
                    "seed": 1_000_003 * i + 17,
                }
            )
            i += 1
    return jobs, skipped


def cmd_eval(args: argparse.Namespace) -> int:
    from ppo.combat_ctx import play as P

    with open(args.fights, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_fights:
        rows = rows[: args.max_fights]
    jobs, skipped = eval_jobs(rows, args.build_reps)
    spec = f"ctx:{args.ckpt}" if ":" not in args.ckpt or os.path.exists(args.ckpt) else args.ckpt
    t0 = time.time()
    res = P.play_many(spec, jobs, args.temperature, args.greedy, args.view, P.MAX_STEPS, args.procs)
    for r in res:
        r["build_rep"] = (
            next(j["rep"] for j in jobs if j["i"] == r["i"]) if args.build_reps > 1 else 0
        )
    won = np.asarray([r["won"] for r in res], bool)
    fight = np.asarray([r["fight"] for r in res])
    out = {
        "policy": spec,
        **P.policy_identity(spec),
        "fights_file": args.fights,
        "fold": None,
        "temperature": None if args.greedy else args.temperature,
        "greedy": args.greedy,
        "view": args.view,
        "build_reps": args.build_reps,
        "skipped": skipped,
        **P.fight_ci(won, fight),
        "decisions_per_playout": float(np.mean([r["decisions"] for r in res])),
        "capped_share": float(np.mean([r["capped"] for r in res])),
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"summary": out, "results": res}, f)
        pathlib.Path(tmp).replace(args.out)
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    """The deployed MCTS pilot on `eval`'s jobs (`pilot.py`)."""
    from ppo.combat_ctx import play as P

    from . import pilot

    with open(args.fights, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_fights:
        rows = rows[: args.max_fights]
    jobs, skipped = eval_jobs(rows, args.build_reps)
    t0 = time.time()
    res = pilot.play_many(jobs, args.sims, args.boss_sims, args.procs)
    won = np.asarray([r["won"] for r in res], bool)
    fight = np.asarray([r["fight"] for r in res])
    errors = sum(r["end"].startswith("error") for r in res)
    out = {
        "policy": f"mcts:{args.sims}/{args.boss_sims}",
        "fights_file": args.fights,
        "greedy": None,
        "view": "raw",
        "build_reps": args.build_reps,
        "skipped": skipped,
        **P.fight_ci(won, fight),
        "errors": errors,
        "capped_share": float(np.mean([r["capped"] for r in res])),
        "hp_after_mean_won": float(np.mean([r["hp_after"] for r in res if r["won"]] or [0])),
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"summary": out, "results": res}, f)
        pathlib.Path(tmp).replace(args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.combat_ppo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    _train_args(t)
    t.add_argument("--out", required=True)
    t.add_argument("--allow-plain-wheel", action="store_true")
    t.set_defaults(fn=cmd_train)
    tp = sub.add_parser("throughput")
    _train_args(tp)
    tp.add_argument("--iters", type=int, default=3)
    tp.add_argument("--json-out")
    tp.set_defaults(fn=cmd_throughput)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True, help="a checkpoint path, or a play.py policy spec")
    e.add_argument("--fights", required=True)
    e.add_argument("--max-fights", type=int, default=0)
    e.add_argument("--build-reps", type=int, default=1)
    e.add_argument("--reps", type=int, default=1, help="(kept for the learner; unused)")
    e.add_argument("--greedy", action="store_true")
    e.add_argument("--temperature", type=float, default=1.0)
    e.add_argument("--view", choices=("raw", "belief"), default="raw")
    e.add_argument("--procs", type=int, default=12)
    e.add_argument("--out")
    e.set_defaults(fn=cmd_eval)
    pl = sub.add_parser("pilot", help="the deployed MCTS pilot on eval's jobs")
    pl.add_argument("--fights", required=True)
    pl.add_argument("--max-fights", type=int, default=0)
    pl.add_argument("--build-reps", type=int, default=1)
    pl.add_argument("--sims", type=int, default=1000)
    pl.add_argument("--boss-sims", type=int, default=4000)
    pl.add_argument("--procs", type=int, default=12)
    pl.add_argument("--out")
    pl.set_defaults(fn=cmd_pilot)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
