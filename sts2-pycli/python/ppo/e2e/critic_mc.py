"""Monte Carlo check of the run critic: `python -m ppo.e2e.critic_mc run|report`.

The lane's `value_ev` scores the critic against λ-return targets, which mix in the critic's
own bootstraps. This measures it against real outcomes instead:

- **run**: replay the runs of a dev-eval arm (its `actions/*.e2el` logs), pick out-of-combat
  decisions (one per run by default, by a seeded hash), score `v_run` there (the learner's
  forward, PRIV tokens on), then play K reseeded rollouts from each state to the run's end
  with the training policy (raw weights, sampling at T=1). A rollout's return G is the env's
  total reward from the state (win 1, boss 0.05, floor 0.002), the critic's target.

      PYTHONPATH=<pin>:python python -m ppo.e2e.critic_mc run --pin PIN --char Regent \\
          --arm ARM_DIR --ckpt RAW.pt --k 8 --out mc_Regent.jsonl

- **report**: from one or more `run` outputs, per slice (all, act, character, decision type):

  - V_MC: the mean of a state's K returns.
  - noise: the mean over states of var(G | state) / K, the Monte Carlo error of V_MC.
  - ceiling EV = (var(V_MC) - noise) / var(G): the best explained variance any critic can
    reach against single returns on these states.
  - critic EV vs G: what the lane's `value_ev` would read with real returns as targets.
  - R² vs true V = 1 - (mean (v - V_MC)² - noise) / (var(V_MC) - noise): how much of the
    states' true value spread the critic explains, with the Monte Carlo noise removed.
  - bias: mean (v - V_MC).

      python -m ppo.e2e.critic_mc report mc_*.jsonl

Use the raw checkpoint (what the lane's actors sample with), not an EMA. Pick an arm whose
logs replay under the pin (`ReplayMismatch` otherwise). See
docs/investigations/2026-11-25-run-critic-monte-carlo.md for the readings so far.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Rollout reseeds: RESEED_BASE + hash % 10M, away from the DEV and trainer seed panels.
RESEED_BASE = 181_000_000
#: Rollouts a worker advances together (one batched forward a step).
GAMES = 32
#: A rollout that is still open after this many decisions counts as an error.
MAX_DECISIONS = 6000
_CLEAN = "STS2_CRITIC_MC_CLEAN"
_W: dict[str, Any] = {}


def _h(*xs: Any) -> int:
    return int.from_bytes(hashlib.sha256(repr(xs).encode()).digest()[:8], "little")


# ── run ───────────────────────────────────────────────────────────────────


def _init(ckpt: str, device: str) -> None:
    import torch

    from . import evaluate as E
    from .net import make_student

    E.require_campaign_build()
    torch.set_num_threads(1)
    _W["pol"] = make_student(ckpt, device=device)


def _value(pol: Any, d: Any) -> float:
    import torch

    from . import tokens
    from .net import to_batch

    tk = tokens.encode(d, priv=True)
    rb = tokens.row_batch([tk], run_uid=[d.run_uid], t=[d.row])
    with torch.no_grad():
        o = pol.net(to_batch(rb, pol.device))
    return float(o.v_run[0])


def _states(path: str, char: str, per_run: int) -> list[dict[str, Any]]:
    """Replay every run of log `path`; keep `per_run` OOC decisions of each (seeded pick)."""
    from . import actionlog as AL
    from .env import E2EDecision, E2EEnv, EnvConfig

    pol = _W["pol"]
    out = []
    for rec in AL.read_records(path):
        h = rec.header
        if h["segment"]["kind"] != "run":
            continue
        seed = int(h["seed"])
        env = E2EEnv(seed, cfg=EnvConfig(character=char))
        d = env.reset()
        ooc = []
        for k, ai in enumerate(rec.idx):
            if not isinstance(d, E2EDecision) or d.bootstrap:
                raise AL.ReplayMismatch(f"seed {seed}: ended early")
            if d.kind != "combat":
                ooc.append(k)
            d = env.step_action(int(ai))
        if env.terminal.fingerprint != h["fingerprint"]:
            raise AL.ReplayMismatch(f"seed {seed}: replay differs from the log")
        if not ooc:
            continue
        pick = {ooc[_h(seed, i) % len(ooc)] for i in range(per_run)}
        env = E2EEnv(seed, cfg=EnvConfig(character=char))
        d = env.reset()
        for k, ai in enumerate(rec.idx):
            if k in pick:
                assert isinstance(d, E2EDecision)  # the first pass checked every row
                out.append(
                    {
                        "seed": seed,
                        "k": k,
                        "snap": env.rm.clone(),
                        "v_run": _value(pol, d),
                        "act": int(d.act),
                        "floor": int(d.floor),
                        "dtype": int(d.dtype),
                        "orig_outcome": h["outcome"],
                        "rows_left": len(rec.idx) - k,
                    }
                )
            d = env.step_action(int(ai))
    return out


def _work(task: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    import torch

    from . import evaluate as E
    from .env import RUN, E2EEnv, EnvConfig
    from .net import concat_rows, to_batch

    path, char, per_run, k_roll = task
    pol = _W["pol"]
    try:
        sts = _states(path, char, per_run)
    except BaseException:
        return [{"file": path, "error": traceback.format_exc()[-2000:]}]
    jobs = [(i, j) for i in range(len(sts)) for j in range(k_roll)]
    rets: dict[tuple[int, int], dict[str, Any]] = {}
    live: list[dict[str, Any]] = []
    while jobs or live:
        while jobs and len(live) < GAMES:
            i, j = jobs.pop()
            s = sts[i]
            rs = RESEED_BASE + (_h(s["seed"], s["k"], j) % 10_000_000)
            try:
                rm = s["snap"].clone()
                rm.reseed_run(int(rs))
                seg = dataclasses.replace(RUN, uid=_h("mc", s["seed"], s["k"], j) % (1 << 62))
                env = E2EEnv(s["seed"], rm=rm, cfg=EnvConfig(character=char), segment=seg)
                gen = torch.Generator().manual_seed(rs)
                live.append({"i": i, "j": j, "env": env, "out": env.reset(), "gen": gen, "n": 0})
            except BaseException:
                rets[i, j] = {"error": traceback.format_exc()[-1500:]}
        ask = []
        for g in live:
            o = g["out"]
            if E.is_decision(o) and not getattr(o, "bootstrap", False) and g["n"] < MAX_DECISIONS:
                ask.append(g)
                continue
            env = g["env"]
            t = env.terminal
            if t is None:
                rets[g["i"], g["j"]] = {"error": f"open after {g['n']} decisions"}
                continue
            rets[g["i"], g["j"]] = {
                "G": float(env.total_reward + env.unassigned_reward),
                "won": getattr(t, "outcome", "") == "won",
                "outcome": getattr(t, "outcome", "open"),
                "n": g["n"],
            }
        live = ask
        if not live:
            continue
        try:
            with torch.no_grad():
                b = to_batch(concat_rows([pol.row(g["out"]) for g in live]), pol.device)
                lg = pol.net.policy(b, bf16=False)[: b.n_real].float().cpu()
            pr = torch.softmax(lg, -1)
            for k, g in enumerate(live):
                c = int(torch.multinomial(pr[k], 1, generator=g["gen"]))
                g["out"] = g["env"].step(c)
                g["n"] += 1
        except BaseException:
            err = traceback.format_exc()[-1500:]
            for g in live:
                rets[g["i"], g["j"]] = {"error": err}
            live = []
    rows = []
    for i, s in enumerate(sts):
        rr = [rets.get((i, j), {"error": "missing"}) for j in range(k_roll)]
        rows.append({k: v for k, v in s.items() if k != "snap"} | {"char": char, "rollouts": rr})
    return rows


def run(a: argparse.Namespace) -> int:
    from . import evaluate as E

    if os.environ.get(_CLEAN) != "1":
        # The env reads its fill events from the environment: the eval's own clean env.
        env = E._clean_env(
            Path(a.pin) if a.pin else None,
            [
                "STS2_E2E_CHARS=1",
                "STS2_E2E_OOC_POTION=1",
                "STS2_E2E_DECK_SELECT=1",
                f"STS2_E2E_CHARACTER={a.char}",
                *a.set_env,
            ],
        )
        env[_CLEAN] = "1"
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
    E.select_campaign_build()
    files = sorted(str(p) for p in Path(a.arm, "actions").glob("*.e2el"))
    if a.files:
        files = files[: a.files]
    if not files:
        print(f"critic_mc: no action logs under {a.arm}/actions", file=sys.stderr)
        return 2
    tasks = [(p, a.char, a.per_run, a.k) for p in files]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with (
        ctx.Pool(a.procs, initializer=_init, initargs=(a.ckpt, a.device)) as pool,
        open(a.out, "w", encoding="utf-8") as fh,
    ):
        fh.write(json.dumps({"meta": {k: v for k, v in vars(a).items() if k != "func"}}) + "\n")
        for n, rows in enumerate(pool.imap_unordered(_work, tasks), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{n}/{len(tasks)} files, {time.time() - t0:.0f}s", flush=True)
    return 0


# ── report ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Summary:
    n: int
    ceiling_ev: float
    critic_ev: float
    r2_true: float
    bias: float
    mean_v_mc: float


def read_states(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """The scored states of `run` outputs with at least 2 finished rollouts; `g` their returns."""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "rollouts" not in r:
                    continue
                g = [x["G"] for x in r["rollouts"] if "G" in x]
                if len(g) >= 2:
                    out.append({**r, "g": np.asarray(g, dtype=np.float64)})
    return out


def summarize(states: Sequence[dict[str, Any]]) -> Summary:
    """The critic against its states' Monte Carlo values (see the module docstring)."""
    v = np.array([s["v_run"] for s in states], dtype=np.float64)
    m = np.array([s["g"].mean() for s in states])
    noise = float(np.mean([s["g"].var(ddof=1) / len(s["g"]) for s in states]))
    allg = np.concatenate([s["g"] for s in states])
    vrep = np.concatenate([np.full(len(s["g"]), s["v_run"]) for s in states])
    var_g = float(allg.var())
    var_true = max(float(m.var()) - noise, 1e-12)
    mse_true = float(np.mean((v - m) ** 2)) - noise
    return Summary(
        n=len(states),
        ceiling_ev=var_true / var_g,
        critic_ev=1.0 - float(np.var(allg - vrep)) / var_g,
        r2_true=1.0 - mse_true / var_true,
        bias=float(np.mean(v - m)),
        mean_v_mc=float(m.mean()),
    )


def _dtype_names() -> dict[int, str]:
    from . import contracts as C

    return {int(v): k for k, v in dict(C.DT).items()}


def report(a: argparse.Namespace) -> int:
    states = read_states(a.paths)
    if not states:
        print("critic_mc: no scored states", file=sys.stderr)
        return 2
    names = _dtype_names()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in states:
        groups[f"act {s['act']}"].append(s)
        groups[f"char {s['char']}"].append(s)
        groups[f"dtype {names.get(s['dtype'], s['dtype'])}"].append(s)

    def line(label: str, ss: Sequence[dict[str, Any]]) -> None:
        if len(ss) < a.min_states:
            return
        x = summarize(ss)
        print(
            f"{label:22} n={x.n:5}  ceiling EV {x.ceiling_ev:5.2f}  critic EV vs G "
            f"{x.critic_ev:5.2f}  R2 vs true V {x.r2_true:5.2f}  bias {x.bias:+.3f}"
            f"  mean V_MC {x.mean_v_mc:.3f}"
        )

    line("ALL", states)
    for key in sorted(groups):
        line(key, groups[key])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ppo.e2e.critic_mc", description=__doc__)
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score states and play rollouts from them")
    r.add_argument("--char", required=True)
    r.add_argument("--arm", required=True, help="a dev-eval arm dir with actions/*.e2el")
    r.add_argument("--ckpt", required=True, help="raw net checkpoint (what the actors use)")
    r.add_argument("--pin", default="", help="wheel pin dir for the clean env")
    r.add_argument("--set-env", action="append", default=[], help="extra KEY=VALUE")
    r.add_argument("--per-run", type=int, default=1)
    r.add_argument("--k", type=int, default=8, help="rollouts per state")
    r.add_argument("--files", type=int, default=0, help="first N log files (0: all)")
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    q = sub.add_parser("report", help="summarize run outputs")
    q.add_argument("paths", nargs="+")
    q.add_argument("--min-states", type=int, default=30)
    q.set_defaults(func=report)
    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
