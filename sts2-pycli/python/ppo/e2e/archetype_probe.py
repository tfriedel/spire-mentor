"""Linear probes of the frozen net for archetype state: `python -m ppo.e2e.archetype_probe`.

    collect  play greedy runs (`forced_rollout` arms) and dump trunk features with labels
    fit      ridge probes, split by run, R² / accuracy against baselines

Probes (features: the trunk's normed output `h` at a token; the baseline is the same
token's input embedding, before any attention, and the label mean):

  B-doom    Necrobinder fights, each enemy with Doom: HP - Doom ("Doom still needed") and
            Doom >= HP ("Doom kills it at its turn"), at the enemy's ENEMY token. HP and Doom
            alone are the controls: Doom sits on a separate EPOWER token, so the input
            embedding of the ENEMY token cannot hold HP - Doom.
  B-poison  Silent fights, each enemy with Poison: turns until Poison alone kills it
            (ticks p, p-1, ...; capped at `CAP`, CAP + 1 = never) and "kills it eventually",
            at the ENEMY token.
  H-deck    out-of-combat decisions: the deck's archetype count (Necrobinder: Doom cards and
            Reaper Form; Silent: Poison sources and payoffs), from the state token (token 0).
  H-option  card rewards: at the option token of an offered payoff card, the deck's
            archetype count (does the option's representation carry the deck it would join).

    PYTHONPATH=<pin>:python python -m ppo.e2e.archetype_probe collect --pin PIN \\
        --char Necrobinder --ckpt EMA.pt --seed-start 160003000 --n 300 --arms none,prefix3 \\
        --out DIR --device cuda:3
    python -m ppo.e2e.archetype_probe fit DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import forced_rollout as FR

#: Poison turns-to-lethal cap; CAP + 1 means Poison alone never kills.
CAP = 10
#: Combat probe rows whose enemy max HP exceeds this are dropped (placeholder enemies).
MAX_HP = 10_000
#: Share of eligible combat decisions sampled (by a seeded hash), and of OOC decisions.
P_COMBAT = 0.35
P_OOC = 0.25
_CLEAN = "STS2_ARCHETYPE_PROBE_CLEAN"
_W: dict[str, Any] = {}

ARCHETYPE = {
    "Necrobinder": FR.DOOM_CARDS | {"REAPER_FORM"},
    "Silent": FR.POISON_SOURCES | FR.POISON_PAYOFFS,
}


def _u(*xs: Any) -> float:
    return int.from_bytes(hashlib.sha256(repr(xs).encode()).digest()[:8], "little") / 2.0**64


def poison_turns(hp: int, p: int, cap: int = CAP) -> int:
    """Enemy turns until Poison `p` (ticking p, p-1, ...) takes `hp`; cap + 1: never."""
    dealt = 0
    for k in range(1, cap + 1):
        dealt += max(p - (k - 1), 0)
        if dealt >= hp:
            return k
    return cap + 1


def _power(m: dict[str, Any], name: str) -> int:
    return sum(int(p.get("amount") or 0) for p in m.get("powers") or [] if p.get("name") == name)


# ── collect ───────────────────────────────────────────────────────────────


def _init(ckpt: str, device: str) -> None:
    import torch

    from . import evaluate as E
    from .net import make_student

    E.require_campaign_build()
    torch.set_num_threads(1)
    _W["pol"] = make_student(ckpt, device=device)


def _features(pol: Any, ds: list[Any]) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """Policy logits, trunk outputs `h`, input embeddings `x0` (`[B, T, D]`) and the batch."""
    import torch

    from .net import concat_rows, to_batch

    b = to_batch(concat_rows([pol.row(d) for d in ds]), pol.device)
    net = pol.net
    with torch.no_grad(), net._autocast(b.cat.device, False):
        x0 = net.embed(b.cat, b.cont, b.sec, b.pool_t, b.pool_ok, b.pool_cnt)
        _, h = net.trunk(b)
        lg = net._logits(h, b)
    return lg[: b.n_real].float().cpu(), h.float().cpu().numpy(), x0.float().cpu().numpy(), b


class _Sink:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def add(self, probe: str, h: np.ndarray, x0: np.ndarray, **labels: Any) -> None:
        self.rows.setdefault(probe, []).append(
            {"h": h.astype(np.float16), "x0": x0.astype(np.float16), **labels}
        )

    def save(self, path: Path) -> dict[str, int]:
        out = {}
        for probe, rows in self.rows.items():
            arr = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(path.with_name(f"{path.name}_{probe}.npz"), **arr)
            out[probe] = len(rows)
        return out


def _sample(g: FR.Game, h: np.ndarray, x0: np.ndarray, b: Any, k: int, sink: _Sink) -> None:
    from . import contracts as C

    d = g.out
    seed, arm = g.seed, g.arm.name
    tag = {"seed": seed, "arm_i": list(FR.ARMS).index(arm), "act": int(d.act)}
    deck = FR.deck_ids(g.env.rm)
    arch = sum(c in ARCHETYPE[g.char] for c in deck)
    if d.kind == "combat":
        enc = (d.d_bv or {}).get("encounter") or []
        sec = b.sec[k].cpu().numpy()
        cat = b.cat[k].cpu().numpy()
        slot_col = C._S["enemy_slot"]
        for t in np.flatnonzero(sec == C.SEC["ENEMY"]):
            slot = int(cat[t, slot_col])
            if not 0 <= slot < len(enc):
                continue
            m = enc[slot]
            if m.get("is_dead") or m.get("is_escaped"):
                continue
            hp, doom, poison = int(m["hp"]), _power(m, "DOOM"), _power(m, "POISON")
            if not (doom or poison) or _u(seed, arm, g.n, slot) >= P_COMBAT:
                continue
            probe = "doom" if g.char == "Necrobinder" else "poison"
            sink.add(
                probe,
                h[k, t],
                x0[k, t],
                hp=hp,
                max_hp=int(m["max_hp"]),
                block=int(m.get("block") or 0),
                doom=doom,
                poison=poison,
                turns=poison_turns(hp, poison),
                round=int(d.turn or 0),
                **tag,
            )
        return
    if d.dtype == C.DT["CARD_REWARD"]:
        offered = g.env.rm.card_reward_option_ids() or []
        cls_pos = b.cls_pos[k].cpu().numpy()
        seen = set()
        for ai, a in enumerate(d.actions):
            if a.get("action") != "pick_card_reward" or a.get("idx") is None:
                continue
            cid = offered[int(a["idx"])].removeprefix("CARD.")
            c = int(d.classes[ai])
            if cid not in g.target.payoff or c in seen:
                continue
            seen.add(c)
            t = int(cls_pos[c])
            sink.add("option", h[k, t], x0[k, t], arch=arch, card=cid, **tag)
    if _u(seed, arm, g.n, "ooc") < P_OOC:
        sink.add("deck", h[k, 0], x0[k, 0], arch=arch, deck_n=len(deck), **tag)


def _hand_ids(d: Any) -> list[str]:
    hand = ((d.d_bv or {}).get("deck") or {}).get("hand") or []
    out = []
    for c in hand:
        cid = c.get("id") if isinstance(c, dict) else c
        out.append(str(cid).removeprefix("CARD."))
    return out


def _work(task: tuple[str, list[tuple[int, str, str]]]) -> dict[str, Any]:
    import torch

    from . import evaluate as E

    path, jobs = task
    pol = _W["pol"]
    sink = _Sink()
    deaths = []
    todo = list(reversed(jobs))
    live: list[tuple[FR.Game, dict[str, Any]]] = []
    errors = 0
    while todo or live:
        while todo and len(live) < FR.GAMES:
            seed, arm, char = todo.pop()
            try:
                pol.begin_run(seed)
                live.append((FR.Game(seed, FR.ARMS[arm], char), {"fight": None, "hand": []}))
            except BaseException:
                errors += 1
        ask = []
        for g, tr in live:
            if E.is_decision(g.out) and not g.out.bootstrap and g.n < FR.MAX_DECISIONS:
                d = g.out
                if d.kind == "combat":
                    if d.fight != tr["fight"]:
                        tr["fight"], tr["hand"] = d.fight, []
                    tr["hand"].append(_hand_ids(d))
                ask.append((g, tr))
            elif g.env.terminal is not None and g.env.terminal.outcome != "won":
                forced = {c for e in g.forcer.events if e["forced"] for c in [e["taken"]] if c}
                deaths.append(
                    {
                        "seed": g.seed,
                        "arm": g.arm.name,
                        "act": int(g.env.rm.act_index),
                        "floor": int(g.env.rm.total_floor()),
                        "forced": sorted(forced),
                        "in_hand_last": bool(forced & set(tr["hand"][-1] if tr["hand"] else [])),
                        "in_hand_any": bool(forced & {c for hh in tr["hand"] for c in hh}),
                        "hand_last": tr["hand"][-1] if tr["hand"] else [],
                        "played_in_fatal": sorted(
                            forced & {x.get("id") for x in g.env.rm.card_play_log()}
                        ),
                    }
                )
        live = ask
        if not live:
            continue
        try:
            lg, h, x0, b = _features(pol, [g.out for g, _ in live])
            for k, (g, _) in enumerate(live):
                _sample(g, h, x0, b, k, sink)
                n_cls = len(g.out.members)
                g.out = g.env.step(g.choose(lg[k, :n_cls].tolist()))
                g.n += 1
        except BaseException:
            print(traceback.format_exc()[-2000:], file=sys.stderr)
            errors += len(live)
            live = []
    counts = sink.save(Path(path))
    return {"counts": counts, "deaths": deaths, "errors": errors}


def collect(a: argparse.Namespace) -> int:
    from . import evaluate as E

    if os.environ.get(_CLEAN) != "1":
        env = E._clean_env(
            Path(a.pin) if a.pin else None,
            [
                "STS2_E2E_CHARS=1",
                "STS2_E2E_OOC_POTION=1",
                "STS2_E2E_DECK_SELECT=1",
                f"STS2_E2E_CHARACTER={a.char}",
            ],
        )
        env[_CLEAN] = "1"
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
    E.select_campaign_build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else []
    seeds = seeds or list(range(a.seed_start, a.seed_start + a.n))
    jobs = [(s, arm, a.char) for s in seeds for arm in a.arms.split(",")]
    tasks = [
        (str(out / f"{a.char}_{i:04d}"), jobs[j : j + a.chunk])
        for i, j in enumerate(range(0, len(jobs), a.chunk))
    ]
    t0 = time.time()
    deaths, errors = [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, initializer=_init, initargs=(a.ckpt, a.device)) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, tasks), 1):
            deaths += r["deaths"]
            errors += r["errors"]
            print(f"{k}/{len(tasks)} {r['counts']} {time.time() - t0:.0f}s", flush=True)
    (out / f"{a.char}_deaths.json").write_text(json.dumps(deaths), encoding="utf-8")
    print(f"archetype_probe: {errors} errored runs", flush=True)
    return 0


# ── fit ───────────────────────────────────────────────────────────────────


def _load(d: Path, probe: str) -> dict[str, np.ndarray] | None:
    files = sorted(d.glob(f"*_{probe}.npz"))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _ridge(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, lam: float) -> np.ndarray:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    b = np.hstack([b, np.ones((len(b), 1))])
    reg = lam * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + reg, a.T @ ytr)
    return b @ w


def _split(seeds: np.ndarray) -> np.ndarray:
    """Test mask: a seeded fifth of the runs' seeds (all of a seed's rows on one side)."""
    return np.array([_u("split", int(s)) < 0.2 for s in seeds])


def _fit_one(x: np.ndarray, y: np.ndarray, te: np.ndarray, *, binary: bool) -> float:
    """Test R² (regression) or accuracy (binary, threshold 0.5), best ridge lambda on a
    held-out slice of the training runs."""
    tr = ~te
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    itr = np.flatnonzero(tr)
    val = np.zeros(len(y), bool)
    val[itr[: len(itr) // 5]] = True
    fit = tr & ~val

    def score(pred: np.ndarray, yy: np.ndarray) -> float:
        if binary:
            return float(((pred > 0.5) == (yy > 0.5)).mean())
        return float(1.0 - ((yy - pred) ** 2).mean() / yy.var())

    best = max(
        (10.0**k for k in range(-1, 5)),
        key=lambda lam: score(_ridge(x[fit], y[fit], x[val], lam), y[val]),
    )
    return score(_ridge(x[tr], y[tr], x[te], best), y[te])


def fit(a: argparse.Namespace) -> int:
    d = Path(a.dir)
    res: dict[str, Any] = {}
    specs = {
        "doom": [
            ("hp - doom", lambda z: z["hp"] - z["doom"], False),
            ("doom >= hp", lambda z: (z["doom"] >= z["hp"]).astype(float), True),
            ("hp", lambda z: z["hp"], False),
            ("doom", lambda z: z["doom"], False),
        ],
        "poison": [
            ("turns to lethal", lambda z: z["turns"], False),
            ("poison kills", lambda z: (z["turns"] <= CAP).astype(float), True),
            ("hp", lambda z: z["hp"], False),
            ("poison", lambda z: z["poison"], False),
        ],
        "deck": [
            ("archetype count", lambda z: z["arch"], False),
            ("archetype >= 2", lambda z: (z["arch"] >= 2).astype(float), True),
        ],
        "option": [
            ("archetype count", lambda z: z["arch"], False),
            ("archetype >= 2", lambda z: (z["arch"] >= 2).astype(float), True),
        ],
    }
    for probe, targets in specs.items():
        z = _load(d, probe)
        if z is None:
            continue
        if "max_hp" in z:
            # A few enemies carry a placeholder max HP (999999999): not a fight to read.
            keep = z["max_hp"] <= MAX_HP
            print(f"\n{probe}: dropped {int((~keep).sum())} rows with max HP > {MAX_HP}")
            z = {k: v[keep] for k, v in z.items()}
        te = _split(z["seed"])
        n_runs = len(np.unique(z["seed"]))
        print(f"\n{probe}: n={len(te)} rows ({int(te.sum())} test) from {n_runs} seeds")
        for name, fn, binary in targets:
            y = fn(z)
            base = (
                max(float((y[te] > 0.5).mean()), 1 - float((y[te] > 0.5).mean())) if binary else 0.0
            )
            r_h = _fit_one(z["h"], y, te, binary=binary)
            r_x = _fit_one(z["x0"], y, te, binary=binary)
            m = "acc" if binary else "R2"
            print(
                f"  {name:16} {m} trunk {r_h:.3f}  input-emb {r_x:.3f}  "
                f"{'majority' if binary else 'mean'} {base:.3f}  (label mean {y.mean():.2f})"
            )
            res[f"{probe}/{name}"] = {
                "metric": m,
                "trunk": r_h,
                "input_emb": r_x,
                "baseline": base,
                "n": len(y),
                "n_test": int(te.sum()),
                "seeds": n_runs,
            }
    (d / "probe_fit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ppo.e2e.archetype_probe", description=__doc__)
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--char", required=True, choices=sorted(FR.TARGETS))
    c.add_argument("--ckpt", required=True)
    c.add_argument("--pin", default="")
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--n", type=int, default=0)
    c.add_argument("--seeds", default="", help="comma-separated seeds instead of a range")
    c.add_argument("--arms", default="none,prefix3")
    c.add_argument("--procs", type=int, default=8)
    c.add_argument("--chunk", type=int, default=40)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--out", required=True)
    c.set_defaults(func=collect)
    f = sub.add_parser("fit")
    f.add_argument("dir")
    f.set_defaults(func=fit)
    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
