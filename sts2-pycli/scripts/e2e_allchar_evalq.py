"""Queue per-character DEV evals for an all-character e2e lane (GH #1197).

A port of the Regent lane's `e2e_regent_evalq.py` (branch exp/e2e-regent-v111) that
loops over the lane's characters. The learner pins a state at every `--pin-every`
decisions; each milestone pin gets, once, the requests its 50M bucket schedules:

- Ironclad, DEV-1000: every 50M up to the end of the mix ramp (`ramp_end`, 205M by
  default: the tripwire), every 100M after it;
- each other character, DEV-500 (DEV-1000's first 500 seeds): every 100M.

A request is queued once per pin (its sha256), character and panel, and its id carries
the sha256's first 8 hex: a lane reverted to an earlier pin (the ramp tripwire) gets its
redone milestones' evals too, under their own ids even at the same decision count.

Every request is the EMA 0.999 variant and carries its `character`: the learner's
built-in eval runner (`ppo.e2e.eval_runner`) then runs `ppo.e2e.evaluate` with
`STS2_E2E_CHARACTER=<character>` and `STS2_E2E_CHARS=1`, one process per character.
Results land in `<lane>/evals/done/`, where the wandb sidecar logs them as
`<character>/dev/ema0999/win_rate` (+ wins, runs, Wilson 95% interval); `--report`
prints them.

    python scripts/e2e_allchar_evalq.py --lane <lane save dir> [--report]

A lane training on human starts (`cfg.json` `human` with p > 0, `ppo.e2e.human`)
also gets, every 100M, a held-out human-start eval per character (DEV-500 with
`"env": "human"`: seed i plays held-out start i), logged as
`<character>/human/ema0999/win_rate`.

`--variant` picks the weights (default ema0999). EMA 0.999 averages over ~131M decisions:
on the Fool fine-tune, whose policy moved fast, it scored 9/500 at 200M where ema0996
scored 60/500 and raw 57/500.

`--only <character>` narrows the lane to one character: DEV-500 at every 50M pin and
nothing else (the Fool fine-tune, which trains the others only to keep them).

Run it from `sts2-pycli` with `PYTHONPATH=python`, on a timer. The characters and
`ramp_end` come from the lane's `cfg.json` mix (`ppo.e2e.mix.MixCfg`); the lane must
pin at every 50M (`--pin-every 50000000`), or the 50M milestones have no pin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ppo.e2e import ckpt
from ppo.e2e import ops_contract as C
from ppo.e2e.mix import BASE, MixCfg

STEP = 50_000_000  # the schedule's bucket: a pin belongs to the last multiple it passed
OTHERS_EVERY = 100_000_000  # every character; Ironclad too after the ramp
BASE_PANEL = "DEV-1000"
OTHER_PANEL = "DEV-500"
VARIANT = "ema0999"
REASON = "milestone"  # only the learner's `pin_every` pins (not K-gate or request pins)
SEEN = "ops/allchar_evalq.json"


def schedule(decisions: int, *, characters: Sequence[str], ramp_end: int) -> list[tuple[str, str]]:
    """(character, panel) pairs the milestone pin at `decisions` queues: its bucket is
    the last multiple of 50M at or below it (none below 50M)."""
    m = int(decisions) // STEP * STEP
    if m <= 0:
        return []
    out: list[tuple[str, str]] = []
    every_100 = m % OTHERS_EVERY == 0
    if BASE in characters and (m <= ramp_end or every_100):
        out.append((BASE, BASE_PANEL))
    if every_100:
        out += [(c, OTHER_PANEL) for c in characters if c != BASE]
    return out


def human_schedule(decisions: int, *, characters: Sequence[str], on: bool) -> list[str]:
    """The characters whose held-out human-start eval the milestone pin at `decisions`
    queues: every character at each 100M, when the lane trains on human starts."""
    m = int(decisions) // STEP * STEP
    return list(characters) if on and m > 0 and m % OTHERS_EVERY == 0 else []


def lane_mix(cfg: dict) -> MixCfg:
    """The lane's mix (its `cfg.json`), or the defaults for a lane without one."""
    return MixCfg.from_dict(cfg.get("mix") or {})


def queue(sd: ckpt.SaveDir, only: str | None = None, variant: str = VARIANT) -> int:
    """Write the requests of the milestone pins not queued yet; returns how many.
    `only`: that character's DEV-500 at every milestone pin, nothing else. `variant`:
    the weights evaluated (`raw`, `ema0996`, `ema0999`)."""
    seen_path = sd.p(SEEN)
    seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else {}
    cfg = C.read_json(sd.p("cfg.json")) or {}
    mix = lane_mix(cfg)
    every = int(cfg.get("pin_every") or 0)
    if every and every != STEP:
        print(f"warning: the lane pins every {every:,}; the schedule wants one pin per {STEP:,}")
    human_on = float((cfg.get("human") or {}).get("p") or 0.0) > 0.0
    n = 0
    for pin in ckpt.pinned_states(sd):
        if pin.reason != REASON:
            continue
        m = pin.decisions // STEP * STEP
        if only:
            jobs = [(only, OTHER_PANEL, "")] if m > 0 else []
        else:
            jobs = [
                (char, panel, "")
                for char, panel in schedule(
                    pin.decisions, characters=mix.characters, ramp_end=mix.ramp_end
                )
            ]
            jobs += [
                (char, OTHER_PANEL, "human")
                for char in human_schedule(pin.decisions, characters=mix.characters, on=human_on)
            ]
        for char, panel, env in jobs:
            tag = f":{env}" if env else ""
            key = f"{pin.sha256}:{char}:{panel}:{variant}{tag}"
            if key in seen:
                continue
            spec = C.EVAL_PANELS[panel]
            # The pin's sha in the id: a pin redone at the same decision count (a
            # tripwire revert) is another net, and the runner keys `evals/out/<id>/`
            # (and the student exported in it), the done file and the wandb point on it.
            rid = (
                f"{sd.root.name}_d{pin.decisions:013d}_{pin.sha256[:8]}"
                f"_{char}_{panel}_{variant}_milestone" + (f"_{env}" if env else "")
            )
            req = {
                "id": rid,
                "lane": sd.root.name,
                "state": str(pin.path(sd)),
                "sha256": pin.sha256,
                "decisions": pin.decisions,
                "milestone": m,
                "character": char,
                "panel": panel,
                "n": spec["n"],
                "seed_start": spec["seed_start"],
                "variant": variant,
                "nice": C.EVAL_NICE,
                "max_procs": C.EVAL_MAX_PROCS,
                "reason": "milestone",
                "requested_by": "allchar_evalq",
                **({"env": env} if env else {}),
                "time": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            C.atomic_write_text(
                sd.p(C.EVAL_QUEUE) / f"{rid}.json", json.dumps(req, indent=1) + "\n"
            )
            seen[key] = rid
            # Saved per request: a crash later in the loop re-queues nothing written.
            C.atomic_write_text(seen_path, json.dumps(seen, indent=1) + "\n")
            n += 1
            print(f"queued {rid}")
    return n


def report(sd: ckpt.SaveDir) -> None:
    from ppo.e2e.stats import wilson

    done = sd.p(C.EVAL_DONE)
    rows = []
    for f in sorted(done.glob("*.json")) if done.is_dir() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append((str(r.get("character") or "-"), int(r.get("decisions", 0)), r))
    for char, d, r in sorted(rows, key=lambda x: (x[0], x[1])):
        w, n = r.get("wins"), r.get("runs")
        if isinstance(w, int) and isinstance(n, int) and n:
            lo, hi = wilson(w, n)
            wr = f"{100.0 * w / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"
        else:
            wr = "  n/a"
        env = r.get("env") or "dev"
        print(
            f"{char:12s} {d / 1e6:6.0f}M  {env:5s} {r.get('panel')} {r.get('variant')} {wr}  ({w}/{n})"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Queue per-character DEV evals for an e2e lane.")
    ap.add_argument("--lane", type=Path, required=True, help="the lane's save dir")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", default="", help="one character: its DEV-500 at every 50M pin")
    ap.add_argument(
        "--variant",
        default=VARIANT,
        choices=("raw", "ema0996", "ema0999"),
        help="the weights evaluated; ema0999 lags ~130M decisions behind a fast-moving lane",
    )
    a = ap.parse_args(argv)
    sd = ckpt.SaveDir(a.lane)
    if a.report:
        report(sd)
    else:
        if a.only and a.only not in lane_mix(C.read_json(sd.p("cfg.json")) or {}).characters:
            ap.error(f"--only {a.only}: not one of the lane's characters")
        queue(sd, only=a.only or None, variant=a.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
