"""Forced archetype picks on the current net: `python -m ppo.e2e.forced_rollout run|report`.

The all-character net takes the sources of some archetypes but almost never their payoffs
(Necrobinder's Doom cards; Silent's Poison payoffs). This driver asks whether taking them
would win more runs with the net's own play: each seed is played under several arms, the
greedy net (as in a DEV eval) making every decision except at the card rewards an arm
overrides.

A card reward is *eligible* when it offers a target card (`Target.payoff`) and the deck
already holds `Target.min_sources` of the archetype's sources (Silent: 2 Poison sources;
Necrobinder: no requirement). At an eligible reward an arm may *force* it:

  none      never (the control: the net's own picks);
  single    the first eligible reward;
  prefix3   the first three eligible rewards;
  late      the first eligible reward in act 2 or later;
  deviate   the first eligible reward, but taking the net's 2nd-ranked option that is not a
            target card (skip included): any override, not the archetype;
  skiponly  the first eligible reward where the net would skip or its top class has
            probability below `UNSURE_P`.

Forcing takes the target card the net rates highest. A forced screen whose pick equals the
net's own still uses the arm's slot (it is not an override). The env replays a seed exactly
under the same choices, so arms are compared per seed (McNemar, paired bootstrap).

    PYTHONPATH=<pin>:python python -m ppo.e2e.forced_rollout run --pin PIN \\
        --char Necrobinder --ckpt EMA.pt --seed-start 160000500 --n 2000 \\
        --arms none,single,prefix3,late,deviate,skiponly --out OUT.jsonl --device cuda:3
    python -m ppo.e2e.forced_rollout report OUT.jsonl [--json report.json]

Each row also carries the run's end (act, floor), its final deck, the forcing events, the
plays of target cards (`rm.card_play_log`, read at each fight's end) and from the fight's
event log (`rm.combat_log`) the Doom and Poison applied to enemies, Doom kills (an enemy
with Doom killed by its own Doom tick: lethal self-damage) and Poison kills (lethal
damage with no dealer on a poisoned enemy).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: The top-class probability below which `skiponly` calls the net unsure.
UNSURE_P = 0.6
#: Runs a worker advances together (one batched forward per step).
GAMES = 32
#: A run still open after this many decisions is booked as an error.
MAX_DECISIONS = 6000
_CLEAN = "STS2_FORCED_ROLLOUT_CLEAN"
_W: dict[str, Any] = {}

# ── what is forced ────────────────────────────────────────────────────────

DOOM_CARDS = frozenset(
    {
        "BLIGHT_STRIKE",
        "SCOURGE",
        "NEGATIVE_PULSE",
        "NO_ESCAPE",
        "COUNTDOWN_CARD",
        "END_OF_DAYS",
        "TIMES_UP",
        "DEATHBRINGER",
        "OBLIVION",
        "SHROUD",
        "DEATHS_DOOR",
    }
)  # Reaper Form is left out: the net already takes it (59%).
POISON_PAYOFFS = frozenset(
    {"ACCELERANT", "BUBBLE_BUBBLE", "MIRAGE", "CORROSIVE_WAVE", "OUTBREAK_CARD"}
)
POISON_SOURCES = frozenset(
    {
        "NOXIOUS_FUMES_CARD",
        "ENVENOM_CARD",
        "SNAKEBITE",
        "DEADLY_POISON",
        "POISONED_STAB",
        "BOUNCING_FLASK",
        "HAZE",
    }
)


@dataclasses.dataclass(frozen=True)
class Target:
    """A character's forced archetype: its payoff cards, and the sources a deck must hold."""

    payoff: frozenset[str]
    sources: frozenset[str] = frozenset()
    min_sources: int = 0


TARGETS: Mapping[str, Target] = {
    "Necrobinder": Target(DOOM_CARDS),
    "Silent": Target(POISON_PAYOFFS, POISON_SOURCES, 2),
}


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    max_forces: int = 0  # eligible rewards the arm may force; 0: the control
    min_act: int = 1  # an eligible reward before this act is neither forced nor counted
    unsure_only: bool = False  # force only where the net skips or is unsure
    deviate: bool = False  # take the net's 2nd non-target option instead


ARMS: Mapping[str, Arm] = {
    a.name: a
    for a in (
        Arm("none"),
        Arm("single", 1),
        Arm("prefix3", 3),
        Arm("late", 1, min_act=2),
        Arm("deviate", 1, deviate=True),
        Arm("skiponly", 1, unsure_only=True),
    )
}


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


class Forcer:
    """One run's card-reward overrides under `arm` (state: the slots used so far).

    `choose` gets a CARD_REWARD decision's offered ids, actions, their merged classes and
    the net's class logits; it returns the class to take and, at an eligible reward, an
    event dict for the run's row."""

    def __init__(self, arm: Arm, target: Target) -> None:
        self.arm = arm
        self.target = target
        self.used = 0
        self.events: list[dict[str, Any]] = []

    def sources_in(self, deck: Iterable[str]) -> int:
        return sum(c in self.target.sources for c in deck)

    def choose(
        self,
        offered: Sequence[str],
        actions: Sequence[Mapping[str, Any]],
        classes: Sequence[int],
        logits: Sequence[float],
        *,
        act: int,
        floor: int,
        deck: Sequence[str],
    ) -> tuple[int, dict[str, Any] | None]:
        greedy = max(range(len(logits)), key=logits.__getitem__)
        picks: dict[int, str | None] = {}  # class -> its card id (None: skip)
        for ai, a in enumerate(actions):
            if a.get("action") != "pick_card_reward":
                continue
            idx = a.get("idx")
            cid = None if idx is None else offered[int(idx)].removeprefix("CARD.")
            picks.setdefault(int(classes[ai]), cid)
        targets = [c for c, cid in picks.items() if cid in self.target.payoff]
        if not targets or self.sources_in(deck) < self.target.min_sources:
            return greedy, None
        p = _softmax(logits)
        net_pick = picks.get(greedy, "?other")
        ev: dict[str, Any] = {
            "act": act,
            "floor": floor,
            "offered": [o.removeprefix("CARD.") for o in offered],
            "net": net_pick,
            "net_p": round(p[greedy], 4),
            "forced": False,
        }
        arm = self.arm
        unsure = net_pick is None or p[greedy] < UNSURE_P
        if self.used >= arm.max_forces or act < arm.min_act or (arm.unsure_only and not unsure):
            ev["taken"] = net_pick
            return greedy, ev
        self.used += 1
        if arm.deviate:
            rest = sorted((c for c in picks if c not in targets), key=lambda c: -logits[c])
            choice = rest[1] if len(rest) > 1 else rest[0] if rest else greedy
        else:
            choice = max(targets, key=logits.__getitem__)
        ev.update(forced=True, taken=picks[choice], override=choice != greedy)
        return choice, ev


# ── combat logs ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class FightStats:
    """Totals over a run's fights, from `card_play_log` and `combat_log`."""

    plays: Counter = dataclasses.field(default_factory=Counter)  # target card id -> plays
    doom_applied: int = 0  # Doom put on enemies
    doom_kills: int = 0
    poison_applied: int = 0
    poison_damage: int = 0
    poison_kills: int = 0
    fights: int = 0
    log_dropped: int = 0

    def add(
        self,
        plays: Sequence[Mapping[str, Any]],
        log: Sequence[Mapping[str, Any]],
        watch: frozenset[str],
        dropped: int = 0,
    ) -> None:
        self.fights += 1
        self.log_dropped += int(dropped)
        for p in plays:
            if p.get("id") in watch:
                self.plays[p["id"]] += 1
        doom: dict[Any, int] = defaultdict(int)
        poison: dict[Any, int] = defaultdict(int)
        for r in log:
            kind = r.get("kind")
            rc = r.get("receiver") or {}
            if rc.get("kind") != "monster":
                continue
            uid = rc.get("uid")
            if kind == "power" and int(r.get("amount") or 0) > 0:
                if r.get("power") == "DOOM":
                    doom[uid] += int(r["amount"])
                    self.doom_applied += int(r["amount"])
                elif r.get("power") == "POISON":
                    poison[uid] += int(r["amount"])
                    self.poison_applied += int(r["amount"])
            elif kind == "damage":
                dealer = r.get("dealer")
                before, after = int(r.get("hp_before") or 0), int(r.get("hp_after") or 0)
                lethal = before > 0 and after == 0
                if dealer is None and poison[uid] > 0:
                    self.poison_damage += max(before - after, 0)
                    self.poison_kills += lethal
                elif (
                    lethal
                    and doom[uid] > 0
                    and dealer.get("kind") == "monster"
                    and dealer.get("uid") == uid
                ):
                    self.doom_kills += 1

    def row(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["plays"] = dict(self.plays)
        return d


# ── run ───────────────────────────────────────────────────────────────────


def _init(ckpt: str, device: str) -> None:
    import torch

    from . import evaluate as E
    from .net import make_student

    E.require_campaign_build()
    torch.set_num_threads(1)
    _W["pol"] = make_student(ckpt, device=device)


def deck_ids(rm: Any) -> list[str]:
    from .decomp import snake

    return [snake(name) for name, _up in rm.master_deck_detailed()]


class Game:
    """One (seed, arm) run: the env, its forcer and its fight statistics."""

    def __init__(self, seed: int, arm: Arm, char: str) -> None:
        from .env import E2EEnv, EnvConfig

        self.seed, self.arm, self.char = seed, arm, char
        self.target = TARGETS[char]
        self.forcer = Forcer(arm, self.target)
        self.stats = FightStats()
        self.env = E2EEnv(seed, cfg=EnvConfig(character=char))
        self.env.rm.set_record_card_plays(True)
        self.env.rm.set_record_combat_log(True)
        self.out = self.env.reset()
        self.n = 0
        self.fight: Any = None  # the fight of the last combat decision; None: out of combat
        self.lost_fights = 0

    def live(self) -> bool:
        from . import evaluate as E

        o = self.out
        return E.is_decision(o) and not getattr(o, "bootstrap", False) and self.n < MAX_DECISIONS

    def harvest(self) -> None:
        """At a decision (or the end) after a combat decision: book the fight that ended."""
        o = self.out
        live = self.live()
        if self.fight is None:
            if live and o.kind == "combat":
                self.fight = o.fight
            return
        if live and o.kind == "combat" and o.fight == self.fight:
            return
        if live and o.kind == "combat":
            self.lost_fights += 1  # fight to fight: its logs already hold the new one
        else:
            rm = self.env.rm
            watch = self.target.payoff | self.target.sources
            self.stats.add(rm.card_play_log(), rm.combat_log(), watch, rm.combat_log_dropped())
        self.fight = o.fight if live and o.kind == "combat" else None

    def choose(self, logits: Sequence[float]) -> int:
        from . import contracts as C

        d = self.out
        greedy = max(range(len(logits)), key=logits.__getitem__)
        if d.dtype != C.DT["CARD_REWARD"]:
            return greedy
        rm = self.env.rm
        choice, ev = self.forcer.choose(
            rm.card_reward_option_ids() or [],
            d.actions,
            d.classes,
            logits,
            act=int(d.act),
            floor=int(d.floor),
            deck=deck_ids(rm),
        )
        if ev is not None:
            self.forcer.events.append(ev)
        return choice

    def result(self) -> dict[str, Any]:
        env, rm = self.env, self.env.rm
        t = env.terminal
        if t is None:
            raise RuntimeError(f"open after {self.n} decisions")
        won = getattr(t, "outcome", "") == "won"
        deck = deck_ids(rm)
        forced = [e for e in self.forcer.events if e["forced"]]
        return {
            "seed": self.seed,
            "arm": self.arm.name,
            "char": self.char,
            "won": won,
            "outcome": str(getattr(t, "outcome", "")),
            "fingerprint": getattr(t, "fingerprint", None),
            "act": int(rm.act_index),
            "floor": int(rm.total_floor()),
            "n_decisions": self.n,
            "log_sha": hashlib.sha256(repr(env.log_idx).encode()).hexdigest()[:16],
            "deck": deck,
            "deck_targets": sum(c in self.target.payoff for c in deck),
            "deck_sources": sum(c in self.target.sources for c in deck),
            "eligible": len(self.forcer.events),
            "forced": len(forced),
            "overrides": sum(bool(e.get("override")) for e in forced),
            "forced_cards": [e["taken"] for e in forced],
            "events": self.forcer.events,
            "lost_fights": self.lost_fights,
            **self.stats.row(),
        }


def _work(jobs: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    import torch

    from .net import concat_rows, to_batch

    pol = _W["pol"]
    todo = list(reversed(jobs))
    rows: list[dict[str, Any]] = []
    live: list[Game] = []

    def fail(seed: int, arm: str, char: str) -> None:
        rows.append(
            {"seed": seed, "arm": arm, "char": char, "error": traceback.format_exc()[-2000:]}
        )

    while todo or live:
        while todo and len(live) < GAMES:
            seed, arm, char = todo.pop()
            try:
                pol.begin_run(seed)
                live.append(Game(seed, ARMS[arm], char))
            except BaseException:
                fail(seed, arm, char)
        ask = []
        for g in live:
            try:
                g.harvest()
                if g.live():
                    ask.append(g)
                else:
                    rows.append(g.result())
            except BaseException:
                fail(g.seed, g.arm.name, g.char)
        live = ask
        if not live:
            continue
        try:
            with torch.no_grad():
                b = to_batch(concat_rows([pol.row(g.out) for g in live]), pol.device)
                lg = pol.net.policy(b, bf16=pol.bf16)[: b.n_real].float().cpu()
        except BaseException:
            for g in live:
                fail(g.seed, g.arm.name, g.char)
            live = []
            continue
        nxt = []
        for k, g in enumerate(live):
            try:
                n_cls = len(g.out.members)
                c = g.choose(lg[k, :n_cls].tolist())
                g.out = g.env.step(c)
                g.n += 1
                nxt.append(g)
            except BaseException:
                fail(g.seed, g.arm.name, g.char)
        live = nxt
    return rows


def _done(path: str) -> set[tuple[int, str]]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "seed" in r and "error" not in r:
                out.add((int(r["seed"]), r["arm"]))
    return out


def run(a: argparse.Namespace) -> int:
    from . import evaluate as E

    if os.environ.get(_CLEAN) != "1":
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
    if a.char not in TARGETS:
        print(f"forced_rollout: no target for {a.char}", file=sys.stderr)
        return 2
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in ARMS:
            print(f"forced_rollout: unknown arm {arm!r}", file=sys.stderr)
            return 2
    done = _done(a.out)
    jobs = [
        (s, arm, a.char)
        for s in range(a.seed_start, a.seed_start + a.n)
        for arm in arms
        if (s, arm) not in done
    ]
    if a.shuffle:
        random.Random(a.shuffle).shuffle(jobs)
    chunks = [jobs[i : i + a.chunk] for i in range(0, len(jobs), a.chunk)]
    print(
        f"forced_rollout: {len(jobs)} runs ({len(done)} done) in {len(chunks)} chunks", flush=True
    )
    t0 = time.time()
    n_rows = 0
    ctx = mp.get_context("spawn")
    with (
        ctx.Pool(a.procs, initializer=_init, initargs=(a.ckpt, a.device)) as pool,
        open(a.out, "a", encoding="utf-8") as fh,
    ):
        if not done:
            meta = {k: v for k, v in vars(a).items() if k != "func"}
            fh.write(json.dumps({"meta": meta}) + "\n")
        for k, rows in enumerate(pool.imap_unordered(_work, chunks), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_rows += len(rows)
            el = time.time() - t0
            print(f"{k}/{len(chunks)} chunks, {n_rows} runs, {el:.0f}s", flush=True)
            if a.max_seconds and el > a.max_seconds:
                print("forced_rollout: time budget reached; rerun to resume", flush=True)
                pool.terminate()
                break
    return 0


# ── report ────────────────────────────────────────────────────────────────


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired(
    base: Sequence[bool], arm: Sequence[bool], *, boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Paired win-rate difference arm - base: point, Wald and bootstrap 95% CIs, McNemar."""
    n = len(base)
    if n == 0:
        return {"n": 0}
    import numpy as np

    diff = np.asarray(arm, dtype=np.int64) - np.asarray(base, dtype=np.int64)
    only_base = int((diff == -1).sum())
    only_arm = int((diff == 1).sum())
    d = (only_arm - only_base) / n
    se = math.sqrt(max(only_arm + only_base - n * d * d, 0.0)) / n
    rng = np.random.default_rng(seed)
    bs = np.sort(
        np.concatenate(
            [
                diff[rng.integers(0, n, (min(1000, boot - i), n))].mean(1)
                for i in range(0, boot, 1000)
            ]
        )
    )
    return {
        "n": n,
        "base": sum(base) / n,
        "arm": sum(arm) / n,
        "delta": d,
        "ci95": (d - 1.96 * se, d + 1.96 * se),
        "boot95": (float(bs[int(0.025 * boot)]), float(bs[int(0.975 * boot) - 1])),
        "only_base": only_base,
        "only_arm": only_arm,
        "mcnemar_p": mcnemar_p(only_base, only_arm),
    }


def load(paths: Iterable[str]) -> tuple[dict[str, dict[int, dict[str, Any]]], int]:
    """Rows by arm and seed (the last row of a (seed, arm) wins), and the number of
    (seed, arm) runs that only errored."""
    by: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    errs: set[tuple[str, int]] = set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "seed" not in r:
                    continue
                if "error" in r:
                    errs.add((r["arm"], int(r["seed"])))
                else:
                    by[r["arm"]][int(r["seed"])] = r
    return by, sum(s not in by.get(a, {}) for a, s in errs)


def summarize(by: Mapping[str, Mapping[int, dict[str, Any]]]) -> dict[str, Any]:
    base = by.get("none", {})
    ctl = by.get("deviate", {})
    out: dict[str, Any] = {}
    for arm, rows in by.items():
        rs = list(rows.values())
        n = len(rs)
        forced = [r for r in rs if r["forced"]]
        fcards = Counter(c for r in rs for c in r["forced_cards"] if c)
        played = [sum(r["plays"].get(c, 0) for c in set(r["forced_cards"]) if c) for r in forced]
        s: dict[str, Any] = {
            "n": n,
            "win": sum(r["won"] for r in rs) / n,
            "deaths_by_act": dict(sorted(Counter(r["act"] for r in rs if not r["won"]).items())),
            "mean_floor": sum(r["floor"] for r in rs) / n,
            "eligible_runs": sum(r["eligible"] > 0 for r in rs) / n,
            "forced_runs": len(forced) / n,
            "override_runs": sum(r["overrides"] > 0 for r in rs) / n,
            "forced_per_run": sum(r["forced"] for r in rs) / n,
            "forced_cards": dict(fcards.most_common()),
            "forced_card_plays_per_forced_run": sum(played) / max(len(forced), 1),
            "forced_runs_playing_it": sum(x > 0 for x in played) / max(len(forced), 1),
            "deck_targets": sum(r["deck_targets"] for r in rs) / n,
            "target_plays": sum(
                v for r in rs for k, v in r["plays"].items() if k in _payoff(r["char"])
            )
            / n,
            "doom_applied": sum(r["doom_applied"] for r in rs) / n,
            "doom_kills": sum(r["doom_kills"] for r in rs) / n,
            "poison_applied": sum(r["poison_applied"] for r in rs) / n,
            "poison_damage": sum(r["poison_damage"] for r in rs) / n,
            "poison_kills": sum(r["poison_kills"] for r in rs) / n,
            "lost_fights": sum(r["lost_fights"] for r in rs),
            "log_dropped": sum(r["log_dropped"] for r in rs),
        }
        for name, ref in (("vs_none", base), ("vs_deviate", ctl)):
            if arm in ("none", name[3:]) or not ref:
                continue
            seeds = sorted(set(rows) & set(ref))
            s[name] = paired([ref[x]["won"] for x in seeds], [rows[x]["won"] for x in seeds])
            ov = [x for x in seeds if rows[x]["overrides"] > 0]
            s[name + "_overridden"] = paired(
                [ref[x]["won"] for x in ov], [rows[x]["won"] for x in ov]
            )
        out[arm] = s
    return out


def _payoff(char: str) -> frozenset[str]:
    return TARGETS[char].payoff


def _ci(x: Mapping[str, Any]) -> str:
    if not x.get("n"):
        return "n=0"
    lo, hi = x["boot95"]
    return (
        f"{100 * x['delta']:+6.2f}pp [{100 * lo:+.2f}, {100 * hi:+.2f}] "
        f"p={x['mcnemar_p']:.3g} n={x['n']} ({x['only_base']}/{x['only_arm']})"
    )


def report(a: argparse.Namespace) -> int:
    by, errors = load(a.paths)
    if not by:
        print("forced_rollout: no rows", file=sys.stderr)
        return 2
    s = summarize(by)
    order = [k for k in ARMS if k in s]
    print(f"runs that only errored: {errors}")
    for arm in order:
        x = s[arm]
        print(
            f"\n{arm:9} n={x['n']} win {100 * x['win']:.2f}%  floor {x['mean_floor']:.1f}  "
            f"deaths by act {x['deaths_by_act']}"
        )
        print(
            f"  eligible {100 * x['eligible_runs']:.1f}% of runs, forced "
            f"{100 * x['forced_runs']:.1f}%, overridden {100 * x['override_runs']:.1f}%, "
            f"{x['forced_per_run']:.2f} forced/run; forced card played "
            f"{x['forced_card_plays_per_forced_run']:.1f}x per forced run "
            f"({100 * x['forced_runs_playing_it']:.0f}% play it)"
        )
        print(
            f"  deck targets {x['deck_targets']:.2f}, target plays {x['target_plays']:.1f}, "
            f"doom applied {x['doom_applied']:.1f} kills {x['doom_kills']:.2f}, poison applied "
            f"{x['poison_applied']:.1f} dmg {x['poison_damage']:.1f} kills {x['poison_kills']:.2f}"
        )
        for key in ("vs_none", "vs_none_overridden", "vs_deviate", "vs_deviate_overridden"):
            if key in x:
                print(f"  {key:22} {_ci(x[key])}")
        if x["forced_cards"]:
            print(f"  forced cards {x['forced_cards']}")
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ppo.e2e.forced_rollout", description=__doc__)
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play the seeds under each arm")
    r.add_argument("--char", required=True, choices=sorted(TARGETS))
    r.add_argument("--ckpt", required=True, help="net checkpoint (a DEV eval's ema0999.pt)")
    r.add_argument("--pin", default="", help="wheel pin dir for the clean env")
    r.add_argument("--set-env", action="append", default=[], help="extra KEY=VALUE")
    r.add_argument("--seed-start", type=int, required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--chunk", type=int, default=64, help="runs per worker task")
    r.add_argument("--shuffle", type=int, default=0, help="shuffle the runs with this seed")
    r.add_argument("--max-seconds", type=float, default=0, help="stop after this long")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", required=True, help="JSONL, appended to (resumes)")
    r.set_defaults(func=run)
    q = sub.add_parser("report", help="paired comparison of the arms")
    q.add_argument("paths", nargs="+")
    q.add_argument("--json", default="")
    q.set_defaults(func=report)
    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
