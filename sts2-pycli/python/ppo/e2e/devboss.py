"""DEVBOSS: 600 boss-entry states x 4 shared reseeds, played by any agent (plan section 11).

    # 1. capture, once: teacher-acted runs from seed 175,000,000 up
    PYTHONPATH=python <W2 python> -m ppo.e2e.devboss capture --out DIR \\
        --run-ckpt RUN.pt --combat-ckpt COMBAT.pt --procs 8
    # 2. play: REF (the shipped MCTS), a combat-PPO net, or the student
    ... -m ppo.e2e.devboss play --states DIR --agent mcts --out REF.jsonl --procs 8
    ... -m ppo.e2e.devboss play --states DIR --agent student --policy MODULE:FUNC \\
        --ckpt FINAL.pt --out FINAL.jsonl --procs 8
    # 3. read (any time; DEVBOSS is not sealed)
    ... -m ppo.e2e.devboss report FINAL.jsonl [REF.jsonl]

**States.** Teacher-acted campaign runs are recorded action by action. A run is
:func:`ppo.e2e.evaluate.construct_run` (A10, v0.111.0, agent mode, potion swap
on), played by the teacher pair the way ``ppo.probe --policy net_greedy
--combat policy`` plays it. Before every Move into a boss room the run is
captured: act 1's boss (``act1_boss``), act 2's (``act2_boss``) and act 3's
first boss, which opens the double boss (``act3_double``). The first 200
captures per cell in seed order are kept; ``state_id`` runs over the cells in
that order, then by seed. A ``RunManager`` does not pickle, so a state is its
run seed plus the action prefix that reaches it, replayed: construct, then
``take_action`` each logged action, binding an event (``materialize_event``)
before each action taken in an event room, as the run did. The capture replays
every kept state from the JSON it wrote and checks a fingerprint of the
replayed run (RNG counters, deck, relics, potions, HP, gold, map position and
the legal actions); a mismatch stops the capture.

**Pairs.** Pair (i, j) is state i with reseed ``175,500,000 + 4i + j``:
``clone``, ``reseed_run``, then the captured Move. Every agent plays the same
pairs, so the reseed fixes the fight's shuffles, AI and rewards. An act-1 or
act-2 pair ends when its boss fight does (won: alive outside combat; lost: the
run is over or a cap hit). An ``act3_double`` pair plays both bosses to the
run's end; the agent plays whatever lies between them.

**Agents.** ``mcts``: the shipped combat search (``sts2_rs.MctsCombatAgent``
seeded with the reseed; 1000 sims, 4000 in boss rooms): REF. ``policy``: a
combat-PPO checkpoint (``ppo.combat_ppo.agent.PolicyAgent``). ``student``:
``--policy`` as in ``ppo.e2e.evaluate``, in ``--segment-env`` (default
:func:`e2e_segment_env`, WP2's env on a practice-shaped segment; ``raw`` is the
stand-in).

**Read.** Final - REF on the pooled win rate, a ratio mean over states with a
cluster bootstrap (a state and its reseeds are one cluster; 20,000 resamples,
``default_rng(20261109)``, a recentred two-sided p): the fourth member of
PREREG-0's descriptive family. P2's rules carry over pair by pair: a REF error
drops the pair, a Final-only error is a Final loss, and more than 0.5% such
pairs make the read INVALID. So do a partial panel, a REF that is not the
shipped MCTS, and two engine builds.

**Running it.** ``capture`` and ``play`` select v0.111.0 for the process and
its spawn workers (``STS2_GAME_VERSION``; another explicit selection is
refused) and refuse any ``STS2_RUNAI_/ROLLOUT_/MCTS_`` knob in the environment.
Both resume: a stopped capture keeps its finished seeds (``work/rows.jsonl``)
and a stopped play its finished pairs (``<out>.part``), as long as the config
(checkpoints, code, wheel) is unchanged; otherwise they refuse. A finished
capture directory and a finished result file are never overwritten. Rows hold
no wall clock, so a repeat on the same binary is byte-identical.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppo.e2e import evaluate as E
from ppo.e2e import panels as P
from ppo.e2e import stats as S

CELLS = P.DEVBOSS_CELLS
#: The cell a boss Move opens, by (act_index, boss slot); act_index is 1-based.
CELL_OF = {(1, "boss"): "act1_boss", (2, "boss"): "act2_boss", (3, "boss"): "act3_double"}
#: WP2's practice-cell names (``ppo.e2e.contracts.CELLS``) for the default student env.
WP2_CELL = {"act1_boss": "BOSS_ACT1", "act2_boss": "BOSS_ACT2", "act3_double": "BOSS_ACT3_DOUBLE"}
#: The teacher's run net must carry the deck-mutation features (plan section 2).
RUN_FEATURES = frozenset({"card_mods", "reward_upgrade"})
#: REF as PREREG-0 defines it: the shipped search budget.
REF_SIMS, REF_BOSS_SIMS = 1000, 4000
STATES = "states.jsonl"
RUNS = "runs.jsonl"
CAPTURE = "capture.json"
SUMS = "SHA256SUMS"
#: Steps per pair (agent and forced steps): a backstop, booked as a cap.
MAX_PAIR_STEPS = 20_000
_RUN_SEED = re.compile(r'\{"seed": (\d+),')


class DevbossError(RuntimeError):
    """A refusal or an integrity failure."""


# ── run state ─────────────────────────────────────────────────────────────


def fingerprint(rm: Any) -> str:
    """A digest of the run's observable and RNG state: a replay must reproduce it."""
    d = {
        "rng": rm.rng_counters(),
        "deck": rm.master_deck_detailed(),
        "ench": rm.master_deck_enchantments(),
        "relics": rm.relics(),
        "potions": rm.potions(),
        "hp": rm.current_hp,
        "max_hp": rm.max_hp,
        "gold": rm.gold,
        "floor": rm.total_floor(),
        "act": rm.act_index,
        "coord": rm.map_coord(),
        "phase": rm.phase(),
        "actions": rm.available_actions(),
    }
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def boss_move(rm: Any, action: dict[str, Any]) -> str | None:
    """ "boss" or "second_boss" if `action` is the Move into that node, else None."""
    if action.get("action") != "move":
        return None
    m = rm.full_map() or {}
    cur = m.get("current_coord")
    if isinstance(cur, dict):
        row = int(cur["row"]) + 1
    elif cur:
        row = int(cur[1]) + 1
    else:
        row = 0
    for slot in ("boss", "second_boss"):
        b = m.get(slot)
        if b and b.get("col") == action.get("col") and b.get("row") == row:
            return slot
    return None


def replay(seed: int, actions: Sequence[dict[str, Any]]) -> Any:
    """The campaign run for `seed` after `actions`, binding events as the run did."""
    rm = E.construct_run(int(seed))
    for a in actions:
        if rm.phase() == "event":
            rm.materialize_event()
        rm.take_action(dict(a))
    if rm.phase() == "event":
        rm.materialize_event()
    return rm


class _Recorder:
    """A `RunManager` stand-in that calls `hook(rm, action)` before every `take_action`."""

    def __init__(self, rm: Any, hook: Any) -> None:
        object.__setattr__(self, "_rm", rm)
        object.__setattr__(self, "_hook", hook)

    def take_action(self, action: dict[str, Any]) -> Any:
        self._hook(self._rm, action)
        return self._rm.take_action(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rm, name)


class _Stop(Exception):
    """Raised by the capture hook once a run has given everything it can."""


def check_run_ckpt(path: str | Path) -> None:
    """Refuse a run net without the deck-mutation features: `NetPolicy` would silently
    read it at the default schema (net_policy.py:35-47)."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    feats = str((cfg or {}).get("features") or "")
    got = frozenset(t.strip() for t in feats.split(",") if t.strip())
    if got != RUN_FEATURES:
        raise DevbossError(f"{path}: cfg features {feats!r}, expected 'card_mods,reward_upgrade'")


# ── capture ───────────────────────────────────────────────────────────────


class Capturer:
    """Pool handler: play one teacher run; return its action log and boss captures."""

    def __init__(self, run_ckpt: str, combat_ckpt: str) -> None:
        E.require_campaign_build()
        self.run_ckpt = run_ckpt
        self.combat_ckpt = combat_ckpt

    def play(self, seed: int) -> dict[str, Any]:
        try:
            return self._play(seed)
        except DevbossError as e:
            return {"seed": seed, "error": str(e), "fatal": True}
        except BaseException as e:  # a Rust panic is a BaseException
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            return {"seed": seed, "error": E._tail(traceback.format_exc())}

    def _play(self, seed: int) -> dict[str, Any]:
        import numpy as np

        from ppo.env import Decision, PpoRunEnv
        from ppo.workers import _net_policy

        log: list[dict[str, Any]] = []
        caps: list[dict[str, Any]] = []

        def hook(rm: Any, action: dict[str, Any]) -> None:
            slot = boss_move(rm, action)
            cell = CELL_OF.get((int(rm.act_index), slot)) if slot else None
            if cell is not None:
                m = rm.full_map() or {}
                caps.append(
                    {
                        "cell": cell,
                        "k": len(log),
                        "move": dict(action),
                        "fingerprint": fingerprint(rm),
                        "act": int(rm.act_index),
                        "floor": int(rm.total_floor()),
                        "hp": int(rm.current_hp),
                        "max_hp": int(rm.max_hp),
                        "gold": int(rm.gold),
                        "potions": sum(1 for p in rm.potions() if p),
                        "deck": len(rm.master_deck_detailed()),
                        "boss": (m.get("boss") or {}).get("id"),
                        "second_boss": (m.get("second_boss") or {}).get("id"),
                    }
                )
                if cell == "act3_double":
                    raise _Stop
            log.append(dict(action))

        # `ppo.probe --policy net_greedy --combat policy`, on the campaign run.
        env = PpoRunEnv(
            seed=seed,
            ascension=E.ASCENSION,
            character=E.CHARACTER,
            combat="policy",
            combat_ckpt=self.combat_ckpt,
            act1="roll",
            game_version=E.GAME_VERSION,
        )
        env.rm = _Recorder(E.construct_run(seed), hook)
        policy = _net_policy(self.run_ckpt)
        rng = np.random.default_rng(seed)
        outcome = "stopped"
        try:
            out = env.reset()
            while isinstance(out, Decision):
                idx = policy.sample(out.obs, out.actions, out.metrics, rng, greedy=True)[0]
                out = env.step(idx)
            outcome = str(out.outcome)
        except _Stop:
            pass
        # Every capture must replay from the JSON written to disk.
        wire = json.loads(json.dumps(log))
        for c in caps:
            got = fingerprint(replay(seed, wire[: c["k"]]))
            if got != c["fingerprint"]:
                raise DevbossError(f"seed {seed} {c['cell']}: the replay fingerprint differs")
        need = max((c["k"] for c in caps), default=0)
        return {"seed": seed, "outcome": outcome, "captures": caps, "actions": wire[:need]}


def _file_sha(path: str | Path) -> str:
    return E.sha256_file(Path(path))


def _clean_or_die() -> None:
    """DEVBOSS numbers must not depend on the caller's engine knobs."""
    bad = sorted(k for k in os.environ if k.startswith(E.KNOB_PREFIXES))
    if bad:
        raise DevbossError(f"engine knobs set in the environment: {bad}; unset them")


def _provenance() -> dict[str, Any]:
    from ppo.wheel_guard import python_source_digest, wheel_identity

    sha, dirty = E.git_head()
    return {
        "git_sha": sha,
        "dirty": dirty,
        "wheel": wheel_identity(),
        "py_src": python_source_digest(),
        "game_version": E.GAME_VERSION,
    }


def _stamp() -> str:
    return dt.datetime.now().astimezone().strftime("%FT%T%z")


def _read_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Rows of an append-only work file by seed; a torn last line (a crash) is dropped."""
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[int(r["seed"])] = r
    return rows


def capture_cmd(args: argparse.Namespace) -> int:
    _clean_or_die()
    E.select_campaign_build()
    per_cell = int(args.per_cell)
    if not 1 <= per_cell <= P.DEVBOSS_STATES_PER_CELL:
        raise DevbossError(f"--per-cell must be in 1..{P.DEVBOSS_STATES_PER_CELL}")
    if args.procs < 1 or args.batch < 1:
        raise DevbossError("--procs and --batch must be positive")
    out = Path(args.out).expanduser().resolve()
    if (out / SUMS).exists():
        raise DevbossError(f"{out} holds a finished capture; capture into a new directory")
    for p in (args.run_ckpt, args.combat_ckpt):
        if not Path(p).is_file():
            raise DevbossError(f"no checkpoint {p}")
    check_run_ckpt(args.run_ckpt)
    handler_args = {
        "run_ckpt": str(Path(args.run_ckpt).resolve()),
        "combat_ckpt": str(Path(args.combat_ckpt).resolve()),
    }
    config = {
        **handler_args,
        "run_ckpt_sha256": _file_sha(args.run_ckpt),
        "combat_ckpt_sha256": _file_sha(args.combat_ckpt),
        "per_cell": per_cell,
        **_provenance(),
    }
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "config.json"
    cfg_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if cfg_path.is_file() and cfg_path.read_text(encoding="utf-8") != cfg_text:
        raise DevbossError(f"{out} was started with another config ({cfg_path}); use a new --out")
    E.write_atomic(cfg_path, cfg_text)
    # Finished seeds survive a restart: the kept states depend only on seed order.
    done = _read_rows(work / "rows.jsonl")
    rows_f = open(work / "rows.jsonl", "a", encoding="utf-8")
    log_f = open(out / "capture.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{_stamp()} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    got: dict[str, list[dict[str, Any]]] = {c: [] for c in CELLS}
    runs: dict[int, list[dict[str, Any]]] = {}
    errors: list[int] = []

    def take(seed: int, row: dict[str, Any]) -> None:
        if row.get("fatal"):
            raise DevbossError(f"seed {seed}: {row['error']}")
        if "error" in row:
            errors.append(seed)
            return
        kept = False
        for c in row["captures"]:
            if len(got[c["cell"]]) < per_cell:
                got[c["cell"]].append({"seed": seed, **c})
                kept = True
        if kept:
            runs[seed] = row["actions"]

    def full() -> bool:
        return all(len(got[c]) >= per_cell for c in CELLS)

    def on_row(s: int, row: dict[str, Any]) -> None:
        done[s] = row
        rows_f.write(json.dumps(row, sort_keys=True) + "\n")
        rows_f.flush()

    cfg = {"handler": "ppo.e2e.devboss:Capturer", "handler_args": handler_args}
    seed = P.DEVBOSS_RUN_SEED_START
    t0 = time.time()
    say(f"capture into {out}: {per_cell} per cell, {len(done)} seed(s) already done")
    try:
        while not full():
            if seed >= P.DEVBOSS_RUN_SEED_STOP:
                raise DevbossError("ran out of DEVBOSS run seeds before every cell filled")
            batch = list(range(seed, min(seed + args.batch, P.DEVBOSS_RUN_SEED_STOP)))
            seed = batch[-1] + 1
            todo = [s for s in batch if s not in done]
            if todo:
                E.Pool(
                    cfg,
                    args.procs,
                    args.hang_timeout,
                    on_row,
                    lambda s, why: {"seed": s, "error": why},
                ).run(todo)
            for s in batch:  # seed order decides, whatever the batch size
                if not full():
                    take(s, done[s])
            say(
                f"seeds ..{batch[-1]}: "
                + " ".join(f"{c}={len(got[c])}" for c in CELLS)
                + f" errors={len(errors)} wall={time.time() - t0:.0f}s"
            )
    finally:
        rows_f.close()
        log_f.close()

    states = []
    for c in CELLS:
        for s in got[c]:
            i = len(states)
            reseeds = [P.devboss_reseed(i, j) for j in range(P.DEVBOSS_RESEEDS_PER_STATE)]
            states.append({**s, "state_id": i, "reseeds": reseeds})
    # Each run's log only as long as its last kept state needs.
    need: dict[int, int] = defaultdict(int)
    for s in states:
        need[s["seed"]] = max(need[s["seed"]], s["k"])
    E.write_atomic(
        out / RUNS,
        "".join(
            json.dumps({"seed": sd, "actions": runs[sd][: need[sd]]}) + "\n" for sd in sorted(need)
        ),
    )
    E.write_atomic(out / STATES, "".join(json.dumps(s, sort_keys=True) + "\n" for s in states))
    meta = {
        "what": "DEVBOSS boss-entry states (plan section 11)",
        **config,
        "seeds_tried": [P.DEVBOSS_RUN_SEED_START, seed - 1],
        "errored_seeds": errors,
        "counts": {c: len(got[c]) for c in CELLS},
    }
    E.write_atomic(out / CAPTURE, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    sums = "".join(f"{_file_sha(out / f)}  {f}\n" for f in (CAPTURE, RUNS, STATES))
    E.write_atomic(out / SUMS, sums)
    print(f"devboss: {len(states)} states in {out}; SHA256SUMS {_file_sha(out / SUMS)}")
    return 0


# ── states ────────────────────────────────────────────────────────────────


@dataclass
class StateSet:
    """A finished capture directory, checked against its SHA256SUMS.

    Run logs are decoded only when a state needs them (a worker plays a few states).
    """

    root: Path
    states: list[dict[str, Any]]
    sums_sha256: str
    run_lines: dict[int, str]

    @classmethod
    def load(cls, root: str | Path) -> StateSet:
        root = Path(root).expanduser().resolve()
        sums = root / SUMS
        if not sums.is_file():
            raise DevbossError(f"{sums}: missing (an unfinished capture?)")
        names = []
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                if _file_sha(root / name) != digest:
                    raise DevbossError(f"{root / name}: sha256 differs from {sums}")
                names.append(name)
        if sorted(names) != sorted((CAPTURE, RUNS, STATES)):
            raise DevbossError(f"{sums} lists {names}")
        rows = [json.loads(x) for x in (root / STATES).read_text(encoding="utf-8").splitlines()]
        if [s["state_id"] for s in rows] != list(range(len(rows))):
            raise DevbossError(f"{root / STATES}: state ids are not 0..{len(rows) - 1}")
        lines: dict[int, str] = {}
        for x in (root / RUNS).read_text(encoding="utf-8").splitlines():
            m = _RUN_SEED.match(x)
            if not m:
                raise DevbossError(f"{root / RUNS}: a line without a leading seed")
            lines[int(m.group(1))] = x
        return cls(root=root, states=rows, sums_sha256=_file_sha(sums), run_lines=lines)

    def prefix(self, i: int) -> list[dict[str, Any]]:
        s = self.states[i]
        actions = json.loads(self.run_lines[int(s["seed"])])["actions"]
        return actions[: s["k"]]

    def pairs(self, limit_states: int | None = None) -> list[tuple[int, int]]:
        """Every (state, reseed) pair; with `limit_states`, only each cell's first N states."""
        ids = [s["state_id"] for s in self.states]
        if limit_states is not None:
            by_cell: dict[str, list[int]] = defaultdict(list)
            for s in self.states:
                if len(by_cell[s["cell"]]) < limit_states:
                    by_cell[s["cell"]].append(s["state_id"])
            ids = sorted(i for v in by_cell.values() for i in v)
        return [(i, j) for i in ids for j in range(len(self.states[i]["reseeds"]))]

    def base(self, i: int) -> Any:
        """State `i` replayed and checked against its capture fingerprint."""
        s = self.states[i]
        rm = replay(s["seed"], self.prefix(i))
        if fingerprint(rm) != s["fingerprint"]:
            raise DevbossError(f"state {i}: the replay fingerprint differs from the capture")
        return rm

    def start(self, i: int, j: int, base: Any = None) -> Any:
        """Pair (i, j)'s start: the state's clone, reseeded, after the captured Move."""
        s = self.states[i]
        rm = (self.base(i) if base is None else base).clone()
        rm.reseed_run(int(s["reseeds"][j]))
        rm.take_action(dict(s["move"]))
        return rm


# ── play ──────────────────────────────────────────────────────────────────


def e2e_segment_env(seed: int, rm: Any, cell: str, *, reseed: int, move: dict[str, Any]) -> Any:
    """The student's env from a pair's start: WP2's ``E2EEnv`` on a segment shaped like
    its practice segments (``practice.segment_for``): act 1 and 2 bosses end at the first
    out-of-combat decision, the double boss at the run's end."""
    from ppo.e2e.env import E2EEnv, Segment  # type: ignore[attr-defined]

    seg = Segment(
        kind="devboss",
        cell=WP2_CELL[cell],
        end="terminal" if cell == "act3_double" else "first_ooc",
        parent_seed=int(seed),
        move=tuple(sorted(move.items())),
        reseed=int(reseed),
    )
    return E2EEnv(int(seed), rm=rm, segment=seg)


def raw_segment_env(seed: int, rm: Any, cell: str, *, reseed: int, move: dict[str, Any]) -> Any:
    """STAND-IN: :class:`ppo.e2e.evaluate.RawEnv` from the pair's start."""
    del cell, reseed, move
    return E.RawEnv(seed, rm=rm)


class _Fights:
    """The boss fights of one pair: entry and end HP, turns, result.

    A fight is entered when the run is first seen in a boss room, and closed when the
    next boss room is entered or the pair ends. Its end HP is read then.
    """

    def __init__(self) -> None:
        self.fights: list[dict[str, Any]] = []
        self._where: tuple[Any, ...] | None = None

    def observe(self, rm: Any) -> None:
        if rm.is_over():
            return
        room = rm.current_room() or {}
        where = (int(rm.act_index), tuple(room.get("map_coord") or ()))
        if room.get("room_type") == "boss" and where != self._where:
            self.close(rm, True)
            slot, boss = E.boss_slot(rm, where[1])
            self.fights.append(
                {
                    "act": where[0],
                    "slot": slot,
                    "boss": boss,
                    "entry_hp": int(rm.current_hp),
                    "max_hp": int(rm.max_hp),
                    "entry_potions": sum(1 for p in rm.potions() if p),
                    "turns": 0,
                    "won": None,
                }
            )
            self._where = where
        if self.fights and self.fights[-1]["won"] is None and rm.phase() == "combat":
            cs = rm.combat_state()
            if cs is not None:
                rnd = int(cs.get_state().get("round") or 0)
                self.fights[-1]["turns"] = max(self.fights[-1]["turns"], rnd)

    def close(self, rm: Any, won: bool) -> None:
        if self.fights and self.fights[-1]["won"] is None:
            f = self.fights[-1]
            f["won"] = bool(won)
            f["end_hp"] = 0 if rm.is_over() and not rm.player_won else int(rm.current_hp)
            f["hp_lost"] = f["entry_hp"] - f["end_hp"]

    def fight_over(self, rm: Any) -> bool:
        """Alive and out of combat after a boss room was entered: that boss is beaten."""
        if rm.is_over() or not self.fights:
            return False
        return rm.phase() != "combat" and not E.is_combat(rm.available_actions())

    def turns_capped(self) -> bool:
        return bool(self.fights) and self.fights[-1]["turns"] > E.MAX_FIGHT_TURNS


def _done(cell: str, fights: _Fights, rm: Any) -> bool:
    """The pair is over: its last boss is beaten (both, for the double boss)."""
    if not fights.fight_over(rm):
        return False
    return cell != "act3_double" or len(fights.fights) >= 2


class Player:
    """Pool handler: play DEVBOSS pairs with one agent."""

    def __init__(
        self,
        states: str,
        agent: str,
        sims: int = REF_SIMS,
        boss_sims: int = REF_BOSS_SIMS,
        combat_ckpt: str | None = None,
        policy: str | None = None,
        ckpt: str | None = None,
        device: str = "cpu",
        segment_env: str = "ppo.e2e.devboss:e2e_segment_env",
    ) -> None:
        E.require_campaign_build()
        E.eval_stamps()  # taken once, here: every row of this worker carries them
        self.set = StateSet.load(states)
        self.agent = agent
        self.sims, self.boss_sims = int(sims), int(boss_sims)
        self._base: tuple[int, Any] | None = None
        if agent == "policy":
            from ppo.combat_ppo.agent import PolicyAgent

            if not combat_ckpt:
                raise DevbossError("--agent policy needs --combat-ckpt")
            self._policy_agent = PolicyAgent(0, combat_ckpt)
        elif agent == "student":
            if not policy:
                raise DevbossError("--agent student needs --policy")
            self.policy = E.make_policy(policy, ckpt, device)
            self.make_env = raw_segment_env if segment_env == "raw" else E.load_spec(segment_env)
            if self.make_env is e2e_segment_env:
                # Fail at worker start (the call stops), not as an error row per pair.
                importlib.import_module("ppo.e2e.env")
        elif agent != "mcts":
            raise DevbossError(f"unknown agent {agent!r}")

    def _start(self, i: int, j: int) -> Any:
        if self._base is None or self._base[0] != i:
            self._base = (i, self.set.base(i))
        return self.set.start(i, j, self._base[1])

    def play(self, task: tuple[int, int]) -> dict[str, Any]:
        i, j = (int(x) for x in task)
        s = self.set.states[i]
        head = {
            "state_id": i,
            "j": j,
            "reseed": s["reseeds"][j],
            "cell": s["cell"],
            **E.eval_stamps(),  # the process's six stamps, as on every eval row
        }
        try:
            rm = self._start(i, j)
            if self.agent == "student":
                body = self._student(s, rm, int(s["reseeds"][j]))
            else:
                body = self._direct(s, rm, int(s["reseeds"][j]))
        except BaseException as e:  # a Rust panic is a BaseException
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            tb = E._tail(traceback.format_exc())
            return {**head, "outcome": "error", "won": False, "error": tb}
        return {**head, **body}

    @staticmethod
    def _end(cell: str, rm: Any, fights: _Fights, cap: str | None, steps: int) -> dict[str, Any]:
        if cap is not None:
            fights.close(rm, False)
            won, outcome = False, "capped"
        elif rm.is_over():
            won = bool(rm.player_won)
            fights.close(rm, won)
            outcome = "won" if won else "lost"
        else:  # alive after the cell's last boss
            fights.close(rm, True)
            won, outcome = True, "won"
        if (
            cell == "act3_double"
            and won
            and not (len(fights.fights) == 2 and all(f["won"] for f in fights.fights))
        ):
            won, outcome = False, "lost"
        return {"outcome": outcome, "won": won, "cap": cap, "steps": steps, "fights": fights.fights}

    def _direct(self, s: dict[str, Any], rm: Any, reseed: int) -> dict[str, Any]:
        import sts2_rs

        if self.agent == "mcts":
            agent: Any = sts2_rs.MctsCombatAgent(
                seed=reseed, sims=self.sims, boss_sims=self.boss_sims
            )
        else:
            agent = self._policy_agent
        fights = _Fights()
        steps = 0
        while True:
            fights.observe(rm)
            if rm.is_over() or _done(s["cell"], fights, rm):
                return self._end(s["cell"], rm, fights, None, steps)
            if steps >= MAX_PAIR_STEPS:
                return self._end(s["cell"], rm, fights, "steps", steps)
            if fights.turns_capped():
                return self._end(s["cell"], rm, fights, "turns", steps)
            if rm.phase() == "event":
                rm.materialize_event()
            if agent.step(rm) is None:
                raise DevbossError(f"the {self.agent} agent had no action at phase {rm.phase()}")
            steps += 1

    def _student(self, s: dict[str, Any], rm: Any, reseed: int) -> dict[str, Any]:
        begin = getattr(self.policy, "begin_run", None)
        if begin is not None:
            begin(reseed)
        env = self.make_env(int(s["seed"]), rm, s["cell"], reseed=reseed, move=dict(s["move"]))
        fights = _Fights()
        out = env.reset()
        decisions = 0
        while E.is_decision(out) and not getattr(out, "bootstrap", False):
            fights.observe(rm)
            if _done(s["cell"], fights, rm):
                return {**self._end(s["cell"], rm, fights, None, decisions), "decisions": decisions}
            (choice,) = self.policy.act_decisions([out])
            if isinstance(choice, bool) or not 0 <= int(choice) < len(out.members):
                raise ValueError(f"policy chose class {choice!r} of {len(out.members)}")
            out = env.step(int(choice))
            decisions += 1
        fights.observe(rm)
        if E.is_decision(out):  # a bootstrap row: the segment ended alive, out of combat
            return {**self._end(s["cell"], rm, fights, None, decisions), "decisions": decisions}
        if out.outcome == "error":
            raise DevbossError(f"env error: {getattr(out, 'error', '')}")
        cap = getattr(out, "cap", None)
        if out.outcome == "capped" and not cap:
            cap = "capped"
        return {**self._end(s["cell"], rm, fights, cap, decisions), "decisions": decisions}


def _play_meta(args: argparse.Namespace, states: StateSet, n_pairs: int) -> dict[str, Any]:
    agent_args: dict[str, Any] = {"agent": args.agent}
    if args.agent == "mcts":
        agent_args.update(sims=args.sims, boss_sims=args.boss_sims)
    elif args.agent == "policy":
        agent_args.update(combat_ckpt=str(Path(args.combat_ckpt).resolve()))
    else:
        agent_args.update(
            policy=args.policy,
            ckpt=str(Path(args.ckpt).resolve()) if args.ckpt else None,
            device=args.device,
            segment_env=args.segment_env,
        )
    return {
        "what": "DEVBOSS results",
        **agent_args,
        "ckpt_sha256": _file_sha(args.ckpt) if args.agent == "student" and args.ckpt else None,
        "combat_ckpt_sha256": _file_sha(args.combat_ckpt) if args.agent == "policy" else None,
        "states": str(states.root),
        "states_sha256sums": states.sums_sha256,
        "pairs": n_pairs,
        "limit_states": args.limit_states,
        **_provenance(),
    }


def play_cmd(args: argparse.Namespace) -> int:
    _clean_or_die()
    E.select_campaign_build()
    if args.procs < 1:
        raise DevbossError("--procs must be positive")
    if args.agent == "policy" and not args.combat_ckpt:
        raise DevbossError("--agent policy needs --combat-ckpt")
    if args.agent == "student" and not args.policy:
        raise DevbossError("--agent student needs --policy")
    for p in (args.ckpt, args.combat_ckpt):
        if p and not Path(p).is_file():
            raise DevbossError(f"no checkpoint {p}")
    states = StateSet.load(args.states)
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise DevbossError(f"{out} exists; results are written once")
    pairs = states.pairs(args.limit_states)
    head = json.dumps({"meta": _play_meta(args, states, len(pairs))}, sort_keys=True) + "\n"
    # Finished pairs survive a restart, under the same meta only.
    part = out.with_name(out.name + ".part")
    results: dict[tuple[int, int], dict[str, Any]] = {}
    if part.is_file():
        lines = part.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines or lines[0] != head:
            raise DevbossError(
                f"{part} was written under another config; delete it or rename --out"
            )
        for x in lines[1:]:
            try:
                r = json.loads(x)
            except ValueError:  # a torn last line
                continue
            results[int(r["state_id"]), int(r["j"])] = r
    part_f: Any = None  # opened at the first row, so a refused call leaves no file

    def write_part(line: str) -> None:
        nonlocal part_f
        if part_f is None:
            fresh = not part.is_file()
            part_f = open(part, "a", encoding="utf-8")
            if fresh:
                part_f.write(head)
        part_f.write(line)
        part_f.flush()

    todo = [p for p in pairs if p not in results]
    handler_args = {
        "states": str(states.root),
        "agent": args.agent,
        "sims": args.sims,
        "boss_sims": args.boss_sims,
        "combat_ckpt": str(Path(args.combat_ckpt).resolve()) if args.combat_ckpt else None,
        "policy": args.policy,
        "ckpt": str(Path(args.ckpt).resolve()) if args.ckpt else None,
        "device": args.device,
        "segment_env": args.segment_env,
    }
    t0 = time.time()
    print(
        f"devboss: {args.agent} on {len(pairs)} pairs ({len(todo)} to play), {args.procs} procs",
        flush=True,
    )

    def on_row(task: tuple[int, int], row: dict[str, Any]) -> None:
        results[int(task[0]), int(task[1])] = row
        write_part(json.dumps(row, sort_keys=True) + "\n")
        n = len(results)
        if n % 50 == 0 or n == len(pairs):
            print(f"{_stamp()} {n}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)

    def make_error(task: tuple[int, int], why: str) -> dict[str, Any]:
        i, j = (int(x) for x in task)
        s = states.states[i]
        return {
            "state_id": i,
            "j": j,
            "reseed": s["reseeds"][j],
            "cell": s["cell"],
            "outcome": "error",
            "won": False,
            "error": why,
            **E.eval_stamps(),  # the parent books it (a hung or dead worker)
        }

    cfg = {"handler": "ppo.e2e.devboss:Player", "handler_args": handler_args}
    try:
        E.Pool(cfg, args.procs, args.hang_timeout, on_row, make_error).run(todo)
    finally:
        if part_f is not None:
            part_f.close()
    body = "".join(json.dumps(results[p], sort_keys=True) + "\n" for p in pairs)
    E.write_atomic(out, head + body)
    part.unlink(missing_ok=True)
    errs = sum(1 for p in pairs if results[p].get("outcome") == "error")
    print(f"devboss: {len(pairs)} pairs ({errs} errors) in {time.time() - t0:.0f}s -> {out}")
    print(f"devboss: sha256 {_file_sha(out)}")
    return 0


# ── read ──────────────────────────────────────────────────────────────────


def load_results(
    path: str | Path,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise DevbossError(f"{path}: empty")
    meta = json.loads(lines[0]).get("meta")
    if not isinstance(meta, dict):
        raise DevbossError(f"{path}: no meta header")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for x in lines[1:]:
        if x.strip():
            r = json.loads(x)
            key = (int(r["state_id"]), int(r["j"]))
            if key in rows:
                raise DevbossError(f"{path}: pair {key} twice")
            rows[key] = r
    if len(rows) != meta.get("pairs"):
        raise DevbossError(f"{path}: {len(rows)} rows, the header says {meta.get('pairs')}")
    return meta, rows


def summary(rows: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    """Per-cell and pooled win rate, HP lost and turns of one agent (errors excluded)."""
    out: dict[str, Any] = {}
    for cell in (*CELLS, "pooled"):
        rs = [r for r in rows.values() if cell == "pooled" or r["cell"] == cell]
        ok = [r for r in rs if r.get("outcome") != "error"]
        f = [x for r in ok for x in r.get("fights") or [] if "hp_lost" in x]
        wins = sum(1 for r in ok if r["won"])
        out[cell] = {
            "pairs": len(rs),
            "errors": len(rs) - len(ok),
            "capped": sum(1 for r in ok if r.get("cap")),
            "wins": wins,
            "wr": wins / len(ok) if ok else None,
            "hp_lost": sum(x["hp_lost"] for x in f) / len(f) if f else None,
            "turns": sum(x["turns"] for x in f) / len(f) if f else None,
        }
    return out


def _wheel(meta: dict[str, Any]) -> dict[str, Any]:
    from ppo.e2e.score_test import WHEEL_KEYS

    w = meta.get("wheel") or {}
    return {k: w.get(k) for k in WHEEL_KEYS}


def paired_results(final: str | Path, ref: str | Path) -> dict[str, Any]:
    """DEVBOSS Final - REF: the descriptive family's fourth member (PREREG-0 P2).

    Pairs are (state, reseed). A REF error drops the pair; a Final-only error is a Final
    loss; more than 0.5% such pairs make the read INVALID, and so do a partial panel, a
    REF that is not the shipped MCTS (1000/4000) and two engine builds. The difference is
    the pooled win rate's, a ratio mean over states, with a cluster bootstrap.
    """
    fm, fr = load_results(final)
    rmeta, rr = load_results(ref)
    if fm.get("states_sha256sums") != rmeta.get("states_sha256sums"):
        raise DevbossError("Final and REF played different state sets")
    if set(fr) != set(rr):
        raise DevbossError("Final and REF played different pairs")
    dropped = final_losses = 0
    per_state: dict[int, list[float]] = {}
    cell_of: dict[int, str] = {}
    for key in sorted(fr):
        a, b = fr[key], rr[key]
        if a["cell"] != b["cell"]:
            raise DevbossError(f"pair {key}: cells differ ({a['cell']} vs {b['cell']})")
        if b.get("outcome") == "error":
            dropped += 1
            continue
        if a.get("outcome") == "error":
            final_losses += 1
            wa = 0
        else:
            wa = int(bool(a["won"]))
        cell_of[key[0]] = b["cell"]
        acc = per_state.setdefault(key[0], [0.0, 0])
        acc[0] += wa - int(bool(b["won"]))
        acc[1] += 1
    n = len(fr)
    max_bad = P.max_bad_seeds(n)
    invalid = []
    if dropped + final_losses > max_bad:
        invalid.append(
            f"{dropped + final_losses} pairs with missing data > {max_bad} (0.5% of {n})"
        )
    full = P.DEVBOSS_N_STATES * P.DEVBOSS_RESEEDS_PER_STATE
    if n != full or fm.get("limit_states") is not None:
        invalid.append(f"a partial panel: {n} pairs, {full} expected")
    shipped = ("mcts", REF_SIMS, REF_BOSS_SIMS)
    got = (rmeta.get("agent"), rmeta.get("sims"), rmeta.get("boss_sims"))
    if got != shipped:
        invalid.append(
            f"REF is {got[0]} {got[1]}/{got[2]}, not the shipped MCTS {REF_SIMS}/{REF_BOSS_SIMS}"
        )
    if _wheel(fm) != _wheel(rmeta):
        invalid.append("Final and REF ran on different engine builds")

    def boot(ids: list[int]) -> dict[str, Any]:
        return S.cluster_bootstrap([per_state[i][0] for i in ids], [per_state[i][1] for i in ids])

    ids = sorted(per_state)
    return {
        "pairs": n,
        "states": len(ids),
        "dropped_ref_missing": dropped,
        "final_missing_as_loss": final_losses,
        "valid": not invalid,
        "invalid": invalid,
        "pooled": boot(ids),
        "cells": {c: boot([i for i in ids if cell_of[i] == c]) for c in CELLS},
        "final": summary(fr),
        "ref": summary(rr),
        "final_meta": {k: fm.get(k) for k in ("agent", "policy", "ckpt", "ckpt_sha256")},
        "ref_meta": {k: rmeta.get(k) for k in ("agent", "sims", "boss_sims")},
    }


def _pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def report_cmd(args: argparse.Namespace) -> int:
    if args.ref:
        out = paired_results(args.final, args.ref)
    else:
        _, rows = load_results(args.final)
        out = {"final": summary(rows)}
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    for who in ("final", "ref"):
        if who not in out:
            continue
        print(f"-- {who}")
        for cell, v in out[who].items():
            wr = "-" if v["wr"] is None else f"{100 * v['wr']:.2f}%"
            print(
                f"   {cell:<12} WR {wr:>7} ({v['wins']}/{v['pairs'] - v['errors']},"
                f" errors {v['errors']}, capped {v['capped']})"
                f"  HP lost {v['hp_lost'] or 0:.1f}  turns {v['turns'] or 0:.1f}"
            )
    if "pooled" in out:
        p = out["pooled"]
        print(
            f"-- Final - REF: {_pp(p['mean'])}pp, cluster95 [{_pp(p['lo'])}, {_pp(p['hi'])}]pp,"
            f" p {p['p']:.3g} over {out['states']} states"
            + ("" if out["valid"] else f"  INVALID: {'; '.join(out['invalid'])}")
        )
        for c, v in out["cells"].items():
            print(f"   {c:<12} {_pp(v['mean'])}pp [{_pp(v['lo'])}, {_pp(v['hi'])}]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ppo.e2e.devboss", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="capture the boss-entry states from teacher runs")
    cap.add_argument("--out", required=True)
    cap.add_argument("--run-ckpt", required=True)
    cap.add_argument("--combat-ckpt", required=True)
    cap.add_argument("--procs", type=int, required=True)
    cap.add_argument("--per-cell", type=int, default=P.DEVBOSS_STATES_PER_CELL)
    cap.add_argument("--batch", type=int, default=200, help="seeds per pool round")
    cap.add_argument("--hang-timeout", type=float, default=E.DEFAULT_HANG_TIMEOUT_S)
    pl = sub.add_parser("play", help="play every pair with one agent")
    pl.add_argument("--states", required=True)
    pl.add_argument("--agent", required=True, choices=("mcts", "policy", "student"))
    pl.add_argument("--out", required=True)
    pl.add_argument("--procs", type=int, required=True)
    pl.add_argument("--sims", type=int, default=REF_SIMS)
    pl.add_argument("--boss-sims", type=int, default=REF_BOSS_SIMS)
    pl.add_argument("--combat-ckpt", default=None)
    pl.add_argument("--policy", default=None, help="first | random | MODULE:FUNC")
    pl.add_argument("--ckpt", default=None)
    pl.add_argument("--device", default="cpu")
    pl.add_argument(
        "--segment-env",
        default="ppo.e2e.devboss:e2e_segment_env",
        help="raw | MODULE:FUNC, called FUNC(seed, rm, cell, reseed=, move=)",
    )
    pl.add_argument("--limit-states", type=int, default=None, help="only each cell's first N")
    pl.add_argument("--hang-timeout", type=float, default=E.DEFAULT_HANG_TIMEOUT_S)
    rep = sub.add_parser("report", help="one agent's numbers, or Final - REF")
    rep.add_argument("final")
    rep.add_argument("ref", nargs="?")
    rep.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        return {"capture": capture_cmd, "play": play_cmd, "report": report_cmd}[args.cmd](args)
    except (DevbossError, E.EvalError) as e:
        print(f"devboss: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
