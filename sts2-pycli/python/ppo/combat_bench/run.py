"""`python -m ppo.combat_bench run`: play one agent spec on one fight set (GH #830).

The output is a played file (record shapes in `SCHEMA.md`): a header line, then
one line per CELL, a (fight, rep) pair, written the moment it finishes.

    python -m ppo.combat_bench run --set data/combat_bench/ironclad_pro_v1.jsonl \\
        --out played/pro_deployed.jsonl --reps 2
    python -m ppo.combat_bench run --set ... --out ... --agent mcts --sims 2000 \\
        --rollout-net net.bin --rollout-temp 0.5
    python -m ppo.combat_bench run --set ... --out ... --agent mypkg.agents:make \\
        --agent-arg depth=3

Pairing. A cell's seed is `_fight_seed(key) * 100 + rep`, the formula
`pilot_human_decks` uses, so every arm plays the same seeds on the same fight
and a rep id means the same thing in every file. The rebuilt fight, the
`RunManager` and the agent all take that seed.

Order. All cells are created up front, shuffled with `--order-seed`, cut into
blocks of `--block-size`, and started in that order, one process per cell and
at most `--procs` at a time, so blocks complete in near-seeded order and no
worker idles on a block's slowest fight. Block membership is fixed by the
seeded order, so a reader that keeps only complete blocks reads a random subset
of cells. Keeping
first-finished cells instead would favour short fights, and fight length
depends on the outcome.

Ends. Only `won` and `died` are fight outcomes. The deterministic caps
(`--max-rounds`, `--max-steps`, `--max-sims-per-fight`) end a fight as
`round_cap` / `step_cap` / `sims_cap`. `no_action` means the agent returned no
move while combat was still on, and `error` means the harness failed (the repr
is kept). `watchdog` (`--watchdog-s`) is wall-clock and therefore depends on
the machine: a SIGALRM in the cell's process records it cleanly between steps,
and the parent kills a cell still running `HARD_KILL_GRACE_S` later (a single
engine search can hang where no Python bytecode runs). A cell process that dies
outright (segfault, OOM kill) is recorded as an `error` cell. A spec that cannot search at all (the engine
panics on an unknown `STS2_RUNAI_MCTS_OVERRIDES` knob when the agent is built,
or on an unparsable `--rollout-net` or `STS2_RUNAI_LEAF=learned` without a
net at the first search) would make every cell an `error`, so the runner
first plays one real decision of the first pending cell in a worker and
refuses the run with exit code 2, before it writes anything. A first block
that ends all in `error` aborts the run the same way.

Retries. Resume skips every finished cell. `--retry-ends watchdog,error`
replays the cells whose record ended that way and appends the fresh record;
`report` keeps the last record of a retried cell. `watchdog_s` is not part of
the header identity, so a retry may raise it.

Cost. Each cell records the 1-minute load average when it started (`load1`)
and its host: `cpu_s` is `process_time`, which contention inflates, and
`report` uses both to decide whether CPU ratios between arms mean anything.
The sims cap defaults to `SIMS_CAP_DECISIONS` full-budget decisions of the
spec (`400 * max(sims, boss_sims)`), not an absolute number, so a
high-budget arm's long boss fights are not truncated where a cheap arm's
are played out.

HP. `dmg_c` sums `hp_before - hp_after` over the player's damage rows, the C#
`DamageTaken` definition. The engine writes some HP loss without a row,
though: every `Effect::DealHpLoss` on the player (Crimson Mantle, Inferno, ...)
takes `hooks::apply_dealhploss`'s direct path. So the runner also reads the
player's exact HP at every step boundary and walks those checkpoints together
with the damage rows in time order. A drop no row explains is counted in
`dmg_unlogged`, and a rise is a heal. `hp_end_combat` needs one more step. The
engine applies the victory heals (Burning Blood +6, Meat on the Bone +12 at
50% HP or less, Black Blood +12) inside the combat state when the last enemy
dies, and it logs no heal rows. So the HP of the finished fight is
post-victory, and the runner inverts those three listeners from the last exact
checkpoint (see `hp_before_victory`).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import signal
import socket
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any, ClassVar

from ppo.combat_bench import agents
from ppo.pilot_human_decks import _fight_seed, fight_character

SCHEMA_ID = "combat_bench.played.v1"
# Seeds are `fight_seed * SEED_STRIDE + rep` (`pilot_human_decks` uses 100).
SEED_STRIDE = 100
# Bench rep ids stay in [0, MAX_REPS). `rollout_learn gen` seeds its imitation
# data with `fight_seed * 100 + 50 + rep` (`--rep0` 50) and the RL start pool
# with `+ 77`, so reps 50-99 would replay the playout net's training states on
# every fight the net trained on.
MAX_REPS = 50
# The parent kills a cell's process this long after `--watchdog-s`, giving the
# in-worker alarm the first chance to record a clean `watchdog` cell.
HARD_KILL_GRACE_S = 60
CHECK_EVERY_S = 5.0
# Header fields a resume may change; every other field must match exactly.
# `watchdog_s` is wall-clock only, so a retry may raise it; `cpus` is the
# host's, like `host`.
HEADER_VOLATILE = frozenset({"argv", "host", "cpus", "watchdog_s"})
# Ends `--retry-ends` may replay: they depend on the machine or the harness,
# not on the agent. The report keeps the last record of a retried cell.
RETRYABLE_ENDS = frozenset({"watchdog", "error"})
# The default `--max-sims-per-fight` for an MCTS spec, in full-budget
# decisions: an absolute cap would truncate a high-budget arm's long boss
# fights (40 decisions at 40k sims) that a 1k arm plays out.
SIMS_CAP_DECISIONS = 400
ROOMS = ("monster", "elite", "boss")
SCORED_ENDS = frozenset({"won", "died"})

# The combat-side victory heals, in the order `relic_effects/mod.rs:131-133`
# registers them: (relic, heal, HP threshold in % of max or None). Each heals
# only a living player (`CombatState::creature_heal`), capped at max HP.
# Meat on the Bone reads HP AFTER Burning Blood has healed.
VICTORY_HEALS: tuple[tuple[str, int, int | None], ...] = (
    ("BURNING_BLOOD", 6, None),
    ("MEAT_ON_THE_BONE", 12, 50),
    ("BLACK_BLOOD", 12, None),
)


@dataclass(frozen=True)
class Caps:
    """Deterministic per-fight limits; hitting one ends the fight unscored."""

    max_rounds: int = 30
    max_steps: int = 3000
    max_sims_per_fight: int = 1_600_000


class WatchdogExpired(Exception):
    """Raised by the worker's SIGALRM handler when a cell overruns `--watchdog-s`."""


def cell_seed(key: str, rep: int) -> int:
    """The seed of cell (`key`, `rep`); identical to `pilot_human_decks.play_fight`.

    Raises:
        ValueError: If `rep` is outside `[0, MAX_REPS)` (reps 50-99 are the
            playout net's training seeds).
    """
    if not 0 <= rep < MAX_REPS:
        raise ValueError(
            f"rep {rep} outside [0, {MAX_REPS}): higher reps reuse rollout_learn's training seeds"
        )
    return _fight_seed(key) * SEED_STRIDE + rep


# ── HP accounting ────────────────────────────────────────────────────────


def after_victory_heals(hp: int, max_hp: int, relics: Iterable[str]) -> int:
    """Player HP after the engine's victory heals fire on `hp`.

    Args:
        hp: HP when the last enemy died.
        max_hp: Max HP at that moment.
        relics: Relic ids the player holds.

    Returns:
        HP after Burning Blood / Meat on the Bone / Black Blood.
    """
    held = set(relics)
    for relic, amount, threshold in VICTORY_HEALS:
        if relic not in held or hp <= 0:
            continue
        if threshold is not None and hp > max_hp * threshold // 100:
            continue
        hp = min(hp + amount, max_hp)
    return hp


def hp_before_victory(
    hp_after: int, max_hp: int, relics: Iterable[str], guess: int
) -> tuple[int, bool, bool]:
    """Invert `after_victory_heals`: the HP a won fight ended on.

    The heals are capped at max HP, so several end HPs can map to the same
    post-heal HP. `guess` breaks the tie. It is the last exact HP checkpoint
    (the final damage row, or the HP before the final step). When it maps onto
    `hp_after`, it is the answer, and that is the normal case. A mismatch means
    HP moved after that checkpoint with no row to show it, for example a heal
    from Feed or Reaper on the killing blow. The nearest consistent HP is
    returned in that case, flagged as inferred, and as ambiguous when more
    than one HP is consistent. With Meat on the Bone the heal map is not
    monotonic (34 and 46 both heal to 52 at 80 max HP), so the nearest one
    can be the wrong one; the caller must not book an ambiguous inverse as
    damage.

    A cap at max HP can also hide an unlogged loss on the killing step (the
    guess still maps onto `hp_after`); nothing in the log can show it, which
    is one reason `dmg_unlogged` is a lower bound.

    Args:
        hp_after: Player HP after combat, victory heals included.
        max_hp: Max HP at the end of combat.
        relics: Relic ids held at the end of combat.
        guess: The last exact HP checkpoint.

    Returns:
        `(hp, inferred, ambiguous)`.
    """
    relics = list(relics)
    if after_victory_heals(guess, max_hp, relics) == hp_after:
        return guess, False, False
    fits = [e for e in range(max_hp + 1) if after_victory_heals(e, max_hp, relics) == hp_after]
    if not fits:
        return hp_after, True, False
    return min(fits, key=lambda e: (abs(e - guess), e)), True, len(fits) > 1


class HpLedger:
    """Exact HP checkpoints in time order: what the damage rows do not explain.

    Two kinds of checkpoint feed it, in time order. A damage row on the player
    carries the HP written on either side of the hit. A step boundary is the
    player's HP read from the live combat. Any move in HP between two
    checkpoints that no row accounts for is unlogged. A drop is counted as
    `unlogged` damage and a rise as `healed`. A heal and an unlogged loss
    inside the same stretch cancel each other, so both totals are lower
    bounds.
    """

    def __init__(self, hp: int) -> None:
        self.cur = hp
        self.unlogged = 0
        self.healed = 0

    def observe(self, hp: int) -> None:
        """A checkpoint: the player's exact HP now."""
        if hp < self.cur:
            self.unlogged += self.cur - hp
        elif hp > self.cur:
            self.healed += hp - self.cur
        self.cur = hp

    def rows(self, rows: Iterable[dict[str, Any]]) -> None:
        """Player damage rows, oldest first."""
        for r in rows:
            self.observe(int(r["hp_before"]))
            self.cur = int(r["hp_after"])


def _is_player_damage(row: dict[str, Any]) -> bool:
    return row["kind"] == "damage" and (row.get("receiver") or {}).get("kind") == "player"


def _searched(rec: dict[str, Any]) -> bool:
    # A forced move still returns a `combat_search` stub, with zero visits.
    search = rec.get("combat_search")
    return bool(search) and sum(a.get("visits", 0) for a in search.get("actions", ())) > 0


# ── one cell ─────────────────────────────────────────────────────────────


def build_run(fight: dict[str, Any], seed: int) -> Any:
    """Rebuild `fight` as a live `RunManager` in combat, with the event log armed.

    Args:
        fight: A fight-set row (SCHEMA).
        seed: The cell seed.

    Returns:
        The `sts2_rs.RunManager`, in phase `combat`, on the selected
        `sts2_client.GAME_VERSION` (v0.107.1 unless `STS2_GAME_VERSION` says
        otherwise).

    Raises:
        sts2_client.GameVersionMismatch: The engine built another version.
    """
    import sts2_client
    import sts2_rs

    ver = sts2_client.GAME_VERSION
    character = fight_character(fight)
    state = sts2_rs.build_scenario_state(
        seed,
        character,
        list(fight["deck"]),
        int(fight["hp_before"]),
        int(fight["max_hp"]),
        list(fight["relics"]),
        fight["encounter"],
        ascension=int(fight["ascension"]),
        potions=list(fight["potions"]),
        monsters=fight.get("monsters"),
        # The human's belt size (A10 TightBelt, slot relics); a set without
        # the field keeps the harness's 3 slots.
        potion_slots=fight.get("potion_slots"),
        game_version=ver,
    )
    rm = sts2_rs.RunManager(
        seed=seed, character=character, ascension_level=int(fight["ascension"]), game_version=ver
    )
    # `debug_enter_combat` takes any state without comparing builds.
    sts2_client.check_game_version(state, ver, what="combat_bench combat state")
    sts2_client.check_game_version(rm, ver, what="combat_bench RunManager")
    rm.debug_enter_combat(state, fight["room_type"])
    # Display-side log only: search never sees it (`MctsTree::new` disarms
    # its root clone), so arming it cannot change the play.
    rm.set_record_combat_log(True)
    return rm


def _in_combat(rm: Any) -> bool:
    return rm.phase() == "combat" and not rm.is_over()


def _view(rm: Any) -> dict[str, Any] | None:
    """The live combat as a state dict, or None outside combat.

    `state_to_dict`, not `CombatState.get_state()`: the bridge view turns into
    a bare `card_select` payload while a choice is open, with no player in it.
    """
    import sts2_rs

    cs = rm.combat_state()
    return None if cs is None else sts2_rs.state_to_dict(cs)


def play(rm: Any, agent: agents.CombatAgent, fight: dict[str, Any], caps: Caps) -> dict[str, Any]:
    """Drive `agent` on the live fight `rm` to an end; the rep record minus ids.

    A `WatchdogExpired` raised while the fight is running ends it as
    `watchdog`, and the record keeps what was measured up to that point.

    Args:
        rm: A `RunManager` from `build_run`.
        agent: The agent for this cell.
        fight: The fight-set row, read for `hp_before` and the given potions.
        caps: The deterministic caps.

    Returns:
        The SCHEMA rep fields from `end` on (no `rep` / `seed` / `cpu_s`).
    """
    ledger = HpLedger(int(fight["hp_before"]))
    steps = decisions = seen = 0
    filt = RootFilterLog()
    end: str | None = None
    spent = agents.sims_spent(agent)
    # The scenario builder already ran combat start (Blood Vial and the like)
    # with the log disarmed; this checkpoint books what that did to HP.
    view = _view(rm)
    assert view is not None, "build_run left the run outside combat"
    ledger.observe(int(view["player"]["hp"]))
    try:
        while _in_combat(rm):
            if int(view["round"]) > caps.max_rounds:
                end = "round_cap"
                break
            if steps >= caps.max_steps:
                end = "step_cap"
                break
            if spent is not None and spent >= caps.max_sims_per_fight:
                end = "sims_cap"
                break
            t0 = time.perf_counter()
            rec = agent.step(rm)
            filt.step(rec, time.perf_counter() - t0)
            if rec is None:
                end = "no_action"
                break
            steps += 1
            decisions += _searched(rec)
            spent = agents.sims_spent(agent)
            log = rm.combat_log()
            ledger.rows(r for r in log[seen:] if _is_player_damage(r))
            seen = len(log)
            if _in_combat(rm):
                view = _view(rm)
                assert view is not None, "in combat without a combat state"
                ledger.observe(int(view["player"]["hp"]))
    except WatchdogExpired:
        end = "watchdog"

    log = rm.combat_log()
    ledger.rows(r for r in log[seen:] if _is_player_damage(r))
    final = _view(rm)
    inferred = ambiguous = False
    if end is None:
        end = "died" if rm.is_over() and not rm.player_won else "won"
    if end == "died":
        hp_end = 0
        ledger.observe(int(final["player"]["hp"]) if final else 0)
    elif end == "won" and final is not None:
        hp_end, inferred, ambiguous = hp_before_victory(
            int(final["player"]["hp"]),
            int(final["player"]["max_hp"]),
            [r["id"] for r in final["player"].get("relics", [])],
            ledger.cur,
        )
        if not ambiguous:
            # An ambiguous inverse is a guess: booking it could invent up to
            # 12 HP of unlogged damage (or hide a heal).
            ledger.observe(hp_end)
    elif end == "won":
        hp_end = int(rm.current_hp)
    else:
        hp_end = int(final["player"]["hp"]) if final else ledger.cur
        ledger.observe(hp_end)
    ended = end in SCORED_ENDS
    # Off the run: the run-side HP / max HP sync only when combat exits.
    hp_after = int(rm.current_hp) if ended or final is None else hp_end
    max_hp_after = int(rm.max_hp) if ended or final is None else int(final["player"]["max_hp"])

    damage = [r for r in log if _is_player_damage(r)]
    drunk = collections.Counter(r.get("potion") for r in log if r["kind"] == "potion")
    turns = max((int(r.get("round", 0)) for r in log), default=0)
    out: dict[str, Any] = {
        "end": end,
        "died": end == "died",
        "hp_end_combat": hp_end,
        "hp_after": hp_after,
        "max_hp_after": max_hp_after,
        "dmg": sum(int(r.get("unblocked", 0)) for r in damage),
        "dmg_c": sum(int(r["hp_before"]) - int(r["hp_after"]) for r in damage),
        "dmg_unlogged": ledger.unlogged,
        "turns": turns or (int(final["round"]) if final else 0),
        "steps": steps,
        "decisions": decisions,
        "sims_spent": spent,
        "potions_left": sum((collections.Counter(fight["potions"]) - drunk).values()),
        "drunk": sum(drunk.values()),
    }
    if rm.combat_log_dropped():
        out["log_dropped"] = int(rm.combat_log_dropped())
    if inferred:
        out["hp_end_inferred"] = True
    if ambiguous:
        out["hp_end_ambiguous"] = True
    reuse = agents.tree_reuse_stats(agent)
    if reuse is not None:
        out["tree_reuse"] = reuse
    if filt.rows:
        out["root_filter"] = filt.summary()
    return out


class RootFilterLog:
    """The turn-plan root filter's per-decision records of one cell (GH #830).

    `MctsCombatAgent.step` carries `root_filter` only when the spec sets
    `STS2_RUNAI_TURNPLAN_ROOT_FILTER`; a cell whose agent never reports one
    gets no `root_filter` field, so records of other arms keep their shape.
    """

    FALLBACK: ClassVar[dict[str | None, str]] = {None: "", "forced": "f", "cap": "c", "empty": "e"}
    VIA: ClassVar[dict[str | None, str]] = {None: "", "topk": "t", "anchor_only": "a"}

    def __init__(self) -> None:
        self.rows: list[list[Any]] = []

    def step(self, rec: dict[str, Any] | None, wall_s: float) -> None:
        f = (rec or {}).get("root_filter")
        if f is None:
            return
        self.rows.append(
            [
                int(f["applied"]),
                self.FALLBACK[f["fallback"]],
                int(f["legal_n"]),
                int(f["plans_n"]),
                int(f["kept_n"]),
                int(f["allowed_n"]),
                self.VIA[f["chosen_via"]],
                round(int(f["filter_us"]) / 1000.0, 2),
                round(wall_s * 1000.0, 1),
            ]
        )

    def summary(self) -> dict[str, Any]:
        """Counts over the cell plus the per-decision rows (SCHEMA `root_filter`)."""
        rows = self.rows
        fb = collections.Counter(r[1] for r in rows if r[1])
        via = collections.Counter(r[6] for r in rows if r[6])
        return {
            "decisions": len(rows),
            "applied": sum(r[0] for r in rows),
            "fallback_forced": fb["f"],
            "fallback_cap": fb["c"],
            "fallback_empty": fb["e"],
            "chosen_topk": via["t"],
            "chosen_anchor_only": via["a"],
            "legal_n": sum(r[2] for r in rows if r[0]),
            "allowed_n": sum(r[5] for r in rows if r[0]),
            "filter_ms": round(sum(r[7] for r in rows), 1),
            "decide_ms": round(sum(r[8] for r in rows), 1),
            "rows": rows,
        }


def machine() -> dict[str, Any]:
    """Where and under what load a cell ran: `load1` (1-minute load average) and `host`."""
    try:
        load1: float | None = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):  # Windows has no os.getloadavg
        load1 = None
    return {"load1": load1, "host": socket.gethostname()}


def play_cell(fight: dict[str, Any], rep: int, spec: dict[str, Any], caps: Caps) -> dict[str, Any]:
    """Play cell (`fight`, `rep`) in this process: the SCHEMA rep record.

    The process env must already be the spec's (`agents.apply_env`); the Pool
    initializer does that for workers.

    Args:
        fight: A fight-set row.
        rep: The rep id (`rep_offset + i`).
        spec: The agent spec.
        caps: The deterministic caps.

    Returns:
        The rep record. A harness failure is `end == "error"` with the repr in
        `error`.
    """
    t0 = time.process_time()
    seed = cell_seed(fight["key"], rep)
    out: dict[str, Any] = {"rep": rep, "seed": seed, **machine()}
    try:
        rm = build_run(fight, seed)
        agent = agents.make_agent(spec, seed)
        out |= play(rm, agent, fight, caps)
    except WatchdogExpired:
        out |= {"end": "watchdog", "died": False}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a PyO3 panic is a BaseException, not an Exception
        out |= {"end": "error", "died": False, "error": repr(e)}
    out["cpu_s"] = round(time.process_time() - t0, 4)
    return _ordered(out)


# SCHEMA field order, so files diff and read the same way.
_REP_ORDER = (
    "rep",
    "seed",
    "end",
    "died",
    "hp_end_combat",
    "hp_after",
    "max_hp_after",
    "dmg",
    "dmg_c",
    "dmg_unlogged",
    "turns",
    "steps",
    "decisions",
    "sims_spent",
    "cpu_s",
    "potions_left",
    "drunk",
    "log_dropped",
    "hp_end_inferred",
    "hp_end_ambiguous",
    "load1",
    "host",
    "error",
)


def _ordered(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in _REP_ORDER if k in rec} | {
        k: v for k, v in rec.items() if k not in _REP_ORDER
    }


# ── workers ──────────────────────────────────────────────────────────────

_WORKER: dict[str, Any] = {}


def _on_alarm(signum: int, frame: Any) -> None:
    if _WORKER.get("armed"):
        raise WatchdogExpired()


def _worker_init(spec: dict[str, Any], caps: dict[str, int], watchdog_s: int) -> None:
    # Before any `sts2_rs` search: some seams are latched on first use.
    agents.apply_env(spec["env"])
    _WORKER.update(spec=spec, caps=Caps(**caps), watchdog_s=watchdog_s, armed=False)
    if watchdog_s > 0:
        signal.signal(signal.SIGALRM, _on_alarm)


def _worker_probe(fight: dict[str, Any], rep: int) -> str | None:
    """Play one real decision of cell (`fight`, `rep`); the error repr, or None.

    Building the agent is not enough. The engine parses
    `STS2_RUNAI_MCTS_OVERRIDES` when an agent is built, but it loads
    `STS2_ROLLOUT_NET` and checks `STS2_RUNAI_LEAF=learned` for its net only at
    the first search, and a factory can fail in `step`. Any of them would turn
    every cell of the run into an `error` record, so the runner searches once
    before it writes anything. A fight that does not build is a cell failure,
    not a spec failure, so then only the agent is built.
    """
    try:
        seed = cell_seed(fight["key"], rep)
        try:
            rm = build_run(fight, seed)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # the fight's own problem; its cell records it
            rm = None
        agent = agents.make_agent(_WORKER["spec"], seed)
        if rm is not None:
            agent.step(rm)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a PyO3 panic is a BaseException, not an Exception
        return repr(e)
    return None


def _worker_play(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    fight, rep = job
    watchdog_s = _WORKER["watchdog_s"]
    if watchdog_s > 0:
        _WORKER["armed"] = True
        signal.alarm(watchdog_s)
    try:
        rec = play_cell(fight, rep, _WORKER["spec"], _WORKER["caps"])
    except WatchdogExpired:
        rec = _ordered(
            {
                "rep": rep,
                "seed": cell_seed(fight["key"], rep),
                "end": "watchdog",
                "died": False,
                **machine(),
            }
        )
    finally:
        _WORKER["armed"] = False
        if watchdog_s > 0:
            signal.alarm(0)
    return {"key": fight["key"], "rep": rec}


def _failed_cell(fight: dict[str, Any], rep: int, end: str, error: str) -> dict[str, Any]:
    rec = {"rep": rep, "seed": cell_seed(fight["key"], rep), "end": end, "died": False}
    return {"key": fight["key"], "rep": _ordered(rec | machine() | {"error": error})}


def _cell_process(
    spec: dict[str, Any],
    caps: dict[str, int],
    watchdog_s: int,
    fight: dict[str, Any],
    rep: int,
    conn: Any,
    probe: bool = False,
) -> None:
    """One cell (or the probe) in its own process; the result goes down `conn`."""
    _worker_init(spec, caps, watchdog_s)
    try:
        out: Any = _worker_probe(fight, rep) if probe else _worker_play((fight, rep))
    except BaseException as e:  # a PyO3 panic is a BaseException
        out = repr(e) if probe else _failed_cell(fight, rep, "error", repr(e))
    conn.send(out)
    conn.close()


class _CellScheduler:
    """Runs each cell in its own spawned process, at most `procs` at a time.

    A process per cell, not a worker pool, because the wall-clock watchdog
    must be able to stop a cell that is stuck INSIDE one engine call. The
    in-worker SIGALRM only fires between Python bytecodes, so a single search
    that never returns (seen on Waterfall Giant at 4000 sims: one decision ran
    for over an hour) held its block forever, and with it every cell queued
    behind it. Here the parent kills a cell's process after `hard_s` and
    records it as `watchdog`; a process that dies without reporting (segfault,
    OOM kill, a panic under panic=abort) becomes an `error` cell instead of
    aborting the run. Startup costs ~0.1 s per cell, against 1-100 s cells.
    """

    def __init__(self, spec: dict[str, Any], caps: Caps, watchdog_s: int, procs: int) -> None:
        self.args = (spec, asdict(caps), watchdog_s)
        self.procs = max(1, procs)
        # The in-worker alarm records a clean `watchdog` cell with its partial
        # measurements when it can; the parent's kill is the backstop.
        self.hard_s = watchdog_s + HARD_KILL_GRACE_S if watchdog_s > 0 else 0
        self.ctx = get_context("spawn")
        self.running: dict[Any, tuple[Any, float, Any]] = {}

    def _start(self, fight: dict[str, Any], rep: int, tag: Any, probe: bool = False) -> None:
        recv, send = self.ctx.Pipe(duplex=False)
        proc = self.ctx.Process(
            target=_cell_process, args=(*self.args, fight, rep, send, probe), daemon=True
        )
        proc.start()
        send.close()
        self.running[recv] = (proc, time.monotonic(), (tag, fight, rep))

    def probe(self, fight: dict[str, Any], rep: int) -> str | None:
        self._start(fight, rep, None, probe=True)
        (recv,) = self.running
        proc, _, _ = self.running.pop(recv)
        try:
            out = recv.recv()
        except EOFError:
            out = f"worker died during the probe search (exit code {proc.exitcode})"
        proc.join()
        return out

    def run(self, cells: Iterable[tuple[Any, dict[str, Any], int]]) -> Iterator[tuple[Any, dict]]:
        """Yield `(tag, result)` per cell as it finishes, starting cells in order."""
        todo = iter(cells)
        exhausted = False
        while True:
            while not exhausted and len(self.running) < self.procs:
                nxt = next(todo, None)
                if nxt is None:
                    exhausted = True
                    break
                tag, fight, rep = nxt
                self._start(fight, rep, tag)
            if not self.running:
                return
            for recv in wait_conns(list(self.running), timeout=CHECK_EVERY_S):
                proc, _, (tag, fight, rep) = self.running.pop(recv)
                try:
                    out = recv.recv()
                except EOFError:
                    proc.join()
                    out = _failed_cell(
                        fight, rep, "error", f"cell process died (exit code {proc.exitcode})"
                    )
                recv.close()
                proc.join()
                yield tag, out
            if self.hard_s:
                now = time.monotonic()
                for recv, (proc, t0, (tag, fight, rep)) in list(self.running.items()):
                    if now - t0 > self.hard_s:
                        proc.kill()
                        proc.join()
                        recv.close()
                        del self.running[recv]
                        yield (
                            tag,
                            _failed_cell(
                                fight, rep, "watchdog", f"killed after {self.hard_s} s wall clock"
                            ),
                        )

    def close(self) -> None:
        for recv, (proc, _, _) in list(self.running.items()):
            proc.kill()
            proc.join()
            recv.close()
        self.running.clear()


# ── set, cells, header, resume ───────────────────────────────────────────


def load_set(path: str) -> list[dict[str, Any]]:
    """Read a fight set, refusing duplicate keys (the key is the pairing id).

    Raises:
        ValueError: On a duplicate key.
    """
    fights = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    seen: set[str] = set()
    for f in fights:
        if f["key"] in seen:
            raise ValueError(f"{path}: duplicate key {f['key']!r}")
        seen.add(f["key"])
    return fights


def read_keys(path: str) -> list[str]:
    """Keys from a JSON list or a file of one key per line."""
    text = open(path, encoding="utf-8").read()
    try:
        keys = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(keys, list):
        raise ValueError(f"{path}: expected a JSON list of keys")
    return [str(k) for k in keys]


def select_fights(
    fights: list[dict[str, Any]],
    keys: Iterable[str] | None = None,
    rooms: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Subset the set, keeping set order: keys, then room types, then the first `limit`.

    Raises:
        ValueError: If `keys` names a key the set does not have.
    """
    if keys is not None:
        want = set(keys)
        missing = want - {f["key"] for f in fights}
        if missing:
            raise ValueError(f"{len(missing)} selected keys not in the set, e.g. {min(missing)!r}")
        fights = [f for f in fights if f["key"] in want]
    if rooms is not None:
        allowed = set(rooms)
        fights = [f for f in fights if f["room_type"] in allowed]
    if limit is not None:
        fights = fights[:limit]
    return fights


def make_cells(
    keys: list[str], reps: int, rep_offset: int, order_seed: int, block_size: int
) -> list[tuple[int, str, int]]:
    """Every (block, key, rep) cell, in the seeded submission order.

    The shuffle runs over the FULL cell list before a resume drops finished
    cells, so a cell's block index never depends on what has already run.
    """
    cells = [(k, rep_offset + i) for k in keys for i in range(reps)]
    random.Random(order_seed).shuffle(cells)
    return [(j // block_size, k, r) for j, (k, r) in enumerate(cells)]


def slowest_first(cells: list[tuple[int, str, int]], hint_path: str) -> list[tuple[int, str, int]]:
    """Start order longest-first by a prior played file's `cpu_s` (LPT scheduling).

    In seeded order a run waits on whichever slow cell happened to start last:
    one cell the watchdog kills at 21 min idled 43 of 44 processes for ~17 min
    per arm. Started first, it overlaps the rest. A cell the prior file did not
    score (watchdog, error) counts as slowest; one it lacks takes its fight's
    mean, else the overall median. Blocks keep their seeded membership, so
    this changes only when a cell starts, never what it plays; but complete
    blocks no longer arrive in seeded order, so an interim read is biased.
    The hint file is only read (a cut last line is skipped), and a cell written
    twice keeps its later line, as a retry appends it.
    """
    cost: dict[tuple[str, int], float] = {}
    for line in open(hint_path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "header" in rec:
            continue
        rep = rec["rep"]
        scored = rep["end"] in ("won", "died") and rep.get("cpu_s") is not None
        cost[rec["key"], rep["rep"]] = rep["cpu_s"] if scored else float("inf")
    finite = sorted(c for c in cost.values() if c != float("inf"))
    median = finite[len(finite) // 2] if finite else 0.0
    by_key: dict[str, list[float]] = collections.defaultdict(list)
    for (k, _), c in cost.items():
        by_key[k].append(c)

    def guess(cell: tuple[int, str, int]) -> float:
        _, k, r = cell
        if (k, r) in cost:
            return cost[k, r]
        if k in by_key:
            return sum(by_key[k]) / len(by_key[k])
        return median

    return sorted(cells, key=guess, reverse=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_header(
    *,
    set_path: str,
    spec: dict[str, Any],
    reps: int,
    rep_offset: int,
    caps: Caps,
    watchdog_s: int,
    block_size: int,
    order_seed: int,
    select: dict[str, Any],
    argv: list[str],
) -> dict[str, Any]:
    """The played file's header (SCHEMA), JSON-normalised for comparison."""
    import sts2_rs

    info = sts2_rs.build_info() if hasattr(sts2_rs, "build_info") else {}
    with open(set_path, "rb") as f:
        set_sha = _sha256(f.read())
    header = {
        "schema": SCHEMA_ID,
        "set_name": Path(set_path).stem,
        "set_sha256": set_sha,
        "spec": spec,
        "src_hash": sts2_rs.src_hash(),
        "build_git": info.get("build_git"),
        "pgo": bool(info.get("pgo")),
        "reps": reps,
        "rep_offset": rep_offset,
        "caps": asdict(caps),
        "watchdog_s": watchdog_s,
        "block_size": block_size,
        "order_seed": order_seed,
        "select": select,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "cpus": os.cpu_count(),
        "argv": argv,
    }
    return json.loads(json.dumps(header))


def header_mismatch(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Header fields that differ, ignoring `HEADER_VOLATILE`."""
    return sorted(k for k in (set(old) | set(new)) - HEADER_VOLATILE if old.get(k) != new.get(k))


def read_played(path: str) -> tuple[dict[str, Any] | None, dict[tuple[str, int], str]]:
    """Header and the end of every finished (key, rep) cell of a played file.

    A line cut short by a killed run is truncated away first. Leaving it would
    glue the next appended line onto it. A cell written twice keeps its first
    end unless that end is retryable (`RETRYABLE_ENDS`), as `report` reads it.

    Returns:
        `(header or None for an empty file, {(key, rep): end})`.
    """
    with open(path, "rb") as f:
        data = f.read()
    cut = data.rfind(b"\n") + 1
    if cut != len(data):
        print(
            f"[combat_bench.run] {path}: dropping a truncated last line ({len(data) - cut} bytes)",
            file=sys.stderr,
        )
        os.truncate(path, cut)
        data = data[:cut]
    lines = data.splitlines()
    if not lines:
        return None, {}
    header = json.loads(lines[0])["header"]
    ends: dict[tuple[str, int], str] = {}
    for line in lines[1:]:
        row = json.loads(line)
        cell = (row["key"], int(row["rep"]["rep"]))
        if cell not in ends or ends[cell] in RETRYABLE_ENDS:
            ends[cell] = row["rep"].get("end")
    return header, ends


# ── main ─────────────────────────────────────────────────────────────────


def _fmt_s(s: float) -> str:
    if s != s or s == float("inf"):
        return "?"
    h, rem = divmod(int(s), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


class _Progress:
    """A progress line every `every` cells, to stderr and an optional log file."""

    def __init__(self, total: int, already: int, path: str | None, every: int = 50) -> None:
        self.total, self.done0, self.path, self.every = total, already, path, every
        self.n = 0
        self.t0 = time.time()
        self.ends: collections.Counter[str] = collections.Counter()

    def emit(self, msg: str) -> None:
        line = f"[combat_bench.run] {msg}"
        print(line, file=sys.stderr, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def cell(self, end: str, block: int, n_blocks: int, force: bool = False) -> None:
        self.n += 1
        self.ends[end] += 1
        if force or self.n % self.every == 0:
            dt = time.time() - self.t0
            rate = self.n / dt if dt > 0 else float("nan")
            left = self.total - self.done0 - self.n
            self.emit(
                f"{self.done0 + self.n}/{self.total} cells, block {block + 1}/{n_blocks}, "
                f"{rate:.3f} cells/s, ETA {_fmt_s(left / rate if rate > 0 else float('nan'))}, "
                f"ends {dict(sorted(self.ends.items()))}"
            )


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.combat_bench run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", required=True, help="fight-set JSONL (SCHEMA)")
    ap.add_argument("--out", required=True, help="played JSONL; resumed when it exists")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=64, help="cells per submission block")
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument(
        "--slow-first",
        default=None,
        metavar="PLAYED",
        help="start cells longest-first by this played file's cpu_s (same set); "
        "shortens the tail, but interim reads are no longer unbiased",
    )
    ap.add_argument("--keys-file", default=None, help="only these keys (JSON list or lines)")
    ap.add_argument("--room", default=None, help="comma list of room types to keep")
    ap.add_argument("--limit", type=int, default=None, help="only the first N fights")
    g = ap.add_argument_group("agent")
    g.add_argument("--agent", default="mcts", help="'mcts' or 'pkg.module:factory'")
    g.add_argument("--agent-arg", action="append", default=[], help="factory kwarg k=v")
    g.add_argument("--sims", type=int, default=None, help="mcts: sims per decision (1000)")
    g.add_argument("--boss-sims", type=int, default=None, help="mcts: boss sims (4000)")
    g.add_argument("--env", action="append", default=[], help="worker env K=V (repeatable)")
    g.add_argument("--overrides", default=None, help="STS2_RUNAI_MCTS_OVERRIDES")
    g.add_argument("--rollout-net", default=None, help="STS2_ROLLOUT_NET")
    g.add_argument("--rollout-temp", type=float, default=None, help="STS2_ROLLOUT_TEMP")
    g.add_argument(
        "--clair",
        action="store_true",
        help="clairvoyant (STS2_RUNAI_HONEST=0). NOT the deployed pilot plus perfect "
        "information: the engine applies the deployed search dialect only under BeliefDraw, "
        "so this is the legacy clairvoyant search (spec search_dialect 'legacy')",
    )
    c = ap.add_argument_group("caps")
    c.add_argument("--max-rounds", type=int, default=Caps.max_rounds)
    c.add_argument("--max-steps", type=int, default=Caps.max_steps)
    c.add_argument(
        "--max-sims-per-fight",
        type=int,
        default=None,
        help=f"default: {SIMS_CAP_DECISIONS} x max(sims, boss_sims) for mcts, "
        f"{Caps.max_sims_per_fight} for a factory",
    )
    c.add_argument("--watchdog-s", type=int, default=3600, help="wall clock per cell; 0 = off")
    ap.add_argument(
        "--retry-ends",
        default=None,
        help=f"comma list of ends to replay on resume ({','.join(sorted(RETRYABLE_ENDS))})",
    )
    ap.add_argument("--progress-log", default=None, help="append progress lines here too")
    return ap.parse_args(argv)


def _spec(args: argparse.Namespace) -> dict[str, Any]:
    env = agents.build_env(
        agents.parse_kv(args.env, "--env"),
        overrides=args.overrides,
        rollout_net=args.rollout_net,
        rollout_temp=args.rollout_temp,
        clair=args.clair,
    )
    if args.agent == "mcts":
        if args.agent_arg:
            raise ValueError("--agent-arg is for a factory agent, not --agent mcts")
        sims = 1000 if args.sims is None else args.sims
        boss_sims = 4000 if args.boss_sims is None else args.boss_sims
        return agents.mcts_spec(sims, boss_sims, env)
    if args.sims is not None or args.boss_sims is not None:
        raise ValueError("--sims / --boss-sims configure --agent mcts; pass --agent-arg instead")
    return agents.factory_spec(args.agent, args.agent_arg, env)


def main(argv: list[str] | None = None) -> int:
    """`python -m ppo.combat_bench run ...`; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse(argv)
    try:
        if args.reps < 1 or args.rep_offset < 0 or args.rep_offset + args.reps > MAX_REPS:
            raise ValueError(
                f"need reps >= 1 and rep_offset + reps <= {MAX_REPS} (seeds are fight_seed * "
                f"{SEED_STRIDE} + rep, and rollout_learn trains on reps 50 and up); got "
                f"{args.rep_offset} + {args.reps}"
            )
        retry = set(args.retry_ends.split(",")) if args.retry_ends else set()
        if retry - RETRYABLE_ENDS:
            raise ValueError(
                f"--retry-ends takes {','.join(sorted(RETRYABLE_ENDS))}, got {args.retry_ends!r}"
            )
        if args.block_size < 1 or args.procs < 1:
            raise ValueError("--block-size and --procs must be >= 1")
        spec = _spec(args)
        rooms = args.room.split(",") if args.room else None
        if rooms and set(rooms) - set(ROOMS):
            raise ValueError(f"--room takes {','.join(ROOMS)}, got {args.room!r}")
        keys = read_keys(args.keys_file) if args.keys_file else None
        fights = select_fights(load_set(args.set), keys, rooms, args.limit)
    except ValueError as e:
        print(f"[combat_bench.run] {e}", file=sys.stderr)
        return 2
    if not fights:
        print("[combat_bench.run] no fights selected", file=sys.stderr)
        return 2

    sims_cap = args.max_sims_per_fight
    if sims_cap is None:
        sims_cap = (
            SIMS_CAP_DECISIONS * max(spec["sims"], spec["boss_sims"])
            if spec["kind"] == "mcts"
            else Caps.max_sims_per_fight
        )
    caps = Caps(args.max_rounds, args.max_steps, sims_cap)
    select = {
        "room": rooms,
        "limit": args.limit,
        "keys_file_sha256": _sha256(open(args.keys_file, "rb").read()) if keys else None,
        # The report needs the planned fights to count missing cells, and a
        # keys file is not part of the played file; so the keys it selected
        # travel in the header (room / limit alone are re-applied on the set).
        "keys": [f["key"] for f in fights] if keys is not None else None,
        "n_fights": len(fights),
        "fights_sha256": _sha256("\n".join(f["key"] for f in fights).encode()),
    }
    header = make_header(
        set_path=args.set,
        spec=spec,
        reps=args.reps,
        rep_offset=args.rep_offset,
        caps=caps,
        watchdog_s=args.watchdog_s,
        block_size=args.block_size,
        order_seed=args.order_seed,
        select=select,
        argv=argv,
    )
    done: set[tuple[str, int]] = set()
    if os.path.exists(args.out):
        old, ends = read_played(args.out)
        done = {c for c, e in ends.items() if e not in retry}
        if old is not None:
            diff = header_mismatch(old, header)
            if diff:
                print(
                    f"[combat_bench.run] refusing to resume {args.out}: the header differs in "
                    f"{diff}. Same set, spec, build, caps and order are required; write a new "
                    "--out instead.",
                    file=sys.stderr,
                )
                for k in diff:
                    print(
                        f"  {k}: file={old.get(k)!r}\n  {k}:  now={header.get(k)!r}",
                        file=sys.stderr,
                    )
                return 2
        else:
            done = set()
    inherited = sorted(k for k in os.environ if agents.is_engine_env(k) and k not in spec["env"])
    if inherited:
        print(
            f"[combat_bench.run] WARNING: ignoring inherited engine env {inherited}; "
            "pass it with --env to make it part of the spec",
            file=sys.stderr,
        )

    by_key = {f["key"]: f for f in fights}
    cells = make_cells(list(by_key), args.reps, args.rep_offset, args.order_seed, args.block_size)
    n_blocks = cells[-1][0] + 1
    pending = [c for c in cells if (c[1], c[2]) not in done]
    progress = _Progress(len(cells), len(cells) - len(pending), args.progress_log)
    progress.emit(
        f"{len(fights)} fights x {args.reps} reps = {len(cells)} cells in {n_blocks} blocks; "
        f"{len(cells) - len(pending)} done, {len(pending)} to play"
        + (f" ({len(retry)} end kind(s) retried)" if retry else "")
        + f"; caps {json.dumps(asdict(caps))}; spec {json.dumps(spec)}"
    )
    if not pending:
        return 0
    # The probe takes the seeded first cell, not the slowest: it has no watchdog.
    _, key0, rep0 = pending[0]
    if args.slow_first:
        pending = slowest_first(pending, args.slow_first)
        progress.emit(f"start order: longest first by {args.slow_first}")
    sched = _CellScheduler(spec, caps, args.watchdog_s, min(args.procs, len(pending)))
    try:
        probe = sched.probe(by_key[key0], rep0)
        if probe is not None:
            progress.emit(f"REFUSED: the spec cannot search ({key0} rep {rep0}): {probe}")
            return 2
        with open(args.out, "a", encoding="utf-8") as sink:
            if Path(args.out).stat().st_size == 0:
                sink.write(json.dumps({"header": header}) + "\n")
                sink.flush()
            last = len(pending)
            first_block = pending[0][0]
            first_size = sum(1 for b, _, _ in pending if b == first_block)
            first_ends: list[dict[str, Any]] = []
            # Cells start in seeded block order, so blocks complete in near-seeded
            # order; block membership is fixed by that order, so a reader that
            # keeps complete blocks reads a random subset.
            jobs = ((block, by_key[key], rep) for block, key, rep in pending)
            for block, out in sched.run(jobs):
                sink.write(
                    json.dumps({"key": out["key"], "block": block, "rep": out["rep"]}) + "\n"
                )
                sink.flush()
                progress.cell(out["rep"]["end"], block, n_blocks, force=progress.n + 1 == last)
                if block == first_block:
                    first_ends.append(out["rep"])
                    if len(first_ends) == first_size and all(
                        r["end"] == "error" for r in first_ends
                    ):
                        # The probe passed but every cell failed: a spec or harness
                        # fault, not the fights. Stop before burning the rest.
                        progress.emit(
                            "ABORT: every cell of the first block ended in error, e.g. "
                            f"{first_ends[0].get('error')}. Fix it and rerun with "
                            "--retry-ends error (or a new --out)."
                        )
                        return 2
    finally:
        sched.close()
    progress.emit(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
