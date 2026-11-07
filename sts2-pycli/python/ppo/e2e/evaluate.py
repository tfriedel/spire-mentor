"""Play a policy on an e2e evaluation panel in the runner's layout: `ppo.e2e.evaluate`.

    PYTHONPATH=python <W2 python> -m ppo.e2e.evaluate run ARM PANEL N SEED_START \\
        --policy MODULE:FUNC --ckpt CKPT --procs 16 --pin "$E2E_PIN" [--sealed]
    PYTHONPATH=python <python> -m ppo.e2e.evaluate report ARM PANEL

``run`` is the Final arm's runner (PREREG-0 P4: "TEST runs only through
``scripts/e2e_arm.sh`` or ``ppo.e2e.evaluate --sealed``"). It writes exactly
what ``e2e_arm.sh`` writes, so ``ppo.e2e.score_test`` reads both the same way::

    <evals>/<PANEL>/<ARM>/shards/s<lo>_<hi>.jsonl   250-seed shards, rows in seed order
    <evals>/<PANEL>/<ARM>/retry/s<seed>.jsonl       the one re-run of an errored seed
    <evals>/<PANEL>/<ARM>/meta/config.txt           what produced the rows (refusal key)
    <evals>/<PANEL>/<ARM>/meta/wheel_guard.txt, calls.log, logs/, work/
    <evals>/<PANEL>/<ARM>.sha256                    sha256sum of every shard and retry file

What it guarantees, matching ``e2e_arm.sh`` (plan section 2, P2, P4):

- It re-executes itself under a clean environment: HOME, PATH, LANG, the
  worker thread caps (``ops_contract.WORKER_THREAD_ENV``: ``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` at 1),
  ``PYTHONPATH=<pin>:<this checkout's python/>``, ``STS2_WHEEL_PIN``,
  ``STS2_GAME_VERSION=v0.111.0`` and the arm's own ``--set-env`` pairs, so no
  ``STS2_RUNAI_/ROLLOUT_/MCTS_`` knob leaks in. Its workers start under the
  caps whoever runs the pool, and set torch to one thread.
- It refuses an ``sts2_rs`` import that does not come from the pin, a wheel
  that fails ``ppo.wheel_guard``, a dirty worktree, and an arm directory whose
  ``meta/config.txt`` differs from this call's. The ``--allow-*`` escapes exist
  for DEV and SMOKE only.
- Rows are repeatable: no wall clock in a row, shards written in seed order
  once complete (``work/`` holds the in-flight rows), so a repeat on the same
  binary gives byte-identical shards and manifest.
- Every row, an error row too, carries the writing process's stamps
  (``stamps.STAMP_KEYS``: game version, Python and client source digests,
  wheel, vocab and schema sha256; plan section 3.1, F8), from
  :func:`eval_stamps`. So does every ``ppo.e2e.devboss`` pair row, the other
  student eval rows the TEST read takes.
- An errored seed re-runs once (P2). A run that hangs past ``--hang-timeout``
  is killed and booked as an error, so it gets that re-run too. The policy
  failing to load stops the whole call instead: a broken checkpoint must not
  turn into rows of "missing data".
- ``--sealed`` (required on TEST and SMOKE_SEALED, refused elsewhere) prints
  only row counts, error counts and the manifest's sha256, sends the workers'
  stdout to /dev/null and keeps the panel directory at mode 700.

The two plug-in points (see EVAL.md next to PREREG.md):

- ``--policy MODULE:FUNC``: ``FUNC(ckpt, device=...)`` returns a
  :class:`StudentPolicy`, whose ``act_decisions(decisions)`` gives one merged
  class index per decision (``ppo.e2e.net.StudentPolicy`` on origin/wp/4b-net
  has this shape). Built in: ``first`` and ``random``, STAND-INS for tests.
- ``--env MODULE:FUNC``: ``FUNC(seed)`` returns an env with ``reset()``,
  ``step(class_index)`` and ``rm``, yielding decisions (objects with
  ``members``) until a terminal (an object with ``outcome`` and ``won``).
  The default is WP2's ``ppo.e2e.env:E2EEnv``; ``raw`` is :class:`RawEnv`, a
  STAND-IN that merges nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import functools
import hashlib
import importlib
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import wait as conn_wait
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ppo.e2e import ops_contract as OC
from ppo.e2e import panels as P

GAME_VERSION = "v0.111.0"
CHARACTER = "Ironclad"
ASCENSION = 10
# Caps (plan section 3.2, PREREG-0 "Reward"): a capped run is a loss.
MAX_FIGHT_TURNS = 100
MAX_FIGHT_DECISIONS = 1000
MAX_RUN_DECISIONS = 6000
# All steps, forced ones included: a backstop for a forced-step loop.
MAX_RUN_STEPS = 200_000

DEFAULT_ENV = "ppo.e2e.env:E2EEnv"
DEFAULT_HANG_TIMEOUT_S = 1800.0
CLEAN_MARKER = "E2E_EVAL_CLEAN"
#: The engine's behaviour knobs (47 + 4 + 1 in sts2-rs); the clean env drops them all.
KNOB_PREFIXES = ("STS2_RUNAI_", "STS2_ROLLOUT_", "STS2_MCTS_")
_PY_DIR = Path(__file__).resolve().parents[2]  # sts2-pycli/python
_REPO = Path(__file__).resolve().parents[4]  # the monorepo checkout


class EvalError(RuntimeError):
    """A refusal: the call stops before (or instead of) writing rows."""


# ── the plug-in contract ─────────────────────────────────────────────────


class StudentPolicy(Protocol):
    """What ``--policy`` returns. ``meta`` (optional) is provenance for config.txt.

    ``act_decisions`` gets the env's decisions and returns one merged-class
    index per decision, in ``range(len(decision.members))``. An optional
    ``begin_run(seed)`` is called before each run.
    """

    def act_decisions(self, decisions: Sequence[Any]) -> list[int]: ...


class FirstOption:
    """STAND-IN policy: always class 0."""

    meta: ClassVar[dict[str, str]] = {"policy": "first"}

    def act_decisions(self, decisions: Sequence[Any]) -> list[int]:
        return [0 for _ in decisions]


class RandomOption:
    """STAND-IN policy: a uniform class, from a stream seeded by the run seed."""

    meta: ClassVar[dict[str, str]] = {"policy": "random"}

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def begin_run(self, seed: int) -> None:
        self._rng = random.Random(f"ppo.e2e.evaluate:{seed}")

    def act_decisions(self, decisions: Sequence[Any]) -> list[int]:
        return [self._rng.randrange(len(d.members)) for d in decisions]


def load_spec(spec: str) -> Any:
    """``module:attr[.attr]`` -> the object."""
    mod, sep, attr = spec.partition(":")
    if not (mod and sep and attr):
        raise EvalError(f"{spec!r} is not MODULE:ATTR")
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_policy(spec: str, ckpt: str | None, device: str) -> Any:
    if spec == "first":
        return FirstOption()
    if spec == "random":
        return RandomOption()
    return load_spec(spec)(ckpt, device=device)


def make_env_factory(spec: str) -> Callable[[int], Any]:
    if spec == "raw":
        return RawEnv
    return load_spec(spec)


# ── the campaign run (plan section 3.1) ───────────────────────────────────


def require_campaign_build() -> None:
    """Refuse unless this process selects v0.111.0 (``sts2_client.GAME_VERSION``).

    Rust ignores the selection, but the Python side reads it (WP1's version-keyed
    combat-context tables, ``PpoRunEnv``'s default build), so a process on another
    selection would encode v0.111.0 runs with another build's tables. WP1's factory
    refuses the same way (``factory.require_selection``).
    """
    import sts2_client

    if sts2_client.GAME_VERSION != GAME_VERSION:
        raise EvalError(
            f"this process selects {sts2_client.GAME_VERSION}, the campaign needs {GAME_VERSION}:"
            f" set {sts2_client.GAME_VERSION_ENV}={GAME_VERSION} before it starts"
        )


def select_campaign_build() -> None:
    """Select v0.111.0 here and in every worker spawned from now on; refuse another
    explicit selection."""
    import sts2_client

    raw = os.environ.get(sts2_client.GAME_VERSION_ENV, "").strip()
    if raw and sts2_client.normalize_game_version(raw) != GAME_VERSION:
        raise EvalError(f"{sts2_client.GAME_VERSION_ENV}={raw}: the campaign runs {GAME_VERSION}")
    sts2_client.set_game_version(GAME_VERSION)


def construct_run(seed: int) -> Any:
    """The campaign run for `seed`: Ironclad A10, v0.111.0, agent mode, Neow, act 1 rolled,
    potion swap on (the student's setting, D4).

    Uses WP1's ``ppo.e2e.factory.new_run`` when that module is present (the merged
    campaign branch), else the same call here, with the same read-back checks.
    """
    try:
        from ppo.e2e import factory  # type: ignore[attr-defined]
    except ImportError:
        factory = None
    if factory is not None:
        return factory.new_run(seed)
    import sts2_rs

    require_campaign_build()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < P.SEED_LIMIT:
        raise AssertionError(f"seed {seed!r} outside [0, 2**31)")
    rm = sts2_rs.RunManager(
        seed,
        character=CHARACTER,
        ascension_level=ASCENSION,
        neow_start=True,
        act1="roll",
        game_version=GAME_VERSION,
        mode="agent",
        interactive_potion_swap=True,
    )
    check_campaign_run(rm)
    return rm


def check_campaign_run(rm: Any) -> None:
    got = (rm.game_version, rm.mode, rm.interactive_potion_swap, rm.character)
    want = (GAME_VERSION, "agent", True, CHARACTER)
    if got != want or int(rm.ascension_level) != ASCENSION:
        raise AssertionError(
            f"not a campaign run: (game_version, mode, interactive_potion_swap, character, "
            f"ascension) = {(*got, rm.ascension_level)}, expected {(*want, ASCENSION)}"
        )


# ── the stand-in env ──────────────────────────────────────────────────────

# The run-level forms of combat actions (`RunManager.available_actions()` in a fight).
COMBAT_RUN_KINDS = frozenset(
    {"play_card", "end_turn", "drink_potion", "combat_resolve_choice"}
    | {"combat_resolve_choice_take_first", "combat_resolve_choice_selection"}
)


def is_combat(actions: Sequence[dict[str, Any]]) -> bool:
    return bool(actions) and all(a.get("action") in COMBAT_RUN_KINDS for a in actions)


@dataclass(eq=False)
class RawDecision:
    """A decision of :class:`RawEnv`: every legal action is its own class."""

    kind: str  # "combat" | "ooc"
    phase: str
    actions: list[dict[str, Any]]
    members: list[list[int]]
    rm: Any
    seed: int
    row: int
    bootstrap: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.members)


@dataclass(eq=False)
class RawTerminal:
    outcome: str  # won | lost | capped | no_actions | error
    won: bool
    cap: str | None
    steps: int
    rows: int
    error: str | None = None


class RawEnv:
    """STAND-IN for WP2's ``E2EEnv``: one campaign run, every multi-option step a decision.

    No merging (each run action is its own class), no tokens: enough to test the
    runner and to play the stand-in policies. It binds an event before reading
    its options (#302), masks a full-belt potion buy (``ppo.env.mask_rejectable``),
    applies single-option steps itself, and ends a run at the plan's caps.
    """

    def __init__(self, seed: int, *, rm: Any = None) -> None:
        self.seed = int(seed)
        self.rm = construct_run(self.seed) if rm is None else rm
        self.steps = 0
        self.rows = 0
        self._fight: tuple[Any, ...] | None = None
        self._fight_rows = 0
        self._round = 0
        self._pending: RawDecision | None = None

    def reset(self) -> RawDecision | RawTerminal:
        return self._drive()

    def step(self, choice: int) -> RawDecision | RawTerminal:
        d = self._pending
        if d is None:
            raise RuntimeError("step() without a pending decision")
        self._pending = None
        self._take(d.actions[d.members[int(choice)][0]])
        return self._drive()

    def _take(self, action: dict[str, Any]) -> None:
        self.rm.take_action(action)
        self.steps += 1

    def _end(self, outcome: str, *, cap: str | None = None) -> RawTerminal:
        won = outcome == "won"
        return RawTerminal(outcome=outcome, won=won, cap=cap, steps=self.steps, rows=self.rows)

    def _drive(self) -> RawDecision | RawTerminal:
        from ppo.env import mask_rejectable

        rm = self.rm
        while True:
            if rm.is_over():
                return self._end("won" if rm.player_won else "lost")
            if self.steps >= MAX_RUN_STEPS:
                return self._end("capped", cap="steps")
            phase = rm.phase()
            if phase == "event":
                rm.materialize_event()
            actions = rm.available_actions()
            if not actions:
                return self._end("no_actions")
            actions = mask_rejectable(rm, phase, actions)
            combat = is_combat(actions)
            if combat:
                where = (int(rm.act_index), tuple(rm.map_coord() or ()))
                if where != self._fight:
                    self._fight, self._fight_rows, self._round = where, 0, 0
            if len(actions) == 1:
                self._take(actions[0])
                continue
            if combat:
                rnd = int(rm.combat_state().get_state().get("round") or 0)
                if rnd < self._round:  # a new fight in the same room
                    self._fight_rows = 0
                self._round = rnd
                if rnd > MAX_FIGHT_TURNS:
                    return self._end("capped", cap="turns")
                if self._fight_rows >= MAX_FIGHT_DECISIONS:
                    return self._end("capped", cap="fight_decisions")
                self._fight_rows += 1
            if self.rows >= MAX_RUN_DECISIONS:
                return self._end("capped", cap="run_decisions")
            self._pending = RawDecision(
                kind="combat" if combat else "ooc",
                phase=phase,
                actions=actions,
                members=[[i] for i in range(len(actions))],
                rm=rm,
                seed=self.seed,
                row=self.rows,
            )
            self.rows += 1
            return self._pending


# ── one run -> one row ────────────────────────────────────────────────────


def is_decision(out: Any) -> bool:
    return hasattr(out, "members")


def boss_slot(rm: Any, coord: Sequence[int]) -> tuple[str | None, str | None]:
    """("boss" | "second_boss", encounter id) of the boss node at `coord`, if it is one."""
    m = rm.full_map() or {}
    for slot in ("boss", "second_boss"):
        b = m.get(slot)
        if b and (b.get("col"), b.get("row")) == tuple(coord):
            return slot, b.get("id")
    return None, None


class BossTracker:
    """Boss fights seen from the stored decisions: entry HP, potions, and whether won.

    A boss counts as entered at the first decision in its room in combat, and as
    won once the run reaches another room (or wins the run while still in it).
    """

    def __init__(self) -> None:
        self.bosses: list[dict[str, Any]] = []
        self._at: tuple[Any, ...] | None = None

    def observe(self, rm: Any) -> None:
        if rm.is_over():
            return
        room = rm.current_room() or {}
        coord = tuple(room.get("map_coord") or ())
        where = (int(rm.act_index), coord)
        if self._at is not None and where != self._at:
            self.bosses[-1]["won"] = True
            self._at = None
        if room.get("room_type") == "boss" and rm.phase() == "combat":
            if self.bosses and self.bosses[-1]["_where"] == where:
                return
            slot, boss = boss_slot(rm, coord)
            self.bosses.append(
                {
                    "act": int(rm.act_index),
                    "slot": slot,
                    "boss": boss,
                    "entry_hp": int(rm.current_hp),
                    "entry_max_hp": int(rm.max_hp),
                    "entry_potions": sum(1 for p in rm.potions() if p),
                    "won": None,
                    "_where": where,
                }
            )
            self._at = where

    def finish(self, won: bool) -> list[dict[str, Any]]:
        if self._at is not None:
            self.bosses[-1]["won"] = bool(won)
            self._at = None
        return [{k: v for k, v in b.items() if k != "_where"} for b in self.bosses]


def _tail(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else "..." + text[-n:]


@functools.cache
def _process_stamps() -> dict[str, Any]:
    from ppo.e2e import contracts as C
    from ppo.e2e import stamps as ST
    from ppo.e2e import vocab as V

    return ST.stamps(vocab_sha256=V.load().sha256, schema_sha256=C.schema_sha256())


def eval_stamps() -> dict[str, Any]:
    """This process's stamps for the rows it writes: ``stamps.stamps`` with the vocab and
    schema sha256 the pre-flight compares (``vocab.load().sha256``,
    ``contracts.schema_sha256()``). Computed once per process (the digests hash files);
    each call returns a fresh copy.

    Raises:
        sts2_client.GameVersionMismatch: The process selects another build than v0.111.0.
    """
    return copy.deepcopy(_process_stamps())


def error_row(seed: int, policy: str, error: str) -> dict[str, Any]:
    """What ``ppo.workers.play_batch`` books for a failed run, plus the stamps of the
    process that books it (:func:`eval_stamps`)."""
    return {
        "seed": seed,
        "policy": policy,
        "outcome": "error",
        "won": False,
        "error": error,
        **eval_stamps(),
    }


class Player:
    """Plays seeds with one env factory and one policy (one per worker process).

    With `log_dir`, every run's action log (plan §3.2) goes to
    `<log_dir>/actions-<pid>.e2el` (`actionlog.LogSink`, an eval run is always kept).
    A terminal without a log (the `RawEnv` stand-in) is skipped."""

    def __init__(
        self,
        env_spec: str,
        policy_spec: str,
        ckpt: str | None,
        device: str,
        log_dir: str | None = None,
    ) -> None:
        require_campaign_build()  # at worker start: a refusal stops the call
        eval_stamps()  # taken once, here: every row of this worker carries them
        self.env_spec = env_spec
        self.policy_spec = policy_spec
        self.make_env = make_env_factory(env_spec)
        self.policy = make_policy(policy_spec, ckpt, device)
        self.sink = None
        if log_dir:
            from .actionlog import LogSink

            self.sink = LogSink(log_dir, worker=os.getpid(), seed=0)

    def _log_run(self, out: Any) -> None:
        """Offer a finished run to the action log; a log that fails is reported on
        stderr, never booked as the run's error."""
        if self.sink is None or not hasattr(out, "log_idx"):
            return
        try:
            self.sink.offer(out, eval_run=True, extra={"policy": self.policy_spec})
        except BaseException as e:  # a Rust panic is a BaseException
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            print(f"evaluate: action log: {_tail(traceback.format_exc())}", file=sys.stderr)

    def play(self, seed: int) -> dict[str, Any]:
        try:
            return self._play(seed)
        except BaseException as e:  # a Rust panic is a BaseException
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            return error_row(seed, self.policy_spec, _tail(traceback.format_exc()))

    def _play(self, seed: int) -> dict[str, Any]:
        from ppo.env import metrics_snapshot
        from ppo.workers import _reached_second_boss

        begin = getattr(self.policy, "begin_run", None)
        if begin is not None:
            begin(seed)
        env = self.make_env(seed)
        tracker = BossTracker()
        out = env.reset()
        decisions = 0
        while is_decision(out) and not getattr(out, "bootstrap", False):
            tracker.observe(env.rm)
            (choice,) = self.policy.act_decisions([out])
            n = len(out.members)
            if isinstance(choice, bool) or not 0 <= int(choice) < n:
                raise ValueError(f"policy chose class {choice!r} of {n}")
            out = env.step(int(choice))
            decisions += 1
        if is_decision(out):
            raise RuntimeError("the env ended a full run on a bootstrap row")
        self._log_run(out)
        if out.outcome == "error":
            return error_row(seed, self.policy_spec, _tail(str(getattr(out, "error", ""))))
        rm = env.rm
        won = bool(out.won)
        room = rm.current_room() or {}
        return {
            "seed": seed,
            "policy": self.policy_spec,
            "ascension": int(rm.ascension_level),
            "character": str(rm.character),
            "act1": str(rm.act1),
            "outcome": "won" if won else str(out.outcome),
            "won": won,
            "cap": getattr(out, "cap", None),
            "steps": int(getattr(out, "steps", 0)),
            "n_decisions": decisions,
            "final": metrics_snapshot(rm),
            "reached_second_boss": _reached_second_boss(rm),
            "bosses": tracker.finish(won),
            "end": {
                "act": int(rm.act_index),
                "floor": int(rm.total_floor()),
                "room": room.get("room_type"),
            },
            **eval_stamps(),  # wheel and py_src among them
        }


# ── the worker pool ───────────────────────────────────────────────────────


def _worker_main(task_conn: Any, result_conn: Any, cfg: dict[str, Any]) -> None:
    """A spawn worker: build the handler once, then answer one task per message.

    ``cfg["handler"]`` names a class (``module:Class``) built with
    ``cfg["handler_args"]``; its ``play(task)`` must return a picklable result and
    should not raise (a raise kills the worker, and the pool books an error).
    Torch, when building the handler loaded it, then runs on one thread (B3).
    """
    if cfg.get("quiet_stdout"):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
    if cfg.get("stderr_log"):
        fd = os.open(cfg["stderr_log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 2)
    try:
        handler = load_spec(cfg["handler"])(**cfg["handler_args"])
    except BaseException:
        result_conn.send(("fatal", _tail(traceback.format_exc())))
        return
    torch = sys.modules.get("torch")  # a handler that plays a net loaded it
    if torch is not None:
        torch.set_num_threads(1)
    result_conn.send(("ready", None))
    while True:
        try:
            task = task_conn.recv()
        except EOFError:
            return
        if task is None:
            return
        result_conn.send(("row", (task, handler.play(task))))


@dataclass
class _Worker:
    proc: Any
    tasks: Any
    results: Any
    ready: bool = False
    task: Any = None
    started: float = 0.0


@dataclass
class Pool:
    """``procs`` spawn workers, each fed one task at a time; hung or dead ones replaced.

    `on_row(task, result)` gets every finished task. A worker that runs one task
    longer than `hang_timeout_s`, or dies, books ``make_error(task, why)`` for it.
    Tasks must not be None.
    """

    cfg: dict[str, Any]
    procs: int
    hang_timeout_s: float
    on_row: Callable[[Any, Any], None]
    make_error: Callable[[Any, str], Any]
    workers: list[_Worker] = field(default_factory=list)
    max_startup_failures: int = 3
    _startup_failures: int = 0

    def _spawn(self) -> _Worker:
        ctx = mp.get_context("spawn")
        task_r, task_w = ctx.Pipe(duplex=False)
        res_r, res_w = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_worker_main, args=(task_r, res_w, self.cfg), daemon=True)
        with OC.worker_thread_env():  # in the worker's environment from exec
            proc.start()
        task_r.close()
        res_w.close()
        return _Worker(proc=proc, tasks=task_w, results=res_r)

    def _kill(self, w: _Worker) -> None:
        with contextlib.suppress(Exception):
            w.proc.kill()
        w.proc.join(5)
        for c in (w.tasks, w.results):
            with contextlib.suppress(Exception):
                c.close()

    def _replace(self, w: _Worker, why: str) -> None:
        task = w.task
        if not w.ready:
            self._startup_failures += 1
            if self._startup_failures >= self.max_startup_failures:
                self._kill(w)
                raise EvalError(f"workers keep dying before they are ready ({why})")
        self._kill(w)
        self.workers[self.workers.index(w)] = self._spawn()
        if task is not None:
            self.on_row(task, self.make_error(task, why))

    def run(self, tasks: Iterable[Any]) -> None:
        todo = deque(tasks)
        if not todo:
            return
        self.workers = [self._spawn() for _ in range(min(self.procs, len(todo)))]
        try:
            while todo or any(w.task is not None for w in self.workers):
                for w in list(self.workers):
                    if w.ready and w.task is None and todo:
                        w.task, w.started = todo.popleft(), time.monotonic()
                        try:
                            w.tasks.send(w.task)
                        except OSError:
                            self._replace(w, "worker died before taking its task")
                by_conn = {id(w.results): w for w in self.workers}
                for conn in conn_wait([w.results for w in self.workers], timeout=1.0):
                    w = by_conn[id(conn)]
                    try:
                        kind, payload = conn.recv()
                    except (EOFError, OSError):
                        self._replace(w, f"worker died (exit code {w.proc.exitcode})")
                        continue
                    if kind == "fatal":
                        raise EvalError(f"a worker could not build its policy/env:\n{payload}")
                    if kind == "ready":
                        w.ready = True
                    elif kind == "row":
                        task, row = payload
                        w.task = None
                        self.on_row(task, row)
                now = time.monotonic()
                for w in list(self.workers):
                    if w.task is not None and now - w.started > self.hang_timeout_s:
                        self._replace(w, f"killed: no result after {self.hang_timeout_s:.0f} s")
                    elif not w.proc.is_alive():
                        self._replace(w, f"worker died (exit code {w.proc.exitcode})")
        finally:
            for w in self.workers:
                with contextlib.suppress(Exception):
                    w.tasks.send(None)
            for w in self.workers:
                w.proc.join(10)
                if w.proc.is_alive():
                    self._kill(w)


# ── files ─────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shard_ranges(start: int, n: int) -> list[tuple[int, int]]:
    end = start + n
    return [(lo, min(lo + P.SHARD_SIZE, end)) for lo in range(start, end, P.SHARD_SIZE)]


def shard_complete(path: Path, lo: int, hi: int) -> bool:
    """e2e_arm.sh's `shard_status`: exactly the seeds lo..hi-1, once each. Reads only `seed`."""
    try:
        with open(path, encoding="utf-8") as f:
            seeds = [json.loads(line)["seed"] for line in f if line.strip()]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return sorted(seeds) == list(range(lo, hi))


def _dump(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True) + "\n"


def write_atomic(path: Path, text: str, tmp_dir: Path | None = None) -> None:
    """Write via a temp file in `tmp_dir` (default: beside `path`) and rename.

    Shards and retries pass the arm's ``work/``, so a crash never leaves a stray
    file in ``shards/`` or ``retry/`` (the scorer stops on one).
    """
    tmp = (tmp_dir or path.parent) / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)


def write_manifest(arm_dir: Path, out: Path) -> str:
    """``(cd ARM; find shards ... | sort | xargs sha256sum; find retry ...) > out``."""
    lines = [
        f"{sha256_file(p)}  {sub}/{p.name}\n"
        for sub in ("shards", "retry")
        for p in sorted((arm_dir / sub).glob("*.jsonl"))
    ]
    write_atomic(out, "".join(lines))
    return sha256_file(out)


def so_digest(so_path: str) -> str:
    """e2e_arm.sh's so_sha256: sha256 of the sorted sha256s of the pin's ``*.so``."""
    hashes = sorted(sha256_file(p) for p in Path(so_path).parent.glob("*.so"))
    return hashlib.sha256("".join(h + "\n" for h in hashes).encode()).hexdigest()


def git_head(repo: Path = _REPO) -> tuple[str, bool]:
    """(HEAD sha, dirty) of the checkout; dirty counts tracked files only, as e2e_arm.sh."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return sha, dirty


# ── the run command ───────────────────────────────────────────────────────


def _clean_env(pin: Path | None, set_env: Sequence[str]) -> dict[str, str]:
    env = {
        "HOME": str(Path.home()),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        **OC.WORKER_THREAD_ENV,
        "PYTHONPATH": os.pathsep.join([*([str(pin)] if pin else []), str(_PY_DIR)]),
        "STS2_GAME_VERSION": GAME_VERSION,
        CLEAN_MARKER: "1",
    }
    if pin is not None:
        env["STS2_WHEEL_PIN"] = str(pin)
    for kv in set_env:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def reexec_clean(
    module: str, argv: Sequence[str], pin: Path | None, set_env: Sequence[str]
) -> None:
    """Replace this process with ``python -m <module> <argv>`` under the clean env.

    Only for a process started as ``python -m <module>``: anything else (a test
    calling ``main`` in-process) would be replaced by the wrong program.
    """
    orig = list(getattr(sys, "orig_argv", []))
    if not any(a == "-m" and b == module for a, b in itertools.pairwise(orig)):
        raise EvalError(f"run this as `python -m {module}` (it re-executes itself in a clean env)")
    env = _clean_env(pin, set_env)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, "-m", module, *argv], env)


def _config_text(args: argparse.Namespace, ident: dict[str, str]) -> str:
    lines = [
        f"arm={args.arm}",
        f"panel={args.panel}",
        f"n={args.n}",
        f"seed_start={args.seed_start}",
        f"sha={ident['sha']}",
        f"pin={ident['pin']}",
        f"so_sha256={ident['so_sha256']}",
        f"python={sys.executable}",
        "runner=ppo.e2e.evaluate",
        f"env_factory={args.env}",
        f"policy={args.policy}",
        f"ckpt={ident['ckpt']}",
        f"ckpt_sha256={ident['ckpt_sha256']}",
        f"device={args.device}",
        f"dirty={ident['dirty']}",
        *(f"env={kv}" for kv in args.set_env),
    ]
    return "\n".join(lines) + "\n"


def _log(arm_dir: Path, msg: str) -> None:
    stamp = dt.datetime.now().astimezone().strftime("%FT%T%z")
    with open(arm_dir / "logs" / "evaluate.log", "a", encoding="utf-8") as f:
        f.write(f"{stamp} {msg}\n")


def _summary(rows: Iterable[dict[str, Any]]) -> str:
    """The unsealed one-line quick look (e2e_arm.sh's), errors after retry excluded."""
    scored = [r for r in rows if r.get("outcome") != "error"]
    n, w = len(scored), sum(1 for r in scored if r.get("won"))
    z = 1.959964
    p = w / n if n else 0.0
    lo = hi = 0.0
    if n:
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = c - h, c + h
    caps = Counter(str(r.get("cap")) for r in scored if r.get("cap"))
    dec = [r["n_decisions"] for r in scored if "n_decisions" in r]
    return (
        f"WR {w}/{n} = {p:.4f} (Wilson 95% {lo:.4f}-{hi:.4f}); caps {dict(caps)};"
        f" decisions/run {sum(dec) / max(1, len(dec)):.1f}"
    )


def run_cmd(args: argparse.Namespace, argv: Sequence[str]) -> int:
    panel = P.runner_panel(args.panel)
    P.check_slice(panel, args.seed_start, args.n)
    if panel.sealed and not args.sealed:
        raise EvalError(f"{panel.name} is sealed: pass --sealed (P4)")
    if args.sealed and not panel.sealed:
        raise EvalError("--sealed is for the sealed panels TEST and SMOKE_SEALED")
    loose = [f for f in ("allow_unpinned", "allow_dirty", "allow_plain_wheel") if getattr(args, f)]
    if panel.sealed and loose:
        raise EvalError(f"{panel.name} takes no --{loose[0].replace('_', '-')}")
    if args.procs < 1:
        raise EvalError("--procs must be positive")
    for kv in args.set_env:
        if "=" not in kv or not kv.split("=", 1)[0]:
            raise EvalError(f"--set-env needs K=V, got {kv!r}")
    pin = Path(args.pin).expanduser().resolve() if args.pin else None
    if pin is None and not args.allow_unpinned:
        raise EvalError("pass --pin <pinned wheel dir> (or E2E_PIN); --allow-unpinned off TEST")
    if pin is not None and not pin.is_dir():
        raise EvalError(f"--pin {pin}: not a directory")

    evals = Path(args.evals).expanduser().resolve()
    if os.environ.get(CLEAN_MARKER) != "1":
        # The clean env drops E2E_PIN / E2E_EVALS: pass what they resolved to.
        full = [*argv, "--evals", str(evals)]
        if pin is not None:
            full += ["--pin", str(pin)]
        reexec_clean("ppo.e2e.evaluate", full, pin, args.set_env)

    # From here on: the clean environment.
    from ppo import wheel_guard

    info = wheel_guard.wheel_build_info()
    so_path = os.path.realpath(str(info.get("so_path") or ""))
    if pin is not None and not so_path.startswith(str(pin) + os.sep):
        raise EvalError(f"sts2_rs imported from {so_path}, not from the pin {pin}")
    try:
        guard = wheel_guard.check_training_wheel(allow_plain=args.allow_plain_wheel, warn=False)
    except SystemExit as e:
        raise EvalError(f"wheel_guard refused the wheel:\n{e}") from None
    import sts2_client

    try:
        eval_stamps()  # the rows the pool books here (a hung or dead worker) carry them
    except sts2_client.GameVersionMismatch as e:
        raise EvalError(str(e)) from None
    sha, dirty = git_head()
    if dirty and not args.allow_dirty:
        raise EvalError(f"worktree {_REPO} has uncommitted changes; evaluate from a clean commit")
    ckpt = str(Path(args.ckpt).resolve()) if args.ckpt else "-"
    if args.ckpt and not Path(ckpt).is_file():
        raise EvalError(f"--ckpt {ckpt}: no such file")
    ident = {
        "sha": sha,
        "pin": str(pin) if pin else "-",
        "so_sha256": so_digest(so_path) if so_path else "-",
        "ckpt": ckpt,
        "ckpt_sha256": sha256_file(Path(ckpt)) if args.ckpt else "-",
        "dirty": "1" if dirty else "0",
    }

    panel_dir = evals / panel.name
    arm_dir = panel_dir / args.arm
    panel_dir.mkdir(parents=True, exist_ok=True)
    if panel.sealed:
        panel_dir.chmod(0o700)
    for sub in ("meta", "shards", "logs", "retry", "work"):
        (arm_dir / sub).mkdir(parents=True, exist_ok=True)
    lock = open(arm_dir / ".lock", "w", encoding="utf-8")  # held until the process exits
    if os.name != "nt":
        import fcntl

        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise EvalError(f"another evaluate holds {arm_dir / '.lock'}") from None

    config = _config_text(args, ident)
    cfg_path = arm_dir / "meta" / "config.txt"
    if cfg_path.is_file():
        old = cfg_path.read_text(encoding="utf-8")
        if old != config:
            diff = [ln for ln in config.splitlines() if ln not in old.splitlines()]
            raise EvalError(
                f"{arm_dir} was produced by a different config ({diff[:3]}...); use a new ARM name"
            )
    else:
        write_atomic(cfg_path, config)
    write_atomic(
        arm_dir / "meta" / "wheel_guard.txt",
        wheel_guard.wheel_banner(guard) + "\n" + json.dumps(guard, indent=2, sort_keys=True) + "\n",
    )
    with open(arm_dir / "meta" / "calls.log", "a", encoding="utf-8") as f:
        stamp = dt.datetime.now().astimezone().strftime("%FT%T%z")
        f.write(f"{stamp} {' '.join(argv)}\n")

    # Shards: run what is not complete, write each once all its seeds are in.
    ranges = shard_ranges(args.seed_start, args.n)
    todo = [
        (lo, hi)
        for lo, hi in ranges
        if not shard_complete(arm_dir / "shards" / f"s{lo}_{hi}.jsonl", lo, hi)
    ]
    worker_cfg = {
        "handler": "ppo.e2e.evaluate:Player",
        "handler_args": {
            "env_spec": args.env,
            "policy_spec": args.policy,
            "ckpt": None if ckpt == "-" else ckpt,
            "device": args.device,
            "log_dir": str(arm_dir / "actions"),
        },
        "quiet_stdout": bool(args.sealed),
        "stderr_log": str(arm_dir / "logs" / "workers.log") if args.sealed else None,
    }
    print(
        f"evaluate: {args.arm} {panel.name} seeds {args.seed_start}..{args.seed_start + args.n - 1}"
        f" at {sha[:12]}; {len(todo)} of {len(ranges)} shard(s) to run, {args.procs} procs",
        flush=True,
    )
    pending: dict[tuple[int, int], dict[int, dict[str, Any]]] = {r: {} for r in todo}
    part_files = {
        r: open(arm_dir / "work" / f"s{r[0]}_{r[1]}.jsonl.part", "w", encoding="utf-8")
        for r in todo
    }
    errors_in_shards = 0

    def make_error(seed: int, why: str) -> dict[str, Any]:
        return error_row(seed, args.policy, why)

    def on_row(seed: int, row: dict[str, Any]) -> None:
        nonlocal errors_in_shards
        r = next(x for x in todo if x[0] <= seed < x[1])
        pending[r][seed] = row
        errors_in_shards += row.get("outcome") == "error"
        part_files[r].write(_dump(row))
        part_files[r].flush()
        if len(pending[r]) == r[1] - r[0]:
            body = "".join(_dump(pending[r][s]) for s in range(r[0], r[1]))
            write_atomic(arm_dir / "shards" / f"s{r[0]}_{r[1]}.jsonl", body, arm_dir / "work")
            part_files[r].close()
            (arm_dir / "work" / f"s{r[0]}_{r[1]}.jsonl.part").unlink()
            done = sum(1 for x in todo if len(pending[x]) == x[1] - x[0])
            _log(
                arm_dir,
                f"shard s{r[0]}_{r[1]} done ({done}/{len(todo)}); errors so far {errors_in_shards}",
            )

    try:
        pool = Pool(worker_cfg, args.procs, args.hang_timeout, on_row, make_error)
        pool.run(s for lo, hi in todo for s in range(lo, hi))
    finally:
        for f in part_files.values():
            f.close()

    # Errored seeds re-run once (P2), on the same binary.
    retried = {int(p.name[1:-6]) for p in (arm_dir / "retry").glob("s*.jsonl")}
    errored = []
    for lo, hi in ranges:
        with open(arm_dir / "shards" / f"s{lo}_{hi}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("outcome") == "error" and r["seed"] not in retried:
                    errored.append(r["seed"])
    if errored:
        print(f"evaluate: re-running {len(errored)} errored seed(s) once", flush=True)

        def on_retry(seed: int, row: dict[str, Any]) -> None:
            write_atomic(arm_dir / "retry" / f"s{seed}.jsonl", _dump(row), arm_dir / "work")

        Pool(worker_cfg, args.procs, args.hang_timeout, on_retry, make_error).run(errored)

    digest = write_manifest(arm_dir, panel_dir / f"{args.arm}.sha256")
    rows, retry_rows = [], {}
    for p in (arm_dir / "retry").glob("s*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                retry_rows[r["seed"]] = r
    n_rows = errs = 0
    for lo, hi in ranges:
        with open(arm_dir / "shards" / f"s{lo}_{hi}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n_rows += 1
                if r.get("outcome") == "error" and r["seed"] in retry_rows:
                    r = retry_rows[r["seed"]]
                errs += r.get("outcome") == "error"
                if not args.sealed:
                    rows.append(r)
    print(f"evaluate: {args.arm} {panel.name} done: rows={n_rows} errors_after_retry={errs}")
    print(f"evaluate: manifest {panel_dir / (args.arm + '.sha256')} ({digest})")
    _log(arm_dir, f"done rows={n_rows} errors_after_retry={errs} manifest {digest}")
    if not args.sealed:
        print(f"evaluate: {_summary(rows)}")
    return 0


# ── the report command (unsealed panels only) ─────────────────────────────


def report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Plan section 11's DEV report: WR, A9-est, deaths, per-boss conversion, entry HP, caps."""
    from ppo.e2e import stats as S

    scored = [r for r in rows if r.get("outcome") not in ("error", "timeout")]
    n = len(scored)
    wins = sum(1 for r in scored if r.get("won"))
    a9 = sum(1 for r in scored if r.get("won") or r.get("reached_second_boss") is True)

    def death_act(r: dict[str, Any]) -> int:
        end = r.get("end") or {}
        return int(end["act"] if "act" in end else (r.get("final") or {}).get("act_index", 0))

    deaths_act = Counter(death_act(r) for r in scored if not r.get("won"))
    deaths_boss: Counter[str] = Counter()
    boss: dict[str, dict[str, float]] = {}
    for r in scored:
        for b in r.get("bosses") or []:
            key = f"a{b['act']} {b.get('slot')} {b.get('boss')}"
            s = boss.setdefault(key, {"entered": 0, "won": 0, "hp_frac": 0.0, "potions": 0.0})
            s["entered"] += 1
            s["won"] += bool(b.get("won"))
            s["hp_frac"] += b["entry_hp"] / max(1, b["entry_max_hp"])
            s["potions"] += b.get("entry_potions", 0)
            if b.get("won") is False:
                deaths_boss[key] += 1
    per_boss = {
        k: {
            "entered": int(v["entered"]),
            "won": int(v["won"]),
            "conversion": v["won"] / v["entered"],
            "entry_hp_frac": v["hp_frac"] / v["entered"],
            "entry_potions": v["potions"] / v["entered"],
        }
        for k, v in sorted(boss.items())
    }
    dec = [r["n_decisions"] for r in scored if isinstance(r.get("n_decisions"), int)]
    return {
        "rows": len(rows),
        "scored": n,
        "missing": len(rows) - n,
        "wins": wins,
        "wr": wins / n if n else None,
        "wilson95": S.wilson(wins, n) if n else None,
        "a9est": a9 / n if n else None,
        "outcomes": dict(Counter(str(r.get("outcome")) for r in rows)),
        "caps": dict(Counter(str(r.get("cap")) for r in scored if r.get("cap"))),
        "deaths_by_act": dict(sorted(deaths_act.items())),
        "deaths_at_boss": dict(deaths_boss.most_common()),
        "per_boss": per_boss,
        "decisions_per_run": sum(dec) / len(dec) if dec else None,
    }


def report_cmd(args: argparse.Namespace) -> int:
    from ppo.e2e import score_test

    panel = P.runner_panel(args.panel)
    if panel.sealed:
        raise EvalError(f"report never reads a sealed panel; {panel.name} is read by score_test")
    evals = Path(args.evals).expanduser()
    try:
        arm = score_test.load_arm(evals, panel, args.arm, sealed_rules=False)
    except score_test.ReadStopped as e:
        raise EvalError(f"integrity: {e}") from None
    out = report([arm.rows[s] for s in sorted(arm.rows)])
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    lo, hi = out["wilson95"] or (math.nan, math.nan)
    print(
        f"{args.arm} {panel.name}: WR {out['wins']}/{out['scored']} = {100 * (out['wr'] or 0):.2f}%"
        f" (Wilson {100 * lo:.2f}-{100 * hi:.2f}%), A9-est {100 * (out['a9est'] or 0):.2f}%,"
        f" missing {out['missing']}, decisions/run {out['decisions_per_run'] or 0:.1f}"
    )
    print(f"outcomes {out['outcomes']}  caps {out['caps']}")
    print(f"deaths by act {out['deaths_by_act']}")
    for k, v in out["per_boss"].items():
        print(
            f"  {k:<36} entered {v['entered']:>4}  won {v['won']:>4}  conv {100 * v['conversion']:5.1f}%"
            f"  entry HP {100 * v['entry_hp_frac']:5.1f}%  potions {v['entry_potions']:.2f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="ppo.e2e.evaluate", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="play a policy on a panel slice")
    run.add_argument("arm")
    run.add_argument("panel", choices=sorted(P.RUNNER_PANELS))
    run.add_argument("n", type=int)
    run.add_argument("seed_start", type=int)
    run.add_argument("--policy", required=True, help="first | random | MODULE:FUNC")
    run.add_argument("--ckpt", default=None)
    run.add_argument(
        "--env", default=DEFAULT_ENV, help=f"raw | MODULE:FUNC (default {DEFAULT_ENV})"
    )
    run.add_argument("--device", default="cpu")
    run.add_argument("--procs", type=int, required=True)
    run.add_argument("--hang-timeout", type=float, default=DEFAULT_HANG_TIMEOUT_S)
    run.add_argument("--sealed", action="store_true")
    run.add_argument("--pin", default=os.environ.get("E2E_PIN") or None)
    run.add_argument(
        "--evals",
        default=os.environ.get("E2E_EVALS") or str(Path.home() / "runs" / "e2e" / "evals"),
    )
    run.add_argument("--set-env", action="append", default=[], metavar="K=V")
    run.add_argument("--allow-unpinned", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--allow-plain-wheel", action="store_true")
    rep = sub.add_parser("report", help="the DEV report of an unsealed arm")
    rep.add_argument("arm")
    rep.add_argument("panel", choices=sorted(P.RUNNER_PANELS))
    rep.add_argument(
        "--evals",
        default=os.environ.get("E2E_EVALS") or str(Path.home() / "runs" / "e2e" / "evals"),
    )
    rep.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            return run_cmd(args, argv)
        return report_cmd(args)
    except (EvalError, P.PanelError) as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
