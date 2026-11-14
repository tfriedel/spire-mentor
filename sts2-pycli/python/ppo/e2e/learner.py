"""The e2e campaign learner (plan §6-§8): serve the workers, cut iterations, run the update.

    PYTHONPATH=python python -u -m ppo.e2e.learner train --save-dir DIR \\
        --sampler-seed-offset N [--resume PATH] [lane args]

The command line, the save-dir files, the cadences, the exit codes and the log
rows are `ppo.e2e.ops_contract`'s (WP7); every state goes through
`ppo.e2e.ckpt`.

**Serving (S0).** The learner process owns the inference device. Each round it
takes one stored decision per env slot from every ready worker (`actors.py`):
a `contracts` row batch per worker, joined into one `rowstore.Chunk`. It runs
the student (`E2ENet.forward` on `net.to_batch`: logits and the value at
acting time) and, while they are used, the teachers (`teachers.py`: the
combat teacher reads its S_ctx tokens from the rows, X-C) in one batch,
samples a merged class per row and sends the classes back. The actor of a row is drawn
once per teacher-acting unit (a fight for combat rows, an act for OOC rows)
with probability β_act (plan §7), by a hash of the segment and the unit, so a
replay draws the same. Every row keeps its behaviour log-prob, actor, value,
the teacher's class probabilities and the acting net's version (updates
loaded). A row's reward arrives with the segment's next row (or its end).

**Iterations** (plan §6, WP4a's `ppo.e2e.algo` protocol). After
`iter_decisions` new decisions the learner cuts the *fresh* table: every row
of a segment that ended (its last row `done`; a practice segment's value-only
end row is a `bootstrap` row with `done` set), and for every open segment
that stored a row this iteration its rows with a known reward plus its newest
row as the bootstrap row (it is stored again, as an ordinary row, next
iteration). A segment that stored no row this iteration is left out, so its
held rows stay dormant. Errored segments are dropped (their held rows via
`HoldBack.drop`), and so are the segments a resume lost (States, below). The
learner's copy of the `AlgoState` (rank 0's after every update) joins the
held run rows in front; every row of that joined table goes to the ranks as
one row batch in one `SharedMemory` block (`rowstore`), which run the update
(`update.py`). The cut decides all of that inline; copying the rows' inputs
into the block (`pack_rows`, `rowstore.gather_to_shm`: ~5 GB a table
mid-run) runs in a thread while serving goes on (`async_pack`, S1-cut), and
the update is submitted once the copy is done, so the rows each update gets
do not depend on the thread. The shipped rows then point into that block
(`rowstore.ShmChunk`; an open segment's newest row stays in its round, which
it ships with again), so their serving rounds and the last held block go. Rank 0 ends
the update by copying the rows its hold-back keeps out of the block into a
block of their own (`update.py` step 6); when the update returns, the
learner maps that block and points its held rows there (`_land_held`,
checked against the new hold-back), so they keep only their own rows alive
and the serving loop copies no row. The update's block goes with the cut's
other rows. The learner keeps collecting
with the policy it has and swaps in the new weights between two inference
batches; at most one update is in flight (`sync_updates` waits for each, for
determinism tests).

**Warm start** (plan §7) is WP4a's `schedules`: V-warm, U0 (teachers act with
probability β_act; KS on every trained row), U1 (λ_KS ramps to 0 by 500M),
U2. A teacherless decision type (the potion swap) gets λ_KS = 0. The learning
check (`--mode learning_check`) has no teacher; its verdict file, passed to
the campaign with `--lc-verdict`, starts the λ_KS ramp (pass) or holds λ_KS at
1 (fail). A failing re-check after a pass (K2, a restart) leaves the ramp
alone (`Clocks.hold_ks`) and raises the OPERATOR alarm `LC_RECHECK_FAIL`.

**States** (plan §8, ops contract item 3). Rank 0 writes them after the
update they follow, on a save job: `state.pt` every `state_every` decisions,
a rolling full state every `full_every` (the newest `full_keep` stay), a
pinned state at every `pin_every` milestone, at each K-gate, on a
`PIN_REQUEST` and at the end, and a slim state (raw weights and every EMA)
every `slim_every`. A full state holds the net, the optimizer, the EMAs, the
`AlgoState` (clocks, KL factor, entropy gates, advantage moments and the
hold-back queue) and the learner's counters. The open segments are lost with
the workers, and a lost run never stores another row, so its held rows would
stay dormant (never aged, never trained) in every later table: a resume
drops them (`HoldBack.resume`) before the ranks get the state, counted in the
first iteration's `assembly.held_lost`. So states leave out the held rows'
inputs (`held` is None; a resume still checks the inputs of a state that has
them). (`lost_uids` drops given runs at the next update on the learner and
every rank, via the job's `drop_uids`.) The action sampler's generator is
reseeded from (seed, sampler seed offset, iteration) at the start of every
collection, so a resume never replays iteration 0's draws (#1081's
learner.py:172-180 bug).

**Stops** (items 5-7). `DONE` or SIGTERM/SIGINT: finish the update in flight,
write `state.pt`, exit 0. The budget: `state.pt`, a final pin, `DONE` with
reason `budget`, exit 0. A non-finite update: save nothing, write `NAN.json`,
exit 86. A wheel or source mismatch (a worker message's stamps or a segment
end's; caught before the batch trains), an lr off its closed form or an F9
invariant hit: `ALARM_<name>`, exit 87.

Rows follow `ppo.e2e.contracts` (SCHEMA.md section 9). The shipped row batch
carries the algorithm's flags (`algo.join`, SCHEMA.md 9.1 since revision 2):
`bootstrap` on a cut's row and, with `done`, on a practice segment's end row;
`done` alone on a terminal. The stamps (section 10) carry
`contracts.schema_sha256()`, the content-ID table's `vocab_sha256`, the combat
teacher's `teacher_stamp` when one is loaded, and `--filled`, the fill events
the rows include; a resume also runs `contracts.check_stamps` with
`restart=True`. `cfg.json` adds the teachers' own `stamps()`.

Evals (item 8): the learner claims each queued request and runs
`ppo.e2e.eval_runner` on it (`eval_cmd`), which exports the request's variant
of the pinned state, plays it with WP6's `ppo.e2e.evaluate` and WP4b's
`make_student`, and leaves `wins` and `runs` for `evals/done/<id>.json`.

STAND-IN (until its package lands): the serving path (S0 here; WP5b's S1b
replaces it at a hot-swap).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import queue
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context, shared_memory
from multiprocessing.connection import wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import actors, ckpt
from . import algo as A
from . import contracts as C
from . import ops_contract as OC
from . import schedules as S
from .actors import M as MC
from .net import (
    E2ENet,
    NetConfig,
    current_vocab_sha256,
    load_checkpoint,
    net_from_state,
    net_state,
    to_batch,
    to_packed,
)
from .rowstore import Chunk, ShmChunk, batch_from_shm, concat, gather_to_shm, take
from .rowstore import sweep as rowstore_sweep
from .update import Rank, rank_main

#: The row layout's stamp (`contracts.schema_sha256()`).
SCHEMA_SHA256 = C.schema_sha256()
STATE_FORMAT = "ppo.e2e.learner/3"
#: The alarm for a learning check that fails after one passed (`Clocks.hold_ks`
#: returned False; WP4a review fix). Its row in `ops_contract.ALARMS` gives the
#: alarm file its severity (OPERATOR) and who acts.
LC_RECHECK_FAIL = "LC_RECHECK_FAIL"
#: Decision types the run teacher does not cover (λ_KS = 0, plan §7): the run nets
#: resolved full-belt potion swaps rule-side, so they never scored one.
RUN_TEACHERLESS = frozenset({C.DT["POTION_SWAP"]})
#: Decision type names by `contracts.DT` id.
DTYPE_NAMES = tuple(d.name for d in C.DECISION_TYPES)
FIGHT_CAPS = frozenset({"turns", "fight_decisions"})
RUN_CAPS = frozenset({"run_decisions", "steps"})


def _f9_hung(e: Mapping[str, Any]) -> bool:
    """An ended segment whose F9 hits do not stop the lane (PREREG Amendment 9): it spun
    to the env's step cap (`OC.F9_HANG_CAP`), so it ended `capped` and scored no win.
    A hang is an engine bug to replay from `ops/f9_hits.jsonl`, not an exploit the policy
    can be paid for; a hit in a segment that ends any other way still stops the lane."""
    return (
        int(e.get("f9_violations", 0)) > 0
        and e.get("outcome") == "capped"
        and e.get("cap") == OC.F9_HANG_CAP
    )


def lc_combat_ev(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The learning check's combat value EV over the stats rows (GH #1153).

    The median over the last half of the rows that carry one (at least the last 5).
    Explained variance is unbounded below, so one blown-up batch dragged the old
    last-5 mean under the gate on a run that learned normally; the median needs half
    the window to go bad. `last5_mean` (the old statistic) is reported, not gated."""
    evs = [(r.get("value_ev") or {}).get("combat") for r in rows]
    evs = [float(x) for x in evs if isinstance(x, (int, float)) and math.isfinite(x)]
    tail = evs[-max(5, len(evs) // 2) :]
    return {
        "median": float(np.median(tail)) if tail else math.nan,
        "rows": len(tail),
        "last5_mean": float(np.mean(evs[-5:])) if evs else math.nan,
    }


class UsageError(RuntimeError):
    """The learner refuses to start (ops contract exit 2)."""


class NanStop(RuntimeError):
    """A non-finite update (exit 86): nothing is saved after it."""


class GuardStop(RuntimeError):
    """A guard refused to continue (exit 87); `name` is the alarm."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"ALARM_{name}: {reason[:2000]}")
        self.name = name


def vocab_sha256() -> str:
    """The content-ID table's sha256 (`ppo.e2e.vocab`), the rows' id space."""
    return current_vocab_sha256()


@dataclass
class Cfg:
    save_dir: str
    mode: str = "train"  # "train" | "learning_check"
    seed: int = 0
    sampler_seed_offset: int = 0
    resume: str = ""  # a full state to resume from ("" = a fresh run)
    # Trainer runs (plan §3.1 "Other seeds", PREREG P1): a campaign lane (mode train, no
    # max_iters) draws from panels.STUDENT_TRAINER_SEEDS (1e9), every other lane (smokes,
    # learning checks, B-T0) from THROWAWAY_TRAINER_SEEDS (1.3e9). 0 = its block's start;
    # a base whose run seeds leave the lane's block is refused.
    run_seed_base: int = 0
    practice: bool = True
    # F9 (FIDELITY.md): `invariants.KNOWN_ISSUES` keys the env checker tolerates
    # (["crossbow_hand_cap"] if G1 launches with engine fault E1); [] = strict.
    f9_tolerate: list[str] = field(default_factory=list)
    # The fill events (`contracts.FILL_EVENTS`) the rows include: the stamps' `filled`
    # (SCHEMA.md 10). A fresh net inits their reserved inputs live; a resume may add
    # events (the identity rule: the checkpoint's zero weights stay), never drop one.
    filled: list[str] = field(default_factory=list)
    # The campaign's character: the one this process plays (`factory.CHARACTER`, from
    # `STS2_E2E_CHARACTER`, which the workers inherit). Recorded here so a run's cfg
    # says what it trained; a mismatch with the process is refused. Any character but
    # Ironclad needs "chars" in `filled` (GH #1373). Ironclad may declare it: its rows
    # then show orbs and Doom too (the learner sets `STS2_E2E_CHARS` for its workers).
    character: str = "Ironclad"
    # Action logs (plan §3.2): the workers keep boss-2, capped and error segments and 2%
    # of runs under <save_dir>/actions (`actionlog.LogSink`, 10 GB in total).
    action_logs: bool = True
    # The workers run the Rust training env (`rsenv.RsE2EEnv`: env loop, bookkeeping and
    # tokenizer in Rust, lockstep-equal to `env.E2EEnv`). Needs the run teacher off: its
    # features read the Python env's `run_obs`.
    rust_env: bool = False
    # Check each start's first table against `filled` (`fillcheck.check`, repeated
    # while inconclusive); a mismatch is FILL_MISMATCH, exit 87, before it trains.
    # Off only for scripted rollout sources (tests), whose rows are not real.
    fill_check: bool = True
    # Processes and devices (L1: GPU0 serves, GPU1-3 update).
    workers: int = 32
    n_envs: int = 32
    infer_device: str = "cuda:0"
    update_devices: str = "cuda:1,cuda:2,cuda:3"
    bf16: bool = True
    rank_threads: int = 2
    sync_rounds: bool = False  # wait for every worker each round, in worker order
    sync_updates: bool = False  # wait for each update before collecting the next batch
    # Receive the workers' messages in a thread (`ProcessSource` `reader`), so the pipe
    # reads overlap the serving loop's work. Which rows a round serves is timing either way.
    pipe_reader: bool = True
    # Serving: the student and the combat teacher score a round as this many forwards,
    # rows sorted by token count, each padded to its own length. 1 = one padded batch,
    # S0's numbers bit for bit. More saves device time but costs host time per forward
    # and changes logits by float rounding: it pays on a device-bound host (laptop: 1.65x
    # at 4 groups over 1), not on the a4000, whose serving is host-bound (S1-lean, #1097).
    serve_groups: int = 1
    # Serve pad-free (#1097 S1-pad): the student through `net.to_packed` and
    # `E2ENet.forward_packed`, the combat teacher through `CombatTeacher.packed`. Rows
    # sit back to back (flash varlen attention under bf16), so no padded token, PRIV
    # slot or option is computed; one forward per model, so `serve_groups` is moot.
    # Off: the padded path, S1's numbers bit for bit. On: equal up to float rounding.
    packed_serve: bool = False
    # The cut (S1-cut, #1097): copy the shipped rows' inputs into the update's row block
    # in a thread while serving goes on, and submit the update when the copy is done.
    # What the cut ships (rows, flags, the join, iteration tags) is decided inline at the
    # same point either way, so the rows each update gets are the same; only the copy,
    # and with it the update's start, moves. False: the copy runs inline (S0's timing).
    async_pack: bool = True
    # Threads of that copy (`rowstore.gather_to_shm`, by chunk; the same bytes at any
    # count). a4000, a 4.6 GB table: 1 thread 4.6 s, 2 2.7 s, 4 2.0 s; serving slows by
    # ~1% per thread while it runs (laptop).
    pack_threads: int = 2
    # The net (`NetConfig` overrides; {} = plan §5's 6 x 256) and its init.
    net: dict[str, Any] = field(default_factory=dict)
    init: str = ""  # a WP4b checkpoint ("" = random init from `seed`)
    # Teachers (plan §7): a named pair ("R2", "R3m") or explicit checkpoints.
    teacher_pair: str = ""
    combat_teacher: str = ""
    run_teacher: str = ""
    # SCHEMA.md open question 15's fallback: the run teacher (and the workers' features
    # for it) read `contracts.honest_obs` instead of the raw obs. Off: the input R2 and
    # R3 were measured on; re-measure the teacher before turning it on.
    run_teacher_honest: bool = False
    # Batches (plan §6).
    iter_decisions: int = 131_072
    epochs: int = 2
    minibatch: int = 8192
    strata_per_mb: int = 24
    probe_strata: int = 24
    fwd_chunk: int = 2048
    hold_iters: int = A.HOLD_ITERS
    lam_c: float = A.LAM_C
    lam_r: float = A.LAM_R
    clip: float = 0.2
    is_cap: float = 2.0
    dual_clip: float = 3.0
    vf_coef: float = 0.5
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    adv_decay: float = 5e-4
    ema_decays: list[float] = field(default_factory=lambda: [0.996, 0.999])
    # `ScheduleCfg` overrides (lr, warm-start, KL and entropy constants, the budget).
    # On a resume an empty dict keeps the saved schedule; a non-empty one replaces it.
    sched: dict[str, Any] = field(default_factory=dict)
    max_iters: int = 0  # stop once this many iterations exist in total (0 = the budget)
    max_wall_s: float = 0.0  # 0 = no wall-clock limit (the learning check uses one)
    # Cadences in decisions (ops contract item 3).
    state_every: int = OC.STATE_EVERY
    full_every: int = OC.FULL_EVERY
    full_keep: int = OC.FULL_KEEP
    pin_every: int = OC.PIN_EVERY
    slim_every: int = OC.SLIM_EVERY
    k_gates: list[list[Any]] = field(
        default_factory=lambda: [[g.name.lower(), g.at_decisions] for g in OC.K_GATES]
    )
    # The eval runner, run at nice 10 on each claimed request: "builtin" is
    # `ppo.e2e.eval_runner` (WP6's evaluate on WP4b's student of the request's
    # state and variant) with `eval_args` appended; "" leaves the requests queued;
    # anything else is a command template with {request} {state} {variant} {panel}
    # {n} {seed_start} {max_procs} {out}. A command that writes {out}/result.json
    # has that file's fields (`wins`, `runs`, ...) added to evals/done/<id>.json.
    eval_cmd: str = "builtin"
    eval_args: str = ""
    heartbeat_s: float = 30.0
    throughput_s: float = OC.THROUGHPUT_EVERY_S
    # The learning check (plan §7): the windows scale with `lc_window`.
    lc_window: int = 10_000
    lc_margin: float = 1.0
    lc_ev: float = 0.1
    lc_verdict: str = ""  # a learning_check.json to apply: pass -> ramp, fail -> hold λ_KS

    def __post_init__(self) -> None:
        if self.mode not in ("train", "learning_check"):
            raise ValueError(f"mode {self.mode!r}")
        if self.mode == "learning_check":
            self.teacher_pair = self.combat_teacher = self.run_teacher = ""
            self.init = ""
        world = len([d for d in self.update_devices.split(",") if d])
        if world < 1 or self.strata_per_mb % world:
            raise ValueError(f"strata_per_mb {self.strata_per_mb} must split over {world} ranks")

    def schedule(self) -> S.ScheduleCfg:
        return S.ScheduleCfg.from_dict({**S.DEFAULT.to_dict(), **self.sched})


class Row:
    """One stored decision: its row is row `i` of `chunk` (a `rowstore.Chunk`)."""

    __slots__ = (
        "action",
        "actor",
        "boot",
        "cell",
        "chunk",
        "comb",
        "dtype",
        "has_t",
        "i",
        "it",
        "logp",
        "ncls",
        "ntok",
        "ref",
        "reward",
        "t",
        "uid",
        "v",
        "version",
    )

    def __init__(
        self,
        ref: int,
        chunk: Chunk,
        i: int,
        uid: int,
        t: int,
        meta: np.ndarray,
        it: int,
        version: int,
    ) -> None:
        self.ref = ref
        self.chunk, self.i = chunk, i
        self.ntok = int(chunk.rb["n_tok"][i])
        self.uid = uid
        self.t = t
        self.comb = bool(meta[MC["is_combat"]])
        self.dtype = int(meta[MC["dtype"]])
        self.ncls = int(meta[MC["n_classes"]])
        self.cell = int(meta[MC["cell"]])
        self.boot = bool(meta[MC["boot"]])
        self.it = it
        self.version = version
        self.action = 0
        self.logp = 0.0
        self.actor = A.ACTOR_STUDENT
        self.v = 0.0
        self.has_t = False  # the chunk's `tprob` holds teacher class probabilities
        self.reward: float | None = None

    @classmethod
    def held(cls, ref: int, chunk: Chunk, i: int) -> Row:
        """A held row restored from a state: only its inputs (the table has the rest)."""
        r = cls.__new__(cls)
        r.ref, r.chunk, r.i = int(ref), chunk, int(i)
        r.ncls = int(chunk.rb["n_classes"][i])
        r.ntok = int(chunk.rb["n_tok"][i])
        r.has_t = bool(chunk.rb["t_src"][i])
        return r


#: `ProcessUpdater.stop`'s end-of-messages mark for its reader thread.
_END = "_end"


def _drop_held(res: Any) -> None:
    """Remove the held rows' block an update result names (`update.Rank._pack_held`)
    that no learner will map: the result of an update in flight at a stop."""
    held = res.get("held") if isinstance(res, dict) else None
    if held is not None:
        with contextlib.suppress(FileNotFoundError):
            shared_memory.SharedMemory(name=held["shm"], track=False).unlink()


def _map_held(held: Mapping[str, Any] | None) -> ShmChunk | None:
    """Rank 0's block of the update's held rows (`update.Rank._pack_held`), mapped
    here. Its name goes at once: the block goes with the last row in it."""
    if held is None:
        return None
    shm = shared_memory.SharedMemory(name=held["shm"], track=False)
    shm.unlink()
    try:
        return ShmChunk(shm, held["layout"])
    except BaseException:
        shm.close()
        raise


class Seg:
    """A segment (a run or a practice segment) with rows not yet shipped as ordinary rows."""

    __slots__ = (
        "actor",
        "boot_end",
        "cell",
        "ended",
        "error",
        "kind",
        "next_t",
        "pending",
        "r_sum",
        "uid",
    )

    def __init__(self, uid: int, kind: str, cell: int) -> None:
        self.uid = uid
        self.kind = kind
        self.cell = cell
        self.pending: list[Row] = []
        self.next_t = 0
        self.r_sum = 0.0  # every reward given to the segment's rows (T4: the env's total)
        self.ended = False
        self.error = False
        self.boot_end = False
        self.actor: dict[tuple[int, int], int] = {}


def mix64(*parts: int) -> int:
    """A splitmix64 hash of `parts` (the actor draws)."""
    x = 0x9E3779B97F4A7C15
    for p in parts:
        x = (x ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x = z ^ (z >> 31)
    return x


def sampler_seed(seed: int, offset: int, it: int) -> int:
    """The action sampler's seed for collection `it` (plan §8: from (seed + offset, iteration))."""
    ss = np.random.SeedSequence([seed & 0xFFFFFFFF, offset & 0xFFFFFFFF, int(it)])
    return int(ss.generate_state(1, np.uint64)[0] >> 1)


WINDOWS = sys.platform == "win32"


def lock_file(f: Any) -> None:
    """An exclusive, non-blocking lock on the open file `f` (released when it closes):
    `fcntl.flock`, or `msvcrt.locking` on Windows. OSError when another process holds it."""
    if WINDOWS:
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def stop_group(p: subprocess.Popen, *, kill: bool) -> None:
    """SIGTERM (`kill`: SIGKILL) to the process group of `p`, started with
    `start_new_session=True`; on Windows, which has no process groups, `p` alone."""
    with contextlib.suppress(OSError):
        if hasattr(os, "killpg"):
            os.killpg(p.pid, signal.SIGKILL if kill else signal.SIGTERM)
        elif kill:
            p.kill()
        else:
            p.terminate()


def eval_argv(cmd: list[str] | str, nice: int, extra: str = "") -> list[str] | str:
    """An eval's command for `Popen`: `cmd` (an argv, or a command line) plus the
    command line `extra`, split POSIX-style and run under `nice -n` where `nice`
    exists. On Windows the command stays one string, which CreateProcess parses
    (POSIX splitting would eat a path's backslashes), and nothing lowers the
    priority."""
    if WINDOWS:
        line = cmd if isinstance(cmd, str) else subprocess.list2cmdline(cmd)
        return f"{line} {extra}".strip()
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    argv += shlex.split(extra)
    return (["nice", "-n", str(nice)] if shutil.which("nice") else []) + argv


def make_stamps(
    *, teacher: Mapping[str, str] | None = None, filled: Sequence[str] = ()
) -> dict[str, Any]:
    """WP1's `stamps.stamps` (game version, py_src, py_client, wheel, vocab, schema) plus
    the remaining `contracts.STAMP_KEYS`: `teacher`, the combat teacher's
    `contracts.teacher_stamp` when one is loaded (the rows' S_ctx content slots use its
    RareMap), `filled`, and `prior_sha256`, the committed content prior's
    (`content_prior.sha256()`) when `filled` names "content-prior" (the learner fills
    the net's prior from it, review finding NT-1). Raises unless the process selects
    v0.111.0, and `contracts.SchemaError` on an unknown fill event."""
    from . import stamps as ST

    base = ST.stamps(vocab_sha256=vocab_sha256(), schema_sha256=SCHEMA_SHA256)
    prior = None
    if "content-prior" in filled:
        from . import content_prior as CP

        prior = CP.sha256()
    return {
        **base,
        **C.stamps(
            vocab_sha256=base["vocab_sha256"],
            prior_sha256=prior,
            wheel=base["wheel"],
            source_digest=base["py_src"],
            teacher=teacher,
            filled=filled,
        ),
        "torch": str(torch.__version__),
    }


def teacher_types(combat: bool, run: bool) -> np.ndarray:
    """Per decision type id: whether a teacher scores it (λ_KS > 0 possible)."""
    t = np.zeros(A.N_TYPES, bool)
    for d in C.DECISION_TYPES:
        if d.status != "live" or not d.id:
            continue
        if d.id in C.COMBAT_DTYPES:
            t[d.id] = combat
        else:
            t[d.id] = run and d.id not in RUN_TEACHERLESS
    return t


def closed_form_lr(row: dict[str, Any], cfg: S.ScheduleCfg) -> dict[str, float]:
    """The lr of each group from the schedule's closed form (ops contract `lr_closed_form`)."""
    frac = float(row["kl_m"]) * S.anneal(float(row["t_wall"]), cfg) * S.warmup(int(row["it"]), cfg)
    return {"trunk": cfg.lr_trunk * frac, "value": cfg.lr_value * frac}


# ── rollout sources ──────────────────────────────────────────────────────


class ProcessSource:
    """`workers` spawned `actors.worker_main` processes over duplex pipes, each started
    under `ops_contract.WORKER_THREAD_ENV`.

    With `reader` (not with `sync`), one daemon thread (`_read`) receives every
    worker's message as soon as it is sent, so the pipe reads (which release the GIL)
    overlap the serving loop's own work instead of running inside it. A worker sends
    again only after its reply, so each pipe has at most one message in flight, and
    `ready` takes whatever has arrived, as `wait` would have shown it ready. The thread
    hands a failure to the main thread instead of dying with it: a pipe that ends or
    breaks, or a message that cannot be unpickled, arrives as an error item that
    `ready` raises (naming the worker), and anything else that stops the thread does
    too; `ready` never waits on a thread that is gone. It keeps reading through `stop`,
    so a worker blocked sending a large message can finish it, read the stop and exit,
    instead of being killed at the deadline (#1151 is the same for the update ranks)."""

    def __init__(self, n: int, wcfg: dict[str, Any], sync: bool, reader: bool = False) -> None:
        ctx = get_context("spawn")
        self.sync = sync
        self.conns: list[Any] = []
        self.procs: list[Any] = []
        with OC.worker_thread_env():  # in each worker's environment from exec
            for wid in range(n):
                a, b = ctx.Pipe(duplex=True)
                p = ctx.Process(target=actors.worker_main, args=(wid, b, wcfg), daemon=True)
                p.start()
                b.close()
                self.conns.append(a)
                self.procs.append(p)
        #: The reader's summed seconds in `Connection.recv` (the pipe read and the
        #: unpickle), which then sit outside the serving loop's `wait_s`; 0 without it.
        self.recv_s = 0.0
        self._got: dict[int, Any] = {}
        self._inbox: queue.Queue[tuple[int, Any, BaseException | None]] | None = None
        self._halt = threading.Event()
        self._reader: threading.Thread | None = None
        if reader and not sync:
            self._inbox = queue.Queue()
            self._reader = threading.Thread(
                target=self._read, args=(list(self.conns),), name="e2e-pipe-reader", daemon=True
            )
            self._reader.start()

    def _read(self, conns: list[Any]) -> None:
        """Every worker's messages into `_inbox` as (wid, msg, None), and its failure as
        (wid, None, exc); once halted, received messages are dropped. Ends when every
        pipe has ended, or a failure of its own ends it, as (-1, None, exc)."""
        inbox = self._inbox
        assert inbox is not None
        sel = selectors.DefaultSelector()
        try:
            for i, c in enumerate(conns):
                sel.register(c, selectors.EVENT_READ, (i, c))
            while sel.get_map():
                for key, _ev in sel.select(timeout=0.5):
                    i, c = key.data
                    t = time.perf_counter()
                    try:
                        msg = c.recv()
                    except BaseException as e:  # EOF, a torn or unpicklable message
                        sel.unregister(c)  # its stream can no longer be trusted
                        if not self._halt.is_set():
                            inbox.put((i, None, e))
                        continue
                    self.recv_s += time.perf_counter() - t
                    if not self._halt.is_set():
                        inbox.put((i, msg, None))
        except BaseException as e:  # handed over: `ready` raises it
            inbox.put((-1, None, e))
        finally:
            sel.close()

    def _take(self, block: bool) -> tuple[int, Any, BaseException | None] | None:
        assert self._inbox is not None and self._reader is not None
        while True:
            try:
                return self._inbox.get(timeout=5.0) if block else self._inbox.get_nowait()
            except queue.Empty:
                if not block:
                    return None
                if not self._reader.is_alive():  # all pipes ended without an error item
                    raise RuntimeError("the pipe reader ended") from None

    def ready(self) -> list[int]:
        if self.sync:
            return list(range(len(self.conns)))
        if self._inbox is None:
            got = wait(self.conns)
            return sorted(self.conns.index(c) for c in got)
        item = self._take(block=True)
        while item is not None:
            i, msg, err = item
            if err is not None:
                raise self._dead(i, err)
            if i in self._got:
                raise RuntimeError(f"rollout worker {i}: a second message before its reply")
            self._got[i] = msg
            item = self._take(block=False)
        return sorted(self._got)

    def _dead(self, wid: int, err: BaseException | None) -> RuntimeError:
        """The error for worker `wid`'s pipe failing with `err`; a worker that is dead
        has its exit code by then, normally (it may lag the EOF a moment)."""
        deadline = time.time() + 0.5
        while wid >= 0 and self.procs[wid].exitcode is None and time.time() < deadline:
            time.sleep(0.02)
        dead = [(w, p.pid, p.exitcode) for w, p in enumerate(self.procs) if p.exitcode is not None]
        who = "the pipe reader" if wid < 0 else f"rollout worker {wid}"
        e = RuntimeError(f"{who} failed: {err!r}; exited (wid, pid, exitcode): {dead}")
        e.__cause__ = err
        return e

    def recv(self, i: int) -> Any:
        if self._inbox is not None:
            return self._got.pop(i)
        try:
            return self.conns[i].recv()
        except EOFError as e:
            raise self._dead(i, e) from e

    def send(self, i: int, msg: Any) -> None:
        try:
            self.conns[i].send(msg)
        except OSError as e:  # BrokenPipe: it died after its message
            raise self._dead(i, e) from e

    def stop(self) -> None:
        self._halt.set()
        for c in self.conns:
            with contextlib.suppress(OSError):
                c.send(("stop",))
        deadline = time.time() + 30  # the reader drains meanwhile, so the workers can exit
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.kill()  # workers ignore SIGTERM
                p.join(timeout=10)
        if self._reader is not None:  # every pipe has ended: it reads their EOFs and ends
            self._reader.join(timeout=10)
        for c in self.conns:
            c.close()
        self.conns, self.procs = [], []
        self._got.clear()


# ── updaters ─────────────────────────────────────────────────────────────


class ProcessUpdater:
    """One spawned `update.rank_main` per update device (torch.distributed over a file store).

    One daemon thread (`_read`) moves the ranks' messages from their queue into a local
    one, and everything else reads only that (#1151). Two reasons. A rank exits only
    once its queue's feeder has written all it sent, so with nobody reading, a rank with
    a large unread result cannot exit on a stop, and gets killed mid-write. And a
    killed rank's half-written message blocks `Queue.get` forever: its timeout covers
    the wait for a message to start, not for its bytes (the parent holds the pipe's
    write end too, so no EOF comes). With the thread, the ranks can always flush, and a
    torn message strands only the thread: `_get` still sees the dead rank and raises,
    and `stop` returns."""

    def __init__(self, devices: list[str], rcfg: SimpleNamespace, init_path: str) -> None:
        ctx = get_context("spawn")
        self.store_dir = tempfile.mkdtemp(prefix="e2e_dist_")
        store = os.path.join(self.store_dir, "store")
        self.outq = ctx.Queue()
        self.inqs: list[Any] = []
        self.procs: list[Any] = []
        for r, d in enumerate(devices):
            q = ctx.Queue()
            p = ctx.Process(
                target=rank_main,
                args=(r, len(devices), d, store, rcfg, init_path, q, self.outq),
                daemon=True,
            )
            p.start()
            self.inqs.append(q)
            self.procs.append(p)
        self._inbox: queue.Queue[tuple[str, int, Any]] = queue.Queue()
        self._reader = threading.Thread(target=self._read, name="e2e-rank-reader", daemon=True)
        self._reader.start()
        for _ in devices:
            kind, r, _ = self._get(True)
            if kind != "ready":
                raise RuntimeError(f"update rank {r}: {kind!r} at start-up")

    def _read(self) -> None:
        """The ranks' messages into `_inbox`, until `_END` (`stop`) or a broken queue."""
        while True:
            try:
                msg = self.outq.get()
            except (EOFError, OSError, ValueError):  # closed under it
                return
            if msg[0] == _END:
                return
            self._inbox.put(msg)

    def _get(self, block: bool) -> tuple[str, int, Any]:
        while True:
            try:
                kind, r, payload = self._inbox.get(timeout=10.0 if block else 0.001)
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"an update rank died: (rank, exitcode) {dead}") from None
                if not block:
                    return ("none", -1, None)
                continue
            if kind == "error":
                raise RuntimeError(f"update rank {r} failed:\n{payload}")
            return kind, r, payload

    def submit(self, job: dict[str, Any]) -> None:
        for q in self.inqs:
            q.put(("update", job))

    def poll(self, block: bool) -> dict[str, Any] | None:
        kind, _r, res = self._get(block)
        if kind == "none":
            return None
        if kind != "done":
            raise RuntimeError(f"unexpected rank message {kind!r}")
        return res

    def save(self, job: dict[str, Any]) -> list[str]:
        self.inqs[0].put(("save", job))
        kind, _r, written = self._get(True)
        if kind != "saved":
            raise RuntimeError(f"unexpected rank message {kind!r}")
        return written

    def stop(self) -> None:
        for q in self.inqs:
            q.put(("stop",))
        deadline = time.time() + 60  # the reader drains meanwhile, so the ranks can flush
        killed = False
        for p in self.procs:
            p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                p.kill()  # ranks ignore SIGTERM
                p.join(timeout=10)
                killed = True
        if not killed:  # all exited, so all they sent is in the pipe: read it all
            self.outq.put((_END, -1, None))
            self._reader.join(timeout=10)  # longer only behind a torn message
        # else a killed rank may have left a torn message, which strands the reader
        # (a daemon): take what it got
        while True:  # an update in flight still answered: its held rows' block goes
            try:
                kind, _r, res = self._inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                _drop_held(res)
        for q in [*self.inqs, self.outq]:
            q.cancel_join_thread()
        shutil.rmtree(self.store_dir, ignore_errors=True)


class LocalUpdater:
    """One in-process `update.Rank` (world 1): tests and laptop smokes."""

    def __init__(self, device: str, rcfg: SimpleNamespace, init_path: str) -> None:
        if torch.device(device).type == "cpu":
            torch.set_num_threads(max(1, int(rcfg.rank_threads)))
        init = torch.load(init_path, map_location="cpu", weights_only=True)
        self.rank = Rank(0, 1, device, rcfg, init)
        self.res: dict[str, Any] | None = None

    def submit(self, job: dict[str, Any]) -> None:
        self.res = self.rank.update(job)

    def poll(self, block: bool) -> dict[str, Any] | None:
        res, self.res = self.res, None
        return res

    def save(self, job: dict[str, Any]) -> list[str]:
        return self.rank.save(job)

    def stop(self) -> None:
        res, self.res = self.res, None
        _drop_held(res)  # an update no one polled


# ── the learner ──────────────────────────────────────────────────────────


def _ready(x: Any) -> Callable[[], Any]:
    """A finished result in the shape of a `teachers` `..._start` call's."""
    return lambda: x


class Learner:
    def __init__(
        self,
        cfg: Cfg,
        *,
        source: Any = None,
        updater: str = "process",
        clock: Any = time.time,
        argv: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.now = clock
        self.updater_kind = updater
        self.source = source
        self.sd = ckpt.SaveDir(cfg.save_dir)
        self.sd.root.mkdir(parents=True, exist_ok=True)
        self._lock_f: Any = None
        self._lock()
        try:
            self._init(argv)
        except BaseException:
            self._unlock()
            raise

    # ── start-up ─────────────────────────────────────────────────────────

    def _lock(self) -> None:
        f = open(self.sd.p(OC.LEARNER_LOCK), "a", encoding="utf-8")
        try:
            lock_file(f)
        except OSError:
            f.close()
            raise UsageError(f"{self.sd.p(OC.LEARNER_LOCK)} is held by another learner") from None
        self._lock_f = f

    def _unlock(self) -> None:
        if self._lock_f is not None:
            self._lock_f.close()
            self._lock_f = None

    def _init(self, argv: list[str] | None) -> None:
        cfg = self.cfg
        if not cfg.resume and self.sd.state.exists():
            raise UsageError(f"{self.sd.state} exists: pass --resume PATH or use a new --save-dir")
        from .invariants import KNOWN_ISSUES

        if unknown := sorted(set(cfg.f9_tolerate) - set(KNOWN_ISSUES)):
            raise UsageError(f"--f9-tolerate {unknown}: not in invariants.KNOWN_ISSUES")
        OC.atomic_write_text(self.sd.p(OC.LEARNER_PID), json.dumps(OC.pid_record(argv=argv)) + "\n")
        import sts2_client

        from . import factory

        if cfg.character != factory.CHARACTER:
            raise UsageError(
                f"--character {cfg.character}: this process plays {factory.CHARACTER}; set "
                f"{factory.CHARACTER_ENV}={cfg.character} before the learner starts"
            )
        if cfg.character != C.BASE_CHARACTER and C.FILL_CHARS not in cfg.filled:
            raise UsageError(
                f"--filled {sorted(cfg.filled)} with --character {cfg.character}: every "
                f'character but {C.BASE_CHARACTER} declares "chars"'
            )
        # The workers (and this process) tokenize Ironclad rows with the event iff declared.
        factory.set_fill_chars(C.FILL_CHARS in cfg.filled)
        try:
            self.stamps = make_stamps(filled=cfg.filled)
        except sts2_client.GameVersionMismatch as e:
            raise UsageError(str(e)) from None
        except C.SchemaError as e:
            raise UsageError(f"--filled: {e}") from None
        self.dev_i = torch.device(cfg.infer_device)
        self.teachers = self._load_teachers()
        combat = getattr(self.teachers, "combat", None)
        if getattr(combat, "teacher_stamp", None) is not None:
            self.stamps["teacher"] = dict(combat.teacher_stamp)
        self.version = 0
        self.decisions = 0
        self.runs_done = 0
        self.next_ref = 0
        self.elapsed0 = 0.0
        self.lc_first: list[float] = []
        self.lc_last: deque[float] = deque(maxlen=max(1, cfg.lc_window))
        self.totals: Counter = Counter()
        self.resumed_from = ""
        self.held: dict[int, Row] = {}
        # Runs whose held rows the next update drops on the learner and every rank.
        self.lost_uids: list[int] = []
        # Held rows a resume dropped (`HoldBack.resume`), counted in the next stats row.
        self.held_lost = 0
        opt_state = ema_state = None
        if cfg.resume:
            st = torch.load(cfg.resume, map_location="cpu", weights_only=True)
            if st.get("format") != STATE_FORMAT:
                raise UsageError(f"{cfg.resume}: not an e2e learner state ({st.get('format')!r})")
            self._check_resume_stamps(st.get("stamps") or {})
            algo = A.AlgoState.from_dict(st["algo"], cfg=cfg.schedule() if cfg.sched else None)
            net_ck, opt_state, ema_state = st["net"], st["opt"], st["ema"]
            ls = st["learner"]
            self.decisions, self.runs_done = int(ls["decisions"]), int(ls["runs_done"])
            self.version, self.next_ref = int(ls["version"]), int(ls["next_ref"])
            self.elapsed0 = float(ls["elapsed"])
            self.lc_first = [float(x) for x in ls["lc_first"]]
            self.lc_last.extend(float(x) for x in ls["lc_last"])
            self.totals.update(ls.get("totals", {}))
            if algo.clocks.it != self.version or algo.clocks.decisions != self.decisions:
                raise UsageError(f"{cfg.resume}: learner counters disagree with the AlgoState")
            if st.get("held") is not None:  # a state from before the learner left them out
                held = held_from_state(st["held"])
                refs = set(algo.hold.cols["ref"].tolist()) if len(algo.hold) else set()
                if refs != set(held):
                    raise UsageError(f"{cfg.resume}: held-row inputs do not match the hold-back")
            # Every held row belongs to a run the restart lost (WP4a): drop them
            # before the first join and before the ranks get the state.
            self.held_lost = algo.hold.resume()
            self.resumed_from = cfg.resume
            # The iterations after this state are replayed: park their rows.
            parked = park_orphan_stats(
                self.sd.p(OC.STATS), self.sd.p(OC.STATS_ORPHAN), int(algo.clocks.it)
            )
            if parked:
                print(
                    f"[e2e] resume at it {algo.clocks.it}: moved {parked} stats rows past it"
                    f" to {OC.STATS_ORPHAN}",
                    flush=True,
                )
            if self.stamps["prior_sha256"] != net_ck["cfg"].get("prior_sha256"):
                # A restart that adds the content prior (check_stamps' restart rule).
                net = net_from_state(
                    net_ck, schema_sha256=SCHEMA_SHA256, vocab_sha256=self.stamps["vocab_sha256"]
                )
                self._fill_prior(net)
                net_ck = net_state(
                    net,
                    schema_sha256=SCHEMA_SHA256,
                    vocab_sha256=self.stamps["vocab_sha256"],
                    stamps=self.stamps,
                )
        else:
            if cfg.init:
                net, ck = load_checkpoint(cfg.init, schema_sha256=SCHEMA_SHA256)
                # The lane restamps it below: first the restart rule (review LRN-2),
                # so a net that read filled columns never trains with them at zero.
                try:
                    C.check_stamps(ck.get("stamps") or {}, self.stamps, restart=True)
                except C.SchemaError as e:
                    raise UsageError(f"--init {cfg.init}: {e}") from None
            else:
                # A fresh net inits the reserved inputs of the launch's fill events
                # live (`E2ENet(filled=...)`); `--init` and `--resume` keep their
                # checkpoint's weights (the identity rule).
                net = E2ENet(NetConfig.from_dict(cfg.net), seed=cfg.seed, filled=cfg.filled)
            self._fill_prior(net)
            net_ck = net_state(
                net,
                schema_sha256=SCHEMA_SHA256,
                vocab_sha256=self.stamps["vocab_sha256"],
                stamps=self.stamps,
            )
            algo = A.AlgoState(
                cfg.schedule(),
                n_types=A.N_TYPES,
                teacher_types=teacher_types(*self._teacher_flags()),
                hold_iters=cfg.hold_iters,
                adv_decay=cfg.adv_decay,
                lam_c=cfg.lam_c,
                lam_r=cfg.lam_r,
            )
        self.lc_applied = None
        if cfg.lc_verdict:
            with open(cfg.lc_verdict, encoding="utf-8") as f:
                verdict = json.load(f)
            if verdict.get("pass"):
                algo.clocks.pass_learning_check(algo.cfg)
            elif not algo.clocks.hold_ks():
                # A re-check failed after a pass: the ramp stands, the operator looks.
                OC.write_alarm(
                    self.sd.root,
                    LC_RECHECK_FAIL,
                    writer="learner",
                    reason=f"{cfg.lc_verdict}: the learning check failed after one passed",
                    verdict=cfg.lc_verdict,
                )
            self.lc_applied = bool(verdict.get("pass"))
        if self.decisions >= algo.cfg.budget_decisions:
            raise UsageError("the budget is already spent")
        if cfg.mode == "train" and algo.cfg.budget_decisions < 10_000_000_000 and not cfg.max_iters:
            raise UsageError("a campaign needs budget_decisions >= 1e10 (plan §6)")
        self.cfg = cfg = self._run_seeds(cfg, algo.cfg.budget_decisions)
        self.algo = algo
        if cfg.rust_env and self.teachers is not None and self.teachers.run is not None:
            # The run teacher's features read `run_obs`, which rsenv rows do not carry,
            # so the workers raise at the first round it is on. Refuse that at launch
            # when it is on now. A loaded teacher whose schedule is already off (the
            # Ironclad lane: --teacher-pair R3, beta and lambda_KS at 0) is fine.
            if self._teacher_on():
                raise UsageError(
                    "--rust-env cannot serve the run teacher, which is on at "
                    f"{self.decisions} decisions (its rows carry no run_obs); "
                    "run with --rust-env false until its schedule ends"
                )
        self.lr_watch = S.LrWatch(algo.cfg)
        self.net_inf = net_from_state(
            net_ck, schema_sha256=SCHEMA_SHA256, vocab_sha256=self.stamps["vocab_sha256"]
        ).to(self.dev_i)
        self.net_inf.eval()
        self.net_inf.bf16 = cfg.bf16 and self.dev_i.type == "cuda"
        self.init_path = str(self.sd.p(f"{OC.CKPT_DIR}/rank_init.pt"))
        os.makedirs(os.path.dirname(self.init_path), exist_ok=True)
        ckpt.atomic_write(
            self.init_path,
            lambda p: torch.save(
                {
                    "net": net_ck,
                    "algo": algo.to_dict(),
                    "opt": opt_state,
                    "ema": ema_state,
                    "schema_sha256": SCHEMA_SHA256,
                    "vocab_sha256": self.stamps["vocab_sha256"],
                    "dtypes": list(DTYPE_NAMES),
                },
                p,
            ),
        )
        self.gen = torch.Generator(device=self.dev_i)
        self.segs: dict[int, Seg] = {}
        self.n_new = 0
        self.n_dispatched = self.algo.clocks.it
        self.inflight: dict[str, Any] | None = None
        # A cut whose row block is still being copied (`dispatch`, `_pack_done`).
        self.packing: dict[str, Any] | None = None
        self._packer: ThreadPoolExecutor | None = None
        self.updater: Any = None
        self.evals: dict[str, tuple[subprocess.Popen, Path]] = {}
        self.t_session = time.time()
        self._t_hb = 0.0
        self._tp = {"t": time.time(), "decisions": self.decisions, "wait": 0.0}
        self._t_row = time.time()
        self._stop_req = ""
        self._t_done = 0.0
        self._t_evals = 0.0
        self._eval_note = False
        self.workers_stats: dict[str, Any] = {}
        self.draw_log: list[tuple[int, int, list[float]]] = []  # (collection, seed, first draws)
        self.stats_rows: list[dict[str, Any]] = []  # this session's rows (tests read them)
        self.k_gates = sorted((str(r), int(d)) for r, d in cfg.k_gates)
        # The fill check runs on this start's tables until one is conclusive.
        self._fill_pending = bool(cfg.fill_check)

    @staticmethod
    def _run_seeds(cfg: Cfg, budget: int) -> Cfg:
        """`cfg` with its trainer seed block resolved (review findings LRN-1, ENVR-6).

        PREREG P1: a campaign lane's runs come from `panels.STUDENT_TRAINER_SEEDS`,
        every other lane's from `THROWAWAY_TRAINER_SEEDS`. Every seed the lane can draw
        (`actors.run_seed`: base + 7 * iteration + (worker * 64 + slot) * 100,000 + run
        count, up to its last iteration) must lie in that block (UsageError)."""
        from . import panels

        campaign = cfg.mode == "train" and not cfg.max_iters
        name = "STUDENT_TRAINER_SEEDS" if campaign else "THROWAWAY_TRAINER_SEEDS"
        block = getattr(panels, name)
        base = cfg.run_seed_base or block.start
        if cfg.n_envs > 64:
            raise UsageError(f"--n-envs {cfg.n_envs} > 64: two slots would share run seeds")
        iters = cfg.max_iters or 2 * math.ceil(budget / max(1, cfg.iter_decisions))
        hi = base + 7 * iters + cfg.workers * 64 * 100_000
        if not (block.start <= base and hi <= block.stop):
            lane = "a campaign lane" if campaign else f"a {cfg.mode} lane that is not the campaign"
            raise UsageError(
                f"--run-seed-base {base}: {lane} draws run seeds from panels.{name} "
                f"[{block.start}, {block.stop}), but its seeds reach [{base}, {hi})"
            )
        return dataclasses.replace(cfg, run_seed_base=base)

    def _fill_prior(self, net: E2ENet) -> None:
        """The "content-prior" fill event (SCHEMA.md 2.9, review finding NT-1): a launch
        or restart that declares it reads the committed prior into the net's `prior`
        buffer, whose stem weights then train. Without the event the buffer stays zero."""
        want = self.stamps["prior_sha256"]
        if want is None or net.prior_sha256 == want:
            return
        from . import content_prior as CP

        net.set_prior(CP.table(), want)
        print(f"[e2e] content prior filled: prior_sha256 {want}", flush=True)

    def _check_resume_stamps(self, have: dict[str, Any]) -> None:
        """A resume may move the wheel or the source (a hot-swap deploys a new SHA), never
        the schema, the vocab or the game version, and follows `contracts.check_stamps`'
        restart rule: the same combat teacher where both name one, fill events and the
        prior only added."""
        from . import stamps as ST

        moved = ST.stamp_change(have, self.stamps)
        if moved:
            print(f"[e2e] resume: stamps moved since the state: {moved}", flush=True)
        for k in ("schema_sha256", "vocab_sha256", "game_version"):
            if have.get(k) != self.stamps.get(k):
                raise UsageError(f"resume: stamp {k} {have.get(k)!r} != {self.stamps.get(k)!r}")
        try:
            C.check_stamps(have, self.stamps, restart=True)
        except C.SchemaError as e:
            raise UsageError(f"resume: {e}") from None

    def _teacher_flags(self) -> tuple[bool, bool]:
        cfg = self.cfg
        return (
            bool(cfg.teacher_pair or cfg.combat_teacher),
            bool(cfg.teacher_pair or cfg.run_teacher),
        )

    def _load_teachers(self) -> Any:
        from . import teachers as TT

        cfg = self.cfg
        if cfg.teacher_pair:
            t = TT.load_pair(cfg.teacher_pair, self.dev_i)
        elif cfg.combat_teacher and cfg.run_teacher:
            # The final R3 by its paths: `load_pair` checks each file's SHA256SUMS.
            t = TT.load_pair(None, self.dev_i, combat=cfg.combat_teacher, run=cfg.run_teacher)
        else:
            t = TT.Teachers(
                TT.CombatTeacher(cfg.combat_teacher, self.dev_i) if cfg.combat_teacher else None,
                TT.RunTeacher(cfg.run_teacher, self.dev_i) if cfg.run_teacher else None,
            )
        if t.combat is None and t.run is None:
            return None
        if t.run is not None:
            t.run.honest_previews = bool(cfg.run_teacher_honest)
        if t.combat is not None:
            t.combat.serve_groups = int(cfg.serve_groups)
            t.combat.packed = bool(cfg.packed_serve)
        return t

    def teacher_provenance(self) -> dict[str, Any] | None:
        """The loaded teachers' `stamps()` (paths, sha256s, settings) for cfg.json."""
        if self.teachers is None:
            return None
        out: dict[str, Any] = {}
        for k in ("combat", "run"):
            x = getattr(self.teachers, k, None)
            out[k] = x.stamps() if x is not None and hasattr(x, "stamps") else None
        return out

    def rank_cfg(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self.cfg))

    def worker_cfg(self) -> dict[str, Any]:
        cfg = self.cfg
        rare = None
        if self.teachers is not None and self.teachers.combat is not None:
            rare = self.teachers.combat.rare_json
        return {
            "n_envs": cfg.n_envs,
            "run_seed_base": cfg.run_seed_base,
            "start_it": self.n_dispatched,
            "start_decisions": self.decisions,
            "seed": cfg.seed,
            "practice": cfg.practice,
            "f9_tolerate": list(cfg.f9_tolerate),
            "teacher_rare": rare,
            "run_teacher_honest": bool(cfg.run_teacher_honest),
            "action_log_dir": str(self.sd.p(OC.ACTIONS_DIR)) if cfg.action_logs else None,
            "rust_env": bool(cfg.rust_env),
        }

    def start(self) -> None:
        cfg = self.cfg
        # Row blocks of a learner killed by SIGKILL (GH #1152): /dev/shm is RAM.
        if gone := rowstore_sweep():
            print(f"[e2e] removed {len(gone)} row blocks of dead learners", flush=True)
        devs = [d for d in cfg.update_devices.split(",") if d]
        if self.updater_kind == "local":
            if len(devs) != 1:
                raise ValueError("the local updater runs one rank")
            self.updater = LocalUpdater(devs[0], self.rank_cfg(), self.init_path)
        else:
            self.updater = ProcessUpdater(devs, self.rank_cfg(), self.init_path)
        Path(self.init_path).unlink(missing_ok=True)  # the ranks hold it now
        if self.source is None:
            self.source = ProcessSource(
                cfg.workers, self.worker_cfg(), cfg.sync_rounds, reader=cfg.pipe_reader
            )
        elif not hasattr(self.source, "ready"):  # a factory: called with the worker cfg
            self.source = self.source(self.worker_cfg())

    def stop(self) -> None:
        if self.packing is not None:  # a cut no update will read: its block goes
            pk, self.packing = self.packing, None
            with contextlib.suppress(Exception):  # a failed copy left no block
                shm, _layout, _s = pk["fut"].result()
                shm.unlink()
                shm.close()
        if self._packer is not None:
            self._packer.shutdown(wait=True)
            self._packer = None
        if self.inflight is not None:
            self.inflight["block"].unlink()  # it closes with the last row in it
            self.inflight = None
        if self.source is not None and hasattr(self.source, "stop"):  # not a factory
            self.source.stop()
        if self.updater is not None:
            self.updater.stop()
        self._stop_evals(requeue=True)
        self._unlock()

    # ── teacher use ──────────────────────────────────────────────────────

    def _beta_act(self) -> float:
        return S.beta_act(self.decisions, self.algo.cfg)

    def _teacher_on(self) -> bool:
        if self.teachers is None:
            return False
        c = self.algo.clocks
        lam = S.lam_ks(self.decisions, self.algo.cfg, hold=c.ks_hold, offset=c.ks_offset)
        return lam > 0 or self._beta_act() > 0

    def _actor(self, seg: Seg, kind: int, idx: int, beta: float, teacher: int) -> int:
        """Who acts in a unit (a fight or an act): drawn once, at its first row (plan §7)."""
        key = (kind, idx)
        a = seg.actor.get(key)
        if a is None:
            u = mix64(self.cfg.seed, self.cfg.sampler_seed_offset, seg.uid, kind, idx) / 2.0**64
            a = teacher if u < beta else A.ACTOR_STUDENT
            seg.actor[key] = a
        return a

    # ── serving ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _student(self, ch: Chunk) -> tuple[torch.Tensor, torch.Tensor]:
        """The student's log-probs `[B, A]` and values `[B]` (fp32, on the device) of
        every row of `ch`. With `serve_groups` > 1 the rows run as that many forwards,
        sorted by visible token count (`teachers.length_groups`), each padded to its
        own token bucket: a row's outputs differ from one padded batch's only by float
        rounding, and `A` is the same (the widest group's option bucket). With
        `packed_serve` the rows run as one pad-free forward (`E2ENet.forward_packed`):
        again the same up to float rounding and the same `A`."""
        from .teachers import length_groups

        if self.cfg.packed_serve:
            out = self.net_inf.forward_packed(to_packed(ch.rb, self.dev_i))
            return out.logp.float(), out.value.float()
        vis = np.concatenate([[0], np.cumsum(C.student_mask(ch.rb), dtype=np.int64)])
        off = ch.off["tok"]
        parts = length_groups(vis[off[1:]] - vis[off[:-1]], self.cfg.serve_groups)
        if len(parts) == 1:
            out = self.net_inf(to_batch(ch.rb, self.dev_i))  # nothing queued: no need to pin
            return out.logp[: ch.n].float(), out.value[: ch.n].float()
        # Pinned: group k + 1's upload must not wait for group k's queued forward.
        outs = [self.net_inf(to_batch(take(ch, p), self.dev_i, pin=True)) for p in parts]
        lp = torch.full(
            (ch.n, max(o.logp.shape[1] for o in outs)), float("-inf"), device=self.dev_i
        )
        val = torch.empty(ch.n, device=self.dev_i)
        for p, o in zip(parts, outs, strict=True):
            idx = torch.from_numpy(p).to(self.dev_i, non_blocking=True)
            lp[idx, : o.logp.shape[1]] = o.logp[: len(p)].float()
            val[idx] = o.value[: len(p)].float()
        return lp, val

    @torch.no_grad()
    def _serve(
        self, rows: list[Row], ch: Chunk, metas: np.ndarray, feats: list[Any], st: dict[str, Any]
    ) -> np.ndarray:
        """Score and sample one round's rows (all of `ch`, in order) and fill their
        acting-time fields in the chunk (`rowstore.ACTING`, `tprob`).

        The student's and the teachers' forwards are all queued (pinned uploads)
        before anything waits on the device, and the actor draws run meanwhile, so
        the host work overlaps the device's. The teacher-acted rows' behaviour
        log-probs go to the device as one block. With `serve_groups` 1 the sampled
        actions, log-probs, values and teacher fields are S0's serving path's
        (bit-identical up to device float nondeterminism); with `serve_groups` > 1
        equal up to the student's float rounding (see `_student`)."""
        b = len(rows)
        tf = time.time()
        lp_s, v_dev = self._student(ch)
        width = lp_s.shape[1]
        teacher_lp: list[np.ndarray | None] = [None] * b
        scored: list[int] = []
        finish = None
        if self._teacher_on():
            scored, finish = self._teacher_start(rows, ch, feats)
        beta = self._beta_act()
        actors_ = np.zeros(b, np.int64)
        acted: list[int] = []
        for i in scored:  # every teacher-scored row (no bootstrap row is scored)
            r, m = rows[i], metas[i]
            seg = self.segs[r.uid]
            who = A.ACTOR_COMBAT_TEACHER if r.comb else A.ACTOR_RUN_TEACHER
            a = self._actor(seg, int(m[MC["unit_kind"]]), int(m[MC["unit_idx"]]), beta, who)
            if a != A.ACTOR_STUDENT:
                acted.append(i)
            actors_[i] = a
        if finish is not None:
            finish(teacher_lp)
        beh = lp_s
        if acted:
            # Each acting row's behaviour is its teacher's class log-probs (fp32), -inf
            # past its class count: S0 wrote the same values row by row.
            tw = np.full((len(acted), width), -np.inf, np.float32)
            for k, i in enumerate(acted):
                tw[k, : rows[i].ncls] = teacher_lp[i]
            beh = lp_s.clone()
            idx = torch.from_numpy(np.asarray(acted, np.int64)).to(self.dev_i, non_blocking=True)
            beh[idx] = torch.from_numpy(tw).to(self.dev_i, non_blocking=True)
        act = torch.multinomial(beh.exp(), 1, generator=self.gen).squeeze(1)
        logp = beh.gather(1, act[:, None]).squeeze(1).cpu().numpy()
        act_np = act.cpu().numpy()
        values = v_dev.cpu().numpy()
        st["fwd_s"] += time.time() - tf
        arg_s = lp_s.argmax(1).cpu().numpy()
        rb = ch.rb
        for i, r in enumerate(rows):
            r.v = float(values[i])
            rb["v_behav"][i] = r.v
            rb["iter"][i] = r.it
            if r.boot:
                continue
            r.action = int(act_np[i])
            r.logp = float(logp[i])
            r.actor = int(actors_[i])
            rb["action"][i], rb["logp_behav"][i], rb["actor"][i] = r.action, r.logp, r.actor
            tl = teacher_lp[i]
            if tl is not None:
                p = np.exp(tl).astype(np.float32)
                rb["tprob"][ch.span("cls", i)] = p / p.sum(dtype=np.float32)
                rb["t_src"][i] = 1 if r.comb else 2
                r.has_t = True
                key = "combat" if r.comb else "ooc"
                st["agree_n"][key] += 1
                st["agree"][key] += int(int(np.argmax(tl)) == int(arg_s[i]))
            if r.actor != A.ACTOR_STUDENT:
                st["teacher_rows"] += 1
        return act_np

    def _teacher_start(
        self, rows: list[Row], ch: Chunk, feats: list[Any]
    ) -> tuple[list[int], Callable[[list[np.ndarray | None]], None]]:
        """Queue the teachers' class log-probs: the combat teacher from the rows' S_ctx
        blocks (X-C), the run teacher from the workers' `featurize_row` inputs.
        Returns the scored rows (in order) and a call that waits for the scores and
        writes each scored row's into its slot of the list it is given."""
        t = self.teachers
        tt = self.algo.teacher_types
        ci = [
            i
            for i, r in enumerate(rows)
            if r.comb and not r.boot and t.combat is not None and tt[r.dtype]
        ]
        ri = [
            i
            for i, r in enumerate(rows)
            if not r.comb
            and not r.boot
            and t.run is not None
            and tt[r.dtype]
            and feats[i] is not None
        ]
        jobs: list[tuple[list[int], Callable[[], list[np.ndarray]]]] = []
        if ri:
            args = ([feats[i] for i in ri], [ch.classes(i) for i in ri], [rows[i].ncls for i in ri])
            start = getattr(t.run, "logp_start", None)
            jobs.append((ri, start(*args) if start else _ready(t.run.logp(*args))))
        if ci:
            start = getattr(t.combat, "logp_rows_start", None)
            jobs.append((ci, start(ch.rb, ci) if start else _ready(t.combat.logp_rows(ch.rb, ci))))

        def finish(out: list[np.ndarray | None]) -> None:
            for idx, done in jobs:
                for i, lp in zip(idx, done(), strict=True):
                    out[i] = lp

        return sorted(ci + ri), finish

    # ── collection ───────────────────────────────────────────────────────

    def _new_stats(self) -> dict[str, Any]:
        return {
            "outcomes": Counter(),
            "ends": [],
            "bad_stamps": [],  # worker messages whose wheel or py_src is not the learner's
            "runs": 0,
            "practice": 0,
            "practice_cells": Counter(),
            "floor": [],
            "won": 0,
            "bosses": 0,
            "boss2": 0,
            "capped": 0,
            "caps": Counter(),
            "fights": 0,
            "fights_capped": 0,
            "runs_capped": 0,
            "errors": 0,
            "errors_sample": [],
            "error_types": Counter(),  # `error/<type>` counts of the error segments
            # The ended segments' summed counters (the end records' `counts`, the env's
            # `E2ETerminal.counts`): what the pre-flight reads beyond the fields above.
            "segment_counts": Counter(),
            "reward_mismatch": 0,
            "reward_mismatch_sample": [],
            "f9_checks": 0,
            # F9 hits that stop the lane: every one but a hung segment's (`_f9_hung`).
            "f9_violations": 0,
            "f9_hung_hits": 0,
            "f9_hung_segments": 0,
            "f9_stop_samples": [],
            "f9_tolerated": 0,
            "unknown": 0,
            "rows": 0,
            "boot_rows": 0,
            "rows_type": Counter(),
            "teacher_rows": 0,
            "agree": Counter(),
            "agree_n": Counter(),
            "batches": 0,
            "batch_rows": 0,
            "fwd_s": 0.0,
            "wait_s": 0.0,
            "recv_s": 0.0,
            "policy_swaps": 0,
        }

    def collect(self) -> dict[str, Any] | None:
        """Serve the workers until `iter_decisions` new decisions are stored.

        Returns None when a stop was requested (`DONE`, SIGTERM) mid-collection.
        """
        cfg = self.cfg
        c = self.n_dispatched
        seed = sampler_seed(cfg.seed, cfg.sampler_seed_offset, c)
        self.gen.manual_seed(seed)
        probe = torch.Generator(device=self.dev_i)
        probe.manual_seed(seed)
        self.draw_log.append((c, seed, torch.rand(4, generator=probe, device=self.dev_i).tolist()))
        st = self._new_stats()
        t0 = time.time()
        r0 = self.source.recv_s
        while self.n_new < cfg.iter_decisions:
            if self._stop_wanted():
                return None
            if self.packing is not None:  # the last cut's block is copied: submit it
                self._pack_done(block=False)
            if self.inflight is not None and not cfg.sync_updates:
                if self._poll(block=False):
                    st["policy_swaps"] += 1
            tw = time.time()
            ready = self.source.ready()
            msgs = [(i, self.source.recv(i)) for i in ready]
            st["wait_s"] += time.time() - tw
            rows: list[Row] = []
            metas: list[np.ndarray] = []
            feats: list[Any] = []
            per_src: list[tuple[int, int]] = []
            ends_all: list[dict[str, Any]] = []
            ch = Chunk(concat([m[4] for _i, m in msgs]))
            j0 = 0
            for i, msg in msgs:
                _kind, wid, meta, r_prev, _rb, ex, ends = msg
                n = len(meta)
                for j in range(n):
                    m = meta[j]
                    rows.append(self._store(ch, j0 + j, m, float(r_prev[j]), c))
                    metas.append(m)
                fx = ex.get("feats")
                feats.extend(fx if fx is not None else [None] * n)
                per_src.append((i, n))
                j0 += n
                ends_all.extend(ends)
                if ex.get("stats"):
                    self.workers_stats[str(wid)] = ex["stats"]
                sx = ex.get("stamps")
                if sx is not None and any(sx.get(k) != self.stamps[k] for k in ("wheel", "py_src")):
                    st["bad_stamps"].append({"wid": int(wid), **sx})
            if j0 != ch.n:
                raise RuntimeError(f"round: {j0} metas for {ch.n} rows")
            acts = np.zeros(0, np.int64)
            if rows:
                acts = self._serve(rows, ch, np.stack(metas), feats, st)
                st["batches"] += 1
                st["batch_rows"] += len(rows)
                for r in rows:
                    if r.boot:
                        st["boot_rows"] += 1
                        continue
                    st["rows"] += 1
                    st["rows_type"][DTYPE_NAMES[r.dtype]] += 1
                    # The decision clock counts every collected policy row, also those
                    # of a segment the cut drops for an env error (review LRN-6).
                    self.n_new += 1
                    self.decisions += 1
            want_feats = bool(
                self.teachers is not None and self.teachers.run is not None and self._teacher_on()
            )
            info = {"decisions": self.decisions, "feats": want_feats}
            o = 0
            for i, k in per_src:
                self.source.send(i, ("act", [int(x) for x in acts[o : o + k]], info))
                o += k
            for e in ends_all:
                self._ingest_end(e, st)
            self._ops_tick()
        st["collect_s"] = time.time() - t0
        st["recv_s"] = self.source.recv_s - r0
        st["collection"] = c
        st["sampler_seed"] = seed
        return st

    def _store(self, ch: Chunk, i: int, m: np.ndarray, r_prev: float, c: int) -> Row:
        uid, t = int(m[MC["uid"]]), int(m[MC["t"]])
        if int(ch.rb["run_uid"][i]) != uid or int(ch.rb["t"][i]) != t:
            raise RuntimeError(f"row {i}: (run_uid, t) disagree with its meta ({uid}, {t})")
        seg = self.segs.get(uid)
        if t == 0:
            if seg is not None:
                raise RuntimeError(f"segment uid {uid} reused")
            kind = "practice" if int(m[MC["practice"]]) else "run"
            seg = self.segs[uid] = Seg(uid, kind, int(m[MC["cell"]]))
        else:
            if seg is None or seg.next_t != t or not seg.pending or seg.pending[-1].t != t - 1:
                raise RuntimeError(f"segment {uid}: row {t} out of order")
            seg.pending[-1].reward = r_prev
            seg.r_sum += r_prev
        row = Row(self.next_ref, ch, i, uid, t, m, c, self.version)
        self.next_ref += 1
        seg.pending.append(row)
        seg.next_t = t + 1
        return row

    def _ingest_end(self, e: dict[str, Any], st: dict[str, Any]) -> None:
        st["ends"].append(e)
        st["segment_counts"].update({str(k): int(v) for k, v in (e.get("counts") or {}).items()})
        out = e["outcome"]
        st["outcomes"][out] += 1
        st["f9_checks"] += int(e.get("f9_checks", 0))
        hits = int(e.get("f9_violations", 0))
        hung = _f9_hung(e)
        if hung:
            st["f9_hung_hits"] += hits
            st["f9_hung_segments"] += 1
        else:
            st["f9_violations"] += hits
        if hits:
            rec = self._f9_note(e, hung)
            if not hung and len(st["f9_stop_samples"]) < 3:
                st["f9_stop_samples"].append(rec)
        st["f9_tolerated"] += int((e.get("counts") or {}).get("f9/tolerated", 0))
        st["unknown"] += int(e.get("unknown", 0))
        st["fights"] += int(e.get("fights", 0))
        cap = e.get("cap")
        if cap in FIGHT_CAPS:
            st["fights_capped"] += 1
        seg = self.segs.get(e["uid"])
        if hung:
            # A fight that spun to the step cap with broken invariants: none of its rows
            # train (the cut drops them like an error segment's) and it counts as no run.
            if seg is not None:
                seg.error = True
            return
        if out == "error":
            st["errors"] += 1
            if len(st["errors_sample"]) < 3:
                st["errors_sample"].append(str(e.get("error"))[-600:])
            for k, v in (e.get("counts") or {}).items():
                if str(k).startswith("error/"):
                    st["error_types"][str(k)[6:]] += int(v)
            if seg is not None:
                seg.error = True
            return
        if seg is None:  # a segment that stored no row
            return
        last = seg.pending[-1] if seg.pending else None
        if last is None or last.t != int(e["t_last"]):
            raise RuntimeError(
                f"segment {e['uid']}: end after row {e['t_last']}, have {last and last.t}"
            )
        if out == "bootstrap":
            if not last.boot:
                raise RuntimeError(f"segment {e['uid']}: a bootstrap end without a bootstrap row")
            seg.boot_end = True
            last.reward = 0.0
        else:
            last.reward = float(e["r_last"])
            seg.r_sum += last.reward
        seg.ended = True
        # T4 at run time: the rows' rewards add up to the env's total for the segment.
        if abs(seg.r_sum - float(e.get("total_reward", 0.0))) > 1e-9:
            st["reward_mismatch"] += 1
            if len(st["reward_mismatch_sample"]) < 3:
                st["reward_mismatch_sample"].append(
                    {"uid": seg.uid, "rows": seg.r_sum, "env": e.get("total_reward")}
                )
        if seg.kind == "practice":
            st["practice"] += 1
            st["practice_cells"][str(e.get("cell"))] += 1
            return
        self.runs_done += 1
        st["runs"] += 1
        st["floor"].append(int(e["floor"]))
        st["won"] += int(e["won"])
        st["bosses"] += int(e["bosses"])
        st["boss2"] += int(e["reached_boss2"])
        if out == "capped":
            st["capped"] += 1
            st["caps"][str(cap)] += 1
            if cap in RUN_CAPS:
                st["runs_capped"] += 1
        if len(self.lc_first) < self.cfg.lc_window:
            self.lc_first.append(float(e["floor"]))
        else:
            self.lc_last.append(float(e["floor"]))

    # ── the cut ──────────────────────────────────────────────────────────

    def cut(self) -> tuple[dict[str, np.ndarray], list[Row], list[int], dict[str, int]]:
        """This collection's fresh table (WP4a `HoldBack.join`'s input), its rows, the dropped segments."""
        c = self.n_dispatched
        fresh: list[tuple[Row, bool, bool]] = []  # (row, done, bootstrap)
        drop: list[int] = []
        ast: Counter = Counter()
        for uid in list(self.segs):
            seg = self.segs[uid]
            if seg.error:
                ast["dropped_error_rows"] += len(seg.pending)
                drop.append(uid)
                del self.segs[uid]
                continue
            p = seg.pending
            if seg.ended:
                for r in p[:-1]:
                    fresh.append((r, False, False))
                fresh.append((p[-1], True, seg.boot_end))
                del self.segs[uid]
                ast["segments_ended"] += 1
                continue
            if p[-1].it < c:
                ast["dormant_segments"] += 1
                continue
            for r in p[:-1]:
                fresh.append((r, False, False))
            fresh.append((p[-1], False, True))
            seg.pending = [p[-1]]
            ast["segments_open"] += 1
        n = len(fresh)
        rows = [x[0] for x in fresh]
        for r, _done, boot in fresh:
            if not boot and r.reward is None:
                raise RuntimeError(f"segment {r.uid}: row {r.t} has no reward at the cut")
        cols = {
            "run_uid": np.fromiter((r.uid for r in rows), np.int64, n),
            "t": np.fromiter((r.t for r in rows), np.int32, n),
            "is_combat": np.fromiter((r.comb for r in rows), bool, n),
            "dtype": np.fromiter((r.dtype for r in rows), np.uint8, n),
            "reward": np.fromiter(
                (0.0 if (b or r.reward is None) else r.reward for r, _d, b in fresh), np.float64, n
            ),
            "done": np.fromiter((d for _r, d, _b in fresh), bool, n),
            "bootstrap": np.fromiter((b for _r, _d, b in fresh), bool, n),
            "actor": np.fromiter((r.actor for r in rows), np.uint8, n),
            "logp_behav": np.fromiter((r.logp for r in rows), np.float32, n),
            "n_classes": np.fromiter((r.ncls for r in rows), np.uint16, n),
            "action": np.fromiter((r.action for r in rows), np.uint16, n),
            "iter": np.fromiter((r.it for r in rows), np.int32, n),
            "v_behav": np.fromiter((r.v for r in rows), np.float32, n),
            "cell": np.fromiter((r.cell for r in rows), np.uint8, n),
            "ref": np.fromiter((r.ref for r in rows), np.int64, n),
            "n_tok": np.fromiter((r.ntok for r in rows), np.int32, n),
            "has_t": np.fromiter((r.has_t for r in rows), bool, n),
            "version": np.fromiter((r.version for r in rows), np.int32, n),
        }
        ast["fresh_rows"] = n
        ast["fresh_bootstrap"] = int(cols["bootstrap"].sum())
        ast["open_segments"] = len(self.segs)
        return cols, rows, drop, dict(ast)

    # ── hand-off to the update ───────────────────────────────────────────

    def _learner_state(self) -> dict[str, Any]:
        """The learner's counters as of the AlgoState it is saved with."""
        return {
            "it": int(self.algo.clocks.it),
            "decisions": int(self.algo.clocks.decisions),
            "runs_done": self.runs_done,
            "version": self.version,
            "next_ref": self.next_ref,
            "elapsed": self.elapsed0 + (time.time() - self.t_session),
            "lc_first": list(self.lc_first),
            "lc_last": list(self.lc_last),
            "totals": dict(self.totals),
            "seed": self.cfg.seed,
            "sampler_seed_offset": self.cfg.sampler_seed_offset,
            "time": float(self.now()),
        }

    def dispatch(self, cst: dict[str, Any]) -> None:
        tc = time.time()
        fresh, rows, drop, ast = self.cut()
        lost = self.algo.hold.drop(self.lost_uids) if self.lost_uids else 0
        ast["held_lost"], self.held_lost = self.held_lost + lost, 0
        drop = sorted(set(drop) | set(self.lost_uids))
        self.lost_uids = []
        if drop:
            self.algo.hold.drop(drop)
        table = self.algo.join(fresh)
        refs = np.asarray(table["ref"], np.int64)
        by_ref = {r.ref: r for r in rows}
        joined: list[Row] = []
        held: dict[int, Row] = {}
        for x in refs.tolist():
            r = by_ref.get(x)
            if r is None:
                r = self.held.get(x)
                if r is None:
                    raise RuntimeError(f"held row ref {x} has no tokens")
                held[x] = r
            joined.append(r)
        self.held = held  # less the rows a drop left out
        # Where each shipped row's inputs are now: the copy reads these chunks, which
        # nothing writes to again (serving fills only the round it serves).
        src = [(r.chunk, r.i) for r in joined]
        # The rows that will read the block: all but an open segment's newest row,
        # which stays in its round (it ships again next iteration, and would keep this
        # block alive until then).
        pending = {id(r) for s in self.segs.values() for r in s.pending}
        move = [k for k, r in enumerate(joined) if id(r) not in pending]
        c = self.n_dispatched
        job: dict[str, Any] = {
            "it": c,
            "n_collected": self.n_new,
            "t_wall": float(self.now()),
            "fresh": fresh,
            "refs": refs,
            "drop_uids": np.asarray(drop, np.int64),
            "stamps": self.stamps,
            "pack_held": True,  # rank 0 copies the rows it holds back (`_land_held`)
            "owner": os.getpid(),  # names rank 0's held block (`rowstore.shm_name`)
        }
        self.n_new = 0
        self.n_dispatched += 1
        ast["held_rows"] = len(refs) - len(rows)
        ast["table_rows"] = len(refs)
        # The first table of a start goes through the fill check before it trains:
        # its copy runs inline, as every copy does without `async_pack`.
        inline = self._fill_pending or not self.cfg.async_pack
        fut: Future = Future()
        if inline:
            fut.set_result(pack_rows(src, table, self.cfg.pack_threads))
        else:
            if self._packer is None:
                self._packer = ThreadPoolExecutor(1, thread_name_prefix="e2e-pack")
            fut = self._packer.submit(pack_rows, src, table, self.cfg.pack_threads)
        self.packing = {
            "fut": fut,
            "job": job,
            "rows": joined,
            "move": move,
            "cst": cst,
            "ast": ast,
        }
        ast["cut_s"] = time.time() - tc  # the serving loop's time on the cut
        if inline:
            self._pack_done(block=True)

    def _pack_done(self, *, block: bool) -> bool:
        """Submit the update of the last cut once its row block is copied (`block`:
        wait for the copy). The shipped rows then read the block, so their serving
        rounds and the last held block go. Returns whether it submitted one."""
        pk = self.packing
        if pk is None or (not block and not pk["fut"].done()):
            return False
        self.packing = None
        tw = time.time()
        shm, layout, pack_s = pk["fut"].result()
        ast = pk["ast"]
        ast["pack_s"] = pack_s
        ast["pack_wait_s"] = time.time() - tw
        try:
            blk = ShmChunk(shm, layout)
        except BaseException:
            shm.unlink()
            raise
        fill = None
        if self._fill_pending:
            try:
                fill = self._fill_check(blk.rb)
            except BaseException:
                blk.unlink()
                raise
        job, joined = pk["job"], pk["rows"]
        job["shm"], job["layout"] = shm.name, layout
        self.inflight = {
            "job": job,
            "block": blk,
            "rows": joined,
            "cst": pk["cst"],
            "ast": ast,
            "fill": fill,
        }
        for k in pk["move"]:
            r = joined[k]
            r.chunk, r.i = blk, k
        ast["submit_t"] = time.time()
        self.updater.submit(job)
        return True

    def _fill_check(self, rb: dict[str, np.ndarray]) -> dict[str, Any]:
        """The fill check of a table against `--filled` (A4): FILL_MISMATCH on a problem
        that holds on any sample; an inconclusive table leaves it for the next one.
        Before a resume's first table, every held row was dropped, so the table is this
        start's own rows."""
        from . import fillcheck as FC

        try:
            fc = FC.check(rb, self.cfg.filled)
            bad = fc.definite()
        except C.SchemaError as e:
            fc, bad = None, (str(e),)
        summary = fc.summary() if fc is not None else ""
        if bad:
            self._guard(
                "FILL_MISMATCH",
                "; ".join(bad),
                filled=sorted(self.cfg.filled),
                summary=summary,
            )
        assert fc is not None
        self._fill_pending = fc.inconclusive
        print(f"[e2e] fill check{' (inconclusive)' if fc.inconclusive else ''}: {summary}")
        return {"ok": fc.ok, "inconclusive": fc.inconclusive, "summary": summary}

    def _poll(self, block: bool) -> bool:
        tw = time.time()
        res = self.updater.poll(block)
        if res is None:
            return False
        inf, self.inflight = self.inflight, None
        assert inf is not None
        # The serving loop's wait for this update (0 when it was already done), and
        # its wall time from submission to here (ranks plus the hand-offs).
        inf["ast"]["poll_wait_s"] = time.time() - tw if block else 0.0
        inf["ast"]["update_wall_s"] = time.time() - inf["ast"].pop("submit_t", tw)
        inf["block"].unlink()  # the ranks are done with it; the rows in it keep it mapped
        t0 = time.time()
        held = _map_held(res.get("held"))  # rank 0's copy of the held rows; its name goes
        map_s = time.time() - t0
        if res.get("nonfinite"):
            self._nan(res, inf)
        before = int(self.algo.clocks.decisions)
        self.net_inf.load_state_dict({k: torch.from_numpy(v) for k, v in res["net"].items()})
        self.version += 1
        self.algo = A.AlgoState.from_dict(res["algo"])
        t1 = time.time()
        self._land_held(held, res.get("held"), inf)
        inf["ast"]["held_land_s"] = map_s + (time.time() - t1)  # the learner's time on them
        self._log(inf, res)
        self._save_due(before, int(self.algo.clocks.decisions))
        return True

    def _land_held(
        self, ch: ShmChunk | None, held: Mapping[str, Any] | None, inf: Mapping[str, Any]
    ) -> None:
        """Make the update's held rows the held rows, pointing at rank 0's copy of
        them (`ch`: rows `held["pos"]` of the update's block, in that order)."""
        joined, block = inf["rows"], inf["block"]
        self.held = {}
        if ch is not None:
            assert held is not None
            for j, k in enumerate(np.asarray(held["pos"], np.int64).tolist()):
                r = joined[k]
                if r.chunk is not block or r.i != k:
                    raise RuntimeError(f"held row {r.ref} is not row {k} of its table's block")
                r.chunk, r.i = ch, j
                self.held[r.ref] = r
        hold = self.algo.hold.cols["ref"] if len(self.algo.hold) else np.zeros(0, np.int64)
        got = np.fromiter(self.held, np.int64, len(self.held))
        if not np.array_equal(np.sort(got), np.sort(np.asarray(hold, np.int64))):
            raise RuntimeError(
                f"rank 0 copied {len(got)} held rows, the hold-back keeps {len(hold)}"
            )

    # ── states ───────────────────────────────────────────────────────────

    def _save_due(self, before: int, after: int) -> None:
        cfg = self.cfg
        pins = []
        if ckpt.crossed(before, after, cfg.pin_every):
            pins.append("milestone")
        pins += [r for r, at in self.k_gates if before < at <= after]
        if self.sd.p(OC.PIN_REQUEST).exists():
            pins.append("request")
        job = {
            "state": ckpt.crossed(before, after, cfg.state_every),
            "full": ckpt.crossed(before, after, cfg.full_every),
            "slim": ckpt.crossed(before, after, cfg.slim_every),
            "pins": pins,
        }
        if job["state"] or job["full"] or job["slim"] or pins:
            self.save(**job)
        if "request" in pins:
            self.sd.p(OC.PIN_REQUEST).unlink(missing_ok=True)

    def save(
        self,
        *,
        state: bool = False,
        full: bool = False,
        slim: bool = False,
        pins: Sequence[str] = (),
    ) -> list[str]:
        """Have rank 0 write the states asked for. They leave out the held rows'
        inputs: a resume drops every held row (`HoldBack.resume`)."""
        job: dict[str, Any] = {
            "state": state,
            "full": full,
            "slim": slim,
            "pins": list(pins),
            "save_dir": str(self.sd.root),
            "full_keep": self.cfg.full_keep,
            "learner_state": self._learner_state(),
            "stamps": self.stamps,
            "format": STATE_FORMAT,
            "held": None,
        }
        written = self.updater.save(job)
        if written:
            print(f"[e2e] saved {', '.join(os.path.basename(p) for p in written)}", flush=True)
        return written

    # ── stops, guards, logs ──────────────────────────────────────────────

    def _stop_wanted(self, *, every_s: float = 1.0) -> bool:
        now = time.time()
        if not self._stop_req and now - self._t_done >= every_s:
            self._t_done = now
            if self.sd.p(OC.DONE).exists():
                self._stop_req = "DONE"
        return bool(self._stop_req)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self._stop_req = signal.Signals(signum).name

    def _nan(self, res: dict[str, Any], inf: dict[str, Any]) -> None:
        rec = {
            "it": int(inf["job"]["it"]),
            "decisions": int(self.algo.clocks.decisions + inf["job"]["n_collected"]),
            "what": list(res["nonfinite"]),
            "sampler_seed_offset": self.cfg.sampler_seed_offset,
            "time": OC.utc_now_iso(),
        }
        OC.atomic_write_text(self.sd.p(OC.NAN_RECORD), json.dumps(rec) + "\n")
        raise NanStop(json.dumps(rec))

    def _f9_note(self, e: dict[str, Any], hung: bool) -> dict[str, Any]:
        """Append a segment with F9 hits to `ops/f9_hits.jsonl`: seed, cell and the first
        violation texts replay it. `dropped` says it was a hung segment (`_f9_hung`)."""
        rec = {
            "time": time.time(),
            "it": int(self.n_dispatched),
            "dropped": bool(hung),
            "hits": int(e.get("f9_violations", 0)),
            "checks": int(e.get("f9_checks", 0)),
            **{k: e.get(k) for k in ("uid", "seed", "kind", "cell", "outcome", "cap", "steps")},
            **{k: e.get(k) for k in ("rows", "floor", "act", "wheel")},
            "samples": list(e.get("f9_samples") or []),
            "counts": e.get("counts"),
        }
        try:
            p = Path(self.sd.p(OC.F9_HITS))
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(rec)) + "\n")
        except OSError as ex:  # a full disk must not hide the hit: the alarm still carries it
            print(f"[learner] f9_hits.jsonl not written: {ex}", flush=True)
        return rec

    def _guard(self, name: str, reason: str, **details: Any) -> None:
        OC.write_alarm(self.sd.root, name, writer="learner", reason=reason, **details)
        raise GuardStop(name, reason)

    def _worker_mismatch(self, ends: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        from ppo.wheel_guard import worker_source_mismatch, worker_wheel_mismatch

        return (
            worker_wheel_mismatch(ends, self.stamps["wheel"]) or None,
            worker_source_mismatch(ends, self.stamps["py_src"]) or None,
        )

    def _stats_row(self, inf: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
        cst, ast = inf["cst"], inf["ast"]
        ust = res["stats"]
        now = time.time()
        floor = cst["floor"]
        lr = {"trunk": float(ust["lr_trunk"]), "value": float(ust["lr_value"])}
        nan = math.nan

        def num(x: Any) -> float:
            return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else nan

        row = {
            # ── the ops contract's fields (ops_contract.STATS_FIELDS) ──
            "it": int(self.algo.clocks.it),
            "decisions": int(self.algo.clocks.decisions),
            "time": now,
            "pid": os.getpid(),
            "dec_per_s": cst["rows"] / max(1e-9, now - self._t_row),
            "epochs": int(self.cfg.epochs),
            "staleness": {
                "mean": ust.get("staleness_mean", 0.0),
                "max": ust.get("staleness_max", 0),
                "fresh_mean": ust.get("staleness_fresh_mean", 0.0),
                "fresh_max": ust.get("staleness_fresh_max", 0),
            },
            "kl": dict(ust.get("kl_t") or {}),
            "clipfrac": {"all": num(ust.get("clipfrac"))},
            "is_tails": dict(ust.get("is_tails") or {}),
            "entropy": dict(ust.get("ent_norm_t") or {}),
            "value_ev": {"combat": num(ust.get("ev_combat")), "run": num(ust.get("ev_run"))},
            "agreement": {k: cst["agree"][k] / v for k, v in cst["agree_n"].items() if v},
            "lambda_ks": float(ust["lam_ks"]),
            "beta": {"combat": float(ust["beta_combat"]), "ooc": float(ust["beta_ooc"])},
            "lr": lr,
            "lr_closed_form": closed_form_lr(ust, self.algo.cfg),
            "caps": {
                "fights": cst["fights"],
                "fights_capped": cst["fights_capped"],
                "runs": cst["runs"],
                "runs_capped": cst["runs_capped"],
            },
            "f9": {
                "checked": cst["f9_checks"],
                "hits": cst["f9_violations"],
                "tolerated": cst["f9_tolerated"],
                "hung_hits": cst["f9_hung_hits"],
                "hung_segments": cst["f9_hung_segments"],
            },
            "losses": {
                "policy": num(ust.get("ploss")),
                "value": num(ust.get("vloss")),
                "ks": num(ust.get("ks_loss")),
                "total": num(ust.get("loss_total")),
            },
            "grad_norm": num(ust.get("gradnorm_mean")),
            "nonfinite": int(bool(res.get("nonfinite"))),
            "sampler_seed_offset": int(self.cfg.sampler_seed_offset),
            "wheel_mismatch": cst.get("wheel_mismatch"),
            "source_mismatch": cst.get("source_mismatch"),
            "stamps": dict(self.stamps),
            # ── the update ──
            "update": {k: v for k, v in ust.items() if k not in ("kl_t", "ent_norm_t")},
            # ── collection ──
            "collection": cst["collection"],
            "sampler_seed": cst["sampler_seed"],
            "runs_done": self.runs_done,
            "collect_s": cst["collect_s"],
            "fwd_s": cst["fwd_s"],
            "wait_s": cst["wait_s"],
            "recv_s": cst["recv_s"],
            "policy_swaps": cst["policy_swaps"],
            "batch_mean": cst["batch_rows"] / max(1, cst["batches"]),
            "rows": cst["rows"],
            "boot_rows": cst["boot_rows"],
            "rows_type": dict(cst["rows_type"]),
            "teacher_rows": cst["teacher_rows"],
            "runs": cst["runs"],
            "practice_segments": cst["practice"],
            "practice_cells": dict(cst["practice_cells"]),
            "floor_mean": float(np.mean(floor)) if floor else nan,
            "win": cst["won"] / max(1, cst["runs"]),
            "bosses_per_run": cst["bosses"] / max(1, cst["runs"]),
            "reached_boss2": cst["boss2"],
            "outcomes": dict(cst["outcomes"]),
            "capped": cst["capped"],
            "cap_reasons": dict(cst["caps"]),
            "errors": cst["errors"],
            "errors_sample": cst["errors_sample"],
            "error_types": dict(cst["error_types"]),
            "segment_counts": dict(cst["segment_counts"]),
            "reward_mismatch": cst["reward_mismatch"],
            "reward_mismatch_sample": cst["reward_mismatch_sample"],
            "unknown": cst["unknown"],
            "assembly": ast,
            "totals": dict(self.totals),
            "workers": self.workers_stats if self.workers_stats else None,
        }
        if self.lc_applied is not None:
            row["lc_verdict_pass"] = self.lc_applied
        if inf.get("fill") is not None:
            row["fill_check"] = inf["fill"]
        self.workers_stats = {}
        return row

    def _log(self, inf: dict[str, Any], res: dict[str, Any]) -> None:
        cst = inf["cst"]
        ust = res["stats"]
        for k in ("runs", "capped", "errors", "unknown", "f9_violations", "fights"):
            self.totals[k] += cst[k]
        row = self._stats_row(inf, res)
        self._t_row = row["time"]
        row = _jsonable(row)
        with open(self.sd.p(OC.STATS), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.stats_rows.append(row)
        print(
            f"[e2e] it {row['it']} {ust['phase']} dec {row['decisions']:,} "
            f"{row['dec_per_s']:.0f}/s runs {row['runs']} floor {row['floor_mean']} "
            f"win {row['win']:.3f} vloss {row['losses']['value']} "
            f"ploss {row['losses']['policy']} kl {ust.get('probe_kl')} "
            f"ev {row['value_ev']['combat']}/{row['value_ev']['run']} "
            f"ep {ust['epochs_run']} m {ust['kl_m']:.2f} lr {ust['lr_trunk']:.2e} "
            f"rows {ust['rows_trained']} held {ust['rows_held']} upd {ust['rank_s']:.1f}s",
            flush=True,
        )
        errs = self.lr_watch.check(ust)
        if errs:
            self._guard("LR_CLOSED_FORM", json.dumps({"it": ust["it"], "problems": errs}))
        if cst["f9_violations"] > OC.F9_MAX_HITS:
            self._guard(
                "F9_INVARIANT",
                json.dumps({"it": ust["it"], "violations": cst["f9_violations"]}),
                samples=_jsonable(cst["f9_stop_samples"]),
            )

    def _ops_tick(self) -> None:
        now = time.time()
        cfg = self.cfg
        if now - self._t_hb >= cfg.heartbeat_s:
            hb = {
                "time": now,
                "pid": os.getpid(),
                "it": self.n_dispatched,
                "decisions": self.decisions,
            }
            OC.atomic_write_text(self.sd.p(OC.HEARTBEAT), json.dumps(hb) + "\n")
            self._t_hb = now
        tp = self._tp
        if now - tp["t"] >= cfg.throughput_s:
            row = {
                "time": now,
                "pid": os.getpid(),
                "decisions": self.decisions,
                "rate": (self.decisions - tp["decisions"]) / (now - tp["t"]),
                "it": self.n_dispatched,
                "open_segments": len(self.segs),
                "held_rows": len(self.held),
                "evals_running": len(self.evals),
            }
            with open(self.sd.p(OC.THROUGHPUT), "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self._tp = {"t": now, "decisions": self.decisions, "wait": 0.0}
        if now - self._t_evals >= 5.0:
            self._t_evals = now
            self._evals_tick()

    # ── evals (ops contract item 8) ──────────────────────────────────────

    def _evals_tick(self) -> None:
        sd, cfg = self.sd, self.cfg
        for rid, (p, path) in list(self.evals.items()):
            rc = p.poll()
            if rc is None:
                continue
            req = OC.read_json(path) or {}
            done = sd.p(OC.EVAL_DONE)
            done.mkdir(parents=True, exist_ok=True)
            res = OC.read_json(Path(str(req["out"])) / "result.json") if req.get("out") else None
            extra = {k: v for k, v in res.items() if k not in req} if isinstance(res, dict) else {}
            rec = {**req, **extra, "rc": rc, "finished": OC.utc_now_iso(), "out": req.get("out")}
            OC.atomic_write_text(done / f"{rid}.json", json.dumps(rec, indent=1) + "\n")
            path.unlink(missing_ok=True)
            del self.evals[rid]
        if sd.p(OC.EVALS_PAUSE).exists():
            self._stop_evals(requeue=True)
            return
        if not cfg.eval_cmd:
            q = sd.p(OC.EVAL_QUEUE)
            if not self._eval_note and q.is_dir() and any(q.glob("*.json")):
                print("[e2e] eval requests queued; no --eval-cmd to run them", flush=True)
                self._eval_note = True
            return
        if self.evals:
            return
        q = sd.p(OC.EVAL_QUEUE)
        reqs = sorted(q.glob("*.json")) if q.is_dir() else []
        if not reqs:
            return
        running = sd.p(OC.EVAL_RUNNING)
        running.mkdir(parents=True, exist_ok=True)
        src = reqs[0]
        dst = running / src.name
        try:
            src.rename(dst)  # the claim
        except OSError:
            return
        req = OC.read_json(dst) or {}
        rid = str(req.get("id") or src.stem)
        out = sd.p(f"evals/out/{rid}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = {
            "request": str(dst),
            "state": req.get("state", ""),
            "variant": req.get("variant", "raw"),
            "panel": req.get("panel", ""),
            "n": req.get("n", 0),
            "seed_start": req.get("seed_start", 0),
            "max_procs": req.get("max_procs", OC.EVAL_MAX_PROCS),
            "out": str(out),
        }
        req["out"] = str(out)
        OC.atomic_write_text(dst, json.dumps(req, indent=1) + "\n")
        nice = int(req.get("nice", OC.EVAL_NICE))
        if cfg.eval_cmd == "builtin":
            line = [sys.executable, "-m", "ppo.e2e.eval_runner", str(dst), "--out", str(out)]
            cmd = eval_argv(line, nice, cfg.eval_args)
        else:
            cmd = eval_argv(cfg.eval_cmd.format(**fields), nice)
        if out.is_dir():
            (out / "result.json").unlink(missing_ok=True)  # a rerun files only its own result
        log = open(str(out) + ".log", "a", encoding="utf-8")
        env = {**os.environ, **OC.WORKER_THREAD_ENV}  # an eval is a worker too
        p = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        log.close()
        self.evals[rid] = (p, dst)

    def _stop_evals(self, *, requeue: bool) -> None:
        for rid, (p, path) in list(self.evals.items()):
            if p.poll() is None:
                stop_group(p, kill=False)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stop_group(p, kill=True)
                    p.wait()
            if requeue and path.exists():
                q = self.sd.p(OC.EVAL_QUEUE)
                q.mkdir(parents=True, exist_ok=True)
                path.rename(q / path.name)
            del self.evals[rid]

    # ── the learning check (plan §7) ─────────────────────────────────────

    def learning_check(self) -> dict[str, Any]:
        cfg = self.cfg
        first, last = self.lc_first, list(self.lc_last)
        rows = []
        p = self.sd.p(OC.STATS)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(x) for x in f if x.strip()]
        ev_stat = lc_combat_ev(rows)
        ev = ev_stat["median"]
        # A NaN a rollback discarded still fails the check: read the parked rows too.
        po = self.sd.p(OC.STATS_ORPHAN)
        parked = []
        if po.exists():
            parked = list(OC.iter_jsonl(po.read_text(encoding="utf-8").splitlines()))
        nan = any(r.get("nonfinite") for r in [*rows, *parked])
        full = len(first) >= cfg.lc_window and len(last) >= cfg.lc_window
        gain = float(np.mean(last) - np.mean(first)) if (first and last) else math.nan
        return {
            "pass": bool(full and gain >= cfg.lc_margin and ev > cfg.lc_ev and not nan),
            "window": cfg.lc_window,
            "complete": full,
            "first_runs": len(first),
            "last_runs": len(last),
            "first_mean": float(np.mean(first)) if first else math.nan,
            "last_mean": float(np.mean(last)) if last else math.nan,
            "gain": gain,
            "margin": cfg.lc_margin,
            "ev_combat": ev,  # the gated statistic: lc_combat_ev's median
            "ev_combat_rows": ev_stat["rows"],
            "ev_combat_last5": ev_stat["last5_mean"],
            "ev_min": cfg.lc_ev,
            "nan": nan,
            "decisions": int(self.algo.clocks.decisions),
            "it": int(self.algo.clocks.it),
            "stamps": self.stamps,
        }

    # ── the main loop ────────────────────────────────────────────────────

    def _limit(self) -> str:
        """Why the run ends now ("" = it goes on)."""
        cfg = self.cfg
        if self.algo.clocks.decisions >= self.algo.cfg.budget_decisions:
            return "budget"
        if cfg.max_iters > 0 and self.n_dispatched >= cfg.max_iters:
            return "max_iters"
        if cfg.max_wall_s > 0 and time.time() - self.t_session >= cfg.max_wall_s:
            return "max_wall_s"
        return ""

    def _check_batch(self, cst: dict[str, Any]) -> None:
        """Refuse a collection that a worker on another wheel or source took part in
        (its messages' stamps or its segment ends): ALARM before the batch trains."""
        wheel, src = self._worker_mismatch(cst["ends"] + cst["bad_stamps"])
        cst["wheel_mismatch"], cst["source_mismatch"] = wheel, src
        if wheel:
            self._guard("WHEEL_MISMATCH", wheel)
        if src:
            self._guard("SOURCE_MISMATCH", src)

    def train(self) -> str:
        """Run until the budget, a limit or a stop; returns why it ended."""
        cfg = self.cfg
        rec = _jsonable(
            asdict(cfg)
            | {
                "stamps": self.stamps,
                "resumed_from": self.resumed_from,
                "teachers": self.teacher_provenance(),
            }
        )
        OC.atomic_write_text(self.sd.p("cfg.json"), json.dumps(rec, indent=1) + "\n")
        old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        with contextlib.suppress(ValueError):  # not the main thread (tests)
            for s in old:
                signal.signal(s, self._on_signal)
        self.start()
        self._t_row = time.time()
        why = ""
        try:
            while not why:
                why = self._limit()
                if why or self._stop_wanted(every_s=0.0):
                    break
                cst = self.collect()
                if cst is None:
                    break
                self._check_batch(cst)
                self._pack_done(block=True)  # the last cut's update, if not yet submitted
                if self.inflight is not None:
                    self._poll(block=True)
                self.dispatch(cst)
                if cfg.sync_updates:
                    self._pack_done(block=True)
                    self._poll(block=True)
            self._pack_done(block=True)
            if self.inflight is not None:
                self._poll(block=True)
            why = why or self._limit() or self._stop_req or "stopped"
            pins = ["final"] if why in ("budget", "max_iters", "max_wall_s") else []
            self.save(state=True, full=False, slim=bool(pins), pins=pins)
            if cfg.mode == "learning_check":
                verdict = self.learning_check()
                OC.atomic_write_text(
                    self.sd.p("learning_check.json"), json.dumps(_jsonable(verdict), indent=1)
                )
                print(f"[e2e] learning check: {json.dumps(_jsonable(verdict))}", flush=True)
            if why in ("budget", "max_iters", "max_wall_s"):
                OC.write_done(
                    self.sd.root,
                    why,
                    by="learner",
                    it=int(self.algo.clocks.it),
                    decisions=int(self.algo.clocks.decisions),
                )
        finally:
            self.stop()
            with contextlib.suppress(ValueError):
                for s, h in old.items():
                    signal.signal(s, h)
        print(f"[e2e] stopped: {why}", flush=True)
        return why


# ── held rows in a state ─────────────────────────────────────────────────


def park_orphan_stats(stats: Path, orphan: Path, it: int) -> int:
    """Move the rows of `stats` past iteration `it` (and a torn line) to `orphan`.

    A stats row is written after its update and before that update's state, and
    state.pt only every `state_every` decisions (5M at launch: ~38 iterations of
    131,072 decisions), so a kill or a rollback leaves the rows of every
    iteration after the state it resumes from, up to ~38, which the resumed lane
    writes again with the same `it`. Every window rule on the rows (the driver's
    CAPS, E1_REVERT and HELD_ROW_LEAK, the learning check, the pre-flight counts)
    would read those twice. After this the file's `it` is unique and increasing.
    The parked rows are appended to `orphan` (fsynced) before `stats` is
    rewritten, so a kill in between duplicates rows there, never loses one.
    Returns how many rows moved."""
    if not stats.is_file():
        return 0
    keep: list[str] = []
    park: list[str] = []
    for line in stats.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:  # a torn last line: the next append would extend it
            park.append(line)
            continue
        n = row.get("it") if isinstance(row, dict) else None
        orphaned = isinstance(n, int) and not isinstance(n, bool) and n > it
        (park if orphaned else keep).append(line)
    if not park:
        return 0
    with open(orphan, "a", encoding="utf-8") as f:
        f.write("".join(x + "\n" for x in park))
        f.flush()
        os.fsync(f.fileno())
    OC.atomic_write_text(stats, "".join(x + "\n" for x in keep))
    return len(park)


def lost_runs(algo: A.AlgoState) -> list[int]:
    """The runs of every held row: at a resume all of them, since the open segments
    die with the old session's workers and their uids never recur."""
    return sorted({int(x) for x in algo.hold.cols["run_uid"]}) if len(algo.hold) else []


def held_from_state(h: dict[str, Any] | None) -> dict[int, Row]:
    """The held rows' inputs a full state carries (`update.held_blob`), by ref."""
    if not h:
        return {}
    ch = Chunk({k: v.numpy() for k, v in h["rows"].items()})
    refs = h["refs"].numpy().tolist()
    if len(refs) != ch.n:
        raise UsageError(f"held rows: {len(refs)} refs for {ch.n} rows")
    return {int(ref): Row.held(ref, ch, i) for i, ref in enumerate(refs)}


def pack_rows(
    src: Sequence[tuple[Chunk, int]], table: Mapping[str, np.ndarray], threads: int = 1
) -> tuple[shared_memory.SharedMemory, dict[str, Any], float]:
    """The update's row block of a cut: the rows `src` (the joined table's, in its
    order) copied into one `SharedMemory` block (`rowstore.gather_to_shm` on
    `threads` threads), the
    table's values written over theirs (`table_to_rows`), and the copy's wall time.
    It only reads `src`'s chunks and `table`, so it runs in the cut's thread."""
    t0 = time.time()
    shm, layout = gather_to_shm(src, threads=threads)
    try:
        rb = batch_from_shm(shm, layout)
        table_to_rows(rb, table)
        del rb
    except BaseException:
        shm.unlink()
        with contextlib.suppress(BufferError):
            shm.close()
        raise
    return shm, layout, time.time() - t0


def table_to_rows(rb: dict[str, np.ndarray], table: Mapping[str, np.ndarray]) -> None:
    """Write the joined table's per-row values into its row batch, in place. SCHEMA.md
    revision 2 (section 9.1) took WP4a's flags: `bootstrap` marks a value-only
    row, a cut's (stored again next iteration) or, with `done`, a practice
    segment's end; `done` alone a terminal. A bootstrap row's reward is 0 (a
    cut row's reward belongs to its next storing). The acting-time fields, the
    run uid, `t` and the cell come from the table too; the ranks rebuild the
    table themselves (`algo.join`) and train on its values."""
    if len(rb["n_tok"]) != len(table["ref"]):
        raise RuntimeError("the row batch and the table have different lengths")
    boot = np.asarray(table["bootstrap"], bool)
    for k in ("reward", "done", "bootstrap"):
        if rb[k].dtype != C.FIELDS[k].dtype:
            raise RuntimeError(f"row batch field {k} is {rb[k].dtype}, not {C.FIELDS[k].dtype}")
    rb["reward"][...] = np.where(boot, 0.0, np.asarray(table["reward"], np.float64))
    rb["done"][...] = np.asarray(table["done"], bool)
    rb["bootstrap"][...] = boot
    for k in ("actor", "logp_behav", "action", "iter", "v_behav", "run_uid", "t", "cell"):
        rb[k][...] = np.asarray(table[k]).astype(rb[k].dtype)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ── CLI ──────────────────────────────────────────────────────────────────


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    d = Cfg(save_dir="")
    p.add_argument(OC.FLAG_SAVE_DIR, dest="save_dir", required=True)
    p.add_argument(OC.FLAG_RESUME, dest="resume", default="", help="a full state to resume from")
    p.add_argument(OC.FLAG_SEED_OFFSET, dest="sampler_seed_offset", type=int, default=0)
    p.add_argument("--updater", default="process", choices=("process", "local"))
    for f in dataclasses.fields(Cfg):
        if f.name in ("save_dir", "resume", "sampler_seed_offset"):
            continue
        val = getattr(d, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(val, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=val)
        elif isinstance(val, (int, float, str)):
            p.add_argument(flag, type=type(val), default=val)
        else:
            p.add_argument(flag, type=json.loads, default=val, help="JSON")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ppo.e2e.learner", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_cfg_args(sub.add_parser(OC.LEARNER_SUBCOMMAND, help="train a lane"))
    ex = sub.add_parser("export", help="write one variant of a state as a WP4b net checkpoint")
    ex.add_argument("--state", required=True)
    ex.add_argument("--variant", default="raw", help="raw, ema0996, ema0999")
    ex.add_argument("--out", required=True)
    return p


def cfg_from_args(args: argparse.Namespace) -> Cfg:
    return Cfg(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(Cfg) if hasattr(args, f.name)}
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.cmd == "export":
        from .update import export_variant

        print(export_variant(args.state, args.variant, args.out))
        return OC.EXIT_OK
    try:
        cfg = cfg_from_args(args)
        learner = Learner(cfg, updater=args.updater, argv=[sys.argv[0], *argv])
        learner.train()
    except UsageError as e:
        print(f"[e2e] refused: {e}", file=sys.stderr, flush=True)
        return OC.EXIT_USAGE
    except NanStop as e:
        print(f"[e2e] non-finite update, nothing saved: {e}", file=sys.stderr, flush=True)
        return OC.EXIT_NAN
    except GuardStop as e:
        print(f"[e2e] guard: {e}", file=sys.stderr, flush=True)
        return OC.EXIT_HUMAN
    return OC.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
