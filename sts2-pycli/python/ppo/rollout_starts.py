"""Mid-fight playout starts for `rollout_learn rl --starts mid` (GH #777, round 2).

`rollout_learn rl` trained the playout net from fight STARTS only, but the
search runs playouts from every decision of a fight. This module records where
the deployed pilot actually goes, so training can start anywhere it goes:

    traj   play human-bank fights with the deployed pilot (`MctsCombatAgent`,
           honest by default, `ppo.combat_bench.agents` env handling) and
           write, per (fight, rep), the seed and the run-action sequence. Any
           decision point is then REBUILT deterministically: build the fight
           from the same seed (`build_fight`) and replay the actions on a
           fresh `RunManager`. Each decision's `CombatState.ordered_hash()`
           (the structural hash plus the order of every card pile, so the
           draw order is checked too) is recorded, and a rebuild that does
           not hash the same raises instead of training on the wrong state.

A record (JSONL; the first line is a header carrying the pilot spec, its
engine env and the engine's `src_hash`) holds `key`, `file`, `rep`,
`seed`, `room`, `encounter`, `act`, the bank row (`fight`), `actions`
(run-action dicts), `hashes`, `searched` (False for a forced move),
`q_best` (the pilot's own search: Q of its recommended move, the search's
root value estimate; None on a forced move), `end` (`won` / `died` /
`step_cap` / `no_action` / `error`), `value` (what the search backs up for
the finished fight, `sts2_rs.search_playout_value`; None unless it ended
`won` / `died`) and, with `--branch-reps`, `branch` (`{t: [values]}` of
extra pilot continuations from decision t, played on a clone with another
agent seed).

The states rebuilt here are the TRUE combat states (actual draw order and
RNG). `rl --view belief` (the default for `--starts mid`) hands
`LearnedPlayout.play_many(belief=True)` those states, and each playout then
starts from its own `belief_root` of the state, the honest view the deployed
search runs its playouts from. The pilot targets (`value`, `branch`) are
outcomes from the true state; averaged over states they estimate the same
expectation the belief playouts do.

`value`, `branch` and `q_best` are search values (`sts2_mcts::playout_return`
units), and those depend on the engine env the pilot workers ran under
(`STS2_RUNAI_TERMINAL_MODE`, `_LOSS_GRADIENT`, the `loss_w_*` override
knobs, ...). `rl` therefore refuses trajectories whose header env differs
from its own in anything but the pilot's playout net (`check_value_env`).

Seeds: `_fight_seed(key) * 100 + rep0 + rep` with `--rep0 80` by default. The
combat bench owns reps 0-49, `gen` 50+ (default 50) and the old `rl` fight
starts 77.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from multiprocessing import get_context
from typing import Any

from ppo.combat_bench import agents
from ppo.combat_bench.holdout import subtract_bench_runs
from ppo.pilot_human_decks import _fight_seed, fight_character

# v2: `hashes` are `ordered_hash()` (v1 recorded the order-free `state_hash()`)
# and the header records the engine's `src_hash`.
SCHEMA = "rollout_learn.traj.v2"
DEFAULT_REP0 = 80
SEED_STRIDE = 100
# Engine env vars that choose the pilot's PLAYOUT policy, not the search
# value a finished playout is scored with: two runs that differ only in
# these still report values in the same units.
PILOT_ONLY_ENV = frozenset({"STS2_ROLLOUT_NET", "STS2_ROLLOUT_TEMP", "STS2_ROLLOUT_NO_EARLY_END"})


def fight_key(fight: dict[str, Any]) -> str:
    return f"{fight['file']}:{fight['floor']}"


def start_id(key: str, seed: int, t: int, ohash: int) -> str:
    """The id of the position at decision `t` of fight `key` built from `seed`.

    It names the position itself (`ohash` is its `ordered_hash()`), not just
    where it came from: two trajectories with another `--rep0`, or another
    pilot, never share an id unless they reached the same position.
    """
    return f"{key}|{seed}|{t}|{ohash:016x}"


def ordered_hash(cs: Any) -> int:
    return int(cs.ordered_hash())


def traj_seed(key: str, rep: int, rep0: int = DEFAULT_REP0) -> int:
    """The seed of trajectory (`key`, `rep`).

    Raises:
        ValueError: When `rep0 + rep` leaves the per-fight seed stride.
    """
    if not 0 <= rep0 + rep < SEED_STRIDE:
        raise ValueError(f"rep0 + rep = {rep0 + rep} outside [0, {SEED_STRIDE})")
    return _fight_seed(key) * SEED_STRIDE + rep0 + rep


def build_fight(fight: dict[str, Any], seed: int, game_version: str | None = None) -> Any:
    """`fight` as a live `RunManager` in combat, built from `seed`.

    The same build as `rollout_learn gen` / the old `rl` starts /
    `pilot_human_decks.play_fight` (belt cut to 3 potions) for a human-bank
    row; a combat-bench row's `monsters` and `potion_slots` are honoured as
    `combat_bench.run.build_run` does. The combat log stays off, so neither
    the states nor their playouts carry one.

    The fight simulates `game_version` (default: the selected
    `sts2_client.GAME_VERSION`, v0.107.1 unless `STS2_GAME_VERSION` says
    otherwise), and the state and the run are read back to prove it.

    Raises:
        sts2_client.GameVersionMismatch: The engine built another version, or
            `game_version` names no supported build.
    """
    import sts2_client
    import sts2_rs

    ver = (
        sts2_client.require_game_version(game_version) if game_version else sts2_client.GAME_VERSION
    )
    character = fight_character(fight)
    slots = fight.get("potion_slots")
    potions = list(fight["potions"]) if slots is not None else list(fight["potions"])[:3]
    state = sts2_rs.build_scenario_state(
        seed,
        character,
        list(fight["deck"]),
        int(fight["hp_before"]),
        int(fight["max_hp"]),
        list(fight["relics"]),
        fight["encounter"],
        ascension=int(fight["ascension"]),
        potions=potions,
        monsters=fight.get("monsters"),
        potion_slots=slots,
        game_version=ver,
    )
    rm = sts2_rs.RunManager(
        seed=seed, character=character, ascension_level=int(fight["ascension"]), game_version=ver
    )
    # `debug_enter_combat` takes any state without comparing builds.
    sts2_client.check_game_version(state, ver, what="build_fight combat state")
    sts2_client.check_game_version(rm, ver, what="build_fight RunManager")
    rm.debug_enter_combat(state, fight["room_type"])
    return rm


def in_combat(rm: Any) -> bool:
    return rm.phase() == "combat" and not rm.is_over()


def searched(rec: dict[str, Any]) -> bool:
    """Whether a step record ran a real search (a forced move has 0 visits)."""
    search = rec.get("combat_search")
    return bool(search) and sum(a.get("visits", 0) for a in search.get("actions", ())) > 0


def q_best(rec: dict[str, Any]) -> float | None:
    """Q of the search's recommended move: its estimate of the position's value.

    Not the visit-weighted mean over the root: at the deployed c_puct the
    root visits are near-uniform (GH #442), so that mean is close to the
    average move's value, not the best one's.
    """
    if not searched(rec):
        return None
    search = rec["combat_search"]
    rec_action = json.dumps(search.get("recommended"), sort_keys=True)
    best = None
    for a in search["actions"]:
        if json.dumps(a["action"], sort_keys=True) == rec_action:
            return float(a["q"])
        if a.get("is_best"):
            best = float(a["q"])
    return best


def final_value(rm: Any) -> float | None:
    """The search's value of the finished fight, or None if it did not finish."""
    import sts2_rs

    cs = rm.combat_state()
    # A finished fight has no legal actions left (`legal_actions` is empty
    # exactly when the state is terminal).
    if cs is None or cs.legal_actions():
        return None
    return float(sts2_rs.search_playout_value(cs))


def _branch_seed(seed: int, t: int, k: int) -> int:
    # `MctsCombatAgent` takes an i32 seed.
    return (seed * 7919 + (t + 1) * 104_729 + k * 1_299_709) % (2**31 - 1)


def play_out(rm: Any, agent: Any, max_steps: int) -> str:
    """Drive `agent` on `rm` to the end of the fight; the end status."""
    steps = 0
    while in_combat(rm):
        if steps >= max_steps:
            return "step_cap"
        if agent.step(rm) is None:
            return "no_action"
        steps += 1
    return "died" if rm.is_over() and not rm.player_won else "won"


def traj_fight(job: tuple[Any, ...]) -> dict[str, Any]:
    """Play one (fight, rep) with the pilot and record it (a Pool worker)."""
    fight, rep, rep0, spec, max_steps, branch_reps, branch_frac = job
    key = fight_key(fight)
    seed = traj_seed(key, rep, rep0)
    out: dict[str, Any] = {
        "key": key,
        "file": fight["file"],
        "rep": rep,
        "seed": seed,
        "room": fight["room_type"],
        "encounter": fight["encounter"],
        "act": fight.get("act"),
        "fight": fight,
    }
    actions: list[Any] = []
    hashes: list[int] = []
    flags: list[bool] = []
    qs: list[float | None] = []
    branch_rms: list[tuple[int, Any]] = []
    rng = random.Random(seed)
    t0 = time.process_time()
    try:
        rm = build_fight(fight, seed)
        agent = agents.make_agent(spec, seed)
        end = None
        while in_combat(rm):
            if len(actions) >= max_steps:
                end = "step_cap"
                break
            cs = rm.combat_state()
            h = ordered_hash(cs)
            pre = rm.clone() if branch_reps and rng.random() < branch_frac else None
            rec = agent.step(rm)
            if rec is None:
                end = "no_action"
                break
            if pre is not None and searched(rec):
                branch_rms.append((len(actions), pre))
            hashes.append(h)
            actions.append(rec["action"])
            flags.append(searched(rec))
            qs.append(q_best(rec))
        if end is None:
            end = "died" if rm.is_over() and not rm.player_won else "won"
        out["end"] = end
        out["won"] = end == "won"
        out["value"] = final_value(rm) if end in ("won", "died") else None
        out["hp_end"] = int(rm.current_hp)
        if branch_rms:
            branch: dict[str, list[float]] = {}
            for t, brm in branch_rms:
                vals = []
                for k in range(branch_reps):
                    b = brm.clone()
                    bagent = agents.make_agent(spec, _branch_seed(seed, t, k))
                    if play_out(b, bagent, max_steps) in ("won", "died"):
                        v = final_value(b)
                        if v is not None:
                            vals.append(v)
                branch[str(t)] = vals
            out["branch"] = branch
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a PyO3 panic is a BaseException
        out["end"] = "error"
        out["error"] = repr(e)
    out["actions"] = actions
    out["hashes"] = hashes
    out["searched"] = flags
    out["q_best"] = qs
    out["cpu_s"] = round(time.process_time() - t0, 3)
    return out


def _worker_init(env: dict[str, str]) -> None:
    agents.apply_env(env)


def read_traj(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the records of a traj file.

    Raises:
        SystemExit: When the file does not start with a header of this
            `SCHEMA` (a v1 file's hashes are not the ones `rebuild` checks).
    """
    header = None
    recs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if header is None:
                if row.get("schema") != SCHEMA:
                    raise SystemExit(
                        f"{path}: schema {row.get('schema')!r}, expected {SCHEMA!r}; "
                        "record it again with `rollout_learn traj`"
                    )
                header = row
            else:
                recs.append(row)
    if header is None:
        raise SystemExit(f"{path}: empty traj file")
    return header, recs


def read_done(path: str) -> tuple[dict[str, Any] | None, set[tuple[str, int]]]:
    """The header and the finished (key, rep) cells of a traj file."""
    if not os.path.exists(path):
        return None, set()
    header, recs = read_traj(path)
    return header, {(r["key"], int(r["rep"])) for r in recs}


def select_fights(
    fights: list[dict[str, Any]],
    holdout_files: list[str] | None,
    holdout_encounters: list[str] | None,
    rooms: set[str],
    allow_bench_overlap: bool,
    tag: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    """Training fights: not held out, in `rooms`, combat-bench runs subtracted.

    Returns:
        `(fights, bench_meta, holdout_runs)`.
    """
    holdout: set[str] = set()
    for path in holdout_files or []:
        with open(path, encoding="utf-8") as f:
            holdout |= {json.loads(line)["file"] for line in f if line.strip()}
    if holdout_encounters:
        holdout |= {f["file"] for f in fights if f["encounter"] in holdout_encounters}
    pool = [f for f in fights if f["file"] not in holdout and f["room_type"] in rooms]
    pool, bench_meta = subtract_bench_runs(pool, lambda f: f["file"], allow_bench_overlap, tag=tag)
    return pool, bench_meta, holdout


def traj(args: argparse.Namespace) -> int:
    with open(args.fights, encoding="utf-8") as f:
        fights = [json.loads(line) for line in f if line.strip()]
    rooms = set(args.rooms.split(","))
    pool, bench_meta, holdout = select_fights(
        fights,
        args.holdout,
        args.holdout_encounter,
        rooms,
        args.allow_bench_overlap,
        "[rollout_learn traj]",
    )
    rng = random.Random(args.sample_seed)
    rng.shuffle(pool)
    if args.max_fights:
        pool = pool[: args.max_fights]
    env = agents.build_env(
        agents.parse_kv(args.env or [], "--env"),
        overrides=args.overrides,
        rollout_net=args.rollout_net,
    )
    spec = agents.mcts_spec(args.sims, args.boss_sims, env)
    import sts2_rs

    header = {
        "schema": SCHEMA,
        "fights": os.path.abspath(args.fights),
        "rooms": sorted(rooms),
        "rep0": args.rep0,
        "reps": args.reps,
        "spec": spec,
        "engine": sts2_rs.src_hash(),
        "max_steps": args.max_steps,
        "branch_reps": args.branch_reps,
        "branch_frac": args.branch_frac,
        "holdout_runs": len(holdout),
        "bench_holdout": bench_meta,
    }
    old, done = read_done(args.out)
    if old is not None:
        diff = [
            k
            for k in ("spec", "engine", "rep0", "max_steps", "branch_reps")
            if old.get(k) != header[k]
        ]
        if diff:
            raise SystemExit(f"{args.out}: header differs in {diff}; use a new --out")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
    jobs = [
        (fight, rep, args.rep0, spec, args.max_steps, args.branch_reps, args.branch_frac)
        for fight in pool
        for rep in range(args.reps)
        if (fight_key(fight), rep) not in done
    ]
    print(
        f"[rollout_learn traj] {len(jobs)} trajectories to play ({len(done)} done, "
        f"{len(holdout)} held-out runs, rooms {sorted(rooms)})",
        flush=True,
    )
    t0 = time.time()
    n = n_err = n_dec = 0
    ends: dict[str, int] = {}
    with (
        get_context("spawn").Pool(args.procs, initializer=_worker_init, initargs=(env,)) as p,
        open(args.out, "a", encoding="utf-8") as out,
    ):
        for rec in p.imap_unordered(traj_fight, jobs):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            n_err += rec["end"] == "error"
            n_dec += sum(rec["searched"])
            ends[rec["end"]] = ends.get(rec["end"], 0) + 1
            if n % 50 == 0 or n == len(jobs):
                print(
                    f"[rollout_learn traj] {n}/{len(jobs)} trajectories, {n_dec} searched "
                    f"decisions, ends {ends}, {time.time() - t0:.0f}s",
                    flush=True,
                )
    print("DONE", flush=True)
    return 1 if n and n_err == n else 0


# ── reading trajectories back ────────────────────────────────────────────


def read_records(paths: list[str]) -> list[dict[str, Any]]:
    """Every trajectory record (headers dropped) of the traj files."""
    out = []
    for path in paths:
        out.extend(read_traj(path)[1])
    return out


def read_headers(paths: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """`(path, header)` of each traj file."""
    return [(path, read_traj(path)[0]) for path in paths]


def value_env(env: dict[str, str]) -> dict[str, str]:
    """The part of an engine env that can change a search value.

    Everything `agents.apply_env` would set except the pilot's playout policy
    (`PILOT_ONLY_ENV`) and the logging seams.
    """
    return {
        k: v
        for k, v in env.items()
        if agents.is_engine_env(k)
        and k not in PILOT_ONLY_ENV
        and not k.startswith("STS2_TRACE_")
        and k != "STS2_COMBAT_LOG"
    }


def check_value_env(
    headers: list[tuple[str, dict[str, Any]]], env: dict[str, str], allow: bool
) -> list[str]:
    """Refuse trajectories whose values were scored under another value env.

    `rl` compares a trajectory's `value` / `branch` (scored in the pilot
    workers under the header's `spec.env`) with its own playouts'
    `search_value` (scored under `env`), so the two must use one value
    function. The comparison is deliberately strict: any engine env var but
    the pilot-only playout seams counts, including the whole
    `STS2_RUNAI_MCTS_OVERRIDES` bundle.

    Args:
        headers: `(path, header)` per traj file.
        env: The engine env `rl` runs under.
        allow: Warn instead of refusing.

    Returns:
        One line per mismatching file (empty when all agree).

    Raises:
        SystemExit: On a mismatch without `allow`.
    """
    want = value_env(env)
    bad = []
    for path, h in headers:
        have = value_env(h["spec"]["env"])
        if have != want:
            bad.append(f"{path}: value env {have} != rl's {want}")
    if bad and not allow:
        raise SystemExit(
            "trajectories were scored under another value env (their `value` / "
            "`branch` and rl's `search_value` would be in different units):\n  "
            + "\n  ".join(bad)
            + "\npass the same --env / --overrides to rl, or --allow-env-mismatch"
        )
    for line in bad:
        print(f"[rollout_learn rl] WARNING (--allow-env-mismatch): {line}", flush=True)
    return bad


def pilot_value(rec: dict[str, Any], t: int) -> float | None:
    """The pilot's mean outcome (search value units) from decision `t` of `rec`."""
    if rec.get("value") is None:
        return None
    vals = [float(rec["value"]), *(float(v) for v in rec.get("branch", {}).get(str(t), []))]
    return sum(vals) / len(vals)


def rebuild(rec: dict[str, Any], ts: list[int]) -> dict[int, Any]:
    """The combat states at decisions `ts` of trajectory `rec`, rebuilt.

    Raises:
        RuntimeError: A rebuilt state does not hash as the recorded one (the
            replay is not deterministic; training on it would be training on
            some other position). The hash is `ordered_hash()`, so a replay
            that only reorders a card pile is caught too.
    """
    want = sorted(set(ts))
    out: dict[int, Any] = {}
    if not want:
        return out
    rm = build_fight(rec["fight"], int(rec["seed"]))
    for t, action in enumerate(rec["actions"]):
        if t == want[len(out)]:
            cs = rm.combat_state()
            h = ordered_hash(cs) if cs is not None else None
            if h != int(rec["hashes"][t]):
                raise RuntimeError(
                    f"{rec['key']} rep {rec['rep']} decision {t}: rebuilt state hash {h} != "
                    f"recorded {rec['hashes'][t]} (the replay diverged; was the trajectory "
                    "recorded by another engine build?)"
                )
            out[t] = cs
            if len(out) == len(want):
                break
        rm.take_action(action)
    if len(out) != len(want):
        raise RuntimeError(f"{rec['key']} rep {rec['rep']}: decisions {want} past its end")
    return out


def start_spec(rec: dict[str, Any], t: int) -> dict[str, Any]:
    """What a worker needs to rebuild decision `t` of `rec` on its own."""
    h = int(rec["hashes"][t])
    return {
        "id": start_id(rec["key"], int(rec["seed"]), t, h),
        "fight": rec["fight"],
        "seed": int(rec["seed"]),
        "actions": rec["actions"][:t],
        "hash": h,
    }


def candidate_points(
    recs: list[dict[str, Any]], per_traj: int, rng: random.Random
) -> list[tuple[int, int]]:
    """`(record index, decision)` for every searched decision (fight start
    always kept), at most `per_traj` per trajectory when positive."""
    out = []
    for i, rec in enumerate(recs):
        if rec.get("end") == "error":
            continue
        ts = [t for t, s in enumerate(rec["searched"]) if s]
        if per_traj and len(ts) > per_traj:
            first = ts[:1]
            ts = sorted(first + rng.sample(ts[1:], per_traj - len(first)))
        out.extend((i, t) for t in ts)
    return out


# ── the deeper-search target (rl --sb-target search) ────────────────────


def q_of(rec: dict[str, Any] | None, action: Any) -> float | None:
    """Q of root child `action` in a search record, None if it has none."""
    if rec is None or not searched(rec):
        return None
    want = json.dumps(action, sort_keys=True)
    for a in rec["combat_search"]["actions"]:
        if json.dumps(a["action"], sort_keys=True) == want:
            return float(a["q"])
    return None


def _search_worker(job: tuple[dict[str, Any], int]) -> tuple[str, float | None]:
    """`V*` of the rebuilt start from two independent `sims`-simulation searches.

    The first search picks the move (Q, then visits); the second, with
    another seed, prices it. The chosen child's Q in the search that chose it
    is the max of near-equal noisy estimates (the deployed search spreads
    its visits almost evenly, GH #442), so it reads high: on 96 recorded
    elite/boss starts at 2000 sims it read +0.052 (se 0.020) above an
    independent search's Q of the same move. The second
    search's Q of the same move has no root-level selection in it. The
    visit-weighted root mean would read low instead (it averages the bad
    moves in). None for a forced move, or when the second search has no
    child for the move.
    """
    spec_, sims = job
    import sts2_rs

    rm = build_fight(spec_["fight"], spec_["seed"])
    for a in spec_["actions"]:
        rm.take_action(a)
    cs = rm.combat_state()
    if cs is None or ordered_hash(cs) != spec_["hash"]:
        raise RuntimeError(f"{spec_['id']}: rebuilt state does not hash as recorded")
    seed = spec_["hash"] % (2**31 - 1)
    first = sts2_rs.MctsCombatAgent(seed=seed, sims=int(sims), boss_sims=int(sims)).step(rm.clone())
    if first is None or not searched(first):
        return spec_["id"], None
    move = first["combat_search"]["recommended"]
    second = sts2_rs.MctsCombatAgent(
        seed=(seed * 7919 + 1) % (2**31 - 1), sims=int(sims), boss_sims=int(sims)
    ).step(rm)
    return spec_["id"], q_of(second, move)


# The `V*` estimator `_search_worker` implements; a cache row computed by
# another one is not reused.
SEARCH_ESTIMATOR = "reeval-v1"


class SearchTargets:
    """`V*(start)` from a deeper honest search, computed lazily and cached.

    The cache is a JSONL file of `{"id", "sims", "fp", "value"}` rows,
    appended as values arrive, so a resumed run (or another run on the same
    starts) reuses them. `id` names the position (`start_id` carries its
    `ordered_hash`), and `fp` fingerprints everything else the value depends
    on: the search spec (sims, engine env, playout-net file hash), the
    engine build and the estimator. A row with another `fp` is ignored.
    """

    def __init__(self, path: str, sims: int, procs: int, env: dict[str, str]) -> None:
        import sts2_rs

        self.path, self.sims, self.procs, self.env = path, sims, procs, env
        self.fp = hashlib.sha256(
            json.dumps(
                {
                    "spec": agents.mcts_spec(sims, sims, env),
                    "engine": sts2_rs.src_hash(),
                    "estimator": SEARCH_ESTIMATOR,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        self.values: dict[str, float | None] = {}
        self.ignored = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("fp") == self.fp:
                            self.values[row["id"]] = row["value"]
                        else:
                            self.ignored += 1
        self._pool = None

    def get(self, specs: list[dict[str, Any]]) -> list[float | None]:
        """Targets for `specs`, searching (in parallel) the ones not cached."""
        need = [s for s in specs if s["id"] not in self.values]
        need = list({s["id"]: s for s in need}.values())
        if need:
            if self._pool is None:
                self._pool = get_context("spawn").Pool(
                    self.procs, initializer=_worker_init, initargs=(self.env,)
                )
            with open(self.path, "a", encoding="utf-8") as f:
                for sid, v in self._pool.imap_unordered(
                    _search_worker, [(s, self.sims) for s in need]
                ):
                    self.values[sid] = v
                    row = {"id": sid, "sims": self.sims, "fp": self.fp, "value": v}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
        return [self.values[s["id"]] for s in specs]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def contested_weights(pwin: Any, floor: float) -> Any:
    """Sampling weight `max(p(1-p), floor)` per start (numpy array in, out)."""
    import numpy as np

    p = np.asarray(pwin, np.float64)
    return np.maximum(p * (1.0 - p), floor)


def add_traj_parser(sub: Any) -> None:
    t = sub.add_parser(
        "traj",
        help="record the deployed pilot's trajectories on bank fights (mid-fight starts)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    t.add_argument("--fights", required=True, help="pilot_human_decks extract JSONL")
    t.add_argument("--out", required=True, help="traj JSONL (appended; resumes)")
    t.add_argument("--holdout", nargs="*", help="fight JSONL files whose runs are never used")
    t.add_argument("--holdout-encounter", nargs="*")
    t.add_argument("--rooms", default="boss,elite")
    t.add_argument("--max-fights", type=int, default=0, help="subsample fights (0 = all)")
    t.add_argument("--sample-seed", type=int, default=0)
    t.add_argument("--reps", type=int, default=1, help="trajectories per fight")
    t.add_argument("--rep0", type=int, default=DEFAULT_REP0)
    t.add_argument("--sims", type=int, default=1000)
    t.add_argument("--boss-sims", type=int, default=4000)
    t.add_argument("--max-steps", type=int, default=3000)
    t.add_argument(
        "--branch-reps",
        type=int,
        default=0,
        help="extra pilot continuations from a sampled decision (the 'pilot' SB target's reps)",
    )
    t.add_argument(
        "--branch-frac",
        type=float,
        default=0.0,
        help="fraction of decisions that get --branch-reps continuations",
    )
    t.add_argument("--env", nargs="*", help="engine env K=V for the pilot workers")
    t.add_argument("--overrides", help="STS2_RUNAI_MCTS_OVERRIDES for the pilot")
    t.add_argument("--rollout-net", help="pilot playout net (STS2_ROLLOUT_NET)")
    t.add_argument("--procs", type=int, default=16)
    t.add_argument(
        "--allow-bench-overlap",
        action="store_true",
        help="do NOT drop combat-bench runs (ppo.combat_bench.holdout)",
    )


def check_k(k: int, sb_beta: float) -> None:
    """Simulation balancing splits the k playouts into two halves of >= 2."""
    if sb_beta > 0 and (k < 4 or k % 2):
        raise SystemExit(f"--sb-beta needs an even --k >= 4 (two halves of >= 2), got {k}")


def nan_to_none(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
