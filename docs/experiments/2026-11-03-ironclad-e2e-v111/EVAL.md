# Evaluation tools (WP6)

This file describes the code that applies PREREG-0's evaluation rules (`PREREG.md`): what each tool does, how a new policy plugs in, and what the files look like. The rules themselves live in PREREG-0 and plan section 11; where this file and PREREG-0 differ, PREREG-0 wins.

Run every command from `sts2-pycli/` with `PYTHONPATH=python` and the campaign venv's Python (W2's for anything after the W2 switch). `<evals>` is `~/runs/e2e/evals` unless you pass `--evals` or set `E2E_EVALS`.

## The tools

| Tool | Writes | Use |
|---|---|---|
| `scripts/e2e_arm.sh` | `<evals>/<PANEL>/<ARM>/` | R1, R2 and R3, through `ppo.probe` |
| `ppo.e2e.evaluate run` | the same layout | Final, or any policy you plug in |
| `ppo.e2e.evaluate report` | stdout | plan section 11's report for an unsealed arm |
| `ppo.e2e.score_test` | stdout, `--json`, the read log | the P2 read, TEST once |
| `ppo.e2e.devboss` | a states directory, result files | DEVBOSS and REF |
| `ppo.e2e.panels`, `ppo.e2e.stats` | nothing | P1's seed ranges and P2's statistics |

## Run a policy on a panel

```text
PYTHONPATH=python <python> -m ppo.e2e.evaluate run FINAL DEV 2000 160000000 \
    --policy MODULE:FUNC --ckpt CKPT --procs 16 --pin "$E2E_PIN"
```

- TEST and SMOKE_SEALED need `--sealed` and `--pin`, and a clean commit. They refuse `--allow-unpinned`, `--allow-dirty` and `--allow-plain-wheel`. With `--sealed`, the command prints only row counts, error counts and the manifest's sha256, sends the workers' stdout to `/dev/null`, and keeps the panel directory at mode 700.
- The command re-executes itself under a clean environment: `HOME`, `PATH`, `LANG`, the worker thread caps (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 1, `ops_contract.WORKER_THREAD_ENV`), `PYTHONPATH=<pin>:python`, `STS2_WHEEL_PIN` and `STS2_GAME_VERSION=v0.111.0`. No `STS2_RUNAI_`, `STS2_ROLLOUT_` or `STS2_MCTS_` knob survives. Its workers start under the caps and run torch on one thread. To set one for an arm, pass `--set-env K=V`; it's recorded in `meta/config.txt`.
- A repeat with the same arguments skips complete shards. A repeat with other arguments in the same arm directory is refused, so use a new arm name.
- An errored seed re-runs once on the same binary (P2). A run that exceeds `--hang-timeout` (default 1,800 s) is killed, booked as an error and re-run once too. A policy or env that fails to load stops the call instead, so a broken checkpoint never turns into rows of missing data.
- Rows hold no wall clock, and shards are written in seed order once complete, so a repeat on the same binary is byte-identical, whatever `--procs` is.

### The policy plug-in

`--policy MODULE:FUNC` names a factory. Each worker process calls `FUNC(ckpt, device=DEVICE)` once. The object it returns must have:

- `act_decisions(decisions) -> list[int]`: one merged-class index per decision, in `range(len(decision.members))`. The runner passes one decision at a time, in play order, and applies the class's first member.
- `begin_run(seed)` (optional): called before each run. DEVBOSS calls it before each pair, with the pair's reseed.
- `meta` (optional): provenance.

WP4b's `ppo.e2e.net.StudentPolicy` has `act_decisions`. It also needs a tokenizer whose fight state (`reset_fight`, `observe`) the caller keeps current, so the student's factory has to wrap it, for example:

```text
def make(ckpt, device="cpu"):
    return StudentPolicy.from_checkpoint(ckpt, device=device, tokenizer=<WP3b tokenizer>)
```

`first` (always class 0) and `random` (seeded by the run seed) are stand-ins for tests.

### The env plug-in

`--env MODULE:FUNC` names `FUNC(seed)`. It returns an env with `reset()`, `step(class_index)` and `rm`, which yields decisions (objects with `members`, and optionally `bootstrap`) until a terminal (an object with `outcome`, `won` and optionally `cap` and `error`). The default is WP2's `ppo.e2e.env:E2EEnv`. `raw` is a stand-in that makes every legal action its own class.

## The files

```text
<evals>/<PANEL>/<ARM>/shards/s<lo>_<hi>.jsonl   250-seed shards, rows in seed order
<evals>/<PANEL>/<ARM>/retry/s<seed>.jsonl       the one re-run of an errored seed
<evals>/<PANEL>/<ARM>/meta/config.txt           what produced the rows (key=value lines)
<evals>/<PANEL>/<ARM>/meta/wheel_guard.txt      the wheel guard's verdict
<evals>/<PANEL>/<ARM>/meta/calls.log            every call's arguments
<evals>/<PANEL>/<ARM>/logs/, work/              logs, and rows still in flight
<evals>/<PANEL>/<ARM>.sha256                    sha256sum of every shard, then every retry file
```

An `evaluate` row has `seed`, `policy`, `ascension`, `character`, `act1`, `outcome` (`won`, `lost`, `capped`, `no_actions`), `won`, `cap`, `steps`, `n_decisions`, `final` (`ppo.env.metrics_snapshot`), `reached_second_boss`, `bosses` (per boss fight: act, slot, encounter, entry HP, entry max HP, entry potions, won), `end` (act, floor, room), `wheel` and `py_src`. An error row has only `seed`, `policy`, `outcome`, `won` and `error`, as `ppo.workers.play_batch` writes it.

For a sealed read, commit each arm's `<ARM>.sha256` byte for byte next to `PREREG.md`, as `sha256/<PANEL>/<ARM>.sha256`. The scorer also accepts `sha256/<PANEL>_<ARM>.sha256`, `sha256/<PANEL>-<ARM>.sha256` and `sha256/<ARM>.sha256`.

## Read TEST

```text
PYTHONPATH=python <python> -m ppo.e2e.score_test \
    --prereg ../docs/experiments/2026-11-03-ironclad-e2e-v111/PREREG.md \
    --final FINAL --cstar R3 --r1 R1 --r2 R2 --r3 R3 \
    --devboss FINAL_DEVBOSS.jsonl REF.jsonl --json read.json
```

The scorer is the only reader of `evals/TEST/`. It stops before computing any statistic (exit 2, `READ STOPPED`) on any integrity failure: a missing or extra seed, a stray file, a file that doesn't match its manifest, a manifest that doesn't match its committed copy, an uncommitted or modified `PREREG.md`, two engine builds inside one arm, a read before Thu Nov 12, 09:00 CET (PREREG Amendment 5; was Mon Nov 9), a second read without `--reread-reason`, or DEVBOSS files that don't pair. Every TEST read is appended to `<evals>/TEST/score_test.reads.jsonl` before the numbers are computed. `--panel DEV` (or `SMOKE`) runs the same code without the sealed rules; `SMOKE_SEALED` rehearses them without the time lock.

The scorer doesn't check that the P5 pick and C* are committed before the read. That stays a step of the read procedure.

## DEVBOSS

```text
PYTHONPATH=python <python> -m ppo.e2e.devboss capture --out STATES \
    --run-ckpt RUN.pt --combat-ckpt COMBAT.pt --procs 8
PYTHONPATH=python <python> -m ppo.e2e.devboss play --states STATES --agent mcts \
    --out REF.jsonl --procs 8
PYTHONPATH=python <python> -m ppo.e2e.devboss play --states STATES --agent student \
    --policy MODULE:FUNC --ckpt FINAL.pt --out FINAL.jsonl --procs 8
PYTHONPATH=python <python> -m ppo.e2e.devboss report FINAL.jsonl REF.jsonl
```

- **capture** plays teacher runs from seed 175,000,000 up: campaign runs (A10, v0.111.0, agent mode, potion swap on), the run net greedy and its combat-PPO net in every fight, as `ppo.probe --policy net_greedy --combat policy` plays them. Before each Move into a boss room it captures the run. It keeps the first 200 captures of each cell (`act1_boss`, `act2_boss`, and `act3_double`, captured before act 3's first boss) in seed order. The run net must carry the features `card_mods,reward_upgrade`.
- **play** starts pair (i, j) from state i: clone, `reseed_run(175,500,000 + 4i + j)`, then the captured Move. The agent plays the boss fight; for `act3_double`, both fights to the run's end. `mcts` is REF: `sts2_rs.MctsCombatAgent(seed=reseed, sims=1000, boss_sims=4000)`. `policy` is a combat-PPO net (`--combat-ckpt`). `student` is `--policy`, as in `evaluate`.
- **report** prints one agent's numbers, or Final − REF: the pooled win rate's difference as a ratio mean over states, with a cluster bootstrap (each state and its four reseeds are one cluster). A REF error drops a pair, and a Final-only error is a Final loss. The read is INVALID when more than 0.5% of pairs are missing, when the panel is partial (`--limit-states`), when REF isn't the shipped 1000/4000 search, or when Final and REF ran on two engine builds. An INVALID read keeps its place in the Holm family with p = 1.

Both `capture` and `play` select v0.111.0 for themselves and their workers and refuse any engine knob in the environment. Both resume after a stop: `capture` keeps the finished seeds in `work/rows.jsonl`, and `play` keeps the finished pairs in `<out>.part`, as long as the config (checkpoints, code and wheel) is unchanged. They never overwrite a finished output.

### The student's env

`--segment-env MODULE:FUNC` is called as `FUNC(seed, rm, cell, reseed=RESEED, move=MOVE)` with the pair's start. The default, `ppo.e2e.devboss:e2e_segment_env`, wraps WP2's `E2EEnv` in the shape of WP2's practice segments: `Segment(kind="devboss", cell=BOSS_ACT1 | BOSS_ACT2 | BOSS_ACT3_DOUBLE, end="first_ooc" | "terminal", parent_seed, move, reseed)`. An act-1 or act-2 pair ends at the first out-of-combat decision after the fight; a double-boss pair ends with the run. `raw` is the stand-in.

### The files

A states directory holds:

- `states.jsonl`: one state per line, with `state_id`, `cell`, `seed` (the teacher run), `k` (the length of its action prefix), `move`, `fingerprint`, `reseeds` and the entry numbers `act`, `floor`, `hp`, `max_hp`, `gold`, `potions`, `deck`, `boss` and `second_boss`.
- `runs.jsonl`: `{"seed": S, "actions": [...]}` for each run that holds a kept state, cut after its last kept state.
- `capture.json`: the checkpoints and their sha256, the code and the wheel, the seeds tried, the seeds that errored, and the counts.
- `SHA256SUMS`: the sha256 of the three files above. `play` refuses a directory that doesn't match it.
- `capture.log` and `work/`: progress, and the files a resumed capture reads.

A state is its seed plus its action prefix, because a `RunManager` doesn't pickle. Replay constructs the run and takes each action, binding an event room (`materialize_event`) before each action taken in one, as the run did. The capture replays every kept state from the JSON it wrote and checks its fingerprint (RNG counters, deck, enchantments, relics, potions, HP, gold, floor, map position, phase and legal actions); a mismatch stops the capture. Every `play` worker checks it again.

A result file starts with one `{"meta": {...}}` line: the agent and its arguments, the checkpoint's sha256, the states' `SHA256SUMS` digest, the pair count, the code and the wheel. Then comes one row per pair, in (state, reseed) order: `state_id`, `j`, `reseed`, `cell`, `outcome`, `won`, `cap`, `steps` and `fights`. Each fight has its act, slot, encounter, entry HP, end HP, HP lost, turns, entry potions and result. A student row adds `decisions`.

### Cost

A teacher capture run took 1.0-1.7 s on the laptop (3 runs measured), so a 600-state capture is well under an hour on one process (est.: about 2-3 runs per act-3 state). REF wasn't timed at 1000/4000; plan section 9 budgets ~20 proc-h for DEVBOSS and REF together.

## Decisions and deviations

- Timeouts count as missing data, like errors: the tighter reading of P2 (the scorer's docstring has the reason).
- Two engine builds, or an act-1 model that differs on a seed, make a contrast INVALID.
- SMOKE_SEALED gets TEST's sealed rules without the time lock, as a rehearsal.
- An empty retry file means the error stands.
- A hung `evaluate` run is killed and booked as an error, then re-run once. There is no cooperative wall-clock timeout inside a run, so rows stay repeatable.
- DEVBOSS stores a state as a seed plus a run-action prefix, not as WP2's action-log record (`actionlog.capture_record`), because the teacher acts through `ppo.env.PpoRunEnv`, not through `E2EEnv`.
- DEVBOSS captures on the campaign run (potion swap on), so the teacher's run net decides full-belt potion swaps there. The teacher's DEV arms ran with potion swap off.
- DEVBOSS has three cells (act-1 boss, act-2 boss, the double boss), as PREREG-0's table lists them; act 3's second boss isn't a cell of its own.
