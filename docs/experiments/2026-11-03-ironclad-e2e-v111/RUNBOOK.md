# Ironclad e2e campaign runbook (v0.111.0)

This runbook covers how to run the Ironclad end-to-end no-search PPO campaign
on the a4000: set up the host, deploy and restart lanes, respond to alarms,
and hand the box over at T0. It implements plan §8 (Ops and Runbook) and
PREREG-0 P6 (GH #1097, draft PR #1098).

- Plan: [`docs/plans/2026-11-02-ironclad-e2e-ppo-v0111.md`](../../plans/2026-11-02-ironclad-e2e-ppo-v0111.md)
- Pre-registration: [`PREREG.md`](PREREG.md)
- The contract every program shares (files, fields, exit codes, alarms):
  [`sts2-pycli/python/ppo/e2e/ops_contract.py`](../../../sts2-pycli/python/ppo/e2e/ops_contract.py)

In this runbook, *you* are the operator: the campaign's Claude session on the
laptop. *The user* decides D1-D11 and every HUMAN alarm. When this runbook and
the code disagree, the code wins; fix the runbook in the same commit.

## Roles

| Role | Who or what | Does |
|---|---|---|
| User | The person who approved the plan | Decides D1-D11 and every HUMAN alarm. |
| Operator | The campaign's Claude session on the laptop (the WP0 owner) | Deploys, reads the K-gates, acts on OPERATOR alarms, and records every decision in `CAMPAIGN_LOG.md`. |
| Driver | `sts2-pycli/scripts/e2e_driver.py`, a systemd user timer on the a4000, every 5 min | Executes only the pre-registered rules. Writes `status.json`, `campaign_state.json`, `ALARM_<name>` files and `CAMPAIGN_LOG.md`. Needs no Claude session. |
| Watchdog | `sts2-pycli/scripts/e2e_watchdog.sh`, one systemd user unit per lane | Starts the learner as its own child, resumes it after a crash, restarts it on a stall, rolls it back after a NaN, and stops it on `DONE`, a crash loop, low disk, low RAM or the memory cap. Its rules live in `ppo/e2e/watchdog.py`. |
| Learner | `python -m ppo.e2e.learner train` (WP4) | Trains. Honors `DONE`, `EVALS_PAUSE` and `PIN_REQUEST`, and writes the states and logs the contract names. |
| Eval runner | WP6 | Consumes the lane's eval queue and writes results. |

The driver never decides D1-D11 and never clears a HUMAN alarm. Only an `ack`
with the user's decision removes a HUMAN alarm.

## Branches

`exp/e2e-ironclad-v111` is the **campaign branch** and is protected by
convention: it carries only what the live lane may run, plus campaign docs.
Every deploy (hot swap) comes from it. A change lands here only if it is an
operator-gated fix to the lane (`pytest tests/e2e` and ruff on the merge
result, F7 and a learning check before deploy) and it keeps the pinned wheel's
Rust sources unchanged, because the wheel guard refuses a tree whose Rust
sources differ from the wheel pin.

Everything else, such as the workbench, the mod, "Instinct" and main merges,
goes to `exp/e2e-ironclad-v111-workbench`. On Nov 7, PRs #1156 and #1159 had
been merged into the campaign branch. They were reverted there (`git revert
-m 1`) and live on in the workbench branch. To bring that work back after the
campaign, revert the two revert commits first, or git will skip the changes.

## Where things live

On the a4000:

```text
~/runs/e2e/                          the campaign dir (the units hard-code it)
  status.json                        written by every driver pass; the laptop polls it
  campaign_state.json                lanes, deploys (SHA, worktree, pin, wheel digest),
                                     stamps, gate flags, queued evals, mutes
  CAMPAIGN_LOG.md                    append-only, one UTC line per action
  driver.env                         E2E_PY and E2E_DRIVER for e2e-driver.service
  ALARM_<name>                       campaign-level alarms (DRIVER_ERROR)
  alarms/acked/                      acknowledged alarms, with note and decision
  evals/<PANEL>/<ARM>/               baseline and TEST arms (scripts/e2e_arm.sh).
                                     TEST is mode 700 and sealed; actions/ holds
                                     every run's action log
  lanes/<lane>/                      one lane: the learner's --save-dir
    lane.env  lane.args  lane.learner_env
    state.pt  state.json             the resume state, every 5M decisions
    ckpt/full/                       rolling full states every 25M, newest 3 kept
    ckpt/pinned/  ckpt/MANIFEST.tsv  pinned full states, never rotated, with sha256
    ckpt/slim/                       weights and EMAs every 100M
    ckpt/quarantine/                 states a rollback moved aside (pruned first)
    stats.jsonl  throughput.jsonl  heartbeat.json  learner.pid  learner.lock
    stats.orphan.jsonl               stats rows a resume replayed (see below)
    DONE  NAN.json  PIN_REQUEST  EVALS_PAUSE  ALARM_<name>
    evals/queue/  evals/running/  evals/done/
    ops/watchdog.log  ops/learner.log  ops/events.jsonl  ops/starts.tsv
    ops/seed_offset  ops/resume_next.json  ops/resume_used/  ops/done_history/
    actions/actions-<worker>.e2el    action logs: boss-2, capped and error segments
                                     and 2% of runs, 10 GB in all (plan §3.2,
                                     `actionlog.replay` rebuilds one)
~/runs/e2e-pin/<sha16>/              the pinned wheel (W1, then W2), from pin_wheel()
~/projects/sts2-sim-e2e-<sha8>/      one worktree per deployed SHA, never edited
```

A stats row is written after its update, state.pt only every 5M decisions
(~38 iterations of 131,072). So a crash, a kill or an OOM replays every
iteration since the last state.pt, up to ~38, and a rollback replays more.
The resumed learner moves the rows past its state's iteration (and a torn
last line) to `stats.orphan.jsonl` before it writes one, so `it` in
`stats.jsonl` stays unique and increasing and the window rules (CAPS,
E1_REVERT, HELD_ROW_LEAK, the learning check, the pre-flight counts) never
read an iteration twice. Its log says how many rows moved. The parked rows
are the discarded trajectory: read them only to diagnose the crash.

On the laptop: the campaign worktree `~/projects/sts2-sim-e2e`, the frozen
baseline worktree `~/projects/sts2-sim-e2e-w`, and the poll log.

After the week, archive the checkpoints, stats, sealed TEST rows and the W2
`.so` to `~/projects/sts2/bench-archive/2026-11-03-e2e-ironclad-v111/` with a
sha256 manifest (plan §13), and commit a copy of `CAMPAIGN_LOG.md` next to
RESULTS.md.

### Shell helpers

The procedures below use these helpers. Run them in a shell on the a4000.
Before the first `init-lane`, `lane.env` doesn't exist yet: set `WT` to the
new worktree and `PY` to its venv python by hand.

```bash
CAMP=~/runs/e2e
LANE=$CAMP/lanes/main
WT=$(sed -n 's/^E2E_WORKTREE=//p' "$LANE/lane.env")   # the deployed worktree
PY=$(sed -n 's/^E2E_PY=//p' "$LANE/lane.env")         # its venv python
drv() { "$PY" "$WT/sts2-pycli/scripts/e2e_driver.py" "$@"; }
ck()  { (cd "$WT/sts2-pycli" && PYTHONPATH=python "$PY" -m ppo.e2e.ckpt "$@"); }
wd()  { (cd "$WT/sts2-pycli" && PYTHONPATH=python "$PY" -m ppo.e2e.watchdog "$@"); }
```

## Set up the host

Do this once on the a4000, as part of the T0 deploy (plan §10.4).

1. Check lingering, so user units run without a login session:

   ```bash
   loginctl show-user "$USER" -p Linger    # must print Linger=yes
   ```

   If it prints `Linger=no`, ask the user to run `sudo loginctl enable-linger "$USER"`.

2. Check that user units can enforce `MemoryMax`:

   ```bash
   cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers
   systemd-run --user --wait --collect --quiet -p MemoryMax=64M -p MemorySwapMax=0 \
     /usr/bin/python3 -c 'b = bytearray(256 << 20)'; echo "rc=$?"
   ```

   The first command must list `memory`. The second must print a nonzero `rc`
   (the kernel killed the 256 MB allocation). If either check fails, the unit's
   `MemoryMax=200G` does nothing, and the watchdog's own memory check
   (`RSS_STOP`, 200 GB of proportional set size over the learner's session) is
   the only cap. Record which one applies with `drv note campaign ...`.

3. Install the units from
   [`systemd/`](systemd/):

   ```bash
   mkdir -p ~/.config/systemd/user
   cp "$WT"/docs/experiments/2026-11-03-ironclad-e2e-v111/systemd/e2e-* ~/.config/systemd/user/
   systemctl --user daemon-reload
   ```

4. Write `~/runs/e2e/driver.env`, two lines that point at the deployed worktree:

   ```text
   E2E_PY=<the venv python>
   E2E_DRIVER=<the worktree>/sts2-pycli/scripts/e2e_driver.py
   ```

   Then start the timer:

   ```bash
   systemctl --user enable --now e2e-driver.timer
   systemctl --user list-timers e2e-driver.timer
   ```

5. Pin the wheel outside every venv, so the wheel-venv garbage collector can't
   reap it (plan §8 "One binary"):

   ```bash
   cd "$WT/sts2-pycli" && PYTHONPATH=python "$PY" -c \
     'import os; from ppo.wheel_guard import pin_wheel; print(pin_wheel(os.path.expanduser("~/runs/e2e-pin")))'
   ```

   The command prints the pin dir, `~/runs/e2e-pin/<sha16>`. Check the glibc
   gate on a laptop-built wheel: the highest `GLIBC_` symbol in
   `objdump -T <pin>/sts2_rs/*.so` must not exceed `ldd --version` on the box,
   and the sha256 must match on both hosts. Export `WHEEL_VENV_NO_GC=1` in
   every build shell.

## Deploy a lane

Each deploy is a new worktree at one SHA. Nobody edits, pulls or switches a
worktree that a lane runs from: the workers import Python from disk.

1. Create the worktree at the campaign SHA, from any clone on the box:

   ```bash
   git -C <clone> fetch origin
   git -C <clone> worktree add --detach ~/projects/sts2-sim-e2e-<sha8> <sha>
   ```

2. Write the learner's arguments, one per line, to a file (WP4 defines them;
   `budget_decisions` must be at least 1e10, plan §6). Leave `--run-seed-base`
   out: a campaign lane (mode `train`, no `--max-iters`) plays its trainer runs
   from 1,000,000,000 (`panels.STUDENT_TRAINER_SEEDS`, PREREG P1), and smokes,
   learning checks and B-T0 from 1,300,000,000; the learner refuses a base whose
   seeds leave the lane's block (at most 40 workers fit). On a wheel with the
   WP-ENG-1 getters (W2), declare the fill: the two lines `--filled` and
   `["WP-ENG-1"]`. A fresh net then inits the telegraph, forecast and encounter
   inputs live; the [pre-flight](#run-the-pre-flight) fails a run whose rows and
   declaration disagree, and so does the learner itself on the first table of
   every start (`FILL_MISMATCH`, exit 87). The content prior is a fill event too:
   with `content-prior` in that list (`["WP-ENG-1","content-prior"]`) the learner
   loads `content_prior_v0111.tsv` into the net's prior buffer and stamps its
   `prior_sha256`, at a launch or at a restart that adds it (never one that drops
   it).

3. Register the lane. The command refuses a dirty worktree, a missing python
   or pin, and a lane that already ran (use `--hot-swap` for that):

   ```bash
   drv init-lane main --worktree ~/projects/sts2-sim-e2e-<sha8> --python "$PY" \
     --pin ~/runs/e2e-pin/<sha16> --args-file <args file> --note "T0 launch"
   ```

   It writes `lane.env` (worktree, python, pin, `E2E_REQUIRE_CLEAN=1`),
   `lane.args`, and the deploy record in `campaign_state.json`, which the
   driver checks every pass: a moved `HEAD`, a dirty worktree or a changed
   `.so` in the pin stops the lane (`SOURCE_MISMATCH`, `WHEEL_MISMATCH`).

4. Start the lane:

   ```bash
   systemctl --user enable --now e2e-lane@main
   ```

5. Start the wandb mirror (it only reads the lane dir; a crash or a wandb
   outage never touches the learner):

   ```bash
   cp docs/experiments/2026-11-03-ironclad-e2e-v111/systemd/e2e-wandb@.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now e2e-wandb@main
   ```

   The run is `ironclad-e2e-main` in wandb project `sts2-ppo-ironclad-e2e`
   (x axis = decisions; stats, throughput, events, `eval/<panel>/win_rate`).
   After a hot-swap, `systemctl --user restart e2e-wandb@main` picks up the new
   worktree and continues the same run (`ops/wandb_sync.json` keeps its id).

6. Watch the first 15 min: `drv status`, `tail -f $LANE/ops/watchdog.log`,
   `journalctl --user -u e2e-lane@main -f`. Expect a `learner <pid> started`
   line in `ops/watchdog.log`, then stats rows. The learner runs under
   `env -i` with only `HOME`, `PATH`, `LANG`, `USER`,
   `PYTHONPATH=<pin>:python`, `STS2_WHEEL_PIN`, the lane environment
   (`STS2_GAME_VERSION=v0.111.0`, `STS2_RUNAI_POTION_SWAP=1`,
   `STS2_PPO_FEATURES=card_mods,reward_upgrade`, `OMP_NUM_THREADS=1`, the
   malloc settings) and `lane.learner_env`. Its rollout workers and eval runs
   start with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`
   at 1 (`ops_contract.WORKER_THREAD_ENV`), whatever `lane.learner_env` sets.

## Run the pre-flight

The pre-flight (plan §3.2, playbook §0.1) runs on a learner run's save dir: at
G0 on the 30-min smoke, and at 100M on the lane. Random actors never leave act
1, so it reads the run's own `stats.jsonl`, where teacher acting takes runs
into acts 2 and 3. Run it from the run's worktree, with the run's venv and
wheel and the lane environment:

```bash
# The G0 smoke (laptop): <save dir> is the smoke learner's --save-dir.
cd <worktree>/sts2-pycli
env STS2_GAME_VERSION=v0.111.0 STS2_RUNAI_POTION_SWAP=1 \
  STS2_PPO_FEATURES=card_mods,reward_upgrade OMP_NUM_THREADS=1 PYTHONPATH=python \
  nice -n 10 <venv>/bin/python -m ppo.e2e.preflight --run-dir <save dir> --json <save dir>/preflight.json

# A lane on the box, with its pinned wheel:
PIN=$(sed -n 's/^E2E_PIN=//p' "$LANE/lane.env")
(cd "$WT/sts2-pycli" && env $(wd env) PYTHONPATH="$PIN:python" STS2_WHEEL_PIN="$PIN" \
  nice -n 10 "$PY" -m ppo.e2e.preflight --run-dir "$LANE" --json "$LANE/ops/preflight.json")
```

It prints one block and exits 0 on `PREFLIGHT PASS`, 1 on `PREFLIGHT FAIL`
(one `problem:` line per failed rule), 2 on a usage error (no stats rows yet,
no declared fill events, another game version). It checks:

- **Stats** (`preflight_stats`, summed over every stats row): each required
  row type and each fine key occurs, `unknown` and `fine_unknown` = 0, capped
  segments ≤ 0.1%, no error segment (an OOC stall with no legal action is
  one, `error/NoActions`), no cap assert, no F9 hit. The fine keys,
  `fine_unknown`, the asserts and the forced steps come from each row's
  `segment_counts` (the ended segments' summed `E2ETerminal.counts`). A
  run with rows from before the learner wrote `segment_counts` lists those
  rules as unchecked; each fine key's surface is a required row type
  (`FINE_DTYPE`), so every surface is still checked.
- **Fill** (`ppo.e2e.fillcheck`, GH #1097 A4): 30 random A10 runs on this
  process's wheel (smoke seeds 162,000,900+), against the fill events the
  run's stamps declare (`--filled JSON` overrides; a disagreement is a
  problem). With WP-ENG-1 declared, RUNCTX's encounter id is nonzero on ≥95%
  of combat rows, and the telegraph columns, the forecast columns and
  PRIV_ENEMY are nonzero on ≥50% of live enemies; without it, all of them are
  zero. Either way, dead enemies carry none, and every combat row with a draw
  pile has PRIV_DRAW.
- **Stamps**: this process's game version, wheel build, Python source, vocab
  and schema must equal the run's, or its rows would not show the run's.

Record the verdict: `drv note main "pre-flight PASS: <the stats and fill lines>"`.
On a fill problem, fix the launch args (`--filled`) or the wheel, and
relaunch: a declaration never changes on a resume except by adding an event.

## Run the F7 check

F7 (plan §3.4) blocks G1: the student's policy logits must not move when the
hidden information does. `ppo.e2e.f7check` plays real v0.111.0 A10 runs (the
recorder's agent), permutes the draw order and every combat RNG stream
(`cs.sample_world`, twice), the room queues and later bosses (a future-map
read raises), and the run RNG, and strips PRIV (T14). The move state
(telegraph, intent forecast) stays fixed. It compares the student's logits
with `torch.equal`, row by row, on each serving path this tree has: **S0** as
the learner serves at launch (`net_inf.bf16`: bf16 autocast on CUDA) and
**fp32 on CPU** (eval's `net.policy`, and the update's unbucketed forward).

```bash
# On the box. At G0 on the laptop: the G0 worktree, its venv and pin, and an
# --out under ~/runs/e2e/.
PIN=$(sed -n 's/^E2E_PIN=//p' "$LANE/lane.env")
cd "$WT/sts2-pycli"
F7ENV="STS2_GAME_VERSION=v0.111.0 OMP_NUM_THREADS=2 PYTHONPATH=$PIN:python STS2_WHEEL_PIN=$PIN"
# G0: the launch net (the learner's default NetConfig, 6x256, fill
# ["WP-ENG-1","content-prior"]); --net JSON overrides the config.
env $F7ENV nice -n 10 "$PY" -m ppo.e2e.f7check --device cuda:0 --runs 12 \
  --out "$LANE/ops/f7_g0.json"
# A trained student: a pinned state (or state.pt, or an exported student).
STATE=$(ck last-pinned "$LANE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["path"])')
env $F7ENV nice -n 10 "$PY" -m ppo.e2e.f7check --device cuda:0 --runs 12 \
  --ckpt "$STATE" --out "$LANE/ops/f7_$(basename "$STATE" .pt).json"
```

Check `nvidia-smi` for free GPU memory first (the G0 run allocated 157 MiB, plus
the CUDA context). Twelve runs take ~5 min, most of it the recording agent. It
exits 0 on `PASS`, 1 on `FAIL`, 2 on a usage error, and writes a JSON verdict:
runs, rows, mismatches per path and kind, device, dtype, the net config and
the stamps. **The pass rule is 0 mismatches and no problem.** A problem means
a leak the tokenizer showed directly (the policy encoding read a PRIV call, or
a future-map read) or a check that ran but proved nothing: a no-op re-deal (it
moved the draw or PRIV on fewer than half the combat rows), a moved move
state, or a net whose logits do not move under an honest change. Exit 3 is a
crash: the verdict has `ok: false` and the `error` with its traceback. It
proves nothing either way, so F7 has not passed: record it, fix the cause, and
run the check again.

When it runs:

- **G0**: once, on CUDA bf16 with the launch config, before G1.
- **Every serving hot-swap** (S1a, S1b, S2): from the new worktree, on the
  lane's newest pinned state (the `hot_swap` pin), before the new commit
  serves: step 6 of [Hot-swap to a new commit](#hot-swap-to-a-new-commit).
- **Each 1B pin**: on that pinned state, next to its DEV-2000 evals.

On a mismatch or a problem: F7 blocks G1, so stop the lane
(`systemctl --user stop e2e-lane@main`), record the verdict with `drv note`,
and tell the user. Never serve a net that fails it: do not start a hot-swap's
new commit, and do not resume a lane whose pinned state fails.

S1b (WP5b's student engine) does not exist in this tree yet, so the tool
checks S0 and CPU fp32 only. When S1b hot-swaps in, F7 on S1b runs then, on
its serving path, before it serves.

A lane that serves with `--packed-serve true` (S1-pad: the pad-free student and
combat-teacher forwards) runs F7 with `--packed` too, which adds that path
(`paths.packed` in the verdict) on `--device`.

Record each verdict: `drv note main "F7 <when> PASS: <runs> runs, <rows> rows, 0 mismatches, <device> <dtype>"`.

## Read the status

- On the box: `drv status` prints the summary line and every open alarm.
  `status.json` has the details per lane: decisions, rate over the last hour,
  learner and watchdog liveness, heartbeat age, `DONE`, the last pinned state,
  the next K-gate, starts in the last 2 h, disk free, checkpoint usage, eval
  counts and the last six eval reads.
- From the laptop: `sts2-pycli/scripts/e2e_poll.sh a4000` (see
  [Poll from the laptop](#poll-from-the-laptop)).
- `CAMPAIGN_LOG.md` gets one line per watchdog event (start, exit, crash,
  stall, pin, refusal) and per driver or operator action. Add your own lines
  with `drv note main <text>` or `drv note campaign <text>`: decisions,
  parity checks, learning checks, gate reads.

The summary line starts with the worst open alarm: `OK`, `OPERATOR` or `HUMAN`.

## Respond to alarms

An alarm is a JSON file `ALARM_<name>` in the lane dir (or the campaign dir)
with `severity`, `writer`, `reason`, `time`, `count` and `who_acts`. A repeat
bumps `count` and `last_time`. The table follows `ALARMS` in
`ops_contract.py`; a test checks that it names every alarm.

| Alarm | Severity | Writer | Trigger | What the writer did | What happens next |
|---|---|---|---|---|---|
| `DISK_PRUNED` | OPERATOR | watchdog | Free disk under 45 GB | Deleted quarantined, slim and old rolling states (never pinned ones); the lane runs on | Find what grew (`usage_gb` in `status.json`) and archive pinned states |
| `DISK_STOP` | OPERATOR | watchdog, driver | Free disk under 30 GB after pruning | Stopped the lane (`DONE`) | Free disk, then [restart the lane](#restart-a-lane-after-a-stop) |
| `CRASH_LOOP` | OPERATOR | watchdog | A 5th learner start within 2 h | Stopped the lane (`DONE`) | Read `ops/learner.log`, fix, restart with `--reset-starts`; HUMAN if you find no fix |
| `RESUME_MISSING` | OPERATOR | watchdog | `ops/resume_next.json` names no file, or `state.pt` is gone while full states exist | Refused to start (no `DONE`; the unit stays down) | Pick a pinned state (`ck rollback`), then start the unit |
| `NAN_ROLLBACK` | OPERATOR | watchdog | First NaN (learner exit 86, or a non-finite stats row) | Resumed from the second-newest full state with sampler seed offset +1e6 | Note it in the log; diagnose while training continues |
| `NAN_HUMAN` | HUMAN | watchdog | A second NaN within 100M decisions, or no full state to roll back to | Stopped the lane (`DONE`) | The user decides |
| `RAM_STOP` | OPERATOR | watchdog, driver | MemAvailable under 20 GB after the evals stopped | Stopped evals (`EVALS_PAUSE`), then the lane (`DONE`) | Check `free -g` and who holds the memory, then restart |
| `RSS_STOP` | OPERATOR | watchdog | The lane's memory above 200 GB (the `MemoryMax` backstop) | Stopped the lane (`DONE`) | Find the leak (`ops/learner.log`); restart with fewer workers or envs |
| `LEARNER_HUMAN` | HUMAN | watchdog | The learner exited 87 (a guard refused to continue) | Stopped the lane (`DONE`); the learner's own alarm says why | The user decides |
| `LR_CLOSED_FORM` | HUMAN | driver, learner | An lr off its closed form by more than 1e-6 (relative) | Stopped the lane (`DONE`) | The user decides |
| `WHEEL_MISMATCH` | HUMAN | driver, learner | `worker_wheel_mismatch`, a changed wheel stamp, or a changed `.so` in the pin | Stopped the lane (`DONE`) | The user decides |
| `SOURCE_MISMATCH` | HUMAN | driver, learner | `worker_source_mismatch`, a changed source, vocab or schema stamp, or a moved or dirty worktree | Stopped the lane (`DONE`) | The user decides |
| `GAME_VERSION` | HUMAN | driver | A stats stamp names a game version other than v0.111.0 | Stopped the lane (`DONE`) | The user decides |
| `F9_INVARIANT` | HUMAN | driver, learner | Any F9 invariant hit (HP at most max, energy at least 0, pile conservation) | Stopped the lane (`DONE`) | The user decides |
| `FILL_MISMATCH` | HUMAN | learner | The first conclusive table of a learner start, fresh or resumed, fails the fill check (`ppo.e2e.fillcheck`) against `--filled`: inputs of an undeclared fill event, or a declared one's inputs missing (the alarm's `summary` has the counts) | Stopped the lane (`DONE`): the learner exits 87 before that batch trains | The user decides: fix `--filled` or the wheel (`fillcheck.wheel_filled()` names what a wheel fills), then relaunch |
| `LC_RECHECK_FAIL` | OPERATOR | learner | A learning check failed after an earlier one passed (a re-check's verdict passed with `--lc-verdict` at a start, e.g. at K2) | Kept the λ_KS ramp; training continues | Read the verdict file the alarm names (`verdict`) and decide whether to investigate; note it in the log and ack it |
| `CAPS` | OPERATOR | driver | Capped fights or runs above 0.1% over the last 50 iterations | Nothing | Look for deadlocks or sim exploits in the capped runs' action logs |
| `CONTRACT` | OPERATOR | driver | A stats or throughput row misses contract fields | Nothing; rules that need the field skip it | Fix the learner's rows at the next restart |
| `HELD_ROW_LEAK` | OPERATOR | driver | `update.rows_overdue` above 0 in each of the last 7 iterations in a row (`HELD_LEAK_ITERS` = `HOLD_ITERS` + 3; a held row is overdue once it waited more than `HOLD_ITERS` + 2 = 6 iterations) | Nothing; training continues | A worker stall of d ≥ 3 iterations gives d + 1 overdue iterations in a row, a leak (held rows nothing releases or drops) gives them for good. Find the runs in `ops/learner.log`; if `rows_overdue` keeps growing, restart the learner (a resume drops the held rows) |
| `PIN_MISSING` | OPERATOR | driver | No pinned state 10M decisions after a 250M milestone or a K-gate | Wrote `PIN_REQUEST` | Check the learner's pinning; pin by hand if needed (`ck pin`) |
| `K2_READ` | OPERATOR | driver | K2 due (~100M decisions) | Queued DEV-1000 on the K2 pinned state | [Read K2](#read-the-k-gates) |
| `K2_FAIL` | OPERATOR | operator | You read K2 as a fail | Stopped the lane (`DONE`) | Fix, then [relaunch from the last good pinned state](#roll-back-on-purpose) |
| `K3_READ` | OPERATOR | driver | K3 due (1.4B decisions, backstop Thu Nov 5 18:00 CET) | Queued DEV-1000 on the K3 pinned state | [Read K3](#read-the-k-gates) |
| `K3_NOGO` | HUMAN | operator | Paired DEV-1000 student − C* ≤ −3pp at K3 | Nothing; training continues | The user decides; you diagnose for up to 6 h; relaunch only for a found bug |
| `K4_READ` | OPERATOR | driver | K4 due (3B decisions, backstop Sat Nov 7 08:00 CET) | The 1B rule queued the 3B DEV-2000 evals | [Read K4](#read-the-k-gates) |
| `K4_NOGO` | HUMAN | operator | Paired DEV-2000 student < C* at K4 | Nothing; training continues | As `K3_NOGO` |
| `STALL_SIGNAL` | HUMAN | driver, operator | DEV-2000 (EMA0.996) below 11% at 2B or below 14% at 3B decisions (plan §1.4) | Nothing; training continues | As `K3_NOGO`; TEST is read regardless |
| `DEV_JUMP` | OPERATOR | driver | A DEV-1000 read more than 5pp above the previous DEV-1000 read of its variant (plan §12); DEV-2000 reads are 1B apart, where +5pp is the median forecast slope, so they don't count | Nothing | Audit the top runs against C# for a sim exploit; raise `SIM_EXPLOIT` if you find one |
| `SIM_EXPLOIT` | HUMAN | operator | You found a sim exploit (a DEV jump, F9 hits, odd card or relic use) | Nothing; training continues until the user decides | The user decides: patch, W3, resume from the last clean pinned state (plan §12) |
| `E1_REVERT` | OPERATOR | driver, operator | E=1 and some type's KL > 0.03 or clipfrac > 0.3 for 20 iterations in a row; or, at +300M, paired DEV-1000 ≥ 3.0pp below pre-switch with exact McNemar p < 0.05 | Nothing | [Revert to E=2](#switch-to-one-epoch-and-back) from the `e1_switch` pinned state |
| `HEARTBEAT_STALE` | OPERATOR | driver | No heartbeat for 15 min while the learner is alive | Nothing; the watchdog's stall rule should restart it | Read `ops/watchdog.log`; restart by hand if the watchdog is stuck |
| `WATCHDOG_DOWN` | OPERATOR | driver | No `DONE`, but no live watchdog for 10 min | Nothing | `systemctl --user status e2e-lane@<lane>`; start the unit |
| `DRIVER_ERROR` | OPERATOR | driver | The driver raised an exception | Nothing; the traceback is in the alarm file | Fix the driver; the lane itself is unaffected |

The learner writes its own guard alarms (`WHEEL_MISMATCH`, `SOURCE_MISMATCH`,
`LR_CLOSED_FORM`, `F9_INVARIANT`, `FILL_MISMATCH`) before it exits 87, so you
can see both `LEARNER_HUMAN` and the guard's alarm. It also writes `LC_RECHECK_FAIL`, which
stops nothing: the learner trains on.

### Handle a HUMAN alarm

1. Don't restart, roll back or edit anything the alarm points at.
2. Send the user a push notification with the poll line, then a message with
   the alarm file, the last stats row and what you know so far.
3. For `K3_NOGO`, `K4_NOGO`, `STALL_SIGNAL` and `SIM_EXPLOIT`, training
   continues: diagnose for up to 6 h and report what you find. Relaunch only
   for a found bug, and only with the user's decision.
4. Wait for the user's decision. Then record it:

   ```bash
   drv ack NAN_HUMAN --lane main --note "what you found" --decision "user: <the decision>"
   ```

   An `ack` moves the alarm to `alarms/acked/` and logs it. It never restarts
   the lane. A HUMAN alarm can't be muted.

5. Carry out the decision with the procedures below.

### Handle an OPERATOR alarm

1. Do what the table says.
2. Acknowledge it with a note: `drv ack CAPS --lane main --note "<what you did>"`.
3. To silence a known, harmless repeat, add `--mute-hours H`. Alarms that stop
   the lane can't be muted, and a mute never skips a stop.

### Raise an operator verdict

The driver can't compute the paired gate verdicts: they need the baseline
arms' per-seed rows. When you read a fail, raise it:

```bash
drv raise K3_NOGO --lane main --reason "paired DEV-1000 student - C* = -3.4pp (McNemar p=0.02)"
```

`drv raise` accepts only the alarms whose writer includes `operator`:
`K2_FAIL` (writes `DONE`), `K3_NOGO`, `K4_NOGO`, `STALL_SIGNAL`,
`SIM_EXPLOIT` and `E1_REVERT`.

## Stop and start a lane

- **Pause** (you resume it later from `state.pt`):
  `systemctl --user stop e2e-lane@main`. The watchdog sends SIGTERM to the
  learner, which finishes the update in flight, writes `state.pt` and exits.
  After 10 min the watchdog sends SIGKILL to the learner's process group;
  systemd kills whatever is left after 15 min. A reboot starts the lane again.
- **Stop for good** (the final stop, or before a fix a reboot must not undo):
  `drv stop main --reason final --note "<why>"`. The learner sees `DONE` at
  its next iteration, writes `state.pt` and exits; if it still runs 15 min
  later, the watchdog stops it. The unit then stays down, after a reboot too,
  until you clear `DONE`. The first `DONE` reason wins.
- **Start**: `systemctl --user start e2e-lane@main`. If systemd hit its own
  start limit (10 starts in 1 h), run `systemctl --user reset-failed e2e-lane@main` first.

Never kill the learner by pid yourself: the watchdog waits on its own child,
and a stray kill counts as a crash.

### Restart a lane after a stop

1. Find why it stopped: `cat $LANE/DONE`, `drv status`,
   `tail -50 $LANE/ops/watchdog.log`, `tail -200 $LANE/ops/learner.log`.
2. Fix the cause. A code fix is a [hot-swap](#hot-swap-to-a-new-commit).
3. Acknowledge the alarms that stopped the lane (a HUMAN one with the user's decision).
4. Clear `DONE`. Add `--reset-starts` after a crash loop, so the old starts no
   longer count:

   ```bash
   drv clear-done main --note "<what you fixed>" [--reset-starts]
   ```

   The command refuses while the learner runs or while an alarm that stops the
   lane is open. HUMAN alarms under which training continues (`K3_NOGO`,
   `K4_NOGO`, `STALL_SIGNAL`, `SIM_EXPLOIT`) don't block it. It archives
   `DONE` to `ops/done_history/`.

5. Start the unit and watch the first 15 min.

### After a reboot

With lingering on and the units enabled, the timer and every enabled lane
start at boot. The watchdog resumes from `state.pt` unless `DONE` exists; the
start counts toward the crash-loop window, and the watchdog pins the resume
state first (reason `restart`). Check `drv status` once the box is back.

## Hot-swap to a new commit

Each step of the hot-swap ladder after the launch (S1a, S1b, S2) keeps the
state dict and deploys as a new worktree (plan §8).

1. WP0 merges the change on the campaign branch and pushes it. Note the SHA.
2. Create the new worktree (see [Deploy a lane](#deploy-a-lane), step 1).
3. Run the parity check on the new worktree: old and new paths in fp32 on the
   same 10K states and a fixed 65K batch must give |Δloss| < 1e-3, max
   |Δlogit| < 1e-2, and equal greedy actions where the top-2 gap exceeds 0.05.
   Record it: `drv note main "parity S1a PASS: ..."`. On a fail, keep the old
   worktree and retry at the next 25M checkpoint.
4. Stop the lane: `systemctl --user stop e2e-lane@main`, and wait until
   `drv status` shows the learner down.
5. Redeploy. `--hot-swap` pins `state.pt` (reason `hot_swap`) and resets the
   stamp baseline, so the new source stamp is expected:

   ```bash
   drv init-lane main --worktree ~/projects/sts2-sim-e2e-<sha8> --python "$PY" \
     --pin ~/runs/e2e-pin/<sha16> --args-file <args file> --hot-swap --note "S1a"
   ```

6. Run [the F7 check](#run-the-f7-check) from the new worktree on the
   `hot_swap` pin (`ck last-pinned "$LANE"`), on `cuda:0`. On a fail, do not
   start the unit: the new commit must not serve. Record it with `drv note`
   and tell the user.
7. If the driver code changed, point `E2E_DRIVER` in `driver.env` at the new worktree.
8. Start the unit. The learning check re-runs at each restart (plan §7); record
   its result with `drv note`.
9. Keep the old worktrees until the campaign ends.

## Roll back on purpose

Deliberate rollbacks (a K2 relaunch, the E=1 revert, an exploit fix) start
only from pinned states. Only the watchdog's NaN rule uses a rolling state.

1. Stop the lane. `ck rollback` refuses while the learner runs.
2. Pick the target: `ck last-pinned $LANE`, or read `$LANE/ckpt/MANIFEST.tsv`.
   Check the sha256 of every pinned state with `ck verify $LANE`.
3. Roll back:

   ```bash
   ck rollback "$LANE" --to pinned_d0000050000000_restart.pt --reason k2_relaunch
   ```

   It moves `state.pt` and every newer rolling or slim state to
   `ckpt/quarantine/`, marks newer pinned states `abandoned` in the manifest,
   and writes `ops/resume_next.json`. The next start resumes from the target.
   Queued evals of abandoned states are dropped, and a K-gate whose pinned
   state was abandoned re-arms.

4. Deploy the fix with `init-lane --hot-swap` (step 5 of
   [Hot-swap to a new commit](#hot-swap-to-a-new-commit)), acknowledge the
   alarm that stopped the lane, clear `DONE` if there is one, and start the unit.

## Switch to one epoch and back

The E=1 switch happens at the first restart after S1b passes parity, if the
learner waited on the update for at least 25% of the last hour and a 20-min
smoke keeps KL ≤ 0.03 and clipfrac ≤ 0.3 (plan §9).

1. Stop the lane.
2. Pin the pre-switch state **before** the hot-swap, so its name carries the
   reason (pinning the same bytes again returns this pin):

   ```bash
   ck pin "$LANE" --src "$LANE/state.pt" --reason e1_switch --note "pre-switch, E=2"
   ```

3. Hot-swap with the E=1 args. The stats rows then carry `epochs: 1`.
4. `E1_REVERT` means revert: stop the lane, `ck rollback "$LANE" --to <the
   e1_switch pin> --reason e1_revert`, hot-swap back to the E=2 args, start.

## Read the K-gates

The driver raises `K2_READ`, `K3_READ` and `K4_READ` once each, with the
gate's pinned state, and queues the gate's evals. If a gate is due by its
backstop time first, the driver writes `PIN_REQUEST` (reason `k3` or `k4`) and
flags the gate on the first pinned state at or after the decision count at
the request.

| Gate | Due | Rule (PREREG-0 P6) | On a fail |
|---|---|---|---|
| K2 | ~100M | Student DEV-1000 ≥ teacher pair − 8pp, paired; agreement ≥ 85% combat and ≥ 75% run; finite value loss; EV within 0.1 of its 50M value | `drv raise K2_FAIL`; fix; relaunch from the last good pinned state |
| K3 | 1.4B, backstop Thu Nov 5 18:00 CET | NO-GO iff paired DEV-1000 student − C* ≤ −3pp | `drv raise K3_NOGO`; diagnose ≤ 6 h while training continues |
| K4 | 3B, backstop Sat Nov 7 08:00 CET | NO-GO iff paired DEV-2000 student < C* | `drv raise K4_NOGO`; as K3 |

Read each gate on the results in `$LANE/evals/done/` against the baseline
arms in `~/runs/e2e/evals/`. `sts2-pycli/scripts/e2e_paired.py NAME=ARM_DIR ...`
prints each arm's rate and every paired contrast on the shared seeds (exact
McNemar, Newcombe CI); it refuses `evals/TEST`. Record the verdict with
`drv note`, then acknowledge the `*_READ` alarm.

## Evals

The driver writes eval requests; the eval runner (WP6) consumes them.

- **Cadence**: DEV-1000 (EMA0.996) on every 250M milestone pin; DEV-2000
  (raw, EMA0.996 and EMA0.999) on every 1B pin, instead of that milestone's
  DEV-1000; DEV-1000 (EMA0.996) on the K2 and K3 pins. Each request is written
  once per pinned state (by sha256), panel and variant.
- **Request**: `$LANE/evals/queue/<id>.json` with `id`, `lane`, `state` (the
  pinned file), `sha256`, `decisions`, `panel`, `n`, `seed_start`
  (160,000,000), `variant`, `nice` (10), `max_procs` (6), `reason`,
  `requested_by` and `time`.
- **Claim**: rename the request into `evals/running/`. The rename is atomic,
  so two runners never claim the same request.
- **Result**: `evals/done/<id>.json`, the request's fields plus `wins` and
  `runs` (finished runs) and any extra fields, such as the path of the
  per-seed rows. The driver reads a result only when `runs` equals `n`, and
  only while the state is still pinned (not abandoned). It logs every read and
  applies `STALL_SIGNAL` and `DEV_JUMP`.
- **F7**: on every 1B pin, also run [the F7 check](#run-the-f7-check) on the
  pinned state; a fail stops the lane.
- **Pause**: while `EVALS_PAUSE` exists (the watchdog or the driver writes it
  under 20 GB MemAvailable), stop every running eval, move its request back to
  the queue and claim nothing. The writer removes it at 24 GB or more.
- **Share**: evals stay at or under 5% of wall clock (plan §8). If they take
  more, halve the DEV-1000 cadence: delete the queued DEV-1000 requests of the
  odd 250M milestones (the driver doesn't write a request twice) and note it.

Baseline and TEST arms run separately, with `scripts/e2e_arm.sh` from the
frozen W1 worktree, into `~/runs/e2e/evals/<PANEL>/<ARM>/`. TEST is sealed:
nobody reads a TEST win rate before the final read.

## Pin requests

`PIN_REQUEST` in the lane dir asks the learner to pin its next full state. It
is a JSON object with `by`, `reason` (one of `PIN_REASONS` in
`ops_contract.py`: `milestone`, `k2`, `k3`, `k4`, `restart`, `nan_rollback`,
`hot_swap`, `e1_switch`, `request`, `manual`, `final`), `why` and `time`. The
learner pins with that reason and deletes the file. To pin by hand instead:
`ck pin "$LANE" --src "$LANE/state.pt" --reason manual --note "<why>"`.

## Disk and memory

| Resource | Threshold | Action | By |
|---|---|---|---|
| Disk | < 45 GB free | Prune: quarantined files, then slim states off the 500M grid, then the other slim states, then rolling full states beyond the newest two; never pinned states or `state.pt` (`DISK_PRUNED`) | watchdog |
| Disk | < 30 GB free after pruning | Stop the lane (`DISK_STOP`) | watchdog; driver if the watchdog is down |
| RAM | MemAvailable < 20 GB | Stop the evals (`EVALS_PAUSE`); if still low after 120 s, stop the lane (`RAM_STOP`) | watchdog; driver if the watchdog is down |
| Lane memory | > 200 GB | Stop the lane (`RSS_STOP`) | watchdog; `MemoryMax=200G` in the unit |

The plan's estimate is at most 40 GB of the ~104 GB free by ~9B decisions.
`E2E_DISK_QUOTA_GB` sets a soft quota for the lane dir. Every watchdog
threshold has an `E2E_WD_<NAME>` override, such as `E2E_WD_DISK_STOP_GB=30`
(`wd limits` prints the current values). Set it in `lane.env`, and in
`driver.env` for the driver's backstops, then restart the lane.

## Hand over the box at T0

The #1081 session does the handover, or you do it with the user's OK (plan §10.4).

1. Post on PR #1081 and message the #1081 session.
2. After the H3 read, stop the H2 and H3 watchdogs, send SIGINT to H2 and H3
   (they resume), and wait on their process IDs. Never use `pgrep -f`.
3. If the main #1081 run was resumed, checkpoint it and let it exit. A
   SIGSTOPped process keeps its GPU memory.
4. Copy both runs' final states, cfg, wheel record and stats to
   `~/projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3/`, with a
   sha256 manifest. The teachers load only from there.
5. Confirm an empty `nvidia-smi`, at least 80 GB of RAM and at least 75 GB of
   disk free.
6. Log the handover: `drv note campaign "T0 handover: ..."`.
7. [Set up the host](#set-up-the-host), then [deploy](#deploy-a-lane): W2,
   cross-host identity, B-T0, the learning check and the 45-min launch smoke
   with a kill and a resume. B-T0's "staleness <= 1" (PREREG) reads the stats
   rows' `staleness.fresh_max`, this iteration's rows; `staleness.max` counts
   the held-back run rows too, which lag up to 5 by design.

## Poll from the laptop

Run one poll every 30 min, with a recurring job (`CronCreate` or `/loop 30m`):

```bash
E2E_POLL_LOG=~/runs/e2e-poll.log ~/projects/sts2-sim-e2e/sts2-pycli/scripts/e2e_poll.sh a4000
```

It runs `tailscale ping` first, then reads `status.json` and every
`ALARM_*` file over ssh (`BatchMode`, 90-s timeout), and prints one line.

| Exit code | Meaning | What you do |
|---|---|---|
| 0 | OK | Nothing. |
| 1 | An OPERATOR alarm | Push a notification; act on it. |
| 2 | A HUMAN alarm (an unreadable alarm counts as HUMAN) | Push a notification; [handle it](#handle-a-human-alarm). |
| 3 | `status.json` is missing or older than 20 min | Push a notification; check `systemctl --user status e2e-driver.timer` and `journalctl --user -u e2e-driver`. |
| 4 | ssh fails | If the line says `via DERP`, the link is relayed and slow, not down: retry. Otherwise push a notification after two failed polls in a row. |

Before you call the box down, run `tailscale ping a4000` yourself: an answer
"via DERP" means the box is up.

## Test the ops code

From `sts2-pycli/` of a worktree, with the campaign venv:

```bash
PYTHONPATH=python <venv>/bin/python -m pytest tests/e2e/test_e2e_ops_*.py -n 0 -q
```

The watchdog tests start the fake learner `tests/e2e/e2e_fake_trainer.py`
for every failure path (crash, crash loop, NaN twice, stall, a learner that
ignores SIGTERM, a HUMAN exit, RAM and disk stops, the memory cap, resume
refusals) and take about a minute. They need Linux; the driver and poller
tests need a POSIX shell. `systemd-analyze --user verify` checks the unit files.

## For other work packages

- **WP4 (learner)**: the obligations are in the `ops_contract.py` docstring:
  the CLI, the pid record and lock, the checkpoint cadences through
  `ppo.e2e.ckpt`, the stats and throughput fields, `DONE`, the NaN exit (86)
  with `NAN.json`, the guard exit (87) with its alarm, and the eval queue.
  `minimal_stats_row()` is a conforming example row.
- **WP6 (evals)**: the request, claim, result and pause protocol in
  [Evals](#evals).
- **WP5 (systems)**: the parity check and the hot-swap procedure above.
