#!/usr/bin/env bash
# e2e_arm.sh: run one evaluation arm of the Ironclad e2e campaign (#1097) on a
# fixed seed panel, in 250-seed shards that skip when already complete.
#
# Usage (from sts2-pycli/ of the frozen worktree ~/projects/sts2-sim-e2e-w):
#
#   E2E_PIN=~/runs/e2e-pin/<sha16> E2E_PY=<venv>/bin/python \
#     scripts/e2e_arm.sh [--env K=V]... ARM PANEL N SEED_START -- <ppo.probe args>
#
#   PANEL  DEV   160,000,000-160,001,999   (DEV-1000 and DEV-2000 start at 160,000,000)
#          TEST  161,000,000-161,003,999   SEALED: prints row and error counts only
#          SMOKE 162,000,000-162,000,999   identity and determinism checks
#          SMOKE_SEALED  SMOKE's seeds with TEST's sealed output, to test sealing
#
# The probe args must name --procs and --run-timeout (ppo.probe defaults to
# cpu_count - 2 workers and 600 s, workers.py:31, :527) and must not name
# --seeds, --seed-start, --out or --game-version, which the runner owns.
#
# What it guarantees (plan section 2, PREREG-0 P2 and P4):
# - The probe runs under `env -i` with only HOME, PATH, LANG, OMP_NUM_THREADS=1,
#   PYTHONPATH=<pin>:python, STS2_WHEEL_PIN=<pin>, STS2_GAME_VERSION=v0.111.0
#   and the arm's own --env pairs, so none of the STS2_RUNAI_/ROLLOUT_/MCTS_
#   knobs in the caller's shell leaks in.
# - It runs the venv's python directly, never `uv run` (wheel_guard.py:202-203).
# - It refuses a dirty worktree, an sts2_rs import that does not come from the
#   pin, and an arm directory whose recorded config (argv, env, SHA, pin)
#   differs from this call, so shards from two configs never mix.
# - An errored seed re-runs once on the same binary (retry/), per P2.
# - It writes <EVALS>/<PANEL>/<ARM>.sha256 (sha256sum of every shard and
#   retry file, in seed order).
# - TEST lives under a mode-700 directory. For TEST the runner reads only each
#   row's `seed` and whether `outcome == "error"`; it never reads or prints
#   `won`, and the probe's own summary goes to a log file nobody reads before
#   the sealed read (P4).
#
# Environment:
#   E2E_PIN        pinned wheel dir (required), e.g. from
#                  `python -c 'from ppo.wheel_guard import pin_wheel; print(pin_wheel("'$HOME'/runs/e2e-pin"))'`
#   E2E_PY         venv python (default: ./.venv/bin/python)
#   E2E_EVALS      output root (default: ~/runs/e2e/evals)
#   E2E_SHARD_PAR  shards run at once, each with procs / E2E_SHARD_PAR workers
#                  (default 4 when --procs >= 8, else 1); staggers shard tails
set -euo pipefail

die() {
  echo "e2e_arm: $*" >&2
  exit 2
}

ARM_ENV=()
while [[ $# -gt 0 && $1 == --env ]]; do
  [[ $# -ge 2 && $2 == *=* ]] || die "--env needs K=V"
  ARM_ENV+=("$2")
  shift 2
done
[[ $# -ge 5 ]] || die "usage: e2e_arm.sh [--env K=V]... ARM PANEL N SEED_START -- <ppo.probe args>"
ARM=$1 PANEL=$2 N=$3 SEED_START=$4
shift 4
[[ $1 == -- ]] || die "expected -- before the ppo.probe args"
shift
PROBE_ARGS=("$@")

[[ $ARM =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "bad ARM name: $ARM"
[[ $N =~ ^[1-9][0-9]*$ ]] || die "N must be a positive integer"
[[ $SEED_START =~ ^[0-9]+$ ]] || die "SEED_START must be a non-negative integer"
case $PANEL in
  DEV) LO=160000000 HI=160002000 ;;
  TEST) LO=161000000 HI=161004000 ;;
  SMOKE | SMOKE_SEALED) LO=162000000 HI=162001000 ;;
  *) die "PANEL must be DEV, TEST, SMOKE or SMOKE_SEALED" ;;
esac
SEALED=0
[[ $PANEL == TEST || $PANEL == SMOKE_SEALED ]] && SEALED=1
((SEED_START >= LO && SEED_START + N <= HI)) ||
  die "seeds $SEED_START..$((SEED_START + N - 1)) leave the $PANEL panel [$LO, $HI)"

# --procs and --run-timeout are required; the runner owns the seed and sink flags.
PROCS="" HAVE_TIMEOUT=0 REST=()
i=0
while ((i < ${#PROBE_ARGS[@]})); do
  a=${PROBE_ARGS[i]}
  case $a in
    --seeds | --seeds=* | --seed-start | --seed-start=* | --out | --out=* | --game-version | --game-version=*)
      die "the runner owns $a" ;;
    --procs)
      PROCS=${PROBE_ARGS[i + 1]:-}
      i=$((i + 2))
      continue
      ;;
    --procs=*)
      PROCS=${a#--procs=}
      i=$((i + 1))
      continue
      ;;
    --run-timeout | --run-timeout=*) HAVE_TIMEOUT=1 ;;
  esac
  REST+=("$a")
  i=$((i + 1))
done
[[ $PROCS =~ ^[1-9][0-9]*$ ]] || die "pass --procs N explicitly"
((HAVE_TIMEOUT)) || die "pass --run-timeout SECONDS explicitly"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
[[ -n ${E2E_PIN:-} && -d $E2E_PIN ]] || die "set E2E_PIN to a pinned wheel dir"
PIN=$(realpath "$E2E_PIN")
# Resolve the venv directory but not the interpreter symlink itself: a venv's
# bin/python links to the base interpreter, and following it drops the venv.
PY=${E2E_PY:-$ROOT/.venv/bin/python}
PY="$(realpath "$(dirname "$PY")")/$(basename "$PY")"
[[ -x $PY ]] || die "no python at $PY (set E2E_PY)"
EVALS=$(realpath -m "${E2E_EVALS:-$HOME/runs/e2e/evals}")

SHA=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
  die "worktree $ROOT has uncommitted changes; baselines run only from a clean commit"
fi

SHARD_PAR=${E2E_SHARD_PAR:-$(((PROCS >= 8) ? 4 : 1))}
[[ $SHARD_PAR =~ ^[1-9][0-9]*$ ]] || die "E2E_SHARD_PAR must be a positive integer"
((SHARD_PAR <= PROCS)) || SHARD_PAR=$PROCS
SHARD_PROCS=$((PROCS / SHARD_PAR))

RUN_ENV=(
  "HOME=$HOME"
  "PATH=/usr/local/bin:/usr/bin:/bin"
  "LANG=${LANG:-C.UTF-8}"
  "OMP_NUM_THREADS=1"
  "PYTHONPATH=$PIN:python"
  "STS2_WHEEL_PIN=$PIN"
  "STS2_GAME_VERSION=v0.111.0"
  "${ARM_ENV[@]}"
)
run_py() { env -i "${RUN_ENV[@]}" "$PY" "$@"; }

PANEL_DIR=$EVALS/$PANEL
ARM_DIR=$PANEL_DIR/$ARM
mkdir -p "$PANEL_DIR"
((SEALED)) && chmod 700 "$PANEL_DIR"
mkdir -p "$ARM_DIR"/{meta,shards,logs,retry}

exec 9>"$ARM_DIR/.lock"
flock -n 9 || die "another e2e_arm.sh holds $ARM_DIR/.lock"

# The pinned import, the worktree SHA and the wheel guard, recorded per call.
SO=$(run_py -c 'import sts2_rs, os; print(os.path.realpath(sts2_rs.__file__))')
[[ $SO == "$PIN"/* ]] || die "sts2_rs imported from $SO, not from the pin $PIN"
SO_SHA=$(sha256sum "$(dirname "$SO")"/*.so | awk '{print $1}' | sort | sha256sum | awk '{print $1}')
GUARD=$(run_py -m ppo.wheel_guard 2>&1) || die "wheel_guard failed:"$'\n'"$GUARD"

CONFIG=$(
  printf 'arm=%s\npanel=%s\nn=%s\nseed_start=%s\nsha=%s\npin=%s\nso_sha256=%s\npython=%s\n' \
    "$ARM" "$PANEL" "$N" "$SEED_START" "$SHA" "$PIN" "$SO_SHA" "$PY"
  printf 'env=%s\n' "${ARM_ENV[@]}"
  printf 'probe_arg=%s\n' "${REST[@]}"
)
if [[ -f $ARM_DIR/meta/config.txt ]]; then
  if [[ $(cat "$ARM_DIR/meta/config.txt") != "$CONFIG" ]]; then
    diff <(cat "$ARM_DIR/meta/config.txt") <(echo "$CONFIG") >&2 || true
    die "$ARM_DIR was produced by a different config (above); use a new ARM name"
  fi
else
  echo "$CONFIG" >"$ARM_DIR/meta/config.txt"
fi
echo "$GUARD" >"$ARM_DIR/meta/wheel_guard.txt"
{
  date '+%FT%T%z'
  printf '%q ' "$0" "$@"
  echo
} >>"$ARM_DIR/meta/calls.log"

# shard_status FILE LO HI: prints "ok" when FILE holds exactly the seeds
# LO..HI-1, once each. Reads only `seed`.
shard_status() {
  run_py -c '
import json, sys
path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    seeds = [json.loads(line)["seed"] for line in open(path, encoding="utf-8") if line.strip()]
except (OSError, ValueError, KeyError):
    print("bad"); sys.exit(0)
print("ok" if sorted(seeds) == list(range(lo, hi)) else "partial")
' "$@"
}

# The probe prints its win rate on stdout, so for TEST stdout is dropped and
# only stderr (tracebacks) reaches the log.
if ((SEALED)); then OUT_SINK=/dev/null; else OUT_SINK=""; fi
probe() { # probe LOG <ppo.probe args>
  local log=$1
  shift
  if [[ -n $OUT_SINK ]]; then
    run_py -m ppo.probe "$@" >"$OUT_SINK" 2>"$log"
  else
    run_py -m ppo.probe "$@" >"$log" 2>&1
  fi
}

run_shard() { # run_shard LO HI
  local lo=$1 hi=$2 name
  name=s${1}_$2
  probe "$ARM_DIR/logs/$name.log" "${REST[@]}" --procs "$SHARD_PROCS" \
    --seeds $((hi - lo)) --seed-start "$lo" --out "$ARM_DIR/shards/$name.jsonl"
}

TODO=()
END=$((SEED_START + N))
for ((lo = SEED_START; lo < END; lo += 250)); do
  hi=$((lo + 250 < END ? lo + 250 : END))
  f=$ARM_DIR/shards/s${lo}_$hi.jsonl
  if [[ -f $f && $(shard_status "$f" "$lo" "$hi") == ok ]]; then
    continue
  fi
  TODO+=("$lo:$hi")
done
echo "e2e_arm: $ARM $PANEL seeds $SEED_START..$((END - 1)) at $SHA; ${#TODO[@]} shard(s) to run," \
  "$SHARD_PAR at a time x $SHARD_PROCS procs"

FAILED=0
running=0
for job in "${TODO[@]}"; do
  run_shard "${job%%:*}" "${job##*:}" &
  running=$((running + 1))
  if ((running >= SHARD_PAR)); then
    wait -n || FAILED=$((FAILED + 1))
    running=$((running - 1))
  fi
done
while ((running > 0)); do
  wait -n || FAILED=$((FAILED + 1))
  running=$((running - 1))
done

INCOMPLETE=0
for ((lo = SEED_START; lo < END; lo += 250)); do
  hi=$((lo + 250 < END ? lo + 250 : END))
  [[ $(shard_status "$ARM_DIR/shards/s${lo}_$hi.jsonl" "$lo" "$hi") == ok ]] || INCOMPLETE=$((INCOMPLETE + 1))
done
((INCOMPLETE == 0)) || die "$INCOMPLETE shard(s) incomplete ($FAILED probe exit(s) nonzero); see $ARM_DIR/logs, then re-run"

# Errored seeds re-run once (P2), one probe per seed, SHARD_PAR x SHARD_PROCS at a time.
mapfile -t ERRORED < <(run_py -c '
import glob, json, os, sys
d = sys.argv[1]
retried = {os.path.basename(p)[1:-6] for p in glob.glob(os.path.join(d, "retry", "s*.jsonl"))}
for p in sorted(glob.glob(os.path.join(d, "shards", "*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("outcome") == "error" and str(r["seed"]) not in retried:
                print(r["seed"])
' "$ARM_DIR")
if ((${#ERRORED[@]})); then
  echo "e2e_arm: re-running ${#ERRORED[@]} errored seed(s) once"
  par=$((PROCS < ${#ERRORED[@]} ? PROCS : ${#ERRORED[@]}))
  running=0
  for s in "${ERRORED[@]}"; do
    probe "$ARM_DIR/logs/retry_s$s.log" "${REST[@]}" --procs 1 --seeds 1 --seed-start "$s" \
      --out "$ARM_DIR/retry/s$s.jsonl" &
    running=$((running + 1))
    if ((running >= par)); then
      wait -n || true
      running=$((running - 1))
    fi
  done
  wait || true
fi

# sha256 manifest over every shard and retry file, in seed order.
(
  cd "$ARM_DIR"
  # Panel seeds all have nine digits, so a plain sort is seed order.
  find shards -name '*.jsonl' | sort | xargs sha256sum
  find retry -name '*.jsonl' | sort | xargs -r sha256sum
) >"$PANEL_DIR/$ARM.sha256"

COUNTS=$(run_py -c '
import glob, json, os, sys
d = sys.argv[1]
retry = {}
for p in glob.glob(os.path.join(d, "retry", "s*.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            retry[r["seed"]] = r.get("outcome") == "error"
rows = errs = 0
for p in glob.glob(os.path.join(d, "shards", "*.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows += 1
            if r.get("outcome") == "error":
                errs += retry.get(r["seed"], True)
print(f"rows={rows} errors_after_retry={errs}")
' "$ARM_DIR")
echo "e2e_arm: $ARM $PANEL done: $COUNTS"
echo "e2e_arm: manifest $PANEL_DIR/$ARM.sha256 ($(sha256sum "$PANEL_DIR/$ARM.sha256" | awk '{print $1}'))"

if ((!SEALED)); then
  run_py -c '
import glob, json, math, os, sys
d = sys.argv[1]
retry = {}
for p in glob.glob(os.path.join(d, "retry", "s*.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            retry[r["seed"]] = r
rows = []
for p in glob.glob(os.path.join(d, "shards", "*.jsonl")):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("outcome") == "error" and r["seed"] in retry:
                r = retry[r["seed"]]
            rows.append(r)
scored = [r for r in rows if r.get("outcome") != "error"]
n, w = len(scored), sum(1 for r in scored if r.get("won"))
z = 1.959964
p = w / n if n else 0.0
den = 1 + z * z / n if n else 1
c = (p + z * z / (2 * n)) / den if n else 0
h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den if n else 0
dec = [r["n_decisions"] for r in scored if "n_decisions" in r]
to = sum(1 for r in scored if r.get("outcome") == "timeout")
print(f"e2e_arm: WR {w}/{n} = {p:.4f} (Wilson 95% {c - h:.4f}-{c + h:.4f}); timeouts {to};"
      f" decisions/run {sum(dec) / max(1, len(dec)):.1f}")
' "$ARM_DIR"
fi
