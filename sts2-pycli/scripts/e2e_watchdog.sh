#!/usr/bin/env bash
# e2e_watchdog.sh: supervise the learner of one lane of the Ironclad e2e
# campaign (#1097, plan section 8 Ops). One systemd user unit per lane runs it
# (docs/experiments/2026-11-03-ironclad-e2e-v111/systemd/e2e-lane@.service).
#
# Usage:
#
#   scripts/e2e_watchdog.sh SAVE_DIR [-- LEARNER_CMD...]
#
# SAVE_DIR is the lane's save dir (~/runs/e2e/lanes/<lane>). It holds:
#   lane.env           KEY=VALUE lines, read here (never sourced): E2E_PY (venv
#                      python, default ./.venv/bin/python), E2E_PIN (pinned
#                      wheel dir), E2E_WORKTREE (the unit's ExecStart uses it),
#                      E2E_REQUIRE_CLEAN (default 1), E2E_WD_<LIMIT> overrides.
#   lane.args          the learner's own args, one per line.
#   lane.learner_env   optional KEY=VALUE lines added to the learner's env.
#
# The learner command is `$E2E_PY -u -m ppo.e2e.learner train`, or
# LEARNER_CMD (tests). The watchdog appends --save-dir SAVE_DIR,
# --sampler-seed-offset N, --resume PATH (when resuming) and lane.args.
#
# What it guarantees:
# - The learner runs under `env -i` with only HOME, PATH, LANG, USER,
#   PYTHONPATH=[<pin>:]python, STS2_WHEEL_PIN=<pin> (when pinned), the lane
#   environment of ppo.e2e.ops_contract.LANE_ENVIRONMENT and lane.learner_env.
# - It runs in its own session (setsid). The watchdog waits on that pid, its
#   own child, and never searches with pgrep. A stop sends SIGTERM to the
#   learner, then SIGKILL to its process group after the grace.
# - One watchdog per save dir: it holds an flock on ops/watchdog.lock, and the
#   learner inherits it, so an orphaned learner also blocks a second watchdog.
# - The rules (crash, DONE, crash loop, disk, RAM, RSS, NaN, stall) live in
#   python/ppo/e2e/watchdog.py, which this script asks before every start,
#   every poll and after every exit.
#
# Exit codes (ppo.e2e.ops_contract.WD_EXIT_*): 0 done or stopped by a signal,
# 1 error, 2 usage or refused start, 3 disk, 4 crash loop, 5 RAM, 6 NaN HUMAN,
# 7 learner HUMAN, 8 memory cap, 9 locked. The unit's
# RestartPreventExitStatus= lists 2-9, so systemd restarts only on 1.
set -uo pipefail

die() {
  echo "e2e_watchdog: $*" >&2
  exit "${2:-1}"
}

[[ $# -ge 1 ]] || die "usage: e2e_watchdog.sh SAVE_DIR [-- LEARNER_CMD...]" 2
SAVE=$(realpath -m "$1")
shift
CUSTOM=()
if [[ $# -gt 0 ]]; then
  [[ $1 == -- ]] || die "expected -- before the learner command" 2
  shift
  CUSTOM=("$@")
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || die "cannot cd to $ROOT"
mkdir -p "$SAVE/ops" || die "cannot create $SAVE/ops" 2
WDLOG=$SAVE/ops/watchdog.log
LLOG=$SAVE/ops/learner.log

log() {
  local line
  printf -v line '%s [%s] %s' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$*"
  printf '%s\n' "$line" >>"$WDLOG"
  printf '%s\n' "$line"
}

# KEY=VALUE lines of FILE into the array named by $2 (comments and blank lines
# skipped, one level of matching quotes stripped, nothing expanded).
read_kv() {
  local file=$1 line key val
  local -n out_=$2
  out_=()
  [[ -f $file ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || die "$file: not KEY=VALUE: $line" 2
    key=${BASH_REMATCH[1]} val=${BASH_REMATCH[2]}
    if [[ $val =~ ^\"(.*)\"$ || $val =~ ^\'(.*)\'$ ]]; then val=${BASH_REMATCH[1]}; fi
    out_+=("$key=$val")
  done <"$file"
}

LANE_ENV=()
read_kv "$SAVE/lane.env" LANE_ENV
for kv in "${LANE_ENV[@]}"; do export "${kv?}"; done
LEARNER_ENV=()
read_kv "$SAVE/lane.learner_env" LEARNER_ENV
LANE_ARGS=()
if [[ -f $SAVE/lane.args ]]; then
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    LANE_ARGS+=("$line")
  done <"$SAVE/lane.args"
fi

# Resolve the venv directory but not the interpreter symlink itself: a venv's
# bin/python links to the base interpreter, and following it drops the venv.
PY=${E2E_PY:-$ROOT/.venv/bin/python}
PY="$(realpath "$(dirname "$PY")")/$(basename "$PY")"
[[ -x $PY ]] || die "no python at $PY (set E2E_PY in lane.env)" 2
PIN=""
if [[ -n ${E2E_PIN:-} ]]; then
  [[ -d $E2E_PIN ]] || die "E2E_PIN=$E2E_PIN is not a directory" 2
  PIN=$(realpath "$E2E_PIN")
fi

# The watchdog's own rule calls: stdlib only, from this worktree.
wdpy() { PYTHONPATH="$ROOT/python" "$PY" -m ppo.e2e.watchdog "$@"; }

exec 9>>"$SAVE/ops/watchdog.lock"
if ! flock -n 9; then
  log "another watchdog, or a learner it left behind, holds $SAVE/ops/watchdog.lock"
  exit 9
fi

LIMITS=$(wdpy limits) || die "ppo.e2e.watchdog limits failed"
while IFS='=' read -r key val; do
  [[ $key =~ ^WD_[A-Z0-9_]+$ ]] && printf -v "$key" '%s' "$val"
done <<<"$LIMITS"
LANE_DEFAULTS=$(wdpy env) || die "ppo.e2e.watchdog env failed"
LENV=(
  "HOME=$HOME"
  "PATH=/usr/local/bin:/usr/bin:/bin"
  "LANG=${LANG:-C.UTF-8}"
  "USER=${USER:-$(id -un)}"
  "PYTHONPATH=${PIN:+$PIN:}python"
)
[[ -n $PIN ]] && LENV+=("STS2_WHEEL_PIN=$PIN")
while IFS= read -r kv; do [[ -n $kv ]] && LENV+=("$kv"); done <<<"$LANE_DEFAULTS"
LENV+=("${LEARNER_ENV[@]}")

# Nobody edits a running worktree (plan section 8 Deploy).
HEAD_SHA=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) || HEAD_SHA=unknown
if [[ ${E2E_REQUIRE_CLEAN:-1} == 1 ]]; then
  [[ $HEAD_SHA != unknown ]] || { log "refusing: $ROOT is not a git worktree"; exit 2; }
  if [[ -n $(git -C "$ROOT" status --porcelain --untracked-files=no 2>&1) ]]; then
    log "refusing: worktree $ROOT has uncommitted changes"
    exit 2
  fi
fi
if [[ -n $PIN ]]; then
  SO=$(env -i "${LENV[@]}" "$PY" -c 'import os, sts2_rs; print(os.path.realpath(sts2_rs.__file__))' 2>&1) || {
    log "refusing: sts2_rs does not import from the pin: $SO"
    exit 2
  }
  [[ $SO == "$PIN"/* ]] || { log "refusing: sts2_rs imported from $SO, not from the pin $PIN"; exit 2; }
fi

STOP_REQ=""
trap 'STOP_REQ=TERM' TERM
trap 'STOP_REQ=INT' INT
trap 'STOP_REQ=HUP' HUP

now_ms() { date +%s%3N; }

# A sleep that a trapped signal interrupts.
isleep() {
  sleep "$1" &
  local s=$!
  wait "$s" 2>/dev/null
  kill "$s" 2>/dev/null
  wait "$s" 2>/dev/null
}

LPID="" RC=0 SINCE=0

# Wait for the learner (our child) and set RC; then SIGKILL what is left of
# its process group (workers, evals).
reap_learner() {
  local rc rc2
  while :; do
    wait "$LPID" 2>/dev/null
    rc=$?
    if ((rc > 128)); then
      kill -0 "$LPID" 2>/dev/null && continue # a trapped signal cut the wait short
      wait "$LPID" 2>/dev/null
      rc2=$?
      ((rc2 != 127)) && rc=$rc2
    fi
    break
  done
  RC=$rc
  kill -KILL -- "-$LPID" 2>/dev/null
}

# SIGTERM the learner; SIGKILL its process group after GRACE_MS.
stop_learner() {
  local grace_ms=$1 t0
  log "SIGTERM to learner $LPID (grace $((grace_ms / 1000)) s)"
  kill -TERM "$LPID" 2>/dev/null
  t0=$(now_ms)
  while kill -0 "$LPID" 2>/dev/null; do
    if (($(now_ms) - t0 >= grace_ms)); then
      log "learner $LPID still runs: SIGKILL to its process group"
      kill -KILL -- "-$LPID" 2>/dev/null
      break
    fi
    isleep 1
  done
  reap_learner
}

rotate_learner_log() {
  local size
  size=$(stat -c %s "$LLOG" 2>/dev/null) || return 0
  ((size > 1073741824)) && mv -f "$LLOG" "$LLOG.1"
  return 0
}

start_learner() { # RESUME OFFSET
  local resume=$1 offset=$2 cmd=()
  if ((${#CUSTOM[@]})); then
    cmd=("${CUSTOM[@]}")
  else
    cmd=("$PY" -u -m ppo.e2e.learner train)
  fi
  cmd+=(--save-dir "$SAVE" --sampler-seed-offset "$offset")
  [[ $resume == - ]] || cmd+=(--resume "$resume")
  cmd+=("${LANE_ARGS[@]}")
  rotate_learner_log
  printf '==== %s watchdog %s starts (HEAD %s): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    "$HEAD_SHA" "${cmd[*]}" >>"$LLOG"
  SINCE=$(date +%s.%N)
  setsid env -i "${LENV[@]}" "${cmd[@]}" </dev/null >>"$LLOG" 2>&1 &
  LPID=$!
  log "learner $LPID started (resume ${resume}, seed offset $offset, HEAD $HEAD_SHA)"
  wdpy started "$SAVE" --watchdog-pid $$ --pid "$LPID" --since "$SINCE" ||
    log "could not write ops/watchdog.json"
}

# Poll the learner until it exits or a rule acts. Sets OUTCOME (exit, signal,
# stop, stall, nan), RC, STOP_CODE and NAN_AT.
supervise() {
  local verdict v1 v2 fails=0
  OUTCOME="" STOP_CODE=0 NAN_AT=-1
  while :; do
    isleep "$WD_POLL_S"
    if [[ -n $STOP_REQ ]]; then
      log "SIG$STOP_REQ: stopping the learner"
      stop_learner "$WD_STOP_GRACE_MS"
      OUTCOME=signal
      return
    fi
    if ! kill -0 "$LPID" 2>/dev/null; then
      reap_learner
      OUTCOME=exit
      return
    fi
    if ! verdict=$(wdpy tick "$SAVE" --since "$SINCE" --pid "$LPID"); then
      fails=$((fails + 1))
      log "tick failed ($fails in a row); the learner runs on"
      continue
    fi
    fails=0
    read -r v1 v2 _ <<<"$verdict"
    case $v1 in
      ok) ;;
      stop)
        log "rule: $verdict"
        stop_learner "$WD_STOP_GRACE_MS"
        OUTCOME=stop STOP_CODE=$v2
        return
        ;;
      restart)
        log "rule: $verdict"
        stop_learner "$WD_STALL_STOP_GRACE_MS"
        OUTCOME=stall
        return
        ;;
      kill_nan)
        log "rule: NaN in stats.jsonl at $v2 decisions: SIGKILL, nothing saved"
        kill -KILL -- "-$LPID" 2>/dev/null
        reap_learner
        OUTCOME=nan NAN_AT=$v2
        return
        ;;
      *) log "tick: unexpected answer '$verdict'" ;;
    esac
  done
}

log "watchdog $$ for $SAVE (worktree $ROOT, HEAD $HEAD_SHA, python $PY, pin ${PIN:-none})"
wdpy started "$SAVE" --watchdog-pid $$ || log "could not write ops/watchdog.json"

while :; do
  if [[ -n $STOP_REQ ]]; then
    log "SIG$STOP_REQ before a start: exiting"
    exit 0
  fi
  verdict=$(wdpy prestart "$SAVE") || die "ppo.e2e.watchdog prestart failed"
  read -r v1 v2 v3 <<<"$verdict"
  case $v1 in
    done)
      log "DONE exists: not starting"
      exit 0
      ;;
    wait)
      log "prestart: wait ($v2)"
      isleep "$WD_POLL_S"
      continue
      ;;
    stop)
      log "prestart: $verdict"
      exit "$v2"
      ;;
    start) start_learner "$v2" "$v3" ;;
    *) die "prestart: unexpected answer '$verdict'" ;;
  esac

  supervise
  case $OUTCOME in
    signal) verdict=$(wdpy exited "$SAVE" --rc "$RC" --intent signal) ;;
    stop) verdict=$(wdpy exited "$SAVE" --rc "$RC" --intent stop) ;;
    stall) verdict=$(wdpy exited "$SAVE" --rc "$RC" --intent stall) ;;
    nan) verdict=$(wdpy exited "$SAVE" --rc "$RC" --nan "$NAN_AT") ;;
    *) verdict=$(wdpy exited "$SAVE" --rc "$RC") ;;
  esac || die "ppo.e2e.watchdog exited failed (learner rc $RC)"
  log "learner $LPID exited rc=$RC ($OUTCOME): $verdict"
  read -r v1 v2 _ <<<"$verdict"
  if [[ $OUTCOME == signal ]]; then
    [[ $v1 == stop ]] && exit "$v2"
    exit 0
  fi
  if [[ $OUTCOME == stop ]]; then
    exit "$STOP_CODE"
  fi
  case $v1 in
    done | stopped) exit 0 ;;
    stop) exit "$v2" ;;
    restart)
      log "restarting in $WD_RESTART_DELAY_S s"
      isleep "$WD_RESTART_DELAY_S"
      ;;
    *) die "exited: unexpected answer '$verdict'" ;;
  esac
done
