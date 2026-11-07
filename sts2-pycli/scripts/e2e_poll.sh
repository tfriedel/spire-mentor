#!/usr/bin/env bash
# e2e_poll.sh: one poll of the Ironclad e2e campaign host (#1097), run from
# the laptop every 30 min (plan section 8 Runbook).
#
# Usage:
#
#   scripts/e2e_poll.sh [HOST]        HOST defaults to $E2E_POLL_HOST, else a4000
#
# It runs `tailscale ping` first: an answer "via DERP" means a relayed, slow
# link, not a dead box. Then it reads status.json and every ALARM_* file of the
# campaign dir over ssh and prints one line.
#
# Exit codes:
#   0  OK
#   1  an OPERATOR alarm
#   2  a HUMAN alarm (the user decides)
#   3  status.json is older than 20 min, or missing (the driver is not running)
#   4  the host does not answer ssh
#
# Environment:
#   E2E_POLL_SSH        ssh command (default ssh)
#   E2E_POLL_TAILSCALE  tailscale command (default tailscale; skipped if missing)
#   E2E_POLL_PY         python3 for parsing (default python3)
#   E2E_POLL_STALE_S    status age that counts as stale (default 1200)
#   E2E_POLL_CAMPAIGN   campaign dir on the host (default ~/runs/e2e there)
#   E2E_POLL_LOG        append each line, time-stamped, to this file
set -uo pipefail

HOST=${1:-${E2E_POLL_HOST:-a4000}}
SSH=${E2E_POLL_SSH:-ssh}
TS=${E2E_POLL_TAILSCALE:-tailscale}
PY=${E2E_POLL_PY:-python3}
STALE=${E2E_POLL_STALE_S:-1200}
REMOTE_DIR=${E2E_POLL_CAMPAIGN:-}

LINK=""
if command -v "$TS" >/dev/null 2>&1; then
  if PING=$("$TS" ping -c 1 --timeout 10s "$HOST" 2>&1); then
    [[ $PING == *"via DERP"* ]] && LINK="link relayed via DERP (slow, not down)"
  else
    LINK="tailscale ping failed"
  fi
fi

# What runs on the host: no python needed there, only coreutils.
read -r -d '' REMOTE <<'EOF'
D=${1:-$HOME/runs/e2e}
echo "NOW $(date +%s)"
if [ -f "$D/status.json" ]; then
  echo "STATUS_BEGIN"
  cat "$D/status.json"
  echo
  echo "STATUS_END"
fi
for f in "$D"/ALARM_* "$D"/lanes/*/ALARM_*; do
  [ -f "$f" ] || continue
  echo "ALARM_BEGIN $f"
  cat "$f"
  echo
  echo "ALARM_END"
done
EOF

OUT=$(printf '%s\n' "$REMOTE" | timeout 90 "$SSH" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
  bash -s -- "$REMOTE_DIR" 2>/dev/null)
RC=$?

emit() { # LINE CODE
  printf '%s\n' "$1"
  if [[ -n ${E2E_POLL_LOG:-} ]]; then
    printf '%s rc=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" "$1" >>"$E2E_POLL_LOG"
  fi
  exit "$2"
}

# A login banner from the host's shell rc may precede the NOW line.
if ((RC != 0)) || [[ $'\n'$OUT != *$'\n'"NOW "* ]]; then
  emit "e2e UNREACHABLE | ssh $HOST failed (rc $RC)${LINK:+ | $LINK}" 4
fi

LINE=$(printf '%s' "$OUT" | E2E_LINK="$LINK" E2E_STALE="$STALE" "$PY" -c '
import json, os, sys

text = sys.stdin.read()
now, status, alarms = None, None, []
block, buf, path = None, [], None
for line in text.splitlines():
    if block is None and line.startswith("NOW "):
        now = float(line.split()[1])
    elif block is None and line == "STATUS_BEGIN":
        block, buf = "status", []
    elif block is None and line.startswith("ALARM_BEGIN "):
        block, buf, path = "alarm", [], line.split(" ", 1)[1]
    elif block == "status" and line == "STATUS_END":
        try:
            status = json.loads("\n".join(buf))
        except ValueError:
            status = {}
        block = None
    elif block == "alarm" and line == "ALARM_END":
        try:
            rec = json.loads("\n".join(buf))
        except ValueError:
            rec = {}
        rec = rec if isinstance(rec, dict) else {}
        name = os.path.basename(path)[len("ALARM_"):]
        lane = rec.get("lane") or os.path.basename(os.path.dirname(path))
        # An alarm without a readable severity counts as HUMAN.
        alarms.append((rec.get("severity") or "HUMAN", name, lane))
        block = None
    elif block is not None:
        buf.append(line)
human = sorted({f"{n}({ln})" for s, n, ln in alarms if s == "HUMAN"})
oper = sorted({f"{n}({ln})" for s, n, ln in alarms if s != "HUMAN"})
parts, code = [], 0
age = None
if isinstance(status, dict) and isinstance(status.get("time"), (int, float)) and now:
    age = now - status["time"]
stale = age is None or age > float(os.environ["E2E_STALE"])
if human:
    code = 2
elif stale:
    code = 3
elif oper:
    code = 1
verdict = {0: "OK", 1: "OPERATOR", 2: "HUMAN", 3: "STALE"}[code]
parts.append(f"e2e {verdict}")
if isinstance(status, dict) and status.get("summary"):
    parts.append(str(status["summary"]))
if human:
    parts.append("HUMAN: " + ", ".join(human))
if oper:
    parts.append("OPERATOR: " + ", ".join(oper))
parts.append("no status.json" if age is None else f"status {age / 60:.0f} min old")
if os.environ.get("E2E_LINK"):
    parts.append(os.environ["E2E_LINK"])
print(" | ".join(parts))
sys.exit(code)
')
emit "$LINE" "$?"
