#!/usr/bin/env bash
# Start (or stop) the servers the Playwright specs attach to, from a worktree.
#
#   scripts/dev-e2e.sh        start whatever is not already answering
#   scripts/dev-e2e.sh stop   STS2_VITE_PORT unset: stop vite on :5173 AND the
#                             backend on :8077 (the old behaviour).
#                             STS2_VITE_PORT set: stop only vite on that port;
#                             the shared backend on :8077 stays up.
#
#   STS2_VITE_PORT=5191 scripts/dev-e2e.sh   run vite on your own port, so a
#   gate run in another worktree on :5173 is neither reused nor killed; point
#   Playwright at it with STS2_E2E_BASE=http://localhost:5191.
#
# Why it exists: playwright.config.ts runs the backend from `..` (the worktree),
# but a worktree's art files are git-LFS pointers, so the backend must run from
# the MAIN checkout to serve real art. Vite runs from this worktree so the
# specs test this worktree's frontend. Both ports are `reuseExistingServer`.
set -euo pipefail

WEB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The main checkout: the git dir every worktree shares sits in it (a plain
# checkout is its own main). STS2_MAIN_PYCLI overrides.
MAIN_PYCLI="${STS2_MAIN_PYCLI:-$(dirname "$(git -C "$WEB" rev-parse --path-format=absolute --git-common-dir)")/sts2-pycli}"
LOGS="$WEB/test-results"
BACKEND_PORT=8077
VITE_PORT=${STS2_VITE_PORT:-5173}
mkdir -p "$LOGS"

if [[ "${1:-}" == "stop" ]]; then
  fuser -k "$VITE_PORT/tcp" 2>/dev/null || true
  if [[ -z "${STS2_VITE_PORT:-}" ]]; then
    fuser -k "$BACKEND_PORT/tcp" 2>/dev/null || true
    echo "stopped servers on :$VITE_PORT and :$BACKEND_PORT"
  else
    echo "stopped vite on :$VITE_PORT (the shared backend on :$BACKEND_PORT stays up)"
  fi
  exit 0
fi

answers() { curl -fsS -o /dev/null --max-time 3 "$1" 2>/dev/null; }

if answers "http://127.0.0.1:$BACKEND_PORT/api/health"; then
  echo "backend already up on :$BACKEND_PORT"
else
  (
    cd "$MAIN_PYCLI"
    STS2_HOST=127.0.0.1 STS2_PORT=$BACKEND_PORT STS2_RELOAD=0 \
      STS2_SKIP_WHEEL_CHECK=1 PYTHONPATH=python \
      setsid nohup .venv/bin/python -m workbench \
      < /dev/null > "$LOGS/backend.log" 2>&1 &
  )
  echo "backend starting (log: $LOGS/backend.log)"
fi

if answers "http://localhost:$VITE_PORT/"; then
  echo "vite already up on :$VITE_PORT"
else
  (
    cd "$WEB"
    STS2_BACKEND="http://127.0.0.1:$BACKEND_PORT" \
      setsid nohup npx vite --port "$VITE_PORT" --strictPort \
      < /dev/null > "$LOGS/vite.log" 2>&1 &
  )
  echo "vite starting (log: $LOGS/vite.log)"
fi

for _ in $(seq 1 90); do
  if answers "http://127.0.0.1:$BACKEND_PORT/api/health" && answers "http://localhost:$VITE_PORT/"; then
    echo "both servers answering"
    exit 0
  fi
  sleep 1
done
echo "servers did not come up in 90s; see $LOGS" >&2
exit 1
