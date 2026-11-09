#!/usr/bin/env bash
# Serialised `git lfs pull` (GH #1162). The SessionStart hooks
# (claude_code_web_init.sh and the data unpack.sh scripts) start in parallel,
# and concurrent pulls race on .git/index.lock: one exits 128 and the caller's
# `|| true` hides it, leaving pointer stubs behind. Every hook pulls through
# this wrapper, which holds one lock per repository.
#
# Usage: scripts/lfs-pull.sh <dir-in-repo> [git lfs pull args...]
set -u
DIR="$1"; shift
command -v git-lfs >/dev/null 2>&1 || exit 0
COMMON="$(git -C "$DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || exit 0
if command -v flock >/dev/null 2>&1; then
  exec flock "$COMMON/lfs-pull.lock" git -C "$DIR" lfs pull "$@"
fi
exec git -C "$DIR" lfs pull "$@"
