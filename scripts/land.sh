#!/usr/bin/env bash
# Land the current branch's PR under a repo-wide lock: merge main, gate, merge.
#
#   just land [PR] [--full]     gate with `just gate` (or `just gate-full`)
#   just land-status            who holds the lock
#   scripts/land.sh record G    (the gate recipes call this) remember that
#                               this tree passed gate G
#
# Why: with no CI, the gate runs on the merge result. Agents used to gate,
# find main had moved under them, merge, re-gate, find it had moved again —
# up to five rounds. Holding the lock from the merge of main through the PR
# merge means main cannot move under a gate that runs inside it.
#
# The lock is the remote branch `land-lock` on origin, so it spans every
# machine and worktree that can push there. A push that creates the ref (or
# fast-forwards your own heartbeat chain) is atomic on the server, so two
# agents can't both win. The lock commit's message names the holder. The
# holder re-pushes a heartbeat every HEARTBEAT seconds; a lock with no
# heartbeat for STALE seconds (holder crashed, laptop asleep) can be taken
# over, with a lease on the stale sha so only one waiter wins that too.
#
# Releasing pushes one more commit onto the chain, marked `state: released`,
# rather than deleting the ref: a Claude Code cloud session's git proxy
# refuses ref deletions (while creates and fast-forwards go through), so a
# delete there failed silently and every land left the lock held until it
# went stale, 20 minutes on. A released lock is free; the next lander
# fast-forwards onto it. A ref that is absent (never taken, or deleted by
# an older land.sh) is free too.
#
# A merge that bypasses the lock (the GitHub UI, a cloud session that can't
# push the lock ref) still moves main. Then the post-gate check below sees it
# and goes round again, so the result is never ungated, only slower.
set -euo pipefail

LOCK_REF=refs/heads/land-lock
HEARTBEAT=${LAND_HEARTBEAT:-240}
STALE=${LAND_STALE:-1200}
POLL=${LAND_POLL:-30}

die() { echo >&2 "land: $*"; exit 1; }

root=$(git rev-parse --show-toplevel)
common=$(cd "$root" && cd "$(git rev-parse --git-common-dir)" && pwd)
passed=$common/land.gated-trees

clean_tree() { [ -z "$(git -C "$root" status --porcelain --untracked-files=no)" ]; }

# `record G`: the gate recipe G passed on this checkout's HEAD tree.
if [ "${1:-}" = record ]; then
  gate=${2:?usage: land.sh record GATE}
  if clean_tree; then
    echo "$gate $(git -C "$root" rev-parse 'HEAD^{tree}')" >>"$passed"
  else
    echo "land: uncommitted changes, so this $gate pass is not recorded" >&2
  fi
  exit 0
fi

# Prints "<sha> <committer epoch>\n<message>" for the lock ref, nothing if
# the ref is absent. A released lock prints too: see `released`.
lock_state() {
  local sha
  sha=$(git ls-remote origin "$LOCK_REF" | cut -f1)
  [ -n "$sha" ] || return 0
  git fetch -q origin "$LOCK_REF" 2>/dev/null || return 0
  git log -1 --format="$sha %ct%n%B" FETCH_HEAD
}

# Is the lock in `lock_state` output $1 released (free to take)?
released() { tail -n +2 <<<"$1" | grep -qx 'state: released'; }

if [ "${1:-}" = status ]; then
  st=$(lock_state)
  if [ -z "$st" ]; then echo "land-lock: free"; exit 0; fi
  if released "$st"; then
    echo "land-lock: free (released)"
    tail -n +2 <<<"$st" | sed '/^$/d; /^land-lock$/d; s/^/  /'
    exit 0
  fi
  age=$(($(date +%s) - $(head -1 <<<"$st" | cut -d' ' -f2)))
  echo "land-lock: held (last heartbeat ${age}s ago, stale after ${STALE}s)"
  tail -n +2 <<<"$st" | sed '/^$/d; s/^/  /'
  exit 0
fi

gate=gate
pr=
for a in "$@"; do
  case $a in
    --full) gate=gate-full ;;
    [0-9]*) pr=$a ;;
    *) die "usage: just land [PR] [--full]" ;;
  esac
done

branch=$(git -C "$root" symbolic-ref --short HEAD) || die "detached HEAD"
[ "$branch" != main ] || die "run it on the PR's branch, not main"
clean_tree || die "commit or stash your changes first"
[ -n "$pr" ] || pr=$(gh pr view "$branch" --json number -q .number) || die "no PR for $branch"
[ "$(gh pr view "$pr" --json headRefName -q .headRefName)" = "$branch" ] ||
  die "PR #$pr is not branch $branch"

empty_tree=$(git hash-object -t tree -w /dev/null)
lock_msg() {
  printf 'land-lock\n\nholder: %s (pid %s)\nbranch: %s\npr: #%s\nworktree: %s\nsince: %s\n' \
    "$(hostname)" $$ "$branch" "$pr" "$root" "$since"
}
since=$(date -Is)
mine=

# Take the lock: create the ref, fast-forward a released one, or replace a
# stale one; the last two under a lease on the sha we saw.
while :; do
  st=$(lock_state)
  if [ -z "$st" ]; then
    c=$(lock_msg | git commit-tree "$empty_tree")
    if git push -q origin "$c:$LOCK_REF" 2>/dev/null; then mine=$c; break; fi
    continue  # someone beat us to it; look again
  fi
  held=$(head -1 <<<"$st" | cut -d' ' -f1)
  if released "$st"; then
    c=$(lock_msg | git commit-tree "$empty_tree" -p "$held")
    if git push -q --force-with-lease="$LOCK_REF:$held" origin "$c:$LOCK_REF" 2>/dev/null; then
      mine=$c
      break
    fi
    continue  # someone beat us to it; look again
  fi
  age=$(($(date +%s) - $(head -1 <<<"$st" | cut -d' ' -f2)))
  if [ "$age" -gt "$STALE" ]; then
    echo "land: taking over a stale lock (no heartbeat for ${age}s):"
    tail -n +2 <<<"$st" | sed '/^$/d; s/^/  /'
    c=$(lock_msg | git commit-tree "$empty_tree")
    if git push -q --force-with-lease="$LOCK_REF:$held" origin "$c:$LOCK_REF" 2>/dev/null; then
      mine=$c
      break
    fi
    continue
  fi
  # Compare the holder's message, not the sha: every heartbeat is a new
  # commit with the same message, so the sha would reprint the holder each time.
  holder=$(tail -n +2 <<<"$st")
  if [ "$holder" != "${last_seen:-}" ]; then
    echo "land: waiting for the lock, held by:"
    sed '/^$/d; s/^/  /' <<<"$holder"
    last_seen=$holder
  fi
  sleep "$POLL"
done
echo "land: lock taken for #$pr ($branch)"

# Heartbeat: fast-forward our own chain. A failed push means we lost it.
lost=$(mktemp)
rm -f "$lost"
headfile=$(mktemp)
echo "$mine" >"$headfile"
# The trap kills the pending `sleep` too: orphaned, it would hold the
# caller's stdout open (a `just land | tee` would hang until it ended).
# It only flags the stop: bash runs a trap after the running command, so a
# TERM during a push that then exits in the trap would leave the pushed
# commit out of $headfile, and release()'s lease would fail on it.
(
  sp= stop=
  trap 'stop=1; kill $sp 2>/dev/null' TERM
  while [ -z "$stop" ]; do
    sleep "$HEARTBEAT" &
    sp=$!
    wait "$sp"
    [ -z "$stop" ] || break
    prev=$(cat "$headfile")
    c=$(lock_msg | git commit-tree "$empty_tree" -p "$prev")
    if git push -q origin "$c:$LOCK_REF" 2>/dev/null; then
      echo "$c" >"$headfile"
    else
      : >"$lost"
      exit 0
    fi
  done
) </dev/null >/dev/null 2>&1 &
hb=$!

release() {
  local head c err
  kill "$hb" 2>/dev/null || true
  wait "$hb" 2>/dev/null || true
  if [ ! -e "$lost" ]; then
    head=$(cat "$headfile")
    c=$({ lock_msg; echo "state: released"; echo "released: $(date -Is)"; } |
      git commit-tree "$empty_tree" -p "$head")
    # The push's own error is shown: a silent failure here is what hid the
    # cloud proxy's refusal of the old ref delete.
    if ! err=$(git push -q --force-with-lease="$LOCK_REF:$head" origin "$c:$LOCK_REF" 2>&1); then
      # The ref moved past $headfile (GH #1939: a heartbeat push recorded
      # nowhere, cause unknown). Still ours if it names this holder and is
      # not released: release it under a lease on what it is now.
      local st cur
      st=$(lock_state)
      cur=${st%% *}
      if [ -n "$cur" ] && ! released "$st" &&
        tail -n +2 <<<"$st" | grep -qxF "holder: $(hostname) (pid $$)"; then
        c=$({ lock_msg; echo "state: released"; echo "released: $(date -Is)"; } |
          git commit-tree "$empty_tree" -p "$cur")
        git push -q --force-with-lease="$LOCK_REF:$cur" origin "$c:$LOCK_REF" 2>/dev/null && err=
      fi
      if [ -n "$err" ]; then
        echo >&2 "land: could not release the lock; it goes stale in ${STALE}s"
        sed 's/^/  /' <<<"$err" >&2
      fi
    fi
  fi
  rm -f "$lost" "$headfile"
}
trap release EXIT
trap 'exit 130' INT TERM

still_mine() {
  [ ! -e "$lost" ] && [ "$(git ls-remote origin "$LOCK_REF" | cut -f1)" = "$(cat "$headfile")" ]
}

cd "$root"
while :; do
  git fetch -q origin main
  if ! git merge-base --is-ancestor origin/main HEAD; then
    echo "land: merging origin/main"
    git merge --no-edit origin/main || {
      git merge --abort
      die "merge conflict with main; resolve it, commit, and run land again"
    }
  fi
  tree=$(git rev-parse 'HEAD^{tree}')
  if grep -qxF "$gate $tree" "$passed" 2>/dev/null; then
    echo "land: this tree already passed $gate; not running it again"
  else
    just "$gate"  # records the tree itself when it passes
  fi
  git push -q origin "HEAD:$branch"
  git fetch -q origin main
  if git merge-base --is-ancestor origin/main HEAD; then break; fi
  echo "land: main moved during the gate (a merge outside the lock); again"
done

still_mine || die "lost the lock during the gate (heartbeat failed or it was taken over)"
gh pr merge "$pr" --merge --match-head-commit "$(git rev-parse HEAD)"
echo "land: merged #$pr"
