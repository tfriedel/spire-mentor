#!/usr/bin/env bash
# land.sh's release() against a throwaway bare origin (GH #1939):
# - our lock one heartbeat past $headfile is still released;
# - a lock another holder took is left alone.
# Usage: scripts/test-land-release.sh   (exit 0 = both hold)
set -u
here=$(cd "$(dirname "$0")" && pwd)
H=$(mktemp -d)
trap 'rm -r -- "$H"' EXIT
cd "$H" && git init -q --bare origin.git && git init -q clone && cd clone || exit 1
git remote add origin "$H/origin.git"
LOCK_REF=refs/heads/land-lock STALE=1200 branch=b pr=1 root=. since=now hb=999999
empty_tree=$(git hash-object -t tree /dev/null)
eval "$(sed -n '/^lock_state() {/,/^}/p; /^released() {/p; /^lock_msg() {/,/^}/p; /^release() {/,/^}/p' "$here/land.sh")"
lost=$H/lost
headfile=$(mktemp -p "$H")
fail=0
a=$(lock_msg | git commit-tree "$empty_tree")
git push -q origin "$a:$LOCK_REF"
echo "$a" >"$headfile"
b=$(lock_msg | git commit-tree "$empty_tree" -p "$a") # a heartbeat $headfile missed
git push -q origin "$b:$LOCK_REF"
release 2>/dev/null
git fetch -q origin "$LOCK_REF"
git log -1 --format=%B FETCH_HEAD | grep -qx 'state: released' || { echo "FAIL: our lock was not released"; fail=1; }
a=$(printf 'land-lock\n\nholder: other (pid 1)\n' | git commit-tree "$empty_tree")
git push -q -f origin "$a:$LOCK_REF"
headfile=$(mktemp -p "$H")
echo deadbeefdeadbeefdeadbeefdeadbeefdeadbeef >"$headfile"
release 2>/dev/null
git fetch -q origin "$LOCK_REF"
git log -1 --format=%B FETCH_HEAD | grep -qx 'holder: other (pid 1)' || { echo "FAIL: another holder's lock was released"; fail=1; }
[ "$fail" = 0 ] && echo "land release: ok"
exit "$fail"
