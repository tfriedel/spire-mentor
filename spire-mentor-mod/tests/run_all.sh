#!/usr/bin/env bash
# The mod's local gate on a box without the game installed: for EVERY
# supported game build, compile SpireMentor against that build's sts2.dll,
# check its binary binding (tests/compile/build_mod.sh) and run the
# BridgeServer wire test on the result (tests/wire); then the game-free unit
# tests (tests/unit). Exit 0 only if all of it passes.
#
# Usage: spire-mentor-mod/tests/run_all.sh [<game-version>...]
#   default: v0.107.1 v0.111.0
# Needs the .NET 9 SDK, sts2_binaries/<ver>/ (see build_mod.sh) and the
# oracle libs in sts2-pycli/lib/<ver>/ (git lfs pull) for the wire test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
versions=("$@")
[[ ${#versions[@]} -gt 0 ]] || versions=(v0.107.1 v0.111.0 v0.113.0)
# The DLL's version const must match the manifest (package.ps1 refuses too).
manifest_version="$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$HERE/../SpireMentor.json" | head -1)"
code_version="$(sed -n 's/.*ModVersion = "\([^"]*\)".*/\1/p' "$HERE/../MainFile.cs" | head -1)"
if [[ "$manifest_version" != "$code_version" ]]; then
  echo "run_all.sh: MainFile.ModVersion '$code_version' != SpireMentor.json '$manifest_version'" >&2
  exit 1
fi
# The versions build in parallel (each writes only tests/compile/.build/<ver>/);
# each one's output is buffered and printed whole, in order, once all finish.
logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT
pids=()
for v in "${versions[@]}"; do
  "$HERE/compile/build_mod.sh" "$v" --wire >"$logs/$v.log" 2>&1 &
  pids+=($!)
done
failed=()
for i in "${!versions[@]}"; do
  v="${versions[$i]}"
  rc=0; wait "${pids[$i]}" || rc=$?
  echo "=== SpireMentor for $v: build + binding + wire test"
  cat "$logs/$v.log"
  [[ "$rc" -eq 0 ]] || failed+=("$v")
done
[[ ${#failed[@]} -eq 0 ]] || { echo "run_all.sh: FAILED for ${failed[*]}" >&2; exit 1; }
echo "=== SpireMentor unit tests"
"$HERE/unit/run_unit_tests.sh"
echo "run_all.sh: OK (${versions[*]})"
