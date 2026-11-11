#!/usr/bin/env bash
# Compile-only build of SpireMentor against ONE game build's sts2.dll, on a box
# without the game installed (Linux dev box, the port gates). The DLL must then
# bind against EVERY game build present: one DLL ships for all of them.
#
# Usage: build_mod.sh <game-version> [--wire]
#   <game-version>  v0.107.1 | v0.111.0 -- which sts2.dll to compile against
#   --wire          then run the BridgeServer wire test (tests/wire) on the
#                   built DLL with that version's oracle libs
#
# Writes tests/compile/.build/<game-version>/mod/SpireMentor.dll. Nothing is
# deployed: the post-build copy into <game>/mods is switched off.
#
# Inputs:
#   STS2_BINARIES  dir holding <game-version>/data_sts2_windows_x86_64/ (the
#                  game's own sts2.dll, 0Harmony.dll, GodotSharp.dll). Default:
#                  <repo>/sts2_binaries, else the main checkout's (worktrees
#                  share it; it is gitignored).
#   GODOTSHARP_DLL GodotSharp.dll override. Default: that version's data dir,
#                  else any other version's -- the stripped v0.107.1 copy has
#                  none, and both builds ship Godot 4.5.1 (GodotSharp
#                  FileVersion 4.5.1.0; the oracle's GodotStubs pin the same).
#   BASELIB_DIR    a REAL BaseLib install (dir with BaseLib.dll). Default: the
#                  compile-only API stub in BaseLibStub/ (BaseLib is not
#                  vendored; install-windows-deps.ps1 fetches it on Windows).
#   DOTNET         dotnet executable (default: dotnet)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(cd "$HERE/../.." && pwd)"
REPO="$(cd "$MOD/.." && pwd)"
DOTNET="${DOTNET:-dotnet}"
die() { echo "build_mod.sh: ERROR: $*" >&2; exit 3; }

[[ $# -ge 1 ]] || { echo "usage: $0 <game-version> [--wire]" >&2; exit 64; }
VER="$1"; shift
WIRE=0
for a in "$@"; do
  case "$a" in
    --wire) WIRE=1 ;;
    *) die "unknown argument: $a" ;;
  esac
done
case "$VER" in
  v0.107.1|v0.111.0) ;;
  *) die "unsupported game version '$VER' (supported: v0.107.1, v0.111.0)" ;;
esac

if [[ -z "${STS2_BINARIES:-}" ]]; then
  STS2_BINARIES="$REPO/sts2_binaries"
  if [[ ! -d "$STS2_BINARIES" ]]; then
    common="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    [[ -n "$common" ]] && STS2_BINARIES="$(dirname "$common")/sts2_binaries"
  fi
fi
DATA="$STS2_BINARIES/$VER/data_sts2_windows_x86_64"
[[ -f "$DATA/sts2.dll" ]] || die "no game sts2.dll at $DATA (set STS2_BINARIES)"
[[ -f "$DATA/0Harmony.dll" ]] || die "no 0Harmony.dll at $DATA"

GODOT="${GODOTSHARP_DLL:-}"
if [[ -z "$GODOT" ]]; then
  for cand in "$DATA/GodotSharp.dll" "$STS2_BINARIES"/v*/data_sts2_windows_x86_64/GodotSharp.dll; do
    [[ -f "$cand" ]] && { GODOT="$cand"; break; }
  done
fi
[[ -n "$GODOT" && -f "$GODOT" ]] || die "no GodotSharp.dll found (set GODOTSHARP_DLL)"

BUILD="$HERE/.build/$VER"
STAGE="$BUILD/data_sts2_windows_x86_64"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ln -s "$(realpath "$DATA/sts2.dll")" "$STAGE/sts2.dll"
ln -s "$(realpath "$DATA/0Harmony.dll")" "$STAGE/0Harmony.dll"
ln -s "$(realpath "$GODOT")" "$STAGE/GodotSharp.dll"

if [[ -n "${BASELIB_DIR:-}" ]]; then
  [[ -f "$BASELIB_DIR/BaseLib.dll" ]] || die "BASELIB_DIR has no BaseLib.dll: $BASELIB_DIR"
  BASELIB="$(realpath "$BASELIB_DIR")"
else
  BASELIB="$BUILD/baselib-stub"
  "$DOTNET" build "$HERE/BaseLibStub/BaseLibStub.csproj" -c Release -nologo -v q \
    -p:Sts2DataDir="$STAGE" \
    -p:BaseIntermediateOutputPath="$BUILD/baselib-obj/" \
    -p:MSBuildProjectExtensionsPath="$BUILD/baselib-obj/" \
    -o "$BASELIB" >"$BUILD/baselib.build.log" 2>&1 \
    || { cat "$BUILD/baselib.build.log" >&2; die "BaseLib stub build failed"; }
fi

echo "build_mod.sh: compiling SpireMentor against $VER ($(realpath "$DATA/sts2.dll"))"
"$DOTNET" build "$MOD/SpireMentor.csproj" -c Release -nologo -v q \
\
  -p:Sts2DataDir="$STAGE" \
  -p:BaseLibDir="$BASELIB" \
  -p:SpireMentorSkipDeploy=true \
  -p:BaseIntermediateOutputPath="$BUILD/mod-obj/" \
  -p:MSBuildProjectExtensionsPath="$BUILD/mod-obj/" \
  -o "$BUILD/mod" >"$BUILD/mod.build.log" 2>&1 && rc=0 || rc=$?
grep -E ": error " "$BUILD/mod.build.log" | sort -u || true
echo "build_mod.sh: $(grep -E ": warning " "$BUILD/mod.build.log" | sort -u | wc -l) distinct warning(s) (log: $BUILD/mod.build.log)"
[[ "$rc" -eq 0 ]] || die "SpireMentor build against $VER failed (log: $BUILD/mod.build.log)"
[[ -f "$BUILD/mod/SpireMentor.dll" ]] || die "build reported success but $BUILD/mod/SpireMentor.dll is missing"
echo "build_mod.sh: OK $BUILD/mod/SpireMentor.dll"

# Binary binding: every sts2 type/method/field the DLL references must resolve
# by name AND signature in every game build's sts2.dll present, not just the one
# compiled against (a source compile does not prove that for members reached
# through `var`, defaults or overloads). A member whose signature differs
# between builds goes through SimGameApi.cs instead.
# Per game version, so run_all.sh can build the versions in parallel.
RC="$BUILD/refcheck"
"$DOTNET" build "$HERE/RefCheck/RefCheck.csproj" -c Release -nologo -v q   -p:BaseIntermediateOutputPath="$BUILD/refcheck-obj/"   -p:MSBuildProjectExtensionsPath="$BUILD/refcheck-obj/"   -o "$RC" >"$BUILD/refcheck.build.log" 2>&1   || { cat "$BUILD/refcheck.build.log" >&2; die "RefCheck build failed"; }
for other in "$STS2_BINARIES"/v*/data_sts2_windows_x86_64/sts2.dll; do
  [[ -f "$other" ]] || continue
  over="$(basename "$(dirname "$(dirname "$other")")")"
  "$DOTNET" "$RC/RefCheck.dll" "$BUILD/mod/SpireMentor.dll" "$other" --optional "$HERE/RefCheck/optional-names.txt" || die "SpireMentor compiled against $VER does not bind against $over"
done

if [[ "$WIRE" -eq 1 ]]; then
  GAME_VERSION="$VER" WIRE_BUILD="$BUILD/wire" \
    "$MOD/tests/wire/run_wire_test.sh" "$BUILD/mod/SpireMentor.dll" "$BUILD/wire-out"
fi
