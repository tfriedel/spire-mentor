#!/usr/bin/env python3
"""Stage the game DLLs into lib/<version>/, IL-patch sts2.dll, and build the C# harness.

The .NET half of ``setup.sh``, in Python so it runs the same on Windows
(PowerShell or Git Bash), Linux and macOS. ``setup.sh`` and the monorepo's
``scripts/import_game.py`` (``just import-game``) both call it.

Usage:
    python scripts/setup_oracle.py [GAME_DATA_DIR] [--version VER] [--install-root DIR]

The oracle is built once per supported game build (GH #789): ``--version``
(default ``$STS2_GAME_VERSION`` or v0.107.1) picks ``lib/<version>/`` for the
DLLs and ``-p:GameVersion=<version>`` for the build. When the install's
``release_info.json`` names a different build, the script refuses (pass
``--allow-version-mismatch`` to stage it anyway).

GAME_DATA_DIR is the folder holding ``sts2.dll`` (``data_sts2_*``). DLLs it
lacks are searched for under it, then under ``--install-root`` (the folder
holding the pck: the Windows build keeps some DLLs outside the data dir).
``0Harmony.dll``, if the install has none, is fetched from NuGet (Lib.Harmony,
MIT; ``STS2_HARMONY_VERSION`` overrides the pinned 2.4.1).

Without GAME_DATA_DIR and with a vendored, pre-patched ``lib/<version>/sts2.dll``
(the private repo's git-lfs copy), the copy and patch steps are skipped and only
the harness is built. Stdlib only. ``DOTNET`` overrides the dotnet binary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYCLI = Path(__file__).resolve().parent.parent
LIB_ROOT = PYCLI / "lib"
PATCHER = PYCLI / "tools" / "IlPatcher" / "IlPatcher.csproj"
HARNESS = PYCLI / "src" / "Sts2Headless" / "Sts2Headless.csproj"

# Mirrors sts2_client.SUPPORTED_GAME_VERSIONS and the csproj's version check.
DEFAULT_VERSION = "v0.107.1"
SUPPORTED_VERSIONS = ("v0.107.1", "v0.111.0", "v0.113.0")

# Files the C# harness links against or loads at run time.
GAME_DLLS = [
    "sts2.dll",
    "SmartFormat.dll",
    "SmartFormat.ZString.dll",
    "Sentry.dll",
    "Steamworks.NET.dll",
    "MonoMod.Backports.dll",
    "MonoMod.ILHelpers.dll",
    "0Harmony.dll",
    "System.IO.Hashing.dll",
]
# Per-build additions: assemblies a build's sts2.dll needs at run time that
# the previous build did not reference.
EXTRA_DLLS: dict[str, list[str]] = {
    "v0.107.1": [],
    # sts2.dll v0.111.0 references Sentry.Godot (the Godot Sentry integration).
    "v0.111.0": ["Sentry.Godot.dll"],
    "v0.113.0": ["Sentry.Godot.dll"],
}


def game_dlls(version: str) -> list[str]:
    return GAME_DLLS + EXTRA_DLLS.get(version, [])


def normalize_version(version: str) -> str:
    v = version.strip()
    if v and not v.startswith("v"):
        v = "v" + v
    if v not in SUPPORTED_VERSIONS:
        raise SystemExit(
            f"unsupported game version {version!r} (supported: {', '.join(SUPPORTED_VERSIONS)})"
        )
    return v


def lib_dir(version: str) -> Path:
    return LIB_ROOT / version


def find_dotnet() -> str:
    """$DOTNET, then the per-user installs (~/.dotnet-arm64, ~/.dotnet), then PATH."""
    if os.environ.get("DOTNET"):
        return os.environ["DOTNET"]
    exe = "dotnet.exe" if os.name == "nt" else "dotnet"
    for base in (Path.home() / ".dotnet-arm64", Path.home() / ".dotnet"):
        if (base / exe).is_file():
            return str(base / exe)
    found = shutil.which("dotnet")
    if not found:
        raise SystemExit(
            ".NET SDK not found. Install the .NET 9 SDK (https://dotnet.microsoft.com/download) "
            "or set DOTNET to your dotnet binary."
        )
    return found


def run(cmd: list[str], cwd: Path = PYCLI) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")


def find_dll(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if (root / name).is_file():
            return root / name
    for root in roots:
        hit = next(root.rglob(name), None) if root.is_dir() else None
        if hit:
            return hit
    return None


def install_version(data_dir: Path, install_root: Path | None) -> str | None:
    """The ``version`` in the install's ``release_info.json``, if one is found."""
    for root in [data_dir, data_dir.parent] + ([install_root] if install_root else []):
        info = root / "release_info.json"
        if info.is_file():
            try:
                return str(json.loads(info.read_text(encoding="utf-8"))["version"])
            except (OSError, ValueError, KeyError):
                return None
    return None


def fetch_harmony(dotnet: str, lib: Path) -> bool:
    """Restore Lib.Harmony through the SDK's own NuGet client and copy 0Harmony.dll to lib."""
    version = os.environ.get("STS2_HARMONY_VERSION", "2.4.1")
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "Harmony.csproj"
        proj.write_text(
            '<Project Sdk="Microsoft.NET.Sdk">\n'
            "  <PropertyGroup><TargetFramework>net9.0</TargetFramework></PropertyGroup>\n"
            f'  <ItemGroup><PackageReference Include="Lib.Harmony" Version="{version}" /></ItemGroup>\n'
            "</Project>\n",
            encoding="utf-8",
        )
        pkgs = Path(tmp) / "pkgs"
        proc = subprocess.run(
            [dotnet, "restore", str(proj), "--packages", str(pkgs)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            print(f"  x could not restore Lib.Harmony {version} from NuGet (offline?)")
            return False
        dlls = sorted(pkgs.rglob("0Harmony.dll"), key=lambda p: "net9.0" not in p.as_posix())
        if not dlls:
            print(f"  x Lib.Harmony {version} restored but no 0Harmony.dll inside")
            return False
        shutil.copy2(dlls[0], lib / "0Harmony.dll")
        print(f"  ok 0Harmony.dll (Lib.Harmony {version} from NuGet)")
        return True


def copy_dlls(data_dir: Path, install_root: Path | None, dotnet: str, version: str) -> None:
    lib = lib_dir(version)
    lib.mkdir(parents=True, exist_ok=True)
    roots = [data_dir] + ([install_root] if install_root and install_root.is_dir() else [])
    print(f"Copying DLLs to lib/{version}/ ...")
    for name in game_dlls(version):
        src = find_dll(name, roots)
        if src:
            shutil.copy2(src, lib / name)
            print(f"  ok {name}" + ("" if src.parent == data_dir else f"  (from {src})"))
        elif name == "0Harmony.dll":
            print("  .. 0Harmony.dll not in the install; fetching Lib.Harmony from NuGet")
            fetch_harmony(dotnet, lib)
        else:
            print(f"  !! {name} not found under {', '.join(map(str, roots))} (the build may fail)")
    # Back up the pristine sts2.dll. A re-stage overwrites the backup, because
    # the copy above just replaced sts2.dll with an unpatched one: keeping an
    # older backup would let a different build's assembly masquerade as this
    # one's original.
    shutil.copy2(lib / "sts2.dll", lib / "sts2.dll.original")
    print("  ok backed up sts2.dll.original")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "data_dir", nargs="?", help="the game's data_sts2_* folder (holding sts2.dll)"
    )
    parser.add_argument(
        "--version",
        default=os.environ.get("STS2_GAME_VERSION") or DEFAULT_VERSION,
        help=f"game build to stage + build ({', '.join(SUPPORTED_VERSIONS)}; "
        f"default: $STS2_GAME_VERSION or {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--install-root", help="further folder to search for DLLs (the folder holding the pck)"
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="stage the install even though its release_info.json names another build",
    )
    parser.add_argument(
        "--no-build", action="store_true", help="stage + patch only; skip the harness build"
    )
    args = parser.parse_args(argv)
    version = normalize_version(args.version)
    dotnet = find_dotnet()
    lib = lib_dir(version)

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else None
    install_root = Path(args.install_root).expanduser() if args.install_root else None
    if data_dir and data_dir.is_dir():
        found = install_version(data_dir, install_root)
        if found and found != version and not args.allow_version_mismatch:
            raise SystemExit(
                f"the install at {data_dir} is {found}, but --version is {version}: its DLLs "
                f"would land in lib/{version}/ and the oracle would link the wrong build. "
                f"Pass --version {found}, or --allow-version-mismatch to stage it anyway."
            )
        print(f"Game data: {data_dir}  (build {found or 'unknown'} -> lib/{version}/)")
        copy_dlls(data_dir, install_root, dotnet, version)
        print("Applying IL patches to sts2.dll ...")
        run([dotnet, "run", "--project", str(PATCHER), "--", str(lib / "sts2.dll")])
    elif (lib / "sts2.dll").is_file():
        print(
            f"No game data dir given - using the vendored lib/{version}/ DLLs (already IL-patched)."
        )
    elif version == DEFAULT_VERSION and (LIB_ROOT / "sts2.dll").is_file():
        print(
            f"No game data dir given - using the LEGACY flat lib/ DLLs; move them to lib/{version}/ "
            "(GH #789). This fallback will be removed."
        )
    else:
        raise SystemExit(
            f"game data dir not found: {data_dir}\n"
            "Pass the folder holding sts2.dll (data_sts2_*), or run `just import-game` from the repo root "
            "(docs/bring-your-own-game.md)."
        )

    if args.no_build:
        return 0
    print(f"Building the C# harness for {version} ...")
    run([dotnet, "build", str(HARNESS), f"-p:GameVersion={version}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
