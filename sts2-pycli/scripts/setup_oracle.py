#!/usr/bin/env python3
"""Stage the game DLLs into lib/, IL-patch sts2.dll, and build the C# harness.

The .NET half of ``setup.sh``, in Python so it runs the same on Windows
(PowerShell or Git Bash), Linux and macOS. ``setup.sh`` and the monorepo's
``scripts/import_game.py`` (``just import-game``) both call it.

Usage:
    python scripts/setup_oracle.py [GAME_DATA_DIR] [--install-root DIR]

GAME_DATA_DIR is the folder holding ``sts2.dll`` (``data_sts2_*``). DLLs it
lacks are searched for under it, then under ``--install-root`` (the folder
holding the pck: the Windows build keeps some DLLs outside the data dir).
``0Harmony.dll``, if the install has none, is fetched from NuGet (Lib.Harmony,
MIT; ``STS2_HARMONY_VERSION`` overrides the pinned 2.4.1).

Without GAME_DATA_DIR and with a vendored, pre-patched ``lib/sts2.dll`` (the
private repo's git-lfs copy), the copy and patch steps are skipped and only the
harness is built. Stdlib only. ``DOTNET`` overrides the dotnet binary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYCLI = Path(__file__).resolve().parent.parent
LIB = PYCLI / "lib"
PATCHER = PYCLI / "tools" / "IlPatcher" / "IlPatcher.csproj"
HARNESS = PYCLI / "src" / "Sts2Headless" / "Sts2Headless.csproj"

# Files the C# harness links against.
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


def fetch_harmony(dotnet: str) -> bool:
    """Restore Lib.Harmony through the SDK's own NuGet client and copy 0Harmony.dll to lib/."""
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
        shutil.copy2(dlls[0], LIB / "0Harmony.dll")
        print(f"  ok 0Harmony.dll (Lib.Harmony {version} from NuGet)")
        return True


def copy_dlls(data_dir: Path, install_root: Path | None, dotnet: str) -> None:
    LIB.mkdir(exist_ok=True)
    roots = [data_dir] + ([install_root] if install_root and install_root.is_dir() else [])
    print("Copying DLLs to lib/ ...")
    for name in GAME_DLLS:
        src = find_dll(name, roots)
        if src:
            shutil.copy2(src, LIB / name)
            print(f"  ok {name}" + ("" if src.parent == data_dir else f"  (from {src})"))
        elif name == "0Harmony.dll":
            print("  .. 0Harmony.dll not in the install; fetching Lib.Harmony from NuGet")
            fetch_harmony(dotnet)
        else:
            print(f"  !! {name} not found under {', '.join(map(str, roots))} (the build may fail)")
    original = LIB / "sts2.dll.original"
    if (LIB / "sts2.dll").is_file() and not original.is_file():
        shutil.copy2(LIB / "sts2.dll", original)
        print("  ok backed up sts2.dll.original")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "data_dir", nargs="?", help="the game's data_sts2_* folder (holding sts2.dll)"
    )
    parser.add_argument(
        "--install-root", help="further folder to search for DLLs (the folder holding the pck)"
    )
    args = parser.parse_args(argv)
    dotnet = find_dotnet()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else None
    if data_dir and data_dir.is_dir():
        print(f"Game data: {data_dir}")
        copy_dlls(data_dir, Path(args.install_root) if args.install_root else None, dotnet)
        print("Applying IL patches to sts2.dll ...")
        run([dotnet, "run", "--project", str(PATCHER), "--", str(LIB / "sts2.dll")])
    elif (LIB / "sts2.dll").is_file():
        print("No game data dir given - using the vendored lib/ DLLs (already IL-patched).")
    else:
        raise SystemExit(
            f"game data dir not found: {data_dir}\n"
            "Pass the folder holding sts2.dll (data_sts2_*), or run `just import-game` from the repo root "
            "(docs/bring-your-own-game.md)."
        )

    print("Building the C# harness ...")
    run([dotnet, "build", str(HARNESS)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
