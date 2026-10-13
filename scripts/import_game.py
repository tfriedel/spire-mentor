#!/usr/bin/env python3
"""Bring your own game: regenerate every game-derived input from a local install.

This repository ships no game content. Everything the C# oracle, the parity
harness and the workbench need from Slay the Spire 2 — the game assembly,
its localization, the decompiled source the extractors read, card/monster
data, art, sound — is produced HERE, from the copy of the game you own, into
paths that are git-ignored in the public repository.

Usage:
    just import-game [GAME_DIR] [--tier oracle|workbench|sound|all]
    just import-game --status
    uv run --no-project scripts/import_game.py [GAME_DIR] [--tier ...]

Runs the same on Windows (PowerShell or Git Bash), Linux and macOS: every step
calls its tool directly (`uv run`, `dotnet`, `ilspycmd`) — no bash, no `just`.

GAME_DIR is the Steam install (the folder holding SlayTheSpire2.pck), the
macOS .app, or the data_sts2_* folder holding sts2.dll; omitted, the usual
Steam locations and $STS2_GAME_DIR are tried.

Tiers (each includes the ones before it):
    engine     nothing to import — the Rust workspace builds and tests as-is
    oracle     sts2-pycli/lib/ (IL-patched game DLLs) + localization_eng/
               -> the C# oracle, the parity harness, the full pytest suite
    workbench  + decompiled source, wiki data JSON, art (atlases, VFX, rooms
               and scenes, card frames, creature rigs), the sfx map, card
               text / card db, the workbench's embedded data bundle
               -> the Python and browser workbenches render real game
               text and art; `just workbench-local` next
    sound      + the sound effects: rendered through FMOD — on Windows the
               game's own fmod.dll, elsewhere the FMOD Studio API
               (FMOD_SDK_ROOT); skipped with a note when neither is there
    all        = sound

Without --tier the import runs everything it can: the sound tier is skipped
with a note when ffmpeg is not installed. Asking for --tier sound or --tier
all explicitly makes a missing ffmpeg an error instead.

Every step is idempotent (re-running skips what exists) and logged to
logs/import-game.log. Read docs/bring-your-own-game.md first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARIES = ROOT / "sts2_binaries"
WIKI = ROOT / "sts2-wiki"
PYCLI = ROOT / "sts2-pycli"
RS = ROOT / "sts2-rs"
LOG = ROOT / "logs" / "import-game.log"

# The build the engine simulates. sts2-rs/crates/sts2-workbench/src/data.rs
# (ENGINE_WIKI_VERSION) and sts2-pycli/python/workbench/localization.py
# (_ENGINE_WIKI_VERSION) pin the same value; move all three together.
PINNED_VERSION = "v0.107.1"

TIERS = ["engine", "oracle", "workbench", "sound"]

# Files the C# harness links against (sts2-pycli/scripts/setup_oracle.py copies + patches).
GAME_DLLS = [
    "sts2.dll", "SmartFormat.dll", "SmartFormat.ZString.dll", "Sentry.dll", "Steamworks.NET.dll",
    "MonoMod.Backports.dll", "MonoMod.ILHelpers.dll", "0Harmony.dll", "System.IO.Hashing.dll",
]

# The FMOD runtime the Windows game installs beside SlayTheSpire2.exe
# (sts2-wiki/scripts/extract_sfx.py loads it through FMOD_LIB_DIR).
FMOD_DLLS = ("fmod.dll", "fmodstudio.dll")

# ilspycmd 10+ needs a newer runtime than the .NET 9 SDK the rest of the repo
# pins; 9.1 is the last release that installs and runs on .NET 9.
ILSPYCMD_INSTALL = "dotnet tool install -g ilspycmd --version 9.1.0.7988"


class Step:
    """One idempotent step: `done()` says whether its output exists."""

    def __init__(self, name: str, tier: str, done, run, needs: tuple[str, ...] = ()):
        self.name, self.tier, self.done, self.run, self.needs = name, tier, done, run, needs


# --------------------------------------------------------------------------
# Logging / shell
# --------------------------------------------------------------------------


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sh(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    log(f"$ (cd {cwd.relative_to(ROOT) if cwd != ROOT else '.'} && {' '.join(cmd)})")
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    with LOG.open("a", encoding="utf-8") as fh:
        # The children print UTF-8 (emoji, arrows); decoding with the locale
        # codec (cp1252 on Windows) would crash on the first such byte.
        proc = subprocess.run(cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        fh.write(proc.stdout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise SystemExit(f"step failed ({proc.returncode}): {' '.join(cmd)}\n{tail}\n(full log: {LOG})")


def uv_python(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run Python in the uv project at `cwd` (its own environment and dependencies)."""
    sh(["uv", "run", "python", *args], cwd, env)


def which(tool: str) -> str | None:
    """`tool` on PATH, else in the per-user .NET locations (`dotnet tool install -g` lands in ~/.dotnet/tools)."""
    found = shutil.which(tool)
    if found:
        return found
    for base in (Path.home() / ".dotnet/tools", Path.home() / ".dotnet"):
        for name in (tool, tool + ".exe"):
            if (base / name).is_file():
                return str(base / name)
    return None


# --------------------------------------------------------------------------
# Locate + stage the game build
# --------------------------------------------------------------------------


def steam_roots() -> list[Path]:
    """Steam installs to look in: the registry's (Windows), then the usual defaults."""
    home = Path.home()
    if platform.system() == "Darwin":
        return [home / "Library/Application Support/Steam"]
    if platform.system() == "Windows":
        roots = []
        try:
            import winreg

            for hive, key, value in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(Path(winreg.QueryValueEx(k, value)[0]))
                except OSError:
                    pass
        except ImportError:
            pass
        return roots + [Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam")]
    return [
        home / ".steam/steam",
        home / ".local/share/Steam",
        home / ".var/app/com.valvesoftware.Steam/.local/share/Steam",  # Flatpak
    ]


def steam_candidates() -> list[Path]:
    """The game's folder in every Steam library: each Steam install's own, plus
    the extra libraries (other drives) its steamapps/libraryfolders.vdf lists."""
    libraries: list[Path] = []
    for root in steam_roots():
        libraries.append(root)
        vdf = root / "steamapps/libraryfolders.vdf"
        if vdf.is_file():
            text = vdf.read_text(encoding="utf-8", errors="replace")
            libraries += [Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', text)]
    game = "SlayTheSpire2.app" if platform.system() == "Darwin" else ""
    seen: set[str] = set()
    out = []
    for lib in libraries:
        cand = lib / "steamapps/common/Slay the Spire 2" / game
        key = str(cand).lower() if platform.system() == "Windows" else str(cand)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def find_pck(root: Path) -> Path | None:
    for name in ("SlayTheSpire2.pck", "Slay the Spire 2.pck"):
        for base in (root, root / "Contents/Resources"):
            if (base / name).is_file():
                return base / name
    hits = sorted(root.rglob("*.pck")) if root.is_dir() else []
    return hits[0] if hits else None


def find_data_dir(root: Path) -> Path | None:
    if (root / "sts2.dll").is_file():
        return root
    for base in (root, root / "Contents/Resources"):
        for d in sorted(base.glob("data_sts2_*")):
            if (d / "sts2.dll").is_file():
                return d
    return None


def locate(arg: str | None) -> tuple[Path, Path, Path]:
    """-> (data_dir with sts2.dll, pck, release_info.json)."""
    tried: list[str] = []
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser())
    elif os.environ.get("STS2_GAME_DIR"):
        candidates.append(Path(os.environ["STS2_GAME_DIR"]).expanduser())
    else:
        candidates.extend(steam_candidates())
    for cand in candidates:
        tried.append(str(cand))
        if not cand.exists():
            # A Steam library without the game: its parents are steamapps/,
            # whose recursive pck search would walk every installed game.
            continue
        roots = [cand, cand.parent, cand.parent.parent]  # a data dir sits one or two levels below the pck
        for root in roots:
            pck = find_pck(root)
            data = find_data_dir(root)
            if pck and data:
                info = next((p for p in (root / "release_info.json", root / "Contents/Resources/release_info.json") if p.is_file()), None)
                if info is None:
                    raise SystemExit(f"found the game at {root} but no release_info.json next to the pck")
                return data, pck, info
    raise SystemExit(
        "could not find a Slay the Spire 2 install. Tried:\n  " + "\n  ".join(tried)
        + "\nPass the install folder (the one holding SlayTheSpire2.pck) as the first argument "
        "or set STS2_GAME_DIR (PowerShell: $env:STS2_GAME_DIR = \"D:\\...\\Slay the Spire 2\"; just import-game)."
    )


def read_version(release_info: Path) -> str:
    return str(json.loads(release_info.read_text(encoding="utf-8"))["version"])


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    """Put `src` at `dst`; returns how ("present", "symlink", "junction", "hardlink", "copy")."""
    if dst.is_symlink() or dst.exists():
        return "present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
            return "symlink"
        except OSError:
            pass
        # Windows without Developer Mode cannot symlink. A directory junction
        # and a hard link (same volume) need no privilege and cost no space.
        if src.is_dir() and os.name == "nt":
            try:
                import _winapi

                _winapi.CreateJunction(str(src), str(dst))
                return "junction"
            except OSError:
                pass
        elif src.is_file():
            try:
                os.link(src, dst)
                return "hardlink"
            except OSError:
                pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy"


def stage(data_dir: Path, pck: Path, release_info: Path, version: str, copy: bool) -> Path:
    """Populate sts2_binaries/<version>/ with the fixed layout every recipe reads."""
    build = BINARIES / version
    link_or_copy(release_info, build / "release_info.json", copy=True)
    how_pck = link_or_copy(pck, build / "SlayTheSpire2.pck", copy)
    how_data = link_or_copy(data_dir, build / "data", copy)
    log(f"staged {build.relative_to(ROOT)}/: SlayTheSpire2.pck ({how_pck}), data/ ({how_data}) -> {data_dir}")
    # The Windows build ships its FMOD runtime next to the pck; staged beside
    # the staged pck it is what the sound tier renders with (no SDK needed).
    for dll in FMOD_DLLS:
        if (pck.parent / dll).is_file():
            link_or_copy(pck.parent / dll, build / dll, copy)
    if how_pck == "copy":
        log(f"note: the pck was COPIED ({pck.stat().st_size / 1e9:.1f} GB) — no symlink or hard link was possible "
            "(on Windows: enable Developer Mode, or keep the repo on the game's drive)")
    return build


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def fmod_env(build: Path) -> dict[str, str]:
    """The FMOD runtime for the sound step, or {} when there is none.

    FMOD_SDK_ROOT (the FMOD Studio API, the Linux/WSL route) wins when set and
    is passed through untouched. Otherwise, on Windows, the game's own
    fmod.dll + fmodstudio.dll that `stage` put beside the pck.
    """
    if os.environ.get("FMOD_SDK_ROOT"):
        return {"FMOD_SDK_ROOT": os.environ["FMOD_SDK_ROOT"]}
    if os.name == "nt" and all((build / d).is_file() for d in FMOD_DLLS):
        return {"FMOD_LIB_DIR": str(build)}
    return {}


def build_steps(build: Path, version: str) -> list[Step]:
    """The import, in order. Each step calls its tool directly, so it runs the
    same on every platform; the sts2-wiki justfile carries the same commands
    for wiki work (`just extract-pck`, `just decompile`, ...)."""
    wiki_env = {"STS2_VERSION": version, "PYTHONUTF8": "1"}
    wiki_data = WIKI / "data" / version
    extracted = WIKI / "extracted" / version
    loc_eng = PYCLI / "localization_eng"
    images = WIKI / "site/public/images"
    audio = WIKI / "site/public/audio"
    pck = str(build / "SlayTheSpire2.pck")

    def real_file(path: Path) -> bool:
        """Present and not an un-smudged LFS pointer (the private repo vendors lib/ that way)."""
        if not path.is_file():
            return False
        with path.open("rb") as fh:
            return not fh.read(24).startswith(b"version https://git-lfs")

    def has_lib() -> bool:
        # The DLLs alone are not the output: setup_oracle.py also IL-patches
        # sts2.dll (leaving sts2.dll.original behind) and builds the harness. A
        # run that was interrupted after the copy must not count as done.
        harness = PYCLI / "src/Sts2Headless/bin/Debug/net9.0"
        return (
            all(real_file(PYCLI / "lib" / d) for d in GAME_DLLS)
            and (PYCLI / "lib/sts2.dll.original").is_file()
            and (harness / "Sts2Headless.dll").is_file()
        )

    def oracle_lib():
        # --install-root: where to look for DLLs the data dir lacks (the folder
        # holding the pck, i.e. the real install root through the link).
        install_root = (build / "SlayTheSpire2.pck").resolve().parent
        sh([sys.executable, "scripts/setup_oracle.py", str(build / "data"), "--install-root", str(install_root)], PYCLI)

    def extract_pck():
        for prefix in ("localization/eng", "images/atlases"):
            uv_python(["-m", "scripts.extract_pck", pck, str(extracted), "--prefix", prefix], WIKI, wiki_env)

    def localization():
        sh([sys.executable, "scripts/vendor_localization.py", pck, "--version", version], PYCLI)

    def decompile():
        env = dict(wiki_env)
        dotnet = which("dotnet")
        if dotnet and not os.environ.get("DOTNET_ROOT"):
            env["DOTNET_ROOT"] = str(Path(dotnet).resolve().parent)
        sh([which("ilspycmd") or "ilspycmd", "-p", "-o", f"decompiled/{version}", str(build / "data/sts2.dll")], WIKI, env)

    def wiki_extract():
        loc = f"extracted/{version}/localization/eng"
        for kind in ("powers", "cards", "monsters", "encounters", "potions", "relics", "ancients",
                     "enchantments", "characters", "epochs"):
            uv_python(["-m", f"scripts.extract_{kind}", f"decompiled/{version}", loc, f"data/{version}"], WIKI, wiki_env)
        uv_python(["-m", "scripts.extract_vfx_bindings", f"decompiled/{version}", f"data/{version}/vfx_bindings.json"],
                  WIKI, wiki_env)
        uv_python(["-m", "scripts.power_flash_sites", f"decompiled/{version}", f"data/{version}/power_flash_sites.json"],
                  WIKI, wiki_env)

    def atlas_art():
        # card_atlas, epoch_atlas and the other sprite atlases, then the VFX.
        uv_python(["scripts/extract_images.py", pck, f"extracted/{version}", "site/public/images"], WIKI, wiki_env)
        uv_python(["-m", "scripts.extract_vfx", pck, "site/public/images"], WIKI, wiki_env)

    def scene_art():
        # Rooms, scenes, map, events, HUD chrome, boss icons, orbs, ancients.
        uv_python(["-m", "scripts.extract_rooms", pck, "site/public/images"], WIKI, wiki_env)
        uv_python(["-m", "scripts.extract_card_template", pck, "site/public/images"], WIKI, wiki_env)

    def rigs():
        uv_python(["-m", "scripts.extract_rig", pck, "--all", "site/public/images"], WIKI, wiki_env)

    def art_tables():
        # The two JSON tables the art tree carries next to the images. Both are
        # committed elsewhere; the art tree gets a copy. vfx_bindings.json is
        # what the browser fetches from /assets/vfx/; creature_scales.json
        # (hand-tuned scales + the authored %Bounds boxes) is the table the
        # Python workbench and the bundle build read from the art root.
        (images / "vfx").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wiki_data / "vfx_bindings.json", images / "vfx/vfx_bindings.json")
        if not (images / "creature_scales.json").is_file():
            shutil.copyfile(RS / "crates/sts2-workbench/data/game/creature_scales.json", images / "creature_scales.json")
        log("copied vfx_bindings.json and creature_scales.json into the art tree")

    def sfx_map():
        uv_python(["-m", "scripts.build_sfx_map", f"decompiled/{version}", "../sts2-pycli/web/public/sfx_map.json"],
                  WIKI, wiki_env)

    def card_text_and_db():
        py_env = {"PYTHONPATH": str(PYCLI / "python")}
        sh([sys.executable, "-m", "synergy.gen_card_text"], PYCLI, env=py_env)
        sh([sys.executable, "-m", "workbench.build_card_db"], PYCLI, env=py_env)

    def workbench_data():
        sh([sys.executable, "scripts/build_asset_manifest.py"], PYCLI)
        sh([sys.executable, "scripts/bundle_workbench_data.py"], PYCLI)

    def portraits():
        uv_python(["-m", "scripts.render_portraits", "site/public/images"], WIKI, wiki_env)

    def sfx():
        uv_python(["-m", "scripts.extract_sfx", pck, f"extracted/{version}", "--render"], WIKI,
                  {**wiki_env, **fmod_env(build)})
        uv_python(["-m", "scripts.encode_sfx", f"extracted/{version}/audio", "site/public/audio"], WIKI, wiki_env)
        # The map's `audioRev` is a hash of the audio tree it is served with
        # (the URLs' cache-buster). The committed map carries the maintainers'
        # hash; re-derive it for the audio just encoded. Only that line moves.
        sfx_map()

    return [
        Step("game DLLs -> sts2-pycli/lib/ (IL-patched) + C# oracle build", "oracle", has_lib, oracle_lib, ("dotnet",)),
        Step("pck localization + atlases -> sts2-wiki/extracted/", "oracle",
             lambda: (extracted / "localization").is_dir(), extract_pck, ("uv",)),
        Step("localization -> sts2-pycli/localization_eng/", "oracle",
             lambda: (loc_eng / "cards.json").is_file() and (loc_eng / "BUILD.txt").is_file(), localization, ("uv",)),
        Step("decompile sts2.dll -> sts2-wiki/decompiled/", "workbench",
             lambda: (WIKI / "decompiled" / version).is_dir(), decompile, ("ilspycmd",)),
        Step("wiki data JSON -> sts2-wiki/data/", "workbench",
             lambda: (wiki_data / "cards.json").is_file() and (wiki_data / "vfx_bindings.json").is_file()
             and (wiki_data / "power_flash_sites.json").is_file(), wiki_extract, ("uv",)),
        Step("atlas art + VFX -> sts2-wiki/site/public/images/", "workbench",
             lambda: (images / "vfx" / "manifest.json").is_file() and any(images.glob("card_atlas/*"))
             and any(images.glob("epoch_atlas/*")), atlas_art, ("uv",)),
        Step("rooms, scenes, map, HUD + card frames -> sts2-wiki/site/public/images/", "workbench",
             lambda: (images / "rooms").is_dir() and (images / "ui/top_panel").is_dir()
             and (images / "boss_icons").is_dir() and (images / "card_template").is_dir()
             and (images / "ancients/neow.png").is_file() and (images / "monsters/hourglass_placeholder.png").is_file(),
             scene_art, ("uv",)),
        Step("creature + room rigs -> sts2-wiki/site/public/images/*_rigs/", "workbench",
             lambda: all(any((images / d).glob("*/*.skel")) for d in ("monster_rigs", "character_rigs", "room_rigs")),
             rigs, ("uv",)),
        Step("creature portraits + character sprites (rendered from the rigs) -> sts2-wiki/site/public/images/",
             "workbench",
             lambda: (images / "character_sprites/ironclad.png").is_file() and (images / "monsters/amalgam.png").is_file(),
             portraits, ("uv", "node", "npm")),
        Step("vfx_bindings.json + creature_scales.json -> the art tree", "workbench",
             lambda: (images / "vfx/vfx_bindings.json").is_file() and (images / "creature_scales.json").is_file(),
             art_tables),
        Step("sfx map -> sts2-pycli/web/public/sfx_map.json", "workbench",
             lambda: (PYCLI / "web/public/sfx_map.json").is_file(), sfx_map, ("uv",)),
        Step("card text + card db -> sts2-pycli/", "workbench",
             lambda: (PYCLI / "web/public/card_text.json").is_file() and (PYCLI / "python/workbench/data/card_db.json").is_file(),
             card_text_and_db),
        Step("workbench bundle -> sts2-rs/crates/sts2-workbench/data/game/", "workbench",
             lambda: (RS / "crates/sts2-workbench/data/game/wiki_cards.json").is_file(), workbench_data),
        Step("sound effects -> sts2-wiki/site/public/audio/ (FMOD: the game's own DLLs on Windows, else FMOD_SDK_ROOT)",
             "sound",
             lambda: any(audio.rglob("*.webm")), sfx, ("uv", "ffmpeg")),
    ]


def ffmpeg_hint() -> str:
    """How to install ffmpeg on this machine."""
    if os.name == "nt":
        return "winget install Gyan.FFmpeg (or .\\install-windows-deps.ps1), then open a new terminal"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "your package manager, e.g. sudo apt install ffmpeg / sudo dnf install ffmpeg"


def preflight(steps: list[Step]) -> None:
    missing: dict[str, str] = {}
    hints = {
        "dotnet": ".NET 9 SDK - https://dotnet.microsoft.com/download (Windows: winget install Microsoft.DotNet.SDK.9)",
        "uv": "https://docs.astral.sh/uv/ (Windows: winget install astral-sh.uv)",
        "ilspycmd": f"{ILSPYCMD_INSTALL} (ILSpy's CLI; the decompile step)",
        "ffmpeg": ffmpeg_hint(),
    }
    for step in steps:
        for tool in step.needs:
            if tool not in missing and not which(tool):
                missing[tool] = hints.get(tool, "")
    if missing:
        lines = [f"  {tool}: {hint}" for tool, hint in missing.items()]
        footer = "\n(Windows: .\\install-windows-deps.ps1 installs all of them)" if os.name == "nt" else ""
        raise SystemExit("missing tools for the requested tier:\n" + "\n".join(lines) + footer)


def status(build: Path | None, version: str) -> int:
    print(f"pinned build: {PINNED_VERSION}")
    if build and build.is_dir():
        print(f"staged build: {build.relative_to(ROOT)}  (version {version})")
    else:
        print("staged build: none (sts2_binaries/<version>/ absent) — run `just import-game [GAME_DIR]`")
    steps = build_steps(build or BINARIES / PINNED_VERSION, version)
    for tier in TIERS[1:]:
        print(f"\n[{tier}]")
        for step in steps:
            if step.tier == tier:
                print(f"  {'ok     ' if step.done() else 'MISSING'}  {step.name}")
    print()
    print("\n".join(art_coverage()))
    return 0


def art_coverage() -> list[str]:
    """scripts/check_art.py's summary: the art + sound against the committed manifest."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_art

    return check_art.summary()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("game_dir", nargs="?", help="Steam install folder / .app / data_sts2_* folder")
    parser.add_argument("--tier", choices=TIERS + ["all"],
                        help="default: everything, skipping sound if ffmpeg is missing")
    parser.add_argument("--status", action="store_true", help="report which artifacts exist and exit")
    parser.add_argument("--copy", action="store_true", help="copy the pck and DLLs into sts2_binaries/ instead of symlinking")
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help=f"proceed even if the install is not {PINNED_VERSION} (the engine is pinned; expect drift)")
    args = parser.parse_args(argv)
    tier = "sound" if args.tier in (None, "all") else args.tier
    for stream in (sys.stdout, sys.stderr):
        # Arrows and dashes in the log lines; a cp1252 console would print mojibake.
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    if args.status:
        staged = sorted(BINARIES.glob("v*")) if BINARIES.is_dir() else []
        build = staged[-1] if staged else None
        version = build.name if build else PINNED_VERSION
        return status(build, version)

    if tier == "engine":
        print("tier engine: nothing to import — `cd sts2-rs && cargo nextest run --workspace` works as-is.")
        return 0

    data_dir, pck, release_info = locate(args.game_dir)
    version = read_version(release_info)
    log(f"game install: data={data_dir} pck={pck} version={version}")
    if version != PINNED_VERSION and not args.allow_version_mismatch:
        raise SystemExit(
            f"the install is {version} but the engine is pinned to {PINNED_VERSION}; the parity oracle and "
            "the extracted data would describe a different game. Pass --allow-version-mismatch to proceed anyway."
        )
    build = stage(data_dir, pck, release_info, version, args.copy)

    wanted = TIERS[: TIERS.index(tier) + 1]
    steps = [s for s in build_steps(build, version) if s.tier in wanted]
    sound_skipped = ""
    if tier == "sound" and not fmod_env(build):
        sound_skipped = "no FMOD runtime"
        log("no FMOD runtime — skipping the sound tier. On Windows the game's own fmod.dll is used (it was not "
            "found next to the pck); elsewhere set FMOD_SDK_ROOT to the unpacked FMOD Studio API (a free, "
            "registration-gated download from fmod.com). The workbench is silent without it.")
    elif (tier == "sound" and args.tier is None and not which("ffmpeg")
          and any(s.tier == "sound" and not s.done() for s in steps)):
        # The default run takes everything it can; only an explicit
        # --tier sound / all makes ffmpeg a hard requirement (preflight).
        sound_skipped = "ffmpeg not installed"
        log(f"ffmpeg not found — skipping the sound tier (art and data are unaffected). Install it ({ffmpeg_hint()}) "
            "and re-run `just import-game`; the workbench is silent until then.")
    if sound_skipped:
        steps = [s for s in steps if s.tier != "sound"]
    preflight([s for s in steps if not s.done()])

    for step in steps:
        if step.done():
            log(f"skip (present): {step.name}")
            continue
        log(f"run: {step.name}")
        step.run()
        if not step.done():
            raise SystemExit(f"step finished but its output is missing: {step.name} (see {LOG})")
    log("done")
    if tier in ("workbench", "sound"):
        for line in art_coverage():
            log(line)
    if sound_skipped:
        log(f"sound skipped: {sound_skipped} (the audio counts above are missing for that reason)")
    if tier in ("workbench", "sound"):
        print("\nnext: the browser workbench -  just workbench-local   (http://127.0.0.1:4173)\n"
              "      the Python workbench  -  just workbench-server  (http://127.0.0.1:8077)\n"
              "      the full test suites  -  see docs/bring-your-own-game.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
