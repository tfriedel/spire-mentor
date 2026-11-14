#!/usr/bin/env python3
"""Bring your own game: regenerate every game-derived input from a local install.

This repository ships no game content. Everything the C# oracle, the parity
harness and the workbench need from Slay the Spire 2 — the game assembly,
its localization, the decompiled source the extractors read, card/monster
data, art, sound — is produced HERE, from the copy of the game you own, into
paths that are git-ignored in the public repository.

Usage:
    just import-game [GAME_DIR] [--tier oracle|workbench|sound|all]
    just import-game --download [VERSION|all] (--steam-user NAME | --steam-qr) [--tier ...]
    just import-game --status
    uv run --no-project scripts/import_game.py [GAME_DIR] [--tier ...]

Runs the same on Windows (PowerShell or Git Bash), Linux and macOS: every step
calls its tool directly (`uv run`, `dotnet`, `ilspycmd`) — no bash, no `just`.

GAME_DIR is the Steam install (the folder holding SlayTheSpire2.pck), the
macOS .app, or the data_sts2_* folder holding sts2.dll; omitted, the usual
Steam locations and $STS2_GAME_DIR are tried.

--download fetches the supported builds instead (all of them unless one is
named) from Steam with DepotDownloader, each pinned to its depot manifest,
into sts2_binaries/<version>/, and imports each one, the default build first.
Steam installs one branch at a time; this is how one machine gets both. It
logs in as you: DepotDownloader asks for your password and Steam Guard code
itself (or shows a QR code with --steam-qr); this script never sees them.

Tiers (each includes the ones before it):
    engine     nothing to import — the Rust workspace builds and tests as-is
    oracle     sts2-pycli/lib/<ver>/ (IL-patched game DLLs) + localization_eng/
               -> the C# oracle, the parity harness, the full pytest suite
    workbench  + decompiled source, wiki data JSON, art (atlases, VFX, rooms
               and scenes, card frames, creature rigs), the sfx map, card
               text / card db, the workbench's embedded data bundle
               -> the Python and browser workbenches render real game
               text and art; `just workbench-local` next
    sound      + the sound effects: rendered through FMOD — on Windows the
               game's own fmod.dll, elsewhere the FMOD Studio API
               (FMOD_SDK_ROOT)
    all        = sound (the default)

A missing tool (ffmpeg included) or FMOD runtime stops the import before any
step runs and says how to install it. Use --tier workbench to import
everything but the sound without them.

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
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARIES = ROOT / "sts2_binaries"
WIKI = ROOT / "sts2-wiki"
PYCLI = ROOT / "sts2-pycli"
RS = ROOT / "sts2-rs"
LOG = ROOT / "logs" / "import-game.log"

# The builds the engine simulates (GH #789). The first is the default — the one
# the shared, unversioned outputs (localization_eng/, the art, the card db, the
# workbench bundle) describe; sts2-rs/crates/sts2-workbench/src/data.rs
# (ENGINE_WIKI_VERSION) and sts2-pycli/python/workbench/localization.py
# (_ENGINE_WIKI_VERSION) pin the same value; move all three together. Every
# other supported build gets only its version-keyed outputs: its oracle
# (sts2-pycli/lib/<ver>/ + bin/<ver>/), sts2-wiki/extracted|decompiled|data/<ver>/,
# its vendored loc (sts2-pycli/localization_eng_<ver>/) and its workbench bundle
# (sts2-rs/crates/sts2-workbench/data/game/<ver>/) — GH #801 — plus its display
# tables: sts2-pycli/web/public/<ver>/{sfx_map,card_text}.json and
# sts2-pycli/python/workbench/data/<ver>/card_db.json (GH #877).
SUPPORTED_VERSIONS = ("v0.107.1", "v0.111.0", "v0.113.0")
DEFAULT_VERSION = SUPPORTED_VERSIONS[0]

TIERS = ["engine", "oracle", "workbench", "sound"]

# Where --download gets each supported build: the Steam app, its Windows depot
# (the pck and the .NET DLLs every step reads are the same on every OS, and the
# Windows depot also carries the FMOD DLLs the sound tier renders with), and
# the depot manifest that IS that build. A branch name alone would not do:
# Steam's `public` and `public-beta` move to newer builds. Steam still wants
# the branch a manifest was published on (it hands out the manifest's
# download code per branch), so each build pins both. A new supported build
# adds its pair here (DepotDownloader prints the manifest;
# sts2_binaries/<ver>/.DepotDownloader/ keeps it as <depot>_<manifest>.manifest).
STEAM_APP = "2868840"
STEAM_DEPOT = "2868841"
DEPOT_MANIFESTS = {
    "v0.107.1": "4733892780027453168",
    "v0.111.0": "5661886308395090154",
    "v0.113.0": "5756684179086186957",
}
DEPOT_BRANCHES = {
    "v0.107.1": "public",
    "v0.111.0": "public-beta",
    "v0.113.0": "public-beta",
}
# What the import reads from a build; the rest (the ~0.9 GB executable) is skipped.
DOWNLOAD_FILES = [
    "release_info.json",
    "SlayTheSpire2.pck",
    "fmod.dll",
    "fmodstudio.dll",
    r"regex:^data_sts2_windows_x86_64[/\\]",
]
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_RETRY_S = 20
DEPOTDOWNLOADER_URL = "https://github.com/SteamRE/DepotDownloader/releases"

# Files the C# harness links against (sts2-pycli/scripts/setup_oracle.py copies + patches).
GAME_DLLS = [
    "sts2.dll", "SmartFormat.dll", "SmartFormat.ZString.dll", "Sentry.dll", "Steamworks.NET.dll",
    "MonoMod.Backports.dll", "MonoMod.ILHelpers.dll", "0Harmony.dll", "System.IO.Hashing.dll",
]

# The sound renderer owns the FMOD runtime rules (which DLLs, which
# environment variable wins); the import asks it rather than restating them.
sys.path.insert(0, str(WIKI / "scripts"))
import extract_sfx  # noqa: E402

# ilspycmd 10+ needs a newer runtime than the .NET 9 SDK the rest of the repo
# pins; 9.1 is the last release that installs and runs on .NET 9.
ILSPYCMD_INSTALL = "dotnet tool install -g ilspycmd --version 9.1.0.7988"


class Step:
    """One idempotent step: `done()` says whether its output exists."""

    def __init__(self, name: str, tier: str, done, run, needs: tuple[str, ...] = (), versioned: bool = False):
        self.name, self.tier, self.done, self.run, self.needs = name, tier, done, run, needs
        # True: the output is keyed by game version, so every supported build
        # gets its own. False: one shared tree describing DEFAULT_VERSION.
        self.versioned = versioned


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
    for dll in extract_sfx.GAME_FMOD_DLLS:
        if (pck.parent / dll).is_file():
            link_or_copy(pck.parent / dll, build / dll, copy)
    if how_pck == "copy":
        log(f"note: the pck was COPIED ({pck.stat().st_size / 1e9:.1f} GB) — no symlink or hard link was possible "
            "(on Windows: enable Developer Mode, or keep the repo on the game's drive)")
    return build


# --------------------------------------------------------------------------
# Download a build (--download)
# --------------------------------------------------------------------------


def find_depotdownloader(explicit: str | None) -> str:
    for cand in (explicit, os.environ.get("DEPOTDOWNLOADER")):
        if cand:
            if Path(cand).expanduser().is_file():
                return str(Path(cand).expanduser())
            raise SystemExit(f"DepotDownloader not found at {cand}")
    found = which("DepotDownloader") or which("depotdownloader")
    if found:
        return found
    raise SystemExit(
        f"--download needs DepotDownloader: unpack a release from {DEPOTDOWNLOADER_URL} and put it on PATH, "
        "or pass --depotdownloader PATH (or set DEPOTDOWNLOADER)."
    )


def downloaded(dest: Path, version: str) -> bool:
    """Whether `dest` already holds `version`'s pck, data folder and release info."""
    info = dest / "release_info.json"
    return (dest / "SlayTheSpire2.pck").is_file() and info.is_file() and read_version(info) == version \
        and find_data_dir(dest) is not None


def depot_command(tool: str, version: str, dest: Path, filelist: Path, steam_user: str | None, qr: bool) -> list[str]:
    cmd = [tool, "-app", STEAM_APP, "-depot", STEAM_DEPOT, "-manifest", DEPOT_MANIFESTS[version],
           "-branch", DEPOT_BRANCHES[version], "-os", "windows", "-dir", str(dest), "-filelist", str(filelist)]
    # -remember-password keeps DepotDownloader's login token (its own store,
    # not this repo), so the second build does not ask again.
    return cmd + (["-qr"] if qr else ["-username", steam_user or "", "-remember-password"])


def download(version: str, tool: str, steam_user: str | None, qr: bool) -> Path:
    """Fetch `version` into sts2_binaries/<version>/ (skipped when it is there)."""
    dest = BINARIES / version
    if downloaded(dest, version):
        log(f"skip download (present): {dest.relative_to(ROOT)}/")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        filelist = Path(tmp) / "files.txt"
        filelist.write_text("\n".join(DOWNLOAD_FILES) + "\n", encoding="utf-8")
        cmd = depot_command(tool, version, dest, filelist, steam_user, qr)
        log(f"download {version} (depot {STEAM_DEPOT} manifest {DEPOT_MANIFESTS[version]}) -> {dest.relative_to(ROOT)}/")
        log("$ " + " ".join("<steam-user>" if steam_user and a == steam_user else a for a in cmd))
        # Inherits the terminal: DepotDownloader prompts for the password and
        # the Steam Guard code (or draws the QR code) itself. Steam sometimes
        # returns no manifest request code ("Encountered 401 for depot
        # manifest") for a manifest it serves a minute later, so a failed
        # attempt is retried; the login token makes the retry silent.
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            rc = subprocess.run(cmd).returncode
            if rc == 0 and downloaded(dest, version):
                break
            if attempt < DOWNLOAD_ATTEMPTS:
                log(f"download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed; retrying in {DOWNLOAD_RETRY_S}s")
                time.sleep(DOWNLOAD_RETRY_S)
    if rc != 0:
        raise SystemExit(f"DepotDownloader failed ({rc}) downloading {version}; its output is above")
    if not downloaded(dest, version):
        info = dest / "release_info.json"
        if not info.is_file():
            # DepotDownloader exits 0 when Steam refuses the manifest (wrong
            # branch, an account that does not own the game, a manifest Steam
            # no longer serves); it says why above.
            raise SystemExit(f"DepotDownloader downloaded nothing for {version} (manifest {DEPOT_MANIFESTS[version]}, "
                             f"branch {DEPOT_BRANCHES[version]}); its output above says why")
        raise SystemExit(f"the download into {dest} is {read_version(info)}, not {version}: the pinned manifest is wrong")
    return dest


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def copy_new_art(extracted: Path, shared: Path, overlay: Path) -> int:
    """Copy into `overlay` every file under `extracted` the `shared` tree lacks.

    -> the number copied. A non-default build's art overlay: its new content
    and the art the default build ships only as a sketch (`beta/`); a file the
    shared tree has stays shared, even where the build redrew it.
    """
    copied = 0
    for src in sorted(extracted.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(extracted)
        dst = overlay / rel
        if (shared / rel).exists() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def write_new_creature_scales(fresh: Path, shared: Path, out: Path) -> int:
    """Write to `out` the `fresh` creature-scale rows whose slug `shared` lacks.

    -> the number written. `fresh` is `extract_creature_bounds`' table for a
    later build; `shared` is the art tree's creature_scales.json (the default
    build's, with its hand-overridden scales). A creature both carry keeps the
    shared row, like the shared art itself (`copy_new_art`).
    """
    rows = json.loads(fresh.read_text(encoding="utf-8"))
    have = json.loads(shared.read_text(encoding="utf-8")) if shared.is_file() else {}
    new = {k: rows[k] for k in sorted(rows) if k not in have}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(new)


def fmod_env(build: Path) -> dict[str, str]:
    """Environment for the sound step: the game's DLLs that `stage` put beside
    the staged pck, unless the user already pointed FMOD_LIB_DIR elsewhere.
    (FMOD_SDK_ROOT and FMOD_LIB_DIR reach the step through the inherited
    environment; `extract_sfx.find_fmod_runtime` decides which one wins.)"""
    return {} if os.environ.get("FMOD_LIB_DIR") else {"FMOD_LIB_DIR": str(build)}


PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR, or None if it is not a PNG."""
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != PNG_MAGIC:
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def rig_page_matches_atlas(rig_dir: Path) -> bool:
    """Whether a rig's first texture page has the size its .atlas states.

    False for a missing rig. An atlas without page sizes (older format)
    cannot be checked and counts as matching.
    """
    atlas = rig_dir / f"{rig_dir.name}.atlas"
    if not atlas.is_file():
        return False
    lines = [ln.strip() for ln in atlas.read_text(encoding="utf-8", errors="replace").splitlines()]
    page = next((i for i, ln in enumerate(lines) if ln.endswith(".png")), None)
    if page is None:
        return False
    for ln in lines[page + 1:]:
        if ":" not in ln:
            return True  # a region name: the page states no size
        if ln.startswith("size:"):
            want = tuple(int(v) for v in ln[5:].replace(" ", "").split(","))
            return png_size(rig_dir / lines[page]) == want
    return True


def build_steps(build: Path, version: str) -> list[Step]:
    """The import, in order. Each step calls its tool directly, so it runs the
    same on every platform; the sts2-wiki justfile carries the same commands
    for wiki work (`just extract-pck`, `just decompile`, ...)."""
    wiki_env = {"STS2_VERSION": version, "PYTHONUTF8": "1"}
    wiki_data = WIKI / "data" / version
    extracted = WIKI / "extracted" / version
    is_default = version == DEFAULT_VERSION
    loc_eng_default = PYCLI / "localization_eng"
    # A non-default build's loc is a sibling copy (workbench.localization.loc_dir_for).
    loc_eng = loc_eng_default if is_default else PYCLI / f"localization_eng_{version}"
    bundle_dir = RS / "crates/sts2-workbench/data/game"
    if not is_default:
        bundle_dir = bundle_dir / version
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
        lib = PYCLI / "lib" / version
        harness = PYCLI / "src/Sts2Headless/bin" / version / "Debug/net9.0"
        return (
            all(real_file(lib / d) for d in GAME_DLLS)
            and (lib / "sts2.dll.original").is_file()
            and (harness / "Sts2Headless.dll").is_file()
        )

    def oracle_lib():
        # --install-root: where to look for DLLs the data dir lacks (the folder
        # holding the pck, i.e. the real install root through the link).
        install_root = (build / "SlayTheSpire2.pck").resolve().parent
        sh([sys.executable, "scripts/setup_oracle.py", str(build / "data"), "--version", version,
            "--install-root", str(install_root)], PYCLI)

    def extract_pck():
        for prefix in ("localization/eng", "images/atlases"):
            uv_python(["-m", "scripts.extract_pck", pck, str(extracted), "--prefix", prefix], WIKI, wiki_env)

    def localization():
        if not is_default and not loc_eng.is_dir() and loc_eng_default.is_dir():
            # Seed from the default build's copy so the vendor merge retains the
            # keys only the older build had (content the engine still implements).
            shutil.copytree(loc_eng_default, loc_eng)
        sh([sys.executable, "scripts/vendor_localization.py", pck, "--version", version,
            "--out", str(loc_eng)], PYCLI)

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

    overlay = images / version

    default_pck = BINARIES / DEFAULT_VERSION / "SlayTheSpire2.pck"

    def art_overlay():
        # images/<version>/ is what a <version> run's art resolver tries before
        # the shared tree (sts2-workbench assets.rs, workbench/assets.py), and
        # the bundle step lists it in that build's art manifest. Every art
        # extractor runs into a scratch tree and the files the shared tree
        # lacks are copied over: atlases + sketch card art, card-frame tints
        # and energy icons, rooms / events / HUD, and the creature rigs with the
        # portraits rendered from them. With no shared tree (only this build
        # was imported) the whole set lands here.
        with tempfile.TemporaryDirectory() as tmp:
            uv_python(["scripts/extract_images.py", pck, f"extracted/{version}", tmp], WIKI, wiki_env)
            uv_python(["-m", "scripts.extract_card_template", pck, tmp], WIKI, wiki_env)
            uv_python(["-m", "scripts.extract_rooms", pck, tmp], WIKI, wiki_env)
            # Only the rigs this build ADDED: the shared tree's rigs keep their
            # own slugs and atlas pages, and a file-level diff against them
            # would pick up alias folders and stray pages (extract_rig.new_rigs).
            rig_args = ["--new-since", str(default_pck)] if default_pck.is_file() else []
            uv_python(["-m", "scripts.extract_rig", pck, "--all", tmp, *rig_args], WIKI, wiki_env)
            for rigs in ("monster_rigs", "character_rigs"):
                (Path(tmp) / rigs).mkdir(exist_ok=True)
            uv_python(["-m", "scripts.render_portraits", tmp], WIKI, wiki_env)
            n = copy_new_art(Path(tmp), images, overlay)
            # The new creatures' authored scale + %Bounds box; the bundle step
            # merges this over the shared creature_scales.json for this build.
            bounds = Path(tmp) / "creature_bounds.json"
            uv_python(["-m", "scripts.extract_creature_bounds", pck, str(bounds)], WIKI, wiki_env)
            n_scales = write_new_creature_scales(bounds, images / "creature_scales.json",
                                                 overlay / "creature_scales.json")
        log(f"art overlay: {n} files the shared art tree lacks, {n_scales} new creature scales "
            f"-> {overlay.relative_to(ROOT)}/")

    def scene_art():
        # Rooms, scenes, map, events, HUD chrome, boss icons, orbs, ancients.
        uv_python(["-m", "scripts.extract_rooms", pck, "site/public/images"], WIKI, wiki_env)
        uv_python(["-m", "scripts.extract_card_template", pck, "site/public/images"], WIKI, wiki_env)

    def rigs():
        # `--all` skips rigs that exist, so a rig an older extractor got wrong
        # is removed first, with the portrait rendered from it.
        for rig_dir in sorted(images.glob("*_rigs/*")):
            if rig_dir.is_dir() and not rig_page_matches_atlas(rig_dir):
                log(f"re-extracting {rig_dir.relative_to(images)}: its page is not the size its atlas states")
                shutil.rmtree(rig_dir)
                for portrait in (images / "monsters" / f"{rig_dir.name}.png",
                                 images / "character_sprites" / f"{rig_dir.name}.png"):
                    portrait.unlink(missing_ok=True)
        uv_python(["-m", "scripts.extract_rig", pck, "--all", "site/public/images"], WIKI, wiki_env)

    def art_tables():
        # The two JSON tables the art tree carries next to the images. Both are
        # committed elsewhere; the art tree gets a copy. vfx_bindings.json is
        # what the browser fetches from /assets/vfx/; creature_scales.json
        # (hand-tuned scales + the authored %Bounds boxes) is the table the
        # Python workbench and the bundle build read from the art root.
        # The art tree is shared by every build and describes the DEFAULT one,
        # so its table comes from the default build's data whichever build is
        # being imported (importing v0.113.0 into a fresh worktree once
        # rewrote it with v0.113.0's bindings).
        (images / "vfx").mkdir(parents=True, exist_ok=True)
        default_bindings = WIKI / "data" / DEFAULT_VERSION / "vfx_bindings.json"
        shutil.copyfile(default_bindings if default_bindings.is_file() else wiki_data / "vfx_bindings.json",
                        images / "vfx/vfx_bindings.json")
        if not (images / "creature_scales.json").is_file():
            shutil.copyfile(RS / "crates/sts2-workbench/data/game/creature_scales.json", images / "creature_scales.json")
        log("copied vfx_bindings.json and creature_scales.json into the art tree")

    # Per-build display tables (GH #877): the default build's at today's paths,
    # another build's under a <version>/ subdir (web/src/gameVersion.ts
    # versionedPublicPath, workbench/deck_analysis.py data_dir_for).
    public = PYCLI / "web/public"
    public_ver = public if is_default else public / version
    card_db_json = PYCLI / "python/workbench/data" / ("" if is_default else version) / "card_db.json"
    sfx_map_json = f"../sts2-pycli/web/public/{'' if is_default else version + '/'}sfx_map.json"

    def sfx_map():
        public_ver.mkdir(parents=True, exist_ok=True)
        uv_python(["-m", "scripts.build_sfx_map", f"decompiled/{version}", sfx_map_json], WIKI, wiki_env)

    def card_text_and_db():
        py_env = {"PYTHONPATH": str(PYCLI / "python")}
        sh([sys.executable, "-m", "synergy.gen_card_text", "--version", version], PYCLI, env=py_env)
        sh([sys.executable, "-m", "workbench.build_card_db", "--version", version], PYCLI, env=py_env)

    def workbench_data():
        if is_default:
            sh([sys.executable, "scripts/build_asset_manifest.py"], PYCLI)
        sh([sys.executable, "scripts/bundle_workbench_data.py", "--version", version], PYCLI)

    def portraits():
        uv_python(["-m", "scripts.render_portraits", "site/public/images"], WIKI, wiki_env)

    # A later build's sound: its render is encoded beside the shared tree and
    # only the events the shared tree lacks are copied in (the Fool's voice,
    # the new monsters). The shared tree keeps the default build's renders, the
    # way the art keeps the default build's images. This file marks it done.
    sfx_merged = extracted / "audio" / ".merged_into_shared"

    def sfx():
        uv_python(["-m", "scripts.extract_sfx", pck, f"extracted/{version}", "--render"], WIKI,
                  {**wiki_env, **fmod_env(build)})
        if is_default:
            uv_python(["-m", "scripts.encode_sfx", f"extracted/{version}/audio", "site/public/audio"], WIKI, wiki_env)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                # The whole render, not just the new events: encode_sfx sets the
                # gain from the loudest cue in the tree it encodes, so a subset
                # would come out at a different level than the shared sounds.
                uv_python(["-m", "scripts.encode_sfx", f"extracted/{version}/audio", tmp], WIKI, wiki_env)
                n = copy_new_art(Path(tmp), audio, audio)
            log(f"sound: {n} events the shared audio tree lacks -> {audio.relative_to(ROOT)}/")
            # This build's map now resolves the events just added.
            uv_python(["-m", "scripts.build_sfx_map", f"decompiled/{version}", sfx_map_json], WIKI, wiki_env)
            sfx_merged.write_text(f"{dt.datetime.now().isoformat()}\n", encoding="utf-8")
        # The map's `audioRev` is a hash of the audio tree it is served with
        # (the URLs' cache-buster). The committed map carries the maintainers'
        # hash; re-derive it for the audio just encoded. Only that line moves:
        # a full rebuild would also drop every event this render missed.
        # The audio tree is shared, so every build's map gets the same rev.
        for sfx_json in [public / "sfx_map.json", *sorted(public.glob("v*/sfx_map.json"))]:
            if sfx_json.is_file():
                uv_python(["-m", "scripts.build_sfx_map", f"decompiled/{version}", str(sfx_json), "--rev-only"],
                          WIKI, wiki_env)

    return [
        Step(f"game DLLs -> sts2-pycli/lib/{version}/ (IL-patched) + C# oracle build", "oracle", has_lib, oracle_lib,
             ("dotnet",), versioned=True),
        Step("pck localization + atlases -> sts2-wiki/extracted/", "oracle",
             lambda: (extracted / "localization").is_dir(), extract_pck, ("uv",), versioned=True),
        Step(f"localization -> sts2-pycli/{loc_eng.name}/", "oracle",
             lambda: (loc_eng / "cards.json").is_file() and (loc_eng / "BUILD.txt").is_file(), localization, ("uv",),
             versioned=True),
        Step("decompile sts2.dll -> sts2-wiki/decompiled/", "workbench",
             lambda: (WIKI / "decompiled" / version).is_dir(), decompile, ("ilspycmd",), versioned=True),
        Step("wiki data JSON -> sts2-wiki/data/", "workbench",
             lambda: (wiki_data / "cards.json").is_file() and (wiki_data / "vfx_bindings.json").is_file()
             and (wiki_data / "power_flash_sites.json").is_file(), wiki_extract, ("uv",), versioned=True),
        Step("atlas art + VFX -> sts2-wiki/site/public/images/", "workbench",
             lambda: (images / "vfx" / "manifest.json").is_file() and any(images.glob("card_atlas/*"))
             and any(images.glob("epoch_atlas/*"))
             # extract_images' ownerless sprites: an older import lacks them.
             and (images / "relic_atlas/yummy_cookie.png").is_file(), atlas_art, ("uv",)),
        *([] if is_default else [
            # creature_scales.json is written last, so it marks a finished
            # overlay (an older import wrote only the atlas sprites).
            Step(f"{version} art overlay (art the shared tree lacks) -> sts2-wiki/site/public/images/{version}/",
                 "workbench", lambda: (overlay / "creature_scales.json").is_file(), art_overlay,
                 ("uv", "node", "npm"), versioned=True),
        ]),
        Step("rooms, scenes, map, HUD + card frames -> sts2-wiki/site/public/images/", "workbench",
             lambda: (images / "rooms").is_dir() and (images / "ui/top_panel").is_dir()
             and (images / "boss_icons").is_dir() and (images / "card_template").is_dir()
             and (images / "ancients/neow.png").is_file() and (images / "monsters/hourglass_placeholder.png").is_file(),
             scene_art, ("uv",)),
        Step("creature + room rigs -> sts2-wiki/site/public/images/*_rigs/", "workbench",
             lambda: all(any((images / d).glob("*/*.skel")) for d in ("monster_rigs", "character_rigs", "room_rigs"))
             # An older extract_rig decoded this page from the same-named event
             # illustration; `rigs` re-extracts it when it did.
             and rig_page_matches_atlas(images / "monster_rigs/battleworn_dummy"),
             rigs, ("uv",)),
        Step("creature portraits + character sprites (rendered from the rigs) -> sts2-wiki/site/public/images/",
             "workbench",
             lambda: all((images / rel).is_file() for rel in (
                 "character_sprites/ironclad.png", "monsters/amalgam.png",
                 # the two `rigs` and newer render_portraits runs can add
                 "monsters/battleworn_dummy.png", "monsters/decimillipede.png")),
             portraits, ("uv", "node", "npm")),
        Step("vfx_bindings.json + creature_scales.json -> the art tree", "workbench",
             lambda: (images / "vfx/vfx_bindings.json").is_file() and (images / "creature_scales.json").is_file(),
             art_tables),
        Step(f"sfx map -> {(public_ver / 'sfx_map.json').relative_to(ROOT).as_posix()}", "workbench",
             lambda: (public_ver / "sfx_map.json").is_file(), sfx_map, ("uv",), versioned=True),
        Step("card text + card db -> sts2-pycli/", "workbench",
             lambda: (public_ver / "card_text.json").is_file() and card_db_json.is_file(),
             card_text_and_db, versioned=True),
        Step(f"workbench bundle -> {bundle_dir.relative_to(ROOT).as_posix()}/", "workbench",
             lambda: (bundle_dir / "wiki_cards.json").is_file(), workbench_data, versioned=True),
        Step("sound effects -> sts2-wiki/site/public/audio/ (FMOD: the game's own DLLs on Windows, else FMOD_SDK_ROOT)",
             "sound",
             (lambda: any(audio.rglob("*.webm"))) if is_default else sfx_merged.is_file,
             sfx, ("uv", "ffmpeg"), versioned=True),
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
    print(f"== {version}{' (default)' if version == DEFAULT_VERSION else ''}  (supported: {', '.join(SUPPORTED_VERSIONS)})")
    if build and build.is_dir():
        print(f"staged build: {build.relative_to(ROOT)}  (version {version})")
    else:
        print("staged build: none (sts2_binaries/<version>/ absent) — run `just import-game [GAME_DIR]`")
    steps = build_steps(build or BINARIES / version, version)
    for tier in TIERS[1:]:
        print(f"\n[{tier}]")
        for step in steps:
            if step.tier == tier and (step.versioned or version == DEFAULT_VERSION):
                print(f"  {'ok     ' if step.done() else 'MISSING'}  {step.name}")
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
    parser.add_argument("--tier", choices=TIERS + ["all"], default="all")
    parser.add_argument("--status", action="store_true", help="report which artifacts exist and exit")
    parser.add_argument("--copy", action="store_true", help="copy the pck and DLLs into sts2_binaries/ instead of symlinking")
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help=f"proceed even if the install is none of {', '.join(SUPPORTED_VERSIONS)} (expect drift)")
    parser.add_argument("--download", nargs="?", const="all", choices=[*SUPPORTED_VERSIONS, "all"],
                        help="download the build(s) from Steam (default: all supported) and import each")
    parser.add_argument("--steam-user", help="--download: your Steam account name (DepotDownloader asks for the password)")
    parser.add_argument("--steam-qr", action="store_true", help="--download: log in by scanning a QR code in the Steam app")
    parser.add_argument("--depotdownloader", help="--download: path to the DepotDownloader executable")
    args = parser.parse_args(argv)
    tier = "sound" if args.tier == "all" else args.tier
    for stream in (sys.stdout, sys.stderr):
        # Arrows and dashes in the log lines; a cp1252 console would print mojibake.
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    if args.status:
        staged = {p.name: p for p in BINARIES.glob("v*")} if BINARIES.is_dir() else {}
        for version in SUPPORTED_VERSIONS:
            status(staged.get(version), version)
            print()
        print("\n".join(art_coverage()))
        return 0

    if tier == "engine":
        print("tier engine: nothing to import — `cd sts2-rs && cargo nextest run --workspace` works as-is.")
        return 0

    if args.download:
        if args.game_dir:
            parser.error("--download fetches the game itself; drop GAME_DIR")
        if not (args.steam_user or args.steam_qr):
            parser.error("--download logs in to Steam as you: pass --steam-user NAME or --steam-qr")
        tool = find_depotdownloader(args.depotdownloader)
        versions = list(SUPPORTED_VERSIONS) if args.download == "all" else [args.download]
        # Every download first (the logins happen up front), then the imports,
        # the default build first: a later build's art overlay is what the
        # default build's shared art tree lacks.
        builds = [download(v, tool, args.steam_user, args.steam_qr) for v in versions]
        for build in builds:
            import_build(str(build), tier, args)
        return 0
    import_build(args.game_dir, tier, args)
    return 0


def import_build(game_dir: str | None, tier: str, args: argparse.Namespace) -> None:
    """Stage the install at `game_dir` (None: find it) and run `tier`'s steps on it."""
    data_dir, pck, release_info = locate(game_dir)
    version = read_version(release_info)
    log(f"game install: data={data_dir} pck={pck} version={version}")
    if version not in SUPPORTED_VERSIONS and not args.allow_version_mismatch:
        raise SystemExit(
            f"the install is {version} but the engine supports {', '.join(SUPPORTED_VERSIONS)}; the parity oracle "
            "and the extracted data would describe a different game. Pass --allow-version-mismatch to proceed anyway."
        )
    build = stage(data_dir, pck, release_info, version, args.copy)

    wanted = TIERS[: TIERS.index(tier) + 1]
    steps = [s for s in build_steps(build, version) if s.tier in wanted]
    if version != DEFAULT_VERSION:
        # The shared trees (localization_eng/, the art, the card db, the
        # workbench bundle) describe the default build; another build gets only
        # its version-keyed outputs, never an overwrite of those.
        for s in steps:
            if not s.versioned:
                log(f"skip ({version} is not the default {DEFAULT_VERSION}): {s.name}")
        steps = [s for s in steps if s.versioned]
    if (tier == "sound" and extract_sfx.find_fmod_runtime(build) is None
            and any(s.tier == "sound" and not s.done() for s in steps)):
        raise SystemExit(
            "no FMOD runtime for the sound tier. On Windows the game's own fmod.dll + fmodstudio.dll are used, "
            "but they were not found next to the pck (or in FMOD_LIB_DIR); elsewhere set FMOD_SDK_ROOT to the "
            "unpacked FMOD Studio API (a free, registration-gated download from fmod.com). Or pass --tier "
            "workbench to import everything but the sound."
        )
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
        print("\nnext: the browser workbench -  just workbench-local   (http://127.0.0.1:4173)\n"
              "      the Python workbench  -  just workbench-server  (http://127.0.0.1:8077)\n"
              "      the full test suites  -  see docs/bring-your-own-game.md")


if __name__ == "__main__":
    sys.exit(main())
