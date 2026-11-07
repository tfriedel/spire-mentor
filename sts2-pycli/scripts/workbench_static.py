#!/usr/bin/env python3
"""Build and serve the browser-only (static) workbench.

    uv run --no-project scripts/workbench_static.py build     # wasm + SPA + art -> web/dist
    uv run --no-project scripts/workbench_static.py preview   # serve web/dist on http://127.0.0.1:4173

Subcommands ``wasm`` and ``art`` run one stage of ``build`` on its own;
``check-dist`` runs only the size gate over ``web/dist``. The
justfile recipes (``wasm-build``, ``workbench-static``, ``_art-into-dist``,
``workbench-static-preview``, and the monorepo root's ``workbench-local``) all
call this script, so the build runs the same from PowerShell, Git Bash, Linux
and macOS. Stdlib only.

Stages of ``build``:
  1. ``wasm``  — cargo builds ``sts2-workbench-wasm`` for wasm32, wasm-bindgen
     writes ``web/src/wasm/pkg``, wasm-opt shrinks it when available, and a
     gzipped copy (``*.wasm.gz``) is written next to it: the static build
     ships ONLY that copy (the worker inflates it with ``DecompressionStream``;
     ``vite.config.ts`` points the worker's import at it), the dev server the
     plain ``.wasm``.
  2. the SPA   — ``npm run build`` in static mode into ``web/dist``.
  3. ``art``   — the extracted art tree (``just import-game``) into
     ``web/dist/assets`` as WebP (``art_to_dist.py``; needs ``cwebp``). Without
     an art tree every image renders the placeholder tile and sound is silent.
  4. the size gate — no file in ``web/dist`` may exceed Cloudflare Pages'
     25 MiB per-file cap (``check-dist``), so an over-size build fails here
     instead of at ``wrangler pages deploy``.

Environment: ``VITE_BUILD_ID`` (default: ``git describe``), ``VITE_ASSETS_BASE``
(build against a hosted art origin; then dist carries no art),
``STS2_WORKBENCH_NO_ART`` (placeholder art even when the tree exists),
``VITE_BRIDGE_SAME_ORIGIN=1`` (the build the mod bundles, see
``web/src/bridge.ts``), ``STS2_WEB_HOST`` (the preview's bind address,
default 127.0.0.1).
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PYCLI = Path(__file__).resolve().parent.parent
RS = PYCLI.parent / "sts2-rs"
WEB = PYCLI / "web"
PKG = WEB / "src" / "wasm" / "pkg"
IMAGES = PYCLI.parent / "sts2-wiki" / "site" / "public" / "images"
WASM = RS / "target" / "wasm32-unknown-unknown" / "wasm-release" / "sts2_workbench_wasm.wasm"
DIST = WEB / "dist"
#: Cloudflare Pages refuses any file over 25 MiB.
PAGES_FILE_CAP = 25 * 1024 * 1024


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}")


def tool(name: str, hint: str) -> str:
    """The full path of `name` (npm.cmd and friends resolve too on Windows), or exit with `hint`."""
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found — {hint}")
    return found


def locked_wasm_bindgen() -> str:
    lock = (RS / "Cargo.lock").read_text(encoding="utf-8")
    m = re.search(r'name = "wasm-bindgen"\nversion = "([^"]+)"', lock)
    if not m:
        raise SystemExit("could not read the wasm-bindgen version from sts2-rs/Cargo.lock")
    return m.group(1)


def web_deps() -> None:
    if not (WEB / "node_modules").is_dir():
        run(
            [
                tool("npm", "install Node.js 20+ (Windows: winget install OpenJS.NodeJS.LTS)"),
                "install",
            ],
            WEB,
        )


def wasm() -> None:
    version = locked_wasm_bindgen()
    install = f"cargo install wasm-bindgen-cli --version {version} --locked"
    bindgen = tool("wasm-bindgen", install)
    have = subprocess.run([bindgen, "--version"], capture_output=True, text=True).stdout.split()
    if version not in have:
        raise SystemExit(
            f"wasm-bindgen {' '.join(have[1:]) or '?'} does not match the crate ({version}): {install}"
        )
    # The wasm target comes with the toolchain sts2-rs/rust-toolchain.toml pins.
    run(
        [
            tool("cargo", "install Rust via https://rustup.rs"),
            "build",
            "--profile",
            "wasm-release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "sts2-workbench-wasm",
        ],
        RS,
        {"RUSTC_WRAPPER": ""},
    )
    PKG.mkdir(parents=True, exist_ok=True)
    # --omit-default-module-path: the glue's `init()` default is
    # `new URL("…_bg.wasm", import.meta.url)`, which Vite follows and emits
    # into dist next to the .gz. The worker always passes the module itself.
    run(
        [
            bindgen,
            "--target",
            "web",
            "--omit-default-module-path",
            "--out-dir",
            str(PKG),
            str(WASM),
        ],
        RS,
    )
    # binaryen's wasm-opt: a system install, else the `binaryen` npm package in
    # web/ (shrinks the PPO-bearing engine ~30%). Skipped, with a note, otherwise.
    web_deps()
    local = WEB / "node_modules" / ".bin"
    opt = shutil.which("wasm-opt") or shutil.which("wasm-opt", path=str(local))
    out = PKG / "sts2_workbench_wasm_bg.wasm"
    if opt:
        # The engine is built with +simd128 (sts2-rs/.cargo/config.toml);
        # say so explicitly rather than rely on wasm-opt reading the
        # target_features section.
        run([opt, "-O3", "--enable-simd", "-o", str(out), str(out)], RS)
    else:
        print(
            ">> note: wasm-opt not found (system or web/node_modules) — shipping the unoptimized wasm"
        )
    gz = compress_wasm(out)
    print(f">> {out} ({out.stat().st_size:,} B), {gz.name} ({gz.stat().st_size:,} B)")


def compress_wasm(wasm_file: Path) -> Path:
    """Write ``<wasm>.gz`` next to ``wasm_file``: what the static build ships.

    The engine outgrew Cloudflare Pages' 25 MiB per-file cap once it was built
    with SIMD; gzipped it is ~7 MB. Byte-stable (no name, mtime 0), so an
    unchanged engine keeps its content hash and the deploy re-uploads nothing.
    The browser inflates it itself (``DecompressionStream`` in the engine
    worker), so the host serves it as opaque bytes, no ``Content-Encoding``.
    """
    dest = wasm_file.with_name(wasm_file.name + ".gz")
    with (
        dest.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz,
    ):
        gz.write(wasm_file.read_bytes())
    return dest


def check_dist() -> None:
    """Fail when a file in web/dist is over Pages' per-file cap (or the engine ships uncompressed)."""
    if not DIST.is_dir():
        raise SystemExit("web/dist is missing — build it first")
    too_big = [
        (f.stat().st_size, f.relative_to(DIST).as_posix())
        for f in DIST.rglob("*")
        if f.is_file() and f.stat().st_size > PAGES_FILE_CAP
    ]
    raw_wasm = sorted(f.relative_to(DIST).as_posix() for f in DIST.rglob("*.wasm"))
    if too_big or raw_wasm:
        lines = [f"  {name}: {size:,} B" for size, name in sorted(too_big, reverse=True)]
        lines += [
            f"  {name}: the engine must ship as .wasm.gz (vite.config.ts)" for name in raw_wasm
        ]
        raise SystemExit(
            f"web/dist fails the size gate (Cloudflare Pages caps a file at {PAGES_FILE_CAP:,} B):\n"
            + "\n".join(lines)
        )
    print(f">> size gate: every web/dist file is under {PAGES_FILE_CAP:,} B")


def build_id() -> str:
    if os.environ.get("VITE_BUILD_ID"):
        return os.environ["VITE_BUILD_ID"]
    proc = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=12", "--exclude", "*"],
        cwd=PYCLI,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "local"


def spa() -> None:
    web_deps()
    run(
        [tool("npm", "install Node.js 20+"), "run", "build"],
        WEB,
        {
            "VITE_WORKBENCH_MODE": "static",
            "VITE_BUILD_ID": build_id(),
            "VITE_ASSETS_BASE": os.environ.get("VITE_ASSETS_BASE", ""),
        },
    )


def need_cwebp() -> None:
    if not shutil.which("cwebp"):
        raise SystemExit(
            "cwebp not found — the static build ships the art as WebP. Install libwebp "
            "(Windows: winget install Google.Libwebp, then open a new terminal; "
            "macOS: brew install webp; Debian: apt install webp)"
        )


def art() -> None:
    """The art tree into web/dist/assets, every PNG as WebP under its .png name.

    Vite empties dist on every build, so this runs after it, and refuses to
    finish unless every file of the art manifest made it (art_to_dist.py).
    """
    need_cwebp()
    run([sys.executable, "scripts/build_asset_manifest.py"], PYCLI)
    run([sys.executable, "scripts/art_to_dist.py"], PYCLI)


def build() -> None:
    with_art = (
        IMAGES.is_dir()
        and not os.environ.get("VITE_ASSETS_BASE")
        and not os.environ.get("STS2_WORKBENCH_NO_ART")
    )
    if with_art:
        need_cwebp()  # before the minutes of compiling, not after
    wasm()
    spa()
    if with_art:
        art()
    elif not os.environ.get("VITE_ASSETS_BASE"):
        print(
            f">> no art tree at {IMAGES} (or STS2_WORKBENCH_NO_ART set) — building in "
            "placeholder-art mode (`just import-game` extracts it)"
        )
    check_dist()


def preview() -> None:
    if not (WEB / "dist" / "index.html").is_file():
        raise SystemExit("web/dist is empty — build it first (`just workbench-local` does both)")
    web_deps()
    run(
        [tool("npx", "install Node.js 20+"), "vite", "preview"],
        WEB,
        {"VITE_WORKBENCH_MODE": "static"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("stage", choices=["build", "preview", "wasm", "art", "check-dist"])
    stage = parser.parse_args(argv).stage
    {"build": build, "preview": preview, "wasm": wasm, "art": art, "check-dist": check_dist}[
        stage
    ]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
