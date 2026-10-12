#!/usr/bin/env python3
"""Render the static creature portraits from the extracted Spine rigs.

The workbench draws creatures from their rigs, but a few views (and the rig-less
fallback) want one still image per creature: ``monsters/<slug>.png`` (512px)
and ``character_sprites/<character>.png`` (640px, the large art on *New run*).
The game ships no such stills, so they are rendered here, idle pose, from the
rigs ``scripts/extract_rig.py --all`` wrote, with ``tools/spine-renderer``
(Node + node-canvas).

Then every monster the workbench files under an older name (the
``_MONSTER_ALIASES`` table in ``sts2-pycli/python/workbench/assets.py``:
``torch_head_amalgam`` -> ``amalgam``, ``globe_head`` -> ``orb_head``,
``ovicopter`` -> ``egg_layer``, ...) also gets its portrait under that name.

Idempotent: existing PNGs are kept (``--force`` re-renders).

Usage:
    uv run python -m scripts.render_portraits [IMAGES_ROOT] [--force]
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from pathlib import Path

WIKI = Path(__file__).resolve().parents[1]
RENDERER = WIKI / "tools" / "spine-renderer"
ASSETS_PY = WIKI.parent / "sts2-pycli" / "python" / "workbench" / "assets.py"

# (rig folder, output folder, edge px)
JOBS = [("monster_rigs", "monsters", 512), ("character_rigs", "character_sprites", 640)]


def _tool(name: str) -> str:
    # npm is npm.cmd on Windows; subprocess needs the resolved path there.
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f"{name} not found on PATH (Node.js: https://nodejs.org; "
            "Windows: winget install OpenJS.NodeJS.LTS)"
        )
    return path


def ensure_renderer() -> None:
    """`npm ci` the renderer once, and prove node-canvas's native binding loads."""
    node = _tool("node")
    if not (RENDERER / "node_modules" / "canvas").is_dir():
        subprocess.run([_tool("npm"), "ci", "--no-audit", "--no-fund"], cwd=RENDERER, check=True)
    probe = subprocess.run(
        [node, "-e", "require('canvas').createCanvas(1, 1).toBuffer('image/png')"],
        cwd=RENDERER,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "node-canvas does not load (its install script fetches a prebuilt binary; "
            f"npm 11+ may have skipped it). In {RENDERER}: `npm approve-scripts canvas` "
            "then `npm rebuild canvas`.\n" + probe.stderr[-2000:]
        )


def monster_aliases() -> dict[str, str]:
    """`_MONSTER_ALIASES` from the workbench's resolver, read without importing it."""
    tree = ast.parse(ASSETS_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_MONSTER_ALIASES" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise SystemExit(f"_MONSTER_ALIASES not found in {ASSETS_PY}")


def write_aliases(monsters: Path, aliases: dict[str, str]) -> int:
    """Copy `<slug>.png` to `<alias>.png` where the alias has no image yet."""
    written = 0
    for slug, alias in sorted(aliases.items()):
        src, dst = monsters / f"{slug}.png", monsters / f"{alias}.png"
        if src.is_file() and not dst.exists():
            shutil.copyfile(src, dst)
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "images_root", nargs="?", type=Path, default=WIKI / "site" / "public" / "images"
    )
    ap.add_argument("--force", action="store_true", help="re-render portraits that already exist")
    args = ap.parse_args(argv)
    images = args.images_root.resolve()

    ensure_renderer()
    for rigs, out, size in JOBS:
        if not (images / rigs).is_dir():
            raise SystemExit(
                f"{images / rigs} is missing: run "
                f"`python -m scripts.extract_rig PCK --all {images}` first"
            )
        cmd = [
            _tool("node"),
            "render.mjs",
            "--rigs",
            str(images / rigs),
            "--out",
            str(images / out),
            "--size",
            str(size),
        ]
        subprocess.run(cmd + (["--force"] if args.force else []), cwd=RENDERER, check=True)
    n = write_aliases(images / "monsters", monster_aliases())
    print(f"portrait aliases written: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
