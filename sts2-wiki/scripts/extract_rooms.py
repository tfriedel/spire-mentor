#!/usr/bin/env python3
"""Extract room / shop / rest-site / event / Neow / treasure art from the STS2 PCK.

Companion to ``extract_images.py`` (which handles atlases/sprites). This pulls
*standalone* VRAM textures used as full-scene art for the workbench's game-like
decision screens. Each logical ``foo.png`` has a ``foo.png.import`` sidecar in
the PCK whose ``path[.fmt]="res://....ctex"`` points at the decodable texture.

Output mirrors the workbench's served-art convention under ``OUTPUT/``:
    events/<slug>.png             per-event illustration  (key: event slug)
    run_history/<slug>.png        map room-type icons
    rooms/<slug>/...              combat-arena / room background layers
                                  (glory, hive, overgrowth, underdocks, boss
                                  rooms, rest_sites/<region>, merchant_room)
    map_bgs/<region>/...          map parchment layers (top/middle/bottom)
    map_icons/...                 map markers / pings
    scene_bgs/<scene>/...         layered scene art from animations/backgrounds
                                  (neow_room, merchant_room, treasure_room, ...)
                                  plus rest_site/<char> rigs and ui transitions

Large source textures are downscaled so the longest edge is <= MAX_EDGE
(1600); the near-fullscreen scene categories (rooms, map_bgs, scene_bgs) get
a higher SCENE_MAX_EDGE (1920) cap instead.
Idempotent: existing outputs are skipped unless --force.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path

from PIL import Image

# Reuse the .ctex decoder (WebP/PNG/BC1/BC3/BC7) from the sibling extractor.
from scripts.extract_images import decode_ctex

MAX_EDGE = 1600

# Arena backdrops / map parchment / scene layers are shown near-fullscreen in
# the workbench, so they keep more resolution than icons and event stills.
SCENE_MAX_EDGE = 1920
CATEGORY_MAX_EDGE = {
    "rooms": SCENE_MAX_EDGE,
    "map_bgs": SCENE_MAX_EDGE,
    "scene_bgs": SCENE_MAX_EDGE,
}

# Sheets a combat room hangs IN THE AIR rather than on a layer: additive fog
# puffs, drifting on `CPUParticles2D` between two parallax slots. They live
# under `images/vfx/` with the spark and mote sheets, but unlike those they are
# room art — the Overgrowth's four teal fog sprites are what lifts its far wall
# ~20% in G and B over the bare layer stack, which is the whole of GH #610's
# "gap that grows with distance". Served under `rooms/air/` so the layer
# manifest can place them (`extract_bg_layers.AIR_SHEETS`).
ROOM_AIR_SHEETS = frozenset({"images/vfx/fog_sheet_1.png"})


def read_pck_index(pck_path: str) -> dict[str, tuple[int, int]]:
    """Return {logical_path: (absolute_offset, size)} for every file in the PCK."""
    index: dict[str, tuple[int, int]] = {}
    with open(pck_path, "rb") as f:
        assert f.read(4) == b"GDPC", "not a Godot PCK"
        fmt_version = struct.unpack("<I", f.read(4))[0]
        f.read(16)  # engine version
        file_base = struct.unpack("<Q", f.read(8))[0]
        directory_offset = struct.unpack("<Q", f.read(8))[0] if fmt_version >= 3 else 0
        f.read(64)  # reserved
        if fmt_version >= 3:
            f.seek(directory_offset)
        file_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(file_count):
            path_len = struct.unpack("<I", f.read(4))[0]
            path = f.read(path_len).rstrip(b"\x00").decode("utf-8")
            offset = struct.unpack("<q", f.read(8))[0]
            size = struct.unpack("<q", f.read(8))[0]
            f.read(16 + 4)  # md5 + flags
            index[path] = (file_base + offset, size)
    return index


def make_reader(pck_path: str, index: dict[str, tuple[int, int]]):
    def read(path: str) -> bytes | None:
        if path not in index:
            return None
        offset, size = index[path]
        with open(pck_path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    return read


def resolve_ctex(import_text: str) -> str | None:
    """Pick the best .ctex referenced by a .png.import (prefer BPTC > S3TC > plain)."""
    paths: list[str] = re.findall(r'path[^=]*="res://([^"]+\.ctex)"', import_text)
    if not paths:
        return None
    paths.sort(key=lambda p: 0 if "bptc" in p else 1 if "s3tc" in p else 2)
    return paths[0]


def downscale(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    longest = max(img.width, img.height)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS
    )


def output_for(source: str) -> str | None:
    """Map a PCK source path (no .import suffix) to an output rel path, or None to skip.

    Families of flat textures extracted:

    * per-event illustrations (``images/events/<slug>.png``) — full-scene art.
    * map room-type icons (``images/ui/run_history/<slug>.png``) — the small
      per-room-type badges the game draws on the map / in the run history
      (monster, elite, shop, rest_site, treasure, event, ancient, and per-boss
      icons). These are flat 128px textures, so the ``.ctex`` decodes cleanly.
    * combat-arena / room background layers (``images/rooms/<slug>/...``) —
      flat parallax layers per region (glory, hive, overgrowth, underdocks),
      boss rooms, rest-site scenes (``rooms/rest_sites/<region>``) and the
      merchant-room props. Subdirectory structure is preserved.
    * map parchment layers (``images/packed/map/map_bgs/<region>/...``) and
      map markers (``images/packed/map/icons/...``).
    * layered scene backgrounds (``animations/backgrounds/<scene>/...``) —
      neow_room, merchant_room, fake_merchant_room, treasure_room, mainmenu,
      ceremonial_beast, tezcatara. Despite living under ``animations/`` these
      are flat layer textures, not Spine atlas pages (the old claim that all
      scene art is Spine-only was wrong — only the character/monster rigs are).
    * per-character rest-site rigs (``animations/rest_site/<char>/...``) and
      full-screen UI transitions (``images/ui/transitions/...``, including the
      game-over transition) → ``scene_bgs/``. The rest-site rigs ARE packed
      Spine atlas pages, extracted anyway as the closest campfire-scene art.

    Args:
        source: e.g. ``images/events/abyssal_baths.png``.
    """
    if source.startswith("images/events/"):
        rest = source[len("images/events/") :]
        if "/" in rest:
            return None  # sub-asset (e.g. crystal_sphere/*), not the main illustration
        return f"events/{rest}"
    if source.startswith("images/ui/run_history/"):
        rest = source[len("images/ui/run_history/") :]
        # Skip the outline overlays (paired ``*_outline.png``) and the
        # run-history card-portrait chrome — the map only needs the filled icon.
        if "/" in rest or rest.endswith("_outline.png"):
            return None
        return f"run_history/{rest}"
    if source.startswith("images/rooms/"):
        return "rooms/" + source[len("images/rooms/") :]
    if source in ROOM_AIR_SHEETS:
        return "rooms/air/" + source.rsplit("/", 1)[1]
    if source.startswith("images/packed/map/map_bgs/"):
        return "map_bgs/" + source[len("images/packed/map/map_bgs/") :]
    if source.startswith("images/packed/map/icons/"):
        return "map_icons/" + source[len("images/packed/map/icons/") :]
    if source.startswith("animations/backgrounds/"):
        return "scene_bgs/" + source[len("animations/backgrounds/") :]
    if source.startswith("animations/rest_site/"):
        return "scene_bgs/rest_site/" + source[len("animations/rest_site/") :]
    if source.startswith("images/ui/transitions/"):
        return "scene_bgs/transitions/" + source[len("images/ui/transitions/") :]
    # Static fallbacks the workbench resolves by name: the map's boss icons
    # (``boss_icons/<boss>_boss_icon.png``), the Defect orb sprites
    # (``orb_atlas/<orb>.png``; the animated orbs are Spine rigs) and the
    # game's own static monster sprites (``monsters/<name>.png`` — e.g. the
    # hourglass Aeonglass is drawn as, which has no rig). The ancients'
    # portraits are the map-node icons; see ``aliases_for``.
    m = re.fullmatch(r"images/map/placeholder/(\w+_boss_icon)\.png", source)
    if m:
        return f"boss_icons/{m.group(1)}.png"
    m = re.fullmatch(r"images/orbs/(\w+)_orb\.png", source)
    if m:
        return f"orb_atlas/{m.group(1)}.png"
    m = re.fullmatch(r"images/monsters/(\w+)\.png", source)
    if m:
        return f"monsters/{m.group(1)}.png"
    # Screen-chrome families: the game's drawn HUD/panel/button art. Flat
    # textures; served under ui/<family>/ for the workbench's game-skin
    # components (top HUD band, energy orb layers, reward banner, rest-site
    # plaques, game-over tally, event option panels, pile/end-turn buttons).
    for prefix, family in (
        ("images/ui/top_panel/", "top_panel"),
        ("images/ui/combat/", "combat"),
        ("images/ui/rest_site/", "rest_site"),
        ("images/ui/reward_screen/", "reward_screen"),
        ("images/ui/game_over_screen/", "game_over_screen"),
        ("images/ui/emote/", "emote"),
        ("images/packed/combat_ui/", "combat_ui"),
        ("images/packed/common_ui/", "common_ui"),
        ("images/packed/map/", "map"),
        ("images/packed/sprite_fonts/", "sprite_fonts"),
    ):
        if source.startswith(prefix):
            return f"ui/{family}/" + source[len(prefix) :]
    return None


def aliases_for(rel_out: str) -> list[str]:
    """Further output paths that carry the same image as ``rel_out``.

    The top-panel character icons double as ``characters/character_icon_<c>.png``,
    the path the run-setup and history views ask for; the map's ancient-node
    icons (278px) as ``ancients/<name>.png``. (``images/ancients/*_placeholder``
    are 2560x1200 scene backdrops, not portraits, and only exist for six of
    the eight ancients.)
    """
    m = re.fullmatch(r"ui/top_panel/(character_icon_\w+)\.png", rel_out)
    if m and not m.group(1).endswith("_outline"):
        return [f"characters/{m.group(1)}.png"]
    m = re.fullmatch(r"ui/map/ancients/ancient_node_(\w+)\.png", rel_out)
    if m and not m.group(1).endswith("_outline"):
        return [f"ancients/{m.group(1)}.png"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract room/event/shop art from the STS2 PCK")
    parser.add_argument("pck_path")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = parser.parse_args()

    pck_path = os.path.expanduser(args.pck_path)
    out_root = Path(args.output_dir)
    index = read_pck_index(pck_path)
    read = make_reader(pck_path, index)

    extracted = 0
    skipped = 0
    failed = 0
    by_category: dict[str, int] = {}
    for path in index:
        if not path.endswith(".png.import"):
            continue
        source = path[: -len(".import")]
        rel_out = output_for(source)
        if rel_out is None:
            continue
        out_path = out_root / rel_out
        category = rel_out.split("/")[0]
        alias_paths = [out_root / a for a in aliases_for(rel_out)]
        if out_path.exists() and all(p.exists() for p in alias_paths) and not args.force:
            skipped += 1
            continue
        import_text = (read(path) or b"").decode("utf-8", "replace")
        ctex = resolve_ctex(import_text)
        if not ctex or ctex not in index:
            print(f"  no .ctex for {source}")
            failed += 1
            continue
        ctex_data = read(ctex)
        img = decode_ctex(ctex_data) if ctex_data else None
        if img is None:
            print(f"  decode failed: {source}")
            failed += 1
            continue
        img = downscale(img, CATEGORY_MAX_EDGE.get(category, MAX_EDGE))
        for p in (out_path, *alias_paths):
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, "PNG")
        extracted += 1
        by_category[category] = by_category.get(category, 0) + 1

    print(
        f"\nextracted {extracted} (skipped {skipped}, failed {failed}); "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items()))
    )


if __name__ == "__main__":
    main()
