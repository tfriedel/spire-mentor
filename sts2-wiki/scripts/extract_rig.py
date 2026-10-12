#!/usr/bin/env python3
"""Extract a monster/character Spine rig (.skel + .atlas + .png) from the game PCK.

The game ships Spine rigs as Godot *imported* resources under `.godot/imported/`:
  <name>.skel-<hash>.spskel   — the raw Spine binary skeleton, verbatim
  <name>.atlas-<hash>.spatlas — JSON `{"atlas_data": "<libgdx .atlas text>"}`
  <name>.png-<hash>.ctex      — Godot CompressedTexture2D (BPTC/S3TC) atlas page

This reconstructs the three plain files the workbench/wiki expect under
  site/public/images/{monster_rigs,character_rigs}/<slug>/<slug>.{skel,atlas,png}

Usage:
  uv run python -m scripts.extract_rig <pck> <source_name> <out_dir> [--slug SLUG]
e.g.
  uv run python -m scripts.extract_rig SlayTheSpire2.pck skulking_colony \
      site/public/images/monster_rigs
"""

import argparse
import json
import os
import re
import struct
import sys

from scripts.extract_images import decode_ctex


def read_pck_index(pck_path):
    """Return {res-path: (offset, size)} for a Godot 4 PCK (mirrors extract_pck)."""
    with open(pck_path, "rb") as f:
        assert f.read(4) == b"GDPC", "bad PCK magic"
        fmt = struct.unpack("<I", f.read(4))[0]
        f.read(12)  # engine major/minor/patch
        f.read(4)  # pack flags
        file_base = struct.unpack("<Q", f.read(8))[0]
        dir_off = struct.unpack("<Q", f.read(8))[0] if fmt >= 3 else None
        f.read(64)  # reserved
        if dir_off is not None:
            f.seek(dir_off)
        n = struct.unpack("<I", f.read(4))[0]
        index = {}
        for _ in range(n):
            pl = struct.unpack("<I", f.read(4))[0]
            path = f.read(pl).rstrip(b"\x00").decode("utf-8")
            offset, size = struct.unpack("<q", f.read(8))[0], struct.unpack("<q", f.read(8))[0]
            f.read(16 + 4)  # md5 + flags
            index[path] = (file_base + offset, size)
    return index


class RigError(Exception):
    """A rig whose skeleton, atlas or texture pages are not in the PCK."""


def _read(pck_path, offset, size):
    with open(pck_path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def _find(index, name, ext):
    """Find `.godot/imported/<name>.<src_ext>-<hash>.<ext>` entries, largest first."""
    pre = f".godot/imported/{name}."
    hits = [p for p in index if p.startswith(pre) and p.endswith(ext) and f".{ext_src(ext)}-" in p]
    return sorted(hits, key=lambda p: index[p][1], reverse=True)


def ext_src(ext):
    return {"spskel": "skel", "spatlas": "atlas", "ctex": "png"}[ext]


def _page_sizes(lines):
    """{page file: (w, h)} from the `size:` line under each atlas page header.

    A page's properties are the `key:value` lines right after its file name;
    the first line without a colon is a region name, whose own `size:` (older
    atlas format) is not the page's.
    """
    sizes = {}
    page = None
    for ln in lines:
        if ln.endswith(".png"):
            page = ln
        elif ":" not in ln:
            page = None
        elif page and ln.startswith("size:"):
            w, h = (int(v) for v in ln[5:].replace(" ", "").split(","))
            sizes[page] = (w, h)
            page = None
    return sizes


def extract_rig(pck_path, name, out_root, slug=None, atlas_name=None, index=None):
    """Reconstruct a Spine rig (``.skel`` + ``.atlas`` + page ``.png``\\ s).

    Most rigs name the skeleton, atlas and texture page identically (``name``).
    Some don't: the Decimillipede segments pair e.g. ``decimillipede3.skel`` with
    ``decimillipede_back.atlas`` (and all three segments share one
    ``decimillipede.png`` page), so the skeleton and atlas are passed separately.
    Texture-page filenames are always read from the atlas itself.

    Args:
        pck_path: Path to the game PCK.
        name: Source skeleton asset name in the PCK (the ``.spskel``).
        out_root: Output root; the rig is written under ``<out_root>/<slug>/``.
        slug: Output dir/file slug (default: ``name``).
        atlas_name: Source atlas asset name, when it differs from ``name``
            (default: ``name``).
        index: A ``read_pck_index`` result to reuse (batch callers); read
            from ``pck_path`` when omitted.

    Raises:
        RigError: the skeleton, atlas or a texture page is missing.
    """
    slug = slug or name
    atlas_name = atlas_name or name
    index = index if index is not None else read_pck_index(pck_path)
    out_dir = os.path.join(out_root, slug)
    os.makedirs(out_dir, exist_ok=True)

    # .skel — the .spskel is the raw Spine binary, verbatim
    skel = _find(index, name, "spskel")
    if not skel:
        raise RigError(f"no .spskel for '{name}' — it likely has no Spine rig (static sprite)")
    with open(os.path.join(out_dir, f"{slug}.skel"), "wb") as o:
        o.write(_read(pck_path, *index[skel[0]]))

    # .atlas — libgdx atlas text inside the .spatlas JSON wrapper. A rig may pack
    # into MULTIPLE atlas pages (kaiser_crab → kaiser_crab.png + kaiser_crab_2.png
    # …); the atlas lists each page by its own image filename. Parse those page
    # names so every page gets extracted, not just the first.
    atl = _find(index, atlas_name, "spatlas")
    if not atl:
        raise RigError(f"no .spatlas for '{atlas_name}'")
    atlas_text = json.loads(_read(pck_path, *index[atl[0]]))["atlas_data"]
    # Page headers are the only lines that end in ".png" (region names never do).
    lines = [ln.strip() for ln in atlas_text.splitlines()]
    page_files = [ln for ln in lines if ln.endswith(".png")]
    if not page_files:
        raise RigError(f"no atlas pages found for '{atlas_name}'")
    page_sizes = _page_sizes(lines)
    # Output page filenames, by slug: <slug>.png, <slug>_2.png, … (the page-index
    # convention the libgdx atlas / spine-webgl loader expects).
    out_pngs = [f"{slug}.png" if i == 0 else f"{slug}_{i + 1}.png" for i in range(len(page_files))]
    # Rewrite the atlas's page references (whatever they were named) to ours.
    for src_pf, out_pf in zip(page_files, out_pngs):
        if src_pf != out_pf:
            atlas_text = atlas_text.replace(src_pf, out_pf, 1)
    with open(os.path.join(out_dir, f"{slug}.atlas"), "w", encoding="utf-8") as o:
        o.write(atlas_text)

    # .png — decode each page's Godot compressed-texture (one .ctex per page; a
    # page may ship >1 compression variant — BPTC/S3TC — so try largest first).
    # The imported names drop the source folder, so an unrelated texture can
    # share the page's name: `battleworn_dummy.png` is both the rig page
    # (396x502) and the event illustration (3440x1613), and largest-first
    # picks the illustration. The atlas states each page's size; a candidate
    # of that size wins.
    sizes = []
    for src_pf, out_pf in zip(page_files, out_pngs):
        page_name = src_pf[:-4]  # source ctex asset name, e.g. "decimillipede"
        ctex = _find(index, page_name, "ctex")
        if not ctex:
            raise RigError(f"no .ctex page for atlas page '{page_name}' (rig '{name}')")
        img = None
        want = page_sizes.get(src_pf)
        for c in ctex:
            decoded = decode_ctex(_read(pck_path, *index[c]))
            if decoded is None:
                continue
            if img is None:
                img = decoded
            if want is None or decoded.size == want:
                img = decoded
                break
        if img is None:
            raise RigError(f"failed to decode any .ctex variant for page '{page_name}'")
        if want is not None and img.size != want:
            print(
                f"  warning: {name}: no {want[0]}x{want[1]} texture for page "
                f"'{page_name}', using {img.size}"
            )
        img.save(os.path.join(out_dir, out_pf))
        sizes.append(img.size)
    return out_dir, sizes[0], len(page_files)


#: Room rigs the workbench draws under a slug no scene or folder carries:
#: the treasure-room chests (``room_rigs/treasure_act_<n>``) and the merchant.
ROOM_RIGS = {
    "treasure_act_1": "chest_room_act_1",
    "treasure_act_2": "chest_room_act_2",
    "treasure_act_3": "chest_room_act_3",
    "shop_merchant_top": "shop_merchant_top",
}


def _rig_of_resource(index, pck_path, res_path):
    """(skel, atlas, skel res-path) named by a Spine skeleton-data ``.tres``, else None."""
    if res_path not in index:
        return None
    text = _read(pck_path, *index[res_path]).decode("utf-8", "replace")
    refs = [r.removeprefix("res://") for r in re.findall(r'path="(res://[^"]+)"', text)]
    skel = next((r for r in refs if r.endswith(".skel")), None)
    atlas = next((r for r in refs if r.endswith(".atlas")), None)
    if not (skel and atlas):
        return None
    return os.path.basename(skel)[: -len(".skel")], os.path.basename(atlas)[: -len(".atlas")], skel


def find_rigs(index, pck_path):
    """Every creature and room rig, as ``{(category, slug): (skel, atlas)}``.

    Creatures resolve through the game's own scenes first:
    ``scenes/creature_visuals/<slug>.tscn`` references a skeleton-data ``.tres``
    that names the ``.skel`` and ``.atlas``. That is the only reliable pairing —
    the Decimillipede segments cross their file names (``segment_front`` draws
    ``decimillipede3.skel`` with ``decimillipede_back.atlas``). A rig several
    scenes share is published under its folder or skeleton name instead
    (``animations/monsters/bowlbug/`` → ``bowlbug``), so those are added too,
    when the atlas of the same name exists. A skeleton under
    ``animations/characters/`` is a character rig; every other creature is a
    monster rig. Rooms come from ``ROOM_RIGS``.
    """
    rigs = {}
    for path in sorted(index):
        m = re.fullmatch(r"scenes/creature_visuals/(\w+)\.tscn", path)
        if not m:
            continue
        scene = _read(pck_path, *index[path]).decode("utf-8", "replace")
        for res in re.findall(r'path="res://([^"]+\.tres)"', scene):
            rig = _rig_of_resource(index, pck_path, res)
            if rig:
                skel, atlas, skel_path = rig
                category = (
                    "character_rigs"
                    if skel_path.startswith("animations/characters/")
                    else "monster_rigs"
                )
                rigs[(category, m.group(1))] = (skel, atlas)
                break

    folders = {}
    for path in index:
        m = re.fullmatch(
            r"animations/(monsters|characters)/(?:.+/)?(\w+)/(\w+)\.skel\.import", path
        )
        if m:
            folders.setdefault((m.group(1), m.group(2)), []).append(m.group(3))
    for (kind, folder), names in sorted(folders.items()):
        category = "character_rigs" if kind == "characters" else "monster_rigs"
        for name in names:
            if not _find(index, name, "spatlas"):
                continue
            slugs = [name] + ([folder] if len(names) == 1 and folder != name else [])
            for slug in slugs:
                rigs.setdefault((category, slug), (name, name))

    for slug, name in ROOM_RIGS.items():
        rigs[("room_rigs", slug)] = (name, name)
    return rigs


def extract_all(pck_path, images_root, force=False):
    """Write every rig ``find_rigs`` names under ``images_root``; returns the failures.

    Idempotent: a rig whose ``<slug>.skel`` exists is skipped unless ``force``.
    """
    index = read_pck_index(pck_path)
    rigs = find_rigs(index, pck_path)
    written = skipped = 0
    failures = []
    for (category, slug), (skel, atlas) in sorted(rigs.items()):
        out_root = os.path.join(images_root, category)
        if not force and os.path.isfile(os.path.join(out_root, slug, f"{slug}.skel")):
            skipped += 1
            continue
        try:
            extract_rig(pck_path, skel, out_root, slug=slug, atlas_name=atlas, index=index)
            written += 1
        except RigError as exc:
            failures.append(f"{category}/{slug}: {exc}")
    print(f"rigs: {written} written, {skipped} present, {len(failures)} failed (of {len(rigs)})")
    for failure in failures:
        print(f"  FAILED {failure}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pck")
    ap.add_argument(
        "name", nargs="?", help="source skeleton asset name in the PCK, e.g. skulking_colony"
    )
    ap.add_argument("out_root", nargs="?", help="e.g. site/public/images/monster_rigs")
    ap.add_argument("--slug", help="output dir/file slug (default: name)")
    ap.add_argument("--atlas", help="source atlas asset name when it differs from <name>")
    ap.add_argument(
        "--all", metavar="IMAGES_ROOT", help="extract every creature and room rig under IMAGES_ROOT"
    )
    ap.add_argument(
        "--force", action="store_true", help="with --all: rewrite rigs that already exist"
    )
    a = ap.parse_args()
    if a.all:
        sys.exit(1 if extract_all(a.pck, a.all, a.force) else 0)
    if not (a.name and a.out_root):
        ap.error("name and out_root are required without --all")
    try:
        out_dir, size, npages = extract_rig(a.pck, a.name, a.out_root, a.slug, a.atlas)
    except RigError as exc:
        sys.exit(str(exc))
    # ASCII only: this is the LAST line of a successful run, and on a cp1252
    # console a non-ASCII glyph here raises UnicodeEncodeError *after* every
    # file is already written - so a clean extraction exits 1 with a traceback.
    print(f"wrote rig -> {out_dir}  (page 1 {size[0]}x{size[1]}, {npages} atlas page(s))")


if __name__ == "__main__":
    main()
