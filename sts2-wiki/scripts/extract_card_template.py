#!/usr/bin/env python3
"""Extract the game's *card template* art — the frame, banner, plaque and cost
icons that `scenes/cards/card.tscn` composites around a card's illustration.

`extract_images.py` pulls the per-card *illustrations* (``card_atlas/…``). This
script pulls the chrome drawn around them, so a card can be reproduced outside
the game (the workbench's ``CardFrame``) instead of approximated with CSS.

Two things make this more than a crop:

1. **The frame and banner ship untinted.** The game draws one greyscale-ish
   source sprite through ``shaders/hsv.gdshader`` with per-variant h/s/v
   uniforms — the frame tinted by *character* (`materials/cards/frames/`) and
   the banner by *rarity* (`materials/cards/banners/`). Rarity is the banner,
   not the border. We re-implement that shader in numpy and bake one PNG per
   variant, so the consumer needs no shader.
2. **Sprites live in an atlas.** ``ui_atlas.sprites/…/*.tres`` name a region of
   ``ui_atlas_{0,1}.png``, which is itself a BPTC-compressed ``.ctex``.

Output (under ``IMAGES_ROOT/card_template/``)::

    frames/{attack,skill,power,ancient,quest}_{red,green,blue,…}.png
    banners/card_banner_{common,uncommon,rare,…}.png
    portrait_borders/{attack,skill,power}.png
    energy/{ironclad,silent,defect,regent,necrobinder,colorless,quest}.png
    plaque.png, energy_star.png, unplayable.png

Usage:
    uv run python -m scripts.extract_card_template PCK_PATH [IMAGES_ROOT]
"""

from __future__ import annotations

import argparse
import logging
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.extract_images import decode_ctex

logger = logging.getLogger(__name__)

# Card frames are keyed by card *type*, tinted by the card pool's colour; both
# lists come from the decompiled source rather than being guessed:
#   CardModel.cs:182  -> card_frame_<type>_s.tres
#   *CardPool.cs      -> CardFrameMaterialPath => "card_frame_<colour>"
FRAME_TYPES = ("attack", "skill", "power", "ancient", "quest")
FRAME_COLORS = ("red", "green", "blue", "pink", "orange", "colorless", "curse", "quest")
# Colours a later build added (v0.113.0's Fool: `card_frame_yellow`). Baked only
# when the pck has the material, so an older pck extracts exactly as before.
LATER_FRAME_COLORS = ("yellow",)

# Banners are keyed by card *rarity* (CardModel.cs:275-282). "common" is the
# fallback arm, so it also covers None/Basic/Token.
BANNER_RARITIES = (
    "common",
    "uncommon",
    "rare",
    "curse",
    "status",
    "event",
    "quest",
    "ancient",
)

PORTRAIT_BORDER_TYPES = ("attack", "skill", "power")
ENERGY_CHARACTERS = (
    "ironclad",
    "silent",
    "defect",
    "regent",
    "necrobinder",
    "colorless",
    "quest",
)
# Characters a later build added (v0.113.0's Fool). `EnergyIconHelper.GetPath`
# tries the ui_atlas sprite, then `packed/energy_icons/energy_<prefix>.png` —
# the Fool ships only the latter. Skipped when the pck has neither.
LATER_ENERGY_CHARACTERS = ("fool",)

# Loose (non-atlas) textures card.tscn references directly, shipped untinted.
LOOSE_TEXTURES = {
    "energy_star.png": "images/ui/combat/energy_star.png",
}

# Loose textures that still get a per-rarity tint baked (see the banner block).
LOOSE_TEXTURES_RARITY = {
    "plaque": "images/ui/cards/card_portrait_border_plaque2.png",
}

_REGION_RE = re.compile(r"^region\s*=\s*Rect2\(([-\d.,\s]+)\)", re.M)
_ATLAS_RE = re.compile(r'path="res://([^"]+\.png)"')
_MARGIN_RE = re.compile(r"^margin\s*=\s*Rect2\(([-\d.,\s]+)\)", re.M)
_SHADER_PARAM_RE = re.compile(r"^shader_parameter/([hsv])\s*=\s*([-\d.]+)", re.M)
_IMPORT_PATH_RE = re.compile(r'^path(?:\.\w+)?="res://([^"]+)"', re.M)


class Pck:
    """Random-access reader over a Godot 4 ``.pck``."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("rb")
        self._index: dict[str, tuple[int, int]] = {}
        self._read_index()

    def has(self, res_path: str) -> bool:
        """Whether the pck carries ``res_path``."""
        return res_path.removeprefix("res://") in self._index

    def _read_index(self) -> None:
        f = self._fh
        magic = f.read(4)
        if magic != b"GDPC":
            raise ValueError(f"not a Godot pck (magic {magic!r})")
        fmt_version = struct.unpack("<I", f.read(4))[0]
        f.read(12)  # engine major/minor/patch
        f.read(4)  # pack flags
        file_base = struct.unpack("<Q", f.read(8))[0]
        directory_offset = None
        if fmt_version >= 3:
            directory_offset = struct.unpack("<Q", f.read(8))[0]
        f.read(64)  # reserved
        if directory_offset is not None:
            f.seek(directory_offset)

        for _ in range(struct.unpack("<I", f.read(4))[0]):
            path_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(path_len).rstrip(b"\x00").decode("utf-8")
            offset, size = struct.unpack("<qq", f.read(16))
            f.read(16)  # md5
            f.read(4)  # flags
            self._index[name.removeprefix("res://")] = (file_base + offset, size)

    def read(self, res_path: str) -> bytes:
        entry = self._index.get(res_path.removeprefix("res://"))
        if entry is None:
            raise KeyError(f"{res_path} not in pck")
        offset, size = entry
        self._fh.seek(offset)
        return self._fh.read(size)

    def read_text(self, res_path: str) -> str:
        return self.read(res_path).decode("utf-8")

    def load_texture(self, res_path: str) -> Image.Image:
        """Load a ``.png`` resource, following its ``.import`` to the ``.ctex``."""
        remap = self.read_text(res_path + ".import")
        match = _IMPORT_PATH_RE.search(remap)
        if match is None:
            raise ValueError(f"no imported path in {res_path}.import")
        image = decode_ctex(self.read(match.group(1)))
        if image is None:
            raise ValueError(f"could not decode {match.group(1)}")
        return image

    def close(self) -> None:
        self._fh.close()


@dataclass(frozen=True)
class HsvParams:
    """The three uniforms of ``shaders/hsv.gdshader``."""

    h: float = 1.0
    s: float = 1.0
    v: float = 1.0


# YIQ basis exactly as spelled in shaders/hsv.gdshader. Godot's mat3(a, b, c)
# takes *column* vectors, and the shader's first multiply is `M * v` (column
# combination), so the rows below are the shader's columns transposed — i.e.
# the standard RGB->YIQ matrix.
_RGB_TO_YIQ = np.array(
    [
        [0.2989, 0.5870, 0.1140],
        [0.5959, -0.2774, -0.3216],
        [0.2115, -0.5229, 0.3114],
    ],
    dtype=np.float64,
)
_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ)


def apply_hsv(image: Image.Image, params: HsvParams) -> Image.Image:
    """Re-implement ``shaders/hsv.gdshader`` on the CPU.

    The shader rotates hue in YIQ space, scales the two chroma axes by ``s``,
    and multiplies through by ``v``. Alpha is untouched. ``v`` is allowed to
    exceed 1.0 by several materials (1.2, 1.14), so the result is clamped only
    at the very end — exactly where the GPU clamps when writing the target.
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:]

    yiq = rgb @ _RGB_TO_YIQ.T

    # `hue = 1.0 - h` then scaled to a full turn; h=1.0 is therefore identity.
    hue = (1.0 - params.h) * 6.283185
    cos_h, sin_h = np.cos(hue), np.sin(hue)
    y = yiq[..., 0]
    i = yiq[..., 1] * cos_h - yiq[..., 2] * sin_h
    q = yiq[..., 1] * sin_h + yiq[..., 2] * cos_h

    # Saturation scales I/Q only; value scales all three.
    yiq = np.stack([y, i * params.s, q * params.s], axis=-1) * params.v

    rgb = np.clip(yiq @ _YIQ_TO_RGB.T, 0.0, 1.0)
    out = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA")


def read_hsv_material(pck: Pck, res_path: str) -> HsvParams:
    """Parse the h/s/v uniforms out of a ShaderMaterial ``.tres``."""
    found = {k: float(v) for k, v in _SHADER_PARAM_RE.findall(pck.read_text(res_path))}
    return HsvParams(**found)


def read_atlas_sprite(pck: Pck, res_path: str, atlas_cache: dict[str, Image.Image]) -> Image.Image:
    """Crop one ``AtlasTexture`` ``.tres`` out of its backing atlas page.

    A trimmed sprite carries ``margin`` — the transparent border the packer
    removed. We pad it back so every sprite keeps the geometry card.tscn's
    offsets were authored against; dropping it shifts the art inside its box.

    ``margin`` is a ``Rect2`` and its two halves mean different things:
    *position* is where the trimmed region sits inside the full sprite, and
    *size* is the TOTAL width/height the trim removed — not a second
    bottom-right offset. Godot's own ``AtlasTexture::get_width()`` is
    ``region.size.x + margin.size.x``, and ``draw_rect`` places the region at
    ``margin.position``.

    Reading it as (left, top, right, bottom) over-pads every trimmed sprite by
    its own top-left margin, which shrinks the art inside its authored box: the
    title banner came out 656x193 instead of 655x170, so the card's cloth band
    was drawn ~14% short and the title sat visibly low on it.
    """
    text = pck.read_text(res_path)
    atlas_match = _ATLAS_RE.search(text)
    region_match = _REGION_RE.search(text)
    if atlas_match is None or region_match is None:
        raise ValueError(f"{res_path} is not an AtlasTexture with a region")

    atlas_path = atlas_match.group(1)
    if atlas_path not in atlas_cache:
        atlas_cache[atlas_path] = pck.load_texture(atlas_path)
    atlas = atlas_cache[atlas_path]

    x, y, w, h = (int(float(n)) for n in region_match.group(1).split(","))
    sprite = atlas.crop((x, y, x + w, y + h))

    margin_match = _MARGIN_RE.search(text)
    if margin_match is not None:
        mx, my, mw, mh = (int(float(n)) for n in margin_match.group(1).split(","))
        padded = Image.new("RGBA", (w + mw, h + mh), (0, 0, 0, 0))
        padded.paste(sprite, (mx, my))
        sprite = padded
    return sprite


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    logger.info("wrote %s (%dx%d)", path, image.width, image.height)


def extract(pck_path: Path, images_root: Path) -> int:
    pck = Pck(pck_path)
    atlas_cache: dict[str, Image.Image] = {}
    out = images_root / "card_template"
    written = 0

    def sprite(name: str) -> Image.Image:
        return read_atlas_sprite(
            pck, f"images/atlases/ui_atlas.sprites/card/{name}.tres", atlas_cache
        )

    # --- frames: one source per card type x one tint per character colour ---
    frame_mat = "materials/cards/frames/card_frame_{}_mat.tres"
    frame_mats = {
        color: read_hsv_material(pck, frame_mat.format(color))
        for color in (
            *FRAME_COLORS,
            *(c for c in LATER_FRAME_COLORS if pck.has(frame_mat.format(c))),
        )
    }
    for card_type in FRAME_TYPES:
        base = sprite(f"card_frame_{card_type}_s")
        for color, params in frame_mats.items():
            _save(apply_hsv(base, params), out / "frames" / f"{card_type}_{color}.png")
            written += 1

    # --- everything the rarity material tints ---
    # NCard.cs:1332-1356 puts Model.BannerMaterial on three nodes, not just the
    # banner: the title banner, the portrait border, and the type plaque. Bake a
    # rarity variant of each so the consumer never needs the shader.
    banner_mats = {
        rarity: read_hsv_material(pck, f"materials/cards/banners/card_banner_{rarity}_mat.tres")
        for rarity in BANNER_RARITIES
    }

    banner_base = sprite("card_banner")
    plaque_base = pck.load_texture(LOOSE_TEXTURES_RARITY["plaque"])
    border_bases = {
        card_type: sprite(f"card_portrait_border_{card_type}_s")
        for card_type in PORTRAIT_BORDER_TYPES
    }

    for rarity, params in banner_mats.items():
        _save(apply_hsv(banner_base, params), out / "banners" / f"{rarity}.png")
        _save(apply_hsv(plaque_base, params), out / "plaques" / f"{rarity}.png")
        written += 2
        for card_type, base in border_bases.items():
            _save(
                apply_hsv(base, params),
                out / "portrait_borders" / f"{card_type}_{rarity}.png",
            )
            written += 1

    # The Ancient banner is its own sprite, drawn untinted (card.tscn gives the
    # AncientBanner node no material).
    _save(sprite("ancient_banner"), out / "banners" / "ancient_sprite.png")
    written += 1

    # --- cost icons + misc chrome ---
    for character in ENERGY_CHARACTERS:
        _save(sprite(f"energy_{character}"), out / "energy" / f"{character}.png")
        written += 1
    for character in LATER_ENERGY_CHARACTERS:
        atlas = f"images/atlases/ui_atlas.sprites/card/energy_{character}.tres"
        packed = f"images/packed/energy_icons/energy_{character}.png"
        if pck.has(atlas):
            icon = sprite(f"energy_{character}")
        elif pck.has(packed + ".import"):
            icon = pck.load_texture(packed)
        else:
            continue
        _save(icon, out / "energy" / f"{character}.png")
        written += 1
    _save(sprite("card_unplayable_icon"), out / "unplayable.png")
    written += 1

    for filename, res_path in LOOSE_TEXTURES.items():
        _save(pck.load_texture(res_path), out / filename)
        written += 1

    pck.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pck_path", type=Path, help="Path to SlayTheSpire2.pck")
    parser.add_argument(
        "images_root",
        type=Path,
        nargs="?",
        default=Path("site/public/images"),
        help="Art root to write card_template/ into (default: site/public/images)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    count = extract(args.pck_path, args.images_root)
    print(f"Wrote {count} card-template files under {args.images_root / 'card_template'}")


if __name__ == "__main__":
    main()
