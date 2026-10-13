#!/usr/bin/env python3
"""How complete is the locally extracted art and sound?

The committed asset manifest (sts2-rs/crates/sts2-workbench/data/game/
asset_manifest.json) lists every image and sound file the maintainers' art
tree holds — the set the workbench was built and tested against. This
compares a local tree (what `just import-game` produced) with it:

    uv run --no-project scripts/check_art.py            # coverage summary
    uv run --no-project scripts/check_art.py --list     # + every missing path
    uv run --no-project scripts/check_art.py --strict   # exit 1 on unexpected gaps

A few manifest files cannot be produced from the current game build at all:
art for content that was removed or renamed before the pinned version but
that the maintainers' tree still carries from an older extraction. Those are
listed in LEGACY_ONLY below and reported separately; anything else missing is
a gap in the import pipeline.

Maintainers with a reference tree can compare the two file by file:

    uv run --with pillow --with numpy --no-project scripts/check_art.py \\
        --reference ../sts2-sim-private/sts2-wiki/site/public

which classifies every file present in both as byte-identical, pixel-identical,
near-identical (same size, mean pixel difference <= NEAR_IDENTICAL: texture
compression noise) or different (for an image, usually the game having
redrawn it since the reference was extracted, or a fresh portrait render).
Audio renders are seeded, so two imports on the same FMOD build give
byte-identical files; against a tree rendered before the seed (or on another
FMOD build) they differ, because FMOD picks a different take of each layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "sts2-rs/crates/sts2-workbench/data/game/asset_manifest.json"
PUBLIC = ROOT / "sts2-wiki/site/public"

# --reference: mean absolute RGBA difference (0-255) up to which two images of
# the same size count as the same art.
NEAR_IDENTICAL = 2.0

# Manifest entries that are not art.
NOT_ART = {".gitattributes"}

# Art for content no longer in v0.107.1: the game removed or renamed it, so its
# pck has no image to extract. The engine still implements some of it (cards
# and powers from v0.103.2), so the workbench can ask for it; it then draws
# the neutral placeholder tile.
LEGACY_ONLY = {
    "boss_icons/doormaker_boss_icon.png": "Doormaker boss, removed after v0.103.2",
    "card_atlas/ironclad/grapple.png": "Grapple card, removed",
    "card_atlas/silent/follow_through.png": "Follow Through card, removed after v0.103.2",
    "card_atlas/silent/prepare.png": "Prepare card, renamed Prepared (prepared.png)",
    "power_atlas/blade_of_ink_power.png": "Blade of Ink power, removed (the card remains)",
    "power_atlas/blade_of_ink_power_plus_power.png": "Blade of Ink+ power, removed",
    "power_atlas/disperse_power.png": "Disperse power, removed",
    "power_atlas/drum_of_battle_power.png": "Drum of Battle power, removed after v0.103.2",
    "power_atlas/eradication_mode_power.png": "Eradication Mode power, removed",
    "power_atlas/grapple_power.png": "Grapple power, removed",
    "power_atlas/grasp_power.png": "Grasp power, removed after v0.103.2",
    "power_atlas/hunger_of_the_void_power.png": "Hunger of the Void power, removed",
    "power_atlas/hunger_power.png": "Hunger power, removed after v0.103.2",
    "power_atlas/invoke_power.png": "Invoke power, removed (the card remains)",
    "power_atlas/metal_power.png": "Metal power, removed",
    "power_atlas/necrosurge_power.png": "Necrosurge power, removed",
    "power_atlas/scrutiny_power.png": "Scrutiny power, removed after v0.103.2",
}


def local_path(public: Path, rel: str) -> Path:
    """Manifest paths are relative to images/, except the audio/ subtree."""
    return public / rel if rel.startswith("audio/") else public / "images" / rel


def group(rel: str) -> str:
    parts = rel.split("/")
    return "/".join(parts[:2]) if parts[0] == "audio" else parts[0]


def coverage(public: Path) -> tuple[list[str], list[str], Counter[str], Counter[str]]:
    """-> (unexpected missing, legacy missing, present per group, total per group)."""
    manifest = [r for r in json.loads(MANIFEST.read_text(encoding="utf-8")) if r not in NOT_ART]
    missing, legacy = [], []
    present: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for rel in manifest:
        total[group(rel)] += 1
        if local_path(public, rel).is_file():
            present[group(rel)] += 1
        elif rel in LEGACY_ONLY:
            legacy.append(rel)
        else:
            missing.append(rel)
    return missing, legacy, present, total


def summary(public: Path = PUBLIC, list_missing: bool = False) -> list[str]:
    """Report lines; the first one is the headline."""
    if not MANIFEST.is_file():
        return [f"art coverage: no manifest at {MANIFEST.relative_to(ROOT)}"]
    missing, legacy, present, total = coverage(public)
    have, want = sum(present.values()), sum(total.values())
    lines = [
        f"art coverage: {have}/{want} manifest files present; {len(legacy)} exist only in older game "
        f"builds; {len(missing)} missing"
    ]
    gaps = {g: total[g] - present[g] for g in total if total[g] != present[g]}
    for g, n in sorted(gaps.items()):
        lines.append(f"  {g}: {present[g]}/{total[g]}")
    if list_missing:
        lines += [f"  missing: {r}" for r in missing]
        lines += [f"  older builds only: {r} ({LEGACY_ONLY[r]})" for r in legacy]
    elif missing:
        lines.append("  (--list prints every missing path)")
    return lines


def compare(public: Path, reference: Path) -> list[str]:
    """Classify files present in both trees: identical / same pixels / different."""
    import numpy as np
    from PIL import Image

    manifest = [r for r in json.loads(MANIFEST.read_text(encoding="utf-8")) if r not in NOT_ART]
    kinds: dict[str, Counter[str]] = {}
    differing: list[str] = []
    for rel in manifest:
        a, b = local_path(public, rel), local_path(reference, rel)
        if not (a.is_file() and b.is_file()):
            continue
        c = kinds.setdefault(group(rel), Counter())
        if a.read_bytes() == b.read_bytes():
            c["identical"] += 1
            continue
        if rel.endswith(".png"):
            ia, ib = Image.open(a).convert("RGBA"), Image.open(b).convert("RGBA")
            if ia.size == ib.size:
                diff = np.abs(np.asarray(ia, np.int16) - np.asarray(ib, np.int16))
                if not diff.any():
                    c["same pixels"] += 1
                    continue
                # Texture-compression noise (the same art decoded from a
                # re-encoded atlas) stays far below this; redrawn art does not.
                if diff.mean() <= NEAR_IDENTICAL:
                    c["near-identical"] += 1
                    continue
            differing.append(f"{rel} {ia.size[0]}x{ia.size[1]} vs {ib.size[0]}x{ib.size[1]}")
        else:
            differing.append(rel)
        c["different"] += 1
    lines = [f"compared with {reference}:"]
    for g, c in sorted(kinds.items()):
        lines.append(f"  {g}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    lines += [f"  different: {d}" for d in differing]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--public", type=Path, default=PUBLIC, help="the tree holding images/ and audio/")
    ap.add_argument("--list", action="store_true", help="print every missing path")
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything outside LEGACY_ONLY is missing")
    ap.add_argument("--reference", type=Path, help="another public/ tree to compare file by file")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    print("\n".join(summary(args.public, args.list)))
    if args.reference:
        print("\n".join(compare(args.public, args.reference)))
    return 1 if args.strict and coverage(args.public)[0] else 0


if __name__ == "__main__":
    sys.exit(main())
