#!/usr/bin/env python3
"""Extract all relic data from STS2 decompiled code + localization."""

import argparse
import os
import re

from scripts.common import (
    class_name_to_loc_key,
    find_loc_key,
    load_localization,
    parse_canonical_vars,
    read_cs_files,
    strip_rich_text,
    write_json,
)
from scripts.extract_cards import render_resolved

POOL_MAP: dict[str, str] = {
    "IroncladRelicPool": "Ironclad",
    "SilentRelicPool": "Silent",
    "DefectRelicPool": "Defect",
    "NecrobinderRelicPool": "Necrobinder",
    "RegentRelicPool": "Regent",
    # v0.113.0's new character; an older decomp simply has no such pool.
    "FoolRelicPool": "Fool",
    "SharedRelicPool": "Shared",
    "EventRelicPool": "Event",
    "DeprecatedRelicPool": "Deprecated",
    "FallbackRelicPool": "Fallback",
}


def build_relic_pool_map(decompiled_dir: str) -> dict[str, str]:
    """Parse RelicPool files to build class_name -> pool mapping."""
    pools_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Models.RelicPools")
    relic_to_pool: dict[str, str] = {}

    if not os.path.exists(pools_dir):
        return relic_to_pool

    for pool_name, content in read_cs_files(pools_dir):
        pool_label = POOL_MAP.get(pool_name, pool_name)
        for m in re.finditer(r"ModelDb\.Relic<(\w+)>\(\)", content):
            relic_class = m.group(1)
            # Character-specific pools override shared
            if relic_class not in relic_to_pool or pool_label != "Shared":
                relic_to_pool[relic_class] = pool_label

    return relic_to_pool


def parse_relic_file(class_name: str, content: str) -> dict | None:
    """Parse a decompiled relic .cs file."""
    if ": RelicModel" not in content:
        return None

    relic: dict = {"class_name": class_name}

    # Rarity
    m = re.search(r"RelicRarity\.(\w+)", content)
    if m:
        relic["rarity"] = m.group(1)

    # Dynamic vars
    vars_found = parse_canonical_vars(content)
    if vars_found:
        relic["vars"] = vars_found

    # Image filename: PascalCase -> snake_case
    image_name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", class_name)
    image_name = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", image_name)
    relic["image"] = image_name.lower()

    return relic


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract STS2 relic data")
    parser.add_argument("decompiled_dir", help="Path to decompiled source directory")
    parser.add_argument("loc_dir", help="Path to localization directory")
    parser.add_argument("output_dir", help="Path to output data directory")
    args = parser.parse_args()

    decompiled_dir = os.path.expanduser(args.decompiled_dir)
    loc_dir = os.path.expanduser(args.loc_dir)
    output_dir = os.path.expanduser(args.output_dir)

    loc_data = load_localization(loc_dir, "relics")
    pool_map = build_relic_pool_map(decompiled_dir)

    # Build acquisition map: which ancients/events offer each relic
    relic_sources: dict[str, list[str]] = {}
    events_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Models.Events")
    if os.path.exists(events_dir):
        for ev_name, ev_content in read_cs_files(events_dir):
            # Find RelicOption<ClassName>() and ModelDb.Relic<ClassName>()
            for m in re.finditer(r"(?:RelicOption|ModelDb\.Relic)<(\w+)>", ev_content):
                relic_class = m.group(1)
                if relic_class not in relic_sources:
                    relic_sources[relic_class] = []
                if ev_name not in relic_sources[relic_class]:
                    relic_sources[relic_class].append(ev_name)

    relics_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Models.Relics")
    relics: list[dict] = []

    for class_name, content in read_cs_files(relics_dir):
        relic = parse_relic_file(class_name, content)
        if not relic:
            continue

        # Pool assignment
        relic["pool"] = pool_map.get(class_name, "Unknown")

        # Localization
        loc_key = find_loc_key(class_name, loc_data)
        if loc_key:
            relic["loc_key"] = loc_key
            relic["title"] = loc_data.get(f"{loc_key}.title", class_name)
            raw_desc = loc_data.get(f"{loc_key}.description", "")
            relic["description"] = strip_rich_text(raw_desc)
            relic["description_resolved"] = render_resolved(raw_desc, relic.get("vars") or [])
            relic["flavor"] = loc_data.get(f"{loc_key}.flavor", "")
        else:
            relic["loc_key"] = class_name_to_loc_key(class_name)
            relic["title"] = class_name
            relic["description"] = ""
            relic["description_resolved"] = ""
            relic["flavor"] = ""

        # Skip deprecated
        if relic["pool"] == "Deprecated":
            continue

        # Acquisition sources
        sources = relic_sources.get(class_name, [])
        if sources:
            relic["sources"] = sources

        # Character lock
        if relic["pool"] in ("Ironclad", "Silent", "Defect", "Necrobinder", "Regent", "Fool"):
            relic["character"] = relic["pool"]

        relics.append(relic)

    output_path = os.path.join(output_dir, "relics.json")
    write_json(output_path, relics)

    print(f"Extracted {len(relics)} relics to {output_path}")
    by_rarity: dict[str, int] = {}
    for r in relics:
        rar = r.get("rarity", "Unknown")
        by_rarity[rar] = by_rarity.get(rar, 0) + 1
    for rar, count in sorted(by_rarity.items()):
        print(f"  {rar}: {count}")


if __name__ == "__main__":
    main()
