#!/usr/bin/env python3
"""Enumerate every `Flash()` call site in the decompiled power classes.

Design spec: `docs/superpowers/specs/2026-08-29-workbench-vfx-creature-events-design.md`
§1.2 and §1.5.

C# `PowerModel.Flash()` (`PowerModel.cs:452`) raises the `Flashed` event,
which `NCreature.OnPowerFlashed` (`NCreature.cs:925`) turns into a
`power_flash_vfx` on the power's owner. Each call site is one place the Rust
engine must record `EventLogEntry::PowerFlashed`.

The output is a LEDGER, not a target. Its purpose is that an unported site is
counted BY NAME rather than silently absent — the same job
`vfx_bindings.json` does for the card/move binding table, and the reason that
table caught a missing `VineShambler` site during sub-project 2.

Two things are deliberately NOT collected:

* **`RelicModel.Flash`** (`RelicModel.cs:459,464`) is a different path — it
  flashes the RELIC's own UI through `NRelic`, never `NPowerFlashVfx`, and a
  relic is not a creature power.
* **Any `Flash()` outside `MegaCrit.Sts2.Core.Models.Powers/`** — e.g.
  `NHandCardHolder.Flash` (`NHandCardHolder.cs:712`), which flashes a card in
  hand.

The power id written to each row is `PowerKind::python_name()`'s form
(`SLITHERING_STRANGLER`), derived from the class stem by CamelCase splitting.
That conversion is exact for all 89 flashing classes — verified against
`power_ids_generated.rs`, and re-verified by
`test_every_power_id_is_a_real_power_kind` — so this module carries no alias
map. If a future power needs one, add it explicitly rather than loosening the
conversion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

#: The decomp namespace directory holding one file per power class.
POWERS_DIR = "MegaCrit.Sts2.Core.Models.Powers"

#: A bare `Flash();` statement. Anchored to the line start (with leading
#: indentation) so a call inside a longer expression -- or a `Flash()` on some
#: other object -- does not match.
_FLASH = re.compile(r"^\s*Flash\(\);", re.M)

#: Matches the point BEFORE each interior capital, for CamelCase splitting.
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def power_id(class_stem: str) -> str:
    """`"SlitheringStranglerPower"` -> `"SLITHERING_STRANGLER"`.

    The form `PowerKind::python_name()` returns, which is what the Rust
    coverage gate joins on.
    """
    name = class_stem.removesuffix("Power")
    return _CAMEL_BOUNDARY.sub("_", name).upper()


def collect(decomp_root: Path) -> list[dict[str, Any]]:
    """Every `Flash();` call, sorted by class then line.

    Args:
        decomp_root: The versioned decomp root, e.g. `decompiled/v0.107.1`.

    Returns:
        One row per call site: ``power``, ``cs_class``, ``cs_line``, plus the
        ledger fields ``ported`` (False until a Rust site is placed), ``rust``
        (where it was placed) and ``reason`` (why it could not be).
    """
    powers_dir = decomp_root / POWERS_DIR
    if not powers_dir.is_dir():
        raise FileNotFoundError(f"no {POWERS_DIR} under {decomp_root}")

    sites: list[dict[str, Any]] = []
    for cs in sorted(powers_dir.glob("*Power.cs"), key=lambda p: p.name):
        text = cs.read_text(encoding="utf-8", errors="replace")
        for m in _FLASH.finditer(text):
            sites.append(
                {
                    "power": power_id(cs.stem),
                    "cs_class": cs.stem,
                    "cs_line": text.count("\n", 0, m.start()) + 1,
                    "ported": False,
                    "rust": None,
                    "reason": None,
                }
            )
    return sites


def merge(fresh: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry the ledger fields of `existing` onto a fresh extraction.

    Re-extracting must not silently un-port every site that has already
    landed, and must not preserve a `ported` mark for a site the C# no longer
    has -- a site that disappeared is news, and it disappears from the output
    entirely so the Rust gate's count moves.
    """
    by_key = {(r["cs_class"], r["cs_line"]): r for r in existing}
    for row in fresh:
        prior = by_key.get((row["cs_class"], row["cs_line"]))
        if prior is not None:
            row["ported"] = prior.get("ported", False)
            row["rust"] = prior.get("rust")
            row["reason"] = prior.get("reason")
    return fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decomp_root", type=Path, help="e.g. decompiled/v0.107.1")
    parser.add_argument("out", type=Path, help="e.g. data/v0.107.1/power_flash_sites.json")
    args = parser.parse_args(argv)

    sites = collect(args.decomp_root)
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        sites = merge(sites, prior.get("sites", []))

    payload = {
        "version": args.decomp_root.name,
        "sites": sites,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # `ensure_ascii=False`: the reasons are prose and carry em dashes and
    # backticked identifiers. Escaping them to `\u2014` makes the committed
    # table unreadable in review AND unstable — a hand edit that writes the
    # literal character gets flipped back on the next extraction, churning
    # rows nobody touched. The file is written UTF-8 either way.
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ported = sum(1 for s in sites if s["ported"])
    print(f"{len(sites)} flash sites, {ported} ported -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
