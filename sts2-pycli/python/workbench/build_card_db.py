"""Distil the wiki card manifest into the workbench's static ``card_db.json``.

Deck analysis must work on *every* screen — including card-reward / map views,
where the Rust backend ships deck cards without numeric ``stats``. So rather
than lean on runtime stats, the analyzer reads a committed static table keyed by
loc-key (which equals the runtime ``card.id``). This script regenerates that
table from the sibling wiki manifest (the same source ``localization.py`` already
trusts for card type / rarity).

Run from the repo root::

    python -m workbench.build_card_db [--version <ver>] [--manifest <path>] [--out <path>]

The output is checked in at ``workbench/data/card_db.json`` so the analyzer never
depends on the wiki sibling being present at runtime.

**One table per game build (GH #877).** The default build's table is
``workbench/data/card_db.json``; every other supported build's is
``workbench/data/<ver>/card_db.json`` (``deck_analysis.data_dir_for``), built
from that build's own wiki manifest (``sts2-wiki/data/<ver>/cards.json``) —
never another build's, which would score the cards the build changed with the
wrong numbers. ``--version`` picks the build; the default is the default build.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS

from workbench import deck_analysis, localization

# target strings the manifest uses that hit more than one enemy.
_AOE_TARGETS = {"AllEnemies"}

_TWICE = re.compile(r"\btwice\b", re.IGNORECASE)
_N_TIMES = re.compile(r"\b(\d+)\s+times\b", re.IGNORECASE)
_X_TIMES = re.compile(r"\bX\s+times\b", re.IGNORECASE)
# A power whose effect recurs every turn (Demon Form, Metallicize, …) — the
# deck simulator banks its Strength/Block each turn rather than once.
_PER_TURN = re.compile(r"(start|end) of your turn|each turn|every turn", re.IGNORECASE)


def _var(vars_: list[dict[str, Any]], kind: str, key: str = "base_value") -> int:
    """Return the named var's value (0 when absent)."""
    for v in vars_ or []:
        if v.get("type") == kind:
            return int(v.get(key, 0) or 0)
    return 0


def _hits(template: str, vars_: list[dict[str, Any]]) -> tuple[int, bool]:
    """Best-effort (hit_count, variable) from the card's description template.

    Multi-hit lives in prose (``"twice"``, ``"3 times"``) or a ``Repeat`` var
    (Sword Boomerang). Whirlwind-style ``"X times"`` scales with energy/another
    stat, so it is flagged variable and counted as a single baseline hit.
    """
    template = template or ""
    repeat = _var(vars_, "Repeat")
    if repeat:
        return repeat, False
    if _X_TIMES.search(template):
        return 1, True
    if _TWICE.search(template):
        return 2, False
    m = _N_TIMES.search(template)
    if m:
        return int(m.group(1)), False
    return 1, False


def build(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Build the ``loc_key -> stats`` table from a wiki ``cards.json`` manifest."""
    raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = raw.get("cards", []) if isinstance(raw, dict) else raw
    db: dict[str, dict[str, Any]] = {}
    for c in cards or []:
        key = c.get("loc_key")
        if not key:
            continue
        vars_ = c.get("vars") or []
        cost = c.get("energy_cost")
        cost_x = cost is None or int(cost) < 0
        hits, variable_hits = _hits(c.get("description_template", ""), vars_)
        db[key] = {
            "title": c.get("title") or key,
            "character": c.get("character"),
            "type": c.get("type"),
            "rarity": c.get("rarity"),
            "cost": 0 if cost_x else int(cost),
            "cost_x": cost_x,
            "target": c.get("target"),
            "aoe": c.get("target") in _AOE_TARGETS,
            "damage": _var(vars_, "Damage"),
            "damage_up": _var(vars_, "Damage", "upgraded_value"),
            "block": _var(vars_, "Block"),
            "block_up": _var(vars_, "Block", "upgraded_value"),
            # Velocity throughput: cards drawn ("Cards" var) and energy gained
            # ("Energy" var) when the card is played. Lets the analyzer quantify
            # velocity as expected cards/energy per turn rather than a card count.
            "draw": _var(vars_, "Cards"),
            "draw_up": _var(vars_, "Cards", "upgraded_value"),
            "energy": _var(vars_, "Energy"),
            "energy_up": _var(vars_, "Energy", "upgraded_value"),
            # Scaling vectors the deck simulator models: Strength/Dexterity gains,
            # and whether the gain recurs every turn (Demon Form vs Inflame).
            "strength": _var(vars_, "Strength"),
            "strength_up": _var(vars_, "Strength", "upgraded_value"),
            "dexterity": _var(vars_, "Dexterity"),
            "dexterity_up": _var(vars_, "Dexterity", "upgraded_value"),
            "per_turn": bool(_PER_TURN.search(c.get("description_template", "") or "")),
            # Keywords the deck simulator needs: ``exhaust`` cards leave play
            # (one-shot — never recycle), ``unplayable`` ones clog the hand.
            "exhaust": "Exhaust" in (c.get("keywords") or []),
            "innate": "Innate" in (c.get("keywords") or []),
            "ethereal": "Ethereal" in (c.get("keywords") or []),
            "unplayable": "Unplayable" in (c.get("keywords") or []),
            "hits": hits,
            "variable_hits": variable_hits,
            "refpow": [
                p.get("class_name") for p in c.get("referenced_powers") or [] if p.get("class_name")
            ],
            "desc": c.get("description_plain") or "",
        }
    return db


def _default_manifest(repo_root: Path, version: str | None = None) -> Path | None:
    """Locate the engine-pinned sibling wiki ``cards.json`` manifest, if any.

    Delegates to ``localization._engine_manifest`` so the generated table is
    keyed to the same wiki version the renderer localizes against
    (``_ENGINE_WIKI_VERSION``), not to whatever version happens to be highest on
    disk — the wiki carries older *and* newer reference data, and a card table
    built off a build the engine does not play mis-scores every deck.
    """
    data_dir = repo_root.parent / "sts2-wiki" / "data"
    if not data_dir.is_dir():
        return None
    return localization._engine_manifest(data_dir, version)


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--version",
        choices=SUPPORTED_GAME_VERSIONS,
        default=LEGACY_GAME_VERSION,
        help="game build whose table to write (default: the default build)",
    )
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.manifest is None:
        args.manifest = _default_manifest(repo_root, args.version)
    if args.out is None:
        args.out = deck_analysis.data_dir_for(args.version) / "card_db.json"
    if not args.manifest or not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest!r} (pass --manifest)")
    db = build(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(db, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(db)} cards -> {args.out}")


if __name__ == "__main__":
    main()
