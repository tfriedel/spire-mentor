"""Generate web/public/card_text.json from the wiki cards.json.

Reads the wiki's per-build ``cards.json`` manifests and produces a
``{loc_key: {title, description}}`` mapping using the pre-resolved
`description_plain` field.  Residual [tag]...[/tag] markup is stripped.
`\n` newlines are preserved for display.

The **engine-pinned** build (``workbench.localization._ENGINE_WIKI_VERSION``)
wins; older manifests only supply cards that build dropped, so the catalog can
still name a retired card (FOLLOW_THROUGH) without describing a live one with
the wrong build's text. It used to read v0.103.2 alone, which is how the catalog
told you Pact's End "Can only be played if you have 3 or more cards in your
Exhaust Pile" — a rule v0.107.1 replaced with an on-play check. 57 cards were
affected; see bug report ``wb-66e36e372eef4bea``.

**One table per game build (GH #877).** ``--version`` picks the build whose
text wins (default: the default build). The default build's table is
``web/public/card_text.json``; every other supported build's is
``web/public/<ver>/card_text.json`` (what ``web/src/catalog/main.ts`` fetches
for that build), so a v0.111.0 card is described with v0.111.0 text.

Usage::

    PYTHONPATH=python uv run python python/synergy/gen_card_text.py [--version VER]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS
from workbench.localization import (
    _ENGINE_WIKI_VERSION,
    _KEYWORDS_AFTER_DESCRIPTION,
    _KEYWORDS_BEFORE_DESCRIPTION,
    engine_wiki_version,
)

_TAG_RE = re.compile(r"\[/?[^\]]+\]")
_VER_NUMS = re.compile(r"\d+")

_DEFAULT_WIKI_DATA = Path(__file__).parent.parent.parent.parent / "sts2-wiki" / "data"
_PUBLIC = Path(__file__).parent.parent.parent / "web" / "public"
_DEFAULT_OUT = _PUBLIC / "card_text.json"


def out_path_for(version: str | None = None) -> Path:
    """``web/public/card_text.json`` for the default build, else
    ``web/public/<ver>/card_text.json``."""
    key = version or LEGACY_GAME_VERSION
    return _DEFAULT_OUT if key == LEGACY_GAME_VERSION else _PUBLIC / key / "card_text.json"


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text)


def _entry(card: dict) -> dict[str, str]:
    raw_desc = card.get("description_plain") or card.get("description_template") or ""
    # The game prints a card's keywords ON its face, above or below the
    # description (C# ``CardModel.GetDescriptionForPile``). ``description_plain``
    # never carries them, so a Tremble read "Apply 3 Vulnerable." with no
    # "Exhaust." — same omission the workbench card face had.
    have = set(card.get("keywords") or [])
    before = [f"{k}." for k in reversed(_KEYWORDS_BEFORE_DESCRIPTION) if k in have]
    after = [f"{k}." for k in _KEYWORDS_AFTER_DESCRIPTION if k in have]
    body = _strip_tags(raw_desc)
    lines = before + ([body] if body else []) + after
    return {
        "title": card.get("title", card["loc_key"]),
        "description": "\n".join(lines),
    }


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(n) for n in _VER_NUMS.findall(name))


def _manifests(wiki_data: Path, version: str | None = None) -> list[Path]:
    """Every ``<ver>/cards.json`` up to build ``version``'s, ``version`` LAST.

    Callers merge in this order, so the pinned build overwrites whatever the
    older ones said and the older ones survive only where it is silent.
    Builds NEWER than the pinned one are skipped: their cards do not exist in
    that build (the default build's table must not describe a v0.111.0 card,
    GH #786), and merging them would describe cards the catalog cannot show.
    When the pinned build has no manifest on disk every build is used, as
    before.
    """
    pinned_ver = engine_wiki_version(version) if version else _ENGINE_WIKI_VERSION
    found = sorted(wiki_data.glob("*/cards.json"), key=lambda p: _version_key(p.parent.name))
    pinned = [p for p in found if p.parent.name == pinned_ver]
    if pinned:
        ceiling = _version_key(pinned_ver)
        found = [p for p in found if _version_key(p.parent.name) <= ceiling]
    return [p for p in found if p not in pinned] + pinned


def generate(wiki_data: Path, out_path: Path, version: str | None = None) -> None:
    result: dict[str, dict] = {}
    for manifest in _manifests(wiki_data, version):
        cards = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(cards, dict):
            cards = cards.get("cards", [])
        for card in cards or []:
            if card.get("loc_key"):
                result[card["loc_key"]] = _entry(card)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(result)} entries to {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate card_text.json from wiki.")
    parser.add_argument(
        "--wiki-data",
        type=Path,
        default=_DEFAULT_WIKI_DATA,
        help="Path to the wiki data/ dir holding versioned <ver>/cards.json manifests",
    )
    parser.add_argument(
        "--version",
        choices=SUPPORTED_GAME_VERSIONS,
        default=LEGACY_GAME_VERSION,
        help="Game build whose text wins (default: the default build)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output card_text.json path (default: per build)"
    )
    args = parser.parse_args(argv)
    generate(args.wiki_data, args.out or out_path_for(args.version), args.version)


if __name__ == "__main__":
    main()
