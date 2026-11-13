"""Derive the engine's card inventory (id -> rarity) from the Rust source.

`parity_fuzz_scenario`'s ``ALL_*_CARDS`` lists were hand-transcribed from the
Rust content crate, with a comment telling the next person to "regenerate them
when content lands". Nobody could, because there was nothing to run — so they
drifted, and the drift was invisible: every coverage test checked a list
against *itself*.

The damage was concrete. `ALL_COLORLESS_CARDS` was transcribed from the
colorless spec slice alone, so every **Event**- and **Ancient**-rarity card —
which live in `status_specs_generated.rs` — had never been driven against the
C# oracle at all. That was found the hard way, one card at a time, while
porting `HelloWorldPower` (GH #636) and `ReboundPower` (GH #637); the second
of those turned out to have a real engine bug the sweep caught the moment the
card was finally included. `NOT_YET` (an Ironclad Rare) and the Defect /
Necrobinder starter cards were missing for no reason at all.

This module is the missing extraction, and
`tests/test_parity_coverage.py::test_every_playable_engine_card_is_covered`
is the gate that makes it stick.

**Why two files, joined on `CardId`.** The rarity lives on the `CardSpec`
literals in `sts2-content`, but the name the harness keys on is
`CardId::to_python_name()` from `card_ids_generated.rs`, and the two do NOT
always agree: `CardId::PanacheCard` is `"PANACHE_CARD"` while its spec's
`name` field says `"PANACHE"`. Keying off the spec's `name` would have quietly
produced a card id neither engine accepts, so the join goes through `CardId`.

Source-parsing rather than importing `sts2_rs` is deliberate: the coverage
tests are pure-Python by design (no wheel, no C# subprocess), and the wheel
exposes no card inventory to bind against.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

#: Repo root — this file is `<root>/sts2-pycli/python/card_inventory.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_CARD_IDS_RS = _REPO_ROOT / "sts2-rs/crates/sts2-core/src/card_ids_generated.rs"
_CONTENT_SRC = _REPO_ROOT / "sts2-rs/crates/sts2-content/src"

#: `Self::Bash => "BASH"` inside `CardId::to_python_name`.
_PY_NAME = re.compile(r'Self::(\w+) => "([A-Z_0-9]+)"')

#: A `CardSpec { .. }` literal body, non-greedy to the closing brace at the
#: same indent. Card specs are `const fn` returns, so the closing brace is
#: always on its own line.
_SPEC_BODY = re.compile(r"CardSpec \{(.*?)\n\s*\}", re.S)
_SPEC_ID = re.compile(r"id: CardId::(\w+)")
_SPEC_RARITY = re.compile(r"rarity: CardRarity::(\w+)")

#: Rarities that are never a deck-holdable, coverage-sweepable card:
#:
#: * ``Status`` / ``Curse`` — unplayable. They enter a deck only as a
#:   punishment and the sweep's shape (put the card in a 5-card deck and play
#:   it) cannot exercise them. Their handling is covered by the encounter and
#:   relic axes instead.
#: * ``Token`` — never in a deck. Shiv / SweepingGaze / SovereignBlade and
#:   friends are minted mid-combat by the effect that needs them, so the card
#:   axis has no way to introduce one.
#: * ``Quest`` — event items (ByrdonisEgg / LanternKey / SpoilsMap), not
#:   combat cards.
#:
#: Everything else — including ``Basic`` starters, ``Event`` and ``Ancient`` —
#: is a card a player can hold and play, and belongs in the sweep.
UNSWEEPABLE_RARITIES = frozenset({"Status", "Curse", "Token", "Quest"})


@lru_cache(maxsize=1)
def card_id_python_names() -> dict[str, str]:
    """`CardId` variant -> the id string both engines key on."""
    text = _CARD_IDS_RS.read_text(encoding="utf-8")
    names = dict(_PY_NAME.findall(text))
    if not names:
        raise RuntimeError(f"no CardId python names found in {_CARD_IDS_RS}")
    return names


@lru_cache(maxsize=1)
def card_rarities() -> dict[str, str]:
    """`CardId` variant -> `CardRarity` variant, from the `CardSpec` literals.

    First spelling wins: a handful of cards are constructed in more than one
    place (upgrade helpers, per-character re-exports) and they agree on rarity
    — a disagreement would be a content bug, and
    `test_engine_card_rarities_are_unambiguous` fails on one.
    """
    out: dict[str, str] = {}
    for path in sorted(_CONTENT_SRC.rglob("*.rs")):
        if path.name == "tests.rs":
            continue
        for body in _SPEC_BODY.findall(path.read_text(encoding="utf-8")):
            card_id = _SPEC_ID.search(body)
            rarity = _SPEC_RARITY.search(body)
            if card_id and rarity:
                out.setdefault(card_id.group(1), rarity.group(1))
    if not out:
        raise RuntimeError(f"no CardSpec literals found under {_CONTENT_SRC}")
    return out


@lru_cache(maxsize=1)
def engine_cards() -> dict[str, str]:
    """Every card the engine defines: id string -> `CardRarity` variant."""
    names = card_id_python_names()
    return {names[cid]: rarity for cid, rarity in card_rarities().items() if cid in names}


_V0111_ONLY = re.compile(r"pub const V0111_ONLY: \[CardId; \d+\] = \[(.*?)\];", re.S)

#: Cards v0.111.0 removed from the game (GH #793): the engine keeps their
#: specs for v0.107.1, so they stay in `engine_cards`, but a v0.111.0 sweep
#: must not name them.
V0111_REMOVED_CARDS = frozenset({"SCARE"})


@lru_cache(maxsize=1)
def v0111_only_cards() -> frozenset[str]:
    """Id strings of `CardId::V0111_ONLY` — cards that exist only in v0.111.0."""
    text = _CARD_IDS_RS.read_text(encoding="utf-8")
    m = _V0111_ONLY.search(text)
    if m is None:
        raise RuntimeError(f"no CardId::V0111_ONLY list found in {_CARD_IDS_RS}")
    names = card_id_python_names()
    return frozenset(names[v] for v in re.findall(r"Self::(\w+)", m.group(1)))


_V0113_ONLY = re.compile(r"pub const V0113_ONLY: \[CardId; \d+\] = \[(.*?)\];", re.S)


@lru_cache(maxsize=1)
def v0113_only_cards() -> frozenset[str]:
    """Id strings of `CardId::V0113_ONLY` — cards that exist from v0.113.0 on."""
    text = _CARD_IDS_RS.read_text(encoding="utf-8")
    m = _V0113_ONLY.search(text)
    if m is None:
        raise RuntimeError(f"no CardId::V0113_ONLY list found in {_CARD_IDS_RS}")
    names = card_id_python_names()
    return frozenset(names[v] for v in re.findall(r"Self::(\w+)", m.group(1)))


def _version_tuple(game_version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in game_version.lstrip("v").split("."))


def has_v0111_cards(game_version: str) -> bool:
    """Whether `game_version` (``"v0.111.0"``-style) has v0.111.0's card set,
    i.e. is v0.111.0 or later."""
    return _version_tuple(game_version) >= (0, 111, 0)


def sweepable_cards(game_version: str = "v0.107.1") -> set[str]:
    """Card ids the coverage sweep's card axis is expected to cover.

    `game_version` picks the build: v0.107.1 (the default) leaves out the
    v0.111.0-only cards; v0.111.0 and later leave out the cards v0.111.0
    removed (Rust gates the same split on ``>= V0_111_0``). Builds before
    v0.113.0 also leave out the v0.113.0-only cards.
    """
    if has_v0111_cards(game_version):
        excluded = set(V0111_REMOVED_CARDS)
    else:
        excluded = set(v0111_only_cards())
    if _version_tuple(game_version) < (0, 113, 0):
        excluded |= v0113_only_cards()
    return {
        name
        for name, rarity in engine_cards().items()
        if rarity not in UNSWEEPABLE_RARITIES and name not in excluded
    }
