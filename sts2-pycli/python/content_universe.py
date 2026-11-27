"""Every content id a game build has, per kind, read from the Rust source.

The parity harness keeps hand-written id lists (``parity_fuzz_scenario``'s
``ALL_RELICS``, ``ALL_POTIONS``, ``ALL_ENCOUNTERS``, ``EVENT_RELICS``,
``event_parity.CONTEXT_DEPENDENT_EVENTS``, ...). Each was right on the day it
was written and nothing noticed when a game build added content: ``ALL_RELICS``
stayed at its 110 v0.103.2 relics while v0.111.0 had 299, and the v0.113.0
relics reached none of the lists (GH #1500). This module is the universe those
lists are checked against, per build, so a new id turns ``just gate`` red.

Like `card_inventory`, it parses the Rust source instead of importing
``sts2_rs``: the checks are pure Python (no wheel, no C#), and the wheel's
registry listings are the PPO vocabularies, which deliberately leave the newer
builds' ids out. The build split mirrors the Rust ``all_for(version)`` helpers:

* relics   — ``RelicId::all()`` + ``V0111_ONLY`` (v0.111.0+) + ``V0113_ONLY``
  (v0.113.0+) in ``sts2-core/src/relic_ids_generated.rs``;
* potions  — ``PotionKind::ALL`` + ``V0111_ONLY`` + ``V0113_ONLY`` in
  ``sts2-core/src/potions.rs``;
* events   — ``EventId::all()``, without ``V0113_ONLY`` before v0.113.0, in
  ``sts2-core/src/events.rs``;
* encounters — every ``EncounterDef`` ``id:`` under
  ``sts2-content/src/encounters/`` (not build-split in Rust).

`tests/test_content_universe.py` checks these against the C# decompilation's
concrete model classes, so the universe itself cannot go stale either.
"""

from __future__ import annotations

import re
from functools import cache, lru_cache
from pathlib import Path

#: Repo root — this file is `<root>/sts2-pycli/python/content_universe.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_SRC = _REPO_ROOT / "sts2-rs/crates/sts2-core/src"
_RELIC_IDS_RS = _CORE_SRC / "relic_ids_generated.rs"
_POTIONS_RS = _CORE_SRC / "potions.rs"
_EVENTS_RS = _CORE_SRC / "events.rs"
_ENCOUNTERS_DIR = _REPO_ROOT / "sts2-rs/crates/sts2-content/src/encounters"
_DECOMP_ROOT = _REPO_ROOT / "sts2-wiki/decompiled"

#: The builds the checks cover: the per-PR gate's builds plus the legacy one.
BUILDS: tuple[str, ...] = ("v0.107.1", "v0.111.0", "v0.113.0", "v0.114.0")

#: Decomp tree of each build. v0.113.0 runs as the v0.113.1 hotfix, whose
#: decompilation replaced v0.113.0's (repo CLAUDE.md).
DECOMP_DIR: dict[str, str] = {
    "v0.107.1": "v0.107.1",
    "v0.111.0": "v0.111.0",
    "v0.113.0": "v0.113.1",
    "v0.114.0": "v0.114.0",
}


def version_tuple(game_version: str) -> tuple[int, ...]:
    """``"v0.111.0"`` -> ``(0, 111, 0)``."""
    return tuple(int(x) for x in game_version.lstrip("v").split("."))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _block(text: str, header: str, path: Path) -> str:
    """The body of the first array literal (``= [`` or ``&[``) after ``header``.

    Skips the ``[RelicId; 2]`` of a type annotation, which is not the list.
    """
    at = text.find(header)
    if at < 0:
        raise RuntimeError(f"{header!r} not found in {path}")
    lit = re.compile(r"(?:=|&)\s*\[").search(text, at + len(header))
    if lit is None:
        raise RuntimeError(f"no array literal after {header!r} in {path}")
    close = text.index("]", lit.end())
    return text[lit.end() : close]


def _variants(body: str, prefix: str) -> list[str]:
    return re.findall(rf"{prefix}::(\w+)", body)


def _names(text: str, fn_header: str, prefix: str, path: Path) -> dict[str, str]:
    """``variant -> python name`` from the ``match self`` arms after ``fn_header``."""
    at = text.find(fn_header)
    if at < 0:
        raise RuntimeError(f"{fn_header!r} not found in {path}")
    body = text[at : text.index("\n    }\n", at)]
    out = dict(re.findall(rf'(?:{prefix}|Self)::(\w+) => "([A-Za-z0-9_]+)"', body))
    if not out:
        raise RuntimeError(f"no {prefix} python-name arms after {fn_header!r} in {path}")
    return out


def _split_universe(
    base: list[str], v0111: list[str], v0113: list[str], names: dict[str, str], version: str
) -> list[str]:
    variants = list(base)
    if version_tuple(version) >= (0, 111, 0):
        variants += v0111
    if version_tuple(version) >= (0, 113, 0):
        variants += v0113
    return [names[v] for v in variants]


@cache
def relics(version: str) -> tuple[str, ...]:
    """Every relic id of ``version``, in ``RelicId::all_for`` order."""
    text = _read(_RELIC_IDS_RS)
    names = _names(text, "pub const fn to_python_name", "RelicId", _RELIC_IDS_RS)
    base = _variants(
        _block(text, "pub const fn all() -> &'static [RelicId]", _RELIC_IDS_RS), "Self"
    )
    v0111 = _variants(_block(text, "pub const V0111_ONLY:", _RELIC_IDS_RS), "Self")
    v0113 = _variants(_block(text, "pub const V0113_ONLY:", _RELIC_IDS_RS), "Self")
    return tuple(_split_universe(base, v0111, v0113, names, version))


@cache
def potions(version: str) -> tuple[str, ...]:
    """Every potion id of ``version``: ``PotionKind::ALL`` plus the build's additions."""
    text = _read(_POTIONS_RS)
    names = _names(text, "pub const fn python_name", "PotionKind", _POTIONS_RS)
    base = _variants(_block(text, "pub const ALL:", _POTIONS_RS), "PotionKind")
    v0111 = _variants(_block(text, "pub const V0111_ONLY:", _POTIONS_RS), "PotionKind")
    v0113 = _variants(_block(text, "pub const V0113_ONLY:", _POTIONS_RS), "PotionKind")
    return tuple(_split_universe(base, v0111, v0113, names, version))


@lru_cache(maxsize=1)
def potion_character_locks() -> dict[str, str]:
    """Class potions: ``python name -> character`` (``PotionKind::character_lock``).

    A character can only drink its own class potions (Rust gates
    ``Action::DrinkPotion`` on it, C# draws them from the class pool), so a
    loadout must not graft another class's potion.
    """
    text = _read(_POTIONS_RS)
    names = _names(text, "pub const fn python_name", "PotionKind", _POTIONS_RS)
    at = text.find("pub const fn character_lock")
    if at < 0:
        raise RuntimeError(f"no character_lock in {_POTIONS_RS}")
    body = text[at : text.index("\n    }\n", at)]
    out: dict[str, str] = {}
    for arm in re.finditer(
        r"((?:\|?\s*PotionKind::\w+\s*)+)=>\s*\{?\s*Some\(CharacterId::(\w+)\)", body
    ):
        for variant in re.findall(r"PotionKind::(\w+)", arm.group(1)):
            out[names[variant]] = arm.group(2)
    if not out:
        raise RuntimeError(f"no character_lock arms parsed in {_POTIONS_RS}")
    return out


@cache
def events(version: str) -> tuple[str, ...]:
    """Every event id (``EventId`` python name, e.g. ``"AbyssalBaths"``) of ``version``."""
    text = _read(_EVENTS_RS)
    names = _names(text, "pub const fn python_name", "EventId", _EVENTS_RS)
    every = _variants(_block(text, "pub const fn all() -> &'static [EventId]", _EVENTS_RS), "Self")
    v0113 = set(_variants(_block(text, "pub const V0113_ONLY:", _EVENTS_RS), "Self"))
    keep = [v for v in every if version_tuple(version) >= (0, 113, 0) or v not in v0113]
    return tuple(names[v] for v in keep)


_ENCOUNTER_ID = re.compile(r'^\s*id: "([A-Z0-9_]+)"', re.M)


@cache
def encounters() -> tuple[str, ...]:
    """Every ``EncounterDef`` id the engine defines (all builds), sorted."""
    ids: set[str] = set()
    for path in sorted(_ENCOUNTERS_DIR.glob("*.rs")):
        ids.update(_ENCOUNTER_ID.findall(_read(path)))
    if not ids:
        raise RuntimeError(f"no EncounterDef ids under {_ENCOUNTERS_DIR}")
    return tuple(sorted(ids))


# ─── the C# decompilation ────────────────────────────────────────────────────

_CAMEL = re.compile(r"(?<=[A-Za-z0-9])(?=[A-Z])")


def slugify(class_name: str) -> str:
    """C# ``StringHelper.Slugify``: ``BurningBlood`` -> ``BURNING_BLOOD``."""
    return re.sub(r"[^A-Z0-9_]", "", _CAMEL.sub("_", class_name.strip()).upper())


_CLASS = re.compile(r"^public (?:sealed |abstract )*class (\w+)\b", re.M)


def decomp_classes(version: str, kind: str) -> dict[str, str]:
    """Concrete model classes of ``Models.<kind>`` in ``version``'s decomp.

    Returns ``class name -> ModelId entry`` (the slug). Abstract classes are
    left out: they are bases, not content.
    """
    folder = _DECOMP_ROOT / DECOMP_DIR[version] / f"MegaCrit.Sts2.Core.Models.{kind}"
    out: dict[str, str] = {}
    for path in sorted(folder.glob("*.cs")):
        text = _read(path)
        for m in _CLASS.finditer(text):
            if m.group(1) == path.stem and "abstract " not in m.group(0):
                out[path.stem] = slugify(path.stem)
    if not out:
        raise RuntimeError(f"no model classes under {folder}")
    return out
