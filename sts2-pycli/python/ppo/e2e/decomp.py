"""Read model classes from the C# decompilation (`sts2-wiki/decompiled/<version>/`).

The decomp is the source of truth for game content and is not part of the
public export, so everything here is offline: the content prior's
extractor and `registry_check` call it, and their tests skip when it is
absent.

`classes(registry)` lists `(registry name, class name, source)` for one
registry of the content-ID table (`ppo.e2e.vocab`): one `.cs` file per model
class under `*.Sts2.Core.Models.<Dir>/`, mocks and deprecated
placeholders left out. The name is the class name after the suffix a
registry drops (`StrengthPower` -> `STRENGTH`, `LightningOrb` -> `LIGHTNING`)
and the few aliases the registry spells differently; `vocab.CANON` folds the
rest (`StrikeIronclad` -> `STRIKE_IRONCLAD`).
"""

from __future__ import annotations

import re
from pathlib import Path

from . import vocab as V

DECOMP_VERSION = "v0.111.0"

# kind -> the last part of the decomp's model namespace: its classes sit in
# `*.Sts2.Core.Models.<part>/`, found by glob (`models_dir`), the way main's
# tests cite the decompilation.
DIRS = {
    "card": "Cards",
    "relic": "Relics",
    "potion": "Potions",
    "power": "Powers",
    "monster": "Monsters",
    "encounter": "Encounters",
    "event": "Events",
    "enchantment": "Enchantments",
    "affliction": "Afflictions",
    "orb": "Orbs",
}
# Class-name suffix the registry drops, per kind.
SUFFIX = {"power": "Power", "orb": "Orb"}
# Class name -> registry key where the plain fold differs (the registry's own
# spelling, `combat_ctx.static_tables.CLASS_ALIASES` for cards).
ALIASES: dict[str, dict[str, str]] = {"card": {"Sloth": "SLOTH_STATUS"}}
# Placeholder classes with no registry id.
SKIP = frozenset(
    {
        "DeprecatedCard",
        "DeprecatedRelic",
        "DeprecatedPotion",
        "DeprecatedPower",
        "DeprecatedMonster",
        "DeprecatedEncounter",
        "DeprecatedEvent",
        "DeprecatedEnchantment",
        "DeprecatedAffliction",
        "DeprecatedOrb",
    }
)


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def snake(cls: str) -> str:
    """`CeremonialBeastBoss` -> `CEREMONIAL_BEAST_BOSS`."""
    return _CAMEL.sub("_", cls).upper()


def repo_root() -> Path:
    """The monorepo root (this file is `sts2-pycli/python/ppo/e2e/decomp.py`)."""
    return Path(__file__).resolve().parents[4]


def decomp_dir(version: str = DECOMP_VERSION) -> Path:
    return repo_root() / "sts2-wiki" / "decompiled" / version


def models_dir(part: str, version: str = DECOMP_VERSION) -> Path | None:
    """The decomp's `*.Sts2.Core.Models.<part>` directory, or None when absent."""
    hits = sorted(p for p in decomp_dir(version).glob(f"*.Sts2.Core.Models.{part}") if p.is_dir())
    return hits[0] if hits else None


def available(version: str = DECOMP_VERSION) -> bool:
    return models_dir(DIRS["card"], version) is not None


def class_key(kind: str, cls: str) -> str:
    """The registry spelling of class `cls` (aliases and dropped suffixes applied)."""
    alias = ALIASES.get(kind, {}).get(cls)
    if alias is not None:
        return alias
    suf = SUFFIX.get(kind)
    if suf and cls.endswith(suf) and len(cls) > len(suf):
        cls = cls[: -len(suf)]
    if kind in ("power", "orb", "encounter"):
        # These registries' CANON only upper-cases: snake-case the class here.
        return snake(cls)
    return cls


def classes(kind: str, version: str = DECOMP_VERSION) -> list[tuple[str, str, str]]:
    """`(vocab key, class name, C# source)` for every model class of `kind`."""
    d = models_dir(DIRS[kind], version)
    out = []
    for p in sorted(d.glob("*.cs")) if d is not None else []:
        cls = p.stem
        if cls in SKIP:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        out.append((class_key(kind, cls), cls, src))
    return out


def registry_check(version: str = DECOMP_VERSION) -> dict[str, dict[str, list[str]]]:
    """Per registry: decomp classes the content-ID table lacks (`missing`) and
    table names with no decomp class (`extra`: abstract helpers, ids of
    another build)."""
    v = V.load()
    out = {}
    for kind in DIRS:
        if kind not in V.REGISTRIES:
            continue  # afflictions have no registry in the table
        dec = {k for k, _c, _s in classes(kind, version)}
        missing = sorted(k for k in dec if not v.has(kind, k))
        canon = {V.CANON[kind](k) for k in dec}
        extra = sorted(
            r.name for r in v.rows if r.registry == kind and V.CANON[kind](r.name) not in canon
        )
        out[kind] = {"missing": missing, "extra": extra}
    return out
