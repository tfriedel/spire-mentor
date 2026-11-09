"""The e2e content-ID table: one append-only, engine-registry-backed table.

Every content reference in a row (a card, relic, potion, power, monster,
encounter, event, enchantment, orb, rest option or monster move) is one id
into a single `contracts.VOCAB_ROWS`-row table (4,096 rows), shared by the
policy trunk, the value module and the content prior.

* id 0 is NONE: the embedding row is all zeros and never trained.
* ids 1-31 are specials: `UNK_<registry>` for a name the table lacks (the
  "missing" row of that registry), the rest spare.
* ids from 32 are the registry blocks, in `REGISTRIES` order, each in the
  engine's registry order at creation.

`vocab.tsv` beside this module is the source of truth. It is append-only: a
later wheel's new names go after the last used id (whatever their registry),
and no row is ever removed or renumbered. `python -m ppo.e2e.vocab --check`
compares it with the installed engine; `--append` appends what is new. Both
fail on a name whose canonical form another spelling already holds.

The sha256 covers the canonical serialization (`id\\tregistry\\tname\\n` per
row, UTF-8, LF), never the file bytes, so CRLF checkouts hash the same.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from .contracts import VOCAB_ROWS

TSV_PATH: Final = Path(__file__).with_name("vocab.tsv")

# Block order. Never reorder; a new registry appends here and in specials.
REGISTRIES: Final = (
    "card",
    "relic",
    "potion",
    "power",
    "monster",
    "encounter",
    "event",
    "enchantment",
    "orb",
    "rest",
    "move",
)
NONE_ID: Final = 0
UNK_ID: Final = {r: i + 1 for i, r in enumerate(REGISTRIES)}  # 1..11
FIRST_BLOCK_ID: Final = 32
# `RestSiteOption::option_id` (sts2-core runstate.rs:1030-1041); the engine
# exposes no registry for it, so the list is pinned here.
REST_OPTIONS: Final = ("HEAL", "SMITH", "DIG", "LIFT", "COOK", "CLONE", "HATCH", "KINDLE", "MEND")
# The event fights' encounters: `list_encounters()` lists the map's encounters
# only, but `CombatState.encounter_id` names an event's fight by its event
# encounter (sts2-wire `emit::names::event_encounter_id`, GH #1097). Pinned
# from the v0.111.0 decomp, `Models.Encounters/*EventEncounter.cs`, in file
# order, as C# `ModelId` slugs. On v0.111.0 the engine reports the game's
# V1-V3 subclass of each BattlewornDummy setting (#1181; v0.107.1 has only the
# abstract `BATTLEWORN_DUMMY_EVENT_ENCOUNTER`); it has no PunchOff or
# TheArchitect fight yet.
# Vocabulary revision 2.1 appended them at ids 1939-1947.
EVENT_ENCOUNTERS: Final = (
    "BATTLEWORN_DUMMY_EVENT_ENCOUNTER",
    "BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER",
    "BATTLEWORN_DUMMY_EVENT_V2_ENCOUNTER",
    "BATTLEWORN_DUMMY_EVENT_V3_ENCOUNTER",
    "DENSE_VEGETATION_EVENT_ENCOUNTER",
    "FAKE_MERCHANT_EVENT_ENCOUNTER",
    "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER",
    "PUNCH_OFF_EVENT_ENCOUNTER",
    "THE_ARCHITECT_EVENT_ENCOUNTER",
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _upper(name: str) -> str:
    return str(name).strip().upper()


def _canon_content(name: str) -> str:
    """`CARD.STRIKE_IRONCLAD` / `StrikeIronclad` -> `STRIKE_IRONCLAD` (ppo.vocab.canonical)."""
    s = str(name).split(".", 1)[-1]
    s = _CAMEL.sub("_", s).upper()
    if s.endswith("_CARD") and len(s) > len("_CARD"):
        s = s[: -len("_CARD")]
    return s


def _canon_enchantment(name: str) -> str:
    """`Tezcatara's Ember` -> `TEZCATARAS_EMBER` (tokens.enchantment_norm)."""
    s = _canon_content(str(name).replace("'", "").replace("’", ""))
    return _NON_ALNUM.sub("_", s).strip("_")


def _canon_event(name: str) -> str:
    """Events are PascalCase in both the registry and `current_event_name()`;
    fold case and separators so `ABYSSAL_BATHS` finds `AbyssalBaths`."""
    return _NON_ALNUM.sub("", _CAMEL.sub("_", str(name)).upper())


def _canon_move(name: str) -> str:
    species, _, move = str(name).partition(".")
    return f"{_NON_ALNUM.sub('', species.upper())}.{_upper(move)}"


def _canon_monster(name: str) -> str:
    return _NON_ALNUM.sub("", str(name).upper())


CANON: Final[dict[str, Callable[[str], str]]] = {
    "card": _canon_content,
    "relic": _canon_content,
    "potion": _canon_content,
    "power": _upper,
    "monster": _canon_monster,
    "encounter": _upper,
    "event": _canon_event,
    "enchantment": _canon_enchantment,
    "orb": _upper,
    "rest": _upper,
    "move": _canon_move,
}


@dataclass(frozen=True)
class Row:
    id: int
    registry: str
    name: str  # the engine's spelling (canonical lookups go through CANON)


def engine_names() -> dict[str, list[str]]:
    """Each registry's names from the installed engine, in registry order."""
    import sts2_rs

    def reg(kind: str) -> list[str]:
        return list(sts2_rs.list_registry_names(kind))

    species = list(sts2_rs.monster_species())
    moves = [f"{s}.{m['id']}" for s in species for m in sts2_rs.monster_moves(s)]
    return {
        "card": reg("card"),
        "relic": reg("relic_id"),
        "potion": reg("potion"),
        "power": reg("power_all"),
        "monster": species,
        "encounter": [str(e["id"]) for e in sts2_rs.list_encounters()] + list(EVENT_ENCOUNTERS),
        "event": reg("event"),
        "enchantment": reg("enchantment"),
        "orb": reg("orb"),
        "rest": list(REST_OPTIONS),
        "move": moves,
    }


def build_rows(names: dict[str, Sequence[str]], existing: Sequence[Row] = ()) -> list[Row]:
    """The table: `existing` unchanged, then every new `(registry, name)` appended.

    With no `existing` rows the registry blocks start at `FIRST_BLOCK_ID`. A
    name already in the table, spelled the same, is skipped.

    Raises:
        ValueError: Two different spellings share a registry's canonical name
            (two engine names, or an engine name and an existing row): they
            would share one id, so a human decides. Or the table would
            outgrow `VOCAB_ROWS`.
    """
    rows = list(existing) or [Row(NONE_ID, "special", "NONE")] + [
        Row(UNK_ID[r], "special", f"UNK_{r.upper()}") for r in REGISTRIES
    ]
    seen = {(r.registry, CANON[r.registry](r.name)): r.name for r in rows if r.registry in CANON}
    nxt = max(FIRST_BLOCK_ID, max(r.id for r in rows) + 1)
    for reg in REGISTRIES:
        for raw in names.get(reg, ()):
            name = str(raw)
            key = (reg, CANON[reg](name))
            had = seen.get(key)
            if had is not None:
                if had != name:
                    raise ValueError(
                        f"{reg} names {had!r} and {name!r} share the canonical name {key[1]!r}"
                    )
                continue
            seen[key] = name
            rows.append(Row(nxt, reg, name))
            nxt += 1
    if nxt > VOCAB_ROWS:
        raise ValueError(f"vocabulary needs {nxt} rows > {VOCAB_ROWS}")
    _check_unique(rows)
    return rows


def _check_unique(rows: Iterable[Row]) -> None:
    ids: set[int] = set()
    keys: set[tuple[str, str]] = set()
    for r in rows:
        if r.id in ids:
            raise ValueError(f"duplicate id {r.id}")
        ids.add(r.id)
        if r.registry in CANON:
            k = (r.registry, CANON[r.registry](r.name))
            if k in keys:
                raise ValueError(f"canonical collision {k}")
            keys.add(k)


def serialize(rows: Iterable[Row]) -> str:
    return "".join(f"{r.id}\t{r.registry}\t{r.name}\n" for r in sorted(rows, key=lambda r: r.id))


def sha256(rows: Iterable[Row]) -> str:
    """sha256 of the canonical serialization."""
    return hashlib.sha256(serialize(rows).encode("utf-8")).hexdigest()


_HEADER = (
    "# e2e content-ID table (GH #1097, SCHEMA.md section 6). APPEND-ONLY: never\n"
    "# reorder, renumber or delete a row; `python -m ppo.e2e.vocab --append` adds\n"
    "# new engine names after the last id. Columns: id, registry, engine name.\n"
)


def write_tsv(rows: Sequence[Row], path: Path = TSV_PATH) -> None:
    path.write_text(_HEADER + serialize(rows), encoding="utf-8", newline="\n")


def read_tsv(path: Path = TSV_PATH) -> list[Row]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        i, reg, name = line.split("\t")
        rows.append(Row(int(i), reg, name))
    _check_unique(rows)
    return rows


class Vocab:
    """Name -> id lookups over the table."""

    def __init__(self, rows: Sequence[Row]) -> None:
        self.rows = list(rows)
        self.sha256 = sha256(self.rows)
        self.n_used = max(r.id for r in self.rows) + 1
        self._idx: dict[tuple[str, str], int] = {}
        for r in self.rows:
            if r.registry in CANON:
                self._idx[r.registry, CANON[r.registry](r.name)] = r.id

    def id(self, registry: str, name: str | None) -> int:
        """The id of `name`; NONE for None/empty, `UNK_<registry>` if absent."""
        if name is None or name == "":
            return NONE_ID
        return self._idx.get((registry, CANON[registry](str(name))), UNK_ID[registry])

    def has(self, registry: str, name: str) -> bool:
        return (registry, CANON[registry](str(name))) in self._idx

    def block(self, registry: str) -> list[int]:
        return [r.id for r in self.rows if r.registry == registry]


@lru_cache(maxsize=1)
def load() -> Vocab:
    """The committed table."""
    return Vocab(read_tsv())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.e2e.vocab", description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="fail if the engine has new names")
    g.add_argument("--append", action="store_true", help="append new engine names")
    g.add_argument("--init", action="store_true", help="write a fresh table (no TSV yet)")
    a = ap.parse_args(argv)
    names = engine_names()
    if a.init:
        if TSV_PATH.exists():
            print(f"{TSV_PATH} exists; use --append", file=sys.stderr)
            return 2
        rows = build_rows(names)
        write_tsv(rows)
        print(f"wrote {len(rows)} rows, sha256 {sha256(rows)}")
        return 0
    old = read_tsv()
    try:
        new = build_rows(names, old)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    added = new[len(old) :]
    if a.check:
        for r in added:
            print(f"new: {r.registry} {r.name}")
        print(f"{len(old)} rows, {len(added)} new, sha256 {sha256(old)}")
        return 1 if added else 0
    write_tsv(new)
    print(f"appended {len(added)} rows, sha256 {sha256(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
