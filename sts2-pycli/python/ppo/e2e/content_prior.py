"""The content prior: 48 static numbers per content id, from the C# decomp (SCHEMA.md §6.3).

A net that reads only an id has to learn from play what a card or relic does.
The prior hands it the printed facts instead: type, rarity, cost, base
numbers, keywords, target, pool and what an upgrade changes, extracted once
from the v0.111.0 decompilation (`decomp.decomp_dir()`) by `extract()`. The rows are indexed by
the e2e content-ID table (`ppo.e2e.vocab`); the net gathers `PRIOR[X_ID_MAIN]`
(the weighted pool mean for summary tokens) from a frozen float32 `[4096, 48]`
buffer. The prior is virtual: it is never stored in rows, so filling it
changes only `prior_sha256` (SCHEMA.md §2.9, open question 8). Changing the
table is another matter: `tactical.hits_of` reads its HITS column into stored
columns, so `TABLE_SHA256` pins the table in this file (under `py_src`).

Columns are `contracts.PRIOR_COLUMNS`; values are scaled to about [0, 1] and
clipped to [-10, 10] (`CLIP`):

* `TYPE_*` (6): card type (Attack, Skill, Power, Status, Curse, Quest).
* `RAR_*` (6): rarity of a card, relic or potion. Basic and Starter are
  `RAR_BASIC`; Ancient and Boss `RAR_BOSS_ANCIENT`; Event, Shop, Token,
  Status, Curse, Quest and the rest `RAR_SPECIAL`.
* `COST` (cost / 3, 0 for X), `X_COST`, `COST_UPG_DELTA` (/ 3): cards.
* `DMG` (/ 20), `HITS` (/ 5; 1 hit for a damage card without a repeat),
  `BLOCK` (/ 20), `DMG_UPG_DELTA` and `BLOCK_UPG_DELTA` (/ 10): the
  canonical (unupgraded) variables of any model class.
* `MAGIC`: the other canonical numbers, each scaled (power amounts / 5, HP
  loss, heal and max HP / 10, gold / 100, named magic numbers / 10), summed;
  `MAGIC_UPG_DELTA` (/ 5) their upgrade deltas.
* `KW_*` (7): exhaust, ethereal, innate, retain, unplayable, sly, eternal.
* `TGT_*` (4): self or none, one enemy, all enemies, a random enemy (cards
  and potions).
* `POOL_*` (3): the Ironclad pool, the colorless pool (cards) or shared pool
  (relics, potions), any other pool (another character, event, curse, token).
* `DRAW` (cards drawn / 5), `ENERGY` (/ 3).
* `APPLIES_DEBUFF`: applies a debuff power to another creature.
* `GAINS_STRENGTH`: applies Strength to its owner.
* `EXHAUSTS_OTHER`: exhausts cards (a card: in its `OnPlay`, so its own
  Exhaust keyword does not count).
* `MON_MAXHP`: a monster's A10 maximum HP / 100.
* `MON_IS_ELITE`, `MON_IS_BOSS`: an elite or boss encounter, or a monster
  that appears in one.
* `ACT`: act (1-3) / 3 of an encounter, an event, or a monster (its first act).
* `POWER_IS_DEBUFF`: a debuff power.
* `PRIOR_RES0`: spare, always 0.
* `MISSING`: 1 for an id with no extracted class (an `UNK_*` row, a rest
  option, a monster move, a registry name the decomp has no class for); such
  a row is otherwise zero.

Row 0 (NONE) stays all zero.

`python -m ppo.e2e.content_prior --regen` rewrites `content_prior_v0111.tsv`
(one line per row with its non-zero columns) from the decomp; `table()` reads
that file, so the public export, which has no decomp, still gets the prior.
`coverage()` reports, per registry, the share of ids with an extracted class.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import contracts as CT
from . import decomp
from . import vocab as V

COLUMNS: tuple[str, ...] = tuple(CT.PRIOR_COLUMNS)
N_PRIOR = CT.N_PRIOR
assert len(COLUMNS) == N_PRIOR == 48
C = {c: i for i, c in enumerate(COLUMNS)}

TSV = Path(__file__).with_name("content_prior_v0111.tsv")
#: `sha256()` of the committed table. Stored rows depend on the TSV, not only the
#: prior fill: `tactical.hits_of` reads its HITS column (E_MAXDMG, E_KILLABLE). This
#: pin puts the file under `py_src` (a stamp; `wheel_guard` hashes `ppo/**/*.py`
#: only), and `table()` refuses a TSV that differs (review finding TOK-4). `--regen`
#: rewrites it.
TABLE_SHA256 = "7bec618fe7257847790fdca483160e084a8eca51e45117a16e2404999fb707e9"
CLIP = 10.0
# Registries with decomp model classes (`decomp.DIRS`); rest options and
# monster moves have none, so their ids are MISSING.
REGISTRIES = (
    "card",
    "relic",
    "potion",
    "power",
    "monster",
    "encounter",
    "event",
    "enchantment",
    "orb",
)

_CARD_TYPES = ("Attack", "Skill", "Power", "Status", "Curse", "Quest")
_RARITY = {
    "Basic": "RAR_BASIC",
    "Starter": "RAR_BASIC",
    "Common": "RAR_COMMON",
    "Uncommon": "RAR_UNCOMMON",
    "Rare": "RAR_RARE",
    "Ancient": "RAR_BOSS_ANCIENT",
    "Boss": "RAR_BOSS_ANCIENT",
}
_TARGET = {
    "AnyEnemy": "TGT_ENEMY",
    "AllEnemies": "TGT_ALL_ENEMIES",
    "RandomEnemy": "TGT_RANDOM",
}
_KEYWORDS = ("Exhaust", "Ethereal", "Innate", "Retain", "Unplayable", "Sly", "Eternal")
_DAMAGE_VARS = ("Damage", "ExtraDamage", "OstyDamage", "CalculatedDamage")
_OTHER_VARS = ("Stars", "Summon", "Forge", "Int", "Dynamic", "CalculationExtra")

_NUM = r"(-?\d+(?:\.\d+)?)m?"
_CARD_BASE = re.compile(
    r":\s*base\(\s*(-?\d+)\s*,\s*CardType\.(\w+)\s*,\s*CardRarity\.(\w+)\s*,\s*TargetType\.(\w+)"
)
_VAR = re.compile(r"new\s+(\w+?)Var(?:<(\w+)>)?\(\s*(?:\"(\w+)\"\s*,\s*)?" + _NUM + r"\s*[,)]")
_UPG_VAR = re.compile(
    r"DynamicVars(?:\.(\w+)|\[\"(\w+)\"\])\.UpgradeValueBy\(\s*" + _NUM + r"\s*\)"
)
_HIT_COUNT = re.compile(r"WithHitCount\(\s*(\d+)\s*\)")
_UPG_COST = re.compile(r"EnergyCost\.UpgradeBy\(\s*(-?\d+)\s*\)")
_PROP = re.compile(r"override\s+(\w+)\s+(\w+)\s*=>\s*([^;]+);")
_ASC = re.compile(r"GetValueIfAscension\(\s*AscensionLevel\.\w+\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_APPLY = re.compile(r"PowerCmd\.Apply<(\w+)>\(\s*[^,]*,\s*([^,]+),")
_POWER_REF = re.compile(r"PowerVar<(\w+)>")
_MODEL_REF = re.compile(r"ModelDb\.(\w+)<(\w+)>\(\)")
_CLASS_BASE = re.compile(r"\bclass\s+(\w+)\s*:\s*(\w+)")
snake = decomp.snake


def _segment(src: str, name: str) -> str:
    """The expression of property `name` (up to its first `;`), or ''."""
    i = src.find(name)
    if i < 0:
        return ""
    j = src.find(";", i)
    return src[i : j if j >= 0 else len(src)]


def _method_body(src: str, sig: str) -> str:
    """The brace-matched body of the first method whose header contains `sig`."""
    i = src.find(sig)
    if i < 0:
        return ""
    j = src.find("{", i)
    if j < 0:
        return ""
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j : k + 1]
    return src[j:]


def _props(src: str) -> dict[str, str]:
    """`override T Name => value;` properties (expression-bodied), by name."""
    return {m.group(2): m.group(3).strip() for m in _PROP.finditer(src)}


def _enum_value(expr: str, enum: str) -> str | None:
    m = re.search(enum + r"\.(\w+)", expr or "")
    return m.group(1) if m else None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class _World:
    """Cross-class facts: debuff powers, pools, acts and encounter rooms."""

    def __init__(self, version: str) -> None:
        root = decomp.decomp_dir(version)
        self.debuffs: set[str] = set()
        for _k, cls, src in decomp.classes("power", version):
            if _enum_value(src[src.find("PowerType Type") :][:200], "PowerType") == "Debuff":
                self.debuffs.add(cls)
        self.pool: dict[tuple[str, str], str] = {}
        for kind, d, own, shared in (
            ("Card", "CardPools", "IroncladCardPool", "ColorlessCardPool"),
            ("Relic", "RelicPools", "IroncladRelicPool", "SharedRelicPool"),
            ("Potion", "PotionPools", "IroncladPotionPool", "SharedPotionPool"),
        ):
            for p in sorted(root.glob(f"*.Sts2.Core.Models.{d}/*.cs")):
                if p.stem.startswith(("Mock", "Deprecated")):
                    continue
                col = (
                    "POOL_IRONCLAD"
                    if p.stem == own
                    else "POOL_COLORLESS"
                    if p.stem == shared
                    else "POOL_OTHER"
                )
                for m in _MODEL_REF.finditer(_read(p)):
                    if m.group(1) != kind:
                        continue
                    key = (kind, m.group(2))
                    # The Ironclad and colorless listings win over another pool's.
                    if self.pool.get(key) not in ("POOL_IRONCLAD", "POOL_COLORLESS"):
                        self.pool[key] = col
        self.act: dict[tuple[str, str], int] = {}
        for p in sorted(root.glob("*.Sts2.Core.Models.Acts/*.cs")):
            src = _read(p)
            m = re.search(r"override\s+int\s+Index\s*=>\s*(\d+)", src)
            if not m:
                continue
            act = int(m.group(1)) + 1
            for r in _MODEL_REF.finditer(src):
                kind = "Event" if r.group(1) == "AncientEvent" else r.group(1)
                key = (kind, r.group(2))
                self.act[key] = min(self.act.get(key, act), act)
        self.room: dict[str, str] = {}
        self.monster_rooms: dict[str, set[str]] = {}
        base: dict[str, str] = {}
        for _k, cls, src in decomp.classes("encounter", version):
            room = _enum_value(_props(src).get("RoomType", ""), "RoomType") or ""
            self.room[cls] = room
            m = _CLASS_BASE.search(src)
            if m and m.group(1) == cls:
                base[cls] = m.group(2)
            act = self.act.get(("Encounter", cls))
            for m in re.finditer(r"ModelDb\.Monster<(\w+)>", src):
                self.monster_rooms.setdefault(m.group(1), set()).add(room)
                if act is not None:
                    key = ("Monster", m.group(1))
                    self.act[key] = min(self.act.get(key, act), act)
        # An abstract encounter declares no room of its own; it takes the one its
        # subclasses share. `BattlewornDummyEventEncounter` is the engine's id for
        # all three of its V1-V3 settings (`vocab.EVENT_ENCOUNTERS`).
        for cls, room in list(self.room.items()):
            subs = {self.room[c] for c, b in base.items() if b == cls}
            if not room and len(subs) == 1:
                self.room[cls] = subs.pop()


def _vars(src: str, row: np.ndarray, is_attack: bool) -> None:
    """Canonical variables -> the base-number columns."""
    seg = _segment(src, "CanonicalVars")
    # A calculated card's base number is damage or block by its calculated var
    # (`CalculatedBlockVar` for Expect a Fight, `CalculatedDamageVar` for
    # Body Slam); its `CalculationExtraVar` is the per-unit increment.
    calc = "block" if "CalculatedBlockVar" in seg else "damage" if is_attack else "other"
    if "CalculatedDamageVar" in seg:
        calc = "damage"
    dmg = blk = hits = 0.0
    for m in _VAR.finditer(seg):
        kind, generic, _name, val = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if kind in _DAMAGE_VARS:
            dmg = dmg or val
        elif kind == "CalculationBase":
            if calc == "damage":
                dmg = dmg or val
            elif calc == "block":
                blk = blk or val
            else:
                row[C["MAGIC"]] += val / 10.0
        elif kind in ("Block", "CalculatedBlock"):
            blk = blk or val
        elif kind == "Repeat":
            hits = val
        elif kind == "Cards":
            row[C["DRAW"]] += val / 5.0
        elif kind == "Energy":
            row[C["ENERGY"]] += val / 3.0
        elif kind == "Power" and generic:
            row[C["MAGIC"]] += val / 5.0
        elif kind in ("HpLoss", "Heal", "MaxHp"):
            row[C["MAGIC"]] += val / 10.0
        elif kind == "Gold":
            row[C["MAGIC"]] += val / 100.0
        elif kind in _OTHER_VARS:
            row[C["MAGIC"]] += val / 10.0
    row[C["DMG"]] = dmg / 20.0
    row[C["BLOCK"]] = blk / 20.0
    if dmg > 0:
        row[C["HITS"]] = (hits or 1.0) / 5.0


def _effects(src: str, row: np.ndarray, w: _World, *, targets_self: bool) -> None:
    """Debuffs applied to others and Strength gained, from `PowerCmd.Apply`."""
    for power, target in _APPLY.findall(src):
        own = "Owner" in target or target.strip() in ("base.Creature", "Creature")
        if power in w.debuffs and not own:
            row[C["APPLIES_DEBUFF"]] = 1.0
        if power == "StrengthPower" and own:
            row[C["GAINS_STRENGTH"]] = 1.0
    if not row[C["APPLIES_DEBUFF"]] and not targets_self:
        # A canonical debuff variable (Bash's Vulnerable) on a card or potion
        # that targets an enemy, when the apply goes through a helper.
        if set(_POWER_REF.findall(_segment(src, "CanonicalVars"))) & w.debuffs:
            row[C["APPLIES_DEBUFF"]] = 1.0


def _card(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    m = _CARD_BASE.search(src)
    if not m:
        return False
    cost, ctype, rarity, target = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    if ctype in _CARD_TYPES:
        row[C["TYPE_" + ctype.upper()]] = 1.0
    row[C[_RARITY.get(rarity, "RAR_SPECIAL")]] = 1.0
    tgt = _TARGET.get(target, "TGT_SELF")
    row[C[tgt]] = 1.0
    xcost = "HasEnergyCostX => true" in src
    row[C["COST"]] = 0.0 if xcost else max(cost, 0) / 3.0
    row[C["X_COST"]] = float(xcost)
    _vars(src, row, ctype == "Attack")
    # A literal hit count (`WithHitCount(2)`: Twin Strike, Thrash) when the
    # card has no repeat variable.
    lit = _HIT_COUNT.findall(src)
    if row[C["DMG"]] > 0 and lit and row[C["HITS"]] <= 0.2 + 1e-6:
        row[C["HITS"]] = max(int(x) for x in lit) / 5.0
    kws = set(re.findall(r"CardKeyword\.(\w+)", _segment(src, "CanonicalKeywords")))
    for kw in _KEYWORDS:
        if kw in kws:
            row[C["KW_" + kw.upper()]] = 1.0
    up = _method_body(src, "void OnUpgrade()")
    for c in _UPG_COST.findall(up):
        row[C["COST_UPG_DELTA"]] += int(c) / 3.0
    for prop, named, val in _UPG_VAR.findall(up):
        v = float(val)
        name = prop or named
        calc_blk = name == "CalculationBase" and "CalculatedBlockVar" in src
        if name in _DAMAGE_VARS or (name == "CalculationBase" and ctype == "Attack"):
            row[C["DMG_UPG_DELTA"]] += v / 10.0
        elif name in ("Block", "CalculatedBlock") or calc_blk:
            row[C["BLOCK_UPG_DELTA"]] += v / 10.0
        else:
            row[C["MAGIC_UPG_DELTA"]] += v / 5.0
    row[C[w.pool.get(("Card", cls), "POOL_OTHER")]] = 1.0
    play = _method_body(src, "OnPlay(")
    _effects(play + _segment(src, "CanonicalVars"), row, w, targets_self=tgt == "TGT_SELF")
    row[C["EXHAUSTS_OTHER"]] = float("CardCmd.Exhaust(" in play)
    return True


def _relic(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    rar = _enum_value(_props(src).get("Rarity", ""), "RelicRarity")
    if rar is None:
        return False
    row[C[_RARITY.get(rar, "RAR_SPECIAL")]] = 1.0
    _vars(src, row, False)
    _effects(src, row, w, targets_self=True)
    row[C["EXHAUSTS_OTHER"]] = float("CardCmd.Exhaust(" in src)
    row[C[w.pool.get(("Relic", cls), "POOL_OTHER")]] = 1.0
    return True


def _potion(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    p = _props(src)
    rar = _enum_value(p.get("Rarity", ""), "PotionRarity")
    if rar is None:
        return False
    row[C[_RARITY.get(rar, "RAR_SPECIAL")]] = 1.0
    tgt = _TARGET.get(_enum_value(p.get("TargetType", ""), "TargetType") or "", "TGT_SELF")
    row[C[tgt]] = 1.0
    _vars(src, row, False)
    _effects(src, row, w, targets_self=tgt == "TGT_SELF")
    row[C["EXHAUSTS_OTHER"]] = float("CardCmd.Exhaust(" in src)
    row[C[w.pool.get(("Potion", cls), "POOL_OTHER")]] = 1.0
    return True


def _power(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    ptype = _enum_value(src[src.find("PowerType Type") :][:200], "PowerType")
    stack = _enum_value(src[src.find("PowerStackType StackType") :][:200], "PowerStackType")
    if ptype is None and stack is None:
        return False
    row[C["POWER_IS_DEBUFF"]] = float(ptype == "Debuff")
    _vars(src, row, False)
    return True


def _hp(expr: str) -> float | None:
    m = _ASC.search(expr)
    if m:
        return float(m.group(1))  # A10 >= ToughEnemies (8): the ascension value
    m = re.fullmatch(r"\s*(-?\d+)\s*", expr)
    return float(m.group(1)) if m else None


def _monster(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    p = _props(src)
    lo = _hp(p.get("MinInitialHp", ""))
    hi = _hp(p.get("MaxInitialHp", ""))
    if hi is None and p.get("MaxInitialHp", "").strip() == "MinInitialHp":
        hi = lo
    if lo is None:
        return False
    row[C["MON_MAXHP"]] = min(hi if hi is not None else lo, 2000.0) / 100.0
    rooms = w.monster_rooms.get(cls, set())
    row[C["MON_IS_ELITE"]] = float("Elite" in rooms)
    row[C["MON_IS_BOSS"]] = float("Boss" in rooms)
    row[C["ACT"]] = w.act.get(("Monster", cls), 0) / 3.0
    return True


def _encounter(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    room = w.room.get(cls)
    if not room:
        return False
    row[C["MON_IS_ELITE"]] = float(room == "Elite")
    row[C["MON_IS_BOSS"]] = float(room == "Boss")
    row[C["ACT"]] = w.act.get(("Encounter", cls), 0) / 3.0
    return True


def _event(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    _vars(src, row, False)
    row[C["ACT"]] = w.act.get(("Event", cls), 0) / 3.0
    return True


def _generic(cls: str, src: str, row: np.ndarray, w: _World) -> bool:
    _vars(src, row, False)
    return True


_EXTRACT = {
    "card": _card,
    "relic": _relic,
    "potion": _potion,
    "power": _power,
    "monster": _monster,
    "encounter": _encounter,
    "event": _event,
    "enchantment": _generic,
    "orb": _generic,
}


def class_names(registry: str, cls: str) -> Iterable[str]:
    """Registry spellings to try for decomp class `cls`, best first."""
    if registry == "card":
        yield from (decomp.ALIASES.get("card", {}).get(cls, cls), cls)
    elif registry in ("power", "orb"):
        suf = decomp.SUFFIX[registry]
        base = cls[: -len(suf)] if cls.endswith(suf) and len(cls) > len(suf) else cls
        yield from (snake(base), snake(cls))
    elif registry == "encounter":
        yield snake(cls)
    else:
        yield cls


def class_id(registry: str, cls: str) -> int | None:
    """The content id of decomp class `cls`, or None if the table lacks it."""
    v = V.load()
    return next((v.id(registry, n) for n in class_names(registry, cls) if v.has(registry, n)), None)


def extract(version: str = decomp.DECOMP_VERSION) -> dict[int, np.ndarray]:
    """`id -> 48 floats` for every content id with a decomp class."""
    w = _World(version)
    out: dict[int, np.ndarray] = {}
    for reg in REGISTRIES:
        for _k, cls, src in decomp.classes(reg, version):
            vid = class_id(reg, cls)
            if vid is None or vid in out:
                continue
            row = np.zeros(N_PRIOR, np.float32)
            if _EXTRACT[reg](cls, src, row, w):
                # Event rows carry gold amounts as named magic numbers: clip so
                # no column dwarfs the rest.
                out[vid] = np.clip(row, -CLIP, CLIP)
    return out


def build(extracted: dict[int, np.ndarray]) -> np.ndarray:
    """The dense `[4096, 48]` table: extracted rows, `MISSING` on every other used id."""
    v = V.load()
    t = np.zeros((CT.VOCAB_ROWS, N_PRIOR), np.float32)
    for r in v.rows:
        if r.id != V.NONE_ID:
            t[r.id, C["MISSING"]] = 1.0
    for vid, row in extracted.items():
        t[vid] = row
    t[V.NONE_ID] = 0.0
    return t


def render(t: np.ndarray) -> str:
    v = V.load()
    lines = [
        "# id\tregistry\tname\tcol=value,... (ppo/e2e/vocab.tsv ids; generated by"
        " `python -m ppo.e2e.content_prior --regen`)"
    ]
    for r in v.rows:
        nz = np.flatnonzero(t[r.id])
        if len(nz) == 0:
            continue
        cells = ",".join(f"{COLUMNS[i]}={float(t[r.id, i]):.9g}" for i in nz)
        lines.append(f"{r.id}\t{r.registry}\t{r.name}\t{cells}")
    return "\n".join(lines) + "\n"


def parse(text: str) -> np.ndarray:
    t = np.zeros((CT.VOCAB_ROWS, N_PRIOR), np.float32)
    for line in text.splitlines():
        line = line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        r, _reg, _name, cells = line.split("\t")
        for cell in cells.split(","):
            name, val = cell.split("=")
            t[int(r), C[name]] = float(val)
    return t


@lru_cache(maxsize=1)
def _text() -> str:
    return TSV.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def table() -> np.ndarray:
    """The committed prior, `[4096, 48]` float32 (row 0 all zero). Read-only.
    Raises unless its sha256 is `TABLE_SHA256`."""
    t = parse(_text())
    t.setflags(write=False)
    if (got := sha256(t)) != TABLE_SHA256:
        raise RuntimeError(
            f"{TSV.name}: table sha256 {got} != TABLE_SHA256 {TABLE_SHA256}; a changed "
            "prior changes stored tactical columns, so pin it (`--regen` does)"
        )
    return t


def sha256(t: np.ndarray | None = None) -> str:
    """`prior_sha256`: sha256 of the float32 buffer's bytes (C order)."""
    t = table() if t is None else t
    return hashlib.sha256(np.ascontiguousarray(t, np.float32).tobytes()).hexdigest()


def coverage(t: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Per registry: ids in the table, ids with an extracted row, their share."""
    t = table() if t is None else t
    v = V.load()
    out = {}
    for reg in V.REGISTRIES:
        ids = v.block(reg)
        got = sum(1 for r in ids if t[r, C["MISSING"]] < 0.5)
        out[reg] = {"ids": len(ids), "extracted": got, "share": got / max(1, len(ids))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--regen", action="store_true", help="rewrite the TSV from the decomp")
    a = ap.parse_args(argv)
    t = build(extract())
    for reg, c in coverage(t).items():
        print(f"{reg:12s} {c['extracted']:4d}/{c['ids']:4d} = {c['share']:.3f}")
    print(f"prior_sha256 {sha256(t)}")
    if a.regen:
        TSV.write_text(render(t), encoding="utf-8", newline="\n")
        print(f"wrote {TSV}")
        src = Path(__file__)
        pin = re.sub(
            r'^TABLE_SHA256 = "[0-9a-f]*"$',
            f'TABLE_SHA256 = "{sha256(t)}"',
            src.read_text(encoding="utf-8"),
            count=1,
            flags=re.M,
        )
        src.write_text(pin, encoding="utf-8", newline="\n")
        print(f"pinned TABLE_SHA256 in {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
