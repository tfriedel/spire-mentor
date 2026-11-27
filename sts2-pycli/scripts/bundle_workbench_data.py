#!/usr/bin/env python3
"""Bundle the workbench's runtime data for the Rust ``sts2-workbench`` crate.

The Python workbench (``python/workbench/localization.py`` / ``assets.py`` and
``server.create_app``) reads its display data from three places at runtime:
the vendored ``localization_eng/*.json`` dumps, the sibling wiki's
``sts2-wiki/data/v0.107.1/`` manifests, and the art manifest that
``scripts/build_asset_manifest.py`` derives from the extracted art tree. The
Rust port runs in the browser (wasm) with no filesystem, so it ``include_str!``s
committed copies instead. This script produces those copies under
``sts2-rs/crates/sts2-workbench/data/`` — self-contained (no include path
leaves ``sts2-rs``), deterministic (sorted keys, fixed indent, sorted globs)
and minimal (the wiki manifests are trimmed to exactly the fields the Python
loaders read).

Regenerate with ``just workbench-data``; ``just check-workbench-data`` fails
on drift between the sources and the committed bundle.

**One bundle per game build (GH #801).** The legacy build (v0.107.1) is
written to ``data/game/`` exactly as before; every other supported build gets a
full sibling set under ``data/game/<ver>/`` (``data/game/v0.111.0/``), built
from that build's vendored loc dir (``localization_eng_<ver>/``) and wiki
aggregates (``sts2-wiki/data/<ver>/``). The art tree is shared — the wiki
serves ONE extracted art set — so a card new in a later build has no art of its
own and the resolver falls back to the placeholder; the per-version asset
manifest records exactly what the art store holds. The deck analyzer's tables
are per build too (GH #877): each set gets its build's ``card_db.json``
(``python/workbench/data/[<ver>/]card_db.json``; a build with none of its own
gets the default build's), and a non-default set also carries that build's
curated ``card_tags.json`` OVERLAY when it has one — the curated default table
stays at the top level, shared.

Usage:
    python scripts/bundle_workbench_data.py [-o OUT_DIR] [--version VER]

With no ``--version`` every build is written (what ``just workbench-data``
and ``just check-workbench-data`` do) — the default build always, every other
build when its sources (``localization_eng_<ver>/`` + ``sts2-wiki/data/<ver>/``)
are on disk, so a checkout that imported only the default build still bundles.
``--version`` writes just that one build and requires its sources.

Stdlib only — it is a build step, not part of the package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_MONO_ROOT = _REPO_ROOT.parent

# Keep in lockstep with ``workbench.localization.LEGACY_GAME_VERSION`` /
# ``SUPPORTED_GAME_VERSIONS`` (and the Rust ``sts2_core::GameVersion``).
_ENGINE_WIKI_VERSION = "v0.107.1"
_VERSIONS = ("v0.107.1", "v0.111.0", "v0.113.0", "v0.114.0")

_WIKI_ROOT = _MONO_ROOT / "sts2-wiki" / "data"
_IMAGES = _MONO_ROOT / "sts2-wiki" / "site" / "public" / "images"
_AUDIO = _MONO_ROOT / "sts2-wiki" / "site" / "public" / "audio"
_DEFAULT_OUT = _MONO_ROOT / "sts2-rs" / "crates" / "sts2-workbench" / "data"

# ``localization_eng`` files copied VERBATIM (byte-for-byte). Every one is a
# flat ``{"KEY.field": "string"}`` dict; the Rust side parses them as
# ``BTreeMap<String, String>`` (``powers.json`` additionally in file order).
_LOC_FILES = (
    "BUILD.txt",
    "afflictions.json",
    "card_keywords.json",
    "cards.json",
    "characters.json",
    "encounters.json",
    "events.json",
    "monsters.json",
    "orbs.json",
    "potions.json",
    "powers.json",
    "relics.json",
    "rest_site_ui.json",
)

# The game's Simplified Chinese tables (``localization_zhs/``, vendored by
# ``scripts/vendor_localization.py`` from the newest build), copied VERBATIM
# into ``game/localization_zhs/``: the browser workbench's DISPLAY text when
# the page runs in Chinese (``sts2-workbench`` ``localization::zh_overlay``).
# One set for every build: a key a build lacks simply stays English. Absent
# sources = no zh bundle (``build.rs`` stubs it; the page shows English).
_ZH_LOC_DIR = _REPO_ROOT / "localization_zhs"
_ZH_LOC_FILES = (
    "BUILD.txt",
    "afflictions.json",
    "card_keywords.json",
    "cards.json",
    "characters.json",
    "encounters.json",
    "events.json",
    "potions.json",
    "relics.json",
    "rest_site_ui.json",
)
# More Chinese tables, merged into ONE ``extra.json`` (``{table: {key: text}}``)
# for the names and words the engine and the page write themselves
# (monster / power / orb / intent names, acts, game modes, ascension levels,
# the game's UI words). Read through ``data::ZhTables::extra``.
_ZH_EXTRA_TABLES = (
    "acts",
    "ancients",
    "ascension",
    "card_reward_ui",
    "card_selection",
    "combat_messages",
    "enchantments",
    "game_modes",
    "game_over_screen",
    "gameplay_ui",
    "intents",
    "map",
    "merchant_room",
    "modifiers",
    "monsters",
    "orbs",
    "powers",
    "static_hover_tips",
)

# The wiki card fields ``localization.load_card_localization`` /
# ``load_card_meta`` read. Nested ``upgrades`` / ``vars`` entries are trimmed
# to their read keys below.
_CARD_FIELDS = (
    "loc_key",
    "title",
    "character",
    "star_cost",
    "x_star_cost",
    "description_template",
    "description_plain",
    "upgraded_description_plain",
    "keywords",
    "type",
    "rarity",
    "energy_cost",
    "upgraded_cost",
)

# The wiki monster fields ``localization.load_monster_hp_ranges`` reads.
_MONSTER_FIELDS = ("class_name", "min_hp", "max_hp", "min_hp_base", "max_hp_base", "hp_ascension")


def _is_int(value: Any) -> bool:
    """``isinstance(x, int)`` minus ``bool`` — the loaders' integer test."""
    return isinstance(value, int) and not isinstance(value, bool)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _trim_card(card: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only what the Python loaders read, dropping values they ignore.

    A missing key and a falsy value are the same thing to ``load_card_
    localization`` (``if c.get("title")``), EXCEPT for numbers — ``energy_cost
    == 0`` and ``star_cost == 0`` are real values (``is not None`` tests) — so
    ``0`` is kept while ``None`` / ``""`` / ``[]`` / ``False`` are dropped.
    """
    key = card.get("loc_key")
    if not key:
        return None
    out: dict[str, Any] = {}
    for field in _CARD_FIELDS:
        value = card.get(field)
        if value is None or value == "" or value == [] or value is False:
            continue
        if field in ("energy_cost", "upgraded_cost", "star_cost"):
            if not _is_int(value):
                raise SystemExit(f"{key}: {field} is {value!r}, expected an int")
        elif field == "x_star_cost":
            if value is not True:
                raise SystemExit(f"{key}: x_star_cost is {value!r}, expected a bool")
        elif field == "keywords":
            if not all(isinstance(k, str) for k in value):
                raise SystemExit(f"{key}: keywords must be strings: {value!r}")
        elif not isinstance(value, str):
            raise SystemExit(f"{key}: {field} is {value!r}, expected a string")
        out[field] = value
    # Keyword-only upgrade deltas (``add_keyword`` / ``remove_keyword``).
    deltas = []
    for up in card.get("upgrades") or []:
        keyword = up.get("keyword")
        if not keyword:
            continue
        entry = {"keyword": keyword}
        if up.get("action") is not None:
            entry["action"] = up["action"]
        deltas.append(entry)
    if deltas:
        out["upgrades"] = deltas
    # Per-variable base / upgraded values.
    var_entries = []
    for var in card.get("vars") or []:
        if not var.get("type"):
            continue
        entry: dict[str, Any] = {"type": var["type"]}
        for vfield in ("base_value", "upgraded_value"):
            if var.get(vfield) is not None:
                if not _is_int(var[vfield]):
                    raise SystemExit(f"{key}: var {var['type']} {vfield} is not an int")
                entry[vfield] = var[vfield]
        var_entries.append(entry)
    if var_entries:
        out["vars"] = var_entries
    return out


def _trim_monster(row: dict[str, Any]) -> dict[str, Any] | None:
    kind = row.get("class_name")
    if not isinstance(kind, str):
        return None
    out: dict[str, Any] = {"class_name": kind}
    for field in _MONSTER_FIELDS[1:]:
        if _is_int(row.get(field)):
            out[field] = row[field]
    return out


def _loc_dir(version: str) -> Path:
    """``localization_eng/`` for the default build, ``localization_eng_<ver>/`` else.

    Mirrors ``workbench.localization.loc_dir_for``."""
    if version == _ENGINE_WIKI_VERSION:
        return _REPO_ROOT / "localization_eng"
    return _REPO_ROOT / f"localization_eng_{version}"


def _game_out(out_dir: Path, version: str) -> Path:
    """``<out>/game`` for the default build, ``<out>/game/<ver>`` else."""
    game = out_dir / "game"
    return game if version == _ENGINE_WIKI_VERSION else game / version


def _owned_items(wiki_data: Path, kind: str) -> list[dict[str, str]]:
    """``{class_name, description}`` per relic / potion, by class name.

    Read from the regex-extracted aggregate (``relics.json`` / ``potions.json``,
    ``just extract`` in sts2-wiki), whose ``description_resolved`` has every
    DynamicVar substituted — the same text the LLM-extracted per-item files
    carried, minus the LLM. Sorted by class name, which is the file-name order
    the per-item loader used, so the Rust ``load_owned_localization`` twin
    keeps its first-wins semantics.
    """
    rows: Any = json.loads((wiki_data / f"{kind}.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{kind}.json: expected a list")
    out = []
    for item in sorted(
        (r for r in rows if isinstance(r, dict)), key=lambda r: r.get("class_name", "")
    ):
        class_name = item.get("class_name")
        desc = item.get("description_resolved")
        if not class_name or not desc:
            continue
        out.append({"class_name": class_name, "description": desc})
    return out


def _check_loc_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a flat JSON object")
    bad = {k: v for k, v in data.items() if not isinstance(v, str)}
    if bad:
        raise SystemExit(f"{path}: non-string localization values: {sorted(bad)[:5]}")


def _has_sources(version: str) -> bool:
    return _loc_dir(version).is_dir() and (_WIKI_ROOT / version / "cards.json").is_file()


def build(out_dir: Path, versions: tuple[str, ...] | None = None) -> dict[str, int]:
    """Write the bundle of ``versions`` under ``out_dir``; return ``{name: bytes}``.

    ``None`` = the default build plus every other build whose sources exist.
    """
    if versions is None:
        versions = tuple(v for v in _VERSIONS if v == _ENGINE_WIKI_VERSION or _has_sources(v))
        for v in _VERSIONS:
            if v not in versions:
                print(
                    f"note: skipping {v} (no localization_eng_{v}/ or wiki data/{v}/)",
                    file=sys.stderr,
                )
    sizes: dict[str, int] = {}
    for version in versions:
        sizes.update(_build_version(out_dir, version))
    if _ZH_LOC_DIR.is_dir():
        sizes.update(_build_zh(out_dir))
    else:
        print(
            f"note: no {_ZH_LOC_DIR.name}/ — the workbench shows game text in English only",
            file=sys.stderr,
        )
    return sizes


def _owned_vars() -> dict[str, dict[str, Any]]:
    """``{LOC_KEY: {var_type_lower: base_value}}`` for every relic and potion,
    newest wiki build last (it wins). The zh overlay renders the Chinese
    description templates with these, since the wiki's pre-resolved plain
    text (``wiki_owned.json``) is English."""
    out: dict[str, dict[str, Any]] = {}
    for version in _VERSIONS:
        for kind in ("relics", "potions"):
            path = _WIKI_ROOT / version / f"{kind}.json"
            if not path.is_file():
                continue
            rows: Any = json.loads(path.read_text(encoding="utf-8"))
            for row in rows if isinstance(rows, list) else []:
                key = row.get("loc_key") if isinstance(row, dict) else None
                if not key:
                    continue
                values = {
                    str(v["type"]).lower(): v["base_value"]
                    for v in row.get("vars") or []
                    if isinstance(v, dict) and v.get("type") and _is_int(v.get("base_value"))
                }
                out[key] = values
    return {k: out[k] for k in sorted(out)}


def _first_line(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return lines[0].strip() if lines else ""


def _write_zh_set(src_dir: Path, zh_out: Path, out_dir: Path) -> dict[str, int]:
    """Copy one Chinese set's tables (and its merged ``extra.json``)."""
    zh_out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name in _ZH_LOC_FILES:
        src = src_dir / name
        if name.endswith(".json"):
            _check_loc_file(src)
        dst = zh_out / name
        shutil.copyfile(src, dst)
        sizes[str(dst.relative_to(out_dir))] = dst.stat().st_size
    extra: dict[str, Any] = {}
    for table in _ZH_EXTRA_TABLES:
        src = src_dir / f"{table}.json"
        _check_loc_file(src)
        extra[table] = json.loads(src.read_text(encoding="utf-8"))
    path = zh_out / "extra.json"
    path.write_text(_dumps(extra), encoding="utf-8", newline="\n")
    sizes[str(path.relative_to(out_dir))] = path.stat().st_size
    return sizes


def _build_zh(out_dir: Path) -> dict[str, int]:
    """Write ``game/localization_zhs/`` (the shared set, from the newest
    build) and, for each other build that has its own vendored set
    (``localization_zhs_<ver>/``), ``game/<ver>/localization_zhs/``: the
    Rust side guards the shared set by that build's English and lays the
    build's own set over it (``data::zh_tables_for``)."""
    sizes = _write_zh_set(_ZH_LOC_DIR, out_dir / "game" / "localization_zhs", out_dir)
    path = out_dir / "game" / "localization_zhs" / "owned_vars.json"
    path.write_text(_dumps(_owned_vars()), encoding="utf-8", newline="\n")
    sizes[str(path.relative_to(out_dir))] = path.stat().st_size
    for version in _VERSIONS:
        own = _REPO_ROOT / f"localization_zhs_{version}"
        if version == _ENGINE_WIKI_VERSION or not own.is_dir():
            continue  # the default build's own set is not bundled (none exists)
        if _first_line(own / "BUILD.txt") == _first_line(_ZH_LOC_DIR / "BUILD.txt"):
            continue  # the shared set IS this build's (no copy needed)
        sizes.update(_write_zh_set(own, out_dir / "game" / version / "localization_zhs", out_dir))
    return sizes


def _build_version(out_dir: Path, version: str) -> dict[str, int]:
    """Write one build's bundle (``game/`` or ``game/<ver>/``)."""
    sys.path.insert(0, str(_HERE))
    from build_asset_manifest import added_audio_of, build_manifest

    if version not in _VERSIONS:
        raise SystemExit(
            f"error: unsupported version {version!r}; supported: {', '.join(_VERSIONS)}"
        )
    loc_dir = _loc_dir(version)
    wiki_data = _WIKI_ROOT / version
    for required in (loc_dir, wiki_data, _IMAGES):
        if not required.exists():
            raise SystemExit(f"error: source not found: {required}")

    # Game-derived files go under ``game/`` (never published; ``build.rs``
    # stubs them when absent); only the curated ``card_tags.json`` sits at the
    # top level and is committed everywhere (shared by every build).
    game_out = _game_out(out_dir, version)
    loc_out = game_out / "localization_eng"
    loc_out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    def rel(path: Path) -> str:
        return str(path.relative_to(out_dir))

    for name in _LOC_FILES:
        src = loc_dir / name
        if name.endswith(".json"):
            _check_loc_file(src)
        dst = loc_out / name
        shutil.copyfile(src, dst)
        sizes[rel(dst)] = dst.stat().st_size
    # The vendored BUILD.txt names the build its tables came from; a bundle
    # labelled with the wrong build would silently describe other rules.
    build_line = (loc_out / "BUILD.txt").read_text(encoding="utf-8").splitlines()[:1]
    if build_line != [version]:
        raise SystemExit(f"error: {loc_dir}/BUILD.txt says {build_line}, expected {version}")

    cards: Any = json.loads((wiki_data / "cards.json").read_text(encoding="utf-8"))
    if isinstance(cards, dict):
        cards = cards.get("cards", [])
    trimmed_cards = [t for t in (_trim_card(c) for c in cards or []) if t is not None]

    monsters: Any = json.loads((wiki_data / "monsters.json").read_text(encoding="utf-8"))
    if not isinstance(monsters, list):
        raise SystemExit("monsters.json: expected a list")
    trimmed_monsters = [
        t for t in (_trim_monster(r) for r in monsters if isinstance(r, dict)) if t is not None
    ]

    owned = _owned_items(wiki_data, "relics") + _owned_items(wiki_data, "potions")

    # The default build never sees an overlay; a later build lists its own
    # (GH #801), so its resolver finds the art that build added.
    overlays = () if version == _ENGINE_WIKI_VERSION else (version,)
    # Likewise for sound: a later build's new events live in the SHARED audio
    # tree, so every other build's manifest leaves out what that build's
    # import recorded as its own (`import_game.record_added_audio`).
    exclude_audio = added_audio_of(_WIKI_ROOT, [v for v in _VERSIONS if v != version])
    manifest = build_manifest(_IMAGES, _AUDIO, overlays=overlays, exclude_audio=exclude_audio)

    scales_src = _IMAGES / "creature_scales.json"
    if not scales_src.is_file():
        raise SystemExit(f"error: {scales_src} not found")
    scales = json.loads(scales_src.read_text(encoding="utf-8"))
    for overlay in overlays:
        # The creatures that build added (import_game's art overlay step).
        own_scales = _IMAGES / overlay / "creature_scales.json"
        if own_scales.is_file():
            scales = {**scales, **json.loads(own_scales.read_text(encoding="utf-8"))}
            scales = {k: scales[k] for k in sorted(scales)}

    # The deck analyzer's two tables (`workbench/deck_analysis.py`), copied
    # as-is: the Rust twin reads the same keys off the same JSON. Per build
    # (GH #877, `deck_analysis.data_dir_for`): the default build's sit in
    # `python/workbench/data/`, another build's in `python/workbench/data/<ver>/`.
    tables_dir = _REPO_ROOT / "python" / "workbench" / "data"
    own_dir = tables_dir if version == _ENGINE_WIKI_VERSION else tables_dir / version
    tables: list[tuple[Path, Path]] = []
    db_src = own_dir / "card_db.json"
    tables.append((db_src if db_src.is_file() else tables_dir / "card_db.json", game_out))
    if version == _ENGINE_WIKI_VERSION:
        tables.append((tables_dir / "card_tags.json", out_dir))
    elif (own_dir / "card_tags.json").is_file():
        # The build's overlay over the shared curated tags (`data.rs` merges).
        tables.append((own_dir / "card_tags.json", game_out))
    for src, dst_dir in tables:
        if not src.is_file():
            raise SystemExit(f"error: {src} not found")
        shutil.copyfile(src, dst_dir / src.name)
        sizes[rel(dst_dir / src.name)] = (dst_dir / src.name).stat().st_size

    for name, obj in (
        ("wiki_cards.json", trimmed_cards),
        ("wiki_monsters.json", trimmed_monsters),
        ("wiki_owned.json", owned),
        ("asset_manifest.json", manifest),
        ("creature_scales.json", scales),
    ):
        path = game_out / name
        path.write_text(_dumps(obj), encoding="utf-8", newline="\n")
        sizes[rel(path)] = path.stat().st_size
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(_DEFAULT_OUT),
        help="Output directory (default: sts2-rs/crates/sts2-workbench/data).",
    )
    parser.add_argument(
        "--version",
        choices=_VERSIONS,
        default=None,
        help="Write only this game build's bundle (default: every build).",
    )
    args = parser.parse_args(argv)
    sizes = build(Path(args.out_dir), (args.version,) if args.version else None)
    total = 0
    for name, size in sorted(sizes.items()):
        print(f"{size:>8}  {name}")
        total += size
    print(f"{total:>8}  total ({total / 1024:.0f} KB) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
