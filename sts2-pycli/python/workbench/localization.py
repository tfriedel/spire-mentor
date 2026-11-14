"""Resolve card localization keys to display strings.

The headless game occasionally emits a raw loc key (e.g. ``"PREPARED.title"``)
when its bundled localization lacks an entry. We rebuild a key -> string map
from the in-repo ``localization_eng/cards.json`` (authoritative) plus the
sibling wiki's extracted card manifest (fills newer cards the in-repo dump is
missing) so those keys can be resolved for display.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS

_VER_NUMS = re.compile(r"\d+")

# The engine simulates the game build a run was started on (GH #801): the
# default v0.107.1 (see ../sts2-rs/CLAUDE.md) or the v0.111.0 beta. Resolve
# display localization against THAT version's wiki data so card/relic/potion
# text matches the rules the engine actually plays — not whatever newer
# ``data/<ver>/`` happens to be highest on disk (the wiki also carries older
# v0.103.2 / v0.105.1 reference data).
#
# ``LEGACY_GAME_VERSION`` is the build every loader reads when no version is
# passed; it mirrors ``sts2_core::GameVersion::default()`` and the Rust
# ``data::ENGINE_WIKI_VERSION``. Each supported build maps to the wiki
# ``data/<ver>/`` dir describing it (``engine_wiki_version``).
# The build list itself lives in ``sts2_client`` (one list for the oracle and
# the workbench); every build's wiki dir is named after the build.
_ENGINE_WIKI_VERSIONS: dict[str, str] = {v: v for v in SUPPORTED_GAME_VERSIONS}
# The default build's wiki dir. Kept under its historical name: scripts
# (``bundle_workbench_data.py``, ``synergy.gen_card_text``) and tests import it.
_ENGINE_WIKI_VERSION = _ENGINE_WIKI_VERSIONS[LEGACY_GAME_VERSION]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def engine_wiki_version(version: str | None = None) -> str:
    """The wiki ``data/<ver>/`` dir name describing game build ``version``.

    Args:
        version: A supported build (``"v0.107.1"`` / ``"v0.111.0"``); ``None``
            means :data:`LEGACY_GAME_VERSION`.

    Raises:
        ValueError: ``version`` is not a supported build.
    """
    key = version or LEGACY_GAME_VERSION
    try:
        return _ENGINE_WIKI_VERSIONS[key]
    except KeyError:
        raise ValueError(
            f"unsupported game version {key!r}; supported: {', '.join(SUPPORTED_GAME_VERSIONS)}"
        ) from None


def loc_dir_for(version: str | None = None, repo_root: Path | None = None) -> Path:
    """The vendored English localization dir for game build ``version``.

    The default build reads ``localization_eng/`` (unchanged); every other
    build reads its sibling ``localization_eng_<ver>/``, vendored by
    ``scripts/vendor_localization.py`` as a MERGE over the default copy (the
    newer build wins its keys, keys only the older build had are retained).

    Args:
        version: A supported build; ``None`` means the default.
        repo_root: The ``sts2-pycli`` root (default: this checkout's).
    """
    key = version or LEGACY_GAME_VERSION
    engine_wiki_version(key)  # validate
    root = repo_root if repo_root is not None else _REPO_ROOT
    if key == LEGACY_GAME_VERSION:
        return root / "localization_eng"
    return root / f"localization_eng_{key}"


def available_game_versions(repo_root: Path | None = None) -> list[str]:
    """The supported builds this checkout can DISPLAY, oldest first.

    The default build is always listed; any other needs its vendored
    ``localization_eng_<ver>/`` (a public snapshot ships without it), or
    every card and relic would render as a raw loc key. Rust twin:
    ``data::embedded_versions``.
    """
    return [
        v
        for v in SUPPORTED_GAME_VERSIONS
        if v == LEGACY_GAME_VERSION or loc_dir_for(v, repo_root).is_dir()
    ]


def _highest_version(paths: list[Path]) -> Path:
    """Pick the path whose parent dir name has the highest ``v<a>.<b>.<c>``."""
    return max(paths, key=lambda p: tuple(int(n) for n in _VER_NUMS.findall(p.parent.name)) or (0,))


def _engine_manifest(wiki_data_dir: Path, version: str | None = None) -> Path | None:
    """Return the engine-pinned ``<ver>/cards.json`` under ``wiki_data_dir``.

    Prefers the build's own wiki dir (:func:`engine_wiki_version`). For the
    DEFAULT build only, falls back to the highest version present if the
    pinned build's data isn't on disk (e.g. a trimmed deploy); a non-default
    build never borrows another build's manifest — it would describe the
    wrong rules."""
    wiki_ver = engine_wiki_version(version)
    pinned = wiki_data_dir / wiki_ver / "cards.json"
    if pinned.is_file():
        return pinned
    if wiki_ver != _ENGINE_WIKI_VERSION:
        return None
    manifests = list(wiki_data_dir.glob("*/cards.json"))
    if not manifests:
        return None
    return _highest_version(manifests)


def load_card_localization(
    repo_loc_json: Path, wiki_data_dir: Path | None, *, version: str | None = None
) -> dict[str, str]:
    """Build a ``loc_key -> string`` map for card titles and descriptions.

    Args:
        repo_loc_json: Path to in-repo ``localization_eng/cards.json`` (a flat
            ``{"KEY.title": "...", "KEY.description": "..."}`` dict).
        wiki_data_dir: Optional path to the wiki's ``data/`` dir holding
            versioned ``<ver>/cards.json`` manifests.
        version: The game build to describe (default: the default build).

    Returns:
        Mapping such as ``{"PREPARED.title": "Prepared",
        "PREPARED.description": "Draw {Cards:diff()} ..."}``. The engine-pinned
        wiki manifest wins; in-repo entries only fill keys it has no value for.
    """
    loc: dict[str, str] = {}
    # The engine-pinned wiki manifest goes in first and WINS. Both sources now
    # describe v0.107.1 — ``localization_eng/`` was re-vendored from the pinned
    # build's pck on 2026-08-24 (``scripts/vendor_localization.py``) — so this
    # ordering no longer decides which BUILD you read. It still matters because
    # the manifest carries the derived fields the raw dump has no notion of
    # (``description_plain``, per-var base values, keywords), and keeping one
    # winner keeps template and plain text from coming out of different files.
    #
    # It mattered a great deal before that: the dump used to override the
    # manifest while being a v0.103.2 asset, which is how 29 cards rendered the
    # wrong build's rules — Pact's End read "Can only be played if you have 3 or
    # more cards in your Exhaust Pile" (a v0.103.2 ``IsPlayable`` gate) while the
    # engine implemented the v0.107.1 ``OnPlay`` gate, letting you play it for no
    # effect. Bug report ``wb-66e36e372eef4bea``.
    if wiki_data_dir and wiki_data_dir.is_dir():
        manifest = _engine_manifest(wiki_data_dir, version)
        if manifest is not None:
            try:
                cards: Any = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cards = []
            if isinstance(cards, dict):
                cards = cards.get("cards", [])
            for c in cards or []:
                key = c.get("loc_key")
                if not key:
                    continue
                if c.get("title"):
                    loc.setdefault(f"{key}.title", c["title"])
                # The card's owning character drives cross-character art lookup
                # (a Defect card held in an Ironclad run resolves under the
                # Defect atlas folder) — see ``assets.AssetResolver.card``.
                if c.get("character"):
                    loc.setdefault(f"{key}.character", c["character"])
                # Regent star cost. A second, independent cost resource the game
                # draws as its own gem below the energy orb; only the 21 cards
                # with a ``CanonicalStarCost`` have one, and it reaches us only
                # here — no backend puts it on the wire.
                if c.get("star_cost") is not None:
                    loc.setdefault(f"{key}.star_cost", str(c["star_cost"]))
                if c.get("x_star_cost"):
                    loc.setdefault(f"{key}.star_cost", "X")
                if c.get("description_template"):
                    loc.setdefault(f"{key}.description", c["description_template"])
                # Pre-resolved plain descriptions (numbers already substituted).
                # The Rust backend can't fill a card's ``stats`` outside combat
                # (it has no instantiated card to read DynamicVars from), so these
                # are the display fallback for deck / reward / shop card views —
                # see ``server._enrich``'s card handling.
                if c.get("description_plain"):
                    loc.setdefault(f"{key}.description_plain", c["description_plain"])
                if c.get("upgraded_description_plain"):
                    loc.setdefault(
                        f"{key}.upgraded_description_plain", c["upgraded_description_plain"]
                    )
                # Keyword-only upgrades (``add_keyword`` / ``remove_keyword``,
                # e.g. Afterimage gains Innate). ~48 cards upgrade by keyword
                # alone and ship NO ``upgraded_description_plain``, so an
                # upgrade preview built from the descriptions alone would read
                # identical to the base card — this is the delta to show
                # alongside it. Keyed ``<KEY>.upgrade_keywords``.
                deltas = []
                for up in c.get("upgrades") or []:
                    keyword = up.get("keyword")
                    if not keyword:
                        continue
                    action = up.get("action")
                    if action == "add_keyword":
                        deltas.append(f"+{keyword}")
                    elif action == "remove_keyword":
                        deltas.append(f"-{keyword}")
                if deltas:
                    loc.setdefault(f"{key}.upgrade_keywords", ", ".join(deltas))
                # The card's own keywords (C# ``CardModel.CanonicalKeywords``).
                # The game PRINTS these on the card face — a Tremble reads
                # "Apply 3 Vulnerable." then "Exhaust." — so they are display
                # data, not just mechanics; see :func:`keyword_card_lines`.
                if c.get("keywords"):
                    loc.setdefault(f"{key}.keywords", ",".join(c["keywords"]))
                # Per-variable base values (``<KEY>.var.<type>`` / ``.uvar.<type>``)
                # so a combat card whose live ``stats`` cover only damage/block can
                # still fill its other vars (Vulnerable, Increase, …) with base
                # values while keeping the wire's live (possibly scaled) damage —
                # see ``server._enrich``'s card handling and :func:`card_var_value`.
                for var in c.get("vars") or []:
                    vtype = (var.get("type") or "").lower()
                    if not vtype:
                        continue
                    if var.get("base_value") is not None:
                        loc.setdefault(f"{key}.var.{vtype}", str(var["base_value"]))
                    if var.get("upgraded_value") is not None:
                        loc.setdefault(f"{key}.uvar.{vtype}", str(var["upgraded_value"]))
    if repo_loc_json.is_file():
        try:
            repo_entries: Any = json.loads(repo_loc_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            repo_entries = {}
        # Gap-filler only. It is still the sole source for the ~33 cards the
        # v0.107.1 manifest dropped, for every ``<KEY>.selectionScreenPrompt``,
        # and for the whole map when no wiki checkout is present (``record.py``).
        for key, value in (repo_entries or {}).items():
            loc.setdefault(key, value)
    return loc


def _engine_versioned_subdir(
    wiki_data_dir: Path, subdir: str, version: str | None = None
) -> Path | None:
    """Return the engine-pinned ``<ver>/<subdir>`` dir under ``wiki_data_dir``.

    Prefers the build's own wiki dir; for the DEFAULT build only, falls back
    to the highest version present if the pinned build's subdir isn't on disk.
    A non-default build has no such fallback: ``data/v0.111.0/`` ships only
    the aggregates, and borrowing v0.107.1's per-item files would fill its
    gaps with the other build's text."""
    wiki_ver = engine_wiki_version(version)
    pinned = wiki_data_dir / wiki_ver / subdir
    if pinned.is_dir():
        return pinned
    if wiki_ver != _ENGINE_WIKI_VERSION:
        return None
    dirs = [p for p in wiki_data_dir.glob(f"*/{subdir}") if p.is_dir()]
    if not dirs:
        return None
    return _highest_version(dirs)


def _class_name_to_loc_key(class_name: str) -> str:
    """``LargeCapsule`` -> ``LARGE_CAPSULE`` (matches the wiki extractor)."""
    result = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", class_name)
    result = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", result)
    return result.upper()


def load_owned_localization(
    wiki_data_dir: Path | None, *, version: str | None = None
) -> dict[str, str]:
    """Build a ``<LOC_KEY>.description_plain`` map for relics and potions.

    The regex-extracted aggregates (``<ver>/relics.json`` / ``potions.json``,
    ``just extract`` in sts2-wiki) carry ``description_resolved`` — the template
    with every DynamicVar substituted by the item's base value. The Rust backend
    can't attach a relic/potion ``vars`` dict (it has no instantiated model to
    read DynamicVars from), so a template like ``"heal {Heal} HP"`` would render
    its bare variable names. These pre-resolved strings are the display fallback
    — the relic/potion analogue of a card's ``description_plain`` — keyed by the
    loc key derived from the item's ``class_name``.

    The aggregates come FIRST so the Python server and the Rust port (whose
    ``wiki_owned.json`` bundle is built from the same aggregates) agree
    byte-for-byte; the LLM-extracted per-item files (``<ver>/relics/*.json``),
    when present, only fill keys the aggregates lack.

    Args:
        wiki_data_dir: Path to the wiki's ``data/`` dir holding versioned
            ``<ver>/{relics,potions}.json`` (and optionally the per-item dirs).
        version: The game build to describe (default: the default build).

    Returns:
        Mapping such as ``{"BURNING_BLOOD.description_plain":
        "At the end of combat, heal 6 HP."}``. Empty when nothing is found.
    """
    loc: dict[str, str] = {}
    if not (wiki_data_dir and wiki_data_dir.is_dir()):
        return loc
    wiki_ver = engine_wiki_version(version)
    for subdir in ("relics", "potions"):
        agg = wiki_data_dir / wiki_ver / f"{subdir}.json"
        if agg.is_file():
            try:
                rows: Any = json.loads(agg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rows = []
            if isinstance(rows, list):
                for row in sorted(
                    (r for r in rows if isinstance(r, dict)), key=lambda r: r.get("class_name", "")
                ):
                    class_name = row.get("class_name")
                    desc = row.get("description_resolved")
                    if not class_name or not desc:
                        continue
                    loc.setdefault(f"{_class_name_to_loc_key(class_name)}.description_plain", desc)
        items_dir = _engine_versioned_subdir(wiki_data_dir, subdir, version)
        if items_dir is None:
            continue
        for path in sorted(items_dir.glob("*.json")):
            try:
                item: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            class_name = item.get("class_name")
            desc = item.get("description")
            if not class_name or not desc:
                continue
            loc.setdefault(f"{_class_name_to_loc_key(class_name)}.description_plain", desc)
    return loc


def named_description_vars(
    needed: set[str], supplied: set[str], loc: dict[str, str]
) -> dict[str, str]:
    """Fill any ``{SomethingDescription}`` placeholder from the item it names.

    A few event options splice another model's rules text into their own:
    Tea Master's ``"Pay {BoneTeaCost} Gold. {BoneTeaDescription}"`` prints the
    Bone Tea RELIC's description. That value lives in a different loc table, so
    the engine cannot supply it — but the name says exactly where to look:
    strip ``Description`` and the rest is the item's loc key
    (``BoneTea`` -> ``BONE_TEA``).

    Args:
        needed: Lowercased var names the template requires.
        supplied: Lowercased var names already provided.
        loc: A loc map carrying ``<KEY>.description_plain`` (see
            :func:`load_owned_localization`).

    Returns:
        ``{var_name: text}`` for the ones that resolved; missing items are
        left out, so the caller still falls back rather than printing a bare
        variable name.
    """
    out: dict[str, str] = {}
    for name in needed - supplied:
        if not name.endswith("description"):
            continue
        # The template's var name is CamelCase; ``needed`` is lowercased, so
        # match case-insensitively against the loc keys we hold.
        stem = name[: -len("description")]
        if not stem:
            continue
        for key, value in loc.items():
            if not key.endswith(".description_plain"):
                continue
            if key[: -len(".description_plain")].replace("_", "").lower() == stem:
                out[name] = value
                break
    return out


def plain_owned_description(item_id: str | None, loc: dict[str, str]) -> str | None:
    """Return a relic/potion's pre-resolved plain description, or None.

    Args:
        item_id: The item's loc-key stem (e.g. ``"BURNING_BLOOD"``).
        loc: A loc map carrying ``<KEY>.description_plain`` entries (see
            :func:`load_owned_localization`).
    """
    if not item_id:
        return None
    return loc.get(f"{item_id}.description_plain")


def load_card_meta(
    wiki_data_dir: Path | None, *, version: str | None = None
) -> dict[str, dict[str, str]]:
    """Build a ``loc_key -> {"card_type", "rarity", "cost", "upgraded_cost"}``
    map from the wiki manifest.

    The headless engine ships every hand/deck card with a generic ``type:
    "card"`` and no rarity, so the real category (Attack/Skill/Power) and rarity
    (Basic/Common/Uncommon/Rare/…) must be looked up by the card's id, which
    equals the manifest ``loc_key``. Returns an empty map when no manifest is
    found (cards then keep the engine's generic type and get no rarity tint).

    Args:
        wiki_data_dir: Path to the wiki's ``data/`` dir holding versioned
            ``<ver>/cards.json`` manifests.
        version: The game build to describe (default: the default build).
    """
    meta: dict[str, dict[str, str]] = {}
    if not (wiki_data_dir and wiki_data_dir.is_dir()):
        return meta
    manifest = _engine_manifest(wiki_data_dir, version)
    if manifest is None:
        return meta
    try:
        cards: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta
    if isinstance(cards, dict):
        cards = cards.get("cards", [])
    for c in cards or []:
        key = c.get("loc_key")
        if not key:
            continue
        entry: dict[str, str] = {}
        if c.get("type"):
            entry["card_type"] = c["type"]
        if c.get("rarity"):
            entry["rarity"] = c["rarity"]
        cost = c.get("energy_cost")
        if cost is not None:
            # Display-ready: the manifest uses -1 for variable (X) cost.
            entry["cost"] = "X" if int(cost) < 0 else str(int(cost))
        # The cost the card prints once upgraded, for the 57 cards whose
        # upgrade is (or includes) a cost reduction. Only these carry the key;
        # for every other card the upgraded cost IS ``cost``. Outside combat no
        # backend puts a cost on the wire at all, so without this an upgraded
        # Stampede+ drew its un-upgraded 2 energy in the deck view and in the
        # rest-site upgrade preview.
        ucost = c.get("upgraded_cost")
        if ucost is not None:
            entry["upgraded_cost"] = "X" if int(ucost) < 0 else str(int(ucost))
        if entry:
            meta[key] = entry
    return meta


# ── Card keywords (C# ``CardKeywordOrder`` / ``CardKeywordExtensions``) ──────
#
# A keyword is not just mechanics: the game PRINTS it on the card face.
# ``CardModel.GetDescriptionForPile`` builds the text as
# ``<before-lines> + <description> + <after-lines>``, one keyword per line,
# each rendered by ``CardKeyword.GetCardText()`` as
# ``"[gold]" + <title> + "[/gold]" + "."``. Nothing in the manifest's
# ``description_template`` / ``description_plain`` carries them, so without
# this the workbench dropped every keyword line — Tremble read "Apply 3
# Vulnerable." where the game reads "Apply 3 Vulnerable." / "Exhaust."
# (210 of 576 v0.107.1 cards carry at least one keyword).

# ``CardKeywordOrder.beforeDescription``, in DECLARATION order. The game walks
# this array and ``list.Insert(0, ...)`` on each hit, so what a card actually
# shows is this order REVERSED: Ascender's Bane prints "Unplayable." above
# "Ethereal.", not the other way round.
_KEYWORDS_BEFORE_DESCRIPTION = ("Ethereal", "Sly", "Retain", "Innate", "Unplayable")
# ``CardKeywordOrder.afterDescription`` — appended below the description, in
# this order.
_KEYWORDS_AFTER_DESCRIPTION = ("Exhaust", "Eternal")


def load_card_keyword_localization(loc_dir: Path) -> dict[str, str]:
    """Build a ``card_keyword.<Name> -> display title`` map.

    Reads the game's own ``localization_eng/card_keywords.json`` (flat
    ``{"EXHAUST.title": "Exhaust", ...}``). The titles are what the card face
    prints. Falls back to the keyword's own name for anything the file lacks,
    so a missing/unreadable dump degrades to English names rather than blanks.

    Args:
        loc_dir: The ``localization_eng/`` directory.
    """
    titles: dict[str, str] = {}
    path = loc_dir / "card_keywords.json"
    entries: Any = {}
    if path.is_file():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
    for keyword in _KEYWORDS_BEFORE_DESCRIPTION + _KEYWORDS_AFTER_DESCRIPTION:
        title = (entries or {}).get(f"{keyword.upper()}.title") or keyword
        titles[f"card_keyword.{keyword}"] = title
    return titles


# Afflictions that PRINT on the card face. ``AfflictionModel.HasExtraCardText``
# is the game's own switch and only four afflictions set it (``Bound.cs``,
# ``Galvanized.cs``, ``Ringing.cs``, ``Tainted.cs`` v0.107.1); the others
# (Entangled, Hexed, Smog, ...) show only their overlay art, so printing their
# text — ``afflictions.json`` carries an ENTANGLED entry the game never reads —
# would put words on a card the real one does not have.
_AFFLICTIONS_WITH_CARD_TEXT = ("BOUND", "GALVANIZED", "RINGING", "TAINTED")


def load_affliction_localization(loc_dir: Path) -> dict[str, str]:
    """Build an ``affliction.<ID>.<field> -> string`` map.

    Reads the game's own ``localization_eng/afflictions.json`` (flat
    ``{"BOUND.title": "Bound", "BOUND.extraCardText": "Bound", ...}``) for the
    afflictions the game prints on a card face. Keys are namespaced with an
    ``affliction.`` prefix so they cannot collide with a card/relic loc key.

    Args:
        loc_dir: The ``localization_eng/`` directory.
    """
    path = loc_dir / "afflictions.json"
    entries: Any = {}
    if path.is_file():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
    out: dict[str, str] = {}
    for aff in _AFFLICTIONS_WITH_CARD_TEXT:
        for field in ("title", "extraCardText"):
            value = (entries or {}).get(f"{aff}.{field}")
            if value:
                out[f"affliction.{aff}.{field}"] = value
    return out


# The nine `RestSiteOption` subclasses the game ships
# (`MegaCrit.Sts2.Core.Entities.RestSite/`). Their `OptionId` is what the
# engine's `RestSiteOption::option_id()` returns and what keys both the
# plaque art (`ui/rest_site/option_<id>.png`) and the loc rows
# (`RestSiteOption.Title` = `rest_site_ui`/`OPTION_<ID>.name`).
_REST_SITE_OPTION_IDS = (
    "HEAL",
    "SMITH",
    "DIG",
    "LIFT",
    "COOK",
    "CLONE",
    "HATCH",
    "KINDLE",
    "MEND",
)


def load_rest_site_localization(loc_dir: Path) -> dict[str, str]:
    """Build a ``rest_site.OPTION_<ID>.name -> display title`` map.

    Reads the game's own ``localization_eng/rest_site_ui.json`` (flat
    ``{"OPTION_HEAL.name": "Rest", ...}``). Keys are namespaced with a
    ``rest_site.`` prefix so they cannot collide with a card/relic loc key.
    Every known option id gets an entry — falling back to the title-cased id —
    so an option the dump lacks still renders a word rather than a raw key.

    Args:
        loc_dir: The ``localization_eng/`` directory.
    """
    path = loc_dir / "rest_site_ui.json"
    entries: Any = {}
    if path.is_file():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = {}
    out: dict[str, str] = {}
    for opt in _REST_SITE_OPTION_IDS:
        out[f"rest_site.OPTION_{opt}.name"] = (entries or {}).get(
            f"OPTION_{opt}.name"
        ) or opt.title()
    return out


def affliction_card_line(affliction: str | None, loc: dict[str, str]) -> str | None:
    """The extra-card-text line an afflicted card prints, markup included.

    ``CardModel.GetDescriptionForPile`` appends
    ``"[purple]" + Affliction.DynamicExtraCardText + "[/purple]"`` under the
    description, which is how a Bound card says it is Bound before you try to
    play it. Returns None for an unafflicted card, and for an affliction the
    game shows no text for.

    ``{Amount}`` is left in the template for the caller to resolve alongside the
    description (Galvanized: "Take {Amount} damage."), so both the plain and the
    marked-up variants go through the same renderer.

    Args:
        affliction: The affliction's name as either backend ships it —
            ``"Bound"`` (the Rust wire's type name) or the C# backend's
            localized title, which is the same string in English.
        loc: The loc map, carrying the ``affliction.`` keys that
            :func:`load_affliction_localization` adds.
    """
    if not affliction:
        return None
    text = loc.get(f"affliction.{affliction.strip().upper()}.extraCardText")
    return f"[purple]{text}[/purple]" if text else None


def card_keywords(card_id: str | None, upgraded: bool, loc: dict[str, str]) -> set[str]:
    """The keywords a card actually has right now (C# ``CardModel.Keywords``).

    Canonical keywords plus/minus whatever the card's ``OnUpgrade`` adds or
    removes — Afterimage gains Innate, Apparition loses Ethereal. Both halves
    come from the wiki manifest via :func:`load_card_localization`.

    Instance keywords granted mid-combat by another model (Hex's Ethereal,
    Bookmark's Retain) are NOT included: they never reach this layer, which
    sees only the card's id and upgrade state.
    """
    if not card_id:
        return set()
    keywords = {k for k in (loc.get(f"{card_id}.keywords") or "").split(",") if k}
    if upgraded:
        for delta in (loc.get(f"{card_id}.upgrade_keywords") or "").split(","):
            delta = delta.strip()
            if delta.startswith("+"):
                keywords.add(delta[1:])
            elif delta.startswith("-"):
                keywords.discard(delta[1:])
    return keywords


def keyword_card_lines(
    card_id: str | None, upgraded: bool, loc: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Return ``(lines above the description, lines below it)`` for a card.

    Each line is one keyword rendered the way the game renders it, markup
    included: ``"[gold]Exhaust[/gold]."``. Feed them through
    ``render.resolve_card_text`` alongside the description so the plain and
    rich variants strip/keep the markup consistently.
    """
    have = card_keywords(card_id, upgraded, loc)
    if not have:
        return [], []

    def line(keyword: str) -> str:
        return f"[gold]{loc.get(f'card_keyword.{keyword}', keyword)}[/gold]."

    before = [line(k) for k in reversed(_KEYWORDS_BEFORE_DESCRIPTION) if k in have]
    after = [line(k) for k in _KEYWORDS_AFTER_DESCRIPTION if k in have]
    return before, after


def resolve(value: str | None, loc: dict[str, str]) -> str:
    """Return ``loc[value]`` if ``value`` is a known loc key, else ``value``."""
    if not value:
        return value or ""
    return loc.get(value, value)


def plain_description(card_id: str | None, upgraded: bool, loc: dict[str, str]) -> str | None:
    """Return a card's pre-resolved plain description, or None if unavailable.

    These come from the wiki manifest (``description_plain`` /
    ``upgraded_description_plain``) and already have the template variables
    substituted with the card's base values. Used as a display fallback for card
    views the backend couldn't attach numeric ``stats`` to (the Rust backend
    outside combat). Falls back to the unupgraded text when the upgraded variant
    is missing.

    Args:
        card_id: The card's loc-key stem (e.g. ``"THUNDERCLAP"``).
        upgraded: Whether the card is upgraded.
        loc: The card loc map from :func:`load_card_localization`.

    Returns:
        Plain text such as ``"Deal 4 damage..."``, or None when no plain
        description is known for ``card_id``.
    """
    if not card_id:
        return None
    if upgraded:
        return loc.get(f"{card_id}.upgraded_description_plain") or loc.get(
            f"{card_id}.description_plain"
        )
    return loc.get(f"{card_id}.description_plain")


def card_var_value(
    card_id: str | None, var_name: str, upgraded: bool, loc: dict[str, str]
) -> str | None:
    """Return a card variable's base value for display, or None if unknown.

    Used to fill a combat card's *missing* template variables (everything the
    backend's live ``stats`` don't cover) from the wiki manifest's per-variable
    base values, so the live (possibly scaled) damage/block from the wire is kept
    while vars like Vulnerable / Increase still resolve to a number instead of a
    bare name.

    The manifest stores power variables under the power name with the ``Power``
    suffix stripped (``VigorPower`` -> ``Vigor``), while templates reference the
    full ``{VigorPower}``; try the bare name first, then the stripped form.

    Args:
        card_id: The card's loc-key stem (e.g. ``"BASH"``).
        var_name: The template variable name to resolve (case-insensitive).
        upgraded: Prefer the upgraded base value when available.
        loc: The card loc map from :func:`load_card_localization`.
    """
    if not card_id or not var_name:
        return None
    name = var_name.lower()
    candidates = [name]
    if name.endswith("power"):
        candidates.append(name[: -len("power")])
    for cand in candidates:
        if upgraded:
            val = loc.get(f"{card_id}.uvar.{cand}")
            if val is not None:
                return val
        val = loc.get(f"{card_id}.var.{cand}")
        if val is not None:
            return val
    return None


# A loc key the game/backend leaks when our bundled loc lacks an entry
# (e.g. the Neow "Winged Boots" relic -> ``WINGED_BOOTS.title``).
_LOC_KEY_RE = re.compile(r"^[A-Z0-9][A-Z0-9_]*\.(title|name|description|flavor)$")


def _prettify_loc_key(value: str) -> str:
    """Turn a leaked loc key like ``WINGED_BOOTS.title`` into ``Winged Boots``."""
    return value.split(".", 1)[0].replace("_", " ").title()


# A dotted loc key (e.g. ``NEOW.pages.INITIAL.description``) that our bundle
# lacks an entry for — distinguishable from real prose because it has no spaces
# and is all key-ish path segments. Used to blank unresolved event bodies.
_DOTTED_KEY_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+$")


def looks_like_unresolved_key(value: str | None) -> bool:
    """True for a leaked dotted loc key (``X.pages.Y.description``), not prose."""
    return bool(value) and bool(_DOTTED_KEY_RE.match(value or ""))


def resolve_name(name: str | None, loc: dict[str, str]) -> str:
    """Resolve a display name, prettifying any still-unresolved loc key."""
    resolved = resolve(name, loc)
    return _prettify_loc_key(resolved) if _LOC_KEY_RE.match(resolved) else resolved


# A rest-site option label (``rest_site.OPTION_SMITH.name``, built by the Rust
# backend / ``game.py``). ``_LOC_KEY_RE`` only matches an UPPERCASE stem, so
# these slip past ``resolve_name`` and used to be printed verbatim.
_REST_OPTION_KEY_RE = re.compile(r"^rest_site\.OPTION_([A-Za-z0-9_]+)\.name$")


def _resolve_option_label(raw: str, loc: dict[str, str]) -> str:
    """Resolve an option label, prettifying an unresolved rest-site key.

    The generic prettifier keeps only the segment before the first dot, which
    would turn every rest-site option alike into "Rest Site"; the option id is
    the informative part, so prettify that instead.
    """
    resolved = resolve(raw, loc)
    match = _REST_OPTION_KEY_RE.match(resolved)
    return (
        match.group(1).replace("_", " ").lower().title() if match else resolve_name(resolved, loc)
    )


def resolve_state_names(state: dict[str, Any] | None, loc: dict[str, str]) -> dict[str, Any]:
    """Return a copy of ``state`` with card/enemy/option display names resolved.

    The log renderers (``render.describe_action`` / ``statediff.diff_states``)
    read the raw ``name``/``title`` fields a backend stored. The Rust backend
    emits localization keys there (``STRIKE_IRONCLAD.title``) and relies on
    display-time enrichment, so a history/event log built straight off the raw
    state would show those keys verbatim. Resolve them so labels read
    ``Play Strike`` regardless of backend. The C# backend already ships display
    names, so this is a no-op there.
    """
    if not state:
        return state or {}
    out = copy.deepcopy(state)
    # ``relics``/``potions`` at the top level are the shop / treasure offerings
    # (the persistent player trays are handled below under ``player``).
    card_lists = (
        "hand",
        "cards",
        "draw_pile",
        "discard_pile",
        "exhaust_pile",
        "enemies",
        "relics",
        "potions",
    )
    for key in card_lists:
        for item in out.get(key) or []:
            if isinstance(item, dict) and "name" in item:
                item["name"] = resolve_name(item.get("name"), loc)
    player = out.get("player") or {}
    # Deck + the persistent relic/potion trays (the diff renderer reads these
    # for "+Relic <name>" / "+Potion <name>" lines).
    for tray in ("deck", "relics", "potions"):
        for item in player.get(tray) or []:
            if isinstance(item, dict) and "name" in item:
                item["name"] = resolve_name(item.get("name"), loc)
    for opt in out.get("options") or []:
        if not isinstance(opt, dict):
            continue
        # ``title`` is the generic screens' label; ``name`` is what a REST SITE
        # carries, and ``describe_action`` prefers ``name`` over ``title`` --
        # resolving only ``title`` logged the raw key ("Choose:
        # rest_site.OPTION_SMITH.name").
        for field in ("title", "name"):
            if isinstance(opt.get(field), str) and opt[field]:
                opt[field] = _resolve_option_label(opt[field], loc)
    return out


def load_monster_hp_ranges(
    wiki_data_dir: Path | None, *, version: str | None = None
) -> dict[str, dict[str, int]]:
    """``MonsterKind -> {min, max, min_asc, max_asc, hp_ascension}`` HP ranges.

    ``Creature.SetUniqueMonsterHpValue`` rolls each monster's max HP inside
    ``[MinInitialHp, MaxInitialHp]``, and ``NCombatRoom.RandomizeEnemyScalesAndHues``
    then sizes the creature by WHERE in that band it landed. Reproducing the
    game's sizing therefore needs the band, which no wire carries -- only the
    rolled result. The wiki manifest has it per monster class.

    The manifest carries both the base band (``min_hp_base``/``max_hp_base``) and
    the one that applies from ``hp_ascension`` upward (``min_hp``/``max_hp``);
    both are kept so the caller can pick by the run's ascension.

    Args:
        wiki_data_dir: Path to the wiki's ``data/`` dir holding versioned
            ``<ver>/monsters.json`` manifests.
        version: The game build to describe (default: the default build).

    Returns:
        Mapping keyed by the manifest's ``class_name``, which IS the
        ``MonsterKind``. Empty when no manifest is on disk (callers then skip
        the sizing, which is what the workbench did before it had this).
    """
    out: dict[str, dict[str, int]] = {}
    if not (wiki_data_dir and wiki_data_dir.is_dir()):
        return out
    cards = _engine_manifest(wiki_data_dir, version)
    manifest = (cards.parent / "monsters.json") if cards is not None else None
    if manifest is None or not manifest.is_file():
        return out
    try:
        with manifest.open(encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = row.get("class_name")
        asc_lo, asc_hi = row.get("min_hp"), row.get("max_hp")
        # The manifest omits a bound that equals its partner: a fixed-HP monster
        # carries `min_hp_base` alone (47 of 117 do), and a monster with no
        # ascension bump carries no base pair at all. Filling both from what IS
        # there keeps every monster in the map -- they roll no range, so they
        # take no jitter, but a MISSING key and a zero-width band are different
        # answers to "why is this one not sized".
        base_lo = row.get("min_hp_base")
        base_hi = row.get("max_hp_base")
        if not isinstance(base_lo, int):
            base_lo = asc_lo
        if not isinstance(base_hi, int):
            base_hi = base_lo
        if (
            not isinstance(kind, str)
            or not isinstance(base_lo, int)
            or not isinstance(base_hi, int)
        ):
            continue
        out[kind] = {
            "min": base_lo,
            "max": base_hi,
            "min_asc": asc_lo if isinstance(asc_lo, int) else base_lo,
            "max_asc": asc_hi if isinstance(asc_hi, int) else base_hi,
            "hp_ascension": row.get("hp_ascension")
            if isinstance(row.get("hp_ascension"), int)
            else 0,
        }
    return out


def build_workbench_loc(
    wiki_data_dir: Path | None,
    *,
    version: str | None = None,
    repo_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """The workbench server's display loc map + monster HP bands for a build.

    Reproduces ``workbench.server.create_app``'s merge order exactly (the Rust
    ``localization::build_loc`` twin is pinned to the same order):

    1. :func:`load_card_localization` (wiki manifest wins, in-repo dump fills);
    2. ``update`` with the card-keyword, affliction and rest-site tables;
    3. ``setdefault`` ``<id>.card_type`` / ``.rarity`` / ``.cost`` /
       ``.upgraded_cost`` from :func:`load_card_meta`;
    4. ``update`` with the raw ``relics.json`` then ``potions.json`` dumps;
    5. ``setdefault`` every ``characters.json`` string;
    6. ``setdefault`` every :func:`load_owned_localization` entry;
    7. ``setdefault`` every ``events.json`` entry.

    Args:
        wiki_data_dir: The wiki's ``data/`` dir (``None`` = no wiki checkout).
        version: The game build (default: :data:`LEGACY_GAME_VERSION`). Picks
            both the vendored loc dir (:func:`loc_dir_for`) and the wiki
            ``data/<ver>/`` manifests (:func:`engine_wiki_version`).
        repo_root: The ``sts2-pycli`` root (default: this checkout's).

    Returns:
        ``(loc, hp_ranges)`` — ``hp_ranges`` is
        :func:`load_monster_hp_ranges` for the same build.
    """
    loc_dir = loc_dir_for(version, repo_root)
    card_loc = load_card_localization(loc_dir / "cards.json", wiki_data_dir, version=version)
    hp_ranges = load_monster_hp_ranges(wiki_data_dir, version=version)
    card_loc.update(load_card_keyword_localization(loc_dir))
    card_loc.update(load_affliction_localization(loc_dir))
    card_loc.update(load_rest_site_localization(loc_dir))
    for key, meta in load_card_meta(wiki_data_dir, version=version).items():
        if meta.get("card_type"):
            card_loc.setdefault(f"{key}.card_type", meta["card_type"])
        if meta.get("rarity"):
            card_loc.setdefault(f"{key}.rarity", meta["rarity"])
        if meta.get("cost") is not None:
            card_loc.setdefault(f"{key}.cost", meta["cost"])
        if meta.get("upgraded_cost") is not None:
            card_loc.setdefault(f"{key}.upgraded_cost", meta["upgraded_cost"])
    for fname in ("relics.json", "potions.json"):
        path = loc_dir / fname
        if path.is_file():
            try:
                card_loc.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    characters_path = loc_dir / "characters.json"
    if characters_path.is_file():
        try:
            for key, value in json.loads(characters_path.read_text(encoding="utf-8")).items():
                if isinstance(value, str):
                    card_loc.setdefault(key, value)
        except (OSError, json.JSONDecodeError):
            pass
    for owned_key, owned_plain in load_owned_localization(wiki_data_dir, version=version).items():
        card_loc.setdefault(owned_key, owned_plain)
    events_path = loc_dir / "events.json"
    if events_path.is_file():
        try:
            for key, val in json.loads(events_path.read_text(encoding="utf-8")).items():
                card_loc.setdefault(key, val)
        except (OSError, json.JSONDecodeError):
            pass
    return card_loc, hp_ranges
