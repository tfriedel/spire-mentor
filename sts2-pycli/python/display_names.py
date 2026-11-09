"""Player-facing names for game ids, and the wire guard that keeps ids off the HUD.

The mod's payloads carry both spellings of most things: a card's ``id``
(``SHRUG_IT_OFF``) and its ``name`` (``Shrug It Off``). The advisor builds its
labels from whichever field it reaches first, and a field that is an id on one
surface (the shop's option ``label`` is the merchant entry's id) put
``SHRUG_IT_OFF`` on the HUD. Labels are built from display names where the
payload has them; this module is the rest:

* :func:`display_name` -- one id (any spelling: ``SHRUG_IT_OFF``,
  ``CARD.SHRUG_IT_OFF``, ``SHRUG_IT_OFF.title``) to the game's English name,
  from the vendored localization tables (the same files
  :mod:`sts2_hint.canonical_maps` reads, gzipped in the packaged advisor);
* :func:`humanize` -- every id-shaped token inside a display string replaced
  by its name (``remove:STRIKE_IRONCLAD`` -> ``Remove Strike``);
* :func:`sanitize_advice` / :func:`sanitize_coach` -- the final guard at the
  wire boundary, applied to every human-facing string the advisor sends, so a
  surface added later cannot leak an id either.

The takeover's cross-check (``SimAdvisor.ReconcileIndex``) compares
``advice.best.label`` against the labels the mod built for the screen, so
:func:`sanitize_advice` never rewrites a ``best.label`` that is the mod's own
spelling of an option on a cross-checked surface (see
:data:`CROSS_CHECKED_SURFACES`).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import runtime_paths

#: Localization tables and the key suffix holding each entry's display name,
#: in lookup order (a card and a relic sharing an id resolve as the card).
_TABLES = (
    ("cards", ".title"),
    ("relics", ".title"),
    ("potions", ".title"),
    ("monsters", ".name"),
    ("events", ".title"),
    ("enchantments", ".title"),
    ("powers", ".title"),
    ("orbs", ".title"),
    ("afflictions", ".title"),
    ("encounters", ".title"),
    ("characters", ".title"),
)

#: ``advice_for`` surfaces whose ``best.label`` the mod cross-checks against
#: its own option labels before clicking (``SimScreenDecision.OptionLabels``,
#: ``SimAdvisor.ReconcileIndex``; a mismatch re-anchors or refuses the click).
#: Shop, map and (non-Fake-Merchant) event screens send no labels to check.
CROSS_CHECKED_SURFACES = frozenset(
    {
        "card_reward",
        "card_select",
        "card_bundle",
        "boss_relic",
        "treasure",
        "rest_site",
        "potion_swap",
        "event",
    }
)


def _loc_dirs() -> list[Path]:
    """The newest vendored build first (dev checkout), the bundled one last."""
    root = runtime_paths.data_root()
    newer = sorted(root.glob("localization_eng_v*"), reverse=True)
    return [*newer, root / "localization_eng"]


@functools.cache
def _names() -> dict[str, str]:
    """``ID`` (upper, ``_``-separated) -> display name, over every table."""
    from sts2_hint.canonical_maps import _load_json

    out: dict[str, str] = {}
    for loc_dir in _loc_dirs():
        if not loc_dir.is_dir():
            continue
        for table, suffix in _TABLES:
            for key, value in _load_json(loc_dir / f"{table}.json").items():
                if not key.endswith(suffix) or "." in key[: -len(suffix)]:
                    continue
                stem = key[: -len(suffix)].upper()
                if not value or "{" in value:
                    continue
                out.setdefault(stem, value)
                if table == "powers" and stem.endswith("_POWER"):
                    out.setdefault(stem[: -len("_POWER")], value)
    return out


def _squash(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


@functools.cache
def _squashed() -> dict[str, str]:
    """The same names keyed without separators (``StrikeIronclad`` engine
    spelling, ``STRIKEIRONCLAD``)."""
    out: dict[str, str] = {}
    for stem, name in _names().items():
        out.setdefault(_squash(stem), name)
    return out


def _prettify(stem: str) -> str:
    return stem.replace("_", " ").strip().title()


def display_name(raw: Any) -> str:
    """The English name for a game id in any spelling; the id prettified when
    no table knows it (``SOME_NEW_CARD`` -> ``Some New Card``)."""
    text = str(raw or "").strip()
    if not text:
        return ""
    upgraded = text.endswith("+")
    text = text.rstrip("+")
    parts = text.split(".")
    if len(parts) > 1 and parts[-1].lower() in ("title", "name"):
        parts = parts[:-1]
    stem = parts[-1]
    name = _names().get(stem.upper()) or _squashed().get(_squash(stem))
    if name is None:
        if re.fullmatch(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+", stem):
            # Engine CamelCase (`ShrugItOff`): split the words.
            stem = re.sub(r"(?<!^)(?=[A-Z])", " ", stem)
        name = _prettify(stem)
    return name + ("+" if upgraded else "")


#: An id-shaped token: SCREAMING_SNAKE (``SHRUG_IT_OFF``), optionally behind a
#: registry prefix (``CARD.``) or before a loc suffix (``.title``), or a lone
#: all-caps word (``BASH``) -- the latter replaced only when a table knows it,
#: so prose capitals (``NOT a recommendation``, ``MCTS``) stay.
_UPPER_ID = re.compile(
    r"(?<![\w./\\-])"
    r"(?:(?:CARD|RELIC|POTION|POWER|MONSTER|EVENT|ENCHANTMENT|ORB)\.)?"
    r"(?P<stem>[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*)"
    r"(?:\.(?:title|name))?"
    r"(?![\w/\\])"
)

#: A machine verb spelled in snake_case (``remove_card``, ``leave_shop``,
#: ``card_select``): its words.
_SNAKE = re.compile(r"(?<![\w./\\-])[a-z]+(?:_[a-z0-9]+)+(?![\w./\\])")

#: ``remove:STRIKE_IRONCLAD`` (the run policy's shop-removal candidates).
_VERB_PREFIX = re.compile(r"\b(remove|buy|upgrade|transform|discard):(?=\S)")


def _upper_token(m: re.Match[str]) -> str:
    whole = m.group(0)
    stem = m.group("stem")
    if "_" not in stem and whole == stem:
        # A bare all-caps word: only a known id is replaced.
        name = _names().get(stem)
        return name if name is not None and len(stem) >= 3 else whole
    return display_name(whole)


def humanize(text: Any) -> Any:
    """``text`` with every id-shaped token replaced by its display name.

    Non-strings pass through unchanged. Idempotent on display text: a name
    has no ``_`` and is not all caps.
    """
    if not isinstance(text, str) or not text:
        return text
    out = _VERB_PREFIX.sub(lambda m: m.group(1).capitalize() + " ", text)
    out = _UPPER_ID.sub(_upper_token, out)
    return _SNAKE.sub(lambda m: m.group(0).replace("_", " "), out)


def card_name(card: Any) -> str:
    """A wire Card object's name: its ``name``, else its id's display name."""
    if not isinstance(card, dict):
        return "?"
    name = card.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    label = display_name(card.get("id")) or "?"
    level = card.get("upgrade_level")
    return label + ("+" if (level or card.get("upgraded")) else "")


def shop_option_label(option: dict[str, Any]) -> str:
    """A ``shop`` option as the player reads it: the item's name.

    The option's own ``label`` is the merchant entry's id (``SHRUG_IT_OFF``,
    ``remove_card``); the display name is in ``item`` (PROTOCOL.md ``shop``).
    """
    action = option.get("action")
    if action == "leave_shop":
        return "Leave shop"
    if action == "remove_card":
        return "Remove a card"
    item = option.get("item")
    if isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        if item.get("id"):
            return display_name(item["id"])
    return humanize(str(option.get("label") or action or "?"))


def potion_swap_label(option: dict[str, Any], belt: list[Any] | None = None) -> str:
    """A full-belt potion swap option (PROTOCOL.md v5.2) in words.

    The mod's ``label`` is ``decline`` / ``discard <POTION_ID>``; the reward
    is named in ``reward`` and the held potion in the belt (``run_state.potions``
    by ``discard_slot``).
    """
    reward = option.get("reward") if isinstance(option.get("reward"), dict) else {}
    new = reward.get("name") or display_name(reward.get("id")) or "the new potion"
    slot = option.get("discard_slot")
    if slot is None:
        return f"Keep your potions (leave {new})"
    held = next(
        (
            p
            for p in belt or []
            if isinstance(p, dict) and p.get("slot") == slot and not p.get("empty")
        ),
        None,
    )
    if held is not None:
        old = held.get("name") or display_name(held.get("id"))
    else:
        raw = str(option.get("label") or "")
        old = display_name(raw.split(" ", 1)[1]) if raw.startswith("discard ") else f"slot {slot}"
    return f"Discard {old} for {new}"


def _mod_labels(state: dict[str, Any] | None) -> set[str]:
    """The option labels the mod itself sent (case-folded): on a cross-checked
    surface these are what ``best.label`` is compared against."""
    out: set[str] = set()
    for key in ("options", "cards", "bundles"):
        for o in (state or {}).get(key) or []:
            if isinstance(o, dict):
                for field in ("label", "name"):
                    v = o.get(field)
                    if isinstance(v, str) and v.strip():
                        out.add(v.strip().casefold())
    return out


def sanitize_advice(
    advice: dict[str, Any], advice_for: str | None = None, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The final guard on every human-facing string of an ``advice`` object.

    Rewrites, in place: ``note``, ``hand_back``, ``instinct_unavailable``,
    ``pv[]``, ``options[].label``, ``plan[].label`` and ``best.label`` --
    except a ``best.label`` the takeover cross-checks: on a
    :data:`CROSS_CHECKED_SURFACES` screen a label that IS one of the mod's own
    option labels stays as sent, or F6 / F7 would refuse the click.

    Returns:
        ``advice`` (the same object).
    """
    for key in ("note", "hand_back", "instinct_unavailable"):
        if isinstance(advice.get(key), str):
            advice[key] = humanize(advice[key])
    if isinstance(advice.get("pv"), list):
        advice["pv"] = [humanize(s) for s in advice["pv"]]
    for key in ("options", "plan"):
        for o in advice.get(key) or []:
            if isinstance(o, dict) and isinstance(o.get("label"), str):
                o["label"] = humanize(o["label"])
    best = advice.get("best")
    if isinstance(best, dict) and isinstance(best.get("label"), str):
        label = best["label"]
        keep = advice_for in CROSS_CHECKED_SURFACES and label.strip().casefold() in _mod_labels(
            state
        )
        if not keep:
            best["label"] = humanize(label)
    return advice


#: Coach record keys holding display text (strings or lists of strings).
_COACH_TEXT_KEYS = frozenset({"label", "you", "ai", "why", "hand", "name", "note"})


def sanitize_coach(record: Any) -> Any:
    """The same guard over a ``coach`` record (PROTOCOL.md §5 ``coach``),
    recursively: the turn lines, the choice labels, a recap's context."""
    if isinstance(record, dict):
        for key, value in record.items():
            if key in _COACH_TEXT_KEYS and isinstance(value, str):
                record[key] = humanize(value)
            elif key in _COACH_TEXT_KEYS and isinstance(value, list):
                record[key] = [
                    humanize(v) if isinstance(v, str) else sanitize_coach(v) for v in value
                ]
            else:
                sanitize_coach(value)
    elif isinstance(record, list):
        for v in record:
            sanitize_coach(v)
    return record
