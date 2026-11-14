"""The sandbox position: a hand-editable combat position and its engine twin.

The engine's own format for a fight is the *wire dict*
(``sts2_rs.state_to_dict`` / ``state_from_dict``). It is a reconstruction
format: card uids and ``stable_idx``, per-enemy ``uid``, an
``intent_history``, RNG cursors. Nothing a person wants to type. The sandbox
edits a compact ``simlab-position/1`` document instead and this module is the
only place that knows how the two relate:

* :func:`position_from_wire` — engine wire → position (what an import or a
  workbench hand-off produces; everything the position cannot carry is listed
  in the returned notes, never silently dropped).
* :func:`wire_from_position` — position → engine wire (what a build submits).
* :func:`build_position` — the round trip through the engine. A position is
  "built" when ``state_from_dict`` accepts the wire it produces; the
  normalised position handed back is what the engine actually holds.

Every field name in the position is stable API for the SPA and for pasted
JSON, so changes here are schema changes.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from enemy_labels import enemy_labels

SCHEMA = "simlab-position/1"

#: PascalCase names the engine's scenario builder accepts, in the game's order.
CHARACTERS: tuple[str, ...] = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")

#: Base orb values by type — what a hand-authored orb starts at (Focus 0).
#: The engine folds Focus in itself on a fresh channel; on a rebuilt board the
#: wire carries the already-scaled numbers and those win.
ORB_BASE: dict[str, tuple[int, int]] = {
    "Lightning": (3, 8),
    "Frost": (2, 5),
    "Dark": (6, 6),
    "Plasma": (1, 2),
    "Glass": (3, 6),
}

#: Phases a position can be edited at. A finished fight has nothing to play.
_TERMINAL_PHASES = frozenset({"Victory", "Defeat"})

_STATUS_PREFIX = re.compile(r"^(status|curse):", re.IGNORECASE)


class PositionError(ValueError):
    """A position the sandbox cannot turn into a fight. The message is user-facing."""


def _camel_to_upper_snake(name: str) -> str:
    """``FuzzyWurmCrawler`` → ``FUZZY_WURM_CRAWLER`` (the engine's own rule)."""
    out = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    out = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", out)
    return out.upper()


def normalize_card_id(raw: str) -> str:
    """Canonical registry spelling for a card id off any wire dialect.

    The live bridge spells statuses ``status:Slimed`` and curses
    ``curse:Clumsy``; the engine registry names them ``SLIMED`` / ``CLUMSY``.
    Plain ids pass through upper-cased.
    """
    stripped = _STATUS_PREFIX.sub("", raw.strip())
    if stripped.upper() == stripped:
        return stripped
    return _camel_to_upper_snake(stripped)


def wire_class(character: str) -> str:
    """``Ironclad`` → the lowercase ``player.class`` the wire parser requires."""
    return character.lower()


def character_from_class(cls: str) -> str:
    """``ironclad`` → ``Ironclad`` (the position's PascalCase spelling)."""
    for name in CHARACTERS:
        if name.lower() == cls.lower():
            return name
    raise PositionError(f"unknown character {cls!r} (expected one of {', '.join(CHARACTERS)})")


# ---------------------------------------------------------------------------
# Position construction helpers
# ---------------------------------------------------------------------------


def empty_position(character: str = "Ironclad", *, ascension: int = 0) -> dict[str, Any]:
    """A blank but well-formed position: no cards, no enemies, full HP."""
    return {
        "schema": SCHEMA,
        "character": character_from_class(character),
        "ascension": int(ascension),
        "turn": 1,
        "isElite": False,
        "encounter": None,
        "seed": 1,
        "player": {
            "hp": 80,
            "maxHp": 80,
            "block": 0,
            "energy": 3,
            "maxEnergy": 3,
            "stars": 0,
            "gold": 99,
            "powers": [],
            "relics": [],
            "potions": [],
            "orbSlots": 0,
            "orbs": [],
            "osty": None,
            "cardsPlayedThisTurn": None,
        },
        "piles": {"hand": [], "draw": [], "discard": [], "exhaust": []},
        "monsters": [],
    }


def _card_row(wire_card: dict[str, Any], notes: list[str], where: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "card": normalize_card_id(str(wire_card.get("id", ""))),
        "upgrade": 1 if wire_card.get("upgraded") else 0,
        # The instance's own cost when the wire says; null means "whatever the
        # registry prints", which is what a hand-authored card wants.
        "cost": wire_card.get("base_cost", wire_card.get("cost_override")),
    }
    dropped = [
        key
        for key in (
            "enchantment",
            "affliction",
            "flags",
            "combat_vars",
            "base_replay_count",
            "tinker_time_rider",
        )
        if wire_card.get(key)
    ]
    if dropped:
        notes.append(
            f"{where}: {row['card']} loses {', '.join(dropped)} (not carried by the position)."
        )
    return row


def _power_rows(wire_powers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows = []
    for p in wire_powers or []:
        rows.append(
            {
                "id": str(p.get("name") or p.get("id") or ""),
                "amount": int(p.get("amount", 0)),
                # The engine emits `counter: 0` for every power; only a live
                # counter (Slow, Orbit, Crimson Mantle) is worth carrying.
                "counter": p.get("counter") or None,
            }
        )
    return rows


def position_from_wire(
    wire: dict[str, Any],
    *,
    encounter: str | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Translate an engine wire dict into a position.

    Args:
        wire: An ``sts2_rs.state_to_dict``-shaped dict.
        encounter: Encounter label to stamp on the position (the wire has none).
        seed: Seed to record; falls back to the wire's ``rng.seed``.

    Returns:
        ``(position, notes)`` — the notes name everything the wire carried that
        the position does not, so the user reads what a rebuild will differ on.

    Raises:
        PositionError: When the wire is not a combat board at a decision point.
    """
    if not isinstance(wire, dict) or "player" not in wire or "deck" not in wire:
        raise PositionError("not an engine combat wire (no player / deck block)")
    phase = str(wire.get("phase") or "PlayerPlayPhase")
    if phase in _TERMINAL_PHASES:
        raise PositionError(f"that fight is already over ({phase.lower()}); nothing to play from")
    notes: list[str] = []
    player = wire["player"]
    deck = wire["deck"]
    rng = wire.get("rng") if isinstance(wire.get("rng"), dict) else {}

    if deck.get("play_pile"):
        raise PositionError(
            "a card is mid-resolution (play pile is not empty); copy the state at a decision point"
        )
    for key in ("pinned_top", "pinned_bottom"):
        if deck.get(key):
            notes.append(f"Draw-pile {key.replace('_', ' ')} pins are not carried; the order is.")
    if deck.get("deferred_draw_order"):
        notes.append("The deferred draw-order flag is not carried.")

    pos = empty_position(character_from_class(str(player.get("class", "ironclad"))))
    pos["ascension"] = int((rng or {}).get("ascension") or 0)
    pos["turn"] = int(wire.get("round", wire.get("turn", 1)) or 1)
    pos["isElite"] = bool(wire.get("is_elite"))
    # A non-default build's wire names it (``$.game_version``); the position
    # carries it on so a rebuild plays the same build (GH #877).
    if isinstance(wire.get("game_version"), str):
        pos["game_version"] = wire["game_version"]
    pos["encounter"] = encounter
    pos["seed"] = int(seed if seed is not None else (rng or {}).get("seed") or 0) or 1

    pp = pos["player"]
    pp["hp"] = int(player["hp"])
    pp["maxHp"] = int(player["max_hp"])
    pp["block"] = int(player.get("block", 0))
    pp["energy"] = int(player.get("energy", 0))
    pp["maxEnergy"] = int(player.get("max_energy", 3))
    pp["stars"] = int(player.get("stars", 0) or 0)
    pp["gold"] = int(player.get("gold", 0) or 0)
    pp["powers"] = _power_rows(player.get("powers"))
    pp["relics"] = [
        {"id": str(r["id"]), "counter": int(r.get("counter", 0) or 0)}
        for r in player.get("relics") or []
        if isinstance(r, dict)
    ]
    pp["potions"] = [str(p) for p in player.get("potions") or [] if p]
    pp["orbSlots"] = int(player.get("orb_slots", 0) or 0)
    pp["orbs"] = [
        {
            "type": str(o.get("type", "Lightning")),
            "passive": int(o.get("passive", 0) or 0),
            "evoke": int(o.get("evoke", 0) or 0),
        }
        for o in player.get("orbs") or []
        if isinstance(o, dict)
    ]
    osty = player.get("osty")
    pp["osty"] = (
        {
            "alive": bool(osty.get("alive", True)),
            "hp": int(osty.get("hp", 0)),
            "maxHp": int(osty.get("max_hp", 1)),
            "block": int(osty.get("block", 0)),
        }
        if isinstance(osty, dict)
        else None
    )
    cptt = player.get("cards_played_this_turn")
    pp["cardsPlayedThisTurn"] = [int(x) for x in cptt] if isinstance(cptt, list) else None

    piles = pos["piles"]
    for pile, key in (
        ("hand", "hand"),
        ("draw", "draw_pile"),
        ("discard", "discard_pile"),
        ("exhaust", "exhaust_pile"),
    ):
        piles[pile] = [_card_row(c, notes, pile) for c in deck.get(key) or []]

    monsters = []
    for m in wire.get("encounter") or []:
        if m.get("is_dead") or m.get("is_escaped"):
            # `state_from_dict` drops corpses anyway; a position lists the fight
            # that is still on.
            continue
        row = {
            "monster": str(m["kind"]),
            "hp": int(m["hp"]),
            "maxHp": int(m["max_hp"]),
            "block": int(m.get("block", 0)),
            "powers": _power_rows(m.get("powers")),
            "intent": m.get("move_state"),
        }
        if row["intent"] is None and m.get("intents"):
            notes.append(
                f"{row['monster']}: the wire telegraphs a move without naming it; "
                "the intent will be re-rolled unless you pick one."
            )
        for key in ("stolen_cards", "fabricator_last_spawned", "curse_of_knowledge_counter"):
            if m.get(key):
                notes.append(f"{row['monster']}: {key} is not carried by the position.")
        monsters.append(row)
    pos["monsters"] = monsters
    return pos, notes


# ---------------------------------------------------------------------------
# Position → wire
# ---------------------------------------------------------------------------

MovesOf = Callable[[str, int], list[dict[str, Any]]]
RollIntents = Callable[
    [str, list[dict[str, Any]], int, int, list[str | None]], list[dict[str, Any]]
]


def _monster_moves(kind: str, ascension: int) -> list[dict[str, Any]]:
    import sts2_rs

    return list(sts2_rs.monster_moves(kind, ascension))


def _strip_move_segment(move_id: str) -> str:
    """``THRASH_MOVE_2`` → ``THRASH_2``; ``WRIGGLE_MOVE`` → ``WRIGGLE``."""
    stripped = move_id.removesuffix("_MOVE")
    return stripped.replace("_MOVE_", "_")


def _int(value: Any, what: str, *, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise PositionError(f"{what} must be a whole number, got {value!r}") from exc
    if minimum is not None and out < minimum:
        raise PositionError(f"{what} must be at least {minimum}, got {out}")
    return out


def _wire_card(row: dict[str, Any], idx: int) -> dict[str, Any]:
    card_id = normalize_card_id(str(row.get("card") or row.get("id") or ""))
    if not card_id:
        raise PositionError(f"card #{idx + 1} has no id")
    out: dict[str, Any] = {
        "id": card_id,
        "stable_idx": idx,
        "kind": "card",
        "upgraded": bool(_int(row.get("upgrade", 0) or 0, "upgrade", minimum=0)),
    }
    cost = row.get("cost")
    if cost is not None:
        out["base_cost"] = _int(cost, f"cost of {card_id}")
    return out


def _wire_powers(rows: list[dict[str, Any]] | None, where: str) -> list[dict[str, Any]]:
    out = []
    for r in rows or []:
        pid = str(r.get("id") or r.get("name") or "").strip().upper()
        if not pid:
            raise PositionError(f"{where}: a power has no id")
        entry: dict[str, Any] = {"name": pid, "amount": _int(r.get("amount", 1), f"{pid} amount")}
        if r.get("counter") is not None:
            entry["counter"] = _int(r["counter"], f"{pid} counter")
        out.append(entry)
    return out


def _roll_intents(
    character: str,
    monsters: list[dict[str, Any]],
    ascension: int,
    seed: int,
    forced: list[str | None],
    game_version: str | None = None,
) -> list[dict[str, Any]]:
    """Ask the engine what each enemy opens with, honouring any forced moves.

    One fresh fight is built with the same lineup; each enemy's ``move_state``
    and telegraph are read off its wire. That is the only way to get an
    opening that respects the AI's random/positional branches (Nibbit's
    BUTT/SLICE/HISS depend on the slot, ScrollOfBiting rotates its starter).
    """
    import sts2_rs

    from simlab.specs import _MONSTER_BASE

    kinds = [str(m["monster"]) for m in monsters]
    cs = sts2_rs.build_scenario_state(
        seed=int(seed) or 1,
        character=character,
        deck=["STRIKE_IRONCLAD"],
        hp=10,
        max_hp=10,
        relics=[],
        encounter=_MONSTER_BASE,
        ascension=int(ascension),
        potions=[],
        monsters=kinds,
        move_states=forced,
        game_version=game_version,
    )
    return list(sts2_rs.state_to_dict(cs)["encounter"])


def position_game_version(pos: dict[str, Any]) -> str | None:
    """The game build a position plays (GH #877): its ``game_version`` field,
    ``None`` for the default build. Written only for a non-default build, so
    every v0.107.1 position is unchanged. Rust twin: ``position_game_version``.

    Raises:
        PositionError: The field names a build the engine does not support.
    """
    from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS

    raw = pos.get("game_version")
    if raw is None or raw == LEGACY_GAME_VERSION:
        return None
    if raw not in SUPPORTED_GAME_VERSIONS:
        raise PositionError(
            f"unsupported game_version {raw!r} (supported: {', '.join(SUPPORTED_GAME_VERSIONS)})"
        )
    return str(raw)


def wire_from_position(
    pos: dict[str, Any],
    *,
    moves_of: MovesOf = _monster_moves,
    roll_intents: RollIntents = _roll_intents,
) -> dict[str, Any]:
    """Translate a position into the engine wire dict ``state_from_dict`` takes.

    Args:
        pos: A ``simlab-position/1`` document.
        moves_of: ``(monster kind, ascension) -> monster_moves rows``; injected
            so the pure mapping is testable without the engine.
        roll_intents: What rolls an unnamed enemy's opening move; injected for
            the same reason. Only called when some enemy has ``intent: null``.

    Raises:
        PositionError: For a shape the engine could never accept. Registry
            errors (an unknown card) are left to the engine, whose message
            names the JSON path.
    """
    if not isinstance(pos, dict):
        raise PositionError("position must be a JSON object")
    if pos.get("schema") not in (None, SCHEMA):
        raise PositionError(f"unsupported schema {pos.get('schema')!r} (expected {SCHEMA})")
    character = character_from_class(str(pos.get("character") or "Ironclad"))
    game_version = position_game_version(pos)
    ascension = _int(pos.get("ascension", 0) or 0, "ascension", minimum=0)
    turn = _int(pos.get("turn", 1) or 1, "turn", minimum=1)
    seed = _int(pos.get("seed", 1) or 1, "seed", minimum=0)
    player = pos.get("player") or {}
    piles = pos.get("piles") or {}

    max_hp = _int(player.get("maxHp", 80), "max HP", minimum=1)
    hp = _int(player.get("hp", max_hp), "HP", minimum=0)
    if hp > max_hp:
        raise PositionError(f"HP ({hp}) is above max HP ({max_hp})")
    max_energy = _int(player.get("maxEnergy", 3), "max energy", minimum=0)
    energy = _int(player.get("energy", max_energy), "energy", minimum=0)

    wire_player: dict[str, Any] = {
        "class": wire_class(character),
        "hp": hp,
        "max_hp": max_hp,
        "block": _int(player.get("block", 0) or 0, "block", minimum=0),
        "energy": energy,
        "max_energy": max_energy,
        "energy_regen_per_turn": max_energy,
        "gold": _int(player.get("gold", 0) or 0, "gold", minimum=0),
        "relics": [
            {
                "id": str(r["id"]).strip().upper(),
                "counter": _int(r.get("counter", 0) or 0, "relic counter"),
            }
            if isinstance(r, dict)
            else {"id": str(r).strip().upper(), "counter": 0}
            for r in player.get("relics") or []
        ],
        "powers": _wire_powers(player.get("powers"), "player"),
        "potions": [str(p).strip().upper() for p in player.get("potions") or [] if p],
    }
    stars = _int(player.get("stars", 0) or 0, "stars", minimum=0)
    if stars:
        wire_player["stars"] = stars
    orb_slots = _int(player.get("orbSlots", 0) or 0, "orb slots", minimum=0)
    orbs = player.get("orbs") or []
    if orb_slots or orbs:
        if len(orbs) > orb_slots:
            raise PositionError(f"{len(orbs)} orbs channelled but only {orb_slots} orb slots")
        wire_orbs = []
        for i, o in enumerate(orbs):
            otype = str(o.get("type", "Lightning")).capitalize()
            base = ORB_BASE.get(otype)
            if base is None:
                raise PositionError(f"orb #{i + 1}: unknown orb type {otype!r}")
            wire_orbs.append(
                {
                    "index": i,
                    "type": otype,
                    "passive": _int(o.get("passive", base[0]), "orb passive"),
                    "evoke": _int(o.get("evoke", base[1]), "orb evoke"),
                }
            )
        wire_player["orbs"] = wire_orbs
        wire_player["orb_slots"] = orb_slots
    osty = player.get("osty")
    if isinstance(osty, dict):
        wire_player["osty"] = {
            "alive": bool(osty.get("alive", True)),
            "hp": _int(osty.get("hp", 0), "Osty HP", minimum=0),
            "max_hp": _int(osty.get("maxHp", 1), "Osty max HP", minimum=1),
            "block": _int(osty.get("block", 0) or 0, "Osty block", minimum=0),
        }
    cptt = player.get("cardsPlayedThisTurn")
    if isinstance(cptt, list) and len(cptt) == 5:
        wire_player["cards_played_this_turn"] = [_int(x, "cards played") for x in cptt]

    wire_deck: dict[str, Any] = {}
    idx = 0
    for pile, key in (
        ("hand", "hand"),
        ("draw", "draw_pile"),
        ("discard", "discard_pile"),
        ("exhaust", "exhaust_pile"),
    ):
        rows = piles.get(pile) or []
        cards = []
        for row in rows:
            cards.append(_wire_card(row, idx))
            idx += 1
        wire_deck[key] = cards
    if idx == 0:
        raise PositionError("the position has no cards in any pile")

    monsters = pos.get("monsters") or []
    if not monsters:
        raise PositionError("the position has no enemies")
    # Resolve every enemy's opening move. Named moves come from the move table;
    # unnamed ones are rolled by the engine for this exact lineup.
    forced: list[str | None] = [(str(m["intent"]) if m.get("intent") else None) for m in monsters]
    if roll_intents is _roll_intents and game_version is not None:
        # The engine rolls the openers in the position's own build.
        roll_intents = functools.partial(_roll_intents, game_version=game_version)
    rolled = (
        roll_intents(character, monsters, ascension, seed, forced)
        if any(f is None for f in forced)
        else []
    )
    encounter = []
    for i, m in enumerate(monsters):
        kind = str(m.get("monster") or "").strip()
        if not kind:
            raise PositionError(f"enemy #{i + 1} has no monster")
        max_m_hp = _int(m.get("maxHp", 1), f"{kind} max HP", minimum=1)
        m_hp = _int(m.get("hp", max_m_hp), f"{kind} HP", minimum=1)
        if m_hp > max_m_hp:
            raise PositionError(f"{kind}: HP ({m_hp}) is above max HP ({max_m_hp})")
        move_state: str | None = forced[i]
        intents: list[dict[str, Any]] = []
        if move_state is not None:
            table = {row["id"]: row for row in moves_of(kind, ascension)}
            move = table.get(move_state)
            if move is None:
                # The live bridge spells C# ids (`WRIGGLE_MOVE`, `THRASH_MOVE_2`)
                # where the engine has `WRIGGLE` / `THRASH_2`; match the way
                # `MonsterAi::force_current_state_by_name_relaxed` does.
                want = _strip_move_segment(move_state)
                for mid, row in table.items():
                    if _strip_move_segment(mid) == want:
                        move, move_state = row, mid
                        break
            if move is None:
                raise PositionError(
                    f"{kind}: {move_state!r} is not one of its moves "
                    f"({', '.join(sorted(table)) or 'none known'})"
                )
            if move.get("intent"):
                intents = [dict(move["intent"])]
        elif i < len(rolled):
            move_state = rolled[i].get("move_state")
            intents = [dict(it) for it in rolled[i].get("intents") or []]
        entry: dict[str, Any] = {
            "kind": kind,
            "uid": i + 1,
            "hp": m_hp,
            "max_hp": max_m_hp,
            "block": _int(m.get("block", 0) or 0, f"{kind} block", minimum=0),
            "powers": _wire_powers(m.get("powers"), kind),
            "intents": intents,
            "intent_history": [],
            "is_dead": False,
            "is_escaped": False,
        }
        if move_state:
            entry["move_state"] = move_state
        encounter.append(entry)

    wire: dict[str, Any] = {
        "wire_version": 1,
        "phase": "PlayerPlayPhase",
        "round": turn,
        "turn": turn,
        "side": "Player",
        "is_elite": bool(pos.get("isElite")),
        "combat_id": 0,
        "intent_amounts": "base",
        "player": wire_player,
        "deck": wire_deck,
        "encounter": encounter,
        "rng": {"seed": seed, "ascension": ascension, "run_seed": seed & 0xFFFF_FFFF},
    }
    # Named only for a non-default build, exactly as ``state_to_dict`` does.
    if game_version is not None:
        wire["game_version"] = game_version
    return wire


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


@dataclass
class BuiltPosition:
    """What the engine holds for a position, plus what it changed on the way."""

    position: dict[str, Any]
    wire: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def _engine_error(exc: Exception) -> PositionError:
    return PositionError(f"the engine rejected this position: {exc}")


def build_position(pos: dict[str, Any]) -> BuiltPosition:
    """Run a position through the engine and hand back its normalised form.

    The wire returned is the engine's own ``state_to_dict`` of the rebuilt
    board (so it carries everything a sweep needs, including the enemy AI's
    resolved move), and the position is re-derived from that wire — which is
    how a user sees a rolled intent or a clamped value the moment it happens.

    Raises:
        PositionError: When the shape is wrong or the engine refuses it; the
            message carries the engine's JSON path for registry misses.
    """
    import sts2_rs

    wire_in = wire_from_position(pos)
    try:
        cs = sts2_rs.state_from_dict(wire_in)
        wire_out = sts2_rs.state_to_dict(cs)
    except PositionError:
        raise
    except Exception as exc:
        raise _engine_error(exc) from exc
    built, notes = position_from_wire(
        wire_out, encounter=pos.get("encounter"), seed=int(pos.get("seed", 1) or 1)
    )
    # The wire the engine emits has no seed of its own; keep the user's so a
    # rebuild from the SAME position is the same fight.
    built["encounter"] = pos.get("encounter")
    return BuiltPosition(position=built, wire=wire_out, notes=notes)


def new_position(
    *,
    character: str,
    encounter: str | None = None,
    monsters: list[str] | None = None,
    ascension: int = 0,
    seed: int = 1,
    hp: int | None = None,
    max_hp: int | None = None,
    relics: list[str] | None = None,
    potions: list[str] | None = None,
    deck: list[str] | None = None,
    starter: dict[str, Any] | None = None,
    game_version: str | None = None,
) -> BuiltPosition:
    """A fresh fight from turn 1, as the game would deal it.

    Args:
        character: PascalCase character name.
        encounter: Registry encounter id, or ``None`` with ``monsters``.
        monsters: Explicit lineup (monster kinds) instead of an encounter.
        ascension: 0–10.
        seed: Combat seed — shuffle, enemy HP rolls and openers.
        hp / max_hp / relics / potions / deck: Overrides; each defaults to the
            character's starter (``starter`` rows from the catalog).
        starter: ``{"hp", "deck", "relic"}`` for the character; taken from
            :mod:`simlab.catalog` when not given.
        game_version: The game build to deal it in (``None`` = the default;
            GH #877, the workbench selector's pick).

    Raises:
        PositionError: On an unknown encounter/character or an empty lineup.
    """
    import sts2_rs

    from simlab.specs import _MONSTER_BASE

    character = character_from_class(character)
    if starter is None:
        from simlab.catalog import starter_for

        starter = starter_for(character)
    base_hp = int(starter.get("hp", 80))
    max_hp = int(max_hp or base_hp)
    hp = int(hp if hp is not None else max_hp)
    deck_ids = list(deck) if deck else list(starter.get("deck", []))
    relic_ids = (
        list(relics) if relics is not None else [starter["relic"]] if starter.get("relic") else []
    )
    lineup = [str(m) for m in monsters or [] if m]
    if not encounter and not lineup:
        raise PositionError("pick an encounter or name at least one monster")
    try:
        cs = sts2_rs.build_scenario_state(
            seed=int(seed),
            character=character,
            deck=deck_ids,
            hp=hp,
            max_hp=max_hp,
            relics=relic_ids,
            encounter=encounter or _MONSTER_BASE,
            ascension=int(ascension),
            potions=list(potions or []),
            monsters=lineup or None,
            game_version=game_version,
        )
        wire = sts2_rs.state_to_dict(cs)
    except Exception as exc:
        raise _engine_error(exc) from exc
    label = encounter or ("monsters:" + ",".join(lineup))
    pos, notes = position_from_wire(wire, encounter=label, seed=int(seed))
    return BuiltPosition(position=pos, wire=wire, notes=notes)


# ---------------------------------------------------------------------------
# Pasted text → position
# ---------------------------------------------------------------------------


def _looks_like_position(value: Any) -> bool:
    return isinstance(value, dict) and (
        value.get("schema") == SCHEMA or ("piles" in value and "monsters" in value)
    )


def _looks_like_engine_wire(value: Any) -> bool:
    return isinstance(value, dict) and "wire_version" in value and "deck" in value


def position_from_paste(text: str) -> tuple[dict[str, Any], list[str], str]:
    """Turn whatever was pasted into a position.

    Accepted, in this order: a ``simlab-position/1`` document, an engine wire
    dict, and anything :func:`workbench.state_import.parse_pasted_state`
    reads — the SpireMentor **Copy state** payload, an advisor log line or a
    whole log (newest fight wins).

    Returns:
        ``(position, notes, source)`` where ``source`` names the dialect.

    Raises:
        PositionError: When nothing in the paste is a fight.
    """
    stripped = text.strip()
    if not stripped:
        raise PositionError("nothing pasted")
    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if _looks_like_position(parsed):
        pos = copy.deepcopy(parsed)
        pos.setdefault("schema", SCHEMA)
        return pos, [], "position"
    if _looks_like_engine_wire(parsed):
        pos, notes = position_from_wire(_normalized_wire(parsed))
        return pos, notes, "engine wire"

    from workbench.state_import import StateImportError, parse_pasted_state

    try:
        imported = parse_pasted_state(stripped)
    except StateImportError as exc:
        raise PositionError(str(exc)) from exc
    if imported.combat_wire is None:
        raise PositionError(
            "That state has no fight in it"
            + (": " + "; ".join(imported.warnings) if imported.warnings else "")
            + ". Copy the state during combat to load it here."
        )
    # Through the engine once, so the bridge's C# move ids (`SHRINKER_MOVE`)
    # come back as the engine's own (`SHRINK`) and the position names moves
    # the way its dropdown does. Run-level warnings (map, RNG cursors) are
    # about rebuilding the RUN and do not apply to a fight sandbox.
    pos, notes = position_from_wire(_normalized_wire(imported.combat_wire))
    pos["ascension"] = int(imported.run.ascension)
    return pos, notes, imported.source


def _normalized_wire(wire: dict[str, Any]) -> dict[str, Any]:
    """The engine's own emission of *wire*, or *wire* itself if it refuses."""
    import sts2_rs

    try:
        return sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
    except Exception:  # the raw wire still converts; the build reports the refusal
        return wire


# ---------------------------------------------------------------------------
# First moves and search
# ---------------------------------------------------------------------------

NameOf = Callable[[str, str], str]


def _default_name_of(kind: str, ident: str) -> str:
    return ident.replace("_", " ").title()


def action_label(
    action: dict[str, Any], wire: dict[str, Any], name_of: NameOf = _default_name_of
) -> str:
    """Human label for an engine legal action against the board it applies to."""
    kind = action.get("action")
    # The engine's ``target_index`` is an encounter SLOT, so it indexes the
    # whole encounter, dead slots included — not the alive-filtered list (which
    # shifts left past a dead enemy and named the wrong one). The name is the
    # board label over the enemies the board shows (neither dead nor escaped,
    # as ``position_from_wire`` keeps them) in slot order (``enemy_labels``):
    # "Two-Tailed Rat #2". Rust twin: ``simlab.rs::action_label``.
    alive = [
        (slot, name_of("monster", str(m["kind"])))
        for slot, m in enumerate(wire.get("encounter") or [])
        if not (m.get("is_dead") or m.get("is_escaped")) and m.get("kind") is not None
    ]
    labels = enemy_labels([{"name": n} for _, n in alive])

    def target() -> str:
        t = action.get("target_index")
        if t is None:
            return ""
        try:
            t = int(t)
        except (TypeError, ValueError):
            return f" → enemy {t}"
        for i, (slot, _) in enumerate(alive):
            if slot == t:
                return f" → {labels[i]}"
        return f" → enemy {t}"

    if kind == "play":
        hand = wire.get("deck", {}).get("hand") or []
        try:
            card = hand[int(action["card_index"])]
            name = name_of("card", normalize_card_id(str(card["id"]))) + (
                "+" if card.get("upgraded") else ""
            )
        except (IndexError, KeyError, ValueError, TypeError):
            name = f"card {action.get('card_index')}"
        return f"Play {name}{target()}"
    if kind == "end_turn":
        return "End turn"
    if kind in ("potion", "drink_potion"):
        potions = wire.get("player", {}).get("potions") or []
        try:
            name = name_of("potion", str(potions[int(action["slot"])]))
        except (IndexError, KeyError, ValueError, TypeError):
            name = f"potion {action.get('slot')}"
        return f"Drink {name}{target()}"
    if kind == "channel":
        return f"Channel {action.get('orb')}{target()}"
    if kind == "evoke":
        return f"Evoke ×{action.get('times', 1)}"
    if kind == "choose":
        return f"Choose #{action.get('index')}"
    return str(kind)


def first_moves(
    wire: dict[str, Any], *, name_of: NameOf = _default_name_of
) -> list[dict[str, Any]]:
    """Every legal move from this board and the position each one leads to.

    A move that opens a choice (a card asking which card to discard) is
    skipped with a note: the resulting board is mid-resolution and cannot be
    swept from.
    """
    import sts2_rs

    try:
        root = sts2_rs.state_from_dict(wire)
    except Exception as exc:
        raise _engine_error(exc) from exc
    out = []
    seen: dict[str, int] = {}
    for action in root.legal_actions():
        label = action_label(action, wire, name_of)
        # Two Defends in hand are two legal actions with one label; a sweep
        # keys its cells on the label, so the second one is numbered.
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} #{seen[label]}"
        clone = root.clone()
        clone.apply(action)
        note = None
        if clone.pending_choice() is not None or clone.has_pending_play():
            note = "opens a choice; cannot sweep from mid-resolution"
        terminal, _ = clone.terminal_value()
        after = sts2_rs.state_to_dict(clone)
        row: dict[str, Any] = {
            "label": label,
            "action": action,
            "terminal": bool(terminal),
            "note": note,
        }
        if note is None:
            try:
                pos, _ = position_from_wire(after)
            except PositionError as exc:
                pos = None
                row["note"] = str(exc)
            row["position"] = pos
            row["wire"] = after
        out.append(row)
    return out


def mcts_eval(
    wire: dict[str, Any],
    *,
    sims: int = 1000,
    seed: int = 0,
    depth: int = 200,
    threads: int = 1,
    hidden_info: str = "beliefdraw",
    name_of: NameOf = _default_name_of,
) -> dict[str, Any]:
    """Per-move MCTS verdict for the board — the sandbox's "search" card.

    Returns ``{"rows": [{label, action, win_prob, q, visits, is_best, …}],
    "sims", "seed"}``. Same engine call the workbench hint uses
    (``sts2_rs.mcts_recommend``), so the numbers agree with the board view.
    """
    import sts2_rs

    try:
        cs = sts2_rs.state_from_dict(wire)
        result = sts2_rs.mcts_recommend(
            cs,
            sims=int(sims),
            depth=int(depth),
            seed=int(seed),
            threads=max(1, int(threads)),
            hidden_info=hidden_info,
        )
    except Exception as exc:
        raise _engine_error(exc) from exc
    rows = []
    for entry in result.get("actions", []):
        action = entry["action"]
        rows.append(
            {
                "label": action_label(action, wire, name_of),
                "action": action,
                "win_prob": float(entry.get("p_win", 0.0)),
                "q": float(entry.get("q", entry.get("value", 0.0))),
                "visits": int(entry.get("visits", 0)),
                "is_best": bool(entry.get("is_best", False)),
                "low_confidence": bool(entry.get("low_confidence", False)),
            }
        )
    rows.sort(key=lambda r: (not r["is_best"], -r["win_prob"], -r["visits"]))
    return {"rows": rows, "sims": int(sims), "seed": int(seed)}
