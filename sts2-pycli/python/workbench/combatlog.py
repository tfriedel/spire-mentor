"""Render the engine's combat event log the way the game's log panel does.

The engine (``sts2_core::event_log``, exposed as ``RunManager.combat_log()``)
records ids, not names: ``STRIKE_IRONCLAD`` dealt 6 to ``Nibbit``. Localization
lives here, so this module is where a row becomes a line a human reads —
"The Ironclad dealt 6 damage to Nibbit using Strike".

Two things it deliberately does NOT do:

* It does not return a formatted string only. Each display row keeps the actor,
  target, amount and source as separate fields (plus a ``text`` fallback) so
  the UI can colour the names the way the game does without re-parsing a
  sentence.
* It does not reproduce the engine's C#-bug-compatible wording. ``text`` on the
  engine row says "discarded" for a DRAW because C#'s own
  ``CardDrawnEntry.Description`` does (that rendering exists to diff against
  the world-side ``[HISTTRACE]`` stream). A human-facing log says "drew".

By default the output is the GAME's log: hits, block, powers, moves, potions
and orbs. Card plays, draws, discards, exhausts and energy are recorded by the
engine but hidden — the game does not show them, and the card that mattered
already rides along as each row's ``source`` — except for an AUTO-played card
(Stampede, Havoc, Mayhem, …), whose start is shown as a ``card_auto_played`` row
(see :data:`AUTO_PLAYED_KIND`, GH #1155). Pass ``verbose=True`` for the
full tape, or ``keep={...}`` to un-hide named kinds and nothing else — which is
what the workbench's timeline slices do for ``power_flash``, a VFX signal the
board needs on the wire but no reader should ever see.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

#: Row kinds hidden unless ``verbose``, or named individually in ``keep``. The
#: card play frames are what give the visible rows their ``source``, so
#: dropping them loses nothing.
_VERBOSE_ONLY = frozenset(
    {
        "card_play_started",
        "card_play_finished",
        "card_drawn",
        "card_discarded",
        "card_exhausted",
        "energy_spent",
        # A power TRIGGERING is not a combat-log LINE -- the game's own log has
        # no such row. It rides this log only so the workbench can draw
        # `power_flash_vfx`, and a busy fight produces dozens, which would
        # bury everything the log exists to show.
        "power_flash",
    }
)


#: The display kind an auto-played ``card_play_started`` row is rendered as.
#: The workbench's step label names only the card the PLAYER chose (or "End
#: turn"), so an auto-play — Stampede at end of turn, Havoc / Cascade / Mayhem
#: / Hellraiser / Howl from Beyond / Distilled Chaos … — had no visible trace
#: except its consequences (GH #1155). Unlike every other card-play row this one
#: is never hidden. Every display row produced INSIDE an auto-play's frame also
#: carries ``"auto_played": True`` (absent, not False, elsewhere) so the history
#: ledger's compact mode can keep it. Mirrors ``combatlog.rs``.
AUTO_PLAYED_KIND = "card_auto_played"


def _prettify(raw: str) -> str:
    """``BUTT_MOVE`` -> ``Butt Move``. The last-resort display name."""
    return " ".join(w.capitalize() for w in raw.replace("-", "_").split("_") if w)


@dataclass(frozen=True)
class LogNames:
    """Id -> display-name resolvers, one per id namespace.

    Each defaults to :func:`_prettify`, so the module is usable (and testable)
    with no localization loaded at all; the workbench passes real resolvers.
    """

    card: Callable[[str], str] = _prettify
    monster: Callable[[str], str] = _prettify
    power: Callable[[str], str] = _prettify
    potion: Callable[[str], str] = _prettify
    orb: Callable[[str], str] = _prettify
    move: Callable[[str], str] = _prettify
    character: Callable[[str], str] = _prettify
    #: Extra fields merged into every display row (unused today; keeps the
    #: dataclass extensible without changing call sites).
    extra: dict[str, Any] = field(default_factory=dict)


def _creature(
    who: dict[str, Any] | None, names: LogNames
) -> tuple[str | None, str | None, int | None]:
    """``(display name, side, creature uid)`` for a serialized ``LogCreature``.

    The uid is the join key the VFX layer needs: it is `c.id.get()`, the same
    `CreatureUid` the board's encounter entries carry, so a per-hit row can be
    pinned to one board slot. Names cannot do that — two Twig Slimes in one
    encounter share a display name.
    """
    if not who:
        return None, None, None
    raw = str(who.get("id") or "?")
    uid = who.get("uid")
    uid = int(uid) if isinstance(uid, int) else None
    if who.get("kind") == "player":
        return names.character(raw), "player", uid
    return names.monster(raw), "enemy", uid


def _source_name(source: dict[str, Any] | None, names: LogNames) -> str | None:
    """Display name of the card / move a row is attributed to."""
    if not source:
        return None
    if source.get("kind") == "card":
        name = names.card(str(source.get("id") or "?"))
        return f"{name}+" if source.get("upgraded") else name
    return names.move(str(source.get("move_id") or "?"))


def _source_ids(source: dict[str, Any] | None) -> dict[str, Any]:
    """The raw, joinable identity of the card or move a row came from.

    `_source_name` renders this same dict for humans; the VFX binding table
    needs the STABLE id, which prettifying destroys ("BUG_STING" -> "Bug
    Sting"). Both are emitted: the UI keeps reading `source`, the effect
    layer reads these.

    A monster's ``move_id`` alone is not unique across monsters (e.g.
    ``THRASH`` exists on both Crusher and BowlbugNectar), so ``source_monster``
    carries the raw, un-prettified monster id alongside it as the other half
    of the VFX binding key. It is left uncased — PascalCase (``"Crusher"``) is
    the join key on both the Rust and the binding-table side.
    """
    if not source:
        return {
            "source_kind": None,
            "source_card": None,
            "source_upgraded": None,
            "source_move_id": None,
            "source_monster": None,
        }
    kind = source.get("kind")
    return {
        "source_kind": kind,
        "source_card": str(source["id"]) if kind == "card" and source.get("id") else None,
        "source_upgraded": bool(source.get("upgraded")) if kind == "card" else None,
        "source_move_id": str(source["move_id"]) if source.get("move_id") else None,
        "source_monster": str(source["monster"]) if source.get("monster") else None,
    }


def format_combat_log(
    rows: list[dict[str, Any]],
    names: LogNames | None = None,
    *,
    verbose: bool = False,
    keep: Collection[str] = frozenset(),
    player_id: str = "PLAYER",
) -> list[dict[str, Any]]:
    """Turn engine event rows into display rows.

    Args:
        rows: ``RunManager.combat_log()`` output, oldest first.
        names: id -> display-name resolvers. Defaults to prettified ids.
        verbose: keep ALL the card-movement / energy / play rows the game's own
            log hides — the whole of :data:`_VERBOSE_ONLY`, in one switch.
        keep: un-hide only these ``_VERBOSE_ONLY`` kinds, leaving the rest
            hidden. This is the knob for a consumer that needs ONE hidden kind
            and none of the others: the workbench's timeline slices need
            ``power_flash`` (the board's VFX layer reads it off the node) but
            emphatically not the card tape, which would bloat every saved run
            and force the client to re-hide six kinds it never asked for.
            Ignored for kinds that are not hidden to begin with.
        player_id: the run's character id (``"IRONCLAD"``), for the rows that
            are implicitly about the player and carry no creature of their own
            — draws, generated cards, potions, orbs, energy. Rows that DO name
            a creature (hits, block, powers) use the id the engine stamped on
            them, so those read right either way; without this the implicit
            ones would say "Player" while the explicit ones say "The Ironclad".

    Returns:
        One display row per kept event::

            {"kind", "round", "side", "text",
             "actor", "actor_side", "actor_uid", "target", "target_side", "target_uid",
             "amount", "noun", "source",
             # The raw, joinable identity behind ``source`` — see `_source_ids`.
             "source_kind", "source_card", "source_upgraded", "source_move_id",
             "source_monster",
             # ``damage`` rows only:
             "blocked", "hp_after"}

        Absent fields are ``None``. ``text`` is always present and always
        renders the whole line, so a UI that does not special-case a kind can
        still show it.
    """
    names = names or LogNames()
    out: list[dict[str, Any]] = []
    # One flag per open card-play frame: is this play (or a play that spawned
    # it) an auto-play? Mirrors the engine's own open-source stack.
    frames: list[bool] = []
    for row in rows:
        kind = str(row.get("kind") or "")
        auto_start = kind == "card_play_started" and bool(row.get("auto_played"))
        if kind == "monster_move":
            frames.clear()
        elif kind == "card_play_started":
            frames.append((frames[-1] if frames else False) or auto_start)
        in_auto = frames[-1] if frames else False
        if kind == "card_play_finished" and frames:
            frames.pop()
        shown = AUTO_PLAYED_KIND if auto_start else kind
        if shown in _VERBOSE_ONLY and not verbose and shown not in keep:
            continue
        display = _format_row(row, shown, names, player_id)
        if display is not None:
            if in_auto:
                display["auto_played"] = True
            out.append(display)
    return out


def _format_row(
    row: dict[str, Any], kind: str, names: LogNames, player_id: str = "PLAYER"
) -> dict[str, Any] | None:
    """Render one engine row, or ``None`` for a kind we do not display."""
    base: dict[str, Any] = {
        "kind": kind,
        "round": row.get("round"),
        "side": row.get("side"),
        # The verb this row reads with. The UI composes coloured spans from the
        # fields, and several kinds have two shapes ("applied … to" vs
        # "gained") that it cannot tell apart from actor/target alone — a
        # self-applied power has actor == target and still reads "applied".
        # Deciding it once, here, keeps that logic in one place.
        "verb": None,
        "actor": None,
        "actor_side": None,
        "actor_uid": None,
        "target": None,
        "target_side": None,
        "target_uid": None,
        "amount": None,
        "noun": None,
        "source": _source_name(row.get("source"), names),
        **_source_ids(row.get("source")),
    }
    # The player is the actor for every player-side row; rows that name their
    # own actor (damage, powers, moves) overwrite this below.
    player = names.character(player_id)

    def card_name(key: str = "card") -> str:
        name = names.card(str(row.get(key) or "?"))
        return f"{name}+" if row.get("upgraded") else name

    if kind == "damage":
        dealer, dealer_side, dealer_uid = _creature(row.get("dealer"), names)
        target, target_side, target_uid = _creature(row.get("receiver"), names)
        amount = int(row.get("unblocked") or 0)
        base |= {
            "actor": dealer,
            "actor_side": dealer_side,
            "actor_uid": dealer_uid,
            "target": target,
            "target_side": target_side,
            "target_uid": target_uid,
            "amount": amount,
            # The block split and resulting HP, for the VFX layer: a hit that
            # block ate entirely plays the game's `block_hit`, not the
            # target's material impact, and `hp_after == 0` is the only death
            # signal the log has (`event_log.rs` has no death entry).
            "blocked": int(row.get("blocked") or 0),
            "hp_after": row.get("hp_after"),
            "verb": "took" if dealer is None else "dealt",
        }
        if dealer is None:
            base["text"] = f"{target} took {amount} damage."
        else:
            using = f" using {base['source']}" if base["source"] else ""
            base["text"] = f"{dealer} dealt {amount} damage to {target}{using}."
        return base

    if kind == "block":
        actor, side, actor_uid = _creature(row.get("receiver"), names)
        amount = int(row.get("amount") or 0)
        base |= {
            "actor": actor,
            "actor_side": side,
            "actor_uid": actor_uid,
            "amount": amount,
            "noun": "Block",
            "verb": "gained",
        }
        base["text"] = f"{actor} gained {amount} Block."
        return base

    if kind == "power_flash":
        target, target_side, target_uid = _creature(row.get("creature"), names)
        power = names.power(str(row.get("power") or "?"))
        base |= {
            "target": target,
            "target_side": target_side,
            "target_uid": target_uid,
            "noun": power,
            "verb": "triggered",
        }
        base["text"] = f"{target}'s {power} triggered."
        return base

    if kind == "power":
        applier, applier_side, applier_uid = _creature(row.get("applier"), names)
        target, target_side, target_uid = _creature(row.get("receiver"), names)
        amount = int(row.get("amount") or 0)
        power = names.power(str(row.get("power") or "?"))
        base |= {
            "actor": applier or target,
            "actor_side": applier_side or target_side,
            "actor_uid": applier_uid if applier is not None else target_uid,
            "target": target,
            "target_side": target_side,
            "target_uid": target_uid,
            "amount": amount,
            "noun": power,
            "verb": "gained" if applier is None else "applied",
        }
        if applier is None:
            base["text"] = f"{target} gained {amount} {power}."
        else:
            base["text"] = f"{applier} applied {amount} {power} to {target}."
        return base

    if kind == "monster_move":
        actor, side, actor_uid = _creature(row.get("monster"), names)
        move = names.move(str(row.get("move_id") or "?"))
        base |= {
            "actor": actor,
            "actor_side": side,
            "actor_uid": actor_uid,
            "noun": move,
            "verb": "used",
        }
        base["text"] = f"{actor} used {move}."
        return base

    if kind == "potion":
        potion = names.potion(str(row.get("potion") or "?"))
        base |= {"actor": player, "actor_side": "player", "noun": potion, "verb": "drank"}
        base["text"] = f"{player} drank {potion}."
        return base

    if kind == "card_generated":
        name = card_name()
        base |= {
            "actor": player,
            "actor_side": "player",
            "noun": name,
            "verb": "generated",
        }
        base["text"] = f"{player} generated {name}."
        return base

    if kind in ("orb_channeled", "orb_evoked"):
        orb = names.orb(str(row.get("orb") or "?"))
        verb = "channeled" if kind == "orb_channeled" else "evoked"
        base |= {"actor": player, "actor_side": "player", "noun": orb, "verb": verb}
        base["text"] = f"{player} {verb} {orb}."
        return base

    if kind == AUTO_PLAYED_KIND:
        name = card_name()
        via = f" via {base['source']}" if base["source"] else ""
        base |= {
            "actor": player,
            "actor_side": "player",
            "noun": name,
            "verb": "auto-played",
            # The raw card identity, for the board's auto-play card flash (the
            # ``source_*`` fields name the PARENT play, e.g. Havoc).
            "card_id": str(row.get("card") or "?"),
            "card_upgraded": bool(row.get("upgraded")),
        }
        base["text"] = f"{player} auto-played {name}{via}."
        return base

    # ── verbose-only kinds ───────────────────────────────────────────────
    if kind in ("card_play_started", "card_play_finished"):
        name = card_name()
        verb = "played" if kind == "card_play_finished" else "started playing"
        base |= {"actor": player, "actor_side": "player", "noun": name, "verb": verb}
        base["text"] = f"{player} {verb} {name}."
        return base

    if kind in ("card_drawn", "card_discarded", "card_exhausted"):
        name = card_name()
        # "drew", not C#'s copy-pasted "discarded" — see the module docstring.
        verb = {
            "card_drawn": "drew",
            "card_discarded": "discarded",
            "card_exhausted": "exhausted",
        }[kind]
        base |= {"actor": player, "actor_side": "player", "noun": name, "verb": verb}
        base["text"] = f"{player} {verb} {name}."
        return base

    if kind == "energy_spent":
        amount = int(row.get("amount") or 0)
        base |= {"actor": player, "actor_side": "player", "amount": amount, "verb": "spent"}
        base["text"] = f"{player} spent {amount} Energy."
        return base

    # An unknown kind is a newer engine talking to an older workbench. Show the
    # engine's own line rather than dropping the event on the floor.
    text = row.get("text")
    if not text:
        return None
    base["text"] = str(text)
    return base
