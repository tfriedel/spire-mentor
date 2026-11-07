"""The honest combat snapshot a context-aware student reads (GH #830 step 0, DESIGN §a.2).

`honest_snapshot(bv, room)` turns `bv = cs.belief_view(0)` into a JSON-able
dict that holds only what the player can see. `belief_view(0)` keeps every
face-up pile and the pinned draw-pile cards, sorts the unknown middle of the
draw pile canonically and reseeds every RNG stream from 0, so everything in
`bv` is a function of the observable position EXCEPT what `belief_root` does
not touch: monster AI internals, uids and RNG counters. The snapshot is built
with a WHITELIST (`FIELD_WHITELIST`), not a blacklist, so a field the wire
grows later stays out until someone decides it is visible.

What is dropped, and why:

* cards: `stable_idx` (the uid) and `combat_vars` (per-instance scratch; the
  visible numbers are in `base_*` / `calculated_value`), and
  `affliction_applied_exhaust`;
* top level: `rng`, `rng_counters`, `master_deck` (always emitted as
  `upgraded: false`, which is misleading), `combat_id`, `side`,
  `wire_version`, `phase`, `turn` (= `round`) and the intent dialect tag;
* enemies: `uid` (kept only as the opaque join key `eid` for `IntentLog`; the
  token builder never reads it), `move_state`, `fabricator_last_spawned`,
  `curse_of_knowledge_counter`, `intent_history` (always empty on the wire)
  and each intent's `modifiers`;
* relics: `counter` for any relic outside the active game build's
  `static_tables.show_counter_relics()` (the game shows no counter for them).
  A ShowCounter relic's counter is replaced by the number C# displays
  (`RELIC_COUNTER_DISPLAY`, per build in `RELIC_COUNTER_DISPLAY_BY_VERSION`):
  the engine's `Relic.counter` is often packed or cumulative (Pocketwatch
  packs last turn's count into bits 8-15, Pen Nib counts every attack of the
  fight) where `DisplayAmount` shows `this turn` or `% 10`.

The unknown middle of the draw pile, the discard and the exhaust pile become
canonical multisets (`[[card, count], ...]`); the hand and the pinned draw
cards keep their (visible) order. Each hand card carries the damage / block
numbers the card face shows (`card_damage_preview` / `card_block_preview`).

`snapshot_from_parts(d, dmg, blk, choice, room)` is the pure core, so the
provenance fuzz test can feed it a doctored `state_to_dict`.

`action_classes(bv, actions)` groups legal actions into the honest form of
the search's transposition merge: two actions are one class iff applying each
to `bv` gives the same `observation_hash()` (content-folded, uid-free), or
they are card plays with one search edge key, or `choose` actions after which
the player sees the same snapshot (`observation_hash` keys a pinned draw-pile
card by uid, so putting back either of two identical Strikes differs there). The
legal actions come from the TRUE state; `obs_actions` re-expresses a `choose`
action in `bv`'s option order (the unknown draw-pile middle is sorted in `bv`,
so a draw-pile choice can list its options in another order there), matching
options by visible content.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .static_tables import active_version, show_counter_relics


def _mod(n: int) -> Callable[[int], int]:
    return lambda c: c % n


# ShowCounter relic -> the number C# displays (`DisplayAmount`, v0.107.1
# decomp) as a function of the engine's `Relic.counter` (0 when the wire
# omits it), or None while C# hides the counter. A relic mapped to None has
# its displayed number nowhere in the engine's counter (it keeps it in run
# state or derives it from the turn), so its counter is dropped. `IsActivating`
# (the flash that shows the threshold for an instant) has no engine state.
RELIC_COUNTER_DISPLAY: dict[str, Callable[[int], int | None] | None] = {
    # The engine counter is the displayed number.
    "GIRYA": lambda c: c,  # TimesLifted
    "PUMPKIN_CANDLE": lambda c: c,  # KindleCount
    "VELVET_CHOKER": lambda c: c,  # _cardsPlayedThisTurn
    "PENDULUM": lambda c: c,  # TurnsSeen, stored already wrapped at Turns (3)
    "EMBER_TEA": lambda c: max(0, c),  # Math.Max(0, CombatsLeft)
    # ShowCounter: CardsPlayedThisTurn < Cards (5).
    "BRILLIANT_SCARF": lambda c: c if c < 5 else None,
    # ShowCounter: OrbsChanneled < OrbCount (7); DisplayAmount = Min(.., 7).
    "METRONOME": lambda c: c if c < 7 else None,
    # Cumulative (or not yet wrapped) counts; C# shows them modulo the threshold.
    "PEN_NIB": _mod(10),  # AttacksPlayed % 10
    "NUNCHAKU": _mod(10),  # AttacksPlayed % Cards (10)
    "TUNING_FORK": _mod(10),  # SkillsPlayed, wrapped at Cards (10)
    "GALACTIC_DUST": _mod(10),  # StarsSpent % Stars (10)
    "HAPPY_FLOWER": _mod(3),  # TurnsSeen = (TurnsSeen + 1) % Turns (3)
    "FAKE_HAPPY_FLOWER": _mod(5),  # TurnsSeen = (TurnsSeen + 1) % Turns (5)
    "POLLINOUS_CORE": _mod(4),  # TurnsSeen, reset to 0 by the turn-4 draw
    "IRON_CLUB": _mod(4),  # CardsPlayed % Cards (4)
    "BOOK_OF_FIVE_RINGS": _mod(5),  # CardsAddedSinceLastTrigger (Cards 5)
    "JOSS_PAPER": _mod(5),  # CardsExhausted, wrapped at ExhaustAmount (5)
    "LASTING_CANDY": _mod(2),  # CombatsSeen % 2
    "FISHING_ROD": _mod(3),  # CombatsSeen % Combats (3)
    "KUNAI": _mod(3),  # AttacksPlayedThisTurn % Cards (3)
    "SHURIKEN": _mod(3),  # AttacksPlayedThisTurn % Cards (3)
    "ORNAMENTAL_FAN": _mod(3),  # AttacksPlayedThisTurn % Cards (3)
    "KUSARIGAMA": _mod(3),  # AttacksPlayedThisTurn % Cards (3)
    "LETTER_OPENER": _mod(3),  # SkillsPlayedThisTurn % Cards (3)
    # ShowCounter: !IsUsedUp (CombatsSeen < Combats 3 * Relics 4).
    "TOY_BOX": lambda c: c % 3 if c < 12 else None,
    # Packed: low byte = cards this turn, bits 8-15 = cards last turn.
    "POCKETWATCH": lambda c: c & 0xFF,
    # The engine counts elites LEFT (5 down to 0); C# shows ElitesDefeated.
    "SWORD_OF_STONE": lambda c: 5 - c,
    # Packed: low byte = TimesUsed; shown as Cards (3) - TimesUsed while < 3.
    "SILVER_CRUCIBLE": lambda c: 3 - (c & 0xFF) if (c & 0xFF) < 3 else None,
    # Counter = CombatsFinished (< 0 once paid out); shown as 5 - n while > 0.
    "WONGOS_MYSTERY_TICKET": lambda c: 5 - c if 0 <= c < 5 else None,
    # Counter = cooldown (-1 pending, 0 armed); shown only while > 0.
    "PAELS_LEGION": lambda c: c if c > 0 else None,
    # Displayed number not in the engine counter: dropped. Diadem shows the
    # cards played this turn (player.cards_played_this_turn); Flesh and Stone
    # Calendar the turn (`round`); Tooth, Wing and Winged Boots run state.
    "DIAMOND_DIADEM": None,
    "PAELS_FLESH": None,
    "STONE_CALENDAR": None,
    "PAELS_TOOTH": None,
    "PAELS_WING": None,
    "WINGED_BOOTS": None,
}

# v0.111.0 (e2e WP1, diffed against the v0.107.1 decomp): Diamond Diadem
# shows no counter (its ShowCounter override is gone), and Toy Box offers 5
# wax relics, so it shows CombatsSeen % 3 while CombatsSeen < 3 * 5. Every
# other ShowCounter relic keeps its DisplayAmount, and the engine's v0.111
# counters fit the same rules: Pendulum stores TurnsSeen % 3, Pollinous Core
# resets at 4, Lasting Candy counts card rewards and still shows them % 2.
RELIC_COUNTER_DISPLAY_V0111: dict[str, Callable[[int], int | None] | None] = {
    **{k: v for k, v in RELIC_COUNTER_DISPLAY.items() if k != "DIAMOND_DIADEM"},
    "TOY_BOX": lambda c: c % 3 if c < 15 else None,
}
RELIC_COUNTER_DISPLAY_BY_VERSION: dict[str, dict[str, Callable[[int], int | None] | None]] = {
    "v0.107.1": RELIC_COUNTER_DISPLAY,
    "v0.111.0": RELIC_COUNTER_DISPLAY_V0111,
}


def display_counter(relic_id: str, counter: int | None, version: str | None = None) -> int | None:
    """The counter C# shows on `relic_id` given the engine's `Relic.counter`
    (None = omitted on the wire = 0), or None when no counter is shown. The
    rules are `version`'s (default: the active game version)."""
    ver = active_version(version)
    if relic_id not in show_counter_relics(ver):
        return None
    rule = RELIC_COUNTER_DISPLAY_BY_VERSION[ver][relic_id]
    return None if rule is None else rule(int(counter or 0))


CARD_KEYS = (
    "id",
    "kind",
    "upgraded",
    "base_cost",
    "base_damage",
    "base_block",
    "calculated_value",
    "flags",
    "enchantment",
    "enchantment_amount",
    "affliction",
    "affliction_stacks",
)
PLAYER_KEYS = (
    "hp",
    "max_hp",
    "block",
    "energy",
    "max_energy",
    "stars",
    "potion_slots",
    "cards_played_this_turn",
    "bound_card_played_this_turn",
)
POWER_KEYS = ("name", "amount", "display_amount", "counter")
INTENT_KEYS = ("kind", "amount", "effective_amount", "times")
ROOMS = ("monster", "elite", "boss")

# Every key the snapshot may hold, by where it sits. `card` covers every card
# dict (hand, piles, play pile); `hand_card` adds the previews.
FIELD_WHITELIST: dict[str, frozenset[str]] = {
    "": frozenset(
        {
            "player",
            "powers",
            "relics",
            "potions",
            "hand",
            "draw_unknown",
            "draw_top",
            "draw_bottom",
            "discard",
            "exhaust",
            "play",
            "enemies",
            "ehist",
            "round",
            "is_elite",
            "room",
            "is_boss",
            "choice",
        }
    ),
    "player": frozenset(PLAYER_KEYS),
    "power": frozenset(POWER_KEYS),
    "relic": frozenset({"id", "counter", "is_wax", "is_melted"}),
    "card": frozenset({*CARD_KEYS, "cost"}),
    "hand_card": frozenset({*CARD_KEYS, "cost", "dmg", "blk"}),
    "enemy": frozenset(
        {
            "eid",
            "kind",
            "hp",
            "max_hp",
            "block",
            "powers",
            "intents",
            "is_dead",
            "removed_after_death",
            "stolen",
        }
    ),
    "intent": frozenset(INTENT_KEYS),
    "stolen": frozenset({"id", "upgraded"}),
    "ehist": frozenset({"lag", *INTENT_KEYS}),
    "choice": frozenset({"kind", "min_select", "max_select", "can_skip", "options"}),
    "option": frozenset({"id", "upgraded", "type", "pile", "hand_index"}),
}
# Names that must never appear anywhere in a snapshot (test 3).
FORBIDDEN = frozenset(
    {
        "rng",
        "rng_counters",
        "uid",
        "stable_idx",
        "move_state",
        "combat_vars",
        "master_deck",
        "fabricator_last_spawned",
        "curse_of_knowledge_counter",
        "intent_history",
        "run_seed",
        "counters",
    }
)


def _card(c: dict[str, Any]) -> dict[str, Any]:
    out = {k: c[k] for k in CARD_KEYS if k in c}
    out["cost"] = c.get("cost_override", c.get("base_cost", 0))
    if "flags" in out:
        out["flags"] = sorted(out["flags"])
    return out


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


def multiset(cards: list[dict[str, Any]]) -> list[list[Any]]:
    """`[[card, count], ...]` sorted by the card's canonical JSON: order-free."""
    groups: dict[str, list[Any]] = {}
    for c in cards:
        k = _canon(c)
        if k in groups:
            groups[k][1] += 1
        else:
            groups[k] = [c, 1]
    return [groups[k] for k in sorted(groups)]


def _power(p: dict[str, Any]) -> dict[str, Any]:
    return {k: p[k] for k in POWER_KEYS if k in p}


def _relic(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": r["id"]}
    shown = display_counter(r["id"], r.get("counter"))
    if shown is not None:
        out["counter"] = shown
    if r.get("is_wax"):
        out["is_wax"] = True
    if r.get("is_melted"):
        out["is_melted"] = True
    return out


def _intent(i: dict[str, Any]) -> dict[str, Any]:
    return {k: i[k] for k in INTENT_KEYS if k in i}


def _enemy(m: dict[str, Any]) -> dict[str, Any]:
    out = {
        "eid": m.get("uid"),
        "kind": m["kind"],
        "hp": m.get("hp", 0),
        "max_hp": m.get("max_hp", 1),
        "block": m.get("block", 0),
        "powers": [_power(p) for p in m.get("powers", [])],
        "intents": [_intent(i) for i in m.get("intents", [])],
        "is_dead": bool(m.get("is_dead")),
        "removed_after_death": bool(m.get("removed_after_death")),
    }
    stolen = m.get("stolen_cards")
    if stolen:
        out["stolen"] = [{"id": s["id"], "upgraded": bool(s.get("upgraded"))} for s in stolen]
    return out


def _choice(pc: dict[str, Any] | None) -> dict[str, Any] | None:
    if pc is None:
        return None
    opts = []
    for o in pc.get("cards", []):
        pile = o.get("pile", "generated")
        opt: dict[str, Any] = {
            "id": o["id"],
            "upgraded": bool(o.get("upgraded")),
            "type": o.get("type"),
            "pile": pile,
        }
        # A draw-pile index is a position in the canonicalized draw pile:
        # dropped. A hand index is the visible hand slot.
        if pile == "hand" and "index" in o:
            opt["hand_index"] = o["index"]
        opts.append(opt)
    return {
        "kind": pc.get("kind"),
        "min_select": pc.get("min_select", 0),
        "max_select": pc.get("max_select", 0),
        "can_skip": bool(pc.get("can_skip")),
        "options": opts,
    }


def snapshot_from_parts(
    d: dict[str, Any],
    dmg: list[dict[int, int]],
    blk: list[int | None],
    choice: dict[str, Any] | None,
    room: str,
) -> dict[str, Any]:
    """The honest snapshot from `state_to_dict(bv)` and `bv`'s previews / choice.

    Pure: reads only its arguments, and of `d` only whitelisted fields.

    Args:
        d: `sts2_rs.state_to_dict(bv)`.
        dmg: `bv.card_damage_preview()` (hand-aligned `{enemy index: dmg}`).
        blk: `bv.card_block_preview()` (hand-aligned, None = no block).
        choice: `bv.pending_choice()`.
        room: `monster` / `elite` / `boss` (the room is visible).
    """
    if room not in ROOMS:
        raise ValueError(f"room {room!r} not in {ROOMS}")
    p = d["player"]
    player = {k: p[k] for k in PLAYER_KEYS if k in p}
    player.setdefault("stars", 0)
    player.setdefault("potion_slots", 0)
    deck = d["deck"]
    hand = []
    for i, c in enumerate(deck.get("hand", [])):
        hc = _card(c)
        pv = dmg[i] if i < len(dmg) else {}
        hc["dmg"] = {str(int(k)): int(v) for k, v in sorted(pv.items(), key=lambda kv: int(kv[0]))}
        hc["blk"] = blk[i] if i < len(blk) else None
        hand.append(hc)
    draw = deck.get("draw_pile", [])
    top_idx = list(deck.get("pinned_top", []))
    bot_idx = list(deck.get("pinned_bottom", []))
    pinned = set(top_idx) | set(bot_idx)
    snap = {
        "player": player,
        "powers": [_power(x) for x in p.get("powers", [])],
        "relics": [_relic(r) for r in p.get("relics", [])],
        "potions": [str(x) for x in p.get("potions", []) if x],
        "hand": hand,
        "draw_unknown": multiset([_card(c) for i, c in enumerate(draw) if i not in pinned]),
        "draw_top": [_card(draw[i]) for i in top_idx],
        "draw_bottom": [_card(draw[i]) for i in bot_idx],
        "discard": multiset([_card(c) for c in deck.get("discard_pile", [])]),
        "exhaust": multiset([_card(c) for c in deck.get("exhaust_pile", [])]),
        "play": [_card(c) for c in deck.get("play_pile", [])],
        "enemies": [_enemy(m) for m in d.get("encounter", [])],
        "round": int(d.get("round", 0)),
        "is_elite": bool(d.get("is_elite")),
        "room": room,
        "is_boss": room == "boss",
        "choice": _choice(choice),
    }
    return snap


def honest_snapshot(bv: Any, room: str) -> dict[str, Any]:
    """`snapshot_from_parts` of the four things read from `bv` (and nothing else)."""
    import sts2_rs

    return snapshot_from_parts(
        sts2_rs.state_to_dict(bv),
        bv.card_damage_preview(),
        bv.card_block_preview(),
        bv.pending_choice(),
        room,
    )


def whitelist_violations(snap: dict[str, Any]) -> list[str]:
    """Every key in `snap` outside `FIELD_WHITELIST` or in `FORBIDDEN` (paths)."""
    bad: list[str] = []

    def check(obj: Any, where: str, path: str) -> None:
        if not isinstance(obj, dict):
            return
        allowed = FIELD_WHITELIST[where]
        for k in obj:
            if k not in allowed or k in FORBIDDEN:
                bad.append(f"{path}.{k}")

    check(snap, "", "$")
    check(snap["player"], "player", "$.player")
    for i, x in enumerate(snap["powers"]):
        check(x, "power", f"$.powers[{i}]")
    for i, x in enumerate(snap["relics"]):
        check(x, "relic", f"$.relics[{i}]")
    for i, x in enumerate(snap["hand"]):
        check(x, "hand_card", f"$.hand[{i}]")
    for pile in ("draw_unknown", "discard", "exhaust"):
        for i, (c, _n) in enumerate(snap[pile]):
            check(c, "card", f"$.{pile}[{i}]")
    for pile in ("draw_top", "draw_bottom", "play"):
        for i, c in enumerate(snap[pile]):
            check(c, "card", f"$.{pile}[{i}]")
    for i, m in enumerate(snap["enemies"]):
        check(m, "enemy", f"$.enemies[{i}]")
        for j, x in enumerate(m["powers"]):
            check(x, "power", f"$.enemies[{i}].powers[{j}]")
        for j, x in enumerate(m["intents"]):
            check(x, "intent", f"$.enemies[{i}].intents[{j}]")
        for j, x in enumerate(m.get("stolen", [])):
            check(x, "stolen", f"$.enemies[{i}].stolen[{j}]")
    for i, hist in enumerate(snap.get("ehist") or []):
        for j, x in enumerate(hist):
            check(x, "ehist", f"$.ehist[{i}][{j}]")
    if snap["choice"] is not None:
        check(snap["choice"], "choice", "$.choice")
        for j, x in enumerate(snap["choice"]["options"]):
            check(x, "option", f"$.choice.options[{j}]")
    return bad


# ── actions ──────────────────────────────────────────────────────────────


def _opt_key(o: dict[str, Any]) -> str:
    """An option's visible content (a hand option also by its slot)."""
    return _canon(
        {
            "id": o["id"],
            "upgraded": bool(o.get("upgraded")),
            "pile": o.get("pile", "generated"),
            "hand_index": o.get("index") if o.get("pile") == "hand" else None,
        }
    )


def obs_actions(
    actions: list[dict[str, Any]],
    cs_choice: dict[str, Any] | None,
    bv_choice: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """`actions` (the true state's) with every `choose` selection re-indexed
    into `bv`'s option order by visible content.

    Identical options are interchangeable to the player, so they are matched
    first-unused in order. Raises if an option has no visible twin (the two
    choices are then not the same choice).
    """
    if cs_choice is None or bv_choice is None:
        return actions
    cs_keys = [_opt_key(o) for o in cs_choice.get("cards", [])]
    bv_keys = [_opt_key(o) for o in bv_choice.get("cards", [])]
    if cs_keys == bv_keys:
        return actions
    if sorted(cs_keys) != sorted(bv_keys):
        raise ValueError("true and belief choices list different options")
    remap: dict[int, int] = {}
    used: set[int] = set()
    for i, k in enumerate(cs_keys):
        for j, kb in enumerate(bv_keys):
            if j not in used and kb == k:
                remap[i] = j
                used.add(j)
                break
    out = []
    for a in actions:
        if a.get("action") == "choose":
            sel = [remap[int(i)] for i in a.get("selection", [])]
            a = dict(a)
            a["selection"] = sel
            a["index"] = sel[0] if sel else 0
        out.append(a)
    return out


def play_edge_key(hand: list[dict[str, Any]], a: dict[str, Any]) -> tuple | None:
    """The search's own identity of a card play (`sts2_mcts::canonicalize_actions`:
    card id, upgraded, cost, free, affliction, target), read from visible card
    content: the search backs every play with this key by ONE tree child (it
    resolves to the first matching hand card), so it never tells such plays
    apart. The effective cost stands in for the engine's `free` flag."""
    if a.get("action") != "play":
        return None
    c = hand[int(a["card_index"])]
    return (
        c["id"],
        bool(c.get("upgraded")),
        c.get("base_cost"),
        c.get("cost_override", c.get("base_cost")),
        c.get("affliction"),
        a.get("target_index"),
    )


def choose_outcome_key(c: Any) -> str:
    """What the player sees after a `choose` action: the honest snapshot of
    the resulting state (room-independent: `room` is fixed)."""
    return _canon(honest_snapshot(c.belief_view(0), "monster"))


def action_classes(bv: Any, actions: list[dict[str, Any]]) -> list[int]:
    """Class id per action. Two actions are one class iff applying each to `bv`
    reaches the same `observation_hash()` (content-folded, uid-free), OR they
    are card plays with the same search edge key (`play_edge_key`), OR they are
    `choose` actions after which the honest snapshot is the same
    (`choose_outcome_key`): the honest form of the search's own merge, plus
    the options a player cannot tell apart (`observation_hash` keys a pinned
    draw-pile card by uid, so putting back either of two identical Strikes
    reaches two hashes). Classes are the connected components; ids are dense,
    in first-occurrence order.

    `actions` must be in `bv`'s index space (`obs_actions`).
    """
    import sts2_rs

    hand = sts2_rs.state_to_dict(bv)["deck"].get("hand", [])
    parent = list(range(len(actions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[Any, int] = {}
    for i, a in enumerate(actions):
        c = bv.clone()
        c.apply(a)
        seen = choose_outcome_key(c) if a.get("action") == "choose" else None
        for key in (
            ("h", int(c.observation_hash())),
            ("k", play_edge_key(hand, a)),
            ("o", seen),
        ):
            if key[1] is None:
                continue
            j = first.setdefault(key, i)
            parent[find(i)] = find(j)
    ids: dict[int, int] = {}
    return [ids.setdefault(find(i), len(ids)) for i in range(len(actions))]
