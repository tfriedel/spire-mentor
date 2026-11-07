"""Exact id vocabularies for the run-level policy encoder.

Card/relic/potion universes come from ``sts2_rs.list_registry_names`` (sorted
canonical UPPER_SNAKE names: card=579, relic_id≈290, potion=63), so embedding
indices are deterministic and stable. Index ``len(names)`` is a reserved
OOV/pad slot, so every table is built with ``len(names)+1`` rows.

The relic universe is ``"relic_id"`` (the run-side ``RelicId`` enum), NOT
``"relic"`` (the ~111-entry combat ``RelicKind`` subset): Neow blessings,
event relics and shop stock exist only in ``RelicId`` and come back OOV
against the combat subset (measured on the M-P4 corpus — every Neow option
label was OOV under ``"relic"``).

Normalization wrinkle: ``card_reward_option_ids`` returns ``"CARD.ULTIMATE_STRIKE"``
while ``master_deck_detailed`` returns PascalCase (``"UltimateStrike"``). Both
must map to the canonical ``"ULTIMATE_STRIKE"``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

import sts2_rs

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Fixed map room-type vocab (sts2 MapPointType, snake form from act_map()).
ROOM_TYPES = (
    "unassigned",
    "unknown",
    "monster",
    "elite",
    "boss",
    "shop",
    "rest_site",
    "treasure",
    "ancient",
)
ROOM_IDX = {r: i for i, r in enumerate(ROOM_TYPES)}
ROOM_OOV = len(ROOM_TYPES)
N_ROOM = len(ROOM_TYPES) + 1


def canonical(name: str) -> str:
    """``"CARD.ULTIMATE_STRIKE"`` / ``"UltimateStrike"`` → ``"ULTIMATE_STRIKE"``.

    Also strips a trailing ``_CARD`` — the ``list_registry_names("card")``
    universe carries it on a subset of entries (``DEBILITATE_CARD``,
    ``BARRICADE_CARD``, ``NECRO_MASTERY_CARD``, …) while scraped ids
    (``CARD.DEBILITATE``) and PascalCase deck names (``Debilitate``) do not,
    so without this ~10% of the Necrobinder card universe canonicalized to a
    key absent from the registry and fell through to the OOV slot. Applying
    it to BOTH the registry names and the queries keeps the two sides aligned.
    """
    s = name.split(".", 1)[-1]  # strip CARD./RELIC./POTION. prefix
    s = _CAMEL_BOUNDARY.sub("_", s).upper()
    if s.endswith("_CARD") and len(s) > len("_CARD"):
        s = s[: -len("_CARD")]
    return s


_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def canonical_boss(name: str) -> str:
    """``"KAISER_CRAB_BOSS"`` / ``"kaiser_crab_boss"`` / ``"KaiserCrab"`` /
    ``"Kaiser Crab"`` → ``"KAISER_CRAB_BOSS"``.

    Plain :func:`canonical` is not enough for the boss identity, because it
    reaches the encoder from three producers with three shapes.
    ``full_map()``'s boss ``id`` normally carries the generated ``UPPER_SNAKE``
    encounter id (``THE_KIN_BOSS``); the ``EncounterSpec`` fallback carries the
    pool's lower-snake ``name``; and the pinned-boss fallback carries a spaced
    *display* name (``"Ceremonial Beast"``) with no ``_BOSS`` suffix at all.
    Folding every non-alphanumeric run to ``_`` and appending the suffix when
    it is missing collapses all three onto the id universe the engine exports,
    so a fallback path costs an embedding row rather than silently landing on
    OOV.
    """
    s = _NON_ALNUM.sub("_", canonical(name)).strip("_")
    if not s or s.endswith("_BOSS"):
        return s
    return f"{s}_BOSS"


class Registry:
    """name→idx with a reserved OOV slot at ``size-1``."""

    def __init__(self, kind: str, norm: Callable[[str], str] = canonical) -> None:
        self.names: list[str] = list(sts2_rs.list_registry_names(kind))
        # Key on the NORMALIZED form (not the raw registry name): a subset of
        # the "card" universe carries a ``_CARD`` suffix the scraped ids /
        # PascalCase deck names lack, so raw-name keys silently sent those to
        # the OOV slot (~10% of the Necrobinder card universe). ``index()``
        # applies the same normalizer to the query, so both sides must agree.
        # First writer wins on the (rare) collision, keeping indices stable.
        self.norm = norm
        self.idx: dict[str, int] = {}
        for i, n in enumerate(self.names):
            self.idx.setdefault(norm(n), i)
        self.oov = len(self.names)
        self.size = len(self.names) + 1  # +1 OOV/pad row

    def index(self, raw_name: str) -> int:
        return self.idx.get(self.norm(raw_name), self.oov)


@lru_cache(maxsize=1)
def cards() -> Registry:
    return Registry("card")


@lru_cache(maxsize=1)
def relics() -> Registry:
    return Registry("relic_id")


@lru_cache(maxsize=1)
def potions() -> Registry:
    return Registry("potion")


@lru_cache(maxsize=1)
def events() -> Registry:
    """Run-event universe (`EventId::all()` python names; ~63 entries)."""
    return Registry("event")


@lru_cache(maxsize=1)
def bosses() -> Registry:
    """Boss-encounter universe (9 entries) — the identity of the boss waiting
    at the top of the current act's map.

    Sourced from the map generator's own ``RoomKind::Boss`` filter
    (``run_room_gen::all_boss_encounter_ids``), so it cannot drift from the
    ids the engine can actually roll. This is the only registry keyed by
    :func:`canonical_boss` rather than :func:`canonical`.

    **Why the encounter id and not the lead monster.** The screening harness
    (GH #431's sibling, `68d56e87a`) keyed its candidate boss feature on the
    boss meta-node's ``kind`` — its lead monster — because that had a registry
    and the encounter id did not. Its own doc comment flagged the cost: two
    boss encounters sharing a lead monster would share an embedding row, and
    a null would then be unreadable. Exporting the id universe from the
    generator removes that risk and shrinks the table from ~121 monster rows
    to 9, so every row is exercised. The shipped champion is trained on this
    keying.
    """
    return Registry("boss", canonical_boss)


def room_index(room: str) -> int:
    return ROOM_IDX.get(room, ROOM_OOV)


# Atomic (non-card/relic/potion/path/event) action kinds — the analogue of the
# fork's FixedAction enum, minus per-event entries (events carry their own
# event_id × option_idx token instead).
FIXED_KINDS = (
    "none",
    "skip_card",
    "take_relic",
    "skip_relic",
    "leave_shop",
    "shop_remove",
    "rest_heal",
    "rest_smith",
    "rest_other",
    "open_chest",
    "sacrifice",
    # Appended (never inserted) so every pre-existing embedding row keeps
    # its index. See `N_FIXED` for why appending was FREE here.
    "decline_potion",
)
FIXED_IDX = {k: i for i, k in enumerate(FIXED_KINDS)}
# The PAD index — `model.RunPolicyNet` allocates `N_FIXED + 1` rows and pins
# `padding_idx` to the last, so the table has always carried exactly one
# SPARE row above the named entries. "decline_potion" (GH #379) consumes
# that spare, which is why this is now `len(...)` and not `len(...) + 1`:
# the pad index does not move, the table does not grow, and every exported
# `.onnx` champion -- whose embedding tables are shape-FROZEN, unlike a
# torch checkpoint that `model.load_compat` can grow -- keeps loading. The
# next entry after this one has no spare left: it WILL move the pad and
# require re-exporting every champion.
# `tests/test_ppo_potion_swap_surface.py` pins this.
N_FIXED = len(FIXED_KINDS)


def fixed_index(kind: str) -> int:
    return FIXED_IDX.get(kind, FIXED_IDX["none"])


# Decision-surface vocab for the state token (the fork's screen_state /
# CardSelectScreenType analogue — lets one CARD section distinguish
# "pick to add" from "pick to remove/upgrade/smith").
SURFACES = (
    "path",
    "card-reward",
    "card-remove",
    "card-upgrade",
    "card-transform",
    "card-enchant",
    "card-grant",
    "card-smith",
    "boss-relic",
    "relic-reward",
    "shop",
    "event",
    "rest",
    "treasure",
    "unknown",
    # Appended AFTER "unknown" on purpose: `surface_index` resolves the
    # fallback by NAME, so the tail is free real estate, and rows 0..14 keep
    # the indices every shipped champion was trained with. Inserting
    # mid-table would silently re-map them.
    "potion-swap",
)
SURFACE_IDX = {s: i for i, s in enumerate(SURFACES)}
# `len(...)`, not `len(...) + 1` -- same spare-row argument as `N_FIXED`
# above. Row 15 was the unused slack every champion carries (it reads
# 0.000000 drift in
# `docs/experiments/2026-08-09-deck-mutation-preflight.md`, which is what
# "never trained" looks like); "potion-swap" takes it, so the pad index
# stays 16 and no exported `.onnx` changes shape.
N_SURFACE = len(SURFACES)


def surface_index(s: str) -> int:
    return SURFACE_IDX.get(s, SURFACE_IDX["unknown"])


@lru_cache(maxsize=1)
def enchantments() -> Registry:
    """Enchantment universe (23 entries) — WHICH enchantment a pending
    deck-enchant pick stamps on the card you choose.

    Sourced from ``EnchantmentId::ALL`` through the append-only pin
    (``sts2-vocab/src/vocab_pins/enchantment.txt``) for the same reason
    the boss registry is pinned: the position IS the embedding row.

    Before this existed, ``pending_deck_enchants()`` returned a bare count and
    an enchant option featurized IDENTICALLY to a removal option — both
    ``card_tok(card_id, upgraded, 0)``. Perfect Fit, Sharp, Adroit and Goopy
    were one decision to the net, and Goopy is a downside that inverts which
    card you want (``wb-27d55ac0``).
    """
    return Registry("enchantment")


# Which deck operation a CARD-section action token belongs to (GH #545).
#
# WHY THIS IS ITS OWN SLOT AND NOT A `FIXED_KINDS` ENTRY. `shop_remove_card`
# separates itself with `cat[6] = fixed_index("shop_remove")`, and copying that
# for the other ops is the obvious fix -- but `N_FIXED` has no spare row left
# (see its comment above), so five more entries would move `pad["fixed"]` from
# 12 to 17. `schema.empty_cat()` writes that pad into `cat[6]` of EVERY token,
# so every committed champion's 13-row fixed table would stop loading at once.
# An APPENDED slot costs nothing instead: both consumers read the `cat` width
# off the graph they loaded, exactly as they do for `emb_boss`.
DECK_OPS = (
    "none",
    "reward",
    "smith",
    "remove",
    "upgrade",
    "transform",
    "enchant",
    "grant",
    "shop_buy",
    "shop_remove",
    # A Scroll Boxes pack: its member cards (obs tokens) and the pick-this-pack
    # action token (GH #978, `card_bundles` feature). Took the spare row.
    "bundle",
)
DECK_OP_IDX = {k: i for i, k in enumerate(DECK_OPS)}
# The PAD row, pinned at 11. The table was defined with one SPARE row above
# the named entries so the next deck op was a free append rather than a pad
# move; "bundle" (GH #978) took it, so this is now `len(DECK_OPS)`. The NEXT
# op moves the pad and every committed champion's `cat[10]` with it (the
# `N_FIXED` trap): widen the table and re-export the fleet, or find a slot.
N_DECK_OP = 11
assert len(DECK_OPS) == N_DECK_OP, "deck-op pad must stay at row 11"


def deck_op_index(kind: str) -> int:
    return DECK_OP_IDX.get(kind, DECK_OP_IDX["none"])


# A card's spec-level type and rarity (`card_kind` feature, GH #977), in the
# engine's enum order (`CardType`, `CardRarity`). Append-only: the position is
# the embedding row, and the pad row is `len(...)`.
CARD_TYPES = ("ATTACK", "SKILL", "POWER", "CURSE", "STATUS")
CARD_RARITIES = (
    "BASIC",
    "COMMON",
    "UNCOMMON",
    "RARE",
    "ANCIENT",
    "STATUS",
    "CURSE",
    "EVENT",
    "TOKEN",
    "QUEST",
)
N_CARD_TYPE = len(CARD_TYPES)
N_CARD_RARITY = len(CARD_RARITIES)


@lru_cache(maxsize=1)
def card_kind_rows() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(type_row, rarity_row)`` per :func:`cards` index, pad where unknown.

    Indexed by the card's embedding row (OOV included), so the encoder stamps
    a token from the ``cat[0]`` it already wrote. Read from the engine's specs
    (``sts2_rs.card_kinds``), the same source the Rust encoder uses.
    """
    reg = cards()
    types = [N_CARD_TYPE] * reg.size
    rarities = [N_CARD_RARITY] * reg.size
    t_idx = {t: i for i, t in enumerate(CARD_TYPES)}
    # GH #1111 gave Quest cards their own engine `CardType`; they used to be
    # Skills. Keep them on the Skill row so the table (and every promoted
    # champion) is unchanged. `encoder.rs::card_kind_rows` folds them alike.
    t_idx["QUEST"] = t_idx["SKILL"]
    r_idx = {r: i for i, r in enumerate(CARD_RARITIES)}
    for name, ctype, rarity in sts2_rs.card_kinds():
        i = reg.index(name)
        if i == reg.oov:
            continue
        types[i] = t_idx.get(ctype, N_CARD_TYPE)
        rarities[i] = r_idx.get(rarity, N_CARD_RARITY)
    return tuple(types), tuple(rarities)


# Event option-index embedding rows (options beyond this clamp to the last).
MAX_OPT_IDX = 8


# Max upgrade count an embedding slot is allocated for (Searing Blow etc. can
# exceed normal +1; clamp into this range).
MAX_UPGRADE = 16
