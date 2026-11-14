"""Frozen interface contracts for the Ironclad e2e no-search PPO (GH #1097, WP3a).

This module is the single source of truth for what every work package shares:

* the token **sections** and their order, the categorical and continuous
  **columns** (the 15 + 122 S_ctx columns of #1081 bit-for-bit, then the
  trailing e2e columns), and which columns each section fills;
* the **action-kind** and **decision-type** id tables, including the ids
  reserved for accepted real-game gaps and the spare ids (D9);
* the **dedupe** rule that turns legal actions into merged option classes;
* the size **caps**;
* what the policy may read (`POLICY_CALLS`, `POLICY_FILTERS`, `honest_obs`,
  `honest_event_deltas`) and what only the value path may (`PRIV_CALLS`);
* the **env decision** the env yields for each stored decision, and its
  validator;
* the actor-to-learner **row batch** (field names, dtypes, ragged layout,
  bootstrap rows) and its validator;
* the **stamps** (`schema_sha256`, vocabulary sha256, fill state, combat
  teacher) recorded in cfg, checkpoints, stats, eval rows and row batches.

`docs/experiments/2026-11-03-ironclad-e2e-v111/SCHEMA.md` is the prose.

Import contract: the standard library and numpy only. Importing this module
never imports torch or `sts2_rs`; the few helpers that need the engine or the
#1081 snapshot code import them inside the function.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

SCHEMA_NAME: Final = "ironclad-e2e-v111"
# Bumped by hand on every change before G-schema; after G-schema the schema is
# frozen and only reserved ids and reserved columns may start carrying data
# (which leaves `schema_sha256` unchanged; see `FILL_EVENTS`).
SCHEMA_REVISION: Final = 2
GAME_VERSION: Final = "v0.111.0"


class SchemaError(ValueError):
    """A row batch, decision or table violates the frozen schema."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. S_ctx mirror (#1081, ppo/combat_ctx/tokens.py). Hardcoded on purpose: the
#    schema hash must not move when that module changes, and a test pins every
#    constant below against it.
# ═══════════════════════════════════════════════════════════════════════════

SCTX_SECTIONS: Final[tuple[str, ...]] = (
    "STATE",
    "HAND",
    "DRAW_UNK",
    "DRAW_TOP",
    "DRAW_BOT",
    "DISCARD",
    "EXHAUST",
    "PLAY",
    "RELIC",
    "POTION",
    "PPOWER",
    "ENEMY",
    "EPOWER",
    "EHIST",
    "ACT_PLAY",
    "ACT_END",
    "ACT_POTION",
    "ACT_CHOOSE",
    "ACT_SKIP",
    "ACT_OTHER",
    "ACT_RL2",
)
SCTX_N_SECTION: Final = 21
SCTX_ACT_SECTIONS: Final = frozenset(range(14, 21))

SCTX_SLOTS: Final[tuple[str, ...]] = (
    "card",
    "upgraded",
    "relic",
    "potion",
    "power",
    "monster",
    "enchantment",
    "affliction",
    "intent",
    "enemy_slot",
    "hand_slot",
    "choice_kind",
    "card_type",
    "choice_pile",
    "rl2key",
)
SCTX_N_CAT: Final = 15
SCTX_CONTENT_SLOTS: Final = ("card", "relic", "potion", "power", "monster", "enchantment", "rl2key")
# Small categorical S_ctx slots the student reads directly. It never reads the
# content slots: those hold the combat teacher's RareMap'd rows; the student
# reads content through the trailing ID columns instead.
SCTX_STUDENT_SLOTS: Final = (
    "upgraded",
    "affliction",
    "intent",
    "enemy_slot",
    "hand_slot",
    "choice_kind",
    "card_type",
    "choice_pile",
)
# `tokens.vocab().digest()` on the W1 engine wheel (879e44015). A table is
# `[names..., UNK, PAD]`; PAD (= size - 1) is the absent value of a slot.
SCTX_TABLE_SIZES: Final[Mapping[str, int]] = {
    "card": 601,
    "upgraded": 4,
    "relic": 299,
    "potion": 66,
    "power": 272,
    "monster": 123,
    "enchantment": 25,
    "affliction": 11,
    "intent": 16,
    "enemy_slot": 7,
    "hand_slot": 14,
    "choice_kind": 33,
    "card_type": 8,
    "choice_pile": 8,
    "rl2key": 1267,
}
SCTX_PAD: Final[tuple[int, ...]] = tuple(SCTX_TABLE_SIZES[s] - 1 for s in SCTX_SLOTS)
SCTX_INTENT_KINDS: Final = (
    "Attack",
    "Defend",
    "Buff",
    "Debuff",
    "DebuffStrong",
    "Sleep",
    "Stun",
    "StatusCard",
    "Summon",
    "Heal",
    "DeathBlow",
    "Escape",
    "Unknown",
    "NONE",
)

# (name, transform) for the 122 continuous S_ctx columns, in column order.
# `x/k` means the raw value divided by k; `flag` is 0/1. Absent values are 0.
_CARD_BLOCK = (
    ("COST", "cost (cost_override, else base_cost) / 3"),
    ("BASE_COST", "base_cost / 3"),
    ("DMG", "base_damage / 20"),
    ("BLK", "base_block / 20"),
    ("CALC", "calculated_value / 20"),
    ("HAS_CALC", "flag: calculated_value present"),
    ("ETHEREAL", "flag"),
    ("UNPLAYABLE", "flag"),
    ("INNATE", "flag"),
    ("SLY", "flag"),
    ("SLY_TURN", "flag"),
    ("RETAIN_TURN", "flag"),
    ("ENCH_AMT", "enchantment_amount / 5"),
    ("AFF_STACKS", "affliction_stacks / 5"),
    ("COUNT", "pile multiplicity / 5; DRAW_TOP/DRAW_BOT: pin rank (1 = first) / 5"),
    ("PLAYABLE", "flag: some legal play uses this hand slot"),
    ("DMG_PREV", "max card_damage_preview over targets / 30"),
    ("BLK_PREV", "card_block_preview / 20"),
)
_STATE_BLOCK = (
    ("HP_FRAC", "hp / max(1, max_hp)"),
    ("HP", "hp / 100"),
    ("MAXHP", "max_hp / 100"),
    ("BLOCK", "block / 30"),
    ("ENERGY", "energy / 5"),
    ("MAXE", "max_energy / 5"),
    ("STARS", "stars / 5"),
    ("ROUND", "round / 20"),
    ("PLAYED", "cards played this turn / 10"),
    ("N_HAND", "hand size / 30"),
    ("N_DRAW", "draw pile size / 30"),
    ("N_DISC", "discard pile size / 30"),
    ("N_EXH", "exhaust pile size / 30"),
    ("POT_SLOTS", "potion slots / 5"),
    ("IS_BOSS", "flag: room == boss"),
    ("IS_ELITE", "flag: room == elite"),
    ("IS_HALL", "flag: room == monster"),
    ("CH_MIN", "pending choice min_select / 5"),
    ("CH_MAX", "pending choice max_select / 5"),
    ("CH_SKIP", "flag: pending choice can_skip"),
    ("INCOMING", "sum of live enemies' attack intent totals / 50"),
    ("UNBLOCKED", "max(0, incoming - block) / 50"),
)
_ENEMY_BLOCK = (
    ("E_HPF", "hp / max(1, max_hp)"),
    ("E_HP", "hp / 200"),
    ("E_MAXHP", "max_hp / 200"),
    ("E_BLOCK", "block / 30"),
    ("E_ALIVE", "flag"),
    ("E_REMOVED", "flag: removed_after_death"),
    ("E_AMT", "head intent amount / 30"),
    ("E_EFF", "head intent effective_amount / 30"),
    ("E_TIMES", "head intent times / 5"),
    ("E_TOT", "head intent attack total (effective * times) / 50"),
)
_EHIST_BLOCK = (
    ("H_LAG", "turns ago / 3"),
    ("H_AMT", "amount / 30"),
    ("H_EFF", "effective_amount / 30"),
    ("H_TIMES", "times / 5"),
    ("H_ALIVE", "flag: the enemy is alive now"),
)
_POWER_RELIC_BLOCK = (
    ("P_AMT", "power amount / 10"),
    ("P_DISP", "power display_amount / 10"),
    ("P_CTR", "power counter / 10"),
    ("R_CTR", "relic display counter / 10"),
    ("R_HAS", "flag: relic shows a counter"),
    ("R_WAX", "flag: wax relic"),
    ("R_MELT", "flag: melted wax relic"),
)
_ACTION_BLOCK = (
    ("T_HPF", "target hp / max(1, max_hp)"),
    ("T_EHP", "target (hp + block) / 100"),
    ("T_DMG", "card_damage_preview against the target / 30"),
    ("N_SEL", "len(selection) / 3"),
)
SCTX_CONT: Final[tuple[tuple[str, str], ...]] = (
    _CARD_BLOCK
    + _STATE_BLOCK
    + _ENEMY_BLOCK
    + _EHIST_BLOCK
    + _POWER_RELIC_BLOCK
    + _ACTION_BLOCK
    + tuple((f"RL2_S{i:02d}", f"rollout_features(cs, 1).state[{i}]") for i in range(32))
    + tuple((f"RL2_X{i:02d}", f"rollout_features(cs, 1).x[action][{i}]") for i in range(24))
)
SCTX_N_CONT: Final = 122
SCTX_C: Final[Mapping[str, int]] = {n: i for i, (n, _) in enumerate(SCTX_CONT)}
SCTX_CARD_COLS: Final = tuple(range(0, 18))
SCTX_PILE_CARD_COLS: Final = tuple(range(0, 15))  # card block without the hand-only extras
SCTX_STATE_COLS: Final = tuple(range(18, 40))
SCTX_ENEMY_COLS: Final = tuple(range(40, 50))
SCTX_EHIST_COLS: Final = tuple(range(50, 55))
SCTX_POWER_COLS: Final = tuple(range(55, 58))
SCTX_RELIC_COLS: Final = tuple(range(58, 62))
SCTX_TARGET_COLS: Final = tuple(range(62, 65))
SCTX_NSEL_COL: Final = 65
SCTX_RL2_STATE_COLS: Final = tuple(range(66, 98))
SCTX_RL2_ACTION_COLS: Final = tuple(range(98, 122))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sections. Ids 0-21 are #1081's (21 = its critic-only SEC_PRIV, the true
#    top draw cards); new sections only append. Reserved ids carry no tokens
#    until their fill event (`FILL_EVENTS`).
# ═══════════════════════════════════════════════════════════════════════════

_SECTION_TABLE: Final[tuple[str, ...]] = (
    *SCTX_SECTIONS,  # 0-20
    "PRIV_DRAW",  # 21: true top draw cards (value only; #1081's SEC_PRIV)
    "RUNCTX",  # 22: the state token, position 0 of every row
    "BOSS",  # 23: the current act's boss(es), 1-2 tokens
    "SUM_DRAW",  # 24: pooled summaries (5 tokens, every row)
    "SUM_DISCARD",  # 25
    "SUM_EXHAUST",  # 26
    "SUM_DECK",  # 27
    "SUM_RELICS",  # 28
    "DECK",  # 29: OOC deck multiset, one token per distinct visible card
    "OFFER",  # 30: offered content that is not itself an option
    "MAPNODE",  # 31: reachable map cone (map decisions only)
    "OPT_CARD",  # 32: option tokens (OOC), one per merged class
    "OPT_RELIC",  # 33
    "OPT_POTION",  # 34
    "OPT_MOVE",  # 35
    "OPT_EVENT",  # 36
    "OPT_REST",  # 37
    "OPT_FIXED",  # 38: content-free options (skip, leave, decline, confirm)
    "RES_SHOP_SHELF",  # 39: reserved - unaffordable shelf items
    "RES_OPT_REWARD",  # 40: reserved - #1069 reward-screen items
    "RES_OPT_SPHERE",  # 41: reserved - Crystal Sphere tools and cells
    "RES_OPT_POTION_USE",  # 42: reserved - OOC potion use and discard
    "RES_OPT_MERCHANT",  # 43: reserved - Fake Merchant shelf
    "RES_44",
    "RES_45",
    "RES_46",
    "RES_47",
    "PRIV_ENEMY",  # 48: true next moves per enemy (value only)
    "PRIV_RUN",  # 49: this act's rooms and the bosses (value only; no future maps)
    "RES_PRIV_50",
    "RES_PRIV_51",
    "RES_PRIV_52",
    "RES_PRIV_53",
    "RES_PRIV_54",
    "RES_PRIV_55",
    "RES_56",
    "RES_57",
    "RES_58",
    "RES_59",
    "RES_60",
    "RES_61",
    "RES_62",
    "RES_63",
)
SECTIONS: Final = _SECTION_TABLE
N_SECTION: Final = 64  # section-embedding table size
SEC: Final[Mapping[str, int]] = {s: i for i, s in enumerate(SECTIONS)}
PRIV_SECTIONS: Final = frozenset({21, 48, 49, 50, 51, 52, 53, 54, 55})
# The sections no fill event has taken ("chars" took RES_44 and RES_45, section 4b).
RESERVED_SECTIONS: Final = frozenset(
    i for i, s in enumerate(SECTIONS) if s.startswith("RES_") and s not in ("RES_44", "RES_45")
)
SUMMARY_SECTIONS: Final = (24, 25, 26, 27, 28)
OOC_OPTION_SECTIONS: Final = frozenset(range(32, 39)) | frozenset({39, 40, 41, 42, 43})
OPTION_SECTIONS: Final = SCTX_ACT_SECTIONS | OOC_OPTION_SECTIONS
# S_ctx sections an out-of-combat row may reuse (same meaning, no S_ctx block).
OOC_SCTX_REUSE: Final = frozenset({SEC["RELIC"], SEC["POTION"]})


# ═══════════════════════════════════════════════════════════════════════════
# 3. Columns. cat = int16 [T, N_CAT]: 15 S_ctx slots, then trailing columns
#    whose absent value is 0. cont = float16 [T, N_CONT]: 122 S_ctx columns,
#    then trailing columns, absent value 0.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Column:
    name: str
    size: int  # embedding-table size (cat) or 0 (cont)
    transform: str
    status: str  # "live" | "reserved:<fill event>"


VOCAB_ROWS: Final = 4096

X_CAT: Final[tuple[Column, ...]] = (
    Column("X_DTYPE", 64, "decision-type id (RUNCTX only)", "live"),
    Column("X_KIND", 128, "action-kind id of the option (every option token)", "live"),
    Column("X_ID_MAIN", VOCAB_ROWS, "primary content id (see SECTION_IDS)", "live"),
    Column("X_ID_MOD", VOCAB_ROWS, "modifier content id, e.g. a card's enchantment", "live"),
    Column("X_ID_REL", VOCAB_ROWS, "related content id, e.g. a play's target monster", "live"),
    Column(
        "X_ID_AUX",
        VOCAB_ROWS,
        "auxiliary id; RUNCTX: the current fight's encounter on combat rows, NONE out of combat",
        "reserved:WP-ENG-1",
    ),
    Column("X_ROOM", 16, "room-type id (ROOM_TYPES)", "live"),
    Column("X_POS", 64, "1 + position (slot, column, option index), capped at 63", "live"),
    Column("X_INTENT_T2", 16, "ENEMY: telegraph intent 2 kind (INTENT_IDS)", "reserved:WP-ENG-1"),
    Column("X_INTENT_T3", 16, "ENEMY: telegraph intent 3 kind (INTENT_IDS)", "reserved:WP-ENG-1"),
    Column("X_FC1", 16, "ENEMY: forecast step 1 head-intent kind", "reserved:WP-ENG-1"),
    Column("X_FC2", 16, "ENEMY: forecast step 2 head-intent kind", "reserved:WP-ENG-1"),
    Column("X_FC3", 16, "ENEMY: forecast step 3 head-intent kind", "reserved:WP-ENG-1"),
    Column("X_RES0", 16, "spare", "reserved:spare"),
    Column("X_RES1", 16, "spare", "reserved:spare"),
    Column("X_RES2", 16, "spare", "reserved:spare"),
    Column("X_RES3", 16, "spare", "reserved:spare"),
)
N_CAT: Final = SCTX_N_CAT + len(X_CAT)  # 32
XC: Final[Mapping[str, int]] = {c.name: SCTX_N_CAT + i for i, c in enumerate(X_CAT)}


def _cols(names: Sequence[tuple[str, str]], status: str = "live") -> tuple[Column, ...]:
    return tuple(Column(n, 0, t, status) for n, t in names)


_RUNCTX_COLS = (
    *_cols(
        (
            ("R_ACT", "act_index (1-3) / 3"),
            ("R_FLOOR", "total_floor / 50"),
            ("R_ACT_FLOOR", "floor within the act / 17"),
            ("R_ROWS_TO_BOSS", "(boss row - current row) / 17"),
            ("R_BOSSES_LEFT", "bosses left in the run, this act's included / 4"),
            ("R_HP", "hp / 100"),
            ("R_MAXHP", "max_hp / 100"),
            ("R_HP_FRAC", "hp / max(1, max_hp)"),
            ("R_GOLD", "gold / 300"),
            ("R_DECK", "deck size / 40"),
            ("R_DECK_UPG", "upgraded deck cards / 20"),
            ("R_RELICS", "relics held / 20"),
            ("R_POTIONS", "potions held / 5"),
            ("R_POT_SLOTS", "potion slots / 5"),
            ("R_ASC", "ascension / 10"),
            ("R_IN_COMBAT", "flag"),
            ("R_SEL_MIN", "selection min picks / 5 (OOC multi-pick, combat choice)"),
            ("R_SEL_MAX", "selection max picks / 5"),
            ("R_SEL_PICKED", "picks already made in the open selection / 5"),
            ("R_SEL_CAN_SKIP", "flag: the selection can end now"),
            ("R_N_CLASSES", "merged options / 64"),
            ("R_DOUBLE_BOSS", "flag: this act ends in two bosses (A10 act 3)"),
        )
    ),
    Column("R_HIST_MISSING", 0, "flag: intent history unavailable (live gap)", "reserved:D9"),
    Column("R_RES0", 0, "spare", "reserved:spare"),
)
_ROUTE_COLS = _cols(
    (
        ("M_ELITE_MIN", "fewest elites on a path from the node to the boss / 5"),
        ("M_ELITE_MAX", "most elites on such a path / 5"),
        ("M_REST_MIN", "fewest rest sites / 5"),
        ("M_REST_MAX", "most rest sites / 5"),
        ("M_SHOP_MAX", "most shops / 5"),
        ("M_UNKNOWN_MAX", "most unknown (?) nodes / 5"),
        ("M_FIGHTS_TO_REST", "fewest monster + elite nodes before the first rest / 5"),
        ("M_ROWS_LEFT", "rows from the node to the boss row / 17"),
    )
)
_MAPNODE_COLS = _cols(
    (
        ("N_ROW_OFF", "(node row - current row) / 17"),
        ("N_COL", "column / 6"),
        ("N_NEXT", "flag: a legal move target now"),
        ("N_PATHS", "log1p(paths from the current node through this node) / 5"),
    )
)
# O_EV_* hold `honest_event_deltas()` (section 11), never the raw
# `event_option_infos()` preview, which resolves the option on a clone and so
# applies a randomly rolled relic's on-pickup effect (SCHEMA.md section 2.7).
_OPTION_COLS = _cols(
    (
        ("O_PRICE", "gold price / 300 (shop items, removal service)"),
        ("O_EV_GOLD", "honest event gold delta / 100 (0 if the option gains a relic)"),
        ("O_EV_HP", "honest event hp delta / 20 (0 if the option gains a relic)"),
        ("O_EV_MAXHP", "honest event max_hp delta / 10 (0 if the option gains a relic)"),
        ("O_EV_RELICS_GAIN", "flag: the option gains at least one relic"),
        ("O_EV_POTIONS_GAIN", "honest potions gained / 3 (0 if the option gains a relic)"),
        ("O_EV_CARDS_GAIN", "honest cards gained / 3 (0 if the option gains a relic)"),
        ("O_EV_CARDS_LOSE", "cards lost / 3"),
        ("O_EV_RELICS_LOSE", "relics lost / 3"),
        ("O_EV_POTIONS_LOSE", "potions lost / 3"),
    )
)
_CARD_X_COLS = _cols(
    (
        ("K_PICKED", "flag: already selected in the open multi-pick selection"),
        ("K_DECK_COPIES", "copies of this card already in the deck / 5"),
    )
)
_SUMMARY_COLS = _cols(
    (
        ("U_N", "pool size / 40"),
        ("U_UPG", "upgraded members / 20"),
        ("U_ENCH", "enchanted members / 10"),
        ("U_DISTINCT", "distinct members / 40"),
    )
)
_BOSS_COLS = _cols((("B_ORDER", "0 first boss, 1 second boss"), ("B_ACT", "act / 3")))
_ENEMY_X_COLS = (
    _cols(
        (
            ("E_N_INTENTS", "intents in the current telegraph / 3"),
            ("E_T2_EFF", "telegraph intent 2 effective_amount / 30"),
            ("E_T2_TIMES", "telegraph intent 2 times / 5"),
            ("E_T3_EFF", "telegraph intent 3 effective_amount / 30"),
            ("E_T3_TIMES", "telegraph intent 3 times / 5"),
        ),
        "reserved:WP-ENG-1",
    )
    + _cols(
        tuple(
            (f"E_F{s}_{k}", f"forecast step {s}: {d}")
            for s in (1, 2, 3)
            for k, d in (
                ("CERT", "certainty (highest-weight branch share, product over steps)"),
                ("AMT", "head-intent amount / 30"),
                ("TIMES", "head-intent times / 5"),
                ("TOT", "attack total (amount * times) / 50"),
            )
        ),
        "reserved:WP-ENG-1",
    )
    + _cols(
        (
            ("E_MAXDMG", "most damage dealable to this enemy this turn (knapsack) / 50"),
            ("E_KILLABLE", "flag: E_MAXDMG >= hp + block"),
        )
    )
)
_STATE_X_COLS = _cols((("S_HP_LOSS_END", "hp lost if the turn ended now / 50"),))
_PRIV_RUN_COLS = _cols(
    tuple(
        (f"P_ROOMS_{r}", f"this act's remaining {r.lower()} rooms / 10")
        for r in ("MONSTER", "ELITE", "EVENT", "SHOP", "REST", "TREASURE")
    )
)
_SPARE_CONT = tuple(Column(f"X_C_RES{i}", 0, "spare", "reserved:spare") for i in range(8))

X_CONT: Final[tuple[Column, ...]] = (
    _RUNCTX_COLS
    + _ROUTE_COLS
    + _MAPNODE_COLS
    + _OPTION_COLS
    + _CARD_X_COLS
    + _SUMMARY_COLS
    + _BOSS_COLS
    + _ENEMY_X_COLS
    + _STATE_X_COLS
    + _PRIV_RUN_COLS
    + _SPARE_CONT
)
N_CONT: Final = SCTX_N_CONT + len(X_CONT)
XF: Final[Mapping[str, int]] = {c.name: SCTX_N_CONT + i for i, c in enumerate(X_CONT)}


def _xf(prefix: str) -> tuple[int, ...]:
    return tuple(i for n, i in XF.items() if n.startswith(prefix))


# Content-prior columns: virtual, never stored in rows. The net gathers
# PRIOR[X_ID_MAIN] (and the weighted pool mean for SUM_* tokens) from a frozen
# float32 [VOCAB_ROWS, 48] buffer that the decomp extractor writes; an
# all-zero buffer (the launch state) is inert.
PRIOR_COLUMNS: Final[tuple[str, ...]] = (
    "TYPE_ATTACK",
    "TYPE_SKILL",
    "TYPE_POWER",
    "TYPE_STATUS",
    "TYPE_CURSE",
    "TYPE_QUEST",
    "RAR_BASIC",
    "RAR_COMMON",
    "RAR_UNCOMMON",
    "RAR_RARE",
    "RAR_SPECIAL",
    "RAR_BOSS_ANCIENT",
    "COST",
    "X_COST",
    "COST_UPG_DELTA",
    "DMG",
    "HITS",
    "BLOCK",
    "DMG_UPG_DELTA",
    "BLOCK_UPG_DELTA",
    "MAGIC",
    "MAGIC_UPG_DELTA",
    "KW_EXHAUST",
    "KW_ETHEREAL",
    "KW_INNATE",
    "KW_RETAIN",
    "KW_UNPLAYABLE",
    "KW_SLY",
    "KW_ETERNAL",
    "TGT_SELF",
    "TGT_ENEMY",
    "TGT_ALL_ENEMIES",
    "TGT_RANDOM",
    "POOL_IRONCLAD",
    "POOL_COLORLESS",
    "POOL_OTHER",
    "DRAW",
    "ENERGY",
    "APPLIES_DEBUFF",
    "GAINS_STRENGTH",
    "EXHAUSTS_OTHER",
    "MON_MAXHP",
    "MON_IS_ELITE",
    "MON_IS_BOSS",
    "ACT",
    "POWER_IS_DEBUFF",
    "PRIOR_RES0",
    "MISSING",
)
N_PRIOR: Final = 48

ROOM_TYPES: Final = (
    "none",
    "unassigned",
    "unknown",
    "monster",
    "elite",
    "boss",
    "shop",
    "rest_site",
    "treasure",
    "ancient",
    "event",
)
ROOM_ID: Final[Mapping[str, int]] = {r: i for i, r in enumerate(ROOM_TYPES)}
# Trailing intent columns: 0 = none, else 1 + index in SCTX_INTENT_KINDS
# (MultiAttack folds into Attack, as #1081's intent log does).
INTENT_IDS: Final[Mapping[str, int]] = {k: i + 1 for i, k in enumerate(SCTX_INTENT_KINDS[:-1])}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-section live columns. A dense row is lossless under the compact
#    layout iff every column outside its section's live set holds the absent
#    value (S_ctx cat: PAD; trailing cat: 0; cont: 0). `check_compact_lossless`
#    asserts it; S1b's compact rows store only the live columns.
# ═══════════════════════════════════════════════════════════════════════════

_S = {s: i for i, s in enumerate(SCTX_SLOTS)}


def _cat(*slots: str) -> tuple[int, ...]:
    return tuple(_S[s] if s in _S else XC[s] for s in slots)


_CARD_CATS = ("card", "upgraded", "card_type", "enchantment", "affliction", "X_ID_MAIN", "X_ID_MOD")
# Out-of-combat tokens never fill S_ctx content slots (they stay PAD): only
# the combat teacher reads those, and it never sees an out-of-combat row.
_OOC_CARD_CATS = ("upgraded", "card_type", "X_ID_MAIN", "X_ID_MOD")
_PILE = (_cat(*_CARD_CATS), (*SCTX_PILE_CARD_COLS, XF["K_PICKED"]))
_ACT_TAIL = SCTX_RL2_ACTION_COLS

_SECTION_LIVE_BASE: Final[Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]]] = {
    SEC["STATE"]: (
        _cat("choice_kind"),
        SCTX_STATE_COLS + SCTX_RL2_STATE_COLS + (XF["S_HP_LOSS_END"],),
    ),
    SEC["HAND"]: (_cat(*_CARD_CATS, "hand_slot"), (*SCTX_CARD_COLS, XF["K_PICKED"])),
    SEC["DRAW_UNK"]: _PILE,
    SEC["DRAW_TOP"]: _PILE,
    SEC["DRAW_BOT"]: _PILE,
    SEC["DISCARD"]: _PILE,
    SEC["EXHAUST"]: _PILE,
    SEC["PLAY"]: _PILE,
    SEC["RELIC"]: (_cat("relic", "X_ID_MAIN"), SCTX_RELIC_COLS),
    SEC["POTION"]: (_cat("potion", "X_ID_MAIN", "X_POS"), ()),
    SEC["PPOWER"]: (_cat("power", "X_ID_MAIN"), SCTX_POWER_COLS),
    SEC["ENEMY"]: (
        _cat(
            "monster",
            "enemy_slot",
            "intent",
            "X_ID_MAIN",
            "X_INTENT_T2",
            "X_INTENT_T3",
            "X_FC1",
            "X_FC2",
            "X_FC3",
        ),
        SCTX_ENEMY_COLS + _xf("E_"),
    ),
    # Powers and stolen cards on an enemy.
    SEC["EPOWER"]: (
        _cat(
            "power",
            "monster",
            "enemy_slot",
            "card",
            "upgraded",
            "card_type",
            "X_ID_MAIN",
            "X_ID_REL",
        ),
        SCTX_POWER_COLS + SCTX_PILE_CARD_COLS,
    ),
    SEC["EHIST"]: (_cat("intent", "monster", "enemy_slot", "X_ID_REL"), SCTX_EHIST_COLS),
    SEC["ACT_PLAY"]: (
        _cat(*_CARD_CATS, "hand_slot", "enemy_slot", "monster", "rl2key", "X_KIND", "X_ID_REL"),
        SCTX_CARD_COLS + SCTX_TARGET_COLS + _ACT_TAIL,
    ),
    SEC["ACT_END"]: (_cat("rl2key", "X_KIND"), _ACT_TAIL),
    SEC["ACT_POTION"]: (
        _cat("potion", "enemy_slot", "rl2key", "X_KIND", "X_ID_MAIN", "X_ID_REL", "X_POS"),
        _ACT_TAIL,
    ),
    SEC["ACT_CHOOSE"]: (
        _cat(
            "card",
            "upgraded",
            "card_type",
            "choice_pile",
            "hand_slot",
            "choice_kind",
            "rl2key",
            "X_KIND",
            "X_ID_MAIN",
            "X_ID_MOD",
        ),
        (SCTX_NSEL_COL, *_ACT_TAIL, XF["K_PICKED"]),
    ),
    SEC["ACT_SKIP"]: (_cat("choice_kind", "rl2key", "X_KIND"), _ACT_TAIL),
    SEC["ACT_OTHER"]: (_cat("rl2key", "X_KIND"), _ACT_TAIL),
    SEC["ACT_RL2"]: (_cat("rl2key"), SCTX_RL2_STATE_COLS + _ACT_TAIL),  # never in e2e rows
    SEC["PRIV_DRAW"]: _PILE,
    SEC["RUNCTX"]: (_cat("X_DTYPE", "X_ID_MAIN", "X_ID_AUX", "X_ROOM"), _xf("R_")),
    SEC["BOSS"]: (_cat("X_ID_MAIN"), _xf("B_")),
    **{s: ((), _xf("U_")) for s in SUMMARY_SECTIONS},
    SEC["DECK"]: (_cat(*_OOC_CARD_CATS), SCTX_PILE_CARD_COLS),
    SEC["OFFER"]: (_cat("X_KIND", "X_ID_MAIN", "X_ID_MOD"), (XF["O_PRICE"],)),
    SEC["MAPNODE"]: (_cat("X_ROOM", "X_POS"), _xf("N_")),
    SEC["OPT_CARD"]: (
        _cat(*_OOC_CARD_CATS, "X_KIND", "X_POS"),
        (*SCTX_PILE_CARD_COLS, XF["O_PRICE"], XF["K_DECK_COPIES"], XF["K_PICKED"]),
    ),
    SEC["OPT_RELIC"]: (_cat("X_KIND", "X_ID_MAIN", "X_POS"), (XF["O_PRICE"],)),
    SEC["OPT_POTION"]: (_cat("X_KIND", "X_ID_MAIN", "X_POS"), (XF["O_PRICE"],)),
    SEC["OPT_MOVE"]: (_cat("X_KIND", "X_ROOM", "X_POS"), _xf("M_")),
    SEC["OPT_EVENT"]: (_cat("X_KIND", "X_ID_MAIN", "X_ID_MOD", "X_POS"), _xf("O_EV_")),
    SEC["OPT_REST"]: (_cat("X_KIND", "X_ID_MAIN", "X_POS"), ()),
    SEC["OPT_FIXED"]: (_cat("X_KIND", "X_POS"), (XF["O_PRICE"],)),
    SEC["PRIV_ENEMY"]: (_cat("enemy_slot", "X_ID_MAIN", "X_ID_MOD", "X_ID_REL", "X_ID_AUX"), ()),
    SEC["PRIV_RUN"]: (_cat("X_ID_MAIN", "X_ID_MOD", "X_ID_REL"), _xf("P_ROOMS_")),
}

# What each X_ID_* column holds, per section ("-" = always 0). Documentation
# and the tokenizer's contract; `vocab.REGISTRIES` names the id blocks.
SECTION_IDS: Final[Mapping[str, tuple[str, str, str, str]]] = {
    "HAND/DRAW_*/DISCARD/EXHAUST/PLAY/DECK/PRIV_DRAW": ("card", "enchantment", "-", "-"),
    "RELIC": ("relic", "-", "-", "-"),
    "POTION": ("potion", "-", "-", "-"),
    "PPOWER": ("power", "-", "-", "-"),
    "ENEMY": ("monster", "-", "-", "-"),
    "EPOWER": ("power, or the stolen card", "-", "owner monster", "-"),
    "EHIST": ("-", "-", "monster", "-"),
    "ACT_PLAY": ("card", "enchantment", "target monster", "-"),
    "ACT_POTION": ("potion", "-", "target monster", "-"),
    "ACT_CHOOSE": ("card", "enchantment", "-", "-"),
    "RUNCTX": ("event (event and Neow decisions)", "-", "-", "this fight's encounter (combat)"),
    "BOSS": ("boss encounter", "-", "-", "-"),
    "OFFER": ("pending potion / pending enchantment / offered card", "enchantment", "-", "-"),
    "OPT_CARD": ("card", "enchantment", "-", "-"),
    "OPT_RELIC": ("relic", "-", "-", "-"),
    "OPT_POTION": ("potion (the one to discard, or to buy)", "-", "-", "-"),
    "OPT_EVENT": ("event", "relic named by the option (Neow)", "-", "-"),
    "OPT_REST": ("rest option", "-", "-", "-"),
    "PRIV_ENEMY": ("next move step 1", "step 2", "step 3", "monster"),
    "PRIV_RUN": ("next act boss", "act-3 second boss", "act-3 first boss", "-"),
}


# ═══════════════════════════════════════════════════════════════════════════
# 4b. The "chars" fill event (GH #1373): the other characters' mechanics.
#
#    The schema was frozen for Ironclad. A row that carries the "chars" fill
#    event has its character (`X_CHAR` on RUNCTX, nonzero), the player's pet
#    (`ALLY`: Necrobinder's Osty), the orb belt (`ORB`) and the doomed flag
#    (`E_DOOMED` on ENEMY). Every row of another character's run carries it.
#    An Ironclad row carries it only in a launch that declares the event
#    (`factory.fill_chars()`): then an Ironclad with orbs (Potion of Capacity,
#    Infused Core) or facing Doom (Potion of Doom) shows them. Without the
#    declaration every Ironclad row, fixture and checkpoint stays as it was.
#
#    It is a fill event: it takes reserved sections and spare columns under
#    alias names and adds no tensor shape. The hashed tables above keep the
#    reserved names (`RES_44`, `X_RES0`, `X_C_RES0`, ...), so `schema_sha256`
#    does not move and an Ironclad checkpoint loads unchanged; the net starts
#    the event's inputs at zero weight (the identity rule, section 2.12).
# ═══════════════════════════════════════════════════════════════════════════

FILL_CHARS: Final = "chars"
# The playable characters, as `rm.character` names them. X_CHAR holds 1 + the
# index; 0, the column's absent value, is a row without "chars" data.
# Fool exists from v0.113.0 on; it has its id so that a later build's run encodes.
CHARACTERS: Final = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool")
CHAR_ID: Final[Mapping[str, int]] = {c: i + 1 for i, c in enumerate(CHARACTERS)}
# The launch character: its rows carry the "chars" event only when declared.
BASE_CHARACTER: Final = CHARACTERS[0]

# Alias -> the reserved name it takes. Aliases resolve through SEC, XC and XF
# like any other name; the hashed tables list only the reserved names.
CHARS_SECTIONS: Final[Mapping[str, str]] = {"ALLY": "RES_44", "ORB": "RES_45"}
CHARS_CAT: Final[Mapping[str, str]] = {"X_CHAR": "X_RES0"}
CHARS_CONT: Final[Mapping[str, str]] = {
    "C_ORB_PASSIVE": "X_C_RES0",
    "C_ORB_EVOKE": "X_C_RES1",
    "E_DOOMED": "X_C_RES2",
}
# What each alias holds (documentation; SCHEMA.md section 13 repeats it).
CHARS_DOC: Final[Mapping[str, str]] = {
    "ALLY": "the player's pet, one token while the run has one (Necrobinder's Osty)",
    "ORB": "one token per orb slot, front (next evoked) first; an empty slot has X_ID_MAIN EMPTY",
    "X_CHAR": "the run's character (CHAR_ID, 1-6; 0 = no chars data; RUNCTX only)",
    "C_ORB_PASSIVE": "ORB: passive value with Focus / 10",
    "C_ORB_EVOKE": "ORB: evoke value with Focus / 20",
    "E_DOOMED": "ENEMY flag: Doom >= hp, the enemy dies at the end of its turn",
}
# An orb belt has at most 10 slots (OrbCmd.AddSlots caps the capacity at 10).
MAX_ORB_TOKENS: Final = 10
MAX_ALLY_TOKENS: Final = 1

SEC.update({a: SEC[r] for a, r in CHARS_SECTIONS.items()})  # type: ignore[attr-defined]
XC.update({a: XC[r] for a, r in CHARS_CAT.items()})  # type: ignore[attr-defined]
XF.update({a: XF[r] for a, r in CHARS_CONT.items()})  # type: ignore[attr-defined]

# Sections that sit between the BOSS tokens and the S_ctx block of a combat
# row, in this order. They never occur out of combat.
COMBAT_EXT_SECTIONS: Final = (SEC["ALLY"], SEC["ORB"])
CHARS_SECTION_IDS: Final = frozenset(SEC[a] for a in CHARS_SECTIONS)

# The live columns the event adds: new sections, and columns on old ones.
_CHARS_LIVE: Final[Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]]] = {
    SEC["ALLY"]: (
        _cat("X_ID_MAIN"),
        tuple(SCTX_C[n] for n in ("E_HPF", "E_HP", "E_MAXHP", "E_ALIVE")),
    ),
    SEC["ORB"]: (_cat("X_ID_MAIN", "X_POS"), (XF["C_ORB_PASSIVE"], XF["C_ORB_EVOKE"])),
    SEC["RUNCTX"]: (_cat("X_CHAR"), ()),
    SEC["ENEMY"]: ((), (XF["E_DOOMED"],)),
}


def _with_fill(
    base: Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]],
    add: Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    out = dict(base)
    for s, (c, f) in add.items():
        bc, bf = out.get(s, ((), ()))
        out[s] = (bc + tuple(x for x in c if x not in bc), bf + tuple(x for x in f if x not in bf))
    return out


# Every section's live columns: the frozen table plus the fill events' slots.
# `schema_sha256` hashes `_SECTION_LIVE_BASE` only.
SECTION_LIVE: Final[Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]]] = _with_fill(
    _SECTION_LIVE_BASE, _CHARS_LIVE
)
# The reserved inputs each fill event outside the column statuses takes
# (`net.layout(filled)` inits them live for a launch that declares the event).
FILL_SLOTS: Final[Mapping[str, Mapping[str, tuple[int, ...]]]] = {
    FILL_CHARS: {
        "sections": tuple(sorted(CHARS_SECTION_IDS)),
        "cat": tuple(XC[a] for a in CHARS_CAT),
        "cont": tuple(XF[a] for a in CHARS_CONT),
    }
}
# `rm.*` reads a fill event adds to the policy facade (`POLICY_CALLS` is hashed).
FILL_POLICY_CALLS: Final[Mapping[str, frozenset[str]]] = {FILL_CHARS: frozenset({"rm.character"})}


def char_id(character: Any) -> int:
    """X_CHAR of a run's character (`rm.character`); `SchemaError` on an unknown one."""
    try:
        return CHAR_ID[str(character)]
    except KeyError:
        raise SchemaError(f"character {character!r} not in {CHARACTERS}") from None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Action kinds (uint8 on the wire, X_KIND in rows). Never renumber.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Kind:
    id: int
    name: str
    status: str  # "live" | "reserved:<gap>" | "spare"
    doc: str


_ACTION_KIND_TABLE: Final[tuple[Kind, ...]] = (
    Kind(0, "NONE", "live", "not an option token"),
    # combat (cs.legal_actions())
    Kind(1, "PLAY", "live", "play {card_index}, no target"),
    Kind(2, "PLAY_TARGET", "live", "play {card_index, target_index}"),
    Kind(3, "END_TURN", "live", "end_turn"),
    Kind(4, "POTION", "live", "drink_potion / potion {slot}, no target"),
    Kind(5, "POTION_TARGET", "live", "drink_potion / potion {slot, target_index}"),
    Kind(6, "CHOOSE", "live", "choose {index, selection: [i]}"),
    Kind(7, "CHOOSE_DONE", "live", "choose {selection: []}: finish or skip the choice"),
    Kind(8, "COMBAT_OTHER", "reserved:non-Ironclad", "channel / evoke / any other combat kind"),
    *(Kind(i, f"COMBAT_SPARE_{i}", "spare", "") for i in range(9, 16)),
    # out of combat (rm.available_actions() through PpoRunEnv)
    Kind(16, "MOVE", "live", "move {col}"),
    Kind(17, "PICK_CARD_REWARD", "live", "pick_card_reward {idx}"),
    Kind(18, "SKIP_CARD_REWARD", "live", "pick_card_reward {idx: None}"),
    Kind(19, "SACRIFICE_CARD_REWARD", "live", "sacrifice_card_reward (Pael's Wing)"),
    Kind(20, "PICK_CARD_BUNDLE", "live", "pick_card_reward {idx} with card_reward_bundles()"),
    Kind(21, "PICK_RELIC_REWARD", "live", "pick_relic_reward: the head of the relic queue"),
    Kind(22, "SKIP_RELIC_REWARD", "live", "skip_relic"),
    Kind(23, "PICK_BOSS_RELIC", "live", "pick_boss_relic {idx}"),
    Kind(24, "SHOP_BUY_CARD", "live", "shop_buy_card {idx}"),
    Kind(25, "SHOP_BUY_RELIC", "live", "shop_buy_relic {idx}"),
    Kind(26, "SHOP_BUY_POTION", "live", "shop_buy_potion {idx}"),
    Kind(27, "SHOP_REMOVE_CARD", "live", "shop_remove_card {deck_idx}: the removal service"),
    Kind(28, "LEAVE", "live", "leave_shop (also leaves a rest site after Miniature Tent)"),
    Kind(29, "REST_OPTION", "live", "rest_option {idx}; the option id rides in X_ID_MAIN"),
    Kind(30, "OOC_SPARE_30", "spare", ""),
    Kind(31, "SMITH_CARD", "live", "smith_card {idx}"),
    Kind(32, "EVENT_CHOICE", "live", "event_choice {idx} (Neow included)"),
    Kind(33, "COLLECT_TREASURE", "live", "collect_treasure"),
    Kind(34, "POTION_SWAP_DISCARD", "live", "potion_swap {discard_slot: i}"),
    Kind(35, "POTION_SWAP_DECLINE", "live", "potion_swap {discard_slot: None}"),
    Kind(36, "REMOVE_DECK_CARD", "live", "remove_deck_card {idx}"),
    Kind(37, "UPGRADE_DECK_CARD", "live", "upgrade_deck_card {idx}"),
    Kind(38, "TRANSFORM_DECK_CARD", "live", "transform_deck_card {idx}"),
    Kind(39, "ENCHANT_DECK_CARD", "live", "enchant_deck_card {idx}"),
    Kind(40, "GRANT_DECK_CARD", "live", "grant_deck_card {idx < len(card_grant_options())}"),
    Kind(41, "GRANT_CONFIRM", "live", "grant_deck_card {idx == len(options)}: end Sea Glass"),
    *(Kind(i, f"OOC_SPARE_{i}", "spare", "") for i in range(42, 48)),
    # accepted real-game gaps (§3.2): reserved, live-only, after the week (D9)
    Kind(48, "TAKE_REWARD_POTION", "reserved:#1069", "reward screen: take a potion"),
    Kind(49, "SKIP_REWARD_POTION", "reserved:#1069", "reward screen: leave a potion"),
    Kind(50, "TAKE_REWARD_GOLD", "reserved:#1069", "reward screen: take gold"),
    Kind(51, "SKIP_TREASURE", "reserved:treasure", "treasure room: leave the chest"),
    Kind(52, "SPHERE_TOOL", "reserved:crystal-sphere", "Crystal Sphere: pick a tool"),
    Kind(53, "SPHERE_CELL", "reserved:crystal-sphere", "Crystal Sphere: reveal a cell"),
    Kind(54, "USE_POTION_OOC", "reserved:ooc-potion", "drink a potion out of combat"),
    Kind(55, "DISCARD_POTION", "reserved:ooc-potion", "discard a potion out of combat"),
    Kind(56, "MERCHANT_BUY", "reserved:fake-merchant", "Fake Merchant: buy one item"),
    Kind(57, "MERCHANT_LEAVE", "reserved:fake-merchant", "Fake Merchant: leave"),
    *(Kind(i, f"GAP_SPARE_{i}", "spare", "reserved for further gaps") for i in range(58, 64)),
    *(Kind(i, f"SPARE_{i}", "spare", "D9 spare") for i in range(64, 96)),
)
ACTION_KINDS: Final = _ACTION_KIND_TABLE
N_ACTION_KIND: Final = 128  # X_KIND table size; ids 96-127 are never assigned
AK: Final[Mapping[str, int]] = {k.name: k.id for k in ACTION_KINDS}
LIVE_ACTION_KINDS: Final = frozenset(k.id for k in ACTION_KINDS if k.status == "live" and k.id)
COMBAT_ACTION_KINDS: Final = frozenset(range(1, 16))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Decision types (uint8 `dtype` in rows, X_DTYPE on RUNCTX). The type picks
#    the policy head; it is classified from the action set, never `phase()`.
# ═══════════════════════════════════════════════════════════════════════════

HEAD_COMBAT: Final = 0
HEAD_OOC: Final = 1


@dataclass(frozen=True)
class DType:
    id: int
    name: str
    status: str
    head: int
    kinds: frozenset[int]  # action kinds a decision of this type may offer


def _k(*names: str) -> frozenset[int]:
    return frozenset(AK[n] for n in names)


_DTYPE_TABLE: Final[tuple[DType, ...]] = (
    DType(0, "NONE", "live", -1, frozenset()),
    DType(
        1,
        "COMBAT_TURN",
        "live",
        HEAD_COMBAT,
        _k("PLAY", "PLAY_TARGET", "END_TURN", "POTION", "POTION_TARGET"),
    ),
    DType(
        2,
        "COMBAT_CHOICE",
        "live",
        HEAD_COMBAT,
        _k("CHOOSE", "CHOOSE_DONE"),
    ),
    *(DType(i, f"COMBAT_SPARE_{i}", "spare", HEAD_COMBAT, frozenset()) for i in range(3, 8)),
    DType(8, "NEOW", "live", HEAD_OOC, _k("EVENT_CHOICE")),
    DType(9, "MAP", "live", HEAD_OOC, _k("MOVE")),
    DType(
        10,
        "CARD_REWARD",
        "live",
        HEAD_OOC,
        _k("PICK_CARD_REWARD", "SKIP_CARD_REWARD", "SACRIFICE_CARD_REWARD"),
    ),
    DType(11, "CARD_BUNDLE", "live", HEAD_OOC, _k("PICK_CARD_BUNDLE")),
    DType(12, "RELIC_REWARD", "live", HEAD_OOC, _k("PICK_RELIC_REWARD", "SKIP_RELIC_REWARD")),
    DType(13, "BOSS_RELIC", "live", HEAD_OOC, _k("PICK_BOSS_RELIC")),
    DType(
        14,
        "SHOP",
        "live",
        HEAD_OOC,
        _k("SHOP_BUY_CARD", "SHOP_BUY_RELIC", "SHOP_BUY_POTION", "SHOP_REMOVE_CARD", "LEAVE"),
    ),
    DType(15, "REST", "live", HEAD_OOC, _k("REST_OPTION", "LEAVE")),
    DType(16, "SMITH", "live", HEAD_OOC, _k("SMITH_CARD")),
    DType(17, "EVENT", "live", HEAD_OOC, _k("EVENT_CHOICE")),
    DType(18, "POTION_SWAP", "live", HEAD_OOC, _k("POTION_SWAP_DISCARD", "POTION_SWAP_DECLINE")),
    DType(19, "CARD_REMOVE", "live", HEAD_OOC, _k("REMOVE_DECK_CARD")),
    DType(20, "CARD_UPGRADE", "live", HEAD_OOC, _k("UPGRADE_DECK_CARD")),
    DType(21, "CARD_TRANSFORM", "live", HEAD_OOC, _k("TRANSFORM_DECK_CARD")),
    DType(22, "CARD_ENCHANT", "live", HEAD_OOC, _k("ENCHANT_DECK_CARD")),
    DType(23, "CARD_GRANT", "live", HEAD_OOC, _k("GRANT_DECK_CARD", "GRANT_CONFIRM")),
    DType(24, "TREASURE", "live", HEAD_OOC, _k("COLLECT_TREASURE")),
    *(DType(i, f"OOC_SPARE_{i}", "spare", HEAD_OOC, frozenset()) for i in range(25, 40)),
    DType(
        40,
        "REWARD_SCREEN",
        "reserved:#1069",
        HEAD_OOC,
        _k("TAKE_REWARD_POTION", "SKIP_REWARD_POTION", "TAKE_REWARD_GOLD"),
    ),
    DType(
        41,
        "TREASURE_CHOICE",
        "reserved:treasure",
        HEAD_OOC,
        _k("COLLECT_TREASURE", "SKIP_TREASURE"),
    ),
    DType(
        42, "CRYSTAL_SPHERE", "reserved:crystal-sphere", HEAD_OOC, _k("SPHERE_TOOL", "SPHERE_CELL")
    ),
    DType(
        43, "POTION_MENU", "reserved:ooc-potion", HEAD_OOC, _k("USE_POTION_OOC", "DISCARD_POTION")
    ),
    DType(
        44,
        "FAKE_MERCHANT",
        "reserved:fake-merchant",
        HEAD_OOC,
        _k("MERCHANT_BUY", "MERCHANT_LEAVE"),
    ),
    *(DType(i, f"GAP_SPARE_{i}", "spare", HEAD_OOC, frozenset()) for i in range(45, 48)),
    *(DType(i, f"SPARE_{i}", "spare", HEAD_OOC, frozenset()) for i in range(48, 64)),
)
DECISION_TYPES: Final = _DTYPE_TABLE
N_DTYPE: Final = 64
DT: Final[Mapping[str, int]] = {d.name: d.id for d in DECISION_TYPES}
LIVE_DTYPES: Final = frozenset(d.id for d in DECISION_TYPES if d.status == "live" and d.id)
COMBAT_DTYPES: Final = frozenset({DT["COMBAT_TURN"], DT["COMBAT_CHOICE"]})

# The section of every live kind's option tokens (SCHEMA.md section 2.7). In
# combat these are #1081's S_ctx action sections (`tokens._action_tokens`); out
# of combat a class's token sits in the section of its payload, and
# content-free options (skip, leave, decline, confirm, sacrifice, the chest)
# in OPT_FIXED. A reserved kind gets its section at its fill event.
KIND_SECTION: Final[Mapping[int, int]] = {
    AK["PLAY"]: SEC["ACT_PLAY"],
    AK["PLAY_TARGET"]: SEC["ACT_PLAY"],
    AK["END_TURN"]: SEC["ACT_END"],
    AK["POTION"]: SEC["ACT_POTION"],
    AK["POTION_TARGET"]: SEC["ACT_POTION"],
    AK["CHOOSE"]: SEC["ACT_CHOOSE"],
    AK["CHOOSE_DONE"]: SEC["ACT_SKIP"],
    **{
        AK[k]: SEC["OPT_CARD"]
        for k in (
            "PICK_CARD_REWARD",
            "PICK_CARD_BUNDLE",
            "SHOP_BUY_CARD",
            "SHOP_REMOVE_CARD",
            "SMITH_CARD",
            "REMOVE_DECK_CARD",
            "UPGRADE_DECK_CARD",
            "TRANSFORM_DECK_CARD",
            "ENCHANT_DECK_CARD",
            "GRANT_DECK_CARD",
        )
    },
    **{AK[k]: SEC["OPT_RELIC"] for k in ("PICK_RELIC_REWARD", "PICK_BOSS_RELIC", "SHOP_BUY_RELIC")},
    **{AK[k]: SEC["OPT_POTION"] for k in ("SHOP_BUY_POTION", "POTION_SWAP_DISCARD")},
    AK["MOVE"]: SEC["OPT_MOVE"],
    AK["EVENT_CHOICE"]: SEC["OPT_EVENT"],
    AK["REST_OPTION"]: SEC["OPT_REST"],
    **{
        AK[k]: SEC["OPT_FIXED"]
        for k in (
            "SKIP_CARD_REWARD",
            "SACRIFICE_CARD_REWARD",
            "SKIP_RELIC_REWARD",
            "LEAVE",
            "COLLECT_TREASURE",
            "POTION_SWAP_DECLINE",
            "GRANT_CONFIRM",
        )
    },
}
# Out-of-combat body sections between the head and the option tokens, in
# their fixed order (SCHEMA.md section 2.1).
OOC_BODY_SECTIONS: Final = (SEC["DECK"], SEC["RELIC"], SEC["POTION"], SEC["OFFER"], SEC["MAPNODE"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. Caps.
# ═══════════════════════════════════════════════════════════════════════════

TOKEN_BUCKETS: Final = (32, 48, 64, 96, 128, 192, 256)
MAX_TOKENS: Final = 256
MAX_OPTIONS: Final = 64  # merged classes per decision, asserted
MAX_MEMBERS: Final = 255  # raw legal actions per decision (uint8-safe), asserted
MAX_FIGHT_TURNS: Final = 100  # capped fight = loss
MAX_FIGHT_DECISIONS: Final = 1000
MAX_RUN_DECISIONS: Final = 6000
CAP_ALARM_RATE: Final = 0.001
MAX_MAPNODES: Final = 40
MAX_BOSS_TOKENS: Final = 2
N_SUMMARY_TOKENS: Final = 5
MAX_PRIV: Final = 16  # PRIV_DRAW <= 10 + PRIV_ENEMY <= 5 + PRIV_RUN 1
PRIV_DRAW_K: Final = 10
PRIV_ENEMY_K: Final = 5  # one per enemy slot
PRIV_RUN_K: Final = 1
MAX_POOL: Final = 1024  # pooled entries per row, trunk (pool_*) and PRIV (ppool_*) each

# Practice cells (§3.3); 255 = not a practice segment.
CELLS: Final = (
    "ELITE_ACT1",
    "ELITE_ACT2",
    "ELITE_ACT3",
    "BOSS_ACT1",
    "BOSS_ACT2",
    "BOSS_ACT3_DOUBLE",
)
CELL_NONE: Final = 255
ACTORS: Final = ("student", "combat_teacher", "run_teacher")

# Token flags (uint8 per token).
TF_OPTION: Final = 1  # an option token (a member of some class)
TF_REP: Final = 2  # the class representative: the student's option token
TF_HIDDEN: Final = 4  # hidden from the student (non-representative member)
TF_TEACHER: Final = 8  # inside the combat teacher's view (the S_ctx block)
TF_POOLED: Final = 16  # a pooled summary token (SUM_*): owns the pool entries
TF_KNOWN: Final = TF_OPTION | TF_REP | TF_HIDDEN | TF_TEACHER | TF_POOLED

# Reserved columns and ids start carrying data only at these events. The
# schema hash does not move; the `filled` stamp records which happened.
FILL_EVENTS: Final = ("WP-ENG-1", "content-prior", "D9", "#1069", FILL_CHARS)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Classification: every decision maps to one decision type and every
#    option to one action kind, or this raises.
# ═══════════════════════════════════════════════════════════════════════════

_OOC_SIMPLE: Final[Mapping[str, str]] = {
    "move": "MOVE",
    "sacrifice_card_reward": "SACRIFICE_CARD_REWARD",
    "pick_relic_reward": "PICK_RELIC_REWARD",
    "skip_relic": "SKIP_RELIC_REWARD",
    "pick_boss_relic": "PICK_BOSS_RELIC",
    "shop_buy_card": "SHOP_BUY_CARD",
    "shop_buy_relic": "SHOP_BUY_RELIC",
    "shop_buy_potion": "SHOP_BUY_POTION",
    "shop_remove_card": "SHOP_REMOVE_CARD",
    "leave_shop": "LEAVE",
    "rest_option": "REST_OPTION",
    "smith_card": "SMITH_CARD",
    "event_choice": "EVENT_CHOICE",
    "collect_treasure": "COLLECT_TREASURE",
    "remove_deck_card": "REMOVE_DECK_CARD",
    "upgrade_deck_card": "UPGRADE_DECK_CARD",
    "transform_deck_card": "TRANSFORM_DECK_CARD",
    "enchant_deck_card": "ENCHANT_DECK_CARD",
}

# Leading kind -> decision type, checked in this order.
_DTYPE_BY_KIND: Final = (
    ("REST_OPTION", "REST"),
    ("SHOP_BUY_CARD", "SHOP"),
    ("SHOP_BUY_RELIC", "SHOP"),
    ("SHOP_BUY_POTION", "SHOP"),
    ("SHOP_REMOVE_CARD", "SHOP"),
    ("LEAVE", "SHOP"),
    ("MOVE", "MAP"),
    ("PICK_CARD_BUNDLE", "CARD_BUNDLE"),
    ("PICK_CARD_REWARD", "CARD_REWARD"),
    ("SKIP_CARD_REWARD", "CARD_REWARD"),
    ("SACRIFICE_CARD_REWARD", "CARD_REWARD"),
    ("PICK_RELIC_REWARD", "RELIC_REWARD"),
    ("SKIP_RELIC_REWARD", "RELIC_REWARD"),
    ("PICK_BOSS_RELIC", "BOSS_RELIC"),
    ("SMITH_CARD", "SMITH"),
    ("POTION_SWAP_DISCARD", "POTION_SWAP"),
    ("POTION_SWAP_DECLINE", "POTION_SWAP"),
    ("REMOVE_DECK_CARD", "CARD_REMOVE"),
    ("UPGRADE_DECK_CARD", "CARD_UPGRADE"),
    ("TRANSFORM_DECK_CARD", "CARD_TRANSFORM"),
    ("ENCHANT_DECK_CARD", "CARD_ENCHANT"),
    ("GRANT_DECK_CARD", "CARD_GRANT"),
    ("GRANT_CONFIRM", "CARD_GRANT"),
    ("COLLECT_TREASURE", "TREASURE"),
    ("EVENT_CHOICE", "EVENT"),
)


def ooc_action_kind(
    action: Mapping[str, Any], *, bundles: bool = False, grant_n_options: int | None = None
) -> int:
    """The action kind of one out-of-combat env action.

    Args:
        action: One entry of `Decision.actions` (after `mask_rejectable`).
        bundles: `rm.card_reward_bundles()` is non-empty (Scroll Boxes).
        grant_n_options: `len(rm.card_grant_options())`; needed only for
            `grant_deck_card`, whose `idx == n` is the Sea Glass confirm.

    Raises:
        SchemaError: The kind is unknown (`unhandled`, `unknown`, or new).
    """
    a = action.get("action")
    if a == "pick_card_reward":
        if action.get("idx") is None:
            if bundles:
                raise SchemaError("a card bundle offers no skip")
            return AK["SKIP_CARD_REWARD"]
        return AK["PICK_CARD_BUNDLE"] if bundles else AK["PICK_CARD_REWARD"]
    if a == "potion_swap":
        slot = action.get("discard_slot")
        return AK["POTION_SWAP_DECLINE"] if slot is None else AK["POTION_SWAP_DISCARD"]
    if a == "grant_deck_card":
        if grant_n_options is None:
            raise SchemaError("grant_deck_card needs grant_n_options")
        idx = int(action["idx"])
        if idx == grant_n_options:
            return AK["GRANT_CONFIRM"]
        if not 0 <= idx < grant_n_options:
            raise SchemaError(f"grant_deck_card idx {idx} outside 0..{grant_n_options}")
        return AK["GRANT_DECK_CARD"]
    name = _OOC_SIMPLE.get(str(a))
    if name is None:
        raise SchemaError(f"unknown out-of-combat action kind {a!r}")
    return AK[name]


def combat_action_kind(action: Mapping[str, Any]) -> int:
    """The action kind of one `cs.legal_actions()` entry."""
    a = action.get("action")
    if a == "play":
        return AK["PLAY"] if action.get("target_index") is None else AK["PLAY_TARGET"]
    if a == "end_turn":
        return AK["END_TURN"]
    if a in ("potion", "drink_potion"):
        return AK["POTION"] if action.get("target_index") is None else AK["POTION_TARGET"]
    if a == "choose":
        return AK["CHOOSE"] if action.get("selection") else AK["CHOOSE_DONE"]
    if a in ("channel", "evoke"):
        return AK["COMBAT_OTHER"]
    raise SchemaError(f"unknown combat action kind {a!r}")


def classify_ooc(
    actions: Sequence[Mapping[str, Any]],
    *,
    event_name: str | None = None,
    bundles: bool = False,
    grant_n_options: int | None = None,
) -> tuple[int, list[int]]:
    """`(dtype, kinds)` of an out-of-combat decision, from its action set.

    `event_name` is `rm.current_event_name()` (after `materialize_event()`);
    `"Neow"` makes an event decision a NEOW decision.
    """
    if not actions:
        raise SchemaError("a decision needs at least one action")
    kinds = [ooc_action_kind(a, bundles=bundles, grant_n_options=grant_n_options) for a in actions]
    names = {ACTION_KINDS[k].name for k in kinds}
    for kind, dtype in _DTYPE_BY_KIND:
        if kind in names:
            if dtype == "EVENT" and event_name == "Neow":
                dtype = "NEOW"
            d = DT[dtype]
            check_kinds(d, kinds)
            return d, kinds
    raise SchemaError(f"no decision type for action kinds {sorted(names)}")


def classify_combat(
    legal: Sequence[Mapping[str, Any]], pending_choice: Mapping[str, Any] | None
) -> tuple[int, list[int]]:
    """`(dtype, kinds)` of a combat decision (`cs.legal_actions()`,
    `cs.pending_choice()`)."""
    if not legal:
        raise SchemaError("a decision needs at least one action")
    kinds = [combat_action_kind(a) for a in legal]
    d = DT["COMBAT_CHOICE"] if pending_choice is not None else DT["COMBAT_TURN"]
    check_kinds(d, kinds)
    return d, kinds


def check_kinds(dtype: int, kinds: Sequence[int]) -> None:
    """Raise unless every kind is allowed for `dtype`."""
    allowed = DECISION_TYPES[dtype].kinds
    bad = sorted({ACTION_KINDS[k].name for k in kinds if k not in allowed})
    if bad:
        raise SchemaError(f"{DECISION_TYPES[dtype].name} cannot offer {bad}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Dedupe. Members of a class share a key; class order = first occurrence;
#    the representative is the first member. Frozen at G-schema.
# ═══════════════════════════════════════════════════════════════════════════


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def card_key(card: Mapping[str, Any]) -> str:
    """The visible identity of a card: #1081's `_canon(_card(card))`.

    Includes cost, enchantment (and amount), affliction, per-copy damage and
    block, calculated value and flags, so an enchanted Strike and a plain
    Strike, or a played and a fresh Rampage, never merge; plain Strikes do.
    """
    from ppo.combat_ctx.snapshot import _card

    return _canon(_card(dict(card)))


_PILES: Final = {
    "hand": "hand",
    "draw": "draw_pile",
    "discard": "discard_pile",
    "exhaust": "exhaust_pile",
}


def choice_option_key(
    choice: Mapping[str, Any], s: int, piles: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str:
    """The visible content of option `s` of a pending choice: its pile and the full
    card (`card_key`) when it sits in a pile the player sees, else the option's own
    fields (id, upgraded, type, pile).

    The pile is part of the key because the tokenizer encodes it (`choice_pile`): the
    same card in hand and in the draw pile are two options (#1097, a card_select
    offering a hand and a draw copy of TEAR_ASUNDER merged them)."""
    opt = (choice.get("cards") or choice.get("options") or [])[s]
    pile = piles.get(_PILES.get(str(opt.get("pile")), ""), None)
    idx = opt.get("index")
    if pile is not None and isinstance(idx, int) and 0 <= idx < len(pile):
        return _canon({"pile": opt.get("pile"), "card": card_key(pile[idx])})
    return _canon({k: opt.get(k) for k in ("id", "upgraded", "type", "pile")})


def combat_class_keys(
    legal: Sequence[Mapping[str, Any]],
    hand: Sequence[Mapping[str, Any]],
    potions: Sequence[Any],
    *,
    choice: Mapping[str, Any] | None = None,
    outcome: Callable[[int], str] | None = None,
    piles: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[str]:
    """One key per legal combat action; equal keys merge.

    Args:
        legal: The legal actions in the belief view's index space
            (`snapshot.obs_actions`).
        hand: `state_to_dict(bv)["deck"]["hand"]`.
        potions: The belief view's potion ids by slot.
        choice: `bv.pending_choice()`; required with `choose` actions.
        outcome: `i -> choose_outcome_key(bv.clone() after legal[i])`;
            required with `choose` actions.
        piles: `state_to_dict(bv)["deck"]` (hand, draw_pile, discard_pile,
            exhaust_pile) for the selected options' content.

    A play keys on `(card_key(card), target_index)`, a potion on
    `(potion id, target_index)`. A choose keys on its outcome AND the content
    of the options it selects: an incremental pick of a multi-select choice
    (`discard_from_hand_and_draw`, max 3-6) leaves the honest snapshot
    unchanged whichever card it picks, so the outcome alone would merge
    different cards.
    """
    keys = []
    for i, a in enumerate(legal):
        kind = combat_action_kind(a)
        if kind in (AK["PLAY"], AK["PLAY_TARGET"]):
            keys.append(
                _canon(["play", card_key(hand[int(a["card_index"])]), a.get("target_index")])
            )
        elif kind in (AK["POTION"], AK["POTION_TARGET"]):
            slot = int(a["slot"])
            pid = potions[slot] if slot < len(potions) else None
            keys.append(_canon(["potion", pid, a.get("target_index")]))
        elif kind in (AK["CHOOSE"], AK["CHOOSE_DONE"]):
            if outcome is None or choice is None:
                raise SchemaError("choose actions need choice and outcome")
            sel = [choice_option_key(choice, int(x), piles or {}) for x in a.get("selection") or []]
            keys.append(_canon(["choose", outcome(i), sorted(sel)]))
        else:
            keys.append(_canon([ACTION_KINDS[kind].name, i]))
    return keys


def ooc_class_keys(kinds: Sequence[int], payloads: Sequence[Any]) -> list[str]:
    """One key per out-of-combat option; equal keys merge.

    Only options whose payload is pure content merge: deck-card picks
    (remove, upgrade, transform, enchant, smith, the shop's removal service;
    payload = the visible deck card), offered cards (card reward, grant;
    payload = the card), shop items (payload = item with price) and
    potion-swap discards (payload = the potion id). Every other kind,
    bundles included, keys on its position and never merges.
    """
    mergeable = {
        AK[n]
        for n in (
            "PICK_CARD_REWARD",
            "SHOP_BUY_CARD",
            "SHOP_BUY_RELIC",
            "SHOP_BUY_POTION",
            "SHOP_REMOVE_CARD",
            "SMITH_CARD",
            "POTION_SWAP_DISCARD",
            "REMOVE_DECK_CARD",
            "UPGRADE_DECK_CARD",
            "TRANSFORM_DECK_CARD",
            "ENCHANT_DECK_CARD",
            "GRANT_DECK_CARD",
        )
    }
    if len(kinds) != len(payloads):
        raise SchemaError("kinds and payloads differ in length")
    return [
        _canon([k, p]) if k in mergeable else _canon([k, "pos", i])
        for i, (k, p) in enumerate(zip(kinds, payloads, strict=True))
    ]


def _at(xs: Any, i: Any) -> Any:
    try:
        return xs[int(i)]
    except (IndexError, KeyError, TypeError, ValueError):
        raise SchemaError(f"option index {i!r} outside its offer list") from None


def _deck_card(obs: Mapping[str, Any], deck_i: Any) -> list[Any]:
    ench = obs.get("deck_enchantments") or []
    e = ench[int(deck_i)] if int(deck_i) < len(ench) else None
    return [_at(obs["deck"], deck_i), e]


def ooc_payloads(
    obs: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], kinds: Sequence[int]
) -> list[Any]:
    """The visible payload of every out-of-combat option (`ooc_class_keys` input).

    `obs` is `PpoRunEnv.observe()` for the decision. A deck-card payload is
    the `master_deck_detailed()` entry plus its `master_deck_enchantments()`
    entry; an offered card is (id, upgraded, enchantment); a shop item is its
    inventory dict (id, price, sale flag, upgraded); a potion-swap discard is
    the potion in that slot.
    """
    out: list[Any] = []
    for a, k in zip(actions, kinds, strict=True):
        n = ACTION_KINDS[k].name
        if n in ("PICK_CARD_REWARD", "PICK_CARD_BUNDLE"):
            i = a["idx"]
            ench = obs.get("card_reward_option_enchantments") or []
            upg = obs.get("card_reward_option_upgraded") or []
            out.append(
                [
                    _at(obs.get("card_reward_option_ids") or [], i),
                    upg[int(i)] if int(i) < len(upg) else None,
                    ench[int(i)] if int(i) < len(ench) else None,
                ]
            )
        elif n == "SMITH_CARD":
            i = a["idx"]
            ench = obs.get("smith_option_enchantments") or []
            out.append([_at(obs["smith_options"], i), ench[int(i)] if int(i) < len(ench) else None])
        elif n == "REMOVE_DECK_CARD":
            i = a["idx"]
            ench = obs.get("card_removal_option_enchantments") or []
            out.append(
                [_at(obs["card_removal_options"], i), ench[int(i)] if int(i) < len(ench) else None]
            )
        elif n in ("UPGRADE_DECK_CARD", "TRANSFORM_DECK_CARD", "ENCHANT_DECK_CARD"):
            field = {
                "UPGRADE_DECK_CARD": "upgrade_select_deck_indices",
                "TRANSFORM_DECK_CARD": "transform_select_deck_indices",
                "ENCHANT_DECK_CARD": "enchant_select_deck_indices",
            }[n]
            out.append(_deck_card(obs, _at(obs[field], a["idx"])))
        elif n == "GRANT_DECK_CARD":
            i = a["idx"]
            ench = obs.get("card_grant_option_enchantments") or []
            out.append(
                [_at(obs["card_grant_options"], i), ench[int(i)] if int(i) < len(ench) else None]
            )
        elif n in ("SHOP_BUY_CARD", "SHOP_BUY_RELIC", "SHOP_BUY_POTION"):
            shelf = {"SHOP_BUY_CARD": "cards", "SHOP_BUY_RELIC": "relics"}.get(n, "potions")
            items = {int(x["idx"]): x for x in obs["shop"][shelf]}
            item = dict(_at(items, a["idx"]))
            item.pop("idx", None)
            out.append(item)
        elif n == "SHOP_REMOVE_CARD":
            out.append(_deck_card(obs, a["deck_idx"]))
        elif n == "POTION_SWAP_DISCARD":
            out.append(_at(obs["potions"], a["discard_slot"]))
        else:
            out.append(None)
    return out


def merge_classes(keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """`(member_class uint16 [M], rep_member uint16 [C])` from member keys."""
    first: dict[str, int] = {}
    cls = np.empty(len(keys), np.uint16)
    reps: list[int] = []
    for i, k in enumerate(keys):
        c = first.get(k)
        if c is None:
            c = first[k] = len(reps)
            reps.append(i)
        cls[i] = c
    return cls, np.asarray(reps, np.uint16)


def merge_probs(member_logits: np.ndarray, member_class: np.ndarray, n_classes: int) -> np.ndarray:
    """Teacher class probabilities from member logits (log-sum-exp merge)."""
    lg = np.asarray(member_logits, np.float64)
    out = np.full(n_classes, -np.inf)
    for c in range(n_classes):
        m = lg[member_class == c]
        if len(m):
            hi = m.max()
            out[c] = hi + np.log(np.exp(m - hi).sum())
    out -= out.max()
    p = np.exp(out)
    return (p / p.sum()).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Actor -> learner row batch: struct of arrays, ragged by per-row counts.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    axis: str  # row | tok | mem | cls | priv | pool | ppool
    width: int  # 0 = 1-D, else the second dimension
    doc: str


ROW_FIELDS: Final[tuple[Field, ...]] = (
    # required (task contract; names fixed)
    Field("run_uid", "int64", "row", 0, "unique per run or practice segment"),
    Field("t", "int32", "row", 0, "order of the stored row within its run or segment"),
    Field("is_combat", "bool", "row", 0, "combat decision"),
    Field("dtype", "uint8", "row", 0, "decision type (DECISION_TYPES)"),
    Field(
        "reward",
        "float32",
        "row",
        0,
        "all reward from this row to the next stored row of the run, or to the end;"
        " 0 on a bootstrap row",
    ),
    Field(
        "done",
        "bool",
        "row",
        0,
        "the run or segment ended: after this row, or at it when it is a bootstrap row",
    ),
    Field(
        "bootstrap",
        "bool",
        "row",
        0,
        "value-only row, its run's last: an iteration cut (stored again next iteration,"
        " same t) or, with done, a practice segment's end; no loss or statistic",
    ),
    Field("actor", "uint8", "row", 0, "0 student, 1 combat teacher, 2 run teacher"),
    Field("logp_behav", "float32", "row", 0, "log prob of the taken class under the actor"),
    Field("n_classes", "uint16", "row", 0, "merged options (>= 2)"),
    Field("action", "uint16", "row", 0, "index of the taken merged class"),
    Field("iter", "int32", "row", 0, "collection iteration"),
    Field("v_behav", "float32", "row", 0, "acting-time value (logging only)"),
    Field("cell", "uint8", "row", 0, "practice cell (CELLS), 255 = not practice"),
    # ragged layout
    Field("n_tok", "int32", "row", 0, "tokens in the row"),
    Field("n_mem", "uint16", "row", 0, "members (raw legal actions)"),
    Field("n_priv", "int32", "row", 0, "PRIV tokens"),
    Field("n_pool", "int32", "row", 0, "pooled entries"),
    Field("n_ppool", "int32", "row", 0, "PRIV pooled entries"),
    Field("sctx_lo", "uint16", "row", 0, "first S_ctx token (combat rows)"),
    Field("n_sctx", "uint16", "row", 0, "S_ctx tokens (0 out of combat)"),
    Field("t_src", "uint8", "row", 0, "teacher of `tprob`: 0 none, 1 combat, 2 run"),
    Field("cat", "int16", "tok", N_CAT, "categorical columns"),
    Field("cont", "float16", "tok", N_CONT, "continuous columns"),
    Field("sec", "int8", "tok", 0, "section id"),
    Field("tok_flags", "uint8", "tok", 0, "TF_* bits"),
    Field("mem_tok", "uint16", "mem", 0, "token of each member, env action order"),
    Field("mem_class", "uint16", "mem", 0, "class of each member"),
    Field("cls_tok", "uint16", "cls", 0, "representative token of each class"),
    Field("cls_kind", "uint8", "cls", 0, "action kind of each class"),
    Field("tprob", "float32", "cls", 0, "teacher class probabilities, NaN when none"),
    Field("pcat", "int16", "priv", N_CAT, "PRIV categorical columns"),
    Field("pcont", "float16", "priv", N_CONT, "PRIV continuous columns"),
    Field("psec", "int8", "priv", 0, "PRIV section id"),
    Field("pool_tok", "uint16", "pool", 0, "token owning the entry"),
    Field("pool_vid", "uint16", "pool", 0, "vocabulary id"),
    Field("pool_w", "float16", "pool", 0, "weight (count)"),
    Field("ppool_tok", "uint16", "ppool", 0, "PRIV token owning the entry"),
    Field("ppool_vid", "uint16", "ppool", 0, "vocabulary id"),
    Field("ppool_w", "float16", "ppool", 0, "weight (count)"),
)
REQUIRED_ROW_FIELDS: Final = (
    "run_uid",
    "t",
    "is_combat",
    "dtype",
    "reward",
    "done",
    "bootstrap",
    "actor",
    "logp_behav",
    "n_classes",
    "action",
    "iter",
    "v_behav",
    "cell",
)
FIELDS: Final[Mapping[str, Field]] = {f.name: f for f in ROW_FIELDS}
_AXIS_COUNT: Final = {
    "tok": "n_tok",
    "mem": "n_mem",
    "cls": "n_classes",
    "priv": "n_priv",
    "pool": "n_pool",
    "ppool": "n_ppool",
}


def row_numpy_dtype() -> np.dtype:
    """The fixed-width per-row fields as one numpy structured dtype."""
    return np.dtype([(f.name, f.dtype) for f in ROW_FIELDS if f.axis == "row"])


def empty_row_batch() -> dict[str, np.ndarray]:
    """A valid zero-row batch (every field present, right dtype and width)."""
    return {
        f.name: np.zeros((0, f.width) if f.width else (0,), np.dtype(f.dtype)) for f in ROW_FIELDS
    }


def _fail(msg: str) -> None:
    raise SchemaError(msg)


def concat_row_batches(batches: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """One batch holding the rows of `batches`, in order (row-local indices
    need no shifting, so every field just concatenates)."""
    if not batches:
        return empty_row_batch()
    return {f.name: np.concatenate([np.asarray(b[f.name]) for b in batches]) for f in ROW_FIELDS}


def take_rows(
    b: Mapping[str, np.ndarray], rows: Sequence[int] | np.ndarray
) -> dict[str, np.ndarray]:
    """The batch of rows `rows` of `b` (indices, in the order given), every
    ragged axis sliced to match: a minibatch or a stratum."""
    rows = np.asarray(rows, np.int64)
    off = row_offsets(b)
    out: dict[str, np.ndarray] = {}
    for f in ROW_FIELDS:
        x = b[f.name]
        if f.axis == "row":
            out[f.name] = x[rows]
            continue
        o = off[f.axis]
        parts = [np.arange(o[r], o[r + 1]) for r in rows]
        out[f.name] = x[np.concatenate(parts) if parts else np.zeros(0, np.int64)]
    return out


def validate_row_batch(
    b: Mapping[str, np.ndarray], *, allow_reserved: bool = False, runs: bool = False
) -> None:
    """Raise `SchemaError` unless `b` is a well-formed actor-to-learner batch.

    Per batch: fields, dtypes and widths; ragged lengths; caps; id ranges;
    finite rewards, values and log probs; the actor and teacher sources
    against the row kind; bootstrap rows (SCHEMA.md section 9.1): reward 0,
    and `done` only on a practice segment's out-of-combat end row.

    Per row (`_check_row`): the head (RUNCTX, the five SUM_* tokens in order,
    1-2 BOSS, then on another character's combat row its ALLY and ORB tokens,
    section 4b); RUNCTX's X_DTYPE equal to `dtype`, its encounter id NONE out of
    combat and its event id only on EVENT and NEOW rows; the member -> class
    map (onto, numbered by first occurrence, `action` in range); class tokens
    = the first member's token; TF_OPTION, TF_REP, TF_HIDDEN, TF_TEACHER and
    TF_POOLED on exactly their tokens; every member's X_KIND and section
    against its class kind (`KIND_SECTION`); a combat row's S_ctx block from
    `sctx_lo` = 6 + bosses (+ its ALLY and ORB tokens) to the end, starting with STATE and ending with its
    `n_mem` action tokens; an out-of-combat row's body sections in their
    fixed order and its option tokens last, in class order; pool entries on
    SUM_* tokens (PRIV: on PRIV_RUN), in range, unique, finite positive
    weights; PRIV sections and their per-section caps; and teacher
    probabilities (all NaN without a teacher, else a distribution).

    Per run (`runs=True`: one iteration's fresh table, every row each of its
    runs stored since the last cut): each run's `t` distinct and contiguous,
    one practice cell, and the run closed by its last row, which is its only
    done or bootstrap row (a terminal, a cut or a practice end). Leave
    `runs=False` for derived batches: minibatches, strata, and the learner's
    table with held rows.
    """
    for f in ROW_FIELDS:
        if f.name not in b:
            _fail(f"missing field {f.name!r}")
        x = b[f.name]
        if not isinstance(x, np.ndarray):
            _fail(f"{f.name}: not a numpy array")
        if x.dtype != np.dtype(f.dtype):
            _fail(f"{f.name}: dtype {x.dtype} != {f.dtype}")
        want = 2 if f.width else 1
        if x.ndim != want or (f.width and x.shape[1] != f.width):
            _fail(f"{f.name}: shape {x.shape}, want ndim {want} width {f.width or '-'}")
    n = len(b["run_uid"])
    for f in ROW_FIELDS:
        if f.axis == "row" and len(b[f.name]) != n:
            _fail(f"{f.name}: {len(b[f.name])} rows != {n}")
    for axis, cnt in _AXIS_COUNT.items():
        total = int(b[cnt].astype(np.int64).sum())
        for f in ROW_FIELDS:
            if f.axis == axis and len(b[f.name]) != total:
                _fail(f"{f.name}: length {len(b[f.name])} != sum({cnt}) = {total}")
    if n == 0:
        return
    nc = b["n_classes"].astype(np.int64)
    nt = b["n_tok"].astype(np.int64)
    nm = b["n_mem"].astype(np.int64)
    if (nc < 2).any() or (nc > MAX_OPTIONS).any():
        _fail(f"n_classes outside 2..{MAX_OPTIONS}")
    if (nm < nc).any() or (nm > MAX_MEMBERS).any():
        _fail("n_mem must be in n_classes..MAX_MEMBERS")
    if (nt < 1).any() or (nt > MAX_TOKENS).any():
        _fail(f"n_tok outside 1..{MAX_TOKENS}")
    if (b["n_priv"] < 0).any() or (b["n_priv"] > MAX_PRIV).any():
        _fail(f"n_priv outside 0..{MAX_PRIV}")
    for cnt in ("n_pool", "n_ppool"):
        if (b[cnt] < 0).any() or (b[cnt] > MAX_POOL).any():
            _fail(f"{cnt} outside 0..{MAX_POOL}")
    if (b["t"] < 0).any() or (b["t"] >= MAX_RUN_DECISIONS).any():
        _fail(f"t outside 0..{MAX_RUN_DECISIONS - 1} (the run cap)")
    if (b["iter"] < 0).any():
        _fail("iter must be >= 0")
    if (b["action"].astype(np.int64) >= nc).any():
        _fail("action >= n_classes")
    if (b["actor"] > 2).any():
        _fail("actor not in 0..2")
    if ((b["cell"] >= len(CELLS)) & (b["cell"] != CELL_NONE)).any():
        _fail("cell not in CELLS or 255")
    if (b["t_src"] > 2).any():
        _fail("t_src not in 0..2")
    comb = b["is_combat"]
    if ((b["actor"] == 1) & ~comb).any() or ((b["actor"] == 2) & comb).any():
        _fail("actor 1 (combat teacher) acts only on combat rows, actor 2 (run teacher) only OOC")
    for name in ("reward", "v_behav"):
        if not np.isfinite(b[name]).all():
            _fail(f"{name} must be finite")
    lp = b["logp_behav"]
    if not np.isfinite(lp).all() or (lp > 1e-6).any():
        _fail("logp_behav must be finite and <= 0")
    boot = b["bootstrap"]
    if (boot & (b["reward"] != 0)).any():
        _fail("a bootstrap row carries reward 0 (its reward belongs to its next storing)")
    end = boot & b["done"]
    if (end & ((b["cell"] == CELL_NONE) | comb)).any():
        _fail("done on a bootstrap row marks a practice segment's out-of-combat end only")
    ok_dt = LIVE_DTYPES | (
        frozenset(d.id for d in DECISION_TYPES if d.status != "live") if allow_reserved else set()
    )
    for d in np.unique(b["dtype"]):
        if int(d) not in ok_dt:
            _fail(f"dtype {int(d)} not live")
    combat_dt = np.isin(b["dtype"], list(COMBAT_DTYPES))
    if (combat_dt != comb).any():
        _fail("is_combat disagrees with dtype")
    ok_kind = LIVE_ACTION_KINDS | (
        frozenset(k.id for k in ACTION_KINDS if k.status != "live" and k.id)
        if allow_reserved
        else set()
    )
    sec, psec = b["sec"].astype(np.int64), b["psec"].astype(np.int64)
    if ((sec < 0) | (sec >= N_SECTION)).any() or ((psec < 0) | (psec >= N_SECTION)).any():
        _fail("section id out of range")
    if np.isin(sec, list(PRIV_SECTIONS)).any():
        _fail("PRIV section in the trunk tokens")
    if len(psec) and not np.isin(psec, list(PRIV_SECTIONS)).all():
        _fail("non-PRIV section in the PRIV arrays")
    if not allow_reserved and (
        np.isin(sec, list(RESERVED_SECTIONS)).any() or np.isin(psec, list(RESERVED_SECTIONS)).any()
    ):
        _fail("reserved section used")
    if (b["tok_flags"] & ~np.uint8(TF_KNOWN)).any():
        _fail("unknown token flag bits")
    _check_cat_ranges(b["cat"], "cat")
    _check_cat_ranges(b["pcat"], "pcat")
    for name in ("cont", "pcont"):
        if not np.isfinite(b[name]).all():
            _fail(f"{name}: non-finite value")
    for pre in ("pool", "ppool"):
        vid, w = b[f"{pre}_vid"].astype(np.int64), b[f"{pre}_w"]
        if ((vid < 1) | (vid >= VOCAB_ROWS)).any():
            _fail(f"{pre}_vid outside 1..{VOCAB_ROWS - 1}")
        if not np.isfinite(w).all() or (w <= 0).any():
            _fail(f"{pre}_w must be finite and > 0")
    off = row_offsets(b)
    for i in range(n):
        _check_row(b, i, off, ok_kind)
    if runs:
        _check_runs(b)


def _check_cat_ranges(cat: np.ndarray, name: str) -> None:
    sizes = np.asarray([SCTX_TABLE_SIZES[s] for s in SCTX_SLOTS] + [c.size for c in X_CAT])
    if len(cat) and ((cat < 0) | (cat >= sizes[None, :])).any():
        _fail(f"{name}: categorical id outside its table")


def _sec_mask(secs: Iterable[int]) -> np.ndarray:
    m = np.zeros(N_SECTION, bool)
    m[list(secs)] = True
    return m


# Lookup tables for `_check_row` (np.isin per row costs a third of the time).
_SUM_SECS: Final = np.asarray(SUMMARY_SECTIONS, np.int64)
_IS_SUM_SEC: Final = _sec_mask(SUMMARY_SECTIONS)
_IS_ACT_SEC: Final = _sec_mask(SCTX_ACT_SECTIONS)
_IS_OPT_SEC: Final = _sec_mask(OPTION_SECTIONS)
_KIND_SEC: Final = np.full(256, -1, np.int64)
for _k, _s in KIND_SECTION.items():
    _KIND_SEC[_k] = _s
_DTYPE_KIND: Final = np.zeros((N_DTYPE, 256), bool)
for _d in DECISION_TYPES:
    _DTYPE_KIND[_d.id, list(_d.kinds)] = True
_IS_EXT_SEC: Final = _sec_mask(COMBAT_EXT_SECTIONS)
_EXT_RANK: Final = np.full(N_SECTION, -1, np.int64)
for _r, _s in enumerate(COMBAT_EXT_SECTIONS):
    _EXT_RANK[_s] = _r
_BODY_RANK: Final = np.full(N_SECTION, -1, np.int64)
for _r, _s in enumerate(OOC_BODY_SECTIONS):
    _BODY_RANK[_s] = _r


def _check_pool(
    tok: np.ndarray, vid: np.ndarray, owner_ok: np.ndarray, n_owner: int, where: str
) -> None:
    """Row-local pool checks: owners inside the row and allowed, no duplicate id."""
    if (tok >= n_owner).any():
        _fail(f"{where}: pool entry owner outside the row")
    if not owner_ok[tok].all():
        _fail(f"{where}: pool entries belong to SUM_* tokens (PRIV: PRIV_RUN)")
    if len(np.unique(tok * VOCAB_ROWS + vid)) != len(tok):
        _fail(f"{where}: duplicate (token, id) pool entry")


def _check_row(
    b: Mapping[str, np.ndarray],
    i: int,
    off: Mapping[str, np.ndarray],
    ok_kind: frozenset[int],
) -> None:
    t0, t1 = int(off["tok"][i]), int(off["tok"][i + 1])
    m0, m1 = int(off["mem"][i]), int(off["mem"][i + 1])
    c0, c1 = int(off["cls"][i]), int(off["cls"][i + 1])
    nt, nm, nc = t1 - t0, m1 - m0, c1 - c0
    sec = b["sec"][t0:t1].astype(np.int64)
    flags = b["tok_flags"][t0:t1]
    cat = b["cat"][t0:t1]
    dtype = int(b["dtype"][i])
    combat = bool(b["is_combat"][i])
    where = f"row {i}"
    # Head: RUNCTX, the five SUM_* tokens in order, then 1-2 BOSS tokens.
    if sec[0] != SEC["RUNCTX"]:
        _fail(f"{where}: token 0 is not RUNCTX")
    if nt < 2 + N_SUMMARY_TOKENS or (sec[1 : 1 + N_SUMMARY_TOKENS] != _SUM_SECS).any():
        _fail(
            f"{where}: tokens 1-5 must be SUM_DRAW, SUM_DISCARD, SUM_EXHAUST, SUM_DECK, SUM_RELICS"
        )
    h = 1 + N_SUMMARY_TOKENS
    nb = 0
    while h + nb < nt and sec[h + nb] == SEC["BOSS"]:
        nb += 1
    if not 1 <= nb <= MAX_BOSS_TOKENS:
        _fail(f"{where}: {nb} BOSS tokens after the summaries (want 1-{MAX_BOSS_TOKENS})")
    body0 = h + nb
    # The "chars" fill event (section 4b): a combat row's ALLY and ORB tokens
    # sit between the head and the S_ctx block; only a row with X_CHAR set
    # (another character's run) carries any of the event's inputs.
    n_ext = 0
    while combat and body0 + n_ext < nt and _IS_EXT_SEC[sec[body0 + n_ext]]:
        n_ext += 1
    ext = sec[body0 : body0 + n_ext]
    if (np.diff(_EXT_RANK[ext]) < 0).any():
        _fail(f"{where}: the ALLY and ORB tokens must come in that order")
    if int((ext == SEC["ALLY"]).sum()) > MAX_ALLY_TOKENS:
        _fail(f"{where}: more than {MAX_ALLY_TOKENS} ALLY token")
    if int((ext == SEC["ORB"]).sum()) > MAX_ORB_TOKENS:
        _fail(f"{where}: more than {MAX_ORB_TOKENS} ORB tokens")
    if _IS_EXT_SEC[sec[body0 + n_ext :]].any():
        _fail(f"{where}: an ALLY or ORB token outside a combat row's head")
    if cat[0, XC["X_CHAR"]] == 0 and (n_ext or (b["cont"][t0:t1][:, XF["E_DOOMED"]] != 0).any()):
        _fail(f'{where}: "chars" inputs on a row without X_CHAR')
    body0 += n_ext
    if cat[0, XC["X_DTYPE"]] != dtype:
        _fail(f"{where}: RUNCTX X_DTYPE {int(cat[0, XC['X_DTYPE']])} != dtype {dtype}")
    if not combat and cat[0, XC["X_ID_AUX"]] != 0:
        _fail(f"{where}: RUNCTX X_ID_AUX (the fight's encounter) must be NONE out of combat")
    if dtype not in (DT["EVENT"], DT["NEOW"]) and cat[0, XC["X_ID_MAIN"]] != 0:
        _fail(f"{where}: RUNCTX X_ID_MAIN (the event) is set only on EVENT and NEOW rows")
    # Members and classes.
    mcls = b["mem_class"][m0:m1].astype(np.int64)
    mtok = b["mem_tok"][m0:m1].astype(np.int64)
    ctok = b["cls_tok"][c0:c1].astype(np.int64)
    ckind = b["cls_kind"][c0:c1].astype(np.int64)
    if (mcls >= nc).any() or len(np.unique(mcls)) != nc:
        _fail(f"{where}: member -> class map not onto 0..{nc - 1}")
    if (mtok >= nt).any() or (ctok >= nt).any():
        _fail(f"{where}: member or class token outside the row")
    _, first = np.unique(mcls, return_index=True)
    if (np.diff(first) <= 0).any():
        _fail(f"{where}: classes are not numbered by first occurrence")
    if (ctok != mtok[first]).any():
        _fail(f"{where}: a class token is not its first member's token")
    is_opt = np.zeros(nt, bool)
    is_opt[mtok] = True
    is_rep = np.zeros(nt, bool)
    is_rep[ctok] = True
    if not np.array_equal((flags & TF_OPTION) != 0, is_opt):
        _fail(f"{where}: TF_OPTION must mark exactly the member tokens")
    if not np.array_equal((flags & TF_REP) != 0, is_rep):
        _fail(f"{where}: TF_REP must mark exactly the class tokens")
    if not np.array_equal((flags & TF_HIDDEN) != 0, is_opt & ~is_rep):
        _fail(f"{where}: TF_HIDDEN must mark exactly the non-representative members")
    for k in np.unique(ckind):
        if int(k) not in ok_kind:
            _fail(f"{where}: action kind {int(k)} not allowed")
    if not _DTYPE_KIND[dtype, ckind].all():
        _fail(f"{where}: class kind not allowed for {DECISION_TYPES[dtype].name}")
    mkind = ckind[mcls]
    if (cat[mtok, XC["X_KIND"]] != mkind).any():
        _fail(f"{where}: X_KIND of a member token != its class kind")
    want_sec = _KIND_SEC[mkind]
    known = want_sec >= 0
    if (sec[mtok][known] != want_sec[known]).any():
        _fail(f"{where}: an option token is not in its kind's section (KIND_SECTION)")
    if not _IS_OPT_SEC[sec[mtok][~known]].all():
        _fail(f"{where}: an option token is not in an option section")
    # Stream layout.
    lo, ns = int(b["sctx_lo"][i]), int(b["n_sctx"][i])
    in_sctx = sec < SCTX_N_SECTION
    if combat:
        if lo != body0 or lo + ns != nt or ns < 1 + nm:
            _fail(f"{where}: the S_ctx block must run from sctx_lo = {body0} to the row's end")
        if not in_sctx[lo:].all():
            _fail(f"{where}: a non-S_ctx section inside the S_ctx block")
        if sec[lo] != SEC["STATE"]:
            _fail(f"{where}: the S_ctx block does not start with STATE")
        acts = np.arange(nt - nm, nt)
        if not _IS_ACT_SEC[sec[acts]].all():
            _fail(f"{where}: the S_ctx block does not end with its {nm} action tokens")
        if _IS_ACT_SEC[sec[lo : nt - nm]].any():
            _fail(f"{where}: an S_ctx action token before the action block")
        if (mtok != acts).any():
            _fail(f"{where}: combat members are not the S_ctx action tokens in order")
        if not np.array_equal((flags & TF_TEACHER) != 0, np.arange(nt) >= lo):
            _fail(f"{where}: TF_TEACHER must mark exactly the S_ctx block")
    else:
        if ns != 0 or lo != 0:
            _fail(f"{where}: an out-of-combat row has an S_ctx block")
        if (flags & TF_TEACHER).any():
            _fail(f"{where}: out-of-combat tokens are never teacher tokens")
        if (ctok != np.arange(nt - nc, nt)).any():
            _fail(f"{where}: out-of-combat option tokens must be the last {nc}, in class order")
        if (mtok != ctok[mcls]).any():
            _fail(f"{where}: an out-of-combat member does not point at its class token")
        body = sec[body0 : nt - nc]
        rank = _BODY_RANK[body]
        if (rank < 0).any() or (np.diff(rank) < 0).any():
            _fail(
                f"{where}: out-of-combat body must be DECK, RELIC, POTION, OFFER, MAPNODE in order"
            )
        n_map = int((body == SEC["MAPNODE"]).sum())
        if n_map > MAX_MAPNODES or (n_map and dtype != DT["MAP"]):
            _fail(f"{where}: MAPNODE tokens only on MAP rows, at most {MAX_MAPNODES}")
    # Pools: TF_POOLED marks the five SUM_* tokens (an empty pile's has no entries).
    is_sum = _IS_SUM_SEC[sec]
    if not np.array_equal((flags & TF_POOLED) != 0, is_sum):
        _fail(f"{where}: TF_POOLED must mark exactly the SUM_* tokens")
    p0, p1 = int(off["pool"][i]), int(off["pool"][i + 1])
    _check_pool(
        b["pool_tok"][p0:p1].astype(np.int64),
        b["pool_vid"][p0:p1].astype(np.int64),
        is_sum,
        nt,
        where,
    )
    v0, v1 = int(off["priv"][i]), int(off["priv"][i + 1])
    ps = b["psec"][v0:v1].astype(np.int64)
    for s, k in ((SEC["PRIV_DRAW"], PRIV_DRAW_K), (SEC["PRIV_ENEMY"], PRIV_ENEMY_K)):
        if int((ps == s).sum()) > k:
            _fail(f"{where}: more than {k} {SECTIONS[s]} tokens")
    if int((ps == SEC["PRIV_RUN"]).sum()) > PRIV_RUN_K:
        _fail(f"{where}: more than {PRIV_RUN_K} PRIV_RUN token")
    q0, q1 = int(off["ppool"][i]), int(off["ppool"][i + 1])
    _check_pool(
        b["ppool_tok"][q0:q1].astype(np.int64),
        b["ppool_vid"][q0:q1].astype(np.int64),
        ps == SEC["PRIV_RUN"],
        v1 - v0,
        f"{where} (PRIV)",
    )
    # Teacher probabilities.
    tp = b["tprob"][c0:c1]
    src = int(b["t_src"][i])
    if src == 0:
        if not np.isnan(tp).all():
            _fail(f"{where}: tprob must be NaN without a teacher")
    elif not np.isfinite(tp).all() or (tp < 0).any() or abs(float(tp.sum()) - 1.0) > 1e-4:
        _fail(f"{where}: tprob must be a distribution over the classes")
    if src == 1 and not combat:
        _fail(f"{where}: combat-teacher probs on an out-of-combat row")
    if src == 2 and combat:
        _fail(f"{where}: run-teacher probs on a combat row")


def _check_runs(b: Mapping[str, np.ndarray]) -> None:
    """Per-run bookkeeping of one iteration's fresh table (SCHEMA.md section 9.1)."""
    order = np.lexsort((b["t"], b["run_uid"]))
    uid = b["run_uid"][order]
    t = b["t"][order].astype(np.int64)
    closer = (b["bootstrap"] | b["done"])[order]
    cell = b["cell"][order]
    starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(uid)], strict=True):
        where = f"run {int(uid[s])}"
        tt = t[s:e]
        if (np.diff(tt) != 1).any():
            _fail(f"{where}: its rows' t are not distinct and contiguous")
        if (cell[s:e] != cell[s]).any():
            _fail(f"{where}: its rows carry different practice cells")
        if not closer[e - 1]:
            _fail(f"{where}: its last row is neither done nor a bootstrap row")
        if closer[s : e - 1].any():
            _fail(f"{where}: a done or bootstrap row before its last row")


def check_compact_lossless(cat: np.ndarray, cont: np.ndarray, sec: np.ndarray) -> None:
    """Raise unless every token is absent-valued outside its section's live
    columns (so the S1b compact layout drops nothing)."""
    pad = np.asarray(list(SCTX_PAD) + [0] * len(X_CAT), np.int64)
    for s in np.unique(sec):
        live_cat, live_cont = SECTION_LIVE.get(int(s), ((), ()))
        rows = sec == s
        dead_cat = np.ones(N_CAT, bool)
        dead_cat[list(live_cat)] = False
        dead_cont = np.ones(N_CONT, bool)
        dead_cont[list(live_cont)] = False
        c = cat[rows][:, dead_cat].astype(np.int64)
        if (c != pad[dead_cat][None, :]).any():
            _fail(f"section {SECTIONS[int(s)]}: categorical value outside its live columns")
        if (cont[rows][:, dead_cont] != 0).any():
            _fail(f"section {SECTIONS[int(s)]}: continuous value outside its live columns")


def compact_token_bytes(section: int) -> int:
    """Bytes one token of `section` takes in the compact layout (int16 cat,
    float16 cont, int8 section, uint8 flags)."""
    live_cat, live_cont = SECTION_LIVE[section]
    return 2 * len(live_cat) + 2 * len(live_cont) + 2


_W_CAT: Final = np.asarray([len(SECTION_LIVE.get(s, ((), ()))[0]) for s in range(N_SECTION)])
_W_CONT: Final = np.asarray([len(SECTION_LIVE.get(s, ((), ()))[1]) for s in range(N_SECTION)])
_ABSENT_CAT: Final = np.asarray(list(SCTX_PAD) + [0] * len(X_CAT), np.int16)


def to_compact(cat: np.ndarray, cont: np.ndarray, sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The compact layout: each token's live columns only, in token order.

    Returns `(ccat int16 [sum of live cat widths], ccont float16 [...])`;
    `sec` stays as it is and fixes every token's widths. Raises unless the
    dense tokens are lossless under the layout.
    """
    check_compact_lossless(cat, cont, sec)
    s64 = sec.astype(np.int64)
    wc, wf = _W_CAT[s64], _W_CONT[s64]
    oc = np.concatenate([[0], np.cumsum(wc)])[:-1]
    of = np.concatenate([[0], np.cumsum(wf)])[:-1]
    ccat = np.empty(int(wc.sum()), np.int16)
    ccont = np.empty(int(wf.sum()), np.float16)
    for sid in np.unique(s64):
        rows = np.flatnonzero(s64 == sid)
        lc, lf = SECTION_LIVE.get(int(sid), ((), ()))
        if lc:
            ccat[oc[rows][:, None] + np.arange(len(lc))] = cat[rows][:, list(lc)]
        if lf:
            ccont[of[rows][:, None] + np.arange(len(lf))] = cont[rows][:, list(lf)]
    return ccat, ccont


def from_compact(
    ccat: np.ndarray, ccont: np.ndarray, sec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Dense `(cat, cont)` from the compact layout (`to_compact`'s inverse)."""
    s64 = sec.astype(np.int64)
    wc, wf = _W_CAT[s64], _W_CONT[s64]
    if int(wc.sum()) != len(ccat) or int(wf.sum()) != len(ccont):
        raise SchemaError("compact arrays do not match the sections' live widths")
    oc = np.concatenate([[0], np.cumsum(wc)])[:-1]
    of = np.concatenate([[0], np.cumsum(wf)])[:-1]
    cat = np.tile(_ABSENT_CAT, (len(sec), 1))
    cont = np.zeros((len(sec), N_CONT), np.float16)
    for sid in np.unique(s64):
        rows = np.flatnonzero(s64 == sid)
        lc, lf = SECTION_LIVE.get(int(sid), ((), ()))
        if lc:
            cat[np.ix_(rows, list(lc))] = ccat[oc[rows][:, None] + np.arange(len(lc))]
        if lf:
            cont[np.ix_(rows, list(lf))] = ccont[of[rows][:, None] + np.arange(len(lf))]
    return cat, cont


def row_offsets(b: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Start offsets (length B + 1) of every ragged axis, keyed by axis."""
    return {
        axis: np.concatenate([[0], np.cumsum(b[cnt].astype(np.int64))])
        for axis, cnt in _AXIS_COUNT.items()
    }


def teacher_slice(b: Mapping[str, np.ndarray], i: int) -> dict[str, Any]:
    """The combat teacher's view of combat row `i` (X-C): the S_ctx block's
    `cat[:, :15]`, `cont[:, :122]`, `sec`, and `n_act`, byte-equal to
    #1081's `Encoder.encode()` arrays for the same decision."""
    if not b["is_combat"][i]:
        raise SchemaError(f"row {i} is not a combat row")
    t0 = int(row_offsets(b)["tok"][i]) + int(b["sctx_lo"][i])
    t1 = t0 + int(b["n_sctx"][i])
    return {
        "cat": b["cat"][t0:t1, :SCTX_N_CAT],
        "cont": b["cont"][t0:t1, :SCTX_N_CONT],
        "sec": b["sec"][t0:t1],
        "n_act": int(b["n_mem"][i]),
    }


def student_mask(b: Mapping[str, np.ndarray]) -> np.ndarray:
    """bool [sum T]: the tokens the student attends to (not `TF_HIDDEN`)."""
    return (b["tok_flags"] & TF_HIDDEN) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Env -> tokenizer interface: what the policy may read, and what only the
#     value path may read. The WP1 AST guard checks call sites against these.
# ═══════════════════════════════════════════════════════════════════════════

# What the policy may read of an event option's preview. `rm.event_option_infos()`
# resolves each option on a clone of the run (sts2-content events/dispatch.rs
# `preview_option_effect`). Under the e2e env's `mode="agent"` a relic the
# option grants is obtained on that clone, so the randomly rolled relic's
# on-pickup effect lands in the gold, hp, max_hp, potion and card deltas, and
# a relic that grants relics changes the relic count: hidden RNG in a policy
# input (69 events x 30 seeds, depth <= 2: 12 of 510 visible option groups
# preview differently across seeds; under this rule, none; SCHEMA.md
# section 2.7).
EVENT_DELTA_FIELDS: Final = (
    "gold_delta",
    "hp_delta",
    "max_hp_delta",
    "relics_gained",
    "relics_lost",
    "potions_gained",
    "potions_lost",
    "cards_gained",
    "cards_lost",
)
# Zeroed on an option that gains a relic: the relic's pickup effect lands here.
EVENT_RELIC_MASKED: Final = (
    "gold_delta",
    "hp_delta",
    "max_hp_delta",
    "potions_gained",
    "cards_gained",
)
# The O_EV_* column of each delta and its scale (`event_delta_columns`).
EVENT_DELTA_COLUMNS: Final[tuple[tuple[str, str, float], ...]] = (
    ("O_EV_GOLD", "gold_delta", 100.0),
    ("O_EV_HP", "hp_delta", 20.0),
    ("O_EV_MAXHP", "max_hp_delta", 10.0),
    ("O_EV_RELICS_GAIN", "relics_gained", 1.0),
    ("O_EV_POTIONS_GAIN", "potions_gained", 3.0),
    ("O_EV_CARDS_GAIN", "cards_gained", 3.0),
    ("O_EV_CARDS_LOSE", "cards_lost", 3.0),
    ("O_EV_RELICS_LOSE", "relics_lost", 3.0),
    ("O_EV_POTIONS_LOSE", "potions_lost", 3.0),
)


def honest_event_deltas(info: Mapping[str, Any]) -> dict[str, int] | None:
    """The preview deltas of one `event_option_infos()` entry the policy may read.

    None when the option has no preview. An option that gains a relic keeps
    only `relics_gained` = 1 and its losses: its gold, hp, max_hp, potion and
    card gains are 0, because on the preview clone they include the rolled
    relic's pickup effect. Idempotent, so applying it twice is harmless.
    """
    if any(info.get(k) is None for k in EVENT_DELTA_FIELDS[:3]):
        return None
    d = {k: int(info.get(k) or 0) for k in EVENT_DELTA_FIELDS}
    if d["relics_gained"] > 0:
        for k in EVENT_RELIC_MASKED:
            d[k] = 0
        d["relics_gained"] = 1
    return d


def honest_event_option(info: Mapping[str, Any]) -> dict[str, Any]:
    """`info` with its preview replaced by `honest_event_deltas(info)` (the delta
    keys dropped when the option has no preview); every other field kept."""
    out = {k: v for k, v in info.items() if k not in EVENT_DELTA_FIELDS}
    d = honest_event_deltas(info)
    if d is not None:
        out.update(d)
    return out


def honest_event_options(infos: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """`rm.event_option_infos()` as the policy may read it (`POLICY_FILTERS`)."""
    return [honest_event_option(o) for o in infos or ()]


def event_delta_columns(info: Mapping[str, Any]) -> dict[str, float]:
    """The O_EV_* values (column name -> value) of one event option's token,
    from `honest_event_deltas(info)`; all 0 when the option has no preview."""
    d = honest_event_deltas(info) or {}
    return {col: float(d.get(key, 0)) / scale for col, key, scale in EVENT_DELTA_COLUMNS}


# `PpoRunEnv.observe()` fields the policy never reads: the run seed.
OBS_HIDDEN_FIELDS: Final = ("seed",)


def honest_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
    """`EnvDecision.obs` as the policy tokenizer may read it: without
    `OBS_HIDDEN_FIELDS`, and with `event_options` through `honest_event_options`.
    The run teacher keeps reading the raw `obs` (SCHEMA.md section 12)."""
    out = {k: v for k, v in obs.items() if k not in OBS_HIDDEN_FIELDS}
    if out.get("event_options") is not None:
        out["event_options"] = honest_event_options(out["event_options"])
    return out


# Engine calls the policy tokenizer may make. In combat it reads the belief
# view `bv = cs.belief_view(0)`; the true state `cs` only for the actions the
# player is offered (`legal_actions`, `pending_choice`, re-indexed to `bv` by
# `snapshot.obs_actions`). Out of combat it reads `honest_obs(d.obs)` and the
# run facade (`PolicySource.run()`), which serves exactly the `rm.*` names
# below and passes each `POLICY_FILTERS` read through its filter.
POLICY_CALLS: Final = frozenset(
    {
        "cs.belief_view",
        "cs.legal_actions",
        "cs.pending_choice",
        "bv.pending_choice",
        "bv.card_damage_preview",
        "bv.card_block_preview",
        "bv.clone",  # choose_outcome_key: a clone of bv after the choose action
        "bv.apply",
        "honest_snapshot(bv)",
        "state_to_dict(bv)",
        "rollout_features(bv)",  # equal to rollout_features(cs) on 4,871 live decisions
        # WP-ENG-1 getters (feat/e2e-getters): read on bv, which carries them
        "bv.telegraph",
        "bv.peek_intents",  # never a step's `move_id`: hidden move state (PRIV_ENEMY)
        "bv.encounter_id",  # this fight's encounter; combat rows only
        "honest_obs(obs)",
        "rm.master_deck_detailed",
        "rm.master_deck_enchantments",
        "rm.relics",
        "rm.potions",
        "rm.max_potion_slots",
        "rm.gold",
        "rm.current_hp",
        "rm.max_hp",
        "rm.ascension_level",
        "rm.full_map",  # the current act's map only
        "rm.map_coord",
        "rm.current_room",
        "rm.current_event_name",
        "rm.event_option_infos",  # through POLICY_FILTERS: honest_event_options
        "rm.materialize_event",
        "rm.rest_site_options",
        "rm.card_reward_bundles",
        "rm.card_grant_options",
        "rm.pending_card_grant_min",
        "rm.pending_enchantment",
        "rm.pending_potion_swap",
        "rm.shop_inventory",
        "rm.available_actions",
        "rm.act_index",
        "rm.act_floor",
        "rm.total_floor",
    }
)
# Policy reads whose raw value carries hidden information: a policy facade
# returns the filter's output instead of the raw value.
POLICY_FILTERS: Final[Mapping[str, Callable[[Any], Any]]] = {
    "rm.event_option_infos": honest_event_options,
}
# Calls only the value path may make (PRIV). Never reachable from a policy
# input; F7 permutes what they read and requires bit-identical logits.
PRIV_CALLS: Final = frozenset(
    {
        "state_to_dict(cs)",
        "rollout_features(cs)",
        "cs.get_state",
        "cs.sample_world",
        "cs.true_draw_top",  # WP-ENG-1: the true top-k draw (PRIV_DRAW)
        "cs.peek_intents:move_id",  # WP-ENG-1: the true next moves (PRIV_ENEMY)
        "rm.generated_rooms",  # this act's and the later acts' rooms (PRIV_RUN)
        "rm.act_map_for",  # a later act's map; not read at launch (SCHEMA.md section 12)
    }
)


@runtime_checkable
class PolicySource(Protocol):
    """What the env hands the policy tokenizer for one decision."""

    def belief(self) -> Any: ...  # cs.belief_view(0) in combat, else None
    def legal(self) -> list[dict[str, Any]]: ...
    def run(self) -> Any: ...  # read-only RunManager facade: POLICY_CALLS' rm.*, POLICY_FILTERS


@runtime_checkable
class PrivSource(Protocol):
    """What the env hands the value path; never passed to the policy tokenizer.

    `act_rooms()` is this act's `rm.generated_rooms()` record itself (a mapping
    with `normal`, `elite`, `events_shuffled`, `boss`, `second_boss`), `{}` past
    the last act; `future()` the later acts' records. PRIV_RUN counts the
    record's queues and reads the later records' bosses (section 2.10).
    """

    def true_state(self) -> Any: ...
    def act_rooms(self) -> Mapping[str, Any]: ...
    def future(self) -> Sequence[Mapping[str, Any]]: ...


# ═══════════════════════════════════════════════════════════════════════════
# 11b. The env decision: what the env (WP2) yields for one stored decision, and
#      what the tokenizer (WP3b) and the actor read from it. In-process only
#      (never persisted), so it is not part of `schema_sha256`.
# ═══════════════════════════════════════════════════════════════════════════

# Decision types with an open multi-pick selection (RUNCTX `R_SEL_*`).
SELECTION_DTYPES: Final = frozenset({DT["COMBAT_CHOICE"], DT["CARD_GRANT"]})
# Kinds that end an open selection now (`Selection.can_skip`).
_SELECTION_END_KINDS: Final = frozenset({AK["CHOOSE_DONE"], AK["GRANT_CONFIRM"]})
# The raw env action names each live or reserved out-of-combat kind may carry.
_OOC_RAW: Final[Mapping[int, frozenset[str]]] = {
    **{AK[k]: frozenset({a}) for a, k in _OOC_SIMPLE.items()},
    AK["PICK_CARD_REWARD"]: frozenset({"pick_card_reward"}),
    AK["SKIP_CARD_REWARD"]: frozenset({"pick_card_reward"}),
    AK["PICK_CARD_BUNDLE"]: frozenset({"pick_card_reward"}),
    AK["POTION_SWAP_DISCARD"]: frozenset({"potion_swap"}),
    AK["POTION_SWAP_DECLINE"]: frozenset({"potion_swap"}),
    AK["GRANT_DECK_CARD"]: frozenset({"grant_deck_card"}),
    AK["GRANT_CONFIRM"]: frozenset({"grant_deck_card"}),
}


@dataclass(frozen=True)
class Selection:
    """The open multi-pick selection: RUNCTX `R_SEL_MIN/MAX/PICKED/CAN_SKIP`.

    Set on `SELECTION_DTYPES` only; every other decision carries the default.

    * COMBAT_CHOICE: `min`/`max` are `bv.pending_choice()`'s `min_select` /
      `max_select`. `picked` is `len(pending_choice()["cards"])` minus the
      CHOOSE options offered: the engine decomposes a `max_select > 1` choice
      into one incremental pick per still-unpicked option and re-opens the
      same choice after each (sts2-core lib.rs:14385-14436); its `picked` list
      is not on the wire.
    * CARD_GRANT: `min` is `rm.pending_card_grant_min` (a property: the picks
      still required), `max` the offered cards (`len(card_grant_options())`),
      `picked` the grant picks the env applied since the grant opened.
    * `can_skip`: a CHOOSE_DONE or GRANT_CONFIRM option is offered, i.e. the
      selection can end now. (On a choice with picks still to make this is
      not `pending_choice()["can_skip"]`: the engine also offers FINALIZE once
      `min_select` picks are made.)
    """

    min: int = 0
    max: int = 0
    picked: int = 0
    can_skip: bool = False


NO_SELECTION: Final = Selection()


@dataclass
class EnvDecision:
    """One stored decision, as the env yields it (SCHEMA.md section 8).

    Members are the legal actions in env order: `PpoRunEnv`'s `Decision.actions`
    (after `mask_rejectable`) out of combat; in combat, `cs.legal_actions()`
    re-indexed to the belief view by `snapshot.obs_actions`. `kinds` holds one
    action kind per member (`classify_combat` / `classify_ooc`), `keys` one
    dedupe key per member (`combat_class_keys` / `ooc_class_keys`), and
    `member_class`, `rep` are `merge_classes(keys)`. The actor takes class `c`
    by applying `actions[rep[c]]` (`member_action`).

    `policy` is all the policy tokenizer may read; `priv` (None when the value
    path is off) is for the value path only and is never passed to the policy
    tokenizer. `obs` and `metrics` are `PpoRunEnv.observe()` and
    `metrics_snapshot()`: required out of combat (the run teacher's
    `featurize_row` input and `ooc_payloads`' input), None in combat. The
    policy tokenizer reads `obs` only as `honest_obs(obs)`.
    """

    run_uid: int
    t: int  # the row's `t`: order of this stored decision within its run or segment
    dtype: int
    actions: Sequence[Mapping[str, Any]]
    kinds: np.ndarray  # uint8 [M]
    keys: Sequence[str]  # [M]
    member_class: np.ndarray  # uint16 [M]
    rep: np.ndarray  # uint16 [C]
    policy: PolicySource
    priv: PrivSource | None = None
    sel: Selection = NO_SELECTION
    event_name: str | None = None  # rm.current_event_name() on EVENT and NEOW
    obs: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    cell: int = CELL_NONE
    fight_turn: int = 0  # combat: the fight's turn, 1-based; 0 out of combat
    fight_decision: int = 0  # combat: stored decisions before this one in the fight

    @property
    def is_combat(self) -> bool:
        return self.dtype in COMBAT_DTYPES

    @property
    def head(self) -> int:
        return DECISION_TYPES[self.dtype].head

    @property
    def n_members(self) -> int:
        return len(self.actions)

    @property
    def n_classes(self) -> int:
        return len(self.rep)

    def class_kinds(self) -> np.ndarray:
        """uint8 [C]: the action kind of each class (the row's `cls_kind`)."""
        return np.asarray(self.kinds, np.uint8)[np.asarray(self.rep, np.intp)]

    def member_action(self, c: int) -> Mapping[str, Any]:
        """The env action that takes class `c`: its representative member."""
        return self.actions[int(self.rep[c])]


def make_decision(
    *,
    run_uid: int,
    t: int,
    dtype: int,
    actions: Sequence[Mapping[str, Any]],
    kinds: Sequence[int],
    keys: Sequence[str],
    policy: PolicySource,
    **kw: Any,
) -> EnvDecision:
    """An `EnvDecision` with `member_class` and `rep` from `merge_classes(keys)`."""
    cls, rep = merge_classes(keys)
    return EnvDecision(
        run_uid=run_uid,
        t=t,
        dtype=dtype,
        actions=list(actions),
        kinds=np.asarray(kinds, np.uint8),
        keys=list(keys),
        member_class=cls,
        rep=rep,
        policy=policy,
        **kw,
    )


def validate_decision(d: EnvDecision, *, allow_reserved: bool = False) -> None:
    """Raise `SchemaError` unless `d` is a well-formed stored decision.

    Checks the decision type (live, or reserved with `allow_reserved`), one
    kind per member and every kind allowed for the type, each member's raw
    action name against its kind, the class map (`merge_classes(keys)`
    exactly: first-occurrence numbering, representative = first member, one
    kind per class), the member and class caps (2 <= classes <= MAX_OPTIONS),
    the selection spec, the event name, `obs`/`metrics` presence, the
    practice cell, the run and fight caps, and that the policy source exposes
    no PRIV accessor.
    """
    if not 0 < d.dtype < N_DTYPE:
        _fail(f"dtype {d.dtype} outside 1..{N_DTYPE - 1}")
    dt = DECISION_TYPES[d.dtype]
    if dt.status == "spare" or (dt.status != "live" and not allow_reserved):
        _fail(f"decision type {dt.name} is {dt.status}")
    m = len(d.actions)
    if not 1 <= m <= MAX_MEMBERS:
        _fail(f"{m} members outside 1..{MAX_MEMBERS}")
    kinds = np.asarray(d.kinds)
    if kinds.dtype != np.uint8 or kinds.shape != (m,):
        _fail(f"kinds must be uint8 [{m}], got {kinds.dtype} {kinds.shape}")
    if len(d.keys) != m:
        _fail(f"{len(d.keys)} keys for {m} members")
    check_kinds(d.dtype, kinds.tolist())
    for a, k in zip(d.actions, kinds.tolist(), strict=True):
        if d.is_combat:
            if combat_action_kind(a) != k:
                _fail(f"combat action {a!r} is not kind {ACTION_KINDS[k].name}")
        elif k in _OOC_RAW:
            if a.get("action") not in _OOC_RAW[k]:
                _fail(f"action {a.get('action')!r} cannot be kind {ACTION_KINDS[k].name}")
        elif ACTION_KINDS[k].status == "live":
            _fail(f"no raw action names for live kind {ACTION_KINDS[k].name}")
    cls, rep = merge_classes(d.keys)
    got_cls, got_rep = np.asarray(d.member_class), np.asarray(d.rep)
    if got_cls.dtype != np.uint16 or got_rep.dtype != np.uint16:
        _fail("member_class and rep must be uint16")
    if not (np.array_equal(got_cls, cls) and np.array_equal(got_rep, rep)):
        _fail("member_class / rep differ from merge_classes(keys)")
    if not 2 <= len(rep) <= MAX_OPTIONS:
        _fail(f"{len(rep)} merged classes outside 2..{MAX_OPTIONS}")
    if (kinds != kinds[rep.astype(np.intp)][cls.astype(np.intp)]).any():
        _fail("a merged class mixes action kinds")
    _check_selection(d, kinds)
    if (d.dtype in (DT["EVENT"], DT["NEOW"])) != bool(d.event_name):
        _fail(f"event_name {d.event_name!r} on a {dt.name} decision")
    if (d.dtype == DT["NEOW"]) != (d.event_name == "Neow"):
        _fail(f"event {d.event_name!r} on a {dt.name} decision")
    if d.is_combat != (d.obs is None) or d.is_combat != (d.metrics is None):
        _fail("obs and metrics are required out of combat and None in combat")
    if d.cell != CELL_NONE and not 0 <= d.cell < len(CELLS):
        _fail(f"practice cell {d.cell}")
    if not 0 <= d.t < MAX_RUN_DECISIONS:
        _fail(f"t {d.t} outside the run cap {MAX_RUN_DECISIONS}")
    if d.is_combat:
        if not 1 <= d.fight_turn <= MAX_FIGHT_TURNS:
            _fail(f"fight_turn {d.fight_turn} outside 1..{MAX_FIGHT_TURNS}")
        if not 0 <= d.fight_decision < MAX_FIGHT_DECISIONS:
            _fail(f"fight_decision {d.fight_decision} outside the fight cap")
    elif d.fight_turn or d.fight_decision:
        _fail("fight counters must be 0 out of combat")
    if not isinstance(d.policy, PolicySource):
        _fail("policy is not a PolicySource")
    if isinstance(d.policy, PrivSource):
        _fail("the policy source exposes PRIV accessors")
    if d.priv is not None and not isinstance(d.priv, PrivSource):
        _fail("priv is not a PrivSource")


def _check_selection(d: EnvDecision, kinds: np.ndarray) -> None:
    s = d.sel
    if d.dtype not in SELECTION_DTYPES:
        if s != NO_SELECTION:
            _fail(f"a selection on a {DECISION_TYPES[d.dtype].name} decision")
        return
    if not 0 <= s.min <= s.max or s.picked < 0:
        _fail(f"selection {s}")
    ends = bool(_SELECTION_END_KINDS & set(kinds.tolist()))
    if s.can_skip != ends:
        _fail(f"selection can_skip={s.can_skip}, end option offered={ends}")
    if d.dtype == DT["CARD_GRANT"]:
        n_cards = int((kinds == AK["GRANT_DECK_CARD"]).sum())
        if s.max != n_cards:
            _fail(f"grant selection max {s.max} != {n_cards} offered cards")


@runtime_checkable
class Tokenizer(Protocol):
    """WP3b's tokenizer, as the env and the actor drive it.

    `reset_fight()` at every fight start (a fresh `IntentLog`); `observe()` at
    EVERY combat step, single-option and auto-advanced ones included, before
    the action (#1081's `Encoder.observe`), so the intent history is complete;
    `encode()` for each stored decision; `encode_priv()` on the value path
    only. Both return their row's arrays under the `ROW_FIELDS` names (`cat`,
    `cont`, `sec`, `tok_flags`, `mem_tok`, `mem_class`, `cls_tok`, `cls_kind`,
    `pool_*`, `sctx_lo`, `n_sctx`; and `pcat`, `pcont`, `psec`, `ppool_*`).
    `encode()` reads only `d.policy` and `honest_obs(d.obs)`, and fills the
    O_EV_* columns with `event_delta_columns`.
    """

    def reset_fight(self) -> None: ...
    def observe(self, policy: PolicySource) -> None: ...
    def encode(self, d: EnvDecision) -> Mapping[str, Any]: ...
    def encode_priv(self, d: EnvDecision) -> Mapping[str, Any]: ...


# ═══════════════════════════════════════════════════════════════════════════
# 12. Stamps.
# ═══════════════════════════════════════════════════════════════════════════

STAMP_KEYS: Final = (
    "schema_sha256",
    "vocab_sha256",
    "prior_sha256",
    "filled",
    "teacher",
    "game_version",
    "wheel",
    "source_digest",
)


def _json_sha256(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def teacher_stamp(
    *,
    ckpt_sha256: str,
    rare: Mapping[str, Sequence[int]] | None,
    sctx_vocab: Mapping[str, int],
) -> dict[str, str]:
    """The combat teacher's stamp: its checkpoint file, its RareMap and its S_ctx
    table sizes. Rows store the S_ctx content slots through this RareMap
    (section 3.1), so rows and a teacher match only if the stamps do.

    Args:
        ckpt_sha256: sha256 of the teacher checkpoint file.
        rare: The checkpoint's RareMap as `RareMap.to_json()` gives it
            (`ck["rare"]`: per content slot, the row each table id maps to),
            or None for none.
        sctx_vocab: The checkpoint's S_ctx table sizes (`ck["vocab"]`).
    """
    return {
        "ckpt_sha256": str(ckpt_sha256),
        "rare_sha256": _json_sha256(
            None if rare is None else {str(k): [int(x) for x in v] for k, v in rare.items()}
        ),
        "sctx_vocab_sha256": _json_sha256({k: int(v) for k, v in sctx_vocab.items()}),
    }


def schema_spec() -> dict[str, Any]:
    """Everything the schema hash covers, as plain JSON-able data."""
    return {
        "name": SCHEMA_NAME,
        "revision": SCHEMA_REVISION,
        "game_version": GAME_VERSION,
        "sctx": {
            "sections": list(SCTX_SECTIONS),
            "slots": list(SCTX_SLOTS),
            "content_slots": list(SCTX_CONTENT_SLOTS),
            "student_slots": list(SCTX_STUDENT_SLOTS),
            "table_sizes": dict(SCTX_TABLE_SIZES),
            "intent_kinds": list(SCTX_INTENT_KINDS),
            "cont": [list(c) for c in SCTX_CONT],
        },
        "sections": list(SECTIONS),
        "priv_sections": sorted(PRIV_SECTIONS),
        "x_cat": [[c.name, c.size, c.transform, c.status] for c in X_CAT],
        "x_cont": [[c.name, c.transform, c.status] for c in X_CONT],
        "section_live": {
            SECTIONS[s]: [list(c), list(f)] for s, (c, f) in _SECTION_LIVE_BASE.items()
        },
        "prior_columns": list(PRIOR_COLUMNS),
        "room_types": list(ROOM_TYPES),
        "action_kinds": [[k.id, k.name, k.status] for k in ACTION_KINDS],
        "decision_types": [
            [d.id, d.name, d.status, d.head, sorted(d.kinds)] for d in DECISION_TYPES
        ],
        "caps": {
            "buckets": list(TOKEN_BUCKETS),
            "max_tokens": MAX_TOKENS,
            "max_options": MAX_OPTIONS,
            "max_members": MAX_MEMBERS,
            "fight_turns": MAX_FIGHT_TURNS,
            "fight_decisions": MAX_FIGHT_DECISIONS,
            "run_decisions": MAX_RUN_DECISIONS,
            "mapnodes": MAX_MAPNODES,
            "boss_tokens": MAX_BOSS_TOKENS,
            "summary_tokens": N_SUMMARY_TOKENS,
            "priv": MAX_PRIV,
            "priv_draw_k": PRIV_DRAW_K,
            "priv_enemy_k": PRIV_ENEMY_K,
            "priv_run_k": PRIV_RUN_K,
            "pool": MAX_POOL,
            "vocab_rows": VOCAB_ROWS,
            "n_section": N_SECTION,
            "n_action_kind": N_ACTION_KIND,
            "n_dtype": N_DTYPE,
            "n_prior": N_PRIOR,
        },
        "cells": list(CELLS),
        "actors": list(ACTORS),
        "token_flags": {
            "OPTION": TF_OPTION,
            "REP": TF_REP,
            "HIDDEN": TF_HIDDEN,
            "TEACHER": TF_TEACHER,
            "POOLED": TF_POOLED,
        },
        "row_fields": [[f.name, f.dtype, f.axis, f.width] for f in ROW_FIELDS],
        "kind_section": {
            ACTION_KINDS[k].name: SECTIONS[s] for k, s in sorted(KIND_SECTION.items())
        },
        "ooc_body_sections": [SECTIONS[s] for s in OOC_BODY_SECTIONS],
        "event_preview": {
            "fields": list(EVENT_DELTA_FIELDS),
            "relic_masked": list(EVENT_RELIC_MASKED),
            "columns": [list(c) for c in EVENT_DELTA_COLUMNS],
        },
        "obs_hidden_fields": list(OBS_HIDDEN_FIELDS),
        "policy_calls": sorted(POLICY_CALLS),
        "policy_filters": {k: f.__name__ for k, f in sorted(POLICY_FILTERS.items())},
        "priv_calls": sorted(PRIV_CALLS),
        "stamp_keys": list(STAMP_KEYS),
    }


def schema_sha256() -> str:
    """sha256 of the canonical JSON of `schema_spec()`."""
    return _json_sha256(schema_spec())


def stamps(
    *,
    vocab_sha256: str,
    prior_sha256: str | None = None,
    filled: Sequence[str] = (),
    teacher: Mapping[str, str] | None = None,
    wheel: str | None = None,
    source_digest: str | None = None,
) -> dict[str, Any]:
    """The stamp dict every cfg, checkpoint, stats file, eval row and row
    batch carries. `teacher` is `teacher_stamp(...)` wherever the combat
    teacher or its RareMap'd rows are involved, else None. `prior_sha256` is
    the filled prior's (`content_prior.sha256()`) exactly when `filled` names
    "content-prior": a fill without its prior, or a prior without its fill
    event, raises `SchemaError`."""
    bad = sorted(set(filled) - set(FILL_EVENTS))
    if bad:
        raise SchemaError(f"unknown fill events {bad}")
    if ("content-prior" in filled) != (prior_sha256 is not None):
        raise SchemaError(
            f"filled {sorted(filled)} with prior_sha256 {prior_sha256!r}: the "
            '"content-prior" event and the prior\'s sha256 come together'
        )
    return {
        "schema_sha256": schema_sha256(),
        "vocab_sha256": vocab_sha256,
        "prior_sha256": prior_sha256,
        "filled": sorted(filled),
        "teacher": None if teacher is None else dict(teacher),
        "game_version": GAME_VERSION,
        "wheel": wheel,
        "source_digest": source_digest,
    }


def check_stamps(
    have: Mapping[str, Any], want: Mapping[str, Any], *, restart: bool = False
) -> None:
    """Raise unless an artifact's stamps (`have`) fit this process's (`want`).

    Always: equal schema, vocabulary and game version; equal teacher stamps
    where both name a teacher. Otherwise the fill state must match: the same
    `filled` events and `prior_sha256`. With `restart=True` (a restart that
    loads `have`, section 2.12's identity rule) `want` may add fill events and
    the prior: `have`'s events must be a subset of `want`'s, and `have`'s
    `prior_sha256` must be `want`'s or None. A checkpoint that read filled
    columns is never served by a process that leaves them at zero.
    """
    for k in ("schema_sha256", "vocab_sha256", "game_version"):
        if have.get(k) != want.get(k):
            raise SchemaError(f"stamp {k}: {have.get(k)!r} != {want.get(k)!r}")
    th, tw = have.get("teacher"), want.get("teacher")
    if th is not None and tw is not None and dict(th) != dict(tw):
        raise SchemaError(f"stamp teacher: {th!r} != {tw!r}")
    fh, fw = set(have.get("filled") or ()), set(want.get("filled") or ())
    ph, pw = have.get("prior_sha256"), want.get("prior_sha256")
    if restart:
        if not fh <= fw:
            raise SchemaError(f"stamp filled: {sorted(fh - fw)} filled before, not now")
        if ph is not None and ph != pw:
            raise SchemaError(f"stamp prior_sha256: {ph!r} != {pw!r}")
    else:
        if fh != fw:
            raise SchemaError(f"stamp filled: {sorted(fh)} != {sorted(fw)}")
        if ph != pw:
            raise SchemaError(f"stamp prior_sha256: {ph!r} != {pw!r}")
