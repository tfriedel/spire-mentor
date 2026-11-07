"""Token schemas for the combat context net (GH #830 step 0, DESIGN §b).

`tokens(obs, actions, rl2feat, arm)` turns one decision into an unordered set
of tokens. It reads ONLY its four arguments: `obs` is the honest snapshot
(`snapshot.honest_snapshot` plus `IntentLog` history), `actions` the legal
actions in the snapshot's option order (`snapshot.obs_actions`), and `rl2feat`
rl2's v1 inputs `{state, x, keys}` (`sts2_rs.rollout_features(cs, 1)`). A
record's search statistics, outcome and hashes cannot reach it (test 10).

Every token is the sum of a section embedding, the embeddings of the
categorical slots it fills (PAD contributes 0) and a sinusoidal code of its
continuous columns. There are no positional encodings, so neither pile order
nor option order can leak; explicit slot ids link tokens instead (the hand
slot ties a play action to its card, the enemy slot ties a target / enemy
power / intent-history entry to its enemy).

Arms (PREREG §3):

* `S_ctx`: every section and slot, plus rl2's 32 state features on the STATE
  token and its 24 action features + action key on every action token, so its
  input is a superset of rl2's (INVALID #8 checks it).
* `S_ctx_tok`: `S_ctx` without the rl2 key and rl2 columns (diagnostic).
* `S_ctxarch_rl2in`: one STATE token with rl2's 32 state features and one
  `ACT_RL2` token per action with the rl2 key and 24 action features only:
  the context control, on the same transformer.

Content tables are full-universe vocabularies from the engine registries
(`[ids..., UNK, PAD]`), so a test-only id never lands on a row that was never
trained (the G-D `<unk>` failure). Rare ids (< 3 TRAIN runs) are mapped to
UNK later, by `data.RareMap`, not here; a name the universe does not know is
UNK at once and counted in `n_oov` (INVALID #7).

The arrays: `cat` int16 `[T, N_CAT]`, `cont` float32 `[T, N_CONT]`, `sec`
int8 `[T]`; observation tokens first (STATE at 0), then one token per legal
action in `actions` order (`n_act`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import intent_log
from .static_tables import CARD_TYPES, card_types

ARMS = ("S_ctx", "S_ctx_tok", "S_ctxarch_rl2in")

SECTIONS = (
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
SEC = {s: i for i, s in enumerate(SECTIONS)}
N_SECTION = len(SECTIONS)
ACT_SECTIONS = frozenset(SEC[s] for s in SECTIONS if s.startswith("ACT_"))

# Categorical slots.
SLOTS = (
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
S = {s: i for i, s in enumerate(SLOTS)}
N_CAT = len(SLOTS)
# Slots whose ids are content (rare-id rule and id dropout apply).
CONTENT_SLOTS = ("card", "relic", "potion", "power", "monster", "enchantment", "rl2key")

AFFLICTIONS = (
    "Bound",
    "Devoured",
    "Entangled",
    "Galvanized",
    "Hexed",
    "Ringing",
    "Smog",
    "Weighted",
    "Tainted",
)
INTENT_KINDS = (
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
    intent_log.NONE_KIND,
)
CHOICE_KINDS = (
    "card_select",
    "put_back",
    "exhaust_pile",
    "orb_slot",
    "enemy_target",
    "add_generated_to_hand",
    "add_generated_retained_to_hand",
    "add_generated_free_to_hand",
    "paels_tooth_stash",
    "upgrade_in_hand",
    "add_from_draw_by_type",
    "add_from_discard_free",
    "discard_from_hand",
    "exhaust_from_hand",
    "discard_from_hand_and_draw",
    "make_free_in_hand",
    "monster_curse_select",
    "add_from_draw_to_hand",
    "foregone_conclusion_from_draw",
    "add_from_discard_to_hand",
    "apply_retain_in_hand",
    "apply_sly_in_hand",
    "apply_ethereal_in_hand",
    "transform_in_hand_to_minion_strike",
    "transform_in_hand_to_minion_sacrifice",
    "transfigure_card_in_hand",
    "transform_draw_card_to_soul",
    "transform_draw_cards_to_minion_dive_bomb",
    "clone_chosen_card_next_turn",
    "retain_at_turn_end",
    "auto_play_skill_from_hand",
)
CHOICE_PILES = ("hand", "draw", "discard", "exhaust", "play", "generated")
MAX_ENEMY_SLOTS = 5
MAX_HAND_SLOTS = 12

# ── continuous columns ───────────────────────────────────────────────────
# card block (card tokens, ACT_PLAY, ACT_CHOOSE)
C_COST, C_BASE_COST, C_DMG, C_BLK, C_CALC, C_HAS_CALC = range(6)
C_ETHEREAL, C_UNPLAYABLE, C_INNATE, C_SLY, C_SLY_TURN, C_RETAIN_TURN = range(6, 12)
C_ENCH_AMT, C_AFF_STACKS, C_COUNT, C_PLAYABLE, C_DMG_PREV, C_BLK_PREV = range(12, 18)
# STATE
C_ST = 18
(
    C_HP_FRAC,
    C_HP,
    C_MAXHP,
    C_BLOCK,
    C_ENERGY,
    C_MAXE,
    C_STARS,
    C_ROUND,
    C_PLAYED,
    C_N_HAND,
    C_N_DRAW,
    C_N_DISC,
    C_N_EXH,
    C_POT_SLOTS,
    C_IS_BOSS,
    C_IS_ELITE,
    C_IS_HALL,
    C_CH_MIN,
    C_CH_MAX,
    C_CH_SKIP,
    C_INCOMING,
    C_UNBLOCKED,
) = range(C_ST, C_ST + 22)
# ENEMY
C_EN = C_ST + 22
(
    C_E_HPF,
    C_E_HP,
    C_E_MAXHP,
    C_E_BLOCK,
    C_E_ALIVE,
    C_E_REMOVED,
    C_E_AMT,
    C_E_EFF,
    C_E_TIMES,
    C_E_TOT,
) = range(C_EN, C_EN + 10)
# EHIST
C_EH = C_EN + 10
C_H_LAG, C_H_AMT, C_H_EFF, C_H_TIMES, C_H_ALIVE = range(C_EH, C_EH + 5)
# powers / relics
C_PW = C_EH + 5
C_P_AMT, C_P_DISP, C_P_CTR = range(C_PW, C_PW + 3)
C_R_CTR, C_R_HAS, C_R_WAX, C_R_MELT = range(C_PW + 3, C_PW + 7)
# action extras
C_AX = C_PW + 7
C_T_HPF, C_T_EHP, C_T_DMG, C_N_SEL = range(C_AX, C_AX + 4)
# rl2 features
C_RL2S = C_AX + 4
RL2_STATE_DIM, RL2_ACTION_DIM = 32, 24
C_RL2X = C_RL2S + RL2_STATE_DIM
N_CONT = C_RL2X + RL2_ACTION_DIM
RL2_COLS = tuple(range(C_RL2S, N_CONT))


class Table:
    """A content table `[names..., UNK, PAD]` keyed by a normalizer."""

    def __init__(self, names: list[str], norm: Any = None) -> None:
        self.names = list(names)
        self.norm = norm or (lambda s: s)
        self.idx: dict[str, int] = {}
        for i, n in enumerate(self.names):
            self.idx.setdefault(self.norm(n), i)
        self.unk = len(self.names)
        self.pad = len(self.names) + 1
        self.size = len(self.names) + 2

    def index(self, name: Any) -> tuple[int, bool]:
        """`(row, oov)`; an unknown name is UNK with `oov` True."""
        i = self.idx.get(self.norm(str(name)))
        return (self.unk, True) if i is None else (i, False)


@dataclass(frozen=True)
class Vocab:
    tables: tuple[Table, ...]

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(t.size for t in self.tables)

    @property
    def pads(self) -> np.ndarray:
        return np.asarray([t.pad for t in self.tables], np.int16)

    @property
    def unks(self) -> np.ndarray:
        return np.asarray([t.unk for t in self.tables], np.int16)

    def table(self, slot: str) -> Table:
        return self.tables[S[slot]]

    def digest(self) -> dict[str, int]:
        return {s: t.size for s, t in zip(SLOTS, self.tables, strict=True)}


def rl2_key_universe(cards: list[str], potions: list[str]) -> list[str]:
    """Every action key `sts2_mcts::rollout_net::ActionKey::name` can produce."""
    out = ["end_turn", "choice", "other"]
    for c in cards:
        out += [f"card:{c}", f"card:{c}+"]
    out += [f"potion:{p}" for p in potions]
    return out


def enchantment_norm(name: str) -> str:
    """The wire spells enchantments as display names (`Tezcatara's Ember`,
    `Perfect Fit`); the registry as `TEZCATARAS_EMBER`. Both normalize to the
    latter."""
    from ppo import vocab as ppo_vocab

    s = ppo_vocab.canonical(str(name).replace("'", "").replace("\u2019", ""))
    return re.sub(r"[^A-Z0-9]+", "_", s).strip("_")


@lru_cache(maxsize=1)
def vocab() -> Vocab:
    """The full-universe vocabularies (engine registries; no data)."""
    import sts2_rs

    from ppo import vocab as ppo_vocab

    cards = ppo_vocab.cards()
    potions = ppo_vocab.potions()
    ench = ppo_vocab.enchantments()
    card_names = list(sts2_rs.list_registry_names("card"))
    potion_names = list(sts2_rs.list_registry_names("potion"))
    tables = (
        Table(cards.names, ppo_vocab.canonical),
        Table(["0", "1"]),
        Table(list(sts2_rs.list_registry_names("relic"))),
        Table(potions.names, ppo_vocab.canonical),
        Table(list(sts2_rs.list_registry_names("power"))),
        Table(list(sts2_rs.monster_species())),
        Table(ench.names, enchantment_norm),
        Table(list(AFFLICTIONS)),
        Table(list(INTENT_KINDS)),
        Table([str(i) for i in range(MAX_ENEMY_SLOTS)]),
        Table([str(i) for i in range(MAX_HAND_SLOTS)]),
        Table(list(CHOICE_KINDS)),
        Table(list(CARD_TYPES)),
        Table(list(CHOICE_PILES)),
        Table(rl2_key_universe(card_names, potion_names)),
    )
    return Vocab(tables)


# ── building ─────────────────────────────────────────────────────────────


class _Builder:
    def __init__(self, v: Vocab) -> None:
        self.v = v
        self.cat: list[np.ndarray] = []
        self.cont: list[np.ndarray] = []
        self.sec: list[int] = []
        self.n_oov = 0
        self.n_ids = 0

    def new(self, section: str) -> tuple[np.ndarray, np.ndarray]:
        cat = self.v.pads.copy()
        cont = np.zeros(N_CONT, np.float32)
        self.cat.append(cat)
        self.cont.append(cont)
        self.sec.append(SEC[section])
        return cat, cont

    def put(self, cat: np.ndarray, slot: str, name: Any) -> None:
        row, oov = self.v.tables[S[slot]].index(name)
        cat[S[slot]] = row
        if slot in CONTENT_SLOTS:
            self.n_ids += 1
            self.n_oov += int(oov)


def _f(x: Any) -> float:
    return 0.0 if x is None else float(x)


F16_MAX = 65504.0


def f16(cont: np.ndarray) -> np.ndarray:
    """`cont` as float16, saturating at its finite range instead of overflowing.

    Waterfall Giant's honest snapshot shows an enemy HP near 1e9; a plain cast
    made it inf and every logit of the net NaN.
    """
    return np.clip(cont, -F16_MAX, F16_MAX).astype(np.float16)


def _card_type(card_id: str) -> str:
    # The active game build's table (`static_tables.card_types`).
    return card_types().get(card_id, "?")


def card_feats(
    b: _Builder, cat: np.ndarray, cont: np.ndarray, c: dict[str, Any], count: float = 0.0
) -> None:
    """A card's categorical slots and card-block columns."""
    b.put(cat, "card", c["id"])
    b.put(cat, "upgraded", "1" if c.get("upgraded") else "0")
    b.put(cat, "card_type", _card_type(c["id"]))
    if c.get("enchantment"):
        b.put(cat, "enchantment", c["enchantment"])
    if c.get("affliction"):
        b.put(cat, "affliction", c["affliction"])
    flags = set(c.get("flags") or ())
    cont[C_COST] = _f(c.get("cost")) / 3.0
    cont[C_BASE_COST] = _f(c.get("base_cost")) / 3.0
    cont[C_DMG] = _f(c.get("base_damage")) / 20.0
    cont[C_BLK] = _f(c.get("base_block")) / 20.0
    if "calculated_value" in c:
        cont[C_CALC] = _f(c["calculated_value"]) / 20.0
        cont[C_HAS_CALC] = 1.0
    cont[C_ETHEREAL] = float("ethereal" in flags)
    cont[C_UNPLAYABLE] = float("unplayable" in flags)
    cont[C_INNATE] = float("innate" in flags)
    cont[C_SLY] = float("sly" in flags)
    cont[C_SLY_TURN] = float("sly_turn" in flags)
    cont[C_RETAIN_TURN] = float("retain_turn" in flags)
    cont[C_ENCH_AMT] = _f(c.get("enchantment_amount")) / 5.0
    cont[C_AFF_STACKS] = _f(c.get("affliction_stacks")) / 5.0
    cont[C_COUNT] = count / 5.0


def _power_feats(b: _Builder, cat: np.ndarray, cont: np.ndarray, p: dict[str, Any]) -> None:
    b.put(cat, "power", p["name"])
    cont[C_P_AMT] = _f(p.get("amount")) / 10.0
    cont[C_P_DISP] = _f(p.get("display_amount")) / 10.0
    cont[C_P_CTR] = _f(p.get("counter")) / 10.0


def _intent_total(i: dict[str, Any]) -> float:
    if i.get("kind") not in ("Attack", "DeathBlow"):
        return 0.0
    eff = i.get("effective_amount", i.get("amount", 0)) or 0
    return float(eff) * float(i.get("times") or 1)


def _hand_extras(obs: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, float]]:
    playable = {int(a["card_index"]) for a in actions if a.get("action") == "play"}
    out = []
    for i, c in enumerate(obs["hand"]):
        dmg = [int(v) for v in (c.get("dmg") or {}).values()]
        out.append(
            {
                "playable": float(i in playable),
                "dmg": (max(dmg) if dmg else 0) / 30.0,
                "blk": _f(c.get("blk")) / 20.0,
            }
        )
    return out


def _hand_card(
    b: _Builder,
    cat: np.ndarray,
    cont: np.ndarray,
    obs: dict[str, Any],
    i: int,
    extras: list[dict[str, float]],
) -> None:
    c = obs["hand"][i]
    card_feats(b, cat, cont, c)
    b.put(cat, "hand_slot", str(i))
    cont[C_PLAYABLE] = extras[i]["playable"]
    cont[C_DMG_PREV] = extras[i]["dmg"]
    cont[C_BLK_PREV] = extras[i]["blk"]


def _state_token(b: _Builder, obs: dict[str, Any], with_rl2: bool, rl2s: Any) -> None:
    cat, cont = b.new("STATE")
    p = obs["player"]
    hp, mhp = _f(p.get("hp")), max(1.0, _f(p.get("max_hp")))
    cont[C_HP_FRAC] = hp / mhp
    cont[C_HP] = hp / 100.0
    cont[C_MAXHP] = mhp / 100.0
    cont[C_BLOCK] = _f(p.get("block")) / 30.0
    cont[C_ENERGY] = _f(p.get("energy")) / 5.0
    cont[C_MAXE] = _f(p.get("max_energy")) / 5.0
    cont[C_STARS] = _f(p.get("stars")) / 5.0
    cont[C_ROUND] = _f(obs.get("round")) / 20.0
    cont[C_PLAYED] = float(sum(p.get("cards_played_this_turn") or [])) / 10.0
    n_draw = sum(n for _, n in obs["draw_unknown"]) + len(obs["draw_top"]) + len(obs["draw_bottom"])
    cont[C_N_HAND] = len(obs["hand"]) / 30.0
    cont[C_N_DRAW] = n_draw / 30.0
    cont[C_N_DISC] = sum(n for _, n in obs["discard"]) / 30.0
    cont[C_N_EXH] = sum(n for _, n in obs["exhaust"]) / 30.0
    cont[C_POT_SLOTS] = _f(p.get("potion_slots")) / 5.0
    cont[C_IS_BOSS] = float(obs["room"] == "boss")
    cont[C_IS_ELITE] = float(obs["room"] == "elite")
    cont[C_IS_HALL] = float(obs["room"] == "monster")
    ch = obs.get("choice")
    if ch is not None:
        b.put(cat, "choice_kind", ch.get("kind"))
        cont[C_CH_MIN] = _f(ch.get("min_select")) / 5.0
        cont[C_CH_MAX] = _f(ch.get("max_select")) / 5.0
        cont[C_CH_SKIP] = float(bool(ch.get("can_skip")))
    inc = 0.0
    for m in obs["enemies"]:
        if not m["is_dead"]:
            inc += sum(_intent_total(i) for i in m["intents"])
    cont[C_INCOMING] = inc / 50.0
    cont[C_UNBLOCKED] = max(0.0, inc - _f(p.get("block"))) / 50.0
    if with_rl2:
        cont[C_RL2S : C_RL2S + RL2_STATE_DIM] = rl2s


def _obs_tokens(b: _Builder, obs: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    extras = _hand_extras(obs, actions)
    for i in range(len(obs["hand"])):
        cat, cont = b.new("HAND")
        _hand_card(b, cat, cont, obs, i, extras)
    for sec, pile in (("DRAW_UNK", "draw_unknown"), ("DISCARD", "discard"), ("EXHAUST", "exhaust")):
        for c, n in obs[pile]:
            cat, cont = b.new(sec)
            card_feats(b, cat, cont, c, float(n))
    for sec, pile in (("DRAW_TOP", "draw_top"), ("DRAW_BOT", "draw_bottom")):
        # The pin rank (1 = first) rides in the count column.
        for r, c in enumerate(obs[pile]):
            cat, cont = b.new(sec)
            card_feats(b, cat, cont, c, float(r + 1))
    for c in obs["play"]:
        cat, cont = b.new("PLAY")
        card_feats(b, cat, cont, c)
    for r in obs["relics"]:
        cat, cont = b.new("RELIC")
        b.put(cat, "relic", r["id"])
        if "counter" in r:
            cont[C_R_CTR] = _f(r["counter"]) / 10.0
            cont[C_R_HAS] = 1.0
        cont[C_R_WAX] = float(bool(r.get("is_wax")))
        cont[C_R_MELT] = float(bool(r.get("is_melted")))
    for pid in obs["potions"]:
        cat, _ = b.new("POTION")
        b.put(cat, "potion", pid)
    for p in obs["powers"]:
        cat, cont = b.new("PPOWER")
        _power_feats(b, cat, cont, p)
    ehist = obs.get("ehist") or [[] for _ in obs["enemies"]]
    for j, m in enumerate(obs["enemies"]):
        alive = float(not m["is_dead"])
        cat, cont = b.new("ENEMY")
        b.put(cat, "monster", m["kind"])
        b.put(cat, "enemy_slot", str(j))
        hi = intent_log.head_intent(m)
        b.put(cat, "intent", hi["kind"])
        mhp = max(1.0, _f(m.get("max_hp")))
        cont[C_E_HPF] = _f(m.get("hp")) / mhp
        cont[C_E_HP] = _f(m.get("hp")) / 200.0
        cont[C_E_MAXHP] = mhp / 200.0
        cont[C_E_BLOCK] = _f(m.get("block")) / 30.0
        cont[C_E_ALIVE] = alive
        cont[C_E_REMOVED] = float(bool(m.get("removed_after_death")))
        cont[C_E_AMT] = _f(hi.get("amount")) / 30.0
        cont[C_E_EFF] = _f(hi.get("effective_amount")) / 30.0
        cont[C_E_TIMES] = _f(hi.get("times")) / 5.0
        cont[C_E_TOT] = _intent_total(hi) / 50.0
        for p in m["powers"]:
            cat, cont = b.new("EPOWER")
            _power_feats(b, cat, cont, p)
            b.put(cat, "monster", m["kind"])
            b.put(cat, "enemy_slot", str(j))
        for s in m.get("stolen", []):
            # A stolen card is visible on its thief: a PLAY-less card token
            # tied to the enemy slot.
            cat, cont = b.new("EPOWER")
            card_feats(b, cat, cont, {"id": s["id"], "upgraded": s.get("upgraded")})
            b.put(cat, "monster", m["kind"])
            b.put(cat, "enemy_slot", str(j))
        for h in ehist[j] if j < len(ehist) else []:
            cat, cont = b.new("EHIST")
            b.put(cat, "intent", h["kind"])
            b.put(cat, "monster", m["kind"])
            b.put(cat, "enemy_slot", str(j))
            cont[C_H_LAG] = _f(h.get("lag")) / 3.0
            cont[C_H_AMT] = _f(h.get("amount")) / 30.0
            cont[C_H_EFF] = _f(h.get("effective_amount")) / 30.0
            cont[C_H_TIMES] = _f(h.get("times")) / 5.0
            cont[C_H_ALIVE] = alive


def _target(
    b: _Builder, cat: np.ndarray, cont: np.ndarray, obs: dict[str, Any], t: Any
) -> dict[str, Any] | None:
    if t is None:
        return None
    t = int(t)
    b.put(cat, "enemy_slot", str(t))
    if t >= len(obs["enemies"]):
        return None
    m = obs["enemies"][t]
    b.put(cat, "monster", m["kind"])
    mhp = max(1.0, _f(m.get("max_hp")))
    cont[C_T_HPF] = _f(m.get("hp")) / mhp
    cont[C_T_EHP] = (_f(m.get("hp")) + _f(m.get("block"))) / 100.0
    return m


def _action_tokens(
    b: _Builder,
    obs: dict[str, Any],
    actions: list[dict[str, Any]],
    rl2: dict[str, Any] | None,
    arm: str,
) -> None:
    extras = _hand_extras(obs, actions)
    ch = obs.get("choice")
    for k, a in enumerate(actions):
        kind = a.get("action")
        if arm == "S_ctxarch_rl2in":
            cat, cont = b.new("ACT_RL2")
        elif kind == "play":
            cat, cont = b.new("ACT_PLAY")
            i = int(a["card_index"])
            _hand_card(b, cat, cont, obs, i, extras)
            if _target(b, cat, cont, obs, a.get("target_index")) is not None:
                dmg = (obs["hand"][i].get("dmg") or {}).get(str(int(a["target_index"])))
                cont[C_T_DMG] = _f(dmg) / 30.0
        elif kind == "end_turn":
            cat, cont = b.new("ACT_END")
        elif kind in ("potion", "drink_potion"):
            cat, cont = b.new("ACT_POTION")
            slot = int(a["slot"])
            if slot < len(obs["potions"]):
                b.put(cat, "potion", obs["potions"][slot])
            t = a.get("target_index")
            if t is not None:
                b.put(cat, "enemy_slot", str(int(t)))
        elif kind == "choose" and a.get("selection"):
            cat, cont = b.new("ACT_CHOOSE")
            sel = [int(s) for s in a["selection"]]
            opts = ch["options"] if ch else []
            if sel[0] < len(opts):
                o = opts[sel[0]]
                b.put(cat, "card", o["id"])
                b.put(cat, "upgraded", "1" if o.get("upgraded") else "0")
                b.put(cat, "card_type", _card_type(o["id"]))
                b.put(cat, "choice_pile", o.get("pile", "generated"))
                if o.get("pile") == "hand" and "hand_index" in o:
                    b.put(cat, "hand_slot", str(o["hand_index"]))
            if ch is not None:
                b.put(cat, "choice_kind", ch.get("kind"))
            cont[C_N_SEL] = len(sel) / 3.0
        elif kind == "choose":
            cat, cont = b.new("ACT_SKIP")
        else:
            cat, cont = b.new("ACT_OTHER")
        if rl2 is not None:
            b.put(cat, "rl2key", rl2["keys"][k])
            cont[C_RL2X : C_RL2X + RL2_ACTION_DIM] = rl2["x"][k]


def tokens(
    obs: dict[str, Any], actions: list[dict[str, Any]], rl2feat: dict[str, Any], arm: str
) -> dict[str, Any]:
    """The decision's tokens for `arm`.

    Args:
        obs: The honest snapshot with `ehist`.
        actions: Legal actions in the snapshot's option order.
        rl2feat: `{state, x, keys}` of `rollout_features(cs, 1)`, aligned
            with `actions`.
        arm: One of `ARMS`.

    Returns:
        `{cat, cont, sec, n_act, n_oov, n_ids}`.
    """
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    if len(rl2feat["keys"]) != len(actions) or len(rl2feat["x"]) != len(actions):
        raise ValueError("rl2feat is not aligned with actions")
    b = _Builder(vocab())
    rl2s = np.asarray(rl2feat["state"], np.float32)
    rl2 = None if arm == "S_ctx_tok" else rl2feat
    if arm == "S_ctxarch_rl2in":
        _cat, cont = b.new("STATE")
        cont[C_RL2S : C_RL2S + RL2_STATE_DIM] = rl2s
    else:
        _state_token(b, obs, arm == "S_ctx", rl2s)
        _obs_tokens(b, obs, actions)
    _action_tokens(b, obs, actions, rl2, arm)
    return {
        "cat": np.stack(b.cat).astype(np.int16),
        "cont": np.stack(b.cont).astype(np.float32),
        "sec": np.asarray(b.sec, np.int8),
        "n_act": len(actions),
        "n_oov": b.n_oov,
        "n_ids": b.n_ids,
    }


def derive(cat: np.ndarray, cont: np.ndarray, sec: np.ndarray, n_act: int, arm: str) -> tuple:
    """`arm`'s tokens from `S_ctx`'s, without the snapshot (the cache stores
    `S_ctx` only). Equal to `tokens(..., arm)` (a test pins it)."""
    if arm == "S_ctx":
        return cat, cont, sec
    v = vocab()
    if arm == "S_ctx_tok":
        cat = cat.copy()
        cont = cont.copy()
        cat[:, S["rl2key"]] = v.tables[S["rl2key"]].pad
        cont[:, C_RL2S:] = 0
        return cat, cont, sec
    if arm == "S_ctxarch_rl2in":
        n = len(sec)
        keep = np.r_[0, np.arange(n - n_act, n)]
        pads = v.pads
        cat2 = np.tile(pads, (len(keep), 1))
        cat2[1:, S["rl2key"]] = cat[n - n_act :, S["rl2key"]]
        cont2 = np.zeros((len(keep), cont.shape[1]), cont.dtype)
        cont2[0, C_RL2S : C_RL2S + RL2_STATE_DIM] = cont[0, C_RL2S : C_RL2S + RL2_STATE_DIM]
        cont2[1:, C_RL2X:] = cont[n - n_act :, C_RL2X:]
        sec2 = np.full(len(keep), SEC["ACT_RL2"], sec.dtype)
        sec2[0] = SEC["STATE"]
        return cat2, cont2, sec2
    raise ValueError(arm)
