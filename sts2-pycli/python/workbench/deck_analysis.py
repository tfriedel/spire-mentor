"""Grade a deck along the three axes — Damage, Block, Velocity.

The framing follows Jorbs' "jobs" model (and the volume-cube analogy from the
untapped.gg deckbuilding article): a deck's strength is roughly
``Damage × Block × Velocity``, so a build starved on any single axis is weak no
matter how tall the others stand. We surface each axis as a 0-100 score plus the
raw numbers behind it (expected damage/block per turn, scaling counts, velocity
engine pieces) so the panel teaches *why*, not just *what*.

Two static tables drive this, both committed under ``workbench/data/``:

* ``card_db.json``  — mechanical stats (cost, type, damage, block, hits, AOE),
  distilled from the wiki manifest by :mod:`workbench.build_card_db`.
* ``card_tags.json`` — curated judgement calls (damage/block *role*: frontload vs
  scaling, and velocity engine type), classified per card.

Both are keyed by loc-key, which equals the runtime ``card.id``, so the analyzer
works on every screen — including card-reward / map views where the engine ships
deck cards without numeric ``stats``.

**Per game build (GH #877).** The tables above describe the default build
(v0.107.1). Every other supported build has its own under
``workbench/data/<ver>/`` (:func:`data_dir_for`): a full ``card_db.json`` built
from that build's wiki manifest, and a ``card_tags.json`` OVERLAY merged over the
default tags (an entry replaces the default's whole entry; ``null`` drops a card
the build removed). A checkout without a build's tables (the public snapshot)
grades that build's decks with the default tables. Every entry point takes
``game_version`` (``None`` = the default); the Rust twin is
``sts2-workbench/src/deck.rs`` (``deck::tables_for``).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS

_DATA = Path(__file__).resolve().parent / "data"

# Standard combat assumptions. Real runs vary (relics, larger hands), but a fixed
# baseline keeps the score comparable turn-to-turn and deck-to-deck.
DEFAULT_ENERGY = 3
DEFAULT_HAND = 5
# Rollouts averaged per analysis. The simulation is multi-turn, so fewer samples
# than a single-turn draw are needed for a stable mean.
DEFAULT_SAMPLES = 200
# A combat is simulated over this many turns — long enough for scaling (Strength,
# per-turn powers) to visibly compound, short enough to stay snappy.
SIM_TURNS = 7
# Single-target (boss) vs group (hallway) enemy counts, blended for the damage
# axis so AOE is rewarded without dominating single-target output.
SIM_ENEMIES_SOLO = 1
SIM_ENEMIES_GROUP = 3
# Safety bound on plays per simulated turn (a net-positive 0-cost engine could
# otherwise loop). Real such engines exhaust; this also caps the worst case.
_MAX_PLAYS_PER_TURN = 40

# Saturation scales mapping simulated per-turn output onto 0-100. Tuned so a
# starter deck lands low-mid and a strong mid-run deck ~65-80 on each axis.
_DMG_SCALE = 40.0
_BLK_SCALE = 20.0
_VEL_SCALE = 16.0

# Velocity dilution (the Reddit model's `1/D` insight): a given draw/energy
# engine is worth less in a bloated deck because key cards are reached less
# often. A deck at or below this size pays no penalty; past it the engine
# credit scales by ``_LEAN_DECK / deck_size`` (fraction-of-deck processed).
_LEAN_DECK = 15

# Context lenses — weighted blends of the three axis scores for different fight
# shapes (Reddit §7). The headline ``volume`` stays a geometric mean (a starved
# axis can't hide); these linear blends are an optional "what matters here" view.
#                  (w_damage, w_block, w_velocity)
_LENSES: dict[str, tuple[float, float, float]] = {
    "hallway": (0.60, 0.20, 0.20),  # simple groups — burst damage dominates
    "boss": (0.34, 0.33, 0.33),  # single big threat — balanced survival
    "scaling": (0.20, 0.40, 0.40),  # long fights — block + velocity carry
}

_DEAD_TYPES = {"Curse", "Status"}


def _version_key(version: str | None) -> str:
    """``version`` validated, ``None`` -> the default build."""
    key = version or LEGACY_GAME_VERSION
    if key not in SUPPORTED_GAME_VERSIONS:
        raise ValueError(
            f"unsupported game version {key!r}; supported: {', '.join(SUPPORTED_GAME_VERSIONS)}"
        )
    return key


def data_dir_for(version: str | None = None) -> Path:
    """Where game build ``version``'s tables live: ``data/`` for the default
    build, ``data/<ver>/`` for every other (``build_card_db --version``)."""
    key = _version_key(version)
    return _DATA if key == LEGACY_GAME_VERSION else _DATA / key


@cache
def card_db_for(version: str | None = None) -> dict[str, Any]:
    """Build ``version``'s ``card_db.json`` (the default build's when that
    build has no table on disk)."""
    path = data_dir_for(version) / "card_db.json"
    if not path.is_file():
        path = _DATA / "card_db.json"
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def card_tags_for(version: str | None = None) -> dict[str, Any]:
    """Build ``version``'s curated tags: the default ``card_tags.json`` with
    that build's overlay (if any) merged over it, ``_meta`` dropped."""
    raw = json.loads((_DATA / "card_tags.json").read_text(encoding="utf-8"))
    raw.pop("_meta", None)
    overlay_path = data_dir_for(version) / "card_tags.json"
    if overlay_path.parent != _DATA and overlay_path.is_file():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        overlay.pop("_meta", None)
        for cid, entry in overlay.items():
            if entry is None:
                raw.pop(cid, None)
            else:
                raw[cid] = entry
    return raw


def _card_db() -> dict[str, Any]:
    """The default build's stats table."""
    return card_db_for(None)


def _card_tags() -> dict[str, Any]:
    """The default build's curated tags."""
    return card_tags_for(None)


def _card_id(card: Any) -> str:
    """Normalise a deck entry to its loc-key (matches ``card_db`` keys)."""
    if isinstance(card, str):
        cid = card
    else:
        cid = card.get("id") or card.get("card_id") or card.get("name") or ""
    cid = cid.split(".")[-1]  # drop any "<KEY>.title" style suffix
    return cid.rstrip("+").split("+")[0] if "+" in cid else cid


def _resolved(card: Any, version: str | None = None) -> dict[str, Any]:
    """Merge a deck entry with build ``version``'s static stats + tags into one
    flat record."""
    cid = _card_id(card)
    card_db = card_db_for(version)
    db = card_db.get(cid, {})
    tags = card_tags_for(version).get(cid, {})
    upgraded = bool(card.get("upgraded")) if isinstance(card, dict) else False
    dmg = db.get("damage", 0)
    blk = db.get("block", 0)
    draw = db.get("draw", 0)
    energy = db.get("energy", 0)
    strength = db.get("strength", 0)
    dexterity = db.get("dexterity", 0)
    if upgraded:
        dmg = db.get("damage_up") or dmg
        blk = db.get("block_up") or blk
        draw = db.get("draw_up") or draw
        energy = db.get("energy_up") or energy
        strength = db.get("strength_up") or strength
        dexterity = db.get("dexterity_up") or dexterity
    ctype = db.get("type") or (card.get("type") if isinstance(card, dict) else None)
    return {
        "id": cid,
        "title": db.get("title", cid),
        "type": ctype,
        "rarity": db.get("rarity"),
        "cost": db.get("cost", 0),
        "cost_x": db.get("cost_x", False),
        "damage": dmg,
        "hits": db.get("hits", 1),
        "variable_hits": db.get("variable_hits", False),
        "block": blk,
        "draw": draw,
        "energy": energy,
        "strength": strength,
        "dexterity": dexterity,
        "per_turn": db.get("per_turn", False),
        "exhaust": db.get("exhaust", False),
        "unplayable": db.get("unplayable", False),
        "aoe": db.get("aoe", False),
        "upgraded": upgraded,
        "known": cid in card_db,
        "damage_role": tags.get("damage_role", "none"),
        "block_role": tags.get("block_role", "none"),
        "velocity": tags.get("velocity", []),
        "note": tags.get("note", ""),
    }


def _effective_cost(c: dict[str, Any], energy_left: int) -> int:
    """Energy a card costs to play now (X-cost spends whatever is left)."""
    return energy_left if c["cost_x"] else max(0, c["cost"])


def _card_priority(c: dict[str, Any], objective: str) -> int:
    """Play order within a turn: engine first, then the objective, then filler.

    Powers (scaling) and tempo cards (draw/energy) are played first so their
    benefit compounds before the turn's attacks/blocks resolve; then the cards
    that serve the rollout's objective (damage or block), then anything else.
    """
    if c["type"] == "Power" or c["strength"] or c["dexterity"]:
        return 4
    if c["draw"] or c["energy"]:
        return 3
    if objective == "damage" and c["damage"] > 0:
        return 2
    if objective == "block" and c["block"] > 0:
        return 2
    return 1


@dataclass
class _Combat:
    """Mutable player state carried across a simulated combat's turns."""

    strength: int = 0
    dexterity: int = 0
    str_per_turn: int = 0  # banked by per-turn Strength powers (Demon Form)
    blk_per_turn: int = 0  # banked by per-turn Block powers (Metallicize)


def _apply_card(c: dict[str, Any], st: _Combat, enemies: int) -> tuple[int, int, int, int]:
    """Resolve one card play. Returns (damage, block, extra_cards, extra_energy).

    Models the scaling vectors the card DB exposes: attacks scale with current
    Strength (per hit) and AOE multiplies by enemy count; block scales with
    Dexterity; Strength/Dexterity gains feed future turns (immediately, or every
    turn for per-turn powers); draw/energy feed the turn's tempo.
    """
    dmg = 0
    if c["damage"] > 0:
        hits = c["hits"]
        per_hit = c["damage"] + st.strength
        targets = enemies if c["aoe"] else 1
        dmg = max(0, per_hit) * hits * targets
    block = (c["block"] + st.dexterity) if c["block"] > 0 else 0
    if c["strength"]:
        if c["per_turn"]:
            st.str_per_turn += c["strength"]
        else:
            st.strength += c["strength"]
    if c["dexterity"] and not c["per_turn"]:
        st.dexterity += c["dexterity"]
    return dmg, block, c["draw"], c["energy"]


def _rollout(
    deck: list[dict[str, Any]],
    *,
    objective: str,
    enemies: int,
    turns: int,
    base_energy: int,
    base_hand: int,
    rng: random.Random,
    no_engine: bool = False,
) -> tuple[float, float, float, int, int]:
    """Play the deck for ``turns`` turns and return per-turn output + bottlenecks.

    Returns ``(damage, block, plays)`` per turn plus the count of turns that
    ended *energy-limited* (cards left in hand we couldn't afford) and
    *draw-limited* (hand emptied with energy to spare) — the diagnostic the
    Reddit model's ``min(Draw, E/C)`` exposes: which constraint actually binds.

    A deliberately simple but faithful-in-spirit policy: each turn draw a hand,
    then greedily play affordable cards (engine → objective → filler), letting
    draw/energy cards extend the turn. Powers leave play; everything else cycles
    through draw → discard → reshuffle, so deck dilution (curses, unupgraded
    bloat) and thinning are felt naturally.

    With ``no_engine`` the deck's own draw/energy gains are ignored, giving the
    baseline plays-per-turn the deck would manage on raw energy alone — the
    reference the velocity axis measures real engines against.
    """
    draw_pile = deck[:]
    rng.shuffle(draw_pile)
    discard: list[dict[str, Any]] = []
    st = _Combat()
    dmg_total = blk_total = plays_total = 0
    draw_limited = energy_limited = 0

    def draw_cards(hand: list[dict[str, Any]], k: int) -> None:
        for _ in range(k):
            if not draw_pile:
                if not discard:
                    return
                draw_pile.extend(discard)
                discard.clear()
                rng.shuffle(draw_pile)
            hand.append(draw_pile.pop())

    for _ in range(turns):
        st.strength += st.str_per_turn  # start-of-turn per-turn powers
        blk_total += st.blk_per_turn
        energy = base_energy
        hand: list[dict[str, Any]] = []
        draw_cards(hand, base_hand)
        plays_this_turn = 0
        while plays_this_turn < _MAX_PLAYS_PER_TURN:
            playable = [
                c
                for c in hand
                if c["type"] not in _DEAD_TYPES
                and not c["unplayable"]
                and _effective_cost(c, energy) <= energy
            ]
            if not playable:
                # Classify why the turn stalled (the binding constraint). Live
                # cards still in hand we couldn't afford → energy-limited; an
                # empty/dead hand with energy to spare → draw-limited.
                live_in_hand = [
                    c for c in hand if c["type"] not in _DEAD_TYPES and not c["unplayable"]
                ]
                if live_in_hand:
                    energy_limited += 1
                elif energy > 0:
                    draw_limited += 1
                break
            card = max(playable, key=lambda c: _card_priority(c, objective))
            hand.remove(card)
            energy -= _effective_cost(card, energy)
            d, b, drew, gained = _apply_card(card, st, enemies)
            dmg_total += d
            blk_total += b
            if no_engine:  # measure baseline tempo without the deck's own engines
                drew = gained = 0
            energy += gained
            plays_total += 1
            plays_this_turn += 1
            if drew:
                draw_cards(hand, drew)
            # Exhausted cards leave play; powers leave play; the rest cycle.
            if not card["exhaust"] and card["type"] != "Power":
                discard.append(card)
        discard.extend(c for c in hand if c["type"] != "Power" and not c["exhaust"])
    return dmg_total / turns, blk_total / turns, plays_total / turns, draw_limited, energy_limited


def _simulate(
    deck: list[dict[str, Any]],
    *,
    enemies_solo: int,
    enemies_group: int,
    turns: int,
    base_energy: int,
    base_hand: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Average many shuffled combat rollouts into per-turn axis numbers.

    Damage is reported both single-target (boss proxy) and vs a group (hallway
    proxy) so AOE is rewarded by the group run without dominating; block is
    enemy-count-independent. All three axes — scaling, AOE and velocity — are
    emergent from playing the deck, not bolted-on bonuses.
    """
    if not [c for c in deck if c["type"] not in _DEAD_TYPES]:
        b = float(base_energy)
        return {
            "damage_solo": 0.0,
            "damage_group": 0.0,
            "block": 0.0,
            "plays": b,
            "plays_base": b,
            "draw_limited": 0.0,
            "energy_limited": 0.0,
        }
    rng = random.Random(seed)
    n = max(1, samples)
    d_solo = d_group = blk = plays = plays_base = 0.0
    draw_lim = energy_lim = 0.0
    common = dict(turns=turns, base_energy=base_energy, base_hand=base_hand)
    for _ in range(n):
        ds, _, ps, dl, el = _rollout(
            deck, objective="damage", enemies=enemies_solo, rng=rng, **common
        )
        dg, _, _, _, _ = _rollout(
            deck, objective="damage", enemies=enemies_group, rng=rng, **common
        )
        _, b, _, _, _ = _rollout(deck, objective="block", enemies=enemies_solo, rng=rng, **common)
        _, _, pb, _, _ = _rollout(
            deck, objective="damage", enemies=enemies_solo, rng=rng, no_engine=True, **common
        )
        d_solo += ds
        d_group += dg
        blk += b
        plays += ps
        plays_base += pb
        draw_lim += dl
        energy_lim += el
    return {
        "damage_solo": d_solo / n,
        "damage_group": d_group / n,
        "block": blk / n,
        "plays": plays / n,
        "plays_base": plays_base / n,
        # Average turns per combat that stalled on each constraint (the binding
        # bottleneck), over the primary single-target damage rollouts.
        "draw_limited": draw_lim / n,
        "energy_limited": energy_lim / n,
    }


def _saturate(raw: float, scale: float) -> float:
    """Map a non-negative raw metric onto 0-100 with diminishing returns."""
    return round(100.0 * (1.0 - math.exp(-max(0.0, raw) / scale)), 1)


def analyze_deck(
    deck: list[Any],
    *,
    energy: int = DEFAULT_ENERGY,
    hand: int = DEFAULT_HAND,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    game_version: str | None = None,
) -> dict[str, Any]:
    """Grade ``deck`` (list of card dicts/ids) on Damage, Block and Velocity.

    Returns a JSON-serialisable report: per-axis 0-100 scores, the overall
    "volume" (geometric mean — a deck weak on one axis can't hide behind the
    others), the raw numbers behind each axis, a composition breakdown, an energy
    curve, and a list of heuristic warnings. Cards are read from game build
    ``game_version``'s tables (``None`` = the default build).
    """
    cards = [_resolved(c, game_version) for c in deck]
    n = len(cards)
    if n == 0:
        return {"empty": True}

    sim = _simulate(
        cards,
        enemies_solo=SIM_ENEMIES_SOLO,
        enemies_group=SIM_ENEMIES_GROUP,
        turns=SIM_TURNS,
        base_energy=energy,
        base_hand=hand,
        samples=samples,
        seed=seed,
    )
    dmg_solo = sim["damage_solo"]
    dmg_group = sim["damage_group"]
    dmg_blend = 0.5 * dmg_solo + 0.5 * dmg_group  # boss + hallway proxy
    blk_pt = sim["block"]
    plays = sim["plays"]
    extra_plays = max(0.0, plays - sim["plays_base"])  # plays the engines buy

    # --- role / content tallies (DISPLAY + warnings only; not score inputs) ------
    # Scaling and AOE no longer add fixed bonuses — they earn their score by
    # boosting the simulated output above. These lists just explain the numbers.
    dmg_scalers = [c for c in cards if c["damage_role"] in ("scaling", "both")]
    blk_scalers = [c for c in cards if c["block_role"] in ("scaling", "both")]
    attacks = [c for c in cards if c["type"] == "Attack"]
    aoe_attacks = [c for c in attacks if c["aoe"]]

    vel_counts: dict[str, int] = {}
    for c in cards:
        for tag in c["velocity"]:
            vel_counts[tag] = vel_counts.get(tag, 0) + 1
    vel_cards = sum(1 for c in cards if c["velocity"])

    dead = [c for c in cards if c["type"] in _DEAD_TYPES]
    basics = [
        c
        for c in cards
        if c["rarity"] == "Basic" and c["type"] in ("Attack", "Skill") and not c["upgraded"]
    ]

    # --- axis scores ------------------------------------------------------------
    # Damage and Block come straight from simulated per-turn output (scaling and
    # AOE are already baked in). Velocity is the extra card plays the deck buys
    # over the bare energy baseline, plus small credits for engine types the
    # sim can't model (extra plays, tutoring, cost cuts).
    extra_play = vel_counts.get("extra_play", 0)
    tutor = vel_counts.get("tutor", 0)
    cost_red = vel_counts.get("cost_reduction", 0)
    vel_raw = 16.0 * extra_plays + 6.0 * extra_play + 6.0 * cost_red + 5.0 * tutor
    # Dilution: the same engine buys less effective velocity in a bloated deck
    # (Reddit's `1/D`). Lean decks (≤ _LEAN_DECK) are unpenalised; larger ones
    # scale by the fraction of the deck a lean reference would process.
    dilution = min(1.0, _LEAN_DECK / n)
    vel_raw *= dilution
    damage_score = _saturate(dmg_blend, _DMG_SCALE)
    block_score = _saturate(blk_pt, _BLK_SCALE)
    velocity_score = _saturate(vel_raw, _VEL_SCALE)
    volume = round((damage_score * block_score * velocity_score) ** (1 / 3), 1)

    # Context lenses (display only): linear blends of the axis scores by fight shape.
    lenses = {
        name: round(wd * damage_score + wb * block_score + wv * velocity_score, 1)
        for name, (wd, wb, wv) in _LENSES.items()
    }

    # Binding bottleneck (Reddit's draw-vs-energy `min`): which constraint ends
    # the most turns, so the advice is "add draw" vs "add energy" — not both.
    draw_lim = sim["draw_limited"]
    energy_lim = sim["energy_limited"]
    if draw_lim < 0.25 and energy_lim < 0.25:
        bottleneck = "none"  # turns end with the hand spent and energy gone — balanced
    elif energy_lim >= draw_lim:
        bottleneck = "energy"  # cards left in hand you can't afford — want more energy
    else:
        bottleneck = "draw"  # hand empties with energy to spare — want more draw

    # --- energy curve + composition --------------------------------------------
    curve: dict[str, int] = {}
    for c in cards:
        if c["type"] in _DEAD_TYPES:  # curses/statuses are unplayable — not on the curve
            continue
        key = "X" if c["cost_x"] else str(c["cost"])
        curve[key] = curve.get(key, 0) + 1
    by_type: dict[str, int] = {}
    by_rarity: dict[str, int] = {}
    for c in cards:
        by_type[c["type"] or "?"] = by_type.get(c["type"] or "?", 0) + 1
        by_rarity[c["rarity"] or "?"] = by_rarity.get(c["rarity"] or "?", 0) + 1

    return {
        "deck_size": n,
        "assumptions": {"energy": energy, "hand": hand, "samples": samples},
        "scores": {
            "damage": damage_score,
            "block": block_score,
            "velocity": velocity_score,
            "volume": volume,
        },
        # Weighted "what matters in this fight" blends (headline stays volume).
        "lenses": lenses,
        "damage": {
            # Simulated average damage per turn (scaling + multi-hit already in).
            "per_turn": round(dmg_blend, 1),
            "single_target": round(dmg_solo, 1),  # boss proxy (1 enemy)
            "vs_group": round(dmg_group, 1),  # hallway proxy (3 enemies)
            "scaling_cards": [c["title"] for c in dmg_scalers],
            "aoe_attacks": [c["title"] for c in aoe_attacks],
            "attack_count": len(attacks),
        },
        "block": {
            # Simulated average block per turn (block engines already in).
            "per_turn": round(blk_pt, 1),
            "scaling_cards": [c["title"] for c in blk_scalers],
        },
        "velocity": {
            "tags": vel_counts,
            "engine_cards": vel_cards,
            # Simulated cards played per turn, and how many of those the deck's
            # own draw/energy engines buy over the same deck on raw energy.
            "cards_per_turn": round(plays, 1),
            "extra_per_turn": round(extra_plays, 1),
            # Deck-size dilution factor applied to the engine credit (1.0 = lean).
            "dilution": round(dilution, 2),
            # Which constraint ends the most turns: "draw", "energy", or "none".
            "bottleneck": bottleneck,
            "draw_limited_turns": round(draw_lim, 1),
            "energy_limited_turns": round(energy_lim, 1),
        },
        "composition": {
            "by_type": by_type,
            "by_rarity": by_rarity,
            "energy_curve": curve,
            "dead_cards": len(dead),
            "unupgraded_basics": len(basics),
        },
        "warnings": _warnings(
            damage_score,
            block_score,
            velocity_score,
            blk_pt,
            dmg_scalers,
            blk_scalers,
            aoe_attacks,
            attacks,
            dead,
            basics,
            n,
            bottleneck,
        ),
    }


_AXIS_KEYS = ("damage", "block", "velocity", "volume")


def reward_preview(
    deck: list[Any],
    candidates: list[Any],
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
    game_version: str | None = None,
) -> dict[str, Any]:
    """Score the deck, then the deck + each candidate card, and return the deltas.

    Powers the card-reward preview: "what does taking *this* card do to my three
    axes?" The same ``seed`` is reused for the baseline and every candidate so the
    Monte-Carlo draws are paired and a delta reflects the card, not RNG noise.

    Args:
        deck: The current master deck (list of card dicts / ids).
        candidates: Offered cards, each a dict with an ``index`` plus the usual
            ``id`` / ``upgraded`` fields.
        samples, seed, game_version: Forwarded to :func:`analyze_deck`.

    Returns:
        ``{"base": <scores>, "candidates": [{"index", "scores", "delta"}, …]}``.
    """
    zero = {k: 0.0 for k in _AXIS_KEYS}
    base = analyze_deck(deck, samples=samples, seed=seed, game_version=game_version)
    base_scores = base.get("scores") or dict(zero)
    out: list[dict[str, Any]] = []
    for cand in candidates:
        r = analyze_deck([*deck, cand], samples=samples, seed=seed, game_version=game_version)
        sc = r.get("scores") or dict(zero)
        out.append(
            {
                "index": cand.get("index") if isinstance(cand, dict) else None,
                "scores": sc,
                "delta": {k: round(sc[k] - base_scores[k], 1) for k in _AXIS_KEYS},
            }
        )
    return {"base": base_scores, "candidates": out}


def deck_signature(deck: list[Any]) -> tuple:
    """A hashable, order-independent fingerprint of a deck (card id + upgrade)."""
    return tuple(
        sorted(
            (_card_id(c), bool(c.get("upgraded")) if isinstance(c, dict) else False) for c in deck
        )
    )


def _warnings(
    dmg_s: float,
    blk_s: float,
    vel_s: float,
    blk_per_turn: float,
    dmg_scalers: list,
    blk_scalers: list,
    aoe: list,
    attacks: list,
    dead: list,
    basics: list,
    n: int,
    bottleneck: str = "none",
) -> list[dict[str, str]]:
    """Heuristic crisis flags — the highest-learning-value output.

    Each is a tunable rule of thumb, not a verdict; severity is "warn" or "info".
    """
    out: list[dict[str, str]] = []

    def add(sev: str, text: str) -> None:
        out.append({"severity": sev, "text": text})

    if attacks and not dmg_scalers:
        add("warn", "No damage scaling — you may stall out against high-HP bosses.")
    if not blk_scalers and blk_per_turn < 10:
        add("warn", "Thin on block and no block engine — big hits will land in Act 2+.")
    if attacks and not aoe:
        add("info", "No AOE — multi-enemy fights (Gremlins, hallways) will be slow.")
    if vel_s < 25:
        add("info", "Low velocity — few draw/energy cards; you'll often draw the wrong hand.")
    if bottleneck == "energy":
        add("info", "Energy-limited — cards stranded in hand; more energy helps, more draw won't.")
    elif bottleneck == "draw":
        add("info", "Draw-limited — hand empties with energy to spare; add card draw, not energy.")
    if n and (len(dead) + len(basics)) / n > 0.45:
        add("warn", "Deck clogged with basics/curses — prioritise removals.")
    if n >= 35 and vel_s < 45:
        add("info", "Large deck with low velocity — consistency suffers as it bloats.")
    if not out:
        add("info", "No structural gaps flagged — keep an eye on your next 5-20 turns.")
    return out
