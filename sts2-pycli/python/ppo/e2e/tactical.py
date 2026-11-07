"""Tactical columns: what this turn can still do (WP3, plan §4).

Three numbers a combat net would otherwise have to add up itself, from the
honest snapshot only (`combat_ctx.snapshot.honest_snapshot`, so they are a
function of what the player sees):

* per enemy, `max_damage`: the most damage this turn's hand can deal to that
  enemy under the current energy. A 0/1 knapsack over the playable hand
  cards' `card_damage_preview()` numbers (per hit, with Strength, Vulnerable
  and the like already applied), times each card's hit count from the
  content prior; an X-cost card spends whatever energy is left. A random-
  target card counts all its hits on the enemy (an upper bound).
* per enemy, `killable`: `max_damage >= hp + block`.
* `hp_loss`: the HP the player loses if the turn ended now, the displayed
  attack intents of every living enemy (`effective_amount x times`) less the
  player's block.

It ignores draw, energy gain and every other effect of the cards it plays, so
`max_damage` is a lower bound on a combo turn and an upper bound when a random
hit can miss its enemy. Pure Python: about 5-30 us on a typical hand.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import content_prior as P
from . import vocab as V

# Divisors that put the columns near [0, 1] (the net applies symlog anyway).
DMG_SCALE = 50.0


@lru_cache(maxsize=4096)
def hits_of(card_id: str) -> int:
    """A card's printed hit count (1 if it has no repeat), from the prior."""
    vid = V.load().id("card", card_id)
    h = float(P.table()[vid, P.C["HITS"]]) * 5.0
    return max(1, round(h))


def _attack_total(i: dict[str, Any]) -> float:
    if i.get("kind") not in ("Attack", "DeathBlow"):
        return 0.0
    eff = i.get("effective_amount", i.get("amount", 0)) or 0
    return float(eff) * float(i.get("times") or 1)


def hp_loss(snap: dict[str, Any], intents: list[list[dict[str, Any]]] | None = None) -> float:
    """Unblocked incoming attack damage if the turn ended now.

    `intents` (aligned with `snap["enemies"]`) replaces the snapshot's
    displayed intents, e.g. with the full telegraph once the getter exists.
    """
    inc = 0.0
    for j, m in enumerate(snap["enemies"]):
        if m.get("is_dead"):
            continue
        ints = intents[j] if intents is not None else m.get("intents") or []
        inc += sum(_attack_total(i) for i in ints)
    return max(0.0, inc - float(snap["player"].get("block") or 0))


def max_damage(snap: dict[str, Any], playable: set[int]) -> list[float]:
    """Per enemy of `snap["enemies"]`, the knapsack damage bound (see module doc)."""
    n = len(snap["enemies"])
    energy = max(0, int(snap["player"].get("energy") or 0))
    fixed = [0.0] * n  # zero-cost cards: always played
    items: list[tuple[int, list[float]]] = []  # (cost, damage per enemy)
    xcards: list[list[float]] = []  # X cost: damage per enemy per energy
    for i, c in enumerate(snap["hand"]):
        if i not in playable:
            continue
        dmg = c.get("dmg")
        if not dmg:
            continue
        h = hits_of(c["id"])
        row = [0.0] * n
        for k, v in dmg.items():
            j = int(k)
            if 0 <= j < n and v > 0:
                row[j] = float(v) * h
        if not any(row):
            continue
        if int(c.get("base_cost") or 0) < 0:
            xcards.append([x / h for x in row])  # per hit; hits = energy spent
            continue
        cost = max(0, int(c.get("cost") or 0))
        if cost == 0:
            fixed = [a + b for a, b in zip(fixed, row, strict=True)]
        elif cost <= energy:
            items.append((cost, row))
    out = []
    for j in range(n):
        dp = [0.0] * (energy + 1)
        for cost, row in items:
            v = row[j]
            if v <= 0.0:
                continue
            for e in range(energy, cost - 1, -1):
                t = dp[e - cost] + v
                if t > dp[e]:
                    dp[e] = t
        best = dp[energy]
        for xc in xcards:
            per = xc[j]
            if per > 0.0:
                for e in range(energy + 1):
                    best = max(best, dp[e] + per * (energy - e))
        out.append(fixed[j] + best)
    return out


def playable_from(actions: list[dict[str, Any]]) -> set[int]:
    """Hand indices some legal `play` action plays (combat action dicts)."""
    return {int(a["card_index"]) for a in actions if a.get("action") == "play"}


def tactical(
    snap: dict[str, Any],
    actions: list[dict[str, Any]],
    intents: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[float], list[float], float]:
    """`(max_damage per enemy, killable per enemy, hp_loss)`, unscaled."""
    md = max_damage(snap, playable_from(actions))
    kill = []
    for j, m in enumerate(snap["enemies"]):
        alive = not m.get("is_dead")
        ehp = float(m.get("hp") or 0) + float(m.get("block") or 0)
        kill.append(float(alive and ehp > 0 and md[j] >= ehp))
    return md, kill, hp_loss(snap, intents)
