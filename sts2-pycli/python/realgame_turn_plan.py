"""The turn plan: this turn's moves, in order, in the CURRENT hand's indices.

``advice.plan`` (PROTOCOL.md §5) lets the HUD number the cards to play this
turn 1, 2, 3. Both brains produce a line of engine actions -- the search its
principal variation, Instinct a greedy rollout of itself -- but every action
after the first indexes a hand that exists only after the earlier ones were
played. This module walks such a line on a copy of the fight and maps each
action back to the hand the player is looking at now, keeping only what it
can map with certainty:

* a play's ``card_index`` is translated through the engine->mod hand map of
  the request (``wire.deck.hand[i].stable_idx``), which is shifted by every
  card the line has already played;
* after every step the engine's hand must be exactly the previous hand minus
  the played card (same cards, same order). A card that draws, generates,
  discards, upgrades or reorders ends the plan AFTER itself: it is still
  certain, the next one is not;
* a step that opens a choice, wins or loses the fight ends the plan after
  itself; a move the walk cannot map (a choice, an orb verb, an index the
  map does not cover, an engine refusal) ends it before itself;
* ``end_turn`` is the last step when the line reaches it;
* the enemy roster must keep its length (``target_index`` addresses it).

Potions use the remap ``best`` uses: the engine's potion slot is dense over
the held potions, the mod's is the belt slot.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from enemy_labels import ally_label, enemy_label

logger = logging.getLogger(__name__)

#: A defensive bound on one turn's plan (a real turn ends long before).
MAX_PLAN_STEPS = 12


def _hand_key(state: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The engine hand as comparable identities (``CombatState.get_state``)."""
    return [
        (c.get("id"), c.get("base_damage"), c.get("base_block"), c.get("type"))
        for c in state.get("hand") or []
        if isinstance(c, dict)
    ]


def _roster_len(state: dict[str, Any]) -> int:
    return len(state.get("enemies") or [])


def step_label(state: dict[str, Any], step: dict[str, Any]) -> str:
    """A plan step named against the request's board (``state``: the mod's)."""
    action = step.get("action")
    if action == "end_turn":
        return "End turn"
    if action == "play":
        hand = state.get("hand") or []
        i = step.get("card_index")
        card = hand[i] if isinstance(i, int) and 0 <= i < len(hand) else {}
        label = str(card.get("name") or card.get("id") or "?")
    elif action == "potion":
        slot = step.get("slot")
        potion = next(
            (
                p
                for p in state.get("potions") or []
                if isinstance(p, dict) and p.get("slot") == slot
            ),
            {},
        )
        label = str(potion.get("name") or potion.get("id") or f"potion {slot}")
    else:
        return str(action or "?").replace("_", " ")
    if step.get("ally_index") is not None:
        target = ally_label(state.get("allies") or [], step.get("ally_index"))
    else:
        target = enemy_label(state.get("enemies") or [], step.get("target_index"))
    if target is not None:
        label += f" → {target}"
    return label


class PlanWalk:
    """Maps one line of engine actions onto the request's hand, step by step.

    Args:
        wire: The engine wire dict the fight was built from (its
            ``deck.hand[].stable_idx`` is the engine->mod hand map).
        state: The mod's ``combat_action`` payload (labels, potion slots).
    """

    def __init__(self, wire: dict[str, Any], state: dict[str, Any]) -> None:
        self.state = state
        self.hand_map: list[int | None] = []
        for c in (wire.get("deck") or {}).get("hand") or []:
            idx = c.get("stable_idx") if isinstance(c, dict) else None
            self.hand_map.append(int(idx) if isinstance(idx, int) else None)
        self.potion_map: list[int] = [
            int(p["slot"])
            for p in state.get("potions") or []
            if isinstance(p, dict) and not p.get("empty") and isinstance(p.get("slot"), int)
        ]
        self.steps: list[dict[str, Any]] = []
        self.done = False

    def to_step(self, action: dict[str, Any]) -> dict[str, Any] | None:
        """``action`` (engine combat form) as a plan step, or ``None``."""
        kind = action.get("action")
        if kind == "end_turn":
            return {"action": "end_turn"}
        step: dict[str, Any]
        if kind == "play":
            ei = action.get("card_index")
            if not isinstance(ei, int) or not 0 <= ei < len(self.hand_map):
                return None
            mod = self.hand_map[ei]
            if mod is None:
                return None
            step = {"action": "play", "card_index": mod}
        elif kind in ("potion", "drink_potion"):
            d = action.get("slot")
            if not isinstance(d, int) or not 0 <= d < len(self.potion_map):
                return None
            step = {"action": "potion", "slot": self.potion_map[d]}
        else:
            return None
        if action.get("target_index") is not None:
            step["target_index"] = int(action["target_index"])
        if action.get("ally_index") is not None:
            step["ally_index"] = int(action["ally_index"])
        return step

    def push(self, step: dict[str, Any]) -> None:
        step["label"] = step_label(self.state, step)
        self.steps.append(step)
        if step["action"] == "end_turn" or len(self.steps) >= MAX_PLAN_STEPS:
            self.done = True

    def advance(
        self,
        action: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        result: str | None = None,
    ) -> None:
        """Account for ``action`` having moved the fight ``before`` -> ``after``
        (engine ``get_state`` dicts); ends the walk unless the next step is
        still certain."""
        if result in ("pending_choice", "combat_won", "combat_lost"):
            self.done = True
            return
        if _roster_len(after) != _roster_len(before):
            self.done = True
            return
        kind = action.get("action")
        expected = _hand_key(before)
        if kind == "play":
            ei = int(action["card_index"])
            expected = expected[:ei] + expected[ei + 1 :]
            if _hand_key(after) != expected:
                self.done = True
                return
            del self.hand_map[ei]
        elif kind in ("potion", "drink_potion"):
            if _hand_key(after) != expected:
                self.done = True
                return
            del self.potion_map[int(action["slot"])]
        else:
            self.done = True


def same_move(step: dict[str, Any], best: dict[str, Any]) -> bool:
    """Whether a plan step is the move ``advice.best`` names."""
    keys = ("action", "card_index", "target_index", "ally_index", "slot")
    return all(step.get(k) == best.get(k) for k in keys)


def first_step(best: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any] | None:
    """``best`` as a plan step (when the line cannot follow it further)."""
    if not isinstance(best, dict) or best.get("action") not in ("play", "potion", "end_turn"):
        return None
    step = {
        k: best[k]
        for k in ("action", "card_index", "target_index", "ally_index", "slot")
        if best.get(k) is not None
    }
    step["label"] = step_label(state, step)
    return step


#: Wall-clock cap on the re-searches that extend a plan past its PV. A plan
#: is a drawing: once spent, the plan simply ends where it got to. A search
#: starts only when the last one's duration still fits, so the cap holds to
#: within the variance of one search.
PLAN_BUDGET_S = 0.5


def search_plan(
    cs: Any,
    pv: list[Any],
    wire: dict[str, Any],
    state: dict[str, Any],
    best: dict[str, Any] | None,
    research: Callable[[Any], list[Any]] | None = None,
    budget_s: float = PLAN_BUDGET_S,
) -> list[dict[str, Any]]:
    """The search's plan for this turn.

    Follows the principal variation, then -- the honest search's PV is one
    ply deep, because its deeper nodes keep no realized state to name an
    action against -- ``research(fight)``: a search from the fight as the
    plan has left it, returning its line (its pick first). Each step is
    applied to a clone of ``cs`` (never ``cs`` itself) and mapped/cut by
    :class:`PlanWalk`. When the PV does not open with ``best`` (the PV
    follows the highest-Q child; the engine's pick rule or the mod's veto can
    crown another), the plan is ``best`` alone.
    """
    if not isinstance(best, dict):
        return []
    lone = first_step(best, state)
    try:
        sim = cs.clone()
    except Exception as exc:
        logger.debug("plan: cannot clone the fight: %s", exc)
        return [lone] if lone else []
    walk = PlanWalk(wire, state)
    line = [a for a in pv if isinstance(a, dict)]
    t0 = time.perf_counter()
    last_search_s = 0.0
    ply = 0
    while not walk.done:
        if not line:
            if research is None or ply == 0:
                break
            if time.perf_counter() - t0 + last_search_s > budget_s:
                break
            t_search = time.perf_counter()
            try:
                line = [a for a in research(sim) if isinstance(a, dict)]
            except Exception as exc:
                logger.debug("plan: re-search failed at ply %d: %s", ply, exc)
                break
            last_search_s = time.perf_counter() - t_search
            if not line:
                break
        action = line.pop(0)
        step = walk.to_step(action)
        if step is None or (ply == 0 and not same_move(step, best)):
            break
        walk.push(step)
        if walk.done:
            break
        try:
            before = sim.get_state()
            result = sim.apply_step(dict(action))
            after = sim.get_state()
        except Exception as exc:
            logger.debug("plan: the line stopped at ply %d: %s", ply, exc)
            walk.steps.pop()
            break
        if result in ("invalid_target", "illegal_play"):
            walk.steps.pop()
            break
        walk.advance(action, before, after, result)
        ply += 1
    if not walk.steps:
        return [lone] if lone else []
    return walk.steps
