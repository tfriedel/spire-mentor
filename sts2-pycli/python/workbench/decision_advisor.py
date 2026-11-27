"""Generic decision advisor: Monte-Carlo win-rate per option for *any* branching
run decision the live engine can fork — in-combat card selects (Knowledge
Daemon's pick, Headbutt's put-back, …), run-level *choose a card* screens
(Lead Paperweight's 2 Colorless, Hefty Tablet's 3 Rares), shops, and events /
treasure / boss relics.

This is the sibling of :mod:`workbench.card_advisor`, which is specialised to
the card-*reward* screen (with a per-card UI). Both
share the same engine trick: fork the live run once per option, take that option
on the clone, then play the **rest of the current act** out many times — MCTS in
combat, the hand-written :func:`run_policy.heuristic_run_action` out of combat —
reseeding the run RNG each rollout. The result is an *act win-rate* per option
plus a composite value (act-clear dominates; floors reached and HP carried to
the wall break near-ties), and the delta of the best pick over the runner-up.

Where card_advisor keys options by the numeric ``pick_card_reward`` idx, this
module keys by a **string** ``key`` (``"choice:0"`` / ``"card:1"`` /
``"relic:0"`` / ``"option:2"`` / ``"leave"`` / ``"choice:skip"``) because a
single decision can mix index spaces — a shop offers cards *and* relics *and*
potions, each numbered from zero. The frontend joins its rendered options to the
result rows by that same key.

Requires the in-process Rust engine (:class:`sts2_rs.RunManager`): the C#
backend is a subprocess and can't be forked cheaply.

Caveat (honest about the signal): :meth:`RunManager.reseed_run` reseeds the
*run-level* RNG (map / rewards / shop stock / encounters), not the active
combat's RNG. So for an **in-combat** choice the current fight resolves nearly
deterministically across rollouts (variance comes from later fights + MCTS seed
exploration); the comparison stays directionally meaningful because it is paired
(same seed, only the chosen option differs) and the floors/HP composite still
separates options when neither clears the act this rollout.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from sts2_hint import display_maps

from workbench.card_advisor import (
    DEFAULT_MAX_STEPS,
    DEFAULT_ROLLOUTS,
    DEFAULT_SIMS,
    DEFAULT_THREADS,
    MAX_ROLLOUTS,
    MAX_SIMS,
    ProgressFn,
    _playout_rest_of_act,
    rank_rows,
)

# Phases (``RunManager.phase()``) whose decision is a discrete fan-out this
# advisor can evaluate. ``card_reward`` is deliberately excluded — a plain
# reward has its own richer advisor (:mod:`workbench.card_advisor`) — EXCEPT
# for a run-level choose screen, which the host renders as a ``card_select``
# and therefore routes here (see :func:`_choose_screen`). ``combat`` is
# included only when an in-combat pending choice is open (both handled in
# :func:`is_advisable`).
_OUT_OF_COMBAT_PHASES = frozenset({"shop", "event", "treasure", "boss_relic"})

# ``phase()`` → the frontend ``decision`` string, so the UI can confirm the
# advice belongs to the screen it is showing.
_PHASE_DECISION = {
    "shop": "shop",
    "event": "event_choice",
    "treasure": "event_choice",
    "boss_relic": "event_choice",
}


def _pending_choice(rm: Any) -> dict[str, Any] | None:
    """The open in-combat ``pending_choice`` dict, or ``None`` when none/out of combat."""
    getter = getattr(rm, "combat_state", None)
    cs = getter() if callable(getter) else None
    if cs is None:
        return None
    pc_getter = getattr(cs, "pending_choice", None)
    return pc_getter() if callable(pc_getter) else None


def _choose_screen(rm: Any) -> bool:
    """True on a run-level *choose a card* screen (C# ``FromChooseACardScreen``).

    Lead Paperweight's 2 Colorless and Hefty Tablet's 3 Rares are offered
    through the ``card_reward`` phase's ``pick_card_reward`` fan-out, but the
    host renders them as a skippable 1-of-N ``card_select`` (min 0 / max 1) —
    so the UI asks *this* advisor for them, not :mod:`workbench.card_advisor`
    (whose gate is the ``card_reward`` decision). Without this the screen fell
    between the two advisors and its "analyze options" button could only ever
    answer that no advisable decision was open (GH #34's decision shape,
    advisor side).

    A plain post-combat reward and a Scroll Boxes bundle pick never set the
    engine's marker, so both stay with the card advisor.
    """
    return rm.phase() == "card_reward" and bool(getattr(rm, "card_offer_choose_screen", False))


def _choose_screen_candidates(rm: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """``(key, label, action)`` per option of a run-level choose screen.

    Keyed ``choice:{idx}`` / ``choice:skip`` — the ``card_select`` key space the
    frontend joins its rendered cards and its Skip button on — while the actions
    stay the engine's own ``pick_card_reward`` picks. ``idx`` is the position in
    ``card_reward_option_ids()``, which is exactly the index the host renders
    each card under, so the chips land on the right card.
    """
    ids = rm.card_reward_option_ids()
    out: list[tuple[str, str, dict[str, Any]]] = []
    for action in rm.available_actions():
        if action.get("action") != "pick_card_reward":
            continue
        idx = action.get("idx")
        if idx is None:
            # ``canSkip: true`` on the C# screen — taking nothing is a real
            # option (deck dilution), so it gets a row like any other.
            out.append(("choice:skip", "skip", action))
            continue
        label = display_maps.loc_id(ids[idx]) if idx < len(ids) else f"card {idx}"
        out.append((f"choice:{idx}", label, action))
    return out


def _candidate_key_label(action: dict[str, Any]) -> tuple[str, str] | None:
    """Map a raw engine action dict to a stable ``(key, label)``, or ``None``.

    ``None`` means "not an option this advisor evaluates" (e.g. a multi-pick
    ``combat_resolve_choice_take_first`` prefix, or a verb from a phase we don't
    advise). ``pick_card_reward`` is one of them: a plain card reward belongs to
    :mod:`workbench.card_advisor`, and the one card-reward fan-out this advisor
    *does* own — a choose screen — needs the RunManager to key its options, so
    it is built by :func:`_choose_screen_candidates` instead. The ``key`` is
    what the frontend joins on; ``label`` is a best-effort human token for logs
    / the progress event (the UI shows the real enriched name next to each
    option).
    """
    name = action.get("action")
    if name == "combat_resolve_choice":
        opt = action.get("option")
        if opt is None:
            return ("choice:skip", "skip")
        return (f"choice:{opt}", f"option {opt}")
    if name == "shop_buy_card":
        return (f"card:{action['idx']}", f"buy card {action['idx']}")
    if name == "shop_buy_relic":
        return (f"relic:{action['idx']}", f"buy relic {action['idx']}")
    if name == "shop_buy_potion":
        return (f"potion:{action['idx']}", f"buy potion {action['idx']}")
    if name == "leave_shop":
        return ("leave", "leave")
    if name == "event_choice":
        return (f"option:{action['idx']}", f"option {action['idx']}")
    if name == "pick_boss_relic":
        return (f"boss_relic:{action['idx']}", f"relic {action['idx']}")
    return None


def _candidates(rm: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """``(key, label, action)`` for each evaluable option of the current decision.

    Built straight off ``rm.available_actions()`` so the candidate set is
    exactly the engine's legal moves; options the advisor can't model (multi-pick
    prefixes) are dropped via :func:`_candidate_key_label` returning ``None``.

    In-combat choices get a nicer label from the pending-choice card ids when
    available (``"Disintegration"`` instead of ``"option 0"``); a run-level
    choose screen is built by :func:`_choose_screen_candidates`.
    """
    if _choose_screen(rm):
        return _choose_screen_candidates(rm)
    out: list[tuple[str, str, dict[str, Any]]] = []
    pc = _pending_choice(rm)
    choice_cards = (pc.get("cards") if pc else None) or []
    for action in rm.available_actions():
        kl = _candidate_key_label(action)
        if kl is None:
            continue
        key, label = kl
        # Prettier in-combat label from the chosen card's id when resolvable.
        if action.get("action") == "combat_resolve_choice":
            opt = action.get("option")
            if opt is not None and 0 <= opt < len(choice_cards):
                cid = choice_cards[opt].get("id")
                if cid:
                    label = display_maps.loc_id(cid)
        out.append((key, label, action))
    return out


def is_advisable(rm: Any) -> bool:
    """True when *rm* sits on a discrete branching decision this advisor evaluates.

    Covers shops, events / treasure / boss-relic picks, run-level choose screens
    (Lead Paperweight / Hefty Tablet — see :func:`_choose_screen`), and
    single-pick in-combat pending choices — anything with at least two evaluable
    options. Excludes plain card rewards (own advisor), the map, plain combat
    (use MCTS hints), and multi-pick choices (no per-option take/compare
    fan-out).
    """
    if rm is None:
        return False
    phase = rm.phase()
    if phase == "combat":
        pc = _pending_choice(rm)
        if not pc or int(pc.get("max_select", 1)) > 1:
            return False
    elif phase == "card_reward":
        if not _choose_screen(rm):
            return False
    elif phase not in _OUT_OF_COMBAT_PHASES:
        return False
    return len(_candidates(rm)) >= 2


def _decision_of(rm: Any) -> str:
    """The frontend ``decision`` string for *rm*'s current phase."""
    phase = rm.phase()
    # A choose screen is a ``card_select`` in the host's decision vocabulary
    # even though its phase is ``card_reward`` (GH #34) — echo what the UI
    # shows, so the advice can be matched to the screen it belongs to.
    if phase == "combat" or _choose_screen(rm):
        return "card_select"
    return _PHASE_DECISION.get(phase, phase)


def advise(
    rm: Any,
    config: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Monte-Carlo act win-rate for each option of live run *rm*'s decision.

    Args:
        rm: An ``sts2_rs.RunManager`` parked on a shop / event / in-combat
            choice. Never mutated — every rollout runs on a deep clone.
        config: Optional overrides: ``rollouts``, ``sims`` (0 = random combat,
            fast), ``threads``, ``max_steps``, ``seed``.
        progress: Optional callback — one ``{"phase": "start", ...}`` (candidate
            list + total playout count) then one ``{"phase": "tick", ...}`` per
            completed playout. Drives the streaming progress bar; ``None`` for
            the plain blocking call.

    Returns:
        On success::

            {"candidates": [{"key", "label", "win_rate", "wins", "n",
                             "avg_final_hp", "value", "delta", "is_best",
                             "inconclusive"}],
             "decision": str, "act": int, "config": {...}}

        ``delta`` is each option's composite value minus the best *other*
        option's. On a non-advisable / unforkable state, or if no candidate is
        buildable, ``{"error": str, "candidates": [], ...}``.
    """
    config = dict(config or {})
    if not is_advisable(rm):
        return {
            "error": "the decision advisor only runs on a shop, event, or "
            "card-select decision with at least two options",
            "candidates": [],
            "config": config,
        }

    rollouts = max(1, min(int(config.get("rollouts", DEFAULT_ROLLOUTS)), MAX_ROLLOUTS))
    sims = min(int(config.get("sims", DEFAULT_SIMS)), MAX_SIMS)
    threads = int(config.get("threads", DEFAULT_THREADS))
    max_steps = int(config.get("max_steps", DEFAULT_MAX_STEPS))
    base_seed = int(config.get("seed", (rm.seed & 0x7FFF_FFFF) ^ 0x5EED))
    start_act = rm.act_index
    decision = _decision_of(rm)
    max_hp = max(1, rm.max_hp)
    candidates = _candidates(rm)

    total_playouts = len(candidates) * rollouts
    done_playouts = 0
    if progress is not None:
        progress(
            {
                "phase": "start",
                "total": total_playouts,
                "rollouts": rollouts,
                "act": start_act,
                "decision": decision,
                "candidates": [{"key": key, "label": label} for key, label, _ in candidates],
            }
        )

    def _tick(candidate_key: str) -> None:
        nonlocal done_playouts
        done_playouts += 1
        if progress is not None:
            progress(
                {
                    "phase": "tick",
                    "done": done_playouts,
                    "total": total_playouts,
                    "candidate_key": candidate_key,
                }
            )

    rows: list[dict[str, Any]] = []
    for key, label, action in candidates:
        wins = 0
        inconclusive = 0
        hp_total = 0
        floors_total = 0
        hp_deep_total = 0
        for r in range(rollouts):
            sim = copy.deepcopy(rm)
            res = sim.take_action(action)
            if res not in ("advanced", "run_over"):
                inconclusive += 1
                _tick(key)
                continue
            # The human's pick is taken as shown; the rollout after it plays
            # as the agent, without the event options that would kill.
            if hasattr(sim, "offer_lethal_event_options"):
                sim.offer_lethal_event_options = False
            # Paired across candidates: same run seed + policy seed per rollout
            # r, so the only difference is the option taken on this branch.
            sim.reseed_run(base_seed + r)
            out = _playout_rest_of_act(
                sim,
                sims=sims,
                threads=threads,
                max_steps=max_steps,
                mcts_seed=base_seed + r * 1009,
                rng=random.Random(base_seed + r),
            )
            if out["inconclusive"]:
                inconclusive += 1
            if out["won"]:
                wins += 1
            hp_total += out["final_hp"]
            floors_total += out["floors_advanced"]
            hp_deep_total += out["hp_at_deepest"]
            _tick(key)
        n = rollouts
        win_rate = wins / n if n else 0.0
        avg_floors = floors_total / n if n else 0.0
        avg_hp_deep = hp_deep_total / n if n else 0.0
        # Composite value (identical weighting to card_advisor): act-clear
        # dominates, with depth reached and HP carried to the wall as
        # fine-grained tiebreakers so two options still separate when neither
        # clears the act this rollout.
        value = win_rate + 0.01 * avg_floors + 0.02 * (avg_hp_deep / max_hp)
        rows.append(
            {
                "key": key,
                "label": label,
                "win_rate": win_rate,
                "wins": wins,
                "n": n,
                "avg_final_hp": hp_total / n if n else 0.0,
                "avg_floors_advanced": avg_floors,
                "avg_hp_at_deepest": avg_hp_deep,
                "value": value,
                "inconclusive": inconclusive,
            }
        )

    rank_rows(rows)

    return {
        "candidates": rows,
        "decision": decision,
        "act": start_act,
        "config": {
            "rollouts": rollouts,
            "sims": sims,
            "threads": threads,
            "max_steps": max_steps,
            "seed": base_seed,
        },
    }
