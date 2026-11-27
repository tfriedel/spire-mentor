"""Live card-pick advisor: Monte-Carlo win-rate per card-reward option.

At a card reward the engine unlocks something a tier list never can: real
grounded truth. Fork the live run once per option (take card *i*, or skip),
then play the **rest of the current act** out many times — MCTS in combat, the
hand-written :func:`run_policy.heuristic_run_action` out of combat — reseeding
the run RNG each rollout so every playout samples a different plausible future
from the same decision point. The result is an *act win-rate* per option and
the delta of the best pick over the runner-up.

This is a deliberately deep, slow analysis (many full-act simulations with
search in every fight), so the workbench only runs it on demand — never as part
of the normal step response. It requires the in-process Rust engine
(:class:`sts2_rs.RunManager`): the C# backend is a subprocess and can't be
forked cheaply.

The comparison is *paired*: at rollout ``r`` every candidate is reseeded with
the same run seed and driven by a policy RNG seeded the same way, so the only
variable between candidates is the card that was added. That variance reduction
(the same trick :func:`run_policy.boss_sim_card_reward` uses with a shared fight
seed) makes the deltas meaningful at a modest rollout budget.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import Any

from run_policy import heuristic_run_action

# A progress callback receives one event dict per emission. The advisor emits a
# single ``{"phase": "start", ...}`` (candidate list + total playout count) then
# a ``{"phase": "tick", ...}`` after every completed playout. Optional: the
# blocking ``/api/live/card-advisor`` endpoint passes ``None``; the streaming
# endpoint passes a callback that forwards each event to the client.
ProgressFn = Callable[[dict[str, Any]], None]

# Defaults tuned so a single "Analyze" press finishes in a handful of seconds on
# a typical run: enough rollouts to separate clearly different picks, light MCTS
# per fight. All overridable from the request config.
DEFAULT_ROLLOUTS = 12
DEFAULT_SIMS = 128
DEFAULT_THREADS = 8
DEFAULT_MAX_STEPS = 600
# Hard ceilings so a malformed request can't wedge the server in a giant sweep.
MAX_ROLLOUTS = 200
MAX_SIMS = 8000


def _label_for(raw_id: str) -> str:
    """``CARD.PERFECTED_STRIKE`` / ``PerfectedStrike`` → a short display token.

    Best-effort: the server enriches the real card name for display, so this
    only needs to be a stable, human-readable fallback for logs/tests."""
    stem = raw_id.removeprefix("CARD.")
    return stem


def _playout_rest_of_act(
    rm: Any,
    *,
    sims: int,
    threads: int,
    max_steps: int,
    mcts_seed: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Drive an already-forked + reseeded *rm* to the end of its current act.

    Combat decisions use ``rm.mcts_pick_action`` (or a uniform-random legal
    action when ``sims <= 0``, a fast/cheap mode); every other decision uses
    :func:`run_policy.heuristic_run_action`. Stops at: death, the act boundary
    (``act_index`` moves on — the act boss was cleared), a won run, or
    ``max_steps`` (inconclusive).

    Returns ``{"won", "died", "final_hp", "hp_at_deepest", "floors_advanced",
    "steps", "reached_act", "inconclusive"}``. *rm* is mutated (it is the
    caller's throwaway clone).

    ``floors_advanced`` and ``hp_at_deepest`` exist because binary act-win is a
    coarse, often-near-zero signal (the act boss is a wall): how *deep* a
    rollout pushed and how much HP it carried to the wall discriminate two
    cards even when neither clears the boss this rollout.
    """
    start_act = rm.act_index
    start_floor = rm.total_floor()
    deepest = start_floor
    hp_at_deepest = rm.current_hp
    steps = 0
    while steps < max_steps:
        if rm.is_over():
            break
        if rm.act_index != start_act:
            break  # cleared the act boss and entered the next act
        # Sample progress out of combat, where ``current_hp`` is the settled
        # between-rooms HP (in combat it reflects the combat-entry value).
        floor = rm.total_floor()
        if floor > deepest:
            deepest = floor
            hp_at_deepest = rm.current_hp
        phase = rm.phase()
        if phase == "combat":
            action: dict[str, Any] | None = None
            if sims > 0:
                action = rm.mcts_pick_action(sims=sims, seed=mcts_seed + steps, threads=threads)
            if action is None:
                # sims<=0 (fast mode), or MCTS declined (orb / multi-select
                # pending choice it can't express) — take a random legal move.
                legal = rm.available_actions()
                if not legal:
                    break
                action = rng.choice(legal)
        else:
            acts = rm.available_actions()
            if not acts:
                break
            action = heuristic_run_action(rm, acts, rng)
        result = rm.take_action(action)
        if result not in ("advanced", "run_over"):
            # Engine rejected the move (shouldn't happen for a legal action) —
            # bail rather than spin. Treated as inconclusive below.
            return {
                "won": False,
                "died": False,
                "final_hp": rm.current_hp,
                "hp_at_deepest": hp_at_deepest,
                "floors_advanced": deepest - start_floor,
                "steps": steps,
                "reached_act": rm.act_index,
                "inconclusive": True,
            }
        steps += 1

    over = rm.is_over()
    cleared_act = rm.act_index != start_act
    won_run = over and rm.player_won
    died = over and not rm.player_won
    inconclusive = steps >= max_steps and not over and not cleared_act
    final_floor = rm.total_floor()
    if final_floor > deepest:
        deepest = final_floor
        hp_at_deepest = rm.current_hp
    return {
        "won": bool(cleared_act or won_run),
        "died": bool(died),
        "final_hp": rm.current_hp,
        "hp_at_deepest": hp_at_deepest,
        "floors_advanced": deepest - start_floor,
        "steps": steps,
        "reached_act": rm.act_index,
        "inconclusive": inconclusive,
    }


def rank_rows(rows: list[dict[str, Any]]) -> None:
    """Mark the best row, attach each row's ``delta`` vs the best *other*, sort.

    Shared by :func:`advise` and :mod:`workbench.decision_advisor`: given rows
    that each carry a composite ``value``, set ``is_best`` on the single highest
    and ``delta`` = ``value`` minus the best *other* row's value (so the
    recommended pick reads ``+x`` over the runner-up and the rest read ``≤ 0``,
    even when act-win rates tie at zero), then sort best-first in place. No-op
    on an empty list.
    """
    if not rows:
        return
    best = max(rows, key=lambda x: x["value"])
    for row in rows:
        others = [x["value"] for x in rows if x is not row]
        best_other = max(others) if others else 0.0
        row["delta"] = row["value"] - best_other
        row["is_best"] = row is best
    rows.sort(key=lambda x: (not x["is_best"], -x["value"]))


def _candidates(rm: Any) -> list[tuple[int | None, str]]:
    """``(idx, label)`` for each evaluable option: every offered card plus skip.

    ``idx`` is the ``pick_card_reward`` option index (``None`` for skip)."""
    acts = rm.available_actions()
    picks = [a for a in acts if a.get("action") == "pick_card_reward" and a.get("idx") is not None]
    ids = rm.card_reward_option_ids()
    out: list[tuple[int | None, str]] = []
    for a in picks:
        idx = a["idx"]
        label = _label_for(ids[idx]) if idx < len(ids) else f"card {idx}"
        out.append((idx, label))
    has_skip = any(a.get("action") == "pick_card_reward" and a.get("idx") is None for a in acts)
    if has_skip:
        out.append((None, "skip"))
    return out


def is_advisable(rm: Any) -> bool:
    """True when *rm* sits on a plain card-pick reward the advisor can evaluate.

    Excludes relic rewards, sacrifice rewards and Scroll-Boxes bundle picks
    (no per-card take/skip fan-out to compare), mirroring the gating in
    :func:`run_policy.boss_sim_card_reward`."""
    if rm is None or rm.phase() != "card_reward":
        return False
    if rm.card_reward_bundles() is not None:
        return False
    acts = rm.available_actions()
    if any(a.get("action") == "pick_relic_reward" for a in acts):
        return False
    return any(a.get("action") == "pick_card_reward" and a.get("idx") is not None for a in acts)


def advise(
    rm: Any,
    config: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Monte-Carlo act win-rate for each card-reward option of live run *rm*.

    Args:
        rm: An ``sts2_rs.RunManager`` parked on a card-reward decision. It is
            never mutated — every rollout runs on a deep clone.
        config: Optional overrides: ``rollouts``, ``sims`` (0 = random combat,
            fast), ``threads``, ``max_steps``, ``seed``.
        progress: Optional callback invoked once with a ``{"phase": "start"}``
            event (candidate list + ``total`` playout count) and then after
            every completed playout with a ``{"phase": "tick"}`` event. Used by
            the streaming endpoint to drive a live progress bar; ``None`` for
            the plain blocking call.

    Returns:
        On success::

            {"candidates": [{"idx", "label", "win_rate", "wins", "n",
                             "avg_final_hp", "delta", "is_best",
                             "inconclusive"}],
             "config": {...}, "act": int}

        ``delta`` is each option's win-rate minus the best *other* option's
        (so the recommended pick reads ``+x`` over the runner-up and the rest
        read ``≤ 0``). On a non-card-reward / unforkable state, or if no
        candidate is buildable, ``{"error": str, "candidates": [], ...}``.
    """
    config = dict(config or {})
    if not is_advisable(rm):
        return {
            "error": "card advisor only runs on a plain card-reward decision",
            "candidates": [],
            "config": config,
        }

    rollouts = max(1, min(int(config.get("rollouts", DEFAULT_ROLLOUTS)), MAX_ROLLOUTS))
    sims = min(int(config.get("sims", DEFAULT_SIMS)), MAX_SIMS)
    threads = int(config.get("threads", DEFAULT_THREADS))
    max_steps = int(config.get("max_steps", DEFAULT_MAX_STEPS))
    base_seed = int(config.get("seed", (rm.seed & 0x7FFF_FFFF) ^ 0x5EED))
    start_act = rm.act_index

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
                "candidates": [{"idx": idx, "label": label} for idx, label in candidates],
            }
        )

    def _tick(candidate_idx: int | None) -> None:
        nonlocal done_playouts
        done_playouts += 1
        if progress is not None:
            progress(
                {
                    "phase": "tick",
                    "done": done_playouts,
                    "total": total_playouts,
                    "candidate_idx": candidate_idx,
                }
            )

    rows: list[dict[str, Any]] = []
    for idx, label in candidates:
        wins = 0
        inconclusive = 0
        hp_total = 0
        floors_total = 0
        hp_deep_total = 0
        for r in range(rollouts):
            sim = copy.deepcopy(rm)
            res = sim.take_action({"action": "pick_card_reward", "idx": idx})
            if res not in ("advanced", "run_over"):
                inconclusive += 1
                _tick(idx)
                continue
            # The rollout plays as the agent: no event option that would kill.
            if hasattr(sim, "offer_lethal_event_options"):
                sim.offer_lethal_event_options = False
            # Paired across candidates: same run seed + policy seed per rollout
            # r, so the only difference is the card added on this branch.
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
            _tick(idx)
        n = rollouts
        win_rate = wins / n if n else 0.0
        avg_floors = floors_total / n if n else 0.0
        avg_hp_deep = hp_deep_total / n if n else 0.0
        # Composite value used for ranking + deltas. Act-clear dominates (each
        # clear is worth a full "act"), with depth reached and HP carried to
        # the wall as fine-grained tiebreakers so two cards still separate when
        # neither clears the boss. Floors/HP are scaled small so they only
        # break near-ties in win-rate, never override a real win-rate gap.
        value = win_rate + 0.01 * avg_floors + 0.02 * (avg_hp_deep / max_hp)
        rows.append(
            {
                "idx": idx,
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

    # Delta vs the best *other* candidate, measured on the composite value (so
    # the recommended pick reads ``+x`` over the runner-up even when act-win
    # rates tie at zero). Mark the winner.
    rank_rows(rows)

    return {
        "candidates": rows,
        "act": start_act,
        "config": {
            "rollouts": rollouts,
            "sims": sims,
            "threads": threads,
            "max_steps": max_steps,
            "seed": base_seed,
        },
    }
