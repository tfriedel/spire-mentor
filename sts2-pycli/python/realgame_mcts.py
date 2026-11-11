"""MCTS-driven play driver for the real-game "SpireMentor" mod (real-game bridge phase 2).

Same session loop as :mod:`realgame_random` (reuses
:class:`realgame_bridge.RealGameBridgeClient`, JSONL logging via
:class:`realgame_random.JsonlLogger`, and every non-combat random-but-legal
policy from :mod:`realgame_random` verbatim), but every ``combat_action``
decision is answered by the real MCTS recommender
(:func:`sts2_hint.recommender.recommend_all`) instead of a random policy:

1. Translate the bridge's ``combat_action`` snapshot to the headless
   ``combat_play`` shape via :func:`realgame_adapter.to_headless_combat_play`.
2. Run MCTS (:func:`sts2_hint.recommender.recommend_all`) and take the
   top-ranked :class:`~sts2_hint.recommender.HintTag`.
3. Convert it back to a bridge action (:func:`hint_tag_to_action`) --
   ``play``/``end_turn``/``potion``.

**Never-crash-the-session policy**: any exception anywhere in that
pipeline (unmapped card/relic/power, ``sts2_rs`` raising
``UnsupportedStateError``, a best action with no bridge-verb mapping such
as Defect's channel/evoke -- see :func:`hint_tag_to_action`) is caught,
logged loudly (``logger.error``), and the decision falls back to
:func:`realgame_random.pick_combat_action` for that one step. The failure
is additionally recorded on that step's JSONL line as
``{"mcts_error": "..."}`` (see :meth:`realgame_random.JsonlLogger.log_step`'s
``extra`` parameter) so failures are visible in the corpus without ever
stalling or crashing the run.

Usage::

    uv run python python/realgame_mcts.py [--host 127.0.0.1] [--port 9002]
        [--log-dir logs/realgame] [--max-steps N] [--seed N]
        [--character IRONCLAD] [--ascension 0]
        [--sims 300] [--depth 200] [--c-puct 1.4] [--mcts-seed 0]
        [--rollout-policy heuristic]

``--character``/``--ascension`` only matter if the mod's
``client_run_config`` flag is enabled (see :mod:`realgame_random`'s module
docstring for the ``run_config`` handshake). ``--seed`` seeds the
random-policy fallback RNG *and* the ``start_run`` seed; ``--mcts-seed``
separately seeds the MCTS rollouts (default 0, matching
:class:`sts2_hint.recommender.RecommendConfig`'s own default).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from combat_mirror import MirrorDivergence, recognize_card_select, skip_arity_from_world
from realgame_adapter import (
    UnsupportedGameVersionError,
    character_display_name,
    install_live_unlock_state,
    install_unlock_state_from_wire,
    run_ascension,
    to_headless_combat_play,
)
from realgame_bridge import DEFAULT_HOST, DEFAULT_PORT, RealGameBridgeClient
from realgame_random import (
    _TERMINAL_TYPES,
    JsonlLogger,
    _progress_line,
    choose_action,
    pick_combat_action,
)
from sts2_hint.recommender import (
    CHANNEL_STABLE_IDX,
    END_TURN_STABLE_IDX,
    POTION_STABLE_IDX_BASE,
    HintTag,
    RecommendConfig,
    recommend_all,
)

logger = logging.getLogger(__name__)

#: Map-node ``type`` string (see ``map_select`` in PROTOCOL.md) that marks the
#: act boss. The bridge's ``combat_action`` snapshot carries no room-type flag,
#: so boss combats are recognised by remembering the type of the map node we
#: last walked into (see :func:`chosen_map_node_type` / :func:`combat_sims_for`).
BOSS_NODE_TYPE = "Boss"

#: Per-encounter MCTS simulation budgets, mirroring the native run agent the
#: PPO champions were trained/evaluated against (``ppo/env.py`` -> the combat
#: MCTS agent at ``sims=1000, boss_sims=4000``). Elites use the
#: regular budget there too (only the boss is special), so we match that.
DEFAULT_SIMS_REGULAR = 1000
DEFAULT_SIMS_BOSS = 4000


def chosen_map_node_type(state: dict[str, Any], action: dict[str, Any]) -> str | None:
    """Room ``type`` of the map node just chosen on a ``map_select`` decision.

    Returns the node ``type`` string (``"Monster"``/``"Elite"``/``"Boss"``/
    ``"RestSite"``/...) for the reachable node the ``action`` selected, or
    ``None`` when ``state`` isn't a ``map_select`` or the choice can't be
    resolved. Used to remember what kind of room the next combat belongs to,
    since ``combat_action`` snapshots carry no room-type field of their own.
    """
    if state.get("type") != "map_select":
        return None
    idx = action.get("index")
    if idx is None:
        return None
    nodes = state.get("nodes") or []
    for n in nodes:
        if n.get("index") == idx:
            return n.get("type")
    if 0 <= idx < len(nodes):
        return nodes[idx].get("type")
    return None


def combat_sims_for(
    current_room_type: str | None,
    sims_regular: int,
    sims_boss: int,
    sims_flat: int | None,
) -> int:
    """Pick the MCTS sim budget for a combat decision.

    ``sims_flat`` (the explicit ``--sims`` override) wins when set; otherwise
    boss rooms get ``sims_boss`` and everything else ``sims_regular`` --
    matching ``ppo/env.py``'s combat-agent ``(sims=, boss_sims=)`` schedule.
    """
    if sims_flat is not None:
        return sims_flat
    return sims_boss if current_room_type == BOSS_NODE_TYPE else sims_regular


def hint_tag_to_action(tag: HintTag) -> dict[str, Any] | None:
    """Convert one MCTS-best :class:`HintTag` into a bridge action dict.

    Returns:
        The bridge action dict, or ``None`` if this ``HintTag`` has no
        corresponding bridge verb (``PROTOCOL.md`` section 4's combat
        verbs are only ``play``/``end_turn``/``potion`` -- there is no
        wire verb for Defect's ``channel``/``evoke`` sentinels or the
        ``choose`` sentinel, which never legitimately fires for a clean
        ``combat_action`` decision anyway since
        :mod:`sts2_hint.adapter` rejects any mid-resolution
        ``pending_choice`` state before MCTS ever runs). Callers should
        treat ``None`` the same as any other translation failure and
        fall back to the random-legal policy.
    """
    if tag.stable_idx == END_TURN_STABLE_IDX:
        return {"action": "end_turn"}
    if tag.stable_idx >= 0:
        action: dict[str, Any] = {"action": "play", "card_index": tag.stable_idx}
        if tag.target_index is not None:
            action["target_index"] = tag.target_index
        if tag.ally_index is not None:
            action["ally_index"] = tag.ally_index
        return action
    if CHANNEL_STABLE_IDX < tag.stable_idx <= POTION_STABLE_IDX_BASE:
        slot = POTION_STABLE_IDX_BASE - tag.stable_idx
        action = {"action": "potion", "slot": slot}
        if tag.target_index is not None:
            action["target_index"] = tag.target_index
        if tag.ally_index is not None:
            # Thrown at that teammate (GH #1017).
            action["ally_index"] = tag.ally_index
        return action
    return None


def build_annotations(
    tags: list[HintTag],
    cfg: RecommendConfig,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the PROTOCOL.md ``annotations`` overlay payload from MCTS tags.

    Per-card scores are the max ``win_prob`` across that card's target
    variants (``recommend_all`` emits one tag per (card, target) pair);
    the ``note`` line summarizes the best action. Purely visual — the mod's
    HUD renders it and it never affects gameplay.

    When ``state`` (the bridge ``combat_action`` snapshot) is given, the
    payload additionally carries the rich advice fields (``p_win``,
    ``options[]`` with labels/Q/visit shares, ``low_confidence`` — see
    PROTOCOL.md section 5) so the HUD's win bar and ranked-move panel light
    up during autopilot runs too. Without ``state`` only the legacy
    ``note``/``cards`` shape is emitted (labels need card/enemy names).
    """
    per_card: dict[int, float] = {}
    for tag in tags:
        if tag.stable_idx >= 0:
            prev = per_card.get(tag.stable_idx)
            if prev is None or tag.win_prob > prev:
                per_card[tag.stable_idx] = tag.win_prob
    if not tags:
        return None
    best = tags[0]
    note = f"MCTS p_win {best.win_prob:.2f} ({cfg.sims} sims)"
    cards = [{"index": i, "score": p, "label": f"{p:.0%}"} for i, p in sorted(per_card.items())]
    out: dict[str, Any] = {"note": note, "cards": cards}
    if state is not None:
        from realgame_advisor import build_advice

        rich = build_advice(tags, state, sims=cfg.sims)
        rich.pop("note", None)  # keep the legacy note line
        rich.pop("best", None)  # annotations are display-only; no executable
        out.update(rich)
    return out


def _remap_potion_slot(action: dict[str, Any], state: dict[str, Any]) -> None:
    """Rewrite a ``potion`` action's ``slot`` from engine index to bridge slot.

    The sts2-rs engine COMPACTS potions to a dense 0-based vector (empty slots
    dropped), so ``hint_tag_to_action`` produces the potion's *dense* index --
    e.g. index 0 for a lone Skill Potion sitting at the bridge's absolute slot
    2. The bridge's ``potion`` verb wants the ABSOLUTE slot (``SafeGetPotionAtSlot``),
    so sending the dense index hits an empty slot -> ``empty_potion_slot`` error
    and the AI can never use its potions. Map dense index i -> the absolute
    ``slot`` of the i-th non-empty potion in ``state["potions"]`` (same order the
    wire/engine compaction preserves). Mutates ``action`` in place; leaves it
    untouched if it isn't a potion action or the index can't be resolved.
    """
    if action.get("action") != "potion":
        return
    dense_idx = action.get("slot")
    if not isinstance(dense_idx, int):
        return
    non_empty = [p for p in (state.get("potions") or []) if not p.get("empty")]
    if 0 <= dense_idx < len(non_empty):
        abs_slot = non_empty[dense_idx].get("slot")
        if isinstance(abs_slot, int):
            action["slot"] = abs_slot


def _wire_unplayable(action: dict[str, Any], state: dict[str, Any]) -> str | None:
    """Reason the real game will reject this action, or ``None`` if it looks fine.

    Only the ``play`` verb is checked: the bridge hand carries the
    authoritative ``CardModel.CanPlay()`` result as ``hand[].playable``
    (PROTOCOL.md -- present on ``combat_action`` only). An absent flag (older
    mod build) or an out-of-range index returns ``None`` -- true index bugs
    still surface as the mod's structured ``invalid_index`` error.
    """
    if action.get("action") != "play":
        return None
    hand = state.get("hand") or []
    idx = action.get("card_index")
    if not isinstance(idx, int) or not 0 <= idx < len(hand):
        return None
    if hand[idx].get("playable") is False:
        return f"{hand[idx].get('id')}: playable=false on the wire"
    return None


def pick_combat_action_mcts(
    state: dict[str, Any],
    cfg: RecommendConfig,
    rng: random.Random,
    *,
    sims: int | None = None,
    mirror: Any | None = None,
    banned: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Pick a combat action via MCTS, falling back to random-legal on any failure.

    Args:
        state: The bridge ``combat_action`` decision message.
        cfg: MCTS knobs (sims/depth/...); ``character`` is overridden
            per-call from ``state["run_state"]["character"]`` when
            mappable.
        rng: Seeded RNG for the random-legal fallback policy
            (:func:`realgame_random.pick_combat_action`).
        sims: Per-decision override for ``cfg.sims`` (the MCTS simulation
            budget for this one decision). ``None`` uses ``cfg.sims`` as-is;
            callers pass the boss/regular budget from :func:`combat_sims_for`.
        mirror: The fight's persistent :class:`combat_mirror.CombatMirror`.
            When given, the state is reconciled against it and the search runs
            on a CLONE of it, so the search sees the RNG cursors and draw-pile
            pins a per-decision reconstruction throws away — and the mirror
            ends the decision parked on whatever choice the played card
            opened, which is what makes the following ``card_select``
            answerable at all.
        banned: Sort-keyed JSON of bridge actions the game already REJECTED
            for this exact decision state (an ``error`` message followed by a
            re-send — see PROTOCOL.md 3a). Same contract as headless_eval's
            ``banned_for_state``: the ranking walk skips them so the search's
            next-best action gets a try instead of a deterministic policy
            re-sending the rejected play forever. Compared AFTER the potion
            slot remap and WITHOUT ``annotations``, i.e. against the action
            exactly as the caller sent it.

    Returns:
        ``(action, log_extra, engine_action)``. ``log_extra`` is ``None`` on a
        clean MCTS recommendation, or ``{"mcts_error": "<message>"}`` when the
        pipeline failed and ``action`` came from the random-legal fallback
        instead. ``engine_action`` is the same choice in the engine's own
        dialect (``None`` without a mirror); the caller applies it AFTER the
        game has accepted the action, so a rejected play never advances the
        mirror.
    """
    try:
        run_state = state.get("run_state") or {}
        character = character_display_name(run_state.get("character")) or cfg.character
        call_cfg = dataclasses.replace(
            cfg, character=character, ascension=run_ascension(run_state, cfg.ascension)
        )
        if sims is not None:
            # Per-encounter budget (regular vs boss); see combat_sims_for.
            call_cfg = dataclasses.replace(call_cfg, sims=sims)

        headless = to_headless_combat_play(state)
        if mirror is not None:
            from sts2_hint import adapter
            from sts2_hint.recommender import tags_from_result

            wire = adapter.translate(
                headless,
                character=call_cfg.character,
                combat_id=call_cfg.combat_id,
                seed=call_cfg.seed,
                ascension=call_cfg.ascension,
            )
            mirror.reconcile(wire)
            tags = tags_from_result(mirror.search(call_cfg), wire["deck"]["hand"])
        else:
            tags = recommend_all(headless, call_cfg)
        if not tags:
            raise RuntimeError("recommend_all() returned no legal actions")

        # Walk the ranking (recommend_all sorts MCTS-best first), skipping
        # plays the REAL game marks unplayable: hand[].playable is the
        # authoritative CardModel.CanPlay() (PROTOCOL.md), and the engine
        # model can miss live-only blocks -- 2026-07-13, Act-3 boss: the
        # Queen's hook made DEFEND unplayable ("BlockedByHook"), the search
        # kept recommending it, and the rejection loop watchdog-killed a
        # healthy run at floor 48. Every play masked -> end_turn.
        action = None
        for tag in tags:
            cand = hint_tag_to_action(tag)
            if cand is None:
                continue  # no bridge-verb mapping (e.g. channel/evoke)
            reason = _wire_unplayable(cand, state)
            if reason is not None:
                logger.warning(
                    "MCTS pick %r rejected by wire legality (%s); trying next-best", cand, reason
                )
                continue
            if banned:
                # Compare in as-sent form: absolute potion slot, no annotations.
                probe = dict(cand)
                _remap_potion_slot(probe, state)
                if json.dumps(probe, sort_keys=True) in banned:
                    logger.warning(
                        "MCTS pick %r already rejected by the game for this state; "
                        "trying next-best",
                        cand,
                    )
                    continue
            action = cand
            break
        if action is None:
            logger.warning("every MCTS action masked by wire legality; ending turn")
            action = {"action": "end_turn"}
        # The engine-dialect twin, captured BEFORE the potion-slot remap below
        # — the engine indexes potions densely, the bridge absolutely.
        engine_action = dict(action) if mirror is not None else None
        # Engine compacts potions to a dense vector; rewrite the potion slot
        # back to the bridge's absolute slot so the play isn't rejected as
        # empty_potion_slot (see _remap_potion_slot).
        _remap_potion_slot(action, state)
        annotations = build_annotations(tags, call_cfg, state)
        if annotations is not None:
            # In-game overlay payload (PROTOCOL.md "annotations"); the mod's
            # SimOverlay renders per-card win-probs. Fire-and-forget.
            action["annotations"] = annotations
        return action, None, engine_action
    except UnsupportedGameVersionError:
        # Also ahead of the blanket handler (GH #807): a build this engine has
        # no rules for is not a bad snapshot, and a random legal move would
        # play an unsupported game blind. The driver refuses instead.
        raise
    except MirrorDivergence:
        # Deliberately ahead of the blanket handler below. `MirrorDivergence`
        # is a `RuntimeError`, so it used to be caught, logged as an "MCTS
        # recommend failed", and answered with a random-legal action — STRICT
        # never fired on `combat_play` in this arm at all.
        #
        # And the fallback makes it worse rather than merely quieter: it
        # returns `engine_action=None`, so the session loop never calls
        # `mirror.advance(...)`, the world commits a move the mirror did not
        # see, and every subsequent reconcile reports cascading FALSE
        # divergences on top of the real one. `pick_card_select_mirrored`
        # was already unwrapped; this makes the two paths agree.
        raise
    except Exception as e:  # Never crash the session on a bad snapshot.
        logger.error(
            "MCTS recommend failed at round=%s, falling back to random-legal policy: %s",
            state.get("round"),
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return pick_combat_action(state, rng), {"mcts_error": str(e)}, None


# ---------------------------------------------------------------------------
# Mid-combat card_select off the persistent combat mirror.
# ---------------------------------------------------------------------------


def pick_card_select_mirrored(
    state: dict[str, Any],
    cfg: RecommendConfig,
    mirror: Any,
    *,
    sims: int | None = None,
    picker: Any = None,
) -> tuple[dict[str, Any], list[int] | None]:
    """Answer a MID-COMBAT ``card_select`` off the fight's persistent mirror.

    Native play (the reference) genuinely SEARCHES these choices: playing
    Armaments/Warcry/True Grit leaves the engine on a ``pending_choice`` whose
    ``ResolveChoice`` options are the only legal actions, so MCTS ranks them
    like any other move. The mirror reproduces that exactly, because it never
    stopped being that engine state — the played card already parked it on the
    choice.

    What this replaced: a three-tier chain that rebuilt the state from the
    previous snapshot, replayed the opening play, and required the engine's
    re-derived option list to match the game's exactly — then fell to a
    prompt-key heuristic, then to uniform random. On the 60-run realdll_v4
    sweep 263 of 635 prompts failed that match gate and 175 were answered at
    random.

    Args:
        state: The bridge ``card_select`` message. ``state["combat"]`` is the
            board the mod now attaches (``SimCombatHandler.
            BuildEmbeddedCombatState``); its absence means this selection is
            not mid-combat and the caller should not be here.
        cfg: MCTS knobs.
        mirror: The fight's :class:`combat_mirror.CombatMirror`.
        sims: Per-decision simulation budget override.
        picker: Who picks once the mirror is parked on the world's choice:
            ``picker(mirror, n_options=, min_select=, max_select=)`` returning
            the option positions. ``None`` = the combat MCTS
            (``mirror.search_choice_picks``). Instinct passes its own
            (``realgame_instinct``), so it answers the SAME parked choice the
            search would, through the same replay and injection.

    Returns:
        ``(bridge action, picked option indices)``. The caller applies the
        picks to the mirror only after the game accepts the action.
    """
    from sts2_hint import adapter

    run_state = state.get("run_state") or {}
    character = character_display_name(run_state.get("character")) or cfg.character
    call_cfg = dataclasses.replace(
        cfg, character=character, ascension=run_ascension(run_state, cfg.ascension)
    )
    if sims is not None:
        call_cfg = dataclasses.replace(call_cfg, sims=sims)

    headless = to_headless_combat_play(state["combat"])
    wire = adapter.translate(
        headless,
        character=call_cfg.character,
        combat_id=call_cfg.combat_id,
        seed=call_cfg.seed,
        ascension=call_cfg.ascension,
    )
    mirror.reconcile(wire)

    parked = mirror.parked_choice()
    bridge_cards = state.get("cards") or []
    options = [
        {
            "id": adapter.registry_card_id(str(c.get("id")), f"$.cards[{i}]"),
            "upgraded": bool(c.get("upgraded") or c.get("upgrade_level")),
        }
        for i, c in enumerate(bridge_cards)
    ]
    # Clamp the arity to what is actually on screen. "Choose any number"
    # arrives as a sentinel max, which is both meaningless past the option
    # count and unrepresentable in the engine's u8 arity fields.
    n = len(options)
    raw_min = state.get("min_count", state.get("min_select")) or 0
    raw_max = state.get("max_count", state.get("max_select")) or 1
    min_select = max(0, min(int(raw_min), n))
    max_select = max(min_select, min(int(raw_max), n))
    # A `FromChooseACardScreen` reports min 0 whether or not it may be
    # declined; the truth is the separate `canSkip` flag the bridge now
    # captures. Normalize BEFORE the recognition, which keys the Gambling-Chip
    # mulligan on `min_select == 0`.
    can_skip, min_select = skip_arity_from_world(
        cancelable=state.get("cancelable"),
        source=state.get("source"),
        min_select=min_select,
        max_select=max_select,
    )
    # The engine's own kind when it derived one; otherwise the recognition
    # SHARED with the DLL driver (`headless_eval.pick_card_select_mirrored`)
    # maps the known unparked screens — generative offers, KnowledgeDemon's
    # curse pick, Gambling Chip's turn-1 mulligan — to their real engine
    # kinds. A still-unrecognized screen keeps `card_select`, which is NOT a
    # usable catch-all: `inject_choice` refuses the unlocated deterministic
    # injection (it would resolve to a silent no-op against the engine's
    # placeholder arm) and drops the mirror, which the `not mirror.alive`
    # branch below then handles.
    kind, turn_start_blocked_on = recognize_card_select(
        parked=parked,
        options=options,
        board=headless,
        wire=wire,
        min_select=min_select,
        source_model=state.get("source_model"),
        prompt_key=state.get("prompt_key"),
    )
    mirror.inject_choice(
        kind,
        options,
        min_select,
        max_select,
        can_skip=can_skip,
        origin={
            "prompt_key": state.get("prompt_key"),
            "source": state.get("source"),
            "source_model": state.get("source_model"),
        },
    )
    if turn_start_blocked_on is not None:
        # C# `Hook.AfterPlayerTurnStart` AWAITS the screen it opened — Gambling
        # Chip's turn-1 mulligan, or (GH #355, any turn) Tyranny's exhaust /
        # Tools of the Trade's discard — so
        # the wire this mirror was just seeded from is a mid-turn-setup
        # snapshot: the hand is drawn and energy is reset, but the rest of the
        # turn-start hook batch has NOT run. Park that batch behind the pick so
        # resolving the injected choice runs it in C# order — armed BEFORE the
        # search so the rollouts see the same continuation. Same order as the
        # DLL driver: inject, arm, then search.
        mirror.arm_turn_start_resume_for(turn_start_blocked_on)

    def _indices(picked: list[int]) -> list[int]:
        """Option positions -> the bridge's own ``cards[i]["index"]`` handles."""
        return [int(bridge_cards[i].get("index", i)) for i in picked]

    if not mirror.alive:
        # OBSERVE only: the injection itself diverged and the mirror was
        # dropped, so there is no state left to search. Take the minimum legal
        # selection and let the run continue TAGGED — STRICT never reaches
        # this line, and no clean measurement may come from a run that did.
        #
        # Do NOT re-seed here: `wire` is the PRE-choice board and the world is
        # about to resolve the very screen the mirror could not answer, so a
        # mirror seeded from it reads the whole resolution as a divergence one
        # decision later. Stay dropped; `reconcile` seeds from the next
        # decision's POST-choice wire. (v28wr census: 50 player-side events.)
        #
        # Picks are `None`, not `[]`: the caller cannot distinguish an empty
        # list from a legitimate skip, and would answer it by calling
        # `advance_through_choice([])` on a mirror that has no choice open —
        # manufacturing a second, entirely spurious divergence.
        picked = list(range(min_select))
        if not picked:
            return {"action": "skip"}, None
        return {"action": "choose", "indices": _indices(picked)}, None

    # One search per pick: the engine decomposes a multi-pick sequentially, so
    # each pick is chosen with the previous ones already in the root state.
    # (A single-pick screen is still exactly one search.) The picks reach the
    # mirror atomically either way, so state fidelity does not depend on this
    # — it is a policy-quality change. Same contract as the DLL driver
    # (`headless_eval.pick_card_select_mirrored`), which calls the same method.
    if picker is not None:
        picked = picker(mirror, n_options=n, min_select=min_select, max_select=max_select)
    else:
        picked = mirror.search_choice_picks(
            call_cfg,
            hand=wire["deck"]["hand"],
            n_options=n,
            min_select=min_select,
            max_select=max_select,
        )
    if not picked and min_select <= 0:
        return {"action": "skip"}, []
    if not picked:
        picked = [0]
    return {"action": "choose", "indices": _indices(picked)}, picked


def run(
    host: str,
    port: int,
    log_dir: Path,
    max_steps: int | None,
    rng: random.Random,
    mcts_cfg: RecommendConfig,
    *,
    sims_regular: int = DEFAULT_SIMS_REGULAR,
    sims_boss: int = DEFAULT_SIMS_BOSS,
    sims_flat: int | None = None,
    character: str | None = None,
    ascension: int | None = None,
    seed: int | None = None,
    map_explore: bool = False,
) -> int:
    """Run the MCTS-driven play loop to completion (or ``max_steps``).

    Identical shape to :func:`realgame_random.run`, except ``combat_action``
    decisions go through :func:`pick_combat_action_mcts` instead of
    :func:`realgame_random.pick_combat_action`. See that function's
    docstring for the full argument reference (mirrored here).

    Returns:
        Process exit code (0 on a clean finish, 1 on a connection error).
    """
    jsonl = JsonlLogger(log_dir)
    step = 0
    result = "unknown"
    mcts_fallback_count = 0
    current_room_type: str | None = None  # type of the map node last walked into

    print(f"Logging to {jsonl.path}")
    # Say which search profile this run used -- the 2026-08-02 clairvoyant
    # confound survived for weeks partly because no live report named it.
    print(
        f"Combat search: hidden_info={mcts_cfg.hidden_info} "
        f"c_puct={mcts_cfg.c_puct} draw_widening_c={mcts_cfg.draw_widening_c}"
    )

    try:
        with RealGameBridgeClient(host=host, port=port) as client:
            while True:
                state = client.receive_state()
                msg_type = state.get("type")

                if msg_type == "pong":
                    continue

                if msg_type == "error":
                    jsonl.log_error(state)
                    print(f"[error] {state.get('error')}: {state.get('detail')}")
                    continue

                if msg_type in _TERMINAL_TYPES:
                    step += 1
                    jsonl.log_terminal(step, state)
                    result = state.get("result") or state.get("message") or msg_type
                    print(f"[{step:4d}] {msg_type} -> result={result!r}")
                    if msg_type == "run_complete":
                        break
                    continue

                # GH #374: the live save's real UnlockState rides the envelope.
                # Latched, so this is a dict lookup after the first message that
                # carries it; it supersedes the census guess installed at
                # startup. Placed before any reconstruction — pool caches are
                # keyed on the lock generation and rebuild on change, but a
                # mirror built against the wrong pool would still have consumed
                # the wrong number of RNG reads getting there.
                install_unlock_state_from_wire(state.get("run_state") or {})

                log_extra: dict[str, Any] | None = None
                if msg_type == "combat_action":
                    sims = combat_sims_for(current_room_type, sims_regular, sims_boss, sims_flat)
                    # Unmirrored lane: engine_action is always None here.
                    action, log_extra, _ = pick_combat_action_mcts(state, mcts_cfg, rng, sims=sims)
                    if log_extra is not None:
                        mcts_fallback_count += 1
                else:
                    action = choose_action(
                        state,
                        rng,
                        run_config_character=character,
                        run_config_ascension=ascension,
                        run_config_seed=seed,
                        map_explore=map_explore,
                    )
                    # Remember the room we're entering so the next combat knows
                    # whether it's the boss (4000 sims) or a regular fight (1000).
                    node_type = chosen_map_node_type(state, action)
                    if node_type is not None:
                        current_room_type = node_type

                step += 1
                jsonl.log_step(step, state, action, extra=log_extra)
                print(_progress_line(step, state, action))
                client.send_action(action)

                if max_steps is not None and step >= max_steps:
                    result = "max_steps_reached"
                    print(f"Stopping after max_steps={max_steps}")
                    break
    except UnsupportedGameVersionError as e:
        # GH #807: `pick_combat_action_mcts` re-raises a build the engine has
        # no rules for rather than answering it with a random legal move.
        logger.error("Refusing to play: %s", e)
        print(f"Steps: {step}\nResult: refused ({e})\nLog: {jsonl.path}")
        jsonl.close()
        return 2
    except (ConnectionError, TimeoutError) as e:
        logger.error("Bridge connection error after %d steps: %s", step, e)
        result = f"error: {e}"
        print(
            f"Steps: {step}\nMCTS fallbacks: {mcts_fallback_count}\n"
            f"Result: {result}\nLog: {jsonl.path}"
        )
        jsonl.close()
        return 1

    jsonl.close()
    print(
        f"Steps: {step}\nMCTS fallbacks: {mcts_fallback_count}\nResult: {result}\nLog: {jsonl.path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bridge mod TCP host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bridge mod TCP port.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/realgame"),
        help="Directory to write the JSONL run log into.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many decisions (default: run to game_over/run_complete).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for the random-legal fallback policy's RNG, and (if the mod's "
            "client_run_config flag is on) the run seed offered in start_run."
        ),
    )
    parser.add_argument(
        "--character",
        default=None,
        help="Character id to request via start_run when a run_config message is offered.",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=None,
        help="Ascension level to request via start_run when a run_config message is offered.",
    )
    parser.add_argument(
        "--explore-map",
        action="store_true",
        help="Bias map choices toward rarely exercised screen types (see realgame_random).",
    )
    parser.add_argument(
        "--sims",
        "--playouts",
        dest="sims",
        type=int,
        default=None,
        help=(
            "Flat MCTS simulations for EVERY combat decision, overriding the "
            "per-encounter --sims-regular/--sims-boss schedule (default: adaptive)."
        ),
    )
    parser.add_argument(
        "--sims-regular",
        type=int,
        default=DEFAULT_SIMS_REGULAR,
        help=f"MCTS simulations for regular (non-boss) fights (default: {DEFAULT_SIMS_REGULAR}).",
    )
    parser.add_argument(
        "--sims-boss",
        type=int,
        default=DEFAULT_SIMS_BOSS,
        help=f"MCTS simulations for boss fights (default: {DEFAULT_SIMS_BOSS}).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=RecommendConfig.depth,
        help=f"MCTS rollout depth cap (default: {RecommendConfig.depth}).",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=None,
        help="Explicit exploration constant; default follows the search profile "
        "(2.5 honest / 1.4 with --clairvoyant).",
    )
    parser.add_argument(
        "--clairvoyant",
        action="store_true",
        help="reproduce the OLD leaky combat search (hidden_info=clairvoyant, "
        "c_puct=1.4, no draw widening). Default matches the DEPLOYED honest "
        "MctsCombatPolicy (beliefdraw, c_puct=2.5, draw_widening_c=2.0) — the "
        "same trio headless_eval and realgame_eval default to. This driver was "
        "missed by the 2026-08-02 sweep that fixed those two (GH #383) and "
        "silently ran the clairvoyant profile until then.",
    )
    parser.add_argument(
        "--mcts-seed",
        type=int,
        default=RecommendConfig.seed,
        help=f"RNG seed for MCTS rollouts, separate from --seed (default: {RecommendConfig.seed}).",
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["heuristic", "uniform", "net"],
        default=RecommendConfig.rollout_policy,
        help="MCTS rollout policy (default: what --hidden-info deploys — the playout net under beliefdraw).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=RecommendConfig.threads,
        help=(
            "Root-parallel MCTS workers (sts2_rs par_search). Default "
            f"{RecommendConfig.threads} preserves the historical single-tree "
            "search byte-for-byte; raise (e.g. 8) to cut boss-decision think "
            "time several-fold at the cost of per-seed comparability with "
            "threads=1 runs (root-parallel splits the sim budget across "
            "independently seeded trees)."
        ),
    )
    parser.add_argument(
        "--time-budget-ms",
        type=int,
        default=RecommendConfig.timeout_ms,
        help=(
            "Forwarded as RecommendConfig.timeout_ms. NOTE: as of this writing "
            "sts2_hint.recommender documents this knob as accepted but not yet "
            "enforced by the underlying sts2_rs.mcts_recommend call."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # GH #367: the live game plays against a real save, whose unrevealed epochs
    # prune every card pool. Install that before the first reconstruction —
    # the pools are memoized, so a late install would leave already-built ones
    # stale. Override or disable with $STS2_LIVE_LOCKED_EPOCHS.
    install_live_unlock_state()

    rng = random.Random(args.seed)
    mcts_cfg = RecommendConfig(
        # Base sims is a placeholder: run() overrides it per decision via the
        # regular/boss schedule (or the flat --sims override). Seed it with the
        # value that will actually be used most so a stray un-overridden call is
        # still sensible.
        sims=args.sims if args.sims is not None else args.sims_regular,
        depth=args.depth,
        # The three honest knobs travel together (see RecommendConfig's
        # hidden_info comment): on reconstructed real-DLL states the RNG cursors
        # are faked, so a clairvoyant root plans against a fictional future.
        hidden_info="clairvoyant" if args.clairvoyant else "beliefdraw",
        c_puct=args.c_puct if args.c_puct is not None else (1.4 if args.clairvoyant else 2.5),
        draw_widening_c=None if args.clairvoyant else 2.0,
        seed=args.mcts_seed,
        threads=args.threads,
        rollout_policy=args.rollout_policy,
        timeout_ms=args.time_budget_ms,
    )
    return run(
        args.host,
        args.port,
        args.log_dir,
        args.max_steps,
        rng,
        mcts_cfg,
        sims_regular=args.sims_regular,
        sims_boss=args.sims_boss,
        sims_flat=args.sims,
        character=args.character,
        ascension=args.ascension,
        seed=args.seed,
        map_explore=args.explore_map,
    )


if __name__ == "__main__":
    sys.exit(main())
