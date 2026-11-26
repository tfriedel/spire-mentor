"""Advisor backend for SpireMentor advisor mode (PROTOCOL.md section 5, v3).

The mod is the server; this client connects to it (same transport as every
other ``realgame_*`` driver) and answers ``advice_request`` messages with a
full ranked ``advice`` payload. It never sends game-driving verbs: advice
is rendered by the mod's HUD, never executed by this process (the phase-7
takeover executes game-side).

Pipeline per ``advice_request`` (combat only in v1):

1. Reinterpret the payload as the ``combat_action`` snapshot it wraps
   (``advice_for`` field) and translate it with
   :func:`realgame_adapter.to_headless_combat_play` — the same path the
   autopilot MCTS driver uses.
2. Run the deployed honest-search configuration
   (``hidden_info="beliefdraw"``, ``c_puct=2.5``, ``draw_widening_c=2.0``,
   and the run agent's EndTurn widening 5.787 / 0.436, which the engine
   applies with the mode — NOT the leaky clairvoyant default) via
   ``sts2_rs.mcts_recommend``, keeping the raw result so the principal
   variation survives (``recommend_all`` drops it). Budget, tree shape, pick
   rule and the last-boss win value are the Rust run agent's
   (:func:`fight_search_cfg`; ``--sims`` / ``--boss-sims`` / ``--threads``).
3. Map the ranked rows through :func:`sts2_hint.recommender.tags_from_result`
   and build the wire ``advice`` object: ``p_win``, ``best`` (executable),
   ``options[]`` (label / p_win / q / visits / visit_share / W-L-T /
   is_best / low_confidence / card_index / target_index), ``pv``.

``human_action`` messages and every advice exchange are appended to a JSONL
log under ``--log-dir`` for the phase-9 review pipeline. Never-crash
policy: any per-request failure is logged, answered with an empty advice
object (the HUD just shows nothing), and the loop continues. Messages this
process does not understand (e.g. autopilot decision messages if it is
accidentally connected to an autopilot session) are logged and ignored —
it must never answer them.

Usage::

    uv run python python/realgame_advisor.py [--host 127.0.0.1] [--port 9002]
        [--sims 4000] [--boss-sims 16000] [--threads 4]
        [--log-dir logs/realgame_advisor]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import realgame_hand_select
import runtime_paths
from coop_coordination import apply_ally_wait
from display_names import sanitize_advice, sanitize_coach
from enemy_labels import ally_label, enemy_label
from realgame_adapter import (
    UnsupportedCharacterError,
    UnsupportedGameVersionError,
    character_display_name,
    check_supported_characters,
    envelope_game_version,
)
from realgame_advisor_mirror import AdvisorMirror
from realgame_bridge import DEFAULT_HOST, DEFAULT_PORT, RealGameBridgeClient
from realgame_coach import CoachGrader, GraderThread
from realgame_control import BRIDGE_MESSAGE_TYPES, BugReportForward, ControlPlane, StaticSite
from realgame_instinct import InstinctAdvisor, InstinctUnavailable
from realgame_livetree import LiveRunTree
from realgame_workbench_log import WorkbenchRunLogger
from sts2_client import BETA_GAME_VERSIONS
from sts2_hint.recommender import (
    CHANNEL_STABLE_IDX,
    CHOOSE_SKIP_STABLE_IDX,
    CHOOSE_STABLE_IDX_BASE,
    END_TURN_STABLE_IDX,
    EVOKE_STABLE_IDX,
    POTION_STABLE_IDX_BASE,
    HintTag,
    RecommendConfig,
)

logger = logging.getLogger(__name__)

#: The deployed honest-search knobs (see RecommendConfig.hidden_info notes:
#: beliefdraw + c_puct 2.5 + draw_widening_c 2.0 travel together).
#:
#: The EndTurn widening is deliberately NOT listed: it is left to the engine,
#: which applies the run agent's values with the mode
#: (`MctsConfig::apply_mode_dialect`, GH #758). Before that the advisor
#: searched EndTurn at the pre-#467 0.5 / 0.4 while every published win rate
#: was measured at 5.787 / 0.436.
#:
#: The `c_puct` here is the value for the DEPLOYED budgets. The search resolves
#: it per decision via `belief_draw_c_puct(sims, threads)` instead, because the
#: optimum rises with the budget and `--sims` is chosen independently of this
#: dict. At every deployed budget the two agree; they diverge only when the
#: budget is trimmed, which is the case this protects.
#:
#: This stays on the BASE dialect on purpose (GH #830). The run agent adds an
#: Ironclad layer on top (`sts2_runai::policy::IRONCLAD_SEARCH_KNOBS` and a BC
#: move prior in elite and boss fights) that the advisor does not take, so for
#: Ironclad "the deployed search" means two things: this one, and the run agent's.
HONEST_SEARCH = {
    "hidden_info": "beliefdraw",
    "c_puct": 2.5,
    "draw_widening_c": 2.0,
}

#: What this advisor advertises on every ``advice`` reply (PROTOCOL.md v5). A
#: mod sends the new ``human_action`` kinds (end turn, run-level choices) only
#: once it has seen ``coach >= 1``, because an older advisor's mirror would
#: drop on them. ``coach: 2`` (v5.1) adds the fight recap and the death
#: autopsy: the mod sends ``combat_ended`` and its Advisor-mode ``game_over``
#: only to an advisor at 2 or above. ``crystal_sphere: 1``: this advisor answers
#: the Crystal Sphere screen (the digger, :mod:`crystal_sphere_policy`).
ADVISOR_CAPABILITIES = {"coach": 2, "potion_swap": 1, "crystal_sphere": 1}

#: Instinct, the e2e student (:mod:`realgame_instinct`): loaded on first use,
#: so an advisor nobody asks for Instinct never reads its 21 MB of weights.
#: ``--instinct-weights`` replaces it in :func:`main`.
INSTINCT = InstinctAdvisor()

#: Acts in a standard run; the fallback for a mod too old to send
#: ``run_state.is_last_act``.
STANDARD_ACT_COUNT = 3


def fight_search_cfg(cfg: RecommendConfig, state: dict[str, Any]) -> RecommendConfig:
    """``cfg`` with the boss win value the run agent uses for THIS fight.

    The run agent (``MctsCombatPolicy::pick_search_setup``, default
    ``STS2_RUNAI_BOSS_CARRY``) changes one thing per fight: on the run's LAST
    boss nothing carries forward, so only winning counts
    (``sts2_mcts::boss_win_carry``). Every other fight keeps the default value.

    The last-boss test reads the mod's ``run_state.is_last_act`` /
    ``awaiting_second_boss``. A mod too old to send them falls back to "act 3,
    and no A10 second boss can be pending" — below A10 the act-3 boss is the
    last one; at A10 it cannot tell which of the two this is, so it keeps the
    default value rather than guess.
    """
    import sts2_rs

    if (state.get("room_type") or "") != "Boss":
        return cfg
    run_state = state.get("run_state") or {}
    ascension = int(run_state.get("ascension") or 0)
    is_last_act = run_state.get("is_last_act")
    awaiting_second = run_state.get("awaiting_second_boss")
    if is_last_act is None:
        is_last_act = int(state.get("act") or 0) >= STANDARD_ACT_COUNT
    if awaiting_second is None:
        if ascension >= 10 and is_last_act:
            return cfg
        awaiting_second = False
    if not is_last_act or awaiting_second:
        return cfg
    carry = sts2_rs.boss_win_carry(True, True, False, ascension)
    if carry is None:
        return cfg
    return dataclasses.replace(cfg, win_hp_carry=carry[0], win_potion_carry=carry[1])


def _tag_label(tag: HintTag, state: dict[str, Any]) -> str:
    """Human-readable option label: card/potion name plus target name (Rust)."""
    import sts2_rs

    return sts2_rs.advice_tag_label(tag._asdict(), state)


def _pv_labels(pv: list[Any], state: dict[str, Any]) -> list[str]:
    """Name the principal variation without lying about the deeper plies: only
    ply 1 indexes the hand the player sees (Rust)."""
    import sts2_rs

    return list(sts2_rs.advice_pv_labels(pv, state))


def _best_to_action(tag: HintTag, state: dict[str, Any]) -> dict[str, Any] | None:
    """The executable ``best`` object (mod-side takeover dialect), a potion's
    dense slot remapped to the bridge's belt slot (Rust)."""
    import sts2_rs

    return sts2_rs.advice_best_to_action(tag._asdict(), state)


def option_move(tag: HintTag, state: dict[str, Any]) -> dict[str, Any]:
    """What a combat option row DOES, in ``best``'s dialect (PROTOCOL.md §5)."""
    import sts2_rs

    return sts2_rs.advice_option_move(tag._asdict(), state)


def _veto_text(reason: str, state: dict[str, Any]) -> str:
    """``hand[2]:EnergyCostTooHigh`` for the HUD: ``Bash (energy cost too high)``."""
    import sts2_rs
    from display_names import _names

    return sts2_rs.advice_veto_text(reason, state, _names())


def wire_unplayable(state: dict[str, Any]) -> dict[int, str]:
    """Hand indices the GAME has already declared unplayable, and why (the
    bridge's ``playable`` / ``unplayable_reason``; absent reads as playable)."""
    import sts2_rs

    return dict(sts2_rs.advice_wire_unplayable(state))


def veto_unplayable(tags: list[HintTag], state: dict[str, Any]) -> tuple[list[HintTag], list[str]]:
    """Drop ranked plays the live game has already said it will refuse (Rust:
    ``sts2_advisor::combat_advice::veto_unplayable``).

    The search runs on a reconstruction of the board, and it disagrees with
    the game about legality often enough to matter (41 refused recommended
    plays across 51 climb runs). Returns the surviving tags, best-first --
    when the veto took the engine's pick, ``is_best`` moves to the engine's
    own next choice -- and one reason per vetoed play. Everything vetoed
    ends the turn.
    """
    import sts2_rs

    if not wire_unplayable(state):
        return tags, []
    kept, reasons = sts2_rs.advice_veto_unplayable([t._asdict() for t in tags], state)
    if not reasons:
        return tags, []
    return [HintTag(**t) for t in kept], list(reasons)


def build_advice(
    tags: list[HintTag],
    state: dict[str, Any],
    *,
    sims: int,
    pv: list[str] | None = None,
    pick: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The wire ``advice`` object (PROTOCOL.md section 5) from ranked tags
    (Rust: ``sts2_advisor::combat_advice::build_advice``).

    ``pick`` is the search result's own explanation of its pick
    (``mcts_recommend()["pick"]``, GH #1022), carried only while the advice
    still recommends that pick.
    """
    import sts2_rs
    from display_names import _names

    return sts2_rs.advice_build([t._asdict() for t in tags], state, sims, _names(), pv, pick)


def compute_combat_advice(
    state: dict[str, Any],
    cfg: RecommendConfig,
    sims: int,
    mirror: Any = None,
    plan_budget_s: float | None = None,
) -> tuple[dict[str, Any], Any, RecommendConfig, list[dict[str, Any]]]:
    """One full search for one ``advice_request`` wrapping a combat board, in
    Rust (``sts2_advisor::advise_combat``, GH #1426).

    Returns ``(advice, cs, call_cfg, rows)``: the wire advice object (with
    the turn plan), the native ``sts2_rs`` combat state the search ran on and
    the resolved config, so callers can run companion searches
    (``mcts_search_tree``) against the very same board, and the search's
    regret rows, which the live coach grades the human's move against.

    ``call_cfg`` is ``cfg`` with the run's character and ascension, ``sims``,
    and ``c_puct=None``: ``sims`` varies per decision (boss vs regular) while
    a pinned ``c_puct`` is chosen once, so the constant is left to
    ``RecommendConfig.resolved_c_puct``, which reads it off the budget this
    decision actually gets.

    Args:
        mirror: The fight's :class:`realgame_advisor_mirror.AdvisorMirror`.
            When it is carrying this fight the search runs on the CARRIED
            state instead of a fresh rebuild, which is the only way it sees
            the per-turn counters the wire does not express (GH #370). A
            mirror parked on a selection the world has finished is dropped
            (a combat screen has no executable ``choose``), and a failed
            mirror search drops it and falls back to the rebuild. ``None``
            keeps the stateless rebuild.
        plan_budget_s: Seconds the turn plan may re-search; `None` is the
            engine's `TURN_PLAN_BUDGET_S`, `inf` no cap (a replay: a plan the
            clock cut depends on the machine's load, GH #1784).

    Raises:
        UnsupportedGameVersionError: a build the engine has no rules for.
    """
    import sts2_rs
    from display_names import _names

    if state.get("type") == "combat_action":
        envelope_game_version(state)
    core = mirror._core if mirror is not None else None
    if plan_budget_s is None:
        plan_budget_s = sts2_rs.TURN_PLAN_BUDGET_S
    try:
        advice, cs, character, ascension, rows, log = sts2_rs.advice_compute_combat(
            state, dataclasses.asdict(cfg), sims, _names(), core, plan_budget_s
        )
    finally:
        if mirror is not None:
            mirror._flush()
    for level, message in log:
        getattr(logger, level)("%s", message)
    call_cfg = dataclasses.replace(
        cfg, character=character, ascension=ascension, sims=sims, c_puct=None
    )
    return advice, cs, call_cfg, rows


def _mcts_kwargs(call_cfg: RecommendConfig, sims: int | None = None) -> dict[str, Any]:
    """``sts2_rs.mcts_recommend``'s keyword arguments for ``call_cfg``."""
    return {
        "sims": call_cfg.sims if sims is None else sims,
        "c_puct": call_cfg.resolved_c_puct(),
        "depth": call_cfg.depth,
        "seed": call_cfg.seed,
        "threads": call_cfg.threads,
        "fair": call_cfg.fair,
        "rollout_policy": call_cfg.rollout_policy,
        "hidden_info": call_cfg.hidden_info,
        "draw_widening_c": call_cfg.resolved_draw_widening_c(),
        "endturn_widening_c": call_cfg.endturn_widening_c,
        "endturn_widening_alpha": call_cfg.endturn_widening_alpha,
        "common_random_numbers": call_cfg.common_random_numbers,
        "win_hp_carry": call_cfg.win_hp_carry,
        "win_potion_carry": call_cfg.win_potion_carry,
    }


#: ``advice_for`` values answered by the PPO run policy rather than by MCTS.
#: These are the screens a human actually sees (autopilot never does: its
#: ICardSelector bypasses them), so the advice has to be built from the wire
#: payload the mod scrapes off the live screen -- see SimScreenAdvice.cs.
SCREEN_ADVICE_FOR = (
    "card_reward",
    "card_select",
    "map_select",
    "shop",
    "event",
    "rest_site",
    "boss_relic",
    "treasure",
    "card_bundle",
    # A full-belt potion reward (PROTOCOL.md v5.2): the mod asks it only of an
    # advisor that advertises `capabilities.potion_swap`.
    "potion_swap",
    # No net: `crystal_sphere_policy` (the digger) answers it, see
    # `_crystal_sphere_advice`. Advertised as `capabilities.crystal_sphere`.
    "crystal_sphere",
)

#: Surfaces answered by a DIFFERENT trained head than their own name suggests,
#: because the champion was never trained on them directly. Each one is a
#: judgement call about which trained question is closest -- and each one is
#: announced in the HUD note, because a ranking whose meaning has been
#: substituted must not read like a native one.
_SURFACE_STANDINS = {
    # A chest relic and a boss relic are the same question ("which of these
    # relics do I want"), only the price differs.
    "treasure": ("boss_relic", "scored with the boss-relic head"),
}

#: Surfaces the advisor answers but has no policy FOR, keyed to the player-facing
#: reason that rides on ``advice.hand_back`` (GH #874). A borrowed head is
#: honest in the HUD note, but it is still not the AI deciding, so F7 hands the
#: screen back instead of playing it. Delete an entry when its surface gets a
#: trained head (GH #379); the mod keeps no list of its own, so the pause goes
#: away with it.
_HAND_BACK_SURFACES = {
    "treasure": "the chest relic pick has no trained policy",
    "card_bundle": "card bundles have no trained policy",
}


def _trained_on_packs(policy: Any) -> bool:
    """Whether this run-policy net reads Scroll Boxes packs (``card_bundles``).

    ``RustPolicy`` carries its features as the manifest string,
    ``OnnxPolicy`` as a parsed set; both name the schema the net was trained at.
    """
    # `ppo.features`, not `ppo.schema`: schema imports numpy, which the
    # packaged advisor does not ship.
    from ppo import features as ppo_features

    features = getattr(policy, "features", "") or ""
    if isinstance(features, str):
        features = ppo_features.parse_features(features)
    return ppo_features.CARD_BUNDLES in features


#: Characters whose run-policy net has already been warmed this session.
_WARMED: set[str] = set()


def warm_policy_async(policies: Any, character: str | None) -> None:
    """Load the run-policy net for ``character`` on a background thread.

    The first screen decision otherwise pays the whole import-torch +
    load-checkpoint bill inline: 45.3 s on the first card reward of the
    2026-08-05 session, by which time the mod had long since cancelled the
    request and the screen showed nothing at all. Combat advice arrives well
    before the first card reward, so warming on any message with a character
    means the net is resident when a screen finally needs it.
    """
    key = (character or "").strip().lower()
    if not key or key in _WARMED:
        return
    _WARMED.add(key)

    def _load() -> None:
        t0 = time.monotonic()
        policy, error = policies.get(character)
        if policy is None:
            logger.warning("run-policy net unavailable for %s: %s", character, error)
        else:
            logger.info("run-policy net ready for %s (%.1fs)", character, time.monotonic() - t0)

    threading.Thread(target=_load, name=f"warm-{key}", daemon=True).start()


def _decline_potion_swap(state: dict[str, Any]) -> dict[str, Any]:
    """The full-belt swap with the run policy's swap head off: decline it.

    What the engines do with ``STS2_RUNAI_POTION_SWAP`` off (the reward stays
    on the screen), and what the mod did before it asked at all. Not a
    ranking, so it says so; Instinct (trained with the swap on) ranks it.
    Rust: ``sts2_advisor::screen_advice::decline_potion_swap``.
    """
    import sts2_rs
    from display_names import _names

    return sts2_rs.screen_decline_potion_swap(state, _names())


def screen_labels(state: dict[str, Any], advice_for: str, decision: Any) -> list[str]:
    """The run policy's option labels, as the player reads them.

    The decision builders (:mod:`realgame_ppo`) label options for logs and
    for the encoder, and three of those spellings are ids: a shop option's
    ``label`` is the merchant entry's id (``SHRUG_IT_OFF``), a shop removal is
    ``remove:<CARD_ID>``, and a Neow option is its relic's id. The screen's
    own display names replace them here; every other surface's builder
    already labels from names (and :func:`display_names.sanitize_advice`
    guards the rest at the wire). Rust: ``screen_advice::screen_labels``.
    """
    import sts2_rs
    from display_names import _names

    return sts2_rs.screen_labels(
        state,
        advice_for,
        list(decision.actions),
        list(decision.responses),
        [str(x) for x in decision.labels],
        _names(),
    )


def _scores_rust_obs(policy: Any) -> bool:
    """Whether `policy` is the Rust net, which scores the Rust-built
    observation (`realgame_ppo.build_wire_decision`)."""
    from ppo.rust_policy import RustPolicy

    return isinstance(policy, RustPolicy)


def _advisor_builders() -> dict[str, Any]:
    """Builders for surfaces only the advisor is asked: not message types the
    autopilot receives, which `realgame_ppo._PPO_NONCOMBAT_BUILDERS` keys."""
    import realgame_ppo

    return {"potion_swap": realgame_ppo._build_potion_swap}


#: Prefix of the one structured line every `_unranked` answer logs (GH #980).
#: `spire-mentor-mod/tools/ascension_climb.py` counts these per run from the
#: archived advisor-log slice, so keep the two in step.
UNRANKED_LOG_PREFIX = "unranked fallback"

#: The engine's spelling for content it has no model of (`run_manager.rs`:
#: ``unknown card id "KI_CHANT"``).
_ENGINE_UNKNOWN_ID = re.compile(r'unknown (card|relic|potion) id "([A-Za-z0-9_]+)"')


def engine_missing_content(exc: BaseException) -> str | None:
    """The fallback reason when Instinct failed only because the engine lacks
    a card / relic / potion on the screen, else ``None``.

    A v0.113.0 card the engine does not model yet (The Fool's, offered to
    another character by a mixed reward) reached Instinct as a bare
    ``ValueError`` and the HUD said "Instinct hit an error" (live,
    2026-11-13). It is a known gap, not a crash: name what is missing.
    """
    if not isinstance(exc, ValueError):
        return None
    m = _ENGINE_UNKNOWN_ID.search(str(exc))
    if m is None:
        return None
    return f"the engine has no {m.group(1)} {m.group(2)} yet"


def _unranked(state: dict[str, Any], why: str, trigger: str | None = None) -> dict[str, Any]:
    """An answer for a surface no head can rank -- honest, but still usable.

    Returning only a note would leave F7 sitting on the screen forever, which
    is the failure this whole pass is about. So the first option is named as
    the pick and the note says plainly that nothing ranked it: an arbitrary
    choice the player can see is arbitrary beats a silent stall.

    Every answer from here also carries ``hand_back`` (GH #874): with the
    mod's default ``pause_on: "unsupported"`` F7 stops here and gives the
    player the screen, so the first-option pick only plays when they opted
    out of pausing (unattended runs).

    ``trigger`` names which way in this was (GH #980): ``card_select``,
    ``no_head:<advice_for>``, ``no_decision:<advice_for>`` or
    ``inference_failed``. It goes into one structured log line
    (:data:`UNRANKED_LOG_PREFIX`) with the prompt key and source model, so a
    climb can count fallbacks per run and `realgame_hand_select`'s direction
    tables can be extended from data. Rust: ``screen_advice::unranked``.
    """
    import sts2_rs

    advice, warning = sts2_rs.screen_unranked(state, why, trigger)
    if warning is not None:
        logger.warning("%s", warning)
    return advice


def _crystal_sphere_advice(state: dict[str, Any]) -> dict[str, Any]:
    """The Crystal Sphere: :func:`crystal_sphere_policy.choose`'s pick, which
    needs no run-policy net (the digger, or the GH #872 rule for an older
    mod's message -- the note says which)."""
    import sts2_rs

    advice, warning = sts2_rs.crystal_sphere_advice(state)
    if warning is not None:
        logging.getLogger("crystal_sphere_policy").warning("%s", warning)
    return advice


def _blitz_shop_answer(state: dict[str, Any]) -> dict[str, Any]:
    """The Blitz merchant (v0.113.0): an admitted rule, never a net ranking.

    `_unranked`'s first-option default would be "Leave shop" here -- three
    free picks thrown away -- so the first uncursed ware is named instead
    (``realgame_ppo.blitz_shop_choice``), and the note says what it is. Still
    a hand-back surface: nothing was trained on it (GH #874). Rust:
    ``screen_advice::blitz_shop_answer``.
    """
    import sts2_rs

    advice, warning = sts2_rs.screen_blitz_shop(state)
    logger.warning("%s", warning)
    return advice


def _build_card_bundle(state: dict[str, Any]) -> Any:
    """Rank Scroll Boxes packs with the obs the engine gives the Rust agent.

    The engine surfaces a bundle pick as a card reward whose options are each
    pack's FIRST card, plus the whole packs in ``card_reward_bundles``
    (``RunManager::surface_pending_bundle``; ``policy.rs::build_obs``). This
    builds exactly that, so the live answer is the decision training saw
    (GH #978): a champion trained with ``card_bundles`` reads every pack
    member; any other champion scores each pack by its first card, as it was
    trained to.

    Returns a ``PpoDecision`` whose actions are the packs.
    """
    import realgame_ppo

    bundles = [b for b in (state.get("options") or []) if isinstance(b, dict)]
    if not bundles:
        return None
    run_state = state.get("run_state") or {}

    packs: list[list[dict[str, Any]]] = []
    for bundle in bundles:
        cards = [c for c in (bundle.get("cards") or []) if isinstance(c, dict)]
        if not cards:
            return None
        packs.append(cards)

    obs = realgame_ppo._base_obs(state, run_state)
    obs["card_reward_option_ids"] = [realgame_ppo._card_id(p[0]) for p in packs]
    # The engine carries no upgrade state on bundle offerings
    # (`card_reward_upgraded_flags`), so neither does this.
    obs["card_reward_option_upgraded"] = []
    obs["card_reward_bundles"] = [[realgame_ppo._card_id(c) for c in p] for p in packs]
    actions = [{"action": "pick_card_reward", "idx": i} for i in range(len(packs))]
    responses = [
        {"action": "choose", "index": int(b.get("index", i))} for i, b in enumerate(bundles)
    ]
    labels = [
        " + ".join(str(c.get("name") or c.get("id")) for c in (b.get("cards") or []))
        for b in bundles
    ]
    return realgame_ppo.PpoDecision(
        obs, actions, realgame_ppo._base_metrics(state, run_state), responses, labels
    )


def _searched_card_select_advice(
    state: dict[str, Any],
    mirror: Any,
    cfg: RecommendConfig | None,
    sims: int | None,
) -> dict[str, Any] | None:
    """Rank an IN-COMBAT ``card_select`` with the combat MCTS, or give up.

    The mirror owns the carried fight, so it answers this
    (:meth:`AdvisorMirror.answer_card_select`). ``None`` for every reason not
    to answer; the caller then falls to the heuristic rung exactly as before.
    """
    if mirror is None:
        return None
    return mirror.answer_card_select(state, cfg, sims)


def _selected_count(state: dict[str, Any]) -> int:
    """Cards the screen already has selected (``selected_count``; absent: none)."""
    try:
        return max(0, int(state.get("selected_count") or 0))
    except (TypeError, ValueError):
        return 0


def _confirm_taken(state: dict[str, Any]) -> dict[str, Any]:
    """Confirm the cards already selected: ``skip``, which the mod plays as
    the screen's confirm (Rust: ``advisor_mirror::confirm_selection_answer``)."""
    cards = state.get("cards") or []
    return {
        "note": f"{_selected_count(state)} pick(s) done · confirm",
        "options": [
            {
                "label": str(c.get("name") or c.get("id") or f"card {i}"),
                "is_best": False,
                "index": int(c.get("index", i)),
                "action": "choose",
            }
            for i, c in enumerate(cards)
        ],
        "best": {"action": "skip", "label": "confirm"},
    }


def is_fight_offer(state: dict[str, Any]) -> bool:
    """A choose-a-card screen opened INSIDE a fight (Discovery, a Skill /
    Attack / Power Potion): a card generated into this fight, not a deck pick.
    The same screen out of a fight (Lead Paperweight) adds to the deck."""
    return state.get("surface") == "choose_a_card" and state.get("combat") is not None


def fight_offer_as_select(state: dict[str, Any]) -> dict[str, Any]:
    """An in-fight choose-a-card as the in-combat ``card_select`` it is.

    The game opens both through ``CardSelectCmd`` (``FromChooseACardScreen``),
    and the fight mirror answers it the same way: replay what opened it
    (``source_model``, sent by the mod since 0.10.0-beta.7) on a fork of the
    carried fight and stand on the engine's own choice. Shared by both brains.
    """
    return {
        **state,
        "advice_for": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cancelable": bool(state.get("can_skip")),
        "source": "FromChooseACardScreen",
    }


def compute_screen_advice(
    state: dict[str, Any],
    advice_for: str,
    policies: Any,
    *,
    session: dict[str, Any] | None = None,
    mirror: Any = None,
    cfg: RecommendConfig | None = None,
    sims: int | None = None,
) -> dict[str, Any]:
    """Rank a card-reward / card-select screen with the run-policy net.

    Reuses :mod:`realgame_ppo`'s decision builders verbatim -- the same
    obs/action encoding the champion was trained and evaluated with -- and
    reports the FULL masked softmax (``NetPolicy.distribution``) instead of
    one sample, so the HUD can show every option's probability.

    The net's per-option probability rides on ``visit_share`` (the field the
    HUD already draws as a bar); there is no per-action Q for these surfaces,
    only the state value V, which goes in the note.

    Args:
        session: Per-run scratch (:func:`realgame_ppo.apply_session_context`).
            The advisor passed nothing here for its whole life, which is why
            its map advice silently degraded on the ~15% of live `map_select`
            messages that arrive with an empty graph. ``None`` keeps the old
            stateless behaviour for one-shot callers and tests.
        mirror: The fight's :class:`AdvisorCombatMirror`, when one is live.
            Only an IN-COMBAT ``card_select`` uses it -- see
            :func:`_searched_card_select_advice`.
        cfg / sims: Search knobs for that path; both required for it to run.

    Rust: ``sts2_advisor::screen_advice::compute_screen_advice``, for the Rust
    net (the default backend). A character whose policy is another backend
    (the onnxruntime fallback, a torch checkpoint) is answered by
    :func:`_other_backend_screen_advice` from the policy lookup on.
    """
    import realgame_advisor_mirror
    import sts2_rs
    from display_names import _names

    other: dict[str, Any] = {}

    def policy_for(character: str) -> Any:
        policy, error = policies.get(character)
        if policy is None:
            return str(error)
        if _scores_rust_obs(policy):
            return policy._net
        other["policy"] = policy
        return None

    try:
        out = sts2_rs.screen_compute(
            state,
            advice_for,
            policy_for,
            _names(),
            realgame_advisor_mirror._wire_for,
            session=session,
            mirror=mirror._core if mirror is not None else None,
            cfg=dataclasses.asdict(cfg) if cfg is not None else None,
            sims=sims,
        )
    finally:
        if mirror is not None:
            mirror._flush()
    for level, name, message in out["log"]:
        getattr(logging.getLogger(name), level)("%s", message)
    if out["advice"] is not None:
        return out["advice"]
    return _other_backend_screen_advice(
        out["state"],
        advice_for,
        other["policy"],
        session=session,
        mirror=mirror,
        cfg=cfg,
        sims=sims,
    )


def _other_backend_screen_advice(
    state: dict[str, Any],
    advice_for: str,
    policy: Any,
    *,
    session: dict[str, Any] | None,
    mirror: Any,
    cfg: RecommendConfig | None,
    sims: int | None,
) -> dict[str, Any]:
    """:func:`compute_screen_advice` from the policy lookup on, for a policy
    that is not the Rust net: the Python decision builders it was paired
    with. ``state`` has had the session applied.
    """
    import realgame_ppo
    import sts2_rs
    from display_names import _names

    standin_note = ""
    # A hand-back surface whose champion WAS trained on it (GH #978).
    trained_surface = False
    if advice_for == "card_select":
        # `_build_card_select` keys the deck-mutation surface off the screen's
        # own prompt_key first (TO_UPGRADE/TO_REMOVE/...), which the mod
        # captures via CardSelectContextPatch, so prev_msg_type is only the
        # fallback and None is fine here. It is the one surface the driver's
        # builder table does not carry (it needs that extra argument).
        decision = realgame_ppo._build_card_select(state, None)
    elif advice_for == "card_bundle":
        decision = _build_card_bundle(state)
        if _trained_on_packs(policy):
            trained_surface = True
        else:
            standin_note = "packs scored by their first card; this net was not trained on packs"
    else:
        # Everything else goes through the driver's own builder table, so a
        # surface the PPO driver learns to handle is one the advisor can
        # explain without a second dispatch to keep in sync.
        head, standin_note = _SURFACE_STANDINS.get(advice_for, (advice_for, ""))
        if advice_for == "potion_swap" and not realgame_ppo._live_potion_swap_enabled():
            return _decline_potion_swap(state)
        builder = _advisor_builders().get(head) or realgame_ppo._PPO_NONCOMBAT_BUILDERS.get(head)
        if builder is None:
            return _unranked(
                state, f"no trained head for {advice_for}", trigger=f"no_head:{advice_for}"
            )
        build_state = state if head == advice_for else {**state, "type": head}
        # This backend keeps the Python builders it was paired with (the Rust
        # net reads the Rust builders' observation, GH #1426 step 2).
        decision = builder(build_state)

    if decision is None or not decision.actions:
        # In-combat card selections (Burning Pact's Exhaust, Gambler's Brew's
        # replace, Dual Wield's copy) land here: no head was ever trained on
        # them, and `_unranked` answers them with the LEFTMOST card, which
        # exhausted a Strike+ over its unupgraded twin in 8 of 8 picks during
        # the ascension climb.
        #
        # The ladder is now three rungs, best first:
        #   1. the combat MCTS, when the fight's mirror is carrying the
        #      choice with its suspended tail (`_searched_card_select_advice`);
        #   2. `realgame_hand_select`, which decides only the one thing the
        #      leftmost default gets wrong -- whether the pick is punished or
        #      rewarded -- and returns None when it cannot tell;
        #   3. `_unranked`'s admitted arbitrary pick.
        # Each rung says in its note which one answered.
        if advice_for == "card_select":
            searched = _searched_card_select_advice(state, mirror, cfg, sims)
            if searched is not None:
                return searched
            if _selected_count(state) > 0:
                # Asked again after a card of a select that takes several
                # (GH #1770): the heuristic's one pick is taken, so confirm
                # rather than pick another "worst card" each time (Rust:
                # `ScreenEnv::in_fight_select`).
                return _confirm_taken(state)
            ranked = realgame_hand_select.rank(state)
            if ranked is not None:
                return ranked
        return _unranked(
            state,
            f"{advice_for}: the net does not cover this surface",
            trigger="card_select" if advice_for == "card_select" else f"no_decision:{advice_for}",
        )

    if len(decision.actions) == 1:
        probs, value = [1.0], None
    else:
        try:
            probs, value = policy.distribution(decision.obs, decision.actions, decision.metrics)
        except Exception as exc:
            # The fixed-shape ONNX backend can exceed its export budget
            # (schema.BudgetExceededError, e.g. a 103-option shop removal --
            # GH #315), and any backend can fail at inference. Advice must
            # always render, so degrade to the unranked shape.
            logger.warning("run-policy inference failed for %s: %s", advice_for, exc)
            return _unranked(
                state,
                f"{advice_for}: run-policy inference failed ({exc})",
                trigger="inference_failed",
            )

    carried = (
        realgame_ppo.shop_removal_carry_index(state, session, decision)
        if advice_for == "card_select"
        else None
    )
    # The shop already chose WHICH card to remove; this screen only executes
    # it. Re-ranking here would answer a different question on a differently
    # trained surface -- see `realgame_ppo.shop_removal_carry_index`. The
    # driver has honoured the shop pick since 2026-07-16; the advisor never
    # did, because it does not route through `pick_noncombat_action_ppo`, so
    # every live shop removal was re-decided (0 carries across
    # `logs/f7-mirror-check18` and `check19`).
    #
    # The rest is `sts2_advisor::screen_advice::rank`: the pick is decoded
    # the way the run agent decodes (the A8 §8 group-mass rule -- pool the
    # mass of equivalent candidates, e.g. one shop removal per deck card,
    # then take the best member; a plain argmax let a split "remove a
    # starter" preference lose to a single stronger-looking buy), map options
    # carry their graph coordinate so the HUD can pin the score to the node,
    # and a forced pick (a single option, which never consulted the net) is
    # not announced as a ranking (GH #399).
    advice, best_i = sts2_rs.screen_rank(
        state,
        advice_for,
        list(decision.actions),
        list(decision.responses),
        [str(x) for x in decision.labels],
        decision.source_note,
        [float(p) for p in probs],
        None if value is None else float(value),
        _names(),
        carried=carried,
        trained_surface=trained_surface,
        standin_note=standin_note,
        fight_offer=is_fight_offer(state),
    )
    if session is not None and advice_for == "shop":
        realgame_ppo.stash_shop_removal_target(
            session, state.get("run_state") or {}, decision.actions[best_i]
        )

    if advice_for == "map_select":
        path = _project_map_path(state, policy, best_i, hp_table=_projection_hp_table())
        if path:
            advice["path"] = path
            advice["note"] += _route_note(path)

    return advice


# Defensive bound only. The act DAG ends itself at the final row (~15 rows),
# so this never binds on a real map; it exists so a malformed graph cannot
# wedge the walk. It used to be 8 -- a fig leaf over a projection that froze
# HP and drifted without bound. Now that HP is walked forward the route can
# run the whole act, which is what the HUD needs to show.
MAX_PROJECTED_HOPS = 32


def _projection_hp_table() -> dict[str, Any] | None:
    """The run-AI's expected-HP-cost tables, straight from the engine.

    `data/runai/heuristic_v.toml` is `include_str!`'d into the Rust binary, so
    asking `sts2_rs` for it keeps the HUD's route and the planner's own
    projection on one set of numbers. A Python-side copy would be a second
    source of truth that nothing gates.

    `None` when the engine cannot answer (older wheel, extension missing) --
    the route then draws frozen, as it did before, rather than not at all.
    """
    try:
        import sts2_rs

        return dict(sts2_rs.projection_hp_table())
    except Exception as exc:  # never let a drawing cost the advice
        logger.debug("projection tables unavailable; route will not decay HP: %s", exc)
        return None


def _route_note(path: list[dict[str, Any]]) -> str:
    """The one-line route caption under the map advice."""
    note = f" · route {len(path)} floors"
    arrival = path[-1].get("hp") if path else None
    if arrival is not None:
        # Naming the frozen axes is the point: a player who knows only HP moves
        # can discount the late hops themselves.
        note += f" → {arrival} HP (deck/gold/relics frozen)"
    return note


def _project_hp(
    hp: float,
    max_hp: float,
    room: str,
    act: int,
    table: dict[str, Any] | None,
) -> float:
    """HP after `room` resolves, from the run-AI's expected-cost tables."""
    if not table:
        return hp
    slot = min(max(act - 1, 0), 2)
    # The bridge sends the C# `MapPointType` enum name (``RestSite``); the
    # training-side obs uses the snake form (``rest_site``). Fold both.
    kind = str(room).lower().replace("_", "")
    if kind == "monster":
        hp -= float(table["monster_hp_loss"][slot])
    elif kind == "elite":
        hp -= float(table["elite_hp_loss"][slot])
    elif kind == "ancient":
        # Event rooms are a chance node; the table's flat prior is the
        # expectation over the pool, matching `projection.rs`.
        hp -= float(table["event_hp_loss"])
    elif kind == "unknown":
        hp -= float(table["unknown_hp_loss"])
    elif kind == "restsite":
        # The engine's own floor(max_hp * 0.3). Projecting the campfire as a
        # smith instead would leave HP flat, and the smith case is the one
        # the player can see for themselves.
        hp += math.floor(max_hp * float(table["rest_heal_frac"]))
    # A drawing, not a death sentence: the planner keeps the negative deficit
    # to score dead paths as dominated, but a HUD line has nothing to do with
    # a projected -40, and the net has never seen an HP below 1.
    if max_hp > 0:
        hp = min(hp, max_hp)
    return max(hp, 1.0)


def route_graph(state: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """The act's map by coordinate, for a route walk, start and boss included.

    The mod's ``map.nodes`` hold rows 1..N only: the act's start (row 0, the
    Ancient) shows up just as the offered node at an act's entrance, and the
    boss just as ``map.boss``. Without them the route at an entrance was empty
    and every route ended a node short of the boss. The game links the start
    to every row-1 node and the last row to the boss (StandardActMap.cs:221-232);
    the last row's ``children`` already name the boss.
    """
    m = state.get("map") or {}
    by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    for n in m.get("nodes") or []:
        try:
            by_coord[int(n["col"]), int(n["row"])] = n
        except (KeyError, TypeError, ValueError):
            continue
    if not by_coord:
        return by_coord
    boss = m.get("boss")
    if isinstance(boss, dict):
        try:
            coord = (int(boss["col"]), int(boss["row"]))
        except (KeyError, TypeError, ValueError):
            coord = None
        if coord is not None and coord not in by_coord:
            by_coord[coord] = {"col": coord[0], "row": coord[1], "type": "Boss", "children": []}
    first_row = min(r for _, r in by_coord)
    for n in state.get("nodes") or []:
        try:
            coord = (int(n["col"]), int(n["row"]))
        except (KeyError, TypeError, ValueError):
            continue
        if coord not in by_coord and coord[1] < first_row:
            by_coord[coord] = {
                **n,
                "children": [{"col": c, "row": r} for (c, r) in by_coord if r == first_row],
            }
    return by_coord


def _project_map_path(
    state: dict[str, Any],
    policy: Any,
    best_index: int,
    hp_table: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Walk the net forward from its chosen node to sketch the route.

    The map surface only ever ranks the CHILDREN of where you stand, so a
    single query answers "which node next", never "which way through the
    act" -- and a bare ``(4,1)`` on the HUD tells a player nothing about
    where that leads. This re-queries the same policy hop by hop, moving
    ``map_coord`` along the graph the bridge already sent us.

    HP is walked forward through :func:`_project_hp` between hops, so hop *n*
    is answered at the HP the player would really arrive with. That is not a
    nicety: the champion rest-seeks 95% of the time below half HP, so a walk
    that freezes HP at its live value is structurally blind to campfires and
    systematically under-draws them (the projection used to stop at 8 hops
    precisely to hide how far that drifted).

    It is still a PROJECTION, not a plan. Deck, gold, relics and potions stay
    frozen, and the HP deltas are the run-AI's flat per-act expectations, not
    this deck against that encounter -- so the route is plausible, not
    accurate. :func:`_route_note` says so on the HUD.
    """
    import realgame_ppo

    if _scores_rust_obs(policy):
        # The Rust net walks in Rust (`screen_advice::project_map_path`), on
        # the Rust-built observation it scores everywhere else. Another
        # backend keeps the walk below, on the Python builders' observation.
        import sts2_rs

        if best_index < 0:
            return []
        return list(sts2_rs.screen_map_path(state, best_index, policy._net))

    m = state.get("map") or {}
    graph = m.get("nodes") or []
    offered = state.get("nodes") or []
    if not graph or not (0 <= best_index < len(offered)):
        return []

    by_coord = route_graph(state)

    run_state = state.get("run_state") or {}
    base_obs = realgame_ppo._base_obs(state, run_state)
    base_metrics = realgame_ppo._base_metrics(state, run_state)
    base_floor = int(base_metrics.get("total_floor", 0) or 0)

    chosen = offered[best_index]
    try:
        cur_coord = (int(chosen["col"]), int(chosen["row"]))
    except (KeyError, TypeError, ValueError):
        return []

    hp = float(base_metrics.get("hp", 0) or 0)
    max_hp = float(base_metrics.get("max_hp", 0) or 0)
    act = int(base_metrics.get("act_index", 1) or 1)

    path: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for hop in range(MAX_PROJECTED_HOPS):
        node = by_coord.get(cur_coord)
        if node is None or cur_coord in seen:
            break
        seen.add(cur_coord)
        room = node.get("type", "?")
        step = {
            "col": cur_coord[0],
            "row": cur_coord[1],
            "type": room,
            "hop": hop,
        }
        # Only claim an arrival HP when there is a cost model behind it. With
        # no table the walk is frozen, and reporting the live HP as "what you
        # arrive with" would be a worse lie than saying nothing.
        if hp_table:
            step["hp"] = round(hp)
        path.append(step)
        hp = _project_hp(hp, max_hp, room, act, hp_table)

        children = []
        for c in node.get("children") or []:
            try:
                coord = (int(c["col"]), int(c["row"]))
            except (KeyError, TypeError, ValueError):
                continue
            if coord in by_coord:
                children.append(coord)
        if not children:
            break
        if len(children) == 1:
            cur_coord = children[0]
            continue

        obs = dict(base_obs)
        obs["map_coord"] = cur_coord
        metrics = dict(base_metrics)
        metrics["total_floor"] = base_floor + hop + 1
        metrics["act_floor"] = cur_coord[1] + 1
        metrics["hp"] = round(hp)
        actions = [{"action": "move", "col": col} for col, _row in children]
        try:
            probs, _ = policy.distribution(obs, actions, metrics)
        except Exception as exc:  # a projection must never break the advice
            logger.debug("map projection stopped at hop %d: %s", hop, exc)
            break
        cur_coord = children[max(range(len(probs)), key=probs.__getitem__)]

    return path


class LastCombat:
    """The most recent combat decision's native search state, for companion
    searches (the workbench's on-demand SearchTree explorer). Thread-safe:
    written by the serve loop, read by the control plane."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_id: Any = None
        self._cs: Any = None
        self._cfg: RecommendConfig | None = None

    def put(self, state_id: Any, cs: Any, cfg: RecommendConfig) -> None:
        with self._lock:
            self._state_id, self._cs, self._cfg = state_id, cs, cfg

    def clear(self) -> None:
        with self._lock:
            self._state_id = self._cs = self._cfg = None

    def get(self) -> tuple[Any, Any, RecommendConfig | None]:
        with self._lock:
            return self._state_id, self._cs, self._cfg


#: Singleton written by ``_handle_advice_request``.
LAST_COMBAT = LastCombat()


def companion_search_tree(cs: Any, cfg: RecommendConfig, sims: int) -> dict[str, Any]:
    """A single-tree search over ``cs`` in the workbench's ``search_tree``
    payload shape, pruned to keep log records bounded.

    This is a COMPANION search (fresh single tree), not the advice's own
    beliefdraw tree — same caveat as the workbench's live explorer.
    """
    import sts2_rs

    tree = sts2_rs.mcts_search_tree(
        cs,
        sims=sims,
        c_puct=cfg.resolved_c_puct(),
        depth=cfg.depth,
        seed=cfg.seed,
        rollout_policy=cfg.rollout_policy,
    )
    return prune_tree(tree)


def prune_tree(tree: dict[str, Any], min_visits: int = 2, max_nodes: int = 2000) -> dict[str, Any]:
    """Drop leaf noise so a logged tree stays a record, not a payload bomb.

    Keeps the root and any node with ``visits >= min_visits``, then caps to
    the ``max_nodes`` most-visited (parents of kept nodes are re-added so
    the tree stays connected).
    """
    nodes = list(tree.get("nodes") or [])
    if not nodes:
        return tree
    root = tree.get("root")
    by_id = {n.get("id"): n for n in nodes}
    kept = [n for n in nodes if n.get("id") == root or (n.get("visits") or 0) >= min_visits]
    kept.sort(key=lambda n: (n.get("id") != root, -(n.get("visits") or 0)))
    kept = kept[:max_nodes]
    ids = {n.get("id") for n in kept}
    # re-add ancestors so every kept node reaches the root
    for n in list(kept):
        cur = n
        while True:
            pid = cur.get("parent")
            if pid is None or pid in ids:
                break
            parent = by_id.get(pid)
            if parent is None:
                break
            kept.append(parent)
            ids.add(pid)
            cur = parent
    out = dict(tree)
    out["nodes"] = kept
    out["pruned_from"] = len(nodes)
    return out


class RequestRecorder:
    """Every raw message from the mod, one JSON line each (``--record-requests``).

    Verbatim and unwrapped, unlike :class:`AdvisorLog`: the point is a corpus
    a test can feed back into the request handler, or into Instinct's run
    graft, exactly as the mod sent it. Never raises into the serve loop.
    """

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = out_dir / f"requests-{stamp}.jsonl"
        self._lock = threading.Lock()

    def write(self, msg: dict[str, Any]) -> None:
        try:
            line = json.dumps(msg, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("request recorder: %s", exc)


#: ``--record-requests``' recorder, or ``None``.
RECORDER: RequestRecorder | None = None


class AdvisorLog:
    """Append-only JSONL event log (advice + human actions), one file/session.

    TODO(phase 9): switch to the workbench ``GameLogger`` schema
    ({step, ts, type: meta|state|mcts|action, data}) so ``/api/runs`` can
    load real-game advisor sessions directly.
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"advisor-{stamp}.jsonl"
        self._step = 0

    def write(self, kind: str, data: dict[str, Any]) -> None:
        try:
            record = {"step": self._step, "ts": time.time(), "type": kind, "data": data}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._step += 1
        except Exception as e:
            logger.warning("advisor log write failed: %s", e)


WORKBENCH_URL = "http://127.0.0.1:8077"


def ensure_workbench_running() -> None:
    """Dev convenience for the mod's "Open Workbench" button: if nothing is
    serving :8077, spawn ``python -m workbench`` as a child process.

    Best-effort by design — the advisor must come up (and advise) whether or
    not the workbench does. The child shares our process group, so the mod's
    job object tears it down with the game like everything else we spawn.
    """
    import subprocess
    import sys
    import urllib.request

    try:
        with urllib.request.urlopen(f"{WORKBENCH_URL}/api/health", timeout=1.5) as resp:
            if resp.status == 200:
                logger.info("workbench already running at %s", WORKBENCH_URL)
                return
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent  # sts2-pycli/
    env = dict(os.environ)
    env["STS2_RELOAD"] = "0"
    # Serve the built SPA from :8077 so the mod's one URL is the whole app.
    # Needs a `npm run build` in web/ at least once; without web/dist the
    # mount is skipped and the page 404s (the API still works).
    env["STS2_SERVE_SPA"] = "1"
    if not (root / "web" / "dist").is_dir():
        logger.warning(
            "web/dist missing — the workbench will serve API only. "
            "Run `npm run build` in sts2-pycli/web once."
        )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root / "python"), env.get("PYTHONPATH")) if p
    )
    try:
        subprocess.Popen(
            [sys.executable, "-m", "workbench"],
            cwd=str(root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("spawned workbench (%s) at %s", sys.executable, WORKBENCH_URL)
    except Exception as exc:
        logger.warning("could not spawn workbench: %s", exc)


def build_coach(
    client: Any,
    wlog: Any,
    args: Any,
    policies: Any = None,
    session: dict[str, Any] | None = None,
    executor: Any = None,
    tree: Any = None,
) -> CoachGrader:
    """The connection's live coach, wired to this client and run log.

    Every record goes to the run log; the ``coach`` message goes to the mod
    only while its last ``advice_request`` advertised ``capabilities.coach``
    (the grader gates that). ``request_id`` / ``state_id`` are explicit
    ``None``: the bridge client would otherwise echo the last request's, and a
    coach record answers no request. Records name the run log and the replay
    step of their decision (``log_name`` / ``log_idx``, from ``tree``) so the
    mod can open the workbench right there.
    """

    def _locate(node_id: Any) -> tuple[str | None, int | None]:
        log_idx = tree.log_idx(node_id) if tree is not None else None
        return wlog.log_name, log_idx

    def _send(record: dict[str, Any]) -> None:
        # The same no-raw-ids guard as the advice (`display_names`).
        record = sanitize_coach(record)
        client.send_action(
            {"action": "coach", "request_id": None, "state_id": None, "coach": record}
        )

    def _log(record: dict[str, Any]) -> None:
        logger.info(
            "coach %s %s: %s",
            record.get("kind"),
            record.get("id") or record.get("ids"),
            record.get("verdict", ""),
        )
        wlog.log_coach(record)

    def _map_path(msg: dict[str, Any], chosen: int) -> list[dict[str, Any]] | None:
        if policies is None:
            return None
        import realgame_ppo

        # A copy: this is a past choice, and folding it into the live session
        # moved the run's map position back (GH #1784).
        state = realgame_ppo.apply_session_context(
            {**msg, "type": "map_select"}, None if session is None else dict(session)
        )
        character = (
            character_display_name((state.get("run_state") or {}).get("character")) or "Ironclad"
        )
        policy, _error = policies.get(character)
        if policy is None:
            return None
        return _project_map_path(state, policy, chosen, hp_table=_projection_hp_table())

    return CoachGrader(
        send=_send,
        log=_log,
        executor=executor if executor is not None else GraderThread(),
        catchup_sims=int(getattr(args, "coach_catchup_sims", 2000)),
        budget_s=float(getattr(args, "coach_budget_s", 6.0)),
        map_path=_map_path,
        locate=_locate,
    )


def serve(args: argparse.Namespace) -> None:
    cfg = RecommendConfig(
        sims=args.sims,
        threads=args.threads,
        hidden_info=HONEST_SEARCH["hidden_info"],
        c_puct=HONEST_SEARCH["c_puct"],
        draw_widening_c=HONEST_SEARCH["draw_widening_c"],
    )
    log = AdvisorLog(Path(args.log_dir))
    logger.info("advisor log: %s", log.path)
    if getattr(args, "record_requests", ""):
        global RECORDER
        RECORDER = RequestRecorder(Path(args.record_requests))
        logger.info("recording raw requests to %s", RECORDER.path)
    wlog = WorkbenchRunLogger(args.workbench_logs)
    if args.workbench == "auto":
        ensure_workbench_running()

    # Rewind lane: the live branch tree + the localhost control plane the
    # workbench's /api/bridge/* proxies talk to (D1: this process owns the
    # one bridge socket, so it is the single gateway).
    tree = LiveRunTree()
    control = ControlPlane(tree)
    control.set_logs_dir(args.workbench_logs)

    def _control_search_tree(sims: Any) -> dict[str, Any]:
        state_id, cs, ccfg = LAST_COMBAT.get()
        if cs is None or ccfg is None:
            raise LookupError("no combat decision cached (not in a fight, or no advice yet)")
        tree_payload = companion_search_tree(cs, ccfg, int(sims or 2000))
        tree_payload["state_id"] = state_id
        return tree_payload

    control.search_tree_fn = _control_search_tree
    site = None
    if args.workbench_dir and StaticSite.is_static_build(Path(args.workbench_dir)):
        site = StaticSite(
            Path(args.workbench_dir), ppo_dir=runtime_paths.data_root() / "data" / "ppo"
        )
        logger.info("hosting the browser workbench from %s", args.workbench_dir)
    # The bundled page's Report button (realgame_control.BugReportForward):
    # forwarded to the site the mod passes, else saved beside the logs.
    bug_reports = (
        BugReportForward(args.bug_report_site, Path(args.log_dir).resolve().parent / "bugreports")
        if site is not None
        else None
    )
    control.start(
        args.control_port,
        allow_origins=args.control_allow_origin or (),
        site=site,
        bug_reports=bug_reports,
    )

    # Screen decisions (card reward / card select) are answered by the same
    # run-policy champion the PPO driver uses; loaded lazily per character on
    # the first screen that needs it, and never fatal if absent.
    import realgame_ppo

    # strict=False: unlike a sweep, a live coached run has no measurement to
    # invalidate, and advice must render even with no net at all (it degrades
    # to the unranked shape, saying so).
    policies = realgame_ppo.PpoPolicyCache(getattr(args, "model", None), strict=False)

    # The two pieces of carried context the advisor used to do without. Both
    # outlive a reconnect on purpose: dropping the connection does not rewind
    # the run, and the mirror re-seeds itself on the next combat board anyway.
    mirror = AdvisorMirror(cfg, enabled=not getattr(args, "no_mirror", False))
    session: dict[str, Any] = {}

    while True:
        # A human is allowed to stare at a screen. The default 60 s read
        # timeout is a stall detector for the autopilot lane, where silence
        # really does mean a wedged run; here it tore the connection down
        # every time the player thought for a minute (seven times in the
        # 2026-08-05 session), and any advice_request sent during the gap was
        # simply lost -- the mod saw no answer and the HUD stayed blank.
        client = RealGameBridgeClient(host=args.host, port=args.port, timeout=args.idle_timeout)
        coach: CoachGrader | None = None
        connected = False
        try:
            client.connect()
            connected = True
            logger.info("connected to SpireMentor at %s:%s", args.host, args.port)
            log.write("meta", {"event": "connected", "sims": args.sims, **HONEST_SEARCH})
            control.set_client(client)
            coach = build_coach(client, wlog, args, policies, session, tree=tree)
            _serve_connected(
                client, cfg, args, log, wlog, policies, tree, control, mirror, session, coach
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning("connection lost/failed: %s; retrying in 3s", e)
            if connected:
                # A session dropped mid-run (not a connect that never got
                # through): the mod's unread requests went with it.
                wlog.note_reconnect()
            time.sleep(3)
        finally:
            if coach is not None:
                coach.executor.stop()
                coach = None
            control.set_client(None)
            client.disconnect()


def _serve_connected(
    client: RealGameBridgeClient,
    cfg: RecommendConfig,
    args: argparse.Namespace,
    log: AdvisorLog,
    wlog: WorkbenchRunLogger,
    policies: Any,
    tree: LiveRunTree,
    control: ControlPlane,
    mirror: AdvisorMirror,
    session: dict[str, Any],
    coach: CoachGrader | None = None,
) -> None:
    while True:
        msg = client.receive_state()
        mtype = msg.get("type")
        if RECORDER is not None:
            RECORDER.write(msg)
        warm_policy_async(policies, (msg.get("run_state") or {}).get("character"))
        control.note_message(msg)

        if mtype == "advice_request" and client.newer_advice_request_pending(msg.get("advice_for")):
            # A board that no longer exists: the mod already asked again (a
            # teammate's next card, #1208) and drops any reply to this one.
            # Answer the newer request instead of paying for this search.
            logger.info(
                "advice: skipped request %s (%s), a newer one is queued",
                msg.get("request_id"),
                msg.get("advice_for"),
            )
            continue
        if mtype == "advice_request":
            _handle_advice_request(
                client, cfg, args, log, wlog, msg, policies, tree, mirror, session, coach
            )
        elif mtype == "human_action":
            # The position the action was taken FROM, for the coach's catch-up
            # search -- it must be cloned before the mirror follows the move.
            pre_state = None
            if coach is not None and mirror.live is not None:
                try:
                    pre_state = mirror.live.state.clone()
                except Exception as exc:
                    logger.debug("coach: cannot clone the mirror: %s", exc)
            # Follow the world BEFORE logging: the mirror is only a mirror
            # while it is advanced through what actually happened.
            mirror.on_world_action(msg)
            INSTINCT.on_human_action(msg)
            logger.info("human: %s", msg.get("detail") or msg.get("action"))
            log.write("action", msg)
            wlog.log_human_action(msg)
            node_id = tree.on_human_action(msg)
            if coach is not None:
                coach.on_human_action(msg, pre_state, node_id)
        elif mtype in BRIDGE_MESSAGE_TYPES:
            # Rewind lane (snapshots / snapshot_saved / snapshot_restored /
            # error): the control plane routes them, including into the tree.
            logger.debug("%s: %s", mtype, msg)
            if mtype in ("snapshot_restored", "combat_undone"):
                # Both carry a fresh state_id precisely because everything
                # older is stale (PROTOCOL.md section 7), and the mod
                # broadcasts them for ANY rewind including the in-game
                # toolbar's. A carried combat state is exactly that kind of
                # stale -- and `combat_undone` keeps the fight running while
                # moving the board backwards, which is the case that would
                # corrupt it silently.
                mirror.on_rewind(msg)
                INSTINCT.reset(mtype)
            control.on_bridge_message(msg)
            if coach is not None and mtype in ("snapshot_restored", "combat_undone"):
                # AFTER the tree moved its active pointer: grades for nodes
                # the rewind abandoned are withdrawn from the mod and the log.
                coach.on_rewind(msg, tree.active_path())
        elif mtype == "pong":
            logger.debug("%s: %s", mtype, msg)
        elif mtype == "combat_ended":
            # The fight's end (PROTOCOL.md v5.1): only the coach's recap
            # needs it. The mirror catches up from the next board and the
            # tree / run log record decisions, not fight boundaries.
            logger.info("combat_ended: %s", {k: msg.get(k) for k in ("victory", "hp", "round")})
            log.write("meta", {"event": mtype, **msg})
            if coach is not None:
                coach.on_combat_ended(msg)
        elif mtype in ("game_over", "run_complete"):
            logger.info("%s: %s", mtype, {k: msg.get(k) for k in ("victory", "floor", "act")})
            log.write("meta", {"event": mtype, **msg})
            wlog.log_run_end(msg)
            tree.on_run_end(msg)
            LAST_COMBAT.clear()
            mirror.drop("run over")
            INSTINCT.reset("run over")
            logger.info("advisor %s", mirror.note())
            session.clear()
            if coach is not None:
                if is_death(msg):
                    # The autopsy is built behind the grading still in flight
                    # and appended to the run log this branch just closed.
                    coach.on_game_over(msg)
                else:
                    coach.on_run_end()
        else:
            # Autopilot-style decision messages (combat_action, map_select,
            # run_config, ...) mean this advisor is connected to a session
            # that expects a DRIVER. Never answer them — log and ignore.
            logger.warning("ignoring non-advisor message type=%r (advisor never drives)", mtype)

        # AFTER the handler, so the file this message just opened (or rotated
        # to) is the one published. Every message, not only advice requests:
        # the workbench's Follow mode asks /status for this path and silently
        # guesses when it is absent, so publishing it once the first advice
        # lands left a window where Follow tailed some other file -- and a
        # human_action, which opens the log just as well, never closed it.
        publish_log_path(control, wlog)


def is_death(msg: dict[str, Any]) -> bool:
    """A ``game_over`` the player lost (PROTOCOL.md section 3: ``result``).

    Only a death gets the coach's autopsy; a victory or a ``run_complete``
    (which also covers an abandoned run, ``terminated``) does not.
    """
    if msg.get("type") != "game_over":
        return False
    return msg.get("result") == "defeat" or msg.get("victory") is False


def publish_log_path(control: Any, wlog: Any) -> None:
    """Publish the run log the control plane serves at ``GET /log``.

    Only an OPEN log replaces the published path: once ``game_over`` closes
    the file the last one stays served until the next run opens its own, so
    the autopsy's "open in workbench" links (which arrive just then) still
    reach their file. Follow mode reloads on a path change, so a finished
    file left published is harmless.
    """
    if wlog.path is not None:
        control.set_log_path(wlog.path)


def _cache_for_coach(
    coach: CoachGrader,
    msg: dict[str, Any],
    advice_for: str,
    state: dict[str, Any],
    advice: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    cs: Any,
    call_cfg: Any,
) -> None:
    """Keep this request's raw result for grading the move made from it."""
    try:
        if advice_for == "combat_action":
            if rows is None:
                return  # the search failed: nothing to grade against
            entry = coach.cache.put(
                msg.get("request_id"),
                msg.get("state_id"),
                kind="combat",
                rows=rows,
                cs=cs,
                cfg=call_cfg,
                advice=advice,
                pv_labels=list(advice.get("pv") or []),
                msg=state,
            )
        elif advice_for in SCREEN_ADVICE_FOR and advice.get("options"):
            entry = coach.cache.put(
                msg.get("request_id"),
                msg.get("state_id"),
                kind="screen",
                options=list(advice.get("options") or []),
                advice=advice,
                msg=state,
            )
        else:
            return
        coach.on_advice(entry)
    except Exception as exc:  # bookkeeping must never cost the advice loop
        logger.warning("coach: caching advice failed: %s", exc)


def advisor_capabilities() -> dict[str, Any]:
    """``capabilities`` on every reply: the coach level, plus what Instinct plays.

    ``instinct`` lists ``{character, game_version}`` pairs (PROTOCOL.md §5,
    v5.2); the mod shows its Search / Instinct switch only for a run it
    lists. Empty when no Instinct net loaded.
    """
    return {**ADVISOR_CAPABILITIES, "instinct": INSTINCT.capabilities()}


def mark_instinct_unavailable(advice: dict[str, Any], reason: str) -> dict[str, Any]:
    """The search's advice, answered in Instinct's place: say so, and why."""
    out = dict(advice)
    out["brain"] = "search"
    out["instinct_unavailable"] = reason
    note = out.get("note")
    out["note"] = f"Search (Instinct: {reason})" + (f" · {note}" if note else "")
    return out


def mark_beta_build(advice: dict[str, Any], game_version: str | None) -> dict[str, Any]:
    """Label advice on a partly ported beta build as approximate.

    The engine runs the beta on the previous build's rules wherever it has not
    ported the beta's changes yet, so the ranking can be off; the player
    should know. Refusals and advisor errors (``hand_back``) carry no ranking
    and stay as they are.
    """
    if game_version not in BETA_GAME_VERSIONS:
        return advice
    if any(k in advice for k in ("unsupported_game_version", "unsupported_character", "hand_back")):
        return advice
    out = dict(advice)
    label = f"Beta {game_version}: approximate advice"
    note = out.get("note")
    out["note"] = label + (f" · {note}" if note else "")
    out["beta_game_version"] = game_version
    return out


def refusal_advice(exc: UnsupportedGameVersionError) -> dict[str, Any]:
    """The advice for a board from a game build this advisor cannot simulate.

    No ranking and no ``best``: there is no engine for that build, and a search
    on the wrong one would rank moves by rules the game is not using. The
    ``note`` is the player-facing explanation the mod shows in the HUD;
    ``unsupported_game_version`` names the build for tooling.
    """
    return {"note": str(exc), "unsupported_game_version": exc.raw}


def note_game_version(session: dict[str, Any] | None, game_version: str, mirror: Any) -> None:
    """Track the game build across requests; a change invalidates carried state.

    The mirror's carried combat state and the cached search board were built
    for the previous build's engine, so neither survives a build change.
    """
    if session is None:
        return
    previous = session.get("game_version")
    if previous == game_version:
        return
    if previous is None:
        logger.info("game build: %s", game_version)
    else:
        logger.warning(
            "game build changed %s -> %s; dropping carried state", previous, game_version
        )
        LAST_COMBAT.clear()
        if mirror is not None:
            mirror.drop(f"game build changed {previous} -> {game_version}")
        INSTINCT.reset("game build changed")
    session["game_version"] = game_version


def _handle_advice_request(
    client: RealGameBridgeClient,
    cfg: RecommendConfig,
    args: argparse.Namespace,
    log: AdvisorLog,
    wlog: WorkbenchRunLogger,
    msg: dict[str, Any],
    policies: Any,
    tree: LiveRunTree | None = None,
    mirror: Any = None,
    session: dict[str, Any] | None = None,
    coach: CoachGrader | None = None,
) -> None:
    request_id = msg.get("request_id")
    state_id = msg.get("state_id")
    advice_for = msg.get("advice_for") or "combat_action"
    if coach is not None:
        coach.on_advice_request(msg)
    rows: list[dict[str, Any]] | None = None
    cs: Any = None
    call_cfg: Any = None
    # State first, advice second: the mcts record attaches to the most recent
    # state line (TimelineStore contract), and the state must survive even if
    # the search below throws.
    logged_step = wlog.log_decision(msg, advice_for)
    if logged_step is not None and tree is not None:
        tree.on_decision(msg, advice_for, log_idx=logged_step)
    search_tree_log: dict[str, Any] | None = None
    state: dict[str, Any] = msg
    t0 = time.monotonic()
    try:
        # The build the mod reports (absent = v0.107.1). The combat engine is
        # built for it downstream (`to_headless_combat_play` -> `$.game_version`
        # -> `state_from_dict`); an unknown build is refused before any search.
        note_game_version(session, envelope_game_version(msg), mirror)
        check_supported_characters(msg)
        state = dict(msg)
        state["type"] = advice_for
        is_boss = (state.get("room_type") or "") == "Boss"
        sims = args.boss_sims if is_boss else args.sims
        fight_cfg = fight_search_cfg(cfg, state)
        # The player's brain (PROTOCOL.md §5, v5.2). Instinct answers when it
        # can read the position; otherwise the search below answers and says
        # why, so autoplay never stalls on a position Instinct cannot read.
        instinct_advice: dict[str, Any] | None = None
        instinct_reason: str | None = None
        if msg.get("brain") == "instinct":
            try:
                instinct_advice = INSTINCT.advise(
                    state, advice_for, cfg=fight_cfg, mirror=mirror, session=session, sims=sims
                )
            except InstinctUnavailable as e:
                instinct_reason = str(e)
                logger.info("instinct unavailable (%s): %s", advice_for, e)
            except Exception as e:
                missing = engine_missing_content(e)
                if missing is not None:
                    instinct_reason = missing
                    logger.info("instinct unavailable (%s): %s", advice_for, missing)
                else:
                    instinct_reason = "Instinct hit an error"
                    logger.error("instinct failed: %s", e, exc_info=True)
        if instinct_advice is not None:
            advice = instinct_advice
            if advice_for == "combat_action":
                advice = apply_ally_wait(state, advice)
        elif advice_for in SCREEN_ADVICE_FOR:
            advice = compute_screen_advice(
                state,
                advice_for,
                policies,
                session=session,
                mirror=mirror,
                cfg=fight_cfg,
                sims=sims,
            )
        elif advice_for == "combat_action":
            advice, cs, call_cfg, rows = compute_combat_advice(
                state, fight_cfg, sims, mirror, getattr(args, "plan_budget_s", None)
            )
            advice = apply_ally_wait(state, advice)
            LAST_COMBAT.put(state_id, cs, call_cfg)
            if args.log_search_tree > 0 and logged_step:
                try:
                    search_tree_log = companion_search_tree(cs, call_cfg, args.log_search_tree)
                except Exception as exc:
                    logger.warning("search-tree logging failed: %s", exc)
        else:
            raise ValueError(f"unsupported advice_for={advice_for!r}")
        if instinct_reason is not None:
            advice = mark_instinct_unavailable(advice, instinct_reason)
    except UnsupportedGameVersionError as e:
        logger.error("advice_request refused: %s", e)
        advice = refusal_advice(e)
    except UnsupportedCharacterError as e:
        logger.warning("advice_request refused: %s", e)
        advice = {"note": str(e), "unsupported_character": e.raw}
    except Exception as e:
        # With the traceback: this handler renders "advisor error" into the
        # HUD and moves on, so without one a live failure leaves only a
        # message and no way to find it. `object of type 'NoneType' has no
        # len()` cost a whole debugging round on 2026-08-24 for exactly that
        # reason.
        logger.error("advice_request failed: %s", e, exc_info=True)
        advice = {"note": f"advisor error: {e}", "hand_back": "the advisor hit an error"}

    # The wire boundary: no raw game id reaches the HUD, from either brain or
    # any surface (`display_names.sanitize_advice`; the cross-checked
    # `best.label` stays the mod's own spelling).
    try:
        advice = sanitize_advice(advice, advice_for, msg)
    except Exception as exc:  # the guard must never cost the advice
        logger.warning("advice label guard failed: %s", exc)
    # An unreadable build was already answered (and refused) above.
    with contextlib.suppress(UnsupportedGameVersionError):
        advice = mark_beta_build(advice, envelope_game_version(msg))
    elapsed = time.monotonic() - t0
    reply = {
        "action": "advice",
        "request_id": request_id,
        "state_id": state_id,
        "advice": advice,
        "capabilities": advisor_capabilities(),
    }
    try:
        client.send_action(reply)
    except ConnectionError:
        raise
    if coach is not None:
        # AFTER the reply: the coach's bookkeeping never delays advice.
        _cache_for_coach(coach, msg, advice_for, state, advice, rows, cs, call_cfg)
    log.write(
        "mcts",
        {
            "request_id": request_id,
            "state_id": state_id,
            "elapsed_s": round(elapsed, 2),
            "advice": advice,
        },
    )
    if logged_step:
        wlog.log_advice(msg, advice, elapsed, tree=search_tree_log)
    best_label = (advice.get("best") or {}).get("label")
    # The note is where an answer admits it is not a ranking ("defaulting to
    # the first option, NOT a recommendation", "heuristic pick: ..."). It was
    # reaching the HUD and nothing else, so an arbitrary pick read exactly like
    # a trained one in every log we keep -- which is how the leftmost-card
    # selection bug survived a whole climb unnoticed. Log it.
    note = advice.get("note")
    logger.info(
        "advice #%s: best=%s p_win=%s (%.1fs)%s",
        request_id,
        best_label,
        advice.get("p_win"),
        elapsed,
        f" [{note}]" if note else "",
    )


#: A synthetic `map_select` decision for the packaged-exe smoke check. Small,
#: character-agnostic (unknown ids land in the encoder's OOV slots) and enough
#: to drive one full forward pass.
_SMOKE_STATE: dict[str, Any] = {
    "type": "map_select",
    "floor": 5,
    "act": 1,
    "run_state": {
        "character": "Ironclad",
        "hp": 60,
        "max_hp": 75,
        "gold": 120,
        "ascension": 0,
        "master_deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}],
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [],
    },
    "nodes": [
        {"index": 0, "type": "Monster", "row": 6, "col": 1},
        {"index": 1, "type": "Shop", "row": 6, "col": 3},
    ],
    "map": {
        "row_count": 10,
        "col_count": 6,
        "current": {"col": 2, "row": 5},
        "boss": {"col": 2, "row": 15},
        "visited": [],
        "nodes": [],
    },
}


class _SmokeSkip(Exception):
    """A smoke step with nothing to check in this build."""


def _smoke_instinct() -> str:
    """The ``--smoke`` Instinct step: the bundled weights, the tokenizer's
    baked-in tables and the forward, through 20 real decisions of the
    campaign run -- the same path a live request takes past the graft.

    The advisor's own student lives inside ``sts2_rs.InstinctAdvisorCore``
    (GH #1426 step 4) and is not reachable from Python, so the self-play
    loads the same weights file into a second ``sts2_rs.E2eStudent``. The
    advisor's own load is checked through ``INSTINCT.available``.

    Raises :class:`_SmokeSkip` for a build without the net (the public export
    ships none, and the spec says so loudly at build time).
    """
    import sts2_rs

    weights = INSTINCT._weights
    if not weights.is_file():
        print(f"instinct SKIP (no net at {weights})")
        raise _SmokeSkip
    if not INSTINCT.available:
        raise RuntimeError(f"the advisor could not load {weights} (see the log above)")
    student = sts2_rs.E2eStudent(str(weights))
    played = sts2_rs.e2e_selfplay(student, 170000000, max_decisions=20)
    if played.get("decisions") != 20 or played.get("error"):
        raise RuntimeError(f"self-play stopped early: {played}")
    return f"instinct OK ({weights.name}, 20 decisions)"


def _smoke() -> int:
    """Deterministic self-check of the packaged exe (build gate, GH #380).

    Exercises every PyInstaller landmine without a game or a bridge: the
    ``sts2_rs`` .pyd, the bundled localization datas, EVERY bundled ONNX
    champion through the full ``load_ppo_policy`` resolver, one real forward
    pass per champion, and a socket bind. Prints ``SMOKE OK`` and exits 0
    only if all four pass.

    The forward pass is the point of the third gate, not the load:
    constructing the policy only parses the graph, while ``ppo.vocab``'s
    registry build (``sts2_rs.list_registry_names``), the featurizer and the
    forward pass are first touched at the first real decision -- so a
    bundling or ABI regression in any of them used to ship green and surface
    as "run-policy inference failed" on the player's first card reward.
    Every champion is loaded because the manifest resolves per character,
    and four of the five were never touched.

    The type assertion is load-bearing, not decoration: the bundle ships
    neither numpy nor onnxruntime, so a resolver that fell back to
    :class:`ppo.onnx_policy.OnnxPolicy` would not merely be slower -- it
    would be unimportable, and the advisor's never-crash contract would turn
    that into a run played at random rather than a crash anyone notices.
    """
    import socket

    ok = True

    try:
        import sts2_rs

        print(f"sts2_rs OK (src_hash {sts2_rs.src_hash()})")
    except Exception as e:
        ok = False
        print(f"sts2_rs FAIL: {e}")

    try:
        from sts2_hint import canonical_maps

        n = len(canonical_maps._CARD_LOC)
        if n == 0:
            raise RuntimeError(f"no card titles loaded from {canonical_maps._LOCALIZATION_DIR}")
        print(f"localization OK ({n} card titles)")
    except Exception as e:
        ok = False
        print(f"localization FAIL: {e}")

    try:
        import champions
        import realgame_ppo
        from ppo.rust_policy import RustPolicy

        decision = realgame_ppo._build_map_select(_SMOKE_STATE)
        if decision is None:
            raise RuntimeError("_build_map_select returned None for the smoke state")
        # The manifest must have made it into the bundle -- without it the
        # frozen exe cannot name a champion at all.
        for character in champions.characters():
            policy, error = realgame_ppo.load_ppo_policy(None, character)
            if not isinstance(policy, RustPolicy):
                raise RuntimeError(
                    f"{character}: expected RustPolicy, got {type(policy).__name__}: {error}"
                )
            probs, value = policy.distribution(decision.obs, decision.actions, decision.metrics)
            if len(probs) != len(decision.actions) or abs(sum(probs) - 1.0) > 1e-6:
                raise RuntimeError(f"{character}: bad distribution {probs} (V={value})")
            print(
                f"run policy OK: {character} (T_MAX={policy.t_max}, A_MAX={policy.a_max}, "
                f"p={[round(p, 3) for p in probs]}, V={value:.3f})"
            )
    except Exception as e:
        ok = False
        print(f"run policy FAIL: {e}")

    try:
        print(_smoke_instinct())
    except _SmokeSkip:
        pass
    except Exception as e:
        ok = False
        print(f"instinct FAIL: {e}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            print(f"socket OK (ephemeral port {s.getsockname()[1]})")
    except Exception as e:
        ok = False
        print(f"socket FAIL: {e}")

    print("SMOKE OK" if ok else "SMOKE FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    # The Rust run agent's search, so a win rate measured in Rust is the one
    # the mod plays: ONE tree (`MctsCombatPolicy` never runs root-parallel)
    # at 4x the `eval-cheap` budget — 4000 sims per decision, 16000 in boss
    # fights, elites on the common budget as in `CombatBudget`. 4x is the
    # strongest budget measured on whole runs (A10 17.0% vs 10.8% at
    # 1000/4000, A9 45.2% vs 40.2%; 16x adds only +1.5pp at 4.2x the time —
    # docs/experiments/2026-10-20-combat-pro-ladder-findings.md, "Full-run
    # check"). Cost on one 2.8 GHz core: ~0.7 s per decision, ~2.3 s per boss
    # decision; `--threads` (default 4) splits that across cores.
    #
    # The advisor used to default to 8000/8000 split across up to 8
    # root-parallel trees (2026-08-10, for latency), with no run-level win
    # rate measured. Root-parallel merges its trees by the same pick rule but
    # drops the single-tree lost-position override; the 2026-10-31 gate below
    # measured that at no detectable cost for 4 trees.
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--boss-sims", type=int, default=16000)
    # Root-parallel workers (sts2_rs par_search: sims split across independent
    # trees; clamped to the core count below). 4 trees of 1000 / 4000 sims
    # played the same as the run agent's single tree on whole runs (Ironclad
    # A10, 1000 paired seeds: 26.6% vs 26.7%, -0.10pp, 95% CI [-3.0, +2.8])
    # at 3.6x less wall time per boss decision (0.31 s vs 1.13 s median) --
    # docs/experiments/2026-10-31-advisor-threads-gate.md. 1 = the single tree.
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--log-dir",
        # Frozen exe: the CWD is the install dir (Program Files on a default
        # Steam install) -- default the JSONL logs to a user-writable spot.
        # SimAdvisorProcess passes --log-dir explicitly; this covers manual
        # launches. Dev default is unchanged (CWD-relative, set by the .cmd).
        default=str(runtime_paths.app_data_dir() / "logs" / "realgame_advisor")
        if runtime_paths.is_frozen()
        else "logs/realgame_advisor",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=3600.0,
        help=(
            "Seconds of socket silence before the advisor treats the bridge as "
            "dead. Long by design: silence means a human is thinking, not that "
            "the game is wedged (the autopilot lane has its own stall guard)."
        ),
    )
    parser.add_argument(
        "--workbench-logs",
        # Same split as --log-dir: dev writes where the workbench already
        # reads (`<sts2-pycli>/logs`), the frozen exe writes user-writable.
        default=str(runtime_paths.app_data_dir() / "logs")
        if runtime_paths.is_frozen()
        else str(Path(__file__).resolve().parent.parent / "logs"),
        help="Directory for workbench-schema run logs (GameLogger JSONL; "
        "the workbench's /api/runs reads this). Empty string disables.",
    )
    parser.add_argument(
        "--record-requests",
        default="",
        metavar="DIR",
        help="Append every raw message the mod sends (advice requests with "
        "their full envelope, human actions, rewinds) to DIR/requests-<time>"
        ".jsonl, verbatim. The corpus Instinct's run graft is checked against "
        "(realgame_instinct). Empty = off.",
    )
    parser.add_argument(
        "--instinct-weights",
        default="",
        metavar="PATH",
        help="Instinct's .e2ew weights (default: data/e2e/ under the data root).",
    )
    parser.add_argument(
        "--log-search-tree",
        type=int,
        default=0,
        metavar="SIMS",
        help="Also log a pruned companion MCTS search tree (single-tree, "
        "SIMS sims) with every combat decision's advice record. 0 = off "
        "(the workbench can recompute trees on demand instead).",
    )
    parser.add_argument(
        "--workbench",
        choices=("off", "auto"),
        # The packaged exe ships no workbench/SPA; dev installs have it one
        # `just workbench` away, so auto-start it for the browser button.
        default="off" if runtime_paths.is_frozen() else "auto",
        help="auto: probe the local workbench (:8077) and spawn "
        "`python -m workbench` if it is not running. off: never spawn.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8079,
        help="Localhost control-plane HTTP port (the workbench's /api/bridge/* "
        "proxy and, with --control-allow-origin, the browser-only workbench).",
    )
    parser.add_argument(
        "--control-allow-origin",
        action="append",
        default=None,
        metavar="ORIGIN",
        help="Page origin allowed to call the control plane from a browser "
        "(repeatable), e.g. https://your-workbench.example.com — the browser-only "
        "workbench's live-game views (GH #709). Off by default: without it no "
        "web page can read the control plane, only the same-machine proxy.",
    )
    parser.add_argument(
        "--bug-report-site",
        default=None,
        metavar="ORIGIN",
        help="Workbench site whose /intake/bugreport stores bug reports: the bundled "
        "workbench's Report button is forwarded there (the mod passes the site its "
        "own reports go to). Unset: reports are saved under <log dir>/../bugreports.",
    )
    parser.add_argument(
        "--workbench-dir",
        default=str(runtime_paths.workbench_dir()),
        help="Browser-only workbench build the control plane hosts at / "
        "(the mod's bundled workbench; dev: web/dist when it is the static "
        "build). Empty string disables.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="PPO run-policy model for screen advice (card reward / card "
        "select). A torch checkpoint extension (.pt/.pth/.ckpt/.bin/.tar) "
        "loads the torch NetPolicy (dev); .onnx loads onnxruntime; anything "
        "else is refused rather than guessed. Default: the committed ONNX "
        "champion for the run's character, resolved by "
        "workbench.ppo_policy.onnx_path.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not carry a per-fight engine mirror. Carrying one lets "
        "in-combat card selections be searched by the combat MCTS instead of "
        "a prompt heuristic, and keeps the per-turn counters the wire does "
        "not carry (GH #370, some of which gate what is legal). The mirror is "
        "only sound because `human_action` carries the machine form of a play "
        "(`card` + `target_index`) so the advisor can follow the fight; on "
        "an older mod build that omits those fields every action is "
        "unfollowable and the mirror drops itself rather than drift. Use this "
        "flag to A/B the mirror or to rule it out of a live problem.",
    )
    parser.add_argument(
        "--coach-catchup-sims",
        type=int,
        default=2000,
        help="Sims for the live coach's catch-up search of a position the "
        "player moved past before its advice arrived (run during the enemy "
        "turn on 2 threads, off the advice path).",
    )
    parser.add_argument(
        "--coach-budget-s",
        type=float,
        default=6.0,
        help="Seconds after an end turn within which the coach must finish "
        "grading that turn; past it, or once the next turn's advice is "
        "requested, the turn stays ungraded and silent (never a late chip).",
    )
    parser.add_argument(
        "--plan-budget-s",
        type=float,
        default=None,
        help="Seconds the turn plan may spend re-searching (default: the "
        "engine's TURN_PLAN_BUDGET_S; `inf`: no cap, for a test comparing "
        "replies, since a clock cut depends on the load).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the packaged-exe self-check (no game needed) and exit.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Clamp --threads to the core count: more workers than
    # cores cannot buy wall-clock, and each extra worker still shrinks the
    # per-tree sim budget, so oversubscription is a pure quality loss.
    cores = os.cpu_count() or 1
    if args.threads > cores:
        logger.warning(
            "--threads %d exceeds %d available cores; using %d",
            args.threads,
            cores,
            cores,
        )
        args.threads = cores
    logger.info(
        "combat search: sims=%d boss_sims=%d threads=%d (%d cores, %d sims/tree)",
        args.sims,
        args.boss_sims,
        args.threads,
        cores,
        args.sims // max(1, args.threads),
    )
    if args.instinct_weights:
        global INSTINCT
        INSTINCT = InstinctAdvisor(Path(args.instinct_weights))
    if args.smoke:
        return _smoke()
    try:
        serve(args)
    except KeyboardInterrupt:
        logger.info("advisor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
