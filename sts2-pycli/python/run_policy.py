"""Run-level policy for sts2_rs full-run self-play.

Shared between ``play_full_run_rust.py`` (CLI driver) and the workbench
recorder. Three layers:

1. Combat-action translation: native MCTS recommendation dicts
   (``play`` / ``drink_potion`` / ``choose`` …) → ``RunManager.take_action``
   dicts (``play_card`` / ``drink_potion`` / ``combat_resolve_choice``).
2. ``heuristic_run_action`` — hand-written out-of-combat rules (pathing,
   rewards, shop, rest, forced removals).
3. ``boss_sim_card_reward`` — pick a card reward by *simulating the
   pinned act-boss fight* once per candidate (each offered card + skip)
   with MCTS, then taking the candidate whose fight went best.
"""

from __future__ import annotations

import random
import re
from typing import Any

import sts2_rs
from sts2_hint.recommender import RecommendConfig

# ---------------------------------------------------------------------------
# Combat-action translation
# ---------------------------------------------------------------------------


def combat_action_to_run_action(combat_action: dict[str, Any]) -> dict[str, Any]:
    """Map a native combat action dict (from ``sts2_rs.Mcts.search()`` /
    ``sts2_rs.mcts_recommend``) to the RunAction dict shape that
    ``RunManager.take_action`` expects.

    Combat side keys: ``play`` / ``end_turn`` / ``potion`` / ``drink_potion`` /
    ``channel`` / ``evoke`` / ``choose`` with ``card_index`` / ``target_index``
    / ``slot`` / ``index``.

    RunAction side keys (combat-relevant subset): ``play_card`` /
    ``end_turn`` / ``drink_potion`` / ``combat_resolve_choice`` with
    ``hand_idx`` / ``target_index`` / ``potion_slot`` / ``option``.

    Orb actions (channel/evoke) raise — ``legal_actions`` never emits
    them, so MCTS can't recommend one at the root.

    Caveat: the native ``choose`` serialization collapses an empty (skip)
    selection to ``index == 0``, so a produced ``combat_resolve_choice``
    should be validated against ``rm.available_actions()`` before
    dispatch (see ``validated_choice``).
    """
    kind = combat_action.get("action")
    if kind == "play":
        return {
            "action": "play_card",
            "hand_idx": combat_action["card_index"],
            "target_index": combat_action.get("target_index"),
        }
    if kind == "end_turn":
        return {"action": "end_turn"}
    if kind in ("potion", "drink_potion"):
        return {
            "action": "drink_potion",
            "potion_slot": combat_action["slot"],
            "target_index": combat_action.get("target_index"),
        }
    if kind == "choose":
        return {
            "action": "combat_resolve_choice",
            "option": combat_action.get("index"),
        }
    raise NotImplementedError(
        f"combat action {kind!r} has no RunAction equivalent "
        f"(orb channel/evoke are engine-internal)"
    )


def validated_choice(rm: Any, action: dict[str, Any]) -> dict[str, Any]:
    """Clamp a translated ``combat_resolve_choice`` to a legal one.

    The native serializer can't distinguish skip (``[]``) from option 0
    (``[0]``); both arrive as ``option == 0``. Check the produced action
    against ``rm.available_actions()`` and fall back to any legal
    resolve-choice action (skip included) on mismatch.
    """
    legal = rm.available_actions()
    for a in legal:
        if a.get("action") == "combat_resolve_choice" and a.get("option") == action.get("option"):
            return action
    for a in legal:
        if a.get("action") == "combat_resolve_choice":
            return a
    raise RuntimeError(
        f"MCTS recommended a choice resolution but no combat_resolve_choice "
        f"is legal (phase={rm.phase()}, legal={legal})"
    )


def best_recommend_action(result: dict[str, Any]) -> dict[str, Any] | None:
    """The ``is_best`` root action from an ``sts2_rs.mcts_recommend``
    result (fallback: max visits), or ``None`` on an empty table."""
    best = None
    most_visited = None
    max_visits = -1
    for entry in result.get("actions", []):
        if entry.get("is_best"):
            best = entry
        visits = int(entry.get("visits", 0))
        if visits > max_visits:
            max_visits = visits
            most_visited = entry
    chosen = best or most_visited
    return chosen.get("action") if chosen else None


# ---------------------------------------------------------------------------
# Heuristic run-level policy (out-of-combat decisions)
# ---------------------------------------------------------------------------

# Card names normalized via _norm_card: "CARD.ASCENDERS_BANE" and
# "AscendersBane" both → "ascendersbane".
_CURSE_NAMES = frozenset(
    {
        "ascendersbane",
        "clumsy",
        "decay",
        "doubt",
        "injury",
        "normality",
        "pain",
        "parasite",
        "regret",
        "shame",
        "writhe",
        "necronomicurse",
        "pride",
        "curseofthebell",
    }
)
_STATUS_NAMES = frozenset({"burn", "dazed", "slimed", "void", "wound"})
_BASIC_NAMES = frozenset({"strike", "defend"})

# Pending interactive deck card-selects (Pomander upgrade, New Leaf transform,
# Wood Carvings enchant, RoomFullOfCheese grant) that preempt the room phase
# with a single-kind pick list — handled together (one pick at a time). The live
# workbench enables these via ``interactive_card_*``; the CLI driver doesn't.
_PENDING_SELECT_KINDS = frozenset(
    {"upgrade_deck_card", "transform_deck_card", "enchant_deck_card", "grant_deck_card"}
)


def _norm_card(card_id: str) -> str:
    return card_id.removeprefix("CARD.").replace("_", "").lower()


def _card_tier(card_id: str) -> int:
    """Coarse desirability: curse=0, status=1, basic Strike/Defend=2, rest=3."""
    n = _norm_card(card_id)
    if n in _CURSE_NAMES:
        return 0
    if n in _STATUS_NAMES:
        return 1
    if n in _BASIC_NAMES:
        return 2
    return 3


def _hp_frac(rm: Any) -> float:
    return rm.current_hp / max(1, rm.max_hp)


def _score_room_type(room_type: str, hp_frac: float, gold: int) -> float:
    if room_type == "rest_site":
        return 95.0 if hp_frac < 0.6 else 35.0
    if room_type in ("treasure", "ancient"):
        return 75.0
    if room_type in ("event", "unknown"):
        return 55.0
    if room_type == "shop":
        return 65.0 if gold >= 180 else 25.0
    if room_type == "monster":
        return 50.0
    if room_type == "elite":
        return 70.0 if hp_frac > 0.8 else 5.0
    return 45.0  # boss / future node types


def map_move_options(rm: Any) -> dict[int, dict[str, Any]]:
    """``col → child node`` (``{col,row,type}``) for the current map decision.

    Derived from graph connectivity (the offered moves are the current
    node's children), mirroring ``RustGame._fill_map`` — NOT ``coord row+1``
    arithmetic, which mislabels the act-start decision (no current coord;
    the engine offers the start node's children)."""
    rows = (rm.full_map() or {}).get("rows") or []
    node_at = {(n["col"], n["row"]): n for row in rows for n in row}
    coord = rm.map_coord()
    current = node_at.get(tuple(coord)) if coord is not None else None
    if current is None:
        current = rows[0][0] if rows and rows[0] else None
    children = (current.get("children") or []) if current else []
    return {c["col"]: node_at.get((c["col"], c["row"]), c) for c in children}


def _pick_map_move(rm: Any, acts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Score each reachable next node by its room type vs current HP/gold."""
    options = map_move_options(rm)
    hp, gold = _hp_frac(rm), rm.gold

    def _score(a: dict[str, Any]) -> float:
        node = options.get(a.get("col")) or {}
        # full_map types are PascalCase UI strings ("RestSite") → snake.
        room = _CAMEL_BOUNDARY.sub("_", str(node.get("type") or "")).lower()
        # jitter so equal-score ties don't always go left
        return _score_room_type(room, hp, gold) + rng.random()

    return max(acts, key=_score)


def _pick_card_reward(rm: Any, acts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Take a random non-curse/non-status option we don't already have 3 of;
    skip otherwise. Bundle picks (Scroll Boxes) choose a random pack."""
    picks = [a for a in acts if a.get("action") == "pick_card_reward" and a.get("idx") is not None]
    if rm.card_reward_bundles() is not None:
        return rng.choice(picks) if picks else rng.choice(acts)
    ids = rm.card_reward_option_ids()
    deck_counts: dict[str, int] = {}
    for c in rm.master_deck():
        n = _norm_card(c)
        deck_counts[n] = deck_counts.get(n, 0) + 1
    candidates = [
        a
        for a in picks
        if a["idx"] < len(ids)
        and _card_tier(ids[a["idx"]]) >= 2
        and deck_counts.get(_norm_card(ids[a["idx"]]), 0) < 3
    ]
    if candidates:
        return rng.choice(candidates)
    skip = next(
        (a for a in acts if a.get("action") == "pick_card_reward" and a.get("idx") is None),
        None,
    )
    return skip or rng.choice(acts)


def _pick_shop_action(rm: Any, acts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Priority: remove a curse > top up potions > buy relics > buy one
    good card when rich > leave."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for a in acts:
        by_kind.setdefault(a["action"], []).append(a)
    inv = rm.shop_inventory() or {}
    deck = rm.master_deck()
    for a in by_kind.get("shop_remove_card", []):
        if a["deck_idx"] < len(deck) and _card_tier(deck[a["deck_idx"]]) == 0:
            return a
    if by_kind.get("shop_buy_potion") and len(rm.potions()) < 2:
        prices = {p["idx"]: p["price"] for p in inv.get("potions", [])}
        return min(by_kind["shop_buy_potion"], key=lambda a: prices.get(a["idx"], 1 << 30))
    if by_kind.get("shop_buy_relic"):
        prices = {r["idx"]: r["price"] for r in inv.get("relics", [])}
        return min(by_kind["shop_buy_relic"], key=lambda a: prices.get(a["idx"], 1 << 30))
    if by_kind.get("shop_buy_card") and rm.gold >= 250:
        cards = {c["idx"]: c for c in inv.get("cards", [])}
        good = [
            a
            for a in by_kind["shop_buy_card"]
            if _card_tier(cards.get(a["idx"], {}).get("id", "")) >= 3
        ]
        if good:
            return rng.choice(good)
    leave = next((a for a in acts if a["action"] == "leave_shop"), None)
    return leave or rng.choice(acts)


def _pick_rest_action(rm: Any, acts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Rest below 65% HP, smith otherwise. In the smith sub-state, prefer
    upgrading a non-basic card."""
    smith_picks = [a for a in acts if a["action"] == "smith_card"]
    if smith_picks:
        names = rm.smith_options()
        non_basic = [
            a for a in smith_picks if a["idx"] < len(names) and _card_tier(names[a["idx"]]) >= 3
        ]
        return rng.choice(non_basic or smith_picks)
    rest_opts = [a for a in acts if a["action"] == "rest_option"]
    if rest_opts:
        # Engine option order is [Rest, Smith, …] (RestSiteOption enum); a
        # disabled Smith (nothing upgradable) keeps idx 1 but is not offered,
        # so pick Smith by idx rather than by position.
        smith = next((a for a in rest_opts if a.get("idx") == 1), None)
        if _hp_frac(rm) < 0.65 or smith is None:
            return rest_opts[0]
        return smith
    return rng.choice(acts)


def _pick_deck_removal(rm: Any, acts: list[dict[str, Any]]) -> dict[str, Any]:
    """Forced removal (e.g. Precise Scissors): drop the worst card."""
    deck = rm.master_deck()
    return min(
        acts,
        key=lambda a: _card_tier(deck[a["idx"]]) if a["idx"] < len(deck) else 99,
    )


def heuristic_run_action(rm: Any, acts: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    """Pick an out-of-combat action with simple hand-written rules.

    v1 policy (bead follow-up of sts2-rs-i6h9i): path by room-type score,
    never take curses/statuses from rewards, rest below 65% HP else
    smith, shop for curse removal + potions + relics, events/boss-relics
    take option 0 / random. Combat is the caller's job (MCTS or random).
    """
    kinds = {a.get("action") for a in acts}
    if kinds == {"remove_deck_card"}:
        return _pick_deck_removal(rm, acts)
    # Other pending interactive deck card-selects preempt the room phase with a
    # single-kind pick list; pick one (multi-pick N is resolved one at a time by
    # the caller). Must precede the phase dispatch, which would otherwise route
    # these through the wrong handler (e.g. _pick_map_move when phase=map_choice).
    if kinds and kinds <= _PENDING_SELECT_KINDS:
        return rng.choice(acts)
    phase = rm.phase()
    if phase == "map_choice":
        return _pick_map_move(rm, acts, rng)
    if phase == "card_reward":
        relic = next((a for a in acts if a.get("action") == "pick_relic_reward"), None)
        if relic:
            return relic
        return _pick_card_reward(rm, acts, rng)
    if phase == "shop":
        return _pick_shop_action(rm, acts, rng)
    if phase == "rest_site":
        return _pick_rest_action(rm, acts, rng)
    if phase == "event":
        choices = [a for a in acts if a.get("action") == "event_choice"]
        return choices[0] if choices else rng.choice(acts)
    if phase == "boss_relic":
        boss = [a for a in acts if a.get("action") == "pick_boss_relic"]
        return rng.choice(boss or acts)
    if phase == "treasure":
        take = next((a for a in acts if a.get("action") == "collect_treasure"), None)
        return take or rng.choice(acts)
    return rng.choice(acts)


# RunAction kinds :func:`to_workbench_verb` maps to a dedicated C# verb. A kind
# OUTSIDE this set falls through to the generic ``(kind, args)`` round-trip, which
# :func:`resolve_workbench_verb` turns into a ``choose_option`` position. Keyed on
# the kind (not the verb name) because ``play_card``/``end_turn`` are verbs whose
# names coincide with their RunAction kind — a name-equality check would
# misclassify them as fallbacks.
_DEDICATED_VERB_KINDS = frozenset(
    {
        "move",
        "play_card",
        "end_turn",
        "drink_potion",
        "combat_resolve_choice",
        "pick_card_reward",
        "shop_buy_card",
        "shop_buy_relic",
        "shop_buy_potion",
        "smith_card",
        "remove_deck_card",
        "shop_remove_card",
        "rest_option",
        "event_choice",
        "pick_boss_relic",
        "leave_shop",
        *_PENDING_SELECT_KINDS,
    }
)


def to_workbench_verb(rm: Any, action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """RunAction dict → the workbench's C#-shaped ``(verb, args)``.

    The SPA's timeline labels (``render.describe_action``), map view
    (``/api/map`` taken-node marking), and the live action endpoint all speak
    this C# verb vocabulary; the raw RunAction dicts are unreadable to them.
    Shared by the full-run recorder (``play_full_run_rust``) and the live
    auto-play recommendation (:func:`recommend_run_action`). Must be called
    BEFORE ``rm.take_action`` (the map row / bundle sub-state are pre-move).

    A kind with no dedicated verb (``pick_relic_reward`` / ``skip_relic`` /
    ``collect_treasure`` / …) round-trips as ``(kind, args)``; those screens are
    option lists, so :func:`recommend_run_action` remaps such an action to its
    ``choose_option`` position. The recorder, in contrast, just logs the kind
    (``describe_action`` falls back to ``str(act)``)."""
    kind = action.get("action")
    if kind == "move":
        node = map_move_options(rm).get(action.get("col")) or {}
        return "select_map_node", {"col": action.get("col"), "row": node.get("row")}
    if kind == "play_card":
        return "play_card", {
            "card_index": action.get("hand_idx"),
            "target_index": action.get("target_index"),
        }
    if kind == "end_turn":
        return "end_turn", {}
    if kind == "drink_potion":
        return "use_potion", {
            "potion_index": action.get("potion_slot"),
            "target_index": action.get("target_index"),
        }
    if kind == "combat_resolve_choice":
        opt = action.get("option")
        return ("skip_select", {}) if opt is None else ("select_cards", {"indices": str(opt)})
    if kind == "pick_card_reward":
        idx = action.get("idx")
        if idx is None:
            return "skip_card_reward", {}
        if rm.card_reward_bundles() is not None:
            return "select_bundle", {"bundle_index": idx}
        return "select_card_reward", {"card_index": idx}
    if kind == "shop_buy_card":
        return "buy_card", {"card_index": action.get("idx")}
    if kind == "shop_buy_relic":
        return "buy_relic", {"relic_index": action.get("idx")}
    if kind == "shop_buy_potion":
        return "buy_potion", {"potion_index": action.get("idx")}
    if kind in ("smith_card", "remove_deck_card") or kind in _PENDING_SELECT_KINDS:
        return "select_cards", {"indices": str(action.get("idx"))}
    if kind == "shop_remove_card":
        return "select_cards", {"indices": str(action.get("deck_idx"))}
    if kind in ("rest_option", "event_choice", "pick_boss_relic"):
        return "choose_option", {"option_index": action.get("idx")}
    if kind == "leave_shop":
        return "leave_room", {}
    # pick_relic_reward / skip_relic / collect_treasure / future kinds: no
    # dedicated verb — return the kind so callers can decide (the recorder logs
    # it; recommend_run_action remaps it to a choose_option position).
    return str(kind), {k: v for k, v in action.items() if k != "action"}


def resolve_workbench_verb(
    rm: Any, action: dict[str, Any], acts: list[dict[str, Any]] | None = None
) -> tuple[str, dict[str, Any]] | None:
    """A raw RunAction → the ``(verb, args)`` the live action endpoint applies.

    :func:`to_workbench_verb` with the option-list fallback both
    :func:`recommend_run_action` and the workbench's faithful "play to end" need:
    a kind with no dedicated verb (``pick_relic_reward`` / ``collect_treasure`` /
    …) is remapped to a ``choose_option`` whose index is the action's position in
    ``available_actions()`` (the order the backend renders the options in).
    Returns ``None`` when such an action isn't present in the option list.
    """
    if action.get("action") in _DEDICATED_VERB_KINDS:
        return to_workbench_verb(rm, action)
    acts = acts if acts is not None else rm.available_actions()
    try:
        return "choose_option", {"option_index": acts.index(action)}
    except ValueError:
        return None


def recommend_run_action(
    rm: Any, rng: random.Random | None = None
) -> tuple[str, dict[str, Any]] | None:
    """The ``(verb, args)`` the out-of-combat heuristic policy would take now.

    Drives the workbench's "play best (G)" / "play to end (F)" off-combat
    autoplay: picks an action with :func:`heuristic_run_action`, then maps it
    to the C# verb the live action endpoint applies. Returns ``None`` in combat
    (use the MCTS hint there), at a finished run, or when no action is
    available. Option-list screens (relic reward, treasure, …) whose action has
    no dedicated verb are resolved to a ``choose_option`` indexing 1:1 into
    ``available_actions()`` — the same order the backend renders the options in.
    """
    if rm is None or rm.phase() in ("combat", "run_over"):
        return None
    acts = rm.available_actions()
    if not acts:
        return None
    rng = rng or random.Random()
    action = heuristic_run_action(rm, acts, rng)
    return resolve_workbench_verb(rm, action, acts)


# ---------------------------------------------------------------------------
# Boss-simulation card-reward chooser
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _pascal_to_upper_snake(name: str) -> str:
    """``PerfectedStrike`` → ``PERFECTED_STRIKE`` (CardId/PotionKind Debug
    names → the canonical UPPER_SNAKE ids ``build_scenario_state`` wants)."""
    return _CAMEL_BOUNDARY.sub("_", name).upper()


def _scenario_loadout(rm: Any) -> dict[str, Any]:
    """Current run loadout in ``build_scenario_state`` id formats."""
    deck = [
        _pascal_to_upper_snake(cid) + ("+1" if upgraded else "")
        for cid, upgraded in rm.master_deck_detailed()
    ]
    relics = [r.removeprefix("RELIC.") for r in rm.relics()]
    potions = [_pascal_to_upper_snake(p) for p in rm.potions()]
    return {
        "character": rm.character,
        "deck": deck,
        "relics": relics,
        "potions": potions,
        "hp": rm.current_hp,
        "max_hp": rm.max_hp,
    }


def play_out_combat(
    cs: Any,
    *,
    sims: int,
    seed: int,
    threads: int = 8,
    max_steps: int = 300,
) -> float:
    """Play ``cs`` to terminal with MCTS (``sims`` per decision); return a
    scalar fight score.

    Score: win → ``10_000 + final player HP`` (rank wins by HP kept);
    loss / step-cap → ``player HP − total living monster HP`` (rank bad
    outcomes by how close the kill was). Mutates ``cs``.
    """
    steps = 0
    while steps < max_steps:
        terminal, _value = cs.terminal_value()
        legal = cs.legal_actions()
        if terminal or not legal:
            break
        # Honest search, matching the deployed `MctsCombatPolicy` — NOT the
        # binding's clairvoyant default. This rollout is the card-reward
        # advisor's yardstick, so a root that can see the draw order scores
        # every candidate deck as it would play for a cheater, not for the
        # agent that will actually play it.
        cfg = RecommendConfig(sims=sims, threads=threads)
        result = sts2_rs.mcts_recommend(
            cs,
            sims=sims,
            seed=seed + steps,
            threads=threads,
            c_puct=cfg.resolved_c_puct(),
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
        )
        action = best_recommend_action(result) or legal[0]
        cs.apply(action)
        steps += 1
    terminal, value = cs.terminal_value()
    st = cs.get_state()
    player_hp = float((st.get("player") or {}).get("hp", 0) or 0)
    enemy_hp = sum(
        max(0.0, float(e.get("hp", 0) or 0)) for e in st.get("enemies") or [] if e.get("is_alive")
    )
    if terminal and value > 0:
        return 10_000.0 + player_hp
    return player_hp - enemy_hp


def boss_sim_card_reward(
    rm: Any,
    acts: list[dict[str, Any]],
    *,
    sims: int = 3000,
    threads: int = 8,
    max_fight_steps: int = 300,
    verbose: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Pick a card reward by simulating the pinned act-boss fight once per
    candidate (each offered card, plus skip) and taking the best score.

    Returns ``(action, details)`` where ``details`` carries the per-candidate
    scores (``{"encounter", "sims", "candidates": [{label, score, win}]}``)
    for run-log attachment. Returns ``(None, None)`` — caller falls back to
    the cheap heuristic — when the decision isn't a plain card pick (relic
    sub-state, Scroll Boxes bundles), the boss isn't resolvable to a
    buildable encounter, or a candidate loadout can't be realized
    (unported content).

    Each candidate fight is built with the same combat seed, so the
    draw-order/HP rolls are identical across candidates and the only
    variable is the added card.
    """
    if any(a.get("action") == "pick_relic_reward" for a in acts):
        return None, None
    if rm.card_reward_bundles() is not None:
        return None, None
    picks = [a for a in acts if a.get("action") == "pick_card_reward" and a.get("idx") is not None]
    if not picks:
        return None, None
    encounter = rm.pinned_boss_encounter_id()
    if not encounter:
        return None, None

    ids = rm.card_reward_option_ids()
    loadout = _scenario_loadout(rm)
    # Same fight seed for every candidate — only the deck differs.
    eval_seed = (rm.seed & 0x7FFF_FFFF) * 31 + rm.total_floor()

    skip = next(
        (a for a in acts if a.get("action") == "pick_card_reward" and a.get("idx") is None),
        None,
    )
    candidates: list[tuple[dict[str, Any], str | None]] = [
        (a, ids[a["idx"]].removeprefix("CARD.")) for a in picks if a["idx"] < len(ids)
    ]
    if skip is not None:
        candidates.append((skip, None))

    best_action: dict[str, Any] | None = None
    best_score = float("-inf")
    scored: list[dict[str, Any]] = []
    for action, extra in candidates:
        deck = loadout["deck"] + ([extra] if extra else [])
        try:
            cs = sts2_rs.build_scenario_state(
                eval_seed,
                loadout["character"],
                deck,
                loadout["hp"],
                loadout["max_hp"],
                loadout["relics"],
                encounter,
                potions=loadout["potions"],
            )
        except ValueError as exc:
            # Unported content in the loadout — no candidate is buildable
            # (the deck is shared), so bail to the heuristic entirely.
            if verbose:
                print(f"    reward-sim unavailable ({exc}); falling back to heuristic")
            return None, None
        score = play_out_combat(
            cs, sims=sims, seed=eval_seed, threads=threads, max_steps=max_fight_steps
        )
        label = extra or "skip"
        scored.append(
            {
                # `idx` is the card_reward option index (None for skip) so
                # UIs can attach the score to the offered card.
                "idx": action.get("idx"),
                "label": label,
                "score": score,
                "win": score >= 10_000,
            }
        )
        if verbose:
            outcome = "WIN" if score >= 10_000 else "loss"
            print(f"    reward-sim vs {encounter}: {label:<24} score={score:8.1f} ({outcome})")
        # Strict > keeps the earlier candidate on ties; skip is evaluated
        # last, so skip must beat every card outright to win.
        if score > best_score:
            best_score = score
            best_action = action
    details = {"encounter": encounter, "sims": sims, "candidates": scored}
    return best_action, details
