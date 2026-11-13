"""Unit tests for headless_eval's boss-relic event rerouting.

The act-start Ancients (Orobas/Pael/Tezcatara, Nonupeipe/Tanx/Vakuu) carry the
act's boss-relic 1-of-3 offer on a generic ``event_choice``; the PPO net only
knows relic identity on the ``boss_relic`` surface (2026-07-15 ablation:
delegating just this decision collapsed the native 10/20 baseline to ~0 —
the untrained event surface picks positionally). ``_boss_relic_reroute`` must
convert exactly these screens to the bridge ``boss_relic`` shape and leave
every other event untouched.
"""

from __future__ import annotations

from typing import Any

import pytest
from headless_eval import (
    _boss_relic_reroute,
    _bridge_answer_to_command,
    _envelope,
    to_bridge_event,
    to_bridge_shop,
)


def _event_state(options: list[dict[str, Any]], event_name: str = "") -> dict[str, Any]:
    return {
        "decision": "event_choice",
        "event_name": event_name,
        "options": options,
        "context": {"floor": 1, "act": 2},
        "player": {"hp": 60, "max_hp": 75, "gold": 99, "deck": [], "relics": [], "potions": []},
    }


def test_dll_ancient_relic_offer_reroutes_to_boss_relic() -> None:
    state = _event_state(
        [
            {
                "index": 0,
                "title": "Very Hot Cocoa",
                "text_key": "TEZCATARA.pages.INITIAL.options.VERY_HOT_COCOA",
            },
            {
                "index": 1,
                "title": "Storybook",
                "text_key": "TEZCATARA.pages.INITIAL.options.STORYBOOK",
            },
            {"index": 2, "title": "Toy Box", "text_key": "TEZCATARA.pages.INITIAL.options.TOY_BOX"},
        ],
        event_name="Tezcatara",
    )
    out = to_bridge_event(state, "Ironclad")
    assert out["type"] == "boss_relic"
    assert [o["relic"]["id"] for o in out["options"]] == [
        "VERY_HOT_COCOA",
        "STORYBOOK",
        "TOY_BOX",
    ]
    assert [o["index"] for o in out["options"]] == [0, 1, 2]


def test_rust_dialect_bare_relic_keys_reroute() -> None:
    """RustGame(csharp) ``_fill_boss_relic``: bare relic-id text_keys, blank
    event_name (see test_csharp_resolve_event_fields_resolves_boss_relic...)."""
    state = _event_state(
        [
            {"index": 0, "title": "Driftwood", "text_key": "DRIFTWOOD"},
            {"index": 1, "title": "Biiig Hug", "text_key": "BIIIG_HUG"},
            {"index": 2, "title": "Glass Eye", "text_key": "GLASS_EYE"},
        ]
    )
    out = to_bridge_event(state, "Ironclad")
    assert out["type"] == "boss_relic"
    assert [o["relic"]["id"] for o in out["options"]] == ["DRIFTWOOD", "BIIIG_HUG", "GLASS_EYE"]


def test_neow_stays_on_event_surface() -> None:
    """Neow is natively an EVENT (enum-labeled options the net trained on)."""
    state = _event_state(
        [
            {"index": 0, "title": "New Leaf", "text_key": "NEOW.pages.INITIAL.options.NEW_LEAF"},
            {
                "index": 1,
                "title": "Winged Boots",
                "text_key": "NEOW.pages.INITIAL.options.WINGED_BOOTS",
            },
        ],
        event_name="Neow",
    )
    assert _boss_relic_reroute(state, "Ironclad") is None
    assert to_bridge_event(state, "Ironclad")["type"] == "event"


def test_ordinary_event_stays_on_event_surface() -> None:
    state = _event_state(
        [
            {
                "index": 0,
                "title": "Trudge On",
                "text_key": "DENSE_VEGETATION.pages.INITIAL.options.TRUDGE_ON",
            },
            {
                "index": 1,
                "title": "Rest",
                "text_key": "DENSE_VEGETATION.pages.INITIAL.options.REST",
            },
        ],
        event_name="Dense Vegetation",
    )
    assert _boss_relic_reroute(state, "Ironclad") is None


def test_ancient_dialogue_page_stays_on_event_surface() -> None:
    """An act-start Ancient page whose options are NOT all relics (dialogue /
    continue) must keep today's event routing."""
    state = _event_state(
        [
            {
                "index": 0,
                "title": "Continue",
                "text_key": "TEZCATARA.pages.INITIAL.options.CONTINUE",
            },
        ],
        event_name="Tezcatara",
    )
    assert _boss_relic_reroute(state, "Ironclad") is None
    state2 = _event_state(
        [
            {
                "index": 0,
                "title": "Very Hot Cocoa",
                "text_key": "TEZCATARA.pages.INITIAL.options.VERY_HOT_COCOA",
            },
            {"index": 1, "title": "Leave", "text_key": "TEZCATARA.pages.INITIAL.options.LEAVE"},
        ],
        event_name="Tezcatara",
    )
    assert _boss_relic_reroute(state2, "Ironclad") is None


def test_boss_relic_answer_maps_back_to_choose_option() -> None:
    cmd = _bridge_answer_to_command("boss_relic", {"action": "choose", "index": 2}, {}, None)
    assert cmd == {"action": "choose_option", "args": {"option_index": 2}}


def test_map_select_with_no_choices_proceeds_instead_of_crashing() -> None:
    """Boss-transition edge: DLL map_select with an empty choices list (killed
    seed-1015 at floor 48 via IndexError in the 2026-07-15 sweep)."""
    cmd = _bridge_answer_to_command(
        "map_select", {"action": "choose", "index": 0}, {"choices": []}, None
    )
    assert cmd == {"action": "proceed", "args": {}}


# ── live-bridge shape: the parallel reroute inside realgame_ppo ─────────────
#
# Live SpireMentor presents the act-start Ancient as a plain `event` message
# (options carry raw `event_id` + `relic_id`); headless_eval never sees those
# states, so realgame_ppo._build_event must do its own detection.


def _live_event_state(options: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "event",
        "floor": 18,
        "act": 2,
        "run_state": {
            "character": "Ironclad",
            "hp": 60,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "master_deck": [],
            "ascension": 0,
        },
        "options": options,
    }


def test_live_ancient_relic_offer_builds_boss_relic_surface() -> None:
    from realgame_ppo import _build_event

    state = _live_event_state(
        [
            {
                "index": 0,
                "label": "Yummy Cookie",
                "text_key": "TEZCATARA.pages.INITIAL.options.YUMMY_COOKIE",
                "relic_id": "YUMMY_COOKIE",
                "event_id": "TEZCATARA",
                "enabled": True,
            },
            {
                "index": 1,
                "label": "Biiig Hug",
                "text_key": "TEZCATARA.pages.INITIAL.options.BIIIG_HUG",
                "relic_id": "BIIIG_HUG",
                "event_id": "TEZCATARA",
                "enabled": True,
            },
            {
                "index": 2,
                "label": "Seal of Gold",
                "text_key": "TEZCATARA.pages.INITIAL.options.SEAL_OF_GOLD",
                "relic_id": "SEAL_OF_GOLD",
                "event_id": "TEZCATARA",
                "enabled": True,
            },
        ]
    )
    decision = _build_event(state)
    assert decision is not None
    assert [a["action"] for a in decision.actions] == ["pick_boss_relic"] * 3
    assert decision.obs["boss_relic_options"] == ["YUMMY_COOKIE", "BIIIG_HUG", "SEAL_OF_GOLD"]
    assert decision.responses == [
        {"action": "choose", "index": 0},
        {"action": "choose", "index": 1},
        {"action": "choose", "index": 2},
    ]


def test_live_ancient_proceed_page_stays_event() -> None:
    from realgame_ppo import _ancient_boss_relic_decision

    state = _live_event_state(
        [
            {
                "index": 0,
                "label": "Proceed",
                "text_key": "PROCEED",
                "relic_id": "",
                "event_id": "TEZCATARA",
                "enabled": True,
            }
        ]
    )
    assert _ancient_boss_relic_decision(state) is None


def test_envelope_dll_dialect_prefers_total_floor() -> None:
    """Real-DLL shape: context dict, act 1-based, floor act-LOCAL, total_floor
    run-wide (RunContext emits it since 2026-07-16). The bridge envelope must
    carry TotalFloor semantics (live SpireMentor parity)."""
    state = {
        "context": {"act": 2, "floor": 5, "total_floor": 22},
        "player": {},
    }
    out = _envelope(state, "map_select", "Ironclad")
    assert out["act"] == 2
    assert out["floor"] == 22


def test_envelope_dll_dialect_without_total_floor_falls_back_to_act_floor() -> None:
    """Older runner build: no total_floor key -- act-local floor beats 0."""
    state = {"context": {"act": 3, "floor": 7}, "player": {}}
    out = _envelope(state, "map_select", "Ironclad")
    assert out["act"] == 3
    assert out["floor"] == 7


def test_envelope_rustgame_dialect_reads_top_level_keys() -> None:
    """RustGame csharp emulation: NO context dict; top-level act is 0-based
    rm.act_index and floor is run-wide rm.total_floor(). Before the 2026-07-16
    fix this dialect produced act=1/floor=0 on EVERY decision (the net always
    thought it was at the start of the game -- the largest surviving obs
    drift in the engine-world deployed harnesses)."""
    state = {"act": 2, "floor": 38, "player": {}}
    out = _envelope(state, "event_choice", "Ironclad")
    assert out["act"] == 3
    assert out["floor"] == 38


def test_envelope_card_select_string_context_uses_top_level() -> None:
    """card_select views carry context as a STRING kind ("remove"/"upgrade");
    the envelope must not crash on it and falls through to top-level keys."""
    state = {"context": "remove", "act": 1, "floor": 20, "player": {}}
    out = _envelope(state, "card_select", "Ironclad")
    assert out["act"] == 2
    assert out["floor"] == 20


def test_run_state_emits_empty_belt_slots_from_potion_slots() -> None:
    """A player holding 1 potion with 3 belt slots must surface 2 EMPTY slots
    -- realgame_ppo._build_shop's belt-full heuristic reads them; without
    potion_slots the deployed net can never buy a potion once it holds one
    (2026-07-16 shop audit: native buys, deployed leaves at p=0.99)."""
    state = {
        "context": {"act": 1, "floor": 3, "total_floor": 3},
        "player": {
            "hp": 60,
            "max_hp": 75,
            "gold": 100,
            "relics": [],
            "potions": [{"index": 0, "id": "SPEED_POTION", "name": "SPEED_POTION.title"}],
            "potion_slots": 3,
            "deck": [],
        },
    }
    out = _envelope(state, "shop", "Ironclad")
    belt = out["run_state"]["potions"]
    assert len(belt) == 3
    assert [p["empty"] for p in belt] == [False, True, True]


def test_shop_options_carry_item_identity() -> None:
    """Shop relic/potion/card option items must carry the vocab-resolvable id
    (2026-07-16 audit: the csharp emulation omitted relic/potion ids entirely,
    so every shop relic/potion featurized OOV -- the net was identity-blind at
    shops, 43.7% same-net disagreement)."""
    state = {
        "decision": "shop",
        "context": {"act": 2, "floor": 5, "total_floor": 22},
        "player": {"gold": 300, "relics": [], "potions": [], "deck": []},
        "cards": [
            {"index": 0, "id": "ANGER", "name": "ANGER.title", "cost": 49, "is_stocked": True},
        ],
        "relics": [
            {
                "index": 0,
                "id": "ORNAMENTAL_FAN",
                "name": "ORNAMENTAL_FAN.title",
                "cost": 154,
                "is_stocked": True,
            },
        ],
        "potions": [
            {
                "index": 0,
                "id": "SPEED_POTION",
                "name": "SPEED_POTION.title",
                "cost": 50,
                "is_stocked": True,
            },
        ],
        "card_removal_cost": 75,
    }
    bridge_state, commands = to_bridge_shop(state, "Ironclad")
    by_action = {}
    for o in bridge_state["options"]:
        by_action.setdefault(o["action"], []).append(o)
    assert by_action["buy_card"][0]["item"]["id"] == "ANGER"
    assert by_action["buy_relic"][0]["item"]["id"] == "ORNAMENTAL_FAN"
    assert by_action["buy_potion"][0]["item"]["id"] == "SPEED_POTION"
    # commands stay index-aligned with options
    for o in bridge_state["options"]:
        assert 0 <= o["index"] < len(commands)


class _RemovalStubPolicy:
    """Peaks on the shop_remove_card action for deck_idx 2 (else index 0)."""

    def distribution(self, obs, actions, metrics):
        probs = [0.0] * len(actions)
        target = next(
            (
                i
                for i, a in enumerate(actions)
                if a.get("action") == "shop_remove_card" and a.get("deck_idx") == 2
            ),
            0,
        )
        probs[target] = 1.0
        return probs, 0.5


class _RemovalStubCache:
    def get(self, character):
        return _RemovalStubPolicy(), None


def _shop_state_with_removal() -> dict:
    return {
        "type": "shop",
        "floor": 20,
        "act": 2,
        "run_state": {
            "character": "Ironclad",
            "hp": 60,
            "max_hp": 75,
            "gold": 300,
            "relics": [],
            "potions": [],
            "master_deck": [
                {"id": "STRIKE_IRONCLAD"},
                {"id": "BASH"},
                {"id": "THUNDERCLAP"},
                {"id": "DEFEND_IRONCLAD", "upgrade_level": 1},
            ],
            "ascension": 0,
        },
        "options": [
            {"index": 0, "id": "leave_shop", "action": "leave_shop", "label": "Leave shop"},
            {"index": 1, "id": "remove_card", "action": "remove_card", "price": 75},
        ],
        "purge_cost": 75,
    }


def test_shop_removal_expansion_covers_the_deck_and_carries_the_target() -> None:
    """The shop surface decides WHICH card to remove (training shape); the
    follow-up card_select must honor that pick instead of re-deciding on the
    remove_deck_card surface (2026-07-16 differ: re-deciding picked a
    different card in ~half the removals).

    Every non-Eternal deck card expands, basics included. This used to assert
    `[1, 2]` and say basics were excluded "like the engine's
    CardId::is_removable action surface" -- but `Phase::Shop` enumerates
    `removal_offer_order()`, which is Eternal-gated and keeps basics, so the
    claim described a bridge-only filter rather than the engine.
    """
    import random

    from realgame_ppo import _build_shop, pick_noncombat_action_ppo

    state = _shop_state_with_removal()
    decision = _build_shop(state)
    assert decision is not None
    removal_idxs = [
        a["deck_idx"] for a in decision.actions if a.get("action") == "shop_remove_card"
    ]
    # Strike (0) and Defend+ (3) are removable at a v0.107.1 shop too.
    assert removal_idxs == [0, 1, 2, 3]

    session: dict = {}
    answer, _extra = pick_noncombat_action_ppo(
        state,
        random.Random(0),
        _RemovalStubCache(),  # type: ignore[arg-type]
        run_config_character="Ironclad",
        session=session,
    )
    assert answer["index"] == 1  # the remove_card bridge option
    assert session["shop_remove_target"] == ("THUNDERCLAP", False)

    # Follow-up card_select (prev=shop): the stashed target must be honored.
    session["prev_msg_type"] = "shop"
    followup = {
        "type": "card_select",
        "floor": 20,
        "act": 2,
        "run_state": state["run_state"],
        "cards": [
            {"index": 1, "id": "BASH", "name": "Bash"},
            {"index": 2, "id": "THUNDERCLAP", "name": "Thunderclap"},
        ],
        "min_count": 1,
        "max_count": 1,
    }
    answer2, _ = pick_noncombat_action_ppo(
        followup,
        random.Random(0),
        _RemovalStubCache(),  # type: ignore[arg-type]
        run_config_character="Ironclad",
        session=session,
    )
    assert answer2["indices"] == [2]
    assert "shop_remove_target" not in session


def test_shop_removal_stash_cleared_by_unrelated_decision() -> None:
    """A stale removal target must not leak into a later card_select."""
    import random

    from realgame_ppo import pick_noncombat_action_ppo

    state = _shop_state_with_removal()
    session: dict = {}
    pick_noncombat_action_ppo(
        state,
        random.Random(0),
        _RemovalStubCache(),  # type: ignore[arg-type]
        run_config_character="Ironclad",
        session=session,
    )
    assert "shop_remove_target" in session
    # An intervening non-card_select decision (shop re-presented) re-stashes
    # or clears; a map_select-shaped unknown message just clears.
    pick_noncombat_action_ppo(
        {"type": "treasure", "run_state": state["run_state"]},
        random.Random(0),
        _RemovalStubCache(),  # type: ignore[arg-type]
        run_config_character="Ironclad",
        session=session,
    )
    assert "shop_remove_target" not in session


def test_live_neow_stays_event() -> None:
    from realgame_ppo import _ancient_boss_relic_decision

    state = _live_event_state(
        [
            {
                "index": 0,
                "label": "Winged Boots",
                "text_key": "NEOW.pages.INITIAL.options.WINGED_BOOTS",
                "relic_id": "WINGED_BOOTS",
                "event_id": "NEOW",
                "enabled": True,
            },
            {
                "index": 1,
                "label": "New Leaf",
                "text_key": "NEOW.pages.INITIAL.options.NEW_LEAF",
                "relic_id": "NEW_LEAF",
                "event_id": "NEOW",
                "enabled": True,
            },
        ]
    )
    assert _ancient_boss_relic_decision(state) is None


def test_session_aborts_on_byte_identical_state_stall(monkeypatch) -> None:
    """A run whose state never changes must bail as ``stall_detected``, not
    grind the full ``max_steps`` cap (the real-DLL EndTurn livelock: combat
    wedges in a non-play phase and every action echoes the same state)."""
    import random
    import types

    import headless_eval

    stuck_state = {
        "type": "decision",
        "decision": "combat_play",
        "context": {"floor": 7, "act": 1},
        "player": {"hp": 44, "max_hp": 80, "energy": 3},
        "hand": [],
        "enemies": [{"name": "Flyconid", "hp": 30}],
    }

    class _StuckGame:
        # Mirrors the real `sts2_client.Game` surface the session touches:
        # the `forward_stderr` kwarg and `world_errors()` / `proc.wait()`, which
        # the harness reads in its `finally` to tally C# throws.
        def __init__(self, *, forward_stderr: bool = False) -> None:
            self.proc = types.SimpleNamespace(kill=lambda: None, wait=lambda timeout=None: 0)

        def world_errors(self, **kwargs) -> dict:
            return {"count": 0, "sample": []}

        def start(self, character: str, **kwargs) -> dict:
            return dict(stuck_state)

        def send(self, cmd: dict) -> dict:
            return dict(stuck_state)

    monkeypatch.setattr(headless_eval, "Game", _StuckGame)
    monkeypatch.setattr(
        headless_eval,
        "pick_combat_action_headless",
        lambda state, cfg, rng, sims, banned=None, mirror=None: (
            {"action": "end_turn", "args": {}},
            None,
            {"action": "end_turn"},
        ),
    )

    outcome = headless_eval.play_one_headless_session(
        random.Random(0),
        run_index=0,
        character="Ironclad",
        seed=1007,
        mcts_cfg=None,  # unused: combat picks are stubbed
        ppo_policies=None,  # unused: no non-combat decisions occur
        max_steps=500,
    )

    assert outcome.result == "stall_detected"
    assert outcome.won is False
    # The breaker must fire near _STALL_REPEAT_LIMIT, far below max_steps.
    assert outcome.decisions <= headless_eval._STALL_REPEAT_LIMIT + 5


def test_session_aborts_on_wall_clock_cap(monkeypatch) -> None:
    """`max_run_seconds` is the backstop for stall variants whose echoed
    state is not byte-identical (a mutating counter defeats the hash check)."""
    import itertools
    import random
    import types

    import headless_eval

    counter = itertools.count()

    class _MutatingStuckGame:
        def __init__(self, *, forward_stderr: bool = False) -> None:
            self.proc = types.SimpleNamespace(kill=lambda: None, wait=lambda timeout=None: 0)

        def world_errors(self, **kwargs) -> dict:
            return {"count": 0, "sample": []}

        def _state(self) -> dict:
            return {
                "type": "decision",
                "decision": "combat_play",
                "context": {"floor": 7, "act": 1},
                "player": {"hp": 44, "max_hp": 80},
                "tick": next(counter),  # never byte-identical
            }

        def start(self, character: str, **kwargs) -> dict:
            return self._state()

        def send(self, cmd: dict) -> dict:
            return self._state()

    monkeypatch.setattr(headless_eval, "Game", _MutatingStuckGame)
    monkeypatch.setattr(
        headless_eval,
        "pick_combat_action_headless",
        lambda state, cfg, rng, sims, banned=None, mirror=None: (
            {"action": "end_turn", "args": {}},
            None,
            {"action": "end_turn"},
        ),
    )

    outcome = headless_eval.play_one_headless_session(
        random.Random(0),
        run_index=0,
        character="Ironclad",
        seed=1009,
        mcts_cfg=None,
        ppo_policies=None,
        max_steps=500,
        max_run_seconds=0.0,
    )

    assert outcome.result == "time_cap_reached"
    assert outcome.won is False
    assert outcome.decisions == 0


def test_bundle_select_maps_to_select_bundle() -> None:
    """Scroll Boxes bundle offers must be answered with select_bundle.

    The old fallthrough answered ``proceed``, which leaves the pending bundle
    unanswered — the state repeats forever and the stall breaker kills the
    run at floor 1 (seed 1016).
    """
    state = {"type": "decision", "decision": "bundle_select", "bundles": [{}, {}, {}]}

    cmd = _bridge_answer_to_command("bundle_select", {"action": "choose", "index": 1}, state, None)
    assert cmd == {"action": "select_bundle", "args": {"bundle_index": 1}}

    # Out-of-range index clamps to the first bundle instead of erroring.
    cmd = _bridge_answer_to_command("bundle_select", {"action": "choose", "index": 7}, state, None)
    assert cmd == {"action": "select_bundle", "args": {"bundle_index": 0}}


def test_combat_pick_skips_dll_rejected_commands(monkeypatch) -> None:
    """A command the DLL bounced with an ``error`` state must not be re-picked
    for the same state — the ranking walk skips it and answers with the
    next-best action (seed-1025 wedge, 2026-07-18 sweep: ``can_play`` true on
    the wire, yet the DLL rejected the play 826 times over 24 min while the
    interleaved error state defeated the consecutive-identical stall check)."""
    import json as _json
    import random as _random
    from types import SimpleNamespace

    import headless_eval as he
    from sts2_hint.recommender import RecommendConfig

    tags = [
        SimpleNamespace(stable_idx=2, target_index=0),  # best: Fight Me! @ enemy 0
        SimpleNamespace(stable_idx=1, target_index=0),  # next: Strike @ enemy 0
        SimpleNamespace(stable_idx=he.END_TURN_STABLE_IDX, target_index=None),
    ]
    monkeypatch.setattr(he, "recommend_all", lambda state, cfg: tags)
    state = {
        "decision": "combat_play",
        "hand": [
            {"index": 0, "can_play": True},
            {"index": 1, "can_play": True},
            {"index": 2, "can_play": True},
        ],
        "enemies": [{"name": "Crusher", "hp": 139}],
    }
    cfg = RecommendConfig(sims=1, depth=1, seed=0)
    rng = _random.Random(0)

    best = {"action": "play_card", "args": {"card_index": 2, "target_index": 0}}
    cmd, extra, engine = he.pick_combat_action_headless(state, cfg, rng)
    assert cmd == best and extra is None
    # The engine-shape twin the caller replays onto the mirror, in HAND
    # coordinates (the DLL's card handle happens to coincide here).
    assert engine == {"action": "play", "card_index": 2, "target_index": 0}

    banned = {_json.dumps(best, sort_keys=True)}
    cmd, extra, _ = he.pick_combat_action_headless(state, cfg, rng, banned=banned)
    assert cmd == {"action": "play_card", "args": {"card_index": 1, "target_index": 0}}

    banned.add(_json.dumps(cmd, sort_keys=True))
    cmd, _, engine = he.pick_combat_action_headless(state, cfg, rng, banned=banned)
    assert cmd == {"action": "end_turn", "args": {}}
    assert engine == {"action": "end_turn"}


def test_a_strict_divergence_escapes_the_mcts_failure_handler(monkeypatch) -> None:
    """STRICT must not be downgraded to "MCTS failed, play something random".

    ``MirrorDivergence`` is a ``RuntimeError``, so the blanket
    ``except Exception`` that catches a crashed search caught it too: STRICT
    never fired on ``combat_play`` at all, and the run carried on answering
    from a state it had PROVEN was wrong. Worse than merely quiet — the
    fallback returns ``engine_action=None``, so the session loop skips
    ``mirror.advance(...)``, the world commits a move the mirror never saw, and
    every later reconcile reports cascading false divergences.

    The fix is a bare ``raise`` with nothing else asserting it, which is
    exactly the shape a later refactor re-swallows silently. This is the guard.
    """
    import random as _random

    import headless_eval as he
    from combat_mirror import MirrorDivergence
    from sts2_hint.recommender import RecommendConfig

    def _boom(state, cfg):
        raise MirrorDivergence("[mirror test] $.player.hp: mirror=40 world=35", {})

    monkeypatch.setattr(he, "recommend_all", _boom)
    state = {
        "decision": "combat_play",
        "hand": [{"index": 0, "can_play": True}],
        "enemies": [{"name": "Nibbit", "hp": 40}],
    }
    cfg = RecommendConfig(sims=1, depth=1, seed=0)

    with pytest.raises(MirrorDivergence, match=r"\$\.player\.hp"):
        he.pick_combat_action_headless(state, cfg, _random.Random(0))

    # ...while an ordinary search failure still degrades to a legal action,
    # which is the case the handler actually exists for.
    def _ordinary(state, cfg):
        raise RuntimeError("rollout exploded")

    monkeypatch.setattr(he, "recommend_all", _ordinary)
    cmd, extra, engine = he.pick_combat_action_headless(state, cfg, _random.Random(0))
    assert cmd["action"] in ("play_card", "end_turn")
    assert extra == {"mcts_error": "rollout exploded"}
    assert engine is None


def test_to_bridge_card_select_carries_prompt_semantics() -> None:
    """prompt_key / cancelable / rarity must survive the bridge conversion.

    They no longer drive a heuristic — a mid-combat selection is answered off
    the persistent combat mirror — but they remain the only description of
    what an OUT-of-combat selection means, and they are what a divergence
    report has to name the prompt by.
    """
    import headless_eval as he

    state = {
        "type": "decision",
        "decision": "card_select",
        "context": {"act": 1, "floor": 3, "room_type": "Monster"},
        "player": {"hp": 50, "max_hp": 80},
        "cards": [
            {
                "index": 0,
                "id": "CARD.INFLAME",
                "name": "Inflame",
                "type": "Power",
                "rarity": "Uncommon",
                "upgraded": True,
            },
            {
                "index": 1,
                "id": "CARD.REGRET",
                "name": "Regret",
                "type": "Curse",
                "rarity": "Curse",
                "upgraded": False,
            },
        ],
        "min_select": 0,
        "max_select": 2,
        "prompt_key": "TO_EXHAUST",
    }
    out = he.to_bridge_card_select(state, "Ironclad")
    assert out["prompt_key"] == "TO_EXHAUST"
    assert out["cancelable"] is True
    assert out["min_count"] == 0 and out["max_count"] == 2
    c0, c1 = out["cards"]
    assert c0["rarity"] == "Uncommon" and c0["upgrade_level"] == 1 and not c0["is_curse"]
    assert c1["is_curse"] and not c1["is_status"]


def test_combat_pick_returns_the_engine_shape_twin(monkeypatch) -> None:
    """Every pick comes back with its engine-dialect twin, which is what the
    session loop replays onto the persistent combat mirror once the DLL has
    accepted the command.

    This replaced ``session["last_combat"]``: the mirror IS the fight's state
    now, so there is no snapshot to stash and replay forward.
    """
    import random as _random
    from types import SimpleNamespace

    import headless_eval as he
    from sts2_hint.recommender import RecommendConfig

    state = {
        "decision": "combat_play",
        "hand": [{"index": 0, "can_play": True}, {"index": 1, "can_play": True}],
        "enemies": [{"name": "Crusher", "hp": 139}],
    }
    cfg = RecommendConfig(sims=1, depth=1, seed=0)
    rng = _random.Random(0)

    tags = [SimpleNamespace(stable_idx=1, target_index=0)]
    monkeypatch.setattr(he, "recommend_all", lambda s, c: tags)
    cmd, _extra, engine = he.pick_combat_action_headless(state, cfg, rng)
    assert cmd == {"action": "play_card", "args": {"card_index": 1, "target_index": 0}}
    assert engine == {"action": "play", "card_index": 1, "target_index": 0}

    tags_end = [SimpleNamespace(stable_idx=he.END_TURN_STABLE_IDX, target_index=None)]
    monkeypatch.setattr(he, "recommend_all", lambda s, c: tags_end)
    cmd, _, engine = he.pick_combat_action_headless(state, cfg, rng)
    assert cmd == {"action": "end_turn", "args": {}}
    assert engine == {"action": "end_turn"}


class TestWorldErrorRollup:
    """`_summarize_world_errors` — the sweep-level "was this a clean corpus?" line.

    Every number in the v7 -> v20 win-count series was quoted without one, which
    is how 11 SLOTH aborts and 2 Kaiser Crab wedges passed as losses. The rollup
    is what makes that visible, so its shape is worth pinning: it must survive
    `world_errors=None` rows (the uninstrumented arm), ignore instrumented rows
    that saw nothing, and carry one representative line per affected seed.
    """

    @staticmethod
    def _outcome(seed: int, world_errors: dict[str, Any] | None):
        from types import SimpleNamespace

        return SimpleNamespace(seed=seed, world_errors=world_errors)

    def test_a_clean_sweep_rolls_up_to_nothing(self) -> None:
        import headless_eval as he

        rows = [self._outcome(1, None), self._outcome(2, {"count": 0, "sample": []})]
        assert he._summarize_world_errors(rows) == {
            "runs": 0,
            "total": 0,
            "seeds": [],
            "first_lines": {},
        }

    def test_only_throwing_seeds_are_listed_with_one_line_each(self) -> None:
        import headless_eval as he

        rows = [
            self._outcome(1000, {"count": 0, "sample": []}),
            self._outcome(1038, {"count": 3, "sample": ["NullReferenceException: boom", "at X"]}),
            self._outcome(1045, {"count": 2, "sample": ["AggregateException: bang"]}),
        ]
        got = he._summarize_world_errors(rows)
        assert got["runs"] == 2
        assert got["total"] == 5
        assert got["seeds"] == [1038, 1045]
        # One line per seed — enough to GROUP by cause without dragging every
        # stack frame into the report.
        assert got["first_lines"] == {
            "1038": "NullReferenceException: boom",
            "1045": "AggregateException: bang",
        }

    def test_a_throwing_run_with_no_sample_still_counts(self) -> None:
        """The count and the sample are independent: the sample is CAPPED, so a
        run that throws in a loop can exhaust it. Losing the line must not lose
        the seed."""
        import headless_eval as he

        got = he._summarize_world_errors([self._outcome(7, {"count": 40})])
        assert got["runs"] == 1
        assert got["total"] == 40
        assert got["first_lines"] == {"7": "?"}


class TestLethalityFlagForwarding:
    """`to_bridge_event` must carry `will_kill_player` through to the choosers.

    It is the only link between the C# probe and both Python no-suicide guards.
    A rename on either side degrades SILENTLY to the HP-cost parse — i.e. back
    to being blind to `Trial.DoubleDown` and `TabletOfTruth.Decipher`, which
    print no HP cost — and no other test would notice.
    """

    def test_the_flag_survives_the_bridge(self) -> None:
        state = _event_state(
            [
                {"index": 0, "title": "Accept", "description": "Stand trial."},
                {
                    "index": 1,
                    "title": "Double Down",
                    "description": "Refuse to be judged.",
                    "will_kill_player": True,
                },
            ]
        )
        out = to_bridge_event(state, "Ironclad")
        assert [o["will_kill_player"] for o in out["options"]] == [None, True]

    def test_an_absent_flag_becomes_none_not_false(self) -> None:
        """`None` means UNKNOWN; `False` means the world said survivable.

        Collapsing the two would tell the guard a pre-flag bridge had cleared
        every option — the failure mode the tri-state exists to prevent.
        """
        out = to_bridge_event(_event_state([{"index": 0, "title": "Leave"}]), "Ironclad")
        assert out["options"][0]["will_kill_player"] is None

    def test_a_false_flag_is_preserved_as_false(self) -> None:
        out = to_bridge_event(
            _event_state([{"index": 0, "title": "Leave", "will_kill_player": False}]),
            "Ironclad",
        )
        assert out["options"][0]["will_kill_player"] is False


class TestEventVarsForwarding:
    """`to_bridge_event` must carry the loc-template `vars` through.

    `realgame_ppo._resolve_option_vars` substitutes them into the option
    description before the encoder's number scrape; without them "Lose
    {DecipherMaxHpLoss} Max HP" carries no number and a 24-max-HP cost reads
    as free (the Tablet of Truth max_hp=1 death spiral, round-4 agent 1).
    """

    def test_vars_survive_the_bridge(self) -> None:
        state = _event_state(
            [
                {
                    "index": 0,
                    "title": "Decipher",
                    "description": "Lose {DecipherMaxHpLoss} Max HP.",
                    "vars": {"DecipherMaxHpLoss": 24},
                },
            ]
        )
        out = to_bridge_event(state, "Ironclad")
        assert out["options"][0]["vars"] == {"DecipherMaxHpLoss": 24}

    def test_absent_vars_forward_as_none(self) -> None:
        out = to_bridge_event(_event_state([{"index": 0, "title": "Leave"}]), "Ironclad")
        assert out["options"][0]["vars"] is None


def test_a_mirrored_pick_targets_the_dlls_living_enemy(monkeypatch) -> None:
    """The mirror's search numbers enemies by encounter slot, corpses kept in
    place; the DLL's ``play_card`` numbers the living only.

    Live diff #1287, Decimillipede: the front segment dead in slot 0, the
    search aimed at slot 1 (the middle segment) and the DLL, reading 1 among
    the living, hit the back segment. The engine twin keeps the slot.
    """
    import random as _random
    from types import SimpleNamespace

    import headless_eval as he
    from sts2_hint.recommender import RecommendConfig

    class _Mirror:
        def reconcile(self, wire):
            pass

        def search(self, cfg):
            return {}

    monkeypatch.setattr(he.adapter, "translate", lambda *a, **k: {"deck": {"hand": []}})
    monkeypatch.setattr(
        he, "tags_from_result", lambda r, h: [SimpleNamespace(stable_idx=0, target_index=2)]
    )
    monkeypatch.setattr(
        he,
        "_mirror_encounter",
        lambda m: [{"is_dead": True}, {"is_dead": False}, {"is_dead": False}],
    )
    state = {
        "decision": "combat_play",
        "hand": [{"index": 0, "can_play": True}],
        "enemies": [{"name": "Middle", "hp": 10}, {"name": "Back", "hp": 22}],
    }
    cfg = RecommendConfig(sims=1, depth=1, seed=0)
    cmd, extra, engine = he.pick_combat_action_headless(
        state, cfg, _random.Random(0), mirror=_Mirror()
    )
    assert extra is None
    assert cmd == {"action": "play_card", "args": {"card_index": 0, "target_index": 1}}
    assert engine == {"action": "play", "card_index": 0, "target_index": 2}
