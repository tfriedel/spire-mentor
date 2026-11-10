"""Offline tests for :mod:`realgame_ppo`.

Three tiers, mirroring :mod:`test_realgame_adapter`'s "pure-dict vs. full
pipeline" split:

* **Builder structure** (no torch/checkpoint needed): the ``_build_*``
  functions produce a well-formed ``PpoDecision`` (equal-length
  actions/responses/labels, in-range bridge indices) from hand-crafted
  fixtures shaped exactly per ``spire-mentor-mod/PROTOCOL.md``.
* **Real inference** (skipped with a clear reason if torch or the
  committed Ironclad checkpoint, ``data/ppo/ppo_fullrun_ironclad_asc_it55.pt``,
  aren't available): the actual committed champion net scores each
  PPO-covered surface and returns a legal, well-annotated bridge action.
* **Fallback + combat reuse**: uncovered surfaces and checkpoint-load
  failures degrade to the random-legal policy with ``ppo_error`` recorded
  (never raising); the combat path re-exports
  :func:`realgame_mcts.pick_combat_action_mcts` / :func:`realgame_random.
  pick_combat_action` verbatim, exercised against the real captured
  ``combat_action`` fixtures under ``tests/fixtures/realgame_adapter/``.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, ClassVar

import pytest
import realgame_ppo as rp
from realgame_mcts import pick_combat_action_mcts
from realgame_random import pick_combat_action
from sts2_realgame import potion_swap_group, reward_screen_choice

try:
    import torch  # noqa: F401

    _TORCH_SKIP_REASON: str | None = None
except ImportError as _e:
    _TORCH_SKIP_REASON = f"torch not importable in this venv: {_e}"

try:
    import sts2_rs  # type: ignore[import-not-found]

    _RS_SKIP_REASON: str | None = None
except ImportError as _e:
    sts2_rs = None  # type: ignore[assignment]
    _RS_SKIP_REASON = f"sts2_rs PyO3 extension not importable in this venv: {_e}"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ppo"
_IRONCLAD_CKPT = _DATA_DIR / "ppo_fullrun_ironclad_asc_it55.pt"
_CKPT_SKIP_REASON = (
    None
    if _IRONCLAD_CKPT.exists()
    else f"committed Ironclad PPO checkpoint missing: {_IRONCLAD_CKPT}"
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "realgame_adapter"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "fixtures" / "sts2_realgame"


def _load_combat_fixture(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)["state"]


# ---------------------------------------------------------------------------
# Synthetic non-combat fixtures, shaped per PROTOCOL.md sections 1 & 3.
# ---------------------------------------------------------------------------

_RUN_STATE = {
    "character": "IRONCLAD",
    "ascension": 0,
    "gold": 99,
    "hp": 50,
    "max_hp": 75,
    "master_deck": [
        {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
        {"id": "DEFEND_IRONCLAD", "upgrade_level": 0},
        {"id": "BASH", "upgrade_level": 1, "upgraded": True},
    ],
    "relics": [{"id": "BURNING_BLOOD"}],
    "potions": [{"slot": 0, "empty": True}],
}


def _envelope(msg_type: str, run_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": msg_type, "floor": 5, "act": 1, "run_state": run_state or _RUN_STATE}


MAP_SELECT_STATE = {
    **_envelope("map_select"),
    "nodes": [
        {"index": 0, "type": "Monster", "row": 6, "col": 2},
        {"index": 1, "type": "RestSite", "row": 6, "col": 3},
    ],
    "map": {
        "row_count": 10,
        "col_count": 6,
        "current": {"col": 2, "row": 5},
        "boss": {"col": 2, "row": 15},
        "visited": [],
        "nodes": [
            {"col": 2, "row": 6, "type": "Monster", "children": [{"col": 2, "row": 7}]},
            {"col": 3, "row": 6, "type": "RestSite", "children": [{"col": 2, "row": 7}]},
            {"col": 2, "row": 7, "type": "Shop", "children": []},
        ],
    },
}

CARD_REWARD_STATE = {
    **_envelope("card_reward"),
    "cards": [
        {"id": "ANGER", "name": "Anger"},
        {"id": "CLEAVE", "name": "Cleave"},
        {"id": "IRON_WAVE", "name": "Iron Wave"},
    ],
    "can_skip": True,
}

CARD_REWARD_NO_SKIP_STATE = {**_envelope("card_reward"), "cards": [], "can_skip": False}

SHOP_STATE = {
    **_envelope("shop"),
    "purge_cost": 75,
    "player_gold": 99,
    "options": [
        {"index": 0, "id": "leave_shop", "action": "leave_shop", "enabled": True},
        {
            "index": 1,
            "id": "buy_card",
            "action": "buy_card",
            "price": 50,
            "enabled": True,
            "item": {"id": "ANGER"},
        },
        {"index": 2, "id": "remove_card", "action": "remove_card", "price": 75, "enabled": True},
        {
            "index": 3,
            "id": "buy_relic",
            "action": "buy_relic",
            "price": 150,
            "enabled": False,
            "item": {"id": "ASTROLABE"},
        },
    ],
}

REST_SITE_STATE = {
    **_envelope("rest_site"),
    "options": [
        {"index": 0, "id": "rest", "action": "rest_option", "label": "Rest", "enabled": True},
        {"index": 1, "id": "smith", "action": "rest_option", "label": "Smith", "enabled": True},
    ],
}

EVENT_STATE = {
    **_envelope("event"),
    "options": [
        {
            "index": 0,
            "id": "event_choice",
            "action": "event_choice",
            "label": "Take it",
            "event_id": "BONFIRE_ELEMENTALS",
            "enabled": True,
        },
        {
            "index": 1,
            "id": "event_choice",
            "action": "event_choice",
            "label": "Leave",
            "event_id": "BONFIRE_ELEMENTALS",
            "enabled": True,
        },
    ],
}

BOSS_RELIC_STATE = {
    **_envelope("boss_relic"),
    "options": [
        {
            "index": 0,
            "id": "r1",
            "action": "pick_relic",
            "label": "Relic A",
            "enabled": True,
            "relic": {"id": "ASTROLABE"},
        },
        {
            "index": 1,
            "id": "r2",
            "action": "pick_relic",
            "label": "Relic B",
            "enabled": True,
            "relic": {"id": "CALIPERS"},
        },
    ],
}

TREASURE_STATE = {
    **_envelope("treasure"),
    "options": [
        {
            "index": 0,
            "id": "open",
            "action": "collect",
            "label": "Open",
            "enabled": True,
            "relic": {"id": "ANCHOR"},
        }
    ],
}


# ---------------------------------------------------------------------------
# Tier 1: builder structure -- no torch/checkpoint needed.
# ---------------------------------------------------------------------------


class TestBuilderStructure:
    @pytest.mark.parametrize(
        "state,builder",
        [
            (MAP_SELECT_STATE, rp._build_map_select),
            (CARD_REWARD_STATE, rp._build_card_reward),
            (SHOP_STATE, rp._build_shop),
            (REST_SITE_STATE, rp._build_rest_site),
            (EVENT_STATE, rp._build_event),
            (BOSS_RELIC_STATE, rp._build_boss_relic),
        ],
    )
    def test_equal_length_parallel_arrays(self, state, builder) -> None:
        decision = builder(state)
        assert decision is not None
        n = len(decision.actions)
        assert n > 0
        assert len(decision.responses) == n
        assert len(decision.labels) == n

    @pytest.mark.parametrize(
        "state,builder,index_field",
        [
            (MAP_SELECT_STATE, rp._build_map_select, "nodes"),
            (REST_SITE_STATE, rp._build_rest_site, "options"),
            (EVENT_STATE, rp._build_event, "options"),
            (BOSS_RELIC_STATE, rp._build_boss_relic, "options"),
        ],
    )
    def test_responses_reference_valid_bridge_indices(self, state, builder, index_field) -> None:
        decision = builder(state)
        assert decision is not None
        valid_indices = {int(o.get("index", i)) for i, o in enumerate(state[index_field])}
        for resp in decision.responses:
            assert resp["action"] == "choose"
            assert resp["index"] in valid_indices

    def test_card_reward_skip_candidate_maps_to_skip_verb(self) -> None:
        decision = rp._build_card_reward(CARD_REWARD_STATE)
        assert decision is not None
        assert decision.responses[-1] == {"action": "skip"}
        assert decision.actions[-1] == {"action": "pick_card_reward", "idx": None}

    def test_card_reward_no_cards_no_skip_returns_none(self) -> None:
        assert rp._build_card_reward(CARD_REWARD_NO_SKIP_STATE) is None

    def test_shop_disabled_option_excluded(self) -> None:
        decision = rp._build_shop(SHOP_STATE)
        assert decision is not None
        # index 3 (buy_relic) is enabled:False -- must not appear.
        assert all(resp["index"] != 3 for resp in decision.responses)

    def test_shop_buy_card_idx_is_position_within_shop_cards_not_bridge_index(self) -> None:
        decision = rp._build_shop(SHOP_STATE)
        assert decision is not None
        buy_card_actions = [a for a in decision.actions if a["action"] == "shop_buy_card"]
        assert buy_card_actions == [{"action": "shop_buy_card", "idx": 0}]
        assert decision.obs["shop"]["cards"] == [{"id": "ANGER", "price": 50, "available": True}]

    def test_shop_remove_card_expands_to_per_deck_card_actions(self) -> None:
        # The net was trained on N per-deck-card shop_remove_card actions
        # (deck_idx), not one opaque service button (previously a "none"
        # token it couldn't value). All candidates answer with the same
        # bridge remove_card option index; the follow-up card_select picks
        # the actual target.
        decision = rp._build_shop(SHOP_STATE)
        assert decision is not None
        removes = [
            (a, r)
            for a, r in zip(decision.actions, decision.responses)
            if a["action"] == "shop_remove_card"
        ]
        # EVERY removable deck card expands, basics included: v0.107.1's shop
        # gates only on `CardModel.IsRemovable` (`!Eternal`), so all three of
        # the [Strike, Defend, Bash] fixture deck are candidates.
        #
        # This assertion used to read `== [2]` and carried a comment saying
        # basics were excluded "like the engine's action surface". They are
        # not: `Phase::Shop` pushes one `ShopRemoveCard` per
        # `removal_offer_order()` entry, and that has been Eternal-gated since
        # 2026-06-19 -- so the test was pinning a bridge-only filter as if it
        # were engine behaviour, which is what kept Strike-thinning off the
        # menu for every deployed champion.
        assert [a["deck_idx"] for a, _ in removes] == [0, 1, 2]
        assert all(r == {"action": "choose", "index": 2} for _, r in removes)
        assert "shop_other_remove_card" not in [a["action"] for a in decision.actions]

    def test_shop_removal_offers_basics_and_withholds_eternal_cards(self) -> None:
        # Both ends of the predicate at once, because the old hand-copied list
        # got both wrong in opposite directions: it dropped basics (which the
        # game offers) and kept Eternal curses (which it never does). Live
        # logs carried `remove:ASCENDERS_BANE` shop picks that no
        # `FromDeckForRemoval` screen could satisfy.
        state = copy.deepcopy(SHOP_STATE)
        state["run_state"] = {
            **_RUN_STATE,
            "master_deck": [
                {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                {"id": "ASCENDERS_BANE", "upgrade_level": 0},
                {"id": "BASH", "upgrade_level": 0},
            ],
        }
        decision = rp._build_shop(state)
        assert decision is not None
        offered = [
            lbl
            for a, lbl in zip(decision.actions, decision.labels)
            if a["action"] == "shop_remove_card"
        ]
        assert offered == ["remove:STRIKE_IRONCLAD", "remove:BASH"]

    def test_an_enchantment_can_make_a_card_non_removable(self) -> None:
        # Removability is not a property of the card id. NutritiousSoup puts
        # TezcatarasEmber on every basic Strike, and that enchantment adds
        # CardKeyword.Eternal (TezcatarasEmber.cs:18) -- so after that pickup
        # the removal screen withholds exactly those Strikes. Goopy is the
        # control: it adds Exhaust, not Eternal.
        state = copy.deepcopy(SHOP_STATE)
        state["run_state"] = {
            **_RUN_STATE,
            "master_deck": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "upgrade_level": 0,
                    "enchantment": "TEZCATARAS_EMBER",
                },
                {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
                {"id": "BASH", "upgrade_level": 0, "enchantment": "GOOPY"},
            ],
        }
        decision = rp._build_shop(state)
        assert decision is not None
        removes = [
            (a["deck_idx"], lbl)
            for a, lbl in zip(decision.actions, decision.labels)
            if a["action"] == "shop_remove_card"
        ]
        # deck_idx is a master-deck index, so skipping index 0 must not
        # renumber what follows.
        assert removes == [(1, "remove:STRIKE_IRONCLAD"), (2, "remove:BASH")]

    def test_an_absent_or_unknown_enchantment_reads_as_removable(self) -> None:
        # An older bridge build omits the field entirely; it must behave
        # exactly as it did before this filter existed.
        state = copy.deepcopy(SHOP_STATE)
        state["run_state"] = {
            **_RUN_STATE,
            "master_deck": [
                {"id": "BASH", "upgrade_level": 0},
                {"id": "ANGER", "upgrade_level": 0, "enchantment": "NOT_AN_ENCHANTMENT"},
            ],
        }
        decision = rp._build_shop(state)
        assert decision is not None
        offered = [
            lbl
            for a, lbl in zip(decision.actions, decision.labels)
            if a["action"] == "shop_remove_card"
        ]
        assert offered == ["remove:BASH", "remove:ANGER"]

    def test_the_wire_removable_flag_outranks_the_reconstruction(self) -> None:
        # GH #592. The bridge carries `CardModel.IsRemovable` -- the single
        # filter `FromDeckForRemoval` applies -- so when it is present it IS
        # the answer, in BOTH directions. check19's OUTBREAK is the case the
        # reconstruction cannot see (removable by id, no enchantment on the
        # wire, and still withheld by the screen); the enchanted Strike is the
        # opposite case, where a wire flag of True must beat a client-side
        # "Tezcatara means Eternal" verdict rather than being ANDed with it.
        state = copy.deepcopy(SHOP_STATE)
        state["run_state"] = {
            **_RUN_STATE,
            "master_deck": [
                {"id": "BASH", "upgrade_level": 0},
                {"id": "OUTBREAK", "upgrade_level": 0, "is_removable": False},
                {
                    "id": "STRIKE_IRONCLAD",
                    "upgrade_level": 0,
                    "enchantment": "TEZCATARAS_EMBER",
                    "is_removable": True,
                },
            ],
        }
        decision = rp._build_shop(state)
        assert decision is not None
        removes = [
            (a["deck_idx"], lbl)
            for a, lbl in zip(decision.actions, decision.labels)
            if a["action"] == "shop_remove_card"
        ]
        assert removes == [(0, "remove:BASH"), (2, "remove:STRIKE_IRONCLAD")]

    def test_a_non_bool_removable_flag_falls_back_to_the_reconstruction(self) -> None:
        # Only a real bool is a verdict. A missing field (older mod) or a
        # null/garbage one must not read as "not removable" -- that would
        # empty the whole fan-out and hand every shop removal back to the
        # differently-trained `remove_deck_card` surface.
        assert rp._wire_card_is_removable({"id": "BASH"}) is True
        assert rp._wire_card_is_removable({"id": "BASH", "is_removable": None}) is True
        assert rp._wire_card_is_removable({"id": "BASH", "is_removable": "yes"}) is True
        assert rp._wire_card_is_removable({"id": "ASCENDERS_BANE"}) is False

    def test_the_eternal_enchantment_set_comes_from_the_engine(self) -> None:
        # Derived by asking the engine, not restated here: the engine applies
        # each enchantment to a probe card and checks its own predicate.
        ench = rp._eternal_granting_enchantments()
        assert "TEZCATARAS_EMBER" in ench
        assert "GOOPY" not in ench
        assert "STEADY" not in ench

    def test_a_target_the_screen_does_not_offer_is_reported(self, caplog) -> None:
        # The miss is harmless -- the screen re-decides -- but it is the ONLY
        # signal that the fan-out's `master_deck` and the screen's
        # `FromDeckForRemoval` list disagree. check19 carries a
        # `remove:OUTBREAK` pick against a screen with no Outbreak, and
        # Outbreak is not Eternal, so nothing in the removability rule
        # explains it. Silence is why it is still open.
        session: dict[str, object] = {"shop_remove_target": ("OUTBREAK", False)}
        decision = rp.PpoDecision(
            obs={},
            actions=[{"action": "remove_deck_card", "deck_idx": 0}],
            metrics={},
            responses=[{"action": "choose", "index": 0}],
            labels=["Strike"],
        )
        state = {
            "cards": [{"id": "STRIKE_SILENT", "index": 0}],
            "run_state": {
                "master_deck": [
                    {"id": "STRIKE_SILENT"},
                    {"id": "OUTBREAK", "is_removable": False},
                ]
            },
        }
        with caplog.at_level(logging.WARNING, logger="realgame_ppo"):
            assert rp.shop_removal_carry_index(state, session, decision) is None
        assert "OUTBREAK" in caplog.text
        assert "STRIKE_SILENT" in caplog.text
        # The deck behind the miss, with the game's own verdict beside each
        # card -- the artifact whose absence kept GH #592 a guess.
        assert "the deck behind it was" in caplog.text
        assert "OUTBREAK(fixed)" in caplog.text
        assert "shop_remove_target" not in session, "a miss still consumes the stash"

    def test_the_non_removable_set_comes_from_the_engine(self) -> None:
        # Not a hand-kept list. Two previous attempts to maintain this rule in
        # Python got it wrong, so the only assertion worth making is that the
        # engine is the source: every entry is Eternal in the C# decompile,
        # and no basic is present.
        non_removable = rp._non_removable_cards()
        assert "ASCENDERS_BANE" in non_removable
        assert "ENTHRALLED" in non_removable
        assert not any(c.startswith(("STRIKE_", "DEFEND_")) for c in non_removable)

    def test_shop_leave_shop_response_is_choose_index_zero(self) -> None:
        decision = rp._build_shop(SHOP_STATE)
        assert decision is not None
        leave_pos = decision.actions.index({"action": "leave_shop"})
        assert decision.responses[leave_pos] == {"action": "choose", "index": 0}

    def test_shop_buy_potion_dropped_when_belt_full(self) -> None:
        # Buying with a full belt is a guaranteed no-op in the real game
        # (PotionCmd.TryToProcure -> TooFull, no gold spent), so a deterministic
        # policy would re-pick the same potion until SpireMentor's 50-attempt cap
        # force-leaves the shop. The option must not reach the net at all.
        potion_opt = {
            "index": 4,
            "id": "buy_potion",
            "action": "buy_potion",
            "price": 39,
            "enabled": True,
            "item": {"id": "ENERGY_POTION"},
        }
        state = {**SHOP_STATE, "options": [*SHOP_STATE["options"], potion_opt]}
        full_belt = [
            {"slot": 0, "empty": False, "id": "FIRE_POTION"},
            {"slot": 1, "empty": False, "id": "BLOCK_POTION"},
            {"slot": 2, "empty": False, "id": "ENERGY_POTION"},
        ]
        state["run_state"] = {**_RUN_STATE, "potions": full_belt}

        decision = rp._build_shop(state)
        assert decision is not None
        assert all(a["action"] != "shop_buy_potion" for a in decision.actions)
        assert all(resp["index"] != 4 for resp in decision.responses)
        assert decision.obs["shop"]["potions"] == []
        assert len(decision.actions) == len(decision.responses) == len(decision.labels)

    def test_shop_buy_potion_kept_when_belt_has_space(self) -> None:
        potion_opt = {
            "index": 4,
            "id": "buy_potion",
            "action": "buy_potion",
            "price": 39,
            "enabled": True,
            "item": {"id": "ENERGY_POTION"},
        }
        state = {**SHOP_STATE, "options": [*SHOP_STATE["options"], potion_opt]}
        state["run_state"] = {
            **_RUN_STATE,
            "potions": [
                {"slot": 0, "empty": False, "id": "FIRE_POTION"},
                {"slot": 1, "empty": True},
            ],
        }

        decision = rp._build_shop(state)
        assert decision is not None
        assert {"action": "shop_buy_potion", "idx": 0} in decision.actions
        assert decision.obs["shop"]["potions"] == [
            {"id": "ENERGY_POTION", "price": 39, "available": True}
        ]

    def test_map_select_move_actions_carry_bridge_cols_in_order(self) -> None:
        decision = rp._build_map_select(MAP_SELECT_STATE)
        assert decision is not None
        assert decision.actions == [
            {"action": "move", "col": 2},
            {"action": "move", "col": 3},
        ]

    def test_map_select_labels_name_the_branch_a_player_can_see(self) -> None:
        # Graph coordinates are not on screen; left/right is. Columns
        # increase left-to-right, so col order IS the on-screen order.
        decision = rp._build_map_select(MAP_SELECT_STATE)
        assert decision is not None
        assert decision.labels == ["Monster (left)", "Rest Site (right)"]

    def test_map_select_single_option_needs_no_side(self) -> None:
        state = {**MAP_SELECT_STATE, "nodes": [{"index": 0, "type": "Elite", "row": 6, "col": 4}]}
        decision = rp._build_map_select(state)
        assert decision is not None
        assert decision.labels == ["Elite"]

    def test_map_select_three_way_branch_gets_a_middle(self) -> None:
        state = {
            **MAP_SELECT_STATE,
            "nodes": [
                {"index": 0, "type": "Shop", "row": 6, "col": 4},
                {"index": 1, "type": "Monster", "row": 6, "col": 1},
                {"index": 2, "type": "Elite", "row": 6, "col": 3},
            ],
        }
        decision = rp._build_map_select(state)
        assert decision is not None
        # Labels stay in OPTION order (index 0 first); only the side word is
        # derived from the column, or the label would name the wrong node.
        assert decision.labels == ["Shop (right)", "Monster (left)", "Elite (middle)"]

    def test_map_select_handles_missing_map_object(self) -> None:
        state = {**MAP_SELECT_STATE, "map": None}
        decision = rp._build_map_select(state)
        assert decision is not None
        # No graph -> None, the same "no map" shape ppo.env.observe produces
        # (mapfeat treats None and an empty rows list identically).
        assert decision.obs["map"] is None
        assert decision.obs["map_coord"] is None

    def test_map_select_no_nodes_returns_none(self) -> None:
        assert rp._build_map_select({**MAP_SELECT_STATE, "nodes": []}) is None

    def test_rest_site_no_options_returns_none(self) -> None:
        assert rp._build_rest_site({**REST_SITE_STATE, "options": []}) is None

    def test_rest_site_idx_is_the_game_slot(self) -> None:
        """The mod renumbers ``index`` after dropping disabled options; the
        ``rest_option`` idx must be the C# slot (``option_index``) so Dig
        behind a disabled Smith is not encoded as ``rest_smith``."""
        options = [
            {"index": 0, "option_index": 0, "id": "HEAL", "label": "Rest", "enabled": True},
            {"index": 1, "option_index": 2, "id": "DIG", "label": "Dig", "enabled": True},
        ]
        decision = rp._build_rest_site({**REST_SITE_STATE, "options": options})
        assert decision is not None
        assert decision.actions == [
            {"action": "rest_option", "idx": 0},
            {"action": "rest_option", "idx": 2},
        ]
        assert decision.responses[1] == {"action": "choose", "index": 1}

    def test_rest_site_idx_falls_back_to_position_for_old_mods(self) -> None:
        options = [
            {"index": 0, "id": "HEAL", "label": "Rest", "enabled": True},
            {"index": 1, "id": "DIG", "label": "Dig", "enabled": True},
        ]
        decision = rp._build_rest_site({**REST_SITE_STATE, "options": options})
        assert decision is not None
        assert [a["idx"] for a in decision.actions] == [0, 1]

    def test_event_no_options_returns_none(self) -> None:
        assert rp._build_event({**EVENT_STATE, "options": []}) is None

    def test_boss_relic_no_options_returns_none(self) -> None:
        assert rp._build_boss_relic({**BOSS_RELIC_STATE, "options": []}) is None


class TestRewardScreenChoice:
    """reward_screen must collect every reward before proceeding, in the same
    order as the headless engine / RealGame backend (reward_screen_choice):
    non-card rewards (gold/potion/relic) first, then CardReward, then proceed.
    The random fallback used to bail via 'proceed' first, starving the deck of
    card rewards -- a major live-win-rate cause. Both drivers share this one
    function so every backend collects rewards identically."""

    def test_collects_non_card_rewards_before_card_and_proceed(self) -> None:
        # gold present -> take gold first (index 0), NOT the card or proceed.
        options = [
            {"index": 0, "id": "gold:18", "action": "pick_reward"},
            {"index": 1, "id": "CardReward", "action": "pick_reward"},
            {"index": 2, "id": "proceed", "action": "proceed"},
        ]
        assert reward_screen_choice(options) == 0

    def test_opens_card_reward_when_only_card_and_proceed_left(self) -> None:
        options = [
            {"index": 0, "id": "CardReward", "action": "pick_reward"},
            {"index": 1, "id": "proceed", "action": "proceed"},
        ]
        assert reward_screen_choice(options) == 0

    def test_proceeds_only_when_nothing_left_to_claim(self) -> None:
        assert reward_screen_choice([{"index": 0, "id": "proceed", "action": "proceed"}]) == 0

    def test_card_reward_deferred_even_when_listed_first(self) -> None:
        # CardReward listed before gold: still take gold (non-card) first.
        options = [
            {"index": 0, "id": "CardReward", "action": "pick_reward"},
            {"index": 1, "id": "gold:9", "action": "pick_reward"},
            {"index": 2, "id": "proceed", "action": "proceed"},
        ]
        assert reward_screen_choice(options) == 1

    def test_no_actionable_option_returns_none(self) -> None:
        assert reward_screen_choice([]) is None


class TestRewardScreenCardSkipLoop:
    """After the policy SKIPS a card reward, the bridge re-shows the reward
    screen with the CardReward button still present. Re-opening it would
    skip-and-loop forever (AutoSlay 3600s timeout). pick_noncombat_action_ppo
    must PROCEED past a just-skipped card instead of re-opening it."""

    _REWARD_SCREEN: ClassVar[dict] = {
        "type": "reward_screen",
        "options": [
            {"index": 0, "id": "CardReward", "action": "pick_reward"},
            {"index": 1, "id": "proceed", "action": "proceed"},
        ],
    }

    def test_opens_card_when_not_yet_skipped(self) -> None:
        session: dict = {}
        action, _ = rp.pick_noncombat_action_ppo(
            self._REWARD_SCREEN, random.Random(0), rp.PpoPolicyCache(None), session=session
        )
        assert action == {"action": "choose", "index": 0}  # opens CardReward

    def test_proceeds_when_card_just_skipped(self) -> None:
        session: dict = {"card_reward_skipped": True}
        action, _ = rp.pick_noncombat_action_ppo(
            self._REWARD_SCREEN, random.Random(0), rp.PpoPolicyCache(None), session=session
        )
        assert action == {"action": "choose", "index": 1}  # proceeds past it
        assert "card_reward_skipped" not in session  # flag consumed


_FULL_BELT = [
    {"slot": 0, "id": "FIRE_POTION", "empty": False},
    {"slot": 1, "id": "BLOCK_POTION", "empty": False},
    {"slot": 2, "id": "ENERGY_POTION", "empty": False},
]


def _swap_group(
    first_index: int, button: int, potion: str = "STABLE_SERUM"
) -> list[dict[str, Any]]:
    """The mod's options for one full-belt potion reward (GH #1069)."""
    reward = {"id": potion, "name": potion.title(), "slot": -1}
    group = [
        {
            "index": first_index,
            "id": "PotionReward",
            "action": "potion_swap",
            "label": "decline",
            "reward": reward,
            "reward_button": button,
            "discard_slot": None,
        }
    ]
    for k, held in enumerate(p["id"] for p in _FULL_BELT):
        group.append(
            {
                "index": first_index + 1 + k,
                "id": "PotionReward",
                "action": "potion_swap",
                "label": f"discard {held}",
                "reward": reward,
                "reward_button": button,
                "discard_slot": k,
            }
        )
    return group


class TestRewardScreenPotionSwap:
    """GH #1069: a potion reward that finds a full belt is a swap decision in
    the real game (discard a held potion, then claim it). The mod offers it as
    a group of ``potion_swap`` options; the driver answers it after the other
    rewards and before the card, as the engines order their ``potion_swap``
    decision, and declines unless ``STS2_RUNAI_POTION_SWAP`` is on."""

    _STATE: ClassVar[dict] = {
        **_envelope("reward_screen", {**_RUN_STATE, "potions": _FULL_BELT}),
        "options": [
            {"index": 0, "id": "CardReward", "action": "pick_reward"},
            *_swap_group(1, 0),
            {"index": 5, "id": "proceed", "action": "proceed"},
        ],
    }

    def test_collector_answers_the_swap_before_the_card(self) -> None:
        assert reward_screen_choice(self._STATE["options"]) == 1  # the decline

    def test_collector_takes_plain_rewards_before_the_swap(self) -> None:
        options = [{"index": 9, "id": "gold:12", "action": "pick_reward"}, *self._STATE["options"]]
        assert reward_screen_choice(options) == 9

    def test_group_is_the_first_blocked_potion_decline_first(self) -> None:
        options = [*_swap_group(4, 1, "FEAR_POTION"), *_swap_group(0, 0)]
        group = potion_swap_group(list(reversed(options)))
        assert [o["discard_slot"] for o in group] == [None, 0, 1, 2]
        assert {o["reward"]["id"] for o in group} == {"STABLE_SERUM"}

    def test_builder_matches_the_engine_surface(self) -> None:
        decision = rp._build_potion_swap(self._STATE)
        assert decision is not None
        assert decision.actions == [
            {"action": "potion_swap", "discard_slot": None},
            {"action": "potion_swap", "discard_slot": 0},
            {"action": "potion_swap", "discard_slot": 1},
            {"action": "potion_swap", "discard_slot": 2},
        ]
        assert decision.obs["pending_potion_swap"] == "STABLE_SERUM"
        assert decision.obs["potions"] == ["FIRE_POTION", "BLOCK_POTION", "ENERGY_POTION"]
        assert decision.responses == [{"action": "choose", "index": i} for i in (1, 2, 3, 4)]
        assert decision.labels[0] == "decline"

    def test_flag_off_declines_without_asking_the_net(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rp, "_live_potion_swap_enabled", lambda: False)
        policies, spy = _spy_policies()
        action, extra = rp.pick_noncombat_action_ppo(
            self._STATE, random.Random(0), policies, session={}
        )
        assert action == {"action": "choose", "index": 1}
        assert extra is None
        assert spy.seen == []

    def test_flag_on_asks_the_net(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rp, "_live_potion_swap_enabled", lambda: True)
        policies, spy = _spy_policies()
        action, _ = rp.pick_noncombat_action_ppo(
            self._STATE, random.Random(0), policies, session={}
        )
        assert len(spy.seen) == 1
        obs, actions, _metrics = spy.seen[0]
        assert obs["pending_potion_swap"] == "STABLE_SERUM"
        assert all(a["action"] == "potion_swap" for a in actions)
        assert action["action"] == "choose"
        assert action["index"] in (1, 2, 3, 4)
        assert "annotations" in action

    def test_flag_reads_the_shared_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if _RS_SKIP_REASON is not None:
            pytest.skip(_RS_SKIP_REASON)
        monkeypatch.delenv("STS2_RUNAI_POTION_SWAP", raising=False)
        assert rp._live_potion_swap_enabled() is False
        monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")
        assert rp._live_potion_swap_enabled() is True


# ---------------------------------------------------------------------------
# Tier 2: real inference against the committed Ironclad checkpoint.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _TORCH_SKIP_REASON is not None, reason=_TORCH_SKIP_REASON or "torch unavailable"
)
@pytest.mark.skipif(_CKPT_SKIP_REASON is not None, reason=_CKPT_SKIP_REASON or "no checkpoint")
class TestRealInference:
    """The real committed Ironclad champion net, exercised against every
    PPO-covered surface. Legality + well-formed annotations only -- these
    are not behavioral/quality assertions (the net's actual pick is
    whatever it learned, not something this test should pin)."""

    @pytest.fixture
    def policies(self) -> rp.PpoPolicyCache:
        return rp.PpoPolicyCache(None)

    @pytest.mark.parametrize(
        "state",
        [
            MAP_SELECT_STATE,
            CARD_REWARD_STATE,
            SHOP_STATE,
            REST_SITE_STATE,
            EVENT_STATE,
            BOSS_RELIC_STATE,
        ],
    )
    def test_returns_legal_action_no_error(self, state, policies) -> None:
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(state, rng, policies)

        assert extra is None
        assert action["action"] in ("choose", "skip")
        if action["action"] == "choose":
            options = state.get("nodes") or state.get("options") or state.get("cards")
            valid = {int(o.get("index", i)) for i, o in enumerate(options)}
            assert action["index"] in valid

    def test_annotations_well_formed(self, policies) -> None:
        rng = random.Random(0)
        action, _ = rp.pick_noncombat_action_ppo(CARD_REWARD_STATE, rng, policies)

        ann = action["annotations"]
        assert isinstance(ann["note"], str) and ann["note"]
        assert len(ann["cards"]) == 4  # 3 cards + skip candidate
        total = 0.0
        for c in ann["cards"]:
            assert 0.0 <= c["score"] <= 1.0
            assert isinstance(c["label"], str)
            total += c["score"]
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_single_option_auto_taken_without_inference(self, policies, monkeypatch) -> None:
        """A one-candidate decision must not touch the net at all (mirrors
        ppo.env's "single-option decisions are auto-taken")."""
        state = {**REST_SITE_STATE, "options": [REST_SITE_STATE["options"][0]]}

        def _boom(*a, **k):
            raise AssertionError("distribution() must not be called for a single candidate")

        policy, _ = policies.get("IRONCLAD")
        monkeypatch.setattr(policy, "distribution", _boom)

        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(state, rng, policies)
        assert extra is None
        assert action["action"] == "choose"
        assert action["index"] == 0

    def test_greedy_is_deterministic(self, policies) -> None:
        rng1, rng2 = random.Random(1), random.Random(2)
        a1, _ = rp.pick_noncombat_action_ppo(SHOP_STATE, rng1, policies, temperature=0.0)
        a2, _ = rp.pick_noncombat_action_ppo(SHOP_STATE, rng2, policies, temperature=0.0)
        assert a1["index"] == a2["index"]


# ---------------------------------------------------------------------------
# card_select -> deck-mutation surface (2026-07-12 live-gap root cause:
# these were all uniformly random, wiping out the net's deck-curation policy).
# ---------------------------------------------------------------------------


def _cs_card(cid: str, index: int, upgrade_level: int = 0) -> dict[str, Any]:
    return {
        "id": cid,
        "name": cid.replace("_", " ").title(),
        "index": index,
        "upgrade_level": upgrade_level,
        "type": "Attack",
        "rarity": "Common",
    }


CARD_SELECT_STATE = {
    **_envelope("card_select"),
    "cards": [_cs_card("STRIKE_IRONCLAD", 0), _cs_card("BASH", 1), _cs_card("THUNDERCLAP", 2)],
    "min_select": 1,
    "max_select": 1,
    "min_count": 1,
    "max_count": 1,
    "cancelable": False,
}


class TestMidCombatSelectionsAreNotThisModulesJob:
    """A mid-combat ``card_select`` no longer reaches the PPO surface at all.

    It is answered off the fight's persistent combat mirror
    (``realgame_mcts.pick_card_select_mirrored``), which searches the choice
    the played card already opened. What this replaced — a pending-choice
    replay whose option list had to match exactly, then a ``prompt_key``
    quality heuristic, then uniform random — is deleted, so the only contract
    left here is that an unmapped context takes the documented fallback rather
    than silently inventing a pick.
    """

    def test_unmapped_context_takes_the_fallback(self) -> None:
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        state = {
            **_envelope("card_select"),
            "cards": [
                {**_cs_card("STRIKE_IRONCLAD", 0), "rarity": "Basic"},
                {**_cs_card("PYRE", 1), "rarity": "Rare"},
            ],
            "min_select": 1,
            "max_select": 1,
            "min_count": 1,
            "max_count": 1,
            "cancelable": False,
            "prompt_key": "TO_UPGRADE",
        }
        session = {"prev_msg_type": "combat_action"}
        action, extra = rp.pick_noncombat_action_ppo(state, rng, policies, session=session)
        assert extra is None
        assert action["action"] in ("choose", "skip", "cancel")


class TestMapCache:
    """Empty live map graphs (~15% of map_select messages) are patched from
    the per-act session cache -- the act's graph is immutable."""

    def test_empty_map_patched_from_cache(self) -> None:
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        good_map = MAP_SELECT_STATE.get("map") or {}
        assert good_map.get("nodes"), "fixture must carry a non-empty map"
        session: dict[str, Any] = {}
        # First call caches the good map.
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, rng, policies, session=session)
        assert session["last_map"] == (MAP_SELECT_STATE["act"], good_map)
        # Second call with an empty map must not clobber the cache.
        degraded = {**MAP_SELECT_STATE, "map": {}}
        rp.pick_noncombat_action_ppo(degraded, rng, policies, session=session)
        assert session["last_map"] == (MAP_SELECT_STATE["act"], good_map)

    def test_cache_not_used_across_acts(self) -> None:
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        session: dict[str, Any] = {}
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, rng, policies, session=session)
        degraded = {**MAP_SELECT_STATE, "map": {}, "act": (MAP_SELECT_STATE["act"] or 0) + 1}
        # Must not raise; the stale other-act graph stays unused.
        action, _ = rp.pick_noncombat_action_ppo(degraded, rng, policies, session=session)
        assert action["action"] in ("choose", "skip")


class _SpyPolicy:
    """Stands in for NetPolicy: uniform distribution, records every obs
    triple -- lets tests observe what the builders + session patching
    actually feed the net, without torch or a checkpoint."""

    def __init__(self) -> None:
        self.seen: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []

    def distribution(self, obs, actions, metrics):  # type: ignore[no-untyped-def]
        self.seen.append((obs, actions, metrics))
        n = len(actions)
        return [1.0 / n] * n, 0.5


def _spy_policies() -> tuple[rp.PpoPolicyCache, _SpyPolicy]:
    policies = rp.PpoPolicyCache(None)
    spy = _SpyPolicy()
    policies._cache["ironclad"] = (spy, None)
    return policies, spy


class TestTotalFloorOffset:
    """GH #1051: the bridge's ``floor`` is the DLL's ``RunState.TotalFloor``,
    which counts every act's start room (Neow, the Ancients); the engine's
    ``total_floor`` the champion trained on counts none. So the offset is the
    1-based act, not a flat 1 -- the old ``- 1`` fed acts 2-3 one and two
    floors too deep. Twin of ``wire.rs``'s ``total_floor_offset_is_the_act_number``."""

    @pytest.mark.parametrize(
        ("act", "floor", "engine_floor", "act_index"),
        [
            (1, 1, 0, 1),  # Neow: engine 0, DLL 1
            (1, 12, 11, 1),
            (2, 19, 17, 2),  # act 2's first map choice after a 17-floor act 1
            (3, 48, 45, 3),  # a won A0 run: game 48, engine 45
            (None, 12, 11, 1),  # no act on the wire reads as act 1
            (0, 12, 11, 1),  # so does a zero act -- and `act_index` with it
            (-1, 12, 11, 1),  # and a negative one (wire.rs agrees)
            (2, 1, 0, 2),  # never negative
        ],
    )
    def test_total_floor_subtracts_the_act(self, act, floor, engine_floor, act_index) -> None:
        state = {**_envelope("map_select"), "floor": floor}
        if act is None:
            state.pop("act")
        else:
            state["act"] = act
        metrics = rp._base_metrics(state, _RUN_STATE)
        assert metrics["total_floor"] == engine_floor
        assert metrics["act_index"] == act_index


class TestMapInjection:
    """The training obs carry the act map (+ coord, + engine act_floor =
    row + 1) on EVERY decision; the bridge only sends the map on map_select.
    2026-07-12 obs-parity audit: 76% of live net decisions had zero map-node
    tokens and act_floor was hardwired 0. The session cache must patch both
    into every other PPO surface."""

    def test_map_select_obs_carries_map_coord_and_act_floor(self) -> None:
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, random.Random(0), policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] == {
            "rows": [MAP_SELECT_STATE["map"]["nodes"]],
            "boss": MAP_SELECT_STATE["map"]["boss"],
        }
        assert obs["map_coord"] == (2, 5)  # fixture's current
        assert metrics["act_floor"] == 6  # row + 1 (runstate.rs semantics)

    def test_chosen_node_recorded_as_map_pos(self) -> None:
        policies, _ = _spy_policies()
        session: dict[str, Any] = {}
        action, _ = rp.pick_noncombat_action_ppo(
            MAP_SELECT_STATE, random.Random(0), policies, session=session
        )
        chosen = next(n for n in MAP_SELECT_STATE["nodes"] if n["index"] == action["index"])
        assert session["map_pos"] == (chosen["col"], chosen["row"])

    def test_act_map_patched_into_card_reward(self) -> None:
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        rng = random.Random(0)
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, rng, policies, session=session)
        rp.pick_noncombat_action_ppo(CARD_REWARD_STATE, rng, policies, session=session)
        obs, _, metrics = spy.seen[-1]
        col, row = session["map_pos"]
        assert obs["map"] is not None
        assert obs["map"]["rows"][0] == MAP_SELECT_STATE["map"]["nodes"]
        assert obs["map_coord"] == (col, row)  # current moved to the chosen node
        assert metrics["act_floor"] == row + 1

    def test_stale_other_act_map_not_injected(self) -> None:
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        rng = random.Random(0)
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, rng, policies, session=session)
        act2_reward = {**CARD_REWARD_STATE, "act": (CARD_REWARD_STATE["act"] or 0) + 1}
        rp.pick_noncombat_action_ppo(act2_reward, rng, policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] is None
        assert obs["map_coord"] is None
        assert metrics["act_floor"] == 0

    def test_fallback_map_answer_still_records_map_pos(self) -> None:
        # No usable policy -> random-legal fallback answers the map_select;
        # the position tracker must still follow the chosen node.
        policies = rp.PpoPolicyCache(None)
        policies._cache["ironclad"] = (None, "no checkpoint (test)")
        session: dict[str, Any] = {}
        action, _ = rp.pick_noncombat_action_ppo(
            MAP_SELECT_STATE, random.Random(0), policies, session=session
        )
        chosen = next(n for n in MAP_SELECT_STATE["nodes"] if n["index"] == action["index"])
        assert session["map_pos"] == (chosen["col"], chosen["row"])


def _wire_map(current: dict[str, Any] | None) -> dict[str, Any]:
    """A bridge ``map`` object as the 2026-08-02 mod attaches to run-level
    envelopes -- same graph as ``MAP_SELECT_STATE``'s, with a chosen
    ``current``."""
    return {**MAP_SELECT_STATE["map"], "current": current}


class TestEnvelopeMapPassthrough:
    """2026-08-02 run-level dialect audit: SpireMentor attaches the act map to
    EVERY run-level message's envelope (deploy-pending mod build), not just
    ``map_select`` -- the live twin of headless_eval's ``get_map()`` seeding
    (headless_eval.py:1510-1519). Before that, every off-map live decision
    (Neow, the act-start Ancient relic offer = the boss-relic surface, and
    the card_selects those open) was decided MAP-BLIND. A map-carrying
    non-map_select message must (a) feed its own map + authoritative
    ``current`` to the net, (b) refresh the per-act cache and position
    tracker for messages that still arrive bare, and (c) keep the pre-act
    row-0 ``current`` OUT of the position tracker so the training-time
    act-start encoding (``map_coord`` None, ``act_floor`` 0) survives."""

    def test_wire_map_used_and_cached(self) -> None:
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        state = {**CARD_REWARD_STATE, "map": _wire_map({"col": 3, "row": 6})}
        rp.pick_noncombat_action_ppo(state, random.Random(0), policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] == {
            "rows": [state["map"]["nodes"]],
            "boss": state["map"]["boss"],
        }
        assert obs["map_coord"] == (3, 6)  # the wire's own current, verbatim
        assert metrics["act_floor"] == 7  # row + 1 (runstate.rs semantics)
        assert session["last_map"] == (state["act"], state["map"])
        assert session["map_pos"] == (3, 6)

    def test_act_start_row0_current_is_not_a_position(self) -> None:
        # Neow / act-start Ancient: the bridge reports the pre-act row-0 node
        # as `current` -- a node the act graph itself never contains. The map
        # tokens must flow, but coord/act_floor must keep the degenerate
        # act-start encoding training always saw (v33wr-class OOD guard).
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        state = {**EVENT_STATE, "map": _wire_map({"col": 3, "row": 0})}
        rp.pick_noncombat_action_ppo(state, random.Random(0), policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] is not None
        assert obs["map"]["rows"][0] == state["map"]["nodes"]
        assert obs["map_coord"] is None
        assert metrics["act_floor"] == 0
        assert "map_pos" not in session  # row 0 is not a position
        assert session["last_map"] == (state["act"], state["map"])

    def test_bare_message_patched_from_envelope_refreshed_cache(self) -> None:
        # A map-carrying event seeds the cache; a later BARE message in the
        # same act (older mod build / mixed deploys) is patched from it, with
        # `current` at the envelope-supplied position.
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        rng = random.Random(0)
        seeded = {**EVENT_STATE, "map": _wire_map({"col": 2, "row": 7})}
        rp.pick_noncombat_action_ppo(seeded, rng, policies, session=session)
        rp.pick_noncombat_action_ppo(CARD_REWARD_STATE, rng, policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] is not None
        assert obs["map"]["rows"][0] == MAP_SELECT_STATE["map"]["nodes"]
        assert obs["map_coord"] == (2, 7)
        assert metrics["act_floor"] == 8

    def test_other_act_envelope_map_replaces_stale_cache(self) -> None:
        # Act boundary: the act-2 Ancient arrives with the ACT-2 graph while
        # the cache still holds act 1 -- the envelope map must win, and the
        # refreshed cache must serve later bare act-2 messages.
        policies, spy = _spy_policies()
        session: dict[str, Any] = {}
        rng = random.Random(0)
        rp.pick_noncombat_action_ppo(MAP_SELECT_STATE, rng, policies, session=session)  # act 1
        act2 = (MAP_SELECT_STATE["act"] or 0) + 1
        ancient = {**EVENT_STATE, "act": act2, "map": _wire_map({"col": 0, "row": 0})}
        rp.pick_noncombat_action_ppo(ancient, rng, policies, session=session)
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] is not None  # no longer map-blind at the act start
        assert obs["map_coord"] is None  # ... but still act-start-encoded
        assert metrics["act_floor"] == 0
        assert session["last_map"][0] == act2
        # The act-1 position must not leak into the act-2 graph: a later
        # BARE act-2 message patches from the refreshed cache with NO
        # current, not the previous act's coordinate.
        assert "map_pos" not in session
        rp.pick_noncombat_action_ppo(
            {**CARD_REWARD_STATE, "act": act2}, rng, policies, session=session
        )
        obs, _, metrics = spy.seen[-1]
        assert obs["map"] is not None
        assert obs["map_coord"] is None
        assert metrics["act_floor"] == 0


@pytest.mark.skipif(_RS_SKIP_REASON is not None, reason=_RS_SKIP_REASON or "")
class TestCardIdVocabTranslation:
    """The Rust registry suffixes colliding card names with ``_CARD``
    (``RAGE_CARD``); the bridge sends the plain game id (``RAGE``), which was
    OOV against the training vocab -- 407/6074 deck tokens in the 12-run live
    corpus (2026-07-12 obs-parity audit)."""

    def test_suffix_family_resolves_non_oov(self) -> None:
        """The contract is the embedding SLOT, not the spelling.

        Since ``vocab.Registry`` started keying on ``canonical()`` (commit
        a082efeef, which strips the trailing ``_CARD``), the bare bridge id
        already lands on the suffixed entry's row, so ``_vocab_card_id``
        leaves it alone -- the suffix rewrite has become the fallback for
        vocabs that predate that commit. Assert what the net actually sees:
        both spellings index the same non-OOV row.
        """
        from ppo import vocab

        reg = vocab.cards()
        for bare in ("RAGE", "FLAME_BARRIER", "SETUP_STRIKE"):
            idx = reg.index(rp._vocab_card_id(bare))
            assert idx != reg.oov, f"{bare} must not be OOV"
            assert idx == reg.index(f"{bare}_CARD")

    def test_status_family_translated(self) -> None:
        """``_STATUS`` does NOT collapse under ``canonical()``, so the
        rewrite is still load-bearing there (2026-07-15 featurizer audit)."""
        from ppo import vocab

        reg = vocab.cards()
        assert reg.index("SLOTH") == reg.oov, "bare SLOTH is genuinely OOV"
        assert rp._vocab_card_id("SLOTH") == "SLOTH_STATUS"

    def test_exact_match_and_unknown_pass_through(self) -> None:
        assert rp._vocab_card_id("ANGER") == "ANGER"
        assert rp._vocab_card_id("TOTALLY_FAKE_CARD_ID") == "TOTALLY_FAKE_CARD_ID"

    def test_card_reward_options_and_deck_translated(self) -> None:
        state = {
            **_envelope("card_reward"),
            "cards": [{"id": "RAGE", "name": "Rage"}, {"id": "ANGER", "name": "Anger"}],
            "can_skip": False,
        }
        state["run_state"] = {**_RUN_STATE, "master_deck": [{"id": "HELLRAISER"}]}
        decision = rp._build_card_reward(state)
        assert decision is not None
        from ppo import vocab

        reg = vocab.cards()
        ids = decision.obs["card_reward_option_ids"]
        assert [reg.index(i) for i in ids] == [reg.index("RAGE_CARD"), reg.index("ANGER")]
        assert reg.index(ids[0]) != reg.oov
        deck = decision.obs["deck"]
        assert [(reg.index(c), up) for c, up in deck] == [(reg.index("HELLRAISER_CARD"), False)]

    def test_card_reward_options_carry_their_upgrade_flag(self) -> None:
        """The real game pre-upgrades reward cards too, over both wire
        spellings. Dropping the flag would put a SpireMentor-served champion on
        a different observation from the one it trained on the moment the
        `reward_upgrade` feature graduates (GH #545)."""
        state = {
            **_envelope("card_reward"),
            "cards": [
                {"id": "RAGE", "name": "Rage"},
                {"id": "ANGER", "name": "Anger+", "upgrade_level": 1},
                {"id": "CLEAVE", "name": "Cleave+", "upgraded": True},
            ],
            "can_skip": False,
        }
        decision = rp._build_card_reward(state)
        assert decision is not None
        assert decision.obs["card_reward_option_upgraded"] == [False, True, True]


class TestEventBbcodeStrip:
    """Bridge event descriptions carry Godot BBCode that broke
    encoder.event_payload's gold/HP parsing (live events scored (0,0))."""

    def test_strip_bbcode_markers_keep_text(self) -> None:
        assert (
            rp._strip_bbcode("Lose [red]6[/red] HP. Gain [blue]43[/blue] [gold]Gold[/gold].")
            == "Lose 6 HP. Gain 43 Gold."
        )

    def test_strip_img_blocks_entirely(self) -> None:
        assert (
            rp._strip_bbcode("Gain [img]res://images/x.png[/img] at the start of each turn.")
            == "Gain  at the start of each turn."
        )

    def test_event_obs_descriptions_parse_payload(self) -> None:
        from ppo.encoder import event_payload

        state = {
            **_envelope("event"),
            "options": [
                {
                    "index": 0,
                    "action": "event_choice",
                    "label": "This",
                    "description": "Lose [red]6[/red] HP. Gain [blue]43[/blue] [gold]Gold[/gold].",
                    "enabled": True,
                    "event_id": "THIS_OR_THAT",
                },
            ],
        }
        decision = rp._build_event(state)
        assert decision is not None
        desc = decision.obs["event_options"][0]["description"]
        gold, hp = event_payload(desc, max_hp=75)
        assert gold == pytest.approx(43.0)
        assert hp == pytest.approx(-6 / 75)


class TestEventNoSuicideGuard:
    """GH #300: the event-blind PPO suicide-loops on SlipperyBridge's
    escalating "Hold On"; ``_build_event`` must withhold options whose
    parsed HP cost is lethal at current HP (unless every option is)."""

    @staticmethod
    def _bridge_options(hp_loss: int) -> list[dict[str, Any]]:
        # Bridge shape: the EVENT-level DynamicVars ride on every option
        # (RunSimulator.cs ~3765); only Hold On's description references
        # {HpLoss}, so the description is what localizes the cost.
        return [
            {
                "index": 0,
                "title": "Overcome",
                "description": "Lose a random card.",
                "enabled": True,
                "vars": {"HpLoss": hp_loss},
            },
            {
                "index": 1,
                "title": "Hold On",
                "description": "Lose [red]{HpLoss}[/red] HP. The card in the above option is randomized.",
                "enabled": True,
                "vars": {"HpLoss": hp_loss},
            },
        ]

    def test_hp_cost_parses_placeholder_and_literal(self) -> None:
        opts = self._bridge_options(7)
        assert rp._event_option_hp_cost(opts[0]) == 0
        assert rp._event_option_hp_cost(opts[1]) == 7
        # Engine-dialect literal form.
        assert rp._event_option_hp_cost({"description": "Take 3 damage, reroll card."}) == 3
        # Unresolvable placeholder (no vars) must not mask anything.
        assert rp._event_option_hp_cost({"description": "Lose {HpLoss} HP."}) == 0

    def test_lethal_option_withheld(self) -> None:
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 5}),
            "options": self._bridge_options(7),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]

    def test_exact_hp_cost_is_lethal(self) -> None:
        # lose_hp(cost) with cost == hp lands on 0 HP = death, same
        # boundary as the engine-side guard's `hp + delta <= 0`.
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 7}),
            "options": self._bridge_options(7),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]

    def test_survivable_option_kept(self) -> None:
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 50}),
            "options": self._bridge_options(7),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2

    def test_all_lethal_leaves_choice_to_policy(self) -> None:
        opts = self._bridge_options(7)
        opts[0]["description"] = "Lose [red]{HpLoss}[/red] HP."
        state = {**_envelope("event", {**_RUN_STATE, "hp": 5}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2

    def test_lethal_option_at_index_zero_is_withheld(self) -> None:
        """The survivor's response index must come from the option's own
        ``index``, not its position in the filtered list.

        Every other case here puts the lethal option last, so dropping it
        leaves position and index equal and an ``o.get("index", i)`` -> ``i``
        regression would pass unnoticed.
        """
        opts = self._bridge_options(7)
        opts[0], opts[1] = opts[1], opts[0]  # lethal Hold On now first
        state = {**_envelope("event", {**_RUN_STATE, "hp": 5}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}], (
            "must answer with Overcome's own index 0, not its new position 0 by luck"
        )
        # And the surviving option really is Overcome, not Hold On.
        assert "HP" not in decision.obs["event_options"][0]["description"]

    def test_veto_is_counted(self) -> None:
        """A guard that silently overrides a trained policy has to be
        countable; otherwise "never fired" and "unreachable" look alike."""
        rp.reset_event_vetoes()
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 5}),
            "options": self._bridge_options(7),
        }
        rp._build_event(state)
        assert rp.EVENT_VETOES == {
            "vetoed": 1,
            "all_lethal": 0,
            "off_model_exit": 0,
            "off_model_no_exit": 0,
        }

        rp.reset_event_vetoes()
        rp._build_event(
            {**_envelope("event", {**_RUN_STATE, "hp": 50}), "options": self._bridge_options(7)}
        )
        assert rp.EVENT_VETOES == {
            "vetoed": 0,
            "all_lethal": 0,
            "off_model_exit": 0,
            "off_model_no_exit": 0,
        }, "no filtering ⇒ no veto"


class TestOffModelPageVeto:
    """GH #487: the engine flattens most multi-page events to their first
    page, so on a live page 2+ the net scores an event id whose learned value
    belongs to a different, cheaper decision -- it deciphered Tablet of Truth
    down to 1/1 max HP at p=94%. ``_build_event`` must take the walk-away
    option on such pages instead of the net's ranking."""

    @staticmethod
    def _tablet_page(page: str) -> list[dict[str, Any]]:
        # Live wire shape (SimNonCombatRoomHandlers.EventOptions): label +
        # text_key + event_id; the later-page decipher cost rides in `vars`.
        return [
            {
                "index": 0,
                "label": "Decipher",
                "description": "Lose [red]{DecipherMaxHpLoss}[/red] Max HP. Upgrade cards.",
                "enabled": True,
                "event_id": "TABLET_OF_TRUTH",
                "text_key": f"TABLET_OF_TRUTH.pages.{page}.options.DECIPHER",
                "vars": {"DecipherMaxHpLoss": 24},
            },
            {
                "index": 1,
                "label": "Give Up",
                "description": "Walk away from the tablet.",
                "enabled": True,
                "event_id": "TABLET_OF_TRUTH",
                "text_key": "TABLET_OF_TRUTH.pages.DECIPHER.options.GIVE_UP",
            },
        ]

    def test_tablet_later_page_forces_walk_away(self) -> None:
        rp.reset_event_vetoes()
        state = {**_envelope("event"), "options": self._tablet_page("DECIPHER_2")}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 1}], (
            "must answer with Give Up's own wire index, not a filtered position"
        )
        assert decision.labels == ["Give Up"]
        assert "GH #487" in decision.source_note
        assert rp.EVENT_VETOES["off_model_exit"] == 1

    def test_initial_page_is_left_to_the_net(self) -> None:
        rp.reset_event_vetoes()
        opts = self._tablet_page("INITIAL")
        opts[0]["text_key"] = "TABLET_OF_TRUTH.pages.INITIAL.options.DECIPHER_1"
        opts[1]["text_key"] = "TABLET_OF_TRUTH.pages.INITIAL.options.SMASH"
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert len(decision.responses) == 2
        assert decision.source_note == ""
        assert rp.EVENT_VETOES["off_model_exit"] == 0
        assert rp.EVENT_VETOES["off_model_no_exit"] == 0

    def test_mixed_initial_and_all_keys_read_as_first_page(self) -> None:
        """EndlessConveyor's FIRST page mixes ``pages.INITIAL`` and shared
        ``pages.ALL`` option keys; the presence of any INITIAL key is what
        marks page 1, not the absence of non-INITIAL ones."""
        rp.reset_event_vetoes()
        opts = self._tablet_page("INITIAL")
        opts[0]["event_id"] = "ENDLESS_CONVEYOR"
        opts[0]["text_key"] = "ENDLESS_CONVEYOR.pages.INITIAL.options.OBSERVE_CHEF"
        opts[1]["event_id"] = "ENDLESS_CONVEYOR"
        opts[1]["text_key"] = "ENDLESS_CONVEYOR.pages.ALL.options.DISH_1"
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert len(decision.responses) == 2
        assert rp.EVENT_VETOES["off_model_exit"] == 0

    def test_engine_modeled_multipage_is_not_vetoed(self) -> None:
        """ColossalFlower's multi-dig pages landed in the engine, so its
        page 2+ is a surface the net trained on and keeps its ranking."""
        rp.reset_event_vetoes()
        opts = self._tablet_page("REACH_DEEPER_2")
        for o in opts:
            o["event_id"] = "COLOSSAL_FLOWER"
        opts[0]["text_key"] = "COLOSSAL_FLOWER.pages.REACH_DEEPER_2.options.REACH_DEEPER_3"
        opts[1]["text_key"] = "COLOSSAL_FLOWER.pages.REACH_DEEPER_2.options.EXTRACT_INSTEAD"
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert len(decision.responses) == 2
        assert rp.EVENT_VETOES["off_model_exit"] == 0

    def test_page_without_exit_is_counted_not_forced(self) -> None:
        """Trial's judgment pages offer GUILTY/INNOCENT -- no walk-away pick.
        The net keeps the (bounded) choice, but the off-model page is counted
        so a sweep can tell "never fired" from "unreachable"."""
        rp.reset_event_vetoes()
        opts = self._tablet_page("NOBLE")
        for o, tail in zip(opts, ("GUILTY", "INNOCENT")):
            o["event_id"] = "TRIAL"
            o["text_key"] = f"TRIAL.pages.NOBLE.options.{tail}"
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert len(decision.responses) == 2
        assert rp.EVENT_VETOES == {
            "vetoed": 0,
            "all_lethal": 0,
            "off_model_exit": 0,
            "off_model_no_exit": 1,
        }

    def test_lethal_filter_runs_before_the_exit_pick(self) -> None:
        """Trial's REJECT page: the only exit-shaped option is Double Down,
        which the world flags WillKillPlayer (it abandons the run). The
        lethal filter must drop it FIRST, leaving Accept forced -- exit-first
        ordering would walk straight into the flagged option."""
        rp.reset_event_vetoes()
        opts = self._tablet_page("REJECT")
        opts[0].update(
            {
                "label": "Accept",
                "event_id": "TRIAL",
                "text_key": "TRIAL.pages.REJECT.options.ACCEPT",
                "description": "Stand trial after all.",
            }
        )
        opts[1].update(
            {
                "label": "Double Down",
                "event_id": "TRIAL",
                "text_key": "TRIAL.pages.REJECT.options.DOUBLE_DOWN",
                "will_kill_player": True,
            }
        )
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]
        assert rp.EVENT_VETOES["vetoed"] == 1
        assert rp.EVENT_VETOES["off_model_exit"] == 0

    def test_options_without_text_keys_never_veto(self) -> None:
        """Headless-dialect fixtures carry no loc keys; no page can be read
        off them, and an unreadable page must never fire the guard."""
        rp.reset_event_vetoes()
        opts = self._tablet_page("DECIPHER_2")
        for o in opts:
            del o["text_key"]
        decision = rp._build_event({**_envelope("event"), "options": opts})
        assert decision is not None
        assert len(decision.responses) == 2
        assert rp.EVENT_VETOES["off_model_exit"] == 0


class TestEventPayloadReachesTheEncoder:
    """The guard stops the suicide; it does not fix why the net wanted it.

    ``ppo.encoder.event_payload`` scrapes numbers out of the option
    description, so an unresolved ``{HpLoss}`` placeholder made the net score
    "Hold On" as free — which is a large part of the p~0.99 prior the guard
    exists to catch. The engine arm never had this problem (it gets
    ``preview_option_effect``'s exact deltas), so the two arms were feeding
    the same checkpoint different features.
    """

    def test_placeholder_is_resolved_before_the_encoder_sees_it(self) -> None:
        opt = {
            "index": 1,
            "title": "Hold On",
            "description": "Lose [red]{HpLoss}[/red] HP.",
            "enabled": True,
            "vars": {"HpLoss": 6},
        }
        state = {**_envelope("event", {**_RUN_STATE, "hp": 50}), "options": [opt]}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["description"] == "Lose 6 HP."

    def test_unknown_placeholder_is_left_visible(self) -> None:
        """A parse failure must still look like one, not like a zero."""
        opt = {
            "index": 0,
            "title": "X",
            "description": "Lose {Mystery} HP.",
            "enabled": True,
            "vars": {"HpLoss": 6},
        }
        state = {**_envelope("event", {**_RUN_STATE, "hp": 50}), "options": [opt]}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["description"] == "Lose {Mystery} HP."

    def test_structured_deltas_are_forwarded_when_present(self) -> None:
        """GH #237's exact numbers beat the regex — but only if they survive
        the trip into the obs, which they previously did not."""
        opt = {
            "index": 0,
            "title": "Overcome",
            "description": "Lose a random card.",
            "enabled": True,
            "gold_delta": 0,
            "hp_delta": -6,
            "max_hp_delta": 0,
        }
        state = {**_envelope("event", {**_RUN_STATE, "hp": 50}), "options": [opt]}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["hp_delta"] == -6
        assert decision.obs["event_options"][0]["max_hp_delta"] == 0

    def test_partial_structured_deltas_do_not_read_as_zero(self) -> None:
        """A missing field must fall back to the description, not assert a
        confident zero — ``structured_event_payload`` needs all three."""
        opt = {
            "index": 0,
            "title": "Overcome",
            "description": "Take 6 damage.",
            "enabled": True,
            "hp_delta": -6,  # gold_delta / max_hp_delta absent
        }
        state = {**_envelope("event", {**_RUN_STATE, "hp": 50}), "options": [opt]}
        decision = rp._build_event(state)
        assert decision is not None
        emitted = decision.obs["event_options"][0]
        assert "gold_delta" not in emitted
        assert "max_hp_delta" not in emitted


def _fake_merchant_state() -> dict[str, Any]:
    path = SNAPSHOT_DIR / "event_fake_merchant.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["state"]


class TestFakeMerchant:
    """GH #873: the live Fake Merchant arrives as the engine's three-option
    event (buy, throw_foul, leave) -- the shape the run policy trained on.
    The net reads an event option by position, so an option that is not
    available keeps its slot instead of letting leave slide onto the throw's.
    """

    def test_a_disabled_throw_keeps_its_slot(self) -> None:
        decision = rp._build_event(_fake_merchant_state())
        assert decision is not None
        assert decision.obs["event_name"] == "FAKE_MERCHANT"
        assert [o["enabled"] for o in decision.obs["event_options"]] == [True, False, True]
        assert decision.actions == [
            {"action": "event_choice", "idx": 0},
            {"action": "event_choice", "idx": 2},
        ]
        assert decision.responses == [
            {"action": "choose", "index": 0},
            {"action": "choose", "index": 2},
        ]
        assert decision.labels == ["Buy a Relic (50g)", "Leave"]

    def test_a_held_foul_potion_offers_the_throw(self) -> None:
        state = _fake_merchant_state()
        state["options"][1]["enabled"] = True
        decision = rp._build_event(state)
        assert decision is not None
        assert [a["idx"] for a in decision.actions] == [0, 1, 2]
        assert [r["index"] for r in decision.responses] == [0, 1, 2]

    def test_every_structured_delta_reaches_the_obs(self) -> None:
        """The relic a purchase brings and the potion a throw costs are what
        tell buy, throw and leave apart; gold alone does not."""
        opts = rp._build_event(_fake_merchant_state()).obs["event_options"]
        assert (opts[0]["gold_delta"], opts[0]["relics_gained"]) == (-50, 1)
        assert opts[1]["potions_lost"] == 1
        assert opts[2]["relics_gained"] == 0

    @pytest.mark.skipif(_RS_SKIP_REASON is not None, reason=_RS_SKIP_REASON or "no sts2_rs")
    def test_matches_the_engines_option_model(self) -> None:
        """Position for position, each option the live message offers
        featurizes like the engine's own option at that position."""
        rm = sts2_rs.RunManager(
            string_seed="FAKEMERCH", character="Ironclad", ascension_level=0, neow_start=True
        )
        assert rm.force_event("FakeMerchant") is not None
        native = rm.event_option_infos()
        assert [o["option_id"] for o in native] == ["buy", "throw_foul", "leave"]

        decision = rp._build_event(_fake_merchant_state())
        wire = decision.obs["event_options"]
        assert len(wire) == len(native)
        # The buy price is rolled per visit (``MerchantRelicEntry.CalcCost``:
        # 50 x Shops.NextFloat(0.85, 1.15)), so the fixture's fixed 50g only
        # matches the engine's roll to within that band.
        keys = [k for k in rp._STRUCTURED_DELTA_KEYS if k != "gold_delta"]
        for action in decision.actions:
            idx = action["idx"]
            assert {k: wire[idx].get(k, 0) for k in keys} == {
                k: native[idx].get(k, 0) for k in keys
            }, f"option {idx} ({native[idx]['option_id']})"
            wire_gold, native_gold = (
                wire[idx].get("gold_delta", 0),
                native[idx].get("gold_delta", 0),
            )
            if wire_gold == 0:
                assert native_gold == 0, f"option {idx}"
            else:
                assert -57 <= native_gold <= -43, f"option {idx}: {native_gold}"

    @pytest.mark.skipif(
        _TORCH_SKIP_REASON is not None, reason=_TORCH_SKIP_REASON or "torch unavailable"
    )
    @pytest.mark.skipif(_CKPT_SKIP_REASON is not None, reason=_CKPT_SKIP_REASON or "no checkpoint")
    def test_the_champion_answers_with_an_offered_option(self) -> None:
        state = _fake_merchant_state()
        action, extra = rp.pick_noncombat_action_ppo(
            state, random.Random(0), rp.PpoPolicyCache(None)
        )
        assert extra is None
        assert action["action"] == "choose"
        assert action["index"] in (0, 2), "never the disabled throw"
        assert action["annotations"]["note"]


class TestEventWorldLethalityFlag:
    """``will_kill_player`` is the C# world's own verdict (``EventOption.
    WillKillPlayer``) and outranks the HP-cost parse.

    The parse cannot see non-damage lethality: ``Trial.DoubleDown`` is
    ``ThatWillKillPlayerIf(_ => true)`` — it opens the abandon-run confirm, so
    live it ENDS the run — and ``TabletOfTruth.Decipher`` keys on MAX HP.
    Neither prints an HP cost to find.
    """

    @staticmethod
    def _trial_options(flag: object) -> list[dict[str, Any]]:
        # Trial's REJECT page. Neither description mentions HP, which is
        # exactly why the cost parse is blind here.
        return [
            {"index": 0, "title": "Accept", "description": "Stand trial.", "enabled": True},
            {
                "index": 1,
                "title": "Double Down",
                "description": "Refuse to be judged.",
                "enabled": True,
                "will_kill_player": flag,
            },
        ]

    def test_hp_cost_parse_is_blind_to_it(self) -> None:
        # Guard the premise: without the flag the option is indistinguishable.
        assert rp._event_option_hp_cost(self._trial_options(True)[1]) == 0

    def test_flagged_option_withheld_at_full_hp(self) -> None:
        # Full HP — the HP-cost guard has nothing to act on, so only the
        # world flag can withhold this.
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 80}),
            "options": self._trial_options(True),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]

    def test_unflagged_options_all_kept(self) -> None:
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 80}),
            "options": self._trial_options(False),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2

    def test_missing_flag_reads_as_unknown_not_safe(self) -> None:
        # A bridge that predates the flag must not have its options treated as
        # cleared; the HP-cost heuristic stays in charge there.
        opts = self._trial_options(True)
        del opts[1]["will_kill_player"]
        state = {**_envelope("event", {**_RUN_STATE, "hp": 80}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2

    def test_all_flagged_leaves_choice_to_policy(self) -> None:
        # AutoSlay.EventRoomHandler keeps the filtered list only `if
        # (list3.Count > 0)`; with nothing left the pick returns to the policy.
        opts = self._trial_options(True)
        opts[0]["will_kill_player"] = True
        state = {**_envelope("event", {**_RUN_STATE, "hp": 80}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2

    def test_response_indices_stay_world_side(self) -> None:
        # Filtering must not renumber: `responses` carries the ORIGINAL index,
        # which is what the C# side resolves the choice against.
        opts = self._trial_options(True)
        opts.insert(0, {"index": 0, "title": "Leave", "description": "", "enabled": True})
        for i, o in enumerate(opts):
            o["index"] = i
        state = {**_envelope("event", {**_RUN_STATE, "hp": 80}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert [r["index"] for r in decision.responses] == [0, 1]

    # ── the flag actually OUTRANKS the parse, in both directions ────────────

    @staticmethod
    def _abyssal_baths_options(flag: object) -> list[dict[str, Any]]:
        """AbyssalBaths' initial page, the real case where the two disagree.

        `ThatDoesDamage(Damage.IntValue - MaxHp.BaseValue)` flags Immerse at a
        threshold of `3 - 2 = 1`, while the description says "Take 3 damage" —
        so the parse extracts 3 and the world extracts 1. At hp in {2, 3} the
        world says survivable and the regex says lethal.
        """
        return [
            {"index": 0, "title": "Abstain", "description": "Leave.", "enabled": True},
            {
                "index": 1,
                "title": "Immerse",
                "description": "Gain 2 Max HP. Take 3 damage.",
                "enabled": True,
                "will_kill_player": flag,
            },
        ]

    def test_an_authoritative_not_lethal_beats_the_parse(self) -> None:
        """The world said survivable; the regex must not veto it anyway.

        This is the whole "checked in order of authority" claim. An earlier
        version ran the flag filter and the parse filter as independent passes
        in sequence, which inverted it — the blind heuristic got the last word
        and withheld an option the world had just cleared.
        """
        assert rp._event_option_hp_cost(self._abyssal_baths_options(False)[1]) == 3
        state = {
            **_envelope("event", {**_RUN_STATE, "hp": 3}),
            "options": self._abyssal_baths_options(False),
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert len(decision.responses) == 2, "the world's `false` was overridden by the regex"

    def test_without_the_flag_the_parse_still_vetoes(self) -> None:
        """The other half — the fallback must keep working where it is all we have."""
        opts = self._abyssal_baths_options(False)
        del opts[1]["will_kill_player"]
        state = {**_envelope("event", {**_RUN_STATE, "hp": 3}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]

    def test_a_null_flag_from_a_throwing_probe_reads_as_unknown(self) -> None:
        """C# emits `null`, not `false`, when the predicate throws.

        `TabletOfTruth`'s closure does a `DynamicVars[...]` lookup and is the one
        predicate that can realistically throw. Reporting `false` there would tell
        this guard the world said SAFE about the very case it exists to catch, so
        the probe degrades to absent and the parse takes over.
        """
        opts = self._abyssal_baths_options(None)
        state = {**_envelope("event", {**_RUN_STATE, "hp": 3}), "options": opts}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.responses == [{"action": "choose", "index": 0}]

        # And at full HP the same unknown reads as "no reason to withhold".
        safe = {**_envelope("event", {**_RUN_STATE, "hp": 80}), "options": opts}
        safe_decision = rp._build_event(safe)
        assert safe_decision is not None
        assert len(safe_decision.responses) == 2


class TestCardSelectBuilder:
    @pytest.mark.parametrize(
        "prev,kind,obs_key",
        [
            ("rest_site", "smith_card", "smith_options"),
            ("shop", "remove_deck_card", "card_removal_options"),
            ("event", "grant_deck_card", "card_grant_options"),
        ],
    )
    def test_surface_dispatch(self, prev, kind, obs_key) -> None:
        decision = rp._build_card_select(CARD_SELECT_STATE, prev)
        assert decision is not None
        assert [a["action"] for a in decision.actions] == [kind] * 3
        assert [a["idx"] for a in decision.actions] == [0, 1, 2]
        assert len(decision.obs[obs_key]) == 3
        # Bounds present -> the bridge expects the indices-list spelling.
        assert decision.responses[1] == {"action": "choose", "indices": [1]}

    def test_smith_options_are_plain_ids(self) -> None:
        decision = rp._build_card_select(CARD_SELECT_STATE, "rest_site")
        assert decision is not None
        assert decision.obs["smith_options"] == ["STRIKE_IRONCLAD", "BASH", "THUNDERCLAP"]

    def test_removal_options_are_id_upgraded_pairs(self) -> None:
        state = dict(CARD_SELECT_STATE)
        state["cards"] = [_cs_card("STRIKE_IRONCLAD", 0), _cs_card("BASH", 1, upgrade_level=1)]
        decision = rp._build_card_select(state, "shop")
        assert decision is not None
        assert decision.obs["card_removal_options"] == [
            ("STRIKE_IRONCLAD", False),
            ("BASH", True),
        ]

    def test_combat_context_returns_none(self) -> None:
        # Mid-combat selections (Armaments/discard/...) are NOT deck
        # mutations; they stay on the random fallback until routed to MCTS.
        assert rp._build_card_select(CARD_SELECT_STATE, "combat_action") is None

    def test_unknown_context_returns_none(self) -> None:
        assert rp._build_card_select(CARD_SELECT_STATE, None) is None

    def test_multi_pick_is_ranked_and_carries_its_arity(self) -> None:
        """It used to return None, which meant uniform random.

        A "remove 2 cards" screen is still a ranking over single cards -- the
        arity says how many the ANSWER names, not that the surface differs.
        Declining it sent 9 of the 19 unsearched selections in
        `logs/f7-mirror-check18` to `realgame_random.pick_card_select`.
        """
        state = dict(CARD_SELECT_STATE)
        state["max_select"] = state["max_count"] = 2
        state["min_select"] = state["min_count"] = 2
        decision = rp._build_card_select(state, "rest_site")
        assert decision is not None
        assert decision.pick_count == 2
        assert len(decision.actions) == len(CARD_SELECT_STATE["cards"])

    def test_a_single_pick_screen_still_names_one_card(self) -> None:
        """The arity only moves for a screen that actually asks for more."""
        decision = rp._build_card_select(dict(CARD_SELECT_STATE), "rest_site")
        assert decision is not None
        assert decision.pick_count == 1

    def test_no_bounds_uses_bare_index_spelling(self) -> None:
        state = {k: v for k, v in CARD_SELECT_STATE.items()}
        for k in ("min_select", "max_select", "min_count", "max_count"):
            state.pop(k)
        decision = rp._build_card_select(state, "rest_site")
        assert decision is not None
        assert decision.responses[2] == {"action": "choose", "index": 2}

    def test_dispatch_without_session_falls_back(self) -> None:
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(CARD_SELECT_STATE, rng, policies)
        assert extra is None  # normal fallback, not an error
        assert action["action"] in ("choose", "cancel", "skip")

    @pytest.mark.skipif(_TORCH_SKIP_REASON is not None, reason=_TORCH_SKIP_REASON or "")
    @pytest.mark.skipif(_CKPT_SKIP_REASON is not None, reason=_CKPT_SKIP_REASON or "")
    def test_dispatch_with_rest_site_context_uses_net(self) -> None:
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        session = {"prev_msg_type": "rest_site"}
        action, extra = rp.pick_noncombat_action_ppo(
            CARD_SELECT_STATE, rng, policies, session=session
        )
        assert extra is None
        assert action["action"] == "choose"
        assert action["indices"][0] in (0, 1, 2)
        assert "annotations" in action  # PPO answered, not the random fallback


# ---------------------------------------------------------------------------
# Tier 3: fallback behavior + combat re-exports (no checkpoint required).
# ---------------------------------------------------------------------------


class TestFallback:
    def test_uncovered_surface_falls_back_silently(self) -> None:
        """treasure isn't a PPO-covered surface -- normal operation, no error."""
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(TREASURE_STATE, rng, policies)
        assert extra is None
        assert action == {"action": "choose", "index": 0}

    # An unloadable EXPLICIT --model only degrades in the advisor lane
    # (`strict=False`): there a missing net costs the ranking, not the run. The
    # driver lanes raise instead, because degrading there still reports a win
    # rate for a checkpoint that was never loaded -- see
    # tests/test_ppo_onnx_policy.py::TestExplicitModelIsNotOptional.
    def test_missing_checkpoint_falls_back_with_ppo_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.pt"
        policies = rp.PpoPolicyCache(str(bogus), strict=False)
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(CARD_REWARD_STATE, rng, policies)

        assert extra is not None
        assert "ppo_error" in extra
        assert action["action"] in ("choose", "skip")

    def test_corrupt_checkpoint_load_failure_falls_back(self, tmp_path: Path) -> None:
        bogus = tmp_path / "corrupt.pt"
        bogus.write_bytes(b"not a torch checkpoint")
        policies = rp.PpoPolicyCache(str(bogus), strict=False)
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo(CARD_REWARD_STATE, rng, policies)

        assert extra is not None
        assert "ppo_error" in extra
        assert action["action"] in ("choose", "skip")

    def test_policy_cache_warns_once_per_character(self, tmp_path: Path, caplog) -> None:
        bogus = tmp_path / "does_not_exist.pt"
        policies = rp.PpoPolicyCache(str(bogus), strict=False)
        with caplog.at_level("ERROR"):
            policies.get("IRONCLAD")
            policies.get("ironclad")  # same normalized key -- must not warn again
            policies.get("SILENT")  # different key -- warns again
        warnings = [r for r in caplog.records if "PPO policy unavailable" in r.message]
        assert len(warnings) == 2

    def test_unknown_message_type_falls_back(self) -> None:
        """reward_screen isn't PPO-covered; with no options offered,
        realgame_random's shared option-choice handler degrades to skip."""
        policies = rp.PpoPolicyCache(None)
        rng = random.Random(0)
        action, extra = rp.pick_noncombat_action_ppo({"type": "reward_screen"}, rng, policies)
        assert extra is None
        assert action == {"action": "skip"}


class TestSelectIndex:
    def test_temperature_zero_is_argmax(self) -> None:
        rng = random.Random(0)
        assert rp._select_index([0.1, 0.7, 0.2], 0.0, rng) == 1

    def test_temperature_positive_favors_higher_prob(self) -> None:
        rng = random.Random(0)
        counts = [0, 0]
        for _ in range(500):
            counts[rp._select_index([0.9, 0.1], 1.0, rng)] += 1
        assert counts[0] > counts[1]

    def test_empty_probs_raises(self) -> None:
        with pytest.raises(ValueError):
            rp._select_index([], 0.0, random.Random(0))

    def test_greedy_with_actions_uses_the_group_decode(self) -> None:
        """The live advisor must decode like the workbench and run_arena.

        Passing the candidates in switches greedy selection to the A8 §8
        group-mass rule (``ppo_decode``): a shop's per-deck-card removal
        candidates pool, so a split "remove a starter" preference stops
        losing to a single buy with less total mass. Without them the plain
        per-candidate argmax is unchanged.
        """
        actions = [
            {"action": "shop_buy_card", "idx": 4},
            {"action": "shop_remove_card", "deck_idx": 0},
            {"action": "shop_remove_card", "deck_idx": 1},
            {"action": "shop_remove_card", "deck_idx": 2},
        ]
        probs = [0.31, 0.24, 0.23, 0.22]
        rng = random.Random(0)
        assert rp._select_index(probs, 0.0, rng) == 0
        assert rp._select_index(probs, 0.0, rng, actions) == 1

    def test_sampling_ignores_the_group_decode(self) -> None:
        # `--temperature > 0` is an explicit "don't be greedy" mode; the
        # group decode is a greedy-argmax rule and must not silently
        # collapse sampling back to one action.
        actions = [{"action": "shop_remove_card", "deck_idx": i} for i in range(2)]
        rng = random.Random(0)
        counts = [0, 0]
        for _ in range(500):
            counts[rp._select_index([0.9, 0.1], 1.0, rng, actions)] += 1
        assert counts[0] > 0 and counts[1] > 0


class TestBuildPpoAnnotations:
    def test_shape(self) -> None:
        decision = rp.PpoDecision(
            obs={}, actions=[{}, {}], metrics={}, responses=[{}, {}], labels=["a", "b"]
        )
        ann = rp.build_ppo_annotations(decision, [0.3, 0.7], 1, 0.5)
        assert ann["note"].startswith("PPO p=0.70")
        assert "-> b" in ann["note"]
        assert ann["cards"] == [
            {"index": 0, "score": 0.3, "label": "30%"},
            {"index": 1, "score": 0.7, "label": "70%"},
        ]


class TestCombatReexports:
    """The combat path never runs through PPO (see module docstring) --
    it reuses realgame_mcts / realgame_random verbatim. These just confirm
    realgame_ppo wires them up correctly, against the real captured
    combat_action fixtures also used by test_realgame_adapter.py."""

    def test_random_combat_returns_legal_action(self) -> None:
        state = _load_combat_fixture("single_enemy_ironclad.json")
        rng = random.Random(0)
        action = pick_combat_action(state, rng)
        assert action["action"] in ("play", "end_turn", "potion")

    @pytest.mark.skipif(_RS_SKIP_REASON is not None, reason=_RS_SKIP_REASON or "no sts2_rs")
    def test_mcts_combat_returns_legal_action(self) -> None:
        from sts2_hint.recommender import RecommendConfig

        state = _load_combat_fixture("single_enemy_ironclad.json")
        rng = random.Random(0)
        cfg = RecommendConfig(sims=20, depth=10, seed=0)
        action, _extra, _engine = pick_combat_action_mcts(state, cfg, rng)
        assert action["action"] in ("play", "end_turn", "potion")


class TestWireLegalityMask:
    """The bridge hand carries the authoritative CardModel.CanPlay() as
    hand[].playable; the engine model can miss live-only blocks (Act-3 boss
    hook made DEFEND unplayable, 2026-07-13 -- the search recommended the
    rejected card forever and the watchdog killed a healthy floor-48 run).
    pick_combat_action_mcts must skip wire-unplayable picks down the
    ranking, ending the turn when everything is masked."""

    @staticmethod
    def _patched(monkeypatch, hand, tag_actions):
        import realgame_mcts as rm

        monkeypatch.setattr(rm, "to_headless_combat_play", lambda s: {})
        monkeypatch.setattr(rm, "recommend_all", lambda h, c: list(tag_actions))
        monkeypatch.setattr(rm, "hint_tag_to_action", lambda t: dict(tag_actions[t]))
        monkeypatch.setattr(rm, "build_annotations", lambda tags, cfg, state=None: None)
        return rm, {"hand": hand, "run_state": {"character": "IRONCLAD"}, "round": 1}

    def test_unplayable_top_pick_falls_to_next_best(self, monkeypatch) -> None:
        from sts2_hint.recommender import RecommendConfig

        rm, state = self._patched(
            monkeypatch,
            hand=[
                {"id": "DEFEND_IRONCLAD", "playable": False},
                {"id": "STRIKE_IRONCLAD", "playable": True},
            ],
            tag_actions={
                0: {"action": "play", "card_index": 0},
                1: {"action": "play", "card_index": 1},
            },
        )
        action, extra, _engine = rm.pick_combat_action_mcts(
            state, RecommendConfig(), random.Random(0)
        )
        assert extra is None
        assert action["card_index"] == 1

    def test_all_plays_masked_ends_turn(self, monkeypatch) -> None:
        from sts2_hint.recommender import RecommendConfig

        rm, state = self._patched(
            monkeypatch,
            hand=[{"id": "DEFEND_IRONCLAD", "playable": False}],
            tag_actions={0: {"action": "play", "card_index": 0}},
        )
        action, extra, _engine = rm.pick_combat_action_mcts(
            state, RecommendConfig(), random.Random(0)
        )
        assert extra is None
        assert action["action"] == "end_turn"

    def test_absent_playable_flag_is_trusted(self, monkeypatch) -> None:
        from sts2_hint.recommender import RecommendConfig

        rm, state = self._patched(
            monkeypatch,
            hand=[{"id": "DEFEND_IRONCLAD"}],  # older mod build: no flag
            tag_actions={0: {"action": "play", "card_index": 0}},
        )
        action, extra, _engine = rm.pick_combat_action_mcts(
            state, RecommendConfig(), random.Random(0)
        )
        assert extra is None
        assert action["card_index"] == 0
        # extra is None on a clean MCTS pick, or a dict carrying mcts_error
        # on fallback -- both are legal outcomes for this smoke test.
        assert extra is None or "mcts_error" in extra

    def test_run_requires_mcts_cfg_when_combat_policy_is_mcts(self) -> None:
        with pytest.raises(ValueError, match="mcts_cfg"):
            rp.run(
                "127.0.0.1",
                1,
                Path("/tmp/does-not-matter"),
                None,
                random.Random(0),
                rp.PpoPolicyCache(None),
                combat_policy="mcts",
                mcts_cfg=None,
            )


class TestNeowEventLabels:
    """Neow options must carry the native enum-form labels (the offered
    blessing's SCREAMING_SNAKE text-key tail) so the PPO encoder registry-
    matches identity like it did in training; every other event keeps prose."""

    @staticmethod
    def _neow_option(index: int, label: str, tail: str, relic_id: str = "") -> dict[str, Any]:
        return {
            "index": index,
            "id": "event_choice",
            "action": "event_choice",
            "label": label,
            "description": "Gain a relic.",
            "event_id": "NEOW",
            "text_key": f"NEOW.pages.INITIAL.options.{tail}" if tail else "",
            "relic_id": relic_id,
            "enabled": True,
        }

    def test_neow_labels_use_text_key_tail(self) -> None:
        state = {
            **_envelope("event"),
            "options": [
                self._neow_option(0, "Winged Boots", "WINGED_BOOTS", "WINGED_BOOTS"),
                self._neow_option(1, "Small Capsule", "SMALL_CAPSULE"),
                self._neow_option(2, "Cursed Pearl", "CURSED_PEARL", "CURSED_PEARL"),
            ],
        }
        decision = rp._build_event(state)
        assert decision is not None
        labels = [o["label"] for o in decision.obs["event_options"]]
        assert labels == ["WINGED_BOOTS", "SMALL_CAPSULE", "CURSED_PEARL"]
        assert decision.labels == labels

    def test_neow_relic_id_fallback_without_text_key(self) -> None:
        state = {
            **_envelope("event"),
            "options": [self._neow_option(0, "Winged Boots", "", "WINGED_BOOTS")],
        }
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["label"] == "WINGED_BOOTS"

    def test_non_neow_event_keeps_prose_labels(self) -> None:
        # Native labels every non-Neow event with prose; an enum-form
        # text_key tail must NOT leak into the label there.
        option = self._neow_option(0, "Pray at the altar", "PRAY", "")
        option["event_id"] = "BONFIRE_ELEMENTALS"
        option["text_key"] = "BONFIRE_ELEMENTALS.pages.INITIAL.options.PRAY"
        state = {**_envelope("event"), "options": [option]}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["label"] == "Pray at the altar"

    def test_neow_prose_kept_when_no_enum_fields(self) -> None:
        # Older mod build: no text_key/relic_id fields at all.
        option = self._neow_option(0, "Winged Boots", "", "")
        del option["text_key"], option["relic_id"]
        state = {**_envelope("event"), "options": [option]}
        decision = rp._build_event(state)
        assert decision is not None
        assert decision.obs["event_options"][0]["label"] == "Winged Boots"


class TestEventNameVocabResolution:
    """`obs["event_name"]` must land in `ppo.vocab.events()`, whatever dialect.

    The vocab is keyed on `EventId::python_name()`; the headless dialect's
    `event_id` is the English DISPLAY name ("Tablet of Truth"), which
    canonicalizes with spaces and never matches — before `_resolve_event_name`,
    72% of DLL event decisions fed the OOV slot and the net was event-blind
    (it walked Tablet of Truth's Decipher escalation to max_hp=1 in 4/60
    v34wr seeds, all defeats). The authoritative id is the first segment of
    any option `text_key`; SpireMentor's SCREAMING_SNAKE `event_id` resolves
    the same way.
    """

    def _decision(self, options, event_name=""):
        state = {**_envelope("event"), "event_name": event_name, "options": options}
        for o in state["options"]:
            o.setdefault("enabled", True)
        return rp._build_event(state)

    def _resolved(self, options, display=""):
        return rp._resolve_event_name(options, display)

    def test_text_key_prefix_wins(self) -> None:
        got = self._resolved(
            [{"text_key": "TABLET_OF_TRUTH.pages.INITIAL.options.0"}], "Tablet of Truth"
        )
        assert got == "TABLET_OF_TRUTH"

    def test_display_name_fallback_resolves_spaces(self) -> None:
        assert self._resolved([], "This or That?") == "THIS_OR_THAT"

    def test_leading_article_is_stripped_to_match_registry(self) -> None:
        # Registry names: Trial / RoundTeaParty / SunkenStatue (no article).
        assert self._resolved([], "The Trial") == "TRIAL"
        assert self._resolved([], "The Round Tea Party") == "ROUND_TEA_PARTY"
        assert self._resolved([], "The Sunken Statue") == "SUNKEN_STATUE"

    def test_aliases(self) -> None:
        # DollRoom's C# loc entry; TheArchitect's continuation page verb.
        assert self._resolved([{"text_key": "MR_STRUGGLES.pages.x"}], "Doll Room") == "DOLL_ROOM"
        assert self._resolved([{"text_key": "PROCEED.options.0"}], "") == "THE_ARCHITECT"

    def test_unknown_event_stays_oov_not_crash(self) -> None:
        got = self._resolved([], "Totally Unknown Event")
        assert got == "TOTALLY_UNKNOWN_EVENT"  # passthrough; encoder OOVs it

    def test_every_registry_event_resolves_to_itself(self) -> None:
        # The vocab universe round-trips: no alias/strip rule may hijack a
        # genuine registry name onto a different slot.
        vocab = pytest.importorskip("ppo.vocab")
        reg = vocab.events()
        for name in reg.names:
            canon = vocab.canonical(name)
            got = self._resolved([{"text_key": f"{canon}.pages.INITIAL.options.0"}], name)
            assert reg.index(got) == reg.index(name), (name, got)

    def test_build_event_obs_uses_resolved_name(self) -> None:
        decision = self._decision(
            [
                {
                    "index": 0,
                    "label": "Decipher",
                    "description": "Lose {DecipherMaxHpLoss} Max HP.",
                    "text_key": "TABLET_OF_TRUTH.pages.INITIAL.options.0",
                    "event_id": "Tablet of Truth",
                    "vars": {"DecipherMaxHpLoss": 24},
                }
            ]
        )
        assert decision is not None
        assert decision.obs["event_name"] == "TABLET_OF_TRUTH"
        # And the vars actually resolve into the description (Defect B).
        assert decision.obs["event_options"][0]["description"] == "Lose 24 Max HP."
