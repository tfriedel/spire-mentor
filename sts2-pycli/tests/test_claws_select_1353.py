"""GH #1353: Claws' "up to 6 into Maul" select is a player choice.

C# `Claws.AfterObtained` opens a min-0/max-6 deck select with a manual confirm
(`Claws.cs:22-29`). The engine surfaces it in agent mode as one
`transform_deck_card` per card still offered plus `confirm_deck_selection`.
These pin each lane's handling of it.
"""

from __future__ import annotations

from typing import Any

import pytest
import sts2_rs
from ppo import encoder, vocab
from ppo.env import assert_encodable, classify_surface
from workbench.backends.rust_game import RustGame

EIGHT = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"] * 4


def _claws_rm(deck: list[str] = EIGHT) -> Any:
    rm = sts2_rs.RunManager(seed=7, character="Ironclad", mode="agent")
    rm.debug_set_master_deck(deck)
    assert rm.force_add_relic("CLAWS")
    return rm


def _ids(rm: Any) -> list[str]:
    """Master-deck ids, spelling-normalized (``StrikeIronclad`` -> ``STRIKEIRONCLAD``)."""
    return [c.replace("_", "").upper() for c, _ in rm.master_deck_detailed()]


def _mauls(rm: Any) -> int:
    return _ids(rm).count("MAUL")


# ── engine, through the bindings ────────────────────────────────────────────


def test_the_select_offers_each_card_and_a_trailing_confirm() -> None:
    rm = _claws_rm()
    assert rm.pending_claws_select
    assert (rm.claws_select_max, rm.claws_select_picked) == (6, 0)
    acts = rm.available_actions()
    assert acts[-1] == {"action": "confirm_deck_selection"}
    assert [a["action"] for a in acts[:-1]] == ["transform_deck_card"] * 8


def test_picks_change_nothing_until_the_confirm() -> None:
    rm = _claws_rm()
    before = _ids(rm)
    for _ in range(2):
        assert rm.take_action({"action": "transform_deck_card", "idx": 0}) == "advanced"
    assert _ids(rm) == before
    assert rm.claws_select_picked == 2
    assert rm.take_action({"action": "confirm_deck_selection"}) == "advanced"
    assert not rm.pending_claws_select
    assert _mauls(rm) == 2


# ── PPO encoder ─────────────────────────────────────────────────────────────


def _obs() -> dict[str, Any]:
    return {
        "phase": "card_reward",
        "surface": "card-pick",
        "hp": 70,
        "max_hp": 80,
        "gold": 99,
        "deck": [("STRIKE_IRONCLAD", False), ("DEFEND_IRONCLAD", False)],
        "relics": ["CLAWS"],
        "potions": [],
        "map": None,
        "map_coord": None,
        "transform_select_deck_indices": [0, 1],
    }


def test_the_confirm_encodes_as_the_skip_token_on_the_transform_surface() -> None:
    actions = [
        {"action": "transform_deck_card", "idx": 0},
        {"action": "transform_deck_card", "idx": 1},
        {"action": "confirm_deck_selection"},
    ]
    assert_encodable(actions, phase="card_reward")
    assert classify_surface("map_choice", actions) == "card-pick"
    assert encoder.fine_surface(actions) == "card-transform"
    assert encoder.fine_surface(actions[-1:]) == "card-transform"
    features = encoder.featurize_row(_obs(), actions, _obs())
    assert features.act_cat[-1, 6] == vocab.fixed_index("skip_card")


# ── workbench backend ───────────────────────────────────────────────────────


def _game() -> RustGame:
    g = RustGame()
    g.start(character="Ironclad", seed="7", ascension=0)
    g.rm.debug_set_master_deck(EIGHT)
    assert g.rm.force_add_relic("CLAWS")
    return g


def test_the_backend_renders_an_up_to_six_card_select() -> None:
    view = _game()._serialize()
    assert view["decision"] == "card_select"
    assert (view["min_select"], view["max_select"]) == (0, 6)
    assert len(view["cards"]) == 8


def test_select_cards_transforms_the_named_cards_at_once() -> None:
    g = _game()
    view = g.act("select_cards", indices="1,3,4")
    assert not view.get("error"), view
    assert not g.rm.pending_claws_select
    # Picked: offers 1, 3, 4 (Defend, Defend, Strike). The five unpicked keep
    # their order; the Mauls join at the tail.
    s, d = "STRIKEIRONCLAD", "DEFENDIRONCLAD"
    assert _ids(g.rm) == [s, s, d, s, d] + ["MAUL"] * 3


def test_skip_select_confirms_with_no_card() -> None:
    g = _game()
    before = _ids(g.rm)
    view = g.act("skip_select")
    assert not view.get("error"), view
    assert not g.rm.pending_claws_select
    assert _ids(g.rm) == before


def test_select_cards_refuses_more_than_six() -> None:
    g = _game()
    view = g.act("select_cards", indices="0,1,2,3,4,5,6")
    assert view.get("type") == "error", view
    assert g.rm.pending_claws_select


# ── e2e env: resolved outside the policy ────────────────────────────────────


def test_the_e2e_env_takes_the_first_six_without_a_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sts2_client
    from ppo.e2e import factory
    from ppo.e2e.env import _RunEnv

    monkeypatch.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
    rm = factory.new_run(7)
    rm.debug_set_master_deck(EIGHT)
    assert rm.force_add_relic("CLAWS")
    env = _RunEnv.wrap(rm, 7, max_steps=1000)
    env._advance("net")
    assert not rm.pending_claws_select
    assert _ids(rm) == ["STRIKEIRONCLAD", "DEFENDIRONCLAD"] + ["MAUL"] * 6
    assert env.forced["CARD_TRANSFORM"] == 7


# ── real-game lanes ─────────────────────────────────────────────────────────


def _claws_screen(n: int, *, selected: int = 0) -> dict[str, Any]:
    return {
        "type": "card_select",
        "prompt_key": "CLAWS.selectionScreenPrompt",
        "min_select": 0,
        "max_select": 6,
        "selected_count": selected,
        "cards": [{"index": i, "id": "STRIKE_IRONCLAD", "name": f"Strike {i}"} for i in range(n)],
        "run_state": {
            "character": "Ironclad",
            "master_deck": [{"id": "STRIKE_IRONCLAD"} for _ in range(n)],
        },
    }


def test_the_ppo_card_select_offers_the_cards_and_a_confirm() -> None:
    import realgame_ppo

    d = realgame_ppo._build_card_select(_claws_screen(3), None)
    assert d is not None
    assert [a["action"] for a in d.actions] == ["transform_deck_card"] * 3 + [
        "confirm_deck_selection"
    ]
    assert d.responses[-1] == {"action": "skip"}
    assert d.optional_room == 6


def test_a_full_claws_grid_offers_only_the_confirm() -> None:
    import realgame_ppo

    d = realgame_ppo._build_card_select(_claws_screen(3, selected=6), None)
    assert d is not None
    assert d.actions == [{"action": "confirm_deck_selection"}]


class _ScriptedPolicy:
    """Answers each ask with a fixed choice and records what it was offered."""

    def __init__(self, picks: list[int]) -> None:
        self.picks = picks
        self.asked: list[tuple[list[int], int]] = []

    def distribution(self, obs: dict[str, Any], actions: list[dict[str, Any]], metrics: Any):
        self.asked.append((list(obs["transform_select_deck_indices"]), len(actions)))
        probs = [0.0] * len(actions)
        probs[self.picks[len(self.asked) - 1]] = 1.0
        return probs, 0.5


def test_the_plan_asks_again_after_each_card_until_the_confirm() -> None:
    import realgame_ppo

    d = realgame_ppo._build_card_select(_claws_screen(4), None)
    assert d is not None
    # Take offer 2, then offer 0, then confirm (the last of three options).
    policy = _ScriptedPolicy([2, 0, 2])
    cards, probs, _value, chosen = realgame_ppo.plan_optional_picks(
        policy, d, lambda p, _a: max(range(len(p)), key=p.__getitem__)
    )
    assert cards == [2, 0]
    assert chosen == 2
    assert len(probs) == 5
    assert [n for _, n in policy.asked] == [5, 4, 3]
    # Each ask's offer drops the cards already taken.
    assert [len(o) for o, _ in policy.asked] == [4, 3, 2]


def test_the_plan_stops_at_the_cap() -> None:
    import realgame_ppo

    screen = _claws_screen(8)
    screen["max_select"] = 2
    d = realgame_ppo._build_card_select(screen, None)
    assert d is not None
    policy = _ScriptedPolicy([0, 0])
    cards, *_ = realgame_ppo.plan_optional_picks(policy, d, lambda p, _a: p.index(max(p)))
    # The third ask has only the confirm, so the net is not consulted.
    assert cards == [0, 1]
    assert len(policy.asked) == 2


@pytest.mark.parametrize(("selected", "best"), [(0, "choose"), (6, "skip")])
def test_instinct_plays_claws_as_its_env_does(selected: int, best: str) -> None:
    import realgame_instinct_screens

    advice = realgame_instinct_screens.deck_select(None, _claws_screen(8, selected=selected), None)
    assert advice["best"]["action"] == best
    if best == "choose":
        assert advice["best"]["index"] == 0
