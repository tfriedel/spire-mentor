"""A human in the workbench is shown the event options that would kill them, as
the game shows them; the AI acting on the run still skips them.

GH #1907 made agent mode withhold every option the game flags
``WillKillPlayer`` while another survives, as every live chooser does. The
workbench builds its runs in agent mode, so its human player lost those options
too. ``offer_lethal_event_options`` gives them back to the human only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sts2_rs")

from ppo.env import mask_rejectable
from workbench.backends.rust_game import RustGame
from workbench.branching import BranchingLiveSession


def _event_idxs(actions: list[dict]) -> list[int]:
    return [a["idx"] for a in actions if a.get("action") == "event_choice"]


def _cheese_at_10_hp(rm) -> None:
    """Room Full of Cheese at 10 HP: Search's 14 damage would kill."""
    assert rm.force_event("RoomFullOfCheese") is not None
    rm.set_player(hp=10)


def test_a_human_run_offers_search_and_the_policy_mask_drops_it() -> None:
    game = RustGame(human_play=True)
    game.start(character="Ironclad", seed="LETHAL", ascension=0)
    rm = game.rm
    assert rm.offer_lethal_event_options
    assert rm.mode == "agent"
    _cheese_at_10_hp(rm)
    actions = rm.available_actions()
    assert _event_idxs(actions) == [0, 1]
    assert rm.agent_withheld_event_options() == [1]
    assert _event_idxs(mask_rejectable(rm, "event", actions)) == [0]


def test_a_harness_run_withholds_it() -> None:
    game = RustGame()
    game.start(character="Ironclad", seed="LETHAL", ascension=0)
    rm = game.rm
    assert not rm.offer_lethal_event_options
    _cheese_at_10_hp(rm)
    assert _event_idxs(rm.available_actions()) == [0]
    assert rm.agent_withheld_event_options() == []


def test_the_workbench_session_builds_human_runs(tmp_path) -> None:
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="rust")
    sess.start("Ironclad", "LETHAL", 0)
    assert sess.game.rm.offer_lethal_event_options  # type: ignore[attr-defined]
