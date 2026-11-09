"""Instinct takes single player's chest relic itself (live beta.6, 2026-11-08).

The chest offers one relic in single player: there is nothing to choose, and
Instinct used to decline it ("taken, not chosen, in Instinct's game"), so
every chest of an Instinct run was answered by the search.

Its answer is a screen reply, so its action is "choose": the mod clicks a
screen only for "choose" / "skip" (SimAdvisor.ExecuteBestOnce). The first
version passed the option's own "collect" through, which the mod took for a
fight move, and AI play sat at every chest (0.10.0, report
wb-25dd89373d1ed925).
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_instinct as ri

pytestmark = pytest.mark.skipif(
    not ri.default_weights_path().is_file(), reason="no Instinct weights (data/e2e)"
)


def _chest(*relics: str) -> dict[str, Any]:
    return {
        "type": "treasure",
        "advice_for": "treasure",
        "surface": "treasure",
        "options": [
            {"index": i, "id": r, "action": "collect", "label": r.title(), "enabled": True}
            for i, r in enumerate(relics)
        ],
        "game_version": "v0.111.0",
        "player_count": 1,
        "run_state": {"character": "Ironclad"},
    }


def test_the_one_chest_relic_is_taken_by_instinct() -> None:
    advice = ri.InstinctAdvisor().advise(_chest("REGAL_PILLOW"), "treasure", session={})
    assert advice["brain"] == "instinct"
    assert advice["best"] == {"action": "choose", "index": 0, "label": "Regal_Pillow"}
    assert [o["action"] for o in advice["options"]] == ["choose"]


def test_a_shared_chest_is_still_the_searchs() -> None:
    state = {**_chest("REGAL_PILLOW", "ANCHOR"), "player_count": 2}
    with pytest.raises(ri.InstinctUnavailable):
        ri.InstinctAdvisor().advise(state, "treasure", session={})
