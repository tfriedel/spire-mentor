"""Instinct in co-op: the act map and Neow, on live co-op payloads (2026-11-08).

The engine builds single-player runs. A co-op act map is one room shorter
(``ActModel.GetNumberOfRooms(isMultiplayer)``, same map RNG), and each
player's Neow is seeded with their slot, with Massive Scroll in the pool. So
Instinct declined every co-op map choice and every co-op Neow. The run
rebuild now builds co-op sized maps (``RunManager.debug_use_coop_maps``) and
stands on the screen's own Neow offers (``debug_set_neow_offers``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_instinct as ri  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ri.default_weights_path().is_file(), reason="no Instinct weights (data/e2e)"
)

FIXTURES = Path(__file__).parent / "fixtures" / "realgame_instinct"


def _payload(name: str) -> dict[str, Any]:
    msg = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return {**msg, "type": msg["advice_for"]}


@pytest.mark.parametrize(
    "name",
    [
        "coop_map_choice.json",
        # The Spoils Map quest's hourglass act map, while the card is in the
        # deck and after the quest took it out.
        "coop_spoils_map_quest.json",
        "coop_spoils_map_done.json",
    ],
)
def test_instinct_routes_a_coop_map(name: str) -> None:
    state = _payload(name)
    _rm, notes = ri.graft_run(state)
    assert ri.COOP_MAP_APPROX not in notes, notes
    assert not [n for n in notes if "act map this engine generates differs" in n], notes

    advice = ri.InstinctAdvisor().advise(state, "map_select", session={})

    assert advice["brain"] == "instinct"
    offered = {(n["col"], n["row"]) for n in state["nodes"]}
    best = next(o for o in advice["options"] if o.get("is_best"))
    assert (best["col"], best["row"]) in offered
    path = advice.get("path") or []
    assert len(path) >= 2
    assert (path[0]["col"], path[0]["row"]) == (best["col"], best["row"])


def test_instinct_answers_a_coop_neow_with_massive_scroll() -> None:
    state = _payload("coop_neow_massive_scroll.json")
    offered = [o["relic_id"] for o in state["options"]]
    assert "MASSIVE_SCROLL" in offered  # co-op only: never in a single-player roll

    advice = ri.InstinctAdvisor().advise(state, "event", session={})

    assert advice["brain"] == "instinct"
    assert advice["best"]["index"] in {o["index"] for o in state["options"]}


def test_the_engine_takes_neow_offers_from_the_screen() -> None:
    rm = sts2_rs.RunManager(
        game_version="v0.111.0", string_seed="KZUFM0ANBN5Q", neow_start=True, mode="agent"
    )
    rm.debug_set_neow_offers("MASSIVE_SCROLL", "RELIC.NEW_LEAF", "PRECARIOUS_SHEARS")
    assert rm.neow_offers() == ("MASSIVE_SCROLL", "NEW_LEAF", "PRECARIOUS_SHEARS")
    with pytest.raises(ValueError, match="not a Neow relic"):
        rm.debug_set_neow_offers("BURNING_BLOOD", "NEW_LEAF", "PRECARIOUS_SHEARS")
