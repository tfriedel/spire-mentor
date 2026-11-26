"""Advice names the board's enemy slots after the game reaps a corpse.

A live autobattle fight (2026-11-23, Ironclad, floor 36, Living Shield +
Turret Operator): Prowess, a Power, set off Lost Wisp, which killed the
5-HP Living Shield. The game took the corpse out of `Enemies`, so the board
listed the Turret alone, at index 0. The advisor's mirror carries the fight
and keeps the corpse in its slot, and Instinct's `target_index` came from
that engine fight verbatim: Pommel Strike at index 1, a slot the board does
not have. The mod refused it three times and ended the turn, then did the
same on every turn left in the fight, 5 energy and a full hand unplayed.

The fixture is that fight's recorded traffic, from its first board to the
first stale answer (a473..a483).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sts2_rs", reason="the advisor needs the sts2_rs extension")

import realgame_advisor_replay as rr
import realgame_instinct as ri

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "realgame_instinct"
    / "reaped_corpse_living_shield.jsonl.gz"
)
FAST = {"sims": 64, "boss_sims": 128, "threads": 1}


def _requests() -> dict[str, dict[str, Any]]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        msgs = [json.loads(line) for line in fh if line.strip()]
    return {m["request_id"]: m for m in msgs if m.get("type") == "advice_request"}


def _off_board(replies: list[dict[str, Any]]) -> list[str]:
    """Every advised move (best and turn plan) aimed at an enemy slot its
    board does not have."""
    requests = _requests()
    bad = []
    for r in replies:
        if r.get("action") != "advice":
            continue
        enemies = requests[r["request_id"]].get("enemies") or []
        advice = r["advice"]
        for move in [advice.get("best") or {}, *(advice.get("plan") or [])]:
            t = move.get("target_index")
            if t is not None and not 0 <= t < len(enemies):
                bad.append(f"{r['request_id']}: {move.get('label')} -> {t} of {len(enemies)}")
    return bad


@pytest.mark.skipif(
    not ri.default_weights_path().is_file(), reason="no Instinct weights (data/e2e)"
)
def test_instinct_aims_at_the_turret_on_the_reaped_board() -> None:
    replies = rr.replay(FIXTURE, brain="instinct", **FAST)
    assert _off_board(replies) == []
    last = next(r for r in replies if r.get("request_id") == "a483" and r.get("action") == "advice")
    best = last["advice"]["best"]
    if best.get("target_index") is not None:
        assert best["target_index"] == 0
        assert "Turret Operator" in best["label"]


def test_the_search_aims_inside_the_reaped_board() -> None:
    assert _off_board(rr.replay(FIXTURE, brain="search", **FAST)) == []
