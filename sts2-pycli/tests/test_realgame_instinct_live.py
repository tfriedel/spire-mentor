"""Instinct over a recorded LIVE run (``realgame_advisor --record-requests``).

The corpus is the first live A10 Ironclad run played with Instinct as the brain
(2026-11-07, seed F7UYBVY0D8WJ, won): every raw message the mod sent, in order.
Replaying it through :class:`realgame_instinct.InstinctAdvisor` pins that the
run graft and every screen handler read REAL payloads -- the shapes the
synthetic fidelity tests (``test_realgame_instinct_screens.py``) can only
imitate: the mod's id spellings, locked event buttons, multi-page events, lone
buttons, an act's entrance.

Declines are allowed only for the reasons the advisor is designed to hand to
the search: in-combat choices need the live fight mirror (not replayed here),
enchant screens do not name their enchantment, and the chest relic is not a
choice in Instinct's game.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_instinct as ri
from realgame_instinct_replay import replay

CORPUS = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "2026-11-07-instinct-live"
    / "run1-requests.jsonl.gz"
)

#: The reasons a decline may carry on this corpus (substrings).
ALLOWED = (
    "no fight mirror",
    "does not say which enchantment",
    "chest relic",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not ri.default_weights_path().is_file(), reason="no Instinct weights (data/e2e)"
)
def test_instinct_reads_every_position_of_a_live_run() -> None:
    r = replay(CORPUS)
    assert r["requests"] > 800
    assert not r["errors"], r["errors"]
    unexpected = {
        (surface, why): n
        for surface, reasons in r["declined"].items()
        for why, n in reasons.items()
        if not any(a in why for a in ALLOWED)
    }
    assert not unexpected, unexpected
    for surface in ("combat_action", "map_select", "event", "shop", "rest_site", "card_reward"):
        assert r["answered"][surface] > 0, surface
