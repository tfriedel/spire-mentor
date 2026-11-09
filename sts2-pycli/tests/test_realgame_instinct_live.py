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


#: Enough of the corpus for a few fights and maps (the run's first act).
PLAN_REQUESTS = 160


@pytest.mark.skipif(
    not ri.default_weights_path().is_file(), reason="no Instinct weights (data/e2e)"
)
def test_instinct_plans_the_turn_and_draws_the_route_on_live_payloads() -> None:
    """``advice.plan`` / ``advice.path`` / ``options[].action`` (PROTOCOL.md
    v5.3) from Instinct on the live run's real payloads: the plan opens with
    ``best``, indexes the request's own hand without repeating a card, and
    only ends the turn last; the route starts at the chosen node."""
    import gzip
    import json

    from display_names import sanitize_advice
    from sts2_hint.recommender import RecommendConfig

    advisor = ri.InstinctAdvisor()
    cfg = RecommendConfig()
    plans = routes = lone_routes = 0
    longer = 0
    n = 0
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            msg = json.loads(line)
            if msg.get("type") == "human_action":
                advisor.on_human_action(msg)
            if msg.get("type") != "advice_request":
                continue
            n += 1
            if n > PLAN_REQUESTS:
                break
            advice_for = msg.get("advice_for") or "combat_action"
            state = {**msg, "type": advice_for}
            try:
                advice = advisor.advise(state, advice_for, cfg=cfg, session={})
            except ri.InstinctUnavailable:
                continue
            advice = sanitize_advice(advice, advice_for, msg)
            if advice_for == "combat_action":
                assert all(
                    o.get("action") in ("play", "potion", "end_turn") for o in advice["options"]
                )
                plan = advice.get("plan")
                assert plan, "every executable combat answer has a plan"
                best = advice["best"]
                assert {k: plan[0].get(k) for k in ("action", "card_index", "target_index")} == {
                    k: best.get(k) for k in ("action", "card_index", "target_index")
                }
                played = [s["card_index"] for s in plan if s["action"] == "play"]
                assert len(played) == len(set(played)), plan
                assert all(0 <= i < len(msg["hand"]) for i in played)
                assert all(s["action"] != "end_turn" for s in plan[:-1])
                assert all(s["label"] for s in plan)
                plans += 1
                longer += len(plan) > 1
            elif advice_for == "map_select":
                # Corridors too (one node on offer): 60% of a live run's map
                # screens, and they drew no route at all (live, 2026-11-08).
                best = advice["best"]
                chosen = next(o for o in advice["options"] if o["index"] == best["index"])
                path = advice["path"]
                assert (path[0]["col"], path[0]["row"]) == (chosen["col"], chosen["row"])
                assert [s["hop"] for s in path] == list(range(len(path)))
                routes += 1
                # The mod draws a line through 2+ nodes.
                lone_routes += len(msg.get("nodes") or []) == 1 and len(path) >= 2
    assert plans > 20 and longer > 5
    assert routes > 0 and lone_routes > 0
