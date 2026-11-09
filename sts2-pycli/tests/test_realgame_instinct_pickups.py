"""The run rebuild drops pickup choices; enchant screens get Instinct's answer.

GH #1183: ``graft_run`` adds the owned relics with their pickup hooks, and a
hook that queues a choice (Lead Paperweight's choose-a-card screen) left that
choice pending in the rebuilt run. The first time the run advanced, the stale
screen surfaced in place of the real one (Tablet of Truth's second page).

GH #1184: enchant screens name their enchantment (``source_model``); Instinct
queues the same enchant and answers with the ``CARD_ENCHANT`` decision.
"""

from __future__ import annotations

from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs", reason="Instinct needs the sts2_rs extension")

import realgame_instinct as ri  # noqa: E402
from test_realgame_instinct import (  # noqa: E402
    WEIGHTS,
    _campaign_run,
    _envelope,
    _play_to,
    _wire_id,
)

SEED = "XQZS12FZEVYP"


def _map_position() -> tuple[Any, dict[str, Any]]:
    """A campaign run on a map choice past its first fight, and its envelope.

    Past the act's entrance, so the rebuild stands on the same map node.
    """
    student = sts2_rs.E2eStudent(str(WEIGHTS))
    rm = _campaign_run(SEED)
    driver = sts2_rs.E2eDriver(rm.seed)
    _play_to(
        student,
        rm,
        driver,
        lambda p: p.decision["decision_type"] == "MAP" and rm.total_floor() >= 2,
    )
    return rm, _envelope(rm, SEED, "map")


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="no Instinct weights (data/e2e)")
def test_the_rebuild_leaves_no_pickup_choice_queued() -> None:
    _base, msg = _map_position()
    relics = msg["run_state"]["relics"]
    if not any(r["id"] == "LEAD_PAPERWEIGHT" for r in relics):
        relics.append({"id": "LEAD_PAPERWEIGHT"})

    rm, _notes = ri.graft_run(msg)
    rm.debug_enter_event("TABLET_OF_TRUTH")
    rm.take_action({"action": "event_choice", "idx": 0})

    # The live run took Lead Paperweight's colorless pick long ago: the
    # event's next page comes next, not that stale choose screen.
    assert not rm.card_offer_choose_screen, rm.card_reward_option_ids()
    assert rm.phase() != "card_reward", rm.card_reward_option_ids()


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="no Instinct weights (data/e2e)")
@pytest.mark.parametrize("enchantment", ["CORRUPTED", "STEADY"])
def test_an_enchant_screen_gets_the_native_students_answer(
    enchantment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, msg = _map_position()

    # The native student on the same enchant pick.
    native_rm = base.clone()
    native_rm.debug_queue_deck_enchant(enchantment)
    driver = sts2_rs.E2eDriver(native_rm.seed)
    peek = driver.peek(native_rm)
    assert peek.kind == "decision"
    assert peek.decision["decision_type"] == "CARD_ENCHANT"
    native = sts2_rs.E2eStudent(str(WEIGHTS)).decide(driver.copy(), peek, native_rm)

    deck = list(native_rm.master_deck_detailed())
    eligible = list(native_rm.enchant_select_deck_indices())
    assert eligible, "no card takes this enchantment"
    msg.update(
        advice_for="card_select",
        prompt_key="TO_ENCHANT",
        source_model=enchantment,
        source_kind="enchantment",
        min_select=1,
        max_select=1,
        cards=[
            {
                "index": k,
                "id": _wire_id(deck[i][0]),
                "name": deck[i][0],
                "upgrade_level": int(bool(deck[i][1])),
            }
            for k, i in enumerate(eligible)
        ],
    )

    advisor = ri.InstinctAdvisor(WEIGHTS)
    advisor.last_screen = "event"
    picks: list[Any] = []
    real = advisor.evaluate

    def spy(*a: Any, **k: Any) -> Any:
        picks.append(real(*a, **k))
        return picks[-1]

    monkeypatch.setattr(advisor, "evaluate", spy)
    advice = advisor.advise(msg, "card_select", session={})

    assert advice["brain"] == "instinct", advice
    assert advice.get("best"), advice
    pick = picks[-1]
    assert pick.decision["decision_type"] == "CARD_ENCHANT"
    assert pick.best == native["class"], (pick.probs, native["probs"])
    # Probabilities only to 2e-2: the advisor's driver starts fresh on the rebuilt run, so the
    # history the native driver carries differs (d1250 drifted 1-4e-3 here, d2500 avg5 up to
    # 1.8e-2 on STEADY / 9e-3 on CARD_GRANT). The pick itself must match exactly (above).
    assert pick.probs == pytest.approx(native["probs"], abs=2e-2)


def test_an_enchant_screen_without_its_enchantment_still_declines() -> None:
    import realgame_instinct_screens as screens

    assert screens._screen_enchantment({"source_kind": "enchantment"}) is None
    assert screens._screen_enchantment({"source_model": "NIMBLE"}) == "NIMBLE"
    assert screens._screen_enchantment({"source_model": "X", "source_kind": "relic"}) is None
    assert screens._shared_card_type([{"type": "Skill"}, {"type": "Skill"}]) == "skill"
    assert screens._shared_card_type([{"type": "Skill"}, {"type": "Attack"}]) is None
