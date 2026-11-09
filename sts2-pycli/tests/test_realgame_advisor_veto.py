"""The wire's own legality verdict vetoes a ranked play.

The search runs on a reconstruction of the live board, and across the first 51
ascension-climb runs the game refused 41 recommended plays in three unrelated
classes. Each case below is one of those classes, taken from an archived
``godot.log`` -- the fixture cards, reasons and hand positions are the ones the
live game actually reported.

Everything here is dict-in/dict-out: no torch, no checkpoint, no game.
"""

from __future__ import annotations

from typing import Any

import realgame_advisor as ra
from sts2_hint.recommender import END_TURN_STABLE_IDX, HintTag


def _tag(stable_idx: int, *, q: float, is_best: bool = False) -> HintTag:
    return HintTag(
        stable_idx=stable_idx,
        win_prob=q,
        bucket_label="likely-win",
        is_best=is_best,
        visits=100,
        q_saturated=False,
        low_confidence=False,
        target_index=0 if stable_idx >= 0 else None,
        q=q,
        wins=1,
        losses=0,
        ties=0,
    )


def _state(hand: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "combat_action", "hand": hand, "potions": []}


def _card(cid: str, *, playable: bool = True, reason: str | None = None) -> dict[str, Any]:
    return {"id": cid, "playable": playable, "unplayable_reason": reason}


def test_chains_of_binding_bound_card_is_vetoed() -> None:
    """Run 050, Queen fight: a second Bound card, refused BlockedByHook.

    22 of the 41 refusals are this one, and every one was in a QUEEN_BOSS
    fight -- the mirror's `chains_bound_card_played_this_turn` never fires on a
    rebuilt state, so it lists a play the game's own hook rejects.
    """
    state = _state(
        [
            _card("DEFEND_SILENT", playable=False, reason="BlockedByHook"),
            _card("STRIKE_SILENT"),
        ]
    )
    tags = [_tag(0, q=0.9, is_best=True), _tag(1, q=0.4)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [1]
    assert vetoed == ["hand[0]:BlockedByHook"]
    assert kept[0].is_best, "the crown must move to the surviving leader"


def test_energy_cost_too_high_is_vetoed() -> None:
    """Run 051: Strike recommended at 0 energy after a three-card turn."""
    state = _state(
        [
            _card("DEFEND_SILENT"),
            _card("STRIKE_SILENT", playable=False, reason="EnergyCostTooHigh"),
        ]
    )
    tags = [_tag(1, q=0.8, is_best=True), _tag(0, q=0.7)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [0]
    assert vetoed == ["hand[1]:EnergyCostTooHigh"]


def test_unplayable_keyword_is_vetoed() -> None:
    """Runs 011/051: the search recommended playing a CLUMSY curse."""
    state = _state(
        [
            _card("CLUMSY", playable=False, reason="HasUnplayableKeyword"),
            _card("SHIV"),
        ]
    )
    tags = [_tag(0, q=0.6, is_best=True), _tag(1, q=0.5)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [1]
    assert vetoed == ["hand[0]:HasUnplayableKeyword"]


def test_end_turn_survives_a_hand_that_is_entirely_unplayable() -> None:
    """Every card vetoed must still leave an action, or autoplay stalls.

    Ending the turn is where the mod's escape hatch lands after three wasted
    refusals; reaching it directly costs nothing and loses no information.
    """
    state = _state([_card("CLUMSY", playable=False, reason="HasUnplayableKeyword")])
    tags = [_tag(0, q=0.6, is_best=True)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert len(kept) == 1
    assert kept[0].stable_idx == END_TURN_STABLE_IDX
    assert kept[0].is_best
    assert kept[0].target_index is None
    assert vetoed == ["hand[0]:HasUnplayableKeyword"]


def test_end_turn_is_never_vetoed_by_a_coincident_hand_index() -> None:
    """`END_TURN_STABLE_IDX` is -1; only non-negative tags index the hand."""
    state = _state([_card("CLUMSY", playable=False, reason="HasUnplayableKeyword")])
    tags = [_tag(END_TURN_STABLE_IDX, q=0.5, is_best=True), _tag(0, q=0.9)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [END_TURN_STABLE_IDX]
    assert vetoed == ["hand[0]:HasUnplayableKeyword"]


def test_a_playable_hand_is_left_exactly_alone() -> None:
    """No veto means no re-stamping, no reordering, no allocation of blame."""
    state = _state([_card("STRIKE_SILENT"), _card("DEFEND_SILENT")])
    tags = [_tag(0, q=0.9, is_best=True), _tag(1, q=0.4)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert kept == tags
    assert vetoed == []


def test_missing_legality_is_treated_as_playable() -> None:
    """A field the producer did not send must never suppress a legal move."""
    state = _state([{"id": "STRIKE_SILENT"}, {"id": "DEFEND_SILENT"}])
    tags = [_tag(0, q=0.9, is_best=True), _tag(1, q=0.4)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert kept == tags
    assert vetoed == []


def test_potion_tags_are_never_vetoed_by_hand_legality() -> None:
    """Potion stable indices are large negatives; they do not index the hand."""
    state = _state([_card("CLUMSY", playable=False, reason="HasUnplayableKeyword")])
    tags = [_tag(-100, q=0.7, is_best=True), _tag(0, q=0.9)]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [-100]
    assert vetoed == ["hand[0]:HasUnplayableKeyword"]


def test_build_advice_reports_the_veto_in_its_note() -> None:
    """The count is the only surviving record of a mirror/game disagreement."""
    state = _state(
        [
            _card("DEFEND_SILENT", playable=False, reason="BlockedByHook"),
            _card("STRIKE_SILENT"),
        ]
    )
    tags = [_tag(0, q=0.9, is_best=True), _tag(1, q=0.4)]

    advice = ra.build_advice(tags, state, sims=4000)

    assert advice["best"]["card_index"] == 1
    assert "vetoed 1" in advice["note"]
    # Named for the player (the card and the game's reason in words), not as
    # `hand[0]:BlockedByHook`.
    assert "(blocked by hook)" in advice["note"]
    assert "hand[" not in advice["note"]


def test_vetoed_pick_passes_the_crown_to_the_engines_next_choice() -> None:
    """GH #1022: the heir is the engine's own next choice (lowest
    ``pick_rank``), not the survivor with the highest win probability.

    The engine ranked Defend second -- an in-band block the tie rule prefers
    -- although Strike reads higher on p_win.
    """
    state = _state(
        [
            _card("BASH", playable=False, reason="EnergyCostTooHigh"),
            _card("STRIKE_IRONCLAD"),
            _card("DEFEND_IRONCLAD"),
        ]
    )
    tags = [
        _tag(0, q=0.9, is_best=True)._replace(pick_rank=0, in_band=True),
        _tag(1, q=0.8)._replace(pick_rank=2, in_band=True),
        _tag(2, q=0.7)._replace(pick_rank=1, in_band=True),
    ]

    kept, vetoed = ra.veto_unplayable(tags, state)

    assert [t.stable_idx for t in kept] == [2, 1]
    assert kept[0].is_best and not kept[1].is_best
    assert vetoed == ["hand[0]:EnergyCostTooHigh"]


def test_advice_carries_the_engine_pick_only_while_it_stands() -> None:
    """``advice.pick`` explains the engine's pick; once a veto moved the
    crown it would explain a move nobody plays, so it is dropped."""
    pick = {
        "rule": "band_visits",
        "reason": "the most searched of the equal moves",
        "gap_q": -0.00412345,
        "gap_z": None,
        "gap_hp": 1.5,
    }
    tags = [
        _tag(0, q=0.9, is_best=True)._replace(pick_rank=0, in_band=True),
        _tag(1, q=0.8)._replace(pick_rank=1, in_band=True),
    ]
    playable = _state([_card("BASH"), _card("STRIKE_IRONCLAD")])
    advice = ra.build_advice(tags, playable, sims=8, pick=pick)
    assert advice["pick"] == {
        "rule": "band_visits",
        "reason": pick["reason"],
        "gap_q": -0.0041,
        "gap_z": None,
        "gap_hp": 1.5,
    }
    assert [(o["in_band"], o["pick_rank"]) for o in advice["options"]] == [(True, 0), (True, 1)]

    refused = _state(
        [_card("BASH", playable=False, reason="EnergyCostTooHigh"), _card("STRIKE_IRONCLAD")]
    )
    advice = ra.build_advice(tags, refused, sims=8, pick=pick)
    assert "pick" not in advice
    assert advice["options"][0]["is_best"] and advice["options"][0]["card_index"] == 1
