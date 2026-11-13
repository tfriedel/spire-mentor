"""Deck-mutation prompts reach the trained net, and the fallback survives.

This file used to pin the GH #432 stopgap: `realgame_advisor.deck_mutation_override`
answering TO_REMOVE / TO_TRANSFORM / TO_ENCHANT with a curse-first heuristic,
because `realgame_ppo._build_card_select` sent those prompts to its `auto_front`
surface — a replica of the flags-off engine's `master_deck.drain(0..take)` that
never consulted the policy. Front-of-deck can only remove what happens to sit at
the front, which is why run 025 carried a **Clumsy from floor 5 to floor 47**.

All five champions are now trained with the deck-mutation surfaces visible
(GH #399/#427), so both halves of that are gone: the override is deleted and the
three prompts route to their own trained heads. What these tests hold shut is
the state that replaced it, plus the one piece that deliberately did **not** go
away (GH #477).
"""

from __future__ import annotations

import random
from typing import Any

import realgame_hand_select
import realgame_ppo as rp


def _card(cid: str, index: int, ctype: str, rarity: str, **kw: Any) -> dict[str, Any]:
    """A card shaped like the wire's, not like a convenient stub.

    `SimStateExtractor.SerializeCard` sends `type`, `rarity`, `upgrade_level`,
    `cost` and `is_curse`, and `card_value` ranks on exactly those. An earlier
    version of this fixture omitted them, every card tied at the default class,
    and the heuristic returned the leftmost card — which is the bug it exists to
    fix. Keep these fields.
    """
    return {
        "id": cid,
        "name": cid.title(),
        "index": index,
        "type": ctype,
        "rarity": rarity,
        "is_curse": ctype == "Curse",
        "upgrade_level": 0,
        "cost": 1,
        **kw,
    }


#: The run-025 shape: a curse sits in the deck behind the starting basics.
_RUN025_CARDS = [
    _card("STRIKE_IRONCLAD", 0, "Attack", "Basic"),
    _card("DEFEND_IRONCLAD", 1, "Skill", "Basic"),
    _card("CLUMSY", 2, "Curse", "Curse"),
]


def _state(prompt: str, cards: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return {
        "type": "card_select",
        "prompt_key": prompt,
        "cards": cards,
        "run_state": {
            "character": "Ironclad",
            "master_deck": [
                {"id": "STRIKE_IRONCLAD"},
                {"id": "DEFEND_IRONCLAD"},
                {"id": "CLUMSY"},
            ],
        },
        **kw,
    }


# ── the three prompts now reach a trained head ───────────────────────────────


def test_the_three_prompts_route_to_their_own_trained_surfaces() -> None:
    """Not `auto_front`, and not each other's head.

    Sending a transform to the `removal` head would featurize the state token
    as `card-remove`, i.e. ask the net a question it was not asked here.
    """
    for prompt, kind in (
        ("TO_REMOVE", "remove_deck_card"),
        ("TO_TRANSFORM", "transform_deck_card"),
        ("TO_ENCHANT", "enchant_deck_card"),
    ):
        decision = rp._build_card_select(_state(prompt, _RUN025_CARDS), None)
        assert decision is not None, prompt
        assert {a["action"] for a in decision.actions} == {kind}, prompt
        # One action per offered card -- a RANKING, not a single forced answer.
        assert len(decision.actions) == len(_RUN025_CARDS), prompt
        assert decision.source_note == "", f"{prompt} must not carry an engine byline"


def test_the_trained_heads_keep_their_own_prompts() -> None:
    """TO_UPGRADE -> `smith`, TO_OBTAIN -> `grant`, unchanged by the retirement."""
    up = rp._build_card_select(_state("TO_UPGRADE", _RUN025_CARDS), None)
    ob = rp._build_card_select(_state("TO_OBTAIN", _RUN025_CARDS), None)
    assert up is not None and {a["action"] for a in up.actions} == {"smith_card"}
    assert ob is not None and {a["action"] for a in ob.actions} == {"grant_deck_card"}


def test_transform_and_enchant_carry_deck_indices_not_card_pairs() -> None:
    """These two are shaped unlike every other surface here (GH #477).

    `encoder._build_action_token` dereferences `transform_idxs[idx]` into
    `ctx.deck`, so the obs must carry master-deck INDICES. Getting this wrong
    is silent: the net would score a different card than the screen offered.
    """
    d = rp._build_card_select(_state("TO_TRANSFORM", _RUN025_CARDS), None)
    assert d is not None
    # deck order is Strike, Defend, Clumsy -- and the offers are in that order.
    assert d.obs["transform_select_deck_indices"] == [0, 1, 2]

    e = rp._build_card_select(_state("TO_ENCHANT", _RUN025_CARDS), None)
    assert e is not None
    assert e.obs["enchant_select_deck_indices"] == [0, 1, 2]


def test_a_deck_index_resolves_to_the_offered_card() -> None:
    """The index is only ever dereferenced back to `(card_id, upgraded)`.

    Offer the deck in a different order than it is stored, and every index must
    still name the card the screen actually offered.
    """
    cards = [
        _card("CLUMSY", 0, "Curse", "Curse"),
        _card("STRIKE_IRONCLAD", 1, "Attack", "Basic"),
    ]
    d = rp._build_card_select(_state("TO_TRANSFORM", cards), None)
    assert d is not None
    deck = d.obs["deck"]
    for offer, deck_i in zip(cards, d.obs["transform_select_deck_indices"], strict=True):
        assert deck[deck_i][0] == offer["id"]


def test_a_card_absent_from_the_master_deck_degrades_instead_of_raising() -> None:
    """The never-crash contract. An out-of-range index costs one option's
    identity (`encoder._at` returns None -> unknown-card token); it must not
    raise and must not silently point at some OTHER card in the deck."""
    cards = [_card("NOT_IN_DECK", 0, "Attack", "Common")]
    d = rp._build_card_select(_state("TO_TRANSFORM", cards), None)
    assert d is not None
    (idx,) = d.obs["transform_select_deck_indices"]
    assert idx >= len(d.obs["deck"]), "must not alias onto an unrelated deck card"


def test_the_response_still_uses_the_option_list_index() -> None:
    """Deck indices are for the OBS only. What goes back on the wire is the
    bridge's own `index` (PROTOCOL.md section 3 `card_select`), which is the
    position in this selection's option list."""
    cards = [
        _card("CLUMSY", 7, "Curse", "Curse"),
        _card("STRIKE_IRONCLAD", 9, "Attack", "Basic"),
    ]
    d = rp._build_card_select(_state("TO_TRANSFORM", cards), None)
    assert d is not None
    assert [r["index"] for r in d.responses] == [7, 9]


# ── what deliberately did NOT go away (GH #477) ──────────────────────────────


def test_hand_select_still_knows_the_deck_mutation_directions() -> None:
    """`realgame_hand_select` is NOT part of the retired stopgap.

    GH #430's checklist said to delete its deck-mutation branch. That would be
    a regression, not a cleanup. The original reason -- `_build_card_select`
    bailing on any `max_select > 1` -- no longer holds (it ranks those now),
    but the table still answers every deck-mutation prompt the net has NO
    surface for, and events ship their own prompt keys
    (`ROOM_FULL_OF_CHEESE.pages.GORGE.selectionScreenPrompt`,
    `BRAIN_LEECH.pages.SHARE_KNOWLEDGE.selectionScreenPrompt`, both live in
    `logs/f7-mirror-check18`). Without it those fall to `_unranked`'s leftmost
    default -- the bug
    `docs/experiments/2026-08-06-ascension-climb-issues.md` issue 5 is about.
    """
    for prompt in ("TO_REMOVE", "TO_TRANSFORM", "TO_UPGRADE", "TO_ENCHANT"):
        assert prompt in realgame_hand_select._PROMPT_DIRECTION, prompt


def test_a_multi_pick_deck_mutation_is_ranked_by_the_net() -> None:
    """It used to be declined, and the decline meant a coin flip.

    The live mod re-asks a "remove 2" screen once per pick with the option
    list one card shorter and min/max unchanged (check18 workbench logs:
    TO_REMOVE min=2 max=2 cards=22, then cards=21), so ranking the shrinking
    list is both correct and all the advisor needs. `pick_count` is what the
    DRIVER needs -- `SimCardSelector` rejects a reply naming fewer than
    `minSelect`.
    """
    state = _state("TO_REMOVE", _RUN025_CARDS, min_select=2, max_select=2)
    decision = rp._build_card_select(state, None)
    assert decision is not None, "a multi-pick removal is still a ranking over cards"
    assert decision.pick_count == 2
    assert len(decision.actions) == len(_RUN025_CARDS)

    # The heuristic is still there for prompts the net has no surface for.
    ranked = realgame_hand_select.rank(state)
    assert ranked is not None
    assert ranked["best"]["index"] == 2, "should still reach the curse, not the lead Strike"


def test_the_reply_to_a_multi_pick_names_every_slot() -> None:
    """`SimCardSelector.cs:368` rejects a short reply, so fill the slots.

    One inference ranked once, top-k -- not the net re-consulted with the
    first pick removed. Close enough for "remove 2 of 22" and unarguably
    better than the uniform random this replaced; not the same thing.
    """

    class _Ranked:
        """Ranks the curse first and the Defend second, so the fill is checkable.

        A uniform stub would pass this test whatever the fill did -- and so
        would the random fallback a null policy degrades to, which is what
        this path used to take.
        """

        def distribution(self, obs, actions, metrics):  # type: ignore[no-untyped-def]
            weights = {0: 0.1, 1: 0.3, 2: 0.6}  # Strike, Defend, Clumsy
            return [weights[int(a["idx"])] for a in actions], 0.5

    policies = rp.PpoPolicyCache(None)
    policies._cache["ironclad"] = (_Ranked(), None)

    state = _state("TO_REMOVE", _RUN025_CARDS, min_select=2, max_select=2)
    action, _ = rp.pick_noncombat_action_ppo(
        state,
        random.Random(0),
        policies,
        session={"prev_msg_type": "event"},
    )
    assert action["indices"] == [2, 1], action
    assert len(set(action["indices"])) == 2, "two slots must name two different cards"


# -- the per-source prompts (static audit) -----------------------------------

#: The prompts whose screen only ever opens DURING a fight, so the combat
#: mirror answers them (`realgame_mcts.pick_card_select_mirrored`) and no deck
#: surface applies. Listed so the audit below can tell "classified as combat"
#: apart from "nobody has looked at it yet".
_COMBAT_PROMPTS = {
    "ASHWATER.selectionScreenPrompt",
    "CHOICES_PARADOX.selectionScreenPrompt",
    "COSMIC_INDIFFERENCE.selectionScreenPrompt",
    "DECISIONS_DECISIONS.selectionScreenPrompt",
    "DREDGE.selectionScreenPrompt",
    "DROPLET_OF_PRECOGNITION.selectionScreenPrompt",
    "DUAL_WIELD.selectionScreenPrompt",
    "FOREGONE_CONCLUSION_POWER.selectionScreenPrompt",
    "GAMBLERS_BREW.selectionScreenPrompt",
    "GAMBLING_CHIP.selectionScreenPrompt",
    "GLIMMER.selectionScreenPrompt",
    "GRAVEBLAST.selectionScreenPrompt",
    "GUARDS.selectionScreenPrompt",
    "HAND_TRICK.selectionScreenPrompt",
    "HEADBUTT.selectionScreenPrompt",
    "HEIRLOOM_HAMMER.selectionScreenPrompt",
    "HOLOGRAM.selectionScreenPrompt",
    "LIQUID_MEMORIES.selectionScreenPrompt",
    "NEOWS_FURY.selectionScreenPrompt",
    "NIGHTMARE.selectionScreenPrompt",
    "PHOTON_CUT.selectionScreenPrompt",
    "PURITY.selectionScreenPrompt",
    "SCULPTING_STRIKE.selectionScreenPrompt",
    "SECRET_TECHNIQUE.selectionScreenPrompt",
    "SECRET_WEAPON.selectionScreenPrompt",
    "SEEKER_STRIKE.selectionScreenPrompt",
    "SNAP.selectionScreenPrompt",
    "STRATAGEM_POWER.selectionScreenPrompt",
    "THINKING_AHEAD.selectionScreenPrompt",
    "TOUCH_OF_INSANITY.selectionScreenPrompt",
    "TRANSFIGURE.selectionScreenPrompt",
    "WELL_LAID_PLANS_POWER.selectionScreenPrompt",
    "WISH.selectionScreenPrompt",
    # v0.113.0: The Fool's cards, all `CardSelectCmd.FromCombatPile`
    # (Models.Cards/ArrowStab.cs, KiChant.cs, LockedIn.cs, Recall.cs, Tutor.cs).
    "ARROW_STAB.selectionScreenPrompt",
    "KI_CHANT.selectionScreenPrompt",
    "LOCKED_IN.selectionScreenPrompt",
    "RECALL.selectionScreenPrompt",
    "TUTOR.selectionScreenPrompt",
}


#: Every `*.selectionScreenPrompt` that is NOT an in-combat hand/draw/discard
#: operation, i.e. the ones a deck screen can raise. Derived from the decompiled
#: effect, not the prompt wording -- see `_PROMPT_SURFACES`.
_NON_COMBAT_PROMPTS = {
    "BRAIN_LEECH.pages.SHARE_KNOWLEDGE.selectionScreenPrompt": "grant",
    "ROOM_FULL_OF_CHEESE.pages.GORGE.selectionScreenPrompt": "grant",
    "SEA_GLASS.selectionScreenPrompt": "grant",
    "CLAWS.selectionScreenPrompt": "transform",
    # Ranked by the heuristic on purpose -- its options are the player's own
    # deck and it adds a COPY, which is not what the grant head saw.
    "DOLLYS_MIRROR.selectionScreenPrompt": None,
    # The shared removal prompt; already covered by `_SOURCE_DIRECTION` too.
    "COMBAT_REWARD_CARD_REMOVAL.selectionScreenPrompt": None,
    # v0.113.0 Blitz Neow: 3 of 10 generated cards into the deck
    # (Models.Blitz/BlitzRunModel.cs:24-35).
    "BLITZ.NEOW.selectionScreenPrompt": "grant",
    # v0.113.0: shipped, but `CardUpgradeReward` asks with `TO_UPGRADE`.
    "COMBAT_REWARD_CARD_UPGRADE.selectionScreenPrompt": "smith",
}

#: Every vendored localization the live advisor can meet: the baseline plus
#: each newer supported build's own copy (a new build's prompts appear only
#: there).
_LOC_DIRS = ("localization_eng", "localization_eng_v0.113.0")


def _loc_selection_prompts() -> dict[str, str]:
    """Every ``*.selectionScreenPrompt`` string shipped in any of `_LOC_DIRS`."""
    import json
    from pathlib import Path

    pycli = Path(__file__).resolve().parents[1]

    def walk(node: Any, path: list[str]):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, [*path, k])
        elif isinstance(node, str):
            yield ".".join(path), node

    out: dict[str, str] = {}
    files = [f for d in _LOC_DIRS for f in sorted((pycli / d).glob("*.json"))]
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for k, v in walk(data, []):
            if k.endswith("selectionScreenPrompt"):
                out[k] = v
    return out


def test_every_non_combat_selection_prompt_has_a_surface() -> None:
    """A source with its own prompt string matches no shared `TO_*` key.

    So it used to reach `_unranked`'s leftmost default with a trained head
    sitting right there -- `SEA_GLASS` and both event prompts did exactly that
    in `logs/f7-mirror-check18`. The set is closed and small (39 strings in
    `localization_eng`, all but a handful in-combat), so it is audited here
    rather than discovered one live run at a time: a NEW prompt fails this
    test instead of silently going leftmost.
    """
    shipped = _loc_selection_prompts()
    assert shipped, "the localization files must be readable for this audit"
    for key in _NON_COMBAT_PROMPTS:
        assert key in shipped, f"{key} is no longer in localization_eng"

    unknown = set(shipped) - set(_NON_COMBAT_PROMPTS) - _COMBAT_PROMPTS
    assert not unknown, (
        "new selectionScreenPrompt(s) -- classify each from its DECOMPILED "
        f"effect and add it to `_PROMPT_SURFACES` (deck screen) or "
        f"`_COMBAT_PROMPTS` (answered by the combat mirror): {sorted(unknown)}"
    )

    for key, surface in _NON_COMBAT_PROMPTS.items():
        assert rp._PROMPT_SURFACES.get(key.upper()) == surface, key


def test_a_relic_that_adds_to_the_deck_is_ranked_not_leftmost() -> None:
    """Sea Glass: "Select any number of cards to add to your Deck."

    `FromSimpleGridForRewards` -> `CardPileCmd.Add(item, Deck)`
    (SeaGlass.cs:74-91), so this is the grant surface. Live in
    `logs/f7-mirror-check18` run 004, answered by the rarity heuristic.
    """
    state = _state("SEA_GLASS.selectionScreenPrompt", _RUN025_CARDS, min_select=0, max_select=3)
    decision = rp._build_card_select(state, None)
    assert decision is not None, "a deck grant is not an unranked screen"
    assert [a["action"] for a in decision.actions] == ["grant_deck_card"] * 3


def test_blitz_neow_pick_is_ranked_not_leftmost() -> None:
    """v0.113.0 Blitz Neow: "add 3 of these 10 cards to your deck".

    `FromSimpleGridForRewards(prefs(3))` -> `CardPileCmd.Add(item, Deck)`
    (Models.Blitz/BlitzRunModel.cs:24-35). Unmapped, the live advisor said
    "the net does not cover this surface" and took the first three cards of
    the rarity-sorted list (2026-11-13). The mod re-asks once per pick with
    min/max 3 and the list one card shorter, so each ask is a grant ranking.
    """
    state = _state("BLITZ.NEOW.selectionScreenPrompt", _RUN025_CARDS, min_select=3, max_select=3)
    decision = rp._build_card_select(state, None)
    assert decision is not None, "the Blitz Neow pick is a deck grant, not an unranked screen"
    assert [a["action"] for a in decision.actions] == ["grant_deck_card"] * 3
