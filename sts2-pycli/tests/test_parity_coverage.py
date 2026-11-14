"""Tests for the coverage-driven parity scenario generator.

`coverage_scenarios()` emits a deterministic, single-variable (one-hot)
sweep that exercises every Ironclad card, colorless card, relic, potion,
and Act-1 encounter at least once, while keeping each trial isolated so a
divergence is attributable to exactly one varied element.

These are pure-Python tests — no C# subprocess / Rust wheel needed.
"""

from __future__ import annotations

import parity_fuzz_scenario as m


def _ic_pool() -> list[str]:
    """Ironclad + colorless cards of the selected build (`--game-version`).

    The sweeps follow `$STS2_GAME_VERSION`; v0.111.0 drops Scare and appends
    its new cards, so a count against the raw v0.107.1 lists would not hold.
    """
    return m.class_cards_for_build("Ironclad") + m.colorless_cards_for_build()


def _subjects(prefix: str) -> set[str]:
    return {
        s.coverage_subject.split(":", 1)[1]
        for s in m.coverage_scenarios()
        if s.coverage_subject and s.coverage_subject.startswith(prefix)
    }


def test_every_ic_card_covered_exactly_once() -> None:
    covered = _subjects("card:")
    for card in m.ALL_IC_CARDS:
        assert card in covered, f"IC card {card} not covered"
    # each card subject appears exactly once
    subjects = [
        s.coverage_subject
        for s in m.coverage_scenarios()
        if s.coverage_subject and s.coverage_subject.startswith("card:")
    ]
    assert len(subjects) == len(set(subjects)), "duplicate card coverage subject"


def test_every_colorless_card_covered() -> None:
    covered = _subjects("card:")
    for card in m.ALL_COLORLESS_CARDS:
        assert card in covered, f"colorless card {card} not covered"


def test_every_relic_covered() -> None:
    covered = _subjects("relic:")
    assert covered == set(m.ALL_RELICS)


def test_every_potion_covered() -> None:
    covered = _subjects("potion:")
    assert covered == set(m.ALL_POTIONS)


def test_every_encounter_covered() -> None:
    covered = _subjects("encounter:")
    assert covered == set(m.ALL_ENCOUNTERS)


def test_additive_not_cartesian() -> None:
    scenarios = m.coverage_scenarios()
    expected = (
        len(_ic_pool())
        + len(m.event_cards_for_build())
        + len(m.ALL_RELICS)
        + len(m.ALL_POTIONS)
        + len(m.ALL_ENCOUNTERS)
    )
    assert len(scenarios) == expected
    # additive sweep, never a product blow-up. The bound tracks the inventory
    # (it was 400 when the pack held 390); it is a blow-up tripwire, not a
    # budget — raise it when content lands, and worry only if it doubles.
    assert len(scenarios) < 500


def test_every_subject_unique() -> None:
    subjects = [s.coverage_subject for s in m.coverage_scenarios()]
    assert all(subjects), "every coverage scenario must carry a subject"
    assert len(subjects) == len(set(subjects)), "coverage subjects must be unique"


def test_card_trials_isolate_one_variable() -> None:
    """A card trial varies only the card: no relics, no potions, the test
    card guaranteed in the opening hand (deck fits in one 5-card draw)."""
    for s in m.coverage_scenarios():
        if not (s.coverage_subject or "").startswith("card:"):
            continue
        card = s.coverage_subject.split(":", 1)[1]
        assert s.relics == (), f"card trial {card} leaked relics {s.relics}"
        assert s.potions == (), f"card trial {card} leaked potions {s.potions}"
        assert card in s.deck, f"card trial {card} missing from its own deck"
        # whole deck fits in the opening hand → the test card is always drawn
        assert len(s.deck) <= 5, f"card trial {card} deck too big to guarantee draw"
        assert s.encounter == m.ARENA_ENCOUNTER


def test_relic_trials_isolate_one_relic() -> None:
    for s in m.coverage_scenarios():
        if not (s.coverage_subject or "").startswith("relic:"):
            continue
        relic = s.coverage_subject.split(":", 1)[1]
        assert s.relics == (relic,), f"relic trial {relic} not isolated: {s.relics}"
        assert s.potions == ()
        assert s.encounter == m.ARENA_ENCOUNTER
        # baseline starter deck, no test cards smuggled in
        assert set(s.deck) <= {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}


def test_potion_trials_isolate_one_potion() -> None:
    for s in m.coverage_scenarios():
        if not (s.coverage_subject or "").startswith("potion:"):
            continue
        potion = s.coverage_subject.split(":", 1)[1]
        assert s.potions == (potion,), f"potion trial {potion} not isolated: {s.potions}"
        assert s.relics == ()
        assert s.encounter == m.ARENA_ENCOUNTER


def test_encounter_trials_use_clean_baseline_loadout() -> None:
    for s in m.coverage_scenarios():
        if not (s.coverage_subject or "").startswith("encounter:"):
            continue
        enc = s.coverage_subject.split(":", 1)[1]
        assert s.encounter == enc
        assert s.relics == ()
        assert s.potions == ()
        assert set(s.deck) <= {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}


def test_deterministic() -> None:
    assert m.coverage_scenarios() == m.coverage_scenarios()


def test_arena_is_a_single_known_encounter() -> None:
    # the arena baseline must itself be a real Act-1 encounter
    assert m.ARENA_ENCOUNTER in m.ALL_ENCOUNTERS


def test_upgraded_sweep_covers_every_card_once() -> None:
    sweep = m.coverage_upgraded_scenarios()
    subjects = [s.coverage_subject for s in sweep]
    assert len(subjects) == len(set(subjects)), "duplicate card+ subject"
    assert all(s.startswith("card+:") for s in subjects)
    covered = {s.split(":", 1)[1] for s in subjects}
    assert covered == set(_ic_pool())


def test_upgraded_sweep_deck_shape() -> None:
    for s in m.coverage_upgraded_scenarios():
        # card under test carries the +1 marker and is first
        assert s.deck[0].endswith("+1")
        assert s.deck[0].removesuffix("+1") == s.coverage_subject.split(":", 1)[1]
        # filler is inert basics, UNUPGRADED (only one varied element)
        assert all("+" not in c for c in s.deck[1:])
        assert s.coverage_subject.startswith("card+:")
        assert s.encounter == m.ARENA_ENCOUNTER
        assert s.relics == () and s.potions == ()


def test_arena_cross_subjects_qualified_and_complete() -> None:
    sweep = m.coverage_arena_cross_scenarios()
    subjects = [s.coverage_subject for s in sweep]
    assert len(subjects) == len(set(subjects))
    # act1 is NOT in the cross — the v1 coverage + Axis A already run
    # every subject against the act1 arena; the cross adds act2 + act3.
    assert all(s.endswith(("@act2", "@act3")) for s in subjects)
    expected_per_act = (
        2 * len(_ic_pool())  # base + upgraded
        + len(m.ALL_RELICS)
        + len(m.ALL_POTIONS)
    )
    assert len(subjects) == 2 * expected_per_act


def test_arena_cross_encounters_match_table() -> None:
    for s in m.coverage_arena_cross_scenarios():
        act = s.coverage_subject.rsplit("@", 1)[1]
        assert s.encounter == m.ARENAS_BY_ACT[act]


def test_arena_preflight_starter_deck() -> None:
    pre = m.arena_preflight_scenarios()
    assert {s.coverage_subject for s in pre} == {"arena:act2", "arena:act3"}
    for s in pre:
        assert s.deck == m._STARTER_DECK
        assert s.relics == () and s.potions == ()


def test_hook_group_ids_resolve_against_pools() -> None:
    cards = set(m.ALL_IC_CARDS + m.ALL_COLORLESS_CARDS)
    relics = set(m.ALL_RELICS)
    for group, spec in m.HOOK_GROUPS.items():
        for gid in spec["generators"]:
            assert gid in cards, f"{group} generator {gid} not a known card"
        for lid in spec["listeners"]:
            assert lid in cards | relics, f"{group} listener {lid} unknown"


def test_pairwise_scenarios_shape() -> None:
    sweep = m.pairwise_hook_scenarios()
    assert len(sweep) > 0
    subjects = [s.coverage_subject for s in sweep]
    assert len(subjects) == len(set(subjects))
    for s in sweep:
        assert s.coverage_subject.startswith("pair:")
        assert s.encounter == m.ARENA_ENCOUNTER
        gen, _, listener = s.coverage_subject[5:].partition("x")
        assert gen in s.deck
        # listener is either in the deck (card) or the relic slot
        assert listener in s.deck or s.relics == (listener,)


# ── engine-derived inventory gate ────────────────────────────────────────
#
# Everything above checks a hand-maintained list against ITSELF: they assert
# each name in `ALL_*_CARDS` gets a trial, never that the lists name every card
# the engine has. So a card the transcription missed was invisible — it showed
# as covered because nothing ever asked the engine.
#
# That is not hypothetical. `ALL_COLORLESS_CARDS` was transcribed from the
# colorless spec slice alone, so all 19 Event- and 18 Ancient-rarity cards
# (which live in `status_specs_generated.rs`) had NEVER been driven against the
# C# oracle — found one card at a time while porting HelloWorld (GH #636) and
# Rebound (GH #637), the second of which had a real engine bug waiting the
# moment its card was finally swept. `NOT_YET` and the Defect / Necrobinder
# starters were missing for no reason at all.
#
# The two tests below close that by deriving the truth from the Rust source.


def test_engine_card_rarities_are_unambiguous() -> None:
    # `card_rarities()` takes the first spelling of each CardId. That is only
    # safe while every duplicate agrees, so prove it rather than assume it.
    import re

    from card_inventory import _CONTENT_SRC, _SPEC_BODY, _SPEC_ID, _SPEC_RARITY

    seen: dict[str, set[str]] = {}
    for path in sorted(_CONTENT_SRC.rglob("*.rs")):
        if path.name == "tests.rs":
            continue
        for body in _SPEC_BODY.findall(path.read_text(encoding="utf-8")):
            cid = _SPEC_ID.search(body)
            rar = _SPEC_RARITY.search(body)
            if cid and rar:
                seen.setdefault(cid.group(1), set()).add(rar.group(1))
    conflicts = {k: v for k, v in seen.items() if len(v) > 1}
    assert not conflicts, f"CardIds built with conflicting rarity: {conflicts}"
    assert re.search  # keep the import honest under lint


#: Every character pack, since a class card only appears in its own pack.
_CHARACTERS = ("Ironclad", "Silent", "Defect", "Necrobinder", "Regent")


def _all_card_subjects() -> set[str]:
    """Card subjects across EVERY character pack, not just Ironclad's.

    `_subjects()` above sweeps the default (Ironclad) pack, which is right for
    the per-pack assertions but would report all ~370 other-class cards as
    uncovered here.
    """
    out: set[str] = set()
    # The Fool and its pack exist from v0.113.0; the other test builds have
    # no such character to sweep.
    fool = ("Fool",) if m.sweep_game_version() == "v0.113.0" else ()
    for character in _CHARACTERS + fool:
        out |= {
            s.coverage_subject.split(":", 1)[1]
            for s in m.coverage_scenarios(character)
            if s.coverage_subject and s.coverage_subject.startswith("card:")
        }
    return out


def test_every_playable_engine_card_is_covered() -> None:
    """Every deck-holdable card the ENGINE defines gets a coverage trial.

    This is the gate the hand-transcribed lists never had. When it fails, add
    the named cards to the right `ALL_*_CARDS` list — or, if a card genuinely
    cannot be swept, widen `card_inventory.UNSWEEPABLE_RARITIES` with a reason.
    Do not delete the assertion.
    """
    from card_inventory import sweepable_cards

    covered = _all_card_subjects()  # the packs of the selected build
    missing = sorted(sweepable_cards(m.sweep_game_version()) - covered)
    assert not missing, (
        f"{len(missing)} playable engine cards have no coverage trial "
        f"(so they have never been run against the C# oracle): {missing}"
    )


def _all_card_subjects_for(game_version: str) -> set[str]:
    out: set[str] = set()
    for character in _CHARACTERS:
        out |= {
            s.coverage_subject.split(":", 1)[1]
            for s in m.coverage_scenarios(character, game_version)
            if s.coverage_subject and s.coverage_subject.startswith("card:")
        }
    return out


def test_every_playable_engine_card_is_covered_v0111() -> None:
    """The v0.111.0 coverage packs cover every card that build can hold (GH #793).

    The v0.111.0-only cards are appended to the packs at that build only, and
    the removed Scare is dropped from them.
    """
    from card_inventory import V0111_REMOVED_CARDS, sweepable_cards

    covered = _all_card_subjects_for("v0.111.0")
    missing = sorted(sweepable_cards("v0.111.0") - covered)
    assert not missing, f"{len(missing)} v0.111.0 engine cards have no coverage trial: {missing}"
    assert not (covered & V0111_REMOVED_CARDS), "a v0.111.0 pack still sweeps a removed card"


def test_v0107_coverage_never_names_a_v0111_card() -> None:
    """The default (v0.107.1) packs must not sweep a card that build lacks."""
    from card_inventory import v0111_only_cards

    # pinned to v0.107.1: under `--game-version v0.111.0` the default packs
    # legitimately carry the v0.111.0 cards.
    leaked = sorted(_all_card_subjects_for("v0.107.1") & v0111_only_cards())
    assert not leaked, f"v0.107.1 coverage sweeps v0.111.0-only cards: {leaked}"


def test_no_coverage_subject_names_a_card_the_engine_lacks() -> None:
    """The other direction: a stale/renamed id would sweep nothing.

    A trial naming a card neither engine can build is worse than an absent one
    — it looks like coverage. `CardId::PanacheCard` is exactly the shape that
    bites: its id string is `PANACHE_CARD` while its `CardSpec.name` says
    `PANACHE`, so a list transcribed from the wrong field would name a
    non-existent card and still read as covered here.
    """
    from card_inventory import engine_cards

    known = set(engine_cards())
    unknown = sorted(_all_card_subjects() - known)
    assert not unknown, f"coverage subjects name cards the engine does not define: {unknown}"
