"""Tests for the parity-fuzz scenario generator (Task 6)."""

from parity_fuzz_scenario import (
    ACT1_ENCOUNTERS,
    ALL_ENCOUNTERS,
    ALL_POTIONS,
    ALL_RELICS,
    IC_CARDS,
    SAFE_RELICS,
    Scenario,
    class_cards_for_build,
    colorless_cards_for_build,
    generate,
    generate_v2,
)


def test_generate_is_deterministic():
    a = generate(42)
    b = generate(42)
    assert a == b


def test_generate_returns_ironclad_v1():
    assert generate(7).character == "Ironclad"


def test_deck_size_in_band():
    for seed in range(20):
        s = generate(seed)
        assert 8 <= len(s.deck) <= 15


def test_relics_in_band():
    # generate() draws 0..min(3, len(SAFE_RELICS)) relics; the band was
    # widened from 2 to 3 when SAFE_RELICS grew past its v1 single entry.
    for seed in range(20):
        assert 0 <= len(generate(seed).relics) <= 3


def test_encounter_in_act1_set():
    for seed in range(20):
        assert generate(seed).encounter in ACT1_ENCOUNTERS


def test_hp_in_band():
    for seed in range(20):
        s = generate(seed)
        assert 20 <= s.hp <= s.max_hp


def test_scenario_is_frozen():
    s = generate(0)
    assert isinstance(s, Scenario)
    try:
        s.character = "Other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Scenario should be frozen")


def test_all_deck_cards_in_ic_pool():
    for seed in range(20):
        s = generate(seed)
        for c in s.deck:
            assert c in IC_CARDS


def test_all_relics_in_safe_pool():
    for seed in range(20):
        for r in generate(seed).relics:
            assert r in SAFE_RELICS


def test_generate_v2_deterministic() -> None:
    assert generate_v2(123) == generate_v2(123)


def test_generate_v2_full_pools() -> None:
    # the pools of the selected build (`--game-version`): v0.111.0 drops/adds cards
    pool = set(class_cards_for_build("Ironclad") + colorless_cards_for_build())
    encounters, relics_seen, upgraded_seen = set(), set(), False
    for seed in range(300):
        s = generate_v2(seed)
        assert 8 <= len(s.deck) <= 20
        assert 0 <= len(s.relics) <= 4
        assert 0 <= len(s.potions) <= 3
        for c in s.deck:
            base = c.removesuffix("+1")
            assert base in pool, c
            upgraded_seen = upgraded_seen or c.endswith("+1")
        assert set(s.relics) <= set(ALL_RELICS)
        assert set(s.potions) <= set(ALL_POTIONS)
        assert s.encounter in ALL_ENCOUNTERS
        encounters.add(s.encounter)
        relics_seen.update(s.relics)
    assert upgraded_seen, "25% upgrade rate must show in 300 seeds"
    assert len(encounters) > 20, "fuzz must reach beyond act1"
    assert len(relics_seen) > 30, "fuzz must actually vary relics now"


def test_builds_after_v0111_keep_its_card_set():
    # v0.113.0 builds on v0.111.0's lists, not v0.107.1's (Rust gates the
    # split on `>= V0_111_0`). Its colorless pool adds MY_SHINIES right after
    # MIND_BLAST (Rust `specs_in` / `COLORLESS_V0113`), and its sweepable set
    # adds only the v0.113.0-only cards (CardId::V0113_ONLY).
    from card_inventory import has_v0111_cards, sweepable_cards, v0113_only_cards

    assert not has_v0111_cards("v0.107.1")
    assert has_v0111_cards("v0.111.0") and has_v0111_cards("v0.113.0")
    assert class_cards_for_build("Ironclad", "v0.113.0") == class_cards_for_build(
        "Ironclad", "v0.111.0"
    )
    v0111_colorless = colorless_cards_for_build("v0.111.0")
    at = v0111_colorless.index("MIND_BLAST") + 1
    assert colorless_cards_for_build("v0.113.0") == [
        *v0111_colorless[:at],
        "MY_SHINIES",
        *v0111_colorless[at:],
    ]
    assert "MY_SHINIES" not in v0111_colorless
    assert sweepable_cards("v0.113.0") == sweepable_cards("v0.111.0") | v0113_only_cards()
    assert not sweepable_cards("v0.111.0") & v0113_only_cards()
