"""Exhaustive adapter-vocab coverage: every registry card/relic/power must
resolve from the form the game emits — zero silent soft-drops.

This guards a whole *class* of bug. The `sts2_hint` adapter translates a game
state view into `sts2_rs` search input; when it can't map a card/relic/power id
it soft-drops the entity (logged at INFO, no error, no fallback counter). That
made a 55%->0% real-world combat regression invisible: the deployed MCTS was
searching a deck with 67 `_CARD`-suffixed cards (Demon Form, Juggernaut,
Inferno, ...) and the starter relic silently deleted, because the game emits
`CARD.INFERNO` / `BURNING_BLOOD.title` while the registry stores `INFERNO_CARD`
/ `BURNING_BLOOD`. The parity suite never caught it: parity drives two ENGINES
through identical actions and diffs STATE; it never exercises this search
adapter.

The tests below reconstruct, for EVERY registry id, the exact string forms the
C# game / live bridge emit (observed in `sts2_hint.log`):

  * card  `<STEM>_CARD`      -> game emits `CARD.<STEM>`
  * card  `<STEM>_STATUS`    -> game emits `CARD.<STEM>` (same shape; the C#
           class is plain `Sloth` while the registry stores `SLOTH_STATUS`)
  * power `<STEM>` (registry drops the `_POWER` suffix) -> game leaks
           `<STEM>_POWER.title` when the power has no localization entry
  * relic `<STEM>`           -> game leaks `<STEM>.title` when unlocalized

and assert the adapter maps each back to a valid registry id. A NEW registry
entry whose emitted form doesn't resolve fails this test loudly — the CI guard
the silent runtime soft-drop lacks.
"""

from __future__ import annotations

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

from sts2_hint.adapter import (  # noqa: E402
    _registry_id_from_leaked,
    _resolve_card_id,
    _soft_translate_list,
    _translate_card,
    _translate_power,
    _translate_relic,
    drop_counts,
    reset_drop_counts,
)


def _registry(kind: str) -> list[str]:
    names = sorted(sts2_rs.list_registry_names(kind))
    if not names:
        pytest.skip(f"empty {kind} registry (build-less env)")
    return names


def test_every_registry_card_resolves_from_csharp_form() -> None:
    reg = set(_registry("card"))
    unresolved: list[tuple[str, str]] = []
    for cid in sorted(reg):
        # The game emits `CARD.<entry>`; for the suffixed registry ids the
        # entry has no suffix (`INFERNO_CARD` -> `CARD.INFERNO`,
        # `SLOTH_STATUS` -> `CARD.SLOTH`).
        #
        # `_STATUS` belongs here for the same reason `_CARD` does, and leaving
        # it out is what let the 2026-07-27 outage through: this test PASSED
        # across the whole 11-run failure because it probed `CARD.SLOTH_STATUS`
        # (which resolves trivially as candidate 0) instead of the `CARD.SLOTH`
        # the DLL actually emits. Strip at most ONE suffix — registry ids carry
        # exactly one, so stripping both off a hypothetical `X_STATUS_CARD`
        # would fabricate a form the game never emits.
        stem = cid
        for suffix in ("_CARD", "_STATUS"):
            if cid.endswith(suffix):
                stem = cid[: -len(suffix)]
                break
        try:
            _cls, resolved = _resolve_card_id(f"CARD.{stem}", "$")
        except Exception as e:  # collect, report all at once
            unresolved.append((cid, f"raise {e}"))
            continue
        if resolved not in reg:
            unresolved.append((cid, f"-> {resolved!r} (not a registry id)"))
    assert not unresolved, (
        f"{len(unresolved)}/{len(reg)} cards do not resolve from their "
        f"`CARD.<entry>` game form: {unresolved[:25]}"
    )


def _csharp_form(cid: str) -> str:
    """The `CARD.<entry>` id the game emits for registry id *cid*."""
    for suffix in ("_CARD", "_STATUS"):
        if cid.endswith(suffix):
            return f"CARD.{cid[: -len(suffix)]}"
    return f"CARD.{cid}"


def test_every_engine_card_resolves_to_itself_from_csharp_form() -> None:
    """Every build's cards, v0.113.0's included, resolve -- to themselves.

    The adapter checked candidates against the PPO vocab ("card", frozen at
    v0.111.0), so My Shinies -- a v0.113.0 card a Colorless Potion offers --
    was soft-dropped from every search state, and an in-fight offer of it
    could not be searched ("card id 'MY_SHINIES' doesn't map to any of ...").
    """
    try:
        engine = sorted(sts2_rs.list_registry_names("card_any"))
    except ValueError:
        pytest.skip("engine predates the card_any registry kind")
    wrong = []
    for cid in engine:
        try:
            _cls, resolved = _resolve_card_id(_csharp_form(cid), "$")
        except Exception as e:
            wrong.append((cid, f"raise {e}"))
            continue
        if resolved != cid:
            wrong.append((cid, resolved))
    assert not wrong, f"{len(wrong)}/{len(engine)} engine cards mis-resolve: {wrong[:25]}"
    assert {"MY_SHINIES", "DEEP_STRETCH"} <= set(engine)


def test_vocab_cards_resolve_as_they_did_against_the_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widening the known set changes nothing for a card that resolved before."""
    from sts2_hint import adapter

    vocab = frozenset(_registry("card"))
    forms = [_csharp_form(cid) for cid in vocab] + [f"CARD.{cid}" for cid in vocab]
    now = {f: _resolve_card_id(f, "$") for f in forms}
    monkeypatch.setattr(adapter, "_registry_card_names", lambda: vocab)
    before = {f: _resolve_card_id(f, "$") for f in forms}
    assert now == before


def test_every_registry_power_recoverable_from_its_lockey() -> None:
    """Exhaustive safety-net: every registry power must be recoverable from a
    loc-key of its own id (`<id>.title`). This is the fallback that makes ANY
    leaked loc-key resolvable; asserting it over the whole registry proves no
    entry can silently soft-drop. (We use the id's OWN loc-key rather than a
    reconstructed `<id>_POWER.title` because a few registry ids already carry
    the `_POWER` suffix — `MANGLE_POWER`, `GRAPPLE_POWER` — so appending it
    again would fabricate a form the game never emits. The exact `_POWER.title`
    emission that broke in the wild is locked by the regression test below.)"""
    reg = set(_registry("power"))
    bad: list[tuple[str, object]] = []
    for pid in sorted(reg):
        got = _registry_id_from_leaked(f"{pid}.title", "power", ("_POWER",))
        if got != pid:
            bad.append((pid, got))
    assert not bad, f"{len(bad)}/{len(reg)} powers not recoverable from `<id>.title`: {bad[:25]}"


def test_leaked_power_emission_patterns_resolve() -> None:
    """Regression: the exact `<STEM>_POWER.title` loc-keys the game leaked for
    potion/relic-granted powers (registry stores the STEM, no `_POWER`)."""
    for leaked, want in [
        ("TAINTED_POWER.title", "TAINTED"),
        ("FLEX_POTION_POWER.title", "FLEX_POTION"),
        ("SPEED_POTION_POWER.title", "SPEED_POTION"),
        ("SHACKLING_POTION_POWER.title", "SHACKLING_POTION"),
        ("REPTILE_TRINKET_POWER.title", "REPTILE_TRINKET"),
    ]:
        got = _registry_id_from_leaked(leaked, "power", ("_POWER",))
        assert got == want, f"{leaked} -> {got!r}, want {want!r}"


def test_every_registry_relic_resolves_from_leaked_lockey() -> None:
    reg = set(_registry("relic"))
    bad: list[tuple[str, object]] = []
    for rid in sorted(reg):
        got = _registry_id_from_leaked(f"{rid}.title", "relic")
        if got != rid:
            bad.append((rid, got))
    assert not bad, (
        f"{len(bad)}/{len(reg)} relics do not resolve from their leaked "
        f"`<id>.title` form: {bad[:25]}"
    )


def test_translate_relic_and_power_accept_leaked_lockeys() -> None:
    """End-to-end: the translate entry points (not just the id resolver) accept
    a leaked loc-key and emit a registry id — the path recommend_all actually
    takes."""
    relic = _translate_relic({"name": "BURNING_BLOOD.title"}, "$")
    assert relic["id"] == "BURNING_BLOOD"
    power = _translate_power({"name": "TAINTED_POWER.title", "amount": 2}, "$")
    assert power["name"] == "TAINTED"
    assert power["amount"] == 2


def test_known_regression_cards_resolve() -> None:
    """The exact cards the real-world search was silently dropping."""
    for stem, want in [
        ("INFERNO", "INFERNO_CARD"),
        ("DEMON_FORM", "DEMON_FORM_CARD"),
        ("JUGGERNAUT", "JUGGERNAUT_CARD"),
        ("HELLRAISER", "HELLRAISER_CARD"),
        ("STAMPEDE", "STAMPEDE_CARD"),
        ("AGGRESSION", "AGGRESSION_CARD"),
        ("JUGGLING", "JUGGLING_CARD"),
        # The `_STATUS` twin of the same split: the Knowledge Demon's Sloth.
        # Unlike the `_CARD` family this one did not soft-drop — it RAISED
        # `UnsupportedStateError` on the boss's card_select and killed the run
        # outright, 11 of 60 seeds in the 2026-07-27 rung-3.5 sweep.
        ("SLOTH", "SLOTH_STATUS"),
    ]:
        _cls, resolved = _resolve_card_id(f"CARD.{stem}", "$")
        assert resolved == want, f"CARD.{stem} -> {resolved!r}, want {want!r}"
    # And a plain card still resolves to its bare id (no regression).
    assert _resolve_card_id("CARD.STRIKE_IRONCLAD", "$")[1] == "STRIKE_IRONCLAD"


def test_translate_card_carries_base_stats_from_view_stats() -> None:
    """An upgraded card's baked base stats must reach the wire.

    The C# headless never emits an ``upgraded`` flag; ``stats.damage`` /
    ``stats.block`` are the ONLY signal that a card is stronger than its
    unupgraded registry default (upgraded Strike deals 9, not 6). Dropping
    them made ``state_from_dict`` rebuild every upgraded card at its unupgraded
    value, so the combat search planned the whole deck weaker than it is
    (2026-07-15 csview recon audit)."""
    up = _translate_card(
        {"id": "CARD.STRIKE_IRONCLAD", "index": 0, "stats": {"damage": 9, "block": 0}},
        "$.deck.hand[0]",
    )
    assert up["id"] == "STRIKE_IRONCLAD"
    assert up["base_damage"] == 9
    assert up["base_block"] == 0

    # A Defend carries block, not damage.
    dfd = _translate_card(
        {"id": "CARD.DEFEND_IRONCLAD", "index": 1, "stats": {"block": 8}},
        "$.deck.hand[1]",
    )
    assert dfd["base_block"] == 8

    # No stats block -> no base_* keys (parser falls back to the registry).
    plain = _translate_card({"id": "CARD.STRIKE_IRONCLAD", "index": 2}, "$")
    assert "base_damage" not in plain
    assert "base_block" not in plain


def test_soft_drop_telemetry_counts_by_kind() -> None:
    """A soft-drop must be COUNTED, not just silently logged — the canary that
    would have made the 55%->0% regression visible in the eval report."""
    reset_drop_counts()
    assert drop_counts() == {}
    # A resolvable loc-key relic (kept) + two unmappable display-name relics.
    # (Fabricated names: with the 2026-07-17 full registry oracle, every real
    # relic display title / loc-key now resolves.)
    out = _soft_translate_list(
        [
            {"name": "BURNING_BLOOD.title"},
            {"name": "Not A Real Relic"},
            {"name": "Also Fake"},
        ],
        _translate_relic,
        "$.player.relics",
    )
    assert len(out) == 1  # only the loc-key relic resolved
    counts = drop_counts()
    assert counts.get("relic") == 2
    assert counts.get("total") == 2
    reset_drop_counts()
    assert drop_counts() == {}


# Relics the C# backend leaks as raw loc-keys (``TOOLBOX.title`` etc.,
# observed 2026-07-15 cellC attribution sweep). Until 2026-07-17 these were
# missing from ``sts2_rs.list_registry_names("relic")`` and needed a
# hard-coded allowlist in the adapter; the engine's registry oracle now
# covers all RelicKinds (297 entries), so they resolve through the normal
# leaked-lockey path. These tests pin that path.
_LEAKED_LOCKEY_RELIC_STEMS = (
    "CIRCLET",
    "JEWELRY_BOX",
    "NEOWS_TORMENT",
    "RADIANT_PEARL",
    "TOOLBOX",
)


def test_relics_missing_from_stale_registry_resolve_from_leaked_lockey() -> None:
    """Regression: 5 relics observed dropped from reconstructed real-game
    states (2026-07-15 cellC attribution sweep, Ironclad A0 seeds 1000-1019)
    despite being fully implemented in the engine. They arrive as raw
    loc-keys (``TOOLBOX.title``) and must resolve via the registry."""
    reg = set(sts2_rs.list_registry_names("relic"))
    for stem in _LEAKED_LOCKEY_RELIC_STEMS:
        assert stem in reg, f"{stem} missing from registry — oracle regressed?"
        relic = _translate_relic({"name": f"{stem}.title"}, "$")
        assert relic["id"] == stem, f"{stem}.title -> {relic['id']!r}, want {stem!r}"


def test_relics_missing_from_stale_registry_accepted_by_state_from_dict() -> None:
    """End-to-end confirmation that the mapping in the test above is
    CORRECT, not just registry-shaped: sts2_rs's actual wire-dict parser
    accepts every one of these ids without raising."""
    for stem in _LEAKED_LOCKEY_RELIC_STEMS:
        d = {
            "wire_version": 1,
            "phase": "PlayerPlayPhase",
            "round": 1,
            "side": "Player",
            "is_elite": False,
            "player": {
                "class": "ironclad",
                "hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "max_energy": 3,
                "energy_regen_per_turn": 3,
                "relics": [{"id": stem, "counter": None}],
                "powers": [],
                "potions": [],
            },
            "deck": {
                "draw_pile": [],
                "discard_pile": [],
                "hand": [],
                "exhaust_pile": [],
                "master_deck": None,
            },
            "encounter": [],
            "turn": 1,
            "combat_id": 0,
            "rng": {"seed": 0, "ascension": 0},
            "active_attack": None,
            "pending_choice": None,
            "pending_play": None,
        }
        sts2_rs.state_from_dict(d)  # must not raise


def _hand_state(card: dict) -> dict:
    """Minimal wire state carrying one card in hand."""
    return {
        "wire_version": 1,
        "phase": "PlayerPlayPhase",
        "round": 1,
        "side": "Player",
        "is_elite": False,
        "player": {
            "class": "ironclad",
            "hp": 70,
            "max_hp": 75,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "energy_regen_per_turn": 3,
            "relics": [],
            "powers": [],
            "potions": [],
        },
        "deck": {
            "draw_pile": [],
            "discard_pile": [],
            "hand": [card],
            "exhaust_pile": [],
            "master_deck": None,
        },
        "encounter": [],
        "turn": 1,
        "combat_id": 0,
        "rng": {"seed": 0, "ascension": 0},
        "active_attack": None,
        "pending_choice": None,
        "pending_play": None,
    }


def test_translate_card_forwards_the_per_instance_card_type() -> None:
    """MadScience's TinkerTime variant is per-INSTANCE, so it rides the wire.

    ``MadScience.Type`` resolves to Attack/Skill/Power on the individual card
    and cannot be recovered from the registry spec, which is the
    default-constructed Attack build. ``_translate_card`` read ``type`` for
    status/curse discrimination and then dropped it, so every live Power-mode
    MadScience rebuilt as an Attack -- wrong damage, wrong cost, invisible to
    every card-type-gated surface (GH #353).

    ``ParsedCard`` lowercases the value itself and tolerates unknown
    spellings, so it is forwarded verbatim.
    """
    for wire_type in ("Attack", "Skill", "Power"):
        spec = _translate_card(
            {"id": "CARD.MAD_SCIENCE", "index": 0, "type": wire_type}, "$.deck.hand[0]"
        )
        assert spec["id"] == "MAD_SCIENCE"
        assert spec["type"] == wire_type
        sts2_rs.state_from_dict(_hand_state(spec))  # must not raise

    # No key at all when the producer does not ship one -- an absent type and
    # a type of "" must not be conflated with an explicit "Attack".
    bare = _translate_card({"id": "CARD.MAD_SCIENCE", "index": 0}, "$.deck.hand[0]")
    assert "type" not in bare
    assert "type" not in _translate_card(
        {"id": "CARD.MAD_SCIENCE", "index": 0, "type": ""}, "$.deck.hand[0]"
    )


def test_translate_card_forwards_the_tinker_time_rider() -> None:
    """The rider half of GH #353, ready for the mod to start emitting it.

    ``MadScience``'s Power arm applies Curious / Improvement / Expertise, and
    the rider is a public ``[SavedProperty]`` on the C# side. No producer puts
    it on the wire yet, so this pins the adapter leg so that lands as a
    mod-only change; ``TinkerTimeRider::from_wire_name`` already accepts the
    enum names.
    """
    spec = _translate_card(
        {"id": "CARD.MAD_SCIENCE", "index": 0, "type": "Power", "tinker_time_rider": "Curious"},
        "$.deck.hand[0]",
    )
    assert spec["tinker_time_rider"] == "Curious"
    sts2_rs.state_from_dict(_hand_state(spec))  # must not raise

    assert "tinker_time_rider" not in _translate_card(
        {"id": "CARD.MAD_SCIENCE", "index": 0, "type": "Power"}, "$.deck.hand[0]"
    )
