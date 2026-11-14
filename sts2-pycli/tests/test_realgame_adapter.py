"""Offline tests for :mod:`realgame_adapter`.

Fixtures under ``tests/fixtures/realgame_adapter/`` are real v2 SimBridge
``combat_action`` captures copied verbatim from
``logs/realgame/20260711T000030Z.jsonl`` (see each fixture's
``_provenance`` field) -- not hand-crafted -- so these tests exercise the
adapter against the actual shape the live game emits, including a
single-enemy fight, a fight with a player debuff (``Shrink``) and an
enemy buff (``Vulnerable``), and a 3-enemy fight (multi-target).

Two tiers:

* Pure-dict translation (:func:`realgame_adapter.to_headless_combat_play`)
  -- always runs, no ``sts2_rs`` dependency.
* Full pipeline (``to_headless_combat_play`` ->
  ``sts2_hint.adapter.translate`` -> ``sts2_rs.state_from_dict``) --
  skipped with a clear reason if the ``sts2_rs`` PyO3 extension isn't
  importable in this venv (``uv run python -c "import sts2_rs"``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from realgame_adapter import character_display_name, to_headless_combat_play

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "realgame_adapter"

try:
    import sts2_rs  # type: ignore[import-not-found]

    _SKIP_ENGINE_REASON = None
except ImportError as _e:
    sts2_rs = None  # type: ignore[assignment]
    _SKIP_ENGINE_REASON = f"sts2_rs PyO3 extension not importable in this venv: {_e}"


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


FIXTURE_NAMES = [
    "single_enemy_ironclad.json",
    "player_power_ironclad.json",
    "multi_enemy_ironclad.json",
    "eye_with_teeth_ironclad.json",
]


# ---------------------------------------------------------------------------
# Pure-dict translation -- no sts2_rs dependency.
# ---------------------------------------------------------------------------


class TestToHeadlessCombatPlayShape:
    """Structural invariants of the translated dict, independent of sts2_rs."""

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_envelope(self, fixture_name: str) -> None:
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        assert out["type"] == "decision"
        assert out["decision"] == "combat_play"
        assert out["context"]["room_type"] == "Combat"
        assert out["round"] == bridge_state["round"]

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_energy_matches_bridge_player(self, fixture_name: str) -> None:
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        assert out["energy"] == bridge_state["player"]["energy"]
        assert out["max_energy"] == bridge_state["player"]["max_energy"]

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_hand_size_and_order_matches_bridge(self, fixture_name: str) -> None:
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_hand = bridge_state["hand"]
        assert len(out["hand"]) == len(bridge_hand)
        for i, (bridge_card, headless_card) in enumerate(
            zip(bridge_hand, out["hand"], strict=True)
        ):
            assert headless_card["id"] == bridge_card["id"]
            assert headless_card["index"] == i, (
                "hand card 'index' must equal its position so HintTag.stable_idx "
                "round-trips as the bridge play verb's card_index"
            )

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_pile_sizes_match_bridge_pile_objects(self, fixture_name: str) -> None:
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        for bridge_key, headless_key in (
            ("draw_pile", "draw_pile"),
            ("discard_pile", "discard_pile"),
            ("exhaust_pile", "exhaust_pile"),
        ):
            bridge_pile = bridge_state.get(bridge_key) or {}
            assert len(out[headless_key]) == bridge_pile.get("count", 0)
            assert len(out[headless_key]) == len(bridge_pile.get("cards") or [])

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_draw_pile_order_preserved(self, fixture_name: str) -> None:
        """Index 0 of both the bridge and headless draw piles must be the
        next card drawn -- no reversal (see module docstring's "Draw-pile
        order" note)."""
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_cards = (bridge_state.get("draw_pile") or {}).get("cards") or []
        for bridge_card, headless_card in zip(bridge_cards, out["draw_pile"], strict=True):
            assert headless_card["id"] == bridge_card["id"]

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_enemy_count_and_index_matches_bridge_position(self, fixture_name: str) -> None:
        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_enemies = bridge_state["enemies"]
        assert len(out["enemies"]) == len(bridge_enemies)
        for i, (bridge_enemy, headless_enemy) in enumerate(
            zip(bridge_enemies, out["enemies"], strict=True)
        ):
            assert headless_enemy["name"] == bridge_enemy["name"]
            assert headless_enemy["hp"] == bridge_enemy["hp"]
            assert headless_enemy["index"] == i, (
                "enemy 'index' must equal its position -- sts2_hint.display.format_target "
                "and the bridge's own target_index both address enemies by position"
            )

    def test_player_power_translated_with_both_id_and_name(self) -> None:
        fixture = _load_fixture("player_power_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_powers = bridge_state["player"]["powers"]
        assert bridge_powers, "fixture should carry at least one player power"
        assert len(out["player_powers"]) == len(bridge_powers)
        first = out["player_powers"][0]
        assert first["id"] == bridge_powers[0]["id"]
        assert first["name"] == bridge_powers[0]["name"]
        assert first["amount"] == bridge_powers[0]["amount"]

    def test_enemy_power_translated(self) -> None:
        fixture = _load_fixture("player_power_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_enemy_powers = bridge_state["enemies"][0].get("powers") or []
        headless_enemy_powers = out["enemies"][0]["powers"]
        assert len(headless_enemy_powers) == len(bridge_enemy_powers)

    def test_relic_translated_by_title(self) -> None:
        fixture = _load_fixture("single_enemy_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        bridge_relics = bridge_state["run_state"]["relics"]
        assert bridge_relics, "fixture should carry at least one relic"
        assert out["player"]["relics"][0]["name"] == bridge_relics[0]["name"]

    def test_relic_counter_and_melt_flags_reach_headless_shape(self) -> None:
        """Counter must land TOP-LEVEL (what sts2_hint reads), not under vars.

        The old shape nested the live counter under ``vars`` and dropped
        ``is_wax``/``is_melted``, so every counter relic reconstructed as
        counter=None and melted ToyBox relics stayed hook-active — on the
        live path only.
        """
        fixture = _load_fixture("single_enemy_ironclad.json")
        bridge_state = fixture["state"]
        bridge_state["run_state"]["relics"] = [
            {"name": "Nunchaku", "counter": 7, "stack_count": 1},
            {"name": "Strike Dummy", "is_wax": True, "is_melted": True},
            {"name": "Burning Blood"},
        ]
        out = to_headless_combat_play(bridge_state)
        nunchaku, dummy, blood = out["player"]["relics"]

        assert nunchaku["counter"] == 7
        assert "counter" not in (nunchaku.get("vars") or {})
        assert dummy.get("is_wax") is True
        assert dummy.get("is_melted") is True
        assert "counter" not in blood
        assert "is_wax" not in blood and "is_melted" not in blood

    def test_empty_potion_slots_pass_through_as_none_name(self) -> None:
        fixture = _load_fixture("single_enemy_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        assert out["player"]["potions"] == bridge_state["potions"]
        for slot in out["player"]["potions"]:
            if slot.get("empty"):
                assert slot.get("name") is None

    def test_potion_slot_count_reaches_the_player_as_potion_slots(self) -> None:
        """GH #893: the mod's ``run_state.potion_slot_count`` (C#
        ``Player.MaxPotionCount``) is the belt capacity; it rides to the
        wire as ``player.potion_slots``."""
        fixture = _load_fixture("single_enemy_ironclad.json")
        bridge_state = fixture["state"]
        bridge_state["run_state"]["potion_slot_count"] = 2
        out = to_headless_combat_play(bridge_state)
        assert out["player"]["potion_slots"] == 2

        del bridge_state["run_state"]["potion_slot_count"]
        assert "potion_slots" not in to_headless_combat_play(bridge_state)["player"]

    def test_the_act_and_a_boss_room_reach_the_wire(self) -> None:
        """GH #989: the mod's top-level ``act`` (1-based) and a ``Boss`` room
        ride to the wire as ``act_index`` / ``is_boss``, off their defaults."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        bridge_state.update(act=3, room_type="Boss")
        wire = translate(to_headless_combat_play(bridge_state))
        assert (wire["act_index"], wire["is_boss"]) == (3, True)

        bridge_state.update(act=1, room_type="Monster")
        wire = translate(to_headless_combat_play(bridge_state))
        assert "act_index" not in wire and "is_boss" not in wire

    def test_the_run_floor_reaches_the_wire(self) -> None:
        """GH #1337: the mod's top-level ``floor`` is C# ``RunState.TotalFloor``,
        which v0.113.0's TheTower multiplies its damage by. It rides to the
        wire as ``total_floor``; without it the engine reads floor 0."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        bridge_state["floor"] = 23
        wire = translate(to_headless_combat_play(bridge_state))
        assert wire["total_floor"] == 23

        del bridge_state["floor"]
        assert "total_floor" not in translate(to_headless_combat_play(bridge_state))

    def test_skittish_latch_and_a_teammates_plays_reach_the_wire(self) -> None:
        """GH #989: Skittish's once-a-turn latch (sent with the power's vars)
        becomes the engine's ``counter``, and a teammate's play buckets ride
        with their seat (their Ripple Basin counts Attacks)."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        bridge_state["enemies"][0]["powers"] = [
            {
                "id": "SKITTISH_POWER",
                "name": "Skittish",
                "amount": 9,
                "vars": {"HasGainedBlockThisTurn": 1},
            },
        ]
        bridge_state["allies"] = [
            {
                "character": "IRONCLAD",
                "hp": 50,
                "max_hp": 70,
                "block": 0,
                "energy": 3,
                "max_energy": 3,
                "powers": [],
                "hand": [],
                "cards_played_this_turn": [2, 1, 0, 0, 0],
            },
        ]
        wire = translate(to_headless_combat_play(bridge_state))
        skittish = wire["encounter"][0]["powers"][0]
        assert (skittish["name"], skittish["counter"]) == ("SKITTISH", 1)
        assert wire["allies"][0]["cards_played_this_turn"] == [2, 1, 0, 0, 0]

    def test_a_powers_applier_reaches_the_wire_as_a_seat(self) -> None:
        """GH #1004: the mod names who applied a power by net id; the wire
        carries the seat (0 us, ``i + 1`` for ``allies[i]``), which Flanking,
        Knockdown and Tag Team key on. Solo, or for an unknown id, nothing."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        bridge_state["player"]["net_id"] = "1"
        bridge_state["enemies"][0]["powers"] = [
            {"id": "FLANKING_POWER", "name": "Flanking", "amount": 2, "applier": "1000"},
            {"id": "FLANKING_POWER", "name": "Flanking", "amount": 2, "applier": "1"},
            {"id": "KNOCKDOWN_POWER", "name": "Knockdown", "amount": 2, "applier": "77"},
        ]
        solo = translate(to_headless_combat_play(bridge_state))
        assert all("applier_seat" not in p for p in solo["encounter"][0]["powers"])

        bridge_state["allies"] = [
            {"character": "SILENT", "hp": 50, "max_hp": 70, "block": 0, "net_id": "1000"},
        ]
        wire = translate(to_headless_combat_play(bridge_state))
        seats = [p.get("applier_seat") for p in wire["encounter"][0]["powers"]]
        assert seats == [1, 0, None]
        assert all("applier" not in p for p in wire["encounter"][0]["powers"])

    def test_imitation_learnings_seat_is_the_teammate_it_copies(self) -> None:
        """GH #992: Imitation Learning's ``applier`` is its own holder; the
        seat the engine needs is ``player_target``, the teammate it copies.
        Without one (an older mod) the seat is left to the engine."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        bridge_state["player"]["net_id"] = "1"
        il = {"id": "IMITATION_LEARNING_POWER", "name": "Imitation Learning", "amount": 2}
        bridge_state["player"]["powers"] = [{**il, "applier": "1", "player_target": "1000"}]
        bridge_state["allies"] = [
            {
                "character": "DEFECT",
                "hp": 50,
                "max_hp": 70,
                "block": 0,
                "net_id": "1000",
                "hand": [],
                "powers": [{**il, "applier": "1000"}],
            },
        ]
        wire = translate(to_headless_combat_play(bridge_state))
        (ours,) = wire["player"]["powers"]
        assert (ours["name"], ours.get("applier_seat")) == ("IMITATION_LEARNING", 1)
        (theirs,) = wire["allies"][0]["powers"]
        assert "applier_seat" not in theirs
        assert "applier" not in theirs

    def test_a_teammates_osty_reaches_their_seat(self) -> None:
        """GH #1017: a Necrobinder teammate's ``pets`` (newer mods) become
        their seat's ``osty``, in our shape, and the engine builds it into
        their seat. Without ``pets`` (an older mod) no Osty is guessed."""
        from sts2_hint.adapter import translate

        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        mate = {
            "character": "NECROBINDER",
            "hp": 50,
            "max_hp": 70,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "hand": [],
        }
        bridge_state["allies"] = [dict(mate)]
        old = translate(to_headless_combat_play(bridge_state))
        assert "osty" not in old["allies"][0]

        osty = {"id": "OSTY", "name": "Osty", "is_alive": True, "hp": 4, "max_hp": 9, "block": 0}
        byrdpip = {"id": "BYRDPIP", "name": "Byrdpip", "is_alive": True, "hp": 1, "max_hp": 1}
        bridge_state["allies"] = [{**mate, "pets": [byrdpip, osty]}]
        wire = translate(to_headless_combat_play(bridge_state))
        assert wire["allies"][0]["osty"] == {"alive": True, "hp": 4, "max_hp": 9, "block": 0}
        assert "osty" not in wire["player"]
        if sts2_rs is not None:
            again = sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))
            assert again["allies"][0]["osty"] == wire["allies"][0]["osty"]

    def test_rejects_non_combat_action_state(self) -> None:
        with pytest.raises(ValueError, match="combat_action"):
            to_headless_combat_play({"type": "map_select"})

    def test_multi_enemy_intents_include_attack_damage(self) -> None:
        fixture = _load_fixture("multi_enemy_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)

        # First enemy in this fixture has a SingleAttackIntent.
        intents = out["enemies"][0]["intents"]
        assert intents
        attack = intents[0]
        assert attack["type"] == "Attack"
        assert attack["damage"] == bridge_state["enemies"][0]["intents"][0]["damage"]


class TestFieldsTheBridgeCaughtUpOn:
    """``play_pile`` / power ``vars`` / power ``display_amount``.

    All three reached the ``Sts2Headless`` wire in the v11/v12 mirror-fidelity
    fixes and the bridge was left behind on them, so the live arm ran the same
    mirror against a wire that still withheld the card mid-play and the
    CrimsonMantle/WitheringPresence counters.

    Every fixture under ``fixtures/realgame_adapter/`` is a capture from a mod
    build that PREDATES the emitters, which is exactly the older half of the
    contract: absent must behave as it did before. The doctored cases below
    cover the present half.
    """

    def test_an_absent_play_pile_is_an_empty_pile_not_a_failure(self) -> None:
        for name in FIXTURE_NAMES:
            out = to_headless_combat_play(_load_fixture(name)["state"])
            assert out["play_pile"] == [], f"{name}: pre-emitter capture must translate cleanly"

    def test_the_card_mid_play_travels(self) -> None:
        """A card parked in ``PileType.Play`` is in no other pile, so if this
        does not travel the reconstruction simply loses it."""
        bridge_state = _load_fixture("single_enemy_ironclad.json")["state"]
        in_flight = ((bridge_state.get("draw_pile") or {}).get("cards") or [])[0]
        bridge_state["play_pile"] = {"type": "Play", "count": 1, "cards": [in_flight]}

        out = to_headless_combat_play(bridge_state)
        assert [c["id"] for c in out["play_pile"]] == [in_flight["id"]]

    def test_power_counters_the_bridge_now_sends_are_forwarded(self) -> None:
        """Forwarded verbatim — ``sts2_hint.adapter`` owns their meaning."""
        bridge_state = _load_fixture("player_power_ironclad.json")["state"]
        bridge_state["player"]["powers"][0]["vars"] = {"SelfDamage": 3}
        bridge_state["player"]["powers"][0]["display_amount"] = 7

        power = to_headless_combat_play(bridge_state)["player_powers"][0]
        assert power["vars"] == {"SelfDamage": 3}
        assert power["display_amount"] == 7

    def test_a_pre_emitter_power_claims_nothing_rather_than_zero(self) -> None:
        """``None``, not ``0``/``{}``: the downstream table skips a missing var
        and leaves the engine's own value alone, but would happily overwrite it
        with a fabricated zero."""
        bridge_state = _load_fixture("player_power_ironclad.json")["state"]
        assert "vars" not in bridge_state["player"]["powers"][0]

        power = to_headless_combat_play(bridge_state)["player_powers"][0]
        assert power["vars"] is None
        assert power["display_amount"] is None


class TestLayeredCostPassthrough:
    """``base_cost`` / ``persistent_cost`` / ``played_cost`` / ``stable_cost``.

    The bridge has emitted the full layered cost model since the v15/v20
    mirror-fidelity catch-ups (``SimStateExtractor.SerializeCard``), but
    ``realgame_adapter._translate_card`` dropped everything except the
    resolved ``cost`` -- so the live reconstruction ran cost_override-only,
    baked transient discounts in permanently, and lost every turn-boundary /
    when-played cost reset. On the first persistent-CombatMirror live sweep
    171 of 451 pile-only desync records differed ONLY in the cost element of
    the pile card key, dominated by dynamic-cost cards (Cascade, Stomp,
    Conflagration, discounted Strikes) -- ``logs/diag/mirror_live1.jsonl``.

    The doctored card below is a Stomp mid-turn: printed 3, shaved to 1 by
    two Attacks played this turn. The bridge (like the headless runner)
    reports ``base_cost=1`` (local modifiers applied,
    ``GetWithModifiers(Local)``) and 3 for all three post-edge re-folds.
    """

    def _bridge_state_with_dynamic_cost_card(self) -> dict[str, Any]:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        stomp = dict(state["hand"][0])
        stomp.update(
            {
                "id": "STOMP",
                "name": "Stomp",
                "type": "Attack",
                "cost": 1,
                "effective_cost": 1,
                "base_cost": 1,
                "persistent_cost": 3,
                "played_cost": 3,
                "stable_cost": 3,
                "costs_x": False,
            }
        )
        state["hand"][0] = stomp
        # The same card sitting in the draw pile (the corpus class is mostly
        # pile cards, and every pile goes through the same translator).
        draw_cards = state["draw_pile"]["cards"]
        cascade = dict(draw_cards[0])
        cascade.update(
            {
                "id": "CASCADE",
                "name": "Cascade",
                "type": "Attack",
                "cost": 2,
                "effective_cost": 2,
                "base_cost": 2,
                "persistent_cost": 3,
                "played_cost": 3,
                "stable_cost": 3,
                "costs_x": False,
            }
        )
        draw_cards[0] = cascade
        return state

    def test_dynamic_cost_fields_survive_to_headless_shape(self) -> None:
        out = to_headless_combat_play(self._bridge_state_with_dynamic_cost_card())
        hand0 = out["hand"][0]
        assert hand0["cost"] == 1
        assert hand0["base_cost"] == 1
        assert hand0["persistent_cost"] == 3
        assert hand0["played_cost"] == 3
        assert hand0["stable_cost"] == 3
        draw0 = out["draw_pile"][0]
        assert draw0["base_cost"] == 2
        assert draw0["persistent_cost"] == 3
        assert draw0["played_cost"] == 3
        assert draw0["stable_cost"] == 3

    def test_dynamic_cost_fields_reach_wire_spec(self) -> None:
        """End-to-end into the translated wire: the exact keys
        ``sts2_hint.adapter._translate_card`` forwards to
        ``sts2_rs.state_from_dict`` (its CardSpec layered-cost slots)."""
        from sts2_hint import adapter

        out = to_headless_combat_play(self._bridge_state_with_dynamic_cost_card())
        wire = adapter.translate(out, character="Ironclad")
        hand0 = wire["deck"]["hand"][0]
        assert hand0["id"] == "STOMP"
        assert hand0["cost_override"] == 1
        assert hand0["base_cost"] == 1
        assert hand0["persistent_cost"] == 3
        assert hand0["played_cost"] == 3
        assert hand0["stable_cost"] == 3
        draw0 = wire["deck"]["draw_pile"][0]
        assert draw0["id"] == "CASCADE"
        assert draw0["cost_override"] == 2
        assert draw0["base_cost"] == 2
        assert draw0["persistent_cost"] == 3

    def test_x_cost_sentinel_travels(self) -> None:
        """``base_cost == -1`` is the engine's X-cost sentinel (headless
        ``BaseEnergyCost`` and bridge ``SafeBaseCost`` both emit it) and must
        not be confused with an absent field."""
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        state["hand"][0]["base_cost"] = -1
        state["hand"][0]["costs_x"] = True
        out = to_headless_combat_play(state)
        assert out["hand"][0]["base_cost"] == -1

    def test_pre_field_mod_build_grows_no_cost_keys(self) -> None:
        """Absent (older mod) and null (no local modifiers) must both stay
        OFF the wire -- the downstream adapter treats an absent key as
        "no claim" but would decode a forwarded ``None`` as a value."""
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        for key in ("base_cost", "persistent_cost", "played_cost", "stable_cost"):
            state["hand"][1].pop(key, None)  # simulate a pre-field mod build
        state["hand"][0]["base_cost"] = 1
        state["hand"][0]["persistent_cost"] = None  # bridge: "cannot differ"
        out = to_headless_combat_play(state)
        assert out["hand"][0]["base_cost"] == 1
        assert "persistent_cost" not in out["hand"][0]
        assert "base_cost" not in out["hand"][1]
        assert "stable_cost" not in out["hand"][1]


class TestGeneratedFreeStarCard:
    """``star_cost`` / ``persistent_star_cost``: the star half of a free card.

    C# ``SetToFreeThisTurn`` (Splash, Discovery, Infernal Blade, the
    generative potions) zeroes the STAR cost as well as the energy cost
    (``CardModel.cs:1267-1271``). Splash on a Silent offers the other
    characters' Attacks, Regent's star cards among them, so its pick is often a
    Falling Star or Gamma Blast that the game lets the Silent play for nothing.
    The bridge only ever shipped the energy half, so the rebuild charged the
    printed star cost, a star-less character could never pay it, and the
    advisor recommended End Turn over a free attack (the user's hands-free
    Silent run, 2026-10-31).

    The doctored card is that pick: a Falling Star (printed 1 energy + 2 stars)
    in hand after Splash, at 0 energy, on a board with no stars.
    """

    def _bridge_state(self, **star: Any) -> dict[str, Any]:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        card = dict(state["hand"][0])
        card.update(
            {
                "id": "FALLING_STAR",
                "name": "Falling Star",
                "type": "Attack",
                "target_type": "AnyEnemy",
                "cost": 0,
                "effective_cost": 0,
                "base_cost": 0,
                "persistent_cost": 1,
                "played_cost": 1,
                "stable_cost": 1,
                "costs_x": False,
                "playable": True,
                "valid_target_indices": [0],
                **star,
            }
        )
        state["hand"].append(card)
        state["player"]["energy"] = 0
        return state

    def test_star_cost_fields_survive_to_the_wire(self) -> None:
        from sts2_hint import adapter

        out = to_headless_combat_play(self._bridge_state(star_cost=0, persistent_star_cost=2))
        assert out["hand"][-1]["star_cost"] == 0
        assert out["hand"][-1]["persistent_star_cost"] == 2
        wire = adapter.translate(out, character="Ironclad")
        assert wire["deck"]["hand"][-1]["star_cost"] == 0
        assert wire["deck"]["hand"][-1]["persistent_star_cost"] == 2

    def test_a_card_without_a_star_cost_grows_no_star_keys(self) -> None:
        """The bridge sends null for every card but the Regent star cards; it
        must stay off the wire, where a forwarded ``None`` is a schema error."""
        out = to_headless_combat_play(self._bridge_state(star_cost=None, persistent_star_cost=None))
        for card in out["hand"]:
            assert "star_cost" not in card
            assert "persistent_star_cost" not in card

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_the_free_star_card_is_playable_in_the_rebuild(self) -> None:
        from sts2_hint import adapter

        def playable(bridge: dict[str, Any]) -> bool:
            wire = adapter.translate(to_headless_combat_play(bridge), character="Ironclad")
            idx = len(wire["deck"]["hand"]) - 1
            cs = sts2_rs.state_from_dict(wire)
            return any(
                a.get("action") == "play" and a.get("card_index") == idx for a in cs.legal_actions()
            )

        # Before: energy fields only -- the free attack is gated on 2 stars.
        assert not playable(self._bridge_state())
        # After: the star rider rides along, and the attack is the play it is.
        assert playable(self._bridge_state(star_cost=0, persistent_star_cost=2))
        # A star card at its printed cost stays gated (nothing over-grants).
        assert not playable(self._bridge_state(star_cost=2, persistent_star_cost=2))


class TestEnvelopeFieldsFromDialectAudit:
    """2026-08-02 dialect audit: fields the bridge sends (or now sends) that
    this module dropped, verified against a live capture
    (``logs/realgame/20260802T191922Z_17912_2.jsonl``).

    * ``rng`` (run_seed + counters): the deployed mod has published it on
      every ``combat_action`` via ``ApplyEnvelope``, but the translation
      dropped it — no live world_wire ever carried ``run_seed``
      (``mirror_live2.jsonl``), so ``CombatMirror.reconcile`` never let the
      world own the dice.
    * ``run_state.gold`` -> ``player.gold``: gates SealOfGold's turn-start
      energy buy (the v27wr DLL sweep's dominant ``$.player.energy`` class,
      fixed on the DLL arm but not here).
    * ``player.stars`` / ``room_type`` / card ``stats`` / ``enchantment``:
      emitted by the deploy-pending mod build (SimCombatHandler /
      SimStateExtractor); forwarded when present, silent when absent.
    """

    _RNG: ClassVar[dict[str, Any]] = {
        "run_seed": 3405856743,
        "counters": {"up_front": 411, "shuffle": 9, "monster_ai": 4},
    }

    def test_gold_reaches_player_and_wire(self) -> None:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        assert state["run_state"]["gold"] == 99  # real capture, not doctored
        out = to_headless_combat_play(state)
        assert out["player"]["gold"] == 99

        from sts2_hint import adapter

        wire = adapter.translate(out, character="Ironclad")
        assert wire["player"]["gold"] == 99

    def test_rng_pair_is_forwarded_to_the_wire(self) -> None:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        state["rng"] = dict(self._RNG)
        out = to_headless_combat_play(state)
        assert out["rng"] == self._RNG

        from sts2_hint import adapter

        wire = adapter.translate(out, character="Ironclad")
        assert wire["rng"]["run_seed"] == self._RNG["run_seed"]
        assert wire["rng"]["counters"] == self._RNG["counters"]

    def test_partial_or_absent_rng_stays_off_the_wire(self) -> None:
        """All-or-nothing: a seed without counters would rebuild every stream
        at count 0 — actively worse than the no-lineage fallback."""
        # Absent (the real pre-field fixture shape).
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        assert state.get("rng") is None
        assert "rng" not in to_headless_combat_play(state)
        # Seed without counters.
        state["rng"] = {"run_seed": 1}
        assert "rng" not in to_headless_combat_play(state)
        # Counters present but empty (the mod's all-or-nothing failure shape
        # is an empty dict, but guard the half-empty one too).
        state["rng"] = {"run_seed": 1, "counters": {}}
        assert "rng" not in to_headless_combat_play(state)

    def test_regent_stars_hoisted_to_top_level(self) -> None:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        assert "stars" not in to_headless_combat_play(state)  # pre-field build
        state["player"]["stars"] = 3
        out = to_headless_combat_play(state)
        assert out["stars"] == 3  # headless dialect: top-level, not player

    def test_room_type_maps_to_context_and_is_elite(self) -> None:
        from sts2_hint import adapter

        state = _load_fixture("single_enemy_ironclad.json")["state"]
        out = to_headless_combat_play(state)
        assert out["context"]["room_type"] == "Combat"  # pre-field default
        assert adapter.translate(out, character="Ironclad")["is_elite"] is False

        state["room_type"] = "Elite"
        out = to_headless_combat_play(state)
        assert out["context"]["room_type"] == "Elite"
        assert adapter.translate(out, character="Ironclad")["is_elite"] is True

    def test_card_stats_reach_base_damage_and_block(self) -> None:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        state["hand"][0]["stats"] = {"damage": 9, "magicnumber": 2}
        state["hand"][1]["stats"] = {"block": 8}
        out = to_headless_combat_play(state)
        assert out["hand"][0]["stats"] == {"damage": 9, "magicnumber": 2}
        assert out["hand"][1]["stats"] == {"block": 8}

        from sts2_hint import adapter

        wire = adapter.translate(out, character="Ironclad")
        assert wire["deck"]["hand"][0]["base_damage"] == 9
        assert wire["deck"]["hand"][1]["base_block"] == 8

    def test_enchantment_and_amount_forwarded(self) -> None:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        state["hand"][0]["enchantment"] = "NIMBLE"
        state["hand"][0]["enchantment_amount"] = 2
        # A one-stack enchantment omits the amount (headless convention:
        # absent means one stack — see ParsedCard::enchantment_amount).
        state["hand"][1]["enchantment"] = "STEADY"
        out = to_headless_combat_play(state)
        assert out["hand"][0]["enchantment"] == "NIMBLE"
        assert out["hand"][0]["enchantment_amount"] == 2
        assert out["hand"][1]["enchantment"] == "STEADY"
        assert "enchantment_amount" not in out["hand"][1]

        from sts2_hint import adapter

        wire = adapter.translate(out, character="Ironclad")
        assert wire["deck"]["hand"][0]["enchantment"] == "NIMBLE"
        assert wire["deck"]["hand"][0]["enchantment_amount"] == 2
        assert wire["deck"]["hand"][1]["enchantment"] == "STEADY"

    def test_pre_field_mod_build_grows_no_new_keys(self) -> None:
        """The untouched pre-field fixture must translate exactly as before:
        no fabricated stats/enchantment/stars/rng claims."""
        out = to_headless_combat_play(_load_fixture("single_enemy_ironclad.json")["state"])
        assert "rng" not in out
        assert "stars" not in out
        for card in out["hand"] + out["draw_pile"]:
            assert "stats" not in card
            assert "enchantment" not in card


class TestCharacterDisplayName:
    def test_known_ids(self) -> None:
        assert character_display_name("IRONCLAD") == "Ironclad"
        assert character_display_name("necrobinder") == "Necrobinder"

    def test_unknown_or_missing(self) -> None:
        assert character_display_name(None) is None
        assert character_display_name("") is None
        assert character_display_name("SOME_UNKNOWN_CHAR") is None


# ---------------------------------------------------------------------------
# Full pipeline -- skipped if sts2_rs isn't importable in this venv.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sts2_rs is None, reason=_SKIP_ENGINE_REASON or "")
class TestFullPipelineAgainstSts2Rs:
    """``to_headless_combat_play`` -> ``sts2_hint.adapter.translate`` ->
    ``sts2_rs.state_from_dict`` on real captured snapshots -- proves the
    adapter's output is not just structurally plausible but genuinely
    accepted by the engine."""

    @pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
    def test_translate_and_state_from_dict_succeed(self, fixture_name: str) -> None:
        from sts2_hint import adapter as hint_adapter

        fixture = _load_fixture(fixture_name)
        bridge_state = fixture["state"]
        headless = to_headless_combat_play(bridge_state)
        character = character_display_name(bridge_state["run_state"]["character"])

        wire = hint_adapter.translate(headless, character=character, seed=0)

        # Sanity invariants: nothing silently dropped card/enemy counts.
        assert len(wire["deck"]["hand"]) == len(bridge_state["hand"])
        bridge_draw = (bridge_state.get("draw_pile") or {}).get("cards") or []
        assert len(wire["deck"]["draw_pile"]) == len(bridge_draw)
        assert len(wire["encounter"]) == len(bridge_state["enemies"])
        assert wire["player"]["energy"] == bridge_state["player"]["energy"]
        assert wire["player"]["max_energy"] == bridge_state["player"]["max_energy"]
        assert wire["player"]["hp"] == bridge_state["player"]["hp"]

        # master_deck: the bridge's run_state.master_deck must survive the
        # full to_headless_combat_play -> sts2_hint.adapter.translate hop
        # (regression for the adapter's former hardcoded
        # ``deck.master_deck = None``). Not necessarily every card: a real
        # deck can carry a card sts2-rs doesn't model yet (this fixture's
        # does -- "Stampede"/``STAMPEDE`` isn't a registered card id), which
        # ``_soft_translate_list`` drops exactly like it already does for
        # hand/draw/discard/exhaust, so we assert "translated, non-empty,
        # no more than the source" rather than an exact count.
        bridge_master_deck = bridge_state["run_state"]["master_deck"]
        assert bridge_master_deck, "fixture should carry a non-empty master_deck"
        assert wire["deck"]["master_deck"] is not None
        assert 0 < len(wire["deck"]["master_deck"]) <= len(bridge_master_deck)

        cs = sts2_rs.state_from_dict(wire)  # type: ignore[union-attr]
        assert cs is not None
        # The engine actually ingested it -- not just present on the wire
        # dict fed in. `state_to_dict` is state_from_dict's documented
        # inverse (state_dict.rs); round-tripping proves master_deck landed
        # in the real `CoreCombatState.master_deck` Vec, not just echoed
        # back from the input dict.
        round_tripped = sts2_rs.state_to_dict(cs)  # type: ignore[union-attr]
        assert len(round_tripped["deck"]["master_deck"]) == len(wire["deck"]["master_deck"])

    def test_mcts_recommend_returns_a_legal_best_action(self) -> None:
        """End-to-end smoke test on one real snapshot: MCTS must return
        exactly one is_best action, and it must be convertible back to a
        bridge action by realgame_mcts.hint_tag_to_action."""
        from sts2_hint.recommender import RecommendConfig, recommend_all

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
        from realgame_mcts import hint_tag_to_action

        fixture = _load_fixture("single_enemy_ironclad.json")
        bridge_state = fixture["state"]
        headless = to_headless_combat_play(bridge_state)
        character = character_display_name(bridge_state["run_state"]["character"])

        cfg = RecommendConfig(sims=40, depth=20, seed=0, character=character)
        tags = recommend_all(headless, cfg)

        assert tags, "expected at least one legal action"
        bests = [t for t in tags if t.is_best]
        assert len(bests) == 1

        action = hint_tag_to_action(bests[0])
        assert action is not None
        assert action["action"] in {"play", "end_turn", "potion"}

    def test_mcts_recommend_survives_eye_with_teeth(self) -> None:
        """Regression, end-to-end: the exact snapshot that produced 12
        real ``mcts_error`` fallbacks in ``logs/realgame/
        20260711T004200Z.jsonl`` (``$.enemies[0]: unknown monster name
        'Eye with Teeth'``) must now flow through
        ``to_headless_combat_play`` -> ``recommend_all`` -> ``sts2_rs``
        and produce exactly one legal best action, instead of raising and
        falling back to random play."""
        from sts2_hint.recommender import RecommendConfig, recommend_all

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
        from realgame_mcts import hint_tag_to_action

        fixture = _load_fixture("eye_with_teeth_ironclad.json")
        bridge_state = fixture["state"]
        assert bridge_state["enemies"][0]["name"] == "Eye with Teeth"
        headless = to_headless_combat_play(bridge_state)
        character = character_display_name(bridge_state["run_state"]["character"])

        cfg = RecommendConfig(sims=40, depth=20, seed=0, character=character)
        tags = recommend_all(headless, cfg)

        assert tags, "expected at least one legal action"
        bests = [t for t in tags if t.is_best]
        assert len(bests) == 1

        action = hint_tag_to_action(bests[0])
        assert action is not None
        assert action["action"] in {"play", "end_turn", "potion"}

    def test_remap_potion_slot_dense_index_to_absolute_bridge_slot(self) -> None:
        """The engine compacts potions to a dense vector, so a potion action's
        ``slot`` is a 0-based index over non-empty potions -- but the bridge
        wants the absolute slot. ``_remap_potion_slot`` must rewrite it, else
        the play hits an empty slot -> ``empty_potion_slot`` and the AI can
        never use potions (root cause of the 0% live win-rate potion spam)."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
        from realgame_mcts import _remap_potion_slot

        # Only non-empty potion sits at absolute bridge slot 2.
        state = {
            "potions": [
                {"slot": 0, "empty": True},
                {"slot": 1, "empty": True},
                {"slot": 2, "empty": False, "id": "SKILL_POTION"},
                {"slot": 3, "empty": True},
            ]
        }
        action = {"action": "potion", "slot": 0}  # engine dense index 0
        _remap_potion_slot(action, state)
        assert action["slot"] == 2, "dense index 0 -> absolute slot 2"

        # Second non-empty potion -> its absolute slot.
        state2 = {
            "potions": [
                {"slot": 0, "empty": False, "id": "FIRE_POTION"},
                {"slot": 1, "empty": True},
                {"slot": 2, "empty": False, "id": "BLOCK_POTION"},
            ]
        }
        a2 = {"action": "potion", "slot": 1}
        _remap_potion_slot(a2, state2)
        assert a2["slot"] == 2

        # Non-potion actions and out-of-range indices are left untouched.
        play = {"action": "play", "card_index": 3}
        _remap_potion_slot(play, state)
        assert play == {"action": "play", "card_index": 3}
        oob = {"action": "potion", "slot": 9}
        _remap_potion_slot(oob, state)
        assert oob["slot"] == 9

    def test_a_strict_divergence_escapes_the_random_legal_fallback(self, monkeypatch) -> None:
        """STRICT must not be downgraded to "MCTS failed, play something random".

        This is the LIVE-game arm, and `realgame_eval` builds its mirror with
        `policy_from_env()` — STRICT by default. `MirrorDivergence` is a
        `RuntimeError`, so the blanket `except Exception` that catches a
        crashed search caught it too: a proven divergence became a log line and
        the run carried on answering from a state known to be wrong, with a
        random-legal action. The fallback also returns `engine_action=None`, so
        the session loop skips `mirror.advance(...)` and every later reconcile
        reports cascading false divergences on top of the real one.

        The fix is a bare `raise` with nothing else asserting it — exactly the
        shape a later refactor re-swallows silently. This is the guard.
        """
        import random as _random

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
        import realgame_mcts as rm
        from combat_mirror import MirrorDivergence
        from sts2_hint.recommender import RecommendConfig

        state = {"type": "combat_action", "round": 1, "hand": [], "enemies": []}
        cfg = RecommendConfig(sims=1, depth=1, seed=0)

        def _boom(*a, **k):
            raise MirrorDivergence("[mirror live] $.player.hp: mirror=40 world=35", {})

        monkeypatch.setattr(rm, "to_headless_combat_play", _boom)
        with pytest.raises(MirrorDivergence, match=r"\$\.player\.hp"):
            rm.pick_combat_action_mcts(state, cfg, _random.Random(0))

        # ...while an ordinary failure still degrades to a legal action, which
        # is the case the handler actually exists for.
        def _ordinary(*a, **k):
            raise RuntimeError("adapter exploded")

        monkeypatch.setattr(rm, "to_headless_combat_play", _ordinary)
        action, extra, engine = rm.pick_combat_action_mcts(state, cfg, _random.Random(0))
        assert extra == {"mcts_error": "adapter exploded"}
        assert engine is None
        assert isinstance(action, dict)


class TestTemplateMonsterTitles:
    """Titles with live-resolved localization variables must still map.

    Regression for sweep-8 (2026-07-13): the Act-3 ``TestSubject`` boss
    renders its title as ``"Test Subject #C8"`` (the ``{Count}`` variable
    resolved by the live game), but ``MONSTER_TITLE_TO_ID`` keeps the raw
    template key ``"Test Subject #C{Count}"`` -- the exact-title lookup
    raised ``unknown monster name`` and the whole boss fight fell back to
    the random-legal policy.
    """

    def test_resolved_count_variable_maps_to_test_subject(self) -> None:
        from sts2_hint.adapter import _translate_enemy

        enemy = _translate_enemy(
            {"name": "Test Subject #C8", "hp": 100, "max_hp": 100, "index": 0},
            "$.enemies[0]",
            0,
        )
        assert enemy["kind"] == "TestSubject"

    def test_unknown_names_still_raise(self) -> None:
        from sts2_hint.adapter import UnsupportedStateError, _translate_enemy

        with pytest.raises(UnsupportedStateError, match="unknown monster name"):
            _translate_enemy(
                {"name": "Totally Unknown 3", "hp": 1, "max_hp": 1, "index": 0},
                "$.enemies[0]",
                0,
            )


class TestCardAfflictionPassthrough:
    """Bound/Tainted card afflictions must reach the sts2_rs wire.

    Regression for the Queen fight (sweep-9 forensics, 2026-07-13): the
    wire dropped ``CardModel.Affliction`` at every layer, so rollouts
    planned through ChainsOfBinding as if every Bound card were freely
    playable (the game vetoes them: ``UnplayableReason.BlockedByHook``).
    """

    def _bridge_state_with_bound_card(self) -> dict[str, Any]:
        state = _load_fixture("single_enemy_ironclad.json")["state"]
        state["hand"][0]["affliction"] = "Bound"
        state["hand"][0]["affliction_stacks"] = 3
        state["player"]["bound_card_played_this_turn"] = True
        return state

    def test_bridge_affliction_survives_to_headless_shape(self) -> None:
        out = to_headless_combat_play(self._bridge_state_with_bound_card())
        assert out["hand"][0]["affliction"] == "Bound"
        assert out["hand"][0]["affliction_stacks"] == 3
        assert out["player"]["bound_card_played_this_turn"] is True
        # Unafflicted cards must not grow the keys.
        assert "affliction" not in out["hand"][1]

    def test_affliction_reaches_wire_spec(self) -> None:
        from sts2_hint import adapter

        out = to_headless_combat_play(self._bridge_state_with_bound_card())
        wire = adapter.translate(out, character="Ironclad")
        hand0 = wire["deck"]["hand"][0]
        assert hand0["affliction"] == "Bound"
        assert hand0["affliction_stacks"] == 3
        assert wire["player"]["bound_card_played_this_turn"] is True

    def test_headless_bilingual_title_dialect_normalizes(self) -> None:
        from sts2_hint.adapter import _translate_card

        spec = _translate_card(
            {
                "index": 0,
                "id": "STRIKE_IRONCLAD",
                "type": "Attack",
                "cost": 1,
                "affliction": "Bound / Gefesselt",
                "affliction_amount": 2,
            },
            "$.hand[0]",
            hand=True,
        )
        assert spec["affliction"] == "Bound"
        assert spec["affliction_stacks"] == 2

    def test_unknown_affliction_fails_loud(self) -> None:
        from sts2_hint.adapter import UnsupportedStateError, _translate_card

        with pytest.raises(UnsupportedStateError, match="unknown card affliction"):
            _translate_card(
                {
                    "index": 0,
                    "id": "STRIKE_IRONCLAD",
                    "type": "Attack",
                    "cost": 1,
                    "affliction": "Cursed",
                },
                "$.hand[0]",
                hand=True,
            )


class TestOrbDialect:
    """Defect orbs: the live game's ``Id.Entry`` is SCREAMING_SNAKE.

    ``PROTOCOL.md`` documents the Orb ``id`` as CamelCase (``"LightningOrb"``),
    and the adapter derived the ``ORB_TITLE_TO_ID`` key by stripping a
    trailing ``"Orb"``. The live v0.107.1 game sends ``"LIGHTNING_ORB"``, which
    survived that strip unchanged and made ``sts2_hint.adapter._translate_orb``
    raise ``unknown orb type 'LIGHTNING_ORB'`` -- so EVERY Defect combat
    decision with an orb on the field fell back to random-legal play (observed
    on the 2026-08-03 Defect A1 live sweep, from round 1 of run 1).
    """

    def test_live_snake_case_orb_ids_translate(self) -> None:
        fixture = _load_fixture("orbs_defect.json")
        bridge_state = fixture["state"]
        assert [o["id"] for o in bridge_state["orbs"]] == ["LIGHTNING_ORB"] * 2, (
            "fixture must keep the live SCREAMING_SNAKE spelling"
        )

        out = to_headless_combat_play(bridge_state)

        assert [o["type"] for o in out["orbs"]] == ["Lightning", "Lightning"]
        assert out["orb_slots"] == bridge_state["orb_capacity"]
        assert out["orbs"][0]["passive"] == bridge_state["orbs"][0]["passive_value"]
        assert out["orbs"][0]["evoke"] == bridge_state["orbs"][0]["evoke_value"]

    @pytest.mark.parametrize(
        ("orb", "expected"),
        [
            ({"id": "LIGHTNING_ORB", "name": "Lightning"}, "Lightning"),  # live
            ({"id": "LightningOrb", "name": "Lightning"}, "Lightning"),  # PROTOCOL.md
            ({"id": "FROST_ORB"}, "Frost"),  # id-only, live dialect
            ({"id": "PlasmaOrb"}, "Plasma"),  # id-only, documented dialect
            ({"id": "DARK_ORB", "name": ""}, "Dark"),  # blank name -> id
            ({"id": "GLASS_ORB"}, "Glass"),
        ],
    )
    def test_both_dialects_resolve_to_the_title_key(
        self, orb: dict[str, Any], expected: str
    ) -> None:
        from realgame_adapter import _orb_type_stem
        from sts2_hint.canonical_maps import ORB_TITLE_TO_ID

        stem = _orb_type_stem(orb)
        assert stem == expected
        assert stem in ORB_TITLE_TO_ID, "stem must be a key the hint adapter accepts"

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_orb_state_reaches_the_engine(self) -> None:
        """The whole point: a Defect board with orbs must survive
        ``translate`` -> ``state_from_dict`` instead of raising
        ``UnsupportedStateError`` and degrading the decision to random."""
        from sts2_hint import adapter as hint_adapter

        fixture = _load_fixture("orbs_defect.json")
        bridge_state = fixture["state"]
        headless = to_headless_combat_play(bridge_state)
        character = character_display_name(bridge_state["run_state"]["character"])
        assert character == "Defect"

        wire = hint_adapter.translate(headless, character=character, seed=0)
        assert [o["type"] for o in wire["player"]["orbs"]] == ["LIGHTNING", "LIGHTNING"]

        cs = sts2_rs.state_from_dict(wire)  # type: ignore[union-attr]
        round_tripped = sts2_rs.state_to_dict(cs)  # type: ignore[union-attr]
        assert len(round_tripped["player"]["orbs"]) == 2

    def test_an_empty_belt_keeps_its_capacity(self) -> None:
        """A Defect whose belt is empty still HAS a belt.

        ``orbs_empty_defect.json`` is a verbatim live capture with
        ``"orbs": [], "orb_capacity": 3``. The adapter gated on
        ``if orbs_raw:`` -- an empty list is falsy -- so BOTH ``orbs`` and
        ``orb_slots`` were dropped and the engine kept ``OrbSlots::default()``
        (capacity 0). The headless twin deliberately emits the pair whenever
        ``OrbQueue.Capacity > 0``, empty list included
        (``RunSimulator.cs``:3955-3967).
        """
        fixture = _load_fixture("orbs_empty_defect.json")
        bridge_state = fixture["state"]
        assert bridge_state["orbs"] == [], "fixture must keep the empty belt"
        assert bridge_state["orb_capacity"] == 3

        out = to_headless_combat_play(bridge_state)

        assert out["orbs"] == []
        assert out["orb_slots"] == 3

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_an_empty_belt_reaches_the_engine_with_its_capacity(self) -> None:
        from sts2_hint import adapter as hint_adapter

        fixture = _load_fixture("orbs_empty_defect.json")
        bridge_state = fixture["state"]
        headless = to_headless_combat_play(bridge_state)
        wire = hint_adapter.translate(headless, character="Defect", seed=0)
        assert wire["player"]["orbs"] == []
        assert wire["player"]["orb_slots"] == 3

        cs = sts2_rs.state_from_dict(wire)  # type: ignore[union-attr]
        round_tripped = sts2_rs.state_to_dict(cs)  # type: ignore[union-attr]
        assert round_tripped["player"]["orb_slots"] == 3, (
            "search root must start with the real belt, not capacity 0"
        )


class TestPetDialect:
    """``PlayerCombatState.Pets`` is a LIST, and Osty is only one of its members.

    ``OstyCmd.Summon`` (OstyCmd.cs:67), the **Byrdpip** relic (Event rarity, ANY
    character -- Byrdpip.cs:71) and **PaelsLegion** (Ancient, any character --
    PaelsLegion.cs:223) all ``AddPet``. The game's own accessor is
    ``Player.Osty => PlayerCombatState.GetPet<Osty>()`` (Player.cs:134 ->
    PlayerCombatState.cs:263, filtering ``p.Monster is Osty``), which is what
    the headless twin serializes (``RunSimulator.cs``:3976). The adapter took
    ``pets[0]`` unconditionally.
    """

    def test_a_paels_legion_pet_is_not_relabelled_osty(self) -> None:
        """Verbatim live capture: an IRONCLAD holding the PaelsLegion relic.

        ``pets[0]`` is the 9999/9999-HP PAELS_LEGION creature. The old code
        emitted it as ``player.osty``, and ``sts2-rs``'s ``restore_osty`` then
        pushed a real ``MonsterKind::Osty`` carrying DieForYou -- a phantom
        9999-HP damage soak on an Ironclad run.
        """
        fixture = _load_fixture("pets_paels_legion_ironclad.json")
        bridge_state = fixture["state"]
        assert [p["id"] for p in bridge_state["pets"]] == ["PAELS_LEGION"]
        assert bridge_state["run_state"]["character"] == "IRONCLAD"

        out = to_headless_combat_play(bridge_state)

        assert "osty" not in out, "a PaelsLegion pet must not become an Osty"

    def test_a_real_osty_capture_still_translates(self) -> None:
        """The other half of the same rule: an actual OSTY pet must survive.

        Verbatim live capture (``dead_osty_ironclad.json``) -- the pet's ``id``
        is ``"OSTY"``, the id ``SimStateExtractor.SerializeCreature`` takes from
        ``Monster.Id.Entry``.
        """
        fixture = _load_fixture("dead_osty_ironclad.json")
        bridge_state = fixture["state"]
        assert [p["id"] for p in bridge_state["pets"]] == ["OSTY"]

        out = to_headless_combat_play(bridge_state)

        assert out["osty"] is not None
        assert out["osty"]["max_hp"] == bridge_state["pets"][0]["max_hp"]

    def test_osty_is_selected_by_id_not_position(self) -> None:
        """Hand-built: no live capture holds two pets at once (a Byrdpip or
        PaelsLegion pet plus a summoned Osty on the same board is legal but
        rare), so the multi-pet ordering case is constructed from the verbatim
        single-pet shapes above."""
        fixture = _load_fixture("dead_osty_ironclad.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        bridge_state["run_state"]["character"] = "NECROBINDER"
        byrdpip = {
            "id": "BYRDPIP",
            "name": "Byrdpip",
            "is_alive": True,
            "hp": 14,
            "max_hp": 14,
            "block": 3,
            "powers": [],
        }
        osty = {
            "id": "OSTY",
            "name": "Osty",
            "is_alive": True,
            "hp": 22,
            "max_hp": 31,
            "block": 0,
            "powers": [],
        }
        # Byrdpip first -- exactly the ordering that used to win.
        bridge_state["pets"] = [byrdpip, osty]

        out = to_headless_combat_play(bridge_state)

        assert out["osty"] == {"alive": True, "hp": 22, "max_hp": 31, "block": 0}

    def test_no_pets_on_a_non_necrobinder_omits_the_key(self) -> None:
        fixture = _load_fixture("orbs_empty_defect.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        bridge_state["pets"] = []
        assert bridge_state["run_state"]["character"] == "DEFECT"

        assert "osty" not in to_headless_combat_play(bridge_state)

    def test_an_osty_less_necrobinder_gets_the_headless_sentinel(self) -> None:
        """The headless twin emits an explicit "no Osty" block for a Necrobinder
        rather than omitting the key (``RunSimulator.cs``:3988-3990); the bridge
        omits it. Hand-built: the live Necrobinder captures on disk all have a
        pet out."""
        fixture = _load_fixture("orbs_empty_defect.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        bridge_state["run_state"]["character"] = "NECROBINDER"
        bridge_state["pets"] = []

        out = to_headless_combat_play(bridge_state)

        assert out["osty"] == {"alive": False, "hp": 0, "max_hp": 0, "block": 0}


class TestDeadOstyRetainsMaxHp:
    """A dead Osty STAYS in ``Pets`` and keeps its accumulated ``MaxHp``.

    ``DieForYouPower.ShouldCreatureBeRemovedFromCombatAfterDeath`` returns false
    for its own owner (DieForYouPower.cs:40-47), so
    ``PlayerCombatState.OnPetDied`` never removes it
    (PlayerCombatState.cs:288-297) -- both producers ship
    ``alive: false, hp: 0, max_hp: N``. Osty's ``MaxHp`` COMPOUNDS across
    summons and is the Necrobinder's main accumulated resource, so zeroing it
    made the next summon rebuild a fresh spawn.
    """

    def test_the_bridge_ships_a_dead_osty_with_its_max_hp(self) -> None:
        fixture = _load_fixture("dead_osty_ironclad.json")
        pet = fixture["state"]["pets"][0]
        assert pet["is_alive"] is False
        assert pet["hp"] == 0
        assert pet["max_hp"] == 10, "the corpse keeps the accumulated MaxHp"
        assert any(p["id"] == "DIE_FOR_YOU_POWER" for p in pet["powers"]), (
            "DieForYou is what keeps the corpse in Pets"
        )

        out = to_headless_combat_play(fixture["state"])
        assert out["osty"] == {"alive": False, "hp": 0, "max_hp": 10, "block": 0}

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_the_corpses_max_hp_survives_to_the_engine(self) -> None:
        from sts2_hint import adapter as hint_adapter

        fixture = _load_fixture("dead_osty_ironclad.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        bridge_state["run_state"]["character"] = "NECROBINDER"
        headless = to_headless_combat_play(bridge_state)

        wire = hint_adapter.translate(headless, character="Necrobinder", seed=0)
        assert wire["player"]["osty"]["max_hp"] == 10, (
            "zeroing it made restore_osty clamp max_hp to 1"
        )

        cs = sts2_rs.state_from_dict(wire)  # type: ignore[union-attr]
        round_tripped = sts2_rs.state_to_dict(cs)  # type: ignore[union-attr]
        assert round_tripped["player"]["osty"]["alive"] is False
        assert round_tripped["player"]["osty"]["max_hp"] == 10


class TestRelicIdentityDialect:
    """Relics must be resolved by their canonical ``id``, not the rendered title.

    ``SimStateExtractor.cs``:666 sends ``relic.Title.GetFormattedText()``, and
    ``RelicModel.Title`` is neither stable nor guaranteed to exist here.
    """

    def test_the_bridge_relic_id_is_forwarded(self) -> None:
        fixture = _load_fixture("relic_fishing_rod_ironclad.json")
        bridge_state = fixture["state"]
        out = to_headless_combat_play(bridge_state)
        assert [r["id"] for r in out["player"]["relics"]] == [
            r["id"] for r in bridge_state["run_state"]["relics"]
        ]

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_a_relic_resolves_by_id_even_when_its_title_cannot(self) -> None:
        """Verbatim live capture: an IRONCLAD holding FISHING_ROD.

        FISHING_ROD had no ``.title`` row in ``localization_eng/relics.json``
        at all -- it was one of three v0.107.1 relics missing from the snapshot
        outright -- so ``RELIC_TITLE_TO_ID`` had no key for it and every one of
        the 1590 occurrences across the recent live logs was soft-dropped.
        ``47e4cb96b`` fixed that by preferring the bridge's canonical ``id``
        when it is a registry name.

        This test originally asserted the *snapshot gap itself* as its premise
        ("Fishing Rod" not in the title map). GH #358 refreshed the snapshot,
        which made that premise false -- so the premise is now constructed
        instead: the titles are overwritten with a deliberate non-title, which
        exercises the id path whatever the snapshot happens to contain, and
        pins the mechanism rather than a data gap that was only ever incidental.
        """
        from sts2_hint import adapter as hint_adapter
        from sts2_hint.canonical_maps import RELIC_TITLE_TO_ID

        fixture = _load_fixture("relic_fishing_rod_ironclad.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        unresolvable = "Not A Relic Title"
        assert unresolvable not in RELIC_TITLE_TO_ID, "premise: this title resolves to nothing"
        for relic in bridge_state["run_state"]["relics"]:
            relic["name"] = unresolvable

        headless = to_headless_combat_play(bridge_state)
        wire = hint_adapter.translate(headless, character="Ironclad", seed=0)

        assert {r["id"] for r in wire["player"]["relics"]} == {"BURNING_BLOOD", "FISHING_ROD"}

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_wax_prefixed_titles_no_longer_soft_drop(self) -> None:
        """Hand-built relic list -- no capture on disk carries a wax relic yet
        (checked: zero ``"is_wax": true`` records across ``logs/realgame``), so
        the ToyBox titles are constructed from the live template.

        ``RelicModel.Title`` prefixes a wax relic with ToyBox's
        ``"Wax {Title}"`` (RelicModel.cs:48-61, ToyBox.cs:26,
        ``localization_eng/relics.json`` ``"TOY_BOX.waxRelicPrefix"``), so ALL
        294 wax-prefixed titles missed ``RELIC_TITLE_TO_ID``. ToyBox grants FOUR
        wax relics at once (ToyBox.cs:87-95).
        """
        from sts2_hint import adapter as hint_adapter

        fixture = _load_fixture("relic_fishing_rod_ironclad.json")
        bridge_state = json.loads(json.dumps(fixture["state"]))
        bridge_state["run_state"]["relics"] = [
            {
                "id": "TOY_BOX",
                "name": "Toy Box",
                "counter": 1,
                "stack_count": 1,
                "is_wax": False,
                "is_melted": False,
            },
            {
                "id": "BURNING_BLOOD",
                "name": "Wax Burning Blood",
                "counter": None,
                "stack_count": 1,
                "is_wax": True,
                "is_melted": False,
            },
            {
                "id": "NUNCHAKU",
                "name": "Wax Nunchaku",
                "counter": 4,
                "stack_count": 1,
                "is_wax": True,
                "is_melted": True,
            },
        ]

        headless = to_headless_combat_play(bridge_state)
        wire = hint_adapter.translate(headless, character="Ironclad", seed=0)

        relics = {r["id"]: r for r in wire["player"]["relics"]}
        assert set(relics) == {"TOY_BOX", "BURNING_BLOOD", "NUNCHAKU"}
        # The melt work only bites if the relic survives translation at all.
        assert relics["BURNING_BLOOD"]["is_wax"] is True
        assert relics["NUNCHAKU"]["is_wax"] is True
        assert relics["NUNCHAKU"]["is_melted"] is True
        assert relics["NUNCHAKU"]["counter"] == 4


class TestWireCarriedUnlockState:
    """GH #374 — the live save's real ``UnlockState`` rides the envelope.

    Stage 1 (#367) inferred the save's unrevealed epochs from a corpus census
    and hardcoded the answer. That is a snapshot of one save at one moment: it
    goes stale silently the instant the timeline advances, and the failure mode
    is exactly #367's original symptom — a ``combat_card_generation`` cursor
    drift of +1 with no signal that anything changed.
    """

    @pytest.fixture(autouse=True)
    def _clear_latch(self):
        from realgame_adapter import _reset_wire_unlock_latch

        _reset_wire_unlock_latch()
        yield
        _reset_wire_unlock_latch()

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_absent_field_is_not_the_same_as_nothing_revealed(self):
        """The load-bearing distinction.

        A mod build predating this field carries no ``unlocked_epochs``. If
        that were read as "revealed nothing", every one of the 60 gated cards
        would be pruned from every pool — reintroducing #367's drift, far
        larger, with no signal. ``None`` must mean "producer could not say".
        """
        from realgame_adapter import locked_epochs_from_wire

        assert locked_epochs_from_wire({}) is None
        assert locked_epochs_from_wire({"unlocked_epochs": None}) is None
        # An EMPTY list is a real answer — UnlockState.none, nothing revealed —
        # so it locks everything the engine models.
        every = locked_epochs_from_wire({"unlocked_epochs": []})
        assert every is not None
        assert set(every) == set(sts2_rs.known_epochs())

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_locked_is_the_complement_over_the_engines_own_epochs(self):
        """C# knows epochs the engine does not; those must not become locks.

        ``UnlockedEpochs`` is the raw C# list and includes ids with no engine
        counterpart (``RELIC3_EPOCH``, ``UNDERDOCKS_EPOCH``, ...). Inverting
        against the engine's own coverage — not against the wire's contents —
        is what keeps an unmodelled id from silently becoming a lock.
        """
        from realgame_adapter import locked_epochs_from_wire

        known = list(sts2_rs.known_epochs())
        revealed = [*known[:-1], "RELIC3_EPOCH", "UNDERDOCKS_EPOCH", "DARV_EPOCH"]
        locked = locked_epochs_from_wire({"unlocked_epochs": revealed})
        assert locked == [known[-1]], "only the one genuinely-unrevealed epoch locks"

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_case_and_whitespace_are_tolerated(self):
        """The wire is another process's JSON; do not desync on formatting."""
        from realgame_adapter import locked_epochs_from_wire

        known = list(sts2_rs.known_epochs())
        sloppy = [f"  {e.lower()} " for e in known]
        assert locked_epochs_from_wire({"unlocked_epochs": sloppy}) == []

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_install_supersedes_the_census_and_latches(self, monkeypatch, caplog):
        """The wire wins over the hardcoded census, once, and says so."""
        import logging

        from realgame_adapter import install_unlock_state_from_wire

        monkeypatch.delenv("STS2_LIVE_LOCKED_EPOCHS", raising=False)
        known = list(sts2_rs.known_epochs())
        # Reveal everything except one epoch that the census does NOT list —
        # so the wire and the census necessarily disagree.
        target = next(e for e in known if e not in {"REGENT5_EPOCH"})
        revealed = [e for e in known if e != target]

        with caplog.at_level(logging.WARNING):
            installed = install_unlock_state_from_wire({"unlocked_epochs": revealed})
        assert installed == [target]
        assert sts2_rs.locked_cards(), "the install must actually prune cards"
        assert any("DISAGREES with the hardcoded census" in r.message for r in caplog.records)

        # Latched: a later message with a DIFFERENT unlock state is ignored,
        # so the process cannot flip-flop its pools mid-run.
        again = install_unlock_state_from_wire({"unlocked_epochs": known})
        assert again == [target]

        sts2_rs.set_locked_epochs([])

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_an_explicit_env_pin_is_not_overwritten(self, monkeypatch):
        """A pinned lock set means someone is bisecting; do not undermine it."""
        from realgame_adapter import install_unlock_state_from_wire

        monkeypatch.setenv("STS2_LIVE_LOCKED_EPOCHS", "REGENT5_EPOCH")
        assert install_unlock_state_from_wire({"unlocked_epochs": []}) is None

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_a_silent_wire_leaves_the_census_install_alone(self, monkeypatch):
        from realgame_adapter import install_unlock_state_from_wire

        monkeypatch.delenv("STS2_LIVE_LOCKED_EPOCHS", raising=False)
        assert install_unlock_state_from_wire({}) is None

    @pytest.mark.skipif(_SKIP_ENGINE_REASON is not None, reason=_SKIP_ENGINE_REASON or "")
    def test_known_epochs_covers_every_id_the_census_names(self):
        """Guards the two tables against drifting apart.

        ``LIVE_LOCKED_EPOCHS`` is hand-maintained and feeds
        ``set_locked_epochs``, which RAISES on an unknown id — so a rename on
        the engine side would turn the startup fallback into a hard crash on
        the live arm.
        """
        from realgame_adapter import LIVE_LOCKED_EPOCHS

        known = {e.upper() for e in sts2_rs.known_epochs()}
        assert set(LIVE_LOCKED_EPOCHS) <= known
