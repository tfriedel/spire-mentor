"""F9 engine invariants (``ppo.e2e.invariants``).

Two halves. Real v0.111.0 A10 runs, played with random legal actions, must
report no violation at any decision: the live gate stops the lane on one, so
a check that fires on a legal state is a bug in the check. And each check must
fire on a state that breaks it: duck-typed run and combat stubs for the run
and engine tiers, tampered real combat dicts for the deck ledger.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any

import pytest
import sts2_rs
from ppo.e2e import invariants as inv
from ppo.e2e.invariant_soak import main as soak_main
from ppo.e2e.invariant_soak import random_run

# -- stubs -----------------------------------------------------------------


class _Combat:
    """The ``CombatState`` surface the checker reads."""

    def __init__(self, *, engine_msg: str | None = None, version: str = "v0.111.0") -> None:
        self.engine_msg = engine_msg
        self.game_version = version

    def check_invariants(self, _tag: str) -> str | None:
        return self.engine_msg

    def pending_choice(self) -> None:
        return None

    def has_pending_play(self) -> bool:
        return False


class _Run:
    """The ``RunManager`` surface the checker reads; healthy by default."""

    def __init__(self, **kw: Any) -> None:
        self.current_hp = kw.get("hp", 50)
        self.max_hp = kw.get("max_hp", 80)
        self.gold = kw.get("gold", 99)
        self.act_index = kw.get("act", 1)
        self.game_version = kw.get("version", "v0.111.0")
        self.held = kw.get("potions", 1)
        self.slots = kw.get("slots", 3)
        self.floor = kw.get("floor", 1)
        self.combat = kw.get("combat")

    def potions(self) -> list[str]:
        return ["FIRE_POTION"] * self.held

    def max_potion_slots(self) -> int:
        return self.slots

    def total_floor(self) -> int:
        return self.floor

    def phase(self) -> str:
        return "combat" if self.combat is not None else "map"

    def is_over(self) -> bool:
        return False

    def combat_state(self) -> _Combat | None:
        return self.combat

    def master_deck(self) -> list[str]:
        return []


def _codes(found: list[inv.Violation]) -> list[str]:
    return sorted(v.code for v in found)


# -- run tier --------------------------------------------------------------


def test_a_healthy_state_reports_nothing() -> None:
    assert inv.InvariantChecker().check(_Run()) == []
    assert inv.InvariantChecker().check(_Run(combat=_Combat())) == []


@pytest.mark.parametrize(
    ("kw", "code"),
    [
        ({"hp": 81, "max_hp": 80}, inv.RUN_HP),
        ({"hp": -1}, inv.RUN_HP),
        ({"hp": 0, "max_hp": 0}, inv.RUN_HP),
        ({"gold": -1}, inv.RUN_GOLD),
        ({"potions": 4, "slots": 3}, inv.RUN_POTIONS),
        ({"act": 0}, inv.RUN_ACT),
        ({"act": 5}, inv.RUN_ACT),
        ({"version": "v0.107.1"}, inv.RUN_VERSION),
    ],
)
def test_each_run_check_fires(kw: dict[str, Any], code: str) -> None:
    assert _codes(inv.InvariantChecker().check(_Run(**kw))) == [code]


def test_the_edges_of_the_run_ranges_are_legal() -> None:
    edges = _Run(hp=0, max_hp=1, gold=0, potions=3, slots=3, act=4)
    assert inv.InvariantChecker().check(edges) == []


def test_no_expected_version_turns_the_version_checks_off() -> None:
    rm = _Run(version="v0.107.1", combat=_Combat(version="v0.107.1"))
    assert inv.InvariantChecker(expected_game_version=None).check(rm) == []


def test_floor_regression_needs_the_run_key() -> None:
    checker = inv.InvariantChecker()
    assert checker.check(_Run(floor=5), run_key="a") == []
    assert checker.check(_Run(floor=6), run_key="a") == []
    assert _codes(checker.check(_Run(floor=4), run_key="a")) == [inv.RUN_FLOOR]
    # The watermark stays at 6, and other runs or no key are unaffected.
    assert _codes(checker.check(_Run(floor=5), run_key="a")) == [inv.RUN_FLOOR]
    assert checker.check(_Run(floor=0), run_key="b") == []
    assert checker.check(_Run(floor=0)) == []


def test_the_floor_watermarks_are_bounded() -> None:
    checker = inv.InvariantChecker(max_runs_tracked=2)
    for key in ("a", "b", "c"):
        checker.check(_Run(floor=9), run_key=key)
    # "a" was evicted, so its lower floor starts a fresh watermark.
    assert checker.check(_Run(floor=1), run_key="a") == []
    assert _codes(checker.check(_Run(floor=1), run_key="c")) == [inv.RUN_FLOOR]


# -- combat tier -----------------------------------------------------------


def test_an_engine_invariant_message_is_a_violation() -> None:
    rm = _Run(combat=_Combat(engine_msg="[e2e] creature 1 hp 120 > max_hp 100"))
    found = inv.InvariantChecker().check(rm)
    assert _codes(found) == [inv.COMBAT_ENGINE]
    assert "hp 120" in found[0].detail


def _wire(hand: int = 5, player_powers=(), enemy_powers=(), relics=()) -> dict[str, Any]:
    hand_cards = [{"id": "STRIKE_IRONCLAD", "stable_idx": i + 1} for i in range(hand)]
    return {
        "deck": {"hand": hand_cards, "draw_pile": [], "discard_pile": [], "exhaust_pile": []},
        "player": {
            "powers": [{"name": n, "amount": a} for n, a in player_powers],
            "relics": [{"id": r} for r in relics],
        },
        "encounter": [{"uid": 2, "powers": [{"name": n, "amount": a} for n, a in enemy_powers]}],
    }


@pytest.mark.parametrize(
    ("msg", "reason"),
    [
        ("duplicate PowerKind ToricToughness on creature 1 (e2e)", "instanced_power"),
        ("duplicate PowerKind TheBomb on player (e2e)", "instanced_power"),
        (
            "card CardUid(6) in pinned_top but not in the pile it annotates (e2e)",
            "stale_pinned_top",
        ),
        (
            "card CardUid(9) in retain_this_turn but not in the pile it annotates (e2e)",
            "stale_retain_this_turn",
        ),
        ("duplicate PowerKind Strength on player (e2e)", None),
        ("card CardUid(6) in pinned_top but not in state.cards arena (e2e)", None),
        ("hand size 11 exceeds cap 10 (e2e)", None),
    ],
)
def test_only_legal_engine_failures_are_tolerated(msg: str, reason: str | None) -> None:
    assert inv.tolerated_engine_message(msg) == reason


def test_a_tolerated_failure_reruns_the_checks_it_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = {"state": _wire()}
    monkeypatch.setattr(sts2_rs, "state_to_dict", lambda _cs: wire["state"])
    checker = inv.InvariantChecker(ledger_every=0)
    stale = _Combat(
        engine_msg="card CardUid(6) in pinned_top but not in the pile it annotates (e2e)"
    )
    assert checker.check(_Run(combat=stale)) == []
    assert checker.stats()["tolerated"] == {"stale_pinned_top": 1}
    assert checker.stats()["violations"] == 0
    # What the engine suite never reached is still checked.
    wire["state"] = _wire(hand=11)
    found = checker.check(_Run(combat=stale))
    assert _codes(found) == [inv.COMBAT_ENGINE]
    assert "hand size 11" in found[0].detail
    wire["state"] = _wire(player_powers=[("STRENGTH", 1), ("STRENGTH", 2)])
    assert "duplicate power STRENGTH" in checker.check(_Run(combat=stale))[0].detail
    wire["state"] = _wire(enemy_powers=[("VULNERABLE", -1)])
    assert "negative VULNERABLE" in checker.check(_Run(combat=stale))[0].detail
    # Instanced powers may repeat on any owner.
    wire["state"] = _wire(player_powers=[("TORIC_TOUGHNESS", 1), ("TORIC_TOUGHNESS", 2)])
    assert checker.check(_Run(combat=stale)) == []


def test_the_crossbow_overflow_passes_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = {"state": _wire(hand=11, relics=["CROSSBOW"])}
    monkeypatch.setattr(sts2_rs, "state_to_dict", lambda _cs: wire["state"])
    over = _Combat(engine_msg="hand size 11 exceeds cap 10 (e2e)")
    assert _codes(inv.InvariantChecker().check(_Run(combat=over))) == [inv.COMBAT_ENGINE]
    known = inv.InvariantChecker(tolerate_known=["crossbow_hand_cap"])
    assert known.check(_Run(combat=over)) == []
    assert known.stats()["tolerated"] == {"known_crossbow_hand_cap": 1}
    # Without Crossbow the same hand is still a fault.
    wire["state"] = _wire(hand=11)
    assert _codes(known.check(_Run(combat=over))) == [inv.COMBAT_ENGINE]
    # A stale annotation can hide the cap check; the re-check honours the key.
    stale = _Combat(
        engine_msg="card CardUid(6) in pinned_top but not in the pile it annotates (e2e)"
    )
    wire["state"] = _wire(hand=11, relics=["CROSSBOW"])
    assert known.check(_Run(combat=stale)) == []
    assert _codes(inv.InvariantChecker().check(_Run(combat=stale))) == [inv.COMBAT_ENGINE]
    with pytest.raises(ValueError, match="unknown known-issue"):
        inv.InvariantChecker(tolerate_known=["no_such_issue"])


def test_a_combat_on_the_wrong_build_is_a_violation() -> None:
    rm = _Run(combat=_Combat(version="v0.107.1"))
    assert _codes(inv.InvariantChecker().check(rm)) == [inv.COMBAT_VERSION]


def test_check_combat_runs_without_a_run() -> None:
    checker = inv.InvariantChecker()
    assert checker.check_combat(_Combat()) == []
    assert _codes(checker.check_combat(_Combat(engine_msg="boom"))) == [inv.COMBAT_ENGINE]
    assert checker.stats()["by_code"] == {inv.COMBAT_ENGINE: 1}


def test_a_raising_check_becomes_checker_error() -> None:
    class Broken(_Run):
        def potions(self) -> list[str]:
            raise RuntimeError("engine gone")

    found = inv.InvariantChecker().check(Broken())
    assert _codes(found) == [inv.CHECKER_ERROR]
    assert "engine gone" in found[0].detail


def test_the_ledger_runs_on_every_nth_combat_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_state_to_dict(_cs: Any) -> dict[str, Any]:
        calls.append(1)
        return {"deck": {p: [] for p in inv._PILES}, "encounter": [], "player": {}}

    monkeypatch.setattr(sts2_rs, "state_to_dict", fake_state_to_dict)
    checker = inv.InvariantChecker(ledger_every=4)
    for _ in range(8):
        checker.check(_Run(combat=_Combat()))
    checker.check(_Run())  # out of combat: not a combat call
    assert len(calls) == 2
    assert checker.stats()["ledger_calls"] == 2
    assert checker.stats()["ledger_skips"] == {"empty_deck": 2}
    off = inv.InvariantChecker(ledger_every=0)
    for _ in range(8):
        off.check(_Run(combat=_Combat()))
    assert len(calls) == 2


def test_maybe_check_samples_from_its_own_rng() -> None:
    checker = inv.InvariantChecker(sample_rate=0.25, seed=7)
    random.seed(123)
    expected = random.random()
    random.seed(123)
    for _ in range(4000):
        checker.maybe_check(_Run())
    assert random.random() == expected  # the global stream was not touched
    stats = checker.stats()
    assert stats["steps_seen"] == 4000
    assert 800 < stats["calls"] < 1200
    always = inv.InvariantChecker(sample_rate=1.0)
    never = inv.InvariantChecker(sample_rate=0.0)
    for _ in range(50):
        assert _codes(always.maybe_check(_Run(gold=-5))) == [inv.RUN_GOLD]
        assert never.maybe_check(_Run(gold=-5)) == []
    assert always.stats()["violations"] == 50
    assert never.stats()["calls"] == 0


# -- deck ledger on real combat states ---------------------------------------


@pytest.fixture(scope="module")
def first_combat() -> tuple[list[str], dict[str, Any]]:
    """``(master_deck, state_to_dict)`` at the first settled decision of a fight."""
    for rm in random_run(1_400_000_002, pin_hp=False, max_steps=400):
        if rm.phase() != "combat" or rm.is_over():
            continue
        cs = rm.combat_state()
        if cs.pending_choice() is None and not cs.has_pending_play():
            return list(rm.master_deck()), sts2_rs.state_to_dict(cs)
    pytest.fail("no combat reached")


def _uid_cards(state: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    return [(p, i, c) for p in inv._PILES for i, c in enumerate(state["deck"][p])]


def _drop_deck_card(state: dict[str, Any], n: int, type_: str) -> tuple[dict[str, Any], int]:
    """A copy of ``state`` with one deck card of spec type ``type_`` removed."""
    types = inv._card_types()
    out = copy.deepcopy(state)
    for pile, i, card in _uid_cards(out):
        if card["stable_idx"] <= n and types.get(inv._canon(card["id"])) == type_:
            del out["deck"][pile][i]
            return out, int(card["stable_idx"])
    pytest.fail(f"no {type_} deck card in the opening state")


def test_the_real_opening_state_is_clean(first_combat) -> None:
    deck, state = first_combat
    assert inv.deck_ledger(deck, state) == ([], None)


def test_a_card_stuck_in_the_play_pile_is_present(first_combat) -> None:
    """The 2026-11-04 F9 stop: a stuck auto-play leaves its card in `play_pile`."""
    deck, state = first_combat
    broken, uid = _drop_deck_card(state, len(deck), "ATTACK")
    broken["deck"]["play_pile"] = [{"stable_idx": uid, "id": deck[len(deck) - uid]}]
    assert inv.deck_ledger(deck, broken) == ([], None)


def test_a_vanished_attack_is_caught(first_combat) -> None:
    deck, state = first_combat
    broken, _uid = _drop_deck_card(state, len(deck), "ATTACK")
    found, skip = inv.deck_ledger(deck, broken)
    assert skip is None
    assert _codes(found) == [inv.COMBAT_DECK_MISSING]


def test_a_played_power_may_leave_every_pile(first_combat) -> None:
    deck, state = first_combat
    broken, uid = _drop_deck_card(state, len(deck), "ATTACK")
    as_power = list(deck)
    as_power[len(deck) - uid] = "Inflame"
    assert inv.deck_ledger(as_power, broken) == ([], None)


def test_a_played_mad_science_may_leave_every_pile(first_combat) -> None:
    """MadScience takes the type its event picked, so a Power copy leaves combat."""
    deck, state = first_combat
    broken, uid = _drop_deck_card(state, len(deck), "ATTACK")
    with_mad = list(deck)
    with_mad[len(deck) - uid] = "MadScience"
    assert inv.deck_ledger(with_mad, broken) == ([], None)


def test_a_transform_result_explains_one_missing_card(first_combat) -> None:
    deck, state = first_combat
    broken, _uid = _drop_deck_card(state, len(deck), "ATTACK")
    rock = {"id": "GIANT_ROCK", "stable_idx": len(deck) + 40}
    broken["deck"]["hand"].append(rock)
    assert inv.deck_ledger(deck, broken) == ([], None)
    twice, _ = _drop_deck_card(broken, len(deck), "ATTACK")
    assert _codes(inv.deck_ledger(deck, twice)[0]) == [inv.COMBAT_DECK_MISSING]


def test_a_card_in_two_piles_is_caught(first_combat) -> None:
    deck, state = first_combat
    broken = copy.deepcopy(state)
    card = _uid_cards(broken)[0][2]
    broken["deck"]["exhaust_pile"].append(dict(card))
    found, skip = inv.deck_ledger(deck, broken)
    assert skip is None
    assert _codes(found) == [inv.COMBAT_DUPLICATE]


def test_the_ledger_skips_what_it_cannot_explain(first_combat) -> None:
    deck, state = first_combat
    broken, _uid = _drop_deck_card(state, len(deck), "ATTACK")

    thief = copy.deepcopy(broken)
    thief["encounter"].append({"kind": "ThievingHopper", "powers": []})
    assert inv.deck_ledger(deck, thief) == ([], "thief")

    entropy = copy.deepcopy(broken)
    entropy["player"]["powers"] = [*entropy["player"].get("powers", []), {"name": "Entropy"}]
    assert inv.deck_ledger(deck, entropy) == ([], "random_transform")

    assert inv.deck_ledger(list(reversed(deck)), state)[1] == "deck_order"
    assert inv.deck_ledger([], state) == ([], "empty_deck")


def test_no_v0111_power_is_typed_as_a_non_power() -> None:
    """The ledger exempts Powers by ``sts2_rs.card_kinds()``, which is one list
    for every build. A card that is a Power in v0.111.0 but not in that list
    would leave every pile legally and read as missing: a false F9 stop."""
    import re
    from pathlib import Path

    decomp = Path(__file__).resolve().parents[3] / "sts2-wiki" / "decompiled" / "v0.111.0"
    ctor = re.compile(r"base\([^;]*?CardType\.([A-Za-z]+)")
    powers = set()
    for path in decomp.glob("*.Sts2.Core.Models.Cards/*.cs"):
        m = ctor.search(path.read_text(encoding="utf-8", errors="replace"))
        if m and m.group(1) == "Power":
            powers.add(inv._canon(path.stem))
    assert len(powers) > 50
    types = inv._card_types()
    mistyped = sorted(c for c in powers if types.get(c, types.get(c + "CARD", "POWER")) != "POWER")
    assert mistyped == []


def test_the_instanced_powers_match_the_decomp() -> None:
    """The tolerated duplicates are exactly C#'s instanced powers, and
    MadScience is the only card whose type is not fixed."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "sts2-wiki" / "decompiled" / "v0.111.0"
    instanced = re.compile(r"InstanceType => PowerInstanceType\.(?!None)")
    found = set()
    for path in root.glob("*.Sts2.Core.Models.Powers/*Power.cs"):
        if instanced.search(path.read_text(encoding="utf-8", errors="replace")):
            found.add(inv._canon(path.stem.removesuffix("Power")))
    assert found == inv._INSTANCED_POWERS
    dynamic = {
        inv._canon(path.stem)
        for path in root.glob("*.Sts2.Core.Models.Cards/*.cs")
        if "override CardType Type =>" in path.read_text(encoding="utf-8", errors="replace")
    }
    assert dynamic == inv._DYNAMIC_TYPE_CARDS


# -- real runs -------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "pin_hp"),
    [(1_400_000_000, True), (1_400_000_001, False), (1_400_000_004, True), (1_400_000_005, False)],
)
def test_random_runs_report_no_violation(seed: int, pin_hp: bool) -> None:
    checker = inv.InvariantChecker(ledger_every=1)
    steps = max_act = 0
    for rm in random_run(seed, pin_hp=pin_hp, max_steps=3000):
        steps += 1
        found = checker.check(rm, run_key=seed)
        assert found == [], (rm.total_floor(), found)
        max_act = max(max_act, rm.act_index)
    stats = checker.stats()
    assert stats["violations"] == 0
    assert stats["calls"] == steps
    assert stats["ledger_calls"] > 0
    if pin_hp:
        assert max_act >= 2, "the pinned walk should leave act 1"


def test_the_soak_crossbow_overflow_is_real_and_known() -> None:
    """Seed 1,400,001,854 of the F9 soak reaches an 11-card hand: MechaKnight's
    Flamethrower adds four Burns, the draw fills the hand, then Crossbow adds a
    free Attack (C# would send it to discard). Skips once the engine is fixed."""
    strict = inv.InvariantChecker(ledger_every=0)
    known = inv.InvariantChecker(ledger_every=0, tolerate_known=["crossbow_hand_cap"])
    for rm in random_run(1_400_001_854, pin_hp=True, max_steps=1200):
        if rm.phase() != "combat" or rm.is_over():
            continue
        found = strict.check(rm)
        if found:
            assert [v.detail for v in found] == ["hand size 11 exceeds cap 10 (e2e)"]
            assert "RELIC.CROSSBOW" in rm.relics()
            assert known.check(rm) == []
            assert known.stats()["tolerated"] == {"known_crossbow_hand_cap": 1}
            return
    pytest.skip("no 11-card hand: the Crossbow hand cap looks fixed in this wheel")


def test_a_run_on_the_wrong_build_is_flagged() -> None:
    checker = inv.InvariantChecker()
    seen: set[str] = set()
    for rm in random_run(1_400_000_001, pin_hp=False, max_steps=300, game_version="v0.107.1"):
        seen.update(v.code for v in checker.check(rm))
        if inv.COMBAT_VERSION in seen:
            break
    assert {inv.RUN_VERSION, inv.COMBAT_VERSION} <= seen


def test_the_module_check_runs_the_ledger() -> None:
    for rm in random_run(1_400_000_002, pin_hp=False, max_steps=400):
        if rm.phase() == "combat":
            assert inv.check(rm) == []
            return
    pytest.fail("no combat reached")


def test_the_soak_cli_writes_its_summary(tmp_path) -> None:
    out = tmp_path / "soak.json"
    rc = soak_main(["--runs", "2", "--workers", "1", "--max-steps", "200", "--out", str(out)])
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["totals"]["runs"] == 2
    assert summary["totals"]["violations"] == 0
    assert summary["totals"]["checks"] == summary["totals"]["steps"]
    assert out.with_suffix(".runs.jsonl").read_text(encoding="utf-8").count("\n") == 2
