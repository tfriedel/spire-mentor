"""Read-only ``CombatState`` getters for the e2e tokenizer (GH #1097).

``cs.telegraph()``, ``cs.peek_intents(k=3)``, ``cs.encounter_id`` and
``cs.true_draw_top(k=10)``, checked on every live combat decision of whole
A10 Ironclad runs on both builds, three seeds each (over 1,000 states per
build). The runs are invincible (HP pinned), fights are played by a small
MCTS and every other decision is a seeded random pick:

* ``telegraph()[i][0]`` is the wire's ``encounter[i]["intents"][0]``, and
  multi-intent moves show their other intents after it.
* ``peek_intents`` leaves every hash and ``state_to_dict`` as they were, a
  clone that called the getters plays on exactly like one that did not, it
  is deterministic, and a certain step 1 names the move the enemy telegraphs
  after the coming enemy turn.
* ``encounter_id`` is the encounter the run generated for the room: the act's
  ``normal`` / ``elite`` lists in visit order, its ``boss`` and then its
  ``second_boss``, or an event's own fight. A lingering finished fight has
  none.
* ``true_draw_top`` is the wire's draw pile, the next turn draws from its top
  in that order, and a ``belief_view`` refuses it.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

VERSIONS = ("v0.111.0", "v0.107.1")
#: Seeds whose runs reach act 3 and end on both builds under this driver.
SEEDS = (1, 5, 14)
#: Per run; these end in 600-1,200.
MAX_DECISIONS = 4000
#: An invincible player can stall a fight for good (Lagavulin Matriarch drains
#: Strength below any damage); such a run is abandoned at this round.
MAX_ROUNDS = 60
EVENT_ENCOUNTERS = {
    "BATTLEWORN_DUMMY_EVENT_ENCOUNTER",
    "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER",
    "FAKE_MERCHANT_EVENT_ENCOUNTER",
    "DENSE_VEGETATION_EVENT_ENCOUNTER",
}
#: Monsters whose next move is decided by something the coming turn does
#: besides the move machine, so a forecast made against the current board can
#: name another move with certainty 1.0. The first three are also in the Rust
#: census (`sts2-wire/tests/combat_getters_1097.rs`, `BOARD_DRIVEN`).
BOARD_DRIVEN = {
    # The Slumber counter runs out at turn end and the wake re-points the
    # telegraph.
    "SlumberingBeetle",
    # The telegraphed move decrements the AI counter that opens
    # CALL_FOR_BACKUP.
    "TwoTailedRat",
    # Asleep for a fixed number of turns, then it wakes.
    "LagavulinMatriarch",
    # A fully blocked HEADBUTT knocks it off balance into DIZZY
    # (`ImbalancedPower`).
    "BowlbugRock",
    # Breaking its block while burrowed stuns it into DIZZY_MOVE
    # (`BurrowedPower.AfterBlockBroken`).
    "Tunneler",
    # An ally's death stuns it while it devours the corpse
    # (`RavenousPower.AfterDeath`).
    "CorpseSlug",
}


def _run(seed: int, version: str) -> Any:
    return sts2_rs.RunManager(
        seed,
        character="Ironclad",
        ascension_level=10,
        neow_start=True,
        act1="roll",
        game_version=version,
        mode="agent",
        interactive_potion_swap=True,
    )


def _bare(intent: dict[str, Any]) -> tuple[Any, Any, Any]:
    """An intent's move-table part, without the board-baked numbers."""
    return (intent["kind"], intent.get("amount"), intent.get("times"))


def _fingerprint(cs: Any) -> tuple[Any, ...]:
    return (
        cs.state_hash(),
        cs.ordered_hash(),
        cs.search_hash(),
        cs.observation_hash(),
        sts2_rs.state_to_dict(cs),
    )


@dataclass
class _Tally:
    states: int = 0
    multi_intent: int = 0
    forecast_checked: int = 0
    forecast_excused: int = 0
    forecast_misses: list[str] = field(default_factory=list)
    draws_checked: int = 0
    replays_checked: int = 0
    stalled: int = 0
    fights: Counter[str] = field(default_factory=Counter)
    acts: set[int] = field(default_factory=set)


@dataclass
class _EndTurn:
    """What an end-turn decision predicted for the next player turn."""

    round: int
    #: uid -> (move_id, certainty, bare intents) of step 1.
    forecast: dict[int, tuple[str, float, list[tuple[Any, Any, Any]]]]
    #: The whole draw pile's card ids, top first, from ``true_draw_top``.
    draw: list[str]
    #: Every card uid in the fight's piles.
    cards: list[int]
    #: Rows already in the run's combat log.
    log_len: int


def _check_state(cs: Any, tally: _Tally, deep: bool) -> dict[str, Any]:
    """Every getter on one state; ``deep`` adds the clone and re-deal checks."""
    before = _fingerprint(cs)
    wire = before[-1]

    tele = cs.telegraph()
    forecast = cs.peek_intents(3)
    top = cs.true_draw_top(10)
    encounter_id = cs.encounter_id
    assert cs.peek_intents(3) == forecast, "peek_intents is deterministic"
    assert _fingerprint(cs) == before, "a getter moved the state"

    enemies = wire["encounter"]
    assert len(tele) == len(enemies) == len(forecast)
    for i, enemy in enumerate(enemies):
        assert tele[i][:1] == enemy["intents"][:1], (i, tele[i], enemy["intents"])
        if len(tele[i]) > 1:
            tally.multi_intent += 1
        steps = forecast[i]
        assert len(steps) in (0, 3), steps
        assert [s["step"] for s in steps] == list(range(1, len(steps) + 1))
        certainties = [s["certainty"] for s in steps]
        assert all(0.0 <= c <= 1.0 for c in certainties), certainties
        assert certainties == sorted(certainties, reverse=True), certainties
        for s in steps:
            assert isinstance(s["move_id"], str)
            assert all("kind" in intent for intent in s["intents"])

    draw = wire["deck"]["draw_pile"]
    assert top == draw[:10]
    assert cs.true_draw_top(len(draw) + 5) == draw

    if deep:
        assert cs.clone().peek_intents(3) == forecast
        with pytest.raises(ValueError, match="deferred_draw_order"):
            cs.belief_view(5).true_draw_top(3)
        world = cs.sample_world(5)
        assert world.true_draw_top(3) == sts2_rs.state_to_dict(world)["deck"]["draw_pile"][:3]
        assert cs.belief_view(5).encounter_id == encounter_id
        assert world.encounter_id == encounter_id
        assert cs.clone().encounter_id == encounter_id
        # A clone that ran every getter plays on exactly like one that did not.
        actions = cs.legal_actions()
        if actions:
            touched, fresh = cs.clone(), cs.clone()
            touched.telegraph()
            touched.peek_intents(3)
            touched.true_draw_top(10)
            touched.apply(actions[0])
            fresh.apply(actions[0])
            assert _fingerprint(touched) == _fingerprint(fresh)
            tally.replays_checked += 1

    tally.states += 1
    return wire


def _check_forecast(
    end: _EndTurn, wire: dict[str, Any], cs: Any, rows: list[dict[str, Any]], tally: _Tally
) -> None:
    """The end-turn forecast against the telegraph the next player turn shows.

    ``rows`` is the combat log since the end turn. An enemy that hit 0 HP in
    between is skipped like one dead at the end turn: its machine is moot, and
    a revive (the Test Subject's next form) opens with that form's own move.
    """
    tele = cs.telegraph()
    killed = {
        r["receiver"].get("uid")
        for r in rows
        if r.get("kind") == "damage"
        and r.get("hp_after") == 0
        and r.get("receiver", {}).get("kind") == "monster"
    }
    for i, enemy in enumerate(wire["encounter"]):
        uid = enemy["uid"]
        if uid not in end.forecast or enemy.get("is_dead") or "move_state" not in enemy:
            continue
        if uid in killed:
            tally.forecast_excused += 1
            continue
        move_id, certainty, intents = end.forecast[uid]
        if certainty < 1.0:
            continue
        tally.forecast_checked += 1
        if enemy["move_state"] == move_id:
            # The turn shows the forecast move's telegraph, head and tail.
            if tele[i] and _bare(tele[i][0]) == intents[0]:
                assert [_bare(t) for t in tele[i]] == intents, (tele[i], intents)
            continue
        if enemy["kind"] in BOARD_DRIVEN:
            tally.forecast_excused += 1
            continue
        tally.forecast_misses.append(
            f"{cs.encounter_id} {enemy['kind']} round {end.round}: "
            f"forecast {move_id}, then {enemy['move_state']}"
        )


def _pile_cards(wire: dict[str, Any]) -> list[int]:
    deck = wire["deck"]
    piles = ("hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile")
    return sorted(c["stable_idx"] for pile in piles for c in deck.get(pile, []))


def _check_draw(end: _EndTurn, wire: dict[str, Any], rm: Any, tally: _Tally) -> None:
    """Every draw since the end turn took the old pile from its top, in order.

    Read from the combat log's ``card_drawn`` rows, so a drawn card that moved
    on (an auto-played Strike, a reshuffle once the pile ran out) still
    counts, and so do draws an enemy move or an end-of-turn play made.
    Skipped when the turn created or removed cards (a status card shuffled
    in, a Thieving Hopper's theft): then the pile itself changed.
    """
    if _pile_cards(wire) != end.cards:
        return
    rows = rm.combat_log()[end.log_len :]
    drawn = [r["card"] for r in rows if r["kind"] == "card_drawn"]
    n = min(len(drawn), len(end.draw))
    if n == 0:
        return
    assert drawn[:n] == end.draw[:n], (drawn, end.draw)
    tally.draws_checked += 1


def _check_new_fight(rm: Any, cs: Any, seen: Counter[tuple[int, str]], tally: _Tally) -> None:
    """A fight's id is the encounter the run generated for its room."""
    act = rm.act_index
    room = rm.current_room()
    kind = room["room_type"]
    generated = rm.generated_rooms()[act - 1]
    nth = seen[act, kind]
    if kind == "event":
        assert cs.encounter_id in EVENT_ENCOUNTERS, cs.encounter_id
    elif kind == "boss":
        want = generated["boss"] if nth == 0 else generated["second_boss"]
        assert cs.encounter_id == want, (act, nth, cs.encounter_id, generated)
    else:
        assert kind in ("monster", "elite"), kind
        ids = generated["normal" if kind == "monster" else "elite"]
        assert cs.encounter_id == ids[nth % len(ids)], (act, kind, nth, cs.encounter_id, ids)
    seen[act, kind] += 1
    tally.fights[kind] += 1
    tally.acts.add(act)


def _drive(seed: int, version: str, tally: _Tally) -> None:
    rm = _run(seed, version)
    assert rm.game_version == version
    rm.set_record_combat_log(True)
    rng = random.Random(seed)
    seen: Counter[tuple[int, str]] = Counter()
    fight: tuple[Any, str | None] | None = None  # (map coord, encounter id)
    end: _EndTurn | None = None
    for n in range(MAX_DECISIONS):
        if rm.is_over():
            return
        rm.set_player(hp=9999, max_hp=9999)
        actions = rm.available_actions()
        assert actions, f"{version} seed {seed}: no action in phase {rm.phase()}"
        if rm.phase() != "combat":
            stale = rm.combat_state()
            assert stale is None or stale.encounter_id is None
            fight = end = None
            rm.take_action(rng.choice(actions))
            continue

        cs = rm.combat_state()
        coord = rm.current_room()["map_coord"]
        if fight is None or fight[0] != coord:
            _check_new_fight(rm, cs, seen, tally)
            fight, end = (coord, cs.encounter_id), None
        assert cs.encounter_id == fight[1], "the id holds for the whole fight"

        wire = _check_state(cs, tally, deep=n % 5 == 0)
        if wire["round"] > MAX_ROUNDS:
            tally.stalled += 1
            return
        if (
            end is not None
            and wire["round"] == end.round + 1
            and wire["phase"] == "PlayerPlayPhase"
        ):
            _check_forecast(end, wire, cs, rm.combat_log()[end.log_len :], tally)
            _check_draw(end, wire, rm, tally)
        end = None

        action = rm.mcts_pick_action(sims=24, depth=6, seed=n, threads=1) or rng.choice(actions)
        if action.get("action") == "end_turn" and wire["phase"] == "PlayerPlayPhase":
            end = _EndTurn(
                round=wire["round"],
                forecast={
                    enemy["uid"]: (
                        steps[0]["move_id"],
                        steps[0]["certainty"],
                        [_bare(t) for t in steps[0]["intents"]],
                    )
                    for enemy, steps in zip(wire["encounter"], cs.peek_intents(1), strict=True)
                    # A dead enemy's machine is moot: a revive (the Test
                    # Subject's next form) swaps it out.
                    if steps and not enemy.get("is_dead")
                },
                draw=[c["id"] for c in cs.true_draw_top(len(wire["deck"]["draw_pile"]))],
                cards=_pile_cards(wire),
                log_len=len(rm.combat_log()),
            )
        rm.take_action(action)
    pytest.fail(f"{version} seed {seed}: the run did not end in {MAX_DECISIONS} decisions")


@pytest.mark.parametrize("version", VERSIONS)
def test_getters_on_live_a10_runs(version: str) -> None:
    tally = _Tally()
    for seed in SEEDS:
        _drive(seed, version, tally)
    summary = (
        f"{version}: {tally.states} states, acts {sorted(tally.acts)}, "
        f"fights {dict(tally.fights)}, {tally.multi_intent} multi-intent telegraphs, "
        f"{tally.forecast_checked} certain forecasts ({tally.forecast_excused} board-driven), "
        f"{tally.draws_checked} draws, {tally.replays_checked} replays, "
        f"{tally.stalled} stalled runs"
    )
    print(summary)
    assert not tally.forecast_misses, "\n".join([summary, *tally.forecast_misses])
    assert tally.states >= 1000, summary
    assert tally.acts == {1, 2, 3}, summary
    assert tally.fights["elite"] > 0 and tally.fights["boss"] >= 4, summary
    assert tally.multi_intent > 0, summary
    assert tally.forecast_checked > 200, summary
    assert tally.draws_checked > 200, summary
    assert tally.replays_checked > 100, summary


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    ("event", "option", "encounter"),
    [
        ("BattlewornDummy", 0, "BATTLEWORN_DUMMY_EVENT_ENCOUNTER"),
        ("DenseVegetation", 1, "DENSE_VEGETATION_EVENT_ENCOUNTER"),
        ("TheLanternKey", 1, "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER"),
    ],
)
def test_an_event_fight_reports_the_event_encounter(
    version: str, event: str, option: int, encounter: str
) -> None:
    rm = _run(7, version)
    assert rm.force_event(event) == event
    choices = [a for a in rm.available_actions() if a.get("action") == "event_choice"]
    rm.take_action(choices[option])
    for _ in range(3):  # through the event's one-option pages
        actions = rm.available_actions()
        if rm.phase() != "event" or len(actions) != 1:
            break
        rm.take_action(actions[0])
    assert rm.phase() == "combat"
    cs = rm.combat_state()
    assert cs.encounter_id == encounter
    assert (
        len(cs.telegraph()) == len(cs.peek_intents()) == len(sts2_rs.state_to_dict(cs)["encounter"])
    )


def test_scenario_states_carry_their_encounter_id() -> None:
    cs = sts2_rs.build_scenario_state(
        7,
        "Ironclad",
        ["DEFEND_IRONCLAD"] * 20,
        400,
        400,
        [],
        "TERROR_EEL_ELITE",
        10,
        game_version="v0.111.0",
    )
    assert cs.game_version == "v0.111.0"
    assert cs.encounter_id == "TERROR_EEL_ELITE"
    assert cs.belief_view(1).encounter_id == "TERROR_EEL_ELITE"
    assert cs.sample_world(1).encounter_id == "TERROR_EEL_ELITE"
    # The wire carries no id, so a rebuilt state has none.
    assert sts2_rs.state_from_dict(sts2_rs.state_to_dict(cs)).encounter_id is None
    # Before its first move the eel's step 1 is the move AFTER the telegraph.
    forecast = cs.peek_intents(3)
    assert [s["move_id"] for s in forecast[0]] == ["ThrashMove", "CRASH_MOVE", "ThrashMove"]
    assert [i["kind"] for i in forecast[0][0]["intents"]] == ["Attack", "Buff"]
    assert cs.peek_intents(0) == [[]]
    with pytest.raises(ValueError, match="limit"):
        cs.peek_intents(256)
    assert sts2_rs.RunManager(7, "Ironclad", game_version="v0.111.0").combat_state() is None
