"""The oracle rolls Rolling Boulder over the enemies one at a time, as the real game does.

Under TestMode, ``RollingBoulderPower.AfterPlayerTurnStart`` deals one
multi-target ``DoDamage(HittableEnemies)``, and that call's
``Kill(killedCreatures)`` waits until every enemy has been hit. The real game
takes the other branch: ``NRollingBoulderVfx`` rolls the boulder left to right
and signals ``HitCreature`` as it reaches each enemy, and each signal starts
its own single-target ``DoDamage``. So an enemy the boulder fells dies, and its
death hooks run, before the boulder reaches the next one. The harness patch
(``RollingBoulderAfterPlayerTurnStartPrefix``) does the same, one
``CreatureCmd.Damage`` per enemy in order.

Gremlin Horn's ``AfterDeath`` energy shows the order in the oracle's state
trace: it lands between the felled enemy's hit and the next enemy's hit.
"""

from __future__ import annotations

from typing import Any

import pytest

# Ironclad has 3 energy, so the opening hand plays one boulder. Turn by turn it
# deals 10, 15, 20, ...: the two Toadpoles (21-26 HP) fall on the same pass,
# the second turn's.
_DECK = ["ROLLING_BOULDER+1"] * 5


def _boulder_pass(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The enemy-HP and Gremlin Horn energy events of the end-turn segments.

    Nothing but the boulder damages an enemy here. A felled enemy leaves the
    list, so the next one is ``monsters[0]`` again: the events are kept in
    order rather than told apart by slot.
    """
    return [
        e
        for seg in segments
        for e in seg.get("events") or []
        if (e["k"] == "hp" and str(e.get("who", "")).startswith("monsters"))
        or (e["k"] == "energy" and "GremlinHorn" in str(e.get("src") or ""))
    ]


@pytest.mark.oracle
def test_a_felled_enemy_dies_before_the_boulder_reaches_the_next(game) -> None:
    state = game.start(character="Ironclad", seed="boulder1")
    game.skip_neow(state)
    r = game.set_player(hp=999, max_hp=999, deck=_DECK, relics=["GREMLIN_HORN"], potions=[])
    assert r.get("type") != "error", r
    state = game.enter_room("combat", encounter="TOADPOLES_WEAK")
    assert state["decision"] == "combat_play", state
    assert game.send({"cmd": "trace", "on": True}).get("enabled") is True

    state = game.play_card(state["hand"][0]["index"], None)

    seq = game.get_state().get("trace_seq") or 0
    for _ in range(8):
        if state.get("decision") != "combat_play":
            break
        state = game.end_turn()
        segs = game.send({"cmd": "get_trace", "since": seq}).get("segments") or []
        seq = game.get_state().get("trace_seq") or seq
        events = _boulder_pass(segs)
        kills = [i for i, e in enumerate(events) if e["k"] == "hp" and e["new"] == 0]
        if not kills:
            continue
        first = kills[0]
        later_hits = [i for i, e in enumerate(events) if e["k"] == "hp" and i > first]
        assert later_hits, f"the pass felled the last enemy first: {events}"
        between = events[first + 1 : later_hits[0]]
        assert any(e["k"] == "energy" for e in between), (
            f"Gremlin Horn's energy for the felled enemy must land before the next hit: {events}"
        )
        return
    pytest.fail("no boulder pass felled an enemy")
