"""The enemy move inferrer against the host's own move ids (v0.111.0).

The fixture holds full-state host observations of long fights (bosses and
elites included) with ``move_state``/``ai`` stripped, and the true move ids
beside them (``bench_agents.move_infer``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

from bench_agents.move_infer import MoveTracker  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "bench_agents" / "move_truth_v0111.json"
ALIASES = Path(__file__).parents[2] / "sts2-rs" / "crates" / "sts2-wire" / "src" / "state_dict.rs"


def _aliases() -> dict[tuple[str, str], str]:
    text = ALIASES.read_text(encoding="utf-8")
    pat = r'\(\s*MonsterKind::(\w+),\s*"(\w+)",\s*"(\w+)",?\s*\)'
    return {(m[1], m[2]): m[3] for m in re.finditer(pat, text)}


def _norm(aliases: dict, kind: str, move: str | None) -> str | None:
    if move is None:
        return None
    # The engine's own relaxed matcher (`force_current_state_by_name_relaxed`).
    move = aliases.get((kind, move), move)
    return move.removesuffix("_MOVE").replace("_MOVE_", "_")


def test_moves_are_inferred_from_the_shown_intents_alone():
    from sts2_hint import adapter

    aliases = _aliases()
    right = total = 0
    wrong = []
    for fight in json.loads(FIXTURE.read_text(encoding="utf-8")):
        tracker = MoveTracker(fight["character"], fight["ascension"], "v0.111.0")
        for obs, truth in zip(fight["observations"], fight["truth"], strict=True):
            assert all("move_state" not in e for e in obs["enemies"])
            got = tracker.infer(obs)
            wire = adapter.translate(
                {**obs, "game_version": "v0.111.0"},
                character=fight["character"],
                ascension=fight["ascension"],
            )
            live = wire["encounter"][: len(truth)]
            for enc, t, g in zip(live, truth, got, strict=True):
                if t is None:
                    continue
                total += 1
                if _norm(aliases, enc["kind"], t) == _norm(aliases, enc["kind"], g["move_state"]):
                    right += 1
                else:
                    wrong.append((fight["encounter"], t, g["move_state"]))
    # 98% over every v0.111.0 encounter (2026-11-20); these long fights are
    # the hard end of it.
    assert total > 150
    assert right / total >= 0.9, (right, total, wrong[:10])


def test_an_enemy_is_tracked_by_its_combat_id_through_another_ones_death():
    # GH #1651: `index` is the position among the living, so it shifts when
    # an enemy before it dies; `combat_id` does not.
    fight = next(
        f
        for f in json.loads(FIXTURE.read_text(encoding="utf-8"))
        if len(f["observations"][0]["enemies"]) >= 3
    )
    obs = json.loads(json.dumps(fight["observations"][0]))
    for e in obs["enemies"]:
        e["combat_id"] = 10 + e["index"]
    tracker = MoveTracker(fight["character"], fight["ascension"], "v0.111.0")
    tracker.infer(obs)
    survivor = tracker.enemies[12]
    # The first enemy dies: the others move up a slot.
    later = {**obs, "enemies": obs["enemies"][1:]}
    for i, e in enumerate(later["enemies"]):
        e["index"] = i
    tracker.infer(later)
    assert tracker.enemies[12] is survivor


@pytest.mark.parametrize("encounter", ["QUEEN_BOSS", "TEST_SUBJECT_BOSS"])
def test_an_act3_boss_fight_is_tracked_on_its_own_build(game, encounter):
    """GH #1933: the catalog is the fight's build's (v0.111.0 renamed the
    TorchHeadAmalgam's moves), and the Queen's CardDebuff opening matches.
    A move the fight's machine lacks left the rebuilt enemy AI-less, rolling
    a stand-in Attack 5 every turn."""
    import json as _json

    from test_referee_rng import Host

    aliases = _aliases()
    state = game.start(character="Ironclad", seed="MVBOSS", ascension=10)
    game.skip_neow(state)
    game.set_player(hp=999, max_hp=999, deck=["DEFEND_IRONCLAD"] * 10, relics=[], potions=[])
    full = game.enter_room("combat", encounter=encounter)
    tracker = MoveTracker("Ironclad", 10, "v0.111.0")
    wrong, total = [], 0
    with Host(referee=True) as host:
        for _ in range(8):
            if full.get("decision") != "combat_play":
                break
            view = _json.loads(host.send(cmd="referee_project", payload=full)["view"])
            got = tracker.infer(view)
            for enemy, guess in zip(full["enemies"], got, strict=True):
                kind = enemy["id"].title().replace("_", "")
                total += 1
                if _norm(aliases, kind, enemy["move_state"]) != _norm(
                    aliases, kind, guess["move_state"]
                ):
                    wrong.append((enemy["id"], enemy["move_state"], guess["move_state"]))
            full = game.end_turn()
    assert total >= 8
    assert not wrong, wrong


def test_a_reviving_corpse_is_rebuilt_on_its_revive_move(game):
    """GH #1933: the Test Subject's corpse sits in ``dead_enemies`` until its
    RESPAWN. Rebuilt without a move, it came back AI-less, rolling a
    stand-in Attack 5 where the boss telegraphs its next form's opening."""
    import json as _json

    from sts2_hint import adapter
    from test_referee_rng import Host

    state = game.start(character="Ironclad", seed="MVREVIVE", ascension=10)
    game.skip_neow(state)
    game.set_player(hp=9999, max_hp=9999, deck=["BLUDGEON"] * 10, relics=[], potions=[])
    full = game.enter_room("combat", encounter="TEST_SUBJECT_BOSS")
    tracker = MoveTracker("Ironclad", 10, "v0.111.0")
    revived = 0
    with Host(referee=True) as host:
        for _ in range(30):
            if full.get("decision") != "combat_play":
                break
            while full.get("decision") == "combat_play" and full.get("energy", 0) >= 3:
                # PainfulStabs shuffles Wounds in; play a Bludgeon.
                ids = [c.get("id") for c in full.get("hand") or []]
                if "CARD.BLUDGEON" not in ids:
                    break
                full = game.play_card(ids.index("CARD.BLUDGEON"), 0)
            if full.get("decision") != "combat_play":
                break
            view = _json.loads(host.send(cmd="referee_project", payload=full)["view"])
            alive, dead = tracker.infer_board(view)
            board = {**view, "game_version": "v0.111.0"}
            for enemy, guess in [
                *zip(board["enemies"], alive),
                *zip(board.get("dead_enemies") or [], dead),
            ]:
                enemy["move_state"], enemy["ai"] = guess["move_state"], guess["ai"]
            if not board.get("dead_enemies"):
                full = game.end_turn()
                continue
            assert [g["move_state"] for g in dead] == ["RESPAWN"]
            rebuilt = sts2_rs.state_from_dict(
                adapter.translate(board, character="Ironclad", ascension=10)
            )
            rebuilt.apply({"action": "end_turn"})
            boss = sts2_rs.state_to_dict(rebuilt)["encounter"][0]
            full = game.end_turn()
            assert boss["move_state"] is not None, "revived AI-less"
            revived += 1
    assert revived == 2
