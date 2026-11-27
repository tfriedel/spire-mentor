"""`ppo.e2e.forced_rollout`: the forcing rule, the combat-log counts and the paired stats."""

from __future__ import annotations

import pytest
import sts2_client
from ppo.e2e import factory
from ppo.e2e import forced_rollout as FR

NECRO = FR.TARGETS["Necrobinder"]
SILENT = FR.TARGETS["Silent"]


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _screen(ids: list[str], potion: bool = True) -> tuple[list[str], list[dict], list[int]]:
    """A CARD_REWARD screen: picks, skip, and a potion discard; one class per action."""
    acts = [{"action": "pick_card_reward", "idx": i} for i in range(len(ids))]
    acts.append({"action": "pick_card_reward", "idx": None})
    if potion:
        acts.append({"action": "discard_potion", "potion_slot": 0})
    return [f"CARD.{c}" for c in ids], acts, list(range(len(acts)))


def _choose(f: FR.Forcer, ids: list[str], logits: list[float], act: int = 1, deck=()):
    offered, acts, classes = _screen(ids)
    return f.choose(offered, acts, classes, logits, act=act, floor=3, deck=list(deck))


# Classes: 0 POKE, 1 SCOURGE, 2 NO_ESCAPE, 3 skip, 4 discard potion. The net leans POKE (p 0.51).
IDS = ["POKE", "SCOURGE", "NO_ESCAPE"]
LOGITS = [2.5, 0.5, 1.0, 2.0, -5.0]


def test_none_never_overrides_but_books_the_eligible_screen() -> None:
    f = FR.Forcer(FR.ARMS["none"], NECRO)
    c, ev = _choose(f, IDS, LOGITS)
    assert c == 0
    assert ev is not None and not ev["forced"] and ev["net"] == "POKE"


def test_a_screen_without_a_target_is_not_eligible() -> None:
    f = FR.Forcer(FR.ARMS["single"], NECRO)
    c, ev = _choose(f, ["POKE", "DEFY", "SNAP"], LOGITS)
    assert (c, ev, f.used) == (0, None, 0)


def test_single_forces_the_best_rated_target_once() -> None:
    f = FR.Forcer(FR.ARMS["single"], NECRO)
    c, ev = _choose(f, IDS, LOGITS)
    assert c == 2  # NO_ESCAPE (1.0) over SCOURGE (0.5)
    assert ev["forced"] and ev["override"] and ev["taken"] == "NO_ESCAPE"
    c, ev = _choose(f, IDS, LOGITS)
    assert c == 0 and not ev["forced"]


def test_a_forced_screen_the_net_agrees_with_uses_the_slot() -> None:
    f = FR.Forcer(FR.ARMS["single"], NECRO)
    c, ev = _choose(f, IDS, [0.0, 0.5, 4.0, 2.0, -5.0])
    assert c == 2 and ev["forced"] and not ev["override"]
    assert f.used == 1


def test_prefix3_forces_three_screens() -> None:
    f = FR.Forcer(FR.ARMS["prefix3"], NECRO)
    got = [_choose(f, IDS, LOGITS)[0] for _ in range(4)]
    assert got == [2, 2, 2, 0]


def test_late_waits_for_act_2() -> None:
    f = FR.Forcer(FR.ARMS["late"], NECRO)
    assert _choose(f, IDS, LOGITS, act=1)[0] == 0
    assert _choose(f, IDS, LOGITS, act=2)[0] == 2
    assert _choose(f, IDS, LOGITS, act=3)[0] == 0


def test_deviate_takes_the_second_non_target_option() -> None:
    f = FR.Forcer(FR.ARMS["deviate"], NECRO)
    c, ev = _choose(f, IDS, LOGITS)
    assert c == 3  # non-targets by logit: POKE 2.5, skip 2.0, (potion is not a pick)
    assert ev["forced"] and ev["override"] and ev["taken"] is None
    assert _choose(f, IDS, LOGITS)[0] == 0


def test_skiponly_forces_only_where_the_net_skips_or_is_unsure() -> None:
    f = FR.Forcer(FR.ARMS["skiponly"], NECRO)
    sure = [9.0, 0.5, 1.0, 2.0, -5.0]  # POKE at p > 0.99
    assert _choose(f, IDS, sure)[0] == 0 and f.used == 0
    assert _choose(f, IDS, LOGITS)[0] == 2  # POKE at p 0.51: unsure
    f = FR.Forcer(FR.ARMS["skiponly"], NECRO)
    skips = [0.0, 0.5, 1.0, 9.0, -5.0]
    assert _choose(f, IDS, skips)[0] == 2


def test_silent_needs_two_poison_sources_in_the_deck() -> None:
    f = FR.Forcer(FR.ARMS["single"], SILENT)
    ids = ["BACKFLIP", "ACCELERANT", "MIRAGE"]
    assert _choose(f, ids, LOGITS, deck=["SNAKEBITE"])[1] is None
    c, ev = _choose(f, ids, LOGITS, deck=["SNAKEBITE", "NOXIOUS_FUMES_CARD", "STRIKE_SILENT"])
    assert c == 2 and ev["taken"] == "MIRAGE"


@pytest.mark.parametrize("char", sorted(FR.TARGETS))
def test_target_ids_are_cards_and_read_back_from_the_deck(char: str) -> None:
    """Every target id is a card the engine adds, and `deck_ids` spells it the same way."""
    t = FR.TARGETS[char]
    rm = factory.new_run(160_000_500, character=char)
    for cid in sorted(t.payoff | t.sources):
        rm.force_add_card(cid)
    assert t.payoff | t.sources <= set(FR.deck_ids(rm))


def _dmg(uid: int, before: int, after: int, dealer: dict | None) -> dict:
    return {
        "kind": "damage",
        "receiver": {"id": "M", "kind": "monster", "uid": uid},
        "dealer": dealer,
        "hp_before": before,
        "hp_after": after,
    }


def _power(uid: int, power: str, amount: int) -> dict:
    rc = {"id": "M", "kind": "monster", "uid": uid}
    return {"kind": "power", "power": power, "amount": amount, "receiver": rc}


def test_fight_stats_count_doom_and_poison_kills() -> None:
    me = {"id": "NECROBINDER", "kind": "player", "uid": None}
    log = [
        _power(2, "DOOM", 13),
        _power(3, "POISON", 5),
        _dmg(3, 9, 4, None),  # poison tick
        _dmg(3, 4, 0, None),  # poison lethal
        _dmg(2, 20, 0, {"id": "M", "kind": "monster", "uid": 2}),  # Doom kill: its own tick
        _dmg(4, 6, 0, me),  # an ordinary kill
        _dmg(5, 30, 0, {"id": "M", "kind": "monster", "uid": 5}),  # self-kill without Doom
    ]
    plays = [{"id": "SCOURGE"}, {"id": "POKE"}, {"id": "SCOURGE"}]
    s = FR.FightStats()
    s.add(plays, log, NECRO.payoff, dropped=0)
    row = s.row()
    assert row["plays"] == {"SCOURGE": 2}
    assert (row["doom_applied"], row["doom_kills"]) == (13, 1)
    assert (row["poison_applied"], row["poison_damage"], row["poison_kills"]) == (5, 9, 1)
    assert row["fights"] == 1


def test_paired_counts_discordant_seeds() -> None:
    base = [True] * 50 + [False] * 50
    arm = [True] * 40 + [False] * 10 + [True] * 30 + [False] * 20
    x = FR.paired(base, arm, boot=2000)
    assert (x["only_base"], x["only_arm"]) == (10, 30)
    assert x["delta"] == pytest.approx(0.2)
    assert x["boot95"][0] < 0.2 < x["boot95"][1]
    assert x["ci95"][0] < 0.2 < x["ci95"][1]
    assert x["mcnemar_p"] < 0.01
    assert FR.mcnemar_p(5, 5) == pytest.approx(1.0)
