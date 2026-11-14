"""The "chars" fill event (GH #1373, SCHEMA.md section 13): every character's mechanics.

The schema was frozen for Ironclad. A row of another character's run carries its
character (`X_CHAR`), the pet (`ALLY`: Osty), the orb belt (`ORB`) and the doomed flag
(`E_DOOMED`); the tactical HP loss counts Osty and the Frost passives. An Ironclad row
carries none of it, the schema hash does not move and a net that does not declare the
event keeps its inputs at zero weight.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import sts2_client
import sts2_rs
import torch
from ppo.e2e import contracts as C
from ppo.e2e import env as E
from ppo.e2e import factory
from ppo.e2e import fillcheck as FC
from ppo.e2e import net as N
from ppo.e2e import tactical as TC
from ppo.e2e import tokens as K
from ppo.e2e import vocab as V
from ppo.e2e.rsenv_lockstep import Lockstep, run_seeds

OTHERS = ("Silent", "Defect", "Regent", "Necrobinder")
SEED = 1_300_000_500  # the throwaway block (plan §3.1)


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _play(character: str, seeds: range, cap: int = 600) -> list[tuple[Any, K.Tok]]:
    """Random-play decisions of `character`'s runs with their rows."""
    out: list[tuple[Any, K.Tok]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "CHARACTER", character)
        for seed in seeds:
            env = E.E2EEnv(seed)
            d = env.reset()
            rng = random.Random(seed)
            n = 0
            while isinstance(d, E.E2EDecision) and n < cap:
                out.append((d, K.encode(d)))
                n += 1
                d = env.step(rng.randrange(d.n_classes))
    return out


def _player(d: Any) -> dict[str, Any]:
    return sts2_rs.state_to_dict(d.bv)["player"]


# ── the frozen schema ────────────────────────────────────────────────────────


def test_the_event_takes_reserved_slots_and_moves_no_hash() -> None:
    assert C.FILL_CHARS in C.FILL_EVENTS
    assert C.SECTIONS[C.SEC["ALLY"]] == "RES_44" and C.SECTIONS[C.SEC["ORB"]] == "RES_45"
    assert C.X_CAT[C.XC["X_CHAR"] - C.SCTX_N_CAT].name == "X_RES0"
    for alias, spare in C.CHARS_CONT.items():
        col = C.X_CONT[C.XF[alias] - C.SCTX_N_CONT]
        assert (col.name, col.status) == (spare, "reserved:spare")
    # The hashed tables are the frozen ones: no tensor grows, no name changes.
    spec = C.schema_spec()
    assert "RES_44" in spec["sections"] and "ALLY" not in spec["sections"]
    assert "RES_44" not in spec["section_live"] and "RES_45" not in spec["section_live"]
    assert C.XF["E_DOOMED"] not in spec["section_live"]["ENEMY"][1]
    assert C.XC["X_CHAR"] not in spec["section_live"]["RUNCTX"][0]
    assert (C.N_CAT, C.N_CONT, C.N_SECTION) == (32, 210, 64)
    # The working table adds the event's slots to the frozen one.
    assert C.XF["E_DOOMED"] in C.SECTION_LIVE[C.SEC["ENEMY"]][1]
    assert C.XC["X_CHAR"] in C.SECTION_LIVE[C.SEC["RUNCTX"]][0]
    assert not C.CHARS_SECTION_IDS & C.RESERVED_SECTIONS
    assert C.CHAR_ID[C.BASE_CHARACTER] == 1 and 0 not in C.CHAR_ID.values()
    assert max(C.CHAR_ID.values()) < C.X_CAT[C.XC["X_CHAR"] - C.SCTX_N_CAT].size
    assert set(factory.CHARACTERS) <= set(C.CHARACTERS)


def test_an_ironclad_row_carries_none_of_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    assert not factory.fill_chars()
    cols = FC.reserved_cols(C.FILL_CHARS)
    rows = _play("Ironclad", range(SEED, SEED + 3), cap=150)
    assert sum(d.is_combat for d, _ in rows) > 20
    for _d, t in rows:
        assert not FC._nonzero(t.cat, t.cont.astype(np.float32), cols).any()
        assert not np.isin(t.sec, list(C.CHARS_SECTION_IDS)).any()
        assert not t.is_combat or t.sctx_lo in (7, 8)


def test_a_declared_ironclad_launch_shows_orbs_and_doom(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Ironclad can hold orbs (Potion of Capacity, Infused Core) and face Doom
    (Potion of Doom). Its rows show them once the launch declares the event
    (`factory.set_fill_chars`, which the learner calls from `--filled`), and not before."""
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    rows = [(d, t) for d, t in _play("Ironclad", range(SEED, SEED + 2), cap=120) if d.is_combat]
    d, _t = next((d, t) for d, t in rows if any(not m.get("is_dead") for m in d.snap["enemies"]))
    j = next(j for j, m in enumerate(d.snap["enemies"]) if not m.get("is_dead"))
    hp = d.snap["enemies"][j]["hp"]
    d.snap["enemies"][j]["powers"].append(
        {"name": "DOOM", "amount": hp, "display_amount": hp, "counter": 0}
    )
    d.d_bv["player"]["orbs"] = [{"index": 0, "type": "Dark", "passive": 6, "evoke": 6}]
    d.d_bv["player"]["orb_slots"] = 2
    off = K.encode(d, priv=False)
    assert off.cat[0, C.XC["X_CHAR"]] == 0
    assert not np.isin(off.sec, list(C.CHARS_SECTION_IDS)).any()
    assert not off.cont[:, C.XF["E_DOOMED"]].any()
    monkeypatch.setattr(factory, "CHARACTER", "Ironclad")
    factory.set_fill_chars(True)
    try:
        assert factory.fill_chars()
        on = K.encode(d, priv=False)
    finally:
        factory.set_fill_chars(False)
    assert on.cat[0, C.XC["X_CHAR"]] == C.CHAR_ID["Ironclad"]
    orb = np.flatnonzero(on.sec == C.SEC["ORB"])
    assert len(orb) == 2 and on.sctx_lo == off.sctx_lo + 2
    v = V.load()
    assert on.cat[orb, C.XC["X_ID_MAIN"]].tolist() == [v.id("orb", "Dark"), v.id("orb", "Empty")]
    enemy = np.flatnonzero(on.sec == C.SEC["ENEMY"])
    assert on.cont[enemy, C.XF["E_DOOMED"]].tolist() == [float(k == j) for k in range(len(enemy))]
    C.validate_row_batch(K.row_batch([on], run_uid=[1]))


def test_the_rust_env_agrees_on_a_declared_ironclad_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(factory.CHARS_ENV, "1")
    ls = Lockstep(f9_p=0.0)
    run_seeds(ls, SEED, 3, pinned=False, practice=0)
    run_seeds(ls, SEED + 50, 1, pinned=True, practice=0)
    assert ls.stats["rows"] > 300
    assert ls.mismatches == 0, {"stats": dict(ls.stats), "examples": ls.examples[:5]}
    rows = _play("Ironclad", range(SEED, SEED + 1), cap=60)
    assert all(t.cat[0, C.XC["X_CHAR"]] == C.CHAR_ID["Ironclad"] for _d, t in rows)


# ── rows of the other characters ─────────────────────────────────────────────


@pytest.mark.parametrize("character", OTHERS)
def test_every_character_tokenizes_and_validates(character: str) -> None:
    rows = _play(character, range(SEED, SEED + 4), cap=250)
    toks = [t for _, t in rows]
    assert sum(t.is_combat for t in toks) > 20 and any(not t.is_combat for t in toks)
    C.validate_row_batch(K.row_batch(toks, run_uid=[1] * len(toks)))
    for d, t in rows:
        C.check_compact_lossless(t.cat, t.cont, t.sec)
        assert t.cat[0, C.XC["X_CHAR"]] == C.CHAR_ID[character]
        ext = int(np.isin(t.sec, list(C.CHARS_SECTION_IDS)).sum())
        if not d.is_combat:
            assert ext == 0
            continue
        nb = int((t.sec == C.SEC["BOSS"]).sum())
        assert t.sctx_lo == 6 + nb + ext
        p = _player(d)
        slots = max(int(p.get("orb_slots") or 0), len(p.get("orbs") or []))
        assert int((t.sec == C.SEC["ALLY"]).sum()) == int("osty" in p)
        assert int((t.sec == C.SEC["ORB"]).sum()) == slots
        loss = TC.hp_loss(d.snap, end_block=TC.end_turn_block(p), ally_hp=TC.ally_hp_of(p))
        got = float(t.cont[t.sctx_lo, C.XF["S_HP_LOSS_END"]])
        assert got == pytest.approx(loss / 50.0, rel=2e-3, abs=1e-3)


def test_osty_is_a_token_with_its_hp() -> None:
    rows = _play("Necrobinder", range(SEED, SEED + 6), cap=250)
    seen = 0
    for d, t in rows:
        if not d.is_combat:
            continue
        o = _player(d).get("osty")
        ally = np.flatnonzero(t.sec == C.SEC["ALLY"])
        if o is None:
            assert len(ally) == 0
            continue
        (i,) = ally
        assert t.cat[i, C.XC["X_ID_MAIN"]] == V.load().id("monster", "Osty")
        hp, mhp = float(o["hp"]), max(1.0, float(o["max_hp"]))
        assert float(t.cont[i, C.SCTX_C["E_HP"]]) == pytest.approx(hp / 200.0, abs=1e-3)
        assert float(t.cont[i, C.SCTX_C["E_MAXHP"]]) == pytest.approx(mhp / 200.0, abs=1e-3)
        assert float(t.cont[i, C.SCTX_C["E_HPF"]]) == pytest.approx(hp / mhp, abs=1e-3)
        assert bool(t.cont[i, C.SCTX_C["E_ALIVE"]]) == bool(o["alive"])
        seen += 1
    # Bound Phylactery summons Osty at the fight start and again every turn, so a
    # decision of random play always sees it alive; the dead Osty is the dict test's.
    assert seen > 20


def test_the_orb_belt_is_one_token_per_slot() -> None:
    rows = _play("Defect", range(SEED, SEED + 6), cap=250)
    kinds: set[str] = set()
    empty = 0
    v = V.load()
    for d, t in rows:
        if not d.is_combat:
            continue
        p = _player(d)
        orbs = p.get("orbs") or []
        orb = np.flatnonzero(t.sec == C.SEC["ORB"])
        assert len(orb) == max(int(p.get("orb_slots") or 0), len(orbs))
        for k, i in enumerate(orb):
            assert t.cat[i, C.XC["X_POS"]] == k + 1
            if k >= len(orbs):
                assert t.cat[i, C.XC["X_ID_MAIN"]] == v.id("orb", "Empty")
                assert not t.cont[i].any()
                empty += 1
                continue
            o = orbs[k]
            kinds.add(o["type"])
            assert t.cat[i, C.XC["X_ID_MAIN"]] == v.id("orb", o["type"]) > 11
            passive = float(t.cont[i, C.XF["C_ORB_PASSIVE"]])
            assert passive == pytest.approx(o["passive"] / 10.0, abs=1e-3)
            evoke = float(t.cont[i, C.XF["C_ORB_EVOKE"]])
            assert evoke == pytest.approx(o["evoke"] / 20.0, abs=1e-3)
    assert {"Lightning", "Frost"} <= kinds and empty > 0, (kinds, empty)


def test_the_chars_tokens_from_a_player_dict() -> None:
    """Osty alive and dead, a mixed belt with Focus in the numbers, empty slots."""
    v = V.load()
    buf = K._Buf()
    K._chars_tokens(
        buf,
        {
            "osty": {"alive": True, "hp": 7, "max_hp": 12, "block": 0},
            "orbs": [
                {"index": 0, "type": "Frost", "passive": 4, "evoke": 7},
                {"index": 1, "type": "Dark", "passive": 8, "evoke": 22},
            ],
            "orb_slots": 4,
        },
    )
    cat, cont, sec, _flags = buf.arrays()
    assert sec.tolist() == [C.SEC["ALLY"]] + [C.SEC["ORB"]] * 4
    assert cont[0, C.SCTX_C["E_HPF"]] == pytest.approx(7 / 12)
    assert cont[0, C.SCTX_C["E_ALIVE"]] == pytest.approx(1.0)
    ids = cat[1:, C.XC["X_ID_MAIN"]].tolist()
    assert ids == [v.id("orb", "Frost"), v.id("orb", "Dark")] + [v.id("orb", "Empty")] * 2
    assert cat[1:, C.XC["X_POS"]].tolist() == [1, 2, 3, 4]
    assert cont[1:, C.XF["C_ORB_PASSIVE"]].tolist() == pytest.approx([0.4, 0.8, 0.0, 0.0])
    assert cont[1:, C.XF["C_ORB_EVOKE"]].tolist() == pytest.approx([0.35, 1.1, 0.0, 0.0])
    dead = K._Buf()
    K._chars_tokens(dead, {"osty": {"alive": False, "hp": 0, "max_hp": 12, "block": 0}})
    _cat, cont, sec, _flags = dead.arrays()
    assert sec.tolist() == [C.SEC["ALLY"]]
    assert not cont[0, C.SCTX_C["E_ALIVE"]] and not cont[0, C.SCTX_C["E_HP"]]
    assert cont[0, C.SCTX_C["E_MAXHP"]] == pytest.approx(12 / 200)
    none = K._Buf()
    K._chars_tokens(none, {})
    assert len(none) == 0


# ── tactical: the decomp's rules, by hand ────────────────────────────────────


def _snap(block: int, enemies: list[dict[str, Any]]) -> dict[str, Any]:
    return {"player": {"block": block, "energy": 0}, "hand": [], "enemies": enemies}


def _attacker(hp: int, dmg: int, times: int, powers: Any = ()) -> dict[str, Any]:
    intent = {"kind": "Attack", "amount": dmg, "effective_amount": dmg, "times": times}
    return {"hp": hp, "block": 0, "is_dead": False, "powers": list(powers), "intents": [intent]}


def test_osty_takes_the_unblocked_damage_first() -> None:
    """`CreatureCmd.Damage`: the owner's block absorbs first; a living Osty then takes
    the unblocked damage of a powered attack (`DieForYouPower`), and the overkill
    passes on to the owner."""
    s = _snap(5, [_attacker(30, 6, 3)])  # 18 incoming, 13 unblocked
    assert TC.hp_loss(s) == pytest.approx(13.0)
    alive = {"osty": {"alive": True, "hp": 8, "max_hp": 10}}
    dead = {"osty": {"alive": False, "hp": 0, "max_hp": 10}}
    assert [TC.ally_hp_of(p) for p in (alive, dead, {})] == [8.0, 0.0, 0.0]
    assert TC.hp_loss(s, ally_hp=TC.ally_hp_of(alive)) == pytest.approx(5.0)  # Osty dies, 5 pass on
    assert TC.hp_loss(s, ally_hp=20.0) == pytest.approx(0.0)
    assert TC.hp_loss(s, ally_hp=TC.ally_hp_of(dead)) == pytest.approx(13.0)


def test_frost_passives_block_before_the_enemies_act() -> None:
    """`OrbQueue.BeforeTurnEnd` -> `FrostOrb.Passive`: unpowered block of `PassiveVal`
    (Focus included), at the end of the player's turn."""
    belt = {
        "orbs": [
            {"type": "Frost", "passive": 2, "evoke": 5},
            {"type": "Lightning", "passive": 3, "evoke": 8},
            {"type": "Frost", "passive": 4, "evoke": 7},
        ]
    }
    assert [TC.end_turn_block(belt), TC.end_turn_block({})] == [6.0, 0.0]
    s = _snap(5, [_attacker(30, 6, 3)])
    assert TC.hp_loss(s, end_block=TC.end_turn_block(belt)) == pytest.approx(7.0)
    md, kill, loss = TC.tactical(s, [], end_block=6.0, ally_hp=4.0)
    assert (md, kill, loss) == ([0.0], [0.0], 3.0)


def test_doomed_is_doom_at_least_hp() -> None:
    """`DoomPower.IsOwnerDoomed`: `Owner.CurrentHp <= Amount`."""

    def doom(n: int) -> list[dict[str, Any]]:
        return [{"name": "DOOM", "amount": n}]

    dead = _attacker(9, 1, 1, doom(50)) | {"is_dead": True}
    s = _snap(
        0,
        [
            _attacker(10, 1, 1, doom(10)),
            _attacker(10, 1, 1, doom(9)),
            _attacker(10, 1, 1, [{"name": "WEAK", "amount": 99}]),
            dead,
        ],
    )
    assert TC.doomed(s) == [1.0, 0.0, 0.0, 0.0]


def test_the_doomed_flag_reaches_the_enemy_token() -> None:
    """A Necrobinder combat row whose enemy carries Doom >= its HP: `E_DOOMED` on that
    ENEMY token only."""
    rows = [(d, t) for d, t in _play("Necrobinder", range(SEED, SEED + 2), cap=120) if d.is_combat]
    d, _t = next((d, t) for d, t in rows if any(not m.get("is_dead") for m in d.snap["enemies"]))
    j = next(j for j, m in enumerate(d.snap["enemies"]) if not m.get("is_dead"))
    hp = d.snap["enemies"][j]["hp"]
    d.snap["enemies"][j]["powers"].append(
        {"name": "DOOM", "amount": hp, "display_amount": hp, "counter": 0}
    )
    t = K.encode(d, priv=False)
    enemy = np.flatnonzero(t.sec == C.SEC["ENEMY"])
    flags = t.cont[enemy, C.XF["E_DOOMED"]].tolist()
    assert flags == [float(k == j) for k in range(len(enemy))]


# ── the net: the identity rule ───────────────────────────────────────────────


def test_a_net_without_the_event_keeps_its_inputs_at_zero() -> None:
    old, new = N.layout(), N.layout([C.FILL_CHARS])
    slots = C.FILL_SLOTS[C.FILL_CHARS]
    assert set(slots["sections"]) <= set(old.reserved_sections)
    assert not set(slots["sections"]) & set(new.reserved_sections)
    assert set(slots["cont"]) <= set(old.zero_cont)
    assert not set(slots["cont"]) & set(new.zero_cont)
    k = old.small_cols.index(C.XC["X_CHAR"])
    assert not old.small_live[k] and new.small_live[k]
    cfg = N.NetConfig(dim=64, n_layers=1, n_heads=1, ffn=128)
    a, b = N.E2ENet(cfg, seed=0), N.E2ENet(cfg, seed=0, filled=[C.FILL_CHARS])
    off, size = old.small_offsets[k], old.small_sizes[k]
    for s in slots["sections"]:
        assert not a.sec.weight[s].any() and b.sec.weight[s].any()
    assert not a.small.weight[off : off + size].any()
    assert b.small.weight[off + 1 : off + size].any()
    cols = list(slots["cont"])
    assert not a.num.inp.weight[:, cols].any() and b.num.inp.weight[:, cols].any()


def test_x_char_and_e_doomed_change_no_output_of_an_undeclared_net() -> None:
    """The identity rule on the event's columns: a net that does not declare "chars"
    (an Ironclad checkpoint) gives the same logits whatever X_CHAR and E_DOOMED hold."""
    rows = [t for d, t in _play("Necrobinder", range(SEED, SEED + 2), cap=80) if d.is_combat]
    net = N.E2ENet(N.NetConfig(dim=64, n_layers=2, n_heads=2, ffn=128), seed=3).eval()

    def logits(t: K.Tok) -> torch.Tensor:
        rb = K.row_batch([t], run_uid=[1])
        with torch.no_grad():
            return net.policy(N.to_batch(rb, "cpu"), bf16=False)

    for t in rows[:6]:
        base = logits(t)
        assert t.cat[0, C.XC["X_CHAR"]] == C.CHAR_ID["Necrobinder"]
        t.cat[0, C.XC["X_CHAR"]] = 0
        enemy = np.flatnonzero(t.sec == C.SEC["ENEMY"])
        t.cont[enemy, C.XF["E_DOOMED"]] = 1.0
        assert torch.equal(base, logits(t))


# ── the fill check ───────────────────────────────────────────────────────────


def _sample(character: str, n: int = 14) -> dict[str, np.ndarray]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "CHARACTER", character)
        rb, _terms = FC.sample_rows(n, SEED)
    return rb


def test_the_fill_check_ties_the_event_to_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    wp = list(FC.wheel_filled())
    necro = _sample("Necrobinder")
    undeclared = FC.check(necro, wp)
    assert C.FILL_CHARS in undeclared.stray
    assert any("chars is not declared" in p for p in undeclared.definite())
    declared = FC.check(necro, [*wp, C.FILL_CHARS])
    assert declared.char_rows == declared.rows and not declared.definite(), declared.problems
    assert C.FILL_CHARS not in declared.unchecked
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    ironclad = _sample("Ironclad", 6)
    assert not FC.check(ironclad, wp).definite()
    wrong = FC.check(ironclad, [*wp, C.FILL_CHARS])
    assert wrong.char_rows == 0
    assert any("chars declared but X_CHAR" in p for p in wrong.definite())


def test_the_factory_reads_the_character_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(factory.CHARACTER_ENV, raising=False)
    assert factory._character() == "Ironclad"
    monkeypatch.setenv(factory.CHARACTER_ENV, "necrobinder")
    assert factory._character() == "Necrobinder"
    monkeypatch.setenv(factory.CHARACTER_ENV, "Watcher")
    with pytest.raises(ValueError, match="not one of"):
        factory._character()
    monkeypatch.setattr(factory, "CHARACTER", "Defect")
    rm = factory.new_run(SEED)
    assert rm.character == "Defect" and factory.check_run(rm) is rm
    monkeypatch.setattr(factory, "CHARACTER", "Ironclad")
    with pytest.raises(AssertionError, match="not a campaign run"):
        factory.check_run(rm)


# ── the Rust tokenizer ───────────────────────────────────────────────────────


@pytest.mark.parametrize("character", OTHERS)
def test_the_rust_env_agrees_row_for_row(character: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`rsenv_lockstep` on another character's runs: the Rust training env's rows equal
    the Python env's, the "chars" tokens and columns included."""
    monkeypatch.setattr(factory, "CHARACTER", character)
    ls = Lockstep(f9_p=0.0)
    run_seeds(ls, SEED, 3, pinned=False, practice=0)
    run_seeds(ls, SEED + 50, 1, pinned=True, practice=0)
    assert ls.stats["rows"] > 300
    assert ls.mismatches == 0, {"stats": dict(ls.stats), "examples": ls.examples[:5]}
