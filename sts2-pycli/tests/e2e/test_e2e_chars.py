"""The "chars" fill event (GH #1373, SCHEMA.md section 13): every character's mechanics.

The schema was frozen for Ironclad. A row of another character's run carries its
character (`X_CHAR`), the pet (`ALLY`: Osty), the orb belt (`ORB`) and the doomed flag
(`E_DOOMED`); the tactical HP loss counts Osty and the Frost passives. An Ironclad row
carries none of it, the schema hash does not move and a net that does not declare the
event keeps its inputs at zero weight.
"""

from __future__ import annotations

import dataclasses
import json
import random
from functools import cache
from typing import Any

import numpy as np
import pytest
import sts2_client
import sts2_rs
import torch
import undeclared_rows as UR  # tests/e2e: the undeclared-row hash (GH #1197)
from ppo.combat_ctx import snapshot as CS
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
        # A "chars" row prices the turn end live (GH #1197, SCHEMA.md 13.5).
        loss = TC.hp_loss_live(TC.live_snapshot(d.snap, d.bv, d.d_bv), p)
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


@pytest.mark.parametrize("character", ("Ironclad", *OTHERS))
def test_lockstep_all_characters(character: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`rsenv_lockstep` on every character's runs (Ironclad with "chars" declared): the
    Rust training env's rows equal the Python env's, the "chars" tokens and columns
    (the live bound, the end-of-turn columns, the S_ctx overrides) included."""
    monkeypatch.setattr(factory, "CHARACTER", character)
    monkeypatch.setenv(factory.CHARS_ENV, "1")
    ls = Lockstep(f9_p=0.0)
    run_seeds(ls, SEED, 3, pinned=False, practice=0)
    run_seeds(ls, SEED + 50, 1, pinned=True, practice=0)
    assert ls.stats["rows"] > 300
    assert ls.mismatches == 0, {"stats": dict(ls.stats), "examples": ls.examples[:5]}
    rows = _play(character, range(SEED, SEED + 1), cap=60)
    assert all(t.cat[0, C.XC["X_CHAR"]] == C.CHAR_ID[character] for _d, t in rows)


# ── undeclared rows: main's bytes (GH #1197) ─────────────────────────────────


@cache
def _undeclared() -> tuple[dict[str, Any], dict[str, tuple[list[str], list[str]]]]:
    """The committed hashes and this checkout's, of the same random-play runs."""
    import os

    want = json.loads(UR.DATA.read_text(encoding="utf-8"))
    # The hashes predate GH #1791's default corpse-power strip: pin the
    # engine's form (every power kept), which those rows were made with.
    assert not CS.STRIP_CORPSE_POWERS, "pin STRIP_CORPSE_POWERS off first"
    assert factory.CHARACTER == "Ironclad" and not os.environ.get(factory.CHARS_ENV)
    return want, {str(s): UR.run_hashes(s, want["cap"]) for s in want["seeds"]}


def _first_diffs(want: dict[str, list[str]], got: dict[str, list[str]]) -> list[Any]:
    out: list[Any] = []
    for s, w in want.items():
        g = got[s]
        if len(g) != len(w):
            out.append((s, "rows", len(w), len(g)))
        out += [(s, i) for i, (a, b) in enumerate(zip(w, g, strict=False)) if a != b][:3]
    return out


def test_undeclared_ironclad_rows_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #1374 check, committed: an Ironclad row without "chars" is main's, byte for
    byte, every policy and PRIV array (`undeclared_rows.py`; the hashes come from
    f6716cae7, main before GH #1197). Never regenerate the data file from a branch:
    only from origin/main without the GH #1197 tokenizer change, in a separate
    worktree (the exact command is in `undeclared_rows.py`'s docstring)."""
    monkeypatch.setattr(CS, "STRIP_CORPSE_POWERS", False)
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    want, got = _undeclared()
    assert want["rows"] > 2000
    diffs = _first_diffs(want["runs"], {s: r for s, (r, _c) in got.items()})
    assert not diffs, diffs[:10]


def test_combat_ctx_rows_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ppo.combat_ctx`'s own S_ctx tokens of the same fights are main's: the chars
    overrides sit in the e2e tokenizer, not in the combat PPO's tokenizer."""
    monkeypatch.setattr(CS, "STRIP_CORPSE_POWERS", False)
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    want, got = _undeclared()
    assert sum(map(len, want["combat_ctx"].values())) > 1500
    diffs = _first_diffs(want["combat_ctx"], {s: c for s, (_r, c) in got.items()})
    assert not diffs, diffs[:10]


# ── the new aliases and the per-character values (GH #1197) ──────────────────

NEW_ALIASES = {
    "C_STAR_COST": "X_C_RES3",
    "E_DIES_FIRST": "X_C_RES4",
    "S_DISCARDS_TURN": "X_C_RES5",
    "S_ATTACKS_TURN": "X_C_RES6",
    "S_PLAYER_DOOMED": "X_C_RES7",
    "R_ARMED": "R_RES0",
}


def test_chars_slots_new_aliases() -> None:
    for alias, spare in NEW_ALIASES.items():
        assert C.CHARS_CONT[alias] == spare and C.XF[alias] == C.XF[spare]
        col = C.X_CONT[C.XF[alias] - C.SCTX_N_CONT]
        assert (col.name, col.status) == (spare, "reserved:spare")
        assert alias in C.CHARS_DOC
    cont = set(C.FILL_SLOTS[C.FILL_CHARS]["cont"])
    assert {C.XF[f"X_C_RES{i}"] for i in range(8)} <= cont
    assert C.XF["R_RES0"] in cont
    live = C.SECTION_LIVE
    for sec, cols in (
        ("HAND", ("C_STAR_COST",)),
        ("ACT_PLAY", ("C_STAR_COST",)),
        ("ENEMY", ("E_DIES_FIRST",)),
        ("STATE", ("S_DISCARDS_TURN", "S_ATTACKS_TURN", "S_PLAYER_DOOMED")),
        ("RELIC", ("R_ARMED",)),
    ):
        assert {C.XF[c] for c in cols} <= set(live[C.SEC[sec]][1]), sec
    assert C.SCTX_C["SLY"] in live[C.SEC["ACT_CHOOSE"]][1]
    for sec in ("DRAW_UNK", "DRAW_TOP", "DRAW_BOT", "DISCARD", "EXHAUST", "PLAY"):
        assert {C.SCTX_C[c] for c in ("SLY", "SLY_TURN", "RETAIN_TURN")} <= set(live[C.SEC[sec]][1])
    # The hashed table stays frozen.
    spec = C.schema_spec()
    assert C.XF["C_STAR_COST"] not in spec["section_live"]["HAND"][1]
    assert C.SCTX_C["SLY"] not in spec["section_live"]["ACT_CHOOSE"][1]
    old, new = N.layout(), N.layout([C.FILL_CHARS])
    assert {C.XF[a] for a in NEW_ALIASES} <= set(old.zero_cont)
    assert not {C.XF[a] for a in NEW_ALIASES} & set(new.zero_cont)


def _scripted(character: str, cs: Any, *, chars: bool = True) -> tuple[Any, K.Tok]:
    """The first combat decision of a `character` run with its fight replaced by `cs`,
    rebuilt as the env builds a decision (`f7check.world_decision`), and its row."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "CHARACTER", character)
        if chars:
            mp.setenv(factory.CHARS_ENV, "1")
        else:
            mp.delenv(factory.CHARS_ENV, raising=False)
        env = E.E2EEnv(SEED)
        d = env.reset()
        rng = random.Random(SEED)
        while not d.is_combat:
            d = env.step(rng.randrange(d.n_classes))
        bv, d_bv, snap = E.combat_view(cs, d.room)
        snap["ehist"] = d.snap["ehist"]
        cc = E.merged_classes(cs, room=d.room)
        d = dataclasses.replace(
            d,
            cs=cs,
            bv=bv,
            d_bv=d_bv,
            snap=snap,
            dtype=cc.dtype,
            kinds=cc.kinds,
            obs_actions=cc.obs_actions,
            actions=cs.legal_actions(),
            classes=cc.classes,
            members=cc.members,
        )
        return d, K.encode(d, priv=False)


def _fight(character: str, card: str, relic: str, edit: Any = None) -> Any:
    cs = factory.scenario_state(
        seed=5,
        character=character,
        deck=[card] * 10,
        hp=70,
        max_hp=80,
        relics=[relic],
        encounter="SLIMES_WEAK",
        ascension=10,
    )
    if edit is None:
        return cs
    sd = sts2_rs.state_to_dict(cs)
    edit(sd)
    return sts2_rs.state_from_dict(sd)


def _sec_rows(t: K.Tok, sec: str) -> np.ndarray:
    return np.flatnonzero(t.sec == C.SEC[sec])


def _f16(x: float) -> float:
    return float(np.float16(np.float32(x)))


def test_chars_row_values() -> None:
    """Per character, a scripted fight's row holds the live value (SCHEMA.md 13.5)."""

    # Regent: Falling Star costs 2 stars; one made free this turn costs none. The paid
    # copies' plays merge into one class, and every member carries the representative's
    # value (`_class_reps`). The free copy is a class of its own (GH #1407,
    # `test_a_free_star_copy_is_its_own_play_class`).
    def stars(sd: dict[str, Any]) -> None:
        sd["player"]["stars"] = 5

    def free(sd: dict[str, Any]) -> None:
        stars(sd)
        sd["deck"]["hand"][0]["star_cost"] = 0  # the wire stamps StarCostFreeThisTurn

    d, t = _scripted("Regent", _fight("Regent", "FALLING_STAR", "DIVINE_RIGHT", stars))
    hand = _sec_rows(t, "HAND")
    assert len(hand) >= 3
    assert t.cont[hand, C.XF["C_STAR_COST"]].tolist() == [_f16(2 / 5)] * len(hand)
    play = _sec_rows(t, "ACT_PLAY")
    assert len(play) >= 3
    assert t.cont[play, C.XF["C_STAR_COST"]].tolist() == [_f16(2 / 5)] * len(play)
    d, t = _scripted("Regent", _fight("Regent", "FALLING_STAR", "DIVINE_RIGHT", free))
    hand = _sec_rows(t, "HAND")
    assert t.cont[hand, C.XF["C_STAR_COST"]].tolist() == [0.0] + [_f16(2 / 5)] * (len(hand) - 1)
    assert any(len(m) > 1 for m in d.members)  # the paid copies still merge
    # Regent: Mini Regent is armed while its relic counter is 0 (wire `armed`).
    d, t = _scripted("Regent", _fight("Regent", "FALLING_STAR", "MINI_REGENT"))
    relic = _sec_rows(t, "RELIC")
    armed = [bool(r.get("armed")) for r in d.d_bv["player"]["relics"]]
    assert True in armed and t.cont[relic, C.XF["R_ARMED"]].tolist() == [float(a) for a in armed]

    # Silent: an enemy whose Poison kills it before it acts.
    def poison(sd: dict[str, Any]) -> None:
        m = sd["encounter"][0]
        m["powers"] = [{"name": "POISON", "amount": int(m["hp"])}]

    _d, t = _scripted("Silent", _fight("Silent", "STRIKE_SILENT", "RING_OF_THE_SNAKE", poison))
    enemy = _sec_rows(t, "ENEMY")
    assert t.cont[enemy[0], C.XF["E_DIES_FIRST"]] > 0.5
    assert not t.cont[enemy[1:], C.XF["E_DIES_FIRST"]].any()

    # Necrobinder: Osty deals Poke, so the player's Strength does not price it.
    def strength(sd: dict[str, Any]) -> None:
        sd["player"]["powers"] = [{"name": "STRENGTH", "amount": 5}]

    rows = [
        _scripted("Necrobinder", _fight("Necrobinder", "POKE", "BOUND_PHYLACTERY", e))[1]
        for e in (None, strength)
    ]
    dmg = [r.cont[_sec_rows(r, "HAND"), C.SCTX_C["DMG_PREV"]] for r in rows]
    assert (dmg[0] > 0).all() and np.array_equal(dmg[0], dmg[1])
    for col in ("DMG_PREV", "T_DMG"):
        tgt = [r.cont[_sec_rows(r, "ACT_PLAY"), C.SCTX_C[col]] for r in rows]
        assert len(tgt[0]) and (tgt[0] > 0).all() and np.array_equal(tgt[0], tgt[1]), col

    # Defect: Barrage hits once per orb.
    def orbs(sd: dict[str, Any]) -> None:
        sd["player"]["orbs"] = [{"type": "Lightning", "passive": 3, "evoke": 8}] * 3

    d, t = _scripted("Defect", _fight("Defect", "BARRAGE", "CRACKED_CORE", orbs))
    live = TC.live_snapshot(d.snap, d.bv, d.d_bv)
    md = TC.max_damage_live(live, TC.playable_from(d.obs_actions))
    old = TC.max_damage(d.snap, TC.playable_from(d.obs_actions))
    enemy = _sec_rows(t, "ENEMY")
    got = t.cont[enemy, C.XF["E_MAXDMG"]].astype(np.float64) * 50.0
    assert got == pytest.approx(md, rel=2e-3)
    assert md[0] > old[0] > 0

    # Ironclad, declared: Body Slam deals the player's block.
    def block(sd: dict[str, Any]) -> None:
        sd["player"]["block"] = 20

    cs = _fight("Ironclad", "BODY_SLAM", "BURNING_BLOOD", block)
    _d, t = _scripted("Ironclad", cs)
    enemy = _sec_rows(t, "ENEMY")
    assert (t.cont[enemy, C.XF["E_MAXDMG"]] > 0).all()
    _d, off = _scripted("Ironclad", cs, chars=False)
    assert not off.cont[_sec_rows(off, "ENEMY"), C.XF["E_MAXDMG"]].any()


def _free_star(sd: dict[str, Any]) -> None:
    sd["player"]["stars"] = 5
    sd["deck"]["hand"][0]["star_cost"] = 0  # the wire stamps StarCostFreeThisTurn


def _play_classes(d: Any) -> dict[tuple[int, Any], int]:
    """`(hand index, target) -> class` of the decision's plays."""
    return {
        (int(a["card_index"]), a.get("target_index")): d.classes[i]
        for i, a in enumerate(d.obs_actions)
        if a.get("action") == "play"
    }


def test_a_free_star_copy_is_its_own_play_class() -> None:
    """GH #1407: two copies of a star-cost card, one free this turn, are two play classes
    on another character's row, so the policy can choose the free one. The paid copies
    still merge, and a play's `ACT_PLAY` token shows its own class's star cost."""
    d, t = _scripted("Regent", _fight("Regent", "FALLING_STAR", "DIVINE_RIGHT", _free_star))
    hand = d.d_bv["deck"]["hand"]
    free = [i for i, c in enumerate(hand) if c.get("star_cost_live") == 0]
    assert len(free) == 1 and sum(c.get("star_cost_live") == 2 for c in hand) >= 2
    plays = _play_classes(d)
    for target in {tg for _, tg in plays}:
        paid = {c for (i, tg), c in plays.items() if tg == target and i != free[0]}
        assert len(paid) == 1, "the paid copies merge"
        assert plays[free[0], target] not in paid, "the free copy is its own class"
    star = t.cont[_sec_rows(t, "ACT_PLAY"), C.XF["C_STAR_COST"]].tolist()
    # One `ACT_PLAY` token per member, in action order: each shows its own class's cost.
    cost = [0.0 if i == free[0] else _f16(2 / 5) for (i, _tg) in plays]
    assert star == cost
    # With no discount there is nothing to refine: the keys are the schema's own.
    d, _t = _scripted("Regent", _fight("Regent", "FALLING_STAR", "DIVINE_RIGHT", _stars5))
    assert E.combat_keys(d.bv, d.d_bv, d.obs_actions) == C.combat_class_keys(
        d.obs_actions, d.d_bv["deck"]["hand"], d.d_bv["player"]["potions"]
    )


def _stars5(sd: dict[str, Any]) -> None:
    sd["player"]["stars"] = 5


@pytest.mark.parametrize("chars", [False, True])
def test_an_ironclad_rows_star_copies_keep_the_schemas_classes(chars: bool) -> None:
    """The refinement is another character's: an Ironclad row's keys stay the frozen
    schema's, declared or not, so no Ironclad class, log or fixture moves (GH #1407)."""
    cs = _fight("Ironclad", "FALLING_STAR", "BURNING_BLOOD", _free_star)
    d, _t = _scripted("Ironclad", cs, chars=chars)
    hand = d.d_bv["deck"]["hand"]
    assert {c.get("star_cost_live") for c in hand} == {0, 2}
    assert E.combat_keys(d.bv, d.d_bv, d.obs_actions) == C.combat_class_keys(
        d.obs_actions, hand, d.d_bv["player"]["potions"]
    )
    plays = _play_classes(d)
    assert len({c for (_i, tg), c in plays.items() if tg == next(iter(plays))[1]}) == 1


def test_live_piles_split_on_pile_flags() -> None:
    """A pile card's `pile_flags` reach its `flags` on a "chars" row's live snapshot, and
    a multiset entry splits where they differ (twin: `snapshot::tests::live_piles_split`)."""
    strike = {"id": "STRIKE_SILENT", "kind": "card", "upgraded": False, "base_cost": 1}
    d_bv = {
        "deck": {
            "draw_pile": [],
            "discard_pile": [strike, {**strike, "pile_flags": ["sly_turn"]}, strike],
            "exhaust_pile": [],
            "play_pile": [],
        }
    }
    snap: dict[str, Any] = {"discard": "honest"}
    TC._live_piles(snap, d_bv)
    assert sorted(snap["discard"], key=lambda e: e[1]) == [
        [{**strike, "cost": 1, "flags": ["sly_turn"]}, 1],
        [{**strike, "cost": 1}, 2],
    ]
    plain = {"discard": "honest"}
    TC._live_piles(plain, {"deck": {"discard_pile": [strike, strike]}})
    assert plain == {"discard": "honest"}  # no pile_flags: the honest piles stay


def test_fill_check_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared rows pass the fill check; a new column set on an undeclared row fails it."""
    wp = list(FC.wheel_filled())
    regent = _sample("Regent")
    assert not FC.check(regent, [*wp, C.FILL_CHARS]).definite()
    monkeypatch.delenv(factory.CHARS_ENV, raising=False)
    ironclad = _sample("Ironclad", 6)
    assert not FC.check(ironclad, wp).definite()
    for sec, alias in (("STATE", "S_PLAYER_DOOMED"), ("RELIC", "R_ARMED"), ("HAND", "C_STAR_COST")):
        rb = {k: v.copy() for k, v in ironclad.items()}
        tok = np.flatnonzero(rb["sec"] == C.SEC[sec])[0]
        rb["cont"][tok, C.XF[alias]] = 1.0
        rep = FC.check(rb, wp)
        assert C.FILL_CHARS in rep.stray, alias


def _mad_science_modes(sd: dict[str, Any]) -> None:
    # The TinkerTime packing (`pack_tinker_time`): mode << 4 | rider. 21 is a Skill (the
    # GH #1854 eval card), 33 a Power; both play untargeted, so only the mode tells them
    # apart. The rest of the hand stays default (Attack, targeted).
    hand = sd["deck"]["hand"]
    hand[0].setdefault("combat_vars", {})["generic"] = 21
    hand[1].setdefault("combat_vars", {})["generic"] = 33
    hand[2].setdefault("combat_vars", {})["generic"] = 21


def test_mad_science_modes_split_the_play_class() -> None:
    """GH #1854: Mad Science keeps its TinkerTime mode in the copy's `generic` combat
    var. Two copies of different modes are two play classes (their RL2 action features
    read the copy's card type, so one class broke `check_members_identical`: Regent
    held-out seed 160000304); two of the same mode still merge."""
    assert E._face_vars({"id": "MAD_SCIENCE", "combat_vars": {"generic": 21}}) == {"generic": 21}
    assert E._face_vars({"id": "MAD_SCIENCE"}) == {}
    assert E._face_vars({"id": "GUILTY", "combat_vars": {"generic": 2}}, stars=True) == {}
    d, t = _scripted("Regent", _fight("Regent", "MAD_SCIENCE", "DIVINE_RIGHT", _mad_science_modes))
    plays = _play_classes(d)
    assert plays[0, None] == plays[2, None], "two Skill copies merge"
    assert plays[0, None] != plays[1, None], "a Skill and a Power copy are two classes"
    K.check_members_identical(t, d.members)
