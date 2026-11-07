"""The tokenizer's WP-ENG-1 reads against the real wheel (GH #1097 A5).

`ppo.e2e.tokens` reads `bv.telegraph()`, `bv.peek_intents(3)`, `bv.encounter_id`,
`cs.true_draw_top(10)` and `cs.peek_intents(3)` when the wheel has them; its unit tests
use fakes. Here every combat row of real A10 runs is checked against the getters,
computed independently from SCHEMA.md's column table:

* RUNCTX's X_ID_AUX is the vocabulary id of `bv.encounter_id`, never NONE or UNK;
* each live enemy's telegraph and forecast columns are the getters' values, and a dead
  enemy (the wire's `is_dead`) gets none of them;
* PRIV_DRAW is the old fallback `state_to_dict(cs)["deck"]["draw_pile"][:10]`, which
  `cs.true_draw_top(10)` equals;
* PRIV_ENEMY carries each live enemy's `cs.peek_intents(3)` move ids and its monster.

On a wheel without the getters (W1) the module skips.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import pytest
import sts2_client
import sts2_rs
from ppo.e2e import contracts as C
from ppo.e2e import factory
from ppo.e2e import fillcheck as FC
from ppo.e2e import tokens as TK
from ppo.e2e import vocab as V
from ppo.e2e.env import E2EDecision, E2EEnv, EnvConfig

SEED = 162_000_980  # plan §3.1 smoke seeds
RUNS = 6
TELE_CAT = ("X_INTENT_T2", "X_INTENT_T3")
FC_CAT = ("X_FC1", "X_FC2", "X_FC3")
TELE_CONT = ("E_N_INTENTS", "E_T2_EFF", "E_T2_TIMES", "E_T3_EFF", "E_T3_TIMES")
FC_CONT = tuple(f"E_F{s}_{k}" for s in (1, 2, 3) for k in ("CERT", "AMT", "TIMES", "TOT"))


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def _kind(i: dict[str, Any] | None) -> int:
    if not i:
        return 0
    k = str(i.get("kind", "Unknown"))
    return C.INTENT_IDS.get("Attack" if k == "MultiAttack" else k, 0)


def _num(x: Any) -> float:
    return 0.0 if x is None else float(x)


def expected_enemy(tele: list[Any], fc: list[Any]) -> tuple[dict[str, int], dict[str, float]]:
    """One live enemy's telegraph and forecast columns (SCHEMA.md section 2.9)."""
    cat: dict[str, int] = {}
    cont: dict[str, float] = {"E_N_INTENTS": len(tele) / 3.0}
    for k, n in ((1, 2), (2, 3)):
        i = tele[k] if k < len(tele) else None
        cat[f"X_INTENT_T{n}"] = _kind(i)
        cont[f"E_T{n}_EFF"] = _num(i.get("effective_amount", i.get("amount")) if i else None) / 30
        cont[f"E_T{n}_TIMES"] = _num(i.get("times") if i else None) / 5
    for s in (1, 2, 3):
        st = fc[s - 1] if s - 1 < len(fc) else None
        ints = (st or {}).get("intents") or []
        head = ints[0] if ints else None
        cat[f"X_FC{s}"] = _kind(head)
        cont[f"E_F{s}_CERT"] = _num(st.get("certainty")) if st else 0.0
        cont[f"E_F{s}_AMT"] = _num(head.get("amount") if head else None) / 30
        cont[f"E_F{s}_TIMES"] = _num(head.get("times") if head else None) / 5
        tot = sum(
            _num(i.get("effective_amount", i.get("amount"))) * float(i.get("times") or 1)
            for i in ints
            if i.get("kind") in ("Attack", "DeathBlow")
        )
        cont[f"E_F{s}_TOT"] = tot / 50
    return cat, {k: float(np.float16(v)) for k, v in cont.items()}


def check_row(d: E2EDecision, tok: TK.Tok, n: collections.Counter) -> None:
    """Assert one combat row's WP-ENG-1 inputs against the getters; count what it saw."""
    v = V.load()
    XC, XF, SEC = C.XC, C.XF, C.SEC
    n["rows"] += 1
    enc = d.bv.encounter_id
    (rc,) = np.flatnonzero(tok.sec == SEC["RUNCTX"])
    aux = int(tok.cat[rc, XC["X_ID_AUX"]])
    assert enc and aux == v.id("encounter", enc) and v.has("encounter", enc), enc
    tele, fc = d.bv.telegraph(), d.bv.peek_intents(3)
    enemies = d.snap.get("enemies") or []
    wire = sts2_rs.state_to_dict(d.bv)["encounter"]
    assert len(tele) == len(fc) == len(enemies) == len(wire)
    toks = np.flatnonzero(tok.sec == SEC["ENEMY"])
    assert len(toks) == len(enemies)
    want_pe = []
    true_fc = d.cs.peek_intents(3)
    for j, t in enumerate(toks):
        got_cat = {c: int(tok.cat[t, XC[c]]) for c in TELE_CAT + FC_CAT}
        got_cont = {c: float(tok.cont[t, XF[c]]) for c in TELE_CONT + FC_CONT}
        assert bool(wire[j].get("is_dead")) == bool(enemies[j].get("is_dead"))
        if enemies[j].get("is_dead"):
            n["dead"] += 1
            assert not any(got_cat.values()) and not any(got_cont.values()), (j, got_cat)
            continue
        n["live"] += 1
        n["multi_intent"] += int(len(tele[j]) > 1)
        assert (got_cat, got_cont) == expected_enemy(tele[j], fc[j]), (j, enemies[j]["kind"])
        kind = enemies[j]["kind"]
        steps = list(true_fc[j])[:3]
        if steps and j < C.PRIV_ENEMY_K:  # the first PRIV_ENEMY_K enemy slots
            ids = [v.id("move", f"{kind}.{s['move_id']}") for s in steps]
            assert all(v.has("move", f"{kind}.{s['move_id']}") for s in steps), steps
            want_pe.append((ids + [0] * (3 - len(ids)), v.id("monster", kind)))
    pe = np.flatnonzero(tok.psec == SEC["PRIV_ENEMY"])
    ids = [XC[c] for c in ("X_ID_MAIN", "X_ID_MOD", "X_ID_REL")]
    got_pe = [(tok.pcat[r, ids].tolist(), int(tok.pcat[r, XC["X_ID_AUX"]])) for r in pe]
    assert got_pe == want_pe
    n["priv_enemy"] += len(pe)
    fb = sts2_rs.state_to_dict(d.cs)["deck"].get("draw_pile", [])[: C.PRIV_DRAW_K]
    assert list(d.cs.true_draw_top(C.PRIV_DRAW_K)) == fb
    pd = np.flatnonzero(tok.psec == SEC["PRIV_DRAW"])
    assert len(pd) == len(fb)
    if fb:
        n["draw_rows"] += 1
        want_cat, want_cont = TK._priv_draw_rows(fb, None)
        assert np.array_equal(tok.pcat[pd], want_cat)
        assert np.array_equal(tok.pcont[pd].view(np.uint16), want_cont.view(np.uint16))


def test_combat_rows_carry_the_getters_values() -> None:
    n: collections.Counter = collections.Counter()
    for k in range(RUNS):
        env = E2EEnv(SEED + k, cfg=EnvConfig(f9_p=0.0))
        act = FC.random_actor(SEED + k)
        d = env.reset()
        while isinstance(d, E2EDecision):
            if d.is_combat:
                if not hasattr(d.bv, "telegraph"):
                    pytest.skip("the wheel has no WP-ENG-1 getters (W1)")
                check_row(d, TK.encode(d, priv=True), n)
            d = env.step(act(d))
        assert env.terminal.outcome != "error", env.terminal.error
    assert n["rows"] >= 150 and n["live"] >= 200 and n["dead"] > 0, n
    assert n["multi_intent"] > 0 and n["draw_rows"] > 0 and n["priv_enemy"] >= n["live"] // 2, n
