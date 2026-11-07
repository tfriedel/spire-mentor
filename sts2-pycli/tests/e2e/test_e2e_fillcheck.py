"""The fill check (GH #1097 A4): the declared fill events match the rows, both ways.

The rows come from real A10 runs on the installed wheel (`fillcheck.sample_rows`). The
wheel without the WP-ENG-1 getters (W1) is simulated by hiding the getters from the
tokenizer, which then takes its fallbacks exactly as on W1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sts2_client
from ppo.e2e import contracts as C
from ppo.e2e import factory
from ppo.e2e import fillcheck as FC
from ppo.e2e import tokens as TK

SEED = 162_000_960  # plan §3.1 smoke seeds
RUNS = 8
WP = ["WP-ENG-1"]
WP_GETTERS = ("telegraph", "peek_intents", "true_draw_top")


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


@pytest.fixture(scope="module")
def getters_rows() -> dict[str, np.ndarray]:
    rb, terms = FC.sample_rows(RUNS, SEED)
    assert len(terms) == RUNS and all(t.outcome != "error" for t in terms)
    return rb


@pytest.fixture(scope="module")
def w1_rows() -> dict[str, np.ndarray]:
    """The same runs tokenized without the WP-ENG-1 getters, as on the W1 wheel."""
    getter = TK._getter

    def no_wp(obj: Any, *names: str) -> Any:
        return getter(obj, *[n for n in names if n not in WP_GETTERS])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TK, "_getter", no_wp)
        mp.setattr(TK, "_encounter_id", lambda d: None)
        rb, _ = FC.sample_rows(RUNS, SEED)
    return rb


def test_the_getters_rows_need_wp_eng_1_declared(getters_rows) -> None:
    rep = FC.check(getters_rows, WP)
    assert rep.ok, rep.problems
    assert rep.combat_rows >= FC.MIN_ROWS and rep.enemies >= FC.MIN_ENEMIES
    assert rep.aux_rows == rep.combat_rows
    assert rep.tele >= 0.95 * rep.enemies and rep.forecast == rep.enemies
    assert rep.priv_enemy >= 0.95 * rep.enemies
    assert rep.draw_rows > 0 and rep.draw_missing == 0 and rep.dead_filled == 0
    bad = FC.check(getters_rows, [])
    n = bad.stray["WP-ENG-1"]
    assert not bad.ok and n > rep.combat_rows
    assert bad.problems == (f"WP-ENG-1 is not declared but {n} token(s) or row(s) carry it",)


def test_rows_without_the_getters_must_not_declare_it(getters_rows, w1_rows) -> None:
    rep = FC.check(w1_rows, [])
    assert rep.ok, rep.problems
    assert rep.combat_rows == FC.check(getters_rows, WP).combat_rows
    assert rep.aux_rows == rep.tele == rep.forecast == rep.priv_enemy == 0
    assert rep.draw_rows > 0 and rep.draw_missing == 0  # the state-dict fallback
    w1, wp = (b["psec"] == C.SEC["PRIV_DRAW"] for b in (w1_rows, getters_rows))
    assert np.array_equal(w1_rows["pcat"][w1], getters_rows["pcat"][wp])  # the same draw
    assert np.array_equal(
        w1_rows["pcont"][w1].view(np.uint16), getters_rows["pcont"][wp].view(np.uint16)
    )
    bad = FC.check(w1_rows, WP)
    assert not bad.ok and len(bad.problems) == 4, bad.problems
    assert "X_ID_AUX is nonzero on 0/" in bad.problems[0]
    assert all(p.startswith("WP-ENG-1 declared but") for p in bad.problems)


def _copy(rb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in rb.items()}


def test_dead_enemies_and_the_draw_fail_either_way(getters_rows) -> None:
    rb = _copy(getters_rows)
    dead = np.flatnonzero((rb["sec"] == C.SEC["ENEMY"]) & (rb["cont"][:, C.SCTX_C["E_ALIVE"]] == 0))
    assert len(dead)
    rb["cont"][dead[0], C.XF["E_F1_CERT"]] = 1.0
    rep = FC.check(rb, WP)
    assert rep.dead_filled == 1 and rep.problems == ("1 dead-enemy token(s) carry WP-ENG-1 values",)
    rb = _copy(getters_rows)  # one row loses its PRIV_DRAW tokens
    off = C.row_offsets(rb)["priv"]
    first = int(np.flatnonzero(rb["psec"] == C.SEC["PRIV_DRAW"])[0])
    row = int(np.searchsorted(off, first, side="right") - 1)
    psec = rb["psec"][off[row] : off[row + 1]]  # a view
    psec[psec == C.SEC["PRIV_DRAW"]] = C.SEC["PRIV_RUN"]
    for filled in (WP, []):
        rep = FC.check(rb, filled)
        assert rep.draw_missing == 1
        assert "1 combat row(s) with a draw pile have no PRIV_DRAW" in rep.problems


def test_an_undeclared_fill_event_leaves_no_trace(getters_rows) -> None:
    rb = _copy(getters_rows)
    rb["cont"][0, C.XF["R_HIST_MISSING"]] = 1.0  # D9's flag
    ooc = int(np.flatnonzero(~rb["is_combat"])[0])
    rb["dtype"][ooc] = C.DT["REWARD_SCREEN"]  # #1069's decision type
    rep = FC.check(rb, WP)
    assert rep.stray == {"D9": 1, "#1069": 1}
    assert not rep.ok and len(rep.problems) == 2
    rep = FC.check(rb, [*WP, "D9", "#1069"])
    assert rep.ok and rep.unchecked == ("D9", "#1069")


def test_a_small_sample_is_inconclusive(getters_rows) -> None:
    few = C.take_rows(getters_rows, np.flatnonzero(getters_rows["is_combat"])[:20])
    for filled in (WP, []):
        rep = FC.check(few, filled)
        assert not rep.ok
        assert f"inconclusive: 20 combat rows < {FC.MIN_ROWS}" in rep.problems
    with pytest.raises(C.SchemaError, match="unknown fill events"):
        FC.check(getters_rows, ["WP-ENG-2"])
