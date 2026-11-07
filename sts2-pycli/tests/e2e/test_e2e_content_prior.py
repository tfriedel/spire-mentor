"""WP3b: the 48-column content prior (`ppo.e2e.content_prior`, SCHEMA.md §6.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ppo.e2e import content_prior as P
from ppo.e2e import contracts as CT
from ppo.e2e import decomp
from ppo.e2e import vocab as V

# Minimum share of a registry's ids with an extracted class (v0.111.0 decomp).
MIN_SHARE = {
    "card": 0.99,
    "relic": 0.99,
    "potion": 1.0,
    "power": 0.85,
    "monster": 0.9,
    "encounter": 1.0,
    "event": 0.95,
    "enchantment": 0.9,
}


def test_columns_are_the_schema_columns() -> None:
    assert tuple(CT.PRIOR_COLUMNS) == P.COLUMNS and P.N_PRIOR == CT.N_PRIOR == 48
    assert P.C["MISSING"] == 47


def test_table_shape_none_row_and_missing_flags() -> None:
    t = P.table()
    assert t.shape == (CT.VOCAB_ROWS, P.N_PRIOR) == (4096, 48)
    assert t.dtype == np.float32 and not t.flags.writeable
    assert not t[V.NONE_ID].any()
    for reg, uid in V.UNK_ID.items():
        assert t[uid, P.C["MISSING"]] == pytest.approx(1.0), reg
        assert t[uid].sum() == pytest.approx(1.0)
    assert not t[:, P.C["PRIOR_RES0"]].any()
    # Rows past the last used id stay zero (a later append fills them).
    assert not t[V.load().n_used :].any()
    assert np.isfinite(t).all() and np.abs(t).max() <= P.CLIP
    missing = t[:, P.C["MISSING"]] > 0.5
    assert not t[missing][:, : P.C["MISSING"]].any(), "a MISSING row has facts"


def test_coverage_per_registry() -> None:
    cov = P.coverage()
    for reg, share in MIN_SHARE.items():
        assert cov[reg]["share"] >= share, (reg, cov[reg])
    assert cov["move"]["extracted"] == 0 and cov["rest"]["extracted"] == 0


def test_printed_facts_of_a_few_ids() -> None:
    v, t = V.load(), P.table()

    def row(reg: str, name: str) -> dict[str, float]:
        r = t[v.id(reg, name)]
        return {c: float(r[i]) for i, c in enumerate(P.COLUMNS) if r[i]}

    bash = row("card", "BASH")
    assert {"TYPE_ATTACK", "RAR_BASIC", "TGT_ENEMY", "POOL_IRONCLAD", "APPLIES_DEBUFF"} <= set(bash)
    assert bash["COST"] == pytest.approx(2 / 3) and bash["DMG"] == pytest.approx(8 / 20)
    assert bash["DMG_UPG_DELTA"] == pytest.approx(0.2) and bash["HITS"] == pytest.approx(0.2)
    strike = row("card", "STRIKE_IRONCLAD")
    assert strike["DMG"] == pytest.approx(0.3) and strike["DMG_UPG_DELTA"] == pytest.approx(0.3)
    assert row("card", "TWIN_STRIKE")["HITS"] == pytest.approx(2 / 5)
    assert "GAINS_STRENGTH" in row("card", "INFLAME")
    assert "EXHAUSTS_OTHER" in row("card", "TRUE_GRIT")
    off = row("card", "OFFERING")
    assert "KW_EXHAUST" in off and "EXHAUSTS_OTHER" not in off and off["DRAW"] > 0
    assert "X_COST" in row("card", "WHIRLWIND") and "COST" not in row("card", "WHIRLWIND")
    assert row("card", "EXPECT_A_FIGHT")["BLOCK"] == pytest.approx(15 / 20)
    assert {"KW_ETERNAL", "TYPE_CURSE", "POOL_OTHER"} <= set(row("card", "ASCENDERS_BANE"))
    assert "POOL_COLORLESS" in row("card", "PROWESS")
    assert row("potion", "FIRE_POTION")["DMG"] == pytest.approx(1.0)
    assert "APPLIES_DEBUFF" in row("potion", "WEAK_POTION")
    assert "GAINS_STRENGTH" in row("relic", "VAJRA")
    assert "POWER_IS_DEBUFF" in row("power", "VULNERABLE")
    assert row("monster", "Nibbit")["MON_MAXHP"] == pytest.approx(0.48)  # A10 HP 44-48
    boss = row("encounter", "CEREMONIAL_BEAST_BOSS")
    assert boss["MON_IS_BOSS"] == pytest.approx(1.0) and boss["ACT"] == pytest.approx(1 / 3)
    assert row("encounter", "SLIMES_WEAK") == {"ACT": pytest.approx(1 / 3)}


def test_sha256_is_the_buffer_digest() -> None:
    t = P.table()
    assert P.sha256() == P.sha256(np.array(t)) and len(P.sha256()) == 64
    u = np.array(t)
    u[40, 0] += 1.0
    assert P.sha256(u) != P.sha256()


def test_the_table_is_pinned_under_py_src(tmp_path, monkeypatch) -> None:
    """Review finding TOK-4: stored tactical columns read the TSV's HITS column, which
    no stamp covered. `TABLE_SHA256` pins the table in a source file that `py_src`
    hashes, and `table()` refuses a TSV that differs."""
    from ppo.wheel_guard import first_party_sources

    assert P.sha256() == P.TABLE_SHA256
    assert Path(P.__file__).resolve() in {q.resolve() for q in first_party_sources()}
    lines = P.TSV.read_text(encoding="utf-8").splitlines(keepends=True)
    i = next(k for k, ln in enumerate(lines) if "HITS=" in ln)
    lines[i] = lines[i].replace("HITS=", "HITS=1", 1)  # one more digit: another value
    bad = tmp_path / P.TSV.name
    bad.write_text("".join(lines), encoding="utf-8")
    monkeypatch.setattr(P, "TSV", bad)
    P._text.cache_clear()
    P.table.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="TABLE_SHA256"):
            P.table()
    finally:
        monkeypatch.undo()
        P._text.cache_clear()
        P.table.cache_clear()
    assert P.sha256() == P.TABLE_SHA256


@pytest.mark.skipif(not decomp.available(), reason="the v0.111.0 decompilation is absent")
def test_committed_table_is_the_extractor_output() -> None:
    fresh = P.build(P.extract())
    np.testing.assert_allclose(fresh, P.table(), rtol=1e-5, atol=1e-6)
    assert P.parse(P.render(fresh)).tobytes() == P.table().tobytes()


@pytest.mark.skipif(not decomp.available(), reason="the v0.111.0 decompilation is absent")
def test_registry_check_finds_every_card_class() -> None:
    rc = decomp.registry_check()
    assert len(rc["card"]["missing"]) <= 6, rc["card"]["missing"]
    assert not rc["potion"]["missing"]
    # A3: the event fights' encounter classes are in the table (vocabulary 2.1).
    assert not rc["encounter"]["missing"] and not rc["encounter"]["extra"]


@pytest.mark.skipif(not decomp.available(), reason="the v0.111.0 decompilation is absent")
def test_an_abstract_encounter_takes_its_subclasses_room() -> None:
    """`BattlewornDummyEventEncounter` declares no `RoomType`; its V1-V3 settings are
    Monster rooms, so its row is extracted (not MISSING), like theirs."""
    w = P._World(decomp.DECOMP_VERSION)
    assert w.room["BattlewornDummyEventEncounter"] == "Monster"
    ex, v = P.extract(), V.load()
    for name in V.EVENT_ENCOUNTERS:
        assert v.id("encounter", name) in ex, name
        assert P.table()[v.id("encounter", name), P.C["MISSING"]] < 0.5, name
