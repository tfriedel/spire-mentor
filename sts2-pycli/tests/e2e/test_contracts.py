"""WP3a contract tests (GH #1097): the frozen schema tables, its hash, the
row-batch validator, the dedupe rule (T6), the content-ID table, and a live
walk over v0.111.0 A10 Ironclad runs that maps every decision to a known
decision type and every option to a known action kind."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from ppo.e2e import contracts as C
from ppo.e2e import vocab as V

# Golden stamps. A change here is a schema change: before G-schema bump
# `SCHEMA_REVISION` and update SCHEMA.md; after G-schema it needs a restart.
SCHEMA_SHA256_GOLDEN = "7afdc933284d80e400138267d46e51e2625a3ff09671d1028b131300c09f7f79"
VOCAB_SHA256_GOLDEN = "bac088c69d8e149cd9cb7ac81a07d4a3af360d35b7da37f9cc16537705667aaf"

try:
    import sts2_rs  # noqa: F401

    HAVE_RS = True
except ImportError:  # pragma: no cover - public export without the wheel
    HAVE_RS = False
needs_rs = pytest.mark.skipif(not HAVE_RS, reason="sts2_rs engine wheel not installed")


# ─── tables ──────────────────────────────────────────────────────────────────


def test_sctx_mirror_matches_1081_tokens() -> None:
    from ppo.combat_ctx import intent_log
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ppo import encode

    assert T.SECTIONS == C.SCTX_SECTIONS
    assert T.N_SECTION == C.SCTX_N_SECTION
    assert C.SCTX_ACT_SECTIONS == T.ACT_SECTIONS
    assert T.SLOTS == C.SCTX_SLOTS
    assert T.N_CAT == C.SCTX_N_CAT
    assert T.CONTENT_SLOTS == C.SCTX_CONTENT_SLOTS
    assert T.INTENT_KINDS == C.SCTX_INTENT_KINDS
    assert C.SCTX_INTENT_KINDS[-1] == intent_log.NONE_KIND
    assert T.N_CONT == C.SCTX_N_CONT == len(C.SCTX_CONT)
    assert encode.SEC_PRIV == C.SEC["PRIV_DRAW"] == 21
    named = {
        n: getattr(T, f"C_{n}")
        for n, _ in C.SCTX_CONT
        if not n.startswith("RL2_") and hasattr(T, f"C_{n}")
    }
    assert len(named) == 66, sorted(set(C.SCTX_C) - set(named))
    for n, i in named.items():
        assert C.SCTX_C[n] == i, n
    assert C.SCTX_C["RL2_S00"] == T.C_RL2S and C.SCTX_C["RL2_X00"] == T.C_RL2X
    assert tuple(C.SCTX_RL2_STATE_COLS) + tuple(C.SCTX_RL2_ACTION_COLS) == T.RL2_COLS
    assert set(C.SCTX_STUDENT_SLOTS).isdisjoint(C.SCTX_CONTENT_SLOTS)


@needs_rs
def test_sctx_table_sizes_match_the_engine() -> None:
    from ppo.combat_ctx import tokens as T

    assert T.vocab().digest() == dict(C.SCTX_TABLE_SIZES)
    assert tuple(int(x) for x in T.vocab().pads) == C.SCTX_PAD


def test_section_ids_are_unique_and_stable() -> None:
    assert len(C.SECTIONS) == C.N_SECTION == 64
    assert len(set(C.SECTIONS)) == len(C.SECTIONS)
    assert C.SECTIONS[: C.SCTX_N_SECTION] == C.SCTX_SECTIONS
    fixed = {
        "PRIV_DRAW": 21,
        "RUNCTX": 22,
        "BOSS": 23,
        "SUM_DRAW": 24,
        "SUM_RELICS": 28,
        "DECK": 29,
        "OFFER": 30,
        "MAPNODE": 31,
        "OPT_CARD": 32,
        "OPT_FIXED": 38,
        "PRIV_ENEMY": 48,
        "PRIV_RUN": 49,
    }
    for name, i in fixed.items():
        assert C.SEC[name] == i, name
    assert np.iinfo(np.int8).max + 1 >= C.N_SECTION
    assert C.PRIV_SECTIONS.isdisjoint(C.OPTION_SECTIONS)
    assert not C.PRIV_SECTIONS & set(C.SUMMARY_SECTIONS)
    live = set(range(C.N_SECTION)) - C.RESERVED_SECTIONS
    missing = {C.SECTIONS[s] for s in live if s not in C.SECTION_LIVE}
    assert not missing, missing


def test_action_kind_ids_unique_and_reserved_do_not_collide() -> None:
    ids = [k.id for k in C.ACTION_KINDS]
    assert ids == list(range(len(ids))), "ids must equal their position"
    assert len({k.name for k in C.ACTION_KINDS}) == len(ids)
    assert max(ids) < C.N_ACTION_KIND <= 256  # uint8 on the wire
    reserved = {k.id for k in C.ACTION_KINDS if k.status.startswith("reserved")}
    spare = {k.id for k in C.ACTION_KINDS if k.status == "spare"}
    assert C.LIVE_ACTION_KINDS.isdisjoint(reserved | spare)
    assert reserved.isdisjoint(spare)
    gaps = {k.status for k in C.ACTION_KINDS if k.status.startswith("reserved:")}
    for gap in ("#1069", "treasure", "crystal-sphere", "ooc-potion", "fake-merchant"):
        assert f"reserved:{gap}" in gaps, gap
    assert len(spare & set(range(64, 96))) == 32, "D9 spare block"
    assert frozenset(range(1, 16)) == C.COMBAT_ACTION_KINDS


def test_decision_type_ids_unique_and_kind_sets_consistent() -> None:
    ids = [d.id for d in C.DECISION_TYPES]
    assert ids == list(range(C.N_DTYPE))
    assert len({d.name for d in C.DECISION_TYPES}) == len(ids)
    for d in C.DECISION_TYPES:
        for k in d.kinds:
            kind = C.ACTION_KINDS[k]
            if d.status == "live":
                assert kind.status == "live", (d.name, kind.name)
            if d.head == C.HEAD_COMBAT:
                assert k in C.COMBAT_ACTION_KINDS, (d.name, kind.name)
            elif d.id:
                assert k not in C.COMBAT_ACTION_KINDS, (d.name, kind.name)
    assert {d.id for d in C.DECISION_TYPES if d.status == "live" and d.head == 0} == C.COMBAT_DTYPES
    reserved = {d.name for d in C.DECISION_TYPES if d.status.startswith("reserved")}
    assert {"REWARD_SCREEN", "TREASURE_CHOICE", "CRYSTAL_SPHERE"} <= reserved
    assert {"POTION_MENU", "FAKE_MERCHANT"} <= reserved
    live_kinds = set().union(*(d.kinds for d in C.DECISION_TYPES if d.status == "live"))
    assert live_kinds == set(C.LIVE_ACTION_KINDS), "every live kind belongs to a live type"


def test_columns_are_consistent() -> None:
    assert C.N_CAT == 32 and len(C.X_CAT) == 17
    assert C.SCTX_N_CONT + len(C.X_CONT) == C.N_CONT
    names = [c.name for c in C.X_CAT] + [c.name for c in C.X_CONT] + [n for n, _ in C.SCTX_CONT]
    assert len(set(names)) == len(names)
    assert len(C.PRIOR_COLUMNS) == C.N_PRIOR == 48
    assert len(set(C.PRIOR_COLUMNS)) == 48 and C.PRIOR_COLUMNS[-1] == "MISSING"
    for c in C.X_CAT:
        assert c.status == "live" or c.status.startswith("reserved:"), c
    for c in C.X_CAT:
        if c.name.startswith("X_ID_"):
            assert c.size == C.VOCAB_ROWS
    used_cat: set[int] = set()
    used_cont: set[int] = set()
    for s, (cat, cont) in C.SECTION_LIVE.items():
        assert all(0 <= i < C.N_CAT for i in cat), C.SECTIONS[s]
        assert all(0 <= i < C.N_CONT for i in cont), C.SECTIONS[s]
        assert len(set(cat)) == len(cat) and len(set(cont)) == len(cont), C.SECTIONS[s]
        used_cat |= set(cat)
        used_cont |= set(cont)
    dead_cat = {c.name for c in C.X_CAT if C.XC[c.name] not in used_cat}
    dead_cont = {c.name for c in C.X_CONT if C.XF[c.name] not in used_cont}
    assert all(n.startswith("X_RES") for n in dead_cat), dead_cat
    assert all(n.startswith(("X_C_RES", "R_RES")) or n == "R_RES0" for n in dead_cont), dead_cont
    # The student reads content only through the trailing ID columns.
    sctx_content = {C.SCTX_SLOTS.index(s) for s in C.SCTX_CONTENT_SLOTS}
    for s in (C.SEC["DECK"], C.SEC["OPT_CARD"], C.SEC["OPT_RELIC"], C.SEC["OPT_POTION"]):
        assert not set(C.SECTION_LIVE[s][0]) & sctx_content, C.SECTIONS[s]
    assert max(C.ROOM_ID.values()) < C.X_CAT[6].size
    assert max(C.INTENT_IDS.values()) < 16


def test_schema_sha256_golden() -> None:
    assert C.schema_sha256() == SCHEMA_SHA256_GOLDEN
    spec = C.schema_spec()
    json.dumps(spec)  # plain data
    assert C.schema_sha256() == C.schema_sha256()


def test_stamps() -> None:
    s = C.stamps(vocab_sha256="v", filled=["WP-ENG-1"])
    assert set(s) == set(C.STAMP_KEYS) and s["teacher"] is None
    assert s["schema_sha256"] == SCHEMA_SHA256_GOLDEN and s["game_version"] == "v0.111.0"
    C.check_stamps(s, dict(s))
    for k, v in (("vocab_sha256", "w"), ("schema_sha256", "x"), ("game_version", "v0.107.1")):
        with pytest.raises(C.SchemaError, match=k):
            C.check_stamps(s, dict(s, **{k: v}))
    with pytest.raises(C.SchemaError):
        C.stamps(vocab_sha256="v", filled=["nope"])
    # The content prior's fill event and its sha256 come together (review finding NT-1).
    with pytest.raises(C.SchemaError, match="content-prior"):
        C.stamps(vocab_sha256="v", filled=["content-prior"])
    with pytest.raises(C.SchemaError, match="content-prior"):
        C.stamps(vocab_sha256="v", prior_sha256="p")
    assert C.stamps(vocab_sha256="v", filled=["content-prior"], prior_sha256="p")["prior_sha256"]
    # Fill state: equal, except that a restart may add fill events and the prior.
    unfilled = dict(s, filled=[])
    with pytest.raises(C.SchemaError, match="filled"):
        C.check_stamps(s, unfilled)
    with pytest.raises(C.SchemaError, match="filled"):
        C.check_stamps(unfilled, s)
    C.check_stamps(unfilled, s, restart=True)
    with pytest.raises(C.SchemaError, match="filled"):
        C.check_stamps(s, unfilled, restart=True)  # filled columns never go back to zero
    with_prior = dict(s, filled=["WP-ENG-1", "content-prior"], prior_sha256="p")
    with pytest.raises(C.SchemaError, match="prior_sha256"):
        C.check_stamps(dict(with_prior, filled=s["filled"]), s)
    C.check_stamps(s, with_prior, restart=True)
    with pytest.raises(C.SchemaError, match="prior_sha256"):
        C.check_stamps(with_prior, dict(with_prior, prior_sha256="q"), restart=True)
    # Teacher: equal wherever both name one.
    t1 = C.teacher_stamp(ckpt_sha256="a", rare={"card": np.asarray([1, 2])}, sctx_vocab={"card": 9})
    assert t1 == C.teacher_stamp(ckpt_sha256="a", rare={"card": [1, 2]}, sctx_vocab={"card": 9})
    t2 = C.teacher_stamp(ckpt_sha256="a", rare={"card": [1, 3]}, sctx_vocab={"card": 9})
    assert t1["rare_sha256"] != t2["rare_sha256"]
    st1 = C.stamps(vocab_sha256="v", teacher=t1)
    C.check_stamps(st1, C.stamps(vocab_sha256="v"))
    with pytest.raises(C.SchemaError, match="teacher"):
        C.check_stamps(st1, C.stamps(vocab_sha256="v", teacher=t2))


def test_import_pulls_in_neither_torch_nor_the_engine() -> None:
    code = (
        "import sys; import ppo.e2e.contracts, ppo.e2e.vocab; "
        "bad = [m for m in ('torch', 'sts2_rs') if m in sys.modules]; "
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(C.__file__).resolve().parents[2]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ─── row batch ───────────────────────────────────────────────────────────────


def _tok(sec: str, **cols: int) -> tuple[np.ndarray, np.ndarray, int]:
    cat = np.asarray(list(C.SCTX_PAD) + [0] * len(C.X_CAT), np.int16)
    for k, v in cols.items():
        idx = C.XC[k] if k in C.XC else C.SCTX_SLOTS.index(k)
        cat[idx] = v
    return cat, np.zeros(C.N_CONT, np.float16), C.SEC[sec]


def make_batch() -> dict[str, np.ndarray]:
    """Two valid rows: a combat turn (two merged Strikes, one end turn) and a
    card reward (three cards and the skip)."""
    play, end = C.AK["PLAY"], C.AK["END_TURN"]
    pick, skip = C.AK["PICK_CARD_REWARD"], C.AK["SKIP_CARD_REWARD"]

    def head(dtype: int) -> list:
        sums = [_tok(s) for s in C.SECTIONS[24:29]]
        return [_tok("RUNCTX", X_DTYPE=dtype), *sums, _tok("BOSS")]

    combat = [
        *head(C.DT["COMBAT_TURN"]),
        _tok("STATE"),
        _tok("HAND", hand_slot=0),
        _tok("HAND", hand_slot=1),
        _tok("ENEMY"),
        _tok("ACT_PLAY", hand_slot=0, X_KIND=play),
        _tok("ACT_PLAY", hand_slot=1, X_KIND=play),
        _tok("ACT_END", X_KIND=end),
    ]
    ooc = [
        *head(C.DT["CARD_REWARD"]),
        _tok("DECK"),
        _tok("DECK"),
        _tok("RELIC"),
        _tok("OPT_CARD", X_KIND=pick),
        _tok("OPT_CARD", X_KIND=pick),
        _tok("OPT_CARD", X_KIND=pick),
        _tok("OPT_FIXED", X_KIND=skip),
    ]
    flags_c = [C.TF_TEACHER if i >= 7 else 0 for i in range(14)]
    flags_c[11] |= C.TF_OPTION | C.TF_REP
    flags_c[12] |= C.TF_OPTION | C.TF_HIDDEN
    flags_c[13] |= C.TF_OPTION | C.TF_REP
    flags_o = [0] * 10 + [C.TF_OPTION | C.TF_REP] * 4
    for flags in (flags_c, flags_o):
        for i in range(1, 6):
            flags[i] |= C.TF_POOLED  # the five SUM_* tokens
    toks = combat + ooc
    priv = [_tok("PRIV_DRAW"), _tok("PRIV_DRAW"), _tok("PRIV_RUN"), _tok("PRIV_RUN")]
    b = {
        "run_uid": np.asarray([7, 8], np.int64),
        "t": np.asarray([3, 4], np.int32),
        "is_combat": np.asarray([True, False]),
        "dtype": np.asarray([C.DT["COMBAT_TURN"], C.DT["CARD_REWARD"]], np.uint8),
        "reward": np.asarray([0.0, 1.0], np.float32),
        "done": np.asarray([False, True]),
        "bootstrap": np.asarray([False, False]),
        "actor": np.asarray([1, 0], np.uint8),
        "logp_behav": np.asarray([-0.3, -1.2], np.float32),
        "n_classes": np.asarray([2, 4], np.uint16),
        "action": np.asarray([0, 3], np.uint16),
        "iter": np.asarray([5, 5], np.int32),
        "v_behav": np.asarray([0.1, 0.4], np.float32),
        "cell": np.asarray([C.CELL_NONE, 2], np.uint8),
        "n_tok": np.asarray([14, 14], np.int32),
        "n_mem": np.asarray([3, 4], np.uint16),
        "n_priv": np.asarray([3, 1], np.int32),
        "n_pool": np.asarray([2, 0], np.int32),
        "n_ppool": np.asarray([0, 0], np.int32),
        "sctx_lo": np.asarray([7, 0], np.uint16),
        "n_sctx": np.asarray([7, 0], np.uint16),
        "t_src": np.asarray([1, 0], np.uint8),
        "cat": np.stack([t[0] for t in toks]),
        "cont": np.stack([t[1] for t in toks]),
        "sec": np.asarray([t[2] for t in toks], np.int8),
        "tok_flags": np.asarray(flags_c + flags_o, np.uint8),
        "mem_tok": np.asarray([11, 12, 13, 10, 11, 12, 13], np.uint16),
        "mem_class": np.asarray([0, 0, 1, 0, 1, 2, 3], np.uint16),
        "cls_tok": np.asarray([11, 13, 10, 11, 12, 13], np.uint16),
        "cls_kind": np.asarray([play, end, pick, pick, pick, skip], np.uint8),
        "tprob": np.asarray([0.75, 0.25] + [math.nan] * 4, np.float32),
        "pcat": np.stack([t[0] for t in priv]),
        "pcont": np.stack([t[1] for t in priv]),
        "psec": np.asarray([t[2] for t in priv], np.int8),
        "pool_tok": np.asarray([4, 4], np.uint16),
        "pool_vid": np.asarray([33, 34], np.uint16),
        "pool_w": np.asarray([5, 4], np.float16),
        "ppool_tok": np.zeros(0, np.uint16),
        "ppool_vid": np.zeros(0, np.uint16),
        "ppool_w": np.zeros(0, np.float16),
    }
    return b


def test_valid_batches_pass() -> None:
    C.validate_row_batch(make_batch())
    C.validate_row_batch(C.empty_row_batch())
    b = make_batch()
    C.check_compact_lossless(b["cat"], b["cont"], b["sec"])
    assert set(C.row_numpy_dtype().names) >= set(C.REQUIRED_ROW_FIELDS)
    for name in C.REQUIRED_ROW_FIELDS:
        assert name in C.FIELDS


def _set(field, value):
    def f(b):
        b[field] = value(b[field]) if callable(value) else value

    return f


def _poke(field, idx, value):
    def f(b):
        b[field] = b[field].copy()
        b[field][idx] = value

    return f


def _all(*mutations):
    def f(b):
        for m in mutations:
            m(b)

    return f


def _flag(tok: int, add: int = 0, drop: int = 0):
    def f(b):
        b["tok_flags"] = b["tok_flags"].copy()
        b["tok_flags"][tok] = (int(b["tok_flags"][tok]) | add) & ~drop & 0xFF

    return f


def _append(field, values, dtype=None):
    """Append ragged entries; they land in row 1, the batch's last row."""

    def f(b):
        b[field] = np.concatenate([b[field], np.asarray(values, dtype or b[field].dtype)])

    return f


def _add_priv(sec: str, k: int):
    """`k` more PRIV tokens of section `sec` on row 1."""
    cat, cont, s = _tok(sec)
    return _all(
        _poke("n_priv", 1, 1 + k),
        _append("pcat", np.tile(cat, (k, 1))),
        _append("pcont", np.tile(cont, (k, 1))),
        _append("psec", [s] * k),
    )


def _ppool(tok: int, vid: int = 40):
    """One PRIV pool entry on row 0's PRIV token `tok` (2 is its PRIV_RUN)."""
    return _all(
        _poke("n_ppool", 0, 1),
        _set("ppool_tok", np.asarray([tok], np.uint16)),
        _set("ppool_vid", np.asarray([vid], np.uint16)),
        _set("ppool_w", np.asarray([1.0], np.float16)),
    )


def _one_class(b):
    """Row 0 with one class (its end turn dropped from the class arrays)."""
    _poke("n_classes", 0, 1)(b)
    for k in ("cls_tok", "cls_kind", "tprob"):
        b[k] = np.delete(b[k], 1)


_EXTRA = C.MAX_OPTIONS + 1 - 4  # classes row 1 needs to pass MAX_OPTIONS
RESERVED_KIND = next(k.id for k in C.ACTION_KINDS if k.status.startswith("reserved"))
PLAY, END = C.AK["PLAY"], C.AK["END_TURN"]

# Batches the validator must accept: edge cases of make_batch().
GOOD = {
    "encounter id on a combat row": _poke("cat", (0, C.XC["X_ID_AUX"]), 40),
    "OOC duplicate on its class token": _all(
        _poke("n_mem", 1, 5), _append("mem_tok", [12]), _append("mem_class", [2])
    ),
    "PRIV pool on PRIV_RUN": _ppool(2),
    "cut row keeps its acting fields": _poke("bootstrap", 0, True),
    "practice end": _all(_poke("bootstrap", 1, True), _poke("reward", 1, 0.0)),
    "no PRIV tokens": _all(
        _poke("n_priv", 0, 0),
        _poke("n_priv", 1, 0),
        _set("pcat", lambda x: x[:0]),
        _set("pcont", lambda x: x[:0]),
        _set("psec", lambda x: x[:0]),
    ),
}


@pytest.mark.parametrize("case", sorted(GOOD))
def test_validation_accepts(case: str) -> None:
    b = make_batch()
    GOOD[case](b)
    C.validate_row_batch(b)


# Each mutation of make_batch() with the reason the validator must give.
BAD = {
    # fields and counts
    "missing field": (lambda b: b.pop("v_behav"), "missing field 'v_behav'"),
    "wrong dtype": (_set("logp_behav", lambda x: x.astype(np.float64)), "logp_behav: dtype"),
    "wrong cat width": (_set("cat", lambda x: x[:, :15]), "cat: shape"),
    "row count mismatch": (_set("reward", lambda x: x[:1]), "reward: 1 rows != 2"),
    "ragged mismatch": (_set("mem_tok", lambda x: x[:-1]), "mem_tok: length"),
    "one class": (_one_class, "n_classes outside"),
    "too many classes": (
        _all(
            _poke("n_classes", 1, C.MAX_OPTIONS + 1),
            _append("cls_tok", [13] * _EXTRA),
            _append("cls_kind", [C.AK["PICK_CARD_REWARD"]] * _EXTRA),
            _append("tprob", [math.nan] * _EXTRA),
        ),
        "n_classes outside",
    ),
    "PRIV count over the cap": (_add_priv("PRIV_DRAW", C.MAX_PRIV), "n_priv outside"),
    "pool count over the cap": (
        _all(
            _poke("n_pool", 1, C.MAX_POOL + 1),
            _append("pool_tok", [4] * (C.MAX_POOL + 1)),
            _append("pool_vid", np.arange(40, 41 + C.MAX_POOL)),
            _append("pool_w", [1.0] * (C.MAX_POOL + 1)),
        ),
        "n_pool outside",
    ),
    # row fields
    "t at the run cap": (_poke("t", 0, C.MAX_RUN_DECISIONS), "t outside"),
    "negative t": (_poke("t", 0, -1), "t outside"),
    "negative iter": (_poke("iter", 0, -1), "iter must be"),
    "action out of range": (_poke("action", 0, 2), "action >= n_classes"),
    "bad actor": (_poke("actor", 0, 3), "actor not in"),
    "bad cell": (_poke("cell", 0, len(C.CELLS)), "cell not in"),
    "bad t_src": (_poke("t_src", 0, 3), "t_src not in"),
    "run teacher acting in combat": (_poke("actor", 0, 2), "acts only on combat rows"),
    "combat teacher acting out of combat": (_poke("actor", 1, 1), "acts only on combat rows"),
    "reward NaN": (_poke("reward", 0, math.nan), "reward must be finite"),
    "v_behav NaN": (_poke("v_behav", 1, math.nan), "v_behav must be finite"),
    "positive logp": (_poke("logp_behav", 0, 0.5), "logp_behav"),
    "logp NaN": (_poke("logp_behav", 0, math.nan), "logp_behav"),
    "bootstrap row with a reward": (_poke("bootstrap", 1, True), "reward 0"),
    "done bootstrap outside practice": (
        _all(_poke("bootstrap", 1, True), _poke("reward", 1, 0.0), _poke("cell", 1, C.CELL_NONE)),
        "practice segment",
    ),
    "done bootstrap in combat": (
        _all(_poke("bootstrap", 0, True), _poke("done", 0, True), _poke("cell", 0, 2)),
        "practice segment",
    ),
    "reserved dtype": (_poke("dtype", 1, C.DT["REWARD_SCREEN"]), "not live"),
    "is_combat mismatch": (_poke("is_combat", 1, True), "is_combat disagrees"),
    # sections, flags, columns
    "PRIV in trunk": (_poke("sec", 3, C.SEC["PRIV_RUN"]), "PRIV section in the trunk"),
    "trunk section in PRIV": (_poke("psec", 0, C.SEC["HAND"]), "non-PRIV section"),
    "reserved section": (_poke("sec", 3, C.SEC["RES_SHOP_SHELF"]), "reserved section"),
    "unknown flag bit": (_flag(0, add=64), "unknown token flag"),
    "cat out of table": (_poke("cat", (0, C.XC["X_DTYPE"]), 64), "outside its table"),
    "non-finite cont": (_poke("cont", (1, 0), np.inf), "cont: non-finite"),
    # head
    "RUNCTX not first": (_poke("sec", 0, C.SEC["BOSS"]), "token 0 is not RUNCTX"),
    "SUM_* token is BOSS": (_poke("sec", 1, C.SEC["BOSS"]), "tokens 1-5"),
    "SUM_* token is DECK": (_poke("sec", 17, C.SEC["DECK"]), "tokens 1-5"),
    "SUM_* tokens out of order": (
        _all(_poke("sec", 2, C.SEC["SUM_EXHAUST"]), _poke("sec", 3, C.SEC["SUM_DISCARD"])),
        "tokens 1-5",
    ),
    "no BOSS token": (_poke("sec", 20, C.SEC["DECK"]), "0 BOSS tokens"),
    "three BOSS tokens": (
        _all(_poke("sec", 21, C.SEC["BOSS"]), _poke("sec", 22, C.SEC["BOSS"])),
        "3 BOSS tokens",
    ),
    "X_DTYPE mismatch": (
        _poke("cat", (0, C.XC["X_DTYPE"]), C.DT["CARD_REWARD"]),
        "RUNCTX X_DTYPE",
    ),
    "encounter id out of combat": (_poke("cat", (14, C.XC["X_ID_AUX"]), 40), "X_ID_AUX"),
    "event id on a card reward": (_poke("cat", (14, C.XC["X_ID_MAIN"]), 40), "X_ID_MAIN"),
    "event id in combat": (_poke("cat", (0, C.XC["X_ID_MAIN"]), 40), "X_ID_MAIN"),
    # members and classes
    "unmapped class": (_poke("mem_class", 2, 0), "not onto"),
    "classes not numbered by first occurrence": (
        _all(
            _poke("mem_class", slice(0, 3), [1, 1, 0]),
            _poke("cls_kind", slice(0, 2), [END, PLAY]),
            _poke("cls_tok", slice(0, 2), [13, 11]),
        ),
        "first occurrence",
    ),
    "member token outside the row": (_poke("mem_tok", 0, 200), "outside the row"),
    "representative not the first member": (
        _all(
            _poke("cls_tok", 0, 12),
            _flag(12, add=C.TF_REP, drop=C.TF_HIDDEN),
            _flag(11, add=C.TF_HIDDEN, drop=C.TF_REP),
        ),
        "first member's token",
    ),
    "visible duplicate": (_flag(12, drop=C.TF_HIDDEN), "TF_HIDDEN must mark"),
    "hidden representative": (_flag(11, add=C.TF_HIDDEN), "TF_HIDDEN must mark"),
    "unflagged representative": (_flag(11, drop=C.TF_REP), "TF_REP must mark"),
    "member without TF_OPTION": (_flag(13, drop=C.TF_OPTION), "TF_OPTION must mark"),
    "TF_OPTION on a non-member": (_flag(10, add=C.TF_OPTION), "TF_OPTION must mark"),
    "X_KIND mismatch": (_poke("cls_kind", 1, PLAY), "X_KIND of a member"),
    "X_KIND of a hidden member": (_poke("cat", (12, C.XC["X_KIND"]), END), "X_KIND of a member"),
    "kind not allowed for dtype": (_poke("cls_kind", 5, C.AK["MOVE"]), "not allowed for CARD_"),
    "reserved kind": (_poke("cls_kind", 5, RESERVED_KIND), "action kind .* not allowed"),
    "member on a non-action token": (
        _all(
            _poke("mem_tok", 0, 10),
            _poke("cls_tok", 0, 10),
            _flag(10, add=C.TF_OPTION | C.TF_REP),
            _flag(11, drop=C.TF_OPTION | C.TF_REP),
            _poke("cat", (10, C.XC["X_KIND"]), PLAY),
        ),
        "kind's section",
    ),
    "combat option in another kind's section": (
        _poke("sec", 13, C.SEC["ACT_PLAY"]),
        "kind's section",
    ),
    "OOC option in another kind's section": (
        _poke("sec", 27, C.SEC["OPT_CARD"]),
        "kind's section",
    ),
    # combat stream
    "block without STATE": (_poke("sec", 7, C.SEC["HAND"]), "does not start with STATE"),
    "trunk section in the block": (_poke("sec", 9, C.SEC["DECK"]), "non-S_ctx section"),
    "action token before the action block": (
        _poke("sec", 10, C.SEC["ACT_PLAY"]),
        "action token before",
    ),
    "combat members out of order": (
        _all(
            _poke("mem_tok", slice(0, 2), [12, 11]),
            _poke("cls_tok", 0, 12),
            _flag(12, add=C.TF_REP, drop=C.TF_HIDDEN),
            _flag(11, add=C.TF_HIDDEN, drop=C.TF_REP),
        ),
        "S_ctx action tokens in order",
    ),
    "sctx_lo off the head": (
        _all(_poke("sctx_lo", 0, 8), _poke("n_sctx", 0, 6)),
        "sctx_lo = 7",
    ),
    "teacher flag outside the block": (_flag(0, add=C.TF_TEACHER), "TF_TEACHER must mark"),
    "teacher flag missing in the block": (_flag(8, drop=C.TF_TEACHER), "TF_TEACHER must mark"),
    # out-of-combat stream
    "S_ctx block on an OOC row": (
        _all(_poke("sctx_lo", 1, 7), _poke("n_sctx", 1, 7)),
        "has an S_ctx block",
    ),
    "teacher flag out of combat": (_flag(21, add=C.TF_TEACHER), "never teacher tokens"),
    "OOC options out of order": (
        _all(
            _poke("cls_tok", slice(2, 6), [11, 10, 12, 13]),
            _poke("mem_tok", slice(3, 7), [11, 10, 12, 13]),
        ),
        "in class order",
    ),
    "OOC member off its class token": (
        _all(_poke("n_mem", 1, 5), _append("mem_tok", [11]), _append("mem_class", [2])),
        "does not point at its class token",
    ),
    "S_ctx section in an OOC body": (_poke("sec", 21, C.SEC["HAND"]), "out-of-combat body"),
    "OOC body out of order": (
        _all(_poke("sec", 22, C.SEC["RELIC"]), _poke("sec", 23, C.SEC["DECK"])),
        "out-of-combat body",
    ),
    "MAPNODE off a MAP row": (_poke("sec", 23, C.SEC["MAPNODE"]), "MAPNODE tokens only"),
    # pools
    "TF_POOLED off a SUM_* token": (_flag(3, drop=C.TF_POOLED), "TF_POOLED must mark"),
    "TF_POOLED on BOSS": (_flag(6, add=C.TF_POOLED), "TF_POOLED must mark"),
    "pool on a non-SUM token": (_poke("pool_tok", 0, 6), "belong to SUM_"),
    "pool owner outside the row": (_poke("pool_tok", 0, 999), "owner outside the row"),
    "duplicate pool entry": (_poke("pool_vid", 1, 33), "duplicate"),
    "pool id NONE": (_poke("pool_vid", 0, 0), "pool_vid outside"),
    "pool id past the table": (_poke("pool_vid", 0, 60000), "pool_vid outside"),
    "pool weight NaN": (_poke("pool_w", 0, math.nan), "pool_w must be finite"),
    "pool weight negative": (_poke("pool_w", 0, -3), "pool_w must be finite"),
    "pool weight zero": (_poke("pool_w", 0, 0), "pool_w must be finite"),
    "PRIV pool off PRIV_RUN": (_ppool(0), "belong to SUM_"),
    "PRIV pool owner outside the row": (_ppool(5), "owner outside the row"),
    "PRIV pool id past the table": (_ppool(2, vid=60000), "ppool_vid outside"),
    # PRIV caps
    "two PRIV_RUN tokens": (_add_priv("PRIV_RUN", 1), "more than 1 PRIV_RUN"),
    "too many PRIV_ENEMY": (_add_priv("PRIV_ENEMY", 6), "more than 5 PRIV_ENEMY"),
    "too many PRIV_DRAW": (_add_priv("PRIV_DRAW", 11), "more than 10 PRIV_DRAW"),
    # teacher probabilities
    "tprob not a distribution": (_poke("tprob", 0, 0.9), "distribution"),
    "tprob without teacher": (_poke("tprob", 2, 0.5), "NaN without a teacher"),
    "run-teacher probs in combat": (_poke("t_src", 0, 2), "run-teacher probs on a combat row"),
    "combat-teacher probs out of combat": (
        _all(_poke("t_src", 1, 1), _poke("tprob", slice(2, 6), 0.25)),
        "combat-teacher probs on an out-of-combat row",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_validation_rejects(case: str) -> None:
    mutate, reason = BAD[case]
    b = make_batch()
    mutate(b)
    with pytest.raises(C.SchemaError, match=reason):
        C.validate_row_batch(b)


def make_table() -> dict[str, np.ndarray]:
    """One iteration's fresh table (`runs=True`): run 7 stored t 2-4 and cut
    at t 4, run 8 ended at t 4, run 9 a practice segment (cell 0) ended at its
    first out-of-combat decision."""
    b = C.take_rows(make_batch(), [1, 0, 0, 0, 0, 1])
    b["run_uid"] = np.asarray([7, 7, 7, 8, 9, 9], np.int64)
    b["t"] = np.asarray([2, 3, 4, 4, 0, 1], np.int32)
    b["bootstrap"] = np.asarray([False, False, True, False, False, True])
    b["done"] = np.asarray([False, False, False, True, False, True])
    b["reward"] = np.asarray([0.0, 0.5, 0.0, 1.0, 0.2, 0.0], np.float32)
    b["cell"] = np.asarray([C.CELL_NONE] * 4 + [0, 0], np.uint8)
    return b


def test_fresh_table_passes_run_checks() -> None:
    b = make_table()
    C.validate_row_batch(b, runs=True)
    C.validate_row_batch(C.take_rows(b, [5, 3, 0, 4, 2, 1]), runs=True)  # any row order
    C.validate_row_batch(C.take_rows(b, [0, 1]))  # a minibatch: open runs are fine


BAD_RUNS = {
    "t gap": (_poke("t", 2, 5), "not distinct and contiguous"),
    "t repeat": (_poke("t", 1, 2), "not distinct and contiguous"),
    "two cells in one run": (_poke("cell", 4, 1), "different practice cells"),
    "open run": (_poke("bootstrap", 2, False), "neither done nor a bootstrap"),
    "done before the last row": (_poke("done", 4, True), "before its last row"),
}


@pytest.mark.parametrize("case", sorted(BAD_RUNS))
def test_run_checks_reject(case: str) -> None:
    mutate, reason = BAD_RUNS[case]
    b = make_table()
    mutate(b)
    C.validate_row_batch(b)  # every row is fine on its own
    with pytest.raises(C.SchemaError, match=reason):
        C.validate_row_batch(b, runs=True)


def test_take_and_concat_rows_roundtrip() -> None:
    b = make_batch()
    parts = [C.take_rows(b, [i]) for i in range(2)]
    for p in parts:
        C.validate_row_batch(p)
    back = C.concat_row_batches(parts)
    assert set(back) == set(b)
    for k, x in b.items():
        assert back[k].dtype == x.dtype, k
        np.testing.assert_array_equal(back[k], x, err_msg=k)
    rev = C.take_rows(b, [1, 0])
    C.validate_row_batch(rev)
    np.testing.assert_array_equal(C.take_rows(rev, [1, 0])["pool_vid"], b["pool_vid"])
    C.validate_row_batch(C.take_rows(b, []))
    C.validate_row_batch(C.concat_row_batches([]))


def test_compact_lossless_rejects_dead_columns() -> None:
    b = make_batch()
    cat, cont = b["cat"].copy(), b["cont"].copy()
    cont[1, C.XF["R_GOLD"]] = 1.0  # a RUNCTX column on a SUM_DRAW token
    with pytest.raises(C.SchemaError):
        C.check_compact_lossless(cat, cont, b["sec"])
    cat = b["cat"].copy()
    cat[0, C.SCTX_SLOTS.index("card")] = 5  # S_ctx content on RUNCTX
    with pytest.raises(C.SchemaError):
        C.check_compact_lossless(cat, b["cont"], b["sec"])


def test_compact_roundtrip_and_teacher_slice() -> None:
    b = make_batch()
    ccat, ccont = C.to_compact(b["cat"], b["cont"], b["sec"])
    cat, cont = C.from_compact(ccat, ccont, b["sec"])
    assert (cat == b["cat"]).all() and (cont == b["cont"]).all()
    rng = np.random.default_rng(1)
    secs = np.asarray(sorted(C.SECTION_LIVE), np.int8)[rng.integers(0, len(C.SECTION_LIVE), 500)]
    cat = np.tile(np.asarray(list(C.SCTX_PAD) + [0] * len(C.X_CAT), np.int16), (500, 1))
    cont = np.zeros((500, C.N_CONT), np.float16)
    for t, s in enumerate(secs):
        lc, lf = C.SECTION_LIVE[int(s)]
        cat[t, list(lc)] = rng.integers(0, 3, len(lc))
        cont[t, list(lf)] = rng.normal(size=len(lf))
    ccat, ccont = C.to_compact(cat, cont, secs)
    back = C.from_compact(ccat, ccont, secs)
    assert (back[0] == cat).all() and (back[1] == cont).all()
    per_tok = 2 + (ccat.nbytes + ccont.nbytes) / 500
    assert per_tok < (cat.nbytes + cont.nbytes) / 500 / 5
    with pytest.raises(C.SchemaError):
        C.from_compact(ccat[:-1], ccont, secs)
    v = C.teacher_slice(b, 0)
    assert v["cat"].shape == (7, 15) and v["cont"].shape == (7, 122)
    assert v["sec"][0] == C.SEC["STATE"] and v["n_act"] == 3
    assert (v["sec"][-3:] >= 14).all()
    with pytest.raises(C.SchemaError):
        C.teacher_slice(b, 1)
    assert C.student_mask(b).sum() == len(b["sec"]) - 1


# ─── classification ──────────────────────────────────────────────────────────


def test_ooc_classification() -> None:
    cases = [
        ([{"action": "event_choice", "idx": 0}, {"action": "event_choice", "idx": 1}], "NEOW"),
        ([{"action": "move", "col": 2}, {"action": "move", "col": 4}], "MAP"),
        (
            [
                {"action": "pick_card_reward", "idx": 0},
                {"action": "pick_card_reward", "idx": None},
                {"action": "sacrifice_card_reward"},
            ],
            "CARD_REWARD",
        ),
        ([{"action": "pick_relic_reward", "idx": 0}, {"action": "skip_relic"}], "RELIC_REWARD"),
        ([{"action": "pick_boss_relic", "idx": i} for i in range(3)], "BOSS_RELIC"),
        (
            [
                {"action": "shop_buy_card", "idx": 0},
                {"action": "shop_remove_card", "deck_idx": 3},
                {"action": "leave_shop"},
            ],
            "SHOP",
        ),
        ([{"action": "rest_option", "idx": 0}, {"action": "leave_shop"}], "REST"),
        ([{"action": "smith_card", "idx": 0}, {"action": "smith_card", "idx": 1}], "SMITH"),
        (
            [
                {"action": "potion_swap", "discard_slot": None},
                {"action": "potion_swap", "discard_slot": 0},
            ],
            "POTION_SWAP",
        ),
        ([{"action": "remove_deck_card", "idx": 0}] * 2, "CARD_REMOVE"),
        ([{"action": "upgrade_deck_card", "idx": 0}] * 2, "CARD_UPGRADE"),
        ([{"action": "transform_deck_card", "idx": 0}] * 2, "CARD_TRANSFORM"),
        ([{"action": "enchant_deck_card", "idx": 0}] * 2, "CARD_ENCHANT"),
        ([{"action": "collect_treasure"}], "TREASURE"),
    ]
    for actions, want in cases:
        d, kinds = C.classify_ooc(actions, event_name="Neow" if want == "NEOW" else "Other")
        assert C.DECISION_TYPES[d].name == want
        assert all(k in C.LIVE_ACTION_KINDS for k in kinds)
    d, _ = C.classify_ooc([{"action": "event_choice", "idx": 0}] * 2, event_name="AbyssalBaths")
    assert d == C.DT["EVENT"]
    # Scroll Boxes bundles and the Sea Glass confirm.
    d, kinds = C.classify_ooc([{"action": "pick_card_reward", "idx": 0}] * 2, bundles=True)
    assert d == C.DT["CARD_BUNDLE"] and kinds == [C.AK["PICK_CARD_BUNDLE"]] * 2
    grant = [{"action": "grant_deck_card", "idx": i} for i in range(4)]
    d, kinds = C.classify_ooc(grant, grant_n_options=3)
    assert d == C.DT["CARD_GRANT"] and kinds[-1] == C.AK["GRANT_CONFIRM"]
    for bad in (
        [{"action": "unhandled"}],
        [{"action": "unknown"}],
        [{"action": "move", "col": 1}, {"action": "smith_card", "idx": 0}],
        [],
    ):
        with pytest.raises(C.SchemaError):
            C.classify_ooc(bad)
    with pytest.raises(C.SchemaError):
        C.classify_ooc([{"action": "pick_card_reward", "idx": None}], bundles=True)
    with pytest.raises(C.SchemaError):
        C.classify_ooc(grant)  # grant_n_options missing


def test_combat_classification() -> None:
    legal = [
        {"action": "play", "card_index": 0, "target_index": None},
        {"action": "play", "card_index": 1, "target_index": 0},
        {"action": "drink_potion", "slot": 0, "target_index": None},
        {"action": "end_turn"},
    ]
    d, kinds = C.classify_combat(legal, None)
    assert d == C.DT["COMBAT_TURN"]
    assert kinds == [C.AK["PLAY"], C.AK["PLAY_TARGET"], C.AK["POTION"], C.AK["END_TURN"]]
    choice = [
        {"action": "choose", "index": 0, "selection": []},
        {"action": "choose", "index": 0, "selection": [0]},
    ]
    d, kinds = C.classify_combat(choice, {"kind": "discard_from_hand_and_draw"})
    assert d == C.DT["COMBAT_CHOICE"] and kinds == [C.AK["CHOOSE_DONE"], C.AK["CHOOSE"]]
    with pytest.raises(C.SchemaError):
        C.classify_combat(choice, None)  # a choose without a pending choice
    with pytest.raises(C.SchemaError):
        C.classify_combat([{"action": "dance"}], None)


# ─── dedupe (T6) ─────────────────────────────────────────────────────────────

STRIKE = {"id": "STRIKE_IRONCLAD", "kind": "Attack", "base_cost": 1, "base_damage": 6}


def test_t6_card_identity() -> None:
    sharp = dict(STRIKE, enchantment="Sharp", enchantment_amount=2)
    fresh = {"id": "RAMPAGE", "kind": "Attack", "base_cost": 1, "base_damage": 9}
    played = dict(fresh, base_damage=14)
    assert C.card_key(STRIKE) == C.card_key(copy.deepcopy(STRIKE))
    assert C.card_key(STRIKE) != C.card_key(sharp)
    assert C.card_key(fresh) != C.card_key(played)
    hand = [STRIKE, sharp, dict(STRIKE), fresh, played]
    legal = [{"action": "play", "card_index": i, "target_index": 0} for i in range(5)]
    legal += [{"action": "end_turn"}]
    cls, reps = C.merge_classes(C.combat_class_keys(legal, hand, []))
    assert cls.tolist() == [0, 1, 0, 2, 3, 4] and reps.tolist() == [0, 1, 3, 4, 5]
    # Targets stay apart.
    two = [{"action": "play", "card_index": 0, "target_index": t} for t in (0, 1)]
    assert len(set(C.combat_class_keys(two, hand, []))) == 2
    # Potions merge by (id, target).
    pots = [{"action": "drink_potion", "slot": s, "target_index": None} for s in (0, 1, 2)]
    keys = C.combat_class_keys(pots, hand, ["FirePotion", "FirePotion", "BlockPotion"])
    assert C.merge_classes(keys)[0].tolist() == [0, 0, 1]


def test_ooc_dedupe() -> None:
    obs = {
        "deck": [("StrikeIronclad", False), ("StrikeIronclad", False), ("Bash", True)],
        "deck_enchantments": [None, ["SHARP", 2], None],
        "card_removal_options": [("StrikeIronclad", False)] * 2 + [("Bash", True)],
        "card_removal_option_enchantments": [None, None, None],
        "shop": {"cards": [{"id": "Cinder", "idx": 0, "price": 51}], "relics": [], "potions": []},
        "potions": ["FirePotion", "FirePotion"],
    }
    acts = [{"action": "remove_deck_card", "idx": i} for i in range(3)]
    kinds = [C.AK["REMOVE_DECK_CARD"]] * 3
    cls, _ = C.merge_classes(C.ooc_class_keys(kinds, C.ooc_payloads(obs, acts, kinds)))
    assert cls.tolist() == [0, 0, 1]
    shop = [{"action": "shop_remove_card", "deck_idx": i} for i in range(3)]
    shop += [{"action": "shop_buy_card", "idx": 0}, {"action": "leave_shop"}]
    _, kinds = C.classify_ooc(shop)
    cls, _ = C.merge_classes(C.ooc_class_keys(kinds, C.ooc_payloads(obs, shop, kinds)))
    assert cls.tolist() == [0, 1, 2, 3, 4], "the enchanted Strike stays apart"
    swap = [{"action": "potion_swap", "discard_slot": s} for s in (None, 0, 1)]
    _, kinds = C.classify_ooc(swap)
    cls, _ = C.merge_classes(C.ooc_class_keys(kinds, C.ooc_payloads(obs, swap, kinds)))
    assert cls.tolist() == [0, 1, 1]
    ev = [{"action": "event_choice", "idx": i} for i in range(2)]
    _, kinds = C.classify_ooc(ev, event_name="X")
    assert C.merge_classes(C.ooc_class_keys(kinds, [None, None]))[0].tolist() == [0, 1]


def test_merge_probs_is_log_sum_exp_and_normalized() -> None:
    p = C.merge_probs(np.log([1.0, 1.0, 2.0]), np.asarray([0, 0, 1]), 2)
    assert np.allclose(p, [0.5, 0.5]) and p.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(2, 30))
        cls = rng.integers(0, 5, m)
        _, cls = np.unique(cls, return_inverse=True)
        lg = rng.normal(size=m) * 3
        p = C.merge_probs(lg, cls, int(cls.max()) + 1)
        assert abs(float(p.sum()) - 1.0) < 1e-5  # T8
        q = np.exp(lg - lg.max())
        q /= q.sum()
        assert np.allclose(p, np.bincount(cls, q), atol=1e-6)


# ─── env decision ────────────────────────────────────────────────────────────


class _Policy:
    """A minimal PolicySource (the env hands the tokenizer one of these)."""

    def belief(self):
        return None

    def legal(self):
        return []

    def run(self):
        return None


class _Priv:
    def true_state(self):
        return None

    def act_rooms(self):
        return {}

    def future(self):
        return []


class _Both(_Policy, _Priv):
    """Exposes PRIV accessors to the policy: must be rejected."""


OBS = {"deck": [], "deck_enchantments": []}


def make_decisions() -> dict[str, C.EnvDecision]:
    """One valid decision per shape: combat turn, combat choice, card reward,
    Sea Glass grant (with its confirm), Neow."""
    hand = [STRIKE, dict(STRIKE), {"id": "BASH", "kind": "Attack", "base_cost": 2}]
    legal = [{"action": "play", "card_index": i, "target_index": 0} for i in range(3)]
    legal += [{"action": "drink_potion", "slot": 0, "target_index": None}, {"action": "end_turn"}]
    _, kinds = C.classify_combat(legal, None)
    keys = C.combat_class_keys(legal, hand, ["FirePotion"])
    out = {
        "combat_turn": C.make_decision(
            run_uid=7,
            t=3,
            dtype=C.DT["COMBAT_TURN"],
            actions=legal,
            kinds=kinds,
            keys=keys,
            policy=_Policy(),
            priv=_Priv(),
            fight_turn=1,
        )
    }
    choice = {"kind": "exhaust_from_hand", "min_select": 0, "max_select": 2, "can_skip": True}
    pick = [{"action": "choose", "selection": []}, {"action": "choose", "selection": [1]}]
    _, kinds = C.classify_combat(pick, choice)
    out["combat_choice"] = C.make_decision(
        run_uid=7,
        t=4,
        dtype=C.DT["COMBAT_CHOICE"],
        actions=pick,
        kinds=kinds,
        keys=["done", "pick 1"],
        policy=_Policy(),
        sel=C.Selection(min=0, max=2, picked=1, can_skip=True),
        fight_turn=1,
        fight_decision=1,
    )
    reward = [{"action": "pick_card_reward", "idx": i} for i in range(3)]
    reward += [{"action": "pick_card_reward", "idx": None}]
    _, kinds = C.classify_ooc(reward)
    out["card_reward"] = C.make_decision(
        run_uid=7,
        t=5,
        dtype=C.DT["CARD_REWARD"],
        actions=reward,
        kinds=kinds,
        keys=["a", "b", "a", "skip"],
        policy=_Policy(),
        obs=OBS,
        metrics={},
    )
    grant = [{"action": "grant_deck_card", "idx": i} for i in range(4)]
    _, kinds = C.classify_ooc(grant, grant_n_options=3)
    out["sea_glass"] = C.make_decision(
        run_uid=7,
        t=6,
        dtype=C.DT["CARD_GRANT"],
        actions=grant,
        kinds=kinds,
        keys=["x", "y", "z", "confirm"],
        policy=_Policy(),
        sel=C.Selection(min=0, max=3, picked=2, can_skip=True),
        obs=OBS,
        metrics={},
    )
    neow = [{"action": "event_choice", "idx": i} for i in range(3)]
    _, kinds = C.classify_ooc(neow, event_name="Neow")
    out["neow"] = C.make_decision(
        run_uid=7,
        t=0,
        dtype=C.DT["NEOW"],
        actions=neow,
        kinds=kinds,
        keys=["0", "1", "2"],
        policy=_Policy(),
        event_name="Neow",
        obs=OBS,
        metrics={},
    )
    return out


def test_valid_decisions_pass() -> None:
    ds = make_decisions()
    for d in ds.values():
        C.validate_decision(d)
    turn = ds["combat_turn"]
    assert turn.is_combat and turn.head == C.HEAD_COMBAT and turn.n_members == 5
    assert turn.member_class.tolist() == [0, 0, 1, 2, 3] and turn.n_classes == 4
    assert turn.class_kinds().tolist() == [
        C.AK["PLAY_TARGET"],
        C.AK["PLAY_TARGET"],
        C.AK["POTION"],
        C.AK["END_TURN"],
    ]
    assert turn.member_action(1) == {"action": "play", "card_index": 2, "target_index": 0}
    reward = ds["card_reward"]
    assert not reward.is_combat and reward.head == C.HEAD_OOC and reward.n_classes == 3
    assert isinstance(_Policy(), C.PolicySource) and not isinstance(_Policy(), C.PrivSource)


def _dset(name: str, **kw):
    def f(ds):
        d = ds[name]
        for k, v in kw.items():
            setattr(d, k, v)
        return d

    return f


BAD_DECISIONS = {
    "dtype_none": _dset("card_reward", dtype=0),
    "dtype_spare": _dset("card_reward", dtype=C.DT["OOC_SPARE_30"]),
    "dtype_reserved": _dset("card_reward", dtype=C.DT["REWARD_SCREEN"]),
    "kind_not_allowed": _dset("card_reward", dtype=C.DT["SHOP"]),
    "kinds_wrong_dtype": lambda ds: (
        setattr(ds["neow"], "kinds", ds["neow"].kinds.astype(np.int64)) or ds["neow"]
    ),
    "kinds_short": lambda ds: setattr(ds["neow"], "kinds", ds["neow"].kinds[:2]) or ds["neow"],
    "raw_action_mismatch": lambda ds: (
        ds["card_reward"].actions.__setitem__(0, {"action": "move", "col": 1}) or ds["card_reward"]
    ),
    "combat_kind_mismatch": lambda ds: (
        ds["combat_turn"].actions.__setitem__(4, {"action": "play", "card_index": 0})
        or ds["combat_turn"]
    ),
    "keys_short": _dset("neow", keys=["0", "1"]),
    "class_numbering": lambda ds: (
        setattr(ds["neow"], "member_class", np.asarray([1, 0, 2], np.uint16)) or ds["neow"]
    ),
    "rep_not_first": lambda ds: (
        setattr(ds["card_reward"], "rep", np.asarray([2, 1, 3], np.uint16)) or ds["card_reward"]
    ),
    "class_map_dtype": lambda ds: (
        setattr(ds["neow"], "member_class", ds["neow"].member_class.astype(np.int64)) or ds["neow"]
    ),
    "one_class": lambda ds: C.make_decision(
        run_uid=1,
        t=0,
        dtype=C.DT["MAP"],
        actions=[{"action": "move", "col": 1}, {"action": "move", "col": 1}],
        kinds=[C.AK["MOVE"]] * 2,
        keys=["same", "same"],
        policy=_Policy(),
        obs=OBS,
        metrics={},
    ),
    "too_many_classes": lambda ds: C.make_decision(
        run_uid=1,
        t=0,
        dtype=C.DT["MAP"],
        actions=[{"action": "move", "col": i} for i in range(C.MAX_OPTIONS + 1)],
        kinds=[C.AK["MOVE"]] * (C.MAX_OPTIONS + 1),
        keys=[str(i) for i in range(C.MAX_OPTIONS + 1)],
        policy=_Policy(),
        obs=OBS,
        metrics={},
    ),
    "class_mixes_kinds": _dset(
        "card_reward",
        keys=["a", "b", "c", "a"],
        member_class=np.asarray([0, 1, 2, 0], np.uint16),
        rep=np.asarray([0, 1, 2], np.uint16),
    ),
    "selection_on_reward": _dset("card_reward", sel=C.Selection(min=1, max=1)),
    "selection_min_gt_max": _dset("combat_choice", sel=C.Selection(min=3, max=2, can_skip=True)),
    "selection_can_skip_wrong": _dset("combat_choice", sel=C.Selection(min=0, max=2)),
    "grant_max_wrong": _dset("sea_glass", sel=C.Selection(min=0, max=15, can_skip=True)),
    "event_name_missing": _dset("neow", event_name=None),
    "neow_named_other": _dset("neow", event_name="AbyssalBaths"),
    "event_name_on_reward": _dset("card_reward", event_name="Neow"),
    "obs_missing": _dset("card_reward", obs=None),
    "metrics_missing": _dset("card_reward", metrics=None),
    "obs_in_combat": _dset("combat_turn", obs=OBS),
    "cell": _dset("combat_turn", cell=6),
    "t_over_run_cap": _dset("card_reward", t=C.MAX_RUN_DECISIONS),
    "fight_turn_zero": _dset("combat_turn", fight_turn=0),
    "fight_turn_over_cap": _dset("combat_turn", fight_turn=C.MAX_FIGHT_TURNS + 1),
    "fight_decision_over_cap": _dset("combat_turn", fight_decision=C.MAX_FIGHT_DECISIONS),
    "fight_counter_ooc": _dset("card_reward", fight_turn=3),
    "policy_exposes_priv": _dset("combat_turn", policy=_Both()),
    "policy_missing": _dset("combat_turn", policy=None),
    "priv_wrong_type": _dset("combat_turn", priv=_Policy()),
}


# What each bad case must be rejected for (a regex on the message), so a case
# cannot pass by tripping an unrelated check.
BAD_DECISION_REASON = {
    "dtype_none": "outside 1",
    "dtype_spare": "is spare",
    "dtype_reserved": "is reserved",
    "kind_not_allowed": "cannot offer",
    "kinds_wrong_dtype": "kinds must be uint8",
    "kinds_short": "kinds must be uint8",
    "raw_action_mismatch": "cannot be kind",
    "combat_kind_mismatch": "is not kind",
    "keys_short": "keys for",
    "class_numbering": "differ from merge_classes",
    "rep_not_first": "differ from merge_classes",
    "class_map_dtype": "must be uint16",
    "one_class": "merged classes outside",
    "too_many_classes": "merged classes outside",
    "class_mixes_kinds": "mixes action kinds",
    "selection_on_reward": "a selection on",
    "selection_min_gt_max": "selection Selection",
    "selection_can_skip_wrong": "can_skip",
    "grant_max_wrong": "offered cards",
    "event_name_missing": "event_name",
    "neow_named_other": "event 'AbyssalBaths'",
    "event_name_on_reward": "event_name",
    "obs_missing": "obs and metrics",
    "metrics_missing": "obs and metrics",
    "obs_in_combat": "obs and metrics",
    "cell": "practice cell",
    "t_over_run_cap": "run cap",
    "fight_turn_zero": "fight_turn",
    "fight_turn_over_cap": "fight_turn",
    "fight_decision_over_cap": "fight cap",
    "fight_counter_ooc": "fight counters",
    "policy_exposes_priv": "exposes PRIV",
    "policy_missing": "not a PolicySource",
    "priv_wrong_type": "not a PrivSource",
}


def test_every_bad_decision_has_a_reason() -> None:
    assert set(BAD_DECISION_REASON) == set(BAD_DECISIONS)


@pytest.mark.parametrize("case", sorted(BAD_DECISIONS))
def test_decision_validation_rejects(case: str) -> None:
    d = BAD_DECISIONS[case](make_decisions())
    with pytest.raises(C.SchemaError, match=BAD_DECISION_REASON[case]):
        C.validate_decision(d)


def test_reserved_decisions_need_allow_reserved() -> None:
    acts = [{"action": "take_reward_potion"}, {"action": "skip_reward_potion"}]
    kinds = [C.AK["TAKE_REWARD_POTION"], C.AK["SKIP_REWARD_POTION"]]
    d = C.make_decision(
        run_uid=1,
        t=0,
        dtype=C.DT["REWARD_SCREEN"],
        actions=acts,
        kinds=kinds,
        keys=["take", "skip"],
        policy=_Policy(),
        obs=OBS,
        metrics={},
    )
    with pytest.raises(C.SchemaError):
        C.validate_decision(d)
    C.validate_decision(d, allow_reserved=True)


# ─── vocabulary ──────────────────────────────────────────────────────────────


def test_vocab_table_is_stable() -> None:
    rows = V.read_tsv()
    v = V.Vocab(rows)
    assert v.sha256 == VOCAB_SHA256_GOLDEN
    assert v.n_used <= C.VOCAB_ROWS
    assert rows[0] == V.Row(0, "special", "NONE")
    for reg, i in V.UNK_ID.items():
        assert v.id(reg, "NO_SUCH_THING_EVER") == i
        assert V.Row(i, "special", f"UNK_{reg.upper()}") in rows
    assert min(r.id for r in rows if r.registry != "special") == V.FIRST_BLOCK_ID
    counts = Counter(r.registry for r in rows)
    assert counts["card"] == 599 and counts["relic"] == 299 and counts["potion"] == 64
    assert counts["encounter"] == 90 and counts["event"] == 69 and counts["rest"] == 9
    # Vocabulary 2.1 appended the event fights' encounters after the move block.
    assert [v.id("encounter", n) for n in V.EVENT_ENCOUNTERS] == list(range(1939, 1948))
    assert v.n_used == 1948 and len(rows) == 1928
    assert v.id("card", "StrikeIronclad") == v.id("card", "CARD.STRIKE_IRONCLAD")
    assert v.id("card", "STRIKE_IRONCLAD") == v.block("card")[0] or v.has("card", "Strike_Ironclad")
    assert v.id("relic", "RELIC.BURNING_BLOOD") == v.id("relic", "BurningBlood")
    assert v.id("enchantment", "Tezcatara's Ember") == v.id("enchantment", "TEZCATARAS_EMBER")
    assert v.id("event", "Neow") != V.UNK_ID["event"]
    assert v.id("card", None) == V.NONE_ID
    text = V.TSV_PATH.read_text(encoding="utf-8")
    assert V.sha256(rows) == V.Vocab(V.read_tsv()).sha256
    assert "\r" not in text


@needs_rs
def test_vocab_is_append_only_complete_for_the_engine() -> None:
    old = V.read_tsv()
    new = V.build_rows(V.engine_names(), old)
    added = [(r.registry, r.name) for r in new[len(old) :]]
    assert not added, (
        f"engine has names the table lacks; run `python -m ppo.e2e.vocab --append`: {added[:10]}"
    )
    assert new == old


def test_vocab_rejects_canonical_collisions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="share the canonical name"):
        V.build_rows({"card": ["FooBar", "FOO_BAR"]})
    rows = V.build_rows({"card": ["FooBar"]})
    assert V.build_rows({"card": ["FooBar"]}, rows) == rows  # the same spelling is skipped
    old = V.read_tsv()
    held = next(r.name for r in old if r.registry == "card" and r.name == "STRIKE_IRONCLAD")
    with pytest.raises(ValueError, match=repr(held)):
        V.build_rows({"card": ["StrikeIronclad"]}, old)  # would silently share its id
    monkeypatch.setattr(V, "engine_names", lambda: {"card": ["StrikeIronclad"]})
    monkeypatch.setattr(V, "write_tsv", lambda *a, **k: pytest.fail("wrote the table"))
    for flag in ("--check", "--append"):
        assert V.main([flag]) == 1
        assert "share the canonical name" in capsys.readouterr().err


# ─── event previews ──────────────────────────────────────────────────────────


def _info(**kw: int | None) -> dict:
    d = dict.fromkeys(C.EVENT_DELTA_FIELDS, 0)
    d.update(kw)
    return {"option_id": "A", "label": "x", **d}


def test_honest_event_previews() -> None:
    plain = _info(gold_delta=50, hp_delta=-3, cards_gained=1)
    assert C.honest_event_deltas(plain) == {k: plain[k] for k in C.EVENT_DELTA_FIELDS}
    relic = _info(gold_delta=50, max_hp_delta=7, potions_gained=1, relics_gained=3, cards_lost=1)
    d = C.honest_event_deltas(relic)
    assert d is not None and d["relics_gained"] == 1 and d["cards_lost"] == 1
    assert all(d[k] == 0 for k in C.EVENT_RELIC_MASKED)
    assert C.honest_event_deltas(C.honest_event_option(relic)) == d  # idempotent
    leave = _info(gold_delta=None, hp_delta=None, max_hp_delta=None)
    assert C.honest_event_deltas(leave) is None
    assert C.honest_event_option(leave) == {"option_id": "A", "label": "x"}
    cols = C.event_delta_columns(relic)
    assert list(cols) == [c for c, _, _ in C.EVENT_DELTA_COLUMNS]
    assert cols["O_EV_RELICS_GAIN"] == pytest.approx(1.0) and not cols["O_EV_GOLD"]
    assert cols["O_EV_CARDS_LOSE"] == pytest.approx(1 / 3)
    assert C.event_delta_columns(plain)["O_EV_GOLD"] == pytest.approx(0.5)
    assert set(C.event_delta_columns(leave).values()) == {0.0}
    assert {c for c, _, _ in C.EVENT_DELTA_COLUMNS} <= set(C.XF)
    obs = {"seed": 5, "deck": [], "event_options": [relic, leave]}
    h = C.honest_obs(obs)
    assert set(h) == {"deck", "event_options"}
    assert h["event_options"] == [C.honest_event_option(relic), C.honest_event_option(leave)]
    assert obs["seed"] == 5 and obs["event_options"][0]["gold_delta"] == 50, "input unchanged"
    assert C.honest_obs({"seed": 1, "event_options": None}) == {"event_options": None}
    assert C.POLICY_FILTERS["rm.event_option_infos"] is C.honest_event_options
    assert set(C.POLICY_FILTERS) <= C.POLICY_CALLS
    assert C.POLICY_CALLS.isdisjoint(C.PRIV_CALLS)


# Everything an event option shows the player.
EVENT_VISIBLE = ("option_id", "label", "description", "enabled", "vars", "title_vars", "loc_key")


def _event_options(rm, ev: str, path: list, depth: int, out: list) -> None:
    """Every option on the forced event's page and, `depth` pages deep, on the
    pages its options lead to, as (everything visible, the option)."""
    for i, o in enumerate(rm.event_option_infos() or []):
        out.append((json.dumps([ev, path, i, *(o.get(k) for k in EVENT_VISIBLE)], default=str), o))
        if depth and o.get("enabled", True):
            c = rm.clone()
            c.take_action({"action": "event_choice", "idx": i})
            if c.current_event_name() == ev and c.event_option_infos():
                # The label is part of the visible path: Endless Conveyor's
                # grab row names the dish taken, so two grabs of different
                # dishes are different histories.
                step = f"{o.get('option_id')}[{o.get('label')}]"
                _event_options(c, ev, [*path, step], depth - 1, out)


@needs_rs
def test_event_previews_are_honest() -> None:
    """Section 2.7: across 30 seeds, two options that look the same to the
    player (same event, page path, position and visible fields) have the same
    honest preview. The raw preview fails this: the relic an option grants is
    rolled from the hidden run RNG and picked up on the preview clone."""
    import sts2_rs

    raw: defaultdict[str, set] = defaultdict(set)
    honest: defaultdict[str, set] = defaultdict(set)
    for ev in sts2_rs.list_registry_names("event"):
        for seed in range(1, 31):
            rm = sts2_rs.RunManager(
                seed,
                character="Ironclad",
                ascension_level=10,
                neow_start=True,
                act1="roll",
                game_version="v0.111.0",
                mode="agent",
                interactive_potion_swap=True,
            )
            assert rm.force_event(ev) is not None, ev
            opts: list = []
            _event_options(rm, ev, [], 2, opts)
            for vis, o in opts:
                raw[vis].add(tuple(o.get(k) for k in C.EVENT_DELTA_FIELDS))
                d = C.honest_event_deltas(o)
                honest[vis].add(None if d is None else tuple(d.values()))
    assert len(raw) > 400, "the walk reaches the event pages"
    leaks = {k: v for k, v in honest.items() if len(v) > 1}
    assert not leaks, list(leaks.items())[:5]
    n_raw = sum(len(v) > 1 for v in raw.values())
    print(json.dumps({"visible_groups": len(raw), "raw_leaking_groups": n_raw}))


# ─── live walk ───────────────────────────────────────────────────────────────

# Six walks plus one (162,000,356) that reaches Sea Glass's optional 0-15 grid.
LIVE_SEEDS = (*range(162_000_200, 162_000_206), 162_000_356)
EXPECT_SEEN = {
    "COMBAT_TURN",
    "COMBAT_CHOICE",
    "NEOW",
    "MAP",
    "CARD_REWARD",
    "EVENT",
    "REST",
    "SHOP",
    "SMITH",
    "BOSS_RELIC",
}


def _walk(seed: int, stats: Counter, check_tokens: bool) -> str:
    from ppo.combat_bench.oracle import combat_to_rm
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.snapshot import choose_outcome_key, honest_snapshot, obs_actions
    from ppo.env import Decision, PpoRunEnv, Terminal

    vocab = V.load()
    rng = random.Random(seed)
    env = PpoRunEnv(
        seed=seed,
        ascension=10,
        character="Ironclad",
        combat="yield",
        act1="roll",
        game_version="v0.111.0",
    )
    rm = env.rm
    out = env.reset()
    fight_steps = 0
    t = 0  # stored decisions so far in the run (the row's `t`)
    fight_t = 0  # stored decisions so far in the fight
    grant_picked = 0  # grant picks applied since the grant opened
    while not isinstance(out, Terminal):
        if isinstance(out, Decision):
            # Invincible: a random policy must survive to act 3 to cover it.
            if rm.max_hp < 900:
                rm.set_player(hp=999, max_hp=999)
            acts = out.actions
            names = {a["action"] for a in acts}
            ev = rm.current_event_name() if "event_choice" in names else None
            grant_n = len(rm.card_grant_options()) if "grant_deck_card" in names else None
            bundles = bool(out.obs.get("card_reward_bundles")) and "pick_card_reward" in names
            d, kinds = C.classify_ooc(acts, event_name=ev, bundles=bundles, grant_n_options=grant_n)
            assert d in C.LIVE_DTYPES and set(kinds) <= C.LIVE_ACTION_KINDS
            keys = C.ooc_class_keys(kinds, C.ooc_payloads(out.obs, acts, kinds))
            cls, _ = C.merge_classes(keys)
            n_cls = int(cls.max()) + 1
            assert n_cls <= C.MAX_OPTIONS, (C.DECISION_TYPES[d].name, n_cls)
            if d != C.DT["CARD_GRANT"]:
                grant_picked = 0
            sel = C.NO_SELECTION
            if d == C.DT["CARD_GRANT"]:
                sel = C.Selection(
                    min=int(rm.pending_card_grant_min),
                    max=int(grant_n or 0),
                    picked=grant_picked,
                    can_skip=C.AK["GRANT_CONFIRM"] in kinds,
                )
            if n_cls >= 2:
                C.validate_decision(
                    C.make_decision(
                        run_uid=seed,
                        t=t,
                        dtype=d,
                        actions=acts,
                        kinds=kinds,
                        keys=keys,
                        policy=_Policy(),
                        priv=_Priv(),
                        sel=sel,
                        event_name=ev,
                        obs=out.obs,
                        metrics=out.metrics,
                    )
                )
                stats["decisions_validated"] += 1
                t += 1
            else:
                stats["ooc_one_class"] += 1
            stats[C.DECISION_TYPES[d].name] += 1
            stats["max_ooc_classes"] = max(stats["max_ooc_classes"], n_cls)
            # Row size without MAPNODE: head + distinct deck cards + relics +
            # potions + an OFFER + one token per class.
            m = out.obs["map"]
            n_boss = 1 + bool(m.get("second_boss"))
            deck = {
                json.dumps([c, e], default=str)
                for c, e in zip(out.obs["deck"], out.obs["deck_enchantments"], strict=False)
            }
            n_pot = sum(p is not None for p in out.obs["potions"])
            offer = int(d in (C.DT["POTION_SWAP"], C.DT["CARD_ENCHANT"]))
            n_tok = 6 + n_boss + len(deck) + len(out.obs["relics"]) + n_pot + offer + n_cls
            stats["max_ooc_tokens_no_map"] = max(stats["max_ooc_tokens_no_map"], n_tok)
            stats["sum_ooc_tokens_no_map"] += n_tok
            stats["n_ooc"] += 1
            if d == C.DT["CARD_GRANT"]:
                confirm = C.AK["GRANT_CONFIRM"] in kinds
                if confirm:
                    assert rm.pending_card_grant_min == 0, "a confirm means an optional pick"
                    stats["grant_optional"] += 1
                stats["grant_max_options"] = max(stats["grant_max_options"], grant_n or 0)
            stats["max_ooc_members"] = max(stats["max_ooc_members"], len(acts))
            if ev is not None:
                assert vocab.has("event", ev), ev
                _check_event_preview(rm, ev, out.obs, stats)
            if d == C.DT["REST"]:
                for oid, _ in rm.rest_site_options():
                    assert vocab.has("rest", oid), oid
            for cid, _ in out.obs["deck"]:
                assert vocab.has("card", cid), cid
            for r in out.obs["relics"]:
                assert vocab.has("relic", r), r
            for p in out.obs["potions"]:
                assert p is None or vocab.has("potion", p), p
            for b in (m.get("boss"), m.get("second_boss")):
                if b:
                    assert vocab.has("encounter", b["id"]), b
            if m.get("second_boss"):
                stats["second_boss_seen"] += 1
            pick = rng.randrange(len(acts))
            if d == C.DT["CARD_GRANT"] and kinds[pick] == C.AK["GRANT_DECK_CARD"]:
                grant_picked += 1
            out = env.step(pick)
            continue
        if rm.is_over() or rm.phase() != "combat":
            out = env.resume()
            fight_steps = fight_t = 0
            continue
        cs = rm.combat_state()
        if int(cs.get_state().get("round") or 0) > C.MAX_FIGHT_TURNS:
            # The evaluator caps a fight past MAX_FIGHT_TURNS as a loss
            # (`ppo.e2e.evaluate`); an invincible random policy can stall a
            # boss that long with an exhausted deck.
            return "capped"
        if cs.get_state().get("player", {}).get("hp", 999) < 500:
            rm.set_player(hp=999, max_hp=999)
            cs = rm.combat_state()
        legal = cs.legal_actions()
        fight_steps += 1
        assert fight_steps <= 20 * C.MAX_FIGHT_DECISIONS, "runaway fight"
        if len(legal) > 1:
            pc = cs.pending_choice()
            d, kinds = C.classify_combat(legal, pc)
            assert set(kinds) <= C.LIVE_ACTION_KINDS
            stats[C.DECISION_TYPES[d].name] += 1
            stats["max_combat_members"] = max(stats["max_combat_members"], len(legal))
            if check_tokens:
                acts, keys, bchoice, rnd = _check_combat_tokens(
                    cs, legal, stats, T, honest_snapshot, obs_actions, choose_outcome_key, vocab
                )
                bd, bkinds = C.classify_combat(acts, bchoice)
                assert (bd, bkinds) == (d, kinds), "belief-view re-indexing keeps kinds"
                sel = C.NO_SELECTION
                if bchoice is not None:
                    n_choose = bkinds.count(C.AK["CHOOSE"])
                    sel = C.Selection(
                        min=int(bchoice["min_select"]),
                        max=int(bchoice["max_select"]),
                        picked=len(bchoice["cards"]) - n_choose,
                        can_skip=C.AK["CHOOSE_DONE"] in bkinds,
                    )
                    stats["choice_reopened"] += int(sel.picked > 0)
                if len(set(keys)) >= 2:
                    C.validate_decision(
                        C.make_decision(
                            run_uid=seed,
                            t=t,
                            dtype=bd,
                            actions=acts,
                            kinds=bkinds,
                            keys=keys,
                            policy=_Policy(),
                            priv=_Priv(),
                            sel=sel,
                            fight_turn=rnd,
                            fight_decision=fight_t,
                        )
                    )
                    stats["decisions_validated"] += 1
                    t += 1
                    fight_t += 1
                else:
                    stats["combat_one_class"] += 1
        plays = [a for a in legal if a["action"] == "play"]
        a = rng.choice(plays) if plays and rng.random() < 0.9 else rng.choice(legal)
        rm.take_action(combat_to_rm(a))
        env.combat_step()
        if rm.is_over() or rm.phase() != "combat":
            out = env.resume()
            fight_steps = fight_t = 0
    return out.outcome


def _check_event_preview(rm, ev: str, obs: dict, stats: Counter, k: int = 6) -> None:
    """Section 2.7 on the live run: on clones whose hidden run RNG is reseeded,
    an option that shows the player the same has the same honest preview."""
    base = rm.event_option_infos() or []
    for o in C.honest_obs(obs).get("event_options") or []:
        assert all(math.isfinite(v) for v in C.event_delta_columns(o).values())
    stats["ev_options"] += len(base)
    stats["ev_masked"] += sum(int(o.get("relics_gained") or 0) > 0 for o in base)
    for j in range(k):
        c = rm.clone()
        c.reseed_run(600_000_000 + j)
        alt = c.event_option_infos() or []
        if len(alt) != len(base):
            stats["ev_reseed_count_moved"] += 1
            continue
        for o, p in zip(base, alt, strict=True):
            if [o.get(x) for x in EVENT_VISIBLE] != [p.get(x) for x in EVENT_VISIBLE]:
                stats["ev_reseed_visible_moved"] += 1
                continue
            stats["ev_reseed_pairs"] += 1
            raw_o = [o.get(f) for f in C.EVENT_DELTA_FIELDS]
            stats["ev_reseed_raw_moved"] += int(raw_o != [p.get(f) for f in C.EVENT_DELTA_FIELDS])
            assert C.honest_event_deltas(o) == C.honest_event_deltas(p), (ev, o, p)


def _check_combat_tokens(
    cs, legal, stats, T, honest_snapshot, obs_actions, choose_outcome_key, vocab
):
    """Members of a merged class have byte-identical S_ctx action tokens
    except the hand_slot column; rl2 features read the same on the belief
    view as on the true state; every visible name is in the vocabulary."""
    import sts2_rs

    bv = cs.belief_view(0)
    snap = honest_snapshot(bv, "monster")
    acts = obs_actions(legal, cs.pending_choice(), bv.pending_choice())
    rf = sts2_rs.rollout_features(cs, version=1)
    rfb = sts2_rs.rollout_features(bv, version=1)
    same = json.dumps(rf, sort_keys=True, default=str) == json.dumps(
        rfb, sort_keys=True, default=str
    )
    stats["rl2_same"] += int(same)
    stats["rl2_checked"] += 1
    tk = T.tokens(snap, acts, rf, "S_ctx")
    n = tk["n_act"]
    cat, cont = tk["cat"][-n:], tk["cont"][-n:]
    hand = sts2_rs.state_to_dict(bv)["deck"].get("hand", [])
    potions = snap["potions"]

    def ck(i: int) -> str:
        c = bv.clone()
        c.apply(acts[i])
        return choose_outcome_key(c)

    piles = sts2_rs.state_to_dict(bv)["deck"]
    keys = C.combat_class_keys(
        acts, hand, potions, choice=bv.pending_choice(), outcome=ck, piles=piles
    )
    cls, reps = C.merge_classes(keys)
    hs = C.SCTX_SLOTS.index("hand_slot")
    for i in range(len(acts)):
        r = int(reps[cls[i]])
        a, b = cat[i].copy(), cat[r].copy()
        a[hs] = b[hs] = 0
        assert (a == b).all() and (cont[i] == cont[r]).all(), (
            acts[i],
            acts[r],
            bv.pending_choice(),
        )
    stats["merged_members"] += len(acts) - len(reps)
    # Exact combat row size: head (RUNCTX, 5 SUMMARY, 1-2 BOSS) + S_ctx.
    sec = tk["sec"]
    n_tok = 8 + len(sec)
    stats["max_combat_tokens"] = max(stats["max_combat_tokens"], n_tok)
    stats["sum_combat_tokens"] += n_tok
    head = C.compact_token_bytes(C.SEC["RUNCTX"]) + 5 * C.compact_token_bytes(C.SEC["SUM_DECK"])
    head += 2 * C.compact_token_bytes(C.SEC["BOSS"])
    stats["sum_combat_compact_bytes"] += head + sum(C.compact_token_bytes(int(s)) for s in sec)
    stats["max_combat_classes"] = max(stats["max_combat_classes"], len(reps))
    for e in snap["enemies"]:
        assert vocab.has("monster", e["kind"]), e["kind"]
        for p in e["powers"]:
            assert vocab.has("power", p["name"]), p["name"]
    for p in snap["powers"]:
        assert vocab.has("power", p["name"]), p["name"]
    for c in hand:
        assert vocab.has("card", c["id"]), c["id"]
    return acts, keys, bv.pending_choice(), int(snap["round"])


@needs_rs
@pytest.mark.pending_e2e_tables
def test_live_walk_maps_every_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """>= 5 full v0.111.0 A10 runs (invincible random policy) through
    PpoRunEnv(combat='yield'): every decision has a live decision type, every
    option a live action kind, merged options stay within MAX_OPTIONS, every
    visible name is in the vocabulary, combat members of one class have
    identical S_ctx action tokens up to hand_slot, and every event option's
    honest preview survives reseeding the hidden run RNG."""
    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")
    monkeypatch.delenv("STS2_RUNAI_POTION_OC", raising=False)
    stats: Counter = Counter()
    outcomes = [_walk(s, stats, check_tokens=True) for s in LIVE_SEEDS]
    assert len(outcomes) >= 5
    seen = {k for k in stats if k in C.DT}
    assert seen >= EXPECT_SEEN, sorted(EXPECT_SEEN - seen)
    assert stats["second_boss_seen"] > 0, "the A10 act-3 second boss is visible on the map"
    assert stats["rl2_checked"] > 0
    assert stats["grant_optional"] > 0, "Sea Glass: optional pick with a confirm option"
    assert stats["decisions_validated"] > 4000, "every stored decision passes validate_decision"
    assert stats["ev_reseed_pairs"] > 100, "event previews checked against reseeded clones"
    print(json.dumps({"outcomes": outcomes, **stats}, sort_keys=True))


# ─── SCHEMA.md stays in step ─────────────────────────────────────────────────

SCHEMA_MD = (
    Path(__file__).resolve().parents[3] / "docs/experiments/2026-11-03-ironclad-e2e-v111/SCHEMA.md"
)


@pytest.mark.skipif(not SCHEMA_MD.exists(), reason="docs/ not in this checkout")
def test_schema_md_matches_contracts() -> None:
    text = SCHEMA_MD.read_text(encoding="utf-8")
    assert C.schema_sha256() in text, "SCHEMA.md quotes a stale schema_sha256"
    assert V.load().sha256 in text, "SCHEMA.md quotes a stale vocabulary sha256"
    names = [s for i, s in enumerate(C.SECTIONS) if i not in C.RESERVED_SECTIONS]
    names += [k.name for k in C.ACTION_KINDS if k.status != "spare"]
    names += [d.name for d in C.DECISION_TYPES if d.status != "spare" and d.id]
    names += [c.name for c in C.X_CAT] + [c.name for c in C.X_CONT]
    names += [f.name for f in C.ROW_FIELDS]
    names += [f"`{f.name}`" for f in dataclasses.fields(C.EnvDecision)]
    names += [f.name for f in dataclasses.fields(C.Selection)]
    names += [f"`{n}`" for n in (*C.POLICY_CALLS, *C.PRIV_CALLS, *C.STAMP_KEYS)]
    names += [f"`{n}`" for n in (*C.EVENT_DELTA_FIELDS, *C.OBS_HIDDEN_FIELDS)]
    names += ["honest_obs(", "honest_event_deltas", "event_delta_columns(", "take_rows("]
    names += ["concat_row_batches(", "check_stamps(", "teacher_stamp(", "runs=True"]
    missing = [n for n in names if n not in text]
    assert not missing, missing


def test_a_choice_option_keys_on_its_pile() -> None:
    """The tokenizer encodes a choose option's pile (`choice_pile`), so the same card in
    hand and in the draw pile must not merge (#1097: eval seed 160001757, a card_select
    offering a hand and a draw copy of TEAR_ASUNDER)."""
    from ppo.e2e import contracts as CC

    card = {"id": "TEAR_ASUNDER", "kind": "card", "upgraded": False, "base_damage": 5, "cost": 2}
    piles = {"hand": [card], "draw": [dict(card)]}
    choice = {
        "kind": "add_from_draw_to_hand",
        "cards": [
            {"id": "TEAR_ASUNDER", "index": 0, "pile": "hand", "upgraded": False},
            {"id": "TEAR_ASUNDER", "index": 0, "pile": "draw", "upgraded": False},
            {"id": "TEAR_ASUNDER", "index": 0, "pile": "draw", "upgraded": False},
        ],
    }
    keys = [CC.choice_option_key(choice, s, piles) for s in range(3)]
    assert keys[0] != keys[1] and keys[1] == keys[2]
