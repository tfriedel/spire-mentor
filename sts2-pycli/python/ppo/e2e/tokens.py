"""The e2e tokenizer: one SCHEMA.md row per decision, combat and out of combat (WP3b).

`encode(d)` turns one stored decision (`ppo.e2e.env.E2EDecision`, or any
object with its fields: `decision_from_state` builds one from a recorded
state) into a `Tok`: the token arrays of one row of the row batch
(`contracts.ROW_FIELDS`), in the layout `docs/experiments/
2026-11-03-ironclad-e2e-v111/SCHEMA.md` fixes:

* token 0 `RUNCTX` (the state token), tokens 1-5 the summaries (draw,
  discard, exhaust, deck, relics; each owns pool entries), then one or two
  `BOSS` tokens (`full_map()["boss"]`, `["second_boss"]`);
* a combat row continues with #1081's S_ctx block, bit for bit: the tokens
  `combat_ctx.tokens.tokens(snapshot, obs_actions, rollout_features(bv),
  "S_ctx")` builds, its 15 categorical slots passed through the combat
  teacher's `RareMap` exactly as `combat_ppo.encode.Encoder.encode` does, its
  122 continuous columns saturated to float16 (`teacher_slice` is byte-equal
  to `Encoder.encode`'s arrays, X-C). The tokenizer only appends trailing
  columns: content ids (`X_ID_*`, from the slots BEFORE the RareMap, so the
  student sees every id), `X_KIND`, `X_POS`, the tactical columns
  (`E_MAXDMG`, `E_KILLABLE`, `S_HP_LOSS_END`), `K_PICKED`, and the WP-ENG-1
  telegraph and forecast once the wheel has the getters. Every legal action
  keeps its token; members of a merged class after the first carry
  `TF_HIDDEN`;
* an out-of-combat (OOC) row continues with the run body (`DECK`, `RELIC`,
  `POTION`, `OFFER`, `MAPNODE` on map decisions) and one `OPT_*` token per
  merged class.

PRIV tokens (value only) go in the separate `pcat`/`pcont`/`psec` arrays:
the true top-10 draw (`PRIV_DRAW`), the true next moves (`PRIV_ENEMY`, once
the wheel has `peek_intents`) and this act's room queues and the later
bosses (`PRIV_RUN`, from the `PrivSource`). The policy part reads only the
belief view, the legal actions in its index space, and the run through the
env's `contracts.POLICY_CALLS` facade; it never reads a PRIV source.

Two entry points:

* `Tokenizer`, the `contracts.Tokenizer` the env and the actor drive with
  `contracts.EnvDecision`s (SCHEMA.md §8.2): `reset_fight`, `observe`,
  `encode`, `encode_priv`. It keeps the fight's `IntentLog` itself.
* `encode(d)` for an `env.E2EDecision`-like object that carries its own
  combat view (`snap` with `ehist`, `d_bv`) and `rm`; `decision_from_state`
  builds one from a recorded state (`RecordedDecision`).

The policy part reads the run as `contracts.POLICY_CALLS` (schema revision 2)
allows: `honest_obs(obs)` out of combat (no run seed; event previews through
`honest_event_options`, so the O_EV_* columns are `event_delta_columns`),
else the run facade (`rm.master_deck_enchantments`, `rm.ascension_level`,
`rm.pending_enchantment`, `rm.act_floor` included), and in combat the belief
view, whose `encounter_id` (WP-ENG-1) fills RUNCTX's `X_ID_AUX`.

Caching (X-T): the run-static reads (map, bosses, PRIV room counts) once per
(run, act, floor), and one-entry memos for the parts that are pure functions
of an input the next decision usually repeats: the deck summary, the relic
counts and the PRIV_DRAW rows (a draw reuses the rows of the cards still in
the window). Outputs never alias a cached array; `memo_state` and
`memo_restore` let a timing harness replay a call with the memos as the run
left them (`ppo.e2e.tokcheck`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ppo.combat_ctx import tokens as T

from . import contracts as CT
from . import tactical as TC
from . import vocab as V

N_CAT, N_CONT = CT.N_CAT, CT.N_CONT
NS_CAT, NS_CONT = CT.SCTX_N_CAT, CT.SCTX_N_CONT
XC, XF, SEC, AK, DT = CT.XC, CT.XF, CT.SEC, CT.AK, CT.DT
_ABSENT_CAT = np.asarray(list(CT.SCTX_PAD) + [0] * (N_CAT - NS_CAT), np.int16)
_ROOM_OF_NODE = {
    "Monster": "monster",
    "Elite": "elite",
    "Boss": "boss",
    "Shop": "shop",
    "RestSite": "rest_site",
    "Treasure": "treasure",
    "Ancient": "ancient",
    "Unknown": "unknown",
    "Event": "event",
}
_CARDLIKE = ("HAND", "DRAW_UNK", "DRAW_TOP", "DRAW_BOT", "DISCARD", "EXHAUST", "PLAY", "ACT_PLAY")


# ── the row ──────────────────────────────────────────────────────────────


def _u16(x: Any) -> np.ndarray:
    return np.asarray(x, np.uint16)


@dataclass
class Tok:
    """One decision's slice of the row batch: everything but the actor's fields."""

    is_combat: bool
    dtype: int
    cat: np.ndarray  # int16 [T, 32]
    cont: np.ndarray  # float16 [T, 210]
    sec: np.ndarray  # int8 [T]
    tok_flags: np.ndarray  # uint8 [T]
    mem_tok: np.ndarray  # uint16 [M]
    mem_class: np.ndarray  # uint16 [M]
    cls_tok: np.ndarray  # uint16 [C]
    cls_kind: np.ndarray  # uint8 [C]
    sctx_lo: int = 0
    n_sctx: int = 0
    pool_tok: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))
    pool_vid: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))
    pool_w: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float16))
    pcat: np.ndarray = field(default_factory=lambda: np.zeros((0, N_CAT), np.int16))
    pcont: np.ndarray = field(default_factory=lambda: np.zeros((0, N_CONT), np.float16))
    psec: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    ppool_tok: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))
    ppool_vid: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint16))
    ppool_w: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float16))

    @property
    def n_tok(self) -> int:
        return len(self.sec)

    @property
    def n_classes(self) -> int:
        return len(self.cls_tok)

    @property
    def n_priv(self) -> int:
        return len(self.psec)

    def policy_arrays(self) -> tuple[np.ndarray, ...]:
        """Everything the policy trunk and heads read (T14 compares these)."""
        return (
            self.cat,
            self.cont,
            self.sec,
            self.tok_flags,
            self.mem_tok,
            self.mem_class,
            self.cls_tok,
            self.cls_kind,
            self.pool_tok,
            self.pool_vid,
            self.pool_w,
            np.asarray([self.sctx_lo, self.n_sctx, self.dtype], np.int64),
        )

    def teacher(self) -> dict[str, Any]:
        """The combat teacher's view (`contracts.teacher_slice` of this row)."""
        lo, hi = self.sctx_lo, self.sctx_lo + self.n_sctx
        return {
            "cat": self.cat[lo:hi, :NS_CAT],
            "cont": self.cont[lo:hi, :NS_CONT],
            "sec": self.sec[lo:hi],
            "n_act": len(self.mem_tok),
        }


ACTOR_DEFAULTS: dict[str, Any] = {
    "reward": 0.0,
    "done": False,
    "bootstrap": False,
    "actor": 0,
    "logp_behav": 0.0,
    "action": 0,
    "iter": 0,
    "v_behav": 0.0,
    "cell": CT.CELL_NONE,
    "t_src": 0,
}


def row_batch(
    toks: list[Tok],
    *,
    run_uid: list[int] | None = None,
    t: list[int] | None = None,
    tprob: list[np.ndarray] | None = None,
    **row_fields: Any,
) -> dict[str, np.ndarray]:
    """A `contracts` row batch of `toks`. Per-row fields default to
    `ACTOR_DEFAULTS` (a scalar or a per-row list overrides), `tprob` to NaN."""
    n = len(toks)
    F = CT.FIELDS

    def col(name: str, vals: Any) -> np.ndarray:
        return np.asarray(vals, np.dtype(F[name].dtype))

    def per_row(name: str, default: Any) -> np.ndarray:
        v = row_fields.get(name, default)
        return col(name, v if isinstance(v, (list, tuple, np.ndarray)) else [v] * n)

    def cat0(name: str, xs: list[np.ndarray]) -> np.ndarray:
        f = F[name]
        if not xs:
            return np.zeros((0, f.width) if f.width else (0,), np.dtype(f.dtype))
        return np.concatenate(xs).astype(np.dtype(f.dtype), copy=False)

    b = {
        "run_uid": col("run_uid", run_uid if run_uid is not None else [0] * n),
        "t": col("t", t if t is not None else list(range(n))),
        "is_combat": col("is_combat", [k.is_combat for k in toks]),
        "dtype": col("dtype", [k.dtype for k in toks]),
        "n_classes": col("n_classes", [k.n_classes for k in toks]),
        "n_tok": col("n_tok", [k.n_tok for k in toks]),
        "n_mem": col("n_mem", [len(k.mem_tok) for k in toks]),
        "n_priv": col("n_priv", [k.n_priv for k in toks]),
        "n_pool": col("n_pool", [len(k.pool_tok) for k in toks]),
        "n_ppool": col("n_ppool", [len(k.ppool_tok) for k in toks]),
        "sctx_lo": col("sctx_lo", [k.sctx_lo for k in toks]),
        "n_sctx": col("n_sctx", [k.n_sctx for k in toks]),
    }
    for name, default in ACTOR_DEFAULTS.items():
        b[name] = per_row(name, default)
    for name in (
        "cat",
        "cont",
        "sec",
        "tok_flags",
        "mem_tok",
        "mem_class",
        "cls_tok",
        "cls_kind",
        "pcat",
        "pcont",
        "psec",
        "pool_tok",
        "pool_vid",
        "pool_w",
        "ppool_tok",
        "ppool_vid",
        "ppool_w",
    ):
        b[name] = cat0(name, [getattr(k, name) for k in toks])
    b["tprob"] = cat0(
        "tprob",
        tprob if tprob is not None else [np.full(k.n_classes, np.nan, np.float32) for k in toks],
    )
    return b


# ── lookup tables ────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _luts(game_version: str) -> dict[str, np.ndarray]:
    """S_ctx table row -> content id, per content slot (PAD -> NONE, UNK -> UNK_*)."""
    del game_version  # the cache key: S_ctx's tables follow the selected build
    tv, v = T.vocab(), V.load()
    out = {}
    for slot, reg in (
        ("card", "card"),
        ("relic", "relic"),
        ("potion", "potion"),
        ("power", "power"),
        ("monster", "monster"),
        ("enchantment", "enchantment"),
    ):
        tab = tv.tables[T.S[slot]]
        lut = np.zeros(tab.size, np.int16)
        for i, name in enumerate(tab.names):
            lut[i] = v.id(reg, name)
        lut[tab.unk] = V.UNK_ID[reg]
        lut[tab.pad] = V.NONE_ID
        out[slot] = lut
    return out


def luts() -> dict[str, np.ndarray]:
    import sts2_client

    return _luts(sts2_client.GAME_VERSION)


_ID_SLOTS = ("card", "relic", "potion", "power", "monster", "enchantment")


@lru_cache(maxsize=4)
def _id_tables(game_version: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The S_ctx content ids per section in one gather: for each section and
    role (MAIN, MOD, REL, and MAIN's EPOWER card fallback), the S_ctx slot
    column it reads (`col`, `[64, 4]`) and that slot's offset (`base`) into
    `joined`, every slot's `luts()` table after a zero block (a role a
    section lacks reads the zero block)."""
    L = _luts(game_version)
    zero = max(len(L[x]) for x in _ID_SLOTS)
    off, parts, at = {}, [np.zeros(zero, np.int16)], zero
    for x in _ID_SLOTS:
        off[x] = at
        parts.append(L[x])
        at += len(L[x])
    col = np.full((CT.N_SECTION, 4), T.S["card"], np.int64)
    base = np.zeros((CT.N_SECTION, 4), np.int64)

    def put(sec: str, role: int, slot: str) -> None:
        col[SEC[sec], role] = T.S[slot]
        base[SEC[sec], role] = off[slot]

    for sec in _CARDLIKE:
        put(sec, 0, "card")
        put(sec, 1, "enchantment")
    put("RELIC", 0, "relic")
    put("POTION", 0, "potion")
    put("ACT_POTION", 0, "potion")
    put("PPOWER", 0, "power")
    put("EPOWER", 0, "power")
    put("EPOWER", 2, "monster")
    put("EPOWER", 3, "card")
    put("ENEMY", 0, "monster")
    put("EHIST", 2, "monster")
    put("ACT_PLAY", 2, "monster")
    return col, base, np.concatenate(parts)


def id_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import sts2_client

    return _id_tables(sts2_client.GAME_VERSION)


@lru_cache(maxsize=8192)
def _vid(registry: str, name: Any) -> int:
    return V.load().id(registry, None if name is None else str(name))


@lru_cache(maxsize=4096)
def _tslot(slot: str, name: str) -> int:
    return T.vocab().tables[T.S[slot]].index(name)[0]


def _f(x: Any) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _at(xs: Any, i: Any) -> Any:
    try:
        i = int(i)
    except (TypeError, ValueError):
        return None
    if xs is None or not 0 <= i < len(xs):
        return None
    return xs[i]


def _ench_name(e: Any) -> str | None:
    if not e:
        return None
    if isinstance(e, (list, tuple)):
        return str(e[0]) if e else None
    if isinstance(e, dict):
        return e.get("id") or e.get("name")
    return str(e)


def _pos(i: int) -> int:
    return min(int(i) + 1, 63)


# ── token buffer ─────────────────────────────────────────────────────────


# One-entry memos (X-T "SUMMARY caching"), one set per run: a part of the row that
# is a pure function of its key is reused while the key equals the same run's
# previous call's. Consecutive decisions of a fight share the master deck and the
# relics on almost every step, and the true top-10 draw within a turn. A worker
# encodes its env slots round-robin, so the memos are keyed by (run, slot): each run
# hits its own entries as if it ran alone (GH #1097 TOK-1). The value depends only
# on the content key, so a run key only decides who hits. Values are never exposed:
# callers copy them into the row. `clear_cache` drops a run's memos with its
# run-static entry; `_MEMO_MAX` bounds the rest.
_MEMO: dict[tuple[Any, str], tuple[Any, Any]] = {}
_MEMO_MAX = 4096


def _memo(slot: str, key: Any, fn: Any, run: Any = None) -> Any:
    hit = _MEMO.get((run, slot))
    if hit is not None and hit[0] == key:
        return hit[1]
    v = fn()
    _memo_put(run, slot, (key, v))
    return v


def _memo_put(run: Any, slot: str, entry: tuple[Any, Any]) -> None:
    if len(_MEMO) >= _MEMO_MAX and (run, slot) not in _MEMO:
        _MEMO.clear()
    _MEMO[run, slot] = entry


def _memo_key(d: Any) -> Any:
    """The memos' run key of a decision: `_Run.key` without building a `_Run`."""
    rk = getattr(d, "run_key", None)
    if rk is not None:
        return rk
    rm = getattr(d, "rm", None)
    return None if rm is None else id(rm)


def memo_state() -> dict[tuple[Any, str], tuple[Any, Any]]:
    """The memos, for a timing harness to replay a call as it ran."""
    return dict(_MEMO)


def memo_restore(state: dict[tuple[Any, str], tuple[Any, Any]]) -> None:
    _MEMO.clear()
    _MEMO.update(state)


_ABSENT_ROWS: dict[int, np.ndarray] = {}


def _absent_rows(n: int) -> np.ndarray:
    """`n` absent cat rows (a copy of a cached tile)."""
    t = _ABSENT_ROWS.get(n)
    if t is None:
        t = _ABSENT_ROWS[n] = np.tile(_ABSENT_CAT, (n, 1))
    return t.copy()


class _Buf:
    """Growable token rows for the head and the OOC body.

    Rows live in preallocated arrays (X-T: no per-token allocation, no
    stacking); `new` returns views of the next row, so a caller finishes a
    row before it asks for the next one (a growth step copies the rows)."""

    def __init__(self, cap: int = 16) -> None:
        self._cat = _absent_rows(cap)
        self._cont = np.zeros((cap, N_CONT), np.float32)
        self._sec = np.zeros(cap, np.int8)
        self._flags = np.zeros(cap, np.uint8)
        self.n = 0
        self.pool_tok: list[int] = []
        self.pool_vid: list[int] = []
        self.pool_w: list[float] = []

    def __len__(self) -> int:
        return self.n

    def new(self, sec: str, flags: int = 0) -> tuple[np.ndarray, np.ndarray]:
        i = self.n
        if i == len(self._sec):
            self._grow()
        self.n = i + 1
        self._sec[i] = SEC[sec]
        self._flags[i] = flags
        return self._cat[i], self._cont[i]

    def _grow(self) -> None:
        n, cap = self.n, 2 * len(self._sec)
        cat = _absent_rows(cap)
        cat[:n] = self._cat
        cont = np.zeros((cap, N_CONT), np.float32)
        cont[:n] = self._cont
        sec = np.zeros(cap, np.int8)
        sec[:n] = self._sec
        flags = np.zeros(cap, np.uint8)
        flags[:n] = self._flags
        self._cat, self._cont, self._sec, self._flags = cat, cont, sec, flags

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        return self._cat[:n], self._cont[:n], self._sec[:n], self._flags[:n]


# ── run view: what the policy may read ───────────────────────────────────


class _Run:
    """The run reads of one decision: `honest_obs(run_obs)` first (out of
    combat), then the policy facade (`contracts.POLICY_CALLS`'s `rm.*`): the
    env's `PolicySource.run()`, or a legacy decision's `rm`."""

    def __init__(self, d: Any) -> None:
        self.d = d
        pol = getattr(d, "policy", None)
        if pol is None:
            self.f = d.rm
        else:
            self.f = (pol() if callable(pol) else pol).run()
        obs = getattr(d, "run_obs", None)
        # The policy reads `obs` only as `honest_obs(obs)`; an `_EnvView`'s is already.
        self.obs = {} if not obs else obs if isinstance(d, _EnvView) else CT.honest_obs(obs)
        self._relics: list[Any] | None = None
        self._deck: list[Any] | None = None
        self._act: int | None = None
        self._floor: int | None = None
        rk = getattr(d, "run_key", None)
        # The run-static cache key: the run uid on the env path, else the run
        # object (checked with `is`).
        self.key, self.anchor = (rk, rk) if rk is not None else (id(d.rm), d.rm)

    def act(self) -> int:
        if self._act is None:
            self._act = int(self.f.act_index)
        return self._act

    def floor(self) -> int:
        if self._floor is None:
            self._floor = int(self.f.total_floor())
        return self._floor

    def read(self, name: str) -> Any:
        """A facade read (`rm.<name>`, called if it is a method); None when the
        facade lacks it."""
        v = getattr(self.f, name, None)
        return v() if callable(v) else v

    def deck(self) -> list[Any]:
        if self._deck is None:
            self._deck = self.obs["deck"] if "deck" in self.obs else self.f.master_deck_detailed()
        return self._deck

    def deck_ench(self) -> list[Any]:
        if "deck_enchantments" in self.obs:
            return self.obs["deck_enchantments"]
        return self.read("master_deck_enchantments") or []

    def relics(self) -> list[Any]:
        if "relics" in self.obs:
            return self.obs["relics"]
        if self._relics is None:
            self._relics = self.f.relics()
        return self._relics

    def potions(self) -> list[Any]:
        return self.obs["potions"] if "potions" in self.obs else self.f.potions()

    def full_map(self) -> dict[str, Any]:
        return self.obs["map"] if "map" in self.obs else self.f.full_map()

    def act_floor(self, fm: dict[str, Any]) -> int:
        """`rm.act_floor()`; without the read, the map row + 1 (0 before the
        first node), which equals it on every one of 916 recorded states."""
        if "act_floor" in self.obs:
            return int(self.obs["act_floor"])
        a = self.read("act_floor")
        if a is not None:
            return int(a)
        cc = fm.get("current_coord")
        return int(cc["row"]) + 1 if cc else 0

    def ascension(self) -> int:
        if "ascension" in self.obs:
            return int(self.obs["ascension"])
        a = self.read("ascension_level")
        if a is not None:
            return int(a)
        from . import factory

        return int(factory.ASCENSION)  # the campaign's construction (plan §3.1)


# ── head: RUNCTX, SUMMARY, BOSS ──────────────────────────────────────────


def _bosses(fm: dict[str, Any]) -> list[str]:
    return [str(b["id"]) for b in (fm.get("boss"), fm.get("second_boss")) if b and b.get("id")]


def bosses_left(act: int, n_this_act: int, ascension: int) -> int:
    """Boss fights left in the run, this act's included: the visible run
    structure (three acts; from A10 act 3 ends in two bosses). `n_this_act`
    counts this act's bosses not yet beaten (`bosses_beaten`)."""
    later = sum(2 if a == 3 and ascension >= 10 else 1 for a in range(act + 1, 4))
    return n_this_act + later


def bosses_beaten(fm: dict[str, Any], in_combat: bool) -> int:
    """This act's bosses already beaten (review finding TOK-3): those on map rows
    below the current one (boss 1 during the A10 act-3 boss 2), and the current row's
    boss once its fight is over (its card reward and boss relic). A boss room's
    decisions before its fight's end are combat decisions."""
    cur = _cur_row(fm)
    rows = [
        int(b["row"])
        for b in (fm.get("boss"), fm.get("second_boss"))
        if b and b.get("id") and b.get("row") is not None
    ]
    return sum(1 for r in rows if r < cur or (r == cur and not in_combat))


def _cur_row(fm: dict[str, Any]) -> int:
    cc = fm.get("current_coord")
    return int(cc["row"]) if cc else -1


def _boss_row(fm: dict[str, Any]) -> int:
    b = fm.get("boss") or {}
    return int(b.get("row", len(fm.get("rows") or [])))


def _sel_spec(d: Any) -> tuple[float, ...]:
    """`(min, max, picked, can_skip)` of the open selection, raw counts: the
    env's `Selection` (`d.sel`, SCHEMA.md §2.4; zeros on every type but
    COMBAT_CHOICE and CARD_GRANT)."""
    s = getattr(d, "sel", None) or CT.NO_SELECTION
    return float(s.min), float(s.max), float(s.picked), float(s.can_skip)


def _encounter_id(d: Any) -> Any:
    """This fight's encounter id: `bv.encounter_id` (WP-ENG-1, `POLICY_CALLS`),
    None before the wheel has the getter."""
    bv = getattr(d, "bv", None)
    enc = getattr(bv, "encounter_id", None) if bv is not None else None
    return enc() if callable(enc) else enc


def _event_name(d: Any, run: _Run) -> Any:
    """`rm.current_event_name()` on EVENT and NEOW: the env's `event_name`."""
    ev = getattr(d, "event_name", None)
    if ev:
        return ev
    return run.obs.get("event_name") or run.f.current_event_name()


def _runctx(
    buf: _Buf,
    d: Any,
    run: _Run,
    fm: dict[str, Any],
    n_bosses: int,
    deck: list[Any],
    sel: tuple[float, ...],
    room: str | None,
) -> None:
    cat, cont = buf.new("RUNCTX")
    cat[XC["X_DTYPE"]] = d.dtype
    if d.dtype in (DT["EVENT"], DT["NEOW"]):
        cat[XC["X_ID_MAIN"]] = _vid("event", _event_name(d, run))
    if d.is_combat:
        enc = _encounter_id(d)
        cat[XC["X_ID_AUX"]] = _vid("encounter", enc) if enc else 0
    if room is None:
        room = (run.f.current_room() or {}).get("room_type")
    cat[XC["X_ROOM"]] = CT.ROOM_ID.get(str(room).lower() if room else "none", CT.ROOM_ID["unknown"])
    act = run.act()
    hp, mhp = _f(run.f.current_hp), max(1.0, _f(run.f.max_hp))
    asc = run.ascension()
    cont[XF["R_ACT"]] = act / 3.0
    cont[XF["R_FLOOR"]] = run.floor() / 50.0
    cont[XF["R_ACT_FLOOR"]] = run.act_floor(fm) / 17.0
    cont[XF["R_ROWS_TO_BOSS"]] = max(0, _boss_row(fm) - _cur_row(fm)) / 17.0
    left = n_bosses - bosses_beaten(fm, bool(d.is_combat))
    cont[XF["R_BOSSES_LEFT"]] = bosses_left(act, left, asc) / 4.0
    cont[XF["R_HP"]] = hp / 100.0
    cont[XF["R_MAXHP"]] = mhp / 100.0
    cont[XF["R_HP_FRAC"]] = hp / mhp
    cont[XF["R_GOLD"]] = _f(run.f.gold) / 300.0
    cont[XF["R_DECK"]] = len(deck) / 40.0
    cont[XF["R_DECK_UPG"]] = sum(1 for _c, up in deck if up) / 20.0
    cont[XF["R_RELICS"]] = len(run.relics()) / 20.0
    cont[XF["R_POTIONS"]] = sum(1 for p in run.potions() if p) / 5.0
    cont[XF["R_POT_SLOTS"]] = _f(run.f.max_potion_slots()) / 5.0
    cont[XF["R_ASC"]] = asc / 10.0
    cont[XF["R_IN_COMBAT"]] = float(d.is_combat)
    cont[XF["R_SEL_MIN"]] = sel[0] / 5.0
    cont[XF["R_SEL_MAX"]] = sel[1] / 5.0
    cont[XF["R_SEL_PICKED"]] = sel[2] / 5.0
    cont[XF["R_SEL_CAN_SKIP"]] = sel[3]
    cont[XF["R_N_CLASSES"]] = len(d.members) / 64.0
    cont[XF["R_DOUBLE_BOSS"]] = float(n_bosses > 1)


def _summary(buf: _Buf, sec: str, members: dict[int, float], n_upg: int, n_ench: int) -> None:
    """A pooled summary token: one pool entry per distinct id (weight = count)."""
    pos = len(buf)
    _cat, cont = buf.new(sec, CT.TF_POOLED)
    n = sum(members.values())
    cont[XF["U_N"]] = n / 40.0
    cont[XF["U_UPG"]] = n_upg / 20.0
    cont[XF["U_ENCH"]] = n_ench / 10.0
    cont[XF["U_DISTINCT"]] = len(members) / 40.0
    if 0 in members:
        members = {v: w for v, w in members.items() if v}
    buf.pool_tok.extend([pos] * len(members))
    buf.pool_vid.extend(members.keys())
    buf.pool_w.extend(members.values())


def _relic_counts(relics: list[Any]) -> dict[int, float]:
    rel: dict[int, float] = {}
    for r in relics:
        v = _vid("relic", r)
        rel[v] = rel.get(v, 0.0) + 1.0
    return rel


def _deck_summary(deck: list[Any], ench: list[Any]) -> tuple[dict[int, float], int, int]:
    c: dict[int, float] = {}
    n_upg = 0
    for cid, up in deck:
        v = _card_vid(cid)
        c[v] = c.get(v, 0.0) + 1.0
        if up:
            n_upg += 1
    n_ench = sum(1 for e in ench[: len(deck)] if e)
    return c, n_upg, n_ench


def _hashable(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(y) for y in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    return x


@lru_cache(maxsize=4096)
def _card_vid(cid: Any) -> int:
    return V.load().id("card", None if cid is None else str(cid))


def _pile_summary(
    counted: list[tuple[dict[str, Any], Any]], *singles: list[dict[str, Any]]
) -> tuple[dict[int, float], int, int]:
    """Per-id counts of a snapshot pile: `(card, count)` pairs, then single
    cards (the draw pile's known top and bottom)."""
    c: dict[int, float] = {}
    n_upg = n_ench = 0
    for card, n in counted:
        n = float(n)
        v = _card_vid(card.get("id"))
        c[v] = c.get(v, 0.0) + n
        if card.get("upgraded"):
            n_upg += int(n)
        if card.get("enchantment"):
            n_ench += int(n)
    for cards in singles:
        for card in cards:
            v = _card_vid(card.get("id"))
            c[v] = c.get(v, 0.0) + 1.0
            if card.get("upgraded"):
                n_upg += 1
            if card.get("enchantment"):
                n_ench += 1
    return c, n_upg, n_ench


# Run-static reads, cached per (run, act, floor): the map, the bosses and the
# PRIV room counts change only when the run moves to another room (X-T
# "tokenizer caching"). Keyed by the run object's identity (checked with `is`).
_STATIC: dict[int, tuple[Any, tuple[int, int], dict[str, Any]]] = {}
_STATIC_MAX = 512
# Run uids one `Tokenizer` tracks for restart detection, least recent dropped
# first (with its run-static entry), so a long-lived worker stays bounded.
_T_MAX = 4096


def _static_hit(run: _Run, key: tuple[int, int]) -> dict[str, Any] | None:
    hit = _STATIC.get(run.key)
    if hit is None or hit[1] != key:
        return None
    if hit[0] is run.anchor or (isinstance(run.anchor, tuple) and hit[0] == run.anchor):
        return hit[2]
    return None


def _static(run: _Run) -> dict[str, Any]:
    key = (run.act(), run.floor())
    st = _static_hit(run, key)
    if st is not None:
        return st
    fm = run.full_map()
    st = {"fm": fm, "bosses": _bosses(fm), "key": key}
    if len(_STATIC) >= _STATIC_MAX:
        _STATIC.clear()
    _STATIC[run.key] = (run.anchor, key, st)
    return st


def clear_cache(key: Any = None) -> None:
    """Drop the run-static cache and the memos: all of them, or one run's (`_Run.key`)."""
    if key is None:
        _STATIC.clear()
        _MEMO.clear()
    else:
        _STATIC.pop(key, None)
        for k in [k for k in _MEMO if k[0] == key]:
            del _MEMO[k]


def _head(
    d: Any, run: _Run, snap: dict[str, Any] | None, room: str | None
) -> tuple[_Buf, list[Any], list[Any]]:
    # A combat head is at most 8 tokens; an OOC row grows from 64.
    buf = _Buf(8 if d.is_combat else 64)
    stc = _static(run)
    fm = stc["fm"]
    bosses = stc["bosses"]
    deck = run.deck()
    ench = run.deck_ench() or []
    _runctx(buf, d, run, fm, len(bosses), deck, _sel_spec(d), room)
    if snap is not None:
        draw = _pile_summary(snap["draw_unknown"], snap["draw_top"], snap["draw_bottom"])
        _summary(buf, "SUM_DRAW", *draw)
        _summary(buf, "SUM_DISCARD", *_pile_summary(snap["discard"]))
        _summary(buf, "SUM_EXHAUST", *_pile_summary(snap["exhaust"]))
    else:
        for sec in ("SUM_DRAW", "SUM_DISCARD", "SUM_EXHAUST"):
            _summary(buf, sec, Counter(), 0, 0)
    deck_key = (list(deck), list(ench))
    _summary(buf, "SUM_DECK", *_memo("deck", deck_key, lambda: _deck_summary(deck, ench), run.key))
    relics = run.relics()
    rel = _memo("relics", list(relics), lambda: _relic_counts(relics), run.key)
    _summary(buf, "SUM_RELICS", rel, 0, 0)
    act = run.act()
    for k, b in enumerate(bosses[: CT.MAX_BOSS_TOKENS]):
        cat, cont = buf.new("BOSS")
        cat[XC["X_ID_MAIN"]] = _vid("encounter", b)
        cont[XF["B_ORDER"]] = float(k)
        cont[XF["B_ACT"]] = act / 3.0
    return buf, deck, ench


# ── combat ───────────────────────────────────────────────────────────────

_HAND_SLOT = CT.SCTX_SLOTS.index("hand_slot")


def combat_picks(snap: dict[str, Any], acts: list[dict[str, Any]]) -> list[int]:
    """Option indices already picked in the open choice (SCHEMA.md §2.4).

    The engine re-opens a multi-pick choice after each pick and offers one
    `choose {selection: [i]}` per still-unpicked option, so an option no
    choose action offers is picked: `env.combat_selection`'s `picked` counts
    the same options.
    """
    ch = snap.get("choice")
    if ch is None:
        return []
    offered = {
        int(a["selection"][0]) for a in acts if a.get("action") == "choose" and a.get("selection")
    }
    return [i for i in range(len(ch["options"])) if i not in offered]


def _intent_cols(cont: np.ndarray, cat: np.ndarray, j: int, tele: Any, fc: Any) -> None:
    """WP-ENG-1 telegraph and forecast columns of enemy `j` (policy-visible
    parts only: a forecast's `move_id` is PRIV)."""
    if tele is not None and j < len(tele):
        ints = list(tele[j] or [])
        cont[XF["E_N_INTENTS"]] = len(ints) / 3.0
        for k, (c_kind, c_eff, c_times) in (
            (1, ("X_INTENT_T2", "E_T2_EFF", "E_T2_TIMES")),
            (2, ("X_INTENT_T3", "E_T3_EFF", "E_T3_TIMES")),
        ):
            if k < len(ints):
                i = ints[k]
                cat[XC[c_kind]] = CT.INTENT_IDS.get(_intent_kind(i), 0)
                cont[XF[c_eff]] = _f(i.get("effective_amount", i.get("amount"))) / 30.0
                cont[XF[c_times]] = _f(i.get("times")) / 5.0
    if fc is not None and j < len(fc):
        cert = 1.0
        for k, st in enumerate(list(fc[j] or [])[:3]):
            cert = _f(st.get("certainty"))
            ints = st.get("intents") or []
            head = ints[0] if ints else {}
            cat[XC[f"X_FC{k + 1}"]] = CT.INTENT_IDS.get(_intent_kind(head), 0) if head else 0
            atk = sum(TC._attack_total(i) for i in ints)
            cont[XF[f"E_F{k + 1}_CERT"]] = cert
            cont[XF[f"E_F{k + 1}_AMT"]] = _f(head.get("amount")) / 30.0
            cont[XF[f"E_F{k + 1}_TIMES"]] = _f(head.get("times")) / 5.0
            cont[XF[f"E_F{k + 1}_TOT"]] = atk / 50.0


def _dead(enemies: list[dict[str, Any]], j: int) -> bool:
    """Whether snapshot enemy `j` is dead: `peek_intents` keeps a dead
    enemy's forecast, to be masked with the wire's `is_dead`."""
    return j < len(enemies) and bool(enemies[j].get("is_dead"))


def _intent_kind(i: dict[str, Any]) -> str:
    k = str(i.get("kind", "Unknown"))
    return "Attack" if k == "MultiAttack" else k


def _getter(obj: Any, *names: str) -> Any:
    for n in names:
        if obj is not None and hasattr(obj, n):
            return getattr(obj, n)
    return None


_POTION_KINDS = frozenset((AK["POTION"], AK["POTION_TARGET"]))
_EXTRA_KINDS = _POTION_KINDS | {AK["CHOOSE"]}


def _sctx_trailing(
    cat15: np.ndarray,
    sec: np.ndarray,
    n_act: int,
    d: Any,
    snap: dict[str, Any],
    d_bv: dict[str, Any] | None,
    picked: list[int],
    tele: Any,
    fc: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """The trailing columns of the S_ctx block: `[T, 17]` cat, `[T, 88]` cont."""
    t = len(sec)
    xc = np.zeros((t, N_CAT - NS_CAT), np.int16)
    xf = np.zeros((t, N_CONT - NS_CONT), np.float32)
    o = NS_CAT
    s = sec.astype(np.int64)
    # Content ids, one gather per role (`id_tables`): MAIN falls back to the
    # card on an EPOWER token whose power slot is PAD (an intent's card).
    col, base, joined = id_tables()
    ids = joined[base[s] + cat15[np.arange(t)[:, None], col[s]]]
    main = ids[:, 0]
    xc[:, XC["X_ID_MAIN"] - o] = np.where(main > 0, main, ids[:, 3])
    xc[:, XC["X_ID_MOD"] - o] = ids[:, 1]
    xc[:, XC["X_ID_REL"] - o] = ids[:, 2]
    # Action tokens: kind, potion slot and target, choose content.
    a0 = t - n_act
    kinds = d.kinds
    xc[a0:, XC["X_KIND"] - o] = kinds
    acts = d.obs_actions
    ch = snap.get("choice")
    piles = (d_bv or {}).get("deck") or {}
    pc_cards: list[Any] | None = None
    for k in [k for k, kind in enumerate(kinds) if kind in _EXTRA_KINDS]:
        a = acts[k]
        if kinds[k] in _POTION_KINDS:
            # The slot the class drinks from: the env applies the class's
            # representative, and two equal potions in two slots are one
            # class (SCHEMA.md §5 keys a potion by id and target), so every
            # member carries its representative's slot and members stay
            # token-identical.
            rep = acts[d.members[int(d.classes[k])][0]]
            xc[a0 + k, XC["X_POS"] - o] = _pos(int(rep["slot"]))
            ti = a.get("target_index")
            if ti is not None:
                e = _at(snap["enemies"], ti)
                xc[a0 + k, XC["X_ID_REL"] - o] = _vid("monster", e["kind"]) if e else 0
        elif ch is not None:  # CHOOSE
            s0 = int(a["selection"][0])
            opt = _at(ch["options"], s0)
            if opt is not None:
                if pc_cards is None:
                    pc = d.bv.pending_choice() if d.bv is not None else None
                    pc_cards = (pc or {}).get("cards") or []
                xc[a0 + k, XC["X_ID_MAIN"] - o] = _vid("card", opt["id"])
                xc[a0 + k, XC["X_ID_MOD"] - o] = _choice_ench(pc_cards, s0, piles)
    # Potion tokens: the belt slot (the snapshot drops empty slots).
    pm = np.flatnonzero(s[:a0] == SEC["POTION"])
    if len(pm):
        belt = ((d_bv or {}).get("player") or {}).get("potions") or []
        slots = [i for i, p in enumerate(belt) if p] or list(range(len(pm)))
        for j, ti in enumerate(pm):
            xc[ti, XC["X_POS"] - o] = _pos(slots[j] if j < len(slots) else j)
    # Picks: the hand cards an open multi-pick choice has selected.
    if picked and ch is not None:
        hand_tok = np.flatnonzero(s[:a0] == SEC["HAND"])
        for i in picked:
            hi = _at(ch["options"], i)
            hi = hi.get("hand_index") if hi else None
            if hi is not None and int(hi) < len(hand_tok):
                xf[hand_tok[int(hi)], XF["K_PICKED"] - NS_CONT] = 1.0
    # Tactical columns and the WP-ENG-1 intent columns, per enemy (a dead
    # enemy keeps a forecast in the getter; it gets none here).
    md, kill, loss = TC.tactical(snap, acts)
    em = np.flatnonzero(s[:a0] == SEC["ENEMY"])
    enemies = snap.get("enemies") or []
    for j, ti in enumerate(em):
        xf[ti, XF["E_MAXDMG"] - NS_CONT] = md[j] / 50.0 if j < len(md) else 0.0
        xf[ti, XF["E_KILLABLE"] - NS_CONT] = kill[j] if j < len(kill) else 0.0
        if (tele is not None or fc is not None) and not _dead(enemies, j):
            cat_row = np.zeros(N_CAT, np.int16)
            cont_row = np.zeros(N_CONT, np.float32)
            _intent_cols(cont_row, cat_row, j, tele, fc)
            xc[ti] |= cat_row[NS_CAT:]
            xf[ti] += cont_row[NS_CONT:]
    xf[0, XF["S_HP_LOSS_END"] - NS_CONT] = loss / 50.0
    return xc, xf


def _choice_ench(pc_cards: list[Any], s: int, piles: dict[str, Any]) -> int:
    """The enchantment id of pending-choice option `s` (its pile card's);
    `pc_cards` is `bv.pending_choice()["cards"]`."""
    opt = _at(pc_cards, s)
    if not opt:
        return 0
    pile = piles.get(CT._PILES.get(str(opt.get("pile")), ""))
    card = _at(pile, opt.get("index")) if pile is not None else None
    return _vid("enchantment", _ench_name(card.get("enchantment"))) if card else 0


def _rare_apply(rare: Any, cat15: np.ndarray) -> np.ndarray:
    if rare is None:
        return cat15
    return rare.apply(cat15).astype(np.int16)


def encode_combat(d: Any, *, rare: Any = None, priv: bool = True, check: bool = True) -> Tok:
    """A combat decision's row (`d.is_combat`)."""
    import sts2_rs

    bv, snap = d.bv, d.snap
    acts = d.obs_actions
    run = _Run(d)
    picked = combat_picks(snap, acts)
    buf, _deck, _ench = _head(d, run, snap, d.room)
    hcat, hcont, hsec, hflags = buf.arrays()
    rf = sts2_rs.rollout_features(bv, version=1)
    tk = T.tokens(snap, acts, rf, "S_ctx")
    cat15 = tk["cat"]
    sec = tk["sec"]
    n_act = int(tk["n_act"])
    tele = fc = None
    g = _getter(bv, "telegraph")
    if g is not None:
        tele = g()
    g = _getter(bv, "peek_intents")
    if g is not None:
        fc = g(3)
    n_mem = len(acts)
    if n_act != n_mem:
        raise CT.SchemaError(f"S_ctx built {n_act} action tokens for {n_mem} members")
    xc, xf = _sctx_trailing(cat15, sec, n_act, d, snap, getattr(d, "d_bv", None), picked, tele, fc)
    # The row's arrays, written in place: the head, then the S_ctx block.
    t = len(sec)
    lo = len(hsec)
    n = lo + t
    cat = np.empty((n, N_CAT), np.int16)
    cat[:lo] = hcat
    # The assignments cast exactly as `Encoder.encode`'s `astype` calls do
    # (`RareMap.apply`'s int64 to int16; `T.f16`'s saturating float16).
    cat[lo:, :NS_CAT] = cat15 if rare is None else rare.apply(cat15)
    cat[lo:, NS_CAT:] = xc
    cont = np.empty((n, N_CONT), np.float16)
    cont[:lo] = hcont
    np.clip(tk["cont"], -T.F16_MAX, T.F16_MAX, out=cont[lo:, :NS_CONT], casting="unsafe")
    cont[lo:, NS_CONT:] = xf
    secs = np.empty(n, np.int8)
    secs[:lo] = hsec
    secs[lo:] = sec
    flags = np.empty(n, np.uint8)
    flags[:lo] = hflags
    flags[lo:] = CT.TF_TEACHER
    reps = np.asarray([m[0] for m in d.members], np.int64)
    a0 = n - n_mem
    flags[a0:] |= CT.TF_OPTION | CT.TF_HIDDEN
    flags[a0 + reps] = CT.TF_TEACHER | CT.TF_OPTION | CT.TF_REP
    mem_class = _u16(d.classes)
    mem_tok = _u16(a0 + np.arange(n_mem))
    cls_tok = mem_tok[reps]
    tok = Tok(
        is_combat=True,
        dtype=int(d.dtype),
        cat=cat,
        cont=cont,
        sec=secs,
        tok_flags=flags,
        mem_tok=mem_tok,
        mem_class=mem_class,
        cls_tok=cls_tok,
        cls_kind=np.asarray([d.kinds[r] for r in reps], np.uint8),
        sctx_lo=lo,
        n_sctx=t,
        **_pool_arrays(buf),
    )
    if check:
        check_members_identical(tok, d.members)
    if priv:
        _priv(tok, d, rare)
    return tok


_KEEP_CAT = np.ones(N_CAT, bool)
_KEEP_CAT[_HAND_SLOT] = False


def check_members_identical(tok: Tok, members: list[list[int]]) -> None:
    """Members of one combat class tokenize identically except `hand_slot`."""
    if len(members) == len(tok.mem_tok):
        return
    mt = tok.mem_tok.astype(np.int64)
    ref = tok.cls_tok.astype(np.int64)[tok.mem_class.astype(np.int64)]
    dup = mt != ref
    a, b = mt[dup], ref[dup]
    if (tok.cat[a][:, _KEEP_CAT] != tok.cat[b][:, _KEEP_CAT]).any() or (
        tok.cont[a].view(np.uint16) != tok.cont[b].view(np.uint16)
    ).any():
        bad = [m for m in members if len(m) > 1]
        raise CT.SchemaError(f"members of one class tokenize differently: {bad}")


def _pool_arrays(buf: _Buf) -> dict[str, np.ndarray]:
    if not buf.pool_tok:
        return {}
    return {
        "pool_tok": np.asarray(buf.pool_tok, np.uint16),
        "pool_vid": np.asarray(buf.pool_vid, np.uint16),
        "pool_w": np.asarray(buf.pool_w, np.float64).astype(np.float16),
    }


# ── out of combat ────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def _card_types_canonical(version: str) -> dict[str, str]:
    """`combat_ctx.tokens.card_types(version)` keyed by `ppo.vocab.canonical`
    (595 keys, no collision on v0.111.0)."""
    from ppo.vocab import canonical

    return {canonical(k): v for k, v in T.card_types(version).items()}


def _card_type(cid: str) -> str:
    """A card's `card_type` slot name ("?" when unknown) under any id spelling:
    S_ctx's table is keyed by registry ids (`STRIKE_IRONCLAD`), while
    `master_deck_detailed()` and the offers name cards in PascalCase
    (`StrikeIronclad`)."""
    t = T._card_type(cid)
    if t != "?":
        return t
    import sts2_client

    from ppo.vocab import canonical

    return _card_types_canonical(sts2_client.GAME_VERSION).get(canonical(cid), "?")


def _card_cats(cat: np.ndarray, cid: Any, upgraded: Any, ench: Any) -> None:
    cat[T.S["upgraded"]] = _tslot("upgraded", "1" if upgraded else "0")
    cat[T.S["card_type"]] = _tslot("card_type", _card_type(str(cid)) if cid else "")
    cat[XC["X_ID_MAIN"]] = _vid("card", cid)
    cat[XC["X_ID_MOD"]] = _vid("enchantment", _ench_name(ench))


def _ench_amount(e: Any) -> float:
    if isinstance(e, (list, tuple)) and len(e) > 1:
        return _f(e[1])
    if isinstance(e, dict):
        return _f(e.get("amount"))
    return 0.0


def _deck_tokens(buf: _Buf, deck: list[Any], ench: list[Any]) -> None:
    """One DECK token per distinct (card, upgraded, enchantment)."""
    groups: dict[Any, int] = {}
    order: list[Any] = []
    for i, (cid, up) in enumerate(deck):
        e = ench[i] if i < len(ench) else None
        key = (str(cid), bool(up), _hashable(e))
        if key not in groups:
            groups[key] = 0
            order.append((key, e))
        groups[key] += 1
    for key, e in order:
        cat, cont = buf.new("DECK")
        _card_cats(cat, key[0], key[1], e)
        cont[CT.SCTX_C["ENCH_AMT"]] = _ench_amount(e) / 5.0
        cont[CT.SCTX_C["COUNT"]] = groups[key] / 5.0


def _map_nodes(fm: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    out = {}
    for r in fm.get("rows") or []:
        for n in r:
            out[int(n["col"]), int(n["row"])] = n
    return out


def _kids(n: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(c["col"]), int(c["row"])) for c in n.get("children") or []]


def _route(
    nodes: dict[tuple[int, int], dict[str, Any]], boss_row: int
) -> dict[tuple[int, int], tuple[float, ...]]:
    """Per node, over the paths from it to the boss (the node included):
    elites min/max, rests min/max, most shops, most unknowns, fewest fights
    before the first rest, rows to the boss row."""
    memo: dict[tuple[int, int], tuple[float, ...]] = {}
    for key in sorted(nodes, key=lambda k: -k[1]):
        n = nodes[key]
        typ = str(n.get("type"))
        e, r = float(typ == "Elite"), float(typ == "RestSite")
        sh, un = float(typ == "Shop"), float(typ in ("Unknown", "Event"))
        fight = float(typ in ("Monster", "Elite"))
        ks = [memo[k] for k in _kids(n) if k in memo]
        if ks:
            emin = e + min(k[0] for k in ks)
            emax = e + max(k[1] for k in ks)
            rmin = r + min(k[2] for k in ks)
            rmax = r + max(k[3] for k in ks)
            shm = sh + max(k[4] for k in ks)
            unm = un + max(k[5] for k in ks)
            f2r = 0.0 if r else fight + min(k[6] for k in ks)
        else:
            emin = emax = e
            rmin = rmax = r
            shm, unm = sh, un
            f2r = 0.0 if r else fight
        memo[key] = (emin, emax, rmin, rmax, shm, unm, f2r, float(boss_row - key[1]))
    return memo


def _paths_through(
    nodes: dict[tuple[int, int], dict[str, Any]], start: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """Paths from the current node through each node (forward counts)."""
    cnt: dict[tuple[int, int], int] = {}
    for k in start:
        if k in nodes:
            cnt[k] = cnt.get(k, 0) + 1
    for key in sorted(nodes, key=lambda k: k[1]):
        c = cnt.get(key, 0)
        if not c:
            continue
        for k in _kids(nodes[key]):
            if k in nodes:
                cnt[k] = cnt.get(k, 0) + c
    return cnt


def _map_body(
    buf: _Buf, fm: dict[str, Any], moves: list[dict[str, Any]]
) -> dict[int, tuple[float, ...]]:
    """MAPNODE tokens (the reachable cone, nearest rows first, at most 40) and
    each move column's route summary."""
    nodes = _map_nodes(fm)
    boss_row = _boss_row(fm)
    route = _route(nodes, boss_row)
    cur = fm.get("current_coord")
    cur_row = int(cur["row"]) if cur else -1
    if cur and (int(cur["col"]), cur_row) in nodes:
        start = _kids(nodes[int(cur["col"]), cur_row])
    else:
        start = [k for k in nodes if k[1] == cur_row + 1]
    paths = _paths_through(nodes, start)
    nxt = {int(a["col"]) for a in moves if a.get("action") == "move"}
    cone = sorted(paths, key=lambda k: (k[1], k[0]))[: CT.MAX_MAPNODES]
    for k in cone:
        n = nodes[k]
        cat, cont = buf.new("MAPNODE")
        cat[XC["X_ROOM"]] = CT.ROOM_ID[_ROOM_OF_NODE.get(str(n.get("type")), "unknown")]
        cat[XC["X_POS"]] = _pos(k[0])
        cont[XF["N_ROW_OFF"]] = (k[1] - cur_row) / 17.0
        cont[XF["N_COL"]] = k[0] / 6.0
        cont[XF["N_NEXT"]] = float(k[1] == cur_row + 1 and k[0] in nxt)
        cont[XF["N_PATHS"]] = math.log1p(paths[k]) / 5.0
    out = {}
    for col in nxt:
        st = route.get((col, cur_row + 1))
        if st is not None:
            out[col] = st
    return out


def _route_cols(cont: np.ndarray, st: tuple[float, ...]) -> None:
    for name, v in zip(
        (
            "M_ELITE_MIN",
            "M_ELITE_MAX",
            "M_REST_MIN",
            "M_REST_MAX",
            "M_SHOP_MAX",
            "M_UNKNOWN_MAX",
            "M_FIGHTS_TO_REST",
        ),
        st[:7],
        strict=True,
    ):
        cont[XF[name]] = v / 5.0
    cont[XF["M_ROWS_LEFT"]] = st[7] / 17.0


def _neow_relic(label: Any) -> int:
    """The relic an event option's label or id names (Neow), or 0."""
    v = V.load()
    for s in label:
        if s and v.has("relic", str(s)):
            return v.id("relic", str(s))
    return 0


def _option(
    buf: _Buf,
    d: Any,
    run: _Run,
    i: int,
    a: dict[str, Any],
    kind: int,
    deck_copies: Counter,
    route: dict[int, tuple[float, ...]],
    fm: dict[str, Any],
) -> None:
    """The token of option `i` (the representative of its class)."""
    obs = run.obs
    name = CT.ACTION_KINDS[kind].name
    flags = CT.TF_OPTION | CT.TF_REP

    def card_opt(cid: Any, up: Any, ench: Any) -> tuple[np.ndarray, np.ndarray]:
        cat, cont = buf.new("OPT_CARD", flags)
        _card_cats(cat, cid, up, ench)
        cont[CT.SCTX_C["ENCH_AMT"]] = _ench_amount(ench) / 5.0
        cont[XF["K_DECK_COPIES"]] = deck_copies.get(str(cid), 0) / 5.0
        return cat, cont

    def deck_opt(deck_i: Any) -> tuple[np.ndarray, np.ndarray]:
        ent = _at(obs.get("deck") or run.deck(), deck_i)
        e = _at(obs.get("deck_enchantments") or [], deck_i)
        cid, up = (ent[0], ent[1]) if ent is not None else (None, False)
        return card_opt(cid, up, e)

    if name in ("PICK_CARD_REWARD", "PICK_CARD_BUNDLE"):
        idx = a["idx"]
        if name == "PICK_CARD_BUNDLE":
            pack = _at(obs.get("card_reward_bundles") or [], idx) or []
            cat, cont = card_opt(pack[0] if pack else None, False, None)
            cont[CT.SCTX_C["COUNT"]] = len(pack) / 5.0
        else:
            cat, cont = card_opt(
                _at(obs.get("card_reward_option_ids"), idx),
                _at(obs.get("card_reward_option_upgraded"), idx),
                _at(obs.get("card_reward_option_enchantments"), idx),
            )
    elif name == "SMITH_CARD":
        idx = a["idx"]
        ent = _at(obs.get("smith_options"), idx)
        cid, up = (ent[0], ent[1]) if isinstance(ent, (list, tuple)) else (ent, False)
        cat, cont = card_opt(cid, up, _at(obs.get("smith_option_enchantments"), idx))
    elif name == "REMOVE_DECK_CARD":
        idx = a["idx"]
        ent = _at(obs.get("card_removal_options"), idx)
        cid, up = (ent[0], ent[1]) if isinstance(ent, (list, tuple)) else (ent, False)
        cat, cont = card_opt(cid, up, _at(obs.get("card_removal_option_enchantments"), idx))
    elif name == "GRANT_DECK_CARD":
        idx = a["idx"]
        ent = _at(obs.get("card_grant_options"), idx)
        cid, up = (ent[0], ent[1]) if isinstance(ent, (list, tuple)) else (ent, False)
        cat, cont = card_opt(cid, up, _at(obs.get("card_grant_option_enchantments"), idx))
    elif name in ("UPGRADE_DECK_CARD", "TRANSFORM_DECK_CARD", "ENCHANT_DECK_CARD"):
        key = {
            "UPGRADE_DECK_CARD": "upgrade_select_deck_indices",
            "TRANSFORM_DECK_CARD": "transform_select_deck_indices",
            "ENCHANT_DECK_CARD": "enchant_select_deck_indices",
        }[name]
        cat, cont = deck_opt(_at(obs.get(key), a["idx"]))
    elif name == "SHOP_REMOVE_CARD":
        cat, cont = deck_opt(a["deck_idx"])
        cont[XF["O_PRICE"]] = _f((obs.get("shop") or {}).get("removal_cost")) / 300.0
    elif name in ("SHOP_BUY_CARD", "SHOP_BUY_RELIC", "SHOP_BUY_POTION"):
        shelf = {"SHOP_BUY_CARD": "cards", "SHOP_BUY_RELIC": "relics"}.get(name, "potions")
        items = {int(x["idx"]): x for x in (obs.get("shop") or {}).get(shelf) or []}
        e = items.get(int(a["idx"]), {})
        if name == "SHOP_BUY_CARD":
            cat, cont = card_opt(e.get("id"), e.get("upgraded"), e.get("enchantment"))
        else:
            reg = "relic" if name == "SHOP_BUY_RELIC" else "potion"
            cat, cont = buf.new("OPT_RELIC" if reg == "relic" else "OPT_POTION", flags)
            cat[XC["X_ID_MAIN"]] = _vid(reg, e.get("id"))
        cont[XF["O_PRICE"]] = _f(e.get("price")) / 300.0
    elif name in ("PICK_RELIC_REWARD", "PICK_BOSS_RELIC"):
        lst = obs.get("relic_reward_ids" if name == "PICK_RELIC_REWARD" else "boss_relic_options")
        cat, cont = buf.new("OPT_RELIC", flags)
        cat[XC["X_ID_MAIN"]] = _vid("relic", _at(lst, a.get("idx", 0)))
    elif name == "POTION_SWAP_DISCARD":
        cat, cont = buf.new("OPT_POTION", flags)
        cat[XC["X_ID_MAIN"]] = _vid("potion", _at(run.potions(), a["discard_slot"]))
    elif name == "MOVE":
        cat, cont = buf.new("OPT_MOVE", flags)
        col = int(a["col"])
        nodes = _map_nodes(fm)
        cur_row = _cur_row(fm)
        n = nodes.get((col, cur_row + 1), {})
        cat[XC["X_ROOM"]] = CT.ROOM_ID[_ROOM_OF_NODE.get(str(n.get("type")), "unknown")]
        st = route.get(col)
        if st is not None:
            _route_cols(cont, st)
        cat[XC["X_KIND"]] = kind
        cat[XC["X_POS"]] = _pos(col)
        return
    elif name == "EVENT_CHOICE":
        idx = int(a["idx"])
        infos = obs.get("event_options")
        if infos is None:
            infos = run.f.event_option_infos()
        opt = _at(infos, idx) or {}
        cat, cont = buf.new("OPT_EVENT", flags)
        cat[XC["X_ID_MAIN"]] = _vid("event", _event_name(d, run))
        cat[XC["X_ID_MOD"]] = _neow_relic((opt.get("label"), opt.get("option_id")))
        # The honest preview (SCHEMA.md §2.7): a relic-gaining option's gains are 0.
        for col, v in CT.event_delta_columns(opt).items():
            cont[XF[col]] = v
    elif name == "REST_OPTION":
        cat, cont = buf.new("OPT_REST", flags)
        opt = _at(_rest_options(run), a.get("idx", 0))
        rid = opt[0] if isinstance(opt, (list, tuple)) else opt
        cat[XC["X_ID_MAIN"]] = _vid("rest", rid) if rid else 0
    else:
        cat, cont = buf.new("OPT_FIXED", flags)
    cat[XC["X_KIND"]] = kind
    cat[XC["X_POS"]] = _pos(i)


def _rest_options(run: _Run) -> list[Any]:
    """`rm.rest_site_options()` (never `REST_OPTION_LABELS`: a relic's third
    option would be mislabeled)."""
    if "rest_site_options" in run.obs:
        return list(run.obs["rest_site_options"])
    return list(run.f.rest_site_options())


def _offer(buf: _Buf, d: Any, run: _Run) -> None:
    """Offered content that is not an option: the potion a swap would add,
    the enchantment an enchant pick stamps."""
    if d.dtype == DT["POTION_SWAP"]:
        pot = run.obs.get("pending_potion_swap")
        if pot is None:
            pot = run.f.pending_potion_swap
        if pot:
            cat, _cont = buf.new("OFFER")
            cat[XC["X_KIND"]] = AK["POTION_SWAP_DISCARD"]
            cat[XC["X_ID_MAIN"]] = _vid("potion", pot)
    elif d.dtype == DT["CARD_ENCHANT"]:
        e = run.obs.get("pending_enchantment")
        if e is None:
            e = run.read("pending_enchantment")
        if e:
            cat, _cont = buf.new("OFFER")
            cat[XC["X_KIND"]] = AK["ENCHANT_DECK_CARD"]
            cat[XC["X_ID_MAIN"]] = _vid("enchantment", _ench_name(e))


def encode_ooc(d: Any, *, priv: bool = True) -> Tok:
    """An out-of-combat decision's row."""
    run = _Run(d)
    buf, deck, ench = _head(d, run, None, None)
    _deck_tokens(buf, deck, ench)
    for r in run.relics():
        cat, _cont = buf.new("RELIC")
        cat[XC["X_ID_MAIN"]] = _vid("relic", r)
    for j, p in enumerate(run.potions()):
        if p:
            cat, _cont = buf.new("POTION")
            cat[XC["X_ID_MAIN"]] = _vid("potion", p)
            cat[XC["X_POS"]] = _pos(j)
    _offer(buf, d, run)
    fm = _static(run)["fm"]
    route = _map_body(buf, fm, d.obs_actions) if d.dtype == DT["MAP"] else {}
    copies: Counter = Counter(str(c) for c, _u in deck)
    reps = [m[0] for m in d.members]
    cls_tok = []
    for c, r in enumerate(reps):
        cls_tok.append(len(buf))
        _option(buf, d, run, c, d.obs_actions[r], d.kinds[r], copies, route, fm)
    cat, cont, sec, flags = buf.arrays()
    cls_tok_a = _u16(cls_tok)
    mem_class = _u16(d.classes)
    tok = Tok(
        is_combat=False,
        dtype=int(d.dtype),
        cat=cat,
        cont=cont.astype(np.float16),
        sec=sec,
        tok_flags=flags,
        mem_tok=cls_tok_a[mem_class.astype(np.int64)],
        mem_class=mem_class,
        cls_tok=cls_tok_a,
        cls_kind=np.asarray([d.kinds[r] for r in reps], np.uint8),
        **_pool_arrays(buf),
    )
    if priv:
        _priv(tok, d, None)
    return tok


# ── PRIV (value only) ────────────────────────────────────────────────────


def _priv_arrays(d: Any, rare: Any) -> dict[str, np.ndarray]:
    """PRIV_DRAW, PRIV_ENEMY and PRIV_RUN (16 at most): the row's PRIV arrays
    under their `ROW_FIELDS` names. Reads `d.priv()` (a `PrivSource`)."""
    p = d.priv() if hasattr(d, "priv") else None
    cs = p.true_state() if p is not None else getattr(d, "cs", None)
    pc: list[np.ndarray] = []
    pf: list[np.ndarray] = []
    ps: list[int] = []
    pool: list[tuple[int, int, float]] = []
    if cs is not None:
        _priv_draw(cs, rare, pc, pf, ps, _memo_key(d))
        _priv_enemy(cs, d.snap, pc, pf, ps)
    if p is not None:
        _priv_run(_Run(d), p, pc, pf, ps, pool)
    if len(ps) > CT.MAX_PRIV:
        raise CT.SchemaError(f"{len(ps)} PRIV tokens > {CT.MAX_PRIV}")
    # `np.concatenate` copies: the PRIV_RUN parts are cached per floor.
    out = {}
    if ps:
        out["pcat"] = np.concatenate(pc).astype(np.int16, copy=False)
        out["pcont"] = np.concatenate(pf).astype(np.float16, copy=False)
        out["psec"] = np.asarray(ps, np.int8)
    else:
        out["pcat"] = np.zeros((0, N_CAT), np.int16)
        out["pcont"] = np.zeros((0, N_CONT), np.float16)
        out["psec"] = np.zeros(0, np.int8)
    if pool:  # one (pos, vids, weights) per pooled PRIV token
        out["ppool_tok"] = np.concatenate([np.full(len(v), pos, np.uint16) for pos, v, _ in pool])
        out["ppool_vid"] = np.concatenate([v for _, v, _ in pool])
        out["ppool_w"] = np.concatenate([w for _, _, w in pool])
    else:
        out["ppool_tok"] = np.zeros(0, np.uint16)
        out["ppool_vid"] = np.zeros(0, np.uint16)
        out["ppool_w"] = np.zeros(0, np.float16)
    return out


def _priv(tok: Tok, d: Any, rare: Any) -> None:
    for k, v in _priv_arrays(d, rare).items():
        setattr(tok, k, v)


def _priv_draw(cs: Any, rare: Any, pc: list, pf: list, ps: list, run: Any = None) -> None:
    """The true top-10 draw: #1081's priv layout (`COUNT` = rank). `run` keys the memo."""
    import sts2_rs

    g = _getter(cs, "true_draw_top")
    if g is not None:
        draw = list(g(CT.PRIV_DRAW_K))
    else:
        draw = sts2_rs.state_to_dict(cs)["deck"].get("draw_pile", [])[: CT.PRIV_DRAW_K]
    if not draw:
        return
    # The rows depend on `rare` (held, compared with `is`) and on the build's
    # content tables (`luts`).
    import sts2_client

    key = (rare, sts2_client.GAME_VERSION, draw)
    hit = _MEMO.get((run, "priv_draw"))
    if hit is not None and hit[0][0] is rare and hit[0][1] == key[1]:
        if hit[0][2] == draw:
            cat, cont = hit[1]
        else:
            cat, cont = _priv_draw_after(hit[0][2], *hit[1], draw, rare)
            _memo_put(run, "priv_draw", (key, (cat, cont)))
    else:
        cat, cont = _priv_draw_rows(draw, rare)
        _memo_put(run, "priv_draw", (key, (cat, cont)))
    pc.append(cat)
    pf.append(cont)
    ps.extend([SEC["PRIV_DRAW"]] * len(cat))


# A PRIV_DRAW row's rank column as `card_feats` writes it (float32 `rank / 5`)
# and `T.f16` casts it.
_DRAW_RANK = T.f16(np.asarray([(r + 1) / 5.0 for r in range(CT.PRIV_DRAW_K)], np.float32))


def _priv_draw_after(
    prev: list[dict[str, Any]],
    pcat: np.ndarray,
    pcont: np.ndarray,
    draw: list[dict[str, Any]],
    rare: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """The rows of `draw` from those of `prev`, the previous call's: after k
    draws, the old window's cards k.. are the new window's 0.., each k ranks
    nearer the top (a card's rows differ only in the rank column), and only
    the cards below them need `card_feats`."""
    n = len(prev)
    for k in range(1, n):
        m = n - k
        if len(draw) >= m and draw[:m] == prev[k:]:
            cat = np.empty((len(draw), N_CAT), np.int16)
            cont = np.empty((len(draw), N_CONT), np.float16)
            cat[:m] = pcat[k:]
            cont[:m] = pcont[k:]
            cont[:m, T.C_COUNT] = _DRAW_RANK[:m]
            if len(draw) > m:
                cat[m:], cont[m:] = _priv_draw_rows(draw[m:], rare, start=m)
            return cat, cont
    return _priv_draw_rows(draw, rare)


def _priv_draw_rows(
    draw: list[dict[str, Any]], rare: Any, start: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """PRIV_DRAW rows of `draw`, the cards at ranks `start + 1`, ..."""
    from ppo.combat_ctx.snapshot import _card

    b = T._Builder(T.vocab())
    for r, c in enumerate(draw):
        cat, cont = b.new("DRAW_TOP")
        T.card_feats(b, cat, cont, _card(c), float(start + r + 1))
    raw = np.stack(b.cat).astype(np.int16)
    k = len(raw)
    L = luts()
    cat = np.zeros((k, N_CAT), np.int16)
    cat[:, :NS_CAT] = _rare_apply(rare, raw)
    cat[:, XC["X_ID_MAIN"]] = L["card"][raw[:, T.S["card"]]]
    cat[:, XC["X_ID_MOD"]] = L["enchantment"][raw[:, T.S["enchantment"]]]
    cont = np.zeros((k, N_CONT), np.float16)
    cont[:, :NS_CONT] = T.f16(np.stack(b.cont))
    return cat, cont


def _priv_enemy(cs: Any, snap: dict[str, Any] | None, pc: list, pf: list, ps: list) -> None:
    """Per live enemy, the true next three moves (WP-ENG-1 `peek_intents(3)`)."""
    g = _getter(cs, "peek_intents")
    if g is None:
        return
    fc = g(3)
    enemies = (snap or {}).get("enemies") or []
    for j, steps in enumerate(list(fc)[: CT.PRIV_ENEMY_K]):
        if not steps or _dead(enemies, j):
            continue
        row = _ABSENT_CAT.copy()
        row[T.S["enemy_slot"]] = _tslot("enemy_slot", str(j))
        kind = enemies[j]["kind"] if j < len(enemies) else None
        for col, st in zip(("X_ID_MAIN", "X_ID_MOD", "X_ID_REL"), steps, strict=False):
            mid = st.get("move_id")
            row[XC[col]] = _vid("move", f"{kind}.{mid}") if mid and kind else 0
        row[XC["X_ID_AUX"]] = _vid("monster", kind) if kind else 0
        pc.append(row[None])
        pf.append(np.zeros((1, N_CONT), np.float16))
        ps.append(SEC["PRIV_ENEMY"])


_ROOM_COUNT_TYPES = (
    ("P_ROOMS_SHOP", "Shop"),
    ("P_ROOMS_REST", "RestSite"),
    ("P_ROOMS_TREASURE", "Treasure"),
)


def _priv_run(run: _Run, p: Any, pc: list, pf: list, ps: list, pool: list) -> None:
    """This act's room queues and the later acts' bosses (`PrivSource.act_rooms`,
    `future`), cached with the run-static reads."""
    key = (run.act(), run.floor())
    st = _static_hit(run, key)
    pr = st.get("priv_run") if st is not None else None
    if pr is None:
        pr = _priv_run_static(p, run.act(), st["fm"] if st is not None else run.full_map())
        if st is not None:
            st["priv_run"] = pr
    row, f, vids, ws = pr
    pos = len(ps)
    pc.append(row)
    pf.append(f)
    ps.append(SEC["PRIV_RUN"])
    if len(vids):
        pool.append((pos, vids, ws))


def _rooms_record(x: Any, what: str) -> Mapping[str, Any]:
    """One act's `generated_rooms()` record (`contracts.PrivSource`). Anything
    else raises: the env's `PrivSource` once returned `list(record)`, the
    record's keys, and PRIV_RUN read zero rooms and no bosses without a sound."""
    if not isinstance(x, Mapping):
        raise CT.SchemaError(f"PrivSource.{what} is {type(x).__name__}, not a rooms record")
    return x


def _priv_run_static(
    p: Any, act: int, fm: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = _rooms_record(p.act_rooms(), "act_rooms()")
    later = [_rooms_record(x, "future()[i]") for x in (p.future() or [])]
    row = _ABSENT_CAT.copy()
    f = np.zeros(N_CONT, np.float32)
    f[XF["P_ROOMS_MONSTER"]] = len(g.get("normal") or []) / 10.0
    f[XF["P_ROOMS_ELITE"]] = len(g.get("elite") or []) / 10.0
    f[XF["P_ROOMS_EVENT"]] = len(g.get("events_shuffled") or []) / 10.0
    # Shops, rests and chests have no queue: the act's nodes ahead.
    cur = _cur_row(fm)
    ahead = Counter(x.get("type") for r in fm.get("rows") or [] for x in r if int(x["row"]) > cur)
    for col, typ in _ROOM_COUNT_TYPES:
        f[XF[col]] = ahead.get(typ, 0) / 10.0
    if later:
        row[XC["X_ID_MAIN"]] = _vid("encounter", later[0].get("boss"))
        if act < 3 and len(later) == 3 - act:
            row[XC["X_ID_REL"]] = _vid("encounter", later[-1].get("boss"))
            row[XC["X_ID_MOD"]] = _vid("encounter", later[-1].get("second_boss"))
    enc: Counter = Counter(
        _vid("encounter", e) for e in (g.get("normal") or []) + (g.get("elite") or [])
    )
    ent = [(vid, float(w)) for vid, w in enc.items() if vid]
    vids = np.asarray([v for v, _w in ent], np.uint16)
    ws = np.asarray([w for _v, w in ent], np.float64).astype(np.float16)
    return row[None], f.astype(np.float16)[None], vids, ws


class RunPriv:
    """A `contracts.PrivSource` over a recorded run (`decision_from_state`):
    `act_rooms()` is this act's `generated_rooms()` record, `future()` the
    later acts' records, whose bosses PRIV_RUN reads (it reads no later map)."""

    __slots__ = ("_cs", "_rm")

    def __init__(self, cs: Any, rm: Any) -> None:
        self._cs = cs
        self._rm = rm

    def true_state(self) -> Any:
        return self._cs

    def act_rooms(self) -> Any:
        rooms = self._rm.generated_rooms()
        a = int(self._rm.act_index) - 1
        return rooms[a] if 0 <= a < len(rooms) else {}

    def future(self) -> Any:
        return list(self._rm.generated_rooms()[int(self._rm.act_index) :])


# ── entry points ─────────────────────────────────────────────────────────


def encode(d: Any, *, rare: Any = None, priv: bool = True) -> Tok:
    """One decision's row arrays, for an `env.E2EDecision`-like object (it has
    `rm`, `policy()`, `priv()`, the combat view and `sel`; `RecordedDecision`
    is one). `rare` is the combat teacher's `RareMap` (None: S_ctx content
    slots stay universe rows). A `contracts.EnvDecision` goes through
    `Tokenizer`, which keeps the fight's intent log."""
    if isinstance(d, CT.EnvDecision):
        raise TypeError("encode a contracts.EnvDecision with a tokens.Tokenizer")
    if d.is_combat:
        return encode_combat(d, rare=rare, priv=priv)
    return encode_ooc(d, priv=priv)


@dataclass(eq=False)
class _EnvView:
    """A `contracts.EnvDecision` read like a decision: `honest_obs(d.obs)` and
    the tokenizer's combat view (`bv`, `d_bv`, `snap`, `room`)."""

    is_combat: bool
    dtype: int
    kinds: list[int]
    obs_actions: list[dict[str, Any]]
    classes: list[int]
    members: list[list[int]]
    sel: Any
    event_name: str | None
    run_obs: dict[str, Any] | None
    run_key: tuple[str, int]
    src: Any
    psrc: Any
    bv: Any = None
    d_bv: dict[str, Any] | None = None
    snap: dict[str, Any] | None = None
    room: str | None = None

    def policy(self) -> Any:
        return self.src

    def priv(self) -> Any:
        return self.psrc


def _members(member_class: list[int], n_classes: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(n_classes)]
    for i, c in enumerate(member_class):
        out[c].append(i)
    return out


def fight_room(run: Any) -> str:
    """The fight's room (`monster`, `elite`, `boss`) from the run facade, as the
    env's `fight_room` reads it."""
    room = (run.current_room() or {}).get("room_type")
    return room if room in ("monster", "elite", "boss") else "monster"


class Tokenizer:
    """WP3b's `contracts.Tokenizer` (SCHEMA.md §8.2), as the env and the actor
    drive it.

    `reset_fight()` at every fight start and `observe(policy)` at every combat
    step, forced ones included, before the action: the fight's `IntentLog`
    (#1081's `Encoder.observe`). `encode(d)` returns one row's policy arrays
    and `encode_priv(d)` its PRIV arrays, under the `ROW_FIELDS` names; `encode`
    never reads `d.priv`. `rare` is the combat teacher's `RareMap`.

    It reads `d.obs` only as `contracts.honest_obs(d.obs)`, and the run
    through `d.policy.run()` (`contracts.POLICY_CALLS`).
    """

    def __init__(self, rare: Any = None, *, check: bool = True) -> None:
        from ppo.combat_ctx.intent_log import IntentLog

        self.rare = rare
        self.check = check
        self._ilog_cls = IntentLog
        self._ilog = IntentLog()
        self._view: tuple[Any, dict[str, Any], dict[str, Any], str] | None = None
        self._t: dict[int, int] = {}
        self._last: tuple[Any, _EnvView, Tok | None] | None = None

    def reset_fight(self) -> None:
        self._ilog = self._ilog_cls()
        self._view = None

    def observe(self, policy: Any) -> None:
        import sts2_rs

        from ppo.combat_ctx.snapshot import snapshot_from_parts

        bv = policy.belief()
        d_bv = sts2_rs.state_to_dict(bv)
        room = fight_room(policy.run())
        snap = snapshot_from_parts(
            d_bv, bv.card_damage_preview(), bv.card_block_preview(), bv.pending_choice(), room
        )
        self._view = (bv, d_bv, self._ilog.observe(snap), room)

    def _env_view(self, d: Any) -> _EnvView:
        if self._last is not None and self._last[0] is d:
            return self._last[1]
        uid = int(d.run_uid)
        if int(d.t) < self._t.pop(uid, -1):
            # The uid's run started again: drop its run-static reads.
            clear_cache(("uid", uid))
        self._t[uid] = int(d.t)  # re-inserted: the dict runs least recent first
        if len(self._t) > _T_MAX:
            old = next(iter(self._t))
            del self._t[old]
            clear_cache(("uid", old))
        obs = CT.honest_obs(d.obs) if d.obs is not None else None
        view: tuple[Any, ...] = (None, None, None, None)
        if d.is_combat:
            bv = d.policy.belief()
            if self._view is None or self._view[0] is not bv:
                # No observe for this state (a caller that skips the env's
                # hooks): observe it now; an intent log needs every step.
                self.observe(d.policy)
            view = self._view  # type: ignore[assignment]
        classes = np.asarray(d.member_class).tolist()
        v = _EnvView(
            is_combat=bool(d.is_combat),
            dtype=int(d.dtype),
            kinds=np.asarray(d.kinds).tolist(),
            obs_actions=list(d.actions),
            classes=classes,
            members=_members(classes, len(d.rep)),
            sel=d.sel,
            event_name=d.event_name,
            run_obs=obs,
            run_key=("uid", uid),
            src=d.policy,
            psrc=d.priv,
            bv=view[0],
            d_bv=view[1],
            snap=view[2],
            room=view[3],
        )
        self._last = (d, v, None)
        return v

    def tok(self, d: Any) -> Tok:
        """The decision's `Tok` without PRIV (cached for the `encode_priv` that follows)."""
        v = self._env_view(d)
        if self._last is not None and self._last[0] is d and self._last[2] is not None:
            return self._last[2]
        if v.is_combat:
            t = encode_combat(v, rare=self.rare, priv=False, check=self.check)
        else:
            t = encode_ooc(v, priv=False)
        self._last = (d, v, t)
        return t

    def encode(self, d: Any) -> dict[str, Any]:
        t = self.tok(d)
        return {
            "cat": t.cat,
            "cont": t.cont,
            "sec": t.sec,
            "tok_flags": t.tok_flags,
            "mem_tok": t.mem_tok,
            "mem_class": t.mem_class,
            "cls_tok": t.cls_tok,
            "cls_kind": t.cls_kind,
            "pool_tok": t.pool_tok,
            "pool_vid": t.pool_vid,
            "pool_w": t.pool_w,
            "sctx_lo": t.sctx_lo,
            "n_sctx": t.n_sctx,
        }

    def encode_priv(self, d: Any) -> dict[str, Any]:
        if d.priv is None:
            return _priv_arrays(_NoPriv(), self.rare)
        return _priv_arrays(self._env_view(d), self.rare)


class _NoPriv:
    snap = None


class SelectionTracker:
    """The grant picks the env applied since the grant opened (`Selection.picked`
    on CARD_GRANT), for recorded runs: feed every decision of one run in order.

    A card grant that requires k picks arrives as k decisions whose offer
    shrinks by the picked card; Sea Glass's optional grid likewise."""

    def __init__(self) -> None:
        self._key: Any = None
        self._n = 0
        self._last = 0

    def picked(self, run: Any, dtype: int) -> int:
        """`run` is the run facade (`card_grant_options`, `act_index`, `total_floor`)."""
        if dtype != DT["CARD_GRANT"]:
            self._key, self._n = None, 0
            return 0
        opts = len(run.card_grant_options())
        key = (int(run.act_index), int(run.total_floor()))
        if key == self._key and opts < self._last:
            self._n += self._last - opts
        elif key != self._key:
            self._n = 0
        self._key, self._last = key, opts
        return self._n


def _grant_selection(run: Any, kinds: list[int], picked: int) -> Any:
    return CT.Selection(
        min=int(run.pending_card_grant_min or 0),
        max=sum(1 for k in kinds if k == AK["GRANT_DECK_CARD"]),
        picked=picked,
        can_skip=AK["GRANT_CONFIRM"] in kinds,
    )


@dataclass(eq=False)
class RecordedDecision:
    """A recorded state (`ppo.e2e.recorder.State`) as an `E2EDecision`-like
    object: the fields the tokenizer reads, classified and deduped as the env
    does (`env.merged_classes`, `env.classify_ooc` + `env.ooc_keys`)."""

    rm: Any
    kind: str
    dtype: int
    kinds: list[int]
    actions: list[dict[str, Any]]
    obs_actions: list[dict[str, Any]]
    classes: list[int]
    members: list[list[int]]
    keys: list[str] = field(default_factory=list)
    sel: Any = CT.NO_SELECTION
    event_name: str | None = None
    cs: Any = None
    bv: Any = None
    d_bv: dict[str, Any] | None = None
    snap: dict[str, Any] | None = None
    room: str | None = None
    run_obs: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    @property
    def is_combat(self) -> bool:
        return self.kind == "combat"

    @property
    def n_classes(self) -> int:
        return len(self.members)

    def policy(self) -> Any:
        from .env import PolicySource

        return PolicySource(self.bv, self.obs_actions, self.rm)

    def priv(self) -> Any:
        return RunPriv(self.cs, self.rm)

    def env_decision(self, *, run_uid: int, t: int) -> Any:
        """This decision as the schema's `contracts.EnvDecision` (SCHEMA.md §8.1)."""
        combat = self.is_combat
        return CT.make_decision(
            run_uid=run_uid,
            t=t,
            dtype=self.dtype,
            actions=self.obs_actions,
            kinds=self.kinds,
            keys=self.keys,
            policy=self.policy(),
            priv=self.priv(),
            sel=self.sel,
            event_name=self.event_name,
            obs=None if combat else self.run_obs,
            metrics=None if combat else (self.metrics or {}),
            fight_turn=int((self.snap or {}).get("round") or 1) if combat else 0,
        )


def decision_from_state(
    st: Any, tracker: SelectionTracker | None = None
) -> RecordedDecision | None:
    """The decision at a recorded state, or None when it has one merged class.
    `tracker` counts a card grant's picks (feed it every state of the run)."""
    import sts2_rs

    from ppo.env import classify_surface, metrics_snapshot, observe

    from . import env as E

    if st.cs is not None:
        cc = E.merged_classes(st.cs, room=st.room or "monster")
        sel = CT.NO_SELECTION
        if cc.dtype in CT.SELECTION_DTYPES:
            sel = E.combat_selection(st.bv.pending_choice(), cc.kinds)
        d = RecordedDecision(
            rm=st.rm,
            kind="combat",
            dtype=cc.dtype,
            kinds=cc.kinds,
            actions=st.cs.legal_actions(),
            obs_actions=cc.obs_actions,
            classes=cc.classes,
            members=cc.members,
            keys=list(cc.keys),
            sel=sel,
            cs=st.cs,
            bv=st.bv,
            d_bv=sts2_rs.state_to_dict(st.bv),
            snap=st.snap,
            room=st.room,
        )
        if tracker is not None:
            tracker.picked(None, cc.dtype)
    else:
        acts = st.actions
        obs = observe(st.rm, st.phase, classify_surface(st.phase, acts))
        dtype, kinds = E.classify_ooc(acts, st.rm, obs)
        keys = E.ooc_keys(acts, kinds, obs)
        cls, members = E.classes_from_keys(keys)
        sel = CT.NO_SELECTION
        if dtype == DT["CARD_GRANT"]:
            n = tracker.picked(st.rm, dtype) if tracker is not None else 0
            sel = _grant_selection(st.rm, kinds, n)
        elif tracker is not None:
            tracker.picked(st.rm, dtype)
        d = RecordedDecision(
            rm=st.rm,
            kind="ooc",
            dtype=dtype,
            kinds=kinds,
            actions=acts,
            obs_actions=acts,
            classes=cls,
            members=members,
            keys=list(keys),
            sel=sel,
            event_name=st.rm.current_event_name() if dtype in (DT["EVENT"], DT["NEOW"]) else None,
            run_obs=obs,
            metrics=metrics_snapshot(st.rm),
        )
    return d if len(d.members) > 1 else None
