"""`E2ENet`: the end-to-end student network (plan §5; precision §6).

One entity transformer plays every decision of an Ironclad run, combat and
out of combat (OOC). Its input is the actor-to-learner row batch that
`ppo.e2e.contracts` freezes (`ROW_FIELDS`): per row, a set of tokens with
categorical columns `cat`, continuous columns `cont`, a section `sec` and
flags `tok_flags`, plus the merged option classes, the summary pools and the
privileged (PRIV) tokens. `to_batch` pads rows into a `TokenBatch`.

The student reads, per token, the sum of:

* the content ids in the trailing `X_ID_MAIN`, `X_ID_MOD`, `X_ID_REL` and
  `X_ID_AUX` columns, each looked up in ONE shared ID table (4,096 x 256,
  row 0 = NONE, zero and never trained) and scaled by a per-column gain
  vector. The gains tell the roles apart: two ids of one registry in one
  token (PRIV_RUN's three bosses) would be order-ambiguous under a plain sum.
  The honest path's `X_ID_AUX` gain starts at zero, because that column is
  reserved there (RUNCTX's encounter id, WP-ENG-1), unless the launch fills
  WP-ENG-1 (`filled`, below); PRIV tokens have their own gains, all at one,
  since PRIV_ENEMY fills `X_ID_AUX` from the start.
* for a pooled summary token, the weighted mean of the ID embeddings of its
  pool entries (and of their content-prior rows), scaled by a gain vector.
* one embedding per small categorical column: the S_ctx slots in
  `contracts.SCTX_STUDENT_SLOTS` and the trailing non-ID columns (`X_DTYPE`,
  `X_KIND`, `X_ROOM`, `X_POS` and the reserved ones). An absent value (S_ctx
  PAD, trailing 0) adds nothing. The student never reads the S_ctx content
  slots: they hold the combat teacher's RareMap'd rows (X-C).
* a section embedding.
* the lean numeric stem of `cont` plus the 48 content-prior columns
  (`contracts.PRIOR_COLUMNS`, gathered from a frozen `[4096, 48]` buffer by
  `X_ID_MAIN`): symlog, then Linear -> SiLU -> Linear. There is no
  normalisation across columns first, so an all-zero column adds nothing.

Reserved inputs start inert (plan §4 "zero-init reserved slots", SCHEMA.md
section 2.12, the identity rule): the first-layer stem weights of every
reserved continuous column and of every prior column are zero, as are the
embedding rows of reserved categorical columns, reserved sections, reserved
and spare action kinds and decision types, the honest `X_ID_AUX` gain, and
the ID-table rows of the registry blocks SCHEMA.md names (`ZERO_ID_REGISTRIES`:
encounters, moves). Filling a reserved input later (a `contracts.FILL_EVENTS`
event, or `E2ENet.set_prior`) leaves every output unchanged until training
moves those weights. A fill event that the launch's rows already carry is not
"later": `E2ENet(filled=...)` (the learner's `--filled`) inits that event's
inputs like live ones, so the launch with `--filled '["WP-ENG-1"]'` starts
with the telegraph and forecast stem weights and embedding rows, the honest
`X_ID_AUX` gain at 1 and the encounter and move ID rows at normal init.

Tokens flagged `TF_HIDDEN` (non-representative members of a merged combat
class) are dropped: the student never sees them, only the combat teacher
does. RUNCTX is token 0 of every row and is the state token.

The trunk is `n_layers` pre-norm blocks (RMSNorm; 4 x 64 attention heads
through `scaled_dot_product_attention` with a key-padding mask; SwiGLU 704;
no biases), with no positional encoding: a token set is unordered.

Heads:

* Policy: a combat MLP and an OOC MLP over `[h_option; h_STATE; e_kind]`
  (a class representative's and the RUNCTX token's trunk outputs and the
  embedding of the option's action kind, `cls_kind`, 128 rows; SCHEMA.md
  section 4) -> 256 -> 1, one logit per merged option class, masked, with an
  fp32 output projection and log-softmax. The decision type picks the head
  (`is_combat`) and reaches it through `h_STATE` (`X_DTYPE` on RUNCTX).
* Value: one more block over `[trunk residual stream; PRIV tokens]`, then a
  `V_combat` and a `V_run` head read at RUNCTX, zero-initialised and
  projected in fp32.

PRIV tokens enter only the value module. The policy path never receives
them: `E2ENet.policy` reads the honest fields of a batch only, so changing
PRIV cannot change a logit (T14 at the net level, plan §3.4).

The valve (`NetConfig.valve`, plan §5): when set, the value module reads the
trunk through a stop-gradient and the PRIV token embeddings are detached, so
the value loss no longer trains the trunk, the ID table or the stems. It's a
config flag, not a buffer: it changes only at a checkpoint, and a Python bool
keeps the compiled graph static.

Precision (plan §6): weights stay fp32; `bf16=True` runs the trunk, the
numeric stem and the head hidden layers under bf16 autocast, while the
embedding lookups and pools, the policy output projection, the log-softmax,
the value norm and the value projection run in fp32. `NetConfig.fp32_heads`
also runs the policy heads' hidden layers in fp32 (X-B's fallback, plan §9).

Token, option, PRIV and pool counts pad to fixed buckets (`LEN_BUCKETS`,
`OPT_BUCKETS`, `PRIV_BUCKETS`, `POOL_BUCKETS`), so `torch.compile` sees a
small set of static shapes. `to_batch` itself is not meant to be compiled.

Serving can skip the padding (#1097 S1-pad, the learner's `--packed-serve`):
`to_packed` lays the rows' visible tokens back to back (`PackedBatch`) and
`E2ENet.forward_packed` runs the per-token layers on them alone, attention per
row through `Varlen` (flash varlen under bf16 on CUDA; a padded
`scaled_dot_product_attention` in fp32), the heads on the real options only,
and the value block's query and MLP at RUNCTX only. Same outputs up to float
rounding; eager, inference only. The update keeps the padded path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import contracts as C

#: Checkpoint format tag (`save_checkpoint`).
FORMAT = "ppo.e2e.net/2"
#: Token-count buckets (contracts: 32-256).
LEN_BUCKETS: tuple[int, ...] = tuple(C.TOKEN_BUCKETS)
#: Merged-option buckets; contracts assert at most 64 options.
OPT_BUCKETS: tuple[int, ...] = (8, 16, 32, 64)
MAX_OPTIONS = C.MAX_OPTIONS
#: PRIV-token buckets; contracts allow at most 16 PRIV tokens.
PRIV_BUCKETS: tuple[int, ...] = (C.MAX_PRIV,)
#: Pooled tokens per row: honest rows carry the 5 summary tokens.
POOL_BUCKETS: tuple[int, ...] = (C.N_SUMMARY_TOKENS, 8, 16)
#: Pooled PRIV tokens per row (0 when no PRIV token owns a pool).
PPOOL_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Ragged axes of a row batch and their per-row count field.
AXES: dict[str, str] = {
    "tok": "n_tok",
    "mem": "n_mem",
    "cls": "n_classes",
    "priv": "n_priv",
    "pool": "n_pool",
    "ppool": "n_ppool",
}
_ID_NAMES = ("X_ID_MAIN", "X_ID_MOD", "X_ID_REL", "X_ID_AUX")
#: ID-table registry blocks whose rows start at zero (SCHEMA.md section 2.12).
#: Note: BOSS (`X_ID_MAIN`) and PRIV_RUN read encounter ids from launch, so
#: those start uninformative and train from zero.
ZERO_ID_REGISTRIES: tuple[str, ...] = ("encounter", "move")
#: The fill event that makes each zero block live: WP-ENG-1's encounter id (RUNCTX
#: `X_ID_AUX`) and `peek_intents` move ids (PRIV_ENEMY). A net built for a launch whose
#: rows already carry the event (`E2ENet(filled=...)`) inits the block like live rows.
ZERO_ID_FILL: Mapping[str, str] = {"encounter": "WP-ENG-1", "move": "WP-ENG-1"}


# ── the input layout (from contracts) ─────────────────────────────────────


@dataclass(frozen=True)
class Layout:
    """Which `contracts` columns the student reads, and how.

    `small_*` describe the small categorical columns: their `cat` index,
    table size, absent value and whether they are live (else reserved,
    zero-initialised). The concatenated small table has a shared zero row 0,
    then one block per column at `small_offsets`.

    `filled` are the fill events (`contracts.FILL_EVENTS`) this layout treats as
    live: an input whose status is `reserved:<event>` for one of them counts as
    live (`layout(filled)`), and `prior_live` says the same of the content-prior
    columns ("content-prior"). Only a fresh net's init reads these flags.
    """

    n_cat: int
    n_cont: int
    n_prior: int
    vocab_rows: int
    n_section: int
    n_dtype: int
    n_kind: int
    live_kinds: tuple[int, ...]
    live_dtypes: tuple[int, ...]
    cat_absent: tuple[int, ...]
    id_cols: tuple[int, ...]
    id_live: tuple[bool, ...]
    main_col: int
    small_cols: tuple[int, ...]
    small_sizes: tuple[int, ...]
    small_absent: tuple[int, ...]
    small_live: tuple[bool, ...]
    zero_cont: tuple[int, ...]
    reserved_sections: tuple[int, ...]
    filled: tuple[str, ...] = ()
    prior_live: bool = False

    @property
    def small_offsets(self) -> tuple[int, ...]:
        return tuple(1 + int(x) for x in np.cumsum((0, *self.small_sizes[:-1])))

    @property
    def small_rows(self) -> int:
        return 1 + sum(self.small_sizes)

    @property
    def stem_in(self) -> int:
        return self.n_cont + self.n_prior

    @property
    def zero_stem_cols(self) -> tuple[int, ...]:
        """Stem inputs that start with zero weights: reserved `cont` columns
        and the prior columns (the "content-prior" fill event) unless filled."""
        prior = () if self.prior_live else tuple(range(self.n_cont, self.n_cont + self.n_prior))
        return self.zero_cont + prior


def fill_key(filled: Iterable[str] = ()) -> tuple[str, ...]:
    """`filled` as a sorted tuple of fill events; `contracts.SchemaError` on an
    event outside `contracts.FILL_EVENTS`."""
    key = tuple(sorted({str(e) for e in filled}))
    bad = sorted(set(key) - set(C.FILL_EVENTS))
    if bad:
        raise C.SchemaError(f"unknown fill events {bad}")
    return key


def _live(status: str, filled: tuple[str, ...]) -> bool:
    """Live at init: live, or reserved for a fill event the launch's rows carry."""
    return status == "live" or (
        status.startswith("reserved:") and status[len("reserved:") :] in filled
    )


def layout(filled: Iterable[str] = ()) -> Layout:
    """The `Layout` of this checkout's `contracts`; with `filled`, the layout of
    a launch whose rows already carry those fill events (`E2ENet(filled=...)`)."""
    return _layout(fill_key(filled))


@lru_cache(maxsize=8)
def _layout(filled: tuple[str, ...]) -> Layout:
    slot = {s: i for i, s in enumerate(C.SCTX_SLOTS)}
    small: list[tuple[int, int, int, bool]] = [
        (slot[s], C.SCTX_TABLE_SIZES[s], C.SCTX_PAD[slot[s]], True) for s in C.SCTX_STUDENT_SLOTS
    ]
    for c in C.X_CAT:
        if c.name in _ID_NAMES:
            continue
        if c.size >= C.VOCAB_ROWS:
            raise AssertionError(f"{c.name}: a {c.size}-row column is not a small column")
        small.append((C.XC[c.name], c.size, 0, _live(c.status, filled)))
    x_cat = {c.name: c for c in C.X_CAT}
    return Layout(
        n_cat=C.N_CAT,
        n_cont=C.N_CONT,
        n_prior=C.N_PRIOR,
        vocab_rows=C.VOCAB_ROWS,
        n_section=C.N_SECTION,
        n_dtype=C.N_DTYPE,
        n_kind=C.N_ACTION_KIND,
        live_kinds=tuple(sorted(k.id for k in C.ACTION_KINDS if k.id and _live(k.status, filled))),
        live_dtypes=tuple(
            sorted(d.id for d in C.DECISION_TYPES if d.id and _live(d.status, filled))
        ),
        cat_absent=tuple(int(x) for x in C.SCTX_PAD) + (0,) * len(C.X_CAT),
        id_cols=tuple(C.XC[n] for n in _ID_NAMES),
        id_live=tuple(_live(x_cat[n].status, filled) for n in _ID_NAMES),
        main_col=C.XC["X_ID_MAIN"],
        small_cols=tuple(s[0] for s in small),
        small_sizes=tuple(s[1] for s in small),
        small_absent=tuple(s[2] for s in small),
        small_live=tuple(s[3] for s in small),
        zero_cont=tuple(C.XF[c.name] for c in C.X_CONT if not _live(c.status, filled)),
        reserved_sections=tuple(sorted(C.RESERVED_SECTIONS)),
        filled=filled,
        prior_live="content-prior" in filled,
    )


def zero_id_rows(filled: Iterable[str] = ()) -> tuple[int, ...]:
    """ID-table rows that start at zero: the `ZERO_ID_REGISTRIES` blocks of
    this checkout's content-ID table (`ppo.e2e.vocab`), less those whose fill
    event (`ZERO_ID_FILL`) is in `filled`."""
    return _zero_id_rows(fill_key(filled))


@lru_cache(maxsize=8)
def _zero_id_rows(filled: tuple[str, ...]) -> tuple[int, ...]:
    from . import vocab as V

    v = V.load()
    return tuple(i for r in ZERO_ID_REGISTRIES if ZERO_ID_FILL[r] not in filled for i in v.block(r))


def current_vocab_sha256() -> str:
    """This checkout's content-ID table sha256 (the `vocab_sha256` stamp)."""
    from . import vocab as V

    return V.load().sha256


# ── configuration ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NetConfig:
    """`E2ENet` sizes. The defaults are plan §5's 6 x 256 net; `n_layers=4`
    is the B-T0 bench variant. Table sizes and column sets come from
    `contracts` (`layout()`); a checkpoint records the schema sha256."""

    dim: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn: int = 704
    num_hidden: int = 512
    head_hidden: int = 256
    value_layers: int = 1
    emb_std: float = 0.4
    valve: bool = False
    #: Run the policy heads' hidden layers in fp32 too (X-B's fallback, plan §9).
    fp32_heads: bool = False

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError(f"dim {self.dim} is not a multiple of n_heads {self.n_heads}")

    def to_dict(self) -> dict[str, Any]:
        """Plain types only, so `torch.load(weights_only=True)` reads it back."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NetConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown NetConfig fields {unknown}")
        return cls(**dict(d))


# ── row batches (numpy) ───────────────────────────────────────────────────


def _offsets(n: np.ndarray) -> np.ndarray:
    o = np.zeros(len(n) + 1, np.int64)
    np.cumsum(np.asarray(n, np.int64), out=o[1:])
    return o


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Flat indices of the concatenated ranges `[starts[i], starts[i] + lens[i])`."""
    lens = np.asarray(lens, np.int64)
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    cum = np.cumsum(lens) - lens
    return np.repeat(np.asarray(starts, np.int64) - cum, lens) + np.arange(total)


def select_rows(
    rb: Mapping[str, np.ndarray], idx: Sequence[int] | np.ndarray
) -> dict[str, np.ndarray]:
    """The rows `idx` of a row batch, as a new row batch (copies).

    Within-row indices (`mem_tok`, `cls_tok`, `pool_tok`, `ppool_tok`,
    `sctx_lo`) are relative to their row, so they carry over unchanged.
    """
    idx = np.asarray(idx, np.int64)
    take = {}
    for axis, o in C.row_offsets(rb).items():
        take[axis] = _ranges(o[idx], o[idx + 1] - o[idx])
    return {
        f.name: (rb[f.name][idx] if f.axis == "row" else rb[f.name][take[f.axis]])
        for f in C.ROW_FIELDS
    }


def concat_rows(batches: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Row batches back to back (the ragged layout concatenates field-wise)."""
    if not batches:
        return C.empty_row_batch()
    return {f.name: np.concatenate([b[f.name] for b in batches]) for f in C.ROW_FIELDS}


# ── padded batches (torch) ────────────────────────────────────────────────


@dataclass
class TokenBatch:
    """A padded batch of decisions (torch tensors on one device).

    Honest fields (the policy reads only these): `cat` `[B, T, N_CAT]`
    int64, `cont` `[B, T, N_CONT]` fp32, `sec` `[B, T]`, `pad` `[B, T]`
    (True = padding), the pools `pool_t` `[B, J]` (token position of pooled
    token j), `pool_ok` `[B, J]` and `pool_cnt` `[B, J, VOCAB_ROWS]` fp32
    (entry weights by vocabulary id), the classes `cls_pos` `[B, A]` (token
    position of each class representative), `cls_kind` `[B, A]` (its action
    kind) and `opt_mask` `[B, A]`, `dtype` `[B]` and `is_combat` `[B]`.
    Token 0 of a row is RUNCTX. Value-only fields: `pcat`, `pcont`, `psec`,
    `ppad` `[B, P, ...]` and the PRIV pools `ppool_t`, `ppool_ok`,
    `ppool_cnt`. `n_real` is the number of real rows; rows past it pad the
    batch and have one token and one option, so nothing in them is NaN.
    """

    cat: torch.Tensor
    cont: torch.Tensor
    sec: torch.Tensor
    pad: torch.Tensor
    pool_t: torch.Tensor
    pool_ok: torch.Tensor
    pool_cnt: torch.Tensor
    cls_pos: torch.Tensor
    cls_kind: torch.Tensor
    opt_mask: torch.Tensor
    dtype: torch.Tensor
    is_combat: torch.Tensor
    pcat: torch.Tensor
    pcont: torch.Tensor
    psec: torch.Tensor
    ppad: torch.Tensor
    ppool_t: torch.Tensor
    ppool_ok: torch.Tensor
    ppool_cnt: torch.Tensor
    n_real: int

    def to(self, device: str | torch.device) -> TokenBatch:
        kw = {
            f.name: getattr(self, f.name).to(device, non_blocking=True)
            for f in dataclasses.fields(self)
            if f.name != "n_real"
        }
        return TokenBatch(**kw, n_real=self.n_real)

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        """`(B, T, A, P, J, Jp)`: the static shape `torch.compile` sees."""
        return (
            int(self.cat.shape[0]),
            int(self.cat.shape[1]),
            int(self.cls_pos.shape[1]),
            int(self.pcat.shape[1]),
            int(self.pool_t.shape[1]),
            int(self.ppool_t.shape[1]),
        )


def bucket(n: int, buckets: Sequence[int] | None) -> int:
    """The smallest bucket >= `n`; past the last bucket (or without buckets),
    `n` itself. Raise if `n` exceeds a bucketed axis's cap elsewhere."""
    if buckets:
        for b in buckets:
            if n <= b:
                return int(b)
    return int(n)


def _pool_slots(
    owner: np.ndarray, row_of: np.ndarray, n_rows: int, buckets: Sequence[int] | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool slots: `(slot of each entry, (row, j, owner) of each pooled
    token, J)`. Pooled tokens of a row get slots 0.. in token order."""
    if len(owner) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.int64), bucket(0, buckets)
    u, inv = np.unique(owner, return_inverse=True)
    urow = row_of[u]
    rank = np.arange(len(u)) - np.searchsorted(urow, urow, side="left")
    j = bucket(int(rank.max()) + 1, buckets)
    slot = urow[inv] * j + rank[inv]
    return slot, np.stack([urow, rank, u], 1), j


@lru_cache(maxsize=1)
def _read_cols() -> tuple[np.ndarray, np.ndarray]:
    """The `cat` columns the net looks up, and each one's table size."""
    L = layout()
    cols = np.asarray(L.id_cols + L.small_cols, np.int64)
    hi = np.asarray((L.vocab_rows,) * len(L.id_cols) + L.small_sizes, np.int64)
    return cols, hi


def _check_lookups(cat: Any, sec: Any, what: str) -> None:
    """Raise unless every id the net looks up in host arrays `cat` and `sec`
    lies inside its table (an id past it would read another column's rows,
    or fail as a device-side assert on CUDA). Device tensors are skipped."""
    L = layout()
    if isinstance(cat, np.ndarray) and len(cat):
        cols, hi = _read_cols()
        v = cat[:, cols]
        out = (v.min(0) < 0) | (v.max(0) >= hi)
        if out.any():
            bad = [int(c) for c in cols[out]]
            raise ValueError(f"to_batch: {what} ids outside their tables in columns {bad}")
    if isinstance(sec, np.ndarray) and len(sec) and not (0 <= sec.min() <= sec.max() < L.n_section):
        raise ValueError(f"to_batch: {what} section outside 0..{L.n_section - 1}")


@dataclass
class _Ix:
    """`_index`'s host index arrays of a row batch (numpy)."""

    b0: int
    flags: np.ndarray  # tok_flags of every token
    vis: np.ndarray  # [n_tok total] student-visible
    cv: np.ndarray  # cumsum(vis): the packed index of visible token g is cv[g] - 1
    row_of_tok: np.ndarray
    newpos: np.ndarray  # each token's position among its row's visible tokens
    n_vis: np.ndarray  # [B] visible tokens per row
    nc: np.ndarray  # [B] merged classes per row
    row_of_cls: np.ndarray
    col_of_cls: np.ndarray
    g_cls: np.ndarray  # global token index of each class representative
    ck: np.ndarray  # each class's action kind
    g_pt: np.ndarray  # global token index of each pool entry's pooled token
    npv: np.ndarray  # [B] PRIV tokens per row
    pv_off: np.ndarray
    row_of_pv: np.ndarray
    col_of_pv: np.ndarray
    g_ppt: np.ndarray  # global PRIV index of each PRIV pool entry's pooled token


def _index(rb: Mapping[str, np.ndarray]) -> _Ix:
    """The host index arithmetic `to_batch` and `to_packed` share, with the
    cheap structural checks (see `to_batch`)."""
    L = layout()
    nt = np.asarray(rb["n_tok"], np.int64)
    b0 = len(nt)
    if b0 == 0:
        raise ValueError("to_batch: no rows")
    flags = np.asarray(rb["tok_flags"])
    vis = C.student_mask(rb)
    t_off = _offsets(nt)
    row_of_tok = np.repeat(np.arange(b0), nt)
    cv = np.cumsum(vis)
    before = np.concatenate([[0], cv])[t_off[:-1]]  # visible tokens before each row
    newpos = cv - 1 - before[row_of_tok]
    n_vis = np.concatenate([[0], cv])[t_off[1:]] - before
    if not vis[t_off[:-1]].all():
        raise ValueError("to_batch: token 0 (RUNCTX) of a row is hidden")

    nc = np.asarray(rb["n_classes"], np.int64)
    if int(nc.min()) < 1 or int(nc.max()) > MAX_OPTIONS:
        raise ValueError(f"to_batch: n_classes outside 1..{MAX_OPTIONS}")
    row_of_cls = np.repeat(np.arange(b0), nc)
    col_of_cls = np.arange(len(row_of_cls)) - np.repeat(_offsets(nc)[:-1], nc)
    g_cls = t_off[row_of_cls] + np.asarray(rb["cls_tok"], np.int64)
    if (np.asarray(rb["cls_tok"], np.int64) >= nt[row_of_cls]).any():
        raise ValueError("to_batch: a class token lies outside its row")
    if not vis[g_cls].all():
        raise ValueError("to_batch: a class representative is hidden")
    ck = np.asarray(rb["cls_kind"], np.int64)
    if len(ck) and int(ck.max()) >= L.n_kind:
        raise ValueError(f"to_batch: cls_kind outside 0..{L.n_kind - 1}")
    _check_lookups(rb["cat"], rb["sec"], "token")
    _check_lookups(rb["pcat"], rb["psec"], "PRIV token")

    npool = np.asarray(rb["n_pool"], np.int64)
    row_of_pe = np.repeat(np.arange(b0), npool)
    ptok = np.asarray(rb["pool_tok"], np.int64)
    if (ptok >= nt[row_of_pe]).any():
        raise ValueError("to_batch: a pool entry's token lies outside its row")
    g_pt = t_off[row_of_pe] + ptok
    if len(g_pt) and not (vis[g_pt] & ((flags[g_pt] & C.TF_POOLED) != 0)).all():
        raise ValueError("to_batch: a pool entry's token is hidden or not TF_POOLED")
    for name in ("pool_vid", "ppool_vid"):
        vid = np.asarray(rb[name], np.int64)
        if len(vid) and (int(vid.min()) < 1 or int(vid.max()) >= L.vocab_rows):
            raise ValueError(f"to_batch: {name} outside 1..{L.vocab_rows - 1} (0 is NONE)")

    npv = np.asarray(rb["n_priv"], np.int64)
    if int(npv.max(initial=0)) > C.MAX_PRIV:
        raise ValueError(f"to_batch: more than {C.MAX_PRIV} PRIV tokens")
    pv_off = _offsets(npv)
    row_of_pv = np.repeat(np.arange(b0), npv)
    col_of_pv = np.arange(len(row_of_pv)) - pv_off[row_of_pv]
    nppool = np.asarray(rb["n_ppool"], np.int64)
    row_of_ppe = np.repeat(np.arange(b0), nppool)
    pptok = np.asarray(rb["ppool_tok"], np.int64)
    if (pptok >= npv[row_of_ppe]).any():
        raise ValueError("to_batch: a PRIV pool entry's token lies outside its row")
    return _Ix(
        b0=b0,
        flags=flags,
        vis=vis,
        cv=cv,
        row_of_tok=row_of_tok,
        newpos=newpos,
        n_vis=n_vis,
        nc=nc,
        row_of_cls=row_of_cls,
        col_of_cls=col_of_cls,
        g_cls=g_cls,
        ck=ck,
        g_pt=g_pt,
        npv=npv,
        pv_off=pv_off,
        row_of_pv=row_of_pv,
        col_of_pv=col_of_pv,
        g_ppt=pv_off[row_of_ppe] + pptok,
    )


def to_batch(
    rb: Mapping[str, np.ndarray],
    device: str | torch.device = "cpu",
    *,
    len_buckets: Sequence[int] | None = LEN_BUCKETS,
    opt_buckets: Sequence[int] | None = OPT_BUCKETS,
    priv_buckets: Sequence[int] | None = PRIV_BUCKETS,
    pool_buckets: Sequence[int] | None = POOL_BUCKETS,
    ppool_buckets: Sequence[int] | None = PPOOL_BUCKETS,
    n_rows: int | None = None,
    check: bool = False,
    pin: bool = False,
) -> TokenBatch:
    """A row batch (`contracts.ROW_FIELDS`, numpy) as a padded `TokenBatch`.

    The index arithmetic runs in numpy on the host; the flat token arrays
    are uploaded once and their student-visible tokens scattered into the
    padded tensors on `device` (every write has its own destination, so the
    result is deterministic). `cat`, `cont`, `sec`, `pcat`, `pcont` and
    `psec` may already be torch tensors on `device` (a server that decodes
    rows there); every other field must be numpy. Hidden tokens are dropped
    and the class and pool token indices remapped. Each axis pads to its
    bucket (`None` = the batch maximum). With `n_rows`, the batch pads to that many rows with inert
    rows. `check=True` runs `contracts.validate_row_batch` first (a per-row
    Python loop). `pin=True` stages host arrays in pinned memory on a CUDA
    device, so an upload never waits for the kernels already queued (a
    pageable copy of more than a few MB does; same values either way). The
    cheap structural checks always run: token, class and
    pool indices inside their rows, RUNCTX and class representatives
    visible, pooled tokens flagged, and every id the net looks up (host
    arrays only) inside its table.
    """
    if check:
        C.validate_row_batch(rb)
    L = layout()
    dev = torch.device(device)
    ix = _index(rb)
    b0, vis, newpos = ix.b0, ix.vis, ix.newpos
    row_of_tok, row_of_cls, col_of_cls = ix.row_of_tok, ix.row_of_cls, ix.col_of_cls
    b = max(b0, int(n_rows or 0))
    t = bucket(int(ix.n_vis.max()), len_buckets)
    a = bucket(int(ix.nc.max()), opt_buckets)
    cls_pos = np.zeros((b, a), np.int64)
    cls_kind = np.zeros((b, a), np.int64)
    opt_mask = np.zeros((b, a), bool)
    cls_pos[row_of_cls, col_of_cls] = newpos[ix.g_cls]
    cls_kind[row_of_cls, col_of_cls] = ix.ck
    opt_mask[row_of_cls, col_of_cls] = True
    opt_mask[b0:, 0] = True

    slot, pooled, j = _pool_slots(ix.g_pt, row_of_tok, b, pool_buckets)
    npv = ix.npv
    p = bucket(max(int(npv.max(initial=0)), 1), priv_buckets)
    row_of_pv, col_of_pv = ix.row_of_pv, ix.col_of_pv
    pslot, ppooled, jp = _pool_slots(ix.g_ppt, row_of_pv, b, ppool_buckets)

    pinned = pin and dev.type == "cuda"

    def up(x: np.ndarray | torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.to(dev, non_blocking=True)
        else:
            h = torch.from_numpy(np.ascontiguousarray(x))
            y = (h.pin_memory() if pinned else h).to(dev, non_blocking=True)
        return y if dtype is None else y.to(dtype)

    long = torch.int64
    absent = torch.tensor(L.cat_absent, dtype=long).to(dev, non_blocking=True)  # no stream sync

    # Honest tokens: upload whole, select the visible ones on the device.
    dst = up(row_of_tok[vis] * t + newpos[vis])
    src = slice(None) if bool(vis.all()) else up(np.flatnonzero(vis))
    cat = absent.expand(b * t, L.n_cat).clone()
    cat[dst] = up(rb["cat"])[src].to(long)
    cont = torch.zeros((b * t, L.n_cont), dtype=torch.float32, device=dev)
    cont[dst] = up(rb["cont"])[src].float()
    sec = torch.zeros(b * t, dtype=long, device=dev)
    sec[dst] = up(rb["sec"])[src].to(long)
    pad = torch.ones(b * t, dtype=torch.bool, device=dev)
    pad[dst] = False
    pad = pad.view(b, t)
    pad[b0:, 0] = False

    pool_t = np.zeros((b, j), np.int64)
    pool_ok = np.zeros((b, j), bool)
    pool_t[pooled[:, 0], pooled[:, 1]] = newpos[pooled[:, 2]]
    pool_ok[pooled[:, 0], pooled[:, 1]] = True
    pool_cnt = torch.zeros((b * j, L.vocab_rows), dtype=torch.float32, device=dev)
    if len(slot):
        pool_cnt.index_put_(
            (up(slot), up(np.asarray(rb["pool_vid"]), long)),
            up(np.asarray(rb["pool_w"]), torch.float32),
            accumulate=True,
        )

    # PRIV tokens.
    pdst = up(row_of_pv * p + col_of_pv)
    pcat = absent.expand(b * p, L.n_cat).clone()
    pcont = torch.zeros((b * p, L.n_cont), dtype=torch.float32, device=dev)
    psec = torch.zeros(b * p, dtype=long, device=dev)
    ppad = torch.ones(b * p, dtype=torch.bool, device=dev)
    if len(row_of_pv):
        pcat[pdst] = up(rb["pcat"], long)
        pcont[pdst] = up(rb["pcont"], torch.float32)
        psec[pdst] = up(rb["psec"], long)
        ppad[pdst] = False
    ppool_t = np.zeros((b, jp), np.int64)
    ppool_ok = np.zeros((b, jp), bool)
    if len(ppooled):
        ppool_t[ppooled[:, 0], ppooled[:, 1]] = col_of_pv[ppooled[:, 2]]
        ppool_ok[ppooled[:, 0], ppooled[:, 1]] = True
    ppool_cnt = torch.zeros((b * jp, L.vocab_rows), dtype=torch.float32, device=dev)
    if len(pslot):
        ppool_cnt.index_put_(
            (up(pslot), up(np.asarray(rb["ppool_vid"]), long)),
            up(np.asarray(rb["ppool_w"]), torch.float32),
            accumulate=True,
        )

    dtype = np.zeros(b, np.int64)
    dtype[:b0] = np.asarray(rb["dtype"], np.int64)
    is_combat = np.zeros(b, bool)
    is_combat[:b0] = np.asarray(rb["is_combat"], bool)
    return TokenBatch(
        cat=cat.view(b, t, L.n_cat),
        cont=cont.view(b, t, L.n_cont),
        sec=sec.view(b, t),
        pad=pad,
        pool_t=up(pool_t),
        pool_ok=up(pool_ok),
        pool_cnt=pool_cnt.view(b, j, L.vocab_rows),
        cls_pos=up(cls_pos),
        cls_kind=up(cls_kind),
        opt_mask=up(opt_mask),
        dtype=up(dtype),
        is_combat=up(is_combat),
        pcat=pcat.view(b, p, L.n_cat),
        pcont=pcont.view(b, p, L.n_cont),
        psec=psec.view(b, p),
        ppad=ppad.view(b, p),
        ppool_t=up(ppool_t),
        ppool_ok=up(ppool_ok),
        ppool_cnt=ppool_cnt.view(b, jp, L.vocab_rows),
        n_real=b0,
    )


# ── packed batches (torch; the serving path) ──────────────────────────────


@dataclass
class PackedBatch:
    """A pad-free batch of decisions for `E2ENet.forward_packed` (serving).

    The student-visible tokens of all rows sit back to back: `cat` `[N, N_CAT]`
    int64, `cont` `[N, N_CONT]` fp32, `sec` `[N]`; row `i` holds tokens
    `cu[i]:cu[i + 1]` (`cu` int32 `[B + 1]`, its longest row `max_len`) and
    its RUNCTX is token `cu[i]`. Pooled tokens: `pool_tok` `[J]` (packed
    index) and `pool_cnt` `[J, VOCAB_ROWS]`. Classes, row-major: `cls_tok`
    (packed index of the representative), `cls_kind`, `cls_row` and
    `cls_dst` (`row * A + column`, the entry of the `[B, A]` logits), with `A`
    from `OPT_BUCKETS` like `to_batch`'s. `dtype`, `is_combat` `[B]`.

    Value-only fields: the PRIV tokens `pcat`, `pcont`, `psec` `[Np, ...]`
    back to back and their pools `ppool_tok`, `ppool_cnt`; the value module's
    sequence of each row is its honest tokens then its PRIV tokens, at
    `h_dst` and `p_dst` of a `[N + Np]` matrix with offsets `vcu` (longest
    row `vmax_len`). No policy input reads a PRIV field.
    """

    cat: torch.Tensor
    cont: torch.Tensor
    sec: torch.Tensor
    cu: torch.Tensor
    max_len: int
    pool_tok: torch.Tensor
    pool_cnt: torch.Tensor
    cls_tok: torch.Tensor
    cls_kind: torch.Tensor
    cls_row: torch.Tensor
    cls_dst: torch.Tensor
    n_opt: int  # A: the logits' width
    dtype: torch.Tensor
    is_combat: torch.Tensor
    pcat: torch.Tensor
    pcont: torch.Tensor
    psec: torch.Tensor
    ppool_tok: torch.Tensor
    ppool_cnt: torch.Tensor
    vcu: torch.Tensor
    vmax_len: int
    h_dst: torch.Tensor
    p_dst: torch.Tensor
    n_real: int


def _pooled(
    g: np.ndarray, vid: Any, w: Any, n_vocab: int, up: Callable[..., torch.Tensor], dev: Any
) -> tuple[np.ndarray, torch.Tensor]:
    """Pooled tokens of pool entries owned by (global) tokens `g`: `(the
    owners in order, their [J, n_vocab] entry weights)`."""
    u, inv = np.unique(g, return_inverse=True)
    cnt = torch.zeros((len(u), n_vocab), dtype=torch.float32, device=dev)
    if len(g):
        cnt.index_put_(
            (up(inv.astype(np.int64)), up(np.asarray(vid), torch.int64)),
            up(np.asarray(w), torch.float32),
            accumulate=True,
        )
    return u, cnt


def to_packed(
    rb: Mapping[str, np.ndarray],
    device: str | torch.device = "cpu",
    *,
    opt_buckets: Sequence[int] | None = OPT_BUCKETS,
    pin: bool = False,
) -> PackedBatch:
    """A row batch (`contracts.ROW_FIELDS`, numpy) as a `PackedBatch`: the
    same rows, checks and option width as `to_batch`, with no token, PRIV or
    pool padding. `cat`, `cont`, `sec`, `pcat`, `pcont` and `psec` may be
    torch tensors on `device`; `pin` as in `to_batch`."""
    L = layout()
    dev = torch.device(device)
    ix = _index(rb)
    b, vis, cv = ix.b0, ix.vis, ix.cv
    a = bucket(int(ix.nc.max()), opt_buckets)
    pinned = pin and dev.type == "cuda"

    def up(x: np.ndarray | torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.to(dev, non_blocking=True)
        else:
            h = torch.from_numpy(np.ascontiguousarray(x))
            y = (h.pin_memory() if pinned else h).to(dev, non_blocking=True)
        return y if dtype is None else y.to(dtype)

    long = torch.int64
    n_vis, npv = ix.n_vis, ix.npv
    cu = _offsets(n_vis)
    vcu = _offsets(n_vis + npv)
    # Small index arrays go up as one int64 block.
    pool_u, pool_cnt = _pooled(ix.g_pt, rb["pool_vid"], rb["pool_w"], L.vocab_rows, up, dev)
    ppool_u, ppool_cnt = _pooled(ix.g_ppt, rb["ppool_vid"], rb["ppool_w"], L.vocab_rows, up, dev)
    vis_row = ix.row_of_tok[vis]
    parts = {
        "cu": cu,
        "vcu": vcu,
        "pool_tok": cv[pool_u] - 1,
        "ppool_tok": ppool_u,
        "cls_tok": cv[ix.g_cls] - 1,
        "cls_kind": ix.ck,
        "cls_row": ix.row_of_cls,
        "cls_dst": ix.row_of_cls * a + ix.col_of_cls,
        "dtype": np.asarray(rb["dtype"], np.int64),
        "is_combat": np.asarray(rb["is_combat"], np.int64),
        "h_dst": vcu[vis_row] + ix.newpos[vis],
        "p_dst": vcu[ix.row_of_pv] + n_vis[ix.row_of_pv] + ix.col_of_pv,
    }
    sizes = [len(v) for v in parts.values()]
    blk = up(np.concatenate([np.asarray(v, np.int64) for v in parts.values()]))
    got = dict(zip(parts, torch.split(blk, sizes), strict=True))

    src = slice(None) if bool(vis.all()) else up(np.flatnonzero(vis))
    has_priv = bool(len(ix.row_of_pv))
    return PackedBatch(
        cat=up(rb["cat"])[src].to(long),
        cont=up(rb["cont"])[src].float(),
        sec=up(rb["sec"])[src].to(long),
        cu=got["cu"].to(torch.int32),
        max_len=int(n_vis.max()),
        pool_tok=got["pool_tok"],
        pool_cnt=pool_cnt,
        cls_tok=got["cls_tok"],
        cls_kind=got["cls_kind"],
        cls_row=got["cls_row"],
        cls_dst=got["cls_dst"],
        n_opt=a,
        dtype=got["dtype"],
        is_combat=got["is_combat"].bool(),
        pcat=up(rb["pcat"], long)
        if has_priv
        else torch.zeros((0, L.n_cat), dtype=long, device=dev),
        pcont=up(rb["pcont"], torch.float32)
        if has_priv
        else torch.zeros((0, L.n_cont), dtype=torch.float32, device=dev),
        psec=up(rb["psec"], long) if has_priv else torch.zeros(0, dtype=long, device=dev),
        ppool_tok=got["ppool_tok"],
        ppool_cnt=ppool_cnt,
        vcu=got["vcu"].to(torch.int32),
        vmax_len=int((n_vis + npv).max()),
        h_dst=got["h_dst"],
        p_dst=got["p_dst"],
        n_real=b,
    )


class Varlen:
    """Attention over rows packed back to back (`cu_q`, `cu_k`: int32 row
    offsets of the queries and the keys, `max_q`, `max_k`: the longest rows).

    `mode="varlen"` runs `torch.nn.attention.varlen.varlen_attn` (flash
    attention: CUDA, bf16/fp16 only); `"pad"` scatters q, k and v into a
    padded batch for `scaled_dot_product_attention` with a key mask and
    gathers the outputs back (any device and dtype, the fp32 path); `"auto"`
    picks varlen whenever it can run on the inputs. No host sync either way.
    """

    def __init__(
        self,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        n_q: int,
        n_k: int,
        mode: str = "auto",
    ) -> None:
        if mode not in ("auto", "varlen", "pad"):
            raise ValueError(f"attention mode {mode!r}")
        self.cu_q, self.cu_k, self.max_q, self.max_k = cu_q, cu_k, int(max_q), int(max_k)
        self.n_q, self.n_k, self.mode = int(n_q), int(n_k), mode
        self._rc: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _row_col(self, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        if which not in self._rc:
            cu, n = (self.cu_q, self.n_q) if which == "q" else (self.cu_k, self.n_k)
            cu = cu.long()
            b = cu.shape[0] - 1
            row = torch.repeat_interleave(
                torch.arange(b, device=cu.device), cu.diff(), output_size=n
            )
            self._rc[which] = (row, torch.arange(n, device=cu.device) - cu[row])
        return self._rc[which]

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """`q` `[n_q, H, d]`, `k` and `v` `[n_k, H, d]` -> `[n_q, H, d]`."""
        mode = self.mode
        if mode == "auto":
            ok = q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
            mode = "varlen" if ok else "pad"
        if mode == "varlen":
            from torch.nn.attention.varlen import varlen_attn

            return varlen_attn(q, k, v, self.cu_q, self.cu_k, self.max_q, self.max_k)
        b = self.cu_q.shape[0] - 1
        rq, cq = self._row_col("q")
        rk, ck = self._row_col("k")
        qp = q.new_zeros((b, self.max_q, *q.shape[1:]))
        kp = k.new_zeros((b, self.max_k, *k.shape[1:]))
        vp = v.new_zeros((b, self.max_k, *v.shape[1:]))
        qp[rq, cq], kp[rk, ck], vp[rk, ck] = q, k, v
        n_k = self.cu_k.long().diff()
        keep = torch.arange(self.max_k, device=q.device)[None, :] < n_k[:, None]
        ap = F.scaled_dot_product_attention(
            qp.transpose(1, 2),
            kp.transpose(1, 2),
            vp.transpose(1, 2),
            attn_mask=keep[:, None, None, :],
        )
        return ap.transpose(1, 2)[rq, cq]


# ── synthetic rows (tests and benches) ────────────────────────────────────

# S_ctx body sections in #1081's native order (SCHEMA.md section 2.1).
_COMBAT_BODY = ("HAND", "DRAW_UNK", "DISCARD", "EXHAUST", "DRAW_TOP", "DRAW_BOT", "PLAY")
_COMBAT_BODY += ("RELIC", "POTION", "PPOWER", "ENEMY", "EPOWER", "EHIST")


def _token(rng: np.random.Generator, sec: str, **fixed: int) -> tuple[np.ndarray, np.ndarray]:
    """One token of section `sec` with random live (non-reserved) columns."""
    L = layout()
    s = C.SEC[sec]
    live_cat, live_cont = C.SECTION_LIVE[s]
    cat = np.asarray(L.cat_absent, np.int16).copy()
    cont = np.zeros(C.N_CONT, np.float16)
    sizes = [C.SCTX_TABLE_SIZES[x] for x in C.SCTX_SLOTS] + [c.size for c in C.X_CAT]
    status = ["live"] * C.SCTX_N_CAT + [c.status for c in C.X_CAT]
    for col in live_cat:
        if status[col] != "live" or rng.random() < 0.15:
            continue
        hi = sizes[col] - 1 if col < C.SCTX_N_CAT else sizes[col]
        cat[col] = rng.integers(0 if col < C.SCTX_N_CAT else 1, hi)
    reserved = set(layout().zero_cont)
    for col in live_cont:
        if col in reserved:
            continue
        cont[col] = rng.standard_normal() * rng.choice([0.0, 0.3, 3.0, 30.0])
    for name, v in fixed.items():
        cat[C.XC[name]] = v
    return cat, cont


def _live_kinds(dt: int) -> list[int]:
    """The live action kinds a decision type offers (`DECISION_TYPES`)."""
    return [k for k in C.DECISION_TYPES[dt].kinds if k in C.LIVE_ACTION_KINDS]


def _option_section(kind: int) -> str:
    """An option token's section: its kind's (`KIND_SECTION`), else OPT_FIXED."""
    return C.SECTIONS[C.KIND_SECTION.get(kind, C.SEC["OPT_FIXED"])]


def synthetic_row(
    rng: np.random.Generator,
    *,
    combat: bool,
    t: int = 0,
    priv: tuple[int, int] = (0, 16),
    body: tuple[int, int] = (8, 60),
    teacher: bool = False,
    ppool: bool = False,
) -> dict[str, np.ndarray]:
    """One random decision as a one-row batch that `validate_row_batch` and
    `check_compact_lossless` accept, in SCHEMA.md section 2.1's token order:
    RUNCTX, the 5 pooled summary tokens and 1-2 BOSS tokens, then (combat)
    an S_ctx block of STATE, body tokens and one action token per member,
    merged by `contracts.merge_classes` with non-representatives hidden, or
    (OOC) body tokens and one option token per class. Decision types, their
    action kinds and the option sections come from `contracts`
    (`DECISION_TYPES`, `KIND_SECTION`). Reserved columns stay absent."""
    toks: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    def add(sec: str, flags: int = 0, **fixed: int) -> int:
        cat, cont = _token(rng, sec, **fixed)
        toks.append((sec, flags, cat, cont))
        return len(toks) - 1

    if combat:
        dname = "COMBAT_TURN" if rng.random() < 0.85 else "COMBAT_CHOICE"
    else:
        ooc = sorted(C.DECISION_TYPES[d].name for d in C.LIVE_DTYPES - C.COMBAT_DTYPES)
        dname = str(rng.choice(ooc))
    dt = C.DT[dname]
    runctx = add("RUNCTX", X_DTYPE=dt)
    if dname not in ("EVENT", "NEOW"):
        toks[runctx][2][C.XC["X_ID_MAIN"]] = 0  # the event id: EVENT and NEOW rows only
    pool_tok: list[int] = []
    pool_vid: list[int] = []
    pool_w: list[float] = []
    for s in C.SUMMARY_SECTIONS:
        k = add(C.SECTIONS[s], C.TF_POOLED)
        n = int(rng.integers(0, 16))
        vids = rng.choice(np.arange(1, C.VOCAB_ROWS), n, replace=False)
        pool_tok += [k] * n
        pool_vid += [int(v) for v in vids]
        pool_w += [float(w) for w in rng.integers(1, 5, n)]
    for _ in range(int(rng.integers(1, C.MAX_BOSS_TOKENS + 1))):
        add("BOSS")
    n_body = int(rng.integers(body[0], body[1] + 1))
    sctx_lo = n_sctx = 0
    kinds_dt = _live_kinds(dt)
    if combat:
        n_cls = int(rng.integers(2, 13))
        keys = [str(i % n_cls) for i in range(n_cls)]
        keys += [str(int(rng.integers(0, n_cls))) for _ in range(int(rng.integers(0, 2 * n_cls)))]
        keys = [keys[i] for i in rng.permutation(len(keys))]
        mem_class, reps = C.merge_classes(keys)
        kinds = [int(rng.choice(kinds_dt)) for _ in range(len(reps))]
        sctx_lo = add("STATE", C.TF_TEACHER)
        for sec in sorted(rng.choice(_COMBAT_BODY, n_body), key=_COMBAT_BODY.index):
            add(str(sec), C.TF_TEACHER)
        rep_set = {int(r) for r in reps}
        mem_tok = []
        for m, c in enumerate(mem_class):
            kind = kinds[int(c)]
            flags = C.TF_TEACHER | C.TF_OPTION | (C.TF_REP if m in rep_set else C.TF_HIDDEN)
            mem_tok.append(add(_option_section(kind), flags, X_KIND=kind))
        n_sctx = len(toks) - sctx_lo
        mem_tok_a = np.asarray(mem_tok, np.uint16)
        cls_tok = mem_tok_a[reps]
        cls_kind = np.asarray(kinds, np.uint8)
    else:
        ooc_body = ("DECK", "RELIC", "POTION", "OFFER") + (("MAPNODE",) if dname == "MAP" else ())
        secs = sorted(rng.choice(ooc_body, n_body), key=ooc_body.index)
        n_map = 0
        for sec in secs:
            n_map += sec == "MAPNODE"
            if n_map <= C.MAX_MAPNODES or sec != "MAPNODE":
                add(str(sec))
        n_cls = int(rng.integers(2, 13))
        picks = [int(rng.choice(kinds_dt)) for _ in range(n_cls)]
        if len(kinds_dt) > 1 and len(set(picks)) == 1:  # a second kind when there is one
            picks[-1] = kinds_dt[1] if picks[0] == kinds_dt[0] else kinds_dt[0]
        cls_tok = np.asarray(
            [add(_option_section(k), C.TF_OPTION | C.TF_REP, X_KIND=k) for k in picks], np.uint16
        )
        cls_kind = np.asarray(picks, np.uint8)
        extra = [int(rng.integers(0, n_cls)) for _ in range(int(rng.integers(0, n_cls)))]
        mem_class = np.asarray(sorted(list(range(n_cls)) + extra), np.uint16)
        mem_tok_a = cls_tok[mem_class]
    n_cls = len(cls_tok)
    n_priv = int(rng.integers(priv[0], priv[1] + 1))
    psecs = ["PRIV_RUN"] + ["PRIV_DRAW"] * min(n_priv - 1, C.PRIV_DRAW_K) if n_priv else []
    psecs += ["PRIV_ENEMY"] * (n_priv - len(psecs))
    ptoks = [_token(rng, s) for s in psecs]
    ppool_tok: list[int] = []
    ppool_vid: list[int] = []
    if ppool and n_priv:
        n = int(rng.integers(1, 6))
        ppool_tok = [0] * n
        ppool_vid = [int(v) for v in rng.choice(np.arange(1, C.VOCAB_ROWS), n, replace=False)]
    tsrc = (1 if combat else 2) if teacher else 0
    tprob = rng.dirichlet(np.ones(n_cls)).astype(np.float32) if teacher else None
    row: dict[str, Any] = {
        "run_uid": np.asarray([int(rng.integers(1, 2**62))], np.int64),
        "t": np.asarray([t], np.int32),
        "is_combat": np.asarray([combat]),
        "dtype": np.asarray([dt], np.uint8),
        "reward": np.asarray([0.0], np.float32),
        "done": np.asarray([False]),
        "bootstrap": np.asarray([False]),
        "actor": np.asarray([0], np.uint8),
        "logp_behav": np.asarray([-np.log(n_cls)], np.float32),
        "n_classes": np.asarray([n_cls], np.uint16),
        "action": np.asarray([int(rng.integers(0, n_cls))], np.uint16),
        "iter": np.asarray([0], np.int32),
        "v_behav": np.asarray([0.0], np.float32),
        "cell": np.asarray([C.CELL_NONE], np.uint8),
        "n_tok": np.asarray([len(toks)], np.int32),
        "n_mem": np.asarray([len(mem_tok_a)], np.uint16),
        "n_priv": np.asarray([n_priv], np.int32),
        "n_pool": np.asarray([len(pool_tok)], np.int32),
        "n_ppool": np.asarray([len(ppool_tok)], np.int32),
        "sctx_lo": np.asarray([sctx_lo], np.uint16),
        "n_sctx": np.asarray([n_sctx], np.uint16),
        "t_src": np.asarray([tsrc], np.uint8),
        "cat": np.stack([x[2] for x in toks]).astype(np.int16),
        "cont": np.stack([x[3] for x in toks]).astype(np.float16),
        "sec": np.asarray([C.SEC[x[0]] for x in toks], np.int8),
        "tok_flags": np.asarray([x[1] for x in toks], np.uint8),
        "mem_tok": np.asarray(mem_tok_a, np.uint16),
        "mem_class": np.asarray(mem_class, np.uint16),
        "cls_tok": np.asarray(cls_tok, np.uint16),
        "cls_kind": cls_kind,
        "tprob": tprob if tprob is not None else np.full(n_cls, np.nan, np.float32),
        "pcat": np.stack([x[0] for x in ptoks]).astype(np.int16)
        if ptoks
        else np.zeros((0, C.N_CAT), np.int16),
        "pcont": np.stack([x[1] for x in ptoks]).astype(np.float16)
        if ptoks
        else np.zeros((0, C.N_CONT), np.float16),
        "psec": np.asarray([C.SEC[s] for s in psecs], np.int8),
        "pool_tok": np.asarray(pool_tok, np.uint16),
        "pool_vid": np.asarray(pool_vid, np.uint16),
        "pool_w": np.asarray(pool_w, np.float16),
        "ppool_tok": np.asarray(ppool_tok, np.uint16),
        "ppool_vid": np.asarray(ppool_vid, np.uint16),
        "ppool_w": np.ones(len(ppool_tok), np.float16),
    }
    return row


def synthetic_row_batch(
    n: int, seed: int = 0, *, combat_share: float = 0.7, **kw: Any
) -> dict[str, np.ndarray]:
    """`n` random decisions (`synthetic_row`) as one row batch."""
    rng = np.random.default_rng(seed)
    return concat_rows(
        [synthetic_row(rng, combat=bool(rng.random() < combat_share), t=i, **kw) for i in range(n)]
    )


# ── modules ───────────────────────────────────────────────────────────────


def symlog(x: torch.Tensor) -> torch.Tensor:
    """`sign(x) * log1p(|x|)`: zero stays zero, large values compress."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class NumStem(nn.Module):
    """symlog -> Linear -> SiLU -> Linear, per token, no column normalisation."""

    def __init__(self, n_num: int, hidden: int, dim: int, zero_cols: Sequence[int] = ()) -> None:
        super().__init__()
        self.inp = nn.Linear(n_num, hidden)
        self.out = nn.Linear(hidden, dim)
        self.zero_columns(zero_cols)

    @torch.no_grad()
    def zero_columns(self, cols: Sequence[int]) -> None:
        """Zero the first-layer weights of `cols`. On a column the tokenizer has
        left at zero this changes no output, and the column then starts from
        the function the net already computes when it is filled."""
        if len(cols):
            self.inp.weight[:, list(cols)] = 0.0

    def forward(self, num: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.inp(symlog(num.float()))))


class Block(nn.Module):
    """Pre-norm transformer block: RMSNorm, attention, RMSNorm, SwiGLU; no biases."""

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1 = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.n2 = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, ffn, bias=False)
        self.v = nn.Linear(dim, ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """`x` `[B, T, D]`; `keep` `[B, 1, 1, T]` True = a key to attend to."""
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=keep
        )
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.v(h))

    def forward_packed(self, x: torch.Tensor, attn: Varlen) -> torch.Tensor:
        """`forward` on packed tokens `x` `[N, D]` (rows given by `attn`)."""
        n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, 3, self.heads, d // self.heads).unbind(1)
        x = x + self.out(attn(q, k, v).reshape(n, d))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.v(h))

    def forward_query(self, x: torch.Tensor, at: torch.Tensor, attn: Varlen) -> torch.Tensor:
        """`forward_packed`'s outputs at the tokens `at` only, `[len(at), D]`:
        every token is a key, only those are queries (`attn` has one query per
        row), and the MLP runs on them alone."""
        n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, 3, self.heads, d // self.heads).unbind(1)
        y = x[at] + self.out(attn(q[at], k, v).reshape(len(at), d))
        h = self.n2(y)
        return y + self.w2(F.silu(self.w1(h)) * self.v(h))


class PolicyHead(nn.Module):
    """`[h_option; h_STATE; e_kind] -> hidden -> 1`, as three split matmuls.

    `Linear([a; s; k])` equals `a @ Wa + s @ Ws + k @ Wk`; the state part is
    computed once per row instead of once per option. `e_kind` is per option
    (the option's action kind). The output projection runs in fp32.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.opt = nn.Linear(dim, hidden)
        self.state = nn.Linear(dim, hidden, bias=False)
        self.kind = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, h_opt: torch.Tensor, h_state: torch.Tensor, e_kind: torch.Tensor
    ) -> torch.Tensor:
        z = F.silu(self.opt(h_opt) + self.kind(e_kind) + self.state(h_state)[:, None, :])
        with torch.autocast(z.device.type, enabled=False):
            return F.linear(z.float(), self.out.weight, self.out.bias).squeeze(-1)

    def forward_packed(
        self, h_opt: torch.Tensor, h_state: torch.Tensor, e_kind: torch.Tensor, row: torch.Tensor
    ) -> torch.Tensor:
        """`forward` on packed options: `h_opt`, `e_kind` `[n, D]`, `h_state`
        `[B, D]`, `row` `[n]` each option's row. Returns `[n]`."""
        z = F.silu(self.opt(h_opt) + self.kind(e_kind) + self.state(h_state)[row])
        with torch.autocast(z.device.type, enabled=False):
            return F.linear(z.float(), self.out.weight, self.out.bias).squeeze(-1)


class ValueModule(nn.Module):
    """One block over `[trunk residual stream; PRIV tokens]`, then `V_combat`
    and `V_run` read at RUNCTX (token 0). Both heads start at zero, and the
    norm and projections run in fp32."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_heads, cfg.ffn) for _ in range(cfg.value_layers)
        )
        self.norm = nn.RMSNorm(cfg.dim)
        self.v_combat = nn.Linear(cfg.dim, 1)
        self.v_run = nn.Linear(cfg.dim, 1)
        for head in (self.v_combat, self.v_run):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, x: torch.Tensor, pad: torch.Tensor, e_priv: torch.Tensor, ppad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x, e_priv.to(x.dtype)], 1)
        keep = ~torch.cat([pad, ppad], 1)[:, None, None, :]
        for blk in self.blocks:
            z = blk(z, keep)
        with torch.autocast(z.device.type, enabled=False):
            hs = self.norm(z[:, 0].float())
            vc = F.linear(hs, self.v_combat.weight, self.v_combat.bias).squeeze(-1)
            vr = F.linear(hs, self.v_run.weight, self.v_run.bias).squeeze(-1)
        return vc, vr

    def forward_packed(
        self, x: torch.Tensor, e_priv: torch.Tensor, pb: PackedBatch, mode: str = "auto"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` on a `PackedBatch`: each row's honest residual stream
        (`x` `[N, D]`) then its PRIV tokens (`e_priv` `[Np, D]`), packed. Only
        RUNCTX is read, so the last block runs its query and MLP there alone."""
        z = x.new_empty((x.shape[0] + e_priv.shape[0], x.shape[1]))
        z[pb.h_dst] = x
        z[pb.p_dst] = e_priv.to(x.dtype)
        b = pb.n_real
        nz = int(z.shape[0])
        at = pb.vcu[:-1].long()
        full = Varlen(pb.vcu, pb.vcu, pb.vmax_len, pb.vmax_len, nz, nz, mode)
        for blk in self.blocks[:-1]:
            z = blk.forward_packed(z, full)
        if len(self.blocks):
            one = torch.arange(b + 1, dtype=torch.int32, device=z.device)
            z0 = self.blocks[-1].forward_query(
                z, at, Varlen(one, pb.vcu, 1, pb.vmax_len, b, nz, mode)
            )
        else:
            z0 = z[at]
        with torch.autocast(z0.device.type, enabled=False):
            hs = self.norm(z0.float())
            vc = F.linear(hs, self.v_combat.weight, self.v_combat.bias).squeeze(-1)
            vr = F.linear(hs, self.v_run.weight, self.v_run.bias).squeeze(-1)
        return vc, vr


class NetOut(NamedTuple):
    """`E2ENet.forward`'s outputs, all fp32."""

    logits: torch.Tensor  # [B, A], -inf off opt_mask
    logp: torch.Tensor  # [B, A] log-softmax over merged classes (-inf off mask)
    v_combat: torch.Tensor  # [B]
    v_run: torch.Tensor  # [B]
    value: torch.Tensor  # [B]: v_combat on combat rows, v_run on OOC rows


class E2ENet(nn.Module):
    """The student: shared ID table, stems, trunk, policy heads, value module.

    Parameter names: everything under `value.` is the value module (the
    "value" optimizer group, `ppo.e2e.algo.param_groups`); the rest is the
    trunk side.
    """

    def __init__(
        self, cfg: NetConfig | None = None, *, seed: int | None = None, filled: Iterable[str] = ()
    ) -> None:
        """`filled`: the fill events (`contracts.FILL_EVENTS`) the launch's rows
        already carry (the learner's `--filled`, the stamps' `filled`). Their
        reserved inputs init like live ones (normal init, the honest `X_ID_AUX`
        gain at 1, normal encounter and move ID rows). Only the init differs: a
        loaded state dict overrides every weight, so an event filled after a
        checkpoint exists keeps that checkpoint's zero weights (the identity rule)."""
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        self.layout = layout(filled)
        #: Run the trunk and head hidden layers under bf16 autocast.
        self.bf16 = False
        #: sha256 of the content prior in the `prior` buffer (None = all zero).
        self.prior_sha256: str | None = None
        # Parameters are created on the CPU, so seeding the CPU generator alone
        # fixes the init; `fork_rng` restores the caller's generator after.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.default_generator.manual_seed(int(seed))
            self._build(cfg)

    def _build(self, cfg: NetConfig) -> None:
        L, d = self.layout, cfg.dim
        self.ids = nn.Embedding(L.vocab_rows, d, padding_idx=0)
        live = torch.tensor([1.0 if x else 0.0 for x in L.id_live])
        self.id_gain = nn.Parameter(live[:, None].repeat(1, d))
        self.priv_id_gain = nn.Parameter(torch.ones(len(L.id_cols), d))
        self.pool_gain = nn.Parameter(torch.ones(d))
        self.small = nn.Embedding(L.small_rows, d, padding_idx=0)
        self.sec = nn.Embedding(L.n_section, d)
        self.num = NumStem(L.stem_in, cfg.num_hidden, d, L.zero_stem_cols)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.ffn) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(d)
        self.kind = nn.Embedding(L.n_kind, d)
        self.head_combat = PolicyHead(d, cfg.head_hidden)
        self.head_ooc = PolicyHead(d, cfg.head_hidden)
        self.value = ValueModule(cfg)
        self.register_buffer("prior", torch.zeros(L.vocab_rows, L.n_prior))
        self.register_buffer("id_cols", torch.tensor(L.id_cols), persistent=False)
        self.register_buffer("small_cols", torch.tensor(L.small_cols), persistent=False)
        self.register_buffer("small_absent", torch.tensor(L.small_absent), persistent=False)
        self.register_buffer("small_off", torch.tensor(L.small_offsets), persistent=False)
        with torch.no_grad():
            for emb in (self.ids, self.sec, self.kind, self.small):
                nn.init.normal_(emb.weight, 0.0, cfg.emb_std)
            self.ids.weight[0] = 0.0
            self.small.weight[0] = 0.0
            for off, size, absent, ok in zip(
                L.small_offsets, L.small_sizes, L.small_absent, L.small_live, strict=True
            ):
                if not ok:
                    self.small.weight[off : off + size] = 0.0
                self.small.weight[off + absent] = 0.0  # never read: absent maps to row 0
            self.sec.weight[list(L.reserved_sections)] = 0.0
            dead_kinds = sorted(set(range(L.n_kind)) - set(L.live_kinds))
            self.kind.weight[dead_kinds] = 0.0
            off = dict(zip(L.small_cols, L.small_offsets, strict=True))
            for name, live, n in (
                ("X_KIND", L.live_kinds, L.n_kind),
                ("X_DTYPE", L.live_dtypes, L.n_dtype),
            ):
                dead = sorted(set(range(1, n)) - set(live))
                self.small.weight[[off[C.XC[name]] + k for k in dead]] = 0.0
            self.ids.weight[list(zero_id_rows(L.filled))] = 0.0

    # ── the valve and the prior ──

    def set_valve(self, on: bool) -> None:
        """Set the value-to-trunk stop-gradient (plan §5; only at a checkpoint)."""
        self.cfg = dataclasses.replace(self.cfg, valve=bool(on))

    @torch.no_grad()
    def set_prior(self, table: np.ndarray | torch.Tensor, sha256: str | None = None) -> None:
        """Fill the content prior (`[VOCAB_ROWS, N_PRIOR]`, row 0 zero): the
        "content-prior" fill event. The prior's stem weights start at zero,
        so this changes no output until training moves them. `prior_sha256`
        becomes the table's (`content_prior.sha256`); a `sha256` that is not
        the table's raises."""
        t = torch.from_numpy(np.array(table, dtype=np.float32))  # a copy: table() is read-only
        if tuple(t.shape) != tuple(self.prior.shape):
            raise ValueError(f"prior shape {tuple(t.shape)} != {tuple(self.prior.shape)}")
        if bool(t[0].any()):
            raise ValueError("prior row 0 (NONE) must be zero")
        got = prior_buffer_sha256(t)
        if sha256 is not None and sha256 != got:
            raise ValueError(f"prior sha256 {sha256!r} is not the table's ({got})")
        self.prior.copy_(t.to(self.prior.device))
        self.prior_sha256 = got

    # ── pieces ──

    def _autocast(self, device: torch.device, bf16: bool | None) -> torch.autocast:
        on = self.bf16 if bf16 is None else bool(bf16)
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=on)

    def embed(
        self,
        cat: torch.Tensor,
        cont: torch.Tensor,
        sec: torch.Tensor,
        pool_t: torch.Tensor,
        pool_ok: torch.Tensor,
        pool_cnt: torch.Tensor,
        *,
        priv: bool = False,
    ) -> torch.Tensor:
        """Token embeddings `[B, T, D]` (the fp32 residual stream).

        Lookups and pools run in fp32 (autocast off); the numeric stem runs
        under the caller's autocast. `priv` selects the PRIV id gains.
        """
        b, t, n_cat = cat.shape
        d = self.cfg.dim
        with torch.autocast(cat.device.type, enabled=False):
            x, prior = self._lookups(cat.reshape(b * t, n_cat), sec.reshape(b * t), priv)
            x, prior = x.view(b, t, d), prior.view(b, t, -1)
            if pool_cnt.shape[1]:
                cnt = pool_cnt.float()
                w = cnt.sum(-1, keepdim=True).clamp(min=1.0)
                pe = torch.matmul(cnt, self.ids.weight) / w
                pp = torch.matmul(cnt, self.prior) / w
                at = torch.arange(t, device=cat.device)
                hot = ((at[None, :, None] == pool_t[:, None, :]) & pool_ok[:, None, :]).float()
                x = x + torch.matmul(hot, pe * self.pool_gain)
                prior = prior + torch.matmul(hot, pp)
        return x + self.num(torch.cat([cont, prior], -1))

    def _lookups(
        self, cat: torch.Tensor, sec: torch.Tensor, priv: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token lookups of flat tokens (`cat` `[N, N_CAT]`, `sec` `[N]`):
        the section, ID and small-column embeddings summed `[N, D]`, and the
        content prior rows of `X_ID_MAIN` `[N, N_PRIOR]`. Call with autocast off."""
        gain = self.priv_id_gain if priv else self.id_gain
        x = self.sec(sec)
        for k, col in enumerate(self.layout.id_cols):
            x = x + self.ids(cat[:, col]) * gain[k]
        v = cat[:, self.small_cols]
        idx = torch.where(v == self.small_absent, 0, v + self.small_off)
        x = x + F.embedding_bag(idx, self.small.weight, mode="sum", padding_idx=0)
        return x, self.prior[cat[:, self.layout.main_col]]

    def embed_packed(
        self,
        cat: torch.Tensor,
        cont: torch.Tensor,
        sec: torch.Tensor,
        pool_tok: torch.Tensor,
        pool_cnt: torch.Tensor,
        *,
        priv: bool = False,
    ) -> torch.Tensor:
        """`embed` of packed tokens `[N, D]`: pooled token `pool_tok[j]` gets
        the weighted mean of `pool_cnt[j]`'s entries."""
        with torch.autocast(cat.device.type, enabled=False):
            x, prior = self._lookups(cat, sec, priv)
            if pool_cnt.shape[0]:
                cnt = pool_cnt.float()
                w = cnt.sum(-1, keepdim=True).clamp(min=1.0)
                pe = torch.matmul(cnt, self.ids.weight) / w
                pp = torch.matmul(cnt, self.prior) / w
                x = x.index_add(0, pool_tok, pe * self.pool_gain)
                prior = prior.index_add(0, pool_tok, pp)
        return x + self.num(torch.cat([cont, prior], -1))

    def trunk(self, b: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """`(residual stream, normed outputs)`, both `[B, T, D]`, from honest
        token fields only (this reads no PRIV field)."""
        x = self.embed(b.cat, b.cont, b.sec, b.pool_t, b.pool_ok, b.pool_cnt)
        keep = (~b.pad)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, keep)
        return x, self.norm(x)

    def _logits(self, h: torch.Tensor, b: TokenBatch) -> torch.Tensor:
        h_opt = torch.gather(h, 1, b.cls_pos[..., None].expand(-1, -1, h.shape[-1]))
        h_state = h[:, 0]
        e_kind = self.kind(b.cls_kind)
        if self.cfg.fp32_heads:
            # X-B's fallback: the heads' hidden layers in fp32 under any autocast.
            with torch.autocast(h.device.type, enabled=False):
                h_opt, h_state, e_kind = h_opt.float(), h_state.float(), e_kind.float()
                zc = self.head_combat(h_opt, h_state, e_kind)
                zo = self.head_ooc(h_opt, h_state, e_kind)
        else:
            # The caller's autocast: bf16 under `bf16=True`, fp32 otherwise. (A
            # nested `torch.autocast(enabled=True)` here would turn bf16 on
            # even under `bf16=False`.)
            zc = self.head_combat(h_opt, h_state, e_kind)
            zo = self.head_ooc(h_opt, h_state, e_kind)
        z = torch.where(b.is_combat[:, None], zc, zo)
        return z.masked_fill(~b.opt_mask, float("-inf"))

    # ── entry points ──

    def policy(self, b: TokenBatch, *, bf16: bool | None = None) -> torch.Tensor:
        """Policy logits `[B, A]` (fp32, -inf off `opt_mask`). Reads no PRIV field.

        The serving path: no value module, no PRIV tokens.
        """
        with self._autocast(b.cat.device, bf16):
            _, h = self.trunk(b)
            return self._logits(h, b)

    def forward(self, b: TokenBatch, *, bf16: bool | None = None) -> NetOut:
        """Logits, fp32 log-probs and both values (the learner's path)."""
        with self._autocast(b.cat.device, bf16):
            x, h = self.trunk(b)
            logits = self._logits(h, b)
            e_priv = self.embed(
                b.pcat, b.pcont, b.psec, b.ppool_t, b.ppool_ok, b.ppool_cnt, priv=True
            )
            if self.cfg.valve:
                x, e_priv = x.detach(), e_priv.detach()
            v_combat, v_run = self.value(x, b.pad, e_priv, b.ppad)
        logp = torch.log_softmax(logits.float(), dim=-1)
        value = torch.where(b.is_combat, v_combat, v_run)
        return NetOut(logits, logp, v_combat, v_run, value)

    def _trunk_packed(self, pb: PackedBatch, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """`trunk` on a `PackedBatch`: `[N, D]` residual stream and normed
        outputs, from honest token fields only (this reads no PRIV field)."""
        x = self.embed_packed(pb.cat, pb.cont, pb.sec, pb.pool_tok, pb.pool_cnt)
        n = int(x.shape[0])
        attn = Varlen(pb.cu, pb.cu, pb.max_len, pb.max_len, n, n, mode)
        for blk in self.blocks:
            x = blk.forward_packed(x, attn)
        return x, self.norm(x)

    def _logits_packed(self, h: torch.Tensor, pb: PackedBatch) -> torch.Tensor:
        """`_logits` of packed outputs `h` `[N, D]`: `[B, A]`, -inf off the classes."""
        h_opt = h[pb.cls_tok]
        h_state = h[pb.cu[:-1].long()]
        e_kind = self.kind(pb.cls_kind)
        row = pb.cls_row
        if self.cfg.fp32_heads:
            with torch.autocast(h.device.type, enabled=False):
                h_opt, h_state, e_kind = h_opt.float(), h_state.float(), e_kind.float()
                zc = self.head_combat.forward_packed(h_opt, h_state, e_kind, row)
                zo = self.head_ooc.forward_packed(h_opt, h_state, e_kind, row)
        else:
            zc = self.head_combat.forward_packed(h_opt, h_state, e_kind, row)
            zo = self.head_ooc.forward_packed(h_opt, h_state, e_kind, row)
        z = torch.where(pb.is_combat[row], zc, zo)
        out = z.new_full((pb.n_real * pb.n_opt,), float("-inf"))
        out[pb.cls_dst] = z
        return out.view(pb.n_real, pb.n_opt)

    def policy_packed(
        self, pb: PackedBatch, *, bf16: bool | None = None, attn: str = "auto"
    ) -> torch.Tensor:
        """`policy` on a `PackedBatch`: logits `[B, A]`. Reads no PRIV field."""
        with self._autocast(pb.cat.device, bf16):
            _, h = self._trunk_packed(pb, attn)
            return self._logits_packed(h, pb)

    def forward_packed(
        self, pb: PackedBatch, *, bf16: bool | None = None, attn: str = "auto"
    ) -> NetOut:
        """`forward` on a `PackedBatch` (the learner's packed serving path):
        the same outputs up to float rounding, with no padded token computed.
        `attn` picks the attention kernel (`Varlen`); inference only."""
        with self._autocast(pb.cat.device, bf16):
            x, h = self._trunk_packed(pb, attn)
            logits = self._logits_packed(h, pb)
            e_priv = self.embed_packed(
                pb.pcat, pb.pcont, pb.psec, pb.ppool_tok, pb.ppool_cnt, priv=True
            )
            if self.cfg.valve:
                x, e_priv = x.detach(), e_priv.detach()
            v_combat, v_run = self.value.forward_packed(x, e_priv, pb, attn)
        logp = torch.log_softmax(logits.float(), dim=-1)
        value = torch.where(pb.is_combat, v_combat, v_run)
        return NetOut(logits, logp, v_combat, v_run, value)

    @torch.no_grad()
    def act_greedy(self, b: TokenBatch, *, bf16: bool | None = None) -> torch.Tensor:
        """Argmax merged class per real row, `[n_real]` int64."""
        return self.policy(b, bf16=bf16)[: b.n_real].argmax(-1)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """`{"trunk": ..., "value": ...}`: the two optimizer groups of plan §6
        (trunk side 1e-4, value module 3e-4). "trunk" is everything outside
        the value module: ID table, stems, trunk and policy heads. Equals
        `ppo.e2e.algo.param_groups(net, ("value.",))`'s split."""
        value = list(self.value.parameters())
        ids = {id(p) for p in value}
        return {"trunk": [p for p in self.parameters() if id(p) not in ids], "value": value}


# ── parameter accounting ──────────────────────────────────────────────────


def n_params(m: nn.Module | Sequence[nn.Parameter]) -> int:
    ps = m.parameters() if isinstance(m, nn.Module) else m
    return sum(p.numel() for p in ps)


def param_breakdown(net: E2ENet) -> dict[str, int]:
    """Parameter counts by plan §5's parts table."""
    stems = n_params(net.sec) + n_params(net.small) + n_params(net.num)
    stems += n_params([net.id_gain, net.priv_id_gain, net.pool_gain])
    trunk = n_params(net.blocks) + n_params(net.norm)
    heads = n_params(net.kind) + n_params(net.head_combat) + n_params(net.head_ooc)
    out = {
        "id_table": n_params(net.ids),
        "stems": stems,
        "trunk": trunk,
        "policy_heads": heads,
        "value_module": n_params(net.value),
    }
    out["total"] = sum(out.values())
    if out["total"] != n_params(net):
        raise AssertionError(f"breakdown {out['total']} != total {n_params(net)}")
    return out


def format_breakdown(net: E2ENet) -> str:
    rows = param_breakdown(net)
    w = max(len(k) for k in rows)
    return "\n".join(f"{k:<{w}}  {v:>10,}  ({v / 1e6:.3f}M)" for k, v in rows.items())


# ── torch.compile ─────────────────────────────────────────────────────────


def static_shapes(batch_sizes: Sequence[int] = (1,)) -> int:
    """How many static shapes the bucketed batches can take: batch sizes x
    token buckets x option buckets x pool buckets (PRIV pads to one bucket,
    and only the 5 summary tokens own pool entries, so J is 5)."""
    return len(batch_sizes) * len(LEN_BUCKETS) * len(OPT_BUCKETS) * len(POOL_BUCKETS)


def raise_recompile_limit(n: int) -> None:
    """Let dynamo keep at least `n` compiled graphs per function. Its default
    (8) is far below `static_shapes()`: past it, a `dynamic=False` compile
    silently falls back to eager for every new shape."""
    import torch._dynamo.config as dc

    for name in (
        "recompile_limit",
        "cache_size_limit",
        "accumulated_recompile_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(dc, name) and getattr(dc, name) < n:
            setattr(dc, name, n)


def compile_net(
    net: E2ENet,
    *,
    batch_sizes: Sequence[int] = (1,),
    entry: str = "policy",
    mode: str | None = None,
) -> Callable[..., Any]:
    """`torch.compile(fullgraph=True, dynamic=False)` of `net.policy` (or
    `entry="forward"`), with dynamo's recompile limit raised to cover every
    static shape of `batch_sizes` (`static_shapes`). Each new shape compiles
    once, on first use (about 7 s on the laptop GPU)."""
    raise_recompile_limit(static_shapes(batch_sizes) + 8)
    fn = getattr(net, entry)
    return torch.compile(fn, fullgraph=True, dynamic=False, mode=mode)


# ── precision (X-B) ───────────────────────────────────────────────────────


def _xb_parts(
    net: E2ENet, b: TokenBatch, bf16_policy: Callable[..., torch.Tensor] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`|logp_fp32 - logp_bf16|` per real merged class, and KL(fp32 || bf16)
    per real row. Log-probs as `E2ENet.forward` computes them (fp32
    log-softmax of fp32 logits)."""
    pol = net.policy if bf16_policy is None else bf16_policy
    lp32 = torch.log_softmax(net.policy(b, bf16=False).float(), -1)[: b.n_real].double()
    lp16 = torch.log_softmax(pol(b, bf16=True).float(), -1)[: b.n_real].double()
    m = b.opt_mask[: b.n_real]
    p32 = lp32.exp()
    kl = torch.where(m, p32 * (lp32 - lp16), torch.zeros_like(p32)).sum(-1)
    return (lp32 - lp16).abs()[m], kl


def _xb_stats(d: torch.Tensor, kl: torch.Tensor) -> dict[str, float]:
    return {
        "p99_abs_dlogp": float(torch.quantile(d, 0.99)),
        "max_abs_dlogp": float(d.max()),
        "mean_kl": float(kl.mean()),
        "n_rows": int(kl.numel()),
        "n_options": int(d.numel()),
    }


@torch.no_grad()
def precision_gap(
    net: E2ENet, b: TokenBatch, *, bf16_policy: Callable[..., torch.Tensor] | None = None
) -> dict[str, float]:
    """fp32 eager vs bf16 log-probs on one batch: the X-B statistics (plan §9).

    Returns the p99 and max of `|logp_fp32 - logp_bf16|` over real merged
    classes, and the mean KL(fp32 || bf16) per real row. `bf16_policy`
    (default `net.policy`) is the bf16 path, for example `compile_net(net)`;
    it is called as `bf16_policy(b, bf16=True)` and returns logits.
    """
    return _xb_stats(*_xb_parts(net, b, bf16_policy))


@torch.no_grad()
def precision_gap_rows(
    net: E2ENet,
    rb: Mapping[str, np.ndarray],
    *,
    device: str | torch.device = "cpu",
    chunk: int = 1024,
    bf16_policy: Callable[..., torch.Tensor] | None = None,
) -> dict[str, float]:
    """X-B over a whole row batch (plan §9: 50K decisions), `chunk` rows at
    a time in bucketed batches padded to `chunk` rows (the static shapes a
    compiled `bf16_policy` sees); statistics over all rows at once."""
    ds, kls = [], []
    for i in range(0, len(rb["n_tok"]), chunk):
        sub = select_rows(rb, np.arange(i, min(i + chunk, len(rb["n_tok"]))))
        d, kl = _xb_parts(net, to_batch(sub, device, n_rows=chunk), bf16_policy)
        ds.append(d.cpu())
        kls.append(kl.cpu())
    return _xb_stats(torch.cat(ds), torch.cat(kls))


# ── checkpoints ───────────────────────────────────────────────────────────


class SchemaMismatch(ValueError):
    """A checkpoint's schema (or vocab) sha256 differs from this checkout's."""


def prior_buffer_sha256(prior: torch.Tensor | np.ndarray) -> str:
    """sha256 of a prior buffer's float32 bytes (C order): `content_prior.sha256`."""
    t = prior.detach().cpu().numpy() if isinstance(prior, torch.Tensor) else prior
    return hashlib.sha256(np.ascontiguousarray(t, np.float32).tobytes()).hexdigest()


def check_serving(stamps: Mapping[str, Any], where: str = "checkpoint") -> None:
    """Refuse to serve a checkpoint on inputs it was not trained on (SCHEMA.md 10, review
    finding LRN-2): stamps of another game version, or a wheel-filled event
    (`fillcheck.WHEEL_EVENTS`) that this process's wheel leaves at zero, such as a
    WP-ENG-1 net on a wheel without the getters. The content prior travels in the
    checkpoint (`check_prior`); D9 and #1069 have no wheel probe."""
    from . import fillcheck as FC

    gv = stamps.get("game_version")
    if gv is not None and gv != C.GAME_VERSION:
        raise SchemaMismatch(f"{where}: stamped for {gv}, this schema is {C.GAME_VERSION}")
    read = set(stamps.get("filled") or ()) & set(FC.WHEEL_EVENTS)
    if missing := sorted(read - set(FC.wheel_filled())):
        raise SchemaMismatch(f"{where}: it read {missing}, which this wheel does not fill")


def check_prior(net: E2ENet, where: str = "checkpoint") -> None:
    """Raise `SchemaMismatch` unless the net's `prior` buffer is what its
    `prior_sha256` says: all zero for None, else those bytes (review finding
    NT-2: an EMA that averaged a prior in would serve a shrunken one)."""
    if net.prior_sha256 is None:
        if bool(net.prior.any()):
            raise SchemaMismatch(f"{where}: prior_sha256 is None but the prior buffer is not zero")
        return
    got = prior_buffer_sha256(net.prior)
    if got != net.prior_sha256:
        raise SchemaMismatch(
            f"{where}: the prior buffer's sha256 {got} != its prior_sha256 {net.prior_sha256}"
        )


def default_stamps(
    *,
    vocab_sha256: str | None = None,
    prior_sha256: str | None = None,
    filled: Sequence[str] = (),
    teacher: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The campaign stamps for a checkpoint: `ppo.e2e.stamps.stamps()` (WP1:
    game version, Python and client source digests, wheel) merged with
    `contracts.stamps()` (schema, vocab and prior sha256, fill events, and
    `teacher`, the combat teacher's `contracts.teacher_stamp` when the net
    trained on its RareMap'd rows).

    A save never fails on a stamp: if WP1's stamps raise (for example, the
    process has not selected v0.111.0), the error is recorded instead. A
    `prior_sha256` adds the "content-prior" fill event (a net that carries the
    prior had the fill).
    """
    vocab_sha256 = vocab_sha256 or current_vocab_sha256()
    if prior_sha256 is not None:
        filled = sorted({*filled, "content-prior"})
    out: dict[str, Any] = {}
    try:
        from ppo.e2e import stamps as wp1

        out.update(wp1.stamps(vocab_sha256=vocab_sha256, schema_sha256=C.schema_sha256()))
    except Exception as e:  # stamps must never fail a save
        out["stamp_error"] = repr(e)
    out.update(
        C.stamps(
            vocab_sha256=vocab_sha256,
            prior_sha256=prior_sha256,
            filled=filled,
            teacher=teacher,
            wheel=out.get("wheel"),
            source_digest=out.get("py_src"),
        )
    )
    out["torch"] = str(torch.__version__)  # a TorchVersion fails a weights-only load
    return out


def _plain(x: Any) -> Any:
    """`x` as plain JSON-like data (dict, list, str, int, float, bool, None),
    which `torch.load(weights_only=True)` reads back; anything else -> `str`."""
    if x is None or type(x) in (bool, int, float, str):
        return x
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return str(x)


def net_state(
    net: E2ENet,
    *,
    vocab_sha256: str | None = None,
    schema_sha256: str | None = None,
    stamps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The net as plain data: `{"format", "cfg", "state_dict"}` (CPU tensors).

    `cfg` carries the `NetConfig`, the schema, vocab and prior sha256 (the
    schema and vocab default to this checkout's) and the stamps
    (`default_stamps` unless given), as plain data so that a weights-only
    `torch.load` reads the file. The learner can embed this dict in its own
    `state.pt`.
    """
    if vocab_sha256 is None:
        vocab_sha256 = current_vocab_sha256()
    if not vocab_sha256:
        raise ValueError("net_state: vocab_sha256 is empty")
    schema = schema_sha256 or C.schema_sha256()
    if stamps is None:
        stamps = default_stamps(vocab_sha256=vocab_sha256, prior_sha256=net.prior_sha256)
    return {
        "format": FORMAT,
        "cfg": {
            "net": net.cfg.to_dict(),
            "schema_sha256": str(schema),
            "vocab_sha256": str(vocab_sha256),
            "prior_sha256": net.prior_sha256,
            "stamps": _plain(dict(stamps)),
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
    }


def net_from_state(
    ck: Mapping[str, Any],
    *,
    vocab_sha256: str | None = None,
    schema_sha256: str | None = None,
    device: str | torch.device = "cpu",
    want_stamps: Mapping[str, Any] | None = None,
    restart: bool = False,
) -> E2ENet:
    """Rebuild a net from `net_state`'s dict; refuses a different schema or
    vocab (defaults: this checkout's `contracts.schema_sha256()` and
    `current_vocab_sha256()`; pass `vocab_sha256=""` to skip the vocab check).

    With `want_stamps` (this process's `contracts.stamps`), also
    `contracts.check_stamps(saved, want_stamps, restart=restart)`: the fill
    state, the prior and the teacher stamp (`SchemaError` on a mismatch).
    The `prior` buffer must match the saved `prior_sha256` (`check_prior`)."""
    if ck.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} checkpoint (format {ck.get('format')!r})")
    cfg = ck["cfg"]
    want = schema_sha256 or C.schema_sha256()
    if cfg.get("schema_sha256") != want:
        raise SchemaMismatch(
            f"checkpoint schema sha256 {cfg.get('schema_sha256')!r} != this checkout's {want!r}"
        )
    if vocab_sha256 is None:
        vocab_sha256 = current_vocab_sha256()
    if vocab_sha256 and cfg.get("vocab_sha256") != vocab_sha256:
        raise SchemaMismatch(
            f"checkpoint vocab sha256 {cfg.get('vocab_sha256')!r} != this checkout's "
            f"{vocab_sha256!r}"
        )
    if want_stamps is not None:
        C.check_stamps(cfg.get("stamps") or {}, want_stamps, restart=restart)
    net = E2ENet(NetConfig.from_dict(cfg["net"]))
    net.load_state_dict(ck["state_dict"])
    net.prior_sha256 = cfg.get("prior_sha256")
    check_prior(net)
    return net.to(device)


def save_checkpoint(path: str | pathlib.Path, net: E2ENet, **kw: Any) -> str:
    """Atomic save of `net_state(net, **kw)`; returns the file's sha256."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(net_state(net, **kw), tmp)
    tmp.replace(path)
    return file_sha256(path)


def load_checkpoint(
    path: str | pathlib.Path,
    *,
    vocab_sha256: str | None = None,
    schema_sha256: str | None = None,
    device: str | torch.device = "cpu",
    want_stamps: Mapping[str, Any] | None = None,
    restart: bool = False,
) -> tuple[E2ENet, dict[str, Any]]:
    """`(net, cfg)` from `save_checkpoint`'s file; refuses a different schema
    or vocab, and with `want_stamps` checks the stamps (`net_from_state`)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = net_from_state(
        ck,
        vocab_sha256=vocab_sha256,
        schema_sha256=schema_sha256,
        device=device,
        want_stamps=want_stamps,
        restart=restart,
    )
    return net, ck["cfg"]


def file_sha256(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── the greedy student (evaluation) ───────────────────────────────────────


def decision_row(
    d: Any, enc: Mapping[str, Any], priv: Mapping[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """A one-row row batch for decision `d` (`contracts.EnvDecision`) from
    the tokenizer's `encode(d)` output and, on the value path, its
    `encode_priv(d)` output.

    Token, member, class and pooled arrays come from `enc` (`priv`), counts
    from their lengths. `mem_class` and `cls_kind` default to the decision's
    (`d.member_class`, `d.class_kinds()`) when the encoding leaves them out.
    The acting-time fields take their defaults: actor 0 (student), action 0,
    reward, `logp_behav`, `v_behav` 0, `t_src` 0, `tprob` NaN.
    """
    rep = np.asarray(d.rep, np.int64)
    n_cls = len(rep)
    src = {**enc, **(priv or {})}
    src.setdefault("mem_class", d.member_class)
    src.setdefault("cls_kind", np.asarray(d.kinds, np.uint8)[rep])
    src.setdefault("tprob", np.full(n_cls, np.nan, np.float32))
    out: dict[str, np.ndarray] = {}
    for f in C.ROW_FIELDS:
        if f.axis == "row":
            continue
        v = src.get(f.name)
        shape = (-1, f.width) if f.width else (-1,)
        if v is None:
            v = np.zeros((0, f.width) if f.width else (0,), f.dtype)
        out[f.name] = np.asarray(v, f.dtype).reshape(shape)
    scalars = {
        "run_uid": d.run_uid,
        "t": d.t,
        "is_combat": d.is_combat,
        "dtype": d.dtype,
        "n_classes": n_cls,
        "cell": d.cell,
        "n_tok": len(out["sec"]),
        "n_mem": len(out["mem_tok"]),
        "n_priv": len(out["psec"]),
        "n_pool": len(out["pool_vid"]),
        "n_ppool": len(out["ppool_vid"]),
        "sctx_lo": src.get("sctx_lo", 0),
        "n_sctx": src.get("n_sctx", 0),
    }
    for f in C.ROW_FIELDS:
        if f.axis == "row":
            out[f.name] = np.asarray([scalars.get(f.name, 0)], f.dtype)
    return out


class TokenizerRows:
    """`StudentPolicy.encode_row` on WP3b's tokenizer (`ppo.e2e.tokens`):
    an `env.E2EDecision` -> its one-row row batch, policy arrays only
    (`priv=False`).

    `tokens.encode` reads the decision's own combat view (the env's intent
    log in `d.snap`, the grant picks in `d.sel`), so it needs no env hooks:
    evaluation's `Player` builds the env without a tokenizer. Its rows equal
    the `contracts.Tokenizer` rows the actor stores (`test_net.py` checks it
    on a real run). `begin_run` drops the tokenizer's run-static cache,
    which otherwise keeps up to 512 finished runs alive. `rare` is the
    combat teacher's RareMap for the S_ctx content slots; the student never
    reads those slots, so None (the default) gives the student the same
    input.
    """

    def __init__(self, rare: Any = None) -> None:
        from . import tokens

        self.tokens = tokens
        self.rare = rare

    def begin_run(self, seed: int | None = None) -> None:
        self.tokens.clear_cache()

    def __call__(self, d: Any) -> dict[str, np.ndarray]:
        tk = self.tokens.encode(d, rare=self.rare, priv=False)
        return self.tokens.row_batch([tk], run_uid=[d.run_uid], t=[d.row])


class StudentPolicy:
    """Greedy play with an `E2ENet`: the argmax merged class of each decision.

    Implements `ppo.e2e.evaluate.StudentPolicy` (`act_decisions`, optional
    `begin_run` and `meta`); `make_student` is the `--policy` factory. `act`
    scores a row batch; `act_decisions` turns env decisions into rows first,
    with `encode_row` (decision -> one-row row batch, for example
    `TokenizerRows`) or else with `tokenizer` (a `contracts.Tokenizer`,
    through `decision_row`). Both return merged-class indices: the env takes
    class `c` with `E2EEnv.step(c)`. A `contracts.Tokenizer`'s `reset_fight`
    and `observe` calls must stay current, which the env does when the
    tokenizer is passed in its `EnvConfig`. Evaluation runs fp32 by default
    (plan section 11 reads greedy argmax, and fp32 removes bf16 ties from
    the read).
    """

    def __init__(
        self,
        net: E2ENet,
        *,
        device: str | torch.device = "cpu",
        bf16: bool = False,
        tokenizer: Any = None,
        encode_row: Callable[[Any], Mapping[str, np.ndarray]] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.bf16 = bool(bf16)
        self.tokenizer = tokenizer
        self.encode_row = encode_row
        self.meta = dict(meta or {})

    @classmethod
    def from_checkpoint(
        cls,
        path: str | pathlib.Path,
        *,
        vocab_sha256: str | None = None,
        device: str | torch.device = "cpu",
        want_stamps: Mapping[str, Any] | None = None,
        **kw: Any,
    ) -> StudentPolicy:
        """Load `save_checkpoint`'s file; refuses a different schema or
        vocabulary sha256, a game version or wheel fill this process cannot serve
        (`check_serving`), and with `want_stamps` (the serving process's
        `contracts.stamps`) a fill state, prior or teacher stamp that differs."""
        net, cfg = load_checkpoint(
            path, vocab_sha256=vocab_sha256, device=device, want_stamps=want_stamps
        )
        check_serving(cfg.get("stamps") or {}, str(path))
        meta = {"path": str(path), "sha256": file_sha256(path), "cfg": cfg}
        return cls(net, device=device, meta=meta, **kw)

    def begin_run(self, seed: int) -> None:
        """Called before each run (`ppo.e2e.evaluate`): resets `encode_row`'s
        per-run state when it has any."""
        begin = getattr(self.encode_row, "begin_run", None)
        if begin is not None:
            begin(seed)

    @torch.no_grad()
    def act(self, rb: Mapping[str, np.ndarray]) -> list[int]:
        """The greedy merged class of each row of a row batch."""
        if len(rb["n_tok"]) == 0:
            return []
        b = to_batch(rb, self.device)
        return [int(i) for i in self.net.act_greedy(b, bf16=self.bf16).cpu()]

    def row(self, d: Any) -> Mapping[str, np.ndarray]:
        """Decision `d` as a one-row row batch: `encode_row(d)`, or else
        `decision_row(e, tokenizer.encode(e))` with `e` a
        `contracts.EnvDecision` (an `env.E2EDecision`'s `decision`)."""
        if self.encode_row is not None:
            return self.encode_row(d)
        if self.tokenizer is None:
            raise RuntimeError("StudentPolicy has neither encode_row nor a tokenizer")
        e = d.decision if hasattr(d, "decision") else d
        return decision_row(e, self.tokenizer.encode(e))

    def act_decisions(self, decisions: Sequence[Any]) -> list[int]:
        """`act` on `row(d)` of each decision."""
        return self.act(concat_rows([self.row(d) for d in decisions]))


def make_student(
    ckpt: str | pathlib.Path, device: str | torch.device = "cpu", **kw: Any
) -> StudentPolicy:
    """`ppo.e2e.evaluate`'s `--policy ppo.e2e.net:make_student`: the greedy
    student of checkpoint `ckpt` on WP3b's tokenizer rows (`TokenizerRows`).
    Refuses a checkpoint of another schema or vocabulary."""
    return StudentPolicy.from_checkpoint(ckpt, device=device, encode_row=TokenizerRows(), **kw)
