"""Row batches the learner keeps, ships to the update ranks and saves (WP4c).

Every row is a row of a `ppo.e2e.contracts` row batch (`ROW_FIELDS`,
SCHEMA.md section 9): per-row fields, then the ragged token, member, class,
PRIV and pool axes. A `Chunk` is one row batch with its axis offsets computed
once; the learner keeps one chunk per serving round and every stored row
points into one (`Row.chunk`, `Row.i`), so a chunk is freed once no stored,
held or in-flight row points into it. Once an iteration's joined table is in
its `SharedMemory` block, the learner points the shipped rows into that block
(a `ShmChunk`), so their rounds and the last update's held block go. Rank 0
ends the update by copying the rows its hold-back keeps out of the block into
a block of their own (`take_to_shm`, `update.Rank`); the learner maps it and
points those rows there, and the update's block goes with the cut's other
rows. A held row would otherwise keep its whole round (or the whole block)
alive, and the held rows of the last `algo.HOLD_ITERS` iterations point into
nearly every round of them.

* `take(chunk, idx)`: rows `idx` of one chunk as a new row batch (copies),
  one vectorised gather per axis.
* `gather(refs)`: rows from any chunks, in the order given (the joined
  table): one `take` per chunk, then one reordering `take`.
* `batch_to_shm` / `batch_from_shm`: a row batch in one `SharedMemory` block
  (the learner writes it, the ranks read views of it); `take_to_shm`: rows
  of a chunk straight into a new block (`batch_to_shm(take(...))` in one
  copy); `gather_to_shm`: rows from any chunks straight into a new block
  (`batch_to_shm(gather(...))` in one copy, the learner's cut);
  `ShmChunk`: a chunk of a block's views that owns the block.

Dense rows cost 486 bytes per token (SCHEMA.md "Size"). The compact layout
(`contracts.to_compact`) would cut that about tenfold; it is not used yet
(open issue, recorded in the WP4c report).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from ppo.combat_ppo.ranks import shm_view, to_shm

from . import contracts as C

FIELDS = C.ROW_FIELDS
#: Ragged axis -> its per-row count field.
AXIS_COUNT = {"tok": "n_tok", "mem": "n_mem", "cls": "n_classes", "priv": "n_priv"}
AXIS_COUNT |= {"pool": "n_pool", "ppool": "n_ppool"}
#: The fields the actor fills at acting time (the worker leaves them at their defaults).
ACTING = ("actor", "logp_behav", "action", "iter", "v_behav", "t_src")


def offsets(rb: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Start offsets (length B + 1) of every ragged axis."""
    out = {}
    for axis, cnt in AXIS_COUNT.items():
        n = np.asarray(rb[cnt], np.int64)
        o = np.zeros(len(n) + 1, np.int64)
        np.cumsum(n, out=o[1:])
        out[axis] = o
    return out


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    lens = np.asarray(lens, np.int64)
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    cum = np.cumsum(lens) - lens
    return np.repeat(np.asarray(starts, np.int64) - cum, lens) + np.arange(total)


#: The most bytes one scattered copy step (`gather_to_shm`) holds in a temporary:
#: under numpy's 4 MiB huge-page threshold, with room.
COPY_BLOCK_BYTES = 1 << 20


def _blocks(
    ch: Chunk, idx: np.ndarray, pos: np.ndarray, like: Mapping[str, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`idx`/`pos` cut into row blocks whose widest field's entries fit in
    `COPY_BLOCK_BYTES` (on the chunk's average entries per row)."""
    n = len(idx)
    if n <= 64:  # a serving round's rows: small whatever the fields
        return [(idx, pos)]
    per_row = 8.0  # a row field's bytes, and the row index itself
    for f in FIELDS:
        x = like[f.name]
        width = x.itemsize * int(np.prod(x.shape[1:], dtype=np.int64))
        if f.axis == "row":
            per_row = max(per_row, float(width))
        else:
            o = ch.off[f.axis]
            mean = float(o[-1] - o[0]) / max(1, ch.n)
            per_row = max(per_row, mean * max(width, 8))  # 8: the entry index
    rows = max(1, int(COPY_BLOCK_BYTES // per_row))
    return [(idx[b : b + rows], pos[b : b + rows]) for b in range(0, n, rows)]


def _places(si: Any, di: Any, o: np.ndarray | None, oo: np.ndarray | None) -> tuple[Any, Any]:
    """Source rows `si` and destination rows `di` as entries on one axis (`o`/`oo`:
    the source's and the block's offsets on it; None for the row axis): slices stay
    slices, and index blocks whose destination is one span become a slice there."""
    if isinstance(si, slice):
        if o is None:
            return si, di
        assert oo is not None
        return slice(int(o[si.start]), int(o[si.stop])), slice(int(oo[di.start]), int(oo[di.stop]))
    if o is None:
        s, d = si, di
    else:
        assert oo is not None
        lens = o[si + 1] - o[si]
        s, d = _ranges(o[si], lens), _ranges(oo[di], lens)
    if len(d) and int(d[-1]) - int(d[0]) + 1 == len(d) and bool((np.diff(d) == 1).all()):
        return s, slice(int(d[0]), int(d[-1]) + 1)
    return s, d


def _copy(v: np.ndarray, x: np.ndarray, s: Any, d: Any) -> None:
    """`v[d] = x[s]` without a large temporary: slice to slice directly, an index
    block into one destination span with `take(out=)`, else through a temporary of
    the block's size."""
    if isinstance(s, slice):
        v[d] = x[s]
    elif not len(s):
        return
    elif isinstance(d, slice):
        np.take(x, s, axis=0, out=v[d], mode="clip")  # in range; "raise" would buffer
    else:
        v[d] = np.take(x, s, axis=0)


class Chunk:
    """One row batch and its axis offsets."""

    __slots__ = ("n", "off", "rb")

    def __init__(self, rb: Mapping[str, np.ndarray]) -> None:
        self.rb = dict(rb)
        self.n = len(self.rb["n_tok"])
        self.off = offsets(self.rb)

    def span(self, axis: str, i: int) -> slice:
        o = self.off[axis]
        return slice(int(o[i]), int(o[i + 1]))

    def classes(self, i: int) -> np.ndarray:
        """Row `i`'s class of each member (`mem_class`), int64."""
        return np.asarray(self.rb["mem_class"][self.span("mem", i)], np.int64)


def take(ch: Chunk, idx: Sequence[int] | np.ndarray) -> dict[str, np.ndarray]:
    """Rows `idx` of `ch` as a new row batch (within-row indices carry over)."""
    idx = np.asarray(idx, np.int64)
    sel = {axis: _ranges(o[idx], o[idx + 1] - o[idx]) for axis, o in ch.off.items()}
    return {
        f.name: ch.rb[f.name][idx] if f.axis == "row" else ch.rb[f.name][sel[f.axis]]
        for f in FIELDS
    }


def axis_rows(ch: Chunk, name: str, idx: Sequence[int] | np.ndarray) -> np.ndarray:
    """Field `name` (a ragged-axis field) of rows `idx`, concatenated in that order."""
    f = C.FIELDS[name]
    idx = np.asarray(idx, np.int64)
    o = ch.off[f.axis]
    return ch.rb[name][_ranges(o[idx], o[idx + 1] - o[idx])]


def concat(batches: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not batches:
        return C.empty_row_batch()
    if len(batches) == 1:
        return {f.name: np.asarray(batches[0][f.name]) for f in FIELDS}
    return {f.name: np.concatenate([b[f.name] for b in batches]) for f in FIELDS}


def _grouped(refs: Sequence[tuple[Chunk, int]]) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """The rows `refs` grouped by chunk (one `take` each): the row batch, each of its
    rows' position in `refs`, and the number of chunks."""
    groups: dict[int, tuple[Chunk, list[int], list[int]]] = {}
    for pos, (ch, i) in enumerate(refs):
        g = groups.get(id(ch))
        if g is None:
            g = groups[id(ch)] = (ch, [], [])
        g[1].append(int(i))
        g[2].append(pos)
    parts = [take(ch, idx) for ch, idx, _ in groups.values()]
    pos = np.concatenate([np.asarray(p, np.int64) for _, _, p in groups.values()])
    return concat(parts), pos, len(groups)


def gather(refs: Sequence[tuple[Chunk, int]]) -> dict[str, np.ndarray]:
    """The rows `refs` (`(chunk, row)` pairs) as one row batch, in that order."""
    if not refs:
        return C.empty_row_batch()
    rb, pos, n_chunks = _grouped(refs)
    if n_chunks == 1 and bool((np.diff(pos) > 0).all()):
        return rb
    return take(Chunk(rb), np.argsort(pos, kind="stable"))


def _layout(
    n_rows: int, totals: Mapping[str, int], like: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """`ranks.to_shm`'s layout (in `FIELDS` order, 64-byte aligned) of a row batch of
    `n_rows` rows with `totals[axis]` entries per ragged axis, each field shaped and
    typed as `like`'s; `"_size"` is the block's size."""
    layout: dict[str, Any] = {}
    off = 0
    for f in FIELDS:
        x = like[f.name]
        shape = (n_rows if f.axis == "row" else int(totals[f.axis]), *x.shape[1:])
        layout[f.name] = (off, tuple(int(s) for s in shape), x.dtype.str)
        off += -(-int(np.prod(shape, dtype=np.int64)) * x.dtype.itemsize // 64) * 64
    layout["_size"] = off
    return layout


def gather_to_shm(
    refs: Sequence[tuple[Chunk, int]], *, threads: int = 1
) -> tuple[shared_memory.SharedMemory, dict[str, Any]]:
    """`batch_to_shm(gather(refs))` in one copy: each chunk's rows go straight to
    their places in a new block (the same bytes, the same layout), with no row
    batch in between, split over `threads` threads by chunk (their rows' places
    are disjoint). It only reads the chunks, and numpy drops the GIL while it
    copies, so it can run in a thread while the chunks' owner goes on (the
    learner's cut, S1-cut). The caller owns the block (`close`, `unlink`)."""
    if not refs:
        return batch_to_shm(C.empty_row_batch())
    n = len(refs)
    groups: dict[int, tuple[Chunk, list[int], list[int]]] = {}
    for pos, (ch, i) in enumerate(refs):
        g = groups.get(id(ch))
        if g is None:
            g = groups[id(ch)] = (ch, [], [])
        g[1].append(int(i))
        g[2].append(pos)
    parts = [
        (ch, np.asarray(idx, np.int64), np.asarray(pos, np.int64))
        for ch, idx, pos in groups.values()
    ]
    like = parts[0][0].rb
    for ch, _idx, _pos in parts[1:]:
        for f in FIELDS:
            a, b = ch.rb[f.name], like[f.name]
            if a.dtype != b.dtype or a.shape[1:] != b.shape[1:]:
                raise ValueError(f"field {f.name}: chunks of different dtypes or widths")
    # Every output row's entry count per axis, then the output offsets.
    out_off: dict[str, np.ndarray] = {}
    for axis in AXIS_COUNT:
        cnt = np.empty(n, np.int64)
        for ch, idx, pos in parts:
            o = ch.off[axis]
            cnt[pos] = o[idx + 1] - o[idx]
        oo = np.zeros(n + 1, np.int64)
        np.cumsum(cnt, out=oo[1:])
        out_off[axis] = oo
    layout = _layout(n, {a: int(o[-1]) for a, o in out_off.items()}, like)
    size = layout.pop("_size")
    # Per chunk, its rows' places. A chunk whose rows land in order on one span (the
    # held rows' block, which leads the table) copies slices. Any other chunk copies
    # in row blocks (`_blocks`): their index arrays and copies stay under
    # `COPY_BLOCK_BYTES`, so no copy allocates a large array. numpy asks for huge
    # pages on large arrays, and on a fragmented host each such allocation can
    # stall in direct compaction.
    plans = []
    for ch, idx, pos in parts:
        if bool((np.diff(pos) == 1).all() and (np.diff(idx) == 1).all()):
            i0, i1, p0, p1 = int(idx[0]), int(idx[-1]) + 1, int(pos[0]), int(pos[-1]) + 1
            plans.append((ch, [(slice(i0, i1), slice(p0, p1))]))
        else:
            plans.append((ch, _blocks(ch, idx, pos, like)))
    shm = shared_memory.SharedMemory(create=True, size=max(size, 64), track=False)

    def copy(group: list[tuple[Chunk, list[tuple[Any, Any]]]]) -> None:
        views = {f.name: shm_view(shm, layout[f.name]) for f in FIELDS}
        try:
            for ch, spans in group:
                for si, di in spans:
                    places = {"row": _places(si, di, None, None)}
                    for axis, o in ch.off.items():
                        places[axis] = _places(si, di, o, out_off[axis])
                    for f in FIELDS:
                        s_, d_ = places[f.axis]
                        _copy(views[f.name], ch.rb[f.name], s_, d_)
        finally:
            views.clear()

    try:
        k = max(1, min(int(threads), len(plans)))
        if k == 1:
            copy(plans)
        else:
            with ThreadPoolExecutor(k, thread_name_prefix="e2e-gather") as ex:
                list(ex.map(copy, [plans[j::k] for j in range(k)]))
    except BaseException:
        shm.unlink()
        with contextlib.suppress(BufferError):
            shm.close()
        raise
    return shm, layout


def batch_to_shm(
    rb: Mapping[str, np.ndarray],
) -> tuple[shared_memory.SharedMemory, dict[str, Any]]:
    """`rb` in one `SharedMemory` block; the caller owns it (`close`, `unlink`)."""
    spec = {}
    for f in FIELDS:
        x = np.asarray(rb[f.name])
        spec[f.name] = (x.shape, x.dtype, x)
    return to_shm(spec)


def take_to_shm(
    ch: Chunk, idx: Sequence[int] | np.ndarray
) -> tuple[shared_memory.SharedMemory, dict[str, Any]]:
    """Rows `idx` of `ch` in a new `SharedMemory` block, laid out as `batch_to_shm`
    lays out `take(ch, idx)`: each field's rows go straight into the block, with
    no row batch in between. The caller owns the block (`close`, `unlink`)."""
    idx = np.asarray(idx, np.int64)
    sel = {axis: _ranges(o[idx], o[idx + 1] - o[idx]) for axis, o in ch.off.items()}
    src = {f.name: (ch.rb[f.name], idx if f.axis == "row" else sel[f.axis]) for f in FIELDS}
    layout: dict[str, Any] = {}
    off = 0
    for name, (x, ix) in src.items():  # `ranks.to_shm`'s layout: 64-byte aligned, in order
        shape = (len(ix), *x.shape[1:])
        layout[name] = (off, shape, x.dtype.str)
        off += -(-int(np.prod(shape, dtype=np.int64)) * x.dtype.itemsize // 64) * 64
    shm = shared_memory.SharedMemory(create=True, size=max(off, 64), track=False)
    try:
        for name, (x, ix) in src.items():
            v = shm_view(shm, layout[name])
            np.take(x, ix, axis=0, out=v, mode="clip")  # in range; "raise" would buffer
            del v
    except BaseException:
        shm.unlink()
        with contextlib.suppress(BufferError):
            shm.close()
        raise
    return shm, layout


def batch_from_shm(shm: shared_memory.SharedMemory, layout: Mapping[str, Any]) -> dict[str, Any]:
    """Views of a `batch_to_shm` block (valid while `shm` is open)."""
    return {k: shm_view(shm, ent) for k, ent in layout.items()}


class ShmChunk(Chunk):
    """A chunk of views of a `batch_to_shm` block, which it owns: the block is
    closed (unmapped) when the chunk goes, that is with the last row pointing
    into it. `unlink` removes the block's name once the ranks are done with
    it; the chunk's views stay valid until it goes."""

    __slots__ = ("shm",)

    def __init__(self, shm: shared_memory.SharedMemory, layout: Mapping[str, Any]) -> None:
        super().__init__(batch_from_shm(shm, layout))
        self.shm = shm

    def unlink(self) -> None:
        with contextlib.suppress(FileNotFoundError):  # already unlinked
            self.shm.unlink()

    def __del__(self) -> None:
        shm = getattr(self, "shm", None)
        if shm is None:  # __init__ failed
            return
        self.rb = {}  # the views first: a block with views left cannot close
        with contextlib.suppress(BufferError):
            shm.close()
