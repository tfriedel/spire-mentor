"""Compact rows: a table-driven, lossless codec for actor-to-learner rows.

A row is one decision. It holds variable-length token arrays and a few
values per row. A table of `Field`s describes the arrays, so a new token
layout needs a new table, not new code (`cppo.cppo_fields` is #1081's table).

Each field belongs to a *stream*. All fields of one stream share a length per
row (for example the honest tokens, or the privileged tokens). Fields of the
`ROW` stream have exactly one element per row (for example `n_act`).

Each stream field uses one of two codecs:

* `raw` stores the array as it is.
* `sparse` stores a bit mask of the cells that differ from the field's `fill`
  value, then the differing cells in row-major order. Cells are compared by
  bit pattern, so `-0.0`, NaN and every other value round-trip exactly.

A row encodes to one blob of bytes. Every segment starts on an 8-byte
boundary, so the blob can be viewed as any field dtype without copying::

    int32 header   one length per stream, then one count per sparse field
    row block      the ROW fields, each at a fixed offset
    segments       per stream field, in table order:
                   raw    -> [length, width] values
                   sparse -> [length, ceil(width / 8)] mask bytes, then values

`encode` runs on the worker (numpy only, no torch). `decode` is the
per-row numpy reference. `decode_torch` decodes many rows at once with
vectorized index arithmetic, on the CPU or a GPU, without a device sync.
`RowStore` keeps encoded rows in one arena on the learner.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROW = ""  # the stream of one-per-row fields
RAW = "raw"
SPARSE = "sparse"
ALIGN = 8

_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
_INT = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}  # torch views per element size
_STRUCT = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "i8": "q",
    "u8": "Q",
}


def align(n: int) -> int:
    """`n` rounded up to the codec's segment alignment (8 bytes)."""
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def _align_np(n: np.ndarray) -> np.ndarray:
    return (n + (ALIGN - 1)) & ~(ALIGN - 1)


@dataclass(frozen=True)
class Field:
    """One array of a row.

    Attributes:
        name: The key in the row mapping and in the decoded output.
        dtype: A numpy dtype name (`int16`, `float16`, ...): the stored type.
        width: Columns per element; 0 means one value per element.
        stream: The stream whose length gives the element count, or `ROW`.
        codec: `raw` or `sparse` (sparse needs a stream and `width >= 1`).
        fill: For `sparse`: the value left out, one per column or one for all.
    """

    name: str
    dtype: str
    width: int = 0
    stream: str = ROW
    codec: str = RAW
    fill: int | float | tuple[int | float, ...] = 0

    def __post_init__(self) -> None:
        dt = np.dtype(self.dtype)
        if dt.kind not in "biuf" or dt.itemsize not in _UINT or dt.byteorder == ">":
            raise ValueError(f"field {self.name!r}: unsupported dtype {self.dtype!r}")
        if self.width < 0:
            raise ValueError(f"field {self.name!r}: width must be >= 0")
        if self.codec not in (RAW, SPARSE):
            raise ValueError(f"field {self.name!r}: codec must be {RAW!r} or {SPARSE!r}")
        if self.codec == SPARSE and (self.width < 1 or self.stream == ROW):
            raise ValueError(f"field {self.name!r}: a sparse field needs a stream and width >= 1")
        if isinstance(self.fill, tuple) and len(self.fill) != self.cols:
            raise ValueError(
                f"field {self.name!r}: fill has {len(self.fill)} values, not {self.cols}"
            )

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def cols(self) -> int:
        return max(1, self.width)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.width,) if self.width else ()

    def fill_array(self) -> np.ndarray:
        """The fill as one value per column, in the field's dtype."""
        return np.broadcast_to(np.asarray(self.fill, self.np_dtype), (self.cols,)).copy()


@dataclass(frozen=True)
class _Seg:
    field: int  # index into RowCodec.fields
    kind: str  # "raw" | "bits" | "vals"
    stream: int  # index into RowCodec.streams
    sparse: int  # index into the sparse fields (-1 if none)
    unit: int  # bytes per counted element


class RowCodec:
    """Encodes rows to blobs and decodes them, as the field table says."""

    def __init__(self, fields: Iterable[Field]) -> None:
        self.fields: tuple[Field, ...] = tuple(fields)
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in {names}")
        self.streams: tuple[str, ...] = tuple(
            dict.fromkeys(f.stream for f in self.fields if f.stream != ROW)
        )
        self.row_fields = tuple(f for f in self.fields if f.stream == ROW)
        self.stream_fields = tuple(f for f in self.fields if f.stream != ROW)
        sparse = [f for f in self.stream_fields if f.codec == SPARSE]
        self.sparse_names = tuple(f.name for f in sparse)
        self.n_hdr = len(self.streams) + len(sparse)
        self.hdr_bytes = align(4 * self.n_hdr)
        # ROW fields: fixed offsets inside the row block, naturally aligned.
        off, o = {}, 0
        for f in self.row_fields:
            it = f.np_dtype.itemsize
            o = (o + it - 1) // it * it
            off[f.name] = o
            o += it * f.cols
        self.row_off = off
        self.row_bytes = align(o)
        self.fixed_bytes = self.hdr_bytes + self.row_bytes
        segs = []
        for f in self.stream_fields:
            fi = self.fields.index(f)
            si = self.streams.index(f.stream)
            it = f.np_dtype.itemsize
            if f.codec == RAW:
                segs.append(_Seg(fi, "raw", si, -1, it * f.cols))
            else:
                pi = self.sparse_names.index(f.name)
                segs.append(_Seg(fi, "bits", si, pi, (f.width + 7) // 8))
                segs.append(_Seg(fi, "vals", si, pi, it))
        self._segs = tuple(segs)
        self._fills = {f.name: f.fill_array() for f in sparse}
        self._fill_bits = {
            f.name: self._fills[f.name].view(_UINT[f.np_dtype.itemsize]) for f in sparse
        }
        self._fill_t: dict[tuple[str, str], Any] = {}
        groups: dict[int, list[tuple[int, int]]] = {}
        for j, sg in enumerate(self._segs):
            if sg.kind == "bits":
                vs = self.fields[sg.field].np_dtype.itemsize
                groups.setdefault(vs, []).append((j, j + 1))  # "vals" follows "bits"
        self._sparse_groups = dict(sorted(groups.items()))
        # Precomputed per-field plans: encode runs per decision on the workers.
        self._sfp = tuple(
            (
                f.name,
                f.np_dtype,
                f.shape,
                self.streams.index(f.stream),
                f.codec == SPARSE,
                _UINT[f.np_dtype.itemsize],
                self._fill_bits.get(f.name),
                f.codec == SPARSE and not self._fill_bits[f.name].any(),
            )
            for f in self.stream_fields
        )
        self._hdr_struct = struct.Struct("<" + "i" * self.n_hdr)
        self._row_structs = tuple(
            (
                f,
                struct.Struct("<" + _STRUCT[f.np_dtype.str[1:]]),
                self.hdr_bytes + self.row_off[f.name],
            )
            if not f.width and f.np_dtype.kind in "iu"
            else (f, None, self.hdr_bytes + self.row_off[f.name])
            for f in self.row_fields
        )

    # ── sizes ────────────────────────────────────────────────────────────

    def max_bytes(self, caps: Mapping[str, int]) -> int:
        """The largest blob a row can need, given each stream's length cap."""
        n = self.fixed_bytes
        for s in self._segs:
            cap = int(caps[self.streams[s.stream]])
            f = self.fields[s.field]
            count = cap * f.cols if s.kind == "vals" else cap
            n += align(count * s.unit)
        return n

    def sizes(self, hdr: np.ndarray) -> np.ndarray:
        """Blob sizes in bytes for header rows `hdr` (`[N, n_hdr]`)."""
        return self.fixed_bytes + self._seg_sizes(np.asarray(hdr, np.int64)).sum(1)

    def _seg_sizes(self, hdr: np.ndarray) -> np.ndarray:
        ns = len(self.streams)
        out = np.empty((len(hdr), len(self._segs)), np.int64)
        for j, s in enumerate(self._segs):
            count = hdr[:, ns + s.sparse] if s.kind == "vals" else hdr[:, s.stream]
            out[:, j] = _align_np(count * s.unit)
        return out

    def describe(self) -> list[dict[str, Any]]:
        """The table as plain dicts (for cfg stamps and reports)."""
        return [
            {
                "name": f.name,
                "dtype": f.dtype,
                "width": f.width,
                "stream": f.stream,
                "codec": f.codec,
                "fill": list(f.fill) if isinstance(f.fill, tuple) else f.fill,
            }
            for f in self.fields
        ]

    # ── encode (worker side, numpy only) ─────────────────────────────────

    def encode(self, row: Mapping[str, Any], out: np.ndarray | None = None) -> Any:
        """Encode one row.

        Args:
            row: Every field's value. Stream fields are arrays of shape
                `[length, width]` (`[length]` if width is 0) in the field's
                exact dtype. ROW fields are scalars or `[width]` arrays and are
                cast to the field's dtype (integers must fit).
            out: A writable uint8 buffer. If given, the blob is written to
                `out[:n]` and `n` is returned; otherwise a new array is returned.
        """
        lens = [-1] * len(self.streams)
        parts: list[np.ndarray] = []
        nnz: list[int] = []
        for name, dt, tail, si, sparse, ut, fb, zf in self._sfp:
            a = row[name]
            if not isinstance(a, np.ndarray) or a.dtype != dt:
                got = getattr(a, "dtype", type(a).__name__)
                raise TypeError(f"field {name!r}: expected a {dt} array, got {got}")
            if a.shape[1:] != tail:
                raise ValueError(f"field {name!r}: shape {a.shape} is not [length, *{tail}]")
            n = a.shape[0]
            if lens[si] < 0:
                lens[si] = n
            elif lens[si] != n:
                raise ValueError(
                    f"stream {self.streams[si]!r}: field {name!r} has length {n}, not {lens[si]}"
                )
            if not a.flags.c_contiguous:
                a = np.ascontiguousarray(a)
            if sparse:
                bits, vals, _ = _sparse_parts(a, ut, fb, zf)
                parts.append(bits)
                parts.append(vals)
                nnz.append(vals.shape[0])
            else:
                parts.append(a)
        size = self.fixed_bytes
        for x in parts:
            size += (x.nbytes + ALIGN - 1) & ~(ALIGN - 1)
        if out is None:
            buf = np.zeros(size, np.uint8)
        else:
            if out.dtype != np.uint8 or out.ndim != 1:
                raise TypeError("out must be a 1-D uint8 array")
            if len(out) < size:
                raise ValueError(f"row needs {size} bytes, the buffer holds {len(out)}")
            buf = out
            buf[:size] = 0  # the padding bytes
        mv = memoryview(buf)
        self._hdr_struct.pack_into(mv, 0, *lens, *nnz)
        for f, st, o in self._row_structs:
            v = row[f.name]
            if st is not None and type(v) is int:
                try:
                    st.pack_into(mv, o, v)
                except struct.error:
                    raise ValueError(f"field {f.name!r}: {v!r} does not fit {f.dtype}") from None
            else:
                a = _row_value(f, v)
                buf[o : o + a.nbytes] = a.reshape(-1).view(np.uint8)
        o = self.fixed_bytes
        for x in parts:
            nb = x.nbytes
            if nb:
                mv[o : o + nb] = memoryview(x).cast("B")
            o += (nb + ALIGN - 1) & ~(ALIGN - 1)
        assert o == size, (o, size)
        return size if out is not None else buf

    def encode_many(
        self, rows: Sequence[Mapping[str, Any]], outs: Sequence[np.ndarray]
    ) -> list[int]:
        """Encode several rows into their buffers; return the blob sizes.

        The same blobs as `encode`, row by row, but the numpy work (compare,
        mask, pack) runs once over all rows, so a worker that steps many envs
        pays it once per round. `outs[i]` receives row `i`.
        """
        k = len(rows)
        if k != len(outs):
            raise ValueError(f"{k} rows but {len(outs)} buffers")
        if k <= 1:
            return [self.encode(rows[0], out=outs[0])] if k else []
        ns = len(self.streams)
        lens: list[list[int]] = [[-1] * ns for _ in range(k)]
        # Per segment: (bytes of all rows, per-row byte offsets), or for a raw
        # field (None, per-row byte views).
        segs: list[tuple[memoryview | None, list[Any]]] = []
        nnz: list[list[int]] = [[] for _ in range(k)]
        for name, dt, tail, si, sparse, ut, fb, zf in self._sfp:
            arrs = []
            for i, r in enumerate(rows):
                a = r[name]
                if not isinstance(a, np.ndarray) or a.dtype != dt:
                    got = getattr(a, "dtype", type(a).__name__)
                    raise TypeError(f"field {name!r}: expected a {dt} array, got {got}")
                if a.shape[1:] != tail:
                    raise ValueError(f"field {name!r}: shape {a.shape} is not [length, *{tail}]")
                n = a.shape[0]
                li = lens[i]
                if li[si] < 0:
                    li[si] = n
                elif li[si] != n:
                    raise ValueError(
                        f"stream {self.streams[si]!r}: field {name!r} has length {n}, not {li[si]}"
                    )
                arrs.append(a)
            if not sparse:
                # Raw: each row's array is copied straight into its blob.
                segs.append((None, [_bytes(np.ascontiguousarray(x)) for x in arrs]))
                continue
            tok = [0]
            for a in arrs:
                tok.append(tok[-1] + a.shape[0])
            cat = np.concatenate(arrs)
            bits, vals, pos = _sparse_parts(cat, ut, fb, zf)
            vo = np.searchsorted(pos, np.asarray(tok, np.int64) * tail[0]).tolist()
            for i in range(k):
                nnz[i].append(vo[i + 1] - vo[i])
            ub = bits.shape[1]
            segs.append((_bytes(bits), [t * ub for t in tok]))
            it = vals.dtype.itemsize
            segs.append((_bytes(vals), [v * it for v in vo]))
        sizes = []
        a8 = ALIGN - 1
        for i in range(k):
            size = self.fixed_bytes
            for src, offs in segs:
                nb = offs[i].nbytes if src is None else offs[i + 1] - offs[i]
                size += (nb + a8) & ~a8
            buf = outs[i]
            if buf.dtype != np.uint8 or buf.ndim != 1:
                raise TypeError("out must be a 1-D uint8 array")
            if len(buf) < size:
                raise ValueError(f"row needs {size} bytes, the buffer holds {len(buf)}")
            buf[:size] = 0
            mv = memoryview(buf)
            self._hdr_struct.pack_into(mv, 0, *lens[i], *nnz[i])
            row = rows[i]
            for f, st, o in self._row_structs:
                v = row[f.name]
                if st is not None and type(v) is int:
                    try:
                        st.pack_into(mv, o, v)
                    except struct.error:
                        raise ValueError(
                            f"field {f.name!r}: {v!r} does not fit {f.dtype}"
                        ) from None
                else:
                    a = _row_value(f, v)
                    buf[o : o + a.nbytes] = a.reshape(-1).view(np.uint8)
            o = self.fixed_bytes
            for src, offs in segs:
                if src is None:
                    x = offs[i]
                    nb = x.nbytes
                    if nb:
                        mv[o : o + nb] = x
                else:
                    b0, b1 = offs[i], offs[i + 1]
                    nb = b1 - b0
                    if nb:
                        mv[o : o + nb] = src[b0:b1]
                o += (nb + a8) & ~a8
            sizes.append(size)
        return sizes

    # ── decode ───────────────────────────────────────────────────────────

    def headers(self, buf: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """`[N, n_hdr]` int32 headers of the rows at byte `offsets` of `buf`."""
        offsets = np.asarray(offsets, np.int64)
        if self.n_hdr == 0:
            return np.zeros((len(offsets), 0), np.int32)
        if len(buf) % 4:
            buf = buf[: len(buf) // 4 * 4]
        words = buf.view(np.int32)
        return words[(offsets // 4)[:, None] + np.arange(self.n_hdr)]

    def decode(self, buf: np.ndarray, offsets: Sequence[int] | np.ndarray | None = None) -> Decoded:
        """Decode rows with numpy, one row at a time (the reference decoder).

        Returns the stream fields concatenated over rows in row order (the
        layout of `combat_ppo.encode.Packed`), the ROW fields stacked, and each
        stream's per-row lengths.
        """
        offs = [0] if offsets is None else [int(x) for x in offsets]
        parts: dict[str, list[np.ndarray]] = {f.name: [] for f in self.fields}
        lens = {s: np.zeros(len(offs), np.int32) for s in self.streams}
        ns = len(self.streams)
        for i, o in enumerate(offs):
            hdr = (
                buf[o : o + 4 * self.n_hdr].view(np.int32) if self.n_hdr else np.zeros(0, np.int32)
            )
            for si, s in enumerate(self.streams):
                lens[s][i] = hdr[si]
            for f in self.row_fields:
                p = o + self.hdr_bytes + self.row_off[f.name]
                nb = f.np_dtype.itemsize * f.cols
                parts[f.name].append(buf[p : p + nb].view(f.np_dtype).reshape(f.shape))
            p = o + self.fixed_bytes
            bits: dict[str, np.ndarray] = {}
            for s in self._segs:
                f = self.fields[s.field]
                n_el = int(hdr[s.stream])
                count = int(hdr[ns + s.sparse]) if s.kind == "vals" else n_el
                nb = count * s.unit
                seg = buf[p : p + nb]
                p += align(nb)
                if s.kind == "raw":
                    parts[f.name].append(seg.view(f.np_dtype).reshape((n_el, *f.shape)))
                elif s.kind == "bits":
                    bits[f.name] = seg.reshape(n_el, s.unit)
                else:
                    m = np.unpackbits(bits[f.name], axis=1, count=f.width, bitorder="little")
                    dense = np.empty((n_el, f.width), f.np_dtype)
                    dense[...] = self._fills[f.name]
                    dense[m.view(bool)] = seg.view(f.np_dtype)
                    parts[f.name].append(dense)
        out: dict[str, np.ndarray] = {}
        for f in self.fields:
            xs = parts[f.name]
            if f.stream == ROW:
                out[f.name] = np.stack(xs) if xs else np.zeros((0, *f.shape), f.np_dtype)
            else:
                out[f.name] = np.concatenate(xs) if xs else np.zeros((0, *f.shape), f.np_dtype)
        return Decoded(out, lens)

    def row_values(self, buf: np.ndarray, offsets: np.ndarray) -> dict[str, np.ndarray]:
        """The ROW fields of the rows at byte `offsets` of `buf` (numpy, vectorized)."""
        offsets = np.asarray(offsets, np.int64)
        out = {}
        for f in self.row_fields:
            dt = f.np_dtype
            it = dt.itemsize
            n = len(buf) // it * it
            words = buf[:n].view(dt)
            base = (offsets + self.hdr_bytes + self.row_off[f.name]) // it
            idx = base[:, None] + np.arange(f.cols) if f.width else base
            out[f.name] = words[idx]
        return out

    def decode_torch(
        self,
        buf: Any,
        offsets: np.ndarray,
        hdr: np.ndarray | None = None,
        host_buf: np.ndarray | None = None,
        up: Any = None,
    ) -> Decoded:
        """Decode many rows at once with torch, on `buf`'s device.

        Args:
            buf: A 1-D `torch.uint8` tensor holding the rows (any device). Its
                length must be a multiple of 8.
            offsets: numpy int64 `[N]`: each row's byte offset in `buf`
                (multiples of 8).
            hdr: numpy `[N, n_hdr]`: the rows' headers. Required when `buf`
                is not on the CPU, so that no device sync is needed; read from
                `buf` otherwise.
            host_buf: The same bytes on the host (numpy). If given (or if `buf`
                is on the CPU), the ROW fields are also returned as numpy in
                `Decoded.host`, so callers never read them back from the device.
            up: `up(np.ndarray) -> tensor on buf's device`, used for the few
                small index arrays (`packed.Uploader` stages them in pinned
                memory). Default: `torch.from_numpy(a).to(device)`.

        All sizes are computed on the host, so the decode enqueues work
        without waiting for the device.
        """
        import torch

        offsets = np.asarray(offsets, np.int64)
        dev = buf.device
        if host_buf is None and dev.type == "cpu":
            host_buf = buf.numpy()
        if hdr is None:
            if host_buf is None:
                raise ValueError("decode_torch on a device needs the host-side headers")
            hdr = self.headers(host_buf, offsets)
        hdr = np.asarray(hdr, np.int64)
        if up is None:

            def up(a: np.ndarray) -> Any:
                return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        ns = len(self.streams)
        seg_sz = self._seg_sizes(hdr)
        seg_start = offsets[:, None] + self.fixed_bytes + (np.cumsum(seg_sz, 1) - seg_sz)
        out: dict[str, Any] = {}
        lens = {s: hdr[:, si].astype(np.int32) for si, s in enumerate(self.streams)}
        host = {} if host_buf is None else self.row_values(host_buf, offsets)
        for f in self.row_fields:
            if f.name in host:
                out[f.name] = up(np.ascontiguousarray(host[f.name]))
                continue
            dt = f.np_dtype
            it = dt.itemsize
            base = (offsets + self.hdr_bytes + self.row_off[f.name]) // it
            idx = base[:, None] + np.arange(f.cols) if f.width else base
            out[f.name] = buf.view(_torch_dtype(dt))[up(idx)]
        # One gather per element size: every segment is a byte range per row.
        # Within a size, raw segments come first, then the masks (size 1) or
        # the values of the sparse fields, in field order, so that each
        # sparse group's masks and values are contiguous slices.
        pieces: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        order = sorted(
            range(len(self._segs)),
            key=lambda j: (self._segs[j].kind != "raw", self._seg_vsize(j), j),
        )
        for j in order:
            sg = self._segs[j]
            f = self.fields[sg.field]
            n_el = hdr[:, sg.stream]
            if sg.kind == "raw":
                it = f.np_dtype.itemsize
                count = n_el * f.cols
            elif sg.kind == "bits":
                it = 1
                count = n_el * sg.unit
            else:
                it = f.np_dtype.itemsize
                count = hdr[:, ns + sg.sparse]
            pieces.setdefault(it, []).append((j, seg_start[:, j] // it, count))
        flat: dict[int, Any] = {}
        for it, ps in pieces.items():
            starts = np.concatenate([p[1] for p in ps])
            counts = np.concatenate([p[2] for p in ps])
            g = _gather(buf.view(_torch_dtype(np.dtype(_INT[it]))), starts, counts, up)
            o = 0
            for j, _st, c in ps:
                n = int(c.sum())
                flat[j] = g.narrow(0, o, n)
                o += n
        for j, sg in enumerate(self._segs):
            if sg.kind == "raw":
                f = self.fields[sg.field]
                total_el = int(hdr[:, sg.stream].sum())
                out[f.name] = flat[j].view(_torch_dtype(f.np_dtype)).view(total_el, *f.shape)
        # Sparse fields, one group per value size: unpack every mask bit at
        # once, find the set bits once, scatter every value once.
        shift = None
        for _vs, js in self._sparse_groups.items():
            masks = [flat[jb] for jb, _jv in js]
            vals = [flat[jv] for _jb, jv in js]
            n_bits = sum(int(m.numel()) * 8 for m in masks)
            n_nz = sum(int(v.numel()) for v in vals)
            m_all = _cat_slices(masks)
            v_all = _cat_slices(vals)
            if shift is None:
                shift = self._shift_tensor(dev)
            bits = ((m_all.unsqueeze(-1) >> shift) & 1).view(-1).bool()
            dense = torch.zeros(n_bits, dtype=v_all.dtype, device=dev)
            if n_nz:
                pos = torch.nonzero_static(bits, size=n_nz).squeeze(1)
                dense[pos] = v_all
            o = 0
            for jb, _jv in js:
                sg = self._segs[jb]
                f = self.fields[sg.field]
                total_el = int(hdr[:, sg.stream].sum())
                w8 = sg.unit * 8
                d = dense.narrow(0, o, total_el * w8).view(total_el, w8)[:, : f.width]
                if self._fill_bits[f.name].any():
                    m = bits.narrow(0, o, total_el * w8).view(total_el, w8)[:, : f.width]
                    d = torch.where(m, d, self._fill_tensor(f.name, dev).view(d.dtype))
                else:
                    d = d.contiguous()
                out[f.name] = d.view(_torch_dtype(f.np_dtype))
                o += total_el * w8
        return Decoded(out, lens, host)

    def _seg_vsize(self, j: int) -> int:
        """The value size of segment `j`'s sparse field (0 for a raw segment)."""
        sg = self._segs[j]
        return 0 if sg.kind == "raw" else self.fields[sg.field].np_dtype.itemsize

    def _shift_tensor(self, dev: Any) -> Any:
        import torch

        key = ("__shift__", str(dev))
        t = self._fill_t.get(key)
        if t is None:
            t = torch.arange(8, dtype=torch.uint8, device=dev)
            self._fill_t[key] = t
        return t

    def _fill_tensor(self, name: str, dev: Any) -> Any:
        import torch

        key = (name, str(dev))
        t = self._fill_t.get(key)
        if t is None:
            t = torch.from_numpy(self._fills[name]).to(dev)
            self._fill_t[key] = t
        return t


@dataclass
class Decoded:
    """Decoded rows: `fields[name]`, per-stream lengths `lens[stream]` and,
    when known on the host, the ROW fields as numpy (`host[name]`)."""

    fields: dict[str, Any]
    lens: dict[str, np.ndarray]
    host: dict[str, np.ndarray] = field(default_factory=dict)

    def row(self, name: str) -> np.ndarray:
        """A ROW field as numpy (from `host` if there, else read back)."""
        if name in self.host:
            return np.asarray(self.host[name])
        v = self.fields[name]
        return np.asarray(v.cpu() if hasattr(v, "cpu") else v)

    def __getitem__(self, name: str) -> Any:
        return self.fields[name]

    def cu(self, stream: str) -> np.ndarray:
        """Row start offsets of `stream` in the concatenated arrays (`[N + 1]`)."""
        c = np.zeros(len(self.lens[stream]) + 1, np.int64)
        np.cumsum(self.lens[stream], out=c[1:])
        return c


def _bytes(x: np.ndarray) -> memoryview:
    """A C-contiguous array's bytes (an empty view if it has no elements)."""
    return memoryview(x).cast("B") if x.size else memoryview(b"")


def _sparse_parts(
    a: np.ndarray, ut: Any, fb: np.ndarray, zero_fill: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(mask [n, ceil(w / 8)] uint8, values, flat cell positions)` of the cells
    of `a` (`[n, w]`, C-contiguous) whose bits differ from `fb`.

    Values are in row-major order, masks in little bit order. (Finding the
    cells costs ~1 ns per cell in numpy, at any batch size, so this is the
    worker's main encode cost; testing 8-byte words at a time was slower.)
    """
    d = a.view(ut) != 0 if zero_fill else a.view(ut) != fb
    pos = np.flatnonzero(d)
    return np.packbits(d, axis=1, bitorder="little"), a.reshape(-1)[pos], pos


def _row_value(f: Field, v: Any) -> np.ndarray:
    a = np.asarray(v)
    if a.shape != f.shape:
        raise ValueError(f"field {f.name!r}: shape {a.shape}, expected {f.shape}")
    if a.dtype == f.np_dtype:
        return np.ascontiguousarray(a)
    c = a.astype(f.np_dtype)
    if f.np_dtype.kind in "biu" and not np.array_equal(c, a):
        raise ValueError(f"field {f.name!r}: {a!r} does not fit {f.dtype}")
    return c


def _torch_dtype(dt: np.dtype) -> Any:
    import torch

    m = {
        np.dtype(np.int8): torch.int8,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.uint16): torch.uint16,  # e2e rows: n_classes, action, mem_tok, ... (NT-3)
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.bool_): torch.bool,
    }
    if dt not in m:
        raise TypeError(f"decode_torch does not support dtype {dt}")
    return m[dt]


def _cat_slices(xs: list[Any]) -> Any:
    """Concatenate adjacent slices of one tensor without a copy (else `torch.cat`)."""
    import torch

    if len(xs) == 1:
        return xs[0]
    a, z = xs[0], xs[-1]
    n = sum(int(x.numel()) for x in xs)
    if (
        all(x.is_contiguous() for x in xs)
        and n
        and a.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
    ):
        first = a.storage_offset()
        if all(
            x.storage_offset() == first + sum(int(y.numel()) for y in xs[:i])
            for i, x in enumerate(xs)
        ):
            return a.as_strided((n,), (1,), first)
    return torch.cat(xs)


def _gather(src: Any, starts: np.ndarray, counts: np.ndarray, up: Any) -> Any:
    """`concat(src[starts[i] : starts[i] + counts[i]] for i)` without a sync."""
    import torch

    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return src[:0]
    shift = np.asarray(starts, np.int64) - (np.cumsum(counts) - counts)
    sc = up(np.stack([shift, counts]))
    idx = torch.repeat_interleave(sc[0], sc[1], output_size=total)
    idx += torch.arange(total, device=src.device)
    return src[idx]


class RowStore:
    """Encoded rows in one growing arena, plus per-row columns (learner side).

    `append` takes rows that sit back to back in a buffer (the batcher's
    staging), so a whole served batch is one copy. Columns hold what the
    server adds after inference (action, log-prob, value, ...).
    """

    def __init__(
        self,
        codec: RowCodec,
        columns: Sequence[Field] = (),
        capacity_bytes: int = 1 << 22,
        capacity_rows: int = 1 << 12,
    ) -> None:
        for c in columns:
            if c.stream != ROW:
                raise ValueError(f"column {c.name!r} must be a ROW field")
        self.codec = codec
        self.columns = tuple(columns)
        self.arena = np.zeros(align(capacity_bytes), np.uint8)
        self.offsets = np.zeros(capacity_rows, np.int64)
        self.sizes = np.zeros(capacity_rows, np.int64)
        self.cols = {c.name: np.zeros((capacity_rows, *c.shape), c.np_dtype) for c in self.columns}
        self.n = 0
        self.used = 0

    def __len__(self) -> int:
        return self.n

    @property
    def nbytes(self) -> int:
        """Bytes held: the used arena plus the per-row bookkeeping and columns."""
        per_row = 16 + sum(c.np_dtype.itemsize * c.cols for c in self.columns)
        return self.used + self.n * per_row

    def _grow(self, rows: int, nbytes: int) -> None:
        if self.used + nbytes > len(self.arena):
            new = np.zeros(align(max(2 * len(self.arena), self.used + nbytes)), np.uint8)
            new[: self.used] = self.arena[: self.used]
            self.arena = new
        if self.n + rows > len(self.offsets):
            cap = max(2 * len(self.offsets), self.n + rows)
            self.offsets = _resized(self.offsets, cap)
            self.sizes = _resized(self.sizes, cap)
            self.cols = {k: _resized(v, cap) for k, v in self.cols.items()}

    def append(
        self, buf: np.ndarray, offsets: np.ndarray, sizes: np.ndarray, **cols: Any
    ) -> np.ndarray:
        """Copy rows (at `offsets` of `buf`) in; return their row ids."""
        offsets = np.asarray(offsets, np.int64)
        sizes = np.asarray(sizes, np.int64)
        k = len(offsets)
        if k == 0:
            return np.zeros(0, np.int64)
        missing = {c.name for c in self.columns} - set(cols)
        if missing:
            raise ValueError(f"append: missing columns {sorted(missing)}")
        total = int(sizes.sum())
        self._grow(k, total)
        lo = int(offsets[0])
        if np.array_equal(offsets, lo + np.cumsum(sizes) - sizes):  # back to back
            self.arena[self.used : self.used + total] = buf[lo : lo + total]
            new_off = self.used + (offsets - lo)
        else:
            new_off = self.used + np.cumsum(sizes) - sizes
            for o, s, d in zip(offsets.tolist(), sizes.tolist(), new_off.tolist(), strict=True):
                self.arena[d : d + s] = buf[o : o + s]
        ids = np.arange(self.n, self.n + k)
        self.offsets[self.n : self.n + k] = new_off
        self.sizes[self.n : self.n + k] = sizes
        for name, v in cols.items():
            if name not in self.cols:
                raise ValueError(f"append: unknown column {name!r}")
            self.cols[name][self.n : self.n + k] = v
        self.n += k
        self.used += total
        return ids

    def column(self, name: str) -> np.ndarray:
        return self.cols[name][: self.n]

    def headers(self, rows: np.ndarray | None = None) -> np.ndarray:
        r = np.arange(self.n) if rows is None else np.asarray(rows, np.int64)
        return self.codec.headers(self.arena, self.offsets[r])

    def decode(self, rows: np.ndarray | None = None) -> Decoded:
        r = np.arange(self.n) if rows is None else np.asarray(rows, np.int64)
        return self.codec.decode(self.arena, self.offsets[r])

    def decode_torch(self, rows: np.ndarray | None = None, device: Any = "cpu") -> Decoded:
        """Decode `rows` on `device` (copies the used arena there once)."""
        import torch

        r = np.arange(self.n) if rows is None else np.asarray(rows, np.int64)
        hdr = self.headers(r)
        host = self.arena[: align(self.used)]
        t = torch.from_numpy(host)
        if torch.device(device).type != "cpu":
            t = t.to(device)
        return self.codec.decode_torch(t, self.offsets[r], hdr, host_buf=host)

    def select(self, rows: np.ndarray) -> RowStore:
        """A new store with `rows` only, in that order (carry-over, compaction)."""
        rows = np.asarray(rows, np.int64)
        s = RowStore(
            self.codec,
            self.columns,
            capacity_bytes=max(ALIGN, int(self.sizes[rows].sum())),
            capacity_rows=max(1, len(rows)),
        )
        s.append(
            self.arena,
            self.offsets[rows],
            self.sizes[rows],
            **{k: v[rows] for k, v in self.cols.items()},
        )
        return s


def _resized(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, *a.shape[1:]), a.dtype)
    out[: len(a)] = a
    return out
