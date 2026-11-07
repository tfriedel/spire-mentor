"""Grouping ready requests into inference batches.

`Batcher.gather` polls the transport, takes up to `max_rows` ready requests,
sorts them by token-length bucket and copies their blobs, back to back, into
a staging buffer. It returns one `Batch` per bucket. Buckets with fewer than
`min_rows` rows ride in the next larger bucket, so the device never runs a
handful of rows at a large padded shape. Without buckets, one `Batch` holds
everything and pads to its longest row.

A `Batch` is the argument of the injected `infer(batch) -> outputs`. It is
pad-free: `buf[offsets[i] : offsets[i] + sizes[i]]` is row `i`'s blob, and
`hdr` holds the header fields. Decoding the blobs (`rows.RowCodec`) is the
inference function's job, so that it can happen on the device.

There are `n_stages` staging buffers (2 by default): batches gathered into one
stage stay valid while the next gather fills the other (double buffering, see
`packed.Pipeline`). With `pin=True` (the default when CUDA is available) the
stages are page-locked, so the host-to-device copy is asynchronous.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .transport import WANT_PROBS, ShmTransport

DEFAULT_BUCKETS = (32, 48, 64, 96, 128, 192, 256)


@dataclass
class Batch:
    """Requests taken together, with their blobs back to back in `buf`."""

    slots: np.ndarray  # int64 [n]: transport slot ids
    seq: np.ndarray  # int64 [n]: request numbers
    rows: np.ndarray  # int64 [n]: transport buffer rows
    flags: np.ndarray  # int64 [n]
    hdr: dict[str, np.ndarray]  # header fields, each [n]
    buf: np.ndarray  # uint8: this batch's part of a staging buffer
    offsets: np.ndarray  # int64 [n]: byte offset of each blob in `buf`
    sizes: np.ndarray  # int64 [n]: blob sizes
    bucket: int  # padded token length (0: pad to the longest row)
    stage: int  # the staging buffer used
    t_gather: float = 0.0  # perf_counter() when gathered
    buf_t: Any = None  # the same bytes as a (pinned) torch tensor, if staged by torch
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.slots)

    @property
    def want_probs(self) -> bool:
        return bool(np.any(self.flags & WANT_PROBS))


def _aligned_empty(n: int) -> np.ndarray:
    raw = np.empty(n + 64, np.uint8)
    o = (-raw.ctypes.data) % 64
    return raw[o : o + n]


class Batcher:
    def __init__(
        self,
        transport: ShmTransport,
        key: str = "n_tok",
        buckets: Sequence[int] | None = DEFAULT_BUCKETS,
        max_rows: int = 1024,
        min_rows: int = 1,
        stage_bytes: int | None = None,
        n_stages: int = 2,
        pin: bool | None = None,
    ) -> None:
        """
        Args:
            transport: The server's `ShmTransport`.
            key: The header field that holds a request's token count.
            buckets: Increasing padded lengths, or None for one mixed batch.
                A longer request gets a bucket of its length rounded up to 64.
            max_rows: The most requests per `gather`.
            min_rows: A bucket with fewer rows joins the next larger bucket.
            stage_bytes: Bytes per staging buffer (default: `max_rows` worst-case
                blobs, capped at 8 MiB, at least one blob). Pinned memory is
                committed, so this stays small: a compact row is ~2 KB, so 8 MiB
                holds ~4,000 typical rows. Requests that do not fit wait.
            n_stages: Staging buffers to rotate through.
            pin: Page-lock the staging buffers (default: if CUDA is available).
        """
        if buckets is not None:
            b = [int(x) for x in buckets]
            if not b or any(x <= 0 for x in b) or b != sorted(set(b)):
                raise ValueError(f"buckets must be increasing positive lengths, got {buckets}")
        if key not in transport.hdr:
            raise ValueError(f"key {key!r} is not a header field of the transport")
        self.t = transport
        self.key = key
        self.buckets = None if buckets is None else np.asarray(buckets, np.int64)
        self.max_rows = int(max_rows)
        self.min_rows = max(1, int(min_rows))
        cap = transport.spec.blob_cap
        if stage_bytes is None:
            stage_bytes = max(cap, min(8 << 20, cap * self.max_rows))
        if stage_bytes < cap:
            raise ValueError(f"stage_bytes {stage_bytes} < one blob's capacity {cap}")
        self.stage_bytes = int(stage_bytes)
        if pin is None:
            pin = _cuda_available()
        self.pinned = bool(pin)
        self.stages: list[np.ndarray] = []
        self.stages_t: list[Any] = []
        for _ in range(n_stages):
            if self.pinned:
                import torch

                t = torch.empty(self.stage_bytes, dtype=torch.uint8, pin_memory=True)
                self.stages_t.append(t)
                self.stages.append(t.numpy())
            else:
                self.stages_t.append(None)
                self.stages.append(_aligned_empty(self.stage_bytes))
        self._next = 0
        self._rot = 0
        self.stats: dict[str, float] = {
            "gathers": 0,
            "batches": 0,
            "rows": 0,
            "overflow": 0,
            "deferred": 0,
            "t_poll": 0.0,
            "t_group": 0.0,
            "t_copy": 0.0,
            "t_reply": 0.0,
        }

    def bucket_of(self, lengths: np.ndarray) -> np.ndarray:
        """The padded length of each token count."""
        lengths = np.asarray(lengths, np.int64)
        if self.buckets is None:
            return np.zeros_like(lengths)
        i = np.searchsorted(self.buckets, lengths, side="left")
        over = i >= len(self.buckets)
        out = self.buckets[np.minimum(i, len(self.buckets) - 1)]
        if over.any():
            out = out.copy()
            out[over] = (lengths[over] + 63) // 64 * 64
        return out

    def ready(self) -> int:
        return self.t.pending()

    def gather(self, stage: int | None = None) -> list[Batch]:
        """Take ready requests into staging buffer `stage` (default: the next)."""
        t0 = time.perf_counter()
        st = self.stats
        slots = self.t.poll()
        if slots.size == 0:
            st["t_poll"] += time.perf_counter() - t0
            return []
        if stage is None:
            stage = self._next
            self._next = (self._next + 1) % len(self.stages)
        if slots.size > self.max_rows:
            # Rotate the start so no worker's slots are always cut first.
            r = self._rot % slots.size
            slots = np.concatenate([slots[r:], slots[:r]])[: self.max_rows]
            self._rot += self.max_rows
            st["deferred"] += 1
        seq, rows = self.t.peek(slots)
        nb = self.t.n_bytes[rows]
        key = self.t.hdr[self.key][rows].astype(np.int64)
        t1 = time.perf_counter()
        if self.buckets is not None:
            bk = self.bucket_of(key)
            order = np.argsort(bk, kind="stable")
            bk = bk[order]
        else:
            order = np.arange(len(slots))
            bk = np.zeros(len(slots), np.int64)
        nb = nb[order]
        cum = np.cumsum(nb)
        keep = int(np.searchsorted(cum, self.stage_bytes, side="right"))
        if keep == 0:
            raise RuntimeError(f"a {int(nb[0])}-byte request does not fit a staging buffer")
        if keep < len(order):
            st["deferred"] += 1
            order, nb, bk, cum = order[:keep], nb[:keep], bk[:keep], cum[:keep]
        slots, seq, rows = slots[order], seq[order], rows[order]
        self.t.mark_taken(slots, seq)
        flags = self.t.flags[rows]
        hdr = {k: a[rows] for k, a in self.t.hdr.items()}
        offs = cum - nb
        t2 = time.perf_counter()
        buf = self.stages[stage]
        blob = self.t.blob
        for r, o, n in zip(rows.tolist(), offs.tolist(), nb.tolist(), strict=True):
            buf[o : o + n] = blob[r, :n]
        t3 = time.perf_counter()
        groups = self._groups(bk)
        out = []
        now = time.perf_counter()
        buf_t = self.stages_t[stage]
        for lo, hi, b in groups:
            blo, bhi = int(offs[lo]), int(cum[hi - 1])
            out.append(
                Batch(
                    slots=slots[lo:hi],
                    seq=seq[lo:hi],
                    rows=rows[lo:hi],
                    flags=flags[lo:hi],
                    hdr={k: v[lo:hi] for k, v in hdr.items()},
                    buf=buf[blo:bhi],
                    offsets=offs[lo:hi] - blo,
                    sizes=nb[lo:hi],
                    bucket=int(b),
                    stage=stage,
                    t_gather=now,
                    buf_t=None if buf_t is None else buf_t[blo:bhi],
                )
            )
        if self.buckets is not None:
            st["overflow"] += int(np.count_nonzero(bk > self.buckets[-1]))
        st["gathers"] += 1
        st["batches"] += len(out)
        st["rows"] += len(slots)
        st["t_poll"] += t1 - t0
        st["t_group"] += (t2 - t1) + (time.perf_counter() - t3)
        st["t_copy"] += t3 - t2
        return out

    def _groups(self, bk: np.ndarray) -> list[tuple[int, int, int]]:
        """`(lo, hi, bucket)` runs of equal bucket; small runs join the next one."""
        if len(bk) == 0:
            return []
        cuts = np.flatnonzero(np.diff(bk)) + 1
        starts = [0, *cuts.tolist()]
        ends = [*cuts.tolist(), len(bk)]
        out: list[tuple[int, int, int]] = []
        lo = None
        for s, e in zip(starts, ends, strict=True):
            if lo is None:
                lo = s
            if e - lo >= self.min_rows:
                out.append((lo, e, int(bk[e - 1])))
                lo = None
        if lo is not None:  # the tail is small: merge it into the last group
            if out:
                plo, _, _ = out.pop()
                out.append((plo, len(bk), int(bk[-1])))
            else:
                out.append((lo, len(bk), int(bk[-1])))
        return out

    def reply(self, batch: Batch, out: Mapping[str, np.ndarray]) -> None:
        t0 = time.perf_counter()
        self.t.reply(batch.slots, batch.rows, batch.seq, dict(out), batch.flags)
        self.stats["t_reply"] += time.perf_counter() - t0

    def serve_once(self, infer: Callable[[Batch], Mapping[str, np.ndarray]]) -> int:
        """Gather, infer and reply synchronously; return the requests served."""
        n = 0
        for b in self.gather():
            self.reply(b, infer(b))
            n += b.n
        return n


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
