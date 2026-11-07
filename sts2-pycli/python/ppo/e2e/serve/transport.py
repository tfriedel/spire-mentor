"""Shared-memory slots between rollout workers and the inference server.

One `SharedMemory` block holds one slot per `(worker, env)`. A slot carries
one request at a time: a few header values and one encoded row (a `rows`
blob), written by the worker. The reply goes back in the same slot: the
chosen class, its log-prob, the value and, when the request asks for it, the
full option distribution. Nothing on this path is pickled, and no memory is
allocated per request.

Each slot has two buffers, used in turn by request number (`seq & 1`). The
worker writes request `k` to buffer `k & 1` and then publishes `k` in
`req_seq`. The server reads it, writes the reply to the same buffer and
publishes `k` in `rep_seq`. Request `k + 1` goes to the other buffer, so the
server may keep reading request `k`'s buffer until it publishes reply `k + 1`,
and a worker may keep reading reply `k` until it publishes request `k + 2`.

**Memory order.** A publish is a plain aligned 8-byte store, made after the
data stores in program order; a read checks the sequence number before it
reads the data. That is enough on x86-64, whose stores and loads are not
reordered with others of their kind (TSO). On other CPUs it is not, so
`ShmTransport` refuses to start there unless `allow_weak_memory=True`.

Workers attach by name from a `TransportHandle` (picklable, spawn-safe) with
`track=False`, so a worker's exit never unlinks the block. The server owns it
and unlinks it in `close`.

Waiting: the server polls `req_seq` with one vectorized compare. A worker
polls its own replies and, when none is ready, spins briefly and then sleeps
with an exponential backoff (`WorkerPort.wait`). There are no syscalls on the
hot path.
"""

from __future__ import annotations

import os
import platform
import time
import uuid
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .rows import Field

WANT_PROBS = 1  # request flag: also reply with the full option distribution
MAGIC = 0x53_54_53_32_53_52_56_31  # "STS2SRV1"

# control words
_C_MAGIC, _C_HASH, _C_STOP, _C_BEAT = 0, 1, 2, 3
_N_CTRL = 8

DEFAULT_HEADER = (Field("n_tok", "int32"),)
DEFAULT_REPLY = (
    Field("action", "int32"),
    Field("logp", "float32"),
    Field("value", "float32"),
    Field("probs", "float32", 64),
)


@dataclass(frozen=True)
class SlotSpec:
    """What a slot carries.

    Attributes:
        blob_bytes: Capacity of one request's encoded row (rounded up to 64).
        header: Scalar request fields the server reads without decoding the
            blob (the batcher's bucket key lives here). `n_bytes` and `flags`
            are always present and need no entry.
        reply: Reply fields, each a scalar or a `[width]` vector.
        probs: The reply field written only for requests with `WANT_PROBS`
            ("" if none).
    """

    blob_bytes: int
    header: tuple[Field, ...] = DEFAULT_HEADER
    reply: tuple[Field, ...] = DEFAULT_REPLY
    probs: str = "probs"

    def __post_init__(self) -> None:
        names = [f.name for f in self.header]
        if {"n_bytes", "flags"} & set(names) or len(set(names)) != len(names):
            raise ValueError(f"bad header fields {names}")
        for f in (*self.header, *self.reply):
            if f.stream:
                raise ValueError(f"slot field {f.name!r} must be a ROW field")
        for f in self.header:
            if f.width:
                raise ValueError(f"header field {f.name!r} must be a scalar")
        rnames = [f.name for f in self.reply]
        if len(set(rnames)) != len(rnames):
            raise ValueError(f"duplicate reply fields {rnames}")
        if self.probs and self.probs not in rnames:
            raise ValueError(f"probs field {self.probs!r} is not a reply field")

    @property
    def blob_cap(self) -> int:
        return (int(self.blob_bytes) + 63) // 64 * 64


def _layout(
    spec: SlotSpec, n_workers: int, n_envs: int
) -> tuple[int, tuple[tuple[str, int, tuple[int, ...], str], ...], int]:
    """`(stride, [(name, offset, shape, dtype)], size)` of the block."""
    stride = (n_envs + 7) // 8 * 8  # one worker's slots fill whole cache lines
    n_slots = n_workers * stride
    n_rows = 2 * n_slots
    arrays: list[tuple[str, tuple[int, ...], np.dtype]] = [
        ("ctrl", (_N_CTRL,), np.dtype(np.int64)),
        ("req_seq", (n_slots,), np.dtype(np.int64)),
        ("rep_seq", (n_slots,), np.dtype(np.int64)),
        ("n_bytes", (n_rows,), np.dtype(np.int64)),
        ("flags", (n_rows,), np.dtype(np.int64)),
    ]
    arrays += [(f"h_{f.name}", (n_rows,), f.np_dtype) for f in spec.header]
    arrays += [(f"r_{f.name}", (n_rows, *f.shape), f.np_dtype) for f in spec.reply]
    arrays.append(("blob", (n_rows, spec.blob_cap), np.dtype(np.uint8)))
    out = []
    off = 0
    for name, shape, dt in arrays:
        out.append((name, off, shape, dt.str))
        off += (int(np.prod(shape, dtype=np.int64)) * dt.itemsize + 63) // 64 * 64
    return stride, tuple(out), max(off, 64)


def _layout_hash(layout: tuple[Any, ...]) -> int:
    import hashlib

    h = hashlib.sha256(repr(layout).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def _views(buf: Any, layout: tuple[Any, ...]) -> dict[str, np.ndarray]:
    return {
        name: np.ndarray(shape, np.dtype(dt), buffer=buf, offset=off)
        for name, off, shape, dt in layout
    }


def _tso() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64", "x64", "i686", "i386", "x86")


@dataclass(frozen=True)
class TransportHandle:
    """Everything a worker needs to attach (picklable; pass it to a spawn worker)."""

    name: str
    spec: SlotSpec
    n_workers: int
    n_envs: int
    stride: int
    layout: tuple[tuple[str, int, tuple[int, ...], str], ...]
    size: int


class ShmTransport:
    """The server's side: owns the block, polls requests, writes replies."""

    def __init__(
        self,
        spec: SlotSpec,
        n_workers: int,
        n_envs: int,
        name: str | None = None,
        allow_weak_memory: bool = False,
    ) -> None:
        if not (_tso() or allow_weak_memory):
            raise RuntimeError(
                f"ShmTransport relies on x86-64 store ordering; this CPU is {platform.machine()!r}"
            )
        if n_workers < 1 or n_envs < 1:
            raise ValueError("need at least one worker and one env")
        self.spec = spec
        self.n_workers, self.n_envs = n_workers, n_envs
        self.stride, self.layout, size = _layout(spec, n_workers, n_envs)
        name = name or f"sts2srv_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        self.size = size
        # A new block is zero-filled by the OS; touching it here would commit every page.
        self.v = _views(self.shm.buf, self.layout)
        self.v["ctrl"][_C_HASH] = _layout_hash(self.layout)
        self.v["ctrl"][_C_MAGIC] = MAGIC
        self.req_seq = self.v["req_seq"]
        self.rep_seq = self.v["rep_seq"]
        self.n_bytes = self.v["n_bytes"]
        self.flags = self.v["flags"]
        self.blob = self.v["blob"]
        self.hdr = {f.name: self.v[f"h_{f.name}"] for f in spec.header}
        self.rep = {f.name: self.v[f"r_{f.name}"] for f in spec.reply}
        n_slots = len(self.req_seq)
        self._taken = np.zeros(n_slots, np.int64)
        self._ne = np.zeros(n_slots, bool)
        # Slot ids of real envs (the stride pads each worker to 8 slots).
        ids = np.arange(n_slots)
        self.valid = (ids % self.stride) < n_envs
        self._closed = False

    # ── wiring ───────────────────────────────────────────────────────────

    def handle(self) -> TransportHandle:
        return TransportHandle(
            name=self.shm.name,
            spec=self.spec,
            n_workers=self.n_workers,
            n_envs=self.n_envs,
            stride=self.stride,
            layout=self.layout,
            size=self.size,
        )

    def slot(self, wid: int, env: int) -> int:
        return wid * self.stride + env

    def worker_env(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(wid, env)` of slot ids."""
        s = np.asarray(slots, np.int64)
        return s // self.stride, s % self.stride

    # ── serving ──────────────────────────────────────────────────────────

    def poll(self) -> np.ndarray:
        """Slot ids holding a request the server has not taken yet."""
        np.not_equal(self.req_seq, self._taken, out=self._ne)
        return np.flatnonzero(self._ne)

    def pending(self) -> int:
        """How many requests `poll` would return."""
        np.not_equal(self.req_seq, self._taken, out=self._ne)
        return int(np.count_nonzero(self._ne))

    def peek(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(seq, row)` of the requests in `slots`, without taking them."""
        seq = self.req_seq[slots]
        return seq, slots * 2 + (seq & 1)

    def mark_taken(self, slots: np.ndarray, seq: np.ndarray) -> None:
        self._taken[slots] = seq

    def take(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Take the requests in `slots`: `(seq, row)`; `row` indexes the buffers."""
        seq, row = self.peek(slots)
        self.mark_taken(slots, seq)
        return seq, row

    def reply(
        self,
        slots: np.ndarray,
        rows: np.ndarray,
        seq: np.ndarray,
        out: dict[str, np.ndarray],
        flags: np.ndarray | None = None,
    ) -> None:
        """Write the replies for taken requests, then publish them."""
        for f in self.spec.reply:
            if f.name == self.spec.probs:
                continue
            self.rep[f.name][rows] = out[f.name]
        p = self.spec.probs
        if p and p in out:
            fl = self.flags[rows] if flags is None else flags
            want = (fl & WANT_PROBS) != 0
            if want.any():
                src = np.asarray(out[p])[want]
                k = min(src.shape[1], self.rep[p].shape[1])
                dst_rows = rows[want]
                self.rep[p][dst_rows, :k] = src[:, :k]
                if k < self.rep[p].shape[1]:
                    self.rep[p][dst_rows, k:] = 0
        self.rep_seq[slots] = seq  # publish last

    def beat(self) -> None:
        self.v["ctrl"][_C_BEAT] = time.monotonic_ns()

    def stop(self) -> None:
        """Tell every worker to stop (`WorkerPort.stopped`)."""
        self.v["ctrl"][_C_STOP] = 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.v = {}
        self.req_seq = self.rep_seq = self.n_bytes = self.flags = self.blob = None  # type: ignore[assignment]
        self.hdr, self.rep = {}, {}
        _close(self.shm)
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> ShmTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class WorkerPort:
    """One worker's side: its `n_envs` slots.

    `publish` (or `submit`) one request per env, `poll` / `wait` for replies,
    `reply` to read one. An env has at most one request in flight. Envs are
    independent, so a worker can keep half its envs in flight while it steps
    the other half (double buffering at the worker).
    """

    def __init__(self, handle: TransportHandle, wid: int, spin: int = 64) -> None:
        if not 0 <= wid < handle.n_workers:
            raise ValueError(f"worker id {wid} out of range")
        self.shm = shared_memory.SharedMemory(name=handle.name, track=False)
        self.v = _views(self.shm.buf, handle.layout)
        ctrl = self.v["ctrl"]
        if int(ctrl[_C_MAGIC]) != MAGIC or int(ctrl[_C_HASH]) != _layout_hash(handle.layout):
            del ctrl
            self.v = {}
            _close(self.shm)
            raise RuntimeError("shared block layout differs from the handle's")
        self.handle = handle
        self.wid = wid
        self.n_envs = handle.n_envs
        self.base = wid * handle.stride
        b, n = self.base, self.n_envs
        self._ctrl = ctrl
        self._req = self.v["req_seq"]
        self._rep_all = self.v["rep_seq"]
        self._rep_mine = self._rep_all[b : b + n]
        self._nbytes = self.v["n_bytes"]
        self._flags = self.v["flags"]
        self._blob = self.v["blob"]
        spec = handle.spec
        self._hdr = {f.name: self.v[f"h_{f.name}"] for f in spec.header}
        self._repf = {f.name: self.v[f"r_{f.name}"] for f in spec.reply}
        self._scalar = tuple(f.name for f in spec.reply if not f.width)
        self.Reply = namedtuple("Reply", self._scalar)  # type: ignore[misc]
        self._seq = [int(x) for x in self._req[b : b + n]]
        self._seq_np = np.asarray(self._seq, np.int64)
        self._got_np = self._seq_np.copy()
        self.spin = spin

    # ── requests ─────────────────────────────────────────────────────────

    def blob(self, env: int) -> np.ndarray:
        """The writable buffer of `env`'s next request (uint8, `blob_cap` bytes)."""
        k = self._seq[env] + 1
        return self._blob[(self.base + env) * 2 + (k & 1)]

    def publish(self, env: int, n_bytes: int, flags: int = 0, **header: Any) -> int:
        """Publish `env`'s next request (its blob already written); return its number."""
        k0 = self._seq[env]
        if self._got_np[env] != k0:
            raise RuntimeError(f"env {env} already has a request in flight")
        if not 0 <= n_bytes <= self._blob.shape[1]:
            raise ValueError(f"n_bytes {n_bytes} outside [0, {self._blob.shape[1]}]")
        if header and not header.keys() <= self._hdr.keys():
            raise ValueError(f"unknown header fields {sorted(set(header) - set(self._hdr))}")
        k = k0 + 1
        slot = self.base + env
        row = slot * 2 + (k & 1)
        self._nbytes[row] = n_bytes
        self._flags[row] = flags
        for name, a in self._hdr.items():
            a[row] = header.get(name, 0)
        self._req[slot] = k  # publish last
        self._seq[env] = k
        self._seq_np[env] = k
        return k

    def submit(self, env: int, codec: Any, row: Any, flags: int = 0, **header: Any) -> int:
        """Encode `row` with `codec` straight into the slot, then publish."""
        n = codec.encode(row, out=self.blob(env))
        return self.publish(env, n, flags, **header)

    def submit_many(
        self,
        envs: Sequence[int],
        codec: Any,
        rows: Sequence[Any],
        flags: int | Sequence[int] = 0,
        **header: Sequence[Any],
    ) -> None:
        """`submit` for several envs at once (`RowCodec.encode_many`: the numpy
        work runs once per call). Header values are sequences, one per env."""
        envs = [int(e) for e in envs]
        outs = [self.blob(e) for e in envs]
        sizes = codec.encode_many(rows, outs)
        fl = [int(flags)] * len(envs) if isinstance(flags, int) else [int(x) for x in flags]
        for j, e in enumerate(envs):
            self.publish(e, sizes[j], fl[j], **{k: v[j] for k, v in header.items()})

    # ── replies ──────────────────────────────────────────────────────────

    def in_flight(self) -> np.ndarray:
        """Envs with a request whose reply has not been read."""
        return np.flatnonzero(self._got_np != self._seq_np)

    def poll(self) -> np.ndarray:
        """Envs whose reply is ready and not read yet."""
        return np.flatnonzero((self._rep_mine == self._seq_np) & (self._got_np != self._seq_np))

    def wait(self, timeout: float | None = None, max_sleep: float = 1e-3) -> np.ndarray:
        """Block until a reply is ready (or stop, or `timeout`); return `poll()`."""
        r = self.poll()
        if r.size or not self.in_flight().size:
            return r
        t_end = None if timeout is None else time.monotonic() + timeout
        for _ in range(self.spin):
            r = self.poll()
            if r.size:
                return r
        nap = 2e-5
        while True:
            if self._ctrl[_C_STOP]:
                return self.poll()
            time.sleep(nap)
            r = self.poll()
            if r.size:
                return r
            if t_end is not None and time.monotonic() >= t_end:
                return r
            nap = min(max_sleep, nap * 2)

    def reply(self, env: int) -> Any:
        """Read `env`'s reply (a `Reply` of the scalar fields) and mark it read."""
        k = self._seq[env]
        if self._rep_mine[env] != k or self._got_np[env] == k:
            raise RuntimeError(f"env {env} has no unread reply")
        row = (self.base + env) * 2 + (k & 1)
        self._got_np[env] = k
        return self.Reply(*[self._repf[n][row].item() for n in self._scalar])

    def reply_field(self, env: int, name: str) -> np.ndarray:
        """A view of a reply field of `env`'s latest request (valid until its
        request after next is published)."""
        k = self._seq[env]
        return self._repf[name][(self.base + env) * 2 + (k & 1)]

    @property
    def stopped(self) -> bool:
        return bool(self._ctrl[_C_STOP])

    def close(self) -> None:
        self.v = {}
        self._ctrl = self._req = self._rep_all = self._rep_mine = None  # type: ignore[assignment]
        self._nbytes = self._flags = self._blob = None  # type: ignore[assignment]
        self._hdr, self._repf = {}, {}
        _close(self.shm)

    def __enter__(self) -> WorkerPort:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _close(shm: shared_memory.SharedMemory) -> None:
    """Close a mapping; if a caller still holds a view, leave it to process exit."""
    try:
        shm.close()
    except BufferError:
        pass


def worker_ports(handle: TransportHandle, wids: Sequence[int]) -> list[WorkerPort]:
    """Several ports in one process (simulated load, tests)."""
    return [WorkerPort(handle, w) for w in wids]
