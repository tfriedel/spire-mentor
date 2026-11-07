"""Packed inference: pad-free batches and a double-buffered serving loop.

A batch from `batcher.Batcher` is pad-free: its rows sit back to back, and
`cu_seqlens` gives each row's token range. The inference function decides
how to run it: padded to the batch's length bucket, or packed with
variable-length attention (`cppo.packed_forward`).

An *engine* runs `infer(batch)`:

* `SyncEngine` calls an `infer` that returns numpy outputs, at launch. Use it
  on the CPU, in tests, and with fake models.
* `TorchEngine` enqueues an `infer` that returns device tensors on its own
  CUDA stream, copies the outputs into pinned host buffers asynchronously and
  records an event. `result` waits for that event only.

`Pipeline.step` serves one round. With `dbuf=True` it gathers (and launches)
batch `k + 1` before it waits for batch `k`, so the CPU packs the next batch
while the GPU runs the current one. The batcher's two staging buffers make
that safe: batch `k`'s buffer is not reused until `k` is complete.

`Uploader` stages the few small host arrays a decode needs (index ranges,
lengths) in pinned memory, so their host-to-device copies are asynchronous
as well. A pageable copy may wait for the stream, which would undo the
overlap.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .batcher import Batch, Batcher


def cu_seqlens(lens: np.ndarray) -> np.ndarray:
    """Row start offsets `[N + 1]` (int32) of rows with token counts `lens`."""
    lens = np.asarray(lens, np.int64)
    out = np.zeros(len(lens) + 1, np.int32)
    np.cumsum(lens, out=out[1:])
    return out


def pad_ratio(lens: np.ndarray, t_pad: int = 0) -> float:
    """Padded tokens over real tokens when rows of `lens` pad to `t_pad`
    (0: to the longest row)."""
    lens = np.asarray(lens, np.int64)
    total = int(lens.sum())
    if total == 0:
        return 1.0
    t = int(t_pad) or int(lens.max())
    return len(lens) * t / total


class Uploader:
    """Small host arrays to a device through one pinned buffer.

    `up(a)` copies `a` into the buffer and returns it on the device. The copy
    is asynchronous on a CUDA device. Call `reset` only when everything
    uploaded since the last reset has arrived (`TorchEngine` does that after
    the batch's event). On the CPU `up` returns `torch.from_numpy(a)`.
    """

    def __init__(self, device: Any, nbytes: int = 1 << 20) -> None:
        import torch

        self.device = torch.device(device)
        self.pinned = self.device.type == "cuda"
        self.np = np.zeros(0, np.uint8)
        self.t = None
        if self.pinned:
            self.t = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
            self.np = self.t.numpy()
        self.used = 0
        self.spills = 0  # uploads that did not fit and went through pageable memory

    def reset(self) -> None:
        self.used = 0

    def __call__(self, a: np.ndarray) -> Any:
        import torch

        from .rows import _torch_dtype

        a = np.ascontiguousarray(a)
        if not self.pinned:
            return torch.from_numpy(a).to(self.device)
        o = (self.used + 63) // 64 * 64
        n = a.nbytes
        if o + n > len(self.np):
            self.spills += 1
            return torch.from_numpy(a).to(self.device)
        self.np[o : o + n] = a.reshape(-1).view(np.uint8)
        self.used = o + n
        assert self.t is not None
        h = self.t[o : o + n].view(_torch_dtype(a.dtype)).view(a.shape)
        return h.to(self.device, non_blocking=True)


@dataclass
class Pending:
    """A launched batch."""

    batch: Batch
    out: Any  # numpy outputs (SyncEngine) or pinned host tensors (TorchEngine)
    event: Any = None
    host: dict[str, Any] | None = None
    up: Uploader | None = None
    t_launch: float = 0.0


class SyncEngine:
    """Runs `infer(batch) -> {name: numpy array}` when a batch is launched."""

    def __init__(self, infer: Callable[[Batch], Mapping[str, np.ndarray]]) -> None:
        self.infer = infer

    def launch(self, batch: Batch) -> Pending:
        t = time.perf_counter()
        return Pending(batch, dict(self.infer(batch)), t_launch=t)

    def ready(self, p: Pending) -> bool:
        return True

    def result(self, p: Pending) -> dict[str, np.ndarray]:
        return p.out

    def release(self, p: Pending) -> None:
        pass


class TorchEngine:
    """Runs `infer(batch) -> {name: device tensor}` on a CUDA stream of its own.

    `infer` must not synchronize with the device (for `cppo.CppoInfer` pass
    `CppoInfer.device_outputs`). It reads `batch.extra["up"]`, an `Uploader`
    for its small host arrays. Outputs are copied to pinned host buffers that
    are reused once `release` returns them.
    """

    def __init__(
        self,
        infer: Callable[[Batch], Mapping[str, Any]],
        device: Any = "cuda",
        aux_bytes: int = 1 << 20,
        max_rows: int = 1024,
    ) -> None:
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TorchEngine needs a CUDA device; use SyncEngine on the CPU")
        self.infer = infer
        self.stream = torch.cuda.Stream(self.device)
        self.aux_bytes = aux_bytes
        self.max_rows = max_rows
        self._ups: list[Uploader] = []
        self._hosts: list[dict[str, Any]] = []
        self._events: list[Any] = []

    def launch(self, batch: Batch) -> Pending:
        import torch

        t = time.perf_counter()
        up = self._ups.pop() if self._ups else Uploader(self.device, self.aux_bytes)
        up.reset()
        batch.extra["up"] = up
        host = self._hosts.pop() if self._hosts else {}
        ev = self._events.pop() if self._events else torch.cuda.Event()
        # The work must wait for anything the default stream did first (weights).
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            out = self.infer(batch)
            res = {}
            for k, v in out.items():
                h = host.get(k)
                if (
                    h is None
                    or h.shape[0] < v.shape[0]
                    or h.shape[1:] != v.shape[1:]
                    or h.dtype != v.dtype
                ):
                    rows = max(int(v.shape[0]), self.max_rows)
                    h = torch.empty((rows, *v.shape[1:]), dtype=v.dtype, pin_memory=True)
                    host[k] = h
                hv = h[: v.shape[0]]
                hv.copy_(v, non_blocking=True)
                res[k] = hv
            ev.record(self.stream)
        return Pending(batch, res, ev, host, up, t)

    def ready(self, p: Pending) -> bool:
        return bool(p.event.query())

    def result(self, p: Pending) -> dict[str, np.ndarray]:
        p.event.synchronize()
        return {k: v.numpy() for k, v in p.out.items()}

    def release(self, p: Pending) -> None:
        if p.up is not None:
            self._ups.append(p.up)
        if p.host is not None:
            self._hosts.append(p.host)
        if p.event is not None:
            self._events.append(p.event)
        p.batch.extra.pop("up", None)


class Pipeline:
    """Gathers, launches and replies, optionally one batch ahead (`dbuf`).

    `on_done(batch, outputs)` runs after each reply is published, for example
    to append the batch's rows and outputs to a `rows.RowStore`. The batch's
    staging buffer stays valid until the next `step` returns.
    """

    def __init__(
        self,
        batcher: Batcher,
        engine: Any,
        dbuf: bool = True,
        on_done: Callable[[Batch, dict[str, np.ndarray]], None] | None = None,
    ) -> None:
        if dbuf and len(batcher.stages) < 2:
            raise ValueError("double buffering needs a batcher with two staging buffers")
        self.batcher = batcher
        self.engine = engine
        self.dbuf = dbuf
        self.on_done = on_done
        self.pending: list[Pending] = []
        self.stats: dict[str, float] = {
            "steps": 0,
            "served": 0,
            "t_gather": 0.0,
            "t_launch": 0.0,
            "t_wait": 0.0,
            "t_reply": 0.0,
            "t_done": 0.0,
        }

    def step(self) -> int:
        """One round; returns the requests replied to."""
        st = self.stats
        t0 = time.perf_counter()
        new = self.batcher.gather()
        t1 = time.perf_counter()
        launched = [self.engine.launch(b) for b in new]
        t2 = time.perf_counter()
        st["t_gather"] += t1 - t0
        st["t_launch"] += t2 - t1
        st["steps"] += 1
        if self.dbuf:
            done, self.pending = self.pending, launched
        else:
            done = launched
        return self._finish(done)

    def flush(self) -> int:
        """Complete the batch launched last (with `dbuf`)."""
        done, self.pending = self.pending, []
        return self._finish(done)

    def _finish(self, ps: list[Pending]) -> int:
        st = self.stats
        n = 0
        for p in ps:
            t0 = time.perf_counter()
            out = self.engine.result(p)
            t1 = time.perf_counter()
            self.batcher.reply(p.batch, out)
            t2 = time.perf_counter()
            if self.on_done is not None:
                self.on_done(p.batch, out)
            self.engine.release(p)
            t3 = time.perf_counter()
            st["t_wait"] += t1 - t0
            st["t_reply"] += t2 - t1
            st["t_done"] += t3 - t2
            n += p.batch.n
        st["served"] += n
        return n

    def run(
        self,
        until: Callable[[], bool],
        idle: Callable[[], None] | None = None,
    ) -> int:
        """`step` until `until()`; call `idle()` after a round with nothing to do."""
        n = 0
        while not until():
            k = self.step()
            n += k
            if k == 0 and not self.pending and idle is not None:
                idle()
        return n + self.flush()
