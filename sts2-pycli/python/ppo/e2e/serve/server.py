"""The serving stack for a learner, wired from `ServeFlags`.

`Server` owns the shared-memory transport, the batcher and the pipeline. A
learner keeps #1081's structure: it spawns workers, keeps their pipes for
episode ends and control, and calls `Server.step` where #1081's `collect`
did `wait(conns)` / `recv` / `_act` / `send`. See the package docstring for
the worker side.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .batcher import DEFAULT_BUCKETS, Batch, Batcher
from .flags import ServeFlags
from .packed import Pipeline, SyncEngine, TorchEngine
from .rows import Field, RowCodec
from .transport import DEFAULT_HEADER, DEFAULT_REPLY, ShmTransport, SlotSpec, TransportHandle


class Server:
    """Transport + batcher + pipeline.

    Args:
        codec: The row codec the workers encode with.
        n_workers, n_envs: Slots: one per `(worker, env)`.
        engine: A `packed.SyncEngine` or `packed.TorchEngine` (see `cppo_engine`).
        flags: Must have `shm`; `buckets` and `dbuf` are read here.
        caps: Each stream's length cap (sizes the slot blobs), e.g.
            `{"tok": 256, "priv": 64}`.
        header: Scalar request fields; the first must be the bucket key
            (`n_tok`). Add, for example, `Field("ep", "int32")` to tag rows
            with their episode.
        reply: Reply fields (default: action, logp, value, probs[64]).
        on_done: `on_done(batch, outputs)` after each batch's replies.
    """

    def __init__(
        self,
        codec: RowCodec,
        n_workers: int,
        n_envs: int,
        engine: Any,
        flags: ServeFlags,
        caps: Mapping[str, int],
        header: Sequence[Field] = DEFAULT_HEADER,
        reply: Sequence[Field] = DEFAULT_REPLY,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        max_rows: int = 1024,
        min_rows: int = 1,
        pin: bool | None = None,
        on_done: Callable[[Batch, dict[str, np.ndarray]], None] | None = None,
        name: str | None = None,
    ) -> None:
        if not flags.shm:
            raise ValueError("Server is the 'shm' path; with 'shm' off keep #1081's pipes")
        self.codec = codec
        self.flags = flags
        spec = SlotSpec(blob_bytes=codec.max_bytes(caps), header=tuple(header), reply=tuple(reply))
        self.transport = ShmTransport(spec, n_workers, n_envs, name=name)
        try:
            self.batcher = Batcher(
                self.transport,
                key=spec.header[0].name,
                buckets=tuple(buckets) if flags.buckets else None,
                max_rows=max_rows,
                min_rows=min_rows,
                pin=pin,
            )
            self.pipeline = Pipeline(self.batcher, engine, dbuf=flags.dbuf, on_done=on_done)
        except Exception:
            self.transport.close()
            raise

    def handle(self) -> TransportHandle:
        """What each worker process needs (picklable; pass it to the spawn target)."""
        return self.transport.handle()

    def step(self) -> int:
        """Serve one round; return the requests replied to."""
        return self.pipeline.step()

    def flush(self) -> int:
        return self.pipeline.flush()

    def pending(self) -> int:
        """Requests waiting to be gathered."""
        return self.transport.pending()

    def worker_env(self, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.transport.worker_env(slots)

    def stop(self) -> None:
        """Tell the workers to stop (they see `WorkerPort.stopped`)."""
        self.transport.stop()

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def cppo_engine(
    net: Any,
    codec: RowCodec,
    flags: ServeFlags,
    device: Any = "cpu",
    bf16: bool = True,
    greedy: bool = False,
    seed: int = 0,
    max_opts: int = 64,
) -> Any:
    """The engine for a #1081 `CombatCtxNet` policy: `TorchEngine` on CUDA,
    `SyncEngine` on the CPU; `flags.packed` picks the packed forward."""
    import torch

    from .cppo import CppoInfer

    dev = torch.device(device)
    inf = CppoInfer(
        net,
        codec,
        device=dev,
        bf16=bf16,
        greedy=greedy,
        seed=seed,
        packed=flags.packed,
        max_opts=max_opts,
    )
    if dev.type == "cuda":
        return TorchEngine(inf.device_outputs, dev)
    return SyncEngine(inf)
