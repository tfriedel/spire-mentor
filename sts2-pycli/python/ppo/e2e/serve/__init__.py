"""S1b serving for the end-to-end trainer: shared-memory transport, length
buckets, packed inference, double buffering and compact rows (WP5b).

Each component sits behind one `ServeFlags` flag; with all flags off a
learner keeps #1081's pipes. `parity` checks every component against #1081's
path and `bench` measures them.

Learner (server) side, keeping #1081's actor-learner structure::

    from ppo.e2e.serve import Field, RowStore, ServeFlags, Server, cppo, cppo_engine

    flags = ServeFlags.parse("all")                       # or "shm,buckets,...", "none"
    codec = cppo.cppo_codec(compact=flags.compact)
    engine = cppo_engine(policy, codec, flags, device="cuda", bf16=True, seed=seed)
    store = RowStore(codec, columns=(Field("action", "int32"), Field("logp", "float32"),
                                     Field("value", "float32")))

    def on_done(batch, out):                              # after the replies are out
        wid, env = srv.worker_env(batch.slots)
        ids = store.append(batch.buf, batch.offsets, batch.sizes, action=out["action"],
                           logp=out["logp"], value=out["value"])
        # map (wid, env, batch.hdr["ep"]) -> ids: the open trajectories

    srv = Server(codec, n_workers, n_envs, engine, flags, caps={"tok": 256, "priv": 64},
                 header=(Field("n_tok", "int32"), Field("ep", "int32")), on_done=on_done)
    handle = srv.handle()                 # spawn args of each worker, next to its Pipe
    while collecting:
        srv.step()                        # was: wait(conns) / recv / _act / send
        ...                               # episode ends still arrive on the pipes
    srv.stop(); srv.close()

Worker side::

    from ppo.e2e.serve import WANT_PROBS, WorkerPort, cppo

    port = WorkerPort(handle, wid)
    port.submit_many(envs, codec, [cppo.decision_row(d) for d in ds],
                     n_tok=[len(d.sec) for d in ds], ep=[ep[e] for e in envs])
    for e in port.wait():                 # envs whose reply is ready
        r = port.reply(e)                 # r.action, r.logp, r.value
    # with flags=WANT_PROBS: port.reply_field(e, "probs") is the distribution
    if port.stopped: ...

`RowStore.select(rows)` hands a finished batch to the update as one arena;
a rank decodes it on its GPU with `RowCodec.decode_torch`, or gets #1081's
`Packed` layout from `cppo.to_packed(store.decode(rows))`.
"""

from .batcher import DEFAULT_BUCKETS, Batch, Batcher
from .flags import ServeFlags
from .packed import Pipeline, SyncEngine, TorchEngine, Uploader, cu_seqlens, pad_ratio
from .rows import RAW, ROW, SPARSE, Decoded, Field, RowCodec, RowStore
from .server import Server, cppo_engine
from .transport import (
    DEFAULT_HEADER,
    DEFAULT_REPLY,
    WANT_PROBS,
    ShmTransport,
    SlotSpec,
    TransportHandle,
    WorkerPort,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_HEADER",
    "DEFAULT_REPLY",
    "RAW",
    "ROW",
    "SPARSE",
    "WANT_PROBS",
    "Batch",
    "Batcher",
    "Decoded",
    "Field",
    "Pipeline",
    "RowCodec",
    "RowStore",
    "ServeFlags",
    "Server",
    "ShmTransport",
    "SlotSpec",
    "SyncEngine",
    "TorchEngine",
    "TransportHandle",
    "Uploader",
    "WorkerPort",
    "cppo_engine",
    "cu_seqlens",
    "pad_ratio",
]
