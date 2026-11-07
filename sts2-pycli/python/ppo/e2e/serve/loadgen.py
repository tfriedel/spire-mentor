"""Simulated workers: many `WorkerPort`s in one process (spawn-safe).

`run_load` drives `len(wids)` workers with `n_envs` envs each. Every env
submits a pre-encoded row, waits for its reply, checks it and submits the
next row, like a rollout worker with an instant environment. It measures the
worker side of the transport and lets one process load the server as 32 x 32
workers would (the host allows at most 2 busy processes for benchmarks).

A reply is checked against `expect_action[i]` (the action a deterministic fake
server returns for row `i`), so a mixed-up slot, buffer or row shows up as a
mismatch.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .transport import WANT_PROBS, TransportHandle, WorkerPort


def run_load(
    handle: TransportHandle,
    wids: list[int],
    flat: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
    n_tok: np.ndarray,
    expect_action: np.ndarray | None = None,
    n_requests: int = 0,
    seconds: float = 0.0,
    probs_every: int = 0,
    seed: int = 0,
    conn: Any = None,
) -> dict[str, Any]:
    """Run until `n_requests` replies were read (all workers together), for
    `seconds`, or until the server says stop. Sends the stats to `conn` too.

    Args:
        flat, offsets, sizes: Encoded rows to cycle through: row `i` is
            `flat[offsets[i] : offsets[i] + sizes[i]]`.
        n_tok: Each row's token count (`[K]`).
        expect_action: The `action` the server must reply for each row.
        probs_every: Ask for the full distribution on every `probs_every`-th
            request (0: never).
    """
    ports = [WorkerPort(handle, w) for w in wids]
    rng = np.random.default_rng(seed)
    k_rows = len(sizes)
    n_envs = handle.n_envs
    nxt = rng.integers(0, k_rows, (len(ports), n_envs))
    sizes_l = [int(x) for x in sizes]
    blobs = [flat[int(o) : int(o) + n] for o, n in zip(offsets, sizes_l, strict=True)]
    ntok_l = [int(x) for x in n_tok]
    exp_l = None if expect_action is None else [int(x) for x in expect_action]
    st = {"served": 0, "mismatch": 0, "t_submit": 0.0, "t_reply": 0.0, "idle_polls": 0}
    sent = 0

    def submit(pi: int, p: WorkerPort, env: int) -> None:
        nonlocal sent
        i = int(nxt[pi, env])
        n = sizes_l[i]
        p.blob(env)[:n] = blobs[i]
        fl = WANT_PROBS if probs_every and sent % probs_every == 0 else 0
        p.publish(env, n, fl, n_tok=ntok_l[i])
        sent += 1

    t0 = time.perf_counter()
    for pi, p in enumerate(ports):
        for env in range(n_envs):
            submit(pi, p, env)
    st["t_submit"] += time.perf_counter() - t0
    t_end = time.perf_counter() + seconds if seconds else None
    done = False
    while not done:
        got = 0
        for pi, p in enumerate(ports):
            r = p.poll()
            if not r.size:
                continue
            ta = time.perf_counter()
            for env in r.tolist():
                rep = p.reply(env)
                i = int(nxt[pi, env])
                if exp_l is not None and rep.action != exp_l[i]:
                    st["mismatch"] += 1
                nxt[pi, env] = (i + 1 + env) % k_rows
                got += 1
            tb = time.perf_counter()
            for env in r.tolist():
                submit(pi, p, env)
            tc = time.perf_counter()
            st["t_reply"] += tb - ta
            st["t_submit"] += tc - tb
        st["served"] += got
        if (
            (n_requests and st["served"] >= n_requests)
            or (t_end is not None and time.perf_counter() >= t_end)
            or ports[0].stopped
        ):
            done = True
        elif not got:
            st["idle_polls"] += 1
    st["elapsed"] = time.perf_counter() - t0
    st["n_ports"] = len(ports)
    for p in ports:
        p.close()
    if conn is not None:
        conn.send(st)
        conn.close()
    return st


def load_process(ctx: Any, handle: TransportHandle, wids: list[int], **kw: Any) -> tuple[Any, Any]:
    """Start `run_load` in a new process of `ctx` (a spawn context); return
    `(process, receiving end of the stats pipe)`."""
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=run_load, args=(handle, wids), kwargs={**kw, "conn": send}, daemon=True)
    p.start()
    send.close()
    return p, recv
