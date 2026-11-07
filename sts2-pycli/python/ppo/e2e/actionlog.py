"""Action logs (plan §3.2): the option index of every stored decision, compressed.

A segment is a pure function of its start and its stored choices (the engine is
deterministic, and auto-advanced steps are recomputed), so a log is the seed plus one
u8 per stored decision (u16 when any index exceeds 255). `replay` rebuilds the segment
from it and asserts that it reaches the recorded state (`run_fingerprint`). That is also
how a boss-entry state is stored for DEVBOSS: `capture_record` keeps the run's seed, its
log prefix and the step of the Move, and `replay_capture` rebuilds the exact clone. A
practice segment's record carries its parent's capture the same way, plus the reseed.

`LogSink` decides what is kept: every eval run, a 2% sample of training runs, every
segment that reaches the act-3 second boss (for a full run that means the whole run, the
only replayable form of its boss-2 segment), and every capped or error segment (what the
CAPS alarm sends the operator to; caps stay under 0.1% of runs). It stops writing at
10 GB in total. The rollout workers (`actors.worker_main`, `<save_dir>/actions`) and the
evaluator's players (`evaluate.Player`, `<arm>/actions`) offer every segment they end.

Records are framed one by one, so appends need no index and a torn tail loses only the
last record:

    b"E2L1" | codec (1 byte) | u32 payload length | payload
    payload (after decompression) = u32 header length | header JSON | index bytes

Deviation, recorded: the campaign venv has no zstd binding (`zstandard` is not
installed, and adding it needs `uv`, which the campaign does not run), so the codec is
zstd when `zstandard` imports and stdlib LZMA otherwise. The codec byte says which.
"""

from __future__ import annotations

import json
import lzma
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"E2L1"
CODEC_ZSTD = b"z"
CODEC_LZMA = b"x"
FORMAT_VERSION = 1

CAP_BYTES = 10 * 10**9
#: The outcome and reason of `worker_error_record`.
WORKER_ERROR = "worker_error"
#: The tail of a worker error's traceback a `worker_error` record keeps.
ERROR_CHARS = 4000
TRAIN_SAMPLE_P = 0.02

try:  # pragma: no cover - depends on the venv
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None


def _compress(raw: bytes) -> tuple[bytes, bytes]:
    if _zstd is not None:
        return CODEC_ZSTD, _zstd.ZstdCompressor(level=10).compress(raw)
    return CODEC_LZMA, lzma.compress(raw, preset=6)


def _decompress(codec: bytes, payload: bytes) -> bytes:
    if codec == CODEC_ZSTD:
        if _zstd is None:
            raise RuntimeError("a zstd action log needs the zstandard module")
        return _zstd.ZstdDecompressor().decompress(payload)
    if codec == CODEC_LZMA:
        return lzma.decompress(payload)
    raise ValueError(f"unknown action-log codec {codec!r}")


@dataclass
class LogRecord:
    """One segment's start (header) and its stored choices (`idx`)."""

    header: dict[str, Any]
    idx: list[int]

    def encode(self) -> bytes:
        top = max(self.idx, default=0)
        if top > 0xFFFF:
            raise ValueError(f"option index {top} does not fit u16")
        width = 1 if top <= 0xFF else 2
        arr = np.asarray(self.idx, dtype=np.uint8 if width == 1 else "<u2").tobytes()
        head = dict(self.header, width=width, n=len(self.idx), format=FORMAT_VERSION)
        hb = json.dumps(head, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raw = struct.pack("<I", len(hb)) + hb + arr
        codec, comp = _compress(raw)
        return MAGIC + codec + struct.pack("<I", len(comp)) + comp


def decode(buf: bytes, off: int = 0) -> tuple[LogRecord, int]:
    """Decode the record at `off`; return it and the offset after it."""
    if buf[off : off + 4] != MAGIC:
        raise ValueError(f"no action-log record at offset {off}")
    codec = buf[off + 4 : off + 5]
    (n,) = struct.unpack_from("<I", buf, off + 5)
    end = off + 9 + n
    if end > len(buf):
        raise ValueError("truncated action-log record")
    raw = _decompress(codec, buf[off + 9 : end])
    (hn,) = struct.unpack_from("<I", raw, 0)
    head = json.loads(raw[4 : 4 + hn].decode("utf-8"))
    dt = np.uint8 if head["width"] == 1 else np.dtype("<u2")
    idx = np.frombuffer(raw[4 + hn :], dtype=dt).astype(int).tolist()
    if len(idx) != head["n"]:
        raise ValueError(f"record holds {len(idx)} indices, header says {head['n']}")
    return LogRecord(header=head, idx=idx), end


def read_records(path: str | os.PathLike[str]) -> Iterator[LogRecord]:
    """Every complete record in `path`; a torn final record is skipped."""
    buf = Path(path).read_bytes()
    off = 0
    while off < len(buf):
        try:
            rec, off = decode(buf, off)
        except ValueError:
            return
        yield rec


# ── records from the env ─────────────────────────────────────────────────────


def _stamps() -> dict[str, Any]:
    import sts2_rs

    try:
        build = str(sts2_rs.build_git())
    except AttributeError:  # an old wheel without the getter; anything else propagates
        build = "unknown"
    return {"game_version": "v0.111.0", "wheel_build": build}


def terminal_record(term: Any, *, reason: str, extra: dict[str, Any] | None = None) -> LogRecord:
    """The log record of a finished segment (`env.E2ETerminal`)."""
    head: dict[str, Any] = {
        "seed": int(term.seed),
        "segment": term.segment.header(),
        "outcome": term.outcome,
        "cap": term.cap,
        "rows": int(term.rows),
        "steps": int(term.steps),
        "floor": int(term.floor),
        "fingerprint": term.fingerprint,
        "reason": reason,
        **_stamps(),
    }
    if extra:
        head["extra"] = extra
    return LogRecord(header=head, idx=list(term.log_idx))


def worker_error_record(env: Any, err: str, *, extra: dict[str, Any] | None = None) -> LogRecord:
    """The log record of a segment a worker ended on an exception of its own (a tokenizer
    `SchemaError`, ...) while `env` waited on a decision: the segment's seed and start, the
    choices so far (which replay to that decision) and the run's fingerprint there.
    `replay_open` rebuilds it. Outcome `worker_error`, kept apart from the env's `error`."""
    from .env import run_fingerprint

    head: dict[str, Any] = {
        "seed": int(env.seed),
        "segment": env.segment.header(),
        "outcome": WORKER_ERROR,
        "cap": None,
        "rows": int(env.rows),
        "steps": len(env.log_idx),
        "floor": int(env.rm.total_floor()),
        "fingerprint": run_fingerprint(env.rm),
        "reason": WORKER_ERROR,
        "error": str(err)[-ERROR_CHARS:],
        **_stamps(),
    }
    if extra:
        head["extra"] = extra
    return LogRecord(header=head, idx=list(env.log_idx))


def capture_record(cap: Any, *, extra: dict[str, Any] | None = None) -> LogRecord:
    """A practice capture (`env.Capture`) as a replayable record: the run's seed, its log
    prefix and the step of the Move. The robust form of a DEVBOSS boss-entry state."""
    from .env import run_fingerprint

    head: dict[str, Any] = {
        "seed": int(cap.seed),
        "segment": {"kind": "capture", "cell": cap.cell},
        "capture_steps": int(cap.steps),
        "move": dict(cap.move),
        "fingerprint": run_fingerprint(cap.snap),
        "reason": "capture",
        **_stamps(),
    }
    if extra:
        head["extra"] = extra
    return LogRecord(header=head, idx=list(cap.idx))


# ── replay ───────────────────────────────────────────────────────────────────


class ReplayMismatch(AssertionError):
    pass


def replay_capture(seed: int, idx: list[int], capture_steps: int) -> Any:
    """Rebuild a capture: replay `idx` from `seed` and return the `env.Capture` taken at
    step `capture_steps`."""
    from .env import E2EDecision, E2EEnv, EnvConfig

    got: dict[int, Any] = {}

    def keep(c: Any) -> None:
        got[int(c.steps)] = c

    env = E2EEnv(int(seed), cfg=EnvConfig(on_capture=keep))
    d = env.reset()
    for ai in idx:
        if not isinstance(d, E2EDecision):
            raise ReplayMismatch(f"seed {seed}: the run ended before the log did")
        d = env.step_action(int(ai))
        if capture_steps in got:
            break
    cap = got.get(int(capture_steps))
    if cap is None:
        raise ReplayMismatch(f"seed {seed}: no capture at step {capture_steps}")
    if list(cap.idx) != [int(i) for i in idx]:
        raise ReplayMismatch(f"seed {seed}: the capture's log prefix differs")
    return cap


def start_from_capture(cap: Any, reseed: int) -> Any:
    """A practice start (plan §3.3): the capture's clone, reseeded, after the Move."""
    rm = cap.snap.clone()
    rm.reseed_run(int(reseed))
    rm.take_action(dict(cap.move))
    return rm


def _segment_env(h: dict[str, Any]) -> Any:
    """A fresh `E2EEnv` at the start of the segment that `h` (a record header) logs."""
    from .env import E2EEnv, EnvConfig, Segment

    seg = h["segment"]
    seed = int(h["seed"])
    if seg["kind"] == "run":
        return E2EEnv(seed)
    cap = replay_capture(
        int(seg["parent_seed"]), list(seg["parent_idx"]), int(seg["capture_steps"])
    )
    rm = start_from_capture(cap, int(seg["reseed"]))
    segment = Segment(
        kind=seg["kind"],
        cell=seg["cell"],
        end=seg["end"],
        parent_seed=int(seg["parent_seed"]),
        parent_idx=tuple(seg["parent_idx"]),
        capture_steps=int(seg["capture_steps"]),
        move=tuple(sorted(seg["move"].items())),
        reseed=int(seg["reseed"]),
        uid=seg.get("uid"),
    )
    return E2EEnv(seed, rm=rm, cfg=EnvConfig(), segment=segment)


def _replay_steps(env: Any, rec: LogRecord) -> Any:
    from .env import E2EDecision

    seed = int(rec.header["seed"])
    d = env.reset()
    for k, ai in enumerate(rec.idx):
        if not isinstance(d, E2EDecision) or d.bootstrap:
            raise ReplayMismatch(f"seed {seed}: the segment ended at row {k} of {len(rec.idx)}")
        d = env.step_action(int(ai))
    return d


def replay(rec: LogRecord) -> Any:
    """Rebuild `rec`'s segment and assert it ends in the recorded state.

    Returns the finished `env.E2ETerminal`. Raises `ReplayMismatch` on any divergence.
    A `worker_error` record has no terminal: use `replay_open`.
    """
    from .env import E2EDecision, run_fingerprint

    h = rec.header
    seed = int(h["seed"])
    if str(h.get("fingerprint", "unavailable")).startswith("unavailable"):
        raise ReplayMismatch(f"seed {seed}: the record has no fingerprint to check")
    if h["segment"]["kind"] == "capture":
        cap = replay_capture(seed, rec.idx, int(h["capture_steps"]))
        fp = run_fingerprint(cap.snap)
        if fp != h["fingerprint"]:
            raise ReplayMismatch(f"capture of seed {seed}: fingerprint {fp} != {h['fingerprint']}")
        return cap
    if h.get("outcome") == WORKER_ERROR:
        raise ReplayMismatch(f"seed {seed}: a worker_error record ends open: use replay_open")
    env = _segment_env(h)
    d = _replay_steps(env, rec)
    if isinstance(d, E2EDecision) and not d.bootstrap:
        raise ReplayMismatch(f"seed {seed}: the log ended before the segment did")
    term = env.terminal
    if term.fingerprint != h["fingerprint"] or term.outcome != h["outcome"]:
        raise ReplayMismatch(
            f"seed {seed}: replay reached {term.outcome}/{term.fingerprint}, "
            f"log says {h['outcome']}/{h['fingerprint']}"
        )
    return term


def replay_open(rec: LogRecord) -> tuple[Any, Any]:
    """Rebuild a `worker_error` record's segment up to the decision the worker failed on.

    Returns `(env, decision)`; asserts the run's fingerprint there matches the record's.
    """
    from .env import E2EDecision, run_fingerprint

    h = rec.header
    seed = int(h["seed"])
    if h.get("outcome") != WORKER_ERROR:
        raise ReplayMismatch(f"seed {seed}: not a worker_error record ({h.get('outcome')})")
    env = _segment_env(h)
    d = _replay_steps(env, rec)
    if not isinstance(d, E2EDecision):
        raise ReplayMismatch(f"seed {seed}: the segment ended before the logged decision")
    fp = run_fingerprint(env.rm)
    if fp != h["fingerprint"]:
        raise ReplayMismatch(f"seed {seed}: fingerprint {fp} != {h['fingerprint']}")
    return env, d


# ── the sink ─────────────────────────────────────────────────────────────────


class LogSink:
    """Appends kept records to `<root>/actions-<worker>.e2el` under a shared byte cap.

    The cap covers every `*.e2el` under `root` (all workers): each sink re-reads the
    directory total every `rescan` records, so the overshoot stays below
    workers x rescan x one record.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        worker: int,
        seed: int,
        cap_bytes: int = CAP_BYTES,
        train_p: float = TRAIN_SAMPLE_P,
        rescan: int = 256,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"actions-{int(worker):03d}.e2el"
        self.cap_bytes = int(cap_bytes)
        self.train_p = float(train_p)
        self.rescan = int(rescan)
        self.rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(worker), 0xAC7])
        self.written = 0
        self.dropped = 0
        self.records = 0
        self._total = self._dir_bytes()
        self._since_scan = 0
        self.full = self._total >= self.cap_bytes

    def _dir_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.e2el"))

    def reason(self, term: Any, *, eval_run: bool) -> str | None:
        """Why `term` is kept, or None. Draws the 2% sample once per training run that no
        other reason keeps."""
        if eval_run:
            return "eval"
        if term.reached_boss2:
            return "boss2"
        if term.outcome in ("capped", "error"):
            return str(term.outcome)
        if term.segment.kind == "run" and self.rng.random() < self.train_p:
            return "sample"
        return None

    def offer(
        self, term: Any, *, eval_run: bool = False, extra: dict[str, Any] | None = None
    ) -> bool:
        why = self.reason(term, eval_run=eval_run)
        if why is None:
            return False
        return self.write(terminal_record(term, reason=why, extra=extra))

    def write(self, rec: LogRecord) -> bool:
        if self._since_scan >= self.rescan:
            self._total = self._dir_bytes()
            self._since_scan = 0
            self.full = self._total >= self.cap_bytes
        if self.full:
            self.dropped += 1
            return False
        blob = rec.encode()
        with self.path.open("ab") as f:
            f.write(blob)
        self.written += len(blob)
        self._total += len(blob)
        self.records += 1
        self._since_scan += 1
        if self._total >= self.cap_bytes:
            self.full = True
        return True
