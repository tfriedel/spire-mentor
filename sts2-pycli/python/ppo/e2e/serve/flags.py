"""One feature flag per S1b component.

With every flag off, a learner keeps #1081's path: pickled `Packed` messages
over pipes, `Padded.from_flat`, one padded forward, `_pack` into shared
memory. Each flag swaps one component:

* `shm`: the shared-memory transport (`transport.ShmTransport`) instead of
  pickled messages on pipes. Episode ends and control messages stay on pipes.
* `buckets`: length-bucketed batches (`batcher.Batcher(buckets=...)`) instead
  of one batch padded to its longest row. Needs `shm`.
* `packed`: pad-free inference with variable-length attention
  (`cppo.packed_forward`) instead of a padded forward.
* `dbuf`: the double-buffered loop (`packed.Pipeline(dbuf=True)`): the CPU
  gathers batch `k + 1` while the GPU runs batch `k`. Needs `shm`.
* `compact`: the sparse row codec (`cppo.cppo_fields(compact=True)`) instead
  of raw rows, for the transport payload and the learner's row store.

`ServeFlags.parse` takes `""` or `"none"` (all off), `"all"`, or a comma list
such as `"shm,buckets,dbuf"`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

NAMES = ("shm", "buckets", "packed", "dbuf", "compact")
_NEEDS_SHM = ("buckets", "dbuf")


@dataclass(frozen=True)
class ServeFlags:
    shm: bool = False
    buckets: bool = False
    packed: bool = False
    dbuf: bool = False
    compact: bool = False

    def __post_init__(self) -> None:
        for name in _NEEDS_SHM:
            if getattr(self, name) and not self.shm:
                raise ValueError(f"serve flag {name!r} needs 'shm'")

    @classmethod
    def parse(cls, s: str | None) -> ServeFlags:
        s = (s or "").strip().lower()
        if s in ("", "none", "off"):
            return cls()
        if s == "all":
            return cls(**dict.fromkeys(NAMES, True))
        parts = [p.strip() for p in s.split(",") if p.strip()]
        bad = sorted(set(parts) - set(NAMES))
        if bad:
            raise ValueError(f"unknown serve flags {bad}; known: {', '.join(NAMES)}")
        return cls(**{n: n in parts for n in NAMES})

    @classmethod
    def all(cls) -> ServeFlags:
        return cls.parse("all")

    @property
    def any(self) -> bool:
        return any(getattr(self, f.name) for f in fields(self))

    def __str__(self) -> str:
        on = [n for n in NAMES if getattr(self, n)]
        return ",".join(on) if on else "none"

    def stamp(self) -> dict[str, bool]:
        """For a run's cfg stamp."""
        return asdict(self)
