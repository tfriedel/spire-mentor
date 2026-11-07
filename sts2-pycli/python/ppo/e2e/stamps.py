"""Stamps: what code, engine and schema made an e2e campaign artifact.

Plan section 3.1 (WP1). Every cfg, checkpoint, `stats.jsonl` row and eval row
of the campaign carries `stamps()`:

* `game_version`: the build this process selects. `stamps` raises unless it is
  the campaign's, v0.111.0 (`factory.require_selection`).
* `py_src`: `wheel_guard.python_source_digest()`, the digest the PPO workers
  already report, over `ppo/**/*.py` and `champions.py`.
* `py_client`: the same kind of digest over `sts2_client.py`, which `py_src`
  leaves out although it holds the build selection and the read-back checks.
* `wheel`: `wheel_guard.wheel_identity()`, the `sts2_rs` build this process
  loaded.
* `vocab_sha256`, `schema_sha256`: the observation vocab and schema. Their
  owner (WP3) passes them in; `canonical_sha256` hashes a JSON-able spec
  independently of key order.

`stamp_change(before, after)` says how two stamps differ (a checkpoint's
against the resuming process, a cfg's against an eval row), or None when they
match. Where the wheel file lives does not count, as in
`wheel_guard.engine_change`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import sts2_client

from ppo import wheel_guard
from ppo.e2e import factory

STAMP_KEYS = ("game_version", "py_src", "py_client", "wheel", "vocab_sha256", "schema_sha256")


def file_digest(path: Path) -> str:
    """sha256[:16] over a file's name and bytes, in `wheel_guard.source_digest`'s
    framing (name, NUL, 8-byte little-endian length, bytes)."""
    data = path.read_bytes()
    h = hashlib.sha256(path.name.encode("utf-8") + b"\0")
    h.update(len(data).to_bytes(8, "little") + data)
    return h.hexdigest()[:16]


@functools.cache
def client_source_digest() -> str:
    """`file_digest` of the `sts2_client.py` this process imported, taken once
    (at the first call), like `wheel_guard.python_source_digest`."""
    return file_digest(Path(sts2_client.__file__))


def canonical_sha256(obj: Any) -> str:
    """sha256 of `obj` as canonical JSON (sorted keys, no whitespace, UTF-8).

    Raises:
        TypeError: `obj` is not JSON-serializable.
    """
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stamps(*, vocab_sha256: str | None = None, schema_sha256: str | None = None) -> dict[str, Any]:
    """This process's stamps (`STAMP_KEYS`), for a cfg, checkpoint or row.

    Args:
        vocab_sha256: The observation vocab's sha256, from its owner.
        schema_sha256: The observation schema's sha256, from its owner.

    Raises:
        sts2_client.GameVersionMismatch: The process selects another build
            than v0.111.0.
    """
    factory.require_selection()
    return {
        "game_version": sts2_client.GAME_VERSION,
        "py_src": wheel_guard.python_source_digest(),
        "py_client": client_source_digest(),
        "wheel": wheel_guard.wheel_identity(),
        "vocab_sha256": vocab_sha256,
        "schema_sha256": schema_sha256,
    }


def stamp_change(before: dict[str, Any] | None, after: dict[str, Any] | None) -> str | None:
    """How stamp `after` differs from `before`, or None when they match.

    A missing stamp, or a missing key in one, counts as a change: every
    campaign artifact carries all of `STAMP_KEYS`.
    """
    if before is None or after is None:
        return f"missing stamp (before: {before is not None}, after: {after is not None})"
    moved = []
    for key in STAMP_KEYS:
        if key not in before or key not in after:
            moved.append(f"{key} missing (before: {key in before}, after: {key in after})")
            continue
        old, new = before[key], after[key]
        if key == "wheel" and old is not None and new is not None:
            change = wheel_guard.engine_change(old, new)
            if change:
                moved.append(f"wheel {change}")
        elif old != new:
            moved.append(f"{key} {old} -> {new}")
    return ", ".join(moved) or None
