"""Named save files for the workbench live session.

A save captures the **active line** of a :class:`workbench.branching.BranchingLiveSession`
— the run identity ``(character, seed, ascension, run_mode)`` plus the ordered
``(action, args)`` pairs from the root to the active leaf. Because the engine
is deterministic, loading replays that line to restore the exact position
(mid-combat included). Abandoned branches are not saved.

Replay-based restore means a save is only valid against the engine that
recorded it: a change to combat or run generation reshapes the run under the
same seed, so an old save's line stops fitting and the load reports where it
diverged. Saves do not survive an engine upgrade, by construction.

Each save is one self-contained JSON file at ``<logs_dir>/../saves/workbench/<slug>.json``
(see :func:`save_dir`). The user-given name is preserved verbatim under ``name``;
the filename is a filesystem-safe slug of it, so re-saving under the same name
overwrites the same file.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_client import LEGACY_GAME_VERSION as _LEGACY_GAME_VERSION

_log = logging.getLogger(__name__)

# Save files written by this module carry this version; readers skip mismatches.
SCHEMA_VERSION = 1

# Backend assumed for saves written before the ``backend`` field existed. Those
# predate the rust backend's save support and were only ever loaded on C#, so
# defaulting to it preserves their original replay behavior.
DEFAULT_BACKEND = "csharp"

# ``act1`` value for saves written before the field existed (GH #544). Those
# runs predate act selection entirely, so they were all Overgrowth — unlike
# ``run_mode``, this one is KNOWN, not unknown, so the loader need not probe.
DEFAULT_ACT1 = "overgrowth"

# ``run_mode`` value for saves written before the field existed: unknown, not
# False. Those files are the ONLY reason the loader probes both constructors —
# guessing False on them is what made a saved "play to end" run unloadable.
UNKNOWN_RUN_MODE = None

# ``game_version`` for saves without the field (GH #801): every save written
# before it existed was played on the default build. Writers omit the key at
# the default, so a v0.107.1 save is byte-identical to one written before.
LEGACY_GAME_VERSION = _LEGACY_GAME_VERSION


def restore_attempts(payload: dict[str, Any]) -> list[bool]:
    """The ``run_mode`` values the loader should try for *payload*, best first
    (see :meth:`workbench.live_runs.RunIdentity.restore_attempts`)."""
    from workbench.live_runs import RunIdentity  # lazy: live_runs imports this module

    return RunIdentity.from_save(payload).restore_attempts()


def save_dir(logs_dir: str | Path) -> Path:
    """Return the workbench saves directory (``<repo>/saves/workbench``).

    Derived from ``logs_dir`` (``<repo>/logs``) so saves sit beside logs under
    the repo root regardless of where the server was launched.
    """
    return Path(logs_dir).resolve().parent / "saves" / "workbench"


def slugify(name: str) -> str:
    """Convert a user-given save name to a filesystem-safe slug.

    Lowercases, replaces runs of non-alphanumeric characters with ``-``, and
    trims leading/trailing dashes. Falls back to ``"save"`` if nothing remains.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "save"


def write_save(
    directory: Path,
    *,
    name: str,
    character: str,
    seed: str,
    ascension: int,
    backend: str,
    run_mode: bool,
    progress: dict[str, Any],
    line: list[dict[str, Any]],
    act1: str = DEFAULT_ACT1,
    node_values: list[float | None] | None = None,
    game_version: str = LEGACY_GAME_VERSION,
) -> Path:
    """Write a save file and return its path. Overwrites any same-slug file.

    ``backend`` records which engine recorded the line (``"csharp"`` /
    ``"rust"``); replaying on a different backend can diverge because the two
    serialize decisions (and map option indices) differently, so the loader
    must restore on the same backend.

    ``run_mode`` records which CONSTRUCTOR built the run: run-mode ("play to
    end") runs take an integer seed, interactive ones a text seed, and the two
    hash to different games. Restoring through the wrong one replays the line
    against an unrelated run, which the engine rejects a few actions in as
    ``invalid_for_phase``. Saves written before this field existed carry
    :data:`UNKNOWN_RUN_MODE` and the loader has to probe (see
    :func:`likely_run_mode`).

    ``act1`` records which of the two Act-1 models the run walked (GH #544).
    They are entirely different acts, so restoring under the wrong one diverges
    the same way a wrong ``run_mode`` does. It matters most for ``"roll"`` runs,
    where the act came from the seed and is not recoverable from the request.

    ``node_values`` is the PPO critic value per node along the same path, root
    first (so it is one longer than ``line``). These are pure cache: replay
    reproduces the identical states, so the loader could recompute them — but
    that is a 320-token transformer forward per non-combat decision, which on a
    long act-3 line is the bulk of a restore. Optional, and absent from saves
    written before it existed; the loader falls back to recomputing.

    ``game_version`` records which game build the run simulates (GH #801):
    the same line replays to a different run under another build. Always
    written, so a save names its build; readers treat a missing key (a save
    from before 2026-11-14 that recorded a v0.107.1 run) as
    :data:`LEGACY_GAME_VERSION`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "character": character,
        "seed": seed,
        "ascension": ascension,
        "backend": backend,
        "run_mode": run_mode,
        "act1": act1,
        "progress": progress,
        "step_count": len(line),
        "line": line,
        "node_values": node_values,
        "game_version": game_version,
    }
    path = directory / f"{slugify(name)}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


#: Per-node fields stripped from a save's list-view header — both are one entry
#: per step, so leaving them in would put a whole run's worth of data per row
#: into the load dialog's listing.
_HEAVY_FIELDS = frozenset({"line", "node_values"})


def _header(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip the per-step fields, leaving list-view metadata."""
    return {k: v for k, v in payload.items() if k not in _HEAVY_FIELDS}


def list_saves(directory: Path) -> list[dict[str, Any]]:
    """Return metadata headers (no ``line``) for every valid save, newest first.

    Corrupt or schema-mismatched files are skipped with a logged warning rather
    than failing the whole listing.
    """
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _log.warning("skipping unreadable save %s", path, exc_info=True)
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            _log.warning("skipping save with bad/missing schema_version: %s", path)
            continue
        out.append(_header(payload))
    out.sort(key=lambda h: h.get("saved_at", ""), reverse=True)
    return out


def read_save(directory: Path, name: str) -> dict[str, Any] | None:
    """Return the full save payload for *name*, or ``None`` if absent/invalid."""
    path = directory / f"{slugify(name)}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _log.warning("save %s is unreadable", path, exc_info=True)
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return None
    return payload
