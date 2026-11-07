"""Checkpoint files of an e2e lane: atomic writes, rotation, pinning, pruning (plan §8).

Layout (names in :mod:`ppo.e2e.ops_contract`)::

    state.pt, state.json            the resume state every 5M decisions, and its
                                    sidecar (decisions, it, time)
    ckpt/full/full_d<D>.pt          rolling full states every 25M; the newest 3 stay
    ckpt/pinned/pinned_d<D>_<reason>.pt
                                    never rotated, sha256 in ckpt/MANIFEST.tsv
    ckpt/slim/slim_d<D>.pt          weights and EMAs every 100M
    ckpt/quarantine/<tag>/          states a rollback moved aside

``<D>`` is the decision count, zero-padded to 13 digits, so names sort by it.

The module is torch-free. Callers pass bytes, text, or a ``writer(path)``
callable such as ``lambda p: torch.save(obj, p)``. Every write goes to a temp
file in the target dir, is fsynced and renamed into place, so a crash never
leaves a partial file under a final name.

``python -m ppo.e2e.ckpt <command> SAVE_DIR ...`` exposes the operator and
watchdog operations: ``usage``, ``verify``, ``second-newest``, ``last-pinned``,
``pin``, ``rollback``, ``disk-check`` and ``prune``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import ops_contract as C

Data = bytes | str | Callable[[Path], None]

DIGITS = 13
STATE_META = "state.json"  # the sidecar of state.pt: {"decisions", "it", "time"}
MANIFEST_HEADER = ("sha256", "name", "decisions", "reason", "status", "bytes", "time", "note")
PINNED = "pinned"
ABANDONED = "abandoned"  # a pinned state on a lineage a rollback left behind
KEEP_FULL_ON_PRUNE = 2  # the NaN rule needs the second-newest full state
SLIM_PRUNE_LAST_EVERY = 500_000_000  # slim states on 500M multiples go last

_NAME = re.compile(r"^(full|slim|pinned)_d(\d{13})(?:_([a-z0-9_]+?))?(?:_b(\d+))?\.pt$")
_REASON = re.compile(r"^[a-z0-9_]+$")


# ── writing ──────────────────────────────────────────────────────────────────


def _fsync_dir(d: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _temp_in(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
    os.close(fd)
    return Path(tmp)


def _write_tmp(tmp: Path, data: Data) -> None:
    if callable(data):
        data(tmp)
    else:
        with open(tmp, "wb") as f:
            f.write(data.encode("utf-8") if isinstance(data, str) else data)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())


def atomic_write(path: str | os.PathLike, data: Data) -> Path:
    """Write ``data`` to ``path`` through a fsynced temp file and a rename."""
    path = Path(path)
    tmp = _temp_in(path.parent, path.name)
    try:
        _write_tmp(tmp, data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_with_sha(src: Path, dst: Path) -> str:
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            h.update(chunk)
            fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    return h.hexdigest()


# ── names ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ckpt:
    path: Path
    kind: str  # "full" | "slim" | "pinned"
    decisions: int
    reason: str | None


def ckpt_name(kind: str, decisions: int, reason: str | None = None, bump: int = 0) -> str:
    if kind not in ("full", "slim", "pinned"):
        raise ValueError(kind)
    if decisions < 0 or decisions >= 10**DIGITS:
        raise ValueError(f"decisions {decisions} out of range")
    if (kind == "pinned") != (reason is not None):
        raise ValueError("a pinned state needs a reason, and only a pinned one")
    if reason is not None and not _REASON.match(reason):
        raise ValueError(f"bad pin reason {reason!r}")
    name = f"{kind}_d{decisions:0{DIGITS}d}"
    if reason:
        name += f"_{reason}"
    if bump:
        name += f"_b{bump}"
    return name + ".pt"


def parse_name(name: str) -> Ckpt | None:
    m = _NAME.match(name)
    if not m:
        return None
    kind, dec, reason, _bump = m.groups()
    if (kind == "pinned") != (reason is not None):
        return None
    return Ckpt(Path(name), kind, int(dec), reason)


# ── the save dir ─────────────────────────────────────────────────────────────


class SaveDir:
    """Paths of one lane's save dir."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"SaveDir({str(self.root)!r})"

    def p(self, rel: str) -> Path:
        return self.root / rel

    @property
    def state(self) -> Path:
        return self.root / C.STATE

    @property
    def state_meta(self) -> Path:
        return self.root / STATE_META

    @property
    def full_dir(self) -> Path:
        return self.root / C.FULL_DIR

    @property
    def pinned_dir(self) -> Path:
        return self.root / C.PINNED_DIR

    @property
    def slim_dir(self) -> Path:
        return self.root / C.SLIM_DIR

    @property
    def quarantine_dir(self) -> Path:
        return self.root / C.QUARANTINE_DIR

    @property
    def manifest(self) -> Path:
        return self.root / C.MANIFEST

    def listing(self, kind: str) -> list[Ckpt]:
        d = {"full": self.full_dir, "slim": self.slim_dir, "pinned": self.pinned_dir}[kind]
        out = []
        if d.is_dir():
            for f in d.iterdir():
                c = parse_name(f.name)
                if c and c.kind == kind and f.is_file():
                    out.append(Ckpt(f, c.kind, c.decisions, c.reason))
        return sorted(out, key=lambda c: (c.decisions, c.path.name))


def _sd(save_dir: str | os.PathLike | SaveDir) -> SaveDir:
    return save_dir if isinstance(save_dir, SaveDir) else SaveDir(save_dir)


def crossed(before: int, after: int, every: int) -> bool:
    """True when ``after`` passed a multiple of ``every`` that ``before`` had not."""
    return every > 0 and after // every > before // every


# ── state.pt, rolling full states, slim states ──────────────────────────────


def save_state(
    save_dir: str | os.PathLike | SaveDir, data: Data, *, decisions: int, it: int
) -> Path:
    """Write ``state.pt``, then its ``state.json`` sidecar."""
    sd = _sd(save_dir)
    atomic_write(sd.state, data)
    meta = {"decisions": int(decisions), "it": int(it), "time": time.time()}
    atomic_write(sd.state_meta, json.dumps(meta) + "\n")
    return sd.state


def state_decisions(save_dir: str | os.PathLike | SaveDir) -> int | None:
    meta = C.read_json(_sd(save_dir).state_meta)
    return int(meta["decisions"]) if isinstance(meta, dict) and "decisions" in meta else None


def save_full(
    save_dir: str | os.PathLike | SaveDir, data: Data, *, decisions: int, keep: int = C.FULL_KEEP
) -> Path:
    """Write a rolling full state and delete all but the newest ``keep``."""
    sd = _sd(save_dir)
    path = atomic_write(sd.full_dir / ckpt_name("full", decisions), data)
    rotate_full(sd, keep)
    return path


def rotate_full(save_dir: str | os.PathLike | SaveDir, keep: int = C.FULL_KEEP) -> list[Path]:
    sd = _sd(save_dir)
    old = sd.listing("full")[:-keep] if keep > 0 else sd.listing("full")
    for c in old:
        c.path.unlink(missing_ok=True)
    return [c.path for c in old]


def save_slim(save_dir: str | os.PathLike | SaveDir, data: Data, *, decisions: int) -> Path:
    sd = _sd(save_dir)
    return atomic_write(sd.slim_dir / ckpt_name("slim", decisions), data)


# ── pinned states and the manifest ──────────────────────────────────────────


@dataclass(frozen=True)
class ManifestRow:
    sha256: str
    name: str
    decisions: int
    reason: str
    status: str
    bytes: int
    time: str
    note: str

    def path(self, save_dir: str | os.PathLike | SaveDir) -> Path:
        return _sd(save_dir).pinned_dir / self.name


def read_manifest(save_dir: str | os.PathLike | SaveDir) -> list[ManifestRow]:
    path = _sd(save_dir).manifest
    rows = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if not line.strip() or line.startswith("sha256\t") or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(MANIFEST_HEADER):
            parts += [""] * (len(MANIFEST_HEADER) - len(parts))
        try:
            rows.append(
                ManifestRow(
                    parts[0], parts[1], int(parts[2]), parts[3], parts[4], int(parts[5] or 0),
                    parts[6], parts[7],
                )
            )  # fmt: skip
        except ValueError:
            continue
    return rows


def _append_manifest(sd: SaveDir, row: ManifestRow) -> None:
    sd.manifest.parent.mkdir(parents=True, exist_ok=True)
    new = not sd.manifest.exists()
    note = row.note.replace("\t", " ").replace("\n", " ")
    with open(sd.manifest, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(MANIFEST_HEADER) + "\n")
        f.write(
            "\t".join(
                [row.sha256, row.name, str(row.decisions), row.reason, row.status,
                 str(row.bytes), row.time, note]
            ) + "\n"
        )  # fmt: skip
        f.flush()
        os.fsync(f.fileno())


def _latest_by_name(rows: list[ManifestRow]) -> dict[str, ManifestRow]:
    latest: dict[str, ManifestRow] = {}
    first: dict[str, ManifestRow] = {}
    for r in rows:
        first.setdefault(r.name, r)
        latest[r.name] = r
    # The sha256 and size come from the pinning row; later rows only change status.
    return {
        n: ManifestRow(first[n].sha256, n, first[n].decisions, first[n].reason, r.status,
                       first[n].bytes, r.time, r.note)
        for n, r in latest.items()
    }  # fmt: skip


def pinned_states(save_dir: str | os.PathLike | SaveDir) -> list[ManifestRow]:
    """Pinned states whose latest manifest status is ``pinned``, oldest first."""
    sd = _sd(save_dir)
    rows = [r for r in _latest_by_name(read_manifest(sd)).values() if r.status == PINNED]
    return sorted((r for r in rows if r.path(sd).is_file()), key=lambda r: (r.decisions, r.name))


def last_pinned(save_dir: str | os.PathLike | SaveDir) -> ManifestRow | None:
    rows = pinned_states(save_dir)
    return rows[-1] if rows else None


def pin(
    save_dir: str | os.PathLike | SaveDir,
    *,
    decisions: int,
    reason: str,
    data: Data | None = None,
    src: str | os.PathLike | None = None,
    note: str = "",
) -> Path:
    """Write (``data``) or copy (``src``) a full state into ``ckpt/pinned/`` and
    record its sha256 in the manifest. Pinning the same bytes at the same
    decision count again returns the existing pin."""
    if (data is None) == (src is None):
        raise ValueError("pin needs exactly one of data and src")
    sd = _sd(save_dir)
    tmp = _temp_in(sd.pinned_dir, "pin")
    try:
        if src is not None:
            sha = _copy_with_sha(Path(src), tmp)
        else:
            assert data is not None
            _write_tmp(tmp, data)
            sha = sha256_file(tmp)
        for r in pinned_states(sd):
            if r.decisions == decisions and r.sha256 == sha:
                tmp.unlink()
                return r.path(sd)
        bump = 0
        while (sd.pinned_dir / ckpt_name("pinned", decisions, reason, bump)).exists():
            bump += 1
        final = sd.pinned_dir / ckpt_name("pinned", decisions, reason, bump)
        size = tmp.stat().st_size
        tmp.replace(final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(sd.pinned_dir)
    _append_manifest(
        sd, ManifestRow(sha, final.name, decisions, reason, PINNED, size, C.utc_now_iso(), note)
    )
    return final


def verify_pinned(save_dir: str | os.PathLike | SaveDir) -> list[str]:
    """Problems with the pinned states: missing files, sha256 mismatches,
    files without a manifest row."""
    sd = _sd(save_dir)
    latest = _latest_by_name(read_manifest(sd))
    out = []
    for r in latest.values():
        f = r.path(sd)
        if not f.is_file():
            out.append(f"{r.name}: in the manifest but missing")
        elif sha256_file(f) != r.sha256:
            out.append(f"{r.name}: sha256 differs from the manifest")
    out += [
        f"{c.path.name}: not in the manifest"
        for c in sd.listing("pinned")
        if c.path.name not in latest
    ]
    return out


# ── full states and rollbacks ───────────────────────────────────────────────


@dataclass(frozen=True)
class FullState:
    decisions: int
    path: Path
    kind: str  # "full" (rolling) or "pinned"


def full_states(save_dir: str | os.PathLike | SaveDir) -> list[FullState]:
    """Rolling and pinned full states, one per decision count (pinned wins), oldest first.
    ``state.pt`` is the working resume file, not one of them."""
    sd = _sd(save_dir)
    by_dec: dict[int, FullState] = {}
    for c in sd.listing("full"):
        by_dec[c.decisions] = FullState(c.decisions, c.path, "full")
    for r in pinned_states(sd):
        by_dec[r.decisions] = FullState(r.decisions, r.path(sd), "pinned")
    return [by_dec[k] for k in sorted(by_dec)]


def second_newest_full_state(save_dir: str | os.PathLike | SaveDir) -> FullState | None:
    states = full_states(save_dir)
    return states[-2] if len(states) >= 2 else None


def _decisions_of(sd: SaveDir, target: Path) -> int | None:
    c = parse_name(target.name)
    if c is not None:
        return c.decisions
    if target.resolve() == sd.state.resolve():
        return state_decisions(sd)
    return None


def quarantine(save_dir: str | os.PathLike | SaveDir, paths: list[Path], tag: str) -> list[Path]:
    """Move ``paths`` into ``ckpt/quarantine/<tag>/``."""
    sd = _sd(save_dir)
    qdir = sd.quarantine_dir / tag
    moved = []
    for p in paths:
        if p.exists():
            qdir.mkdir(parents=True, exist_ok=True)
            dst = qdir / p.name
            p.replace(dst)
            moved.append(dst)
    return moved


def rollback(
    save_dir: str | os.PathLike | SaveDir,
    target: str | os.PathLike,
    *,
    reason: str,
    allow_unpinned: bool = False,
) -> dict[str, Any]:
    """Prepare a resume from ``target``: quarantine ``state.pt`` and every
    rolling and slim state newer than it, mark newer pinned states abandoned,
    and write ``ops/resume_next.json``, which the watchdog uses at its next start.

    Deliberate rollbacks (K2 relaunch, E=1 revert, an exploit fix) start only
    from pinned states (plan §8). Only the NaN rule passes ``allow_unpinned``.
    The lane must be stopped first: a learner that still runs would write a
    new ``state.pt`` at its exit and undo the rollback.
    """
    sd = _sd(save_dir)
    if C.record_alive(C.read_json(sd.root / C.LEARNER_PID)):
        raise RuntimeError("the learner still runs: stop the lane before a rollback")
    target = Path(target)
    if not target.is_absolute():
        target = sd.pinned_dir / target if (sd.pinned_dir / target).exists() else sd.root / target
    if not target.is_file():
        raise FileNotFoundError(target)
    pinned = {r.name: r for r in pinned_states(sd)}
    is_pinned = target.parent.resolve() == sd.pinned_dir.resolve() and target.name in pinned
    if not (is_pinned or allow_unpinned):
        raise ValueError(f"{target} is not a pinned state; deliberate rollbacks need one")
    dec = _decisions_of(sd, target)
    if dec is None:
        raise ValueError(f"cannot tell the decision count of {target}")
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{reason}"
    newer = [c.path for k in ("full", "slim") for c in sd.listing(k) if c.decisions > dec]
    moved = quarantine(sd, [sd.state, sd.state_meta, *newer], tag)
    abandoned = [r for r in pinned.values() if r.decisions > dec]
    for r in abandoned:
        _append_manifest(
            sd,
            ManifestRow(r.sha256, r.name, r.decisions, r.reason, ABANDONED, r.bytes,
                        C.utc_now_iso(), f"rollback to {target.name} ({reason})"),
        )  # fmt: skip
    rec = {
        "path": str(target),
        "decisions": dec,
        "reason": reason,
        "time": C.utc_now_iso(),
        "unix": time.time(),
        "quarantined": [str(p) for p in moved],
        "abandoned": [r.name for r in abandoned],
    }
    C.atomic_write_text(sd.p(C.RESUME_NEXT), json.dumps(rec, indent=1) + "\n")
    return rec


# ── disk ─────────────────────────────────────────────────────────────────────


def tree_bytes(root: str | os.PathLike) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            with contextlib.suppress(OSError):
                total += (Path(dirpath) / f).lstat().st_size
    return total


def quota_gb_from_env() -> float | None:
    raw = os.environ.get(C.DISK_QUOTA_ENV, "")
    return float(raw) if raw else None


def disk_free_bytes(save_dir: str | os.PathLike | SaveDir, quota_gb: float | None = None) -> int:
    """Free bytes for the lane: the file system's, or less under a soft quota
    (``E2E_DISK_QUOTA_GB``: the save dir may hold at most that much)."""
    sd = _sd(save_dir)
    free = shutil.disk_usage(sd.root).free
    if quota_gb is None:
        quota_gb = quota_gb_from_env()
    if quota_gb is not None:
        free = min(free, int(quota_gb * 1e9) - tree_bytes(sd.root))
    return free


def usage(save_dir: str | os.PathLike | SaveDir) -> dict[str, int]:
    """Bytes per category, plus ``total``."""
    sd = _sd(save_dir)
    cats = {
        "state": sum(p.stat().st_size for p in (sd.state, sd.state_meta) if p.is_file()),
        "full": tree_bytes(sd.full_dir),
        "pinned": tree_bytes(sd.pinned_dir),
        "slim": tree_bytes(sd.slim_dir),
        "quarantine": tree_bytes(sd.quarantine_dir),
        "logs": sum(p.stat().st_size for p in (sd.p(C.STATS), sd.p(C.THROUGHPUT)) if p.is_file())
        + tree_bytes(sd.p(C.OPS_DIR)),
        "evals": tree_bytes(sd.p("evals")),
    }
    cats["total"] = tree_bytes(sd.root)
    cats["other"] = cats["total"] - sum(v for k, v in cats.items() if k != "total")
    return cats


def prune_candidates(save_dir: str | os.PathLike | SaveDir) -> list[Path]:
    """What pruning may delete, in order: quarantined files (oldest tag first),
    slim states off the 500M grid, the other slim states, and rolling full
    states beyond the newest two (each oldest first). Never pinned states,
    never state.pt, never the newest two rolling states."""
    sd = _sd(save_dir)
    out: list[Path] = []
    if sd.quarantine_dir.is_dir():
        for tag in sorted(sd.quarantine_dir.iterdir()):
            out += sorted(p for p in tag.rglob("*") if p.is_file()) if tag.is_dir() else [tag]
    slim = sd.listing("slim")
    on_grid = [
        c for c in slim
        if (c.decisions // C.SLIM_EVERY * C.SLIM_EVERY) % SLIM_PRUNE_LAST_EVERY == 0
    ]  # fmt: skip
    out += [c.path for c in slim if c not in on_grid] + [c.path for c in on_grid]
    full = sd.listing("full")
    out += [c.path for c in full[: max(0, len(full) - KEEP_FULL_ON_PRUNE)]]
    return out


def prune(
    save_dir: str | os.PathLike | SaveDir,
    *,
    target_free_gb: float,
    quota_gb: float | None = None,
) -> list[Path]:
    """Delete :func:`prune_candidates` in order until ``target_free_gb`` is free."""
    sd = _sd(save_dir)
    deleted = []
    for p in prune_candidates(sd):
        if disk_free_bytes(sd, quota_gb) >= target_free_gb * 1e9:
            break
        p.unlink(missing_ok=True)
        deleted.append(p)
    if sd.quarantine_dir.is_dir():
        for tag in sd.quarantine_dir.iterdir():
            if tag.is_dir() and not any(tag.rglob("*")):
                tag.rmdir()
    return deleted


@dataclass
class DiskVerdict:
    free_gb: float
    action: str  # "ok" | "pruned" | "stop"
    deleted: list[str]


def disk_check(
    save_dir: str | os.PathLike | SaveDir,
    *,
    prune_below_gb: float,
    stop_below_gb: float,
    quota_gb: float | None = None,
) -> DiskVerdict:
    """Below ``prune_below_gb`` free, prune back up to it; then ``stop`` if
    still below ``stop_below_gb``."""
    sd = _sd(save_dir)
    free = disk_free_bytes(sd, quota_gb) / 1e9
    if free >= prune_below_gb:
        return DiskVerdict(free, "ok", [])
    deleted = prune(sd, target_free_gb=prune_below_gb, quota_gb=quota_gb)
    free = disk_free_bytes(sd, quota_gb) / 1e9
    action = "stop" if free < stop_below_gb else ("pruned" if deleted else "ok")
    if action == "ok" and free < prune_below_gb:
        action = "pruned"  # nothing left to prune, but still under the prune line
    return DiskVerdict(free, action, [str(p) for p in deleted])


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.e2e.ckpt", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("usage", "verify", "second-newest", "last-pinned", "disk-check"):
        sub.add_parser(name).add_argument("save_dir")
    p = sub.add_parser("pin", help="pin a full state by hand (hot-swap, E=1 switch, manual)")
    p.add_argument("save_dir")
    p.add_argument("--src", required=True)
    p.add_argument("--decisions", type=int, help="default: from the name or state.json")
    p.add_argument("--reason", required=True, choices=C.PIN_REASONS)
    p.add_argument("--note", default="")
    r = sub.add_parser("rollback", help="resume the next start from a pinned state")
    r.add_argument("save_dir")
    r.add_argument("--to", required=True, help="a pinned state's name or path")
    r.add_argument("--reason", required=True)
    pr = sub.add_parser("prune")
    pr.add_argument("save_dir")
    pr.add_argument("--target-free-gb", type=float, required=True)
    a = ap.parse_args(argv)
    sd = SaveDir(a.save_dir)
    if a.cmd == "usage":
        u = usage(sd)
        print(json.dumps({**u, "free_bytes": disk_free_bytes(sd)}, indent=1))
    elif a.cmd == "verify":
        problems = verify_pinned(sd)
        for x in problems:
            print(x)
        print(f"{len(pinned_states(sd))} pinned state(s), {len(problems)} problem(s)")
        return 1 if problems else 0
    elif a.cmd == "second-newest":
        s = second_newest_full_state(sd)
        if s is None:
            return 1
        print(s.path)
    elif a.cmd == "last-pinned":
        row = last_pinned(sd)
        if row is None:
            return 1
        print(json.dumps({**asdict(row), "path": str(row.path(sd))}))
    elif a.cmd == "pin":
        src = Path(a.src)
        dec = a.decisions
        if dec is None:
            dec = _decisions_of(sd, src)
        if dec is None:
            print("pass --decisions: the source has no decision count", file=sys.stderr)
            return 2
        print(pin(sd, decisions=dec, reason=a.reason, src=src, note=a.note))
    elif a.cmd == "rollback":
        print(json.dumps(rollback(sd, a.to, reason=a.reason), indent=1))
    elif a.cmd == "prune":
        for x in prune(sd, target_free_gb=a.target_free_gb):
            print(x)
    elif a.cmd == "disk-check":
        lim = C.Limits.from_env()
        v = disk_check(sd, prune_below_gb=lim.disk_prune_gb, stop_below_gb=lim.disk_stop_gb)
        print(json.dumps(asdict(v)))
        return C.WD_EXIT_DISK if v.action == "stop" else 0
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
