"""Warm-start teachers (plan §7): log-probabilities over the student's merged
option classes.

Two teachers act and supply the kickstarting (KS) targets while the student
warms up (U0, U1):

* `CombatTeacher`: #1081's combat net (`combat_ctx.model.CombatCtxNet`, an
  `S_ctx` checkpoint such as `run/it06000.pt` or `hybrid/it02500.pt`), loaded
  the way `combat_ctx.play.CtxNet` loads it. It reads #1081's `S_ctx` tokens
  (`combat_ppo.encode.Decision`: RareMap-applied `cat`, float16 `cont`, `sec`,
  `n_act`). The student's combat rows keep those tokens bit for bit (plan §4,
  X-C): `sctx_rows` cuts them out of a `contracts` row batch by the section
  mask and column slice that `contracts` fixes (the `[sctx_lo, sctx_lo +
  n_sctx)` block, its first `SCTX_N_CAT` categorical and `SCTX_N_CONT`
  continuous columns, the block's last `n_mem` tokens as the options, and
  `mem_class` as the merge). `CombatTeacher.encode` builds the same tokens
  from an env decision with #1081's own encoder (~0.6 ms of worker CPU per
  decision), for a caller without rows. Both combat checkpoints (R2's
  it06000, R3m's it02500) share one RareMap (`rare_sha256` 11014549...),
  the one the tokenizer must apply to the S_ctx content slots.
* `RunTeacher`: a run net (`ppo.model.RunPolicyNet`, such as R2's run net,
  the cont300 Ironclad champion's checkpoint at iteration 1150, or
  `hybrid/run_ckpt_v00375.pt`), fed by its own `ppo.encoder.featurize_row` on
  the run decision beside the student's stream. The loader asserts
  `ck["cfg"]["features"] == "card_mods,reward_upgrade"`, because
  `ppo.net_policy.NetPolicy` silently falls back to the default features.

The named pairs are pinned, file and sha256, in `teachers.toml` beside this
module (`PINS`): a champion promotion must not move a teacher, so a teacher
is never resolved through `data/ppo/champions.toml`.

A teacher scores ITS options: the engine's legal actions in the order it
encodes them, which is the env decision's `actions` order (combat:
`cs.legal_actions()`; out of combat: the run actions). The env merges
identical options into classes (plan §3.2): `classes[k]` is the class of
option `k`, as in `ppo.e2e.env.E2EDecision.classes` or
`contracts.EnvDecision.member_class` (`decision_classes` reads either). The
map must be total
and onto (every option has a class, every class has an option), so a
class's log-probability is the log-sum-exp of its members' and the merged
distribution sums to 1 (T8). `merge_logp` does that, batched.

Both teachers run under `torch.no_grad`, in fp32 by default, on any device;
the serving GPU runs them during U0 and U1 (plan §7). Their log-softmax and
the merge run in float64, so a merged distribution sums to 1 within 1e-12.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

#: The run nets' feature set (PREREG-0 "Loaders").
RUN_FEATURES = "card_mods,reward_upgrade"


class TeacherMismatch(ValueError):
    """A teacher checkpoint is not what the wrapper can serve faithfully."""


class OptionMapError(ValueError):
    """A teacher-option -> merged-class map is not total and onto (T8)."""


def rare_sha256(rare: Any) -> str:
    """sha256 of a #1081 checkpoint's RareMap (`ck["rare"]`) as canonical JSON."""
    blob = json.dumps(rare, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_root() -> pathlib.Path:
    """The monorepo root (this file is `sts2-pycli/python/ppo/e2e/teachers.py`)."""
    return pathlib.Path(__file__).resolve().parents[4]


#: The campaign's frozen teacher inputs (plan D1; sha256 in its SHA256SUMS).
ARCHIVE = pathlib.Path.home() / "projects/sts2/bench-archive/2026-11-02-e2e-inputs"
#: The final R3's archive (PREREG-0 amendment 2): #1081's H2 pair at it04000,
#: copied read-only with a sha256 manifest on the laptop and on the a4000.
H2H3_ARCHIVE = pathlib.Path.home() / "projects/sts2/bench-archive/2026-11-03-combat-ppo-h2-h3"
#: The pinned teacher pairs, a data file: `[<pair>.<role>]` tables of `root`,
#: `path` and `sha256`.
PINS_FILE = pathlib.Path(__file__).with_name("teachers.toml")
#: A pair's roles: one combat and one run teacher.
ROLES = ("combat", "run")
_PIN_KEYS = {"root", "path", "sha256"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Pin:
    """A pinned teacher checkpoint: its file and the sha256 it must have."""

    path: pathlib.Path
    sha256: str


def load_pins(path: str | pathlib.Path = PINS_FILE) -> dict[str, dict[str, Pin]]:
    """The pinned pairs of a `teachers.toml`: pair name -> role -> `Pin`.

    `root` "archive" is `ARCHIVE`, "h2h3" is `H2H3_ARCHIVE`, "repo" the
    monorepo root; `path` is relative to it.

    Raises:
        ValueError: a pair without exactly the roles `ROLES`, a pin without
            exactly `root`, `path` and `sha256`, an unknown root, a path that
            is absolute or climbs out of its root, or a sha256 that is not 64
            lowercase hex digits.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    roots = {"archive": ARCHIVE, "h2h3": H2H3_ARCHIVE, "repo": repo_root()}
    out: dict[str, dict[str, Pin]] = {}
    for name, pair in raw.items():
        if not isinstance(pair, dict) or set(pair) != set(ROLES):
            got = sorted(pair) if isinstance(pair, dict) else pair
            raise ValueError(f"{path}: pair {name!r} needs the roles {list(ROLES)}, got {got!r}")
        out[name] = {}
        for role in ROLES:
            pin = pair[role]
            where = f"{path}: [{name}.{role}]"
            if not isinstance(pin, dict) or set(pin) != _PIN_KEYS:
                got = sorted(pin) if isinstance(pin, dict) else pin
                raise ValueError(f"{where} needs the keys {sorted(_PIN_KEYS)}, got {got!r}")
            if pin["root"] not in roots:
                raise ValueError(f"{where}: root {pin['root']!r} is not one of {sorted(roots)}")
            rel = pathlib.PurePosixPath(str(pin["path"]))
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"{where}: path {pin['path']!r} must be relative to its root")
            sha = pin["sha256"]
            if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
                raise ValueError(f"{where}: sha256 {sha!r} is not 64 lowercase hex digits")
            out[name][role] = Pin(roots[pin["root"]] / rel, sha)
    return out


#: The teacher pairs, pinned in `teachers.toml`: PREREG-0's R2, the
#: provisional R3m (amendment 1) and the final R3, H2's matched it04000 pair
#: (amendment 2). The teacher is the better of R2 and R3 on DEV-2000 (within
#: 2pp, R3): R3, by +2.90pp on W2 (BASELINES.md), so a launch names `R3`.
PINS: dict[str, dict[str, Pin]] = load_pins()
#: `PINS` without the sha256: pair name -> role -> file.
PAIRS: dict[str, dict[str, pathlib.Path]] = {
    name: {role: pin.path for role, pin in pair.items()} for name, pair in PINS.items()
}


def manifest_sha256(path: str | pathlib.Path) -> str | None:
    """The sha256 that the nearest `SHA256SUMS` file in an ancestor directory
    records for `path`, or None if it has no entry."""
    p = pathlib.Path(path).resolve()
    for d in p.parents:
        m = d / "SHA256SUMS"
        if not m.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        for line in m.read_text(encoding="utf-8").splitlines():
            sha, _, name = line.strip().partition("  ")
            if name.removeprefix("./") == rel:
                return sha
        return None
    return None


def _verify(path: str | pathlib.Path, want: str | None = None) -> str:
    """The file's sha256; raises if it is not `want` (a pin) or a `SHA256SUMS`
    manifest disagrees."""
    sha = file_sha256(path)
    if want is not None and want != sha:
        raise TeacherMismatch(f"{path}: sha256 {sha} != pinned {want}")
    listed = manifest_sha256(path)
    if listed is not None and listed != sha:
        raise TeacherMismatch(f"{path}: sha256 {sha} != manifest {listed}")
    return sha


# ── merging teacher options into the student's classes ────────────────────


def check_option_map(classes: Sequence[int] | np.ndarray, n_classes: int) -> None:
    """Raise `OptionMapError` unless `classes` maps every teacher option to a
    class in `[0, n_classes)` and every class has at least one option."""
    m = np.asarray(classes)
    if m.ndim != 1 or len(m) == 0:
        raise OptionMapError(f"classes must be a non-empty vector, got shape {m.shape}")
    if n_classes < 1 or int(m.min()) < 0 or int(m.max()) >= n_classes:
        raise OptionMapError(f"classes {m.tolist()} outside [0, {n_classes})")
    hit = np.bincount(m.astype(np.int64), minlength=n_classes)
    if (hit == 0).any():
        raise OptionMapError(f"classes {np.flatnonzero(hit == 0).tolist()} have no teacher option")


def merge_logp(logp: torch.Tensor, classes: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Class log-probabilities `[B, n_classes]` from option log-probabilities.

    `logp` `[B, A]` are per-option log-probabilities (-inf allowed), `classes`
    `[B, A]` the class of each option (-1 = no option). A class's value is
    the log-sum-exp of its members'; a class with no member is -inf.
    """
    b, _ = logp.shape
    valid = classes >= 0
    idx = classes.clamp(min=0)
    lp = logp.masked_fill(~valid, float("-inf"))
    neg = torch.full((b, n_classes), float("-inf"), dtype=logp.dtype, device=logp.device)
    top = neg.scatter_reduce(1, idx, lp, reduce="amax", include_self=True)
    safe = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    shifted = torch.where(valid, (lp - safe.gather(1, idx)).exp(), torch.zeros_like(lp))
    s = torch.zeros_like(top).scatter_add(1, idx, shifted)
    return torch.where(s > 0, safe + s.log(), neg)


def _merge(
    logp: torch.Tensor,
    n_opts: Sequence[int],
    classes: Sequence[Sequence[int]],
    n_classes: Sequence[int] | None,
) -> list[np.ndarray]:
    """Per decision: check its map, then merge (float64 numpy out).

    `n_classes[i]` is the student's class count for decision `i`; None infers
    it as `max(classes[i]) + 1`, which cannot catch a missing last class.
    """
    mem, ncls = _member_classes(tuple(logp.shape), n_opts, classes, n_classes)
    return _merge_start(logp, mem, ncls)()


def _member_classes(
    shape: tuple[int, ...],
    n_opts: Sequence[int],
    classes: Sequence[Sequence[int]],
    n_classes: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """`_merge`'s host half: the checked member-class matrix `[B, A]` (-1 past
    each decision's options) and the class counts."""
    if len(classes) != shape[0]:
        raise OptionMapError(f"{len(classes)} class maps for {shape[0]} decisions")
    if n_classes is not None and len(n_classes) != len(classes):
        raise OptionMapError(f"{len(n_classes)} class counts for {len(classes)} decisions")
    ncls = []
    mem = np.full(shape, -1, np.int64)
    for i, (n, c_of) in enumerate(zip(n_opts, classes, strict=True)):
        m = np.asarray(c_of, np.int64)
        if len(m) != n:
            raise OptionMapError(f"decision {i}: {len(m)} classes for {n} teacher options")
        c = int(n_classes[i]) if n_classes is not None else (int(m.max()) + 1 if len(m) else 0)
        check_option_map(m, c)
        mem[i, :n] = m
        ncls.append(c)
    return mem, np.asarray(ncls, np.int64)


def _merge_start(
    logp: torch.Tensor, mem: np.ndarray, ncls: np.ndarray
) -> Callable[[], list[np.ndarray]]:
    """Queue the merge of option log-probs `logp` (any device) by the checked
    member-class matrix `mem`; the returned call waits for it and gives each
    decision's float64 class log-probs."""
    m = torch.from_numpy(mem)
    if logp.device.type == "cuda":
        m = m.pin_memory().to(logp.device, non_blocking=True)
    out = merge_logp(logp.double(), m.to(logp.device), int(ncls.max()))

    def finish() -> list[np.ndarray]:
        z = out.cpu().numpy()
        return [z[i, :c] for i, c in enumerate(ncls.tolist())]

    return finish


#: The fewest rows `length_groups` puts in a group: a smaller forward costs
#: more in kernel launches than its padding saves.
MIN_GROUP_ROWS = 32


def length_groups(lens: Sequence[int] | np.ndarray, groups: int) -> list[np.ndarray]:
    """Row indices split into up to `groups` groups of about equal size,
    sorted by `lens` (stable), so each group pads to a shorter length than
    the whole batch would. Each group has at least `MIN_GROUP_ROWS` rows;
    `groups` <= 1, or too few rows, gives one group in the original order."""
    n = len(lens)
    g = min(int(groups), n // MIN_GROUP_ROWS)
    if g <= 1:
        return [np.arange(n, dtype=np.int64)]
    order = np.argsort(np.asarray(lens, np.int64), kind="stable")
    return list(np.array_split(order, g))


def decision_classes(d: Any) -> tuple[np.ndarray, int]:
    """`(class of each member, class count)` of an env decision: an
    `env.E2EDecision` (`classes`, `members`) or a `contracts.EnvDecision`
    (`member_class`, `rep`)."""
    if hasattr(d, "member_class"):
        return np.asarray(d.member_class, np.int64), len(d.rep)
    return np.asarray(d.classes, np.int64), len(d.members)


# ── the combat teacher ────────────────────────────────────────────────────


class SctxDecision(Protocol):
    """#1081's per-decision `S_ctx` tokens (`combat_ppo.encode.Decision`)."""

    cat: np.ndarray  # int16 [T, 15], RareMap applied
    cont: np.ndarray  # float16 [T, 122]
    sec: np.ndarray  # int8 [T]
    n_act: int  # the last n_act tokens are the actions, in legal-action order


@dataclass
class SctxTokens:
    """`SctxDecision` built from arrays (`CombatTeacher.encode`, `sctx_rows`)."""

    cat: np.ndarray
    cont: np.ndarray
    sec: np.ndarray
    n_act: int


@dataclass
class SctxRow:
    """X-C's view of one combat row: its `S_ctx` tokens and option merge."""

    row: int  # index in the row batch
    tokens: SctxTokens
    classes: np.ndarray  # class of each option (= member), env order
    n_classes: int


def sctx_rows(rb: Mapping[str, np.ndarray], rows: Sequence[int] | None = None) -> list[SctxRow]:
    """The combat teacher's view (X-C) of combat rows of a `contracts` row batch.

    `rows` default to every combat row. Each row's tokens are
    `contracts.teacher_slice(rb, i)` (tests check byte equality; this
    function computes the offsets once for all rows): the `S_ctx` block
    `[sctx_lo, sctx_lo + n_sctx)`, its first `contracts.SCTX_N_CAT`
    categorical and `contracts.SCTX_N_CONT` continuous columns and its
    section ids. The block's last `n_mem` tokens are the options in
    legal-action order (`validate_row_batch` checks that layout), merged by
    `mem_class`.

    Raises:
        ValueError: a selected row is not a combat row, or its block is not
            an `S_ctx` block.
    """
    from . import contracts as C

    off = C.row_offsets(rb)
    t_off, m_off = off["tok"], off["mem"]
    nm = np.asarray(rb["n_mem"], np.int64)
    is_c = np.asarray(rb["is_combat"], bool)
    sel = np.flatnonzero(is_c) if rows is None else np.asarray(rows, np.int64)
    out = []
    for i in sel.tolist():
        if not is_c[i]:
            raise ValueError(f"row {i} is not a combat row")
        lo = int(t_off[i] + rb["sctx_lo"][i])
        n = int(rb["n_sctx"][i])
        sec = np.asarray(rb["sec"][lo : lo + n])
        if n < 1 + nm[i] or (sec >= C.SCTX_N_SECTION).any() or lo + n > t_off[i + 1]:
            raise ValueError(f"row {i}: no S_ctx block of {nm[i]} options")
        tok = SctxTokens(
            cat=np.asarray(rb["cat"][lo : lo + n, : C.SCTX_N_CAT]),
            cont=np.asarray(rb["cont"][lo : lo + n, : C.SCTX_N_CONT]),
            sec=sec,
            n_act=int(nm[i]),
        )
        cls = np.asarray(rb["mem_class"][m_off[i] : m_off[i + 1]], np.int64)
        out.append(SctxRow(i, tok, cls, int(rb["n_classes"][i])))
    return out


@dataclass
class _SctxFlat:
    """S_ctx tokens of `len(n)` decisions stored back to back: decision `i`'s
    tokens are rows `[off[i], off[i] + n[i])` of `cat`, `cont` and `sec`; its
    last `n_act[i]` tokens are its options."""

    cat: np.ndarray
    cont: np.ndarray
    sec: np.ndarray
    n: np.ndarray  # int64 [B]
    n_act: np.ndarray  # int64 [B]

    def take(self, idx: np.ndarray) -> _SctxFlat:
        off = np.concatenate([[0], np.cumsum(self.n)])
        src = _ranges(off[idx], self.n[idx])
        return _SctxFlat(self.cat[src], self.cont[src], self.sec[src], self.n[idx], self.n_act[idx])


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """The concatenated index ranges `[starts[i], starts[i] + lens[i])`."""
    lens = np.asarray(lens, np.int64)
    if not len(lens) or int(lens.sum()) == 0:
        return np.zeros(0, np.int64)
    first = np.cumsum(lens) - lens
    return np.repeat(np.asarray(starts, np.int64) - first, lens) + np.arange(int(lens.sum()))


def _row_col(lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, column) of every element of rows of lengths `lens`, row-major."""
    lens = np.asarray(lens, np.int64)
    r = np.repeat(np.arange(len(lens)), lens)
    return r, _ranges(np.zeros(len(lens), np.int64), lens)


def _flat_of(ds: Sequence[SctxDecision]) -> _SctxFlat:
    return _SctxFlat(
        np.concatenate([np.asarray(d.cat) for d in ds]),
        np.concatenate([np.asarray(d.cont) for d in ds]),
        np.concatenate([np.asarray(d.sec) for d in ds]),
        np.asarray([len(d.sec) for d in ds], np.int64),
        np.asarray([int(d.n_act) for d in ds], np.int64),
    )


def _sctx_flat(
    rb: Mapping[str, np.ndarray], rows: Sequence[int] | None
) -> tuple[np.ndarray, _SctxFlat, np.ndarray]:
    """`sctx_rows` without the per-row loop: the selected rows, their S_ctx
    tokens back to back, and each row's first member offset. Same checks."""
    from . import contracts as C

    off = C.row_offsets(rb)
    t_off, m_off = off["tok"], off["mem"]
    nm = np.asarray(rb["n_mem"], np.int64)
    is_c = np.asarray(rb["is_combat"], bool)
    sel = np.flatnonzero(is_c) if rows is None else np.asarray(rows, np.int64)
    if len(sel) and not is_c[sel].all():
        raise ValueError(f"row {int(sel[~is_c[sel]][0])} is not a combat row")
    lo = t_off[sel] + np.asarray(rb["sctx_lo"], np.int64)[sel]
    n = np.asarray(rb["n_sctx"], np.int64)[sel]
    src = _ranges(lo, n)
    sec = np.asarray(rb["sec"])[src]
    bad = (n < 1 + nm[sel]) | (lo + n > t_off[sel + 1])
    if len(src):
        out = np.zeros(len(sel), bool)
        np.logical_or.at(out, np.repeat(np.arange(len(sel)), n), sec >= C.SCTX_N_SECTION)
        bad |= out
    if bad.any():
        i = int(sel[np.flatnonzero(bad)[0]])
        raise ValueError(f"row {i}: no S_ctx block of {nm[i]} options")
    cat = np.asarray(rb["cat"])[src, : C.SCTX_N_CAT]
    cont = np.asarray(rb["cont"])[src, : C.SCTX_N_CONT]
    return sel, _SctxFlat(cat, cont, sec, n, nm[sel]), m_off[sel]


def _check_option_maps(mem: np.ndarray, n_opts: np.ndarray, n_classes: np.ndarray) -> None:
    """`check_option_map` of every row of a member-class matrix at once: row
    `i`'s first `n_opts[i]` entries map its options onto `[0, n_classes[i])`."""
    b, a = mem.shape
    real = np.arange(a)[None, :] < n_opts[:, None]
    c = n_classes[:, None]
    oob = real & ((mem < 0) | (mem >= c))
    bad = (n_opts < 1) | (n_classes < 1) | oob.any(1)
    if not bad.any():
        w = int(n_classes.max())
        hit = np.zeros((b, w), np.int64)
        r, k = np.nonzero(real)
        np.add.at(hit, (r, mem[r, k]), 1)
        bad = ((hit == 0) & (np.arange(w)[None, :] < c)).any(1)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        check_option_map(mem[i, : n_opts[i]], int(n_classes[i]))  # raises its message
        raise OptionMapError(f"decision {i}: classes {mem[i, : n_opts[i]].tolist()}")


class CombatTeacher:
    """#1081's `S_ctx` combat net as a teacher over merged classes."""

    kind = "combat"

    def __init__(
        self,
        path: str | pathlib.Path,
        device: str | torch.device = "cpu",
        *,
        bf16: bool = False,
        expect_sha256: str | None = None,
    ) -> None:
        """`expect_sha256`: the file's pinned sha256 (`PINS`); a file that
        differs raises `TeacherMismatch` before it is loaded."""
        from ppo.combat_ctx import tokens as T
        from ppo.combat_ctx.data import RareMap
        from ppo.combat_ctx.model import CombatCtxNet

        from . import contracts as C

        self.path = str(path)
        self.sha256 = _verify(path, expect_sha256)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("arm") != "S_ctx":
            raise TeacherMismatch(f"{path}: arm {ck.get('arm')!r}, the teacher reads S_ctx")
        if ck["vocab"] != T.vocab().digest() or ck["n_cont"] != T.N_CONT:
            raise TeacherMismatch(f"{path}: token schema differs from this checkout's")
        self.config = dict(ck["config"], dropout=0.0)
        self.net = CombatCtxNet.from_config(self.config)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.device = torch.device(device)
        self.bf16 = bool(bf16)
        self.rare_json = ck["rare"]
        self.rare = RareMap.from_json(ck["rare"])
        #: The RareMap a tokenizer must apply to the S_ctx content slots.
        self.rare_sha256 = rare_sha256(ck["rare"])
        #: `contracts.teacher_stamp`: rows store the S_ctx content slots through
        #: this teacher's RareMap, so rows and a teacher match only if it does.
        self.teacher_stamp = C.teacher_stamp(
            ckpt_sha256=self.sha256, rare=ck["rare"], sctx_vocab=ck["vocab"]
        )
        self.decisions = (ck.get("settings") or {}).get("decisions")
        self.T = T
        self._enc: Any = None
        #: Forwards per scoring call (`logits_tensor`); the learner sets its
        #: `serve_groups`. 1 = one batch padded to its longest row.
        self.serve_groups = 1
        #: Score pad-free (`_logits_packed`): no padded token is computed, and
        #: the last block runs at the option tokens only. Same logits up to float
        #: rounding; `serve_groups` is then moot. The learner's `--packed-serve`.
        self.packed = False

    def stamps(self) -> dict[str, Any]:
        """Provenance; `teacher` is the contract's `teacher_stamp`, the value
        `contracts.stamps(teacher=...)` takes."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "decisions": self.decisions,
            "rare_sha256": self.rare_sha256,
            "teacher": dict(self.teacher_stamp),
        }

    def encode(self, d: Any) -> SctxTokens:
        """#1081's `S_ctx` tokens of an env combat decision (worker side).

        Reads `d.snap` (the honest snapshot with this fight's intent history),
        `d.bv`, `d.cs` and `d.actions` (`cs.legal_actions()`), as
        `combat_ppo.encode.Encoder.encode` does. The X-C path reads the same
        tokens from the student's rows instead (`sctx_rows`).
        """
        if self._enc is None:
            from ppo.combat_ppo.encode import Encoder

            self._enc = Encoder("S_ctx", self.rare_json, priv_k=0)
        x = self._enc.encode(d.snap, d.bv, d.cs, d.actions)
        return SctxTokens(x.cat, x.cont, x.sec, int(x.n_act))

    def _inputs(self, ds: Sequence[SctxDecision]) -> dict[str, torch.Tensor]:
        """`combat_ctx.play.CtxNet.logits`'s padded batch (same layout)."""
        return self._inputs_flat(_flat_of(ds))

    def _inputs_flat(self, f: _SctxFlat) -> dict[str, torch.Tensor]:
        """`_inputs` of decisions stored back to back (no per-decision loop)."""
        T = self.T
        b = len(f.n)
        tm, am = int(f.n.max()), int(f.n_act.max())
        r, c = _row_col(f.n)
        cat = np.tile(T.vocab().pads.astype(np.int64), (b, tm, 1))
        cont = np.zeros((b, tm, T.N_CONT), np.float32)
        sec = np.zeros((b, tm), np.int64)
        pad = np.ones((b, tm), bool)
        cat[r, c] = f.cat
        cont[r, c] = np.asarray(f.cont, np.float16).astype(np.float32)
        sec[r, c] = f.sec
        pad[r, c] = False
        act_pos = np.zeros((b, am), np.int64)
        amask = np.zeros((b, am), bool)
        ra, ca = _row_col(f.n_act)
        act_pos[ra, ca] = (f.n - f.n_act)[ra] + ca
        amask[ra, ca] = True
        dev = self.device
        # Pinned staging on CUDA: a pageable upload this size waits for every kernel
        # already queued (the student's forwards), a pinned one does not.
        pin = dev.type == "cuda"
        return {
            k: (torch.from_numpy(v).pin_memory() if pin else torch.from_numpy(v)).to(
                dev, non_blocking=pin
            )
            for k, v in (
                ("cat", cat),
                ("cont", cont),
                ("sec", sec),
                ("pad", pad),
                ("act_pos", act_pos),
                ("act_mask", amask),
            )
        }

    @torch.no_grad()
    def logits_tensor(self, ds: Sequence[SctxDecision]) -> torch.Tensor:
        """Per-option logits `[B, A]` (fp32; -inf past each row's `n_act`).

        With `serve_groups` > 1 the decisions run as that many forwards of
        about equal size, sorted by token count, each padded to its own
        longest row instead of the batch's (`length_groups`). A row's logits
        then differ from the one-batch forward's only by float rounding.
        """
        return self._logits_flat(_flat_of(ds))

    @torch.no_grad()
    def _logits_flat(self, f: _SctxFlat) -> torch.Tensor:
        if self.packed:
            return self._logits_packed(f)
        parts = length_groups(f.n, self.serve_groups)
        if len(parts) == 1:
            return self._logits_one(f)
        outs = [self._logits_one(f.take(p)) for p in parts]
        z = torch.full(
            (len(f.n), max(o.shape[1] for o in outs)), float("-inf"), device=outs[0].device
        )
        for p, o in zip(parts, outs, strict=True):
            z[torch.from_numpy(p).to(z.device, non_blocking=True), : o.shape[1]] = o
        return z

    def _logits_one(self, f: _SctxFlat) -> torch.Tensor:
        x = self._inputs_flat(f)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            logits, _ = self.net(
                x["cat"], x["cont"], x["sec"], x["pad"], x["act_pos"], x["act_mask"]
            )
        return logits.float()

    def _logits_packed(self, f: _SctxFlat, attn: str = "auto") -> torch.Tensor:
        """`_logits_one` over packed tokens (`net.Varlen`; `attn` its mode):
        embeddings, norms and MLPs see no padding, and the last block's query
        and MLP run at the option tokens only (the policy head reads nothing else)."""
        import torch.nn.functional as F

        from .net import Varlen

        net, dev = self.net, self.device
        b = len(f.n)
        cu = np.zeros(b + 1, np.int64)
        np.cumsum(f.n, out=cu[1:])
        cq = np.zeros(b + 1, np.int64)
        np.cumsum(f.n_act, out=cq[1:])
        am = int(f.n_act.max())
        ra, ca = _row_col(f.n_act)
        pin = dev.type == "cuda"

        def up(x: np.ndarray) -> torch.Tensor:
            h = torch.from_numpy(np.ascontiguousarray(x))
            return (h.pin_memory() if pin else h).to(dev, non_blocking=pin)

        idx = up(np.concatenate([cu, cq, (cu[1:] - f.n_act)[ra] + ca, ra * am + ca]))
        cu_t, cq_t, at, dst = torch.split(idx, [b + 1, b + 1, len(ra), len(ra)])
        cat = up(f.cat).long()
        cont = up(np.asarray(f.cont, np.float16)).float()  # `_inputs_flat`'s rounding
        sec = up(f.sec).long()
        n = int(cu[-1])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=self.bf16):
            # SinusoidalProj flattens from dim 2: one batch row of all the tokens.
            x = net.emb_sec(sec) + net.cont_proj(cont[None])[0]
            for i, e in enumerate(net.emb):
                x = x + e(cat[:, i])
            full = Varlen(cu_t.int(), cu_t.int(), int(f.n.max()), int(f.n.max()), n, n, attn)
            blocks = list(net.blocks)
            for k, blk in enumerate(blocks):
                d = x.shape[1]
                q, kk, v = blk.qkv(blk.n1(x)).view(n, 3, blk.heads, d // blk.heads).unbind(1)
                if k < len(blocks) - 1:
                    x = x + blk.out(full(q, kk, v).reshape(n, d))
                else:  # the last block: queries at the option tokens only
                    one = Varlen(cq_t.int(), cu_t.int(), am, int(f.n.max()), len(ra), n, attn)
                    x = x[at] + blk.out(one(q[at], kk, v).reshape(len(ra), d))
                h = blk.n2(x)
                x = x + blk.w2(F.silu(blk.w1(h)) * blk.v(h))
            if not blocks:
                x = x[at]
            z = net.policy_head(net.norm(x)).squeeze(-1).float()
        out = z.new_full((b * am,), float("-inf"))
        out[dst] = z
        return out.view(b, am)

    def logits(self, ds: Sequence[SctxDecision]) -> list[np.ndarray]:
        """Per decision, its options' logits (float64), as `CtxNet.logits`."""
        z = self.logits_tensor(ds).double().cpu().numpy()
        return [z[i, : int(d.n_act)] for i, d in enumerate(ds)]

    def logp(
        self,
        ds: Sequence[SctxDecision],
        classes: Sequence[Sequence[int]],
        n_classes: Sequence[int] | None = None,
    ) -> list[np.ndarray]:
        """Per decision, log-probabilities over its merged classes (float64).

        `classes[i][k]` is the class of decision `i`'s option `k` (legal-action
        order); `n_classes[i]` its class count (see `_merge`).
        """
        lp = torch.log_softmax(self.logits_tensor(ds).double(), dim=-1)
        return _merge(lp, [int(d.n_act) for d in ds], classes, n_classes)

    def logp_decisions(
        self, decisions: Sequence[Any], tokens: Sequence[SctxDecision] | None = None
    ) -> list[np.ndarray]:
        """`logp` for env combat decisions (`classes`, `members`); `tokens`
        default to `encode(d)` per decision."""
        toks = list(tokens) if tokens is not None else [self.encode(d) for d in decisions]
        cls = [decision_classes(d) for d in decisions]
        return self.logp(toks, [c for c, _ in cls], [n for _, n in cls])

    def logp_rows(
        self, rb: Mapping[str, np.ndarray], rows: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """X-C: `logp` over the merged classes of combat rows of a row batch
        (`sctx_rows`; default every combat row), float64. Bit-identical to
        `logp` on the rows' `sctx_rows` tokens, without a per-row loop."""
        return self.logp_rows_start(rb, rows)()

    @torch.no_grad()
    def logp_rows_start(
        self, rb: Mapping[str, np.ndarray], rows: Sequence[int] | None = None
    ) -> Callable[[], list[np.ndarray]]:
        """`logp_rows` in two halves: this call checks the rows and queues the
        forward and the merge on the device; the returned call waits for them."""
        sel, f, m_lo = _sctx_flat(rb, rows)
        if not len(sel):
            return list
        lp = torch.log_softmax(self._logits_flat(f).double(), dim=-1)
        ra, ca = _row_col(f.n_act)
        mem = np.full(tuple(lp.shape), -1, np.int64)
        mem[ra, ca] = np.asarray(rb["mem_class"], np.int64)[m_lo[ra] + ca]
        ncls = np.asarray(rb["n_classes"], np.int64)[sel]
        _check_option_maps(mem, f.n_act, ncls)
        return _merge_start(lp, mem, ncls)

    def tprob_rows(
        self, rb: Mapping[str, np.ndarray], rows: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """The row batch's `tprob` for combat rows (`t_src` 1): class
        probabilities, float32, renormalised after the cast."""
        out = []
        for lp in self.logp_rows(rb, rows):
            p = np.exp(lp).astype(np.float32)
            out.append(p / p.sum(dtype=np.float32))
        return out


# ── the run teacher ───────────────────────────────────────────────────────


class RunTeacher:
    """A `RunPolicyNet` run net as a teacher over merged classes.

    It reads the raw `obs`, raw event previews included, the input its
    baselines were measured on (SCHEMA.md section 3.2). On an option that
    gains a relic the raw preview carries the rolled relic's pickup, which
    the student never sees (section 2.7), so the KS target there is partly
    out of the student's reach. `honest_previews=True` is SCHEMA open
    question 15's fallback: featurize `contracts.honest_obs(obs)` instead
    (masked previews, no seed). Re-measure the teacher before acting on it.
    """

    kind = "run"

    def __init__(
        self,
        path: str | pathlib.Path,
        device: str | torch.device = "cpu",
        *,
        honest_previews: bool = False,
        expect_sha256: str | None = None,
    ) -> None:
        """`expect_sha256`: the file's pinned sha256 (`PINS`); a file that
        differs raises `TeacherMismatch` before it is loaded."""
        from ppo import schema
        from ppo.model import RunPolicyNet, load_compat

        self.path = str(path)
        self.sha256 = _verify(path, expect_sha256)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ck.get("cfg") if isinstance(ck, dict) else None
        got = cfg.get("features") if isinstance(cfg, dict) else None
        if got != RUN_FEATURES:
            raise TeacherMismatch(
                f"{path}: cfg features {got!r} != {RUN_FEATURES!r} (NetPolicy would fall "
                "back to the default features silently)"
            )
        self.features = schema.parse_features(got)
        self.net = RunPolicyNet(
            dim=int(cfg.get("dim", 192)),
            n_layers=int(cfg.get("n_layers", 3)),
            n_heads=int(cfg.get("n_heads", 4)),
        )
        # `load_compat`, as NetPolicy loads it: the baselines serve it so.
        load_compat(self.net, ck.get("state_dict", ck))
        self.net.eval().to(device)
        self.device = torch.device(device)
        self.iterations = cfg.get("iterations")
        self.honest_previews = bool(honest_previews)

    def stamps(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "features": RUN_FEATURES,
            "iterations": self.iterations,
            "honest_previews": self.honest_previews,
        }

    def featurize(self, obs: dict, actions: list[dict], metrics: dict) -> Any:
        """The worker-side half: `featurize_row` with this checkpoint's features."""
        from ppo.encoder import featurize_row

        return featurize_row(obs, actions, metrics, features=self.features)

    def featurize_decision(self, d: Any) -> Any:
        """`featurize` of an env OOC decision: an `env.E2EDecision` (`run_obs`,
        `actions`, `metrics`) or a `contracts.EnvDecision` (`obs`, `actions`,
        `metrics`). The raw `obs`, or `honest_obs(obs)` with `honest_previews`."""
        obs = d.run_obs if hasattr(d, "run_obs") else d.obs
        if self.honest_previews:
            from . import contracts as C

            obs = C.honest_obs(obs)
        return self.featurize(obs, d.actions, d.metrics)

    @torch.no_grad()
    def logits_tensor(self, feats: Sequence[Any]) -> torch.Tensor:
        """Per-option logits `[B, A]` (fp32; -inf past each row's options)."""
        logits, _value, _aux = self.net(self._batch(feats))
        return logits.float()

    def _batch(self, feats: Sequence[Any]) -> Any:
        """`model.collate` onto the device; on CUDA through pinned memory, so the
        upload does not wait for kernels already queued (same tensors)."""
        from ppo.model import Batch, collate

        if self.device.type != "cuda":
            return collate(list(feats), self.device)
        b = collate(list(feats))
        up = {
            f.name: getattr(b, f.name).pin_memory().to(self.device, non_blocking=True)
            for f in dataclasses.fields(b)
            if isinstance(getattr(b, f.name), torch.Tensor)
        }
        return Batch(**up, device=self.device)

    def logits(self, feats: Sequence[Any]) -> list[np.ndarray]:
        z = self.logits_tensor(feats).double().cpu().numpy()
        return [z[i, : f.n_actions] for i, f in enumerate(feats)]

    def logp(
        self,
        feats: Sequence[Any],
        classes: Sequence[Sequence[int]],
        n_classes: Sequence[int] | None = None,
    ) -> list[np.ndarray]:
        """Per decision, log-probabilities over its merged classes (float64).

        `classes[i][k]` is the class of action `k` of the `actions` list the
        decision was featurized with; `n_classes[i]` its class count.
        """
        return self.logp_start(feats, classes, n_classes)()

    @torch.no_grad()
    def logp_start(
        self,
        feats: Sequence[Any],
        classes: Sequence[Sequence[int]],
        n_classes: Sequence[int] | None = None,
    ) -> Callable[[], list[np.ndarray]]:
        """`logp` in two halves: this call queues the forward and the merge on
        the device; the returned call waits for them."""
        lp = torch.log_softmax(self.logits_tensor(feats).double(), dim=-1)
        mem, ncls = _member_classes(
            tuple(lp.shape), [f.n_actions for f in feats], classes, n_classes
        )
        return _merge_start(lp, mem, ncls)

    def logp_decisions(
        self, decisions: Sequence[Any], feats: Sequence[Any] | None = None
    ) -> list[np.ndarray]:
        """`logp` for env OOC decisions; `feats` default to `featurize_decision`."""
        fs = list(feats) if feats is not None else [self.featurize_decision(d) for d in decisions]
        cls = [decision_classes(d) for d in decisions]
        return self.logp(fs, [c for c, _ in cls], [n for _, n in cls])


@dataclass
class Teachers:
    """The teacher pair the warm start uses (either may be None: λ_KS = 0 for
    a row type without a teacher, plan §7)."""

    combat: CombatTeacher | None
    run: RunTeacher | None

    def stamps(self) -> dict[str, Any]:
        return {
            "combat": None if self.combat is None else self.combat.stamps(),
            "run": None if self.run is None else self.run.stamps(),
        }

    def logp_decisions(self, decisions: Sequence[Any]) -> list[np.ndarray | None]:
        """Per env decision (`is_combat` picks the teacher), its teacher's
        class log-probabilities, or None where that row type has no teacher."""
        out: list[np.ndarray | None] = [None] * len(decisions)
        for combat, t in ((True, self.combat), (False, self.run)):
            idx = [i for i, d in enumerate(decisions) if bool(d.is_combat) == combat]
            if t is None or not idx:
                continue
            for i, lp in zip(idx, t.logp_decisions([decisions[i] for i in idx]), strict=True):
                out[i] = lp
        return out


def load_pair(
    name: str | None = None,
    device: str | torch.device = "cpu",
    *,
    combat: str | pathlib.Path | None = None,
    run: str | pathlib.Path | None = None,
) -> Teachers:
    """A named pair from `PINS` ("R2", "R3m" or "R3"), each file checked against
    its pinned sha256, or the pair at the paths `combat` and `run`.
    Each file is also checked against the nearest `SHA256SUMS` manifest, when
    it has one. A file that differs raises `TeacherMismatch`."""
    want: dict[str, str | None] = dict.fromkeys(ROLES)
    if name is not None:
        if combat is not None or run is not None:
            raise ValueError("give a pair name or its paths, not both")
        pair = PINS[name]
        combat, run = pair["combat"].path, pair["run"].path
        want = {role: pair[role].sha256 for role in ROLES}
    if combat is None or run is None:
        raise ValueError("give a pair name, or both the combat and the run path")
    return Teachers(
        CombatTeacher(combat, device, expect_sha256=want["combat"]),
        RunTeacher(run, device, expect_sha256=want["run"]),
    )
