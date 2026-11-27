"""Carry an e2e net to the next lane build: the warm start of a later lane (GH #1535).

A lane on a later build (`ppo.e2e.build`) has its own schema hash (the build is part of
it), vocabulary (`vocab_<tag>.tsv`: the previous build's rows, then the new names) and
content prior, so `--init` refuses the previous build's checkpoint as it stands. The
shapes are the same (`VOCAB_ROWS` rows either way), so this tool carries the weights
one build forward (`build.previous`): v0.111.0 -> v0.113.0, v0.113.0 -> v0.114.0.

    STS2_E2E_GAME_VERSION=v0.114.0 STS2_GAME_VERSION=v0.114.0 \\
        python -m ppo.e2e.migrate SRC.pt DST.pt [--filled chars,fool,...] [--twins default]

* SRC is a `net_state` checkpoint of a lane on the previous build (`learner export
  --variant ema0999` of a pinned state). Its schema, vocab and game_version stamp are
  checked against that build's, so a file from another lane does not migrate silently.
* Every id of the previous build's table keeps its id and its rows (a card the new
  build removed, v0.114.0's RAPID_SHOT, keeps its row unused). Each new id starts with
  its own random init (the rows the net was built with and never trained), except the
  **twins**: a new card whose role an old one plays takes the mean of those cards' rows
  (`DEFAULT_TWINS`, per destination build). Per-card id rows carry most of what a net
  knows about a character's cards (docs/experiments/2026-11-16-card-id-ablation.md),
  and copying the starters' rows gave +3-6pp per character there.

  - v0.113.0: the Fool's Strike and Defend from the five starters' Strikes and Defends.
  - v0.114.0 (the Fool rework): TWIN_SHOT from RAPID_SHOT, which it replaces in
    `FoolCardPool` (the same slot; `CardId::REMOVED_IN_V0114`). Both are 1-cost
    `Shoot` attacks at one enemy that fire an arrow for a multi-hit (Rapid Shot 3x3,
    +1 hit; Twin Shot 7x2, +3 damage), so the old row is the new card's role.
    OPTIMISM (rare Skill: exhaust every Sadness in hand, draw one per Sadness) from
    GROWTH and PUNCH_OF_JOY, the two v0.113 cards whose effect reads the Sadness in
    hand and cashes it in (Growth transforms one into a copy, Punch of Joy turns them
    into Joy); no old card draws off Sadness, so its row starts as the Sadness
    payoff's.
* The prior buffer becomes the lane build's (`content_prior.table()`) when the lane
  declares "content-prior", and DST carries
  this process's stamps (schema, vocab, prior, game version, `--filled`), so the lane's
  `--init DST` passes `contracts.check_stamps`. The optimizer and the EMA shadows start
  fresh with the lane, as with any `--init`.

`provenance` in DST's cfg names SRC, its sha256 and stamps, and the twins applied.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import build as _build
from . import contracts as C
from . import vocab as V

# Per destination build: new card -> the old cards whose mean row it starts from
# (`--twins default`; the module docstring gives the reasons).
STARTERS = ("IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER")
DEFAULT_TWINS_BY_BUILD: dict[str, dict[str, tuple[str, ...]]] = {
    "v0.113.0": {
        "STRIKE_FOOL": tuple(f"STRIKE_{c}" for c in STARTERS),
        "DEFEND_FOOL": tuple(f"DEFEND_{c}" for c in STARTERS),
    },
    "v0.114.0": {
        "TWIN_SHOT": ("RAPID_SHOT",),
        "OPTIMISM": ("GROWTH", "PUNCH_OF_JOY"),
    },
}
#: This process's lane build's default twins (empty on v0.111.0, which has no source).
DEFAULT_TWINS: dict[str, tuple[str, ...]] = DEFAULT_TWINS_BY_BUILD.get(_build.GAME_VERSION, {})


class MigrateError(ValueError):
    """The source is not a checkpoint this tool carries to the lane build."""


def source_build() -> str:
    """The build whose nets this process's lane migrates (`build.previous`)."""
    try:
        return _build.previous(_build.GAME_VERSION)
    except ValueError:
        raise MigrateError(
            f"this process's lane build is {_build.GAME_VERSION}: set "
            f"{_build.BUILD_ENV} to the build to migrate to"
        ) from None


def _base_vocab() -> V.Vocab:
    """The source build's content-ID table."""
    return V.Vocab(V.read_tsv(V.tsv_path(source_build())))


def _base_schema_sha256() -> str:
    """The schema hash of a lane on the source build: this checkout's spec with the
    build and its S_ctx table sizes swapped (the only build-dependent parts)."""
    src = source_build()
    spec = C.schema_spec()
    spec["game_version"] = src
    spec["sctx"]["table_sizes"] = dict(C.SCTX_TABLE_SIZES_BY_BUILD[src])
    return C._json_sha256(spec)


def twin_rows(vocab: V.Vocab, twins: Mapping[str, Sequence[str]]) -> dict[int, list[int]]:
    """`new id -> old ids` for each twin the table has: every name must be a card row,
    the new one past the source build's rows, the old ones in it."""
    base_n = _base_vocab().n_used
    out: dict[int, list[int]] = {}
    for new, olds in twins.items():
        if not vocab.has("card", new):
            raise MigrateError(f"twin {new}: not a card of {V.TSV_PATH.name}")
        nid = vocab.id("card", new)
        if nid < base_n:
            raise MigrateError(f"twin {new}: id {nid} is an old row, not a new one")
        ids = []
        for o in olds:
            if not vocab.has("card", o):
                raise MigrateError(f"twin {new}: source {o} is not a card")
            oid = vocab.id("card", o)
            if oid >= base_n:
                raise MigrateError(f"twin {new}: source {o} (id {oid}) is new itself")
            ids.append(oid)
        out[nid] = ids
    return out


def vocab_tensors(state: Mapping[str, torch.Tensor]) -> list[str]:
    """The state-dict keys indexed by content id (dim 0 is `VOCAB_ROWS`), the prior
    buffer left out (it is replaced whole)."""
    return [
        k
        for k, v in state.items()
        if v.ndim >= 1 and v.shape[0] == C.VOCAB_ROWS and k.rsplit(".", 1)[-1] != "prior"
    ]


@torch.no_grad()
def apply_twins(state: dict[str, torch.Tensor], rows: Mapping[int, Sequence[int]]) -> list[str]:
    """Set each new id's rows to the mean of its sources' rows, in every id tensor."""
    keys = vocab_tensors(state)
    if not keys:
        raise MigrateError("no tensor of the state dict is indexed by content id")
    for k in keys:
        t = state[k]
        for nid, olds in rows.items():
            t[nid] = t[list(olds)].float().mean(0).to(t.dtype)
    return keys


def migrate(
    src: Mapping[str, Any],
    *,
    filled: Sequence[str],
    twins: Mapping[str, Sequence[str]],
    src_name: str = "",
    src_sha256: str = "",
) -> dict[str, Any]:
    """`src` (a `net_state` dict of the source build, `source_build()`) as a
    `net_state` dict of this process's lane."""
    from . import content_prior as CP
    from . import net as N

    src_build = source_build()
    cfg = src["cfg"]
    if src.get("format") != N.FORMAT:
        raise MigrateError(f"not an {N.FORMAT} checkpoint (format {src.get('format')!r})")
    if cfg.get("schema_sha256") != _base_schema_sha256():
        raise MigrateError(
            f"source schema {cfg.get('schema_sha256')!r} is not the {src_build} "
            f"schema {_base_schema_sha256()!r}"
        )
    if cfg.get("vocab_sha256") != _base_vocab().sha256:
        raise MigrateError(
            f"source vocab {cfg.get('vocab_sha256')!r} is not {V.tsv_path(src_build).name}'s"
        )
    old_stamps = dict(cfg.get("stamps") or {})
    # An unstamped net is a v0.111.0 one (the stamp came with the later builds).
    ok_gv = (None, src_build) if src_build == _build.DEFAULT_BUILD else (src_build,)
    if old_stamps.get("game_version") not in ok_gv:
        raise MigrateError(
            f"source game_version {old_stamps.get('game_version')!r}, not {src_build}"
        )
    if missing := sorted(set(old_stamps.get("filled") or ()) - set(filled)):
        raise MigrateError(f"source filled {missing}: --filled must keep them")

    vocab = V.load()
    names = {r.id: r.name for r in vocab.rows}
    rows = twin_rows(vocab, twins)
    state = {k: v.clone() for k, v in src["state_dict"].items()}
    keys = apply_twins(state, rows)
    net = N.E2ENet(N.NetConfig.from_dict(cfg["net"]))
    net.load_state_dict(state)
    # The prior goes with the "content-prior" fill event (`contracts.stamps`).
    prior = CP.sha256() if "content-prior" in filled else None
    want = C.stamps(
        vocab_sha256=vocab.sha256,
        prior_sha256=prior,
        filled=filled,
        teacher=None,
        wheel=old_stamps.get("wheel"),
        source_digest=old_stamps.get("source_digest"),
    )
    if prior is not None:
        net.set_prior(CP.table(), prior)
    out = N.net_state(
        net,
        schema_sha256=C.schema_sha256(),
        vocab_sha256=vocab.sha256,
        stamps={**old_stamps, **want},
    )
    out["cfg"]["provenance"] = {
        "migrated_from": src_name,
        "source_sha256": src_sha256,
        "source_stamps": old_stamps,
        "from_build": src_build,
        "to_build": _build.GAME_VERSION,
        "twins": {names[n]: [names[o] for o in olds] for n, olds in rows.items()},
        "id_tensors": keys,
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from . import net as N

    ap = argparse.ArgumentParser(prog="python -m ppo.e2e.migrate", description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--filled", default="", help="the lane's fill events, comma-separated")
    ap.add_argument(
        "--twins", choices=("default", "none"), default="default", help="the new cards' rows"
    )
    a = ap.parse_args(argv)
    if a.dst.exists():
        print(f"{a.dst} exists; not overwriting", file=sys.stderr)
        return 2
    src = torch.load(a.src, map_location="cpu", weights_only=True)
    try:
        out = migrate(
            src,
            filled=[f for f in a.filled.split(",") if f],
            twins=DEFAULT_TWINS if a.twins == "default" else {},
            src_name=str(a.src),
            src_sha256=N.file_sha256(a.src),
        )
    except (MigrateError, C.SchemaError) as e:
        print(f"migrate: {e}", file=sys.stderr)
        return 1
    torch.save(out, a.dst)
    st = out["cfg"]["stamps"]
    print(
        f"wrote {a.dst}: build {st['game_version']}, vocab {st['vocab_sha256'][:12]}, "
        f"prior {st['prior_sha256'][:12]}, twins {len(out['cfg']['provenance']['twins'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
