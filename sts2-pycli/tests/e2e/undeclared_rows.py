"""The undeclared-Ironclad-row hash (GH #1197; the #1374 check, SCHEMA.md section 13.4).

Random-play Ironclad A10 runs on the e2e env, in a process that does not declare the
"chars" fill event, each decision tokenized as a worker does (`tokens.encode(d)`, PRIV
included). Every policy and PRIV array of every row is hashed. The hashes of `main`'s
tokenizer and wheel are committed (`data/undeclared_ironclad_rows.json`), and
`test_e2e_chars.py::test_undeclared_ironclad_rows_unchanged` holds this checkout to them:
a change that leaks into rows without the event moves a hash.

The module uses only what main had before GH #1197, so the same file generates the
committed hashes on a checkout of main.

REGENERATING THE DATA FILE. Never regenerate it from a branch, this one included: a
branch's own tokenizer would write its own rows into the file and the test would pass
on exactly the leak it exists to catch. Generate it ONLY from `origin/main` WITHOUT the
GH #1197 tokenizer change (before this branch merges: `origin/main` itself, or
`git merge-base origin/main HEAD`; the committed file came from f6716cae7), in a
separate, detached worktree with its own freshly built wheel, then remove the worktree:

    REPO=$(git rev-parse --show-toplevel)           # this checkout
    W=~/runs/e2e/tmp/undeclared-main                 # not /tmp (the laptop quota)
    git -C "$REPO" fetch origin
    git -C "$REPO" worktree add --detach "$W" origin/main
    cd "$W"/sts2-pycli
    uv sync --frozen                                 # its own .venv and wheel
    uv run --frozen python -c 'import sts2_rs; print(sts2_rs.__file__)'  # must be under $W
    STS2_E2E_CORPSE_POWERS=keep PYTHONPATH=python uv run --frozen python \\
        "$REPO/sts2-pycli/tests/e2e/undeclared_rows.py" \\
        --out "$REPO/sts2-pycli/tests/e2e/data/undeclared_ironclad_rows.json"
    cd "$REPO" && git worktree remove "$W"

The run is deterministic (two runs `cmp` equal); the file should change only when
main itself changes undeclared Ironclad rows on purpose.

`STS2_E2E_CORPSE_POWERS=keep` pins the engine's corpse powers, as the test does: the
hashes predate GH #1791's default strip, and without it 31 of the 96 runs move.

An engine-only change that moves random play (GH #1907 withholds lethal event
options, so seed 1300000623 survives its event) regenerates the file on its own
branch, but only when `git diff origin/main -- sts2-pycli` is empty, so the tokenizer
is main's. Check that just one or two seeds move, each from the step the engine
change explains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

SEED0 = 1_300_000_600  # the throwaway block (plan section 3.1)
N_RUNS = 96  # random A10 play dies in act 1: ~26 rows a run
CAP = 400  # decisions per run
DATA = Path(__file__).parent / "data" / "undeclared_ironclad_rows.json"


def _digest(arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]  # 64 bits a row: the file stays small


def row_digest(tok: Any) -> str:
    """sha256 of a row's policy arrays (`Tok.policy_arrays`) and its PRIV arrays."""
    priv = (tok.pcat, tok.pcont, tok.psec, tok.ppool_tok, tok.ppool_vid, tok.ppool_w)
    return _digest((*tok.policy_arrays(), *priv))


def combat_ctx_digest(d: Any) -> str:
    """sha256 of `ppo.combat_ctx`'s own S_ctx tokens of combat decision `d`: the honest
    snapshot of its belief view, its actions and rl2's features, as the combat PPO
    tokenizes them (no e2e tokenizer in the path)."""
    import sts2_rs
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.snapshot import honest_snapshot

    snap = honest_snapshot(d.bv, d.room)
    tk = T.tokens(snap, d.obs_actions, sts2_rs.rollout_features(d.bv, version=1), "S_ctx")
    return _digest((tk["cat"], tk["cont"], tk["sec"], np.asarray([tk["n_act"]], np.int64)))


def run_hashes(seed: int, cap: int = CAP) -> tuple[list[str], list[str]]:
    """Row hashes of one random-play Ironclad run (`random.Random(seed)` picks classes),
    and the `combat_ctx_digest` of each of its combat decisions."""
    from ppo.e2e import env as E
    from ppo.e2e import tokens as K

    env = E.E2EEnv(seed)
    d = env.reset()
    rng = random.Random(seed)
    rows: list[str] = []
    ctx: list[str] = []
    while isinstance(d, E.E2EDecision) and len(rows) < cap:
        rows.append(row_digest(K.encode(d)))
        if d.is_combat:
            ctx.append(combat_ctx_digest(d))
        d = env.step(rng.randrange(d.n_classes))
    return rows, ctx


def generate(seeds: list[int], cap: int = CAP) -> dict[str, Any]:
    """`{"seeds", "cap", "policy", "rows", "runs": {seed: [row hash, ...]},
    "combat_ctx": {seed: [hash, ...]}}` of this process."""
    import sts2_client
    from ppo.e2e import factory

    if factory.CHARACTER != "Ironclad" or os.environ.get(factory.CHARS_ENV):
        raise SystemExit("run as an undeclared Ironclad process (no STS2_E2E_CHARACTER/_CHARS)")
    sts2_client.GAME_VERSION = factory.GAME_VERSION
    got = {str(s): run_hashes(s, cap) for s in seeds}
    runs = {s: r for s, (r, _c) in got.items()}
    return {
        "seeds": seeds,
        "cap": cap,
        "policy": "random.Random(seed).randrange(n_classes)",
        "rows": sum(len(v) for v in runs.values()),
        "runs": runs,
        "combat_ctx": {s: c for s, (_r, c) in got.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    a = ap.parse_args()
    import sts2_rs

    print("sts2_rs:", sts2_rs.__file__)
    res = generate([SEED0 + i for i in range(N_RUNS)], a.cap)
    a.out.write_text(json.dumps(res, indent=0) + "\n", encoding="utf-8")
    print(f"{res['rows']} rows -> {a.out}")


if __name__ == "__main__":
    main()
