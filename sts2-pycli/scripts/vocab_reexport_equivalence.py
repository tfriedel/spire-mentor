#!/usr/bin/env python3
"""Prove a vocab re-export answers exactly like the net it replaces (GH #799).

The re-export (``reexport_champions_vocab.py``) widens the card / relic /
potion tables; on a v0.107.1 run no v0.111.0 id can appear, so the claim is
that the widened net reads exactly what the shipped one read. This checks the
claim on REAL decisions instead of synthetic ones:

1. Drive ``PpoRunEnv`` (default game build, v0.107.1) greedily with the NEW
   graph, over several seeds and ascensions, ``--act1 roll``.
2. At every yielded decision featurize the observation twice: once with the
   current (widened) registries, once with the OLD registries, rebuilt as the
   first ``old_n`` ids of each widened list (the pins are append-only; the old
   height is read off the shipped graph's own tables and the old ids are
   checked against the pin at ``--old-rev``).
3. Feed the old encoding to the shipped graph and the new one to the re-export
   and compare logits (over the real actions) and value, exactly.
4. Check the two encodings differ ONLY by the OOV/PAD index remap of the
   widened columns, i.e. no observation produced a v0.111.0 id and no label
   newly resolved to one.

Writes one JSON line per decision with the max |diff| and a summary line, so a
long sweep can be watched and resumed.

USAGE (from sts2-pycli, worktree wheel venv)::

    python scripts/vocab_reexport_equivalence.py --character Ironclad \\
        --old data/ppo/OLD.onnx --new data/ppo/NEW.onnx --min-decisions 2000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

#: (vocab accessor, pin file, categorical column, ONNX table) per widened kind.
KINDS = [
    ("cards", "card", 0, "emb_card"),
    ("relics", "relic_id", 2, "emb_relic"),
    ("potions", "potion", 3, "emb_potion"),
]


def table_rows(path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path))
    return {
        t: init.dims[0]
        for init in model.graph.initializer
        for t in (k[3] for k in KINDS)
        if init.name.endswith(f"{t}.weight")
    }


def pin_ids(kind: str, rev: str) -> list[str]:
    path = f"sts2-rs/crates/sts2-vocab/src/vocab_pins/{kind}.txt"
    text = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def old_registries(old_onnx: Path, old_rev: str) -> dict:
    """The pre-widening registries, as the shipped graph was exported over."""
    from ppo import vocab

    rows = table_rows(old_onnx)
    out = {}
    for acc, pin_kind, _col, table in KINDS:
        cur = getattr(vocab, acc)()
        old_n = rows[table] - 2
        assert cur.names[:old_n] == pin_ids(pin_kind, old_rev)[:old_n], pin_kind
        assert len(pin_ids(pin_kind, old_rev)) == old_n, (
            f"{pin_kind}: shipped graph indexes {old_n} ids, pin at {old_rev} lists "
            f"{len(pin_ids(pin_kind, old_rev))}"
        )
        reg = vocab.Registry.__new__(vocab.Registry)
        reg.names = cur.names[:old_n]
        reg.norm = cur.norm
        reg.idx = {}
        for i, n in enumerate(reg.names):
            reg.idx.setdefault(reg.norm(n), i)
        reg.oov = old_n
        reg.size = old_n + 1
        out[acc] = reg
    return out


@contextlib.contextmanager
def patched_vocab(regs: dict):
    from ppo import vocab

    saved = {acc: getattr(vocab, acc) for acc in regs}
    try:
        for acc, reg in regs.items():
            setattr(vocab, acc, lambda reg=reg: reg)
        yield
    finally:
        for acc, fn in saved.items():
            setattr(vocab, acc, fn)


def session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in s.get_inputs()}
    return (
        s,
        int(shapes["cont"][1]),
        int(shapes["act_mask"][1]),
        int(shapes["cat"][2]),
        int(shapes["cont"][2]),
    )


def feed(feat, t_max: int, a_max: int, n_cat: int, n_cont: int) -> dict:
    from ppo.schema import pad_to_fixed

    inp = pad_to_fixed(feat, t_max, a_max)
    inp["cat"] = np.ascontiguousarray(inp["cat"][:, :, :n_cat])
    inp["cont"] = np.ascontiguousarray(inp["cont"][:, :, :n_cont])
    return inp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--character", required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old-rev", default="HEAD", help="git rev whose pins the old graph indexes")
    ap.add_argument("--min-decisions", type=int, default=2000)
    ap.add_argument("--max-seeds", type=int, default=400)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--ascensions", type=int, nargs="+", default=[0, 3, 5, 8, 10])
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--boss-sims", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from ppo import vocab
    from ppo.encoder import featurize_row
    from ppo.env import Decision, PpoRunEnv, Terminal

    olds = old_registries(a.old, a.old_rev)
    news = {acc: getattr(vocab, acc)() for acc, *_ in KINDS}
    s_old, t_old, a_old, c_old, k_old = session(a.old)
    s_new, t_new, a_new, c_new, k_new = session(a.new)
    assert (t_old, a_old, c_old, k_old) == (t_new, a_new, c_new, k_new), "graph contract moved"

    # old -> new index map per widened column: ids keep their row, OOV and
    # PAD move to the new tail.
    remap = {}
    for acc, _k, col, _t in KINDS:
        o, n = olds[acc], news[acc]
        remap[col] = (o.oov, n.oov, o.size, n.size)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_dec = n_exact = 0
    max_dl = max_dv = 0.0
    n_runs = 0
    t0 = time.time()
    with a.out.open("w") as fh:
        for i in range(a.max_seeds):
            if n_dec >= a.min_decisions:
                break
            seed = a.seed_base + i
            asc = a.ascensions[i % len(a.ascensions)]
            env = PpoRunEnv(
                seed=seed,
                ascension=asc,
                character=a.character,
                sims=a.sims,
                boss_sims=a.boss_sims,
                act1="roll",
            )
            # The default build: the one every shipped champion was gated on.
            assert env.rm.game_version == "v0.107.1", env.rm.game_version
            d = env.reset()
            n_runs += 1
            while not isinstance(d, Terminal):
                # Combat resolves in-engine here (no --combat policy), so every
                # stop is a run-level Decision, never a CombatTurn.
                assert isinstance(d, Decision), type(d)
                feat_new = featurize_row(d.obs, d.actions, d.metrics)
                with patched_vocab(olds):
                    feat_old = featurize_row(d.obs, d.actions, d.metrics)
                    in_old = feed(feat_old, t_old, a_old, c_old, k_old)
                in_new = feed(feat_new, t_new, a_new, c_new, k_new)

                # (4) the encodings differ only by the tail remap.
                exp = in_old["cat"].copy()
                for col, (o_oov, n_oov, o_pad, n_pad) in remap.items():
                    c = in_old["cat"][..., col]
                    exp[..., col] = np.where(c == o_oov, n_oov, np.where(c == o_pad, n_pad, c))
                assert np.array_equal(exp, in_new["cat"]), (
                    f"seed {seed} d{d.step_index}: encoding differs beyond the OOV/PAD remap"
                )
                for k in ("cont", "sec", "pad_mask", "obs_len", "act_mask"):
                    assert np.array_equal(in_old[k], in_new[k]), k

                lo, vo = s_old.run(["logits", "value"], in_old)
                ln, vn = s_new.run(["logits", "value"], in_new)
                na = feat_new.n_actions
                lo, ln = lo[0, :na], ln[0, :na]
                dl = float(np.max(np.abs(lo - ln)))
                dv = float(np.max(np.abs(vo - vn)))
                exact = bool(np.array_equal(lo, ln) and np.array_equal(vo, vn))
                n_dec += 1
                n_exact += exact
                max_dl, max_dv = max(max_dl, dl), max(max_dv, dv)
                fh.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "asc": asc,
                            "step": d.step_index,
                            "surface": d.surface,
                            "n_actions": na,
                            "exact": exact,
                            "max_dlogit": dl,
                            "max_dvalue": dv,
                        }
                    )
                    + "\n"
                )
                d = env.step(int(np.argmax(ln)))
            fh.flush()
            print(
                f"[{a.character}] seed {seed} asc {asc} {d.outcome}: {n_dec} decisions, "
                f"{n_exact} exact, max dlogit {max_dl:.3g} dvalue {max_dv:.3g} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        summary = {
            "summary": True,
            "character": a.character,
            "old": a.old.name,
            "new": a.new.name,
            "runs": n_runs,
            "decisions": n_dec,
            "exact": n_exact,
            "max_dlogit": max_dl,
            "max_dvalue": max_dv,
        }
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    ok = n_dec >= a.min_decisions and max_dl < 1e-6 and max_dv < 1e-6
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
