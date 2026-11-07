"""X-C and X-T on recorded states, through the env path (plan §9, WP3b).

Replays recorded runs (`ppo.e2e.recorder`) with a `tokens.Tokenizer` driven
by the replay's `reset_fight` / `observe` hooks, builds each sampled state's
`contracts.EnvDecision` (validated), and checks:

* **X-C**: on every combat row, the combat teacher's slice of
  `Tokenizer.encode` is byte-equal to #1081's `Encoder.encode` (cat, cont,
  sec, n_act) and the PRIV_DRAW tokens' S_ctx part to its PRIV; every 25th
  row, the teacher's logits through both are `torch.equal`.
* **X-T**: on the first `--time` combat rows, interleaved, the minimum of
  `--reps` timings of `Encoder.encode` against `Tokenizer.encode` +
  `encode_priv`, and of #1081's observe (`honest_snapshot` + `IntentLog`)
  against `Tokenizer.observe`; plus the tokens per row over all rows. Every
  timed call sees the tokenizer's one-entry memos as the replay left them
  before this row (`tokens.memo_state`), so a repeat does not hit a memo the
  first call filled, and the next row sees them as this row's
  `encode` + `encode_priv` left them. The memos are kept per run (`tokens._memo`),
  so a worker's round-robin over its env slots hits them as this one-run replay
  does (GH #1097 TOK-1).
* The token-count distribution over `contracts.TOKEN_BUCKETS` (SCHEMA.md
  §2.11: WP3b measures it and may move bucket edges, never the cap): rows per
  bucket (the smallest edge that holds the row) and the padded tokens per
  real token.

Run it pinned to one performance core:

    taskset -c 7 env PYTHONPATH=python STS2_GAME_VERSION=v0.111.0 \\
        python -m ppo.e2e.tokcheck STATES.jsonl.gz.0,STATES.jsonl.gz.1

The teacher checkpoint defaults to the e2e inputs archive under the home
directory; its `RareMap` is the tokenizer's.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TEACHER = Path.home() / "projects/sts2/bench-archive/2026-11-02-e2e-inputs/run/it06000.pt"


def run_checks(paths: list[str], teacher: Path, n_time: int, reps: int) -> dict[str, Any]:
    import torch

    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.intent_log import IntentLog
    from ppo.combat_ctx.model import CombatCtxNet
    from ppo.combat_ctx.snapshot import honest_snapshot
    from ppo.combat_ppo import nets
    from ppo.combat_ppo.encode import Decision, Encoder

    from . import contracts as CT
    from . import recorder as R
    from . import tokens as K

    torch.set_num_threads(1)
    ck = torch.load(teacher, map_location="cpu", weights_only=False)
    if ck["vocab"] != T.vocab().digest():
        raise SystemExit("the teacher's S_ctx vocabulary differs from this build's")
    net = CombatCtxNet.from_config(dict(ck["config"], dropout=0.0))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    enc = Encoder("S_ctx", ck["rare"], CT.PRIV_DRAW_K)
    runs = [r for p in paths for r in R.read_runs(p)]
    n_states = n_xc = n_logits = 0
    bad: list[tuple[int, int, str]] = []
    times: dict[str, list[float]] = collections.defaultdict(list)
    ntok: list[int] = []
    combat: list[bool] = []
    by_type: collections.Counter = collections.Counter()
    for run in runs:
        tok = K.Tokenizer(rare=enc.rare)
        tr = K.SelectionTracker()
        for t, st in enumerate(R.replay(run, tokenizer=tok)):
            n_states += 1
            d = K.decision_from_state(st, tr)
            if d is None:
                continue
            memo = K.memo_state()
            ed = d.env_decision(run_uid=int(run["seed"]), t=t)
            CT.validate_decision(ed)
            got = K.Tok(
                is_combat=bool(ed.is_combat),
                dtype=int(ed.dtype),
                **tok.encode(ed),
                **tok.encode_priv(ed),
            )
            after = K.memo_state()
            by_type[CT.DECISION_TYPES[ed.dtype].name] += 1
            ntok.append(got.n_tok)
            combat.append(bool(ed.is_combat))
            if not ed.is_combat:
                continue
            legal = st.cs.legal_actions()
            dec = enc.encode(st.snap, st.bv, st.cs, legal)
            sl = got.teacher()
            pd = got.psec == CT.SEC["PRIV_DRAW"]
            same = (
                sl["cat"].tobytes() == dec.cat.tobytes()
                and sl["cont"].tobytes() == dec.cont.tobytes()
                and sl["sec"].tobytes() == dec.sec.tobytes()
                and sl["n_act"] == dec.n_act
                and got.pcat[pd][:, : CT.SCTX_N_CAT].tobytes() == dec.pcat.tobytes()
                and got.pcont[pd][:, : CT.SCTX_N_CONT].tobytes() == dec.pcont.tobytes()
            )
            if not same:
                bad.append((st.seed, st.step, "bytes"))
            n_xc += 1
            if n_xc % 25 == 0:
                ours = Decision(
                    cat=sl["cat"].copy(),
                    cont=sl["cont"].copy(),
                    sec=sl["sec"].copy(),
                    n_act=sl["n_act"],
                )
                with torch.no_grad():
                    a = nets.run(net, nets.collate([dec], "cpu", priv=False))[0]
                    b = nets.run(net, nets.collate([ours], "cpu", priv=False))[0]
                if not torch.equal(a, b):
                    bad.append((st.seed, st.step, "logits"))
                n_logits += 1
            if n_xc <= n_time:
                ilog = IntentLog()
                ilog.observe(st.snap)
                best = dict.fromkeys(
                    ("old", "new", "new_policy", "old_observe", "new_observe"), 1e9
                )
                for _ in range(reps):
                    t0 = time.perf_counter()
                    enc.encode(st.snap, st.bv, st.cs, legal)
                    t1 = time.perf_counter()
                    K.memo_restore(memo)
                    t2 = time.perf_counter()
                    tok._last = None
                    tok.encode(ed)
                    tok.encode_priv(ed)
                    t3 = time.perf_counter()
                    K.memo_restore(memo)
                    t4 = time.perf_counter()
                    tok._last = None
                    tok.encode(ed)
                    t5 = time.perf_counter()
                    ilog.observe(honest_snapshot(st.bv, st.room))
                    t6 = time.perf_counter()
                    tok.observe(ed.policy)
                    t7 = time.perf_counter()
                    for k, v in zip(
                        best, (t1 - t0, t3 - t2, t5 - t4, t6 - t5, t7 - t6), strict=True
                    ):
                        best[k] = min(best[k], v)
                for k, v in best.items():
                    times[k].append(v * 1e6)
                K.memo_restore(after)
        print(f"progress states {n_states} xc {n_xc} bad {len(bad)}", file=sys.stderr, flush=True)
    n, is_c = np.asarray(ntok), np.asarray(combat)
    edges = np.asarray(CT.TOKEN_BUCKETS)
    bucket = edges[np.minimum(np.searchsorted(edges, n), len(edges) - 1)]
    out: dict[str, Any] = {
        "states": n_states,
        "rows": len(ntok),
        "by_type": dict(by_type),
        "xc_rows": n_xc,
        "logit_checks": n_logits,
        "mismatches": bad[:20],
        "n_mismatches": len(bad),
        "tokens_mean": float(np.mean(ntok)),
        "tokens_p99": float(np.percentile(ntok, 99)),
        "tokens_max": int(np.max(ntok)),
        "tokens_mean_combat": float(n[is_c].mean()) if is_c.any() else 0.0,
        "tokens_mean_ooc": float(n[~is_c].mean()) if (~is_c).any() else 0.0,
        "tokens_over_cap": int((n > edges[-1]).sum()),
        "rows_per_bucket": {int(e): int((bucket == e).sum()) for e in edges},
        "padded_per_token": float(bucket.sum() / n.sum()),
        "n_timed": len(times["old"]),
    }
    if times["old"]:
        m = {k: float(np.mean(v)) for k, v in times.items()}
        old, new = np.asarray(times["old"]), np.asarray(times["new"])
        out |= {
            "us_mean": m,
            "ratio_mean": m["new"] / m["old"],
            "ratio_median": float(np.median(new / old)),
            "ratio_policy_only": m["new_policy"] / m["old"],
            "ratio_with_observe": (m["new"] + m["new_observe"]) / (m["old"] + m["old_observe"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("states", help="recorded runs: comma-separated gzip JSONL files")
    ap.add_argument("--teacher", default=str(TEACHER))
    ap.add_argument("--time", type=int, default=2000, help="combat rows to time")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    out = run_checks(a.states.split(","), Path(a.teacher), a.time, a.reps)
    print(json.dumps(out, indent=1))
    return 1 if out["n_mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
