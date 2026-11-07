"""F7: the student's policy logits do not move when the hidden information does.

Plan section 3.4 (F7 blocks G1). On real v0.111.0 A10 states, permute what the
player cannot know and keep what it can (the move state: telegraph and the
intent forecast), and require the student's policy logits to be `torch.equal`
row by row on every serving path this tree has:

* ``s0``: exactly as the learner serves at launch (``Learner._serve``:
  ``net_inf(to_batch(rb, dev))`` with ``net_inf.bf16 = cfg.bf16 and cuda``, so
  bf16 autocast on CUDA, fp32 on CPU); logits and log-probs;
* ``cpu_fp32``: the net in fp32 on CPU, bucketed (eval's ``StudentPolicy``
  path: ``net.policy``) and unbucketed (the update's ``NO_BUCKETS`` forward).

* ``packed`` (with ``--packed``): the learner's ``--packed-serve`` path,
  ``net.forward_packed(to_packed(rb, dev))`` on ``--device`` (bf16 varlen
  attention on CUDA, fp32 on CPU); logits and log-probs.

On a CPU run the first two coincide and only ``cpu_fp32`` runs. S1b (WP5b's student
engine) does not exist in this tree; F7 on S1b runs when it hot-swaps in.

The permutations ("kinds"), each a (base, permuted) row pair:

* ``world1``, ``world2``: ``cs.sample_world`` re-deals the unknown draw order
  and reseeds every combat RNG stream (combat rows);
* ``rooms``: other room queues and other later bosses; any read of a future
  map raises (``HiddenRooms``);
* ``run_rng``: the run RNG reseeded on a clone (``rm.reseed_run``);
* ``no_priv``: PRIV not encoded at all, every PRIV read raising (T14).

The check also refuses a vacuous pass: the re-deal must move the true draw or
PRIV on most combat rows, the room permutation must move PRIV_RUN, and an
honest change (every continuous cell x1.25) must move the logits.

    PYTHONPATH=$PIN:python python -m ppo.e2e.f7check --device cuda:0 --runs 12 \\
        --out f7.json                       # the launch net, S0 on CUDA bf16
    ... --ckpt LANE/ckpt/pinned/<it>.pt     # a trained student (hot-swap, 1B pin)
    ... --shards STATES.jsonl.gz            # replay recorded runs instead of playing

The runs are played by the recorder's agent (``recorder.record_run``: the
Ironclad champion at the recorder's budget, 300 sims and 1,200 at bosses, so
most runs reach act 3; a run takes ~25 s). Exit 0 on a pass, 1 on any mismatch
or a vacuous pass, 2 on a usage error, 3 on a crash (the verdict then has
``ok`` false and the ``error``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The launch fill (RUNBOOK "Launch"): WP-ENG-1 getters and the committed content prior.
LAUNCH_FILLED = ("WP-ENG-1", "content-prior")
KINDS = ("world1", "world2", "rooms", "run_rng", "no_priv")
WORLD_SEEDS = {"world1": 9001, "world2": 9002}
#: The share of combat rows on which the re-deal must move the true draw or PRIV.
MIN_WORLD_MOVED = 0.5
SEED_START = 162_000_200  # the recorder's default range (no eval panel)
RESEED_BASE = 600_000_000
MAX_EXAMPLES = 20


class UsageError(Exception):
    """Bad arguments: exit 2."""


# ── the hidden-information permutations ─────────────────────────────────


class HiddenRead(AssertionError):
    """A tokenizer read of what the check hides (a PRIV call, a future map): counted
    as a problem, so the check still writes its verdict."""


class NoPriv:
    """A run whose PRIV reads raise: the policy part must never make one."""

    def __init__(self, rm: Any) -> None:
        self._rm = rm

    def __getattr__(self, name: str) -> Any:
        from . import contracts as C

        if f"rm.{name}" in C.PRIV_CALLS:
            raise HiddenRead(f"the policy tokenizer read rm.{name}")
        return getattr(self._rm, name)


class HiddenRooms:
    """A run whose hidden future differs: other room queues, other later bosses.
    A read of a future map raises."""

    def __init__(self, rm: Any) -> None:
        self._rm = rm

    def __getattr__(self, name: str) -> Any:
        if name in ("act_map_for", "act_map", "future_maps"):
            raise HiddenRead(f"the tokenizer read {name} (a future map)")
        return getattr(self._rm, name)

    def generated_rooms(self) -> list[dict[str, Any]]:
        gr = [dict(g) for g in self._rm.generated_rooms()]
        a = int(self._rm.act_index) - 1
        for i, g in enumerate(gr):
            for key in ("normal", "elite", "events_shuffled"):
                g[key] = list(reversed(g.get(key) or []))[:-1]
            if i > a:
                g["boss"] = (
                    "CEREMONIAL_BEAST_BOSS"
                    if g.get("boss") != "CEREMONIAL_BEAST_BOSS"
                    else "WATERFALL_GIANT_BOSS"
                )
        return gr


def same_arrays(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> bool:
    return len(a) == len(b) and all(
        x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        for x, y in zip(a, b, strict=True)
    )


def _sec(t: Any, name: str) -> bytes:
    from . import contracts as C

    m = t.psec == C.SEC[name]
    return t.pcat[m].tobytes() + t.pcont[m].tobytes()


def _priv(t: Any) -> tuple[bytes, ...]:
    return tuple(
        x.tobytes() for x in (t.pcat, t.pcont, t.psec, t.ppool_tok, t.ppool_vid, t.ppool_w)
    )


def _enc(d: Any, **kw: Any) -> Any:
    from . import tokens as K

    K.clear_cache()
    return K.encode(d, **kw)


def world_decision(st: Any, d: Any, seed: int) -> tuple[Any, Any]:
    """`(world, decision)`: `d` in the world `st.cs.sample_world(seed)` deals."""
    import sts2_rs

    from ppo.combat_ctx.snapshot import honest_snapshot

    from . import env as E

    w = st.cs.sample_world(seed)
    bvw = w.belief_view(0)
    snapw = honest_snapshot(bvw, st.room)
    snapw["ehist"] = st.snap["ehist"]
    cc = E.merged_classes(w, room=st.room)
    dw = dataclasses.replace(
        d,
        cs=w,
        bv=bvw,
        d_bv=sts2_rs.state_to_dict(bvw),
        snap=snapw,
        kinds=cc.kinds,
        obs_actions=cc.obs_actions,
        classes=cc.classes,
        members=cc.members,
    )
    return w, dw


def decisions(runs: Sequence[dict[str, Any]]) -> Iterator[tuple[Any, Any]]:
    """`(state, decision)` of every sampled multi-option state of `runs`, each state's
    run cloned so it outlives the replay's next step."""
    from . import recorder as R
    from . import tokens as K

    for run in runs:
        tr = K.SelectionTracker()
        for st in R.replay(run):
            st.rm = st.rm.clone()
            if st.cs is not None:
                st.cs = st.rm.combat_state()
                st.bv = st.cs.belief_view(0)
            d = K.decision_from_state(st, tr)
            if d is not None:
                yield st, d


@dataclasses.dataclass
class Pairs:
    """Per kind, the (base, permuted) `Tok` pairs and where each came from."""

    pairs: dict[str, list[tuple[Any, Any]]]
    where: dict[str, list[tuple[int, int, str]]]  # (seed, step, decision type)
    counts: Counter
    problems: list[str]
    examples: dict[str, list[Any]]


def make_pairs(runs: Sequence[dict[str, Any]], *, max_rows: int | None = None) -> Pairs:
    """Every kind's pairs over `runs`, with the row-level checks (policy arrays
    equal, move state fixed) and the non-vacuity counts."""
    from . import contracts as C

    out = Pairs({k: [] for k in KINDS}, {k: [] for k in KINDS}, Counter(), [], {})

    def note(key: str, item: Any) -> None:
        ex = out.examples.setdefault(key, [])
        if len(ex) < MAX_EXAMPLES:
            ex.append(item)

    def add(kind: str, base: Any, other: Any, at: tuple[int, int, str]) -> None:
        out.counts[f"{kind}/rows"] += 1
        if not same_arrays(base.policy_arrays(), other.policy_arrays()):
            out.counts[f"{kind}/row_mismatch"] += 1
            note(f"{kind}/row_mismatch", list(at))
        out.pairs[kind].append((base, other))
        out.where[kind].append(at)

    for i, (st, d) in enumerate(decisions(runs)):
        if max_rows is not None and i >= max_rows:
            break
        at = (int(st.seed), int(st.step), C.DECISION_TYPES[d.dtype].name)
        out.counts["rows"] += 1
        out.counts["combat_rows"] += int(d.is_combat)
        base = _enc(d)
        try:
            pol = _enc(dataclasses.replace(d, rm=NoPriv(d.rm)), priv=False)
        except HiddenRead as e:  # the most direct leak: the policy part read PRIV
            out.counts["no_priv/priv_read"] += 1
            note("no_priv/priv_read", [*at, str(e)])
        else:
            if pol.n_priv:
                out.problems.append(
                    f"no_priv: a row encoded {pol.n_priv} PRIV tokens with priv=False"
                )
            add("no_priv", base, pol, at)
        try:
            hid = _enc(dataclasses.replace(d, rm=HiddenRooms(d.rm)))
        except HiddenRead as e:
            out.counts["rooms/future_map_read"] += 1
            note("rooms/future_map_read", [*at, str(e)])
        else:
            out.counts["rooms/priv_run_moved"] += _sec(base, "PRIV_RUN") != _sec(hid, "PRIV_RUN")
            add("rooms", base, hid, at)
        c = d.rm.clone()
        c.reseed_run(RESEED_BASE + i)
        rs = _enc(dataclasses.replace(d, rm=c))
        out.counts["run_rng/priv_moved"] += _priv(base) != _priv(rs)
        add("run_rng", base, rs, at)
        if not d.is_combat:
            continue
        tele, fc = st.cs.telegraph(), st.cs.peek_intents(3)
        top = list(st.cs.true_draw_top(10))
        for kind, seed in WORLD_SEEDS.items():
            w, dw = world_decision(st, d, seed)
            if w.telegraph() != tele or w.peek_intents(3) != fc:
                out.counts[f"{kind}/move_state_moved"] += 1
                note(f"{kind}/move_state_moved", list(at))
            tw = _enc(dw)
            moved_draw = list(w.true_draw_top(10)) != top
            moved_priv = _priv(base) != _priv(tw)
            out.counts[f"{kind}/true_draw_moved"] += moved_draw
            out.counts[f"{kind}/priv_moved"] += moved_priv
            out.counts[f"{kind}/moved"] += moved_draw or moved_priv
            add(kind, base, tw, at)
    _vacuity(out)
    return out


def _vacuity(p: Pairs) -> None:
    """Problems for a permutation that moved nothing it should have."""
    n, nc = p.counts["rows"], p.counts["combat_rows"]
    if n == 0:
        p.problems.append("no decision rows: the runs sampled no state")
        return
    if nc == 0:
        p.problems.append("no combat rows: the re-deal was never exercised")
    for kind in WORLD_SEEDS:
        moved = p.counts[f"{kind}/moved"]
        if nc and moved < MIN_WORLD_MOVED * nc:
            p.problems.append(
                f"{kind}: the re-deal moved the draw or PRIV on {moved} of {nc} combat rows"
                f" (< {MIN_WORLD_MOVED:.0%}): a no-op re-deal?"
            )
        if p.counts[f"{kind}/move_state_moved"]:
            p.problems.append(
                f"{kind}: sample_world moved the move state on"
                f" {p.counts[f'{kind}/move_state_moved']} rows (it must stay fixed)"
            )
    if p.counts["rooms/rows"] and p.counts["rooms/priv_run_moved"] == 0:
        p.problems.append("rooms: the room permutation never moved PRIV_RUN")
    if p.counts["no_priv/priv_read"]:
        p.problems.append(
            f"no_priv: {p.counts['no_priv/priv_read']} rows' policy encoding read PRIV"
        )
    if p.counts["rooms/future_map_read"]:
        p.problems.append(f"rooms: {p.counts['rooms/future_map_read']} rows read a future map")


# ── the net ──────────────────────────────────────────────────────────────


def launch_net(net_json: dict[str, Any] | None = None, seed: int = 0) -> Any:
    """A fresh student as the learner builds it at launch (``Learner.__init__`` and
    ``_fill_prior``): ``NetConfig.from_dict(net_json or {})`` (the default 6 x 256),
    ``filled`` = the launch fill, the committed content prior live."""
    from . import content_prior as CP
    from . import net as N

    net = N.E2ENet(N.NetConfig.from_dict(net_json or {}), seed=seed, filled=list(LAUNCH_FILLED))
    net.set_prior(CP.table(), CP.sha256())
    return net.eval()


def load_net(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """The student of `path`: a learner state (state.pt, a full, pinned or slim
    state: the net under ``"net"``) or a student checkpoint (``net_state``'s
    format). Refuses another schema or vocab (``net_from_state``)."""
    import torch

    from . import net as N

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and ck.get("format") != N.FORMAT and isinstance(ck.get("net"), dict):
        ck = ck["net"]
    if not isinstance(ck, dict) or ck.get("format") != N.FORMAT:
        raise UsageError(f"{path}: neither a learner state nor an {N.FORMAT} checkpoint")
    net = N.net_from_state(ck)
    return net.eval(), dict(ck["cfg"])


def describe_net(net: Any, ckpt: str | None, ck_cfg: dict[str, Any] | None) -> dict[str, Any]:
    from . import net as N

    out: dict[str, Any] = {
        "cfg": net.cfg.to_dict(),
        "params": N.n_params(net),
        "prior_sha256": net.prior_sha256,
        "prior_live": bool(net.layout.prior_live),
    }
    if ckpt is None:
        out["source"] = "fresh launch net"
        out["filled"] = list(net.layout.filled)
    else:
        out["source"] = str(Path(ckpt).resolve())
        out["ckpt_sha256"] = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
        out["filled"] = list((ck_cfg or {}).get("stamps", {}).get("filled") or [])
    return out


# ── the comparisons ──────────────────────────────────────────────────────


def _row_equal(x: Any, y: Any, i: int) -> bool:
    import torch

    return x.shape[1:] == y.shape[1:] and torch.equal(x[i], y[i])


def _batches(pairs: list[tuple[Any, Any]], batch: int) -> Iterator[tuple[list[int], Any, Any]]:
    """Row batches of mixed lengths (sorted by length, as a serving round mixes
    rows), base and permuted in the same order."""
    from . import tokens as K

    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0].n_tok)
    for lo in range(0, len(order), batch):
        idx = order[lo : lo + batch]
        yield (
            idx,
            K.row_batch([pairs[i][0] for i in idx]),
            K.row_batch([pairs[i][1] for i in idx]),
        )


def compare(
    net: Any,
    p: Pairs,
    *,
    path: str,
    device: str,
    bf16: bool,
    batch: int,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """One serving path over every kind: `{kind: {"rows", "mismatches", ...}}`.

    ``s0`` compares ``net(to_batch(rb, device))``'s logits and log-probs with
    ``net.bf16 = bf16``, as the learner serves. ``cpu_fp32`` compares
    ``net.policy`` bucketed and ``net(...)``'s logits unbucketed (``NO_BUCKETS``)."""
    import torch

    from . import net as N
    from .update import NO_BUCKETS

    net = net.to(device).eval()
    net.bf16 = bf16
    res: dict[str, Any] = {}
    with torch.no_grad():
        for kind in KINDS:
            pairs = p.pairs[kind]
            r = {"rows": 0, "mismatches": 0, "examples": []}
            for idx, a, b in _batches(pairs, batch):
                n = len(idx)
                ba, bb = N.to_batch(a, device), N.to_batch(b, device)
                checks: list[tuple[Any, Any]] = []
                if path == "s0":
                    fa, fb = net(ba), net(bb)
                    checks += [(fa.logits, fb.logits), (fa.logp, fb.logp)]
                elif path == "packed":
                    fa = net.forward_packed(N.to_packed(a, device))
                    fb = net.forward_packed(N.to_packed(b, device))
                    checks += [(fa.logits, fb.logits), (fa.logp, fb.logp)]
                else:
                    checks.append((net.policy(ba, bf16=False), net.policy(bb, bf16=False)))
                    ua = N.to_batch(a, device, **NO_BUCKETS)
                    ub = N.to_batch(b, device, **NO_BUCKETS)
                    checks.append((net(ua, bf16=False).logits, net(ub, bf16=False).logits))
                for j in range(n):
                    if not all(_row_equal(x, y, j) for x, y in checks):
                        r["mismatches"] += 1
                        if len(r["examples"]) < MAX_EXAMPLES:
                            r["examples"].append(list(p.where[kind][idx[j]]))
                r["rows"] += n
            res[kind] = r
            log(f"{path} {kind}: {r['rows']} rows, {r['mismatches']} mismatches")
    return res


def control(net: Any, p: Pairs, *, device: str, bf16: bool, batch: int) -> dict[str, int]:
    """The positive control: an honest change (every continuous cell x1.25) must move
    the logits of most rows, or bit-identity means a net that reads nothing."""
    import torch

    from . import net as N

    pairs = p.pairs["no_priv"]
    if not pairs:
        return {"rows": 0, "moved": 0}
    _idx, a, _b = next(_batches(pairs, batch))
    a2 = dict(a)
    a2["cont"] = (a["cont"].astype(np.float32) * 1.25).astype(a["cont"].dtype)
    net = net.to(device).eval()
    net.bf16 = bf16
    with torch.no_grad():
        x, y = net(N.to_batch(a, device)).logits, net(N.to_batch(a2, device)).logits
    n = len(a["n_tok"])
    return {"rows": n, "moved": sum(not _row_equal(x, y, i) for i in range(n))}


# ── the command ──────────────────────────────────────────────────────────


def collect_runs(args: argparse.Namespace, log: Callable[[str], None]) -> list[dict[str, Any]]:
    from . import recorder as R

    if args.shards:
        runs = [r for s in args.shards for r in R.read_runs(s)][: args.runs]
        log(f"replaying {len(runs)} recorded run(s)")
        return runs
    runs = []
    for s in range(args.seed_start, args.seed_start + args.runs):
        t = time.time()
        run = R.record_run(s, args.p, args.sims, args.boss_sims)
        log(
            f"played seed {s}: {len(run['states'])} states, act {run['act']}"
            f" floor {run['floor']}, {time.time() - t:.0f} s"
        )
        runs.append(run)
    return runs


def run_check(args: argparse.Namespace, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The verdict (`main` writes it): ok, counts, mismatches per path and kind."""
    import torch

    from . import contracts as C
    from . import stamps as ST
    from . import vocab as V

    t0 = time.time()
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise UsageError(f"--device {args.device}: CUDA is not available")
    if args.ckpt:
        net, ck_cfg = load_net(args.ckpt)
    else:
        net, ck_cfg = launch_net(args.net), None
    desc = describe_net(net, args.ckpt, ck_cfg)
    runs = collect_runs(args, log)
    p = make_pairs(runs, max_rows=args.max_rows)
    log(f"{p.counts['rows']} rows ({p.counts['combat_rows']} combat) from {len(runs)} run(s)")
    s0_bf16 = dev.type == "cuda" and not args.fp32
    paths: dict[str, Any] = {}
    if dev.type == "cuda":
        paths["s0"] = compare(
            net, p, path="s0", device=str(dev), bf16=s0_bf16, batch=args.batch, log=log
        )
    paths["cpu_fp32"] = compare(
        net, p, path="cpu_fp32", device="cpu", bf16=False, batch=args.batch, log=log
    )
    if args.packed:
        paths["packed"] = compare(
            net, p, path="packed", device=str(dev), bf16=s0_bf16, batch=args.batch, log=log
        )
    ctl = control(net, p, device=str(dev), bf16=s0_bf16, batch=args.batch)
    problems = list(p.problems)
    if ctl["rows"] and ctl["moved"] < 0.9 * ctl["rows"]:
        problems.append(
            f"control: an honest change moved the logits of {ctl['moved']}/{ctl['rows']}"
        )
    row_mismatches = sum(p.counts[f"{k}/row_mismatch"] for k in KINDS)
    net_mismatches = sum(r["mismatches"] for path in paths.values() for r in path.values())
    mismatches = row_mismatches + net_mismatches
    return {
        "ok": mismatches == 0 and not problems,
        "check": "F7 (plan section 3.4)",
        "runs": len(runs),
        "seeds": [int(r["seed"]) for r in runs],
        "rows": p.counts["rows"],
        "combat_rows": p.counts["combat_rows"],
        "mismatches": mismatches,
        "row_mismatches": row_mismatches,
        "logit_mismatches": net_mismatches,
        "problems": problems,
        "device": str(dev),
        "dtype": "bf16" if s0_bf16 else "fp32",
        "paths": paths,
        "counts": dict(sorted(p.counts.items())),
        "examples": p.examples,
        "control": ctl,
        "net": desc,
        "stamps": ST.stamps(vocab_sha256=V.load().sha256, schema_sha256=C.schema_sha256()),
        "acts": {str(k): v for k, v in sorted(Counter(int(r["act"]) for r in runs).items())},
        "gpu_peak_mib": (
            round(torch.cuda.max_memory_allocated(dev) / 2**20) if dev.type == "cuda" else None
        ),
        "torch": str(torch.__version__),
        "argv": list(args.argv),
        "secs": round(time.time() - t0, 1),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.e2e.f7check", description=__doc__.split("\n", 1)[0]
    )
    ap.add_argument("--out", required=True, help="the JSON verdict")
    ap.add_argument("--device", default="cpu", help="S0's device (cuda:0 serves bf16)")
    ap.add_argument("--fp32", action="store_true", help="S0 on CUDA without bf16 (not launch)")
    ap.add_argument(
        "--packed", action="store_true", help="also check the --packed-serve path on --device"
    )
    ap.add_argument("--net", type=json.loads, default=None, help="NetConfig overrides (JSON)")
    ap.add_argument("--ckpt", default=None, help="a trained student or learner state")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=SEED_START)
    ap.add_argument("--p", type=float, default=0.4, help="share of states sampled per run")
    # The recorder's budget: most runs reach act 3, so every act's rows are checked.
    ap.add_argument("--sims", type=int, default=300, help="the recording agent's MCTS sims")
    ap.add_argument("--boss-sims", type=int, default=1200)
    ap.add_argument("--shards", nargs="+", default=None, help="recorded runs to replay")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    args = ap.parse_args(list(argv))
    args.argv = list(argv)
    if args.ckpt and args.net is not None:
        raise UsageError("--net and --ckpt exclude each other (a checkpoint names its config)")
    if args.ckpt and not Path(args.ckpt).is_file():
        raise UsageError(f"--ckpt {args.ckpt}: no such file")
    if args.net is not None and not isinstance(args.net, dict):
        raise UsageError("--net takes a JSON object")
    if args.runs < 1 or args.batch < 1 or not 0 < args.p <= 1:
        raise UsageError("--runs and --batch must be positive, --p in (0, 1]")
    for s in args.shards or ():
        if not Path(s).is_file():
            raise UsageError(f"--shards {s}: no such file")
    return args


def crash_verdict(args: argparse.Namespace, e: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "check": "F7 (plan section 3.4)",
        "error": f"{type(e).__name__}: {e}",
        "traceback": traceback.format_exc()[-4000:],
        "device": str(args.device),
        "argv": list(args.argv),
    }


def _write(path: str, verdict: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    from .evaluate import EvalError, select_campaign_build

    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:  # argparse
        return 2 if e.code else 0
    except (UsageError, json.JSONDecodeError) as e:
        print(f"f7check: {e}", file=sys.stderr)
        return 2
    try:
        select_campaign_build()
        torch.set_num_threads(max(1, args.threads))
        verdict = run_check(args, log=lambda s: print(f"f7check: {s}", flush=True))
    except (UsageError, EvalError) as e:
        print(f"f7check: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # a crash still leaves a verdict to record
        traceback.print_exc()
        _write(args.out, crash_verdict(args, e))
        print(f"f7check: CRASH: {type(e).__name__}: {e}; verdict {args.out}", flush=True)
        return 3
    out = _write(args.out, verdict)
    word = "PASS" if verdict["ok"] else "FAIL"
    print(
        f"f7check: {word}: {verdict['runs']} runs, {verdict['rows']} rows,"
        f" {verdict['mismatches']} mismatches on {verdict['device']} {verdict['dtype']};"
        f" problems {verdict['problems']}; verdict {out}",
        flush=True,
    )
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
