"""The data-parallel update (plan §6, §8 L1) on WP4a's algorithm core.

Every rank holds an identical replica of the net, the AdamW optimizer and the
`ppo.e2e.algo.AlgoState` (clocks, KL factor, entropy-collapse counters,
advantage moments, hold-back queue). One update follows the protocol of
`ppo.e2e.algo`:

1. ``algo.hold.drop(errored segments)``, ``sched = algo.begin(n, t_wall)``,
   ``table = algo.join(fresh)``. The learner made the same join on its copy
   and sent every table row in that order, as one `contracts` row batch in
   shared memory (`rowstore`); the rank checks the order (the ``ref``
   column). Each forward pads its rows with WP4b's `net.to_batch`.
2. The masks (trained, student-acted, held, dormant) do not depend on the
   values, so `length_strata` cuts the trained rows first. The pre-pass then
   runs every stratum (owner: stratum index mod world) and every other
   non-dormant row once, with no gradient: V of every row, log π_prox of the
   taken class, the raw entropy per type of student-acted trained rows and
   the π_prox logits of the probe strata (`probe_strata`). One all-reduce
   assembles them.
3. ``ret = algo.returns(table, values)``, ``adv = algo.normalize(table, ret)``,
   ``set_lr(opt, sched.lr_frac)``.
4. E epochs of `epoch_minibatches` (24 strata per minibatch, `24 / world` per
   rank, redrawn every epoch). Each stratum is one micro-chunk, padded on its
   own: forward, `ppo_losses` with the minibatch's global trained-row count as
   the denominator, backward. Then one all-reduce of the flat gradient and
   one `clip_and_step`. After each epoch the exact probe KL; ``algo.kl.stop``
   skips the remaining epochs (U1 and U2 only).
5. ``algo.end(table, ret, probe_kl, entropy_per_type)``, then the EMAs (rank 0).
6. On the learner's jobs (``pack_held``) rank 0 copies the rows the hold-back
   now keeps out of the row block into a block of their own
   (`rowstore.take_to_shm`) and names it in the result (``held``: the block,
   its layout, the rows' table positions). The learner maps it, removes its
   name and points its held rows there, so they keep no other row alive and
   the learner's serving loop never copies them (WP4c's stall: 2-5 s per
   iteration). Not on a non-finite update: the learner saves nothing then.

Rank 0 answers ``("done", 0, result)`` with the new weights, the AlgoState
and the statistics. The other ranks answer only on an error. A non-finite
gradient skips its step (on every rank: the gradient is all-reduced first) and
the result names it (``nonfinite``); the learner then saves nothing and exits
(ops contract item 6). Rank 0 writes states on a ``("save", job)`` message,
through `ppo.e2e.ckpt` (`save`). `Rank` also runs in-process (`world == 1`).
"""

from __future__ import annotations

import dataclasses
import math
import os
import queue
import signal
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import algo as A
from . import ckpt, rowstore
from . import contracts as C
from . import schedules as S
from .net import (
    INPUT_ROOTS,
    E2ENet,
    NetConfig,
    check_prior,
    file_sha256,
    load_checkpoint,
    net_from_state,
    net_state,
    save_checkpoint,
    to_batch,
    to_packed,
)
from .rowstore import Chunk, axis_rows, batch_from_shm, take, take_to_shm

#: The update's micro-chunks pad each axis to their own maximum (no buckets: the
#: update is not compiled, and a length-sorted stratum pads by about 1%).
NO_BUCKETS = {
    "len_buckets": None,
    "opt_buckets": None,
    "priv_buckets": None,
    "pool_buckets": None,
    "ppool_buckets": None,
}
LEARNER_FORMATS = ("ppo.e2e.learner/3",)


def ahead(make: Callable[[Any], Any], items: Iterable[Any], depth: int = 2) -> Iterator[Any]:
    """`make(item)` for each of `items`, in order, built by one thread up to `depth`
    results ahead of the consumer (#1550): the update's steps wait on the device,
    and the host builds the next micro-batches meanwhile instead of in between.
    An error in `make` is raised where its result would have come. Closing the
    iterator (or dropping it) ends the thread."""
    out: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()
    done = object()

    def put(x: tuple[bool, Any]) -> bool:
        while not stop.is_set():
            try:
                out.put(x, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run() -> None:
        try:
            for it in items:
                if not put((True, make(it))):
                    return
            put((True, done))
        except BaseException as e:  # handed to the consumer
            put((False, e))

    t = threading.Thread(target=run, name="e2e-batch-ahead", daemon=True)
    t.start()
    try:
        while True:
            ok, x = out.get()
            if not ok:
                raise x
            if x is done:
                return
            yield x
    finally:
        stop.set()
        t.join()


def upload(x: np.ndarray, dev: torch.device) -> torch.Tensor:
    """`x` on `dev` without a host sync: a CUDA copy goes through pinned memory
    and does not block, so the host builds the next batch while the device runs
    (a pageable copy waits for every queued kernel). Same values either way."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dev.type != "cuda":
        return t.to(dev)
    return t.pin_memory().to(dev, non_blocking=True)


#: `ppo_losses` statistics (scalars, then per-type vectors), summed over chunks and ranks.
STAT_SCALARS = (
    "n_v",
    "vloss",
    "n_pi",
    "ploss",
    "ent",
    "ent_n",
    "ent_bonus",
    "kl",
    "clipfrac",
    "dualfrac",
    "isw",
    "is_clamped",
    "n_ks",
    "n_ks_missing",
    "ks_kl",
    "ks_ce",
    "ks_loss",
    "n_fl",
    "fl_loss",
    "fl_below",
)
STAT_VECTORS = ("n_pi_t", "ent_t", "kl_t", "n_v_t", "vloss_t", "n_ks_t", "ks_kl_t")
#: The update bins its per-type statistics by (character, type), `char * n_types +
#: type` with the character from each row's X_CHAR (`rowstore.row_chars`), and folds
#: the bins back to per-type sums: pooled per type, and split by character (GH #1197).
N_CHARS = len(C.CHARACTERS)
#: The value module's parameter-name prefix (`E2ENet.value`): the 3e-4 AdamW group.
VALUE_PREFIXES = ("value.",)


def loss_cfg(cfg: Any) -> A.LossCfg:
    return A.LossCfg(
        clip=float(cfg.clip),
        is_clamp=float(cfg.is_cap),
        dual_clip=float(cfg.dual_clip),
        vf_coef=float(cfg.vf_coef),
        vf_coef_run=None if float(cfg.vf_coef_run) < 0 else float(cfg.vf_coef_run),
    )


def fold_chars(v: Any, n_types: int) -> np.ndarray:
    """(character, type) bins summed over the characters: the per-type sums."""
    return np.asarray(v, np.float64).reshape(-1, n_types).sum(0)


def char_type_means(
    s: Any, n: Any, dtypes: tuple[str, ...], n_types: int | None = None
) -> dict[str, dict[str, float]]:
    """{character: {type: s / n}} of (character, type) bin sums `s` over counts `n`
    (`n_types` types per character, default `len(dtypes)`), for the bins with rows;
    characters without any are left out."""
    nt = int(n_types or len(dtypes))
    s2 = np.asarray(s, np.float64).reshape(-1, nt)
    n2 = np.asarray(n, np.float64).reshape(-1, nt)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in range(len(s2)):
            d = type_dict(np.where(n2[c] > 0, s2[c] / n2[c], np.nan), dtypes)
            if d:
                out[C.CHARACTERS[c]] = d
    return out


def distill_char_ids(names: Any) -> np.ndarray:
    """`ScheduleCfg.distill_chars` as indices into `contracts.CHARACTERS` (what
    `rowstore.row_chars` returns); ValueError on a name that is not a character."""
    bad = [c for c in names if c not in C.CHARACTERS]
    if bad or not len(names):
        raise ValueError(f"distill_chars {list(names)!r}: each must be one of {C.CHARACTERS}")
    return np.array([C.CHARACTERS.index(c) for c in names], np.int64)


def checkpoint_filled(path: str) -> tuple[str, ...]:
    """The fill events a checkpoint's net was trained on (its `stamps["filled"]`)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    return tuple(sorted((ck.get("cfg") or {}).get("stamps", {}).get("filled") or ()))


def check_distill_teacher_inputs(
    teacher_filled: Any, lane_filled: Any, source: str, where: str = "distill_ckpt"
) -> None:
    """Refuse a teacher fed inputs it was not trained on (GH #1197, the 791 -> 728 case).

    The "chars" fill event does not only add inputs: on an Ironclad row it rewrites
    existing ones (live damage previews, tactical columns), so a teacher trained without
    it must never read a "chars" lane's rows. `source` "update" feeds it the lane's own
    rows, so its "chars" must match the lane's; "acting" feeds it the workers' chars-off
    shadow rows, so it must be a teacher without "chars"."""
    t_chars = C.FILL_CHARS in set(teacher_filled or ())
    l_chars = C.FILL_CHARS in set(lane_filled or ())
    if source == "acting":
        if t_chars:
            raise ValueError(
                f'{where}: it was trained with "{C.FILL_CHARS}"; distill_source "acting" '
                'scores chars-off shadow rows: use distill_source "update"'
            )
        return
    if t_chars != l_chars:
        if l_chars:
            raise ValueError(
                f'{where}: it was trained without "{C.FILL_CHARS}" (filled '
                f"{sorted(teacher_filled or ())}) but this lane's rows carry it: use "
                'distill_source "acting" (it scores chars-off shadow rows)'
            )
        raise ValueError(
            f'{where}: it reads "{C.FILL_CHARS}", which this lane\'s rows do not carry'
        )


def check_distill_cfg(cfg: S.ScheduleCfg, lane_filled: Any = None) -> None:
    """Fail fast on a bad distillation config (the learner calls it before any worker
    starts): the characters exist, the checkpoint's file sha256 is the declared one and,
    with `lane_filled` (the lane's `--filled`), its inputs fit the distill source
    (`check_distill_teacher_inputs`)."""
    if cfg.distill_ckpt is None:
        return
    distill_char_ids(cfg.distill_chars)
    got = file_sha256(cfg.distill_ckpt)
    if got != cfg.distill_sha256:
        raise ValueError(
            f"distill_ckpt {cfg.distill_ckpt}: sha256 {got} != distill_sha256 {cfg.distill_sha256}"
        )
    if lane_filled is not None:
        check_distill_teacher_inputs(
            checkpoint_filled(cfg.distill_ckpt),
            lane_filled,
            cfg.distill_source,
            f"distill_ckpt {cfg.distill_ckpt}",
        )


def load_distill_rare(cfg: S.ScheduleCfg) -> Any:
    """The RareMap JSON the shadow applies (`ScheduleCfg.distill_rare`: a #1081 combat
    checkpoint's `ck["rare"]`), None without one. Refuses a map whose
    `teachers.rare_sha256` is not `distill_rare_sha256`."""
    if cfg.distill_rare is None:
        return None
    from .teachers import rare_sha256

    ck = torch.load(str(cfg.distill_rare), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "rare" not in ck:
        raise ValueError(f"distill_rare {cfg.distill_rare}: no RareMap (ck['rare'])")
    got = rare_sha256(ck["rare"])
    if got != cfg.distill_rare_sha256:
        raise ValueError(
            f"distill_rare {cfg.distill_rare}: rare_sha256 {got} != distill_rare_sha256"
            f" {cfg.distill_rare_sha256}"
        )
    return ck["rare"]


def load_distill_teacher(
    cfg: S.ScheduleCfg,
    device: torch.device | str,
    *,
    schema_sha256: str,
    vocab_sha256: str,
    student: E2ENet | None = None,
    lane_filled: Any = None,
) -> E2ENet:
    """The frozen distillation teacher of `cfg` (`ScheduleCfg.distill_ckpt`) on `device`,
    in eval mode with no parameter requiring a gradient. Refuses a file whose sha256 is
    not `distill_sha256`, a different schema or vocab, (with `student`) a different
    `NetConfig` or content prior, and (with `lane_filled`) inputs that do not fit the
    distill source (`check_distill_teacher_inputs`)."""
    check_distill_cfg(cfg, lane_filled)
    net, _ = load_checkpoint(
        str(cfg.distill_ckpt), schema_sha256=schema_sha256, vocab_sha256=vocab_sha256
    )
    if student is not None:
        if net.cfg.to_dict() != student.cfg.to_dict():
            raise ValueError(
                f"distill_ckpt {cfg.distill_ckpt}: net {net.cfg.to_dict()} != the student's "
                f"{student.cfg.to_dict()}"
            )
        if net.prior_sha256 != student.prior_sha256:
            raise ValueError(
                f"distill_ckpt {cfg.distill_ckpt}: content prior {net.prior_sha256} != the "
                f"student's {student.prior_sha256}"
            )
    net = net.to(device)
    net.eval()
    net.requires_grad_(False)
    return net


def distill_teacher_probs(
    t_logits: torch.Tensor, n_classes: torch.Tensor, sel: torch.Tensor, width: int
) -> torch.Tensor:
    """[B, width] fp32 teacher probabilities for `ppo_losses`' teacher term: row
    `sel[j]` gets the softmax of `t_logits[j]` (the teacher's logits of the selected
    rows, in order) over its `n_classes` merged classes, masked exactly like the
    student's log-softmax (`algo.class_mask`, `algo.NEG`); every other row is NaN
    (no teacher). `n_classes` is that of all B rows; `t_logits` may be narrower or
    wider than `width` (it pads its own sub-batch), as long as it covers the classes."""
    b = int(sel.shape[0])
    out = torch.full((b, width), math.nan, dtype=torch.float32, device=t_logits.device)
    if not int(t_logits.shape[0]):
        return out
    nc = n_classes[sel].long()
    lg = torch.full(
        (int(t_logits.shape[0]), width), A.NEG, dtype=torch.float32, device=t_logits.device
    )
    w = min(width, int(t_logits.shape[1]))
    lg[:, :w] = t_logits[:, :w].float()
    cm = A.class_mask(nc, width)
    out[sel] = torch.softmax(lg.masked_fill(~cm, A.NEG), dim=-1)
    return out


def type_dict(v: np.ndarray, dtypes: tuple[str, ...]) -> dict[str, float]:
    """A per-type vector as {type name: value} for the types that had rows."""
    out = {}
    for i, x in enumerate(np.asarray(v, np.float64)):
        if math.isfinite(x):
            out[dtypes[i] if i < len(dtypes) else str(i)] = float(x)
    return out


#: `_adv_diag`'s candidate filter thresholds, as fractions of the iteration's max |advantage|.
ADV_DIAG_FRACS = (0.001, 0.003, 0.01, 0.03, 0.1)


def _adv_diag(
    a: A.AlgoState,
    table: dict[str, np.ndarray],
    ret: A.Returns,
    values: np.ndarray,
    v_behav: np.ndarray,
    it: int,
    dtype: np.ndarray,
    dtypes: tuple[str, ...],
) -> dict[str, Any]:
    """`Cfg.adv_diag` (stats only, GH #1538 / #1550): what the advantage filter and the
    reused pre-pass would do to this iteration, without doing it.

    - ``below``: per candidate fraction of the max |advantage|, the share of trained
      rows under it, overall and per decision type; ``q``: quantiles of |advantage|.
    - ``reuse``: this iteration's rows take ``v_behav`` for their value, the returns
      are computed again, and the two sets of advantages are compared on the
      student rows (correlation, sign agreement, mean absolute difference), next
      to the value gap itself.
    """
    tr = ret.trained
    out: dict[str, Any] = {}
    if not tr.any():
        return out
    aa = np.abs(ret.adv[tr])
    top = float(aa.max())
    dt = dtype[tr]
    n_t = len(dtypes)
    cnt = np.maximum(np.bincount(dt, minlength=n_t), 1)
    out["a_max"] = top
    out["q"] = {f"p{q}": float(np.quantile(aa, q / 100)) for q in (10, 25, 50, 75, 90, 99)}
    out["below"] = {
        str(f): {
            "all": float((aa < f * top).mean()),
            "t": type_dict(np.bincount(dt[aa < f * top], minlength=n_t) / cnt, dtypes),
        }
        for f in ADV_DIAG_FRACS
    }
    fresh = np.asarray(table["iter"], np.int64) == it
    v_mix = np.where(fresh, v_behav.astype(np.float64), values)
    ret_b = a.returns(table, v_mix)
    m = ret.pi & fresh
    if int(m.sum()) > 2:
        x, y = ret.adv[m], ret_b.adv[m]
        dv = v_behav[m].astype(np.float64) - values[m]
        out["reuse"] = {
            "rows": int(m.sum()),
            "v_gap_mean": float(dv.mean()),
            "v_gap_abs": float(np.abs(dv).mean()),
            "v_gap_p99": float(np.quantile(np.abs(dv), 0.99)),
            "adv_corr": float(np.corrcoef(x, y)[0, 1]),
            "adv_sign": float((np.sign(x) == np.sign(y)).mean()),
            "adv_gap_abs": float(np.abs(x - y).mean()),
            "adv_abs": float(np.abs(x).mean()),
        }
    return out


class Rank:
    """One update rank. `world == 1` needs no torch.distributed."""

    def __init__(
        self, rank: int, world: int, device: str, cfg: SimpleNamespace, init: dict[str, Any]
    ) -> None:
        self.rank, self.world = rank, world
        self.dev = torch.device(device)
        self.cfg = cfg
        self.schema_sha256 = init["schema_sha256"]
        self.vocab_sha256 = init["vocab_sha256"]
        self.dtypes = tuple(init["dtypes"])
        self.net = net_from_state(
            init["net"], schema_sha256=self.schema_sha256, vocab_sha256=self.vocab_sha256
        ).to(self.dev)
        self.net.bf16 = bool(cfg.bf16) and self.dev.type == "cuda"
        self.net.train()
        self.algo = A.AlgoState.from_dict(init["algo"])
        self.opt = A.make_optimizer(
            self.net,
            VALUE_PREFIXES,
            cfg=self.algo.cfg,
            input_roots=INPUT_ROOTS,
            weight_decay=float(cfg.weight_decay),
        )
        if init.get("opt") is not None:
            # A state saved with or without the "input" group (lr_input) loads either way.
            A.load_opt_state(
                self.opt, init["opt"], self.net, VALUE_PREFIXES, input_roots=INPUT_ROOTS
            )
            # A resume restores the moments, not the saved hyperparameters: the cfg's
            # base lrs (what LrWatch checks) and weight decay hold (review finding LRN-4).
            base = self.algo.cfg.base_lrs()
            for g in self.opt.param_groups:
                g["base_lr"] = float(base[g["name"]])
                g["weight_decay"] = float(cfg.weight_decay)
        self.ema: A.Ema | None = None
        if rank == 0:
            # Buffers (the frozen prior) are copied into the shadows, not averaged.
            state = self.net.state_dict()
            buffers = {k for k, _ in self.net.named_buffers() if k in state}
            self.ema = A.Ema(state, tuple(cfg.ema_decays), copy=buffers)
            if init.get("ema") is not None:
                self.ema.load_state_dict(init["ema"])
                self.ema.copy_from(state)
        self.loss_cfg = loss_cfg(cfg)
        self._packed: Any = None  # `_fwd_packed`'s forward
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        # The distillation teacher (`ScheduleCfg.distill_ckpt`): a plain attribute, never
        # a submodule of `net`, so the optimizer, the EMA, the sync check and every save
        # leave it out. Loaded whenever a checkpoint is named with distill_source
        # "update" (its sha is checked even at lambda 0); it only runs at lambda > 0.
        # With "acting" the learner scored the rows when it served them (`tprob`): the
        # rank loads no teacher.
        self.distill: E2ENet | None = None
        self.distill_ids = np.zeros(0, np.int64)
        dc = self.algo.cfg
        if dc.distill_ckpt is not None:
            self.distill_ids = distill_char_ids(dc.distill_chars)
            if dc.distill_source == "update":
                self.distill = load_distill_teacher(
                    dc,
                    self.dev,
                    schema_sha256=self.schema_sha256,
                    vocab_sha256=self.vocab_sha256,
                    student=self.net,
                    lane_filled=getattr(cfg, "filled", None),
                )
                self.distill.bf16 = self.net.bf16

    # ── collectives ─────────────────────────────────────────────────────

    def _sum(self, x: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            import torch.distributed as dist

            dist.all_reduce(x)
        return x

    def _flat_grads(self) -> torch.Tensor:
        return torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in self.params
            ]
        )

    def _set_grads(self, flat: torch.Tensor) -> None:
        o = 0
        for p in self.params:
            k = p.numel()
            p.grad = flat[o : o + k].view_as(p).clone()
            o += k

    def _check_in_sync(self) -> None:
        """Every rank holds bit-identical weights (a broken all-reduce would drift them)."""
        if self.world == 1:
            return
        import torch.distributed as dist

        with torch.no_grad():
            s = torch.stack([p.double().sum() for p in self.params]).sum().reshape(1)
            got = [torch.zeros_like(s) for _ in range(self.world)]
            dist.all_gather(got, s)
        vals = [float(x) for x in got]
        if len(set(vals)) != 1:
            raise RuntimeError(f"update ranks out of sync: weight checksums {vals}")

    # ── one update ──────────────────────────────────────────────────────

    def _batch(self, rows: Chunk, idx: np.ndarray) -> Any:
        return to_batch(take(rows, idx), self.dev, pin=True, **NO_BUCKETS)

    def _fwd(self, rows: Chunk, idx: np.ndarray) -> Any:
        return self.net(self._batch(rows, idx))

    def _distill_logits(self, rows: Chunk, idx: np.ndarray) -> torch.Tensor:
        """The distillation teacher's policy logits of rows `idx` (no gradient, the
        student's autocast). The serving path: the teacher's value module never runs."""
        assert self.distill is not None
        with torch.no_grad():
            b = to_batch(take(rows, idx), self.dev, pin=True, **NO_BUCKETS)
            return self.distill.policy(b)

    def _fwd_packed(self, rows: Chunk, idx: np.ndarray) -> Any:
        """The pre-pass's forward of `idx` on the packed (pad-free) path, compiled
        once per rank on CUDA as the learner's serving is (`Cfg.compile_serve`)."""
        if self._packed is None:
            self._packed = self.net.forward_packed
            if self.dev.type == "cuda":
                self._packed = torch.compile(self.net.forward_packed, dynamic=True)
        return self._packed(self._packed_batch(rows, idx))

    def _packed_batch(self, rows: Chunk, idx: np.ndarray) -> Any:
        return to_packed(take(rows, idx), self.dev, pin=True)

    def update(self, job: dict[str, Any]) -> dict[str, Any]:
        a = self.algo
        t0 = time.time()
        it = int(job["it"])
        if a.clocks.it != it:
            raise RuntimeError(f"rank {self.rank}: AlgoState at iteration {a.clocks.it}, job {it}")
        dropped = a.hold.drop(job["drop_uids"]) if len(job["drop_uids"]) else 0
        sched = a.begin(int(job["n_collected"]), float(job["t_wall"]))
        table = a.join(job["fresh"])
        if not np.array_equal(table["ref"], job["refs"]):
            raise RuntimeError(
                f"rank {self.rank}: the joined table's order differs from the learner's"
            )
        shm = shared_memory.SharedMemory(name=job["shm"], track=False)
        try:
            rows = Chunk(batch_from_shm(shm, job["layout"]))
            if rows.n != len(table["ref"]):
                raise RuntimeError(f"shm holds {rows.n} rows, the table {len(table['ref'])}")
            res = self._update(sched, table, rows, it, t0, dropped)
            if job.get("pack_held") and self.rank == 0 and not res.get("nonfinite"):
                res["held"], res["stats"]["held_pack_s"] = self._pack_held(
                    table, rows, job.get("owner")
                )
            del rows
        finally:
            try:
                shm.close()
            except BufferError:  # a view outlived an error; the mapping goes with the process
                pass
        return res

    def _pack_held(
        self, table: dict[str, np.ndarray], rows: Chunk, owner: int | None = None
    ) -> tuple[dict[str, Any] | None, float]:
        """The rows the hold-back keeps (`HoldBack.carry`), copied out of the row
        block into a block of their own for the learner (step 6), and the copy's
        wall time. None when it keeps no row. The block's name carries `owner`
        (the learner's pid, `rowstore.shm_name`): it is the learner's to unlink."""
        t0 = time.time()
        hold = self.algo.hold
        if not len(hold):
            return None, time.time() - t0
        pos = np.flatnonzero(np.isin(table["ref"], hold.cols["ref"]))
        if len(pos) != len(hold):
            raise RuntimeError(f"{len(hold)} held rows, {len(pos)} of them in the table")
        shm, layout = take_to_shm(rows, pos, owner=owner)
        shm.close()  # the learner maps it by name and owns it from then on
        return {"shm": shm.name, "layout": layout, "pos": pos}, time.time() - t0

    def _update(
        self,
        sched: S.IterSchedule,
        table: dict[str, np.ndarray],
        rows: Chunk,
        it: int,
        t0: float,
        dropped: int,
    ) -> dict[str, Any]:
        cfg, a, dev = self.cfg, self.algo, self.dev
        n = len(table["ref"])
        world, rank = self.world, self.rank
        masks = a.returns(table, np.zeros(n))
        trained_idx = np.flatnonzero(masks.trained)
        n_tok = np.asarray(table["n_tok"], np.int64)
        per_mb = int(cfg.strata_per_mb)
        n_mb = max(1, round(len(trained_idx) / max(1, int(cfg.minibatch))))
        n_strata = per_mb * n_mb
        strata = [trained_idx[s] for s in A.length_strata(n_tok[trained_idx], n_strata)]
        probe = {int(j) for j in A.probe_strata(n_strata, seed=cfg.seed, it=it, k=cfg.probe_strata)}
        nc_all = np.asarray(table["n_classes"], np.int64)
        act_all = np.asarray(table["action"], np.int64)
        dt_all = np.asarray(table["dtype"], np.int64)
        comb_all = np.asarray(table["is_combat"], bool)
        n_types = a.n_types
        # GH #1197: the statistics' (character, type) bins (see N_CHARS).
        n_ct = N_CHARS * n_types
        ct_all = rowstore.row_chars(rows.rb) * n_types + dt_all

        # ── pre-pass ──
        tp = time.time()
        values = torch.zeros(n, dtype=torch.float32, device=dev)
        lp_prox = torch.zeros(n, dtype=torch.float32, device=dev)
        ent_s = torch.zeros(n_types, device=dev)
        ent_c = torch.zeros(n_types, device=dev)
        ent_n = torch.zeros(n_types, device=dev)  # normalised entropy (stats only)
        ent_n_ct = torch.zeros(n_ct, device=dev)  # the same by (character, type)
        probe_prox: dict[int, tuple[np.ndarray, torch.Tensor]] = {}
        other = np.flatnonzero(~masks.trained & ~masks.dormant)
        other = other[np.argsort(n_tok[other], kind="stable")]
        step = max(1, int(cfg.fwd_chunk))
        chunks = [other[i : i + step] for i in range(0, len(other), step)]
        # `packed_prepass` (#1550): the trained rows go through the packed forward in
        # length-sorted chunks of `fwd_chunk` rows, instead of one padded forward per
        # stratum (~340 rows): 20 s -> 11 s of the lane's update. Only the probe
        # strata keep their padded forward, whose logits `_probe_kl` compares with
        # the same forward after the epoch.
        packed = bool(getattr(cfg, "packed_prepass", False))
        fwd = self._fwd_packed if packed else self._fwd
        # `prepass_reuse` (GH #1550): this iteration's rows keep what the serving net
        # wrote, so only the carried rows and the teacher-acted ones get a forward.
        # The probe strata then get their own forward (as with `packed_prepass`),
        # which is also the entropy sample: a random 24 strata of the trained rows.
        reuse = bool(getattr(cfg, "prepass_reuse", False))
        v_behav = np.asarray(table["v_behav"], np.float32)
        lp_behav = np.asarray(table["logp_behav"], np.float32)
        served = np.zeros(n, bool)  # rows that take the serving net's value and log-prob
        if reuse:
            fresh_rows = np.asarray(table["iter"], np.int64) == it
            student = np.asarray(table["actor"], np.int64) == A.ACTOR_STUDENT
            served = fresh_rows & (student | ~masks.trained)
            other = other[~served[other]]
            chunks = [other[i : i + step] for i in range(0, len(other), step)]
        own_probe = packed or reuse  # the probe strata get a forward of their own
        if own_probe:
            tr = trained_idx[~served[trained_idx]]
            tr = tr[np.argsort(n_tok[tr], kind="stable")]
            pre = [tr[i : i + step] for i in range(0, len(tr), step)]
        else:
            pre = strata
        n_fwd = int(sum(len(x) for x in pre) + sum(len(x) for x in chunks))

        def absorb(idx: np.ndarray, out: Any, *, store: bool = True, ent: bool = True) -> None:
            nonlocal ent_s, ent_c, ent_n, ent_n_ct
            ti = upload(idx, dev)
            nc = upload(nc_all[idx], dev)
            if store:
                values[ti] = out.value.float()
                lp_prox[ti] = A.logp_of(out.logits, nc, upload(act_all[idx], dev))
            if not ent:
                return
            es, ec = A.entropy_by_type(
                out.logits,
                nc,
                upload(dt_all[idx], dev),
                upload(masks.pi[idx], dev),
                n_types,
            )
            ent_s += es
            ent_c += ec
            ent_n += normalized_entropy_by_type(out.logits, nc, dt_all[idx], masks.pi[idx], n_types)
            ent_n_ct += normalized_entropy_by_type(out.logits, nc, ct_all[idx], masks.pi[idx], n_ct)

        self.net.eval()
        with torch.no_grad():
            for j, idx in enumerate(pre):
                if j % world != rank or not len(idx):
                    continue
                out = fwd(rows, idx)
                absorb(idx, out, ent=not reuse)
                if not own_probe and j in probe:
                    probe_prox[j] = (idx, out.logits.float())
            if own_probe:
                for j in sorted(probe):
                    if j % world == rank and len(strata[j]):
                        out = self._fwd(rows, strata[j])
                        probe_prox[j] = (strata[j], out.logits.float())
                        if reuse:
                            absorb(strata[j], out, store=False)
            for c, idx in enumerate(chunks):
                if c % world == rank:
                    values[upload(idx, dev)] = fwd(rows, idx).value.float()
        self.net.train()
        flat = self._sum(torch.cat([values, lp_prox, ent_s, ent_c, ent_n, ent_n_ct]))
        values_np = flat[:n].double().cpu().numpy()
        lp_prox_np = flat[n : 2 * n].cpu().numpy()
        o = 2 * n
        ent_s_np = flat[o : o + n_types].double().cpu().numpy()
        ent_c_np = flat[o + n_types : o + 2 * n_types].double().cpu().numpy()
        ent_n_np = flat[o + 2 * n_types : o + 3 * n_types].double().cpu().numpy()
        ent_n_ct_np = flat[o + 3 * n_types :].double().cpu().numpy()
        if reuse:
            values_np[served] = v_behav[served]
            lp_prox_np[served] = lp_behav[served]
        # The pre-pass covers the strata, i.e. the trained rows: their counts per bin.
        # (With `prepass_reuse` the entropy sample is the probe strata's rows.)
        ent_rows = (
            np.concatenate([strata[j] for j in sorted(probe)] or [trained_idx[:0]])
            if reuse
            else trained_idx
        )
        pi_tr = ent_rows[masks.pi[ent_rows]]
        ent_c_ct_np = np.bincount(ct_all[pi_tr], minlength=n_ct).astype(np.float64)
        prepass_s = time.time() - tp
        nonfinite: list[str] = []
        live = ~masks.dormant
        if not np.isfinite(values_np[live]).all():
            nonfinite.append(f"values (it {it})")
        if not np.isfinite(lp_prox_np[masks.pi]).all():
            nonfinite.append(f"log pi_prox (it {it})")

        # ── returns, advantages, lr ──
        ret = a.returns(table, values_np)
        if not (np.array_equal(ret.trained, masks.trained) and np.array_equal(ret.pi, masks.pi)):
            raise RuntimeError("the loss masks moved with the values")
        adv = a.normalize(table, ret)
        # `adv_filter` (GH #1538), after the moments: the rows under the threshold lose
        # their policy terms ("pi") or leave the minibatches ("pi+v"). `ret` itself
        # stays whole for the hold-back, the probe and the statistics.
        flt = str(getattr(cfg, "adv_filter", "off"))
        if flt not in A.ADV_FILTER_MODES:
            raise ValueError(f"adv_filter {flt!r}: one of {A.ADV_FILTER_MODES}")
        pi_rows, train_strata = ret.pi, strata
        flt_stats: dict[str, Any] | None = None
        if flt != "off":
            keep, eta = a.filter_rows(ret, float(cfg.adv_filter_frac))
            n_tr = max(1, len(trained_idx))
            flt_stats = {
                "mode": flt,
                "eta": eta,
                "a_max": a.adv_max,
                "keep": float(keep.sum()) / n_tr,
                "rows_kept": int(keep.sum()),
                "keep_t": type_dict(
                    np.bincount(dt_all[keep], minlength=n_types)
                    / np.maximum(np.bincount(dt_all[ret.trained], minlength=n_types), 1),
                    self.dtypes,
                ),
            }
            if flt == "pi":
                pi_rows = ret.pi & keep
            else:
                kept = trained_idx[keep[trained_idx]]
                n_mb_k = max(1, round(len(kept) / max(1, int(cfg.minibatch))))
                train_strata = [kept[s] for s in A.length_strata(n_tok[kept], per_mb * n_mb_k)]
        diag = (
            _adv_diag(a, table, ret, values_np, v_behav, it, dt_all, self.dtypes)
            if (rank == 0 and bool(getattr(cfg, "adv_diag", False)))
            else None
        )
        applied = A.set_lr(self.opt, sched.lr_frac)
        has_t = np.asarray(table["has_t"], bool)
        beta = sched.beta_rows(dt_all, comb_all)
        # `adv_filter_beta_rescale`: the filter takes the entropy bonus away with the
        # rows it drops (the screen's arm C lost a quarter of its combat entropy). Scale
        # the kept rows' coefficient per decision type by policy rows / kept policy rows,
        # so each type's summed coefficient is what it is without the filter.
        if flt_stats is not None and bool(getattr(cfg, "adv_filter_beta_rescale", False)):
            n_pi_t = np.bincount(dt_all[ret.pi], minlength=n_types).astype(np.float64)
            n_kept_t = np.bincount(dt_all[ret.pi & keep], minlength=n_types).astype(np.float64)
            cap = max(1.0, float(getattr(cfg, "adv_filter_beta_cap", A.ADV_FILTER_BETA_CAP)))
            scale_t = np.minimum(n_pi_t / np.maximum(n_kept_t, 1.0), cap)
            scale_t[n_pi_t == 0] = 1.0
            beta = (beta * scale_t[dt_all]).astype(beta.dtype)
            flt_stats["beta_scale_t"] = type_dict(
                np.where(n_pi_t > 0, scale_t, np.nan), self.dtypes
            )
        ks = sched.ks_rows(dt_all, has_t)
        floor_p = sched.floor_rows(dt_all, nc_all)
        floor_w = None
        if floor_p.any():
            floor_w = np.full(n, sched.floor_coef, np.float32)
        else:
            floor_p = None
        target = ret.target
        use_teacher = bool(np.any(sched.ks_types > 0))
        # Learner-side distillation (ScheduleCfg.distill_*): the frozen teacher's probs
        # on the trained rows of the distill characters, through the same teacher term.
        dcfg = a.cfg
        # "acting": the probs are the rows' own `tprob` (scored at serving time on the
        # chars-off shadow rows); a distill row without them (its shadow was lost) gets
        # no term and is counted.
        acting = bool(dcfg.distill_acting)
        distill_live = bool(
            dcfg.distill_on and sched.policy_on and (acting or self.distill is not None)
        )
        if dcfg.distill_on and use_teacher:
            raise RuntimeError(
                "distillation (distill_lambda > 0) and the stored-row KS teacher (lambda_KS > 0)"
                " are both on: they share the teacher term; turn one off"
            )
        d_sel_all = np.zeros(n, bool)
        distill_missing = 0
        if distill_live:
            d_sel_all = ret.trained & np.isin(rowstore.row_chars(rows.rb), self.distill_ids)
            if acting:
                distill_missing = int((d_sel_all & ~has_t).sum())
                d_sel_all &= has_t
        d_lam = float(dcfg.distill_lambda)
        distill_s = 0.0
        distill_rows = 0

        def loss_rows(idx: np.ndarray, width: int) -> A.LossRows:
            nonlocal distill_s, distill_rows
            teacher = None
            ks_i = ks[idx]
            if use_teacher and has_t[idx].any():
                flat_t = axis_rows(rows, "tprob", idx)
                teacher = A.pad_ragged(flat_t, nc_all[idx], width=width)
            sel = d_sel_all[idx]
            d_teacher = None
            if distill_live and acting and sel.any():
                flat_t = axis_rows(rows, "tprob", idx)
                teacher = A.pad_ragged(flat_t, nc_all[idx], width=width)
                distill_rows += int(sel.sum())
                ks_i = np.where(sel, d_lam, 0.0)
            elif distill_live and sel.any():
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t1 = time.time()
                t_logits = self._distill_logits(rows, idx[sel])
                d_teacher = distill_teacher_probs(
                    t_logits, upload(nc_all[idx], dev), upload(sel, dev), width
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                distill_s += time.time() - t1
                distill_rows += int(sel.sum())
                ks_i = np.where(sel, d_lam, 0.0)
            lr_ = A.LossRows.from_numpy(
                dev,
                n_classes=nc_all[idx],
                action=act_all[idx],
                lp_prox=lp_prox_np[idx],
                lp_behav=lp_behav[idx],
                adv=adv[idx],
                target=target[idx],
                pi=pi_rows[idx],
                v=ret.trained[idx],
                beta=beta[idx],
                ks=ks_i,
                dtype=ct_all[idx],  # the statistics' bins; the loss itself reads no type
                teacher=teacher,
                floor_p=None if floor_p is None else floor_p[idx],
                floor_w=None if floor_w is None else floor_w[idx],
            )
            if d_teacher is not None:  # already on the device (from_numpy copies numpy)
                lr_ = dataclasses.replace(lr_, teacher=d_teacher)
            return lr_

        acc: dict[str, torch.Tensor] = {k: torch.zeros((), device=dev) for k in STAT_SCALARS}
        acc.update({k: torch.zeros(n_ct, device=dev) for k in STAT_VECTORS})
        gnorms: list[torch.Tensor] = []
        kls: list[float] = []
        steps = epochs_run = 0
        stopped = False
        tu = time.time()
        # This rank's forwards of the epoch, built ahead of the steps. With
        # `packed_update` a step's strata go through ONE packed (pad-free)
        # forward instead of one padded forward each (#1550): the lane's
        # micro-batches (~340 rows) are small enough that launching their
        # kernels costs more than computing them, and merged padded strata
        # would pad every row to the longest. The loss is a sum over rows, so
        # the step's gradient is the same up to float rounding.
        merged = bool(getattr(cfg, "packed_update", False))
        chunk = max(1, int(getattr(cfg, "update_chunk", 3072)))

        def groups(mb: list[list[np.ndarray]]) -> list[np.ndarray]:
            own = [idx for idx in mb[rank] if len(idx)]
            if not (merged and own):
                return own
            # At most `update_chunk` rows a forward: with fewer ranks a rank gets more
            # of the step, and one forward of it all does not fit a 24 GB device.
            rows_ = np.concatenate(own)
            return np.array_split(rows_, -(-len(rows_) // chunk))

        build = self._packed_batch if merged else self._batch
        for epoch in range(int(cfg.epochs)):
            if not trained_idx.size or nonfinite:
                break
            mbs = A.epoch_minibatches(
                train_strata, seed=cfg.seed, it=it, epoch=epoch, per_mb=per_mb, world=world
            )
            mine = [idx for mb in mbs for idx in groups(mb)]
            batches = ahead(lambda idx: build(rows, idx), mine)
            for mb in mbs:
                denom = float(sum(len(s) for rr in mb for s in rr))
                if denom == 0:
                    continue
                for idx in groups(mb):
                    batch = next(batches)
                    out = self.net.forward_packed(batch) if merged else self.net(batch)
                    lo = A.ppo_losses(
                        out.logits,
                        out.value,
                        loss_rows(idx, int(out.logits.shape[1])),
                        policy_on=sched.policy_on,
                        denom=denom,
                        cfg=self.loss_cfg,
                        n_types=n_ct,
                    )
                    lo.loss.backward()
                    for k, v in lo.stats.items():
                        acc[k] = acc[k] + v
                g = self._sum(self._flat_grads())
                if not bool(torch.isfinite(g).all()):
                    nonfinite.append(f"gradient (it {it}, epoch {epoch})")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self._set_grads(g)
                gnorms.append(A.clip_and_step(self.opt, float(cfg.grad_clip)).detach())
                self.opt.zero_grad(set_to_none=True)
                steps += 1
            batches.close()
            epochs_run += 1
            kl = self._probe_kl(rows, probe_prox, nc_all, masks.pi)
            kls.append(kl)
            if a.kl.stop(kl, sched, a.cfg):
                stopped = True
                break
        update_s = time.time() - tu
        ent_mean = np.where(ent_c_np > 0, ent_s_np / np.maximum(ent_c_np, 1.0), np.nan)
        info = a.end(table, ret, probe_kl=kls[-1] if kls else None, entropy_per_type=ent_mean)
        if self.ema is not None:
            self.ema.update(self.net.state_dict())
        self._check_in_sync()

        keys = list(acc)
        flat_s = self._sum(torch.cat([acc[k].reshape(-1).float() for k in keys]))
        if not bool(torch.isfinite(flat_s).all()):
            nonfinite.append("loss statistics")
        sums: dict[str, Any] = {}
        o = 0
        for k in keys:
            m = acc[k].numel()
            sums[k] = flat_s[o : o + m].reshape(acc[k].shape).cpu()
            o += m
        sums_ct = {k: sums[k].numpy() for k in STAT_VECTORS}
        for k in STAT_VECTORS:
            sums[k] = fold_chars(sums_ct[k], n_types)
        means = A.stats_means(sums)
        if rank != 0:
            return {"nonfinite": nonfinite}

        stats: dict[str, Any] = sched.log_fields(applied)
        stats.update(
            {
                "kl_m_next": info["kl_m"],
                "collapsed_types": info["collapsed_types"],
                "probe_kl": kls[-1] if kls else math.nan,
                "probe_kl_epochs": kls,
                "kl_stopped": stopped,
                "epochs_run": epochs_run,
                "steps": steps,
                "gradnorm_mean": float(torch.stack(gnorms).mean()) if gnorms else math.nan,
                "gradnorm_max": float(torch.stack(gnorms).max()) if gnorms else math.nan,
                "n_table": n,
                "n_strata": n_strata,
                "n_probe_strata": len(probe),
                "padding": A.padding_ratio(n_tok, strata),
                "held_dropped": dropped,
                "prepass_s": prepass_s,
                "prepass_rows_fwd": n_fwd,
                "update_s": update_s,
                "rank_s": time.time() - t0,
                "nonfinite": nonfinite,
            }
        )
        stats.update(ret.stats(table))
        if flt_stats is not None:
            stats["adv_filter"] = flt_stats
        if reuse:
            stats["prepass_rows_served"] = int(served.sum())
        if diag is not None:
            stats["adv_diag"] = diag
        for k, v in means.items():
            if isinstance(v, np.ndarray):
                if k not in ("n_pi_t", "n_v_t", "n_ks_t"):
                    stats[k] = type_dict(v, self.dtypes)
            else:
                stats[k] = v
        stats["ent_prox_t"] = type_dict(ent_mean, self.dtypes)
        ent_norm = np.where(ent_c_np > 0, ent_n_np / np.maximum(ent_c_np, 1.0), np.nan)
        stats["ent_norm_t"] = type_dict(ent_norm, self.dtypes)
        # GH #1197, by character: the same normalised entropy and k3 KL per type, and
        # the value loss over all types (the learner's `by_char`).
        dts = self.dtypes
        stats["ent_norm_ct"] = char_type_means(ent_n_ct_np, ent_c_ct_np, dts, n_types)
        stats["kl_ct"] = char_type_means(sums_ct["kl_t"], sums_ct["n_pi_t"], dts, n_types)
        vl_c = sums_ct["vloss_t"].reshape(N_CHARS, n_types).sum(1)
        nv_c = sums_ct["n_v_t"].reshape(N_CHARS, n_types).sum(1)
        stats["vloss_c"] = {
            C.CHARACTERS[c]: float(vl_c[c] / nv_c[c]) for c in range(N_CHARS) if nv_c[c] > 0
        }
        if dcfg.distill_on:
            # Learner-side distillation (its loss is the teacher term: also ks_kl/ks_loss).
            n_v_all = float(sums["n_v"])
            stats.update(
                {
                    "distill_lambda": d_lam,
                    "distill_source": dcfg.distill_source,
                    # Teacher rows over epochs: forwards ("update"), stored probs ("acting").
                    "distill_rows": distill_rows,
                    # "acting": distill rows the table trained without probs (shadow lost).
                    "distill_rows_missing": distill_missing,
                    "distill_fwd_s": distill_s,
                    "distill_kl": means.get("ks_kl", math.nan),
                    "distill_loss": means.get("ks_loss", math.nan),
                    # Its share of the minibatch-mean loss (sum over the trained rows).
                    "distill_loss_per_row": (
                        float(sums["ks_loss"]) / n_v_all if n_v_all > 0 else math.nan
                    ),
                    "distill_kl_t": type_dict(means.get("ks_kl_t", np.zeros(0)), dts),
                    "distill_kl_ct": char_type_means(
                        sums_ct["ks_kl_t"], sums_ct["n_ks_t"], dts, n_types
                    ),
                }
            )
        stats["is_tails"] = is_tails(lp_prox_np, lp_behav, ret.pi, float(cfg.is_cap))
        stats["loss_total"] = float(
            np.nansum([stats.get(k, math.nan) for k in ("vloss", "ploss", "ks_loss")])
            - (stats.get("ent_bonus") if math.isfinite(stats.get("ent_bonus", math.nan)) else 0.0)
        )
        tr = ret.trained
        for name, m in (("combat", tr & comb_all), ("run", tr & ~comb_all)):
            stats[f"ev_{name}"] = A.explained_variance(target, values_np, m)
            stats[f"v_mean_{name}"] = float(values_np[m].mean()) if m.any() else math.nan
            stats[f"target_mean_{name}"] = float(target[m].mean()) if m.any() else math.nan
        if tr.any():
            stale = it - np.asarray(table["version"], np.int64)[tr]
            stats["staleness_mean"] = float(stale.mean())
            stats["staleness_max"] = int(stale.max())
            age = it - np.asarray(table["iter"], np.int64)[tr]
            stats["row_age_mean"] = float(age.mean())
            stats["row_age_max"] = int(age.max())
            # This iteration's own rows: the serving lag of the async loop (at most 1 by
            # construction), apart from the held-back run rows' age (review LRN-3).
            fresh = age == 0
            if fresh.any():
                stats["staleness_fresh_mean"] = float(stale[fresh].mean())
                stats["staleness_fresh_max"] = int(stale[fresh].max())
        return {
            "net": {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()},
            "algo": a.to_dict(),
            "stats": stats,
            "nonfinite": nonfinite,
        }

    def _probe_kl(
        self,
        rows: Chunk,
        probe_prox: dict[int, tuple[np.ndarray, torch.Tensor]],
        nc_all: np.ndarray,
        pi: np.ndarray,
    ) -> float:
        s = torch.zeros(2, device=self.dev)
        self.net.eval()
        with torch.no_grad():
            for idx, lg_prox in probe_prox.values():
                ks, kc = A.probe_kl(
                    self._fwd(rows, idx).logits,
                    lg_prox,
                    upload(nc_all[idx], self.dev),
                    upload(pi[idx], self.dev),
                )
                s[0] += ks
                s[1] += kc
        self.net.train()
        s = self._sum(s)
        return float(s[0] / s[1]) if float(s[1]) > 0 else math.nan

    # ── checkpoints (rank 0) ────────────────────────────────────────────

    def full_state(self, job: dict[str, Any]) -> dict[str, Any]:
        """Everything a resume needs (the learner's `STATE_FORMAT`)."""
        stamps = dict(job["stamps"])
        return {
            "format": job["format"],
            "net": net_state(
                self.net,
                schema_sha256=self.schema_sha256,
                vocab_sha256=self.vocab_sha256,
                stamps=stamps,
            ),
            "opt": self.opt.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
            "algo": self.algo.to_dict(),
            "learner": job["learner_state"],
            "held": held_blob(job.get("held")),
            "stamps": stamps,
        }

    def slim_state(self, job: dict[str, Any]) -> dict[str, Any]:
        """The weights and every EMA (ops contract: a slim state)."""
        kw = {
            "schema_sha256": self.schema_sha256,
            "vocab_sha256": self.vocab_sha256,
            "stamps": dict(job["stamps"]),
        }
        return {
            "format": SLIM_FORMAT,
            "net": net_state(self.net, **kw),
            "ema": None if self.ema is None else self.ema.state_dict(),
            "learner": job["learner_state"],
            "stamps": dict(job["stamps"]),
        }

    def save(self, job: dict[str, Any]) -> list[str]:
        """Write what `job` asks for through `ppo.e2e.ckpt`: state.pt, a rolling full
        state, pinned states (one per reason) and a slim state."""
        sd = ckpt.SaveDir(job["save_dir"])
        d = int(job["learner_state"]["decisions"])
        it = int(job["learner_state"]["it"])
        written: list[str] = []
        if job.get("state") or job.get("full") or job.get("pins"):
            full = self.full_state(job)

            def writer(p: Any) -> None:
                torch.save(full, p)

            if job.get("state"):
                written.append(str(ckpt.save_state(sd, writer, decisions=d, it=it)))
            if job.get("full"):
                path = ckpt.save_full(sd, writer, decisions=d, keep=int(job["full_keep"]))
                written.append(str(path))
            for reason in job.get("pins") or []:
                written.append(str(ckpt.pin(sd, decisions=d, reason=reason, data=writer)))
        if job.get("slim"):
            slim = self.slim_state(job)
            written.append(str(ckpt.save_slim(sd, lambda p: torch.save(slim, p), decisions=d)))
        return written


SLIM_FORMAT = "ppo.e2e.slim/1"


def normalized_entropy_by_type(
    logits: torch.Tensor,
    n_classes: torch.Tensor,
    dtype: np.ndarray,
    mask: np.ndarray,
    n_types: int,
) -> torch.Tensor:
    """Entropy / log(n_classes) summed per decision type over ``mask`` rows (stats only)."""
    dev = logits.device
    m = upload(np.asarray(mask, bool), dev)
    nc = torch.where(m, n_classes.long(), torch.full_like(n_classes.long(), 2))
    cm = A.class_mask(nc, logits.shape[-1])
    lg = torch.where(m[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    lp = torch.log_softmax(lg.masked_fill(~cm, A.NEG), dim=-1)
    h = -(torch.where(cm, lp.exp() * lp, torch.zeros_like(lp))).sum(-1)
    hn = h / torch.log(nc.float().clamp(min=2.0))
    dt = upload(np.asarray(dtype, np.int64), dev)
    dt = torch.where(m, dt, torch.zeros_like(dt))
    return torch.zeros(n_types, device=dev).index_add(0, dt, hn * m.float())


def is_tails(
    lp_prox: np.ndarray, lp_behav: np.ndarray, pi: np.ndarray, cap: float
) -> dict[str, float]:
    """IS weight pi_prox / pi_behav over student-acted trained rows: p01, p99, max and the
    share above the clamp (ops contract `is_tails`)."""
    m = np.asarray(pi, bool)
    if not m.any():
        return {"p01": math.nan, "p99": math.nan, "max": math.nan, "frac_clamped": math.nan}
    w = np.exp(np.asarray(lp_prox, np.float64)[m] - np.asarray(lp_behav, np.float64)[m])
    return {
        "p01": float(np.quantile(w, 0.01)),
        "p99": float(np.quantile(w, 0.99)),
        "max": float(w.max()),
        "frac_clamped": float((w > cap).mean()),
    }


def held_blob(h: dict[str, Any] | None) -> dict[str, Any] | None:
    """The held rows (a `contracts` row batch: tokens, teacher probabilities, acting-time
    fields) from the learner's shm, as tensors a `weights_only` load reads back
    (`learner.held_from_state`)."""
    if not h:
        return None
    shm = shared_memory.SharedMemory(name=h["shm"], track=False)
    try:
        views = batch_from_shm(shm, h["layout"])
        rows = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in views.items()}
        del views
    finally:
        try:
            shm.close()
        except BufferError:
            pass
    return {"refs": torch.from_numpy(np.asarray(h["refs"], np.int64)), "rows": rows}


def variant_decay(variant: str) -> float | None:
    """ "raw" -> None; "ema0996" -> 0.996 (the ops contract's eval variant names)."""
    if variant == "raw":
        return None
    digits = variant.removeprefix("ema")
    if not digits.isdigit() or not digits.startswith("0"):
        raise ValueError(f"eval variant {variant!r}: raw or ema<digits> (ema0996)")
    return float("0." + digits[1:])


def export_variant(state_path: str, variant: str, out: str) -> str:
    """Write one variant (raw weights or an EMA) of a full or slim state as a WP4b net
    checkpoint (`net.save_checkpoint`, what an evaluator loads); returns its sha256."""
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    if st.get("format") not in (*LEARNER_FORMATS, SLIM_FORMAT):
        raise ValueError(f"{state_path}: not an e2e learner or slim state")
    cfg = st["net"]["cfg"]
    net = E2ENet(NetConfig.from_dict(cfg["net"]))
    decay = variant_decay(variant)
    if decay is None:
        net.load_state_dict(st["net"]["state_dict"])
    else:
        ema = st.get("ema")
        key = A.Ema.key(decay)
        if not ema or key not in ema["shadows"]:
            raise ValueError(f"{state_path}: no EMA {key}")
        net.load_state_dict({k: v.float() for k, v in ema["shadows"][key].items()})
    net.prior_sha256 = cfg.get("prior_sha256")
    check_prior(net, f"{state_path} {variant}")
    stamps = {
        **cfg.get("stamps", {}),
        "variant": variant,
        "from_state": os.path.basename(state_path),
    }
    return save_checkpoint(
        out,
        net,
        schema_sha256=cfg["schema_sha256"],
        vocab_sha256=cfg["vocab_sha256"],
        stamps=stamps,
    )


def rank_main(
    rank: int,
    world: int,
    device: str,
    store: str,
    cfg: SimpleNamespace,
    init_path: str,
    inq: Any,
    outq: Any,
) -> None:
    """A spawned update rank: ``("update", job)``, ``("save", job)`` or ``("stop",)``.

    It ignores SIGINT and SIGTERM: a signal sent to the whole process group (a
    Ctrl-C, `timeout`) reaches the learner, which finishes the update in flight,
    has rank 0 save, then stops the ranks. A rank whose learner died exits.
    """
    import faulthandler

    faulthandler.enable()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    parent = os.getppid()
    dist = None
    try:
        torch.set_num_threads(max(1, int(cfg.rank_threads)))
        if world > 1:
            import torch.distributed as dist

            backend = "nccl" if device.startswith("cuda") else "gloo"
            if device.startswith("cuda"):
                torch.cuda.set_device(torch.device(device))
            dist.init_process_group(
                backend, init_method=f"file://{store}", rank=rank, world_size=world
            )
        init = torch.load(init_path, map_location="cpu", weights_only=True)
        r = Rank(rank, world, device, cfg, init)
        del init
        outq.put(("ready", rank, None))
        while True:
            try:
                msg = inq.get(timeout=5.0)
            except queue.Empty:
                if os.getppid() != parent:  # the learner died without a stop
                    break
                continue
            if msg[0] == "stop":
                break
            kind, job = msg
            if kind == "update":
                res = r.update(job)
                if rank == 0:
                    outq.put(("done", rank, res))
            elif kind == "save" and rank == 0:
                outq.put(("saved", rank, r.save(job)))
    except BaseException:
        outq.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
