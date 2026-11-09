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

import math
import os
import queue
import signal
import time
import traceback
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import algo as A
from . import ckpt
from . import schedules as S
from .net import (
    E2ENet,
    NetConfig,
    check_prior,
    net_from_state,
    net_state,
    save_checkpoint,
    to_batch,
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
)
STAT_VECTORS = ("n_pi_t", "ent_t", "kl_t")
#: The value module's parameter-name prefix (`E2ENet.value`): the 3e-4 AdamW group.
VALUE_PREFIXES = ("value.",)


def loss_cfg(cfg: Any) -> A.LossCfg:
    return A.LossCfg(
        clip=float(cfg.clip),
        is_clamp=float(cfg.is_cap),
        dual_clip=float(cfg.dual_clip),
        vf_coef=float(cfg.vf_coef),
    )


def type_dict(v: np.ndarray, dtypes: tuple[str, ...]) -> dict[str, float]:
    """A per-type vector as {type name: value} for the types that had rows."""
    out = {}
    for i, x in enumerate(np.asarray(v, np.float64)):
        if math.isfinite(x):
            out[dtypes[i] if i < len(dtypes) else str(i)] = float(x)
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
            self.net, VALUE_PREFIXES, cfg=self.algo.cfg, weight_decay=float(cfg.weight_decay)
        )
        if init.get("opt") is not None:
            self.opt.load_state_dict(init["opt"])
            # A resume restores the moments, not the saved hyperparameters: the cfg's
            # base lrs (what LrWatch checks) and weight decay hold (review finding LRN-4).
            base = {"trunk": self.algo.cfg.lr_trunk, "value": self.algo.cfg.lr_value}
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
        self.params = [p for p in self.net.parameters() if p.requires_grad]

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

    def _fwd(self, rows: Chunk, idx: np.ndarray) -> Any:
        return self.net(to_batch(take(rows, idx), self.dev, **NO_BUCKETS))

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

        # ── pre-pass ──
        tp = time.time()
        values = torch.zeros(n, dtype=torch.float32, device=dev)
        lp_prox = torch.zeros(n, dtype=torch.float32, device=dev)
        ent_s = torch.zeros(n_types, device=dev)
        ent_c = torch.zeros(n_types, device=dev)
        ent_n = torch.zeros(n_types, device=dev)  # normalised entropy (stats only)
        probe_prox: dict[int, tuple[np.ndarray, torch.Tensor]] = {}
        other = np.flatnonzero(~masks.trained & ~masks.dormant)
        other = other[np.argsort(n_tok[other], kind="stable")]
        step = max(1, int(cfg.fwd_chunk))
        chunks = [other[i : i + step] for i in range(0, len(other), step)]
        self.net.eval()
        with torch.no_grad():
            for j, idx in enumerate(strata):
                if j % world != rank or not len(idx):
                    continue
                out = self._fwd(rows, idx)
                ti = torch.from_numpy(idx).to(dev)
                nc = torch.from_numpy(nc_all[idx]).to(dev)
                values[ti] = out.value.float()
                lp_prox[ti] = A.logp_of(out.logits, nc, torch.from_numpy(act_all[idx]).to(dev))
                es, ec = A.entropy_by_type(
                    out.logits,
                    nc,
                    torch.from_numpy(dt_all[idx]).to(dev),
                    torch.from_numpy(masks.pi[idx]).to(dev),
                    n_types,
                )
                ent_s += es
                ent_c += ec
                ent_n += normalized_entropy_by_type(
                    out.logits, nc, dt_all[idx], masks.pi[idx], n_types
                )
                if j in probe:
                    probe_prox[j] = (idx, out.logits.float())
            for c, idx in enumerate(chunks):
                if c % world == rank:
                    values[torch.from_numpy(idx).to(dev)] = self._fwd(rows, idx).value.float()
        self.net.train()
        flat = self._sum(torch.cat([values, lp_prox, ent_s, ent_c, ent_n]))
        values_np = flat[:n].double().cpu().numpy()
        lp_prox_np = flat[n : 2 * n].cpu().numpy()
        o = 2 * n
        ent_s_np = flat[o : o + n_types].double().cpu().numpy()
        ent_c_np = flat[o + n_types : o + 2 * n_types].double().cpu().numpy()
        ent_n_np = flat[o + 2 * n_types :].double().cpu().numpy()
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
        applied = A.set_lr(self.opt, sched.lr_frac)
        has_t = np.asarray(table["has_t"], bool)
        beta = sched.beta_rows(dt_all, comb_all)
        ks = sched.ks_rows(dt_all, has_t)
        lp_behav = np.asarray(table["logp_behav"], np.float32)
        target = ret.target
        use_teacher = bool(np.any(sched.ks_types > 0))

        def loss_rows(idx: np.ndarray, width: int) -> A.LossRows:
            teacher = None
            if use_teacher and has_t[idx].any():
                flat_t = axis_rows(rows, "tprob", idx)
                teacher = A.pad_ragged(flat_t, nc_all[idx], width=width)
            return A.LossRows.from_numpy(
                dev,
                n_classes=nc_all[idx],
                action=act_all[idx],
                lp_prox=lp_prox_np[idx],
                lp_behav=lp_behav[idx],
                adv=adv[idx],
                target=target[idx],
                pi=ret.pi[idx],
                v=ret.trained[idx],
                beta=beta[idx],
                ks=ks[idx],
                dtype=dt_all[idx],
                teacher=teacher,
            )

        acc: dict[str, torch.Tensor] = {k: torch.zeros((), device=dev) for k in STAT_SCALARS}
        acc.update({k: torch.zeros(n_types, device=dev) for k in STAT_VECTORS})
        gnorms: list[torch.Tensor] = []
        kls: list[float] = []
        steps = epochs_run = 0
        stopped = False
        tu = time.time()
        for epoch in range(int(cfg.epochs)):
            if not trained_idx.size or nonfinite:
                break
            mbs = A.epoch_minibatches(
                strata, seed=cfg.seed, it=it, epoch=epoch, per_mb=per_mb, world=world
            )
            for mb in mbs:
                denom = float(sum(len(s) for rr in mb for s in rr))
                if denom == 0:
                    continue
                for idx in mb[rank]:
                    if not len(idx):
                        continue
                    out = self._fwd(rows, idx)
                    lo = A.ppo_losses(
                        out.logits,
                        out.value,
                        loss_rows(idx, int(out.logits.shape[1])),
                        policy_on=sched.policy_on,
                        denom=denom,
                        cfg=self.loss_cfg,
                        n_types=n_types,
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
                "update_s": update_s,
                "rank_s": time.time() - t0,
                "nonfinite": nonfinite,
            }
        )
        stats.update(ret.stats(table))
        for k, v in means.items():
            if isinstance(v, np.ndarray):
                if k != "n_pi_t":
                    stats[k] = type_dict(v, self.dtypes)
            else:
                stats[k] = v
        stats["ent_prox_t"] = type_dict(ent_mean, self.dtypes)
        ent_norm = np.where(ent_c_np > 0, ent_n_np / np.maximum(ent_c_np, 1.0), np.nan)
        stats["ent_norm_t"] = type_dict(ent_norm, self.dtypes)
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
                    torch.from_numpy(nc_all[idx]).to(self.dev),
                    torch.from_numpy(pi[idx]).to(self.dev),
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
    m = torch.from_numpy(np.asarray(mask, bool)).to(dev)
    nc = torch.where(m, n_classes.long(), torch.full_like(n_classes.long(), 2))
    cm = A.class_mask(nc, logits.shape[-1])
    lg = torch.where(m[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    lp = torch.log_softmax(lg.masked_fill(~cm, A.NEG), dim=-1)
    h = -(torch.where(cm, lp.exp() * lp, torch.zeros_like(lp))).sum(-1)
    hn = h / torch.log(nc.float().clamp(min=2.0))
    dt = torch.from_numpy(np.asarray(dtype, np.int64)).to(dev)
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
