"""The data-parallel PPO update: one process per update device.

Every rank holds a replica of the policy, the critic and both optimizers. They
start from the same weights and stay identical, because every rank applies the
same summed gradient (one `all_reduce` per minibatch) with the same optimizer.
`_check_in_sync` asserts this after each update.

The learner hands over an iteration's decisions as one `SharedMemory` block
(`to_shm`): the flat token arrays of `encode.Packed`, plus the sampled action,
its behaviour log-prob, and the episode lengths and terminal rewards. Episodes
are contiguous, in order. Each rank:

1. Splits the length-sorted decisions into minibatches of `cfg.minibatch` and
   takes its contiguous `1/world` share of each. It pads only those rows onto
   its device.
2. Computes, for its rows, the critic's values and the policy's log-prob of the
   taken action (the proximal policy, i.e. the weights before this update).
   It gathers the values of all rows from all ranks, then computes GAE and the
   EWMA advantage normalisation over the whole batch. Every rank does the same
   and gets the same result.
3. Runs `cfg.epochs` passes over the minibatches, in an order every rank draws
   from the same seed. Each minibatch's loss is the sum over the rank's rows
   divided by the minibatch's total size, so the summed gradient is the
   gradient of the whole minibatch's mean. The KL used for early stopping and
   the logged statistics ride in the same `all_reduce`.

**Decoupled PPO** (Hilton et al. 2021, "Batch size-invariance for policy
optimization"). The learner keeps collecting while a rank set updates, so a
batch was sampled by the policy one update behind the one that trains on it.
The clipped ratio is therefore taken against the proximal policy (this
update's starting weights). The behaviour-to-proximal importance weight
`π_prox / π_behav` (truncated at `cfg.is_trunc`) multiplies each row's clipped
surrogate. With no lag, the weight is 1 and the loss is standard PPO.
"""

from __future__ import annotations

import os
import pathlib
import queue
import time
import traceback
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ppo.trainer import RunningMoments

from . import nets

# ── shared-memory batches ─────────────────────────────────────────────────


def to_shm(
    spec: dict[str, tuple[tuple[int, ...], Any, np.ndarray | list[np.ndarray]]],
    *,
    name: str | None = None,
) -> tuple[shared_memory.SharedMemory, dict[str, tuple[int, tuple[int, ...], str]]]:
    """One `SharedMemory` block holding every array of `spec`.

    `spec[name] = (shape, dtype, fill)`: `fill` is an array, or a list of
    arrays concatenated along axis 0. The caller owns the block (`unlink`).
    `name` names it (default: a random name).
    """
    layout: dict[str, tuple[int, tuple[int, ...], str]] = {}
    off = 0
    for k, (shape, dt, _fill) in spec.items():
        d = np.dtype(dt)
        layout[k] = (off, tuple(int(x) for x in shape), d.str)
        off += -(-int(np.prod(shape, dtype=np.int64)) * d.itemsize // 64) * 64
    shm = shared_memory.SharedMemory(name=name, create=True, size=max(off, 64), track=False)
    for k, (_shape, _dt, fill) in spec.items():
        v = shm_view(shm, layout[k])
        if isinstance(fill, list):
            if v.size:
                np.concatenate(fill, out=v)
        else:
            v[...] = fill
        del v
    return shm, layout


def shm_view(shm: shared_memory.SharedMemory, ent: tuple[int, tuple[int, ...], str]) -> np.ndarray:
    off, shape, dt = ent
    return np.ndarray(shape, np.dtype(dt), buffer=shm.buf, offset=off)


# ── PPO pieces ────────────────────────────────────────────────────────────


def gae(values: np.ndarray, reward: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns of one episode with a terminal reward only (γ = 1)."""
    n = len(values)
    nxt = np.append(values[1:], 0.0)
    r = np.zeros(n)
    r[-1] = reward
    delta = r + nxt - values
    adv = np.zeros(n)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + values


def entropy_coef(cfg: Any, frac: float) -> float:
    return cfg.ent0 * (cfg.ent1 / cfg.ent0) ** min(1.0, frac)


def minibatches(n_total: np.ndarray, size: int, world: int) -> list[list[np.ndarray]]:
    """Length-bucketed minibatches, each split into `world` contiguous shares."""
    order = np.argsort(n_total, kind="stable")
    return [np.array_split(order[i : i + size], world) for i in range(0, len(order), size)]


# Statistics summed across ranks with every minibatch's gradient.
_STATS = ("vloss", "ploss", "entropy", "kl", "clipfrac", "is_w")


class Rank:
    def __init__(self, rank: int, world: int, dev: torch.device, cfg: Any) -> None:
        self.rank, self.world, self.dev, self.cfg = rank, world, dev, cfg
        self.policy, self.critic, self.meta = nets.load_init(cfg.init)
        run_head = getattr(cfg, "run_reward", "fight") == "value"
        if run_head:
            nets.add_run_head(self.critic)
        self.policy.to(dev)
        self.critic.to(dev)
        self.opt_p = torch.optim.AdamW(self.policy.parameters(), lr=cfg.lr, weight_decay=1e-4)
        # The run head is its own param group, after the rest: a state saved
        # before the head existed restores group 0 and leaves the head fresh.
        base_v = [
            p for n, p in self.critic.named_parameters() if not n.startswith("value_head_run.")
        ]
        self.opt_v = torch.optim.AdamW(base_v, lr=cfg.critic_lr, weight_decay=1e-4)
        if run_head:
            self.opt_v.add_param_group({"params": list(self.critic.value_head_run.parameters())})
        self.adv_rm = RunningMoments()
        p = os.path.join(cfg.out, "state.pt")
        if os.path.exists(p):
            st = torch.load(p, map_location="cpu", weights_only=True)
            self.policy.load_state_dict(st["policy"])
            missing, unexpected = self.critic.load_state_dict(st["critic"], strict=False)
            if unexpected or any(not k.startswith("value_head_run.") for k in missing):
                raise RuntimeError(f"critic state: missing {missing}, unexpected {unexpected}")
            self.opt_p.load_state_dict(st["opt_p"])
            ov = st["opt_v"]
            if run_head and missing:  # branching from a state without the run head
                saved = self.opt_v.state_dict()
                ov = {
                    "state": ov["state"],
                    "param_groups": [ov["param_groups"][0], saved["param_groups"][1]],
                }
            self.opt_v.load_state_dict(ov)
            self.adv_rm.__dict__.update(st["adv_rm"])
        # The parameters each loss reaches: the policy's value head and the
        # critic's policy head never get a gradient, so they are left out of
        # the reduction and keep `grad = None` (AdamW skips them, no decay).
        self.params = [
            p
            for n, p in self.policy.named_parameters()
            if not n.startswith(("value_head.", "value_norm."))
        ] + [p for n, p in self.critic.named_parameters() if not n.startswith("policy_head.")]

    def _autocast(self) -> Any:
        return torch.autocast(
            "cuda", torch.bfloat16, enabled=self.cfg.bf16 and self.dev.type == "cuda"
        )

    @torch.no_grad()
    def _forward_all(
        self, g: dict[str, torch.Tensor], act: torch.Tensor, chunk: int = 8192
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Critic values and the policy's log-prob of `act`, for every local row."""
        self.policy.eval()
        self.critic.eval()
        n = len(g["n_total"])
        v_out = torch.empty(n, device=self.dev)
        lp_out = torch.empty(n, device=self.dev)
        for i in range(0, n, chunk):
            idx = torch.arange(i, min(n, i + chunk), device=self.dev)
            with self._autocast():
                _, v = nets.run(self.critic, nets.batch_inputs(g, idx, critic=True))
                z, _ = nets.run(self.policy, nets.batch_inputs(g, idx, critic=False))
            v_out[idx] = v.float()
            lp = torch.log_softmax(z.float(), -1)
            lp_out[idx] = lp.gather(1, act[idx][:, None]).squeeze(-1)
        return v_out, lp_out

    def _gather_rows(self, local: torch.Tensor, rows: list[np.ndarray], n: int) -> np.ndarray:
        """Assemble per-row values from every rank's share (`rows[r]` = rank r's rows)."""
        m = max(len(r) for r in rows)
        buf = torch.zeros(m, device=self.dev)
        buf[: len(local)] = local
        outs = [torch.empty_like(buf) for _ in range(self.world)]
        dist.all_gather(outs, buf)
        full = np.empty(n, np.float32)
        for r, o in zip(rows, outs, strict=True):
            full[r] = o[: len(r)].cpu().numpy()
        return full

    def _check_in_sync(self) -> None:
        ps = list(self.policy.parameters()) + list(self.critic.parameters())
        s = torch.stack([p.detach().double().sum() for p in ps]).sum()[None]
        outs = [torch.empty_like(s) for _ in range(self.world)]
        dist.all_gather(outs, s)
        vals = [float(o) for o in outs]
        if max(vals) != min(vals):
            raise RuntimeError(f"update ranks diverged: parameter sums {vals}")

    def update(self, job: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.time()
        shm = shared_memory.SharedMemory(name=job["shm"], track=False)
        try:
            f = {k: shm_view(shm, e) for k, e in job["layout"].items()}
            n = len(f["n_tok"])
            mbs = minibatches(
                f["n_tok"].astype(np.int64) + f["n_priv"].astype(np.int64),
                cfg.minibatch,
                self.world,
            )
            rows = [np.concatenate([mb[r] for mb in mbs]) for r in range(self.world)]
            mine = rows[self.rank]
            pad = nets.Padded.from_flat(SimpleNamespace(**f), mine, priv=True)
            act = f["action"][mine].astype(np.int64)
            blp = f["logp"][mine].astype(np.float32)
            ep_len = np.array(f["ep_len"], np.int64)
            ep_run = np.array(f["ep_run"], bool) if "ep_run" in f else np.zeros(len(ep_len), bool)
            reward = np.array(f["reward"], np.float64)
            f = {}
        finally:
            shm.close()
        g = pad.to(self.dev)
        del pad
        if ep_run.any():  # the rows the critic values with `value_head_run`
            g["is_run"] = torch.from_numpy(np.repeat(ep_run, ep_len)[mine]).to(self.dev)
        act_g = torch.from_numpy(act).to(self.dev)
        t_prep = time.time()
        micro = int(getattr(cfg, "micro_rows", 0) or 0)
        v_mine, lp_prox = self._forward_all(g, act_g, chunk=micro or 8192)
        vals = self._gather_rows(v_mine, rows, n)
        adv = np.empty(n, np.float32)
        ret = np.empty(n, np.float32)
        o = 0
        for length, r in zip(ep_len, reward, strict=True):
            a, rt = gae(vals[o : o + length].astype(np.float64), float(r), cfg.gae_lambda)
            adv[o : o + length] = a
            ret[o : o + length] = rt
            o += length
        assert o == n, (o, n)
        self.adv_rm.update(adv)
        adv_n = (adv - self.adv_rm.mean) / self.adv_rm.std
        ev = 1.0 - float(np.var(ret - vals)) / max(1e-8, float(np.var(ret)))
        rr = np.repeat(ep_run, ep_len)
        ev_run = (
            1.0 - float(np.var(ret[rr] - vals[rr])) / max(1e-8, float(np.var(ret[rr])))
            if rr.sum() > 1
            else float("nan")
        )
        t_fwd = time.time()

        frac = float(job["frac"])
        lr = cfg.lr * (1.0 - (1.0 - cfg.lr_final_frac) * frac)
        for pg in self.opt_p.param_groups:
            pg["lr"] = lr
        ent_c = entropy_coef(cfg, frac)
        train_policy = bool(job["train_policy"])

        dv = self.dev
        adv_t = torch.from_numpy(adv_n[mine].astype(np.float32)).to(dv)
        ret_t = torch.from_numpy(ret[mine]).to(dv)
        ov_t = torch.from_numpy(vals[mine]).to(dv)
        blp_t = torch.from_numpy(blp).to(dv)
        # Local row ranges of this rank's share of each minibatch.
        lens = np.asarray([len(mb[self.rank]) for mb in mbs])
        loc = np.concatenate([[0], np.cumsum(lens)])
        sizes = [sum(len(x) for x in mb) for mb in mbs]

        rng = np.random.default_rng(cfg.seed * 7 + int(job["it"]))
        sums: dict[str, list[float]] = {k: [] for k in _STATS}
        gnorms: list[float] = []
        stop = False
        epochs_run = 0
        self.policy.train()
        self.critic.train()
        n_stats = len(_STATS)
        for _ep in range(cfg.epochs):
            epochs_run += 1
            for ci in rng.permutation(len(mbs)):
                nc = sizes[ci]
                lo, hi = int(loc[ci]), int(loc[ci + 1])
                self.opt_v.zero_grad(set_to_none=True)
                self.opt_p.zero_grad(set_to_none=True)
                stats = torch.zeros(n_stats, device=dv)
                # Gradient accumulation over micro-chunks (`cfg.micro_rows`):
                # every loss term is a sum over rows divided by `nc`, so the
                # accumulated gradient equals the whole share's.
                step = micro if micro > 0 else max(1, hi - lo)
                for mlo in range(lo, hi, step):
                    idx = torch.arange(mlo, min(hi, mlo + step), device=dv)
                    xv = nets.batch_inputs(g, idx, critic=True)
                    with self._autocast():
                        _, v = nets.run(self.critic, xv)
                    v = v.float()
                    ov, rt = ov_t[idx], ret_t[idx]
                    v_clip = ov + (v - ov).clamp(-cfg.clip, cfg.clip)
                    vloss = 0.5 * torch.max((v - rt) ** 2, (v_clip - rt) ** 2).sum() / nc
                    vloss.backward()
                    stats[0] += vloss.detach()
                    if train_policy:
                        xp = nets.batch_inputs(g, idx, critic=False)
                        with self._autocast():
                            logits, _ = nets.run(self.policy, xp)
                        lp_all = torch.log_softmax(logits.float(), -1)
                        lp = lp_all.gather(1, act_g[idx][:, None]).squeeze(-1)
                        ent = -(lp_all.exp() * lp_all.masked_fill(~xp["act_mask"], 0.0)).sum(-1)
                        lpp = lp_prox[idx]
                        ratio = (lp - lpp).exp()
                        w = (lpp - blp_t[idx]).exp().clamp(max=cfg.is_trunc)
                        a_t = adv_t[idx]
                        surr = torch.min(ratio * a_t, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_t)
                        ploss = -(w * surr).sum() / nc
                        loss = ploss - ent_c * ent.sum() / nc
                        loss.backward()
                        with torch.no_grad():
                            d = lp - lpp
                            stats[1] += ploss.detach()
                            stats[2] += ent.sum() / nc
                            stats[3] += ((d.exp() - 1) - d).sum() / nc
                            stats[4] += ((ratio - 1).abs() > cfg.clip).float().sum() / nc
                            stats[5] += w.sum() / nc
                # One all_reduce: both nets' gradients and the statistics.
                flat = torch.cat(
                    [
                        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                        for p in self.params
                    ]
                    + [stats]
                )
                if self.world > 1:
                    dist.all_reduce(flat)
                o = 0
                for p in self.params:
                    k = p.numel()
                    if p.grad is None:
                        p.grad = flat[o : o + k].view_as(p).clone()
                    else:
                        p.grad.copy_(flat[o : o + k].view_as(p))
                    o += k
                tot = flat[o:].tolist()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                self.opt_v.step()
                sums["vloss"].append(tot[0])
                if not train_policy:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                self.opt_p.step()
                gnorms.append(float(gn))
                for k, x in zip(_STATS[1:], tot[1:], strict=True):
                    sums[k].append(x)
                if tot[3] > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        t_ep = time.time()
        self._check_in_sync()
        # The lag between the behaviour and the proximal policy, over this rank's rows.
        with torch.no_grad():
            d = lp_prox - blp_t
            lag = torch.stack([((d.exp() - 1) - d).sum(), torch.tensor(float(len(d)), device=dv)])
            if self.world > 1:
                dist.all_reduce(lag)
        del g
        out = {k: float(np.mean(v)) for k, v in sums.items() if v}
        if gnorms:
            out["gradnorm"] = float(np.mean(gnorms))
        out.update(
            {
                "epochs_run": epochs_run,
                "kl_stop": stop,
                "lr": lr,
                "ent_coef": ent_c,
                "value_ev": ev,
                "value_ev_run": ev_run,
                "adv_mean": float(adv.mean()),
                "adv_std": float(adv.std()),
                "ret_mean": float(ret.mean()),
                "train_policy": train_policy,
                "n_steps": n,
                "lag_kl": float(lag[0] / lag[1].clamp(min=1)),
                "upd_prep_s": t_prep - t0,
                "upd_fwd_s": t_fwd - t_prep,
                "upd_epochs_s": t_ep - t_fwd,
                "update_s": time.time() - t0,
            }
        )
        return out

    def save(self, job: dict[str, Any]) -> None:
        """Rank 0: the checkpoint (if asked) and `state.pt`."""
        if job.get("ckpt"):
            nets.save_policy(job["ckpt"], self.policy, self.critic, self.meta, job["ckpt_extra"])
        st = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_p": self.opt_p.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "adv_rm": dict(self.adv_rm.__dict__),
            **job["state"],
        }
        path = os.path.join(self.cfg.out, "state.pt")
        torch.save(st, path + ".tmp")
        pathlib.Path(path + ".tmp").replace(path)

    def policy_numpy(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.policy.state_dict().items()}


def rank_main(rank: int, world: int, dev_s: str, store: str, cfg: Any, inq: Any, outq: Any) -> None:
    """Process entry: join the group, then serve `update` / `save` / `stop` jobs."""
    import faulthandler

    faulthandler.enable()
    torch.set_num_threads(2)
    try:
        dev = torch.device(dev_s)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        dist.init_process_group(
            "nccl" if dev.type == "cuda" else "gloo",
            init_method=f"file://{store}",
            rank=rank,
            world_size=world,
        )
        r = Rank(rank, world, dev, cfg)
        outq.put(("ready", rank, None))
        parent = os.getppid()
        while True:
            try:
                msg = inq.get(timeout=30)
            except queue.Empty:
                if os.getppid() != parent:  # the learner died: do not linger on the GPU
                    break
                continue
            if msg[0] == "stop":
                break
            job = msg[1]
            if msg[0] == "update":
                res = r.update(job)
                if rank == 0:
                    r.save(job)
                    res["policy"] = r.policy_numpy()
                    outq.put(("done", rank, res))
            elif msg[0] == "save" and rank == 0:
                r.save(job)
                outq.put(("saved", rank, None))
    except BaseException:  # every failure goes to the learner
        outq.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
