"""Combat-PPO screen: GAE, policy-input honesty, logit parity with `play.py`, holdouts, a learner smoke.

A hand-made fight (the starter deck plus Pommel Strike against Nibbits, as in
`test_combat_ctx`) stands in for a bank fight. The nets are randomly initialised
`CombatCtxNet`s saved in the step-0 checkpoint format, so no private checkpoint
is needed.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
from collections.abc import Iterator

import numpy as np
import pytest
import sts2_client
import torch
from ppo import rollout_starts
from ppo.combat_bench import agents
from ppo.combat_ctx import gen as G
from ppo.combat_ctx import play as P
from ppo.combat_ctx import tokens as T
from ppo.combat_ctx.intent_log import IntentLog
from ppo.combat_ctx.model import CombatCtxNet
from ppo.combat_ppo import nets
from ppo.combat_ppo.encode import SEC_PRIV, Encoder, pack, unpack
from ppo.combat_ppo.learner import Cfg, Learner, Step, gae
from ppo.combat_ppo.pool import train_pool

FIGHT = {
    "file": "unit.run.gz",
    "floor": 3,
    "deck": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH", "POMMEL_STRIKE"],
    "hp_before": 70,
    "max_hp": 80,
    "relics": ["BURNING_BLOOD", "CENTENNIAL_PUZZLE"],
    "encounter": "NIBBITS_WEAK",
    "ascension": 10,
    "potions": ["FIRE_POTION"],
    "room_type": "elite",
}


@pytest.fixture(autouse=True)
def engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if agents.is_engine_env(key):
            monkeypatch.delenv(key)


@pytest.fixture(scope="module", autouse=True)
def _v0107_engine() -> Iterator[None]:
    """Run this module on v0.107.1 under any `--game-version`.

    Its fights and expected numbers are v0.107.1's: until e2e WP1 plumbed the
    active game version into `rollout_starts.build_fight` and friends, no
    `game_version` reached the engine here, so every fight was the engine's
    default build, v0.107.1.
    """
    before = sts2_client.GAME_VERSION
    sts2_client.set_game_version("v0.107.1")
    yield
    sts2_client.set_game_version(before)


def _identity_rare() -> dict:
    v = T.vocab()
    return {s: list(range(v.tables[T.S[s]].size)) for s in T.CONTENT_SLOTS}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch.manual_seed(0)
    v = T.vocab()
    net = CombatCtxNet(v.sizes, T.N_CONT, T.N_SECTION, dim=32, n_layers=1, n_heads=2)
    # A non-trivial policy head, so parity is not checked on near-zero logits.
    torch.nn.init.normal_(net.policy_head.weight, std=1.0)
    path = str(tmp_path_factory.mktemp("cppo") / "init.pt")
    nets.save_policy(path, net, None, {"arm": "S_ctx", "rare": _identity_rare()})
    return path


def _walk(seed: int = 7, n: int = 25) -> list:
    rm = rollout_starts.build_fight(FIGHT, seed)
    cs = rm.combat_state()
    ilog = IntentLog()
    enc = Encoder("S_ctx", _identity_rare(), priv_k=10)
    out = []
    rng = random.Random(seed)
    while len(out) < n:
        acts = cs.legal_actions()
        if not acts:
            break
        snap, bv = enc.observe(ilog, cs, FIGHT["room_type"])
        out.append((cs.clone(), acts, snap, bv))
        cs.apply(acts[rng.randrange(len(acts))])
    return out


def test_gae_terminal_reward_only() -> None:
    v = np.asarray([0.5, 0.6, 0.2])
    adv, ret = gae(v, 1.0, lam=1.0)
    # λ = 1, γ = 1: every advantage is R - V(s_t); every return is R.
    assert np.allclose(adv, 1.0 - v)
    assert np.allclose(ret, 1.0)
    adv0, ret0 = gae(v, 1.0, lam=0.0)
    # λ = 0: one-step TD errors.
    assert np.allclose(adv0, [0.6 - 0.5, 0.2 - 0.6, 1.0 - 0.2])
    assert np.allclose(ret0, [0.6, 0.2, 1.0])


def test_policy_tokens_are_honest_and_priv_tokens_are_not() -> None:
    """The policy input is identical across worlds the player cannot tell apart.

    The critic-only `PRIV_DRAW` tokens see the true draw order, so some of them
    differ.
    """
    enc = Encoder("S_ctx", _identity_rare(), priv_k=10)
    priv_differs = 0
    for cs, acts, snap, bv in _walk(seed=13):
        base = enc.encode(snap, bv, cs, acts)
        assert len(base.pcat) <= 10
        for r in range(3):
            w = cs.sample_world(r)
            other = enc.encode(snap, bv, w, w.legal_actions())
            for k in ("cat", "cont", "sec"):
                assert np.array_equal(getattr(base, k), getattr(other, k)), k
            assert base.n_act == other.n_act
            if len(base.pcat) != len(other.pcat) or not np.array_equal(base.pcat, other.pcat):
                priv_differs += 1
    assert priv_differs > 0, "the PRIV_DRAW tokens never saw a different hidden draw order"


def test_pack_round_trip() -> None:
    enc = Encoder("S_ctx", _identity_rare(), priv_k=5)
    ds = [enc.encode(s, b, cs, a) for cs, a, s, b in _walk(n=8)]
    back = unpack(pack(ds))
    for a, b in zip(ds, back, strict=True):
        for k in ("cat", "cont", "sec", "pcat", "pcont"):
            assert np.array_equal(getattr(a, k), getattr(b, k))
        assert a.n_act == b.n_act


def test_policy_logits_match_play_ctxnet(ckpt: str) -> None:
    """Encoder + collate + net == `combat_ctx.play.CtxNet` on the same states."""
    ref = P.CtxNet(ckpt)
    policy, critic, meta = nets.load_init(ckpt)
    policy.eval()
    enc = Encoder(meta["arm"], meta["rare"], priv_k=10)
    rm = rollout_starts.build_fight(FIGHT, 11)
    cs = rm.combat_state()
    job = {"row": FIGHT}
    ilog = IntentLog()
    n = 0
    while n < 25:
        acts = cs.legal_actions()
        if not acts:
            break
        ref.observe(job, cs)
        snap, bv = enc.observe(ilog, cs, FIGHT["room_type"])
        if len(acts) > 1:
            z_ref = ref.logits([(job, cs, acts)])[0]
            d = enc.encode(snap, bv, cs, acts)
            with torch.no_grad():
                z, _ = nets.run(policy, nets.collate([d], "cpu", priv=False))
                _zc, v = nets.run(critic, nets.collate([d], "cpu", priv=True))
            assert np.allclose(z[0, : d.n_act].double().numpy(), z_ref, atol=1e-5)
            assert torch.isfinite(v).all()
            n += 1
        cs.apply(acts[n % len(acts)])
    assert n >= 10
    assert critic.emb_sec.weight.shape[0] == T.N_SECTION + 1
    assert torch.equal(critic.emb_sec.weight[SEC_PRIV], critic.emb_sec.weight[T.SEC["DRAW_TOP"]])


def test_policy_batch_padding_does_not_change_logits(ckpt: str) -> None:
    policy, _c, meta = nets.load_init(ckpt)
    policy.eval()
    enc = Encoder(meta["arm"], meta["rare"], priv_k=10)
    ds = [enc.encode(s, b, cs, a) for cs, a, s, b in _walk(seed=5, n=12) if len(a) > 1]
    with torch.no_grad():
        zb, _ = nets.run(policy, nets.collate(ds, "cpu", priv=False))
        for i, d in enumerate(ds):
            z1, _ = nets.run(policy, nets.collate([d], "cpu", priv=False))
            assert torch.allclose(zb[i, : d.n_act], z1[0, : d.n_act], atol=1e-5)


def test_train_pool_holds_out_every_holdout_run(tmp_path) -> None:
    files = [f"r{i:04d}.run.gz" for i in range(400)]
    rows = [dict(FIGHT, file=f, floor=1) for f in files]
    rows += [dict(FIGHT, file=files[0], floor=9, encounter=G.WG, room_type="boss")]
    hb = tmp_path / "hbank.jsonl"
    hb.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    b8 = tmp_path / "boss800.jsonl"
    b8.write_text(json.dumps(dict(FIGHT, file=files[1])) + "\n", encoding="utf-8")
    pool, meta = train_pool(str(hb), str(b8))
    kept = {r["file"] for rs in pool.values() for r in rs}
    assert files[0] not in kept  # a run with a Waterfall Giant fight
    assert files[1] not in kept  # a boss800 run
    assert all(G.fold_of(f) != 2 for f in kept)
    assert any(G.fold_of(f) == 2 for f in files)
    assert kept and meta["runs"] == len(kept)


def test_learner_smoke(ckpt: str, tmp_path) -> None:
    """Two tiny CPU iterations on a fixed pool: stats, checkpoints, state, resume."""
    out = str(tmp_path / "run")
    cfg = Cfg(
        init=ckpt,
        out=out,
        fixed_rows=[FIGHT],
        weights={"elite": 1.0},
        workers=2,
        n_envs=3,
        iter_decisions=120,
        minibatch=64,
        epochs=2,
        warmup_iters=1,
        infer_device="cpu",
        update_devices="cpu,cpu",
        bf16=False,
        max_iters=2,
        ckpt_every=1,
        eval_every=10**12,
    )
    Learner(cfg).train()
    with open(os.path.join(out, "stats.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    assert [r["it"] for r in rows] == [1, 2]
    assert rows[0]["errors"] == 0 and rows[1]["errors"] == 0
    assert not rows[0]["train_policy"] and rows[1]["train_policy"]
    assert rows[1]["decisions"] >= 240
    assert os.path.exists(os.path.join(out, "ckpt", "final.pt"))
    # The final checkpoint plays through `play.py`'s loader.
    pol = P.policy_from_spec(f"ctx:{os.path.join(out, 'ckpt', 'final.pt')}")
    res = P.play_jobs(pol, P.make_jobs([FIGHT], 1)[0], 1.0, True)
    assert len(res) == 1
    # Resume picks up the iteration counter.
    cfg.max_iters = 3
    lr = Learner(cfg)
    assert lr.it == 2


def test_own_pool_rows_build_and_mix_by_share(tmp_path) -> None:
    """Own-deck rows (bench schema) build, and the sampler mixes them at `own_share`."""
    from ppo.combat_ppo.pool import StartSampler, is_own, own_pool

    own_row = dict(
        FIGHT,
        file="seed152000000",
        key="seed152000000:3",
        character="Ironclad",
        cohort="own",
        potion_slots=3,
        monsters=None,
    )
    big_belt = dict(own_row, potion_slots=6)
    path = tmp_path / "own.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (own_row, big_belt)) + "\n", encoding="utf-8")
    own = own_pool([str(path)])
    assert len(own["elite"]) == 1  # the 6-slot belt cannot be built and is dropped
    rm = rollout_starts.build_fight(own["elite"][0], 2**30 + 5)
    assert rm.combat_state().legal_actions()
    human = {"boss": [], "elite": [dict(FIGHT)], "monster": []}
    s = StartSampler(human, {"elite": 1.0}, seed=0, own=own, own_share=0.25)
    share = np.mean([is_own(s.draw()[0]) for _ in range(4000)])
    assert abs(share - 0.25) < 0.03
    assert not any(is_own(StartSampler(human, {"elite": 1.0}, seed=1).draw()[0]) for _ in range(50))


def test_padded_rows_of_flat_equal_padded_of_subset(ckpt: str) -> None:
    """`Padded.from_flat(pack(ds), rows)` is `Padded([ds[i] for i in rows])`."""
    _, _, meta = nets.load_init(ckpt)
    enc = Encoder(meta["arm"], meta["rare"], priv_k=10)
    ds = [enc.encode(s, b, cs, a) for cs, a, s, b in _walk(seed=5, n=24) if len(a) > 1]
    rows = np.asarray([3, 0, 7, 7, len(ds) - 1, 2])
    for priv in (True, False):
        a = nets.Padded.from_flat(pack(ds), rows, priv=priv)
        b = nets.Padded([ds[i] for i in rows], priv=priv)
        for k in ("cat", "cont", "sec", "act_pos", "n_honest", "n_total", "n_act"):
            assert np.array_equal(getattr(a, k), getattr(b, k)), (priv, k)


def _one_update(cfg: Cfg, done: list) -> dict:
    """One update by `cfg.update_devices`' ranks on a hand-made batch."""
    lr = Learner(cfg)
    lr.start_ranks()
    try:
        lr._send(done, {"n_done": sum(len(t) for t, _ in done)})
        kind, _r, res = lr._rank_get()
        assert kind == "done"
    finally:
        if lr.inflight is not None:
            lr.inflight["shm"].unlink()
            lr.inflight["shm"].close()
        lr.stop_ranks()
    return res


def test_data_parallel_update_matches_one_rank(ckpt: str, tmp_path) -> None:
    """Two ranks, each with half of every minibatch, train like one rank on all of it.

    The reported losses are global means (they ride in the gradient all_reduce),
    so they must match the one-rank run. Adam's first steps are nearly
    sign-only, so the weights are compared on their mean change.
    """
    _, _, meta = nets.load_init(ckpt)
    enc = Encoder(meta["arm"], meta["rare"], priv_k=10)
    rng = np.random.default_rng(0)
    done = []
    for seed in range(6):
        traj = []
        for cs, a, s, b in _walk(seed=seed, n=20):
            if len(a) > 1:
                d = enc.encode(s, b, cs, a)
                traj.append(Step(d, int(rng.integers(d.n_act)), float(np.log(1.0 / d.n_act))))
        done.append((traj, float(rng.random())))
    res = {}
    for devs, micro in (("cpu", 0), ("cpu,cpu", 0), ("cpu", 5)):
        cfg = Cfg(
            init=ckpt,
            out=str(tmp_path / f"{devs.replace(',', '_')}_{micro}"),
            infer_device="cpu",
            update_devices=devs,
            micro_rows=micro,
            minibatch=16,
            epochs=2,
            warmup_iters=0,
            target_kl=1e9,
            bf16=False,
        )
        res[devs, micro] = _one_update(cfg, done)
    one, two = res["cpu", 0], res["cpu,cpu", 0]
    for k in ("vloss", "ploss", "entropy", "kl", "is_w", "value_ev", "adv_std", "lag_kl"):
        assert one[k] == pytest.approx(two[k], rel=1e-3, abs=1e-6), k
    # Micro-chunk accumulation (`micro_rows`) is the same update.
    acc = res["cpu", 5]
    for k in ("vloss", "ploss", "entropy", "kl", "is_w", "value_ev", "adv_std", "lag_kl"):
        assert one[k] == pytest.approx(acc[k], rel=1e-3, abs=1e-6), k
    init, _, _ = nets.load_init(ckpt)
    for k, w0 in init.state_dict().items():
        if k == "policy_head.bias":
            continue  # shared by every logit, cancels in the softmax: its gradient is noise
        d1 = one["policy"][k] - w0.numpy()
        d2 = two["policy"][k] - w0.numpy()
        if np.abs(d1).mean() > 0:
            assert np.abs(d1 - d2).mean() < 0.05 * np.abs(d1).mean(), k


def test_policy_ignores_masked_priv_tokens(ckpt: str) -> None:
    """`Padded` + `batch_inputs` == `collate` for both nets.

    The policy reads the shared padded arrays with its priv tokens masked. Its
    logits must equal those of a batch built with no priv tokens at all.
    """
    policy, critic, meta = nets.load_init(ckpt)
    policy.eval()
    critic.eval()
    enc = Encoder(meta["arm"], meta["rare"], priv_k=10)
    ds = [enc.encode(s, b, cs, a) for cs, a, s, b in _walk(seed=9, n=20) if len(a) > 1]
    assert any(len(d.pcat) for d in ds)
    g = nets.Padded(ds).to("cpu")
    idx = torch.arange(len(ds) - 1, -1, -1)  # a reordered selection
    with torch.no_grad():
        zp, _ = nets.run(policy, nets.batch_inputs(g, idx, critic=False))
        zr, _ = nets.run(policy, nets.collate([ds[i] for i in idx], "cpu", priv=False))
        _, vp = nets.run(critic, nets.batch_inputs(g, idx, critic=True))
        _, vr = nets.run(critic, nets.collate([ds[i] for i in idx], "cpu", priv=True))
    fin = torch.isfinite(zr)
    assert torch.equal(fin, torch.isfinite(zp))
    assert torch.allclose(zp[fin], zr[fin], atol=1e-5)
    assert torch.allclose(vp, vr, atol=1e-5)


def test_bench_agent_plays_like_play_py(ckpt: str) -> None:
    """The `combat_bench` agent (through `RunManager`) == `play.py` greedy (on `CombatState`)."""
    from ppo.combat_ppo.agent import make

    for seed in (3, 4):
        job = {"i": 0, "fight": 0, "rep": 0, "row": FIGHT, "build_seed": seed, "seed": 17}
        ref = P.play_jobs(P.policy_from_spec(f"ctx:{ckpt}"), [job], 1.0, True)[0]
        rm = rollout_starts.build_fight(FIGHT, seed)
        agent = make(seed=seed, ckpt=ckpt)
        steps = 0
        while not rm.is_over() and rm.phase() == "combat" and steps < 600:
            if agent.step(rm) is None:
                break
            steps += 1
        assert agent.decisions == ref["decisions"]
        assert bool(rm.player_won) == ref["won"]


@pytest.fixture(scope="module")
def run_ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A random-init run net in the `ppo.trainer` checkpoint format (default schema)."""
    from ppo.model import RunPolicyNet
    from ppo.trainer import PPOConfig

    torch.manual_seed(1)
    cfg = PPOConfig()
    net = RunPolicyNet(dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    path = str(tmp_path_factory.mktemp("cppo_run") / "run_init.pt")
    torch.save(
        {"state_dict": net.state_dict(), "cfg": dict(cfg.__dict__), "adv_mean": 0.0, "adv_sq": 1.0},
        path,
    )
    return path


def test_yield_combat_run_equals_policy_combat_run(ckpt: str, run_ckpt: str) -> None:
    """`combat="yield"` driven by the bench agent == `combat="policy"`, run for run."""
    from ppo.combat_ppo.agent import make
    from ppo.env import CombatTurn, Decision, PpoRunEnv
    from ppo.net_policy import NetPolicy

    pol = NetPolicy(run_ckpt)
    rng = np.random.default_rng(0)
    for seed in (11, 12):
        ref = PpoRunEnv(seed=seed, ascension=10, combat="policy", combat_ckpt=ckpt)
        out = ref.reset()
        picks = []
        while isinstance(out, Decision):
            idx = pol.sample(out.obs, out.actions, out.metrics, rng, greedy=True)[0]
            picks.append(idx)
            out = ref.step(idx)
        env = PpoRunEnv(seed=seed, ascension=10, combat="yield")
        agent = make(seed=seed, ckpt=ckpt)
        out2 = env.reset()
        got = []
        while True:
            if isinstance(out2, CombatTurn):
                rm = env.rm
                while not rm.is_over() and rm.phase() == "combat":
                    assert agent.step(rm) is not None
                    env.combat_step()
                out2 = env.resume()
            elif isinstance(out2, Decision):
                idx = pol.sample(out2.obs, out2.actions, out2.metrics, rng, greedy=True)[0]
                got.append(idx)
                out2 = env.step(idx)
            else:
                break
        assert got == picks
        assert out2.won == out.won and out2.metrics == out.metrics
        assert out2.steps == out.steps


def test_hybrid_learner_smoke(ckpt: str, run_ckpt: str, tmp_path) -> None:
    """Run slots next to fight slots: run fights train, run rows reach the run learner."""
    out = str(tmp_path / "hyb")
    cfg = Cfg(
        init=ckpt,
        out=out,
        fixed_rows=[FIGHT],
        weights={"elite": 1.0},
        workers=2,
        n_envs=3,
        iter_decisions=400,
        minibatch=128,
        epochs=1,
        warmup_iters=0,
        infer_device="cpu",
        update_devices="cpu",
        bf16=False,
        max_iters=5,
        ckpt_every=10**12,
        eval_every=10**12,
        init_decisions=1000,
        run_slots=2,
        run_init=run_ckpt,
        run_device="cpu",
        run_games_per_iter=2,
        run_warmup=1,
        run_ckpt_every=1,
    )
    Learner(cfg).train()
    with open(os.path.join(out, "stats.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    assert rows[0]["decisions"] > 1000  # the branch starts at init_decisions
    assert sum(sum(r["fights_run"].values()) for r in rows) > 0
    assert sum(r["run_fight_decisions"] for r in rows) > 0
    assert sum(r["runs_done"] for r in rows) >= 2
    with open(os.path.join(out, "run", "stats.jsonl"), encoding="utf-8") as f:
        rrows = [json.loads(x) for x in f]
    assert rrows and rrows[0]["warmup"] and rrows[0]["errors"] == 0
    assert rrows[0]["n_trans"] > 0
    assert os.path.exists(os.path.join(out, "run", "ckpt", "v00001.pt"))


_DEAD_READER_SCRIPT = """
import sys
from multiprocessing import get_context

from ppo.combat_ppo.learner import _release_queues


def reader(q):
    pass  # exits without reading, like an OOM-killed update rank


if __name__ == "__main__":
    ctx = get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=reader, args=(q,))
    p.start()
    p.join()
    q.put(b"x" * 10_000_000)  # more than the pipe holds
    if sys.argv[1] == "release":
        _release_queues([q, None])
    raise SystemExit(3)
"""


@pytest.mark.parametrize("mode", ["release", "keep"])
def test_learner_exits_when_a_queue_reader_died(tmp_path, mode):
    """The learner hung on exit after a rank OOM: the queue's feeder thread blocked on the full pipe."""
    import subprocess
    import sys

    script = tmp_path / "dead_reader.py"
    script.write_text(_DEAD_READER_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "python")}
    try:
        rc = subprocess.run([sys.executable, str(script), mode], env=env, timeout=30).returncode
    except subprocess.TimeoutExpired:
        rc = "hang"
    assert rc == (3 if mode == "release" else "hang")


def test_runslot_value_reward(ckpt: str, run_ckpt: str) -> None:
    """`run_reward = "value"`: every in-run fight ends with `scale · (m1 − m0)`.

    m0 is the run value at the last run decision before the fight, m1 the one
    at the first decision after it, or the terminal base value when the fight
    ended the run (a loss: the death floor's progress only).
    """
    import itertools

    from ppo.combat_ppo.runslot import RunSlot
    from ppo.net_policy import NetPolicy
    from ppo.rewards import ShapingConfig, base_value

    _, _, meta = nets.load_init(ckpt)
    rc = torch.load(run_ckpt, map_location="cpu", weights_only=True)["cfg"]
    shaping = {
        "hp_coef": rc["shaping_hp_coef"],
        "upg_coef": rc["shaping_upg_coef"],
        "relic_coef": rc["shaping_relic_coef"],
        "offset": rc["shaping_offset"],
    }
    cfg = {
        "max_steps": 600,
        "run_ascension": 10,
        "run_character": "Ironclad",
        "run_act1": Cfg.__dataclass_fields__["run_act1"].default,
        "run_reward": "value",
        "run_reward_scale": 2.0,
        "run_shaping": shaping,
        "_run_ver": lambda: 0,
    }
    pol = NetPolicy(run_ckpt)
    ends: list[dict] = []
    run_rows: list[dict] = []
    slot = RunSlot(
        0,
        itertools.count(21),
        cfg,
        Encoder(meta["arm"], meta["rare"], priv_k=10),
        lambda: pol,
        ends,
        run_rows,
    )
    rng = np.random.default_rng(0)
    while len(run_rows) < 2:
        d = slot.advance()
        slot.apply(int(rng.integers(d.n_act)))
    fights = [e for e in ends if not e["error"]]
    assert fights and all(e["run"] for e in ends)
    assert sum(r["fights"] for r in run_rows) == len(ends)  # none left waiting
    for e in fights:
        assert e["r_run"] == pytest.approx(2.0 * (e["m1"] - e["m0"]))
    lost = [e for e in fights if not e["won"]]
    assert len(lost) == 2  # each run ends on its lost fight
    for e, row in zip(lost, run_rows, strict=True):
        m = {"ascension": 10, **row["final"]}
        assert e["m1"] == pytest.approx(base_value(m, False, ShapingConfig(**shaping)))


def test_hybrid_value_reward_branch(ckpt: str, run_ckpt: str, tmp_path) -> None:
    """A fight-reward hybrid branches into `run_reward = "value"` and resumes again."""
    out = str(tmp_path / "hyb")
    base = dict(
        init=ckpt,
        out=out,
        fixed_rows=[FIGHT],
        weights={"elite": 1.0},
        workers=2,
        n_envs=3,
        iter_decisions=400,
        minibatch=128,
        epochs=1,
        warmup_iters=0,
        infer_device="cpu",
        update_devices="cpu",
        bf16=False,
        ckpt_every=10**12,
        eval_every=10**12,
        init_decisions=1000,
        run_slots=2,
        run_init=run_ckpt,
        run_device="cpu",
        run_games_per_iter=2,
        run_warmup=1,
        run_ckpt_every=1,
    )
    Learner(Cfg(**base, max_iters=2)).train()
    val = dict(base, run_reward="value", run_reward_scale=3.0, policy_frozen_until=4)
    Learner(Cfg(**val, max_iters=6)).train()
    st = torch.load(os.path.join(out, "state.pt"), map_location="cpu", weights_only=True)
    assert "value_head_run.weight" in st["critic"]
    assert len(st["opt_v"]["param_groups"]) == 2
    Learner(Cfg(**val, max_iters=7)).train()  # resume with the head in the state
    with open(os.path.join(out, "stats.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    assert [r["it"] for r in rows] == list(range(1, 8))
    assert all(math.isnan(r["reward_run_mean"]) for r in rows[:2])
    assert any(math.isfinite(r["reward_run_mean"]) for r in rows[2:])
    assert any(math.isfinite(r["value_ev_run"]) for r in rows[2:])
