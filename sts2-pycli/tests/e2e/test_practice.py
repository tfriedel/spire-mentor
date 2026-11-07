"""Practice on own runs (`ppo.e2e.practice`, plan §3.3): the schedule, the queues, the
reseed stream, and live starts from captured runs (T13 and the segment ends)."""

from __future__ import annotations

import collections
import math
import random

import pytest
import sts2_client
from ppo.e2e import contracts as C
from ppo.e2e import env as E
from ppo.e2e import factory
from ppo.e2e import practice as P
from ppo.e2e.env import Capture, E2EDecision, E2EEnv, EnvConfig

SMOKE = 162_000_000


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


def fake(cell: str, k: int = 0) -> Capture:
    return Capture(cell=cell, snap=None, move={"action": "move"}, seed=k, steps=k, rows=0, idx=())


# ── the schedule ─────────────────────────────────────────────────────────────


def test_shares_are_linear_over_0_to_2b_then_flat() -> None:
    assert P.practice_share(0) == pytest.approx(0.40)
    assert P.practice_share(1e9) == pytest.approx(0.325)
    assert P.practice_share(2e9) == pytest.approx(0.25)
    assert P.practice_share(9e9) == pytest.approx(0.25)
    assert P.boss_share(0) == pytest.approx(0.40)
    assert P.boss_share(1e9) == pytest.approx(0.575)
    assert P.boss_share(2e9) == pytest.approx(0.75)
    assert P.boss_share(-5) == pytest.approx(0.40)


@pytest.mark.parametrize("t", [0, 3e8, 1e9, 2e9])
def test_cell_probabilities_follow_the_mix(t: float) -> None:
    p = P.cell_probs(t)
    assert set(p) == set(P.CELLS)
    assert sum(p.values()) == pytest.approx(1.0)
    b = P.boss_share(t)
    assert p["BOSS_ACT3_DOUBLE"] == pytest.approx(0.65 * b)
    assert p["BOSS_ACT2"] == pytest.approx(0.20 * b)
    assert p["BOSS_ACT1"] == pytest.approx(0.15 * b)
    assert p["ELITE_ACT1"] == pytest.approx(0.20 * (1 - b))
    assert p["ELITE_ACT2"] == pytest.approx(0.40 * (1 - b))
    assert p["ELITE_ACT3"] == pytest.approx(0.40 * (1 - b))


def test_practice_slots_round_the_share() -> None:
    assert P.practice_slots(8, 0) == 3  # 3.2
    assert P.practice_slots(8, 2e9) == 2
    assert P.practice_slots(64, 0) == 26  # 25.6
    assert [P.is_practice_slot(s, 8, 0) for s in range(8)] == [True] * 3 + [False] * 5


def test_draws_match_the_cell_probabilities() -> None:
    q = P.PracticeQueues(worker=3, seed=11)
    n = 40_000
    got = collections.Counter(q.draw_cell(0) for _ in range(n))
    for cell, p in P.cell_probs(0).items():
        assert abs(got[cell] / n - p) < 5 * math.sqrt(p * (1 - p) / n), cell


def test_reseeds_are_in_range_and_deterministic_per_worker() -> None:
    def stream(worker: int, seed: int) -> list[int]:
        q = P.PracticeQueues(worker=worker, seed=seed)
        return [q.reseed() for _ in range(200)]

    a = stream(2, 7)
    assert a == stream(2, 7)
    assert a != stream(3, 7) and a != stream(2, 8)
    assert all(P.RESEED_LO <= s <= P.RESEED_HI for s in a)
    assert (P.RESEED_LO, P.RESEED_HI) == (600_000_000, 639_999_999)


def test_practice_uids_never_collide_with_seeds_or_each_other() -> None:
    uids = [P.practice_uid(w, 7, n) for w in range(4) for n in range(500)]
    assert len(set(uids)) == len(uids)
    assert all(u >> 62 == 1 and u < 2**63 for u in uids)  # bit 62: above any run seed
    assert P.practice_uid(1, 7, 3) == P.practice_uid(1, 7, 3)
    starts = [P.start_uid(fake("ELITE_ACT1", k), r) for k in range(20) for r in range(50)]
    assert len(set(starts)) == len(starts) and not set(starts) & set(uids)
    assert all(u >> 62 == 1 and u < 2**63 for u in starts)
    assert P.segment_for(fake("BOSS_ACT1", 3), 9).uid == P.start_uid(fake("BOSS_ACT1", 3), 9)


def test_cells_are_the_schemas() -> None:
    assert P.CELLS == C.CELLS and P.DOUBLE_BOSS in C.CELLS
    assert set(P.BOSS_MIX) | set(P.ELITE_MIX) == set(C.CELLS)


# ── the queues ───────────────────────────────────────────────────────────────


def test_a_queue_is_fifo_with_32_slots() -> None:
    q = P.PracticeQueues(worker=0)
    for k in range(40):
        q.add(fake("ELITE_ACT1", k), now=k)
    assert q.size("ELITE_ACT1") == 32
    assert [e.cap.seed for e in q.q["ELITE_ACT1"]] == list(range(8, 40))
    assert q.stats["evict_full/ELITE_ACT1"] == 8


def test_entries_expire_50m_decisions_after_capture() -> None:
    q = P.PracticeQueues(worker=0)
    q.add(fake("BOSS_ACT1", 0), now=0)
    q.add(fake("BOSS_ACT1", 1), now=10_000_000)
    q.expire(50_000_000)
    assert q.size("BOSS_ACT1") == 2  # an age of exactly 50M is still live
    q.expire(50_000_001)
    assert q.size("BOSS_ACT1") == 1
    q.expire(60_000_001)
    assert q.size("BOSS_ACT1") == 0
    assert q.stats["expired/BOSS_ACT1"] == 2


def test_an_entry_serves_at_most_four_starts() -> None:
    q = P.PracticeQueues(worker=0)
    q.add(fake("BOSS_ACT2"), now=0)
    uses = [q.take("BOSS_ACT2").uses for _ in range(4)]
    assert uses == [1, 2, 3, 4]
    assert q.size("BOSS_ACT2") == 0 and q.stats["used_up/BOSS_ACT2"] == 1


def test_activation_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []
    monkeypatch.setattr(
        P, "start_env", lambda cap, s, cfg=None, uid=None: built.append((cap.cell, s)) or "ENV"
    )
    q = P.PracticeQueues(worker=0, seed=1)
    monkeypatch.setattr(q, "draw_cell", lambda t: "BOSS_ACT3_DOUBLE")
    for k in range(7):
        q.add(fake("BOSS_ACT3_DOUBLE", k), now=0)
    st = q.start(now=0, t=0)  # 7 < 8 entries, and the fallbacks are empty
    assert st.env is None and st.cell is None and st.wanted == "BOSS_ACT3_DOUBLE"
    assert q.stats["fallback/BOSS_ACT3_DOUBLE->run"] == 1
    for k in range(8):
        q.add(fake("BOSS_ACT1", k), now=0)
    st = q.start(now=0, t=0)  # act 2 is empty too: act 1
    assert (st.env, st.cell) == ("ENV", "BOSS_ACT1")
    assert q.stats["fallback/BOSS_ACT3_DOUBLE->BOSS_ACT1"] == 1
    q.add(fake("BOSS_ACT3_DOUBLE", 99), now=0)  # 8 entries: active
    st = q.start(now=0, t=0)
    assert st.cell == "BOSS_ACT3_DOUBLE" and P.RESEED_LO <= st.reseed <= P.RESEED_HI
    assert [c for c, _ in built] == ["BOSS_ACT1", "BOSS_ACT3_DOUBLE"]
    assert q.fallback_rates()["BOSS_ACT3_DOUBLE"] == pytest.approx(2 / 3)


def test_a_bad_start_tries_another_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def start_env(cap, s, cfg=None, uid=None):
        calls.append(cap.seed)
        return None if len(calls) == 1 else "ENV"

    monkeypatch.setattr(P, "start_env", start_env)
    q = P.PracticeQueues(worker=0, seed=2)
    monkeypatch.setattr(q, "draw_cell", lambda t: "ELITE_ACT2")
    for k in range(8):
        q.add(fake("ELITE_ACT2", k), now=0)
    st = q.start(now=0, t=0)
    assert st.env == "ENV" and len(calls) == 2
    assert q.stats["bad_start/ELITE_ACT2"] == 1


def test_practice_off_changes_nothing_else() -> None:
    q = P.PracticeQueues(worker=0, enabled=False)
    assert q.sink(lambda: 0) is None  # no capture, no clones
    q.add(fake("BOSS_ACT1"), now=0)
    assert q.size("BOSS_ACT1") == 0
    assert q.start(now=0, t=0).env is None
    assert not q.wants_practice(0, 8, 0)
    assert P.PracticeQueues(worker=0).wants_practice(0, 8, 0)


def test_a_segment_source_files_captures_at_the_live_clock() -> None:
    """Review finding ENVR-7: a capture is filed at the actor's clock when it happens,
    not at the decision clock of its run's start."""
    box = {"t": 5}
    q = P.PracticeQueues(worker=0, seed=1)
    src = P.SegmentSource(q, iter([SMOKE]), n_slots=1, clock=lambda: box["t"])
    env = src.next(0, box["t"])
    box["t"] = 777  # the run plays on; the actor's clock moves
    env.cfg.on_capture(fake("BOSS_ACT1"))
    assert q.q["BOSS_ACT1"][-1].born == 777
    old = P.SegmentSource(q, iter([SMOKE + 1]), n_slots=1)  # no clock: the run's start
    old.next(0, 9).cfg.on_capture(fake("BOSS_ACT1"))
    assert q.q["BOSS_ACT1"][-1].born == 9


def test_a_segment_source_feeds_and_draws_practice(won_run_captures) -> None:
    caps, _ = won_run_captures
    q = P.PracticeQueues(worker=1, seed=3, min_active=1)
    src = P.SegmentSource(q, seeds=iter(range(SMOKE + 30, SMOKE + 40)), n_slots=8)
    env = src.next(slot=0, t=0)  # a practice slot, but no cell is active yet
    assert env.segment.kind == "run" and src.stats["practice/run_fallback"] == 1
    assert env.cfg.on_capture is not None  # full runs feed the queues
    for c in caps:
        q.add(c, now=0)
    env = src.next(slot=0, t=0)
    assert env.segment.kind == "practice" and env.cfg.on_capture is None
    assert env.segment.uid is not None and env.segment.uid >> 62 == 1
    assert src.next(slot=7, t=0).segment.kind == "run"  # slot 7 of 8 is never practice
    off = P.SegmentSource(P.PracticeQueues(worker=1, enabled=False), iter([SMOKE]), n_slots=8)
    env = off.next(slot=0, t=0)
    assert env.segment.kind == "run" and env.cfg.on_capture is None


# ── live: captures, starts, segment ends ─────────────────────────────────────


def _actor(seed: int):
    rng = random.Random(seed)

    def act(d: E2EDecision) -> int:
        if d.kind == "combat":
            ends = {c for c, a in zip(d.classes, d.actions) if a["action"] == "end_turn"}
            others = [c for c in range(d.n_classes) if c not in ends]
            if others and rng.random() < 0.85:
                return rng.choice(others)
        return rng.randrange(d.n_classes)

    return act


def _walk(env: E2EEnv, act) -> tuple[list[E2EDecision], object]:
    rows = []
    d = env.reset()
    while isinstance(d, E2EDecision) and not d.bootstrap:
        rows.append(d)
        d = env.step(act(d))
    if isinstance(d, E2EDecision):
        rows.append(d)
    return rows, env.terminal


@pytest.fixture(scope="module")
def won_run_captures() -> tuple[list[Capture], object]:
    """The captures of the won HP-pinned smoke run (one per elite and boss room)."""
    caps: list[Capture] = []
    rm = factory.new_run(SMOKE)
    rm.set_player(hp=100_000, max_hp=100_000)
    env = E2EEnv(SMOKE, rm=rm, cfg=EnvConfig(on_capture=caps.append))
    _, t = _walk(env, _actor(SMOKE))
    assert t.won
    return caps, t


def test_captures_cover_every_cell_and_skip_boss_2(won_run_captures) -> None:
    caps, t = won_run_captures
    by = collections.Counter(c.cell for c in caps)
    assert set(by) == set(P.CELLS)
    assert by["BOSS_ACT1"] == by["BOSS_ACT2"] == by["BOSS_ACT3_DOUBLE"] == 1
    assert t.counts["capture/boss2_skipped"] == 1 and t.reached_boss2
    for c in caps:
        assert c.move["action"] == "move"
        assert c.snap.phase() != "combat"  # taken before the Move
        factory.check_run(c.snap)


@pytest.mark.parametrize("cell", P.CELLS)
def test_t13_reseeds_deal_distinct_hands(won_run_captures, cell: str) -> None:
    cap = next(c for c in won_run_captures[0] if c.cell == cell)
    hands, worlds = set(), set()
    for s in range(P.RESEED_LO, P.RESEED_LO + 100):
        env = P.start_env(cap, s)
        assert env is not None and env.rm.current_room()["room_type"] == P.ROOM_OF[cell]
        hands.add(P.opening_key(env))
        worlds.add(env.rm.combat_state().ordered_hash())
    assert len(hands) >= 95 and len(worlds) >= 95
    a, b = P.start_env(cap, P.RESEED_LO + 5), P.start_env(cap, P.RESEED_LO + 5)
    assert P.opening_key(a) == P.opening_key(b)
    assert a.rm.combat_state().ordered_hash() == b.rm.combat_state().ordered_hash()
    assert cap.snap.phase() != "combat"  # starts never touch the capture


@pytest.mark.parametrize("cell", ["ELITE_ACT1", "BOSS_ACT1", "BOSS_ACT2"])
def test_a_segment_ends_at_its_first_ooc_decision(won_run_captures, cell: str) -> None:
    cap = next(c for c in won_run_captures[0] if c.cell == cell)
    env = P.start_env(cap, P.RESEED_LO + 1)
    rows, t = _walk(env, _actor(1))
    assert t.outcome == "bootstrap" and env.done
    assert len(t.fingerprint) == 24  # a real digest of the bootstrap state
    last = rows[-1]
    assert last.bootstrap and last.kind == "ooc" and not any(d.bootstrap for d in rows[:-1])
    assert all(d.kind == "combat" for d in rows[:-1])
    assert t.rewards[-1] == pytest.approx(0.0, abs=0.0)  # the bootstrap row earns nothing
    assert sum(t.rewards) == pytest.approx(t.total_reward)
    assert t.unassigned_reward == pytest.approx(0.0, abs=0.0)
    assert t.bosses == (1 if cell.startswith("BOSS") else 0)
    assert t.total_reward == pytest.approx(0.05 * t.bosses)
    with pytest.raises(RuntimeError, match="bootstrap"):
        env.step(0)
    b = E.segment_rows(rows, t)  # a practice end: value only, done (SCHEMA.md §9.1)
    assert b["bootstrap"].tolist() == [False] * (len(rows) - 1) + [True]
    assert b["done"].tolist() == [False] * (len(rows) - 1) + [True]
    assert not b["reward"][-1]  # reward 0
    assert (b["cell"] == C.CELLS.index(cell)).all()
    for d in rows:  # every row, the bootstrap row included, is a valid env decision
        C.validate_decision(d.decision)
        assert d.decision.cell == C.CELLS.index(cell) and d.decision.run_uid == d.segment.uid


def test_a_start_keeps_every_setting_but_capture(won_run_captures) -> None:
    cap = won_run_captures[0][0]
    tok = object()
    cfg = EnvConfig(max_fight_turns=7, f9_p=0.5, tokenizer=tok, on_capture=print)
    env = P.start_env(cap, P.RESEED_LO + 4, cfg=cfg)
    assert env is not None and env.cfg.on_capture is None
    assert env.cfg.tokenizer is tok and env.cfg.max_fight_turns == 7
    assert env.f9.sample_rate == pytest.approx(0.5)


def test_a_double_boss_segment_plays_to_the_outcome(won_run_captures) -> None:
    cap = next(c for c in won_run_captures[0] if c.cell == "BOSS_ACT3_DOUBLE")
    env = P.start_env(cap, P.RESEED_LO + 2)
    rows, t = _walk(env, _actor(2))
    assert t.outcome in ("won", "lost", "capped")
    assert not any(d.bootstrap for d in rows)
    assert env.segment.end == "terminal" and env.segment.kind == "practice"
    assert t.counts["fights/boss"] >= 1
    if t.won:
        assert t.reached_boss2 and t.bosses == 1
        assert t.total_reward == pytest.approx(
            1.0 + 0.05 + 0.002 * (t.floor - cap.snap.total_floor() - 1)
        )
