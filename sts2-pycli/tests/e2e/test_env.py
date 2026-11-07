"""The campaign env (`ppo.e2e.env`, WP2): decision typing and dedupe on the schema's
classifiers (T6), reward accounting (T4), caps (T5), auto-advance, units, F9 wiring, the
build guard, the policy and PRIV sources, and the env's row-batch fields.

Live tests play real v0.111.0 A10 runs with a seeded random actor. "Pinned" runs set the
player's HP to 100,000 so that random play reaches acts 2 and 3.
"""

from __future__ import annotations

import collections
import json
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import sts2_client
import sts2_rs
from ppo.combat_ctx.snapshot import choose_outcome_key, obs_actions, play_edge_key
from ppo.e2e import contracts as C
from ppo.e2e import env as E
from ppo.e2e import factory
from ppo.e2e import preflight as PF
from ppo.e2e.env import E2EDecision, E2EEnv, EnvConfig
from ppo.e2e.invariants import Violation

SMOKE = 162_000_000  # plan §3.1: smoke seeds 162,000,000-162,000,999


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


# ── helpers ──────────────────────────────────────────────────────────────────


def random_actor(seed: int, play_bias: float = 0.85):
    """A seeded random actor that prefers playing to ending the turn."""
    rng = random.Random(seed)

    def act(d: E2EDecision) -> int:
        if d.kind == "combat":
            ends = {c for c, a in zip(d.classes, d.actions) if a["action"] == "end_turn"}
            others = [c for c in range(d.n_classes) if c not in ends]
            if others and rng.random() < play_bias:
                return rng.choice(others)
        return rng.randrange(d.n_classes)

    return act


def end_turn_actor(d: E2EDecision) -> int:
    """Always end the turn (a deadlock against a pinned player), else option 0."""
    for c, a in zip(d.classes, d.actions):
        if a.get("action") == "end_turn":
            return c
    return 0


def play(env: E2EEnv, act) -> tuple[list[E2EDecision], Any]:
    rows = []
    d = env.reset()
    while isinstance(d, E2EDecision) and not d.bootstrap:
        rows.append(d)
        d = env.step(act(d))
    if isinstance(d, E2EDecision):
        rows.append(d)  # the bootstrap row
    return rows, env.terminal


def pinned_env(seed: int, cfg: EnvConfig | None = None) -> E2EEnv:
    rm = factory.new_run(seed)
    rm.set_player(hp=100_000, max_hp=100_000)
    return E2EEnv(seed, rm=rm, cfg=cfg)


def scenario(deck: list[str], encounter: str = "NIBBITS_WEAK") -> Any:
    return factory.scenario_state(
        seed=5,
        character="Ironclad",
        deck=deck,
        hp=80,
        max_hp=80,
        relics=[],
        encounter=encounter,
        ascension=10,
    )


def plays_of(cs: Any, card_id: str) -> list[int]:
    hand = sts2_rs.state_to_dict(cs)["deck"]["hand"]
    return [
        i
        for i, a in enumerate(cs.legal_actions())
        if a["action"] == "play" and hand[a["card_index"]]["id"] == card_id
    ]


# ── decision types (no engine) ───────────────────────────────────────────────


class _Rm:
    def __init__(self, event: str | None = None, grants: int = 0) -> None:
        self.event = event
        self.grants = grants

    def current_event_name(self) -> str | None:
        return self.event

    def card_grant_options(self) -> list[str]:
        return ["X"] * self.grants


@pytest.mark.parametrize(
    ("actions", "rm", "obs", "want"),
    [
        ([{"action": "move", "col": 0}, {"action": "move", "col": 1}], _Rm(), {}, "MAP"),
        ([{"action": "event_choice", "idx": i} for i in range(2)], _Rm("Neow"), {}, "NEOW"),
        ([{"action": "event_choice", "idx": i} for i in range(2)], _Rm("BigFish"), {}, "EVENT"),
        (
            [{"action": "pick_card_reward", "idx": 0}, {"action": "pick_card_reward", "idx": None}],
            _Rm(),
            {},
            "CARD_REWARD",
        ),
        (
            [{"action": "pick_card_reward", "idx": i} for i in range(2)],
            _Rm(),
            {"card_reward_bundles": [["A"], ["B"]]},
            "CARD_BUNDLE",
        ),
        (
            [{"action": "grant_deck_card", "idx": 0}, {"action": "grant_deck_card", "idx": 1}],
            _Rm(grants=1),
            {},
            "CARD_GRANT",
        ),
        ([{"action": "rest_option", "idx": 0}, {"action": "leave_shop"}], _Rm(), {}, "REST"),
        ([{"action": "shop_buy_card", "idx": 0}, {"action": "leave_shop"}], _Rm(), {}, "SHOP"),
        (
            [{"action": "potion_swap", "discard_slot": 0}, {"action": "potion_swap"}],
            _Rm(),
            {},
            "POTION_SWAP",
        ),
    ],
)
def test_ooc_types_come_from_the_schema(actions, rm, obs, want) -> None:
    dtype, kinds = E.classify_ooc(actions, rm, obs)
    assert E.dtype_name(dtype) == want
    assert len(kinds) == len(actions) and set(kinds) <= C.LIVE_ACTION_KINDS


def test_an_unknown_kind_fails_loudly() -> None:
    with pytest.raises(C.SchemaError):
        E.classify_ooc([{"action": "teleport"}, {"action": "teleport"}], _Rm(), {})


def test_a_deck_operation_preempting_a_fight_is_not_combat() -> None:
    # Classification is by kind, never phase(): a pending removal shows phase map_choice.
    assert not E.is_combat([{"action": "remove_deck_card"}, {"action": "play_card"}])
    assert E.is_combat([{"action": "play_card"}, {"action": "end_turn"}])
    assert not E.is_combat([])


def test_classes_from_keys_is_dense_in_first_occurrence_order() -> None:
    cls, members = E.classes_from_keys(["a", "b", "a", "c", "b"])
    assert cls == [0, 1, 0, 2, 1]
    assert members == [[0, 2], [1, 4], [3]]
    assert E.classes_from_keys([]) == ([], [])


def test_reward_and_cap_constants_match_prereg_and_the_schema() -> None:
    assert (E.R_WIN, E.R_BOSS, E.R_FLOOR) == (1.0, 0.05, 0.002)
    assert (E.MAX_FIGHT_TURNS, E.MAX_FIGHT_DECISIONS, E.MAX_RUN_DECISIONS) == (100, 1000, 6000)
    assert E.MAX_FIGHT_TURNS == C.MAX_FIGHT_TURNS and E.MAX_RUN_DECISIONS == C.MAX_RUN_DECISIONS


# ── T6: dedupe ───────────────────────────────────────────────────────────────


def test_t6_plain_strikes_merge() -> None:
    cs = scenario(["STRIKE_IRONCLAD"] * 5)
    cc = E.merged_classes(cs)
    cls, members = cc.classes, cc.members
    strikes = plays_of(cs, "STRIKE_IRONCLAD")
    assert len(strikes) == 5
    assert len({cls[i] for i in strikes}) == 1  # five plays, one class
    assert len(members) == 2  # the Strike, and end_turn
    assert members[cls[strikes[0]]] == strikes  # members in action order
    assert E.dtype_name(cc.dtype) == "COMBAT_TURN"
    assert {C.ACTION_KINDS[k].name for k in cc.kinds} == {"PLAY_TARGET", "END_TURN"}


def test_t6_an_enchanted_strike_stays_apart_from_plain_ones() -> None:
    d = sts2_rs.state_to_dict(scenario(["STRIKE_IRONCLAD"] * 5))
    d["deck"]["hand"][0]["enchantment"] = "SHARP"
    d["deck"]["hand"][0]["enchantment_amount"] = 2
    cs = sts2_rs.state_from_dict(json.loads(json.dumps(d)))
    hand = sts2_rs.state_to_dict(cs)["deck"]["hand"]
    assert hand[0].get("enchantment")  # the engine kept it
    legal = cs.legal_actions()
    cc = E.merged_classes(cs)
    cls, members = cc.classes, cc.members
    strikes = plays_of(cs, "STRIKE_IRONCLAD")
    sharp = [i for i in strikes if legal[i]["card_index"] == 0]
    plain = [i for i in strikes if legal[i]["card_index"] != 0]
    assert len(sharp) == 1 and len(plain) == 4
    assert len({cls[i] for i in plain}) == 1
    assert cls[sharp[0]] not in {cls[i] for i in plain}
    assert len(members) == 3
    # The search's edge key would have merged them (snapshot.py:487-503).
    keys = {play_edge_key(hand, legal[i]) for i in strikes}
    assert len(keys) == 1


def test_t6_a_played_rampage_stays_apart_from_a_fresh_one() -> None:
    cs = scenario(["RAMPAGE", "RAMPAGE", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"])
    cs.apply(cs.legal_actions()[plays_of(cs, "RAMPAGE")[0]])
    cs.apply({"action": "end_turn"})
    for _ in range(8):  # through the enemy turn to the next player decision
        if plays_of(cs, "RAMPAGE"):
            break
        cs.apply(cs.legal_actions()[0])
    rams = plays_of(cs, "RAMPAGE")
    assert len(rams) == 2, "both Rampages back in hand on turn 2"
    raw = sts2_rs.state_to_dict(cs)["deck"]["hand"]
    legal = cs.legal_actions()
    played = [
        i
        for i in rams
        if (raw[legal[i]["card_index"]].get("combat_vars") or {}).get("rampage_damage")
    ]
    assert len(played) == 1, "exactly one Rampage carries its per-copy bonus"
    cls = E.merged_classes(cs).classes
    assert cls[rams[0]] != cls[rams[1]]
    assert play_edge_key(raw, legal[rams[0]]) == play_edge_key(raw, legal[rams[1]])
    # The deviation: the schema's own key merges them on W1 (both base_damage 10).
    bv = cs.belief_view(0)
    d = sts2_rs.state_to_dict(bv)
    acts = obs_actions(legal, cs.pending_choice(), bv.pending_choice())
    schema = C.combat_class_keys(acts, d["deck"]["hand"], d["player"]["potions"])
    assert schema[rams[0]] == schema[rams[1]]


def test_t6_identical_potions_merge_by_id_and_target() -> None:
    d = sts2_rs.state_to_dict(scenario(["STRIKE_IRONCLAD"] * 5))
    d["player"]["potions"] = ["FIRE_POTION", "FIRE_POTION", "BLOCK_POTION"]
    d["player"]["potion_slots"] = 3
    cs = sts2_rs.state_from_dict(d)
    legal = cs.legal_actions()
    cc = E.merged_classes(cs)
    cls, members = cc.classes, cc.members
    drinks = {a["slot"]: i for i, a in enumerate(legal) if a["action"] == "drink_potion"}
    assert cls[drinks[0]] == cls[drinks[1]]  # two Fire Potions, one class
    assert cls[drinks[2]] != cls[drinks[0]]
    assert len(members) == 4  # Strike, Fire, Block, end_turn


def test_t6_choices_merge_only_on_the_same_visible_outcome() -> None:
    cs = scenario(
        ["ARMAMENTS", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"]
    )
    cs.apply(cs.legal_actions()[plays_of(cs, "ARMAMENTS")[0]])
    assert cs.pending_choice() is not None
    legal = cs.legal_actions()
    cc = E.merged_classes(cs)
    assert all(a["action"] == "choose" for a in legal)
    assert E.dtype_name(cc.dtype) == "COMBAT_CHOICE"
    # Upgrading the first or the second Strike leaves a different hand order, so the
    # honest snapshot differs: the schema's choose key keeps them apart.
    assert len(cc.members) == len(legal)
    # The key carries the selected card's content, so a choice that leaves the visible
    # outcome unchanged still never merges different cards (SCHEMA.md §5).
    bv = cs.belief_view(0)
    d = sts2_rs.state_to_dict(bv)

    def outcome(i: int) -> str:
        c = bv.clone()
        c.apply(cc.obs_actions[i])
        return choose_outcome_key(c)

    keys = C.combat_class_keys(
        cc.obs_actions,
        d["deck"]["hand"],
        [],
        choice=bv.pending_choice(),
        outcome=lambda i: "same",
        piles=d["deck"],
    )
    strikes = [i for i, k in enumerate(keys) if "STRIKE" in k]
    defends = [i for i, k in enumerate(keys) if "DEFEND" in k]
    assert len({keys[i] for i in strikes}) == 1 and len({keys[i] for i in defends}) == 1
    assert keys[strikes[0]] != keys[defends[0]]
    assert outcome(strikes[0]) != outcome(strikes[1])


def _schema_classes(bv: Any, d_bv: dict[str, Any], acts: list[dict[str, Any]]):
    """The schema's classes with every choose outcome computed (no shortcut)."""

    def outcome(i: int) -> str:
        c = bv.clone()
        c.apply(acts[i])
        return choose_outcome_key(c)

    keys = C.combat_class_keys(
        acts,
        d_bv["deck"]["hand"],
        d_bv["player"]["potions"],
        choice=bv.pending_choice(),
        outcome=outcome,
        piles=d_bv["deck"],
    )
    return E.classes_from_keys(keys)


def test_only_a_shared_selection_needs_its_outcome() -> None:
    cs = scenario(["ARMAMENTS", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"])
    cs.apply(cs.legal_actions()[plays_of(cs, "ARMAMENTS")[0]])
    bv, d, _ = E.combat_view(cs, "monster")
    acts = obs_actions(cs.legal_actions(), cs.pending_choice(), bv.pending_choice())
    hand = d["deck"]["hand"]
    picked = [hand[int(a["selection"][0])]["id"] for a in acts]
    shared = E._shared_selections(bv.pending_choice(), acts, d["deck"])
    assert {picked[i] for i in shared} == {"STRIKE_IRONCLAD"} and len(shared) == 2
    # The two Strikes still split on their outcomes, exactly as the schema splits them.
    assert E.classes_from_keys(E.combat_keys(bv, d, acts)) == _schema_classes(bv, d, acts)
    assert len(E.merged_classes(cs).members) == len(acts)


def test_choice_rows_carry_the_schemas_classes() -> None:
    n = 0
    for k in range(3):
        rows, _ = play(pinned_env(SMOKE + 40 + k), random_actor(40 + k))
        for d in rows:
            if d.is_combat and d.dtype == C.DT["COMBAT_CHOICE"]:
                n += 1
                assert (d.classes, d.members) == _schema_classes(d.bv, d.d_bv, d.obs_actions)
    assert n >= 20, n


def _stream(seed: int, pinned: bool) -> tuple[list[tuple], Any]:
    env = pinned_env(seed) if pinned else E2EEnv(seed)
    rows, t = play(env, random_actor(seed))
    return [(d.kind, d.dtype, d.classes, d.members, d.snap, d.reward, d.turn) for d in rows], t


@pytest.mark.parametrize(("seed", "pinned"), [(SMOKE + 44, False), (SMOKE + 45, True)])
def test_the_fast_paths_change_no_row(monkeypatch: pytest.MonkeyPatch, seed: int, pinned: bool):
    fast, tf = _stream(seed, pinned)
    monkeypatch.setattr(E, "FAST_PATHS", False)
    plain, tp = _stream(seed, pinned)
    assert len(fast) == len(plain) and fast == plain
    assert tf.fingerprint == tp.fingerprint and tf.rewards == tp.rewards
    assert tf.counts == tp.counts and tf.log_idx == tp.log_idx


def test_the_earlier_helper_forms_still_work() -> None:
    cs = scenario(["STRIKE_IRONCLAD"] * 5)
    bv, d, snap = E.combat_view(cs, "monster")
    acts = obs_actions(cs.legal_actions(), cs.pending_choice(), bv.pending_choice())
    assert E.combat_keys(bv, d, snap, acts) == E.combat_keys(bv, d, acts)
    assert E.decision_type([{"action": "play_card"}], _Rm()) == C.DT["COMBAT_TURN"]
    assert E.decision_type([{"action": "move"}] * 2, _Rm()) == C.DT["MAP"]
    # The one-argument sources (the tokenizer's `RecordedDecision.policy()` / `.priv()`).
    rm = _Rm()
    one = SimpleNamespace(bv=bv, obs_actions=acts, rm=rm, cs=cs)
    for pol in (E.PolicySource(one), E.PolicySource(bv, acts, rm)):
        assert pol.belief() is bv and pol.legal() is acts
        assert object.__getattribute__(pol.run(), "_rm") is rm
    for priv in (E.PrivSource(one), E.PrivSource(cs, rm)):
        assert priv.true_state() is cs and priv._rm is rm


def test_face_variables_only_split_schema_classes() -> None:
    # The refinement is finer than the schema's key: equal env keys imply equal schema keys.
    cs = scenario(["RAMPAGE", "RAMPAGE", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD"])
    bv = cs.belief_view(0)
    d = sts2_rs.state_to_dict(bv)
    acts = obs_actions(cs.legal_actions(), cs.pending_choice(), bv.pending_choice())
    ours = E.combat_keys(bv, d, acts)
    schema = C.combat_class_keys(acts, d["deck"]["hand"], d["player"]["potions"])
    assert ours == schema  # no per-copy variables yet: identical keys
    pairs = {(o, k) for o, k in zip(ours, schema)}
    assert len({o for o, _ in pairs}) == len(pairs)


# ── T4: reward accounting ────────────────────────────────────────────────────


def _check_stream(rows: list[E2EDecision], t: Any) -> None:
    """Each row's `reward` is the previous row's final reward; the terminal's is the last."""
    assert len(t.rewards) == t.rows == len(rows)
    assert [d.row for d in rows] == list(range(len(rows)))
    assert [d.reward for d in rows[1:]] + [t.reward] == t.rewards
    assert sum(t.rewards) == pytest.approx(t.total_reward, abs=1e-12)


@pytest.mark.parametrize("seed", [SMOKE + 1, SMOKE + 5])
def test_t4_rewards_over_stored_rows_equal_the_env_total(seed: int) -> None:
    env = E2EEnv(seed)
    floor0 = env.max_floor
    rows, t = play(env, random_actor(seed))
    assert t.outcome == "lost" and not t.won and t.cap is None
    _check_stream(rows, t)
    assert t.unassigned_reward == pytest.approx(0.0, abs=0.0)
    want = E.R_FLOOR * (t.floor - floor0) + E.R_BOSS * t.bosses
    assert t.total_reward == pytest.approx(want, abs=1e-12)


def test_t4_a_won_run_earns_floors_bosses_and_the_win() -> None:
    env = pinned_env(SMOKE)
    rows, t = play(env, random_actor(SMOKE))
    assert t.outcome == "won" and t.won
    # Act 1, act 2 and act-3 boss 1 pay 0.05; boss 2's win pays only the win.
    assert t.bosses == 3 and t.reached_boss2
    assert t.floor == 46
    assert t.total_reward == pytest.approx(1.0 + 3 * 0.05 + 46 * 0.002, abs=1e-12)
    _check_stream(rows, t)
    assert t.unassigned_reward == pytest.approx(0.0, abs=0.0)
    assert t.counts["round_reset"] == 0  # the act-3 bosses are two combat phases on W1


# ── T5 and the other caps ────────────────────────────────────────────────────


def test_t5_a_deadlock_ends_at_turn_100_as_a_loss() -> None:
    env = pinned_env(SMOKE + 2)
    rows, t = play(env, end_turn_actor)
    assert t.outcome == "capped" and t.cap == "turns" and not t.won
    combat = [d for d in rows if d.kind == "combat"]
    last = combat[-1]
    assert last.turn == E.MAX_FIGHT_TURNS  # 100 turns played, the 101st never starts
    assert t.counts["cap/turns"] == 1 and t.counts["outcome/capped"] == 1
    assert t.bosses == 0
    _check_stream(rows, t)
    # A cap pays nothing: only the floors reached before it.
    assert t.total_reward == pytest.approx(E.R_FLOOR * t.floor, abs=1e-12)


def _returns(env: E2EEnv) -> tuple[dict[str, np.ndarray], np.ndarray, Any]:
    """The segment's returns as the learner computes them, on random values."""
    from ppo.e2e import algo

    b = env.segment_rows()
    tab = {k: b[k] for k in ("run_uid", "t", "is_combat", "dtype", "reward", "done", "bootstrap")}
    tab["actor"] = np.zeros(len(b["t"]), np.uint8)
    v = np.random.default_rng(0).uniform(0, 1, len(b["t"]))
    return tab, v, algo.compute_returns(tab, v, hold_iters=0)


def test_t5_a_capped_fight_starts_at_turn_1_and_ends_done_not_bootstrap() -> None:
    """T5's other halves: the capped fight plays turns 1..100 (a round counter from 0
    would pass the last-turn check at 101 turns), and the capped segment ends `done`,
    not `bootstrap`, so the returns score it from V(end) = 0, a loss."""
    env = pinned_env(SMOKE + 2)
    rows, t = play(env, end_turn_actor)
    assert t.outcome == "capped" and t.cap == "turns" and not t.won
    fight = [d for d in rows if d.kind == "combat" and d.fight == rows[-1].fight]
    assert sorted({d.turn for d in fight}) == list(range(1, E.MAX_FIGHT_TURNS + 1))
    tab, _v, ret = _returns(env)
    assert tab["done"][-1] and not tab["bootstrap"][-1]
    assert int(ret.succ[-1]) == -1 and ret.boot_w[-1] == pytest.approx(0.0, abs=0.0)
    assert ret.target[-1] == pytest.approx(float(tab["reward"][-1]), abs=1e-12)


# ── T2: the act-3 double boss on the live env ────────────────────────────────


def _check_double_boss(rows: list[E2EDecision], t: Any, env: E2EEnv) -> None:
    """Boss 1's last stored row is followed directly by boss 2's first (the algo
    test's T2 assumes it: the two fights are combat phases joined by one forced
    Move), the returns chain them, and the boss-1 bonus lands once, there."""
    from ppo.e2e import algo

    boss = [d for d in rows if d.kind == "combat" and d.room == "boss" and d.act == 3]
    fights = sorted({d.fight for d in boss})
    assert len(fights) == 2 and t.won and t.reached_boss2
    last1 = max(d.row for d in boss if d.fight == fights[0])
    first2 = min(d.row for d in boss if d.fight == fights[1])
    assert first2 == last1 + 1, "a stored row sits between boss 1 and boss 2"
    tab, v, ret = _returns(env)
    assert int(ret.succ[last1]) == first2 and tab["is_combat"][first2]
    delta = float(tab["reward"][last1]) + v[first2] - v[last1]
    assert ret.adv[last1] == pytest.approx(delta + algo.LAM_C * ret.adv[first2], abs=1e-12)
    # The boss-1 bonus and the Move's floor land once, on boss 1's last row.
    assert t.rewards[last1] == pytest.approx(E.R_BOSS + E.R_FLOOR, abs=1e-12)


def test_t2_live_boss_one_chains_into_boss_two_in_a_full_run() -> None:
    env = pinned_env(SMOKE)
    rows, t = play(env, random_actor(SMOKE))
    _check_double_boss(rows, t, env)
    assert t.bosses == 3


@pytest.mark.parametrize("r", [2, 3])
def test_t2_live_boss_one_chains_into_boss_two_in_practice(r: int) -> None:
    from ppo.e2e import practice as P

    caps: list[Any] = []
    play(pinned_env(SMOKE, cfg=EnvConfig(on_capture=caps.append)), random_actor(SMOKE))
    cap = next(c for c in caps if c.cell == "BOSS_ACT3_DOUBLE")
    env = P.start_env(cap, P.RESEED_LO + r)
    rows, t = play(env, random_actor(r))
    _check_double_boss(rows, t, env)
    assert t.bosses == 1


@pytest.mark.parametrize(
    ("cfg", "cap"),
    [
        (EnvConfig(max_fight_decisions=3), "fight_decisions"),
        (EnvConfig(max_run_decisions=5), "run_decisions"),
        (EnvConfig(max_run_steps=20), "steps"),
    ],
)
def test_caps_end_the_segment_as_a_loss(cfg: EnvConfig, cap: str) -> None:
    env = E2EEnv(SMOKE + 3, cfg=cfg)
    rows, t = play(env, random_actor(SMOKE + 3))
    assert t.outcome == "capped" and t.cap == cap and not t.won
    assert t.counts[f"cap/{cap}"] == 1
    _check_stream(rows, t)
    if cap == "run_decisions":
        assert t.rows == 5
    if cap == "fight_decisions":
        assert max(d.fight_row for d in rows if d.kind == "combat") == 2
    if cap == "steps":
        assert t.steps == 20


# ── auto-advance, units, determinism ─────────────────────────────────────────


def test_every_stored_row_has_two_classes_and_forced_steps_store_nothing() -> None:
    rows, t = play(E2EEnv(SMOKE + 4), random_actor(SMOKE + 4))
    assert rows and all(d.n_classes >= 2 for d in rows)
    for d in rows:
        assert sorted(i for m in d.members for i in m) == list(range(len(d.actions)))
        assert all(d.classes[i] == c for c, m in enumerate(d.members) for i in m)
        if d.kind == "combat":
            assert len(d.obs_actions) == len(d.actions)
            assert d.cs is not None and d.snap is not None and "ehist" in d.snap
        else:
            assert d.run_obs is not None and d.metrics is not None
    assert t.counts["forced/combat"] > 0
    assert t.counts["unknown"] == 0 and t.counts["fine_unknown"] == 0


def test_one_merged_class_auto_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every combat action in one class: no combat row, the fight still plays out.
    monkeypatch.setattr(E, "combat_keys", lambda bv, d_bv, acts: ["same"] * len(acts))
    rows, t = play(E2EEnv(SMOKE + 6, cfg=EnvConfig(max_run_steps=4000)), random_actor(1))
    assert t.outcome != "error", t.error
    assert t.counts["rows/COMBAT_TURN"] == t.counts["rows/COMBAT_CHOICE"] == 0
    assert t.counts["forced/combat_merged"] > 0
    assert t.counts["fights/monster"] >= 1
    assert all(d.kind == "ooc" for d in rows)


def test_units_mark_each_fight_and_act_once() -> None:
    rows, t = play(pinned_env(SMOKE), random_actor(SMOKE))  # the won run of T4
    seen: set = set()
    fights = []
    for d in rows:
        assert d.unit_start == (d.unit not in seen)
        seen.add(d.unit)
        if d.kind == "combat":
            assert d.unit == ("fight", d.fight)
            fights.append(d.fight)
        else:
            assert d.unit == ("act", d.act)
    assert fights == sorted(fights)
    assert t.won and {u[1] for u in seen if u[0] == "act"} == {1, 2, 3}
    # One unit per fight with a stored row (a fully forced fight stores none).
    assert len({u for u in seen if u[0] == "fight"}) <= sum(
        v for k, v in t.counts.items() if k.startswith("fights/")
    )


def test_the_same_seed_and_choices_replay_bit_for_bit() -> None:
    a = play(E2EEnv(SMOKE + 8), random_actor(8))[1]
    b = play(E2EEnv(SMOKE + 8), random_actor(8))[1]
    assert a.log_idx == b.log_idx and a.rewards == b.rewards
    assert a.fingerprint == b.fingerprint and a.outcome == b.outcome
    # Replaying the log by action index reaches the same state.
    env = E2EEnv(SMOKE + 8)
    d = env.reset()
    for ai in a.log_idx:
        d = env.step_action(ai)
    assert not isinstance(d, E2EDecision)
    assert env.terminal.fingerprint == a.fingerprint


# ── F9 wiring, the build guard ───────────────────────────────────────────────


def test_f9_sees_every_step_at_rate_one() -> None:
    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(f9_p=1.0))
    _, t = play(env, random_actor(9))
    assert t.counts["f9/checks"] == t.steps == env.f9.steps_seen
    assert t.counts["f9/violations"] == 0, env.f9_violations


def test_f9_tolerated_states_are_counted_not_violations() -> None:
    """Seed 162,000,201 (pinned, random actor) holds two Toric Toughness instances (a power
    C# instances). Engines before #1166 flag them and the checker tolerates them; from
    #1166 on the engine's own check allows them. Either way they are no violation, and
    every tolerated failure is counted."""
    seed = SMOKE + 201
    env = pinned_env(seed, EnvConfig(f9_p=1.0))
    _, t = play(env, random_actor(seed))
    assert t.counts["f9/violations"] == 0, env.f9_violations
    assert t.counts["f9/tolerated"] == sum(env.f9.tolerated.values())


def test_f9_violations_are_counted_and_kept() -> None:
    class Always:
        calls = 0

        def maybe_check(self, rm, cs=None, *, run_key=None):
            self.calls += 1
            return [Violation("run.hp_range", "stub")]

    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(f9=Always()))  # type: ignore[arg-type]
    _, t = play(env, random_actor(9))
    assert t.counts["f9/violations"] == t.counts["f9/run.hp_range"] == t.steps
    assert env.f9_violations[:1] == ["run.hp_range: stub"] and len(env.f9_violations) == 8


def test_known_f9_issues_are_tolerated_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from ppo.e2e import invariants as inv

    seen: list[tuple[str, ...]] = []

    class Spy(inv.InvariantChecker):
        def __init__(self, **kw: Any) -> None:
            seen.append(tuple(kw.get("tolerate_known", ())))
            super().__init__(**kw)

    monkeypatch.setattr(E, "InvariantChecker", Spy)
    E2EEnv(SMOKE + 9)
    E2EEnv(SMOKE + 9, cfg=EnvConfig(f9_tolerate=("crossbow_hand_cap",)))
    assert seen == [(), ("crossbow_hand_cap",)]
    with pytest.raises(ValueError, match="unknown known-issue keys"):
        E2EEnv(SMOKE + 9, cfg=EnvConfig(f9_tolerate=("no_such_issue",)))


def test_a_run_on_another_build_is_refused() -> None:
    rm = sts2_rs.RunManager(
        SMOKE,
        character="Ironclad",
        ascension_level=10,
        neow_start=True,
        act1="roll",
        game_version="v0.107.1",
        mode="agent",
        interactive_potion_swap=True,
    )
    with pytest.raises(sts2_client.GameVersionMismatch):
        E2EEnv(SMOKE, rm=rm)


def test_rule_side_potion_swaps_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sts2_rs, "potion_swap_is_rule_side", lambda: True)
    with pytest.raises(RuntimeError, match="potion swaps"):
        E2EEnv(SMOKE)


# ── OOC dedupe, sources, row fields ──────────────────────────────────────────


def test_ooc_rows_merge_by_the_schemas_keys() -> None:
    """Identical deck cards merge (smith, remove, upgrade); positions never do."""
    env = pinned_env(SMOKE)
    act = random_actor(SMOKE)
    ooc = []
    merged = 0
    d = env.reset()
    while isinstance(d, E2EDecision):
        if d.kind == E.OOC:  # re-derive at the decision, while the run is live there
            dtype, kinds = E.classify_ooc(d.actions, d.rm, d.run_obs)
            assert dtype == d.dtype and kinds == d.kinds
            keys = E.ooc_keys(d.actions, d.kinds, d.run_obs)
            assert d.classes == E.classes_from_keys(keys)[0]
            if d.dtype_name in ("MAP", "NEOW", "EVENT", "BOSS_RELIC"):
                assert d.n_classes == len(d.actions)  # positional: never merges
            merged += len(d.actions) - d.n_classes
            ooc.append(d)
        d = env.step(act(d))
    assert merged > 0
    smith = [d for d in ooc if d.dtype_name == "SMITH"]
    assert smith and any(d.n_classes < len(d.actions) for d in smith)


def test_the_facade_filters_event_previews() -> None:
    """`rm.event_option_infos` is a `POLICY_FILTERS` read: the facade serves
    `honest_event_options(...)`, never the raw previews (schema revision 2)."""
    raw = [
        {"label": "a", "gold_delta": 75, "hp_delta": -5, "max_hp_delta": 0, "relics_gained": 2},
        {"label": "b", "gold_delta": -30, "hp_delta": 0, "max_hp_delta": 0, "cards_lost": 1},
        {"label": "c", "gold_delta": None, "hp_delta": None, "max_hp_delta": None},
    ]
    run = E.RunFacade(SimpleNamespace(event_option_infos=lambda: raw, gold=99))
    got = run.event_option_infos()
    assert got == C.honest_event_options(raw) and got != raw
    assert got[0]["gold_delta"] == 0 and got[0]["relics_gained"] == 1  # masked relic option
    assert got[1]["gold_delta"] == -30 and got[1]["cards_lost"] == 1
    assert "gold_delta" not in got[2] and got[2]["label"] == "c"  # no preview
    assert run.gold == 99  # unfiltered reads pass through
    env = E2EEnv(SMOKE + 13)
    d = env.reset()
    assert d.dtype == C.DT["NEOW"]
    live = d.policy().run().event_option_infos()
    assert live == C.honest_event_options(d.rm.event_option_infos())


def test_the_policy_source_reads_only_policy_calls() -> None:
    env = E2EEnv(SMOKE + 13)
    d = env.reset()
    while d.kind != E.COMBAT:
        d = env.step(0)
    pol = d.policy()
    assert pol.belief() is d.bv and pol.legal() is d.obs_actions
    run = pol.run()
    assert run.gold == d.rm.gold and run.master_deck_detailed() == d.rm.master_deck_detailed()
    for name in ("combat_state", "generated_rooms", "take_action", "clone", "reseed_run"):
        with pytest.raises(AttributeError):
            getattr(run, name)
    with pytest.raises(AttributeError):
        run.gold = 5
    priv = d.priv()
    assert priv.true_state() is d.cs
    rooms = d.rm.generated_rooms()
    assert priv.act_rooms() == rooms[d.act - 1]
    assert priv.future() == list(rooms[d.act :])


def _priv_run(arrays: Any) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """The one PRIV_RUN token of a row's PRIV arrays (a `Tok` or `encode_priv`'s dict):
    `(cat, cont, pooled encounter id -> weight)`."""
    get = arrays.get if isinstance(arrays, dict) else lambda k: getattr(arrays, k)
    (j,) = np.flatnonzero(get("psec") == C.SEC["PRIV_RUN"])
    own = get("ppool_tok") == j
    pool = dict(zip(get("ppool_vid")[own].tolist(), get("ppool_w")[own].astype(float).tolist()))
    return get("pcat")[j], get("pcont")[j], pool


def test_priv_run_reads_the_acts_rooms_and_later_bosses_on_the_env_path() -> None:
    """A1: `PrivSource.act_rooms()` is this act's `generated_rooms()` record and
    `future()` the later acts' records, which `tokens._priv_run_static` reads. So on the
    env path an act-1 PRIV_RUN token carries the act's room counts, the act-2 boss and
    both act-3 bosses (A10), and the pooled encounter ids, through `tokens.encode` and
    through the `Tokenizer` the env drives alike. (The env once returned the records'
    keys, and every one of these read zero.)"""
    from ppo.e2e import tokens as K
    from ppo.e2e import vocab as V

    v = V.load()
    f16 = np.float16
    tok = K.Tokenizer()
    env = E2EEnv(SMOKE + 13, cfg=EnvConfig(tokenizer=tok))
    act = random_actor(13)
    d = env.reset()
    kinds: collections.Counter = collections.Counter()
    while isinstance(d, E2EDecision) and d.act == 1 and min(kinds.values(), default=0) < 3:
        g, later = d.rm.generated_rooms()[0], d.rm.generated_rooms()[1:]
        assert len(later) == 2 and later[-1]["second_boss"], "A10: act 3 ends in two bosses"
        pool = collections.Counter(v.id("encounter", e) for e in g["normal"] + g["elite"])
        for arrays in (K.encode(d), tok.encode_priv(d.decision)):
            cat, cont, got_pool = _priv_run(arrays)
            assert cont[C.XF["P_ROOMS_MONSTER"]] == f16(len(g["normal"]) / 10.0) > 0
            assert cont[C.XF["P_ROOMS_ELITE"]] == f16(len(g["elite"]) / 10.0) > 0
            assert cont[C.XF["P_ROOMS_EVENT"]] == f16(len(g["events_shuffled"]) / 10.0) > 0
            assert cat[C.XC["X_ID_MAIN"]] == v.id("encounter", later[0]["boss"])
            assert cat[C.XC["X_ID_REL"]] == v.id("encounter", later[-1]["boss"])
            assert cat[C.XC["X_ID_MOD"]] == v.id("encounter", later[-1]["second_boss"])
            ids = [cat[C.XC[c]] for c in ("X_ID_MAIN", "X_ID_MOD", "X_ID_REL")]
            assert all(x not in (V.NONE_ID, V.UNK_ID["encounter"]) for x in ids)
            assert got_pool == {k: float(f16(n)) for k, n in pool.items()} and got_pool
        kinds[d.kind] += 1
        d = env.step(act(d))
    assert kinds[E.COMBAT] >= 3 and kinds[E.OOC] >= 3, kinds


def check_decision(d: E2EDecision) -> C.EnvDecision:
    """`d.decision` passes the schema's validator and agrees with the env's own fields."""
    e = d.decision
    C.validate_decision(e)
    assert e.member_class.tolist() == d.classes
    assert e.rep.tolist() == [m[0] for m in d.members]
    assert e.class_kinds().tolist() == d.cls_kind and e.n_classes == d.n_classes
    assert e.t == d.row and e.run_uid == d.run_uid and e.cell == d.cell
    assert e.is_combat == d.is_combat and e.dtype == d.dtype
    if d.is_combat:
        assert e.actions is d.obs_actions and e.obs is None and e.metrics is None
        assert e.fight_turn == d.turn and e.fight_decision == d.fight_row
        assert e.policy.belief() is d.bv and e.priv.true_state() is d.cs
    else:
        assert e.actions is d.actions and e.obs is d.run_obs and e.metrics is d.metrics
    return e


def test_every_stored_row_is_a_valid_env_decision() -> None:
    seen: collections.Counter = collections.Counter()
    for k in range(4):
        rows, _ = play(pinned_env(SMOKE + 50 + k), random_actor(50 + k))
        for d in rows:
            e = check_decision(d)
            seen[d.dtype_name] += 1
            seen["reopened"] += int(e.sel.picked > 0)
            if d.dtype_name == "NEOW":
                assert e.event_name == "Neow"
    for name in ("COMBAT_TURN", "COMBAT_CHOICE", "NEOW", "EVENT", "MAP", "CARD_REWARD"):
        assert seen[name] > 0, (name, seen)


def test_a_card_grant_counts_its_picks() -> None:
    """A 2-of-8 grant arrives as two decisions: picks still required, offered, made."""
    env = pinned_env(SMOKE + 65, EnvConfig(f9_p=0.0))
    act = random_actor(65)
    d = env.reset()
    sels = []
    while isinstance(d, E2EDecision) and not d.bootstrap:
        if d.dtype_name == "CARD_GRANT":
            sels.append(check_decision(d).sel)
        d = env.step(act(d))
    assert [(s.min, s.max, s.picked, s.can_skip) for s in sels] == [
        (2, 8, 0, False),
        (1, 7, 1, False),
    ]


class _Tok:
    """A `contracts.Tokenizer` stub recording what the env drives."""

    def __init__(self) -> None:
        self.resets = 0
        self.seen: list[Any] = []

    def reset_fight(self) -> None:
        self.resets += 1

    def observe(self, policy: Any) -> None:
        assert isinstance(policy, C.PolicySource) and not isinstance(policy, C.PrivSource)
        self.seen.append(policy)

    def encode(self, d: Any) -> dict[str, Any]:
        return {}

    def encode_priv(self, d: Any) -> dict[str, Any]:
        return {}


def test_the_env_drives_the_tokenizer_at_every_combat_step() -> None:
    tok = _Tok()
    assert isinstance(tok, C.Tokenizer)
    env = pinned_env(SMOKE + 70, EnvConfig(tokenizer=tok))
    act = random_actor(70)
    d = env.reset()
    stored = 0
    while isinstance(d, E2EDecision) and not d.bootstrap:
        if d.is_combat:
            stored += 1
            last = tok.seen[-1]  # observed before the env yields the decision
            assert last.belief() is d.bv and last.legal() == d.obs_actions
        d = env.step(act(d))
    t = env.terminal
    assert t.outcome in ("won", "lost") or t.cap == "turns"  # caps checked before observe
    assert tok.resets == sum(v for k, v in t.counts.items() if k.startswith("fights/"))
    forced = t.counts["forced/combat"] + t.counts["forced/combat_merged"]
    assert len(tok.seen) == stored + forced and forced > 0


def test_a_tokenizer_assert_errors_the_segment_with_a_counted_reason() -> None:
    """G-schema: an assert, the tokenizer's included, ends the segment as an error with a
    counted reason (`error/<type>`); the worker lives on."""

    class Failing(_Tok):
        def observe(self, policy: Any) -> None:
            super().observe(policy)
            if len(self.seen) == 5:
                raise C.SchemaError("stub tokenizer assert")

    tok = Failing()
    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(tokenizer=tok))
    _, t = play(env, random_actor(9))
    assert t.outcome == "error" and env.done and "stub tokenizer assert" in (t.error or "")
    assert t.counts["error"] == t.counts["error/SchemaError"] == t.counts["outcome/error"] == 1
    pf = PF.preflight(PF.tally([t]))
    assert pf.errors == 1 and not pf.ok and pf.error_types == {"SchemaError": 1}
    assert any(p.endswith("error (engine or assert): SchemaError 1") for p in pf.problems())
    assert E2EEnv(SMOKE + 9).reset() is not None  # a fresh env in the same process still works


# ── what ends a segment, and what ends the process (A2) ─────────────────────


@pytest.fixture(scope="module")
def rust_panic(tmp_path_factory) -> BaseException:
    """A real `pyo3_runtime.PanicException`: the engine panics loading a corrupt ONNX."""
    bad = tmp_path_factory.mktemp("panic") / "not_an.onnx"
    bad.write_text("this is not an onnx model", encoding="utf-8")
    try:
        sts2_rs.PpoRunAgent(model_path=str(bad))
    except BaseException as ex:  # the panic under test
        assert not isinstance(ex, Exception) and type(ex).__name__ == "PanicException"
        return ex
    pytest.fail("a corrupt ONNX loaded")


def test_a_build_mismatch_at_fight_start_propagates(monkeypatch) -> None:
    """A2: `factory.check_fight` raising `GameVersionMismatch` at a fight start leaves the
    env: the process is on the wrong build, so the segment is not an ordinary error."""
    calls = []

    def mismatch(cs: Any, **_kw: Any) -> Any:
        calls.append(cs)
        raise sts2_client.GameVersionMismatch("stub: this fight simulates v0.107.1")

    env = E2EEnv(SMOKE + 21)
    monkeypatch.setattr(factory, "check_fight", mismatch)
    with pytest.raises(sts2_client.GameVersionMismatch, match="stub: this fight") as ei:
        play(env, random_actor(21))
    assert len(calls) == 1 and any(f.name == "_start_fight" for f in ei.traceback)
    assert not env.done and env.counts["error"] == 0


def test_a_process_on_another_build_fails_the_fight_start(monkeypatch) -> None:
    """A2, unstubbed: the process selection flips to v0.107.1 after the run is built; the
    real fight-start check (`require_selection`) raises and the env lets it through."""
    env = E2EEnv(SMOKE + 21)
    d = env.reset()
    monkeypatch.setattr(sts2_client, "GAME_VERSION", "v0.107.1")
    act = random_actor(21)
    with pytest.raises(sts2_client.GameVersionMismatch, match=r"selects v0\.107\.1") as ei:
        while isinstance(d, E2EDecision):
            d = env.step(act(d))
    assert any(f.name == "_start_fight" for f in ei.traceback)


@pytest.mark.parametrize(
    "exc", [KeyboardInterrupt, SystemExit, GeneratorExit, sts2_client.GameVersionMismatch]
)
def test_interrupts_and_build_mismatches_leave_the_env(exc: type[BaseException]) -> None:
    class Failing(_Tok):
        def observe(self, policy: Any) -> None:
            super().observe(policy)
            if len(self.seen) == 5:
                raise exc("stub")

    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(tokenizer=Failing()))
    with pytest.raises(exc):
        play(env, random_actor(9))
    assert not env.done and env.counts["error"] == 0


@pytest.mark.parametrize("kind", ["panic", "runtime", "assert"])
def test_a_rust_panic_or_an_exception_errors_the_segment(kind: str, rust_panic) -> None:
    """A Rust panic (a `BaseException`) stays an errored run, as does any `Exception`
    that is not a build mismatch, `AssertionError` (its base class) included."""
    ex = {"panic": rust_panic, "runtime": RuntimeError("stub"), "assert": AssertionError("x")}

    class Failing(_Tok):
        def observe(self, policy: Any) -> None:
            super().observe(policy)
            if len(self.seen) == 5:
                raise ex[kind]

    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(tokenizer=Failing()))
    _, t = play(env, random_actor(9))
    name = type(ex[kind]).__name__
    assert t.outcome == "error" and env.done and t.counts[f"error/{name}"] == 1
    assert E.ends_segment(ex[kind]) and E.is_rust_panic(ex[kind]) == (kind == "panic")


def test_the_error_fingerprint_lets_interrupts_through(monkeypatch, rust_panic) -> None:
    """The error terminal's fingerprint read keeps the same rule: a panic reading a broken
    state gives `unavailable`, an interrupt propagates."""

    class Failing(_Tok):
        def observe(self, policy: Any) -> None:
            super().observe(policy)
            if len(self.seen) == 5:
                raise RuntimeError("stub")

    def broken(rm: Any) -> str:
        raise rust_panic

    monkeypatch.setattr(E, "run_fingerprint", broken)
    env = E2EEnv(SMOKE + 9, cfg=EnvConfig(tokenizer=Failing()))
    _, t = play(env, random_actor(9))
    assert t.outcome == "error" and t.fingerprint == "unavailable"

    def interrupted(rm: Any) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(E, "run_fingerprint", interrupted)
    with pytest.raises(KeyboardInterrupt):
        play(E2EEnv(SMOKE + 9, cfg=EnvConfig(tokenizer=Failing())), random_actor(9))


def test_segment_rows_carry_the_schemas_row_fields() -> None:
    env = E2EEnv(SMOKE + 14)
    rows, t = play(env, random_actor(14))
    b = E.segment_rows(rows, t)
    n = len(rows)
    for name, arr in b.items():
        assert arr.dtype == C.FIELDS[name].dtype, name
    assert all(len(b[k]) == n for k in ("run_uid", "t", "dtype", "reward", "done", "n_mem"))
    assert len(b["mem_class"]) == int(b["n_mem"].sum())
    assert len(b["cls_kind"]) == int(b["n_classes"].sum())
    assert b["done"].tolist() == [False] * (n - 1) + [True]
    assert not b["bootstrap"].any() and (b["cell"] == C.CELL_NONE).all()
    assert (b["run_uid"] == SMOKE + 14).all() and (b["n_classes"] >= 2).all()
    assert b["reward"].sum() == pytest.approx(t.total_reward, abs=1e-6)
    assert set(b["dtype"].tolist()) <= C.LIVE_DTYPES
    assert set(b["cls_kind"].tolist()) <= C.LIVE_ACTION_KINDS
    with pytest.raises(ValueError):
        E.segment_rows(rows[1:], t)
    # The env keeps each row's fields itself, so the actor can drop the decisions.
    mine = env.segment_rows()
    assert mine.keys() == b.keys() and all(np.array_equal(mine[k], b[k]) for k in b)
    assert env.row_meta == [d.meta() for d in rows]
    with pytest.raises(RuntimeError, match="before the segment ended"):
        E2EEnv(SMOKE + 14).segment_rows()


# ── pre-flight ───────────────────────────────────────────────────────────────


def test_preflight_flags_missing_surfaces_unknown_caps_and_f9() -> None:
    good = {f"rows/{t}": 1 for t in PF.REQUIRED_DTYPES}
    good.update({f"fine/{k}": 1 for k in PF.REQUIRED_FINE})
    good.update({"segments": 2000, "segments/run": 2000})
    pf = PF.preflight(good)
    assert pf.ok, pf.problems()
    assert set(pf.gaps) == set(PF.GAP_KEYS)
    bad = dict(good, **{"rows/CARD_GRANT": 0, "unknown": 1, "outcome/capped": 3})
    bad["f9/violations"] = 1
    pf = PF.preflight(bad)
    assert not pf.ok
    probs = " | ".join(pf.problems())
    assert "rows/CARD_GRANT" in probs and "unknown = 1" in probs
    assert "capped 3/2000" in probs and "F9" in probs
    assert PF.preflight(dict(good, **{"outcome/capped": 2})).ok  # 0.1% is not above 0.1%
    row = pf.stats_row()
    assert row["dec_per_run/CARD_REWARD"] == pytest.approx(1 / 2000)
    assert not row["preflight/ok"] and not row["preflight/asserts"]
    over = PF.preflight(dict(good, **{"assert/members": 1, "outcome/error": 1}))
    assert not over.ok and over.asserts == {"members": 1} and over.errors == 1
    assert "1 decision(s) over the schema's members cap" in over.problems()
    assert over.stats_row()["preflight/asserts"] == pytest.approx(1)


def test_a_decision_over_the_caps_is_an_assert_with_its_own_count() -> None:
    counts: collections.Counter = collections.Counter()
    E.check_caps(255, 64, "COMBAT_CHOICE", counts)
    assert not counts
    with pytest.raises(C.SchemaError, match="266 members / 12 classes exceed 255 / 64"):
        E.check_caps(266, 12, "COMBAT_CHOICE", counts)
    with pytest.raises(C.SchemaError):
        E.check_caps(300, 65, "SHOP", counts)
    with pytest.raises(C.SchemaError):
        E.check_caps(70, 65, "SHOP")  # no counter: still raises
    assert counts == {"assert/members": 2, "assert/classes": 1}


@pytest.mark.slow
def test_preflight_over_twenty_pinned_runs() -> None:
    """Twenty HP-pinned random runs present every required surface, no `unknown`."""
    terms = []
    for k in range(20):
        seed = SMOKE + 200 + k
        _, t = play(pinned_env(seed, EnvConfig(f9_p=0.0)), random_actor(seed))
        assert t.outcome in ("won", "lost", "capped"), t.error
        terms.append(t)
    pf = PF.preflight(PF.tally(terms))
    assert pf.unknown == 0 and pf.errors == 0
    assert pf.missing == [], pf.missing
    assert sum(t.won for t in terms) >= 5
    assert 0.1 < pf.combat_auto_share < 0.35, pf.combat_auto_share  # random play: 20-22%
    assert pf.stats_row()["preflight/combat_auto_share"] == pf.combat_auto_share


def test_an_ooc_stall_is_an_error_not_a_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run stepper's `no_actions` (no legal action in a run that is not over) is an
    engine stall, not a death: the segment ends as a counted error (`error/NoActions`)
    and the learner drops it, like the combat-side stall (review finding ENVR-2; plan
    §3 and §6: only deaths and caps are losses)."""
    from ppo.env import Terminal

    env = E2EEnv(SMOKE + 9)
    d = env.reset()
    assert isinstance(d, E2EDecision) and not d.is_combat

    def stall(idx: int) -> Terminal:
        return Terminal(outcome="no_actions", won=False, metrics={}, steps=0, decisions=0)

    monkeypatch.setattr(env.run, "step", stall)
    t = env.step(0)
    assert t.outcome == "error" and env.done and not t.won
    assert t.counts["error/NoActions"] == 1 and t.counts["outcome/error"] == 1
    assert "outcome/lost" not in t.counts and "NoActions" in t.error
    pf = PF.preflight(PF.tally([t]))
    assert pf.errors == 1 and pf.error_types == {"NoActions": 1}


def test_a_spent_one_shot_enchantment_splits_the_play_class() -> None:
    """Vigorous adds its amount until a card's first play (C# `Vigorous.AfterCardPlayed`);
    an Anger clone keeps it. A spent and a fresh copy preview different damage, so they
    are two classes (#1097: eval seed 160000444, two Vigorous ANGERs, 21 vs 12)."""
    from ppo.e2e import env as EE

    base = {"id": "ANGER", "kind": "card", "enchantment": "Vigorous", "enchantment_amount": 8}
    fresh = dict(base, combat_vars={"generated_in_combat": 1})
    spent = dict(base, combat_vars={"enchantment_used": 1})
    assert EE._face_vars(fresh) == {}
    assert EE._face_vars(spent) == {"enchantment_used": 1}
