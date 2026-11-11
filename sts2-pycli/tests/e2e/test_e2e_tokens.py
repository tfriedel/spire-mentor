"""WP3b: the e2e tokenizer (`ppo.e2e.tokens`) against SCHEMA.md.

On a short recorded v0.111.0 A10 run: every row passes `validate_row_batch`
and `check_compact_lossless`; the combat teacher's slice is byte-equal to
#1081's `Encoder.encode` (X-C; with the it06000 checkpoint, its logits too);
merged members tokenize identically; PRIV never reaches the policy arrays
(T14); and the policy arrays do not move when the hidden state does (F7: a
re-dealt draw pile and reseeded combat RNG, other room queues and later
bosses).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sts2_client
import sts2_rs
from ppo.combat_ctx.snapshot import honest_snapshot
from ppo.e2e import contracts as CT
from ppo.e2e import factory
from ppo.e2e import recorder as R
from ppo.e2e import tokens as K
from ppo.e2e import vocab as V

TEACHER = Path.home() / "projects/sts2/bench-archive/2026-11-02-e2e-inputs/run/it06000.pt"


@pytest.fixture(autouse=True, scope="module")
def _campaign_process():
    """The factory needs the process to select v0.111.0 (`STS2_GAME_VERSION`)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sts2_client, "GAME_VERSION", factory.GAME_VERSION)
        yield


@lru_cache(maxsize=1)
def _run() -> dict[str, Any]:
    # Seed 13: the run whose sampled decisions hold exactly one EVENT, the
    # RoomFullOfCheese one `test_event_previews_are_honest` reads. Its 8-sim
    # combat searches draw from state-hash-seeded RNG, so an engine change
    # that moves the card hash moves the run (the cost model moved seed 7's
    # first divergent search at step 18, on an identical state, and its run
    # no longer meets an event).
    return R.record_run(13, 0.6, 8, 8)


@lru_cache(maxsize=1)
def _decisions() -> list[tuple[R.State, K.RecordedDecision]]:
    out = []
    tr = K.SelectionTracker()
    for st in R.replay(_run()):
        # A State's rm is live until the next yield: keep a clone.
        st.rm = st.rm.clone()
        if st.cs is not None:
            st.cs = st.rm.combat_state()
            st.bv = st.cs.belief_view(0)
        d = K.decision_from_state(st, tr)
        if d is not None:
            out.append((st, d))
    return out


def _tok_of(row: dict[str, Any], prow: dict[str, Any], d: Any) -> K.Tok:
    """A `Tok` from a `Tokenizer`'s `encode` and `encode_priv` outputs."""
    return K.Tok(is_combat=bool(d.is_combat), dtype=int(d.dtype), **row, **prow)


def _same(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    return all(x.dtype == y.dtype and x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def test_rows_validate_and_are_compact_lossless() -> None:
    ds = _decisions()
    toks = [K.encode(d) for _st, d in ds]
    kinds = {CT.DECISION_TYPES[t.dtype].name for t in toks}
    assert {"COMBAT_TURN", "MAP"} <= kinds and len(kinds) >= 4, kinds
    b = K.row_batch(toks, run_uid=[7] * len(toks))
    CT.validate_row_batch(b)
    CT.check_compact_lossless(b["cat"], b["cont"], b["sec"])
    CT.check_compact_lossless(b["pcat"], b["pcont"], b["psec"])
    # The compact codec round-trips the rows, PRIV included.
    for c, f, s in (("cat", "cont", "sec"), ("pcat", "pcont", "psec")):
        cat, cont = CT.from_compact(*CT.to_compact(b[c], b[f], b[s]), b[s])
        assert cat.tobytes() == b[c].tobytes() and cont.tobytes() == b[f].tobytes(), c
    nt = np.asarray([t.n_tok for t in toks])
    assert nt.max() <= CT.MAX_TOKENS and nt.mean() <= 65, (nt.mean(), nt.max())
    for t in toks:
        assert t.sec[:6].tolist() == [CT.SEC["RUNCTX"], *CT.SUMMARY_SECTIONS]
        nb = int((t.sec == CT.SEC["BOSS"]).sum())
        assert 1 <= nb <= 2
        assert t.n_priv <= CT.MAX_PRIV
        if t.is_combat:
            assert t.sctx_lo == 6 + nb


def test_teacher_slice_is_encoder_encode_bit_for_bit() -> None:
    """X-C at the row level, with an identity RareMap (no checkpoint needed)."""
    from ppo.combat_ppo.encode import Encoder

    enc = Encoder("S_ctx", {}, CT.PRIV_DRAW_K)
    n = 0
    for st, d in _decisions():
        if not d.is_combat:
            continue
        dec = enc.encode(st.snap, st.bv, st.cs, st.cs.legal_actions())
        tok = K.encode(d)
        b = K.row_batch([tok])
        sl = CT.teacher_slice(b, 0)
        assert sl["cat"].tobytes() == dec.cat.tobytes()
        assert sl["cont"].tobytes() == dec.cont.tobytes()
        assert sl["sec"].tobytes() == dec.sec.tobytes() and sl["n_act"] == dec.n_act
        pd = tok.psec == CT.SEC["PRIV_DRAW"]
        assert tok.pcat[pd][:, : CT.SCTX_N_CAT].tobytes() == dec.pcat.tobytes()
        assert tok.pcont[pd][:, : CT.SCTX_N_CONT].tobytes() == dec.pcont.tobytes()
        n += 1
    assert n > 10


@pytest.mark.skipif(not TEACHER.exists(), reason="it06000.pt (the #1081 teacher) absent")
def test_teacher_logits_through_the_slice_are_equal() -> None:
    """X-C with the teacher's own RareMap and net: `torch.equal` logits."""
    import torch
    from ppo.combat_ctx import tokens as T
    from ppo.combat_ctx.model import CombatCtxNet
    from ppo.combat_ppo import nets
    from ppo.combat_ppo.encode import Decision, Encoder

    ck = torch.load(TEACHER, map_location="cpu", weights_only=False)
    assert ck["vocab"] == T.vocab().digest()
    net = CombatCtxNet.from_config(dict(ck["config"], dropout=0.0))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    enc = Encoder("S_ctx", ck["rare"], 0)
    ours, theirs = [], []
    for st, d in _decisions():
        if not d.is_combat:
            continue
        theirs.append(enc.encode(st.snap, st.bv, st.cs, st.cs.legal_actions()))
        sl = K.encode(d, rare=enc.rare, priv=False).teacher()
        ours.append(Decision(cat=sl["cat"], cont=sl["cont"], sec=sl["sec"], n_act=sl["n_act"]))
        assert ours[-1].cat.tobytes() == theirs[-1].cat.tobytes()
    with torch.no_grad():
        a = nets.run(net, nets.collate(theirs, "cpu", priv=False))[0]
        b = nets.run(net, nets.collate(ours, "cpu", priv=False))[0]
    assert torch.equal(a, b)


def _dup_fight() -> Any:
    return factory.scenario_state(
        3,
        "Ironclad",
        ["STRIKE_IRONCLAD"] * 10,
        70,
        80,
        ["BURNING_BLOOD"],
        "SLIMES_WEAK",
        ascension=10,
    )


class _Fight:
    """A scenario fight as a decision (no run: the head reads a stub run)."""

    def __init__(self, cs: Any) -> None:
        from ppo.e2e import env as E

        self.cs = cs
        self.bv = cs.belief_view(0)
        self.d_bv = sts2_rs.state_to_dict(self.bv)
        self.snap = honest_snapshot(self.bv, "monster")
        self.snap["ehist"] = [[] for _ in self.snap["enemies"]]
        cc = E.merged_classes(cs, room="monster")
        self.kind, self.dtype, self.kinds = "combat", cc.dtype, cc.kinds
        self.obs_actions, self.classes, self.members = cc.obs_actions, cc.classes, cc.members
        self.sel = CT.NO_SELECTION
        if cc.dtype in CT.SELECTION_DTYPES:
            self.sel = E.combat_selection(self.bv.pending_choice(), cc.kinds)
        self.actions = cs.legal_actions()
        self.room, self.run_obs = "monster", None
        self.rm = factory.new_run(5)

    is_combat = True


def test_merged_members_tokenize_identically() -> None:
    f = _Fight(_dup_fight())
    assert any(len(m) > 1 for m in f.members), f.members
    tok = K.encode(f, priv=False)
    hidden = tok.tok_flags[tok.mem_tok.astype(int)] & CT.TF_HIDDEN
    assert int((hidden > 0).sum()) == len(f.obs_actions) - len(f.members)
    # Break a merge on purpose: a Strike and the end turn cannot share a class.
    f.members = [[0, len(f.obs_actions) - 1], *[[i] for i in range(1, len(f.obs_actions) - 1)]]
    f.classes = [0, *range(1, len(f.obs_actions) - 1), 0]
    with pytest.raises(CT.SchemaError):
        K.encode(f, priv=False)


def test_combat_picks_flag_the_selected_hand_card() -> None:
    cs = factory.scenario_state(
        3,
        "Ironclad",
        ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 5,
        70,
        80,
        ["BURNING_BLOOD"],
        "SLIMES_WEAK",
        ascension=10,
        potions=["GAMBLERS_BREW"],
    )
    cs.apply(next(a for a in cs.legal_actions() if a["action"] == "drink_potion"))
    cs.apply(next(a for a in cs.legal_actions() if a.get("selection") == [1]))
    f = _Fight(cs)
    assert K.combat_picks(f.snap, f.obs_actions) == [1]
    tok = K.encode(f, priv=False)
    hand = np.flatnonzero(tok.sec == CT.SEC["HAND"])
    picked = tok.cont[hand, CT.XF["K_PICKED"]].astype(float)
    assert picked.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert float(tok.cont[0, CT.XF["R_SEL_PICKED"]]) == pytest.approx(0.2, abs=1e-3)
    assert float(tok.cont[0, CT.XF["R_SEL_CAN_SKIP"]]) == pytest.approx(1.0)


class _NoPriv:
    """A run whose PRIV reads raise: the policy part must never make one."""

    def __init__(self, rm: Any) -> None:
        self._rm = rm

    def __getattr__(self, name: str) -> Any:
        if f"rm.{name}" in CT.PRIV_CALLS:
            raise AssertionError(f"the policy tokenizer read rm.{name}")
        return getattr(self._rm, name)


def test_priv_never_reaches_the_policy_arrays() -> None:
    """T14 at the row level: the policy arrays with and without PRIV are equal,
    and the policy part makes no PRIV call."""
    import dataclasses

    for _st, d in _decisions()[:80]:
        full = K.encode(d)
        d2 = dataclasses.replace(d, rm=_NoPriv(d.rm))
        K.clear_cache()
        pol = K.encode(d2, priv=False)
        assert _same(full.policy_arrays(), pol.policy_arrays())
        assert pol.n_priv == 0 and not np.isin(full.sec, list(CT.PRIV_SECTIONS)).any()


def test_policy_arrays_ignore_the_draw_order_and_combat_rng() -> None:
    """F7 at the row level: `sample_world` re-deals the unknown draw pile and
    reseeds every RNG stream; the policy arrays must not move."""
    import dataclasses

    from ppo.e2e import env as E

    n = pairs = moved = 0
    for st, d in _decisions():
        if not d.is_combat:
            continue
        K.clear_cache()
        full = K.encode(d)
        base = K.encode(d, priv=False)
        top = list(st.cs.true_draw_top(10))
        for k in (1, 2):
            w = st.cs.sample_world(9000 + k)
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
            assert _same(base.policy_arrays(), K.encode(dw, priv=False).policy_arrays())
            K.clear_cache()
            tw = K.encode(dw)
            pairs += 1
            moved += list(w.true_draw_top(10)) != top or (
                full.pcat.tobytes() != tw.pcat.tobytes()
                or full.pcont.tobytes() != tw.pcont.tobytes()
            )
        n += 1
    assert n > 10
    # Not vacuous: the re-deal moved the true draw or PRIV on most rows (a few rows
    # have a draw pile of at most one card, or one the player knows, so no re-deal
    # can move it). A no-op `sample_world` fails here.
    assert moved >= 0.7 * pairs, (moved, pairs)


class _HiddenRooms:
    """A run whose hidden future differs: other room queues, other later bosses."""

    def __init__(self, rm: Any) -> None:
        self._rm = rm

    def __getattr__(self, name: str) -> Any:
        if name in ("act_map_for", "act_map", "future_maps"):
            raise AssertionError(f"the tokenizer read {name} (a future map)")
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


def test_policy_arrays_ignore_rooms_and_later_bosses() -> None:
    """F7 at the row level for the run: room queues and later acts' bosses
    reach PRIV only, and no future map is read."""
    import dataclasses

    changed = 0
    for _st, d in _decisions():
        a = K.encode(d)
        b = K.encode(dataclasses.replace(d, rm=_HiddenRooms(d.rm)))
        assert _same(a.policy_arrays(), b.policy_arrays())
        changed += int(
            a.pcont.tobytes() != b.pcont.tobytes() or a.pcat.tobytes() != b.pcat.tobytes()
        )
    assert changed > 0


def test_event_previews_are_honest() -> None:
    """Schema revision 2: an event option's O_EV_* columns are
    `contracts.event_delta_columns` of its preview (a relic-gaining option
    keeps only the relic flag and its losses), and the run seed `honest_obs`
    drops never reaches a row."""
    import dataclasses

    (d,) = [d for _st, d in _decisions() if CT.DECISION_TYPES[d.dtype].name == "EVENT"]
    names = [c for c, _k, _s in CT.EVENT_DELTA_COLUMNS]
    cols = [CT.XF[c] for c in names]

    def ev_cols(dd: K.RecordedDecision) -> dict[int, dict[str, float]]:
        tok = K.encode(dd)
        out = {}
        for t in np.flatnonzero(tok.sec == CT.SEC["OPT_EVENT"]):
            (cls,) = np.flatnonzero(tok.cls_tok == t)
            idx = int(dd.obs_actions[dd.members[int(cls)][0]]["idx"])
            out[idx] = dict(zip(names, tok.cont[t, cols].astype(np.float32).tolist(), strict=True))
        return out

    infos = d.run_obs["event_options"]
    relic = [i for i, o in enumerate(infos) if (o.get("relics_gained") or 0) > 0]
    assert relic, "the RoomFullOfCheese event has a relic-gaining option"
    got = ev_cols(d)
    assert sorted(got) == list(range(len(infos)))
    for idx, cols_got in got.items():
        want = CT.event_delta_columns(infos[idx])
        assert cols_got == {c: float(np.float16(want[c])) for c in names}, idx
    # A preview inflated by the relic's pickup effect reads as the bare flag.
    fake = [
        dict(o, gold_delta=75, max_hp_delta=4, cards_gained=1, relics_gained=2) if i in relic else o
        for i, o in enumerate(infos)
    ]
    got = ev_cols(dataclasses.replace(d, run_obs=dict(d.run_obs, event_options=fake)))
    masked = ("O_EV_GOLD", "O_EV_HP", "O_EV_MAXHP", "O_EV_CARDS_GAIN", "O_EV_POTIONS_GAIN")
    want = dict.fromkeys(masked, 0.0) | {"O_EV_RELICS_GAIN": 1.0}
    for i in relic:
        assert {c: got[i][c] for c in want} == want, i
    # The run seed is not a policy input.
    a = K.encode(d)
    b = K.encode(dataclasses.replace(d, run_obs=dict(d.run_obs, seed="NOT_THE_SEED")))
    assert _same(a.policy_arrays(), b.policy_arrays())


def test_run_reads_use_the_revision_2_facade() -> None:
    """In combat the deck enchantments, ascension, act floor and pending
    enchantment come from the run facade (POLICY_CALLS, schema revision 2);
    out of combat `honest_obs(obs)` answers first and never holds the seed;
    a facade without a read falls back to the map row and the campaign."""
    from types import SimpleNamespace as NS

    ench = [{"card_index": 0, "enchantment": "Sharp"}]
    fac = NS(
        master_deck_enchantments=lambda: ench,
        ascension_level=7,
        act_floor=lambda: 5,
        pending_enchantment="Swift",
    )

    def run_of(facade: Any, obs: dict[str, Any] | None) -> Any:
        return K._Run(NS(policy=lambda: NS(run=lambda: facade), run_obs=obs, run_key=("uid", 1)))

    run = run_of(fac, None)
    assert run.deck_ench() == ench and run.ascension() == 7 and run.act_floor({}) == 5
    assert run.read("pending_enchantment") == "Swift"
    run = run_of(fac, {"ascension": 3, "deck_enchantments": [], "seed": "S"})
    assert run.ascension() == 3 and run.deck_ench() == [] and "seed" not in run.obs
    run = run_of(NS(), None)
    assert run.deck_ench() == [] and run.act_floor({"current_coord": {"row": 4}}) == 5
    assert run.ascension() == factory.ASCENSION


def test_head_and_option_content() -> None:
    v = V.load()
    seen: set[str] = set()
    for _st, d in _decisions():
        tok = K.encode(d)
        name = CT.DECISION_TYPES[tok.dtype].name
        seen.add(name)
        assert tok.cat[0, CT.XC["X_DTYPE"]] == tok.dtype
        boss = tok.cat[tok.sec == CT.SEC["BOSS"], CT.XC["X_ID_MAIN"]]
        assert set(boss.tolist()) <= set(v.block("encounter")), boss
        rep = tok.cls_tok.astype(int)
        assert (tok.cat[rep, CT.XC["X_KIND"]] == tok.cls_kind).all()
        if name == "MAP":
            nodes = tok.sec == CT.SEC["MAPNODE"]
            assert 0 < nodes.sum() <= CT.MAX_MAPNODES
            nxt = tok.cont[nodes, CT.XF["N_NEXT"]].astype(float)
            assert int(nxt.sum()) == len({a["col"] for a in d.obs_actions})
            moves = tok.sec == CT.SEC["OPT_MOVE"]
            assert (tok.cont[moves, CT.XF["M_ROWS_LEFT"]] > 0).all()
        if name == "REST":
            ids = tok.cat[tok.sec == CT.SEC["OPT_REST"], CT.XC["X_ID_MAIN"]].tolist()
            assert v.id("rest", "HEAL") in ids
        if tok.is_combat:
            enemy = tok.sec == CT.SEC["ENEMY"]
            assert (tok.cat[enemy, CT.XC["X_ID_MAIN"]] >= v.block("monster")[0]).all()
            hand = tok.sec == CT.SEC["HAND"]
            assert (tok.cat[hand, CT.XC["X_ID_MAIN"]] >= v.block("card")[0]).all()
    assert {"COMBAT_TURN", "MAP"} <= seen


def test_ooc_card_tokens_carry_their_card_type() -> None:
    """Deck and offer ids are PascalCase (`StrikeIronclad`) while S_ctx's
    card-type table is keyed by registry ids: DECK and OPT_CARD tokens still
    get the type, the same slot value a combat card token of that card gets."""
    from ppo.combat_ctx import tokens as T

    assert K._card_type("StrikeIronclad") == K._card_type("STRIKE_IRONCLAD") == "Attack"
    assert K._card_type("DefendIronclad") == "Skill" and K._card_type("NoSuchCard") == "?"
    unknown = K._tslot("card_type", "?")
    slot = T.S["card_type"]
    n = 0
    for _st, d in _decisions():
        tok = K.encode(d)
        if tok.is_combat:
            continue
        cards = (tok.sec == CT.SEC["DECK"]) | (tok.sec == CT.SEC["OPT_CARD"])
        n += int((tok.sec == CT.SEC["OPT_CARD"]).sum())
        assert (tok.cat[cards, slot] != unknown).all(), CT.DECISION_TYPES[tok.dtype].name
    assert n > 0, "the run offers cards"


def test_selection_tracker_counts_grant_picks() -> None:
    class _Rm:
        def __init__(self, n: int) -> None:
            self.n, self.act_index = n, 1

        def card_grant_options(self) -> list[Any]:
            return [("STRIKE_IRONCLAD", False)] * self.n

        def total_floor(self) -> int:
            return 4

    tr = K.SelectionTracker()
    grant = CT.DT["CARD_GRANT"]
    assert [tr.picked(_Rm(n), grant) for n in (8, 7, 6)] == [0, 1, 2]
    assert tr.picked(None, CT.DT["COMBAT_TURN"]) == 0
    assert tr.picked(_Rm(5), grant) == 0


def test_bosses_left_counts_the_double_boss() -> None:
    assert K.bosses_left(1, 1, 10) == 4  # acts 1, 2 and act 3's two
    assert K.bosses_left(3, 2, 10) == 2
    assert K.bosses_left(2, 1, 0) == 2


def test_a_beaten_boss_is_not_left() -> None:
    """Review finding TOK-3: R_BOSSES_LEFT counts this act's bosses not yet beaten. The
    maps are the A10 act-3 ones the review logged (boss rows 14 and 15)."""
    fm = {
        "boss": {"id": "AEONGLASS_BOSS", "row": 14},
        "second_boss": {"id": "TEST_SUBJECT_BOSS", "row": 15},
    }
    at = [(None, False, 0), (13, False, 0), (14, True, 0), (15, True, 1)]
    for row, combat, beaten in at:
        cur = {**fm, "current_coord": None if row is None else {"col": 3, "row": row}}
        assert K.bosses_beaten(cur, combat) == beaten, (row, combat)
    act1 = {"boss": {"id": "B", "row": 16}, "current_coord": {"col": 3, "row": 16}}
    assert K.bosses_beaten(act1, True) == 0 and K.bosses_beaten(act1, False) == 1


def test_post_boss_rewards_count_one_boss_fewer() -> None:
    """The act-1 boss fight, then its card reward and boss relic, on a real run (the
    review's probe 15): 4 bosses left in the fight, 3 at the rewards."""
    from ppo.e2e import env as E
    from ppo.e2e import fillcheck as FC

    seed = 162_000_201
    rm = factory.new_run(seed)
    rm.set_player(hp=100_000, max_hp=100_000)
    env = E.E2EEnv(seed, rm=rm, cfg=E.EnvConfig(max_fight_turns=400))
    act, d, seen = FC.random_actor(seed), env.reset(), {}
    while isinstance(d, E.E2EDecision) and rm.act_index == 1:
        fm = rm.full_map()
        cur, boss = fm.get("current_coord"), (fm.get("boss") or {}).get("row")
        if cur is not None and cur["row"] == boss:
            left = round(float(K.encode(d).cont[0][CT.XF["R_BOSSES_LEFT"]]) * 4)
            seen.setdefault(d.dtype_name, left)
        d = env.step(act(d))
    assert seen.get("COMBAT_TURN") == 4, seen
    assert seen.get("CARD_REWARD") == 3 and seen.get("BOSS_RELIC") == 3, seen


def test_tokenizer_protocol_matches_encode() -> None:
    """The env path (`Tokenizer` over `contracts.EnvDecision`s, driven by the
    replay's `reset_fight` / `observe` hooks) builds the same rows as `encode`
    over the recorded decision, PRIV included; every `EnvDecision` validates."""
    tok = K.Tokenizer()
    assert isinstance(tok, CT.Tokenizer)
    tr = K.SelectionTracker()
    toks, n_combat = [], 0
    for t, st in enumerate(R.replay(_run(), tokenizer=tok)):
        d = K.decision_from_state(st, tr)
        if d is None:
            continue
        ed = d.env_decision(run_uid=7, t=t)
        CT.validate_decision(ed)
        row, prow = tok.encode(ed), tok.encode_priv(ed)
        assert set(row) == {
            "cat",
            "cont",
            "sec",
            "tok_flags",
            "mem_tok",
            "mem_class",
            "cls_tok",
            "cls_kind",
            "pool_tok",
            "pool_vid",
            "pool_w",
            "sctx_lo",
            "n_sctx",
        }
        got = _tok_of(row, prow, ed)
        want = K.encode(d)
        assert _same(got.policy_arrays(), want.policy_arrays()), (
            t,
            CT.DECISION_TYPES[d.dtype].name,
        )
        for name in ("pcat", "pcont", "psec", "ppool_tok", "ppool_vid", "ppool_w"):
            assert getattr(got, name).tobytes() == getattr(want, name).tobytes(), name
        toks.append(got)
        n_combat += int(d.is_combat)
    assert n_combat > 10 and len(toks) > n_combat
    b = K.row_batch(toks, run_uid=[7] * len(toks))
    CT.validate_row_batch(b)


class _Raising:
    """A PRIV source whose every read fails: `encode` must never touch it."""

    def true_state(self) -> Any:
        raise AssertionError("encode read d.priv")

    def act_rooms(self) -> Any:
        raise AssertionError("encode read d.priv")

    def future(self) -> Any:
        raise AssertionError("encode read d.priv")


def test_tokenizer_encode_never_reads_priv() -> None:
    import dataclasses

    tok = K.Tokenizer()
    tr = K.SelectionTracker()
    seen = 0
    for t, st in enumerate(R.replay(_run(), tokenizer=tok)):
        d = K.decision_from_state(st, tr)
        if d is None:
            continue
        ed = d.env_decision(run_uid=9, t=t)
        bad = dataclasses.replace(ed, priv=_Raising())
        row = tok.encode(bad)
        assert row["cat"].tobytes() == tok.encode(ed)["cat"].tobytes()
        with pytest.raises(AssertionError):
            K.Tokenizer().encode_priv(bad)
        seen += 1
        if seen >= 40:
            break
    assert seen >= 40


class _KeysOnly:
    """The PRIV source the env once had: `act_rooms()` and `future()` hand back
    each record's keys (`list(record)`), not the record."""

    def __init__(self, p: Any) -> None:
        self._p = p

    def true_state(self) -> Any:
        return self._p.true_state()

    def act_rooms(self) -> Any:
        return list(self._p.act_rooms())

    def future(self) -> Any:
        return [list(r) for r in self._p.future()]


def test_encode_priv_rejects_a_rooms_record_that_is_not_one() -> None:
    """A1: PRIV_RUN read a keys-only `act_rooms()` as an act with no rooms and no
    later bosses, without a sound. A PRIV source that is not a
    `contracts.PrivSource` now raises at the first PRIV_RUN read."""
    import dataclasses

    tr = K.SelectionTracker()
    for t, st in enumerate(R.replay(_run(), tokenizer=K.Tokenizer())):
        d = K.decision_from_state(st, tr)
        if d is None:
            continue
        ed = d.env_decision(run_uid=9, t=t)
        K.Tokenizer().encode_priv(ed)
        with pytest.raises(CT.SchemaError, match="not a rooms record"):
            K.Tokenizer().encode_priv(dataclasses.replace(ed, priv=_KeysOnly(ed.priv)))
        return
    pytest.fail("the recorded run has no decision")


@pytest.mark.parametrize("seed", [11, 12])
def test_tokenizer_in_the_env(seed: int) -> None:
    """The env drives the Tokenizer (`EnvConfig(tokenizer=...)`): its rows over
    `E2EDecision.decision` equal `encode` over the `E2EDecision` itself (the
    env's own intent log and combat view) on a random-play run."""
    import random

    from ppo.e2e import env as E

    tok = K.Tokenizer()
    env = E.E2EEnv(seed, cfg=E.EnvConfig(tokenizer=tok))
    d = env.reset()
    rng = random.Random(seed)
    toks, n_combat = [], 0
    while isinstance(d, E.E2EDecision) and len(toks) < 400:
        ed = d.decision
        CT.validate_decision(ed)
        got = _tok_of(tok.encode(ed), tok.encode_priv(ed), ed)
        want = K.encode(d)
        assert _same(got.policy_arrays(), want.policy_arrays()), (len(toks), d.dtype_name)
        assert got.pcat.tobytes() == want.pcat.tobytes()
        assert got.pcont.tobytes() == want.pcont.tobytes()
        toks.append(got)
        n_combat += int(d.is_combat)
        d = env.step(rng.randrange(d.n_classes))
    assert n_combat >= 10 and len(toks) > n_combat
    CT.validate_row_batch(K.row_batch(toks, run_uid=[seed] * len(toks)))


def test_memos_do_not_change_rows() -> None:
    """The one-entry memos (master deck, relics, true top-10 draw) reuse a part
    only for an equal key: each row from the memos the previous decision left
    equals the row from empty memos, PRIV included, over random-play runs."""
    import random

    from ppo.e2e import env as E

    def fields(tok: K.Tokenizer, ed: Any) -> tuple[np.ndarray, ...]:
        tok._last = None
        t = _tok_of(tok.encode(ed), tok.encode_priv(ed), ed)
        return (*t.policy_arrays(), t.pcat, t.pcont, t.psec, t.ppool_tok, t.ppool_vid, t.ppool_w)

    n = hits = 0
    for seed in range(13, 19):
        tok = K.Tokenizer()
        env = E.E2EEnv(seed, cfg=E.EnvConfig(tokenizer=tok))
        d = env.reset()
        rng = random.Random(seed)
        while isinstance(d, E.E2EDecision):
            ed = d.decision
            warm = K.memo_state()
            got = fields(tok, ed)
            hits += any(K.memo_state().get(k) is v for k, v in warm.items())
            K.memo_restore({})
            assert _same(got, fields(tok, ed)), (seed, n, d.dtype_name)
            n += 1
            d = env.step(rng.randrange(d.n_classes))
        if n >= 150:
            break
    assert n >= 100 and hits >= n // 2, (n, hits)


class _Getters:
    """A combat state with the WP-ENG-1 getters (`feat/e2e-getters` shapes)."""

    encounter_id = "TERROR_EEL_ELITE"

    def telegraph(self) -> list[list[dict[str, Any]]]:
        return [[{"kind": "MultiAttack", "amount": 6, "times": 2}, {"kind": "Buff"}]]

    def peek_intents(self, k: int = 3) -> list[list[dict[str, Any]]]:
        step = {
            "certainty": 0.5,
            "move_id": "CRASH_MOVE",
            "intents": [{"kind": "Attack", "amount": 9}],
        }
        return [[dict(step, step=i + 1) for i in range(k)]]


def test_wp_eng1_getter_columns() -> None:
    """Where the W2 getters exist, the encounter id, telegraph and forecast
    fill their policy columns, and the forecast's move ids reach only
    PRIV_ENEMY."""
    from types import SimpleNamespace

    g = _Getters()
    # `bv.encounter_id` (POLICY_CALLS, schema revision 2); None before the getter.
    assert K._encounter_id(SimpleNamespace(bv=g)) == "TERROR_EEL_ELITE"
    assert K._encounter_id(SimpleNamespace(bv=None)) is None
    assert K._encounter_id(SimpleNamespace(bv=SimpleNamespace())) is None
    assert V.load().id("encounter", "TERROR_EEL_ELITE") > 0

    cat = np.zeros(CT.N_CAT, np.int16)
    cont = np.zeros(CT.N_CONT, np.float32)
    K._intent_cols(cont, cat, 0, g.telegraph(), g.peek_intents(3))
    XC, XF = CT.XC, CT.XF
    assert cont[XF["E_N_INTENTS"]] == np.float32(2 / 3.0)
    assert cat[XC["X_INTENT_T2"]] == CT.INTENT_IDS["Buff"]
    assert cat[XC["X_FC1"]] == cat[XC["X_FC3"]] == CT.INTENT_IDS["Attack"]
    assert cont[XF["E_F2_CERT"]] == np.float32(0.5) and cont[XF["E_F1_TOT"]] > 0
    move = V.load().id("move", "TerrorEel.CRASH_MOVE")
    assert move > 0 and move not in cat

    pc: list[np.ndarray] = []
    pf: list[np.ndarray] = []
    ps: list[int] = []
    K._priv_enemy(g, {"enemies": [{"kind": "TerrorEel"}]}, pc, pf, ps)
    assert ps == [CT.SEC["PRIV_ENEMY"]]
    row = pc[0][0]
    assert row[XC["X_ID_MAIN"]] == row[XC["X_ID_MOD"]] == row[XC["X_ID_REL"]] == move
    assert row[XC["X_ID_AUX"]] == V.load().id("monster", "TerrorEel")
    # A dead enemy keeps a forecast in the getter: no PRIV_ENEMY token for it.
    ps.clear()
    K._priv_enemy(g, {"enemies": [{"kind": "TerrorEel", "is_dead": True}]}, pc, pf, ps)
    assert ps == []


#: The engine's event fights: (event, the fight option's `option_id`, its encounter id).
EVENT_FIGHTS = (
    # v0.111.0 has one encounter class per setting (#1181).
    ("BattlewornDummy", "setting_1", "BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER"),
    ("BattlewornDummy", "setting_2", "BATTLEWORN_DUMMY_EVENT_V2_ENCOUNTER"),
    ("BattlewornDummy", "setting_3", "BATTLEWORN_DUMMY_EVENT_V3_ENCOUNTER"),
    ("DenseVegetation", "rest", "DENSE_VEGETATION_EVENT_ENCOUNTER"),
    ("TheLanternKey", "keep_key", "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER"),
    ("FakeMerchant", "throw_foul", "FAKE_MERCHANT_EVENT_ENCOUNTER"),
)


def event_fight_env(seed: int, event: str, option: str, **cfg: Any) -> Any:
    """An env whose run starts at `event` (forced over Neow) and whose first decision is
    that event; `option` (an `option_id`) starts its fight. FakeMerchant's fight needs a
    Foul Potion on the belt."""
    from ppo.e2e import env as E

    rm = factory.new_run(seed)
    if event == "FakeMerchant":
        rm.debug_set_potion_state(["FOUL_POTION"], [])
    assert rm.force_event(event) == event
    env = E.E2EEnv(seed, rm=rm, cfg=E.EnvConfig(**cfg))
    return env, [o["option_id"] for o in rm.event_option_infos()].index(option)


@pytest.mark.skipif(not hasattr(sts2_rs.CombatState, "encounter_id"), reason="pre-W2 wheel")
@pytest.mark.parametrize(("event", "option", "encounter"), EVENT_FIGHTS)
def test_event_fights_have_an_encounter_id(event: str, option: str, encounter: str) -> None:
    """A3: an event's fight fills RUNCTX's X_ID_AUX with its event encounter's id, not
    `UNK_ENCOUNTER` (vocabulary revision 2.1 appended them)."""
    from ppo.e2e import env as E

    v = V.load()
    env, opt = event_fight_env(162_000_400, event, option)
    d = env.reset()
    assert isinstance(d, E.E2EDecision) and d.dtype == CT.DT["EVENT"]
    d = env.step_action(next(i for i, a in enumerate(d.actions) if a.get("idx") == opt))
    rows = 0
    while isinstance(d, E.E2EDecision) and d.is_combat and rows < 20:
        assert d.bv.encounter_id == encounter
        tok = K.encode(d)
        aux = int(tok.cat[tok.sec == CT.SEC["RUNCTX"], CT.XC["X_ID_AUX"]][0])
        assert aux == v.id("encounter", encounter) != V.UNK_ID["encounter"]
        rows += 1
        d = env.step(0)
    assert rows > 0


def test_tokenizer_run_tracking_is_bounded(monkeypatch) -> None:
    """A long-lived Tokenizer keeps at most `_T_MAX` run uids; dropping one
    drops its run-static entry too, so its next row rebuilds it."""
    monkeypatch.setattr(K, "_T_MAX", 3)
    d = next(d for _st, d in _decisions() if not d.is_combat)
    want = K.encode(d).policy_arrays()
    tok = K.Tokenizer()
    for uid in range(6):
        ed = d.env_decision(run_uid=uid, t=0)
        assert _same(_tok_of(tok.encode(ed), tok.encode_priv(ed), d).policy_arrays(), want)
        assert len(tok._t) <= 3
    assert list(tok._t) == [3, 4, 5]
    assert not any(("uid", u) in K._STATIC for u in range(3))


def test_dead_enemies_get_no_forecast_columns() -> None:
    """The telegraph and forecast columns of a dead enemy's ENEMY token stay
    zero (`peek_intents` keeps a dead enemy's forecast)."""
    import copy

    from ppo.combat_ctx import tokens as T

    d = next(d for _st, d in _decisions() if d.is_combat and d.snap["enemies"])
    n = len(d.snap["enemies"])
    g = _Getters()
    tele, fc = g.telegraph() * n, g.peek_intents(3) * n
    tk = T.tokens(d.snap, d.obs_actions, sts2_rs.rollout_features(d.bv, version=1), "S_ctx")
    em = np.flatnonzero(tk["sec"] == CT.SEC["ENEMY"])
    assert len(em) == n
    fc_cat = [CT.XC[f"X_FC{k}"] - K.NS_CAT for k in (1, 2, 3)]
    fc_cont = [CT.XF[f"E_F{k}_CERT"] - K.NS_CONT for k in (1, 2, 3)]

    def forecast(snap: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xc, xf = K._sctx_trailing(
            tk["cat"], tk["sec"], int(tk["n_act"]), d, snap, d.d_bv, [], tele, fc
        )
        return xc[em][:, fc_cat], xf[em][:, fc_cont]

    live_c, live_f = forecast(d.snap)
    assert (live_c > 0).all() and (live_f > 0).all()
    dead = copy.deepcopy(d.snap)
    dead["enemies"][0]["is_dead"] = True
    dead_c, dead_f = forecast(dead)
    assert not dead_c[0].any() and not dead_f[0].any()
    assert (dead_c[1:] == live_c[1:]).all() and (dead_f[1:] == live_f[1:]).all()


def test_priv_draw_reuse_matches_fresh_rows() -> None:
    """PRIV_DRAW rows carried over from the previous window equal fresh rows:
    a shift with new cards below, a shorter pile, a one-card pile, no
    overlap, and repeated identical cards."""
    ids = ["StrikeIronclad", "DefendIronclad", "Bash"]
    pile = [
        {"id": ids[i % 3], "upgraded": i % 4 == 0, "base_cost": 1 + i % 2, "base_damage": 6 + i}
        for i in range(16)
    ]
    same = [dict(pile[0]) for _ in range(12)]
    for prev, news in (
        (pile[:10], (pile[2:12], pile[3:10], pile[9:10], pile[10:16], pile[1:11])),
        (same[:10], ([*same[:9], pile[5]], [*same[:8], pile[1], pile[2]], same[:4])),
    ):
        pcat, pcont = K._priv_draw_rows(prev, None)
        for new in news:
            got = K._priv_draw_after(prev, pcat, pcont, new, None)
            want = K._priv_draw_rows(new, None)
            assert got[0].tobytes() == want[0].tobytes(), len(new)
            assert got[1].tobytes() == want[1].tobytes(), len(new)


def test_the_memos_hit_per_run_when_slots_interleave(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker encodes its env slots round-robin, so consecutive `encode` calls come
    from different runs. The one-entry memos (deck, relics, PRIV_DRAW) are kept per
    run, so each run hits its own entries as if it ran alone (review finding TOK-1:
    process-global memos missed on nearly every interleaved call, about 11% of the
    encode time, and X-T, which replays one run, credited them anyway)."""
    from collections import Counter

    from ppo.e2e import env as E

    calls: Counter = Counter()
    for name in ("_deck_summary", "_relic_counts", "_priv_draw_rows", "_priv_draw_after"):
        real = getattr(K, name)

        def spy(*a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
            calls[_name] += 1
            return _real(*a, **kw)

        monkeypatch.setattr(K, name, spy)

    def play(seeds: list[int], steps: int) -> tuple[Counter, list[K.Tok]]:
        K.clear_cache()
        K.memo_restore({})
        calls.clear()
        envs = [E.E2EEnv(s, cfg=E.EnvConfig(f9_p=0.0)) for s in seeds]
        ds = [e.reset() for e in envs]
        toks = []
        for _ in range(steps):
            for i, d in enumerate(ds):  # round-robin, as `actors.worker_main` does
                if isinstance(d, E.E2EDecision):
                    toks.append(K.encode(d, priv=True))
                    ds[i] = envs[i].step(0)
        return Counter(calls), toks

    alone = Counter()
    for s in (162_000_211, 162_000_212):
        alone.update(play([s], 60)[0])
    both, toks = play([162_000_211, 162_000_212], 60)
    assert sum(alone.values()) > 0 and both == alone, (both, alone)
    assert len(toks) > 100


# ── T14 on the S0 path: the student's logits over permuted real rows ──────


def _student(bf16: bool) -> Any:
    """A small student with O(1) policy logits (test_net's `_loud_heads`), so that
    equality means something; `bf16` as the learner's `net_inf.bf16` (CPU autocast
    here: no test needs a GPU)."""
    import torch
    from ppo.e2e import net as N

    cfg = N.NetConfig(num_hidden=64, dim=64, n_layers=2, ffn=176, head_hidden=64)
    net = N.E2ENet(cfg, seed=0)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for head in (net.head_combat, net.head_ooc):
            w = head.out.weight
            w.copy_(torch.randn(w.shape, generator=g) * 3.0 / w.shape[1] ** 0.5)
    net.bf16 = bf16
    return net.eval()


def _s0_logits(net: Any, toks: list[K.Tok]) -> Any:
    """`Learner._serve`'s forward: `net(to_batch(rb))`."""
    import torch
    from ppo.e2e import net as N

    rb = K.row_batch(toks)
    CT.validate_row_batch(rb)
    with torch.no_grad():
        return net(N.to_batch(rb)).logits


def _priv_bytes(t: K.Tok) -> bytes:
    return t.pcat.tobytes() + t.pcont.tobytes() + t.psec.tobytes() + t.ppool_vid.tobytes()


@pytest.mark.parametrize("bf16", [False, True])
def test_t14_a_re_deal_leaves_the_students_logits_bit_identical(bf16: bool) -> None:
    """The draw order and combat RNG (`sample_world`) move the PRIV arrays of real
    rows (asserted, so the check is not vacuous); the student's S0 logits over the
    re-encoded rows stay `torch.equal`."""
    import torch
    from ppo.e2e import f7check as F7

    base, moved = [], []
    for st, d in _decisions():
        if not d.is_combat:
            continue
        K.clear_cache()
        b = K.encode(d)  # PRIV included: what the learner stores
        for k in (1, 2):
            _w, dw = F7.world_decision(st, d, 9000 + k)
            K.clear_cache()
            base.append(b)
            moved.append(K.encode(dw))
    changed = sum(_priv_bytes(a) != _priv_bytes(b) for a, b in zip(base, moved, strict=True))
    assert len(base) > 20 and changed >= 0.7 * len(base), (changed, len(base))
    net = _student(bf16)
    assert torch.equal(_s0_logits(net, base), _s0_logits(net, moved))


@pytest.mark.parametrize("bf16", [False, True])
def test_t14_hidden_rooms_and_bosses_leave_the_students_logits_bit_identical(bf16: bool) -> None:
    import dataclasses

    import torch

    a, b = [], []
    for _st, d in _decisions():
        K.clear_cache()
        a.append(K.encode(d))
        K.clear_cache()
        b.append(K.encode(dataclasses.replace(d, rm=_HiddenRooms(d.rm))))
    assert sum(_priv_bytes(x) != _priv_bytes(y) for x, y in zip(a, b, strict=True)) > 0
    net = _student(bf16)
    assert torch.equal(_s0_logits(net, a), _s0_logits(net, b))


@pytest.mark.parametrize("bf16", [False, True])
def test_t14_dropping_priv_leaves_the_students_logits_bit_identical(bf16: bool) -> None:
    """An n_priv change (here: no PRIV at all) does not reach the trunk's shapes."""
    import torch

    full, bare = [], []
    for _st, d in _decisions()[:60]:
        K.clear_cache()
        full.append(K.encode(d))
        K.clear_cache()
        bare.append(K.encode(d, priv=False))
    assert any(t.n_priv for t in full) and not any(t.n_priv for t in bare)
    net = _student(bf16)
    assert torch.equal(_s0_logits(net, full), _s0_logits(net, bare))
