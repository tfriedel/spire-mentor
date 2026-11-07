"""GH #830 step 0: the combat context probe's honesty, data path, trainer and metrics.

A hand-made fight (the starter deck plus Pommel Strike against Nibbits, as in
`test_deepbets_value_label`) stands in for a bank fight. DESIGN §g numbers the
honesty tests; each test below names the one it implements.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import random
import re
from collections.abc import Iterator

import numpy as np
import pytest
import sts2_client
import sts2_rs
from ppo import rollout_starts
from ppo.combat_bench import agents
from ppo.combat_ctx import data as D
from ppo.combat_ctx import gen as G
from ppo.combat_ctx import metrics as M
from ppo.combat_ctx import play as P
from ppo.combat_ctx import power as PW
from ppo.combat_ctx import static_tables as ST
from ppo.combat_ctx import tokens as T
from ppo.combat_ctx import train as TR
from ppo.combat_ctx.intent_log import IntentLog
from ppo.combat_ctx.snapshot import (
    action_classes,
    honest_snapshot,
    obs_actions,
    snapshot_from_parts,
    whitelist_violations,
)
from ppo.deepbets import value_data as vd

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
DECOMP = os.path.join(vd.REPO_SUBDIR, "sts2-wiki", "decompiled", "v0.107.1")


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


def _walk(seed: int = 7, n: int = 30) -> list:
    """`(cs, acts)` along a fixed policy (first card play, else first action)."""
    rm = rollout_starts.build_fight(FIGHT, seed)
    cs = rm.combat_state()
    out = []
    rng = random.Random(seed)
    while len(out) < n:
        acts = cs.legal_actions()
        if not acts:
            break
        out.append((cs.clone(), acts))
        cs.apply(acts[rng.randrange(len(acts))])
    return out


def _snap(cs, room: str = "elite") -> dict:
    return honest_snapshot(cs.belief_view(0), room)


def _rl2(cs) -> dict:
    rf = sts2_rs.rollout_features(cs, version=1)
    return {"state": rf["state"], "x": rf["x"], "keys": rf["keys"]}


def _toks(snap: dict, cs, acts, arm: str = "S_ctx") -> dict:
    bv = cs.belief_view(0)
    a = obs_actions(acts, cs.pending_choice(), bv.pending_choice())
    return T.tokens(snap, a, _rl2(cs), arm)


def _read(path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _same_tokens(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in ("cat", "cont", "sec"))


# ── test 1 / 2a: the snapshot reads only bv ──────────────────────────────


def test_snapshot_reads_only_belief_view() -> None:
    states = _walk(seed=11, n=25)
    for cs, acts in states:
        base = _snap(cs)
        base["ehist"] = []
        toks = _toks(base, cs, acts)
        hashes = set()
        for r in range(4):
            w = cs.sample_world(r)
            hashes.add(int(w.ordered_hash()))
            s2 = _snap(w)
            s2["ehist"] = []
            assert s2 == base
            assert _same_tokens(_toks(s2, w, acts), toks)
    # Control: the worlds really differ somewhere along the walk.
    assert any(
        len({int(cs.sample_world(r).ordered_hash()) for r in range(4)}) > 1 for cs, _ in states
    )


def leaky_snapshot(cs) -> dict:
    """A snapshot that reads the TRUE draw order (negative control 2a)."""
    d = sts2_rs.state_to_dict(cs)
    return {"draw": [c["id"] for c in d["deck"]["draw_pile"]]}


def test_leaky_snapshot_fails_test_1() -> None:
    hit = n = 0
    for cs, _ in _walk(seed=3, n=25):
        d = sts2_rs.state_to_dict(cs.belief_view(0))
        known = set(d["deck"].get("pinned_top", [])) | set(d["deck"].get("pinned_bottom", []))
        if len(d["deck"]["draw_pile"]) - len(known) < 2:
            continue
        ids = [c["id"] for c in d["deck"]["draw_pile"]]
        if len(set(ids)) < 2:
            continue
        n += 1
        base = leaky_snapshot(cs)
        hit += any(leaky_snapshot(cs.sample_world(r)) != base for r in range(4))
    assert n > 0 and hit / n >= 0.9


# ── test 2b / 2c / 3 / 9: whitelist and provenance ──────────────────────


def _snap_with_move_state(cs) -> dict:
    """Negative control 2b: AI state copied in under its own key."""
    bv = cs.belief_view(0)
    s = _snap(cs)
    d = sts2_rs.state_to_dict(bv)
    for m, e in zip(d["encounter"], s["enemies"], strict=True):
        e["move_state"] = m.get("move_state")
    return s


def test_whitelist_and_negative_control_2b() -> None:
    for cs, _ in _walk(seed=5, n=10):
        s = _snap(cs)
        assert whitelist_violations(s) == []
        leaky = _snap_with_move_state(cs)
        assert any("move_state" in v for v in whitelist_violations(leaky))  # test 3 fails
        assert _snap_with_move_state(cs.sample_world(1)) == leaky  # test 1 passes


def _fuzz(d: dict, rng: random.Random) -> dict:
    """Every non-whitelisted field randomized (same type), uids relabelled."""
    d = copy.deepcopy(d)

    def rnd(v):
        if isinstance(v, bool):
            return rng.random() < 0.5
        if isinstance(v, int):
            return rng.randrange(1, 10**6)
        if isinstance(v, float):
            return rng.random()
        if isinstance(v, str):
            return f"X{rng.randrange(10**6)}"
        if isinstance(v, dict):
            return {k: rnd(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rnd(x) for x in v]
        return v

    for k in ("rng", "rng_counters", "combat_id", "side", "wire_version", "turn", "phase"):
        if k in d:
            d[k] = rnd(d[k])
    for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile", "play_pile"):
        for c in d["deck"].get(pile, []):
            c["stable_idx"] = rng.randrange(10**6)
            if "combat_vars" in c:
                c["combat_vars"] = rnd(c["combat_vars"])
    uids = [m["uid"] for m in d["encounter"]]
    perm = rng.sample(range(1000, 2000), len(uids))
    for m, u in zip(d["encounter"], perm, strict=True):
        m["uid"] = u
        m["move_state"] = f"MOVE{rng.randrange(99)}"
        m["intent_history"] = [{"kind": "Attack"}]
    d["player"]["gold"] = rng.randrange(999)
    return d


def test_provenance_fuzz_9_and_negative_control_2c() -> None:
    rng = random.Random(0)
    leaked = 0
    for cs, acts in _walk(seed=9, n=12):
        bv = cs.belief_view(0)
        d = sts2_rs.state_to_dict(bv)
        parts = (bv.card_damage_preview(), bv.card_block_preview(), bv.pending_choice(), "elite")
        base = snapshot_from_parts(d, *parts)
        base["ehist"] = []
        fz = snapshot_from_parts(_fuzz(d, rng), *parts)
        fz["ehist"] = []
        assert _same_tokens(_toks(base, cs, acts), _toks(fz, cs, acts))

        # 2c: a value-level leak of the AI state through a whitelisted field.
        def leak(dd: dict, parts: tuple = parts) -> dict:
            s = snapshot_from_parts(dd, *parts)
            ms = "|".join(str(m.get("move_state")) for m in dd["encounter"])
            h = int(__import__("hashlib").sha1(ms.encode()).hexdigest(), 16) % 997 + 1
            s["powers"] = [
                *s["powers"],
                {"name": "STRENGTH", "amount": 0, "display_amount": h, "counter": 0},
            ]
            s["ehist"] = []
            return s

        assert whitelist_violations(leak(d)) == []  # test 3 passes
        assert _snap(cs.sample_world(2)) == _snap(cs)  # test 1 passes
        leaked += not _same_tokens(_toks(leak(d), cs, acts), _toks(leak(_fuzz(d, rng)), cs, acts))
    assert leaked > 0  # test 9 catches it


# ── test 4: rl2 features are honest ──────────────────────────────────────


def test_rl2_features_invariant_under_sample_world() -> None:
    for cs, _ in _walk(seed=13, n=20):
        base = sts2_rs.rollout_features(cs, version=1)
        for r in range(3):
            o = sts2_rs.rollout_features(cs.sample_world(r), version=1)
            assert (o["state"], o["x"], o["keys"]) == (base["state"], base["x"], base["keys"])


# ── test 5: draw-pile choice options ─────────────────────────────────────


def test_choice_options_drop_non_hand_indices_and_remap() -> None:
    cs = _walk(n=1)[0][0]
    d = sts2_rs.state_to_dict(cs.belief_view(0))
    choice = {
        "kind": "add_from_draw_to_hand",
        "min_select": 1,
        "max_select": 1,
        "can_skip": False,
        "cards": [
            {"id": "BASH", "pile": "draw", "index": 4, "type": "Attack", "upgraded": False},
            {
                "id": "STRIKE_IRONCLAD",
                "pile": "hand",
                "index": 2,
                "type": "Attack",
                "upgraded": False,
            },
            {
                "id": "DEFEND_IRONCLAD",
                "pile": "draw",
                "index": 1,
                "type": "Skill",
                "upgraded": False,
            },
        ],
    }
    s = snapshot_from_parts(d, [], [], choice, "elite")
    for o in s["choice"]["options"]:
        assert ("hand_index" in o) == (o["pile"] == "hand")
        assert "index" not in o
    # The true state lists the draw options in another order than bv.
    cs_choice = copy.deepcopy(choice)
    cs_choice["cards"] = [choice["cards"][2], choice["cards"][1], choice["cards"][0]]
    acts = [{"action": "choose", "index": i, "selection": [i]} for i in range(3)]
    mapped = obs_actions(acts, cs_choice, choice)
    assert [a["selection"] for a in mapped] == [[2], [1], [0]]


# ── test 6: classes ──────────────────────────────────────────────────────


def test_action_classes_merge_identical_copies_only() -> None:
    for cs, acts in _walk(seed=17, n=8):
        bv = cs.belief_view(0)
        cls = action_classes(bv, acts)
        d = sts2_rs.state_to_dict(bv)
        hand = d["deck"]["hand"]
        for i, a in enumerate(acts):
            for j, b in enumerate(acts):
                if a.get("action") == b.get("action") == "play":
                    same_card = (
                        hand[a["card_index"]]["id"] == hand[b["card_index"]]["id"]
                        and hand[a["card_index"]]["upgraded"] == hand[b["card_index"]]["upgraded"]
                    )
                    same_t = a.get("target_index") == b.get("target_index")
                    if same_card and same_t:
                        assert cls[i] == cls[j]
                    if not same_t:
                        assert cls[i] != cls[j]
                    if hand[a["card_index"]]["id"] != hand[b["card_index"]]["id"]:
                        assert cls[i] != cls[j]
    # Class visits sum, class Q is visit-weighted over visited members.
    v, q = D.class_stats(
        np.array([0, 0, 1]), np.array([10.0, 0.0, 5.0]), np.array([0.5, np.nan, 0.2])
    )
    assert v.tolist() == [10.0, 5.0] and np.allclose(q, [0.5, 0.2])
    v, q = D.class_stats(np.array([0, 0]), np.array([3.0, 1.0]), np.array([1.0, 0.0]))
    assert np.allclose(q, [0.75])


def test_choose_options_a_player_cannot_tell_apart_are_one_class() -> None:
    """Putting back either of two identical discard Strikes: two observation
    hashes (the pinned card keeps its uid), one class, one token."""
    fight = {**FIGHT, "deck": ["STRIKE_IRONCLAD"] * 6 + ["HEADBUTT"] * 4 + ["DEFEND_IRONCLAD"] * 2}
    n_dup = 0
    for seed in range(1, 16):
        cs = rollout_starts.build_fight(fight, seed).combat_state()
        rng = random.Random(seed)
        for _ in range(60):
            acts = cs.legal_actions()
            if not acts:
                break
            pc = cs.pending_choice()
            if pc is not None and pc["kind"] == "put_back":
                bv = cs.belief_view(0)
                a2 = obs_actions(acts, pc, bv.pending_choice())
                cls = action_classes(bv, a2)
                opts = bv.pending_choice()["cards"]
                vis = []
                hashes = set()
                for a in a2:
                    o = opts[a["selection"][0]]
                    vis.append((o["id"], bool(o.get("upgraded")), o.get("pile")))
                    c = bv.clone()
                    c.apply(a)
                    hashes.add(int(c.observation_hash()))
                for i in range(len(a2)):
                    for j in range(len(a2)):
                        assert (cls[i] == cls[j]) == (vis[i] == vis[j]), (vis, cls)
                if len(set(vis)) < len(vis):
                    n_dup += 1
                    assert len(hashes) > len(set(cls))  # the hash alone would split them
                    tk = _toks(_snap(cs), cs, acts)
                    rows = tk["cat"][-len(acts) :]
                    for i in range(len(a2)):
                        for j in range(len(a2)):
                            if np.array_equal(rows[i], rows[j]):
                                assert cls[i] == cls[j]
            cs.apply(acts[rng.randrange(len(acts))])
    assert n_dup >= 2


# ── test 7: split and leakage ────────────────────────────────────────────


def test_split_schemes() -> None:
    names = [f"{i:016x}.run.gz" for i in range(3000)]
    for n in names[:200]:
        assert G.fold_of(n) == int(__import__("hashlib").sha1(n.encode()).hexdigest(), 16) % 10
        assert vd.split_of(n) == G.split_of(n, "S1")
    assert G.SCHEMES["S1"] == {0: "test", 1: "dev", **{f: "train" for f in range(2, 10)}}
    assert G.SCHEMES["S2"] == {0: "test", 1: "test", 2: "dev", **{f: "train" for f in range(3, 10)}}
    # Neither smoke nor pilot ever plays a fold that is TEST under S1 or S2.
    assert not {0, 1} & (set(G.SCHEMES["SMOKE"]) | set(G.SCHEMES["PILOT"]))


def test_jobs_hold_out_every_holdout_run() -> None:
    with open(G.BOSS800, encoding="utf-8") as f:
        b8 = [json.loads(line) for line in f][:5]
    bench = sorted(G.holdout_runs())[:5]
    rows = []
    for k, f in enumerate([f"{i:016x}.run.gz" for i in range(400)]):
        rows.append(
            {
                **FIGHT,
                "file": f,
                "floor": 1 + k % 3,
                "room_type": ("boss", "elite", "monster")[k % 3],
            }
        )
    rows += [{**FIGHT, "file": r["file"], "room_type": "elite"} for r in b8]
    rows += [{**FIGHT, "file": b, "room_type": "boss"} for b in bench]
    rows += [{**FIGHT, "file": "wg0.run.gz", "encounter": G.WG, "room_type": "boss"},
             {**FIGHT, "file": "wg0.run.gz", "room_type": "elite"}]  # fmt: skip
    held = G.holdout(rows)
    assert "wg0.run.gz" in held["wg"]
    for scheme in ("S1", "S2"):
        jobs = G.build_jobs(rows, scheme, held["all"])
        files = {j["row"]["file"] for j in jobs}
        assert not files & held["all"]
        by = {
            s: {j["row"]["file"] for j in jobs if j["split"] == s} for s in ("train", "dev", "test")
        }
        assert (
            not (by["train"] & by["dev"])
            and not (by["train"] & by["test"])
            and not (by["dev"] & by["test"])
        )
        for j in jobs:
            assert j["split"] == G.split_of(j["row"]["file"], scheme)
            assert j["fold"] == G.fold_of(j["row"]["file"])
            if j["row"]["room_type"] == "monster":
                assert j["split"] == "train"
            assert 0 <= j["rep"] <= 1
        assert {j["rep"] for j in jobs if j["split"] != "train"} == {0}
    assert all(j["split"] == "test" for j in G.build_jobs(rows, "S1", held["all"], extension_rep=2))


# ── test 8: driver logit parity ─────────────────────────────────────────


def test_rl2_torch_twin_matches_rust_logits() -> None:
    rust, twin = P.Rl2Explain(None), P.Rl2Torch(None)
    n = 0
    for cs, acts in _walk(seed=21, n=12):
        if len(acts) < 2:
            continue
        a = rust.logits([({}, cs, acts)])[0]
        b = twin.logits([({}, cs, acts)])[0]
        assert np.max(np.abs(a - b)) <= 1e-4
        n += 1
        if n >= 5:
            break
    assert n == 5


# ── test 10: the token builder's inputs ──────────────────────────────────


def _record(seed: int = 7, split: str = "dev", rep: int = 0) -> dict:
    job = {"row": FIGHT, "idx": 0, "rep": rep, "split": split, "fold": G.fold_of(FIGHT["file"])}
    return G.play_job(job)


@pytest.fixture(scope="module")
def played() -> dict:
    return _record()


def test_record_inputs_cannot_carry_search_or_outcome(played: dict) -> None:
    rec = played["decisions"][0]
    import inspect

    assert list(inspect.signature(T.tokens).parameters) == ["obs", "actions", "rl2feat", "arm"]
    base = T.tokens(*D.record_inputs(rec), "S_ctx")
    poisoned = copy.deepcopy(rec)
    poisoned["z"] = {"value": 99.0, "won": True}
    poisoned["A"] = {k: None for k in rec["A"]}
    poisoned["B"] = {"visits": [7] * 9}
    poisoned["ohash"] = 1
    poisoned["rl2"]["logits"] = [123.0] * len(rec["actions"])
    assert _same_tokens(base, T.tokens(*D.record_inputs(poisoned), "S_ctx"))


def test_record_round_trip_and_featurizer_determinism(played: dict) -> None:
    for rec in played["decisions"]:
        a = T.tokens(*D.record_inputs(rec), "S_ctx")
        back = json.loads(json.dumps(rec))
        b = T.tokens(*D.record_inputs(back), "S_ctx")
        assert _same_tokens(a, b)
        n_act = len(rec["actions"])
        assert a["cat"].shape == (len(a["sec"]), T.N_CAT) and a["cont"].shape == (
            len(a["sec"]),
            T.N_CONT,
        )
        assert a["sec"][0] == T.SEC["STATE"]
        assert all(s in T.ACT_SECTIONS for s in a["sec"][-n_act:])
        assert not any(s in T.ACT_SECTIONS for s in a["sec"][:-n_act])
        assert a["n_oov"] == 0
        for arm in T.ARMS:
            got = T.derive(a["cat"], a["cont"], a["sec"], n_act, arm)
            want = T.tokens(*D.record_inputs(rec), arm)
            assert all(
                np.array_equal(x, y)
                for x, y in zip(got, (want["cat"], want["cont"], want["sec"]), strict=True)
            )


def test_gen_record_shape_and_asserts(played: dict) -> None:
    assert "assert" not in played and played["traj"]["end"] in ("won", "died")
    assert played["traj"]["seed"] == G.seed_of("unit.run.gz:3", 0)
    for rec in played["decisions"]:
        n = len(rec["actions"])
        assert n >= 2 and sum(rec["A"]["visits"]) > 0
        assert len(rec["cls"]) == n == len(rec["rl2"]["x"]) == len(rec["rl2"]["logits"])
        assert rec["B"] is not None and rec["B"]["seed"] == G.shadow_seed(rec["seed"], rec["t"])
        assert rec["A"]["unmatched"] == 0 and 0 <= rec["A"]["rec"] < n
        assert rec["z"]["end"] == played["traj"]["end"]
    # The traj record rebuilds every decision's true state, hash-checked.
    ts = [r["t"] for r in played["decisions"]]
    states, verified = vd.walk_record(played["traj"], ts)
    assert verified >= max(ts) + 1
    for r in played["decisions"]:
        assert int(states[r["t"]].ordered_hash()) == r["ohash"]


# ── test 11 / 13: static tables ──────────────────────────────────────────


@pytest.mark.skipif(not os.path.isdir(DECOMP), reason="decomp absent")
def test_static_tables_match_the_decomp() -> None:
    from ppo import vocab

    # An override that returns a literal `false` (BoneTea, LavaRock,
    # TeaOfDiscourtesy) shows no counter either.
    never = re.compile(r"override bool ShowCounter\s*=>\s*false\s*;")
    rel = {
        os.path.basename(p)[:-3]
        for p in glob.glob(f"{DECOMP}/*.Sts2.Core.Models.Relics/*.cs")
        if "override bool ShowCounter" in _read(p) and not never.search(_read(p))
    }
    rcanon = {vocab.canonical(n): n for n in sts2_rs.list_registry_names("relic")}
    assert {rcanon[vocab.canonical(c)] for c in rel} == ST.SHOW_COUNTER_RELICS
    assert len(ST.SHOW_COUNTER_RELICS) == 36
    assert not {"BONE_TEA", "LAVA_ROCK", "TEA_OF_DISCOURTESY"} & ST.SHOW_COUNTER_RELICS
    pat = re.compile(r":\s*base\(\s*-?\d+\s*,\s*CardType\.(\w+)")
    canon = {}
    for n in sts2_rs.list_registry_names("card"):
        canon.setdefault(vocab.canonical(n), n)
    want = {}
    for p in glob.glob(f"{DECOMP}/*.Sts2.Core.Models.Cards/*.cs"):
        cls = os.path.basename(p)[:-3]
        if cls in ST.SKIPPED_CLASSES:
            continue
        name = ST.CLASS_ALIASES.get(cls) or canon[vocab.canonical(cls)]
        want[name] = pat.search(_read(p)).group(1)
    assert want == ST.CARD_TYPE


def test_non_show_counter_relic_counter_is_dropped() -> None:
    cs, acts = _walk(n=1)[0]
    bv = cs.belief_view(0)
    d = sts2_rs.state_to_dict(bv)
    parts = (bv.card_damage_preview(), bv.card_block_preview(), bv.pending_choice(), "elite")
    d1 = copy.deepcopy(d)
    for r in d1["player"]["relics"]:
        if r["id"] == "CENTENNIAL_PUZZLE":
            r["counter"] = 1
    d0 = copy.deepcopy(d)
    for r in d0["player"]["relics"]:
        r.pop("counter", None)
    assert "CENTENNIAL_PUZZLE" not in ST.SHOW_COUNTER_RELICS
    s1, s0 = snapshot_from_parts(d1, *parts), snapshot_from_parts(d0, *parts)
    s1["ehist"] = s0["ehist"] = []
    assert _same_tokens(_toks(s1, cs, acts), _toks(s0, cs, acts))
    d2 = copy.deepcopy(d1)
    d2["player"]["relics"].append({"id": "PEN_NIB", "counter": 4})
    assert any(r.get("counter") == 4 for r in snapshot_from_parts(d2, *parts)["relics"])
    # A ShowCounter=false override keeps the engine's live counter out too.
    d3 = copy.deepcopy(d)
    d3["player"]["relics"].append({"id": "BONE_TEA", "counter": 1})
    assert [r for r in snapshot_from_parts(d3, *parts)["relics"] if r["id"] == "BONE_TEA"] == [
        {"id": "BONE_TEA"}
    ]


# The counter C# shows (`DisplayAmount`, v0.107.1) on a ShowCounter relic:
# its largest value at rest (the `IsActivating` flash aside). None = no bound.
DISPLAY_MAX = {
    "PEN_NIB": 9, "NUNCHAKU": 9, "TUNING_FORK": 9, "GALACTIC_DUST": 9, "HAPPY_FLOWER": 2,
    "PENDULUM": 2, "FAKE_HAPPY_FLOWER": 4, "POLLINOUS_CORE": 3, "IRON_CLUB": 3,
    "BOOK_OF_FIVE_RINGS": 4, "JOSS_PAPER": 4, "LASTING_CANDY": 1, "FISHING_ROD": 2, "KUNAI": 2,
    "SHURIKEN": 2, "ORNAMENTAL_FAN": 2, "KUSARIGAMA": 2, "LETTER_OPENER": 2, "TOY_BOX": 2,
    "SWORD_OF_STONE": 5, "SILVER_CRUCIBLE": 3, "WONGOS_MYSTERY_TICKET": 5, "PAELS_LEGION": 2,
    "BRILLIANT_SCARF": 4, "METRONOME": 6, "EMBER_TEA": 5, "GIRYA": 3, "VELVET_CHOKER": 6,
    "POCKETWATCH": None, "PUMPKIN_CANDLE": None,
}  # fmt: skip


def test_relic_counters_are_the_displayed_numbers() -> None:
    from ppo.combat_ctx.snapshot import RELIC_COUNTER_DISPLAY, display_counter

    assert set(RELIC_COUNTER_DISPLAY) == ST.SHOW_COUNTER_RELICS
    shown = {k for k, v in RELIC_COUNTER_DISPLAY.items() if v is not None}
    assert shown == set(DISPLAY_MAX)
    # Engine counters seen in the smoke -> C#'s DisplayAmount.
    cases = {
        ("POCKETWATCH", 515): 3,  # (2 cards last turn << 8) | 3 this turn
        ("POCKETWATCH", 768): 0,
        ("PEN_NIB", 40): 0,  # AttacksPlayed % 10
        ("PEN_NIB", 13): 3,
        ("PEN_NIB", None): 0,  # the wire omits a zero counter; C# shows 0
        ("NUNCHAKU", 23): 3,
        ("HAPPY_FLOWER", 7): 1,
        ("PENDULUM", 2): 2,  # the engine stores TurnsSeen % 3, like C#
        ("POLLINOUS_CORE", 4): 0,  # reset by the turn-4 hand draw
        ("SHURIKEN", 5): 2,
        ("SWORD_OF_STONE", 5): 0,  # 5 elites left = 0 defeated
        ("PAELS_LEGION", 2): 2,
        ("PAELS_LEGION", -1): None,
        ("PAELS_LEGION", None): None,
        ("SILVER_CRUCIBLE", (1 << 8) | 1): 2,  # (treasure rooms << 8) | times used
        ("SILVER_CRUCIBLE", 3): None,
        ("WONGOS_MYSTERY_TICKET", 2): 3,
        ("WONGOS_MYSTERY_TICKET", -1): None,
        ("BRILLIANT_SCARF", 5): None,
        ("TOY_BOX", 12): None,
        ("DIAMOND_DIADEM", 3): None,
        ("CENTENNIAL_PUZZLE", 1): None,
        ("BONE_TEA", 1): None,
    }
    for (rid, c), want in cases.items():
        assert display_counter(rid, c) == want, (rid, c)


def test_relic_counters_lie_in_the_displayed_range() -> None:
    """Real engine states: every snapshot counter is in C#'s displayed range,
    while the raw counters leave it (the control)."""
    relics = ["BURNING_BLOOD", "POCKETWATCH", "PEN_NIB", "NUNCHAKU", "HAPPY_FLOWER", "PENDULUM",
              "KUNAI", "SHURIKEN", "ORNAMENTAL_FAN", "KUSARIGAMA", "LETTER_OPENER", "TUNING_FORK",
              "POLLINOUS_CORE", "IRON_CLUB", "SWORD_OF_STONE", "EMBER_TEA", "PAELS_LEGION",
              "FAKE_HAPPY_FLOWER", "JOSS_PAPER", "GIRYA"]  # fmt: skip
    fight = {**FIGHT, "relics": relics, "encounter": "THE_KIN_BOSS", "room_type": "boss",
             "hp_before": 80, "deck": [*FIGHT["deck"], "SHRUG_IT_OFF", "TRUE_GRIT"]}  # fmt: skip
    out_of_range: set[str] = set()
    pendulum_counters: set[int] = set()
    n = 0
    for seed in (7, 8):
        cs = rollout_starts.build_fight(fight, seed).combat_state()
        rng = random.Random(seed)
        for _ in range(300):
            acts = cs.legal_actions()
            if not acts:
                break
            bv = cs.belief_view(0)
            for r in sts2_rs.state_to_dict(bv)["player"]["relics"]:
                hi = DISPLAY_MAX.get(r["id"])
                if hi is not None and not 0 <= r.get("counter", 0) <= hi:
                    out_of_range.add(r["id"])
                if r["id"] == "POCKETWATCH" and r.get("counter", 0) > 0xFF:
                    out_of_range.add(r["id"])
                if r["id"] == "PENDULUM":
                    pendulum_counters.add(r.get("counter", 0))
            for r in honest_snapshot(bv, "boss")["relics"]:
                if "counter" in r:
                    hi = DISPLAY_MAX[r["id"]]
                    assert 0 <= r["counter"] <= (0xFF if hi is None else hi), r
                    n += 1
            plays = [a for a in acts if a.get("action") == "play"]
            pick = plays if plays and rng.random() < 0.85 else acts
            cs.apply(pick[rng.randrange(len(pick))])
    assert n > 0
    assert {"POCKETWATCH", "HAPPY_FLOWER"} <= out_of_range
    # Pendulum's engine counter IS C#'s TurnsSeen (wrapped at 3 on every tick,
    # both builds), so its raw counter must stay in range and cycle through it.
    assert "PENDULUM" not in out_of_range
    assert pendulum_counters == {0, 1, 2}


def test_card_type_agrees_with_rl2_features() -> None:
    names = ("Attack", "Skill", "Power")
    for cs, acts in _walk(seed=23, n=20):
        d = sts2_rs.state_to_dict(cs)
        rf = sts2_rs.rollout_features(cs, version=1)
        for a, x in zip(rf["actions"], rf["x"], strict=True):
            if a.get("action") != "play":
                continue
            c = d["deck"]["hand"][a["card_index"]]
            onehot = [int(v) for v in x[8:12]]
            t = ST.CARD_TYPE[c["id"]]
            want = [int(t == n) for n in names] + [int(t in ("Status", "Curse"))]
            assert onehot == want, (c["id"], onehot, t)
            assert (c["kind"] == "status") == (t == "Status") and (c["kind"] == "curse") == (
                t == "Curse"
            )


# ── test 12: intent log ──────────────────────────────────────────────────


def _enemy(eid: int, kind: str, intents: list) -> dict:
    return {"eid": eid, "kind": kind, "intents": intents, "hp": 10, "max_hp": 10, "block": 0,
            "powers": [], "is_dead": False, "removed_after_death": False}  # fmt: skip


def test_intent_log_reads_only_displayed_intents() -> None:
    a1 = [
        {"kind": "Attack", "amount": 5, "effective_amount": 7, "times": 2},
        {"kind": "Buff", "amount": 3},
    ]
    seq = [
        {"round": 1, "enemies": [_enemy(1, "Nibbit", a1)]},
        {"round": 1, "enemies": [_enemy(1, "Nibbit", [{"kind": "Defend"}])]},
        {"round": 2, "enemies": [_enemy(1, "Nibbit", [{"kind": "Buff", "amount": 2}])]},
        {"round": 3, "enemies": [_enemy(1, "Nibbit", [])]},
    ]
    log = IntentLog()
    outs = [log.observe(copy.deepcopy(s))["ehist"] for s in seq]
    assert outs[0] == [[]] and outs[1] == [[]]
    assert outs[2] == [
        [{"lag": 1, "kind": "Attack", "amount": 5, "effective_amount": 7, "times": 2}]
    ]
    assert outs[3][0][0]["kind"] == "Buff" and outs[3][0][1]["lag"] == 2
    assert all(h["kind"] != "Buff" or h.get("amount") != 3 for h in outs[3][0])  # no tail
    # Other fields randomized -> same history.
    log2 = IntentLog()
    for s, want in zip(seq, outs, strict=True):
        s2 = copy.deepcopy(s)
        for e in s2["enemies"]:
            e.update(hp=99, block=7, powers=[{"name": "X"}], is_dead=True)
        s2["player"] = {"hp": 1}
        assert log2.observe(s2)["ehist"] == want


def test_intent_log_gen_and_driver_agree(played: dict) -> None:
    # Replay the trajectory's states through a fresh log, as the driver would.
    rec = played["traj"]
    rm = rollout_starts.build_fight(rec["fight"], rec["seed"])
    log = IntentLog()
    by_t = {r["t"]: r["obs"]["ehist"] for r in played["decisions"]}
    for t, action in enumerate(rec["actions"]):
        snap = log.observe(_snap(rm.combat_state()))
        if t in by_t:
            assert snap["ehist"] == by_t[t]
        rm.take_action(action)


# ── test 14 / 15 ─────────────────────────────────────────────────────────


def test_superset_rl2_columns_survive_the_cache(played: dict, tmp_path) -> None:
    sp = _cache(tmp_path, played)
    a = sp.a
    for d in range(sp.n):
        rec = played["decisions"][d]
        s, n = a["tok_start"][d], a["n_tok"][d]
        na = a["n_act"][d]
        st = a["cont"][s, T.C_RL2S : T.C_RL2S + T.RL2_STATE_DIM]
        assert np.array_equal(st, np.asarray(rec["rl2"]["state"], np.float16))
        x = a["cont"][s + n - na : s + n, T.C_RL2X :]
        assert np.array_equal(x, np.asarray(rec["rl2"]["x"], np.float16))


def test_runtime_asserts_fire() -> None:
    with pytest.raises(G.RuntimeAssert):
        G.check_step_record({"retained_visits": 5, "combat_search": None}, "x")
    with pytest.raises(G.RuntimeAssert):
        G.check_step_record(
            {"retained_visits": None, "combat_search": {"expand_order": {"considered": 1}}}, "x"
        )
    G.check_step_record(
        {"retained_visits": None, "combat_search": {"expand_order": {"considered": 0}}}, "x"
    )
    with pytest.raises(G.RuntimeAssert):
        G.check_end_of_fight({"reused": 1}, "x")
    with pytest.raises(G.RuntimeAssert):
        G.check_worker_env({"STS2_ROLLOUT_NET": "x"})
    G.check_worker_env({"STS2_OTHER": "1"})
    assert G.env_hash({"STS2_X": "1", "HOME": "a"}) == G.env_hash({"STS2_X": "1", "HOME": "b"})
    assert G.env_hash({"STS2_X": "1"}) != G.env_hash({"STS2_X": "2"})


# ── cache, trainer, metrics ──────────────────────────────────────────────


def _cache(tmp_path, played: dict, extra: list | None = None) -> D.Split:
    gdir = tmp_path / "gen"
    parts = G.Parts(str(gdir / "dec"), 1000)
    parts.write(played["decisions"])
    for e in extra or []:
        parts.write(e["decisions"])
    parts.close()
    D.build_cache([str(gdir)], str(tmp_path / "cache"), procs=1, log=lambda m: None)
    return D.Split(str(tmp_path / "cache"), played["split"])


def test_trainer_smoke_every_arm(tmp_path, played: dict) -> None:
    import torch

    tr = _record(seed=7, split="train", rep=1)
    for d in tr["decisions"]:
        d["split"] = "train"
    _cache(tmp_path, played, [tr])
    train_sp = D.Split(str(tmp_path / "cache"), "train")
    for arm in T.ARMS:
        n_act = train_sp.a["n_act"]
        full = TR.derive_split(train_sp.a, arm)
        for d in range(min(3, train_sp.n)):
            s, n = train_sp.a["tok_start"][d], train_sp.a["n_tok"][d]
            want = T.derive(train_sp.a["cat"][s : s + n], train_sp.a["cont"][s : s + n],
                            train_sp.a["sec"][s : s + n], int(n_act[d]), arm)  # fmt: skip
            s2, n2 = full["tok_start"][d], full["n_tok"][d]
            assert np.array_equal(full["cat"][s2 : s2 + n2], want[0])
            assert np.array_equal(full["cont"][s2 : s2 + n2], want[1])
            assert np.array_equal(full["sec"][s2 : s2 + n2], want[2])
    for arm in ("S_ctx", "S_ctxarch_rl2in", "S_rl2arch"):
        out = tmp_path / arm
        args = argparse.Namespace(
            cache=str(tmp_path / "cache"), arm=arm, out=str(out), seed=0, device="cpu",
            target="primary", dim=32, layers=1, heads=4, dropout=0.1, hidden=16, emb=8,
            rl2_mode="normal", lr=None, wd=None, batch=8, epochs=2, warmup_epochs=1, schedule=None,
            lambda_v=0.1, id_dropout=0.05, clip=None, train_frac=1.0, rooms="all", overfit=0,
            max_minutes=0,
        )  # fmt: skip
        TR.cmd_train(args)
        summ = json.loads(_read(out / "summary.json"))
        assert summ["epochs_run"] == 2 and summ["best_epoch"] >= 1
        ck = torch.load(out / "best.pt", weights_only=True)
        assert ck["arm"] == arm
        assert os.path.exists(out / "dev_picks.npz")
        # The pick file records what the checkpoint was trained as, and for a
        # transformer the trainer's DEV picks are the predict path's (fp32).
        ckpt = out / ("rl2.json" if arm == "S_rl2arch" else "best.pt")
        pf = tmp_path / f"pred_{arm}.npz"
        TR.cmd_predict(argparse.Namespace(cache=str(tmp_path / "cache"), ckpt=str(ckpt),
                                          split="dev", out=str(pf), device="cpu", final=False,
                                          final_dir=None))  # fmt: skip
        z = np.load(pf)
        assert str(z["arm"]) == arm and int(z["seed"]) == 0
        assert z["has_unk"].shape == z["pick"].shape
        if arm != "S_rl2arch":
            assert np.array_equal(np.load(out / "dev_picks.npz")["pick"], z["pick"])
    # The rl2arch export loads in the Rust playout and plays.
    import sts2_rs as rs

    lp = rs.LearnedPlayout(_read(tmp_path / "S_rl2arch" / "rl2.json"))
    cs = _walk(n=1)[0][0]
    assert len(lp.explain(cs)["logits"]) == len(cs.legal_actions())
    # The transformer checkpoint drives the play loop.
    job = {"i": 0, "fight": 0, "rep": 0, "row": FIGHT, "build_seed": 7, "seed": 17}
    res = P.play_jobs(P.CtxNet(str(tmp_path / "S_ctx" / "best.pt")), [job], 1.0, True, max_steps=60)
    assert res[0]["steps"] > 0


def test_huge_enemy_hp_saturates_in_fp16() -> None:
    """Mid-fight, Waterfall Giant's honest snapshot shows an HP near 1e9; the
    fp16 cast saturates it instead of making it inf (NaN logits in play)."""
    cs, acts = _walk(n=1)[0]
    snap = _snap(cs)
    snap["enemies"][0].update(hp=999_999_999, max_hp=999_999_999)
    cont = _toks(snap, cs, acts)["cont"]
    big = np.abs(cont) > T.F16_MAX
    assert big.any() and not np.isfinite(cont.astype(np.float16)).all()
    c = T.f16(cont)
    assert np.isfinite(c).all() and (np.abs(c[big]) == T.F16_MAX).all()
    assert np.array_equal(c[~big], cont[~big].astype(np.float16))


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a GPU (bf16 autocast path)"
)
def test_dev_picks_are_the_predict_path_on_cuda(tmp_path, played: dict) -> None:
    tr = _record(seed=7, split="train", rep=1)
    for d in tr["decisions"]:
        d["split"] = "train"
    _cache(tmp_path, played, [tr])
    out = tmp_path / "S_ctx"
    args = argparse.Namespace(
        cache=str(tmp_path / "cache"), arm="S_ctx", out=str(out), seed=0, device="cuda",
        target="primary", dim=32, layers=1, heads=4, dropout=0.1, hidden=16, emb=8,
        rl2_mode="normal", lr=None, wd=None, batch=8, epochs=3, warmup_epochs=1, schedule=None,
        lambda_v=0.1, id_dropout=0.05, clip=None, train_frac=1.0, rooms="all", overfit=0,
        max_minutes=0,
    )  # fmt: skip
    TR.cmd_train(args)
    dsp = D.Split(str(tmp_path / "cache"), "dev")
    want = TR.ctx_picks(str(out / "best.pt"), dsp, "cuda")
    assert np.array_equal(np.load(out / "dev_picks.npz")["pick"], want)


def _train_args(cache: str, arm: str, out: str, **kw) -> argparse.Namespace:
    base = dict(
        cache=cache, arm=arm, out=out, seed=0, device="cpu", target="primary", dim=32,
        layers=1, heads=4, dropout=0.1, hidden=16, emb=8, rl2_mode="normal", lr=None, wd=None,
        batch=8, epochs=2, warmup_epochs=1, schedule=None, lambda_v=0.1, id_dropout=0.05,
        clip=None, train_frac=1.0, rooms="all", overfit=0, max_minutes=0,
    )  # fmt: skip
    base.update(kw)
    return argparse.Namespace(**base)


def test_overfit_probe_drops_every_regulariser(tmp_path, played: dict) -> None:
    """INVALID #6: `--overfit` trains with no dropout, no id dropout and no
    weight decay, and its TRAIN top-1 (predicted on the probe's own decisions
    only) is the frozen-weights top-1 on exactly those decisions."""
    tr = _record(seed=7, split="train", rep=1)
    for d in tr["decisions"]:
        d["split"] = "train"
    _cache(tmp_path, played, [tr])
    cache = str(tmp_path / "cache")
    train_sp = D.Split(cache, "train")
    n_sub = max(2, train_sp.n // 2)
    for arm in ("S_ctx", "S_rl2arch"):
        normal = TR.run(_cfg(cache, arm, str(tmp_path / f"{arm}_n")))
        assert normal["regularisation"]["weight_decay"] > 0
        assert normal["regularisation"]["id_dropout"] > 0
        out = tmp_path / f"{arm}_overfit"
        summ = TR.run(_cfg(cache, arm, str(out), overfit=n_sub, epochs=3))
        assert summ["regularisation"] == {"dropout": 0.0, "id_dropout": 0.0, "weight_decay": 0.0}
        assert summ["train_decisions"] == n_sub
        assert summ["train_top1_max"] >= summ["train_top1"]
        if arm == "S_ctx":
            import torch

            ck = torch.load(out / "last.pt", weights_only=True)
            assert not ck["config"]["dropout"]
            order = np.argsort([hashlib.sha1(i.encode()).hexdigest() for i in train_sp.meta["ids"]])
            order = order[:n_sub]
            pick = TR.ctx_picks(str(out / "last.pt"), train_sp)
            has = train_sp.a["rec_clsA"][order] >= 0
            want = float((train_sp.a["cls"][pick[order]][has]
                          == train_sp.a["rec_clsA"][order][has]).mean())  # fmt: skip
            assert summ["train_top1"] == pytest.approx(want, abs=1e-12)


def _cfg(cache: str, arm: str, out: str, **kw) -> argparse.Namespace:
    """`cmd_train`'s argument defaults resolved (lr, wd, schedule, clip) for `TR.run`."""
    a = _train_args(cache, arm, out, **kw)
    if a.arm == "S_rl2arch":
        a.lr, a.wd, a.schedule, a.clip = 3e-3, 1e-4, "const", 0.0
    else:
        a.lr, a.wd, a.schedule, a.clip = 3e-4, 0.01, "cosine", 1.0
    return a


def test_rare_rule_counts_only_the_students_own_train_decisions(tmp_path, played: dict) -> None:
    """A learning-curve fraction (or the boss/elite-only retrain) fits the
    rare-id rule on the decisions it trains on: an id seen only outside them
    maps to UNK, not to a row that never gets a gradient. `keep` = all of
    TRAIN is the default fit."""
    tr = _record(seed=7, split="train", rep=1)
    for d in tr["decisions"]:
        d["split"] = "train"
    _cache(tmp_path, played, [tr])
    sp = D.Split(str(tmp_path / "cache"), "train")
    full = D.RareMap.fit(sp, min_runs=1)
    same = D.RareMap.fit(sp, min_runs=1, keep=np.ones(sp.n, bool))
    for slot in T.CONTENT_SLOTS:
        assert np.array_equal(full.maps[slot], same.maps[slot])
    keep = np.zeros(sp.n, bool)
    keep[0] = True
    part = D.RareMap.fit(sp, min_runs=1, keep=keep)
    v = T.vocab()
    tok = slice(sp.a["tok_start"][0], sp.a["tok_start"][0] + sp.a["n_tok"][0])
    outside_only = 0
    for slot in T.CONTENT_SLOTS:
        s = T.S[slot]
        tab = v.tables[s]
        inside = {int(i) for i in sp.a["cat"][tok, s] if i < tab.unk}
        everywhere = {int(i) for i in sp.a["cat"][:, s] if i < tab.unk}
        outside_only += len(everywhere - inside)
        for i in everywhere:
            assert full.maps[slot][i] == i
            assert part.maps[slot][i] == (i if i in inside else tab.unk)
    assert outside_only > 0  # the fixture has ids outside decision 0: not vacuous
    m, names = TR.rl2_key_rows(sp, "normal", min_runs=1, keep=keep)
    unk = v.tables[T.S["rl2key"]].unk
    seen = {int(k) for k in sp.a["rl2_key"][: sp.a["n_act"][0]] if k < unk}
    assert {int(i) for i in np.flatnonzero(m)} == seen
    assert len(names) == 1 + len(seen)


def _fake_ckpts(tmp_path, seeds: dict) -> list[str]:
    """`train`-shaped checkpoints that carry only an arm and a seed."""
    import torch

    tmp_path.mkdir(exist_ok=True)
    paths = []
    for arm, ss in seeds.items():
        for sd in ss:
            if arm == "S_rl2arch":
                p = tmp_path / f"{arm}_{sd}.json"
                p.write_text(json.dumps({"keys": [], "meta": {"combat_ctx": {"arm": arm, "seed": sd, "rl2_mode": "normal"}}}))  # fmt: skip
            else:
                p = tmp_path / f"{arm}_{sd}.pt"
                torch.save({"arm": arm, "settings": {"seed": sd}}, p)
            paths.append(str(p))
    return paths


def test_sealed_read_checks_arm_labels_seeds_and_coverage(tmp_path) -> None:
    sp = _FakeSplit()
    dev = tmp_path / "dev_reads.json"
    R = {"S_ctxarch_rl2in": 0.05, "S_rl2arch": 0.06, "ref_teacher": 0.04}
    dev.write_text(json.dumps({"R": R, "tau": 0.02}))
    three = {a: [0, 1, 2] for a in M.STUDENTS}

    def freeze(final, ckpts, n=3):
        M.cmd_freeze(argparse.Namespace(final_dir=str(final), dev_reads=str(dev), ckpt=ckpts,
                                        seeds_per_arm=n))  # fmt: skip
        return json.loads(_read(final / "FROZEN.json"))

    # One seed per arm is refused at the default (3), and a repeated seed too.
    with pytest.raises(SystemExit):
        freeze(tmp_path / "f1", _fake_ckpts(tmp_path / "x", {a: [0] for a in M.STUDENTS}))
    dup = _fake_ckpts(tmp_path / "y", {"S_ctx": [0, 1], "S_ctxarch_rl2in": [0, 1, 2], "S_rl2arch": [0, 1, 2]})  # fmt: skip
    torch_save_dup = tmp_path / "y" / "S_ctx_dup.pt"
    __import__("torch").save({"arm": "S_ctx", "settings": {"seed": 1}, "x": 1}, torch_save_dup)
    with pytest.raises(SystemExit):
        freeze(tmp_path / "f2", [*dup, str(torch_save_dup)])
    ckpts = _fake_ckpts(tmp_path / "z", three)
    frozen = freeze(tmp_path / "final", ckpts)
    assert frozen["seeds_per_arm"] == 3 and len(frozen["identity"]) == 9

    files = {}
    for c in ckpts:
        ident = M.ckpt_identity(c)
        f = tmp_path / f"pick_{ident['arm']}_{ident['seed']}.npz"
        np.savez(f, pick=np.array([0, 2]), ids_sha=M.ids_sha(sp), ckpt_sha256=M.sha256(c),
                 arm=ident["arm"], seed=ident["seed"], has_unk=np.array([False, True]))  # fmt: skip
        files[ident["arm"], ident["seed"]] = str(f)
    good = [f"{a}={files[a, sd]}" for a in M.STUDENTS for sd in range(3)]
    picks, unk = M.load_picks(good, sp, frozen)
    assert {a: len(v) for a, v in picks.items()} == dict.fromkeys(M.STUDENTS, 3)
    assert unk.tolist() == [False, True]
    # The reviewer's swap: two arms' labels exchanged.
    swapped = [g.replace("S_ctx=", "TMP=").replace("S_ctxarch_rl2in=", "S_ctx=").replace("TMP=", "S_ctxarch_rl2in=") for g in good]  # fmt: skip
    with pytest.raises(SystemExit, match="labelled"):
        M.load_picks(swapped, sp, frozen)
    with pytest.raises(SystemExit, match="cover every frozen checkpoint"):
        M.load_picks(good[:-1], sp, frozen)
    with pytest.raises(SystemExit, match="cover every frozen checkpoint"):
        M.load_picks([*good[:-1], good[0]], sp, frozen)
    # A file whose own seed disagrees with the frozen checkpoint's.
    bad = tmp_path / "bad.npz"
    z = dict(np.load(files["S_ctx", 0]))
    z["seed"] = np.array(5)
    np.savez(bad, **z)
    with pytest.raises(SystemExit, match="seed"):
        M.load_picks([f"S_ctx={bad}", *good[1:]], sp, frozen)
    # The compat twin is not a student.
    compat = tmp_path / "compat.json"
    compat.write_text(json.dumps({"meta": {"combat_ctx": {"arm": "S_rl2arch", "seed": 0, "rl2_mode": "compat"}}}))  # fmt: skip
    with pytest.raises(SystemExit, match="compat"):
        freeze(tmp_path / "f3", [*ckpts[:-1], str(compat)])


def test_sealed_table_carries_the_prereg_diagnostics(tmp_path, played: dict) -> None:
    rep1 = _record(seed=7, split="dev", rep=1)
    sp = _cache(tmp_path, played, [rep1])
    # Two plays of one bank fight: two fight-plays, one fight cluster.
    assert len(sp.fights) == 2 and len(sp.bank_fights) == 1
    rp = M.reference_picks(sp)
    picks = {"S_ctx": [rp["heur"]], "S_ctx_tok": [rp["rl2_zs"]],
             "S_ctxarch_rl2in": [rp["rl2_zs"]]}  # fmt: skip
    unk = np.zeros(sp.n, bool)
    unk[::3] = True
    contrasts = [("S_ctxarch_rl2in", "S_ctx"), *M.DIAG_CONTRASTS]
    t = M.evaluate(sp, picks, contrasts, {"S_ctxarch_rl2in-S_ctx": 0.002}, n_boot=200,
                   tau=0.0, unk=unk)  # fmt: skip
    assert t["n_fights"] == 1 and t["n_fight_plays"] == 2
    assert t["contrasts"]["S_ctxarch_rl2in-S_ctx"]["fight_clustered"]["clusters"] == 1
    assert "outcome" in t["contrasts"]["S_ctxarch_rl2in-S_ctx"]
    assert "outcome" not in t["contrasts"]["S_ctx_tok-S_ctx"]
    er = t["e_rules"]
    removed = sum(er["removed"].values())
    assert er["decisions"] == sp.n and er["decisions"] - removed == er["in_E"] == t["n_E"]
    assert t["all_searched"]["n"] >= t["n_E"]
    sub = t["subsets"]
    assert sub["contested"]["n_E"] == t["n_E"]  # tau 0 keeps every E decision
    assert sub["zero_unk"]["n_E"] == int((sp.a["in_E"] & ~unk).sum())
    want_hv = 0
    for d in np.flatnonzero(sp.a["in_E"]):
        s, k = sp.a["act_start"][d], sp.a["n_act"][d]
        cls = sp.a["cls"][s : s + k].astype(np.int64)
        va, _ = D.class_stats(cls, sp.a["visA"][s : s + k], sp.a["qA"][s : s + k])
        vb, _ = D.class_stats(cls, sp.a["visB"][s : s + k], sp.a["qB"][s : s + k])
        want_hv += min(va.min(), vb.min()) >= M.HIGH_VISITS
    assert sub["high_visit"]["n_E"] == want_hv
    for name in ("contested", "high_visit", "zero_unk"):
        assert set(sub[name]["contrasts"]) == {"S_ctxarch_rl2in-S_ctx", "S_ctx_tok-S_ctx"}
    # The student-free E report needs no picks.
    out = tmp_path / "e.json"
    M.cmd_eval(argparse.Namespace(cache=str(tmp_path / "cache"), split="dev", picks=[],
                                  e_report=True, final=False, final_dir=None, stage=1,
                                  disagreements=None, referee=None, dev_reads_out=None,
                                  n_boot=10, out=str(out)))  # fmt: skip
    assert json.loads(_read(out)) == er


def test_play_contrast_pairs_jobs_and_clusters_by_run(tmp_path) -> None:
    rows = [{**FIGHT, "file": f"r{k % 2}.run.gz", "floor": 3 + k, "room_type": "elite"}
            for k in range(3)]  # fmt: skip
    jobs, _ = P.make_jobs(rows, 1)

    def run(spec: str, name: str, js: list) -> str:
        res = P.play_jobs(P.policy_from_spec(spec), js, 1.0, True, max_steps=200)
        summ = {"policy": spec, **P.policy_identity(spec), "fights_file": "x", "fold": None,
                "greedy": True, "temperature": None, "view": "raw"}  # fmt: skip
        path = tmp_path / name
        path.write_text(json.dumps({"summary": summ, "results": res}))
        return str(path)

    a, b = run("heur", "a.json", jobs), run("rl2", "b.json", jobs)
    for r in json.loads(_read(a))["results"]:
        assert r["file"] in ("r0.run.gz", "r1.run.gz") and r["key"].startswith(r["file"])
    out = tmp_path / "c.json"
    P.cmd_contrast(argparse.Namespace(a=[a], b=[b], n_boot=200, level=0.95, out=str(out)))
    c = json.loads(_read(out))
    wa = np.mean([r["won"] for r in json.loads(_read(a))["results"]])
    wb = np.mean([r["won"] for r in json.loads(_read(b))["results"]])
    assert np.isclose(c["a_minus_b"]["delta"], wa - wb)
    assert c["runs"] == 2 and c["a_minus_b"]["clusters"] == 2 and c["jobs"] == 3
    other = run("heur", "o.json", jobs[:2])
    with pytest.raises(SystemExit):
        P.cmd_contrast(argparse.Namespace(a=[a], b=[other], n_boot=10, level=0.95, out=None))


def test_select_temperature_records_and_pins_its_checkpoint(tmp_path, monkeypatch) -> None:
    """T* names the checkpoint it played (sha256, arm, training seed), so it can
    be matched to the frozen seed-0 checkpoint; a checkpoint that changes while
    the temperatures are played is refused."""
    import torch

    (ck,) = _fake_ckpts(tmp_path / "ck", {"S_ctx": [0]})
    monkeypatch.setattr(P, "select_rows", lambda args: [FIGHT])
    monkeypatch.setattr(P, "make_jobs", lambda rows, reps: ([{"i": 0}], []))
    monkeypatch.setattr(P, "play_many", lambda pol, jobs, t, *a: [{"won": abs(t - 0.5) < 1e-9}])
    out = tmp_path / "tstar.json"
    ns = argparse.Namespace(policy=f"ctx:{ck}", fights="x", scheme="S2", fold=2, max_fights=0,
                            procs=1, out=str(out))  # fmt: skip
    assert P.cmd_select_temperature(ns) == 0
    t = json.loads(_read(out))
    assert t["T_star"] == pytest.approx(0.5)
    assert (t["ckpt_sha256"], t["arm"], t["seed"]) == (M.sha256(ck), "S_ctx", 0)

    def swap(pol, jobs, t, *a):
        torch.save({"arm": "S_ctx", "settings": {"seed": 0}, "t": t}, ck)
        return [{"won": False}]

    monkeypatch.setattr(P, "play_many", swap)
    ns2 = argparse.Namespace(**{**vars(ns), "out": str(tmp_path / "t2.json")})
    with pytest.raises(SystemExit, match="changed while T"):
        P.cmd_select_temperature(ns2)
    assert not os.path.exists(tmp_path / "t2.json")


class _FakeSplit:
    """Two decisions: d0 has classes {0: Q .5/.4, 1: Q .3/.6}; d1 one class each act."""

    def __init__(self) -> None:
        self.n = 2
        cls = np.array([0, 1, 0, 1, 2])
        regA = np.array([0.0, 0.2, 0.0, 0.1, 0.3])
        regB = np.array([0.2, 0.0, 0.05, 0.0, 0.2])
        self.a = {
            "act_start": np.array([0, 2]),
            "n_act": np.array([2, 3]),
            "cls": cls,
            "regA": regA,
            "regB": regB,
            "rec_clsA": np.array([0, 0]),
            "rec_clsB": np.array([1, 1]),
            "recA": np.array([0, 0]),
            "recB": np.array([1, 1]),
            "in_E": np.array([True, True]),
            "room": np.array([2, 1]),
        }
        self.run = np.array([0, 1])
        self.fight = np.array([0, 1])
        self.bank_fight = np.array([0, 1])
        self.meta = {"ids": ["d0", "d1"]}


def test_metrics_on_a_hand_computed_toy() -> None:
    sp = _FakeSplit()
    pick = np.array([0, 4])  # d0 class 0, d1 class 2
    rows = M.pick_rows(sp, pick)
    assert np.allclose(rows["regA"], [0.0, 0.3]) and np.allclose(rows["regB"], [0.2, 0.2])
    s = M.summarize(rows, sp.a["in_E"])
    assert np.isclose(s["R"], 0.5 * (0.15 + 0.2))
    assert np.isclose(s["Agr"], 0.5 * (0.5 + 0.0))
    ref = M.ref_rows(sp)  # A's rec scored by B: d0 0.2, d1 0.05; B's rec by A: 0.2, 0.1
    assert np.allclose(ref["regB"], [0.2, 0.05]) and np.allclose(ref["regA"], [0.2, 0.1])
    u = M.uniform_rows(sp)
    assert np.allclose(u["regA"], [0.1, (0.0 + 0.1 + 0.3) / 3])
    res = M.bootstrap_contrast(
        np.array([[0.2, 0.4], [0.2, 0.4]]), np.array([[0.1, 0.1]]), sp.run, n_boot=500
    )
    assert np.isclose(res["delta"], 0.2) and res["lo"] <= 0.2 <= res["hi"]
    assert M.outcome(0.001, 0.01, 0.004, 0.003, 1) == "PASS"
    assert M.outcome(0.001, 0.01, 0.002, 0.003, 1) == "SMALL"
    assert M.outcome(-0.001, 0.002, 0.0, 0.003, 1) == "NULL"
    assert M.outcome(-0.001, 0.004, 0.0, 0.003, 1) == "INCONCLUSIVE"
    assert M.outcome(-0.001, 0.004, 0.0, 0.003, 2) == "UNDERPOWERED"
    assert np.allclose(
        [M.margin(0.05, 0.04), M.margin(0.1, 0.0), M.margin(0.0, 0.0)], [0.0025, 0.003, 0.002]
    )
    # ICC: identical within groups -> 1; pure noise -> ~0.
    v = np.repeat([1.0, 2.0, 3.0], 4)
    assert np.isclose(PW.icc_oneway(v, np.repeat([0, 1, 2], 4)), 1.0)
    assert np.isclose(PW.icc_oneway(np.array([1.0, -1.0, 1.0, -1.0]), np.array([0, 0, 1, 1])), 0.0)


# ── the honesty tests on real pilot decisions (slow, needs a gen dir) ────


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("COMBAT_CTX_GEN"), reason="set COMBAT_CTX_GEN=<gen dir>")
def test_honesty_on_200_real_pilot_decisions() -> None:
    import itertools

    gdir = os.environ["COMBAT_CTX_GEN"]
    trajs = {}
    with open(os.path.join(gdir, "traj.jsonl"), encoding="utf-8") as f:
        next(f)
        for line in f:
            t = json.loads(line)
            trajs[t["key"], int(t["rep"])] = t
    recs = list(itertools.islice(G.iter_records(gdir), 200))
    by: dict = {}
    for r in recs:
        by.setdefault((r["key"], int(r["rep"])), []).append(r)
    rng = random.Random(1)
    n = 0
    for k, rs in by.items():
        states, _ = vd.walk_record(trajs[k], [r["t"] for r in rs])
        for r in rs:
            cs = states[r["t"]]
            assert int(cs.ordered_hash()) == r["ohash"]
            bv = cs.belief_view(0)
            snap = honest_snapshot(bv, r["room"])
            snap["ehist"] = r["obs"]["ehist"]
            assert snap == r["obs"]
            assert whitelist_violations(snap) == []
            acts = r.get("act_obs") or r["actions"]
            want = T.tokens(*D.record_inputs(r), "S_ctx")
            for w in range(4):
                sw = cs.sample_world(w)
                s2 = honest_snapshot(sw.belief_view(0), r["room"])
                s2["ehist"] = r["obs"]["ehist"]
                assert _same_tokens(T.tokens(s2, acts, _rl2(sw), "S_ctx"), want)
            d = sts2_rs.state_to_dict(bv)
            parts = (
                bv.card_damage_preview(),
                bv.card_block_preview(),
                bv.pending_choice(),
                r["room"],
            )
            fz = snapshot_from_parts(_fuzz(d, rng), *parts)
            fz["ehist"] = r["obs"]["ehist"]
            assert _same_tokens(T.tokens(fz, acts, _rl2(cs), "S_ctx"), want)
            n += 1
    assert n == len(recs) >= 100


def test_wire_names_land_in_the_universe() -> None:
    v = T.vocab()
    for name, want in (
        ("Tezcatara's Ember", "TEZCATARAS_EMBER"),
        ("Perfect Fit", "PERFECT_FIT"),
        ("Soul's Power", "SOULS_POWER"),
        ("Sharp", "SHARP"),
    ):
        row, oov = v.table("enchantment").index(name)
        assert not oov and v.table("enchantment").names[row] == want
    for slot, name in (("card", "ACCURACY_CARD"), ("card", "STRIKE_IRONCLAD"), ("relic", "PEN_NIB"),
                       ("power", "STRENGTH"), ("monster", "Nibbit"), ("potion", "FIRE_POTION")):  # fmt: skip
        assert not v.table(slot).index(name)[1], (slot, name)
    assert v.table("card").index("NOT_A_CARD") == (v.table("card").unk, True)


def test_potion_actions_are_potion_tokens_and_in_E() -> None:
    cs, acts = _walk(n=1)[0]
    assert any(a["action"] == "drink_potion" for a in acts)
    tk = _toks(_snap(cs), cs, acts)
    secs = tk["sec"][-len(acts) :]
    for a, s in zip(acts, secs, strict=True):
        if a["action"] == "drink_potion":
            assert s == T.SEC["ACT_POTION"]
    assert "drink_potion" in D.E_ACTIONS


def test_power_projection_rule() -> None:
    rng = np.random.default_rng(0)
    runs = np.repeat(np.arange(50), 20)
    d = {"d_ref": rng.normal(0, 0.01, 1000), "d_heur": rng.normal(0, 0.01, 1000), "runs": runs.astype(str),
         "fights": runs.astype(str), "rooms": np.where(runs % 2 == 0, "boss", "elite")}  # fmt: skip
    res = PW.project(d, {"S1": 249, "S2": 479})
    assert res["scheme"] == "S1" and not res["projected_underpowered"]
    d["d_ref"] = rng.normal(0, 0.5, 1000)
    res = PW.project(d, {"S1": 249, "S2": 479})
    assert res["scheme"] == "S2" and res["eval_rep1"] and res["projected_underpowered"]
