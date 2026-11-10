"""Pure-logic tests for the deep act-2/3 state-parity probe (issue #158).

These exercise the nav-instrumentation and the C# mid-combat-snapshot HP
oracle-artifact classifier WITHOUT driving the engines (no C# subprocess), so
they are fast and deterministic. The engine-driven sweep is validated by running
the CLI on real seeds.
"""

from __future__ import annotations

import io
import sys

import act23_state_parity as a23
import pytest
from run_parity import Snap


def _snap(phase: str, hp: int | None = None, enemies: tuple[int, ...] = ()) -> Snap:
    """Minimal Snap for classifier tests (only the fields the helpers read)."""
    return Snap(
        phase=phase,
        hp=hp,
        max_hp=hp,
        block=0,
        gold=0,
        deck={},
        deck_size=0,
        relics=[],
        potions=[],
        enemies=enemies,
    )


# --------------------------------------------------------------------------- #
# act / floor extraction reads the C# ``context`` nesting
# --------------------------------------------------------------------------- #
def test_act_of_reads_csharp_context() -> None:
    # C# nests act under ``context`` at event/combat/card_reward decisions.
    assert a23._act_of({"decision": "event_choice", "context": {"act": 2}}) == 2
    # Rust puts it top-level.
    assert a23._act_of({"decision": "event_choice", "act": 3}) == 3
    # Absent everywhere -> None (the old behaviour that pinned max_act to Rust).
    assert a23._act_of({"decision": "event_choice"}) is None
    assert a23._act_of(None) is None


def test_floor_of_reads_context() -> None:
    assert a23._floor_of({"context": {"floor": 7}}) == 7
    assert a23._floor_of({"floor": 22}) == 22
    assert a23._floor_of({"decision": "combat_play"}) is None


# --------------------------------------------------------------------------- #
# current-node coord: Rust top-level dict vs C# per-node ``current`` flag
# --------------------------------------------------------------------------- #
class _MapBackend:
    def __init__(self, m: dict) -> None:
        self._m = m

    def get_map(self) -> dict:
        return self._m


def test_current_coord_rust_style() -> None:
    b = _MapBackend({"current_coord": {"col": 3, "row": 16}, "rows": []})
    assert a23._current_coord(b) == (3, 16)


def test_current_coord_csharp_style() -> None:
    b = _MapBackend(
        {
            "rows": [
                [{"col": 1, "row": 5, "current": False}],
                [{"col": 2, "row": 6, "current": True}],
            ]
        }
    )
    assert a23._current_coord(b) == (2, 6)


def test_current_coord_absent() -> None:
    # No current node and no current_coord -> None (the C# off-map case).
    b = _MapBackend({"rows": [[{"col": 1, "row": 5, "current": False}]]})
    assert a23._current_coord(b) is None

    class _Boom:
        def get_map(self):
            raise RuntimeError("no map in this phase")

    assert a23._current_coord(_Boom()) is None


# --------------------------------------------------------------------------- #
# diff_nav: act + coord, with one-sided None skipped
# --------------------------------------------------------------------------- #
def test_diff_nav_flags_act_and_coord() -> None:
    cs = a23.Nav(act=2, floor=7, coord=(3, 0))
    rs = a23.Nav(act=1, floor=16, coord=(3, 16))
    fields = a23.diff_nav(cs, rs)
    assert any(f.startswith("nav.act:") for f in fields)
    assert any(f.startswith("nav.coord:") for f in fields)


def test_diff_nav_skips_one_sided_none() -> None:
    # C# exposes no current node (coord None) -> coord comparison skipped.
    cs = a23.Nav(act=2, floor=None, coord=None)
    rs = a23.Nav(act=2, floor=16, coord=(3, 16))
    assert a23.diff_nav(cs, rs) == []


def test_diff_nav_clean_when_aligned() -> None:
    cs = a23.Nav(act=2, floor=7, coord=(2, 6))
    rs = a23.Nav(act=2, floor=22, coord=(2, 6))  # floor differs but is NOT compared
    assert a23.diff_nav(cs, rs) == []


# --------------------------------------------------------------------------- #
# HP oracle-artifact classifier (issue #158 step 3)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Claws: C# is sent the same first-6 candidates Rust transformed atomically
# --------------------------------------------------------------------------- #
def _claws_view(n: int) -> dict:
    return {
        "decision": "card_select",
        "prompt_key": "CLAWS.selectionScreenPrompt",
        "source": "FromDeckForTransformation",
        "min_select": 0,
        "max_select": 6,
        "cards": [{"index": i, "id": "CARD.STRIKE_IRONCLAD"} for i in range(n)],
    }


def test_claws_atomic_choice_picks_first_six_candidates() -> None:
    assert a23.claws_atomic_choice(_claws_view(20)) == [
        ("select_cards", {"indices": "0,1,2,3,4,5"})
    ]


def test_claws_atomic_choice_clamps_to_candidate_count() -> None:
    assert a23.claws_atomic_choice(_claws_view(3)) == [("select_cards", {"indices": "0,1,2"})]
    assert a23.claws_atomic_choice(_claws_view(0)) == []


def test_claws_atomic_choice_ignores_other_selects() -> None:
    view = _claws_view(10)
    view["prompt_key"] = "TO_TRANSFORM"
    assert a23.claws_atomic_choice(view) == []


def test_hp_artifact_peeled_in_combat() -> None:
    cs, rs = _snap("combat", hp=99988), _snap("combat", hp=99986)
    assert a23._is_combat_hp_artifact(cs, rs, ["hp: cs=99988 rs=99986"])


def test_hp_artifact_peeled_in_combat_exit_window() -> None:
    for phase in ("card_reward", "card_select"):
        cs, rs = _snap(phase, hp=99791), _snap(phase, hp=99790)
        assert a23._is_combat_hp_artifact(cs, rs, ["hp: cs=99791 rs=99790"]), phase


def test_hp_artifact_not_peeled_with_enemy_diff() -> None:
    # A co-occurring enemies_hp field => real combat divergence, never peeled.
    cs, rs = _snap("combat", hp=99969, enemies=(15,)), _snap("combat", hp=99956, enemies=(12,))
    fields = ["hp: cs=99969 rs=99956", "enemies_hp: cs=[15] rs=[12]"]
    assert not a23._is_combat_hp_artifact(cs, rs, fields)


def test_hp_artifact_not_peeled_at_event_or_rest() -> None:
    # Events/rest are where REAL HP-changing effects live.
    for phase in ("event", "rest"):
        cs, rs = _snap(phase, hp=99988), _snap(phase, hp=99986)
        assert not a23._is_combat_hp_artifact(cs, rs, ["hp: cs=99988 rs=99986"]), phase


def test_hp_artifact_not_peeled_above_tolerance() -> None:
    delta = a23.HP_ARTIFACT_TOLERANCE + 1
    cs, rs = _snap("combat", hp=100_000), _snap("combat", hp=100_000 - delta)
    assert not a23._is_combat_hp_artifact(cs, rs, [f"hp: cs=100000 rs={100_000 - delta}"])


# --------------------------------------------------------------------------- #
# survival policy: the walk must beat The Insatiable's Sandpit timer
# --------------------------------------------------------------------------- #
def _combat_view(hand: list[str], enemy_powers: list[str] = ()) -> dict:
    return {
        "decision": "combat_play",
        "hand": [{"index": i, "id": cid} for i, cid in enumerate(hand)],
        "enemies": [{"powers": [{"id": p, "amount": 1} for p in enemy_powers]}],
    }


def test_survival_choice_picks_frantic_escape() -> None:
    view = _combat_view(["CARD.STRIKE_IRONCLAD", "CARD.FRANTIC_ESCAPE"])
    choices = [
        ("play_card", {"card_index": 0, "target_index": 0}),
        ("play_card", {"card_index": 1, "target_index": None}),
        ("end_turn", {}),
    ]
    assert a23._survival_choice(choices, view) == 1


def test_survival_choice_none_without_survival_card() -> None:
    view = _combat_view(["CARD.STRIKE_IRONCLAD"])
    assert a23._survival_choice([("play_card", {"card_index": 0}), ("end_turn", {})], view) is None


def test_survival_pool_drops_end_turn_only_under_a_timer_kill() -> None:
    choices = [("play_card", {"card_index": 0}), ("end_turn", {})]
    sandpit = _combat_view(["CARD.STRIKE_IRONCLAD"], ["SANDPIT_POWER"])
    assert a23._survival_pool(choices, sandpit) == choices[:1]
    assert a23._survival_pool(choices, _combat_view(["CARD.STRIKE_IRONCLAD"])) == choices
    # Nothing left to play: ending the turn stays legal.
    assert a23._survival_pool([("end_turn", {})], sandpit) == [("end_turn", {})]


# --------------------------------------------------------------------------- #
# checkpoints: recorder, naming, stale detection
# --------------------------------------------------------------------------- #
class _FakeRust:
    def __init__(self) -> None:
        self.n = 0

    def start(self, **_kw):
        self.n = 0
        return {"n": self.n}

    def act(self, verb, **_kw):
        self.n += 1
        return {"n": self.n, "verb": verb}

    def set_player(self, **_kw):
        return {"ok": True}

    def get_map(self):
        return {"map": True}


def test_recorder_logs_only_mutating_calls() -> None:
    rec = a23._RustRecorder(_FakeRust())
    rec.start(character="Ironclad", seed="1", ascension=0)
    rec.act("end_turn")
    rec.set_player(hp=5)
    assert rec.get_map() == {"map": True}
    assert [c[0] for c in rec.calls] == ["start", "act", "set_player"]
    assert rec.calls[1] == ["act", ["end_turn"], {}]


def test_checkpoint_name_lineage() -> None:
    assert a23.checkpoint_name("Ironclad", "1000", 0, 2) == "Ironclad-1000-a0-act2"
    assert a23.checkpoint_name("Ironclad", "1000", 0, 3, "Ironclad-1000-a0-act2.b3") == (
        "Ironclad-1000-a0-act2.b3-act3"
    )


class _FakeCs:
    def load_save(self, path):
        return {"decision": "map_select", "path": path}


def test_resume_replays_rust_and_flags_a_stale_checkpoint(tmp_path) -> None:
    rec = a23._RustRecorder(_FakeRust())
    rec.start(character="Ironclad", seed="1", ascension=0)
    view = rec.act("end_turn")
    meta = {"rust_calls": rec.calls, "rs_view": view}

    cs_view, rs_view = a23.resume_checkpoint(
        tmp_path, _FakeCs(), a23._RustRecorder(_FakeRust()), meta
    )
    assert rs_view == view and cs_view["decision"] == "map_select"

    stale = {**meta, "rs_view": {"n": 99}}
    try:
        a23.resume_checkpoint(tmp_path, _FakeCs(), a23._RustRecorder(_FakeRust()), stale)
    except a23.CheckpointStale:
        pass
    else:
        raise AssertionError("a changed Rust replay must raise CheckpointStale")


# --------------------------------------------------------------------------- #
# end to end: a checkpoint resumes in lockstep (needs both engines)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.oracle
def test_checkpoint_round_trip(tmp_path) -> None:
    """Build an act-2 checkpoint, then resume it: the resumed engines must
    start in agreement and stay in lockstep for a stretch of act 2."""
    built = a23.run_one_invincible(
        "1000",
        "Ironclad",
        max_steps=4000,
        checkpoint_dir=tmp_path,
        checkpoint_acts=frozenset({2}),
        stop_at_checkpoint=True,
    )
    assert built.result.terminal == "checkpointed", built.result.divergence
    (ck,) = a23.list_checkpoints(tmp_path)
    assert a23.read_checkpoint_meta(ck)["act"] == 2

    resumed = a23.run_one_invincible("", from_checkpoint=ck, branch=0, max_steps=60)
    assert resumed.result.divergence is None, resumed.result.divergence
    assert resumed.result.terminal in ("max_steps", "game_over_match")


@pytest.mark.slow
@pytest.mark.oracle
def test_an_a10_run_walks_on_to_the_second_boss_in_lockstep() -> None:
    """#1181 H1 + H2: after the first act-3 boss at A10 the headless returns to
    the map (it used to call ``EnterNextAct``, so the run stopped at
    ``phase: cs='event' rs='map'``), and its map lists the second boss in
    ``rows`` (it used to stop on ``map.nodes: only_rs=[(3, 15)]``). Seed 1015
    stalled at (3, 14) before; it must now end at the second boss in step."""
    import sts2_client

    prev = sts2_client.GAME_VERSION
    sts2_client.set_game_version("v0.111.0")
    try:
        run = a23.run_one_invincible("1015", "Ironclad", ascension=10, max_steps=3000)
    finally:
        sts2_client.set_game_version(prev)
    assert run.result.divergence is None, run.result.divergence
    assert run.result.terminal == "game_over_match", run.result.terminal
    assert run.max_act == 3


#: Monsters whose ENGINE move log differs from C#'s StateLog (#1181 lockstep
#: check), so a log restored from C# rightly differs from the engine's own.
#: Only CeremonialBeast is left (#1215): its STUN is the engine's real state,
#: whose follow-up C# sends as the resume. Both roll identically.
_ENGINE_LOG_DIVERGES = {"CeremonialBeast"}


@pytest.mark.slow
@pytest.mark.oracle
def test_the_csharp_move_log_restores_the_engines_own(monkeypatch) -> None:
    """#1181: at every lockstep combat step of an A10 run, each living enemy's
    C# ``ai`` object (``StateLog`` ids, ``performed_first_move``,
    ``performed``, a synthetic stun's ``forced_next``), put on the engine's own
    dict, rebuilds the engine's own move log, current move and queued move.
    Only the monsters in ``_ENGINE_LOG_DIVERGES`` may differ."""
    import sts2_client
    import sts2_rs

    tally = {"checked": 0, "unexpected": []}
    orig = a23.full_state_divs

    def check(g, r):
        cs_state, divs = orig(g, r)
        combat = r.combat_state() if cs_state else None
        if combat is None or divs:
            return cs_state, divs
        d = sts2_rs.state_to_dict(combat)
        cs_alive = sorted(
            (m for m in cs_state.get("monsters") or [] if m.get("alive")), key=lambda m: m["uid"]
        )
        rs_alive = [i for i, e in enumerate(d["encounter"]) if not e.get("is_dead") and e.get("ai")]
        if len(cs_alive) != len(rs_alive) or any(m.get("ai") is None for m in cs_alive):
            return cs_state, divs

        def pick(e):
            return (e["ai"]["log"], e.get("move_state"), e["ai"].get("forced_next"))

        want = [pick(d["encounter"][i]) for i in rs_alive]
        for i, m in zip(rs_alive, cs_alive, strict=True):
            d["encounter"][i]["ai"] = {k: v for k, v in m["ai"].items()}
        back = sts2_rs.state_to_dict(sts2_rs.state_from_dict(d))
        for j, i in enumerate(rs_alive):
            tally["checked"] += 1
            got = pick(back["encounter"][i])
            kind = d["encounter"][i].get("kind")
            if got != want[j] and kind not in _ENGINE_LOG_DIVERGES:
                tally["unexpected"].append((kind, want[j], got))
        return cs_state, divs

    monkeypatch.setattr(a23, "full_state_divs", check)
    prev = sts2_client.GAME_VERSION
    sts2_client.set_game_version("v0.111.0")
    try:
        run = a23.run_one_invincible(
            "1049", "Ironclad", ascension=10, max_steps=3000, full_state=True
        )
    finally:
        sts2_client.set_game_version(prev)
    assert run.result.terminal == "game_over_match", run.result.terminal
    assert not tally["unexpected"], tally["unexpected"][:5]
    # Steps with a full-state split are skipped (the hand-affliction class of
    # #1188 splits act 2 of this seed), so ~900 enemy states are checked.
    assert tally["checked"] > 500, tally["checked"]


# --------------------------------------------------------------------------- #
# u8 option-index cap: Rust rejects a pick past option 255
# --------------------------------------------------------------------------- #
def _selects(n: int) -> list[tuple[str, dict]]:
    return [("select_cards", {"indices": str(i)}) for i in range(n)]


def test_select_index_cap_is_noop_at_256_options() -> None:
    # Every index 0..255 fits a u8: nothing dropped, so no other run's walk
    # (its choice list or its RNG draws) moves.
    choices = [*_selects(256), ("skip_select", {})]
    assert a23._within_select_index_cap(choices) == choices


def test_select_index_cap_drops_picks_past_255() -> None:
    # Ironclad-2012-a0-act2#b1: Headbutt over a 445-card discard pile. The walk
    # picked '317', which C# took and Rust rejected (invalid_for_phase).
    kept = a23._within_select_index_cap(_selects(445))
    assert len(kept) == 256
    assert max(int(kw["indices"]) for _, kw in kept) == a23.MAX_SELECT_INDEX


def test_select_index_cap_checks_every_index_of_a_multi_pick() -> None:
    choices = [
        ("select_cards", {"indices": "0,1"}),
        ("select_cards", {"indices": "0,300"}),
        ("end_turn", {}),
    ]
    assert a23._within_select_index_cap(choices) == [
        ("select_cards", {"indices": "0,1"}),
        ("end_turn", {}),
    ]


# --------------------------------------------------------------------------- #
# C#-alone atomic-select gate
# --------------------------------------------------------------------------- #
def test_cs_alone_select_when_rust_left_the_select() -> None:
    assert a23._cs_alone_select("card_select", "map")
    assert a23._cs_alone_select("card_select", "card_reward")


def test_cs_alone_select_not_for_a_shared_select() -> None:
    # A select Rust surfaces too (both paused) is driven through the lockstep,
    # Dolly's Mirror included: Rust offers it as a pick-1 card grant.
    assert not a23._cs_alone_select("card_select", "card_select")
    assert not a23._cs_alone_select("map", "card_select")


# --------------------------------------------------------------------------- #
# A C# view that hid its pending select (Toasty Mittens, "combat phase settle")
# --------------------------------------------------------------------------- #
class _ProbeGame:
    """Answers an argument-less ``select_cards`` like ``DoSelectCards``."""

    def __init__(self, pending: bool) -> None:
        self.pending = pending
        self.calls: list[tuple[str, dict]] = []

    def act(self, verb: str, **kw: object) -> dict:
        self.calls.append((verb, kw))
        if verb == "select_cards" and not kw:
            msg = (
                "select_cards requires 'indices' (comma-separated card indices)"
                if self.pending
                else "No pending card selection"
            )
            return {"type": "error", "message": msg}
        return {"type": "decision", "decision": "combat_play"}


def test_cs_select_still_pending_reads_the_probe_error() -> None:
    # ckpt_act2_2000 Ironclad-2091-a0-act2#b1: C# logged "Card selection
    # pending" then returned combat_play from the settle fallback.
    hidden = _ProbeGame(pending=True)
    assert a23._cs_select_still_pending(hidden)
    assert hidden.calls == [("select_cards", {})]
    assert not a23._cs_select_still_pending(_ProbeGame(pending=False))


def test_stale_cs_select_pick_matches_the_rust_finalize_prefix() -> None:
    cards = [{"index": i} for i in range(5)]
    view = {"decision": "card_select", "cards": cards, "min_select": 1}
    assert a23._stale_cs_select_pick(view) == ("select_cards", {"indices": "0"})
    view["min_select"] = 2
    assert a23._stale_cs_select_pick(view) == ("select_cards", {"indices": "0,1"})
    view["min_select"] = 0
    assert a23._stale_cs_select_pick(view) == ("skip_select", {})


# --------------------------------------------------------------------------- #
# --random-loadout
# --------------------------------------------------------------------------- #
class _Rm:
    def max_potion_slots(self) -> int:
        return 2


class _Side:
    """A backend whose ``set_player`` echoes what it took, minus ``drop``."""

    def __init__(self, drop: str | None = None) -> None:
        self.rm = _Rm()
        self.drop = drop
        self.kw: dict = {}

    def set_player(self, **kw: object) -> dict:
        self.kw = kw
        deck = [{"id": c.removesuffix("+1"), "upgraded": c.endswith("+1")} for c in kw["deck"]]
        return {
            "player": {
                "deck": [c for c in deck if c["id"] != self.drop],
                "relics": [{"id": x} for x in kw["relics"]],
                "potions": [{"id": x} for x in kw["potions"]],
                "gold": kw["gold"],
            }
        }


LOADOUT = a23.Loadout(
    deck=("BASH+1", "ANGER"),
    relics=("BURNING_BLOOD", "ANCHOR"),
    potions=("FIRE_POTION", "BLOCK_POTION", "ATTACK_POTION"),
    gold=77,
)
RS_VIEW = {"player": {"relics": [{"id": "RELIC.BURNING_BLOOD"}]}}


def test_graft_keeps_owned_relics_and_fits_the_belt() -> None:
    g, r = _Side(), _Side()
    assert a23._graft_loadout(g, r, LOADOUT, RS_VIEW) == []
    assert r.kw["relics"] == ["BURNING_BLOOD", "ANCHOR"]
    assert r.kw["potions"] == ["FIRE_POTION", "BLOCK_POTION"]
    assert g.kw["relics"] == r.kw["relics"]


def test_graft_strips_rust_card_suffix_for_csharp_only() -> None:
    lo = a23.Loadout(deck=("BARRICADE_CARD+1",), relics=(), potions=(), gold=0)
    g, r = _Side(), _Side()
    a23._graft_loadout(g, r, lo, RS_VIEW)
    assert g.kw["deck"] == ["BARRICADE+1"]
    assert r.kw["deck"] == ["BARRICADE_CARD+1"]


def test_graft_reports_a_card_the_oracle_skipped() -> None:
    # C# skips an unknown id silently; that must not reach the lockstep diff
    # as an engine divergence.
    problems = a23._graft_loadout(_Side(drop="ANGER"), _Side(), LOADOUT, RS_VIEW)
    assert any("cs deck_size 1 != loadout 2" in p for p in problems)


def test_graft_reports_an_engine_refusal() -> None:
    class Refuses(_Side):
        def set_player(self, **kw: object) -> dict:
            return {"type": "error", "message": "unknown relic"}

    problems = a23._graft_loadout(_Side(), Refuses(), LOADOUT, RS_VIEW)
    assert problems == ["Rust set_player failed: 'unknown relic'"]


def test_utf8_stdio_survives_a_cp1252_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A piped stdout on Windows is cp1252, which has no ``Δ``; the run summary
    prints ``max |Δhp|`` and used to crash there after the whole run had passed."""
    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", cp1252)
    with pytest.raises(UnicodeEncodeError):
        print("max |Δhp|=0")
    a23._utf8_stdio()
    print("max |Δhp|=0")
    sys.stdout.flush()
    assert raw.getvalue().endswith("max |Δhp|=0\n".encode())


def test_utf8_stdio_leaves_a_non_textiowrapper_stream_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced stream without ``reconfigure`` (StringIO) is left as it is."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    a23._utf8_stdio()
    assert sys.stdout is buf
