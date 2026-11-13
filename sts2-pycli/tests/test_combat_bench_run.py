"""`ppo.combat_bench run`: seeds, end statuses, HP accounting, resume, honesty (GH #830).

The fights are real human fights from the 2026-10-18 holdout banks
(`logs/2026-10-18-rollout-net-recipe/{hall800,elite800}.jsonl`), converted to
the fight-set SCHEMA and played at a toy budget (40 / 80 sims). Assertions are
invariants of the runner, not the pilot's results, except where a fight was
chosen because its outcome is not in doubt (a full-HP hallway fight against
a weak encounter).
"""

from __future__ import annotations

import collections
import copy
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from ppo.combat_bench import agents, run
from ppo.pilot_human_decks import _fight_seed

sts2_rs = pytest.importorskip("sts2_rs")

SIMS, BOSS_SIMS = 40, 80
SPEC = agents.mcts_spec(SIMS, BOSS_SIMS, {})


def _fight(key: str, room: str, act: int, **kw: Any) -> dict[str, Any]:
    file, floor = key.split(":")
    return {
        "key": key,
        "file": file,
        "floor": int(floor),
        "act": act,
        "room_type": room,
        "character": "Ironclad",
        "ascension": 10,
        "player": "p_fixture",
        "cohort": "other",
        "run_won": False,
        "run_abandoned": False,
        "belt": None,
        "belt_unused": False,
        "monsters": None,
        "n_enchanted": sum("@" in c for c in kw["deck"]),
        "human": {
            "hp_after": None,
            "damage_taken": 0,
            "hp_healed": 0,
            "max_hp_gained": 0,
            "max_hp_lost": 0,
            "turns": None,
            "died": False,
            "quit": False,
        },
        "flags": [],
        **kw,
    }


# Burning Blood, no healing anywhere in deck or relics, 63/80 against a weak
# hallway: won, and the +6 victory heal is not capped.
CLEAN = _fight(
    "685434af6c06add7.run.gz:3",
    "monster",
    1,
    encounter="SLUDGE_SPINNER_WEAK",
    deck=[
        *("STRIKE_IRONCLAD+1", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD+1", "DEFEND_IRONCLAD+1"),
        *("DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE", "PERFECTED_STRIKE+1"),
        *["STRIKE_IRONCLAD"] * 4,
        "DEFEND_IRONCLAD",
    ],
    relics=["BURNING_BLOOD", "LARGE_CAPSULE", "WAR_PAINT", "MOLTEN_EGG"],
    potions=[],
    hp_before=63,
    max_hp=80,
)
# Toadpoles carry Thorns. The reflected damage takes the engine's unlogged
# `apply_dealhploss` path, so this fight always has HP loss that no damage row
# shows.
THORNS = _fight(
    "aafd68f38d0026b9.run.gz:2",
    "monster",
    1,
    encounter="TOADPOLES_WEAK",
    deck=[
        *["STRIKE_IRONCLAD"] * 5,
        *["DEFEND_IRONCLAD"] * 4,
        *("BASH", "ASCENDERS_BANE", "CONFLAGRATION", "INJURY"),
    ],
    relics=["BURNING_BLOOD", "HEFTY_TABLET"],
    potions=[],
    hp_before=64,
    max_hp=80,
)
# An act-2 elite with Burning Blood + Meat on the Bone, Feed and Blood Vial.
ELITE = _fight(
    "83b6edbaed9754ce.run.gz:30",
    "elite",
    2,
    encounter="INFESTED_PRISMS_ELITE",
    deck=[
        *("STRIKE_IRONCLAD+1", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD@SPIRAL:1"),
        *("STRIKE_IRONCLAD", "DEFEND_IRONCLAD+1", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"),
        *("DEFEND_IRONCLAD", "BASH", "ASCENDERS_BANE", "DISMANTLE", "THUNDERCLAP", "DOMINATE+1"),
        *("RAGE+1", "SHRUG_IT_OFF", "ARMAMENTS+1", "FEED+1", "RUPTURE+1", "TREMBLE"),
        *("BLOODLETTING", "BREAKTHROUGH", "TWIN_STRIKE"),
    ],
    relics=[
        *("BURNING_BLOOD", "LAVA_ROCK", "LUCKY_FYSH", "HAPPY_FLOWER", "CAPTAINS_WHEEL"),
        *("RED_MASK", "CENTENNIAL_PUZZLE", "VERY_HOT_COCOA", "BLOOD_VIAL", "WAR_PAINT"),
        *("MEAT_ON_THE_BONE", "PEN_NIB"),
    ],
    potions=[],
    hp_before=50,
    max_hp=87,
)
# A potion to drink, Crimson Mantle (unlogged self-damage), 31/91 HP.
POTION = _fight(
    "f5bcb57deede874f.run.gz:24",
    "monster",
    2,
    encounter="HUNTER_KILLER_NORMAL",
    deck=[
        *("STRIKE_IRONCLAD", "STRIKE_IRONCLAD"),
        *["DEFEND_IRONCLAD@GOOPY:1"] * 4,
        *("BASH", "ASCENDERS_BANE", "SHRUG_IT_OFF+1", "SHRUG_IT_OFF", "BLUDGEON"),
        *("BLOODLETTING", "BLOODLETTING", "DOMINATE", "CRIMSON_MANTLE+1", "SPITE", "PYRE"),
        *("BREAKTHROUGH", "ULTIMATE_STRIKE", "BATTLE_TRANCE", "STAMPEDE", "BLOOD_WALL+1"),
    ],
    relics=[
        *("BURNING_BLOOD", "NUTRITIOUS_OYSTER", "VAJRA", "REPTILE_TRINKET", "JOSS_PAPER"),
        *("AMETHYST_AUBERGINE", "PAELS_CLAW"),
    ],
    potions=["LIQUID_BRONZE"],
    hp_before=31,
    max_hp=91,
)
FIGHTS = [CLEAN, THORNS, ELITE, POTION]


# The CLI arms its per-cell watchdog with SIGALRM, which Windows lacks.
_SIGALRM = pytest.mark.skipif(sys.platform == "win32", reason="no SIGALRM on Windows")


@pytest.fixture
def engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-process equivalent of `apply_env({})`: no inherited engine seam."""
    for key in list(os.environ):
        if agents.is_engine_env(key):
            monkeypatch.delenv(key)


def _player_damage(rm: Any) -> list[dict[str, Any]]:
    return [r for r in rm.combat_log() if run._is_player_damage(r)]


def _play(fight: dict[str, Any], rep: int = 0, caps: run.Caps | None = None, spec=SPEC):
    """Play in-process, keeping the live run for log inspection."""
    seed = run.cell_seed(fight["key"], rep)
    rm = run.build_run(fight, seed)
    rec = run.play(rm, agents.make_agent(spec, seed), fight, caps or run.Caps())
    return rm, rec


# ── agents a factory spec can name (spawned workers import this module) ──


class NoMove:
    """Never has a move: the `no_action` end."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def step(self, rm: Any) -> dict[str, Any] | None:
        return None


class Crash:
    """Kills its worker process outright, like a segfault or an OOM kill.

    `after` legal moves are played first, so `after=1` passes the runner's
    one-decision probe and dies inside a real cell.
    """

    def __init__(self, seed: int, after: int = 0) -> None:
        self.seed, self.left = seed, after

    def step(self, rm: Any) -> dict[str, Any] | None:
        if self.left <= 0:
            os._exit(13)
        self.left -= 1
        act = rm.available_actions()[0]
        rm.take_action(act)
        return {"action": act, "combat_search": None}


class Hang:
    """Blocks SIGALRM and sleeps: a search stuck inside one engine call.

    The in-worker alarm cannot fire while the signal is blocked, exactly as it
    cannot while a single Rust search runs, so only the parent's kill ends it.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def step(self, rm: Any) -> dict[str, Any] | None:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        time.sleep(600)
        return None


class FirstLegal:
    """Plays the first available action; no search, no budget readout.

    With `stall_after=n` it sleeps 600 s instead of taking decision n + 1: a
    stuck agent whose partial record is exactly n steps, whatever the load.
    """

    def __init__(self, seed: int, stall_after: int = 0) -> None:
        self.seed, self.stall_after, self.steps = seed, stall_after, 0

    def step(self, rm: Any) -> dict[str, Any] | None:
        if self.stall_after and self.steps >= self.stall_after:
            time.sleep(600)
        acts = rm.available_actions()
        if not acts:
            return None
        rm.take_action(acts[0])
        self.steps += 1
        return {"action": acts[0], "combat_search": None}


# ── pure parts ───────────────────────────────────────────────────────────


def test_cell_seed_is_the_pilot_human_decks_formula():
    for fight in FIGHTS:
        for rep in (0, 1, 49):
            assert run.cell_seed(fight["key"], rep) == _fight_seed(fight["key"]) * 100 + rep
    # Reps 50+ are rollout_learn's training seeds (gen `--rep0` 50, the RL pool's 77).
    for rep in (50, 77, 100):
        with pytest.raises(ValueError, match="training seeds"):
            run.cell_seed(CLEAN["key"], rep)


def test_victory_heals_follow_the_engine_listener_order():
    bb, motb, black = ["BURNING_BLOOD"], ["BURNING_BLOOD", "MEAT_ON_THE_BONE"], ["BLACK_BLOOD"]
    assert run.after_victory_heals(50, 80, bb) == 56
    assert run.after_victory_heals(78, 80, bb) == 80  # capped
    assert run.after_victory_heals(0, 80, bb) == 0  # a dead player is not healed
    # Meat on the Bone reads HP after Burning Blood: 34+6 = 40 <= 40 heals...
    assert run.after_victory_heals(34, 80, motb) == 52
    # ...35+6 = 41 > 40 does not.
    assert run.after_victory_heals(35, 80, motb) == 41
    assert run.after_victory_heals(30, 80, black) == 42
    assert run.after_victory_heals(30, 80, []) == 30


def test_hp_before_victory_prefers_the_exact_checkpoint():
    bb = ["BURNING_BLOOD"]
    assert run.hp_before_victory(56, 80, bb, guess=50) == (50, False, False)
    # Capped: 75..80 all heal to 80; the checkpoint picks the one it saw.
    assert run.hp_before_victory(80, 80, bb, guess=77) == (77, False, False)
    # An unlogged 2 HP loss after the checkpoint: inverted exactly, flagged.
    assert run.hp_before_victory(54, 80, bb, guess=50) == (48, True, False)
    assert run.hp_before_victory(56, 80, [], guess=56) == (56, False, False)
    # Meat on the Bone makes the heal map non-monotonic: 34 (+6, then +12 at
    # <= 40) and 46 (+6) both end on 52. A stale guess cannot tell them
    # apart, so the inverse is ambiguous and `play` books no damage from it.
    motb = ["BURNING_BLOOD", "MEAT_ON_THE_BONE"]
    assert run.hp_before_victory(52, 80, motb, guess=40) == (34, True, True)
    assert run.hp_before_victory(52, 80, motb, guess=46) == (46, False, False)


def test_ledger_splits_unexplained_drops_and_rises():
    led = run.HpLedger(50)
    led.observe(52)  # start-of-combat heal (Blood Vial)
    led.rows([{"hp_before": 52, "hp_after": 45}])  # logged 7
    led.observe(44)  # 1 unlogged (Inferno tick)
    led.rows([{"hp_before": 41, "hp_after": 41}, {"hp_before": 41, "hp_after": 30}])
    assert (led.cur, led.unlogged, led.healed) == (30, 4, 2)


def test_cells_are_seeded_shuffled_and_blocked():
    keys = [f["key"] for f in FIGHTS]
    a = run.make_cells(keys, reps=3, rep_offset=2, order_seed=7, block_size=5)
    assert a == run.make_cells(keys, reps=3, rep_offset=2, order_seed=7, block_size=5)
    assert a != run.make_cells(keys, reps=3, rep_offset=2, order_seed=8, block_size=5)
    assert sorted((k, r) for _, k, r in a) == sorted((k, r) for k in keys for r in (2, 3, 4))
    assert [b for b, _, _ in a] == [i // 5 for i in range(12)]


def test_slowest_first_starts_unscored_then_slow_cells_and_keeps_blocks(tmp_path: Path):
    def line(key: str, rep: int, end: str, cpu_s: float | None) -> str:
        return json.dumps({"key": key, "block": 0, "rep": {"rep": rep, "end": end, "cpu_s": cpu_s}})

    hint = tmp_path / "prior.jsonl"
    hint.write_text(
        "\n".join(
            [
                json.dumps({"header": {}}),
                line("a", 0, "won", 1.0),
                line("a", 1, "died", 9.0),
                line("b", 0, "watchdog", None),
                line("c", 0, "error", None),
                line("c", 0, "won", 2.0),  # a retry's later line wins
                line("d", 0, "won", 5.0),
            ]
        )
        + '\n{"key": "cut'
    )
    cells = [(0, "a", 0), (0, "a", 1), (1, "b", 0), (1, "c", 0), (2, "a", 2), (2, "e", 0)]
    order = run.slowest_first(cells, str(hint))
    assert sorted(order) == sorted(cells)  # same cells, same blocks
    # b unscored (inf) > a1 9 > a2 (a's mean 5) = e (median of 1,9,2,5 -> 5) > c 2 > a0 1
    assert order[0] == (1, "b", 0)
    assert order[1] == (0, "a", 1)
    assert set(order[2:4]) == {(2, "a", 2), (2, "e", 0)}
    assert order[4:] == [(1, "c", 0), (0, "a", 0)]


def test_spec_env_merges_flags_and_refuses_silent_no_ops(tmp_path: Path):
    net = tmp_path / "net.bin"
    net.write_bytes(b"weights")
    env = agents.build_env({"STS2_X": "1"}, overrides="c_puct=2", rollout_net=str(net), clair=True)
    spec = agents.mcts_spec(10, 20, env)
    assert spec["env"] == {
        "STS2_ROLLOUT_NET": str(net),
        "STS2_RUNAI_HONEST": "0",
        "STS2_RUNAI_MCTS_OVERRIDES": "belief_dialect=1,c_puct=2",
        "STS2_X": "1",
    }
    assert spec["overrides"] == "belief_dialect=1,c_puct=2" and spec["rollout_net"] == str(net)
    # --clair keeps the deployed search dialect, so it differs in information alone.
    assert spec["search_dialect"] == "belief-draw-clairvoyant"
    assert agents.search_dialect({"STS2_RUNAI_HONEST": "0"}) == "legacy"
    assert spec["rollout_net_sha256"] is not None
    with pytest.raises(ValueError, match="rollout-temp"):
        agents.build_env({}, rollout_temp=0.5)
    with pytest.raises(ValueError, match="both"):
        agents.build_env({"STS2_RUNAI_HONEST": "0"}, clair=True)
    with pytest.raises(ValueError, match="STS2_RUNAI_HONEST"):
        agents.build_env({"STS2_RUNAI_HONEST": "1"})  # the engine would panic on every cell
    assert agents.build_env({"STS2_RUNAI_HONEST": "full"}) == {"STS2_RUNAI_HONEST": "full"}
    f = agents.factory_spec("a.b:c", ["n=3", "name=x"], {})
    assert f["args"] == {"n": 3, "name": "x"} and f["kind"] == "factory"
    with pytest.raises(ValueError):
        agents.factory_spec("a.b:c", ["seed=1"], {})


def test_apply_env_scrubs_inherited_engine_seams(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STS2_RUNAI_HONEST", "0")
    monkeypatch.setenv("STS2_ROLLOUT_TEMP", "2")
    monkeypatch.setenv("STS2_GAME_DIR", "/keep")
    scrubbed = agents.apply_env({"STS2_ROLLOUT_TEMP": "1"})
    assert scrubbed == ["STS2_RUNAI_HONEST"]
    assert "STS2_RUNAI_HONEST" not in os.environ
    assert os.environ["STS2_ROLLOUT_TEMP"] == "1" and os.environ["STS2_GAME_DIR"] == "/keep"


# ── one fight, in-process ────────────────────────────────────────────────


def test_runner_reproduces_the_legacy_pilot_harness(engine_env):
    """Same wheel, same seeds: `run` plays exactly what `pilot_human_decks.play_fight` plays.

    That keeps the new bank continuous with every number the old harness
    produced, as long as the engine is the same.
    """
    from ppo import pilot_human_decks as phd

    for fight in (CLEAN, ELITE, POTION):
        legacy = phd.play_fight(
            (fight, 2, set(), _fight_seed(fight["key"]), (SIMS, BOSS_SIMS, False, None), {"orig"})
        )
        for rep, old in enumerate(legacy["arms"]["orig"]):
            new = run.play_cell(fight, rep, SPEC, run.Caps())
            for field in ("died", "steps", "dmg", "turns", "hp_after", "drunk", "potions_left"):
                assert old[field] == new[field], (fight["key"], rep, field)


def test_play_is_deterministic(engine_env):
    for fight in (ELITE, POTION):
        a = run.play_cell(fight, 3, SPEC, run.Caps())
        b = run.play_cell(fight, 3, SPEC, run.Caps())
        assert a["end"] in ("won", "died")
        for x in (a, b):  # what the machine measured, not what the fight did
            x.pop("cpu_s"), x.pop("load1")
        assert a == b


def test_damage_rows_and_hp_end_on_a_clean_burning_blood_win(engine_env):
    rm, rec = _play(CLEAN)
    rows = _player_damage(rm)
    assert rec["end"] == "won" and not rec["died"]
    assert rec["dmg_c"] == sum(r["hp_before"] - r["hp_after"] for r in rows)
    assert rec["dmg"] == sum(r["unblocked"] for r in rows) >= rec["dmg_c"]
    assert 0 < rec["dmg_c"] <= CLEAN["hp_before"]
    # No heal source and no unlogged self-damage in this fight, so the end HP is
    # entry HP minus the rows, found without inference...
    assert rec["dmg_unlogged"] == 0 and "hp_end_inferred" not in rec
    assert rec["hp_end_combat"] == CLEAN["hp_before"] - rec["dmg_c"]
    assert rows and rows[-1]["hp_after"] == rec["hp_end_combat"]
    # ...and the run's HP afterwards carries Burning Blood's +6 on top.
    assert rec["hp_after"] == min(rec["hp_end_combat"] + 6, rec["max_hp_after"])
    assert rec["hp_after"] > rec["hp_end_combat"]
    assert rec["max_hp_after"] == CLEAN["max_hp"]
    assert rec["turns"] == max(r["round"] for r in rm.combat_log())
    assert 0 < rec["decisions"] <= rec["steps"]
    assert rec["sims_spent"] == SIMS * rec["decisions"]


def test_unlogged_thorns_damage_is_booked_and_the_end_hp_still_balances(engine_env):
    # Whether the agent hits a Toadpole while its Thorns are up depends on the
    # line it plays, which moves with the build's card content (v0.113.0 rep 0
    # kills both before Spike Spit's Thorns matter). Play a few reps: every one
    # must balance, and at least one must book Thorns damage.
    unlogged = []
    for rep in range(4):
        rm, rec = _play(THORNS, rep)
        assert rec["end"] == "won", (rep, rec)
        rows = _player_damage(rm)
        assert rec["dmg_c"] == sum(r["hp_before"] - r["hp_after"] for r in rows)
        # No heal source: every HP lost is a row or unlogged, nothing else.
        assert rec["hp_end_combat"] == THORNS["hp_before"] - rec["dmg_c"] - rec["dmg_unlogged"]
        assert rec["hp_after"] == min(rec["hp_end_combat"] + 6, rec["max_hp_after"])
        unlogged.append(rec["dmg_unlogged"])
    assert max(unlogged) > 0, unlogged


def test_every_fixture_balances_and_heals_are_never_negative(engine_env):
    for fight in FIGHTS:
        for rep in (0, 1):
            rm, rec = _play(fight, rep)
            assert rec["end"] in ("won", "died"), (fight["key"], rec)
            healed = rec["hp_end_combat"] - fight["hp_before"] + rec["dmg_c"] + rec["dmg_unlogged"]
            assert healed >= 0, (fight["key"], rec)
            if rec["died"]:
                assert rec["hp_end_combat"] == 0 and rec["hp_after"] == 0
            else:
                assert rec["hp_end_combat"] > 0
                relics = [
                    r["id"] for r in sts2_rs.state_to_dict(rm.combat_state())["player"]["relics"]
                ]
                final_hp = sts2_rs.state_to_dict(rm.combat_state())["player"]["hp"]
                assert (
                    run.after_victory_heals(rec["hp_end_combat"], rec["max_hp_after"], relics)
                    == final_hp
                )
            drunk = collections.Counter(
                r["potion"] for r in rm.combat_log() if r["kind"] == "potion"
            )
            assert rec["drunk"] == sum(drunk.values())
            left = collections.Counter(fight["potions"]) - drunk
            assert rec["potions_left"] == sum(left.values())


def test_caps_end_the_fight_deterministically(engine_env):
    _, rec = _play(ELITE, caps=run.Caps(max_steps=3))
    assert (rec["end"], rec["steps"], rec["died"]) == ("step_cap", 3, False)
    _, rec = _play(ELITE, caps=run.Caps(max_rounds=1))
    assert rec["end"] == "round_cap" and rec["steps"] > 0 and rec["turns"] == 2
    _, rec = _play(ELITE, caps=run.Caps(max_sims_per_fight=50))
    assert rec["end"] == "sims_cap" and 50 <= rec["sims_spent"] < 50 + SIMS
    # An unscored end reports the in-combat HP: no victory heal to undo.
    assert rec["hp_after"] == rec["hp_end_combat"]


def test_no_action_error_and_non_mcts_agents(engine_env):
    spec = agents.factory_spec(f"{__name__}:NoMove", [], {})
    rec = run.play_cell(CLEAN, 0, spec, run.Caps())
    assert (rec["end"], rec["steps"], rec["sims_spent"]) == ("no_action", 0, None)
    spec = agents.factory_spec(f"{__name__}:FirstLegal", [], {})
    rec = run.play_cell(CLEAN, 0, spec, run.Caps(max_steps=5))
    assert rec["sims_spent"] is None and rec["decisions"] == 0 and rec["steps"] > 0
    bad = copy.deepcopy(CLEAN) | {"relics": ["NOT_A_RELIC_ANYWHERE"]}
    rec = run.play_cell(bad, 0, SPEC, run.Caps())
    assert rec["end"] == "error" and "NOT_A_RELIC" in rec["error"] and not rec["died"]
    assert rec["seed"] == run.cell_seed(CLEAN["key"], 0)


def test_searching_a_clone_leaves_the_live_run_untouched(engine_env):
    seed = run.cell_seed(ELITE["key"], 0)
    rm = run.build_run(ELITE, seed)
    agent = agents.make_agent(SPEC, seed)
    for _ in range(3):
        agent.step(rm)

    def fingerprint(r: Any) -> tuple[Any, ...]:
        cs = r.combat_state()
        state = json.dumps(sts2_rs.state_to_dict(cs), sort_keys=True)
        return cs.state_hash(), state, len(r.combat_log()), r.phase(), r.current_hp, r.max_hp

    before = fingerprint(rm)
    trial = rm.clone()
    rec = agents.make_agent(SPEC, seed + 1).step(trial)
    assert rec is not None and rec["combat_search"]
    assert fingerprint(rm) == before
    # The positive control: the clone itself did move.
    assert fingerprint(trial) != before


# ── honesty ──────────────────────────────────────────────────────────────

# `blindfold::belief_root` reseeds exactly these (`RunRngSet::combat_streams_mut`).
COMBAT_STREAMS = [
    *("shuffle", "monster_ai", "combat_targets", "combat_card_selection"),
    *("combat_energy_costs", "combat_orbs", "combat_card_generation"),
    *("combat_potion_generation", "niche"),
]


def _root_search(d: dict[str, Any], room: str, seed: int, sims: int = 200) -> Any:
    rm = sts2_rs.RunManager(seed=seed, character="Ironclad", ascension_level=10)
    rm.debug_enter_combat(sts2_rs.state_from_dict(d), room)
    rec = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=sims).step(rm)
    return rec["action"], rec["combat_search"]["actions"]


def _perturbed(d: dict[str, Any], streams: list[str], seed_bump: int) -> dict[str, Any]:
    out = copy.deepcopy(d)
    for name in streams:
        out["rng"]["counters"][name] += 5
        out["rng_counters"][name] += 5
    out["rng"]["seed"] += seed_bump
    return out


def _probe_states() -> list[tuple[dict[str, Any], str, int]]:
    out = []
    for fight in (ELITE, POTION, THORNS):
        seed = run.cell_seed(fight["key"], 0)
        rm = run.build_run(fight, seed)
        out.append((sts2_rs.state_to_dict(rm.combat_state()), fight["room_type"], seed))
    return out


def test_honest_search_cannot_see_the_hidden_combat_rng(engine_env, monkeypatch):
    """Permute only the hidden dice: the deployed (honest) root search is unchanged.

    The hidden half of a fight is the draw-pile order and the RNG cursors that
    decide future shuffles, monster rolls and generated cards. This moves only
    the cursors, through a `state_to_dict` round trip, so every observable is
    identical. The honest search must return the same root record, bit for
    bit. The clairvoyant search is the positive control: it plays the real
    dice, so the same perturbation has to move it, or the probe cannot tell
    the two apart.
    """
    honest_same, clair_moved = [], []
    for d, room, seed in _probe_states():
        hidden = _perturbed(d, COMBAT_STREAMS, seed_bump=777)
        honest_same.append(_root_search(d, room, seed) == _root_search(hidden, room, seed))
        monkeypatch.setenv("STS2_RUNAI_HONEST", "0")
        clair_moved.append(_root_search(d, room, seed) != _root_search(hidden, room, seed))
        monkeypatch.delenv("STS2_RUNAI_HONEST")
    assert all(honest_same)
    assert any(clair_moved)


@pytest.mark.xfail(
    strict=True,
    reason="belief_root does not reseed the RUN-level streams (up_front, rewards, shops, "
    "transformations) and chance nodes seed from the parent state_hash "
    "(sts2-mcts lib.rs alloc_chance_node), so their cursors change the honest search's "
    "noise. This is noise, not information (it enters through a hash), but it breaks "
    "belief_root's byte-identity claim. Remove this marker once the root hash ignores them.",
)
def test_honest_search_ignores_run_level_stream_cursors(engine_env):
    for d, room, seed in _probe_states():
        hidden = _perturbed(d, ["up_front", "rewards", "shops", "transformations"], 0)
        assert _root_search(d, room, seed) == _root_search(hidden, room, seed)


# ── the CLI: header, workers, resume ─────────────────────────────────────


def _write_set(path: Path, fights: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(f) + "\n" for f in fights), encoding="utf-8")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _argv(set_path: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--set", str(set_path), "--out", str(out), "--reps", "2", "--procs", "2",
        "--block-size", "3", "--order-seed", "7", "--sims", str(SIMS), "--boss-sims",
        str(BOSS_SIMS), *extra,
    ]  # fmt: skip


@_SIGALRM
def test_cli_writes_header_and_cells_matching_in_process_play(engine_env, tmp_path: Path):
    set_path, out = tmp_path / "ironclad_toy_v1.jsonl", tmp_path / "played.jsonl"
    _write_set(set_path, [CLEAN, POTION])
    assert run.main(_argv(set_path, out)) == 0
    header, *cells = _lines(out)
    h = header["header"]
    assert h["schema"] == "combat_bench.played.v1" and h["set_name"] == "ironclad_toy_v1"
    assert h["spec"] == json.loads(json.dumps(SPEC))
    assert h["src_hash"] == sts2_rs.src_hash()
    assert (h["reps"], h["rep_offset"], h["block_size"], h["order_seed"]) == (2, 0, 3, 7)
    # The sims cap scales with the spec: 400 full-budget decisions (boss_sims 80).
    assert h["caps"] == {"max_rounds": 30, "max_steps": 3000, "max_sims_per_fight": 400 * 80}
    assert h["spec"]["search_dialect"] == "belief-draw" and h["cpus"] == os.cpu_count()
    order = run.make_cells([CLEAN["key"], POTION["key"]], 2, 0, 7, 3)
    assert sorted((c["block"], c["key"], c["rep"]["rep"]) for c in cells) == sorted(order)
    by_key = {f["key"]: f for f in (CLEAN, POTION)}
    for c in cells:
        rep = dict(c["rep"])
        assert rep["seed"] == _fight_seed(c["key"]) * 100 + rep["rep"]
        # Where it ran, for `report`'s CPU-ratio checks.
        assert rep["host"] == socket.gethostname() and isinstance(rep["load1"], float)
        here = run.play_cell(by_key[c["key"]], rep["rep"], SPEC, run.Caps())
        for x in (rep, here):
            x.pop("cpu_s"), x.pop("load1")
        assert rep == here  # a spawned worker plays exactly what this process plays


@_SIGALRM
def test_cli_resume_skips_done_cells_and_refuses_a_changed_header(engine_env, tmp_path: Path):
    set_path, out = tmp_path / "ironclad_toy_v1.jsonl", tmp_path / "played.jsonl"
    _write_set(set_path, [CLEAN])
    assert run.main(_argv(set_path, out)) == 0
    full = out.read_text()
    assert len(full.splitlines()) == 3
    # Same run again: nothing to do, the file is untouched (procs is not in the header).
    assert run.main([*_argv(set_path, out), "--procs", "1"]) == 0
    assert out.read_text() == full
    # A different spec or cap is a different experiment: refused, file untouched.
    assert run.main([*_argv(set_path, out), "--sims", "41"]) == 2
    assert run.main([*_argv(set_path, out), "--max-rounds", "10"]) == 2
    assert out.read_text() == full
    # A run killed mid-write: last cell gone, the next one half-written.
    head, first, second = full.splitlines()
    out.write_text(f"{head}\n{first}\n{second[:20]}")
    assert run.main(_argv(set_path, out)) == 0
    lines = out.read_text().splitlines()
    assert lines[:2] == [head, first] and len(lines) == 3
    replayed, original = json.loads(lines[2]), json.loads(second)
    for x in (replayed, original):
        x["rep"].pop("cpu_s"), x["rep"].pop("load1")
    assert replayed == original  # same cell, same block, same result
    # The watchdog is wall-clock only: a different --watchdog-s may resume.
    assert run.main([*_argv(set_path, out), "--watchdog-s", "7200"]) == 0


def test_cli_rejects_reps_past_the_seed_space(tmp_path: Path):
    set_path = tmp_path / "s.jsonl"
    _write_set(set_path, [CLEAN])
    assert run.main([*_argv(set_path, tmp_path / "o.jsonl"), "--rep-offset", "99"]) == 2
    # Reps 50+ would replay rollout_learn's training seeds.
    assert run.main([*_argv(set_path, tmp_path / "o.jsonl"), "--rep-offset", "49"]) == 2
    assert not (tmp_path / "o.jsonl").exists()


@_SIGALRM
def test_cli_watchdog_ends_a_stuck_cell(engine_env, tmp_path: Path):
    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [CLEAN])
    # The alarm is armed before the spawned cell builds its run and imports this
    # module, which took over a second on a loaded host: a 1 s watchdog then
    # fired before the first decision and the record had no `steps`. The agent
    # stalls after one decision instead, and the alarm has room to spare.
    argv = [
        "--set", str(set_path), "--out", str(out), "--procs", "1", "--watchdog-s", "5",
        "--agent", f"{__name__}:FirstLegal", "--agent-arg", "stall_after=1",
    ]  # fmt: skip
    assert run.main(argv) == 0
    _, cell = _lines(out)
    assert cell["rep"]["end"] == "watchdog" and cell["rep"]["steps"] == 1
    header = _lines(out)[0]["header"]
    assert header["spec"]["kind"] == "factory"
    assert header["caps"]["max_sims_per_fight"] == run.Caps.max_sims_per_fight  # no budget


@_SIGALRM
def test_cli_records_a_dead_cell_process_and_never_hangs(engine_env, tmp_path: Path):
    """A cell whose process dies outright is an `error` cell; the run never hangs.

    When every cell of the first block dies, the spec is at fault and the run
    stops with exit code 2 after writing those cells.
    """
    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [CLEAN, POTION])
    argv = [
        "--set", str(set_path), "--out", str(out), "--procs", "2",
        "--agent", f"{__name__}:Crash", "--agent-arg", "after=1",
    ]  # fmt: skip
    assert run.main(argv) == 2
    lines = _lines(out)
    assert "header" in lines[0] and len(lines) == 3
    assert all(line["rep"]["end"] == "error" for line in lines[1:])
    assert all("exit code 13" in line["rep"]["error"] for line in lines[1:])
    # One that dies on its first decision never gets past the probe search.
    out2 = tmp_path / "o2.jsonl"
    assert run.main([*argv[:3], str(out2), *argv[4:-2]]) == 2
    assert not out2.exists()


@_SIGALRM
def test_cli_kills_a_cell_stuck_inside_one_engine_call(
    engine_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The parent kills a cell past the watchdog even when no bytecode runs in it."""
    monkeypatch.setattr(run, "CHECK_EVERY_S", 0.2)
    set_path = tmp_path / "s.jsonl"
    _write_set(set_path, [CLEAN])
    sched = run._CellScheduler(
        agents.factory_spec(f"{__name__}:Hang", [], {}), run.Caps(30, 3000, 10**6), 600, 1
    )
    # The in-worker alarm is armed before the cell builds its run, so on a
    # loaded host a short one fired before `Hang` could block it and the cell
    # ended cleanly. Keep the alarm out of reach and bring the kill in.
    sched.hard_s = 2
    fight = run.load_set(str(set_path))[0]
    t0 = time.monotonic()
    results = list(sched.run([("tag", fight, 0)]))
    sched.close()
    assert time.monotonic() - t0 < 30
    ((tag, res),) = results
    assert tag == "tag" and res["rep"]["end"] == "watchdog"
    assert "killed after 2 s" in res["rep"]["error"]


def test_cli_refuses_a_spec_that_cannot_build_an_agent(engine_env, tmp_path: Path):
    """An unknown override knob panics the engine in every cell: refuse before writing."""
    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [CLEAN])
    assert run.main([*_argv(set_path, out), "--overrides", "no_such_knob=1"]) == 2
    assert not out.exists()


def test_cli_refuses_a_spec_that_panics_at_its_first_search(engine_env, tmp_path: Path):
    """The agent builds, but the engine panics at the first search: still refused.

    `STS2_RUNAI_LEAF=learned` needs `STS2_RUNAI_COMBAT_NET`, and a file that is
    JSON but not a playout net fails when `STS2_ROLLOUT_NET` is loaded; both
    only at the first search, so a constructor-only probe passed them.
    """
    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [CLEAN])
    assert run.main([*_argv(set_path, out), "--env", "STS2_RUNAI_LEAF=learned"]) == 2
    bad = tmp_path / "bad_net.json"
    bad.write_text("{}")
    assert run.main([*_argv(set_path, out), "--rollout-net", str(bad)]) == 2
    assert not out.exists()


@_SIGALRM
def test_cli_aborts_when_the_whole_first_block_errors(engine_env, tmp_path: Path):
    """A fight that cannot build passes the probe (a cell failure), but a first
    block of nothing but errors stops the run."""
    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [{**CLEAN, "encounter": "NO_SUCH_ENCOUNTER"}])
    assert run.main(_argv(set_path, out)) == 2
    cells = _lines(out)[1:]
    assert cells and all(c["rep"]["end"] == "error" for c in cells)


@_SIGALRM
def test_cli_retry_ends_replays_watchdog_cells(engine_env, tmp_path: Path):
    """`--retry-ends watchdog` replays such cells; resume alone never does."""
    from ppo.combat_bench import report

    set_path, out = tmp_path / "s.jsonl", tmp_path / "o.jsonl"
    _write_set(set_path, [CLEAN])
    assert run.main(_argv(set_path, out)) == 0
    head, first, second = out.read_text().splitlines()
    good = json.loads(first)
    stuck = {**good, "rep": {"rep": good["rep"]["rep"], "seed": 1, "end": "watchdog"}}
    out.write_text("\n".join([head, json.dumps(stuck), second]) + "\n")
    assert run.main(_argv(set_path, out)) == 0
    assert len(out.read_text().splitlines()) == 3  # a plain resume leaves it
    assert run.main([*_argv(set_path, out), "--retry-ends", "watchdog"]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 4
    again = json.loads(lines[3])
    for x in (again, good):
        x["rep"].pop("cpu_s"), x["rep"].pop("load1")
    assert again == good  # the same deterministic cell
    assert run.main([*_argv(set_path, out), "--retry-ends", "won"]) == 2  # not machine-made
    played = report.load_played(str(out), "x", report.load_set(str(set_path)))
    assert played.retried == 1 and played.duplicates == 0
    assert played.cells[good["key"], good["rep"]["rep"]]["end"] == good["rep"]["end"]


@_SIGALRM
def test_cli_keys_file_selection_is_what_the_report_expects(engine_env, tmp_path: Path):
    """The header carries a keys-file selection, and `report` rebuilds the same plan."""
    from ppo.combat_bench import report

    set_path, out, keys = tmp_path / "s.jsonl", tmp_path / "o.jsonl", tmp_path / "k.json"
    _write_set(set_path, [CLEAN, THORNS, POTION])
    keys.write_text(json.dumps([POTION["key"], CLEAN["key"]]))
    assert run.main(_argv(set_path, out, "--keys-file", str(keys))) == 0
    header = _lines(out)[0]["header"]
    assert header["select"]["keys"] == [CLEAN["key"], POTION["key"]]  # set order
    fights = report.load_set(str(set_path))
    assert report.selected_keys(header, fights) == [CLEAN["key"], POTION["key"]]
    played = report.load_played(str(out), "x", fights)
    report.resolve_selection(played, fights)
    assert report.cell_summary(played, fights)["expected"] == 4  # 2 planned fights x 2 reps
