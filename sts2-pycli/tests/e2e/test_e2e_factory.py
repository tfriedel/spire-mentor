"""e2e WP1: the version factory and gate F8 (a v0.107.1 object fails loudly).

Rust never reads `STS2_GAME_VERSION`, so an engine call that drops
`game_version=` builds v0.107.1 under any selection. These tests build such
objects on purpose, and inject them into the fight builders and the run env
by wrapping the engine constructors, and expect `GameVersionMismatch`: an
`AssertionError`, never the `ValueError` a fight builder skips a row on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import sts2_client
import sts2_rs
from ppo import rollout_starts
from ppo.combat_bench import run as bench_run
from ppo.e2e import factory as F
from sts2_client import GameVersionMismatch

V107, V111 = "v0.107.1", "v0.111.0"

FIGHT = {
    "file": "unit.run.gz",
    "floor": 3,
    "deck": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH", "POMMEL_STRIKE"],
    "hp_before": 70,
    "max_hp": 80,
    "relics": ["BURNING_BLOOD"],
    "encounter": "NIBBITS_WEAK",
    "ascension": 10,
    "potions": ["FIRE_POTION"],
    "room_type": "elite",
}
SCENARIO = (7, "Ironclad", FIGHT["deck"], 70, 80, FIGHT["relics"], FIGHT["encounter"])
BUILDERS = {"build_fight": rollout_starts.build_fight, "build_run": bench_run.build_run}


def _state(**kw: Any) -> Any:
    return sts2_rs.build_scenario_state(*SCENARIO, ascension=10, potions=["FIRE_POTION"], **kw)


def _run(**kw: Any) -> Any:
    base = {
        "character": "Ironclad",
        "ascension_level": 10,
        "neow_start": True,
        "act1": "roll",
        "mode": "agent",
        "interactive_potion_swap": True,
    }
    return sts2_rs.RunManager(11, **{**base, **kw})


def _mismatch(call: Callable[[], Any]) -> GameVersionMismatch:
    """The `GameVersionMismatch` `call` raises; it must not be a `ValueError`."""
    with pytest.raises(GameVersionMismatch) as info:
        call()
    assert not isinstance(info.value, ValueError)
    return info.value


def _dropping_version(fn: Callable[..., Any], to: str | None = None) -> Callable[..., Any]:
    """`fn` with its `game_version=` dropped (the engine default), or forced to `to`."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("game_version", None)
        if to is not None:
            kwargs["game_version"] = to
        return fn(*args, **kwargs)

    return wrapped


@pytest.fixture(autouse=True)
def _campaign_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select v0.111.0, as a campaign process started with
    `STS2_GAME_VERSION=v0.111.0` does. A test may select another build."""
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V111)


# ── the factory ──────────────────────────────────────────────────────────


def test_new_run_is_a_campaign_run() -> None:
    rm = F.new_run(123)
    got = (rm.game_version, rm.mode, rm.interactive_potion_swap, rm.character, rm.ascension_level)
    assert got == (V111, "agent", True, "Ironclad", 10)
    assert rm.seed == 123
    assert rm.phase() == "event"  # Neow
    assert rm.combat_state() is None


def test_the_factory_refuses_a_process_on_another_build(monkeypatch: pytest.MonkeyPatch) -> None:
    rm, cs = F.new_run(5), F.scenario_state(*SCENARIO)
    monkeypatch.setattr(sts2_client, "GAME_VERSION", V107)
    calls = {
        "new_run": lambda: F.new_run(5),
        "check_run": lambda: F.check_run(rm),
        "check_fight": lambda: F.check_fight(cs),
        "scenario_state": lambda: F.scenario_state(*SCENARIO),
        "require_selection": F.require_selection,
    }
    for name, call in calls.items():
        assert "STS2_GAME_VERSION=v0.111.0" in str(_mismatch(call)), name


def test_new_run_rolls_act_1_from_the_seed() -> None:
    # `act1="roll"`, not the engine's default (always Overgrowth).
    assert {F.new_run(s).act1 for s in range(8)} == {"overgrowth", "underdocks"}


def test_new_run_without_the_potion_swap() -> None:
    rm = F.new_run(5, interactive_potion_swap=False)
    assert rm.interactive_potion_swap is False
    assert F.check_run(rm, interactive_potion_swap=False) is rm
    with pytest.raises(AssertionError, match="interactive_potion_swap") as info:
        F.check_run(rm)
    assert type(info.value) is AssertionError


@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.5, "7", None])
def test_new_run_rejects_seeds_outside_the_range(seed: Any) -> None:
    with pytest.raises(AssertionError, match="outside"):
        F.new_run(seed)


@pytest.mark.parametrize("kw", [{"game_version": V107}, {}], ids=["v0.107.1", "unversioned"])
def test_check_run_rejects_a_v0107_run(kw: dict[str, Any]) -> None:
    rm = _run(**kw)
    assert rm.game_version == V107
    assert "v0.107.1" in str(_mismatch(lambda: F.check_run(rm)))


@pytest.mark.parametrize(
    "kw",
    [{"mode": "parity"}, {"character": "Silent"}, {"ascension_level": 9}],
    ids=["parity", "silent", "a9"],
)
def test_check_run_rejects_other_runs(kw: dict[str, Any]) -> None:
    rm = _run(game_version=V111, **kw)
    with pytest.raises(AssertionError, match="not a campaign run") as info:
        F.check_run(rm)
    assert type(info.value) is AssertionError


def test_check_run_passes_a_clone() -> None:
    rm = F.new_run(9)
    assert F.check_run(rm.clone()).game_version == V111


def test_check_fight_passes_a_v0111_state_only() -> None:
    cs = _state(game_version=V111)
    assert F.check_fight(cs) is cs
    _mismatch(lambda: F.check_fight(_state()))  # unversioned: v0.107.1
    _mismatch(lambda: F.check_fight(_state(game_version=V107)))
    _mismatch(lambda: F.check_fight(None))  # outside combat
    _mismatch(lambda: F.check_fight(object()))  # no getter


def test_check_fight_on_a_run_combat_state() -> None:
    ok = rollout_starts.build_fight(FIGHT, 7, game_version=V111)
    assert F.check_fight(ok.combat_state()).game_version == V111
    old = rollout_starts.build_fight(FIGHT, 7, game_version=V107)
    _mismatch(lambda: F.check_fight(old.combat_state()))


def test_scenario_state_builds_v0111_only() -> None:
    cs = F.scenario_state(*SCENARIO, ascension=10, potions=["FIRE_POTION"])
    assert cs.game_version == V111
    assert F.scenario_state(*SCENARIO, game_version="0.111.0").game_version == V111
    _mismatch(lambda: F.scenario_state(*SCENARIO, game_version=V107))
    _mismatch(lambda: F.scenario_state(*SCENARIO, game_version="v0.103.2"))


def test_version_errors_are_assertion_errors() -> None:
    assert issubclass(GameVersionMismatch, AssertionError)
    assert not issubclass(GameVersionMismatch, ValueError)
    _mismatch(lambda: sts2_client.check_game_version(_state(), "v0.103.2"))
    assert sts2_client.require_game_version("0.111.0") == V111
    _mismatch(lambda: sts2_client.require_game_version("v0.103.2"))


# ── F8: the fight builders ───────────────────────────────────────────────


@pytest.mark.parametrize("version", [V107, V111])
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_fight_builders_follow_the_selection(
    name: str, version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sts2_client, "GAME_VERSION", version)
    rm = BUILDERS[name](FIGHT, 7)
    assert rm.phase() == "combat"
    assert (rm.game_version, rm.combat_state().game_version) == (version, version)


@pytest.mark.parametrize("target", ["build_scenario_state", "RunManager"])
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_f8_a_v0107_object_in_a_fight_builder_fails_loudly(
    name: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sts2_rs, target, _dropping_version(getattr(sts2_rs, target)))
    err = _mismatch(lambda: BUILDERS[name](FIGHT, 7))
    assert "'v0.107.1', expected 'v0.111.0'" in str(err)


def test_f8_a_mismatch_escapes_a_skip_on_value_error_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reset loop shape of `combat_ppo.actors`: a `ValueError` skips the row."""
    monkeypatch.setattr(
        sts2_rs, "build_scenario_state", _dropping_version(sts2_rs.build_scenario_state)
    )
    tried = []

    def reset() -> Any:
        for seed in range(3):
            tried.append(seed)
            try:
                return rollout_starts.build_fight(FIGHT, seed)
            except ValueError:
                continue
        return None

    _mismatch(reset)
    assert tried == [0]


def test_build_fight_rejects_an_unsupported_build() -> None:
    _mismatch(lambda: rollout_starts.build_fight(FIGHT, 7, game_version="v0.103.2"))


# ── F8: the run env ──────────────────────────────────────────────────────


def test_f8_a_v0107_run_in_the_run_env_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PpoRunEnv`, which the e2e env wraps, builds its run on the selected
    build; `check_run` on `env.rm` passes it and catches an injected
    v0.107.1 run (WP2 calls it after every reset and clone)."""
    from ppo.env import PpoRunEnv

    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")
    kw = {"seed": 5, "ascension": 10, "character": "Ironclad", "combat": "yield", "act1": "roll"}
    assert F.check_run(PpoRunEnv(**kw).rm).game_version == V111
    with monkeypatch.context() as m:
        m.setattr(sts2_rs, "RunManager", _dropping_version(sts2_rs.RunManager, to=V107))
        env = PpoRunEnv(**kw)
    assert env.rm.game_version == V107
    _mismatch(lambda: F.check_run(env.rm))
