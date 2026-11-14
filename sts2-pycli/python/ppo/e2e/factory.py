"""The version factory: the only construction site for e2e campaign runs.

Plan section 3.1 (WP1, GH #1097). Every run of the Ironclad A10 end-to-end
campaign is built here, on v0.111.0, in agent mode, in a process started with
`STS2_GAME_VERSION=v0.111.0`:

    rm = new_run(seed)                     # a fresh run, checked
    check_run(rm)                          # any run the env did not build (a clone)
    cs = check_fight(rm.combat_state())    # at every fight start
    cs = scenario_state(seed, ...)         # a scenario fight, on v0.111.0

Rust never reads `STS2_GAME_VERSION`, so a `RunManager(...)` or
`build_scenario_state(...)` call without `game_version=` silently simulates
v0.107.1. Each helper names the build explicitly and then reads it back from
the engine object. The Python side does read the selection (the combat
context's static tables, `PpoRunEnv`'s default build, the fight builders), so
every helper also requires this process to select v0.111.0
(`require_selection`). A mismatch raises `sts2_client.GameVersionMismatch` and
any other broken invariant an `AssertionError`. Neither is a `ValueError`,
on purpose: fight builders read a `ValueError` as "this row does not build"
and skip the row. The checks are explicit raises, not `assert` statements, so
`python -O` keeps them.

`tests/e2e/test_version_guard.py` fails on any engine call without
`game_version=` in this package, `ppo/combat_ppo/`, `ppo/rollout_starts.py`
and `ppo/combat_bench/`.
"""

from __future__ import annotations

import os
from typing import Any

import sts2_client

GAME_VERSION = "v0.111.0"
# The campaign's character. A process plays one: `STS2_E2E_CHARACTER` names it
# before the process starts (spawned workers inherit it, as with the game
# version); unset, it is Ironclad. Another character's rows carry the "chars"
# fill event (GH #1373, SCHEMA.md section 13), so its learner declares
# `--filled` with "chars" and `--character`.
CHARACTER_ENV = "STS2_E2E_CHARACTER"
CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")


def _character() -> str:
    raw = os.environ.get(CHARACTER_ENV, "").strip()
    if not raw:
        return CHARACTERS[0]
    by_lower = {c.lower(): c for c in CHARACTERS}
    if raw.lower() not in by_lower:
        raise ValueError(f"{CHARACTER_ENV}={raw!r}: not one of {CHARACTERS}")
    return by_lower[raw.lower()]


CHARACTER = _character()

# Whether this process's Ironclad rows carry the "chars" fill event (orbs and
# Doom on an Ironclad run). Another character's rows always do. The learner
# sets it from `--filled` before it spawns its workers; unset, Ironclad rows
# are the frozen schema's.
CHARS_ENV = "STS2_E2E_CHARS"


def fill_chars() -> bool:
    """Whether a row of this process's `CHARACTER` carries the "chars" event."""
    return CHARACTERS[0] != CHARACTER or os.environ.get(CHARS_ENV, "") == "1"


def set_fill_chars(on: bool) -> None:
    """Declare (or drop) the "chars" event for this process and its children."""
    if on:
        os.environ[CHARS_ENV] = "1"
    else:
        os.environ.pop(CHARS_ENV, None)


ASCENSION = 10
MODE = "agent"
ACT1 = "roll"
# `RunManager` takes an i32 seed; campaign seeds are non-negative.
SEED_LIMIT = 2**31


def require_selection() -> None:
    """Raise unless this process selects the campaign build, v0.111.0.

    Rust ignores the selection, but the Python side reads it: the combat
    context's static tables (`static_tables.card_types()`), `PpoRunEnv`'s
    default build, and the `rollout_starts` and `combat_bench` fights. A
    process on another selection would build v0.111.0 runs here and encode
    them with v0.107.1 tables.

    Raises:
        sts2_client.GameVersionMismatch: The process selects another build.
    """
    if sts2_client.GAME_VERSION != GAME_VERSION:
        raise sts2_client.GameVersionMismatch(
            f"this process selects {sts2_client.GAME_VERSION}, the campaign needs "
            f"{GAME_VERSION}: set {sts2_client.GAME_VERSION_ENV}={GAME_VERSION} before "
            "the process starts (spawned workers inherit it) or call "
            "sts2_client.set_game_version"
        )


def new_run(seed: int, *, interactive_potion_swap: bool = True) -> Any:
    """A fresh campaign run: `CHARACTER` (Ironclad unless `STS2_E2E_CHARACTER`
    names another) at A10 on v0.111.0, agent mode, from Neow, act 1 rolled
    from the seed.

    Args:
        seed: The run seed, in [0, 2**31).
        interactive_potion_swap: Whether a potion reward on a full belt
            surfaces a `potion_swap` decision. The student plays with it on
            (D4); an R1-style arm passes False.

    Raises:
        sts2_client.GameVersionMismatch: The process selects another build
            (`require_selection`), or the engine built one.
        AssertionError: The seed is out of range, or the engine built another
            kind of run (`check_run`).
    """
    import sts2_rs

    require_selection()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_LIMIT:
        raise AssertionError(f"seed {seed!r} outside [0, 2**31)")
    rm = sts2_rs.RunManager(
        seed,
        character=CHARACTER,
        ascension_level=ASCENSION,
        neow_start=True,
        act1=ACT1,
        game_version=GAME_VERSION,
        mode=MODE,
        interactive_potion_swap=interactive_potion_swap,
    )
    return check_run(rm, interactive_potion_swap=interactive_potion_swap)


def check_run(rm: Any, *, interactive_potion_swap: bool = True) -> Any:
    """Return `rm` once it reads back as a campaign run.

    Checks the build (v0.111.0), the mode (agent), the potion-swap flag, the
    character and the ascension. Call it on every run the env holds that
    `new_run` did not just build: a clone, a practice snapshot, a run another
    module constructed.

    Raises:
        sts2_client.GameVersionMismatch: `rm` simulates another build, or the
            process selects one (`require_selection`).
        AssertionError: `rm` differs in mode, flag, character or ascension.
    """
    require_selection()
    sts2_client.check_game_version(rm, GAME_VERSION, what="RunManager")
    got = (rm.mode, rm.interactive_potion_swap, rm.character, rm.ascension_level)
    want = (MODE, interactive_potion_swap, CHARACTER, ASCENSION)
    if got != want:
        raise AssertionError(
            "RunManager is not a campaign run: (mode, interactive_potion_swap, "
            f"character, ascension_level) = {got}, expected {want}"
        )
    return rm


def check_fight(cs: Any, *, what: str = "combat state") -> Any:
    """Return `cs` once it simulates v0.111.0: the fight-start check.

    Reads the `CombatState.game_version` getter, so a state from an
    unversioned `build_scenario_state` call or from a v0.107.1 run fails here
    instead of playing the old build's cards and monsters.

    Raises:
        sts2_client.GameVersionMismatch: `cs` simulates another build (or is
            not a combat state at all, such as None outside combat), or the
            process selects one (`require_selection`).
    """
    require_selection()
    return sts2_client.check_game_version(cs, GAME_VERSION, what=what)


def scenario_state(*args: Any, game_version: str = GAME_VERSION, **kwargs: Any) -> Any:
    """`sts2_rs.build_scenario_state(*args, **kwargs)` on v0.111.0, checked.

    Takes `build_scenario_state`'s own arguments: seed, character, deck, hp,
    max_hp, relics, encounter, then its keywords.

    Raises:
        sts2_client.GameVersionMismatch: `game_version` names another build
            (this helper builds v0.111.0 only), the engine built one, or the
            process selects one (`require_selection`).
    """
    import sts2_rs

    require_selection()
    # Not `normalize_game_version`: its ValueError on an unknown build is the
    # exception fight builders skip on.
    if str(game_version).removeprefix("v") != GAME_VERSION.removeprefix("v"):
        raise sts2_client.GameVersionMismatch(
            f"scenario_state builds {GAME_VERSION} only, not {game_version!r}"
        )
    cs = sts2_rs.build_scenario_state(*args, game_version=GAME_VERSION, **kwargs)
    return check_fight(cs, what="scenario state")
