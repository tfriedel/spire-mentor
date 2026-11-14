"""GH #803: the PPO env simulates the selected game build.

`PpoRunEnv` used to build its `RunManager` without `game_version`, so a probe
or campaign under `STS2_GAME_VERSION=v0.111.0` silently ran v0.107.1. These
tests pin that the env follows the explicit field first, then the
process-wide selection, and that the default stays v0.107.1.
"""

from __future__ import annotations

import os

import pytest
import sts2_client

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))


@pytest.fixture(autouse=True)
def _syspath(monkeypatch):
    monkeypatch.syspath_prepend(PY)


@pytest.fixture
def restore_version():
    """Undo any ``set_game_version`` a test does (it is process-global)."""
    before = sts2_client.GAME_VERSION
    env = os.environ.get(sts2_client.GAME_VERSION_ENV)
    yield
    sts2_client.set_game_version(before)
    if env is None:
        os.environ.pop(sts2_client.GAME_VERSION_ENV, None)
    else:
        os.environ[sts2_client.GAME_VERSION_ENV] = env


def _env_version(**kwargs) -> str:
    from ppo.env import PpoRunEnv

    env = PpoRunEnv(seed=0, **kwargs)
    env.reset()
    return env.rm.game_version


def test_an_env_on_the_legacy_build_runs_it(restore_version) -> None:
    sts2_client.set_game_version(sts2_client.LEGACY_GAME_VERSION)
    assert _env_version() == "v0.107.1"


def test_env_follows_the_process_selection(restore_version) -> None:
    sts2_client.set_game_version("v0.111.0")
    assert _env_version() == "v0.111.0"


def test_explicit_field_overrides_the_process_selection(restore_version) -> None:
    sts2_client.set_game_version("v0.107.1")
    assert _env_version(game_version="0.111.0") == "v0.111.0"
