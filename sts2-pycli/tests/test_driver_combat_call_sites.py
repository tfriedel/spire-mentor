"""Drive each play driver's session loop one combat action deep (GH #401).

``pick_combat_action_mcts`` itself is well covered (see
``test_realgame_ppo.py::TestCombatReexports`` / ``TestWireLegalityMask``).
What was **not** covered is the two ``run()`` loops that call it, so when
``1be2a92a73`` changed the signature -- dropping ``session``, adding
``mirror``/``banned``, returning a 3-tuple instead of a 2-tuple -- both
drivers kept calling it the old way and nothing went red:

* ``realgame_ppo.run`` passed ``session=`` -> ``TypeError`` on the first
  combat action of its default ``--combat-policy mcts`` lane.
* ``realgame_mcts.run`` unpacked 2 values from the 3-tuple -> ``ValueError``.

Neither call sits under a local ``except``, and the loop's only handler is
``(ConnectionError, TimeoutError)``, so either one kills the session.

These tests substitute a scripted fake for the TCP client and let the loop
run for real. The picker is deliberately **not** patched: it is
never-crash by contract (any failure degrades to the random-legal policy),
so it returns a well-formed 3-tuple with or without a working ``sts2_rs``,
and calling it for real is what pins the call site's arity and kwargs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "realgame_adapter"


def _combat_fixture() -> dict[str, Any]:
    """The captured ``combat_action`` message itself.

    The fixture files wrap it with ``_provenance`` / ``recorded_action``
    metadata; the decision message is the ``state`` member (same unwrap as
    ``test_realgame_ppo._load_combat_fixture``). Feeding the wrapper to a
    driver is a silent no-op -- ``type`` is absent, so the loop classifies it
    as an unknown non-combat message and never reaches the combat branch.
    """
    raw = json.loads((_FIXTURES / "single_enemy_ironclad.json").read_text(encoding="utf-8"))
    return raw["state"]


class _ScriptedBridgeClient:
    """Stands in for :class:`realgame_bridge.RealGameBridgeClient`.

    Yields a fixed list of decision messages and records what the driver
    sent back. Accepts (and ignores) the real client's constructor kwargs so
    it can be monkeypatched straight over the name the driver imported.
    """

    def __init__(self, states: list[dict[str, Any]], **_kwargs: Any) -> None:
        self._states = list(states)
        self.sent: list[dict[str, Any]] = []

    def __enter__(self) -> _ScriptedBridgeClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def receive_state(self) -> dict[str, Any]:
        if not self._states:
            raise AssertionError("driver asked for more states than the script provides")
        return self._states.pop(0)

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent.append(action)


def _script(states: list[dict[str, Any]]) -> tuple[Any, list[_ScriptedBridgeClient]]:
    """Return a client factory plus the list it appends each instance to."""
    made: list[_ScriptedBridgeClient] = []

    def factory(**kwargs: Any) -> _ScriptedBridgeClient:
        client = _ScriptedBridgeClient(states, **kwargs)
        made.append(client)
        return client

    return factory, made


#: One combat decision, then a clean end of run so the loop breaks instead of
#: running off the end of the script.
def _one_fight_then_done() -> list[dict[str, Any]]:
    return [_combat_fixture(), {"type": "run_complete", "result": "victory"}]


@pytest.fixture
def tiny_cfg():
    """Smallest search that still exercises the real pipeline."""
    from sts2_hint.recommender import RecommendConfig

    return RecommendConfig(sims=1, depth=2, seed=0)


class TestCombatCallSites:
    def test_ppo_driver_survives_a_combat_action(self, tmp_path, monkeypatch, tiny_cfg) -> None:
        import realgame_ppo as rp

        factory, made = _script(_one_fight_then_done())
        monkeypatch.setattr(rp, "RealGameBridgeClient", factory)

        rc = rp.run(
            "localhost",
            9999,
            tmp_path,
            None,
            random.Random(0),
            rp.PpoPolicyCache(None, strict=False),
            combat_policy="mcts",
            mcts_cfg=tiny_cfg,
            sims_flat=1,
        )

        assert rc == 0
        assert len(made) == 1
        # The whole point: the loop reached send_action rather than dying on
        # the call itself.
        assert len(made[0].sent) == 1
        assert made[0].sent[0]["action"] in ("play", "end_turn", "potion")

    def test_mcts_driver_survives_a_combat_action(self, tmp_path, monkeypatch, tiny_cfg) -> None:
        import realgame_mcts as rm

        factory, made = _script(_one_fight_then_done())
        monkeypatch.setattr(rm, "RealGameBridgeClient", factory)

        rc = rm.run(
            "localhost",
            9999,
            tmp_path,
            None,
            random.Random(0),
            tiny_cfg,
            sims_flat=1,
        )

        assert rc == 0
        assert len(made) == 1
        assert len(made[0].sent) == 1
        assert made[0].sent[0]["action"] in ("play", "end_turn", "potion")


def _unsupported_build_fight() -> list[dict[str, Any]]:
    """A combat decision from a game build no engine exists for (GH #807)."""
    return [{**_combat_fixture(), "game_version": "v9.9.9"}, {"type": "run_complete"}]


class TestUnsupportedBuildIsRefused:
    """GH #807: an unsupported game build stops the driver instead of falling
    into the never-crash handler's random legal move -- which would play the
    whole run blind while the log looked like an ordinary MCTS fallback."""

    def test_the_picker_reraises_instead_of_playing_random(self, tiny_cfg) -> None:
        import realgame_mcts as rm
        from realgame_adapter import UnsupportedGameVersionError

        with pytest.raises(UnsupportedGameVersionError, match=r"v9\.9\.9"):
            rm.pick_combat_action_mcts(_unsupported_build_fight()[0], tiny_cfg, random.Random(0))

    def test_mcts_driver_refuses(self, tmp_path, monkeypatch, tiny_cfg) -> None:
        import realgame_mcts as rm

        factory, made = _script(_unsupported_build_fight())
        monkeypatch.setattr(rm, "RealGameBridgeClient", factory)

        rc = rm.run("localhost", 9999, tmp_path, None, random.Random(0), tiny_cfg, sims_flat=1)

        assert rc == 2
        assert made[0].sent == []

    def test_ppo_driver_refuses(self, tmp_path, monkeypatch, tiny_cfg) -> None:
        import realgame_ppo as rp

        factory, made = _script(_unsupported_build_fight())
        monkeypatch.setattr(rp, "RealGameBridgeClient", factory)

        rc = rp.run(
            "localhost",
            9999,
            tmp_path,
            None,
            random.Random(0),
            rp.PpoPolicyCache(None, strict=False),
            combat_policy="mcts",
            mcts_cfg=tiny_cfg,
            sims_flat=1,
        )

        assert rc == 2
        assert made[0].sent == []
