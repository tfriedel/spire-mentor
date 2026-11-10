"""Unit tests for the Rust-backend rest-site option rendering.

The offered set is dynamic past the base ``[Rest, Smith]`` (Shovel appends
Dig, Girya Lift, PumpkinCandle Kindle, MeatCleaver Cook, a ByrdonisEgg Hatch),
so the backend must name each option from the engine's own ``option_id``
rather than from its index — an index-keyed table shipped a bare "Option 2"
plaque for Kindle (bug report ``wb-500a91f0d29256f6``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("sts2_rs", reason="rust backend requires the sts2_rs extension")

from fake_rm import FakeRm
from workbench.backends.rust_game import RustGame


class _RestRm(FakeRm):
    """A rest site offering ``options`` as ``(option_id, enabled)`` pairs."""

    def __init__(self, options: list[tuple[str, bool]]) -> None:
        super().__init__(
            phase="rest_site",
            actions=[{"action": "rest_option", "idx": i} for i in range(len(options))],
        )
        self._options = options

    def rest_site_options(self) -> list[tuple[str, bool]]:
        return list(self._options)


def _state(options: list[tuple[str, bool]]) -> dict:
    g = RustGame()
    g.rm = _RestRm(options)
    return g.get_state()


def test_relic_added_option_is_named_not_numbered() -> None:
    state = _state([("HEAL", True), ("SMITH", True), ("KINDLE", True)])
    opts = state["options"]
    assert [o["index"] for o in opts] == [0, 1, 2]
    # The engine's own ids — what keys both the plaque art
    # (ui/rest_site/option_<id>.png) and the rest_site_ui loc row.
    assert [o["option_id"] for o in opts] == ["HEAL", "SMITH", "KINDLE"]
    assert [o["name"] for o in opts] == [
        "rest_site.OPTION_HEAL.name",
        "rest_site.OPTION_SMITH.name",
        "rest_site.OPTION_KINDLE.name",
    ]
    assert all("Option" not in o["option_id"] for o in opts)


def test_disabled_option_reports_is_enabled_false() -> None:
    state = _state([("HEAL", True), ("SMITH", False)])
    assert [o["is_enabled"] for o in state["options"]] == [True, False]


def test_falls_back_to_index_labels_without_the_accessor() -> None:
    """A wheel predating ``rest_site_options()`` still names the base two."""

    def _no_accessor() -> list[tuple[str, bool]]:
        raise AttributeError("rest_site_options")

    rm = _RestRm([("HEAL", True), ("SMITH", True)])
    rm.rest_site_options = _no_accessor
    g = RustGame()
    g.rm = rm
    opts = g.get_state()["options"]
    assert [o["option_id"] for o in opts] == ["Rest", "Smith"]


class _EngineRestRm(_RestRm):
    """Like the real engine: a disabled option is listed but not an action."""

    def __init__(self, options: list[tuple[str, bool]], extra: list[dict] | None = None) -> None:
        super().__init__(options)
        self._actions = [
            {"action": "rest_option", "idx": i} for i, (_, en) in enumerate(options) if en
        ] + list(extra or [])

    def available_actions(self) -> list[dict]:
        return list(self._actions)


def test_disabled_option_not_offered_by_engine_is_still_rendered() -> None:
    """C# shows a disabled Smith greyed; the engine just doesn't offer it."""
    g = RustGame()
    g.rm = _EngineRestRm([("HEAL", True), ("SMITH", False)])
    opts = g.get_state()["options"]
    assert [(o["option_id"], o["is_enabled"]) for o in opts] == [
        ("HEAL", True),
        ("SMITH", False),
    ]
    # choose_option 0 still resolves to Rest.
    assert g._option_actions[0] == {"action": "rest_option", "idx": 0}


def test_only_disabled_left_after_tent_renders_leave() -> None:
    g = RustGame()
    g.rm = _EngineRestRm([("SMITH", False)], extra=[{"action": "leave_shop"}])
    opts = g.get_state()["options"]
    assert opts[0]["option_id"] == "SMITH" and opts[0]["is_enabled"] is False
    assert g._option_actions[-1] == {"action": "leave_shop"}
