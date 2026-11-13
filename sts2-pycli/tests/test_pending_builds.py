"""The pending-build table in ``tests/conftest.py`` (GH #1303)."""

from __future__ import annotations

import sts2_client


def test_pending_builds_are_supported_builds(pending_builds: dict[str, frozenset[str]]) -> None:
    # A pending entry for a build that is no longer supported is dead weight:
    # its markers would never fire.
    supported = set(sts2_client.SUPPORTED_GAME_VERSIONS)
    for name, builds in pending_builds.items():
        assert builds <= supported, (name, builds - supported)


def test_the_default_build_is_never_pending(pending_builds: dict[str, frozenset[str]]) -> None:
    # `just test` runs the default build only: a pending default would hide
    # every marked test from the everyday gate.
    for name, builds in pending_builds.items():
        assert sts2_client.DEFAULT_GAME_VERSION not in builds, name
