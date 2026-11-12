"""Every game method the host overrides is pinned to a reviewed fingerprint (GH #1299).

The host skips some game methods and runs a copy of their rules, or nothing, in
their place. Such an override keeps the rules of the build it was written for, so
a game update that changes the original must not pass unnoticed. If a test here
fails, read ``python/host_override_fingerprints.py``: re-check each listed method
against the new decomp, fix or delete the override, then refresh the table.
"""

from __future__ import annotations

import host_override_fingerprints as hof
import pytest
import sts2_client

_HOW = (
    "A game method the host overrides changed, appeared or disappeared. Re-read each "
    "one against sts2-wiki/decompiled/{v}/ and the host's prefix, fix or delete the "
    "override, then run: uv run python python/host_override_fingerprints.py --write "
    "--version {v}\n"
)


@pytest.mark.oracle
def test_harmony_overrides_match_the_reviewed_table(game) -> None:
    version = sts2_client.GAME_VERSION
    current = hof.harmony_fingerprints(game)
    reviewed = hof.load_table()[version]["harmony"]
    lines = hof.diff(reviewed, {k: v["hash"] for k, v in current.items()})
    for i, line in enumerate(lines):
        key = line.split()[1]
        if key in current:
            lines[i] += f"  (prefix {', '.join(current[key]['prefixes'])})"
    assert not lines, _HOW.format(v=version) + "\n".join(lines)


@pytest.mark.parametrize("version", sts2_client.SUPPORTED_GAME_VERSIONS)
def test_ilpatcher_rewrites_match_the_reviewed_table(version: str) -> None:
    lines = hof.diff(hof.load_table()[version]["cecil"], hof.cecil_fingerprints(version))
    assert not lines, _HOW.format(v=version) + "\n".join(lines)


@pytest.mark.oracle
def test_a_fingerprint_follows_the_state_machine_and_private_helpers(game) -> None:
    # RollingBoulderPower.AfterPlayerTurnStart is async, so its logic lives in
    # the state machine's MoveNext; its damage is in the private DoDamage, and the
    # live branch hits through a lambda. All three must be in the fingerprint.
    game.start(character="Ironclad", seed="FINGERPRINT")
    rows = game.send({"cmd": "override_fingerprints", "dump": True})["overrides"]
    il = next(r for r in rows if "RollingBoulderPower::AfterPlayerTurnStart" in r["method"])["il"]
    assert "<AfterPlayerTurnStart>d__" in il
    assert "--- MegaCrit.Sts2.Core.Models.Powers.RollingBoulderPower::DoDamage(" in il
    assert "CreatureCmd::Damage(" in il
    assert "<AfterPlayerTurnStart>b__" in il


def test_the_table_sees_a_real_rule_change() -> None:
    # v0.107.1 Crusher.Background throws when no scene is loaded; v0.111.0 is
    # null-safe. Different code must give different fingerprints, and code that
    # did not change between the builds (RollingBoulderPower) must not.
    table = hof.load_table()
    old, new = table["v0.107.1"]["harmony"], table["v0.111.0"]["harmony"]
    background = "MegaCrit.Sts2.Core.Models.Monsters.Crusher::get_Background()"
    assert old[background] != new[background]
    boulder = next(k for k in old if "RollingBoulderPower::AfterPlayerTurnStart" in k)
    assert old[boulder] == new[boulder]


def test_diff_names_new_gone_and_changed() -> None:
    lines = hof.diff({"a": "1", "b": "2"}, {"b": "3", "c": "4"})
    assert [line.split()[0] for line in lines] == ["GONE", "CHANGED", "NEW"]
