"""`ascension_climb` finds the game's saves with a save-sharing mod too.

A modded game writes `steam/<id>/modded/profileN/` of its own; a save-sharing
mod points it at the unmodded `steam/<id>/profileN/`. The climb looked in the
modded tree alone, so every run there came back `win=None` (2026-11-23).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def climb() -> Any:
    path = Path(__file__).resolve().parents[2] / "spire-mentor-mod" / "tools" / "ascension_climb.py"
    spec = importlib.util.spec_from_file_location("ascension_climb_saves_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its own module by name
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


@pytest.mark.parametrize("profile", ["modded/profile2", "profile2"])
def test_the_profile_the_log_names_is_found_in_either_tree(climb: Any, profile: str) -> None:
    log = (
        f"Wrote 266345 bytes to path=user://steam/76561198000000001/{profile}/saves/progress.save\n"
    )
    found = climb.profile_in_log(log)
    root = climb.APPDATA / "SlayTheSpire2" / "steam"
    assert found == root / "76561198000000001" / profile / "saves" / "progress.save"


def test_history_and_saves_are_searched_in_both_trees(
    climb: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(climb, "APPDATA", tmp_path)
    steam = tmp_path / "SlayTheSpire2" / "steam" / "76561198000000001"
    modded = _touch(steam / "modded" / "profile1" / "saves" / "history" / "1.run")
    shared = _touch(steam / "profile2" / "saves" / "history" / "2.run")
    _touch(steam / "profile2" / "saves" / "history" / "notes.txt")
    assert sorted(climb.history_files()) == sorted([modded, shared])

    save = _touch(steam / "profile2" / "saves" / "progress.save")
    assert climb.all_progress_saves() == [save]


@pytest.mark.parametrize(
    "line",
    [
        "[SimAdvisor] Takeover 'Strike' changed nothing on the board (1 in a row); re-requesting advice.",
        "[SimAdvisor] Takeover: the game refuses DEFEND (reason=EnergyCostTooHigh, preventer=none, energy=0).",
        "[SimAdvisor] Takeover: recommended target no longer hittable.",
        "[SimAdvisor] Takeover: no valid target for the recommended potion.",
    ],
)
def test_every_refusal_the_mod_logs_is_counted(climb: Any, line: str) -> None:
    assert len(climb.REFUSED_RE.findall(line)) == 1
