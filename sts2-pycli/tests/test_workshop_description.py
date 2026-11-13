"""The Steam Workshop descriptions fit Steam's 8000-byte cap.

`workshop.ps1` checks this too, but only at upload time -- after a ~20 minute
advisor and workbench build. Release 0.11.2 (#1317) merged a description of
8128 bytes and the upload failed there; this moves the check to the gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PACKAGING = Path(__file__).resolve().parents[2] / "spire-mentor-mod" / "packaging"

#: Steam rejects a longer description (k_EResultInvalidParam) at the very end
#: of an upload; `workshop.ps1` measures the UTF-8 bytes of the trimmed text.
_STEAM_CAP = 8000


@pytest.mark.parametrize("item", ["workshop", "workshop-beta"])
def test_description_fits_steam_cap(item: str) -> None:
    path = _PACKAGING / item / "description.bbcode"
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    size = len(path.read_text(encoding="utf-8").rstrip().encode("utf-8"))
    assert size <= _STEAM_CAP, (
        f"{path.name} ({item}) is {size} bytes; Steam caps it at {_STEAM_CAP}"
    )
