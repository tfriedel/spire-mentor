"""The "fool" fill event (GH #1535, SCHEMA.md section 13.6): v0.113.0's mechanics.

v0.113.0 adds the Fool's arrows and melody and the Wuppy pet. A "chars" row of a
v0.113.0 board carries them (`$.player.fool`, `$.player.wuppy`): ARROW and MELODY
tokens, a second ALLY token and the impaled-arrow count on ENEMY. The event takes
reserved slots, so the schema hash does not move and an undeclared net keeps the
inputs at zero weight. The build is per process, so the v0.113.0 checks run in a
subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ppo.e2e import contracts as C
from ppo.e2e import fillcheck as FC
from ppo.e2e import tokens as K
from ppo.e2e import vocab as V

PYCLI = Path(__file__).resolve().parents[2]


def test_the_event_takes_reserved_slots() -> None:
    assert C.FILL_FOOL in C.FILL_EVENTS
    assert (C.SEC["ARROW"], C.SEC["MELODY"]) == (C.SEC["RES_46"], C.SEC["RES_47"])
    assert [C.XC[f"X_FOOL{i}"] for i in (1, 2, 3)] == [C.XC[f"X_RES{i}"] for i in (1, 2, 3)]
    assert not C.FOOL_SECTION_IDS & C.RESERVED_SECTIONS
    slots = C.FILL_SLOTS[C.FILL_FOOL]
    assert set(slots["sections"]) == set(C.FOOL_SECTION_IDS)
    assert not set(slots["cat"]) & set(C.FILL_SLOTS[C.FILL_CHARS]["cat"])


def test_the_fool_tokens_from_a_player_dict() -> None:
    """The board of the Rust twin (`tokens::tests::fool_tokens_from_a_player_dict`)."""
    v = V.load()
    player = {
        "osty": {"alive": True, "hp": 7, "max_hp": 12, "block": 0},
        "wuppy": {"alive": True, "hp": 3, "max_hp": 6, "block": 0},
        "fool": {
            "arrows": [[1, 0], [2, 1], [3, 0]],
            "pending": True,
            "next": 0,
            "melody": [2, 0, 40, 3],
            "impaled": [[1, 1]],
        },
    }
    buf = K._Buf()
    K._chars_tokens(buf, player)
    cat, cont, sec, _flags = buf.arrays()
    assert sec.tolist() == [C.SEC["ALLY"]] * 2 + [C.SEC["ARROW"]] * 4 + [C.SEC["MELODY"]]
    assert cat[:2, C.XC["X_ID_MAIN"]].tolist() == [
        v.id("monster", "Osty"),
        v.id("monster", "Wuppy"),
    ]
    assert cont[1, C.SCTX_C["E_HPF"]] == pytest.approx(0.5)
    arrows = cat[2:6][:, [C.XC["X_POS"], C.XC["X_FOOL1"], C.XC["X_FOOL2"]]].tolist()
    # Slot 0 ready and next; slot 1 away with quietus; slot 2 destroyed; the pending slot.
    assert arrows == [[1, 1, 3], [2, 2, 2], [3, 3, 1], [4, 1, 1]]
    m = cat[6, [C.XC["X_FOOL1"], C.XC["X_FOOL2"], C.XC["X_FOOL3"], C.XC["X_POS"]]].tolist()
    assert m == [2, 0, C.FOOL_COUNT_CAP, 4]
    assert K._impaled(player) == {1: 1}
    lone = K._Buf()
    K._chars_tokens(lone, {"fool": {"arrows": [], "pending": True, "next": -1}})
    cat, _cont, sec, _flags = lone.arrays()
    assert sec.tolist() == [C.SEC["ARROW"], C.SEC["MELODY"]]
    assert cat[0, C.XC["X_FOOL2"]] == 3  # the pending slot is the next shot


def test_arrow_tokens_stop_at_the_cap() -> None:
    buf = K._Buf()
    K._fool_tokens(buf, {"arrows": [[1, 0]] * (C.MAX_ARROW_TOKENS + 3), "next": 0})
    _cat, _cont, sec, _flags = buf.arrays()
    assert int((sec == C.SEC["ARROW"]).sum()) == C.MAX_ARROW_TOKENS


_V0113 = r"""
import collections, json, sys
import numpy as np
from ppo.e2e import contracts as C, fillcheck as FC

rb, _terms = FC.sample_rows(3, 1_300_000_500)
sec, cat = rb["sec"].astype(np.int64), rb["cat"]
arrow = sec == C.SEC["ARROW"]
enemy = sec == C.SEC["ENEMY"]
base = ["WP-ENG-1", "content-prior", "chars"]
print(json.dumps({
    "combat": int(np.asarray(rb["is_combat"], bool).sum()),
    "arrow": int(arrow.sum()),
    "melody": int((sec == C.SEC["MELODY"]).sum()),
    "away": int((arrow & (cat[:, C.XC["X_FOOL1"]] == 2)).sum()),
    "impaled": int(cat[enemy, C.XC["X_FOOL1"]].sum()),
    "declared": list(FC.check(rb, [*base, "fool"]).definite()),
    "undeclared": list(FC.check(rb, base).definite()),
}))
"""


@pytest.mark.slow
@pytest.mark.parametrize("build", ["v0.113.0", "v0.114.0"])
def test_a_later_fool_run_carries_the_event(build: str) -> None:
    """Random Fool runs on v0.113.0 and on v0.114.0 (the reworked Fool): every combat
    row has its arrows and the melody, an away arrow is impaled in an enemy, and the
    fill check ties the event to the rows."""
    env = {
        **os.environ,
        "STS2_E2E_GAME_VERSION": build,
        "STS2_GAME_VERSION": build,
        "STS2_E2E_CHARS": "1",
        "STS2_E2E_CHARACTER": "Fool",
        "CUDA_VISIBLE_DEVICES": "",
        # pytest's `pythonpath` does not reach a child interpreter.
        "PYTHONPATH": str(PYCLI / "python"),
    }
    out = subprocess.run(
        [sys.executable, "-c", _V0113],
        cwd=PYCLI,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=900,
    )
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["combat"] >= 50
    assert got["melody"] == got["combat"]
    assert got["arrow"] >= got["combat"]
    assert got["away"] == got["impaled"]
    assert got["declared"] == []
    assert any("fool is not declared" in p for p in got["undeclared"])


def test_an_undeclared_wuppy_is_a_stray() -> None:
    """A Wuppy ALLY token is the "fool" event's even though "chars" owns ALLY."""
    v = V.load()
    if not v.has("monster", "Wuppy"):
        pytest.skip("this build's vocab has no Wuppy")
    assert FC._wuppy_id() == v.id("monster", "Wuppy")


def test_the_row_check_bounds_the_new_tokens() -> None:
    assert C.MAX_ALLY_TOKENS == 2
    assert C.COMBAT_EXT_SECTIONS[-2:] == (C.SEC["ARROW"], C.SEC["MELODY"])
    assert np.isin(list(C.FOOL_SECTION_IDS), list(C.COMBAT_EXT_SECTIONS)).all()
