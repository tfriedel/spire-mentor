"""C#-vs-Rust parity for Neow run-start offers.

The Neow blessing offers are generated from the run seed. Both engines must
produce the *same* three offers for the same (canonicalized) seed — the sts2_rs
port of `Neow.GenerateInitialOptions` is expected to match the real game
(exercised here via the C# headless).

This guards against two regressions we actually hit:
  * a stale C# headless binary silently disagreeing with current source, and
  * a future drift in either engine's Neow offer generation.

Skips if the Rust extension isn't built. Spawns the C# headless per seed, so
it's a touch slow; keep the seed list short.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("sts2_rs", reason="parity test needs the sts2_rs extension")

from sts2_client import Game
from workbench.backends.rust_game import RustGame
from workbench.options import normalize_options

# Mix of case/digits, including seeds that previously appeared to diverge
# (the stale-build artifact) and seeds that surface the Scroll Boxes curse.
SEEDS = ["BRAVO", "ECHO", "GOLF", "HOTEL", "seed2", "WORLD", "PARITY", "q9w8e7"]


def _norm(name: str) -> str:
    """Fold a display name / loc-key to a comparable token.

    C# emits resolved names ("Neow's Torment"); the Rust backend emits loc keys
    ("NEOWS_TORMENT.title"). Strip to upper-case alphanumerics so the two line up.
    """
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _csharp_offers(seed: str) -> list[str]:
    g = Game()
    try:
        state = g.start(character="Ironclad", seed=seed)
        opts = state.get("options") or state.get("event_options") or []
        return [_norm(o.get("name") or o.get("title") or "") for o in opts]
    finally:
        g.close()


def _rust_offers(seed: str) -> list[str]:
    g = RustGame()
    g.start(character="Ironclad", seed=seed, ascension=0)
    state = g.get_state()
    return [_norm((o.get("label") or "").replace(".title", "")) for o in normalize_options(state)]


@pytest.mark.pending_run_level
@pytest.mark.parametrize("seed", SEEDS)
def test_neow_offers_match_between_engines(seed: str) -> None:
    cs = _csharp_offers(seed)
    rs = _rust_offers(seed)
    assert cs, f"C# produced no Neow offers for seed {seed!r}"
    assert cs == rs, (
        f"Neow offers diverge for seed {seed!r}:\n  C#  = {cs}\n  Rust = {rs}\n"
        "If the C# side looks wrong, rebuild the headless (`just build-cs`) — a "
        "stale binary is the usual cause."
    )
