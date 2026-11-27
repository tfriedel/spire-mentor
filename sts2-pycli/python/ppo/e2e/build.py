"""The e2e lane's game build: which engine build every run, row and stamp is on (GH #1535).

A lane plays one build. `STS2_E2E_GAME_VERSION` names it before the process starts,
and spawned workers inherit it, as with `STS2_GAME_VERSION`. Unset, it is v0.111.0,
the build of the Ironclad campaign (#1097) and of the all-character lane (#1197), so
every existing lane, test and artifact reads what it always did. A hotfix spelling
names its base build (`v0.113.1` is the `v0.113.0` engine, which follows the v0.113.1
decomp; `sts2_client.normalize_game_version`).

Everything that used to hardcode "v0.111.0" reads `GAME_VERSION` here: the factory and
its `require_selection`, the schema and stamps (`contracts`), the evaluator, the decomp
reader and the invariants. A process must also select the same build for the Python
side (`STS2_GAME_VERSION`), which `factory.require_selection` checks.

`CHARACTERS` is the build's playable roster: The Fool exists from v0.113.0 on.
`previous(build)` is the build a lane on `build` warm-starts from (`ppo.e2e.migrate`):
v0.111.0 for v0.113.0, v0.113.0 for v0.114.0 (whose reworked Fool keeps its card ids).
"""

from __future__ import annotations

import os
from typing import Final

import sts2_client

BUILD_ENV: Final = "STS2_E2E_GAME_VERSION"
# The builds an e2e lane may play, oldest first.
BUILDS: Final = ("v0.111.0", "v0.113.0", "v0.114.0")
DEFAULT_BUILD: Final = BUILDS[0]

# Each build's engine card universe (sts2-wire `registry_names_for`), for the e2e
# content-ID table and the S_ctx card table: a later build's is frozen as that build
# left it, since `card_any` grows with every build and its sorted order is a trained
# net's rows (GH #1924). Rust twin: `sts2_e2e::sctx::reg_kind`.
CARD_KIND: Final[dict[str, str]] = {
    "v0.111.0": "card",
    "v0.113.0": "card_v0113",
    "v0.114.0": "card_v0114",
}

_BASE_CHARACTERS: Final = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
# Characters added by a build: they play on it and every later one.
_ADDED: Final[dict[str, tuple[str, ...]]] = {"v0.113.0": ("Fool",)}


def resolve(raw: str | None) -> str:
    """The lane build `raw` names (None or empty: the default)."""
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_BUILD
    v = sts2_client.normalize_game_version(raw)
    if v not in BUILDS:
        raise ValueError(f"{BUILD_ENV}={raw!r}: an e2e lane plays one of {BUILDS}, not {v}")
    return v


def previous(build: str) -> str:
    """The lane build before `build` in `BUILDS` (whose nets `migrate` carries to it)."""
    if build not in BUILDS or build == BUILDS[0]:
        raise ValueError(f"build {build!r}: no lane build before it in {BUILDS}")
    return BUILDS[BUILDS.index(build) - 1]


def characters(build: str) -> tuple[str, ...]:
    """The characters a lane on `build` may play, in `contracts.CHARACTERS` order."""
    if build not in BUILDS:
        raise ValueError(f"build {build!r} not in {BUILDS}")
    chars: list[str] = list(_BASE_CHARACTERS)
    for b in BUILDS[: BUILDS.index(build) + 1]:
        chars.extend(_ADDED.get(b, ()))
    return tuple(chars)


GAME_VERSION: Final = resolve(os.environ.get(BUILD_ENV))
CHARACTERS: Final = characters(GAME_VERSION)
