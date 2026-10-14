"""Pin monster→art coverage so combatants don't render blank in the workbench.

Every monster the engine can put on the board is shown by its display name
(``enemy["name"]``), which the workbench resolves to a sprite + Spine rig via
``AssetResolver`` and the ``_MONSTER_ALIASES`` table. That alias table is
hand-written and mirrors the web frontend's ``MONSTER_IMAGE_ALIASES``
(``sts2-wiki/site/src/utils.ts``); it silently drifted once — Crusher, Rocket,
and a dozen other monsters resolved to nothing and showed up as empty slots.

These tests fail loud the moment a modeled monster stops resolving to art, so
the gap can't reopen unnoticed. The two allow-lists below document the only
monsters that legitimately lack art (no extraction yet, or sprite-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sts2_hint.canonical_maps import MONSTER_TITLE_TO_ID
from sts2_hint.display_maps import monster_title
from workbench.assets import _MONSTER_KIND_RIGS, AssetResolver


def _wire_name(title: str) -> str:
    """The name the WIRE actually carries for a monster.

    Two of the 114 catalogue titles are not what a combatant is called on the
    board: the Test Subject's title is the loc TEMPLATE
    ``"Test Subject #C{Count}"`` (the count comes from cross-run save progress,
    which the workbench has no value for), and ``"Tough Egg"`` ships as
    ``"Hatchling"``. ``display_maps.monster_title`` is what the backend runs
    them through, so it is what the art lookup must be tested against.

    Testing the raw title instead made this guard BLIND for the Test Subject:
    ``"Test Subject #C{Count}"`` slugifies to ``test_subject_c_count``, which
    resolves to nothing, so the boss was parked in NO_ART as "image not yet
    rendered" while `monsters/test_subject.png` and its three-form Spine rig
    had been sitting in the tree the whole time.
    """
    return monster_title(MONSTER_TITLE_TO_ID[title])


# Art tree lives in the sibling sts2-wiki checkout; absent in lean CI.
_IMAGES = Path(__file__).resolve().parents[2].parent / "sts2-wiki" / "site" / "public" / "images"

# Monsters with NO extracted art of any kind. Either synthetic/test content the
# player never sees, or real bosses whose sprite/rig has not been rendered yet
# (mirrors SKIP_MONSTERS + KNOWN_MISSING_MONSTERS in
# ``sts2-wiki/scripts/check_images.py``). Remove an entry once its art lands.
NO_ART: frozenset[str] = frozenset(
    {
        "The Adversary Mk 1",  # test/unused content
        "The Adversary Mk 2",  # test/unused content
        "The Adversary Mk 3",  # test/unused content
    }
)

# Monsters that have a static sprite but no Spine rig (they animate via the
# sprite fallback). Supersets NO_ART, since "no art" implies "no rig".
NO_RIG: frozenset[str] = NO_ART | frozenset(
    {
        "Aeonglass",  # v0.107.1 boss: no Spine rig; renders the static hourglass placeholder
        # Decimillipede is NOT listed: besides its per-segment rigs (addressed
        # by MonsterKind, guarded by test_kind_addressed_rigs_resolve below),
        # the game ships scenes/creature_visuals/decimillipede.tscn, which
        # extract_rig.find_rigs publishes as the name-addressed
        # monster_rigs/decimillipede.
    }
)

pytestmark = pytest.mark.skipif(
    not (_IMAGES / "monsters").is_dir(),
    reason=f"extracted art tree not present at {_IMAGES}",
)


@pytest.fixture(scope="module")
def assets() -> AssetResolver:
    return AssetResolver(_IMAGES)


@pytest.mark.parametrize("title", sorted(MONSTER_TITLE_TO_ID))
def test_every_enemy_has_an_image(assets: AssetResolver, title: str) -> None:
    """Every modeled monster resolves to a sprite or rig (no blank combatant)."""
    if title in NO_ART:
        pytest.skip(f"{title!r} is a documented NO_ART monster")
    name = _wire_name(title)
    assert assets.monster(name) or assets.monster_rig(name), (
        f"monster {title!r} resolves to no sprite AND no rig — it will render "
        f"blank in the workbench. Add an alias to workbench.assets._MONSTER_ALIASES, "
        f"or to NO_ART here if its art genuinely does not exist."
    )


@pytest.mark.parametrize("title", sorted(MONSTER_TITLE_TO_ID))
def test_every_enemy_has_a_rig(assets: AssetResolver, title: str) -> None:
    """Every modeled monster resolves to a Spine rig (so it animates)."""
    if title in NO_RIG:
        pytest.skip(f"{title!r} is a documented sprite-only / NO_RIG monster")
    assert assets.monster_rig(_wire_name(title)), (
        f"monster {title!r} has no Spine rig — add an alias to "
        f"workbench.assets._MONSTER_ALIASES, or to NO_RIG here if it is sprite-only."
    )


@pytest.mark.parametrize("kind", sorted(_MONSTER_KIND_RIGS))
def test_kind_addressed_rigs_resolve(assets: AssetResolver, kind: str) -> None:
    """Rigs picked by MonsterKind (the Decimillipede segments, which share one
    display name) resolve to a real Spine rig.

    The name-based tests above can't see these — ``monster_rig`` needs the
    ``kind`` to disambiguate — so without this a missing or renamed segment rig
    would silently fall back to the shared sprite. That is exactly what a stale
    deployed manifest did: the segment rigs were absent from it, so the whole
    body rendered as three static sprites instead of the connected rig.
    """
    assert assets.monster_rig("", kind), (
        f"MonsterKind {kind!r} resolves to no Spine rig (expected "
        f"{_MONSTER_KIND_RIGS[kind]!r}) — it will fall back to the shared sprite. "
        f"Check workbench.assets._MONSTER_KIND_RIGS and the rig files."
    )


def test_allowlists_are_not_stale(assets: AssetResolver) -> None:
    """An allow-listed monster that gained art must be removed from the list."""
    gained_art = sorted(t for t in NO_ART if assets.monster(t) or assets.monster_rig(t))
    assert not gained_art, f"these NO_ART monsters now have art — remove them: {gained_art}"
    gained_rig = sorted(t for t in NO_RIG if assets.monster_rig(t))
    assert not gained_rig, f"these NO_RIG monsters now have a rig — remove them: {gained_rig}"
