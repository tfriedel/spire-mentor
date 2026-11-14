"""A game build's art overlay (GH #801): ``images/<version>/…`` is the art that
build added or redrew. Its resolver tries the overlay first; the default
build's resolver and manifest never see it."""

from __future__ import annotations

import sys
from pathlib import Path

from workbench.assets import AssetResolver

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_asset_manifest import build_manifest, overlay_versions


def _tree(root: Path) -> Path:
    for rel in (
        "card_atlas/defect/beta/voltaic.png",
        "v0.111.0/card_atlas/defect/voltaic.png",
        "v0.111.0/relic_atlas/dowsing_rod.png",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"png")
    return root


def test_the_overlay_wins_only_for_its_build(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    base = AssetResolver(root)
    beta = AssetResolver(root, overlay="v0.111.0")
    assert base.card("Defect", "CARD.VOLTAIC", "Voltaic") == "card_atlas/defect/beta/voltaic.png"
    assert (
        beta.card("Defect", "CARD.VOLTAIC", "Voltaic") == "v0.111.0/card_atlas/defect/voltaic.png"
    )
    assert base.relic("Dowsing Rod") is None
    assert beta.relic("Dowsing Rod") == "v0.111.0/relic_atlas/dowsing_rod.png"


def test_manifest_mode_matches_filesystem_mode(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    manifest = set(build_manifest(root, overlays=["v0.111.0"]))
    beta = AssetResolver(root, manifest=manifest, overlay="v0.111.0")
    assert (
        beta.card("Defect", "CARD.VOLTAIC", "Voltaic") == "v0.111.0/card_atlas/defect/voltaic.png"
    )


def test_the_default_manifest_lists_no_overlay(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert overlay_versions(root) == ["v0.111.0"]
    assert build_manifest(root) == ["card_atlas/defect/beta/voltaic.png"]
    assert "v0.111.0/relic_atlas/dowsing_rod.png" in build_manifest(root, overlays=["v0.111.0"])


def test_a_rig_only_the_overlay_has_resolves_to_the_overlay(tmp_path: Path) -> None:
    """The Fool and the v0.113.0 monsters ship their rigs in the overlay only.

    Mutation caught: the rig lookups returning the bare ``character_rigs/…``
    path after finding the ``.skel`` under the overlay — a URL that 404s.
    """
    for rel in (
        "v0.113.0/character_rigs/fool/fool.skel",
        "v0.113.0/monster_rigs/wuppy/wuppy.skel",
        "monster_rigs/nibbit/nibbit.skel",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"skel")
    (tmp_path / "creature_scales.json").write_text('{"nibbit": {"scale": 0.2}}', encoding="utf-8")
    (tmp_path / "v0.113.0/creature_scales.json").write_text(
        '{"fool": {"scale": 0.34}, "nibbit": {"scale": 9.0}}', encoding="utf-8"
    )
    base = AssetResolver(tmp_path)
    new = AssetResolver(tmp_path, overlay="v0.113.0")
    assert base.character_rig("Fool") is None
    assert new.character_rig("Fool") == "v0.113.0/character_rigs/fool/fool"
    assert new.monster_rig("Wuppy") == "v0.113.0/monster_rigs/wuppy/wuppy"
    assert new.monster_rig("Nibbit") == "monster_rigs/nibbit/nibbit"
    # The overlay adds the new creatures' scales; a shared row is never replaced.
    assert new.character_rig_scale("Fool") == {"scale": 0.34}
    assert new.monster_rig_scale("Nibbit") == {"scale": 0.2}
    assert base.character_rig_scale("Fool") is None
