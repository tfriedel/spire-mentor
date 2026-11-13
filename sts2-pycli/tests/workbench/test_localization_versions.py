"""Per-game-version display localization (GH #801).

The default build (v0.107.1) keeps reading ``localization_eng/`` and the wiki's
``data/v0.107.1/``; every other supported build reads its own sibling
``localization_eng_<ver>/`` and ``data/<ver>/`` and never borrows the default
build's files.
"""

import json
from pathlib import Path

import pytest
from workbench import localization

_REPO = Path(__file__).resolve().parents[2]
_WIKI_DATA = _REPO.parent / "sts2-wiki" / "data"


def test_engine_wiki_version_maps_every_supported_build():
    assert localization.DEFAULT_GAME_VERSION == "v0.107.1"
    assert localization.SUPPORTED_GAME_VERSIONS == ("v0.107.1", "v0.111.0", "v0.113.0")
    assert localization.engine_wiki_version() == localization._ENGINE_WIKI_VERSION
    for v in localization.SUPPORTED_GAME_VERSIONS:
        assert localization.engine_wiki_version(v) == v
    with pytest.raises(ValueError, match=r"v0\.107\.1, v0\.111\.0"):
        localization.engine_wiki_version("v0.99.1")


def test_loc_dir_for_keeps_the_default_path():
    assert localization.loc_dir_for() == _REPO / "localization_eng"
    assert localization.loc_dir_for("v0.107.1") == _REPO / "localization_eng"
    assert localization.loc_dir_for("v0.111.0") == _REPO / "localization_eng_v0.111.0"
    assert localization.loc_dir_for("v0.111.0", Path("/x")) == Path("/x/localization_eng_v0.111.0")


def test_non_default_build_never_borrows_another_builds_wiki_data(tmp_path: Path):
    """``data/v0.111.0/`` ships only the aggregates; the default build's per-item
    files must not leak into it, and a missing manifest is no manifest."""
    wiki = tmp_path / "data"
    (wiki / "v0.107.1" / "relics").mkdir(parents=True)
    (wiki / "v0.107.1" / "cards.json").write_text(json.dumps([{"loc_key": "BASH", "title": "Old"}]))
    (wiki / "v0.107.1" / "relics" / "x.json").write_text(
        json.dumps({"class_name": "OldRelic", "description": "old"})
    )
    assert localization._engine_manifest(wiki, "v0.111.0") is None
    assert localization._engine_versioned_subdir(wiki, "relics", "v0.111.0") is None
    assert localization.load_card_meta(wiki, version="v0.111.0") == {}
    assert localization.load_owned_localization(wiki, version="v0.111.0") == {}

    (wiki / "v0.111.0").mkdir()
    (wiki / "v0.111.0" / "cards.json").write_text(json.dumps([{"loc_key": "BASH", "title": "New"}]))
    (wiki / "v0.111.0" / "relics.json").write_text(
        json.dumps([{"class_name": "NewRelic", "description_resolved": "new"}])
    )
    loc = localization.load_card_localization(tmp_path / "none.json", wiki, version="v0.111.0")
    assert loc["BASH.title"] == "New"
    assert localization.load_owned_localization(wiki, version="v0.111.0") == {
        "NEW_RELIC.description_plain": "new"
    }
    # The default build is untouched by the sibling's presence.
    assert localization.load_card_localization(tmp_path / "none.json", wiki)["BASH.title"] == "Old"
    assert "OLD_RELIC.description_plain" in localization.load_owned_localization(wiki)


def test_vendored_v0111_localization_is_a_merge_over_the_default():
    loc_dir = localization.loc_dir_for("v0.111.0")
    if not loc_dir.is_dir():
        pytest.skip("localization_eng_v0.111.0/ not vendored")
    build = (loc_dir / "BUILD.txt").read_text(encoding="utf-8").splitlines()[0]
    assert build == "v0.111.0"
    cards = json.loads((loc_dir / "cards.json").read_text(encoding="utf-8"))
    base = json.loads((localization.loc_dir_for() / "cards.json").read_text(encoding="utf-8"))
    assert cards["CACOPHONY.title"] == "Cacophony"
    assert "CACOPHONY.title" not in base
    # Every key the default build has survives (content the engine implements).
    assert set(base) <= set(cards)


def test_build_workbench_loc_v0111_knows_its_new_content():
    if not (_WIKI_DATA / "v0.111.0" / "cards.json").is_file():
        pytest.skip("no sibling sts2-wiki data/v0.111.0")
    if not localization.loc_dir_for("v0.111.0").is_dir():
        pytest.skip("localization_eng_v0.111.0/ not vendored")
    loc, hp = localization.build_workbench_loc(_WIKI_DATA, version="v0.111.0")
    assert loc["CACOPHONY.title"] == "Cacophony"
    assert loc["CACOPHONY.card_type"] == "Power"
    assert loc["DOWSING_ROD.title"] == "Dowsing Rod"
    assert "DOWSING_ROD.description_plain" in loc
    assert loc["AMBERGRIS.title"] == "Ambergris"
    assert loc["GRAPPLE.title"] == "Grapple"
    assert loc["rest_site.OPTION_LIFT.name"] == "Train"
    assert hp
    default, _ = localization.build_workbench_loc(_WIKI_DATA)
    assert "CACOPHONY.title" not in default
    assert default["BASH.card_type"] == "Attack"


def test_build_workbench_loc_default_matches_the_rust_bundle_sources():
    """The default map is still built from ``localization_eng/`` + ``data/v0.107.1``
    — the same inputs ``bundle_workbench_data.py`` copies for the Rust twin."""
    if not (_WIKI_DATA / "v0.107.1" / "cards.json").is_file():
        pytest.skip("no sibling sts2-wiki data/v0.107.1")
    loc, _ = localization.build_workbench_loc(_WIKI_DATA)
    relics = json.loads((_REPO / "localization_eng" / "relics.json").read_text(encoding="utf-8"))
    for key, value in relics.items():
        assert loc[key] == value
