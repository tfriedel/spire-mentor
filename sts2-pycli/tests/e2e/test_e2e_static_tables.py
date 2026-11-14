"""e2e WP1: the combat context's static tables, per game build.

The v0.107.1 tables are pinned by content hash, taken at e0185edde before the
tables became version-keyed, so selecting by game build cannot have moved a
single v0.107.1 entry. The v0.111.0 tables are checked against the diff that
was audited against the v0.111.0 decompilation, and, where the decompilation
is present, re-derived from it.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import sts2_client
import sts2_rs
from ppo.combat_ctx import snapshot as SN
from ppo.combat_ctx import static_tables as ST
from ppo.combat_ctx import static_tables_gen as GEN
from ppo.combat_ctx import static_tables_v0111 as V111
from ppo.combat_ctx import tokens as T

V107, V111_ID = "v0.107.1", "v0.111.0"

# Engine counters to render every display rule over: small and cumulative
# counts, the packed `(hi << 8) | lo` counters, a sentinel, and an omitted one.
GRID = [*range(-3, 300), *((k << 8) | j for k in range(1, 4) for j in range(21)), 0xFFFF, None]

# sha256 of `json.dumps(x, sort_keys=True)`, computed at e0185edde.
PINS_V107 = {
    "card_type": "5c45b4e64d57f6fcbfa3d3457b9bcce1c1c320425249b7a3357da489a2cff240",
    "show_counter": "c42f5ec15ddb8a39b2c1855ea9902702024ae7d25ee74ac3a095a5f7b56a519f",
    # display re-pinned after main#1143 (PENDULUM shows its already-wrapped counter).
    "display": "91f5c05893676a693611da61c5dd5835737fd4437d681d234f0e190bb99a39a8",
    "consts": "acc3d6283108d90ef5559a7913192b9146025e6372d6a1f142abcbf1a1041623",
}

# The v0.107.1 -> v0.111.0 card diff, audited against both decompilations.
V111_ADDED = {
    "ABUNDANCE": "Skill",
    "BLADE_SYMPHONY": "Skill",
    "BLAZE": "Skill",
    "CACOPHONY_CARD": "Power",
    "CONCOCT_CARD": "Skill",
    "CONSTELLATION": "Skill",
    "DOWSING": "Quest",
    "FADE_CARD": "Skill",
    "HIBERNATE_CARD": "Skill",
    "IMITATION_LEARNING_CARD": "Skill",
    "MIDNIGHT": "Attack",
    "ONE_FOR_ALL_CARD": "Power",
    "OUTRAGE": "Attack",
    "PLOT": "Skill",
    "SIDESTEP": "Skill",
    "SOULBOUND_CARD": "Power",
    "THE_BALL": "Attack",
    "TUTOR": "Skill",
    "UNDERWORLD_CARD": "Skill",
}


def _sha(x: object) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()


def _display_grid(version: str | None) -> dict[str, list[int | None]]:
    rules = SN.RELIC_COUNTER_DISPLAY_BY_VERSION[version or ST.active_version()]
    return {r: [SN.display_counter(r, c, version) for c in GRID] for r in sorted(rules)}


def _has_decomp(version: str) -> bool:
    return os.path.isdir(GEN.decomp_dir(version))


@pytest.fixture
def select(monkeypatch: pytest.MonkeyPatch):
    """Make `version` the active game build for the rest of the test."""

    def _select(version: str) -> None:
        monkeypatch.setattr(sts2_client, "GAME_VERSION", version)

    return _select


@pytest.mark.parametrize("how", ["module", "explicit", "selected"])
def test_v0107_tables_are_unchanged(how: str, select) -> None:
    select(V107)
    if how == "module":
        cards, show, grid = ST.CARD_TYPE, ST.SHOW_COUNTER_RELICS, _display_grid(V107)
    elif how == "explicit":
        cards, show, grid = ST.card_types(V107), ST.show_counter_relics(V107), _display_grid(V107)
    else:
        cards, show, grid = ST.card_types(), ST.show_counter_relics(), _display_grid(None)
    assert (len(cards), len(show), len(grid)) == (577, 36, 36)
    assert _sha(cards) == PINS_V107["card_type"]
    assert _sha(sorted(show)) == PINS_V107["show_counter"]
    assert _sha(grid) == PINS_V107["display"]
    consts = [list(ST.CARD_TYPES), ST.CLASS_ALIASES, sorted(ST.SKIPPED_CLASSES)]
    assert _sha(consts) == PINS_V107["consts"]


def test_the_legacy_build_reads_the_v0107_tables(select) -> None:
    select(sts2_client.LEGACY_GAME_VERSION)
    assert sts2_client.LEGACY_GAME_VERSION == V107
    assert ST.card_types() is ST.CARD_TYPE
    assert ST.show_counter_relics() is ST.SHOW_COUNTER_RELICS


# Builds the advisor accepts whose e2e tables are not generated yet. The e2e
# brain does not serve them; GH #1303 tracks generating them.
PENDING_TABLES = {"v0.113.0"}


def test_every_supported_build_has_its_tables() -> None:
    builds = set(sts2_client.SUPPORTED_GAME_VERSIONS) - PENDING_TABLES
    assert set(ST.CARD_TYPE_BY_VERSION) == builds
    assert set(ST.SHOW_COUNTER_RELICS_BY_VERSION) == builds
    assert set(SN.RELIC_COUNTER_DISPLAY_BY_VERSION) == builds
    for v in builds:
        # A display rule for exactly the relics that build shows a counter on.
        assert set(SN.RELIC_COUNTER_DISPLAY_BY_VERSION[v]) == ST.show_counter_relics(v), v
        assert set(ST.card_types(v).values()) <= set(ST.CARD_TYPES), v


def test_pending_builds_are_still_pending() -> None:
    # Fails once a pending build gets its tables: then drop it from the set.
    assert set(sts2_client.SUPPORTED_GAME_VERSIONS) >= PENDING_TABLES
    assert not PENDING_TABLES & set(ST.CARD_TYPE_BY_VERSION)


def test_v0111_tables_are_the_audited_diff() -> None:
    old, new = ST.CARD_TYPE, V111.CARD_TYPE
    assert {k: new[k] for k in set(new) - set(old)} == V111_ADDED
    assert set(old) - set(new) == {"SCARE"}
    changed = {k: (old[k], new[k]) for k in set(old) & set(new) if old[k] != new[k]}
    assert changed == {"OUTBREAK_CARD": ("Power", "Skill")}
    assert len(new) == 595
    show = V111.SHOW_COUNTER_RELICS
    assert show == ST.SHOW_COUNTER_RELICS - {"DIAMOND_DIADEM"}
    # Every id is an engine registry name, so a wire id finds its row.
    assert set(new) <= set(sts2_rs.list_registry_names("card"))
    assert show <= set(sts2_rs.list_registry_names("relic"))


def test_v0111_display_rules_differ_only_where_the_decomp_does() -> None:
    old, new = _display_grid(V107), _display_grid(V111_ID)
    assert set(old) - set(new) == {"DIAMOND_DIADEM"}
    assert set(new) <= set(old)
    assert {r for r in new if new[r] != old[r]} == {"TOY_BOX"}
    # Toy Box: 3 combats per wax relic, 4 relics on v0.107.1, 5 on v0.111.0.
    toy = {c: SN.display_counter("TOY_BOX", c, V111_ID) for c in (0, 4, 11, 12, 13, 14, 15, 16)}
    assert toy == {0: 0, 4: 1, 11: 2, 12: 0, 13: 1, 14: 2, 15: None, 16: None}
    assert SN.display_counter("TOY_BOX", 12, V107) is None
    assert SN.display_counter("TOY_BOX", 11, V107) == 2


@pytest.mark.parametrize(
    ("version", "card", "want"),
    [
        (V107, "ABUNDANCE", "?"),
        (V111_ID, "ABUNDANCE", "Skill"),
        (V107, "OUTBREAK_CARD", "Power"),
        (V111_ID, "OUTBREAK_CARD", "Skill"),
        (V107, "SCARE", "Skill"),
        (V111_ID, "SCARE", "?"),
        (V107, "BASH", "Attack"),
        (V111_ID, "BASH", "Attack"),
    ],
)
def test_card_tokens_follow_the_active_build(version: str, card: str, want: str, select) -> None:
    select(version)
    assert T._card_type(card) == want


def test_relic_tokens_follow_the_active_build(select) -> None:
    for version, want in ((V107, {"id": "TOY_BOX"}), (V111_ID, {"id": "TOY_BOX", "counter": 1})):
        select(version)
        assert SN._relic({"id": "TOY_BOX", "counter": 13}) == want, version
    select(V111_ID)
    assert "DIAMOND_DIADEM" not in ST.show_counter_relics()
    assert SN.display_counter("DIAMOND_DIADEM", 3) is None


def test_an_explicit_build_overrides_the_active_one(select) -> None:
    select(V107)
    assert ST.card_types(V111_ID) is V111.CARD_TYPE
    assert ST.card_types("0.111.0") is V111.CARD_TYPE
    assert SN.display_counter("TOY_BOX", 13, "0.111.0") == 1
    select(V111_ID)
    assert ST.show_counter_relics(V107) is ST.SHOW_COUNTER_RELICS
    with pytest.raises(ValueError, match="unsupported game version"):
        ST.card_types("v0.103.2")


@pytest.mark.skipif(not _has_decomp(V107), reason="v0.107.1 decompilation absent")
def test_the_generator_reproduces_the_v0107_tables() -> None:
    assert GEN.derive(GEN.decomp_dir(V107)) == (ST.CARD_TYPE, ST.SHOW_COUNTER_RELICS)


@pytest.mark.skipif(not _has_decomp(V111_ID), reason="v0.111.0 decompilation absent")
def test_the_v0111_module_is_generated_from_its_decomp() -> None:
    cards, show = GEN.derive(GEN.decomp_dir(V111_ID))
    assert (cards, show) == (V111.CARD_TYPE, V111.SHOW_COUNTER_RELICS)
    module = GEN.module_path(V111_ID).read_text(encoding="utf-8")
    assert GEN.render(V111_ID, cards, show) == module
    assert GEN.main([V111_ID, "--check"]) == 0
