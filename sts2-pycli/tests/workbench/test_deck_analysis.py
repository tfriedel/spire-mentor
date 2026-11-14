"""Tests for the deck-analysis axis grader and its endpoints."""

import json

import pytest
from fastapi.testclient import TestClient
from workbench import deck_analysis
from workbench.deck_analysis import analyze_deck
from workbench.server import create_app

IRONCLAD_STARTER = (
    [{"id": "STRIKE_IRONCLAD"}] * 5 + [{"id": "DEFEND_IRONCLAD"}] * 4 + [{"id": "BASH"}]
)

STRONG = [
    {"id": "STRIKE_IRONCLAD", "upgraded": True},
    {"id": "BASH", "upgraded": True},
    {"id": "INFLAME"},
    {"id": "DEMON_FORM"},
    {"id": "POMMEL_STRIKE", "upgraded": True},
    {"id": "ADRENALINE"},
    {"id": "WHIRLWIND"},
    {"id": "BACKFLIP"},
    {"id": "FEEL_NO_PAIN"},
    {"id": "TWIN_STRIKE"},
    {"id": "DEFEND_IRONCLAD"},
]


def test_static_tables_cover_each_other():
    """Every card in the stat DB has a curated tag entry, and vice versa."""
    db = deck_analysis._card_db()
    tags = deck_analysis._card_tags()
    assert set(db) == set(tags)
    roles = {"frontload", "scaling", "both", "none"}
    for t in tags.values():
        assert t["damage_role"] in roles
        assert t["block_role"] in roles
        assert isinstance(t["velocity"], list)


def test_empty_deck():
    assert analyze_deck([]) == {"empty": True}


def test_card_id_normalisation():
    # "CARD." prefix (engine) and "+" upgrade suffix both strip to the loc key.
    assert deck_analysis._card_id({"id": "CARD.STRIKE_IRONCLAD"}) == "STRIKE_IRONCLAD"
    assert deck_analysis._card_id("BASH+1") == "BASH"
    assert deck_analysis._card_id({"id": "BASH"}) == "BASH"


def test_starter_deck_is_low_velocity_and_warned():
    r = analyze_deck(IRONCLAD_STARTER)
    s = r["scores"]
    # Starter has no draw/energy → velocity collapses, dragging volume to ~0
    # (the geometric-mean "cube" point: a starved axis can't be hidden).
    assert s["velocity"] == pytest.approx(0.0)
    assert s["volume"] == pytest.approx(0.0)
    assert s["damage"] > 0 and s["block"] > 0
    texts = " ".join(w["text"] for w in r["warnings"])
    assert "velocity" in texts.lower()
    # All starter attacks are single-target — no AOE emerges.
    assert r["damage"]["aoe_attacks"] == []
    assert r["damage"]["vs_group"] == pytest.approx(r["damage"]["single_target"], abs=2.0)


def test_strong_deck_scores_on_all_axes():
    r = analyze_deck(STRONG)
    s = r["scores"]
    assert s["damage"] > 40
    assert s["velocity"] > 40
    assert s["volume"] > 0
    # Scaling and AOE are detected from the curated tags / stats.
    assert "Demon Form" in r["damage"]["scaling_cards"]
    assert "Whirlwind" in r["damage"]["aoe_attacks"]
    assert r["velocity"]["tags"].get("draw", 0) >= 1


def test_velocity_is_quantified_per_turn():
    # No-engine starter plays ~3 cards/turn (≈ base energy, 1-cost cards).
    starter = analyze_deck(IRONCLAD_STARTER)["velocity"]
    assert 2.0 <= starter["cards_per_turn"] <= 3.5
    # A deck with draw/energy engines plays more cards per turn.
    strong = analyze_deck(STRONG)["velocity"]
    assert strong["cards_per_turn"] > starter["cards_per_turn"]


def test_scaling_emerges_from_simulation():
    # A pure Strength-scaling power (no count bonus) must lift simulated damage
    # over many turns vs the same deck without it.
    base = [{"id": "STRIKE_IRONCLAD"}] * 6 + [{"id": "DEFEND_IRONCLAD"}] * 4
    scaled = [*base, {"id": "DEMON_FORM"}]
    assert (
        analyze_deck(scaled)["damage"]["single_target"]
        > analyze_deck(base)["damage"]["single_target"]
    )


def test_aoe_emerges_vs_group():
    # An AOE attack should out-damage a single-target one against a group, with
    # no aoe_share fudge — purely from hitting every enemy in the simulation.
    deck = [{"id": "WHIRLWIND"}] * 3 + [{"id": "DEFEND_IRONCLAD"}] * 4
    r = analyze_deck(deck)["damage"]
    assert r["vs_group"] > r["single_target"] * 1.5


def test_velocity_dilution_penalises_bloat():
    # Same engine, bigger deck → the Reddit `1/D` dilution lowers the velocity
    # score and reports a <1 dilution factor (lean decks stay at 1.0).
    lean = analyze_deck(STRONG)
    bloated = analyze_deck([*STRONG, *([{"id": "STRIKE_IRONCLAD"}] * 30)])
    assert lean["velocity"]["dilution"] == pytest.approx(1.0)
    assert bloated["velocity"]["dilution"] < 0.5
    assert bloated["scores"]["velocity"] < lean["scores"]["velocity"]


def test_bottleneck_reports_energy_vs_draw():
    # A 1-cost-heavy deck strands cards in hand → energy-limited; an all-0-cost
    # deck empties the hand with energy to spare → draw-limited.
    energy_bound = analyze_deck(IRONCLAD_STARTER)["velocity"]
    assert energy_bound["bottleneck"] == "energy"
    assert energy_bound["energy_limited_turns"] > energy_bound["draw_limited_turns"]
    shivs = analyze_deck([{"id": "SHIV"}] * 8 + [{"id": "DEFEND_SILENT"}] * 2)["velocity"]
    assert shivs["bottleneck"] == "draw"
    assert shivs["draw_limited_turns"] > shivs["energy_limited_turns"]


def test_bottleneck_surfaces_in_warnings():
    texts = " ".join(w["text"] for w in analyze_deck(IRONCLAD_STARTER)["warnings"]).lower()
    assert "energy-limited" in texts


def test_lenses_reweight_axes_by_fight():
    # The lenses are weighted blends of the three axis scores. A damage-leaning,
    # block-light deck scores higher under the hallway lens than the scaling one.
    r = analyze_deck(STRONG)
    lenses = r["lenses"]
    assert set(lenses) == {"hallway", "boss", "scaling"}
    s = r["scores"]
    # Hallway weights damage 0.6/block 0.2/velocity 0.2 — recompute and match.
    assert lenses["hallway"] == pytest.approx(
        0.6 * s["damage"] + 0.2 * s["block"] + 0.2 * s["velocity"], abs=0.1
    )
    # STRONG is damage-forward and block-light → hallway lens beats scaling lens.
    assert lenses["hallway"] > lenses["scaling"]


def test_deterministic():
    assert analyze_deck(STRONG, seed=7) == analyze_deck(STRONG, seed=7)


def test_upgrade_raises_damage():
    base = analyze_deck([{"id": "BASH"}] * 5, seed=1)
    up = analyze_deck([{"id": "BASH", "upgraded": True}] * 5, seed=1)
    assert up["damage"]["per_turn"] > base["damage"]["per_turn"]


# --- endpoint wiring --------------------------------------------------------


def _client(tmp_path, deck):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    log = tmp_path / "run.jsonl"
    state = {"decision": "card_reward", "round": 0, "player": {"name": "Ironclad", "deck": deck}}
    log.write_text(
        json.dumps({"step": 1, "ts": "2026-05-24T00:00:00", "type": "state", "data": state}) + "\n"
    )
    client = TestClient(create_app(images_root=images, logs_dir=tmp_path))
    client.post("/api/load", json={"path": str(log)})
    return client


def test_step_deck_analysis_endpoint(tmp_path):
    c = _client(tmp_path, IRONCLAD_STARTER)
    r = c.get("/api/step/0/deck-analysis")
    assert r.status_code == 200
    body = r.json()
    assert body["deck_size"] == len(IRONCLAD_STARTER)
    assert set(body["scores"]) == {"damage", "block", "velocity", "volume"}


def test_reward_preview_velocity_card_lifts_volume():
    # The starter deck has velocity 0 → volume 0. Adding a velocity engine
    # (Adrenaline) breaks the collapse, so its volume delta is strongly positive,
    # while a pure-damage card (Bludgeon) leaves volume pinned at 0.
    r = deck_analysis.reward_preview(
        IRONCLAD_STARTER,
        [{"index": 0, "id": "ADRENALINE"}, {"index": 1, "id": "BLUDGEON"}],
    )
    assert r["base"]["volume"] == pytest.approx(0.0)
    by_idx = {c["index"]: c for c in r["candidates"]}
    assert by_idx[0]["delta"]["velocity"] > 10
    assert by_idx[0]["delta"]["volume"] > 10
    assert by_idx[1]["delta"]["volume"] == pytest.approx(0.0)


def test_deck_signature_order_independent():
    a = deck_analysis.deck_signature([{"id": "BASH"}, {"id": "STRIKE_IRONCLAD"}])
    b = deck_analysis.deck_signature([{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}])
    assert a == b
    up = deck_analysis.deck_signature([{"id": "BASH", "upgraded": True}])
    assert up != deck_analysis.deck_signature([{"id": "BASH"}])


def test_reward_preview_endpoint(tmp_path):
    c = _client(tmp_path, IRONCLAD_STARTER)
    r = c.post(
        "/api/deck/reward-preview",
        json={
            "deck": [{"id": x["id"]} for x in IRONCLAD_STARTER],
            "candidates": [{"index": 0, "id": "ADRENALINE"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "base" in body and len(body["candidates"]) == 1
    assert body["candidates"][0]["delta"]["velocity"] > 10


def test_trend_endpoint_tracks_deck_changes(tmp_path):
    # Two steps with different decks → two trend points.
    images = tmp_path / "images"
    images.mkdir(parents=True)
    log = tmp_path / "run.jsonl"
    small = {"decision": "card_reward", "player": {"name": "Ironclad", "deck": IRONCLAD_STARTER}}
    big = {
        "decision": "card_reward",
        "player": {"name": "Ironclad", "deck": [*IRONCLAD_STARTER, {"id": "ADRENALINE"}]},
    }
    log.write_text(
        "\n".join(
            json.dumps({"step": i + 1, "ts": "2026-05-24T00:00:00", "type": "state", "data": d})
            for i, d in enumerate([small, small, big])  # dup is deduped
        )
        + "\n"
    )
    client = TestClient(create_app(images_root=images, logs_dir=tmp_path))
    client.post("/api/load", json={"path": str(log)})
    pts = client.get("/api/deck/trend").json()["points"]
    assert len(pts) == 2  # the duplicate deck collapses into one point
    assert pts[0]["deck_size"] == len(IRONCLAD_STARTER)
    assert pts[1]["deck_size"] == len(IRONCLAD_STARTER) + 1


def test_step_deck_analysis_no_run(tmp_path):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    c = TestClient(create_app(images_root=images, logs_dir=tmp_path))
    assert c.get("/api/step/0/deck-analysis").status_code == 400


# --- per game build (GH #877) -------------------------------------------------

_V111 = "v0.111.0"
_V111_DB = deck_analysis.data_dir_for(_V111) / "card_db.json"
_needs_v111 = pytest.mark.skipif(not _V111_DB.is_file(), reason="no v0.111.0 card db (public)")


def test_data_dir_for_keeps_the_default_build_at_todays_path():
    assert deck_analysis.data_dir_for(None) == deck_analysis.data_dir_for("v0.107.1")
    assert deck_analysis.data_dir_for(None).name == "data"
    assert deck_analysis.data_dir_for(_V111).name == _V111
    with pytest.raises(ValueError, match="unsupported game version"):
        deck_analysis.data_dir_for("v9.9.9")


@_needs_v111
def test_v0111_tables_cover_each_other():
    """The v0.111.0 tag overlay tags every card its db knows, and drops the
    card it removed."""
    db = deck_analysis.card_db_for(_V111)
    tags = deck_analysis.card_tags_for(_V111)
    assert set(db) == set(tags)
    assert "CACOPHONY" in db and "SCARE" not in db
    roles = {"frontload", "scaling", "both", "none"}
    for t in tags.values():
        assert t["damage_role"] in roles
        assert t["block_role"] in roles
        assert isinstance(t["velocity"], list)
    # The default build is untouched by the overlay.
    assert "CACOPHONY" not in deck_analysis._card_tags()
    assert "SCARE" in deck_analysis._card_tags()


@_needs_v111
def test_a_v0111_card_is_known_only_to_its_build():
    deck = [*IRONCLAD_STARTER, {"id": "OUTRAGE"}]
    v111 = analyze_deck(deck, game_version=_V111)
    base = analyze_deck(deck)
    assert v111["damage"]["attack_count"] == base["damage"]["attack_count"] + 1
    assert (
        analyze_deck(IRONCLAD_STARTER, game_version=_V111)["scores"]
        == analyze_deck(IRONCLAD_STARTER)["scores"]
    )


@_needs_v111
def test_shipped_v0111_card_db_matches_a_fresh_build(tmp_path):
    from workbench import build_card_db

    out = tmp_path / "card_db.json"
    build_card_db.main(["--version", _V111, "--out", str(out)])
    assert out.read_bytes() == _V111_DB.read_bytes()


def test_reward_preview_endpoint_takes_a_game_version(tmp_path):
    c = _client(tmp_path, IRONCLAD_STARTER)
    body = {
        "deck": [{"id": x["id"]} for x in IRONCLAD_STARTER],
        "candidates": [{"index": 0, "id": "OUTRAGE"}],
    }
    assert (
        c.post("/api/deck/reward-preview", json={**body, "game_version": "v9"}).status_code == 400
    )
    r = c.post("/api/deck/reward-preview", json={**body, "game_version": _V111})
    assert r.status_code == 200
    if _V111_DB.is_file():
        # Unknown to v0.107.1 (no stats, no damage), a real attack in v0.111.0.
        base = c.post("/api/deck/reward-preview", json={**body, "game_version": "v0.107.1"}).json()
        assert r.json()["candidates"][0]["delta"] != base["candidates"][0]["delta"]
