"""The workbench's per-session game build (GH #801).

The wire contract shared with the Rust ``sts2-workbench`` port:

* ``/api/live/start``, ``/api/live/start-faithful`` and
  ``/api/live/import-state`` take an optional ``game_version``
  (``"v0.107.1"`` default, or ``"v0.111.0"``); anything else is a 400 that
  names the supported builds.
* ``/api/capabilities`` lists ``game_versions`` + ``default_game_version``.
* The live state, saves and the run-log meta carry ``game_version`` — written
  only off the default, read as the default when missing.
* A bug report's ``meta.engine_build`` is the session's build.
"""

from __future__ import annotations

import json

import pytest
from fake_game import FakeGame  # shared stub from tests/workbench/fake_game.py
from fastapi.testclient import TestClient
from workbench import saves as saves_mod
from workbench.live_runs import RunIdentity
from workbench.server import create_app
from workbench.timeline import TimelineStore

V107 = "v0.107.1"
V111 = "v0.111.0"


def _client(tmp_path, monkeypatch=None) -> TestClient:
    if monkeypatch is not None:
        from workbench import branching

        monkeypatch.setattr(branching, "Game", FakeGame)
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return TestClient(create_app(images_root=images, logs_dir=logs))


def _rust() -> None:
    pytest.importorskip("sts2_rs")


def _start(c: TestClient, version: str | None, seed: str = "G0LDEN0001") -> dict:
    body = {"character": "Ironclad", "seed": seed, "backend": "rust", "act1": "roll"}
    if version is not None:
        body["game_version"] = version
    r = c.post("/api/live/start", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _first_enabled_action(c: TestClient, step: dict) -> dict:
    opt = next(o for o in step["state"]["options"] if o.get("enabled", True))
    r = c.post("/api/live/action", json={"action": opt["action"], "args": opt["args"]})
    assert r.status_code == 200, r.text
    return r.json()["step"]


# ── capabilities / validation ───────────────────────────────────────────────


def test_capabilities_list_the_supported_builds(tmp_path):
    """Lists the builds this checkout can render: a public snapshot has no
    v0.111.0 loc dir, so only the default is offered there (#855)."""
    from workbench import localization

    caps = _client(tmp_path).get("/api/capabilities").json()
    assert caps["game_versions"] == localization.available_game_versions()
    assert caps["game_versions"][0] == V107
    # A new run starts on the default build when this checkout can show it.
    assert caps["default_game_version"] == (V111 if V111 in caps["game_versions"] else V107)
    assert "backends" in caps


def test_a_build_without_its_loc_dir_is_not_offered(tmp_path):
    """A public snapshot ships without ``localization_eng_v0.111.0/``; offering
    the build there would render every card as a raw loc key (#855)."""
    from workbench import localization

    (tmp_path / "localization_eng").mkdir()
    assert localization.available_game_versions(tmp_path) == [V107]
    (tmp_path / f"localization_eng_{V111}").mkdir()
    assert localization.available_game_versions(tmp_path) == [V107, V111]


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/api/live/start", {"character": "Ironclad", "seed": "S", "backend": "rust"}),
        ("/api/live/start-faithful", {"character": "Ironclad", "seed": "run_1"}),
        ("/api/live/import-state", {"payload": "{}"}),
    ],
)
@pytest.mark.parametrize("bad", ["v0.103.2", "0.111.0", "latest", ""])
def test_an_unsupported_build_is_a_400_naming_the_supported_ones(tmp_path, route, body, bad):
    c = _client(tmp_path)
    r = c.post(route, json={**body, "game_version": bad})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert V107 in detail and V111 in detail


def _other_build() -> str:
    """A supported build this process's C# oracle was NOT selected for."""
    import sts2_client

    return next(v for v in (V107, V111) if v != sts2_client.GAME_VERSION)


def test_another_build_than_the_process_oracle_on_the_csharp_engine_is_a_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post(
        "/api/live/start",
        json={
            "character": "Ironclad",
            "seed": "s",
            "backend": "csharp",
            "game_version": _other_build(),
        },
    )
    assert r.status_code == 400
    assert "rust" in r.json()["detail"]


@pytest.mark.parametrize("named", [True, False])
def test_the_csharp_engine_runs_the_process_oracle_build(tmp_path, monkeypatch, named):
    """Named or not: the C# backend has one build, the process's."""
    import sts2_client

    c = _client(tmp_path, monkeypatch)
    body = {"character": "Ironclad", "seed": "s", "backend": "csharp"}
    if named:
        body["game_version"] = sts2_client.GAME_VERSION
    r = c.post("/api/live/start", json=body)
    assert r.status_code == 200


# ── readers: a missing key is the default ───────────────────────────────────


def test_a_save_without_the_key_restores_on_the_default_build():
    legacy = {"character": "Ironclad", "seed": "S", "ascension": 0, "backend": "rust"}
    assert RunIdentity.from_save(legacy).game_version == V107
    assert RunIdentity.from_save({**legacy, "game_version": V111}).game_version == V111


def test_a_recording_without_the_key_is_the_default_build():
    ident = {
        "character": "Ironclad",
        "seed": "S",
        "ascension": 0,
        "backend": "rust",
        "act1": "roll",
        "run_mode": False,
    }
    assert RunIdentity.from_recording(ident).game_version == V107
    assert RunIdentity.from_recording({**ident, "game_version": V111}).game_version == V111


def test_write_save_omits_the_default_build(tmp_path):
    common = dict(
        character="Ironclad",
        seed="S",
        ascension=0,
        backend="rust",
        run_mode=False,
        progress={},
        line=[],
    )
    # Every save names its build, the legacy one included.
    p107 = saves_mod.write_save(tmp_path, name="a", game_version=V107, **common)
    assert json.loads(p107.read_text())["game_version"] == V107
    p111 = saves_mod.write_save(tmp_path, name="b", game_version=V111, **common)
    assert json.loads(p111.read_text())["game_version"] == V111


# ── a real v0.111.0 run (rust engine) ───────────────────────────────────────


def test_start_on_v0111_runs_that_build(tmp_path):
    _rust()
    c = _client(tmp_path)
    step = _start(c, V111)
    assert step["state"]["game_version"] == V111
    assert c.get("/api/live/state").json()["state"]["game_version"] == V111
    # The legacy build's STATE never carries the key (its golden bytes are
    # unchanged); a start that names no build gets the default one.
    legacy = _start(c, V107)
    assert "game_version" not in legacy["state"]
    assert _start(c, None)["state"]["game_version"] == V111
    # v0.111.0 hashes the text seed differently: the same seed is another run.
    assert step["state"]["bg_seed"] != legacy["state"]["bg_seed"]


def test_start_faithful_on_v0111(tmp_path):
    _rust()
    c = _client(tmp_path)
    r = c.post(
        "/api/live/start-faithful",
        json={
            "character": "Ironclad",
            "seed": "run_1",
            "sims": 8,
            "boss_sims": 16,
            "game_version": V111,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"]["game_version"] == V111


def test_save_and_restore_keep_the_build(tmp_path):
    _rust()
    c = _client(tmp_path)
    step = _start(c, V111)
    for _ in range(3):
        step = _first_enabled_action(c, step)
    before = c.get("/api/live/state").json()["state"]

    assert c.post("/api/live/save", json={"name": "v111 mid"}).status_code == 200
    payload = saves_mod.read_save(saves_mod.save_dir(tmp_path / "logs"), "v111 mid")
    assert payload is not None and payload["game_version"] == V111
    listed = c.get("/api/saves").json()["saves"][0]
    assert listed["game_version"] == V111

    # Replace the live run with a default-build one, then load the save back.
    _start(c, None)
    r = c.post("/api/saves/load", json={"name": "v111 mid"})
    assert r.status_code == 200, r.text
    restored = r.json()["state"]
    assert restored["game_version"] == V111
    assert restored["floor"] == before["floor"]
    assert restored["player"] == before["player"]


def test_seek_rebuilds_on_the_same_build(tmp_path):
    _rust()
    c = _client(tmp_path)
    step = _start(c, V111)
    root_seed = step["state"]["bg_seed"]
    for _ in range(2):
        step = _first_enabled_action(c, step)
    tl = c.get("/api/live/timeline").json()
    root_id = tl["nodes"][0]["id"]
    r = c.post("/api/live/seek", json={"node_id": root_id})
    assert r.status_code == 200, r.text
    assert r.json()["step"]["state"]["game_version"] == V111
    assert r.json()["step"]["state"]["bg_seed"] == root_seed


def test_the_run_log_meta_carries_the_build_and_resumes_on_it(tmp_path):
    _rust()
    c = _client(tmp_path)
    step = _start(c, V111)
    for _ in range(2):
        step = _first_enabled_action(c, step)
    log_path = c.post("/api/live/quit").json()["log_path"]
    store = TimelineStore.load_jsonl(log_path)
    ident = store.run_identity()
    assert ident is not None and ident["game_version"] == V111

    # "Play from here" on the recording rebuilds the run on v0.111.0.
    assert c.post("/api/load", json={"path": log_path}).status_code == 200
    r = c.post("/api/live/from_replay", json={"idx": 1})
    assert r.status_code == 200, r.text
    assert r.json()["state"]["game_version"] == V111


def test_a_legacy_build_run_log_names_its_build_too(tmp_path):
    _rust()
    c = _client(tmp_path)
    step = _start(c, V107)
    _first_enabled_action(c, step)
    log_path = c.post("/api/live/quit").json()["log_path"]
    ident = TimelineStore.load_jsonl(log_path).run_identity()
    assert ident is not None and ident["game_version"] == V107


def test_a_bug_report_names_the_sessions_build(tmp_path, monkeypatch):
    _rust()
    monkeypatch.delenv("STS2_BUGREPORT_BUCKET", raising=False)
    c = _client(tmp_path)
    _start(c, V111)
    r = c.post("/api/bugreport", json={"mode": "live"})
    assert r.status_code == 200, r.text
    dump = c.get(f"/api/bugreport/{r.json()['id']}").json()
    assert dump["meta"]["engine_build"] == V111

    _start(c, V107)
    r = c.post("/api/bugreport", json={"mode": "live"})
    dump = c.get(f"/api/bugreport/{r.json()['id']}").json()
    assert dump["meta"]["engine_build"] == V107


def test_imported_combat_is_rebuilt_in_the_sessions_build(tmp_path):
    """The session's build wins over a build the wire names, as in the Rust
    ``Game::start_imported`` — else one paste builds two different boards."""
    from workbench.branching import BranchingLiveSession

    sess = BranchingLiveSession.from_imported_combat(
        str(tmp_path), None, {"game_version": V107}, "Ironclad", game_version=V111
    )
    wire, character = sess._backend_cls.args
    assert wire["game_version"] == V111
    assert character == "Ironclad"
