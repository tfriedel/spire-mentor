import json
import tempfile
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from simlab.specs import DECK_PRESETS, SweepSpec
from simlab.web.app import create_app
from simlab.web.jobs import JobManager


def _wait_done(client: TestClient, sweep_id: str, *, tries: int = 120) -> dict:
    """Poll a sweep's status until done (mirrors test_sweep_lifecycle)."""
    st: dict = {}
    for _ in range(tries):
        st = client.get(f"/api/sweeps/{sweep_id}").json()
        if st["status"] == "done":
            break
        time.sleep(0.5)
    assert st["status"] == "done"
    return st


def test_meta_endpoint(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    meta = client.get("/api/meta").json()
    assert len(meta["strategies"]) >= 5
    assert {"random", "mcts", "min_overblock"} <= set(meta["strategies"])
    assert len(meta["deck_presets"]) >= 5
    assert len(meta["encounters"]) == 12
    # card lists are present and non-empty
    assert all(len(cards) > 0 for cards in meta["deck_presets"].values())


def test_sweep_lifecycle(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)

    spec = {
        "name": "webtest",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random", "min_overblock"],
        "n_seeds": 4,
    }
    r = client.post("/api/sweeps", json=spec)
    assert r.status_code == 200
    sweep_id = r.json()["sweep_id"]

    for _ in range(120):
        st = client.get(f"/api/sweeps/{sweep_id}").json()
        if st["status"] == "done":
            break
        time.sleep(0.5)
    assert st["status"] == "done"
    assert st["done"] == st["total"] == 8

    lst = client.get("/api/sweeps").json()
    assert any(s["sweep_id"] == sweep_id for s in lst["sweeps"])
    # Every entry carries an mtime and the list is sorted newest-first.
    mtimes = [s["mtime"] for s in lst["sweeps"]]
    assert all(isinstance(m, (int, float)) for m in mtimes)
    assert mtimes == sorted(mtimes, reverse=True)

    res = client.get(f"/api/sweeps/{sweep_id}/results?baseline=random").json()
    bucket = res["buckets"]["MAWLER_NORMAL/0/ironclad_starter/min_overblock"]
    assert bucket["n"] == 4
    assert len(bucket["kde"]["x"]) == len(bucket["kde"]["y"]) > 0
    assert "delta_vs_baseline" in bucket


def test_list_sweeps_sorted_by_mtime_desc(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)

    # Two sweeps; the second is submitted later, so it must sort first.
    base = {
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 1,
    }
    id_a = client.post("/api/sweeps", json={**base, "name": "alpha"}).json()["sweep_id"]
    time.sleep(1.1)
    id_b = client.post("/api/sweeps", json={**base, "name": "beta"}).json()["sweep_id"]

    sweeps = client.get("/api/sweeps").json()["sweeps"]
    mtimes = [s["mtime"] for s in sweeps]
    assert mtimes == sorted(mtimes, reverse=True)
    ids = [s["sweep_id"] for s in sweeps]
    assert ids.index(id_b) < ids.index(id_a)


def test_submitted_sweep_is_listable_before_its_runner_writes_anything() -> None:
    """A sweep must be listable the instant `submit` returns, not once its
    worker thread happens to run.

    `list_sweeps` skips any directory without a `spec.json`, and that file used
    to be written only by the runner — from inside a detached daemon thread. So
    the sweep was invisible for however long the OS took to schedule that
    thread: usually microseconds, occasionally long enough to fail under load
    (the flake in `test_list_sweeps_sorted_by_mtime_desc`), and a real gap for a
    UI that POSTs then lists.

    Block the injected runner on an event so the worker provably has NOT run,
    which is what makes this a test of the guarantee rather than of the
    scheduler's mood: without the submit-time write it fails every time, not
    one run in fifty.
    """
    released = threading.Event()
    entered = threading.Event()

    def blocking_runner(sweep, *, out_dir, jobs, progress, cancel):
        entered.set()
        released.wait(30)
        return out_dir

    with tempfile.TemporaryDirectory() as tmp:
        mgr = JobManager(results_root=Path(tmp), runner=blocking_runner)
        spec = SweepSpec.from_json(
            {
                "encounters": ["MAWLER_NORMAL"],
                "ascensions": [0],
                "strategies": ["random"],
                "n_seeds": 1,
                "name": "immediate",
            }
        )
        sweep_id = None
        try:
            sweep_id = mgr.submit(spec)
            # The runner is parked inside its first line (or has not started at
            # all); either way it has written nothing.
            listed = mgr.list_sweeps()
            assert [e["sweep_id"] for e in listed] == [sweep_id]
            assert listed[0]["spec"]["encounters"] == ["MAWLER_NORMAL"]
        finally:
            released.set()
            entered.wait(5)
            # The worker still rewrites progress.json after the runner returns.
            # Join it before the temp dir goes: deleting files it holds open
            # fails on Windows (WinError 32/145) and races it on Linux.
            for t in threading.enumerate():
                if sweep_id is not None and t.name == f"sweep-{sweep_id}":
                    t.join(30)


def test_submitted_sweep_has_a_status_before_its_worker_starts(tmp_path, monkeypatch) -> None:
    """`GET /api/sweeps/{id}` must answer "running", not 404, the instant
    `submit` returns.

    progress.json used to be written only from inside the worker thread, so a
    client that POSTed and polled before the thread was scheduled got a 404
    (`test_results_unknown_baseline_422` failed on CI with ``KeyError:
    'status'``). The worker thread is held back entirely here — constructed but
    not started until the asserts are done — so only a submit-time write can
    make this pass, independent of the scheduler.
    """
    from simlab.web import jobs

    held: list[threading.Thread] = []
    real_thread = threading.Thread

    class HeldThread(real_thread):
        def start(self) -> None:
            held.append(self)

    monkeypatch.setattr(jobs.threading, "Thread", HeldThread)
    mgr = JobManager(results_root=tmp_path, runner=lambda sweep, **kw: kw["out_dir"])
    spec = SweepSpec.from_json(
        {
            "encounters": ["MAWLER_NORMAL"],
            "ascensions": [0],
            "strategies": ["random"],
            "n_seeds": 3,
            "name": "polled",
        }
    )
    sweep_id = mgr.submit(spec)
    try:
        assert held, "the worker thread must not have started"
        st = mgr.status(sweep_id)
        assert st["status"] == "running"
        assert (st["done"], st["total"]) == (0, 3)
    finally:
        for t in held:
            real_thread.start(t)
            t.join(30)


def test_submit_bad_spec_returns_422(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    spec = {
        "name": "x",
        "encounters": ["E"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 1,
        "deck_preset": "typo",
    }
    r = client.post("/api/sweeps", json=spec)
    assert r.status_code == 422


def test_meta_ships_cards_and_monsters(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    meta = client.get("/api/meta").json()
    assert "STRIKE_IRONCLAD" in meta["cards"]
    assert "Nibbit" in meta["monsters"]
    # existing keys preserved
    assert "deck_presets" in meta
    assert "strategies" in meta
    assert "encounters" in meta


def test_submit_rejects_bad_deck_with_suggestion(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    spec = {
        "name": "x",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 1,
        "decks": [["x", ["STRIKE_IRONCLD"]]],
    }
    r = client.post("/api/sweeps", json=spec)
    assert r.status_code == 422
    assert "STRIKE_IRONCLAD" in r.text


def test_submit_rejects_bad_monster(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    spec = {
        "name": "x",
        "encounters": ["monsters:Nibit"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 1,
    }
    r = client.post("/api/sweeps", json=spec)
    assert r.status_code == 422
    assert "Nibbit" in r.text


def test_results_baseline_deck(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    starter = list(DECK_PRESETS["ironclad_starter"])
    # "b" is the starter minus one Strike so it's a distinct, valid deck.
    b_deck = starter.copy()
    b_deck.remove("STRIKE_IRONCLAD")
    spec = {
        "name": "baselinedeck",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 2,
        "decks": [["a", starter], ["b", b_deck]],
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]
    _wait_done(client, sweep_id)

    res = client.get(f"/api/sweeps/{sweep_id}/results?baseline_deck=a").json()
    b_bucket = res["buckets"]["MAWLER_NORMAL/0/b/random"]
    assert "delta_vs_baseline" in b_bucket
    assert len(b_bucket["delta_vs_baseline"]["values"]) == 2
    # the baseline deck self-pairs to a zero delta (integer subtraction)
    a_bucket = res["buckets"]["MAWLER_NORMAL/0/a/random"]
    assert a_bucket["delta_vs_baseline"]["values"] == [0, 0]
    assert a_bucket["delta_vs_baseline"]["mean"] == 0


def test_baseline_params_mutually_exclusive(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    # The mutex guard runs before the rows lookup, so a nonexistent sweep id
    # still 422s — no need to run a sweep here.
    r = client.get("/api/sweeps/nope/results?baseline=random&baseline_deck=a")
    assert r.status_code == 422


def test_results_unknown_baseline_deck_422(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    starter = list(DECK_PRESETS["ironclad_starter"])
    b_deck = starter.copy()
    b_deck.remove("STRIKE_IRONCLAD")
    spec = {
        "name": "unknownbasedeck",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 2,
        "decks": [["a", starter], ["b", b_deck]],
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]
    _wait_done(client, sweep_id)

    r = client.get(f"/api/sweeps/{sweep_id}/results?baseline_deck=nope")
    assert r.status_code == 422


def test_results_unknown_baseline_422(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    starter = list(DECK_PRESETS["ironclad_starter"])
    spec = {
        "name": "unknownbase",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 2,
        "decks": [["a", starter]],
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]
    _wait_done(client, sweep_id)

    r = client.get(f"/api/sweeps/{sweep_id}/results?baseline=nope")
    assert r.status_code == 422


def test_stop_unknown_sweep_404(tmp_path) -> None:
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    # No such sweep / not running -> 404 (nothing to terminate).
    assert client.post("/api/sweeps/nope/stop").status_code == 404


def test_stop_running_sweep_marks_cancelled(tmp_path) -> None:
    """Stopping a running sweep hard-kills it and leaves a partial 'cancelled'.

    Uses a slow mcts sweep on 2 workers so it is reliably still running when we
    stop it; tolerant of a fast box racing to completion first.
    """
    app = create_app(results_root=tmp_path, jobs=2)
    client = TestClient(app)
    spec = {
        "name": "stoptest",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["mcts"],
        "n_seeds": 12,
        "mcts_sims": 200,
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]

    # Wait until the job registers (status leaves the pre-run void).
    for _ in range(50):
        st = client.get(f"/api/sweeps/{sweep_id}").json()
        if st["status"] in ("running", "cancelled", "done"):
            break
        time.sleep(0.05)

    r = client.post(f"/api/sweeps/{sweep_id}/stop")
    if r.status_code == 404:
        # Raced to completion before we could stop it — nothing to assert about
        # cancellation, but it must have finished cleanly.
        assert client.get(f"/api/sweeps/{sweep_id}").json()["status"] == "done"
        return
    assert r.status_code == 200
    assert r.json()["status"] == "stopping"

    # The runner thread flips the status to cancelled shortly after the kill.
    for _ in range(60):
        st = client.get(f"/api/sweeps/{sweep_id}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "cancelled"
    assert st["done"] < st["total"]  # stopped before every combat ran
    # Partial summary is kept so the sweep stays browsable in history.
    assert (tmp_path / sweep_id / "summary.json").exists()


def test_sweep_exceeding_runtime_cap_is_cancelled(tmp_path) -> None:
    """A sweep that runs past max_runtime_s is hard-stopped and marked cancelled.

    Uses a tiny cap + a slow mcts sweep so the watchdog reliably trips before the
    sweep finishes; the stopped_reason distinguishes it from a user Stop.
    """
    app = create_app(results_root=tmp_path, jobs=2, max_runtime_s=0.5)
    client = TestClient(app)
    spec = {
        "name": "captest",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["mcts"],
        "n_seeds": 24,
        "mcts_sims": 400,
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]

    st = {}
    for _ in range(120):
        st = client.get(f"/api/sweeps/{sweep_id}").json()
        if st["status"] not in ("running", "unknown"):
            break
        time.sleep(0.5)
    assert st["status"] == "cancelled"
    assert st["stopped_reason"] == "max_runtime"
    assert st["done"] < st["total"]  # capped before every combat ran


def test_resolve_max_runtime_explicit_and_env(monkeypatch) -> None:
    from simlab.web.jobs import DEFAULT_MAX_RUNTIME_S, JobManager

    # Explicit value wins over everything.
    monkeypatch.setenv("SIMLAB_MAX_RUNTIME_S", "42")
    assert JobManager._resolve_max_runtime(120) == 120
    # SIMLAB_MAX_RUNTIME_S overrides the default when unset (0).
    assert JobManager._resolve_max_runtime(0) == 42
    # Garbage env falls back to the default.
    monkeypatch.setenv("SIMLAB_MAX_RUNTIME_S", "nonsense")
    assert JobManager._resolve_max_runtime(0) == DEFAULT_MAX_RUNTIME_S
    monkeypatch.delenv("SIMLAB_MAX_RUNTIME_S", raising=False)
    assert JobManager._resolve_max_runtime(0) == DEFAULT_MAX_RUNTIME_S


def test_resolve_jobs_explicit_and_env(monkeypatch) -> None:
    from simlab.web.jobs import JobManager

    # Explicit jobs wins over everything.
    monkeypatch.setenv("SIMLAB_JOBS", "7")
    assert JobManager._resolve_jobs(3) == 3
    # SIMLAB_JOBS overrides the half-cores default when jobs == 0.
    assert JobManager._resolve_jobs(0) == 7
    # Garbage / non-positive env is ignored; fall back to the half-cores default.
    monkeypatch.setenv("SIMLAB_JOBS", "nonsense")
    assert JobManager._resolve_jobs(0) >= 1
    monkeypatch.setenv("SIMLAB_JOBS", "0")
    assert JobManager._resolve_jobs(0) >= 1
    monkeypatch.delenv("SIMLAB_JOBS", raising=False)
    assert JobManager._resolve_jobs(0) >= 1


def test_progress_reports_live_rate(tmp_path) -> None:
    """A running/finished sweep's progress.json carries jobs + combats/s."""
    app = create_app(results_root=tmp_path)
    client = TestClient(app)
    spec = {
        "name": "ratetest",
        "encounters": ["MAWLER_NORMAL"],
        "ascensions": [0],
        "strategies": ["random"],
        "n_seeds": 4,
    }
    sweep_id = client.post("/api/sweeps", json=spec).json()["sweep_id"]
    st = _wait_done(client, sweep_id)
    assert "jobs" in st
    assert st["elapsed_s"] >= 0
    assert st["combats_per_s"] >= 0


def test_initial_progress_total_includes_deck_axis(tmp_path) -> None:
    """The pre-runner progress.json write must count the deck axis.

    Stub out the runner so the JobManager's own initial _progress(0, total)
    is the only write — the real one would otherwise overwrite total with its
    (correct) chunk-derived count and mask a wrong initial value.

    The stub is INJECTED into this manager rather than monkeypatched onto
    ``simlab.web.jobs.run_sweep``: sweeps run on detached daemon threads that
    outlive the test, so a module-global stub also no-ops any sibling test's
    sweep that is still in flight, leaving it stuck at ``status: "running"``
    (an intermittent failure across the other sweep tests in this file).
    """
    from simlab.specs import SweepSpec
    from simlab.web.jobs import JobManager

    starter = list(DECK_PRESETS["ironclad_starter"])
    mgr = JobManager(results_root=tmp_path, jobs=1, runner=lambda *a, **kw: None)
    sweep_id = mgr.submit(
        SweepSpec.from_json(
            {
                "name": "totaltest",
                "encounters": ["MAWLER_NORMAL"],
                "ascensions": [0],
                "strategies": ["random"],
                "n_seeds": 3,
                "decks": [["a", starter], ["b", starter[1:]]],
            }
        )
    )
    for _ in range(120):
        st = mgr.status(sweep_id)
        if st["status"] == "done":
            break
        time.sleep(0.05)
    assert st["status"] == "done"
    # 1 encounter x 1 ascension x 2 decks x 1 strategy x 3 seeds
    assert st["total"] == 6


# ---------------------------------------------------------------------------
# Sandbox routes
# ---------------------------------------------------------------------------


def test_catalog_route(tmp_path) -> None:
    client = TestClient(create_app(results_root=tmp_path))
    cat = client.get("/api/simlab/catalog").json()
    assert {c["id"] for c in cat["characters"]} == {
        "Ironclad",
        "Silent",
        "Defect",
        "Regent",
        "Necrobinder",
    }
    assert all("moves" in m for m in cat["monsters"])
    assert any(e["id"] == "VANTOM_BOSS" for e in cat["encounters"])
    assert "mcts" in cat["strategies"]


def test_sandbox_deals_and_keeps_the_selected_game_version(tmp_path) -> None:
    """GH #877: ``game_version`` on ``/position/new`` deals the fight in that
    build, the position remembers it through build / moves, and an unknown
    build is a 400. A legacy-build position names no version at all; a request
    that names none deals the default build."""
    client = TestClient(create_app(results_root=tmp_path))
    body = {"character": "Ironclad", "encounter": "FUZZY_WURM_CRAWLER_WEAK", "seed": 4}
    base = client.post("/api/simlab/position/new", json={**body, "game_version": "v0.107.1"}).json()
    assert "game_version" not in base["position"] and "game_version" not in base["wire"]
    unnamed = client.post("/api/simlab/position/new", json=body).json()
    assert unnamed["position"]["game_version"] == "v0.111.0"

    r = client.post("/api/simlab/position/new", json={**body, "game_version": "v0.111.0"})
    assert r.status_code == 200, r.text
    pos = r.json()["position"]
    assert pos["game_version"] == "v0.111.0"
    assert r.json()["wire"]["game_version"] == "v0.111.0"
    built = client.post("/api/simlab/position/build", json={"position": pos}).json()
    assert built["position"]["game_version"] == "v0.111.0"
    assert built["wire"]["game_version"] == "v0.111.0"
    moves = client.post("/api/simlab/position/moves", json={"position": pos}).json()["moves"]
    assert any(m["label"] == "End turn" for m in moves)

    assert (
        client.post("/api/simlab/position/new", json={**body, "game_version": "v9"}).status_code
        == 400
    )
    assert client.get("/api/simlab/catalog?game_version=v9").status_code == 400
    cat = client.get("/api/simlab/catalog?game_version=v0.111.0").json()
    assert {c["id"] for c in cat["characters"]} >= {"Ironclad", "Necrobinder"}


def test_position_new_build_moves_and_sweep(tmp_path) -> None:
    client = TestClient(create_app(results_root=tmp_path))
    r = client.post(
        "/api/simlab/position/new",
        json={
            "character": "Ironclad",
            "encounter": "FUZZY_WURM_CRAWLER_WEAK",
            "ascension": 0,
            "seed": 4,
        },
    )
    assert r.status_code == 200, r.text
    pos = r.json()["position"]
    assert pos["monsters"][0]["monster"] == "FuzzyWurmCrawler"

    pos["player"]["hp"] = 30
    built = client.post("/api/simlab/position/build", json={"position": pos}).json()
    assert built["position"]["player"]["hp"] == 30
    assert built["notes"] == []

    bad = dict(pos)
    bad["monsters"] = [{**pos["monsters"][0], "intent": "NOPE"}]
    r = client.post("/api/simlab/position/build", json={"position": bad})
    assert r.status_code == 422
    assert "NOPE" in r.json()["detail"]

    moves = client.post("/api/simlab/position/moves", json={"position": pos}).json()["moves"]
    labels = [m["label"] for m in moves]
    assert "End turn" in labels and any(lbl.startswith("Play ") for lbl in labels)

    # One sweep cell per first move — what the sandbox's Run submits.
    spec = {
        "name": "moves",
        "encounters": [],
        "ascensions": [],
        "strategies": ["random"],
        "n_seeds": 2,
        "start_states": [[m["label"], json.dumps(m["wire"])] for m in moves if m["note"] is None][
            :2
        ],
        "positions": [[m["label"], json.dumps(m["position"])] for m in moves if m["note"] is None][
            :2
        ],
    }
    r = client.post("/api/sweeps", json=spec)
    assert r.status_code == 200, r.text
    sweep_id = r.json()["sweep_id"]
    st = _wait_done(client, sweep_id)
    # The total is known from the first tick (it used to read 0 for a
    # start-state sweep until the first chunk reported).
    assert st["total"] == 4
    lst = client.get("/api/sweeps").json()["sweeps"]
    mine = next(s for s in lst if s["sweep_id"] == sweep_id)
    assert len(mine["spec"]["positions"]) == 2
    buckets = client.get(f"/api/sweeps/{sweep_id}/results").json()["buckets"]
    assert len(buckets) == 2


def test_position_import_and_mcts(tmp_path) -> None:
    client = TestClient(create_app(results_root=tmp_path))
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "realgame_adapter"
        / "single_enemy_ironclad.json"
    )
    wire = json.loads(fixture.read_text(encoding="utf-8"))
    paste = json.dumps({**wire["state"], "type": "advice_request", "advice_for": "combat_action"})
    r = client.post("/api/simlab/position/import", json={"text": paste})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "advice_request"
    assert body["position"]["monsters"][0]["monster"] == "ShrinkerBeetle"
    r = client.post("/api/simlab/position/import", json={"text": "nonsense"})
    assert r.status_code == 422

    r = client.post("/api/simlab/position/mcts", json={"position": body["position"], "sims": 100})
    assert r.status_code == 200, r.text
    rows = r.json()["rows"]
    assert rows and rows[0]["is_best"]


def test_status_retries_a_read_refused_mid_replace(tmp_path: Path, monkeypatch) -> None:
    """Windows refuses to open progress.json while a writer's os.replace swaps it."""
    from simlab.web import jobs

    sweep = tmp_path / "s1"
    sweep.mkdir()
    (sweep / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    real_read_text = Path.read_text
    refusals = [2]

    def flaky_read_text(self: Path, *args, **kwargs) -> str:
        if self.name == "progress.json" and refusals[0]:
            refusals[0] -= 1
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    monkeypatch.setattr(jobs, "_REPLACE_BACKOFF_S", 0)
    assert JobManager(results_root=tmp_path).status("s1") == {"status": "running"}
    assert refusals == [0]
