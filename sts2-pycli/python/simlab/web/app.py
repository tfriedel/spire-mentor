"""SimLab FastAPI app: sweep jobs + chart-ready aggregates + SPA hosting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sts2_rs
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from scipy.stats import gaussian_kde

from simlab.specs import (
    DECK_PRESETS,
    UI_ENCOUNTERS,
    SweepSpec,
    validate_deck,
    validate_encounter,
)
from simlab.telemetry import bucket_key, read_rows, summarize
from simlab.web.jobs import JobManager

DEFAULT_RESULTS = Path("results/simlab")
# Path arithmetic: python/simlab/web/app.py → parents[0]=web, [1]=simlab, [2]=python, [3]=repo root
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def _kde(values: list[int]) -> dict[str, list[float]]:
    """Compute a kernel density estimate for a list of damage values.

    Args:
        values: List of integer damage values.

    Returns:
        Dict with "x" (200 evenly-spaced points) and "y" (KDE densities),
        or a single-spike representation for degenerate (all-identical) input.
    """
    if len(set(values)) < 2:
        # degenerate: all identical — emit a single spike
        v = float(values[0]) if values else 0.0
        return {"x": [v], "y": [1.0]}
    arr = np.asarray(values, dtype=float)
    # fixed bandwidth — damage is discrete/integer; Scott's rule over-narrows on
    # quantized data.
    kde = gaussian_kde(arr, bw_method=0.3)
    x = np.linspace(0.0, max(arr.max() * 1.1, 1.0), 200)
    return {"x": x.round(2).tolist(), "y": kde(x).round(6).tolist()}


def create_app(
    results_root: Path = DEFAULT_RESULTS,
    jobs: int = 0,
    *,
    serve_spa: bool = True,
    max_runtime_s: float = 0,
) -> FastAPI:
    """Create the SimLab FastAPI application.

    Args:
        results_root: Directory where sweep results are stored and served from.
        jobs: Worker count for the process-pool runner; 0 means cpu_count.
        serve_spa: When True (standalone `just sim-ui`), mount the built SPA at
            "/". Set False when embedding this app inside another server (the
            workbench mounts it for its API routes only; the workbench-web dev
            server / its own SPA mount serves the pages).
        max_runtime_s: Per-sweep wall-clock cap; 0 falls back to
            SIMLAB_MAX_RUNTIME_S / the 10-minute default (see JobManager).

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="SimLab")
    mgr = JobManager(results_root=results_root, jobs=jobs, max_runtime_s=max_runtime_s)
    app.include_router(build_api_router(mgr))
    if serve_spa and WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="spa")
    return app


def build_api_router(mgr: JobManager) -> APIRouter:
    """Build the SimLab API routes (meta + sweep jobs + chart aggregates).

    Kept separate from :func:`create_app` so a host that already owns ``/`` — the
    workbench's single-container Cloud Run image mounts its SPA there as a
    catch-all ``StaticFiles`` — can ``include_router`` these explicit ``/api/...``
    routes *before* that mount. A nested ``Mount("/")`` (what the old embedding
    used) would instead be shadowed by the SPA mount, so every POST fell through
    to ``StaticFiles`` and 405'd.

    Args:
        mgr: The :class:`JobManager` the routes submit sweeps to and read from.

    Returns:
        An :class:`fastapi.APIRouter` carrying the SimLab API routes.
    """
    router = APIRouter()

    @router.get("/api/meta")
    def meta() -> dict:
        from simlab.agents import AGENTS

        return {
            "deck_presets": {name: list(cards) for name, cards in DECK_PRESETS.items()},
            "strategies": sorted(AGENTS),
            "encounters": list(UI_ENCOUNTERS),
            "cards": sorted(sts2_rs.list_registry_names("card")),
            "monsters": sorted(sts2_rs.list_registry_names("monster")),
        }

    # ── Sandbox: catalog + positions ─────────────────────────────────────
    # The SimLab position editor. Every route is stateless: the position
    # travels in the request, the engine validates it, the reply is what the
    # engine holds. See `simlab/position.py` for the schema.

    def _game_version_or_400(raw: object) -> str | None:
        """A request's ``game_version`` (GH #877): ``None`` for the default
        build, else a supported build; anything else is a 400 naming them."""
        from sts2_client import (
            DEFAULT_GAME_VERSION,
            LEGACY_GAME_VERSION,
            SUPPORTED_GAME_VERSIONS,
        )

        if raw is None:
            # A request that names no build gets the default one. (`None` out
            # of here means "the engine's own default", which is the legacy
            # build: a position FILE that names none.)
            raw = DEFAULT_GAME_VERSION
        if raw == LEGACY_GAME_VERSION:
            return None
        if raw not in SUPPORTED_GAME_VERSIONS:
            raise HTTPException(
                400,
                f"unsupported game_version {raw!r} "
                f"(supported: {', '.join(SUPPORTED_GAME_VERSIONS)})",
            )
        return str(raw)

    @router.get("/api/simlab/catalog")
    def simlab_catalog(game_version: str | None = None) -> dict:
        from simlab.catalog import catalog

        return catalog(_game_version_or_400(game_version))

    def _position_or_422(fn, *args, **kwargs):
        from simlab.position import PositionError

        try:
            return fn(*args, **kwargs)
        except PositionError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/simlab/position/build")
    def position_build(body: dict) -> dict:
        """Round-trip a position through the engine; 422 names what it refused."""
        from simlab.position import build_position

        built = _position_or_422(build_position, body.get("position") or {})
        return {"position": built.position, "wire": built.wire, "notes": built.notes}

    @router.post("/api/simlab/position/new")
    def position_new(body: dict) -> dict:
        """A fresh fight from turn 1: encounter (or lineup) × character × seed."""
        from simlab.position import new_position

        built = _position_or_422(
            new_position,
            character=str(body.get("character") or "Ironclad"),
            encounter=body.get("encounter") or None,
            monsters=body.get("monsters") or None,
            ascension=int(body.get("ascension") or 0),
            seed=int(body.get("seed") or 1),
            hp=body.get("hp"),
            max_hp=body.get("maxHp"),
            relics=body.get("relics"),
            potions=body.get("potions"),
            deck=body.get("deck"),
            # The workbench selector's build, like /api/live/start (GH #877).
            game_version=_game_version_or_400(body.get("game_version")),
        )
        return {"position": built.position, "wire": built.wire, "notes": built.notes}

    @router.post("/api/simlab/position/import")
    def position_import(body: dict) -> dict:
        """A pasted position / engine wire / SpireMentor Copy-state → position."""
        from simlab.position import position_from_paste

        pos, notes, source = _position_or_422(position_from_paste, str(body.get("text") or ""))
        return {"position": pos, "notes": notes, "source": source}

    @router.post("/api/simlab/position/moves")
    def position_moves(body: dict) -> dict:
        """Every legal first move and the position it leads to."""
        from simlab.catalog import name_of
        from simlab.position import build_position, first_moves, position_game_version

        pos = body.get("position") or {}
        built = _position_or_422(build_position, pos)
        version = _position_or_422(position_game_version, pos)
        moves = _position_or_422(
            first_moves, built.wire, name_of=lambda k, i: name_of(k, i, version)
        )
        return {"moves": moves, "wire": built.wire}

    @router.post("/api/simlab/position/mcts")
    def position_mcts(body: dict) -> dict:
        """MCTS verdict per first move (the same search the workbench hint runs)."""
        from simlab.catalog import name_of
        from simlab.position import build_position, mcts_eval, position_game_version

        pos = body.get("position") or {}
        built = _position_or_422(build_position, pos)
        version = _position_or_422(position_game_version, pos)
        sims = max(1, min(int(body.get("sims") or 1000), 20000))
        return _position_or_422(
            mcts_eval,
            built.wire,
            sims=sims,
            seed=int(body.get("seed") or 0),
            depth=int(body.get("depth") or 200),
            threads=int(body.get("threads") or 1),
            hidden_info=str(body.get("hidden_info") or "beliefdraw"),
            name_of=lambda k, i: name_of(k, i, version),
        )

    @router.post("/api/sweeps")
    def submit(spec: dict) -> dict:
        try:
            sweep = SweepSpec.from_json(spec)
            # Validate decks/encounters against the registries up front so typos
            # surface as a 422 (with suggestions) here rather than crashing the
            # background thread. A start-state sweep has neither axis — the
            # pasted board carries its own deck and lineup — so there is
            # nothing to look up; `cells()` below still validates the states by
            # parsing them.
            if not sweep.start_states:
                for name, cards in sweep.decks:
                    validate_deck(name, cards)
                for enc in sweep.encounters:
                    validate_encounter(enc)
            # Materialize cells eagerly so deck_preset/scenario validation (which
            # ScenarioSpec only runs on construction) surfaces as a 422 here
            # rather than crashing the background thread.
            sweep.cells()
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(422, f"bad sweep spec: {exc}") from exc
        return {"sweep_id": mgr.submit(sweep)}

    @router.get("/api/sweeps")
    def list_sweeps() -> dict:
        return {"sweeps": mgr.list_sweeps()}

    @router.post("/api/sweeps/{sweep_id}/stop")
    def stop(sweep_id: str) -> dict:
        # 404 when the sweep isn't running (already done/cancelled, unknown id,
        # or predates a restart) — there's nothing to terminate.
        if not mgr.stop(sweep_id):
            raise HTTPException(404, f"{sweep_id} not running")
        return {"status": "stopping"}

    @router.get("/api/sweeps/{sweep_id}")
    def status(sweep_id: str) -> dict:
        st = mgr.status(sweep_id)
        if st["status"] == "unknown":
            raise HTTPException(404, sweep_id)
        return st

    @router.get("/api/sweeps/{sweep_id}/results")
    def results(sweep_id: str, baseline: str = "", baseline_deck: str = "") -> dict:
        if baseline and baseline_deck:
            raise HTTPException(422, "baseline and baseline_deck are mutually exclusive")
        rows_path = Path(mgr.results_root) / sweep_id / "rows.jsonl"
        if not rows_path.exists():
            raise HTTPException(404, "no rows yet")
        all_rows = read_rows(rows_path)
        rows = [r for r in all_rows if "error" not in r]
        total_errors = sum(1 for r in all_rows if "error" in r)
        summary = summarize(rows)["buckets"]
        by_bucket: dict[str, list[dict]] = {}
        for r in rows:
            by_bucket.setdefault(bucket_key(r), []).append(r)
        # An unknown baseline would silently yield no deltas; reject it loudly.
        if baseline_deck:
            known = {k.rsplit("/", 3)[2] for k in by_bucket}
            if baseline_deck not in known:
                raise HTTPException(422, f"baseline_deck {baseline_deck!r} not found in results")
        if baseline:
            known = {k.rsplit("/", 3)[3] for k in by_bucket}
            if baseline not in known:
                raise HTTPException(422, f"baseline {baseline!r} not found in results")
        out: dict[str, dict] = {}
        for key, rs in by_bucket.items():
            dmg = [r["dmg_taken"] for r in rs]
            bucket = dict(summary[key])
            # raw values intentionally shipped — the SPA renders client-side
            # histograms and delta distributions; ~80KB per 10k-seed bucket,
            # acceptable for a localhost tool.
            bucket["dmg_values"] = dmg
            bucket["kde"] = _kde(dmg)
            if baseline or baseline_deck:
                # encounter names never contain "/" so rsplit("/", 3) is safe;
                # key is encounter/ascension/deck/strategy. A strategy baseline
                # swaps the last part; a deck baseline swaps the deck part.
                parts = key.rsplit("/", 3)
                if baseline:
                    base_key = f"{parts[0]}/{parts[1]}/{parts[2]}/{baseline}"
                else:
                    base_key = f"{parts[0]}/{parts[1]}/{baseline_deck}/{parts[3]}"
                base_rows = {r["seed"]: r["dmg_taken"] for r in by_bucket.get(base_key, [])}
                deltas = [
                    r["dmg_taken"] - base_rows[r["seed"]] for r in rs if r["seed"] in base_rows
                ]
                if deltas:
                    bucket["delta_vs_baseline"] = {
                        "mean": round(sum(deltas) / len(deltas), 3),
                        # raw values intentionally shipped — the SPA renders the
                        # delta distribution client-side (see dmg_values above).
                        "values": deltas,
                    }
            out[key] = bucket
        return {"buckets": out, "errors_total": total_errors}

    return router


app = create_app()
