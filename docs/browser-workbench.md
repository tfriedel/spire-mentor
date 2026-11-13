# Browser-only workbench (M1 – M5)

The workbench SPA running with **no backend**: the Rust engine, compiled to
WebAssembly, answers every `/api/*` call from a Web Worker; runs persist in
IndexedDB; the page is an installable PWA that works offline. This is the
Android/offline route decided on 2026-09-02 — the Python FastAPI workbench
(`sts2-pycli/python/workbench`) and its Cloud Run deploy stay as-is and act as
the **oracle** the port is measured against until parity is complete.

Milestone 1 ships the **play loop, rewind (seek / branch tree) and named
saves** with full localization and art. Milestone 2 adds the **advisors**
(combat hint, search tree, choice ranker, "play best", regret coach / review,
the card and decision EV advisors with streamed progress) — see "Advisors"
below. Milestone 3 adds **run mode**: the PPO champion net runs in the
browser (tract-onnx compiled to wasm), the native run agent plays "play to
end" and autobattle, every non-combat decision is scored for the WPA review
and verify, and run-mode saves restore — see "Run mode" below. Milestone 4 adds the
**replay viewer** (a finished run leaves its JSONL log in IndexedDB; the
timeline, map, step view, recommend / search tree, regret and run reviews,
the MCTS Runs page and "play from here" all work over it) and the **deck
analysis** routes — see "Replay viewer" below. Milestone 5 adds **bug
reports** on the Pages host and **state import** — paste a run position
copied out of the live game and rebuild it in the browser (see those two
sections below), and the SimLab sandbox plus its distribution sweeps (GH
#707). The static build's `/api/capabilities` flags are `advisors, run_mode,
saves, replay, state_import, simlab, simlab_sweeps: true` and `bug_reports:
false` — the engine has no upload target for reports, so the SPA turns the
report button on itself when the Pages host answers the probe.

## Architecture

```
sts2-pycli/web (React SPA, VITE_WORKBENCH_MODE=static)
  api.ts ──► transport/apiFetch ──► Web Worker (engine.worker.ts)
                 │                      ├── wasm: sts2-workbench-wasm  ──► sts2-workbench (Rust port)
                 │                      │        Workbench::call(method, route, body) -> {status, body}
                 │                      └── IndexedDB `sts2-workbench` (stores: saves, live)
                 └── non-/api URLs (art, catalog): real network fetch
public/sw.js: shell cache-first + art runtime cache (offline)
```

| Layer | Where | What it is |
|---|---|---|
| `sts2-workbench` | `sts2-rs/crates/sts2-workbench` | Rust port of the Python workbench (`backends/rust_game.py`, `branching.py`, `server.py::_enrich`, `localization.py`, `assets.py`, `render.py`, `options.py`, `statediff.py`, `combatlog.py`, `saves.py`, …). JSON in / JSON out, no I/O, no threads, no clock. `Workbench::call` is shaped exactly like the FastAPI routes (same paths, bodies, 400/404/409 status codes). |
| `sts2-wire::emit` | `sts2-rs/crates/sts2-wire/src/emit/` | `serde_json::Value` twins of the PyO3 dict builders (combat state, run actions, full map, shop inventory, event options, …, and since M2 `emit::mcts`: the `mcts_recommend` / `mcts_search_tree` result dicts plus the one parser of their `config`). `sts2-py` now returns `pythonize(&value)` over them, so the Python workbench and the port read **the same** dicts by construction. |
| `sts2-vocab` | `sts2-rs/crates/sts2-vocab` | The PPO vocabulary, split out of `sts2-runai` so `sts2-wire` no longer pulls tract-onnx / rayon into the wasm graph. |
| `sts2-workbench-wasm` | `sts2-rs/crates/sts2-workbench-wasm` | The `wasm-bindgen` shell: `new(config_json)`, `call`, `build_save`, `restore_save`, `save_header`, `export_persist`, `import_persist(payload, progress_cb)`. |
| Embedded data | `sts2-rs/crates/sts2-workbench/data/` | `include_str!` bundle (localization_eng copies, trimmed wiki card/monster/owned tables, the art manifest, creature scales; ~1 MB raw). Regenerate with `just workbench-data`; `just check-workbench-data` gates drift. |
| Worker + transport | `web/src/worker/`, `web/src/transport/` | `apiFetch` replaces the 58 `fetch("/api/…")` calls in `api.ts`. In static mode it posts `{id, method, route, body}` to the worker and wraps the reply in a real `Response`, so nothing above `api.ts` (or in vitest) can tell the two modes apart. The worker owns the save routes and writes `export_persist()` to IndexedDB after every mutating call. |
| PWA | `web/public/sw.js`, `manifest.webmanifest`, `src/pwa.ts` | Hand-written service worker (shell + wasm cache-first, art cache with quota-aware eviction). Registered only in static mode. **Do not put `"orientation"` back in the manifest**: Chrome bakes it into the WebAPK's Android activity, and Android throws `IllegalStateException: Only fullscreen opaque activities can request orientation` on a `standalone` activity — the installed app then dies at launch (the icon does nothing) while a browser tab, which ignores the field, stays fine. Lock the orientation from JS (`screen.orientation.lock`) if it is ever wanted. |

### Engine prerequisites that landed with M1

- `wasm32-unknown-unknown` in `rust-toolchain.toml`; `[profile.wasm-release]`.
- `sts2-runai` features `ppo` (tract-onnx) and `sweeps` (rayon / `Instant`) —
  both default-on, both off for the wasm graph.
- `getrandom` 0.2 `js` feature on wasm targets (rand 0.8 in sts2-mcts / sts2-runai).
- Content dispatch registries (cards, relics, potions) are installed by the
  workbench's run constructor (`game::ensure_content_installed`), the same
  way the `sts2_rs` module import and the CLI do. Without it a wasm build
  links none of the relic / potion hooks — dead-code elimination drops them
  and every on-obtain effect silently no-ops. The golden replay caught this.

## Build, run, gates

### One-time toolchain

`wasm-build` shells out to the **`wasm-bindgen` CLI**, which cargo does not
install for you, and its version must equal the `wasm-bindgen` crate version in
`sts2-rs/Cargo.lock` — wasm-bindgen refuses an artifact produced by a different
version. The build (`sts2-pycli/scripts/workbench_static.py`) checks both
before it compiles and prints the install command. The wasm target itself
comes with the toolchain `sts2-rs/rust-toolchain.toml` pins.

```
cargo install wasm-bindgen-cli --locked \
    --version "$(grep -A1 'name = "wasm-bindgen"' sts2-rs/Cargo.lock | grep version | head -1 | cut -d'"' -f2)"
```

`wasm-opt` (binaryen) is optional — a system install is used if present, else
the `binaryen` npm package `_web-deps` puts in `web/node_modules`; without
either, the build ships the unoptimized wasm with a note (~30% larger).

**The engine ships gzipped.** The wasm32 target is built with `+simd128`
(`sts2-rs/.cargo/config.toml`; Chrome 91+, Firefox 89+, Safari 16.4+), which
makes the e2e student's net ~3.6x faster and the engine ~26.2 MB — over
Cloudflare Pages' 25 MiB (26,214,400 B) per-file cap. So `wasm-build` also
writes `web/src/wasm/pkg/sts2_workbench_wasm_bg.wasm.gz` (gzip -9, byte-stable,
~7.5 MB), the static BUILD points the worker's `?url` import at it
(`vite.config.ts`; dist then carries `wb-assets/sts2_workbench_wasm_bg.wasm-<hash>.gz`
and no `.wasm`), and the worker inflates it with `DecompressionStream`
(`web/src/worker/wasmLoad.ts`, still `instantiateStreaming`; a host that adds
`Content-Encoding: gzip` anyway is handled by sniffing the gzip magic; a
browser without `DecompressionStream` gets a clear error). `vite dev` serves
the plain wasm. `workbench_static.py build` ends with a **size gate**
(`check-dist`): any `web/dist` file over 25 MiB, or any `.wasm` in dist,
fails the build (and `just deploy-static-cf` re-runs it before uploading).

All from `sts2-pycli/`:

```
just wasm-check            # cargo check of the wasm graph (+ --no-default-features mcts/runai)
just wasm-build            # wasm-release build → wasm-bindgen --target web → web/src/wasm/pkg/
just workbench-static      # wasm-build + VITE_WORKBENCH_MODE=static npm run build → web/dist
just workbench-static-preview   # vite preview on :4173 with ../sts2-wiki art mounted at /assets
just e2e-static            # Playwright against the static build (web/e2e-static/)

just workbench-data / check-workbench-data       # embedded data bundle
just workbench-golden-dump / workbench-golden-check   # oracle fixtures (below)
```

Dev without a build: `cd web && npm run dev:static` (art served from
`../../sts2-wiki/site/public/images` by a Vite plugin). The static build ships
that same tree inside `web/dist/assets` as WebP under the `.png` names
(`scripts/art_to_dist.py`; ~70 MB instead of ~300), so the deployed site is
self-contained (no GCS bucket: its downloads are billed to the project, which
is on the free plan) and a full download fits the host's 360 MB/day quota.
`VITE_ASSETS_BASE=https://…` builds against a hosted art origin instead.

Offline: the service worker precaches the shell and caches art and PPO nets
as they are fetched; the topbar's **⬇ Download for offline** button
(`src/offline.ts`) fills both caches up front from `wb-art-list.json` +
`/ppo/index.json` (~110 MB once per device, resumable), after which any fight
renders with the network off.

Hosted: the owner's deployment (URL not published; `https://your-workbench.example.com` stands in for it below) on Cloudflare Pages (public
unlisted, free plan with no bandwidth cap; `just deploy-static-cf`, and
`just deploy-static-cf-preview [branch]` for a preview origin to test a phone
against). It started on Firebase Hosting, whose free plan caps transfer at
360 MB/day per project — three offline downloads; that site was disabled on
2026-09-03 and there is no fallback host. The build has an art-manifest gate
on the copied tree. Headers, cache rules and the one-time bootstrap:
`sts2-pycli/deploy/DEPLOY.md` § "Browser-only workbench on Cloudflare Pages".
The service worker precaches the whole shell (from `wb-precache.json`,
written by the Vite build) so the app works offline from the first visit; the
art cache survives redeploys.

### Gates

Rust (`sts2-rs/`): the usual six plus gate 7 in `sts2-rs/CLAUDE.md`
(`cargo check --target wasm32-unknown-unknown -p sts2-workbench-wasm`).
`cargo nextest run -p sts2-workbench` runs the ported unit tests, the
`play_smoke` scripted run and the **golden parity replay**.

Python (`sts2-pycli/`): `pytest tests/` — `tests/workbench/golden/` pins the
dump script; `sts2-py` parity now rides on the emit twins.

Web (`sts2-pycli/web/`): `npm run lint && npm test && npm run build`, then
`just e2e-static`.

## Parity against the Python oracle

`sts2-pycli/scripts/dump_workbench_golden.py` drives the **Python** workbench
(`create_app(...)` under FastAPI's TestClient, `assets_base=https://assets.test`,
a fixed `saved_at`) through the scenario matrix in
`sts2-pycli/tests/workbench/golden/scenarios/*.json` — 5 characters × 2 seeds
× {A0, A10} × {overgrowth, underdocks} plus three long `act1=roll` runs to
`game_over`, with `state` / `map` / `timeline` / `seek` / `save` / `load`
probes — and records every request plus the SHA-256 of its normalized
response (full bodies for the first occurrence of each decision kind and every
non-200) under `sts2-rs/crates/sts2-workbench/tests/golden/`. M3 adds three
`runmode_*` scenarios started through `/api/live/start-faithful` at a tiny
combat budget, with `run_mode` / `run_mode_step` / `autobattle_*` /
`ppo_policy` / `wpa_review` / `wpa_verify` probes; two of them are flagged
`"ppo": true`, which lifts the dump's champion stub so the Python side serves
the committed net through its Rust ONNX backend — the same tract inference
over the same bytes the browser runs, so the panel probabilities, the node
values and the WPA review hash-match. Every other scenario runs with the
loader stubbed to answer `no PPO net loaded for <character>` — the browser
port's wording before its host has fetched a champion — so both hosts
answer the same "no net" panel.

`crates/sts2-workbench/tests/golden_parity.rs` replays the same requests
through `Workbench::call` (save routes emulated in memory) and compares
hashes. Normalization on both sides: sorted keys, `log_path` / `path`
dropped, floats rounded like Python's `round(x, 4)` (exact, ties-to-even —
`(x*1e4).round()/1e4` is *not* that), compact JSON. **43 scenarios, 5917
requests, 0 divergences** at M1; **6709 requests** at M2 with the advisor
probes (`hint` / `search_tree` / `choice_ranker` at fixed steps — a 409 off
combat, a seeded 48-sim search in a fight — plus `auto_action`,
`last_move_regret` — since removed, GH #869 — and `regret_review`); **46 scenarios, 7218 requests** at M3;
**8454 requests** at M4, every scenario ending with a quit and the replay
block (list, load, timeline, map, steps, deck grades, the reviews, the
searches, the MCTS Runs page, play-from-here) plus live deck-analysis and
reward-preview probes. The run-mode routes hash-match because the run agent
is the same Rust code on both hosts; `autobattle/verdict` is structural in
the scenarios without a net, where the brain's pick is each host's own
heuristic; `/api/runs` and `/api/runs/stat` (host paths, mtimes) and the
replay's `recommend` / `search_tree` / `regret-review` (exact-state
re-searches, see "Replay viewer") are structural too.

Three of those routes are compared by status + key shape instead of hash
(`STRUCTURAL_ROUTES` in the test), each for a reason the port will not
undo: the regret coach and review re-score the parent node's **native**
combat snapshot, where the Python session — which no longer holds that
state — re-scores a lossy wire dict and answers 409 for moves it cannot
resolve (a potion's `potion_index` it never read, a card the round-trip
dropped); "play best" off combat asks the Rust `HeuristicPolicy`, where
Python asks its own `run_policy` heuristic (or the PPO brain). The test
allows `python 409 / rust 200` on the coach and never the reverse.

Oracle quirks fixed in Python on the way (both re-dumped):

- `/api/live/state` did not pass `hp_ranges` to `_live_step`, so a
  rehydrated combat lost its `size_jitter`.
- `RustGame._serialize` guarded `bg_seed` with `callable(run_seed)`, but
  `run_seed` is a PyO3 getter (an int), so the room background seed was never
  emitted for live runs.
- (M3) `BranchingLiveSession._node_value` never passed `character=`, so every
  run's node values — the WPA review's whole data — came from the DEFAULT
  character's champion (Ironclad's net valued Silent's decisions). Caught
  by the Silent `ppo: true` scenario: the Rust port's root value differed.
- (M3) `ppo_policy._get_rust_agent` built `PpoRunAgent(model_path=…)`, which
  falls back to the process-wide `STS2_PPO_FEATURES` for the encoder schema
  — empty on the server — so Ironclad's `reward_upgrade` champion was served
  at the default schema (GH #545's exact failure). The binding now takes
  `features=` and the loader passes the manifest's spec.

Rules for the parity window: any Python workbench change that touches a
response re-dumps the goldens (`just workbench-golden-dump`), and any new
normalization ignore needs a comment naming a GitHub issue.

## Advisors (M2)

`crates/sts2-workbench/src/advisors.rs` ports `reasoning.py`,
`choice_ranker.py`, `regret_review.py`, `card_advisor.py` and
`decision_advisor.py` over the live session; `api.rs` mounts them on the
Python routes (`/api/live/hint`, `search_tree`, `choice-ranker`,
`auto-action`, `last-turn-grade`, `regret-review`, `card-advisor[/stream]`,
`decision-advisor[/stream]`). Everything is single-threaded — a config's
`threads` is honoured as 1 — and the search-backed routes are deterministic
for a config, which is what makes them golden-testable.

**Wasm-only: the choice coach** (`src/session_choice.rs`). `POST
/api/live/last-choice-grade` and `POST /api/live/last-choice-check/stream`
have no Python twin and no golden — the browser build is the only target.
`live_action` snapshots the PPO policy and a `RunManager` clone before each
human run-level choice; the grade is policy agreement (flag only when the
top pick is ≥ 40% and ≥ 2× the pick's probability, else toss-up), and the
check plays paired rollouts of your pick vs the AI's top pick from that
snapshot. The snapshot never changes a ported response, so the goldens stay
byte-identical.

- **Search-backed**: one `sts2_wire::emit::mcts::recommend_value` /
  `search_tree_value` over the engine `CombatState`. The hint's rows, the
  ranker's per-option join (`choice:{opt}` / `choice:skip`) and "play best"
  (the ranker's starred option in an in-combat choice, the tier-1
  `HeuristicPolicy` elsewhere, verbs looked up in the rendered option list)
  are the Python code line for line. `Session` nodes now keep the engine's
  combat state (`Node::combat`, in memory only, rebuilt by replay on
  restore) so the regret coach scores the real parent position.
- **Rollout-backed**: the card / decision advisors fork the `RunManager`
  per option, take it, and play the rest of the act out `rollouts` times
  (cheap depth-10 MCTS at `sims` in combat — `sims: 0` is a random move —
  the heuristic run policy off combat), paired by seed across options.
  Progress goes through a `FnMut(&Value)` sink → `Workbench::call_stream`
  → the wasm `call_stream(…, on_frame)` → `{id, frame}` worker messages →
  a `ReadableStream` body in `apiFetch` with `application/x-ndjson`, so
  `api.ts`'s line readers and the progress bars are untouched. A
  non-applicable stream is an `{"type": "error"}` line on a 200, like the
  Python server (`advice_stream.py`).

Cost on the tablet (Xiaomi Pad Mini, September 2026, clairvoyant search,
before the October engine speed-up; 0.023 ms/sim): the hint at the
default 2000 sims is ~50 ms; the EV advisors at their defaults (12 rollouts
× 128 sims × the rest of the act) run for tens of seconds — the SPA's
budget selectors apply as before.

## Run mode (M3)

`crates/sts2-workbench/src/runmode.rs`, `session_runmode.rs` and `ppo.rs`
port `BranchingLiveSession.start_run_mode` / `run_mode_step` /
`play_to_end`, `autobattle.py` + the autobattle loop and verdict,
`ppo_policy.py` + `ppo_policy_panel`, and `wpa_review.py` + `wpa_verify`;
`api.rs` mounts `/api/live/start-faithful`, `run-mode`, `run-mode-step`,
`faithful-step`, `play-to-end`, `autobattle/{advance,step,verdict}`,
`ppo-policy`, `wpa-review` and `wpa-verify`.

- **The champion net.** `sts2-runai`'s `ppo` feature (tract-onnx) is on in
  the wasm graph (`PpoNet::from_bytes` / `PpoPolicy::from_bytes_with`; the
  only wasm-specific fix was getrandom 0.4's `wasm_js` opt-in, pulled in by
  tract's `rand`). The worker fetches `/ppo/<file>` — the exact filename
  `champions.toml`'s generated table names for the run's character,
  `Workbench::ppo_champion_file` — before any route that starts or restores
  a run and hands the bytes to `Workbench::load_ppo`, once per character. A
  failed fetch (offline on first use) is recorded (`set_ppo_error`) and the
  run goes on exactly as Python does without a checkpoint: the panel says
  `no PPO net loaded for <character>: …`, node values stay `null`, the `ppo`
  brain falls back to the heuristic. The static build copies the five
  champion files into `dist/ppo/` (`vite.config.ts`, read off
  `champions.toml`; the dev/preview server mounts `../data/ppo`); they are
  NOT precached (8.5 MB each), the service worker keeps them cache-first
  (`wb-ppo-v1`, across deploys — a promotion changes the filename) and
  `_headers` serves `/ppo/*` immutable.
- **Size.** With the net reachable (parser + optimizer + ops) tract adds
  ~17 MB raw to the engine (5.5 → 22.9 MB raw, 1.65 → 6.0 MB gzip);
  `wasm-opt -O3` brings that to 19.5 MB raw / 5.4 MB gzip. `just wasm-build` runs a system `wasm-opt`
  if there is one, else the `binaryen` npm package in `web/` (a dev
  dependency, so `_web-deps` installs it). A hand-written forward pass for
  the 3-block transformer is the M5-era option if the compile cost matters
  on the tablet.
- **The run agent** is `sts2_rs.HeuristicRunAgent` — the deployed
  `MctsCombatPolicy<HeuristicPolicy>` stepped one decision at a time
  (`RunAgent`), single-threaded, carrying its MCTS RNG and decision counter
  across pauses. For Ironclad it carries the compiled-in run-agent layer
  (GH #830): the forms + combo knobs in every fight and the built-in BC move
  prior in elite and boss fights. The wasm build reads no environment, so the
  browser always plays that default. The hint and the search-tree inspector
  (`RecommendParams`) do not take the layer; they search in the base
  BeliefDraw dialect for every character. `start-faithful` builds the run
  the PPO env's way (`Game::start_faithful`: integer seed via `hash_seed`,
  Neow start, the deck-mutation flags, Overgrowth); a hand-played run gets its agent lazily
  on the first `run-mode-step`. The step record, the `search` rows, the
  `debug` summary and the timeline labels (`to_workbench_verb` port) are
  the Python shapes.
- **Node values.** `Session::add_node` scores every non-combat, non-game-over
  node with the loaded net (the critic's V(s)); a save / IndexedDB restore
  replays the recorded values instead (`node_values`, and `value` per
  persisted node) so a long restore never re-runs the net. Under the `ppo`
  brain the map overlay's route is seeded on the net's pick, as the Python
  session does (`_driving_first_col`).
- **Run-mode runs** rebuild through the faithful constructor on seek /
  restore / persist import, and native-agent nodes (`__agent_native__`)
  replay through `take_action`.
- **No clock on wasm.** `wpa-verify`'s `max_seconds` deadline reads
  `Config::now_ms`, which the wasm shell wires to `Date.now`; native tests
  use `Instant`.
- The `ppo_impl` toggle is accepted for parity (`python|rust`) but the
  browser always runs tract; the panel hides the toggle in static mode.

## Instinct — the e2e brain (one net, no search)

The UI calls this brain and advisor **Instinct** ("Instinct (PPO only)" in the pickers); code, routes and stored values keep `e2e`.

`off_combat_brain: "e2e"` is the e2e Ironclad student (`sts2-rs/crates/sts2-e2e`,
weights `sts2-pycli/data/e2e/*.e2ew` + `.provenance.md`): one transformer that
makes EVERY decision of the run, fights included, with no search. Plan:
`docs/plans/2026-11-06-workbench-e2e-brain.md`.

- **Engine.** `sts2-workbench` feature `e2e` (default; the wasm shell has it).
  `e2e.rs` is the pure core (load, decide, `selfplay`); `session_e2e.rs` is the
  brain: `run_mode_step` routes every phase to `e2e_step`, which asks the
  `sts2_e2e::env::Driver` what comes next (`peek`), applies forced steps as the
  env does, otherwise tokenizes + runs the net greedily, applies the action
  natively and records a `__agent_native__` node. The step answers the
  agent-step shape: `search.scores` are the merged-class probabilities with
  labels, `debug` carries `decision_type`, `prob`, `ms_tokenize`, `ms_net`,
  `ms` (from `Config.perf_ms` = `performance.now()` in the browser). The
  student's per-run state (driver + tokenizer cache) is kept per node and
  rebuilt by shadow replay after seek / restore / import.
  `recommended_action` and the autobattle champion pick ask the student too.
- **Start.** `start-faithful {e2e: true}` forces the training run: Ironclad,
  v0.111.0, act 1 rolled, potion swap on, ascension default 10. The run
  identity carries `e2e_setup` through saves, persists and run logs.
- **Net, on demand.** ~21 MB. The worker fetches `/e2e/<file>` only when a call
  selects the brain (or the engine answers 424 for a restored run, retried
  once): not at boot, not in "Download for offline". The static build copies
  it to `dist/e2e/`; the service worker caches it cache-first in `wb-ppo-v1`;
  `_headers` serves `/e2e/*` immutable. The public export has no `data/e2e`
  (the unit allowlist names only the PPO champions), so there the build ships
  without it and the brain answers 424 with the fetch's 404.
- **Probe.** `POST /api/e2e/selfplay {seed, ascension?, max_decisions?}` plays
  a fresh run headless: `{won, decisions, chosen, ms_total, ms_per_decision,
  ms_{tokenize,net,env}_per_decision}`; `chosen` equals a native `E2eEnv` run.
- **Size + SIMD.** The brain adds ~0.7 MB to the engine wasm. `+simd128`
  (enabled since 2026-11-06) grows it 26,097,190 → 26,269,945 B after
  `wasm-opt -O3`, over Pages' 25 MiB per-file cap, so the engine ships
  gzipped (7,526,128 B; "Build, run, gates" above). Measured under node 26
  on the same source, full `selfplay` of seed 170000000 (634 decisions, A10,
  won, identical choices): 30.7 → 8.6 ms/decision (net 30.3 → 8.2 ms); the
  MCTS bench (`scripts/site-bench/wasm_hint_bench.mjs`, BENCH1) got faster
  too, 0.0302 → 0.0241 ms/sim at 4000 sims (0.0293 → 0.0211 at 200).

### Instinct as advisor (hand-played runs)

A run you play yourself can take the student as its **advisor** instead of the
search: `advisor: "e2e"` (Ironclad on v0.111.0 only — any other run keeps the
search). Every surface that advises a hand-played move then shows the
student's pick and its softmax probabilities (a POLICY's, never win chances):

| Surface | Route | Under the e2e advisor |
|---|---|---|
| combat hint, best-move star on hand / potions / end turn, reasoning sidebar | `/api/live/hint` | the student's merged classes as `Score` rows (`stable_idx` = hand index, `-1` end turn, `-100-s` potion; `prob` on every row); the sidebar draws label / bar / % and captions "e2e net" |
| play best (G), autoplay a turn (A), peek (P) | `/api/live/hint`, `/api/live/auto-action` | the student's pick |
| policy panel (card reward, shop, event, map, rest, Neow, card selects, potion swap, bundles) | `/api/live/ppo-policy` | `policy: "e2e"` payload: the student's classes, no V(s), no backend toggle |
| in-combat card-select chips | `/api/live/choice-ranker` | the student's probability per option ("e2e NN%") |
| choice coach chip | `/api/live/last-choice-grade` | agreement with the student (`policy: "e2e"`, no rollout check) |
| map route overlay | `full_map` / `/api/live/map` | seeded on the student's map pick |
| per-turn combat coach | `/api/live/last-turn-grade` | silent (409): it grades against the search's line |

**Win estimate (GH #1222).** The shipped `.e2ew` also carries a small probe
(`ppo.e2e.probe`: a ~0.13M-parameter MLP on the student's `h_STATE`, the
RUNCTX output its policy forward already computes, so it costs no extra
forward). `E2eNet::evaluate` returns it with the logits; `e2e-advice`, the
hint and the policy panel carry it as `estimate` — `{p_run, p_fight,
hp_loss: {q10, q25, q50, q75, q90}, hp_now}` (`p_fight` / `hp_loss` only in a
fight; `hp_loss` includes end-of-fight heals and is clamped to `hp_now`) — and
the sidebar shows one line under Instinct's advice: "If Instinct plays on:
fight 93% · −9 HP (−2 HP to −20 HP) · run 34%". The top bar shows the two
chances side by side as a chip, "Run 34% · Fight 93%" (the fight only in
combat), banded green / amber / red (`advice/estimate.ts` `estimateChip`);
the idle "Instinct ready" shrinks to its dot to make room. Every number assumes Instinct
plays every move from here; it is a state estimate, not a per-option value
(per-choice numbers wait on #1222 step 5). Run mode adds it to the step's
`debug.estimate` and summary. The probe is fitted per student: re-fit it
(`ppo.e2e.probe collect` / `fit` / `report`, then `export_wb weights --probe`)
whenever the weights change. Calibration: the `.provenance.md` next to the
weights.

Left on the search, deliberately: the search tree (`search_tree`, an explicit
MCTS tool), the on-demand EV analyzers (card / decision advisor, choice
"check it" — Monte-Carlo rollouts with MCTS fights, a different question than
"what would the net play"), the regret / WPA reviews and the autobattle
champion verdict (the run brain's, not the advisor's).

- **One evaluation per node.** `POST /api/live/e2e-advice` is the student's
  evaluation of the live node — `{available, reason, forced, is_combat,
  decision_type, ms, pick, options, scores}`; each option carries the class's
  workbench `action` / `args` (what `/api/live/action` takes), `label`,
  `prob`, `members`, `is_best` and the env's `run_action`. It is cached per
  node (`E2eSlot::advice`), so the hint, panel, auto-action and coach of one
  position share ONE forward (~20–45 ms in wasm).
- **Student state on a human's run.** While the student advises, every human
  move goes through `Driver::peek` + `note_external` (the per-node state the
  brain keeps); a run played without it (advisor switched on mid-run, a
  reload, a save-load) is rebuilt once by shadow replay from the deepest node
  that has a state. A position the driver cannot read (an unencodable action,
  a cap) answers `available: false` with the reason — and does not poison the
  positions after it (the driver's terminal is sticky, so the tracked state
  skips the peek that ended it). Out-of-distribution runs (Overgrowth act 1,
  potion swap off, other ascensions) still get an answer.
- **Choosing it.** `POST /api/advisor {advisor}` is the host's preference: the
  engine applies it to every run it installs after (start, save-load, import,
  play-from-here, resume) that the student can read, and to the live run.
  `POST /api/live/run-mode {advisor}` sets one run's advisor (400 for a run the
  student cannot read); the run-mode config persists it (named only when
  `e2e`). The page keeps the preference in `localStorage` (`wb.advisor`): the
  launcher's "Play yourself" card has an **advisor** picker and the top bar an
  **🔎 search / 🧠 e2e net** chip, both shown only for an Ironclad v0.111.0
  Rust run; picking the e2e brain to watch picks it to advise too. The default
  stays the search: the student's ~21 MB net downloads on first use (the
  worker fetches it when a call selects the advisor or answers 424), and
  v0.111.0 is not the default build. Tests: `tests/e2e_advisor.rs` (a
  hand-played run checked position by position against an independent
  `Driver` + net loop, through seek and a save-load rebuild).

## Replay viewer + deck analysis (M4)

`crates/sts2-workbench/src/runlog.rs` is `BranchingLiveSession.quit` +
`game_log.GameLogger` (the writer) and `timeline.py`'s `TimelineStore` +
`aggregate.py` (the reader); `replay.rs` mounts `/api/timeline`, `/api/map`,
`/api/step/{idx}[/deck-analysis]`, `/api/deck/trend`,
`/api/deck/reward-preview`, `/api/live/deck-analysis`, `/api/recommend`,
`/api/search_tree`, `/api/regret-review`, `/api/wpa-review` and
`/api/live/from_replay`; `deck.rs` is `deck_analysis.py`.

- **The log.** `POST /api/live/quit` writes the active path as the same
  `{step, ts, type, data}` records Python writes (identity header with
  `run_mode` / `act1` / `backend`, a `state` per node, each `action` with
  its native args + display label + replayable verb, `combat_log` rows,
  the `wpa` values, the terminal `summary`). The worker stamps `now`, keeps
  the text under its filename (`{stamp}_{character}_{seed}.jsonl`) in the
  `runs` store, lists it newest first (`/api/runs`, `/api/runs/stat`) and
  hands it back to `Workbench::load_run` on `/api/load`; `/api/mcts-runs`
  and its aggregate read every stored log.
- **The engine-backed replay routes rebuild the run.** Where Python
  re-scores a fight rebuilt from the rendered state through a lossy adapter,
  the port replays the recorded line to the step (the log carries the run
  identity and every action's replayable form) and searches the exact engine
  state — one forward replay for a whole regret review. Those three routes
  are structural twins in the golden parity (like the live regret coach);
  "play from here" is the same rebuild through `Session::load_line`.
- **Deck analysis** is bit-exact: `deck.rs` carries a port of CPython's
  `random.Random` (MT19937, `_randbelow`, `shuffle`) so the Monte-Carlo
  rollouts draw the same shuffles; the two tables (`card_db.json`,
  `card_tags.json`) ride in the data bundle (`just workbench-data` copies
  them from `python/workbench/data/`). They are per game build (GH #877): a
  non-default build has its own `card_db.json` and a `card_tags.json`
  overlay under `python/workbench/data/<ver>/`, and every deck route grades
  with the run's own build's tables.
- The run log is written with the Python normaliser's action shape (a
  hand-played potion records `slot: null` — the M2 asymmetry stays in the
  live coach only), so a recorded line hashes like the oracle's.

## Seed check (`/seedcheck.html`)

A page for a seed seen in a post ("maybe unwinnable"): paste the game seed,
and Instinct plays it until it wins. Linked from the top bar (🎲 Seed check).
The page picks character, build and ascension. Instinct only plays Ironclad
on v0.111.0, so the other characters and builds are listed but disabled (the
engine refuses them with a 400 too); any ascension 0–10 runs, with a note
below A10 that the net trained at A10. Design:
`docs/superpowers/specs/2026-11-08-workbench-seed-checker-design.md`.

- **The seed is the game's.** `sts2_e2e::env::new_run_from_game_seed`
  canonicalises (uppercase, `O`→`0`, `I`→`1`) and hashes with the v0.111.0
  run-seed hash, the same as `sts2_rs.RunManager(string_seed=…)` and the C#
  oracle. The page shows the Neow offers and the bosses before anything runs,
  so a typo is visible. A character outside the game's alphabet is a 400; a
  length other than 12 is shown as a warning.
- **Check.** Attempt 0 is greedy, then T=1.0 samples, each attempt's PRNG
  seeded from (seed, `check`, k) (`e2e::attempt`), so attempt k replays from
  its index alone. `pool.ts` runs `min(8, cores − 1)` workers
  (`seedcheck.worker.ts`, each its own wasm and net), stops at the first win
  or at 5 minutes. No win: the rule-of-three bound over counted attempts
  (engine errors and decision caps excluded), the best attempt saved, and
  "Keep searching" continues the same sequence from the next k.
- **Win rate.** A separate sequence (seed, `rate`, k), 200 sampled attempts,
  Wilson 95% interval and deaths per floor with the room.
- **Replay.** The winning (or best) attempt's class sequence is replayed
  through a live e2e session (`POST /api/seedcheck/log`) and stored like any
  run log; the result links to `/?open=<path>`. The run identity carries the
  string seed as `game:<SEED>` (`ModeSeed`), so seek and restore rebuild it.
- **The text.** `verdict.ts` writes the paste-ready verdict; it never says
  "impossible". It carries the route floor by floor
  (`seedcheck::line_summary` `route`, built from `visited_map_coords` so
  forced moves are there too): the node's room ("?" says whether it was a
  fight or an event), which node to take ("go left/straight/right" from the
  node you stand on, "Nth of M from the left" on that floor), the encounter
  or event, the choices and relics, and the HP on leaving. With no win, the
  best attempt's route. Paragraphs are blank-line separated and the route is a
  `- ` list, so Reddit and Discord markdown keep the lines apart.
- **Acceptance (node, no browser).** `node web/scripts/seedcheck-node.mjs`
  after `just wasm-build`: the greedy attempt must win both reference seeds and
  the sampled rate on `1SNVTCEXKR0Q` must fall in [4.6%, 12.0%], the Wilson
  interval of the Python `ppo.e2e.solve` d1250 rate (15/200). That reference
  predates the d2500 avg5 (2026-11-08) and d5250 (2026-11-13) nets and needs re-measuring.

## Persistence

IndexedDB `sts2-workbench`:

- `saves` — key = slug, value = today's `saves.py` payload
  (`schema_version: 1`, identity, `progress`, `step_count`, `line`,
  `node_values`). `saved_at` is injected by the worker's clock.
- `live/current` — identity + the branch tree **without node states** (but
  with each node's `value`, the run mode and its config) + the active leaf,
  rewritten after every mutating call. On load the worker rebuilds the root
  and replays the active path (progress frames drive the "Restoring run
  n/N" banner); other branches materialize lazily on `seek`.
- `runs` — the JSONL run logs finished runs leave behind, keyed by
  filename, `{text, size, mtime}` (M4).

`/api/live/quit` clears `live/current` and answers `{ok: true, log_path: null}`.

## Phone-speed probe (`POST /api/bench`)

`{sims, playouts, seed}` runs the hint endpoint's single-tree search
(`mcts_recommend`'s config, `threads=1`) over the live combat plus seeded
heuristic playouts, and reports the work done; the caller times it. In the
static build the transport is exposed as `window.__sts2ApiFetch` so a script
can call it from outside the bundle. Numbers: `sts2-rs/docs/perf/2026-09-02-browser-workbench-bench.md`.

## Bug reports

The Python workbench builds a report server-side and puts it in GCS
(`sts2-pycli/CLAUDE.md`, "Bug reports"). The browser build has no server, so
the roles move: the **page** builds the same schema-1 dump
(`web/src/bugreport.ts`) from what the worker exposes, and the **host** —
Cloudflare Pages — stores it through a Pages Function (`web/functions/api/`)
in a Workers KV namespace (`web/wrangler.toml`; deploy/DEPLOY.md has the
why-KV and the one-time setup). Same id format (`wb-` + 16 hex), same
`?report=<id>` view.

What differs is what a report carries, because the reporter's run exists
nowhere but in their IndexedDB:

- a **live** report carries the run's persisted tree (`GET /api/live/persist`,
  the payload a reload resumes from) plus the enriched step and the map;
- a **replay** report carries the run log's text (`GET /api/runs/read?path=`)
  plus the viewed step.

The report view's "Load into this workbench" takes either back
(`POST /api/live/import {persist}` replaces the live run and persists it, so
the reload resumes the reporter's run; `POST /api/runs/put {path, text}` adds
the log to the Runs list under `<report id>_<original name>.jsonl`). Those
four are host routes in `engineRoutes.ts`, not wasm ones.

The 🐞 Report button appears only where a capture would land: the SPA probes
`GET /api/bugreport` once and wants the Function's JSON (`{ok: true}`) — the
SPA fallback of a plain static host and `vite preview` answer HTML / 404 and keep it hidden.
From the desk, `just bug-reports` lists the store and `just bug-report <id>`
downloads one (a replay report's log unpacks to `logs/bugreports/<id>.jsonl`,
which the desktop workbench opens).

Abuse budget: Pages' free plan has no rate-limit binding, so the caps are a
20 MiB body, an origin check on POST (a foreign `Origin` is refused; the
intake below also takes a post with none — a non-browser client, which could
forge the header anyway), and KV's own 1,000 writes/day — a flood costs a day
of lost reports, never money.

### The sign-in, and the intake outside it

The hosted site is behind a sign-in (Cloudflare Access, Google accounts):
the app, its assets and `/api/bugreport/<id>` all need it. The clients below
cannot sign in, so they post to the store's **intake**, `POST
/intake/bugreport` (`web/functions/intake/bugreport.ts`) — the one path
exempt from it, with its own Access application (`sts2-pycli/deploy/DEPLOY.md`,
"Bug-report intake"). It only writes; every other path under `/intake/` is a
404, so the exemption exposes no page, asset or report. Exempting
`/api/bugreport` instead would have opened every stored report: an Access
path covers everything under it. The page itself keeps posting to
`/api/bugreport`, signed in and same-origin.

### Reports from the real game

Two more clients post to the same store, so one `just bug-reports` lists
everything players send:

- **The Spire Mentor mod**'s in-game **Report a bug** (`Ctrl+Shift+B`;
  `spire-mentor-mod/SimBugReport.cs`, the README's "Bug reports"). Its dump
  says `meta.source = "mod"`, `meta.mode = "game"`, and carries the game's
  position as `state.game_state` (the Copy-state message), the start-of-room
  save as `repro.run_save`, the advisor's run log as `repro.run_log` when
  the advisor is up, and log tails as `logs` (mod / game / advisor). The mod
  posts it itself to the intake, with no `Origin` and following no redirect
  (a redirect is the sign-in page, reported as such); the site is baked into
  release builds (`package.ps1 -BugReportSite`), never named in source.
- **The mod-bundled workbench** (served by the advisor's control plane on
  :8079). Its 🐞 Report posts to `/api/bugreport` on that origin as usual,
  and the control plane (`realgame_control.BugReportForward`) forwards it to
  the intake of the site named by `--bug-report-site` (the mod passes its own
  site), adding
  the game's latest position as `state.live_game` and the run envelope as
  `meta.advisor`. With no site, or no store to reach, it saves the report
  under `<log dir>/../bugreports/<id>.json` and the toast shows that path
  instead of a link. It refuses a foreign `Origin` and any body that is not
  `application/json` (a cross-site page could send that only after a
  preflight it refuses). Opening a sent report (`?report=<id>`) works on the
  hosted site, signed in; the bundled page can open only the ones it saved.

The report view shows the game side of either (mod and game build, run, coach
mode, the logs) and offers **▶ Import the game position**, which runs the
position through the state import (`POST /api/live/import-state`, lane
`auto`, in the build it was taken in) and makes it this workbench's live run.
`just bug-report <id>` unpacks it as `<id>.state.json` (paste it into Import
state), `<id>.run_save.json` and `<id>.<mod|game|advisor>.log`.

## State import

Press **📋 Import state** and paste a run position copied out of the live game
(Spire Mentor's **Copy state** / `Ctrl+Shift+C`, or an advisor log — the newest
position in a pasted log wins). The dialog previews which run it is and what
the import will and will not reproduce *before* it replaces the run you have
open, then rebuilds it in the wasm engine.

Two lanes, as on the desktop. The **run lane** reconstructs the position on
the real seed's maps, and the deck, relics, potions, gold, HP and every RNG
cursor are grafted on from the paste, so the next card reward, shop and event
roll are the ones your run was about to get. It parks on the node you were
standing on. The **combat lane** (GH #708) resumes the exact fight — the hand,
the draw / discard / exhaust piles in order, every power, every intent, the
energy and the block, plus the game's own RNG lineage — bit-identical to the
board on screen. The dialog offers it whenever the paste carried a fight the
adapter could translate ("auto" prefers it), and the **open in SimLab** handoff
loads the same board into the sandbox.

The port lives in `sts2-rs/crates/sts2-workbench`: `state_import.rs` is the
paste parser (of `python/workbench/state_import.py`), `combat_import.rs` is
the mod-wire → engine-wire adapter (of `realgame_adapter.to_headless_combat_play`
+ `sts2_hint.adapter.translate`, composed into one pass over the mod's
objects; the title maps come from the embedded localization, the id checks
from the engine registries), `Game::start_imported` is the graft (of
`RustGame.start_imported`), and `/api/live/import-state[/preview]` are the
routes (of the Python endpoints). The graft order is load-bearing — place on
the map, empty the deck **before** granting relics (Whetstone / War Paint
upgrade on landing), deck then upgrades, HP/gold, RNG cursors, and the fight
last of all (`RunManager::debug_enter_combat`: the rebuilt `CombatState`
dropped onto the run as it stands, nothing re-rolled) — and
`Session::start_imported` keeps the snapshot, fight included, so `seek`, a
reload and a named save all rebuild through the graft rather than as a fresh
seed run.

One difference from the desktop is deliberate: there the combat lane is a
bare fight with no run behind it (`ImportedCombatBackend`); here the fight
sits on the run the run lane rebuilt, so when it ends the reward screen and
the map are there — on whatever node the paste placed the run (a fight paste
carries no map, so that is the act entrance, and the warning says so). Nothing
is guessed: a field the paste omits (no map, no RNG) becomes a warning shown
above the board, because a half-restored state that *looks* whole is the
expensive failure. The adapter keeps the desktop's fail-loud / soft-drop
split — an unknown monster or orb refuses the fight (the run lane still
imports, with the reason), an unknown card / relic / power / potion is
dropped — but where the desktop counted drops into a tally nobody read, every
drop here is a named warning, as is a pet with no engine representation.
`tests/state_import_combat_parity.rs` pins the adapter against the Python
chain's wire dicts for every real capture under
`sts2-pycli/tests/fixtures/realgame_adapter/`
(`just state-import-golden-dump` re-records them).

## SimLab sandbox (M5)

`/simlab.html` is a position editor with no run behind it: set up a fight
(character, ascension, turn, encounter, seed), edit the player (HP, block,
energy, relics, potions, powers, orbs), lay out each pile — the draw pile **in
order** — and each enemy with HP, block, powers and its telegraphed move, then
ask the engine two questions. Every edit re-asks the engine to build the board
(the **Position** card says "Built." or names the field it refused); **search**
runs the workbench's MCTS on the board and returns win probability per first
move. You can paste a position in, deal a fresh fight, or hand-author one.

The distribution **sweeps** ("play it out" as a distribution / CDF / box
charts, and the Studies tab's deck × encounter × ascension batches) run in
the browser too since GH #707, so `capabilities.features.simlab_sweeps` is
`true`. On the desktop a sweep is a background job over a process pool with
its state on disk; the wasm is single-threaded and has no threads, files or
clock, so `simlab_sweep.rs` makes it a **host-stepped job**: the Web Worker
calls `Workbench::sweep_tick` (plays trials for ~50 ms), yields to its message
queue — so a status poll or the sandbox's own search queued meanwhile is
answered between trials and the page never freezes — and enqueues the next
tick until nothing is running (`engine.worker.ts::pumpSweeps`). **Stop** sets
a flag the next trial honours; the desktop's 10-minute wall-clock cap applies
(`stopped_reason: max_runtime`). Finished sweeps — and, every few seconds, the
running one — are exported to the `sweeps` IndexedDB store, so the history
survives a reload (a sweep cut by a reload comes back `cancelled` with its
partial rows). The strategies (`random`, `block_first`, `attack_first`,
`min_overblock`, `mcts`), the seed pairing, the row schema and the bucket
statistics are ports of `simlab/{runner,agents,intel,telemetry}.py` and
`app._kde`; `tests/sweep_parity.rs` replays two small sweeps and compares the
rows and the results payload against the Python runner's
(`just simlab-sweep-golden-dump` re-records `tests/golden/simlab_sweep.json`).
The `random` strategy carries CPython's `random.Random` (the `deck.rs` port), so
even its rows match. One deliberate deviation: an unknown strategy is a 422 at
submit, where the desktop runner would have produced a row of errors.

Importing a paste accepts a `simlab-position/1` document, an engine wire
dict, or a live-game Spire Mentor fight paste (translated through the same
`combat_import.rs` adapter the state-import combat lane uses; a fight the
engine cannot rebuild is refused with the reason).

The port lives in `sts2-rs/crates/sts2-workbench`: `simlab.rs` is the
position ↔ wire logic (of `python/simlab/position.py`), `simlab_catalog.rs` is
the catalog + move/encounter tables (of `catalog.py`), and
`GET /api/meta`, `GET /api/simlab/catalog` and
`POST /api/simlab/position/{build,new,import,moves,mcts}` are the routes (of
`python/simlab/web/app.py`), plus `POST /api/sweeps`, `GET /api/sweeps`,
`GET /api/sweeps/{id}`, `POST /api/sweeps/{id}/stop` and
`GET /api/sweeps/{id}/results` (of the same file's sweep half + `jobs.py`,
in `simlab_sweep.rs`). The correctness-critical fresh-fight construction
is the **shared** `sts2_wire::scenario::build_scenario_state` — lifted out of
`sts2-py` so the Python parity oracle and the browser build boards
byte-identically — while the wire round-trip and the MCTS are the same
pure-Rust `sts2_wire` emitters the advisors already use. The whole
`web/src/simlab/` SPA was already built for the desktop server; the browser
change is routing its `/api/*` calls through the worker transport
(`src/simlab/api.ts` → `apiFetch`).

## Live-game bridge (GH #709)

The live-game views — the **Game tree** of the real run (click a save point
to rewind the actual game), **Live search** over the live board, and **rewind
to the viewed floor** — talk to the advisor's control plane (the Spire Mentor
mod's rewind lane: `GET /status`, `/tree`, `/snapshots`, `POST /restore`,
`/search_tree` on `127.0.0.1:8079`, `sts2-pycli/python/realgame_control.py`).
The Python workbench reached it through a same-origin proxy (`/api/bridge/*`);
the browser build has no server, so the page calls the control plane itself:
a cross-origin fetch from `https://your-workbench.example.com` to a loopback
`http://` address. Whether a browser allows that was the open question.

### The transport question, measured (2026-09-08)

`sts2-pycli/scripts/spike_bridge_transport.py` starts two fake control
planes on loopback — one with the page origin allow-listed, one without —
and `web/scripts/bridge-transport-spike.mjs` loads the **real deployed page**
in Playwright's browsers and fetches both from it (`GET /status`, a simple
request, and `POST /restore` with a JSON body, which preflights):

| Browser | Control plane with the opt-in (CORS + PNA headers) | Without |
|---|---|---|
| Firefox 151.0 | **works**: GET and the preflighted POST, `127.0.0.1` and `localhost` | blocked (`Access-Control-Allow-Origin` missing) |
| Chromium 149.0.7827.55 | blocked in headless: `Permission was denied for this request to access the loopback address space` — Chrome's **Local Network Access** permission (a prompt in a headed browser; the fetch waits on it, so a headed run under xvfb hangs on the prompt). With that check off (`--disable-features=LocalNetworkAccessChecks`, i.e. the permission granted) **works**: GET and the preflighted POST | blocked (CORS) |

So the three gates the issue named resolve as: **mixed content** is not a
blocker (a loopback `http://` target is exempt in both browsers);
**CORS** is, and is answered by the control plane's opt-in; **Private Network
Access** in current Chrome is a user permission rather than the
`Access-Control-Allow-Private-Network` header alone — the header is sent
too, and the page passes `targetAddressSpace: "loopback"` on its requests
(Chrome's hint), but the user has to allow the prompt once. Viable, with
two opt-ins, both off by default.

### The setup

1. **On the advisor** (the machine running the game): allow this page's
   origin. In the mod's settings, **Browser workbench origin** =
   `https://your-workbench.example.com` (`SpireMentorModConfig.BrowserWorkbenchOrigin`
   → the advisor's `--control-allow-origin`), or pass the flag yourself.
   The control plane then answers that origin — and no other — with
   `Access-Control-Allow-Origin` (echoed, never `*`), the methods / headers
   allow-list, `Access-Control-Allow-Private-Network: true` and a
   `do_OPTIONS` preflight; with no origin listed it sends no CORS headers at
   all, so no web page anywhere can read it (the same-machine proxy never
   sends an `Origin` and is unaffected). `tests/test_realgame_control.py::TestBrowserAccess`
   pins both halves.
2. **In the page**: Tools → **🎮 Live game…**, enter the control plane URL
   (default `http://127.0.0.1:8079`; the game's machine if that is another
   one), connect. `src/bridge.ts` keeps it in `localStorage`; nothing is
   probed until a URL is set. The status probe then runs in every view (not
   only the replay of an advisor log, which the browser build cannot follow),
   and the Analyze menu grows **Game tree** and **Live search** while the
   advisor answers `connected`; **rewind to the viewed floor** stays a replay
   command (it needs a viewed floor). The Run menu grows **🎮 Import from
   live game**, which opens the state import pre-filled from the control
   plane's `GET /state` — the latest `advice_request` the mod sent, i.e.
   exactly what Copy state would have put on the clipboard (the import
   dialog also offers it as a button while connected). The Tools entry shows the state
   (`connected` / `unreachable`), and the dialog shows the probe error.
3. **In Chrome**, allow the "access devices on your local network" prompt
   the first request raises. Firefox does not ask.

**The bundled workbench needs none of this.** The Spire Mentor release zip
ships this build in `mods/SpireMentor/workbench/` (`spire-mentor-mod/package.ps1`),
and the control plane serves it itself (`realgame_control.StaticSite`): the
page lives on `http://127.0.0.1:8079`, the control plane's own origin, so no
CORS, no local-network prompt, and `VITE_BRIDGE_SAME_ORIGIN=1` makes
`src/bridge.ts` connect to that origin on load. The champion nets are not
duplicated into it: `/ppo/<file>` falls back to the advisor's bundled
`data/ppo`. Follow (`?follow=1`, the replay view tailing the advisor's run
log) works there too: the page cannot read the advisor's file, so
`src/liveLog.ts` polls the control plane's `GET /log?stat=1` and, when the
log changed, copies `GET /log` into the worker's run store
(`POST /api/runs/put`), after which App.tsx's follow loop runs unchanged.

The proxy's routes mapped 1:1 onto the control plane's; the one transform it
did — rendering each tree node's `combat_log_raw` with the localization —
is the wasm's `POST /api/bridge/render-tree` (`enrich::log_names` +
`combatlog::format_combat_log`, the same formatter a simulated run goes
through). `/api/bridge/status` and friends stay 501 in the worker: with a URL
configured, `api.bridge.*` never goes through it.

## Claude link (terminal Claude Code ↔ the open workbench)

A Claude Code session started in the repo can read the run on screen, call
any workbench route against it, and put charts and explanations into a dock
next to the board. The run in the browser build exists only in the tab's
worker, so the link goes through the page:

```
Claude Code ──stdio MCP──► sts2-pycli/python/claude_link.py ◄──ws://127.0.0.1:8091── the workbench tab
```

1. `.mcp.json` (repo root and `sts2-pycli/`) registers the relay as the
   `workbench` MCP server; Claude Code asks once to enable it. It runs in its
   own `uv run --no-project` env, so it never builds the engine wheel.
2. In the workbench: Tools → **🤖 Claude link** (remembered in
   `localStorage`). The first time, it asks for the **pairing token** — ask
   Claude for it (`wb_status` carries it), or open the workbench with
   `#claude-link=<token>` appended. The dock shows `connected` once the
   relay accepts the page.
3. Ask Claude about the run. Its tools: `wb_status`, `wb_context` (mode + the
   shown step, render-only fields stripped), `wb_call(route, method, body)`
   (through `apiFetch`, so the same engine and live run; NDJSON advisor
   streams come back as their final frame), `wb_show(title, html | markdown |
   plotly | file)`. The relay's MCP instructions list the useful routes.

Details that matter:

- **Mutations.** A non-GET `/api/live/*` call that is not one of the
  read-only searches / advisors / reviews (`READ_ONLY_LIVE` in
  `web/src/claude/link.ts`) makes the page re-read the live state and the
  timeline afterwards, so a Claude-taken action or seek shows on the board.
- **Pairing.** The relay closes any connection whose hello lacks its token
  (code 4003; the page then stops retrying and offers *pair…*). Without it,
  any local process or localhost page could pose as the page: feed Claude a
  made-up run and receive what Claude shows. The token lives in
  `~/.config/spire-mentor/claude-link-token` (0600, created on first use;
  `CLAUDE_LINK_TOKEN` overrides), so each workbench origin pairs once.
- **Sandbox.** Every dock item — HTML, markdown, Plotly — renders in an
  `allow-scripts` iframe with no `allow-same-origin`: opaque origin, no access
  to the page, its worker or IndexedDB. Markdown (`marked`) and Plotly load
  from a CDN inside the frame; the link only runs next to an online Claude
  Code session anyway.
- **Origins.** The relay binds loopback only and accepts a localhost page or
  an origin in `CLAUDE_LINK_ORIGINS` (comma-separated). The hosted build needs
  that variable set for the relay, and Chrome's local-network prompt allowed
  once (see "Live-game bridge"). `CLAUDE_LINK_PORT` moves the port; the page's
  stored URL (`sts2.claudeLink`) must match.
- **One link.** One relay holds the port; another Claude Code session's relay
  retries every 5 s and its tools say the port is taken. One page holds the
  relay; a newer tab takes over (close code 4000) and the older one stops
  retrying.

## What is not there yet (roadmap)

| Milestone | Scope |
|---|---|
| M2 | **Shipped** (see "Advisors"): the search-backed and rollout advisors over worker RPC, streamed progress. |
| M3 | **Shipped** (see "Run mode"): PPO in-browser, run mode / autobattle, node values, WPA review + verify, run-mode saves. |
| M4 | **Shipped** (see "Replay viewer"): run logs on quit, the replay viewer and its reviews / searches / play-from-here, the MCTS Runs page, deck analysis. |
| M5 | **Shipped**: bug reports, state import (both lanes since GH #708), the SimLab sandbox and — since GH #707 — the SimLab distribution sweeps (see those sections). |
| M6 | wasm threads (COOP/COEP: the `/*` block in `sts2-pycli/web/public/_headers`; `crossorigin` on art elements only for a remote art origin). |
| M7 | Android TWA; native `sts2-coach` binary over `Workbench::call` for the Spire Mentor lane. Then retire the Python workbench modules and `deploy/`. (The live-game views already reach the advisor directly — see "Live-game bridge".) |

Known M1 limits: single restore attempt on load; `size_jitter` / `hue_jitter`
are the only floats in the wire; the `csharp` backend dialect and
`start_faithful` are not ported. `start_imported` and the `force_*` /
`debug_*` graft ARE ported now (M5, both lanes since GH #708 — see "State
import").
