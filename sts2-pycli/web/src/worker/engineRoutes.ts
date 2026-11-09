/**
 * The `/api/*` router that lives INSIDE the engine worker.
 *
 * Pure logic over two injected seams — the wasm `Workbench` (or anything with
 * its shape) and a `KvStore` — so it is unit-testable without WebAssembly or
 * IndexedDB. `engine.worker.ts` is the thin message loop around it.
 *
 * Contract (mirrors the Python routes the SPA was written against):
 *   - Routes the wasm serves directly go through `Workbench.call`.
 *   - Saves are the worker's job, over the `saves` store:
 *       POST /api/live/save {name}  -> build_save, store under slugify(name)
 *       GET  /api/saves             -> every payload through save_header
 *       POST /api/saves/load {name} -> restore_save(stored payload)
 *   - The streaming advisor routes (`…/stream`) go through
 *     `Workbench.call_stream`; every frame it hands back is forwarded to the
 *     caller's `onFrame` (the worker posts them as `{id, frame}`), and the
 *     reply is a 200 with a null body — the frames ARE the response.
 *   - Run logs (M4): a `POST /api/live/quit` leaves the run's JSONL behind,
 *     and so does a start / import / resume / save-load for the live run it
 *     replaces (`Workbench.take_run_log`); the worker stores each under its
 *     filename (suffixed `_2`, … on a clash) in
 *     the `runs` store, lists it (`GET /api/runs`, newest first, `GET
 *     /api/runs/stat`), reopens it (`POST /api/load` → `Workbench.load_run`)
 *     and feeds every stored log to the MCTS Runs page
 *     (`GET /api/mcts-runs[/aggregate]`). The replay routes themselves go
 *     through `Workbench.call` like everything else.
 *   - The PPO champion (M3): before a route that starts or restores a run
 *     the worker fetches that character's champion ONNX (`/ppo/<file>`, the
 *     file `Workbench.ppo_champion_file` names) and hands the bytes to
 *     `Workbench.load_ppo`, once per character. A failed fetch (offline on
 *     first use, a 404) is recorded with `set_ppo_error` so the panel and the
 *     brain can say why, and is retried on the next start / panel request.
 *   - The e2e student: its weights (`/e2e/<file>`, `Workbench.e2e_file`,
 *     ~21 MB) are fetched ON DEMAND only — when a call selects the `e2e`
 *     brain or advisor (`start-faithful {e2e: true}`, `run-mode
 *     {off_combat_brain: "e2e"}` / `{advisor: "e2e"}`, `/api/e2e/selfplay`,
 *     `/api/live/e2e-advice`) or the engine answers 424 (a restored
 *     or resumed e2e run with no net yet), after which the call is retried
 *     once. Never at boot, never by "download for offline". A failed fetch
 *     is recorded with `set_e2e_error` and retried on the next such call.
 *   - Bug reports (M5): the page builds the dump (src/bugreport.ts) and
 *     posts it to the host's Pages Function; the worker only supplies the
 *     two things a report carries that the wasm routes do not expose —
 *     the live tree (`GET /api/live/persist` = `export_persist`) and a
 *     stored run log's text (`GET /api/runs/read?path=`) — and takes them
 *     back when a reader loads a report into their own workbench
 *     (`POST /api/live/import {persist}`, `POST /api/runs/put {path, text}`).
 *   - SimLab sweeps (GH #707): the wasm owns the jobs (`/api/sweeps*`), but
 *     it is single-threaded and cannot run one in the background, so the
 *     worker STEPS it: `sweepTick` plays trials for ~50 ms and returns, and
 *     `engine.worker.ts` calls it between the page's requests (a status
 *     poll, the sandbox's own search are answered between trials) until
 *     nothing is running. A sweep that ends — and, every few seconds, one
 *     still running — is exported to the `sweeps` store so the history
 *     survives a reload; the first listing after a reload imports them back.
 *   - Everything else under /api/ answers 501 (replay tools — the SPA
 *     already tolerates those failing).
 *   - After every successful mutating call the live run is persisted to
 *     `live/current`; a GET /api/live/state with no run in memory imports it
 *     back (that is how a reload resumes the run), forwarding the wasm's
 *     progress callback as `{done, total}` frames.
 */
import type { KvStore } from "./idb";

/** Reply envelope, exactly what the Python server would have answered. */
export interface Reply {
  status: number;
  body: unknown;
}

export interface WorkbenchLike {
  call(method: string, route: string, body_json: string): string;
  /** Streaming routes: `on_frame` gets each NDJSON frame as a JSON string. */
  call_stream(method: string, route: string, body_json: string, on_frame: (frame: string) => void): string;
  build_save(name: string, saved_at: string): string;
  /** `now` (ISO-8601) stamps the log of the live run the restore replaces. */
  restore_save(payload_json: string, now: string): string;
  export_persist(): string;
  import_persist(payload_json: string, progress: (done: number, total: number) => void): string;
  has_live_run(): boolean;
  /** PPO champion (M3): load from the fetched ONNX bytes / record why not. */
  load_ppo(character: string, bytes: Uint8Array): string;
  set_ppo_error(character: string, reason: string): void;
  has_ppo(character: string): boolean;
  /** The e2e student: load from the fetched `.e2ew` bytes / record why not.
   *  Absent = no e2e brain in this host (older fakes). */
  load_e2e?(bytes: Uint8Array): string;
  set_e2e_error?(reason: string): void;
  has_e2e?(): boolean;
  /** Run logs (M4): the log the last quit wrote (`{filename, text}` JSON or
   *  `"null"`), and a stored log's text parsed into the replay store. */
  /** The oldest run log not yet collected (`{filename, text}` JSON), or
   *  `"null"` when none is left. */
  take_run_log(): string;
  load_run(path: string, text: string): string;
  /** SimLab sweeps (GH #707): step the oldest running sweep (`{active,
   *  finished}` JSON), export one for storage (`"null"` if unknown), take a
   *  stored one back. Absent = no sweeps in this host (older fakes). */
  sweep_tick?(max_trials: number): string;
  has_running_sweep?(): boolean;
  export_sweep?(id: string): string;
  import_sweep?(doc_json: string): string;
}

export interface EngineDeps {
  wb: WorkbenchLike;
  /** `Workbench.save_header` (a static, so passed separately). */
  saveHeader: (payload_json: string) => string;
  /** `Workbench.mcts_runs` / `aggregate_runs` (statics) over
   *  `[[filename, text], …]` JSON; absent = the MCTS Runs page is empty. */
  mctsRuns?: (logs_json: string) => string;
  aggregateRuns?: (logs_json: string) => string;
  /** `Workbench.ppo_champion_file` (a static): the champion ONNX filename
   *  for a character, as a reply JSON. */
  championFile?: (character: string) => string;
  /** Fetch a champion file's bytes (`/ppo/<file>` in the worker). Absent =
   *  no PPO in this host (tests). */
  fetchChampion?: (file: string) => Promise<Uint8Array>;
  /** `Workbench.e2e_file()`: the e2e weights filename; absent / empty = no
   *  e2e brain in this host. */
  e2eFile?: string;
  /** Fetch the e2e weights (`/e2e/<file>` in the worker), on demand only. */
  fetchE2e?: (file: string) => Promise<Uint8Array>;
  kv: KvStore;
  /** Clock for `saved_at`; defaults to now, second precision, `Z`. */
  now?: () => string;
}

export type ProgressCb = (done: number, total: number) => void;
export type FrameCb = (frame: unknown) => void;

export interface Engine {
  handle(
    method: string,
    route: string,
    body?: unknown,
    onProgress?: ProgressCb,
    onFrame?: FrameCb,
  ): Promise<Reply>;
  /** Play sweep trials for about `budgetMs` (default 50) and store what
   *  ended; resolves to whether a sweep is still running (call again). */
  sweepTick(budgetMs?: number): Promise<boolean>;
  /** Whether a sweep still has trials to play. */
  hasRunningSweep(): boolean;
}

/** Routes `Workbench.call` serves, by `METHOD /path`. */
const WASM_ROUTES = new Set([
  "GET /api/health",
  "GET /api/capabilities",
  "POST /api/live/start",
  "GET /api/live/state",
  "GET /api/live/map",
  "POST /api/live/action",
  "GET /api/live/timeline",
  "POST /api/live/seek",
  "POST /api/live/quit",
  "POST /api/bench",
  // Advisors (M2).
  "POST /api/live/hint",
  "POST /api/live/search_tree",
  "POST /api/live/choice-ranker",
  "POST /api/live/auto-action",
  "POST /api/live/last-turn-grade",
  "POST /api/live/last-choice-grade",
  "POST /api/live/regret-review",
  "POST /api/live/card-advisor",
  "POST /api/live/decision-advisor",
  "POST /api/live/wpa-review",
  "POST /api/live/ppo-policy",
  // The e2e student's evaluation of the live node (the e2e advisor), and
  // the host's advisor preference.
  "POST /api/live/e2e-advice",
  "POST /api/advisor",
  // Run mode / autobattle / WPA verify (M3).
  "POST /api/live/start-faithful",
  // The e2e student's headless self-play (speed / parity probe).
  "POST /api/e2e/selfplay",
  // The seed checker (seedcheck.html): a seed's preview and an attempt's
  // named line. Its attempts run in its own workers; its run logs are
  // stored by `POST /api/seedcheck/log` below.
  "POST /api/seedcheck/preview",
  "POST /api/seedcheck/summary",
  "POST /api/live/run-mode",
  "POST /api/live/run-mode-step",
  "POST /api/live/faithful-step",
  "POST /api/live/play-to-end",
  "POST /api/live/autobattle/advance",
  "POST /api/live/autobattle/step",
  "GET /api/live/autobattle/verdict",
  "POST /api/live/wpa-verify",
  // The replay viewer + deck analysis (M4).
  "GET /api/timeline",
  "GET /api/map",
  "GET /api/coach",
  "GET /api/deck/trend",
  "POST /api/deck/reward-preview",
  "GET /api/live/deck-analysis",
  "POST /api/recommend",
  "POST /api/search_tree",
  "POST /api/regret-review",
  "POST /api/wpa-review",
  "POST /api/live/from_replay",
  // State import (M5): rebuild a pasted live-game RUN position in the engine.
  "POST /api/live/import-state/preview",
  "POST /api/live/import-state",
  // SimLab sandbox (M5): stateless position editor + per-move MCTS. The
  // distribution sweeps (/api/sweeps*) are desktop-only, so they are NOT here
  // — the SimLab page hides that UI in static mode.
  "GET /api/meta",
  "GET /api/simlab/catalog",
  "POST /api/simlab/position/build",
  "POST /api/simlab/position/new",
  "POST /api/simlab/position/import",
  "POST /api/simlab/position/moves",
  "POST /api/simlab/position/mcts",
  // SimLab sweeps (GH #707): submit + list; `/api/sweeps/{id}…` matches
  // SWEEP_ROUTE below. Stepping the job is the worker's (see sweepTick).
  "POST /api/sweeps",
  "GET /api/sweeps",
  // Live-game bridge (GH #709): the page talks to the advisor's control
  // plane itself (src/bridge.ts); the wasm only renders the tree's raw
  // combat logs. The proxy routes (`/api/bridge/status` …) stay 501 here.
  "POST /api/bridge/render-tree",
]);

/** `GET /api/sweeps/{id}`, `POST /api/sweeps/{id}/stop`,
 *  `GET /api/sweeps/{id}/results` (query string passed through). */
const SWEEP_ROUTE = /^\/api\/sweeps\/[^/]+(\/(stop|results))?$/;

/** `GET /api/step/{idx}` and `GET /api/step/{idx}/deck-analysis`. */
const STEP_ROUTE = /^\/api\/step\/\d+(\/deck-analysis)?$/;

/** Routes answered as a frame stream (`Workbench.call_stream`). */
const STREAM_ROUTES = new Set([
  "POST /api/live/card-advisor/stream",
  "POST /api/live/decision-advisor/stream",
  "POST /api/live/last-choice-check/stream",
]);

/** Calls after which the live run must be re-persisted (or dropped). */
const MUTATING = new Set([
  "POST /api/live/start",
  "POST /api/live/action",
  "POST /api/live/seek",
  "POST /api/live/quit",
  "POST /api/saves/load",
  "POST /api/live/start-faithful",
  "POST /api/live/run-mode",
  // NOT `POST /api/advisor`: it may come before a lazy resume, when the
  // engine has no run and a persist would DELETE the stored one. It persists
  // only when a run is live (below).
  "POST /api/live/run-mode-step",
  "POST /api/live/faithful-step",
  "POST /api/live/play-to-end",
  "POST /api/live/autobattle/advance",
  "POST /api/live/autobattle/step",
  "POST /api/live/wpa-verify",
  "POST /api/live/from_replay",
  // Replaces the live run with the imported position — must be re-persisted so
  // a reload resumes it (preview touches nothing, so it stays out).
  "POST /api/live/import-state",
]);

/** A stored run log. */
interface StoredRun {
  text: string;
  size: number;
  /** Seconds since the epoch, like `st_mtime`. */
  mtime: number;
}

/** What `/api/runs/put` accepts as a key: a filename, no path parts. */
const RUN_FILENAME = /^[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl$/;

function pathOf(body: unknown): string | null {
  const p = (body as { path?: unknown } | null)?.path;
  return typeof p === "string" ? p : null;
}

/** Routes before which the character's champion must be in the engine
 *  (they create or restore the run whose nodes get scored), keyed by how
 *  the character is found. */
const PPO_BEFORE_START = new Set(["POST /api/live/start", "POST /api/live/start-faithful"]);
/** Routes that replace the live run. The engine quits the run they replace
 *  the way `/api/live/quit` does, writing its log stamped with the `now` the
 *  body carries, as the Python server's `LiveRuns.replace` keeps it. */
const REPLACES_RUN = new Set([
  "POST /api/live/start",
  "POST /api/live/start-faithful",
  "POST /api/live/import-state",
  "POST /api/live/from_replay",
]);

/** The status the engine answers when the e2e brain has no net yet. */
export const NO_E2E_NET = 424;

/** Whether a call needs the e2e net now (so it is fetched first). Not
 *  `POST /api/advisor`: the page re-posts a stored preference at boot, and
 *  the net is fetched by the first advice call that needs it (its 424). */
function selectsE2e(key: string, body: unknown): boolean {
  const b = (body ?? {}) as { e2e?: unknown; off_combat_brain?: unknown; advisor?: unknown };
  if (key === "POST /api/e2e/selfplay" || key === "POST /api/live/e2e-advice") return true;
  if (key === "POST /api/seedcheck/log") return true;
  if (key === "POST /api/live/start-faithful") return b.e2e === true;
  if (key === "POST /api/live/run-mode") return b.off_combat_brain === "e2e" || b.advisor === "e2e";
  return false;
}

function characterOf(body: unknown): string {
  const c = (body as { character?: unknown } | null)?.character;
  return typeof c === "string" && c ? c : "Ironclad";
}

export const LIVE_KEY = "current";

/** Save-name → IDB key: lowercase, runs of non-alphanumerics → "-", trimmed,
 *  "save" if nothing is left (what the Python side did for file names). */
export function slugify(name: string): string {
  const s = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return s || "save";
}

/** ISO-8601 to the second, e.g. `2026-09-02T10:11:12Z` (no millis). */
export function isoNow(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

function parseReply(json: string): Reply {
  const r = JSON.parse(json) as Partial<Reply> | null;
  if (!r || typeof r.status !== "number") {
    throw new Error(`malformed engine reply: ${json.slice(0, 200)}`);
  }
  return { status: r.status, body: r.body };
}

function detail(status: number, msg: string): Reply {
  return { status, body: { detail: msg } };
}

function nameOf(body: unknown): string | null {
  const n = (body as { name?: unknown } | null)?.name;
  return typeof n === "string" ? n : null;
}

export function createEngine(deps: EngineDeps): Engine {
  const { wb, kv, saveHeader, championFile, fetchChampion, e2eFile, fetchE2e } = deps;
  const now = deps.now ?? isoNow;

  /** Make sure `character`'s champion is loaded; never throws — a failure
   *  is recorded on the engine (the panel shows it) and the run goes on
   *  with the heuristic brain. */
  async function ensurePpo(character: string): Promise<void> {
    if (!championFile || !fetchChampion) return;
    if (wb.has_ppo(character)) return;
    const named = parseReply(championFile(character));
    if (named.status !== 200 || typeof named.body !== "string") {
      wb.set_ppo_error(character, String((named.body as { detail?: string } | null)?.detail ?? "no champion"));
      return;
    }
    try {
      const bytes = await fetchChampion(named.body);
      const r = parseReply(wb.load_ppo(character, bytes));
      if (r.status !== 200) {
        wb.set_ppo_error(character, String((r.body as { detail?: string } | null)?.detail ?? "load failed"));
      }
    } catch (err) {
      wb.set_ppo_error(character, `fetch of /ppo/${named.body} failed: ${String(err)}`);
    }
  }

  /** Make sure the e2e student is loaded; resolves to whether it is. Never
   *  throws — a failure is recorded on the engine (the brain's 424 says
   *  why) and the next e2e call tries again. */
  async function ensureE2e(): Promise<boolean> {
    if (!e2eFile || !fetchE2e || !wb.load_e2e || !wb.has_e2e) return false;
    if (wb.has_e2e()) return true;
    try {
      const bytes = await fetchE2e(e2eFile);
      const r = parseReply(wb.load_e2e(bytes));
      if (r.status !== 200) {
        wb.set_e2e_error?.(String((r.body as { detail?: string } | null)?.detail ?? "load failed"));
      }
    } catch (err) {
      wb.set_e2e_error?.(`fetch of /e2e/${e2eFile} failed: ${String(err)}`);
    }
    return wb.has_e2e();
  }

  async function persist(): Promise<void> {
    const raw = wb.export_persist();
    if (raw.trim() === "null" || raw === "") await kv.delete("live", LIVE_KEY);
    else await kv.put("live", LIVE_KEY, raw);
  }

  async function resumeIfPersisted(onProgress?: ProgressCb): Promise<Reply | null> {
    const raw = await kv.get<string>("live", LIVE_KEY);
    if (raw == null) return null;
    let character = "Ironclad";
    try {
      character = characterOf(JSON.parse(raw));
    } catch {
      /* a malformed persist fails in import_persist below */
    }
    await ensurePpo(character);
    return parseReply(wb.import_persist(raw, onProgress ?? (() => {})));
  }

  async function save(body: unknown): Promise<Reply> {
    const name = nameOf(body);
    if (name == null) return detail(400, "missing save name");
    const r = parseReply(wb.build_save(name, now()));
    if (r.status !== 200) return r;
    const payload = r.body as { step_count?: number };
    await kv.put("saves", slugify(name), payload);
    return { status: 200, body: { name, path: null, step_count: payload.step_count } };
  }

  async function list(): Promise<Reply> {
    const payloads = await kv.getAll<unknown>("saves");
    const saves: { saved_at?: string }[] = [];
    for (const p of payloads) {
      const r = parseReply(saveHeader(JSON.stringify(p)));
      if (r.status === 200) saves.push(r.body as { saved_at?: string });
    }
    saves.sort((a, b) => (b.saved_at ?? "").localeCompare(a.saved_at ?? ""));
    return { status: 200, body: { saves } };
  }

  async function load(body: unknown): Promise<Reply> {
    const name = nameOf(body);
    if (name == null) return detail(400, "missing save name");
    const payload = await kv.get<unknown>("saves", slugify(name));
    if (payload === undefined) return detail(404, `unknown save: ${name}`);
    await ensurePpo(characterOf(payload));
    const reply = parseReply(wb.restore_save(JSON.stringify(payload), now()));
    if (reply.status === 200) await storeRunLogs();
    return reply;
  }

  // ── run logs ─────────────────────────────────────────────────────────

  /** After a route that can end a run (a quit, or one that replaces the
   *  live run): store every log the engine wrote. A name already taken gets
   *  `_2`, `_3`, … as the Python server's `game_log.GameLogger` does on disk
   *  (320ed156f), so two runs of the same seed in the same second keep both
   *  logs instead of the second overwriting the first. Returns the name the
   *  last log was stored under, or null when none was written. */
  async function storeRunLogs(): Promise<string | null> {
    let last: string | null = null;
    for (let raw = wb.take_run_log(); raw.trim() !== "null" && raw !== ""; raw = wb.take_run_log()) {
      const log = JSON.parse(raw) as { filename: string; text: string };
      last = await storeLog(log.filename, log.text);
    }
    return last;
  }

  /** Store one log under `filename`, suffixed `_2`, … on a clash; returns
   *  the name it was stored under. */
  async function storeLog(filename: string, text: string): Promise<string> {
    const stored: StoredRun = { text, size: new TextEncoder().encode(text).length, mtime: Date.now() / 1000 };
    const taken = new Set(await kv.keys("runs"));
    const stem = filename.replace(/\.jsonl$/, "");
    let name = filename;
    for (let n = 2; taken.has(name); n++) name = `${stem}_${n}.jsonl`;
    await kv.put("runs", name, stored);
    return name;
  }

  /** `GET /api/runs`: every stored log, newest first (mtime, then name). */
  async function listRuns(): Promise<Reply> {
    const keys = await kv.keys("runs");
    const rows: { path: string; mtime: number }[] = [];
    for (const k of keys) {
      const r = await kv.get<StoredRun>("runs", k);
      rows.push({ path: k, mtime: r?.mtime ?? 0 });
    }
    rows.sort((a, b) => b.mtime - a.mtime || (b.path < a.path ? -1 : b.path > a.path ? 1 : 0));
    return { status: 200, body: { runs: rows.map((r) => r.path) } };
  }

  async function runStat(route: string): Promise<Reply> {
    const q = route.split("?")[1] ?? "";
    const path = new URLSearchParams(q).get("path");
    if (!path) return detail(400, "path outside logs dir: ");
    const r = await kv.get<StoredRun>("runs", path);
    if (r === undefined) return detail(404, `file not found: ${path}`);
    return { status: 200, body: { size: r.size, mtime: r.mtime } };
  }

  async function loadRun(body: unknown): Promise<Reply> {
    const path = pathOf(body);
    if (path == null) return detail(422, "path must be a string");
    const r = await kv.get<StoredRun>("runs", path);
    if (r === undefined) return detail(404, `file not found: ${path}`);
    return parseReply(wb.load_run(path, r.text));
  }

  /** `GET /api/runs/read?path=`: a stored log, text and all — what a
   *  replay-mode bug report carries so the run can be reopened elsewhere. */
  async function readRun(route: string): Promise<Reply> {
    const q = route.split("?")[1] ?? "";
    const path = new URLSearchParams(q).get("path");
    if (!path) return detail(400, "path outside logs dir: ");
    const r = await kv.get<StoredRun>("runs", path);
    if (r === undefined) return detail(404, `file not found: ${path}`);
    return { status: 200, body: { path, text: r.text, size: r.size, mtime: r.mtime } };
  }

  /** `POST /api/runs/put {path, text}`: store a log under a bare filename
   *  (a report's run log, loaded on the reader's side), so `/api/runs`
   *  lists it and `/api/load` reopens it like one this browser produced. */
  async function putRun(body: unknown): Promise<Reply> {
    const path = pathOf(body);
    const text = (body as { text?: unknown } | null)?.text;
    if (path == null || !RUN_FILENAME.test(path)) return detail(422, "path must be a bare .jsonl filename");
    if (typeof text !== "string" || !text.trim()) return detail(422, "text must be a non-empty string");
    const stored: StoredRun = { text, size: new TextEncoder().encode(text).length, mtime: Date.now() / 1000 };
    await kv.put("runs", path, stored);
    return { status: 200, body: { path, size: stored.size } };
  }

  // ── the live tree, in and out (bug reports) ──────────────────────────

  /** `GET /api/live/persist`: the live run's tree as `export_persist` emits
   *  it — the same payload a reload resumes from, so a reader can stand
   *  exactly where the reporter stood. */
  function livePersist(): Reply {
    const raw = wb.export_persist();
    if (raw.trim() === "null" || raw === "") return detail(400, "no live run");
    return { status: 200, body: { persist: JSON.parse(raw) } };
  }

  /** `POST /api/live/import {persist}`: replace the live run with a
   *  persisted tree (progress frames as the active path replays), and
   *  persist it so the next reload resumes THIS run. */
  async function liveImport(body: unknown, onProgress?: ProgressCb): Promise<Reply> {
    const p = (body as { persist?: unknown } | null)?.persist;
    if (p == null || typeof p !== "object") return detail(422, "persist must be an object");
    const r = parseReply(wb.import_persist(JSON.stringify(p), onProgress ?? (() => {})));
    if (r.status === 200) await persist();
    return r;
  }

  // ── SimLab sweeps ────────────────────────────────────────────────────

  /** Stored sweeps are imported once, before the first route that reads
   *  the registry, so a reload's Studies history is what it was. */
  let sweepsImported: Promise<void> | null = null;
  function importStoredSweeps(): Promise<void> {
    if (!wb.import_sweep) return Promise.resolve();
    sweepsImported ??= (async () => {
      const docs = await kv.getAll<unknown>("sweeps");
      for (const doc of docs) wb.import_sweep!(JSON.stringify(doc));
    })();
    return sweepsImported;
  }

  async function storeSweep(id: string): Promise<void> {
    if (!wb.export_sweep) return;
    const raw = wb.export_sweep(id);
    if (raw.trim() === "null" || raw === "") return;
    await kv.put("sweeps", id, JSON.parse(raw));
  }

  /** The running sweep is checkpointed every few seconds so a reload
   *  mid-sweep keeps the trials played so far (as a cancelled sweep). */
  let lastCheckpoint = 0;
  const CHECKPOINT_MS = 5000;

  async function sweepTick(budgetMs = 50): Promise<boolean> {
    if (!wb.sweep_tick) return false;
    await importStoredSweeps();
    const t0 = Date.now();
    let active = true;
    const finished: string[] = [];
    // At least one trial per tick, then as many as the budget allows.
    do {
      const r = JSON.parse(wb.sweep_tick(1)) as { active: boolean; finished: string[] };
      active = r.active;
      finished.push(...r.finished);
    } while (active && Date.now() - t0 < budgetMs);
    for (const id of finished) await storeSweep(id);
    if (active && Date.now() - lastCheckpoint > CHECKPOINT_MS) {
      lastCheckpoint = Date.now();
      const list = parseReply(wb.call("GET", "/api/sweeps", "null"));
      const running = ((list.body as { sweeps?: { sweep_id: string; status: string }[] } | null)?.sweeps ?? []).filter(
        (s) => s.status === "running",
      );
      for (const s of running) await storeSweep(s.sweep_id);
    }
    return active;
  }

  async function allLogs(): Promise<string> {
    const keys = await kv.keys("runs");
    const pairs: [string, string][] = [];
    for (const k of keys) {
      const r = await kv.get<StoredRun>("runs", k);
      if (r) pairs.push([k, r.text]);
    }
    return JSON.stringify(pairs);
  }

  return {
    async handle(method, route, body = null, onProgress, onFrame) {
      const m = method.toUpperCase();
      const path = route.split("?")[0];
      const key = `${m} ${path}`;
      let reply: Reply;
      if (STREAM_ROUTES.has(key)) {
        const bodyJson = body == null ? "null" : JSON.stringify(body);
        reply = parseReply(
          wb.call_stream(m, path, bodyJson, (frame) => onFrame?.(JSON.parse(frame))),
        );
      } else if (key === "POST /api/live/save") reply = await save(body);
      else if (key === "GET /api/saves") reply = await list();
      else if (key === "POST /api/saves/load") reply = await load(body);
      else if (key === "GET /api/runs") reply = await listRuns();
      else if (key === "GET /api/runs/stat") reply = await runStat(route);
      else if (key === "POST /api/load") reply = await loadRun(body);
      else if (key === "GET /api/runs/read") reply = await readRun(route);
      else if (key === "POST /api/runs/put") reply = await putRun(body);
      else if (key === "GET /api/live/persist") reply = livePersist();
      else if (key === "POST /api/live/import") reply = await liveImport(body, onProgress);
      else if (key === "GET /api/mcts-runs") {
        reply = deps.mctsRuns ? parseReply(deps.mctsRuns(await allLogs())) : { status: 200, body: { runs: [] } };
      } else if (key === "GET /api/mcts-runs/aggregate") {
        reply = deps.aggregateRuns
          ? parseReply(deps.aggregateRuns(await allLogs()))
          : { status: 200, body: { n_runs: 0, n_losses: 0, final_turn_histogram: {}, turn1_q: { min: null, max: null, mean: null }, pattern_tags: {}, representative_losses: [] } };
      } else if (key === "POST /api/live/quit") {
        // The engine stamps the log with the host's clock; collect what it wrote.
        reply = parseReply(wb.call(m, path, JSON.stringify({ ...(body as object | null ?? {}), now: now() })));
        if (reply.status === 200) {
          // The reply names the log by the name it is stored under, which a
          // clash suffixed.
          const stored = await storeRunLogs();
          const out = reply.body as { log_path?: string | null } | null;
          if (stored && out?.log_path) out.log_path = stored;
        }
      } else if (key === "POST /api/seedcheck/log") {
        // The engine writes the attempt's log; the worker keeps it with the
        // other run logs, so the replay viewer lists and opens it.
        await ensureE2e();
        reply = parseReply(wb.call(m, path, JSON.stringify({ ...((body as object | null) ?? {}), now: now() })));
        if (reply.status === 200) {
          const log = reply.body as { path: string; text: string };
          reply = { status: 200, body: { path: await storeLog(log.path, log.text) } };
        }
      } else if (key.startsWith("POST /api/sweeps") || key.startsWith("GET /api/sweeps")) {
        if (!WASM_ROUTES.has(key) && !SWEEP_ROUTE.test(path)) return detail(501, `not available in the browser build: ${key}`);
        await importStoredSweeps();
        reply = parseReply(wb.call(m, route, body == null ? "null" : JSON.stringify(body)));
        // A stop is a status change worth keeping; the tick stores the end.
        if (key === "POST /api/sweeps" && reply.status === 200) lastCheckpoint = Date.now();
      } else if (WASM_ROUTES.has(key) || STEP_ROUTE.test(path)) {
        if (PPO_BEFORE_START.has(key)) await ensurePpo(characterOf(body));
        // A run that started offline gets its net on the next panel fetch.
        else if (key === "POST /api/live/ppo-policy" && wb.has_live_run()) {
          const st = parseReply(wb.call("GET", "/api/live/state", "null"));
          const character = (st.body as { state?: { player?: { name?: string } } } | null)?.state?.player?.name;
          if (typeof character === "string" && character) await ensurePpo(character);
        }
        const resumed =
          key === "GET /api/live/state" && !wb.has_live_run()
            ? await resumeIfPersisted(onProgress)
            : null;
        const triedE2e = selectsE2e(key, body);
        if (triedE2e) await ensureE2e();
        const sent = REPLACES_RUN.has(key) ? { ...((body as object | null) ?? {}), now: now() } : body;
        const send = () => parseReply(wb.call(m, path, sent == null ? "null" : JSON.stringify(sent)));
        reply = resumed ?? send();
        // An e2e run restored or resumed before its net arrived: fetch it
        // now and ask again, once (not when this call already tried).
        if (!resumed && !triedE2e && reply.status === NO_E2E_NET && !wb.has_e2e?.() && (await ensureE2e())) reply = send();
        if (REPLACES_RUN.has(key) && reply.status === 200) await storeRunLogs();
      } else {
        return detail(501, `not available in the browser build: ${key}`);
      }
      if (MUTATING.has(key) && reply.status === 200) await persist();
      // The live run's advisor, kept so a reload resumes it with the one the
      // page shows.
      else if (key === "POST /api/advisor" && reply.status === 200 && wb.has_live_run()) await persist();
      return reply;
    },
    sweepTick,
    hasRunningSweep: () => wb.has_running_sweep?.() ?? false,
  };
}
