/**
 * Bug reports in the browser-only workbench (static mode).
 *
 * The Python workbench builds a report server-side (`workbench/bugreport.py`)
 * and puts it in GCS. There is no server here — the engine is in a Web
 * Worker — so the PAGE builds the same schema-1 dump from what the worker
 * exposes and posts it to the host's Pages Function (`functions/api/
 * bugreport.ts`), which stamps `meta.id` / `meta.captured_at` and stores it
 * in Workers KV. `?report=<id>` then opens it read-only, like a Python one.
 *
 * What a browser report carries that a Python one does not: for a live run
 * the persisted tree (`GET /api/live/persist`, the payload a reload resumes
 * from) and for a replayed run the log text itself (`GET /api/runs/read`),
 * because neither exists anywhere but in the reporter's IndexedDB. The
 * reader's workbench takes them back with `loadReportIntoWorkbench` — that
 * is how a report on a phone becomes a run on the desk.
 *
 * Two more kinds of report reach the same store, through its signed-out
 * intake `/intake/bugreport` (docs/browser-workbench.md, "Bug reports"):
 * the Spire Mentor mod's in-game "Report a bug"
 * (`meta.source === "mod"`, its position in `state.game_state`), and the
 * mod-bundled workbench's, forwarded by the advisor's control plane with the
 * game's latest position added as `state.live_game`. Either position is
 * what the mod's Copy state yields, so `importReportPosition` hands it to
 * the state import.
 *
 * `api.captureBugReport` / `api.fetchBugReport` delegate here in static mode.
 * Whether the host HAS the Function is probed once (`bugReportEndpointAvailable`):
 * Cloudflare Pages answers the probe with JSON, a plain static host / `vite preview`
 * answer HTML or 404, and the SPA shows the 🐞 Report button only when it
 * would work.
 */
import type { BugReportDump } from "./api";
import { LEGACY_GAME_VERSION, declaredGameVersion, gameVersionBody, runLogGameVersion } from "./gameVersion";
import { apiFetch } from "./transport";

/** The engine build a report is stamped with when its session names none
 *  (mirrors `workbench/bugreport.py`'s constant). Since GH #801 the stamp is
 *  the session's own `game_version`; this is what a missing key means. */
export const ENGINE_BUILD = LEGACY_GAME_VERSION;
export const BUGREPORT_ENDPOINT = "/api/bugreport";

export interface CaptureRequest {
  mode: string;
  step?: number | null;
  screen?: string | null;
  overlay?: string | null;
  note?: string | null;
  run_path?: string | null;
}

/** Injection seams (tests): the worker-bound API fetch, the network fetch,
 *  and the identity fields the dump stamps. */
export interface BugReportDeps {
  api?: typeof apiFetch;
  fetch?: typeof globalThis.fetch;
  origin?: string;
  userAgent?: string;
  buildId?: string;
}

function resolve(deps: BugReportDeps) {
  return {
    api: deps.api ?? apiFetch,
    fetch: deps.fetch ?? ((input: RequestInfo | URL, init?: RequestInit) => globalThis.fetch(input, init)),
    origin: deps.origin ?? (typeof location !== "undefined" ? location.origin : ""),
    userAgent: deps.userAgent ?? (typeof navigator !== "undefined" ? navigator.userAgent : ""),
    buildId: deps.buildId ?? import.meta.env?.VITE_BUILD_ID ?? "dev",
  };
}

async function failure(res: Response): Promise<Error> {
  const text = await res.text().catch(() => "");
  let msg = text;
  try {
    const d = (JSON.parse(text) as { detail?: unknown } | null)?.detail;
    if (typeof d === "string") msg = d;
  } catch {
    /* not JSON */
  }
  return new Error(`${res.status} ${msg}`.trim());
}

/** One worker call, JSON in and out; throws with the engine's `detail`. */
async function engine<T>(api: typeof apiFetch, method: string, route: string, body?: unknown): Promise<T> {
  const res = await api(
    route,
    body === undefined
      ? { method }
      : { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) },
  );
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

/** Build the schema-1 dump for the moment `req` describes. Live: the
 *  enriched step, the map and the persisted tree. Replay: the viewed step
 *  and the run log's text. Throws "nothing to capture" when the mode's
 *  state is absent, like the Python side. */
export async function buildBrowserDump(req: CaptureRequest, deps: BugReportDeps = {}): Promise<BugReportDump> {
  const d = resolve(deps);
  const ui = { screen: req.screen ?? null, overlay: req.overlay ?? null };
  let meta: Record<string, unknown>;
  let repro: Record<string, unknown>;
  let state: Record<string, unknown>;
  let engineBuild: string;
  if (req.mode === "live") {
    let step: { state?: unknown };
    try {
      step = await engine<{ state?: unknown }>(d.api, "GET", "/api/live/state");
    } catch {
      throw new Error("nothing to capture");
    }
    const { persist } = await engine<{ persist: Record<string, unknown> }>(d.api, "GET", "/api/live/persist");
    const map = await engine<unknown>(d.api, "GET", "/api/live/map").catch(() => null);
    meta = {
      mode: "live",
      character: persist.character ?? null,
      seed: persist.seed ?? null,
      ascension: persist.ascension ?? null,
      backend: "wasm",
    };
    repro = { persist };
    state = { game_state: step.state ?? step, map };
    engineBuild = declaredGameVersion(persist) ?? declaredGameVersion(step) ?? LEGACY_GAME_VERSION;
  } else if (req.mode === "replay") {
    const idx = req.step;
    if (idx == null || idx < 0) throw new Error("nothing to capture");
    let step: { state?: unknown };
    try {
      step = await engine<{ state?: unknown }>(d.api, "GET", `/api/step/${idx}`);
    } catch {
      throw new Error("nothing to capture");
    }
    let log: { text: string } | null = null;
    if (req.run_path) {
      log = await engine<{ text: string }>(d.api, "GET", `/api/runs/read?path=${encodeURIComponent(req.run_path)}`).catch(
        () => null,
      );
    }
    const gs = step.state ?? step;
    const player = (gs as { player?: { name?: unknown } } | null)?.player;
    meta = {
      mode: "replay",
      character: typeof player?.name === "string" ? player.name : null,
      seed: null,
      ascension: null,
      backend: null,
    };
    repro = { run_file: req.run_path ?? null, line: idx, run_log: log?.text ?? null };
    state = { game_state: gs, map: null };
    // The step's state names a non-default build; the run log's head meta is
    // the fallback when it does not. Absent from both means v0.107.1.
    engineBuild = declaredGameVersion(step) ?? runLogGameVersion(log?.text) ?? LEGACY_GAME_VERSION;
  } else {
    throw new Error(`unknown mode: ${req.mode}`);
  }
  meta.engine_build = engineBuild;
  meta.workbench_commit = d.buildId;
  meta.host = d.origin;
  meta.user_agent = d.userAgent;
  return { schema_version: 1, meta, repro, state, ui, note: req.note ?? null };
}

/** Build the dump and post it; resolves to the id and the shareable URL. */
export async function captureBrowserBugReport(
  req: CaptureRequest,
  deps: BugReportDeps = {},
): Promise<{ id: string; url: string; detail?: string }> {
  const d = resolve(deps);
  const dump = await buildBrowserDump(req, deps);
  const res = await d.fetch(BUGREPORT_ENDPOINT, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(dump),
  });
  if (!res.ok) throw await failure(res);
  return (await res.json()) as { id: string; url: string; detail?: string };
}

export async function fetchBrowserBugReport(id: string, deps: BugReportDeps = {}): Promise<BugReportDump> {
  const d = resolve(deps);
  const res = await d.fetch(`${BUGREPORT_ENDPOINT}/${encodeURIComponent(id)}`);
  if (!res.ok) throw await failure(res);
  return (await res.json()) as BugReportDump;
}

let probe: Promise<boolean> | null = null;

/** Does this host run the bug-report Function? Probed once per page load;
 *  a host that answers HTML (a static host's SPA fallback), 404 (`vite preview`)
 *  or nothing at all (offline) is "no", and the button stays hidden. */
export function bugReportEndpointAvailable(deps: BugReportDeps = {}): Promise<boolean> {
  probe ??= (async () => {
    try {
      const res = await resolve(deps).fetch(BUGREPORT_ENDPOINT, { cache: "no-store" });
      if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) return false;
      const body = (await res.json()) as { ok?: unknown } | null;
      return body?.ok === true;
    } catch {
      return false;
    }
  })();
  return probe;
}

/** Tests: forget the probe's answer. */
export function resetBugReportProbe(): void {
  probe = null;
}

export type LoadedReport = { kind: "live" } | { kind: "replay"; path: string };

/** What `loadReportIntoWorkbench` would do with this dump, or null. */
export function reportLoadKind(dump: BugReportDump): "live" | "replay" | null {
  const r = dump.repro;
  if (r.persist && typeof r.persist === "object") return "live";
  if (typeof r.run_log === "string" && r.run_log.trim()) return "replay";
  return null;
}

/** The filename a report's run log is stored under on the reader's side:
 *  the report id, then the original name — recognisable in the Runs list
 *  and never colliding with the reader's own logs. */
export function reportRunFilename(dump: BugReportDump): string {
  const id = String(dump.meta.id ?? "wb-report");
  const raw = String(dump.repro.run_file ?? "run.jsonl").split(/[\\/]/).pop() ?? "run.jsonl";
  let base = raw.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._-]+/, "");
  if (!base.endsWith(".jsonl")) base = `${base.replace(/\.[^.]*$/, "") || "run"}.jsonl`;
  return `${id}_${base}`;
}

/** Take a report's run into THIS workbench: a live tree replaces the live
 *  run (and is persisted, so a reload resumes it); a run log lands in the
 *  Runs list under `reportRunFilename`. Throws when the dump carries neither. */
export async function loadReportIntoWorkbench(dump: BugReportDump, deps: BugReportDeps = {}): Promise<LoadedReport> {
  const d = resolve(deps);
  const kind = reportLoadKind(dump);
  if (kind === "live") {
    await engine(d.api, "POST", "/api/live/import", { persist: dump.repro.persist });
    return { kind: "live" };
  }
  if (kind === "replay") {
    const path = reportRunFilename(dump);
    await engine(d.api, "POST", "/api/runs/put", { path, text: dump.repro.run_log });
    return { kind: "replay", path };
  }
  throw new Error("this report carries no run to load");
}

/** The live-game position a report carries — the mod's Copy-state message
 *  (an `advice_request` or a `run_snapshot`) — or null. A mod report keeps
 *  it as its `state.game_state`; a report from the mod-bundled workbench has
 *  the game's latest one as `state.live_game` (the advisor adds it). */
export function reportGamePosition(dump: BugReportDump): Record<string, unknown> | null {
  const isPosition = (v: unknown): v is Record<string, unknown> => {
    if (!v || typeof v !== "object" || Array.isArray(v)) return false;
    const t = (v as { type?: unknown }).type;
    return t === "advice_request" || t === "run_snapshot";
  };
  const state = dump.state ?? {};
  if (dump.meta?.source === "mod" && isPosition(state.game_state)) return state.game_state;
  return isPosition(state.live_game) ? state.live_game : null;
}

/** Rebuild a report's game position as THIS workbench's live run (the
 *  state import, lane "auto": the exact fight when it is one). Resolves to
 *  the import's fidelity report; throws when the dump carries no position. */
export async function importReportPosition(dump: BugReportDump, deps: BugReportDeps = {}): Promise<unknown> {
  const pos = reportGamePosition(dump);
  if (!pos) throw new Error("this report carries no game position");
  const d = resolve(deps);
  const step = await engine<{ import?: unknown }>(d.api, "POST", "/api/live/import-state", {
    payload: JSON.stringify(pos),
    lane: "auto",
    ...gameVersionBody(declaredGameVersion(pos)),
  });
  return step.import ?? null;
}
