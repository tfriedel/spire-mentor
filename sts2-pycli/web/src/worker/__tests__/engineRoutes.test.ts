import { describe, expect, it, vi } from "vitest";
import { createEngine, isoNow, slugify, LIVE_KEY, type WorkbenchLike } from "../engineRoutes";
import { memoryKv } from "../idb";

const reply = (status: number, body: unknown) => JSON.stringify({ status, body });

/** A `Workbench` double with just enough state to exercise the router:
 *  a live run that start/quit/restore/import toggle, and a call log. */
class FakeWorkbench implements WorkbenchLike {
  live: Record<string, unknown> | null = null;
  calls: string[] = [];
  progressCalls = 0;
  /** PPO champions loaded / errors recorded (M3). */
  nets = new Map<string, number>();
  ppoErrors = new Map<string, string>();
  call(method: string, route: string, body_json: string): string {
    this.calls.push(`${method} ${route} ${body_json}`);
    const sweep = this.sweepRoute(method, route, body_json);
    if (sweep != null) return sweep;
    switch (route) {
      case "/api/live/start":
      case "/api/live/start-faithful":
        this.quitReplaced(JSON.parse(body_json).now);
        this.live = { seed: JSON.parse(body_json).seed, steps: 0, character: JSON.parse(body_json).character ?? "Ironclad" };
        return reply(200, { idx: 0 });
      case "/api/live/from_replay":
        this.quitReplaced(JSON.parse(body_json).now);
        this.live = { resumed: JSON.parse(body_json).idx, steps: 0 };
        return reply(200, { idx: 0 });
      case "/api/live/run-mode-step":
      case "/api/live/autobattle/step":
        if (!this.live) return reply(400, { detail: "no live run" });
        this.live = { ...this.live, steps: (this.live.steps as number) + 1 };
        return reply(200, { step: { idx: this.live.steps }, done: false });
      case "/api/live/state":
        return this.live ? reply(200, { idx: this.live.steps }) : reply(400, { detail: "no live run" });
      case "/api/live/action":
        if (!this.live) return reply(400, { detail: "no live run" });
        this.live = { ...this.live, steps: (this.live.steps as number) + 1 };
        return reply(200, { idx: this.live.steps });
      case "/api/live/quit": {
        const had = this.live != null;
        this.live = null;
        const now = JSON.parse(body_json || "{}").now as string | undefined;
        if (had) {
          const filename = this.logName(now);
          this.pendingLogs.push({ filename, text: '{"type":"meta"}\n{"type":"state"}\n' });
          return reply(200, { ok: true, log_path: filename });
        }
        return reply(400, { detail: "no live run" });
      }
      default:
        return reply(200, { route });
    }
  }
  call_stream(method: string, route: string, body_json: string, on_frame: (frame: string) => void): string {
    this.calls.push(`${method} ${route} ${body_json} (stream)`);
    on_frame(JSON.stringify({ type: "start", total: 1 }));
    on_frame(JSON.stringify({ type: "tick", done: 1, total: 1 }));
    on_frame(JSON.stringify(this.live ? { type: "result", candidates: [] } : { type: "error", error: "no live run" }));
    return reply(200, null);
  }
  build_save(name: string, saved_at: string): string {
    if (!this.live) return reply(400, { detail: "no live run" });
    return reply(200, { name, saved_at, step_count: this.live.steps, line: [1, 2, 3] });
  }
  restore_save(payload_json: string, now: string): string {
    const p = JSON.parse(payload_json);
    this.quitReplaced(now);
    this.live = { restored: p.name, steps: p.step_count };
    return reply(200, { step: { idx: p.step_count } });
  }
  export_persist(): string {
    return JSON.stringify(this.live);
  }
  import_persist(payload_json: string, progress: (done: number, total: number) => void): string {
    progress(1, 2);
    progress(2, 2);
    this.progressCalls += 2;
    this.live = JSON.parse(payload_json);
    return reply(200, { idx: this.live!.steps, resumed: true });
  }
  has_live_run(): boolean {
    return this.live != null;
  }
  /** SimLab sweeps (GH #707): a registry of jobs with `left` trials each;
   *  a tick plays one trial of the oldest running job. */
  sweeps = new Map<string, { spec: unknown; left: number; done: number; status: string; imported?: boolean }>();
  sweepCalls: string[] = [];
  private sweepSeq = 0;
  private sweepRoute(method: string, route: string, body_json: string): string | null {
    const [path, query] = route.split("?");
    if (method === "POST" && path === "/api/sweeps") {
      const spec = JSON.parse(body_json) as { n_seeds: number };
      const id = `sweep-${++this.sweepSeq}`;
      this.sweeps.set(id, { spec, left: spec.n_seeds, done: 0, status: "running" });
      return reply(200, { sweep_id: id });
    }
    if (method === "GET" && path === "/api/sweeps") {
      return reply(200, { sweeps: [...this.sweeps].map(([sweep_id, j]) => ({ sweep_id, status: j.status, done: j.done })) });
    }
    const m = /^\/api\/sweeps\/([^/]+)(?:\/(stop|results))?$/.exec(path);
    if (!m) return null;
    const job = this.sweeps.get(m[1]);
    if (!job) return reply(404, { detail: m[1] });
    if (m[2] === "stop") {
      if (job.status !== "running") return reply(404, { detail: `${m[1]} not running` });
      job.status = "cancelled";
      return reply(200, { status: "stopping" });
    }
    if (m[2] === "results") return reply(200, { buckets: {}, errors_total: 0, query: query ?? "" });
    return reply(200, { status: job.status, done: job.done });
  }
  sweep_tick(max_trials: number): string {
    this.sweepCalls.push(`tick ${max_trials}`);
    const finished: string[] = [];
    const running = [...this.sweeps].find(([, j]) => j.status === "running");
    if (running) {
      const [id, job] = running;
      job.left -= 1;
      job.done += 1;
      if (job.left <= 0) {
        job.status = "done";
        finished.push(id);
      }
    }
    return JSON.stringify({ active: this.has_running_sweep(), finished });
  }
  has_running_sweep(): boolean {
    return [...this.sweeps.values()].some((j) => j.status === "running");
  }
  export_sweep(id: string): string {
    const job = this.sweeps.get(id);
    return job ? JSON.stringify({ sweep_id: id, spec: job.spec, rows: Array<number>(job.done).fill(1), progress: { status: job.status, done: job.done } }) : "null";
  }
  import_sweep(doc_json: string): string {
    const doc = JSON.parse(doc_json) as { sweep_id: string; spec: unknown; rows: unknown[]; progress: { status: string } };
    if (!this.sweeps.has(doc.sweep_id)) {
      this.sweeps.set(doc.sweep_id, { spec: doc.spec, left: 0, done: doc.rows.length, status: doc.progress.status === "done" ? "done" : "cancelled", imported: true });
    }
    return reply(200, { ok: true });
  }
  load_ppo(character: string, bytes: Uint8Array): string {
    if (bytes.length === 0) return reply(400, { detail: "empty model" });
    this.nets.set(character, bytes.length);
    this.ppoErrors.delete(character);
    return reply(200, { ok: true, character });
  }
  set_ppo_error(character: string, reason: string): void {
    this.ppoErrors.set(character, reason);
  }
  has_ppo(character: string): boolean {
    return this.nets.has(character);
  }
  /** Run logs (M4): quit leaves one log behind until collected. */
  pendingLogs: { filename: string; text: string }[] = [];
  logName(now: string | undefined): string {
    return `${(now ?? "2026-01-01T00:00:00Z").replace(/\D/g, "").slice(0, 14)}_Ironclad_S.jsonl`;
  }
  /** The engine quits a run a start / resume / restore replaces, leaving its log. */
  quitReplaced(now: string | undefined): void {
    if (this.live) this.pendingLogs.push({ filename: this.logName(now), text: '{"type":"meta"}\n' });
  }
  loadedRun: string | null = null;
  take_run_log(): string {
    const log = this.pendingLogs.shift();
    return log ? JSON.stringify(log) : "null";
  }
  load_run(path: string, text: string): string {
    this.calls.push(`load_run ${path}`);
    if (!text.startsWith("{")) return reply(422, { detail: "invalid JSONL: bad" });
    this.loadedRun = path;
    return reply(200, { count: text.split("\n").filter(Boolean).length, resume: { kind: "run", reason: null } });
  }
}

const championFile = (character: string) =>
  character === "Ironclad" || character === "Silent"
    ? reply(200, `ppo_${character.toLowerCase()}.onnx`)
    : reply(400, { detail: `unknown character: ${character}` });

const saveHeader = (payload_json: string) => {
  const p = JSON.parse(payload_json);
  return reply(200, { name: p.name, saved_at: p.saved_at, step_count: p.step_count });
};

function setup(opts: { now?: () => string; fetchChampion?: (file: string) => Promise<Uint8Array> } = {}) {
  const wb = new FakeWorkbench();
  const kv = memoryKv();
  const engine = createEngine({
    wb,
    kv,
    saveHeader,
    now: opts.now,
    ...(opts.fetchChampion ? { championFile, fetchChampion: opts.fetchChampion } : {}),
  });
  return { wb, kv, engine };
}

describe("slugify / isoNow", () => {
  it("lowercases, collapses non-alphanumerics to '-', trims, falls back to 'save'", () => {
    expect(slugify("My Run!")).toBe("my-run");
    expect(slugify("  Act 2 -- boss  ")).toBe("act-2-boss");
    expect(slugify("***")).toBe("save");
    expect(slugify("")).toBe("save");
  });
  it("isoNow has second precision and a Z suffix", () => {
    expect(isoNow()).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
  });
});

describe("engine routes — dispatch", () => {
  it("answers 501 for every route the browser build does not serve, without touching the wasm", async () => {
    const { wb, engine } = setup();
    const r = await engine.handle("GET", "/api/bridge/status");
    expect(r.status).toBe(501);
    expect((r.body as { detail: string }).detail).toMatch(/not available in the browser build: GET \/api\/bridge\/status/);
    expect(wb.calls).toEqual([]);
    expect((await engine.handle("GET", "/api/bridge/tree")).status).toBe(501);
  });

  it("forwards served routes to Workbench.call with the query stripped and the body as JSON", async () => {
    const { wb, engine } = setup({ now: () => "2026-09-03T04:22:57Z" });
    await engine.handle("get", "/api/capabilities?x=1");
    await engine.handle("POST", "/api/live/start", { seed: "ABC", character: "Ironclad" });
    expect(wb.calls).toEqual([
      "GET /api/capabilities null",
      // A run-replacing route carries the host's clock for the replaced run's log.
      'POST /api/live/start {"seed":"ABC","character":"Ironclad","now":"2026-09-03T04:22:57Z"}',
    ]);
  });

  it("passes the wasm's status/body through untouched", async () => {
    const { engine } = setup();
    expect(await engine.handle("GET", "/api/live/state")).toEqual({ status: 400, body: { detail: "no live run" } });
  });
});

describe("engine routes — saves over the kv store", () => {
  it("POST /api/live/save builds the save, stores it under the slug, and answers the Python shape", async () => {
    const now = vi.fn(() => "2026-09-02T10:00:00Z");
    const { wb, kv, engine } = setup({ now });
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    await engine.handle("POST", "/api/live/action", { action: "end_turn" });
    const r = await engine.handle("POST", "/api/live/save", { name: "My Run!" });
    expect(r).toEqual({ status: 200, body: { name: "My Run!", path: null, step_count: 1 } });
    expect(await kv.get("saves", "my-run")).toEqual({
      name: "My Run!", saved_at: "2026-09-02T10:00:00Z", step_count: 1, line: [1, 2, 3],
    });
    expect(wb.calls.filter((c) => c.includes("/api/live/save"))).toEqual([]);
  });

  it("a non-200 from build_save passes through and nothing is stored", async () => {
    const { kv, engine } = setup();
    const r = await engine.handle("POST", "/api/live/save", { name: "x" });
    expect(r).toEqual({ status: 400, body: { detail: "no live run" } });
    expect(await kv.getAll("saves")).toEqual([]);
    expect((await engine.handle("POST", "/api/live/save", {})).status).toBe(400);
  });

  it("GET /api/saves lists every stored save's header, newest first", async () => {
    const { kv, engine } = setup();
    await kv.put("saves", "old", { name: "old", saved_at: "2026-01-01T00:00:00Z", step_count: 2 });
    await kv.put("saves", "new", { name: "new", saved_at: "2026-09-01T00:00:00Z", step_count: 9 });
    await kv.put("saves", "mid", { name: "mid", saved_at: "2026-05-01T00:00:00Z", step_count: 5 });
    const r = await engine.handle("GET", "/api/saves");
    expect(r.status).toBe(200);
    expect((r.body as { saves: { name: string }[] }).saves.map((s) => s.name)).toEqual(["new", "mid", "old"]);
  });

  it("POST /api/saves/load restores the stored payload (404 when unknown) and persists the run", async () => {
    const { wb, kv, engine } = setup();
    expect(await engine.handle("POST", "/api/saves/load", { name: "nope" })).toEqual({
      status: 404, body: { detail: "unknown save: nope" },
    });
    await kv.put("saves", "my-run", { name: "My Run!", saved_at: "2026-09-02T10:00:00Z", step_count: 4 });
    const r = await engine.handle("POST", "/api/saves/load", { name: "My Run!" });
    expect(r).toEqual({ status: 200, body: { step: { idx: 4 } } });
    expect(wb.live).toEqual({ restored: "My Run!", steps: 4 });
    expect(await kv.get("live", LIVE_KEY)).toBe(JSON.stringify(wb.live));
  });
});

describe("engine routes — persisting the live run", () => {
  it("writes live/current after every successful mutating call and drops it after quit", async () => {
    const { kv, engine } = setup();
    expect(await kv.get("live", LIVE_KEY)).toBeUndefined();
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    expect(JSON.parse((await kv.get<string>("live", LIVE_KEY))!)).toEqual({ seed: "S", steps: 0, character: "Ironclad" });
    await engine.handle("GET", "/api/live/map"); // not mutating: unchanged
    await engine.handle("POST", "/api/live/action", { action: "end_turn" });
    expect(JSON.parse((await kv.get<string>("live", LIVE_KEY))!)).toEqual({ seed: "S", steps: 1, character: "Ironclad" });
    await engine.handle("POST", "/api/live/quit");
    expect(await kv.get("live", LIVE_KEY)).toBeUndefined();
  });

  it("a failed mutation leaves the persisted run alone", async () => {
    const { kv, engine } = setup();
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    await engine.handle("POST", "/api/live/quit");
    // quit again: the double 400 from the wasm must not resurrect anything
    const r = await engine.handle("POST", "/api/live/action", { action: "end_turn" });
    expect(r.status).toBe(400);
    expect(await kv.get("live", LIVE_KEY)).toBeUndefined();
  });

  it("GET /api/live/state with no run in memory imports the persisted one, streaming progress", async () => {
    const { wb, kv, engine } = setup();
    await kv.put("live", LIVE_KEY, '{"seed":"S","steps":6}');
    const frames: [number, number][] = [];
    const r = await engine.handle("GET", "/api/live/state", null, (d, t) => frames.push([d, t]));
    expect(r).toEqual({ status: 200, body: { idx: 6, resumed: true } });
    expect(frames).toEqual([[1, 2], [2, 2]]);
    expect(wb.calls).toEqual([]); // answered from import_persist, not call()
    // Now the run is in memory: the next state read is an ordinary call.
    await engine.handle("GET", "/api/live/state");
    expect(wb.calls).toEqual(["GET /api/live/state null"]);
    expect(wb.progressCalls).toBe(2);
  });

  it("GET /api/live/state with nothing persisted is the wasm's own answer", async () => {
    const { wb, engine } = setup();
    expect(await engine.handle("GET", "/api/live/state")).toEqual({ status: 400, body: { detail: "no live run" } });
    expect(wb.calls).toEqual(["GET /api/live/state null"]);
  });
});

describe("streaming advisor routes", () => {
  it("forwards every call_stream frame, parsed, and answers 200/null", async () => {
    const wb = new FakeWorkbench();
    const engine = createEngine({ wb, saveHeader: () => reply(200, {}), kv: memoryKv() });
    const frames: unknown[] = [];
    const r = await engine.handle(
      "POST",
      "/api/live/card-advisor/stream",
      { config: { rollouts: 1 } },
      undefined,
      (f) => frames.push(f),
    );
    expect(r).toEqual({ status: 200, body: null });
    expect(wb.calls[wb.calls.length - 1]).toBe('POST /api/live/card-advisor/stream {"config":{"rollouts":1}} (stream)');
    expect(frames).toEqual([
      { type: "start", total: 1 },
      { type: "tick", done: 1, total: 1 },
      { type: "error", error: "no live run" },
    ]);
  });

  it("the advisor and run-mode routes are served by the wasm; replay tools still answer 501", async () => {
    const wb = new FakeWorkbench();
    const engine = createEngine({ wb, saveHeader: () => reply(200, {}), kv: memoryKv() });
    for (const route of ["/api/live/hint", "/api/live/auto-action", "/api/live/regret-review", "/api/live/ppo-policy", "/api/live/wpa-review"]) {
      const r = await engine.handle("POST", route, { config: {} });
      expect(r.status).toBe(200);
      expect(r.body).toEqual({ route });
    }
    const r = await engine.handle("GET", "/api/bridge/status");
    expect(r.status).toBe(501);
  });
});

describe("engine routes — run mode + the PPO champion (M3)", () => {
  it("serves the run-mode routes through the wasm and persists after each", async () => {
    const { wb, kv, engine } = setup();
    await engine.handle("POST", "/api/live/start-faithful", { character: "Silent", seed: "run_1" });
    expect(wb.live).toMatchObject({ seed: "run_1", character: "Silent" });
    expect(JSON.parse((await kv.get<string>("live", LIVE_KEY))!)).toMatchObject({ steps: 0 });
    for (const route of ["/api/live/run-mode-step", "/api/live/autobattle/step"]) {
      const r = await engine.handle("POST", route);
      expect(r.status).toBe(200);
    }
    expect(JSON.parse((await kv.get<string>("live", LIVE_KEY))!)).toMatchObject({ steps: 2 });
    // Read-only run-mode routes reach the wasm too.
    for (const key of [
      "POST /api/live/run-mode",
      "GET /api/live/autobattle/verdict",
      "POST /api/live/wpa-verify",
      "POST /api/live/play-to-end",
    ]) {
      const [m, route] = key.split(" ");
      expect((await engine.handle(m, route, {})).status).toBe(200);
    }
    expect(wb.calls.some((c) => c.startsWith("GET /api/live/autobattle/verdict"))).toBe(true);
  });

  it("fetches the character's champion once before a start and hands the bytes to the engine", async () => {
    const fetched: string[] = [];
    const fetchChampion = vi.fn(async (file: string) => {
      fetched.push(file);
      return new Uint8Array([1, 2, 3]);
    });
    const { wb, engine } = setup({ fetchChampion });
    await engine.handle("POST", "/api/live/start", { character: "Silent", seed: "A" });
    expect(fetched).toEqual(["ppo_silent.onnx"]);
    expect(wb.has_ppo("Silent")).toBe(true);
    // Loaded once per character: a second start of the same character does
    // not refetch; another character does.
    await engine.handle("POST", "/api/live/start", { character: "Silent", seed: "B" });
    expect(fetchChampion).toHaveBeenCalledTimes(1);
    await engine.handle("POST", "/api/live/start-faithful", { character: "Ironclad", seed: "run_2" });
    expect(fetched).toEqual(["ppo_silent.onnx", "ppo_ironclad.onnx"]);
    // The engine sees the champion before the start (the root gets scored).
    const startIdx = wb.calls.findIndex((c) => c.startsWith("POST /api/live/start "));
    expect(startIdx).toBeGreaterThanOrEqual(0);
    expect(wb.nets.get("Silent")).toBe(3);
  });

  it("records a failed champion fetch on the engine and still starts the run", async () => {
    const fetchChampion = vi.fn(async () => {
      throw new Error("TypeError: Failed to fetch");
    });
    const { wb, engine } = setup({ fetchChampion });
    const r = await engine.handle("POST", "/api/live/start", { character: "Ironclad", seed: "A" });
    expect(r.status).toBe(200);
    expect(wb.has_ppo("Ironclad")).toBe(false);
    expect(wb.ppoErrors.get("Ironclad")).toMatch(/fetch of \/ppo\/ppo_ironclad.onnx failed: Error: TypeError: Failed to fetch/);
    // An unknown character never fetches; the engine's own reason is kept.
    await engine.handle("POST", "/api/live/start", { character: "Watcher", seed: "A" });
    expect(fetchChampion).toHaveBeenCalledTimes(1);
    expect(wb.ppoErrors.get("Watcher")).toBe("unknown character: Watcher");
    // A rejected load (bad bytes) is recorded too.
    const { wb: wb2, engine: e2 } = setup({ fetchChampion: async () => new Uint8Array([]) });
    await e2.handle("POST", "/api/live/start", { character: "Silent", seed: "A" });
    expect(wb2.ppoErrors.get("Silent")).toBe("empty model");
  });

  it("loads the champion before restoring a save or resuming a persisted run", async () => {
    const fetchChampion = vi.fn(async () => new Uint8Array([9]));
    const { wb, kv, engine } = setup({ fetchChampion });
    await kv.put("saves", "s1", { name: "s1", step_count: 3, character: "Silent" });
    await engine.handle("POST", "/api/saves/load", { name: "s1" });
    expect(wb.has_ppo("Silent")).toBe(true);
    expect(fetchChampion).toHaveBeenCalledTimes(1);
    // A persisted Ironclad run: resumed on the first state read, net first.
    const { wb: wb2, kv: kv2, engine: e2 } = setup({ fetchChampion });
    await kv2.put("live", LIVE_KEY, JSON.stringify({ steps: 4, character: "Ironclad" }));
    const r = await e2.handle("GET", "/api/live/state");
    expect(r.status).toBe(200);
    expect(wb2.has_ppo("Ironclad")).toBe(true);
    expect(fetchChampion).toHaveBeenCalledTimes(2);
  });

  it("without a champion source the routes still work and nothing is fetched", async () => {
    const { wb, engine } = setup();
    await engine.handle("POST", "/api/live/start", { character: "Silent", seed: "A" });
    expect(wb.has_ppo("Silent")).toBe(false);
    expect(wb.ppoErrors.size).toBe(0);
  });
});

describe("engine routes — run logs and the replay viewer (M4)", () => {
  it("stores the log a quit leaves behind, lists it newest first and reopens it", async () => {
    const now = () => "2026-09-03T04:22:57Z";
    const { wb, kv, engine } = setup({ now });
    expect((await engine.handle("GET", "/api/runs")).body).toEqual({ runs: [] });
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    const q = await engine.handle("POST", "/api/live/quit");
    expect(q.status).toBe(200);
    expect((q.body as { log_path: string }).log_path).toBe("20260903042257_Ironclad_S.jsonl");
    // The engine got the host's clock, the store the text.
    expect(wb.calls.some((c) => c.startsWith("POST /api/live/quit ") && c.includes('"now":"2026-09-03T04:22:57Z"'))).toBe(true);
    const keys = await kv.keys("runs");
    expect(keys).toEqual(["20260903042257_Ironclad_S.jsonl"]);
    expect(await kv.get("live", LIVE_KEY)).toBeUndefined();
    // A second run: two logs, newest first.
    await kv.put("runs", "older.jsonl", { text: "{}", size: 2, mtime: 1 });
    const runs = (await engine.handle("GET", "/api/runs")).body as { runs: string[] };
    expect(runs.runs).toEqual(["20260903042257_Ironclad_S.jsonl", "older.jsonl"]);
    const stat = await engine.handle("GET", "/api/runs/stat?path=older.jsonl");
    expect(stat).toEqual({ status: 200, body: { size: 2, mtime: 1 } });
    expect((await engine.handle("GET", "/api/runs/stat?path=missing.jsonl")).status).toBe(404);
    expect((await engine.handle("GET", "/api/runs/stat")).status).toBe(400);
    // Loading hands the stored text to the engine.
    const loaded = await engine.handle("POST", "/api/load", { path: "20260903042257_Ironclad_S.jsonl" });
    expect(loaded.status).toBe(200);
    expect((loaded.body as { count: number }).count).toBe(2);
    expect(wb.loadedRun).toBe("20260903042257_Ironclad_S.jsonl");
    expect((await engine.handle("POST", "/api/load", { path: "nope.jsonl" })).status).toBe(404);
    expect((await engine.handle("POST", "/api/load", {})).status).toBe(422);
    // The replay routes reach the wasm.
    for (const key of ["GET /api/timeline", "GET /api/map", "GET /api/coach", "GET /api/step/3", "GET /api/step/3/deck-analysis", "GET /api/deck/trend", "POST /api/regret-review", "POST /api/wpa-review", "POST /api/recommend"]) {
      const [m, route] = key.split(" ");
      const r = await engine.handle(m, route, {});
      expect(r.status).toBe(200);
      expect(r.body).toEqual({ route });
    }
    expect((await engine.handle("GET", "/api/step/x")).status).toBe(501);
  });

  it("keeps the log of a run another start replaces, and both of a name clash (320ed156f)", async () => {
    const now = () => "2026-09-03T04:22:57Z";
    const { wb, kv, engine } = setup({ now });
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    // Starting over quits the live run; the engine got the host's clock.
    await engine.handle("POST", "/api/live/start", { seed: "S" });
    expect(wb.calls.filter((c) => c.startsWith("POST /api/live/start ") && c.includes('"now":"2026-09-03T04:22:57Z"')).length).toBe(2);
    expect(await kv.keys("runs")).toEqual(["20260903042257_Ironclad_S.jsonl"]);
    // The quit of the second run, in the same second, clashes and is suffixed.
    const quit = await engine.handle("POST", "/api/live/quit");
    expect((quit.body as { log_path: string }).log_path).toBe("20260903042257_Ironclad_S_2.jsonl");
    expect((await kv.keys("runs")).sort()).toEqual([
      "20260903042257_Ironclad_S.jsonl",
      "20260903042257_Ironclad_S_2.jsonl",
    ]);
  });

  it("feeds every stored log to the MCTS Runs statics, or answers empty without them", async () => {
    const seen: string[] = [];
    const wb = new FakeWorkbench();
    const kv = memoryKv();
    await kv.put("runs", "a.jsonl", { text: "A", size: 1, mtime: 1 });
    await kv.put("runs", "b.jsonl", { text: "B", size: 1, mtime: 2 });
    const engine = createEngine({
      wb,
      kv,
      saveHeader,
      mctsRuns: (logs) => {
        seen.push(logs);
        return reply(200, { runs: [{ stem: "a" }] });
      },
      aggregateRuns: (logs) => {
        seen.push(logs);
        return reply(200, { n_runs: 2 });
      },
    });
    expect((await engine.handle("GET", "/api/mcts-runs")).body).toEqual({ runs: [{ stem: "a" }] });
    expect((await engine.handle("GET", "/api/mcts-runs/aggregate")).body).toEqual({ n_runs: 2 });
    expect(JSON.parse(seen[0])).toEqual([["a.jsonl", "A"], ["b.jsonl", "B"]]);
    const bare = createEngine({ wb: new FakeWorkbench(), kv: memoryKv(), saveHeader });
    expect((await bare.handle("GET", "/api/mcts-runs")).body).toEqual({ runs: [] });
    expect(((await bare.handle("GET", "/api/mcts-runs/aggregate")).body as { n_runs: number }).n_runs).toBe(0);
  });

  it("play-from-here replaces the live run and is persisted", async () => {
    const { kv, engine } = setup();
    const r = await engine.handle("POST", "/api/live/from_replay", { idx: 3 });
    expect(r.status).toBe(200);
    expect(await kv.get("live", LIVE_KEY)).toBeDefined();
  });
});

describe("engine routes — what a bug report carries (M5)", () => {
  it("hands out the live tree and takes one back, persisting it", async () => {
    const { wb, kv, engine } = setup();
    expect((await engine.handle("GET", "/api/live/persist")).status).toBe(400);
    await engine.handle("POST", "/api/live/start", { seed: "S", character: "Silent" });
    await engine.handle("POST", "/api/live/action", {});
    const out = await engine.handle("GET", "/api/live/persist");
    expect(out.status).toBe(200);
    const persist = (out.body as { persist: Record<string, unknown> }).persist;
    expect(persist).toEqual({ seed: "S", steps: 1, character: "Silent" });
    // Another browser: import replaces whatever run was there, with progress.
    const { wb: wb2, kv: kv2, engine: engine2 } = setup();
    await engine2.handle("POST", "/api/live/start", { seed: "OTHER" });
    const frames: [number, number][] = [];
    const r = await engine2.handle("POST", "/api/live/import", { persist }, (d, t) => frames.push([d, t]));
    expect(r.status).toBe(200);
    expect(r.body).toEqual({ idx: 1, resumed: true });
    expect(frames).toEqual([[1, 2], [2, 2]]);
    expect(wb2.live).toEqual(persist);
    expect(JSON.parse((await kv2.get<string>("live", LIVE_KEY))!)).toEqual(persist);
    expect((await engine2.handle("POST", "/api/live/import", {})).status).toBe(422);
    expect((await engine2.handle("POST", "/api/live/import", { persist: "x" })).status).toBe(422);
    void wb;
    void kv;
  });

  it("reads a stored run log out and puts one in under a bare filename", async () => {
    const { wb, kv, engine } = setup();
    await kv.put("runs", "a.jsonl", { text: '{"type":"meta"}\n', size: 16, mtime: 7 });
    const r = await engine.handle("GET", "/api/runs/read?path=a.jsonl");
    expect(r).toEqual({ status: 200, body: { path: "a.jsonl", text: '{"type":"meta"}\n', size: 16, mtime: 7 } });
    expect((await engine.handle("GET", "/api/runs/read?path=nope.jsonl")).status).toBe(404);
    expect((await engine.handle("GET", "/api/runs/read")).status).toBe(400);
    const put = await engine.handle("POST", "/api/runs/put", { path: "wb-0123456789abcdef_run.jsonl", text: '{"type":"meta"}\n{"type":"state"}\n' });
    expect(put.status).toBe(200);
    expect(put.body).toEqual({ path: "wb-0123456789abcdef_run.jsonl", size: 33 });
    const runs = (await engine.handle("GET", "/api/runs")).body as { runs: string[] };
    expect(runs.runs[0]).toBe("wb-0123456789abcdef_run.jsonl");
    const loaded = await engine.handle("POST", "/api/load", { path: "wb-0123456789abcdef_run.jsonl" });
    expect(loaded.status).toBe(200);
    expect(wb.loadedRun).toBe("wb-0123456789abcdef_run.jsonl");
    for (const bad of [{ path: "../x.jsonl", text: "{}" }, { path: "x.txt", text: "{}" }, { path: "ok.jsonl", text: "" }, { text: "{}" }]) {
      expect((await engine.handle("POST", "/api/runs/put", bad)).status).toBe(422);
    }
  });
});

describe("engine routes — SimLab sweeps (GH #707)", () => {
  it("submits through the wasm, then a tick plays trials until the budget or the end, storing what finished", async () => {
    const wb = new FakeWorkbench();
    const kv = memoryKv();
    const e = createEngine({ wb, kv, saveHeader: () => reply(200, {}) });
    const submitted = await e.handle("POST", "/api/sweeps", { name: "t", n_seeds: 3 });
    expect(submitted).toEqual({ status: 200, body: { sweep_id: "sweep-1" } });
    expect(e.hasRunningSweep()).toBe(true);
    // The status / stop / results routes reach the wasm with the query intact.
    expect(await e.handle("GET", "/api/sweeps/sweep-1")).toEqual({ status: 200, body: { status: "running", done: 0 } });
    expect((await e.handle("GET", "/api/sweeps/sweep-1/results?baseline=random")).body).toMatchObject({ query: "baseline=random" });
    expect((await e.handle("GET", "/api/sweeps/nope")).status).toBe(404);
    expect((await e.handle("POST", "/api/sweeps/sweep-1/bogus")).status).toBe(501);
    // A tick with a generous budget runs the job to its end and stores it.
    expect(await e.sweepTick(1000)).toBe(false);
    expect(wb.sweepCalls).toEqual(["tick 1", "tick 1", "tick 1"]);
    expect(await kv.get("sweeps", "sweep-1")).toMatchObject({ sweep_id: "sweep-1", progress: { status: "done", done: 3 } });
    expect(e.hasRunningSweep()).toBe(false);
    expect((await e.handle("POST", "/api/sweeps/sweep-1/stop")).status).toBe(404);
  });

  it("a stop is honoured by the next tick; a zero-budget tick still plays one trial", async () => {
    const wb = new FakeWorkbench();
    const kv = memoryKv();
    const e = createEngine({ wb, kv, saveHeader: () => reply(200, {}) });
    await e.handle("POST", "/api/sweeps", { name: "t", n_seeds: 50 });
    expect(await e.sweepTick(0)).toBe(true);
    expect(wb.sweepCalls).toHaveLength(1);
    expect(await e.handle("POST", "/api/sweeps/sweep-1/stop")).toEqual({ status: 200, body: { status: "stopping" } });
    expect(await e.sweepTick(0)).toBe(false);
  });

  it("imports the stored sweeps once, before the first listing, and never a live id twice", async () => {
    const kv = memoryKv();
    await kv.put("sweeps", "old-1", { sweep_id: "old-1", spec: { name: "old" }, rows: [1, 1], progress: { status: "done", done: 2 } });
    await kv.put("sweeps", "old-2", { sweep_id: "old-2", spec: { name: "cut" }, rows: [1], progress: { status: "running", done: 1 } });
    const wb = new FakeWorkbench();
    const e = createEngine({ wb, kv, saveHeader: () => reply(200, {}) });
    const listed = (await e.handle("GET", "/api/sweeps")).body as { sweeps: { sweep_id: string; status: string }[] };
    expect(listed.sweeps.map((s) => `${s.sweep_id}:${s.status}`)).toEqual(["old-1:done", "old-2:cancelled"]);
    await e.handle("GET", "/api/sweeps");
    expect(wb.sweeps.size).toBe(2);
    // No pump needed for what came back: nothing is running.
    expect(e.hasRunningSweep()).toBe(false);
  });

  it("without sweep support in the wasm the routes still go through and a tick is a no-op", async () => {
    const wb = new FakeWorkbench();
    const bare = Object.assign(Object.create(Object.getPrototypeOf(wb)), wb) as FakeWorkbench;
    // Older shell: none of the sweep methods.
    (bare as unknown as { sweep_tick?: unknown }).sweep_tick = undefined;
    (bare as unknown as { has_running_sweep?: unknown }).has_running_sweep = undefined;
    (bare as unknown as { export_sweep?: unknown }).export_sweep = undefined;
    (bare as unknown as { import_sweep?: unknown }).import_sweep = undefined;
    const e = createEngine({ wb: bare, kv: memoryKv(), saveHeader: () => reply(200, {}) });
    expect(await e.sweepTick()).toBe(false);
    expect(e.hasRunningSweep()).toBe(false);
  });
});

/** The e2e student on top of the double: the brain answers 424 until the
 *  net is loaded. */
class E2eWorkbench extends FakeWorkbench {
  e2e = 0;
  e2eError: string | null = null;
  call(method: string, route: string, body_json: string): string {
    if (
      ((route === "/api/live/run-mode-step" && this.live?.e2e) || route === "/api/live/e2e-advice")
      && !this.e2e
    ) {
      this.calls.push(`${method} ${route} ${body_json}`);
      return reply(424, { detail: `no e2e net loaded: ${this.e2eError ?? "not fetched"}` });
    }
    const r = super.call(method, route, body_json);
    if (route === "/api/live/start-faithful" && JSON.parse(body_json).e2e) this.live = { ...this.live, e2e: true };
    return r;
  }
  load_e2e(bytes: Uint8Array): string {
    if (bytes.length === 0) return reply(400, { detail: "not a weights file" });
    this.e2e = bytes.length;
    this.e2eError = null;
    return reply(200, { ok: true });
  }
  set_e2e_error(reason: string): void {
    this.e2eError = reason;
  }
  has_e2e(): boolean {
    return this.e2e > 0;
  }
}

function setupE2e(fetchE2e: (file: string) => Promise<Uint8Array>) {
  const wb = new E2eWorkbench();
  const kv = memoryKv();
  const engine = createEngine({ wb, kv, saveHeader, e2eFile: "student.e2ew", fetchE2e });
  return { wb, kv, engine };
}

describe("engine routes — the e2e student, fetched on demand", () => {
  it("fetches nothing until a call selects the e2e brain", async () => {
    const fetchE2e = vi.fn(async () => new Uint8Array([1, 2]));
    const { wb, engine } = setupE2e(fetchE2e);
    await engine.handle("POST", "/api/live/start-faithful", { seed: "run_1" });
    await engine.handle("POST", "/api/live/run-mode", { off_combat_brain: "ppo" });
    await engine.handle("POST", "/api/live/run-mode-step", {});
    expect(fetchE2e).not.toHaveBeenCalled();
    await engine.handle("POST", "/api/live/run-mode", { off_combat_brain: "e2e" });
    expect(fetchE2e).toHaveBeenCalledWith("student.e2ew");
    expect(wb.has_e2e()).toBe(true);
    // Loaded once.
    await engine.handle("POST", "/api/e2e/selfplay", { seed: 1 });
    await engine.handle("POST", "/api/live/start-faithful", { e2e: true, seed: "run_2" });
    expect(fetchE2e).toHaveBeenCalledTimes(1);
    expect(wb.calls.some((c) => c.startsWith("POST /api/e2e/selfplay"))).toBe(true);
  });

  it("choosing the e2e advisor, or asking for its advice, fetches the net first", async () => {
    const fetchE2e = vi.fn(async () => new Uint8Array([5]));
    const { wb, engine } = setupE2e(fetchE2e);
    await engine.handle("POST", "/api/live/start", { character: "Ironclad", seed: "X" });
    await engine.handle("POST", "/api/live/run-mode", { advisor: "search" });
    expect(fetchE2e).not.toHaveBeenCalled();
    await engine.handle("POST", "/api/live/run-mode", { advisor: "e2e" });
    expect(fetchE2e).toHaveBeenCalledTimes(1);
    expect(wb.has_e2e()).toBe(true);

    // The host's preference does NOT: the page re-posts it at every boot.
    const pref = setupE2e(fetchE2e);
    await pref.engine.handle("POST", "/api/advisor", { advisor: "e2e" });
    expect(fetchE2e).toHaveBeenCalledTimes(1);

    const fresh = setupE2e(fetchE2e);
    await fresh.engine.handle("POST", "/api/live/start", { character: "Ironclad", seed: "X" });
    await fresh.engine.handle("POST", "/api/live/e2e-advice", {});
    expect(fetchE2e).toHaveBeenCalledTimes(2);
    expect(fresh.wb.calls.some((c) => c.startsWith("POST /api/live/e2e-advice"))).toBe(true);
  });

  it("an e2e run with no net yet (a resumed one) fetches on the 424 and retries once", async () => {
    const fetchE2e = vi.fn(async () => new Uint8Array([7]));
    const { wb, kv, engine } = setupE2e(fetchE2e);
    await kv.put("live", LIVE_KEY, JSON.stringify({ steps: 2, character: "Ironclad", e2e: true }));
    await engine.handle("GET", "/api/live/state");
    expect(fetchE2e).not.toHaveBeenCalled();
    const r = await engine.handle("POST", "/api/live/run-mode-step", {});
    expect(r.status).toBe(200);
    expect(fetchE2e).toHaveBeenCalledTimes(1);
    expect(wb.calls.filter((c) => c.startsWith("POST /api/live/run-mode-step")).length).toBe(2);
  });

  it("a call that already tried the fetch does not fetch again on its 424", async () => {
    const fetchE2e = vi.fn(async (): Promise<Uint8Array> => {
      throw new Error("404 Not Found");
    });
    const { engine } = setupE2e(fetchE2e);
    await engine.handle("POST", "/api/live/start", { character: "Ironclad", seed: "X" });
    const r = await engine.handle("POST", "/api/live/e2e-advice", {});
    expect(r.status).toBe(424);
    expect(fetchE2e).toHaveBeenCalledTimes(1);
  });

  it("the advisor preference persists the live run, and only a live one", async () => {
    const { wb, kv, engine } = setupE2e(vi.fn(async () => new Uint8Array([1])));
    await kv.put("live", LIVE_KEY, JSON.stringify({ steps: 2, character: "Ironclad" }));
    // No live run yet (a lazy resume is pending): the stored run stays.
    await engine.handle("POST", "/api/advisor", { advisor: "search" });
    expect(await kv.get("live", LIVE_KEY)).not.toBeNull();
    await engine.handle("POST", "/api/live/start", { character: "Ironclad", seed: "X" });
    const before = wb.calls.length;
    await engine.handle("POST", "/api/advisor", { advisor: "search" });
    expect(wb.calls.length).toBeGreaterThan(before);
    expect(await kv.get("live", LIVE_KEY)).toBe(wb.export_persist());
  });

  it("a failed fetch is recorded, the 424 passes through, and the next call tries again", async () => {
    let fail = true;
    const fetchE2e = vi.fn(async () => {
      if (fail) throw new Error("404 Not Found");
      return new Uint8Array([3]);
    });
    const { wb, engine } = setupE2e(fetchE2e);
    await engine.handle("POST", "/api/live/start-faithful", { e2e: true, seed: "run_1" });
    expect(wb.e2eError).toMatch(/fetch of \/e2e\/student.e2ew failed: Error: 404 Not Found/);
    const r = await engine.handle("POST", "/api/live/run-mode-step", {});
    expect(r.status).toBe(424);
    expect((r.body as { detail: string }).detail).toMatch(/404 Not Found/);
    fail = false;
    expect((await engine.handle("POST", "/api/live/run-mode-step", {})).status).toBe(200);
    expect(fetchE2e).toHaveBeenCalledTimes(3);
  });
});
