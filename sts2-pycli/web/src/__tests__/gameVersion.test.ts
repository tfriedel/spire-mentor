import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { buildBrowserDump } from "../bugreport";
import {
  LEGACY_GAME_VERSION,
  declaredGameVersion,
  effectiveGameVersion,
  gameVersionBody,
  gameVersionLabel,
  payloadGameVersion,
  runLogGameVersion,
  versionedPublicPath,
} from "../gameVersion";

afterEach(() => vi.unstubAllGlobals());

describe("game version helpers (GH #801)", () => {
  it("a missing key reads as the default", () => {
    expect(LEGACY_GAME_VERSION).toBe("v0.107.1");
    expect(payloadGameVersion({})).toBe("v0.107.1");
    expect(payloadGameVersion(null)).toBe("v0.107.1");
    expect(declaredGameVersion({ idx: 1 })).toBeNull();
    expect(payloadGameVersion({ game_version: "v0.111.0" })).toBe("v0.111.0");
    expect(payloadGameVersion({ meta: { game_version: "v0.111.0" } })).toBe("v0.111.0");
  });

  it("reads a live / replay Step's version off its state, the shape both servers send", () => {
    // Rust `Game::serialize` and Python `rust_game` write the key onto the
    // game-state dict, which is served as `Step.state`.
    const step = { idx: 3, decision: "map_select", state: { floor: 2, game_version: "v0.111.0" } };
    expect(declaredGameVersion(step)).toBe("v0.111.0");
    expect(payloadGameVersion({ idx: 3, state: { floor: 2 } })).toBe("v0.107.1");
    expect(declaredGameVersion({ idx: 3, state: null })).toBeNull();
  });

  it("reads a run log's head meta record", () => {
    const meta = JSON.stringify({ step: 0, ts: 1, type: "meta", data: { character: "Silent", game_version: "v0.111.0" } });
    const act = JSON.stringify({ step: 1, ts: 2, type: "action", data: {} });
    expect(runLogGameVersion(`${meta}\n${act}\n`)).toBe("v0.111.0");
    expect(runLogGameVersion(JSON.stringify({ type: "meta", data: { character: "Silent" } }))).toBeNull();
    expect(runLogGameVersion("not json\n")).toBeNull();
    expect(runLogGameVersion(null)).toBeNull();
  });

  it("every named build goes on the wire, the legacy one included", () => {
    expect(gameVersionBody("v0.107.1")).toEqual({ game_version: "v0.107.1" });
    expect(gameVersionBody(undefined)).toEqual({});
    expect(gameVersionBody("v0.111.0")).toEqual({ game_version: "v0.111.0" });
  });

  it("a remembered pick the server cannot run falls back without being lost", () => {
    expect(effectiveGameVersion("v0.111.0", ["v0.107.1", "v0.111.0"])).toBe("v0.111.0");
    expect(effectiveGameVersion("v0.111.0", ["v0.107.1"])).toBe("v0.107.1");
    expect(effectiveGameVersion("bogus", ["v0.107.1", "v0.111.0"], "v0.111.0")).toBe("v0.111.0");
  });

  it("labels the known builds", () => {
    expect(gameVersionLabel("v0.107.1")).toBe("game v0.107.1 (stable)");
    expect(gameVersionLabel("v0.111.0")).toBe("v0.111.0 (beta)");
    expect(gameVersionLabel("v9")).toBe("v9");
  });
});

describe("api.live.importState", () => {
  function captureBodies() {
    const bodies: unknown[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ import: { warnings: [] } }), {
        status: 200, headers: { "content-type": "application/json" },
      });
    }));
    return bodies;
  }

  it("sends game_version whenever a build is named", async () => {
    const bodies = captureBodies();
    await api.live.importState("{}", "run");
    await api.live.importState("{}", "run", "v0.107.1");
    await api.live.importState("{}", "combat", "v0.111.0");
    expect(bodies).toEqual([
      { payload: "{}", lane: "run" },
      { payload: "{}", lane: "run", game_version: "v0.107.1" },
      { payload: "{}", lane: "combat", game_version: "v0.111.0" },
    ]);
  });
});

describe("bug report engine_build", () => {
  const jsonRes = (body: unknown) =>
    new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  const identity = { origin: "https://wb.test", userAgent: "UA/1", buildId: "deadbeef" };

  function liveApi(persist: Record<string, unknown>) {
    return async (input: string): Promise<Response> => {
      const path = input.split("?")[0];
      if (path === "/api/live/state") return jsonRes({ idx: 0, state: { floor: 1 } });
      if (path === "/api/live/persist") return jsonRes({ persist });
      return jsonRes(null);
    };
  }

  it("is the live session's game_version", async () => {
    const d = await buildBrowserDump(
      { mode: "live" }, { api: liveApi({ character: "Silent", game_version: "v0.111.0" }), ...identity });
    expect(d.meta.engine_build).toBe("v0.111.0");
  });

  it("falls back to v0.107.1 when the session names none", async () => {
    const d = await buildBrowserDump({ mode: "live" }, { api: liveApi({ character: "Silent" }), ...identity });
    expect(d.meta.engine_build).toBe("v0.107.1");
  });

  function replayApi(stepState: Record<string, unknown>, logText: string | null) {
    return async (input: string): Promise<Response> => {
      const path = input.split("?")[0];
      if (path === "/api/step/4") return jsonRes({ idx: 4, state: stepState });
      if (path === "/api/runs/read" && logText !== null) return jsonRes({ text: logText });
      return jsonRes(null);
    };
  }

  it("in replay is the step state's game_version", async () => {
    const d = await buildBrowserDump(
      { mode: "replay", step: 4 },
      { api: replayApi({ floor: 3, game_version: "v0.111.0" }, null), ...identity },
    );
    expect(d.meta.engine_build).toBe("v0.111.0");
  });

  it("in replay falls back to the run log's head meta, then the default", async () => {
    const head = JSON.stringify({ step: 0, ts: 0, type: "meta", data: { game_version: "v0.111.0" } });
    const d = await buildBrowserDump(
      { mode: "replay", step: 4, run_path: "runs/a.jsonl" },
      { api: replayApi({ floor: 3 }, `${head}\n`), ...identity },
    );
    expect(d.meta.engine_build).toBe("v0.111.0");
    const plain = await buildBrowserDump(
      { mode: "replay", step: 4, run_path: "runs/a.jsonl" },
      { api: replayApi({ floor: 3 }, `${JSON.stringify({ type: "meta", data: {} })}\n`), ...identity },
    );
    expect(plain.meta.engine_build).toBe("v0.107.1");
  });
});

describe("versionedPublicPath (GH #877)", () => {
  it("keeps the default build's table where it always was", () => {
    expect(versionedPublicPath("sfx_map.json", null)).toBe("/sfx_map.json");
    expect(versionedPublicPath("sfx_map.json", LEGACY_GAME_VERSION)).toBe("/sfx_map.json");
  });
  it("puts another build's under its version dir", () => {
    expect(versionedPublicPath("card_text.json", "v0.111.0")).toBe("/v0.111.0/card_text.json");
  });
});
