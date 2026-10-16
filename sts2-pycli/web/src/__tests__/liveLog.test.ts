import { beforeEach, describe, expect, it, vi } from "vitest";

const bridgeFetch = vi.fn();
const apiFetch = vi.fn();
vi.mock("../bridge", () => ({ bridgeFetch: (...a: unknown[]) => bridgeFetch(...a) }));
vi.mock("../transport", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));

import { mirrorLiveLog, resetLiveLogMirror } from "../liveLog";

const json = (body: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(body), { status }));

describe("liveLog.ts — Follow for the bundled workbench", () => {
  let log = { name: "20261016_IRONCLAD_S1.jsonl", size: 12, mtime: 100, text: '{"step":0}\n' };
  beforeEach(() => {
    resetLiveLogMirror();
    bridgeFetch.mockReset();
    apiFetch.mockReset();
    log = { name: "20261016_IRONCLAD_S1.jsonl", size: 12, mtime: 100, text: '{"step":0}\n' };
    bridgeFetch.mockImplementation((path: string) =>
      path === "/log?stat=1" ? json({ name: log.name, size: log.size, mtime: log.mtime }) : json(log),
    );
    apiFetch.mockImplementation(() => json({ path: log.name, size: log.size }));
  });

  it("puts the advisor's log into the run store, then only again when it changed", async () => {
    expect(await mirrorLiveLog()).toBe(log.name);
    expect(apiFetch).toHaveBeenCalledTimes(1);
    const [route, init] = apiFetch.mock.calls[0] as [string, RequestInit];
    expect(route).toBe("/api/runs/put");
    expect(JSON.parse(init.body as string)).toEqual({ path: log.name, text: log.text });

    expect(await mirrorLiveLog()).toBe(log.name); // unchanged: a stat, no transfer
    expect(apiFetch).toHaveBeenCalledTimes(1);
    expect(bridgeFetch).toHaveBeenCalledTimes(3);

    log = { ...log, size: 24, mtime: 101, text: '{"step":0}\n{"step":1}\n' };
    expect(await mirrorLiveLog()).toBe(log.name);
    expect(apiFetch).toHaveBeenCalledTimes(2);
  });

  it("is null while the advisor has no log, or only a torn first line", async () => {
    bridgeFetch.mockImplementationOnce(() => json({ error: "no_log" }, 404));
    expect(await mirrorLiveLog()).toBeNull();
    log = { ...log, text: "" };
    expect(await mirrorLiveLog()).toBeNull();
    expect(apiFetch).not.toHaveBeenCalled();
  });
});
