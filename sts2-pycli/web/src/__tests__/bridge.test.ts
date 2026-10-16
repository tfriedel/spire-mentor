import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { bridgeBase, bridgeFetch, DEFAULT_BRIDGE_URL, normalizeBridgeUrl, setBridgeBase } from "../bridge";

describe("bridge.ts — the browser build's control-plane client (GH #709)", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("normalises what the user types to an http(s) origin", () => {
    expect(normalizeBridgeUrl("127.0.0.1:8079")).toBe("http://127.0.0.1:8079");
    expect(normalizeBridgeUrl(" http://localhost:8079/ ")).toBe("http://localhost:8079");
    expect(normalizeBridgeUrl("https://pc.lan:8079/status")).toBe("https://pc.lan:8079");
    expect(normalizeBridgeUrl("")).toBeNull();
    expect(normalizeBridgeUrl("ws://x")).toBeNull();
    expect(normalizeBridgeUrl("not a url at all")).toBeNull();
    expect(DEFAULT_BRIDGE_URL).toBe("http://127.0.0.1:8079");
  });

  it("is off by default and remembers a configured origin", () => {
    expect(bridgeBase()).toBeNull();
    setBridgeBase("http://127.0.0.1:8079");
    expect(bridgeBase()).toBe("http://127.0.0.1:8079");
    setBridgeBase(null);
    expect(bridgeBase()).toBeNull();
  });

  it("fetches the control plane directly, with Chrome's loopback hint, and refuses when unconfigured", async () => {
    await expect(bridgeFetch("/status")).rejects.toThrow(/no live-game bridge/);
    setBridgeBase("http://127.0.0.1:8079");
    const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));
    await bridgeFetch("/restore", { method: "POST", body: "{}" });
    expect(spy).toHaveBeenCalledTimes(1);
    const [url, init] = spy.mock.calls[0] as [string, RequestInit & { targetAddressSpace?: string }];
    expect(url).toBe("http://127.0.0.1:8079/restore");
    expect(init.method).toBe("POST");
    expect(init.targetAddressSpace).toBe("loopback");
  });

  describe("the build the mod bundles (VITE_BRIDGE_SAME_ORIGIN=1)", () => {
    beforeEach(() => vi.stubEnv("VITE_BRIDGE_SAME_ORIGIN", "1"));
    afterEach(() => vi.unstubAllEnvs());

    it("connects to its own origin by default, and stays off once disconnected", () => {
      expect(bridgeBase()).toBe(window.location.origin);
      setBridgeBase(null);
      expect(bridgeBase()).toBeNull();
      setBridgeBase("http://pc.lan:8079");
      expect(bridgeBase()).toBe("http://pc.lan:8079");
    });

    it("sends no loopback hint to its own origin", async () => {
      const spy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));
      await bridgeFetch("/status");
      const [url, init] = spy.mock.calls[0] as [string, (RequestInit & { targetAddressSpace?: string }) | undefined];
      expect(url).toBe(`${window.location.origin}/status`);
      expect(init?.targetAddressSpace).toBeUndefined();
    });
  });
});
