import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createWebAudioBackend } from "../webAudioBackend";

/**
 * Final review finding 12 (minor, 2026-08-23): this file had no tests "by
 * design" — the real WebAudio API is thin glue over jsdom-unsupported
 * browser primitives, and the decision-shaped logic (voice cap, debounce,
 * buffer caching) already lives in `player.ts`, which IS unit-tested. But
 * `voices -= 1` in `play()`'s catch is itself decision-shaped: removing it
 * left all 549 tests green, and it is the one thing standing between a
 * failed `start()` call and PERMANENT silence for the tab (enough failed
 * plays pin `voices` at `maxVoices`, and `player.ts`'s cap then refuses
 * every future cue). This file stays otherwise untested by design; only
 * that one line is pinned here, via a minimal fake AudioContext.
 */
function fakeAudioContext(startBehavior: "throws" | "succeeds", sources: { playbackRate: { value: number } }[] = []) {
  return class {
    // Already running: these tests are about the voice count, not the
    // autoplay seam below, so no gesture listeners get hooked.
    state = "running";
    destination = {};
    createBufferSource() {
      const src = {
        buffer: null as unknown,
        playbackRate: { value: 1 },
        onended: null as (() => void) | null,
        connect() {
          return this;
        },
        start() {
          if (startBehavior === "throws") throw new Error("boom");
        },
      };
      sources.push(src);
      return src;
    }
    createGain() {
      return {
        gain: { value: 0 },
        connect() {
          return this;
        },
      };
    }
  };
}

describe("createWebAudioBackend", () => {
  let originalAudioContext: typeof AudioContext | undefined;

  beforeEach(() => {
    originalAudioContext = globalThis.AudioContext;
  });

  afterEach(() => {
    globalThis.AudioContext = originalAudioContext as typeof AudioContext;
    vi.restoreAllMocks();
  });

  it("unwinds the voice count when start() throws synchronously, instead of pinning it forever", () => {
    // Mutation caught: removing `voices -= 1` from the catch block.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    // @ts-expect-error jsdom has no AudioContext; a minimal throwing stub.
    globalThis.AudioContext = fakeAudioContext("throws");

    const backend = createWebAudioBackend();
    backend.play({}, 0.5, 1);

    expect(backend.activeVoices()).toBe(0);
    expect(warn).toHaveBeenCalled();
  });

  it("counts a voice while it plays and releases it via onended when start() succeeds", () => {
    // Companion to the above: pins the non-error path this test file
    // otherwise never exercises, so the throw-path assertion above isn't
    // trivially true (e.g. `voices` never incrementing at all).
    // @ts-expect-error jsdom has no AudioContext; a minimal succeeding stub.
    globalThis.AudioContext = fakeAudioContext("succeeds");

    const backend = createWebAudioBackend();
    backend.play({}, 0.5, 1);
    expect(backend.activeVoices()).toBe(1);
  });

  it("applies the requested playback rate to the source (the pitch-variance draw)", () => {
    // Mutation caught: dropping the `playbackRate` assignment, which would
    // leave every pitch-varied sound at one fixed pitch again.
    const sources: { playbackRate: { value: number } }[] = [];
    // @ts-expect-error jsdom has no AudioContext; a minimal succeeding stub.
    globalThis.AudioContext = fakeAudioContext("succeeds", sources);

    createWebAudioBackend().play({}, 0.5, 1.07);
    expect(sources[0].playbackRate.value).toBeCloseTo(1.07);
  });
});

/**
 * The autoplay-policy seam. Sound is on by default, so the AudioContext is
 * usually constructed at page load with no gesture behind it and comes back
 * `suspended`; a suspended context plays nothing and never un-suspends on
 * its own. These pin the two ways out: the first gesture on the page, and a
 * play that finds it still suspended.
 */
class SuspendedContext {
  state: "suspended" | "running" = "suspended";
  resume = vi.fn(async () => { this.state = "running"; });
  destination = {};
  createBufferSource() {
    return {
      connect: () => ({ connect: () => {} }), start: () => {}, buffer: null, onended: null,
      playbackRate: { value: 1 },
    };
  }
  createGain() { return { gain: { value: 0 }, connect: () => ({}) }; }
}

describe("createWebAudioBackend: the suspended context", () => {
  let ctx: SuspendedContext;
  const original = globalThis.AudioContext;
  beforeEach(() => {
    ctx = new SuspendedContext();
    // @ts-expect-error jsdom has no AudioContext
    globalThis.AudioContext = class { constructor() { return ctx; } };
  });
  afterEach(() => { globalThis.AudioContext = original; });

  it("resumes on the first gesture, then stops listening", async () => {
    createWebAudioBackend();
    expect(ctx.resume).not.toHaveBeenCalled();
    window.dispatchEvent(new Event("pointerdown"));
    expect(ctx.resume).toHaveBeenCalledTimes(1);
    await Promise.resolve();
    expect(ctx.state).toBe("running");
    window.dispatchEvent(new Event("keydown"));
    expect(ctx.resume).toHaveBeenCalledTimes(1);
  });

  it("a play on a still-suspended context asks for a resume itself", () => {
    const backend = createWebAudioBackend();
    backend.play({}, 0.5, 1);
    expect(ctx.resume).toHaveBeenCalledTimes(1);
  });

  it("hooks nothing when the context is already running", () => {
    ctx.state = "running";
    createWebAudioBackend();
    window.dispatchEvent(new Event("pointerdown"));
    expect(ctx.resume).not.toHaveBeenCalled();
  });
});
