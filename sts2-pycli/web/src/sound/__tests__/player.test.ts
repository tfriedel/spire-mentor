import { describe, it, expect, afterEach, vi } from "vitest";
import { SoundPlayer, type AudioBackend } from "../player";

function fake() {
  const loads: string[] = [];
  const plays: { gain: number; rate: number }[] = [];
  let voices = 0;
  const backend: AudioBackend = {
    async load(url) { loads.push(url); return { url }; },
    play(_b, gain, rate) { plays.push({ gain, rate }); voices += 1; },
    activeVoices() { return voices; },
  };
  return { backend, loads, plays, setVoices: (n: number) => { voices = n; } };
}

const CUE = { event: "event:/sfx/block_gain", cls: "informative" as const };

describe("SoundPlayer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads a file once and reuses the buffer", async () => {
    const f = fake();
    const p = new SoundPlayer(f.backend);
    await p.play(CUE, 1, 0);
    await p.play(CUE, 1, 1000);
    expect(f.loads).toHaveLength(1);
    expect(f.plays).toHaveLength(2);
  });

  it("debounces the same event inside the window", async () => {
    const f = fake();
    const p = new SoundPlayer(f.backend, { debounceMs: 60 });
    await p.play(CUE, 1, 0);
    await p.play(CUE, 1, 30);
    expect(f.plays).toHaveLength(1);
  });

  it("plays the same event again once the window has passed", async () => {
    const f = fake();
    const p = new SoundPlayer(f.backend, { debounceMs: 60 });
    await p.play(CUE, 1, 0);
    await p.play(CUE, 1, 100);
    expect(f.plays).toHaveLength(2);
  });

  it("drops a cue when the voice cap is reached", async () => {
    const f = fake();
    const p = new SoundPlayer(f.backend, { maxVoices: 4 });
    f.setVoices(4);
    await p.play(CUE, 1, 0);
    expect(f.plays).toHaveLength(0);
  });

  it("multiplies the cue gain by the user volume", async () => {
    const f = fake();
    const p = new SoundPlayer(f.backend);
    await p.play({ ...CUE, gain: 0.5 }, 0.4, 0);
    expect(f.plays[0].gain).toBeCloseTo(0.2);
  });

  it("plays at rate 1, with no draw, when the cue asks for no pitch variance", async () => {
    // `GetRandomPitchScale` returns 1 without touching the RNG for
    // `PitchVariance.None`; every FMOD cue and most debug ones take this path.
    const f = fake();
    const rand = vi.fn(() => 0.9);
    const p = new SoundPlayer(f.backend, { rand });
    await p.play(CUE, 1, 0);
    expect(f.plays[0].rate).toBe(1);
    expect(rand).not.toHaveBeenCalled();
  });

  it("draws the playback rate uniformly from 1 ± pitchVariance", async () => {
    // Both ends of the draw, so a sign error or a missing `* 2 - 1` (which
    // would only ever raise the pitch) fails.
    const f = fake();
    const draws = [0, 0.999999];
    const p = new SoundPlayer(f.backend, { rand: () => draws.shift()!, debounceMs: 0 });
    await p.play({ ...CUE, pitchVariance: 0.1 }, 1, 0);
    await p.play({ ...CUE, pitchVariance: 0.1 }, 1, 1);
    expect(f.plays[0].rate).toBeCloseTo(0.9);
    expect(f.plays[1].rate).toBeCloseTo(1.1);
  });

  it("swallows a failed load rather than breaking the frame", async () => {
    const p = new SoundPlayer({
      async load() { throw new Error("404"); },
      play() { throw new Error("should not play"); },
      activeVoices() { return 0; },
    });
    await expect(p.play(CUE, 1, 0)).resolves.toBeUndefined();
  });

  it("warns when a load fails, instead of failing silently", async () => {
    // A systematic regression (wrong asset URL, bad encode, broken node
    // graph) must not degrade to permanent, undiagnosable silence — nothing
    // in devtools, nothing in the workbench's bug-report capture.
    // Mutation caught: deleting the `console.warn` call in the catch.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const p = new SoundPlayer({
      async load() { throw new Error("404"); },
      play() { throw new Error("should not play"); },
      activeVoices() { return 0; },
    });
    await p.play(CUE, 1, 0);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain(CUE.event);
  });

  it("retries a failed load on a later call, instead of pinning the rejection", async () => {
    // The `catch` in player.ts deletes the cache entry precisely so a later
    // attempt can succeed. Earlier in this feature, `loadSfxMap` used
    // `cached ??=`, which pins a REJECTED promise forever (a rejection is
    // neither null nor undefined, so `??=` never reassigns it) — one
    // transient fetch blip silenced sound for the rest of the session.
    // Mutation caught: removing `this.buffers.delete(cue.event)` from the
    // catch block — the second `play` below would then still reuse the
    // failed promise and never reach `backend.play`.
    let attempt = 0;
    const plays: { gain: number }[] = [];
    const backend: AudioBackend = {
      async load(url) {
        attempt += 1;
        if (attempt === 1) throw new Error("network blip");
        return { url };
      },
      play(_b, gain) { plays.push({ gain }); },
      activeVoices() { return 0; },
    };
    const p = new SoundPlayer(backend, { debounceMs: 60 });

    await p.play(CUE, 1, 0);
    expect(plays).toHaveLength(0);

    // Advance `now` well past the debounce window so the retry isn't
    // rejected by the debounce instead of actually re-attempting the load.
    await p.play(CUE, 1, 1000);
    expect(attempt).toBe(2);
    expect(plays).toHaveLength(1);
  });

  it("does not start a debounce window for a cue dropped by the voice cap", async () => {
    // The cap check runs BEFORE `lastPlayed.set(...)` on purpose: a cue that
    // never played must not suppress the next attempt at (or near) the same
    // `now`. Mutation caught: moving `lastPlayed.set` above the cap check (or
    // otherwise stamping it on a cap-dropped call) — the second `play` below
    // would then be swallowed by the debounce instead of playing.
    const f = fake();
    const p = new SoundPlayer(f.backend, { maxVoices: 4, debounceMs: 60 });
    f.setVoices(4);
    await p.play(CUE, 1, 0);
    expect(f.plays).toHaveLength(0);

    f.setVoices(0);
    await p.play(CUE, 1, 1);
    expect(f.plays).toHaveLength(1);
  });
});
