import { describe, it, expect } from "vitest";
import { noiseDirection, turbulenceDisplacement } from "../turbulence";
import type { TurbulenceParams } from "../turbulence";

// vfx_attack_blunt's "specks" emitter — the real authored parameter set this
// task is transcribed against (see task-13-brief.md).
const P: TurbulenceParams = {
  strength: 5.0,
  scale: 10.1,
  speed: [0.1, 0.1, 0.1],
  speedRandom: 1.0,
  influenceMin: 0.65,
  influenceMax: 0.85,
  // Representative baked influence-over-life ramp (not extracted real bake
  // data — Task 13 ports the noise math, not the manifest's curve loader).
  influenceOverLifeLut: [0.1, 0.4, 0.75, 1.0],
};

describe("noiseDirection", () => {
  it("is deterministic in position and time", () => {
    expect(noiseDirection([1, 2, 0], 0.5, P)).toEqual(noiseDirection([1, 2, 0], 0.5, P));
  });

  it("returns a bounded direction", () => {
    for (let i = 0; i < 1000; i++) {
      const d = noiseDirection([i * 0.1, i * 0.3, 0], i * 0.01, P);
      expect(Math.hypot(...d)).toBeLessThanOrEqual(1.001);
    }
  });

  it("varies with position at the game's authored noise scale", () => {
    // vfx_attack_blunt's specks: noise_scale 10.1. Two points a
    // particle-width apart must not return the same direction, or the whole
    // field is constant and the specks fly parallel.
    const a = noiseDirection([0, 0, 0], 0, P);
    const b = noiseDirection([0.05, 0, 0], 0, P);
    expect(a).not.toEqual(b);
  });

  it("magnitude reflects the game's authored strength=5.0 extrapolation", () => {
    // GLSL mix() is unclamped: mix(0.9*dir, dir, strength - 9.0) at
    // strength=5.0 has t=-4.0, an extrapolation collapsing the result to
    // 0.5*dir, not a unit vector. A clamped-lerp implementation would give a
    // different (larger) magnitude here — this is the regression test for
    // that specific mistake.
    const d = noiseDirection([1, 2, 3], 0.25, P);
    const unitScaled = noiseDirection([1, 2, 3], 0.25, { ...P, strength: 9.9 });
    // At strength ~9.9, t ~= 0.9, close to the true curl unit vector, so its
    // magnitude should be visibly larger than the strength=5.0 case's 0.5x.
    expect(Math.hypot(...d)).toBeLessThan(Math.hypot(...unitScaled));
  });

  it("never produces NaN across a wide position/time sweep at the game's authored parameters", () => {
    // curl_3d's normalize() is unguarded (matches Godot — see module doc), so
    // a zero-length curl anywhere in this sweep would surface as NaN here.
    // 51,005 samples across a wide position/time grid at the authored
    // strength/scale/speed found none.
    //
    // The assertion is HOISTED out of the loop deliberately. Calling
    // `expect` 51,005 times costs far more than the noise math it is
    // checking, and put this test's runtime right at vitest's 5s limit — so
    // it passed or failed on machine load, which is a broken gate rather
    // than a flaky one. Recording the first bad sample and asserting once
    // keeps every sample and the diagnostic (the failure message names the
    // exact position and time), at a fraction of the cost.
    let sampled = 0;
    let firstBad: { pos: [number, number, number]; time: number; d: readonly number[] } | null = null;
    for (let ix = -50; ix <= 50; ix++) {
      for (let iy = -50; iy <= 50; iy++) {
        for (let it = 0; it < 5; it++) {
          const pos: [number, number, number] = [ix * 0.37, iy * 0.29, (ix - iy) * 0.13];
          const time = it * 3.7 + ix * 0.011;
          const d = noiseDirection(pos, time, P);
          sampled++;
          if (firstBad === null && !d.every((c) => Number.isFinite(c))) {
            firstBad = { pos, time, d };
          }
        }
      }
    }
    expect(firstBad, `non-finite direction at ${JSON.stringify(firstBad)}`).toBeNull();
    expect(sampled).toBe(51005);
    // ~2.7 s alone, but 9.8 s measured with the full suite running beside a
    // Rust build: the 5 s default still made this a load test.
  }, 30_000);
});

describe("turbulenceDisplacement", () => {
  it("scales displacement by influence and the influence-over-life curve", () => {
    const early = turbulenceDisplacement(0.0, P);
    const late = turbulenceDisplacement(1.0, P);
    expect(Math.abs(late)).toBeGreaterThan(Math.abs(early));
  });

  it("falls back to constant full influence when no curve is baked", () => {
    // This is the ONLY branch every real turbulence-enabled emitter actually
    // takes today — see TurbulenceParams.influenceOverLifeLut's doc: 10/10
    // in-scope emitters have this field null.
    const noCurve: TurbulenceParams = { ...P, influenceOverLifeLut: null };
    const early = turbulenceDisplacement(0.0, noCurve);
    const late = turbulenceDisplacement(1.0, noCurve);
    expect(early).toBeCloseTo(late, 10);
    // With no explicit turbInfluence, the default is the range's midpoint.
    expect(early).toBeCloseTo((P.influenceMin + P.influenceMax) / 2, 10);
  });

  it("uses the caller's per-particle turbInfluence draw over the midpoint default", () => {
    // dynamic_params.turb_influence is drawn once per particle uniformly
    // over [influenceMin, influenceMax] — this is the only real variation
    // turbulence has today (the influence-over-life curve is null for every
    // in-scope emitter). A caller passing a specific drawn value must see
    // that value used, not the placeholder midpoint.
    const noCurve: TurbulenceParams = { ...P, influenceOverLifeLut: null };
    expect(turbulenceDisplacement(0.0, noCurve, P.influenceMin)).toBeCloseTo(P.influenceMin, 10);
    expect(turbulenceDisplacement(0.0, noCurve, P.influenceMax)).toBeCloseTo(P.influenceMax, 10);
  });

  it("stays within [0, 1] regardless of influence range", () => {
    const hot: TurbulenceParams = { ...P, influenceMin: 1.5, influenceMax: 3.0 };
    expect(turbulenceDisplacement(1.0, hot)).toBeLessThanOrEqual(1.0);
    expect(turbulenceDisplacement(1.0, hot)).toBeGreaterThanOrEqual(0.0);
  });
});
