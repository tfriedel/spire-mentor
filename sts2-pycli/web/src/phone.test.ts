import { describe, expect, it } from "vitest";
import { PHONE_BELOW_GROUND, PHONE_LANDSCAPE_BELOW_GROUND, atLeast, floorPx, isSmallPhone, phoneGroundLimit } from "./phone";

describe("phone floor maths", () => {
  it("converts a real-px floor into design px under the stage scale k", () => {
    expect(floorPx(44, 0.44)).toBeCloseTo(100);
  });
  it("atLeast never goes below the real-px floor", () => {
    expect(atLeast(26, 28, 0.44)).toBeCloseTo(63.6, 1);
  });
  it("keeps the design size where the floor is already met", () => {
    expect(atLeast(102, 60, 1)).toBe(102); // desktop zoom: design size wins
  });
});

describe("phoneGroundLimit", () => {
  it("is null off a phone", () => {
    expect(phoneGroundLimit(null, 1000, 1)).toBeNull();
  });
  it("portrait leaves the strip, the chrome row and a two-row power block under the ground line", () => {
    // 96 strip + 130 below ground + 80 chrome + 20 wrapped name, real px
    expect(phoneGroundLimit("portrait", 1928, 0.5)).toBeCloseTo(1928 - 326 / 0.5);
  });
  it("landscape reserves only the HP bar and a one-line name: the powers stand beside the creature", () => {
    // 96 strip + PHONE_LANDSCAPE_BELOW_GROUND, real px, at k = 0.5
    expect(phoneGroundLimit("landscape", 768, 0.5)).toBeCloseTo(768 - (96 + PHONE_LANDSCAPE_BELOW_GROUND) / 0.5);
    expect(PHONE_LANDSCAPE_BELOW_GROUND).toBeLessThan(PHONE_BELOW_GROUND / 2);
    expect(phoneGroundLimit("portrait", 1928, 0.46)).toBeLessThan(phoneGroundLimit("landscape", 1928, 0.46)!);
  });
});

describe("isSmallPhone", () => {
  it("is the short side under 390, upright or sideways", () => {
    for (const [w, h, small] of [[375, 667, true], [390, 844, false], [412, 915, false]] as const) {
      expect(isSmallPhone(w, h)).toBe(small);
      expect(isSmallPhone(h, w)).toBe(small);
    }
  });
});
