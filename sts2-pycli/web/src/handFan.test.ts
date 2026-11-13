import { describe, expect, it } from "vitest";
import { fanLayout, handFloors, handInsets } from "./handFan";
import { floorPx } from "./phone";

describe("fanLayout flat", () => {
  it("a flat fan has no tilt and no arc", () => {
    const slots = fanLayout(5, 280, 800, -1, { flat: true });
    expect(slots.every((s) => s.rot === 0)).toBe(true);
    expect(slots.every((s) => s.drop === 0)).toBe(true);
  });
  it("without `flat` the fan still tilts and arcs", () => {
    const slots = fanLayout(5, 280, 800, -1);
    expect(slots[0].rot).toBeLessThan(0);
    expect(slots[0].drop).toBeGreaterThan(0);
  });
  it("a flat fan's hover never moves the hovered card", () => {
    for (let h = 0; h < 7; h++) expect(fanLayout(7, 280, 1400, h, { flat: true })[h].dx).toBe(0);
  });
});

describe("handFloors", () => {
  it("portrait / landscape floors", () => {
    expect(handFloors("portrait", false)).toEqual({ cardW: 124, showH: 96 });
    expect(handFloors("landscape", false)).toEqual({ cardW: 136, showH: 96 });
  });
  it("a small portrait phone drops the card to 108", () => {
    expect(handFloors("portrait", true)!.cardW).toBe(108);
  });
  it("the small phone is the same phone sideways: landscape keeps 136", () => {
    expect(handFloors("landscape", true)!.cardW).toBe(136);
  });
  it("off a phone there are no floors", () => {
    expect(handFloors(null, false)).toBeNull();
    expect(handInsets(null)).toBeNull();
  });
});

describe("ten cards on a phone", () => {
  // What Hand.tsx computes: card width = the real floor / k, the fan's room =
  // the stage less the insets. Every card must keep a 24 real-px strip.
  const cells = [
    { mode: "portrait" as const, w: 375, k: 0.42 },
    { mode: "portrait" as const, w: 412, k: 0.46 },
    { mode: "landscape" as const, w: 667, k: 0.46 },
    { mode: "landscape" as const, w: 915, k: 0.5 },
  ];
  for (const { mode, w, k } of cells) {
    it(`${mode} ${w}px keeps a 24px uncovered strip per card`, () => {
      const cw = Math.round(floorPx(handFloors(mode, w < 390)!.cardW, k));
      const ins = handInsets(mode)!;
      const availW = w / k - floorPx(ins.left + ins.right, k);
      const slots = fanLayout(10, cw, availW, -1, { flat: true });
      for (const s of slots.slice(1)) expect((cw - s.overlap) * k).toBeGreaterThanOrEqual(24);
      // and the fan fits the room the chrome leaves it
      expect((cw + slots.slice(1).reduce((a, s) => a + cw - s.overlap, 0)) * k).toBeLessThanOrEqual(availW * k + 1);
    });
  }
});
