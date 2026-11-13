import { describe, expect, it } from "vitest";
import { CARD_GEM_PROUD, CARD_H, CARD_STAR_PROUD, CARD_W, cardBoxHeight } from "./cardGeometry";
import { splitBandReal, splitZoomRise, splitZoomWidth } from "./portraitSplit";
import {
  HAND_OVERLAP, LANDSCAPE_LEFT, TOP_MARGIN, fitWidth, portraitLeft, portraitZoomLeft, zoomHitHeight, zoomPlacement, type Rect, type ZoomArgs,
} from "./zoomPlacement";

const rect = (left: number, top: number, right: number, bottom: number): Rect => ({ left, top, right, bottom });
const overlaps = (a: Rect, b: Rect) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
const box = (p: { left: number; top: number; width: number }): Rect =>
  rect(p.left, p.top, p.left + p.width, p.top + cardBoxHeight(p.width));
const inside = (a: Rect, s: Rect) =>
  a.left >= s.left - 0.01 && a.top >= s.top - 0.01 && a.right <= s.right + 0.01 && a.bottom <= s.bottom + 0.01;

// The portrait zoom stands over the player band, horizontally centred (spec "Phase 2:
// portrait split", "The zoomed card"). The old centred-under-the-stage-top
// spot covered the enemies, which now fill the top of the stage.
// Stages as the split lays them out: the hand strip is the bottom 96 px.
const large = rect(0, 0, 412, 692);
const smallStage = rect(0, 0, 375, 475);
const portraitOf = (stage: Rect, small: boolean, enemies: Rect[] = []): ZoomArgs => ({
  mode: "portrait", stage, enemies, hand: rect(stage.left, stage.bottom - 96, stage.right, stage.bottom + 40), small,
});
const portrait = portraitOf(large, false, [rect(30, 40, 190, 380), rect(222, 40, 382, 380)]);
const hOf = (w: number) => (w * CARD_H) / CARD_W + 1;

describe("zoomPlacement, portrait: one fixed spot over the band", () => {
  it("portrait: centred over the band", () => {
    const p = zoomPlacement(portrait);
    // 7 px left of centre on the 412 px stage: End Turn keeps 134 px.
    expect(p.left).toBe(large.left + 120);
    expect(p.left + p.width / 2).toBe((large.left + large.right) / 2 - 7);
    expect(p.left - p.width * CARD_STAR_PROUD).toBeGreaterThanOrEqual(large.left);
    expect(p.left - p.width * CARD_GEM_PROUD).toBeGreaterThanOrEqual(large.left);
    expect(p.width).toBe(158);
    expect(p.width).toBe(splitZoomWidth(false));
    expect(p.top + hOf(158)).toBeCloseTo(portrait.hand.top + HAND_OVERLAP, 6);
    expect(p.covers).toBeUndefined();
    expect(inside(box(p), large)).toBe(true);
  });

  it("small phone: 150 wide, the same edges", () => {
    const a = portraitOf(smallStage, true);
    const p = zoomPlacement(a);
    expect(p.width).toBe(150);
    expect(p.width).toBe(splitZoomWidth(true));
    // 21.5 px left of centre: End Turn (and the status block: a power cell
    // and the energy orb) keeps 134 px right of the card.
    expect(p.left).toBe(375 - 150 - 134);
    expect(p.left - p.width * CARD_STAR_PROUD).toBeGreaterThanOrEqual(0);
    expect(p.top + hOf(150)).toBeCloseTo(a.hand.top + HAND_OVERLAP, 6);
    expect(p.covers).toBeUndefined();
  });

  it("portrait: the card's top is the band's top, or 20 px above it on the small phone", () => {
    for (const [stage, small, edge] of [[large, false, 296], [smallStage, true, 284]] as const) {
      const p = zoomPlacement(portraitOf(stage, small));
      expect(splitBandReal(small) + splitZoomRise(small)).toBe(edge);
      expect(p.top).toBeGreaterThanOrEqual(stage.bottom - splitBandReal(small) - splitZoomRise(small) - 0.5);
    }
  });

  it("the same spot whatever the enemies are", () => {
    const none = zoomPlacement(portraitOf(large, false));
    const five = zoomPlacement(portraitOf(large, false, [0, 1, 2, 3, 4].map((i) => rect(10 + i * 80, 40, 80 + i * 80, 380))));
    expect(zoomPlacement(portrait)).toEqual(none);
    expect(five).toEqual(none);
  });

  it("follows the stage: a stage lower on the page moves the spot with it", () => {
    const st = rect(0, 172, 411, 864);
    const p = zoomPlacement(portraitOf(st, false));
    expect(p.left).toBe(411 - 158 - 134);
    expect(p.top + hOf(p.width)).toBeCloseTo(864 - 96 + HAND_OVERLAP, 6);
  });

  it("safety net: an enemy reaching down into the spot stays covered, the spot unmoved", () => {
    // 158 - 10 is under MIN_WIDTH, so the shrink loop has no step left in
    // portrait: the split never puts an enemy here, `covers` reports it if
    // something else does.
    const low = rect(140, 300, 300, 450);
    const p = zoomPlacement({ ...portrait, enemies: [low] });
    expect(p).toMatchObject({ left: 120, width: 158, covers: true });
    expect(p.top + hOf(p.width)).toBeCloseTo(portrait.hand.top + HAND_OVERLAP, 6);
  });

  it("a short stage shrinks it to fit, never past the hand's overlap line", () => {
    const short = portraitOf(rect(0, 0, 412, 300), false);
    const p = zoomPlacement(short);
    expect(p.width).toBe(fitWidth(short));
    expect(p.top).toBeGreaterThanOrEqual(TOP_MARGIN - 0.01);
    expect(box(p).bottom).toBeLessThanOrEqual(204 + 24 + 0.01);
  });
});

// The large phone sideways without browser bars: a 313px stage.
const landscape: ZoomArgs = {
  mode: "landscape",
  stage: rect(0, 52, 914, 365),
  enemies: [rect(560, 140, 700, 340)],
  hand: rect(0, 269, 914, 400),
  small: false,
  orb: rect(8, 160, 68, 220),
};

describe("zoomPlacement, landscape: one fixed spot on the player's side", () => {
  it("right of the orb's column, a fixed margin under the stage top, 170 wide", () => {
    const p = zoomPlacement(landscape);
    expect(p).toEqual({ left: LANDSCAPE_LEFT, top: 52 + TOP_MARGIN, width: 170 });
    expect(inside(box(p), landscape.stage)).toBe(true);
  });

  it("the same spot with one enemy, three, or none", () => {
    const three = zoomPlacement({ ...landscape, enemies: [rect(480, 140, 560, 340), rect(600, 140, 700, 340), rect(740, 140, 840, 340)] });
    expect(three).toEqual(zoomPlacement(landscape));
    expect(zoomPlacement({ ...landscape, enemies: [] })).toEqual(zoomPlacement(landscape));
  });

  it("a wider orb pushes the spot right of it (still the same for every card)", () => {
    const p = zoomPlacement({ ...landscape, orb: rect(8, 160, 100, 250) });
    expect(p.left).toBe(108);
  });

  it("the short stages take the widest card they hold", () => {
    // 915x332 page (Android Chrome bars): a 233 px stage.
    const bars = { ...landscape, stage: rect(0, 40, 914, 273), hand: rect(0, 177, 914, 300) };
    const p = zoomPlacement(bars);
    expect(p.width).toBe(fitWidth(bars));
    expect(p.width).toBeGreaterThanOrEqual(150);
    expect(inside(box(p), bars.stage)).toBe(true);
    // The small phone with bars: a 196 px stage.
    const small = { ...landscape, small: true, stage: rect(0, 40, 666, 236), hand: rect(0, 140, 666, 260) };
    const q = zoomPlacement(small);
    expect(q.width).toBe(fitWidth(small));
    expect(inside(box(q), small.stage)).toBe(true);
  });

  it("an enemy standing in the spot shrinks the card, its left and top edges unmoved", () => {
    const near = rect(236, 60, 300, 300);
    const p = zoomPlacement({ ...landscape, enemies: [near] });
    expect(p.left).toBe(LANDSCAPE_LEFT);
    expect(p.top).toBe(52 + TOP_MARGIN);
    expect(overlaps(box(p), near)).toBe(false);
    expect(p.width).toBe(160);
  });
});

describe("zoomHitHeight", () => {
  it("landscape: only the part above the hand strip takes taps", () => {
    expect(zoomHitHeight("landscape", 100, 260)).toBe(160);
    expect(zoomHitHeight("landscape", 300, 260)).toBe(0);
  });
  it("portrait: the whole card", () => {
    expect(zoomHitHeight("portrait", 100, 260)).toBeUndefined();
  });
});

describe("zoomPlacement, hard rules", () => {
  it("holds the invariants over a sweep of stages and enemy shapes", () => {
    for (const mode of ["portrait", "landscape"] as const) {
      for (let h = 200; h <= 900; h += 70) {
        for (let w = 300; w <= 950; w += 130) {
          for (const n of [0, 1, 2, 3, 5]) {
            const stage = rect(0, 30, w, 30 + h);
            const hand = rect(0, 30 + h - 96, w, 30 + h + 40);
            const enemies = Array.from({ length: n }, (_, i) =>
              rect(w * 0.45 + i * 40, 30 + h * 0.3, w * 0.45 + i * 40 + 36, 30 + h * 0.7));
            const a: ZoomArgs = { mode, stage, enemies, hand, small: w < 400 };
            const p = zoomPlacement(a);
            const b = box(p);
            expect(inside(b, stage), `${mode} ${w}x${h} n${n}`).toBe(true);
            if (mode === "portrait") expect(b.bottom).toBeLessThanOrEqual(hand.top + 24 + 0.01);
            if (mode === "landscape") expect(p.top).toBe(stage.top + TOP_MARGIN);
            else {
              expect(p.left).toBe(stage.left + portraitZoomLeft(w, p.width));
              expect(p.left).toBeGreaterThanOrEqual(stage.left + portraitLeft(p.width));
              expect(p.left - p.width * CARD_STAR_PROUD).toBeGreaterThanOrEqual(stage.left - 0.01);
            }
            if (!p.covers) for (const e of enemies) expect(overlaps(b, e), `${mode} ${w}x${h} n${n}`).toBe(false);
          }
        }
      }
    }
  });
});
