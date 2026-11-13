import { describe, expect, it } from "vitest";
import { CARD_H, CARD_W } from "./cardGeometry";
import { decimillipedeSpan } from "./components/decimillipedeLayout";
import { SEGMENT_SCALE } from "./components/DecimillipedeGroup";
import { PHONE_HAND_SHOW } from "./phone";
import { rigBox } from "./rigBounds";
import { aspectFit, boundsWidth, fitScene, placeEnemies, placePlayer, renderedSize, rowHalfWidths } from "./scene";
import {
  CHROME, SPLIT, heldSplitHeight, shiftSplit, splitBandReal, splitChromeBelow, splitIntentReal, splitLayout, splitOstyRise,
  splitPowerCells, splitZoomRise, splitZoomWidth,
  type SplitCreature, type SplitInput, type SplitLayout,
} from "./portraitSplit";

/** A stage whose battlefield (the part above the hand strip) is `realW` x
 *  `realH` real px, at `k` real px per design px, in design px. The stage the
 *  solver gets runs on under the hand strip: `splitBandReal` counts it. */
const stage = (realW: number, realH: number, k: number) =>
  ({ stageW: realW / k, stageH: (realH + PHONE_HAND_SHOW) / k, k });

/** The battlefields the real app has at the first fight on a 412x915 and a
 *  375x667 phone (Phase 2: portrait split, real-app stage height): a stage of
 *  674 / 456 real px, hand strip included, under the HUD, the tool row and
 *  the bottom bar. (The spike measured 692 / 475.) */
const LARGE = { ...stage(412, 578, 0.25), small: false };
const SMALL = { ...stage(375, 360, 0.22), small: true };

/** A creature from its shipped rig box at its authored scene scale: the crown
 *  is how far the rig reaches above its root, the footprint the row's width. */
function rigCreature(slug: string, rigScale: number): SplitCreature {
  const rig = `/monster_rigs/${slug}/${slug}`;
  const size = renderedSize({ rig, rigScale });
  if (!size) throw new Error(`no rig box for ${slug}`);
  return { crownH: size.top, footW: boundsWidth({ rig, rigScale }), drawW: size.w };
}

const CULTIST = rigCreature("cultists", 0.28);
const BEETLE = rigCreature("shrinker_beetle", 0.25);
const GREMLIN = rigCreature("fat_gremlin", 0.33);
const NIBBIT = rigCreature("nibbit", 0.28);
const TWIG = rigCreature("twig_slime_s", 0.275);
const ROSTER = [CULTIST, BEETLE, GREMLIN, NIBBIT, TWIG];
/** The Ironclad's crown, scene px. */
const PLAYER_CROWN = 278;
const PLAYER_W = 250;

/** The phase-1 row's scale for this roster on this stage (design px per scene
 *  px): the scene fitted to the stage, the row laid out for the width it shows,
 *  and `aspectFit`'s shrink when the row still overhangs it. */
function rowScaleOf(st: { stageW: number; stageH: number }, enemies: SplitCreature[]): number {
  const fit = fitScene(st.stageW, st.stageH);
  const widths = enemies.map((e) => e.footW);
  const half = rowHalfWidths(fit.visibleHalfW, 1, widths);
  const slots = placeEnemies(widths, 1, half.enemies);
  const player = placePlayer(PLAYER_W, 1, false, half.player);
  return fit.k * aspectFit([...slots, player], [...widths, PLAYER_W], 1, fit.visibleHalfW);
}

function input(st: typeof LARGE, enemies: SplitCreature[]): SplitInput {
  return {
    stageW: st.stageW, stageH: st.stageH, k: st.k, small: st.small, enemies,
    refCrown: PLAYER_CROWN, rowScale: rowScaleOf(st, enemies),
  };
}

type Box = { left: number; right: number; top: number; bottom: number };

/** Enemy `i`'s chrome box in REAL px: its slot across (the body or the
 *  chrome's minimum width, plus the power column on its left), from the intent
 *  over its crown down to the chrome under its feet. */
function box(l: SplitLayout, i: number, inp: SplitInput): Box {
  const c = CHROME[l.chrome];
  const e = inp.enemies[i];
  const p = l.enemies[i];
  const half = Math.max(Math.max(e.footW, e.drawW ?? 0) * l.scale * inp.k, c.slot) / 2;
  const x = p.x * inp.k;
  const foot = p.foot * inp.k;
  return {
    left: x - half - c.side,
    right: x + half,
    top: foot - e.crownH * l.scale * inp.k - splitIntentReal(inp.small),
    bottom: foot + splitChromeBelow(l.chrome, inp.small),
  };
}

const disjoint = (a: Box, b: Box) =>
  a.right <= b.left + 0.5 || b.right <= a.left + 0.5 || a.bottom <= b.top + 0.5 || b.bottom <= a.top + 0.5;

const hOf = (w: number) => (w * CARD_H) / CARD_W + 1;

const CASES = [LARGE, SMALL].flatMap((st) =>
  [1, 2, 3, 4, 5].map((n) => ({ st, n, name: `${n} on the ${st.small ? "small" : "large"} phone` })));

describe("splitBandReal", () => {
  it("is the hand, the clearance, the controls and the padded figure", () => {
    expect(splitBandReal(false)).toBe(296); // 96 + 12 + 60 + 6 + 116 + 6
    expect(splitBandReal(true)).toBe(264); // 96 + 12 + 60 + 6 + 84 + 6
  });
});

describe("splitLayout", () => {
  it("the zoom's width fits the band", () => {
    expect(splitZoomWidth(false)).toBe(158);
    expect(splitZoomRise(false)).toBe(0);
    expect(splitZoomWidth(true)).toBe(150);
    expect(splitZoomRise(true)).toBe(20);
    for (const small of [false, true]) {
      expect(hOf(splitZoomWidth(small))).toBeLessThanOrEqual(splitBandReal(small) - 96 + 24 + splitZoomRise(small));
    }
  });

  /** Real px from an enemy's feet to its HP bar's bottom edge (2.9..7.6
   *  under them, measured on the small phone). */
  const HP_UNDER_FEET = 8;

  it("the small phone's risen zoom stays under every enemy's HP bar", () => {
    // The floor no longer keeps the zoom's 20 px clear (enemy size is
    // permanent, the zoom transient); what hangs under the feet keeps every
    // body, intent and HP bar above the zoom's top.
    for (const { st, n, name } of CASES) {
      const inp = input(st, ROSTER.slice(0, n));
      const l = splitLayout(inp);
      const zoomTop = l.bandTop * st.k - splitZoomRise(st.small);
      for (let i = 0; i < n; i++) {
        expect(l.enemies[i].foot * st.k + HP_UNDER_FEET, `${name}: ${i}`).toBeLessThanOrEqual(zoomTop + 0.5);
      }
    }
    expect(splitChromeBelow("column", true)).toBeGreaterThanOrEqual(splitZoomRise(true) + HP_UNDER_FEET);
    for (const mode of ["rows", "column", "compact"] as const) {
      expect(splitChromeBelow(mode, false), `${mode}: the large phone keeps its chrome`).toBe(CHROME[mode].below);
    }
  });

  it("one Cultist on the large phone fills the stage height", () => {
    const inp = input(LARGE, [CULTIST]);
    const l = splitLayout(inp);
    expect(l.chrome).toBe("column");
    expect(l.scale * LARGE.k * CULTIST.crownH).toBeGreaterThan(2 * inp.rowScale * LARGE.k * CULTIST.crownH);
    expect(box(l, 0, inp).top).toBeGreaterThanOrEqual(6 - 0.5);
  });

  it("no chrome box crosses another or the band", () => {
    for (const { st, n, name } of CASES) {
      const inp = input(st, ROSTER.slice(0, n));
      const l = splitLayout(inp);
      const boxes = l.enemies.map((_, i) => box(l, i, inp));
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) expect(disjoint(boxes[i], boxes[j]), `${name}: ${i} x ${j}`).toBe(true);
        const b = boxes[i];
        expect(b.bottom, `${name}: ${i} under the band`).toBeLessThanOrEqual(l.bandTop * st.k + 0.5);
        expect(b.left, `${name}: ${i} left`).toBeGreaterThanOrEqual(-0.5);
        expect(b.right, `${name}: ${i} right`).toBeLessThanOrEqual(st.stageW * st.k + 0.5);
        expect(b.top, `${name}: ${i} top`).toBeGreaterThanOrEqual(-0.5);
      }
    }
  });

  // At the real stage heights every roster here fits at the row's own
  // scale, five on the small phone included (before the small phone stopped
  // keeping the zoom's rise clear, one Cultist alone was drawn under it).
  it("never under the row layout's scale", () => {
    for (const { st, n, name } of CASES) {
      const inp = input(st, ROSTER.slice(0, n));
      // On the small phone a body drawn shorter than its tap box pushes its
      // chrome down (SPLIT.tap): at these test stages' k (0.22, the real app
      // measures 0.42) the row's scale draws the beetle 29 px tall, so the
      // row's placement no longer fits and the split may go under it. The
      // app's own k is pinned below ("never under ..., on either phone").
      const smallest = Math.min(...inp.enemies.map((e) => e.crownH)) * inp.rowScale * st.k;
      if (st.small && smallest < SPLIT.tap - SPLIT.tapHp) continue;
      expect(splitLayout(inp).scale, name).toBeGreaterThanOrEqual(inp.rowScale);
    }
  });

  it("a back rank only when it pays 8%", () => {
    expect(splitLayout(input(LARGE, ROSTER.slice(0, 2))).backCount).toBe(0);
    // Tall enough that a second rank beats one rank of five held by the width.
    const narrow = { ...stage(375, 900, 0.25), small: false };
    expect(splitLayout(input(narrow, [CULTIST, CULTIST, CULTIST, CULTIST, CULTIST])).backCount).toBeGreaterThan(0);
    for (const { st, n } of CASES) {
      const l = splitLayout(input(st, ROSTER.slice(0, n)));
      if (l.chrome === "column") expect(l.backCount).toBe(0);
    }
  });

  it("compact only for a crowd too small otherwise", () => {
    expect(splitLayout(input(SMALL, ROSTER)).chrome).toBe("compact");
    const tall = { ...stage(412, 2000, 0.25), small: false };
    expect(splitLayout(input(tall, [CULTIST])).chrome).not.toBe("compact");
  });

  it("pure", () => {
    const inp = input(LARGE, ROSTER.slice(0, 3));
    expect(splitLayout(inp)).toEqual(splitLayout({ ...inp, enemies: inp.enemies.map((e) => ({ ...e })) }));
  });

  it("one creature for a multi-rig boss", () => {
    const rigs = ["back", "middle", "front"].map((p) => `/monster_rigs/decimillipede_segment_${p}/decimillipede_segment_${p}`);
    const span = decimillipedeSpan(rigs);
    const crown = Math.max(...rigs.map((r) => {
      const b = rigBox(r);
      return b ? (b.y + b.h) * SEGMENT_SCALE : 0;
    }));
    const boss: SplitCreature = { crownH: crown, footW: (span.right - span.left) * SEGMENT_SCALE };
    const inp = input(LARGE, [boss]);
    const l = splitLayout(inp);
    expect(l.enemies).toHaveLength(1);
    const b = box(l, 0, inp);
    expect((b.left + b.right) / 2).toBeCloseTo((LARGE.stageW * LARGE.k) / 2, 1);
  });

  it("puts the figure on the controls and the band's top under the enemies", () => {
    const l = splitLayout(input(LARGE, [CULTIST]));
    expect(l.figureReal).toBe(SPLIT.figure);
    expect(l.figureFoot * LARGE.k).toBeCloseTo(674 - (96 + 12 + 60 + 6), 6);
    expect(l.bandTop * LARGE.k).toBeCloseTo(674 - 296, 6);
    expect(splitLayout(input(SMALL, [CULTIST])).figureReal).toBe(SPLIT.figureSmall);
  });

  it("pins the large phone's drawn heights", () => {
    const one = splitLayout(input(LARGE, [CULTIST]));
    expect(Math.abs(one.scale * LARGE.k * CULTIST.crownH - 260)).toBeLessThanOrEqual(1);
    const three = splitLayout(input(LARGE, ROSTER.slice(0, 3)));
    expect(three.chrome).toBe("rows");
    const drawn = ROSTER.slice(0, 3).map((e) => three.scale * LARGE.k * e.crownH);
    [168, 73, 94].forEach((h, i) => expect(Math.abs(drawn[i] - h)).toBeLessThanOrEqual(1));
  });

  it("five Cultists on the 375-wide large-phone battlefield stand in one compact rank", () => {
    const st = { ...stage(375, 596, 0.25), small: false };
    const five = [CULTIST, CULTIST, CULTIST, CULTIST, CULTIST];
    const inp = input(st, five);
    const l = splitLayout(inp);
    expect(l.backCount).toBe(0);
    expect(l.chrome).toBe("compact");
    expect(l.cramped).toBe(false);
    const boxes = l.enemies.map((_, i) => box(l, i, inp));
    for (let i = 1; i < 5; i++) expect(disjoint(boxes[i - 1], boxes[i])).toBe(true);
  });

  it("gives every enemy its slot's width: at least the chrome's minimum, and its body", () => {
    for (const { st, n, name } of CASES) {
      const inp = input(st, ROSTER.slice(0, n));
      const l = splitLayout(inp);
      expect(l.slots, name).toHaveLength(n);
      const c = CHROME[l.chrome];
      l.slots.forEach((w, i) => {
        const e = inp.enemies[i];
        expect(w * st.k, `${name}: ${i}`).toBeGreaterThanOrEqual(c.slot + c.side - 1e-6);
        expect(w, `${name}: ${i}`).toBeCloseTo(Math.max(Math.max(e.footW, e.drawW ?? 0) * l.scale, c.slot / st.k) + c.side / st.k, 6);
      });
    }
  });

  it("Osty's rise: 18 px over the small phone's band, none over the large one's", () => {
    expect(splitOstyRise(true)).toBe(18); // 64 + 44 - 84 - 6
    expect(splitOstyRise(false)).toBe(0);
    for (const { st, n, name } of CASES) {
      const inp = input(st, ROSTER.slice(0, n));
      const plain = splitLayout(inp);
      const withOsty = splitLayout({ ...inp, osty: true });
      if (!st.small) {
        // The large phone's enemies are not shrunk for it.
        expect(withOsty, name).toEqual(plain);
        continue;
      }
      // The small phone keeps Osty's 18 px over the band clear.
      const limit = withOsty.bandTop * st.k - splitOstyRise(true);
      for (let i = 0; i < n; i++) expect(box(withOsty, i, inp).bottom, `${name}: ${i}`).toBeLessThanOrEqual(limit + 0.5);
    }
  });

  it("a rows slot at the chrome's minimum holds two power cells across", () => {
    for (const k of [0.2, 0.25, 0.42, 0.59]) {
      expect(splitPowerCells(CHROME.rows.slot, k), `k ${k}`).toBe(2);
      expect(splitPowerCells(88, k), `k ${k}`).toBe(1); // 2 x 44 + the gap is over 88
    }
    // Past k = 44/26 the cell is its 26 design px, wider than 44 real.
    expect(splitPowerCells(200, 2)).toBe(3); // (200 + 10) / (52 + 10)
    expect(splitPowerCells(10, 0.25)).toBe(1);
  });

  it("a rig of shoulder-to-shoulder parts never takes the column chrome", () => {
    // One Cultist on the small phone takes the column; as a rig whose parts
    // stand side by side (the Decimillipede) each part's column would stand
    // on its neighbour.
    expect(splitLayout(input(SMALL, [CULTIST])).chrome).toBe("column");
    const l = splitLayout(input(SMALL, [{ ...CULTIST, noColumn: true }]));
    expect(l.chrome).not.toBe("column");
    expect(l.cramped).toBe(false);
  });

  it("is not cramped while some chrome fits", () => {
    for (const { st, n, name } of CASES) expect(splitLayout(input(st, ROSTER.slice(0, n))).cramped, name).toBe(false);
  });
});

describe("splitLayout on inputs no chrome fits", () => {
  /** Finite and positive, every slot inside the stage, and within each rank
   *  the bodies stand left to right without overlapping. Where the stage
   *  holds the band at all, every body stands on the stage, the intent
   *  headroom the layout claims included, with its feet and its claimed
   *  chrome room above the band's top. */
  function expectSane(l: SplitLayout, inp: SplitInput, name: string) {
    expect(Number.isFinite(l.scale) && l.scale > 0, `${name}: scale ${l.scale}`).toBe(true);
    const W = Number.isFinite(inp.stageW) && inp.stageW > 0 ? inp.stageW : Infinity;
    const H = Number.isFinite(inp.stageH) && inp.stageH > 0 ? inp.stageH : Infinity;
    expect(l.enemies).toHaveLength(inp.enemies.length);
    for (const p of l.enemies) {
      expect(Number.isFinite(p.x) && Number.isFinite(p.foot), name).toBe(true);
      expect(p.x, name).toBeGreaterThanOrEqual(0);
      expect(p.x, name).toBeLessThanOrEqual(W);
      expect(p.foot, name).toBeGreaterThanOrEqual(0);
      expect(p.foot, name).toBeLessThanOrEqual(H);
    }
    const k = Number.isFinite(inp.k) && inp.k > 0 ? inp.k : 1;
    if (l.bandTop >= 0) {
      l.enemies.forEach((p, i) => {
        const c = inp.enemies[i].crownH;
        const crown = (Number.isFinite(c) && c > 0 ? c : 1) * l.scale;
        expect((p.foot - crown - l.intentReal / k) * k, `${name}: ${i} intent top`).toBeGreaterThanOrEqual(-1e-6);
        // Above the band (Osty's rise counted: `bandTop` is the band's own
        // top), unless a cramped fight reports its HP bars over it; then
        // still clear of the controls row, whenever an intent and an HP bar
        // fit above that at all.
        const lowest = l.bandTop - (inp.osty ? splitOstyRise(inp.small) / k : 0);
        if (l.figureFoot * k < l.intentReal + SPLIT.tapHp + 1) return;
        if (l.overReal > 0) {
          expect(l.cramped, `${name}: over the band only when cramped`).toBe(true);
          expect((p.foot - lowest) * k + SPLIT.tapHp, `${name}: ${i} over by overReal`).toBeCloseTo(l.overReal, 6);
          expect(p.foot * k + SPLIT.tapHp, `${name}: ${i} HP bar over the controls`)
            .toBeLessThanOrEqual(l.figureFoot * k + 1e-6);
          return;
        }
        expect(p.foot, `${name}: ${i} feet over the band`).toBeLessThanOrEqual(lowest + 1e-6);
        expect(p.foot + l.belowReal / k, `${name}: ${i} chrome over the band`).toBeLessThanOrEqual(lowest + 1e-6);
        expect(l.belowReal, `${name}: ${i} HP bar over the band`).toBeGreaterThanOrEqual(SPLIT.tapHp - 1e-6);
      });
    }
    const half = (e: SplitCreature) => Math.max(e.footW || 0, e.drawW || 0) * l.scale / 2;
    for (const [from, to] of [[0, l.backCount], [l.backCount, l.enemies.length]]) {
      for (let i = from + 1; i < to; i++) {
        const [a, b] = [l.enemies[i - 1], l.enemies[i]];
        expect(b.x, `${name}: ${i} ordered`).toBeGreaterThan(a.x);
        expect(b.x - a.x + 1e-6, `${name}: ${i} apart`)
          .toBeGreaterThanOrEqual(half(inp.enemies[i - 1]) + half(inp.enemies[i]));
      }
    }
  }

  const crowd = (n: number) => Array.from({ length: n }, (_, i) => ROSTER[i % ROSTER.length]);

  it("crowds of 6, 7 and 8: two compact ranks on the large phone, one cramped rank on the small", () => {
    for (const st of [LARGE, SMALL]) {
      for (const n of [6, 7, 8]) {
        const inp = input(st, crowd(n));
        const l = splitLayout(inp);
        const name = `${n} on the ${st.small ? "small" : "large"} phone`;
        expectSane(l, inp, name);
        expect(l.chrome, name).toBe("compact");
        if (st.small) {
          // Six 64 px compact slots and their gaps are wider than 375 - 2 * 6.
          expect(l.cramped, name).toBe(true);
          expect(l.backCount, name).toBe(0);
        } else {
          expect(l.cramped, name).toBe(false);
          expect(l.backCount, name).toBeGreaterThan(0);
          const boxes = l.enemies.map((_, i) => box(l, i, inp));
          for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) expect(disjoint(boxes[i], boxes[j]), `${name}: ${i} x ${j}`).toBe(true);
          }
        }
      }
    }
  });

  // A phone in a browser has a shorter stage than the app measured at the
  // first fight (a small phone with its bars: ~340-390 px). On every stage
  // from 300 to 700 real px nothing leaves the stage or reaches into the
  // band, and no chrome mode passes for a fit that draws the tallest enemy
  // as a speck.
  it("short and tall stages, 300 to 700 real px: on the stage, above the band", () => {
    const phones = [
      { ...SMALL, w: 375, label: "small" }, { ...LARGE, w: 412, label: "large" },
      { ...SMALL, k: 0.42, w: 375, label: "small, app k" }, { ...LARGE, k: 0.46, w: 412, label: "large, app k" },
    ];
    for (const ph of phones) {
      for (let real = 300; real <= 700; real += 10) {
        const st = { ...stage(ph.w, real - PHONE_HAND_SHOW, ph.k), small: ph.small };
        for (const n of [1, 3, 5]) {
          for (const osty of [false, true]) {
            const inp = { ...input(st, ROSTER.slice(0, n)), osty };
            const l = splitLayout(inp);
            const name = `${ph.label} ${real} px, ${n}${osty ? " + Osty" : ""}`;
            expectSane(l, inp, name);
            expect(l.bandTop, name).toBeGreaterThanOrEqual(0);
            if (!l.cramped) {
              // A player-sized creature would be drawn at least `minCrown`.
              const ref = Math.max(inp.refCrown, ...inp.enemies.map((e) => e.crownH)) * l.scale * ph.k;
              expect(ref, `${name}: ${l.chrome}`).toBeGreaterThanOrEqual(SPLIT.minCrown - 1e-6);
              expect(l.intentReal, name).toBe(splitIntentReal(ph.small));
              expect(l.belowReal, name).toBe(splitChromeBelow(l.chrome, ph.small));
            }
          }
        }
      }
    }
  });

  // An enemy the player cannot see is worse than one over the band: a
  // cramped fight keeps a player-sized enemy `minCrown` tall on every stage
  // where an intent, that crown and an HP bar fit above the controls row
  // (letting the HP bars down over the band's upper part), unless the
  // stage's width holds it smaller.
  it("cramped: a player-sized enemy stays minCrown tall while it fits above the controls row", () => {
    const phones = [
      { ...SMALL, w: 375, label: "small" }, { ...LARGE, w: 412, label: "large" },
      { ...SMALL, k: 0.42, w: 375, label: "small, app k" }, { ...LARGE, k: 0.46, w: 412, label: "large, app k" },
    ];
    let checked = 0;
    for (const ph of phones) {
      for (let real = 240; real <= 520; real += 2) {
        const st = { ...stage(ph.w, real - PHONE_HAND_SHOW, ph.k), small: ph.small };
        for (const roster of [[CULTIST], ROSTER.slice(0, 3), ROSTER.slice(0, 5)]) {
          for (const osty of [false, true]) {
            const inp = { ...input(st, roster), osty };
            const l = splitLayout(inp);
            const name = `${ph.label} ${real} px, ${roster.length}${osty ? " + Osty" : ""}`;
            expectSane(l, inp, name);
            if (!l.cramped) continue;
            if (l.figureFoot * ph.k < l.intentReal + SPLIT.minCrown + SPLIT.tapHp) continue;
            const ref = Math.max(inp.refCrown, ...roster.map((e) => e.crownH)) * l.scale * ph.k;
            const byWidth = roster.some((e, i) => Math.max(e.footW, e.drawW ?? 0) * l.scale >= l.slots[i] - 1e-6);
            if (!byWidth) checked++;
            expect(ref >= SPLIT.minCrown - 1e-6 || byWidth, `${name}: player-sized crown ${ref.toFixed(1)}`).toBe(true);
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("a stage shorter than the band", () => {
    const st = { ...stage(412, 100, 0.25), small: false };
    const inp = input(st, ROSTER.slice(0, 3));
    const l = splitLayout(inp);
    expectSane(l, inp, "short stage");
    expect(l.cramped).toBe(true);
  });

  it("k of 0 or NaN is taken as 1", () => {
    for (const k of [0, Number.NaN, -1]) {
      const inp = { ...input(LARGE, ROSTER.slice(0, 3)), k };
      const l = splitLayout(inp);
      expectSane(l, inp, `k=${k}`);
      expect(l).toEqual(splitLayout({ ...inp, k: 1 }));
    }
  });

  it("zero or NaN crowns, widths and stage sizes", () => {
    const bad: SplitCreature[] = [
      { crownH: 0, footW: 0 }, { crownH: Number.NaN, footW: Number.NaN, drawW: Number.NaN }, CULTIST,
    ];
    const inp = input(LARGE, bad);
    expectSane(splitLayout(inp), inp, "bad creatures");
    for (const [stageW, stageH] of [[0, 0], [Number.NaN, Number.NaN], [-5, 100]]) {
      const inp2 = { ...input(LARGE, ROSTER.slice(0, 2)), stageW, stageH, rowScale: Number.NaN, refCrown: Number.NaN };
      const l = splitLayout(inp2);
      expect(Number.isFinite(l.scale) && l.scale > 0, `stage ${stageW}x${stageH}`).toBe(true);
      for (const p of l.enemies) expect(Number.isFinite(p.x) && Number.isFinite(p.foot)).toBe(true);
    }
  });
});

describe("the split holds through a stage height wobble", () => {
  const k = LARGE.k;
  it("keeps the held height under SPLIT.margin real px, either way", () => {
    expect(heldSplitHeight(null, 2000, k)).toBe(2000);
    expect(heldSplitHeight(2000, 2000 - 12, k)).toBe(2000); // 3 real px
    expect(heldSplitHeight(2000, 2000 + 12, k)).toBe(2000);
    expect(heldSplitHeight(2000, 2000 + (SPLIT.margin - 0.5) / k, k)).toBe(2000);
  });
  it("re-solves at SPLIT.margin real px and on a big change", () => {
    expect(heldSplitHeight(2000, 2000 + SPLIT.margin / k, k)).toBe(2000 + SPLIT.margin / k);
    expect(heldSplitHeight(2000, 2000 - 24 / k, k)).toBe(2000 - 24 / k);
    expect(heldSplitHeight(2000, 900, k)).toBe(900);
  });
  // `shiftSplit` is only exact while every vertical output is linear in the
  // stage height at a fixed scale: pinned here against the solver itself.
  it("a shifted solve is the solve of the taller stage at the held scale", () => {
    let sameScale = 0;
    for (const enemies of [[CULTIST], ROSTER.slice(0, 3), ROSTER]) {
      const inp: SplitInput = { ...LARGE, enemies, refCrown: PLAYER_CROWN, rowScale: 0, osty: false };
      const held = splitLayout(inp);
      const dy = 8;
      const shifted = shiftSplit(held, dy);
      const fresh = splitLayout({ ...inp, stageH: inp.stageH + dy });
      expect(shifted.bandTop).toBeCloseTo(fresh.bandTop, 6);
      expect(shifted.figureFoot).toBeCloseTo(fresh.figureFoot, 6);
      // The fresh solve may grow the scale; at the held one the feet are where
      // the taller stage's floor puts them.
      if (Math.abs(fresh.scale - held.scale) < 1e-9) {
        sameScale++;
        shifted.enemies.forEach((e, i) => expect(e.foot).toBeCloseTo(fresh.enemies[i].foot, 6));
      }
      expect(shifted.scale).toBe(held.scale);
      expect(shifted.enemies.map((e) => e.x)).toEqual(held.enemies.map((e) => e.x));
      expect(shiftSplit(held, 0)).toBe(held);
    }
    expect(sameScale, "a width-bound roster pins the feet").toBeGreaterThan(0);
  });

  // A cramped solve clamps its feet to the stage; held and shifted by a
  // wobble it must still stand on the stage, intents included, and keep its
  // chrome above the band.
  it("a cramped roster, held and shifted by ±3 real px, stays on the stage and above the band", () => {
    const crowd = Array.from({ length: 8 }, (_, i) => ROSTER[i % ROSTER.length]);
    const cases: { name: string; inp: SplitInput }[] = [
      { name: "8 on the small phone", inp: { ...SMALL, enemies: crowd, refCrown: PLAYER_CROWN, rowScale: 0, osty: false } },
      // Height-bound: the intent stands exactly at the top margin, the feet
      // as far into the compact chrome's room as `SPLIT.minCrown` needs.
      { name: "8 on the large phone, short", inp: { ...stage(412, 360, 0.25), small: false, enemies: crowd, refCrown: PLAYER_CROWN, rowScale: 0, osty: false } },
    ];
    for (const { name, inp } of cases) {
      const held = splitLayout(inp);
      expect(held.cramped, name).toBe(true);
      // Its own solve stands at or below the top margin.
      held.enemies.forEach((p, i) => expect(
        (p.foot - inp.enemies[i].crownH * held.scale - held.intentReal / inp.k) * inp.k, `${name}: enemy ${i} held top`,
      ).toBeGreaterThanOrEqual(SPLIT.margin - 1e-6));
      for (const real of [-3, 3]) {
        const dy = real / inp.k;
        const l = shiftSplit(held, dy);
        const H = inp.stageH + dy;
        const at = `${name}, ${real} px`;
        expect(l.bandTop, at).toBeCloseTo(H - splitBandReal(inp.small) / inp.k, 6);
        l.enemies.forEach((p, i) => {
          const top = p.foot - inp.enemies[i].crownH * l.scale - l.intentReal / inp.k;
          expect(top, `${at}: enemy ${i} intent top`).toBeGreaterThanOrEqual(-1e-6);
          // The chrome room it keeps (on this short stage less than the
          // compact chrome's 72: its feet stand lower so the crowd keeps
          // `SPLIT.minCrown`).
          expect(p.foot + l.belowReal / inp.k, `${at}: enemy ${i} chrome bottom`)
            .toBeLessThanOrEqual(l.bandTop + 1e-6);
        });
      }
    }
  });
});

/** The real app's solver inputs at the first fight, logged from Board on
 *  2026-11-12 (Phase 2: portrait split, real-app stage height): the fixture
 *  page on 412x742 / 375x524, which gives it the real app's 674 / 456 real px
 *  stage, and the real first fight (one Shrinker Beetle) on 412x915 / 375x667.
 *  Unlike `LARGE` / `SMALL` above, `k` is the app's own. Heights here are the
 *  solver's (crown x scale x k); the drawn ink measures ~3 px less. */
describe("the real app's first fight (Phase 2: portrait split, real-app stage height)", () => {
  const APP_SMALL = { stageW: 892, stageH: 1088, k: 0.42, small: true };
  const APP_LARGE = { stageW: 896, stageH: 1464, k: 0.46, small: false };
  const REF = 338.52; // the Ironclad's crown, scene px
  const cultist: SplitCreature = { crownH: 341.32, footW: 292.32, drawW: 306.88 };
  const beetle: SplitCreature = { crownH: 148.5, footW: 253.2, drawW: 247.5 };
  /** The real fight's beetle stands on a narrower footprint. */
  const realBeetle: SplitCreature = { crownH: 148.5, footW: 240, drawW: 247.5 };
  const lagavulin: SplitCreature = { crownH: 485.4, footW: 894.24, drawW: 834 };
  const queen: SplitCreature = { crownH: 493.64, footW: 354.648, drawW: 346.92 };
  const effigy: SplitCreature = { crownH: 587.84, footW: 403.2, drawW: 338.56 };
  /** [name, roster, the row's scale on the small phone, on the large]. */
  const FIGHTS: [string, SplitCreature[], number, number][] = [
    ["1 Cultist", [cultist], 0.6272, 0.9796],
    ["real 1 Beetle", [realBeetle], 0.8641, 1.0221],
    ["2 Beetles", [beetle, beetle], 0.6016, 0.6043],
    ["3 Beetles", [beetle, beetle, beetle], 0.4956, 0.4978],
    ["4 Beetles", [beetle, beetle, beetle, beetle], 0.4421, 0.4440],
    ["5 Beetles", [beetle, beetle, beetle, beetle, beetle], 0.3919, 0.3937],
    ["Lagavulin", [lagavulin], 0.4204, 0.4223],
    ["Queen", [queen], 0.4336, 0.8594],
    ["Effigy", [effigy], 0.3978, 0.7856],
  ];
  const solve = (st: typeof APP_SMALL, roster: SplitCreature[], rowScale: number) =>
    splitLayout({ ...st, enemies: roster, refCrown: REF, rowScale });
  const drawn = (st: typeof APP_SMALL, l: SplitLayout, e: SplitCreature) => e.crownH * l.scale * st.k;
  const fight = (name: string) => FIGHTS.find((f) => f[0] === name)!;

  it("the stages are the real app's 456 and 674 real px", () => {
    expect(Math.round(APP_SMALL.stageH * APP_SMALL.k)).toBe(457);
    expect(Math.round(APP_LARGE.stageH * APP_LARGE.k)).toBe(673);
  });

  it("small phone: one Cultist is drawn at least 85 px tall (was 55)", () => {
    const [, roster, rs] = fight("1 Cultist");
    expect(drawn(APP_SMALL, solve(APP_SMALL, roster, rs), cultist)).toBeGreaterThanOrEqual(85);
  });

  it("small phone: Lagavulin, the Queen and the Effigy at least 80 px (were under 60)", () => {
    for (const name of ["Lagavulin", "Queen", "Effigy"]) {
      const [, roster, rs] = fight(name);
      expect(drawn(APP_SMALL, solve(APP_SMALL, roster, rs), roster[0]), name).toBeGreaterThanOrEqual(80);
    }
  });

  it("small phone: three Shrinker Beetles at least 40 px (were 31)", () => {
    const [, roster, rs] = fight("3 Beetles");
    const l = solve(APP_SMALL, roster, rs);
    expect(l.chrome).toBe("column");
    expect(drawn(APP_SMALL, l, beetle)).toBeGreaterThanOrEqual(40);
  });

  it("never under the row layout's scale, on either phone", () => {
    for (const st of [APP_SMALL, APP_LARGE]) {
      for (const [name, roster, small, large] of FIGHTS) {
        const rs = st.small ? small : large;
        expect(solve(st, roster, rs).scale, `${name}, ${st.small ? "small" : "large"}`).toBeGreaterThanOrEqual(rs);
      }
    }
  });

  it("small phone: the real first fight's beetle is drawn at the row's own scale", () => {
    // Its row draws it larger than the player-crown cap would (`refCrown`).
    const [, roster, rs] = fight("real 1 Beetle");
    expect(solve(APP_SMALL, roster, rs).scale).toBeCloseTo(rs, 6);
  });

  it("large phone: unchanged by the small phone's budget", () => {
    // Scales as logged before and after the small phone's fix.
    const want: Record<string, [string, number]> = {
      "1 Cultist": ["column", 1.6524], "real 1 Beetle": ["column", 1.6661], "2 Beetles": ["column", 1.3057],
      "3 Beetles": ["rows", 1.0753], "4 Beetles": ["rows", 0.8332], "5 Beetles": ["compact", 0.7279],
      Lagavulin: ["column", 0.8610], Queen: ["column", 1.1425], Effigy: ["column", 0.9594],
    };
    for (const [name, roster, , rs] of FIGHTS) {
      const l = solve(APP_LARGE, roster, rs);
      expect(l.chrome, name).toBe(want[name][0]);
      expect(l.scale, name).toBeCloseTo(want[name][1], 3);
    }
  });
});

describe("slots cut to the ink (a rig's measured idle box)", () => {
  /** `c` with its rig's ink box, `frac` of its footprint wide. */
  const inked = (c: SplitCreature, frac: number): SplitCreature => ({ ...c, inkW: c.footW * frac });

  it("four width-bound enemies are drawn larger by the ink's share of the footprint", () => {
    const plain = splitLayout(input(LARGE, [BEETLE, BEETLE, BEETLE, BEETLE]));
    const tight = splitLayout(input(LARGE, [0, 1, 2, 3].map(() => inked(BEETLE, 0.8))));
    expect(tight.chrome).toBe(plain.chrome);
    expect(tight.scale).toBeGreaterThan(plain.scale * 1.15);
    expect(tight.scale).toBeLessThanOrEqual(plain.scale / 0.8 + 1e-9);
  });

  it("the ink boxes, centred on their places, keep the gap and stay on the stage", () => {
    const es = [0, 1, 2, 3].map(() => inked(BEETLE, 0.8));
    const inp = input(LARGE, es);
    const l = splitLayout(inp);
    const m = SPLIT.margin / inp.k;
    const boxes = l.enemies.map((p, i) => {
      const c = p.x;
      const half = (es[i].inkW! * l.scale) / 2;
      return { left: c - half, right: c + half };
    });
    expect(boxes[0].left).toBeGreaterThanOrEqual(m - 1e-6);
    expect(boxes[3].right).toBeLessThanOrEqual(inp.stageW - m + 1e-6);
    for (let i = 1; i < boxes.length; i++) {
      expect(boxes[i].left - boxes[i - 1].right).toBeGreaterThanOrEqual(SPLIT.gap / inp.k - 1e-6);
    }
  });

  it("a creature without an ink box is laid out as before", () => {
    const a = splitLayout(input(LARGE, [BEETLE, CULTIST]));
    const b = splitLayout(input(LARGE, [{ ...BEETLE, inkW: undefined }, CULTIST]));
    expect(b).toEqual(a);
  });
});

describe("a body of parts takes the compact chrome whenever it draws it 1.4x larger", () => {
  // The Decimillipede: 1146 scene px across, 330 tall, its parts shoulder to shoulder.
  const DECI: SplitCreature = { crownH: 330, footW: 1146, noColumn: true };
  it("on a 496 px stage of the small phone it is not the 160 px speck the rows chrome leaves", () => {
    const st = { ...stage(375, 496 - PHONE_HAND_SHOW, 0.22), small: true };
    const l = splitLayout({ ...input(st, [DECI]), rowScale: 0.3 });
    expect(l.chrome).toBe("compact");
    expect(DECI.footW * l.scale * st.k).toBeGreaterThan(300);
  });
  it("a tall stage keeps the rows chrome: compact draws it no larger there", () => {
    const st = { ...stage(412, 714 - PHONE_HAND_SHOW, 0.25), small: false };
    expect(splitLayout({ ...input(st, [DECI]), rowScale: 0.3 }).chrome).toBe("rows");
  });
});
