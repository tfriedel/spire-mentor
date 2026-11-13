/**
 * The phone-portrait combat screen's layout: the enemies across the top of the
 * stage, the player in a band right above the controls row and the hand, the
 * zoomed card over that band.
 *
 * Pure: solved once per fight from the roster and the stage size, never from
 * powers, selection or any other per-turn state, so nothing moves mid-fight
 * except when a never-seen enemy (a summon) re-solves it.
 *
 * Units: `stageW` / `stageH` and every returned position are DESIGN px (the
 * `.gs-stage-scale` layer); a creature's `crownH` / `footW` are SCENE px; the
 * fixed sizes (`SPLIT`, `CHROME`) are REAL px and go through `k` on the way
 * in. `scale` is design px per scene px.
 */
import { CARD_H, CARD_W } from "./cardGeometry";
import { PHONE_HAND_SHOW, atLeast } from "./phone";
import { POWER_GAP } from "./powerRow";
import { HAND_OVERLAP, MIN_WIDTH, zoomWidth } from "./zoomPlacement";

/** How an enemy's powers hang off it: two rows under the feet, a column
 *  beside the body, or one cell under the feet. */
export type PowerChrome = "rows" | "column" | "compact";

/** The split's fixed sizes, real px. */
export const SPLIT = {
  /** The player's figure in the band; `figureSmall` on the small phone. */
  figure: 116,
  figureSmall: 84,
  /** Air above and below the figure inside the band. */
  bandPad: 6,
  /** The controls row (End Turn, piles, energy) on top of the hand. */
  controls: 60,
  /** Clear between the controls row and the hand strip. */
  handClear: 12,
  /** The band's right gutter. */
  gutter: 16,
  /** Air between an enemy's chrome and the stage edge. */
  margin: 6,
  /** Air between neighbouring enemy slots. */
  gap: 4,
  /** Intent headroom over an enemy's crown. */
  intent: 64,
  /** The small phone's: what the intent chip really takes there. Its 44 px
   *  box stands `25 design px - 1/4 box` over the crown (game-skin.css
   *  `.gs-intent`), so its top is `25k + 33` real px over it: 43.5 at the
   *  small phone's k of 0.42 (measured in the app). */
  intentSmall: 44,
  /** The Necrobinder's Osty in the band. */
  osty: 64,
  /** Osty's HP bar and name under its feet in the band: it stands this much
   *  higher than the figure's feet. */
  ostyChrome: 44,
  /** The smallest crown a chrome mode may draw a player-sized creature at
   *  (`refCrown`, or the roster's tallest if taller): below it the mode does
   *  not fit (a 2 px enemy is no layout). An enemy smaller than the player
   *  may be drawn smaller, in proportion, as the scale cap draws it. 26, not
   *  24: a Cultist's ink is ~83% of its rig crown, and 24 drew 19.9 px. */
  minCrown: 26,
  /** An enemy's tap box: however small its body is drawn, it takes this
   *  much from its crown down (phone.css `[data-enemy-index]`), and the
   *  chrome under the feet hangs under that box. */
  tap: 44,
  /** What the tap box holds under the feet in every mode: the HP bar
   *  (measured 2.9-7.6 under them). Only a body shorter than `tap` less this
   *  pushes the chrome down. */
  tapHp: 8,
} as const;

/** Per power-chrome mode, real px: how far the chrome hangs below the feet,
 *  the narrowest an enemy's slot gets, and the column beside the body. */
export const CHROME: Record<PowerChrome, { below: number; belowSmall: number; slot: number; side: number }> = {
  rows: { below: 140, belowSmall: 140, slot: 92, side: 0 },
  column: { below: 48, belowSmall: 32, slot: 60, side: 46 },
  compact: { below: 72, belowSmall: 72, slot: 64, side: 0 },
};

/** How far a mode's chrome hangs below the feet, real px. On the small phone
 *  a column enemy's HP bar and one-line name end ~27 px under its feet
 *  (measured); 32 keeps the HP bar clear of the zoomed card, which rises 20
 *  px over the band's top there (`splitZoomRise`), and lets it cover at most
 *  the name's bottom while it is open. */
export const splitChromeBelow = (mode: PowerChrome, small: boolean) =>
  (small ? CHROME[mode].belowSmall : CHROME[mode].below);

/** One solver creature, scene px. A multi-rig boss (Kaiser Crab,
 *  Decimillipede) is ONE creature spanning its whole rig. */
export interface SplitCreature {
  /** Height the rig reaches above its root. */
  crownH: number;
  /** Footprint width. */
  footW: number;
  /** Drawn rig width, when wider than the footprint. */
  drawW?: number;
  /** The width of the rig's measured idle ink box. When known the slot is
   *  cut to the INK (the footprint is the game's hitbox, which keeps a 16:9
   *  row airy; a phone has no width to spare); the caller centres the ink
   *  in the slot (Board's `bodyDx`), the place stays the slot's centre. */
  inkW?: number;
  /** Its targetable parts stand shoulder to shoulder (the Decimillipede's
   *  segments): each part's power column would stand on its neighbour, so
   *  the fight never takes the `column` chrome. */
  noColumn?: boolean;
}

export interface SplitInput {
  /** Stage size, design px. */
  stageW: number;
  stageH: number;
  /** `--gs-k`: real px per design px. */
  k: number;
  /** The small phone (`isSmallPhoneScreen`). */
  small: boolean;
  /** The fight's roster, in slot order. */
  enemies: SplitCreature[];
  /** The player's crown, scene px: the scale cap's floor, so a lone small
   *  enemy is drawn no larger than one as tall as the player would be. */
  refCrown: number;
  /** The row layout's scale, design px per scene px: the scale's floor. */
  rowScale: number;
  /** The fight has an Osty (the Necrobinder's; alive or not, so its death
   *  never re-solves). Osty stands on its chrome beside the figure, and on the
   *  small phone that reaches above the band's top: the enemies' chrome
   *  keeps clear of it (`splitOstyRise`). */
  osty?: boolean;
}

/** An enemy's slot: its rig root's x (the slot's centre, less the ink's
 *  offset from the root) and its feet, design px. */
export interface SplitPlace {
  x: number;
  foot: number;
}

export interface SplitLayout {
  /** Design px per scene px, the same for every enemy. */
  scale: number;
  chrome: PowerChrome;
  /** Enemies `0..backCount-1` stand in the back rank. */
  backCount: number;
  /** Index-aligned with `SplitInput.enemies`. */
  enemies: SplitPlace[];
  /** Each enemy's slot width, design px, its power column (if any)
   *  included: what its power chrome may span. Index-aligned. */
  slots: number[];
  /** Design px: the band's top edge, the lowest any enemy chrome reaches. */
  bandTop: number;
  /** Design px: the player figure's feet, on top of the controls row. */
  figureFoot: number;
  /** The figure's height, real px: 116, or 84 on the small phone. */
  figureReal: number;
  /** No chrome fits the roster at its minimum slot width (a crowd of six or
   *  more, a stage shorter than the band): the enemies stand in one compact
   *  rank of equal slots, `(available width) / n` each, so a body may be
   *  narrower than a 44 px tap box. */
  cramped: boolean;
  /** Real px of intent headroom the layout keeps over each crown, and of
   *  chrome room under each enemy's feet before the band (Osty's rise
   *  aside): `splitIntentReal` / `splitChromeBelow`, less on a cramped
   *  stage too short for them (`cramped`). */
  intentReal: number;
  belowReal: number;
  /** Real px by which a cramped fight's HP bars reach down over the band's
   *  top (or Osty's, if it stands taller): 0 unless the stage is too short
   *  to draw a player-sized enemy `SPLIT.minCrown` tall otherwise. Never
   *  down to the controls row. */
  overReal: number;
}

const figureOf = (small: boolean) => (small ? SPLIT.figureSmall : SPLIT.figure);

/** The band's height in real px from the stage bottom: the hand strip, the
 *  clearance over it, the controls row and the padded figure. Nothing is added
 *  for the zoomed card: enemy size is permanent, the zoom transient. */
export function splitBandReal(small: boolean): number {
  return PHONE_HAND_SHOW + SPLIT.handClear + SPLIT.controls + SPLIT.bandPad + figureOf(small) + SPLIT.bandPad;
}

/** A `rows` enemy's power cells across: how many of its drawn cells fit its
 *  solver slot (`SplitLayout.slots`, here in real px) at stage zoom `k`.
 *  Counted with the cells and the gap the grid really draws (`Powers`: a
 *  `max(26, 44 / k)` design-px tap box, `POWER_GAP` design px apart), so the
 *  92 px minimum slot holds two 44 px cells (~2 px apart on a phone), never
 *  one stacked over "+N". At least one. */
export function splitPowerCells(slotReal: number, k: number): number {
  const cell = atLeast(26, 44, k) * k;
  const gap = POWER_GAP * k;
  return Math.max(1, Math.floor((slotReal + gap) / (cell + gap) + 1e-6));
}

/** Real px Osty (64 tall, on its 44 px of chrome) reaches above the band's
 *  top: 18 on the small phone (84 px figure), 0 on the large one. */
export function splitOstyRise(small: boolean): number {
  return Math.max(0, SPLIT.osty + SPLIT.ostyChrome - figureOf(small) - SPLIT.bandPad);
}

/** Real px above the band's top no enemy chrome may take: Osty's rise, in a
 *  fight that has one. The zoomed card's rise (`splitZoomRise`, 20 on the
 *  small phone) is NOT kept clear: the zoom is transient and closes with its
 *  x, and the small phone's chrome under the feet (`splitChromeBelow`) keeps
 *  every HP bar, body and intent above it. Enemy size is permanent. */
const riseOf = (inp: SplitInput) => (inp.osty ? splitOstyRise(inp.small) : 0);

/** Intent headroom over an enemy's crown, real px. */
export const splitIntentReal = (small: boolean) => (small ? SPLIT.intentSmall : SPLIT.intent);

/** The zoomed card's height at width `w`, plus CardFrame's rounding px. */
const hOf = (w: number) => (w * CARD_H) / CARD_W + 1;

/** Real px from the hand strip's top (less the zoom's allowed overlap) up to
 *  the band's top: the height a zoom can take without rising above the band. */
const zoomRoom = (small: boolean) => splitBandReal(small) - PHONE_HAND_SHOW + HAND_OVERLAP;

/** The zoomed card's width: the largest whose top stays at or under the
 *  band's top, floored at `MIN_WIDTH` and capped at the portrait width. */
export function splitZoomWidth(small: boolean): number {
  const fit = Math.floor(((zoomRoom(small) - 1) * CARD_W) / CARD_H);
  return Math.min(zoomWidth("portrait", small), Math.max(MIN_WIDTH, fit));
}

/** Real px the zoomed card rises above the band's top: 0 on the large phone;
 *  on the small one even `MIN_WIDTH` does not fit, so it rises 20, over the
 *  air and at most the names under the enemies' feet (`splitChromeBelow`). */
export function splitZoomRise(small: boolean): number {
  return Math.max(0, Math.ceil(hOf(splitZoomWidth(small)) - zoomRoom(small)));
}

const bodyW = (e: SplitCreature) => e.inkW ?? Math.max(e.footW, e.drawW ?? 0);
/** The ink centre's offset right of the root, scene px. */

/** Bisection steps for the largest scale that fits. */
const STEPS = 30;
/** A richer chrome wins over rows at this much more scale. */
const COLUMN_PAYS = 1.05;
/** A back rank must draw them this much bigger. */
const BACK_PAYS = 1.08;
/** Compact must draw them this much bigger than the richer chrome... */
const COMPACT_PAYS = 1.4;
/** ...and the richer chrome's smallest enemy must be under this, real px. */
const COMPACT_SMALLEST = 40;

interface ModeLayout {
  scale: number;
  backCount: number;
  enemies: SplitPlace[];
  slots: number[];
  /** The placement fits this mode's constraints (false: nothing did). */
  ok: boolean;
  /** Real px: the intent headroom and the chrome room under the feet it keeps. */
  intentReal: number;
  belowReal: number;
  overReal: number;
}

/** `v` when it is a finite number above 0, else `fallback`. */
const positive = (v: number, fallback: number) => (Number.isFinite(v) && v > 0 ? v : fallback);

/** Clamp every input to something the solver can divide by: a non-finite or
 *  non-positive `k`, stage size, crown or width becomes 1 (a crown or width of
 *  one scene px; `k` 1 = design px are real px), a bad `rowScale` 0. */
function sanitize(inp: SplitInput): SplitInput {
  return {
    ...inp,
    stageW: positive(inp.stageW, 1),
    stageH: positive(inp.stageH, 1),
    k: positive(inp.k, 1),
    refCrown: positive(inp.refCrown, 1),
    rowScale: Number.isFinite(inp.rowScale) && inp.rowScale > 0 ? inp.rowScale : 0,
    enemies: inp.enemies.map((e) => ({
      crownH: positive(e.crownH, 1),
      footW: positive(e.footW, 1),
      ...(Number.isFinite(e.drawW) && (e.drawW as number) > 0 ? { drawW: e.drawW } : {}),
      ...(e.noColumn ? { noColumn: true } : {}),
      ...(Number.isFinite(e.inkW) && (e.inkW as number) > 0
        ? { inkW: e.inkW } : {}),
    })),
  };
}

/** The best scale, ranks and placements for one chrome mode. */
function solveMode(inp: SplitInput, mode: PowerChrome): ModeLayout {
  const { stageW, stageH, k, small, enemies } = inp;
  const c = CHROME[mode];
  const m = SPLIT.margin / k;
  const gap = SPLIT.gap / k;
  const intent = splitIntentReal(small) / k;
  const below = splitChromeBelow(mode, small) / k;
  const minSlot = c.slot / k;
  const side = c.side / k;
  const floor = stageH - (splitBandReal(small) + riseOf(inp)) / k - below;

  const slotW = (e: SplitCreature, s: number) => Math.max(bodyW(e) * s, minSlot) + side;
  const rowW = (idx: number[], s: number) =>
    idx.reduce((a, i) => a + slotW(enemies[i], s), 0) + gap * Math.max(0, idx.length - 1);
  const tallest = (idx: number[], s: number) => Math.max(0, ...idx.map((i) => enemies[i].crownH * s));
  /** How far a rank's tap boxes reach below its feet (a body drawn under
   *  `SPLIT.tap` less its HP bar): its chrome hangs that much lower. The
   *  large phone draws no body that short at a fitting scale, so it is left
   *  out there and the large phone's layouts stay as they were. */
  const tapUnder = (idx: number[], s: number) => !small ? 0 :
    Math.max(0, (SPLIT.tap - SPLIT.tapHp) / k - Math.min(...idx.map((i) => enemies[i].crownH * s)));

  const place = (s: number, back: number) => {
    const n = enemies.length;
    const backIdx = Array.from({ length: back }, (_, i) => i);
    const front = Array.from({ length: n - back }, (_, i) => back + i);
    const out: SplitPlace[] = new Array<SplitPlace>(n);
    const fFoot = floor - tapUnder(front, s);
    const frontTop = fFoot - tallest(front, s) - intent;
    let ok = rowW(front, s) <= stageW - 2 * m && frontTop >= m;
    const lay = (idx: number[], foot: number) => {
      let x = m + Math.max(0, (stageW - 2 * m - rowW(idx, s)) / 2);
      for (const i of idx) {
        const w = slotW(enemies[i], s);
        // The power column stands on the body's LEFT.
        out[i] = { x: x + side + (w - side) / 2, foot };
        x += w + gap;
      }
    };
    lay(front, fFoot);
    if (back > 0) {
      const bFoot = frontTop - below - tapUnder(backIdx, s);
      ok = ok && rowW(backIdx, s) <= stageW - 2 * m && bFoot - tallest(backIdx, s) - intent >= m;
      lay(backIdx, bFoot);
    }
    return { ok, out };
  };

  const cap = Math.max(
    inp.rowScale,
    (floor - intent - m) / Math.max(inp.refCrown, ...enemies.map((e) => e.crownH)),
  );
  /** The largest scale <= cap that fits, never under the row's own. */
  const largest = (back: number) => {
    if (place(cap, back).ok) return cap;
    let lo = 0;
    let hi = cap;
    for (let i = 0; i < STEPS; i++) {
      const mid = (lo + hi) / 2;
      if (place(mid, back).ok) lo = mid;
      else hi = mid;
    }
    return lo;
  };

  let best = { s: -1, back: 0 };
  // A power column grows up beside the body, into whatever stands behind it:
  // a column chrome has no back rank.
  const ranks = mode === "column" ? 1 : Math.max(1, enemies.length);
  for (let back = 0; back < ranks; back++) {
    const s = largest(back);
    if (best.s < 0 || s > best.s * BACK_PAYS) best = { s, back };
  }
  const placed = place(best.s, best.back);
  const allowed = mode !== "column" || !enemies.some((e) => e.noColumn);
  const bigEnough = Math.max(inp.refCrown, ...enemies.map((e) => e.crownH)) * best.s * k >= SPLIT.minCrown;
  return { scale: best.s, backCount: best.back, enemies: placed.out, ok: allowed && bigEnough && placed.ok && best.s > 0,
    slots: enemies.map((e) => slotW(e, best.s)),
    intentReal: splitIntentReal(small), belowReal: splitChromeBelow(mode, small), overReal: 0 };
}

/** The last resort when no mode fits: one compact rank of `n` equal slots
 *  across the stage, the bodies scaled to fit their slot and the height
 *  left above the band. On a stage too short for the full overheads it
 *  gives up, in turn and only as far as it must to draw the tallest at
 *  `SPLIT.minCrown` (in proportion if it is smaller than the player, or at
 *  its slot's width, if that is smaller): the compact chrome under the feet
 *  (the fight is drawn bare, `belowReal` under `splitChromeBelow`), the top
 *  margin, then crown height. The intent and the HP bar always stay on the
 *  stage and above the band; only a stage shorter than the band plus an
 *  intent puts anything over the band. */
function cramped(inp: SplitInput): ModeLayout {
  const { stageW, stageH, k, small, enemies } = inp;
  const n = Math.max(1, enemies.length);
  const m = SPLIT.margin / k;
  const inner = stageW - 2 * m > 0 ? stageW - 2 * m : stageW;
  const left = (stageW - inner) / 2;
  const pitch = inner / n;
  const gap = Math.min(SPLIT.gap / k, pitch / 2);
  const widest = Math.max(1, ...enemies.map(bodyW));
  const tallest = Math.max(1, ...enemies.map((e) => e.crownH));
  const byWidth = (pitch - gap) / widest;
  // The lowest a foot may stand: the band's top, less Osty's rise.
  const lowest = stageH - (splitBandReal(small) + riseOf(inp)) / k;
  /** The tallest's crown, design px, worth giving up overheads for:
   *  `SPLIT.minCrown` for a player-sized one, in proportion for a smaller. */
  const want = Math.min(byWidth * tallest, (SPLIT.minCrown / k) * (tallest / Math.max(inp.refCrown, tallest)));
  const intentFull = splitIntentReal(small) / k;
  const below = splitChromeBelow("compact", small) / k;
  const smallest = Math.max(1e-9, Math.min(...enemies.map((e) => e.crownH)));
  /** How far the tap boxes reach below the feet at scale `sc`. */
  const tapUnder = (sc: number) => Math.max(0, (SPLIT.tap - SPLIT.tapHp) / k - smallest * sc);
  const intent = intentFull;
  let top = m;
  const fit = (f: number) => {
    const room = f - intent - top;
    return room > 0 ? Math.min(byWidth, room / tallest) : 0;
  };
  // Every overhead kept: the feet over the chrome and the tap boxes.
  let foot = lowest - below;
  for (let i = 0; i < 4; i++) foot = lowest - below - tapUnder(fit(foot));
  // Too short: give up the chrome under the feet (Board draws the fight
  // bare, `gs-split-bare`: no powers, no names, and no tap box hanging under
  // a small body, whose 44 px intent chip is its tap target then) down to
  // the HP bar, then the top margin, then Osty's rise, then let the HP bars
  // come down over the band's upper part (an enemy the player cannot see is
  // worse than one over the figure), never past the figure's feet (the
  // controls row stays clear), and only then the crown. Never the intent:
  // its chip is drawn over the crown however little room the layout keeps.
  let bare = false;
  const hp = SPLIT.tapHp / k;
  if (foot - intent - top < want) {
    bare = true;
    foot = Math.min(lowest - hp, top + intent + want);
  }
  if (foot - intent - top < want) top = Math.min(m, Math.max(0, foot - intent - want));
  const bandTop = stageH - splitBandReal(small) / k;
  if (foot - intent - top < want) foot = Math.max(foot, Math.min(bandTop - hp, top + intent + want));
  const deepest = stageH - (PHONE_HAND_SHOW + SPLIT.handClear + SPLIT.controls + SPLIT.bandPad) / k - hp;
  if (foot - intent - top < want) foot = Math.max(foot, Math.min(deepest, top + intent + want));
  // A stage shorter than the band plus an intent: a one-real-px speck whose
  // intent stands on the stage's top, over the band.
  const scale = fit(foot) > 0 ? fit(foot) : 1 / k / tallest;
  foot = Math.min(stageH, Math.max(foot, intent + tallest * scale));
  return {
    scale,
    backCount: 0,
    enemies: enemies.map((_, i) => ({ x: left + pitch * (i + 0.5), foot })),
    slots: enemies.map(() => pitch - gap),
    ok: true,
    intentReal: intent * k,
    belowReal: (bare ? Math.max(0, lowest - foot) : below) * k,
    overReal: Math.max(0, foot + hp - lowest) * k,
  };
}

/** Solve the split for one fight. */
export function splitLayout(raw: SplitInput): SplitLayout {
  const inp = sanitize(raw);
  const { k, small, stageH } = inp;
  const fitted = (l: ModeLayout) => (l.ok ? l.scale : 0);
  const rows = solveMode(inp, "rows");
  const column = solveMode(inp, "column");
  const compact = solveMode(inp, "compact");
  let chrome: PowerChrome = fitted(column) > fitted(rows) * COLUMN_PAYS ? "column" : "rows";
  const richer = chrome === "column" ? column : rows;
  const minCrown = Math.min(...inp.enemies.map((e) => e.crownH));
  // A body of targetable parts (the Decimillipede, the Kaiser Crab) is wide
  // and low: it can be far too small across while still `COMPACT_SMALLEST`
  // tall (160 of 375 px on a 496 px stage), so for it the scale alone decides.
  const parts = inp.enemies.some((e) => e.noColumn);
  if (fitted(compact) > fitted(richer) * COMPACT_PAYS
    && (parts || fitted(richer) * k * minCrown < COMPACT_SMALLEST)) {
    chrome = "compact";
  }
  let pick = chrome === "column" ? column : chrome === "compact" ? compact : rows;
  const isCramped = !pick.ok;
  if (isCramped) {
    chrome = "compact";
    pick = cramped(inp);
  }
  return {
    scale: pick.scale,
    chrome,
    backCount: pick.backCount,
    enemies: pick.enemies,
    slots: pick.slots,
    bandTop: stageH - splitBandReal(small) / k,
    figureFoot: stageH - (PHONE_HAND_SHOW + SPLIT.handClear + SPLIT.controls + SPLIT.bandPad) / k,
    figureReal: figureOf(small),
    cramped: isCramped,
    intentReal: pick.intentReal,
    belowReal: pick.belowReal,
    overReal: pick.overReal,
  };
}

/** The stage height to solve at, design px, given the height the split was
 *  last solved at (`held`, null for none) and the stage's height now (`h`).
 *  A change under `SPLIT.margin` real px keeps `held`: a tool row wrapping or
 *  the browser's bar moving must not rescale the enemies mid-fight. The
 *  threshold is the solver's top margin, so the held solve, shifted to the
 *  real stage (`shiftSplit`), never lifts an intent off the stage's top. A
 *  rotation or a larger change re-solves. */
export function heldSplitHeight(held: number | null, h: number, k: number): number {
  if (held == null || !Number.isFinite(held)) return h;
  return Math.abs(h - held) * positive(k, 1) < SPLIT.margin ? held : h;
}

/** A layout solved for a stage `dy` design px shorter, moved down onto this
 *  one. Every vertical position the solver returns is measured from the
 *  stage top and moves one-for-one with its height at a fixed scale (the
 *  floor, the band and the figure stand on the stage's bottom), so this is the
 *  layout of the taller stage at the held scale. */
export function shiftSplit(l: SplitLayout, dy: number): SplitLayout {
  if (dy === 0) return l;
  return {
    ...l,
    enemies: l.enemies.map((e) => ({ ...e, foot: e.foot + dy })),
    bandTop: l.bandTop + dy,
    figureFoot: l.figureFoot + dy,
  };
}
