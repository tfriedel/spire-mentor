import { CARD_COST_PROUD, CARD_H, CARD_W } from "./cardGeometry";
import type { PhoneMode } from "./phone";
import { splitZoomWidth } from "./portraitSplit";

/** Where the zoomed card goes on a phone.
 *
 *  ONE fixed spot per orientation, the same for every card (attack or skill,
 *  any number of enemies): the owner found a zoom that moved from card to card
 *  hard to read. Everything is in REAL px (viewport coordinates,
 *  `getBoundingClientRect`).
 *
 *  - Portrait (spec "Phase 2: portrait split", "The zoomed card"):
 *    horizontally centred on the stage (`portraitZoomLeft`; never closer to
 *    the left edge than `portraitLeft`: 8 px, or more so every cost icon
 *    stays on the stage), its bottom `HAND_OVERLAP` into the hand strip, i.e. over
 *    the player's band, never over the enemies across the top of the stage.
 *    `splitZoomWidth` wide (158, small phone 150): the split solver sized the
 *    band (and the rise above it) for exactly this card. It never shrinks for
 *    an enemy (the next 10 px step is under `MIN_WIDTH`); should an enemy
 *    ever stand in the spot it stays put and reports `covers`.
 *  - Landscape: on the player's (left) side, `LANDSCAPE_LEFT` from the stage's
 *    left edge (clear of the energy orb's column, see `landscapeLeft`), top
 *    edge `TOP_MARGIN` below the stage top. The stage is too short for the card
 *    to stay above the hand strip, so it hangs over the strip; that part passes
 *    taps through to the slots (`zoomHitHeight`).
 *
 *  Width: `splitZoomWidth` portrait, 170 landscape, 160 on the small phone in
 *  landscape. It shrinks only
 *  (a) on a stage too short for it (portrait: the card's bottom may reach
 *  `HAND_OVERLAP` into the hand strip; landscape: the stage bottom), and
 *  (b) landscape only in practice, LAST, when the fixed spot would cover an
 *  enemy's body (which an armed attack's next tap must reach): by 10s down to
 *  `MIN_WIDTH`, its top-left corner never moving. If even that covers one it
 *  stays at `MIN_WIDTH`, covering it (`covers`). Intent chips and power rows
 *  may be covered there: they are inspect targets, back as soon as the zoom
 *  is closed. */

export type Rect = { left: number; top: number; right: number; bottom: number };

export type ZoomArgs = {
  mode: PhoneMode;
  stage: Rect;
  enemies: Rect[];
  hand: Rect;
  small: boolean;
  /** Landscape: the energy orb, whose column the card stands right of. */
  orb?: Rect | null;
};

export type ZoomPlacement = {
  left: number; top: number; width: number;
  /** How much of the card, from its top edge, takes taps (real px). Unset:
   *  all of it. Set in landscape, where the card hangs over the hand strip:
   *  the part over the strip passes taps through to the slots under it
   *  (`zoomHitHeight`). */
  hit?: number;
  /** The spot covers an enemy even at `MIN_WIDTH`. */
  covers?: boolean;
};

/** How far the zoom may hang over the resting hand strip in portrait. */
export const HAND_OVERLAP = 24;
/** Real px between the stage top and the card's top edge. Room for the cost
 *  gem (it stands ~0.06 of the width proud of the edge, 11px at 190) and for
 *  the close button, which straddles the card's top-right corner. */
export const TOP_MARGIN = 12;
/** Landscape: the card's left edge, in real px from the stage's left edge,
 *  at the least (`landscapeLeft`). */
export const LANDSCAPE_LEFT = 76;
/** Portrait: the card's left edge, real px from the stage's left edge. */
export const PORTRAIT_LEFT = 8;

/** Portrait: the card's left edge from the stage's left edge at width `w`:
 *  `PORTRAIT_LEFT`, or the furthest cost icon's overhang rounded up where
 *  that is more. That is the Regent's star (36/300 of the width: 19 at 158,
 *  18 at 150), not the energy gem (16/300): ONE spot for every card, so a
 *  star-cost card keeps its star on the stage and no card moves for it. The
 *  band's status block reads it too, to stand right of the card. */
export function portraitLeft(w: number): number {
  return Math.max(PORTRAIT_LEFT, Math.ceil(w * CARD_COST_PROUD));
}

/** Portrait: the card's left edge from the stage's left edge on a stage
 *  `stageW` real px wide: horizontally CENTRED (the owner found a card that
 *  opened at the left edge unintuitive), a little left of centre where the
 *  status block and End Turn need the room (`ZOOM_STATUS_ROOM`), never left
 *  of `portraitLeft`. */
export function portraitZoomLeft(stageW: number, w: number): number {
  return Math.max(portraitLeft(w), Math.min((stageW - w) / 2, stageW - w - ZOOM_STATUS_ROOM));
}
/** Real px the card leaves free on its right: End Turn (124 px wide, 8 px
 *  off the stage's edge) plus a 2 px gap, which also holds the player's
 *  status block (Board's `splitBand`: an 8 px gap, the block shield's 17 px
 *  overhang, one 44 px power cell, 4 px, the 44 px energy orb and an 8 px
 *  gutter). On a 412 px stage the 158 px card stands 7 px left of centre, on
 *  a 375 px one the 150 px card 21.5 px. */
export const ZOOM_STATUS_ROOM = 134;
/** The status block's right gutter while a card is zoomed. */
export const ZOOM_GUTTER = 8;
/** The narrowest zoom: the card's title is still 13px on the glass. */
export const MIN_WIDTH = 150;
const STEP = 10;

/** Phase 1 spec widths: 190 portrait, 170 landscape, 160 on a small phone.
 *  The portrait split caps its own width (`splitZoomWidth`) at this. */
export function zoomWidth(mode: PhoneMode, small: boolean): number {
  if (small) return 160;
  return mode === "landscape" ? 170 : 190;
}

/** The card's height, plus a px of slack: CardFrame rounds its height in design
 *  px, which can land up to a real px taller than the exact ratio. */
const hOf = (w: number) => (w * CARD_H) / CARD_W + 1;
const overlaps = (a: Rect, b: Rect) =>
  a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;

/** The lowest the card's bottom edge may go. */
const floorY = (a: ZoomArgs) =>
  a.mode === "landscape" ? a.stage.bottom : Math.min(a.stage.bottom, a.hand.top + HAND_OVERLAP);

/** Landscape: the card's left edge, right of the energy orb's column. */
export function landscapeLeft(a: Pick<ZoomArgs, "stage" | "orb">): number {
  const orbRight = a.orb && a.orb.right > a.orb.left ? a.orb.right + 8 : -Infinity;
  return Math.max(a.stage.left + LANDSCAPE_LEFT, orbRight);
}

/** The widest card the stage holds on the fixed spot: landscape, from the
 *  fixed top edge down; portrait, from the floor up to `TOP_MARGIN` under the
 *  stage top. */
export function fitWidth(a: ZoomArgs): number {
  const room = floorY(a) - (a.stage.top + TOP_MARGIN);
  const across = a.mode === "landscape"
    ? a.stage.right - landscapeLeft(a)
    : a.stage.right - a.stage.left - 2 * PORTRAIT_LEFT;
  return Math.max(0, Math.floor(Math.min(across, (room - 1) * (CARD_W / CARD_H))));
}

/** The card's rect at width `w` on the fixed spot. */
function at(a: ZoomArgs, w: number): ZoomPlacement {
  if (a.mode === "landscape") return { left: landscapeLeft(a), top: a.stage.top + TOP_MARGIN, width: w };
  return { left: a.stage.left + portraitZoomLeft(a.stage.right - a.stage.left, w), top: floorY(a) - hOf(w), width: w };
}

const coversFoe = (a: ZoomArgs, p: ZoomPlacement) => {
  const r = { left: p.left, top: p.top, right: p.left + p.width, bottom: p.top + hOf(p.width) };
  return a.enemies.some((e) => overlaps(r, e));
};

export function zoomPlacement(a: ZoomArgs): ZoomPlacement {
  const want = a.mode === "landscape" ? zoomWidth(a.mode, a.small) : splitZoomWidth(a.small);
  const w0 = Math.min(want, fitWidth(a));
  let p = at(a, w0);
  for (let w = w0 - STEP; coversFoe(a, p) && w >= MIN_WIDTH; w -= STEP) p = at(a, w);
  return coversFoe(a, p) ? { ...p, covers: true } : p;
}

/** The tappable height of a zoom whose top edge is `top` (real px, same
 *  origin as `handTop`): in landscape only the part ABOVE the hand strip,
 *  so a tap on the strip under the card reaches the slot there (which only
 *  selects, never plays). Portrait: undefined, the whole card. */
export function zoomHitHeight(mode: PhoneMode, top: number, handTop: number): number | undefined {
  if (mode !== "landscape") return undefined;
  return Math.max(0, handTop - top);
}
