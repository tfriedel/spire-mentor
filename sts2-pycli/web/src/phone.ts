import { createContext, useContext } from "react";
import {
  PHONE_LANDSCAPE_TOUCH_QUERY,
  PHONE_PORTRAIT_QUERY,
  useMediaQuery,
} from "./useMediaQuery";

/** How the workbench is being held, or `null` when it is not a phone at all. */
export type PhoneMode = "portrait" | "landscape" | null;

/** Phone mode from the live media queries. Portrait wins when both match (a
 *  narrow AND short viewport is a phone upright in a split view; the portrait
 *  layout is the one that copes with the narrower axis). */
export function usePhone(): PhoneMode {
  const portrait = useMediaQuery(PHONE_PORTRAIT_QUERY);
  const landscape = useMediaQuery(PHONE_LANDSCAPE_TOUCH_QUERY);
  return portrait ? "portrait" : landscape ? "landscape" : null;
}

/** The stage is drawn in design px and scaled by `k` (`--gs-k`) to fit the
 *  screen, so a size that must be `realPx` on the glass is `realPx / k` in
 *  design px. */
export function floorPx(realPx: number, k: number): number {
  return realPx / k;
}

/** A design-px size that never renders smaller than `realPx` on the glass.
 *  At desktop zoom (k near 1) the design size wins. */
export function atLeast(designPx: number, realPx: number, k: number): number {
  return Math.max(designPx, floorPx(realPx, k));
}

/** The stage's zoom (`--gs-k`), for children that size themselves in design px
 *  and need a real-px floor (see `atLeast`). Board provides it; outside a
 *  board it is 1, i.e. design px are real px. */
export const StageZoomContext = createContext(1);
export function useStageZoom(): number {
  return useContext(StageZoomContext);
}

/** How much of a resting hand card stands above the stage's bottom edge on a
 *  phone, in real px: the strip the hand owns (see `handFan.handFloors`). */
export const PHONE_HAND_SHOW = 96;

/** Portrait: real px from a fighter's ground line down to the bottom of its
 *  power block at the cap of two rows (HP bar, nameplate, two 44px power rows), plus a
 *  few px of air. Constant because every one of those is floored in real px
 *  on a phone. */
export const PHONE_BELOW_GROUND = 130;

/** Landscape: real px from the ground line to the bottom of the one-line
 *  nameplate (HP bar ~14, nameplate gap and line ~15 at the 0.42 zoom floor),
 *  plus air. The powers stand in a column BESIDE the creature in landscape
 *  (powerColumn.ts), so nothing else hangs under the feet. */
export const PHONE_LANDSCAPE_BELOW_GROUND = 36;

/** Real px of the chrome row that sits on top of the hand strip in portrait
 *  (End Turn plate, piles, energy orb — phone.css), with its gaps. */
export const PHONE_PORTRAIT_CHROME = 80;

/** Portrait only: the fighters stand close enough that a long name wraps onto
 *  a second line (`Shrinker Beetle #2`), which pushes that creature's power
 *  block ~15 real px further down. */
export const PHONE_PORTRAIT_NAME_WRAP = 20;

/** The lowest the fighters' ground line may sit on a phone, in design px from
 *  the top of a `stageH`-tall (design px) stage. Portrait: a two-row power
 *  block under any fighter ends above the hand strip and the chrome row
 *  standing on it, where the power chips stay tappable. Landscape: only the HP
 *  bar and the nameplate hang under the feet (the powers stand beside the
 *  creature), so the fighters are drawn as large as the stage allows. `null`
 *  off a phone. */
export function phoneGroundLimit(mode: PhoneMode, stageH: number, k: number): number | null {
  if (mode == null) return null;
  const below = PHONE_HAND_SHOW + (mode === "portrait"
    ? PHONE_BELOW_GROUND + PHONE_PORTRAIT_CHROME + PHONE_PORTRAIT_NAME_WRAP
    : PHONE_LANDSCAPE_BELOW_GROUND);
  return stageH - floorPx(below, k);
}

/** The small phone (iPhone SE class, 375 CSS px across) by the SHORT side of
 *  the screen: the same answer upright and sideways. The large phone is 412
 *  across; 390 splits them. One predicate for every phone size rule (the
 *  hand's card floor, the zoom's width). */
export const SMALL_PHONE_SHORT_SIDE = 390;
export function isSmallPhone(w: number, h: number): boolean {
  return Math.min(w, h) < SMALL_PHONE_SHORT_SIDE;
}

/** `isSmallPhone` for this device. The SCREEN, not the window: held sideways
 *  the window's short side is the screen's less the browser's status and URL
 *  bars (412 becomes ~330 on a large Android), which would call every phone
 *  small. Falls back to the window where the screen reports nothing. */
export function isSmallPhoneScreen(): boolean {
  const s = typeof screen === "undefined" ? null : screen;
  const w = s?.width || window.innerWidth;
  const h = s?.height || window.innerHeight;
  return isSmallPhone(w, h);
}
