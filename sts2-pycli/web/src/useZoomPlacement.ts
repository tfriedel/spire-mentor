import { useLayoutEffect, useState, type RefObject } from "react";
import { isSmallPhoneScreen, type PhoneMode } from "./phone";
import { zoomHitHeight, zoomPlacement, type Rect, type ZoomPlacement } from "./zoomPlacement";

const rectOf = (el: Element): Rect => {
  const b = el.getBoundingClientRect();
  return { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
};
const real = (r: Rect) => r.right - r.left > 0 && r.bottom - r.top > 0;

/** Each enemy's BODY: what an armed attack's next tap aims at. Its intent chip
 *  (which bobs, so it measures differently from frame to frame) and power rows
 *  are inspect targets; the zoom may cover them, and the close button or a
 *  tap on empty stage brings them back. In portrait the body is the drawn rig
 *  and its HP bar: the enemy's box also holds its name, whose foot the small
 *  phone's zoom may cover (the split's floor keeps only the HP bar above the
 *  zoom's rise, `splitChromeBelow`). */
function enemyRects(root: ParentNode, mode: PhoneMode): Rect[] {
  const sel = mode === "portrait"
    ? "[data-enemy-index] > .gs-sprite-wrap, [data-enemy-index] > .gs-hpwrap" : "[data-enemy-index]";
  return Array.from(root.querySelectorAll(sel)).map(rectOf).filter(real);
}

/** Measure the stage and run `zoomPlacement`. Returns the placement in REAL px
 *  relative to the stage's top-left, or null when there is nothing to place.
 *  The spot depends on the stage alone (never on which card is zoomed), so a
 *  switch from one card to another leaves it where it is. Re-measured when the
 *  stage resizes (a rotation) and once more on the next frame, after the
 *  fighters have been re-fitted. */
export function useZoomPlacement(
  stageRef: RefObject<HTMLElement | null>,
  active: boolean,
  mode: PhoneMode,
  k: number,
): ZoomPlacement | null {
  const [p, setP] = useState<ZoomPlacement | null>(null);
  useLayoutEffect(() => {
    const stageEl = stageRef.current;
    if (!active || mode == null || !stageEl) { setP(null); return; }
    const measure = () => {
      const stage = rectOf(stageEl);
      const handEl = stageEl.querySelector(".gs-hand");
      if (!handEl) return;
      const hand = rectOf(handEl);
      const orbEl = stageEl.querySelector(".gs-energy");
      const z = zoomPlacement({
        mode, stage, enemies: enemyRects(stageEl, mode), hand, small: isSmallPhoneScreen(),
        orb: orbEl ? rectOf(orbEl) : null,
      });
      const next = { left: z.left - stage.left, top: z.top - stage.top, width: z.width,
        hit: zoomHitHeight(mode, z.top, hand.top), ...(z.covers ? { covers: true } : {}) };
      setP((prev) => (prev && Math.abs(prev.left - next.left) < 0.5 && Math.abs(prev.top - next.top) < 0.5
        && prev.width === next.width && prev.covers === next.covers && Math.abs((prev.hit ?? -1) - (next.hit ?? -1)) < 0.5 ? prev : next));
    };
    measure();
    let raf = requestAnimationFrame(() => { raf = requestAnimationFrame(measure); });
    const late = window.setTimeout(measure, 300);
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => {
      measure();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(measure);
    });
    ro?.observe(stageEl);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(late);
      ro?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [stageRef, active, mode, k]);
  return active && mode != null ? p : null;
}
