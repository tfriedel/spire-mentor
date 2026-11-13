// The Decimillipede's segment boxes in rig units — layout math kept out of the
// component file so it exports only the component (react-refresh/only-export-
// components).
import { rigBox } from "../rigBounds";

/** A segment's mirrored box in rig units, relative to the shared root. */
export function mirroredBox(rig: string | null | undefined) {
  const box = rigBox(rig);
  if (!box) return null;
  // flipX mirrors about the root: [x, x+w] becomes [-(x+w), -x].
  return { left: -(box.x + box.w), right: -box.x, w: box.w, h: box.h, bottom: box.y };
}

/** The whole animal's box in rig units, relative to the shared root — what the
 *  scene layout spaces the elite by. */
export function decimillipedeSpan(rigs: (string | null | undefined)[]): { left: number; right: number } {
  let left = 0, right = 0, any = false;
  for (const rig of rigs) {
    const b = mirroredBox(rig);
    if (!b) continue;
    left = any ? Math.min(left, b.left) : b.left;
    right = any ? Math.max(right, b.right) : b.right;
    any = true;
  }
  return any ? { left, right } : { left: -600, right: 600 };
}

/** How far the animal's legs reach below the shared root, rig units (0 when
 *  no rig has measured bounds). */
export function decimillipedeDrop(rigs: (string | null | undefined)[]): number {
  return Math.max(0, ...rigs.map((rig) => -(mirroredBox(rig)?.bottom ?? 0)));
}
