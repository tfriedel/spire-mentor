/** How a creature's buff/debuff icons pack into rows — `NPowerContainer
 *  .UpdatePositions`, ported.
 *
 *  Lives outside `Powers.tsx` for the same reason `cardGeometry.ts` lives
 *  outside `CardFrame.tsx`: it is a layout rule with its own tests, and the
 *  component file stays a component file (Fast Refresh).
 */

/** Gap between icon cells. The game's cells are flush (it steps by the icon's
 *  own width); ours are small enough that they need the air. */
export const POWER_GAP = 5;

/** The row width to assume when the caller has no creature box to give — six
 *  26px icons and their gaps, which is what the old flex cap allowed. Reached
 *  by a combatant rendered outside the arena (no `pxPerScene`) and by tests. */
export const DEFAULT_POWER_BOX = 190;

/** `NPowerContainer.SetCreatureBounds`'s `+ 25f`: the row may run this much
 *  wider than the creature's authored box before it wraps. Scene px. */
export const POWER_BOX_MARGIN = 25;

/**
 * How many icons one row holds. `NPowerContainer.UpdatePositions`, verbatim:
 *
 *     num = CeilToInt(Size.X / x)                 // x = one icon's width
 *     num = Max(CeilToInt(count / 2), num)
 *     position(i) = (x * (i % num), floor(i / num) * x)
 *
 * The second line is the load-bearing one: a row is never narrower than HALF
 * the powers, so the block is **at most two rows** for any number of them. Past
 * the creature's own width the game widens the row rather than starting a third
 * — powers grow sideways, into the room, never down into the hand strip.
 *
 * Ours divides by the GAPPED cell (`size + POWER_GAP`) and floors, which is the
 * same question asked of a row that carries gaps: how many whole cells fit in
 * `boxWidth`.
 */
export function powerColumns(count: number, boxWidth: number, size: number): number {
  const fit = Math.floor((boxWidth + POWER_GAP) / (size + POWER_GAP));
  return Math.max(1, Math.ceil(count / 2), fit);
}

/** Phone: whole tap-box cells that fit `boxWidth`, never fewer than 2 (so three
 *  powers and the "+N" chip always fit as 2x2). Only that clamp lets a block
 *  run wider than a very narrow creature's box. */
export function powerFit(boxWidth: number, cell: number): number {
  return Math.max(2, Math.floor((boxWidth + POWER_GAP) / (cell + POWER_GAP)));
}
