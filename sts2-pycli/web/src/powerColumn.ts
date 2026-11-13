/** Phone landscape: a creature's powers stand in a COLUMN beside it rather than
 *  in rows under it (the stage is ~280-310 real px tall, and two 44px rows
 *  under every fighter left the fighters ~20-60px tall). How many cells that
 *  column gets.
 *
 *  Pure geometry in real px, so `Powers` can measure the stage and ask. */

export type Rect = { left: number; right: number; top: number; bottom: number };

/** What the column is sized for before it has measured: three powers and the
 *  "+N" chip (spec: at least three powers before collapsing). It then takes
 *  every cell the room up to the stage top gives it. */
export const COLUMN_WANT = 4;

/** And never fewer than this — one power and the "+N" chip — even where a
 *  neighbour's intent or the stage top crowds it. */
export const COLUMN_MIN = 2;

/** Of several places a column could stand (`left` of each), the index of the
 *  one with the most room; the first wins ties. */
export function bestColumnPlace(rooms: readonly number[]): number {
  let best = 0;
  rooms.forEach((r, i) => { if (r > rooms[best]) best = i; });
  return best;
}

/**
 * Cells in a bottom-anchored column of `cell`-px boxes `gap` apart, whose
 * bottom edge is `bottom` and whose x-range is `left..right`.
 *
 * - It reaches up to `ceiling` (the stage's top edge, less a margin), past the
 *   creature's own top: a short creature still gets three powers and the chip
 *   where the stage has the room;
 * - and never into an obstacle (another creature's intent, the energy orb)
 *   whose x-range overlaps the column's: the column stops `gap` under it.
 *   An obstacle reaching below `bottom` cannot be cleared by shortening: no
 *   room at all (`columnRoom` 0), left to the floor.
 *
 * Clamped to `COLUMN_MIN`.
 */
export type ColumnRoom = {
  bottom: number; cell: number; gap: number; ceiling: number;
  left: number; right: number; obstacles: readonly Rect[];
};

export function columnCells(o: ColumnRoom): number {
  return Math.max(COLUMN_MIN, columnRoom(o));
}

/** `columnCells` before the `COLUMN_MIN` clamp: 0 or 1 means the column does
 *  not fit there without covering something. For choosing between places. */
export function columnRoom(o: ColumnRoom): number {
  let top = o.ceiling;
  // (An obstacle reaching below the column's bottom leaves no room at all.)
  for (const r of o.obstacles) {
    if (r.right <= o.left || r.left >= o.right) continue; // beside the column
    if (r.top >= o.bottom) continue; // under it
    if (r.bottom > o.bottom) return 0; // beside the column's bottom: unclearable
    top = Math.max(top, r.bottom + o.gap);
  }
  return Math.max(0, Math.floor((o.bottom - top + o.gap) / (o.cell + o.gap) + 1e-6));
}

/** Where a column's bottom edge goes: on `bottom` (the ground line), unless a
 *  NEIGHBOUR's HP bar or name (`floors`) hangs across the column's x-range
 *  above that line — in a crowded row every second creature stands higher, and
 *  its HP bar and name reach beside the next creature's body. Then the column
 *  stands on top of the highest such floor, `gap` above it. */
export function columnBottom(bottom: number, left: number, right: number, gap: number,
  floors: readonly Rect[]): number {
  let b = bottom;
  for (const r of floors) {
    if (r.right <= left || r.left >= right) continue;
    if (r.top >= bottom || r.bottom <= r.top) continue;
    b = Math.min(b, r.top - gap);
  }
  return b;
}
