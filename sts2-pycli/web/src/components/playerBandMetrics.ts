// The phone-portrait player band's arithmetic, kept out of PlayerBand.tsx so
// that file exports only the component (react-refresh/only-export-components).
import { POWER_GAP } from "../powerRow";

/** A power cell in the band: a 44 px tap box. */
export const BAND_CELL = 44;
/** An orb in the band's row, and the gap after each, while they fit. */
export const BAND_ORB = 30;
export const BAND_ORB_GAP = 4;
/** The smallest orb the row draws (a Defect with ten slots on a small phone). */
export const BAND_ORB_MIN = 20;
/** The smallest an orb gets on a line too narrow for the floor (the status
 *  block beside a zoomed card, which is on screen only while it is open). */
export const BAND_ORB_TINY = 8;
/** The energy orb's size in the band while a card is zoomed, and its gap. */
export const BAND_ENERGY = 44;
/** The narrowest status block beside a zoomed card: one power cell, a gap
 *  and the energy orb. */
export const BAND_ZOOM_MIN = BAND_CELL + 4 + BAND_ENERGY;
/** The tap area around the band's 34 px block shield. */
export const BAND_SHIELD_HIT = 44;
/** Icon size inside a band cell: the phone floor for a power icon. */
export const BAND_ICON = 28;

export interface BandOrbRow { size: number; gap: number; width: number }

/** `n` orbs packed into `budget` px: 30 px orbs with 4 px gaps, shrinking to
 *  the 20 px floor, and below that the gaps close up. */
function packOrbs(budget: number, n: number, tinyOk = false): BandOrbRow {
  const fit = Math.floor((budget + BAND_ORB_GAP) / n - BAND_ORB_GAP);
  const size = Math.max(BAND_ORB_MIN, Math.min(BAND_ORB, fit));
  const gap = size > BAND_ORB_MIN || n < 2 ? BAND_ORB_GAP
    : Math.max(0, Math.min(BAND_ORB_GAP, Math.floor((budget - n * size) / (n - 1))));
  // Narrower than the floor allows (ten orbs beside a zoomed card): the orbs
  // shrink under it rather than run off the stage.
  if (tinyOk && n * size > budget && budget > 0) {
    const tiny = Math.max(BAND_ORB_TINY, Math.floor(budget / n));
    return { size: tiny, gap: 0, width: n * tiny };
  }
  return { size, gap, width: n * size + (n - 1) * gap };
}

/** The band's orbs row: ONE line, never wider than the status block. It keeps
 *  room for one power cell (and its gap) beside it while the 20 px floor
 *  allows; past that the orbs take the whole line and the powers wrap to a
 *  line of their own. Narrower than `orbCount * 20` px nothing fits, and the
 *  row overflows at the floor (ten orbs need 200 px). `ownLine` (a zoomed
 *  card: the energy orb shares the powers' line): the orbs take their line
 *  whole and shrink under the floor to fit it (`BAND_ORB_TINY`). */
export function bandOrbRow(widthPx: number, orbCount: number, ownLine = false): BandOrbRow {
  if (orbCount <= 0) return { size: BAND_ORB, gap: BAND_ORB_GAP, width: 0 };
  if (ownLine) return packOrbs(widthPx, orbCount, true);
  const beside = packOrbs(widthPx - BAND_ORB_GAP - BAND_CELL, orbCount);
  if (beside.width <= widthPx - BAND_ORB_GAP - BAND_CELL) return beside;
  return packOrbs(widthPx, orbCount);
}

/** Does the orbs row leave a power cell room on its own line? */
function cellBeside(widthPx: number, row: BandOrbRow): boolean {
  return row.width + BAND_ORB_GAP + BAND_CELL <= widthPx;
}

/** How many power cells fit on the band's second line after the orbs row
 *  (`bandOrbRow`) and a 4 px gap; each cell is 44 px with a 5 px gap (none
 *  after the last). With 30 px orbs that is `floor((widthPx - 34 * orbs + 5)
 *  / 49)`. When the orbs fill their line the powers get the full width on the
 *  next. Always at least one, so a crowded line still has its "+N". */
export function bandPowerCells(widthPx: number, orbCount: number): number {
  const row = bandOrbRow(widthPx, orbCount);
  const used = orbCount > 0 && cellBeside(widthPx, row) ? row.width + BAND_ORB_GAP : 0;
  return Math.max(1, Math.floor((widthPx - used + POWER_GAP) / (BAND_CELL + POWER_GAP)));
}

/** Power cells on the line the energy orb shares with them (a zoomed card):
 *  what fits left of the orb and its gap. At least one. */
export function bandPowerCellsBesideEnergy(widthPx: number): number {
  return Math.max(1, Math.floor((widthPx - BAND_ENERGY - 4 + POWER_GAP) / (BAND_CELL + POWER_GAP)));
}
