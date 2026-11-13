import type { Power } from "./api";

/** The fewest cells a capped block may have: three icons and the "+N" chip. */
export const MIN_SHOWN = 4;

/** Split a creature's powers into what the capped block draws and what hides
 *  behind the "+N" chip. When anything is hidden the chip takes the last cell,
 *  so `shown.length === maxShown - 1`. `minShown` is the floor on the cap: a
 *  two-row block keeps 4, a landscape column may go down to 2 (powerColumn). */
export function splitPowers(
  powers: Power[], maxShown: number, minShown: number = MIN_SHOWN,
): { shown: Power[]; hidden: Power[] } {
  const cap = Math.max(minShown, Math.floor(maxShown));
  if (powers.length <= cap) return { shown: powers, hidden: [] };
  return { shown: powers.slice(0, cap - 1), hidden: powers.slice(cap - 1) };
}
