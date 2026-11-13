import type { PowerChrome } from "../portraitSplit";
import { hpBarWidth } from "./combatantMetrics";

/** Phone portrait (the split): the power chrome a multi-rig boss's part
 *  takes, the same as an ordinary enemy's — the fight's chrome mode, and the
 *  part's share of the boss's solver slot (`Board`: the slot split evenly
 *  among the targetable parts). Absent off the split. */
export interface PartChrome {
  chrome: PowerChrome;
  /** `rows`: cells across (`splitPowerCells` of `slotReal`). */
  cells: number;
  /** The part's slot less its power column, design px: what its body, HP
   *  bar and name may span. The HP bar is capped to it. */
  bodySlot: number;
  /** The part's slot, design px: what its power rows may span. */
  slot: number;
  /** The part's slot, real px, for `data-split-slot`. */
  slotReal: number;
}

/** The HP bar width a part draws: its own (`own`, the game's where known,
 *  else the max-HP curve), capped to its split slot. */
export function partHpWidth(own: number | undefined, maxHp: number, part?: PartChrome): number | undefined {
  if (!part) return own;
  return Math.min(own ?? hpBarWidth(maxHp), part.bodySlot);
}
