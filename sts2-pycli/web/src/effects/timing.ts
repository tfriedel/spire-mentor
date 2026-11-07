import type { EffectEvent } from "./types";

/**
 * How far apart consecutive hits land.
 *
 * NOT measured from the game — a considered starting value. The damage number
 * arc is ~1.2s (`NDamageNumVfx`, mirrored by `.gs-dmg`), so hits much closer
 * than this stack their numerals into one blob, which is the exact failure
 * `damageNumberJitter` exists to avoid on a single multi-hit.
 */
export const HIT_SPACING_MS = 180;

/** How long an auto-played card (`cardPlay`) stays on screen. Matches the
 *  `gs-autoplay-in-out` keyframes' duration in `game-skin.css`. */
export const AUTO_CARD_SHOW_MS = 1100;

/** How long the board keeps playing after the last event fires. */
export const TOTAL_SETTLE_MS = 1200;

/**
 * How much the clock advances AFTER an event of each kind.
 *
 * The one place pacing is tuned. A zero means "the next event is simultaneous
 * with this one": several blocks gained by one card are a single beat, not a
 * queue.
 */
export const EVENT_ADVANCE_MS: Record<EffectEvent["kind"], number> = {
  attack: 120,
  hurt: HIT_SPACING_MS,
  fullyBlocked: HIT_SPACING_MS,
  death: 0,
  blockGain: 0,
  blockBreak: 0,
  powerAdd: 120,
  powerRemove: 0,
  // PROVISIONAL — a reasoned starting value, not a measured one, and the task
  // that draws the flash should re-tune it against the real thing. 0 because a
  // flash is the highlight that ANNOUNCES what the next row does (Thorns
  // flashes, then the attacker's damage row lands), so it should fire
  // simultaneously with it rather than delaying it, and several powers
  // triggering off one action are one beat rather than a queue. Nothing pins
  // the number: `timing.test.ts` only pins that every kind HAS an entry, so
  // this stays tunable without a test lying about how settled it is.
  powerFlash: 0,
  heal: 120,
  // The auto-played card is shown BEFORE the hits it causes land, so they read
  // as its consequences rather than arriving out of nowhere. Not measured from
  // the game; long enough for the card to register, short enough not to stall
  // an end-of-turn with several Stampede plays.
  cardPlay: 450,
};

/** Offsets in ms at which each event fires, parallel to *events*. */
export function scheduleTimes(events: EffectEvent[]): number[] {
  let t = 0;
  const times: number[] = [];
  for (const e of events) {
    times.push(t);
    t += EVENT_ADVANCE_MS[e.kind];
  }
  return times;
}
