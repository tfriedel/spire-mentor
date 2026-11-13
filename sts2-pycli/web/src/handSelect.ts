/** Phone hand selection: which card is zoomed (`selected`), which is armed for
 *  a target (`pending`), and whether a tap plays something. Pure, so Board can
 *  route every touch tap through it and the rules can be tested without a DOM.
 *
 *  The rules (spec "Interaction rules"):
 *  - The first tap on a card zooms it. A card that needs an enemy target is
 *    also armed, so the next tap on an enemy plays it — but only in live play,
 *    and only when the card can be played at all.
 *  - A second tap on the zoomed card plays it when it needs no target, or when
 *    it needs one and there is exactly one enemy (reading the card before it
 *    resolves is the point, so a lone enemy is never auto-targeted on the
 *    first tap). With several enemies it stays armed.
 *  - Tapping another card switches the zoom (and the arming) to it.
 *  - Replay (`live: false`) and unplayable cards zoom, nothing else. */

export type TapState = { selected: number | null; pending: number | null };

export type TapResult = TapState & { play?: { card: number; target?: number } };

export type TapCard = { index: number; needsTarget: boolean; canPlay: boolean };

export type TapContext = { live: boolean; enemies: number[] };

function select(card: TapCard, ctx: TapContext): TapResult {
  const arms = ctx.live && card.canPlay && card.needsTarget;
  return { selected: card.index, pending: arms ? card.index : null };
}

export function tapCard(state: { selected: number | null; pending?: number | null }, card: TapCard, ctx: TapContext): TapResult {
  if (state.selected !== card.index) return select(card, ctx);
  const again = select(card, ctx);
  if (!ctx.live || !card.canPlay) return again;
  if (!card.needsTarget) return { selected: null, pending: null, play: { card: card.index } };
  if (ctx.enemies.length === 1) {
    return { selected: null, pending: null, play: { card: card.index, target: ctx.enemies[0] } };
  }
  return again;
}

export function tapEnemy(state: TapState, enemyIndex: number): TapResult {
  if (state.pending == null) return { selected: state.selected, pending: null };
  return { selected: null, pending: null, play: { card: state.pending, target: enemyIndex } };
}

export function tapStage(): TapResult {
  return { selected: null, pending: null };
}

/** A selection pinned to the state it was made in (Board holds the combat
 *  state object here). */
export type Held<T> = { index: number; at: T } | null;

export function hold<T>(index: number | null, at: T): Held<T> {
  return index == null ? null : { index, at };
}

/** The held index, but only while the state is still the one it was picked
 *  in: from the first render of a new state on, it is null. */
export function heldIndex<T>(held: Held<T>, at: T): number | null {
  return held != null && held.at === at ? held.index : null;
}
