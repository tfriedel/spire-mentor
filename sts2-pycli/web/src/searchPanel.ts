// What the "Advice" sidebar (formerly "Machine reasoning") shows, as a pure function of the run
// state. Kept out of App.tsx so the precedence rules below are unit-testable
// without standing up the whole network-driven app.
import type { Enemy, LastSearch, Score } from "./api";
import { isE2eScores } from "./advisor";

// `LastSearch` is a wire shape, so it is declared in api.ts alongside the rest
// of them; re-exported here because this module is where it gets interpreted.
export type { LastSearch };

export type PanelView =
  | {
      kind: "scores";
      scores: Score[];
      pv: string[] | null;
      title: string | null;
      /** Board the scores belong to, or null to use the live one. */
      enemies: Enemy[] | null;
    }
  | { kind: "message"; text: string };

/**
 * Pick the sidebar's content.
 *
 * Precedence, and why:
 *
 * 1. **Live scores win whenever they exist.** They belong to the position on
 *    screen; a recorded search does not. So a hint you pressed H for is never
 *    shadowed by the autoplay leftovers underneath it.
 * 2. **Otherwise show the last search, running or paused.** During autoplay
 *    this is the whole point — the previous behaviour replaced the panel with
 *    a flat "autoplaying…" so the bars you wanted were never drawn even once
 *    the data existed. Keeping it after a Stop is what makes "we're losing,
 *    let me look at why" possible: the panel holds the last decision instead
 *    of blanking the moment the loop exits.
 * 3. **Otherwise a status line**, which differs by whether anything is
 *    actually being computed for you.
 */
export function reasoningView(opts: {
  autoRunning: boolean;
  scores: Score[];
  pv: string[] | null;
  lastSearch: LastSearch | null;
  hintBusy: boolean;
  autoHints: boolean;
}): PanelView {
  const { autoRunning, scores, pv, lastSearch, hintBusy, autoHints } = opts;
  // The e2e advisor's rows are a policy's probabilities, not a search: the
  // caption says whose they are.
  if (scores.length > 0) {
    return { kind: "scores", scores, pv, title: isE2eScores(scores) ? "Instinct" : null, enemies: null };
  }
  if (lastSearch && lastSearch.scores.length > 0) {
    return {
      kind: "scores",
      scores: lastSearch.scores,
      // The agent's step record carries no principal variation — only the
      // root distribution — so there is nothing honest to put here.
      pv: null,
      title: `${isE2eScores(lastSearch.scores) ? "last Instinct pick" : "last search"} → ${lastSearch.action}`,
      enemies: lastSearch.enemies,
    };
  }
  if (autoRunning) return { kind: "message", text: "searching…" };
  return {
    kind: "message",
    text: hintBusy || autoHints ? "thinking…" : "press H for a hint",
  };
}

/**
 * Whether the reasoning sidebar is shown at all.
 *
 * Advice is opt-in: with the global hints switch off, the workbench must not
 * volunteer its read of a position you are playing yourself. Two carve-outs:
 *
 * - `hintShown` — you pressed H. Asking is not being told.
 * - **whole-run automation** — the AI is playing, so there is no move of yours
 *   for it to spoil. Watching it search IS the feature. `autoPaused &&
 *   lastSearch` extends that across a Stop, which is the whole point of
 *   stopping: to read why the run is going the way it is. It lapses the moment
 *   you take a move yourself, because `liveAction` clears `lastSearch` — so
 *   taking the run back also takes the advice away, with no extra bookkeeping.
 */
export function panelVisible(opts: {
  hintsOn: boolean;
  hintShown: boolean;
  autoRunning: boolean;
  autoPaused: boolean;
  lastSearch: LastSearch | null;
}): boolean {
  const { hintsOn, hintShown, autoRunning, autoPaused, lastSearch } = opts;
  return hintsOn || hintShown || autoRunning || (autoPaused && lastSearch != null);
}
