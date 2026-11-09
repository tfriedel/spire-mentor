// The marks of the decision on screen, handed to whatever draws the thing they
// mark (a hand card, the End Turn plate, a reward card, a shop slot, …), and
// the "engine numbers" setting that decides whether the engine's own units
// (win %, Q, W/L/T, V(s)) are drawn beside the words.
//
// App builds the value (marks.ts decides what is marked) and supplies it with
// AdviceMarksProvider (AdviceContext.tsx); a component reads it here and looks
// its own thing up by the key it would dispatch. Outside a provider nothing is
// marked, so a unit test or a stray subtree shows no advice rather than
// leaking it.

import { createContext, useContext, useEffect, useState } from "react";
import { actionKey, type Mark } from "./marks";

export interface AdviceMarksValue {
  marks: Map<string, Mark>;
  /** Draw the engine's numbers (win %, Q, shares) next to the marks and rows. */
  showNumbers: boolean;
}

export const AdviceMarksCtx = createContext<AdviceMarksValue>({ marks: new Map(), showNumbers: false });

export function useAdviceMarks(): AdviceMarksValue {
  return useContext(AdviceMarksCtx);
}

/** The mark on a screen action, looked up by the verb + args the screen would
 *  send (marks.ts `actionKey`). */
export function markFor(marks: Map<string, Mark>, action: string, args: Record<string, unknown> = {}): Mark | null {
  return marks.get(actionKey(action, args)) ?? null;
}

/** The class list of a marked thing: ` wb-mark wb-mark--best` etc., or "". */
export function markClass(m: Mark | null): string {
  return m ? ` wb-mark wb-mark--${m.tier}` : "";
}

const NUMBERS_KEY = "wb.engineNumbers";

function readNumbers(): boolean {
  try {
    return localStorage.getItem(NUMBERS_KEY) === "1";
  } catch {
    return false;
  }
}

/** The "engine numbers" setting (default off, like the mod's "Show engine
 *  numbers"): persisted per browser. */
export function useEngineNumbersPref(): [boolean, (v: boolean) => void] {
  const [on, setOn] = useState(readNumbers);
  useEffect(() => {
    try {
      localStorage.setItem(NUMBERS_KEY, on ? "1" : "0");
    } catch {
      /* storage unavailable: the setting still holds for this tab */
    }
  }, [on]);
  return [on, setOn];
}
