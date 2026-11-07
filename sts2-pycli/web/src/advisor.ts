import type { Advisor, Score } from "./api";

/** The e2e student plays (and reads) only the run it was trained on. */
export const E2E_CHARACTER = "Ironclad";
export const E2E_GAME_VERSION = "v0.111.0";

/** Whether the e2e student can advise a run of `character` on `gameVersion`
 *  played on `backend` (it reads Ironclad on v0.111.0, Rust engine only). */
export function e2eAdvisable(
  e2eAvailable: boolean, character: string, gameVersion: string | null | undefined, backend = "rust",
): boolean {
  return e2eAvailable && backend === "rust" && character === E2E_CHARACTER && gameVersion === E2E_GAME_VERSION;
}

/** localStorage key of the hand-played run's advisor preference. */
export const ADVISOR_KEY = "wb.advisor";

/** The stored advisor preference; anything but "e2e" is the search. */
export function readAdvisorPref(): Advisor {
  try {
    return localStorage.getItem(ADVISOR_KEY) === "e2e" ? "e2e" : "search";
  } catch {
    return "search";
  }
}

/** Whether combat `scores` are the e2e advisor's (policy probabilities,
 *  `prob` on every row) rather than the search's (win chances + visits). */
export function isE2eScores(scores: Score[]): boolean {
  return scores.length > 0 && scores.every((s) => typeof s.prob === "number");
}
