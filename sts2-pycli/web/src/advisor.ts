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

/** The stored advisor preference. A fresh browser prefers Instinct, as the
 *  mod does since 0.10 (it advises only a run it can read; every other run is
 *  the search's whatever the preference). "search" is kept once chosen. */
export function readAdvisorPref(): Advisor {
  try {
    const v = localStorage.getItem(ADVISOR_KEY);
    return v === "search" ? "search" : v === "e2e" || v == null ? "e2e" : "search";
  } catch {
    return "e2e";
  }
}

/** Whether combat `scores` are the e2e advisor's (policy probabilities,
 *  `prob` on every row) rather than the search's (win chances + visits). */
export function isE2eScores(scores: Score[]): boolean {
  return scores.length > 0 && scores.every((s) => typeof s.prob === "number");
}
