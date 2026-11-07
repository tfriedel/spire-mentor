import { afterEach, describe, expect, it } from "vitest";
import { ADVISOR_KEY, e2eAdvisable, isE2eScores, readAdvisorPref } from "./advisor";
import { bestAction } from "./hint";
import { reasoningView } from "./searchPanel";
import type { Score } from "./api";

/** The e2e advisor's hint rows as `/api/live/hint` answers them. */
function e2eRow(over: Partial<Score>): Score {
  const p = over.prob ?? 0.1;
  return {
    stable_idx: 0, action: "Strike", win_prob: p, prob: p, q: 2 * p - 1, visits: Math.round(p * 1e6),
    is_best: false, q_saturated: false, low_confidence: false, target_index: null, ...over,
  };
}

afterEach(() => localStorage.clear());

describe("the e2e advisor", () => {
  it("is offered only on a Rust Ironclad v0.111.0 run of a build that has the student", () => {
    expect(e2eAdvisable(true, "Ironclad", "v0.111.0")).toBe(true);
    expect(e2eAdvisable(false, "Ironclad", "v0.111.0")).toBe(false);
    expect(e2eAdvisable(true, "Silent", "v0.111.0")).toBe(false);
    expect(e2eAdvisable(true, "Ironclad", "v0.107.1")).toBe(false);
    expect(e2eAdvisable(true, "Ironclad", null)).toBe(false);
    expect(e2eAdvisable(true, "Ironclad", "v0.111.0", "csharp")).toBe(false);
  });

  it("the preference defaults to the search and survives a reload", () => {
    expect(readAdvisorPref()).toBe("search");
    localStorage.setItem(ADVISOR_KEY, "e2e");
    expect(readAdvisorPref()).toBe("e2e");
    localStorage.setItem(ADVISOR_KEY, "garbage");
    expect(readAdvisorPref()).toBe("search");
  });

  it("tells its rows from the search's by `prob`", () => {
    expect(isE2eScores([])).toBe(false);
    expect(isE2eScores([e2eRow({}), e2eRow({ stable_idx: -1 })])).toBe(true);
    const search: Score = { ...e2eRow({}), prob: undefined };
    expect(isE2eScores([search])).toBe(false);
  });

  it("play best / autoplay read its pick off the hand index the engine maps the class to", () => {
    const rows = [
      e2eRow({ stable_idx: 3, target_index: 1, prob: 0.7, is_best: true }),
      e2eRow({ stable_idx: -1, prob: 0.2, action: "end turn" }),
      e2eRow({ stable_idx: -102, prob: 0.1, action: "drink Fire Potion" }),
    ];
    expect(bestAction(rows)).toEqual({ kind: "play", cardIndex: 3, targetIndex: 1 });
    expect(bestAction([{ ...rows[1], is_best: true }])).toEqual({ kind: "end" });
    expect(bestAction([{ ...rows[2], is_best: true }])).toEqual({ kind: "potion", slot: 2, targetIndex: undefined });
  });

  it("the reasoning panel captions its rows as the e2e net's", () => {
    const rows = [e2eRow({ is_best: true, prob: 0.9 })];
    const v = reasoningView({ autoRunning: false, scores: rows, pv: null, lastSearch: null, hintBusy: false, autoHints: true });
    expect(v.kind === "scores" && v.title).toBe("Instinct");
    const last = reasoningView({
      autoRunning: true, scores: [], pv: null, hintBusy: false, autoHints: true,
      lastSearch: { scores: rows, action: "Strike", enemies: [] },
    });
    expect(last.kind === "scores" && last.title).toBe("last Instinct pick → Strike");
  });
});
