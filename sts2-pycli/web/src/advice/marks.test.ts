import { describe, expect, it } from "vitest";
import type { Score } from "../api";
import {
  actionKey, buildMarks, cardKey, END_TURN_KEY, gradeMove, humanize, policyOptions, potionKey,
  rowVerdict, scoreKey, scoreOptions, verdict, type MarkOption,
} from "./marks";

function score(p: Partial<Score> & { stable_idx: number }): Score {
  return {
    action: "x", win_prob: 0.5, q: 0, visits: 0, is_best: false, q_saturated: false,
    low_confidence: false, target_index: null, ...p,
  };
}

function opt(p: Partial<MarkOption> & { key: string | null }): MarkOption {
  return { label: p.key ?? "", isBest: false, inBand: false, share: null, q: null, pickRank: null,
    targetIndex: null, ...p };
}

describe("thing keys", () => {
  it("maps the recommender's sentinels", () => {
    expect(scoreKey({ stable_idx: 3 })).toBe(cardKey(3));
    expect(scoreKey({ stable_idx: -1 })).toBe(END_TURN_KEY);
    expect(scoreKey({ stable_idx: -102 })).toBe(potionKey(2));
    expect(scoreKey({ stable_idx: -300 })).toBeNull();
  });

  it("keys a screen action by verb and args, order- and type-blind", () => {
    expect(actionKey("buy_card", { card_index: 2 })).toBe(actionKey("buy_card", { card_index: "2" }));
    expect(actionKey("select_cards", { indices: [1, 4] })).toBe(actionKey("select_cards", { indices: "1,4" }));
    expect(actionKey("select_map_node", { col: 1, row: 2 })).toBe(actionKey("select_map_node", { row: 2, col: 1 }));
    expect(actionKey("leave_room", {})).toBe("leave_room");
  });
});

describe("buildMarks", () => {
  it("gives a targeted card one mark, its best row's, with the target", () => {
    const marks = buildMarks(scoreOptions([
      score({ stable_idx: 0, is_best: true, q: 0.5, visits: 60, target_index: 1 }),
      score({ stable_idx: 0, q: 0.4, visits: 30, target_index: 0 }),
      score({ stable_idx: -1, q: 0.1, visits: 10 }),
    ]));
    const card = marks.filter((m) => m.key === cardKey(0));
    expect(card).toHaveLength(1);
    expect(card[0]).toMatchObject({ tier: "best", order: 1, targetIndex: 1 });
  });

  it("drops moves under 0.5% and keeps at most two alternatives", () => {
    const marks = buildMarks([
      opt({ key: "a", isBest: true, share: 0.4 }),
      opt({ key: "b", share: 0.2 }),
      opt({ key: "c", share: 0.18 }),
      opt({ key: "d", share: 0.16 }),
      opt({ key: "e", share: 0.001 }),
    ]);
    expect(marks.map((m) => m.key)).toEqual(["a", "b", "c"]);
    expect(marks.slice(1).every((m) => m.tier === "alt" && m.order == null)).toBe(true);
  });

  it("rings a move as good as the pick (in the band, or within the Q delta)", () => {
    const marks = buildMarks([
      opt({ key: "a", isBest: true, q: 0.5, share: 0.3 }),
      opt({ key: "b", q: 0.498, share: 0.05 }),
      opt({ key: "c", inBand: true, q: 0.4, share: 0.05 }),
      opt({ key: "d", q: 0.3, share: 0.05 }),
    ]);
    expect(Object.fromEntries(marks.map((m) => [m.key, m.tier]))).toEqual({ a: "best", b: "good", c: "good" });
  });

  it("numbers a plan in order, including steps the options did not list", () => {
    const marks = buildMarks(
      [opt({ key: "a", isBest: true, share: 0.5 }), opt({ key: "b", share: 0.01 })],
      [{ key: "a", targetIndex: null, label: "A" }, { key: "b", targetIndex: 2, label: "B" },
        { key: "z", targetIndex: null, label: "Z" }],
    );
    expect(marks.map((m) => [m.key, m.order, m.tier])).toEqual([["a", 1, "best"], ["b", 2, "good"], ["z", 3, "good"]]);
    expect(marks[1].targetIndex).toBe(2);
  });

  it("marks the run policy's options by the action the screen sends", () => {
    const marks = buildMarks(policyOptions([
      { action: "select_card_reward", args: { card_index: 1 }, label: "Inflame", prob: 0.7, is_best: true },
      { action: "skip_card_reward", args: {}, label: "Skip", prob: 0.3, is_best: false },
    ]));
    expect(marks.map((m) => m.key)).toEqual([actionKey("select_card_reward", { card_index: 1 }), "skip_card_reward"]);
  });
});

describe("verdict words", () => {
  it("grades the search by loss against the pick", () => {
    const v = (q: number) => verdict({ isBest: false, inBand: false, q, share: 0.1 }, 0.5);
    expect([v(0.497), v(0.49), v(0.45), v(0.3)]).toEqual(["Also good", "Okay", "Weak", "Mistake"]);
  });

  it("grades a policy by how sure the net is", () => {
    const v = (share: number) => verdict({ isBest: false, inBand: false, q: null, share }, null);
    expect([v(0.3), v(0.1), v(0.02)]).toEqual(["Also good", "Okay", "Unlikely"]);
  });

  it("reads a later plan step as Then", () => {
    expect(rowVerdict({ isBest: false, inBand: false, q: null, share: 0.02 }, null, 2)).toBe("Then");
    expect(rowVerdict({ isBest: true, inBand: false, q: null, share: 0.9 }, null, 1)).toBe("Best");
  });
});

describe("gradeMove", () => {
  const rows = scoreOptions([
    score({ stable_idx: 0, is_best: true, q: 0.5, visits: 50, target_index: 0 }),
    score({ stable_idx: 0, q: 0.3, visits: 10, target_index: 1 }),
    score({ stable_idx: 1, q: 0.49, visits: 40 }),
  ]);
  it("grades the exact move, target included", () => {
    expect(gradeMove(rows, cardKey(0), 0)).toBe("best");
    expect(gradeMove(rows, cardKey(0), 1)).toBe("mistake");
    expect(gradeMove(rows, cardKey(1), null)).toBe("okay");
  });
  it("says nothing about a move the advice did not cover", () => {
    expect(gradeMove(rows, cardKey(5), null)).toBeNull();
  });
});

describe("humanize", () => {
  it("turns raw ids into words and leaves the rest alone", () => {
    expect(humanize("Play SHRUG_IT_OFF → AI")).toBe("Play Shrug It Off → AI");
    expect(humanize("CARD.PERFECTED_STRIKE")).toBe("Perfected Strike");
    expect(humanize("GO_FOR_THE_EYES")).toBe("Go for the Eyes");
  });
});
