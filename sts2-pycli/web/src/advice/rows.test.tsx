import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card, Score } from "../api";
import AdviceList from "./AdviceList";
import { buildMarks, markIndex, scoreOptions } from "./marks";
import { policyRows, scoreRows } from "./rows";

function score(p: Partial<Score> & { stable_idx: number }): Score {
  return {
    action: "x", win_prob: 0.5, q: 0, visits: 0, is_best: false, q_saturated: false,
    low_confidence: false, target_index: null, ...p,
  };
}

const hand = [
  { index: 0, name: "Strike", type: "Attack" },
  { index: 1, name: "Defend", type: "Skill" },
] as Card[];

describe("scoreRows", () => {
  const scores = [
    score({ stable_idx: -1, action: "end turn", q: 0.1, visits: 10 }),
    score({ stable_idx: 0, action: "Strike", q: 0.5, visits: 60, is_best: true }),
    score({ stable_idx: 1, action: "Defend", q: 0.47, visits: 30 }),
    score({ stable_idx: 1, action: "Defend", q: 0.2, visits: 0 }),
  ];
  const marks = markIndex(buildMarks(scoreOptions(scores)));
  const rows = scoreRows(scores, marks, { hand });

  it("puts the pick first with its step, then by share, hidden rows last", () => {
    expect(rows.map((r) => r.label)).toEqual(["Strike", "Defend", "end turn", "Defend"]);
    expect(rows[0]).toMatchObject({ step: 1, verdict: "Best", kind: "attack", best: true });
    expect(rows[1]).toMatchObject({ kind: "skill", verdict: "Weak" });
    expect(rows[2]).toMatchObject({ kind: "end_turn", verdict: "Mistake" });
    expect(rows[3].hidden).toBe(true);
  });

  it("carries the engine's numbers for the search only", () => {
    expect(rows[0].q).toBe(0.5);
    expect(rows[0].wlt).toBe("0/0/0");
    const e2e = scoreRows([score({ stable_idx: 0, prob: 0.9, visits: 900000, is_best: true })], new Map(), { hand });
    expect(e2e[0]).toMatchObject({ q: null, wlt: null, share: 0.9 });
  });
});

describe("policyRows", () => {
  it("ranks the run policy's options in words, skip in grey", () => {
    const rows = policyRows([
      { action: "skip_card_reward", args: {}, label: "Skip", prob: 0.1, is_best: false },
      { action: "select_card_reward", args: { card_index: 0 }, label: "CARD.INFLAME", prob: 0.9, is_best: true },
    ], new Map());
    expect(rows.map((r) => [r.label, r.verdict, r.kind])).toEqual([
      ["Inflame", "Best", "pick"], ["Skip", "Okay", "skip"]]);
  });
});

describe("AdviceList", () => {
  const many = Array.from({ length: 11 }, (_, i) => ({
    id: String(i), label: `Move ${i}`, verdict: i === 0 ? "Best" as const : "Okay" as const, step: i === 0 ? 1 : null,
    kind: "pick" as const, best: i === 0, hidden: i === 10, share: 0.1, q: 0.1 * i, wlt: "1/0/0",
  }));

  it("shows eight rows and folds the rest behind '… N more'", () => {
    render(<AdviceList rows={many} showNumbers={false} />);
    expect(screen.getAllByText(/^Move /)).toHaveLength(8);
    fireEvent.click(screen.getByRole("button", { name: /3 more/ }));
    expect(screen.getAllByText(/^Move /)).toHaveLength(11);
  });

  it("lays out the engine columns only with engine numbers on", () => {
    const { rerender } = render(<AdviceList rows={many.slice(0, 2)} showNumbers={false} />);
    expect(screen.queryByText("1/0/0")).toBeNull();
    rerender(<AdviceList rows={many.slice(0, 2)} showNumbers />);
    expect(screen.getAllByText("1/0/0")).toHaveLength(2);
  });
});
