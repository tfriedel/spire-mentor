import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Reasoning from "./Reasoning";
import type { Score } from "../api";

function row(over: Partial<Score>): Score {
  return {
    stable_idx: 0, action: "Strike", win_prob: 0.5, q: 0, visits: 100, is_best: false,
    q_saturated: false, low_confidence: false, target_index: null, ...over,
  };
}

describe("Reasoning under the e2e advisor", () => {
  it("lists the student's classes with their probabilities, starred pick, no Q or W/L/T", () => {
    const scores = [
      row({ stable_idx: 2, action: "Bash", prob: 0.62, win_prob: 0.62, visits: 620000, is_best: true, target_index: 0 }),
      row({ stable_idx: -1, action: "end turn", prob: 0.38, win_prob: 0.38, visits: 380000 }),
    ];
    render(<Reasoning scores={scores} pv={null} />);
    expect(screen.getByTestId("e2e-rows")).toBeInTheDocument();
    expect(screen.getByText(/^★ Bash/)).toBeInTheDocument();
    expect(screen.getByText("62.0%")).toBeInTheDocument();
    expect(screen.getByText("38.0%")).toBeInTheDocument();
    expect(screen.getByText(/policy probabilities, not win chances/)).toBeInTheDocument();
    // No search columns: Q, visit share, rollout W/L/T.
    expect(screen.queryByText(/^\+0\.\d\d$/)).toBeNull();
    expect(screen.queryByText("0/0/0")).toBeNull();
  });

  it("search rows still render as the search's", () => {
    render(<Reasoning scores={[row({ is_best: true, q: 0.4, wins: 3, losses: 1, ties: 0 })]} pv={null} />);
    expect(screen.queryByTestId("e2e-rows")).toBeNull();
    expect(screen.getByText("3/1/0")).toBeInTheDocument();
  });
});
