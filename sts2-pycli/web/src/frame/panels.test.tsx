import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AdvicePanel } from "./panels";
import { INITIAL_STATE, type Advice } from "./store";

const base: Advice = {
  note: null, p_win: null, best_q: null, low_confidence: false, best: null, pv: [],
  pick_reason: null, pick_gap_hp: null, hand_back: null, brain: "instinct", source_tag: "Instinct", options: [],
};

describe("AdvicePanel: Instinct's estimate (GH #1222)", () => {
  it("heads the panel with the run and fight chances, the HP range and what they assume", () => {
    const advice = { ...base, estimate: { p_run: 0.71, p_fight: 0.55, hp_loss: { q10: 2, q50: 9.4, q90: 20 } } };
    render(<AdvicePanel state={{ ...INITIAL_STATE, advice }} />);
    const head = screen.getByTestId("frame-estimate");
    expect(head).toHaveTextContent("Run 71% · Fight 55%");
    expect(head).toHaveTextContent("This fight: −9 HP (−2 HP to −20 HP)");
    expect(head).toHaveTextContent("if Instinct plays every move from here");
    expect(head.querySelector(".fr-est-bad")).toHaveTextContent("55%");
  });

  it("shows only the run off a fight, and nothing from the search", () => {
    const { rerender } = render(
      <AdvicePanel state={{ ...INITIAL_STATE, advice: { ...base, estimate: { p_run: 0.25, p_fight: null, hp_loss: null } } }} />,
    );
    expect(screen.getByTestId("frame-estimate")).toHaveTextContent(/^Run 25%if Instinct/);
    rerender(<AdvicePanel state={{ ...INITIAL_STATE, advice: { ...base, brain: "search", p_win: 0.6 } }} />);
    expect(screen.queryByTestId("frame-estimate")).toBeNull();
    expect(screen.getByText(/Win chance 60%/)).toBeInTheDocument();
  });
});
