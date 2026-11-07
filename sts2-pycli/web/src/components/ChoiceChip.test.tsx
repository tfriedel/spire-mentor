import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ordinal } from "../coach/coachText";
import ChoiceChip from "./ChoiceChip";
import { checkSummary } from "./choiceCheck";
import { api, type ChoiceCheck, type ChoiceGrade, type DecisionAdvisorCandidate } from "../api";

function grade(over: Partial<ChoiceGrade> = {}): ChoiceGrade {
  return {
    node_id: 7, parent_id: 6, decision: "card_reward", kind: "Card reward",
    verdict: "flag", strength: "strong preference",
    played: { label: "Skip", prob: 0.1, rank: 3 },
    top: { label: "Inflame", prob: 0.72 },
    alternatives: [], n_options: 4, checkable: true,
    ...over,
  };
}

function row(key: string, wins: number, n = 12, hp = 40): DecisionAdvisorCandidate {
  return { key, label: key, win_rate: wins / n, wins, n, avg_final_hp: hp, avg_floors_advanced: 5,
    avg_hp_at_deepest: hp, value: wins / n, delta: 0, is_best: false, inconclusive: 0 };
}

describe("ordinal", () => {
  it("handles 2nd, 3rd, 11th, 22nd", () => {
    expect([1, 2, 3, 4, 11, 12, 13, 22, 101].map(ordinal))
      .toEqual(["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "22nd", "101st"]);
  });
});

describe("ChoiceChip", () => {
  it("flag reads as agreement, never cost", () => {
    const { container } = render(<ChoiceChip grade={grade()} onDismiss={() => {}} />);
    const t = container.textContent ?? "";
    expect(t).toContain("Card reward · you: Skip");
    expect(t).toContain("AI would take: Inflame (strong preference) · Skip was its 3rd choice");
    expect(t).not.toMatch(/%|cost/);
  });

  it("graded against the e2e advisor, it names the e2e net and offers no rollout check", () => {
    const { container } = render(
      <ChoiceChip grade={grade({ policy: "e2e", checkable: false })} onDismiss={() => {}} />,
    );
    const t = container.textContent ?? "";
    expect(t).toContain("Instinct would take: Inflame (strong preference)");
    expect(t).not.toContain("AI would take");
    expect(screen.queryByRole("button", { name: /check/ })).toBeNull();
  });

  it("toss-up names both options", () => {
    const { container } = render(<ChoiceChip grade={grade({ verdict: "tossup", alternatives: [{ label: "Inflame", prob: 0.3 }, { label: "Anger", prob: 0.25 }] })} onDismiss={() => {}} />);
    const t = container.textContent ?? "";
    expect(t).toContain("✓ toss-up — Skip · no strong AI preference");
    expect(t).toContain("Card reward · AI's top picks: Inflame, Anger");
  });

  it("agreement hides the check button", () => {
    render(<ChoiceChip grade={grade({ verdict: "agree", checkable: false, played: { label: "Inflame", prob: 0.72, rank: 1 } })} onDismiss={() => {}} />);
    expect(screen.getByText(/✓ AI agrees/)).toBeInTheDocument();
    expect(screen.queryByText("check it")).toBeNull();
  });

  it("check it streams and shows the rollout summary", async () => {
    const spy = vi.spyOn(api.live, "lastChoiceCheckStream").mockImplementation(async (nodeId, _cfg, onProgress) => {
      expect(nodeId).toBe(7);
      onProgress({ type: "start", total: 24, rollouts: 12, act: 0, candidates: [] });
      return { candidates: [row("you", 1, 12, 31), row("ai", 11, 12, 44)] };
    });
    render(<ChoiceChip grade={grade()} onDismiss={() => {}} />);
    fireEvent.click(screen.getByText("check it"));
    await waitFor(() => expect(screen.getByText(/rollouts to end of act/)).toBeInTheDocument());
    expect(screen.getByText(/you 8% · AI 92% · avg HP 31 vs 44/)).toBeInTheDocument();
    spy.mockRestore();
  });
});

describe("checkSummary", () => {
  it("calls a gap inside two standard errors within noise", () => {
    const c: ChoiceCheck = { candidates: [row("ai", 8), row("you", 5)] };
    expect(checkSummary(c)).toBe("rollouts to end of act (n=12): you 42% · AI 67% · within noise · avg HP 40 vs 40");
  });
  it("names a clear gap without the caveat", () => {
    expect(checkSummary({ candidates: [row("ai", 11), row("you", 1)] })).not.toContain("within noise");
  });
  it("is null without both rows", () => {
    expect(checkSummary({ candidates: [row("you", 1)] })).toBeNull();
  });
});
