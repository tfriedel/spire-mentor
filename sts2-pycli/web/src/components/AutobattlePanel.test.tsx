import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import AutobattlePanel, { AutobattleDiffCard, AutobattleVerdictCard } from "./AutobattlePanel";
import type { AutobattleDiff, AutobattleVerdict } from "../api";

function diff(over: Partial<AutobattleDiff> = {}): AutobattleDiff {
  return {
    from_floor: 8,
    to_floor: 12,
    autoplayed: 3,
    hp: 52,
    max_hp: 75,
    hp_delta: -21,
    gold: 145,
    gold_delta: 30,
    deck_size: 14,
    deck: { added: [], removed: [], upgraded: [] },
    relics_gained: [],
    relics_lost: [],
    potions_gained: [],
    potions_lost: [],
    lines: [],
    ...over,
  };
}

describe("AutobattleDiffCard", () => {
  it("shows what moved since the last stop", () => {
    render(
      <AutobattleDiffCard
        diff={diff({
          deck: { added: ["Inflame"], removed: ["Strike"], upgraded: ["Defend"] },
          relics_gained: ["Bag of Prep"],
        })}
      />,
    );
    expect(screen.getByText("Inflame")).toBeTruthy();
    expect(screen.getByText("Strike")).toBeTruthy();
    expect(screen.getByText("Defend")).toBeTruthy();
    expect(screen.getByText("Bag of Prep")).toBeTruthy();
    expect(screen.getByText(/HP 52\/75 \(−21\)/)).toBeTruthy();
    expect(screen.getByText(/Gold 145 \(\+30\)/)).toBeTruthy();
    expect(screen.getByText(/3 decisions auto-played/)).toBeTruthy();
  });

  it("labels the opening card as the run start and shows no deltas", () => {
    render(<AutobattleDiffCard diff={diff({ from_floor: null, hp_delta: null, gold_delta: null })} />);
    expect(screen.getByText(/Run start/)).toBeTruthy();
    expect(screen.queryByText(/\(−/)).toBeNull();
  });

  it("says so explicitly when nothing changed", () => {
    render(<AutobattleDiffCard diff={diff({ hp_delta: 0, gold_delta: 0 })} />);
    expect(screen.getByText("Nothing changed.")).toBeTruthy();
  });

  it("uses the singular for a single auto-played decision", () => {
    render(<AutobattleDiffCard diff={diff({ autoplayed: 1 })} />);
    expect(screen.getByText(/1 decision auto-played/)).toBeTruthy();
  });
});

describe("AutobattleVerdictCard", () => {
  const agreed: AutobattleVerdict = {
    available: true,
    agreed: true,
    champion_label: "Take Inflame",
    your_label: "Take Inflame",
    brain: "ppo",
  };

  it("confirms a match without restating the pick", () => {
    render(<AutobattleVerdictCard verdict={agreed} />);
    expect(screen.getByText(/Matched the champion/)).toBeTruthy();
    expect(screen.queryByText(/Champion:/)).toBeNull();
  });

  it("shows both picks on a disagreement", () => {
    render(
      <AutobattleVerdictCard
        verdict={{ ...agreed, agreed: false, your_label: "Skip", champion_label: "Take Inflame" }}
      />,
    );
    expect(screen.getByText(/Differed from the champion/)).toBeTruthy();
    expect(screen.getByText("Take Inflame")).toBeTruthy();
    expect(screen.getByText("Skip")).toBeTruthy();
  });

  it("never implies a win probability", () => {
    const { container } = render(
      <AutobattleVerdictCard verdict={{ ...agreed, agreed: false, band: "differed" }} />,
    );
    expect(container.textContent).toMatch(/not a win-rate judgement/);
  });

  it("bands a close call as a near-tie rather than a miss", () => {
    render(
      <AutobattleVerdictCard
        verdict={{
          ...agreed,
          agreed: false,
          band: "near_tie",
          margin: 0.02,
          champion_prob: 0.4,
          your_prob: 0.38,
          your_label: "Take Anger",
        }}
      />,
    );
    expect(screen.getByText(/Near-tie/)).toBeTruthy();
    expect(screen.queryByText(/Differed/)).toBeNull();
    expect(screen.getByText(/38%/)).toBeTruthy();
    expect(screen.getByText(/40%/)).toBeTruthy();
  });

  it("labels the near-tie percentages as policy preference, not win rate", () => {
    const { container } = render(
      <AutobattleVerdictCard
        verdict={{ ...agreed, agreed: false, band: "near_tie", champion_prob: 0.4, your_prob: 0.38 }}
      />,
    );
    expect(container.textContent).toMatch(/how much the policy preferred each option, not win rates/);
  });

  it("never claims near-tied options are equally good", () => {
    // A flat policy also happens when the net is uncertain or undertrained;
    // the probabilities alone cannot tell those apart.
    const { container } = render(
      <AutobattleVerdictCard verdict={{ ...agreed, agreed: false, band: "near_tie", margin: 0.01 }} />,
    );
    expect(container.textContent).toMatch(/not that the picks are equally good/);
    // No affirmative equivalence claim anywhere in the card.
    expect(container.textContent).not.toMatch(/just as good|either is fine|equally strong/i);
  });

  it("falls back to two bands when the payload carries no band", () => {
    render(<AutobattleVerdictCard verdict={{ ...agreed, agreed: false, band: undefined }} />);
    expect(screen.getByText(/Differed from the champion/)).toBeTruthy();
  });

  it("omits a probability that the brain did not supply", () => {
    const { container } = render(
      <AutobattleVerdictCard
        verdict={{ ...agreed, agreed: false, band: "differed", champion_prob: null, your_prob: null }}
      />,
    );
    expect(container.textContent).not.toMatch(/%/);
  });

  it("explains itself when the decision could not be graded", () => {
    render(<AutobattleVerdictCard verdict={{ available: false, reason: "no graded decision pending" }} />);
    expect(screen.getByText("no graded decision pending")).toBeTruthy();
  });
});

describe("AutobattlePanel", () => {
  const props = {
    diff: null,
    verdict: null,
    busy: false,
    over: false,
    won: false,
    watch: false,
    onWatchChange: () => {},
    onAdvance: () => {},
    onStop: () => {},
  };

  it("offers to start before the first stop", () => {
    render(<AutobattlePanel {...props} />);
    expect(screen.getByRole("button", { name: "Start autobattle" })).toBeTruthy();
  });

  it("advances on click", () => {
    const onAdvance = vi.fn();
    render(<AutobattlePanel {...props} diff={diff()} onAdvance={onAdvance} />);
    fireEvent.click(screen.getByRole("button", { name: "Skip to next decision" }));
    expect(onAdvance).toHaveBeenCalledOnce();
  });

  it("locks the button while the agent is playing", () => {
    render(<AutobattlePanel {...props} busy diff={diff()} />);
    expect(screen.getByRole("button", { name: "Playing…" }).hasAttribute("disabled")).toBe(true);
  });

  it("reports the outcome and stops offering advances once the run ends", () => {
    render(<AutobattlePanel {...props} over won diff={diff()} />);
    expect(screen.getByText("Run won.")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Start autobattle|decision/ }).hasAttribute("disabled"))
      .toBe(true);
  });

  it("plays the fights out rather than skipping them when watching", () => {
    render(<AutobattlePanel {...props} watch diff={diff()} />);
    expect(screen.getByRole("button", { name: "Play to next decision" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Skip to next decision" })).toBeNull();
  });

  it("toggles watching the fights", () => {
    const onWatchChange = vi.fn();
    render(<AutobattlePanel {...props} onWatchChange={onWatchChange} />);
    fireEvent.click(screen.getByLabelText(/watch fights/));
    expect(onWatchChange).toHaveBeenCalledWith(true);
  });

  it("offers a stop only while a watched fight is actually running", () => {
    const onStop = vi.fn();
    const { rerender } = render(<AutobattlePanel {...props} watch diff={diff()} onStop={onStop} />);
    expect(screen.queryByRole("button", { name: /stop/ })).toBeNull();
    rerender(<AutobattlePanel {...props} watch busy diff={diff()} onStop={onStop} />);
    fireEvent.click(screen.getByRole("button", { name: /stop/ }));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("says it is fighting, not merely playing, while a watched leg runs", () => {
    render(<AutobattlePanel {...props} watch busy diff={diff()} />);
    expect(screen.getByRole("button", { name: "Fighting…" }).hasAttribute("disabled")).toBe(true);
  });

  it("shows the verdict above the next stop's diff", () => {
    const { container } = render(
      <AutobattlePanel
        {...props}
        diff={diff()}
        verdict={{ available: true, agreed: true, brain: "ppo" }}
      />,
    );
    const text = container.textContent ?? "";
    expect(text.indexOf("Matched the champion")).toBeLessThan(text.indexOf("Since your last stop"));
  });
});

describe("the brain behind the verdict", () => {
  it("names the e2e student in the verdict head", () => {
    render(
      <AutobattleVerdictCard
        verdict={{ available: true, agreed: true, band: "matched", brain: "e2e", champion_label: "Strike", your_label: "Strike" }}
      />,
    );
    expect(screen.getByText(/Instinct \(one net, no search\)/)).toBeInTheDocument();
  });
});
