import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MentorBar, { type MentorBarProps } from "./MentorBar";
import { mentorStatus } from "./mentorStatus";

function props(over: Partial<MentorBarProps> = {}): MentorBarProps {
  return {
    mode: "hints", setMode: vi.fn(), status: { text: "Search ready", state: "idle" },
    aiPlaying: false, onStop: vi.fn(), ...over,
  };
}

describe("MentorBar", () => {
  it("is one Mode switch, the current segment lit, the others picking their mode", () => {
    const p = props();
    render(<MentorBar {...p} />);
    const group = screen.getByRole("radiogroup", { name: "Mode" });
    expect(group).toHaveTextContent("OffCoachHintsAI plays");
    expect(screen.getByRole("radio", { name: "Hints" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("radio", { name: "Coach" }));
    expect(p.setMode).toHaveBeenCalledWith("coach");
    fireEvent.click(screen.getByRole("radio", { name: "Hints" }));
    expect(p.setMode).toHaveBeenCalledTimes(1);
  });

  it("says why AI plays cannot start, and does not start it", () => {
    const p = props({ aiPlaysDisabled: "start a run first" });
    render(<MentorBar {...p} />);
    const ai = screen.getByRole("radio", { name: "AI plays" });
    expect(ai).toBeDisabled();
    expect(ai).toHaveAttribute("title", "start a run first");
  });

  it("leads with Stop while the AI plays", () => {
    const p = props({ mode: "aiplays", aiPlaying: true });
    const { container } = render(<MentorBar {...p} />);
    fireEvent.click(screen.getByRole("button", { name: /Stop AI/ }));
    expect(p.onStop).toHaveBeenCalled();
    expect(container.firstChild).toHaveClass("wb-mentor--ai");
  });

  it("offers the Brain switch only where it is given, and Peek only in Coach", () => {
    const set = vi.fn();
    const { rerender } = render(<MentorBar {...props()} />);
    expect(screen.queryByRole("radiogroup", { name: "Brain" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Peek/ })).toBeNull();
    rerender(<MentorBar {...props({ mode: "coach", onPeek: () => {}, brain: { value: "search", set } })} />);
    fireEvent.click(screen.getByRole("radio", { name: "Instinct" }));
    expect(set).toHaveBeenCalledWith("instinct");
    expect(screen.getByRole("button", { name: /Peek/ })).toBeInTheDocument();
  });
});

describe("mentorStatus", () => {
  it("says what the AI is doing in words", () => {
    const base = { mode: "hints" as const, aiPlaying: false, thinking: false, brain: "search" as const, autobattle: false };
    expect(mentorStatus(base)).toEqual({ text: "Search ready", state: "idle" });
    expect(mentorStatus({ ...base, thinking: true, brain: "instinct" }).text).toBe("Instinct thinking…");
    expect(mentorStatus({ ...base, mode: "off", thinking: true })).toEqual({ text: "AI off", state: "off" });
    expect(mentorStatus({ ...base, mode: "aiplays", aiPlaying: true, aiStep: 12 })).toEqual({
      text: "AI playing · step 12", state: "ai" });
    expect(mentorStatus({ ...base, mode: "coach" }).text).toMatch(/your call/);
  });
});
