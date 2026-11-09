import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Board from "../Board";
import GameHud from "../GameHud";
import RelicTray from "../RelicTray";
import type { Card, Step } from "../../api";

function card(name: string): Card {
  return { index: 0, id: name, name, cost: 1, type: "Attack", description: "", image: null };
}

function pileStep(): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: {
      decision: "combat_play", round: 1, energy: 3, max_energy: 3,
      draw_pile_count: 2, discard_pile_count: 1, exhaust_pile_count: 0,
      draw_pile: [card("Bash"), card("Strike")],
      discard_pile: [card("Defend")],
      exhaust_pile: [],
      player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 },
      enemies: [{ index: 0, name: "Cultist", hp: 50, max_hp: 50, block: 0, image: null }],
      hand: [],
    },
  };
}

describe("pile inspection", () => {
  it("opens the draw pile viewer (with order-hidden note) on click", () => {
    render(<Board step={pileStep()} />);
    fireEvent.click(screen.getByRole("button", { name: /^draw 2$/i }));
    const dialog = screen.getByRole("dialog", { name: /draw pile/i });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText(/order hidden/i)).toBeInTheDocument();
    // Both cards listed.
    expect(screen.getByText("Bash")).toBeInTheDocument();
    expect(screen.getByText("Strike")).toBeInTheDocument();
  });

  it("opens the discard pile viewer without an order note", () => {
    render(<Board step={pileStep()} />);
    fireEvent.click(screen.getByRole("button", { name: /^discard 1$/i }));
    expect(screen.getByRole("dialog", { name: /discard pile/i })).toBeInTheDocument();
    expect(screen.queryByText(/order hidden/i)).toBeNull();
    expect(screen.getByText("Defend")).toBeInTheDocument();
  });

  it("closes the viewer on the close button", () => {
    render(<Board step={pileStep()} />);
    fireEvent.click(screen.getByRole("button", { name: /^draw 2$/i }));
    fireEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("hides the exhaust pile button while nothing is exhausted (game HUD has none)", () => {
    render(<Board step={pileStep()} />);
    expect(screen.queryByRole("button", { name: /^exhaust/i })).toBeNull();
  });

  it("shows the exhaust pile button once something is exhausted, and opens its viewer", () => {
    const step = pileStep();
    step.state.exhaust_pile_count = 1;
    step.state.exhaust_pile = [card("Ashen Strike")];
    render(<Board step={step} />);
    fireEvent.click(screen.getByRole("button", { name: /^exhaust 1$/i }));
    expect(screen.getByRole("dialog", { name: /exhaust pile/i })).toBeInTheDocument();
    expect(screen.getByText("Ashen Strike")).toBeInTheDocument();
  });
});

describe("RelicTray", () => {
  it("renders an icon per relic and reveals name + description on hover", () => {
    render(<RelicTray relics={[
      { name: "Burning Blood", description: "Heal 6 HP.", image: "/assets/relics/burning_blood.png" },
    ]} />);
    const btn = screen.getByRole("button", { name: "Burning Blood" });
    expect(btn).toBeInTheDocument();
    fireEvent.mouseEnter(btn);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Heal 6 HP.");
  });

  it("renders nothing when there are no relics", () => {
    const { container } = render(<RelicTray relics={[]} />);
    expect(container.firstChild).toBeNull();
  });
});

describe("PotionTray empty slots", () => {
  it("pads to max_potion_slot_count with empty slots", () => {
    const step = pileStep();
    step.state.player.potions = [{ index: 0, name: "Fire Potion", description: "Deal 20 damage." }];
    step.state.player.max_potion_slot_count = 3;
    render(<GameHud state={step.state} onUsePotion={vi.fn()} />);
    expect(screen.getByText("Fire Potion")).toBeInTheDocument();
    expect(screen.getAllByLabelText("empty potion slot")).toHaveLength(2);
  });
});
