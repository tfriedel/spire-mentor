import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import type { Step } from "../api";
import EventScreen from "./EventScreen";
import Shop from "./Shop";
import RestSite from "./RestSite";
import CardSelect from "./CardSelect";
import DecisionScreen from "./DecisionScreen";
import RichText from "./RichText";

/** Build a Step with the given decision + state fields (other fields are unused by screens). */
function step(decision: string, state: Record<string, unknown>): Step {
  return {
    idx: 0,
    decision,
    state: { decision, ...state } as Step["state"],
    scores: [],
    pv: null,
    chosen_action: null,
    source: "live",
    meta: { turn: null, step_no: 0 },
  };
}

describe("EventScreen", () => {
  const s = step("event_choice", {
    event_name: "This or That?",
    description: "A choice.",
    art: { illustration: "/assets/events/this_or_that.png" },
    event_options: [
      { index: 0, title: "This", description: "Lose 6 HP.", is_locked: false },
      { index: 1, title: "That", description: "Locked.", is_locked: true },
    ],
  });

  it("renders the illustration as a backdrop and the title", () => {
    const { container } = render(<EventScreen step={s} onChoose={() => {}} />);
    expect(screen.getByText("This or That?")).toBeInTheDocument();
    // The backdrop rides the shared .gs-scene primitive via its --gs-scene-bg var.
    const scene = container.querySelector(".gs-scene") as HTMLElement;
    expect(scene.style.getPropertyValue("--gs-scene-bg")).toContain("/assets/events/this_or_that.png");
  });

  it("fires choose_option with the option index and disables locked options", () => {
    const onChoose = vi.fn();
    render(<EventScreen step={s} onChoose={onChoose} />);
    fireEvent.click(screen.getByRole("button", { name: /This/ }));
    expect(onChoose).toHaveBeenCalledWith("choose_option", { option_index: 0 });
    expect(screen.getByRole("button", { name: /That/ })).toBeDisabled();
  });

  it("is read-only when no onChoose (Replay)", () => {
    render(<EventScreen step={s} />);
    expect(screen.getByRole("button", { name: /This/ })).toBeDisabled();
  });

  it("renders an option icon when image is present", () => {
    const withIcon = step("event_choice", {
      event_name: "Neow",
      description: "",
      event_options: [
        { index: 0, title: "WINGED BOOTS", description: "Ignore paths 3 times.", is_locked: false, image: "/assets/relic_atlas/winged_boots.png" },
      ],
    });
    const { container } = render(<EventScreen step={withIcon} onChoose={() => {}} />);
    const img = container.querySelector("img") as HTMLImageElement;
    expect(img).not.toBeNull();
    expect(img.getAttribute("src")).toBe("/assets/relic_atlas/winged_boots.png");
    expect(screen.getByText("Ignore paths 3 times.")).toBeInTheDocument();
  });
});

describe("Shop", () => {
  const s = step("shop", {
    player: { name: "Ironclad", gold: 100 },
    cards: [{ index: 0, id: "CARD.X", name: "Anger", type: "Attack", cost: 50, card_cost: 0, is_stocked: true, image: null, description: "" }],
    relics: [
      { index: 0, name: "Akabeko", cost: 75, is_stocked: true, image: "/assets/relic_atlas/akabeko.png" },
      { index: 1, name: "Pricey", cost: 999, is_stocked: true },
    ],
    potions: [],
    card_removal_cost: 60,
  });

  it("buys an affordable relic and disables an unaffordable one", () => {
    const onChoose = vi.fn();
    render(<Shop step={s} onChoose={onChoose} />);
    fireEvent.click(screen.getByRole("button", { name: /Akabeko/ }));
    expect(onChoose).toHaveBeenCalledWith("buy_relic", { relic_index: 0 });
    expect(screen.getByRole("button", { name: /Pricey/ })).toBeDisabled();
  });

  it("offers leave + card removal", () => {
    const onChoose = vi.fn();
    render(<Shop step={s} onChoose={onChoose} />);
    fireEvent.click(screen.getByRole("button", { name: /Leave shop/ }));
    expect(onChoose).toHaveBeenCalledWith("leave_room", {});
  });
});

describe("RestSite", () => {
  /** Regression: the hero used to be painted BEFORE the logs, on the theory
   *  that the left log occluding the legs would read as "seated". It did not —
   *  it read as the character standing BEHIND the scenery and dissolving into
   *  it, and a rig with a wide silhouette (Regent carries its throne) showed a
   *  slab hovering behind the log. The hero must paint AFTER both logs. */
  it("paints the hero in front of the logs", () => {
    // The hero is only mounted once the scene has a measured height (its rig
    // box is sized in px off it); jsdom measures every box at 0.
    const rect = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockReturnValue({ height: 800, width: 1400, top: 0, left: 0, bottom: 800,
        right: 1400, x: 0, y: 0, toJSON: () => ({}) } as DOMRect);
    try {
      const { container } = render(
        <RestSite step={step("rest_site", {
          player: { name: "Ironclad" },
          rest_options: [{ index: 0, name: "Rest", is_enabled: true }],
        })} onChoose={() => {}} />,
      );
      const scenery = container.querySelector(".gs-rest-scenery") as HTMLElement;
      const kids = Array.from(scenery.children);
      const hero = kids.findIndex((n) => n.classList.contains("gs-rest-hero"));
      const lastLog = kids.map((n) => n.classList.contains("gs-rest-log")).lastIndexOf(true);
      expect(hero).toBeGreaterThan(-1);
      expect(lastLog).toBeGreaterThan(-1);
      expect(hero).toBeGreaterThan(lastLog);
    } finally {
      rect.mockRestore();
    }
  });

  it("fires choose_option for an enabled rest option", () => {
    const onChoose = vi.fn();
    const s = step("rest_site", { rest_options: [{ index: 0, name: "Rest", is_enabled: true }, { index: 1, name: "Smith", is_enabled: false }] });
    render(<RestSite step={s} onChoose={onChoose} />);
    fireEvent.click(screen.getByRole("button", { name: "Rest" }));
    expect(onChoose).toHaveBeenCalledWith("choose_option", { option_index: 0 });
    expect(screen.getByRole("button", { name: "Smith" })).toBeDisabled();
  });
});

describe("CardSelect", () => {
  it("submits selected indices and respects max", () => {
    const onChoose = vi.fn();
    const s = step("card_select", {
      min_select: 1,
      max_select: 1,
      cards: [
        { index: 0, id: "a", name: "A", type: "Skill", cost: 1, image: null, description: "" },
        { index: 1, id: "b", name: "B", type: "Skill", cost: 1, image: null, description: "" },
      ],
    });
    render(<CardSelect step={s} onChoose={onChoose} />);
    // One card to pick: the tap is the pick, with no Confirm button.
    expect(screen.queryByRole("button", { name: /Confirm|Done/ })).toBeNull();
    fireEvent.click(screen.getByText("A").closest("div[data-played], div") as HTMLElement);
    expect(onChoose).toHaveBeenCalledWith("select_cards", { indices: "0" });
  });

  // Regression for wb-1115a56f102d6af1: confirming a SINGLE card on a
  // sequential multi-pick choice (e.g. Ashwater's "choose up to 3 to
  // exhaust") must finalize the choice, not leave it re-opened looking stuck.
  describe("multi-pick (e.g. Ashwater exhaust_from_hand)", () => {
    function multiPickStep() {
      return step("card_select", {
        context: "exhaust_from_hand",
        min_select: 0,
        max_select: 3,
        cards: [
          { index: 0, id: "INFERNO", name: "Inferno", type: "Power", cost: 1, image: null, description: "" },
          { index: 1, id: "HELLRAISER", name: "Hellraiser", type: "Power", cost: 1, image: null, description: "" },
          { index: 2, id: "STRIKE_IRONCLAD", name: "Strike", type: "Attack", cost: 1, image: null, description: "" },
        ],
      });
    }

    it("finalizes a single-card confirm with a follow-up skip_select (accumulate-then-finalize)", async () => {
      const onChoose = vi.fn();
      render(<CardSelect step={multiPickStep()} onChoose={onChoose} />);
      fireEvent.click(screen.getByText("Inferno").closest("div[data-played], div") as HTMLElement);
      fireEvent.click(screen.getByRole("button", { name: /Done/ }));
      expect(onChoose.mock.calls[0]).toEqual(["select_cards", { indices: "0" }]);
      await waitFor(() => expect(onChoose.mock.calls[1]).toEqual(["skip_select", {}]));
      expect(onChoose).toHaveBeenCalledTimes(2);
    });

    it("resolves a 2-card confirm atomically with no follow-up call", async () => {
      const onChoose = vi.fn();
      render(<CardSelect step={multiPickStep()} onChoose={onChoose} />);
      fireEvent.click(screen.getByText("Inferno").closest("div[data-played], div") as HTMLElement);
      fireEvent.click(screen.getByText("Strike").closest("div[data-played], div") as HTMLElement);
      fireEvent.click(screen.getByRole("button", { name: /Done/ }));
      await Promise.resolve();
      expect(onChoose).toHaveBeenCalledTimes(1);
      expect(onChoose).toHaveBeenCalledWith("select_cards", { indices: "0,2" });
    });

    it("routes a 0-card confirm through skip_select", async () => {
      const onChoose = vi.fn();
      render(<CardSelect step={multiPickStep()} onChoose={onChoose} />);
      fireEvent.click(screen.getByRole("button", { name: /Done/ }));
      await waitFor(() => expect(onChoose).toHaveBeenCalledWith("skip_select", {}));
      expect(onChoose).toHaveBeenCalledTimes(1);
    });
  });
});

describe("CardSelect, exactly N cards", () => {
  it("commits on the N-th tap, with no button", () => {
    const onChoose = vi.fn();
    const s = step("card_select", {
      min_select: 2, max_select: 2,
      cards: ["A", "B", "C"].map((name, index) => ({ index, id: name, name, type: "Skill", cost: 1, image: null, description: "" })),
    });
    render(<CardSelect step={s} onChoose={onChoose} />);
    expect(screen.queryByRole("button", { name: /Confirm|Done/ })).toBeNull();
    fireEvent.click(screen.getByText("A").closest("div[data-played], div") as HTMLElement);
    expect(onChoose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("C").closest("div[data-played], div") as HTMLElement);
    expect(onChoose).toHaveBeenCalledTimes(1);
    expect(onChoose).toHaveBeenCalledWith("select_cards", { indices: "0,2" });
  });
});

describe("DecisionScreen routing", () => {
  it("routes shop to the Shop screen", () => {
    render(<DecisionScreen step={step("shop", { player: { name: "X", gold: 0 }, cards: [], relics: [], potions: [] })} />);
    expect(screen.getByText("Merchant")).toBeInTheDocument();
  });
  it("falls back to a muted note for unknown decisions in Replay", () => {
    render(<DecisionScreen step={step("mystery_decision", { options: [] })} />);
    expect(screen.getByText(/mystery_decision/)).toBeInTheDocument();
  });
});

describe("RichText", () => {
  it("colors tagged spans and leaves plain text", () => {
    const { container } = render(<RichText text="Lose [red]6[/red] HP." />);
    expect(container.textContent).toBe("Lose 6 HP.");
    const colored = container.querySelector("span[style*='--hp']");
    expect(colored?.textContent).toBe("6");
  });
});
