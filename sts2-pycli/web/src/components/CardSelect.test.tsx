import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import CardSelect from "./CardSelect";
import { cardBoxHeight } from "../cardGeometry";
import { isCombatChoiceStep } from "../choiceKinds";
import { HintsFixture } from "../hints/HintsContext";
import { MarksFixture } from "../advice/AdviceContext";
import type { Step } from "../api";

/** Run `body` with `window.matchMedia` answering true for exactly `query`.
 *  jsdom has no matchMedia at all, which is why the rest of this file gets the
 *  desktop path (see useMediaQuery) without doing anything. */
function withMediaQuery(query: string, body: () => void): void {
  const win = window as unknown as { matchMedia?: unknown };
  const had = "matchMedia" in win;
  const prev = win.matchMedia;
  win.matchMedia = (q: string) => ({
    matches: q === query,
    media: q,
    addEventListener() {},
    removeEventListener() {},
  });
  try {
    body();
  } finally {
    if (had) win.matchMedia = prev;
    else delete win.matchMedia;
  }
}

function step(context: string | undefined): Step {
  return {
    decision: "card_select",
    state: {
      decision: "card_select",
      context,
      cards: [
        { index: 0, name: "Strike", id: "STRIKE", upgraded: false },
        { index: 1, name: "Defend", id: "DEFEND", upgraded: false },
      ],
      min_select: 1,
      max_select: 1,
    },
  } as unknown as Step;
}

describe("CardSelect title", () => {
  it("labels the Touch of Insanity choice by purpose, not 'Select cards'", () => {
    render(<CardSelect step={step("make_free_in_hand")} onChoose={() => {}} />);
    expect(screen.getByRole("heading", { name: "Make a card free for this combat" })).toBeTruthy();
  });

  it("labels deck operations (upgrade) with their wording", () => {
    render(<CardSelect step={step("upgrade")} onChoose={() => {}} />);
    expect(screen.getByRole("heading", { name: "Upgrade a card" })).toBeTruthy();
  });

  it("falls back to 'Choose a card' for an unknown/absent context", () => {
    render(<CardSelect step={step(undefined)} onChoose={() => {}} />);
    expect(screen.getByRole("heading", { name: "Choose a card" })).toBeTruthy();
  });
});

// ── before → after upgrade preview (the game's NUpgradePreview) ──
import { fireEvent } from "@testing-library/react";

function upgradeStep(context: string): Step {
  return {
    decision: "card_select",
    state: {
      decision: "card_select",
      context,
      cards: [
        {
          index: 0, name: "Bash", id: "BASH", upgraded: false, description: "Deal 8 damage.",
          upgrade_preview: {
            index: 0, name: "Bash", id: "BASH", upgraded: true, description: "Deal 10 damage.",
          },
        },
        {
          index: 1, name: "Afterimage", id: "AFTERIMAGE", upgraded: false, description: "Gain 1 Block.",
          upgrade_preview: {
            index: 1, name: "Afterimage", id: "AFTERIMAGE", upgraded: true,
            description: "Gain 1 Block.", keyword_delta: "+Innate",
          },
        },
      ],
      min_select: 1,
      max_select: 1,
    },
  } as unknown as Step;
}

describe("CardSelect upgrade preview", () => {
  it("shows the un-upgraded and upgraded card side by side for the hovered option", () => {
    render(<CardSelect step={upgradeStep("upgrade")} onChoose={() => {}} />);
    expect(screen.getByText("Hover or pick a card to preview its upgrade")).toBeTruthy();
    fireEvent.mouseEnter(screen.getByText("Deal 8 damage.").closest("div.gs-card")!.parentElement!);
    expect(screen.getByText("now")).toBeTruthy();
    expect(screen.getByText("upgraded")).toBeTruthy();
    // The preview card carries the upgraded text + the "+" name suffix.
    expect(screen.getByText("Deal 10 damage.")).toBeTruthy();
    expect(screen.getByText("Bash+")).toBeTruthy();
  });

  it("keeps previewing the picked card once the cursor leaves the grid", () => {
    render(<CardSelect step={upgradeStep("upgrade")} onChoose={() => {}} />);
    const card = screen.getByText("Deal 8 damage.").closest("div.gs-card")!;
    fireEvent.click(card);
    fireEvent.mouseLeave(card.parentElement!);
    expect(screen.getByText("Deal 10 damage.")).toBeTruthy();
  });

  it("names the keyword an upgrade adds, the only change on such cards", () => {
    render(<CardSelect step={upgradeStep("upgrade_in_hand")} onChoose={() => {}} />);
    fireEvent.mouseEnter(screen.getByText("Gain 1 Block.").closest("div.gs-card")!.parentElement!);
    expect(screen.getByText("+Innate")).toBeTruthy();
  });

  it("keeps the preview box exactly as tall empty as filled", () => {
    // The card grid sits directly under this box. When the box was a
    // `minHeight` guess ~20px short of its filled layout, every hover pushed
    // the grid down and out from under the cursor — and walking the row made
    // it flicker as the grid chased the pointer.
    render(<CardSelect step={upgradeStep("upgrade")} onChoose={() => {}} />);
    const row = screen.getByTestId("upgrade-preview-row");
    const empty = row.style.height;
    expect(parseInt(empty, 10)).toBeGreaterThan(cardBoxHeight(160));

    fireEvent.mouseEnter(screen.getByText("Deal 8 damage.").closest("div.gs-card")!.parentElement!);
    expect(screen.getByText("now")).toBeTruthy();
    expect(screen.getByTestId("upgrade-preview-row").style.height).toBe(empty);
  });

  it("shows no comparison box for non-upgrade selects", () => {
    render(<CardSelect step={step("remove")} onChoose={() => {}} />);
    expect(screen.queryByText("Hover or pick a card to preview its upgrade")).toBeNull();
  });

  it("collapses the empty box on a touchscreen, where nothing can hover", () => {
    // The reservation above exists only to stop a HOVER moving the grid. With
    // no pointer there is nothing to hold still, and two reserved card heights
    // of empty panel ate ~290px of a 517px tablet screen — the cards under it
    // were pushed off the bottom.
    withMediaQuery("(hover: none)", () => {
      render(<CardSelect step={upgradeStep("upgrade")} onChoose={() => {}} />);
      expect(screen.getByTestId("upgrade-preview-row").style.height).toBe("");
      // …and the copy stops asking for a hover the device cannot do.
      expect(screen.getByText("Pick a card to preview its upgrade")).toBeTruthy();

      // A tap still fills it, at which point it takes its full height again.
      fireEvent.click(screen.getByText("Deal 8 damage.").closest("div.gs-card")!);
      expect(screen.getByText("Deal 10 damage.")).toBeTruthy();
      expect(screen.getByTestId("upgrade-preview-row").style.height).not.toBe("");
    });
  });
});

// ── eager combat-MCTS choice ranker (auto-fetched when the choice opens) ──
import { vi, afterEach } from "vitest";
import { waitFor } from "@testing-library/react";

const RANKING = {
  candidates: [
    { key: "choice:0", label: "Strike", win_prob: 0.61, q: 0.2, visits: 700, is_best: true },
    { key: "choice:1", label: "Defend", win_prob: 0.44, q: 0.1, visits: 300, is_best: false },
  ],
  decision: "card_select",
  act: 1,
  config: {},
};

function stubFetch(handler: (url: string) => unknown) {
  const spy = vi.fn(async (input: RequestInfo | URL) => ({
    ok: true,
    status: 200,
    json: async () => handler(String(input)),
  }));
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => vi.unstubAllGlobals());

describe("CardSelect choice-ranker chips", () => {
  it("rings the ranker's pick like every other pick, its search % hidden", async () => {
    stubFetch(() => RANKING);
    const { container } = render(
      <HintsFixture on>
        <CardSelect step={step("add_generated_free_to_hand")} onChoose={() => {}} />
      </HintsFixture>,
    );
    await waitFor(() => expect(container.querySelector(".gs-mark-best")).not.toBeNull());
    expect(container.querySelector(".gs-mark-best")).toHaveTextContent("1");
    expect(screen.queryByText(/61(.d)?%/)).toBeNull();
  });

  it("fetches the choice ranker eagerly on a live in-combat single-pick and badges options", async () => {
    const spy = stubFetch(() => RANKING);
    render(
      <HintsFixture on>
        <MarksFixture marks={[]} showNumbers>
          <CardSelect step={step("add_generated_free_to_hand")} onChoose={() => {}} />
        </MarksFixture>
      </HintsFixture>,
    );
    await waitFor(() => expect(screen.getByText(/61(\.\d)?%/)).toBeTruthy());
    expect(screen.getByText(/44(\.\d)?%/)).toBeTruthy();
    expect(spy.mock.calls.some(([u]) => String(u).includes("/api/live/choice-ranker"))).toBe(true);
  });

  it("shows no ranker chips in Coach style — that is unprompted advice", async () => {
    stubFetch(() => RANKING);
    render(
      <HintsFixture style="coach">
        <CardSelect step={step("add_generated_free_to_hand")} onChoose={() => {}} />
      </HintsFixture>,
    );
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByText(/61(\.\d)?%/)).toBeNull();
    expect(screen.queryByText(/44(\.\d)?%/)).toBeNull();
  });

  it("does not fetch for deck-op selects (Smith etc.)", () => {
    const spy = stubFetch(() => RANKING);
    render(<CardSelect step={step("upgrade")} onChoose={() => {}} />);
    expect(spy.mock.calls.some(([u]) => String(u).includes("/api/live/choice-ranker"))).toBe(false);
  });

  it("does not fetch in replay (no onChoose)", () => {
    const spy = stubFetch(() => RANKING);
    render(<CardSelect step={step("card_select")} />);
    expect(spy.mock.calls.some(([u]) => String(u).includes("/api/live/choice-ranker"))).toBe(false);
  });
});

describe("CardSelect under the e2e advisor", () => {
  it("badges the options with the e2e net's probability, not a search win chance", async () => {
    stubFetch(() => ({
      ...RANKING,
      advisor: "e2e",
      candidates: [
        { key: "choice:0", label: "Strike", win_prob: 0.83, prob: 0.83, q: 0.66, visits: 0, is_best: true },
        { key: "choice:1", label: "Defend", win_prob: 0.17, prob: 0.17, q: -0.66, visits: 0, is_best: false },
      ],
    }));
    render(
      <HintsFixture on>
        <MarksFixture marks={[]} showNumbers>
          <CardSelect step={step("add_generated_free_to_hand")} onChoose={() => {}} />
        </MarksFixture>
      </HintsFixture>,
    );
    await waitFor(() => expect(screen.getByText("★ Instinct 83.0%")).toBeTruthy());
    expect(screen.getByText("Instinct 17.0%")).toBeTruthy();
    expect(screen.queryByText(/search \d/)).toBeNull();
  });
});

describe("CardSelect hints gate", () => {
  it("does not fetch the choice ranker while hints are off", async () => {
    const spy = stubFetch(() => RANKING);
    render(<CardSelect step={step("add_generated_free_to_hand")} onChoose={() => {}} />);
    await new Promise((r) => setTimeout(r, 0));
    expect(spy.mock.calls.some(([u]) => String(u).includes("/api/live/choice-ranker"))).toBe(false);
    expect(screen.queryByText(/search \d/)).toBeNull();
  });
});

// ── the Skip button: `can_skip` is not the same flag as `min_select === 0` ──
describe("CardSelect skip", () => {
  function skipStep(extra: Record<string, unknown>): Step {
    return {
      decision: "card_select",
      state: {
        decision: "card_select",
        context: "add_generated_free_to_hand",
        cards: [{ index: 0, name: "Strike", id: "STRIKE", upgraded: false }],
        min_select: 1,
        max_select: 1,
        ...extra,
      },
    } as unknown as Step;
  }

  it("offers Skip on a can_skip choice that still wants a card", () => {
    // Discovery and the Colorless / Skill / Attack / Power potions are
    // min 1 / max 1 / can_skip true. Reading `min_select` alone hid the button
    // while the MCTS chips still scored a skip (bug wb-31c2d79f995fe1a6).
    render(<CardSelect step={skipStep({ can_skip: true })} onChoose={() => {}} />);
    expect(screen.getByRole("button", { name: "Skip" })).toBeTruthy();
  });

  it("still offers Skip when nothing is a valid selection (min 0)", () => {
    render(<CardSelect step={skipStep({ min_select: 0 })} onChoose={() => {}} />);
    expect(screen.getByRole("button", { name: "Skip" })).toBeTruthy();
  });

  it("offers no Skip on a mandatory choice", () => {
    render(<CardSelect step={skipStep({ can_skip: false })} onChoose={() => {}} />);
    expect(screen.queryByRole("button", { name: "Skip" })).toBeNull();
  });
});

// ── in-combat vs deck-op classification (drives "play best" multi-pick) ──
describe("isCombatChoiceStep", () => {
  it("separates in-combat pending choices from deck operations", () => {
    expect(isCombatChoiceStep(step("exhaust_from_hand"))).toBe(true);
    expect(isCombatChoiceStep(step("add_generated_free_to_hand"))).toBe(true);
    // Rest-site Smith / Neow removals resolve in one action per press.
    expect(isCombatChoiceStep(step("upgrade"))).toBe(false);
    expect(isCombatChoiceStep(step("remove"))).toBe(false);
    // Not a card_select at all.
    expect(isCombatChoiceStep({ decision: "map_select", state: {} } as unknown as Step)).toBe(false);
  });
});
