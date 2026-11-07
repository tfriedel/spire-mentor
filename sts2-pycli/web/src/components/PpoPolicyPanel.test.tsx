import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import PpoPolicyPanel from "./PpoPolicyPanel";
import type { ReactElement } from "react";
import { HintsFixture } from "../hints/HintsContext";
import type { PpoPolicyPayload } from "../api";

const PAYLOAD: PpoPolicyPayload = {
  ppo_available: true,
  impl: "rust",
  surface: "event",
  value: 0.523,
  error: null,
  options: [
    { action: "choose", args: { index: 0 }, label: "Take blessing", prob: 0.82, is_best: true },
    { action: "choose", args: { index: 1 }, label: "Skip", prob: 0.18, is_best: false },
  ],
};

// Route each fetched URL to its canned payload, mirroring CardReward.test.tsx.
function stubFetch(byUrl: (url: string) => unknown) {
  const spy = vi.fn(async (url: string) => ({ ok: true, json: async () => byUrl(url) }) as Response);
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => {
  vi.unstubAllGlobals();
  // The collapse/height prefs persist, so a test that folds the panel would
  // otherwise fold it for every test after it.
  localStorage.clear();
});

// The panel is gated on the global hints switch (default OFF), so every test
// that asserts on its content must opt in — exactly as the user must.
function renderPanel(ui: ReactElement, on = true) {
  return render(<HintsFixture on={on}>{ui}</HintsFixture>);
}

describe("PpoPolicyPanel", () => {
  it("renders option probabilities when available", async () => {
    stubFetch(() => PAYLOAD);
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />);
    await waitFor(() => expect(screen.getByText("Take blessing")).toBeInTheDocument());
    expect(screen.getByText("82.0%")).toBeInTheDocument();
    expect(screen.getByText(/V\(s\)/)).toBeInTheDocument();
  });

  it("impl toggle posts run-mode", async () => {
    const spy = stubFetch((url) =>
      url.includes("run-mode") ? { ppo_impl: "python" } : PAYLOAD,
    );
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />);
    await waitFor(() => expect(screen.getByText("rust")).toBeInTheDocument());
    fireEvent.click(screen.getByText("rust"));
    await waitFor(() =>
      expect(spy.mock.calls.some(([u]) => String(u).includes("/api/live/run-mode"))).toBe(true),
    );
  });

  it("renders nothing — and fetches nothing — while hints are off", async () => {
    const spy = stubFetch(() => PAYLOAD);
    const { container } = renderPanel(
      <PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />,
      false,
    );
    expect(container.firstChild).toBeNull();
    // The policy call must not even be issued: the payload names the net's
    // preferred option, and autobattle mode grades you on not having seen it.
    await new Promise((r) => setTimeout(r, 0));
    expect(spy.mock.calls.some(([u]) => String(u).includes("ppo-policy"))).toBe(false);
  });

  it("renders nothing in combat", () => {
    stubFetch(() => PAYLOAD);
    const { container } = renderPanel(<PpoPolicyPanel live={{ decision: "combat_play", meta: { floor: 1 } }} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the unavailable message instead of hiding", async () => {
    const unavailable: PpoPolicyPayload = {
      ...PAYLOAD,
      ppo_available: false,
      impl: "rust",
      error: "no ONNX",
    };
    stubFetch(() => unavailable);
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />);
    await waitFor(() => expect(screen.getByText(/PPO net unavailable/)).toBeInTheDocument());
  });
});

// The dock shipped as `position: sticky; bottom: 0` INSIDE the scrolling
// screen box, which keeps it in view the way an overlay does: laid over the
// bottom of whatever screen is up. On a card reward that bottom strip is the
// advisor rail — the per-option EV chips the "analyze picks" button exists to
// produce — so the panel covered the evidence it was sitting next to, and even
// folded to its header it clipped the chips. It is now a FLOW dock: App renders
// it as a sibling of the scroll box, which is `flex: 1` against it, so the
// screen re-fits into what is left. Nothing here may take `position` back.
it("is a flow dock, never positioned over the screen above it", async () => {
  stubFetch(() => PAYLOAD);
  const { container } = renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 3 } }} />);
  await waitFor(() => expect(screen.getByText("PPO backend")).toBeInTheDocument());
  const root = container.firstChild as HTMLElement;
  expect(root.style.position).toBe("");
  expect(root.style.zIndex).toBe("");
  // Sized by its own content in the column, never stretched or squeezed by it.
  expect(root.style.flex).toBe("0 0 auto");
  // Opaque and separated from the screen it sits under.
  expect(root.style.background).not.toBe("");
  expect(root.style.borderTop).not.toBe("");
});

// A shop lists every buy AND every removal target — thirty-odd rows, enough to
// push the board itself off a tablet screen back when this dock grew with its
// content. It now caps, folds and drags.
describe("PpoPolicyPanel space controls", () => {
  it("caps its body and scrolls inside instead of growing with the option list", async () => {
    stubFetch(() => PAYLOAD);
    renderPanel(<PpoPolicyPanel live={{ decision: "shop", meta: { floor: 5 } }} />);
    const body = (await screen.findByText(/V\(s\)/)).parentElement as HTMLElement;
    expect(body.style.overflowY).toBe("auto");
    // Two caps: the dragged px preference AND a viewport share no stored
    // height can exceed — the game must keep more than half the screen.
    expect(body.style.maxHeight).toMatch(/min\(\d+px, 45vh\)/);
  });

  it("folds to a header that still names the net's pick, and remembers it", async () => {
    stubFetch(() => PAYLOAD);
    const { unmount } = renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />);
    await waitFor(() => expect(screen.getByText("Take blessing")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /collapse the PPO panel/ }));
    // The rows and V(s) are gone; the one line the panel exists to say — the
    // net's pick and how sure it is — rides along in the header.
    expect(screen.queryByText("Skip")).not.toBeInTheDocument();
    expect(screen.queryByText(/V\(s\)/)).not.toBeInTheDocument();
    expect(screen.getByText("Take blessing")).toBeInTheDocument();
    expect(screen.getByText(/82\.0%/)).toBeInTheDocument();
    expect(screen.getByText("PPO backend")).toBeInTheDocument();

    // Folded stays folded on the next screen — a dock someone tucked away must
    // not come back every floor.
    unmount();
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 2 } }} />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /expand the PPO panel/ })).toBeInTheDocument(),
    );
    expect(screen.queryByText("Skip")).not.toBeInTheDocument();
  });

  it("offers a resize grip while open (and none while folded)", async () => {
    stubFetch(() => PAYLOAD);
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 1 } }} />);
    await waitFor(() => expect(screen.getByText("Take blessing")).toBeInTheDocument());
    const grip = screen.getByRole("separator", { name: /resize the PPO panel/ });
    fireEvent.keyDown(grip, { key: "ArrowUp" });
    expect(Number(localStorage.getItem("wb.ppo.height"))).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: /collapse the PPO panel/ }));
    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
  });

  it("under the e2e advisor it is the e2e net's panel: its classes, no V(s), no backend toggle", async () => {
    stubFetch(() => ({
      ...PAYLOAD, policy: "e2e", impl: "e2e", value: null,
      options: [
        { action: "select_card_reward", args: { card_index: 1 }, label: "Take Inflame", prob: 0.71, is_best: true },
        { action: "skip_card_reward", args: {}, label: "Skip", prob: 0.29, is_best: false },
      ],
    }));
    renderPanel(<PpoPolicyPanel live={{ decision: "card_reward", meta: { floor: 2 } }} />);
    await waitFor(() => expect(screen.getByText("Take Inflame")).toBeInTheDocument());
    expect(screen.getByText("Instinct")).toBeInTheDocument();
    expect(screen.getByText("71.0%")).toBeInTheDocument();
    expect(screen.queryByText("PPO backend")).toBeNull();
    expect(screen.queryByText(/V\(s\)/)).toBeNull();
    expect(screen.queryByText("rust")).toBeNull();
  });

  it("says why the e2e net has no pick", async () => {
    stubFetch(() => ({
      ...PAYLOAD, policy: "e2e", impl: "e2e", value: null, ppo_available: false, options: [],
      error: "Instinct cannot read this position: unencodable action kind",
    }));
    renderPanel(<PpoPolicyPanel live={{ decision: "event", meta: { floor: 2 } }} />);
    await waitFor(() => expect(screen.getByText(/Instinct has no pick here: Instinct cannot read/)).toBeInTheDocument());
  });
});
