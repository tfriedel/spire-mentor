import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { InspectProvider } from "../inspect/InspectContext";
import InspectSheet from "../inspect/InspectSheet";
import RelicTray from "./RelicTray";

const RELICS = ["A", "B", "C", "D", "E"].map((n) => ({ name: `Relic ${n}`, description: `desc ${n}` }));

function mockPhone(on: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: on, media: q, addEventListener() {}, removeEventListener() {},
  }));
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("RelicTray maxShown", () => {
  it("collapses the overflow into one +N button in the last slot", () => {
    mockPhone(false);
    render(<RelicTray relics={RELICS} skin="hud" maxShown={4} />);
    // 4 slots: three relics and the +2 button.
    expect(screen.getAllByRole("button")).toHaveLength(4);
    const more = screen.getByRole("button", { name: "2 more relics" });
    expect(more).toHaveClass("gs-hud-relic-more");
    expect(more).toHaveTextContent("+2");
  });

  it("never shows fewer than three relics, even when the room fits fewer", () => {
    mockPhone(false);
    render(<RelicTray relics={RELICS} skin="hud" maxShown={2} />);
    expect(screen.getAllByRole("button", { name: /^Relic / })).toHaveLength(3);
    expect(screen.getByRole("button", { name: "2 more relics" })).toBeInTheDocument();
  });

  it("shows everything when it fits", () => {
    mockPhone(false);
    render(<RelicTray relics={RELICS} skin="hud" maxShown={5} />);
    expect(screen.getAllByRole("button")).toHaveLength(5);
    expect(document.querySelector(".gs-hud-relic-more")).toBeNull();
  });

  it("without the inspect sheet the +N button expands the tray in place", () => {
    mockPhone(false);
    render(<RelicTray relics={RELICS} skin="hud" maxShown={4} />);
    fireEvent.click(screen.getByRole("button", { name: "2 more relics" }));
    expect(screen.getByRole("button", { name: "Relic E" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "fewer relics" }));
    expect(screen.queryByRole("button", { name: "Relic E" })).toBeNull();
  });

  it("on a phone with the sheet mounted +N opens a list of the hidden relics", () => {
    mockPhone(true);
    render(
      <InspectProvider>
        <RelicTray relics={RELICS} skin="hud" maxShown={4} />
        <InspectSheet />
      </InspectProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "2 more relics" }));
    const rows = screen.getByRole("dialog").querySelectorAll(".gs-inspect-list > *");
    expect(rows).toHaveLength(2);
    expect(screen.getByRole("dialog")).toHaveTextContent("Relic E");
    expect(screen.getByRole("dialog")).not.toHaveTextContent("Relic A");
  });
});
