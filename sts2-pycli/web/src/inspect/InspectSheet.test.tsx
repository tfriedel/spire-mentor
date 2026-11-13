import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { InspectProvider } from "./InspectContext";
import { useInspect, useInspectSuppress, type InspectItem } from "./useInspect";
import { powerItem } from "./powerItem";
import InspectSheet from "./InspectSheet";
import RelicTray from "../components/RelicTray";

const POWER: InspectItem = { key: "p:vuln", name: "Vulnerable", amount: 2, description: "Takes 50% more damage." };

function Opener({ item }: { item: InspectItem }) {
  const { open, enabled } = useInspect();
  return <button onClick={() => open(item)}>open {String(enabled)}</button>;
}

function mockPhone(on: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: on, media: q, addEventListener() {}, removeEventListener() {},
  }));
}

beforeEach(() => mockPhone(true));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function ui(item: InspectItem, disabled = false) {
  return (
    <InspectProvider disabled={disabled}>
      <div><Opener item={item} /><InspectSheet /></div>
    </InspectProvider>
  );
}

describe("InspectSheet", () => {
  it("opens with name, amount and description", () => {
    render(ui(POWER));
    fireEvent.click(screen.getByRole("button"));
    const sheet = screen.getByRole("dialog");
    expect(sheet).toHaveClass("gs-inspect");
    expect(sheet).toHaveTextContent("Vulnerable");
    expect(sheet).toHaveTextContent("2");
    expect(sheet).toHaveTextContent("Takes 50% more damage.");
  });

  it("opening the same key again closes it", () => {
    render(ui(POWER));
    fireEvent.click(screen.getByRole("button"));
    fireEvent.click(screen.getByRole("button"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("tapping the sheet closes it", () => {
    render(ui(POWER));
    fireEvent.click(screen.getByRole("button"));
    fireEvent.click(screen.getByRole("dialog"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("renders one row per list entry", () => {
    const list = [1, 2, 3].map((n) => ({ key: `l${n}`, name: `Power ${n}`, amount: n }));
    render(ui({ key: "more", name: "Powers", list }));
    fireEvent.click(screen.getByRole("button"));
    const rows = screen.getByRole("dialog").querySelectorAll(".gs-inspect-list > *");
    expect(rows).toHaveLength(3);
  });

  it("disabled makes open a no-op", () => {
    render(ui(POWER, true));
    fireEvent.click(screen.getByRole("button"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("is not enabled off a phone", () => {
    mockPhone(false);
    render(ui(POWER));
    fireEvent.click(screen.getByRole("button"));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button")).toHaveTextContent("false");
  });
});

describe("powerItem", () => {
  const vuln = { name: "Vulnerable", amount: 2, description: "x", image: null } as never;
  it("scopes the key by creature", () => {
    expect(powerItem(vuln, "a").key).not.toBe(powerItem(vuln, "b").key);
    expect(powerItem(vuln, "a").key).toBe(powerItem(vuln, "a").key);
  });
  it("the same power on another creature switches the sheet instead of closing it", () => {
    function Two() {
      const { open } = useInspect();
      return <>
        <button onClick={() => open(powerItem(vuln, "a"))}>a</button>
        <button onClick={() => open(powerItem(vuln, "b"))}>b</button>
      </>;
    }
    render(<InspectProvider><Two /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByText("a"));
    fireEvent.click(screen.getByText("b"));
    expect(screen.getByRole("dialog")).toHaveTextContent("Vulnerable");
  });
});

describe("Escape", () => {
  it("closes the sheet", () => {
    render(ui(POWER));
    fireEvent.click(screen.getByRole("button"));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("useInspectSuppress", () => {
  function Armed({ on }: { on: boolean }) {
    useInspectSuppress(on);
    return null;
  }
  const withArm = (on: boolean) => (
    <InspectProvider><Armed on={on} /><div><Opener item={POWER} /><InspectSheet /></div></InspectProvider>
  );

  it("disables inspect while on, clears an open item, re-enables on release", () => {
    const { rerender } = render(withArm(false));
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    rerender(withArm(true));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button")).toHaveTextContent("false");
    fireEvent.click(screen.getByRole("button"));
    expect(screen.queryByRole("dialog")).toBeNull();
    rerender(withArm(false));
    expect(screen.getByRole("button")).toHaveTextContent("true");
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });
});

describe("outside a provider", () => {
  it("useInspect is a disabled no-op", () => {
    render(<Opener item={POWER} />);
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("button")).toHaveTextContent("false");
  });
});

describe("RelicTray inspect", () => {
  const relics = [{ name: "Burning Blood", description: "Heal 6 HP.", image: null }] as never;

  it("a tap opens the sheet with the relic and no popover", () => {
    render(
      <InspectProvider><RelicTray relics={relics} skin="hud" /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Burning Blood" }));
    const sheet = screen.getByRole("dialog");
    expect(sheet).toHaveTextContent("Burning Blood");
    expect(sheet).toHaveTextContent("Heal 6 HP.");
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("off a phone keeps its popover", () => {
    mockPhone(false);
    render(<InspectProvider><RelicTray relics={relics} skin="hud" /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Burning Blood" }));
    expect(screen.getByRole("tooltip")).toHaveTextContent("Heal 6 HP.");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("on a phone with the sheet off (a card armed) a tap does nothing: no clipped popover", () => {
    render(<InspectProvider disabled><RelicTray relics={relics} skin="hud" /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Burning Blood" }));
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("a relic tap still explains the relic while a card is armed, and the arm keeps it open", () => {
    // The owner's phone: a card armed (a tap on an attack arms it) and a tap on
    // a relic in the HUD did nothing at all.
    function Armed({ on }: { on: boolean }) { useInspectSuppress(on); return null; }
    const tree = (on: boolean) => (
      <InspectProvider><Armed on={on} /><RelicTray relics={relics} skin="hud" /><InspectSheet /></InspectProvider>);
    const { rerender } = render(tree(true));
    fireEvent.click(screen.getByRole("button", { name: "Burning Blood" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("Heal 6 HP.");
    rerender(tree(false));
    expect(screen.getByRole("dialog")).toHaveTextContent("Heal 6 HP.");
    rerender(tree(true));
    expect(screen.getByRole("dialog")).toHaveTextContent("Heal 6 HP.");
  });

  it("a relic with no text says so instead of showing a bare name", () => {
    const bare = [{ name: "Mystery Relic", image: null }] as never;
    render(<InspectProvider><RelicTray relics={bare} skin="hud" /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Mystery Relic" }));
    expect(screen.getByRole("dialog")).toHaveTextContent("Mystery RelicNo description available");
  });

  it("on a phone with the sheet off \"+N\" never expands the tray in place", () => {
    const many = [1, 2, 3, 4, 5, 6].map((n) => ({ name: `Relic ${n}`, description: "x", image: null })) as never;
    render(<InspectProvider disabled><RelicTray relics={many} skin="hud" maxShown={4} /><InspectSheet /></InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "3 more relics" }));
    expect(screen.queryByRole("button", { name: "Relic 6" })).toBeNull();
    expect(screen.getByRole("button", { name: "3 more relics" })).toBeInTheDocument();
  });

  it("the screen host (no board) opens on a relic tap and closes on a tap elsewhere", () => {
    render(
      <InspectProvider>
        <RelicTray relics={relics} skin="hud" />
        <button type="button">elsewhere</button>
        <div><InspectSheet screen /></div>
      </InspectProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Burning Blood" }));
    const sheet = screen.getByRole("dialog");
    expect(sheet).toHaveTextContent("Heal 6 HP.");
    expect(sheet).toHaveAttribute("aria-live", "polite");
    expect(sheet.style.position).toBe("fixed");
    fireEvent.click(screen.getByRole("button", { name: "elsewhere" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
