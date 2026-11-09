import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HintsProvider } from "../HintsContext";
import { useHints } from "../useHints";

function Probe() {
  const { on, style, toggle } = useHints();
  return (
    <button onClick={toggle} data-testid="probe">
      {style} / hints: {on ? "on" : "off"}
    </button>
  );
}

/** jsdom here exposes a `localStorage` object with no methods, so the suite
 *  supplies its own in-memory store — which is also what makes the persistence
 *  contract observable. */
function memoryStorage(seed: Record<string, string> = {}) {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
    dump: () => Object.fromEntries(map),
  };
}

beforeEach(() => vi.stubGlobal("localStorage", memoryStorage()));
afterEach(() => vi.unstubAllGlobals());

function mount() {
  return render(
    <HintsProvider>
      <Probe />
    </HintsProvider>,
  );
}

describe("HintsContext", () => {
  it("a fresh browser starts in Hints, as the mod does", () => {
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("hints / hints: on");
  });

  it("defaults to Coach without a provider, so an ungated subtree cannot leak", () => {
    render(<Probe />);
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
  });

  it("switching style persists across a remount", () => {
    const store = memoryStorage();
    vi.stubGlobal("localStorage", store);
    const { unmount } = mount();
    fireEvent.click(screen.getByTestId("probe"));
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
    expect(store.dump().adviceStyle).toBe("coach");
    unmount();
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
  });

  it("migrates the old hints switch: on becomes Hints", () => {
    vi.stubGlobal("localStorage", memoryStorage({ hintsOn: "1" }));
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("hints / hints: on");
  });

  it("migrates the old hints switch: off becomes Coach", () => {
    vi.stubGlobal("localStorage", memoryStorage({ hintsOn: "0" }));
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
  });

  it("a stored style wins over the old key", () => {
    vi.stubGlobal("localStorage", memoryStorage({ hintsOn: "1", adviceStyle: "coach" }));
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
  });

  it("survives storage being unavailable (private mode), in Hints", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    });
    mount();
    expect(screen.getByTestId("probe")).toHaveTextContent("hints / hints: on");
    fireEvent.click(screen.getByTestId("probe"));
    expect(screen.getByTestId("probe")).toHaveTextContent("coach / hints: off");
  });
});

describe("Off", () => {
  it("is a stored style of its own, quiet and not Hints", () => {
    vi.stubGlobal("localStorage", memoryStorage({ adviceStyle: "off" }));
    function OffProbe() {
      const h = useHints();
      return <span data-testid="off">{h.style}/{String(h.on)}/{String(h.off)}</span>;
    }
    render(<HintsProvider><OffProbe /></HintsProvider>);
    expect(screen.getByTestId("off")).toHaveTextContent("off/false/true");
  });
});
