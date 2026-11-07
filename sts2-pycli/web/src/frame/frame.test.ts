import { describe, expect, it } from "vitest";
import {
  DEFAULT_LAYOUT,
  dockOf,
  MIN_SIZE,
  movePanel,
  normalizeLayout,
  PANELS,
  setPanelHidden,
  visiblePanels,
} from "./layout";
import { hotkeyAction } from "./host";
import { addChip, expireChips, INITIAL_STATE, reduce, tieBandLines, type Chip } from "./store";

const chip = (id: string, kind = "choice"): Chip => ({
  id,
  kind,
  title: id,
  lines: [],
  tone: "info",
  log_name: null,
  log_idx: null,
  in_combat: false,
});

describe("layout", () => {
  it("moves a panel between docks and within one", () => {
    let l = movePanel(DEFAULT_LAYOUT, "log", "right", 0);
    expect(l.docks.right).toEqual(["log", "advice", "coach"]);
    l = movePanel(l, "coach", "bottom");
    expect(l.docks.bottom).toEqual(["workbench", "coach"]);
    l = movePanel(l, "coach", "right", 2);
    expect(l.docks.right).toEqual(["log", "advice", "coach"]);
    // later within the same dock: index counts the slot before removal
    l = movePanel(l, "log", "right", 3);
    expect(l.docks.right).toEqual(["advice", "coach", "log"]);
    expect(dockOf(l, "log")).toBe("right");
  });

  it("moving a hidden panel shows it", () => {
    const l = movePanel(DEFAULT_LAYOUT, "workbench", "left");
    expect(l.hidden).not.toContain("workbench");
    expect(visiblePanels(l, "left")).toEqual(["workbench"]);
  });

  it("repairs a stored layout", () => {
    const l = normalizeLayout({ docks: { top: ["log", "log", "nope"] }, sizes: { top: 5 }, hidden: ["x"] });
    // the duplicate and the unknown go; the missing toolbar rejoins its default dock
    expect(l.docks.top).toEqual(["log", "toolbar"]);
    for (const p of PANELS) expect(Object.values(l.docks).flat().filter((q) => q === p)).toHaveLength(1);
    expect(l.sizes.top).toBe(MIN_SIZE);
    expect(l.hidden).toEqual([]);
    expect(normalizeLayout("garbage")).toBe(DEFAULT_LAYOUT);
  });

  it("hides and shows", () => {
    const l = setPanelHidden(DEFAULT_LAYOUT, "advice", true);
    expect(visiblePanels(l, "right")).toEqual(["coach", "log"]);
    expect(visiblePanels(setPanelHidden(l, "advice", false), "right")).toEqual(["advice", "coach", "log"]);
  });

});

describe("store", () => {
  it("replaces the unpinned chip of the same group, keeps the other group", () => {
    let chips = addChip([], chip("a"), 0);
    chips = addChip(chips, chip("ab", "autobattle"), 0);
    chips = addChip(chips, chip("b"), 0);
    expect(chips.map((c) => c.id)).toEqual(["ab", "b"]);
  });

  it("keeps pinned chips and expires the rest on their timers", () => {
    let s = reduce(INITIAL_STATE, { type: "from_mod", msg: { type: "chip", chip: chip("t", "turn") } }, 0);
    s = reduce(s, { type: "from_mod", msg: { type: "chip", chip: chip("ab", "autobattle") } }, 0);
    expect(expireChips(s.chips, 21_000).map((c) => c.id)).toEqual(["ab"]);
    s = reduce(s, { type: "pin_chip", id: "t", pinned: true });
    expect(expireChips(s.chips, 60_000).map((c) => c.id)).toEqual(["t", "ab"]);
  });

  it("keeps a pin and the chip's age across a mod reconnect", () => {
    let s = reduce(INITIAL_STATE, { type: "from_mod", msg: { type: "chip", chip: chip("p") } }, 0);
    s = reduce(s, { type: "pin_chip", id: "p", pinned: true });
    s = reduce(s, { type: "from_mod", msg: { type: "chip", chip: chip("t", "turn") } }, 0);
    // the welcome: hello, then the chips the mod still shows, with their age
    s = reduce(s, { type: "from_mod", msg: { type: "hello", protocol: 1, actions: [] } }, 15_000);
    s = reduce(s, { type: "from_mod", msg: { type: "chip", chip: chip("p"), age_ms: 15_000 } }, 15_000);
    s = reduce(s, { type: "from_mod", msg: { type: "chip", chip: chip("t", "turn"), age_ms: 15_000 } }, 15_000);
    expect(s.chips.find((c) => c.id === "p")?.pinned).toBe(true);
    // the turn chip's 20 s run from when the mod showed it, not from the replay
    expect(expireChips(s.chips, 21_000).map((c) => c.id)).toEqual(["p"]);
  });

  it("withdraw removes pinned chips too", () => {
    let s = reduce(INITIAL_STATE, { type: "from_mod", msg: { type: "chip", chip: chip("a") } }, 0);
    s = reduce(s, { type: "pin_chip", id: "a", pinned: true });
    s = reduce(s, { type: "from_mod", msg: { type: "chip_withdraw", ids: ["a"] } });
    expect(s.chips).toEqual([]);
  });

  it("hides the coach panel only for its own id", () => {
    const panel = { id: "p", kind: "recap", title: "", summary: "", sections: [], footer: "", log_name: null };
    let s = reduce(INITIAL_STATE, { type: "from_mod", msg: { type: "coach_panel", panel } });
    s = reduce(s, { type: "from_mod", msg: { type: "coach_panel_hide", ids: ["other"] } });
    expect(s.coachPanel).not.toBeNull();
    s = reduce(s, { type: "from_mod", msg: { type: "coach_panel_hide", ids: null } });
    expect(s.coachPanel).toBeNull();
  });

  it("matches TieBandText", () => {
    expect(tieBandLines(["A"], "r", 3)).toEqual([]);
    expect(tieBandLines(["A", "B", "C", "D"], "keeps block", 2.6)).toEqual([
      "about equal: A ≈ B ≈ C (+1) (~3 HP)",
      "→ A: keeps block",
    ]);
    expect(tieBandLines(["A", "B"], null, 0.2)).toEqual(["about equal: A ≈ B"]);
  });
});

describe("hotkeys", () => {
  const key = (k: string, mods: Partial<{ ctrlKey: boolean; shiftKey: boolean; altKey: boolean }> = {}) =>
    hotkeyAction({ key: k, ctrlKey: false, shiftKey: false, altKey: false, ...mods });

  it("maps the workbench's play keys to the mod's verbs", () => {
    expect(key("g")).toBe("ai_once");
    expect(key("p")).toBe("peek");
    expect(key("h")).toBe("peek");
    expect(key("f")).toBe("ai_auto");
    expect(key("a")).toBe("ai_auto");
    expect(key("?", { shiftKey: true })).toBe("help");
  });

  it("keeps the mod's own keys and leaves chords and the game's keys alone", () => {
    expect(key("F6")).toBe("ai_once");
    expect(key("F6", { shiftKey: true })).toBe("partner_go");
    expect(key("z", { ctrlKey: true })).toBe("undo_turn");
    expect(key("g", { ctrlKey: true })).toBeNull();
    expect(key("G", { shiftKey: true })).toBeNull();
    expect(key("e")).toBeNull();
    expect(key("1")).toBeNull();
  });
});
