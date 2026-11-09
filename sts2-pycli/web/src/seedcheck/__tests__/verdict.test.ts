import { describe, expect, it } from "vitest";
import { checkVerdict, rateVerdict, routeLines, type RouteStep, type Summary } from "../verdict";
import { tally, type AttemptResult } from "../stats";

const summary: Summary = {
  won: true, outcome: "won", hp: 40, max_hp: 107, floor: 46,
  neow_pick: "Lava Rock", boss_relics: ["Pael's Legion", "Claws"],
  events: [{ floor: 2, event: "Brain Leech", option: "Rip" }],
  cards_taken: [{ floor: 1, card: "Anger" }],
  shop: [{ floor: 36, bought: "Anchor" }, { floor: 36, removed: "Doubt" }],
  rests: [{ floor: 6, choice: "SMITH", card: "Omnislice" }],
  deck: ["Strike", "Strike", "Defend", "Bash", "Ascender's Bane", "Demon Form", "Maul", "Maul"],
  relics: ["Burning Blood", "Lava Rock"], death: null,
};

function r(k: number, over: Partial<AttemptResult> = {}): AttemptResult {
  return {
    k, seq: "check", temp: k === 0 ? 0 : 1, won: false, outcome: "lost", error: null, floor: 22,
    act: 2, decisions: 300, classes: [], hp: 0, death_room: "Terror Eel", ...over,
  };
}

const base = { seed: "1SNVTCEXKR0Q", build: "wb-test" };

describe("verdict", () => {
  it("states a win with its conditions and line", () => {
    const rs = [r(0, { won: true, outcome: "won", floor: 46, death_room: null })];
    const t = checkVerdict({ ...base, tally: tally(rs), firstWin: rs[0], summary, deaths: [] });
    for (const s of ["1SNVTCEXKR0Q", "Ironclad", "A10", "v0.111.0", "simulator", "wb-test",
      "won on attempt 0 (greedy)", "Lava Rock", "Pael's Legion", "Demon Form", "Maul ×2", "40/107"]) {
      expect(t).toContain(s);
    }
    expect(t).not.toMatch(/impossible|unwinnable/i);
    expect(t).not.toContain("Strike");
  });

  it("states a miss as a bound, never as impossible", () => {
    const rs = Array.from({ length: 440 }, (_, k) => r(k, k % 2 ? { floor: 46, death_room: "Queen" } : {}));
    rs[7] = r(7, { outcome: "error", error: "boom" });
    const t = checkVerdict({
      ...base, tally: tally(rs), firstWin: null, summary: null,
      deaths: [{ floor: 46, room: "Queen", count: 220 }, { floor: 22, room: "Terror Eel", count: 219 }],
    });
    expect(t).toContain("No win in 439 attempts");
    expect(t).toContain("below 0.7%");
    expect(t).toContain("floor 46 (Queen, 220)");
    expect(t).toContain("1 attempt ended in an engine error");
    expect(t).not.toMatch(/impossible|unwinnable/i);
  });

  it("states a sampled win rate", () => {
    const t = rateVerdict({ ...base, wins: 36, n: 200, deaths: [{ floor: 46, room: "Queen", count: 80 }] });
    expect(t).toContain("won 36 of 200 sampled runs (18%");
    expect(t).toContain("95% interval 13–24%");
    expect(t).toContain("floor 46 (Queen, 80)");
  });

  it("claims nothing when no attempt finished", () => {
    const rs = [r(0, { outcome: "error", error: "boom" })];
    const t = checkVerdict({ ...base, tally: tally(rs), firstWin: null, summary: null, deaths: [] });
    expect(t).toContain("No attempt finished");
    expect(t).not.toMatch(/below|No win in/);
  });

  it("writes the route floor by floor so a reader can follow it", () => {
    const step = (o: Partial<RouteStep>): RouteStep => ({
      act: 1, floor: 1, room: "Monster", pos: 1, of: 3, dir: null, options: 3, name: null, notes: [], hp: 62, max_hp: 80, ...o,
    });
    const route = [
      step({ name: "Shrinker Beetle", notes: ["took Anger"] }),
      step({ floor: 2, room: "Unknown", kind: "event", pos: 1, of: 4, dir: "straight", options: 2, name: "Brain Leech", notes: ["Rip", "took Omnislice"], hp: 57 }),
      step({ floor: 3, room: "Treasure", pos: 2, of: 3, dir: "right", options: 1, notes: ["got Red Mask"] }),
      step({ floor: 16, room: "Boss", pos: 1, of: 1, dir: null, options: 1, name: "Vantom", notes: ["boss relic: Pael's Legion"], hp: 70 }),
      step({ act: 2, floor: 17, pos: 3, of: 3, name: "Bowlbugs", notes: ["skipped the card"], hp: 74 }),
    ];
    expect(routeLines(route)).toEqual([
      "Act 1:",
      "",
      "- F1 Fight, start on the 1st of 3 from the left: Shrinker Beetle; took Anger (HP 62/80)",
      "- F2 ? (event), go straight (1st of 4 from the left): Brain Leech; Rip; took Omnislice (HP 57/80)",
      "- F3 Chest, only path (2nd of 3 from the left): got Red Mask (HP 62/80)",
      "- F16 Boss: Vantom; boss relic: Pael's Legion (HP 70/80)",
      "",
      "Act 2:",
      "",
      "- F17 Fight, start on the 3rd of 3 from the left: Bowlbugs; skipped the card (HP 74/80)",
    ]);
    const rs = [r(0, { won: true, outcome: "won", floor: 46, death_room: null })];
    const t = checkVerdict({
      ...base, conditions: { character: "Ironclad", ascension: 7, version: "v0.111.0" },
      tally: tally(rs), firstWin: rs[0], summary: { ...summary, route }, deaths: [],
    });
    expect(t).toContain("Ironclad A7, v0.111.0");
    expect(t).toContain("one line per floor");
    expect(t).toContain("- F2 ? (event), go straight (1st of 4 from the left)");
    // The route carries events and shops; the overview does not repeat them.
    expect(t).not.toContain("Events:");
    // Paragraphs are blank-line separated (Reddit joins single newlines).
    expect(t.split("\n\n").length).toBeGreaterThan(4);
  });

  it("titles the route of a best attempt as such", () => {
    const rs = [r(0, { floor: 30 })];
    const route: RouteStep[] = [
      { act: 1, floor: 1, room: "Monster", pos: 1, of: 2, dir: null, options: 2, name: "Nibbits", notes: [], hp: 70, max_hp: 80 },
    ];
    const t = checkVerdict({ ...base, tally: tally(rs), firstWin: null, summary: { ...summary, won: false, route }, deaths: [] });
    expect(t).toContain("The best attempt's route: one line per floor");
    expect(t).not.toContain("Winning line");
  });
});
