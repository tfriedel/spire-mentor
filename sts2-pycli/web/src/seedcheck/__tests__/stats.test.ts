import { describe, expect, it } from "vitest";
import { deathsByFloor, tally, upperBound95, wilson95, type AttemptResult } from "../stats";

function r(k: number, over: Partial<AttemptResult> = {}): AttemptResult {
  return {
    k, seq: "check", temp: 1, won: false, outcome: "lost", error: null, floor: 22, act: 2,
    decisions: 300, classes: [], hp: 0, death_room: "Terror Eel", ...over,
  };
}

describe("stats", () => {
  it("bounds the per-attempt win chance by the rule of three", () => {
    expect(upperBound95(440)).toBeCloseTo(0.00679, 5);
    expect(upperBound95(1)).toBeCloseTo(0.95, 6);
  });

  it("gives the Wilson 95% interval", () => {
    const [lo, hi] = wilson95(36, 200);
    expect(lo).toBeCloseTo(0.133, 2);
    expect(hi).toBeCloseTo(0.239, 2);
    expect(wilson95(0, 0)).toEqual([0, 1]);
  });

  it("groups losses by floor and room, dropping wins and errors", () => {
    const rs = [
      r(0, { won: true, outcome: "won", floor: 46, death_room: null }),
      r(1), r(2), r(3, { floor: 46, death_room: "Queen" }),
      r(4, { outcome: "error", error: "boom" }),
    ];
    expect(deathsByFloor(rs)).toEqual([
      { floor: 22, room: "Terror Eel", count: 2 },
      { floor: 46, room: "Queen", count: 1 },
    ]);
  });

  it("tallies wins, errors and the best attempt", () => {
    const rs = [r(3, { floor: 40 }), r(1, { floor: 40 }), r(2, { outcome: "stopped" })];
    const t = tally(rs);
    expect(t).toMatchObject({ played: 3, wins: 0, errored: 1, bestFloor: 40 });
    expect(t.best?.k).toBe(1);
    const w = tally([...rs, r(9, { won: true, outcome: "won", floor: 46 }), r(5, { won: true, outcome: "won", floor: 46 })]);
    expect(w.wins).toBe(2);
    expect(w.best?.k).toBe(5);
  });

  it("never picks an unfinished attempt as the best", () => {
    const t = tally([r(0, { floor: 20 }), r(1, { outcome: "stopped", floor: 45 }), r(2, { outcome: "error", floor: 30 })]);
    expect(t.best?.k).toBe(0);
    expect(t.bestFloor).toBe(20);
  });
});
