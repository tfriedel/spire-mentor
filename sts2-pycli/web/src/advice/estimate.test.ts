import { describe, expect, it } from "vitest";
import { estimateChip, estimateText, hpRangeText } from "./estimate";

describe("estimateText", () => {
  it("is empty without an estimate", () => {
    expect(estimateText(null)).toBe("");
    expect(estimateText(undefined)).toBe("");
  });

  it("gives only the run outside a fight", () => {
    expect(estimateText({ p_run: 0.342, p_fight: null, hp_loss: null, hp_now: 60 })).toBe(
      "If Instinct plays on: run 34%",
    );
  });

  it("gives the fight, its HP range and the run in a fight", () => {
    const e = { p_run: 0.34, p_fight: 0.931, hp_loss: { q10: 2, q25: 5, q50: 9.4, q75: 14, q90: 20 }, hp_now: 50 };
    expect(estimateText(e)).toBe("If Instinct plays on: fight 93% · −9 HP (−2 HP to −20 HP) · run 34%");
  });

  it("shows a net heal as a gain and collapses an empty range", () => {
    const e = { p_run: 1, p_fight: 1, hp_loss: { q10: -6, q25: -6, q50: -6, q75: -6, q90: -6 }, hp_now: 70 };
    expect(estimateText(e)).toBe("If Instinct plays on: fight 100% · +6 HP · run 100%");
  });
});

describe("estimateChip", () => {
  it("is empty without an estimate", () => {
    expect(estimateChip(null)).toEqual([]);
  });

  it("gives the run alone outside a fight", () => {
    expect(estimateChip({ p_run: 0.71, p_fight: null })).toEqual([
      { label: "Run", text: "71%", tone: "good" },
    ]);
  });

  it("puts the fight beside the run and bands both", () => {
    const e = { p_run: 0.25, p_fight: 0.75, hp_loss: null, hp_now: 30 };
    expect(estimateChip(e)).toEqual([
      { label: "Run", text: "25%", tone: "bad" },
      { label: "Fight", text: "75%", tone: "mid" },
    ]);
  });
});

describe("hpRangeText", () => {
  it("is empty outside a fight", () => {
    expect(hpRangeText(null)).toBe("");
  });

  it("gives the median and the 10–90 range, or the median alone when the ends agree", () => {
    expect(hpRangeText({ q10: 2, q50: 9.4, q90: 20 })).toBe("−9 HP (−2 HP to −20 HP)");
    expect(hpRangeText({ q10: -6, q50: -6, q90: -6 })).toBe("+6 HP");
  });
});
