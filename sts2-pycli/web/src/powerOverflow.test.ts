import { describe, expect, it } from "vitest";
import type { Power } from "./api";
import { splitPowers } from "./powerOverflow";

const mk = (n: number): Power[] =>
  Array.from({ length: n }, (_, i) => ({ name: `P${i}`, amount: i + 1 }) as Power);

describe("splitPowers", () => {
  it("hides nothing when everything fits", () => {
    const r = splitPowers(mk(3), 6);
    expect(r.shown).toHaveLength(3);
    expect(r.hidden).toHaveLength(0);
  });
  it("leaves the last cell for the +N chip", () => {
    const r = splitPowers(mk(9), 6);
    expect(r.shown).toHaveLength(5);
    expect(r.hidden).toHaveLength(4);
    expect(r.shown.map((p) => p.name)).toEqual(["P0", "P1", "P2", "P3", "P4"]);
  });
  it("treats maxShown below 4 as 4", () => {
    const r = splitPowers(mk(9), 2);
    expect(r.shown).toHaveLength(3);
    expect(r.hidden).toHaveLength(6);
  });
  it("shows exactly maxShown with no chip when the count equals it", () => {
    expect(splitPowers(mk(6), 6).hidden).toHaveLength(0);
  });
  it("a column may go down to one power and the +N chip", () => {
    const r = splitPowers(mk(5), 2, 2);
    expect(r.shown).toHaveLength(1);
    expect(r.hidden).toHaveLength(4);
  });
  it("a column of three shows two powers and the chip", () => {
    const r = splitPowers(mk(5), 3, 2);
    expect(r.shown).toHaveLength(2);
    expect(r.hidden).toHaveLength(3);
  });
});
