import { describe, expect, it } from "vitest";
import { COLUMN_MIN, COLUMN_WANT, bestColumnPlace, columnBottom, columnCells, columnRoom } from "./powerColumn";

// 44px cells, 5px gaps: n cells are 49n - 5 tall.
const base = { bottom: 300, cell: 44, gap: 5, ceiling: 100, left: 100, right: 144, obstacles: [] };

describe("columnCells", () => {
  it("is sized for three powers and the +N chip before it measures", () => {
    expect(COLUMN_WANT).toBe(4);
  });
  it("takes the room up to the ceiling, however short the creature", () => {
    // 200 px under the ceiling: four cells (191 px) = three powers and the chip
    expect(columnCells(base)).toBe(4);
    expect(columnCells({ ...base, ceiling: 0 })).toBe(6);
  });
  it("never rises above the ceiling (the stage top)", () => {
    // only 120 px under the ceiling: two cells (93 px)
    expect(columnCells({ ...base, ceiling: 180 })).toBe(2);
  });
  it("stops under another creature's intent that hangs over the column", () => {
    const intent = { left: 120, right: 164, top: 150, bottom: 194 };
    // room from 199 to 300: two cells
    expect(columnCells({ ...base, ceiling: 0, obstacles: [intent] })).toBe(2);
  });
  it("ignores obstacles beside the column", () => {
    const intent = { left: 150, right: 194, top: 150, bottom: 260 };
    expect(columnCells({ ...base, ceiling: 0, obstacles: [intent] })).toBe(6);
  });
  it("never goes under one power and the +N chip", () => {
    expect(COLUMN_MIN).toBe(2);
    expect(columnCells({ ...base, ceiling: 290 })).toBe(2);
    const low = { left: 100, right: 144, top: 200, bottom: 280 };
    expect(columnCells({ ...base, obstacles: [low] })).toBe(2);
  });
  it("columnRoom reports a column that fits nowhere as 0 or 1, unclamped", () => {
    expect(columnRoom({ ...base, ceiling: 290 })).toBe(0);
    const low = { left: 100, right: 144, top: 200, bottom: 320 };
    expect(columnRoom({ ...base, obstacles: [low] })).toBe(0);
    expect(columnCells({ ...base, obstacles: [low] })).toBe(COLUMN_MIN);
    const intent = { left: 120, right: 164, top: 200, bottom: 250 };
    expect(columnRoom({ ...base, ceiling: 0, obstacles: [intent] })).toBe(1);
  });
});

describe("bestColumnPlace", () => {
  it("takes the place with the most room, the first on a tie", () => {
    expect(bestColumnPlace([1, 3, 3])).toBe(1);
    expect(bestColumnPlace([3, 3])).toBe(0);
    expect(bestColumnPlace([0])).toBe(0);
  });
});

describe("columnBottom", () => {
  it("stays on the ground line with nothing across it", () => {
    expect(columnBottom(300, 100, 144, 5, [])).toBe(300);
    // a neighbour's HP bar beside the column, or under the ground line
    expect(columnBottom(300, 100, 144, 5, [{ left: 150, right: 220, top: 250, bottom: 255 }])).toBe(300);
    expect(columnBottom(300, 100, 144, 5, [{ left: 90, right: 220, top: 305, bottom: 310 }])).toBe(300);
  });
  it("stands on a raised neighbour's HP bar or name that reaches across it", () => {
    const hp = { left: 80, right: 120, top: 250, bottom: 255 };
    const name = { left: 80, right: 125, top: 259, bottom: 275 };
    expect(columnBottom(300, 100, 144, 5, [name, hp])).toBe(245);
  });
});
