import { describe, expect, it } from "vitest";
import { heldIndex, hold, tapCard, tapEnemy, tapStage } from "./handSelect";

const attack = (index: number, canPlay = true) => ({ index, needsTarget: true, canPlay });
const skill = (index: number, canPlay = true) => ({ index, needsTarget: false, canPlay });
const live2 = { live: true, enemies: [0, 1] };
const live1 = { live: true, enemies: [4] };
const none = { selected: null, pending: null };

describe("tapCard", () => {
  it("first tap on an attack selects and arms it", () => {
    const r = tapCard(none, attack(2), live2);
    expect(r).toEqual({ selected: 2, pending: 2 });
  });

  it("first tap on a skill selects it without arming", () => {
    expect(tapCard(none, skill(1), live2)).toEqual({ selected: 1, pending: null });
  });

  it("second tap on a selected skill plays it", () => {
    const r = tapCard({ selected: 1, pending: null }, skill(1), live2);
    expect(r.play).toEqual({ card: 1 });
    expect(r.selected).toBeNull();
    expect(r.pending).toBeNull();
  });

  it("one enemy: the first tap still only zooms and arms", () => {
    const r = tapCard(none, attack(3), live1);
    expect(r).toEqual({ selected: 3, pending: 3 });
  });

  it("one enemy: the second tap on the card plays it on that enemy", () => {
    const r = tapCard({ selected: 3, pending: 3 }, attack(3), live1);
    expect(r).toEqual({ selected: null, pending: null, play: { card: 3, target: 4 } });
  });

  it("two enemies: the second tap on the card does not play", () => {
    const r = tapCard({ selected: 2, pending: 2 }, attack(2), live2);
    expect(r).toEqual({ selected: 2, pending: 2 });
  });

  it("tapping another card switches selection and arming", () => {
    expect(tapCard({ selected: 2, pending: 2 }, skill(0), live2)).toEqual({ selected: 0, pending: null });
    expect(tapCard({ selected: 0, pending: null }, attack(3), live2)).toEqual({ selected: 3, pending: 3 });
  });

  it("an unplayable card zooms but never arms or plays", () => {
    const first = tapCard(none, attack(2, false), live1);
    expect(first).toEqual({ selected: 2, pending: null });
    const second = tapCard(first, attack(2, false), live1);
    expect(second).toEqual({ selected: 2, pending: null });
    expect(tapCard({ selected: 1, pending: null }, skill(1, false), live2).play).toBeUndefined();
  });

  it("replay: selects, never arms or plays", () => {
    const replay = { live: false, enemies: [0] };
    const first = tapCard(none, attack(2), replay);
    expect(first).toEqual({ selected: 2, pending: null });
    expect(tapCard(first, attack(2), replay)).toEqual({ selected: 2, pending: null });
    expect(tapCard({ selected: 1, pending: null }, skill(1), replay)).toEqual({ selected: 1, pending: null });
  });
});

describe("tapEnemy", () => {
  it("plays the armed card on that enemy and clears", () => {
    expect(tapEnemy({ selected: 2, pending: 2 }, 1)).toEqual({
      selected: null, pending: null, play: { card: 2, target: 1 },
    });
  });

  it("does nothing without an armed card", () => {
    expect(tapEnemy({ selected: 1, pending: null }, 0)).toEqual({ selected: 1, pending: null });
  });
});

describe("tapStage", () => {
  it("clears everything", () => {
    expect(tapStage()).toEqual({ selected: null, pending: null });
  });
});

describe("held selection", () => {
  it("reads back only against the state it was made in", () => {
    const s1 = { hand: [0, 1, 2] };
    const s2 = { hand: [0, 1] };
    const h = hold(2, s1);
    expect(heldIndex(h, s1)).toBe(2);
    // The very first render of the next state already sees no selection.
    expect(heldIndex(h, s2)).toBeNull();
    expect(heldIndex(hold(null, s1), s1)).toBeNull();
  });
});
