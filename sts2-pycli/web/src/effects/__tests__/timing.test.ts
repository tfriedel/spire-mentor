import { describe, it, expect } from "vitest";
import { scheduleTimes, EVENT_ADVANCE_MS, HIT_SPACING_MS } from "../timing";
import type { EffectEvent } from "../types";

const E = { side: "enemy", index: 0 } as const;

describe("scheduleTimes", () => {
  it("fires a single event immediately", () => {
    expect(scheduleTimes([{ kind: "hurt", target: E, amount: 6 }])).toEqual([0]);
  });

  it("spaces successive hits so their numbers do not overlap", () => {
    const hits: EffectEvent[] = [
      { kind: "hurt", target: E, amount: 6 },
      { kind: "hurt", target: E, amount: 6 },
      { kind: "hurt", target: E, amount: 6 },
    ];
    expect(scheduleTimes(hits)).toEqual([0, HIT_SPACING_MS, HIT_SPACING_MS * 2]);
  });

  it("fires a block gain with no delay after it", () => {
    const evs: EffectEvent[] = [
      { kind: "blockGain", target: { side: "player" }, amount: 5 },
      { kind: "blockGain", target: { side: "player" }, amount: 5 },
    ];
    expect(scheduleTimes(evs)).toEqual([0, 0]);
  });

  it("puts a death after the hit that caused it", () => {
    const evs: EffectEvent[] = [
      { kind: "hurt", target: E, amount: 6 },
      { kind: "death", target: E },
    ];
    const [hit, death] = scheduleTimes(evs);
    expect(death).toBeGreaterThan(hit);
  });

  it("returns [] for no events", () => {
    expect(scheduleTimes([])).toEqual([]);
  });

  it("is parallel to its input", () => {
    const evs: EffectEvent[] = [
      { kind: "attack", target: E },
      { kind: "hurt", target: E, amount: 3 },
      { kind: "powerAdd", target: E, power: "Weak" },
    ];
    expect(scheduleTimes(evs)).toHaveLength(evs.length);
  });

  it("schedules a flash without stalling the row it announces", () => {
    // The one thing about `powerFlash`'s advance that is NOT provisional: a
    // flash announces the row after it, so however the value is later tuned,
    // the flash and that row must not drift into two separate beats.
    const evs: EffectEvent[] = [
      { kind: "powerFlash", target: E, power: "Thorns" },
      { kind: "hurt", target: E, amount: 3 },
    ];
    const [flash, hit] = scheduleTimes(evs);
    expect(hit! - flash!).toBeLessThan(HIT_SPACING_MS);
  });
});

describe("EVENT_ADVANCE_MS", () => {
  // Every kind the union can carry, written out. The annotation makes a typo
  // or a removed kind a COMPILE error; the equality below makes a kind that
  // exists in both the union and the record but is missing here a RED test.
  // Between them the record cannot go partial and this list cannot go stale —
  // which is what `scheduleTimes` needs, since a missing entry would advance
  // its clock by `undefined` and NaN every later offset in silence.
  const ALL_KINDS: EffectEvent["kind"][] = [
    "attack", "hurt", "fullyBlocked", "death", "blockGain", "blockBreak",
    "powerAdd", "powerRemove", "powerFlash", "heal", "cardPlay",
  ];

  it("has an entry for every event kind", () => {
    expect(Object.keys(EVENT_ADVANCE_MS).sort()).toEqual([...ALL_KINDS].sort());
  });

  it("advances by a finite, non-negative number for every kind", () => {
    for (const k of ALL_KINDS) {
      expect(Number.isFinite(EVENT_ADVANCE_MS[k]), k).toBe(true);
      expect(EVENT_ADVANCE_MS[k], k).toBeGreaterThanOrEqual(0);
    }
  });
});
