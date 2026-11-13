import { describe, expect, it } from "vitest";
import { INTENT_TYPES, intentInfo, intentName } from "./intentNames";

describe("intent names", () => {
  // MegaCrit.Sts2.Core.MonsterMoves.Intents.IntentType (v0.107.1), in order.
  const ENUM = ["Attack", "Buff", "Debuff", "DebuffStrong", "Defend", "Escape", "Heal", "Hidden",
    "Summon", "Sleep", "Stun", "StatusCard", "CardDebuff", "DeathBlow", "Unknown"];

  it("covers every IntentType", () => {
    expect([...INTENT_TYPES].sort()).toEqual([...ENUM].sort());
  });

  it("gives every known type a human name and a one-sentence description", () => {
    for (const t of ENUM) {
      const { name, description } = intentInfo({ type: t });
      expect(name, t).not.toMatch(/[a-z][A-Z]/); // never a raw CamelCase id
      expect(name.length, t).toBeGreaterThan(0);
      expect(description, t).toMatch(/^[A-Z].*\.$/);
      expect(description, t).not.toMatch(/[[\]{}]/); // no BBCode, no placeholders
    }
  });

  it("uses the game's own hover tip text", () => {
    expect(intentInfo({ type: "DebuffStrong" })).toEqual({
      name: "Strategic", description: "This enemy intends to apply a powerful Debuff to you." });
    expect(intentName("Defend")).toBe("Defensive");
  });

  it("fills in an attack's damage and hits", () => {
    expect(intentInfo({ type: "Attack", damage: 12 }).description)
      .toBe("This enemy intends to Attack for 12 damage.");
    expect(intentInfo({ type: "Attack", damage: 5, hits: 3 }).description)
      .toBe("This enemy intends to Attack for 5 damage 3 times.");
    expect(intentInfo({ type: "Attack" }).description).toBe("This enemy intends to Attack.");
    expect(intentInfo({ type: "DeathBlow", damage: 30 }).description).toContain("for 30 damage");
  });

  it("spells out an unknown type's id rather than showing it raw", () => {
    expect(intentName("DebuffMega")).toBe("Debuff Mega");
    expect(intentInfo({ type: "NewThingX" })).toEqual({ name: "New Thing X", description: undefined });
  });
});
