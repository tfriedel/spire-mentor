import { describe, it, expect } from "vitest";
import { actionStream } from "../actionStream";
import type { CombatLogRow, CombatState } from "../../api";

function state(over: Partial<CombatState> = {}): CombatState {
  return {
    decision: "combat_play", round: 1, energy: 3, max_energy: 3,
    draw_pile_count: 0, discard_pile_count: 0, exhaust_pile_count: 0,
    player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
    player_powers: [], enemies: [], hand: [],
    ...over,
  } as unknown as CombatState;
}

function enemy(over: Record<string, unknown> = {}) {
  return { index: 0, uid: 11, name: "Nibbit", hp: 20, max_hp: 20, block: 0, image: null, ...over };
}

function row(over: Partial<CombatLogRow> = {}): CombatLogRow {
  return {
    kind: "damage", round: 1, side: "player", text: "", verb: "dealt",
    actor: "The Ironclad", actor_side: "player", actor_uid: null,
    target: "Nibbit", target_side: "enemy", target_uid: 11,
    amount: 6, blocked: 0, hp_after: 14, noun: null, source: "Strike",
    ...over,
  } as CombatLogRow;
}

/** A `power_flash` row: a VIEW signal, so no amount, no source, no hp. */
function flashRow(over: Partial<CombatLogRow> = {}): CombatLogRow {
  return row({
    kind: "power_flash", verb: "triggered", noun: "Thorns",
    actor: null, actor_side: null, actor_uid: null,
    target: "Nibbit", target_side: "enemy", target_uid: 11,
    amount: null, blocked: null, hp_after: null, source: null,
    ...over,
  });
}

describe("actionStream — per-hit fidelity", () => {
  it("emits one hurt per damage row, not one per net HP change", () => {
    const prev = state({ enemies: [enemy({ hp: 20 })] });
    const next = state({ enemies: [enemy({ hp: 2 })] });
    const rows = [
      row({ amount: 6, hp_after: 14 }),
      row({ amount: 6, hp_after: 8 }),
      row({ amount: 6, hp_after: 2 }),
    ];
    const hurts = actionStream(rows, prev, next).filter((e) => e.kind === "hurt");
    expect(hurts).toEqual([
      { kind: "hurt", target: { side: "enemy", index: 0 }, amount: 6 },
      { kind: "hurt", target: { side: "enemy", index: 0 }, amount: 6 },
      { kind: "hurt", target: { side: "enemy", index: 0 }, amount: 6 },
    ]);
  });

  it("emits fullyBlocked, not hurt, when block ate the whole hit", () => {
    const prev = state({ enemies: [enemy({ hp: 20, block: 6 })] });
    const next = state({ enemies: [enemy({ hp: 20, block: 0 })] });
    const out = actionStream([row({ amount: 0, blocked: 6, hp_after: 20 })], prev, next);
    expect(out.some((e) => e.kind === "hurt")).toBe(false);
    expect(out).toContainEqual({ kind: "fullyBlocked", target: { side: "enemy", index: 0 } });
  });

  it("emits a death for the killing blow, resolved from the previous state", () => {
    const prev = state({ enemies: [enemy({ hp: 6 })] });
    const next = state({ enemies: [] });
    const out = actionStream([row({ amount: 6, hp_after: 0 })], prev, next);
    expect(out).toContainEqual({ kind: "death", target: { side: "enemy", index: 0 } });
  });

  it("emits blockGain from a block row", () => {
    const prev = state();
    const next = state({ player: { name: "Ironclad", hp: 80, max_hp: 80, block: 5 } });
    const out = actionStream(
      [row({ kind: "block", target_uid: null, target_side: null, actor_side: "player", amount: 5 })],
      prev, next,
    );
    expect(out).toContainEqual({ kind: "blockGain", target: { side: "player" }, amount: 5 });
  });

  it("emits powerAdd from a power row", () => {
    const prev = state({ enemies: [enemy()] });
    const next = state({ enemies: [enemy({ powers: [{ name: "Weak", amount: 2 }] })] });
    const out = actionStream(
      [row({ kind: "power", amount: 2, noun: "Weak" })],
      prev, next,
    );
    expect(out).toContainEqual({
      kind: "powerAdd", target: { side: "enemy", index: 0 }, power: "Weak", amount: 2,
    });
  });

  it("takes powerAdd's amount from the snapshot TOTAL, not the row's delta", () => {
    // The row says "+2 stacks were applied"; the creature ends the action on 5.
    // `GetTypeForAmount` classifies by the total, so 5 is the number that must
    // ride the event — see `EffectEvent.powerAdd`'s doc.
    const prev = state({ enemies: [enemy({ powers: [{ name: "Strength", amount: 3 }] })] });
    const next = state({ enemies: [enemy({ powers: [{ name: "Strength", amount: 5 }] })] });
    const out = actionStream([row({ kind: "power", amount: 2, noun: "Strength" })], prev, next);
    expect(out).toContainEqual({
      kind: "powerAdd", target: { side: "enemy", index: 0 }, power: "Strength", amount: 5,
    });
  });

  it("omits powerAdd's amount rather than guessing when the snapshot has no total", () => {
    // A power the log named but that is not in `next` (removed again within the
    // same action, say). Absent beats the delta: `powerTypeForAmount` reads an
    // absent amount as "classify by name", which is the pre-amount behaviour.
    const prev = state({ enemies: [enemy()] });
    const next = state({ enemies: [enemy()] });
    const out = actionStream([row({ kind: "power", amount: 2, noun: "Weak" })], prev, next);
    expect(out).toContainEqual({
      kind: "powerAdd", target: { side: "enemy", index: 0 }, power: "Weak", amount: undefined,
    });
  });

  it("turns a power_flash row into a powerFlash event on the right slot", () => {
    const prev = state({ enemies: [enemy()] });
    const next = state({ enemies: [enemy()] });
    expect(actionStream([flashRow()], prev, next)).toContainEqual({
      kind: "powerFlash",
      target: { side: "enemy", index: 0 },
      power: "Thorns",
    });
  });

  it("does not confuse a flash with an application", () => {
    // Both rows name the same power on the same creature, and only the
    // `power` row means "you gained stacks". Feeding BOTH is what makes this
    // able to fail in both directions: mapping `power_flash` to `powerAdd`
    // yields two adds, and dropping the flash branch yields no flash.
    const prev = state({ enemies: [enemy()] });
    const next = state({ enemies: [enemy({ powers: [{ name: "Thorns", amount: 3 }] })] });
    const out = actionStream(
      [row({ kind: "power", amount: 3, noun: "Thorns" }), flashRow()],
      prev, next,
    );
    expect(out.filter((e) => e.kind === "powerAdd")).toEqual([
      { kind: "powerAdd", target: { side: "enemy", index: 0 }, power: "Thorns", amount: 3 },
    ]);
    expect(out.filter((e) => e.kind === "powerFlash")).toEqual([
      { kind: "powerFlash", target: { side: "enemy", index: 0 }, power: "Thorns" },
    ]);
  });

  it("emits an attack for the actor of a monster_move row", () => {
    const prev = state({ enemies: [enemy()] });
    const next = state({ enemies: [enemy()] });
    const out = actionStream(
      [row({ kind: "monster_move", actor_side: "enemy", actor_uid: 11, target_uid: null, target_side: null })],
      prev, next,
    );
    expect(out).toContainEqual({ kind: "attack", target: { side: "enemy", index: 0 } });
  });
});

describe("actionStream — source", () => {
  it("carries the card a hit came from", () => {
    const prev = state({ enemies: [enemy({ hp: 20 })] });
    const next = state({ enemies: [enemy({ hp: 13 })] });
    const rows = [
      row({ amount: 7, hp_after: 13, source_kind: "card", source_card: "BASH" }),
    ];
    const [e] = actionStream(rows, prev, next);
    expect(e.kind).toBe("hurt");
    expect((e as { source?: unknown }).source).toEqual({ kind: "card", id: "BASH" });
  });

  it("carries the monster and move a monster hit came from", () => {
    const prev = state({
      enemies: [enemy()],
      player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
    });
    const next = state({
      enemies: [enemy()],
      player: { name: "Ironclad", hp: 73, max_hp: 80, block: 0 },
    });
    const rows = [
      row({
        actor_side: "enemy", actor_uid: 11, target_uid: null, target_side: "player",
        amount: 7, hp_after: 73, source_kind: "move", source_move_id: "THRASH",
        source_monster: "Crusher",
      }),
    ];
    const [e] = actionStream(rows, prev, next);
    expect(e.kind).toBe("hurt");
    expect((e as { source?: unknown }).source).toEqual({
      kind: "move", monster: "Crusher", id: "THRASH",
    });
  });

  it("leaves source undefined for diff-derived events", () => {
    // The snapshot diff cannot know a source; it must not invent one.
    const prev = state({ player: { name: "Ironclad", hp: 60, max_hp: 80, block: 0 } });
    const next = state({ player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 } });
    const [e] = actionStream([], prev, next);
    expect(e.kind).toBe("heal");
    expect((e as { source?: unknown }).source).toBeUndefined();
  });
});

describe("actionStream — reconciliation", () => {
  it("appends heals, which the engine log does not record", () => {
    const prev = state({ player: { name: "Ironclad", hp: 60, max_hp: 80, block: 0 } });
    const next = state({ player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 } });
    const out = actionStream([], prev, next);
    expect(out).toContainEqual({ kind: "heal", target: { side: "player" }, amount: 10 });
  });

  it("appends power removal, which the engine log does not record", () => {
    const prev = state({ enemies: [enemy({ powers: [{ name: "Weak", amount: 1 }] })] });
    const next = state({ enemies: [enemy({ powers: [] })] });
    const out = actionStream([], prev, next);
    expect(out).toContainEqual({
      kind: "powerRemove", target: { side: "enemy", index: 0 }, power: "Weak", amount: 1,
    });
  });

  it("does NOT double-count a target the log already described", () => {
    const prev = state({ enemies: [enemy({ hp: 20 })] });
    const next = state({ enemies: [enemy({ hp: 14 })] });
    const out = actionStream([row({ amount: 6, hp_after: 14 })], prev, next);
    expect(out.filter((e) => e.kind === "hurt")).toHaveLength(1);
  });

  it("covers a target the log could not address (Osty) from the diff", () => {
    const prev = state({ osty: { hp: 30, max_hp: 30, block: 0, alive: true } as never });
    const next = state({ osty: { hp: 22, max_hp: 30, block: 0, alive: true } as never });
    const out = actionStream([], prev, next);
    expect(out).toContainEqual({ kind: "hurt", target: { side: "osty" }, amount: 8 });
  });

  it("leaves a flashed target undescribed, so the diff still reconciles it", () => {
    // A flash says nothing about damage or stacks. If it marked the creature
    // described, an HP change only the diff can see would be silently dropped.
    const prev = state({ enemies: [enemy({ hp: 20 })] });
    const next = state({ enemies: [enemy({ hp: 14 })] });
    const out = actionStream([flashRow()], prev, next);
    expect(out).toContainEqual({ kind: "hurt", target: { side: "enemy", index: 0 }, amount: 6 });
  });

  it("in one call: a log-described enemy hurt exactly once, an undescribed Osty hurt covered by the diff", () => {
    const prev = state({
      enemies: [enemy({ hp: 20 })],
      osty: { hp: 30, max_hp: 30, block: 0, alive: true } as never,
    });
    const next = state({
      enemies: [enemy({ hp: 14 })],
      osty: { hp: 22, max_hp: 30, block: 0, alive: true } as never,
    });
    const out = actionStream([row({ amount: 6, hp_after: 14 })], prev, next);
    expect(out.filter((e) => e.kind === "hurt" && e.target.side === "enemy")).toHaveLength(1);
    expect(out).toContainEqual({ kind: "hurt", target: { side: "osty" }, amount: 8 });
  });
});

describe("actionStream — auto-played cards (GH #1155)", () => {
  function autoRow(over: Partial<CombatLogRow> = {}): CombatLogRow {
    return row({
      kind: "card_auto_played", verb: "auto-played", noun: "Bash",
      target: null, target_side: null, target_uid: null,
      amount: null, blocked: null, hp_after: null, source: null,
      auto_played: true, card_id: "BASH", card_upgraded: false,
      ...over,
    });
  }

  it("shows the auto-played card BEFORE the hit it causes, using the hand card", () => {
    const bash = { index: 0, id: "BASH", name: "Bash", cost: 2, type: "Attack",
      description: "Deal 8 damage.", upgraded: false, image: "bash.png" };
    const prev = state({ hand: [bash], enemies: [enemy({ hp: 20 })] } as Partial<CombatState>);
    const next = state({ enemies: [enemy({ hp: 12 })] });
    const out = actionStream(
      [autoRow(), row({ amount: 8, hp_after: 12, source: "Bash", auto_played: true })],
      prev, next,
    );
    expect(out.map((e) => e.kind)).toEqual(["cardPlay", "hurt"]);
    expect(out[0]).toEqual({ kind: "cardPlay", target: { side: "player" }, card: bash });
  });

  it("falls back to a stub card named by the row when no pile holds it", () => {
    const out = actionStream([autoRow({ noun: "Strike+", card_id: "STRIKE_IRONCLAD", card_upgraded: true })],
      state(), state());
    expect(out).toHaveLength(1);
    const e = out[0];
    expect(e.kind).toBe("cardPlay");
    if (e.kind !== "cardPlay") return;
    expect(e.card).toMatchObject({ id: "STRIKE_IRONCLAD", name: "Strike+", upgraded: true, image: null });
  });
});
