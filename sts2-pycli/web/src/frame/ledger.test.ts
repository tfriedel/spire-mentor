import { describe, expect, it } from "vitest";
import { decisionLabel, toTimeline } from "./ledger";
import type { LogEntry, LogRow } from "./store";

const at = (actor: string, action: string, extra: Partial<LogEntry> = {}): LogEntry => ({
  ts: "2026-11-06T20:00:00Z",
  actor,
  act: 1,
  floor: 3,
  decision: "",
  action,
  advice: null,
  ...extra,
});

const row = (r: Partial<LogRow>): LogRow => ({
  kind: "other",
  verb: null,
  actor: null,
  actor_side: null,
  target: null,
  target_side: null,
  amount: null,
  noun: null,
  text: "",
  killed: false,
  ...r,
});

describe("toTimeline", () => {
  it("files a fight the way the workbench timeline does", () => {
    const tl = toTimeline([
      at("game", "════ floor 3 · Monster  HP 70/80  gold 99"),
      at("mod", "Hints: the AI's opinion shows while you decide."),
      at("game", "— turn 1 —  HP 70/80"),
      at("You", "Strike → Jaw Worm", { decision: "combat", advice: "AI: Bash → Jaw Worm" }),
      at("game", "The Ironclad dealt 6 damage to Jaw Worm.", {
        row: row({ kind: "damage", verb: "dealt", actor: "The Ironclad", actor_side: "player", target: "Jaw Worm", target_side: "enemy", amount: 6 }),
      }),
      at("You", "Defend", { decision: "combat", advice: "AI: Defend" }),
      at("game", "Jaw Worm used Chomp.", { row: row({ kind: "monster_move", actor: "Jaw Worm", actor_side: "enemy", noun: "Chomp" }) }),
      at("game", "Jaw Worm dealt 6 damage to The Ironclad.", {
        row: row({ kind: "damage", verb: "dealt", actor: "Jaw Worm", actor_side: "enemy", target: "The Ironclad", target_side: "player", amount: 6 }),
      }),
      at("game", "— turn 2 —  HP 64/80"),
      at("AI", "Bash → Jaw Worm", { decision: "combat" }),
      at("game", "The Ironclad dealt 34 damage to Jaw Worm.", {
        row: row({ kind: "damage", actor: "The Ironclad", actor_side: "player", target: "Jaw Worm", target_side: "enemy", amount: 34, killed: true }),
      }),
      at("game", "— combat ended —  HP 64/80"),
      at("AI", "Inflame", { decision: "card_reward", floor: 3 }),
    ]);
    expect(tl.map((t) => t.label)).toEqual([
      "Strike → Jaw Worm  ·  AI: Bash → Jaw Worm",
      "Defend",
      "End turn",
      "🤖 Bash → Jaw Worm",
      "🤖 Card reward: Inflame",
    ]);
    expect(tl.map((t) => t.turn)).toEqual([1, 1, 1, 2, null]);
    expect(tl[0].combat_log?.[0]).toMatchObject({ kind: "damage", round: 1, side: "player", amount: 6 });
    // the enemy's turn: its rows are "enemy" side (the workbench's net view keeps them)
    expect(tl[2].combat_log?.map((r) => r.side)).toEqual(["enemy", "enemy"]);
    expect(tl[3].details).toEqual(["Jaw Worm ☠ defeated"]);
    expect(tl.map((t) => t.idx)).toEqual([0, 1, 2, 3, 4]);
  });

  it("keeps an enemy's reaction on your turn under your play", () => {
    const tl = toTimeline([
      at("game", "— turn 1 —"),
      at("You", "Strike → Louse", { decision: "combat" }),
      // Curl Up: an enemy actor, but on the player's turn
      at("game", "Louse gained 7 Block.", {
        row: row({ kind: "block", actor: "Louse", actor_side: "enemy", amount: 7, side: "player" }),
      }),
      at("You", "Defend", { decision: "combat" }),
      at("game", "Louse used Bite.", {
        row: row({ kind: "monster_move", actor: "Louse", actor_side: "enemy", side: "enemy" }),
      }),
    ]);
    expect(tl.map((t) => t.label)).toEqual(["Strike → Louse", "Defend", "End turn"]);
    expect(tl[0].combat_log?.map((r) => r.side)).toEqual(["player"]);
    expect(tl[2].combat_log?.map((r) => r.side)).toEqual(["enemy"]);
  });

  it("opens the enemy's turn on a monster move when an older mod sends no side", () => {
    const tl = toTimeline([
      at("game", "— turn 1 —"),
      at("You", "Strike → Louse", { decision: "combat" }),
      at("game", "Louse gained 7 Block.", { row: row({ kind: "block", actor: "Louse", actor_side: "enemy" }) }),
      at("game", "Louse used Bite.", { row: row({ kind: "monster_move", actor: "Louse", actor_side: "enemy" }) }),
    ]);
    expect(tl.map((t) => t.label)).toEqual(["Strike → Louse", "End turn"]);
  });

  it("keeps an older mod's plain lines as details", () => {
    const tl = toTimeline([at("game", "— turn 1 —"), at("game", "IRONCLAD gained 5 block")]);
    expect(tl).toHaveLength(1);
    expect(tl[0]).toMatchObject({ label: "Start of turn", details: ["IRONCLAD gained 5 block"] });
  });

  it("names decisions", () => {
    expect(decisionLabel("combat")).toBe("");
    expect(decisionLabel("rest_site")).toBe("Rest site");
  });
});
