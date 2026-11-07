import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, it, expect } from "vitest";
import EventLog from "./EventLog";
import type { TimelineEntry } from "../api";

function entry(over: Partial<TimelineEntry>): TimelineEntry {
  return {
    idx: 0, decision: "combat_play", label: "Play Strike", details: [],
    turn: 1, floor: 1, act: 1, act_name: "Act 1", ...over,
  };
}

function damageRow(over: Partial<import("../api").CombatLogRow> = {}): import("../api").CombatLogRow {
  return {
    kind: "damage", round: 1, side: "player",
    text: "The Ironclad dealt 6 damage to Nibbit using Strike.",
    verb: "dealt", actor: "The Ironclad", actor_side: "player",
    target: "Nibbit", target_side: "enemy", amount: 6, noun: null, source: "Strike",
    ...over,
  };
}

describe("EventLog", () => {
  // The runner's localStorage stub lacks removeItem; the components guard
  // every access the same way.
  beforeEach(() => {
    try {
      globalThis.localStorage?.removeItem?.("wb.history.detail");
    } catch {
      // no storage in this environment — the default mode applies anyway
    }
  });

  it("renders a (coloured) detail line under its action", () => {
    const entries = [entry({ idx: 0, label: "Play Strike", details: ["Cultist HP 50→44 (−6)"] })];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    expect(screen.getByText("Play Strike")).toBeInTheDocument();
    const detail = container.querySelector("[data-detail]");
    expect(detail?.textContent).toBe("Cultist HP 50→44 (−6)");
  });

  it("does not scrub when the click ends a text selection in the row", () => {
    const picks: number[] = [];
    const entries = [entry({ idx: 0, label: "Play Strike" }), entry({ idx: 1, label: "Play Bash" })];
    render(<EventLog entries={entries} current={0} onPick={(i) => picks.push(i)} />);
    const label = screen.getByText("Play Bash");
    getSelection()!.selectAllChildren(label);
    fireEvent.click(label);
    expect(picks).toEqual([]);
    getSelection()!.removeAllRanges();
    fireEvent.click(label);
    expect(picks).toEqual([1]);
  });

  it("renders no detail lines when details is empty", () => {
    const entries = [entry({ idx: 0, label: "End turn", details: [] })];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    expect(container.querySelectorAll("[data-detail]").length).toBe(0);
  });

  it("groups combat entries under a turn header with net chips", () => {
    const entries = [
      entry({ idx: 0, label: "Play Strike", turn: 1, combat_log: [damageRow()] }),
      entry({ idx: 1, label: "End turn", turn: 1, combat_log: [
        damageRow({ side: "enemy", actor: "Nibbit", actor_side: "enemy",
          target: "The Ironclad", target_side: "player", amount: 7, source: null,
          text: "Nibbit dealt 7 damage to The Ironclad." }),
      ] }),
      entry({ idx: 2, label: "Play Bash", turn: 2, combat_log: [damageRow({ amount: 8, round: 2 })] }),
    ];
    render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    expect(screen.getByLabelText("Turn 1")).toHaveTextContent("⚔ 6");
    expect(screen.getByLabelText("Turn 1")).toHaveTextContent("−7 HP");
    expect(screen.getByLabelText("Turn 2")).toHaveTextContent("⚔ 8");
  });

  it("never shows a view-only row, even in full mode", () => {
    // The recorded slice carries the view-only `power_flash` rows, because the
    // board's effect stream needs them — so the ledger drops them itself.
    const flash = damageRow({
      kind: "power_flash", side: "enemy", actor: null, actor_side: null,
      target: "Nibbit", target_side: "enemy", amount: null, noun: "Thorns",
      source: null, verb: "triggered", text: "Nibbit's Thorns triggered.",
    });
    const entries = [
      entry({ idx: 0, label: "Play Strike", turn: 1, combat_log: [damageRow(), flash] }),
    ];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    fireEvent.click(screen.getByText("full"));
    expect(screen.queryByText("Thorns")).not.toBeInTheDocument();
    expect(container.querySelectorAll("[data-detail]").length).toBe(1);
  });

  it("falls back to diff details for a step whose only rows are view-only", () => {
    // `StepEvents` shows the tape INSTEAD of the diff lines when a step has
    // one. A slice of nothing but hidden rows is not a tape.
    const flash = damageRow({
      kind: "power_flash", amount: null, noun: "Thorns", source: null,
      verb: "triggered", text: "Nibbit's Thorns triggered.",
    });
    const entries = [
      entry({ idx: 0, label: "Play Strike", details: ["Nibbit HP 50→44 (−6)"],
        combat_log: [flash] }),
    ];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    const details = [...container.querySelectorAll("[data-detail]")].map((d) => d.textContent);
    expect(details).toEqual(["Nibbit HP 50→44 (−6)"]);
    // …and the net/full toggle, which only exists when some step has a tape,
    // stays away too.
    expect(screen.queryByText("full")).not.toBeInTheDocument();
  });

  it("net mode shows only the enemy turn's rows; full shows every row", () => {
    const entries = [
      entry({ idx: 0, label: "Play Strike", turn: 1, combat_log: [
        damageRow(),
        damageRow({ side: "enemy", actor: "Nibbit", actor_side: "enemy",
          target: "The Ironclad", target_side: "player", amount: 7, source: null,
          text: "Nibbit dealt 7 damage to The Ironclad." }),
      ] }),
    ];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    // Default (net): only the enemy-turn row.
    expect(container.querySelectorAll("[data-detail]").length).toBe(1);
    expect(screen.getByText("Nibbit")).toBeInTheDocument();
    // Full: the player's row appears too.
    fireEvent.click(screen.getByText("full"));
    expect(container.querySelectorAll("[data-detail]").length).toBe(2);
    expect(screen.getByText("Strike")).toBeInTheDocument();
  });

  it("net mode keeps an auto-played card's rows (GH #1155)", () => {
    // Stampede plays Bash at end of turn. The step is labelled "End turn", so
    // dropping the player-side rows here hid the play entirely.
    const entries = [
      entry({ idx: 0, label: "End turn", turn: 1, combat_log: [
        damageRow({ kind: "card_auto_played", verb: "auto-played", noun: "Bash",
          target: null, target_side: null, amount: null, source: null,
          text: "The Ironclad auto-played Bash.", auto_played: true,
          card_id: "BASH", card_upgraded: false }),
        damageRow({ amount: 8, source: "Bash", auto_played: true,
          text: "The Ironclad dealt 8 damage to Nibbit using Bash." }),
        damageRow({ side: "enemy", actor: "Nibbit", actor_side: "enemy",
          target: "The Ironclad", target_side: "player", amount: 5, source: null,
          text: "Nibbit dealt 5 damage to The Ironclad." }),
      ] }),
    ];
    const { container } = render(<EventLog entries={entries} current={0} onPick={() => {}} />);
    const lines = [...container.querySelectorAll("[data-detail]")].map((d) => d.textContent);
    expect(lines).toEqual([
      "The Ironclad auto-played Bash.",
      "The Ironclad dealt 8 damage to Nibbit using Bash.",
      "Nibbit dealt 5 damage to The Ironclad.",
    ]);
  });
});
