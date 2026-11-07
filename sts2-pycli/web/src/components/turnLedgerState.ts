// The turn ledger's persisted detail mode and its per-turn chips — kept out of
// the component file so it exports only components (react-refresh/only-export-
// components).
import { useState } from "react";
import type { CombatLogRow } from "../api";

const DETAIL_KEY = "wb.history.detail";

export type DetailMode = "net" | "full";

/** Persisted detail mode. localStorage is absent/throwing in some contexts. */
export function useDetailMode(): [DetailMode, (m: DetailMode) => void] {
  const [mode, setMode] = useState<DetailMode>(() => {
    try {
      return globalThis.localStorage?.getItem(DETAIL_KEY) === "full" ? "full" : "net";
    } catch {
      return "net";
    }
  });
  const set = (m: DetailMode) => {
    setMode(m);
    try {
      globalThis.localStorage?.setItem(DETAIL_KEY, m);
    } catch {
      // Preference not persisted; the toggle still works this session.
    }
  };
  return [mode, set];
}

/**
 * Chips summarizing one turn: total damage the player dealt, block gained,
 * HP the player lost, and kills. Kills only exist in the diff-derived detail
 * lines (the engine log has no death event), so those ride in via `details`.
 */
export function turnChips(rows: CombatLogRow[], details: string[]): string[] {
  let dealt = 0, taken = 0, block = 0;
  for (const r of rows) {
    if (r.kind === "damage" && typeof r.amount === "number") {
      if (r.actor_side === "player") dealt += r.amount;
      if (r.target_side === "player") taken += r.amount;
    }
    if (r.kind === "block" && r.actor_side === "player" && typeof r.amount === "number") {
      block += r.amount;
    }
  }
  const kills = details.filter((d) => d.includes("☠")).length;
  const out: string[] = [];
  if (dealt) out.push(`⚔ ${dealt}`);
  if (block) out.push(`🛡 ${block}`);
  if (taken) out.push(`−${taken} HP`);
  if (kills) out.push(`☠ ${kills}`);
  return out;
}

/** Player-side row kinds the `net` view keeps: each names a card the step
 * label does not (GH #1015, #1155). The engine log hides the card moves
 * unless verbose, so in the workbench these arrive only from the mod. */
const NET_CARD_KINDS = new Set([
  "card_exhausted",
  "card_discarded",
  "card_generated",
  "card_auto_played",
  "card_replayed",
]);

export function keptInNet(r: CombatLogRow): boolean {
  return r.side === "enemy" || r.auto_played === true || NET_CARD_KINDS.has(r.kind);
}
