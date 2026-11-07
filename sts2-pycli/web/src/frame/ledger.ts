// The mod's flat action log (FrameLink `log` entries) as the workbench's
// timeline, so the frame can render it with the workbench's own EventLog:
// "Act · floor" headers, TURN headers with chips, one row per decision, and
// each decision's combat-log rows (coloured by LogLine) under it -- the
// enemy's turn under a synthetic "End turn" row, as the workbench has it.

import type { CombatLogRow, TimelineEntry } from "../api";
import type { LogEntry } from "./store";

const FLOOR = /^════ floor (\d+)/;
const TURN = /^— turn (\d+) —/;
const ENDED = /^— combat ended —/;

/** "card_reward" -> "Card reward"; combat steps need no prefix. */
export function decisionLabel(decision: string): string {
  if (!decision || decision === "combat") return "";
  const s = decision.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Loose label equality: "Strike → Jaw Worm" vs "AI: strike → jaw worm". */
function sameMove(a: string, b: string): boolean {
  const norm = (s: string) => s.replace(/^AI:\s*/i, "").replace(/\s+/g, " ").trim().toLowerCase();
  return norm(a) === norm(b);
}

export function toTimeline(entries: LogEntry[]): TimelineEntry[] {
  const out: TimelineEntry[] = [];
  let turn: number | null = null;
  let step: TimelineEntry | null = null;
  let enemyTurn = false;

  const open = (e: LogEntry, label: string, decision = "combat"): TimelineEntry => {
    const s: TimelineEntry = {
      idx: out.length,
      decision,
      label,
      details: [],
      combat_log: [],
      turn,
      floor: e.floor || null,
      act: e.act || null,
      act_name: null,
    };
    out.push(s);
    step = s;
    return s;
  };

  for (const e of entries) {
    if (e.actor === "game" && (FLOOR.test(e.action) || ENDED.test(e.action))) {
      turn = null;
      step = null;
      enemyTurn = false;
      continue;
    }
    const t = e.actor === "game" ? TURN.exec(e.action) : null;
    if (t) {
      turn = Number(t[1]);
      step = null;
      enemyTurn = false;
      continue;
    }
    if (e.actor === "mod") continue; // the mod's own notes are not the run's history

    if (e.actor === "game") {
      const r = e.row;
      // The enemy's turn opens under the "End turn" that handed it over, as
      // the workbench files it. The mod says whose turn a row happened on
      // (`side`); an enemy ACTOR is not enough -- Thorns or Curl Up reacting
      // to your Strike is an enemy row on your turn. An older mod without
      // `side`: only a monster's move opens it.
      const enemySide = r ? (r.side != null ? r.side === "enemy" : r.kind === "monster_move") : false;
      if (r && !enemyTurn && enemySide) {
        enemyTurn = true;
        open(e, "End turn");
      }
      const target: TimelineEntry = step ?? open(e, "Start of turn");
      if (r) {
        const row: CombatLogRow = {
          kind: r.kind,
          round: turn,
          side: enemyTurn ? "enemy" : "player",
          text: r.text,
          verb: r.verb,
          actor: r.actor,
          actor_side: r.actor_side,
          target: r.target,
          target_side: r.target_side,
          amount: r.amount,
          noun: r.noun,
          source: null,
        };
        // Every entry here was made by this call, so append in place (a copy
        // per row is quadratic over a long turn).
        (target.combat_log ??= []).push(row);
        if (r.killed && r.target) target.details.push(`${r.target} ☠ defeated`);
      } else {
        // An older mod: no row, just the sentence.
        target.details.push(e.action);
      }
      continue;
    }

    // A decision: yours or the AI's.
    enemyTurn = false;
    const prefix = decisionLabel(e.decision);
    let label = prefix ? `${prefix}: ${e.action}` : e.action;
    if (e.actor === "AI") label = `🤖 ${label}`;
    else if (e.advice && !sameMove(e.advice, e.action)) label += `  ·  ${e.advice}`;
    open(e, label, e.decision || "combat");
  }
  return out;
}
