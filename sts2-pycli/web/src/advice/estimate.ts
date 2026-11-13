// Instinct's win estimate (GH #1222) in words. The numbers come from a small
// net on the student's state embedding; each one assumes Instinct plays every
// move from here, so the text says so.

import type { InstinctEstimate } from "../api";

const pct = (p: number) => `${Math.round(Math.max(0, Math.min(1, p)) * 100)}%`;

/** `−9 HP`, `+3 HP` (a heal), `±0 HP`: the fight's HP change, signed from the player's side. */
function hpDelta(loss: number): string {
  const v = Math.round(loss);
  if (v === 0) return "±0 HP";
  return v > 0 ? `−${v} HP` : `+${-v} HP`;
}

/** One line, or "" without an estimate. Combat:
 *  "If Instinct plays on: fight 93% · −9 HP (−2 to −20) · run 34%".
 *  Elsewhere: "If Instinct plays on: run 34%". */
export function estimateText(e: InstinctEstimate | null | undefined): string {
  if (!e || typeof e.p_run !== "number") return "";
  const parts: string[] = [];
  if (e.p_fight != null) parts.push(`fight ${pct(e.p_fight)}`);
  if (e.hp_loss) {
    const lo = hpDelta(e.hp_loss.q10);
    const hi = hpDelta(e.hp_loss.q90);
    parts.push(lo === hi ? hpDelta(e.hp_loss.q50) : `${hpDelta(e.hp_loss.q50)} (${lo} to ${hi})`);
  }
  parts.push(`run ${pct(e.p_run)}`);
  return `If Instinct plays on: ${parts.join(" · ")}`;
}

export type EstimateTone = "good" | "mid" | "bad";

/** One number of the top bar's chip: its label, percent and colour band. */
export interface ChipPart {
  label: "Run" | "Fight";
  text: string;
  tone: EstimateTone;
}

/** The top bar's "Run 71% · Fight 100%": the run always, the fight in combat.
 *  Bands: a run is good from 60%, bad under 30%; a fight good from 90%, bad
 *  under 60% (most hallway fights sit near 100%, so a fight under 90% is news). */
export function estimateChip(e: InstinctEstimate | null | undefined): ChipPart[] {
  if (!e || typeof e.p_run !== "number") return [];
  const band = (p: number, good: number, bad: number): EstimateTone =>
    p >= good ? "good" : p < bad ? "bad" : "mid";
  const parts: ChipPart[] = [{ label: "Run", text: pct(e.p_run), tone: band(e.p_run, 0.6, 0.3) }];
  if (e.p_fight != null) parts.push({ label: "Fight", text: pct(e.p_fight), tone: band(e.p_fight, 0.9, 0.6) });
  return parts;
}
