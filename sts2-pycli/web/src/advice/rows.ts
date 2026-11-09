// The ranked list in a player's words — the rows of the mod's advice panel
// (SimHud.BuildOptionRow, and the frame's OptionRow): a chip in the move's
// colour carrying its plan step, the move in words (wrapped, never cut), a
// verdict, and the AI's share. The engine's own units (Q, W/L/T) only with
// "engine numbers" on. Pure, so the ordering and the words are tested here.

import type { Card, PpoOption, Score } from "../api";
import {
  hidden, humanize, policyOptions, rowVerdict, scoreKey, scoreOptions, type Mark, type MarkOption, type Verdict,
} from "./marks";

export type RowKind = "attack" | "skill" | "power" | "potion" | "end_turn" | "skip" | "pick" | "other";

export interface AdviceRow {
  id: string;
  label: string;
  verdict: Verdict;
  /** The plan step on this row's chip (the number on the card), or null. */
  step: number | null;
  kind: RowKind;
  best: boolean;
  /** A move nobody would make: folded behind "… N more". */
  hidden: boolean;
  /** Share of the search's visits, or the net's probability. */
  share: number | null;
  /** The engine's value for the move (search only). */
  q: number | null;
  /** Rollout wins / losses / ties, or visits (search only). */
  wlt: string | null;
}

function kindOfCard(card: Card | undefined): RowKind {
  const t = (card?.type ?? "").toLowerCase();
  return t === "attack" || t === "skill" || t === "power" ? t : "other";
}

function finish(rows: AdviceRow[]): AdviceRow[] {
  // The pick first, then the hidden rows last, then the plan's order, then share.
  return rows.sort((a, b) =>
    Number(b.best) - Number(a.best)
    || Number(a.hidden) - Number(b.hidden)
    || (a.step ?? 1e9) - (b.step ?? 1e9)
    || (b.share ?? -1) - (a.share ?? -1));
}

/** The step a row's chip carries: its thing's mark number, on the row the mark
 *  was taken from (a targeted card's other targets get no number). */
function stepOf(o: MarkOption, marks: Map<string, Mark>): number | null {
  const m = o.key != null ? marks.get(o.key) : undefined;
  if (!m || m.order == null) return null;
  return (m.targetIndex ?? null) === (o.targetIndex ?? null) ? m.order : null;
}

/** Combat rows (the search's, or Instinct's). */
export function scoreRows(scores: Score[], marks: Map<string, Mark>, opts: {
  hand?: Card[];
  /** "Strike → Cultist": the row's label with its target named. */
  labelOf?: (s: Score) => string;
} = {}): AdviceRow[] {
  const options = scoreOptions(scores, opts.labelOf);
  const bestQ = options.find((o) => o.isBest)?.q ?? null;
  return finish(scores.map((s, i) => {
    const o = options[i];
    const key = scoreKey(s);
    const step = stepOf(o, marks);
    const kind: RowKind = key === "end_turn" ? "end_turn"
      : key?.startsWith("potion:") ? "potion"
      : kindOfCard(opts.hand?.find((c) => c.index === s.stable_idx));
    const searched = s.prob == null && s.visits > 0;
    return {
      id: `${s.stable_idx}:${s.target_index ?? ""}`,
      label: humanize(o.label),
      verdict: rowVerdict(o, bestQ, step),
      step,
      kind,
      best: o.isBest,
      hidden: hidden(o),
      share: o.share,
      q: searched ? s.q : null,
      wlt: searched ? `${s.wins ?? 0}/${s.losses ?? 0}/${s.ties ?? 0}` : null,
    };
  }));
}

/** The run policy's rows (PPO or Instinct, every screen but a fight). */
export function policyRows(options: PpoOption[], marks: Map<string, Mark>): AdviceRow[] {
  const mo = policyOptions(options);
  return finish(options.map((p, i) => {
    const o = mo[i];
    const step = stepOf(o, marks);
    return {
      id: `${i}:${o.key}`,
      label: humanize(p.label),
      verdict: rowVerdict(o, null, step),
      step,
      kind: /^skip|leave|proceed/.test(p.action) ? "skip" : "pick",
      best: o.isBest,
      hidden: hidden(o),
      share: o.share,
      q: null,
      wlt: null,
    };
  }));
}
