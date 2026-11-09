// What the workbench highlights, decided without React — the port of the
// mod's AdviceMarks.cs (spire-mentor-mod/), so the browser and the game ring
// the same things with the same words.
//
// The advice is a ranked list of moves: the combat search's (or Instinct's)
// `Score` rows, or the run policy's options on every other screen. This file
// turns it into MARKS on the things the player will click — a ring and a step
// number on the card to play, the End Turn plate, a potion slot, a reward
// card, a shop slot, a campfire button — and decides which things get one and
// how loud it is. The components draw them.
//
// Rules (the mod's, reported 2026-11-07 after the first hand-played Instinct
// run):
//   * one mark per thing: a targeted card has one row PER TARGET; the card's
//     mark is its best row's, and the target travels with it;
//   * a move nobody would make (under 0.5% of the search or of the net's
//     probability) gets no mark at all — "0%" over a card is noise;
//   * a plan, when there is one, numbers the things 1, 2, 3 in order; without
//     one only the pick is numbered;
//   * the words a row is ranked with are a player's ("Best", "Also good",
//     "Okay", "Weak", "Mistake"); the engine's units stay behind the "engine
//     numbers" setting.

import type { ChoiceRankRow, PpoOption, Score } from "../api";

/** How loud a mark is. Ordered: a higher tier is louder. */
export type MarkTier = "alt" | "good" | "best";
const TIER_RANK: Record<MarkTier, number> = { alt: 1, good: 2, best: 3 };

/** The part of an advice row the marks need. */
export interface MarkOption {
  /** The thing the row acts on (`thingKey`), or null when it has none on screen. */
  key: string | null;
  label: string;
  isBest: boolean;
  inBand: boolean;
  /** Share of the search's visits, or the net's probability. */
  share: number | null;
  q: number | null;
  pickRank: number | null;
  targetIndex: number | null;
}

/** One step of a turn plan. */
export interface MarkPlanStep {
  key: string | null;
  targetIndex: number | null;
  label: string;
}

/** One thing to highlight. */
export interface Mark {
  key: string;
  tier: MarkTier;
  /** The step number drawn on the thing (1 = now), or null for an unnumbered ring. */
  order: number | null;
  targetIndex: number | null;
  label: string;
  share: number | null;
  q: number | null;
}

/** Below this share / probability a move is not shown at all. */
export const HIDDEN_SHARE = 0.005;
/** From this share a non-plan move is worth a thin ring. */
export const ALT_SHARE = 0.15;
/** A Q this close to the pick's is "as good" (search only). */
export const GOOD_Q_DELTA = 0.005;
/** Most rings drawn for alternatives, so a wide hand stays readable. */
export const MAX_ALTS = 2;

// ---------------------------------------------------------------------------
// Thing keys
// ---------------------------------------------------------------------------

// The recommender's sentinel stable_idx values (hint.ts mirrors them too).
const END_TURN = -1;
const POTION_BASE = -100;
const POTION_MIN = -200;

export const cardKey = (handIndex: number) => `card:${handIndex}`;
export const potionKey = (slot: number) => `potion:${slot}`;
export const END_TURN_KEY = "end_turn";

/** The thing a combat score row acts on: a hand card, a potion slot, End Turn. */
export function scoreKey(s: Pick<Score, "stable_idx">): string | null {
  const si = s.stable_idx;
  if (si === END_TURN) return END_TURN_KEY;
  if (si <= POTION_BASE && si > POTION_MIN) return potionKey(POTION_BASE - si);
  return si >= 0 ? cardKey(si) : null;
}

/**
 * The thing a screen action acts on, as the screen itself dispatches it:
 * `select_card_reward {card_index: 2}` → `select_card_reward|card_index=2`.
 * The run policy's options carry the same verb + args the screens send
 * (`resolve_workbench_verb`), so a screen asks for its button's mark with the
 * action it is about to send. Args are compared as strings, and a list (a
 * multi-select's indices) as its comma-joined form.
 */
export function actionKey(action: string, args: Record<string, unknown> = {}): string {
  const parts = Object.keys(args)
    .sort()
    .map((k) => {
      const v = args[k];
      return `${k}=${Array.isArray(v) ? v.join(",") : String(v)}`;
    });
  return parts.length ? `${action}|${parts.join("&")}` : action;
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/** Combat rows (the search's or Instinct's) as mark options. */
export function scoreOptions(scores: Score[], labelOf?: (s: Score) => string): MarkOption[] {
  const totalVisits = scores.reduce((n, s) => n + (s.visits || 0), 0);
  return scores.map((s) => ({
    key: scoreKey(s),
    label: labelOf ? labelOf(s) : s.action,
    isBest: !!s.is_best,
    inBand: !!s.in_band,
    share: s.prob != null ? s.prob : totalVisits > 0 ? s.visits / totalVisits : null,
    // Instinct's rows carry a probability, never a value.
    q: s.prob != null || totalVisits === 0 ? null : s.q,
    pickRank: s.pick_rank ?? null,
    targetIndex: s.target_index ?? null,
  }));
}

/** The in-fight card choice's ranking (POST /api/live/choice-ranker: one
 *  combat search, or Instinct's probabilities) as mark options, keyed by its
 *  own `choice:<index>` / `choice:skip`. */
export function choiceOptions(rows: ChoiceRankRow[]): MarkOption[] {
  const totalVisits = rows.reduce((n, r) => n + (r.visits || 0), 0);
  return rows.map((r) => ({
    key: r.key,
    label: r.label,
    isBest: !!r.is_best,
    inBand: false,
    share: r.prob != null ? r.prob : totalVisits > 0 ? r.visits / totalVisits : null,
    q: r.prob != null ? null : r.q,
    pickRank: null,
    targetIndex: null,
  }));
}

/** The run policy's options (PPO or Instinct) as mark options. */
export function policyOptions(options: PpoOption[]): MarkOption[] {
  return options.map((o) => ({
    key: actionKey(o.action, o.args),
    label: o.label,
    isBest: !!o.is_best,
    inBand: false,
    share: o.prob,
    q: null,
    pickRank: null,
    targetIndex: null,
  }));
}

// ---------------------------------------------------------------------------
// Marks
// ---------------------------------------------------------------------------

function better(a: MarkOption, b: MarkOption): boolean {
  if (a.isBest !== b.isBest) return a.isBest;
  if (a.pickRank != null && b.pickRank != null && a.pickRank !== b.pickRank) return a.pickRank < b.pickRank;
  if (a.q != null && b.q != null && Math.abs(a.q - b.q) > 1e-9) return a.q > b.q;
  return (a.share ?? 0) > (b.share ?? 0);
}

/** A move nobody would make: no mark, and folded away in the list. */
export function hidden(o: Pick<MarkOption, "isBest" | "share">): boolean {
  return !o.isBest && o.share != null && o.share < HIDDEN_SHARE;
}

function asGood(o: MarkOption, bestQ: number | null): boolean {
  return o.q != null && bestQ != null && o.q >= bestQ - GOOD_Q_DELTA && !hidden(o);
}

/** The marks for one advice: at most one per thing, best first, then the plan
 *  in order, then the alternatives by share. */
export function buildMarks(options: MarkOption[], plan?: MarkPlanStep[] | null): Mark[] {
  const bestQ = options.find((o) => o.isBest)?.q ?? null;

  const order = new Map<string, { order: number; step: MarkPlanStep }>();
  (plan ?? []).forEach((s, i) => {
    if (s.key != null && !order.has(s.key)) order.set(s.key, { order: i + 1, step: s });
  });

  const byThing = new Map<string, MarkOption>();
  for (const o of options) {
    if (o.key == null) continue;
    const have = byThing.get(o.key);
    if (!have || better(o, have)) byThing.set(o.key, o);
  }

  const marks: Mark[] = [];
  for (const [key, o] of byThing) {
    const planned = order.get(key);
    let tier: MarkTier;
    if (o.isBest || planned?.order === 1) tier = "best";
    else if (planned || o.inBand || asGood(o, bestQ)) tier = "good";
    else if (hidden(o)) continue;
    else if ((o.share ?? 0) >= ALT_SHARE) tier = "alt";
    else continue;
    marks.push({
      key,
      tier,
      order: planned ? planned.order : tier === "best" ? 1 : null,
      targetIndex: planned ? (planned.step.targetIndex ?? o.targetIndex) : o.targetIndex,
      label: o.label,
      share: o.share,
      q: o.q,
    });
  }
  // A planned step the options did not list still gets its number.
  for (const [key, { order: n, step }] of order) {
    if (!byThing.has(key)) {
      marks.push({ key, tier: n === 1 ? "best" : "good", order: n, targetIndex: step.targetIndex,
        label: step.label, share: null, q: null });
    }
  }

  marks.sort((a, b) =>
    TIER_RANK[b.tier] - TIER_RANK[a.tier]
    || (a.order ?? Infinity) - (b.order ?? Infinity)
    || (b.share ?? 0) - (a.share ?? 0));

  // Keep the alternatives few: rings everywhere say nothing.
  let alts = 0;
  return marks.filter((m) => m.tier !== "alt" || ++alts <= MAX_ALTS);
}

/** Marks by thing key, for the components' lookups. */
export function markIndex(marks: Mark[]): Map<string, Mark> {
  return new Map(marks.map((m) => [m.key, m]));
}

// ---------------------------------------------------------------------------
// Words for a row
// ---------------------------------------------------------------------------

export type Verdict = "Best" | "Also good" | "Then" | "Okay" | "Weak" | "Mistake" | "Unlikely" | "";

/**
 * A player's word for how a row compares with the pick. With a Q (the search)
 * it is the loss against the best move; with only a probability (Instinct, the
 * run policy) it is how sure the net is.
 */
export function verdict(o: Pick<MarkOption, "isBest" | "inBand" | "q" | "share">, bestQ: number | null): Verdict {
  if (o.isBest) return "Best";
  if (o.inBand) return "Also good";
  if (o.q != null && bestQ != null) {
    const loss = bestQ - o.q;
    return loss <= GOOD_Q_DELTA ? "Also good" : loss <= 0.02 ? "Okay" : loss <= 0.06 ? "Weak" : "Mistake";
  }
  if (o.share != null) return o.share >= 0.25 ? "Also good" : o.share >= 0.08 ? "Okay" : "Unlikely";
  return "";
}

/** A later plan step is judged at THIS decision, where it is not the move to
 *  make yet: it reads "Then", not "Unlikely". */
export function rowVerdict(o: Pick<MarkOption, "isBest" | "inBand" | "q" | "share">, bestQ: number | null,
  planStep: number | null): Verdict {
  return planStep != null && planStep >= 2 && !o.isBest ? "Then" : verdict(o, bestQ);
}

/** How a move that was MADE compares with the advice for the position it was
 *  made from — the float over a played card (the mod's SimMoveFeedback). */
export type Grade = "best" | "good" | "okay" | "weak" | "mistake";

export function gradeOf(v: Verdict): Grade | null {
  switch (v) {
    case "Best": return "best";
    case "Also good": case "Then": return "good";
    case "Okay": return "okay";
    case "Weak": case "Unlikely": return "weak";
    case "Mistake": return "mistake";
    default: return null;
  }
}

export const GRADE_TEXT: Record<Grade, string> = {
  best: "✓ Best",
  good: "✓ Good",
  okay: "Okay",
  weak: "?! Weak",
  mistake: "?? Mistake",
};

/** Is a grade on the AI's line (counted by the fight's accuracy line)? */
export const onLine = (g: Grade) => g === "best" || g === "good";

/**
 * Grade a move against the advice rows for the position it was made from: the
 * row for exactly that move (same thing, same target) when there is one, else
 * the thing's best row. Null when the advice does not cover the move.
 */
export function gradeMove(options: MarkOption[], key: string, targetIndex: number | null): Grade | null {
  const bestQ = options.find((o) => o.isBest)?.q ?? null;
  const same = options.filter((o) => o.key === key);
  if (same.length === 0) return null;
  const exact = same.find((o) => (o.targetIndex ?? null) === (targetIndex ?? null));
  const row = exact ?? same.reduce((a, b) => (better(b, a) ? b : a));
  return gradeOf(verdict(row, bestQ));
}

// ---------------------------------------------------------------------------
// No raw ids on screen
// ---------------------------------------------------------------------------

// An id the game writes in its tables: SHRUG_IT_OFF, CARD.SHRUG_IT_OFF. Two or
// more upper-case words joined by underscores, or a dotted CATEGORY.ID. A lone
// upper-case word ("AI", "HP") is left alone.
const ID_TOKEN = /\b(?:[A-Z][A-Z0-9]*\.)?[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|\b[A-Z][A-Z0-9]*\.[A-Z][A-Z0-9_]*\b/g;
const SMALL_WORDS = new Set(["a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"]);

/** "SHRUG_IT_OFF" → "Shrug It Off" (small words lower after the first). */
export function titleCase(id: string): string {
  return id.split("_").filter(Boolean).map((w, i) => {
    const l = w.toLowerCase();
    return i > 0 && SMALL_WORDS.has(l) ? l : l.charAt(0).toUpperCase() + l.slice(1);
  }).join(" ");
}

/** The last guard before a label reaches the screen: a raw game id in it
 *  becomes words. */
export function humanize(label: string | null | undefined): string {
  if (!label) return label ?? "";
  return label.replace(ID_TOKEN, (id) => titleCase(id.slice(id.lastIndexOf(".") + 1)));
}
