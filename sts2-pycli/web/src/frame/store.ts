// What the frame's panels show, folded from the mod's FrameLink feed
// (spire-mentor-mod/FrameLink.cs; protocol in spire-mentor-frame/README.md)
// and the host's own state messages. Pure reducer.

export interface Ui {
  mode: string;
  mode_name: string;
  power_label: string;
  style_label: string;
  coach_style: boolean;
  takeover_enabled: boolean;
  autoplay_armed: boolean;
  autobattle_on: boolean;
  speed: string;
  /** The advisor brain switch (mod protocol v5.2): shown where the advisor's
   *  Instinct can play this run, or while Instinct is still chosen. Optional:
   *  an older mod's `ui` has none. */
  brain?: { show: boolean; value: string; label: string };
  partner_go: { show: boolean; waiting: boolean; seconds: number };
  advisor_status: string;
  build_elapsed: string | null;
  hud_shown: boolean;
  hidden: boolean;
  last_action: string;
  note: string | null;
  status: string;
  /** Framed: the floating panels are around the game (in the frame). */
  ui_outside: boolean;
}

export interface AdviceOption {
  label: string;
  p_win: number | null;
  q: number | null;
  visits: number | null;
  visit_share: number | null;
  wins: number | null;
  losses: number | null;
  ties: number | null;
  is_best: boolean;
  low_confidence: boolean;
  in_band: boolean;
  pick_rank: number | null;
  card_index: number | null;
  score: number | null;
}

export interface Advice {
  note: string | null;
  p_win: number | null;
  best_q: number | null;
  low_confidence: boolean;
  best: string | null;
  pv: string[];
  pick_reason: string | null;
  pick_gap_hp: number | null;
  hand_back: string | null;
  /** v5.2: who answered (`search` / `instinct`), why Instinct did not, and
   *  the HUD's own source line. Optional: older mods send none. */
  brain?: string | null;
  instinct_unavailable?: string | null;
  source_tag?: string | null;
  options: AdviceOption[];
}

export interface Chip {
  id: string;
  kind: string;
  title: string;
  lines: string[];
  tone: "good" | "slip" | "blunder" | "info" | string;
  log_name: string | null;
  log_idx: number | null;
  in_combat: boolean;
}

export interface ShownChip extends Chip {
  shownAt: number;
  pinned: boolean;
}

export interface CoachPanelText {
  id: string;
  kind: string;
  title: string;
  summary: string;
  sections: { heading: string; lines: string[]; log_idx: number | null }[];
  footer: string;
  log_name: string | null;
}

export interface LogEntry {
  ts: string;
  actor: string;
  act: number;
  floor: number;
  decision: string;
  action: string;
  advice: string | null;
  /** Combat-history lines: the workbench combat-log row (CombatEventText). */
  row?: LogRow | null;
}

/** A combat event as the workbench's combat-log display row, minus the
 * round the frame fills in from where it sits in the log. */
export interface LogRow {
  kind: string;
  verb: string | null;
  actor: string | null;
  actor_side: "player" | "enemy" | null;
  target: string | null;
  target_side: "player" | "enemy" | null;
  amount: number | null;
  noun: string | null;
  text: string;
  killed: boolean;
  /** Whose turn it happened on (the entry's CurrentSide); absent from older mods. */
  side?: "player" | "enemy" | null;
}

export interface HostState {
  game: string;
  mod_connected: boolean;
}

export interface FrameState {
  host: HostState;
  ui: Ui | null;
  advice: Advice | null;
  peek: string | null;
  chips: ShownChip[];
  coachPanel: CoachPanelText | null;
  log: LogEntry[];
  /** Bumped by the mod's Ctrl+L (`toggle_panel` log). */
  toggles: { panel: string; seq: number } | null;
}

export const INITIAL_STATE: FrameState = {
  host: { game: "starting", mod_connected: false },
  ui: null,
  advice: null,
  peek: null,
  chips: [],
  coachPanel: null,
  log: [],
  toggles: null,
};

export const MAX_LOG = 1000;
export const MAX_CHIPS = 4;
const AUTOBATTLE = "autobattle";

/** Seconds a chip stays up unpinned (SimCoachChip's timers). */
export function chipSeconds(kind: string): number {
  if (kind === AUTOBATTLE) return Infinity;
  return kind === "turn" ? 20 : 25;
}

/** Drop unpinned chips past their time. */
export function expireChips(chips: ShownChip[], now: number): ShownChip[] {
  const kept = chips.filter((c) => c.pinned || now - c.shownAt < chipSeconds(c.kind) * 1000);
  return kept.length === chips.length ? chips : kept;
}

export type Incoming =
  | { type: "host"; game: string; mod_connected: boolean }
  | { type: "from_mod"; msg: ModMessage }
  | { type: "pin_chip"; id: string; pinned: boolean }
  | { type: "close_chip"; id: string }
  | { type: "close_coach_panel" }
  | { type: "tick"; now: number }
  | { type: "menu_pick"; id: string; index: number }
  | { type: "menu_closed"; id: string };

export type ModMessage =
  | { type: "hello"; protocol: number; actions: string[] }
  | { type: "ui"; ui: Ui }
  | { type: "advice"; advice: Advice | null; peek: string | null }
  | { type: "chip"; chip: Chip; age_ms?: number }
  | { type: "chip_withdraw"; ids: string[] }
  | { type: "chip_clear_unpinned" }
  | { type: "coach_panel"; panel: CoachPanelText }
  | { type: "coach_panel_hide"; ids: string[] | null }
  | { type: "log"; entry: LogEntry }
  | { type: "log_backlog"; entries: LogEntry[] }
  | { type: "toggle_panel"; panel: string }
  | { type: "pong" };

export function reduce(state: FrameState, m: Incoming, now = Date.now()): FrameState {
  switch (m.type) {
    case "host":
      return { ...state, host: { game: m.game, mod_connected: m.mod_connected } };
    case "pin_chip":
      return { ...state, chips: state.chips.map((c) => (c.id === m.id ? { ...c, pinned: m.pinned } : c)) };
    case "close_chip":
      return { ...state, chips: state.chips.filter((c) => c.id !== m.id) };
    case "close_coach_panel":
      return { ...state, coachPanel: null };
    case "menu_pick":
    case "menu_closed":
      return state; // host.ts showNativeMenu answers these
    case "tick": {
      const chips = expireChips(state.chips, m.now);
      return chips === state.chips ? state : { ...state, chips };
    }
    case "from_mod":
      return reduceMod(state, m.msg, now);
  }
}

function reduceMod(state: FrameState, m: ModMessage, now: number): FrameState {
  switch (m.type) {
    case "hello":
      // A (re)connected mod re-sends everything it is showing.
      return { ...state, chips: state.chips.filter((c) => c.pinned), coachPanel: null };
    case "ui":
      return { ...state, ui: m.ui };
    case "advice":
      return { ...state, advice: m.advice, peek: m.peek };
    case "chip":
      return { ...state, chips: addChip(state.chips, m.chip, now - (m.age_ms ?? 0)) };
    case "chip_withdraw":
      return { ...state, chips: state.chips.filter((c) => !m.ids.includes(c.id)) };
    case "chip_clear_unpinned":
      return { ...state, chips: state.chips.filter((c) => c.pinned) };
    case "coach_panel":
      return { ...state, coachPanel: m.panel };
    case "coach_panel_hide":
      if (!state.coachPanel) return state;
      if (m.ids && !m.ids.includes(state.coachPanel.id)) return state;
      return { ...state, coachPanel: null };
    case "log": {
      const log = [...state.log, m.entry];
      return { ...state, log: log.length > MAX_LOG ? log.slice(log.length - MAX_LOG) : log };
    }
    case "log_backlog":
      return { ...state, log: m.entries.slice(-MAX_LOG) };
    case "toggle_panel":
      return { ...state, toggles: { panel: m.panel, seq: (state.toggles?.seq ?? 0) + 1 } };
    case "pong":
      return state;
  }
}

/**
 * SimCoachChip's replacement rules: the same record replaces itself (and
 * keeps its pin: a reconnected mod replays the chips it showed); a new chip
 * replaces the unpinned one of its group (the coach's verdicts vs
 * autobattle's "since your last stop"); at most MAX_CHIPS, oldest unpinned
 * first out. `shownAt` is when the mod showed it (a replay sends its age).
 */
export function addChip(chips: ShownChip[], chip: Chip, shownAt: number): ShownChip[] {
  const group = chip.kind === AUTOBATTLE;
  const pinned = chips.some((c) => c.id === chip.id && c.pinned);
  let next = chips.filter((c) => c.id !== chip.id && (c.pinned || (c.kind === AUTOBATTLE) !== group));
  next = [...next, { ...chip, shownAt, pinned }];
  while (next.length > MAX_CHIPS) {
    const drop = next.findIndex((c) => !c.pinned);
    next.splice(drop >= 0 ? drop : 0, 1);
  }
  return next;
}

/** TieBandText.Lines: the "about equal" lines under the ranked list. */
export function tieBandLines(inBand: string[], reason: string | null, gapHp: number | null): string[] {
  if (inBand.length < 2) return [];
  const maxNamed = 3;
  let moves = inBand.slice(0, maxNamed).join(" ≈ ");
  if (inBand.length > maxNamed) moves += ` (+${inBand.length - maxNamed})`;
  let line = "about equal: " + moves;
  if (gapHp != null && Math.abs(gapHp) >= 0.5) line += ` (~${Math.abs(gapHp).toFixed(0)} HP)`;
  const lines = [line];
  if (reason) lines.push(`→ ${inBand[0]}: ${reason}`);
  return lines;
}

/** SimHud's ranked order: best first, then by visit share / p_win. */
export function rankedOptions(advice: Advice): AdviceOption[] {
  return advice.options
    .filter((o) => o.p_win != null || o.q != null || o.visit_share != null)
    .sort((a, b) => {
      if (a.is_best !== b.is_best) return a.is_best ? -1 : 1;
      return (b.visit_share ?? b.p_win ?? 0) - (a.visit_share ?? a.p_win ?? 0);
    });
}

/** The in-band labels, the pick first, then the engine's preference order. */
export function inBandLabels(advice: Advice): string[] {
  return advice.options
    .filter((o) => o.in_band)
    .sort((a, b) => (a.is_best ? -1 : (a.pick_rank ?? 1e9)) - (b.is_best ? -1 : (b.pick_rank ?? 1e9)))
    .map((o) => o.label);
}
