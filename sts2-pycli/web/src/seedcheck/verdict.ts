/**
 * The seed checker's text verdict, for pasting into a thread. It always
 * names the conditions (character, ascension, build, "simulator") and never
 * claims more than was measured: no "impossible", a bound instead. A run's
 * line carries its route floor by floor (which map node, what was there,
 * what was chosen, HP on leaving), so a reader can follow it. Paragraphs are
 * blank-line separated and the route is a "- " list, which Reddit and Discord
 * markdown keep apart.
 */
import { upperBound95, wilson95, type AttemptResult, type FloorDeaths, type Tally } from "./stats";

/** `POST /api/seedcheck/summary`. */
export interface Summary {
  won: boolean;
  outcome: string;
  hp: number;
  max_hp: number;
  floor: number;
  neow_pick: string | null;
  boss_relics: string[];
  events: { floor: number; event: string; option: string }[];
  cards_taken: { floor: number; card: string }[];
  shop: { floor: number; bought?: string; removed?: string }[];
  rests: { floor: number; choice: string; card?: string }[];
  /** One entry per map node walked, in order (`seedcheck::line_summary`). */
  route?: RouteStep[];
  deck: string[];
  relics: string[];
  death: { floor: number; act: number; room: string | null } | null;
}

/** A map node the run walked through. */
export interface RouteStep {
  act: number;
  floor: number;
  /** The map's node type: Monster, Elite, Unknown (a ?), RestSite, Shop, Treasure, Boss. */
  room: string;
  /** Its place in its row, counted from the left (1-based), of `of`. */
  pos: number;
  of: number;
  /** From the previous node; null at an act's first node and for a boss. */
  dir: "left" | "straight" | "right" | null;
  /** How many nodes the move could have reached. */
  options: number;
  /** The fight's encounter or the event, when there was one. */
  name: string | null;
  /** What the node turned out to be (a "?" can be either). */
  kind?: "fight" | "event" | null;
  notes: string[];
  hp?: number;
  max_hp?: number;
}

/** What the run was: shown in every header. */
export interface Conditions {
  character: string;
  ascension: number;
  version: string;
}

export const DEFAULT_CONDITIONS: Conditions = { character: "Ironclad", ascension: 10, version: "v0.111.0" };

const ROOM: Record<string, string> = {
  Monster: "Fight",
  Elite: "Elite",
  Unknown: "?",
  RestSite: "Rest",
  Shop: "Shop",
  Treasure: "Chest",
  Boss: "Boss",
  Ancient: "Ancient",
};

function ordinal(n: number): string {
  const s = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${s}`;
}

/** Which node to take, in words a player can act on. */
function whereTo(r: RouteStep): string {
  if (r.room === "Boss") return "";
  const place = `${ordinal(r.pos)} of ${r.of} from the left`;
  if (r.dir === null) return `start on the ${place}`;
  if (r.options <= 1) return `only path (${place})`;
  return `go ${r.dir} (${place})`;
}

/** The route as one "- " line per floor, grouped by act. */
export function routeLines(route: RouteStep[]): string[] {
  const out: string[] = [];
  let act = -1;
  for (const r of route) {
    if (r.act !== act) {
      act = r.act;
      if (out.length) out.push("");
      out.push(`Act ${act}:`, "");
    }
    const where = whereTo(r);
    const what = [r.name, ...r.notes].filter(Boolean).join("; ");
    const hp = r.hp !== undefined && r.max_hp !== undefined ? ` (HP ${r.hp}/${r.max_hp})` : "";
    const base = ROOM[r.room] ?? r.room;
    const room = r.room === "Unknown" && r.kind ? `${base} (${r.kind})` : base;
    out.push(`- F${r.floor} ${room}${where ? `, ${where}` : ""}${what ? `: ${what}` : ""}${hp}`);
  }
  return out;
}

const STARTERS = new Set(["Strike", "Defend", "Bash", "Ascender's Bane"]);
const CAVEAT =
  "Simulator result: following it in the real game only works where the engine matches the game on this seed. " +
  "Your fights will go differently from the bot's, so HP and some rewards can differ from the route.";

const ROUTE_HELP =
  "Route: one line per floor. \"Go left/straight/right\" is from the node you stand on; \"Nth of M from the left\" counts that floor's nodes on the map. HP is when leaving the floor.";

function header(seed: string, build: string, c: Conditions): string {
  return `Seed ${seed} — ${c.character} A${c.ascension}, ${c.version} (simulator, workbench ${build})`;
}

/** 0.0679 → "6.8", 0.18 → "18". */
function pct(x: number, digits = 1): string {
  return String(Number((x * 100).toFixed(digits)));
}

/** A bound rounded UP so it never claims more than was measured. */
function pctUp(x: number): string {
  return String(Math.ceil(x * 1000) / 10);
}

function deathLine(deaths: FloorDeaths[], label: string): string | null {
  if (!deaths.length) return null;
  const parts = deaths.slice(0, 3).map((d) => `floor ${d.floor} (${d.room ?? "unknown"}, ${d.count})`);
  return `${label}: ${parts.join(", ")}.`;
}

function deckHighlights(deck: string[]): string {
  const counts = new Map<string, number>();
  for (const c of deck) if (!STARTERS.has(c)) counts.set(c, (counts.get(c) ?? 0) + 1);
  return [...counts.entries()]
    .slice(0, 12)
    .map(([c, n]) => (n > 1 ? `${c} ×${n}` : c))
    .join(", ");
}

function winningLine(s: Summary): string[] {
  const out: string[] = [];
  if (s.neow_pick) out.push(`Neow: ${s.neow_pick}.`);
  if (s.boss_relics.length) out.push(`Boss relics: ${s.boss_relics.join(", ")}.`);
  // Events and shops are in the route, floor by floor; without one (an older
  // summary) name them here.
  if (!s.route?.length) {
    if (s.events.length) {
      out.push(`Events: ${s.events.map((e) => `${e.event} → ${e.option} (floor ${e.floor})`).join("; ")}.`);
    }
    const bought = s.shop.filter((x) => x.bought).map((x) => x.bought);
    const removed = s.shop.filter((x) => x.removed).map((x) => x.removed);
    if (bought.length || removed.length) {
      const bits = [bought.length ? `bought ${bought.join(", ")}` : "", removed.length ? `removed ${removed.join(", ")}` : ""];
      out.push(`Shops: ${bits.filter(Boolean).join("; ")}.`);
    }
  }
  const deck = deckHighlights(s.deck);
  if (deck) out.push(`Deck: ${deck}.`);
  out.push(`Ended with ${s.hp}/${s.max_hp} HP.`);
  return out;
}

export function checkVerdict(p: {
  seed: string;
  build: string;
  conditions?: Conditions;
  tally: Tally;
  firstWin: AttemptResult | null;
  summary: Summary | null;
  deaths: FloorDeaths[];
}): string {
  const lines = [header(p.seed, p.build, p.conditions ?? DEFAULT_CONDITIONS)];
  const counted = p.tally.played - p.tally.errored;
  if (p.firstWin) {
    const how = p.firstWin.temp === 0 ? "greedy" : "sampled";
    const n = p.tally.played;
    lines.push(
      `Winnable: the bot won on attempt ${p.firstWin.k} (${how}); ${n} attempt${n === 1 ? "" : "s"} played.`,
    );
    if (p.summary) lines.push(`Winning line (overview): ${winningLine(p.summary).join(" ")}`);
  } else if (counted === 0) {
    lines.push("No attempt finished, so there is no result for this seed.");
  } else {
    lines.push(
      `No win in ${counted} attempts: the bot's chance per attempt is below ${pctUp(upperBound95(counted))}% (95% confidence).`,
    );
    lines.push(`The best attempt reached floor ${p.tally.bestFloor}.`);
    const d = deathLine(p.deaths, "Most attempts died on");
    if (d) lines.push(d);
  }
  const route = p.summary?.route ?? [];
  if (route.length) {
    lines.push(p.firstWin ? ROUTE_HELP : ROUTE_HELP.replace("Route:", "The best attempt's route:"));
    lines.push(routeLines(route).join("\n"));
  }
  if (p.tally.errored) {
    const e = p.tally.errored;
    lines.push(`(${e} attempt${e === 1 ? "" : "s"} ended in an engine error and ${e === 1 ? "is" : "are"} not counted.)`);
  }
  lines.push(CAVEAT);
  return lines.join("\n\n");
}

export function rateVerdict(p: {
  seed: string;
  build: string;
  conditions?: Conditions;
  wins: number;
  n: number;
  deaths: FloorDeaths[];
  errored?: number;
}): string {
  const [lo, hi] = wilson95(p.wins, p.n);
  const lines = [
    header(p.seed, p.build, p.conditions ?? DEFAULT_CONDITIONS),
    `The bot won ${p.wins} of ${p.n} sampled runs (${pct(p.n ? p.wins / p.n : 0)}%, 95% interval ${pct(lo, 0)}–${pct(hi, 0)}%).`,
  ];
  const d = deathLine(p.deaths, "Most losses");
  if (d) lines.push(d);
  if (p.errored) lines.push(`(${p.errored} more ended in an engine error and are not counted.)`);
  lines.push(CAVEAT);
  return lines.join("\n\n");
}
