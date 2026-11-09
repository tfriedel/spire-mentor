/**
 * Seed-check statistics: what a set of attempts says about a seed.
 *
 * An attempt that ended in an engine error or hit the decision cap is
 * "errored": it is neither a win nor a loss, so it never tightens the bound.
 */

/** One attempt as `POST /api/seedcheck/attempt` returns it. */
export interface AttemptResult {
  k: number;
  seq: "check" | "rate";
  temp: number;
  won: boolean;
  /** `won`, `lost`, `capped`, `error` or `stopped` (decision cap). */
  outcome: string;
  error: string | null;
  floor: number;
  act: number;
  decisions: number;
  classes: number[];
  hp: number;
  /** The encounter the run ended in (display name); null for a win. */
  death_room: string | null;
}

export interface Tally {
  /** Attempts played, errored ones included. */
  played: number;
  wins: number;
  errored: number;
  bestFloor: number;
  /** The lowest-k win, else the furthest attempt (ties: lowest k). */
  best: AttemptResult | null;
}

export interface FloorDeaths {
  floor: number;
  room: string | null;
  count: number;
}

/** Whether an attempt counts toward wins and losses. */
export function isCounted(r: AttemptResult): boolean {
  return r.won || r.outcome === "lost";
}

/** With no win in `n` counted attempts, the per-attempt win chance is below
 *  this with 95% confidence (`1 - 0.05^(1/n)`, about 3/n). */
export function upperBound95(n: number): number {
  return n <= 0 ? 1 : 1 - 0.05 ** (1 / n);
}

/** Wilson 95% interval for `wins` of `n`. */
export function wilson95(wins: number, n: number): [number, number] {
  if (n <= 0) return [0, 1];
  const z = 1.959964;
  const p = wins / n;
  const den = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / den;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Losses per (floor, room), most first; wins and errored attempts dropped. */
export function deathsByFloor(rs: AttemptResult[]): FloorDeaths[] {
  const m = new Map<string, FloorDeaths>();
  for (const r of rs) {
    if (r.won || !isCounted(r)) continue;
    const key = `${r.floor}|${r.death_room ?? ""}`;
    const row = m.get(key) ?? { floor: r.floor, room: r.death_room, count: 0 };
    row.count += 1;
    m.set(key, row);
  }
  return [...m.values()].sort((a, b) => b.count - a.count || a.floor - b.floor);
}

export function tally(rs: AttemptResult[]): Tally {
  let wins = 0;
  let errored = 0;
  let best: AttemptResult | null = null;
  for (const r of rs) {
    if (r.won) wins += 1;
    else if (!isCounted(r)) {
      errored += 1;
      continue; // never the "best attempt": it did not finish
    }
    if (best === null) {
      best = r;
      continue;
    }
    const better = r.won !== best.won
      ? r.won
      : r.won
        ? r.k < best.k
        : r.floor > best.floor || (r.floor === best.floor && r.k < best.k);
    if (better) best = r;
  }
  const bestFloor = rs.reduce((m, r) => (isCounted(r) ? Math.max(m, r.floor) : m), 0);
  return { played: rs.length, wins, errored, bestFloor, best };
}
