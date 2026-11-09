/** The seed checker's engine routes (answered by the page's engine worker). */
import { apiFetch } from "../transport";
import type { Seq } from "./protocol";
import type { Summary } from "./verdict";

export interface Preview {
  seed: string;
  length_ok: boolean;
  neow: string[];
  act1: string;
  bosses: string[][];
  ascension: number;
}

async function post<T>(route: string, body: unknown): Promise<T> {
  const r = await apiFetch(route, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = (await r.json().catch(() => null)) as (T & { detail?: string }) | null;
  if (!r.ok) throw new Error(j?.detail ?? `${route}: ${r.status}`);
  return j as T;
}

export interface SeedCheckApi {
  preview(seed: string, ascension: number): Promise<Preview>;
  summary(seed: string, ascension: number, classes: number[]): Promise<Summary>;
  saveLog(seed: string, ascension: number, classes: number[], seq: Seq, k: number): Promise<{ path: string }>;
}

export const seedCheckApi: SeedCheckApi = {
  preview: (seed, ascension) => post("/api/seedcheck/preview", { seed, ascension }),
  summary: (seed, ascension, classes) => post("/api/seedcheck/summary", { seed, ascension, classes }),
  saveLog: (seed, ascension, classes, seq, k) =>
    post("/api/seedcheck/log", { seed, ascension, classes, seq, k }),
};
