/**
 * Follow mode for the workbench the Spire Mentor mod bundles.
 *
 * Follow (`?follow=1`, App.tsx) tails the run log the advisor is writing. The
 * Python workbench reads that file from disk; the browser build's run store
 * is IndexedDB, so this mirrors the file into it instead: the control plane
 * that serves the page answers `GET /log` (realgame_control.py), and a
 * changed log is written to the worker's run store with `POST /api/runs/put`
 * under its bare filename. From there the follow loop is the desktop one —
 * `/api/load` reopens it, and `/api/runs/stat` changes whenever a put did.
 */
import { bridgeFetch } from "./bridge";
import { apiFetch } from "./transport";

interface LogStat {
  name: string;
  size: number;
  mtime: number;
}

let mirrored: LogStat | null = null;

/** Mirror the advisor's current run log into the page's run store (only
 *  when it changed) and return its key there; null while there is none. */
export async function mirrorLiveLog(): Promise<string | null> {
  const r = await bridgeFetch("/log?stat=1");
  if (!r.ok) return null;
  const st = (await r.json()) as LogStat;
  if (mirrored && mirrored.name === st.name && mirrored.size === st.size && mirrored.mtime === st.mtime) {
    return st.name;
  }
  const full = await bridgeFetch("/log");
  if (!full.ok) return null;
  const log = (await full.json()) as LogStat & { text: string };
  // A log opened a moment ago holds no complete line yet; the run store
  // refuses empty text, and there is nothing to show anyway.
  if (!log.text.trim()) return null;
  const put = await apiFetch("/api/runs/put", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path: log.name, text: log.text }),
  });
  if (!put.ok) return null;
  mirrored = { name: log.name, size: log.size, mtime: log.mtime };
  return log.name;
}

/** Tests: forget what was mirrored. */
export function resetLiveLogMirror(): void {
  mirrored = null;
}
