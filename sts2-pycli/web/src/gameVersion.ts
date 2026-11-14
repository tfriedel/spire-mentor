/**
 * Which game build a run plays (GH #801).
 *
 * The engine runs the stable v0.107.1 content or a later beta (v0.111.0,
 * v0.113.0 — the build that adds the Fool). The
 * server advertises what it can run in `GET /api/capabilities`
 * (`game_versions` / `default_game_version`); the launcher sends the pick as
 * `game_version` on start / start-faithful / import-state. Every reader treats
 * a missing key as the default, so payloads from before the selector existed
 * (and every 0.107 payload, whose writers omit the key) read as v0.107.1.
 */

export const DEFAULT_GAME_VERSION = "v0.107.1";

/** Launcher labels for the versions we know; an unknown one shows as-is. */
const LABELS: Record<string, string> = {
  "v0.107.1": "game v0.107.1 (stable)",
  "v0.111.0": "v0.111.0 (beta)",
  "v0.113.0": "v0.113.0 (beta, + the Fool)",
};

export function gameVersionLabel(v: string): string {
  return LABELS[v] ?? v;
}

/** The `game_version` a payload declares; null when it declares none.
 *
 *  Looks where each writer puts it: inside `state` for a live / replay `Step`
 *  (both backends write the key onto the game-state dict they serve as
 *  `Step.state`), at the top level for `export_persist` and saves, and in
 *  `meta`. */
export function declaredGameVersion(p: unknown): string | null {
  if (p && typeof p === "object") {
    const o = p as {
      game_version?: unknown;
      state?: { game_version?: unknown } | null;
      meta?: { game_version?: unknown } | null;
    };
    const s = o.state && typeof o.state === "object" ? o.state.game_version : undefined;
    for (const v of [s, o.game_version, o.meta?.game_version]) {
      if (typeof v === "string" && v) return v;
    }
  }
  return null;
}

/** The `game_version` a JSONL run log's head `meta` record declares
 *  (`{"type":"meta","data":{"game_version":...}}`); null when the log has no
 *  meta record or the record names no version. */
export function runLogGameVersion(text: string | null | undefined): string | null {
  if (!text) return null;
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    let rec: unknown;
    try {
      rec = JSON.parse(t);
    } catch {
      continue;
    }
    const r = rec as { type?: unknown; data?: { game_version?: unknown } | null } | null;
    if (r && typeof r === "object" && r.type === "meta") {
      const v = r.data?.game_version;
      return typeof v === "string" && v ? v : null;
    }
  }
  return null;
}

/** The `game_version` a payload carries, or the default when it carries none. */
export function payloadGameVersion(p: unknown): string {
  return declaredGameVersion(p) ?? DEFAULT_GAME_VERSION;
}

/** The version to actually send: the remembered pick when this server can run
 *  it, else the server's default. The pick itself is left alone, so a visit to
 *  an older (single-version) server does not overwrite the preference. */
export function effectiveGameVersion(
  pick: string, available: readonly string[], serverDefault: string = DEFAULT_GAME_VERSION,
): string {
  if (available.includes(pick)) return pick;
  return available.includes(serverDefault) ? serverDefault : (available[0] ?? DEFAULT_GAME_VERSION);
}

/** The URL of a per-build static table under `public/` (GH #877): the
 *  default build's at `/<file>` (where it always was), every other build's at
 *  `/<ver>/<file>` — `sfx_map.json`, `card_text.json`. */
export function versionedPublicPath(file: string, v: string | null | undefined): string {
  return v && v !== DEFAULT_GAME_VERSION ? `/${v}/${file}` : `/${file}`;
}

/** The request-body fragment for `v`: empty for the default, so a 0.107 run
 *  posts exactly the body it always did. */
export function gameVersionBody(v: string | null | undefined): { game_version?: string } {
  return v && v !== DEFAULT_GAME_VERSION ? { game_version: v } : {};
}
