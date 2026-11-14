/** The playable characters, keyed the way the art tree names them. */
export const CHARACTER_KEYS: ReadonlySet<string> = new Set([
  "ironclad", "silent", "defect", "regent", "necrobinder", "fool",
]);

/** Characters a later game build added → the first build that has them. Their
 *  art is in that build's overlay (`<ver>/character_sprites/…`, written by
 *  `import_game`'s art-overlay step), not in the shared tree. */
export const LATER_CHARACTERS: Readonly<Record<string, string>> = { fool: "v0.113.0" };

/** `rel` (an art path for character `key`) where the art actually is: under
 *  the overlay of the build that added the character, else as-is. */
export function characterArt(key: string, rel: string): string {
  const since = LATER_CHARACTERS[key];
  return since ? `${since}/${rel}` : rel;
}

/** `v0.113.0` -> [0, 113, 0]; unparsable parts read as 0. */
function versionParts(v: string): number[] {
  return v.replace(/^v/, "").split(".").map((n) => Number.parseInt(n, 10) || 0);
}

/** Whether game build `version` has the character with art key `key`. */
export function characterInBuild(key: string | null, version: string): boolean {
  const since = key ? LATER_CHARACTERS[key] : undefined;
  if (!since) return true;
  const [a, b] = [versionParts(version), versionParts(since)];
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    if (d !== 0) return d > 0;
  }
  return true;
}

/** ``state.player.name`` → the art-tree key, or ``null`` when we ship no art.
 *
 *  Mirrors `workbench/assets.py slugify`, **including its leading-"the"
 *  strip**. The two backends disagree on the shape of this field: the engine
 *  lane ships the bare `"Ironclad"`, while the live-game lane ships the
 *  localized title — `sts2_realgame._character_title` builds `"The " +
 *  <Title Case>` from `characters.json`'s `<ID>.title`, which every one of the
 *  five carries. Both have to land on the same key.
 *
 *  This lives in one place because it did not: three components each derived
 *  it separately and none of them stripped the article, so every live-game run
 *  silently lost the rest-site hero, the run-end sprite and the HUD portrait
 *  while combat still looked right (that path uses the server-resolved
 *  `player.rig`). */
export function characterKey(name?: string | null): string | null {
  const k = (name ?? "")
    .trim()
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/[^a-z]/g, "");
  return CHARACTER_KEYS.has(k) ? k : null;
}
