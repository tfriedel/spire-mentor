import { assetUrl } from "../api";
import { versionedPublicPath } from "../gameVersion";

// `hits`/`damage` are omitted from sfx_map.json (never null) when the v0.107.1
// decompilation didn't let Task 4 resolve them for that move — 181/257 moves
// have `hits`, 178 have `damage`. Absent beats wrong: an omitted field lets
// `resolveMonsterAttack` skip that move rather than pin it to a made-up shape.
export interface MonsterMoveSfx {
  name: string;
  intent: string;
  hits?: number;
  damage?: number;
  event: string;
}
export interface MonsterSfx {
  moves: MonsterMoveSfx[];
  defaultAttack?: string;
  /** This monster's own death sound, when the decomp names one. */
  death?: string;
  /** This monster's own taking-a-hit sound, when the decomp names one. */
  hurt?: string;
  /** What its BODY sounds like when a hit gets through block — the material
   *  impact `SfxCmd.PlayDamage` plays (`MonsterModel.TakeDamageSfx`). Distinct
   *  from `hurt`, which is the creature's voice, and from `generic.blockHit`,
   *  which fires only when block absorbed the whole hit. */
  impact?: string;
}
/**
 * One orb's three sounds, as `OrbModel.cs:171-203` resolves them.
 *
 * All three are optional and the gaps are the GAME's, not missing data: an
 * orb class that never calls `PlayEvokeSfx()` has no evoke sound (Glass), and
 * one that does gets either its FMOD override (Lightning) or the
 * `res://debug_audio/<slug>_evoke.mp3` fallback (Dark, Frost, Plasma). The
 * file tree cannot be used to guess this — `glass_orb_evoke.mp3` ships and is
 * never played.
 */
export interface OrbSfx {
  passive?: string;
  evoke?: string;
  channel?: string;
}

/** A playable character's own three sounds (`CharacterModel.cs:200-206`). */
export interface CharacterSfx {
  attack?: string;
  cast?: string;
  die?: string;
}

/**
 * Which of its character's sounds a card makes the PLAYER produce.
 *
 * `except` names the ONE character for whom this card animates a trigger the
 * game's `CreatureCmd.TriggerAnim` switch doesn't handle, i.e. who makes no
 * sound at all: Ironclad's heavy-attack cards (Mangle, Bludgeon, ...) fire
 * `"heavyAttack"` for him and plain `"Attack"` for anyone else; Necrobinder's
 * summon cards fire `"summonTrigger"` for her and `"Cast"` for anyone else.
 */
export interface CardAnim {
  kind: "attack" | "cast";
  except?: string;
}

// `{"Vulnerable": "Debuff", "Strength": "Buff", …}` — every power the
// decompilation could classify, keyed by its display name (the exact string
// `EffectEvent.power` carries — see sts2-wiki/scripts/build_sfx_map.py's
// `load_power_table`). A power absent here (never observed in v0.107.1, but
// possible after a game update adds one before the map is regenerated) is not
// an error — `triggers.ts` falls back to a fixed direction rather than
// silencing the cue.
export interface SfxMap {
  version: string;
  generic: Record<string, string>;
  powers: Record<string, "Buff" | "Debuff">;
  /** The display names whose amount may go BELOW zero
   *  (`PowerModel.AllowNegative`) — v0.107.1: Strength, Dexterity, Focus,
   *  Shriek, Shrink. `powers` above is the power's static `Type`, which is not
   *  what the game renders: it renders `GetTypeForAmount(Amount)`, and this
   *  list is the half of that rule a name cannot express. See
   *  `powerValence.ts`. Optional for the same reason as `characters` below —
   *  a bucket may still be serving a map generated before it existed, and an
   *  absent list degrades to the name-only classification that shipped then. */
  powersAllowNegative?: string[];
  cards: Record<string, string>;
  relics: Record<string, string>;
  monsters: Record<string, MonsterSfx>;
  /** Keyed by CLASS name ("Ironclad"), which is what the wire sends as
   *  `state.player.name` — NOT the display title ("The Ironclad") the
   *  `cards`/`relics`/`monsters` tables are keyed by. Optional because a
   *  deployed bucket can still be serving a map generated before this table
   *  existed; the player then simply makes no sound, as it did before. */
  characters?: Record<string, CharacterSfx>;
  /** Keyed by card display title, like `cards`. Optional for the same reason
   *  as `characters`. */
  cardAnims?: Record<string, CardAnim>;
  /** The sound a card's attack makes ON ITS TARGET —
   *  `AttackCommand.WithHitFx`'s sfx/tmpSfx argument. 80 of the 81 entries are
   *  `res://debug_audio/` paths, which is why this layer was missing until the
   *  non-FMOD scheme existed. Keyed by card display title, like `cards`;
   *  optional for the same reason as `characters`. */
  cardHits?: Record<string, string>;
  /** Keyed by orb TYPE ("Frost"), which is what the wire sends as `Orb.type`.
   *  Optional for the same reason as `characters`. */
  orbs?: Record<string, OrbSfx>;
  /** The chest-opening sound per ACT NUMBER ("1", "2", "3") — the key the
   *  treasure-reveal payload carries. Optional, as above. */
  treasure?: Record<string, string>;
  /** Content hash of the rendered audio tree, appended to every audio URL as
   *  `?v=` — see `eventAssetUrl`. Optional: a map built without an audio root
   *  (a test fixture) has none, and the URLs are then simply un-busted. */
  audioRev?: string;
}

/**
 * The audio-tree fingerprint from the last loaded map, or "" before one loads.
 *
 * Module-level rather than threaded through every caller, mirroring
 * `api.ts`'s `assetsBase`: `SoundPlayer` builds asset URLs from a bare cue
 * and has no map reference, and giving it one to carry a single string would
 * push the map through `SoundContext` -> `SoundPlayer` -> `AudioBackend` for
 * no other reason. Set in `loadSfxMap`, which every sound path goes through
 * before a cue can exist.
 */
let audioRev = "";

/**
 * The game's two sound schemes, as the decomp writes them.
 *
 * `event:/…` is an FMOD event, rendered out of the banks. `res://debug_audio/…`
 * is a plain Godot stream played by `NDebugAudioManager` — the game's second,
 * non-FMOD sound layer, whose own doc comment calls it "temporary sound
 * effects (hopefully they make it into FMOD someday)". It is not a leftover:
 * combat victory, obtaining a relic and the per-card hit sounds all live
 * there and nowhere else.
 *
 * `encode_sfx` puts both families in one tree as `.webm`, so the only
 * difference here is how the path is derived — see `eventAssetUrl`.
 */
const DEBUG_AUDIO_SCHEME = "res://debug_audio/";

/**
 * `event:/sfx/block_gain` -> the URL of its audio file.
 *
 * The `audio/` prefix matters: the deployed bucket root holds the art tree
 * (so `audio/…` resolves to `$BUCKET/audio/…`), and the local dev server
 * mirrors that same layout by mounting `sts2-wiki/site/public/audio` at
 * `/assets/audio` alongside the images mount (`workbench/server.py`, final
 * review finding 2) — a sibling of `images_root` on disk, not a child of it,
 * so this prefix is load-bearing in BOTH environments, not just the bucket.
 * Also, not every event lives under `sfx/` (temp_sfx.bank events are
 * `event:/temp/sfx/…`).
 */
export function soundAssetRelPath(event: string): string {
  // `res://debug_audio/victory.mp3` -> `debug_audio/victory`. The source
  // extension is DROPPED rather than kept: the encoder transcodes mp3 and wav
  // alike into one `.webm` tree, so the scheme a cue came from must not leak
  // into the path.
  return event.startsWith(DEBUG_AUDIO_SCHEME)
    ? event.slice("res://".length).replace(/\.[^./]+$/, "")
    : event.replace(/^event:\//, "");
}

export function eventAssetUrl(event: string): string {
  const url = assetUrl(`audio/${soundAssetRelPath(event)}.webm`);
  // Cache-buster, and it is load-bearing rather than cosmetic. These URLs are
  // NOT content-addressed — the path comes from the FMOD event name and never
  // changes — yet the deploy serves them `max-age=31536000, immutable`
  // (`sts2-pycli/deploy/sync-assets.sh`). Re-encoding the corpus therefore
  // puts new bytes at an unchanged URL, and `immutable` means an already-
  // primed browser will not revalidate for a YEAR. Changing the URL is the
  // only thing that reaches those clients; it also makes `immutable` honest,
  // since the URL now moves whenever the bytes do.
  return audioRev ? `${url}?v=${audioRev}` : url;
}

/**
 * Pick which of a monster's attack sounds to play.
 *
 * The web never learns which move fired — `Enemy.intents` carries only
 * `{type, damage, hits}`, and even that `hits` can be missing — so we match
 * on hit count, the one discriminator both sides share, and fall back to the
 * monster's default attack sound (or its first attack move) whenever the
 * shape doesn't pin down exactly one move. A missing hit count, on either
 * side, means "cannot discriminate" — it is never treated as 0 or 1, since
 * that would silently pick whichever move happens to have that count.
 * Same idea as `effects/clips.ts::pickAttackClip`.
 */
export function resolveMonsterAttack(
  m: MonsterSfx | undefined,
  intent: { hits?: number } | undefined,
  fallback: string,
): string {
  if (!m) return fallback;
  const attacks = m.moves.filter((mv) => mv.intent === "Attack");
  const hits = intent?.hits;
  const exact = hits !== undefined ? attacks.filter((mv) => mv.hits === hits) : [];
  if (exact.length === 1) return exact[0].event;
  return m.defaultAttack ?? attacks[0]?.event ?? fallback;
}

/**
 * The sound of one monster MOVE, any kind — the attack, but also the buff,
 * the debuff, the summon (Axebot's `BOOT_UP_MOVE` is its `axebot_buff`).
 *
 * With the engine log, the workbench knows exactly which move fired
 * (`monster_move` rows carry its state id), so this picks that move's own
 * sound instead of guessing from the intent's hit count; before it existed,
 * 78 of the 259 mapped moves — every non-attack one — could never be heard.
 * The id join is the same one `effects/bindings.ts::ownerKeyCandidates`
 * makes for the VFX: the engine writes `THRASH` where most C# states are
 * `THRASH_MOVE` but some are bare (`SWING_1`), so both spellings are tried.
 *
 * Without an id that resolves (no log, or one of the handful of engine ids
 * that RENAME the C# state), it falls back to the intent the move was shown
 * with: an attack goes through `resolveMonsterAttack` exactly as before, and
 * any other intent is voiced only when the monster has exactly one move of
 * that kind. Otherwise `undefined` — silence beats a wrong sound.
 */
export function resolveMonsterMove(
  m: MonsterSfx | undefined,
  moveId: string | undefined,
  intent: { type: string; hits?: number } | undefined,
  fallback: string,
): string | undefined {
  if (m && moveId) {
    const exact = m.moves.find((mv) => mv.name === `${moveId}_MOVE`) ?? m.moves.find((mv) => mv.name === moveId);
    if (exact) return exact.event;
  }
  if (!intent) return moveId ? undefined : resolveMonsterAttack(m, undefined, fallback);
  if (intent.type === "Attack") return resolveMonsterAttack(m, intent, fallback);
  const same = (m?.moves ?? []).filter((mv) => mv.intent === intent.type);
  return same.length === 1 ? same[0].event : undefined;
}

/** One memoised fetch per map URL (one per game build). */
const cached = new Map<string, Promise<SfxMap>>();

/**
 * Fetch and memoise the generated map of game build `gameVersion` (GH #877):
 * the default build's is `/sfx_map.json`, every other build's
 * `/<ver>/sfx_map.json` (built from that build's decompile, so its new
 * cards / powers / monsters have their own cues). Called on the first sound
 * enable. A build whose map is missing (the public snapshot ships only the
 * default build's) falls back to the default build's map rather than going
 * silent.
 *
 * A failed fetch is NOT memoised: a rejected promise left in the cache would
 * wedge sound off for the rest of the session (a page opened before assets
 * settle, a network blip) — every later call would replay the same rejection
 * instead of retrying.
 */
export function loadSfxMap(
  fetchFn: typeof fetch = fetch,
  gameVersion?: string | null,
): Promise<SfxMap> {
  const url = versionedPublicPath("sfx_map.json", gameVersion);
  const hit = cached.get(url);
  if (hit) return hit;
  const fallback = url === "/sfx_map.json" ? null : () => loadSfxMap(fetchFn);
  const p = fetchFn(url)
    .then((r) => {
      if (!r.ok) throw new Error(`sfx_map.json: ${r.status}`);
      return r.json() as Promise<SfxMap>;
    })
    .then((map) => {
      // Before any cue can be built, so `eventAssetUrl` is never called with
      // a stale (or empty) rev while a map that carries one is in hand. The
      // audio tree is shared by every build, so every map carries the same.
      audioRev = map.audioRev ?? "";
      return map;
    })
    .catch((err: unknown) => {
      cached.delete(url);
      if (fallback) return fallback();
      throw err;
    });
  cached.set(url, p);
  return p;
}
