import type { CardAnim, SfxMap } from "./sfxMap";
import { PITCH_VARIANCE, type SoundCue } from "./types";

// CORRECTION (2026-08-24, round 2). An earlier amendment here recorded that
// `victory` "is NOT a real key" and that "a win is silent by design, since the
// decomp has no victory event". Both halves were wrong, and the mistake was
// looking for it in the wrong place: there is no FMOD *event* for a win, but
// the game plays a win sound all the same —
// `NRewardsScreen._Ready` (`:400`) calls
// `NDebugAudioManager.Instance?.Play("victory.mp3")`, a plain Godot stream out
// of `res://debug_audio/`. The generated map now carries it (and `relicGet`
// and `neowWelcome` beside it), so `cueForUi(map, "victory")` resolves.
//
// The general lesson, worth more than this one key: the workbench's sound
// pipeline used to harvest FMOD banks ONLY, so every moment the game voices
// through `NDebugAudioManager` looked "silent by design" from the inside.
// Before concluding a moment has no sound, grep the decomp for BOTH
// `SfxCmd.Play` and `NDebugAudioManager`.
//
// RULING (2026-08-24, gold-tier fix): `gold` was removed from this union for
// the same reason `relic` was, one paragraph down — the sound is not a single
// event. `PlayerCmd.cs:152` picks between three by AMOUNT, so it needs the
// gained amount, which `cueForUi` has no way to take. `cueForGold` below owns
// it now and reads `map.generic` directly, exactly like `cueForRelic`.
//
// RULING (task-11 fix round 2, 2026-08-24): `relic` was removed from this
// union. Unlike `victory` (a real moment — a win happens, it's just silent by
// design), `relic` named no real code path: nothing ever called
// `cueForUi(map, "relic")` or `emitUi("relic")`. The actual relic-activation
// sound is `cueForRelic` below, a separate function that reads
// `map.relics[name] ?? map.generic.relic` directly and never touches
// `UiCueKind`. A coverage sweep's false-positive match (`"relic"` also being
// Shop.tsx's unrelated ware-kind string) is what surfaced this — see
// task-11-report.md. Absent beats wrong: a union member nothing can produce
// doesn't belong here.
export type UiCueKind =
  | "click" | "mapOpen" | "mapSelect" | "energy"
  | "victory" | "defeat" | "relicGet" | "neowWelcome"
  | "battleStart" | "playerTurn" | "enemyTurn"
  | "cardDeal" | "cardExhaust" | "cardSelect"
  | "cardSmith" | "cardTransform" | "enchant"
  | "potionGain" | "potionSlosh"
  | "restSleep" | "restBlanket" | "restDig"
  | "screenOpen" | "merchantWelcome" | "merchantThanks" | "summon";

/**
 * `keys` is a LIST because three of these families are randomised in the game,
 * not because the others might grow one: `NCombatStartBanner.cs:99`,
 * `NPotionHolder.cs:286` and `HealRestSiteOption.cs:98` each call
 * `Rng.Chaotic.NextItem(...)` over exactly the members listed here (the rest
 * jingles via the `sleep.tres` `AudioStreamRandomizer`, whose three streams
 * the map names directly). A single-key kind is the degenerate case of the
 * same shape, so there is one code path rather than two.
 */
//
// `gain` / `pitch` are the `volume` and `PitchVariance` arguments of the
// game's own `NDebugAudioManager.Play` call for that sound. The non-FMOD layer
// has no mixer data of its own, so the call site IS its mix: without these the
// card deal played four times louder than the game's (0.25), the combat-start
// slosh more than three times (0.3), and every one of them at one fixed pitch
// where the game spreads it. Absent = the call passes the defaults (1,
// `PitchVariance.None`). FMOD events carry their volume/pitch randomisation
// inside the bank (the renderer bakes one seeded take), so they take neither.
const UI_KEYS: Record<UiCueKind, {
  keys: readonly string[];
  cls: SoundCue["cls"];
  gain?: number;
  pitch?: number;
}> = {
  click: { keys: ["uiClick"], cls: "chatter" },
  mapOpen: { keys: ["mapOpen"], cls: "chatter" },
  mapSelect: { keys: ["mapSelect"], cls: "informative" },
  energy: { keys: ["energy"], cls: "chatter" },
  // Winning a fight: the sting the rewards screen opens with. See the
  // CORRECTION above for why this was believed not to exist.
  victory: { keys: ["victory"], cls: "informative" },
  defeat: { keys: ["defeat"], cls: "informative" },
  // Any relic entering the inventory (`RelicCmd.Obtain`), which is what a
  // Neow boon, a boss relic and a shop purchase all funnel through. Distinct
  // from `cueForRelic`, which is a relic ACTIVATING later in a fight.
  relicGet: { keys: ["relicGet"], cls: "informative" },
  // Neow greeting you as her room opens.
  neowWelcome: { keys: ["neowWelcome"], cls: "informative" },
  // --- combat rhythm ---
  battleStart: { keys: ["battleStart1", "battleStart2"], cls: "informative" },
  playerTurn: { keys: ["playerTurn"], cls: "informative" },
  enemyTurn: { keys: ["enemyTurn"], cls: "informative" },
  // Chatter, both of them: `card_deal` fires per card at 0.25 volume in the
  // game and `card_exhaust` rides a vfx the workbench does not draw in
  // `reduced` mode, so dropping them there keeps sound level with the visuals.
  // `CardPileCmd.cs:875`.
  cardDeal: { keys: ["cardDeal"], cls: "chatter", gain: 0.25, pitch: PITCH_VARIANCE.small },
  cardExhaust: { keys: ["cardExhaust"], cls: "chatter" },
  // `NMouseCardPlay.cs:336`: the workbench is mouse-driven (the controller
  // path, `NControllerCardPlay.cs:199`, plays it at 1).
  cardSelect: { keys: ["cardSelect"], cls: "chatter", gain: 0.5 },
  // --- deck operations ---
  // `NCardSmithVfx.cs:179` (its `SfxVolume` defaults to 1).
  cardSmith: { keys: ["cardSmith"], cls: "informative", pitch: PITCH_VARIANCE.small },
  cardTransform: { keys: ["cardTransform"], cls: "informative" },
  enchant: { keys: ["enchant"], cls: "informative" },
  // --- potions ---
  // `PotionCmd.cs:44`.
  potionGain: { keys: ["potionGain"], cls: "informative", pitch: PITCH_VARIANCE.small },
  // `NPotionHolder.cs:662` (`ShineOnStartOfCombat`), the only slosh the
  // workbench plays: entering a fight while holding a potion.
  potionSlosh: {
    keys: ["potionSlosh1", "potionSlosh2", "potionSlosh3"], cls: "chatter",
    gain: 0.3, pitch: PITCH_VARIANCE.large,
  },
  // --- rest site ---
  restSleep: { keys: ["restSleep1", "restSleep2", "restSleep3"], cls: "informative" },
  // `HealRestSiteOption.PlayRestSiteHealSfx` (`:99`).
  restBlanket: { keys: ["restBlanket"], cls: "informative", pitch: PITCH_VARIANCE.small },
  // `DigRestSiteOption.DoLocalPostSelectVfx` (`:35`).
  restDig: { keys: ["restDig"], cls: "informative", pitch: PITCH_VARIANCE.small },
  // --- screens / NPCs ---
  screenOpen: { keys: ["screenOpen"], cls: "chatter" },
  // Necrobinder's Osty arriving on the board.
  summon: { keys: ["summon"], cls: "informative" },
  merchantWelcome: { keys: ["merchantWelcome"], cls: "informative" },
  merchantThanks: { keys: ["merchantThanks"], cls: "informative" },
};

/**
 * Every `UiCueKind` the type can hold, for a coverage test to enumerate at
 * runtime — the type itself has no runtime representation, and hand-copying
 * the kind list into a test would silently stop covering a kind added here
 * later. `UI_KEYS` stays module-private (it carries the `cls` classification
 * map, which the test doesn't need) — this exports only the key list.
 */
export const ALL_UI_CUE_KINDS: readonly UiCueKind[] = Object.keys(UI_KEYS) as UiCueKind[];

/**
 * The sound the PLAYER's own body makes when this card is played, or `null`.
 *
 * This is the layer the workbench used to be missing entirely, and the reason
 * an Ironclad Strike sounded like nothing but a card-shuffle woosh. In the
 * game the swoosh below is the card FLYING TO THE DISCARD PILE; the hit you
 * actually hear comes from `CreatureCmd.TriggerAnim`, which — for a creature
 * that `IsPlayer` — plays `character.AttackSfx`/`CastSfx` before setting the
 * animation trigger (v0.107.1 `CreatureCmd.cs:928`).
 *
 * Both halves must resolve, and neither is assumed: a map generated before
 * `characters`/`cardAnims` existed (a stale deploy) has neither table, and a
 * card the decomp says animates NOTHING — Defend is the headline case, along
 * with 179 others — is simply absent from `cardAnims`. Every one of those
 * cases means silence here, not a substituted sound.
 */
function characterCueEvent(
  map: SfxMap,
  cardName: string,
  character: string | undefined,
): string | undefined {
  if (!character) return undefined;
  const anim: CardAnim | undefined = map.cardAnims?.[cardName];
  const sfx = map.characters?.[character];
  if (!anim || !sfx) return undefined;
  // The one character this card animates with a trigger name the game's
  // switch doesn't handle — Ironclad swinging `heavyAttack`, Necrobinder
  // casting `summonTrigger`. Those play nothing in the real game, so they
  // must play nothing here; every OTHER character gets the ordinary sound.
  if (anim.except === character) return undefined;
  return anim.kind === "cast" ? sfx.cast : sfx.attack;
}

/**
 * Every sound playing one card makes, in the order the game layers them.
 *
 * Up to four, and they are ADDITIVE, not alternatives — a mistake the
 * previous single-cue version baked in by returning the card's own sound
 * INSTEAD of the generic one. Playing Whirlwind really does stack all three:
 * `Whirlwind.OnPlay` plays `ironclad_whirlwind` directly, its `FromCard`
 * attack chain then triggers the "Attack" animation (so `ironclad_attack`
 * too — `AttackCommand.cs:220`/`:562`/`:570` fire the card's own attacker
 * sfx and the animation trigger one after the other, not one or the other),
 * and the card still swooshes into the discard pile.
 *
 * The fourth layer is the HIT — what the attack sounds like landing on the
 * target, from `AttackCommand.WithHitFx`. It is the layer a player notices
 * most and the workbench had none of, because 80 of the 81 cards that carry
 * one name a `res://debug_audio/` mp3 rather than an FMOD event (Bash and
 * Headbutt thud with `blunt_attack`, Mangle and Break with `heavy_attack`,
 * Slice with `slash_attack`, Precise Cut with `dagger_throw`). It is ordered
 * after the character's own grunt and before the swoosh, matching
 * `AttackCommand.cs:562-586`: attacker sfx, attacker anim, then hit sfx.
 *
 * Returns `[]` when nothing resolves — never a cue with an `undefined`
 * event, which would build a garbage asset URL downstream.
 *
 * AMENDMENT (i), still load-bearing: `cardType` is compared
 * case-INSENSITIVELY. `Card.type` is typed bare `string` with no union
 * (api.ts), and CardFrame.tsx:86 — the only other consumer of this field in
 * the app — normalises with `(type ?? "").toLowerCase()` before comparing,
 * which is direct evidence the backend's casing is not to be relied on.
 * Matching case-sensitively against `"Power"` would silently give every
 * lowercase-`"power"` card the ordinary play sound instead.
 */
export function cuesForCard(
  map: SfxMap,
  cardName: string,
  cardType: string,
  character?: string,
): SoundCue[] {
  const cues: SoundCue[] = [];
  const player = characterCueEvent(map, cardName, character);
  if (player) cues.push({ event: player, cls: "informative" });

  const own = map.cards[cardName];
  if (own) cues.push({ event: own, cls: "informative" });

  const hit = map.cardHits?.[cardName];
  if (hit) cues.push({ event: hit, cls: "informative" });

  const t = (cardType ?? "").toLowerCase();
  const key = t === "power" ? "cardPower" : "cardPlay";
  const swoosh = map.generic[key] ?? map.generic.cardPlay;
  if (swoosh) cues.push({ event: swoosh, cls: "informative" });

  return cues;
}

/**
 * The gold-pickup sound for gaining `amount`, or `null` when the map names none.
 *
 * Three tiers, thresholded on the amount exactly as `PlayerCmd.cs:152` does:
 * `amount >= 100 ? gold_3 : amount > 30 ? gold_2 : gold_1`. Note `>= 100` but
 * `> 30` — the asymmetry is the decomp's, not a typo, so 30 gold takes the
 * SMALL sound and 100 takes the large one.
 *
 * Falls back to the untiered `gold` key so a map generated before the tiers
 * existed still makes the sound it used to, rather than going silent.
 */
export function cueForGold(map: SfxMap, amount: number): SoundCue | null {
  const key = amount >= 100 ? "goldLarge" : amount > 30 ? "goldMedium" : "goldSmall";
  const event = map.generic[key] ?? map.generic.gold;
  return event ? { event, cls: "informative" } : null;
}

/** A relic's activation sound: its own if named, else the generic one, else `null`. */
export function cueForRelic(map: SfxMap, relicName: string): SoundCue | null {
  const event = map.relics[relicName] ?? map.generic.relic;
  return event ? { event, cls: "informative" } : null;
}

/**
 * A UI moment's sound, or `null` when this map has no entry for it.
 *
 * For a randomised family (battle start, potion slosh, rest jingle) one of the
 * variants is picked uniformly, the way the game does. `rand` is injected so
 * the choice is testable; only the variants the map actually resolves are
 * candidates, so a partially-regenerated map degrades to the ones it has
 * rather than going silent a third of the time.
 */
export function cueForUi(
  map: SfxMap,
  kind: UiCueKind,
  rand: () => number = Math.random,
): SoundCue | null {
  const spec = UI_KEYS[kind];
  const events = spec.keys.map((k) => map.generic[k]).filter(Boolean);
  if (events.length === 0) return null;
  // `Math.min` guards the one input this cannot control: a `rand` returning
  // exactly 1 (or a hair under, then rounding up) would index past the end.
  const event = events[Math.min(events.length - 1, Math.floor(rand() * events.length))];
  const cue: SoundCue = { event, cls: spec.cls };
  if (spec.gain !== undefined) cue.gain = spec.gain;
  if (spec.pitch !== undefined) cue.pitchVariance = spec.pitch;
  return cue;
}

/**
 * The chest-opening sound for `act`, or `null`.
 *
 * `ActModel.ChestOpenSfx` is per act and both act-1 variants (Overgrowth and
 * Underdocks) point at the same file, so the map is keyed by act NUMBER —
 * which is also the only thing the treasure-reveal payload carries.
 */
export function cueForTreasure(map: SfxMap, act: number): SoundCue | null {
  const event = map.treasure?.[String(act)];
  return event ? { event, cls: "informative" } : null;
}
