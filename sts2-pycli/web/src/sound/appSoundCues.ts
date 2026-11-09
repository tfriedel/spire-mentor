import type { Card, CombatState } from "../api";
import type { SfxMap } from "./sfxMap";
import type { SoundCue } from "./types";
import { cuesForCard, cueForGold, cueForUi } from "./runCues";

/**
 * Run-level cue DECISIONS that App.tsx's handlers/effects call straight
 * through into `sound.emit`. Pulled out as plain functions (no React, no
 * `sound` context) for the same reason `runControl.ts` / `liveSeek.ts` /
 * `regretCache.ts` already sit beside App.tsx rather than inside it: App.tsx
 * itself has no test harness in this codebase (nothing renders it — every
 * existing test targets a standalone component), so the only way these
 * decisions get mutation-provable coverage is to make them testable without
 * rendering App at all.
 */

/**
 * The cues for playing `cardIndex` out of `hand` — the character's own
 * attack/cast sound, the card's signature sound, and the card swoosh, as
 * many of the three as resolve (see `cuesForCard`).
 *
 * `character` is the run's character CLASS name, which the wire sends as
 * `state.player.name`. It is read from the live state rather than from
 * App.tsx's `character` select, because that select is the NEXT run's
 * setting: loading a save or opening a replay doesn't touch it, so it would
 * give a loaded Silent run the Ironclad's grunts.
 *
 * Returns `[]` when the index isn't in hand (defensive — the caller always
 * sources `cardIndex` from a click on a card actually in `hand`, but the
 * lookup can't be assumed to succeed) or the map has nothing for it.
 */
export function cardPlayCues(
  map: SfxMap,
  hand: Card[],
  cardIndex: number,
  character?: string,
): SoundCue[] {
  const card = hand.find((c) => c.index === cardIndex);
  if (!card) return [];
  return cuesForCard(map, card.name, card.type, character);
}

/**
 * The gold cue exactly when gold INCREASED since the last observation —
 * never on a shop spend (which decreases it), and never on the very first
 * observation of a run (`prevGold` null: there is nothing to diff against,
 * and treating a fresh run's starting gold as a "gain" would fire on load).
 *
 * The DELTA is what picks the sound, not the new total: the game thresholds
 * on the amount gained (`PlayerCmd.cs:152`), so clearing a 20-gold combat
 * while sitting on 400 gold must still be the small chime. See `cueForGold`.
 */
export function goldGainCue(map: SfxMap, prevGold: number | null, nextGold: number): SoundCue | null {
  if (prevGold == null || nextGold <= prevGold) return null;
  return cueForGold(map, nextGold - prevGold);
}

/**
 * The run-end cue: only at the `game_over` decision, victory or defeat.
 * `decision` is passed in (rather than assumed) so a caller that got the
 * gating wrong upstream shows up here instead of silently mis-firing.
 */
export function runEndCue(map: SfxMap, decision: string, victory: boolean | undefined): SoundCue | null {
  if (decision !== "game_over") return null;
  return cueForUi(map, victory ? "victory" : "defeat");
}

/**
 * The relic-obtained cue, exactly when the inventory GREW.
 *
 * `RelicCmd.Obtain` (`:49`) plays `relic_get.mp3` for every relic the local
 * player gains, whatever handed it over — the Neow boon, a boss drop, a shop
 * purchase, a chest. Diffing the count rather than watching any one screen is
 * what makes all four work from one place.
 *
 * `prevCount` null means no baseline: the first state of a run, or a step
 * whose payload omits `relics` altogether (it is optional on the wire, and a
 * screen that drops it would otherwise read as "lost every relic", then as
 * "gained them all back" on the next step that carries them).
 *
 * Gaining several at once is still ONE cue. The game would play one per
 * `Obtain` call, but they land in the same frame and the player's per-event
 * debounce would collapse them anyway; emitting one is the same sound with
 * less to explain.
 */
export function relicGainCue(map: SfxMap, prevCount: number | null, nextCount: number): SoundCue | null {
  if (prevCount == null || nextCount <= prevCount) return null;
  return cueForUi(map, "relicGet");
}

/**
 * The potion-obtained cue, exactly when the belt GREW.
 *
 * `PotionCmd` (`:44`) plays `gain_potion.mp3` for every potion the local
 * player picks up. Same diff shape, and same `null` baseline rule, as
 * `relicGainCue` — `potions` is optional on the wire too.
 */
export function potionGainCue(map: SfxMap, prevCount: number | null, nextCount: number): SoundCue | null {
  if (prevCount == null || nextCount <= prevCount) return null;
  return cueForUi(map, "potionGain");
}

/**
 * The cues for ARRIVING at a decision screen the game voices on open.
 *
 * Three of them, all `_Ready`-time sounds in the game and all keyed off the
 * same transition, so they share one function rather than three near-copies:
 *
 * - `card_reward` -> the victory sting (`NRewardsScreen.cs:400`)
 * - `combat_play` -> one of the two battle-start stings
 *   (`NCombatStartBanner.cs:99`), PLUS a potion slosh if you are holding one
 *   (`NPotionHolder.ShineOnStartOfCombat`, `:662`)
 * - `shop`       -> the merchant's welcome (`NMerchantInventory.cs:384`)
 *
 * `prevDecision` null means there is nothing to transition FROM — a page
 * load, or a save restored straight onto one of these screens. Announcing a
 * victory the player didn't just win (or a fight that started before they
 * opened the tab) is worse than silence, so it returns nothing.
 */
export function arrivalCues(
  map: SfxMap,
  prevDecision: string | null,
  decision: string,
  opts: { holdsPotion?: boolean; rand?: () => number } = {},
): SoundCue[] {
  if (prevDecision === null || prevDecision === decision) return [];
  // Written as direct `cueForUi(...)` calls rather than a kinds array, and
  // that is load-bearing: `sfxMapCoverage`'s reachability sweep looks for the
  // kind literal AT a call site, so an indirection through a helper would let
  // a `UiCueKind` nothing can produce sit in the union unnoticed — the exact
  // rot that test exists to catch (see its `relic` note).
  const cues: (SoundCue | null)[] = [];
  if (decision === "card_reward") cues.push(cueForUi(map, "victory", opts.rand));
  if (decision === "combat_play") {
    cues.push(cueForUi(map, "battleStart", opts.rand));
    if (opts.holdsPotion) cues.push(cueForUi(map, "potionSlosh", opts.rand));
  }
  if (decision === "shop") cues.push(cueForUi(map, "merchantWelcome", opts.rand));
  return cues.filter((c) => c !== null);
}

/**
 * The cues an ACTION makes as it is taken, before its result comes back.
 *
 * These are the moments no state diff can see, because the sound belongs to
 * the choice rather than to its outcome: the shop's thank-you is the same
 * whatever you bought, and Rest vs Dig differ only in which option index you
 * sent. `liveAction` is the single funnel every interactive action goes
 * through, so this reads the state the action was taken FROM.
 *
 * Deliberately conservative — an action this doesn't recognise is silent, not
 * guessed at.
 */
export function actionCues(
  map: SfxMap,
  state: CombatState,
  decision: string,
  action: string,
  args: Record<string, unknown>,
  rand: () => number = Math.random,
): SoundCue[] {
  const cues: (SoundCue | null)[] = [];
  // Playing a card. It used to be voiced only from the Board's click handler,
  // so every card the ADVISOR played — "play recommended", the A-key autoplay
  // — went silent; this is the funnel all of them share. `card_index` is the
  // card's position in the hand the action was taken from, i.e. `state.hand`.
  if (action === "play_card" && typeof args.card_index === "number") {
    cues.push(...cardPlayCues(map, state.hand ?? [], args.card_index, state.player?.name));
  }
  // Any purchase. `NMerchantInventory.cs:460` plays the thank-you on every
  // successful buy, whatever the ware was.
  if (action.startsWith("buy_")) cues.push(cueForUi(map, "merchantThanks", rand));

  if (decision === "rest_site" && action === "choose_option") {
    const index = args.option_index;
    const id = (state.rest_options ?? []).find((o) => o.index === index)?.option_id;
    // `HealRestSiteOption.cs:98-99` plays BOTH: the `sleep.tres` randomiser
    // and then the blanket. `DigRestSiteOption.cs:35` plays the shovel. The
    // other seven options (Smith, Cook, Clone, …) have no sound of their own.
    if (id === "HEAL") {
      cues.push(cueForUi(map, "restSleep", rand), cueForUi(map, "restBlanket", rand));
    }
    if (id === "DIG") cues.push(cueForUi(map, "restDig", rand));
  }

  // A deck operation committing. `context` is the same field `choiceKinds.ts`
  // reads to tell a deck op from an in-combat pick; the three that animate
  // are upgrade (`NCardSmithVfx`), transform (`NCardTransformVfx`) and
  // enchant (`NCardEnchantVfx`). `remove` and `grant` animate no sound.
  if (decision === "card_select") {
    const context = (state as { context?: string }).context;
    if (context === "upgrade") cues.push(cueForUi(map, "cardSmith", rand));
    if (context === "transform") cues.push(cueForUi(map, "cardTransform", rand));
    if (context === "enchant") cues.push(cueForUi(map, "enchant", rand));
  }
  return cues.filter((c) => c !== null);
}

/**
 * The cues for a step the ENGINE took on the player's behalf — "play to end"
 * (`runModeStep`) and the watched autobattle leg (`autobattleStep`).
 *
 * Those loops never see the action they dispatched; the step comes back with
 * only its human-readable label (`debug.action`, built by the wasm's
 * `describe_action`: `"Play Strike"` or `"Play Strike → Nibbit"`). The card
 * name in it is read off the hand of the state the action was taken FROM, so
 * it joins `prev.hand` exactly — after dropping the `+` the label appends to
 * an upgraded card (`render.rs::card_name`), since `Card.name` never carries
 * it. Only card plays are voiced; any other label, a non-combat step, or a
 * name the hand doesn't hold yields nothing rather than a guess.
 */
export function steppedActionCues(
  map: SfxMap,
  prev: { decision: string; state: CombatState } | null,
  label: string | undefined,
): SoundCue[] {
  if (!prev || prev.decision !== "combat_play" || !label) return [];
  const m = /^Play (.+?)(?: → .*)?$/.exec(label);
  if (!m) return [];
  const name = m[1].replace(/\+$/, "");
  const card = (prev.state.hand ?? []).find((c) => c.name === name);
  if (!card) return [];
  return cuesForCard(map, card.name, card.type, prev.state.player?.name);
}

/**
 * Neow's greeting, on ARRIVAL at her room.
 *
 * Gated on the event's own name rather than the decision, because
 * `event_choice` is every ancient/event room and only this one is Neow.
 * Compared case-insensitively and trimmed, exactly as `EventScreen.tsx:66`
 * does for the same field — one screen deciding it is Neow while the sound
 * layer disagrees would be worse than either answer alone.
 */
export function neowArrivalCue(
  map: SfxMap,
  prevEventName: string | null,
  decision: string,
  eventName: string | undefined,
): SoundCue | null {
  if (decision !== "event_choice") return null;
  const name = (eventName ?? "").trim().toLowerCase();
  if (name !== "neow" || prevEventName === name) return null;
  return cueForUi(map, "neowWelcome");
}
