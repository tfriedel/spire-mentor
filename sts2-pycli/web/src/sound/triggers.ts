import type { CombatState } from "../api";
import type { EffectEvent } from "../effects/types";
import { resolveMonsterAttack, resolveMonsterMove, type SfxMap } from "./sfxMap";
import { powerValenceLookup, type PowerValence } from "./powerValence";
import { cueForUi, cuesForCard } from "./runCues";
import type { CueClass, SoundCue } from "./types";

/** What resolving one `EffectEvent`'s sounds needs — see `cuesForEvent`. */
export interface CombatEventCueContext {
  map: SfxMap;
  enemyName: (index: number) => string | undefined;
  enemyIntent: (index: number) => { type: string; damage?: number; hits?: number } | undefined;
  /** The run's character CLASS name (`state.player.name`), for the player's
   *  own death cry and an auto-played card's grunt. Optional: a caller that
   *  doesn't know it gets silence there, exactly as before this cue existed. */
  character?: string;
}

export interface CombatCueContext extends CombatEventCueContext {
  /** Enemy indices that struck this transition (from `effects/enemyTurn.ts::attackingEnemies`). */
  attackers: number[];
  /** The two snapshots this transition spans, for the cues that are a
   *  COUNT change rather than an `EffectEvent` — turn banners, card deal,
   *  exhaust, orbs. `combatDiff` deliberately does not model these: they have
   *  no visual in Board.tsx, and adding target-less members to `EffectEvent`
   *  for sound alone would push sound-only concepts into the visual layer.
   *  Optional so an existing caller keeps working unchanged. */
  prev?: CombatState;
  next?: CombatState;
  /** Uniform [0,1), injected for the randomised families (see `cueForUi`). */
  rand?: () => number;
}

/**
 * The cues that come from a COUNT moving, not from an `EffectEvent`.
 *
 * All four are things the game voices through a banner or a per-card vfx that
 * the workbench renders differently or not at all, so there is nothing in
 * `combatDiff` to hang them on:
 *
 * - the two turn banners, once per round boundary (`NPlayerTurnBanner.cs:112`,
 *   `NEnemyTurnBanner.cs:84`). One workbench transition spans the end of your
 *   turn, the whole enemy turn and your next draw, so a round increment means
 *   BOTH banners fired, in that order.
 * - `card_deal` when the hand grew (`CardPileCmd.cs:875`, per card at 0.25
 *   volume — one cue here, since the player debounces per event anyway).
 * - `card_exhaust` when the exhaust pile grew (`NExhaustVfx.cs:102`).
 *
 * A missing `prev`/`next` yields nothing rather than guessing, and so does the
 * first snapshot of a combat (no `prev` to diff).
 */
function rhythmCues(ctx: Pick<CombatCueContext, "map" | "prev" | "next" | "rand">): SoundCue[] {
  const { prev, next, map } = ctx;
  if (!prev || !next) return [];
  // Direct `cueForUi(...)` calls, not a kinds array — see the same note in
  // `appSoundCues.arrivalCues`: the coverage sweep looks for the kind literal
  // at a call site.
  const cues: (SoundCue | null)[] = [];
  if (next.round > prev.round) {
    cues.push(cueForUi(map, "enemyTurn", ctx.rand), cueForUi(map, "playerTurn", ctx.rand));
  }
  if ((next.hand?.length ?? 0) > (prev.hand?.length ?? 0)) {
    cues.push(cueForUi(map, "cardDeal", ctx.rand));
  }
  if ((next.exhaust_pile_count ?? 0) > (prev.exhaust_pile_count ?? 0)) {
    cues.push(cueForUi(map, "cardExhaust", ctx.rand));
  }
  // Necrobinder's Osty arriving (`OstyCmd.cs:47`). `osty` stays on the wire
  // once he exists — with `alive: false` when slain — so this keys on the
  // field APPEARING, not on him being alive, and a resurrect after a death
  // therefore does not re-summon. That matches `OstyCmd`, which plays this
  // only on the initial summon.
  if (!prev.osty && next.osty) cues.push(cueForUi(map, "summon", ctx.rand));
  return cues.filter((c) => c !== null);
}

/**
 * Channel / evoke / passive, from the orb queue changing.
 *
 * The wire sends the whole queue each snapshot, so a type present in `next`
 * beyond its `prev` count was CHANNELLED and one short of it was EVOKED —
 * matching by type rather than by list position, since evoking the front orb
 * shifts every index behind it. Counting by type also gets a same-turn
 * channel-then-evoke of two different orbs right, which a length comparison
 * could not see at all.
 *
 * `passive` fires on the round boundary for every orb still in the queue —
 * `OrbModel.PlayPassiveSfx` is called from each orb's own turn trigger, and
 * only Frost and Glass ever call it (see `load_orb_sfx`).
 */
function orbCues(ctx: Pick<CombatCueContext, "map" | "prev" | "next">): SoundCue[] {
  const { prev, next, map } = ctx;
  if (!prev || !next || !map.orbs) return [];
  const count = (orbs: CombatState["orbs"]) => {
    const by = new Map<string, number>();
    for (const o of orbs ?? []) by.set(o.type, (by.get(o.type) ?? 0) + 1);
    return by;
  };
  const before = count(prev.orbs);
  const after = count(next.orbs);
  const cues: SoundCue[] = [];
  const add = (event: string | undefined) => {
    if (event) cues.push({ event, cls: "informative" });
  };
  for (const type of new Set([...before.keys(), ...after.keys()])) {
    const delta = (after.get(type) ?? 0) - (before.get(type) ?? 0);
    if (delta > 0) add(map.orbs[type]?.channel);
    else if (delta < 0) add(map.orbs[type]?.evoke);
  }
  if (next.round > prev.round) {
    for (const type of after.keys()) add(map.orbs[type]?.passive);
  }
  return cues;
}

// One transition can carry six `hurt` events (an AoE) or two `blockBreak`s.
// Playing each would machine-gun the mix, so each generic kind fires at most
// once per transition — the player's debounce is a second line of defence,
// not the first.
//
// `hurt` is NOT in here. It used to map to `blockHit`, which was simply the
// wrong sound: `CreatureCmd.cs` plays `block_hit` only on its `WasFullyBlocked`
// arm, i.e. when block ate the entire hit. A hit that lands plays the target's
// own material impact (`SfxCmd.PlayDamage` -> `MonsterModel.TakeDamageSfx`),
// which is per-monster and so is resolved below rather than through this table.
//
// `fullyBlocked` is ALSO not in here (moved out below `cuesForCombat`'s
// attacker loop) — it needs to be checked ahead of the enemy-turn/no-log
// fallback heuristic that shares its `block_hit` key, not folded into a
// same-pass table lookup that can't see that ordering.
const GENERIC: Record<string, { key: string; cls: CueClass }> = {
  heal: { key: "heal", cls: "informative" },
  blockGain: { key: "blockGain", cls: "informative" },
  blockBreak: { key: "blockBreak", cls: "informative" },
};

/**
 * Which generic sound a power change should play, given the valence
 * `powerValence.ts` resolved for this power AT THIS AMOUNT — the power's own
 * `Type` from sfx_map.json's `powers` table, flipped where the game's
 * `GetTypeForAmount` flips it (a negative Strength is a debuff). The caller
 * passes `valenceOf(e.power, e.amount)`, and `Board.tsx`'s panel colour reads
 * the same function, so the chime and the colour cannot disagree.
 *
 * `powerAdd` plays the valence straight: gaining a Debuff (e.g. Vulnerable)
 * sounds bad, gaining a Buff (e.g. Strength) sounds good. `powerRemove`
 * INVERTS it — losing a Debuff is good news, losing a Buff is bad, so it
 * plays the opposite chime from an add of the same power.
 *
 * A power absent from the table (not yet classified for this game version,
 * or a future update added one before the map was regenerated) still MUST
 * make a sound rather than going silent — `valence` is `undefined` for it,
 * and this falls back to the same fixed direction the file used before
 * per-power valence existed: gain -> buff chime, loss -> debuff chime.
 */
function powerCueKey(
  kind: "powerAdd" | "powerRemove",
  valence: PowerValence | undefined,
): "buff" | "debuff" {
  if (valence === undefined) return kind === "powerAdd" ? "buff" : "debuff";
  const isDebuff = kind === "powerAdd" ? valence === "Debuff" : valence === "Buff";
  return isDebuff ? "debuff" : "buff";
}

/**
 * The cues that belong to the TRANSITION as a whole rather than to any one
 * effect event — the turn banners, the card deal and exhaust, Osty arriving,
 * the orb queue (`rhythmCues` / `orbCues`) — deduplicated, banners first.
 *
 * They lead because they announce what is about to happen. `Board.tsx` plays
 * these at once when a transition lands, then each event's own cues
 * (`cuesForEvent`) as the scheduler reaches that event.
 */
export function transitionCues(ctx: Pick<CombatCueContext, "map" | "prev" | "next" | "rand">): SoundCue[] {
  const seen = new Set<string>();
  return [...rhythmCues(ctx), ...orbCues(ctx)].filter((cue) => {
    if (seen.has(cue.event)) return false;
    seen.add(cue.event);
    return true;
  });
}

/**
 * Every sound ONE effect event makes, in the order the game layers them. Pure.
 *
 * This is the unit `Board.tsx` plays per hit in `full` fx mode — the scheduler
 * fires a 3-hit attack as three `hurt` events 180 ms apart, so the body's
 * impact now sounds three times, as `SfxCmd.PlayDamage` does in the game,
 * instead of once for the whole transition. `cuesForCombat` is the same
 * resolution folded into one deduplicated beat.
 *
 * Within an event the cues are distinct; ACROSS events nothing is deduplicated
 * here (the player's per-event debounce collapses a same-instant repeat).
 */
export function cuesForEvent(e: EffectEvent, ctx: CombatEventCueContext): SoundCue[] {
  const out: SoundCue[] = [];
  const add = (event: string | undefined, cls: CueClass = "informative") => {
    if (event && !out.some((c) => c.event === event)) out.push({ event, cls });
  };
  switch (e.kind) {
    case "hurt": {
      // The sound of an attack LANDING, and it belongs to the body it lands on:
      // armor, fur, slime, stone… A monster with no entry of its own falls back
      // to `generic.impact`, which is MonsterModel's own `Armor` default.
      //
      // Enemy-only, exactly as in the decomp: `CreatureCmd.cs` guards the
      // `PlayDamage` call with `Side == CombatSide.Enemy`. A player taking a hit
      // is voiced by the attacker's own attack cue — giving the player an
      // impact too would double every incoming hit.
      if (e.target.side !== "enemy") break;
      const monster = ctx.map.monsters[ctx.enemyName(e.target.index) ?? ""];
      // The creature's VOICE, before its body's impact and in ADDITION to it —
      // `CreatureCmd.cs:326-328` plays `Monster.HurtSfx` alongside the "Hit"
      // animation trigger, then `:352` plays the material impact. Only 14 of
      // the 97 monsters declare one (`MonsterModel.HurtSfx` is virtual `null`).
      add(monster?.hurt);
      add(monster?.impact ?? ctx.map.generic.impact);
      break;
    }
    case "death": {
      // Every monster fades out with the same sound on top of whatever its own
      // class names — `NMonsterDeathVfx.cs:227` plays `enemy_fade` for all of
      // them, and only 91 of the 97 declare a `DeathSfx` at all.
      //
      // The player dies to their own character's cry (`SfxCmd.PlayDeath(player)`
      // -> `player.Character.DeathSfx`, SfxCmd.cs:98). The game ALSO starts
      // `event:/temp/sfx/game_over` here, but as MUSIC on a different bus
      // (`CreatureCmd.cs:460`) — `runEndCue` emits that one, so they do not
      // duplicate. A monster whose class declares no `DeathSfx` stays silent
      // rather than borrowing another monster's.
      if (e.target.side === "enemy") {
        add(ctx.map.generic.enemyFade);
        add(ctx.map.monsters[ctx.enemyName(e.target.index) ?? ""]?.death);
      } else if (e.target.side === "player") {
        add(ctx.map.characters?.[ctx.character ?? ""]?.die);
      }
      break;
    }
    case "powerAdd":
    case "powerRemove":
      // Both kinds are `informative`: Ruling 47 (final review finding 8,
      // 2026-08-23) once left `powerRemove` as `chatter` because a removal had
      // NO visual in any mode; Task 12 gave it one (icon + name + "Wears Off",
      // `power_removed_vfx`), so `chatter` would now make sound quieter than
      // what is on screen.
      add(ctx.map.generic[powerCueKey(e.kind, powerValenceLookup(ctx.map)(e.power, e.amount))]);
      break;
    case "fullyBlocked":
      // `CreatureCmd.cs:400-406`: creature-agnostic, on `WasFullyBlocked`
      // alone — the same call site as the "Blocked" text and the block spark.
      add(ctx.map.generic.blockHit);
      break;
    case "attack": {
      // A monster performing a move (a `monster_move` row, or the Board's
      // intent-guess for a recording with no log). The PLAYER's own swing is
      // voiced by the card it came from (`cuesForCard`), not here.
      if (e.target.side !== "enemy") break;
      const index = e.target.index;
      const moveId = e.source?.kind === "move" ? e.source.id : undefined;
      add(resolveMonsterMove(
        ctx.map.monsters[ctx.enemyName(index) ?? ""],
        moveId,
        ctx.enemyIntent(index),
        ctx.map.generic.blockHit,
      ));
      break;
    }
    case "cardPlay":
      // A card the ENGINE played (Havoc, Mayhem, Stampede, … — GH #1155). It
      // makes the same sounds as one the player clicked; the click path voices
      // those from `actionCues`, which never sees an auto-play.
      for (const cue of cuesForCard(ctx.map, e.card.name, e.card.type, ctx.character)) add(cue.event, cue.cls);
      break;
    default: {
      const spec = GENERIC[e.kind];
      if (spec) add(ctx.map.generic[spec.key], spec.cls);
    }
  }
  return out;
}

/**
 * Derive the sounds one combat transition makes when it all lands in ONE
 * beat — every fx mode but `full`, where the board settles instantly — with
 * each sound at most once (one transition can carry six `hurt` events from an
 * AoE; playing each would machine-gun the mix). Pure — no WebAudio, no React.
 *
 * `ctx.attackers` is the intent-sniffing guess at who struck
 * (`effects/enemyTurn.ts::attackingEnemies`), for a transition whose events
 * carry no `attack` of their own (no combat log); a caller with the log
 * passes `[]` and the `attack` events voice the exact moves instead. This
 * does not re-filter by `enemyIntent(...).type`, so an enemy listed as an
 * attacker still gets an attack cue even if its intent reads `DeathBlow`
 * rather than `Attack` (an exploding bomb) — `resolveMonsterAttack` restricts
 * hit-count matching to `Attack`-tagged moves and falls back to
 * `defaultAttack`/the generic hit sound, so this never fabricates a move.
 */
export function cuesForCombat(events: EffectEvent[], ctx: CombatCueContext): SoundCue[] {
  const cues: SoundCue[] = [];
  const seen = new Set<string>();
  const push = (cue: SoundCue) => {
    if (seen.has(cue.event)) return;
    seen.add(cue.event);
    cues.push(cue);
  };

  for (const cue of transitionCues(ctx)) push(cue);

  for (const index of ctx.attackers) {
    const monster = ctx.map.monsters[ctx.enemyName(index) ?? ""];
    const event = resolveMonsterAttack(monster, ctx.enemyIntent(index), ctx.map.generic.blockHit);
    if (event) push({ event, cls: "informative" });
  }

  // A direct `fullyBlocked` event — from `combatDiff` (gated `sameRound`, see
  // combatDiff.ts) or a per-hit log row — wins regardless of what the
  // fallback below would have concluded, so it is checked ahead of it.
  const blockedEvent = events.some((e) => e.kind === "fullyBlocked");
  if (blockedEvent && ctx.map.generic.blockHit) push({ event: ctx.map.generic.blockHit, cls: "informative" });

  // The one case the direct event can't cover: the ENEMY TURN's own
  // fully-blocked hit on the player. `combatDiff` only emits `fullyBlocked`
  // when `sameRound` — block also zeroes at end of turn, so a snapshot pair
  // spanning the enemy turn can't tell "absorbed a hit" from "expired" — and
  // a recording with no combat log carries no per-hit `fullyBlocked` row at
  // all. Inherently PLAYER-only: `ctx.attackers` only ever names enemies
  // striking the player.
  if (
    !blockedEvent &&
    ctx.attackers.length &&
    !events.some((e) => e.kind === "hurt" && e.target.side === "player") &&
    ctx.map.generic.blockHit
  ) {
    push({ event: ctx.map.generic.blockHit, cls: "informative" });
  }

  for (const e of events) {
    if (e.kind === "fullyBlocked") continue; // voiced above
    for (const cue of cuesForEvent(e, ctx)) push(cue);
  }
  return cues;
}

/** Apply the visual fx mode to sound, so the two never disagree. */
export function filterForMode(cues: SoundCue[], mode: "full" | "reduced" | "off"): SoundCue[] {
  if (mode === "off") return [];
  if (mode === "reduced") return cues.filter((c) => c.cls === "informative");
  return cues;
}
