import type { Card } from "../api";

// The combatant an effect targets. Enemies are addressed by their stable index
// (CombatState.enemies[].index), which survives across snapshots.
export type TargetRef =
  | { side: "player" }
  | { side: "osty" }
  | { side: "enemy"; index: number };

/** The stable identity of the card or monster move an event came from — the
 *  join key into the VFX binding table (`effects/bindings.ts`), never the
 *  prettified display name.
 *
 *  A move is identified by `{ monster, id }`, NOT by `id` alone: engine move
 *  ids are only unique per monster (e.g. `THRASH` exists on both `Crusher`
 *  and `BowlbugNectar`), and the binding table is keyed
 *  `PascalCase-monster : SCREAMING_SNAKE-move`. Dropping the monster would
 *  make the lookup ambiguous and risk playing the wrong effect.
 *
 *  `id` is the RAW engine id, unreconciled against the table's `_MOVE`-suffix
 *  convention — that bidirectional join (try `id + "_MOVE"`, then bare `id`)
 *  is the binding lookup's job (Task 9), not this stream's.
 *
 *  `undefined` on diff-derived events — the snapshot diff cannot know a
 *  source, and inventing one would put wrong rows into the binding lookup. */
export type EffectSource =
  | { kind: "card"; id: string }
  | { kind: "move"; monster: string; id: string };

// One observable change between two combat snapshots.
export type EffectEvent =
  | { kind: "hurt"; target: TargetRef; amount: number; source?: EffectSource }
  // hp rising for a target present in both snapshots. This event exists for
  // SOUND ONLY (cuesForCombat maps it to the "heal" generic key). Board.tsx
  // dispatches on `e.kind` through an if-chain that does not name "heal" — and
  // for that reason SKIPS it before the `fx` write, rather than falling
  // through it. "Names no branch" is not the same as "is ignored": an entry
  // that sets no flag still bumps the shared per-target nonce, which replays
  // the enemy's hurt clip and drops the flags a co-occurring event set. That
  // was live for as long as this comment claimed otherwise. Heals are still
  // intentionally not modeled VISUALLY in v1; do not read this member as
  // licence to add a heal animation without a separate decision to do so.
  | { kind: "heal"; target: TargetRef; amount: number }
  | { kind: "blockGain"; target: TargetRef; amount: number; source?: EffectSource }
  | { kind: "blockBreak"; target: TargetRef }
  // A hit landed and block ate ALL of it. This is the one and only thing the
  // game plays `block_hit` for (CreatureCmd.cs's `WasFullyBlocked` arm); a hit
  // that gets THROUGH block plays the target's material impact instead. The
  // SAME call site also spawns `NBlockSparkVfx` and `NDamageBlockedVfx` — so
  // this is not sound-only: `effectsForEvent` plays `vfx_block_spark_vfx` off
  // it (`bindings.ts`) and `Board.tsx`'s `Fx.blocked` drives the "Blocked"
  // text (Task 12). This comment used to say "sound only, like `heal`"; that
  // was already wrong the moment Task 12 landed the text and stayed
  // uncorrected — see CLAUDE.md's rule about leaving a stale justification
  // beside changed code.
  | { kind: "fullyBlocked"; target: TargetRef; source?: EffectSource }
  /** `amount` is the power's TOTAL stack count at the moment the event fires —
   *  `power.Amount`, the value the game hands `GetTypeForAmount` to pick the
   *  buff/debuff rendering (`NCreature.cs:869-885`, `PowerModel.cs:462-472`).
   *  Deliberately NOT the delta that was just applied: the engine log's own
   *  `amount` for a `power` row IS the delta ("stacks applied, post-modify
   *  chain" — `event_log.rs`), and using it would call a -1 Strength on a
   *  creature already at +3 a debuff. Optional because a producer that cannot
   *  resolve a total must omit it rather than substitute the delta;
   *  `powerTypeForAmount` reads an absent amount as "classify by name", which
   *  is what shipped before this field existed. */
  | { kind: "powerAdd"; target: TargetRef; power: string; amount?: number }
  | { kind: "powerRemove"; target: TargetRef; power: string; amount?: number }
  /** A power TRIGGERING, as opposed to being applied — C#
   *  `NCreature.OnPowerFlashed`. Carries no amount: nothing changed, the
   *  power just did its job. */
  | { kind: "powerFlash"; target: TargetRef; power: string }
  | { kind: "death"; target: TargetRef }
  // The ACTOR's lunge, from a `monster_move` row or a card play. Distinct
  // from `hurt`, which is about the receiver: an attack that is fully blocked
  // still lunges. Supersedes the intent-sniffing guess in `enemyTurn.ts`
  // whenever the engine log is present and names the attacker; `Board.tsx`
  // still falls back to that guess (synthesizing this same event kind) for a
  // recording with no combat log at all, or a log row that can't resolve to
  // a board slot.
  | { kind: "attack"; target: TargetRef; source?: EffectSource }
  /** A card the ENGINE played on its own — Stampede at end of turn, Havoc,
   *  Mayhem, Cascade, … (a `card_auto_played` log row). The game shows such a
   *  card fly to the play area and resolve; the board flashes it the same way,
   *  since nothing else on screen would say it was played (GH #1155). `target`
   *  is always the player: the card is theirs. `card` is the hand card as it
   *  was drawn before the play, or a stub built from the log row when the card
   *  was not in hand (a draw-pile Havoc / Mayhem pull). */
  | { kind: "cardPlay"; target: TargetRef; card: Card };

export function sameTarget(a: TargetRef, b: TargetRef): boolean {
  if (a.side !== b.side) return false;
  if (a.side === "enemy" && b.side === "enemy") return a.index === b.index;
  return true;
}
