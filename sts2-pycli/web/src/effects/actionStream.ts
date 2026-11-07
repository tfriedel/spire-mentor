import type { Card, CombatLogRow, CombatState } from "../api";
import { combatDiff } from "./combatDiff";
import type { EffectEvent, EffectSource, TargetRef } from "./types";
import { targetForUid } from "./uidTargets";

// Event kinds the engine log CANNOT produce — `event_log.rs`'s `EventLogEntry`
// has no heal and no power-removed entry — so these always come from the
// snapshot diff, regardless of what the log said.
const DIFF_ONLY: ReadonlySet<EffectEvent["kind"]> = new Set(["heal", "powerRemove"]);

function key(t: TargetRef): string {
  return t.side === "enemy" ? `e${t.index}` : t.side;
}

/**
 * A power's stack TOTAL on one target in one snapshot, or `undefined` when the
 * target or the power is not in it.
 *
 * The log row for a power application carries an `amount` of its own and it is
 * the WRONG number here: `event_log.rs`'s `PowerReceived.amount` is the stacks
 * that application added, post-modify-chain, while the game classifies the
 * rendering by `power.Amount`, the running total (`NCreature.cs:869-885`). For
 * a power at +3 taking a -1, those are -1 and +2 — opposite valences. The
 * snapshot is where the total lives, so that is where this reads it.
 */
function powerAmountIn(
  state: CombatState | null,
  target: TargetRef,
  power: string,
): number | undefined {
  if (!state) return undefined;
  const powers =
    target.side === "player"
      ? state.player_powers
      : target.side === "enemy"
        ? (state.enemies ?? []).find((e) => e.index === target.index)?.powers
        : undefined; // osty carries no powers on the wire (see `combatDiff`)
  return (powers ?? []).find((p) => p.name === power)?.amount;
}

// The stable card/move identity a row came from, for the VFX binding lookup
// (Task 9). Both branches are guarded on `source_kind` — not just field
// truthiness — so a future producer change can't misroute a card row into
// the move branch (or vice versa).
function sourceOf(r: CombatLogRow): EffectSource | undefined {
  if (r.source_kind === "card" && r.source_card) return { kind: "card", id: r.source_card };
  if (r.source_kind === "move" && r.source_move_id && r.source_monster) {
    return { kind: "move", monster: r.source_monster, id: r.source_move_id };
  }
  return undefined;
}

/**
 * The card an auto-play row names, as the board should draw it: the matching
 * card from the snapshot BEFORE the action (it has its art, cost and text) —
 * the hand first (Stampede), then the piles an auto-play can pull from
 * (Havoc / Mayhem take the draw pile's top, Howl from Beyond the exhaust
 * pile) — else a stub from the row itself.
 */
function autoPlayedCard(r: CombatLogRow, prev: CombatState | null): Card {
  const id = r.card_id ?? "";
  const upgraded = r.card_upgraded === true;
  const piles = [prev?.hand, prev?.draw_pile, prev?.discard_pile, prev?.exhaust_pile];
  for (const exact of [true, false]) {
    for (const pile of piles) {
      const hit = (pile ?? []).find(
        (c) => c.id === id && (!exact || (c.upgraded ?? false) === upgraded),
      );
      if (hit) return hit;
    }
  }
  return {
    index: -1,
    id,
    name: r.noun ?? id,
    cost: 0,
    type: "Attack",
    description: "",
    upgraded,
    image: null,
  };
}

/**
 * One action's ordered effect events.
 *
 * The engine's combat log is the primary source: it records damage PER HIT
 * with the block split, so a 3-hit attack yields three events rather than one
 * net HP change. `combatDiff` is demoted to a reconciler, supplying only what
 * the log cannot know.
 *
 * The reconciliation rule, in order:
 *   1. the log's events, in log order;
 *   2. every `heal` / `powerRemove` from the diff (the log has neither);
 *   3. the diff's events for any target the log said NOTHING about.
 *
 * Rule 3 is what keeps Osty working: it carries no uid on the board, so its
 * log rows do not resolve to a slot, and without this it would silently lose
 * the damage numbers it shows today. Restricting it to targets with zero log
 * events is what stops every ordinary enemy hit rendering twice.
 */
export function actionStream(
  rows: CombatLogRow[],
  prev: CombatState | null,
  next: CombatState,
): EffectEvent[] {
  const out: EffectEvent[] = [];
  const described = new Set<string>();
  // `[next, prev]`: an enemy killed by this action is already gone from
  // `next`, but Board still draws its slot through the `dying` overlay.
  const states = [next, prev];

  for (const r of rows) {
    const actor = targetForUid(r.actor_uid, r.actor_side, states);
    const target = targetForUid(r.target_uid, r.target_side, states);

    if (r.kind === "damage" && target) {
      const amount = r.amount ?? 0;
      const source = sourceOf(r);
      // Only mark the target "described" if a row actually produced an event
      // for it — don't lean on the invariant that amount/blocked is always
      // nonzero, which lives a layer down in Rust and isn't checked here.
      if (amount > 0) {
        out.push({ kind: "hurt", target, amount, source });
        described.add(key(target));
      } else if ((r.blocked ?? 0) > 0) {
        out.push({ kind: "fullyBlocked", target, source });
        described.add(key(target));
      }
      // `hp_after === 0` is the log's only death signal.
      if (r.hp_after === 0) {
        out.push({ kind: "death", target });
        described.add(key(target));
      }
      continue;
    }
    if (r.kind === "block" && actor) {
      described.add(key(actor));
      out.push({ kind: "blockGain", target: actor, amount: r.amount ?? 0, source: sourceOf(r) });
      continue;
    }
    if (r.kind === "power" && target) {
      described.add(key(target));
      const power = r.noun ?? "";
      out.push({ kind: "powerAdd", target, power, amount: powerAmountIn(next, target, power) });
      continue;
    }
    if (r.kind === "power_flash" && target) {
      // NOT `described.add`: a flash says nothing about damage or stacks, so
      // it must not suppress the diff's reconciliation for this target.
      out.push({ kind: "powerFlash", target, power: r.noun ?? "" });
      continue;
    }
    if (r.kind === "card_auto_played") {
      // NOT `described.add`: showing the card says nothing about the player's
      // HP or block, so the diff must still reconcile the player.
      out.push({ kind: "cardPlay", target: { side: "player" }, card: autoPlayedCard(r, prev) });
      continue;
    }
    if (r.kind === "monster_move" && actor) {
      described.add(key(actor));
      out.push({ kind: "attack", target: actor, source: sourceOf(r) });
      continue;
    }
  }

  for (const e of combatDiff(prev, next)) {
    if (DIFF_ONLY.has(e.kind) || !described.has(key(e.target))) out.push(e);
  }
  return out;
}
