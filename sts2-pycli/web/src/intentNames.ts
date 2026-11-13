/** What each intent type is called and what it means, for the phone's inspect
 *  sheet and the intent icon's tooltip, which used to show the engine's raw
 *  enum id ("DebuffStrong").
 *
 *  The text is the game's own hover tip: `AbstractIntent.GetHoverTip` reads
 *  `intents.json` `<PREFIX>.title` / `<PREFIX>.description` (localization_eng
 *  and the v0.111.0 dump agree), with the BBCode stripped and the placeholders
 *  filled from what the wire carries. Three kinds share the title "Strategic"
 *  in the game; the description is what tells them apart. `Hidden` has no tip
 *  in the game (`HasIntentTip => false`), so its text is ours. */

/** `MegaCrit.Sts2.Core.MonsterMoves.Intents.IntentType`, v0.107.1. */
export const INTENT_TYPES = [
  "Attack", "Buff", "Debuff", "DebuffStrong", "Defend", "Escape", "Heal", "Hidden",
  "Summon", "Sleep", "Stun", "StatusCard", "CardDebuff", "DeathBlow", "Unknown",
] as const;

type Entry = { name: string; description: string | ((d?: number, hits?: number) => string) };

const forDamage = (d?: number, hits?: number) =>
  d == null ? "" : ` for ${d} damage${hits != null && hits > 1 ? ` ${hits} times` : ""}`;

const TABLE: Record<(typeof INTENT_TYPES)[number], Entry> = {
  Attack: { name: "Aggressive", description: (d, h) => `This enemy intends to Attack${forDamage(d, h)}.` },
  Buff: { name: "Empower", description: "This enemy intends to use a Buff." },
  Debuff: { name: "Strategic", description: "This enemy intends to apply a Debuff to you." },
  DebuffStrong: { name: "Strategic", description: "This enemy intends to apply a powerful Debuff to you." },
  Defend: { name: "Defensive", description: "This enemy intends to Block on its turn." },
  Escape: { name: "Cowardly", description: "This enemy intends to Escape." },
  Heal: { name: "Heal", description: "This enemy intends to Heal." },
  Hidden: { name: "Hidden", description: "This enemy's intention is hidden." },
  Summon: { name: "Summon", description: "This enemy intends to summon Monsters." },
  Sleep: { name: "Sleeping", description: "This enemy is doing nothing this turn." },
  Stun: { name: "Stunned", description: "This enemy can't act on its next turn." },
  // The game says how many ({CardCount}); the wire does not carry it.
  StatusCard: { name: "Strategic", description: "This enemy intends to give you Status cards." },
  CardDebuff: { name: "Malicious", description: "This enemy intends to apply Afflictions on your cards." },
  DeathBlow: { name: "Death Blow", description: (d) => d == null
    ? "This creature is trying to take you down with it before being destroyed."
    : `This creature is trying to take you down with it. It will attack you for ${d} damage before being destroyed.` },
  Unknown: { name: "Unknown", description: "This enemy's intentions are Unknown." },
};

const known = (type: string): type is keyof typeof TABLE => Object.prototype.hasOwnProperty.call(TABLE, type);

/** "DebuffStrong" -> "Debuff Strong": an id with no entry still reads as words. */
function spaced(id: string): string {
  return id.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/([A-Z])([A-Z][a-z])/g, "$1 $2");
}

export function intentName(type: string): string {
  return known(type) ? TABLE[type].name : spaced(type);
}

/** Name and one-sentence description of an intent; `description` is absent
 *  for a type this table does not know. */
export function intentInfo(intent: { type: string; damage?: number; hits?: number }):
  { name: string; description: string | undefined } {
  if (!known(intent.type)) return { name: spaced(intent.type), description: undefined };
  const e = TABLE[intent.type];
  return { name: e.name,
    description: typeof e.description === "string" ? e.description : e.description(intent.damage, intent.hits) };
}
