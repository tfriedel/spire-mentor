import { describe, it, expect } from "vitest";
import {
  actionCues, arrivalCues, cardPlayCues, goldGainCue, neowArrivalCue, potionGainCue, relicGainCue, runEndCue,
  steppedActionCues,
} from "../appSoundCues";
import type { SfxMap } from "../sfxMap";
import type { Card } from "../../api";

const MAP: SfxMap = {
  version: "test",
  generic: {
    cardPlay: "event:/sfx/cards/play",
    cardPower: "event:/sfx/cards/power",
    gold: "event:/sfx/gold",
    defeat: "event:/sfx/defeat",
  },
  cards: {}, relics: {}, monsters: {}, powers: {},
};

function card(over: Partial<Card>): Card {
  return { index: 0, id: "x", name: "Strike", cost: 1, type: "Attack", description: "", image: null, ...over };
}

describe("cardPlayCues", () => {
  it("resolves the cues for the card at the given index", () => {
    const hand = [card({ index: 0, name: "Strike", type: "Attack" }), card({ index: 1, name: "Inflame", type: "Power" })];
    expect(cardPlayCues(MAP, hand, 1).map((c) => c.event)).toEqual(["event:/sfx/cards/power"]);
  });

  // Mutation caught: looking up by array position instead of by `.index` —
  // the hand array's position and a card's `.index` field are not the same
  // thing once cards have been played/drawn out of order.
  it("looks up by the card's `.index` field, not its array position", () => {
    const hand = [card({ index: 5, name: "Inflame", type: "Power" })];
    expect(cardPlayCues(MAP, hand, 5).map((c) => c.event)).toEqual(["event:/sfx/cards/power"]);
  });

  it("returns nothing when no card in hand has that index", () => {
    const hand = [card({ index: 0 })];
    expect(cardPlayCues(MAP, hand, 9)).toEqual([]);
  });

  // Mutation caught: dropping the `character` argument on the way through to
  // `cuesForCard`, which silently reverts the whole fix — every card would
  // play the swoosh alone again, exactly the bug this replaced.
  it("forwards the character through, so the player's own sound is added", () => {
    const m: SfxMap = {
      ...MAP,
      characters: { Ironclad: { attack: "event:/sfx/characters/ironclad/ironclad_attack" } },
      cardAnims: { Strike: { kind: "attack" } },
    };
    const hand = [card({ index: 0, name: "Strike", type: "Attack" })];
    expect(cardPlayCues(m, hand, 0, "Ironclad").map((c) => c.event))
      .toEqual(["event:/sfx/characters/ironclad/ironclad_attack", "event:/sfx/cards/play"]);
  });
});

describe("goldGainCue", () => {
  it("fires on a gold increase", () => {
    expect(goldGainCue(MAP, 10, 25)?.event).toBe("event:/sfx/gold");
  });

  // Mutation caught: passing `nextGold` (the new TOTAL) instead of the delta.
  // Both fixtures end at 420 gold; only the amount GAINED differs, so a
  // total-based implementation gives them the same sound.
  it("picks the tier from the amount GAINED, not the running total", () => {
    const m: SfxMap = {
      ...MAP,
      generic: {
        ...MAP.generic,
        goldSmall: "event:/sfx/gold_1",
        goldMedium: "event:/sfx/gold_2",
        goldLarge: "event:/sfx/gold_3",
      },
    };
    expect(goldGainCue(m, 400, 420)?.event).toBe("event:/sfx/gold_1");
    expect(goldGainCue(m, 320, 420)?.event).toBe("event:/sfx/gold_3");
  });

  // Mutation caught: comparing with `>=` instead of `>` (a no-op tick would
  // then fire the cue every time gold is merely re-observed unchanged).
  it("does not fire when gold is unchanged", () => {
    expect(goldGainCue(MAP, 25, 25)).toBeNull();
  });

  // Mutation caught: dropping the `<=` branch and firing on ANY delta
  // (including a shop purchase, which decreases gold).
  it("does not fire on a decrease (a shop purchase)", () => {
    expect(goldGainCue(MAP, 25, 10)).toBeNull();
  });

  it("does not fire on the first observation (nothing to diff against)", () => {
    expect(goldGainCue(MAP, null, 99)).toBeNull();
  });
});

describe("runEndCue", () => {
  it("resolves the defeat cue at game_over when victory is false", () => {
    expect(runEndCue(MAP, "game_over", false)?.event).toBe("event:/sfx/defeat");
  });

  // Mutation caught: swapping the ternary (`victory ? "defeat" : "victory"`)
  // — since the map has no `victory` generic key, this null-vs-event
  // difference is the only way to tell the branches apart.
  it("resolves to null (not defeat) for an actual victory — no victory event ships", () => {
    expect(runEndCue(MAP, "game_over", true)).toBeNull();
  });

  // Mutation caught: dropping the `decision !== "game_over"` gate, which
  // would fire a defeat/victory cue on every other decision too.
  it("does not fire outside the game_over decision", () => {
    expect(runEndCue(MAP, "map_select", false)).toBeNull();
  });
});

describe("arrivalCues", () => {
  const M: SfxMap = {
    ...MAP,
    generic: {
      ...MAP.generic,
      victory: "res://debug_audio/victory.mp3",
      battleStart1: "res://debug_audio/battle_start_1.mp3",
      battleStart2: "res://debug_audio/battle_start_2.mp3",
      potionSlosh1: "res://debug_audio/potion_slosh_1.mp3",
      merchantWelcome: "event:/sfx/npcs/merchant/merchant_welcome",
    },
  };
  const events = (cues: { event: string }[]) => cues.map((c) => c.event);

  it("plays the victory sting on arrival at the rewards screen", () => {
    expect(events(arrivalCues(M, "combat_play", "card_reward"))).toEqual(["res://debug_audio/victory.mp3"]);
  });

  it("plays a battle-start sting on arrival in combat", () => {
    expect(events(arrivalCues(M, "map_select", "combat_play", { rand: () => 0 })))
      .toEqual(["res://debug_audio/battle_start_1.mp3"]);
  });

  // `Rng.Chaotic.NextItem(TmpSfx.BattleStart)` picks between exactly two.
  it("picks the other battle-start variant at the top of the range", () => {
    expect(events(arrivalCues(M, "map_select", "combat_play", { rand: () => 0.99 })))
      .toEqual(["res://debug_audio/battle_start_2.mp3"]);
  });

  // `NPotionHolder.ShineOnStartOfCombat` — only when there is a potion to shine.
  it("adds a potion slosh when the player is holding one", () => {
    expect(events(arrivalCues(M, "map_select", "combat_play", { holdsPotion: true, rand: () => 0 })))
      .toEqual(["res://debug_audio/battle_start_1.mp3", "res://debug_audio/potion_slosh_1.mp3"]);
  });

  it("plays the merchant's welcome on arrival at the shop", () => {
    expect(events(arrivalCues(M, "map_select", "shop"))).toEqual(["event:/sfx/npcs/merchant/merchant_welcome"]);
  });

  // Mutation caught: reading the current decision instead of the transition.
  // A background poll re-delivering the same step would re-announce it.
  it("does not re-fire while sitting on the same screen", () => {
    expect(arrivalCues(M, "card_reward", "card_reward")).toEqual([]);
  });

  it("stays silent on the first observation of a run", () => {
    expect(arrivalCues(M, null, "card_reward")).toEqual([]);
  });

  it("stays silent on a decision none of them voices", () => {
    expect(arrivalCues(M, "combat_play", "map_select")).toEqual([]);
  });

  // A map generated before the keys existed. The old comment in runCues.ts
  // asserted a win was "silent by design"; it must degrade to that, not throw.
  it("stays silent when the map has none of the keys", () => {
    expect(arrivalCues(MAP, "combat_play", "card_reward")).toEqual([]);
  });
});

describe("potionGainCue", () => {
  const M: SfxMap = { ...MAP, generic: { ...MAP.generic, potionGain: "res://debug_audio/gain_potion.mp3" } };

  it("fires when the belt grows", () => {
    expect(potionGainCue(M, 0, 1)?.event).toBe("res://debug_audio/gain_potion.mp3");
  });

  // Drinking one must not sound like picking one up.
  it("stays silent when the belt shrinks or holds", () => {
    expect(potionGainCue(M, 2, 1)).toBeNull();
    expect(potionGainCue(M, 2, 2)).toBeNull();
  });

  it("stays silent on the first observation of a run", () => {
    expect(potionGainCue(M, null, 2)).toBeNull();
  });
});

describe("relicGainCue", () => {
  const M: SfxMap = { ...MAP, generic: { ...MAP.generic, relicGet: "res://debug_audio/relic_get.mp3" } };

  it("fires when the inventory grows", () => {
    expect(relicGainCue(M, 1, 2)?.event).toBe("res://debug_audio/relic_get.mp3");
  });

  it("stays silent on the first observation of a run", () => {
    expect(relicGainCue(M, null, 3)).toBeNull();
  });

  // Mutation caught: `!==` instead of `<=`. Relics can LEAVE the inventory
  // (Ectoplasm-style swaps, an event that takes one), and losing one must not
  // sound like gaining one.
  it("stays silent when the count falls or holds", () => {
    expect(relicGainCue(M, 3, 2)).toBeNull();
    expect(relicGainCue(M, 3, 3)).toBeNull();
  });

  it("fires once for several relics arriving together", () => {
    expect(relicGainCue(M, 0, 3)?.event).toBe("res://debug_audio/relic_get.mp3");
  });
});

describe("neowArrivalCue", () => {
  const M: SfxMap = { ...MAP, generic: { ...MAP.generic, neowWelcome: "event:/sfx/npcs/neow/neow_welcome" } };

  it("fires on arrival in Neow's room", () => {
    expect(neowArrivalCue(M, null, "event_choice", "neow")?.event).toBe("event:/sfx/npcs/neow/neow_welcome");
  });

  // Matches EventScreen.tsx:66, which decides the same question the same way.
  // The two disagreeing would draw Neow's room in silence, or greet her in
  // someone else's.
  it("normalises the event name the way the screen does", () => {
    expect(neowArrivalCue(M, null, "event_choice", "  Neow ")?.event).toBe("event:/sfx/npcs/neow/neow_welcome");
  });

  it("does not re-fire while sitting in her room", () => {
    expect(neowArrivalCue(M, "neow", "event_choice", "neow")).toBeNull();
  });

  // Mutation caught: gating on the decision alone. `event_choice` is EVERY
  // event room, so that would greet you as Neow at the Dead Adventurer.
  it("stays silent in any other event room", () => {
    expect(neowArrivalCue(M, null, "event_choice", "Golden Idol")).toBeNull();
  });

  it("stays silent on a decision that is not an event", () => {
    expect(neowArrivalCue(M, null, "card_reward", "neow")).toBeNull();
  });
});

describe("actionCues", () => {
  const M: SfxMap = {
    ...MAP,
    generic: {
      ...MAP.generic,
      merchantThanks: "event:/sfx/npcs/merchant/merchant_thank_yous",
      restSleep1: "res://debug_audio/STS_SleepJingle_1a_NewMix_v1.mp3",
      restBlanket: "res://debug_audio/SOTE_SFX_SleepBlanket_v1.mp3",
      restDig: "res://debug_audio/sts_sfx_shovel_v1.mp3",
      cardSmith: "res://debug_audio/card_smith.mp3",
      cardTransform: "event:/sfx/ui/cards/card_transform",
      enchant: "event:/sfx/ui/enchant_shimmer",
    },
  };
  const events = (cues: { event: string }[]) => cues.map((c) => c.event);
  const state = (over: Record<string, unknown> = {}) => ({ ...over } as never);

  it("thanks you for any purchase", () => {
    expect(events(actionCues(M, state(), "shop", "buy_relic", { relic_index: 0 })))
      .toEqual(["event:/sfx/npcs/merchant/merchant_thank_yous"]);
    expect(events(actionCues(M, state(), "shop", "buy_card", { card_index: 1 })))
      .toEqual(["event:/sfx/npcs/merchant/merchant_thank_yous"]);
  });

  it("says nothing when you just leave the shop", () => {
    expect(actionCues(M, state(), "shop", "leave_room", {})).toEqual([]);
  });

  // `HealRestSiteOption.cs:98-99` plays BOTH, in this order.
  it("plays the jingle and the blanket for a Rest", () => {
    const s = state({ rest_options: [{ index: 0, option_id: "HEAL" }, { index: 1, option_id: "DIG" }] });
    expect(events(actionCues(M, s, "rest_site", "choose_option", { option_index: 0 })))
      .toEqual([
        "res://debug_audio/STS_SleepJingle_1a_NewMix_v1.mp3",
        "res://debug_audio/SOTE_SFX_SleepBlanket_v1.mp3",
      ]);
  });

  // Mutation caught: keying on the option INDEX rather than looking its id up.
  // The rest-site menu is not a fixed order, so index 1 is not always Dig.
  it("plays the shovel for a Dig, resolved through the option id", () => {
    const s = state({ rest_options: [{ index: 0, option_id: "DIG" }, { index: 1, option_id: "HEAL" }] });
    expect(events(actionCues(M, s, "rest_site", "choose_option", { option_index: 0 })))
      .toEqual(["res://debug_audio/sts_sfx_shovel_v1.mp3"]);
  });

  it("says nothing for a rest-site option with no sound of its own", () => {
    const s = state({ rest_options: [{ index: 0, option_id: "COOK" }] });
    expect(actionCues(M, s, "rest_site", "choose_option", { option_index: 0 })).toEqual([]);
  });

  it("picks the deck-op sound from the card_select context", () => {
    expect(events(actionCues(M, state({ context: "upgrade" }), "card_select", "select_card", {})))
      .toEqual(["res://debug_audio/card_smith.mp3"]);
    expect(events(actionCues(M, state({ context: "transform" }), "card_select", "select_card", {})))
      .toEqual(["event:/sfx/ui/cards/card_transform"]);
    expect(events(actionCues(M, state({ context: "enchant" }), "card_select", "select_card", {})))
      .toEqual(["event:/sfx/ui/enchant_shimmer"]);
  });

  // An in-combat pick wears the same `card_select` decision as a deck op —
  // `choiceKinds.ts` exists precisely because they are not the same thing.
  it("says nothing for a card_select with no deck-op context", () => {
    expect(actionCues(M, state(), "card_select", "select_card", {})).toEqual([]);
  });
});

describe("actionCues — playing a card", () => {
  const hand = [card({ index: 0, name: "Strike", type: "Attack" }), card({ index: 1, name: "Inflame", type: "Power" })];
  const state = { hand, player: { name: "Ironclad" } } as never;
  const m: SfxMap = {
    ...MAP,
    characters: { Ironclad: { attack: "event:/sfx/characters/ironclad/ironclad_attack" } },
    cardAnims: { Strike: { kind: "attack" } },
  };

  // The regression this pins: card sounds came only from the Board's click
  // handler, so "play recommended" and the A-key autoplay played silently.
  // Mutation caught: deleting the `play_card` branch.
  it("voices the card at `card_index` in the hand it is played from", () => {
    expect(actionCues(m, state, "combat_play", "play_card", { card_index: 0, target_index: 0 }).map((c) => c.event))
      .toEqual(["event:/sfx/characters/ironclad/ironclad_attack", "event:/sfx/cards/play"]);
    expect(actionCues(m, state, "combat_play", "play_card", { card_index: 1 }).map((c) => c.event))
      .toEqual(["event:/sfx/cards/power"]);
  });

  it("says nothing for an index the hand doesn't hold", () => {
    expect(actionCues(m, state, "combat_play", "play_card", { card_index: 7 })).toEqual([]);
  });
});

describe("steppedActionCues", () => {
  const hand = [
    card({ index: 0, name: "Strike", type: "Attack" }),
    card({ index: 1, name: "Inflame", type: "Power", upgraded: true }),
  ];
  const prev = { decision: "combat_play", state: { hand, player: { name: "Ironclad" } } as never };
  const events = (cues: { event: string }[]) => cues.map((c) => c.event);

  // The two shapes `describe_action` renders a play as (render.rs).
  it("joins a targeted and an untargeted play label to the hand it came from", () => {
    expect(events(steppedActionCues(MAP, prev, "Play Strike → Fuzzy Wurm Crawler"))).toEqual(["event:/sfx/cards/play"]);
    expect(events(steppedActionCues(MAP, prev, "Play Strike"))).toEqual(["event:/sfx/cards/play"]);
  });

  // `render.rs::card_name` appends "+" to an upgraded card; `Card.name` never
  // carries it. Mutation caught: dropping the strip.
  it("strips the upgrade suffix the label adds", () => {
    expect(events(steppedActionCues(MAP, prev, "Play Inflame+"))).toEqual(["event:/sfx/cards/power"]);
  });

  it("is silent for anything that is not a card play from combat", () => {
    expect(steppedActionCues(MAP, prev, "End turn")).toEqual([]);
    expect(steppedActionCues(MAP, prev, "drink Fire Potion")).toEqual([]);
    expect(steppedActionCues(MAP, prev, "Play Bash")).toEqual([]); // not in hand
    expect(steppedActionCues(MAP, prev, undefined)).toEqual([]);
    expect(steppedActionCues(MAP, null, "Play Strike")).toEqual([]);
    expect(steppedActionCues(MAP, { ...prev, decision: "map" }, "Play Strike")).toEqual([]);
  });
});
