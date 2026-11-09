import { describe, it, expect } from "vitest";
import { cuesForCard, cueForGold, cueForRelic, cueForTreasure, cueForUi } from "../runCues";
import type { SfxMap } from "../sfxMap";

const MAP: SfxMap = {
  version: "test",
  generic: {
    cardPlay: "event:/sfx/ui/cards/card_movement_B_play_into_discard",
    cardPower: "event:/sfx/ui/cards/card_movement_B_power",
    uiClick: "event:/sfx/ui/clicks/ui_click",
    mapOpen: "event:/sfx/ui/map/map_open",
    mapSelect: "event:/sfx/ui/map/map_select",
    gold: "event:/sfx/ui/gold/gold_1",
  },
  // "Blood Wall" (a multi-word DISPLAY TITLE, not the class name "BloodWall")
  // deliberately, so this exercises the same class-name -> display-title
  // lookup the wire actually does — see the 2026-08-23 final review, finding
  // 1: every fixture in this file used to key by class name, which let the
  // real keying bug ship (the web reads `card.name`, the wire's display
  // name, never the decomp's class name).
  cards: { "Blood Wall": "event:/sfx/characters/ironclad/ironclad_bloodwall" },
  relics: {}, monsters: {}, powers: {},
};

// The same map with the two player-side tables the generator now ships.
// `characters` is keyed by CLASS name ("Ironclad"), unlike every other table
// here — that is the real wire shape (`state.player.name`), not an oversight.
const CHAR_MAP: SfxMap = {
  ...MAP,
  cards: {
    ...MAP.cards,
    Whirlwind: "event:/sfx/characters/ironclad/ironclad_whirlwind",
  },
  characters: {
    Ironclad: {
      attack: "event:/sfx/characters/ironclad/ironclad_attack",
      cast: "event:/sfx/characters/ironclad/ironclad_cast",
      die: "event:/sfx/characters/ironclad/ironclad_die",
    },
    Silent: {
      attack: "event:/sfx/characters/silent/silent_attack",
      cast: "event:/sfx/characters/silent/silent_cast",
      die: "event:/sfx/characters/silent/silent_die",
    },
  },
  cardAnims: {
    Strike: { kind: "attack" },
    Whirlwind: { kind: "attack" },
    Inflame: { kind: "cast" },
    Mangle: { kind: "attack", except: "Ironclad" },
  },
};

describe("cueForUi — randomised families", () => {
  const M: SfxMap = {
    ...MAP,
    generic: {
      ...MAP.generic,
      battleStart1: "res://debug_audio/battle_start_1.mp3",
      battleStart2: "res://debug_audio/battle_start_2.mp3",
    },
  };

  it("picks the first variant at the bottom of the range", () => {
    expect(cueForUi(M, "battleStart", () => 0)?.event).toBe("res://debug_audio/battle_start_1.mp3");
  });

  it("picks the last variant at the top of the range", () => {
    expect(cueForUi(M, "battleStart", () => 0.999)?.event).toBe("res://debug_audio/battle_start_2.mp3");
  });

  // Mutation caught: indexing with `Math.floor(rand() * n)` alone. A `rand`
  // returning exactly 1 (permitted by nothing, but cheap to survive) would
  // index past the end and build a cue with an `undefined` event.
  it("never indexes past the end", () => {
    expect(cueForUi(M, "battleStart", () => 1)?.event).toBe("res://debug_audio/battle_start_2.mp3");
  });

  // A partially-regenerated map must degrade to the variants it has rather
  // than going silent a third of the time.
  it("falls back to the variants that resolve", () => {
    const partial: SfxMap = {
      ...MAP,
      generic: { ...MAP.generic, battleStart2: "res://debug_audio/battle_start_2.mp3" },
    };
    expect(cueForUi(partial, "battleStart", () => 0)?.event).toBe("res://debug_audio/battle_start_2.mp3");
  });

  it("returns null when the map names none of them", () => {
    expect(cueForUi(MAP, "battleStart", () => 0)).toBeNull();
  });
});

describe("cueForTreasure", () => {
  const M: SfxMap = { ...MAP, treasure: { "1": "event:/sfx/ui/treasure/treasure_act1" } };

  it("resolves the chest sound for the act", () => {
    expect(cueForTreasure(M, 1)?.event).toBe("event:/sfx/ui/treasure/treasure_act1");
  });

  it("stays silent for an act the map has no chest sound for", () => {
    expect(cueForTreasure(M, 4)).toBeNull();
  });

  it("stays silent when the map predates the table", () => {
    expect(cueForTreasure(MAP, 1)).toBeNull();
  });
});

describe("cuesForCard", () => {
  const events = (cues: { event: string }[]) => cues.map((c) => c.event);

  it("uses the card's own sound when the decomp gave it one", () => {
    expect(events(cuesForCard(MAP, "Blood Wall", "Power")))
      .toContain("event:/sfx/characters/ironclad/ironclad_bloodwall");
  });
  it("falls back to the generic play sound", () => {
    expect(events(cuesForCard(MAP, "Strike", "Attack")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });
  it("uses the power sound for a Power card", () => {
    expect(events(cuesForCard(MAP, "Inflame", "Power")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_power"]);
  });
  // The HIT layer — what the attack sounds like landing. 80 of the 81 cards
  // that carry one name a `res://debug_audio/` mp3, so this was the layer the
  // FMOD-only pipeline could not see at all. Mutation caught: dropping
  // `cardHits` from the stack, which silently returns the workbench to the
  // "Ironclad thuds on nothing" state.
  it("adds the card's hit sound, after the character's own and before the swoosh", () => {
    const m: SfxMap = {
      ...MAP,
      characters: { Ironclad: { attack: "event:/sfx/characters/ironclad/ironclad_attack" } },
      cardAnims: { Bash: { kind: "attack" } },
      cardHits: { Bash: "res://debug_audio/blunt_attack.mp3" },
    };
    expect(events(cuesForCard(m, "Bash", "Attack", "Ironclad"))).toEqual([
      "event:/sfx/characters/ironclad/ironclad_attack",
      "res://debug_audio/blunt_attack.mp3",
      "event:/sfx/ui/cards/card_movement_B_play_into_discard",
    ]);
  });

  it("plays no hit sound for a card the decomp gives none", () => {
    // 77 cards are `WithHitFx("vfx/…")` with no sound argument — Strike among
    // them. Absent beats substituted.
    const m: SfxMap = { ...MAP, cardHits: { Bash: "res://debug_audio/blunt_attack.mp3" } };
    expect(events(cuesForCard(m, "Strike", "Attack")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });

  it("plays no hit sound when the map predates the table", () => {
    // A deployed bucket can still be serving a map built before `cardHits`
    // existed; that must degrade to the old sound, not throw.
    expect(events(cuesForCard(MAP, "Bash", "Attack")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });

  it("marks card play as informative so reduced mode keeps it", () => {
    expect(cuesForCard(MAP, "Strike", "Attack")[0].cls).toBe("informative");
  });

  // AMENDMENT (i): the backend's card-type casing is not to be relied on —
  // CardFrame.tsx:86 normalises with .toLowerCase() before comparing, so this
  // must too. Mutation caught: comparing `cardType === "Power"` directly
  // (case-sensitive) instead of lower-casing first.
  it("uses the power sound for a lowercase 'power' type", () => {
    expect(events(cuesForCard(MAP, "Inflame", "power")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_power"]);
  });

  // AMENDMENT (ii): absent beats wrong — when neither the card's own sound
  // nor either generic key resolves, the result must be empty, never a cue
  // with an `undefined` event (which would build a garbage asset URL
  // downstream). Mutation caught: pushing
  // `{ event: map.generic[key] ?? map.generic.cardPlay, cls: "informative" }`
  // unconditionally instead of gating on `swoosh`.
  it("returns nothing when the card has no own sound and both generic keys are absent", () => {
    const bare: SfxMap = { ...MAP, generic: {}, cards: {} };
    expect(cuesForCard(bare, "Strike", "Attack")).toEqual([]);
  });

  // --- the player's own body, the layer this file used to be missing ----
  // The bug this fixes: an Ironclad Strike played ONLY the card swoosh, so
  // the whole game sounded like shuffling paper. Each of these asserts the
  // character sound is ADDED, never substituted — the swoosh is still there
  // in every case, because the game plays both.
  it("adds the character's attack sound for an attack card", () => {
    expect(events(cuesForCard(CHAR_MAP, "Strike", "Attack", "Ironclad")))
      .toEqual([
        "event:/sfx/characters/ironclad/ironclad_attack",
        "event:/sfx/ui/cards/card_movement_B_play_into_discard",
      ]);
  });
  it("adds the character's cast sound for a card that animates Cast/PowerUp", () => {
    expect(events(cuesForCard(CHAR_MAP, "Inflame", "Power", "Ironclad")))
      .toContain("event:/sfx/characters/ironclad/ironclad_cast");
  });

  // Mutation caught: keying `characters` by the same display title the
  // `cards` table uses ("The Ironclad"), or reading the anim table with the
  // character instead of the card name.
  it("reads the character table by CLASS name, the string the wire sends", () => {
    expect(events(cuesForCard(CHAR_MAP, "Strike", "Attack", "The Ironclad")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });

  // `Whirlwind.OnPlay` plays its own sfx AND runs a `FromCard` attack chain,
  // so the game stacks three sounds. Mutation caught: returning the card's
  // own sound INSTEAD of the generic one (what the pre-fix single-cue version
  // did), which silently drops one layer.
  it("stacks all three layers for a card with its own sound", () => {
    expect(events(cuesForCard(CHAR_MAP, "Whirlwind", "Attack", "Ironclad")))
      .toEqual([
        "event:/sfx/characters/ironclad/ironclad_attack",
        "event:/sfx/characters/ironclad/ironclad_whirlwind",
        "event:/sfx/ui/cards/card_movement_B_play_into_discard",
      ]);
  });

  // Defend calls no TriggerAnim at all in the decomp, so it is absent from
  // `cardAnims` and must stay silent apart from the swoosh. Mutation caught:
  // defaulting a missing `cardAnims` entry to "attack".
  it("plays no character sound for a card that animates nothing", () => {
    expect(events(cuesForCard(CHAR_MAP, "Defend", "Skill", "Ironclad")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });

  // Mangle fires the `heavyAttack` trigger for Ironclad — a name
  // `CreatureCmd.TriggerAnim`'s switch does not handle, so the real game
  // plays nothing — but plain "Attack" for everyone else. Mutation caught:
  // ignoring `except`, which would give Ironclad a sound the game never
  // makes; or inverting it, which would silence everyone BUT Ironclad.
  it("honours `except`: silent for the exempt character, sounded for others", () => {
    expect(events(cuesForCard(CHAR_MAP, "Mangle", "Attack", "Ironclad")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
    expect(events(cuesForCard(CHAR_MAP, "Mangle", "Attack", "Silent")))
      .toContain("event:/sfx/characters/silent/silent_attack");
  });

  // A stale deployed bucket can still be serving a map generated before
  // these tables existed. Mutation caught: dropping the `?.` and throwing.
  it("degrades to the old swoosh-only behaviour on a map with no character tables", () => {
    expect(events(cuesForCard(MAP, "Strike", "Attack", "Ironclad")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });

  it("plays no character sound when the caller knows no character", () => {
    expect(events(cuesForCard(CHAR_MAP, "Strike", "Attack")))
      .toEqual(["event:/sfx/ui/cards/card_movement_B_play_into_discard"]);
  });
});

describe("cueForGold", () => {
  const TIERED: SfxMap = {
    ...MAP,
    generic: {
      ...MAP.generic,
      goldSmall: "event:/sfx/ui/gold/gold_1",
      goldMedium: "event:/sfx/ui/gold/gold_2",
      goldLarge: "event:/sfx/ui/gold/gold_3",
    },
  };

  // The thresholds are `>= 100` and `> 30` — asymmetric, straight from
  // `PlayerCmd.cs:152`. Both boundaries are pinned because that asymmetry is
  // the exact thing a "tidy-up" would flatten to `>=` on both.
  it.each([
    [1, "gold_1"], [30, "gold_1"], [31, "gold_2"], [99, "gold_2"],
    [100, "gold_3"], [500, "gold_3"],
  ])("picks the right tier for %i gold", (amount, expected) => {
    expect(cueForGold(TIERED, amount as number)?.event).toContain(expected as string);
  });

  // A map generated before the tiers existed must still make a sound.
  // Mutation caught: dropping the `?? map.generic.gold` fallback, which turns
  // gold pickups silent against a stale deployed bucket.
  it("falls back to the untiered gold key", () => {
    const legacy: SfxMap = { ...MAP, generic: { gold: "event:/sfx/ui/gold/gold_1" } };
    expect(cueForGold(legacy, 250)?.event).toBe("event:/sfx/ui/gold/gold_1");
  });

  it("returns null when the map names no gold sound at all", () => {
    expect(cueForGold({ ...MAP, generic: {} }, 50)).toBeNull();
  });
});

describe("cueForRelic", () => {
  // The base MAP fixture carries no generic `relic` key, so a version of this
  // test using it alone would pass just as well with `cueForRelic`'s `??`
  // order swapped (generic-first) — the fallback would simply never resolve
  // either way. Both a generic `relic` key AND the relic's own key must be
  // present so the own-sound branch is the ONLY thing distinguishing them.
  // Mutation caught: `map.generic.relic ?? map.relics[relicName]`.
  it("uses the relic's own sound when the decomp gave it one, even when a generic fallback also exists", () => {
    // "Iron Club" (display title), not the class name "IronClub" — see the
    // cards fixture note above.
    const m = {
      ...MAP,
      generic: { ...MAP.generic, relic: "event:/sfx/ui/relic_activate_general" },
      relics: { "Iron Club": "event:/sfx/relics/iron_club" },
    };
    expect(cueForRelic(m, "Iron Club")?.event).toBe("event:/sfx/relics/iron_club");
  });
  it("falls back to the generic relic-activate sound", () => {
    const m = { ...MAP, generic: { ...MAP.generic, relic: "event:/sfx/ui/relic_activate_general" } };
    expect(cueForRelic(m, "Anchor")?.event).toBe("event:/sfx/ui/relic_activate_general");
  });
  it("returns null when neither exists", () => {
    expect(cueForRelic(MAP, "Anchor")).toBeNull();
  });
});

describe("cueForUi", () => {
  it("maps a known kind", () => {
    expect(cueForUi(MAP, "mapSelect")?.event).toBe("event:/sfx/ui/map/map_select");
  });
  it("marks clicks as chatter", () => {
    expect(cueForUi(MAP, "click")?.cls).toBe("chatter");
  });
  it("returns null when the map has no entry", () => {
    expect(cueForUi({ ...MAP, generic: {} }, "mapSelect")).toBeNull();
  });

  // AMENDMENT (iii): `victory` is a permanently dead key — no shipped map
  // will ever carry a `generic.victory` entry (the decomp has no victory
  // event), so this must always resolve to null. Asserted explicitly (rather
  // than left to be discovered) so a future change that adds a bogus
  // `victory` fallback elsewhere doesn't quietly un-silence it.
  it("returns null for victory even when every other generic key is present — no map ships a victory event", () => {
    expect(cueForUi(MAP, "victory")).toBeNull();
  });
});

describe("cueForUi — the game's per-call volume and pitch spread", () => {
  const M: SfxMap = {
    ...MAP,
    generic: {
      ...MAP.generic,
      cardDeal: "res://debug_audio/card_deal.mp3",
      potionSlosh1: "res://debug_audio/potion_slosh_1.mp3",
      relicGet: "res://debug_audio/relic_get.mp3",
    },
  };

  // `CardPileCmd.cs:875` — `Play("card_deal.mp3", 0.25f, PitchVariance.Small)`.
  // Shipped at full volume and one fixed pitch until this table existed.
  it("carries the call site's volume and variance", () => {
    expect(cueForUi(M, "cardDeal")).toMatchObject({ gain: 0.25, pitchVariance: 0.02 });
    expect(cueForUi(M, "potionSlosh", () => 0)).toMatchObject({ gain: 0.3, pitchVariance: 0.1 });
  });

  // `RelicCmd.cs:49` passes the defaults; FMOD cues never take either field.
  it("leaves both unset where the game passes the defaults", () => {
    const relic = cueForUi(M, "relicGet");
    expect(relic?.gain).toBeUndefined();
    expect(relic?.pitchVariance).toBeUndefined();
    expect(cueForUi(M, "mapOpen")?.pitchVariance).toBeUndefined();
  });
});
