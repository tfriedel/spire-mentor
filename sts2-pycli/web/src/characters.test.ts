import { describe, expect, it } from "vitest";
import { characterArt, characterInBuild, characterKey } from "./characters";

describe("characterKey", () => {
  it("accepts the engine lane's bare name", () => {
    expect(characterKey("Ironclad")).toBe("ironclad");
    expect(characterKey("Necrobinder")).toBe("necrobinder");
  });

  it("accepts the live-game lane's localized title", () => {
    // `sts2_realgame._character_title` builds "The " + <Title Case> from
    // `characters.json`'s `<ID>.title`, which all five carry. Three components
    // each derived this key separately without the strip, so every live-game
    // run silently lost the rest-site hero, the run-end sprite and the HUD
    // portrait.
    expect(characterKey("The Ironclad")).toBe("ironclad");
    expect(characterKey("The Silent")).toBe("silent");
    expect(characterKey("The Defect")).toBe("defect");
    expect(characterKey("The Regent")).toBe("regent");
    expect(characterKey("The Necrobinder")).toBe("necrobinder");
  });

  it("returns null rather than a key we have no art for", () => {
    for (const bad of [null, undefined, "", "The", "Watcher", "the "]) {
      expect(characterKey(bad)).toBeNull();
    }
  });

  it("strips only a leading article, not one inside the name", () => {
    // Guard against a blanket `replace(/the/g, "")`, which would turn
    // "The Silent" into "silen" via the "t" — and quietly break all five.
    expect(characterKey("  THE   silent  ")).toBe("silent");
  });
});

describe("later-build characters (the Fool, v0.113.0)", () => {
  it("keys the Fool like the five", () => {
    expect(characterKey("Fool")).toBe("fool");
    expect(characterKey("The Fool")).toBe("fool");
  });

  it("finds the Fool's art in its build's overlay, everyone else's in the shared tree", () => {
    expect(characterArt("fool", "character_sprites/fool.png")).toBe("v0.113.0/character_sprites/fool.png");
    expect(characterArt("ironclad", "character_sprites/ironclad.png")).toBe("character_sprites/ironclad.png");
  });

  it("offers the Fool only on a build that has it", () => {
    expect(characterInBuild("fool", "v0.107.1")).toBe(false);
    expect(characterInBuild("fool", "v0.111.0")).toBe(false);
    expect(characterInBuild("fool", "v0.113.0")).toBe(true);
    expect(characterInBuild("fool", "v0.120.2")).toBe(true);
    expect(characterInBuild("silent", "v0.107.1")).toBe(true);
    expect(characterInBuild(null, "v0.107.1")).toBe(true);
  });
});
