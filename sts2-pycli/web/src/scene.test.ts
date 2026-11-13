import { describe, expect, it } from "vitest";
import {
  GROUND_Y, INTENT_HEADROOM_PX, MIN_EDGE, SCENE_CX, SCENE_CY,
  aspectFit, fitScene, placeEnemies, placePlayer, rowHalfWidths, lowestFootY, sceneFitHeight, sceneFramingFor,
} from "./scene";
import { MAX_FRAMING } from "./stageZoom";

describe("rowHalfWidths", () => {
  it("is the scene's own half on a stage that shows the whole scene", () => {
    // A desktop stage shows 960+ scene px each side of centre: nothing moves.
    for (const half of [960, 1100, 1400]) {
      expect(rowHalfWidths(half, 1, [250])).toEqual({ player: SCENE_CX, enemies: SCENE_CX });
    }
  });

  it("brings the player in to the edge on a narrow stage, enemies unchanged", () => {
    // A phone in portrait shows ~440 scene px each side. The game parked the
    // player in the middle of the left half (x ≈ -480); against a 440 half
    // the player stands at the 150 px minimum edge instead.
    const wide = placePlayer(200, 1, false);
    const narrow = placePlayer(200, 1, false, rowHalfWidths(440, 1, [250]).player);
    expect(wide.x).toBeCloseTo(-480, 0);
    expect(narrow.x).toBeCloseTo(-(MIN_EDGE + 100), 0);
    // A single enemy starts at the same 150 px edge either way.
    const [e] = placeEnemies([250], 1, rowHalfWidths(440, 1, [250]).enemies);
    expect(e.x).toBeCloseTo(MIN_EDGE + 125, 0);
  });

  it("never narrows the enemies' half past what the row needs", () => {
    // Two 250-wide enemies need 2*150 + 500 + 5 = 805 of half-width at the
    // tightest gap. Below that `placeEnemies`'s collapse branch would centre
    // the row on the narrow half and walk the first enemy left of centre,
    // into the player.
    const h = rowHalfWidths(440, 1, [250, 250]);
    expect(h.enemies).toBe(805);
    const row = placeEnemies([250, 250], 1, h.enemies);
    expect(row[0].x - 125).toBeGreaterThanOrEqual(MIN_EDGE);
    // ...and `aspectFit` then shrinks the whole row to the stage.
    const fit = aspectFit(row, [250, 250], 1, 440);
    expect(fit).toBeLessThan(1);
    expect(Math.max(...row.map((s, i) => (s.x + [250, 250][i] / 2))) * fit).toBeLessThanOrEqual(440);
    // A row too wide even for the scene gets the game's own crowded layout.
    const three = rowHalfWidths(440, 1, [250, 250, 250]);
    expect(three.enemies).toBe(SCENE_CX);
    expect(placeEnemies([250, 250, 250], 1, three.enemies)).toEqual(placeEnemies([250, 250, 250], 1));
  });

  it("scales the tightest half by the camera, like the halves it compares to", () => {
    // Under a 0.8 camera the halves are divided by 0.8 inside placeEnemies,
    // so the floor has to be pre-multiplied to survive that division.
    const h = rowHalfWidths(440, 0.8, [250, 250]);
    expect(h.enemies).toBe(805 * 0.8);
  });
});

describe("aspectFit", () => {
  it("holds the LEFT edge too, not only the game's right edge", () => {
    // The player 600 px left of centre with a 200 px box reaches -700; a
    // stage showing 440 each side needs the row shrunk to fit that side.
    const fit = aspectFit([{ x: -600, y: 0 }, { x: 300, y: 0 }], [200, 200], 1, 440);
    expect(fit).toBeCloseTo(440 / 715, 3);
    // Both inside: untouched.
    expect(aspectFit([{ x: -250, y: 0 }, { x: 300, y: 0 }], [200, 200], 1, 440)).toBe(1);
  });
});

describe("sceneFramingFor", () => {
  // The zoom layer an 8.8" tablet in landscape lays the scene out in: a
  // 1002x465 stage at a zoom of 0.6 (stageZoom.ts), i.e. the box `fitScene`
  // is actually measured off.
  const W = 1670;
  const H = 775;
  const full = MAX_FRAMING;
  // Crowns in scene rows, through each fight's own camera: the Bygone Effigy
  // (588 px rig, 0.88 / +50 camera) and a Cultist (341 px rig, no camera).
  const EFFIGY = SCENE_CY + (GROUND_Y - 588 - SCENE_CY) * 0.88 + 50;
  const CULTIST = GROUND_Y - 341;

  it("leaves an ordinary fight at the stage's full framing", () => {
    expect(sceneFramingFor(W, H, full, CULTIST)).toBe(full);
    expect(sceneFramingFor(W, H, full, null)).toBe(full);
  });

  it("pulls the Effigy fight back exactly as far as its intent chip needs", () => {
    const f = sceneFramingFor(W, H, full, EFFIGY);
    expect(f).toBeLessThan(full);
    expect(f).toBeGreaterThan(1);
    // At the full framing the chip was ~100 layer px above the stage.
    expect(fitScene(W, H, full).y(EFFIGY)).toBeLessThan(0);
    // At the answer it is on the stage with the headroom the chip needs —
    // and one step larger would not be.
    expect(fitScene(W, H, f).y(EFFIGY)).toBeGreaterThanOrEqual(INTENT_HEADROOM_PX);
    expect(fitScene(W, H, f + 0.01).y(EFFIGY)).toBeLessThan(INTENT_HEADROOM_PX);
    expect(f).toBe(Number(f.toFixed(2)));
  });

  it("never goes below the game's own frame", () => {
    // A crown at the very top of the scene cannot be given room; the pure fit
    // is the game's own picture and stays.
    expect(sceneFramingFor(W, H, full, 0)).toBe(1);
    expect(sceneFramingFor(W, H, full, -500)).toBe(1);
  });

  it("does nothing on a monitor, where the framing is 1 to begin with", () => {
    expect(sceneFramingFor(1920, 1002, 1, EFFIGY)).toBe(1);
    expect(sceneFramingFor(1920, 1002, 1, 0)).toBe(1);
  });

  it("does nothing on a portrait stage, which crops no rows", () => {
    // A phone in portrait: the width binds, the stage shows scene rows above
    // and below the frame, and the crown has all the room it needs.
    const f = sceneFramingFor(412 / 0.5, 700 / 0.5, 2.2, EFFIGY);
    expect(f).toBe(2.2);
  });

  it("is monotone in the crown: a taller enemy never gets MORE framing", () => {
    let prev = sceneFramingFor(W, H, full, 400);
    for (let crown = 400; crown >= 100; crown -= 10) {
      const f = sceneFramingFor(W, H, full, crown);
      expect(f).toBeLessThanOrEqual(prev);
      prev = f;
    }
  });
});

describe("sceneFitHeight", () => {
  it("keeps the stage height when the ground already clears the limit", () => {
    expect(sceneFitHeight(1600, 800, 1, null, GROUND_Y, 800)).toBe(800);
  });
  it("lifts the ground line to the limit on a short wide stage (phone landscape)", () => {
    const [w, h, framing] = [1830, 768, MAX_FRAMING];
    const limit = 320;
    const fitH = sceneFitHeight(w, h, framing, 300, GROUND_Y, limit);
    expect(fitH).toBeLessThan(h);
    const g = fitScene(w, fitH, sceneFramingFor(w, fitH, framing, 300)).y(GROUND_Y);
    expect(g).toBeLessThanOrEqual(limit);
    expect(g).toBeGreaterThan(limit - 4); // the largest box that clears it
  });
  it("in portrait only shifts the scene up, at the same scale", () => {
    const [w, h] = [896, 1928];
    const limit = fitScene(w, h, 1).y(GROUND_Y) - 100;
    const fitH = sceneFitHeight(w, h, 1, null, GROUND_Y, limit);
    expect(fitScene(w, fitH, 1).k).toBeCloseTo(fitScene(w, h, 1).k);
    expect(fitScene(w, fitH, 1).y(GROUND_Y)).toBeLessThanOrEqual(limit);
  });
});

describe("lowestFootY", () => {
  it("is the ground line when nothing stands lower", () => {
    expect(lowestFootY([])).toBe(GROUND_Y);
    expect(lowestFootY([715, 737])).toBe(GROUND_Y);
  });
  it("is an authored marker below the ground line (Exoskeletons' second slot)", () => {
    expect(lowestFootY([700, 756])).toBe(756);
  });
  it("a creature on a low marker still lands above the phone limit", () => {
    const [w, h, framing] = [1830, 768, MAX_FRAMING];
    const limit = 320;
    const foot = lowestFootY([756]);
    const fitH = sceneFitHeight(w, h, framing, 300, foot, limit);
    const fitFor = (fh: number) => fitScene(w, fh, sceneFramingFor(w, fh, framing, 300));
    expect(fitFor(fitH).y(foot)).toBeLessThanOrEqual(limit);
    // Fitting the ground line alone would have left that creature below it.
    const groundOnly = sceneFitHeight(w, h, framing, 300, GROUND_Y, limit);
    expect(fitFor(groundOnly).y(foot)).toBeGreaterThan(limit);
  });
});
