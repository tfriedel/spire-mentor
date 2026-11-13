/**
 * The combat scene's coordinate system — the game's, not ours.
 *
 * Everything the arena draws (the room art, where each fighter stands, how big
 * its rig is) is laid out in the game's own 1920x1080 design pixels and then
 * mapped onto whatever box the workbench stage actually got. The numbers here
 * are read off the v0.107.1 decomp, not tuned by eye:
 *
 *   * `NCombatRoom.CreateEnemyNodes` offsets the encounter scene by
 *     `new Vector2(1920f, 1080f) * 0.5f` — the scene IS 1920x1080, and the
 *     ally/enemy containers hang off its CENTRE.
 *   * `PositionEnemies` / `PositionPlayersAndPets` place every creature at
 *     `y = 200` below that centre (so the ground line is y=740) and at an x
 *     derived from each creature's `Visuals.Bounds` width — ported below.
 *   * `SceneContainer.Scale = encounter.GetCameraScaling()` and
 *     `SceneContainer.Position += encounter.GetCameraOffset()` zoom the WHOLE
 *     scene (room art included) per encounter — that is what makes the Kaiser
 *     Crab's room read as one giant crab rather than a monster in a field.
 *
 * A creature's rig is drawn at its authored scene scale (`rig_scale`, from
 * `creature_scales.json`) pixels per rig unit, with the rig's root ON the
 * ground line. That is the whole sizing rule: the workbench used to multiply it
 * by two eyeballed constants and then clamp the result to the arena's height,
 * which is why a Decimillipede came out at under half its size and every giant
 * came out at exactly the same size as every other giant.
 */
import { rigBox, type RigBox } from "./rigBounds";

/** The game's design viewport. */
export const SCENE_W = 1920;
export const SCENE_H = 1080;
export const SCENE_CX = SCENE_W / 2;
export const SCENE_CY = SCENE_H / 2;

/** The ground line: `Position.Y = 200f` under the scene centre, for allies and
 *  enemies alike (`NCombatRoom.PositionEnemies` / `PositionPlayersAndPets`). */
export const GROUND_Y = SCENE_CY + 200;

/** How much of the scene's top the game's own HUD band covers, measured off
 *  refs/game-combat-1.png (the torn edge lands at y≈78 of 1080). The workbench
 *  draws its own band ABOVE the stage, so the stage shows scene rows
 *  `HUD_BAND_H..1080` and this is what makes the two framings line up. */
export const HUD_BAND_H = 78;
/** Scene height the stage is fitted to. */
export const SCENE_VIEW_H = SCENE_H - HUD_BAND_H;

/** Gap between neighbouring creatures, and the minimum inset from the scene's
 *  centre line to the first one (`num2` / the `Math.Max(val, 150f)` clamp). */
export const CREATURE_GAP = 70;
export const MIN_EDGE = 150;

/** The room art's authoring canvas, and the scale its scene draws it at.
 *
 *  The extracted layers are 1920x900; the combat-room background scene puts
 *  them up CENTRED on the scene — so the visible frame is the middle 1920x1080
 *  of the painting, not the whole canvas fitted to the box. Confirmed by fitting
 *  the layer stack to refs/game-combat-1.png: a scale sweep bottoms out at
 *  exactly 1.44 with the canvas centred (mean abs error 3.0/255, against 4.3 at
 *  the next-best scale).
 *
 *  `ROOM_CANVAS_SCALE` is that Overgrowth number and is now only the FALLBACK:
 *  the size is authored per room and read out of the PCK into
 *  `RoomScene.scale`, which is 1.50078 for the Hive/Glory family. */
export const ROOM_CANVAS_W = 1920;
export const ROOM_CANVAS_H = 900;
export const ROOM_CANVAS_SCALE = 1.44;

/** How far right of the scene's centre line the room hangs, in scene px.
 *
 *  `combat_room.tscn` anchors `%BgContainer` to the centre (0.5/0.5, zero size)
 *  and then nudges it: `offset_left = 23.0`. Every layer's placement is
 *  relative to that node, so the whole painting is 23 px right of where the
 *  fighters' containers sit. Small, but measurable: sampling the wall bands of
 *  refs/game-combat-2.png against an offline composite, adding it takes the
 *  left band from 2.6% off to 0.2% and the right from 6.1% to 2.9%
 *  (docs/experiments/2026-08-30-combat-room-grade). */
export const ROOM_BG_OFFSET_X = 23;

/** Per-encounter camera, from the `GetCameraScaling` / `GetCameraOffset` /
 *  `FullyCenterPlayers` overrides on `EncounterModel` subclasses. */
export interface Camera {
  /** `SceneContainer.Scale` — zooms the room art and the fighters together. */
  scale: number;
  /** `SceneContainer.Position +=`, in scene px (positive y = down). */
  offsetY: number;
  /** `FullyCenterPlayers` — the player stands on the scene's centre line
   *  instead of in the left half (the "surrounded" bosses). */
  centerPlayer: boolean;
}

export const DEFAULT_CAMERA: Camera = { scale: 1, offsetY: 0, centerPlayer: false };

/** The camera each encounter with an override asks for, keyed by a rig slug
 *  that only that encounter fields.
 *
 *  The wire carries no encounter id (the engine does not expose one), so the
 *  roster is the handle we have. Every entry below is one encounter's own
 *  override; an encounter without one keeps `DEFAULT_CAMERA`, which is what
 *  the base `EncounterModel` returns. */
const CAMERA_BY_RIG: Record<string, Camera> = {
  // KaiserCrabBoss — 0.75 / Down*35, and the only combat that centres the
  // player: the crab is the room, with a claw reaching in from each side.
  kaiser_crab: { scale: 0.75, offsetY: 35, centerPlayer: true },
  // DecimillipedeElite — 0.87 / Down*50.
  decimillipede_segment_front: { scale: 0.87, offsetY: 50, centerPlayer: false },
  decimillipede_segment_middle: { scale: 0.87, offsetY: 50, centerPlayer: false },
  decimillipede_segment_back: { scale: 0.87, offsetY: 50, centerPlayer: false },
  aeonglass: { scale: 0.9, offsetY: 60, centerPlayer: false },
  hourglass_placeholder: { scale: 0.9, offsetY: 60, centerPlayer: false },
  bygone_effigy: { scale: 0.88, offsetY: 50, centerPlayer: false },
  ceremonial_beast: { scale: 0.9, offsetY: 50, centerPlayer: false },
  fabricator: { scale: 0.85, offsetY: 60, centerPlayer: false },
  knowledge_demon: { scale: 0.85, offsetY: 70, centerPlayer: false },
  lagavulin_matriarch: { scale: 0.9, offsetY: 0, centerPlayer: false },
  living_fog: { scale: 0.9, offsetY: 0, centerPlayer: false },
  mecha_knight: { scale: 0.9, offsetY: 50, centerPlayer: false },
  myte: { scale: 0.9, offsetY: 50, centerPlayer: false },
  ovicopter: { scale: 0.8, offsetY: 0, centerPlayer: false },
  phantasmal_gardener: { scale: 0.85, offsetY: 40, centerPlayer: false },
  queen: { scale: 0.9, offsetY: 60, centerPlayer: false },
  slumbering_beetle: { scale: 0.85, offsetY: 0, centerPlayer: false },
  terror_eel: { scale: 0.9, offsetY: 0, centerPlayer: false },
  the_insatiable: { scale: 0.9, offsetY: 0, centerPlayer: false },
  vantom: { scale: 0.9, offsetY: 50, centerPlayer: false },
  waterfall_giant: { scale: 0.9, offsetY: 0, centerPlayer: false },
  // KnightsElite (0.87 / Down*50) — the elite that fields the Magi/Flail/Spectral
  // knights together. `mysterious_knight` shares the flail rig, so key on the
  // two that only this encounter brings.
  magi_knight: { scale: 0.87, offsetY: 50, centerPlayer: false },
  spectral_knight: { scale: 0.87, offsetY: 50, centerPlayer: false },
  // SlimesNormal (0.9 / Down*50) and TwoTailedRatsNormal (0.85 / Down*25).
  twig_slime_m: { scale: 0.9, offsetY: 50, centerPlayer: false },
  leaf_slime_m: { scale: 0.9, offsetY: 50, centerPlayer: false },
  two_tailed_rat: { scale: 0.85, offsetY: 25, centerPlayer: false },
  // TheKinBoss — 0.85 / Down*50.
  kin_priest: { scale: 0.85, offsetY: 50, centerPlayer: false },
  kin_follower: { scale: 0.85, offsetY: 50, centerPlayer: false },
};

/** The camera for a fight, from the rigs on the board. First match wins; the
 *  rosters above are disjoint, so the order among enemies does not matter. */
export function cameraFor(rigBases: (string | null | undefined)[]): Camera {
  for (const rig of rigBases) {
    const cam = CAMERA_BY_RIG[slugOf(rig)];
    if (cam) return cam;
  }
  return DEFAULT_CAMERA;
}

function slugOf(rigBase: string | null | undefined): string {
  if (!rigBase) return "";
  return rigBase.split("?")[0].split("/").filter(Boolean).pop() ?? "";
}

/** What the layout needs to know about one fighter. */
export interface SceneCreature {
  /** Rig base URL, or null for a sprite-only combatant. */
  rig?: string | null;
  /** Authored scene scale — px per rig unit (`creature_scales.json`). */
  rigScale?: number | null;
  /** Authored `Visuals.Bounds` height in scene px, when the game has one. */
  boundsH?: number | null;
  /** Authored `Visuals.Bounds` WIDTH in scene px — the number
   *  `NCombatRoom.PositionEnemies` actually spaces the row by. */
  boundsW?: number | null;
}

/** Fallback footprint for a combatant with neither a rig nor authored bounds —
 *  roughly a mid-size monster, so a sprite-only fight still spaces out. */
const FALLBACK_BOUNDS_W = 300;

/** How much wider than the rig it draws a creature's `%Bounds` box is.
 *
 *  Only reached now by a combatant the manifest has no `bounds_w` for —
 *  the eleven `creature_scales.json` entries with no `creature_visuals` scene
 *  (the Kaiser Crab's shared rig pages, `door`, the Battleworn Dummy) and any
 *  older backend that does not send the field. Everything else uses the
 *  authored width, which is the game's own number.
 *
 *  It is measured, not guessed — from a game frame of the Living Shield +
 *  Turret Operator fight, where the row algorithm below is invertible. Two
 *  enemies at screen x 0.675 and 0.89 of a 1920 scene give `w1 + w2 = 720` and,
 *  through the `MIN_EDGE` clamp the pair sits on, `w1 = 400` / `w2 = 320` —
 *  1.14x and 1.20x the widths those two rigs are DRAWN at (350 and 254 scene
 *  px). The PCK has since confirmed the shape of that answer: those two boxes
 *  are authored 356 and 280 wide, so the estimate was right to within 11%
 *  (GH #607).
 *
 *  It is emphatically not a ratio to take from `bounds_h`, which this used to
 *  do. That was safe-looking and wrong twice over: it is a different axis, and
 *  the `bounds_h` it read was itself about twice the authored box (GH #611, now
 *  fixed — both edges come off the one rect and both run ~1.07x the drawn rig).
 *  Reading a ~2x height ratio into x made every creature's footprint about 1.75x
 *  too wide, which is a different LAYOUT and not merely a rounder one: a row
 *  wider than `960/scaling` takes the collapse branch in `placeEnemies`, whose
 *  `val` runs negative and walks the whole row left into the player. That is the
 *  Hive fight where a Bowlbug stood ON the Ironclad — its Slumbering Beetle was
 *  being spaced as 1560 scene px of monster. */
const BOUNDS_MARGIN = 1.2;

/** A creature's footprint in scene px: the width the game's row layout spaces
 *  by (`NCreature.Visuals.Bounds.Size.X`).
 *
 *  Read straight off the authored box when the backend sends one. Everything
 *  below that is the fallback ladder for a combatant with no authored width:
 *  margin the rig it draws, then the authored box height through the rig's
 *  aspect, then a mid-size monster. */
export function boundsWidth(c: SceneCreature): number {
  if (c.boundsW != null && c.boundsW > 0) return c.boundsW;
  const box: RigBox | null = rigBox(c.rig);
  if (!box || box.h <= 0 || box.w <= 0) return c.boundsH ?? FALLBACK_BOUNDS_W;
  const scale = Math.abs(c.rigScale ?? 0);
  if (scale > 0) return box.w * scale * BOUNDS_MARGIN;
  return c.boundsH != null && c.boundsH > 0
    ? c.boundsH * (box.w / box.h)
    : FALLBACK_BOUNDS_W;
}

/** A creature's rendered size in scene px at its authored scale: `w`/`h` are
 *  the drawn bounding box, `top` is how far the rig reaches ABOVE its root —
 *  the creature's height as the scene sees it, since the root is what stands on
 *  the ground line. (A rig's box usually starts a little under the root: a
 *  contact shadow, kicked dust.) */
export function renderedSize(c: SceneCreature): { w: number; h: number; top: number } | null {
  const box = rigBox(c.rig);
  const scale = Math.abs(c.rigScale ?? 0);
  if (!box || !scale) return null;
  return { w: box.w * scale, h: box.h * scale, top: (box.y + box.h) * scale };
}

/** The per-instance size multiplier from `NCombatRoom.RandomizeEnemyScalesAndHues`.
 *
 *  The game scales the CREATURE after the row has been laid out, so this
 *  belongs to the drawn rig and to the column reserved above it — never to the
 *  `Visuals.Bounds` width the row spaces by (`scene.ts:boundsWidth`), which
 *  stays the authored box. Absent means "not one of two or more of a kind", and
 *  is drawn at the authored size. */
export function instanceScale(c: { size_jitter?: number | null }): number {
  const j = c.size_jitter;
  return typeof j === "number" && j > 0 ? j : 1;
}

/** Where one fighter stands, in scene px: `x` is its Bounds centre and `y` its
 *  ground line (the rig's root sits here). */
export interface Slot { x: number; y: number }

/**
 * `NCombatRoom.PositionEnemies`, verbatim.
 *
 * Widths are Bounds widths in scene px; `scaling` is the encounter camera's
 * scale (the room's usable half-width grows as the camera pulls back). The
 * returned x's are relative to the scene CENTRE, positive = right.
 *
 * `halfScene` is the half-width the row is laid out for, in scene px before
 * the camera — the game's `SCENE_CX` (its window is always 1920 wide), or
 * what a NARROW stage can actually show (see `rowHalfWidths`): the algorithm
 * then closes the gaps itself, exactly as it does for a crowded row.
 */
export function placeEnemies(widths: number[], scaling: number, halfScene: number = SCENE_CX): Slot[] {
  const n = widths.length;
  if (n === 0) return [];
  const half = halfScene / scaling;
  let gap = CREATURE_GAP;
  const sum = widths.reduce((a, w) => a + w, 0);
  let total = sum + (n - 1) * gap;
  let val = Math.max((half - total) / 2, MIN_EDGE);
  let stagger = 0;
  if (val + total > half && n > 1) {
    gap = Math.max((half - MIN_EDGE - sum) / (n - 1), 5);
    total = sum + (n - 1) * gap;
    val = (half - total) / 2;
    // A row squeezed past ~30px of gap lifts every SECOND creature so the
    // overlapping bodies stay readable.
    if (gap < 30) stagger = 60 + (40 - 60) * ((gap - 5) / 25);
  }
  const out: Slot[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ x: val + widths[i] / 2, y: GROUND_Y - (i % 2 !== 0 ? stagger : 0) });
    val += widths[i] + gap;
  }
  return out;
}

/**
 * `NCombatRoom.PositionPlayersAndPets` for the single local player the
 * workbench ever has (the multiplayer grid collapses to one row, one column).
 * Returns the player's x relative to the scene centre — negative, i.e. left —
 * except under `fullyCenterPlayers`, which puts them exactly on the centre
 * line (the Kaiser Crab's "surrounded" framing).
 */
export function placePlayer(width: number, scaling: number, fullyCenter: boolean, halfScene: number = SCENE_CX): Slot {
  if (fullyCenter) return { x: 0, y: GROUND_Y };
  const half = halfScene / scaling;
  const edge = Math.max((half - width) / 2, MIN_EDGE);
  return { x: -(edge + width / 2), y: GROUND_Y };
}

/** Gap the enemy row closes down to before the workbench shrinks it. */
const ROW_MIN_GAP = 5;

/**
 * The scene half-widths to lay the player and the enemy row out for on a
 * stage that shows LESS than the game's 1920 px — a phone or a tablet in
 * portrait. The game never sees one: its window is always the full scene,
 * so `placePlayer` parks the player in the middle of the left half (x ≈ -480)
 * and the enemies start 150 px right of centre, and on a 412 px screen the
 * two stood at opposite edges with the room between them. Laying the row out
 * for the width the stage can show — the same algorithm, a narrower `half` —
 * brings the player in to the 150 px edge and closes the enemies' gaps
 * before `aspectFit` has to shrink anyone.
 *
 * The enemies' half never drops below what the row needs at its tightest —
 * `placeEnemies`'s collapse branch centres the row on the half, so that is
 * the widths plus `ROW_MIN_GAP`s with `MIN_EDGE` on BOTH sides. Past that the
 * first enemy would walk left of centre, into the player. (A row too wide
 * even for the scene gets the game's own crowded layout, as it always did.)
 * `aspectFit` takes it from there.
 *
 * On a stage that shows the whole scene both are `SCENE_CX` — no change.
 */
export function rowHalfWidths(
  visibleHalfW: number, scaling: number, enemyWidths: number[],
): { player: number; enemies: number } {
  const player = Math.min(SCENE_CX, visibleHalfW);
  const n = enemyWidths.length;
  const tightest = (2 * MIN_EDGE + enemyWidths.reduce((a, w) => a + w, 0) + ROW_MIN_GAP * Math.max(0, n - 1)) * scaling;
  return { player, enemies: Math.min(SCENE_CX, Math.max(player, tightest)) };
}

/** Where the Necrobinder's Osty stands, given the player's slot and Osty's own
 *  max HP (`NCreature.GetOstyOffsetFromPlayer` + `PositionLocalPlayerOsty`:
 *  the player steps 150px left, and Osty takes half the player's hitbox plus a
 *  150..250px reach that grows with its max HP, 75px up). */
export function placeOsty(playerSlot: Slot, playerWidth: number, ostyMaxHp: number): Slot {
  const t = Math.min(Math.max(ostyMaxHp / 150, 0), 1);
  const reach = 150 + (250 - 150) * t;
  return { x: playerSlot.x + 150 + playerWidth / 2 + reach, y: playerSlot.y - 75 };
}

/** Player x shift when Osty is on the board. */
export const OSTY_PLAYER_SHIFT = -150;

/**
 * `NCombatRoom.AdjustCreatureScaleForAspectRatio`: when the rightmost fighter
 * would stand past the room's right edge, both containers shrink to fit rather
 * than letting it hang off the screen. `visibleHalfW` is half of what the stage
 * can actually show, in scene px (a stage wider than 16:9 shows MORE scene, so
 * this normally does nothing).
 */
export function aspectFit(slots: Slot[], widths: number[], scaling: number, visibleHalfW: number): number {
  let right = 0;
  let left = 0;
  for (let i = 0; i < slots.length; i++) {
    right = Math.max(right, (slots[i].x + widths[i] / 2) * scaling);
    // The game checks the right edge only — its player is always inside the
    // frame. On a stage narrower than the scene (a phone in portrait) the
    // player can stand past the LEFT edge, so that side is held to too.
    left = Math.max(left, -(slots[i].x - widths[i] / 2) * scaling);
  }
  const reach = Math.max(right, left) + 15;
  return reach > visibleHalfW && reach > 0 ? visibleHalfW / reach : 1;
}

/** Maps scene px onto the stage box. Built once per stage measurement. */
export interface SceneFit {
  /** stage px per scene px. */
  k: number;
  /** Stage x of a scene x measured from the scene CENTRE. */
  x: (sceneDx: number) => number;
  /** Stage y of an absolute scene y. */
  y: (sceneY: number) => number;
  /** Half the scene width the stage can show, in scene px. */
  visibleHalfW: number;
  /** The scene rows the stage shows: `[viewTop, viewTop + viewH]`. */
  viewTop: number;
  viewH: number;
}

/** The centre of the band the stage stands in for — the scene minus the HUD the
 *  workbench draws above it. Extra vertical room opens symmetrically about it. */
const SCENE_VIEW_CY = (HUD_BAND_H + SCENE_H) / 2;

/** When a framed scene is cropped, the share of the lost rows taken off the
 *  TOP. Symmetric (0.5) put the hand's rest line — `HAND_REST_VISIBLE` scene
 *  px up from the stage's bottom edge — right on the ground line, hiding every
 *  HP bar behind the fan; anchoring the bottom (1.0) instead cut the 470px
 *  Ironclad off at the brow. 0.75 keeps ~60 scene px of floor under the feet
 *  clear of the fan at the full `MAX_FRAMING` and still clears that rig. */
const CROP_FROM_TOP = 0.75;

/**
 * Fit the scene to a stage box, the way the game's viewport does it.
 *
 * Godot's `canvas_items` + `expand` stretch scales by the SMALLER of the two
 * axis ratios and reveals extra scene on the axis with room to spare — it never
 * crops. So does this: on the workbench's usual wide stage the height binds and
 * the extra width simply shows more room to the sides (exactly the old
 * fit-by-height behaviour), and on a stage NARROWER than 16:9 the width binds
 * instead, so the fighters keep their full 1920 px of elbow room and the stage
 * shows scene rows above and below the 1080 frame rather than slicing the
 * player and the boss off at the sides.
 *
 * This is the whole of the workbench's resolution handling: everything in the
 * scene is in scene px and goes through `k`, so any stage box — a 4K window, a
 * tablet in landscape, the debug drawer open, a 21:9 monitor — is the same
 * composition at a different size.
 *
 * `framing` (stageZoom.ts) is the one deliberate departure from the game's
 * fit: on a stage too small to read a fitted scene across a table, the scene
 * is drawn `framing`x larger than its fit and CROPPED to the stage. The crop
 * comes mostly off the top (`CROP_FROM_TOP`), the part of the room nothing
 * stands in: the ground line and the HP bars under it keep their distance
 * from the hand's rest line, and the tallest ordinary rigs keep their heads.
 * At `framing` 1 this is exactly the fit above.
 */
export function fitScene(stageW: number, stageH: number, framing: number = 1): SceneFit {
  if (!(stageW > 0) || !(stageH > 0)) {
    return { k: 0, x: () => 0, y: () => 0, visibleHalfW: SCENE_CX, viewTop: HUD_BAND_H, viewH: SCENE_VIEW_H };
  }
  const k = Math.min(stageW / SCENE_W, stageH / SCENE_VIEW_H) * Math.max(1, framing);
  const viewH = stageH / k;
  const viewTop = viewH >= SCENE_VIEW_H
    ? SCENE_VIEW_CY - viewH / 2
    : HUD_BAND_H + (SCENE_VIEW_H - viewH) * CROP_FROM_TOP;
  return {
    k,
    x: (dx: number) => stageW / 2 + dx * k,
    y: (sceneY: number) => (sceneY - viewTop) * k,
    visibleHalfW: stageW / 2 / k,
    viewTop,
    viewH,
  };
}

/** How far above a creature's crown its intent chip reaches, in the px of the
 *  zoom layer the chip is drawn in (`.gs-intent`, game-skin.css): the chip's
 *  box bottom sits `25 - 0.25*size` above `--rig-top`, the box is `size` tall
 *  (58px for a numberless intent), and the dark disc under it overhangs the box
 *  by 34% of `size` — 25 + 1.09 * 58 ≈ 88px — plus a few px so the disc does
 *  not kiss the stage's edge. The crown it hangs off is the rig's measured
 *  visible top, which `renderedSize().top` (the rig's idle box) matches to
 *  within a few px. */
export const INTENT_HEADROOM_PX = 96;

/**
 * The framing a FIGHT can take: `framing` (stageZoom.ts's compact framing),
 * pulled back as far as it takes to keep the intent chip over the fight's
 * tallest enemy on the stage.
 *
 * The compact framing crops the scene mostly off the top — the part of the
 * room nothing stands in for the Ironclad and a Cultist, and the part the
 * Bygone Effigy's scythe and the chip telling you what it will do for are in
 * for a 588 scene-px statue. At the full `MAX_FRAMING` on a tablet the stage
 * showed scene rows 269 and down; the Effigy's crown is at 249 (through its
 * own 0.88 camera) and the chip hangs ~90 layer px above that, so the number
 * you most need to read was the one thing off the screen. Some twenty
 * creatures' crowns sit above row 346, so this is not one boss's problem.
 *
 * `crownY` is the scene y (absolute, encounter camera already applied) of the
 * highest crown an intent hangs over. The answer is the largest framing in
 * `[1, framing]` at which `fitScene` puts that crown at least `headroomPx`
 * below the stage's top edge — never below 1, which is the game's own frame
 * as the game shows it, and `framing` itself whenever the chip already fits
 * (every ordinary fight on a tablet, and every fight on a monitor, where the
 * framing is 1 to begin with). Floored to 0.01 so a resize drag does not
 * re-fit the scene for every pixel, and so the flooring can only ADD room.
 *
 * Only the scene is pulled back — the fighters and the room, exactly what the
 * game's own per-encounter `SceneContainer.Scale` zooms. The cards, the orb
 * and the End Turn plate keep the stage's zoom, so a hand reads the same
 * across a table whichever monster is standing behind it.
 */
export function sceneFramingFor(
  stageW: number, stageH: number, framing: number, crownY: number | null | undefined,
  headroomPx: number = INTENT_HEADROOM_PX,
): number {
  if (!(framing > 1) || crownY == null || !Number.isFinite(crownY) || !(stageW > 0) || !(stageH > 0)) {
    return Math.max(1, framing || 1);
  }
  const room = (f: number) => fitScene(stageW, stageH, f).y(crownY);
  if (room(framing) >= headroomPx) return framing;
  if (room(1) < headroomPx) return 1;
  // `room` falls as the framing rises (a larger framing crops more off the
  // top and magnifies what is left), so the largest framing that still leaves
  // `headroomPx` is a bisection away.
  let lo = 1;
  let hi = framing;
  for (let i = 0; i < 40 && hi - lo > 1e-4; i++) {
    const mid = (lo + hi) / 2;
    if (room(mid) >= headroomPx) lo = mid; else hi = mid;
  }
  return Math.max(1, Math.floor(lo * 100) / 100);
}

/** The lowest foot line among `slotYs` (absolute scene y, before the camera)
 *  and the ground line itself — where the player stands (`placePlayer`), with
 *  Osty and every row slot at or above it. Authored encounter markers can sit
 *  below it (`encounterSlots.data.ts`: up to 756 against the ground's 740). */
export function lowestFootY(slotYs: readonly number[]): number {
  return Math.max(GROUND_Y, ...slotYs);
}

/**
 * The height to fit the scene into so the ground line `groundY` (scene y,
 * camera applied) lands no lower than `maxGroundY` px down the stage.
 *
 * A phone needs room UNDER the fighters that the game's frame does not leave:
 * a 96px hand strip, and above it the HP bar, nameplate and up to two rows of
 * 44px power chips that must stay tappable — about 230 real px under the feet,
 * on a 347-384px landscape stage. So the scene is fitted into the stage's top
 * part and the room floor fills the rest. In landscape (the fit is bound by
 * the height) that also draws the fighters smaller; in portrait (bound by the
 * width) it only lifts them, into the empty sky.
 *
 * Returns `stageH` whenever the ground already clears the limit (every stage
 * that is not a phone), otherwise the largest height that does, floored to a
 * whole px. The framing is re-derived at each height through
 * `sceneFramingFor`, exactly as the caller will, so the crown keeps its
 * headroom.
 */
export function sceneFitHeight(
  stageW: number, stageH: number, framing: number, crownY: number | null | undefined,
  groundY: number, maxGroundY: number,
): number {
  const groundAt = (h: number) => fitScene(stageW, h, sceneFramingFor(stageW, h, framing, crownY)).y(groundY);
  if (!(stageW > 0) || !(stageH > 0) || groundAt(stageH) <= maxGroundY) return stageH;
  // The ground line is inside the box it is fitted into, so a box no taller
  // than the limit always clears it.
  let lo = Math.max(1, Math.min(stageH, maxGroundY));
  let hi = stageH;
  for (let i = 0; i < 40 && hi - lo > 0.25; i++) {
    const mid = (lo + hi) / 2;
    if (groundAt(mid) <= maxGroundY) lo = mid; else hi = mid;
  }
  return Math.floor(lo);
}

/**
 * The scale to draw the room canvas at so it still covers the stage.
 *
 * `sceneScale` is the size the ROOM itself authors its canvas at
 * (`RoomScene.scale`, 1.44 or 1.50078 — see `roomScenes.ts`), which covers the
 * 1920x1080 frame with room to spare and is what ships at every ordinary size.
 * But the canvas is a finite painting, so a stage far wider or squarer than the
 * game's own frame can reach past its edge, and a room that stops before the
 * stage does is a black band, not a framing choice. Scale up to cover when that
 * happens; the canvas stays centred on the scene either way.
 *
 * `centreY` / `pxPerScene` are the canvas's placement on the stage as Board
 * computes it (both already carry the encounter camera).
 */
export function roomCanvasScale(
  stageW: number,
  stageH: number,
  centreY: number,
  pxPerScene: number,
  sceneScale: number = ROOM_CANVAS_SCALE,
): number {
  if (!(pxPerScene > 0)) return sceneScale;
  const needW = stageW / (ROOM_CANVAS_W * pxPerScene);
  // The canvas is centred on `centreY`, so covering the stage needs twice the
  // larger of the two overhangs.
  const needH = (2 * Math.max(centreY, stageH - centreY)) / (ROOM_CANVAS_H * pxPerScene);
  return Math.max(sceneScale, needW, needH);
}
