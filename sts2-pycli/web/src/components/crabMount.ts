/**
 * The Kaiser Crab as the game mounts it, read off the v0.107.1 PCK.
 *
 * The crab is not a creature in the row: `KaiserCrabBossBackground`
 * (`scenes/backgrounds/kaiser_crab_boss/kaiser_crab_boss_background.tscn`) is the
 * fight's combat-room background, and it instances the crab rig as a child:
 *
 *     [node name="KaiserCrab" ... instance=kaiser_crab_boss_setup.tscn]
 *     position = Vector2(-1029, -544)
 *
 * and the setup scene (`scenes/creature_visuals/kaiser_crab_boss_setup.tscn`)
 * puts the skeleton at
 *
 *     [node name="Visuals" type="SpineSprite"]
 *     position = Vector2(979, 450)
 *     scale = Vector2(0.55, 0.55)
 *
 * So the rig's root is (-50, -94) from the background's origin, drawn at 0.55
 * scene px per rig unit — NOT the 0.5 of `creature_visuals/kaiser_crab_boss.tscn`
 * the wire's `rig_scale` comes from (that scene is the bestiary's). The
 * background's origin is `%BgContainer`'s, which hangs `ROOM_BG_OFFSET_X` right
 * of the scene's centre line at mid-height; the encounter camera (0.75, Down*35)
 * zooms it with everything else.
 *
 * Drawing the crab from this, rather than flushing the rig's top to the stage
 * top as the workbench used to, is what puts its face where the game has it
 * (scene row ~326 through the camera, not ~480) and at the game's size, on any
 * stage: everything here is scene px through `fitScene`.
 *
 * The two arms, Crusher and Rocket, are ordinary creatures standing on the
 * encounter's markers (`scenes/encounters/kaiser_crab_boss.tscn`, in
 * `encounterSlots.data.ts`), each with its own `creature_visuals` scene: the
 * HP bar is its `Bounds` width + 24 (`NHealthBar.UpdateLayoutForCreatureBounds`)
 * and the intent hangs at its `IntentPos`.
 */
import { ROOM_BG_OFFSET_X, SCENE_CY } from "../scene";
import { ENCOUNTER_SLOTS } from "./encounterSlots.data";

/** Scene px per rig unit (`kaiser_crab_boss_setup.tscn` Visuals.scale). */
export const CRAB_SETUP_SCALE = 0.55;
/** The rig root's x from the scene's centre line, in scene px. */
export const CRAB_ROOT_DX = ROOM_BG_OFFSET_X + (-1029 + 979);
/** The rig root's absolute scene y. */
export const CRAB_ROOT_Y = SCENE_CY + (-544 + 450);

/** `NHealthBar`: the bar is the creature's Bounds width plus this (less the
 *  monster's `HpBarSizeReduction`, which neither arm sets). */
const HP_BAR_PAD = 24;

export interface CrabArmMount {
  /** Feet: x from the scene's centre line, absolute y — scene px, pre-camera. */
  dx: number;
  y: number;
  /** Centre of the creature's `Bounds`, from its feet (the bar centres on it). */
  boundsCx: number;
  /** HP bar width, scene px. */
  hpW: number;
  /** `IntentPos` height above the feet, scene px (positive = up). */
  intentH: number;
}

/** Each arm's creature scene: `Bounds` x-extent and `IntentPos` y. */
const ARM_VISUALS: Record<string, { slot: string; left: number; right: number; intentY: number }> = {
  // creature_visuals/crusher.tscn: Bounds -220..237, IntentPos (0, -580)
  left: { slot: "crusher", left: -220, right: 237, intentY: -580 },
  // creature_visuals/rocket.tscn: Bounds -205..226, IntentPos (1, -450)
  right: { slot: "rocket", left: -205, right: 226, intentY: -450 },
};

/** Where the arm on `part` ("left" = Crusher, "right" = Rocket) stands. */
export function crabArmMount(part: string): CrabArmMount | null {
  const v = ARM_VISUALS[part];
  const slot = ENCOUNTER_SLOTS.kaiser_crab_boss?.slots.find((s) => s.name === v?.slot);
  if (!v || !slot) return null;
  return {
    dx: slot.x,
    y: slot.y,
    boundsCx: (v.left + v.right) / 2,
    hpW: v.right - v.left + HP_BAR_PAD,
    intentH: -v.intentY,
  };
}
