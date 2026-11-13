import { useState, type ComponentProps } from "react";
import type { Enemy } from "../api";
import type { DissolveEffect } from "../effects/manifest";
import Combatant from "./Combatant";
import KaiserCrab, { type ArmAction, type CrabMeasure } from "./KaiserCrab";
import { partHpWidth, type PartChrome } from "./partChrome";

/**
 * One targetable arm of the Kaiser Crab — the per-enemy render info Board would
 * otherwise hand straight to a <Combatant>, plus which `part` (animation-track
 * family) of the shared rig this arm drives.
 */
export interface CrabArm {
  c: Enemy;
  part: string; // "left" | "right"
  dataIndex: number;
  /** The board label (`enemyLabeller`) — see `<Combatant label>`. */
  label?: string;
  hotkey?: number;
  onClick?: () => void;
  onHoverChange?: (hovering: boolean) => void;
  animation?: string;
  actionAnim: string; // "attack" | "hurt" | "die"
  /** Explicit clip names for the one-shot, most specific first (rigs whose
   *  clips are suffixed per revive form). Unused by the shared-rig groups,
   *  which route their one-shot onto a named track instead. */
  actionAnimCandidates?: string[];
  actionNonce?: number;
  holdLast: boolean;
  dying: boolean;
  /** The WebGL dissolve (Task 14). **Always `null` for a Kaiser Crab arm**, and
   *  that is the game's own answer rather than a gap: `Crusher.cs:57` and
   *  `Rocket.cs:37` both override `ShouldFadeAfterDeath => false`, so
   *  `NCreature.cs:1106` never constructs an `NMonsterDeathVfx` for either one.
   *  There is no dissolve to give the shared body.
   *
   *  An earlier version of this doc called the missing body dissolve a
   *  not-yet-implemented consumer, and GH #633 was opened to build a
   *  group-level one. Reading the decomp retired both: what the crab actually
   *  gets is `NKaiserCrabBossExplosionVfx`, a particle burst fired off the
   *  shared rig's own `left_embers_start` spine event, which is a different
   *  effect and a different mechanism.
   *
   *  The prop stays because `DecimillipedeGroup` shares this descriptor type
   *  and because a future crab whose arms DO fade would need no plumbing
   *  change. What paces the arm chrome's fade is `deathAnimSeconds` below. */
  dissolve?: { effect: DissolveEffect; rev: string; startedAt: number } | null;
  /** `DeathAnimLengthOverride` (`Crusher.cs:61` / `Rocket.cs:41`: `2.5f`), the
   *  wait `NCreature.cs:1093` takes for a creature with no `_spineAnimator` of
   *  its own — which is exactly these arms, whose body is on the shared rig.
   *  It is what the chrome-only overlay's fade follows now that `dissolve` is
   *  correctly null; before GH #633 the same 2.5s arrived via the dissolve
   *  CEILING, which agreed by coincidence rather than by derivation. */
  deathAnimSeconds?: number | null;
  effects?: Parameters<typeof Combatant>[0]["effects"];
}

/**
 * Renders the Kaiser Crab boss as a SINGLE shared rig with one targetable
 * overlay per arm, instead of a whole crab per combatant. The rig (body + both
 * claws) is drawn once, behind; each arm's chrome (intent / block / HP / name /
 * powers / click hitbox) is a spriteless <Combatant> pinned over its claw. Each
 * arm's one-shot (attack/hurt/die) is routed to the matching `left/*` or
 * `right/*` animation track on the shared rig.
 */
/** Width of an arm's readout column (HP bar / name / intent), px. The two claws
 *  are at opposite edges of the screen, so nothing here has to squeeze. */
const OVERLAY_W = 190;
/** How far above a combatant's body top its intent chip's centre sits, px
 *  (.gs-intent in game-skin.css: box bottom 25 - 0.25 * 58 over the top, the box
 *  58 tall — 10.5 + 29). */
const INTENT_CHIP_RISE = 40;

/** Where one arm stands, in px from the rig's root (Board: crabMount.ts
 *  through the stage fit). */
export interface ArmAt {
  /** Its feet. */
  x: number;
  y: number;
  /** HP bar width, px. */
  hpW: number;
  /** `IntentPos` height above the feet, px. */
  intentH: number;
}

export default function KaiserCrabGroup({
  rig,
  pxPerUnit,
  armAt,
  clampX,
  arms,
  locStrings,
  onRigEvent,
  split,
}: {
  rig: string;
  /** Rendered px per rig unit — the crab's authored scene scale (0.5) times the
   *  stage's scene factor and the encounter camera. The crab is NOT fitted to a
   *  box: at its authored scale it is wider than the scene, which is the point
   *  (see KaiserCrab). */
  pxPerUnit: number;
  /** Each arm's own creature node, by `part`.
   *
   *  The crab rig and the two arm creatures are separate things in the game:
   *  the rig hangs off the combat-room background, while Crusher and Rocket are
   *  ordinary creatures standing on the encounter's markers, each with its own
   *  HP bar (its `Bounds` width) and intent (at its `IntentPos`). So a
   *  readout stands where its creature does, not on the claw it moves. */
  armAt: Record<string, ArmAt | undefined>;
  /** The stage's x-extent in px from the rig's root: a readout is kept inside
   *  it, so on a stage narrower than the game's frame an arm whose marker
   *  falls off the edge still has a legible HP bar. */
  clampX: { min: number; max: number };
  arms: CrabArm[];
  /** Forwarded straight to each arm's `<Combatant>` — see that prop's own
   *  doc. Board.tsx holds the manifest; this group has no fetch of its own. */
  locStrings?: Record<string, string>;
  /** Forwarded straight to `<KaiserCrab>` — see that prop's doc. Passed
   *  through rather than handled here because the effect sink lives in
   *  Board.tsx, the same place every other `sink.play` call is made. */
  onRigEvent?: ComponentProps<typeof KaiserCrab>["onRigEvent"];
  /** Phone portrait: each arm's split power chrome (`PartChrome`). */
  split?: PartChrome;
}) {
  const [m, setM] = useState<CrabMeasure | null>(null);

  const armAction = (part: string): ArmAction | undefined => {
    const arm = arms.find((a) => a.part === part);
    if (!arm || arm.actionNonce == null) return undefined;
    return { nonce: arm.actionNonce, clip: arm.actionAnim, hold: arm.holdLast };
  };

  // The group is a ZERO-SIZE anchor on the rig's root, which the parent places;
  // the canvas and both arm readouts hang off it in px measured from that root.
  // Nothing here has a box for the crab to be fitted into — it is drawn at its
  // authored size and overflows the room on purpose.
  const fallback: Record<string, number> = { left: -600, right: 600 };

  return (
    <div className="gs-crab-group" style={{ position: "relative", width: 0, height: 0 }}>
      <KaiserCrab
        rigBase={rig}
        pxPerUnit={pxPerUnit}
        leftAction={armAction("left")}
        rightAction={armAction("right")}
        onMeasure={setM}
        onRigEvent={onRigEvent}
      />
      {arms.map((arm) => {
        const at = armAt[arm.part];
        const box = m?.claw[arm.part];
        const hpW = partHpWidth(at?.hpW, arm.c.max_hp, split);
        const w = Math.max(OVERLAY_W, hpW ?? 0);
        const cx = at?.x ?? box?.cx ?? fallback[arm.part] ?? 0;
        // Keep the readout on screen when its marker is not — the game's own
        // frame always shows both, a narrower stage may not.
        // In the split, the arm's power rows span its whole slot: that
        // stays on screen too.
        const margin = Math.max(w, split?.slot ?? 0) / 2 + 12;
        const x = Math.max(clampX.min + margin, Math.min(clampX.max - margin, cx));
        // The spacer is the creature's body: its feet up to its IntentPos, so
        // the HP bar sits under the feet and the intent where the game hangs it.
        // The game centres the intent ON IntentPos, where our chip hangs wholly
        // above the spacer (.gs-intent: its centre ~40px over the top), so the
        // spacer stops that much short.
        const spacerH = Math.max(64, Math.round((at?.intentH ?? 360) - INTENT_CHIP_RISE));
        const feetY = at?.y ?? 0;
        return (
          <div key={arm.dataIndex} data-crab-arm={arm.part}
            data-split-slot={split ? split.slotReal.toFixed(1) : undefined}
            style={{ position: "absolute", left: x, top: feetY - spacerH,
              transform: "translateX(-50%)",
              // Hidden (and untargetable) until the rig has been measured, then
              // faded in with it — SpineSprite-backed combatants hold their own
              // chrome back the same way while their rig loads. Painted against
              // the fallback claw split first, each arm's HP bar, name and intent
              // sits ~200px from where its claw actually is; this rig is four
              // atlas pages / ~5 MB, so that wrong-place phase lasts about a
              // second and reads as the boss shifting.
              pointerEvents: m ? "auto" : "none",
              opacity: m ? 1 : 0, transition: "opacity 140ms ease-out" }}>
            <Combatant
              c={arm.c}
              isEnemy
              dataIndex={arm.dataIndex}
              label={arm.label}
              noSprite={{ w, h: spacerH }}
              hpWidth={hpW}
              powerChrome={split?.chrome}
              powerCells={split?.cells}
              hotkey={arm.hotkey}
              onClick={arm.onClick}
              onHoverChange={arm.onHoverChange}
              dying={arm.dying}
              // Syncs THIS arm's chrome (HP/name/intent) fade to how long the
              // corpse actually lives — `Board.tsx`'s removal timer reads the
              // same two values in the same order. `dissolve` is always null
              // for a crab arm (see its doc above), so in practice it is
              // `deathAnimSeconds`'s 2.5s doing the work; without either, the
              // chrome fell back to `DEFAULT_DYING_FADE_SECONDS` (0.9s) and
              // vanished ~1.6s before the corpse was actually removed.
              dissolve={arm.dissolve}
              deathAnimSeconds={arm.deathAnimSeconds}
              effects={arm.effects}
              locStrings={locStrings}
            />
          </div>
        );
      })}
    </div>
  );
}
