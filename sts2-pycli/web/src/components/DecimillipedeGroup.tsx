import Combatant from "./Combatant";
import SpineSprite from "./SpineSprite";
import type { CrabArm } from "./KaiserCrabGroup";
import { partHpWidth, type PartChrome } from "./partChrome";
import { mirroredBox } from "./decimillipedeLayout";

/**
 * Renders the Decimillipede elite as its real connected body: three segment rigs
 * (front / middle / back), each an independently targetable combatant, drawn at
 * the authored scene scale from ONE shared root.
 *
 * The three rigs are authored as pieces of a single animal around a common
 * origin — measured in rig units, `back` lies entirely LEFT of its root
 * (-1307..-290), `middle` straddles it (-514..+308) and `front` — the head — is
 * almost entirely RIGHT of it (-1058..+1240), with all three sharing a
 * baseline. Stacked on one point they tile into one 2547-unit millipede with
 * overlapping seams; that is why the game's encounter scene can park all three
 * slots together, and why the workbench's old "space them out, then overlap by
 * 18%" layout had to invent a spacing constant to hide the seams.
 *
 * They are mirrored (`flipX`): the game renders the segments with a negative X
 * scale (`creature_scales` has -0.45 for all three), which is what turns the
 * head to face the player standing on the left.
 *
 * Sizing is the same rule as every other creature — `rig_scale` px per rig unit
 * in scene px (see src/scene.ts). The old layout drew them at less than half
 * that, which is why the elite read as three small grubs.
 */
/** Authored scene scale for every segment (`creature_scales`, magnitude of the
 *  shared -0.45). The three share it so the body tube keeps one diameter. */
export const SEGMENT_SCALE = 0.45;

/** Draw order, back of the animal first, so the head sits nearest the camera. */
const ORDER = ["back", "middle", "front"];

export default function DecimillipedeGroup({ segments, pxPerScene = 1, locStrings, split, drop = 0 }: {
  segments: CrabArm[];
  /** Phone portrait: each segment's split power chrome (`PartChrome`). */
  split?: PartChrome;
  /** Phone portrait: stage px the legs reach below the shared root. The
   *  HP bars, names and powers hang under the legs there, not across them. */
  drop?: number;
  /** Stage px per SCENE px (`scene.fitScene().k`). */
  pxPerScene?: number;
  /** Forwarded straight to each segment's `<Combatant>` — see that prop's
   *  own doc. Board.tsx holds the manifest; this group has no fetch of its
   *  own. */
  locStrings?: Record<string, string>;
}) {
  const ordered = [...segments].sort(
    (a, b) => (ORDER.indexOf(a.part) + 1 || 99) - (ORDER.indexOf(b.part) + 1 || 99),
  );
  // px per rig unit — the authored scale carried into this stage's pixels.
  const k = SEGMENT_SCALE * pxPerScene;

  return (
    // A ZERO-SIZE anchor on the shared root, which Board pins to the scene's
    // ground line; every segment and overlay hangs off it in px.
    <div className="gs-decimillipede" style={{ position: "relative", width: 0, height: 0 }}>
      {ordered.map((s, i) => {
        const b = mirroredBox(s.c.rig);
        return (
          <div key={`rig-${s.dataIndex}`}
            // The sprite box centres the rig's ROOT (see SpineSprite), so the
            // wrapper's own centre goes on the shared root.
            style={{ position: "absolute", left: 0, bottom: 0, zIndex: i,
              transform: "translateX(-50%)" }}>
            <SpineSprite
              rigBase={s.c.rig as string}
              flipX
              worldScale={SEGMENT_SCALE}
              pxPerScene={pxPerScene}
              boundsH={b ? (b.bottom + b.h) * SEGMENT_SCALE : undefined}
              fallback={s.c.image}
              alt={s.c.name}
              actionAnim={s.actionAnim}
              actionNonce={s.actionNonce}
              holdLast={s.holdLast}
              dissolve={s.dissolve}
            />
          </div>
        );
      })}
      {/* Targetable overlays (in front): HP / intent / block / name / click,
          each over the middle of its own segment. */}
      {ordered.map((s, i) => {
        const b = mirroredBox(s.c.rig);
        const cx = b ? ((b.left + b.right) / 2) * k : 0;
        const spacerH = Math.max(48, Math.round((b ? b.h + b.bottom : 400) * k)) + drop;
        // Keep neighbouring readouts from colliding: a segment overlay is at
        // most as wide as the gap to the next segment's centre.
        const overlayW = Math.max(90, Math.min(170, Math.abs(b ? b.w * k : 150) * 0.5));
        return (
          <div key={`ov-${s.dataIndex}`}
            data-split-slot={split ? split.slotReal.toFixed(1) : undefined}
            style={{ position: "absolute", left: cx, bottom: -drop, transform: "translateX(-50%)",
              zIndex: 10 + i }}>
            <Combatant
              c={s.c}
              isEnemy
              dataIndex={s.dataIndex}
              label={s.label}
              noSprite={{ w: overlayW, h: spacerH }}
              hpWidth={partHpWidth(undefined, s.c.max_hp, split)}
              powerChrome={split?.chrome}
              powerCells={split?.cells}
              hotkey={s.hotkey}
              onClick={s.onClick}
              onHoverChange={s.onHoverChange}
              dying={s.dying}
              // Syncs this overlay's chrome (HP/name/intent) fade to whatever
              // paces the corpse — the same two values `Board.tsx`'s removal
              // timer reads, in the same order. Without it the chrome fell back
              // to `DEFAULT_DYING_FADE_SECONDS` (0.9s) and vanished while the
              // body was still visible.
              //
              // Since GH #633 a segment's `dissolve` is always null:
              // `DecimillipedeSegment.cs:91` overrides
              // `ShouldFadeAfterDeath => false`, so the game constructs no
              // `NMonsterDeathVfx` for one and the threshold dissolve this
              // group used to run was never the game's behaviour.
              dissolve={s.dissolve}
              deathAnimSeconds={s.deathAnimSeconds}
              effects={s.effects}
              locStrings={locStrings}
            />
          </div>
        );
      })}
    </div>
  );
}
