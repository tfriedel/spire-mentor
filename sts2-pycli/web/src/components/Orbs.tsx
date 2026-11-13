import { useState, type CSSProperties } from "react";
import { assetUrl, type Orb } from "../api";
import { orbArc, GAME_BODY_H, ORB_TO_BODY } from "./orbArc";

/** The game's empty-slot art: the dashed ring `NOrb` shows while its `Model` is
 *  null (the `%Outline` TextureRect, drawn for the local player only).
 *
 *  Resolved through `assetUrl` like every other art path. A hardcoded
 *  `/assets/...` is only correct when the backend serves the art tree itself;
 *  point `STS2_ASSETS_BASE_URL` at the art bucket (which is what the deployed
 *  workbench does) and it 404s — and a broken `<img>` renders its ALT TEXT,
 *  which is how the words "empty orb slot" came to be wrapped across the belt.
 *  `EmptySlot` below falls back to a drawn ring if the file is missing from the
 *  bucket too, so a slot is never nothing. */
const EMPTY_ORB_ART = "orb_atlas/empty.png";

/** One unfilled slot. The art first, a drawn ring if it doesn't load: the art
 *  tree and the app are deployed separately (`just deploy-art` vs
 *  `just deploy-workbench`), so a bucket that is one deploy behind would
 *  otherwise leave a hole in the belt. */
function EmptySlot({ style }: { style: CSSProperties }) {
  const [missing, setMissing] = useState(false);
  if (missing) return <span className="gs-orb-empty" role="img" aria-label="empty orb slot" style={style} />;
  return (
    <img className="gs-orb-empty-art" src={assetUrl(EMPTY_ORB_ART)} alt="" role="img"
      aria-label="empty orb slot" draggable={false} style={style}
      onError={() => setMissing(true)} />
  );
}

/** The Defect's orb queue: the channelled orbs (each with its per-turn passive
 *  value) followed by empty slots up to the queue's capacity, hung on the arc
 *  the game sweeps around the character. Renders nothing for characters with no
 *  orb slots.
 *
 *  Positions are relative to the container's origin, which `.gs-orbs` pins to
 *  the rig's feet — the creature's root. The arc itself is centred ABOVE that
 *  root (see `ORB_CENTRE_TO_BODY`), the way the game hangs `NOrbManager` off a
 *  marker rather than off the root — see `Combatant`. */
export default function Orbs({ orbs, slots, bodyH, size, row }: {
  orbs?: Orb[];
  /** A plain left-to-right row of `size` px slots (`.gs-orbs-row`) instead of
   *  the arc: the phone-portrait player band, which has no body to hang the
   *  belt around. */
  row?: boolean;
  slots?: number;
  /** The owner's rendered body height in px; the belt scales with it. */
  bodyH?: number | null;
  /** Orb diameter in px. Derived from `bodyH` when omitted. */
  size?: number;
}) {
  const filled = orbs ?? [];
  const capacity = Math.max(slots ?? 0, filled.length);
  if (capacity === 0) return null;
  // An unmeasured (or not-yet-laid-out, or headless) owner falls back to the
  // game's own body height, so the belt comes out at the game's scale rather
  // than collapsing onto a point.
  const body = bodyH && bodyH > 0 ? bodyH : GAME_BODY_H;
  const px = size ?? Math.round(body * ORB_TO_BODY);
  const pts = orbArc(capacity, body);
  const slot = (i: number): CSSProperties => row ? { position: "relative", width: px, height: px, flex: "none" } : ({
    position: "absolute",
    left: pts[i].x,
    top: pts[i].y,
    width: px,
    height: px,
    // The arc gives orb CENTRES, so each slot is centred on its point rather
    // than hung off it by a corner.
    transform: "translate(-50%, -50%)",
  });
  return (
    <div className={`gs-orbs${row ? " gs-orbs-row" : ""}`} data-combatant="orbs">
      {filled.map((o, i) => (
        <div key={o.index} style={slot(i)}
          title={`${o.name}${o.passive != null ? ` — passive ${o.passive}, evoke ${o.evoke}` : ""}`}>
          {o.image ? (
            <img src={o.image} alt={o.name} draggable={false} style={{ width: px, height: px, display: "block" }} />
          ) : (
            <span className="mono" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
              justifyContent: "center", fontSize: 10, borderRadius: "50%",
              border: "1px solid var(--border)", background: "var(--panel)" }}>
              {o.type}
            </span>
          )}
          {/* The per-turn passive value. The game prints it INSIDE the orb (a
              MegaLabel that auto-sizes to the sprite) and hides it outright for
              Plasma, whose value is the energy it will hand you; a corner chip
              is our own affordance, so it tracks the orb's size rather than
              staying a 10px pip on a head-sized sprite. */}
          {o.passive != null && (
            <span className="mono" style={{ position: "absolute", right: -3, bottom: -3,
              fontSize: Math.max(10, Math.round(px * 0.26)), fontWeight: 700,
              lineHeight: 1, padding: "1px 3px", borderRadius: 6, color: "var(--text)",
              background: "var(--bg)", border: "1px solid var(--text)",
              textShadow: "0 1px 2px oklch(0% 0 0 / 0.8)" }}>
              {o.passive}
            </span>
          )}
        </div>
      ))}
      {Array.from({ length: capacity - filled.length }, (_, i) => (
        <EmptySlot key={`empty-${i}`} style={slot(filled.length + i)} />
      ))}
    </div>
  );
}
