import type React from "react";
import type { Orb, Power } from "../api";
import { powerItem } from "../inspect/powerItem";
import { useInspect } from "../inspect/useInspect";
import HealthBar from "./HealthBar";
import Orbs from "./Orbs";
import Powers from "./Powers";
import type { CombatantEffects, PlayerLike } from "./Combatant";
import { BAND_CELL, BAND_ICON, BAND_SHIELD_HIT, bandOrbRow, bandPowerCells, bandPowerCellsBesideEnergy } from "./playerBandMetrics";

/** The phone-portrait player's status block, drawn beside the player figure:
 *  line 1 is the HP bar (with its block shield on the left end), line 2 the
 *  orbs, then ONE row of power cells that collapses its tail into a "+N" chip.
 *  Laid out in REAL px: the caller supplies the 1/k counter-scale. No panel,
 *  no border — it stands on the scene like a creature's own bar.
 *
 *  The block shield is centred on the bar's left end, so it OVERHANGS the
 *  status block's left edge by 17 px (half its 34 px), and its 44 px tap area
 *  by 22 px. The host must leave that room to the left (between the figure
 *  and the bar) and must not clip it (`overflow: hidden` on any ancestor).
 *
 *  The orbs (a Defect's up to ten) stay on ONE row no wider than `widthPx`:
 *  they shrink from 30 px to 20 px, then their gaps close (`bandOrbRow`). */
export default function PlayerBand({ player, powers, widthPx, orbs, orbSlots, effects, energy }: {
  player: PlayerLike; powers: Power[]; widthPx: number;
  /** While a card is zoomed: the energy orb, which the centred card covers
   *  where it normally sits. Drawn at the right end of the block's last
   *  line; the orbs then take a line of their own. */
  energy?: React.ReactNode;
  orbs?: Orb[]; orbSlots?: number; effects?: CombatantEffects;
  /** `VfxManifest.locStrings`, the slice `Combatant` takes; nothing in the
   *  band prints VFX text yet. */
  locStrings?: Record<string, string>;
}) {
  const inspect = useInspect();
  const inspectId = `player${player.name}`;
  const orbCount = Math.max(orbSlots ?? 0, orbs?.length ?? 0);
  const cells = energy ? bandPowerCellsBesideEnergy(widthPx) : bandPowerCells(widthPx, orbCount);
  const orbRow = bandOrbRow(widthPx, orbCount, !!energy);
  return (
    <div className="gs-playerband-status"
      style={{ width: widthPx, "--gs-band-orb-gap": `${orbRow.gap}px` } as React.CSSProperties}>
      <HealthBar c={player} widthPx={widthPx} effects={effects} inspectId={inspectId} badgeFor="player"
        shieldHitPx={BAND_SHIELD_HIT} />
      {orbCount > 0 && <Orbs orbs={orbs} slots={orbSlots} size={orbRow.size} row />}
      {powers.length > 0 && (
        <div className="gs-powers">
          <Powers powers={powers} size={BAND_ICON} hitSize={BAND_CELL} cells={cells} maxShown={cells} minShown={1}
            inspectScope={inspectId}
            onOverflow={(hidden) => inspect.open({ key: `powers:${inspectId}`, name: "More powers",
              list: hidden.map((p) => powerItem(p, inspectId)) })} />
        </div>
      )}
      {energy && <div className="gs-band-energy">{energy}</div>}
    </div>
  );
}
