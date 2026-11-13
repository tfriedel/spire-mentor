import { useState } from "react";
import { assetUrl } from "../api";
import { useInspect } from "../inspect/useInspect";
import { inspectTrigger } from "../inspect/trigger";
import { doomAmount } from "./combatantMetrics";
import type { CombatantEffects, PlayerLike } from "./Combatant";

/** A creature's health bar: the chevron bar, its numerals, the Doom band and
 *  the block shield. `Combatant` draws it under every creature's feet; the
 *  phone-portrait `PlayerBand` draws the player's on its own.
 *
 *  `widthPx` is the bar's length (`--gs-hpw`). `badgeFor` names whose shield
 *  this is (`data-block-badge-for`), for a reader that has to tell the band's
 *  shield from a creature's; absent, the attribute is not rendered at all, so
 *  `Combatant`'s markup is the one it always had.
 *
 *  `shieldHitPx` gives the shield a square tap area that big, centred on it
 *  and invisible (`.gs-block-hit`), for a shield drawn smaller than the 44 px
 *  tap floor; a tap on it is a tap on the shield. Absent, no such element. */
export default function HealthBar({ c, widthPx, effects, inspectId, badgeFor, shieldHitPx }: {
  c: PlayerLike; widthPx: number; effects?: CombatantEffects; inspectId: string; badgeFor?: "player";
  shieldHitPx?: number;
}) {
  const inspect = useInspect();
  const pct = Math.max(0, Math.min(100, (c.hp / c.max_hp) * 100));
  // Doom: the kill line painted on the bar (see doomAmount). A lethal Doom
  // takes the whole remaining bar; otherwise the band stops at doom/maxHp but
  // never runs past the live fill (a Doom above current HP IS lethal, and a
  // band overhanging an empty track would read as phantom health). The 4% floor
  // mirrors the game's own 12px minimum width, scaled to our shorter bar, so a
  // 1-point Doom on a 200 HP boss is still visible.
  const doom = c.hp > 0 ? doomAmount(c.powers) : 0;
  const doomLethal = doom > 0 && doom >= c.hp;
  const doomPct = doomLethal ? pct : Math.min(pct, Math.max(4, (doom / c.max_hp) * 100));
  const blocked = c.block > 0;
  // `AnimateInBlock` returns early unless `oldBlock == 0`: only a shield that
  // was not showing drops in; topping up an existing one just changes the
  // number. `blockPop` fires on every gain, so remember which kind this was
  // (React's "state from the previous render" pattern, no effect lag).
  const [prevBlock, setPrevBlock] = useState(c.block);
  const [blockFromZero, setBlockFromZero] = useState(false);
  if (c.block !== prevBlock) {
    setPrevBlock(c.block);
    setBlockFromZero(prevBlock <= 0 && c.block > 0);
  }
  return (
    /* Three layers so the chevron silhouette survives: frame (teal edge) ->
       track (dark) -> fill, each with its own clip-path; the numerals ride on
       top and deliberately overflow the bar (see .gs-hptext).

       `gs-hp-blocked`: NHealthBar.RefreshBlockUi — while the creature has
       block the bar shows `%BlockOutline`, the fill is modulated to `#3B6FA3`
       and the HP numerals take the `#1B3045` outline (RefreshText). */
    <div className={`gs-hpwrap${blocked ? " gs-hp-blocked" : ""}`}
      style={{ "--gs-hpw": `${widthPx}px` } as React.CSSProperties}>
      <div className="gs-hpbar">
        <div className="gs-hptrack">
          {/* Doom band. The game draws it from the bar's LOW end out to
              `doom/maxHp` of the track, under the red fill, so the stretch of
              health that will not save you is a different colour; once Doom
              reaches the current HP the red fill is hidden outright and the
              whole remaining bar is the Doom colour (NHealthBar
              .RefreshForeground). Same here.

              The lethality test upstream is `doom >= hp - poisonNextTurn`.
              Poison's next-turn total is a hook-modified sum the wire does
              not carry (PoisonPower.CalculateTotalDamageNextTurn), so this
              uses the poison-free form: exact whenever the creature is not
              poisoned, and one turn late when it is. */}
          {doom > 0 && (
            <div className={`gs-hpdoom${doomLethal ? " gs-hpdoom-lethal" : ""}`}
              style={{ width: `${doomPct}%` }} />
          )}
          {!doomLethal && <div className="gs-hpfill" style={{ width: `${pct}%` }} />}
        </div>
      </div>
      <span className={`gs-hptext${doomLethal ? " gs-hptext-doomed" : ""}`}>{c.hp}/{c.max_hp}</span>
      {/* `%BlockContainer`: the game's shield (ui/combat/block.png) with the
          count on it, centred on the bar's LEFT end (`SetCreatureBounds`
          puts its centre on the creature box's left edge, which is where
          the bar starts) — part of the health bar, not floating by the head.

          `data-block-badge`: `VfxLayer`'s anchor for `vfx_block_broken`
          (`NHealthBar.cs:575`: `_blockContainer.GlobalPosition +
          _blockContainer.Size * 0.5`). ALWAYS MOUNTED — never gated on
          `c.block > 0` — and only its NUMBER/visibility toggle with it.
          That mirrors Godot's own `Control.Visible`: a hidden `Control`
          stays in the scene tree with a valid `GlobalPosition`, it just
          doesn't draw. It is also load-bearing for TIMING: the `blockBreak`
          event that plays `vfx_block_broken` fires from a `useEffect` that
          runs AFTER `c.block` has already committed to 0 in this same
          render pass — gating the DIV itself on `c.block > 0` would unmount
          the anchor before the VFX ever got a position to query (verified: a
          `blockShatter`-flag gate looked like a fix but READS THE SAME
          POST-COMMIT DOM, one render too late). Keeping the container
          always-present and toggling only its CONTENTS/opacity sidesteps the
          race entirely.

          `gs-block-pop` is `AnimateInBlock`: on a 0 -> n gain the shield
          fades in while dropping 20px into place (0.5 s, Back ease). */}
      <div
        className={`gs-block${effects?.blockPop && blockFromZero ? " gs-block-pop" : ""}`}
        style={blocked ? undefined : { opacity: 0, pointerEvents: "none" }}
        title={`block ${c.block}`}
        data-block-badge="true"
        data-block-badge-for={badgeFor}
        {...inspectTrigger(inspect.enabled && blocked, `Block ${c.block}`, () => {
          inspect.open({ key: `block:${inspectId}`, icon: assetUrl("ui/combat/block.png"), name: "Block",
            amount: c.block, description: "Absorbs damage before it reaches HP." });
        })}
      >
        <img className="gs-block-icon" src={assetUrl("ui/combat/block.png")} alt="" aria-hidden="true" draggable={false} />
        {blocked ? <span className="gs-block-num">{c.block}</span> : null}
        {shieldHitPx != null && (
          <span className="gs-block-hit" aria-hidden="true" style={{ position: "absolute", left: "50%", top: "50%",
            width: shieldHitPx, height: shieldHitPx, transform: "translate(-50%, -50%)" }} />
        )}
      </div>
      {effects?.blockShatter && <div key={`bs${effects.nonce}`} className="gs-block-shatter" />}
    </div>
  );
}
