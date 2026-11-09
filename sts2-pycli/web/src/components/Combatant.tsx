import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { assetUrl, type Enemy, type Orb, type Power } from "../api";
import type { DissolveEffect } from "../effects/manifest";
import CardFrame from "./CardFrame";
import Orbs from "./Orbs";
import Powers from "./Powers";
import SpineSprite from "./SpineSprite";
import { monsterSkin } from "../monsterSkins";
import { describeIntent, intentDirection } from "../intentBreakdown";
import { boundsWidth, instanceScale } from "../scene";
import { POWER_BOX_MARGIN } from "../powerRow";
import { ANIM_SHAKE_AMPLITUDE, ANIM_SHAKE_DURATION, animShakeKeyframes } from "../effects/animShake";
import {
  nameplateText,
  damageNumberJitter,
  spriteBodyH,
  FALLBACK_BODY_H,
  hpBarWidth,
  doomAmount,
  bodyScale,
  dyingChromeFadeSeconds,
} from "./combatantMetrics";

interface PlayerLike {
  name: string; hp: number; max_hp: number; block: number;
  image?: string | null;
  rig?: string | null;
  rig_scale?: number | null;
  rig_bounds_h?: number | null;
  /** Authored `Visuals.Bounds` WIDTH in scene px — the box the power row is
   *  laid out against (`NPowerContainer.SetCreatureBounds`). */
  rig_bounds_w?: number | null;
  size_jitter?: number | null;
  hue_jitter?: number | null;
  intents?: Enemy["intents"];
  powers?: Power[];
  stolen_cards?: Enemy["stolen_cards"];
}

/** The game's OWN intent art (images/intent_atlas/), for the kinds the server
 *  sends no `intent_image` for. r4 drew these as text glyphs, which is how the
 *  second slime ended up wearing a bare white diamond (❖, the StatusCard
 *  fallback) where the game shows a painted icon — both critics logged it.
 *  `intent_hidden.png` is deliberately absent from this table: that file is
 *  fully transparent in the extract, so it would render as nothing. */
const INTENT_ART: Record<string, string> = {
  Attack: "intent_atlas/attack/intent_attack_3.png",
  Defend: "intent_atlas/intent_defend.png",
  Buff: "intent_atlas/intent_buff.png",
  Debuff: "intent_atlas/intent_debuff.png",
  CardDebuff: "intent_atlas/intent_card_debuff.png",
  StatusCard: "intent_atlas/intent_status_card.png",
  Stun: "intent_atlas/intent_stun.png",
  Sleep: "intent_atlas/intent_sleep.png",
  Escape: "intent_atlas/intent_escape.png",
  Summon: "intent_atlas/intent_summon.png",
  Heal: "intent_atlas/intent_heal.png",
  DeathBlow: "intent_atlas/intent_death_blow.png",
  Unknown: "intent_atlas/intent_unknown.png",
};
// Last resort for a kind with neither a server image nor atlas art: the game's
// own "unknown" disc — never raw enum text in the scene.
const INTENT_ART_FALLBACK = "intent_atlas/intent_unknown.png";

export default function Combatant({ c, isEnemy, dataIndex, onClick, onHoverChange, hotkey, anchor, animation, actionAnim, actionAnimCandidates, actionNonce, holdLast, dying, dissolve, deathAnimSeconds, effects, noSprite, hpWidth, maxBodyH, pxPerScene = 1, orbs, orbSlots, locStrings, label, reticle }: {
  c: PlayerLike; isEnemy: boolean; dataIndex?: number; onClick?: () => void;
  /** The advice's target: the enemy the plan's first targeted move aims at
   *  (advice/marks.ts). Gold targeting brackets around the creature, as the
   *  mod draws a copy of the game's own reticle — it stays up while you decide. */
  reticle?: boolean;
  /** The enemy's board label (`enemyLabeller`): `c.name`, plus `" #k"` when
   *  several live enemies share the name. The nameplate shows the number so
   *  three Two-Tailed Rats say WHICH one; absent = the plain name (the
   *  player, a dying corpse no longer in the live list). */
  label?: string;
  // Hover enter (true) / leave (false) — drives the live damage preview on the
  // pending attack card while the player chooses a target.
  onHoverChange?: (hovering: boolean) => void;
  hotkey?: number; anchor?: string;
  animation?: string; actionAnim?: string; actionNonce?: number;
  /** Explicit one-shot clip names, most specific first — for rigs whose
   *  action clips are suffixed per variant (Test Subject: attack_big2,
   *  hurt3). Falls back to `actionAnim`'s prefix match when absent. */
  actionAnimCandidates?: string[];
  holdLast?: boolean;
  // Render only the combatant CHROME (intent/block/hotkey/name/HP/powers) over a
  // transparent {w,h} spacer, with NO own rig/sprite. Used for the Kaiser Crab
  // arms, whose body is drawn once by a shared rig BEHIND these overlays; the
  // spacer reserves the rig's footprint so the HP bar sits below it and the
  // intent above it.
  noSprite?: { w: number; h: number };
  /** The HP bar's width in px, where the game's own is known (the Kaiser Crab
   *  arms: their `Bounds` width + 24, see crabMount.ts). Absent, the bar
   *  follows `hpBarWidth`'s max-HP curve. */
  hpWidth?: number;
  /** Tallest rig body the arena has room for — see SpineSprite's `maxBodyH`.
   *  The arena does NOT set this (the game has no such clamp; it shrinks the
   *  whole row instead — see `scene.aspectFit`); the preview harnesses do. */
  maxBodyH?: number | null;
  /** Stage px per SCENE px (`scene.fitScene().k`). The rig's authored scale and
   *  bounds are both in the game's 1920x1080 scene units, so this is what turns
   *  them into the pixels this stage has. */
  pxPerScene?: number;
  /** The Defect's orb queue and its capacity. The belt is drawn HERE, inside
   *  the sprite wrap, and not as a row above the combatant: it hangs on an arc
   *  around the creature (see Orbs), so it needs the rig's own crown and feet —
   *  which only the wrap knows. Stacked above the column instead, it sat at the
   *  top of the (much taller) reserved sprite box, i.e. adrift in the sky. */
  orbs?: Orb[];
  orbSlots?: number;
  dying?: boolean;
  /** The WebGL dissolve (Task 14). Forwarded to the internal `SpineSprite`'s
   *  own `dissolve` prop — see that prop's doc — AND, independently of any
   *  sprite, read by `dyingChromeFadeSeconds` to set `--gs-dying-ms`, which
   *  paces the chrome's own fade-out.
   *
   *  So this is NOT inert for a `noSprite` combatant: there is no
   *  `SpineSprite` to feed, but the chrome fade still tracks the dissolve's
   *  duration instead of the `DEFAULT_DYING_FADE_SECONDS` fallback. That is
   *  the whole mechanism `KaiserCrabGroup` and `DecimillipedeGroup`'s
   *  chrome-only overlays rely on — `grep -rn "noSprite=" src/components/`
   *  outside tests finds those two and nothing else. An earlier version of
   *  this doc claimed the opposite; it was written before those overlays
   *  forwarded the prop. */
  dissolve?: { effect: DissolveEffect; rev: string; startedAt: number } | null;
  /** `DeathAnimLengthOverride` (`MonsterModel.cs:321`), when this creature's
   *  monster sets one. Only reaches the chrome fade, and only matters for a
   *  creature that gets no dissolve — see `dyingChromeFadeSeconds`. */
  deathAnimSeconds?: number | null;
  /** `VfxManifest.locStrings` (Board.tsx holds the manifest; this is the one
   *  slice of it the DOM VFX panels below need). `undefined` — manifest
   *  hasn't loaded yet, or 404'd, or a fixture/test passes none — falls back
   *  to the English literal baked in beside each consumer below; see the
   *  fallback note there for why a literal is still correct to keep, not a
   *  stopgap this component quietly depends on. */
  locStrings?: Record<string, string>;
  effects?: { dmgs?: { amount: number; nonce: number }[]; dmgKind?: "hp" | "block"; flash?: boolean;
    /** `NCreature.AnimShake` — the creature's own 1 s wobble, fired when a
     *  DEBUFF lands (`NCreature.cs:895-902`). The value is the fx nonce, not a
     *  flag: two debuffs in a row must retrigger, and a boolean staying `true`
     *  across them would not. `undefined` means no shake pending.
     *
     *  It used to be a damage tier, which was wrong twice over — the game's
     *  amplitude is fixed, and damage is the one case `AnimShake` explicitly
     *  refuses (`!Visuals.IsPlayingHurtAnimation()`). See `effects/animShake.ts`
     *  for the full reading; the whole-screen punch that the old wobble was
     *  really standing in for lives in `effects/screenShake.ts` and fires from
     *  `Board.tsx`. */
    shake?: number;
    blockPop?: boolean; blockShatter?: boolean;
    /** A hit landed and block ate all of it — `vfx_blocked_text.tscn`'s
     *  floating "Blocked" label (`NDamageBlockedVfx.cs`). Replaces the old
     *  `fullyBlocked` no-op: that event used to drive only the hit-punch
     *  flash (`flash` above), with nothing telling the player block was the
     *  reason no number appeared. */
    blocked?: boolean;
    /** A power being gained or lost, faithfully split into the game's own
     *  TWO panels rather than one invented chip:
     *  - `mode: "applied"` == `power_applied_vfx.tscn` (icon over a
     *    green/red-by-`kind` name) — `kind` only exists here because
     *    `NPowerRemovedVfx` has no `_isBuff` field at all (Task 12
     *    addendum #3); a removal cannot carry a colour the game itself
     *    never assigns it.
     *  - `mode: "removed"` == `power_removed_vfx.tscn` (icon beside the
     *    name AND a second "Wears Off" label).
     *  `icon` is the same `p.image` field `Powers.tsx` renders — `null`
     *  when the power has no art, same fallback as that component's own
     *  chip path. No `amount`: `NPowerAppliedVfx` stores one but never
     *  reads it back (only `_power.Title`), and `NPowerRemovedVfx.Create`
     *  doesn't even take one — showing a count here would be exactly the
     *  invented-chip mistake this task exists to delete. */
    powerPop?: { power: string; icon: string | null } &
      ({ mode: "applied"; kind: "buff" | "debuff" } | { mode: "removed" });
    nonce?: number };
}) {
  const pct = Math.max(0, Math.min(100, (c.hp / c.max_hp) * 100));
  // The box the power row is laid out against: `NPowerContainer
  // .SetCreatureBounds` takes the creature's `%Bounds` control and sizes the
  // container to `Bounds.Size.X * Bounds.Scale.X + 25`. Ours is the same
  // authored width, in the scene px the row is drawn in (`pxPerScene`, which
  // already carries the encounter camera and the row's shrink-to-fit).
  // The size jitter is deliberately NOT in it, for the reason `scene.ts`
  // gives on `instanceScale`: the jitter belongs to the drawn rig, not to the
  // authored box the game's layouts space by.
  const powerBoxW = Math.round(
    (boundsWidth({ rig: c.rig, rigScale: c.rig_scale,
      boundsH: c.rig_bounds_h, boundsW: c.rig_bounds_w }) + POWER_BOX_MARGIN)
    * (pxPerScene ?? 1));
  // Doom: the kill line painted on the bar (see doomAmount). A lethal Doom
  // takes the whole remaining bar; otherwise the band stops at doom/maxHp but
  // never runs past the live fill (a Doom above current HP IS lethal, and a
  // band overhanging an empty track would read as phantom health). The 4% floor
  // mirrors the game's own 12px minimum width, scaled to our shorter bar, so a
  // 1-point Doom on a 200 HP boss is still visible.
  const doom = c.hp > 0 ? doomAmount(c.powers) : 0;
  const doomLethal = doom > 0 && doom >= c.hp;
  const doomPct = doomLethal ? pct : Math.min(pct, Math.max(4, (doom / c.max_hp) * 100));
  const intent = c.intents?.[0];
  // Server image first (it knows the exact per-enemy variant), then the kind's
  // own atlas art, then the generic disc — always a painted icon.
  // Why this telegraph reads what it reads. `intentMods` drops the residual
  // `other` row: it exists so the printed arithmetic reconciles, but it names
  // no listener and so has no icon to draw.
  const intentTip = intent ? describeIntent(intent) : undefined;
  const intentDir = intent ? intentDirection(intent) : null;
  const intentMods = (intent?.modifiers ?? []).filter((m) => m.kind !== "other");
  const intentArt = intent
    ? intent.intent_image || assetUrl(INTENT_ART[intent.type] ?? INTENT_ART_FALLBACK)
    : null;
  // The Spine rig reports where its head sits inside the (boundsH-sized) layout
  // box; we expose it as `--rig-top` so the box-top-anchored overlays below pin
  // to the rig head, not the (often taller) box top. Until the rig is measured
  // we keep the overlays hidden so they fade in WITH the rig at the right spot
  // instead of popping in high and then dropping. Combatants with no rig (static
  // image) never report, so they show immediately at the box top as before.
  const [rigHeadTop, setRigHeadTop] = useState<number | null>(null);
  // A noSprite arm has no rig of its own, so it never "loads" — show chrome at once.
  const rigLoading = !noSprite && !!c.rig && rigHeadTop == null;
  // How tall this creature is actually DRAWN — crown to feet, i.e. the box less
  // the head offset above. The orb belt is scaled to it (see Orbs), and it can
  // only be read off the laid-out wrap: a rig is fitted to `maxBodyH` at load
  // time, so neither the authored bounds nor the box height is a proxy for it.
  // Observed rather than measured once, because the static-sprite path has NO
  // height until the image lands — a one-shot read there would size the belt
  // against a zero-tall box and collapse it onto a point.
  const wrapRef = useRef<HTMLDivElement>(null);
  const [bodyH, setBodyH] = useState<number | null>(null);
  const wantOrbs = (orbSlots ?? 0) > 0 || (orbs?.length ?? 0) > 0;
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!wantOrbs || !el) return;
    const measure = () => {
      const h = el.offsetHeight - (rigHeadTop ?? 0);
      setBodyH((prev) => (prev === h ? prev : h));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [wantOrbs, rigHeadTop, maxBodyH, c.rig, c.image]);

  const shakeAnimRef = useRef<Animation | null>(null);
  const shakeNonce = effects?.shake;
  useEffect(() => {
    if (shakeNonce == null) return;
    // `AnimShake` moves `Visuals` — the sprite subtree — and NOT the creature
    // node, so the health bar, the name and the power tray stay put while the
    // body wobbles. `gs-sprite-wrap` is the nearest thing to `Visuals` here.
    // (It also holds the intent chip and the orb belt, which the game keeps
    // outside `Visuals`; those ride along, which is the one place this differs.)
    const el = wrapRef.current;
    if (!el) return;
    // `_shakeTween == null || !_shakeTween.IsRunning()` (`NCreature.cs:1243`):
    // a second debuff inside the second is DROPPED, not restarted. That is the
    // opposite of the screen shake, whose single `_shakeInstance` is
    // overwritten — so the two cannot share a firing helper.
    if (shakeAnimRef.current?.playState === "running") return;
    // The other half of the C# guard, `!Visuals.IsPlayingHurtAnimation()`, has
    // nothing to apply to here. It exists because both the hurt animation and
    // the shake wrote `Visuals.Position`, one node's one property. In the
    // workbench the hurt clip is a Spine one-shot inside the sprite's own
    // canvas and the hit punch is a CSS transform on the wrap, so neither
    // contends with a transform on this element — they nest and compose.
    if (typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    // `animate` is absent in jsdom, so the attribute below is what tests
    // assert. Guarded rather than polyfilled, same as `fireScreenShake`.
    // The observable marker goes on the ROOT, which is what
    // `[data-combatant=...]` addresses — the animated node is an inner div.
    (el.closest(".gs-combatant") as HTMLElement | null)?.setAttribute("data-anim-shake", String(shakeNonce));
    if (typeof el.animate !== "function") return;
    shakeAnimRef.current = el.animate(
      animShakeKeyframes(ANIM_SHAKE_DURATION, ANIM_SHAKE_AMPLITUDE * pxPerScene),
      {
        duration: ANIM_SHAKE_DURATION * 1000,
        // `add`, not `replace`: this element also carries `gs-hit-punch`, a CSS
        // transform, and one card can both damage and debuff. In Godot the two
        // were separate properties on separate nodes; here they share
        // `transform`, so the shake has to add to the punch rather than mask it
        // for its whole second. The re-entry guard above is what keeps `add`
        // from stacking two shakes on top of each other.
        composite: "add",
      },
    );
  }, [shakeNonce, pxPerScene]);

  // Is the WebGL dissolve actually going to eat this sprite? Only then may
  // `game-skin.css` skip the sprite when it fades the dying creature — see
  // that rule's comment for the three states where nothing dissolves and the
  // CSS fade is still the only thing standing between a corpse and popping
  // out at full opacity. Needs all three: a manifest entry to dissolve WITH
  // (`dissolve`), a rig for `SpineSprite` to draw (`c.rig` — a `c.image`
  // enemy is a plain `<img>`), and a sprite of its own at all (a `noSprite`
  // spacer's body is drawn by a different combatant). `SpineSprite` covers
  // the fourth state, a rig that failed to load, from the inside.
  const dyingSpriteDissolves = dying && !!dissolve && !!c.rig && !noSprite;
  // Shrink (see bodyScale). The body scales about its feet; the overlays that
  // pin to the crown (`--rig-top`: intent, damage numbers, power pops) follow
  // it down, as the game re-places `IntentContainer` by `Visuals.Scale`
  // (`NCreature.cs`'s bounds refresh). `100%` resolves against the wrap at
  // each consumer's `top`/`bottom`, i.e. the body box the feet stand on.
  const scale = noSprite ? 1 : bodyScale(c.powers);
  const rigTop = scale === 1
    ? `${rigHeadTop ?? 0}px`
    : `calc(100% - (100% - ${rigHeadTop ?? 0}px) * ${scale})`;
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
    <div
      className={`gs-combatant${onClick ? " gs-clickable" : ""}${dying ? " gs-dying" : ""}${dyingSpriteDissolves ? " gs-dissolving" : ""}`}
      onClick={onClick}
      onMouseEnter={onHoverChange ? () => onHoverChange(true) : undefined}
      onMouseLeave={onHoverChange ? () => onHoverChange(false) : undefined}
      data-enemy-index={isEnemy ? dataIndex : undefined}
      data-combatant={anchor}
      // `--gs-dying-ms`: only meaningful while `.gs-dying` is present, but
      // harmless to set unconditionally — see `dyingChromeFadeSeconds`'s doc
      // for why this must match the WebGL dissolve's own duration.
      style={{ "--gs-dying-ms": `${dyingChromeFadeSeconds(dissolve, deathAnimSeconds)}s` } as React.CSSProperties}
    >
      <div ref={wrapRef} className={`gs-sprite-wrap${effects?.flash ? " gs-hit-punch" : ""}`}
        style={{ "--rig-top": rigTop } as React.CSSProperties}>
        {wantOrbs && !rigLoading && <Orbs orbs={orbs} slots={orbSlots} bodyH={bodyH} />}
        {hotkey != null && !rigLoading && <span aria-hidden="true" className="gs-hotkey">{hotkey}</span>}
        {reticle && !rigLoading && (
          <span className="gs-reticle" data-testid="advice-reticle" aria-label="the AI's target">
            <i className="gs-reticle-tl" /><i className="gs-reticle-tr" />
            <i className="gs-reticle-bl" /><i className="gs-reticle-br" />
          </span>
        )}
        {isEnemy && intent && !rigLoading && (
          /* The game centres the ICON over the creature's crown and hangs the
             damage numeral off its lower-LEFT (refs/game-combat-1.png: the "4"
             sits against the dagger's tip and overhangs it downward). r5
             centred the whole [number][icon] row instead, which pushed the icon
             right of the body and the number out over the background — both r5
             critics read it as "drifted off its owner". `gs-intent-nodmg` gives
             a numberless intent the bigger icon box: with no numeral beside it
             a 48px StatusCard chip reads as an unresolved grey blob. Layout
             lives in .gs-intent (game-skin.css). */
          <div className={`gs-intent${intent.damage == null ? " gs-intent-nodmg" : ""}`}
            title={intentTip}>
            {intent.damage != null && (
              /* `data-mod` says the number is NOT the move's own: down = the
                 board is reducing it (Weak/Shrink), up = raising it
                 (Strength/Vulnerable). Without it a telegraph that already has
                 a debuff baked in looks identical to one that has none — the
                 whole of bug report `wb-4eae3bad6ebd57f8`. Workbench-only: the
                 game paints this numeral plain white (NIntent.UpdateVisuals).
                 The modifier icons below name WHICH listeners did it. */
              <span className="gs-intent-dmg" data-mod={intentDir ?? undefined}>
                {intent.damage}{intent.hits && intent.hits > 1 ? `×${intent.hits}` : ""}
              </span>
            )}
            {intentMods.length > 0 && (
              <div className="gs-intent-mods">
                {intentMods.map((m) => (
                  m.image ? (
                    <img key={m.name} src={m.image} alt={m.title || m.name}
                      title={m.title || m.name} className="gs-intent-mod-icon" />
                  ) : (
                    <span key={m.name} className="gs-intent-mod-txt"
                      title={m.title || m.name}>{(m.title || m.name).slice(0, 2)}</span>
                  )
                ))}
              </div>
            )}
            {intentArt && (
              <img className="gs-intent-img" src={intentArt} alt={intent.type} title={intent.type} />
            )}
          </div>
        )}
        {/* One node per hit — a multi-hit attack shows every number in flight
            at once, like NDamageNumVfx does in the real game, rather than
            the single net total `combatDiff` alone could ever produce. Each
            is keyed and jittered off its OWN nonce so concurrent numbers on
            the same target don't trace the same throw. */}
        {effects?.dmgs?.map((d) => (
          <div key={`d${d.nonce}`} className={`gs-dmg${effects.dmgKind === "block" ? " gs-dmg-block" : ""}`}
            style={damageNumberJitter(d.nonce)}>
            <span>{d.amount}</span>
          </div>
        ))}
        {/* The card this thief is holding. The game gives the Thieving Hopper a
            `%StolenCardPos` marker and parents a real card node to it
            (ThievingHopper.ThieveryMove), because a bare "Swipe 1" pip does not
            answer the only question that matters: WHICH card left your deck —
            and whether killing the thief before it escapes is worth it. Drawn
            inside the sprite wrap so it rides the creature's own box (and so a
            click on it still targets the enemy, like the intent icon). */}
        {c.stolen_cards && c.stolen_cards.length > 0 && !rigLoading && (
          <div className="gs-stolen">
            {c.stolen_cards.map((sc, i) => (
              <div key={`${sc.id}-${i}`} className="gs-stolen-card"
                title={`Stolen from your deck: ${sc.name}${sc.upgraded ? "+" : ""} — kill ${c.name} before it escapes to get it back`}
                aria-label={`${c.name} stole ${sc.name}${sc.upgraded ? "+" : ""}`}>
                <CardFrame card={sc} width={64} />
              </div>
            ))}
          </div>
        )}
        {/* `vfx_blocked_text.tscn`: a lone Label, authored "Blocked" and
            overwritten at `_Ready` from loc key `vfx/BLOCKED` — v0.107.1 PCK
            `localization/eng/vfx.json`: `{"BLOCKED": "Blocked"}`.
            `manifest.locStrings.blocked` IS that loc row (`extract_vfx.py`'s
            `VFX_LOC_STRINGS` reads it out of the pck, same as every other
            asset this manifest carries) — the literal here is only the
            FALLBACK for a manifest that hasn't loaded yet (or 404'd), same
            "additive, never blank" contract `loadManifest`'s other callers
            already get. `sts2-pycli/localization_eng/` is a stale v0.103.2
            dump and was never the source for either string. */}
        {effects?.blocked && (
          <div key={`bt${effects.nonce}`} className="gs-blocked-text">
            {locStrings?.blocked ?? "Blocked"}
          </div>
        )}
        {effects?.powerPop && (
          <div key={`pp${effects.nonce}`} data-testid="power-pop" data-mode={effects.powerPop.mode}
            className={`gs-power-pop gs-power-pop-${effects.powerPop.mode}`
              + (effects.powerPop.mode === "applied" ? ` gs-${effects.powerPop.kind}` : "")}>
            {effects.powerPop.icon && (
              <img className="gs-power-pop-icon" src={effects.powerPop.icon} alt={effects.powerPop.power} />
            )}
            {effects.powerPop.mode === "applied" ? (
              // power_applied_vfx.tscn: Icon (TextureRect) above Label. The
              // label colour is `_isBuff ? green : red` (NPowerAppliedVfx.cs
              // line 161) — never a count: `_amount` is stored and never
              // read again (see the `powerPop` type doc above).
              <span className={`gs-power-pop-label gs-${effects.powerPop.kind}`}>{effects.powerPop.power}</span>
            ) : (
              // power_removed_vfx.tscn: HBox[ TextureRect, VBox[ PowerField,
              // WearsOff ] ]. `WearsOff`'s authored "Where's Off?" is
              // overwritten at `_Ready` from loc key `vfx/POWER_WEARS_OFF` =
              // "Wears Off" — `manifest.locStrings.powerWearsOff`, fallback
              // literal below for the same reason `blocked`'s has one. No
              // colour class: `NPowerRemovedVfx` has no `_isBuff` to drive one.
              <span className="gs-power-pop-textcol">
                <span className="gs-power-pop-label">{effects.powerPop.power}</span>
                <span className="gs-power-pop-wearsoff">{locStrings?.powerWearsOff ?? "Wears Off"}</span>
              </span>
            )}
          </div>
        )}
        {/* `Visuals`: the body Shrink scales (bodyScale; the tween is in
            game-skin.css). Its own box, so the scale reaches the drawn creature
            and its ground shadow but none of the chrome riding on the wrap.
            Unset at 1 rather than `scale: 1`, which would still open a
            stacking context and lift the ground shadow's `z-index: -1`. */}
        <div className="gs-sprite-body" style={scale === 1 ? undefined : { scale: String(scale) }}>
          {/* Ground-contact shadow (skipped for noSprite spacers — their shared
              rig draws one body elsewhere and per-arm ellipses would tile). */}
          {!noSprite && !rigLoading && <div className="gs-groundshadow" aria-hidden="true" />}
          {noSprite
            ? <div aria-hidden="true" style={{ width: noSprite.w, height: noSprite.h }} />
            : c.rig
              ? <SpineSprite rigBase={c.rig} animation={animation}
                  skin={isEnemy ? monsterSkin(c.name, dataIndex) : undefined}
                  /* The authored scene scale: it IS px per rig unit in the game's
                     scene (see scene.ts), so the rig comes out the size the game
                     draws it — times this instance's own jitter, which the game
                     applies to repeated monsters of one type. */
                  worldScale={c.rig_scale != null
                    ? Math.abs(c.rig_scale) * instanceScale(c)
                    : undefined}
                  hueShift={c.hue_jitter}
                  pxPerScene={pxPerScene}
                  /* Reserve the rig's REAL body, from the generated rig-bounds
                     table, rather than the authored Bounds box — that box is a
                     hitbox, drawn a median 1.07x the rig, and the overlays below
                     pin to the head this reports. */
                  boundsH={spriteBodyH(c, isEnemy)}
                  maxBodyH={maxBodyH}
                  fallbackHeight={(isEnemy ? FALLBACK_BODY_H.enemy : FALLBACK_BODY_H.player) * pxPerScene}
                  fallback={c.image} alt={c.name}
                  actionAnim={actionAnim} actionAnimCandidates={actionAnimCandidates}
                  actionNonce={actionNonce} holdLast={holdLast} dissolve={dissolve} onMeasure={setRigHeadTop} />
              : c.image
                ? <img className={`gs-sprite ${isEnemy ? "gs-enemy" : "gs-player"}`} src={c.image} alt={c.name} />
                : <div className="gs-sprite-fallback" />}
        </div>
      </div>
      {/* Game order: HP bar right under the feet, name under the bar.
          Three layers so the chevron silhouette survives: frame (teal edge) ->
          track (dark) -> fill, each with its own clip-path; the numerals ride
          on top and deliberately overflow the bar (see .gs-hptext). The bar's
          LENGTH tracks the creature — r3 gave a 13-HP and a 27-HP slime the
          same 150px, which both critics flagged. sqrt keeps small enemies
          distinguishable without making a boss's bar span the arena. */}
      {/* `gs-hp-blocked`: NHealthBar.RefreshBlockUi — while the creature has
          block the bar shows `%BlockOutline`, the fill is modulated to
          `#3B6FA3` and the HP numerals take the `#1B3045` outline
          (RefreshText). */}
      <div className={`gs-hpwrap${blocked ? " gs-hp-blocked" : ""}`}
        style={{ "--gs-hpw": `${hpWidth ?? hpBarWidth(c.max_hp)}px` } as React.CSSProperties}>
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
        >
          <img className="gs-block-icon" src={assetUrl("ui/combat/block.png")} alt="" aria-hidden="true" draggable={false} />
          {blocked ? <span className="gs-block-num">{c.block}</span> : null}
        </div>
        {effects?.blockShatter && <div key={`bs${effects.nonce}`} className="gs-block-shatter" />}
      </div>
      {/* Name at reduced at-rest weight (hover brightens — see .gs-nameplate);
          the exact-HP numerals on the bar above stay always-visible. Debug
          size suffixes ("Twig Slime (M)") stay OFF the scene — display the
          bare name; the full name (with the tag) lives on the tooltip and
          the accessible label so nothing is lost. */}
      {(() => {
        const { text, full } = nameplateText(c.name, label);
        return (
          <div className="gs-nameplate" title={full}
            aria-label={full}>{text}</div>
        );
      })()}
      {c.powers && c.powers.length > 0 && (
        <div className="gs-powers"><Powers powers={c.powers} boxWidth={powerBoxW} /></div>
      )}
    </div>
  );
}
