import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { assetUrl, type Enemy, type Orb, type Power } from "../api";
import type { DissolveEffect } from "../effects/manifest";
import CardFrame from "./CardFrame";
import Orbs from "./Orbs";
import Powers from "./Powers";
import HealthBar from "./HealthBar";
import { powerItem } from "../inspect/powerItem";
import { useInspect } from "../inspect/useInspect";
import { atLeast, usePhone, useStageZoom } from "../phone";
import SpineSprite from "./SpineSprite";
import { monsterSkin } from "../monsterSkins";
import { describeIntent, intentDirection } from "../intentBreakdown";
import { intentInfo, intentName } from "../intentNames";
import { boundsWidth, instanceScale } from "../scene";
import type { PowerChrome } from "../portraitSplit";
import { POWER_BOX_MARGIN, powerFit } from "../powerRow";
import { inspectTrigger } from "../inspect/trigger";
import { ANIM_SHAKE_AMPLITUDE, ANIM_SHAKE_DURATION, animShakeKeyframes } from "../effects/animShake";
import {
  nameplateText,
  damageNumberJitter,
  spriteBodyH,
  FALLBACK_BODY_H,
  hpBarWidth,
  bodyScale,
  dyingChromeFadeSeconds,
} from "./combatantMetrics";

export interface PlayerLike {
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

/** A creature's one-shot effects for this render: damage numbers, flashes,
 *  the block shield's pop and shatter, power pops. */
export type CombatantEffects = { dmgs?: { amount: number; nonce: number }[]; dmgKind?: "hp" | "block"; flash?: boolean;
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

export default function Combatant({ c, isEnemy, dataIndex, onClick, onHoverChange, hotkey, anchor, animation, actionAnim, actionAnimCandidates, actionNonce, holdLast, dying, dissolve, deathAnimSeconds, effects, noSprite, noChrome, onCrown, powerChrome, powerCells, hpWidth, bodyDx = 0, maxBodyH, pxPerScene = 1, orbs, orbSlots, locStrings, label, reticle }: {
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
  /** The body alone: no HP bar, nameplate or powers under the feet. The
   *  phone-portrait player figure, whose status block (`PlayerBand`) stands
   *  beside it instead. */
  noChrome?: boolean;
  /** Reports how far the DRAWN body reaches above the feet, in scene px (the
   *  rig's measured head, not its authored box), once the rig has measured
   *  and again on every re-fit. A caller drawing a creature at a fixed height
   *  on the glass scales by this. */
  onCrown?: (sceneCrown: number) => void;
  /** Phone portrait (the split, portraitSplit.ts): how this enemy's powers
   *  hang off it, the fight's chrome mode. `rows`: two rows of `powerCells`
   *  44 px cells under the name; `column`: the landscape column on the
   *  body's left; `compact`: one cell (a lone power, or "+N" listing them
   *  all) and no nameplate. Absent: today's rules (rows measured off the
   *  stage in portrait, the column in landscape). */
  powerChrome?: PowerChrome;
  /** `rows` only: cells across, from the solver's slot width. */
  powerCells?: number;
  /** The HP bar's width in px, where the game's own is known (the Kaiser Crab
   *  arms: their `Bounds` width + 24, see crabMount.ts). Absent, the bar
   *  follows `hpBarWidth`'s max-HP curve. */
  hpWidth?: number;
  /** Design px the drawn body stands right of its box's centre (phone
   *  portrait: the rig's ink is centred in its slot, the chrome stays put). */
  bodyDx?: number;
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
  effects?: CombatantEffects;
}) {
  // Phone: power icons never render under 28px on the glass, in a 44px tap box
  // (the stage is scaled by `--gs-k`, so that is `atLeast` in design px). The
  // block is capped at two rows of whatever fits the creature's width; the rest
  // hides behind a "+N" chip. Desktop passes nothing and keeps the game's rule.
  const phone = usePhone();
  const stageK = useStageZoom();
  const powerSize = phone ? atLeast(26, 28, stageK) : undefined;
  const powerHit = phone ? atLeast(26, 44, stageK) : undefined;
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
  const intent = c.intents?.[0];
  const inspect = useInspect();
  // Distinguishes this creature's sheet entries from another's (toggle-closes
  // only the SAME entry).
  const inspectId = `${anchor ?? ""}${dataIndex ?? ""}${c.name}`;
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

  // Reported from the rig's own measurement callback, not from an effect on
  // `pxPerScene`: the head it measured belongs to the scale it drew at, and a
  // new scale reaching here before the rig has re-measured would pair the old
  // head with it.
  const authoredCrown = spriteBodyH(c, isEnemy);
  const onRigMeasure = (headTop: number | null) => {
    setRigHeadTop(headTop);
    if (headTop != null && pxPerScene > 0) onCrown?.(authoredCrown - headTop / pxPerScene);
  };

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
            title={intentTip}
            // Phone: a tap shows the same explanation `title` carries on
            // desktop, plus one row per modifier.
            // Phone only (desktop gets nothing): a named button, Enter / Space
            // open it too.
            {...inspectTrigger(inspect.enabled, `Intent: ${intentTip ?? intentInfo(intent).name}`, () => {
              // The breakdown's arithmetic when there is one (it is what the
              // desktop tooltip shows); the kind's own name otherwise, never
              // the raw enum id. Either way the game's sentence under it.
              inspect.open({ key: `intent:${inspectId}:${intent.type}:${intent.damage ?? ''}:${intentMods.map((m) => m.name).join(',')}`, icon: intentArt,
                name: intentTip ?? intentInfo(intent).name, description: intentInfo(intent).description,
                list: intentMods.map((m) => ({ key: m.name, icon: m.image, name: m.title || m.name })) });
            })}>
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
              <img className="gs-intent-img" src={intentArt} alt={intentName(intent.type)} title={intentName(intent.type)} />
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
        <div className="gs-sprite-body" style={scale === 1 && !bodyDx ? undefined
          : { ...(scale === 1 ? {} : { scale: String(scale) }), ...(bodyDx ? { left: bodyDx } : {}) }}>
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
                  actionNonce={actionNonce} holdLast={holdLast} dissolve={dissolve} onMeasure={onRigMeasure} />
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
      {!noChrome && <HealthBar c={c} widthPx={hpWidth ?? hpBarWidth(c.max_hp)} effects={effects} inspectId={inspectId} />}
      {/* Name at reduced at-rest weight (hover brightens — see .gs-nameplate);
          the exact-HP numerals on the bar above stay always-visible. Debug
          size suffixes ("Twig Slime (M)") stay OFF the scene — display the
          bare name; the full name (with the tag) lives on the tooltip and
          the accessible label so nothing is lost. */}
      {!noChrome && powerChrome !== "compact" && (() => {
        const { text, full } = nameplateText(c.name, label);
        return (
          <div className="gs-nameplate" title={full}
            aria-label={full}>{text}</div>
        );
      })()}
      {!noChrome && c.powers && c.powers.length > 0 && (() => {
        // Phone landscape, and a `column` split fight: a column BESIDE the
        // creature (powerColumn.ts), so nothing but the HP bar and the name
        // hangs under the feet and the fighters can be drawn as tall as the
        // short stage allows.
        const column = powerChrome ? powerChrome === "column" : phone === "landscape";
        // A split fight's rows and compact cell take their count across from
        // the solver's slot, never from measuring the stage.
        const cells = powerChrome === "compact" ? 1
          : powerChrome === "rows" ? Math.max(1, powerCells ?? 1) : undefined;
        return (
          <div className={`gs-powers${column ? " gs-powers-col" : ""}`}>
            <Powers powers={c.powers} boxWidth={powerBoxW} size={powerSize} hitSize={powerHit}
              column={column}
              cells={cells}
              maxShown={cells != null ? (powerChrome === "compact" ? 1 : 2 * cells)
                : powerHit ? 2 * powerFit(powerBoxW, powerHit) : undefined}
              minShown={cells != null ? (powerChrome === "compact" ? 1 : 2 * cells) : undefined}
              inspectScope={inspectId}
              onOverflow={(hidden) => inspect.open({ key: `powers:${inspectId}`, name: "More powers",
                list: hidden.map((p) => powerItem(p, inspectId)) })} />
          </div>
        );
      })()}
    </div>
  );
}
