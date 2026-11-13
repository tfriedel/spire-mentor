// VfxLayer's live-effect records and the pure math that builds, ages and
// anchors them — kept out of the component file so it exports only the
// component (react-refresh/only-export-components).
import * as spine from "@esotericsoftware/spine-webgl";
import {
  type DissolveEffect,
  type FlipbookEffect,
  type ParticlesEffect,
  type SpineEffect,
  type SpriteShaderEffect,
  type TransformAnimEffect,
  type VfxEffect
} from "../effects/manifest";
import type { TargetRef } from "../effects/types";
import { startComposite, type LiveComposite } from "../effects/particles/composite";
import { makeRng } from "../effects/particles/rng";
import type { SimState } from "../effects/particles/simulate";

/**
 * The DOM box an effect spawns on, or `null` when that creature is not on the
 * board. Anchors are the ones `Combatant.tsx` already emits, and *arena* is
 * `.gs-actors` — the container they are drawn in (see `Board.tsx`).
 *
 * Osty deliberately returns `null`: it has no `data-combatant` anchor, and
 * sub-project 1 leaves it to the existing CSS effects.
 */
export function anchorRect(target: TargetRef, arena: HTMLElement): DOMRect | null {
  const sel =
    target.side === "enemy"
      ? `[data-enemy-index="${target.index}"]`
      : target.side === "player"
        ? `[data-combatant="player"]`
        : null;
  if (!sel) return null;
  const el = arena.querySelector(sel);
  return el ? (el as HTMLElement).getBoundingClientRect() : null;
}

/**
 * The block BADGE's own box, or `null` when it is not on screen —
 * `vfx_block_broken`'s anchor (`sink.ts`'s `PlayedEffect.anchor` doc:
 * `NHealthBar.cs:575`'s `_blockContainer.GlobalPosition +
 * _blockContainer.Size * 0.5`), which is a DIFFERENT element than the
 * creature `anchorRect` resolves — `block_spark_vfx` anchors to the creature
 * itself, `vfx_block_broken` does not, and the two must not be unified (see
 * that doc for the full C# citation).
 *
 * Looked up as a DESCENDANT of the creature's own anchor element
 * (`Combatant.tsx`'s `data-block-badge` div), not a standalone selector — the
 * badge has no target-identifying attribute of its own, only the creature
 * container does. Returns `null` only when the creature itself is off-board
 * (a rig still loading, or a slot with nothing in it) — `Combatant.tsx` keeps
 * the badge div ALWAYS mounted once the rig is up (only its number/opacity
 * toggle with `c.block`), specifically so this never returns null for the one
 * case that matters: a `blockBreak` event, which by construction only ever
 * fires when block was just non-zero (`combatDiff.ts`: `prev.block > 0 &&
 * next.block === 0`), so the badge was genuinely on screen right up to this
 * transition. See `Combatant.tsx`'s doc on that div for why a
 * conditionally-mounted badge doesn't work here: `Board.tsx`'s `blockBreak`
 * handling runs in a `useEffect` AFTER `c.block` has already committed to 0,
 * so gating the anchor's OWN presence on `c.block > 0` (or even a same-tick
 * `Fx` flag meant to bridge the gap) reads the same too-late DOM — it looked
 * like a fix and wasn't, verified by tracing React's render-then-effect
 * ordering, not assumed.
 */
export function badgeRect(target: TargetRef, arena: HTMLElement): DOMRect | null {
  const sel =
    target.side === "enemy"
      ? `[data-enemy-index="${target.index}"]`
      : target.side === "player"
        ? `[data-combatant="player"]`
        : null;
  if (!sel) return null;
  const creature = arena.querySelector(sel);
  if (!creature) return null;
  // The portrait split draws the player's figure bare (`noChrome`): its block
  // shield is the band's, beside it (`PlayerBand`), not inside the figure.
  const badge = creature.querySelector("[data-block-badge]")
    ?? (target.side === "player" ? arena.querySelector('[data-block-badge-for="player"]') : null);
  return badge ? (badge as HTMLElement).getBoundingClientRect() : null;
}

interface LiveFlipbook {
  kind: "flipbook";
  effect: FlipbookEffect;
  texturesRev: string;
  /** Canvas-space centre, Y-up. */
  x: number;
  y: number;
  /** Godot's own `Node2D.RotationDegrees`, straight from the scene's authored
   *  `_rotationRange` — NOT a draw angle. Convert with
   *  {@link drawAngleFromGodotDegrees} at the draw call. */
  angleDeg: number;
  startedAt: number;
}

/**
 * A Godot 2D rotation, in DEGREES, as the `angle` argument of `spine-webgl`'s
 * `SceneRenderer.drawTextureRotated`.
 *
 * Two conversions, and both are needed — either one alone is wrong:
 *
 * 1. **Unit.** `drawTextureRotated`'s `angle` is DEGREES: `SceneRenderer.js:247`
 *    hands it to `MathUtils.cosDeg`/`sinDeg`, which multiply by `Math.PI / 180`
 *    themselves (`Utils.js:170`). Handing it radians makes it re-convert an
 *    already-converted value, which is what the flipbook branch used to do —
 *    an authored 0-25° jitter came out at 1/57th of a degree, i.e. inert.
 * 2. **Sign.** Godot's 2D rotation is visually CLOCKWISE in its Y-down space;
 *    the rotation matrix `drawTextureRotated` builds is counter-clockwise in a
 *    Y-up one, and this canvas is Y-up. The same axis flip that already negates
 *    particle and sprite Y for position inverts rotational chirality too. The
 *    flipbooks' `_rotationRange` is `(0, 25)` — asymmetric — so fixing only the
 *    unit would spin all 14 shipped composites the wrong way.
 *
 * `NSpriteAnimator._Ready` (`NSpriteAnimator.cs:91-93`) is the authored source:
 * `RotationDegrees = new Random().Next((int)_rotationRange.X, (int)_rotationRange.Y)`.
 */
export function drawAngleFromGodotDegrees(deg: number): number {
  return -deg;
}

/** {@link drawAngleFromGodotDegrees} for a Godot rotation in RADIANS — the unit
 *  `Animation` rotation tracks (and `Node2D.Rotation`) are authored in. */
export function drawAngleFromGodotRadians(rad: number): number {
  return drawAngleFromGodotDegrees((rad * 180) / Math.PI);
}

export interface LiveSpine {
  kind: "spine";
  effect: SpineEffect;
  /** Canvas-space centre, Y-up — same convention as `LiveFlipbook`. */
  x: number;
  y: number;
  /** Built lazily on the first frame whose rig assets have finished loading
   *  (the atlas/skeleton binary race the first hits of a fight, same as a
   *  flipbook frame still in flight). `null` until then. */
  skeleton: spine.Skeleton | null;
  state: spine.AnimationState | null;
  /** The one-shot track `state.setAnimation` returned — `isComplete()` on
   *  this, not a timer, is what frees the instance under the normal path, so
   *  a clip's own duration (never modeled separately here) is exactly how
   *  long it lives. `null` until `skeleton`/`state` are built, or if the rig
   *  has no `clip` at all. See `spineExpired` for the ABNORMAL path (a rig
   *  whose assets never finish loading). */
  track: spine.TrackEntry | null;
  /** When this instance was created (`performance.now()`), for
   *  `spineExpired`'s wall-clock backstop — independent of load success or
   *  `track`, unlike `LiveFlipbook`'s `startedAt` (which drives its actual
   *  frame index, not just a safety cap). */
  startedAt: number;
  /** Slot-parented emitters this rig has STARTED, keyed by emitter name, with
   *  the `performance.now()` its rig event fired at and its own simulation
   *  (GH #669). Empty until the rig raises the event — the burst is not part
   *  of the effect's opening frame, it is `NKaiserCrabBossExplosionVfx`
   *  reacting to `left_embers_start` partway through the clip. Absent
   *  entirely for a rig with no `emitters`. */
  slotEmitters?: Map<string, { startedAt: number; sim: SimState | null; seed: number }>;
}

/** A started particle composite (`kind: "particles"`, Task 15). The heavy
 *  lifting — which emitters exist, when each starts, the resolved tint, the
 *  bespoke-script randomiser — lives in `startComposite`/`isDone`
 *  (`effects/particles/composite.ts`), which is what's unit-tested; this is
 *  just the wrapper the draw loop's `Live` union needs (anchor position +
 *  wall-clock start), the same shape `LiveFlipbook`/`LiveSpine` already have. */
export interface LiveParticles {
  kind: "particles";
  composite: LiveComposite;
  /** One resumable simulation per emitter, positionally aligned with
   *  `composite.emitters`, each advanced once per display frame. `null` until
   *  the draw loop can build that emitter's spec — its curve LUTs load
   *  asynchronously, so an emitter may be several frames late; whenever it
   *  does start, `advanceTo` catches it up in one jump, which is the same
   *  frame sequence it would have run had it started on time.
   *
   *  Held here rather than on `LiveComposite` to keep `composite.ts` pure
   *  data: this is renderer state, and it is mutable by design. */
  sims: (SimState | null)[];
  /** Canvas-space centre, Y-up — same convention as `LiveFlipbook`/`LiveSpine`. */
  x: number;
  y: number;
  startedAt: number;
}

/** A started `transform_anim` effect (`vfx_block_broken`'s two halves, Task
 *  13). Purely a wall-clock playhead into `effect.sprites`' baked tracks —
 *  `sampleTrack` (a pure function, unit-tested on its own) does the actual
 *  sampling; this is only the anchor + start-time wrapper the draw loop's
 *  `Live` union needs, the same shape every other kind here has. */
interface LiveTransformAnim {
  kind: "transform_anim";
  effect: TransformAnimEffect;
  /** Canvas-space spawn point, Y-up — the block BADGE's centre for
   *  `vfx_block_broken` (`sink.ts`'s `anchor: "badge"`), NOT the creature. */
  x: number;
  y: number;
  startedAt: number;
}

/** A started `sprite_shader` effect (`vfx_power_flash_vfx`, Task 13) — one
 *  additive sprite over a power's icon, animated by `effect.tween`
 *  (`sampleSpriteShaderTween`, a pure function tested on its own below). */
export interface LiveSpriteShader {
  kind: "sprite_shader";
  effect: SpriteShaderEffect;
  x: number;
  y: number;
  /** The flashing power's own icon URL (`PlayedEffect.iconUrl` — see that
   *  field's doc for why the manifest itself carries no texture for this
   *  kind). `null` when the power had no icon on record: nothing draws for
   *  an instance with no texture, same as every other "still loading, keep
   *  waiting" texture path in this file — except this one never resolves,
   *  since a `null` icon URL has nothing to eventually load. */
  iconUrl: string | null;
  startedAt: number;
}

export type Live = LiveFlipbook | LiveSpine | LiveParticles | LiveTransformAnim | LiveSpriteShader;

/**
 * Wall-clock lifetime cap for a `LiveSpine` instance, independent of
 * `TrackEntry.isComplete()` or whether its rig ever finished loading.
 *
 * `AssetManagerBase.error()` (the vendored spine-core loader) records a
 * failed fetch only in its own `errors` map — never in `cache.assets` — so a
 * transient failure (a real network error, not the by-design "never asked
 * for" case `vfx_adrenaline` sits in) leaves `getSkeletonData` returning
 * `null` FOREVER for that rig: nothing here ever retries. Without this cap,
 * the draw loop's "still loading, keep waiting" branch would keep every
 * `LiveSpine` pushed for that rig in `liveRef.current` for the life of the
 * `manifest` reference — an unbounded leak, and a direct violation of the
 * "freed when the clip ends" contract this renderer exists to satisfy.
 *
 * `getSkeletonData` failing to resolve is ALSO checked directly (via
 * `assets.getErrors()`) so a load failure drops its instances promptly
 * rather than waiting out the full cap — but this constant is the
 * load-bearing half: it holds even for a failure mode neither check
 * anticipates. Generous relative to every real clip here (well under a
 * second once playing) and to a legitimately slow-but-successful load, so it
 * never cuts off a real instance — it only bounds a genuinely stuck one.
 */
export const SPINE_MAX_LIFETIME_MS = 8000;

/** Pure wall-clock check backing the cap above — no DOM/canvas/GL, so it's
 *  directly unit-testable under jsdom without reproducing a real network
 *  failure (see `spineVfxSelection.test.ts`). */
export function spineExpired(l: LiveSpine, now: number): boolean {
  return now - l.startedAt > SPINE_MAX_LIFETIME_MS;
}

/**
 * Which overlay draw path a manifest entry routes to, as a fresh `Live`
 * record ready to push. Pure: touches no DOM/canvas/GL (the particle branch
 * only calls `startComposite`, itself pure — see its own doc), which is what
 * makes the ROUTING decision unit-testable under jsdom (no WebGL there) —
 * the same seam `anchorRect` above already established for the anchor half
 * of `sink.play`. See `spineVfxSelection.test.ts` / `composite.test.ts`.
 *
 * `tint` is the binding table's tint (`PlayedEffect.tint`) — consumed only by
 * the particle branch (design spec §1.4/§5), `null` by default so every
 * existing flipbook/Spine call site (and every pre-Task-15 test) keeps
 * working unchanged. `iconUrl` (`PlayedEffect.iconUrl`) is consumed only by
 * the `sprite_shader` branch (Task 13's `vfx_power_flash_vfx`), `null` by
 * default for the same reason.
 */
export function buildLiveEffect(
  entry: VfxEffect,
  x: number,
  y: number,
  manifestRev: string,
  tint: string | null = null,
  iconUrl: string | null = null,
  dissolve: { emissionRadius: number; lifetime: number } | null = null,
): Live | null {
  if (entry.kind === "particles") {
    // Math.random(), not a seeded stream: this is the ONE instancing draw
    // that picks which random composite a live event gets (mirroring how
    // LiveFlipbook's randomizeRotation already draws below) — the
    // determinism contract ("no Math.random() in the simulator") is about
    // `simulate.ts`'s own particle physics, not the top-level UI choosing a
    // fresh seed per play.
    const seed = Math.floor(Math.random() * 0xffffffff);
    return {
      kind: "particles",
      composite: startComposite(entry, makeRng(seed), tint),
      sims: entry.emitters.map(() => null),
      x,
      y,
      startedAt: performance.now(),
    };
  }
  if (entry.kind === "flipbook") {
    const [lo, hi] = entry.rotationRange;
    const angle = entry.randomizeRotation ? lo + Math.random() * (hi - lo) : 0;
    return { kind: "flipbook", effect: entry, texturesRev: manifestRev, x, y, angleDeg: angle, startedAt: performance.now() };
  }
  if (entry.kind === "spine") {
    // `skeleton`/`state`/`track` are built lazily by the draw loop once the
    // rig's assets finish loading — see the module doc comment.
    return { kind: "spine", effect: entry, x, y, skeleton: null, state: null, track: null, startedAt: performance.now() };
  }
  if (entry.kind === "transform_anim") {
    return { kind: "transform_anim", effect: entry, x, y, startedAt: performance.now() };
  }
  if (entry.kind === "sprite_shader") {
    // `iconUrl` rides in on the play call, not the manifest entry — see
    // `LiveSpriteShader`'s doc for why the manifest carries no texture here.
    return { kind: "sprite_shader", effect: entry, x, y, iconUrl, startedAt: performance.now() };
  }
  if (entry.kind === "dissolve") {
    // The dying creature's DEBRIS BURST. The dissolve SPRITE is not drawn
    // here at all — it lives per-creature inside `SpineSprite`, which owns
    // the FBO the rig renders into (`effects/dissolve.ts`'s module doc says
    // why it cannot be shared). This branch is the other half of
    // `vfx_monster_death`, and it is an ordinary particle composite.
    //
    // `dissolve` is required, not optional-with-a-default: the manifest's own
    // `emissionSphereRadius` is deliberately blanked to 0 (`extract_vfx.py`),
    // because the game overwrites it per creature. No `dissolve` means no
    // measurable creature, and playing nothing is the honest outcome — a
    // burst at the blanked value would be every particle on one point.
    if (!dissolve) return null;
    return {
      kind: "particles",
      composite: startComposite(dissolveBurstEffect(entry, dissolve), makeRng(Math.floor(Math.random() * 0xffffffff)), tint),
      sims: entry.emitters.map(() => null),
      x,
      y,
      startedAt: performance.now(),
    };
  }
  // Returning `null` — which the sink already handles by not pushing
  // anything — plays nothing, which is the honest outcome for a kind with no
  // draw path. This used to be an unconditional `return { kind: "spine", ... }`
  // fall-through, which typed a `transform_anim` entry AS a Spine effect and
  // then read `entry.rig`/`entry.clip` off it at draw time.
  //
  // The annotation is the guard, and the reason this is not simply a bare
  // `return null`: a kind added to `VfxEffect` without a draw path fails to
  // COMPILE here rather than silently joining the do-nothing set. `never`
  // because every kind now HAS a branch above — add one to `VfxEffect` and
  // this stops compiling.
  const unrendered: never = entry;
  void unrendered;
  return null;
}

/**
 * `vfx_monster_death`'s debris burst as a `ParticlesEffect` the ordinary
 * composite path can start, with the two things the C# does to it that the
 * manifest cannot bake in.
 *
 * 1. **The radius**, and with it the composite's **lifetime** — both
 *    per-creature, both from `live` (see `PlayedEffect.dissolve`).
 *    `NMonsterDeathVfx.cs:231` assigns
 *    `EmissionSphereRadius = node.Size.Y / 4f` per creature, so the scene's
 *    authored 99.23 never reaches the screen and `extract_vfx.py` blanks it
 *    to 0.
 *
 * 2. **The parent's `modulate`.** The burst's node is `parent="Visual"` in
 *    the scene, and `Visual` carries `modulate = Color(1, 1, 1, 0.466667)`.
 *    Godot's `modulate` descends to children (`self_modulate` is the one that
 *    does not), so the game draws the debris at that alpha too. Folded into
 *    the emitter's `selfModulate` HERE, at play time, rather than in the
 *    extractor: `selfModulate` is the scene node's own verbatim value, and
 *    baking a parent's alpha into it would make the field a lie for every
 *    other reader (`DissolveEffect.modulateAlpha`'s doc makes the same point
 *    from the other side).
 *
 * The third item rides along for free: routing through the particle path
 * draws the burst at that path's alpha-mix blend rather than the dissolve
 * sprite's `blend_premul_alpha`, which is what the game does — a
 * `CanvasItem`'s `material` does NOT descend to children.
 *
 * `mirrorX`/`pivotScale` are null and `tinted` false because the death scene
 * has neither — those are `NVfxParticleSystem` authoring concepts and this
 * is a bespoke `NMonsterDeathVfx`.
 */
export function dissolveBurstEffect(
  entry: DissolveEffect,
  live: { emissionRadius: number; lifetime: number },
): ParticlesEffect {
  return {
    kind: "particles",
    // The composite's life is the DISSOLVE TWEEN's, not the debris'. The C#
    // frees the whole death node once that tween finishes
    // (`NMonsterDeathVfx.cs:240-241`), and the tween is per-creature — so a
    // small monster's burst really is cut off before its 1.8 s particles
    // would have expired. `entry.lifetime` is the 2.5 s CEILING; the caller
    // narrows it, the same way `Board`'s `dissolveEffectFor` narrows
    // `dissolveSeconds` for the sprite half.
    lifetime: live.lifetime,
    emitters: entry.emitters.map((e) => ({
      ...e,
      selfModulate: [
        e.selfModulate[0],
        e.selfModulate[1],
        e.selfModulate[2],
        e.selfModulate[3] * entry.modulateAlpha,
      ] as [number, number, number, number],
      process: { ...e.process, emissionSphereRadius: live.emissionRadius },
    })),
    mirrorX: null,
    pivotScale: null,
    tinted: false,
  };
}

/** `TransitionType`s reported by `evaluateEase` already, so a tween that
 *  names one this port has never seen logs once rather than every frame. */
const reportedUnknownTrans = new Set<number>();

/**
 * One `SpriteShaderTweener`'s eased progress at local fraction `t` (clamped
 * to [0, 1], not wrapped — a tween runs once, it does not loop).
 *
 * Godot's `Tween` defaults to `TRANS_LINEAR` / `EASE_IN_OUT` when a tweener
 * calls neither `SetEase` nor `SetTrans` (`ease`/`trans` both `null` in the
 * manifest) — confirmed against Godot's own `Tween` class reference ("Before
 * this method is called, the default transition type is TRANS_LINEAR" / "...
 * the default ease type is EASE_IN_OUT"), not assumed. A LINEAR transition's
 * curve is `t` regardless of ease direction, so `null`/`TRANS_LINEAR` (0)
 * never needs the ease branch at all — which is also every `scale` tweener in
 * the committed `power_flash_vfx` data (`ease`/`trans` both `null`).
 *
 * `TRANS_SINE` (1) is the only other transition the committed tween data
 * uses (`modulateAlpha`'s two tweeners, `ease` 1/EASE_OUT and 0/EASE_IN), so
 * that is the only one this port implements — Godot's own Penner-style sine
 * curves (`math_funcs.cpp`). A `trans` this has never seen reports once and
 * falls back to linear rather than silently guessing a curve shape.
 */
function evaluateEase(t: number, trans: number | null, ease: number | null): number {
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
  if (trans === null || trans === 0) return clamped; // TRANS_LINEAR (or the unset default)
  if (trans === 1) {
    // TRANS_SINE.
    if (ease === 0) return 1 - Math.cos((clamped * Math.PI) / 2); // EASE_IN
    if (ease === 1) return Math.sin((clamped * Math.PI) / 2); // EASE_OUT
    return -(Math.cos(Math.PI * clamped) - 1) / 2; // EASE_IN_OUT (ease null/2/unrecognized)
  }
  if (!reportedUnknownTrans.has(trans)) {
    reportedUnknownTrans.add(trans);
    console.error(`power flash tween: unrecognized TransitionType ${trans}; using linear`);
  }
  return clamped;
}

/**
 * `vfx_power_flash_vfx`'s scale/alpha at `elapsed` seconds into its life —
 * `NPowerFlashVfx.cs:97-102`'s tween, transcribed as data by Task 10
 * (`SpriteShaderEffect.tween`) and evaluated here. Pure: no DOM/canvas/GL, so
 * it is unit-testable without a WebGL context, the same seam every other
 * pure piece of this file already gets.
 *
 * Steps run in SEQUENCE; the tweeners inside one step run in PARALLEL, each
 * over its OWN duration — `modulateAlpha`'s first tweener (0.25s) finishes
 * and holds while `scale`'s sibling in the same step (0.4s) keeps animating,
 * and the NEXT step does not start until the step's longest tweener does
 * (`SpriteShaderTweener`'s own doc: "outer array = STEPS run in sequence;
 * inner = tweeners run in parallel within their step"). `remaining` tracks
 * how far past each step's start `elapsed` is — negative once a step has not
 * started yet, `evaluateEase`'s clamp then holds that property at its
 * pre-step value, matching a `Tween` that has not reached that leg.
 */
export function sampleSpriteShaderTween(
  effect: SpriteShaderEffect,
  elapsed: number,
): { scale: number; modulateAlpha: number } {
  let scale = effect.scale;
  let modulateAlpha = effect.modulateAlpha;
  let remaining = elapsed;
  for (const step of effect.tween) {
    const fromScale = scale;
    const fromAlpha = modulateAlpha;
    for (const tw of step) {
      const t = tw.duration > 0 ? remaining / tw.duration : remaining >= 0 ? 1 : 0;
      const eased = evaluateEase(t, tw.trans, tw.ease);
      if (tw.property === "scale") scale = fromScale + (tw.to - fromScale) * eased;
      else if (tw.property === "modulateAlpha") modulateAlpha = fromAlpha + (tw.to - fromAlpha) * eased;
    }
    const stepDuration = step.reduce((m, tw) => Math.max(m, tw.duration), 0);
    remaining -= stepDuration;
  }
  return { scale, modulateAlpha };
}
