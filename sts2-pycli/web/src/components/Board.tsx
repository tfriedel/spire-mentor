import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { assetUrl, type Card, type CombatLogRow, type Enemy, type MctsConfig, type Score, type Step } from "../api";
import Combatant from "./Combatant";
import { DEFAULT_DYING_FADE_SECONDS, hpBarWidth, spriteBodyH } from "./combatantMetrics";
import KaiserCrabGroup, { type ArmAt, type CrabArm } from "./KaiserCrabGroup";
import type { PartChrome } from "./partChrome";
import { CRAB_ROOT_DX, CRAB_ROOT_Y, CRAB_SETUP_SCALE, crabArmMount } from "./crabMount";
import DecimillipedeGroup, { SEGMENT_SCALE } from "./DecimillipedeGroup";
import { decimillipedeDrop, decimillipedeSpan } from "./decimillipedeLayout";
import MctsControls from "./MctsControls";
import Hand from "./Hand";
import CardFrame from "./CardFrame";
import PileViewer from "./PileViewer";
import { isBrowserChord, isTextInput } from "../keys";
import { enemyLabeller } from "../enemyLabels";
import { stageFraming, stageZoom } from "../stageZoom";
import { StageZoomContext, isSmallPhoneScreen, phoneGroundLimit, usePhone } from "../phone";
import { CHROME, SPLIT, heldSplitHeight, shiftSplit, splitChromeBelow, splitLayout, splitPowerCells, splitZoomWidth, type SplitCreature, type SplitLayout, type SplitPlace } from "../portraitSplit";
import { ZOOM_GUTTER, portraitZoomLeft } from "../zoomPlacement";
import { rigBox } from "../rigBounds";
import PlayerBand from "./PlayerBand";
import { useInspect, useInspectSuppress } from "../inspect/useInspect";
import InspectSheet from "../inspect/InspectSheet";
import {
  ROOM_BG_OFFSET_X, ROOM_CANVAS_W, ROOM_CANVAS_H, SCENE_CY, SCENE_W,
  aspectFit, boundsWidth, cameraFor, fitScene, lowestFootY, instanceScale, placeEnemies, placeOsty, placePlayer, rowHalfWidths,
  renderedSize, roomCanvasScale, sceneFitHeight, sceneFramingFor, type Camera, type Slot,
} from "../scene";
import { deathBurstRadiusFor, dissolveSecondsFor } from "../effects/dissolve";
import { useHints } from "../hints/useHints";
import { useEffectsMode } from "../effects/useEffectsMode";
import { attackingEnemies } from "../effects/enemyTurn";
import { actionStream } from "../effects/actionStream";
import { EffectScheduler, rafClock, type Clock } from "../effects/scheduler";
import { AUTO_CARD_SHOW_MS, HIT_SPACING_MS, TOTAL_SETTLE_MS } from "../effects/timing";
import { loadManifest, type DissolveEffect, type VfxManifest } from "../effects/manifest";
import {
  loadBindings, effectsForEvent, reportSubstitution, deathFadeFor, shouldFadeAfterDeath,
  type Bindings,
} from "../effects/bindings";
import { nullSink, type EffectSink } from "../effects/sink";
import { crabRigEventVfx } from "../effects/crabRigEvents";
import type { EffectEvent, TargetRef } from "../effects/types";
import VfxLayer from "./VfxLayer";
import { useSound } from "../sound/useSound";
import { loadSfxMap } from "../sound/sfxMap";
import { declaredGameVersion } from "../gameVersion";
import { powerValenceLookup, type PowerValence } from "../sound/powerValence";
import {
  SHAKE_DURATION,
  SHAKE_STRENGTH,
  screenShakeKeyframes,
  shakeStrengthForDamage,
  type ShakeStrength,
} from "../effects/screenShake";
import { cuesForCombat, cuesForEvent, transitionCues } from "../sound/triggers";
import type { SoundCue } from "../sound/types";
import { cueForUi } from "../sound/runCues";
import TargetingArrow from "./TargetingArrow";
import { DESIGN_HEIGHT, type Point } from "./arrowGeometry";
import { artSlugs, atmosphereVars, pickScene, resolveScene, sceneAtmosphere,
  sceneScale } from "./roomScenes";
import { matchEncounter, nextSpawnSlot, slotNamed } from "./encounterSlots";
import RoomLayers from "./RoomLayers";
import { useAdviceMarks } from "../advice/useAdviceMarks";
import { END_TURN_KEY, GRADE_TEXT, type Grade } from "../advice/marks";
import { roomArtUrls, sceneArtUrls, useArtReady } from "./roomPreload";
import ZoomCard from "./ZoomCard";
import { heldIndex, hold, tapCard, tapEnemy, tapStage, type Held, type TapResult } from "../handSelect";
import { useZoomPlacement } from "../useZoomPlacement";

function needsTarget(card: { target_type?: string }): boolean {
  const t = card.target_type ?? "";
  return t.includes("Enemy") && t !== "AllEnemies";
}

type PileKind = "draw" | "discard" | "exhaust";

/** A rig's drawn width in scene px as a box SYMMETRIC about its root (a slot
 *  centres the root, so a scythe reaching far left counts on both sides), or
 *  `undefined` when the rig has no measured bounds. */
function rootSymWidth(rig: string | null | undefined, rigScale: number | null | undefined): number | undefined {
  const b = rigBox(rig);
  const sc = Math.abs(rigScale ?? 0);
  if (!b || !sc) return undefined;
  return 2 * Math.max(-b.x, b.x + b.w) * sc;
}

/** A rig's measured idle ink box in scene px: its width and its centre right
 *  of the root. `undefined` without measured bounds, or for a rig drawn
 *  mirrored (a negative scale), whose box this does not flip. */
function inkBox(rig: string | null | undefined, rigScale: number | null | undefined): { w: number; cx: number } | undefined {
  const b = rigBox(rig);
  if (!b || !(rigScale != null && rigScale > 0) || !(b.w > 0)) return undefined;
  return { w: b.w * rigScale, cx: (b.x + b.w / 2) * rigScale };
}

/** The phone-portrait split's figure gap: px between the stage's left edge,
 *  the figure, Osty and the status block (real px). */
const SPLIT_FIG_GAP = 8;
/** Half the band's 34 px block shield, which overhangs the status block's
 *  left edge (PlayerBand). */
const SPLIT_SHIELD_HALF = 17;
/** Real px Osty's HP bar and nameplate take under its feet in the band. */
const SPLIT_OSTY_CHROME = 44;
/** The narrowest Osty's HP bar gets in the band, real px. */
const SPLIT_OSTY_MIN_W = 60;

/**
 * Which creature this is, for anything that must survive the fight changing
 * shape — the row order, the width and slot a creature was handed, the React
 * key its rig hangs off.
 *
 * NOT `index`. That is a position in the encounter ARRAY, and the array is kept
 * in the game's slot order (`CombatState.SortEnemiesBySlotName`, re-run on
 * every add), so a mid-fight spawn INSERTED ahead of a creature renumbers it:
 * an Ovicopter that laid three eggs into `egg5`/`egg4`/`egg3` goes from index 0
 * to index 3 in one step. Keying the per-fight memo by `index` therefore handed
 * the Ovicopter's slot to whichever hatchling inherited index 0 and stood the
 * Ovicopter itself in an egg slot — the row visibly shuffling on a lay.
 *
 * `uid` is the engine's `CreatureUid` and never moves. The `index` fallback is
 * for a producer that predates the field (`uid` is optional on the wire); it
 * restores the old behaviour rather than crossing the two numberings, which is
 * why the two forms are tagged apart.
 */
const enemyKey = (e: Enemy): string => (e.uid != null ? `u${e.uid}` : `i${e.index}`);

/** The game's own pile-button art (ui/combat_ui/): a card-stack icon per pile
 *  plus the red count gem overlaid in a corner. */
const PILE_ART: Record<PileKind, string> = {
  draw: "ui/combat_ui/draw_pile.png",
  discard: "ui/combat_ui/discard_pile.png",
  exhaust: "ui/combat_ui/exhaust_pile.png",
};

/** In-scene pile button (draw bottom-left, discard bottom-right, like the
 *  game's corner piles): the real card-stack art with the count gem riding a
 *  corner — still the clickable pile-viewer button. */
function PilePlaque({ label, count, kind, onClick }: {
  label: string; count: number; kind: PileKind; onClick: () => void;
}) {
  return (
    // gs-pilebtn is kept as a bare selector hook (no styles) — external tooling
    // (the gauntlet capture harness) locates the pile buttons by it.
    <button type="button" onClick={onClick} className={`gs-pilebtn gs-pileplaque gs-pile-${kind}`}
      aria-label={`${label} ${count}`} title={`${label} pile (${count})`}>
      <img className="gs-pileplaque-art" src={assetUrl(PILE_ART[kind])} alt="" aria-hidden="true"
        draggable={false} />
      <span className="gs-pileplaque-count" aria-hidden="true">
        <img src={assetUrl("ui/combat_ui/pile_button_count.png")} alt="" draggable={false} />
        <span>{count}</span>
      </span>
    </button>
  );
}

/** Characters with extracted layered energy-orb art under
 *  ui/combat/energy_counters/<char>/. */
const ORB_CHARACTERS = new Set(["ironclad", "silent", "defect", "regent", "necrobinder"]);

/** Derive the orb-art character from the backend's energy icon path
 *  (`sprite_fonts/<char>_energy_icon.png`) — the step payload carries no
 *  explicit character name in combat. Unknown/missing → ironclad. */
function orbCharacter(energyImage?: string | null): string {
  const m = /([a-z_]+)_energy_icon\.png$/.exec(energyImage ?? "");
  const c = m?.[1] ?? "";
  return ORB_CHARACTERS.has(c) ? c : "ironclad";
}

/** One-shot clip candidates for a rig whose clips are suffixed by revive form
 *  (the Test Subject's `attack_big1/2/3`, `hurt1/2/3`, `knockout1/2/3`). Most
 *  specific first; SpineSprite plays the first name the skeleton actually has,
 *  falling back to the unsuffixed prefix match for rigs without forms. */
function formClips(base: string, form: number): string[] {
  if (base === "attack") return [`attack_big${form}`, `attack_double${form}`, `attack${form}`, "attack"];
  if (base === "die") return ["die", `knockout${form}`, "death"];
  return [`${base}${form}`, base];
}

/** Which SOURCE currently arms one rig's one-shot clip, and that source's own
 *  nonce. `null` = nothing armed. */
type OneShotTrigger = { src: string; nonce: number } | null;

/**
 * Turn per-rig one-shot triggers into EDGES, so a clip replays only when a
 * trigger actually fires.
 *
 * Every rig's `actionNonce` is picked from several independent sources — a
 * death, an enemy's lunge, the hurt flag on `fx`, the player's `attack` pulse
 * — by a priority chain, and `SpineSprite` replays its clip on ANY change of
 * that value. Those two facts do not compose: when the winning source goes
 * away the slot falls back to a lower-priority one whose nonce has not moved,
 * and handing the slot back is indistinguishable from a fresh trigger.
 *
 * That is the "Ironclad punches the air on End Turn" bug. `attack` is a pulse
 * `App` never clears (`liveAttack`), so it sat there holding the nonce of the
 * Strike played BEFORE End Turn; when the enemy's hit finished and the state
 * effect's `setFx({})` dropped the player's flinch, the slot fell back to that
 * nonce and the rig swung a second time at nobody. The struck enemy flinched
 * again for the same reason, and so did an enemy whose own lunge timed out
 * while an older `fx` entry was still on it.
 *
 * The rule that closes it: fire only when the selected source's nonce differs
 * from the last nonce this rig fired FOR THAT SAME SOURCE. Falling back to a
 * source already played is then silent, while a value that genuinely moved —
 * including one that moved while it was being outranked — still fires. The
 * nonce handed out is a plain counter, which also removes a latent hazard in
 * feeding the raw ones straight through: they come from different number
 * lines (`fx` counts up from 0, `attack` is a `Date.now()`), so a switch
 * between sources could in principle repeat a value and swallow a real hit.
 *
 * Triggers already armed at MOUNT are seeded as fired, never played: a board
 * that has just appeared did not witness them. Without that, `liveAttack`
 * surviving the end of a fight made the next fight open with the player
 * lunging at an enemy it had not touched.
 */
function useOneShots(triggers: Record<string, OneShotTrigger>): Record<string, number | undefined> {
  const fired = useRef<Map<string, number> | null>(null);
  if (fired.current == null) fired.current = firedKeys(triggers);
  const counter = useRef(0);
  const [nonces, setNonces] = useState<Record<string, number>>({});
  // `triggers` is rebuilt on every render, so this effect runs on every commit
  // — which is what it is for. The body is idempotent: it calls `setNonces`
  // only when a trigger really moved, so it settles in one extra pass rather
  // than looping.
  useEffect(() => {
    const bumped: Record<string, number> = {};
    for (const [rig, t] of Object.entries(triggers)) {
      if (!t) continue;
      const key = firedKey(rig, t);
      if (fired.current!.get(key) === t.nonce) continue;
      fired.current!.set(key, t.nonce);
      bumped[rig] = ++counter.current;
    }
    if (Object.keys(bumped).length > 0) setNonces((cur) => ({ ...cur, ...bumped }));
  }, [triggers]);
  return nonces;
}

const firedKey = (rig: string, t: { src: string }) => `${rig}:${t.src}`;

function firedKeys(triggers: Record<string, OneShotTrigger>): Map<string, number> {
  const m = new Map<string, number>();
  for (const [rig, t] of Object.entries(triggers)) if (t) m.set(firedKey(rig, t), t.nonce);
  return m;
}

/** A trigger, or `null` when that source has no nonce to fire on. */
const trig = (src: string, nonce: number | undefined): OneShotTrigger =>
  nonce == null ? null : { src, nonce };

/** One arrow's endpoints in board-relative px: where it hangs off the card,
 *  and what it points at. */
interface Aim { from: Point; tip: Point }

/** How long a corpse may sit on the board waiting for its `death` event
 *  before it is swept. Comfortably past one action's playback (`TOTAL_SETTLE_MS`
 *  plus its per-hit pacing) so it can only ever fire when the event was lost.
 *  Also comfortably past the FADING half of a corpse's life (Task 14: the
 *  WebGL dissolve's 2.5s ceiling plus its own small removal buffer, well
 *  under half of this) — this timeout only ever governs the NOT-yet-fading
 *  half, but re-check it here if the dissolve's own duration ever grows.
 *  The per-monster duration (`dissolveSecondsFor`) can only take a creature
 *  BELOW that ceiling, so it does not move this bound. */
const STUCK_CORPSE_MS = 4000;

/** One target's transient combat-fx state — see `applyEvent`. */
type Fx = NonNullable<Parameters<typeof Combatant>[0]["effects"]>;

/**
 * Which colour/drift a `powerAdd` panel gets, given the valence
 * `powerValence.ts` resolved for this power AT THIS AMOUNT — the SAME rule
 * `sound/triggers.ts`'s `powerCueKey` applies, so a negative Strength goes red
 * and plays the debuff chime together rather than disagreeing. The fallback for
 * a power the table doesn't (yet) classify is unchanged: a gain reads as a
 * buff.
 *
 * `triggers.ts` is deliberately not imported here for the direction — it also
 * INVERTS for `powerRemove`, which is a SOUND-only distinction; `powerRemove`'s
 * panel has no colour at all (Task 12 addendum #3), so that inversion has
 * nothing to apply to.
 */
function powerPopKind(valence: PowerValence | undefined): "buff" | "debuff" {
  return valence === "Debuff" ? "debuff" : "buff";
}

export default function Board({
  step,
  onPlayCard,
  onEndTurn,
  onHint,
  onPeek,
  attack,
  scores,
  onPlayRecommended,
  onAutoplay,
  onAutoplayToEnd,
  onSearch,
  hintBusy,
  moveGrade,
  showScores = false,
  hintConfig,
  onHintConfig,
  enemyAnim,
  actionNode,
  clock = rafClock,
}: {
  step: Step;
  onPlayCard?: (cardIndex: number, targetIndex?: number) => void;
  onEndTurn?: () => void;
  onHint?: () => void;
  /** P: peek at the AI's next action (Coach style; press-first). */
  onPeek?: () => void;
  /** The player just struck this enemy: cues the attack lunge on the player rig
   *  and the hurt clip on the target. NOT an arrow — the game draws none once a
   *  card resolves, and drawing one here put a second arrow on the board. */
  attack?: { target: number; nonce: number };
  scores?: Score[];
  onPlayRecommended?: () => void;
  onAutoplay?: () => void;
  onAutoplayToEnd?: () => void;
  onSearch?: () => void;
  hintBusy?: boolean;
  /** The verdict over the card just played (advice/marks.ts `gradeMove`),
   *  floating up from the hand; `nonce` restarts it. */
  moveGrade?: { grade: Grade; nonce: number } | null;
  /** Live play: reveal the per-card win-prob overlays for THIS position. True
   *  when the global hints switch is on, or when the player just asked for a
   *  hint here (H / the hint button). Replay ignores it — see `shownScores`. */
  showScores?: boolean;
  hintConfig?: MctsConfig;
  onHintConfig?: (config: MctsConfig) => void;
  // Dev/preview only: force a looping animation on enemies (e.g. "hurt") so a
  // rig's full motion envelope can be eyeballed for framing/crop checks.
  enemyAnim?: string;
  /** The timeline node whose action produced `step` — see `combatlog.ts`'s
   *  `activeNode`. Its `combat_log` is THIS action's per-hit rows, and is what
   *  drives the effect stream below. Deliberately NOT `step.state.combat_log`:
   *  that field does not exist on a workbench live step, and where it does
   *  exist (replayed / real-game bridge steps) it is the whole fight, so
   *  feeding it to `actionStream` would replay every hit since turn 1. */
  actionNode?: { id: number; parent_id: number | null; combat_log?: CombatLogRow[] } | null;
  /** Injectable clock for effect playback. Tests drive it by hand; production
   *  uses `rafClock`. */
  clock?: Clock;
}) {
  const s = step.state;
  const p = s.player;
  const phone = usePhone();
  // The card armed for a target, and (PHONE only) the hand card shown zoomed
  // (ZoomCard). Selection lives here rather than in Hand so one tap can both
  // zoom a card and arm it — see handSelect.ts. `selectedCard` is always null
  // on a desktop or a touch tablet.
  //
  // Both are held TOGETHER WITH THE STATE THEY WERE PICKED IN (`Held`) and
  // read back only against that same state. A hand index can be a position,
  // so once `s` moves on (the card was played or discarded, the turn ended)
  // the same number may name a different card; clearing it in an effect left
  // one painted frame zooming / arming that other card. A desktop's armed
  // card keeps its old behaviour and survives a state change.
  const sRef = useRef(s);
  sRef.current = s;
  const [pendingHeld, setPendingHeld] = useState<Held<typeof s>>(null);
  const [selectedHeld, setSelectedHeld] = useState<Held<typeof s>>(null);
  const setPendingCard = useCallback((i: number | null) => setPendingHeld(hold(i, sRef.current)), []);
  const setSelectedCard = useCallback((i: number | null) => setSelectedHeld(hold(i, sRef.current)), []);
  const pendingCard = phone ? heldIndex(pendingHeld, s) : pendingHeld?.index ?? null;
  const selectedCard = phone ? heldIndex(selectedHeld, s) : null;
  const [hoverEnemy, setHoverEnemy] = useState<number | null>(null);
  const [pile, setPile] = useState<PileKind | null>(null);
  // Memoized so a state without enemies doesn't hand the callbacks below a
  // fresh [] (and a fresh identity) every render.
  const enemies = useMemo(() => s.enemies ?? [], [s.enemies]);

  // ---- the room this fight is painted in ------------------------------------
  //
  // Resolved up here, not in the JSX below, because the layer stack decides
  // whether the room may be drawn AT ALL yet: a room is up to 37 separate PNGs
  // and a cold cache paints them one at a time, so the fight would start in
  // front of a room still assembling itself. `useRoomArtReady` holds the whole
  // stack back until the browser has all of it (see roomPreload.ts); until then
  // the stage shows its own gradient and the room fades in as one piece.
  const roomScene = pickScene(step.meta?.act_name, enemies);
  // The map node this fight stands on, when the producer knows it: that is
  // what lets the room roll the SAME layer variants the real fight did (see
  // roomScenes.resolveScene / megaRng).
  const bgSeed = step.meta?.bg_seed;
  const bgRow = step.meta?.bg_row;
  const bgCol = step.meta?.bg_col;
  // A seed past 2^53 arrives as a decimal string (GH #878), so it is exact.
  const seedOk =
    typeof bgSeed === "number" || (typeof bgSeed === "string" && /^\d+$/.test(bgSeed));
  const bgRoll =
    seedOk && typeof bgRow === "number" && typeof bgCol === "number"
      ? {
          seed: bgSeed,
          row: bgRow,
          col: bgCol,
          gameVersion: declaredGameVersion(step),
        }
      : null;
  const room = resolveScene(roomScene, step.meta?.act, step.meta?.floor, bgRoll);
  // The whole fight, not just its room: the rigs of everyone on stage and the
  // opening hand's portraits are gated with the layer stack, keyed to the
  // FIGHT (floor + who is in it) so that a card drawn later pops in as it
  // always did instead of hiding a scene that is already up.
  const roomUrls = roomArtUrls(assetUrl, room.far, room.near);
  const sceneUrls = sceneArtUrls(
    roomUrls,
    [s.player?.rig, s.osty?.rig, ...enemies.map((e) => e.rig)],
    s.hand.map((c) => c.image),
  );
  const fightKey = [
    step.meta?.floor ?? "",
    roomUrls.join(","),
    ...enemies.map((e) => `${e.index}:${e.rig ?? e.image ?? e.name}`),
  ].join("|");
  const roomReady = useArtReady(sceneUrls, fightKey);

  // Live modified per-hit damage to show on an attack card, for the target it's
  // currently aimed at: the lone enemy when there's only one, otherwise the
  // enemy being hovered while THIS card is the one awaiting a target. Returns
  // null when no target is determined (card text stays at its base value).
  const previewDamageFor = useCallback((card: Card): number | null => {
    const dp = card.damage_preview;
    if (!dp) return null;
    const target =
      enemies.length === 1 ? enemies[0].index
      : pendingCard === card.index && hoverEnemy != null ? hoverEnemy
      : null;
    if (target == null) return null;
    const v = dp[String(target)];
    return v == null ? null : v;
  }, [enemies, pendingCard, hoverEnemy]);

  // Global hints switch (default OFF). It governs advice the app shows you
  // WITHOUT being asked — here, the per-card win-prob overlays that would
  // otherwise appear on their own. The controls below (hint, play best, the
  // search-tree inspector, the auto-hint toggle, the sims knobs, autoplay,
  // play to end) are all press-first: nothing is revealed until you click, so
  // they stay available with the switch off. `showScores` carries the "you
  // just asked for a hint here" case down from the parent.
  // Phone: tap-to-inspect. Off while a card is armed for a target, when a tap
  // on a creature means "target this" (the provider's `disabled` rule, which
  // needs Board's state, so it is registered rather than passed as a prop).
  const inspect = useInspect();
  useInspectSuppress(pendingCard != null);
  // The open item is a snapshot: close it whenever the combat state moves (a
  // power expires, an enemy dies, the turn ends) rather than show stale text.
  const closeInspect = inspect.close;
  useEffect(() => { closeInspect(); }, [s, closeInspect]);
  const hints = useHints();
  const { mode } = useEffectsMode();
  const sound = useSound();
  // Held in a ref rather than added to the diff effect's dependency array:
  // `sound` is a memoized context value that changes whenever the volume
  // slider, the on/off switch, or the loaded map changes, and that effect's
  // body sets damage numbers, flashes, shakes and 1-second timeouts — adding
  // `sound` would re-run all of that (re-flashing the board) on every volume
  // tick. The ref is refreshed every render so the effect still reads the
  // current emitter without depending on it.
  const soundRef = useRef(sound);
  soundRef.current = sound;
  // Per-event sounds for `full` mode, keyed by the event OBJECT the scheduler
  // later hands to `applyEvent` (it passes the same objects through). Weak, so
  // an event that has fired holds nothing.
  const eventCuesRef = useRef(new WeakMap<EffectEvent, SoundCue[]>());
  const prevStateRef = useRef<typeof s | null>(null);
  // The `actionNode` whose rows the effect below last played. See its use
  // there: it is how a slice that has caught up with the state is told from
  // the previous action's, and an action from a seek.
  const playedNodeRef = useRef<number | null>(null);
  const [fx, setFx] = useState<Record<string, Fx>>({});
  const fxNonce = useRef(0);
  // The card the engine just auto-played (Stampede, Havoc, Mayhem, …), shown
  // over the stage for a moment — see `applyEvent`'s `cardPlay` arm.
  const [autoCard, setAutoCard] = useState<{ card: Card; nonce: number } | null>(null);
  // Enemies drawn as corpses: killed by the action being played, so already
  // gone from `s.enemies`. `fading` is the second half of a death — see the
  // render-phase mount below and `applyEvent`'s `death` arm.
  const [dying, setDying] = useState<{ enemy: Enemy; nonce: number; fading: boolean;
    /** `performance.now()` when `fading` flipped true — the dissolve tween's
     *  own clock (`SpineSprite`'s `dissolve.startedAt`). `undefined` while
     *  not yet fading. */
    fadedAt?: number }[]>([]);
  // Enemies playing their attack-lunge clip this enemy turn (index -> nonce).
  const [enemyActions, setEnemyActions] = useState<Record<number, number>>({});
  const dyingTimers = useRef<number[]>([]);
  useEffect(() => () => { dyingTimers.current.forEach((t) => window.clearTimeout(t)); }, []);

  // The enemies drawn LAST render. A killing blow removes its target from
  // `s.enemies` for the WHOLE action (`rust_game.py:2078` skips dead
  // creatures), so this is the only place its board entry can still be found.
  const shownEnemiesRef = useRef<Enemy[]>(enemies);
  // Enemies that have left `enemies` but whose `dying` entry has not landed
  // yet. Written AND read during render, so a corpse is on the board on the
  // very frame its creature leaves it. The `setDying` below cannot manage that
  // alone: it is a render-phase update, and StrictMode re-invokes the render
  // from the PRE-update state, so the pass React actually commits saw an empty
  // `dying`. For one committed frame the board then held neither the live
  // creature nor its corpse — which unmounted `<KaiserCrabGroup>` and made the
  // remount re-fetch the crab's four atlas pages, ~5 MB, right as its death
  // clip should have been playing (GH #668). Emptied again as soon as `dying`
  // carries the entry, so nothing here outlives the state catching up.
  const pendingCorpsesRef = useRef<Enemy[]>([]);
  // Which fight `shownEnemiesRef` / `dying` belong to — see the corpse mount
  // below, where a change means "different encounter", not "everyone died".
  const combatKeyRef = useRef<string>(`${step.meta?.act ?? ""}|${step.meta?.floor ?? ""}`);
  // The enemy row's per-fight slot assignment, keyed by `enemyKey` — every
  // enemy this fight has put on the board. Filled during render, where the row
  // is laid out — see the row-layout block below for why the layout is a
  // per-fight memo rather than a function of the live roster.
  const rowRosterRef = useRef<{ key: string; widths: Map<string, number>;
    /** `enemyKey` -> the authored slot it was handed, for a fight whose
     *  encounter authors slots. Given out once and kept, like the widths. */
    slots: Map<string, string>;
    /** The row's own left-to-right order, as `enemyKey`s. Appended to as the
     *  fight puts new creatures on the board; a spawn goes in beside its live
     *  neighbours, never at the end just because it arrived last. */
    order: string[];
    /** `enemyKey` -> how far its rig reaches above the ground line, in scene
     *  px (`spriteBodyH`, size jitter included). Remembered for the fight like
     *  the widths, so the scene framing it caps (`sceneFramingFor`) holds
     *  still when the tall one dies — a corpse leaving must not zoom the room. */
    crowns: Map<string, number>;
    /** `enemyKey` -> its drawn width symmetric about its root, scene px
     *  (`rootSymWidth`), for the portrait split; kept for the fight like the
     *  widths. */
    draws: Map<string, number>;
    /** `enemyKey` -> its rig's measured idle ink box, scene px: width and
     *  centre right of the root (`inkBox`), for the portrait split. */
    inks: Map<string, { w: number; cx: number }>;
    /** A multi-rig boss's part count (`rig_group` -> the most parts this
     *  fight has shown), for the portrait split: each part's share of the
     *  boss's slot is cut by it, so a part's death never re-shares it. */
    parts: Map<string, number>;
    /** Every Decimillipede segment rig this fight has shown: the split
     *  stands the animal on their span, so a segment leaving does not move
     *  the others. */
    segRigs: Set<string> }>({ key: "", widths: new Map(), slots: new Map(), order: [], crowns: new Map(), draws: new Map(),
    inks: new Map(), parts: new Map(), segRigs: new Set() });
  // Mount the corpse DURING RENDER — not from the state effect below.
  //
  // The scheduler emits an action's FIRST event synchronously inside that
  // effect, and `VfxLayer`'s sink resolves a spawn point by querying
  // `[data-enemy-index=N]`. A `setDying` from inside the effect commits too
  // late (measured: a render-phase update lands in the DOM before passive
  // effects run; a layout effect's and a `flushSync`'s do not), so the killing
  // hit would find no node, `anchorRect` would return `null`, and NO impact
  // VFX would play on any kill — the most dramatic hit of every fight was the
  // one with no effect.
  //
  // Mounted NOT fading: the scheduled `death` event, one `hurt` beat behind
  // the blow that caused it, is what starts the fade and the rig's `die` clip.
  // Gated exactly like that effect (`onPlayCard`, `mode !== "off"`), because
  // that is what guarantees a `death` event will arrive to take it down again
  // — `combatDiff` emits one for every enemy in `prev` and absent from `next`.
  const liveEnemyKeys = new Set(enemies.map(enemyKey));
  // A fight the board has never seen — the next floor, or a rewind onto a
  // different combat — swaps the WHOLE roster in one render while this
  // component stays mounted. Every creature of the old fight is then "gone",
  // and the corpse mount below would stand each of them up beside the new
  // encounter, at the old fight's HP, until the stuck-corpse sweeper took them
  // away ~a second later: the user's "why do I see two Axebots?" (2026-09-12),
  // the ghost being the PREVIOUS Axebot (its own rolled max HP, hence 73 next
  // to 74). A fight change is not a death — drop the bookkeeping rather than
  // mourn it.
  const combatKey = `${step.meta?.act ?? ""}|${step.meta?.floor ?? ""}`;
  const combatChanged = combatKeyRef.current !== combatKey;
  if (combatChanged) {
    combatKeyRef.current = combatKey;
    shownEnemiesRef.current = enemies;
    pendingCorpsesRef.current = [];
  }
  if (combatChanged && dying.length > 0) setDying([]);
  if (!combatChanged && onPlayCard && mode !== "off") {
    const gone = shownEnemiesRef.current.filter((e) => !liveEnemyKeys.has(enemyKey(e)));
    if (gone.length > 0) {
      pendingCorpsesRef.current = [
        ...pendingCorpsesRef.current.filter(
          (p) => !gone.some((g) => enemyKey(g) === enemyKey(p)),
        ),
        ...gone,
      ];
      setDying((d) => [
        ...d,
        ...gone
          .filter((e) => !d.some((x) => enemyKey(x.enemy) === enemyKey(e)))
          .map((e) => ({ enemy: e, nonce: ++fxNonce.current, fading: false })),
      ]);
    }
  }
  shownEnemiesRef.current = enemies;
  // Which of them still need covering is decided per render, at the use site
  // below, NOT by pruning the ref here. A render-phase update makes React
  // re-run this function from the pre-update state, so a prune that ran on the
  // superseded pass would have emptied the ref before the pass React commits
  // ever read it — which is the same one-frame hole in a new place. The ref
  // only ever grows during render; the effect further down empties it once the
  // commit has happened.

  // Empty the pending-corpse ref once the commit has happened. Doing it here
  // rather than during render is what keeps a discarded render pass from
  // clearing an entry the committed pass still needs (GH #668); the render
  // side only appends, so without this the ref would keep every creature the
  // fight ever killed. No dependency array on purpose — "after every commit,
  // drop whatever that commit caught up with" is the whole rule, and the list
  // it filters is at most one dying encounter long.
  useEffect(() => {
    pendingCorpsesRef.current = pendingCorpsesRef.current.filter(
      (p) =>
        !enemies.some((e) => enemyKey(e) === enemyKey(p)) &&
        !dying.some((d) => enemyKey(d.enemy) === enemyKey(p)),
    );
  });

  // Safety net for a corpse whose `death` event never arrives (an action
  // superseded mid-flight, a board that changed without one). A not-yet-fading
  // corpse must not outlive one action's playback; without this it would sit
  // on the board until the fight ended.
  useEffect(() => {
    const stuck = dying.filter((d) => !d.fading);
    if (stuck.length === 0) return;
    const t = window.setTimeout(() => {
      setDying((cur) => {
        const kept = cur.filter((x) => x.fading || !stuck.some((k) => k.nonce === x.nonce));
        // Same array when nothing was stuck after all (the death landed while
        // the timer was pending), so this cannot re-trigger its own effect.
        return kept.length === cur.length ? cur : kept;
      });
    }, STUCK_CORPSE_MS);
    return () => window.clearTimeout(t);
  }, [dying]);

  // The WebGL overlay's sink, and the manifest that names what it can play.
  // Held in a ref (not state) for the same reason `soundRef` is: nothing here
  // should re-run the effect below.
  const sinkRef = useRef<EffectSink>(nullSink);
  /** The Crusher arm's `TargetRef`, for `onCrabRigEvent` to name.
   *  A ref because that callback is built once, long before `crabParts` is
   *  derived each render. `null` when the crab is not on the board. */
  const crabLeftTargetRef = useRef<TargetRef | null>(null);
  const [manifest, setManifest] = useState<VfxManifest | null>(null);
  // `applyEvent` (below) is a `useCallback(..., [mode])` and must not gain
  // `manifest` as a dependency — same reasoning as `bindingsRef`'s own doc:
  // the manifest arrives once, well after mount, and adding it would
  // re-create `applyEvent` (and re-run whatever effect subscribes it) the
  // moment it loads, mid-fight if one is already in progress.
  const manifestRef = useRef<VfxManifest | null>(null);
  manifestRef.current = manifest;
  useEffect(() => {
    let live = true;
    loadManifest()
      .then((m) => { if (live) setManifest(m); })
      // Additive: with no manifest the board keeps all of its CSS effects.
      .catch((err) => console.warn("vfx manifest unavailable", err));
    return () => { live = false; };
  }, []);
  // The card/move -> VFX binding table (design spec §1.4). Held BOTH ways, the
  // same split `manifest` above uses and for the same two reasons: `applyEvent`
  // is a `useCallback(..., [mode])` that must read it without gaining it as a
  // dependency (a ref), and `dissolveEffectFor` reads its `deathFade` section
  // during RENDER (state, so the table arriving re-renders). It was a ref alone
  // until `deathFade` gave it a render-time consumer; a ref alone would have
  // left an enemy that died in the window before the fetch resolved dissolving
  // forever, because nothing would re-render to notice the table had landed.
  //
  // `null` while the fetch is in flight (or on a browser that never gets
  // one) is the SAME additive fallback as `manifest` above —
  // `effectsForEvent` treats it as "keep sub-project 1's slash/block
  // defaults". A failed fetch is NOT that case: it means the table exists
  // but this client couldn't load it (e.g. it 404s because it was never
  // published to the art tree), which would otherwise render
  // vfx_attack_slash for every hit while looking healthy. `console.error`
  // (not `console.warn`, unlike the manifest above) keeps that failure loud
  // rather than indistinguishable from the benign race.
  const [bindings, setBindings] = useState<Bindings | null>(null);
  const bindingsRef = useRef<Bindings | null>(null);
  bindingsRef.current = bindings;
  useEffect(() => {
    let live = true;
    loadBindings()
      .then((b) => { if (live) setBindings(b); })
      .catch((err) => console.error("vfx bindings unavailable — every hit falls back to vfx_attack_slash", err));
    return () => { live = false; };
  }, []);
  const onSink = useCallback((sk: EffectSink) => { sinkRef.current = sk; }, []);

  /** The Kaiser Crab's rig-driven VFX (GH #642).
   *
   * Every other effect on this board is fired from a MODEL-side event — a
   * snapshot diff, a card play, an enemy move. The crab's are fired by its own
   * animation: `NKaiserCrabBossVfx` listens on the rig's spine events, and the
   * one that draws today is `claw_explode_l`, the Crusher's death explosion at
   * t=1.53 s of `left/die`.
   *
   * `anchor: "point"` because the spawn is the live `claw_upper` bone, which at
   * that instant is high in the air and nowhere near the arm's ground-line
   * chrome — see `PlayedEffect.point`. `boneAt` returning `null` (rig not
   * measured, or gone) plays nothing, which is the honest outcome rather than
   * dropping the explosion on the wrong spot.
   *
   * `target` is the Crusher arm so the record says whose effect this is, but
   * nothing reads it for a point play. */
  const onCrabRigEvent = useCallback((
    name: string, boneAt: (bone: string) => { x: number; y: number } | null,
  ) => {
    const vfx = crabRigEventVfx(name);
    if (!vfx) return;
    const point = boneAt(vfx.bone);
    if (!point) return;
    sinkRef.current.play({
      effect: vfx.effect,
      target: crabLeftTargetRef.current ?? { side: "player" },
      anchor: "point",
      point,
      tint: null,
    });
  }, []);

  // Enemies alive going INTO the action the scheduler is currently playing —
  // written once per transition, from the SAME `prev` the event stream below
  // was built from. `applyEvent`'s `death` case reads this rather than
  // `prevStateRef`: that ref is overwritten at the TOP of the state effect,
  // so by the time a scheduled `death` event actually fires (paced behind
  // earlier hits), the dying enemy is already gone from it.
  const dyingSnapshotsRef = useRef<Map<number, Enemy>>(new Map());

  // Icon per power NAME, accumulated across every transition this fight has
  // seen (an icon never changes once extracted, so nothing here is ever
  // invalidated). Needed because a `powerRemove` event's target has ALREADY
  // lost that power by `s` — the icon has to come from wherever the power
  // was last seen, exactly the same "read it before it's gone" reason
  // `dyingSnapshotsRef` above exists.
  const powerIconsRef = useRef<Map<string, string | null>>(new Map());

  // Every power's buff/debuff valence, from the SAME generated table sound
  // already classifies by (`sfx_map.json`'s `powers` — see
  // `sound/triggers.ts`'s `powerCueKey`). Loaded independently of
  // `useSound()`: that context's map is fetched only while sound is ON
  // (`SoundContext.tsx`), and a player can switch it off for good, so a panel
  // that read `sound.map` would show the `gain ⇒ buff` fallback colour
  // forever for them — the visual must not depend on whether (or how) the
  // SOUND feature happens to be configured. `loadSfxMap` is memoised (`sfxMap.ts`'s module-level
  // `cached` promise), so this shares the one fetch with `SoundContext`
  // rather than doubling it.
  const powerValenceRef = useRef<ReturnType<typeof powerValenceLookup>>(powerValenceLookup(null));
  // The shown run's own build's table (GH #877): a power new in a later
  // build is classified by that build's map.
  const sfxVersion = declaredGameVersion(step);
  useEffect(() => {
    let live = true;
    loadSfxMap(fetch, sfxVersion)
      .then((map) => { if (live) powerValenceRef.current = powerValenceLookup(map); })
      // Additive, like `manifest`/`bindings` above: an unclassified power (or
      // a map that never loads) just takes `powerPopKind`'s fallback rather
      // than the panel failing to render at all.
      .catch(() => {});
    return () => { live = false; };
  }, [sfxVersion]);

  /** The board root — the whole-screen shake's target, and the positioned
   *  ancestor every absolute child in this file measures against. */
  const ref = useRef<HTMLDivElement>(null);

  // The whole-board shake (`NGame.ScreenShake`). A ref, not state: the shake is
  // a fire-and-forget animation on one DOM node, and routing it through a React
  // render would re-render every combatant for a transform that belongs to
  // their common ancestor. `pxPerScene` is read through a ref for the same
  // reason `applyEvent`'s other reads are — this callback is memoised with an
  // empty dep list and must not go stale when the stage resizes.
  const pxPerSceneRef = useRef(1);
  /** The encounter camera's scale, for the same reason — the corpse-removal
   *  timer scheduled from `applyEvent` needs it to reproduce the duration the
   *  rig is dissolving at. */
  const cameraScaleRef = useRef(1);
  const effectsModeRef = useRef(mode);
  useEffect(() => { effectsModeRef.current = mode; }, [mode]);
  const fireScreenShake = useCallback((strength: ShakeStrength) => {
    const el = ref.current;
    if (!el) return;
    // `mode` gates it exactly as it gates the per-creature wobble, and
    // `prefers-reduced-motion` kills it the way `game-skin.css:1358` kills
    // every other animation — this one is driven from JS, so the media query
    // has to be asked rather than declared.
    if (effectsModeRef.current !== "full") return;
    if (typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    // `NGame.cs:1325-1332`: both `CreatureCmd` sites omit `degAngle`, so every
    // shake picks a fresh uniform direction — `Rng.Chaotic.NextFloat(360f)`.
    // `Math.random` for the same reason `buildLiveEffect` uses it: this is the
    // UI choosing a seed, not the simulator consuming one.
    const angleRad = Math.random() * 2 * Math.PI;
    const duration = SHAKE_DURATION.short;
    const strengthPx = SHAKE_STRENGTH[strength] * pxPerSceneRef.current;
    // `animate` is absent in jsdom, so a test asserts the ATTRIBUTE below
    // rather than the animation. Guarded rather than polyfilled: a shake that
    // silently does not play is the correct degradation for a headless DOM.
    el.dataset.screenShake = strength;
    window.setTimeout(() => {
      if (el.dataset.screenShake === strength) delete el.dataset.screenShake;
    }, duration * 1000);
    if (typeof el.animate !== "function") return;
    el.animate(screenShakeKeyframes(duration, strengthPx, angleRad), {
      duration: duration * 1000,
      // `replace`: a second hit inside 0.3 s restarts the punch rather than
      // summing two transforms, which is what `NScreenShake` does too — it
      // holds ONE `_shakeInstance` and `Shake` overwrites it.
      composite: "replace",
    });
  }, []);

  // Applies ONE per-hit event as the scheduler dispatches it — this is what
  // turns a 3-hit attack into three separately-timed damage numbers instead
  // of one net total landing all at once.
  const applyEvent = useCallback((e: EffectEvent) => {
    // This event's sounds, resolved when its transition landed (`full` mode
    // only; every other mode voices the whole transition at once). Read off
    // the event object itself, so a drained, superseded event still plays
    // its own transition's sounds.
    const cues = eventCuesRef.current.get(e);
    if (cues?.length) soundRef.current.emit(cues);
    if (e.kind === "cardPlay") {
      // An auto-played card (GH #1155). The game flies it to the play area and
      // resolves it there; without this the board showed only its
      // consequences, and under "End turn" nothing said a card was played at
      // all. Not an `fx` entry: bumping the player's nonce would make them
      // flinch, and the card is not something that happened TO them.
      const an = ++fxNonce.current;
      setAutoCard({ card: e.card, nonce: an });
      const t = window.setTimeout(() => {
        setAutoCard((cur) => (cur?.nonce === an ? null : cur));
        dyingTimers.current = dyingTimers.current.filter((id) => id !== t);
      }, AUTO_CARD_SHOW_MS);
      dyingTimers.current.push(t);
      return;
    }
    if (e.kind === "attack") {
      // The `attack` variant is the actor's lunge (types.ts). The state
      // effect below only ever produces one for an enemy — either from a
      // `monster_move` log row via `actionStream`, or synthesized from the
      // `attackingEnemies` intent-guess for a recording with no log — so
      // either source ends up here and drives the SAME lunge clip.
      if (e.target.side === "enemy") {
        const idx = e.target.index;
        const an = ++fxNonce.current;
        setEnemyActions((cur) => ({ ...cur, [idx]: an }));
        const ea = window.setTimeout(() => {
          setEnemyActions((cur) => {
            if (cur[idx] !== an) return cur; // superseded by a later attack
            const rest = { ...cur };
            delete rest[idx];
            return rest;
          });
          dyingTimers.current = dyingTimers.current.filter((id) => id !== ea);
        }, 1000);
        dyingTimers.current.push(ea);
      }
      return;
    }

    // `powerFlash` is deliberately NOT written into `fx`. Two reasons, both
    // about this block's side effects rather than about the flash itself —
    // its visual is the `vfx_power_flash_vfx` node played through the sink
    // further down, never an `Fx` flag:
    //
    //  - The `nonce` bump REPLAYS the target's one-shot clip. An enemy's
    //    `actionNonce` reads ``fx[`e${index}`]?.nonce`` straight through
    //    while `actionAnim` falls back to "hurt" (see the enemy props
    //    below), so every flash made its owner flinch — a Cultist's Ritual,
    //    a Byrdonis's Territorial, a Zapbot's HighVoltage, Thorns on every
    //    hit received. In game the flash is a tint on the power's own icon:
    //    `NCreature.cs:925` spawns the flash node and triggers nothing on
    //    the creature. (The player side never had this; its `actionNonce`
    //    is already gated on `fx["player"]?.flash`.)
    //  - `cur` is rebuilt from scratch, so an entry that sets no flag would
    //    also DROP whatever `flash` / `blocked` / `powerPop` /
    //    `blockShatter` / `shake` an earlier event in the SAME action put on
    //    this target. One line, one fix: don't write the entry.
    //
    // `heal` is here for exactly the same two reasons, and it is the same bug:
    // it names no branch below, so its entry sets no flag and its only effects
    // are the flinch and the clobber. It is not rarer in kind, only in
    // frequency — a healing monster or a Regent lifesteal recoiled as though
    // struck. The game heals with no creature reaction at all.
    if (e.kind !== "powerFlash" && e.kind !== "heal") {
      const k = e.target.side === "enemy" ? `e${e.target.index}` : e.target.side;
      const nonce = ++fxNonce.current;
      setFx((prevFx) => {
        // Only `dmgs` carries forward — WITHIN one action (the state effect
        // below clears `fx` at the start of each new one, so this never
        // reaches across actions). Every OTHER flag belongs to this event
        // alone: spreading the whole previous entry forward (as this used to)
        // is what let a block-break's `blockShatter` outlive its own event —
        // the shared per-target `nonce` bumps on every later hit and remounts
        // (replays) the shatter node for a hit that has nothing to do with a
        // block breaking.
        const cur: Fx = { nonce, dmgs: prevFx[k]?.dmgs ?? [] };
        // Both of `CreatureCmd.cs`'s shake sites are wired, from here and from
        // the `fullyBlocked` arm below — see `effects/screenShake.ts` for which
        // site is which and why they partition rather than overlap. They are the
        // same global `NGame.ScreenShake`, so both land on `fireScreenShake`.
        //
        // What the game does to the struck CREATURE is `CreatureCmd.cs:324-329`'s
        // `TriggerAnim(receiver, "Hit")` — the rig's own hurt clip, which the
        // rig path already plays — and nothing else. There is deliberately no
        // `cur.shake` here: the creature wobble that used to fire on damage was
        // a stand-in for the screen shake above, and `NCreature.AnimShake` (its
        // real counterpart) is guarded on `!Visuals.IsPlayingHurtAnimation()`,
        // i.e. damage is the one case the game EXCLUDES. It now fires from the
        // `powerAdd` arm below, which is where `NCreature.cs:901` calls it
        // (GH #632).
        if (e.kind === "hurt") {
          cur.dmgs = [...(prevFx[k]?.dmgs ?? []), { amount: e.amount, nonce }];
          cur.dmgKind = "hp";
          cur.flash = true;
          fireScreenShake(shakeStrengthForDamage(e.amount));
        } else if (e.kind === "fullyBlocked") {
          cur.flash = true;
          cur.blocked = true;
          // `CreatureCmd.cs:406`'s fixed Weak, not a damage tier — there is no
          // unblocked damage here to tier by.
          fireScreenShake("weak");
        } else if (e.kind === "blockGain") {
          cur.blockPop = true;
        } else if (e.kind === "blockBreak") {
          cur.blockShatter = true;
        } else if (e.kind === "powerAdd") {
          const valence = powerValenceRef.current(e.power, e.amount);
          cur.powerPop = {
            power: e.power,
            icon: powerIconsRef.current.get(e.power) ?? null,
            mode: "applied",
            kind: powerPopKind(valence),
          };
          // `NCreature.cs:901`: `if (!flag) AnimShake()`, where `flag` is the
          // same `GetTypeForAmount` valence the panel colour above reads — so
          // the wobble, the red panel and the debuff chime cannot disagree
          // about what just landed. Note the C# asks `!flag` rather than
          // `== Debuff`: a power the table cannot classify falls through
          // `powerPopKind`'s buff fallback and gets no shake, which is the same
          // side both surfaces land on.
          if (mode === "full" && valence === "Debuff") cur.shake = nonce;
        } else if (e.kind === "powerRemove") {
          cur.powerPop = {
            power: e.power,
            icon: powerIconsRef.current.get(e.power) ?? null,
            mode: "removed",
          };
        }
        return { ...prevFx, [k]: cur };
      });
    }

    if (e.kind === "death" && e.target.side === "enemy") {
      const idx = e.target.index;
      const dn = ++fxNonce.current;
      // The dissolve tween's own clock (`SpineSprite`'s `dissolve.startedAt`,
      // Task 14) — captured once here, not inside the `setDying` updater
      // (which may run twice under StrictMode and must not restart the
      // dissolve's clock the second time).
      const fadedAt = performance.now();
      setDying((d) => {
        const at = d.findIndex((x) => x.enemy.index === idx);
        if (at >= 0) {
          // Already on the board as a not-yet-fading corpse (mounted at
          // render, so the killing hit's VFX had something to anchor to).
          // THIS event is the death proper: start the fade, and bump the
          // nonce so the rig restarts on its `die` clip.
          return d.map((x, i) => (i === at ? { ...x, nonce: dn, fading: true, fadedAt } : x));
        }
        // Nothing mounted it — a `death` drained from a superseded action,
        // whose target left the board before this Board ever rendered it
        // missing. Put the corpse up already fading.
        const snap = dyingSnapshotsRef.current.get(idx);
        return snap ? [...d, { enemy: snap, nonce: dn, fading: true, fadedAt }] : d;
      });
      // Corpse removal: comfortably past THIS creature's own dissolve so the
      // WebGL dissolve is never cut off mid-fade. Per-monster, matching the
      // duration `dissolveEffectFor` hands the rig — the manifest's 2.5 s is
      // the game's ceiling, and reading it flat here would leave a small
      // monster's corpse standing, fully dissolved and invisible, for seconds
      // after it finished.
      //
      // Falls back to `DEFAULT_DYING_FADE_SECONDS` (the pre-Task-14 CSS
      // timing) when the manifest hasn't loaded yet or lacks the entry — the
      // SAME fallback `Combatant.tsx`'s `dyingChromeFadeSeconds` uses for the
      // chrome fade, so a manifest-less board removes a corpse exactly when
      // its (still-CSS) chrome fade finishes, same as before Task 14.
      const eff = manifestRef.current?.effects["vfx_monster_death"];
      // Read through refs: `applyEvent` is memoised and this fires from the
      // scheduler, so the render-time `camera`/`displayEnemies` are not in
      // scope and would be stale if they were.
      const snap = dyingSnapshotsRef.current.get(idx);
      // `NCreature.cs:1106`'s guard, the same one `dissolveEffectFor` applies
      // at render. Nulling `deathFx` here rather than branching each consumer
      // is what makes the sprite dissolve, the debris burst and the corpse
      // timer agree: in the game they are three parts of ONE node that is
      // never created for these monsters, so there is no state where some of
      // them fire.
      const fadeRow = deathFadeFor(snap?.kind, bindingsRef.current);
      const deathFx = eff && eff.kind === "dissolve" && fadeRow.fade !== "never" ? eff : null;
      const size = snap ? renderedSize({ rig: snap.rig, rigScale: snap.rig_scale }) : null;
      // This creature's own dissolve duration, or `null` when nothing can be
      // measured — the two consumers below want opposite fallbacks for that,
      // so neither gets to see a substituted number.
      const dissolveSeconds =
        deathFx && size && snap
          ? dissolveSecondsFor(
              size.w * instanceScale(snap),
              size.h * instanceScale(snap),
              cameraScaleRef.current,
              deathFx.dissolveSeconds,
              SCENE_W,
            )
          : null;
      const dissolveMs =
        dissolveSeconds !== null
          ? dissolveSeconds * 1000
          : deathFx
            ? deathFx.dissolveSeconds * 1000
            // No dissolve to outlast, so the corpse lives as long as the game
            // waits for the death animation: `DeathAnimLengthOverride` when the
            // monster sets one (`NCreature.cs:1093`), else the pre-Task-14
            // fallback. The two Kaiser Crab arms are why this is not just the
            // fallback — their 2.5 s is currently supplied by the dissolve
            // CEILING happening to be the same number, which is an agreement
            // by coincidence rather than by derivation.
            : (fadeRow.deathAnimSeconds ?? DEFAULT_DYING_FADE_SECONDS) * 1000;
      // The debris burst (`NMonsterDeathVfx`'s `%Particles`, a child of the
      // node the dissolve sprite draws). Fired HERE rather than from the
      // binding table because it is not a bound reaction to an event — it is
      // the other half of `vfx_monster_death`, whose sprite half is already
      // driven straight from these same measurements.
      //
      // Both per-creature numbers come from the same box: the game sizes its
      // `SubViewport` from the creature's padded bounds and derives the
      // emission radius from that size, so a Byrdpip's burst is small and
      // brief and a Fabricator's is neither. Skipped entirely when nothing is
      // measurable — see `PlayedEffect.dissolve`.
      if (deathFx && size && snap && dissolveSeconds !== null) {
        const radius = deathBurstRadiusFor(
          size.w * instanceScale(snap),
          size.h * instanceScale(snap),
          cameraScaleRef.current,
          deathFx.emissionRadiusPerCreatureExtent,
        );
        if (radius !== null) {
          // `center`, matching `NMonsterDeathVfx`'s own origin: it sets
          // `GlobalPosition` to the merged bounds' centre (:196-199), not to
          // the creature node's origin at its feet.
          sinkRef.current.play({
            effect: "vfx_monster_death",
            target: e.target,
            anchor: "center",
            tint: null,
            dissolve: { emissionRadius: radius, lifetime: dissolveSeconds },
          });
        }
      }
      // Unconditional: the updater above runs on a later render (and twice
      // under StrictMode), so nothing here can observe whether it mounted
      // anything. If it did not, this filter is a no-op.
      const rm = window.setTimeout(() => {
        setDying((d) => d.filter((x) => x.nonce !== dn));
        dyingTimers.current = dyingTimers.current.filter((id) => id !== rm);
      }, dissolveMs + 100);
      dyingTimers.current.push(rm);
    }

    // A list, not one pick: 15 owners chain more than one spawn call in the
    // same C# builder and the game plays all of them (see `effectsForEvent`).
    // Only some kinds carry a source; `reportSubstitution` names it in its
    // log line and tolerates its absence.
    const source = "source" in e ? e.source : undefined;
    // `vfx_power_flash_vfx` draws the flashing POWER's own icon
    // (`NPowerFlashVfx.cs:94`: `_sprite.Texture = _power.BigIcon`, assigned
    // at runtime — the manifest carries no texture for this effect at all).
    // `effectsForEvent` is a pure lookup with no access to the icon cache, so
    // this is the one place that can attach it: the same
    // `powerIconsRef`/`p.image` source `powerAdd`/`powerRemove` already read
    // above, keyed by the event's own `power` name.
    const iconUrl = e.kind === "powerFlash" ? (powerIconsRef.current.get(e.power) ?? null) : undefined;
    for (const pick of effectsForEvent(e, bindingsRef.current)) {
      reportSubstitution(source, pick);
      sinkRef.current.play({ effect: pick.effect, target: e.target, anchor: pick.anchor, tint: pick.tint, iconUrl });
    }
    // `fireScreenShake` is itself `useCallback(..., [])`, so naming it here
    // keeps this dep list honest without widening what re-creates `applyEvent`.
  }, [mode, fireScreenShake]);

  // `applyEvent` held in a ref so the scheduler's identity does not depend on
  // it. It is a `useCallback(..., [mode])`, so keying the effect below on it
  // tore the scheduler down on every mode transition — autoplay starting or
  // stopping, fast-step, a `prefers-reduced-motion` change — and `dispose()`
  // clears the queue WITHOUT emitting, contradicting the supersede contract
  // the class documents for itself: a queued `death` or `powerAdd` carries
  // board state that must still land or the board disagrees with the engine.
  const applyEventRef = useRef(applyEvent);
  applyEventRef.current = applyEvent;

  const schedulerRef = useRef<EffectScheduler | null>(null);
  useEffect(() => {
    schedulerRef.current = new EffectScheduler(clock, (e) => applyEventRef.current(e));
    return () => { schedulerRef.current?.dispose(); schedulerRef.current = null; };
  }, [clock]);

  useEffect(() => {
    const prev = prevStateRef.current;
    // Live play only: onPlayCard is set solely in interactive mode. Replay steps
    // are decision-points (net deltas, not play-by-play) → no fabricated juice.
    if (!onPlayCard) {
      prevStateRef.current = null;
      playedNodeRef.current = actionNode?.id ?? null;
      return;
    }

    // Do the log rows on hand belong to the state we were just handed?
    //
    // App lands `setLive(s)` and `refreshTimeline()` in two DIFFERENT renders,
    // so this effect can be invoked with action N's state and action N-1's
    // slice. The timeline's active leaf advances exactly once per action, so
    // an unchanged node id means the rows have not caught up yet.
    const nodeId = actionNode?.id ?? null;
    const caughtUp = actionNode == null || nodeId !== playedNodeRef.current;
    // ...and did we get here by ACTING, rather than by jumping? Seeking moves
    // the leaf without an action happening, and replaying the slice of the
    // node we landed on would re-fire hits the player already watched. Only a
    // CHILD of the node we last played is a step forward.
    const isStep = actionNode == null || actionNode.parent_id === playedNodeRef.current;
    // In `full` mode — the only mode that actually paces per hit — wait for
    // the rows rather than falling back: return WITHOUT consuming `prev`, so
    // the re-run one render later sees the matching pair. Every other mode
    // settles in a single beat anyway, and the autoplay loops refresh the
    // timeline only every 6th step (App's `TIMELINE_STRIDE`), so deferring
    // there would batch six actions into one burst. They use no rows at all
    // and fall back to the snapshot diff — exactly what the board did before
    // per-hit playback existed.
    if (mode === "full" && !caughtUp) return;
    playedNodeRef.current = nodeId;
    prevStateRef.current = s;
    // `prev === s` is not a transition: the effect also re-runs when only
    // `mode` or `actionNode` changed (a timeline refresh with no action, the
    // combat-entry node arriving one render after the board mounted). The
    // snapshot diff was always empty there; now that rows can arrive
    // independently of the state, say so explicitly rather than relying on it.
    if (!prev || prev === s || mode === "off") return; // fx:off settles instantly; autoplay/reduced still shows calm cues
    // Enemies that struck this enemy turn — the intent-sniffing fallback for
    // a recording with no combat log (or one whose rows don't resolve to a
    // board slot). Used below only when the log-driven stream stayed silent.
    const attackers = attackingEnemies(prev, s);

    // Per-hit events from the engine log, reconciled against the snapshot
    // diff for the kinds the log has no entry for (heal, power removal) and
    // for targets it cannot address (Osty). This SUPERSEDES `attackingEnemies`'s
    // intent-sniffing guess whenever the log actually names an attacker (see
    // the `attack` variant's doc comment, types.ts) — a recording with no
    // combat log carries no `attack` events here, so the fallback below
    // covers it with the same guess the board used before this task.
    const rows = caughtUp && isStep ? (actionNode?.combat_log ?? []) : [];
    const events = actionStream(rows, prev, s);
    const streamHasAttack = events.some((e) => e.kind === "attack");
    const hadFallback = !streamHasAttack && attackers.length > 0;
    // No log to pace a per-hit sequence from — just "this enemy struck". The
    // player still reacts (flinch) even when the hit was fully blocked — that
    // yields no `hurt` event, log or no log — unless the stream already
    // described a real player hit itself.
    const fallbackEvents: EffectEvent[] = !hadFallback ? [] : [
      ...attackers.map((index): EffectEvent => ({ kind: "attack", target: { side: "enemy", index } })),
      ...(events.some((e) => (e.kind === "hurt" || e.kind === "fullyBlocked") && e.target.side === "player")
        ? []
        : [{ kind: "fullyBlocked", target: { side: "player" } } as EffectEvent]),
    ];

    // Sound. `sound.emit` is a no-op while the switch is off and applies the
    // fx mode itself. The intent is read from `prev`: by `s` the enemy has
    // already rolled its NEXT move. `enemyName` resolves from `prev` first —
    // a killed enemy is absent from `s` by definition, so a `s`-only lookup
    // would drop every death cue.
    const map = soundRef.current.map;
    if (map) {
      const cueCtx = {
        map,
        enemyName: (i: number) =>
          (prev.enemies ?? []).find((e) => e.index === i)?.name ??
          (s.enemies ?? []).find((e) => e.index === i)?.name,
        enemyIntent: (i: number) => (prev.enemies ?? []).find((e) => e.index === i)?.intents?.[0],
        character: prev.player?.name,
        // The two snapshots, for the cues that are a count moving rather
        // than an `EffectEvent`: the turn banners, card deal/exhaust and
        // the orb queue. See `triggers.ts`'s `rhythmCues`/`orbCues`.
        prev,
        next: s,
      };
      if (mode === "full") {
        // Sound rides the per-hit stream, like the visuals: the transition's
        // own cues (banners, deal, orbs) now, and each event's when the
        // scheduler reaches it (`applyEvent`). A 3-hit attack therefore
        // sounds three impacts in step with its three numbers and shakes,
        // and each enemy's move is heard as it acts — not the whole enemy
        // turn as one chord on the first frame. Resolved HERE, not in
        // `applyEvent`, because a superseded transition's queued events are
        // drained after the next transition's snapshots have replaced these.
        const now = transitionCues(cueCtx);
        if (now.length) soundRef.current.emit(now);
        for (const e of [...fallbackEvents, ...events]) eventCuesRef.current.set(e, cuesForEvent(e, cueCtx));
      } else {
        // Every other mode settles in one beat, so it sounds as one beat —
        // each sound once. With the log, the `attack` events name the exact
        // moves and the intent guess stands down, as it does for the visuals.
        soundRef.current.emit(cuesForCombat(events, { ...cueCtx, attackers: hadFallback ? attackers : [] }));
      }
    }

    if (events.length === 0 && !hadFallback) return;
    // Replace, not merge: this is a NEW action, and each target's fx entry
    // (dmgs, flash, shake, blocked, blockPop, blockShatter, powerPop) belongs
    // to exactly one action. `applyEvent` only ever carries `dmgs` forward
    // across events WITHIN one action; without this clear, a flag set by a
    // PRIOR action's last event stuck around until the settle timeout below
    // finally fired — which gets cancelled and rescheduled on every new
    // action, so a busy fight could leave a shatter/flash showing well past
    // the action that caused it.
    setFx({});
    // The no-log guess is applied immediately (bypassing the scheduler)
    // exactly like the board did before per-hit playback, rather than paying
    // the scheduler's per-kind advance (see timing.ts) for a single guessed beat.
    for (const e of fallbackEvents) applyEvent(e);
    // MERGE, don't replace: `play()` below drains any still-queued events
    // from the PREVIOUS transition first (the scheduler's supersede rule —
    // see scheduler.ts), and a queued `death` among them needs its own
    // snapshot to still be here when it fires, not already overwritten by
    // this transition's `prev`.
    for (const e of prev.enemies ?? []) dyingSnapshotsRef.current.set(e.index, e);
    // Same idea, for power icons: a `powerRemove` fired from THIS transition
    // needs the icon from `prev` (it's already gone from `s`), and a
    // `powerAdd` needs it from `s` (it wasn't in `prev` yet) — so both
    // snapshots get scanned, keyed by the power's display name (the same
    // string `EffectEvent.power` carries).
    for (const p of [
      ...(prev.player_powers ?? []), ...(s.player_powers ?? []),
      ...(prev.enemies ?? []).flatMap((e) => e.powers ?? []),
      ...(s.enemies ?? []).flatMap((e) => e.powers ?? []),
      // `image_big`, with `image` as the fallback: all three consumers of this
      // cache are VFX surfaces, and every one of them draws the game's
      // `_power.BigIcon` rather than the tray sprite. The FLASH needed it most
      // — `vfx_power_flash_vfx`'s authored `scale: 0.4` is applied to
      // `img.width`, and the game applies it to a 256px sprite, so feeding the
      // 64px tray icon rendered the flash at a quarter size. The fallback keeps
      // a bucket serving a pre-`power_big` art tree working exactly as before.
      // `Powers.tsx`'s tray still reads `p.image` directly and must keep doing
      // so: it draws at 26px.
    ]) powerIconsRef.current.set(p.name, p.image_big ?? p.image ?? null);
    schedulerRef.current?.play(events);
    // Anything but `full` settles in one beat — autoplay, fast-step and
    // prefers-reduced-motion keep exactly today's behaviour.
    if (mode !== "full") schedulerRef.current?.drain();
    // Outlasts the damage number's 1.2s arc (NDamageNumVfx); unmounting at 1s
    // cut it off mid-fall. Measured from the LAST event, not the first.
    const t = window.setTimeout(
      () => setFx({}),
      TOTAL_SETTLE_MS + HIT_SPACING_MS * events.length,
    );
    return () => window.clearTimeout(t);
  }, [step.idx, s, mode, onPlayCard, applyEvent, actionNode]);

  // Picking a card UP. `NMouseCardPlay.StartCardDrag` (`:336`) plays this
  // the moment the card leaves the hand under the cursor; the workbench's
  // click IS that gesture, so it fires on the click (on a phone, on the first
  // tap that selects a playable card) rather than on the play itself (which
  // has its own three-layer cue — see `cuesForCard`).
  const cardSelectCue = useCallback(() => {
    const snd = soundRef.current;
    if (snd?.map) {
      const cue = cueForUi(snd.map, "cardSelect");
      if (cue) snd.emit([cue]);
    }
  }, []);

  const clickCard = useCallback((card: { index: number; target_type?: string }) => {
    if (!onPlayCard) return;
    cardSelectCue();
    if (!needsTarget(card)) onPlayCard(card.index, undefined);
    // One legal target, so we pick it for you rather than making you aim at
    // the only thing there is.
    else if (enemies.length === 1) onPlayCard(card.index, enemies[0].index);
    else setPendingCard(card.index);
  }, [onPlayCard, enemies, cardSelectCue, setPendingCard]);

  // ---- phone: zoom-on-select, two-tap play (handSelect.ts) -----------------
  const applyTap = useCallback((r: TapResult) => {
    setSelectedCard(r.selected);
    setPendingCard(r.pending);
    if (r.play && onPlayCard) onPlayCard(r.play.card, r.play.target);
  }, [onPlayCard, setPendingCard, setSelectedCard]);

  const tapHandCard = useCallback((card: Card) => {
    const canPlay = card.can_play !== false;
    const r = tapCard({ selected: selectedCard, pending: pendingCard },
      { index: card.index, needsTarget: needsTarget(card), canPlay },
      { live: !!onPlayCard, enemies: enemies.map((e) => e.index) });
    if (onPlayCard && canPlay && selectedCard !== card.index) cardSelectCue();
    applyTap(r);
  }, [selectedCard, pendingCard, onPlayCard, enemies, applyTap, cardSelectCue]);

  /** A thumb dragged along the hand: the card under it is selected (zoomed,
   *  an attack armed) as a first tap would, and never played. */
  const browseHandCard = useCallback((card: Card) => {
    if (selectedCard !== card.index) tapHandCard(card);
  }, [selectedCard, tapHandCard]);

  const clickEnemy = useCallback((enemyIndex: number) => {
    if (phone) { applyTap(tapEnemy({ selected: selectedCard, pending: pendingCard }, enemyIndex)); return; }
    if (onPlayCard && pendingCard != null) {
      onPlayCard(pendingCard, enemyIndex);
      setPendingCard(null);
    }
  }, [onPlayCard, pendingCard, phone, selectedCard, applyTap, setPendingCard]);

  // A tap on the stage that reaches it (powers, intents, block and the sheet
  // stop propagation while they can inspect) closes the inspect sheet; on a
  // phone a tap on EMPTY stage also cancels the zoom and the armed card. The
  // hand, the zoom and — while a card is armed — the enemies are not empty
  // stage: their own handlers have already acted on the tap.
  const onStageClick = (ev: React.MouseEvent) => {
    if (inspect.item) inspect.close();
    if (!phone || (selectedCard == null && pendingCard == null)) return;
    const t = ev.target as Element;
    if (t.closest(".gs-handslot, .gs-zoomcard")) return;
    if (pendingCard != null && t.closest("[data-enemy-index]")) return;
    applyTap(tapStage());
  };

  useEffect(() => {
    if (!onPlayCard) return; // interactive (live) combat only
    function onKey(e: KeyboardEvent) {
      if (isTextInput(e.target) || isBrowserChord(e)) return;
      if (e.key === "Escape") { setPendingCard(null); setSelectedCard(null); return; }
      if (e.key === "e" || e.key === "E") { e.preventDefault(); onEndTurn?.(); return; }
      // H / G / S mirror the buttons below, which are press-first and so are
      // not gated by the hints switch. Pressing a key IS asking.
      if (e.key === "h" || e.key === "H") { e.preventDefault(); onHint?.(); return; }
      if (e.key === "g" || e.key === "G") { e.preventDefault(); onPlayRecommended?.(); return; }
      if (e.key === "a" || e.key === "A") { e.preventDefault(); onAutoplay?.(); return; }
      if (e.key === "f" || e.key === "F") { e.preventDefault(); onAutoplayToEnd?.(); return; }
      if (e.key === "s" || e.key === "S") { e.preventDefault(); onSearch?.(); return; }
      if (e.key === "p" || e.key === "P") { e.preventDefault(); onPeek?.(); return; }
      const n = Number(e.key);
      if (!Number.isInteger(n) || n < 1 || n > 9) return;
      e.preventDefault();
      if (pendingCard != null) {
        const en = enemies[n - 1];
        if (en) clickEnemy(en.index);
      } else {
        const card = s.hand[n - 1];
        if (card && card.can_play !== false) clickCard(card);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onPlayCard, onEndTurn, onHint, onPeek, onPlayRecommended, onAutoplay, onAutoplayToEnd, onSearch, pendingCard, enemies, s.hand, clickCard, clickEnemy, setPendingCard, setSelectedCard]);

  // Per-card / per-potion / end-turn win-prob overlays are UNPROMPTED advice —
  // they appear next to every option without you doing anything — so in LIVE
  // play they need either the hints switch or an explicit hint request for
  // this position (`showScores`). Replay (no `onPlayCard`) is review of a
  // recorded run, not a drill, so its recorded scores stay visible.
  const shownScores = !onPlayCard || showScores || hints.on ? scores : undefined;
  // A zero-visit row was never searched (the run-AI skips a forced move), so it
  // has no win-prob to overlay.
  // The engine's number on the plate rides only with "engine numbers" on;
  // the advice itself is the mark (advice/marks.ts).
  const { marks: adviceMarks, showNumbers } = useAdviceMarks();
  const endScore = showNumbers
    ? shownScores?.find((sc) => sc.stable_idx === -1 && sc.visits !== 0) : undefined;
  const endMark = adviceMarks.get(END_TURN_KEY) ?? null;
  // The plan's first targeted move: its target wears the reticle.
  const reticleTarget = [...adviceMarks.values()]
    .filter((m) => m.order != null && m.targetIndex != null)
    .sort((a, b) => a.order! - b.order!)[0]?.targetIndex ?? null;

  const action = step.chosen_action as { action?: string; args?: { card_index?: number; target_index?: number } } | null;
  const playedIndex = action?.action === "play_card" ? action.args?.card_index : undefined;
  const targetIndex = action?.action === "play_card" ? action.args?.target_index : undefined;

  // The arrow's every constant is measured against the game's 1080-tall design
  // viewport, so it needs to know how tall this board actually is. 0 means
  // "not measured yet" — or jsdom, where every box measures 0 — and falls back
  // to the design height rather than suppressing the arrow entirely.
  const [boardHeight, setBoardHeight] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBoardHeight(el.getBoundingClientRect().height));
    ro.observe(el);
    setBoardHeight(el.getBoundingClientRect().height);
    return () => ro.disconnect();
  }, []);

  /** Replay only: a static "this is the card that was played, and what it hit"
   *  marker on a recorded step. Live play draws no arrow after the play — the
   *  game calls `StopDrawing()` the instant a card resolves, and the hit VFX
   *  are what say who got struck. */
  const [replayArrow, setReplayArrow] = useState<Aim | null>(null);

  useLayoutEffect(() => {
    function measure() {
      const el = ref.current;
      if (!el || onPlayCard || playedIndex == null || targetIndex == null) { setReplayArrow(null); return; }
      const card = el.querySelector('[data-played="true"]');
      const enemy = el.querySelector(`[data-enemy-index="${targetIndex}"]`);
      if (!card || !enemy) { setReplayArrow(null); return; }
      const b = el.getBoundingClientRect();
      const c = card.getBoundingClientRect();
      const e = enemy.getBoundingClientRect();
      setReplayArrow({
        from: { x: c.left + c.width / 2 - b.left, y: c.top - b.top },
        tip: { x: e.left + e.width / 2 - b.left, y: e.top + e.height * 0.55 - b.top },
      });
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [step.idx, playedIndex, targetIndex, onPlayCard]);

  // ---- the aiming arrow -------------------------------------------------
  // While a card waits for a target the game hangs the arrow off it, throws it
  // at the cursor and snaps it to whatever enemy is under it.
  const [aimTo, setAimTo] = useState<Point | null>(null);
  // Not on a phone: there is no pointer to follow (spec: no targeting arrow on
  // touch); the valid targets keep the enemy highlight instead.
  useEffect(() => {
    const el = ref.current;
    if (pendingCard == null || !el || phone) { setAimTo(null); return; }
    const onMove = (ev: MouseEvent) => {
      const b = el.getBoundingClientRect();
      setAimTo({ x: ev.clientX - b.left, y: ev.clientY - b.top });
    };
    el.addEventListener("mousemove", onMove);
    return () => el.removeEventListener("mousemove", onMove);
  }, [pendingCard, phone]);

  // Where the aiming arrow currently points: the hovered enemy if there is one
  // (the game snaps), else the raw cursor. Read from the DOM at render time
  // like the replay arrow — the fan and the arena both move under it.
  let aimArrow: Aim | null = null;
  if (pendingCard != null && aimTo && ref.current && !phone) {
    const el = ref.current;
    const card = el.querySelector(`[data-card-index="${pendingCard}"]`);
    if (card) {
      const b = el.getBoundingClientRect();
      const c = card.getBoundingClientRect();
      let tip = aimTo;
      const snap = hoverEnemy != null
        ? el.querySelector(`[data-enemy-index="${hoverEnemy}"]`) : null;
      if (snap) {
        const e = snap.getBoundingClientRect();
        tip = { x: e.left + e.width / 2 - b.left, y: e.top + e.height * 0.55 - b.top };
      }
      aimArrow = {
        from: { x: c.left + c.width / 2 - b.left, y: c.top - b.top + 6 },
        tip,
      };
    }
  }

  // `vfx_monster_death` (Task 10) — the shader-driven dissolve every dying
  // enemy now dissolves through (`effects/dissolve.ts`, `SpineSprite`'s
  // `dissolve` prop), replacing the old `.gs-dying` CSS fade. `null` before
  // the manifest has loaded (or if a future extraction ever drops the
  // entry) — a dying rig just holds its last `die` frame, un-dissolved,
  // until the corpse is removed, the same additive-degradation contract
  // every other manifest-driven visual in this file already has.
  // The encounter camera (`SceneContainer.Scale` / `.Position`), which zooms the
  // room art and the fighters together — 0.75 for the Kaiser Crab, 0.87 for the
  // Decimillipede, 1 for an ordinary fight.
  const camera: Camera = cameraFor([
    ...enemies.map((e) => e.rig), s.osty?.rig ?? null,
  ]);

  const dissolveEffect: DissolveEffect | null = (() => {
    const eff = manifest?.effects["vfx_monster_death"];
    return eff && eff.kind === "dissolve" ? eff : null;
  })();

  /**
   * `dissolveEffect` with `dissolveSeconds` narrowed to ONE creature's own
   * duration (`effects/dissolve.ts`'s `dissolveSecondsFor`). The manifest's
   * 2.5 s is the game's CEILING, not a constant: the game divides the
   * creature's own bounds by the screen width, so a Byrdpip dissolves in well
   * under a second where a Fabricator takes the full 2.5.
   *
   * Narrowing the effect OBJECT rather than threading a second number is what
   * keeps this one change: `SpineSprite`'s dissolve, `Combatant`'s chrome fade,
   * `KaiserCrabGroup` and `DecimillipedeGroup` all already read
   * `effect.dissolveSeconds`, and each now reads its own creature's.
   */
  const dissolveEffectFor = (c: {
    kind?: string | null; rig?: string | null; rig_scale?: number | null; size_jitter?: number | null;
  }): DissolveEffect | null => {
    if (!dissolveEffect) return null;
    // `NCreature.cs:1106` gates the whole `NMonsterDeathVfx` on the monster's
    // own `ShouldFadeAfterDeath`, and ten classes override it away — the two
    // Kaiser Crab arms, the Decimillipede's segments, and six ordinary
    // monsters. Nothing modelled that until GH #633, so every one of them
    // dissolved here while the game leaves it alone. Absent from the table
    // means `MonsterModel`'s `true` default, so an unclassified monster keeps
    // dissolving.
    if (!shouldFadeAfterDeath(c.kind, bindings)) return null;
    const size = renderedSize({ rig: c.rig, rigScale: c.rig_scale });
    if (!size) return dissolveEffect;
    const jitter = instanceScale(c);
    return {
      ...dissolveEffect,
      dissolveSeconds: dissolveSecondsFor(
        size.w * jitter,
        size.h * jitter,
        camera.scale,
        dissolveEffect.dissolveSeconds,
        SCENE_W,
      ),
    };
  };

  // The pending corpses that are still doing something: dropped as soon as
  // `dying` carries the entry, or the creature is back on the board (a seek
  // backwards, an action superseded before its death landed). Derived, so
  // every render pass — including one React later throws away — reaches the
  // same answer from the same inputs.
  const pendingCorpses = pendingCorpsesRef.current.filter(
    (p) => !liveEnemyKeys.has(enemyKey(p)) && !dying.some((d) => enemyKey(d.enemy) === enemyKey(p)),
  );
  const displayEnemies = [
    ...enemies.map((e) => ({
      e, isDying: false as boolean, isFading: false as boolean,
      nonce: undefined as number | undefined, fadedAt: undefined as number | undefined,
    })),
    ...dying
      .filter((d) => !enemies.some((e) => enemyKey(e) === enemyKey(d.enemy)))
      .map((d) => ({
        e: d.enemy, isDying: true as boolean, isFading: d.fading,
        nonce: d.nonce as number | undefined, fadedAt: d.fadedAt as number | undefined,
      })),
    // The corpses `dying` has not caught up with yet (GH #668). Not fading and
    // carrying no nonce: this is the same not-yet-dead-looking body the
    // render-phase mount puts up, one render earlier — the scheduled `death`
    // event is still what starts the fade and the `die` clip.
    ...pendingCorpses.map((e) => ({
      e, isDying: true as boolean, isFading: false as boolean,
      nonce: undefined as number | undefined, fadedAt: undefined as number | undefined,
    })),
  ].sort((a, b) => a.e.index - b.e.index);

  // Which source arms each rig's one-shot right now, in the SAME priority
  // order the clip itself is picked in (`armOf` below, and the player's props
  // at the bottom of the render). `useOneShots` edge-triggers them — see its
  // doc comment for why the priority chain cannot be handed to `SpineSprite`
  // as-is.
  const oneShot = useOneShots({
    player: fx["player"]?.flash
      ? trig("fx", fx["player"].nonce)
      : trig("atk", attack?.nonce),
    ...Object.fromEntries(displayEnemies.map((d): [string, OneShotTrigger] => {
      const key = `e${d.e.index}`;
      const isDead = d.isFading || d.e.hp <= 0;
      const lunge = enemyActions[d.e.index];
      const hit = fx[key];
      return [key,
        isDead ? trig("die", d.nonce)
        : lunge != null ? trig("lunge", lunge)
        : hit ? trig("fx", hit.nonce)
        : attack?.target === d.e.index ? trig("atk", attack.nonce)
        : null];
    })),
  });

  // The nameplate label: a name shared by several live enemies carries its
  // " #k" — the same numbering every action label uses ("Strike → Rat #2").
  const labelOf = enemyLabeller(enemies);

  // Build a CrabArm descriptor (the same per-enemy render info a
  // <Combatant> gets, plus its shared-rig `part`) for one enemy.
  const armOf = (d: typeof displayEnemies[number]): CrabArm => {
    const { e, isDying, isFading, fadedAt } = d;
    const liveIdx = enemies.findIndex((x) => x.index === e.index);
    // A corpse mounted for THIS action's killing blow is on the board but not
    // yet dead-looking: it still plays `hurt` and still shows the blow's
    // damage number, and only turns into a `die`-clipped, fading body when
    // the scheduled `death` event fires. `isDying` (present as a corpse at
    // all) is what disables targeting; `isFading` is what makes it a corpse.
    const isDead = isFading || e.hp <= 0;
    const atkNonce = enemyActions[e.index];
    const isAttacking = !isDead && atkNonce != null;
    const dissolveFx = dissolveEffectFor(e);
    return {
      c: e, part: e.rig_part ?? "left", dataIndex: e.index,
      // Numbered over the LIVE list only — a dying corpse keeps its plain name.
      label: liveIdx >= 0 ? labelOf(e.index) : undefined,
      // Not on a phone: there is no keyboard to press it on, and the badge
      // stood every enemy 17px higher the moment an attack was armed.
      hotkey: !phone && !isDying && onPlayCard && pendingCard != null && liveIdx >= 0 && liveIdx < 9 ? liveIdx + 1 : undefined,
      onClick: !isDying && onPlayCard && pendingCard != null ? () => clickEnemy(e.index) : undefined,
      onHoverChange: !isDying && onPlayCard ? (h: boolean) => setHoverEnemy(h ? e.index : null) : undefined,
      // Test Subject's rig has no shared "idle" clip, only one
      // idle_loop{N} per revive form; force the clip matching its
      // current form so it doesn't default to whichever form's
      // clip happens to sort first (see SpineSprite idle resolution).
      animation: e.rig_form ? `idle_loop${e.rig_form}` : (enemyAnim ?? e.rig_anim ?? undefined),
      actionAnim: isDead ? "die" : isAttacking ? "attack" : "hurt",
      // ...and its ONE-SHOTS are per-form too (attack_big2, hurt3, …).
      // A bare "attack" resolves by prefix, which lands on whichever
      // numbered clip sorts first — so a three-headed Test Subject
      // lunged with its one-headed body and snapped back to the
      // three-headed idle. Name the form's clips explicitly; the
      // unsuffixed fallbacks keep this harmless for any other rig
      // that ever reports a form.
      actionAnimCandidates: e.rig_form
        ? formClips(isDead ? "die" : isAttacking ? "attack" : "hurt", e.rig_form)
        : undefined,
      actionNonce: oneShot[`e${e.index}`],
      holdLast: isDead, dying: isFading, effects: isFading ? undefined : fx[`e${e.index}`],
      // The WebGL dissolve (Task 14) — only once this corpse's `death` event
      // has actually fired (`fadedAt` set) and the manifest carries the
      // entry; `SpineSprite`'s plain draw path runs untouched otherwise.
      dissolve: isFading && fadedAt != null && dissolveFx
        ? { effect: dissolveFx, rev: manifest?.rev ?? "", startedAt: fadedAt }
        : null,
      // `DeathAnimLengthOverride`. Only the two Kaiser Crab arms set one at
      // v0.107.1, and they are exactly the pair that gets no dissolve — so
      // without this their chrome faded on the 0.9s fallback while the corpse
      // itself stayed up for 2.5s.
      deathAnimSeconds: deathFadeFor(e.kind, bindings).deathAnimSeconds ?? null,
    };
  };

  // Scene zoom: the stage is `flex: 1 1 0` in a viewport-height column, so its
  // box is whatever the chrome around it leaves — a full 1920x1052 in a
  // maximised 1080p window, less than half that on a tablet in landscape. The
  // whole composition inside is authored at ONE size (the game's scene, plus
  // the design-px furniture standing on it), so it is laid out at that size and
  // scaled to the box we got. clientWidth/Height (not getBoundingClientRect)
  // because the stage's own box must be read unscaled; only its
  // .gs-stage-scale child carries the transform.
  const stageRef = useRef<HTMLDivElement>(null);
  // `.gs-actors`, the combatants' own container — the box `VfxLayer` measures
  // and the one its canvas is a child of. See the overlay's JSX below.
  const actorsRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  // The compact framing (stageZoom.ts): 1 on a monitor, up to 1.35 on a tablet,
  // where the composition is drawn larger than its fit and cropped. The zoom
  // already carries it; `fitScene` below and the hand need the factor itself,
  // to crop the scene and size the rest line by the same amount.
  const [framing, setFraming] = useState(1);
  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    // The zoom changes the layer's size, never the stage's (the stage is a
    // flex:1 child of a fixed column), so observing the stage cannot loop.
    const measure = () => {
      setZoom(stageZoom(el.clientWidth, el.clientHeight));
      setFraming(stageFraming(el.clientWidth, el.clientHeight));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The combat SCENE, in the game's own 1920x1080 design px (see src/scene.ts).
  // Everything below — the room art, where each fighter stands, how big its rig
  // is drawn — is placed in those coordinates and mapped onto the stage here,
  // rather than composed out of flex padding and tuned pixel budgets. That is
  // the whole point: the game's numbers are knowable (they are in the decomp),
  // so the workbench can simply use them.
  //
  // Measured off `.gs-stage-scale`, i.e. the layer's UNSCALED layout size:
  // `--gs-k` scales that layer as a whole, so a rig built against it is built
  // at native resolution and then scaled with everything else. Rounded to 4px
  // so a resize drag does not re-scale every rig once per pixel.
  const sceneRef = useRef<HTMLDivElement>(null);
  const [stageBox, setStageBox] = useState({ w: 1600, h: 800 });
  useLayoutEffect(() => {
    const el = sceneRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      // offsetWidth/Height, NOT getBoundingClientRect: the rect is the box
      // AFTER `--gs-k`, and feeding a scaled box back in would compound.
      const w = Math.round(el.offsetWidth / 4) * 4;
      const h = Math.round(el.offsetHeight / 4) * 4;
      if (w < 4 || h < 4) return;
      setStageBox((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The Kaiser Crab boss is not a creature standing in the enemy column: in the
  // game its rig is mounted on the combat-room BACKGROUND
  // (`NKaiserCrabBossBackground` at %KaiserCrab) and reaches a claw in from each
  // side of the screen, with the player standing between them. Rendering it as a
  // combatant produced a small crab hunched in the right-hand column with both
  // claws side by side — nothing like the fight.
  const crabParts = displayEnemies.filter((d) => d.e.rig_group === "kaiser_crab" && d.e.rig);
  const crabRig = crabParts[0]?.e;
  // `armOf` defaults a missing `rig_part` to "left", so match that here rather
  // than requiring the field — the fallback is what actually renders.
  const crabLeft = crabParts.find((d) => (d.e.rig_part ?? "left") === "left");
  crabLeftTargetRef.current = crabLeft ? { side: "enemy", index: crabLeft.e.index } : null;
  // Decimillipede segments: three rigs, one animal, one shared root.
  const segParts = displayEnemies.filter((d) => d.e.rig_group === "decimillipede" && d.e.rig);

  useEffect(() => { cameraScaleRef.current = camera.scale; }, [camera.scale]);

  // The fighters, laid out by the game's own row algorithm.
  //
  // Multi-rig bosses count as ONE row entry (they share a slot in the game's
  // encounter scene): the Kaiser Crab is not in the row at all — it is the
  // room, drawn from the scene centre — and the Decimillipede's three segments
  // stand on one root, so the row spaces the whole animal, not each segment.
  const rowEnemies = displayEnemies.filter(
    (d) => d.e.rig_group !== "kaiser_crab" && d.e.rig_group !== "decimillipede",
  );
  const segSpan = decimillipedeSpan(segParts.map((d) => d.e.rig));
  const segWidth = (segSpan.right - segSpan.left) * SEGMENT_SCALE;
  // The ROW IS LAID OUT ONCE PER FIGHT, over the fight's whole roster — not per
  // render over whoever is still standing.
  //
  // `NCombatRoom.PositionEnemies` runs exactly once, from `CreateEnemyNodes` at
  // combat setup. Nothing calls it again: a creature that dies has its node
  // removed and the survivors keep the `Position` they were given, and the only
  // other placement path in the room, `AddCreature`, moves a creature ONLY when
  // it carries an authored `SlotName`. So in the game the row never reflows
  // mid-combat, and here it did — killing one of two enemies slid the other
  // across the floor, because the widths were recomputed from the live list.
  //
  // Kept as a per-fight memo rather than derived from the wire because the wire
  // does not carry the encounter's original roster (only who is alive now). It
  // is additive: a slot, once given out, is that creature's for the rest of the
  // fight whether or not it is still alive, and the row only ever recomputes if
  // a creature the fight had never seen appears. That second case is a
  // mid-combat SPAWN — an Ovicopter egg, a Two-Tailed Rat, a Living Fog bomb —
  // and every encounter in the game that spawns has authored slots
  // (`EncounterModel.Slots`), i.e. is already outside what this row algorithm
  // can place exactly.
  const rowKey = `${step.meta?.act ?? ""}|${step.meta?.floor ?? ""}`;
  if (rowRosterRef.current.key !== rowKey) {
    rowRosterRef.current = { key: rowKey, widths: new Map(), slots: new Map(), order: [], crowns: new Map(), draws: new Map(),
      inks: new Map(), parts: new Map(), segRigs: new Set() };
  }
  const roster = rowRosterRef.current.widths;
  // The most parts each multi-rig boss has shown this fight: a part's death
  // must not re-share the boss's split slot among the survivors.
  const partsSeen = rowRosterRef.current.parts;
  const partCount = (group: string, live: number) => {
    const n = Math.max(partsSeen.get(group) ?? 0, live);
    partsSeen.set(group, n);
    return n;
  };
  const crabPartN = partCount("kaiser_crab", crabParts.length);
  const segPartN = partCount("decimillipede", segParts.length);
  for (const d of segParts) if (d.e.rig) rowRosterRef.current.segRigs.add(d.e.rig);
  /** The span of every segment the fight has shown, for the split. */
  const segSpanFight = decimillipedeSpan([...rowRosterRef.current.segRigs]);
  const crowns = rowRosterRef.current.crowns;
  for (const d of rowEnemies) {
    const k = enemyKey(d.e);
    if (!roster.has(k)) {
      roster.set(k, boundsWidth({
        rig: d.e.rig, rigScale: d.e.rig_scale,
        boundsH: d.e.rig_bounds_h, boundsW: d.e.rig_bounds_w,
      }));
      crowns.set(k, spriteBodyH(d.e, true));
      const dw = rootSymWidth(d.e.rig, d.e.rig_scale);
      if (dw != null) rowRosterRef.current.draws.set(k, dw * instanceScale(d.e));
      const ink = inkBox(d.e.rig, d.e.rig_scale);
      if (ink) rowRosterRef.current.inks.set(k, { w: ink.w * instanceScale(d.e), cx: ink.cx * instanceScale(d.e) });
    }
  }
  // Wire order IS the game's row order (the engine keeps enemies in slot order
  // — `CombatState.SortEnemiesBySlotName`), so the order is READ off the board
  // and remembered, never re-derived from `index`. A spawn inserted mid-list
  // renumbers every enemy behind it (`index` is a position in the encounter
  // array, `rust_game.py`), so a creature already standing somewhere must be
  // recognised by something the renumbering cannot touch — see `enemyKey`.
  const rowOrder = rowRosterRef.current.order;
  const liveKeys = rowEnemies.map((d) => enemyKey(d.e));
  for (const [at, k] of liveKeys.entries()) {
    if (rowOrder.includes(k)) continue;
    // In beside its neighbours: ahead of the first creature the board already
    // knows and lists AFTER this one.
    let pos = rowOrder.length;
    for (let j = at + 1; j < liveKeys.length; j++) {
      const known = rowOrder.indexOf(liveKeys[j]);
      if (known >= 0) { pos = known; break; }
    }
    rowOrder.splice(pos, 0, k);
  }
  const rowWidths = [
    ...rowOrder.map((k) => roster.get(k) as number),
    ...(segParts.length ? [segWidth] : []),
  ];

  /** Scene y of an absolute scene y, through the encounter camera. */
  const camY = (sceneY: number) => SCENE_CY + (sceneY - SCENE_CY) * camera.scale + camera.offsetY;
  // The highest crown an intent chip hangs over, in scene rows through the
  // camera — what caps the compact framing below (`sceneFramingFor`). Taken
  // over the fight's whole roster, dead included, so it never moves mid-fight.
  // The row is the game's own here (`placeEnemies` at the scene's full width,
  // for the stagger a crowded row lifts every second creature by): where the
  // framing crops at all the stage is at least 16:9 and shows the whole 1920,
  // so that IS the row drawn below, and `aspectFit` is 1 — and where it is not,
  // no framing is cropped and the cap does nothing.
  const gameSlots = placeEnemies(rowOrder.map((k) => roster.get(k) as number), camera.scale);
  let crownY: number | null = null;
  rowOrder.forEach((k, i) => {
    const y = camY(gameSlots[i].y) - (crowns.get(k) ?? 0) * camera.scale;
    crownY = crownY == null ? y : Math.min(crownY, y);
  });
  // The Kaiser Crab's arms stand off the row, on their own markers; their
  // intents hang at each arm's IntentPos.
  if (enemies.some((e) => e.rig_group === "kaiser_crab")) {
    for (const part of ["left", "right"]) {
      const a = crabArmMount(part);
      if (!a) continue;
      const y = camY(a.y) - a.intentH * camera.scale;
      crownY = crownY == null ? y : Math.min(crownY, y);
    }
  }
  // The scene's framing: the stage's compact framing, pulled back for a fight
  // whose tallest enemy would otherwise lose its intent (and its head) off the
  // top. Only the fighters and the room take it; `zoom` — the cards, the orb,
  // the plate — keeps the full framing, as the game's own encounter camera
  // scales the scene container and not the UI over it.
  //
  // On a phone the scene is fitted into the stage's TOP part, so every
  // fighter's power rows end above the hand strip (and the portrait chrome row
  // on it) rather than under the cards' tap boxes — see `sceneFitHeight`.
  //
  // The line fitted is the LOWEST foot line on the stage, not just the
  // ground: an encounter with authored slots (see `authored` below) can stand
  // a creature up to 16 scene px under it (Exoskeletons' "second" at 756).
  // Every slot the encounter authors counts, occupied or not, so a spawn onto
  // a low marker mid-fight does not re-fit the scene under the player.
  const authored = matchEncounter(rowEnemies.map((d) => d.e), artSlugs);
  const groundLimit = phoneGroundLimit(phone, stageBox.h, zoom);
  // The zoomed card, only while it is still in the hand (the state effect above
  // clears the selection on the next commit; this covers the frame between).
  const zoomCard = phone && selectedCard != null ? s.hand.find((c) => c.index === selectedCard) ?? null : null;
  const zoomAt = useZoomPlacement(stageRef, zoomCard != null, phone, zoom);
  const fitH = groundLimit == null ? stageBox.h
    : sceneFitHeight(stageBox.w, stageBox.h, framing, crownY,
      camY(lowestFootY(authored?.slots.map((m) => m.y) ?? [])), groundLimit);
  const sceneFraming = sceneFramingFor(stageBox.w, fitH, framing, crownY);
  const fit = fitScene(stageBox.w, fitH, sceneFraming);
  /** Stage px per SCENE px, camera included — what every rig is drawn at. */
  const pxPerScene = fit.k * camera.scale;
  // The screen shake's amplitudes are authored in scene px, so it converts
  // through this like everything else drawn on the stage; `fireScreenShake` is
  // memoised with no deps, hence the ref rather than a closure over the value.
  useEffect(() => { pxPerSceneRef.current = pxPerScene; }, [pxPerScene]);
  /** Stage x of a scene x measured from the scene's centre line. */
  const stageX = (sceneDx: number) => fit.x(sceneDx * camera.scale);
  /** Stage y of an absolute scene y, through the camera. */
  const stageY = (sceneY: number) => fit.y(camY(sceneY));
  // On a stage that shows less than the scene's 1920 px (a phone or a tablet
  // in portrait) the row is laid out for what the stage CAN show, so the
  // player and the enemies close up instead of standing at opposite edges —
  // see `rowHalfWidths`. On a desktop stage both halves are the scene's own.
  const rowHalf = rowHalfWidths(fit.visibleHalfW, camera.scale, rowWidths);
  const rowSlots = placeEnemies(rowWidths, camera.scale, rowHalf.enemies);

  // AUTHORED SLOTS BEAT THE ROW. Twenty-four encounters ship their own
  // `scenes/encounters/<id>.tscn` full of `Marker2D`s, and for those
  // `NCombatRoom` runs `PositionCreaturesWithSlots` INSTEAD of the row
  // algorithm — so the row above is at best an approximation of them, and for
  // the ones that stack in depth (the Living Fog's five bombs in an arc, the
  // Fabricator's four corner bots) not even that.
  //
  // Handed out the same way the widths are, once and for keeps: the opening
  // roster takes the encounter's `start` list in wire order, and anything that
  // appears later — a spawn — takes a free slot, from whichever end of the list
  // that encounter's summoner scans from (`nextSpawnSlot`).
  //
  // "Free" is judged against the creatures ON THE BOARD, which is what C#'s
  // `Enemies.All(c => c.SlotName != s)` asks: a slot whose occupant has died
  // and left is free for the next spawn. A corpse still dissolving is still on
  // the board and still holds its slot, so nothing lands on top of one.
  //
  // WHICH newcomer gets WHICH of the slots just claimed is a separate question
  // from the order they were claimed in, and the two disagree: an Ovicopter
  // lays into egg5, then egg4, then egg3, while the wire lists its board in
  // SLOT order (`SortEnemiesBySlotName`) — egg3 first. So the claimed slots are
  // sorted before they are handed out, or the three hatchlings come out mirrored.
  const held = rowRosterRef.current.slots;
  if (authored) {
    const newcomers = rowEnemies.filter((d) => !held.has(enemyKey(d.e)));
    if (held.size === 0) {
      // The opening roster: `start` is already in the wire's order.
      newcomers.forEach((d, at) => {
        const name = authored.start[at];
        if (name) held.set(enemyKey(d.e), name);
      });
    } else if (newcomers.length > 0) {
      const taken = new Set(
        rowEnemies.map((o) => held.get(enemyKey(o.e))).filter((n) => n != null),
      );
      const claimed: string[] = [];
      for (let i = 0; i < newcomers.length; i++) {
        const name = nextSpawnSlot(authored, taken);
        if (!name) break;
        taken.add(name);
        claimed.push(name);
      }
      const rank = new Map(authored.slots.map((s2, i) => [s2.name, i]));
      claimed.sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0));
      newcomers.forEach((d, i) => {
        const name = claimed[i];
        if (name) held.set(enemyKey(d.e), name);
      });
    }
  }

  /** The slot this enemy was given when it joined the fight. */
  const slotOf = (e: Enemy): Slot | undefined => {
    const k = enemyKey(e);
    const name = authored && held.get(k);
    const marker = name ? slotNamed(authored, name) : undefined;
    if (marker) return { x: marker.x, y: marker.y };
    const at = rowOrder.indexOf(k);
    return at < 0 ? undefined : rowSlots[at];
  };
  // The Decimillipede's root is not its span's centre (the head reaches far to
  // one side), so convert the slot's Bounds centre back to the shared root.
  const segSlot: Slot | null = segParts.length
    ? (() => {
        const slot = rowSlots[rowSlots.length - 1];
        const centre = (segSpan.left + segSpan.right) / 2 * SEGMENT_SCALE;
        return { x: slot.x - centre, y: slot.y };
      })()
    : null;

  const playerWidth = boundsWidth({
    rig: p.rig, rigScale: p.rig_scale, boundsH: p.rig_bounds_h, boundsW: p.rig_bounds_w,
  });
  const playerSlot = placePlayer(playerWidth, camera.scale, camera.centerPlayer, rowHalf.player);
  const ostyAlive = !!s.osty?.alive;
  const ostySlot = ostyAlive
    ? placeOsty(playerSlot, playerWidth, s.osty?.max_hp ?? 0)
    : null;
  // `PositionLocalPlayerOsty` steps the player 150 scene px left to make room.
  const playerX = playerSlot.x + (ostyAlive ? -150 : 0);

  // `AdjustCreatureScaleForAspectRatio`: a row that would stand past the right
  // edge shrinks to fit rather than hanging off the screen. Normally a no-op —
  // the workbench stage is usually WIDER than 16:9, which shows more room, not
  // less.
  const rowFit = aspectFit(
    [...rowSlots, playerSlot], [...rowWidths, playerWidth], camera.scale, fit.visibleHalfW);
  /** What the FIGHTERS are drawn at — the scene factor with that fit applied.
   *  The room art keeps `pxPerScene`: the game scales the ally/enemy containers
   *  here, never the background. */
  const actorScale = pxPerScene * rowFit;

  // ---- Phone portrait: the split (portraitSplit.ts) ----
  // Every portrait fight: the enemies across the top of the stage, the player
  // figure in a band right above the controls row, its status block beside it.
  // Solved ONCE PER FIGHT from the fight's roster (the additive per-fight maps
  // above, so a death keeps its slot empty and only a never-seen enemy — a
  // summon — re-solves) and the stage size: nothing per-turn reaches it.
  // Authored encounter slots are not used here; a Kaiser Crab or a
  // Decimillipede is ONE solver creature. Landscape, a touch tablet and a
  // desktop never get here (`split` is null).
  const splitSmall = phone === "portrait" && isSmallPhoneScreen();
  const pCrown = spriteBodyH(p, false);
  // The band draws the figure and Osty at a fixed height ON THE GLASS, which
  // is the rig's measured head (`Combatant`'s `onCrown`), not its authored
  // box: the box reaches ~9% (the Ironclad) to ~50% (Osty) above the drawn
  // head. Until a rig has measured, its authored crown stands in. Keyed by
  // rig, so a new character re-measures.
  const [inkCrowns, setInkCrowns] = useState<Record<string, number>>({});
  const reportCrown = useCallback((rig: string | null | undefined) => (crown: number) => {
    if (!rig || !(crown > 0)) return;
    setInkCrowns((prev) => (Math.abs((prev[rig] ?? 0) - crown) < 0.5 ? prev : { ...prev, [rig]: crown }));
  }, []);
  const drawnCrown = (rig: string | null | undefined, authored: number) => (rig && inkCrowns[rig]) || authored;
  /** The fight's crab rig (the room in the row layout) as one creature: its
   *  whole rig box, standing on the arms' markers. Scene px. */
  const crabSplit = (() => {
    if (!crabRig?.rig) return null;
    const b = rigBox(crabRig.rig);
    if (!b) return null;
    const feetY = Math.max(...["left", "right"].map((part) => crabArmMount(part)?.y ?? CRAB_ROOT_Y));
    /** How far the arms' feet stand below the rig's root. */
    const drop = feetY - CRAB_ROOT_Y;
    return {
      drop,
      /** The rig box's centre, right of the root. */
      cx: (b.x + b.w / 2) * CRAB_SETUP_SCALE,
      creature: { crownH: (b.y + b.h) * CRAB_SETUP_SCALE + drop, footW: b.w * CRAB_SETUP_SCALE } as SplitCreature,
    };
  })();
  // A Necrobinder fight, Osty alive or dead (a dead Osty is `{ alive: false }`
  // on the wire), so Osty's death and return never re-solve the split.
  const hasOsty = !!s.osty;
  const splitRosterSize = rowOrder.length + (segParts.length ? 1 : 0) + (crabSplit ? 1 : 0);
  // Hysteresis on the stage height (`heldSplitHeight`): within one fight,
  // roster, width and zoom, a height change under `SPLIT.margin` real px
  // keeps the last solve, shifted onto the stage below, instead of
  // rescaling every enemy. Read and written during render, like the roster
  // maps; it is idempotent, so a repeated render lands on the same value.
  const splitHoldKey = `${phone}|${rowKey}|${splitRosterSize}|${stageBox.w}|${zoom}|${splitSmall}|${hasOsty}`;
  const splitHoldRef = useRef<{ key: string; h: number } | null>(null);
  const prevHold = splitHoldRef.current;
  const splitSolveH = heldSplitHeight(prevHold?.key === splitHoldKey ? prevHold.h : null, stageBox.h, zoom);
  if (prevHold?.key !== splitHoldKey || prevHold.h !== splitSolveH) {
    splitHoldRef.current = { key: splitHoldKey, h: splitSolveH };
  }
  /** Scene px the Decimillipede's legs reach below its root. */
  const segLegs = decimillipedeDrop([...rowRosterRef.current.segRigs]) * SEGMENT_SCALE;
  const solved: { layout: SplitLayout; segDrop: number } | null = useMemo(() => {
    if (phone !== "portrait") return null;
    const { crowns: cr, widths, draws, inks } = rowRosterRef.current;
    const solve = (drop: number) => splitLayout({
      stageW: stageBox.w, stageH: splitSolveH, k: zoom, small: splitSmall,
      enemies: [
        ...rowOrder.map((k) => ({ crownH: cr.get(k) ?? 0, footW: widths.get(k) ?? 0, drawW: draws.get(k),
          inkW: inks.get(k)?.w })),
        ...(segParts.length
          ? [{ crownH: Math.max(...segParts.map((d) => spriteBodyH(d.e, true))) + drop,
            footW: (segSpanFight.right - segSpanFight.left) * SEGMENT_SCALE, noColumn: true }] : []),
        ...(crabSplit ? [crabSplit.creature] : []),
      ],
      refCrown: pCrown, rowScale: actorScale, osty: hasOsty,
    });
    const plain = solve(0);
    if (!segParts.length || !(segLegs > 0)) return { layout: plain, segDrop: 0 };
    // The Decimillipede stands on its legs, its HP bars and powers under
    // them, where that costs it no size (a stage with height to spare: the
    // animal is as wide as the stage). On a stage where height binds, the
    // bars hang from its root, across the legs, and it keeps its size.
    const stood = solve(segLegs);
    return stood.chrome === plain.chrome && stood.scale >= plain.scale * 0.98
      ? { layout: stood, segDrop: segLegs } : { layout: plain, segDrop: 0 };
    // Pinned per fight: the roster maps are additive refs read here, and the
    // row scale and the player's crown follow from the fight and the stage.
    // Only a new fight, a never-seen enemy, the stage size or the zoom (a
    // rotation) re-solves it. The key leaves out values this solve reads that
    // are assumed fixed for a fight: `actorScale` (the row's), `pCrown` (the
    // player's authored crown), the segments' span and `crabSplit` (the boss rigs).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phone, rowKey, splitRosterSize, stageBox.w, splitSolveH, zoom, splitSmall, hasOsty]);
  const splitDy = stageBox.h - splitSolveH;
  const solvedSplit = solved?.layout ?? null;
  const segDrop = solved?.segDrop ?? 0;
  const split = useMemo(() => solvedSplit && shiftSplit(solvedSplit, splitDy), [solvedSplit, splitDy]);
  /** An enemy's split place, by the slot the fight gave it (`rowOrder`). */
  const splitPlaceOf = (e: Enemy): SplitPlace | undefined => {
    const at = rowOrder.indexOf(enemyKey(e));
    return split && at >= 0 ? split.enemies[at] : undefined;
  };
  /** An enemy's solver slot, design px (`SplitLayout.slots`). */
  const splitSlotOf = (e: Enemy): number | undefined => {
    const at = rowOrder.indexOf(enemyKey(e));
    return split && at >= 0 ? split.slots[at] : undefined;
  };
  /** The slot less its power column: what the body, its HP bar and its
   *  name may span, design px. The HP bar is capped to it, so neighbouring
   *  bars keep the solver's gap. */
  const splitBodySlot = (slot: number) => slot - (split ? CHROME[split.chrome].side / zoom : 0);
  /** A multi-rig boss's targetable parts (the crab's arms, the
   *  Decimillipede's segments) share its ONE solver slot evenly and take the
   *  fight's chrome from it, like any enemy: `at` is the boss's solver index,
   *  `n` its parts. */
  const splitPartChrome = (at: number, n: number): PartChrome | undefined => {
    if (!split || !(n > 0) || split.slots[at] == null) return undefined;
    const slot = split.slots[at] / n;
    return { chrome: split.chrome, cells: splitPowerCells(slot * zoom, zoom),
      bodySlot: splitBodySlot(slot), slot, slotReal: slot * zoom };
  };
  /** Feet at `place.foot`, centred on `place.x`, drawn at `scale`. */
  const splitStyle = (place: SplitPlace, spriteH: number, scale: number, z = 1): React.CSSProperties => ({
    position: "absolute",
    left: place.x,
    top: place.foot - spriteH * scale,
    transform: "translateX(-50%)",
    zIndex: z,
  });
  // The band, real px. The figure is drawn `figureReal` tall; its width is
  // the drawn rig's (symmetric about the root) or the sprite box's (0.7x its
  // height), whichever is wider, so nothing stands on it.
  const splitBand = split ? (() => {
    const figS = split.figureReal / zoom / drawnCrown(p.rig, pCrown);
    // Sized off the AUTHORED crown, which is the taller: the band's left edge
    // never moves when the rig measures.
    const figW = Math.max(
      Math.max(playerWidth, rootSymWidth(p.rig, p.rig_scale) ?? 0) * (split.figureReal / pCrown), 0.7 * split.figureReal);
    const figure = { x: (SPLIT_FIG_GAP + figW / 2) / zoom, foot: split.figureFoot };
    let osty: { place: SplitPlace; scale: number; w: number } | null = null;
    if (ostyAlive && s.osty) {
      const oCrown = spriteBodyH({ rig: s.osty.rig, rig_scale: s.osty.rig_scale, rig_bounds_h: s.osty.rig_bounds_h }, false);
      const oS = SPLIT.osty / zoom / drawnCrown(s.osty.rig, oCrown);
      const oFoot = boundsWidth({ rig: s.osty.rig, rigScale: s.osty.rig_scale, boundsH: s.osty.rig_bounds_h });
      const w = Math.max(SPLIT_OSTY_MIN_W, 0.7 * SPLIT.osty,
        // At the DRAWN scale: Osty's authored box reaches ~50% over its drawn
        // head, so sized off the box it would stand on the status block once
        // its rig measured. (The band then settles once, as the rig loads.)
        Math.max(oFoot, rootSymWidth(s.osty.rig, s.osty.rig_scale) ?? 0) * oS * zoom);
      osty = { scale: oS, w,
        place: { x: (SPLIT_FIG_GAP + figW + SPLIT_FIG_GAP + w / 2) / zoom,
          foot: split.figureFoot - SPLIT_OSTY_CHROME / zoom } };
    }
    const rest = SPLIT_FIG_GAP + figW + SPLIT_FIG_GAP + (osty ? osty.w + SPLIT_FIG_GAP : 0) + SPLIT_SHIELD_HALF;
    // While a card is zoomed, the status block (shield, HP bar, orbs, powers)
    // steps right of the zoom, so the player keeps their block and powers in
    // sight while choosing a card. The zoom's right edge comes from the same
    // numbers `zoomPlacement` uses, not from a measurement; only the block
    // moves (figure, Osty, enemies and controls stay put).
    const zoomW = splitZoomWidth(splitSmall);
    const zoomRight = portraitZoomLeft(stageBox.w * zoom, zoomW) + zoomW;
    const left = zoomCard ? Math.max(rest, zoomRight + SPLIT_FIG_GAP + SPLIT_SHIELD_HALF) : rest;
    return {
      figure, figS, osty, left,
      width: Math.max(0, stageBox.w * zoom - (zoomCard ? ZOOM_GUTTER : SPLIT.gutter) - left),
      top: split.figureFoot - split.figureReal / zoom,
    };
  })() : null;

  /** Place one fighter: its rig root on the scene's ground line at `slot.x`,
   *  with the combatant column hung above so the sprite's feet land there. */
  const slotStyle = (slot: Slot, spriteH: number, z = 1): React.CSSProperties => ({
    position: "absolute",
    left: stageX(slot.x * rowFit),
    top: stageY(slot.y) - spriteH * actorScale,
    transform: "translateX(-50%)",
    zIndex: z,
  });

  // A stage too short for the enemies' chrome (a small phone with its
  // browser bars): the cramped split gave up the room under the feet, so the
  // powers and names go rather than hang over the player band.
  const splitBare = split != null && split.cramped
    && split.belowReal + 0.5 < splitChromeBelow(split.chrome, splitSmall);

  /** The character's layered faceted orb (ui/combat/energy_counters/):
   *  base + swirl + core + rim + facets composite the game's ruby. */
  const energyOrb = (
    <div className="gs-energy" title={`energy ${s.energy}/${s.max_energy}`}>
      {(() => {
        const ch = orbCharacter(s.energy_image);
        return [1, 2, 3, 4, 5].map((n) => (
          <img key={n} className="gs-orb-layer" alt="" aria-hidden="true" draggable={false}
            src={assetUrl(`ui/combat/energy_counters/${ch}/${ch}_orb_layer_${n}.png`)} />
        ));
      })()}
      <span className="gs-gem-val">{s.energy}/{s.max_energy}</span>
    </div>
  );

  return (
    <div ref={ref}
      className={`gs-board${phone ? ` gs-phone gs-phone-${phone}` : ""}${split ? " gs-split" : ""}${splitBare ? " gs-split-bare" : ""}`}
      data-armed={pendingCard != null ? "1" : undefined}
      data-split={split ? `${split.chrome} s=${(split.scale * zoom).toFixed(3)} back=${split.backCount}`
        + (split.overReal > 0 ? ` over=${split.overReal.toFixed(1)}` : "") : undefined}
      data-awaits-target={phone && pendingCard != null ? "1" : undefined} data-scene-ready={roomReady ? "1" : "0"}
      // `clip`, not `hidden`: see .gs-arena-stage in game-skin.css — a hidden
      // box is still a scroll container, and focusing a card scrolled it.
      style={{ position: "relative", height: "100%", display: "flex", flexDirection: "column",
        overflow: "clip" }}>
      {/* No typed status strip here: the shared GameHud band above the screen
          carries hp/gold/relics/potions; in-scene furniture carries the rest
          (turn on the End Turn plate, energy on the gem, block on the shield). */}
      {/* One continuous scene surface: the backdrop covers the arena AND the
          hand strip (the game's hand overlaps the arena floor), so cards never
          render on page-black outside the scene. The corner HUD (energy orb,
          piles, End Turn) anchors to the STAGE bottom — the strip the hand
          shares — like the game's corner furniture. */}
      {/* `data-room-ready` gates everything the ROOM paints — its two layer
          stacks and the weather that hangs in them. Until the whole stack has
          decoded it is held at opacity 0 and the stage's own gradient shows
          through, so a fight never opens on a room assembling itself layer by
          layer; when it lands, the room fades in whole. See roomPreload.ts and
          the rules on .gs-arena-stage[data-room-ready]. */}
      <div className="gs-arena-stage" ref={stageRef} data-room-ready={roomReady ? "1" : "0"}
        // A tap on empty stage closes the inspect sheet. Powers, intents,
        // block, relics and the sheet stop propagation, so they never reach it.
        onClick={phone || inspect.item ? onStageClick : undefined}
        style={{ "--gs-k": zoom, ...atmosphereVars(roomScene) } as React.CSSProperties}>
      <StageZoomContext.Provider value={zoom}>
      {/* The zoom layer. Everything in the scene is authored in px against a
          ~800px stage, so on anything shorter (a laptop at 768, a tablet in
          landscape at 640) the actors oversail the frame and get clipped away
          — see stageZoom.ts. This lays the scene out at its authored size and
          scales the whole composition down to the box we have. */}
      <div className="gs-stage-scale" ref={sceneRef}>
        {/* Painted room parallax.

            WHICH room comes from the ENCOUNTER, not the act: thirteen
            encounters declare their own background in the game (see
            roomScenes.ts), which is why the Insatiable stands on sand and
            Vantom in an oil cavern rather than everyone getting one of four
            act rooms.

            WHERE it goes is the game's own placement: every layer is authored
            on one 1920x900 canvas, and the combat-room scene draws that canvas
            at 1.44x, CENTRED on the 1920x1080 scene — so the frame shows the
            middle ~70% of a 2765x1296 painting, not the whole canvas fitted to
            the box. (Fitting the layer stack to refs/game-combat-1.png bottoms
            out at exactly that scale and offset.) The canvas element below IS
            that painting; each layer then sits where its own scene puts it on
            it — see roomLayerFit. */}
        {(() => {
          // `roomScene` and `room` are resolved at the top of the component:
          // the stack has to be known before render to gate the room on its
          // art (see `roomReady`).
          const { far, near } = room;
          // The canvas is the room's own painting, placed where the game puts
          // it — 1.44x, centred on the scene — and scaled up beyond that only
          // when a stage far off the game's own aspect would otherwise see past
          // its edge (see roomCanvasScale).
          const centreY = stageY(SCENE_CY);
          // The size the ROOM authors its canvas at, not one global constant:
          // the Hive/Glory family paints 1.50078x where the Overgrowth one
          // paints 1.44x, and drawing both at 1.44 left the former 4.2% small.
          const canvasScale = roomCanvasScale(
            stageBox.w, stageBox.h, centreY, pxPerScene, sceneScale(roomScene),
          );
          const canvas: React.CSSProperties = {
            width: ROOM_CANVAS_W * canvasScale * pxPerScene,
            height: ROOM_CANVAS_H * canvasScale * pxPerScene,
            // `%BgContainer` hangs 23 scene px right of the centre line the
            // fighters stand on — see ROOM_BG_OFFSET_X.
            left: stageX(ROOM_BG_OFFSET_X), top: centreY,
            transform: "translate(-50%, -50%)",
          };
          return (
            <>
              <RoomLayers className="gs-arena-bg" testId="arena-bg" scene={roomScene}
                layers={far} style={canvas} url={assetUrl} />
              {/* The room's OWN weather, where it has any — the Insatiable's
                  sand storm, the Waterfall Giant's spray. Colour and density
                  come out of the scene. Rendered ONLY for a room that has some:
                  the layer screen-blends, and `mix-blend-mode` opens a stacking
                  context that changes how everything beneath it composites even
                  at zero alpha. Left mounted it shifted all fourteen clear rooms
                  by ~0.6 levels — invisible, but not nothing, and there is no
                  reason for a clear room to carry the element at all. */}
              {sceneAtmosphere(roomScene) && (
                <div className="gs-arena-dust" aria-hidden="true" data-testid="arena-dust" />
              )}
              {/* Firefly motes. Pure CSS dots (no DOM churn, no rig): the ref
                  has warm sparks hanging in the canopy and none of ours did.
                  The drift is opt-in per prefers-reduced-motion; the dots
                  themselves are static in that mode, never hidden. */}
              <div className="gs-motes" aria-hidden="true" data-testid="arena-motes" />
              {/* The `_fg_` layer and anything the room draws after it: the
                  frontmost thing in the ROOM, and still behind every
                  combatant — see .gs-arena-fg. */}
              <RoomLayers className="gs-arena-fg" testId="arena-fg"
                layers={near} style={canvas} url={assetUrl} />
            </>
          );
        })()}
        {/* The character's layered faceted orb (ui/combat/energy_counters/):
            base + swirl + core + rim + facets composite the game's ruby. */}
        {/* Portrait split, a card zoomed: the centred card covers this spot,
            so the orb is drawn in the player band instead (`PlayerBand`). */}
        {!(splitBand && zoomCard) && energyOrb}
        <PilePlaque label="draw" kind="draw" count={s.draw_pile_count} onClick={() => setPile("draw")} />
        <PilePlaque label="discard" kind="discard" count={s.discard_pile_count} onClick={() => setPile("discard")} />
        {/* The game has no persistent exhaust pile in the HUD — the affordance
            appears (small, tucked left of the discard pile) only once
            something has actually been exhausted. */}
        {s.exhaust_pile_count > 0 && (
          <PilePlaque label="exhaust" kind="exhaust" count={s.exhaust_pile_count} onClick={() => setPile("exhaust")} />
        )}
        {/* End Turn plate lives in the scene (right edge, above the discard
            pile) — inert/disabled in replay, clickable in live play. The real
            button art (ui/combat_ui/) with its extracted glow layer on hover;
            turn number and the win-prob hint overlay the plate. */}
        <button type="button" className={`gs-endturn${endMark ? ` gs-mark-${endMark.tier}` : ""}`}
          onClick={onEndTurn} disabled={!onEndTurn}
          title={onEndTurn ? "End Turn (E)" : "End Turn (replay)"}>
          <img className="gs-endturn-art" src={assetUrl("ui/combat_ui/end_turn_button.png")}
            alt="" aria-hidden="true" draggable={false} />
          <img className="gs-endturn-glow" src={assetUrl("ui/combat_ui/end_turn_button_glow.png")}
            alt="" aria-hidden="true" draggable={false} />
          <span className="gs-endturn-face">
            <span>End Turn {s.round}</span>
            {endScore && <span className="gs-endturn-score">{endScore.win_prob.toFixed(2)}</span>}
            {endMark?.order != null && (
              <span className={`gs-badge gs-step-tag gs-step-${endMark.tier}`} aria-label={`step ${endMark.order}`}>
                {endMark.order}
              </span>
            )}
          </span>
        </button>
      {/* The fighters, in SCENE coordinates. Every slot below is the game's
          own: `NCombatRoom` puts the ground line 200px under the scene centre
          and derives each x from the creature's footprint (see src/scene.ts).
          The row used to be a flex box with tuned percentage padding, which is
          why the actors drifted with the window and why nothing here was ever
          the size the game draws it.

          The layer is inert and sits between the room's foreground and the
          hand; each combatant re-enables its own pointer. */}
      <div className="gs-actors" aria-hidden={false} ref={actorsRef}>
        {/* The Kaiser Crab boss AS THE ROOM (see crabParts above), mounted
            where the game mounts it: the rig hangs off the combat-room
            background at the setup scene's own position and 0.55 scale
            (crabMount.ts, read off the PCK), through the encounter camera like
            the room art — so its face, its claws and the player between them
            land where the game draws them on any stage. Its two arms are
            ORDINARY creatures on the encounter's markers, with the game's own
            HP bar width and intent height. */}
        {crabRig?.rig && crabParts.length > 0 && (() => {
          // Portrait split: the whole rig is the last solver creature, its
          // arms' feet on that creature's foot line.
          const crabAt = split && crabSplit ? split.enemies[split.enemies.length - 1] : null;
          const k = crabAt ? split!.scale : pxPerScene;
          const pxPerUnit = CRAB_SETUP_SCALE * k;
          const rootLeft = crabAt ? crabAt.x - crabSplit!.cx * k : stageX(CRAB_ROOT_DX);
          const rootTop = crabAt ? crabAt.foot - crabSplit!.drop * k : stageY(CRAB_ROOT_Y);
          const armAt: Record<string, ArmAt | undefined> = {};
          for (const part of ["left", "right"]) {
            const a = crabArmMount(part);
            if (!a) continue;
            armAt[part] = {
              x: crabAt ? (a.dx + a.boundsCx - CRAB_ROOT_DX) * k : stageX(a.dx + a.boundsCx) - rootLeft,
              y: crabAt ? (a.y - CRAB_ROOT_Y) * k : stageY(a.y) - rootTop,
              hpW: a.hpW * k,
              intentH: a.intentH * k,
            };
          }
          return (
            <div style={{ position: "absolute", left: rootLeft, top: rootTop, zIndex: 0 }}>
              <KaiserCrabGroup rig={crabRig.rig} arms={crabParts.map(armOf)}
                pxPerUnit={pxPerUnit} armAt={armAt}
                clampX={{ min: -rootLeft, max: stageBox.w - rootLeft }}
                locStrings={manifest?.locStrings}
                split={crabAt ? splitPartChrome(split!.enemies.length - 1, crabPartN) : undefined}
                onRigEvent={onCrabRigEvent} />
            </div>
          );
        })()}
        {/* Player. The Defect's orb belt goes down INTO the Combatant: the game
            hangs it on an arc around the creature, which needs the rig's crown
            and feet. */}
        {/* Portrait split: the figure alone (`noChrome`), drawn at its fixed
            height in the band; its HP bar, block, orbs and powers are the
            band's (below). */}
        <div className="gs-side gs-side-player"
          style={splitBand ? splitStyle(splitBand.figure, pCrown, splitBand.figS, 2)
            : slotStyle({ x: playerX, y: playerSlot.y }, spriteBodyH(p, false), 2)}>
          <Combatant c={{ ...p, powers: s.player_powers }} isEnemy={false} anchor="player"
            pxPerScene={splitBand ? splitBand.figS : actorScale}
            orbs={splitBand ? undefined : s.orbs} orbSlots={splitBand ? undefined : s.orb_slots}
            noChrome={!!splitBand} onCrown={splitBand ? reportCrown(p.rig) : undefined}
            actionAnim={fx["player"]?.flash ? "hurt" : "attack"}
            actionNonce={oneShot["player"]}
            effects={fx["player"]} locStrings={manifest?.locStrings} />
        </div>
        {ostySlot && s.osty && (() => {
          const osty = {
            name: s.osty.name ?? "Osty", hp: s.osty.hp ?? 0, max_hp: s.osty.max_hp ?? 1,
            block: s.osty.block ?? 0, image: s.osty.image, rig: s.osty.rig,
            rig_scale: s.osty.rig_scale, rig_bounds_h: s.osty.rig_bounds_h,
          };
          return (
            <div className="gs-side gs-side-osty"
              style={splitBand?.osty ? splitStyle(splitBand.osty.place, spriteBodyH(osty, false), splitBand.osty.scale, 2)
                : slotStyle(ostySlot, spriteBodyH(osty, false), 2)}>
              <Combatant c={osty} isEnemy={false} anchor="osty"
                pxPerScene={splitBand?.osty ? splitBand.osty.scale : actorScale}
                hpWidth={splitBand?.osty ? splitBand.osty.w / zoom : undefined}
                onCrown={splitBand?.osty ? reportCrown(s.osty?.rig) : undefined}
                locStrings={manifest?.locStrings} />
            </div>
          );
        })()}
        {/* Portrait split: the player's status block beside the figure, laid
            out in REAL px (a 1/k counter-scale inside the zoom layer), centred
            on the figure and running to the right gutter. Inside .gs-actors so
            VfxLayer finds the orbs anchor on it. */}
        {splitBand && (
          <div className="gs-playerband"
            style={{ position: "absolute", left: splitBand.left / zoom, top: splitBand.top,
              width: splitBand.width, height: split!.figureReal,
              transform: `scale(${1 / zoom})`, transformOrigin: "0 0", zIndex: 2,
              "--gs-k": 1 } as React.CSSProperties}>
            <StageZoomContext.Provider value={1}>
              <PlayerBand player={p} powers={s.player_powers ?? []} widthPx={splitBand.width}
                orbs={s.orbs} orbSlots={s.orb_slots} effects={fx["player"]}
                energy={zoomCard ? energyOrb : undefined}
                locStrings={manifest?.locStrings} />
            </StageZoomContext.Provider>
          </div>
        )}
        {/* Enemies, in row order. A multi-rig boss occupies ONE row slot. */}
        {rowEnemies.map((d) => {
          const { e, isDying } = d;
          const a = armOf(d);
          // Its OWN slot, the one the fight handed it — not its position in the
          // list of who is still alive. See `slotOf`.
          const slot = slotOf(e);
          const place = splitPlaceOf(e);
          if (split ? !place : !slot) return null;
          const sSlot = splitSlotOf(e);
          return (
            <div key={`${enemyKey(e)}-${isDying ? "d" : "l"}`} className="gs-side"
              data-split-slot={sSlot != null ? (sSlot * zoom).toFixed(1) : undefined}
              style={place && split ? splitStyle(place, spriteBodyH(e, true), split.scale, 1)
                : slotStyle(slot!, spriteBodyH(e, true), 1)}>
              <Combatant c={e} isEnemy dataIndex={e.index} pxPerScene={split ? split.scale : actorScale} label={a.label}
                powerChrome={split?.chrome}
                powerCells={sSlot != null ? splitPowerCells(sSlot * zoom, zoom) : undefined}
                hpWidth={sSlot != null && split ? Math.min(hpBarWidth(e.max_hp), splitBodySlot(sSlot)) : undefined}
                // The ink is centred in the slot; the box and its chrome are too.
                bodyDx={split ? -(rowRosterRef.current.inks.get(enemyKey(e))?.cx ?? 0) * split.scale : 0}
                reticle={reticleTarget === e.index}
                hotkey={a.hotkey} onClick={a.onClick} onHoverChange={a.onHoverChange}
                animation={a.animation} actionAnim={a.actionAnim}
                actionAnimCandidates={a.actionAnimCandidates} holdLast={a.holdLast}
                dying={a.dying} dissolve={a.dissolve} deathAnimSeconds={a.deathAnimSeconds}
                actionNonce={a.actionNonce} effects={a.effects}
                locStrings={manifest?.locStrings} />
            </div>
          );
        })}
        {segSlot && segParts.length > 0 && (
          <div style={split
            ? { position: "absolute", zIndex: 1,
              left: split.enemies[rowOrder.length].x - (segSpanFight.left + segSpanFight.right) / 2 * SEGMENT_SCALE * split.scale,
              top: split.enemies[rowOrder.length].foot - segDrop * split.scale }
            : { position: "absolute", left: stageX(segSlot.x * rowFit), top: stageY(segSlot.y), zIndex: 1 }}>
            <DecimillipedeGroup segments={segParts.map(armOf)} pxPerScene={split ? split.scale : actorScale}
              split={split ? splitPartChrome(rowOrder.length, segPartN) : undefined}
              drop={split ? segDrop * split.scale : 0}
              locStrings={manifest?.locStrings} />
          </div>
        )}
        {/* The WebGL impact-VFX overlay, drawn above every combatant.
            Main restructured the arena while this branch was in flight: the
            combatants moved out of `.gs-arena` (now a bare flex spacer) into
            this absolutely-positioned scene, so the overlay follows them.

            It is the last child of `.gs-actors` — NOT of `.gs-arena-stage`,
            which is two levels up — and `.gs-actors` is what it measures and
            anchors against. That is deliberate: measuring the stage instead
            would only work by coincidence (`.gs-actors` is `inset: 0` on a box
            coincident with the stage), and would silently put every effect in
            the wrong place the day anyone gives `.gs-actors` an inset or a
            transform of its own. No test can catch that — jsdom has no WebGL.

            The scene's actors sit at z-index 1-2 (`slotStyle`), so the canvas
            takes 3 — above them, below the HUD furniture at 6.

            `sceneScale` is SCREEN px per scene px, which is `actorScale`
            (layout px per scene px) times `zoom`. The overlay measures
            `.gs-actors` with `getBoundingClientRect`, which is taken AFTER
            `.gs-stage-scale`'s `transform: scale(--gs-k)`, so its canvas works
            in screen px — while `actorScale` is derived from that layer's
            UNSCALED `offsetWidth/Height`. Dropping the `zoom` factor leaves
            the effect a constant size as the window shrinks (1.43x oversize at
            1366x740, 1.92x at 1000x900), which is the whole bug. The game
            parents a VFX into the creature's own container, so it scales with
            the scene rather than staying a constant screen size. */}
        <VfxLayer arenaRef={actorsRef} manifest={manifest} onSink={onSink}
          sceneScale={actorScale * zoom} />
      </div>
      {/* Flex spacer: the scene above is absolutely placed over the whole
          stage, so this is all that keeps the hand strip at the bottom of the
          column (and the room's ground line above it). */}
      <div className="gs-arena" />
      {moveGrade && (
        <div key={moveGrade.nonce} className={`gs-move-grade gs-move-grade--${moveGrade.grade}`}
          role="status" data-testid="move-grade">
          {GRADE_TEXT[moveGrade.grade]}
        </div>
      )}
      <Hand cards={s.hand} playedIndex={playedIndex} onCardClick={onPlayCard ? clickCard : undefined} scores={shownScores} previewDamageFor={previewDamageFor} framing={sceneFraming}
        selectedIndex={phone ? selectedCard : null} onCardTap={phone ? tapHandCard : undefined}
        onCardBrowse={phone ? browseHandCard : undefined} />
      {zoomCard && zoomAt && (
        <ZoomCard card={zoomCard} placement={zoomAt} onTap={() => tapHandCard(zoomCard)}
          onClose={() => applyTap(tapStage())} live={!!onPlayCard}
          awaitsTarget={pendingCard === zoomCard.index && enemies.length > 1}
          wouldPlay={tapCard({ selected: zoomCard.index, pending: pendingCard },
            { index: zoomCard.index, needsTarget: needsTarget(zoomCard), canPlay: zoomCard.can_play !== false },
            { live: !!onPlayCard, enemies: enemies.map((e) => e.index) }).play != null}
          winProb={showNumbers ? shownScores?.find((x) => x.stable_idx === zoomCard.index)?.win_prob : undefined}
          previewDamage={previewDamageFor(zoomCard)} />
      )}
      {/* The potion belt is the HUD band's (GameHud → PotionTray), where the
          game keeps it; the scene no longer carries a second copy. */}
      </div>
      </StageZoomContext.Provider>
      {/* Outside the zoom layer: real px, overlaying the stage (no layout height). */}
      <InspectSheet />
      </div>

      {onEndTurn && (
        <div className="mono gs-toolrow">
          {/* End Turn itself moved into the arena as the in-scene hex plate;
              this row keeps the workbench tooling (press-first advice etc.).
              Sizes live in .gs-toolrow (game-skin.css) rather than inline, so
              the compact rules for a short viewport can shrink the row — it
              was 143px of a 640px-tall tablet screen, taken off the board. */}
          {onPlayRecommended && (
            <button onClick={onPlayRecommended} disabled={hintBusy}>
              play best (G)
            </button>
          )}
          {onAutoplay && (
            <button onClick={onAutoplay} disabled={hintBusy}>
              autoplay (A)
            </button>
          )}
          {onAutoplayToEnd && (
            <button onClick={onAutoplayToEnd} disabled={hintBusy} title="play the MCTS-recommended move every step until the fight ends (win or lose)">
              play to end (F)
            </button>
          )}
          {onSearch && (
            <button onClick={onSearch} title="inspect the MCTS search tree">
              search (S)
            </button>
          )}
          {hintBusy && <span style={{ color: "var(--muted)" }}>thinking…</span>}
          {pendingCard != null && <span style={{ color: "var(--accent-strong)" }}>select a target…</span>}
          {hintConfig && onHintConfig && (
            <span className="gs-toolrow-adv">
              <MctsControls config={hintConfig} onChange={onHintConfig} disabled={hintBusy} />
            </span>
          )}
        </div>
      )}

      {autoCard && (
        // Keyed by nonce so a second auto-play in the same action (Stampede 2,
        // a Havoc chain) restarts the animation instead of being swallowed.
        <div key={autoCard.nonce} className="gs-autoplay-card" data-testid="autoplay-card"
          aria-live="polite" aria-label={`auto-played ${autoCard.card.name}`}>
          <CardFrame card={autoCard.card} width={150} />
          <span className="gs-autoplay-tag">auto-played</span>
        </div>
      )}

      {(aimArrow || replayArrow) && (
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 20 }}>
          {/* Live: the card waits for a target, the arrow follows the cursor
              and snaps to the enemy under it, tinting red the moment it has
              something to hit — the game's whole targeting affordance. */}
          {aimArrow && (
            <TargetingArrow
              from={aimArrow.from} tip={aimArrow.tip} boardHeight={boardHeight || DESIGN_HEIGHT}
              tint={hoverEnemy != null ? "enemy" : "none"} hovering={hoverEnemy != null}
              testId="aim-arrow" />
          )}
          {/* Replay: which card was played, and at what. Not live juice — a
              recorded step is a decision point, and it has already resolved. */}
          {!aimArrow && replayArrow && (
            <TargetingArrow
              from={replayArrow.from} tip={replayArrow.tip} boardHeight={boardHeight || DESIGN_HEIGHT}
              tint="enemy" testId="play-arrow" />
          )}
        </svg>
      )}

      {pile && (() => {
        const piles: Record<PileKind, { title: string; cards: Card[]; note?: string }> = {
          draw: { title: "Draw pile", cards: s.draw_pile ?? [], note: "order hidden — sorted, not draw order" },
          discard: { title: "Discard pile", cards: s.discard_pile ?? [] },
          exhaust: { title: "Exhaust pile", cards: s.exhaust_pile ?? [] },
        };
        const v = piles[pile];
        return <PileViewer title={v.title} cards={v.cards} note={v.note} onClose={() => setPile(null)} />;
      })()}
    </div>
  );
}
