import { describe, it, expect, vi } from "vitest";
import { render, act } from "@testing-library/react";
import Board from "../Board";
import { SoundFixture } from "../../sound/SoundContext";
import { EffectsProvider } from "../../effects/EffectsContext";
import { HIT_SPACING_MS } from "../../effects/timing";
import type { Clock } from "../../effects/scheduler";
import type { Card, CombatLogRow, CombatState, Step } from "../../api";
import type { SfxMap } from "../../sound/sfxMap";
import type { SoundCue } from "../../sound/types";

const newEmit = () => vi.fn<(cues: SoundCue[]) => void>();
type Emit = ReturnType<typeof newEmit>;

/**
 * Combat sound driven by the engine log, as the board plays it: per hit in
 * `full` mode (each event's sounds when the scheduler reaches it), and as one
 * deduplicated beat in the modes that settle instantly. `boardSound.test.tsx`
 * covers the no-log (intent-guess) path.
 */

/** A hand-driven clock: nothing the scheduler queued fires until `advance`. */
function fakeClock() {
  let t = 0;
  let cbs: (() => void)[] = [];
  const clock: Clock = {
    now: () => t,
    frame: (cb) => {
      cbs.push(cb);
      return () => { cbs = cbs.filter((c) => c !== cb); };
    },
  };
  return {
    clock,
    advance(ms: number) {
      t += ms;
      const due = cbs;
      cbs = [];
      due.forEach((c) => c());
    },
  };
}

const SLIME = "event:/sfx/e/slime_impact";
const SLAM = "event:/sfx/e/slime_slam";
const GROW = "event:/sfx/e/slime_grow";
const SWOOSH = "event:/sfx/cards/play";

const MAP: SfxMap = {
  version: "test",
  generic: { blockHit: "event:/sfx/block_hit", cardPlay: SWOOSH },
  powers: {},
  cards: {},
  relics: {},
  monsters: {
    "Twig Slime": {
      moves: [
        { name: "SLAM_MOVE", intent: "Attack", hits: 1, event: SLAM },
        { name: "GROW_MOVE", intent: "Buff", event: GROW },
      ],
      defaultAttack: SLAM,
      impact: SLIME,
    },
  },
};

/** A live step whose enemy carries the uid the log rows join on. */
function loggedStep(round: number, enemyHp: number, intent = "Attack", hand: Card[] = []): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: round, step_no: 0 },
    state: {
      decision: "combat_play", round, energy: 3, max_energy: 3,
      draw_pile_count: 5, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
      player_powers: [],
      enemies: [{ index: 0, uid: 11, name: "Twig Slime", hp: enemyHp, max_hp: 50, block: 0,
        image: null, intents: [{ type: intent, damage: 7, hits: 1 }] }],
      hand,
    } as unknown as CombatState,
  };
}

const row = (over: Partial<CombatLogRow>): CombatLogRow => ({
  kind: "damage", round: 1, side: "player", text: "", verb: null,
  actor: null, actor_side: null, target: null, target_side: null,
  amount: null, noun: null, source: null, ...over,
});
const hit = (hpAfter: number) => row({
  kind: "damage", verb: "dealt", actor: "The Ironclad", actor_side: "player",
  target: "Twig Slime", target_side: "enemy", target_uid: 11, amount: 6, blocked: 0, hp_after: hpAfter,
});
const move = (id: string) => row({
  kind: "monster_move", round: 2, side: "enemy", verb: "used", actor: "Twig Slime", actor_side: "enemy",
  actor_uid: 11, source_kind: "move", source_move_id: id, source_monster: "TwigSlime",
});
/** One timeline node, as `App` hands it to `Board`. */
const node = (id: number, rows: CombatLogRow[]) => ({ id, parent_id: id - 1, combat_log: rows });

/** Every cue event `emit` received, flattened across calls, in order. */
const heard = (emit: Emit) =>
  emit.mock.calls.flatMap((c) => c[0].map((cue) => cue.event));

/** Render `from`, then step to `to` with `rows` on its node. */
function transition(emit: Emit, from: Step, to: Step, rows: CombatLogRow[], clock?: Clock) {
  const board = (s: Step, n: ReturnType<typeof node>) => (
    <SoundFixture on map={MAP} emit={emit}>
      <Board step={s} actionNode={n} onPlayCard={() => {}} onEndTurn={() => {}} clock={clock} />
    </SoundFixture>
  );
  const { rerender } = render(board(from, node(1, [])));
  act(() => { rerender(board(to, node(2, rows))); });
}

describe("board combat sound — per hit, from the engine log", () => {
  // The game plays `SfxCmd.PlayDamage` once per HIT. The board used to voice
  // a whole transition as one chord, so a 3-hit attack showed three numbers
  // and three shakes but made a single impact sound.
  // Mutation caught: emitting the event cues at transition time (or through
  // `cuesForCombat`) instead of from `applyEvent` as each hit lands.
  it("sounds the struck monster's impact once per hit, in step with the hits", () => {
    const emit = newEmit();
    const { clock, advance } = fakeClock();
    transition(emit, loggedStep(1, 50), loggedStep(1, 32), [hit(44), hit(38), hit(32)], clock);
    const impacts = () => heard(emit).filter((e) => e === SLIME).length;
    act(() => { advance(0); });
    expect(impacts()).toBe(1);
    act(() => { advance(HIT_SPACING_MS); });
    expect(impacts()).toBe(2);
    act(() => { advance(HIT_SPACING_MS); });
    expect(impacts()).toBe(3);
  });

  // 78 of the 259 mapped moves are not attacks, and the intent-based guess
  // could only ever voice attacks. Mutation caught: dropping the exact
  // move-id join in `resolveMonsterMove`.
  it("plays the exact move the log names, a buff included", () => {
    const emit = newEmit();
    const { clock, advance } = fakeClock();
    transition(emit, loggedStep(1, 50, "Buff"), loggedStep(2, 50), [move("GROW")], clock);
    act(() => { advance(1000); });
    expect(heard(emit)).toContain(GROW);
    expect(heard(emit)).not.toContain(SLAM);
  });

  // Autoplay settles in one beat (`reduced`), which voices the transition
  // through `cuesForCombat` — the log must win there too, not the intent
  // guess. The intent deliberately says Attack while the log says GROW.
  // Mutation caught: passing the intent-guess `attackers` to `cuesForCombat`
  // even when the stream carries the log's own `attack` events.
  it("lets the log name the move in the one-beat modes as well", () => {
    const emit = newEmit();
    const board = (s: Step, n: ReturnType<typeof node>) => (
      <EffectsProvider autoplaying>
        <SoundFixture on map={MAP} emit={emit}>
          <Board step={s} actionNode={n} onPlayCard={() => {}} onEndTurn={() => {}} />
        </SoundFixture>
      </EffectsProvider>
    );
    const { rerender } = render(board(loggedStep(1, 50, "Attack"), node(1, [])));
    act(() => { rerender(board(loggedStep(2, 50), node(2, [move("GROW")]))); });
    expect(heard(emit)).toContain(GROW);
    expect(heard(emit)).not.toContain(SLAM);
  });

  // GH #1155 made an engine auto-play (Havoc, Stampede, …) visible; it made
  // no sound, because card sounds were emitted only by the click path.
  it("voices a card the engine auto-played", () => {
    const emit = newEmit();
    const { clock, advance } = fakeClock();
    const strike: Card = { index: 0, id: "STRIKE_IRONCLAD", name: "Strike", cost: 1, type: "Attack", description: "", image: null };
    transition(
      emit, loggedStep(1, 50, "Attack", [strike]), loggedStep(1, 50),
      [row({ kind: "card_auto_played", card_id: "STRIKE_IRONCLAD", noun: "Strike" })], clock,
    );
    act(() => { advance(1000); });
    expect(heard(emit)).toContain(SWOOSH);
  });
});
