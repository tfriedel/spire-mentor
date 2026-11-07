import { render, act } from "@testing-library/react";
import { useEffect } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Board from "../Board";
import { EffectsProvider } from "../../effects/EffectsContext";
import type { Clock } from "../../effects/scheduler";
import type { EffectSink, PlayedEffect } from "../../effects/sink";
import type { CombatLogRow, CombatState, Step } from "../../api";
import { EVENT_ADVANCE_MS, HIT_SPACING_MS } from "../../effects/timing";
import * as sfxMapModule from "../../sound/sfxMap";
import type { SfxMap } from "../../sound/sfxMap";
import * as manifestModule from "../../effects/manifest";
import type { DissolveEffect, VfxManifest } from "../../effects/manifest";
import { deathBurstRadiusFor, dissolveSecondsFor } from "../../effects/dissolve";
import { RIG_BOUNDS } from "../../rigBounds";
import { SCENE_W } from "../../scene";
import * as bindingsModule from "../../effects/bindings";
import { DEFAULT_DYING_FADE_SECONDS } from "../combatantMetrics";

/** Everything the overlay was asked to play, recorded at the moment it was
 *  asked — see the `VfxLayer` mock below. */
const { sinkPlays } = vi.hoisted(() => ({
  sinkPlays: [] as {
    effect: string; index: number; anchored: boolean; iconUrl?: string | null;
    dissolve?: { emissionRadius: number; lifetime: number } | null;
  }[],
}));

// The real overlay is a WebGL canvas and a silent no-op in jsdom, so it can
// never report what it played. This stand-in publishes the same `EffectSink`
// and records each call — crucially including whether the target was actually
// ON THE BOARD at the instant it was asked to spawn, which is the one thing
// the real sink needs (`anchorRect` returns null otherwise and drops the
// effect entirely).
vi.mock("../VfxLayer", () => {
  function VfxLayerMock({ arenaRef, onSink }: {
    arenaRef: { current: HTMLElement | null };
    onSink: (s: EffectSink) => void;
  }) {
    useEffect(() => {
      onSink({
        play(p: PlayedEffect) {
          const index = p.target.side === "enemy" ? p.target.index : -1;
          const sel = index >= 0
            ? `[data-enemy-index="${index}"]` : `[data-combatant="${p.target.side}"]`;
          sinkPlays.push({
            effect: p.effect, index, anchored: !!arenaRef.current?.querySelector(sel),
            iconUrl: p.iconUrl, dissolve: p.dissolve,
          });
        },
      });
    }, [arenaRef, onSink]);
    return null;
  }
  return { default: VfxLayerMock };
});

beforeEach(() => { sinkPlays.length = 0; });

// A stand-in for the real rig renderer (WebGL/Spine, unavailable in jsdom):
// surfaces the one-shot action clip it was asked to play as a DOM attribute,
// so a test can observe "the enemy's lunge fired" without a GL context.
// Enemies with no `rig` field never reach this (Combatant falls back to a
// plain `<img>`), so it is inert for every OTHER fixture in this file.
vi.mock("../SpineSprite", () => ({
  default: (props: { actionAnim?: string; actionNonce?: number }) => (
    <div data-testid="spine-mock" data-action-anim={props.actionAnim} data-action-nonce={props.actionNonce} />
  ),
}));

// Build a live combat step at a given round, player hp, and enemy intent.
function step(round: number, playerHp: number, enemyIntent: string, enemyHp = 50): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: round, step_no: 0 },
    state: {
      decision: "combat_play", round, energy: 3, max_energy: 3,
      draw_pile_count: 5, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: playerHp, max_hp: 80, block: 0 },
      enemies: [{ index: 0, name: "Twig Slime", hp: enemyHp, max_hp: 50, block: 0,
        image: null, intents: [{ type: enemyIntent, damage: 7 }] }],
      hand: [],
    },
  };
}

/** A hand-driven clock: nothing advances until the test says so. */
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

function hitRow(amount: number, hpAfter: number): CombatLogRow {
  return {
    kind: "damage", round: 1, side: "player", text: "", verb: "dealt",
    actor: "The Ironclad", actor_side: "player", actor_uid: null,
    target: "Twig Slime", target_side: "enemy", target_uid: 11,
    amount, blocked: 0, hp_after: hpAfter, noun: null, source: "Sword Boomerang",
  };
}

/**
 * Same shape as `step()`, plus a uid on the enemy — and deliberately NO
 * `state.combat_log`.
 *
 * That is how production looks: `CombatState.combat_log` is written only for
 * replayed / real-game-bridge steps, never for a workbench live step, so the
 * rows have to arrive the way `App` supplies them — on the timeline node (see
 * `actionNodeAt` below). These tests used to hand-build `state.combat_log`,
 * which is why they agreed with a board that read the wrong field and rendered
 * one number for a three-hit attack.
 */
function loggedStep(enemyHp: number): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: {
      decision: "combat_play", round: 1, energy: 3, max_energy: 3,
      draw_pile_count: 5, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
      enemies: [{ index: 0, uid: 11, name: "Twig Slime", hp: enemyHp, max_hp: 50,
        block: 0, image: null, intents: [{ type: "Attack", damage: 7 }] }],
      hand: [],
    },
  } as unknown as Step;
}

/** `loggedStep`'s enemy killed: the engine drops a dead creature from the
 *  board entirely (`rust_game.py`'s `if c.get("is_dead"): continue`), so it is
 *  absent for the WHOLE action, not just after the death lands. */
function killedStep(): Step {
  const st = loggedStep(1);
  return { ...st, state: { ...st.state, enemies: [] } };
}

/** One timeline node, exactly as `App` hands it to `Board`: the id advances
 *  once per action and `parent_id` chains it to the node before. */
function actionNodeAt(id: number, combatLog: CombatLogRow[]) {
  return { id, parent_id: id - 1, combat_log: combatLog };
}

/** The engine log's own signal for "this enemy just struck" — no `damage`
 *  row, just the actor's move. `actionStream` turns this into an `attack`
 *  event on its own, independent of any hp change. */
function monsterMoveRow(): CombatLogRow {
  return {
    kind: "monster_move", round: 2, side: "enemy", text: "", verb: "used",
    actor: "Twig Slime", actor_side: "enemy", actor_uid: 11,
    target: null, target_side: null, amount: null, blocked: null, hp_after: null,
    noun: "Attack", source: null,
  };
}

/** Like `loggedStep`, but the enemy carries a `powers` entry with a real
 *  icon URL — needed to drive `powerFlash` end to end: `Board.tsx` resolves
 *  `PlayedEffect.iconUrl` from this SAME `powers` list (`powerIconsRef`),
 *  keyed by display name, the same source `powerAdd`/`powerRemove`'s panels
 *  already read (Task 12). */
function poweredStep(round: number, enemyHp: number, power = "Thorns", icon: string | null = "/img/thorns.png"): Step {
  const st = loggedStep(enemyHp);
  return {
    ...st,
    state: {
      ...st.state,
      round,
      enemies: (st.state.enemies ?? []).map((e) => ({
        ...e,
        powers: [{ name: power, amount: 3, image: icon }],
      })),
    },
  };
}

/** The engine log's own signal for a power TRIGGERING (as opposed to being
 *  applied/removed) — `NCreature.OnPowerFlashed`, Task 1/11's
 *  `EventLogEntry::PowerFlashed` -> `power_flash` combat-log row. */
function powerFlashRow(power: string): CombatLogRow {
  return {
    kind: "power_flash", round: 2, side: "enemy", text: "", verb: "triggered",
    actor: null, actor_side: null, actor_uid: null,
    target: "Twig Slime", target_side: "enemy", target_uid: 11,
    amount: null, blocked: null, hp_after: null, noun: power, source: null,
  };
}

/** Like `loggedStep`, but at a given round and with a `rig` on the enemy —
 *  needed so `Combatant` actually mounts `SpineSprite` (mocked above)
 *  instead of falling back to a plain `<img>`. */
function rigStep(round: number, enemyHp: number): Step {
  const st = loggedStep(enemyHp);
  return {
    ...st,
    state: {
      ...st.state,
      round,
      enemies: (st.state.enemies ?? []).map((e) => ({ ...e, rig: "monsters/twig_slime" })),
    },
  };
}

/** `rigStep`'s rigged enemy, plus a `powers` entry and a NON-attacking
 *  intent — the shape needed to watch what a `powerFlash` does to the RIG
 *  (the cases above only watch the sink). The intent matters: with an
 *  `Attack` intent `attackingEnemies` (Board.tsx) guesses a lunge, and
 *  `actionNonce` would then come from `enemyActions` rather than from `fx`. */
function poweredRigStep(round: number, power = "Thorns"): Step {
  const st = rigStep(round, 50);
  return {
    ...st,
    state: {
      ...st.state,
      enemies: (st.state.enemies ?? []).map((e) => ({
        ...e,
        intents: [{ type: "StatusCard" }],
        powers: [{ name: power, amount: 3, image: "/img/thorns.png" }],
      })),
    },
  };
}

/** `rigStep`'s rigged enemy at an arbitrary hp, with the same non-attacking
 *  intent `poweredRigStep` needs and for the same reason. A pair of these with
 *  hp RISING is a `heal` on that enemy and nothing else. */
function healedRigStep(round: number, enemyHp: number): Step {
  const st = rigStep(round, enemyHp);
  return {
    ...st,
    state: {
      ...st.state,
      enemies: (st.state.enemies ?? []).map((e) => ({ ...e, intents: [{ type: "StatusCard" }] })),
    },
  };
}

describe("Board combat-feel effects (live mode)", () => {
  it("shows a damage number + SCREEN shake when an enemy attacks and deals hp damage", () => {
    const onPlayCard = () => {};
    // First render establishes the previous snapshot (no effects yet).
    const { rerender, container } = render(
      <Board step={step(1, 80, "Attack")} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    // Enemy turn: round advances, player took 7 hp (an enemy that telegraphed Attack).
    act(() => {
      rerender(<Board step={step(2, 73, "StatusCard")} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    const dmg = container.querySelector(".gs-dmg");
    expect(dmg?.textContent).toBe("7");
    // The SCREEN shakes, at the tier `CreatureCmd.cs:351-365` picks for 7
    // unblocked damage. jsdom has no `Element.animate`, so the attribute is
    // what is observable here; the curve itself is pinned in
    // `effects/__tests__/screenShake.test.ts`.
    expect(container.querySelector(".gs-board")).toHaveAttribute("data-screen-shake", "medium");
    // The CREATURE does not wobble. `NCreature.AnimShake` is guarded on
    // `!Visuals.IsPlayingHurtAnimation()`, so damage is the one case the game
    // excludes — the hit reads as the rig's hurt clip plus the board punch
    // (GH #632). This assertion is the whole point of that change and would
    // pass vacuously if `data-anim-shake` were ever renamed, so the debuff test
    // below asserts the positive case against the same attribute.
    expect(container.querySelector('[data-combatant="player"][data-anim-shake]')).toBeNull();
  });

  it("flinches the player even when the hit is fully blocked (no hp loss)", () => {
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={step(1, 80, "Attack")} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      // round advances, hp unchanged (blocked) — still an attack landed.
      rerender(<Board step={step(2, 80, "StatusCard")} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    // The player's sprite plays the hit "punch" (the "got hit" signal), no damage number.
    expect(container.querySelector('[data-combatant="player"] .gs-hit-punch')).not.toBeNull();
    expect(container.querySelector(".gs-dmg")).toBeNull();
    // Task 13: the SAME `fullyBlocked` event also plays the block spark on
    // the CREATURE (`CreatureCmd.cs`'s `WasFullyBlocked` arm spawns
    // `NBlockSparkVfx` alongside the sound and the "Blocked" text) —
    // `anchored: true` confirms it resolved against the player, not a badge.
    expect(sinkPlays).toContainEqual({ effect: "vfx_block_spark_vfx", index: -1, anchored: true });
    // ...and the screen shakes, from `CreatureCmd.cs:406` — a FIXED Weak, not a
    // damage tier. This is the site the damage-scaled one at :349-368 cannot
    // reach (`:338`'s `if (damage <= 0) continue`), so "fully blocked" is not
    // "no shake" as this file's comment used to say.
    expect(container.querySelector(".gs-board")).toHaveAttribute("data-screen-shake", "weak");
  });

  it("does NOT animate in replay mode (no onPlayCard)", () => {
    const { rerender, container } = render(<Board step={step(1, 80, "Attack")} />);
    act(() => {
      rerender(<Board step={step(2, 73, "StatusCard")} />);
    });
    expect(container.querySelector(".gs-dmg")).toBeNull();
  });
});

describe("Board — per-hit playback from the combat log", () => {
  it("shows one damage number per hit, spaced, for a multi-hit attack", () => {
    // The whole point of the log-driven stream: combatDiff can only see the
    // net 18 and would render a single "18".
    //
    // This is also the C1 regression test: the rows arrive ONLY on
    // `actionNode`, exactly as `App` supplies them, and the state carries no
    // `combat_log` at all. A board that reads `step.state.combat_log` sees
    // nothing here, degrades to `combatDiff`, and renders one "18".
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={loggedStep(50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    act(() => {
      rerender(
        <Board
          step={loggedStep(32)}
          actionNode={actionNodeAt(2, [hitRow(6, 44), hitRow(6, 38), hitRow(6, 32)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock}
        />,
      );
    });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(1);
    act(() => { advance(HIT_SPACING_MS); });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(2);
    act(() => { advance(HIT_SPACING_MS); });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(3);
  });

  it("waits for the node's rows rather than replaying the previous action's", () => {
    // `App` lands `setLive(s)` and `refreshTimeline()` in two different
    // renders, so the board is genuinely rendered once with action N's state
    // and action N-1's node. Playing then would put the WRONG hits on the
    // board and consume `prev`, so the real rows would arrive with nothing
    // left to diff against.
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={loggedStep(50)} actionNode={actionNodeAt(1, [hitRow(4, 50)])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    // Render 1: new state, STALE node (still id 1).
    act(() => {
      rerender(
        <Board step={loggedStep(32)} actionNode={actionNodeAt(1, [hitRow(4, 50)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(0);
    // Render 2: the timeline catches up. Same state, this action's rows.
    act(() => {
      rerender(
        <Board step={loggedStep(32)}
          actionNode={actionNodeAt(2, [hitRow(6, 44), hitRow(6, 38), hitRow(6, 32)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    act(() => { advance(HIT_SPACING_MS * 2); });
    const texts = [...container.querySelectorAll(".gs-dmg")].map((n) => n.textContent);
    expect(texts).toEqual(["6", "6", "6"]);
  });

  it("shows each hit's own amount, not the summed total", () => {
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={loggedStep(50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    act(() => {
      rerender(
        <Board step={loggedStep(38)} actionNode={actionNodeAt(2, [hitRow(5, 45), hitRow(7, 38)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    act(() => { advance(HIT_SPACING_MS); });
    const texts = [...container.querySelectorAll(".gs-dmg")].map((n) => n.textContent);
    expect(texts).toEqual(["5", "7"]);
  });

  it("plays the enemy's lunge from a monster_move log row, and does not ALSO fire the intent-sniffing fallback", () => {
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={rigStep(1, 50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        // Round advances and the enemy still telegraphs "Attack" — exactly
        // the shape `attackingEnemies` (the intent-sniffing fallback) would
        // ALSO fire on. The log must win. The hit is fully blocked (hp
        // unchanged) and the log carries ONLY the `monster_move` row, no
        // `damage` row — so if `streamHasAttack` mis-gates and the fallback
        // fires too, it synthesizes a `fullyBlocked` flinch for the player
        // that the correct, log-driven path does NOT (the log said nothing
        // happened to the player at all).
        <Board step={rigStep(2, 50)} actionNode={actionNodeAt(2, [monsterMoveRow()])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    // The stream applied the log's own `attack` event — the enemy actually
    // lunged, not sat idle.
    const rig = container.querySelector('[data-testid="spine-mock"]');
    expect(rig?.getAttribute("data-action-anim")).toBe("attack");
    // The fallback did NOT also fire: no synthesized player flinch.
    expect(container.querySelector('[data-combatant="player"] .gs-hit-punch')).toBeNull();
  });
});

describe("Board — an auto-played card is shown (GH #1155)", () => {
  it("flashes the card Stampede played at end of turn, before its hit lands", () => {
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const bash = { index: 0, id: "BASH", name: "Bash", cost: 2, type: "Attack",
      description: "Deal 8 damage.", upgraded: false, image: null };
    const before = loggedStep(50);
    before.state = { ...before.state, hand: [bash] } as typeof before.state;
    const { rerender, container, queryByTestId } = render(
      <Board step={before} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    expect(queryByTestId("autoplay-card")).toBeNull();
    const auto: CombatLogRow = {
      kind: "card_auto_played", round: 1, side: "player", text: "The Ironclad auto-played Bash.",
      verb: "auto-played", actor: "The Ironclad", actor_side: "player", actor_uid: null,
      target: null, target_side: null, target_uid: null, amount: null, noun: "Bash", source: null,
      auto_played: true, card_id: "BASH", card_upgraded: false,
    };
    act(() => {
      rerender(
        <Board step={loggedStep(42)}
          actionNode={actionNodeAt(2, [auto, { ...hitRow(8, 42), auto_played: true }])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    // The card is up first; the hit it causes waits its beat.
    expect(queryByTestId("autoplay-card")).toHaveAttribute("aria-label", "auto-played Bash");
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(0);
    act(() => { advance(EVENT_ADVANCE_MS.cardPlay); });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(1);
  });
});

describe("Board — the killing blow", () => {
  it("plays the impact VFX on the dying enemy, which is off the board by then", () => {
    // A killing blow's target is absent from `next.enemies` for the whole
    // action. The `dying` overlay that puts it back used to mount only when
    // the paced `death` event fired — one `hurt` beat LATER — so at the hit
    // itself there was no `[data-enemy-index]` node, `anchorRect` returned
    // null, and no slash played on any kill.
    const { clock } = fakeClock();
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={loggedStep(6)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    act(() => {
      rerender(
        // `hp_after: 0` is the log's only death signal, so this one row is
        // both the hit and the kill.
        <Board step={killedStep()} actionNode={actionNodeAt(2, [hitRow(6, 0)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    expect(sinkPlays).toContainEqual({ effect: "vfx_attack_slash", index: 0, anchored: true });
  });

  it("does not start the death fade until the death event lands", () => {
    // The corpse is mounted early so the hit has something to anchor to, but
    // it must still be a live-looking creature at that moment — the fade and
    // the rig's `die` clip belong to the scheduled `death`, one beat later.
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={loggedStep(6)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    act(() => {
      rerender(
        <Board step={killedStep()} actionNode={actionNodeAt(2, [hitRow(6, 0)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    expect(container.querySelector('[data-enemy-index="0"]')).not.toBeNull();
    expect(container.querySelector(".gs-dying")).toBeNull();
    // ...and it still shows the blow that killed it.
    expect(container.querySelector(".gs-dmg")?.textContent).toBe("6");
    act(() => { advance(HIT_SPACING_MS); });
    expect(container.querySelector(".gs-dying")).not.toBeNull();
  });

  it("does not mourn the last fight's enemies when the next fight starts", () => {
    // The next floor swaps the WHOLE roster in one render while Board stays
    // mounted, so every creature of the old fight is "gone" — and the corpse
    // mount stood each of them up beside the new encounter, at the old
    // fight's HP, until the stuck-corpse sweeper removed it a second later.
    // That is the user's "why do I see two Axebots?" (2026-09-12): one live,
    // one ghost with its own rolled max HP. Mutation caught: dropping the
    // `combatChanged` guard on the corpse mount in Board.tsx.
    const { clock } = fakeClock();
    const onPlayCard = () => {};
    const floor = (step: Step, n: number): Step =>
      ({ ...step, meta: { ...step.meta, floor: n, act: 1 } }) as Step;
    const { rerender, container } = render(
      <Board step={floor(loggedStep(6), 43)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    // A different creature (its own uid and max HP) on the next floor.
    const next = floor(loggedStep(74), 44);
    const nextFight = {
      ...next,
      state: {
        ...next.state,
        enemies: [{ index: 0, uid: 12, name: "Axebot", hp: 74, max_hp: 74,
          block: 0, image: null, intents: [{ type: "Attack", damage: 14 }] }],
      },
    } as unknown as Step;
    act(() => {
      rerender(
        <Board step={nextFight} actionNode={actionNodeAt(2, [])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    expect(container.querySelector(".gs-dying")).toBeNull();
    expect(container.querySelectorAll("[data-enemy-index]")).toHaveLength(1);
  });
});

/** A `vfx_monster_death` entry carrying nothing but the fields the duration
 *  path reads. The dissolve's own art (the two baked noise gradients, the
 *  shader) never loads under jsdom — `SpineSprite` owns that, and jsdom has no
 *  WebGL — so the chrome fade is what is observable here, and it reads the
 *  same `dissolveSeconds`. */
const DISSOLVE: DissolveEffect = {
  kind: "dissolve", shader: "dissolve", blendMode: "blend_premul_alpha",
  gradient1: "vfx/g1.png", gradient2: "vfx/g2.png", modulateAlpha: 0.466667,
  thresholdFrom: 1, thresholdTo: 0, dissolveEase: 1, dissolveTrans: 1,
  dissolveSeconds: 2.5, lifetime: 2.5, emitters: [], emissionRadiusPerCreatureExtent: 0.5,
};

/** `loggedStep`'s enemy wearing a real rig at a real authored scale — what the
 *  per-monster dissolve duration needs to measure at all. It goes on the
 *  PREVIOUS step: the engine drops a dead creature from the board, so the
 *  corpse Board draws is `dyingSnapshotsRef`'s copy of the enemy as it was. */
function riggedStep(hp: number, rig: string, rigScale: number): Step {
  const st = loggedStep(hp);
  return {
    ...st,
    state: {
      ...st.state,
      enemies: (st.state.enemies ?? []).map((e) => ({ ...e, rig, rig_scale: rigScale })),
    },
  };
}

describe("Board — the dissolve's duration is per-monster (NMonsterDeathVfx.cs:235)", () => {
  async function fadeSecondsAfterDeath(rig: string, rigScale: number): Promise<number> {
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={riggedStep(6, rig, rigScale)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    // Board fetches the manifest in an effect; the dissolve entry has to be in
    // hand before the death, or every corpse takes the manifest-less fallback.
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board step={killedStep()} actionNode={actionNodeAt(2, [hitRow(6, 0)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    act(() => { advance(HIT_SPACING_MS); });
    const dying = container.querySelector(".gs-dying") as HTMLElement | null;
    expect(dying).not.toBeNull();
    return parseFloat(dying!.style.getPropertyValue("--gs-dying-ms"));
  }

  beforeEach(() => {
    vi.spyOn(manifestModule, "loadManifest").mockResolvedValue({
      rev: "test", effects: { vfx_monster_death: DISSOLVE }, locStrings: {},
    } as VfxManifest);
  });

  it("gives a small rig a duration well under the 2.5s ceiling", async () => {
    // `byrdpip` is 583x599 rig units; at 0.11 its larger padded extent is far
    // short of the 5%-of-screen point where the `Math.Min` binds.
    const scale = 0.11;
    const box = RIG_BOUNDS.byrdpip!;
    const expected = dissolveSecondsFor(box.w * scale, box.h * scale, 1, 2.5, SCENE_W);
    expect(expected).toBeLessThan(2.5);
    expect(await fadeSecondsAfterDeath("monsters/byrdpip", scale)).toBeCloseTo(expected, 5);
  });

  it("still gives a big rig the full ceiling", async () => {
    // `fabricator` is 4067x3185 — comfortably past the point the cap binds, so
    // this half also proves the shortening above is the BOUNDS talking and not
    // a blanket change to every corpse.
    //
    // (The real Fabricator overrides `ShouldFadeAfterDeath => false` and never
    // dissolves in the game at all — see the deathFade block below. It still
    // dissolves HERE because the fixture enemy carries no `kind`, which is the
    // documented "producer that predates the field" fallback. The rig is being
    // used as a big BOX, not as that monster.)
    expect(await fadeSecondsAfterDeath("monsters/fabricator", 0.11)).toBeCloseTo(2.5, 5);
  });
});

describe("Board — the dissolve's DEBRIS BURST is sized per monster", () => {
  beforeEach(() => {
    vi.spyOn(manifestModule, "loadManifest").mockResolvedValue({
      rev: "test", effects: { vfx_monster_death: DISSOLVE }, locStrings: {},
    } as VfxManifest);
  });

  async function burstAfterDeath(rig: string, rigScale: number) {
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={riggedStep(6, rig, rigScale)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board step={killedStep()} actionNode={actionNodeAt(2, [hitRow(6, 0)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />,
      );
    });
    act(() => { advance(HIT_SPACING_MS); });
    return sinkPlays.filter((x) => x.effect === "vfx_monster_death");
  }

  it("plays the burst on the dying enemy, carrying its own radius and duration", async () => {
    // The other half of `vfx_monster_death`: the sprite dissolve is drawn
    // per creature inside `SpineSprite`, the debris burst on the shared
    // overlay. Both numbers are derived from the SAME padded box the
    // duration uses, because the game derives both from one `SubViewport`
    // size (`NMonsterDeathVfx.cs:202-204,231`).
    const scale = 0.11;
    const box = RIG_BOUNDS.byrdpip!;
    const plays = await burstAfterDeath("monsters/byrdpip", scale);
    expect(plays).toHaveLength(1);
    expect(plays[0]!.index).toBe(0);
    // Anchored: the corpse is still mounted at the instant the death event
    // fires (it goes up for the killing blow's own VFX), so the real sink's
    // `anchorRect` has something to measure. Without that the burst would be
    // dropped silently.
    expect(plays[0]!.anchored).toBe(true);
    expect(plays[0]!.dissolve?.emissionRadius).toBeCloseTo(
      deathBurstRadiusFor(box.w * scale, box.h * scale, 1, 0.5)!, 10,
    );
    expect(plays[0]!.dissolve?.lifetime).toBeCloseTo(
      dissolveSecondsFor(box.w * scale, box.h * scale, 1, 2.5, SCENE_W), 10,
    );
  });

  it("gives a big rig a bigger burst — the radius is the BOUNDS talking", async () => {
    // Same assertion shape as the duration pair above, and for the same
    // reason: a constant would satisfy either test alone.
    const small = await burstAfterDeath("monsters/byrdpip", 0.11);
    sinkPlays.length = 0;
    const big = await burstAfterDeath("monsters/fabricator", 0.11);
    expect(big[0]!.dissolve!.emissionRadius).toBeGreaterThan(
      small[0]!.dissolve!.emissionRadius * 4,
    );
  });
});

describe("Board — the scheduler survives a mode change", () => {
  it("still emits queued events when autoplay flips the effects mode mid-playback", () => {
    // `applyEvent` is a `useCallback(..., [mode])`. Keying the scheduler
    // effect on it tore the scheduler down on every mode transition, and
    // `dispose()` clears the queue WITHOUT emitting — so a hit (or a death,
    // or a power) still queued at that instant was simply lost.
    const { clock, advance } = fakeClock();
    const onPlayCard = () => {};
    const board = (autoplaying: boolean) => (
      <EffectsProvider autoplaying={autoplaying}>
        <Board step={loggedStep(32)}
          actionNode={actionNodeAt(2, [hitRow(6, 44), hitRow(6, 38), hitRow(6, 32)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />
      </EffectsProvider>
    );
    const { rerender, container } = render(
      <EffectsProvider autoplaying={false}>
        <Board step={loggedStep(50)} actionNode={actionNodeAt(1, [])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} clock={clock} />
      </EffectsProvider>,
    );
    act(() => { rerender(board(false)); });
    act(() => { advance(HIT_SPACING_MS); });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(2);
    // Autoplay starts: mode full -> reduced, so `applyEvent` is rebuilt.
    act(() => { rerender(board(true)); });
    act(() => { advance(HIT_SPACING_MS); });
    expect(container.querySelectorAll(".gs-dmg")).toHaveLength(3);
  });
});

describe("Board — the real Blocked-text VFX (Task 12)", () => {
  it("shows the 'Blocked' panel, not just the hit-punch flash, on a fully-blocked hit", () => {
    // Same fixture/mechanism as "flinches the player even when the hit is
    // fully blocked" above (round advances, hp unchanged -> the `hadFallback`
    // path's synthesized `fullyBlocked`) — this just asserts the NEW panel
    // that event now also drives.
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={step(1, 80, "Attack")} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(<Board step={step(2, 80, "StatusCard")} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    expect(container.querySelector(".gs-blocked-text")?.textContent).toBe("Blocked");
  });

  it("prefers the loaded manifest's own 'Blocked' loc string over the fallback literal", async () => {
    // Fix round 1: `Combatant.tsx` used to type "Blocked" as a bare literal
    // with a comment falsely claiming it came from the loc row. This proves
    // Board actually THREADS `manifest.locStrings` down to the panel, not
    // just that `Combatant` alone can render one when handed it directly
    // (see `creatureEventVfx.test.tsx` for that half).
    const manifest: VfxManifest = {
      rev: "test", effects: {}, locStrings: { blocked: "BLOCKED (from manifest)" },
    };
    vi.spyOn(manifestModule, "loadManifest").mockResolvedValue(manifest);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={step(1, 80, "Attack")} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    // Let Board's `loadManifest()` effect resolve before the transition.
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(<Board step={step(2, 80, "StatusCard")} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    expect(container.querySelector(".gs-blocked-text")?.textContent).toBe("BLOCKED (from manifest)");
  });
});

/** Same shape as `step()`, but carrying `player_powers` and enemy `powers` —
 *  needed to drive `powerAdd`/`powerRemove` (`combatDiff.ts` diffs those
 *  fields, not `step()`'s bare player/enemy hp/block). Round and intent are
 *  held FIXED across prev/next so combatDiff emits ONLY the power event —
 *  no incidental `hurt`/`fullyBlocked`/`attack` noise to pace around. */
function powerState(over: Partial<CombatState> = {}): CombatState {
  return {
    decision: "combat_play", round: 1, energy: 3, max_energy: 3,
    draw_pile_count: 0, discard_pile_count: 0, exhaust_pile_count: 0,
    player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
    player_powers: [],
    enemies: [{ index: 0, name: "Cultist", hp: 50, max_hp: 50, block: 0, image: null, powers: [] }],
    hand: [],
    ...over,
  } as CombatState;
}

function powerStep(idx: number, over: Partial<CombatState>): Step {
  return {
    idx, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: idx },
    state: powerState(over),
  };
}

describe("Board — the real power-applied/removed panels (Task 12)", () => {
  it("shows the applied panel with the power's own icon, coloured by sfx_map's valence", async () => {
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Vulnerable: "Debuff" }, cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    // Let Board's own `loadSfxMap()` effect (independent of `useSound()` —
    // see its comment in Board.tsx) resolve before the transition that
    // reads it.
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Vulnerable", amount: 2, image: "/img/vulnerable.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    const pop = container.querySelector('[data-testid="power-pop"]');
    expect(pop).not.toBeNull();
    expect(pop).toHaveAttribute("data-mode", "applied");
    expect(pop?.querySelector("img")).toHaveAttribute("src", "/img/vulnerable.png");
    // Vulnerable classified "Debuff" in the (mocked) table -> the red class.
    expect(pop?.querySelector(".gs-power-pop-label")?.className).toMatch(/gs-debuff/);
  });

  it("wobbles the creature when a DEBUFF lands — `NCreature.cs:901`'s AnimShake", async () => {
    // `if (!flag) AnimShake()`, where `flag` is the same `GetTypeForAmount`
    // valence the panel colour reads. jsdom has no `Element.animate`, so the
    // attribute is what is observable; the curve is pinned in
    // `effects/__tests__/animShake.test.ts`.
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Vulnerable: "Debuff" }, cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Vulnerable", amount: 2, image: "/img/vulnerable.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    expect(container.querySelector('[data-combatant="player"][data-anim-shake]')).not.toBeNull();
  });

  it("does NOT wobble when a buff lands", async () => {
    // The C# guard is `if (!flag)`, inside the same block that picks the buff
    // chime — so a Strength gain gets the green panel and no shake at all.
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Strength: "Buff" }, powersAllowNegative: ["Strength"],
      cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Strength", amount: 3, image: "/img/strength.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    expect(container.querySelector('[data-combatant="player"][data-anim-shake]')).toBeNull();
  });

  it("wobbles for a NEGATIVE Strength, because the shake follows the amount too", async () => {
    // The one case that proves the shake reads `GetTypeForAmount` rather than
    // the power's static `Type`: Strength is typed Buff, so a name-keyed guard
    // would leave a debuff landing silently.
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Strength: "Buff" }, powersAllowNegative: ["Strength"],
      cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Strength", amount: -3, image: "/img/strength.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    expect(container.querySelector('[data-combatant="player"][data-anim-shake]')).not.toBeNull();
  });

  it("colours a NEGATIVE Strength red, the way GetTypeForAmount does", async () => {
    // `PowerModel.cs:462-472`: an `AllowNegative` power's rendered valence
    // follows its AMOUNT, not its `Type`. Strength is typed Buff, so a
    // name-keyed lookup drew a green panel drifting upward for what is, on the
    // board, a debuff.
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Strength: "Buff" }, powersAllowNegative: ["Strength"],
      cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Strength", amount: -3, image: "/img/strength.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    const pop = container.querySelector('[data-testid="power-pop"]');
    expect(pop?.querySelector(".gs-power-pop-label")?.className).toMatch(/gs-debuff/);
  });

  it("still colours a POSITIVE Strength green, so the flip is the amount and not the name", async () => {
    const map: SfxMap = {
      version: "test", generic: {}, powers: { Strength: "Buff" }, powersAllowNegative: ["Strength"],
      cards: {}, relics: {}, monsters: {},
    };
    vi.spyOn(sfxMapModule, "loadSfxMap").mockResolvedValue(map);
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Strength", amount: 3, image: "/img/strength.png" }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    const pop = container.querySelector('[data-testid="power-pop"]');
    expect(pop?.querySelector(".gs-power-pop-label")?.className).toMatch(/gs-buff\b/);
  });

  it("falls back to the buff colour (gain ⇒ buff) when the valence table hasn't loaded", async () => {
    // Never resolves within this test — pins the addendum's hard constraint:
    // the panel must still render, not wait on (or silently depend on)
    // `sfx_map.json`'s load state.
    vi.spyOn(sfxMapModule, "loadSfxMap").mockReturnValue(new Promise(() => {}));
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={powerStep(0, {})} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board
          step={powerStep(1, { player_powers: [{ name: "Strength", amount: 1, image: null }] })}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    const pop = container.querySelector('[data-testid="power-pop"]');
    expect(pop).not.toBeNull();
    expect(pop?.querySelector(".gs-power-pop-label")?.className).toMatch(/gs-buff\b/);
  });

  it("shows the removed panel with the icon from BEFORE the power disappeared", async () => {
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board
        step={powerStep(0, { player_powers: [{ name: "Weak", amount: 1, image: "/img/weak.png" }] })}
        onPlayCard={onPlayCard} onEndTurn={() => {}}
      />,
    );
    act(() => {
      // Weak is gone from `next` — `combatDiff` can only see it via `prev`,
      // exactly the case `powerIconsRef` (Board.tsx) exists to cover.
      rerender(<Board step={powerStep(1, { player_powers: [] })} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    const pop = container.querySelector('[data-testid="power-pop"]');
    expect(pop).not.toBeNull();
    expect(pop).toHaveAttribute("data-mode", "removed");
    expect(pop?.querySelector("img")).toHaveAttribute("src", "/img/weak.png");
    // No colour class at all on a removal.
    expect(pop?.querySelector(".gs-power-pop-label")?.className).not.toMatch(/gs-buff|gs-debuff/);
  });
});

describe("Board — the block-break VFX, anchored to the badge (Task 13)", () => {
  /** Same shape as `step()`, but with a settable player `block` — needed to
   *  drive `blockBreak` (`combatDiff.ts`: `prev.block > 0 && next.block ===
   *  0`), which `step()`'s fixed `block: 0` can never trigger. */
  function blockStep(round: number, playerBlock: number): Step {
    const st = step(round, 80, "StatusCard");
    return { ...st, state: { ...st.state, player: { ...st.state.player, block: playerBlock } } };
  }

  it("plays vfx_block_broken when block empties out", () => {
    // The mocked sink (see the module doc above) doesn't inspect `p.anchor`
    // itself — `badgeRect`'s own unit tests and `effectsForEvent`'s
    // `anchor: "badge"` assertion (`bindings.test.ts`) cover THAT half. This
    // proves the OTHER half end to end: a `blockBreak` event from a real
    // Board transition actually reaches `sink.play` with the right effect.
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={blockStep(1, 12)} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    // NOT a discrimination on block, despite the fixture's `block: 12`.
    // `Combatant.tsx` mounts this badge on `!rigLoading` (`:371`) and hides
    // it with `opacity: 0`, deliberately never gating the DIV on
    // `c.block > 0` (`:354-355`, with the reasoning; `creatureEventVfx`'s
    // "mounts [data-block-badge] even at block: 0" case pins it). So this
    // assertion cannot fail for a block reason at all.
    //
    // What it does pin is that the ANCHOR exists in this fixture's tree: the
    // real `VfxLayer` resolves `anchor: "badge"` by querying
    // `[data-block-badge]` inside the combatant (`VfxLayer.tsx:94`), and the
    // mocked sink above resolves against the COMBATANT instead — so if the
    // badge stopped being rendered, the `anchored: true` expectation below
    // would still pass and nothing else in this file would notice.
    expect(container.querySelector("[data-block-badge]")).not.toBeNull();
    act(() => {
      rerender(<Board step={blockStep(2, 0)} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    });
    expect(sinkPlays).toContainEqual({ effect: "vfx_block_broken", index: -1, anchored: true });
  });
});

describe("Board — the power flash VFX (Task 13)", () => {
  it("plays vfx_power_flash_vfx, carrying the flashing power's own icon URL", () => {
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={poweredStep(1, 50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board
          step={poweredStep(2, 50)}
          actionNode={actionNodeAt(2, [powerFlashRow("Thorns")])}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    expect(sinkPlays).toContainEqual({
      effect: "vfx_power_flash_vfx", index: 0, anchored: true, iconUrl: "/img/thorns.png",
    });
  });

  it("draws the power's BIG icon, not the tray sprite", () => {
    // `NPowerFlashVfx.cs:94` assigns `_power.BigIcon`, and the effect's
    // authored `scale: 0.4` is sized against that 256px sprite — fed the 64px
    // tray icon instead, the flash rendered at a quarter of its intended size.
    // So this pins an ASSET CHOICE that is also a geometry fix.
    const onPlayCard = () => {};
    const withBoth = (round: number): Step => {
      const st = poweredStep(round, 50);
      return {
        ...st,
        state: {
          ...st.state,
          enemies: (st.state.enemies ?? []).map((e) => ({
            ...e,
            powers: [{ name: "Thorns", amount: 3, image: "/img/tray.png", image_big: "/img/big.png" }],
          })),
        },
      };
    };
    const { rerender } = render(
      <Board step={withBoth(1)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board step={withBoth(2)} actionNode={actionNodeAt(2, [powerFlashRow("Thorns")])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    expect(sinkPlays).toContainEqual({
      effect: "vfx_power_flash_vfx", index: 0, anchored: true, iconUrl: "/img/big.png",
    });
  });

  it("falls back to the tray icon when the art tree predates the big icons", () => {
    // A deployed bucket can still be serving an art tree extracted before the
    // `power_big/` pass existed. That must degrade to the previous behaviour,
    // not to a missing image.
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={poweredStep(1, 50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board step={poweredStep(2, 50)} actionNode={actionNodeAt(2, [powerFlashRow("Thorns")])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    expect(sinkPlays).toContainEqual({
      effect: "vfx_power_flash_vfx", index: 0, anchored: true, iconUrl: "/img/thorns.png",
    });
  });

  it("carries iconUrl: null rather than crashing when the flashing power has no icon on record", () => {
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={poweredStep(1, 50, "Thorns", null)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board
          step={poweredStep(2, 50, "Thorns", null)}
          actionNode={actionNodeAt(2, [powerFlashRow("Thorns")])}
          onPlayCard={onPlayCard} onEndTurn={() => {}}
        />,
      );
    });
    expect(sinkPlays).toContainEqual({ effect: "vfx_power_flash_vfx", index: 0, anchored: true, iconUrl: null });
  });

  it("does NOT replay the target's hurt clip: a flash is a tint on the power's icon, not a creature reaction", () => {
    // `NCreature.cs:925` spawns the flash node on the power's owner and
    // nothing else — there is no hurt trigger anywhere on that path. The
    // board used to bump the shared per-target `fx` nonce for EVERY event
    // reaching `applyEvent`, and an enemy's `actionNonce` reads that nonce
    // directly while `actionAnim` defaults to "hurt" — so a Cultist's
    // Ritual, a Byrdonis's Territorial and Thorns on every hit received
    // made the enemy visibly flinch on each tick.
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={poweredRigStep(1)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    const before = container.querySelector('[data-testid="spine-mock"]');
    expect(before).not.toBeNull();
    expect(before?.getAttribute("data-action-nonce")).toBeNull();
    act(() => {
      rerender(
        <Board step={poweredRigStep(2)} actionNode={actionNodeAt(2, [powerFlashRow("Thorns")])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    const after = container.querySelector('[data-testid="spine-mock"]');
    // The clip it WOULD play is still "hurt" (nothing else is in flight), so
    // the nonce is what discriminates: the one-shot must never be armed.
    expect(after?.getAttribute("data-action-anim")).toBe("hurt");
    expect(after?.getAttribute("data-action-nonce")).toBeNull();
    // ...while the flash node itself still plays, unchanged.
    expect(sinkPlays.map((p) => p.effect)).toContain("vfx_power_flash_vfx");
  });

  it("does NOT replay the target's hurt clip on a heal, for the same reason", () => {
    // The twin of the case above, and the same one-line cause: `heal` names no
    // branch in `applyEvent`'s if-chain, so its `fx` entry carried nothing but
    // the nonce bump that arms the enemy's one-shot. Healing is not a hit.
    const onPlayCard = () => {};
    const { rerender, container } = render(
      <Board step={healedRigStep(1, 30)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    expect(container.querySelector('[data-testid="spine-mock"]')?.getAttribute("data-action-nonce")).toBeNull();
    act(() => {
      rerender(
        <Board step={healedRigStep(2, 45)} actionNode={actionNodeAt(2, [])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    const after = container.querySelector('[data-testid="spine-mock"]');
    expect(after?.getAttribute("data-action-anim")).toBe("hurt");
    expect(after?.getAttribute("data-action-nonce")).toBeNull();
  });

  it("does NOT attach an iconUrl to an ordinary hurt event's pick", () => {
    // `iconUrl` is `powerFlash`-only (Board.tsx) — an unrelated hit must not
    // pick up a stale value from a previous power flash in the same action.
    const onPlayCard = () => {};
    const { rerender } = render(
      <Board step={loggedStep(50)} actionNode={actionNodeAt(1, [])}
        onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    act(() => {
      rerender(
        <Board step={loggedStep(44)} actionNode={actionNodeAt(2, [hitRow(6, 44)])}
          onPlayCard={onPlayCard} onEndTurn={() => {}} />,
      );
    });
    const hit = sinkPlays.find((p) => p.effect === "vfx_attack_slash");
    expect(hit?.iconUrl).toBeUndefined();
  });
});


/** A bindings table carrying just the `deathFade` rows a test needs — the rest
 *  of the shape is what `loadBindings` would return for a decomp with no
 *  card/move bindings at all, which no other assertion here reads. */
function bindingsWithDeathFade(deathFade: Record<string, { fade?: string; deathAnimSeconds?: number }>) {
  return { version: "test", effects: {}, bindings: [], deathFade } as unknown as Awaited<
    ReturnType<typeof bindingsModule.loadBindings>
  >;
}

describe("Board — ShouldFadeAfterDeath gates the dissolve (NCreature.cs:1106)", () => {
  beforeEach(() => {
    vi.spyOn(manifestModule, "loadManifest").mockResolvedValue({
      rev: "test", effects: { vfx_monster_death: DISSOLVE }, locStrings: {},
    } as VfxManifest);
  });

  /** Kill one enemy of `kind` and report what its corpse ended up with.
   *
   *  `rig` matters to the chrome-fade assertions and is NOT incidental: at
   *  0.11 the fabricator's box is past the point `NMonsterDeathVfx.cs:235`'s
   *  `Math.Min` binds, so a dissolve on it would also read 2.5s — the exact
   *  number `Crusher`'s `DeathAnimLengthOverride` is. Asserting 2.5 against
   *  that rig would pass whether the gate worked or not, so the death-wait test
   *  below uses `byrdpip`, whose dissolve would be well under it.
   */
  async function killOne(kind: string | undefined, rig = "monsters/fabricator", rigScale = 0.11) {
    const { clock, advance } = fakeClock();
    const rigged = (hp: number): Step => {
      const st = riggedStep(hp, rig, rigScale);
      return {
        ...st,
        state: {
          ...st.state,
          enemies: (st.state.enemies ?? []).map((e) => ({ ...e, kind })),
        },
      } as unknown as Step;
    };
    const { rerender, container } = render(
      <Board step={rigged(6)} actionNode={actionNodeAt(1, [])}
        onPlayCard={() => {}} onEndTurn={() => {}} clock={clock} />,
    );
    // Both tables have to be in hand BEFORE the death: the manifest for the
    // dissolve entry, the bindings for the gate. Two awaits, because the two
    // fetches resolve on separate microtask chains.
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await Promise.resolve(); });
    act(() => {
      rerender(
        <Board step={killedStep()} actionNode={actionNodeAt(2, [hitRow(6, 0)])}
          onPlayCard={() => {}} onEndTurn={() => {}} clock={clock} />,
      );
    });
    act(() => { advance(HIT_SPACING_MS); });
    const dying = container.querySelector(".gs-dying") as HTMLElement | null;
    return {
      chromeFadeSeconds: dying ? parseFloat(dying.style.getPropertyValue("--gs-dying-ms")) : null,
      burstPlays: sinkPlays.filter((p) => p.effect === "vfx_monster_death"),
      container,
      advanceBy: advance,
    };
  }

  it("gives a `fade: never` monster NO dissolve and NO debris burst", async () => {
    // The game constructs one `NMonsterDeathVfx` or none; the sprite dissolve
    // and the burst are two halves of that ONE node, so there is no state where
    // only one of them fires. Asserting both is what pins that.
    vi.spyOn(bindingsModule, "loadBindings").mockResolvedValue(
      bindingsWithDeathFade({ Fabricator: { fade: "never" } }),
    );
    const { burstPlays } = await killOne("Fabricator");
    expect(burstPlays).toHaveLength(0);
  });

  it("still dissolves a monster the table says nothing about", async () => {
    // The gate's other direction. `deathFade` publishes only the exceptions, so
    // a bug that read "absent" as "never" would silently stop every ordinary
    // corpse dissolving — and nothing else here would catch it.
    vi.spyOn(bindingsModule, "loadBindings").mockResolvedValue(
      bindingsWithDeathFade({ Fabricator: { fade: "never" } }),
    );
    const { burstPlays } = await killOne("Cultist");
    expect(burstPlays).toHaveLength(1);
  });

  it("paces a non-fading monster's chrome by DeathAnimLengthOverride", async () => {
    // `Crusher`/`Rocket`'s 2.5s (`NCreature.cs:1093`). With no dissolve to read
    // a duration from, the chrome would otherwise take the 0.9s fallback and
    // vanish while the corpse was still on the board.
    vi.spyOn(bindingsModule, "loadBindings").mockResolvedValue(
      bindingsWithDeathFade({ Crusher: { fade: "never", deathAnimSeconds: 2.5 } }),
    );
    // `byrdpip` deliberately: its dissolve would be well under 2.5s, so 2.5s
    // can only have come from the death-wait override. On a big rig this
    // assertion would hold with the gate removed entirely.
    const wouldBe = dissolveSecondsFor(
      RIG_BOUNDS.byrdpip!.w * 0.11, RIG_BOUNDS.byrdpip!.h * 0.11, 1, 2.5, SCENE_W,
    );
    expect(wouldBe).toBeLessThan(2.4);
    const { chromeFadeSeconds } = await killOne("Crusher", "monsters/byrdpip", 0.11);
    expect(chromeFadeSeconds).toBeCloseTo(2.5, 5);
  });

  it("schedules a non-fading corpse's REMOVAL off the death wait too", async () => {
    // The chrome fade and the removal timer are two separate reads of the same
    // pair of numbers, and only the fade is visible in the assertions above. If
    // they disagreed, the corpse would either vanish mid-fade or linger until
    // `STUCK_CORPSE_MS`'s 4s safety net swept it — a bug the safety net would
    // then HIDE. The timer runs on `window.setTimeout` (not the injected
    // `Clock`, which only drives the effect scheduler), so the delay is what is
    // observable here without fighting two clocks at once.
    vi.spyOn(bindingsModule, "loadBindings").mockResolvedValue(
      bindingsWithDeathFade({ Crusher: { fade: "never", deathAnimSeconds: 2.5 } }),
    );
    const timer = vi.spyOn(window, "setTimeout");
    await killOne("Crusher", "monsters/byrdpip", 0.11);
    // `dissolveMs + 100` — the same "comfortably past" margin the dissolving
    // path uses. Byrdpip's own dissolve would be well under 2.5s, so this delay
    // can only have come from the death wait.
    const delays = timer.mock.calls.map((c) => c[1]);
    expect(delays).toContain(2500 + 100);
    timer.mockRestore();
  });

  it("falls back to 0.9s for a non-fading monster that sets no override", async () => {
    vi.spyOn(bindingsModule, "loadBindings").mockResolvedValue(
      bindingsWithDeathFade({ Fabricator: { fade: "never" } }),
    );
    const { chromeFadeSeconds } = await killOne("Fabricator");
    expect(chromeFadeSeconds).toBeCloseTo(DEFAULT_DYING_FADE_SECONDS, 5);
  });
});
