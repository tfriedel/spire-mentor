import { render } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, it, expect } from "vitest";
import Board from "../Board";
import fixture from "./fixture-step.json";
import type { Step } from "../../api";
import { SCENE_CX, SCENE_CY, cameraFor, fitScene } from "../../scene";
import { CRAB_ROOT_DX, CRAB_ROOT_Y, crabArmMount } from "../crabMount";

/**
 * The Kaiser Crab boss is a ROOM, not a combatant.
 *
 * The game mounts its one shared rig on the combat-room BACKGROUND
 * (`NKaiserCrabBossBackground` at `%KaiserCrab`) and reaches a claw in from each
 * side of the screen, with the player standing between them. Rendered as a
 * creature in the enemy column it came out as a small crab hunched on the right
 * with both claws side by side — "nothing like the fight".
 */
const crab = (part: string, index: number, name: string) => ({
  index, name, hp: 209, max_hp: 209, block: 0,
  image: null, rig: "/assets/monster_rigs/kaiser_crab/kaiser_crab", rig_scale: 0.5,
  rig_group: "kaiser_crab", rig_part: part,
  intents: [{ type: "Attack", damage: 12, hits: 1 }],
});

function boardWith(enemies: unknown[]) {
  const base = fixture as unknown as Step;
  const step = { ...base, state: { ...base.state, enemies } as Step["state"] };
  return render(<Board step={step} />).container;
}

/** The scene layer's own slots — Board places every fighter into these. */
const slots = (c: Element) => [...c.querySelectorAll<HTMLElement>(".gs-actors > *")];

describe("Kaiser Crab boss layout", () => {
  it("draws the shared rig once, on the scene layer, at its authored scale", () => {
    const c = boardWith([crab("left", 0, "Crusher"), crab("right", 1, "Rocket")]);
    const groups = c.querySelectorAll(".gs-crab-group");
    expect(groups).toHaveLength(1);
    expect(c.querySelector(".gs-actors")!.contains(groups[0])).toBe(true);
    // No crab in the enemy row: the boss is not standing opposite the player.
    for (const side of c.querySelectorAll(".gs-side")) {
      expect(side.querySelector(".gs-crab-group")).toBeNull();
    }
  });

  it("mounts the rig where the game's background does, through the camera", () => {
    // kaiser_crab_boss_background.tscn puts the setup scene at (-1029, -544)
    // and the setup scene its skeleton at (979, 450): the root is (-50, -94)
    // from %BgContainer's origin, which the encounter camera (0.75, Down*35)
    // zooms with the room. Not flushed to the stage top, which is what drew
    // the crab small and low.
    expect(CRAB_ROOT_DX).toBe(23 - 50);
    expect(CRAB_ROOT_Y).toBe(SCENE_CY - 94);
    const c = boardWith([crab("left", 0, "Crusher"), crab("right", 1, "Rocket")]);
    const anchor = c.querySelector(".gs-crab-group")!.parentElement!;
    const stage = c.querySelector<HTMLElement>(".gs-stage-scale")!;
    // jsdom reports 0-size boxes, so Board falls back to its seeded stage box.
    const fit = fitScene(stage.offsetWidth || 1600, stage.offsetHeight || 800);
    const cam = cameraFor(["/assets/monster_rigs/kaiser_crab/kaiser_crab"]);
    const camY = (y: number) => SCENE_CY + (y - SCENE_CY) * cam.scale + cam.offsetY;
    expect(parseFloat(anchor.style.left)).toBeCloseTo(fit.x(CRAB_ROOT_DX * cam.scale), 0);
    expect(parseFloat(anchor.style.top)).toBeCloseTo(fit.y(camY(CRAB_ROOT_Y)), 0);
  });

  it("stands each arm on its own encounter marker, with the game's bar width", () => {
    // Crusher and Rocket are creatures on kaiser_crab_boss.tscn's markers;
    // NHealthBar sizes each bar to its creature's Bounds width + 24.
    expect(crabArmMount("left")).toMatchObject({ dx: -824, y: 736, hpW: 457 + 24, intentH: 580 });
    expect(crabArmMount("right")).toMatchObject({ dx: 896, y: 712, hpW: 431 + 24, intentH: 450 });
    const c = boardWith([crab("left", 0, "Crusher"), crab("right", 1, "Rocket")]);
    const anchor = c.querySelector(".gs-crab-group")!.parentElement!;
    const stage = c.querySelector<HTMLElement>(".gs-stage-scale")!;
    const fit = fitScene(stage.offsetWidth || 1600, stage.offsetHeight || 800);
    const cam = cameraFor(["/assets/monster_rigs/kaiser_crab/kaiser_crab"]);
    const left = c.querySelector<HTMLElement>("[data-crab-arm=left]")!;
    const right = c.querySelector<HTMLElement>("[data-crab-arm=right]")!;
    const rootLeft = parseFloat(anchor.style.left);
    // Each readout is centred on its marker (plus its Bounds' own centre).
    expect(rootLeft + parseFloat(left.style.left)).toBeCloseTo(fit.x((-824 + 8.5) * cam.scale), 0);
    expect(rootLeft + parseFloat(right.style.left)).toBeCloseTo(fit.x((896 + 10.5) * cam.scale), 0);
    const bar = left.querySelector<HTMLElement>(".gs-hpwrap")!;
    expect(parseFloat(bar.style.getPropertyValue("--gs-hpw"))).toBeCloseTo(481 * fit.k * cam.scale, 0);
  });

  it("centres the player, because the claws come in from both sides", () => {
    // `KaiserCrabBoss.FullyCenterPlayers` — the player's slot is x = 0, the
    // scene's own centre line, not the left-hand third an ordinary fight uses.
    const cam = cameraFor(["/assets/monster_rigs/kaiser_crab/kaiser_crab"]);
    expect(cam.centerPlayer).toBe(true);
    const c = boardWith([crab("left", 0, "Crusher"), crab("right", 1, "Rocket")]);
    const stage = c.querySelector<HTMLElement>(".gs-stage-scale")!;
    const player = c.querySelector<HTMLElement>(".gs-side-player")!;
    expect(parseFloat(player.style.left)).toBeCloseTo((stage.offsetWidth || 1600) / 2, 0);
    // One readout per arm, each hung from its own feet like any creature.
    const readouts = [...c.querySelectorAll<HTMLElement>(".gs-crab-group > div")];
    expect(readouts).toHaveLength(2);
    for (const r of readouts) expect(r.style.transform).toBe("translateX(-50%)");
  });

  it("keeps one targetable, named overlay per claw", () => {
    const c = boardWith([crab("left", 0, "Crusher"), crab("right", 1, "Rocket")]);
    const group = c.querySelector(".gs-crab-group")!;
    const names = Array.from(group.querySelectorAll(".gs-nameplate")).map((n) => n.textContent);
    expect(names).toEqual(["Crusher", "Rocket"]);
    expect(group.querySelectorAll("[data-enemy-index]").length).toBe(2);
  });

  it("leaves an ordinary encounter standing in the enemy half", () => {
    const c = render(<Board step={fixture as unknown as Step} />).container;
    expect(c.querySelector(".gs-crab-group")).toBeNull();
    const stage = c.querySelector<HTMLElement>(".gs-stage-scale")!;
    const mid = (stage.offsetWidth || 1600) / 2;
    const enemy = slots(c).find((el) => el.textContent?.includes("Scroll of Biting"));
    expect(enemy).toBeDefined();
    // Right of the centre line; the player is left of it.
    expect(parseFloat(enemy!.style.left)).toBeGreaterThan(mid);
    const player = c.querySelector<HTMLElement>(".gs-side-player")!;
    expect(parseFloat(player.style.left)).toBeLessThan(mid);
    // …and both are inside the scene's own 1920 px, at the default camera.
    const k = fitScene(stage.offsetWidth || 1600, stage.offsetHeight || 800).k;
    expect(parseFloat(enemy!.style.left)).toBeLessThan(mid + SCENE_CX * k);
  });
});

describe("Kaiser Crab boss mount stability", () => {
  /**
   * GH #668. The arms leave `state.enemies` on the frame they die, and the
   * corpse that replaces them used to arrive one render later — so for one
   * COMMITTED frame the board held neither, `<KaiserCrabGroup>` unmounted, and
   * the remount re-fetched the crab's four atlas pages (~5 MB) right as its
   * death clip should have been playing. The node identity is the assertion:
   * a remount would hand back a different element.
   *
   * Rendered in `StrictMode` deliberately. The mount is a render-phase
   * `setDying`, and StrictMode re-invokes the render from the PRE-update
   * state — the pass React commits is the one that saw an empty `dying`, so
   * without the fix this test only fails under StrictMode. Both entry points
   * (`main.tsx`, `boardpreview.tsx`) use it.
   */
  it("holds the same rig node through the frame its arms die", () => {
    const base = fixture as unknown as Step;
    const withArms = {
      ...base,
      state: { ...base.state, enemies: [crab("left", 0, "Crusher"), crab("right", 1, "Rocket")] },
    } as Step;
    const dead = { ...base, state: { ...base.state, enemies: [] } } as Step;
    const view = render(
      <StrictMode><Board step={withArms} onPlayCard={() => {}} /></StrictMode>,
    );
    const before = view.container.querySelector(".gs-crab-group");
    expect(before).not.toBeNull();

    view.rerender(<StrictMode><Board step={dead} onPlayCard={() => {}} /></StrictMode>);
    const after = view.container.querySelector(".gs-crab-group");
    expect(after).not.toBeNull();
    expect(after).toBe(before);
  });
});
