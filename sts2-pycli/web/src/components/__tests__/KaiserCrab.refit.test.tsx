import { act, render } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { CrabArm } from "../KaiserCrabGroup";
import type { Enemy } from "../../api";

/**
 * The Kaiser Crab is drawn at its authored scale times the stage's scene
 * factor, and that factor is a live measurement (Board re-measures the stage on
 * every resize). Three things follow, and all three used to be wrong:
 *
 *  - The rig is NOT fitted to a box. At 0.5 px per rig unit it is wider than
 *    the 1920px scene, which is the fight: a claw past each edge of the screen.
 *  - Re-scaling must be a pure resize. The size used to be a dependency of the
 *    WebGL load effect, so a resize disposed the canvas and re-fetched all four
 *    atlas pages (~5 MB) — the boss gone for about a second, then back.
 *  - Until the rig is measured the arm overlays are pinned at a guessed claw
 *    split, so painting them straight away made every arm's HP bar / name /
 *    intent jump once the rig landed.
 */

const constructions: string[] = [];
// When set, SpineCanvas parks its initialize() instead of running it inline, so
// a test can observe the window where the rig is loading but not yet measured.
const pending: Array<() => void> = [];
let defer = false;

vi.mock("@esotericsoftware/spine-webgl", () => {
  class Vector2 {
    constructor(public x = 0, public y = 0) {}
    set(x: number, y: number) { this.x = x; this.y = y; return this; }
  }
  // A region attachment that paints a box: `computeWorldVertices` is how the
  // renderer (and so the claw measurement) learns where a slot's art actually
  // lands, and a claw's bone is nowhere near the middle of it.
  class RegionAttachment {
    constructor(public name = "region", public box = { l: 0, r: 0, b: 0, t: 0 }) {}
    computeWorldVertices(_slot: unknown, out: Float32Array, offset: number, stride: number) {
      const { l, r, b, t } = this.box;
      const pts = [[l, b], [l, t], [r, t], [r, b]];
      pts.forEach((p, i) => { out[offset + i * stride] = p[0]; out[offset + i * stride + 1] = p[1]; });
    }
  }
  class MeshAttachment { constructor(public name = "mesh") {} }
  class Skin { constructor(public name: string) {} addSkin() {} getAttachment() { return null; } }
  // 4000 x 1600 world units at (-2000, 0) — the crab's wide, low silhouette.
  const BOUNDS = { x: -2000, y: 0, w: 4000, h: 1600 };
  // Each claw's ART spans 600 units around its bone, 400 tall.
  const slot = (name: string, worldX: number, worldY: number) => {
    let attachment: RegionAttachment | null = new RegionAttachment(name,
      { l: worldX - 300, r: worldX + 300, b: worldY - 200, t: worldY + 200 });
    return {
      data: { name }, bone: { worldX, worldY },
      getAttachment: () => attachment,
      setAttachment: (a: RegionAttachment | null) => { attachment = a; },
    };
  };
  class Skeleton {
    slots = [slot("claw_l_1", -1500, 200), slot("claw_r_1", 1500, 100)];
    constructor(public data: unknown) {}
    setSkin() {} setSlotsToSetupPose() {} setToSetupPose() {}
    update() {} updateWorldTransform() {}
    getBounds(off: Vector2, size: Vector2) {
      off.set(BOUNDS.x, BOUNDS.y); size.set(BOUNDS.w, BOUNDS.h);
    }
  }
  class AnimationStateData { constructor(public skeletonData: unknown) {} }
  // Every listener the component registered, in order (see `addListener`).
  const spineEventListeners: unknown[] = [];
  //: Every `setAnimation` the component made, as `track:clip`.
  const setAnimations: string[] = [];
  class AnimationState {
    constructor(public data: AnimationStateData) {}
    setAnimation(track: number, name: string) { setAnimations.push(`${track}:${name}`); return {}; }
    addAnimation() { return {}; }
    update() {} apply() {}
    // The component subscribes to the rig's SPINE EVENTS here (GH #642's
    // `claw_explode_l`). This mock lacking `addListener` is not a harmless
    // gap: `initialize` is wrapped in a `try/catch` that sets `failed`, so a
    // missing method silently turns every assertion in this file into the
    // never-measured fallback — which is exactly how it failed when the
    // listener landed. `spineEventListeners` below is what keeps it honest.
    addListener(l: unknown) { spineEventListeners.push(l); }
  }
  const skeletonData = {
    animations: [
      { name: "body/idle_loop", duration: 1 },
      { name: "left/idle_loop", duration: 1 },
      { name: "right/idle_loop", duration: 1 },
      // The Crusher's death clip, which carries `claw_explode_l` at t=1.53s.
      { name: "left/die", duration: 2.97 },
    ],
    slots: [], skins: [], defaultSkin: null,
    findAnimation(n: string) { return this.animations.find((a) => a.name === n) ?? null; },
    findSkin() { return null; },
  };
  class AtlasAttachmentLoader { constructor(public atlas: unknown) {} }
  class SkeletonBinary {
    constructor(public loader: unknown) {}
    readSkeletonData() { return skeletonData; }
  }
  class SpineCanvas {
    constructor(public canvas: HTMLCanvasElement, public config: { app: Record<string, (c: unknown) => void> }) {
      constructions.push("new");
      const c = {
        assetManager: { loadBinary() {}, loadTextureAtlas() {}, hasErrors: () => false, require: () => ({}) },
        gl: null, renderer: null,
      };
      config.app.loadAssets?.(c);
      if (defer) pending.push(() => config.app.initialize?.(c));
      else config.app.initialize?.(c);
    }
    dispose() {}
  }
  return {
    Vector2, RegionAttachment, MeshAttachment, Skin, Skeleton,
    AnimationState, AnimationStateData, AtlasAttachmentLoader, SkeletonBinary,
    SpineCanvas, Physics: { update: 1 }, ResizeMode: { Expand: 1 },
    // Test-only handle on what the component subscribed (see `addListener`).
    __spineEventListeners: spineEventListeners,
    __setAnimations: setAnimations,
  };
});

const { default: KaiserCrab } = await import("../KaiserCrab");
const { default: KaiserCrabGroup } = await import("../KaiserCrabGroup");

const spineMock = await import("@esotericsoftware/spine-webgl") as unknown as {
  __spineEventListeners: { event?: (entry: unknown, ev: { data: { name: string } }) => void }[];
  __setAnimations: string[];
};

beforeEach(() => {
  constructions.length = 0; pending.length = 0; defer = false;
  spineMock.__spineEventListeners.length = 0;
  spineMock.__setAnimations.length = 0;
});

const canvasW = (c: HTMLElement) =>
  parseFloat((c.querySelector("canvas") as HTMLCanvasElement).style.width);

const arm = (part: string, index: number, name: string): CrabArm => ({
  c: { index, name, hp: 209, max_hp: 209, block: 0, intents: [{ type: "Attack", damage: 12, hits: 1 }] } as unknown as Enemy,
  part, dataIndex: index, actionAnim: "hurt", holdLast: false, dying: false,
});

describe("Kaiser Crab rig events", () => {
  it("forwards a spine event, with a live bone resolver (GH #642)", () => {
    // The Crusher's death explosion is fired by the RIG, not by anything in
    // the model layer: `NKaiserCrabBossVfx` listens for `claw_explode_l` at
    // t=1.53 s of `left/die`. Nothing else in this suite would notice the
    // subscription disappearing.
    const seen: string[] = [];
    render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5}
        onRigEvent={(name) => seen.push(name)} />,
    );
    expect(spineMock.__spineEventListeners).toHaveLength(1);
    spineMock.__spineEventListeners[0].event?.({}, { data: { name: "claw_explode_l" } });
    expect(seen).toEqual(["claw_explode_l"]);
  });

  it("forwards events even with no handler attached", () => {
    // The listener is registered unconditionally, so a parent that passes no
    // `onRigEvent` must not throw on the rig's ten authored events.
    render(<KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5} />);
    expect(spineMock.__spineEventListeners).toHaveLength(1);
    expect(() =>
      spineMock.__spineEventListeners[0].event?.({}, { data: { name: "plow_chunks_end" } }),
    ).not.toThrow();
  });
});

describe("Kaiser Crab one-shots across a rig reload", () => {
  it("plays a death that arrived while the rig was still loading", () => {
    // Board REMOUNTS this component at the moment the arms die, so the whole
    // 5 MB rig is disposed and re-fetched and the `die` one-shot lands in the
    // reload window. `playArm` used to mark the nonce played BEFORE checking
    // the rig was ready, so the action was swallowed and never replayed: the
    // crab's death animation never ran, and neither did the spine events on
    // it — which is what hid GH #642's explosion.
    defer = true;
    render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5}
        leftAction={{ nonce: 1, clip: "die", hold: true }} />,
    );
    expect(spineMock.__setAnimations, "nothing can play before the rig loads")
      .toEqual([]);

    act(() => { pending.forEach((f) => f()); });

    expect(spineMock.__setAnimations).toContain("1:left/die");
  });

  it("does not replay a one-shot the loaded rig already played", () => {
    // The held action is cleared when it is replayed, so a LATER reload does
    // not resurrect a death that already ran.
    render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5}
        leftAction={{ nonce: 1, clip: "die", hold: true }} />,
    );
    const plays = spineMock.__setAnimations.filter((a) => a === "1:left/die");
    expect(plays).toHaveLength(1);
  });
});

describe("Kaiser Crab refit", () => {
  it("draws at the given px-per-rig-unit, with no box fit", () => {
    const { container } = render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5} />,
    );
    // 4000 units wide + 8% overscan each side, at 0.5 px/unit.
    expect(canvasW(container)).toBeCloseTo(4000 * 1.16 * 0.5, 1);
  });

  it("re-scales to a resized stage without rebuilding the rig", () => {
    const { container, rerender } = render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5} />,
    );
    expect(constructions).toHaveLength(1);
    const before = canvasW(container);

    rerender(<KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.4} />);
    expect(constructions).toHaveLength(1);
    expect(canvasW(container)).toBeCloseTo(before * 0.8, 1);
  });

  it("places the canvas relative to the rig root, not centred in a box", () => {
    const { container } = render(
      <KaiserCrab rigBase="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5} />,
    );
    const canvas = container.querySelector("canvas") as HTMLCanvasElement;
    // Bounds start at x=-2000 with 8% (320u) of overscan to its left, at 0.5.
    expect(parseFloat(canvas.style.left)).toBeCloseTo((-2000 - 320) * 0.5, 1);
    // Bounds bottom is the root (y=0) less 4% (64u) of ground overscan.
    expect(parseFloat(canvas.style.bottom)).toBeCloseTo(-64 * 0.5, 1);
  });

  it("holds each arm's chrome back until the shared rig is measured", () => {
    defer = true;
    const { container } = render(
      <KaiserCrabGroup rig="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5}
        armAt={{}} clampX={{ min: -960, max: 960 }}
        arms={[arm("left", 0, "Crusher"), arm("right", 1, "Rocket")]} />,
    );
    const overlays = () => [...container.querySelectorAll<HTMLElement>(".gs-crab-group > div")];
    expect(overlays()).toHaveLength(2);
    // Still loading: pinned at the guessed claw split, and invisible.
    for (const o of overlays()) expect(o.style.opacity).toBe("0");

    act(() => { pending.forEach((f) => f()); });

    for (const o of overlays()) expect(o.style.opacity).toBe("1");
    // With no marker given (Board always passes one — Board.crab.test), the
    // claw's ART is the fallback, measured, not its bone: the mock paints it
    // across world x -1800..-1200, so its centre is 750px left of the root at
    // 0.5 px/unit — not the -600px guess the overlays were parked at.
    expect(parseFloat(overlays()[0].style.left)).toBeCloseTo(-750, 1);
  });
});
