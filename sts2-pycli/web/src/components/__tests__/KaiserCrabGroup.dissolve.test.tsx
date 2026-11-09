import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import type { CrabArm } from "../KaiserCrabGroup";
import type { Enemy } from "../../api";
import type { DissolveEffect } from "../../effects/manifest";

/**
 * Task 14 review finding 1: the Kaiser Crab arm's chrome-only `<Combatant>`
 * overlay (`KaiserCrabGroup.tsx`'s "Targetable overlays" block) never
 * received the `dissolve` prop, even though `CrabArm` declares one — so
 * `dyingChromeFadeSeconds` always fell back to `DEFAULT_DYING_FADE_SECONDS`
 * (0.9s) for a dying arm's chrome, while `Board.tsx`'s corpse-removal timer
 * uses the manifest's real `dissolveSeconds` (2.5s) for every dying entry,
 * grouped or not. Net effect: the arm's HP/name/intent overlay vanished
 * ~1.6s before the corpse the timer keeps around actually got removed.
 *
 * This does NOT give the Kaiser Crab's shared BODY rig a dissolve — that
 * gap is real and stays open (see `CrabArm.dissolve`'s own doc) — it only
 * syncs the arm's own DOM chrome fade to how long the corpse actually lives.
 */
vi.mock("@esotericsoftware/spine-webgl", () => {
  class Vector2 {
    constructor(public x = 0, public y = 0) {}
    set(x: number, y: number) { this.x = x; this.y = y; return this; }
  }
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
  const BOUNDS = { x: -2000, y: 0, w: 4000, h: 1600 };
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
  class AnimationState {
    constructor(public data: AnimationStateData) {}
    setAnimation() { return {}; } addAnimation() { return {}; }
    update() {} apply() {}
  }
  const skeletonData = {
    animations: [
      { name: "body/idle_loop", duration: 1 },
      { name: "left/idle_loop", duration: 1 },
      { name: "right/idle_loop", duration: 1 },
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
      const c = {
        assetManager: { loadBinary() {}, loadTextureAtlas() {}, hasErrors: () => false, require: () => ({}) },
        gl: null, renderer: null,
      };
      config.app.loadAssets?.(c);
      config.app.initialize?.(c);
    }
    dispose() {}
  }
  return {
    Vector2, RegionAttachment, MeshAttachment, Skin, Skeleton,
    AnimationState, AnimationStateData, AtlasAttachmentLoader, SkeletonBinary,
    SpineCanvas, Physics: { update: 1 }, ResizeMode: { Expand: 1 },
  };
});

const { default: KaiserCrabGroup } = await import("../KaiserCrabGroup");

const dissolveEffect = (dissolveSeconds: number) => ({ dissolveSeconds }) as DissolveEffect;

const dyingArm = (part: string, index: number): CrabArm => ({
  c: { index, name: "Crusher", hp: 0, max_hp: 209, block: 0 } as unknown as Enemy,
  part, dataIndex: index, actionAnim: "die", holdLast: true, dying: true,
  dissolve: { effect: dissolveEffect(1.4), rev: "r1", startedAt: 0 },
});

describe("KaiserCrabGroup — a dying arm's chrome fade duration", () => {
  it("carries the manifest's own dissolve duration, not the 0.9s fallback", () => {
    const { container } = render(
      <KaiserCrabGroup rig="/rig/kaiser_crab/kaiser_crab" pxPerUnit={0.5}
        armAt={{}} clampX={{ min: -960, max: 960 }}
        arms={[dyingArm("left", 0)]} />,
    );
    const wrap = container.querySelector(".gs-combatant") as HTMLElement | null;
    expect(wrap, ".gs-combatant overlay not found").not.toBeNull();
    expect(wrap!.style.getPropertyValue("--gs-dying-ms")).toBe("1.4s");
  });
});
