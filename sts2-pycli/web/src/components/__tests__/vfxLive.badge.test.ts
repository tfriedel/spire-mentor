// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { badgeRect } from "../vfxLive";

/** A rect jsdom reports for `el`: its own, by a stubbed getBoundingClientRect. */
function stubRect(el: Element, x: number): void {
  (el as HTMLElement).getBoundingClientRect = () => new DOMRect(x, 0, 10, 10);
}

describe("badgeRect", () => {
  it("falls back to the band's shield when the player figure has none (portrait split)", () => {
    const arena = document.createElement("div");
    arena.innerHTML = `<div data-combatant="player"><div class="gs-sprite-wrap"></div></div>
      <div class="gs-playerband"><div data-block-badge="true" data-block-badge-for="player"></div></div>`;
    stubRect(arena.querySelector('[data-block-badge-for="player"]')!, 42);
    expect(badgeRect({ side: "player" }, arena)?.x).toBe(42);
  });

  it("prefers the figure's own badge", () => {
    const arena = document.createElement("div");
    arena.innerHTML = `<div data-combatant="player"><div data-block-badge="true" class="own"></div></div>
      <div class="gs-playerband"><div data-block-badge="true" data-block-badge-for="player"></div></div>`;
    stubRect(arena.querySelector(".own")!, 7);
    stubRect(arena.querySelector(".gs-playerband [data-block-badge]")!, 42);
    expect(badgeRect({ side: "player" }, arena)?.x).toBe(7);
  });

  it("an enemy without a badge has none", () => {
    const arena = document.createElement("div");
    arena.innerHTML = `<div data-enemy-index="0"></div>
      <div class="gs-playerband"><div data-block-badge="true" data-block-badge-for="player"></div></div>`;
    expect(badgeRect({ side: "enemy", index: 0 }, arena)).toBeNull();
  });
});
