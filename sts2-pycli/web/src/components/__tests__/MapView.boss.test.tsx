import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MapView from "../MapView";
import { bossArt } from "../mapArt";
import { DEFAULT_GAME_VERSION } from "../../gameVersion";
import { HintsFixture } from "../../hints/HintsContext";
import type { FullMap, MapGraph } from "../../api";

/** An act whose current node is the last floor before the boss, so the boss is
 *  the one legal move — the shape of the real pre-boss map decision. */
function graph(boss: NonNullable<FullMap["boss"]>): MapGraph {
  return {
    idx: 0, act: 1, act_name: "act 1", floor: 15,
    full_map: {
      rows: [[
        { col: 0, row: 0, type: "Rest", children: [{ col: 0, row: 1 }], visited: true },
      ]],
      boss,
      current_coord: { col: 0, row: 0 },
    },
  };
}

const BOSS = { col: 0, row: 1, type: "Boss", name: "The Guardian" };
const SELECTABLE = new Set(["0,1"]);

/** An A10 (DoubleBoss) act: the boss node with a SECOND boss node directly
 *  above it, each a different encounter — the shape `attach_second_boss`
 *  produces and `NBossMapPoint` draws with two different illustrations. */
function doubleBossGraph(): MapGraph {
  return {
    idx: 0, act: 3, act_name: "act 3", floor: 15,
    full_map: {
      rows: [
        [{ col: 0, row: 0, type: "Rest", children: [{ col: 0, row: 1 }], visited: true }],
        // The second boss arrives as an ordinary row node (only `ActMap::boss`
        // is grid-exempt in `full_map_view`) — its identity comes from the
        // `second_boss` meta key.
        [], [{ col: 0, row: 2, type: "Boss", children: [] }],
      ],
      boss: { col: 0, row: 1, type: "Boss", id: "QUEEN_BOSS", name: "Queen" },
      second_boss: { col: 0, row: 2, type: "Boss", id: "VANTOM_BOSS", name: "Vantom" },
      current_coord: { col: 0, row: 0 },
    },
  };
}

describe("the act boss on the parchment map", () => {
  it("can be clicked to take the boss node", () => {
    // The boss illustration is `pointer-events: none`, so the node needs its
    // own hit target. Without one the LAST decision of the act cannot be made
    // at all — the map is the picker whenever the wire ships a full graph.
    const onSelect = vi.fn();
    render(
      <HintsFixture on={false}>
        <MapView rows={[]} graphs={[graph(BOSS)]} currentIdx={0}
          selectable={SELECTABLE} onSelect={onSelect} live />
      </HintsFixture>,
    );
    const node = screen.getByRole("button", { name: /act boss/i });
    fireEvent.click(node);
    expect(onSelect).toHaveBeenCalledWith(0, 1);
  });

  it("takes Enter from the keyboard like every other node", () => {
    const onSelect = vi.fn();
    render(
      <HintsFixture on={false}>
        <MapView rows={[]} graphs={[graph(BOSS)]} currentIdx={0}
          selectable={SELECTABLE} onSelect={onSelect} live />
      </HintsFixture>,
    );
    fireEvent.keyDown(screen.getByRole("button", { name: /act boss/i }), { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith(0, 1);
  });

  it("still names itself when there is no silhouette to print", () => {
    // `assets.py room_icon()` returns None for a Boss with no `boss_id`, so a
    // graph with no name and no icon is a state the server actually produces.
    // The illustration is then the only thing in the node — it must not be
    // allowed to render as an empty group.
    const { container } = render(
      <HintsFixture on={false}>
        <MapView rows={[]} graphs={[graph({ col: 0, row: 1, type: "Boss" })]} currentIdx={0} live />
      </HintsFixture>,
    );
    expect(container.textContent).toContain("BOSS");
  });
});

describe("A10's two bosses", () => {
  it("names each boss node after its own encounter", () => {
    // C# `NBossMapPoint._Ready` picks `Point == Map.SecondBossMapPoint ?
    // Act.SecondBossEncounter : Act.BossEncounter`. Keying both nodes off
    // `full_map.boss` printed the act boss twice — same face, same name.
    render(
      <HintsFixture on={false}>
        <MapView rows={[]} graphs={[doubleBossGraph()]} currentIdx={0} live />
      </HintsFixture>,
    );
    expect(screen.getByRole("img", { name: "Act boss: Queen" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Second boss: Vantom" })).toBeTruthy();
  });

  it("draws the two silhouettes from different art", () => {
    const { container } = render(
      <HintsFixture on={false}>
        <MapView rows={[]} graphs={[doubleBossGraph()]} currentIdx={0} live />
      </HintsFixture>,
    );
    const hrefs = [...container.querySelectorAll("image.gsm-boss-ink")]
      .map((i) => i.getAttribute("href"));
    expect(hrefs).toHaveLength(2);
    expect(hrefs[0]).not.toBe(hrefs[1]);
  });
});

describe("the boss silhouette's art chain", () => {
  it("reaches the run-history badge only after the creature illustration", () => {
    // The badge is the opaque hexagonal run-history stamp: its picture is in
    // its COLOUR, so flooding its alpha (which is what the boss filter does)
    // prints the boss as a featureless black lozenge — the Queen, live, on
    // 2026-08-26, because the badge used to sit AHEAD of `monsters/`.
    const art = bossArt({ col: 0, row: 1, type: "Boss", id: "QUEEN_BOSS", name: "Queen" });
    const monster = art.candidates.findIndex((u) => u.endsWith("monsters/queen.png"));
    const badge = art.candidates.findIndex((u) => u.includes("run_history/"));
    expect(monster).toBeGreaterThanOrEqual(0);
    expect(badge).toBeGreaterThan(monster);
  });

  it("keeps a server-resolved badge behind the creature illustration too", () => {
    // `_enrich_map_nodes` resolves that same badge onto the node as `icon`;
    // taking it just because the server sent it is the same bug.
    const art = bossArt(
      { col: 0, row: 1, type: "Boss", id: "QUEEN_BOSS", name: "Queen" },
      "/art/run_history/queen_boss.png",
    );
    const monster = art.candidates.findIndex((u) => u.endsWith("monsters/queen.png"));
    expect(art.candidates.indexOf("/art/run_history/queen_boss.png")).toBeGreaterThan(monster);
  });

  it("prefers the game's own map art, then the creature illustration", () => {
    const art = bossArt({ col: 0, row: 1, type: "Boss", id: "QUEEN_BOSS", name: "Queen" });
    // `boss_icons/<id>_icon.png` is C# `BossNodePath` for the placeholder
    // bosses; the creature render's alpha IS the silhouette the game prints.
    expect(art.candidates[0]).toContain("boss_icons/queen_boss_icon.png");
    expect(art.candidates[1]).toContain("monsters/queen.png");
  });

  it("tries the article-stripped monster slug too", () => {
    // `monsters/` is generated WITH the leading article dropped, so "The Kin"
    // lives at `monsters/kin.png` and never at `monsters/the_kin.png`.
    const art = bossArt({ col: 0, row: 1, type: "Boss", id: "THE_KIN_BOSS", name: "The Kin" });
    expect(art.candidates.some((u) => u.endsWith("monsters/kin.png"))).toBe(true);
  });

  it("looks in a later build's art overlay before the shared tree", () => {
    // v0.113's Aeonglass ships only in `images/v0.113.0/` (the import's art
    // overlay); without the overlay candidates its map node fell through to
    // the server's small stamp.
    const node = { col: 0, row: 1, type: "Boss", id: "AEONGLASS_BOSS", name: "Aeonglass" };
    const art = bossArt(node, null, "v0.113.0");
    expect(art.candidates[0]).toContain("v0.113.0/boss_icons/aeonglass_boss_icon.png");
    expect(art.candidates[1]).toContain("/boss_icons/aeonglass_boss_icon.png");
    expect(art.candidates[1]).not.toContain("v0.113.0/");
    expect(art.candidates).toContainEqual(expect.stringContaining("v0.113.0/monsters/aeonglass.png"));
  });

  it("adds no overlay candidates for the default build", () => {
    const node = { col: 0, row: 1, type: "Boss", id: "QUEEN_BOSS", name: "Queen" };
    expect(bossArt(node, null, DEFAULT_GAME_VERSION)).toEqual(bossArt(node));
  });
});
