import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Ctx, type Internal } from "../../inspect/useInspect";
import PlayerBand from "../PlayerBand";
import { BAND_ORB_MIN, bandOrbRow, bandPowerCells, bandPowerCellsBesideEnergy } from "../playerBandMetrics";
import type { Orb, Power } from "../../api";

const player = (block: number) => ({ name: "Ironclad", hp: 58, max_hp: 72, block });

const powers = (n: number): Power[] =>
  Array.from({ length: n }, (_, i) => ({
    name: `Power ${i}`, amount: i + 1, image: `/p${i}.png`,
  })) as Power[];

const orbs = (n: number): Orb[] =>
  Array.from({ length: n }, (_, i) => ({
    index: i, name: `Orb ${i}`, type: "Lightning", passive: 3, evoke: 8, image: `/o${i}.png`,
  })) as Orb[];

describe("bandPowerCells", () => {
  it("fits 44 px cells with 5 px gaps beside the orbs, at least one", () => {
    expect(bandPowerCells(262, 0)).toBe(5);
    expect(bandPowerCells(262, 3)).toBe(3);
    expect(bandPowerCells(40, 0)).toBe(1);
    expect(bandPowerCells(40, 10)).toBe(1);
  });
});

describe("the player band", () => {
  it("bar, shield and powers", () => {
    const { container } = render(<PlayerBand player={player(12)} powers={powers(8)} widthPx={262} />);
    expect(container.querySelectorAll(".gs-hpbar")).toHaveLength(1);
    expect(container.querySelector(".gs-hptext")!.textContent).toBe("58/72");
    const shield = container.querySelectorAll<HTMLElement>('[data-block-badge-for="player"]');
    expect(shield).toHaveLength(1);
    expect(shield[0].textContent).toBe("12");
    expect(bandPowerCells(262, 0)).toBe(5);
    expect(container.querySelectorAll(".gs-power")).toHaveLength(4);
    const more = container.querySelectorAll(".gs-power-more");
    expect(more).toHaveLength(1);
    expect(more[0].textContent).toBe("+4");
  });

  it("lays the cells out in one row", () => {
    const { container } = render(<PlayerBand player={player(0)} powers={powers(8)} widthPx={262} />);
    const grid = container.querySelector<HTMLElement>(".gs-powers > div")!;
    expect(grid.style.gridTemplateColumns).toBe("repeat(5, 44px)");
  });

  it("no powers, no row", () => {
    const { container } = render(<PlayerBand player={player(0)} powers={[]} widthPx={262} />);
    expect(container.querySelector(".gs-powers")).toBeNull();
  });

  it("orbs lead the second line", () => {
    const { container } = render(
      <PlayerBand player={player(0)} powers={powers(3)} orbs={orbs(3)} orbSlots={3} widthPx={262} />,
    );
    const o = container.querySelector(".gs-orbs")!;
    const p = container.querySelector(".gs-powers")!;
    expect(o).not.toBeNull();
    expect(o.compareDocumentPosition(p) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(bandPowerCells(262, 3)).toBe(3);
    expect(container.querySelectorAll(".gs-power")).toHaveLength(3);
    expect(container.querySelector(".gs-power-more")).toBeNull();
  });

  it("shield keeps its anchor at block 0", () => {
    const { container } = render(<PlayerBand player={player(0)} powers={[]} widthPx={262} />);
    const shield = container.querySelector<HTMLElement>('[data-block-badge-for="player"]');
    expect(shield).not.toBeNull();
    expect(shield!.style.opacity).toBe("0");
  });

  it("draws the bar at the band's width", () => {
    const { container } = render(<PlayerBand player={player(0)} powers={[]} widthPx={262} />);
    expect(container.querySelector<HTMLElement>(".gs-hpwrap")!.style.getPropertyValue("--gs-hpw")).toBe("262px");
  });
});

/** The drawn width of the orbs row, from the slot sizes and the gap PlayerBand
 *  sets (jsdom does no layout, so this reads the same inline numbers CSS uses). */
const orbRowWidth = (c: HTMLElement) => {
  const slots = Array.from(c.querySelectorAll<HTMLElement>(".gs-orbs > *"));
  const gap = parseFloat(c.querySelector<HTMLElement>(".gs-playerband-status")!
    .style.getPropertyValue("--gs-band-orb-gap"));
  return { n: slots.length, size: parseFloat(slots[0].style.width),
    width: slots.reduce((w, s) => w + parseFloat(s.style.width), 0) + gap * (slots.length - 1) };
};

describe("the band's orbs row", () => {
  for (const widthPx of [262, 225]) {
    for (const n of [3, 7, 10]) {
      it(`${n} orbs fit one row at ${widthPx} px`, () => {
        const { container } = render(
          <PlayerBand player={player(0)} powers={powers(3)} orbs={orbs(n)} orbSlots={n} widthPx={widthPx} />,
        );
        const row = orbRowWidth(container);
        expect(row.n).toBe(n);
        expect(row.width).toBeLessThanOrEqual(widthPx);
        expect(row.size).toBeGreaterThanOrEqual(BAND_ORB_MIN);
        expect(row.size).toBeLessThanOrEqual(30);
      });
    }
  }

  it("keeps 30 px orbs and 4 px gaps while they fit beside a cell", () => {
    expect(bandOrbRow(262, 3)).toEqual({ size: 30, gap: 4, width: 98 });
  });

  it("leaves a power cell beside the orbs while the floor allows it", () => {
    // 262 - 4 - 44 = 214 for ten orbs: 20 px with a 1 px gap.
    const row = bandOrbRow(262, 10);
    expect(row.width).toBeLessThanOrEqual(262 - 4 - 44);
    expect(bandPowerCells(262, 10)).toBe(1);
  });

  it("gives the powers a line of their own when even the floor fills the band", () => {
    // 225 - 48 = 177 < 10 x 20: the orbs take the whole line (<= 225) and the
    // powers wrap below with the full width.
    expect(bandOrbRow(225, 10).width).toBeLessThanOrEqual(225);
    expect(bandPowerCells(225, 10)).toBe(4);
  });
});

describe("the band's shield", () => {
  it("has a 44 x 44 tap area centred on the 34 px shield", () => {
    const { container } = render(<PlayerBand player={player(12)} powers={[]} widthPx={262} />);
    const hit = container.querySelector<HTMLElement>('[data-block-badge-for="player"] > .gs-block-hit')!;
    expect(hit).not.toBeNull();
    expect(parseFloat(hit.style.width)).toBeGreaterThanOrEqual(44);
    expect(parseFloat(hit.style.height)).toBeGreaterThanOrEqual(44);
    expect(hit.style.left).toBe("50%");
    expect(hit.style.top).toBe("50%");
    expect(hit.style.transform).toBe("translate(-50%, -50%)");
  });

  it("a tap on it opens the Block sheet", () => {
    const open = vi.fn();
    const ctx: Internal = { item: null, open, close() {}, enabled: true, chromeEnabled: true,
      registerHost: () => () => {}, registerSuppress: () => () => {} };
    const { container } = render(
      <Ctx.Provider value={ctx}>
        <PlayerBand player={player(12)} powers={[]} widthPx={262} />
      </Ctx.Provider>,
    );
    fireEvent.click(container.querySelector(".gs-block-hit")!);
    expect(open).toHaveBeenCalledTimes(1);
    expect(open.mock.calls[0][0]).toMatchObject({ name: "Block", amount: 12 });
    expect(open.mock.calls[0][0].key).toMatch(/^block:/);
  });
});

describe("big numbers", () => {
  // jsdom does no layout, so the "not clipped" half of this is measured in a
  // real browser (task-2-report.md, fix round 1). What jsdom can pin: every
  // digit is in the DOM, and nothing on the way up from the numerals hides
  // overflow or ellipsises.
  it("999/999 HP and block 999 render in full", () => {
    const { container } = render(
      <PlayerBand player={{ name: "Ironclad", hp: 999, max_hp: 999, block: 999 }} powers={[]} widthPx={225} />,
    );
    const text = container.querySelector<HTMLElement>(".gs-hptext")!;
    const num = container.querySelector<HTMLElement>(".gs-block-num")!;
    expect(text.textContent).toBe("999/999");
    expect(num.textContent).toBe("999");
    for (const el of [text, num]) {
      for (let n: HTMLElement | null = el; n && n !== container; n = n.parentElement) {
        expect(n.style.overflow).toBe("");
        expect(n.style.textOverflow).toBe("");
      }
    }
  });
});

describe("a zoomed card: the energy orb joins the band", () => {
  it("draws the orb last, one power cell beside it at the narrowest block", () => {
    const { container } = render(
      <PlayerBand player={player(12)} powers={powers(3)} widthPx={92} energy={<div className="gs-energy" />} />,
    );
    const status = container.querySelector(".gs-playerband-status")!;
    expect(status.lastElementChild!.className).toBe("gs-band-energy");
    expect(status.querySelector(".gs-band-energy .gs-energy")).not.toBeNull();
    expect(bandPowerCellsBesideEnergy(92)).toBe(1);
    expect(bandPowerCellsBesideEnergy(190)).toBe(3);
  });

  it("the orbs take their own line and shrink to fit it", () => {
    for (const n of [3, 5, 10]) {
      const row = bandOrbRow(92, n, true);
      expect(row.width, `${n} orbs`).toBeLessThanOrEqual(92);
    }
    // Off the zoom the floor holds.
    expect(bandOrbRow(92, 10).size).toBe(BAND_ORB_MIN);
  });
});
