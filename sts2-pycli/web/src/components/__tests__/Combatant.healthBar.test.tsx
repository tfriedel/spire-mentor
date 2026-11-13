import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Combatant from "../Combatant";

/**
 * The HP bar's DOM, pinned. `HealthBar` was lifted out of `Combatant` so the
 * phone-portrait player band can draw the same bar on its own; every other
 * layout (desktop, tablet, phone landscape) must keep exactly the markup it
 * had. The snapshots were written from the unmodified, inline bar.
 */
const enemy = (over: Record<string, unknown> = {}) => ({
  name: "Cultist", hp: 30, max_hp: 72, block: 9, ...over,
});

const hpwrap = (c: HTMLElement) => c.querySelector(".gs-hpwrap")!.outerHTML;

describe("the health bar's markup", () => {
  it("is unchanged for a blocked enemy", () => {
    const { container } = render(<Combatant c={enemy()} isEnemy dataIndex={0} />);
    expect(hpwrap(container)).toMatchSnapshot();
  });

  it("is unchanged with no block, a Doom band and a fixed bar width", () => {
    const { container } = render(
      <Combatant c={enemy({ block: 0, powers: [{ name: "Doom", id: "DOOM", amount: 12 }] })}
        isEnemy dataIndex={1} hpWidth={137} />,
    );
    expect(hpwrap(container)).toMatchSnapshot();
  });

  it("is unchanged for the player mid block-pop and shatter", () => {
    const { container } = render(
      <Combatant c={{ name: "Ironclad", hp: 58, max_hp: 80, block: 5 }} isEnemy={false} anchor="player"
        effects={{ blockPop: true, blockShatter: true, nonce: 3 }} />,
    );
    expect(hpwrap(container)).toMatchSnapshot();
  });

  it("drops the shield in only on a gain from zero", () => {
    const p = (block: number) => ({ name: "Ironclad", hp: 58, max_hp: 80, block });
    const { container, rerender } = render(<Combatant c={p(0)} isEnemy={false} anchor="player" />);
    rerender(<Combatant c={p(5)} isEnemy={false} anchor="player" effects={{ blockPop: true, nonce: 1 }} />);
    expect(container.querySelector(".gs-block")!.className).toContain("gs-block-pop");
    rerender(<Combatant c={p(8)} isEnemy={false} anchor="player" effects={{ blockPop: true, nonce: 2 }} />);
    expect(container.querySelector(".gs-block")!.className).not.toContain("gs-block-pop");
  });

  it("keeps the block badge mounted at block 0", () => {
    const { container } = render(<Combatant c={enemy({ block: 0 })} isEnemy dataIndex={0} />);
    const badge = container.querySelector<HTMLElement>("[data-block-badge]");
    expect(badge).not.toBeNull();
    expect(badge!.title).toBe("block 0");
    expect(badge!.style.opacity).toBe("0");
  });
});
