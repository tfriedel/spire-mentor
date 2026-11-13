import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Combatant from "../Combatant";

/**
 * `noChrome`: the phone-portrait player figure is the body alone — its HP bar,
 * block, powers and name are the band's (`PlayerBand`), beside it.
 */
const player = { name: "The Ironclad", hp: 58, max_hp: 72, block: 12,
  powers: [{ name: "Strength", amount: 2, image: null }] };

describe("Combatant noChrome", () => {
  it("draws no HP bar, nameplate or powers", () => {
    const { container } = render(<Combatant c={player} isEnemy={false} anchor="player" noChrome />);
    expect(container.querySelector('[data-combatant="player"]')).not.toBeNull();
    expect(container.querySelector(".gs-hpwrap")).toBeNull();
    expect(container.querySelector(".gs-nameplate")).toBeNull();
    expect(container.querySelector(".gs-powers")).toBeNull();
  });

  it("keeps all three without it", () => {
    const { container } = render(<Combatant c={player} isEnemy={false} anchor="player" />);
    expect(container.querySelector(".gs-hpwrap")).not.toBeNull();
    expect(container.querySelector(".gs-nameplate")).not.toBeNull();
    expect(container.querySelector(".gs-powers")).not.toBeNull();
  });
});
