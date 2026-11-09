import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import type { ReactElement } from "react";
import Board from "../Board";
import GameHud from "../GameHud";
import { HintsFixture } from "../../hints/HintsContext";
import { MarksFixture } from "../../advice/AdviceContext";
import { buildMarks, scoreOptions } from "../../advice/marks";
import type { Step } from "../../api";

function combatStep(): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: {
      decision: "combat_play", round: 1, energy: 3, max_energy: 3,
      draw_pile_count: 5, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 },
      enemies: [{ index: 0, name: "Cultist", hp: 50, max_hp: 50, block: 0, image: null }],
      hand: [
        { index: 0, id: "STRIKE", name: "Strike", cost: 1, type: "Attack",
          description: "", can_play: true, target_type: "AnyEnemy", image: null },
        { index: 1, id: "DEFEND", name: "Defend", cost: 1, type: "Skill",
          description: "", can_play: true, target_type: "None", image: null },
      ],
    },
  };
}

/** Board's MCTS advice (hint / play best / search / score badges) is gated on
 *  the global hints switch, which defaults OFF. Tests about that advice opt in. */
function withHints(ui: ReactElement) {
  return render(<HintsFixture on>{ui}</HintsFixture>);
}

/** The engine's own numbers ("engine numbers" on), no marks. */
function withNumbers(ui: ReactElement) {
  return <MarksFixture marks={[]} showNumbers>{ui}</MarksFixture>;
}

describe("Board (interactive)", () => {
  it("auto-targets the only enemy for a single-target card", () => {
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    fireEvent.click(screen.getByText("Strike"));
    expect(onPlayCard).toHaveBeenCalledWith(0, 0);
  });

  it("plays an untargeted card immediately", () => {
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    fireEvent.click(screen.getByText("Defend"));
    expect(onPlayCard).toHaveBeenCalledWith(1, undefined);
  });

  it("fires onEndTurn from the End Turn button", () => {
    const onEndTurn = vi.fn();
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={onEndTurn} />);
    fireEvent.click(screen.getByRole("button", { name: /end turn/i }));
    expect(onEndTurn).toHaveBeenCalled();
  });

  it("is read-only without handlers (End Turn plate rendered but inert)", () => {
    render(<Board step={combatStep()} />);
    // The in-scene End Turn plate stays visible in replay (game furniture) but
    // is disabled — replay must not be able to end a turn.
    expect(screen.getByRole("button", { name: /end turn/i })).toBeDisabled();
  });

  function potionStep(): Step {
    const s = combatStep();
    s.state.player.potions = [
      { index: 0, name: "Weak Potion" },
      { index: 1, name: "Fire Potion" },
    ];
    return s;
  }

  // The belt is the HUD band's, as in the game (the scene carries no copy).
  it("renders a tile per potion and fires onUsePotion with the slot", () => {
    const onUsePotion = vi.fn();
    render(<GameHud state={potionStep().state} onUsePotion={onUsePotion} />);
    // One "use" button per potion tile, in slot order [Weak, Fire].
    const useButtons = screen.getAllByRole("button", { name: /^use$/i });
    fireEvent.click(useButtons[1]); // Fire Potion = slot 1
    expect(onUsePotion).toHaveBeenCalledWith(1); // slot 1, no target
  });

  it("fires onDiscardPotion with the slot from the discard button", () => {
    const onDiscardPotion = vi.fn();
    render(<GameHud state={potionStep().state} onUsePotion={() => {}} onDiscardPotion={onDiscardPotion} />);
    fireEvent.click(screen.getByRole("button", { name: /discard Fire Potion/i }));
    expect(onDiscardPotion).toHaveBeenCalledWith(1);
  });

  it("shows potions read-only when no handlers (no use/discard buttons)", () => {
    render(<GameHud state={potionStep().state} />);
    expect(screen.getByText("Fire Potion")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^use$/i })).toBeNull();
  });

  it("rings and numbers the potion the AI would drink, on the band's belt", () => {
    const marks = buildMarks([
      { key: "potion:1", label: "Fire Potion", isBest: true, inBand: true, share: 0.7, q: 0.6, pickRank: 0, targetIndex: null },
      { key: "end_turn", label: "end turn", isBest: false, inBand: false, share: 0.3, q: 0.4, pickRank: 1, targetIndex: null },
    ]);
    render(<MarksFixture marks={marks}><GameHud state={potionStep().state} onUsePotion={() => {}} /></MarksFixture>);
    const ringed = screen.getByTestId("potion-mark");
    expect(ringed).toHaveClass("gs-mark-best");
    expect(ringed).toHaveTextContent("Fire Potion");
    expect(ringed.querySelector(".gs-step-tag")).toHaveTextContent("1");
  });

  function press(key: string) {
    act(() => { window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true })); });
  }

  it("plays the Nth hand card on its number key (auto-target single enemy)", () => {
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    press("1");
    expect(onPlayCard).toHaveBeenCalledWith(0, 0); // Strike (AnyEnemy) auto-targets the only enemy
  });

  it("plays an untargeted card by number immediately", () => {
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    press("2");
    expect(onPlayCard).toHaveBeenCalledWith(1, undefined); // Defend (None)
  });

  it("'e' ends the turn and 'h' requests a hint", () => {
    const onEndTurn = vi.fn();
    const onHint = vi.fn();
    withHints(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={onEndTurn} onHint={onHint} />);
    press("e");
    expect(onEndTurn).toHaveBeenCalled();
    press("h");
    expect(onHint).toHaveBeenCalled();
  });

  it("'p' peeks, in either style (it is press-first)", () => {
    const onPeek = vi.fn();
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} onPeek={onPeek} />);
    press("p");
    expect(onPeek).toHaveBeenCalledTimes(1);
  });

  it("does not react to keys when read-only (no handlers)", () => {
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} />);
    press("1");
    expect(onPlayCard).not.toHaveBeenCalled(); // nothing wired; just assert no throw
  });

  function twoEnemyStep(): Step {
    const s = combatStep();
    s.state.enemies = [
      { index: 0, name: "Cultist", hp: 50, max_hp: 50, block: 0, image: null },
      { index: 1, name: "Looter", hp: 40, max_hp: 40, block: 0, image: null },
    ];
    return s;
  }

  it("two-step targets: number selects card, then number selects enemy", () => {
    const onPlayCard = vi.fn();
    render(<Board step={twoEnemyStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />);
    press("1");                       // Strike needs a target, 2 enemies -> pending
    expect(onPlayCard).not.toHaveBeenCalled();
    expect(screen.getByText(/select a target/i)).toBeInTheDocument();
    press("2");                       // pick the 2nd enemy (index 1)
    expect(onPlayCard).toHaveBeenCalledWith(0, 1);
  });

  it("draws no arrow from the player when an attack lands", () => {
    // A resolved attack used to hang a SECOND arrow off the character while the
    // card's own arrow was still on screen. The game has exactly one arrow and
    // it never starts at a creature — `attack` is an animation cue, not a line.
    const { container } = render(
      <Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} attack={{ target: 0, nonce: 1 }} />,
    );
    expect(container.querySelector('[data-testid="attack-arrow"]')).toBeNull();
    expect(container.querySelectorAll("svg [data-testid$='-arrow']")).toHaveLength(0);
  });

  function withScores(over: Partial<import("../../api").Score>): import("../../api").Score {
    return {
      // A searched row has visits; a zero-visit row is a forced move the
      // run AI did not search, and carries no win-prob to overlay (M03).
      stable_idx: 0, action: "Strike", win_prob: 0.5, q: 0, visits: 100,
      is_best: false, q_saturated: false, low_confidence: false, target_index: null, ...over,
    };
  }

  it("shows per-card win-prob badges and the end-turn score", () => {
    const scores = [withScores({ stable_idx: 0, win_prob: 0.62, is_best: true }), withScores({ stable_idx: -1, win_prob: 0.4 })];
    withHints(withNumbers(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} scores={scores} />));
    expect(screen.getByText("62%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /end turn/i })).toHaveTextContent("0.40");
  });

  it("keeps the engine's numbers off the board unless engine numbers is on", () => {
    const scores = [withScores({ stable_idx: 0, win_prob: 0.62, is_best: true }), withScores({ stable_idx: -1, win_prob: 0.4 })];
    withHints(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} scores={scores} />);
    expect(screen.queryByText("62%")).toBeNull();
    expect(screen.getByRole("button", { name: /end turn/i })).not.toHaveTextContent("0.40");
  });

  it("rings and numbers the move to make, and puts the reticle on its target", () => {
    const scores = [
      withScores({ stable_idx: 0, win_prob: 0.62, is_best: true, visits: 80, target_index: 0, q: 0.3 }),
      withScores({ stable_idx: -1, win_prob: 0.4, visits: 20, q: -0.2 }),
    ];
    const { container } = withHints(
      <MarksFixture marks={buildMarks(scoreOptions(scores))}>
        <Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} scores={scores} />
      </MarksFixture>);
    const card = container.querySelector('[data-card-index="0"] .gs-card');
    expect(card).toHaveClass("gs-mark-best");
    expect(container.querySelector('[data-card-index="0"]')).toHaveTextContent("1");
    expect(container.querySelector(".gs-handslot.gs-pulled")).not.toBeNull();
    // End turn at 20% of the search is a real alternative: ringed, unnumbered.
    expect(screen.getByRole("button", { name: /end turn/i })).toHaveClass("gs-mark-alt");
    expect(screen.getByTestId("advice-reticle")).toBeInTheDocument();
  });

  it("shows no end-turn score for a forced, unsearched move", () => {
    const scores = [withScores({ stable_idx: -1, win_prob: 0, visits: 0, is_best: true })];
    withHints(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} scores={scores} />);
    expect(screen.getByRole("button", { name: /end turn/i })).not.toHaveTextContent("0.00");
  });

  it("'g' plays the recommended move and 'a' autoplays", () => {
    const onPlayRecommended = vi.fn();
    const onAutoplay = vi.fn();
    withHints(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}}
      onPlayRecommended={onPlayRecommended} onAutoplay={onAutoplay} />);
    press("g");
    expect(onPlayRecommended).toHaveBeenCalled();
    press("a");
    expect(onAutoplay).toHaveBeenCalled();
  });

  it("keeps every PRESS-FIRST control available while hints are off", () => {
    // The switch governs advice that appears unasked, not access to the tools.
    // Nothing below reveals anything until it is clicked, so all of it stays.
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}}
      onHint={() => {}} onPlayRecommended={() => {}} onAutoplay={() => {}}
      onAutoplayToEnd={() => {}} onSearch={() => {}} />);
    expect(screen.getByRole("button", { name: /play best/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^search/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /autoplay/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /play to end/i })).toBeInTheDocument();
  });

  it("honours the H / G / S shortcuts while hints are off — pressing a key IS asking", () => {
    const onHint = vi.fn();
    const onPlayRecommended = vi.fn();
    const onSearch = vi.fn();
    const onAutoplay = vi.fn();
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}}
      onHint={onHint} onPlayRecommended={onPlayRecommended} onSearch={onSearch}
      onAutoplay={onAutoplay} />);
    press("h");
    press("g");
    press("s");
    press("a");
    expect(onHint).toHaveBeenCalled();
    expect(onPlayRecommended).toHaveBeenCalled();
    expect(onSearch).toHaveBeenCalled();
    expect(onAutoplay).toHaveBeenCalled();
  });

  it("hides per-card win-prob badges in live play while hints are off and none was asked for", () => {
    // These are the UNPROMPTED half: they sit next to every option without any
    // click, so they stay behind the switch.
    const scores = [withScores({ stable_idx: 0, win_prob: 0.62, is_best: true }), withScores({ stable_idx: -1, win_prob: 0.4 })];
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} scores={scores} />);
    expect(screen.queryByText("62%")).toBeNull();
    expect(screen.getByRole("button", { name: /end turn/i })).not.toHaveTextContent("0.40");
  });

  it("shows the badges with hints off once a hint was asked for at this position", () => {
    const scores = [withScores({ stable_idx: 0, win_prob: 0.62, is_best: true }), withScores({ stable_idx: -1, win_prob: 0.4 })];
    render(withNumbers(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}}
      scores={scores} showScores />));
    expect(screen.getByText("62%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /end turn/i })).toHaveTextContent("0.40");
  });

  it("still shows recorded scores in replay, which is review rather than play", () => {
    const scores = [withScores({ stable_idx: 0, win_prob: 0.62, is_best: true })];
    render(withNumbers(<Board step={combatStep()} scores={scores} />));
    expect(screen.getByText("62%")).toBeInTheDocument();
  });
});