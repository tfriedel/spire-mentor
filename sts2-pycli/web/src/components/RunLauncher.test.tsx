import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import RunLauncher from "./RunLauncher";
import { SEED_ALPHABET, SEED_LENGTH } from "../seed";

function renderLauncher(over: Partial<React.ComponentProps<typeof RunLauncher>> = {}) {
  const props: React.ComponentProps<typeof RunLauncher> = {
    character: "Ironclad", onCharacter: vi.fn(),
    seed: "Y3X2YZSBQJ", onSeed: vi.fn(),
    ascension: 0, onAscension: vi.fn(),
    act1: "roll", onAct1: vi.fn(),
    backend: "rust", onBackend: vi.fn(), backends: ["rust", "csharp"],
    onPlay: vi.fn(), onWatch: vi.fn(), busy: false, error: null,
    offCombatBrain: "heuristic", onOffCombatBrain: vi.fn(),
    combat: {
      preset: "benchmark", monsterSims: 1000, bossSims: 4000, threads: 1,
      onPreset: vi.fn(), onMonsterSims: vi.fn(), onBossSims: vi.fn(),
    },
    ...over,
  };
  return { ...render(<RunLauncher {...props} />), props };
}

describe("RunLauncher", () => {
  it("rolls a game-style seed when it opens with none", () => {
    const { props } = renderLauncher({ seed: "" });
    expect(props.onSeed).toHaveBeenCalledTimes(1);
    const rolled = (props.onSeed as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(rolled).toHaveLength(SEED_LENGTH);
    for (const ch of rolled) expect(SEED_ALPHABET).toContain(ch);
  });

  it("keeps a seed it was given, and re-rolls on demand", () => {
    const { props } = renderLauncher();
    expect(props.onSeed).not.toHaveBeenCalled();
    expect(screen.getByLabelText("seed")).toHaveValue("Y3X2YZSBQJ");
    fireEvent.click(screen.getByRole("button", { name: "new seed" }));
    expect(props.onSeed).toHaveBeenCalledTimes(1);
    expect((props.onSeed as ReturnType<typeof vi.fn>).mock.calls[0][0]).not.toBe("Y3X2YZSBQJ");
  });

  it("tells the player how a typed seed will actually run", () => {
    renderLauncher({ seed: "foo1oi" });
    expect(screen.getByText("F00101")).toBeInTheDocument();
  });

  it("offers the five characters as a radio group and reports the pick", () => {
    const { props } = renderLauncher();
    const radios = screen.getAllByRole("radio", { name: /Ironclad|Silent|Defect|Regent|Necrobinder/ });
    expect(radios).toHaveLength(5);
    expect(screen.getByRole("radio", { name: "Ironclad" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("radio", { name: "Necrobinder" }));
    expect(props.onCharacter).toHaveBeenCalledWith("Necrobinder");
  });

  it("names the ascension level and stacks the modifiers below it", () => {
    const { props } = renderLauncher({ ascension: 3 });
    expect(screen.getByText("Ascension 3 · Poverty")).toBeInTheDocument();
    // A3 stacks A1..A3: three modifier lines, the top one first.
    const mods = screen.getAllByRole("listitem");
    expect(mods).toHaveLength(3);
    expect(mods[0]).toHaveTextContent(/25% less/);
    expect(screen.getByRole("radio", { name: "ascension 3" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("radio", { name: "ascension 10" }));
    expect(props.onAscension).toHaveBeenCalledWith(10);
    // The game caps at A10 — no rung past it.
    expect(screen.queryByRole("radio", { name: "ascension 11" })).toBeNull();
  });

  it("explains both launchers and fires the right one", () => {
    const { props } = renderLauncher();
    const play = screen.getByRole("button", { name: "Play yourself" });
    const watch = screen.getByRole("button", { name: "Watch the AI play" });
    expect(play).toHaveAccessibleDescription(/You make every decision/);
    expect(watch).toHaveAccessibleDescription(/plays the whole run out/);
    fireEvent.click(play);
    expect(props.onPlay).toHaveBeenCalledTimes(1);
    fireEvent.click(watch);
    expect(props.onWatch).toHaveBeenCalledTimes(1);
    // The AI's knobs live with the AI launcher.
    expect(screen.getByRole("combobox", { name: "brain" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "combat" })).toBeInTheDocument();
  });

  it("hides the AI launcher and its knobs when run mode is off", () => {
    renderLauncher({ onWatch: undefined });
    expect(screen.queryByRole("button", { name: "Watch the AI play" })).toBeNull();
    expect(screen.queryByRole("combobox", { name: "brain" })).toBeNull();
    expect(screen.getByRole("button", { name: "Play yourself" })).toBeInTheDocument();
  });

  it("disables everything while a start is in flight and shows the error", () => {
    renderLauncher({ busy: true, error: "start failed: 500" });
    expect(screen.getByRole("button", { name: "Play yourself" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("start failed: 500");
  });

  it("hides the game-version select when the server runs one build", () => {
    renderLauncher();
    expect(screen.queryByLabelText("version")).toBeNull();
    renderLauncher({ gameVersions: ["v0.107.1"], gameVersion: "v0.107.1", onGameVersion: vi.fn() });
    expect(screen.queryByLabelText("version")).toBeNull();
  });

  it("offers the game builds when there are two, and reports the pick", () => {
    const onGameVersion = vi.fn();
    renderLauncher({ gameVersions: ["v0.107.1", "v0.111.0"], gameVersion: "v0.107.1", onGameVersion });
    const sel = screen.getByLabelText("version");
    expect(sel).toHaveValue("v0.107.1");
    expect(screen.getByRole("option", { name: "game v0.107.1 (stable)" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "v0.111.0 (beta)" })).toBeInTheDocument();
    fireEvent.change(sel, { target: { value: "v0.111.0" } });
    expect(onGameVersion).toHaveBeenCalledWith("v0.111.0");
  });

  it("pins the version to the default under the C# engine, which plays only that build", () => {
    renderLauncher({
      backend: "csharp", backends: ["rust", "csharp"],
      gameVersions: ["v0.107.1", "v0.111.0"], gameVersion: "v0.111.0", onGameVersion: vi.fn(),
    });
    const sel = screen.getByLabelText("version");
    expect(sel).toBeDisabled();
    expect(sel).toHaveValue("v0.107.1");
  });

  it("keeps the engine and act-1 selects reachable by label", () => {
    const { props } = renderLauncher();
    fireEvent.change(screen.getByLabelText("engine"), { target: { value: "csharp" } });
    expect(props.onBackend).toHaveBeenCalledWith("csharp");
    fireEvent.change(screen.getByLabelText("act 1"), { target: { value: "underdocks" } });
    expect(props.onAct1).toHaveBeenCalledWith("underdocks");
  });

  it("offers the e2e net as the advisor of a hand-played Ironclad v0.111.0 run only", () => {
    const { unmount } = renderLauncher({ e2eAvailable: true, onAdvisor: vi.fn(), gameVersion: "v0.107.1" });
    expect(screen.queryByRole("combobox", { name: "advisor" })).toBeNull();
    unmount();
    const silent = renderLauncher({ e2eAvailable: true, onAdvisor: vi.fn(), character: "Silent", gameVersion: "v0.111.0" });
    expect(screen.queryByRole("combobox", { name: "advisor" })).toBeNull();
    silent.unmount();
    const { props } = renderLauncher({ e2eAvailable: true, onAdvisor: vi.fn(), gameVersion: "v0.111.0" });
    const pick = screen.getByRole("combobox", { name: "advisor" });
    expect(pick).toHaveValue("search");
    fireEvent.change(pick, { target: { value: "e2e" } });
    expect(props.onAdvisor).toHaveBeenCalledWith("e2e");
  });

  it("offers the e2e student only where the engine has it, and drops the sim knobs for it", () => {
    const { unmount } = renderLauncher();
    expect(screen.queryByRole("option", { name: /e2e/ })).toBeNull();
    unmount();
    const { props, unmount: un2 } = renderLauncher({ e2eAvailable: true });
    fireEvent.change(screen.getByRole("combobox", { name: "brain" }), { target: { value: "e2e" } });
    expect(props.onOffCombatBrain).toHaveBeenCalledWith("e2e");
    un2();
    renderLauncher({ e2eAvailable: true, offCombatBrain: "e2e" });
    expect(screen.getByRole("option", { name: "Instinct (PPO only)" })).toBeInTheDocument();
    expect(screen.getByTestId("e2e-note")).toHaveTextContent("Ironclad · v0.111.0");
    expect(screen.queryByTitle(/Combat search budget/)).toBeNull();
  });
});
