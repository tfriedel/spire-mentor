import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import SeedCheckPage, { type PoolLike } from "../SeedCheckPage";
import type { Preview, SeedCheckApi } from "../api";
import type { Job, JobResult } from "../pool";
import type { AttemptResult } from "../stats";

const PREVIEW = {
  seed: "1SNVTCEXKR0Q", length_ok: true, neow: ["Arcane Scroll", "Lava Rock", "Neow's Bones"],
  act1: "Overgrowth", bosses: [["Vantom"], ["The Insatiable"], ["Aeonglass", "Queen"]],
};

function attempt(k: number, won: boolean): AttemptResult {
  return {
    k, seq: "check", temp: k === 0 ? 0 : 1, won, outcome: won ? "won" : "lost", error: null,
    floor: won ? 46 : 22, act: 2, decisions: 5, classes: [k], hp: won ? 40 : 0, death_room: won ? null : "Terror Eel",
  };
}

function fakeApi(over: Partial<SeedCheckApi> = {}): SeedCheckApi {
  return {
    preview: vi.fn(async (s: string, ascension: number) => ({
      ...PREVIEW,
      seed: s.toUpperCase(),
      length_ok: s.length === 12,
      ascension,
      bosses: ascension === 10 ? PREVIEW.bosses : [["Vantom"], ["The Insatiable"], ["Aeonglass"]],
    })),
    summary: vi.fn(async () => ({
      won: true, outcome: "won", hp: 40, max_hp: 107, floor: 46, neow_pick: "Lava Rock", boss_relics: ["Claws"],
      events: [], cards_taken: [], shop: [], rests: [], deck: ["Demon Form"], relics: [], death: null,
    })),
    saveLog: vi.fn(async () => ({ path: "x_Ironclad_game_1SNVTCEXKR0Q.jsonl" })),
    ...over,
  };
}

/** A pool that answers each run from `script`, recording the jobs. */
function fakePool(script: (job: Job) => JobResult) {
  const jobs: Job[] = [];
  let release: (() => void) | null = null;
  const pool: PoolLike & { jobs: Job[]; hold: boolean; release: () => void } = {
    jobs, hold: false, alive: 2,
    async run(job, onProgress) {
      jobs.push(job);
      const r = script(job);
      onProgress(r.results);
      if (pool.hold) await new Promise<void>((res) => (release = res));
      return r;
    },
    stop: vi.fn(),
    dispose: vi.fn(),
    release: () => release?.(),
  };
  return pool;
}

async function typeSeed(seed: string) {
  fireEvent.change(screen.getByLabelText("seed"), { target: { value: seed } });
  await screen.findByText(/Arcane Scroll/);
}

describe("SeedCheckPage", () => {
  it("previews the seed and flags a non-12-character one", async () => {
    render(<SeedCheckPage api={fakeApi()} makePool={() => fakePool(() => ({ results: [], reason: "done", nextK: 0, errors: [], pending: [] }))} />);
    await typeSeed("1snvtcexkr0q");
    expect(screen.getByText(/Vantom → The Insatiable → Aeonglass \+ Queen/)).toBeTruthy();
    expect(screen.queryByText(/Game seeds are 12 characters/)).toBeNull();
    fireEvent.change(screen.getByLabelText("seed"), { target: { value: "ABC" } });
    await screen.findByText(/Game seeds are 12 characters; this one is 3/);
  });

  it("shows the preview's error", async () => {
    const api = fakeApi({ preview: vi.fn(async () => { throw new Error("seed character '-' is not in the game's alphabet"); }) });
    render(<SeedCheckPage api={api} makePool={() => fakePool(() => ({ results: [], reason: "done", nextK: 0, errors: [], pending: [] }))} />);
    fireEvent.change(screen.getByLabelText("seed"), { target: { value: "AB-C" } });
    await screen.findByText(/not in the game's alphabet/);
  });

  it("a won check shows the verdict and the replay link", async () => {
    const api = fakeApi();
    const pool = fakePool(() => ({ results: [attempt(0, true)], reason: "won", nextK: 1, errors: [], pending: [] }));
    render(<SeedCheckPage api={api} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    const text = (await screen.findByLabelText("Verdict text")) as HTMLTextAreaElement;
    expect(text.value).toContain("won on attempt 0 (greedy)");
    expect(text.value).toContain("Lava Rock");
    const link = screen.getByText(/Open the winning run/).closest("a")!;
    expect(link.getAttribute("href")).toBe(`/?open=${encodeURIComponent("x_Ironclad_game_1SNVTCEXKR0Q.jsonl")}`);
    expect(api.saveLog).toHaveBeenCalledWith("1SNVTCEXKR0Q", 10, [0], "check", 0);
    expect(pool.jobs[0]).toMatchObject({ seq: "check", ascension: 10, startK: 0, stopOnWin: true, capMs: 300_000 });
  });

  it("disables Check while running", async () => {
    const pool = fakePool(() => ({ results: [attempt(0, false)], reason: "cap", nextK: 1, errors: [], pending: [] }));
    pool.hold = true;
    render(<SeedCheckPage api={fakeApi()} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    await waitFor(() => expect((screen.getByRole("button", { name: "Check" }) as HTMLButtonElement).disabled).toBe(true));
    expect((screen.getByLabelText("seed") as HTMLInputElement).disabled).toBe(true);
    pool.release();
    await screen.findByLabelText("Verdict text");
  });

  it("no win: shows the bound and keeps searching from nextK", async () => {
    const pool = fakePool((job) => ({
      results: [attempt(job.startK, false), attempt(job.startK + 1, false)],
      reason: "cap", nextK: job.startK + 2, errors: [], pending: [],
    }));
    render(<SeedCheckPage api={fakeApi()} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    const text = (await screen.findByLabelText("Verdict text")) as HTMLTextAreaElement;
    expect(text.value).toContain("No win in 2 attempts");
    fireEvent.click(screen.getByText(/Keep searching/));
    await waitFor(() => expect(pool.jobs).toHaveLength(2));
    expect(pool.jobs[1].startK).toBe(2);
    await waitFor(() => expect((screen.getByLabelText("Verdict text") as HTMLTextAreaElement).value).toContain("No win in 4 attempts"));
  });

  it("win rate runs 200 sampled attempts", async () => {
    const pool = fakePool(() => ({
      results: Array.from({ length: 200 }, (_, k) => ({ ...attempt(k, k < 15), seq: "rate" as const, temp: 1 })),
      reason: "done", nextK: 200, errors: [], pending: [],
    }));
    render(<SeedCheckPage api={fakeApi()} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByText(/Win rate/));
    const text = (await screen.findByLabelText("Win rate text")) as HTMLTextAreaElement;
    expect(text.value).toContain("won 15 of 200 sampled runs (7.5%");
    expect(pool.jobs[0]).toMatchObject({ seq: "rate", maxAttempts: 200, stopOnWin: false });
  });

  it("keeps searching with the attempts a crash left unplayed, counting none twice", async () => {
    const pool = fakePool((job) =>
      job.startK === 0
        ? { results: [attempt(1, false)], reason: "cap", nextK: 2, errors: [], pending: [0] }
        : {
            results: [attempt(0, false), attempt(1, false), attempt(2, false)],
            reason: "cap", nextK: 3, errors: [], pending: [],
          },
    );
    render(<SeedCheckPage api={fakeApi()} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    await screen.findByText(/Keep searching/);
    fireEvent.click(screen.getByText(/Keep searching/));
    await waitFor(() => expect(pool.jobs).toHaveLength(2));
    expect(pool.jobs[1]).toMatchObject({ startK: 2, pending: [0] });
    await waitFor(() =>
      expect((screen.getByLabelText("Verdict text") as HTMLTextAreaElement).value).toContain("No win in 3 attempts"),
    );
  });

  it("offers no keep searching after a failed check", async () => {
    const pool = fakePool(() => ({ results: [], reason: "failed", nextK: 0, errors: ["no net"], pending: [] }));
    render(<SeedCheckPage api={fakeApi()} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    await screen.findByText(/The check stopped: no net/);
    expect(screen.queryByText(/Keep searching/)).toBeNull();
  });

  it("ignores a preview reply for text that has since changed", async () => {
    let release: (() => void) | null = null;
    const api = fakeApi({
      preview: vi.fn((s: string) =>
        s === "1SNV"
          ? new Promise<Preview>((res) => (release = () => res({ ...PREVIEW, seed: "1SNV", length_ok: false } as Preview)))
          : Promise.reject(new Error("seed character '-' is not in the game's alphabet")),
      ),
    });
    render(<SeedCheckPage api={api} makePool={() => fakePool(() => ({ results: [], reason: "done", nextK: 0, errors: [], pending: [] }))} />);
    fireEvent.change(screen.getByLabelText("seed"), { target: { value: "1SNV" } });
    await waitFor(() => expect(api.preview).toHaveBeenCalledWith("1SNV", 10));
    fireEvent.change(screen.getByLabelText("seed"), { target: { value: "1SNV-" } });
    await screen.findByText(/not in the game's alphabet/);
    release!();
    await new Promise((r) => setTimeout(r, 20));
    expect(screen.queryByText(/Arcane Scroll/)).toBeNull();
    expect((screen.getByRole("button", { name: "Check" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("offers every character and build but only Ironclad v0.111.0 is selectable", async () => {
    render(<SeedCheckPage api={fakeApi()} makePool={() => fakePool(() => ({ results: [], reason: "done", nextK: 0, errors: [], pending: [] }))} />);
    const options = (label: string) =>
      [...(screen.getByLabelText(label) as HTMLSelectElement).options].map((o) => [o.value, o.disabled] as const);
    expect(options("character")).toEqual([
      ["Ironclad", false], ["Silent", true], ["Defect", true], ["Necrobinder", true], ["Regent", true],
    ]);
    expect(options("version")).toEqual([["v0.111.0", false], ["v0.107.1", true]]);
    expect(options("ascension").map(([v]) => v)).toEqual(Array.from({ length: 11 }, (_, a) => String(a)));
    expect((screen.getByLabelText("ascension") as HTMLSelectElement).value).toBe("10");
  });

  it("plays the picked ascension and names it in the verdict", async () => {
    const api = fakeApi();
    const pool = fakePool(() => ({ results: [attempt(0, true)], reason: "won", nextK: 1, errors: [], pending: [] }));
    render(<SeedCheckPage api={api} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.change(screen.getByLabelText("ascension"), { target: { value: "4" } });
    await screen.findByText(/Instinct trained at A10; it plays A4/);
    await waitFor(() => expect(api.preview).toHaveBeenLastCalledWith("1SNVTCEXKR0Q", 4));
    await screen.findByText(/Vantom → The Insatiable → Aeonglass$/);
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    const text = (await screen.findByLabelText("Verdict text")) as HTMLTextAreaElement;
    expect(text.value).toContain("Ironclad A4, v0.111.0");
    expect(pool.jobs[0].ascension).toBe(4);
    expect(api.summary).toHaveBeenCalledWith("1SNVTCEXKR0Q", 4, [0]);
    expect(api.saveLog).toHaveBeenCalledWith("1SNVTCEXKR0Q", 4, [0], "check", 0);
  });

  it("follows the best attempt's route when nothing won", async () => {
    const api = fakeApi();
    const pool = fakePool(() => ({
      results: [attempt(0, false), { ...attempt(1, false), floor: 30 }],
      reason: "cap", nextK: 2, errors: [], pending: [],
    }));
    render(<SeedCheckPage api={api} makePool={() => pool} />);
    await typeSeed("1SNVTCEXKR0Q");
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    await screen.findByLabelText("Verdict text");
    expect(api.summary).toHaveBeenCalledWith("1SNVTCEXKR0Q", 10, [1]);
  });
});
