import { describe, expect, it } from "vitest";
import { SeedCheckPool, type JobResult } from "../pool";
import type { FromWorker, PoolWorker, ToWorker } from "../protocol";
import type { AttemptResult } from "../stats";

type Behaviour = {
  /** Fail init with this message. */
  initError?: string;
  /** Crash (onerror) on the n-th attempt this worker gets (0-based). */
  crashOn?: number;
  /** Report an engine trap (an attempt-level error) on the n-th attempt. */
  panicOn?: number;
  /** Report the engine refusing the job (a bad seed) on the n-th attempt. */
  fatalOn?: number;
};

const issued: { seq: string; k: number; ascension: number }[] = [];

class FakeWorker implements PoolWorker {
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((ev: ErrorEvent) => void) | null = null;
  private got = 0;
  terminated = false;
  constructor(private b: Behaviour, private winAt: (k: number) => boolean) {}
  postMessage(msg: ToWorker): void {
    queueMicrotask(() => {
      if (msg.type === "init") {
        this.emit(this.b.initError ? { type: "error", message: this.b.initError } : { type: "ready" });
        return;
      }
      const n = this.got++;
      issued.push({ seq: msg.seq, k: msg.k, ascension: msg.ascension });
      if (this.b.crashOn === n) {
        this.onerror?.({ message: "boom" } as ErrorEvent);
        return;
      }
      if (this.b.panicOn === n) {
        this.emit({ type: "error", id: msg.id, message: "unreachable executed" });
        return;
      }
      if (this.b.fatalOn === n) {
        this.emit({ type: "error", id: msg.id, message: "bad seed", fatal: true });
        return;
      }
      const won = this.winAt(msg.k);
      const result: AttemptResult = {
        k: msg.k, seq: msg.seq, temp: msg.k === 0 && msg.seq === "check" ? 0 : 1, won,
        outcome: won ? "won" : "lost", error: null, floor: won ? 46 : 22, act: won ? 3 : 2,
        decisions: 10, classes: [1, 2], hp: won ? 30 : 0, death_room: won ? null : "Terror Eel",
      };
      this.emit({ type: "result", id: msg.id, result });
    });
  }
  private emit(m: FromWorker) {
    this.onmessage?.({ data: m } as MessageEvent<FromWorker>);
  }
  terminate(): void {
    this.terminated = true;
  }
}

function pool(size: number, winAt: (k: number) => boolean, behaviours: Behaviour[] = [], now = () => 0) {
  let i = 0;
  return new SeedCheckPool({ size, spawn: () => new FakeWorker(behaviours[i++] ?? {}, winAt), now });
}

const check = { seed: "S", ascension: 10, seq: "check" as const, startK: 0, stopOnWin: true, maxAttempts: Infinity, capMs: 300_000 };

describe("SeedCheckPool", () => {
  it("stops at the first win", async () => {
    issued.length = 0;
    const p = pool(3, (k) => k === 4);
    const r: JobResult = await p.run(check, () => {});
    expect(r.reason).toBe("won");
    expect(r.results.some((x) => x.won && x.k === 4)).toBe(true);
    expect(Math.max(...issued.map((x) => x.k))).toBeLessThan(4 + 3);
  });

  it("stops at the cap and keeps searching from nextK", async () => {
    issued.length = 0;
    let t = 0;
    const p = pool(2, () => false, [], () => t);
    const progress: number[] = [];
    const r = await p.run(check, (rs) => {
      progress.push(rs.length);
      t += 100_000;
    });
    expect(r.reason).toBe("cap");
    expect(progress.length).toBeGreaterThan(0);
    const next = r.nextK;
    issued.length = 0;
    t = 0;
    const r2 = await p.run({ ...check, startK: next }, () => {
      t += 100_000;
    });
    expect(issued[0].k).toBe(next);
    expect(r2.results.every((x) => x.k >= next)).toBe(true);
  });

  it("stops when asked", async () => {
    const p = pool(2, () => false);
    const r = await p.run(check, () => p.stop());
    expect(r.reason).toBe("stopped");
  });

  it("plays exactly maxAttempts for a win rate", async () => {
    issued.length = 0;
    const p = pool(4, (k) => k % 10 === 0);
    const r = await p.run({ seed: "S", ascension: 10, seq: "rate", startK: 0, stopOnWin: false, maxAttempts: 200, capMs: Infinity }, () => {});
    expect(r.reason).toBe("done");
    expect(r.results).toHaveLength(200);
    expect(new Set(r.results.map((x) => x.k))).toEqual(new Set(Array.from({ length: 200 }, (_, k) => k)));
    expect(issued.every((x) => x.seq === "rate")).toBe(true);
    expect(r.results.filter((x) => x.won)).toHaveLength(20);
  });

  it("drops a worker that fails to start and finishes on the rest", async () => {
    const p = pool(2, () => false, [{ initError: "no net" }]);
    const r = await p.run({ ...check, stopOnWin: false, maxAttempts: 5 }, () => {});
    expect(r.reason).toBe("done");
    expect(r.results).toHaveLength(5);
    expect(r.errors.join(" ")).toContain("no net");
  });

  it("fails when every worker fails", async () => {
    const p = pool(2, () => false, [{ initError: "a" }, { initError: "b" }]);
    const r = await p.run(check, () => {});
    expect(r.reason).toBe("failed");
    expect(r.errors).toEqual(expect.arrayContaining([expect.stringContaining("a"), expect.stringContaining("b")]));
  });

  it("re-issues a crashed worker's attempt", async () => {
    const p = pool(2, () => false, [{ crashOn: 1 }]);
    const r = await p.run({ ...check, stopOnWin: false, maxAttempts: 6 }, () => {});
    expect(r.reason).toBe("done");
    expect(r.results.map((x) => x.k).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(r.errors.join(" ")).toContain("boom");
  });

  it("hands back an attempt a crash left unplayed, and the next job plays it once", async () => {
    issued.length = 0;
    let t = 0;
    const p = pool(2, () => false, [{ crashOn: 0 }], () => t);
    const r = await p.run(check, () => {
      t = 300_000; // the cap: nothing more is issued
    });
    expect(r.reason).toBe("cap");
    expect(r.pending).toEqual([0]);
    expect(r.nextK).toBe(2);
    issued.length = 0;
    t = 0;
    const r2 = await p.run({ ...check, startK: r.nextK, pending: r.pending, stopOnWin: false, maxAttempts: 3 }, () => {});
    expect(issued[0].k).toBe(0);
    const ks = [...r.results, ...r2.results].map((x) => x.k).sort((a, b) => a - b);
    expect(ks).toEqual([0, 1, 2, 3, 4]);
  });

  it("counts an engine trap as an errored attempt and replaces the worker", async () => {
    let spawned = 0;
    const p = new SeedCheckPool({
      size: 2,
      spawn: () => new FakeWorker(spawned++ === 0 ? { panicOn: 1 } : {}, () => false),
      now: () => 0,
    });
    const r = await p.run({ ...check, stopOnWin: false, maxAttempts: 6 }, () => {});
    expect(r.reason).toBe("done");
    expect(r.results.map((x) => x.k).sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(r.results.filter((x) => x.outcome === "error")).toHaveLength(1);
    expect(spawned).toBe(3);
    expect(r.errors.join(" ")).toContain("unreachable");
  });

  it("fails the job when the engine refuses it", async () => {
    const p = pool(2, () => false, [{ fatalOn: 0 }, { fatalOn: 0 }]);
    const r = await p.run(check, () => {});
    expect(r.reason).toBe("failed");
    expect(r.errors.join(" ")).toContain("bad seed");
  });

  it("respawns dead workers for the next job", async () => {
    const p = pool(2, () => false, [{ initError: "a" }, { initError: "b" }]);
    expect((await p.run(check, () => {})).reason).toBe("failed");
    const r = await p.run({ ...check, stopOnWin: false, maxAttempts: 4 }, () => {});
    expect(r.reason).toBe("done");
    expect(r.results).toHaveLength(4);
    expect(r.errors).toEqual([]);
  });

  it("sends the job's ascension with every attempt", async () => {
    issued.length = 0;
    const p = pool(2, () => false);
    await p.run({ ...check, ascension: 3, stopOnWin: false, maxAttempts: 4 }, () => {});
    expect(issued).toHaveLength(4);
    expect(issued.every((x) => x.ascension === 3)).toBe(true);
  });
});
