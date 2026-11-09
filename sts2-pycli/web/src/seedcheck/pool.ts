/**
 * A pool of seed-check workers playing one job's attempts in parallel.
 *
 * Attempt indices go out in order from `startK`. With `stopOnWin` the pool
 * stops issuing at the first win; it also stops at `capMs`, at `maxAttempts`
 * or on `stop()`, and resolves once every attempt in flight has come back
 * (their results are kept). A worker that fails to start or crashes is
 * dropped, and its in-flight index goes back in the queue; an index left in
 * the queue when the job ends comes back as `pending`, for the next job to
 * play first. An attempt that traps inside the engine (a panic) is recorded
 * as an errored attempt and its worker is replaced. Dead workers are
 * respawned when the next job starts.
 */
import type { FromWorker, PoolWorker, Seq } from "./protocol";
import type { AttemptResult } from "./stats";

export interface Job {
  seed: string;
  /** 0-10; the e2e student trained at 10. */
  ascension: number;
  seq: Seq;
  startK: number;
  /** Indices below `startK` an earlier job left unplayed; played first. */
  pending?: number[];
  stopOnWin: boolean;
  maxAttempts: number;
  capMs: number;
}

export interface JobResult {
  results: AttemptResult[];
  reason: "won" | "cap" | "stopped" | "done" | "failed";
  /** The first index never issued: where "keep searching" resumes. */
  nextK: number;
  /** Indices below `nextK` that were issued but never played (their worker
   *  crashed when no more could be issued): pass them as `Job.pending`. */
  pending: number[];
  errors: string[];
}

interface Slot {
  w: PoolWorker;
  state: "starting" | "idle" | "busy" | "dead";
  inflight: number | null;
}

/** Workers to run: all cores but one, at most 8. */
export function defaultPoolSize(): number {
  const n = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;
  return Math.max(1, Math.min(8, n - 1));
}

export class SeedCheckPool {
  private slots: Slot[] | null = null;
  private stopped = false;
  private nextId = 1;
  private readonly size: number;
  private readonly spawn: () => PoolWorker;
  private readonly now: () => number;
  /** The running job's hooks, set by `run`. */
  private onReady: (() => void) | null = null;
  private onResult: ((slot: Slot, r: AttemptResult) => void) | null = null;
  private onDead: ((slot: Slot, message: string) => void) | null = null;
  private onFatal: ((slot: Slot, message: string) => void) | null = null;
  private onAttemptError: ((slot: Slot, message: string) => void) | null = null;

  constructor(opts: { size: number; spawn: () => PoolWorker; now?: () => number }) {
    this.size = opts.size;
    this.spawn = opts.spawn;
    this.now = opts.now ?? (() => performance.now());
  }

  /** Workers alive (started or starting). */
  get alive(): number {
    return (this.slots ?? []).filter((s) => s.state !== "dead").length;
  }

  stop(): void {
    this.stopped = true;
  }

  dispose(): void {
    for (const s of this.slots ?? []) s.w.terminate();
    this.slots = null;
  }

  private makeSlot(): Slot {
    const slot: Slot = { w: this.spawn(), state: "starting", inflight: null };
    slot.w.onmessage = (ev: MessageEvent<FromWorker>) => {
      const m = ev.data;
      if (m.type === "ready") {
        slot.state = "idle";
        this.onReady?.();
      } else if (m.type === "result") {
        this.onResult?.(slot, m.result);
      } else if (m.id === undefined) {
        this.kill(slot, `worker failed to start: ${m.message}`);
      } else if (m.fatal) {
        // The engine refused the job (a bad seed): it cannot go on.
        this.onFatal?.(slot, m.message);
      } else {
        this.onAttemptError?.(slot, m.message);
      }
    };
    slot.w.onerror = (ev: ErrorEvent) => this.kill(slot, `worker crashed: ${ev.message}`);
    slot.w.postMessage({ type: "init" });
    return slot;
  }

  /** The slots, with every dead one replaced by a fresh worker. */
  private ensureSlots(): Slot[] {
    if (!this.slots) this.slots = Array.from({ length: this.size }, () => this.makeSlot());
    else this.slots = this.slots.map((s) => (s.state === "dead" ? this.makeSlot() : s));
    return this.slots;
  }

  private kill(slot: Slot, message: string): void {
    if (slot.state === "dead") return;
    slot.state = "dead";
    slot.w.terminate();
    this.onDead?.(slot, message);
  }

  run(job: Job, onProgress: (rs: AttemptResult[]) => void): Promise<JobResult> {
    this.stopped = false;
    let slots = this.ensureSlots();
    const t0 = this.now();
    const results: AttemptResult[] = [];
    const errors: string[] = [];
    const requeue: number[] = [...(job.pending ?? [])].sort((a, b) => a - b);
    let counter = job.startK;
    const end = job.startK + job.maxAttempts;
    let won = false;
    let fatal = false;

    return new Promise<JobResult>((resolve) => {
      let done = false;
      const note = (message: string) => {
        if (!errors.includes(message)) errors.push(message);
      };
      const reason = (): JobResult["reason"] => {
        if (won && job.stopOnWin) return "won";
        if (fatal || slots.every((s) => s.state === "dead")) return "failed";
        if (this.stopped) return "stopped";
        if (requeue.length === 0 && counter >= end) return "done";
        return "cap";
      };
      const canIssue = () =>
        !fatal &&
        !this.stopped &&
        !(won && job.stopOnWin) &&
        this.now() - t0 < job.capMs &&
        (requeue.length > 0 || counter < end);
      const finish = () => {
        if (done) return;
        done = true;
        this.onReady = this.onResult = this.onDead = this.onFatal = this.onAttemptError = null;
        resolve({ results, reason: reason(), nextK: counter, pending: [...requeue].sort((a, b) => a - b), errors });
      };
      const pump = () => {
        if (done) return;
        for (const s of slots) {
          if (s.state !== "idle" || !canIssue()) continue;
          const k = requeue.length ? requeue.shift()! : counter++;
          s.state = "busy";
          s.inflight = k;
          s.w.postMessage({
            type: "attempt",
            id: this.nextId++,
            seed: job.seed,
            ascension: job.ascension,
            seq: job.seq,
            k,
          });
        }
        const busy = slots.some((s) => s.state === "busy");
        const starting = slots.some((s) => s.state === "starting");
        if (!busy && (!canIssue() || !starting) && !slots.some((s) => s.state === "idle" && canIssue())) {
          finish();
        }
      };
      const settle = (slot: Slot) => {
        slot.state = "idle";
        slot.inflight = null;
      };
      this.onReady = pump;
      this.onResult = (slot, r) => {
        settle(slot);
        results.push(r);
        if (r.won) won = true;
        onProgress(results);
        pump();
      };
      this.onDead = (slot, message) => {
        if (slot.inflight !== null) requeue.push(slot.inflight);
        slot.inflight = null;
        note(message);
        pump();
      };
      this.onFatal = (slot, message) => {
        if (slot.inflight !== null) requeue.push(slot.inflight);
        settle(slot);
        fatal = true;
        note(message);
        pump();
      };
      this.onAttemptError = (slot, message) => {
        const k = slot.inflight;
        if (k !== null) {
          results.push({
            seq: job.seq, k, temp: k === 0 && job.seq === "check" ? 0 : 1, won: false, outcome: "error",
            error: message, floor: 0, act: 0, decisions: 0, classes: [], hp: 0, death_room: null,
          });
          onProgress(results);
        }
        note(`an attempt failed in the engine: ${message}`);
        // Its wasm instance may be poisoned: replace the worker.
        slot.state = "dead";
        slot.inflight = null;
        slot.w.terminate();
        slots = this.ensureSlots();
        pump();
      };
      pump();
    });
  }
}
