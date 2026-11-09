/** Messages between the seed-check page and its workers (`seedcheck.worker.ts`). */
import type { AttemptResult } from "./stats";

export type Seq = "check" | "rate";

export type ToWorker =
  | { type: "init" }
  | { type: "attempt"; id: number; seed: string; ascension: number; seq: Seq; k: number };

export type FromWorker =
  | { type: "ready" }
  | { type: "result"; id: number; result: AttemptResult }
  /** `fatal`: the engine refused the job itself (a bad seed, no net), so
   *  every attempt would fail the same way. Otherwise an attempt-level error
   *  (an engine panic): that attempt counts as errored. */
  | { type: "error"; id?: number; message: string; fatal?: boolean };

/** The part of `Worker` the pool uses (tests pass fakes). */
export interface PoolWorker {
  postMessage(msg: ToWorker): void;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
  terminate(): void;
}
