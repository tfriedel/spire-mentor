/**
 * Engine Web Worker: hosts the Rust workbench compiled to WASM and answers the
 * page's `/api/*` requests (see `src/transport/`).
 *
 * Wire protocol: see `../transport/protocol.ts`.
 *
 * The wasm is single-threaded, so requests are processed strictly in order —
 * the client already serialises them, and the queue here keeps that true even
 * if a second client ever appears. Any thrown error becomes a 500 reply.
 *
 * A SimLab sweep (GH #707) is a job the wasm cannot run in the background, so
 * once one is submitted the worker pumps it: a tick plays trials for ~50 ms,
 * then the loop yields to the message queue (so a status poll or any other
 * request queued meanwhile is answered) and enqueues the next tick, until
 * nothing is running. A stop request lands within one trial.
 */
import init, { Workbench } from "sts2-workbench-wasm";
// The static build resolves this to the gzipped engine (`.wasm.gz`,
// vite.config.ts); engineResponse inflates it. `vite dev`: the plain wasm.
import wasmUrl from "sts2-workbench-wasm/sts2_workbench_wasm_bg.wasm?url";
import { createEngine, type Engine } from "./engineRoutes";
import { engineResponse } from "./wasmLoad";
import { openKv } from "./idb";
import type { RequestMessage, WorkerInbound, WorkerOutbound } from "../transport/protocol";


const post = (m: WorkerOutbound) => self.postMessage(m);

let engine: Promise<Engine> | null = null;
let chain: Promise<void> = Promise.resolve();

async function boot(assetsBase: string): Promise<Engine> {
  await init({ module_or_path: engineResponse(wasmUrl) });
  const wb = new Workbench(JSON.stringify({ assets_base: assetsBase.replace(/\/$/, "") }));
  return createEngine({
    wb,
    saveHeader: (payload) => Workbench.save_header(payload),
    mctsRuns: (logs) => Workbench.mcts_runs(logs),
    aggregateRuns: (logs) => Workbench.aggregate_runs(logs),
    championFile: (character) => Workbench.ppo_champion_file(character),
    // Same-origin: the static build ships the champions under /ppo/ and the
    // service worker keeps them cache-first, so this is one download per
    // character per device.
    fetchChampion: async (file) => {
      const res = await fetch(`/ppo/${file}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return new Uint8Array(await res.arrayBuffer());
    },
    // The e2e student (~21 MB): fetched only when the e2e brain is chosen,
    // then kept cache-first by the service worker like the champions.
    e2eFile: Workbench.e2e_file(),
    fetchE2e: async (file) => {
      const res = await fetch(`/e2e/${file}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return new Uint8Array(await res.arrayBuffer());
    },
    kv: openKv(),
  });
}

async function serve(msg: RequestMessage): Promise<void> {
  const { id } = msg;
  try {
    if (!engine) throw new Error("engine worker used before init");
    const e = await engine;
    const reply = await e.handle(
      msg.method,
      msg.route,
      msg.body,
      (done, total) => post({ id, progress: { done, total } }),
      (frame) => post({ id, frame }),
    );
    post({ id, status: reply.status, body: reply.body });
  } catch (err) {
    post({ id, status: 500, body: { detail: String(err) } });
  }
}

/** Let queued messages register on `chain` before the next tick does. */
const yieldToQueue = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

let pumping = false;

/** Keep ticking the sweep registry, interleaved with requests, until idle. */
function pumpSweeps(e: Engine): void {
  if (pumping || !e.hasRunningSweep()) return;
  pumping = true;
  const tick = async (): Promise<void> => {
    let active = false;
    try {
      active = await e.sweepTick();
    } catch {
      // A crashed tick ends the pump; the sweep's own status says why.
    }
    if (!active) {
      pumping = false;
      return;
    }
    await yieldToQueue();
    chain = chain.then(tick);
  };
  chain = chain.then(tick);
}

self.onmessage = (ev: MessageEvent<WorkerInbound>) => {
  const msg = ev.data;
  if ("init" in msg) {
    engine ??= boot(msg.init.assets_base);
    // Surface a failed boot on the first request rather than as an unhandled
    // rejection nobody can see.
    engine.catch(() => {});
    return;
  }
  chain = chain.then(async () => {
    await serve(msg);
    if (msg.route.startsWith("/api/sweeps") && engine) pumpSweeps(await engine);
  });
};
