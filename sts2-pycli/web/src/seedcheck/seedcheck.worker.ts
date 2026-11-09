/**
 * A seed-check worker: its own wasm `Workbench` and Instinct net, answering
 * one attempt at a time (`POST /api/seedcheck/attempt`). The page runs
 * several in parallel (`pool.ts`); the wasm is single-threaded.
 */
import init, { Workbench } from "sts2-workbench-wasm";
import wasmUrl from "sts2-workbench-wasm/sts2_workbench_wasm_bg.wasm?url";
import { engineResponse } from "../worker/wasmLoad";
import type { FromWorker, ToWorker } from "./protocol";
import type { AttemptResult } from "./stats";

const post = (m: FromWorker) => self.postMessage(m);
let wb: Workbench | null = null;

async function boot(): Promise<void> {
  await init({ module_or_path: engineResponse(wasmUrl) });
  const w = new Workbench(JSON.stringify({ assets_base: "" }));
  const file = Workbench.e2e_file();
  const res = await fetch(`/e2e/${file}`);
  if (!res.ok) throw new Error(`Instinct net download failed: ${res.status} ${res.statusText}`);
  const loaded = JSON.parse(w.load_e2e(new Uint8Array(await res.arrayBuffer()))) as {
    status: number;
    body: { detail?: string };
  };
  if (loaded.status !== 200) throw new Error(loaded.body?.detail ?? `load_e2e ${loaded.status}`);
  wb = w;
}

self.onmessage = async (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  if (msg.type === "init") {
    try {
      await boot();
      post({ type: "ready" });
    } catch (e) {
      post({ type: "error", message: String(e instanceof Error ? e.message : e) });
    }
    return;
  }
  const { id, seed, ascension, seq, k } = msg;
  try {
    if (!wb) throw new Error("seed-check worker used before init");
    const r = JSON.parse(wb.call("POST", "/api/seedcheck/attempt", JSON.stringify({ seed, ascension, seq, k }))) as {
      status: number;
      body: unknown;
    };
    if (r.status !== 200) {
      const detail = (r.body as { detail?: string } | null)?.detail;
      post({ type: "error", id, message: detail ?? `attempt ${r.status}`, fatal: r.status === 400 || r.status === 424 });
      return;
    }
    post({ type: "result", id, result: r.body as AttemptResult });
  } catch (e) {
    // A trap (an engine panic) inside the wasm: this instance may be poisoned.
    post({ type: "error", id, message: String(e instanceof Error ? e.message : e) });
  }
};
