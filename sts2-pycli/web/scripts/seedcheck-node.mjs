#!/usr/bin/env node
// The seed checker's acceptance check, run on the wasm the static build ships
// (in node, no browser): the greedy check attempt must reproduce the Python
// d1250 reference, and the sampled win rate must fall inside the reference's
// Wilson 95% interval (design: docs/superpowers/specs/2026-11-08-workbench-seed-checker-design.md,
// acceptance 3 and 5).
//
//   node scripts/seedcheck-node.mjs [--runs 200] [--threads 8] [--seed 1SNVTCEXKR0Q]
//
// Needs `just wasm-build` (src/wasm/pkg) and the net in ../data/e2e/.
// The reference (Python `ppo.e2e.solve` on ema0999_d1250.pt, 2026-11-08). STALE: the
// shipped net is now ironclad_e2e_d2500_avg5_ema0999 (stronger), so the sampled
// interval below is d1250's and needs re-measuring on the new net:
// 1SNVTCEXKR0Q greedy won, sampled 15/200 = 7.5%, Wilson 95% [4.6%, 12.0%];
// 1SNVTCEXKR0G greedy won.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Worker, isMainThread, parentPort } from "node:worker_threads";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = path.join(here, "..", "src", "wasm", "pkg");

async function makeWorkbench() {
  const mod = await import(pathToFileURL(path.join(pkg, "sts2_workbench_wasm.js")).href);
  mod.initSync({ module: fs.readFileSync(path.join(pkg, "sts2_workbench_wasm_bg.wasm")) });
  const wb = new mod.Workbench(JSON.stringify({ assets_base: "" }));
  const net = path.join(here, "..", "..", "data", "e2e", mod.Workbench.e2e_file());
  const r = JSON.parse(wb.load_e2e(new Uint8Array(fs.readFileSync(net))));
  if (r.status !== 200) throw new Error(`load_e2e: ${JSON.stringify(r.body)}`);
  return wb;
}

function attempt(wb, seed, seq, k) {
  const r = JSON.parse(wb.call("POST", "/api/seedcheck/attempt", JSON.stringify({ seed, seq, k })));
  if (r.status !== 200) throw new Error(`attempt ${seq} ${k}: ${JSON.stringify(r.body)}`);
  return r.body;
}

if (!isMainThread) {
  const wb = await makeWorkbench();
  parentPort.on("message", ({ seed, seq, k }) => {
    try {
      parentPort.postMessage({ ok: true, r: attempt(wb, seed, seq, k) });
    } catch (e) {
      parentPort.postMessage({ ok: false, k, error: String(e) });
    }
  });
  parentPort.postMessage({ ready: true });
} else {
  const arg = (name, dflt) => {
    const i = process.argv.indexOf(`--${name}`);
    return i > 0 ? process.argv[i + 1] : dflt;
  };
  const runs = Number(arg("runs", "200"));
  const threads = Number(arg("threads", String(Math.max(1, Math.min(8, os.cpus().length - 1)))));
  const seed = arg("seed", "1SNVTCEXKR0Q");
  let failed = false;

  // Greedy check attempts (acceptance 3), on the main thread.
  const wb = await makeWorkbench();
  for (const s of ["1SNVTCEXKR0Q", "1SNVTCEXKR0G"]) {
    const t0 = performance.now();
    const r = attempt(wb, s, "check", 0);
    const ok = r.won === true;
    failed ||= !ok;
    console.log(
      `${ok ? "PASS" : "FAIL"} greedy ${s}: ${r.outcome} floor ${r.floor} hp ${r.hp} ` +
        `(${r.decisions} decisions, ${((performance.now() - t0) / 1000).toFixed(1)} s); reference: won`,
    );
  }

  // Sampled rate (acceptance 5), across worker threads.
  const t0 = performance.now();
  const results = [];
  let next = 0;
  await new Promise((resolve, reject) => {
    let live = 0;
    for (let i = 0; i < threads; i++) {
      const w = new Worker(fileURLToPath(import.meta.url));
      live++;
      const feed = () => {
        if (next >= runs) {
          w.terminate();
          if (--live === 0) resolve();
          return;
        }
        w.postMessage({ seed, seq: "rate", k: next++ });
      };
      w.on("message", (m) => {
        if (!m.ready) {
          results.push(m.ok ? m.r : { k: m.k, won: false, outcome: "error", error: m.error });
          if (results.length % 20 === 0) {
            const wins = results.filter((r) => r.won).length;
            console.log(`  ${results.length}/${runs} played, ${wins} won, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
          }
        }
        feed();
      });
      w.on("error", reject);
    }
  });
  const errors = results.filter((r) => r.outcome !== "won" && r.outcome !== "lost");
  const n = results.length - errors.length;
  const wins = results.filter((r) => r.won).length;
  const p = wins / n;
  const LO = 0.046;
  const HI = 0.12;
  const ok = errors.length === 0 && p >= LO && p <= HI;
  failed ||= !ok;
  console.log(
    `${ok ? "PASS" : "FAIL"} rate ${seed}: ${wins}/${n} = ${(100 * p).toFixed(1)}% ` +
      `(${errors.length} errors, ${((performance.now() - t0) / 1000).toFixed(0)} s on ${threads} threads); ` +
      `reference interval [${100 * LO}%, ${100 * HI}%]`,
  );
  for (const e of errors.slice(0, 5)) console.log(`  error k=${e.k}: ${e.error ?? e.outcome}`);
  process.exit(failed ? 1 : 0);
}
