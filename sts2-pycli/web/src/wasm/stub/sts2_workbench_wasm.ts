/**
 * Stand-in for the wasm-bindgen package in NON-static builds.
 *
 * `vite.config.ts` aliases `sts2-workbench-wasm` here unless
 * `VITE_WORKBENCH_MODE=static`, so the engine worker still bundles (Vite
 * resolves `new Worker(new URL(...))` statically, whether or not the page ever
 * spawns it) without the generated `src/wasm/pkg/` being present. Nothing
 * reaches this at runtime: the transport only spawns the worker in static
 * mode. The real signatures live in `src/wasm/pkg.d.ts`, which is what the
 * worker is type-checked against; this file only has to load and throw.
 */
const MISSING =
  "sts2-workbench-wasm is not available in this build (built without " +
  "VITE_WORKBENCH_MODE=static, or `just wasm-build` was not run)";

const missing = (): never => {
  throw new Error(MISSING);
};

export function version(): string {
  return "stub";
}

export class Workbench {
  constructor() {
    missing();
  }
  call = missing;
  call_stream = missing;
  build_save = missing;
  restore_save = missing;
  export_persist = missing;
  import_persist = missing;
  has_live_run = () => false;
  load_ppo = missing;
  set_ppo_error = missing;
  has_ppo = () => false;
  load_e2e = missing;
  set_e2e_error = missing;
  has_e2e = () => false;
  take_run_log = () => "null";
  load_run = missing;
  free = () => {};
  static save_header = missing;
  static ppo_champion_file = missing;
  static e2e_file = () => "";
  static mcts_runs = missing;
  static aggregate_runs = missing;
}

export default async function init(): Promise<never> {
  return missing();
}
