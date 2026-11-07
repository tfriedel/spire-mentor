/**
 * Type contract for the wasm-bindgen module built from
 * `sts2-rs/crates/sts2-workbench-wasm` (`just wasm-build`, `--target web`).
 *
 * The generated package lands in `src/wasm/pkg/` (gitignored). Source imports
 * it through the bare specifier `sts2-workbench-wasm`, which `vite.config.ts`
 * aliases to that directory in static mode and to `src/wasm/stub/` otherwise —
 * so a non-static `npm run build` never needs the generated files, and `tsc -b`
 * types the module from THIS file in both cases. Keep it in step with the
 * crate's `#[wasm_bindgen]` surface.
 */
declare module "sts2-workbench-wasm" {
  /** JSON `{status, body}` — `status` is the HTTP status the Python server
   *  would have answered, `body` its JSON body (`{detail}` on error). */
  export type ReplyJson = string;

  export interface InitOutput {
    readonly memory: WebAssembly.Memory;
  }

  export function version(): string;

  export class Workbench {
    /** `config_json` = `{"assets_base": "<origin or path, no trailing slash>"}`. */
    constructor(config_json: string);
    call(method: string, route: string, body_json: string): ReplyJson;
    /** Streaming advisor routes: `on_frame` gets each NDJSON frame as a JSON
     *  string, the last one terminal; the reply is `{status: 200, body: null}`. */
    call_stream(
      method: string,
      route: string,
      body_json: string,
      on_frame: (frame: string) => void,
    ): ReplyJson;
    build_save(name: string, saved_at: string): ReplyJson;
    restore_save(payload_json: string, now: string): ReplyJson;
    static save_header(payload_json: string): ReplyJson;
    /** Raw JSON value (`null` when no run is live). */
    export_persist(): string;
    import_persist(
      payload_json: string,
      progress: (done: number, total: number) => void,
    ): ReplyJson;
    has_live_run(): boolean;
    /** PPO champion (M3): the ONNX filename to fetch for a character
     *  (`{status, body: "<file>"}`), the loader over its bytes, why a
     *  load was not possible, whether one is loaded. */
    static ppo_champion_file(character: string): ReplyJson;
    /** Run logs (M4): the oldest log not yet collected — a quit's, or the
     *  one a run-replacing route wrote (raw JSON `{filename, text}` or
     *  `null` when none is left), a stored log parsed into the replay
     *  store, and the MCTS Runs listings over `[[filename, text], …]`. */
    take_run_log(): string;
    load_run(path: string, text: string): ReplyJson;
    static mcts_runs(logs_json: string): ReplyJson;
    static aggregate_runs(logs_json: string): ReplyJson;
    load_ppo(character: string, bytes: Uint8Array): ReplyJson;
    set_ppo_error(character: string, reason: string): void;
    has_ppo(character: string): boolean;
    /** The e2e student: the `.e2ew` filename to fetch on demand
     *  (`/e2e/<file>`), the loader over its bytes, why a load was not
     *  possible, whether it is loaded. */
    static e2e_file(): string;
    load_e2e(bytes: Uint8Array): ReplyJson;
    set_e2e_error(reason: string): void;
    has_e2e(): boolean;
    free(): void;
  }

  type InitInput = string | URL | Request | Response | BufferSource | WebAssembly.Module;
  /** wasm-bindgen's init: the single-object form (a bare argument still
   *  works but logs a deprecation warning). */
  export default function init(
    input?: { module_or_path: InitInput | Promise<InitInput> },
  ): Promise<InitOutput>;
}
