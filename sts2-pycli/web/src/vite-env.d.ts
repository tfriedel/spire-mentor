/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** `"static"` = browser-only workbench: every `/api/*` call is answered by
   *  the Rust engine (WASM) in a Web Worker instead of the Python backend.
   *  Unset = the classic dev / Cloud Run deployment against FastAPI. */
  readonly VITE_WORKBENCH_MODE?: string;
  /** Where the game art is served from in static mode (an origin or a path,
   *  trailing slash tolerated). Defaults to `/assets`, which the Vite dev /
   *  preview server mounts from the sibling sts2-wiki checkout. */
  readonly VITE_ASSETS_BASE?: string;
  /** Build identity (`git describe --always --dirty`, set by `just
   *  workbench-static`): the cache-buster on the VFX manifest / bindings
   *  fetched from the immutable art bucket, and the service worker's shell
   *  cache name (public/sw.js, stamped by vite.config.ts). Unset in plain
   *  `npm run build` / dev: the VFX fetch then falls back to Date.now(). */
  readonly VITE_BUILD_ID?: string;
  /** `"1"` = the build the Spire Mentor mod bundles: the advisor's control
   *  plane serves the page, so the live-game bridge is the page's own origin
   *  (src/bridge.ts). Set by spire-mentor-mod/package.ps1. */
  readonly VITE_BRIDGE_SAME_ORIGIN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
