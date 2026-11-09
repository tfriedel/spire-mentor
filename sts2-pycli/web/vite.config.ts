import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";

declare const process: { env: Record<string, string | undefined>; cwd(): string };

const here = path.dirname(fileURLToPath(import.meta.url));

// The workbench backend also serves the SimLab API (it mounts the SimLab app),
// so a single backend covers all of /api/* — no separate :8078 needed in dev.
const backend = process.env.STS2_BACKEND ?? "http://127.0.0.1:8077";

// Content types for the art tree the dev/preview server mounts in static mode
// (see `artMount`). Anything else goes out as octet-stream.
const MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".atlas": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".webm": "video/webm",
  ".ogg": "audio/ogg",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
};

/**
 * Static-mode art mount: with no backend to serve `/assets`, and
 * `VITE_ASSETS_BASE` unset, serve the sibling sts2-wiki checkout's art tree —
 * `site/public/images` at `/assets` and `site/public/audio` at
 * `/assets/audio`, the same layout the Python server's images mount had.
 * ~30 lines instead of a dependency (sirv is not in the tree).
 */
function artMount(): Plugin {
  const images = path.resolve(here, "../../sts2-wiki/site/public/images");
  const audio = path.resolve(here, "../../sts2-wiki/site/public/audio");
  const handler = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    let rel: string;
    try {
      rel = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
    } catch {
      return next();
    }
    let root = images;
    if (rel === "/audio" || rel.startsWith("/audio/")) {
      root = audio;
      rel = rel.slice("/audio".length);
    }
    const file = path.normalize(path.join(root, rel));
    if (!file.startsWith(root + path.sep)) return next();
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) return next();
      res.setHeader("content-type", MIME[path.extname(file).toLowerCase()] ?? "application/octet-stream");
      res.setHeader("content-length", st.size);
      res.setHeader("cache-control", "no-cache");
      if (req.method === "HEAD") return res.end();
      fs.createReadStream(file).pipe(res);
    });
  };
  return {
    name: "sts2-art-mount",
    configureServer(server) {
      server.middlewares.use("/assets", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/assets", handler);
    },
  };
}

/**
 * The PPO champions for the browser build (M3). `data/ppo/champions.toml`
 * names one ONNX per character; the worker fetches `/ppo/<file>` for the
 * run's character and loads it into the wasm engine. Two jobs:
 *   1. dev / preview: serve those files from `../data/ppo` at `/ppo`.
 *   2. build: copy them into `<outDir>/ppo/` so the static host ships them
 *      (8.5 MB each — NOT precached; the service worker stores them
 *      cache-first on first use, and `_headers` serves `/ppo/*` immutable).
 * Reading the manifest here (rather than a hand-kept list) keeps the shipped
 * set equal to what `just check-champions` gates on the Rust side.
 */
function championFiles(): string[] {
  const manifest = path.resolve(here, "../data/ppo/champions.toml");
  const text = fs.readFileSync(manifest, "utf8");
  const files = [...text.matchAll(/^onnx\s*=\s*"([^"]+)"/gm)].map((m) => m[1]);
  if (files.length === 0) throw new Error(`no champions listed in ${manifest}`);
  return files;
}

function ppoChampions(): Plugin {
  const root = path.resolve(here, "../data/ppo");
  let outDir = "dist";
  const handler = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    let rel: string;
    try {
      rel = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
    } catch {
      return next();
    }
    const name = rel.replace(/^\/+/, "");
    if (name === "index.json") {
      res.setHeader("content-type", "application/json");
      res.setHeader("cache-control", "no-cache");
      return res.end(JSON.stringify({ files: championFiles() }));
    }
    if (!championFiles().includes(name)) return next();
    const file = path.join(root, name);
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) return next();
      res.setHeader("content-type", "application/octet-stream");
      res.setHeader("content-length", st.size);
      res.setHeader("cache-control", "no-cache");
      if (req.method === "HEAD") return res.end();
      fs.createReadStream(file).pipe(res);
    });
  };
  return {
    name: "sts2-ppo-champions",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    configureServer(server) {
      server.middlewares.use("/ppo", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/ppo", handler);
    },
    closeBundle() {
      const dest = path.resolve(here, outDir, "ppo");
      fs.mkdirSync(dest, { recursive: true });
      for (const name of championFiles()) {
        const src = path.join(root, name);
        if (!fs.existsSync(src)) throw new Error(`champion missing: ${src} (see data/ppo/champions.toml)`);
        fs.copyFileSync(src, path.join(dest, name));
      }
      // The list "Download for offline" (src/offline.ts) walks.
      fs.writeFileSync(path.join(dest, "index.json"), JSON.stringify({ files: championFiles() }));
    },
  };
}

/**
 * The e2e student's weights (`../data/e2e/*.e2ew`, ~21 MB). Like the
 * champions: served at `/e2e` in dev / preview and copied to `<outDir>/e2e/`
 * at build. Unlike them it is fetched ONLY when the e2e brain is chosen —
 * no index.json, so "Download for offline" never sees it, and it is not
 * precached; the service worker keeps it cache-first after the first use and
 * `_headers` serves `/e2e/*` immutable. The public export ships no
 * `data/e2e`, so a missing directory just builds without the brain's net
 * (the brain then answers 424 with the fetch's 404).
 */
function e2eWeights(): Plugin {
  const root = path.resolve(here, "../data/e2e");
  const files = (): string[] =>
    fs.existsSync(root) ? fs.readdirSync(root).filter((f) => f.endsWith(".e2ew")) : [];
  let outDir = "dist";
  const handler = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    let rel: string;
    try {
      rel = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
    } catch {
      return next();
    }
    const name = rel.replace(/^\/+/, "");
    if (!files().includes(name)) return next();
    const file = path.join(root, name);
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) return next();
      res.setHeader("content-type", "application/octet-stream");
      res.setHeader("content-length", st.size);
      res.setHeader("cache-control", "no-cache");
      if (req.method === "HEAD") return res.end();
      fs.createReadStream(file).pipe(res);
    });
  };
  return {
    name: "sts2-e2e-weights",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    configureServer(server) {
      server.middlewares.use("/e2e", handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/e2e", handler);
    },
    closeBundle() {
      const names = files();
      if (names.length === 0) return;
      const dest = path.resolve(here, outDir, "e2e");
      fs.mkdirSync(dest, { recursive: true });
      for (const name of names) fs.copyFileSync(path.join(root, name), path.join(dest, name));
    },
  };
}

/**
 * Static-build companion of public/sw.js. Two jobs, both at build time:
 *   1. emit `wb-precache.json` (dist root — NOT under .vite/, which a static
 *      host's dot-file ignore would drop) listing every `wb-assets/*` file the
 *      build produced: the JS/CSS chunks of all pages, the engine worker chunk
 *      and the wasm the worker imports by `?url`. The service worker precaches
 *      that list at install, so the app works offline from the first visit.
 *   2. stamp the build id and the art origin into dist/sw.js in place of the
 *      `__BUILD_ID__` / `__ASSETS_BASE__` literals — the id names the shell
 *      cache (a new id per deploy = fresh shell, old chunks dropped) and is
 *      what makes the browser see a byte-different sw.js and re-install it.
 */
function swPrecache(buildId: string, assetsBase: string): Plugin {
  let outDir = "dist";
  return {
    name: "sts2-sw-precache",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    generateBundle(_options, bundle) {
      const files = Object.keys(bundle)
        .filter((f) => f.startsWith("wb-assets/"))
        .sort();
      this.emitFile({
        type: "asset",
        fileName: "wb-precache.json",
        source: JSON.stringify({ build: buildId, files }),
      });
    },
    closeBundle() {
      // Vite copies public/ into outDir before this hook; the multi-page
      // build runs closeBundle once, after every page is written.
      const sw = path.resolve(here, outDir, "sw.js");
      if (!fs.existsSync(sw)) return;
      const src = fs.readFileSync(sw, "utf8");
      fs.writeFileSync(
        sw,
        src.replace('"__BUILD_ID__"', JSON.stringify(buildId)).replace('"__ASSETS_BASE__"', JSON.stringify(assetsBase)),
      );
    },
  };
}

export default defineConfig(({ mode, command }) => {
  // VITE_* from the shell (the npm scripts set VITE_WORKBENCH_MODE) or a .env.
  const env = { ...loadEnv(mode, here, "VITE_"), ...process.env };
  // Static = the browser-only workbench: /api/* is answered by the Rust engine
  // (WASM) inside a Web Worker, nothing is proxied, and the art comes from
  // VITE_ASSETS_BASE or the local mount above.
  const isStatic = env.VITE_WORKBENCH_MODE === "static";
  // The engine worker imports the wasm-bindgen package by the bare name
  // `sts2-workbench-wasm`. Static builds resolve it to the generated (gitignored)
  // src/wasm/pkg/ from `just wasm-build`; every other build gets the throwing
  // stub so the worker still bundles without the generated files present
  // (Vite resolves `new Worker(new URL(...))` whether or not it ever runs).
  const wasmDir = path.resolve(here, isStatic ? "src/wasm/pkg" : "src/wasm/stub");
  const wasmEntry = path.join(wasmDir, isStatic ? "sts2_workbench_wasm.js" : "sts2_workbench_wasm.ts");
  // The static BUILD ships the engine gzipped: `just wasm-build` writes
  // `sts2_workbench_wasm_bg.wasm.gz` next to the wasm, and the worker's
  // `?url` import of the wasm is pointed at it here, so dist carries only the
  // .gz (the plain wasm is over Cloudflare Pages' 25 MiB per-file cap;
  // scripts/workbench_static.py gates dist on it). The worker inflates it
  // with DecompressionStream (src/worker/wasmLoad.ts). `vite dev` keeps the
  // plain wasm.
  const wasmGz = path.join(wasmDir, "sts2_workbench_wasm_bg.wasm.gz");
  const shipGz = isStatic && command === "build";
  if (shipGz && !fs.existsSync(wasmGz)) {
    throw new Error(`${wasmGz} is missing: run \`just wasm-build\` (it writes the gzipped engine the static build ships)`);
  }
  // What the service worker's shell cache is named after (see swPrecache);
  // `just workbench-static` passes `git describe`, a bare `npm run build` gets "dev".
  const buildId = env.VITE_BUILD_ID || "dev";
  const assetsBase = (env.VITE_ASSETS_BASE || "").replace(/\/$/, "");

  return {
    plugins: [
      react(),
      ...(isStatic && !env.VITE_ASSETS_BASE ? [artMount()] : []),
      ...(isStatic ? [ppoChampions(), e2eWeights(), swPrecache(buildId, assetsBase)] : []),
    ],
    resolve: {
      alias: [
        ...(shipGz
          ? [{ find: /^sts2-workbench-wasm\/sts2_workbench_wasm_bg\.wasm\?url$/, replacement: `${wasmGz}?url` }]
          : []),
        { find: /^sts2-workbench-wasm\/(.+)$/, replacement: `${wasmDir}/$1` },
        { find: /^sts2-workbench-wasm$/, replacement: wasmEntry },
      ],
    },
    build: {
      // NOT the default "assets": the backend mounts the game-art tree at
      // /assets (server.py images mount), which shadows the SPA's bundles when
      // it serves web/dist itself (STS2_SERVE_SPA=1) -- every JS chunk 404'd.
      assetsDir: "wb-assets",
      rollupOptions: {
        // Relative to the config root; vite resolves these against the project dir.
        input: {
          main: "index.html",
          gallery: "gallery.html",
          cardskin: "cardskin.html",
          simlab: "simlab.html",
          seedcheck: "seedcheck.html",
          catalog: "catalog.html",
          frame: "frame.html",
        },
      },
    },
    // `vite preview` inherits host/allowedHosts from here. Default: this
    // machine only — the build carries the game's art, which is for the
    // user's own use. STS2_WEB_HOST=0.0.0.0 (or an address) opens it to the
    // LAN; STS2_WEB_ALLOWED_HOSTS=name1,name2 admits other Host headers.
    server: {
      host: process.env.STS2_WEB_HOST || "127.0.0.1",
      allowedHosts: (process.env.STS2_WEB_ALLOWED_HOSTS ?? "")
        .split(",")
        .map((h) => h.trim())
        .filter(Boolean),
      // Single backend serves both the workbench and SimLab APIs. Nothing to
      // proxy in static mode: /api/* never leaves the page.
      proxy: isStatic ? undefined : { "/api": backend, "/assets": backend },
    },
  };
});
