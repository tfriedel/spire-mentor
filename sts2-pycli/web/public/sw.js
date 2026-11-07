// Service worker for the browser-only workbench (static build only; the
// page registers it from src/pwa.ts when VITE_WORKBENCH_MODE=static).
//
// Two caches:
//   * SHELL — same-origin app files: index.html and the other pages, the
//     hashed wb-assets/* chunks (JS, CSS, the worker chunk, the wasm engine),
//     fonts, icons, sfx_map.json. Precached at install from
//     /wb-precache.json, which the Vite build writes next to sw.js (the list
//     of every wb-assets/* file it emitted), so the app works offline from
//     the FIRST online visit — not only once every chunk happened to be
//     fetched while a worker was in control. Cache-first afterwards for the
//     hashed files; the page itself is network-first (falls back to the
//     cache offline) so a new deploy shows on the next reload, not the one
//     after.
//     The cache is named after the build id the Vite build stamps below, so
//     each deploy installs a fresh shell and drops the previous build's
//     chunks (the gzipped wasm engine is ~7 MB; they must not pile up).
//   * ART — every request to the art origin (the asset bucket or the local
//     /assets mount), cache-first, as it is fetched. That is how a fight you
//     have already seen renders offline; art you never loaded is not there.
//     Named WITHOUT the build id: a redeploy must not throw away the art a
//     phone spent its data plan on.
//
// `__BUILD_ID__` / `__ASSETS_BASE__` are replaced by the sw-precache plugin
// in vite.config.ts when the static build copies this file into dist/. Under
// the dev server they stay literal and the worker behaves like the M1 one
// (no precache list, art base only via postMessage).
const BUILD = "__BUILD_ID__";
const BAKED_ASSETS_BASE = "__ASSETS_BASE__";
const SHELL = `wb-shell-${BUILD.startsWith("__") ? "dev" : BUILD}`;
const ART = "wb-art-v1";
// The PPO champions (/ppo/<file>.onnx, ~8.5 MB each): fetched on first use
// per character, kept cache-first across deploys — a promotion changes the
// filename, so a stale entry is simply never asked for again.
const PPO = "wb-ppo-v1";
// The e2e student (/e2e/<file>.e2ew, ~21 MB) lives in the same cache: fetched
// only when that brain is chosen (never precached, never in "Download for
// offline"), then cache-first like a champion.
const PRECACHE_LIST = "/wb-precache.json";
// Files under public/ the shell needs that carry no content hash.
const SHELL_EXTRAS = [
  "/",
  "/manifest.webmanifest",
  "/fonts/Kreon.ttf",
  "/sfx_map.json",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
];

async function precacheList() {
  try {
    const res = await fetch(PRECACHE_LIST, { cache: "no-cache" });
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body.files) ? body.files.map((f) => `/${f}`) : [];
  } catch {
    return [];
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL);
      const files = await precacheList();
      // addAll is atomic: a failed install keeps the previous shell serving
      // and the browser retries at the next update check.
      await cache.addAll([...SHELL_EXTRAS, ...files]).catch(() => undefined);
    })(),
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => (k.startsWith("wb-shell-") && k !== SHELL) || k.startsWith("wb-sw-v1"))
          .map((k) => caches.delete(k)),
      ),
    ),
  );
  self.clients.claim();
});

function assetsBase() {
  // The page posts the base on every load; the baked value covers the gap
  // after the browser restarted an idle worker (the message is not replayed).
  if (self.__ASSETS_BASE__) return self.__ASSETS_BASE__;
  return BAKED_ASSETS_BASE.startsWith("__") ? "" : BAKED_ASSETS_BASE;
}

function isArt(url) {
  const base = assetsBase();
  if (base && url.href.startsWith(base)) return true;
  return url.origin === self.location.origin && url.pathname.startsWith("/assets/");
}

function isChampion(url) {
  return (
    url.origin === self.location.origin &&
    (url.pathname.startsWith("/ppo/") || url.pathname.startsWith("/e2e/"))
  );
}

// Art and champions are cached under their BARE url. VFX frames, the vfx
// bindings and the audio clips are requested as `…?v=<rev>` while "Download
// for offline" (src/offline.ts) fetches the bare path, and the bytes are the
// same file either way -- so the query must not reach the cache key.
//
// This used to be expressed as `cache.match(req, { ignoreSearch: true })`,
// which Chromium cannot answer from the url index: it walks every entry, so
// the per-lookup cost grew with the cache (measured: 0.44 ms at 250 entries,
// 0.95 at 1000, 1.87 at 2000, 3.74 at 4000 -- linear, and ~180x an exact
// match by the end) while an exact match stayed flat at ~0.02 ms. Cache ops
// serialise per cache handle, so the ~4,200 scans did not overlap either:
// "Download for offline" started fast and then crawled as it filled its own
// cache. Normalising the KEY keeps one entry per file and every lookup
// indexed.
function cacheKey(url) {
  return url.origin + url.pathname;
}

self.addEventListener("message", (event) => {
  if (!event.data) return;
  if (event.data.type === "assets-base") {
    self.__ASSETS_BASE__ = String(event.data.base || "");
  }
  // The page's list of art it fetched before this worker controlled it (see
  // src/pwa.ts): store what is not cached yet, one by one, failures ignored.
  if (event.data.type === "warm" && Array.isArray(event.data.urls)) {
    const urls = event.data.urls.filter((u) => typeof u === "string" && isArt(new URL(u, self.location.href)));
    const job = caches.open(ART).then(async (cache) => {
      for (const u of urls) {
        // Same bare key as the fetch handler, or a `?v=`-busted url warmed
        // here would sit in the cache under a key nothing ever looks up.
        const key = cacheKey(new URL(u, self.location.href));
        if (await cache.match(key)) continue;
        try {
          const res = await fetch(u);
          if (res && (res.ok || res.type === "opaque")) await cache.put(key, res);
        } catch {
          /* offline or blocked: nothing to store */
        }
      }
    });
    if (event.waitUntil) event.waitUntil(job);
  }
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  // /api/* never reaches the network in the static build (the worker answers
  // it in-page); if one slips through, let it fail normally.
  if (url.pathname.startsWith("/api/")) return;

  if (isArt(url) || isChampion(url)) {
    event.respondWith(
      caches.open(isChampion(url) ? PPO : ART).then(async (cache) => {
        const key = cacheKey(url);
        const hit = await cache.match(key);
        if (hit) return hit;
        const res = await fetch(req);
        if (res && (res.ok || res.type === "opaque")) cache.put(key, res.clone()).catch(() => undefined);
        return res;
      }),
    );
    return;
  }

  if (url.origin === self.location.origin) {
    const isNav = req.mode === "navigate";
    event.respondWith(
      caches.open(SHELL).then(async (cache) => {
        // Navigations: the multi-page build's /gallery.html, /catalog.html …
        // are their own documents; every other navigation (deep links, query
        // strings) is the SPA shell at "/".
        const key = isNav ? (url.pathname.endsWith(".html") ? url.pathname : "/") : req;
        // ignoreVary: the shell is static same-origin content, but a host may
        // answer with `Vary: Origin` (CORS-enabled servers do) and a module
        // script's request carries an Origin header while the precache fetch
        // did not — without this the precached chunks never match offline.
        const hit = await cache.match(key, { ignoreVary: true });
        const refresh = fetch(req)
          .then((res) => {
            if (res && res.ok) cache.put(key, res.clone()).catch(() => undefined);
            return res;
          })
          .catch(() => undefined);
        if (isNav) {
          // The page itself is network-first: it is tiny, the host serves it
          // no-cache (a 304 when unchanged), and cache-first meant a new deploy
          // only showed up on the SECOND reload. Offline falls back to the
          // precached shell.
          const fresh = await refresh;
          if (fresh) return fresh;
          if (hit) return hit;
          return new Response("offline", { status: 503, statusText: "offline" });
        }
        if (hit) {
          // Hashed chunks and the like: serve from cache, refresh in the background.
          event.waitUntil(refresh);
          return hit;
        }
        const res = await refresh;
        return res || new Response("offline", { status: 503, statusText: "offline" });
      }),
    );
  }
});
