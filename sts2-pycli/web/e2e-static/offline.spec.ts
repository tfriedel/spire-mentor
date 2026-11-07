import { test, expect } from "@playwright/test";
import { openMenu, closeMenu } from "../e2e/topbarMenu";

/**
 * Offline from the FIRST visit: the service worker precaches the whole shell
 * (index.html, every wb-assets chunk incl. the wasm engine, fonts, icons)
 * from /wb-precache.json at install, so a page that was loaded online once
 * comes back with the network cut. M1's worker only cached chunks on use, so
 * a phone that installed the app and went offline got a blank page.
 */
test("the shell is precached at install and the app loads offline", async ({ page, context }) => {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();

  // The build stamps the id into sw.js; the shell cache is named after it.
  const precache = await page.evaluate(async () => {
    const r = await fetch("/wb-precache.json", { cache: "no-cache" });
    return (await r.json()) as { build: string; files: string[] };
  });
  // The engine ships gzipped (vite.config.ts), never as a plain .wasm.
  expect(precache.files.some((f) => /^wb-assets\/sts2_workbench_wasm_bg\.wasm-[\w-]+\.gz$/.test(f))).toBe(true);
  expect(precache.files.some((f) => f.endsWith(".wasm"))).toBe(false);
  const shell = `wb-shell-${precache.build}`;

  await page.evaluate(() => navigator.serviceWorker.ready);
  // Install finishes after `ready` resolves its activation; wait for the last
  // precached file to land rather than for the cache to merely exist.
  const last = `/${precache.files[precache.files.length - 1]}`;
  await expect
    .poll(
      () =>
        page.evaluate(
          async ([name, url]) => {
            const c = await caches.open(name);
            return (await c.match(url)) != null && (await c.match("/")) != null;
          },
          [shell, last] as const,
        ),
      { timeout: 60_000 },
    )
    .toBe(true);

  await context.setOffline(true);
  await page.reload();
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible({ timeout: 30_000 });
  // The engine (wasm from the shell cache) still answers: starting a run works.
  await page.getByLabel("seed", { exact: true }).fill("OFFLINE1");
  await page.getByRole("button", { name: "Play yourself" }).click();
  await expect(page.locator(".gs-event-option").first()).toBeVisible({ timeout: 60_000 });
  await context.setOffline(false);
});

/**
 * "Download for offline" fills the art and PPO caches up front (src/offline.ts
 * over the lists the static build emits), so a fight the device has never
 * seen renders with the network off — the worker alone only keeps what was
 * fetched. The build ships the art as WebP under the .png names; the browser
 * decodes by content, so the images below must have a real width.
 */
test("download for offline: every art file is cached and an unseen fight renders offline", async ({ page, context }) => {
  await page.goto("/");
  await page.evaluate(() => navigator.serviceWorker.ready);
  const list = await page.evaluate(async () => (await (await fetch("/wb-art-list.json")).json()) as { files: unknown[] });
  expect(list.files.length).toBeGreaterThan(1000);

  // The button is a row of the Tools menu (GH #703); the menu stays open
  // while the download runs, since the control keeps its one DOM node.
  await openMenu(page, "Tools");
  const download = page.getByTestId("offline-download");
  await expect(download).toBeVisible();
  await download.click();
  await expect(page.getByTestId("offline-ready")).toBeVisible({ timeout: 170_000 });
  await closeMenu(page);

  // A card the start screen never showed is served from the cache offline,
  // and it decodes (WebP bytes under a .png name).
  await context.setOffline(true);
  const unseen = await page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img.naturalWidth);
        img.onerror = () => resolve(-1);
        img.src = "/assets/card_atlas/ironclad/anger.png";
      }),
  );
  expect(unseen).toBeGreaterThan(0);

  // The art cache is keyed by the BARE url, so a `?v=<rev>`-busted request
  // (VFX frames, vfx_bindings, audio clips) still hits the entry the download
  // stored -- the behaviour `{ ignoreSearch: true }` used to provide, before
  // its whole-cache scan made the download crawl as it filled. Both halves
  // matter: the query must not reach the key, and it must not reach a lookup.
  const busted = await page.evaluate(async () => {
    const res = await fetch("/assets/card_atlas/ironclad/anger.png?v=cachebust1");
    const keys = await (await caches.open("wb-art-v1")).keys();
    return { ok: res.ok, queried: keys.filter((r) => r.url.includes("?")).length };
  });
  expect(busted.ok).toBe(true);
  expect(busted.queried).toBe(0);

  await page.getByLabel("seed", { exact: true }).fill("OFFLINE2");
  await page.getByRole("button", { name: "Play yourself" }).click();
  await expect(page.locator(".gs-event-option").first()).toBeVisible({ timeout: 60_000 });
  const rendered = await page.evaluate(() => [...document.images].filter((i) => i.naturalWidth > 0).length);
  expect(rendered).toBeGreaterThan(0);
  await context.setOffline(false);
});

