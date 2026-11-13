import { expect, test, type Locator, type Page } from "@playwright/test";

/** CSS-px viewports of the two phones the layout is checked against: a large
 *  Android (Pixel-class) and the small end (iPhone SE-class). */
export const PHONES = {
  large: { width: 412, height: 915 },
  small: { width: 375, height: 667 },
} as const;

export type PhoneSize = keyof typeof PHONES;
export type Orientation = "portrait" | "landscape";

/** Playwright `use` options for a phone: touch + `isMobile`, which is what
 *  makes `(hover: none)` match in Chromium. Landscape swaps the axes. */
export function phoneContext(size: PhoneSize, orientation: Orientation) {
  const { width, height } = PHONES[size];
  const viewport = orientation === "portrait" ? { width, height } : { width: height, height: width };
  return { viewport, hasTouch: true, isMobile: true, deviceScaleFactor: 2 };
}

/** Viewports that give the board FIXTURE (boardpreview.html, no HUD, tool
 *  row or bottom bar) the stage the real app has on these phones at the
 *  first fight: 714 / 496 real px, hand strip included (Phase 2: portrait
 *  split, real-app stage height; 40 px more since the tool row is no longer
 *  shown on a phone). On the full phone viewport the fixture's
 *  stage is taller than anything a player sees (~850 / 599). */
export const REAL_STAGE: Record<PhoneSize, { width: number; height: number }> = {
  large: { width: 412, height: 714 },
  small: { width: 375, height: 496 },
};

/** Playwright `use` options for the fixture at the real app's stage height,
 *  with the phone's own screen (what `phone.isSmallPhoneScreen` reads). */
export function realStageContext(size: PhoneSize) {
  return { ...phoneContext(size, "portrait"), viewport: REAL_STAGE[size], screen: PHONES[size] };
}

/** The fixture at the stage a phone held upright in a real browser gives the
 *  app (Phase 2: portrait split, short stages): the browser's bars take
 *  ~114 / ~135 px of the screen, so the app's stage is ~342 / ~539 real px
 *  instead of 456 / 674 (the fixture's viewport is its stage). With
 *  the tool row gone from phones a real stage is 40 px taller than these;
 *  the cases are kept at their heights for the cramped modes they cover.
 *  `small` (342) and `browser-bars` (375, mid-range of a small phone with
 *  its bars) are under every chrome mode's floor: the split is cramped and
 *  bare (no powers or names). `near-limit` (410) is just above the stage
 *  where five enemies still fit the compact mode. */
export const SHORT_PORTRAIT = {
  small: { viewport: { width: 375, height: 342 }, screen: PHONES.small, stage: 342 },
  large: { viewport: { width: 412, height: 539 }, screen: PHONES.large, stage: 539 },
  "browser-bars": { viewport: { width: 375, height: 375 }, screen: PHONES.small, stage: 375 },
  "near-limit": { viewport: { width: 375, height: 410 }, screen: PHONES.small, stage: 410 },
} as const;
export type ShortPortrait = keyof typeof SHORT_PORTRAIT;

/** Playwright `use` options for the fixture at a short portrait stage. */
export function shortPortraitContext(which: ShortPortrait) {
  const { viewport, screen } = SHORT_PORTRAIT[which];
  return { ...phoneContext(which === "large" ? "large" : "small", "portrait"), viewport, screen };
}

/** The same two phones held sideways in a real browser: Android Chrome's
 *  status and URL bars take ~80 px off the short side, so the page gets
 *  915x332 / 667x295 while the SCREEN is still 915x412 / 667x375 (which is
 *  what `phone.isSmallPhoneScreen` reads). */
export const SHORT_LANDSCAPE = {
  large: { viewport: { width: 915, height: 332 }, screen: { width: 915, height: 412 } },
  small: { viewport: { width: 667, height: 295 }, screen: { width: 667, height: 375 } },
} as const;

/** Playwright `use` options for a phone sideways with the browser's bars. */
export function shortLandscapeContext(size: PhoneSize) {
  return { ...SHORT_LANDSCAPE[size], hasTouch: true, isMobile: true, deviceScaleFactor: 2 };
}

/** Open the backend-free combat board fixture (boardpreview.html) and wait for
 *  the board and its first hand slot to exist. `query` is the part after `?`. */
export async function openPreview(page: Page, query: string): Promise<void> {
  await page.goto(`/boardpreview.html?${query}`);
  // This host's network flaps (`net::ERR_NETWORK_CHANGED` in the page log): a
  // module request dropped that way leaves the page without a board, and the
  // test then sat out its whole 2-minute timeout. A board that is not up in
  // 20s gets one reload.
  try {
    await page.locator(".gs-board").waitFor({ state: "visible", timeout: 20_000 });
  } catch {
    // Recorded, so the flake rate stays visible in the report and the log.
    const note = `no board after 20s on ?${query}; reloaded once`;
    test.info().annotations.push({ type: "preview-reload", description: note });
    console.warn(`[preview-reload] ${test.info().title}: ${note}`);
    await page.reload();
    await page.locator(".gs-board").waitFor({ state: "visible" });
  }
  await page.locator(".gs-sprite-wrap").first().waitFor({ state: "visible" });
  await page.waitForTimeout(800); // rigs load and the stage zoom settles
}

/** Tap hand card `i` in the middle of the strip of it no later card covers,
 *  inside the visible band at the stage's bottom edge. */
export async function tapSlot(page: Page, i: number): Promise<void> {
  const pt = await page.evaluate((idx) => {
    const hits = Array.from(document.querySelectorAll(".gs-handslot-hit")).map((h) => h.getBoundingClientRect());
    const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
    const b = hits[idx];
    const right = idx < hits.length - 1 ? hits[idx + 1].left : b.right;
    return { x: (b.left + right) / 2, y: stage.bottom - 30 };
  }, i);
  await page.touchscreen.tap(pt.x, pt.y);
}

/** Bounding box in CSS px (what is on the glass, transforms included). */
export async function realBox(loc: Locator) {
  const box = await loc.boundingBox();
  if (!box) throw new Error("realBox: element has no box (hidden or detached)");
  return box;
}

/** Rendered font size in CSS px: the computed size times the cumulative scale
 *  of every transformed ancestor (the stage is drawn in design px and scaled
 *  by `--gs-k`, so the computed size alone says nothing about legibility). */
export async function fontPx(loc: Locator): Promise<number> {
  return loc.evaluate((el) => {
    let scale = 1;
    for (let n: Element | null = el; n; n = n.parentElement) {
      const t = getComputedStyle(n).transform;
      if (t && t !== "none") {
        const m = new DOMMatrixReadOnly(t);
        scale *= Math.hypot(m.a, m.b);
      }
    }
    return parseFloat(getComputedStyle(el).fontSize) * scale;
  });
}

/** A creature's DRAWN height in real px: crown (`--rig-top`, design px inside
 *  the stage's `--gs-k` zoom) to the ground line (the sprite box's bottom).
 *  `anchor` is its `data-combatant` ("player", "osty"). */
export async function drawnHeight(page: Page, anchor = "player"): Promise<number> {
  return page.evaluate((a) => {
    const wrap = document.querySelector(`.gs-combatant[data-combatant="${a}"] > .gs-sprite-wrap`) as HTMLElement;
    const stage = document.querySelector(".gs-arena-stage")!;
    const k = parseFloat(getComputedStyle(stage).getPropertyValue("--gs-k")) || 1;
    const rigTop = parseFloat(getComputedStyle(wrap).getPropertyValue("--rig-top")) || 0;
    const b = wrap.getBoundingClientRect();
    return b.bottom - (b.top + rigTop * k);
  }, anchor);
}

/** `drawnHeight` once the rig has settled: two readings 200 ms apart that
 *  agree. */
export async function settledDrawnHeight(page: Page, anchor = "player"): Promise<number> {
  let prev = -1;
  let h = await drawnHeight(page, anchor);
  await expect.poll(async () => {
    prev = h;
    await page.waitForTimeout(200);
    h = await drawnHeight(page, anchor);
    return h > 0 && Math.abs(h - prev) < 0.5;
  }, { timeout: 15_000 }).toBe(true);
  return h;
}

/** Start a cheap run (Ironclad, seed "tablet-landscape", 20 sims) and wait for
 *  its first screen (Neow, before any board). */
export async function startRun(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
  await page.getByLabel("seed", { exact: true }).fill("tablet-landscape");
  // Cheap budgets: a combat step blocks on MCTS, and this test only needs the
  // first fight to exist.
  await page.getByLabel("monster combat sims").fill("20");
  await page.getByLabel("boss combat sims").fill("20");
  await page.getByRole("button", { name: "Play yourself" }).click();
  // Starting a run is a blocking engine call, so wait for the first screen to
  // exist rather than for a fixed delay — a loaded machine takes seconds.
  await expect(page.locator(".gs-hud")).toBeVisible({ timeout: 90_000 });
}

/** Start a cheap run and click through Neow / rewards / the map until the
 *  combat board is up. The workbench's own "best" buttons are not used: the
 *  point is to arrive at a real fight, not to exercise the advisor. */
export async function reachCombat(page: Page): Promise<void> {
  await startRun(page);

  const stage = page.locator(".gs-arena-stage");
  const choices = [
    ".gs-event-option:not(:disabled)",
    ".gs-treasure-hit:not(:disabled)",
    "g[data-can]", // an act-map node we may move to
    "button:has-text('Skip'):not(:disabled)",
  ];
  // Every screen between Neow and the first fight offers at least one of these.
  // Wait for whichever appears, take it, repeat.
  const anyChoice = page.locator(choices.join(", ")).first();
  for (let i = 0; i < 12 && (await stage.count()) === 0; i++) {
    await anyChoice.waitFor({ state: "visible", timeout: 90_000 });
    for (const sel of choices) {
      const loc = page.locator(sel).first();
      if ((await loc.count()) > 0) {
        await loc.click();
        break;
      }
    }
    await page.waitForTimeout(1500);
  }
  await expect(stage, "never reached combat").toBeVisible();
  // Let the rigs load and the zoom settle before measuring anything.
  await expect(page.locator(".gs-sprite-wrap").first()).toBeVisible();
  await expect(page.locator(".gs-handslot").first()).toBeVisible();
  await page.waitForTimeout(2000);
}
