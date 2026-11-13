import { test, expect, type Page } from "@playwright/test";
import {
  fontPx, phoneContext, reachCombat, realBox, settledDrawnHeight, shortLandscapeContext, startRun, type PhoneSize,
} from "./phoneHarness";

/**
 * The fighters in a REAL fight on a phone held sideways.
 *
 * The backend-free fixture has no workbench chrome, so its landscape stage is
 * ~350-384 real px tall; in the real app the HUD band and the tool row leave
 * ~276-313. With two 44px power rows reserved under every fighter the
 * Ironclad came out 57 real px tall on the large phone and 17 on the small one.
 * The powers now stand in a column beside each creature (powerColumn.ts) and
 * only the HP bar and the nameplate hang under the feet.
 */

for (const size of ["large", "small"] as PhoneSize[]) {
  test.describe(`${size} phone, landscape: a real fight`, () => {
    test.use(phoneContext(size, "landscape"));

    test("the fighters are drawn large and every floor holds", async ({ page }) => {
      test.setTimeout(240_000);
      await reachCombat(page);
      const stage = await realBox(page.locator(".gs-arena-stage"));
      expect(stage.height).toBeGreaterThanOrEqual(240);
      // Measured before the column: 57 (large) and 17 (small). After: 120 / 96.
      const h = await settledDrawnHeight(page);
      console.log(`[phone-fight] ${size} landscape: stage ${stage.height.toFixed(0)} px, player ${h.toFixed(1)} px`);
      expect(h).toBeGreaterThanOrEqual(90);

      // Under every fighter only its HP bar and nameplate, all above the
      // hand's 96px strip; the nameplate stays one line.
      const stripTop = stage.y + stage.height - 96;
      for (const plate of await page.locator(".gs-nameplate").all()) {
        const b = await realBox(plate);
        expect(b.y + b.height).toBeLessThanOrEqual(stripTop + 1);
        expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
        expect(await fontPx(plate)).toBeGreaterThanOrEqual(11.9);
      }
      for (const side of await page.locator(".gs-side").all()) {
        const b = await realBox(side);
        expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
      }
      expect(await fontPx(page.locator(".gs-hptext").first())).toBeGreaterThanOrEqual(13.9);
      for (const intent of await page.locator(".gs-intent").all()) {
        const b = await realBox(intent);
        expect(b.width).toBeGreaterThanOrEqual(43.5);
        expect(b.height).toBeGreaterThanOrEqual(43.5);
        expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
      }
      // Powers, where the fight has any: a column of 44px chips above the strip.
      for (const chip of await page.locator(".gs-power, .gs-power-more").all()) {
        const b = await realBox(chip);
        expect(b.width).toBeGreaterThanOrEqual(43.5);
        expect(b.height).toBeGreaterThanOrEqual(43.5);
        expect(b.y + b.height).toBeLessThanOrEqual(stripTop + 1);
        expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
      }
      const end = await realBox(page.locator(".gs-endturn"));
      expect(end.width).toBeGreaterThanOrEqual(123.5);
      expect(end.height).toBeGreaterThanOrEqual(55.5);
      expect((await realBox(page.locator(".gs-energy").first())).width).toBeGreaterThanOrEqual(59.5);
      for (const pile of await page.locator(".gs-pilebtn").all()) {
        const b = await realBox(pile);
        expect(b.width).toBeGreaterThanOrEqual(43.5);
        expect(b.height).toBeGreaterThanOrEqual(43.5);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(PAGE_W[size]);
    });
  });
}

const PAGE_W: Record<PhoneSize, number> = { large: 915, small: 667 };

type R = { left: number; top: number; right: number; bottom: number };
const hits = (a: R, b: R) => a.left < b.right - 0.5 && b.left < a.right - 0.5
  && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;

/** Tap hand card `i` in the strip of it no later card covers. */
async function tapSlot(page: Page, i: number): Promise<void> {
  const pt = await page.evaluate((idx) => {
    const hs = Array.from(document.querySelectorAll(".gs-handslot-hit")).map((h) => h.getBoundingClientRect());
    const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
    const b = hs[idx];
    const right = idx < hs.length - 1 ? hs[idx + 1].left : b.right;
    return { x: (b.left + right) / 2, y: stage.bottom - 30 };
  }, i);
  await page.touchscreen.tap(pt.x, pt.y);
}

/** Zoom card `i` and read what the placement rules care about, in real px. */
async function zoomed(page: Page, i: number) {
  await tapSlot(page, i);
  const z = page.locator(".gs-zoomcard");
  await expect(z).toHaveAttribute("data-zoom-index", /\d+/);
  await z.evaluate((el) => Promise.all(el.getAnimations().map((an) => an.finished)));
  await page.waitForTimeout(350); // the re-measure after the slot's lift
  return page.evaluate(() => {
    const r = (el: Element) => { const b = el.getBoundingClientRect(); return { left: b.left, top: b.top, right: b.right, bottom: b.bottom }; };
    const enemies = Array.from(document.querySelectorAll("[data-enemy-index]")).map((el) =>
      [el, ...Array.from(el.querySelectorAll(".gs-intent, .gs-powers"))].map(r).filter((x) => x.right > x.left)
        .reduce((u, x) => ({ left: Math.min(u.left, x.left), top: Math.min(u.top, x.top),
          right: Math.max(u.right, x.right), bottom: Math.max(u.bottom, x.bottom) })));
    return {
      zoom: r(document.querySelector(".gs-zoomcard")!),
      stage: r(document.querySelector(".gs-arena-stage")!),
      end: r(document.querySelector(".gs-endturn")!),
      enemies,
      armed: !!document.querySelector(".gs-board[data-armed]"),
    };
  });
}

/** Deselect by tapping the empty stage corner away from the zoom. */
async function deselect(page: Page, zoom: R, stage: R): Promise<void> {
  const left = (zoom.left + zoom.right) / 2 > (stage.left + stage.right) / 2;
  await page.touchscreen.tap(left ? stage.left + 4 : stage.right - 4, stage.top + 6);
  await expect(page.locator(".gs-zoomcard")).toHaveCount(0);
}

/** Every rule the zoom keeps in landscape, for the attack (Bash, slot 0) and a
 *  skill (Defend, slot 2) of the Ironclad's opening hand. */
async function checkZoom(page: Page, minW: number, label: string): Promise<void> {
  for (const [slot, attack] of [[0, true], [2, false]] as const) {
    const g = await zoomed(page, slot);
    const w = g.zoom.right - g.zoom.left;
    console.log(`[phone-fight] ${label} ${attack ? "attack" : "skill"}: stage ${(g.stage.bottom - g.stage.top).toFixed(0)} px, zoom ${w.toFixed(1)} px`);
    expect(g.armed).toBe(attack);
    expect(w, `${label} zoom width`).toBeGreaterThanOrEqual(minW - 0.5);
    // Inside the stage (the cost gem stands a few px proud of the top edge).
    expect(g.zoom.left).toBeGreaterThanOrEqual(g.stage.left - 0.5);
    expect(g.zoom.top).toBeGreaterThanOrEqual(g.stage.top - 0.5);
    expect(g.zoom.right).toBeLessThanOrEqual(g.stage.right + 0.5);
    expect(g.zoom.bottom).toBeLessThanOrEqual(g.stage.bottom + 0.5);
    if (attack) for (const e of g.enemies) expect(hits(g.zoom, e), "an attack's zoom covers an enemy").toBe(false);
    // No lit-edge rule any more: the zoom stands on ONE fixed spot for every
    // card (owner, 2026-11-10), so it may hang over the selected slot itself;
    // the part over the strip passes taps through (zoomHitHeight).
    expect(hits(g.zoom, g.end), "the zoom covers End Turn").toBe(false);
    await deselect(page, g.zoom, g.stage);
  }
}

/** Measured in the real app (stage height, real px): 313 / 276 with no
 *  browser bars, 233 / 196 with Android Chrome's (915x332, 667x295). A card
 *  is 1.407 x its width tall plus its gem; its top edge stands
 *  `zoomPlacement.TOP_MARGIN` (12) under the stage top (room for the gem and
 *  the close button), so a 233 px stage holds a 156 px card and a 196 px one
 *  130: the zoom takes all of that. */
const ZOOM_W: Record<PhoneSize, { full: number; short: number }> = {
  large: { full: 170, short: 156 }, small: { full: 160, short: 130 },
};
const STAGE_MIN: Record<PhoneSize, { full: number; short: number }> = {
  large: { full: 310, short: 230 }, small: { full: 273, short: 193 },
};

for (const size of ["large", "small"] as PhoneSize[]) {
  test.describe(`${size} phone, landscape: zoom and touch in a real fight`, () => {
    test.use(phoneContext(size, "landscape"));

    test("the zoom is full width, keeps every rule, and double-tap zoom is off", async ({ page }) => {
      test.setTimeout(240_000);
      await reachCombat(page);
      const stage = await realBox(page.locator(".gs-arena-stage"));
      expect(stage.height).toBeGreaterThanOrEqual(STAGE_MIN[size].full);
      await checkZoom(page, ZOOM_W[size].full, `${size} landscape`);
      // No double-tap zoom on what is tapped twice in a row: `manipulation`,
      // and `none` on the hand, which also keeps the browser from taking a
      // thumb drag across the cards as a scroll (drag-browse, 2026-11-13).
      await tapSlot(page, 2);
      for (const [sel, want] of [[".gs-handslot-hit", "none"], [".gs-zoomcard", "manipulation"], [".gs-hud-relic", "manipulation"]]) {
        expect(await page.locator(sel).first().evaluate((el) => getComputedStyle(el).touchAction), sel)
          .toBe(want);
      }
    });
  });

  test.describe(`${size} phone, landscape with the browser's bars: a real fight`, () => {
    test.use(shortLandscapeContext(size));

    test("the stage keeps its measured height and the zoom takes all of it", async ({ page }) => {
      test.setTimeout(240_000);
      await reachCombat(page);
      const stage = await realBox(page.locator(".gs-arena-stage"));
      expect(stage.height).toBeGreaterThanOrEqual(STAGE_MIN[size].short);
      await checkZoom(page, ZOOM_W[size].short, `${size} short landscape`);
    });
  });

  for (const orientation of ["portrait", "landscape"] as const) {
    test.describe(`${size} phone, ${orientation}: relics off the board`, () => {
      test.use(phoneContext(size, orientation));

      test("a relic tap on the Neow screen opens the sheet; a tap elsewhere closes it", async ({ page }) => {
        test.setTimeout(180_000);
        await startRun(page);
        await expect(page.locator(".gs-arena-stage")).toHaveCount(0);
        const relic = page.locator(".gs-hud-relic:not(.gs-hud-relic-more)").first();
        const name = (await relic.getAttribute("aria-label"))!;
        await relic.tap({ timeout: 5000 });
        const sheet = page.locator(".gs-inspect");
        await expect(sheet).toBeVisible();
        await expect(sheet).toContainText(name);
        expect((await sheet.locator(".gs-inspect-desc").innerText()).length).toBeGreaterThan(5);
        await expect(page.getByRole("tooltip")).toHaveCount(0);
        const b = await realBox(sheet);
        const vp = page.viewportSize()!;
        expect(b.x).toBeGreaterThanOrEqual(0);
        expect(b.y).toBeGreaterThanOrEqual(0);
        expect(b.x + b.width).toBeLessThanOrEqual(vp.width + 0.5);
        expect(b.y + b.height).toBeLessThanOrEqual(vp.height + 0.5);
        // Never under the HUD band.
        const hud = await realBox(page.locator(".gs-hud"));
        expect(b.y).toBeGreaterThanOrEqual(hud.y + hud.height - 0.5);
        // Beside the sheet (it is at most 360 wide and centred), on the
        // screen's backdrop rather than on one of Neow's options.
        await page.touchscreen.tap(6, b.y + b.height / 2);
        await expect(sheet).toHaveCount(0);
      });

      // The owner tapped relics mid-fight and got nothing: with a card armed
      // the board switched ALL inspection off, the HUD's relics included.
      test("with an attack armed, a relic tap still explains the relic; the attack stays armed", async ({ page }) => {
        test.setTimeout(240_000);
        await reachCombat(page);
        await tapSlot(page, 0); // Bash, the opening hand's attack
        await expect(page.locator(".gs-zoomcard")).toHaveCount(1, { timeout: 5000 });
        await expect(page.locator(".gs-board[data-armed]")).toBeVisible();
        const relic = page.locator(".gs-hud-relic:not(.gs-hud-relic-more)").first();
        const name = (await relic.getAttribute("aria-label"))!;
        await relic.tap({ timeout: 5000 });
        const sheet = page.locator(".gs-inspect");
        await expect(sheet).toBeVisible({ timeout: 5000 });
        await expect(sheet).toContainText(name);
        expect((await sheet.locator(".gs-inspect-desc").innerText()).length).toBeGreaterThan(5);
        await expect(page.locator(".gs-board[data-armed]"), "still armed").toBeVisible();
        await expect(page.locator(".gs-zoomcard")).toHaveCount(1);
      });
    });
  }
}
