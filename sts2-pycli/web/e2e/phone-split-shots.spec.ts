import { test, type Page } from "@playwright/test";
import {
  PHONES, openPreview, phoneContext, reachCombat, realStageContext, settledDrawnHeight, tapSlot, type PhoneSize,
} from "./phoneHarness";

/**
 * Screenshots of the phone-portrait split for the write-up
 * (docs/experiments/2026-11-10-phone-ux.md, "Portrait split (phase 2)").
 * Not a test: it asserts nothing and only writes PNGs and log lines. Tagged
 * `@shots` so a normal run leaves it out:
 *
 *   npx playwright test --grep-invert @shots          # every spec but this
 *   npx playwright test e2e/phone-split-shots.spec.ts # just the shots
 */

const OUT = "../../docs/experiments/2026-11-10-phone-ux";

/** `data-split` and every enemy's drawn height (crown to feet), real px. */
async function splitLine(page: Page): Promise<string> {
  const split = await page.locator(".gs-board").getAttribute("data-split");
  const heights = await page.evaluate(() => {
    const stage = document.querySelector(".gs-arena-stage")!;
    const k = parseFloat(getComputedStyle(stage).getPropertyValue("--gs-k")) || 1;
    return Array.from(document.querySelectorAll("[data-enemy-index] > .gs-sprite-wrap")).map((w) => {
      const rigTop = parseFloat(getComputedStyle(w).getPropertyValue("--rig-top")) || 0;
      const b = w.getBoundingClientRect();
      return Math.round(b.bottom - (b.top + rigTop * k));
    });
  });
  const stageH = await page.evaluate(() => Math.round(document.querySelector(".gs-arena-stage")!.getBoundingClientRect().height));
  return `${split} heights ${heights.join(",")} (stage ${stageH})`;
}

async function shot(page: Page, size: PhoneSize, name: string): Promise<void> {
  console.log(`[split] ${size} ${name}: ${await splitLine(page)}`);
  await page.screenshot({ path: `${OUT}/split-${size}-${name}.png` });
}

test.describe.configure({ mode: "serial" });

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait split shots @shots`, () => {
    test.use(phoneContext(size, "portrait"));

    test("a real fight, resting and zoomed", async ({ page }) => {
      test.setTimeout(240_000);
      await reachCombat(page);
      await settledDrawnHeight(page);
      await shot(page, size, "real-1");
      await tapSlot(page, 0);
      await page.waitForTimeout(600);
      await shot(page, size, "real-zoomed");
    });
  });

  test.describe(`${size} phone, portrait split fixture shots @shots`, () => {
    // The fixture at the real app's stage height (`REAL_STAGE`).
    test.use(realStageContext(size));

    test("a hit on the band", async ({ page }) => {
      await openPreview(page, "enemies=2&hit=4000");
      await settledDrawnHeight(page);
      // The hit fires 4000 ms after load; openPreview already waited ~800 ms
      // plus the settle, so wait out the rest of the 4300 ms frame.
      const since = await page.evaluate(() => performance.now());
      await page.waitForTimeout(Math.max(0, 4300 - since));
      await shot(page, size, "hit");
    });

    for (const n of [3, 5]) {
      test(`${n} enemies with powers`, async ({ page }) => {
        await openPreview(page, `enemies=${n}&powers=3&ppowers=3&pblock=12`);
        await settledDrawnHeight(page);
        await page.waitForTimeout(500);
        await shot(page, size, `${n}`);
      });
    }

    // Numbers only (no PNG) for the write-up's table against the spike's
    // np=4 "mirror" cells: the same fights at 3 and 8 powers.
    test("the spike's fights, numbers only", async ({ page }) => {
      test.setTimeout(240_000);
      const fights = [
        "enemy=cultists", "enemies=2", "enemies=3", "enemies=4", "enemies=5", "enemies=5&enemy=cultists",
        "enemy=lagavulin_matriarch", "enemy=queen", "enemy=bygone_effigy",
        "char=necrobinder&osty=1&enemies=2", "char=defect&orbs=3,3&enemies=2",
      ];
      for (const q of fights) {
        for (const p of [3, 8]) {
          await openPreview(page, `${q}&powers=${p}`);
          await settledDrawnHeight(page);
          await page.waitForTimeout(300);
          console.log(`[split] ${size} ${q} powers=${p}: ${await splitLine(page)}`);
        }
      }
    });
  });
}
