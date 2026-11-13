import { test, expect, type Page } from "@playwright/test";
import { PHONES, fontPx, phoneContext, reachCombat, realBox, type Orientation, type PhoneSize } from "./phoneHarness";

/**
 * The chrome around the board (HUD band, combat tool row, folded top bar) on a
 * phone, in a REAL run: boardpreview.html has no HUD. Touch floors in CSS px:
 * 44 for tap targets, 20 for HUD numerals, 13 for the deck badge, 40 tall /
 * 14px text for the tool row, 44 tall for top-bar menu items.
 */
for (const size of Object.keys(PHONES) as PhoneSize[]) {
  for (const orientation of ["portrait", "landscape"] as Orientation[]) {
    test.describe(`${size} phone, ${orientation}: chrome`, () => {
      test.use(phoneContext(size, orientation));
      test.setTimeout(240_000);

      const menuButton = (page: Page) => page.getByRole("button", { name: "Menu", exact: true });
      /** The panel's tappable rows: menu items and hosted controls (not the
       *  section headings, which take no tap). */
      const panelRows = (page: Page) =>
        page.locator('.wb-menu-list--phone > [role="menuitem"], .wb-menu-list--phone > [role="menuitemcheckbox"], '
          + ".wb-menu-list--phone > .wb-menu-row:not(:has(> .wb-menu-heading)):not(:empty)");

      test("HUD and tool row meet the floors", async ({ page }) => {
        await reachCombat(page);
        const hud = page.locator(".gs-hud");
        const min = async (sel: string, px: number, what: string) => {
          const all = page.locator(sel);
          const n = await all.count();
          expect(n, `${what}: none found`).toBeGreaterThan(0);
          for (let i = 0; i < n; i++) {
            const b = await realBox(all.nth(i));
            expect(b.width, `${what} #${i} width`).toBeGreaterThanOrEqual(px - 0.5);
            expect(b.height, `${what} #${i} height`).toBeGreaterThanOrEqual(px - 0.5);
          }
        };
        await min(".gs-hud-relic", 44, "relic button");
        await min(".gs-hud-socket", 44, "potion socket");
        await min(".gs-hud-map", 44, "map button");
        await expect(hud.locator(".gs-hud-deck"), "deck button exists in combat").toHaveCount(1);
        await min(".gs-hud-deck", 44, "deck button");
        expect(await fontPx(hud.locator(".gs-hud-deckcount")), "deck badge text").toBeGreaterThanOrEqual(13);
        expect(await fontPx(hud.locator(".gs-hud-hpnum")), "hp text").toBeGreaterThanOrEqual(20);
        await expect(hud.locator(".gs-hud-goldnum"), "gold exists in combat").toHaveCount(1);
        expect(await fontPx(hud.locator(".gs-hud-goldnum")), "gold text").toBeGreaterThanOrEqual(20);
        // Never collapse potion sockets: all of them are on screen, inside the band.
        const hb = await realBox(hud);
        const sockets = hud.locator(".gs-hud-socket");
        for (let i = 0; i < (await sockets.count()); i++) {
          const b = await realBox(sockets.nth(i));
          expect(b.x + b.width, `socket #${i} inside band`).toBeLessThanOrEqual(hb.x + hb.width + 0.5);
        }
        // The workbench's tool row is not shown on a phone (owner feedback,
        // 2026-11-12): Mode "AI plays", Stop and the Menu cover it.
        await expect(page.locator(".gs-toolrow")).toBeHidden();
        // The old unlabelled ≡ toggle is gone on a phone: the labelled Menu
        // button replaces it (its own test below).
        await expect(page.locator(".wb-chrome-toggle"), "no ≡ toggle on a phone").toHaveCount(0);
      });

      test("the Menu button is labelled, 44 tall, high-contrast and always on screen", async ({ page }) => {
        await reachCombat(page);
        const btn = menuButton(page);
        await expect(btn).toHaveCount(1);
        await expect(btn).toBeVisible();
        await expect(btn).toContainText("Menu");
        const b = await realBox(btn);
        const vw = page.viewportSize()!.width;
        expect(b.height, "height").toBeGreaterThanOrEqual(43.5);
        expect(b.width, "width").toBeGreaterThanOrEqual(71.5);
        expect(b.x + b.width, "pinned at the right edge").toBeGreaterThan(vw - 16);
        expect(b.x + b.width, "on screen").toBeLessThanOrEqual(vw + 0.5);
        expect(b.y, "on screen").toBeGreaterThanOrEqual(-0.5);
        // Outside the bar's sideways scroller, so scrolling it never hides the button.
        expect(await btn.evaluate((el) => !!el.closest(".wb-topbar-scroll")), "outside the scroller").toBe(false);
        const scroller = page.locator(".wb-topbar-scroll");
        if (await scroller.count()) {
          await scroller.evaluate((el) => { el.scrollLeft = el.scrollWidth; });
          const b2 = await realBox(btn);
          expect(b2.x, "unmoved by the scroller").toBeCloseTo(b.x, 0);
        }
        // Strong contrast: text against its own background, WCAG ratio >= 7.
        const ratio = await btn.evaluate((el) => {
          const cs = getComputedStyle(el);
          const lum = (c: string) => {
            const [r, g, bl] = (c.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number).map((v) => {
              const x = v / 255;
              return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
            });
            return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
          };
          const [a, z] = [lum(cs.color), lum(cs.backgroundColor)].sort((x, y) => y - x);
          return (a + 0.05) / (z + 0.05);
        });
        expect(ratio, "contrast").toBeGreaterThanOrEqual(7);
      });

      // The one dropdown on a phone: the Menu button's panel (the bar's own
      // Run / Analyze / Tools are its sections).
      const triggers = ["Menu"];
      for (const name of triggers) {
        test(`"${name}" opens a fixed panel under the bar without scrolling it; every entry reachable`, async ({ page }) => {
          await reachCombat(page);
          const trigger = name === "Menu" ? menuButton(page)
            : page.locator(".wb-topbar-scroll").getByRole("button", { name: new RegExp(`^${name}\\b`) });
          await trigger.scrollIntoViewIfNeeded();
          const scroller = page.locator(".wb-topbar-scroll");
          const sl = async () => ((await scroller.count()) ? scroller.evaluate((el) => el.scrollLeft) : 0);
          const before = await sl();
          await trigger.click({ timeout: 5000 });
          const panel = page.locator(".wb-menu-list--phone");
          await expect(panel).toHaveCount(1, { timeout: 5000 });
          await page.waitForTimeout(200);
          expect(await sl(), "the bar did not scroll").toBe(before);
          const vp = page.viewportSize()!;
          const pb = await realBox(panel);
          expect(await panel.evaluate((el) => getComputedStyle(el).position)).toBe("fixed");
          expect(pb.x, "8px gutter left").toBeCloseTo(8, 0);
          expect(pb.x + pb.width, "8px gutter right").toBeCloseTo(vp.width - 8, 0);
          expect(pb.y + pb.height, "inside the viewport").toBeLessThanOrEqual(vp.height + 0.5);
          if (orientation === "portrait") {
            const bar = await realBox(page.locator(".wb-topbar"));
            expect(pb.y, "under the bar").toBeGreaterThanOrEqual(bar.y + bar.height - 0.5);
          } else {
            const tb = await realBox(trigger);
            expect(pb.y, "under the button").toBeGreaterThanOrEqual(tb.y + tb.height - 0.5);
          }
          // 44px rows, 14px text; scroll the panel and every row is reachable.
          const rows = panelRows(page);
          const n = await rows.count();
          expect(n).toBeGreaterThan(0);
          for (let i = 0; i < n; i++) {
            const r = rows.nth(i);
            await r.scrollIntoViewIfNeeded();
            const rb = await realBox(r);
            const now = await realBox(panel);
            expect(rb.height, `row #${i} height`).toBeGreaterThanOrEqual(43.5);
            expect(rb.y, `row #${i} reachable (top)`).toBeGreaterThanOrEqual(now.y - 0.5);
            expect(rb.y + rb.height, `row #${i} reachable (bottom)`).toBeLessThanOrEqual(now.y + now.height + 0.5);
            const label = r.locator(".wb-menu-label");
            if (await label.count()) expect(await fontPx(label.first()), `row #${i} text`).toBeGreaterThanOrEqual(14);
          }
          console.log(`[menu] ${size} ${orientation} ${name}: panel ${pb.x},${pb.y.toFixed(0)} ${pb.width}x${pb.height.toFixed(0)}, ${n} rows`);
        });
      }

      test("a second tap on the trigger, or a tap outside, closes the menu", async ({ page }) => {
        await reachCombat(page);
        const panel = page.locator(".wb-menu-list--phone");
        const btn = menuButton(page);
        await btn.tap({ timeout: 5000 });
        await expect(panel).toHaveCount(1, { timeout: 5000 });
        await btn.tap({ timeout: 5000 });
        await expect(panel, "second tap on the trigger").toHaveCount(0, { timeout: 5000 });
        await btn.tap({ timeout: 5000 });
        await expect(panel).toHaveCount(1, { timeout: 5000 });
        // A tap on the board below the panel.
        const vp = page.viewportSize()!;
        const pb = await realBox(panel);
        const y = Math.min(vp.height - 4, pb.y + pb.height + 10);
        await page.touchscreen.tap(vp.width / 2, y);
        await expect(panel, "tap outside").toHaveCount(0, { timeout: 5000 });
      });

      test("the bar is four labelled tap-size buttons, nothing to scroll; Undo and Log work", async ({ page }) => {
        await reachCombat(page);
        // No workbench row under the bar pushes the game down.
        await expect(page.locator(".gs-toolrow")).toBeHidden();
        await expect(page.locator(".wb-runbar")).toBeHidden();
        if (orientation !== "portrait") return;
        await expect(page.locator(".wb-topbar-scroll")).toHaveCount(0);
        const btns = page.locator(".wb-phonebar button:visible");
        await expect(btns).toHaveCount(3); // mode, Undo, Log (Menu is beside the bar)
        const vp = page.viewportSize()!;
        for (let i = 0; i < 3; i++) {
          const b = await realBox(btns.nth(i));
          expect(b.height, `button #${i} height`).toBeGreaterThanOrEqual(43.5);
          expect(b.width, `button #${i} width`).toBeGreaterThanOrEqual(43.5);
          expect(b.x + b.width, `button #${i} on screen`).toBeLessThanOrEqual(vp.width);
          expect(await fontPx(btns.nth(i)), `button #${i} text`).toBeGreaterThanOrEqual(14);
        }
        expect(await page.locator(".wb-topbar").evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(0);
        // The mode button offers the four modes.
        await page.locator(".wb-phonemode button").first().click();
        await expect(page.getByRole("menuitemcheckbox")).toHaveCount(4);
        await page.keyboard.press("Escape");
        // Log: a full-screen sheet over everything, with its own Close.
        const side = page.locator(".wb-side");
        await page.getByRole("button", { name: /Log$/ }).tap();
        await expect(side).toBeVisible();
        expect((await realBox(side)).width).toBeGreaterThanOrEqual(vp.width - 1);
        await side.getByRole("button", { name: /Close/ }).tap();
        await expect(side).toBeHidden();
        // Undo: a step was taken to get here, so it is offered and goes back.
        const undo = page.getByRole("button", { name: /Undo$/ });
        await expect(undo).toBeEnabled();
        await undo.tap();
        await expect(page.locator(".gs-arena-stage")).toHaveCount(0, { timeout: 30_000 });
      });

      test("the stage does not resize when a card is armed", async ({ page }) => {
        await reachCombat(page);
        const stage = page.locator(".gs-arena-stage");
        const before = await realBox(stage);
        // A tap zooms (and, for an attack, arms) a card; the stage must hold still.
        await page.locator(".gs-handslot-hit").last().click({ timeout: 5000 });
        await expect(page.locator(".gs-zoomcard"), "the tap zoomed a card").toHaveCount(1, { timeout: 5000 });
        await page.waitForTimeout(500);
        const after = await realBox(stage);
        expect(after).toEqual(before);
      });

      test("the stage keeps its height", async ({ page }) => {
        await reachCombat(page);
        const b = await realBox(page.locator(".gs-arena-stage"));
        expect(b.height, `${orientation} stage`).toBeGreaterThanOrEqual(240);
      });

      test("folded top-bar menu items are 44 tall", async ({ page }) => {
        await reachCombat(page);
        await menuButton(page).click({ timeout: 5000 });
        const items = page.getByRole("menu").getByRole("menuitem");
        const n = await items.count();
        expect(n).toBeGreaterThan(0);
        for (let i = 0; i < n; i++) {
          expect((await realBox(items.nth(i))).height, `menu item #${i}`).toBeGreaterThanOrEqual(43.5);
        }
      });

      test("many relics collapse into a +N that opens the list", async ({ page }) => {
        // Give the real run 14 relics (more than fit even on a 915px-wide band): patch every live-API JSON body that carries
        // a player's relic list (the HUD reads them from the step responses).
        await page.route("**/api/live/**", async (route) => {
          const resp = await route.fetch();
          let body = await resp.text();
          try {
            const json = JSON.parse(body);
            const patch = (o: unknown): void => {
              if (!o || typeof o !== "object") return;
              const rec = o as Record<string, unknown>;
              if (Array.isArray(rec.relics) && rec.relics.length > 0 && typeof rec.relics[0] === "object"
                  && "hp" in rec) {
                const base = rec.relics[0] as Record<string, unknown>;
                rec.relics = Array.from({ length: 14 }, (_, n) => ({
                  ...base, name: `Test Relic ${n + 1}`, description: `Does thing ${n + 1}.`, index: n }));
              }
              for (const v of Object.values(rec)) patch(v);
            };
            patch(json);
            body = JSON.stringify(json);
          } catch { /* not JSON (a stream): pass through */ }
          await route.fulfill({ response: resp, body });
        });
        await reachCombat(page);
        const hud = page.locator(".gs-hud");
        const hb = await realBox(hud);
        const shown = hud.locator(".gs-hud-relic:not(.gs-hud-relic-more)");
        const nShown = await shown.count();
        expect(nShown, "relic buttons shown").toBeGreaterThanOrEqual(3);
        expect(nShown, "some relics collapsed").toBeLessThan(14);
        for (let i = 0; i < nShown; i++) {
          const b = await realBox(shown.nth(i));
          expect(b.width).toBeGreaterThanOrEqual(43.5);
          expect(b.height).toBeGreaterThanOrEqual(43.5);
        }
        const more = hud.locator(".gs-hud-relic-more");
        await expect(more).toHaveCount(1);
        const b = await realBox(more);
        expect(b.width, "+N width").toBeGreaterThanOrEqual(43.5);
        expect(b.height, "+N height").toBeGreaterThanOrEqual(43.5);
        expect(b.x, "+N inside band (left)").toBeGreaterThanOrEqual(hb.x - 0.5);
        expect(b.x + b.width, "+N inside band (right)").toBeLessThanOrEqual(hb.x + hb.width + 0.5);
        expect(b.y + b.height, "+N inside band (bottom)").toBeLessThanOrEqual(hb.y + hb.height + 0.5);
        // Potion sockets all still on screen.
        const sockets = hud.locator(".gs-hud-socket");
        const ns = await sockets.count();
        expect(ns).toBeGreaterThan(0);
        for (let i = 0; i < ns; i++) {
          const sb = await realBox(sockets.nth(i));
          expect(sb.x + sb.width, `socket #${i} inside band`).toBeLessThanOrEqual(hb.x + hb.width + 0.5);
          expect(sb.x, `socket #${i} on screen`).toBeGreaterThanOrEqual(0);
        }
        await more.click({ timeout: 5000 });
        await expect(page.locator(".gs-inspect")).toBeVisible({ timeout: 5000 });
        await expect(page.locator(".gs-inspect-list > *")).toHaveCount(14 - nShown, { timeout: 5000 });
      });
    });
  }
}
