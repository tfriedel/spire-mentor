import { test, expect, type Page } from "@playwright/test";
import {
  PHONES, fontPx, openPreview, phoneContext, realBox, shortLandscapeContext, tapSlot, type Orientation, type PhoneSize,
} from "./phoneHarness";
import { splitBandReal, splitZoomRise, splitZoomWidth } from "../src/portraitSplit";
import { portraitZoomLeft } from "../src/zoomPlacement";

/**
 * The combat board in phone mode, on the backend-free preview fixture.
 *
 * Phone = `(hover: none)` AND (`max-width: 480px` OR `max-height: 520px`). Each
 * describe is one of {large, small} x {portrait, landscape}. Later tasks add
 * their layout assertions here; these are the baseline every phone must hold.
 */
const PREVIEW = "enemies=2&enemy=cultists";

/** The zoom's width per `zoomPlacement`: 170 landscape (160 on the small
 *  phone); portrait, the split's `splitZoomWidth`, 158 (150 on the small
 *  phone; spec "Phase 2: portrait split", "The zoomed card"). */
function zoomWidthFor(size: PhoneSize, orientation: Orientation): number {
  if (orientation === "portrait") return splitZoomWidth(size === "small");
  return size === "small" ? 160 : 170;
}

/** Real px from the stage bottom the portrait zoom's top may reach: the band
 *  (296, small 264) plus the card's rise above it (0, small 20). */
const zoomCeiling = (size: PhoneSize) => splitBandReal(size === "small") + splitZoomRise(size === "small");

type R = { left: number; top: number; right: number; bottom: number };
const overlaps = (z: { x: number; y: number; width: number; height: number }, e: R) =>
  z.x < e.right - 0.5 && e.left < z.x + z.width - 0.5 && z.y < e.bottom - 0.5 && e.top < z.y + z.height - 0.5;


/** The zoom's box once its short entry animation (a scale-in) has settled. */
async function zoomBox(page: Page) {
  const z = page.locator(".gs-zoomcard");
  await z.evaluate((el) => Promise.all(el.getAnimations().map((an) => an.finished)));
  return realBox(z);
}

/** Real-px rects: the stage, the hand, each enemy as a tap sees it (its
 *  body, intent chip and power rows) and each enemy's body alone (what an
 *  armed attack's tap aims at, which the zoom keeps clear of). */
async function geometry(page: Page): Promise<{ stage: R; hand: R; enemies: R[]; bodies: R[] }> {
  return page.evaluate(() => {
    const r = (el: Element) => { const b = el.getBoundingClientRect(); return { left: b.left, top: b.top, right: b.right, bottom: b.bottom }; };
    const enemies = Array.from(document.querySelectorAll("[data-enemy-index]")).map((el) => {
      const rs = [el, ...Array.from(el.querySelectorAll(".gs-intent, .gs-powers"))].map(r)
        .filter((x) => x.right > x.left && x.bottom > x.top);
      return { left: Math.min(...rs.map((x) => x.left)), top: Math.min(...rs.map((x) => x.top)),
               right: Math.max(...rs.map((x) => x.right)), bottom: Math.max(...rs.map((x) => x.bottom)) };
    });
    const bodies = Array.from(document.querySelectorAll("[data-enemy-index]")).map(r);
    return { stage: r(document.querySelector(".gs-arena-stage")!), hand: r(document.querySelector(".gs-hand")!), enemies, bodies };
  });
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  for (const orientation of ["portrait", "landscape"] as Orientation[]) {
    test.describe(`${size} phone, ${orientation}`, () => {
      test.use(phoneContext(size, orientation));

      test("board is in phone mode", async ({ page }) => {
        await openPreview(page, PREVIEW);
        // Verify the emulation actually yields a touch device rather than assuming.
        expect(await page.evaluate(() => matchMedia("(hover: none)").matches)).toBe(true);
        const board = page.locator(".gs-board");
        await expect(board).toHaveClass(/(^|\s)gs-phone(\s|$)/);
        await expect(board).toHaveClass(new RegExp(`gs-phone-${orientation}`));
        const other = orientation === "portrait" ? "landscape" : "portrait";
        await expect(board).not.toHaveClass(new RegExp(`gs-phone-${other}`));
      });

      test("no horizontal overflow", async ({ page }) => {
        await openPreview(page, PREVIEW);
        const m = await page.evaluate(() => ({
          w: document.documentElement.scrollWidth, vw: window.innerWidth,
        }));
        expect(m.w).toBeLessThanOrEqual(m.vw);
      });

      test("stage controls meet the floors", async ({ page }) => {
        await openPreview(page, "enemies=2&powers=3");
        const power = page.locator(".gs-power").first();
        const pb = await realBox(power);
        expect(pb.width).toBeGreaterThanOrEqual(43.5);
        expect(pb.height).toBeGreaterThanOrEqual(43.5);
        expect((await realBox(power.locator("img"))).width).toBeGreaterThanOrEqual(27.5);
        expect(await fontPx(page.locator(".gs-power-amt").first())).toBeGreaterThanOrEqual(13.9);
        expect(await fontPx(page.locator(".gs-hptext").first())).toBeGreaterThanOrEqual(13.9);
        expect(await fontPx(page.locator(".gs-nameplate").first())).toBeGreaterThanOrEqual(11.9);
        // Block badge: the player's block is 12 in the fixture.
        expect((await realBox(page.locator(".gs-block").first())).width).toBeGreaterThanOrEqual(27.5);
        expect(await fontPx(page.locator(".gs-block-num").first())).toBeGreaterThanOrEqual(13.9);
        for (const count of await page.locator(".gs-pileplaque-count span").all()) {
          expect(await fontPx(count)).toBeGreaterThanOrEqual(13.9);
        }
        const intent = await realBox(page.locator(".gs-intent").first());
        expect(intent.width).toBeGreaterThanOrEqual(43.5);
        expect(intent.height).toBeGreaterThanOrEqual(43.5);
        expect((await realBox(page.locator(".gs-energy").first())).width).toBeGreaterThanOrEqual(59.5);
        const end = await realBox(page.locator(".gs-endturn"));
        expect(end.width).toBeGreaterThanOrEqual(123.5);
        expect(end.height).toBeGreaterThanOrEqual(55.5);
        for (const pile of await page.locator(".gs-pilebtn").all()) {
          const b = await realBox(pile);
          expect(b.width).toBeGreaterThanOrEqual(43.5);
          expect(b.height).toBeGreaterThanOrEqual(43.5);
        }
      });

      test("intent modifier icons meet the floor", async ({ page }) => {
        await openPreview(page, "enemies=2&mods=1");
        const icon = page.locator(".gs-intent-mod-icon").first();
        const b = await realBox(icon);
        expect(b.width).toBeGreaterThanOrEqual(19.5);
        expect(b.height).toBeGreaterThanOrEqual(19.5);
      });

      for (const q of ["enemies=2&enemy=cultists&powers=3", "enemies=2&enemy=cultists&powers=8"]) {
        test(`power chips stay on the stage (${q})`, async ({ page }) => {
          await openPreview(page, q);
          const stage = await realBox(page.locator(".gs-arena-stage"));
          const chips = await page.locator(".gs-power, .gs-power-more").all();
          expect(chips.length).toBeGreaterThan(0);
          for (const chip of chips) {
            const b = await realBox(chip);
            expect(b.x).toBeGreaterThanOrEqual(stage.x - 1);
            expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
            expect(b.x + b.width).toBeLessThanOrEqual(stage.x + stage.width + 1);
            expect(b.y + b.height).toBeLessThanOrEqual(stage.y + stage.height + 1);
          }
        });
      }

      test("fallback power chips meet the text floor", async ({ page }) => {
        await openPreview(page, "enemies=2&powers=3");
        // The fixture's powers all have art, so strip the images to take the
        // no-art path through the same component: render one chip directly.
        await page.evaluate(() => {
          const host = document.querySelector(".gs-powers")!;
          const chip = document.createElement("span");
          chip.className = "gs-power-name";
          chip.textContent = "Fallback";
          chip.style.fontSize = "10px";
          host.appendChild(chip);
        });
        expect(await fontPx(page.locator(".gs-power-name").first())).toBeGreaterThanOrEqual(12.9);
      });

      test(`eight powers collapse into ${orientation === "landscape" ? "a column" : "the fight's chrome"} and a +N chip`, async ({ page }) => {
        await openPreview(page, "enemies=2&powers=8");
        expect(await page.locator(".gs-power").count()).toBeGreaterThanOrEqual(2);
        const more = page.locator(".gs-power-more").first();
        await expect(more).toBeVisible();
        await expect(more).toHaveText(/^\+\d+$/);
        if (orientation === "portrait") {
          // The split: each enemy's block is what the fight's chrome mode
          // draws; the player's band, one row.
          const chrome = (await page.locator(".gs-board").getAttribute("data-split"))?.split(" ")[0];
          expect(chrome).toMatch(/^(rows|column|compact)$/);
          const band = page.locator(".gs-playerband .gs-powers");
          await expect(band).toHaveCount(1);
          const bb = await realBox(band);
          expect(bb.height, "the band's powers are one row").toBeLessThanOrEqual((await realBox(band.locator(".gs-power").first())).height + 1);
          const blocks = await page.locator("[data-enemy-index] .gs-powers").all();
          expect(blocks.length).toBe(2);
          for (const block of blocks) {
            const b = await realBox(block);
            const cell = await realBox(block.locator(".gs-power, .gs-power-more").first());
            const cells = await block.locator(".gs-power, .gs-power-more").count();
            if (chrome === "rows") {
              expect(b.height, "rows: two rows at most").toBeLessThanOrEqual(cell.height * 2 + 12);
            } else if (chrome === "column") {
              expect(b.width, "column: one column").toBeLessThanOrEqual(cell.width + 1);
            } else {
              expect(cells, "compact: one cell").toBe(1);
              await expect(block.locator(".gs-power-more")).toHaveText("+8");
            }
          }
          return;
        }
        for (const block of await page.locator(".gs-powers").all()) {
          const b = await realBox(block);
          const one = await realBox(block.locator(".gs-power").first());
          if (orientation === "landscape") {
            // One column: three powers before the "+N" chip where the stage
            // holds four cells; the small phone's (as tall as in the real app
            // since the tool row keeps its height) holds three: two and "+N".
            expect(b.width).toBeLessThanOrEqual(one.width + 1);
            expect(await block.locator(".gs-power").count()).toBeGreaterThanOrEqual(size === "small" ? 2 : 3);
            await expect(block.locator(".gs-power-more")).toHaveCount(1);
          } else {
            expect(b.height).toBeLessThanOrEqual(one.height * 2 + 12);
          }
        }
      });

      // The owner: "when multiple enemies were shown the icons below them could
      // cause a mess". Portrait: each creature's power block stays inside its
      // own column (half-way to each neighbour), never on a neighbour's block,
      // HP bar or name; it collapses into "+N" earlier when that column is
      // narrow (one power and the chip at the least, three where there is room).
      if (orientation === "portrait") {
        for (const n of [2, 3, 4, 5]) {
          for (const pw of [3, 8]) {
            const q = `enemies=${n}&powers=${pw}`;
            test(`power blocks keep to their own creature's column (${q})`, async ({ page }) => {
              await openPreview(page, q);
              const cs = await page.evaluate(() => {
                const r = (el: Element | null) => {
                  if (!el) return null;
                  const b = el.getBoundingClientRect();
                  return b.width > 0 ? { left: b.left, top: b.top, right: b.right, bottom: b.bottom } : null;
                };
                // The portrait split's player figure is the body alone: its
                // powers are in the band beside it, not under its feet.
                return Array.from(document.querySelectorAll(".gs-combatant:not(.gs-split [data-combatant=\"player\"])")).map((c) => {
                  const chips = Array.from(c.querySelectorAll(".gs-power, .gs-power-more")).map(r)
                    .filter((x): x is NonNullable<typeof x> => !!x);
                  const block = chips.length ? {
                    left: Math.min(...chips.map((x) => x.left)), top: Math.min(...chips.map((x) => x.top)),
                    right: Math.max(...chips.map((x) => x.right)), bottom: Math.max(...chips.map((x) => x.bottom)),
                  } : null;
                  const more = c.querySelector(".gs-power-more")?.textContent ?? null;
                  // Distinct cell columns and rows, 4 px tolerance.
                  const distinct = (vs: number[]) => vs.sort((a, b) => a - b)
                    .filter((v, i, a) => i === 0 || v - a[i - 1] > 4).length;
                  // The solver's slot (real px) and the stage zoom: what a
                  // `rows` block's cells across are computed from.
                  const slot = parseFloat(c.closest(".gs-side")?.getAttribute("data-split-slot") ?? "NaN");
                  const k = parseFloat(getComputedStyle(document.querySelector(".gs-arena-stage")!).getPropertyValue("--gs-k")) || 1;
                  return { block, cells: chips.length, more, slot, k, box: r(c)!, hp: r(c.querySelector(".gs-hpwrap")),
                    across: distinct(chips.map((x) => x.left)), rows: distinct(chips.map((x) => x.top)),
                    name: r(c.querySelector(".gs-nameplate")) };
                }).sort((a, b) => a.box.left - b.box.left);
              });
              expect(cs.length).toBe(n);
              const chrome = (await page.locator(".gs-board").getAttribute("data-split"))?.split(" ")[0];
              console.log(`[blocks] ${size} ${q}: ${chrome}; cells ${cs.map((c) => `${c.cells}/${c.across}x${c.rows}`).join(",")}`);
              type B = { left: number; top: number; right: number; bottom: number };
              const hitB = (a: B, b: B) => a.left < b.right - 0.5 && b.left < a.right - 0.5
                && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;
              for (const [i, c] of cs.entries()) {
                expect(c.block, `creature ${i} shows its powers`).not.toBeNull();
                // Exactly what the fight's chrome mode draws (`data-split`):
                // `rows` fills two rows of its cells across before the "+N"
                // chip, `column` one column of at least one power and the
                // chip, `compact` one cell ("+N" listing them all).
                if (chrome === "rows") {
                  // Cells across from the slot, not from what was drawn: the
                  // grid's real cell (a 44 px tap box at least) and gap.
                  expect(Number.isFinite(c.slot), `creature ${i} has a solver slot`).toBe(true);
                  const cell = Math.max(26 * c.k, 44);
                  const across = Math.max(1, Math.floor((c.slot + 5 * c.k) / (cell + 5 * c.k) + 1e-6));
                  expect(across, `creature ${i}: a ${c.slot} px slot holds two cells`).toBeGreaterThanOrEqual(2);
                  expect(c.rows, `creature ${i}: two rows at most`).toBeLessThanOrEqual(2);
                  expect(c.across, `creature ${i}: ${across} across`).toBe(Math.min(across, c.cells));
                  expect(c.cells, `creature ${i}: two full rows of ${across} before the chip`)
                    .toBe(Math.min(pw, 2 * across));
                } else if (chrome === "column") {
                  expect(c.across, `creature ${i}: one column`).toBe(1);
                  expect(c.cells, `creature ${i}: at least one power and the chip`).toBeGreaterThanOrEqual(Math.min(2, pw));
                } else {
                  expect(chrome, "the chrome mode").toBe("compact");
                  expect(c.cells, `creature ${i}: one cell`).toBe(1);
                }
                // Nothing lost: the chip counts exactly the powers not drawn.
                if (c.cells < pw) expect(c.more, `creature ${i}'s chip`).toBe(`+${pw - (c.cells - 1)}`);
                else expect(c.more, `creature ${i}: no chip with every power drawn`).toBeNull();
                for (const [j, o] of cs.entries()) {
                  if (j === i) continue;
                  if (o.block) expect(hitB(c.block!, o.block), `block ${i} on block ${j}`).toBe(false);
                  if (o.hp) expect(hitB(c.block!, o.hp), `block ${i} on HP bar ${j}`).toBe(false);
                  if (o.name) expect(hitB(c.block!, o.name), `block ${i} on name ${j}`).toBe(false);
                }
              }
            });
          }
        }
      }

      if (orientation === "landscape") {
        for (const hand of [5, 10]) {
          test(`the zoom passes strip taps to the slots under it (hand=${hand}, one enemy)`, async ({ page }) => {
            test.setTimeout(240_000);
            await openPreview(page, `enemies=1&enemy=cultists&hand=${hand}`);
            let under = 0;
            for (const sel of [2, 4]) under += await strayTapsSwitch(page, sel);
            console.log(`[passthrough] ${size} landscape hand=${hand}: ${under} taps under the zoom`);
            await zoomAboveStripPlays(page, 4);
          });
        }

        // ---- Landscape: powers in a column BESIDE each creature ----
        type Box = { left: number; top: number; right: number; bottom: number };
        const hit = (a: Box, b: Box) => a.left < b.right - 0.5 && b.left < a.right - 0.5
          && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;
        /** Per creature: its column, its body (the sprite box), its intent
         *  (icon and numeral). Real px. */
        const columns = (page: Page) => page.evaluate(() => {
          const r = (el: Element | null) => {
            if (!el) return null;
            const b = el.getBoundingClientRect();
            return b.width > 0 ? { left: b.left, top: b.top, right: b.right, bottom: b.bottom } : null;
          };
          return Array.from(document.querySelectorAll(".gs-combatant")).map((c) => ({
            col: r(c.querySelector(".gs-powers")),
            cells: c.querySelectorAll(".gs-power, .gs-power-more").length,
            body: r(c.querySelector(":scope > .gs-sprite-wrap")),
            intents: [r(c.querySelector(".gs-intent")), r(c.querySelector(".gs-intent-dmg"))].filter(Boolean) as Box[],
            chrome: [r(c.querySelector(".gs-hpwrap")), r(c.querySelector(".gs-nameplate"))].filter(Boolean) as Box[],
          }));
        });
        for (const [n, pw] of [[2, 3], [2, 8], [3, 3], [3, 8], [4, 3], [4, 8], [5, 3], [5, 8]]) {
          const q = `enemies=${n}&powers=${pw}`;
          test(`power columns stand beside their creature and off the neighbours (${q})`, async ({ page }) => {
            await openPreview(page, q);
            const cs = await columns(page);
            expect(cs.length).toBe(n + 1);
            console.log(`[columns] ${size} ${q}: cells per creature ${cs.map((c) => c.cells).join(",")}`);
            const hand = await realBox(page.locator(".gs-hand"));
            const stage = await realBox(page.locator(".gs-arena-stage"));
            for (const [i, c] of cs.entries()) {
              expect(c.col, `creature ${i} has a column`).not.toBeNull();
              const col = c.col!;
              // Beside the body, not under it.
              expect(col.right <= c.body!.left + 0.5 || col.left >= c.body!.right - 0.5, `column ${i} beside its body`).toBe(true);
              expect(col.bottom).toBeLessThanOrEqual(hand.y + 1);
              expect(col.top).toBeGreaterThanOrEqual(stage.y - 1);
              // Three powers and the chip where the row has room; never under
              // one power and the chip.
              // (Five enemies on the small phone: the gaps between their intents
              // are narrower than a tap box, and a crowded creature's column is
              // one "+N" cell standing on its neighbour's intent.)
              expect(c.cells).toBeGreaterThanOrEqual(n <= 2 && size === "large" ? Math.min(4, pw)
                : size === "small" && n >= 5 ? 1 : 2);
              for (const [j, o] of cs.entries()) {
                if (j === i) continue;
                if (o.col) expect(hit(col, o.col), `column ${i} on column ${j}`).toBe(false);
                for (const it of o.intents) expect(hit(col, it), `column ${i} on intent ${j}`).toBe(false);
                for (const it of o.chrome) expect(hit(col, it), `column ${i} on HP bar/name ${j}`).toBe(false);
              }
            }
          });

          test(`every power chip takes its own tap (${q})`, async ({ page }) => {
            await openPreview(page, q);
            const chips = await page.locator(".gs-power, .gs-power-more").all();
            expect(chips.length).toBeGreaterThan(0);
            for (const chip of chips) {
              const b = await realBox(chip);
              expect(b.width).toBeGreaterThanOrEqual(43.5);
              expect(b.height).toBeGreaterThanOrEqual(43.5);
              const own = await chip.evaluate((el, [x, y]) => el.contains(document.elementFromPoint(x, y)),
                [b.x + b.width / 2, b.y + b.height / 2]);
              expect(own, "a chip's centre is under something else").toBe(true);
            }
            for (const amt of await page.locator(".gs-power-amt").all()) {
              expect(await fontPx(amt)).toBeGreaterThanOrEqual(12.9);
            }
          });
        }
      }

      // The block clears the resting hand: the peeking hand is a 96px strip and
      // the fighters stand high enough that two power rows end above it.
      test("powers clear the resting hand", async ({ page }) => {
        await openPreview(page, "enemies=2&powers=8");
        const hand = await realBox(page.locator(".gs-hand"));
        for (const block of await page.locator(".gs-powers").all()) {
          const b = await realBox(block);
          expect(b.y + b.height).toBeLessThanOrEqual(hand.y + 1);
        }
      });

      const T = { timeout: 5000 };
      const thing = (page: import("@playwright/test").Page, enemyOnly = false) =>
        page.locator(`${enemyOnly ? ".gs-side:not(.gs-side-player) " : ""}.gs-power`);

      test("tapping a power opens the inspect sheet", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await page.locator(".gs-power").first().tap(T);
        await expect(page.locator(".gs-inspect")).toContainText("Vulnerable");
        expect(await fontPx(page.locator(".gs-inspect .gs-inspect-name").first())).toBeGreaterThanOrEqual(13.9);
      });

      test("tapping the intent explains it", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&mods=1");
        const intent = page.locator(".gs-intent").first();
        const title = (await intent.getAttribute("title")) ?? "";
        expect(title).not.toBe("");
        await intent.tap(T);
        const sheet = page.locator(".gs-inspect");
        await expect(sheet.locator(".gs-inspect-name").first()).toHaveText(title.replace(/\n/g, " "), { useInnerText: true });
        await expect(sheet.locator(".gs-inspect-list > *")).toHaveCount(1);
      });

      test("an intent with no breakdown is named, not shown as its id", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&intent=DebuffStrong");
        await page.locator(".gs-intent").first().tap(T);
        const sheet = page.locator(".gs-inspect");
        await expect(sheet.locator(".gs-inspect-name").first()).toHaveText("Strategic");
        await expect(sheet.locator(".gs-inspect-desc").first()).toHaveText("This enemy intends to apply a powerful Debuff to you.");
        await expect(sheet).not.toContainText("DebuffStrong");
      });

      test("+N lists every hidden power", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=8");
        const more = page.locator(".gs-power-more").first();
        const hidden = Number((await more.innerText()).replace("+", ""));
        expect(hidden).toBeGreaterThan(0);
        await more.tap(T);
        await expect(page.locator(".gs-inspect .gs-inspect-list > *")).toHaveCount(hidden);
      });

      test("the sheet closes when the combat state moves on", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3&debuff=2500");
        await thing(page).first().tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
        await expect(page.locator(".gs-inspect")).toHaveCount(0, { timeout: 6000 });
      });

      test("the same power on another enemy switches the sheet", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3");
        const powersOf = (i: number) => page.locator(`.gs-combatant[data-enemy-index="${i}"] .gs-power`);
        // Landscape: the SECOND power (of three): on the small phone sideways
        // a column of three reaches the stage top, where the open sheet sits
        // over its first cell.
        const first = powersOf(0).nth(orientation === "landscape" ? 1 : 0);
        const label = (await first.getAttribute("aria-label"))!;
        await first.tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
        await powersOf(1).and(page.locator(`[aria-label="${label}"]`)).tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
      });

      test("no inspect sheet while a card is armed", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3&hand=3");
        // The last of three is Bash, an attack that needs a target, and the fan's top-most
        // slot (earlier slots' hit boxes are partly overlapped); two enemies stand.
        // The slot's `.gs-handslot-hit` overlay is the tap target, not the card.
        // On a phone ONE tap zooms the card and arms it.
        const hit = page.locator(".gs-handslot-hit").last();
        await hit.tap(T);
        await expect(page.locator(".gs-board[data-armed]")).toBeVisible();
        await thing(page, true).first().tap(T);
        await expect(page.locator(".gs-inspect")).toHaveCount(0);
      });

      test("the sheet never covers the hand", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=8");
        await page.locator(".gs-power-more").first().tap(T);
        const sheet = await realBox(page.locator(".gs-inspect"));
        const hand = await realBox(page.locator(".gs-hand"));
        const stage = await realBox(page.locator(".gs-arena-stage"));
        expect(sheet.y + sheet.height).toBeLessThanOrEqual(hand.y + 1);
        expect(sheet.x).toBeGreaterThanOrEqual(stage.x - 1);
        expect(sheet.x + sheet.width).toBeLessThanOrEqual(stage.x + stage.width + 1);
      });

      test("the sheet takes no layout height", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3");
        const before = await realBox(page.locator(".gs-arena-stage"));
        await thing(page).first().tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
        const after = await realBox(page.locator(".gs-arena-stage"));
        expect(after.height).toBe(before.height);
        expect(after.y).toBe(before.y);
      });

      test("tapping empty stage closes the sheet", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3");
        await thing(page).first().tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
        // The stage's top-left corner is room backdrop, under no creature or sheet's tap target.
        const stage = await realBox(page.locator(".gs-arena-stage"));
        const sheet = await realBox(page.locator(".gs-inspect"));
        await page.touchscreen.tap(stage.x + 2, Math.min(sheet.y + sheet.height + 6, stage.y + stage.height / 2));
        await expect(page.locator(".gs-inspect")).toHaveCount(0);
      });

      test("tapping the sheet closes it", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3");
        await thing(page).first().tap(T);
        await page.locator(".gs-inspect").tap(T);
        await expect(page.locator(".gs-inspect")).toHaveCount(0);
      });

      {
        // The landscape fit assumes a one-line name (phone.PHONE_BELOW_GROUND).
        // Portrait (the split): every fight whose power chrome is not
        // `compact` (a compact fight has no nameplate); the player figure has
        // none, its name is not on the board.
        for (const q of ["enemies=4", "enemies=4&enemy=cultists"]) {
          test(`names stay on one line (${q})`, async ({ page }) => {
            await openPreview(page, q);
            if (orientation === "portrait") {
              const chrome = (await page.locator(".gs-board").getAttribute("data-split"))?.split(" ")[0];
              test.skip(chrome === "compact", "a compact split fight draws no nameplates");
            }
            const stage = await realBox(page.locator(".gs-arena-stage"));
            const plates = await page.locator(".gs-nameplate").all();
            expect(plates.length).toBe(orientation === "portrait" ? 4 : 5);
            for (const plate of plates) {
              const { h, line } = await plate.evaluate((el) => {
                const cs = getComputedStyle(el);
                const fs = parseFloat(cs.fontSize);
                const lh = cs.lineHeight === "normal" ? fs * 1.2 : parseFloat(cs.lineHeight);
                return { h: (el as HTMLElement).offsetHeight, line: lh };
              });
              expect(h).toBeLessThan(1.6 * line);
              const b = await realBox(plate);
              expect(b.x).toBeGreaterThanOrEqual(stage.x - 1);
              expect(b.x + b.width).toBeLessThanOrEqual(stage.x + stage.width + 1);
              expect(b.y).toBeGreaterThanOrEqual(stage.y - 1);
              expect(b.y + b.height).toBeLessThanOrEqual(stage.y + stage.height + 1);
              expect(await fontPx(plate)).toBeGreaterThanOrEqual(11.9);
            }
          });
        }
      }

      // ---- The peeking hand (Task 5) ----
      const floorW = orientation === "landscape" ? 136 : PHONES[size].width < 390 ? 108 : 124;
      for (const hand of [5, 10]) {
        test(`resting cards are wide half-cards (hand=${hand})`, async ({ page }) => {
          await openPreview(page, `hand=${hand}`);
          const stage = await realBox(page.locator(".gs-arena-stage"));
          const lifts = await page.locator(".gs-handslot-lift").all();
          expect(lifts.length).toBe(hand);
          for (const lift of lifts) {
            const b = await realBox(lift);
            expect(b.width).toBeGreaterThanOrEqual(floorW - 0.5);
            const shown = stage.y + stage.height - b.y;
            expect(shown).toBeGreaterThanOrEqual(95.5);
            expect(shown).toBeLessThanOrEqual(97); // the band is pinned at 96, not the desktop ~110
            // The cost gem and the name banner are inside the visible part (the
            // gem overhangs the card's top edge a little; it is still on stage).
            for (const sel of [".gs-card-energy", ".gs-card-banner"]) {
              const part = await realBox(lift.locator(sel).first());
              expect(part.y).toBeGreaterThanOrEqual(stage.y);
              expect(part.y + part.height).toBeLessThanOrEqual(stage.y + stage.height + 0.5);
            }
          }
        });

        test(`End Turn, piles and orb clear the resting cards (hand=${hand})`, async ({ page }) => {
          await openPreview(page, `hand=${hand}`);
          const stage = await realBox(page.locator(".gs-arena-stage"));
          const bottom = stage.y + stage.height;
          const chrome = [page.locator(".gs-endturn"), page.locator(".gs-energy"),
            ...(await page.locator(".gs-pilebtn").all())];
          for (const lift of await page.locator(".gs-handslot-lift").all()) {
            const c = await realBox(lift);
            // Only the part of the card above the stage's bottom edge is drawn.
            const card = { x: c.x, y: c.y, w: c.width, b: Math.min(c.y + c.height, bottom) };
            for (const loc of chrome) {
              const o = await realBox(loc);
              const apart = o.x + o.width <= card.x + 0.5 || o.x >= card.x + card.w - 0.5
                || o.y + o.height <= card.y + 0.5 || o.y >= card.b - 0.5;
              expect(apart, `${await loc.getAttribute("class")} overlaps a resting card`).toBe(true);
            }
          }
        });
      }

      // The owner: "the icons for energy, discard etc almost touched the cards
      // in the hand". At least 12 real px between End Turn / the piles / the
      // orb and any resting card (a pulled plan card included), and 8 between
      // the controls themselves. Distance = the gap along the axis that
      // separates two boxes.
      for (const q of ["hand=5", "hand=10", "hand=5&pulled=1", "hand=10&pulled=1"]) {
        test(`the chrome keeps its distance from the cards and itself (${q})`, async ({ page }) => {
          await openPreview(page, q);
          const m = await page.evaluate(() => {
            const r = (el: Element) => { const b = el.getBoundingClientRect(); return { l: b.left, t: b.top, r: b.right, b: b.bottom }; };
            const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
            const chrome = Array.from(document.querySelectorAll(".gs-endturn, .gs-energy, .gs-pilebtn"))
              .map((el) => ({ name: (el.className as string).split(" ").slice(0, 2).join("."), ...r(el) }));
            const cards = Array.from(document.querySelectorAll(".gs-handslot-lift")).map((el) => {
              const x = r(el); return { ...x, b: Math.min(x.b, stage.bottom) };
            });
            return { chrome, cards };
          });
          type Bx = { l: number; t: number; r: number; b: number };
          const gap = (a: Bx, b: Bx) => Math.max(b.l - a.r, a.l - b.r, b.t - a.b, a.t - b.b);
          let minCard = Infinity, minSelf = Infinity;
          const bad: string[] = [];
          for (const c of m.chrome) {
            const toCard = Math.min(...m.cards.map((card) => gap(c, card)));
            minCard = Math.min(minCard, toCard);
            if (toCard < 11.5) bad.push(`${c.name} ${toCard.toFixed(1)}px from a card`);
            for (const o of m.chrome) {
              if (o === c) continue;
              const g = gap(c, o);
              minSelf = Math.min(minSelf, g);
              if (g < 7.5) bad.push(`${c.name} ${g.toFixed(1)}px from ${o.name}`);
            }
          }
          console.log(`[chrome gaps] ${size} ${orientation} ${q}: to cards ${minCard.toFixed(1)}, between ${minSelf.toFixed(1)}`);
          expect(bad, `${size} ${orientation} ${q}`).toEqual([]);
        });
      }

      // ---- The advice's plan cards (`.gs-pulled`) stay in the strip ----
      for (const hand of [5, 10]) {
        const q = `hand=${hand}&pulled=1`;
        test(`pulled cards stay in the strip, clear of the chrome (${q})`, async ({ page }) => {
          await openPreview(page, q);
          const stage = await realBox(page.locator(".gs-arena-stage"));
          const bottom = stage.y + stage.height;
          const pulled = page.locator(".gs-handslot.gs-pulled > .gs-handslot-lift");
          await expect(pulled).toHaveCount(2);
          const chrome = [page.locator(".gs-endturn"), page.locator(".gs-energy"),
            ...(await page.locator(".gs-pilebtn").all())];
          for (const lift of await page.locator(".gs-handslot-lift").all()) {
            const c = await realBox(lift);
            const isPulled = await lift.evaluate((el) => el.parentElement!.classList.contains("gs-pulled"));
            const shown = bottom - c.y;
            expect(shown).toBeGreaterThanOrEqual(95.5);
            expect(shown).toBeLessThanOrEqual(isPulled ? 96 + 12 : 97);
            const card = { x: c.x, y: c.y, w: c.width, b: Math.min(c.y + c.height, bottom) };
            for (const loc of chrome) {
              const o = await realBox(loc);
              const apart = o.x + o.width <= card.x + 0.5 || o.x >= card.x + card.w - 0.5
                || o.y + o.height <= card.y + 0.5 || o.y >= card.b - 0.5;
              expect(apart, `${await loc.getAttribute("class")} overlaps a ${isPulled ? "pulled" : "resting"} card`).toBe(true);
            }
          }
          // The numbered badge: inside the visible band, 13px+, and in the
          // strip of its card that no later card covers.
          for (const badge of await page.locator(".gs-handslot.gs-pulled .gs-step-tag").all()) {
            const b = await realBox(badge);
            expect(b.y).toBeGreaterThanOrEqual(bottom - 96 - 12);
            expect(b.y + b.height).toBeLessThanOrEqual(bottom);
            expect(await fontPx(badge)).toBeGreaterThanOrEqual(13);
            // (The badge takes no pointer, so elementFromPoint cannot be asked.)
            const covered = await badge.evaluate((el, [x, y]) => {
              const mine = el.closest(".gs-handslot")!;
              const slots = Array.from(document.querySelectorAll(".gs-handslot"));
              return slots.slice(slots.indexOf(mine) + 1).some((sl) => {
                const r = sl.querySelector(".gs-handslot-lift")!.getBoundingClientRect();
                return x > r.left && x < r.right && y > r.top && y < r.bottom;
              });
            }, [b.x + b.width / 2, b.y + b.height / 2]);
            expect(covered, "a later card covers the step badge").toBe(false);
          }
        });

        test(`the other cards stay flat beside a pulled card (${q})`, async ({ page }) => {
          await openPreview(page, q);
          const poses = await page.locator(".gs-handslot:not(.gs-pulled) > .gs-handslot-hit, .gs-handslot:not(.gs-pulled) > .gs-handslot-lift").evaluateAll(
            (els) => els.map((el) => {
              const t = getComputedStyle(el).transform;
              const m = new DOMMatrixReadOnly(t === "none" ? undefined : t);
              return { b: m.b, f: m.f };
            }));
          expect(poses.length).toBe(2 * (hand - 2));
          for (const m of poses) {
            expect(Math.abs(m.b)).toBeLessThan(1e-6);
            expect(Math.abs(m.f)).toBeLessThan(0.5);
          }
          const floorW = orientation === "landscape" ? 136 : PHONES[size].width < 390 ? 108 : 124;
          for (const lift of await page.locator(".gs-handslot-lift").all()) {
            expect((await realBox(lift)).width).toBeGreaterThanOrEqual(floorW - 0.5);
          }
        });
      }

      test("ten cards stay tappable", async ({ page }) => {
        await openPreview(page, "hand=10");
        const hits = await page.locator(".gs-handslot-hit").all();
        expect(hits.length).toBe(10);
        const boxes = await Promise.all(hits.map((h) => realBox(h)));
        // Each slot paints over the one before it, so a card's uncovered strip
        // is the distance to the next card's left edge (the last is whole).
        for (let i = 0; i < boxes.length - 1; i++) {
          expect(boxes[i + 1].x - boxes[i].x).toBeGreaterThanOrEqual(24);
        }
        // And a tap on the middle of each strip lands on that card's slot.
        for (let i = 0; i < boxes.length; i++) {
          const right = i < boxes.length - 1 ? boxes[i + 1].x : boxes[i].x + boxes[i].width;
          const x = (boxes[i].x + right) / 2;
          const y = boxes[i].y + 40;
          const hitIndex = await page.evaluate(([px, py]) => {
            const el = document.elementFromPoint(px, py);
            const slots = Array.from(document.querySelectorAll(".gs-handslot"));
            return slots.findIndex((sl) => sl.contains(el));
          }, [x, y]);
          expect(hitIndex).toBe(i);
        }
      });

      // The owner: "when clicking on cards in the hand ... the cards to the
      // right of the clicked would flicker". Every slot is recorded on every
      // animation frame for 600ms after a tap: no slot but the tapped one may
      // move, change opacity, z-order or class, and the tapped one only lifts,
      // in ONE step (a .14s transition under its overlapping neighbours made
      // the browser re-layer them, which is the flicker).
      for (const hand of [5, 10]) {
        test(`a tap moves only the tapped card, in one step (hand=${hand})`, async ({ page }) => {
          // Portrait draws the fight with the split's larger enemies, and two
          // Cultists that size cost headless Chromium's software WebGL ~110 ms
          // a frame: too few frames to watch the slots. Nibbits are cheap; the
          // hand is the same.
          const foe = orientation === "portrait" ? "nibbit" : "cultists";
          await openPreview(page, `enemies=2&enemy=${foe}&hand=${hand}`);
          const target = hand === 5 ? 1 : 4;
          await page.evaluate(() => {
            const w = window as unknown as { __frames: string[][] };
            w.__frames = [];
            const snap = () => Array.from(document.querySelectorAll(".gs-handslot")).map((s) => {
              const lift = s.querySelector(".gs-handslot-lift")!;
              const card = s.querySelector(".gs-card")!;
              const r = lift.getBoundingClientRect();
              const cs = getComputedStyle(s);
              return [r.left.toFixed(1), r.top.toFixed(1), r.width.toFixed(1), r.height.toFixed(1), cs.zIndex,
                cs.transform, getComputedStyle(lift).transform, getComputedStyle(card).opacity,
                getComputedStyle(lift).opacity, s.className].join("|");
            });
            const t0 = performance.now();
            const loop = () => { w.__frames.push(snap()); if (performance.now() - t0 < 800) requestAnimationFrame(loop); };
            requestAnimationFrame(loop);
          });
          await page.waitForTimeout(120);
          const before = await page.evaluate(() => (window as unknown as { __frames: string[][] }).__frames.length);
          await tapSlot(page, target);
          await page.waitForTimeout(800);
          const frames = await page.evaluate(() => (window as unknown as { __frames: string[][] }).__frames);
          expect(frames.length, "frames recorded after the tap").toBeGreaterThan(before + 10);
          const base = frames[before - 1];
          const seen = new Set<string>();
          for (const f of frames.slice(before)) {
            f.forEach((v, i) => {
              if (i === target) { seen.add(v.split("|").slice(0, 7).join("|")); return; }
              expect(v, `slot ${i} changed after a tap on ${target}`).toBe(base[i]);
            });
          }
          // The tapped slot: at most one pose before and one after, the
          // second only higher (same x and size).
          const poses = [...seen];
          expect(poses.length, `tapped slot poses: ${poses.join(" / ")}`).toBeLessThanOrEqual(2);
          const [l0, t0, w0, h0] = base[target].split("|").map(Number);
          for (const p of poses) {
            const [l, t, w, h] = p.split("|").map(Number);
            expect(Math.abs(l - l0)).toBeLessThan(0.2);
            expect(Math.abs(w - w0)).toBeLessThan(0.2);
            expect(Math.abs(h - h0)).toBeLessThan(0.2);
            expect(t).toBeLessThanOrEqual(t0 + 0.05);
          }
          await expect(page.locator(".gs-handslot.gs-selected")).toHaveCount(1);
        });
      }

      test("the fan is flat on a phone", async ({ page }) => {
        await openPreview(page, "hand=10");
        const poses = await page.locator(".gs-handslot-hit, .gs-handslot-lift").evaluateAll(
          (els) => els.map((el) => {
            const t = getComputedStyle(el).transform;
            const m = new DOMMatrixReadOnly(t === "none" ? undefined : t);
            return { b: m.b, f: m.f };
          }));
        expect(poses.length).toBe(20);
        for (const m of poses) {
          expect(Math.abs(m.b)).toBeLessThan(1e-6); // no rotation
          expect(Math.abs(m.f)).toBeLessThan(0.5); // no arc drop
        }
      });

      // ---- zoom-on-select and two-tap play (handSelect.ts, ZoomCard) ----
      // Preview hand: 0 Iron Wave (attack), 1 Shrug It Off (skill), 2 Bash
      // (attack), 3 Pommel Strike (attack), 4 Bloodletting (skill). The fixture
      // records a play in `data-last-play` ("card" or "card>target") and takes
      // the card out of the hand.
      const zoomW = zoomWidthFor(size, orientation);
      const zoom = (page: Page) => page.locator(".gs-zoomcard");
      // The tool row's "select a target…" is not shown on a phone.
      const armed = (page: Page) => page.locator(".gs-board[data-armed]");
      const lastPlay = (page: Page) => page.locator("[data-last-play]");

      test("first tap zooms and arms, enemy tap plays", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await expect(zoom(page)).toBeVisible();
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "2");
        expect((await zoomBox(page)).width).toBeGreaterThanOrEqual(Math.min(zoomW, 150) - 0.5);
        await expect(armed(page)).toBeVisible();
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await page.locator('.gs-combatant[data-enemy-index="1"]').tap(T);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "2>1");
        // The card left the hand, so its zoom and the targeting went with it.
        await expect(zoom(page)).toHaveCount(0);
        await expect(armed(page)).toHaveCount(0);
      });

      test("the zoomed card's text meets the floor", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await zoomBox(page); // the scale-in animation would shrink the reading
        expect(await fontPx(zoom(page).locator(".gs-card-desc"))).toBeGreaterThanOrEqual(13);
        expect(await fontPx(zoom(page).locator(".gs-card-title"))).toBeGreaterThanOrEqual(13);
      });

      test("a skill plays on the second tap of the card", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 1);
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "1");
        await expect(armed(page)).toHaveCount(0);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await tapSlot(page, 1);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "1");
        await expect(zoom(page)).toHaveCount(0);
      });

      test("tapping the zoom is the second tap", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 4);
        await zoom(page).tap(T);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "4");
        // An armed attack with two enemies: tapping its zoom does not play it.
        await tapSlot(page, 2);
        await zoom(page).tap(T);
        await expect(armed(page)).toBeVisible();
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "4");
      });

      test("one enemy: the first tap zooms, the second plays on it", async ({ page }) => {
        await openPreview(page, "enemies=1&enemy=cultists");
        await tapSlot(page, 2);
        await expect(zoom(page)).toBeVisible();
        await expect(armed(page)).toBeVisible();
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await tapSlot(page, 2);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "2>0");
      });

      // The owner: "cards when zoomed in were not always in the same place".
      // ONE spot per orientation for every card, attack or skill, whatever the
      // enemies (zoomPlacement.ts). Landscape: right of the orb column on the
      // player's side; it covers no enemy here, where it would it shrinks in
      // place (unit-tested). Portrait (spec "Phase 2: portrait split", "The
      // zoomed card"): bottom-left over the player's band, off every enemy.
      for (const q of ["enemies=1&enemy=cultists", "enemies=2&enemy=cultists", "enemies=2&enemy=cultists&powers=8",
        "enemies=3&enemy=cultists", "enemies=4", "enemies=5", "char=necrobinder&osty=1&enemies=2"]) {
        test(`the zoom stands on one fixed spot for every card (${q})`, async ({ page }) => {
          await openPreview(page, q);
          const boxes: { x: number; y: number; width: number; height: number }[] = [];
          for (const i of [0, 1, 2, 3, 4]) {
            await tapSlot(page, i);
            await expect(zoom(page)).toHaveAttribute("data-zoom-index", String(i));
            await page.waitForTimeout(100);
            boxes.push(await zoomBox(page));
          }
          const z = boxes[0];
          for (const [i, b] of boxes.entries()) {
            for (const k of ["x", "y", "width", "height"] as const) {
              expect(Math.abs(b[k] - z[k]), `card ${i} ${k}: ${b[k]} vs ${z[k]}`).toBeLessThan(0.75);
            }
          }
          const g = await geometry(page);
          const covers = (await zoom(page).getAttribute("data-zoom-covers")) === "1";
          console.log(`[zoom spot] ${size} ${orientation} ${q}: x=${z.x.toFixed(1)} y=${(z.y - g.stage.top).toFixed(1)} w=${z.width.toFixed(1)} covers=${covers}`);
          // Readable: never under the 150px floor (13px title).
          expect(z.width).toBeGreaterThanOrEqual(149.5);
          expect(z.x).toBeGreaterThanOrEqual(g.stage.left - 0.5);
          expect(z.x + z.width).toBeLessThanOrEqual(g.stage.right + 0.5);
          if (orientation === "portrait") {
            // Horizontally centred, or a little left of centre where the
            // status block needs the room (`portraitZoomLeft`).
            expect(Math.abs(z.x - (g.stage.left + portraitZoomLeft(g.stage.right - g.stage.left, z.width)))).toBeLessThan(0.75);
            expect(covers, "data-zoom-covers in portrait").toBe(false);
            expect(Math.abs(z.width - zoomWidthFor(size, orientation))).toBeLessThan(0.75);
            expect(z.y + z.height).toBeLessThanOrEqual(g.hand.top + 24.5);
            if (size === "small") expect(z.y).toBeGreaterThanOrEqual(g.stage.bottom - zoomCeiling(size) - 0.5);
            // Off every enemy body (its drawn rig: the enemy's box also holds
            // its name, whose foot the small phone's zoom may cover), intent
            // and HP bar.
            const { hit, seen, minLeft } = await page.evaluate((zz) => {
              const out: string[] = [];
              let seenN = 0;
              document.querySelectorAll("[data-enemy-index]").forEach((e) => {
                const i = e.getAttribute("data-enemy-index");
                [...Array.from(e.querySelectorAll(":scope > .gs-sprite-wrap")).map((n) => [n, "body"]),
                  ...Array.from(e.querySelectorAll(".gs-intent")).map((n) => [n, "intent"]),
                  ...Array.from(e.querySelectorAll(".gs-hpwrap")).map((n) => [n, "hp"])].forEach(([n, what]) => {
                  const b = (n as Element).getBoundingClientRect();
                  if (b.right <= b.left || b.bottom <= b.top) return;
                  seenN++;
                  if (zz.x < b.right - 0.5 && b.left < zz.x + zz.width - 0.5 && zz.y < b.bottom - 0.5 && b.top < zz.y + zz.height - 0.5) {
                    out.push(`e${i} ${what as string} ${JSON.stringify(b)}`);
                  }
                });
              });
              // The zoom's leftmost painted part (the cost gem stands proud).
              let left = Infinity;
              document.querySelectorAll(".gs-zoomcard, .gs-zoomcard *").forEach((el) => {
                const b = el.getBoundingClientRect();
                if (b.width > 0 && b.height > 0) left = Math.min(left, b.left);
              });
              return { hit: out, seen: seenN, minLeft: left };
            }, z);
            expect(seen, "enemy bodies/intents/HP bars collected").toBeGreaterThan(0);
            expect(minLeft, "the cost gem stays on the stage").toBeGreaterThanOrEqual(g.stage.left - 0.5);
            if (size === "small" && /enemies=5/.test(q)) {
              console.log(`[zoom spot] ${size} ${q}: covers ${hit.length ? hit.join("; ") : "no enemy chrome"}`);
            } else {
              expect(hit, "enemy chrome under the zoom").toEqual([]);
            }
            // End Turn and the energy numeral stay clear.
            for (const sel of [".gs-endturn", ".gs-energy .gs-gem-val"]) {
              const c = await realBox(page.locator(sel));
              expect(overlaps(z, { left: c.x, top: c.y, right: c.x + c.width, bottom: c.y + c.height }), sel).toBe(false);
            }
          } else {
            const intents = g.enemies.filter((e) => overlaps(z, e)).length;
            console.log(`[zoom spot] ${size} ${orientation} ${q}: covers part of ${intents}/${g.enemies.length} enemies' intent/powers`);
            if (!covers) for (const e of g.bodies) expect(overlaps(z, e), `covers an enemy ${JSON.stringify(e)}`).toBe(false);
            // The card's own top edge sits 12px under the stage top (its box
            // starts a few px higher: the cost gem stands proud of it).
            expect(z.y).toBeGreaterThanOrEqual(g.stage.top - 0.5);
            expect(z.y).toBeLessThanOrEqual(g.stage.top + 12.5);
            const orb = await realBox(page.locator(".gs-energy"));
            expect(z.x).toBeGreaterThanOrEqual(orb.x + orb.width - 0.5);
            expect(z.x + z.width / 2).toBeLessThan((g.stage.left + g.stage.right) / 2);
            expect(z.y + z.height).toBeLessThanOrEqual(g.stage.bottom + 0.5);
          }
        });
      }

      test("the zoom's close button puts the card back, plays nothing", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        for (const i of [2, 1]) { // an armed attack, then a skill
          await tapSlot(page, i);
          await expect(zoom(page)).toHaveAttribute("data-zoom-index", String(i));
          await zoomBox(page);
          const close = page.getByRole("button", { name: "Close", exact: true });
          await expect(close).toHaveCount(1);
          const c = await realBox(close);
          expect(c.width).toBeGreaterThanOrEqual(43.5);
          expect(c.height).toBeGreaterThanOrEqual(43.5);
          const st = await realBox(page.locator(".gs-arena-stage"));
          expect(c.x).toBeGreaterThanOrEqual(st.x - 0.5);
          expect(c.y).toBeGreaterThanOrEqual(st.y - 0.5);
          expect(c.x + c.width).toBeLessThanOrEqual(st.x + st.width + 0.5);
          // At the zoom's top-right corner.
          const z = await zoomBox(page);
          expect(Math.abs(c.x + c.width / 2 - (z.x + z.width))).toBeLessThan(23);
          if (orientation === "portrait") {
            // Spec "Phase 2: portrait split", "The zoomed card": the x lies
            // fully inside the stage and on neither End Turn nor the orb.
            expect(c.y + c.height).toBeLessThanOrEqual(st.y + st.height + 0.5);
            for (const sel of [".gs-endturn", ".gs-energy"]) {
              const o = await realBox(page.locator(sel));
              expect(overlaps(c, { left: o.x, top: o.y, right: o.x + o.width, bottom: o.y + o.height }), sel).toBe(false);
            }
          }
          await close.tap(T);
          await expect(zoom(page)).toHaveCount(0);
          await expect(armed(page)).toHaveCount(0);
          await expect(page.locator(".gs-handslot.gs-selected")).toHaveCount(0);
          await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
          await expect(page.locator(".gs-handslot")).toHaveCount(5);
        }
        if (orientation === "portrait") {
          // Both stay usable with the zoom open. The orb has no tap of its
          // own (no inspect sheet), so "usable" is: a tap at its centre lands
          // on the orb, not on the zoom. End Turn ends the turn.
          await tapSlot(page, 1);
          await zoomBox(page);
          const orbHit = await page.evaluate(() => {
            const b = document.querySelector(".gs-energy")!.getBoundingClientRect();
            const n = document.elementFromPoint((b.left + b.right) / 2, (b.top + b.bottom) / 2);
            return { orb: !!n?.closest(".gs-energy"), zoom: !!n?.closest(".gs-zoomcard") };
          });
          expect(orbHit).toEqual({ orb: true, zoom: false });
          await expect(zoom(page)).toHaveAttribute("data-zoom-index", "1");
          await page.locator(".gs-endturn").tap(T);
          await expect(page.locator("[data-last-play]")).toHaveAttribute("data-end-turns", "1");
          await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        }
      });

      // The zoom's fixed spot may cover an enemy (`data-zoom-covers`): the tap
      // goes at the covered part of each enemy when there is one.
      for (const q of ["enemies=3&enemy=cultists", "enemies=5&enemy=cultists", "enemies=2"]) {
        test(`an armed attack reaches every enemy, also one under the zoom (${q})`, async ({ page }) => {
          const n = Number(/enemies=(\d)/.exec(q)![1]);
          for (let idx = 0; idx < n; idx++) {
            await openPreview(page, q);
            await tapSlot(page, 2);
            await expect(armed(page)).toBeVisible();
            await zoomBox(page);
            const at = await page.evaluate((i) => {
              const z = document.querySelector(".gs-zoomcard")!.getBoundingClientRect();
              const e = document.querySelector(`.gs-combatant[data-enemy-index="${i}"]`)!.getBoundingClientRect();
              const l = Math.max(z.left, e.left), r = Math.min(z.right, e.right);
              const t = Math.max(z.top, e.top), b = Math.min(z.bottom, e.bottom);
              return r - l > 4 && b - t > 4
                ? { x: (l + r) / 2, y: (t + b) / 2, under: true }
                : { x: (e.left + e.right) / 2, y: (e.top + e.bottom) / 2, under: false };
            }, idx);
            await page.touchscreen.tap(at.x, at.y);
            await expect(lastPlay(page), `enemy ${idx}${at.under ? " (under the zoom)" : ""}`)
              .toHaveAttribute("data-last-play", `2>${idx}`);
          }
        });
      }

      test("a skill's zoom keeps off End Turn", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 1);
        const z = await zoomBox(page);
        const end = await realBox(page.locator(".gs-endturn"));
        expect(overlaps(z, { left: end.x, top: end.y, right: end.x + end.width, bottom: end.y + end.height })).toBe(false);
      });

      test("a zoomed skill keeps every intent it does not cover inspectable", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&powers=3");
        await tapSlot(page, 1);
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "1");
        await expect(armed(page)).toHaveCount(0);
        // The zoom has one fixed spot; in portrait it stands over the
        // player's band (spec "Phase 2: portrait split", "The zoomed card"),
        // so every intent stays free on every phone.
        const z = await zoomBox(page);
        const free: number[] = [];
        for (const i of [0, 1]) {
          const ib = await realBox(page.locator(`.gs-combatant[data-enemy-index="${i}"] .gs-intent`));
          if (!overlaps(z, { left: ib.x, top: ib.y, right: ib.x + ib.width, bottom: ib.y + ib.height })) free.push(i);
        }
        console.log(`[zoom intents] ${size} ${orientation}: ${free.length}/2 intents uncovered`);
        expect(free).toEqual([0, 1]);
        for (const i of free) {
          await page.locator(`.gs-combatant[data-enemy-index="${i}"] .gs-intent`).tap(T);
          await expect(page.locator(".gs-inspect")).toBeVisible();
          await page.locator(".gs-inspect").tap(T);
          await expect(page.locator(".gs-inspect")).toHaveCount(0);
        }
        // Inspecting neither played the skill nor dropped its zoom.
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "1");
      });

      test("a long card's zoomed text holds the floor and scrolls", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&longcard=1");
        await tapSlot(page, 1);
        const desc = zoom(page).locator(".gs-card-desc");
        await expect(desc).toContainText("Exhaust");
        await zoomBox(page); // the scale-in animation would shrink the reading
        expect(await fontPx(desc)).toBeGreaterThanOrEqual(13);
        const m = await desc.evaluate((el) => ({
          over: el.scrollHeight > el.clientHeight + 1, overflowY: getComputedStyle(el).overflowY,
        }));
        // Text that no longer fits at the floor scrolls rather than clips.
        if (m.over) expect(m.overflowY).toBe("auto");
        console.log(`[long card] ${size} ${orientation}: ${(await fontPx(desc)).toFixed(2)}px, scrolls=${m.over}`);
      });

      test("the zoom is a button named for what a tap does", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        // The button is the zoom's hit area (all of it in portrait, the part
        // above the hand strip in landscape).
        const btn = (page: Page) => zoom(page).locator(".gs-zoomcard-hit");
        await tapSlot(page, 1);
        await expect(btn(page)).toHaveAttribute("role", "button");
        await expect(btn(page)).toHaveAttribute("aria-label", /^Play Shrug It Off: Gain 8 Block\. Draw 1 card\.$/);
        await tapSlot(page, 2);
        await expect(btn(page)).toHaveAttribute("aria-label", /^Bash, zoomed: Deal 8 damage/);
        await expect(page.getByRole("button", { name: /^Bash, zoomed/ })).toHaveCount(1);
      });

      test("with a card selected, End Turn and the piles still work once", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        const root = page.locator("[data-last-play]");
        await tapSlot(page, 1);
        await page.locator(".gs-endturn").tap(T);
        await expect(root).toHaveAttribute("data-end-turns", "1");
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await tapSlot(page, 1);
        // The centred zoom (owner feedback round 2) no longer stands over the
        // piles in portrait: a pile takes its tap with the card still open.
        await page.locator(".gs-pile-draw").tap(T);
        await expect(page.locator(".gs-pileview")).toHaveCount(1);
        await expect(root).toHaveAttribute("data-end-turns", "1");
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
      });

      // Owner feedback round 5: with many cards the half-cards cannot be told
      // apart, so a thumb dragged along the strip shows each card in full.
      test("dragging a thumb along the hand browses it and plays nothing", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&hand=10");
        const hits = await page.locator(".gs-handslot-hit").evaluateAll((els) => els.map((e) => e.getBoundingClientRect().left));
        const stage = await realBox(page.locator(".gs-arena-stage"));
        const y = stage.y + stage.height - 30;
        const mid = (i: number) => (hits[i] + (i < hits.length - 1 ? hits[i + 1] : hits[i] + 30)) / 2;
        const cdp = await page.context().newCDPSession(page);
        const touch = (type: string, x: number, yy: number) => cdp.send("Input.dispatchTouchEvent",
          { type, touchPoints: type === "touchEnd" ? [] : [{ x, y: yy }] });
        await touch("touchStart", mid(1), y);
        const seen: string[] = [];
        for (let i = 1; i <= 8; i++) {
          // Several steps per card, drifting up over the zoom as a thumb does.
          for (const f of [0.34, 0.67, 1]) await touch("touchMove", mid(i - 1) + (mid(i) - mid(i - 1)) * f, y - i * 6);
          await expect(zoom(page)).toHaveAttribute("data-zoom-index", String(i));
          seen.push((await zoom(page).getAttribute("data-zoom-index"))!);
        }
        await touch("touchEnd", 0, 0);
        expect(seen).toEqual(["1", "2", "3", "4", "5", "6", "7", "8"]);
        // The card the thumb left stays open; nothing was played.
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "8");
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
        await expect(page.locator(".gs-handslot")).toHaveCount(10);
        // A plain tap still works afterwards.
        await tapSlot(page, 2);
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "2");
      });

      test("tapping empty stage cancels zoom and targeting", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await expect(armed(page)).toBeVisible();
        const stage = await realBox(page.locator(".gs-arena-stage"));
        // The stage's top corner on the far side from the zoom: in landscape
        // the zoom (and its cost gem, which stands proud of its left edge) can
        // sit right against the stage's left edge.
        const z = await zoomBox(page);
        const right = z.x + z.width / 2 < stage.x + stage.width / 2;
        await page.touchscreen.tap(right ? stage.x + stage.width - 4 : stage.x + 4, stage.y + 6);
        await expect(armed(page)).toHaveCount(0);
        await expect(zoom(page)).toHaveCount(0);
        await expect(page.locator(".gs-handslot.gs-selected")).toHaveCount(0);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
      });

      test("no targeting arrow on touch", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await expect(armed(page)).toBeVisible();
        const e = await realBox(page.locator('.gs-combatant[data-enemy-index="0"]'));
        await page.mouse.move(e.x + e.width / 2, e.y + e.height / 2);
        await page.waitForTimeout(200);
        await expect(page.getByTestId("aim-arrow")).toHaveCount(0);
      });

      test("rotating keeps zoom and armed card in step", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await expect(zoom(page)).toBeVisible();
        await expect(armed(page)).toBeVisible();
        const vp = page.viewportSize()!;
        await page.setViewportSize({ width: vp.height, height: vp.width });
        const turned = orientation === "portrait" ? "landscape" : "portrait";
        await expect(page.locator(".gs-board")).toHaveClass(new RegExp(`gs-phone-${turned}`));
        await page.waitForTimeout(600);
        await expect(zoom(page)).toBeVisible();
        await expect(armed(page)).toBeVisible();
        // Placement was recomputed for the new layout.
        const z = await zoomBox(page);
        expect(z.width).toBeGreaterThanOrEqual(Math.min(zoomWidthFor(size, turned), 150) - 0.5);
        const g = await geometry(page);
        // Clear of every enemy body unless the fixed spot cannot be (small
        // portrait: it then stays put at its narrowest and says so).
        if ((await zoom(page).getAttribute("data-zoom-covers")) == null) {
          // Turned upright, the drawn rigs: the centred zoom may cover the
          // foot of a name on the small phone (as in "one fixed spot").
          const rigs = turned !== "portrait" ? g.bodies : await page.evaluate(() =>
            Array.from(document.querySelectorAll("[data-enemy-index] > .gs-sprite-wrap")).map((n) => {
              const r = n.getBoundingClientRect();
              return { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
            }));
          for (const e of rigs) expect(overlaps(z, e)).toBe(false);
        }
        expect(z.x + z.width).toBeLessThanOrEqual(g.stage.right + 0.5);
        expect(z.y + z.height).toBeLessThanOrEqual(turned === "portrait" ? g.hand.top + 24.5 : g.stage.bottom + 0.5);
        // And both go together (a tap on the stage corner away from the zoom).
        const stage = await realBox(page.locator(".gs-arena-stage"));
        const right = z.x + z.width / 2 < stage.x + stage.width / 2;
        await page.touchscreen.tap(right ? stage.x + stage.width - 4 : stage.x + 4, stage.y + 6);
        await expect(zoom(page)).toHaveCount(0);
        await expect(armed(page)).toHaveCount(0);
      });

      test("replay mode: tap zooms, never arms", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&replay=1");
        await tapSlot(page, 2);
        await expect(zoom(page)).toBeVisible();
        await tapSlot(page, 2);
        await expect(zoom(page)).toBeVisible();
        await expect(page.locator(".gs-combatant.gs-clickable")).toHaveCount(0);
        await expect(armed(page)).toHaveCount(0);
        // Nothing is armed, so a tap on an enemy power still explains it.
        await page.locator(".gs-side:not(.gs-side-player) .gs-power").first().tap(T);
        await expect(page.locator(".gs-inspect")).toBeVisible();
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
      });

      test("the inspect sheet is suppressed while armed", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists");
        await tapSlot(page, 2);
        await expect(armed(page)).toBeVisible();
        // A tap on an enemy's intent chip targets that enemy. The zoom stands
        // on one fixed spot and may cover an intent on the small phone (it
        // shrinks for enemy BODIES only), so tap one it leaves uncovered.
        const target = await page.evaluate(() => {
          const z = document.querySelector(".gs-zoomcard")!.getBoundingClientRect();
          const free = Array.from(document.querySelectorAll<HTMLElement>(".gs-combatant[data-enemy-index]")).find((c) => {
            const b = c.querySelector(".gs-intent")!.getBoundingClientRect();
            const x = (b.left + b.right) / 2, y = (b.top + b.bottom) / 2;
            return !(x > z.left && x < z.right && y > z.top && y < z.bottom);
          });
          return free?.dataset.enemyIndex ?? null;
        });
        expect(target, "an intent the zoom leaves uncovered").not.toBeNull();
        await page.locator(`.gs-combatant[data-enemy-index="${target}"] .gs-intent`).tap(T);
        await expect(page.locator(".gs-inspect")).toHaveCount(0);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", `2>${target}`);
      });

      test("an unplayable card zooms but never arms or plays", async ({ page }) => {
        await openPreview(page, "enemies=1&enemy=cultists&unplayable=2");
        await tapSlot(page, 2);
        await expect(zoom(page)).toHaveAttribute("data-zoom-index", "2");
        await expect(armed(page)).toHaveCount(0);
        await tapSlot(page, 2);
        await zoom(page).tap(T);
        // The zoom stands off the enemies (portrait: over the band, spec
        // "Phase 2: portrait split", "The zoomed card"), so a plain tap on the
        // enemy reaches it.
        await page.locator('.gs-combatant[data-enemy-index="0"]').tap(T);
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
      });

      test("ten cards: every strip selects its own card", async ({ page }) => {
        await openPreview(page, "enemies=2&enemy=cultists&hand=10");
        // In order and then back, so every tap is a neighbour switching the zoom.
        for (const i of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 3]) {
          await tapSlot(page, i);
          await expect(zoom(page)).toHaveAttribute("data-zoom-index", String(i));
          await expect(page.locator(".gs-handslot.gs-selected")).toHaveCount(1);
        }
        await expect(lastPlay(page)).toHaveAttribute("data-last-play", "");
      });

      test("fighters are inside the stage", async ({ page }) => {
        await openPreview(page, PREVIEW);
        const r = await page.evaluate(() => {
          const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
          return Array.from(document.querySelectorAll(".gs-side")).map((el) => {
            const b = el.getBoundingClientRect();
            return { left: b.left - stage.left, right: stage.right - b.right,
                     top: b.top - stage.top, bottom: stage.bottom - b.bottom };
          });
        });
        expect(r.length).toBeGreaterThan(0);
        for (const b of r) {
          // 1px of rounding slack; the rigs' transparent overscan is not in .gs-side.
          expect(b.left).toBeGreaterThanOrEqual(-1);
          expect(b.right).toBeGreaterThanOrEqual(-1);
          expect(b.top).toBeGreaterThanOrEqual(-1);
          expect(b.bottom).toBeGreaterThanOrEqual(-1);
        }
      });
    });
  }
}

test.describe("a touch tablet", () => {
  test.use({ viewport: { width: 1024, height: 640 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2 });

  test("a touch tablet is not a phone", async ({ page }) => {
    await openPreview(page, PREVIEW);
    expect(await page.evaluate(() => matchMedia("(hover: none)").matches)).toBe(true);
    await expect(page.locator(".gs-phone")).toHaveCount(0);
  });

  test("a tap on a touch tablet raises the card, never zooms it", async ({ page }) => {
    await openPreview(page, PREVIEW);
    await page.locator(".gs-handslot-hit").last().tap({ timeout: 5000 });
    await expect(page.locator(".gs-handslot.gs-hovered")).toHaveCount(1);
    await page.waitForTimeout(300);
    await expect(page.locator(".gs-zoomcard")).toHaveCount(0);
  });
});

/** Landscape: with hand card `sel` zoomed, every OTHER slot has a point in the
 *  hand strip that hits it, and a tap there (under the zoom where the zoom
 *  hangs over the strip) switches the zoom to that card and plays nothing.
 *  Returns how many of those taps landed under the zoom. */
async function strayTapsSwitch(page: Page, sel: number): Promise<number> {
  const lastPlay = page.locator("[data-last-play]");
  const n = await page.locator(".gs-handslot").count();
  let under = 0;
  for (let j = 0; j < n; j++) {
    if (j === sel) continue;
    // (Tapping it while it is already zoomed would play it.)
    if (await page.locator(".gs-zoomcard").getAttribute("data-zoom-index", { timeout: 1000 }).catch(() => null) !== String(sel)) {
      await tapSlot(page, sel);
    }
    await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", String(sel));
    await zoomBox(page);
    await page.waitForTimeout(350); // the re-measure after the slot's lift
    // A point inside the strip that hits slot j, preferring one under the zoom.
    const pt = await page.evaluate((idx) => {
      const slots = Array.from(document.querySelectorAll(".gs-handslot"));
      const hit = slots[idx].querySelector(".gs-handslot-hit")!.getBoundingClientRect();
      const band = document.querySelector(".gs-hand")!.getBoundingClientRect().top;
      const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
      const z = document.querySelector(".gs-zoomcard")!.getBoundingClientRect();
      let fallback: { x: number; y: number; under: boolean } | null = null;
      for (let y = band + 6; y < stage.bottom - 2; y += 8) {
        for (let x = hit.left + 2; x < hit.right - 1; x += 4) {
          if (x < stage.left || x > stage.right) continue;
          const el = document.elementFromPoint(x, y);
          if (el?.closest(".gs-handslot") !== slots[idx]) continue;
          const under = x >= z.left && x <= z.right && y >= z.top && y <= z.bottom;
          if (under) return { x, y, under };
          fallback ??= { x, y, under };
        }
      }
      return fallback;
    }, j);
    expect(pt, `slot ${j} has no hittable point in the strip`).not.toBeNull();
    if (pt!.under) under++;
    await page.touchscreen.tap(pt!.x, pt!.y);
    await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", String(j));
    await expect(lastPlay).toHaveAttribute("data-last-play", "");
    await expect(page.locator(".gs-handslot")).toHaveCount(n);
  }
  return under;
}

/** Landscape: tapping the zoom ABOVE the strip is still the second tap. */
async function zoomAboveStripPlays(page: Page, sel: number): Promise<void> {
  if (await page.locator(".gs-zoomcard").getAttribute("data-zoom-index", { timeout: 1000 }).catch(() => null) !== String(sel)) {
    await tapSlot(page, sel);
  }
  await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", String(sel));
  const z = await zoomBox(page);
  await page.waitForTimeout(350);
  const band = (await realBox(page.locator(".gs-hand"))).y;
  await page.touchscreen.tap(z.x + z.width / 2, Math.min(z.y + z.height / 2, band - 8));
  await expect(page.locator("[data-last-play]")).toHaveAttribute("data-last-play", String(sel));
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, short landscape (the browser's bars)`, () => {
    // 40 px less than the page: the fixture no longer draws the 40 px tool
    // row the real app's HUD band stood for here.
    const ctx = shortLandscapeContext(size);
    test.use({ ...ctx, viewport: { width: ctx.viewport.width, height: ctx.viewport.height - 40 } });
    for (const hand of [5, 10]) {
      test(`the zoom passes strip taps to the slots under it (hand=${hand}, one enemy)`, async ({ page }) => {
        test.setTimeout(240_000);
        await openPreview(page, `enemies=1&enemy=cultists&hand=${hand}`);
        let under = 0;
        for (const sel of [2, 4]) under += await strayTapsSwitch(page, sel);
        console.log(`[passthrough] ${size} short hand=${hand}: ${under} taps under the zoom`);
        expect(under, "no slot lay under the zoom: the check proved nothing").toBeGreaterThan(0);
        await zoomAboveStripPlays(page, 4);
      });
    }
  });
}

test.describe("a desktop", () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test("desktop still uses title tooltips", async ({ page }) => {
    await openPreview(page, "enemies=2&enemy=cultists&powers=3");
    const power = page.locator(".gs-power").first();
    await expect(power).toHaveAttribute("title", /.+/);
    await power.click();
    await expect(page.locator(".gs-inspect")).toHaveCount(0);
  });

  test("desktop: a click arms an attack with the arrow, no zoom", async ({ page }) => {
    await openPreview(page, "enemies=2&enemy=cultists");
    await page.locator(".gs-handslot-hit").nth(2).click({ position: { x: 20, y: 40 } });
    await expect(page.locator(".gs-board[data-armed]")).toBeVisible();
    await expect(page.locator(".gs-zoomcard")).toHaveCount(0);
    const e = await realBox(page.locator('.gs-combatant[data-enemy-index="1"]'));
    await page.mouse.move(e.x + e.width / 2, e.y + e.height / 2);
    await expect(page.getByTestId("aim-arrow")).toBeVisible();
    await page.mouse.click(e.x + e.width / 2, e.y + e.height / 2);
    await expect(page.locator("[data-last-play]")).toHaveAttribute("data-last-play", "2>1");
  });
});
