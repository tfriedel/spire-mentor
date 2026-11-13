import { test, expect, type Page } from "@playwright/test";
import { PHONES, phoneContext, type Orientation, type PhoneSize } from "./phoneHarness";

/**
 * Every non-combat decision screen on a phone, from gallery.html (one fixture
 * per screen): each tap target is at least 44x44 real CSS px and no screen
 * overflows the page horizontally. Combat boards in the gallery are covered by
 * phone-layout.spec.ts and are skipped here by label.
 *
 * "Tap target" = visible, enabled `button`, `[role=button]` and map node
 * `g[data-can]`. A control smaller than 44px passes only when its HIT area is
 * 44x44: elementFromPoint at the corners and centre of the 44x44 box centred
 * on it lands on the control itself (a padded hit pad or ::after). Nothing is
 * skipped by selector; the only exclusions are hidden or zero-size elements,
 * disabled controls, and inline text links inside prose (none are matched by
 * the selector, so none exist to exclude).
 */
const FLOOR = 44;
const COMBAT = /^(Combat|Defect orb belt|Test Subject|Kaiser Crab)/;

type Offender = { screen: string; what: string; w: number; h: number; hit: boolean };
type Overflow = { screen: string; what: string; px: number };

async function audit(page: Page): Promise<{ offenders: Offender[]; overflows: Overflow[]; mapScroller: string; mapOverhang: number; count: number; screens: number; nodes: number }> {
  return page.evaluate(({ floor, combat }) => {
    const skip = new RegExp(combat);
    const offenders: Offender[] = [];
    const overflows: Overflow[] = [];
    let mapScroller = "none";
    let mapOverhang = -1;
    let count = 0;
    let screens = 0;
    let nodes = 0;
    const describe = (el: Element) => {
      const t = (el.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 30);
      const cls = (el.getAttribute("class") ?? "").split(/\s+/)[0];
      return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""}${t ? ` "${t}"` : ""}`;
    };
    for (const section of Array.from(document.querySelectorAll("section"))) {
      const screen = section.querySelector("h2")?.textContent ?? "?";
      if (skip.test(screen)) continue;
      screens++;
      // The screen's own box: every gallery screen sits in a `.surface` with
      // overflow:hidden, which would CLIP any in-screen overflow and hide it
      // from a page-level scrollWidth check, so it is measured here.
      const surface = section.querySelector<HTMLElement>(":scope > .surface") ?? section;
      if (surface.scrollWidth > surface.clientWidth + 1) {
        overflows.push({ screen, what: `screen body scrollWidth ${surface.scrollWidth} > clientWidth ${surface.clientWidth}`, px: surface.scrollWidth - surface.clientWidth });
      }
      // The act map's sheet (.gsm-parch-body, overflow:auto) scrolls down
      // through the act; sideways it fits (`mapOverhang`). Its nodes are
      // checked after being scrolled into view, below.
      const scroller = section.querySelector<HTMLElement>(".gsm-parch-body");
      if (scroller) {
        mapScroller = getComputedStyle(scroller).overflowX;
        // In the app the sheet is the screen, full-bleed; the gallery frames
        // it narrower than any phone (305 px on the 375 px one), so the
        // overhang is taken against the sheet at the phone's own width.
        mapOverhang = Math.max(0, scroller.scrollWidth - (scroller.clientWidth + window.innerWidth - surface.clientWidth));
      }
      const els = Array.from(section.querySelectorAll<HTMLElement | SVGElement>("button, [role=button], g[data-can]"));
      for (const el of els) {
        // Disabled controls are audited too: a disabled price tag that shrinks
        // shifts the shelf.
        el.scrollIntoView({ block: "center", inline: "center" });
        // A map node's `g` is as big as its drawn ink ring; what is TAPPED is
        // the transparent disc inside it, so that disc is the box to measure.
        const isNode = el.matches("g[data-can]");
        const r = (isNode ? el.querySelector("circle[data-hit]") ?? el : el).getBoundingClientRect();
        if (r.width === 0 || r.height === 0 || getComputedStyle(el).visibility === "hidden") continue;
        count++;
        if (isNode) nodes++;
        // Inside its screen horizontally (nodes: after scrolling into view).
        const sb = surface.getBoundingClientRect();
        const eb = el.getBoundingClientRect();
        if (eb.left < sb.left - 1 || eb.right > sb.right + 1) {
          overflows.push({ screen, what: `${describe(el)} spans ${Math.round(eb.left - sb.left)}..${Math.round(eb.right - sb.left)} of ${Math.round(sb.width)}`, px: Math.max(sb.left - eb.left, eb.right - sb.right) });
        }
        if (r.width >= floor - 0.5 && r.height >= floor - 0.5) continue;
        // Hit area: the floor-sized box centred on the control (a disc covers
        // its centre and edge midpoints, a box also its corners).
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2, h = floor / 2 - 1;
        const pts: [number, number][] = isNode
          ? [[cx, cy], [cx - h, cy], [cx + h, cy], [cx, cy - h], [cx, cy + h]]
          : [[cx, cy], [cx - h, cy - h], [cx + h, cy - h], [cx - h, cy + h], [cx + h, cy + h]];
        const hit = pts.every(([x, y]) => {
          const top = document.elementFromPoint(x, y);
          return !!top && (el === top || el.contains(top));
        });
        if (!hit) offenders.push({ screen, what: describe(el), w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10, hit });
      }
    }
    return { offenders, overflows, mapScroller, mapOverhang, count, screens, nodes };
  }, { floor: FLOOR, combat: COMBAT.source });
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  for (const orientation of ["portrait", "landscape"] as Orientation[]) {
    test.describe(`${size} phone, ${orientation}: non-combat screens`, () => {
      test.use(phoneContext(size, orientation));

      test("tap targets are 44px and nothing overflows", async ({ page }) => {
        await page.goto("/gallery.html");
        await page.locator("section").first().waitFor({ state: "visible" });
        await page.waitForTimeout(1500); // art and rigs settle
        const { offenders, overflows, mapScroller, mapOverhang, count, screens, nodes } = await audit(page);
        expect(screens, "non-combat screens audited").toBeGreaterThanOrEqual(10);
        expect(count, "tap targets audited").toBeGreaterThan(10);
        expect(nodes, "reachable map nodes audited").toBe(4);
        expect(mapScroller, "the act map scrolls horizontally on its own").toMatch(/^(auto|scroll)$/);
        // The act fits the paper: the columns close up on a narrow sheet
        // (MapView's `colFit`), so the map never scrolls sideways.
        expect(mapOverhang, "px the act map overhangs its sheet sideways").toBeGreaterThanOrEqual(0);
        expect(mapOverhang, "px the act map overhangs its sheet sideways").toBeLessThanOrEqual(1);
        expect(overflows, "controls or screen bodies wider than their screen").toEqual([]);
        expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth),
          "horizontal page overflow (px)").toBeLessThanOrEqual(0);
        expect(offenders, "tap targets under 44x44 (hit area included)").toEqual([]);
      });
    });
  }
}

// Owner feedback (2026-11-13): no hover on a phone, so a shop ware's effect
// could not be read before buying. The first tap opens its sheet.
test.describe("shop wares on a phone", () => {
  test.use(phoneContext("large", "portrait"));
  test("a tap on a relic or potion opens its sheet; nothing is bought until Buy", async ({ page }) => {
    await page.goto("/gallery.html");
    const item = page.locator(".gs-shop-item").first();
    await item.scrollIntoViewIfNeeded();
    await item.tap();
    const sheet = page.locator(".gs-shop-sheet");
    await expect(sheet).toBeVisible();
    const vp = page.viewportSize()!;
    const b = (await sheet.boundingBox())!;
    expect(b.y + b.height).toBeLessThanOrEqual(vp.height + 0.5);
    for (const name of [/Buy for|Too expensive|Sold out/, /Close/]) {
      const btn = sheet.getByRole("button", { name });
      await expect(btn).toBeVisible();
      expect((await btn.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    }
    // On top of everything: its Close button takes the tap.
    await sheet.getByRole("button", { name: /Close/ }).tap();
    await expect(sheet).toHaveCount(0);
  });
});

