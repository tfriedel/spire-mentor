import { test, expect, type Page } from "@playwright/test";
import { splitIntentReal, splitZoomRise } from "../src/portraitSplit";
import {
  PHONES, SHORT_PORTRAIT, drawnHeight, fontPx, openPreview, phoneContext, realBox, realStageContext, settledDrawnHeight,
  shortPortraitContext, tapSlot, type PhoneSize, type ShortPortrait,
} from "./phoneHarness";

/**
 * The phone-portrait combat screen's split layout (portraitSplit.ts): the
 * enemies across the top of the stage, the player figure in a band right
 * above the controls row, its status block (HP bar, block shield, powers)
 * beside it in real px. Solved once per fight, for every portrait fight.
 *
 * Landscape, a touch tablet and a desktop never get any of it.
 */

/** The figure's drawn height, real px (`SPLIT.figure` / `figureSmall`). */
const FIGURE: Record<PhoneSize, number> = { large: 116, small: 84 };
/** How far Osty may stand over the band's top, real px (`splitOstyRise`). */
const OSTY_RISE: Record<PhoneSize, number> = { large: 0, small: 18 };

type Box = { left: number; top: number; right: number; bottom: number };

/** The player figure's drawn body (crown to feet), real px. */
async function figureBox(page: Page, anchor = "player"): Promise<Box> {
  return page.evaluate((a) => {
    const wrap = document.querySelector(`.gs-combatant[data-combatant="${a}"] > .gs-sprite-wrap`) as HTMLElement;
    const stage = document.querySelector(".gs-arena-stage")!;
    const k = parseFloat(getComputedStyle(stage).getPropertyValue("--gs-k")) || 1;
    const rigTop = parseFloat(getComputedStyle(wrap).getPropertyValue("--rig-top")) || 0;
    const b = wrap.getBoundingClientRect();
    return { left: b.left, top: b.top + rigTop * k, right: b.right, bottom: b.bottom };
  }, anchor);
}

/** The band's drawn content: the union of its HP bar line and its power row. */
async function statusContent(page: Page): Promise<Box> {
  return page.evaluate(() => {
    const rs = Array.from(document.querySelectorAll(".gs-playerband .gs-hpwrap, .gs-playerband .gs-powers"))
      .map((el) => el.getBoundingClientRect()).filter((b) => b.width > 0 && b.height > 0);
    return { left: Math.min(...rs.map((b) => b.left)), top: Math.min(...rs.map((b) => b.top)),
      right: Math.max(...rs.map((b) => b.right)), bottom: Math.max(...rs.map((b) => b.bottom)) };
  });
}

const cy = (b: Box) => (b.top + b.bottom) / 2;

/** Every chrome rect on the stage, real px, by owner: each enemy (`e<i>`:
 *  its box, intent, damage numeral, HP bar, name and every power cell),
 *  `player` (the figure from its crown, Osty and everything the band draws)
 *  and `controls` (End Turn, energy, the piles). */
type Owned = { owner: string; what: string; r: Box };
async function chromeBoxes(page: Page): Promise<{
  rects: Owned[]; stage: Box; band: Box; osty: Box | null; slots: Record<string, number>;
}> {
  return page.evaluate(() => {
    const r = (el: Element): Box | null => {
      const b = el.getBoundingClientRect();
      return b.width > 0 && b.height > 0 ? { left: b.left, top: b.top, right: b.right, bottom: b.bottom } : null;
    };
    const out: Owned[] = [];
    const slots: Record<string, number> = {};
    const add = (owner: string, what: string, el: Element | null) => {
      const b = el && r(el);
      if (b) out.push({ owner, what, r: b });
    };
    document.querySelectorAll("[data-enemy-index]").forEach((e) => {
      const o = `e${e.getAttribute("data-enemy-index")}`;
      add(o, "body", e);
      e.querySelectorAll(".gs-intent").forEach((n) => add(o, "intent", n));
      e.querySelectorAll(".gs-intent-dmg").forEach((n) => add(o, "intent-dmg", n));
      e.querySelectorAll(":scope > .gs-hpwrap").forEach((n) => add(o, "hp", n));
      // The solver's slot, real px, on the enemy's wrapper.
      const slot = parseFloat(e.closest("[data-split-slot]")?.getAttribute("data-split-slot") ?? "NaN");
      slots[o] = slot;
      e.querySelectorAll(".gs-nameplate").forEach((n) => add(o, "name", n));
      e.querySelectorAll(".gs-power, .gs-power-more").forEach((n) => add(o, "power", n));
    });
    const stageEl = document.querySelector(".gs-arena-stage")!;
    const k = parseFloat(getComputedStyle(stageEl).getPropertyValue("--gs-k")) || 1;
    // The figure from its drawn crown (the sprite box reaches above it).
    const wrap = document.querySelector('.gs-combatant[data-combatant="player"] > .gs-sprite-wrap');
    if (wrap) {
      const b = r(wrap)!;
      const rigTop = parseFloat(getComputedStyle(wrap).getPropertyValue("--rig-top")) || 0;
      out.push({ owner: "player", what: "figure", r: { ...b, top: b.top + rigTop * k } });
    }
    let osty: Box | null = null;
    const ow = document.querySelector('.gs-combatant[data-combatant="osty"]');
    if (ow) {
      const sw = ow.querySelector(":scope > .gs-sprite-wrap");
      if (sw) {
        const b = r(sw)!;
        const rigTop = parseFloat(getComputedStyle(sw).getPropertyValue("--rig-top")) || 0;
        osty = { ...b, top: b.top + rigTop * k };
        out.push({ owner: "player", what: "osty", r: osty });
      }
      ow.querySelectorAll(".gs-hpwrap, .gs-nameplate").forEach((n) => add("player", "osty-chrome", n));
    }
    // What the band draws, not the boxes that only lay it out (the status
    // block runs the figure's full height, empty above and below its lines).
    const painted = (n: Element) => {
      if (n.childElementCount === 0) return true;
      const cs = getComputedStyle(n);
      return cs.backgroundColor !== "rgba(0, 0, 0, 0)" || cs.backgroundImage !== "none"
        || parseFloat(cs.borderTopWidth) > 0 || cs.boxShadow !== "none";
    };
    document.querySelectorAll(".gs-playerband *").forEach((n) => {
      if (painted(n)) add("player", `band ${n.className}`, n);
    });
    document.querySelectorAll(".gs-endturn, .gs-energy, .gs-pilebtn").forEach((n) => add("controls", n.className, n));
    return { rects: out, stage: r(stageEl)!, band: r(document.querySelector(".gs-playerband")!)!, osty, slots };
  });
}

/** The overlap matrix: rosters x player characters, each with 0, 3 and 8
 *  powers. `strict` cases also keep every enemy rect above the band. */
const MATRIX: { q: string; sizes: PhoneSize[]; pws: number[]; rig?: string }[] = [
  { q: "enemy=cultists", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemies=2", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemies=3", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemies=4", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemies=5", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemies=5&enemy=cultists", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemy=lagavulin_matriarch", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemy=queen", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "enemy=bygone_effigy", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "char=necrobinder&osty=1&enemies=2", sizes: ["large", "small"], pws: [0, 3, 8] },
  { q: "char=defect&orbs=3,3&enemies=2", sizes: ["large", "small"], pws: [0, 3, 8] },
  // A wider rig than the Ironclad's against the band's 17 px shield overhang.
  { q: "char=defect&enemies=2", sizes: ["large", "small"], pws: [3] },
  { q: "char=silent&enemies=2", sizes: ["large", "small"], pws: [3] },
  { q: "char=regent&enemies=2", sizes: ["large", "small"], pws: [3] },
  // The multi-rig bosses: one solver creature each, drawn as several
  // targetable parts that share its slot and take the fight's chrome.
  // Skipped where the rigs are not linked into public/preview-rig (see
  // boardpreview.tsx's `?scene=` note).
  { q: "scene=kaiser_crab", sizes: ["large", "small"], pws: [0, 3, 8], rig: "kaiser_crab/kaiser_crab.atlas" },
  { q: "scene=decimillipede", sizes: ["large", "small"], pws: [0, 3, 8],
    rig: "decimillipede_segment_front/decimillipede_segment_front.atlas" },
];

const hits = (a: Box, b: Box) => a.left < b.right - 0.5 && b.left < a.right - 0.5
  && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5;

/** One overlap-matrix case: no two owners' chrome rects overlap, every
 *  rect stays on the stage, every enemy rect above the band (and Osty), and
 *  each enemy's HP bar fits its solver slot. */
async function overlapCase(page: Page, size: PhoneSize, q: string, pw: number, rig?: string): Promise<void> {
  const query = `${q}&powers=${pw}&ppowers=${pw}`;
  if (rig) {
    // A missing static file falls through to the SPA's index.html.
    const res = await page.request.get(`/preview-rig/${rig}`);
    test.skip(!res.ok() || (await res.text()).startsWith("<"), `${rig} is not linked into public/preview-rig`);
  }
  await openPreview(page, query);
  await settledDrawnHeight(page);
  if (q.includes("osty")) await settledDrawnHeight(page, "osty");
  // The band settles once a rig measures: read the boxes after it.
  await page.waitForTimeout(300);
  const { rects, stage, band, osty, slots } = await chromeBoxes(page);
  const split = await page.locator(".gs-board").getAttribute("data-split");
  const lowest = Math.max(...rects.filter((x) => x.owner.startsWith("e")).map((x) => x.r.bottom));
  console.log(`[overlap] ${size} ${query}: ${split}; lowest enemy rect ${lowest.toFixed(1)}, `
    + `band top ${band.top.toFixed(1)}${osty ? `, osty top ${osty.top.toFixed(1)}` : ""}`);
  // Every owner collected its rects: an empty one would make the
  // overlap check below vacuous.
  expect(rects.filter((x) => x.owner.startsWith("e")).length).toBeGreaterThan(0);
  expect(rects.filter((x) => x.owner === "player" && x.what === "figure")).toHaveLength(1);
  expect(rects.filter((x) => x.owner === "player" && x.what.startsWith("band ")).length).toBeGreaterThan(0);
  // End Turn, the energy orb and the three piles.
  expect(rects.filter((x) => x.owner === "controls")).toHaveLength(5);
  if (q.includes("osty")) expect(rects.filter((x) => x.owner === "player" && x.what === "osty")).toHaveLength(1);
  // Each enemy's HP bar fits its solver slot, and side-by-side bars
  // keep the solver's 4 px apart.
  const bars = rects.filter((x) => x.what === "hp");
  expect(bars.length).toBe(Object.keys(slots).length);
  for (const b of bars) {
    expect(Number.isFinite(slots[b.owner]), `${b.owner} has a solver slot`).toBe(true);
    expect(b.r.right - b.r.left, `${b.owner}'s HP bar vs its ${slots[b.owner]} px slot`)
      .toBeLessThanOrEqual(slots[b.owner] + 0.5);
  }
  for (const [i, a] of bars.entries()) {
    for (const b of bars.slice(i + 1)) {
      if (a.r.top >= b.r.bottom || b.r.top >= a.r.bottom) continue; // another rank
      const apart = Math.max(b.r.left - a.r.right, a.r.left - b.r.right);
      expect(apart, `${a.owner} and ${b.owner}'s HP bars`).toBeGreaterThanOrEqual(4 - 0.5);
    }
  }
  // A cramped fight on a stage too short to draw a player-sized enemy
  // `SPLIT.minCrown` tall above the band lets its HP bars down over the
  // band's upper part (`over=` in data-split): there the enemies may cover
  // the player's figure, band and Osty, never the controls.
  const over = /\bover=/.test(split ?? "");
  const controlsTop = Math.min(...rects.filter((x) => x.owner === "controls").map((x) => x.r.top));
  const bad: string[] = [];
  for (const [i, a] of rects.entries()) {
    for (const b of rects.slice(i + 1)) {
      const enemyOnPlayer = (a.owner.startsWith("e") && b.owner === "player") || (b.owner.startsWith("e") && a.owner === "player");
      if (over && enemyOnPlayer) continue;
      if (a.owner !== b.owner && hits(a.r, b.r)) {
        bad.push(`${a.owner} ${a.what} [${a.r.left.toFixed(0)},${a.r.top.toFixed(0)},${a.r.right.toFixed(0)},${a.r.bottom.toFixed(0)}]`
          + ` x ${b.owner} ${b.what} [${b.r.left.toFixed(0)},${b.r.top.toFixed(0)},${b.r.right.toFixed(0)},${b.r.bottom.toFixed(0)}]`);
      }
    }
    const inside = a.r.left >= stage.left - 0.5 && a.r.right <= stage.right + 0.5
      && a.r.top >= stage.top - 0.5 && a.r.bottom <= stage.bottom + 0.5;
    if (!inside) bad.push(`${a.owner} ${a.what} leaves the stage`);
    // Above the band, and above Osty where it stands taller than the
    // figure (the small phone).
    const ceiling = over ? controlsTop : Math.min(band.top, osty?.top ?? Infinity);
    if (a.owner.startsWith("e") && a.r.bottom > ceiling + 0.5) {
      bad.push(`${a.owner} ${a.what} bottom ${a.r.bottom.toFixed(1)} under the band/Osty top ${ceiling.toFixed(1)}`);
    }
  }
  expect(bad, bad.join("\n")).toEqual([]);
}

/** The matrix's tests for one phone, `keep` picking the cases. */
function overlapMatrix(size: PhoneSize, label: string, keep: (q: string, pw: number) => boolean): void {
  for (const { q, sizes, pws, rig } of MATRIX) {
    if (!sizes.includes(size)) continue;
    for (const pw of pws) {
      if (!keep(q, pw)) continue;
      test(`no two creatures' chrome boxes overlap${label} (${q}&powers=${pw}&ppowers=${pw})`,
        async ({ page }) => overlapCase(page, size, q, pw, rig));
    }
  }
}

/** The matrix rosters also run on the full phone viewport. */
const FULL_VIEWPORT_SET = ["enemy=cultists", "enemies=3", "enemies=5", "enemy=queen", "char=necrobinder&osty=1&enemies=2",
  "scene=kaiser_crab"];

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait: the split`, () => {
    test.use(phoneContext(size, "portrait"));

    test("the split is on in portrait only", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists");
      await expect(page.locator(".gs-split")).toHaveCount(1);
      await expect(page.locator(".gs-board")).toHaveAttribute("data-split", /^(rows|column|compact) s=\d+\.\d{3} back=\d+$/);
      await expect(page.locator(".gs-actors .gs-playerband")).toHaveCount(1);
      // One orbs anchor at most, and never on the figure.
      await expect(page.locator('.gs-side-player [data-combatant="orbs"]')).toHaveCount(0);
    });

    test("the figure stands in the band at its fixed height", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists");
      const h = await settledDrawnHeight(page);
      console.log(`[split figure] ${size}: ${h.toFixed(1)} px`);
      expect(Math.abs(h - FIGURE[size])).toBeLessThanOrEqual(2);
      // The figure stands ON the controls row: its feet at least 6 px above
      // the top of every control under it (End Turn, at the far right, is
      // taller than the row and stands beside the status block, not under the
      // figure).
      const fig = await figureBox(page);
      let under = 0;
      for (const sel of [".gs-endturn", ".gs-energy", ".gs-pile-draw", ".gs-pile-discard", ".gs-pile-exhaust"]) {
        const c = await realBox(page.locator(sel));
        if (c.x >= fig.right || c.x + c.width <= fig.left) continue;
        under++;
        expect(c.y, `${sel} top vs the figure's feet`).toBeGreaterThanOrEqual(fig.bottom + 6 - 0.5);
      }
      expect(under, "no control stands under the figure: the check proved nothing").toBeGreaterThan(0);
      // No player chrome under the figure: no HP bar, no nameplate, no powers.
      await expect(page.locator(".gs-side-player .gs-hpwrap, .gs-side-player .gs-nameplate, .gs-side-player .gs-powers"))
        .toHaveCount(0);
      expect(await page.locator(".gs-playerband").evaluate((el) => getComputedStyle(el).backgroundColor))
        .toBe("rgba(0, 0, 0, 0)");
    });

    test("the status block is centred on the figure and runs to the gutter", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists&ppowers=3&pblock=12");
      await settledDrawnHeight(page);
      const fig = await figureBox(page);
      const st = await statusContent(page);
      expect(Math.abs(cy(st) - cy(fig)), "status block centre vs figure centre").toBeLessThanOrEqual(2);
      const bar = await realBox(page.locator(".gs-playerband .gs-hpbar"));
      expect(Math.abs(bar.height - 20)).toBeLessThanOrEqual(0.5);
      const vw = page.viewportSize()!.width;
      expect(Math.abs(bar.x + bar.width - (vw - 16)), "the bar ends at the right gutter").toBeLessThanOrEqual(1);
      expect(await fontPx(page.locator(".gs-playerband .gs-hptext"))).toBeGreaterThanOrEqual(16);
      // The shield sits on the bar's left end, clear of the numerals.
      const shield = await realBox(page.locator(".gs-playerband .gs-block"));
      const barMid = bar.y + bar.height / 2;
      expect(shield.x).toBeLessThanOrEqual(bar.x);
      expect(shield.x + shield.width).toBeGreaterThanOrEqual(bar.x);
      expect(shield.y).toBeLessThanOrEqual(barMid);
      expect(shield.y + shield.height).toBeGreaterThanOrEqual(barMid);
      const nums = await page.locator(".gs-playerband .gs-hptext").evaluate((el) => {
        const r = document.createRange();
        r.selectNodeContents(el);
        const b = r.getBoundingClientRect();
        return { left: b.left, right: b.right, top: b.top, bottom: b.bottom };
      });
      const apart = shield.x + shield.width <= nums.left + 0.5 || shield.x >= nums.right - 0.5
        || shield.y + shield.height <= nums.top + 0.5 || shield.y >= nums.bottom - 0.5;
      expect(apart, "the shield is off the numerals").toBe(true);
      const powers = await page.locator(".gs-playerband .gs-power").all();
      expect(powers.length).toBeGreaterThan(0);
      const first = await realBox(powers[0]);
      expect(first.y).toBeGreaterThanOrEqual(bar.y + bar.height + 6 - 0.5);
      for (const pw of powers) {
        const b = await realBox(pw);
        expect(Math.abs(b.width - 44)).toBeLessThanOrEqual(0.5);
        expect(Math.abs(b.height - 44)).toBeLessThanOrEqual(0.5);
      }
    });

    test("no powers: the bar alone, centred", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists&ppowers=0");
      await settledDrawnHeight(page);
      const fig = await figureBox(page);
      const bar = await realBox(page.locator(".gs-playerband .gs-hpbar"));
      expect(Math.abs(bar.y + bar.height / 2 - cy(fig))).toBeLessThanOrEqual(2);
      await expect(page.locator(".gs-playerband .gs-powers")).toHaveCount(0);
    });

    test("Osty stands between figure and status", async ({ page }) => {
      await openPreview(page, "char=necrobinder&osty=1&enemies=2");
      await settledDrawnHeight(page);
      const h = await settledDrawnHeight(page, "osty");
      console.log(`[split osty] ${size}: ${h.toFixed(1)} px`);
      expect(Math.abs(h - 64)).toBeLessThanOrEqual(2);
      const fig = await realBox(page.locator(".gs-side-player"));
      const osty = await realBox(page.locator(".gs-side-osty"));
      const status = await realBox(page.locator(".gs-playerband .gs-playerband-status"));
      expect(osty.x).toBeGreaterThanOrEqual(fig.x + fig.width - 0.5);
      // The status block's shield overhangs its left edge by 17 px.
      expect(osty.x + osty.width).toBeLessThanOrEqual(status.x - 17 + 0.5);
      // Osty rises no higher over the band's top (the solver's: 6 px of
      // padding over the figure) than the solver keeps clear for it
      // (`splitOstyRise`: 18 on the small phone, 0 on the large).
      const drawnTop = (await figureBox(page, "osty")).top;
      const rise = (status.y - 6) - drawnTop;
      console.log(`[split osty rise] ${size}: ${rise.toFixed(1)} px`);
      expect(rise).toBeLessThanOrEqual(OSTY_RISE[size] + 1);
    });

    test("enemies stand in the top part, above the band", async ({ page }) => {
      for (const n of [1, 3, 5]) {
        await openPreview(page, `enemies=${n}&enemy=cultists&powers=3`);
        await settledDrawnHeight(page);
        const band = await realBox(page.locator(".gs-playerband"));
        const bodies = await page.locator("[data-enemy-index]").all();
        expect(bodies.length).toBe(n);
        for (const b of bodies) {
          const e = await realBox(b);
          expect(e.y + e.height, `enemies=${n}: an enemy body reaches into the band`).toBeLessThanOrEqual(band.y + 0.5);
        }
      }
    });

    // The full matrix runs at the real app's stage height (below); on the
    // full phone viewport the fixture's stage is taller than any a player
    // gets, so a reduced set is enough here.
    overlapMatrix(size, "", (q, pw) => pw === 3 && FULL_VIEWPORT_SET.includes(q));

    test("the chrome mode is the roster's, not the powers'", async ({ page }) => {
      for (const q of ["enemy=cultists", "enemies=3", "enemies=5"]) {
        await openPreview(page, `${q}&powers=0`);
        const none = await page.locator(".gs-board").getAttribute("data-split");
        await openPreview(page, `${q}&powers=8`);
        const eight = await page.locator(".gs-board").getAttribute("data-split");
        console.log(`[chrome mode] ${size} ${q}: ${none} / ${eight}`);
        expect(none?.split(" ")[0], q).toBe(eight?.split(" ")[0]);
      }
    });
    // While a card is zoomed, the status block (shield, HP bar, orbs, powers)
    // steps right of the zoom: the player keeps their block and powers in
    // sight while choosing a card. Nothing else moves; × puts it back.
    for (const q of ["enemies=2&enemy=cultists", "char=necrobinder&osty=1&enemies=2", "char=defect&orbs=3,3&enemies=2"]) {
      test(`the band stays readable beside the zoom (${q})`, async ({ page }) => {
        await openPreview(page, `${q}&pblock=12&powers=3`);
        const enemyBoxes = () => page.evaluate(() => Array.from(document.querySelectorAll("[data-enemy-index]")).map((e) => {
          const b = e.getBoundingClientRect();
          return { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
        }));
        const statusLeft = async () => (await realBox(page.locator(".gs-playerband-status"))).x;
        const before = await enemyBoxes();
        const left0 = await statusLeft();
        await tapSlot(page, 1);
        const zl = page.locator(".gs-zoomcard");
        await expect(zl).toHaveAttribute("data-zoom-index", "1");
        await zl.evaluate((el) => Promise.all(el.getAnimations().map((an) => an.finished)));
        await page.waitForTimeout(150);
        const zb = await realBox(zl);
        const z: Box = { left: zb.x, top: zb.y, right: zb.x + zb.width, bottom: zb.y + zb.height };
        const stage = (await chromeBoxes(page)).stage;
        const band = await page.evaluate(() => {
          const r = (el: Element) => { const b = el.getBoundingClientRect(); return { left: b.left, top: b.top, right: b.right, bottom: b.bottom }; };
          const out: { what: string; r: { left: number; top: number; right: number; bottom: number } }[] = [];
          const sh = document.querySelector('[data-block-badge-for="player"]');
          if (sh) out.push({ what: "shield", r: r(sh) });
          const hp = document.querySelector(".gs-playerband .gs-hptext");
          if (hp) out.push({ what: "hp text", r: r(hp) });
          document.querySelectorAll(".gs-playerband .gs-power, .gs-playerband .gs-power-more")
            .forEach((n) => out.push({ what: `power ${n.className}`, r: r(n) }));
          return out;
        });
        expect(band.map((b) => b.what)).toContain("shield");
        expect(band.map((b) => b.what)).toContain("hp text");
        expect(band.filter((b) => b.what.startsWith("power")).length).toBeGreaterThan(0);
        for (const b of band) {
          expect(hits(b.r, z), `${b.what} under the zoom`).toBe(false);
          expect(b.r.left >= stage.left - 0.5 && b.r.right <= stage.right + 0.5
            && b.r.top >= stage.top - 0.5 && b.r.bottom <= stage.bottom + 0.5, `${b.what} inside the stage`).toBe(true);
        }
        // The HP numerals stay at the floor and inside the narrowed bar, also
        // a three-digit "120/120".
        const hpText = page.locator(".gs-playerband .gs-hptext");
        expect(await fontPx(hpText)).toBeGreaterThanOrEqual(13);
        for (const txt of [null, "120/120"]) {
          const fit = await page.evaluate((t) => {
            const el = document.querySelector(".gs-playerband .gs-hptext")!;
            if (t != null) el.textContent = t;
            const range = document.createRange();
            range.selectNodeContents(el);
            const g = range.getBoundingClientRect();
            const bar = document.querySelector(".gs-playerband .gs-hpbar")!.getBoundingClientRect();
            return { text: el.textContent, gl: g.left, gr: g.right, bl: bar.left, br: bar.right };
          }, txt);
          console.log(`[band beside zoom] ${size} ${q}: "${fit.text}" glyphs ${fit.gl.toFixed(1)}..${fit.gr.toFixed(1)} bar ${fit.bl.toFixed(1)}..${fit.br.toFixed(1)}`);
          expect(fit.gl, `"${fit.text}" left of the bar`).toBeGreaterThanOrEqual(fit.bl - 0.5);
          expect(fit.gr, `"${fit.text}" right of the bar`).toBeLessThanOrEqual(fit.br + 0.5);
        }
        // The Defect's orbs keep one row.
        const orbTops = await page.evaluate(() => Array.from(document.querySelectorAll(".gs-playerband .gs-orbs-row > *"))
          .map((o) => Math.round(o.getBoundingClientRect().top)));
        if (orbTops.length) expect(new Set(orbTops).size, `orb rows ${orbTops.join(",")}`).toBe(1);
        // Nothing else moved: every enemy within 0.5 px of where it stood.
        const after = await enemyBoxes();
        expect(after.length).toBe(before.length);
        for (const [i, a] of after.entries()) {
          for (const k of ["left", "top", "right", "bottom"] as const) {
            expect(Math.abs(a[k] - before[i][k]), `enemy ${i} ${k}`).toBeLessThan(0.5);
          }
        }
        const ostyEl = page.locator('.gs-combatant[data-combatant="osty"]');
        if (await ostyEl.count()) {
          const o = await figureBox(page, "osty");
          console.log(`[band beside zoom] ${size} Osty [${o.left.toFixed(0)},${o.top.toFixed(0)},${o.right.toFixed(0)},${o.bottom.toFixed(0)}]`
            + ` zoom [${z.left.toFixed(0)},${z.top.toFixed(0)},${z.right.toFixed(0)},${z.bottom.toFixed(0)}] under=${hits(o, z)}`);
        }
        console.log(`[band beside zoom] ${size} ${q}: status left ${left0.toFixed(1)} -> ${(await statusLeft()).toFixed(1)}, zoom right ${z.right.toFixed(1)}`);
        await page.getByRole("button", { name: "Close", exact: true }).tap({ timeout: 5000 });
        await expect(zl).toHaveCount(0);
        await expect.poll(statusLeft, { timeout: 2000 }).toBeCloseTo(left0, 0);
        expect(Math.abs((await statusLeft()) - left0)).toBeLessThan(0.5);
      });
    }
  });

  test.describe(`${size} phone, portrait: a star-cost card zoomed`, () => {
    test.use(phoneContext(size, "portrait"));

    // The zoom stands at one fixed spot for every card, far enough in that
    // the Regent's star cost (the icon furthest proud of the card's left
    // edge) is painted on the stage.
    test("the star cost stays on the stage", async ({ page }) => {
      await openPreview(page, "char=regent&enemies=2&stars=1&pblock=12&powers=3");
      await tapSlot(page, 1);
      const zl = page.locator(".gs-zoomcard");
      await expect(zl).toHaveAttribute("data-zoom-index", "1");
      await zl.evaluate((el) => Promise.all(el.getAnimations({ subtree: true }).map((an) => an.finished)));
      await expect(zl.locator(".gs-card-star")).toHaveCount(1);
      const m = await page.evaluate(async () => {
        const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
        const z = document.querySelector(".gs-zoomcard")!;
        // The star image's leftmost non-transparent column, on the glass.
        const img = z.querySelector(".gs-card-star img") as HTMLImageElement;
        await img.decode();
        const r = img.getBoundingClientRect();
        const c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        const g = c.getContext("2d")!;
        g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, c.width, c.height).data;
        let minX = c.width;
        for (let y = 0; y < c.height; y++) {
          for (let x = 0; x < minX; x++) if (d[(y * c.width + x) * 4 + 3] > 8) { minX = x; break; }
        }
        const zoom = z.getBoundingClientRect();
        return { stageLeft: stage.left, painted: r.left + (minX / c.width) * r.width, box: r.left,
          zoomLeft: zoom.left, minX, natural: c.width };
      });
      console.log(`[star] ${size}: stage ${m.stageLeft.toFixed(1)}, zoom ${m.zoomLeft.toFixed(1)}, star box ${m.box.toFixed(1)}, painted ${m.painted.toFixed(1)} (col ${m.minX}/${m.natural})`);
      expect(m.minX, "the star image has painted pixels").toBeLessThan(m.natural);
      expect(m.painted, "the star's leftmost painted pixel").toBeGreaterThanOrEqual(m.stageLeft);
      // The box too: the fixed spot clears the icon's whole box, not only ink.
      expect(m.box).toBeGreaterThanOrEqual(m.stageLeft - 0.5);
    });
  });

  test.describe(`${size} phone, landscape: no split`, () => {
    test.use(phoneContext(size, "landscape"));

    test("the split is on in portrait only", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists&ppowers=3");
      await expect(page.locator(".gs-phone-landscape")).toHaveCount(1);
      await expect(page.locator(".gs-split")).toHaveCount(0);
      await expect(page.locator(".gs-playerband")).toHaveCount(0);
      await expect(page.locator("[data-split]")).toHaveCount(0);
      // The player keeps its own chrome under its feet.
      await expect(page.locator(".gs-side-player .gs-hpwrap")).toHaveCount(1);
    });
  });
}

for (const [name, use] of [
  ["a desktop", { viewport: { width: 1280, height: 800 } }],
  ["a touch tablet", { viewport: { width: 1024, height: 640 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2 }],
] as const) {
  test.describe(`${name}: no split`, () => {
    test.use(use);

    test("the split is on in portrait only", async ({ page }) => {
      await openPreview(page, "enemies=2&enemy=cultists&ppowers=3");
      await expect(page.locator(".gs-split")).toHaveCount(0);
      await expect(page.locator(".gs-playerband")).toHaveCount(0);
      await expect(page.locator("[data-split]")).toHaveCount(0);
      await expect(page.locator(".gs-side-player .gs-hpwrap")).toHaveCount(1);
    });
  });
}

/** Every enemy's drawn body (crown to feet, the rig's ink) and its intent
 *  chip, real px. */
async function enemyInk(page: Page): Promise<{ h: number; crown: number; intentTop: number | null }[]> {
  return page.evaluate(() => {
    const stage = document.querySelector(".gs-arena-stage")!;
    const k = parseFloat(getComputedStyle(stage).getPropertyValue("--gs-k")) || 1;
    return Array.from(document.querySelectorAll("[data-enemy-index]")).map((e) => {
      const w = e.querySelector(":scope > .gs-sprite-wrap")!;
      const rigTop = parseFloat(getComputedStyle(w).getPropertyValue("--rig-top")) || 0;
      const b = w.getBoundingClientRect();
      const crown = b.top + rigTop * k;
      const it = e.querySelector(".gs-intent")?.getBoundingClientRect();
      return { h: b.bottom - crown, crown, intentTop: it ? it.top : null };
    });
  });
}

// The fixture at the stage height the real app has at the first fight
// (Phase 2: portrait split, real-app stage height): 674 / 456 real px, where
// the full phone viewport gives the fixture ~850 / 599.
for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait: the split at the real stage height`, () => {
    test.use(realStageContext(size));

    test("the stage is the real app's", async ({ page }) => {
      await openPreview(page, "enemy=cultists");
      const h = (await realBox(page.locator(".gs-arena-stage"))).height;
      expect(Math.abs(h - (size === "small" ? 496 : 714))).toBeLessThanOrEqual(2);
    });

    overlapMatrix(size, " at the real stage", () => true);

    // The intent chip stands inside the headroom the solver keeps for it
    // (`splitIntentReal`: 44 on the small phone, measured 43.5).
    test("every intent stands inside the solver's headroom", async ({ page }) => {
      for (const q of ["enemy=cultists", "enemies=3", "enemy=queen"]) {
        await openPreview(page, `${q}&powers=3`);
        await settledDrawnHeight(page);
        const ink = await enemyInk(page);
        expect(ink.length).toBeGreaterThan(0);
        for (const [i, e] of ink.entries()) {
          expect(e.intentTop, `${q}: enemy ${i} has an intent`).not.toBeNull();
          expect(e.crown - e.intentTop!, `${q}: enemy ${i}`).toBeLessThanOrEqual(splitIntentReal(size === "small") + 0.5);
        }
      }
    });

    // The small phone's zoom rises 20 px over the band (`splitZoomRise`) and
    // the enemies' floor no longer keeps that clear: it may cover a name's
    // foot, never an enemy body, intent or HP bar.
    for (const q of ["enemy=cultists", "enemies=2", "enemies=3", "enemies=4", "enemy=queen", "enemy=lagavulin_matriarch",
      "char=necrobinder&osty=1&enemies=2"]) {
      test(`the zoom covers no enemy body, intent or HP bar (${q})`, async ({ page }) => {
        await openPreview(page, `${q}&powers=3`);
        await settledDrawnHeight(page);
        await tapSlot(page, 1);
        const zl = page.locator(".gs-zoomcard");
        await expect(zl).toHaveAttribute("data-zoom-index", "1");
        await zl.evaluate((el) => Promise.all(el.getAnimations().map((an) => an.finished)));
        const zb = await realBox(zl);
        const band = await realBox(page.locator(".gs-playerband"));
        // The band's box starts at the figure's top; the solver's band top
        // is its 6 px of padding above that.
        const rise = band.y - 6 - zb.y;
        console.log(`[zoom at real stage] ${size} ${q}: zoom top ${zb.y.toFixed(1)}, rise over the band ${rise.toFixed(1)}`);
        expect(Math.abs(rise - splitZoomRise(size === "small"))).toBeLessThanOrEqual(1.5);
        const { hit, seen } = await page.evaluate((z) => {
          const out: string[] = [];
          let n = 0;
          document.querySelectorAll("[data-enemy-index]").forEach((e) => {
            const i = e.getAttribute("data-enemy-index");
            // The body is the drawn rig (the enemy's box also holds its
            // name, which the zoom may cover).
            const parts: [Element, string][] = [
              ...Array.from(e.querySelectorAll(":scope > .gs-sprite-wrap")).map((x): [Element, string] => [x, "body"]),
              ...Array.from(e.querySelectorAll(".gs-intent")).map((x): [Element, string] => [x, "intent"]),
              ...Array.from(e.querySelectorAll(":scope > .gs-hpwrap")).map((x): [Element, string] => [x, "hp"])];
            for (const [el, what] of parts) {
              const b = el.getBoundingClientRect();
              if (b.right <= b.left || b.bottom <= b.top) continue;
              n++;
              if (z.x < b.right - 0.5 && b.left < z.x + z.width - 0.5 && z.y < b.bottom - 0.5 && b.top < z.y + z.height - 0.5) {
                out.push(`e${i} ${what} [${b.left.toFixed(0)},${b.top.toFixed(0)},${b.right.toFixed(0)},${b.bottom.toFixed(0)}]`);
              }
            }
          });
          return { hit: out, seen: n };
        }, zb);
        expect(seen, "enemy bodies, intents and HP bars collected").toBeGreaterThan(0);
        expect(hit, hit.join("\n")).toEqual([]);
      });
    }

    // A multi-rig boss's parts share its ONE solver slot. A part's death
    // must not re-share it: the survivors' tap box, HP bar and power
    // cells stay where they were (the crab's Rocket, the Decimillipede's back).
    for (const [scene, dying, rig] of [["kaiser_crab", 1, "kaiser_crab/kaiser_crab.atlas"],
      ["decimillipede", 2, "decimillipede_segment_front/decimillipede_segment_front.atlas"]] as const) {
      test(`a part's death leaves the survivors' chrome in place (${scene})`, async ({ page }) => {
        const res = await page.request.get(`/preview-rig/${rig}`);
        test.skip(!res.ok() || (await res.text()).startsWith("<"), `${rig} is not linked into public/preview-rig`);
        await openPreview(page, `scene=${scene}&powers=3&die=6000&dying=${dying}`);
        await rigsLoaded(page);
        const survivors = (): Promise<{ id: string; boxes: Box[]; cells: number; slot: string | null }[]> =>
          page.evaluate((dead) => Array.from(document.querySelectorAll("[data-enemy-index]"))
            .filter((e) => e.getAttribute("data-enemy-index") !== String(dead))
            .map((e) => {
              const r = (el: Element | null): Box => {
                const b = el ? el.getBoundingClientRect() : { left: 0, top: 0, right: 0, bottom: 0 };
                return { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
              };
              return { id: e.getAttribute("data-enemy-index")!,
                // (The intent rides on the body; it bobs, so it is not compared.)
                boxes: [r(e), r(e.querySelector(":scope > .gs-hpwrap"))],
                cells: e.querySelectorAll(".gs-power, .gs-power-more").length,
                slot: e.closest("[data-split-slot]")?.getAttribute("data-split-slot") ?? null };
            }), dying);
        const before = await survivors();
        expect(await page.evaluate(() => performance.now()), "recorded before the death").toBeLessThan(6000);
        const n0 = await page.locator("[data-enemy-index]").count();
        await expect(page.locator("[data-enemy-index]")).toHaveCount(n0 - 1, { timeout: 20_000 });
        await page.waitForTimeout(500);
        const after = await survivors();
        console.log(`[part death] ${size} ${scene}: slots ${before.map((x) => x.slot).join(",")} -> ${after.map((x) => x.slot).join(",")}`);
        expect(after.map((x) => x.id)).toEqual(before.map((x) => x.id));
        for (const [i, a] of after.entries()) {
          expect(a.slot, `${a.id} slot`).toBe(before[i].slot);
          expect(a.cells, `${a.id} power cells`).toBe(before[i].cells);
          expectSameBoxes(a.boxes, before[i].boxes, `enemy ${a.id} (body, HP bar)`, 1);
        }
      });
    }

    if (size === "small") {
      // What the small phone's budget is for: at the real stage, sizes a
      // player can read (before: one Cultist 53, the bosses 48-55, three
      // Shrinker Beetles 29).
      for (const [q, min] of [["enemy=cultists", 85], ["enemy=lagavulin_matriarch", 80], ["enemy=queen", 80],
        ["enemy=bygone_effigy", 80], ["enemies=3", 38]] as const) {
        test(`enemies are drawn a readable size (${q}: at least ${min} px)`, async ({ page }) => {
          await openPreview(page, `${q}&powers=3`);
          await settledDrawnHeight(page);
          // The ink's crown moves with the idle animation (a beetle's
          // antennae: +-10%), so each enemy's tallest of a few frames.
          let tallest: number[] = [];
          for (let f = 0; f < 6; f++) {
            await page.waitForTimeout(150);
            const ink = await enemyInk(page);
            tallest = ink.map((e, i) => Math.max(e.h, tallest[i] ?? 0));
          }
          console.log(`[real stage size] ${size} ${q}: ${tallest.map((h) => h.toFixed(1)).join(", ")}`);
          expect(tallest.length).toBeGreaterThan(0);
          for (const [i, h] of tallest.entries()) expect(h, `enemy ${i}`).toBeGreaterThanOrEqual(min);
        });
      }
    }
  });
}

// A phone held upright in a browser: the bars leave the app a shorter stage
// than the first-fight measurement (C1 of the final review). Nothing may
// leave the stage or overlap another owner's chrome there.
for (const which of Object.keys(SHORT_PORTRAIT) as ShortPortrait[]) {
  const size: PhoneSize = which === "large" ? "large" : "small";
  test.describe(`${which} phone, portrait: the split on a short stage (${SHORT_PORTRAIT[which].stage} px)`, () => {
    test.use(shortPortraitContext(which));

    test("the stage is the short one", async ({ page }) => {
      await openPreview(page, "enemy=cultists");
      const h = (await realBox(page.locator(".gs-arena-stage"))).height;
      expect(Math.abs(h - SHORT_PORTRAIT[which].stage)).toBeLessThanOrEqual(2);
    });

    overlapMatrix(size, ` on a ${SHORT_PORTRAIT[which].stage} px stage`,
      (q) => ["enemy=cultists", "enemies=3", "enemies=5", "enemy=queen", "char=necrobinder&osty=1&enemies=2"].includes(q));

    // An enemy the player can see: the cramped fight keeps a player-sized
    // one `SPLIT.minCrown` (26) tall, its HP bar over the band if it must.
    if (which === "small") {
      test("a Cultist stays visible beside Osty (at least 20 px)", async ({ page }) => {
        await openPreview(page, "char=necrobinder&osty=1&enemy=cultists");
        await settledDrawnHeight(page);
        await settledDrawnHeight(page, "osty");
        let tallest = 0;
        for (let f = 0; f < 6; f++) {
          await page.waitForTimeout(150);
          tallest = Math.max(tallest, ...(await enemyInk(page)).map((e) => e.h));
        }
        const split = await page.locator(".gs-board").getAttribute("data-split");
        console.log(`[short stage] cultist beside Osty: ${tallest.toFixed(1)} px, ${split}`);
        expect(tallest).toBeGreaterThanOrEqual(20);
      });
    }
  });
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait: crowded rosters`, () => {
    test.use(phoneContext(size, "portrait"));

    // Every enemy stays a 44 px tap target however small the split draws it:
    // an armed attack lands on each one from its box's centre.
    // The large phone fits 6-8 enemies without going `cramped`, but draws
    // them ~11 px tall: the tap box must hold whatever the solver says.
    const crowds = size === "small" ? [5] : [5, 8];
    for (const n of crowds) test(`every enemy keeps a 44px tap box (${n} enemies, ${size})`, async ({ page }) => {
      test.setTimeout(400_000);
      for (let i = 0; i < n; i++) {
        await openPreview(page, `enemies=${n}&powers=3`);
        if (i === 0) console.log(`[tap box] ${size} enemies=${n}: ${await page.locator(".gs-board").getAttribute("data-split")}`);
        const box = await realBox(page.locator(`[data-enemy-index="${i}"]`));
        console.log(`[tap box] ${size} e${i}: ${box.width.toFixed(1)}x${box.height.toFixed(1)}`);
        expect(box.width).toBeGreaterThanOrEqual(44 - 0.5);
        expect(box.height).toBeGreaterThanOrEqual(44 - 0.5);
        await tapSlot(page, 2);
        await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", "2");
        await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
        await expect(page.locator("[data-last-play]")).toHaveAttribute("data-last-play", `2>${i}`);
      }
    });
  });
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait: the multi-rig bosses' parts`, () => {
    test.use(phoneContext(size, "portrait"));

    // Every arm / segment is a 44 px tap target an armed attack lands on.
    for (const { scene, parts, rig } of [
      { scene: "kaiser_crab", parts: 2, rig: "kaiser_crab/kaiser_crab.atlas" },
      { scene: "decimillipede", parts: 3, rig: "decimillipede_segment_front/decimillipede_segment_front.atlas" },
    ]) {
      test(`each part keeps a 44px tap box and takes an armed attack (${scene})`, async ({ page }) => {
        const res = await page.request.get(`/preview-rig/${rig}`);
        test.skip(!res.ok() || (await res.text()).startsWith("<"), `${rig} is not linked into public/preview-rig`);
        for (let i = 0; i < parts; i++) {
          await openPreview(page, `scene=${scene}&powers=3`);
          const part = page.locator(`[data-enemy-index="${i}"]`);
          await expect(part).toBeVisible();
          await page.waitForTimeout(400); // the crab's arms fade in with the rig
          const box = await realBox(part);
          console.log(`[part tap] ${size} ${scene} e${i}: ${box.width.toFixed(1)}x${box.height.toFixed(1)}`);
          expect(box.width).toBeGreaterThanOrEqual(44 - 0.5);
          expect(box.height).toBeGreaterThanOrEqual(44 - 0.5);
          await tapSlot(page, 2);
          await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", "2");
          await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
          await expect(page.locator("[data-last-play]")).toHaveAttribute("data-last-play", `2>${i}`);
        }
      });
    }
  });
}

test.describe("large phone: rotating mid-fight", () => {
  test.use(phoneContext("large", "portrait"));

  test("landscape drops the split, portrait brings it back", async ({ page }) => {
    await openPreview(page, "enemies=3&enemy=cultists&powers=3");
    await expect(page.locator(".gs-split")).toHaveCount(1);
    const split0 = await page.locator(".gs-board").getAttribute("data-split");
    const vp = page.viewportSize()!;
    await page.setViewportSize({ width: vp.height, height: vp.width });
    await expect(page.locator(".gs-phone-landscape")).toHaveCount(1);
    await expect(page.locator(".gs-split")).toHaveCount(0);
    await expect(page.locator(".gs-playerband")).toHaveCount(0);
    await page.setViewportSize(vp);
    await expect(page.locator(".gs-split")).toHaveCount(1);
    await page.waitForTimeout(600);
    expect(Math.abs((await drawnHeight(page)) - FIGURE.large)).toBeLessThanOrEqual(2);
    // Re-solved for the portrait stage again: the same split, and no two
    // owners' chrome overlapping.
    expect(await page.locator(".gs-board").getAttribute("data-split")).toBe(split0);
    const { rects, band } = await chromeBoxes(page);
    expect(rects.filter((x) => x.owner.startsWith("e")).length).toBeGreaterThan(0);
    const bad: string[] = [];
    for (const [i, a] of rects.entries()) {
      for (const b of rects.slice(i + 1)) {
        if (a.owner !== b.owner && hits(a.r, b.r)) bad.push(`${a.owner} ${a.what} x ${b.owner} ${b.what}`);
      }
      if (a.owner.startsWith("e") && a.r.bottom > band.top + 0.5) bad.push(`${a.owner} ${a.what} under the band top`);
    }
    expect(bad, bad.join("\n")).toEqual([]);
  });
});

/** Every `[data-enemy-index]` box, real px, in document order. */
async function enemyBoxes(page: Page): Promise<Box[]> {
  return page.evaluate(() => Array.from(document.querySelectorAll("[data-enemy-index]")).map((e) => {
    const b = e.getBoundingClientRect();
    return { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
  }));
}

/** Each box of `now` within `tol` px of the same box in `then`. */
function expectSameBoxes(now: Box[], then: Box[], what: string, tol = 0.5): void {
  expect(now.length, `${what}: enemy count`).toBe(then.length);
  for (const [i, a] of now.entries()) {
    for (const k of ["left", "top", "right", "bottom"] as const) {
      expect(Math.abs(a[k] - then[i][k]), `${what}: enemy ${i} ${k} ${then[i][k].toFixed(1)} -> ${a[k].toFixed(1)}`)
        .toBeLessThanOrEqual(tol);
    }
  }
}

/** Wait until every rig on the board has measured its head (`--rig-top`
 *  set), which is when the band settles (`Combatant.onCrown`). */
async function rigsLoaded(page: Page): Promise<void> {
  await settledDrawnHeight(page);
  if (await page.locator('.gs-combatant[data-combatant="osty"]').count()) await settledDrawnHeight(page, "osty");
  await page.waitForTimeout(300);
}

for (const size of Object.keys(PHONES) as PhoneSize[]) {
  test.describe(`${size} phone, portrait: a hit on the band`, () => {
    test.use(phoneContext(size, "portrait"));

    // The player's damage number, hurt flash and block-break play on the band,
    // never up among the enemies' chrome: the figure stands right under them.
    test("a hit plays on the figure and stays in the band", async ({ page }) => {
      await openPreview(page, "enemies=2&hit=4000");
      await rigsLoaded(page);
      const band = (await chromeBoxes(page)).band;
      const fig = await figureBox(page);
      const badge = page.locator('[data-block-badge-for="player"]');
      await expect(badge).toHaveCSS("opacity", "1");
      // The number's box on every frame of its flight.
      const samples: Box[] = await page.evaluate(() => new Promise<Box[]>((resolve) => {
        const out: Box[] = [];
        const t0 = performance.now();
        const tick = () => {
          // Every number in flight, not only the first.
          const els = Array.from(document.querySelectorAll(".gs-dmg > span"));
          for (const el of els) {
            const r = el.getBoundingClientRect();
            out.push({ left: r.left, top: r.top, right: r.right, bottom: r.bottom });
          }
          if ((out.length && !els.length) || performance.now() - t0 > 12_000) resolve(out);
          else requestAnimationFrame(tick);
        };
        tick();
      }));
      console.log(`[hit] ${size}: band ${band.top.toFixed(1)}..${band.bottom.toFixed(1)}, figure [${fig.left.toFixed(0)},${fig.top.toFixed(0)},${fig.right.toFixed(0)},${fig.bottom.toFixed(0)}], `
        + `${samples.length} frames, number top >= ${Math.min(...samples.map((b) => b.top)).toFixed(1)}, bottom <= ${Math.max(...samples.map((b) => b.bottom)).toFixed(1)}`);
      expect(samples.length).toBeGreaterThan(5);
      for (const b of samples) {
        const at = `[${b.left.toFixed(0)},${b.top.toFixed(0)},${b.right.toFixed(0)},${b.bottom.toFixed(0)}]`;
        expect(hits(b, fig) || hits(b, band), `number ${at} on the figure or band`).toBe(true);
        expect(b.top, `number ${at}: its top stays in the band`).toBeGreaterThanOrEqual(band.top - 2);
        expect(b.bottom, `number ${at}: its bottom stays in the band`).toBeLessThanOrEqual(band.bottom + 2);
      }
      // The block broke (12 -> 0): the band's shield is gone.
      await expect(badge).toHaveCSS("opacity", "0");
      expect((await badge.textContent())?.trim()).toBe("");
    });
  });
}

test.describe("large phone, portrait: effects and a pinned split", () => {
  test.use(phoneContext("large", "portrait"));

  /** The controls row's boxes (End Turn, energy, the three piles). */
  const controlBoxes = (page: Page, energy = true) => page.evaluate((sel) => Array.from(document.querySelectorAll(sel))
    .map((e) => { const b = e.getBoundingClientRect(); return { left: b.left, top: b.top, right: b.right, bottom: b.bottom }; }),
  energy ? ".gs-endturn, .gs-energy, .gs-pilebtn" : ".gs-endturn, .gs-pilebtn");
  /** The page's clock, ms since navigation: the preview's timed steps count
   *  from the board's mount, which is later. */
  const pageNow = (page: Page) => page.evaluate(() => performance.now());

  // Solved once per fight: arming a card, a power landing and the zoom
  // closing move no enemy and no control. (The band's status block steps
  // right of the zoom while it is open, by design; it is not checked here.)
  test("nothing moves while you play", async ({ page }) => {
    await openPreview(page, "enemies=3&enemy=cultists&powers=3&buff=6000");
    await rigsLoaded(page);
    const split0 = await page.locator(".gs-board").getAttribute("data-split");
    const enemies0 = await enemyBoxes(page);
    const controls0 = await controlBoxes(page);
    const fixed0 = await controlBoxes(page, false);
    expect(await pageNow(page), "recorded before the buff").toBeLessThan(6000);
    expect(enemies0).toHaveLength(3);
    expect(controls0).toHaveLength(5);
    const zoom = page.locator(".gs-zoomcard");
    await tapSlot(page, 2);
    await expect(zoom).toHaveAttribute("data-zoom-index", "2");
    expectSameBoxes(await enemyBoxes(page), enemies0, "armed");
    // The energy orb steps into the band beside the zoom, by design.
    expectSameBoxes(await controlBoxes(page, false), fixed0, "controls, armed");
    await page.getByRole("button", { name: "Close", exact: true }).tap({ timeout: 5000 });
    await expect(zoom).toHaveCount(0);
    expectSameBoxes(await enemyBoxes(page), enemies0, "zoom closed");
    // Armed again when the buff lands (a fourth power on the player). The new
    // combat state closes the zoom by itself.
    await tapSlot(page, 2);
    await expect(zoom).toHaveAttribute("data-zoom-index", "2");
    const powers = page.locator(".gs-playerband .gs-power, .gs-playerband .gs-power-more");
    const n0 = await powers.count();
    await expect.poll(async () => (await page.locator(".gs-playerband").innerHTML()).includes("Strength"), { timeout: 10_000 }).toBe(true);
    await page.waitForTimeout(2000);
    console.log(`[pinned] ${split0}; power cells ${n0} -> ${await powers.count()}`);
    await expect(zoom).toHaveCount(0);
    expectSameBoxes(await enemyBoxes(page), enemies0, "after the buff");
    expectSameBoxes(await controlBoxes(page), controls0, "controls");
    expect(await page.locator(".gs-board").getAttribute("data-split")).toBe(split0);
  });

  // A death keeps its slot empty: nobody slides over. A never-seen enemy (a
  // summon) re-solves once, then holds.
  test("a death keeps the slot empty; a summon re-solves once", async ({ page }) => {
    await openPreview(page, "enemies=3&enemy=cultists&powers=3&die=5000&dying=1");
    await rigsLoaded(page);
    const before = await enemyBoxes(page);
    const split0 = await page.locator(".gs-board").getAttribute("data-split");
    expect(await pageNow(page), "recorded before the death").toBeLessThan(5000);
    expect(before).toHaveLength(3);
    // The corpse plays out and leaves.
    await expect(page.locator("[data-enemy-index]")).toHaveCount(2, { timeout: 15_000 });
    await page.waitForTimeout(500);
    expectSameBoxes(await enemyBoxes(page), [before[0], before[2]], "survivors");
    expect(await page.locator(".gs-board").getAttribute("data-split")).toBe(split0);

    await openPreview(page, "enemies=3&enemy=cultists&powers=3&summon=5000&buff=9000");
    await rigsLoaded(page);
    const s0 = await page.locator(".gs-board").getAttribute("data-split");
    expect(await pageNow(page), "recorded before the summon").toBeLessThan(5000);
    await expect(page.locator("[data-enemy-index]")).toHaveCount(4, { timeout: 15_000 });
    await settledDrawnHeight(page);
    await page.waitForTimeout(800);
    const s1 = await page.locator(".gs-board").getAttribute("data-split");
    console.log(`[summon] ${s0} -> ${s1}`);
    expect(s1, "a summon re-solves").not.toBe(s0);
    const { rects } = await chromeBoxes(page);
    expect(rects.filter((x) => x.owner === "e99").length, "the summon is placed").toBeGreaterThan(0);
    const bad: string[] = [];
    for (const [i, a] of rects.entries()) {
      for (const b of rects.slice(i + 1)) {
        if (a.owner !== b.owner && hits(a.r, b.r)) bad.push(`${a.owner} ${a.what} x ${b.owner} ${b.what}`);
      }
    }
    expect(bad, bad.join("\n")).toEqual([]);
    const placed = await enemyBoxes(page);
    // Then it holds: the buff re-solves nothing.
    await expect.poll(async () => (await page.locator(".gs-playerband").innerHTML()).includes("Strength"), { timeout: 15_000 }).toBe(true);
    await page.waitForTimeout(1000);
    expect(await page.locator(".gs-board").getAttribute("data-split")).toBe(s1);
    expectSameBoxes(await enemyBoxes(page), placed, "after the buff");
  });

  // A stage height change of a few px (a tool row wrapping, the browser's
  // bar) does not re-solve: the enemies keep their size and place.
  // The tall single enemies are the height-bound solves.
  for (const q of ["enemies=3&enemy=cultists", "enemy=queen", "enemy=bygone_effigy", "enemies=5&enemy=cultists"]) test(`a stage height wobble does not rescale the enemies (${q})`, async ({ page }) => {
    await openPreview(page, `${q}&powers=3`);
    await rigsLoaded(page);
    const vp = page.viewportSize()!;
    const split0 = await page.locator(".gs-board").getAttribute("data-split");
    const before = await enemyBoxes(page);
    const stageH = () => page.evaluate(() => document.querySelector(".gs-arena-stage")!.getBoundingClientRect().height);
    const h0 = await stageH();
    for (const dh of [-3, 3]) {
      await page.setViewportSize({ width: vp.width, height: vp.height + dh });
      await page.waitForTimeout(600);
      // Not vacuous: the stage itself changed height.
      expect(Math.abs((await stageH()) - h0), `stage height at ${dh} px`).toBeGreaterThanOrEqual(2);
      const now = await enemyBoxes(page);
      const s = await page.locator(".gs-board").getAttribute("data-split");
      console.log(`[wobble] ${q} ${dh}: ${split0} -> ${s}; widths ${before.map((b) => (b.right - b.left).toFixed(2)).join(",")} -> ${now.map((b) => (b.right - b.left).toFixed(2)).join(",")}`);
      expect(s, `data-split at ${dh} px`).toBe(split0);
      expect(now.length).toBe(before.length);
      for (const [i, b] of now.entries()) {
        expect(Math.abs((b.right - b.left) - (before[i].right - before[i].left)), `enemy ${i} width at ${dh} px`).toBeLessThanOrEqual(0.5);
      }
      // The held solve, shifted onto the shorter stage, keeps every enemy
      // rect on it and above the band.
      const { rects, stage, band } = await chromeBoxes(page);
      for (const x of rects.filter((r) => r.owner.startsWith("e"))) {
        expect(x.r.top, `${x.owner} ${x.what} top at ${dh} px`).toBeGreaterThanOrEqual(stage.top - 0.5);
        expect(x.r.bottom, `${x.owner} ${x.what} bottom at ${dh} px`).toBeLessThanOrEqual(band.top + 0.5);
      }
    }
    await page.setViewportSize(vp);
    await page.waitForTimeout(600);
    expectSameBoxes(await enemyBoxes(page), before, "back to the first height");
    // A real change still re-solves (here, where the height binds the scale).
    if (q !== "enemies=3&enemy=cultists") {
      await page.setViewportSize({ width: vp.width, height: vp.height - 30 });
      await page.waitForTimeout(600);
      const s = await page.locator(".gs-board").getAttribute("data-split");
      console.log(`[wobble] ${q} -30: ${split0} -> ${s}`);
      expect(s, "a 30 px change re-solves").not.toBe(split0);
      await page.setViewportSize(vp);
    }
  });
});

test.describe("large phone, portrait: the Kaiser Crab", () => {
  test.use(phoneContext("large", "portrait"));

  // One solver creature for the whole rig, never a row fallback; both arms
  // take an armed attack. Skips where the crab rig is not linked into
  // public/preview-rig (see boardpreview.tsx's `?scene=` note).
  test("the crab is one split creature and both arms take an attack", async ({ page }) => {
    const res = await page.request.get("/preview-rig/kaiser_crab/kaiser_crab.atlas");
    test.skip(!res.ok(), "the kaiser_crab rig is not linked into public/preview-rig");
    for (const idx of [0, 1]) {
      await openPreview(page, "scene=kaiser_crab");
      await expect(page.locator(".gs-split")).toHaveCount(1);
      const arm = page.locator(`.gs-combatant[data-enemy-index="${idx}"]`);
      await expect(arm).toBeVisible();
      const band = await realBox(page.locator(".gs-playerband"));
      const a = await realBox(arm);
      expect(a.y + a.height, `arm ${idx} stands above the band`).toBeLessThanOrEqual(band.y + 0.5);
      // Bash, then the arm.
      const pt = await page.evaluate(() => {
        const hits = Array.from(document.querySelectorAll(".gs-handslot-hit")).map((h) => h.getBoundingClientRect());
        const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
        return { x: (hits[2].left + hits[3].left) / 2, y: stage.bottom - 30 };
      });
      await page.touchscreen.tap(pt.x, pt.y);
      await expect(page.locator(".gs-zoomcard")).toHaveAttribute("data-zoom-index", "2");
      await arm.tap({ timeout: 5000 });
      await expect(page.locator("[data-last-play]")).toHaveAttribute("data-last-play", `2>${idx}`);
    }
  });
});
