import { test, expect, devices, type Page } from "@playwright/test";
import { reachCombat } from "./phoneHarness";
import { menuItem } from "./topbarMenu";

/**
 * The workbench on an Android tablet in landscape.
 *
 * Everything in the combat scene is authored in absolute px against a desktop
 * window, and nothing used to re-derive it from the box actually available. On
 * a 1024x640 tablet that meant the chrome ate 350 of the 640px, the fixed
 * 320px side panel ate a third of the width, and the ~290px of stage left over
 * clipped the PLAYER off the top of the screen entirely — you could not see
 * your own character or read a card.
 *
 * These are the assertions that would have caught it: the actors are inside
 * the frame, the page does not overflow, the tool row is reachable rather than
 * silently cut off, and a held card can be read without a mouse.
 *
 * Galaxy Tab S9 landscape (1024x640 CSS px, dpr 2.5, touch) is the tightest
 * mainstream Android tablet in landscape — anything larger has more room.
 */
test.use({ ...devices["Galaxy Tab S9 landscape"] });

/** Walk on from the first fight until the card reward for winning it opens.
 *  The workbench's own "play best" button drives the fight — this test is about
 *  the screen AFTER it, not about how the cards were played. */
async function reachCardReward(page: Page): Promise<void> {
  await reachCombat(page);
  const best = page.getByRole("button", { name: /play best/ });
  for (let i = 0; i < 40; i++) {
    if (await page.locator(".gs-cardreward-wrap").count()) {
      await page.waitForTimeout(800); // let the card art land before measuring
      return;
    }
    if ((await best.count()) && (await best.first().isEnabled())) await best.first().click();
    await page.waitForTimeout(700);
  }
  throw new Error("never reached a card reward");
}

/** Walk the map to a rest site and take Smith, which opens the upgrade select
 *  (the "Upgrade a card" screen with the before -> after preview).
 *
 *  Seeded to a run that meets one early: the screen under test is the same
 *  whichever rest site opens it, so there is no reason to make the walk long. */
async function reachRestSiteSmith(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
  await page.getByLabel("seed", { exact: true }).fill("rest7");
  await page.getByLabel("monster combat sims").fill("20");
  await page.getByLabel("boss combat sims").fill("20");
  await page.getByRole("button", { name: "Play yourself" }).click();
  await expect(page.locator(".gs-hud")).toBeVisible({ timeout: 90_000 });

  const best = page.getByRole("button", { name: /play best/ });
  for (let i = 0; i < 80; i++) {
    const smith = page.getByRole("button", { name: /Smith/i }).first();
    if (await smith.isVisible().catch(() => false)) {
      await smith.click();
      await expect(page.getByRole("heading", { name: "Upgrade a card" })).toBeVisible();
      return;
    }
    // Prefer a rest node whenever the map offers one; otherwise take anything.
    const rest = page.locator('g[data-can][aria-label*="Rest"]').first();
    if (await rest.count()) { await rest.click(); await page.waitForTimeout(1200); continue; }
    if ((await best.count()) && (await best.first().isEnabled())) {
      await best.first().click();
      await page.waitForTimeout(700);
      continue;
    }
    // `has-text` matches a SUBSTRING anywhere in the button, and the timeline's
    // hidden "rewind" controls carry one — hence the exact accessible name and
    // the visibility check, not a text match.
    const fallbacks = [
      page.locator(".gs-event-option:not(:disabled)").first(),
      page.locator(".gs-treasure-hit:not(:disabled)").first(),
      page.locator("g[data-can]").first(),
      page.getByRole("button", { name: "Skip", exact: true }).first(),
    ];
    for (const loc of fallbacks) {
      if (await loc.isVisible().catch(() => false)) { await loc.click(); break; }
    }
    await page.waitForTimeout(900);
  }
  throw new Error("never reached a rest site");
}

test.describe("tablet in landscape", () => {
  test.setTimeout(240_000);

  test("fits the whole fight on screen and keeps the hand readable", async ({ page }) => {
    await reachCombat(page);

    // --- the page itself fits the viewport (no scrolling the app away) ---
    const doc = await page.evaluate(() => ({
      w: document.documentElement.scrollWidth,
      h: document.documentElement.scrollHeight,
      vw: window.innerWidth,
      vh: window.innerHeight,
    }));
    expect(doc.w).toBeLessThanOrEqual(doc.vw);
    expect(doc.h).toBeLessThanOrEqual(doc.vh);

    // --- every combatant is inside the stage ---
    // The rigs overscan upward by a few px of TRANSPARENT canvas (raised
    // weapons, ears), which the stage clips on a desktop too — 24px of slack
    // covers that. It does not cover the 396px of Ironclad that used to be
    // above the top edge.
    const clip = await page.evaluate(() => {
      const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
      return Array.from(document.querySelectorAll(".gs-sprite-wrap")).map((el) => {
        const r = el.getBoundingClientRect();
        return { above: Math.round(stage.top - r.top), below: Math.round(r.bottom - stage.bottom) };
      });
    });
    expect(clip.length).toBeGreaterThan(1); // player + at least one enemy
    for (const c of clip) {
      expect(c.above).toBeLessThanOrEqual(24);
      expect(c.below).toBeLessThanOrEqual(0);
    }

    // --- the hand is on the stage, not off the bottom of the window ---
    const slots = page.locator(".gs-handslot");
    await expect(slots.first()).toBeVisible();
    const handTops = await page.evaluate(() => {
      const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
      return Array.from(document.querySelectorAll(".gs-handslot")).map(
        (el) => Math.round(el.getBoundingClientRect().top - stage.top),
      );
    });
    for (const top of handTops) expect(top).toBeGreaterThan(0);
  });

  test("gives the board the full width and the side panel a toggle", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByLabel("seed", { exact: true })).toBeVisible();

    // 1024px is too narrow to spend 320 of it on the panel, so it starts closed
    // and the board owns the whole width.
    const side = page.locator(".wb-side");
    await expect(side).toBeHidden();
    const toggle = page.locator(".wb-side-toggle");
    await expect(toggle).toBeVisible();

    // Opening it lays it OVER the board (a drawer), not beside it — otherwise
    // it would take the width back.
    await toggle.click();
    await expect(side).toBeVisible();
    await expect(side).toHaveClass(/wb-side--overlay/);
    await toggle.click();
    await expect(side).toBeHidden();
  });

  test("the top tool row stays reachable instead of being clipped", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByLabel("seed", { exact: true })).toBeVisible();

    // The row used to hold ~25 buttons and simply CUT the tail off at this
    // width — Load and the review tools were unreachable. They live in menus
    // now, so the LAST trigger fits without scrolling, and Load is one click
    // deep.
    const tools = page.getByRole("button", { name: /^Tools/ });
    await expect(tools).toBeInViewport();
    const load = await menuItem(page, "Run", "Load");
    await expect(load).toBeInViewport();
  });

  test("the reward screen fits the trio and the Skip plate on screen", async ({ page }) => {
    await reachCardReward(page);

    // The scene is `overflow: hidden` — a composition taller than its box is
    // not scrolled, it is CUT, and what got cut was the bottom of the cards and
    // the whole Skip plate. The bug was invisible to a height check on the page
    // (nothing overflowed the document) which is why this measures the SCENE.
    const fit = await page.evaluate(() => {
      const scene = document.querySelector(".gs-cardreward")!.getBoundingClientRect();
      const cards = [...document.querySelectorAll(".gs-cardreward-slot")].map((el) =>
        el.getBoundingClientRect());
      const skip = document.querySelector(".gs-cardreward-skip")?.getBoundingClientRect();
      return {
        n: cards.length,
        cardsBelow: Math.max(...cards.map((r) => Math.round(r.bottom - scene.bottom))),
        cardsAbove: Math.max(...cards.map((r) => Math.round(scene.top - r.top))),
        skipBelow: skip ? Math.round(skip.bottom - scene.bottom) : 0,
      };
    });
    expect(fit.n).toBeGreaterThan(0);
    expect(fit.cardsAbove).toBeLessThanOrEqual(0);
    expect(fit.cardsBelow).toBeLessThanOrEqual(0);
    expect(fit.skipBelow).toBeLessThanOrEqual(0);
  });

  test("the upgrade preview does not bury the cards it is previewing", async ({ page }) => {
    await reachRestSiteSmith(page);

    // Empty, the preview used to reserve two full card heights — ~290px of a
    // 517px screen for one line of placeholder text, which pushed the deck it
    // was previewing off the bottom. There is no hover on a tablet, so there is
    // nothing for that reservation to hold still.
    const firstCard = page.locator(".gs-card").first();
    await expect(firstCard).toBeInViewport();
    const before = await page.evaluate(() => ({
      row: document.querySelector<HTMLElement>('[data-testid="upgrade-preview-row"]')!
        .getBoundingClientRect().height,
      card: document.querySelector(".gs-card")!.getBoundingClientRect(),
      vh: window.innerHeight,
    }));
    expect(before.row).toBeLessThan(120);
    expect(before.card.bottom).toBeLessThanOrEqual(before.vh);

    // A tap fills it — the before/after comparison is the point of the screen.
    await firstCard.click();
    await expect(page.getByText("upgraded")).toBeVisible();
  });

  test("a tap raises a held card before a second tap plays it", async ({ page }) => {
    await reachCombat(page);

    const slots = page.locator(".gs-handslot");
    const before = await slots.count();
    expect(before).toBeGreaterThan(0);

    // There is no hover on a touchscreen, and the fan is deliberately cropped
    // by the stage's bottom edge, so the first tap has to raise the card clear
    // to be read rather than play it blind.
    const first = slots.first();
    await first.click();
    await expect(first).toHaveClass(/gs-hovered/);
    await expect(slots).toHaveCount(before);

    // Raised means readable: the card's whole box is inside the stage's clip.
    const lifted = await page.evaluate(() => {
      const stage = document.querySelector(".gs-arena-stage")!.getBoundingClientRect();
      const r = document
        .querySelector(".gs-handslot.gs-hovered .gs-handslot-lift")!
        .getBoundingClientRect();
      return { above: Math.round(stage.top - r.top), below: Math.round(r.bottom - stage.bottom) };
    });
    expect(lifted.above).toBeLessThanOrEqual(0);
    expect(lifted.below).toBeLessThanOrEqual(0);

    // Second tap commits it.
    await first.click();
    await expect(slots).toHaveCount(before - 1);
  });
});
