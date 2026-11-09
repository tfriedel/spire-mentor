import { test, expect, type Page } from "@playwright/test";
import { closeMenu, menuItem, openMenu } from "../e2e/topbarMenu";

/**
 * The browser-only workbench, end to end: the Rust engine in the Web Worker
 * answers every /api call, the run survives a reload through IndexedDB,
 * seek rewinds it, and named saves round-trip.
 *
 * Deliberately shallow on game content (a few decisions) — the golden
 * parity tests own the "is the state right" question; this owns "does the
 * page work with no server".
 */

/** The reachable next rooms on the SVG map ("Go to Monster (0,1)" …). */
function mapChoices(page: Page) {
  return page.getByRole("button", { name: /^Go to / });
}

/** Clickable history rows in the live branch-tree log. */
function historyRows(page: Page) {
  return page.locator(
    '[role="button"][title="rewind the run to this step"], [role="button"][title="current position"]',
  );
}

async function startRun(page: Page, seed: string): Promise<void> {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
  // Static build: the engine dropdown offers only the in-browser Rust engine.
  await expect(page.getByLabel("engine").locator("option")).toHaveText(["Rust"]);
  await page.getByLabel("seed", { exact: true }).fill(seed);
  await page.getByRole("button", { name: "Play yourself" }).click();
  // Every run opens on Neow: the event screen with its option plaques.
  await expect(page.locator(".gs-event-option").first()).toBeVisible({ timeout: 60_000 });
}

/** Take the first enabled choice on whatever non-combat screen is up. */
async function chooseFirst(page: Page): Promise<void> {
  const event = page.locator(".gs-event-option:not([disabled])").first();
  const map = mapChoices(page).first();
  if (await event.isVisible().catch(() => false)) {
    await event.click();
    return;
  }
  if (await map.isVisible().catch(() => false)) {
    await map.click();
    return;
  }
  throw new Error("no event option or map choice visible");
}

/** Neow → (maybe a follow-up card select) → map → the first room, which
 *  every scripted seed here makes a fight. */
async function reachFirstCombat(page: Page): Promise<void> {
  // Neow → (maybe a follow-up card select) → map → first room.
  await chooseFirst(page);
  await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBeGreaterThan(1);
  // Walk until the map offers a path, then take one.
  for (let i = 0; i < 6; i++) {
    const map = mapChoices(page).first();
    if (await map.isVisible().catch(() => false)) break;
    const opt = page.locator(".gs-event-option:not([disabled])").first();
    if (await opt.isVisible().catch(() => false)) {
      await opt.click();
    } else {
      // A card select the Neow relic opened (New Leaf's transform, Precise
      // Scissors' removal, Lead Paperweight's choose screen): Skip when it is
      // offered, otherwise pick the first card and confirm.
      const confirm = page.getByRole("button", { name: /^Confirm \(/ });
      if (!(await confirm.isVisible().catch(() => false))) break;
      const skip = page.getByRole("button", { name: "Skip", exact: true });
      if (await skip.isVisible().catch(() => false)) {
        await skip.click();
      } else {
        // The Confirm row's grandparent is the CardSelect root; its first
        // <img> is the first offered card (CardFrame).
        await confirm.locator("xpath=../..").getByRole("img").first().click();
        // A pick can resolve the screen on its own (a one-card select
        // auto-confirms); only press Confirm while it is still there.
        await expect(confirm).toBeEnabled({ timeout: 10_000 }).catch(() => undefined);
        if ((await confirm.isVisible().catch(() => false)) && (await confirm.isEnabled())) {
          await confirm.click();
        }
      }
    }
    await page.waitForTimeout(500);
  }
  const map = mapChoices(page).first();
  await expect(map).toBeVisible({ timeout: 30_000 });
  await map.click();
  // The first room is a fight: the End Turn plate is the combat board's tell.
  await expect(page.locator(".gs-endturn")).toBeAttached({ timeout: 60_000 });
}

test("no server: start, play a few decisions, reload, seek, save and load", async ({ page }) => {
  await startRun(page, "E2E1");

  await reachFirstCombat(page);
  const rowsBefore = await expect
    .poll(() => historyRows(page).count(), { timeout: 60_000 })
    .toBeGreaterThan(2)
    .then(() => historyRows(page).count());

  // RELOAD: the worker rebuilds the session from IndexedDB (replay of the
  // active line) and the page rehydrates onto the same step.
  await page.reload();
  await expect.poll(() => historyRows(page).count(), { timeout: 90_000 }).toBe(rowsBefore);

  // SEEK: rewind to the root row; it becomes the current step.
  const firstRow = historyRows(page).first();
  await firstRow.click();
  await expect(firstRow).toHaveAttribute("aria-current", "step", { timeout: 60_000 });

  // SAVE (window.prompt for the name, window.alert on success) then LOAD.
  const dialogs: string[] = [];
  page.on("dialog", (d) => {
    dialogs.push(d.message());
    void (d.type() === "prompt" ? d.accept("e2e save") : d.accept());
  });
  await (await menuItem(page, "Run", "Save")).click();
  await expect.poll(() => dialogs.some((m) => /Saved/.test(m)), { timeout: 30_000 }).toBe(true);

  await (await menuItem(page, "Run", "Load")).click();
  const dialog = page.getByRole("dialog", { name: "Load save" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText("e2e save")).toBeVisible();
  // Save rows are <tr title="click to load"> (SaveLoadDialog.tsx), not buttons.
  await dialog.locator('tr[title="click to load"]').first().click();
  await expect(dialog).toBeHidden({ timeout: 60_000 });
  await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBeGreaterThan(0);
});

test("advisors run in the worker: hint chips, play best, the EV stream", async ({ page }) => {
  await startRun(page, "E2E2");
  // Static capabilities now advertise the advisors: the Mode switch is
  // offered, in Hints by default (the mod's default).
  await expect(page.getByRole("radio", { name: "Hints" })).toHaveAttribute("aria-checked", "true");
  // The streaming route, end to end through the worker: NDJSON frames in
  // order, terminal `result`, at a budget that finishes in a moment.
  const frames = await page.evaluate(async () => {
    const f = (window as unknown as { __sts2ApiFetch: typeof fetch }).__sts2ApiFetch;
    const r = await f("/api/live/decision-advisor/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ config: { rollouts: 1, sims: 0, max_steps: 8, seed: 1 } }),
    });
    const text = await r.text();
    return { status: r.status, type: r.headers.get("content-type"), lines: text.trim().split("\n") };
  });
  expect(frames.status).toBe(200);
  expect(frames.type).toBe("application/x-ndjson");
  const parsed = frames.lines.map((l) => JSON.parse(l) as { type: string; total?: number });
  expect(parsed[0].type).toBe("start");
  expect(parsed[parsed.length - 1].type).toBe("result");
  expect(parsed.filter((f) => f.type === "tick")).toHaveLength(parsed[0].total ?? -1);

  await reachFirstCombat(page);
  // H asks the combat search: the move to make is ringed and numbered (one
  // thing: a card, End Turn or a potion), the side panel ranks the moves in
  // words, and the engine's own win % stays off the board.
  await page.keyboard.press("h");
  await expect(page.locator(".gs-mark-best")).toHaveCount(1, { timeout: 60_000 });
  await expect(page.getByTestId("advice-list").getByText("Best", { exact: true })).toBeVisible();
  await expect(page.locator(".gs-badge.gs-win")).toHaveCount(0);
  // G plays the starred move: the history grows by one row.
  const rows = await historyRows(page).count();
  await page.getByRole("button", { name: "play best (G)" }).click();
  await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBe(rows + 1);
});

test("run mode in the worker: the PPO panel at Neow, play to end, autobattle", async ({ page }) => {
  await startRun(page, "E2E3");
  // Static capabilities advertise run mode: the live band's "play to end (F)"
  // is there (the lobby's "Watch the AI play" launcher is gone once a run is
  // live — GH #711), the Autobattle verb on the Mode bar.
  await expect(page.getByRole("button", { name: /play to end \(F\)/ }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Autobattle", exact: true })).toBeVisible();
  // The worker fetched /ppo/<champion>.onnx before the start and the engine
  // scored Neow with it: in Hints (the default) the net's pick is ringed on
  // the screen and ranked in the Advice panel; with Tools → engine numbers
  // the panel adds V(s). Not the "unavailable" line.
  await expect(page.locator(".wb-mark--best")).toHaveCount(1, { timeout: 120_000 });
  await (await openMenu(page, "Tools")).getByRole("menuitemcheckbox", { name: /engine numbers/ }).click();
  await expect(page.getByText(/V\(s\) = -?\d+\.\d{3}/)).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/is unavailable/)).toHaveCount(0);
  // F plays the run with the native agent (combat MCTS + the PPO brain):
  // the history grows on its own, Escape pauses it.
  const before = await historyRows(page).count();
  await page.keyboard.press("f");
  await expect.poll(() => historyRows(page).count(), { timeout: 120_000 }).toBeGreaterThan(before + 5);
  await page.keyboard.press("Escape");
  // The loop finishes its in-flight step, then the band's stop button goes
  // away and "play to end (F)" is back, and the history stops growing (its
  // list refreshes with a stride, so it may catch up by a few rows first).
  await expect(page.getByRole("button", { name: /stop \(Esc\)/ })).toHaveCount(0, { timeout: 60_000 });
  await expect(page.getByRole("button", { name: /play to end \(F\)/ }).first()).toBeVisible();
  // Let the final timeline refresh land, then the count must hold still.
  await page.waitForTimeout(2500);
  const paused = await historyRows(page).count();
  await page.waitForTimeout(2000);
  expect(await historyRows(page).count()).toBe(paused);
  // Autobattle: the agent resolves the fights and stops at the next
  // run-level decision with the change card ("Run start" at the first stop,
  // "Since your last stop — floor N" afterwards).
  await page.getByRole("button", { name: "Autobattle", exact: true }).click();
  await expect(page.getByText("Autobattle", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: /Start autobattle|Skip to next decision|Play to next decision/ }).click();
  await expect(page.getByText(/^Run start|^Since your last stop/)).toBeVisible({ timeout: 180_000 });
});

test("the replay viewer in the worker: end run → review the recorded log → play from here", async ({ page }) => {
  await startRun(page, "E2E5");
  // A few agent steps, then end the run: the engine writes the JSONL run
  // log, the worker stores it in IndexedDB, the summary offers the review.
  await page.keyboard.press("f");
  const before = await historyRows(page).count();
  await expect.poll(() => historyRows(page).count(), { timeout: 120_000 }).toBeGreaterThan(before + 5);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Running…" })).toHaveCount(0, { timeout: 60_000 });
  page.once("dialog", (d) => void d.accept());
  await (await menuItem(page, "Run", "End run")).click();
  const review = page.getByRole("button", { name: /Review replay/ });
  await expect(review).toBeVisible({ timeout: 30_000 });
  await review.click();
  // Replay mode: the recorded steps in the event log, newest run listed.
  const entries = page.locator('div[title^="#"]');
  await expect.poll(() => entries.count(), { timeout: 60_000 }).toBeGreaterThan(5);
  await expect(await menuItem(page, "Analyze", /regret review/)).toBeVisible();
  await expect(await menuItem(page, "Analyze", /run review/)).toBeVisible();
  await closeMenu(page);
  await expect(page.locator("select option", { hasText: /_Ironclad_/ }).first()).toBeAttached();
  // Play from here rebuilds the run at the viewed step and returns to play.
  const resume = page.getByRole("button", { name: /Play from here/ });
  await expect(resume).toBeEnabled({ timeout: 30_000 });
  await resume.click();
  await expect.poll(() => historyRows(page).count(), { timeout: 120_000 }).toBeGreaterThan(0);
  await expect(await menuItem(page, "Run", "End run")).toBeVisible();
  await closeMenu(page);
});

test("Coach: nothing unprompted, P peeks and rings the pick, End turn grades the turn", async ({ page }) => {
  await startRun(page, "E2E2");
  await page.getByRole("radio", { name: "Coach" }).click();
  await reachFirstCombat(page);
  // Coach: nothing marked on the board until something asks for it.
  await expect(page.locator(".gs-mark-best")).toHaveCount(0);
  // P reveals the AI's next action, one line, and rings that one move only.
  await page.keyboard.press("p");
  const peek = page.getByTestId("peek-banner");
  await expect(peek).toContainText("AI's next action:", { timeout: 60_000 });
  await expect(page.locator(".gs-mark-best")).toHaveCount(1);
  await expect(page.locator(".gs-mark-good, .gs-mark-alt")).toHaveCount(0);
  // No grade before the turn ends: combat is graded per turn, never per card.
  await expect(page.getByTestId("turn-chip")).toHaveCount(0);
  // End turn with the hand untouched: the turn is graded (a chip, unless the
  // grade is a toss-up or silent). The verdict itself is pinned in Rust
  // (last_turn_grade_wasted_turn_is_costly); here the page must take it
  // without crashing, and the peek is gone with the turn.
  const rows = await historyRows(page).count();
  await page.keyboard.press("e");
  await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBeGreaterThan(rows);
  await expect(peek).toHaveCount(0);
  await expect(page.getByRole("button", { name: /play best \(G\)/ })).toBeVisible({ timeout: 60_000 });
});
