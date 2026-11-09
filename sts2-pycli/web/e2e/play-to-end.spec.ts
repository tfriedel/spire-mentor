import { test, expect, type Page } from "@playwright/test";

/**
 * Smoke coverage for the "Watch the AI play" launcher (start-form contract, live
 * progress, Esc→pause→Resume, click-to-seek). These exercise the real React
 * loop + backend that unit tests can't: vitest mocks the API and never mounts
 * the autoplay loop, and the Python parity tests cover the engine but not the
 * browser UI.
 *
 * The run is NOT played to completion (a full run is minutes of MCTS combat
 * even at cheap budgets — full-run-replay.spec.ts covers that); we let it
 * progress a few steps, then pause/resume/seek.
 */

/** Open the app on the play-mode start form. Play is the default view on a
 *  fresh session (no live run, nothing loaded for replay), so no mode toggle
 *  needs clicking — the old "▶ Play" button no longer exists. */
async function openStartForm(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
}

/** Count the clickable history rows in the live branch-tree log. */
function historyRows(page: Page) {
  return page.locator(
    '[role="button"][title="rewind the run to this step"], [role="button"][title="current position"]',
  );
}

test.describe("unified Play to end", () => {
  test("combat preset drives the sim-budget inputs (quick ↔ benchmark)", async ({ page }) => {
    await openStartForm(page);

    // Two launchers: manual "Play yourself" + the autoplay "Watch the AI play".
    await expect(page.getByRole("button", { name: "Play yourself" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Watch the AI play" })).toBeVisible();

    // The old fast/faithful "mode" select is gone; the sim-budget inputs are
    // always visible now, and the combat *preset* drives the monster/boss sim
    // values (Quick = 200/800, Benchmark = the offline sweep's 1000/4000).
    await expect(page.getByRole("combobox", { name: "mode", exact: true })).toHaveCount(0);
    const preset = page.getByRole("combobox", { name: "combat", exact: true });
    const monster = page.getByLabel("monster combat sims");
    const boss = page.getByLabel("boss combat sims");
    // Benchmark is the DEFAULT: a fresh run should be comparable to a
    // benchmarked win rate without anyone remembering to change a dropdown.
    await expect(preset).toHaveValue("benchmark");
    await expect(monster).toHaveValue("1000");
    await expect(boss).toHaveValue("4000");

    await preset.selectOption("quick");
    await expect(monster).toHaveValue("200");
    await expect(boss).toHaveValue("800");
    await preset.selectOption("benchmark");
    await expect(monster).toHaveValue("1000");
    await expect(boss).toHaveValue("4000");

    // The card-reward sim budget is GONE — the server documents the field as
    // accepted-and-ignored, so an input for it only invited people to tune a
    // number nothing reads.
    // The worker-thread box stays, disabled: HeuristicRunAgent discards the
    // value, so it must not look settable.
    const threads = page.getByLabel("root-parallel MCTS workers (not wired)");
    await expect(threads).toBeVisible();
    await expect(threads).toBeDisabled();
    await expect(page.getByRole("combobox", { name: "brain", exact: true })).toBeVisible();

    // The old "Benchmark run" button stays gone.
    await expect(page.getByRole("button", { name: "Benchmark run" })).toHaveCount(0);
  });

  test("play-to-end run shows live progress, pauses/resumes, and seeks", async ({ page }) => {
    await openStartForm(page);

    // Cheap budgets so steps tick fast. This is the successor of the old
    // "faithful" launch — same live loop (startFaithful + run-mode stepper)
    // server-side.
    await page.getByLabel("seed", { exact: true }).fill("1");
    await page.getByLabel("monster combat sims").fill("20");
    await page.getByLabel("boss combat sims").fill("20");

    await page.getByRole("button", { name: "Watch the AI play" }).click();

    // LIVE PROGRESS: the branch tree grows beyond the root while the run plays.
    // The "⛶ branch tree" button only appears once there is more than one node.
    await expect(page.getByRole("button", { name: /branch tree/ })).toBeVisible({
      timeout: 60_000,
    });
    await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBeGreaterThan(1);

    // PAUSE: Esc interrupts the loop (global handler, works on any screen).
    await page.keyboard.press("Escape");
    // Wait for the in-flight step to finish and the row count to stabilize.
    const settled = async () => {
      let prev = -1;
      for (let i = 0; i < 20; i++) {
        const n = await historyRows(page).count();
        if (n === prev) return n;
        prev = n;
        await page.waitForTimeout(1000);
      }
      return prev;
    };
    const pausedCount = await settled();
    expect(pausedCount).toBeGreaterThan(1);

    // RESUME: an explicit "resume" button off-combat, else the Board's
    // "play to end (F)" in combat.
    const resumeBtn = page.getByRole("button", { name: /resume/ });
    if (await resumeBtn.isVisible().catch(() => false)) {
      await resumeBtn.click();
    } else {
      await page.keyboard.press("f");
    }
    // Progress continues beyond where we paused.
    await expect
      .poll(() => historyRows(page).count(), { timeout: 60_000 })
      .toBeGreaterThan(pausedCount);

    // Pause again so the run stops growing before we seek.
    await page.keyboard.press("Escape");
    await settled();

    // SEEK: clicking an earlier history row rewinds the live session to it —
    // that row becomes the current step (aria-current="step").
    const firstRow = historyRows(page).first();
    await firstRow.click();
    await expect(firstRow).toHaveAttribute("aria-current", "step");
  });

  test("an autoplayed combat move shows its MCTS branch scores in the sidebar", async ({ page }) => {
    await openStartForm(page);
    await page.getByLabel("seed", { exact: true }).fill("1");
    await page.getByLabel("monster combat sims").fill("20");
    await page.getByLabel("boss combat sims").fill("20");
    await page.getByRole("button", { name: "Watch the AI play" }).click();

    // The sidebar used to read a flat "autoplaying…" for the whole run: the
    // step payload carried `scores: []`, and the panel was behind the hints
    // switch, which is OFF by default. Both had to change for this to appear.
    // Note we never touch the hints toggle — whole-run automation is exempt.
    await expect(page.getByText(/last search →/)).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText("Advice", { exact: true })).toBeVisible();

    // The rows themselves, in words (the engine's numbers are behind Tools →
    // engine numbers): one per searched action, the pick verdicted "Best".
    const list = page.getByTestId("advice-list");
    await expect(list).toBeVisible({ timeout: 30_000 });
    await expect(list.getByText("Best", { exact: true })).toBeVisible();

    // It survives a Stop — that is the "we're losing, why?" case.
    await page.keyboard.press("Escape");
    await expect(page.getByText(/last search →/)).toBeVisible();
  });

  test("the combat sim budget can be changed mid-run", async ({ page }) => {
    await openStartForm(page);
    await page.getByLabel("seed", { exact: true }).fill("2");
    await page.getByLabel("monster combat sims").fill("20");
    await page.getByLabel("boss combat sims").fill("20");
    await page.getByRole("button", { name: "Watch the AI play" }).click();

    // A live run reports the budget it is ACTUALLY searching at — the whole
    // point, since `set_run_mode` used to accept a new budget and then leave
    // the running agent on the old one with nothing on screen to say so.
    const searchingAt = page.getByText(/searching at \d+\/\d+/);
    await expect(searchingAt).toBeVisible({ timeout: 60_000 });
    await expect(searchingAt).toHaveText("searching at 20/20");

    await page.keyboard.press("Escape");

    // Typing a new budget does NOT post it — an explicit apply does, so a step
    // landing mid-keystroke can't pick up a half-typed number.
    const monster = page.getByLabel("monster combat sims").last();
    await monster.fill("40");
    await expect(searchingAt).toHaveText("searching at 20/20");

    const apply = page.getByRole("button", { name: /apply 40\/20/ });
    await expect(apply).toBeVisible();
    await apply.click();
    await expect(searchingAt).toHaveText("searching at 40/20");
    // Applied — so the affordance goes away until the boxes diverge again.
    await expect(apply).toHaveCount(0);
  });
});
