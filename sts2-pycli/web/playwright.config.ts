import { defineConfig, devices } from "@playwright/test";

// End-to-end tests drive the real workbench: the Vite dev server (proxying
// /api to the FastAPI backend) + the in-process Rust engine. They are slow and
// stateful (one shared backend), so run serially with generous timeouts.
//
// Launch both servers via `webServer`. The backend reuses the already-built
// `sts2_rs` wheel in `.venv` (no maturin rebuild), reload off for stability.
// `STS2_E2E_BASE` points the run at a vite on its own port (`STS2_VITE_PORT=…
// scripts/dev-e2e.sh`), so a run never starts or reuses another worktree's
// :5173. The vite `webServer` follows it: it waits on that URL and, when
// nothing answers there, starts vite on that URL's port (the protocol's
// default, 80 / 443, when the URL names none).
const base = process.env.STS2_E2E_BASE ?? "http://localhost:5173";
const baseUrl = new URL(base);
const vitePort = baseUrl.port || (baseUrl.protocol === "https:" ? "443" : "80");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    baseURL: base,
    headless: true,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "PYTHONPATH=python STS2_RELOAD=0 .venv/bin/python -m workbench",
      cwd: "..",
      url: "http://127.0.0.1:8077/api/health",
      timeout: 180_000,
      reuseExistingServer: true,
      env: { STS2_HOST: "127.0.0.1", STS2_PORT: "8077" },
    },
    {
      command: `npm run dev -- --port ${vitePort} --strictPort`,
      url: base,
      timeout: 60_000,
      reuseExistingServer: true,
    },
  ],
});
