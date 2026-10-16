/**
 * The live-game bridge from the BROWSER build (GH #709).
 *
 * The Python workbench reaches the advisor's control plane (the SpireMentor
 * mod's rewind lane: `GET /status`, `/tree`, `/snapshots`, `POST /restore`,
 * `/search_tree` on `127.0.0.1:8079`) through a same-origin proxy. The
 * browser-only build has no server, so the page calls the control plane
 * itself — a cross-origin fetch from `https://your-workbench.example.com` to a
 * loopback `http://` address. That clears the browser's gates only with two
 * opt-ins, both off by default:
 *
 *  - on the advisor: `--control-allow-origin https://your-workbench.example.com`
 *    (the mod's "Browser workbench origin" setting), which makes the control
 *    plane answer CORS + Private Network Access headers for that origin and
 *    no other;
 *  - in this page: the control plane's URL, stored here in `localStorage`
 *    (nothing is probed until the user enters one).
 *
 * Measured 2026-09-08 (docs/browser-workbench.md, "Live-game bridge"):
 * Firefox 151 then just works; Chromium 149 additionally asks the user for
 * the "local network" permission (its Local Network Access check) the first
 * time — the fetch waits on that prompt. `targetAddressSpace` is Chrome's
 * hint that a loopback target is intended.
 *
 * The workbench the Spire Mentor mod ships is different: the advisor's
 * control plane serves the page itself, so the bridge is this page's own
 * origin — connected from the first load, no opt-in, no prompt. That build
 * sets `VITE_BRIDGE_SAME_ORIGIN=1` (spire-mentor-mod/package.ps1).
 */

const KEY = "sts2.bridgeUrl";
/** Stored when the user disconnects a same-origin build (see setBridgeBase). */
const OFF = "off";

/** True for the build the mod bundles: the control plane serves this page. */
export function servedByControlPlane(): boolean {
  return import.meta.env.VITE_BRIDGE_SAME_ORIGIN === "1";
}

/** What the dialog pre-fills: the advisor's default control-plane port. */
export const DEFAULT_BRIDGE_URL = "http://127.0.0.1:8079";

/** Normalise a user-entered control-plane URL: scheme required, no path. */
export function normalizeBridgeUrl(raw: string): string | null {
  const s = raw.trim();
  if (!s) return null;
  let url: URL;
  try {
    url = new URL(s.includes("://") ? s : `http://${s}`);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  return url.origin;
}

/** The configured control-plane origin, or null. The default is null, or
 *  this page's own origin when the control plane serves it. */
export function bridgeBase(): string | null {
  try {
    const v = localStorage.getItem(KEY);
    if (v === OFF) return null;
    if (v) return normalizeBridgeUrl(v);
  } catch {
    /* no storage: fall through to the default */
  }
  return servedByControlPlane() ? window.location.origin : null;
}

/** Remember (or forget, with null) the control-plane origin. */
export function setBridgeBase(url: string | null): void {
  try {
    if (url) localStorage.setItem(KEY, url);
    // A same-origin build would otherwise reconnect itself on the next read.
    else if (servedByControlPlane()) localStorage.setItem(KEY, OFF);
    else localStorage.removeItem(KEY);
  } catch {
    /* private mode: the setting lives for this page load only */
  }
}

/** A request straight to the control plane (base must be configured). */
export function bridgeFetch(path: string, init?: RequestInit): Promise<Response> {
  const base = bridgeBase();
  if (!base) return Promise.reject(new Error("no live-game bridge configured"));
  // `targetAddressSpace` is Chrome's Local Network Access hint, for a
  // cross-origin target only; unknown RequestInit keys are ignored elsewhere.
  if (base === window.location.origin) return fetch(`${base}${path}`, init);
  const extra = { targetAddressSpace: "loopback" } as unknown as RequestInit;
  return fetch(`${base}${path}`, { ...init, ...extra });
}
