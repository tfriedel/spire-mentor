import { useEffect, useState } from "react";
import { DEFAULT_BRIDGE_URL, normalizeBridgeUrl, servedByControlPlane } from "../bridge";

/**
 * "Live game…" (browser build, GH #709): connect this page to the advisor's
 * control plane on the machine running the game, so the Game tree, Live
 * search and rewind work with no server in between. The dialog is where the
 * page-side opt-in lives (the URL) and where the other two are explained:
 * the advisor must allow this page's origin, and Chrome asks for its local
 * network permission on the first request.
 */
export default function BridgeConnectDialog({
  open,
  url,
  state,
  error,
  onChange,
  onClose,
}: {
  open: boolean;
  /** The configured control-plane origin, or null. */
  url: string | null;
  state: "off" | "connected" | "unreachable";
  error: string | null;
  onChange: (url: string | null) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState(url ?? DEFAULT_BRIDGE_URL);
  useEffect(() => {
    if (open) setText(url ?? DEFAULT_BRIDGE_URL);
  }, [open, url]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;

  const normalized = normalizeBridgeUrl(text);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const status =
    state === "connected" ? "connected — the advisor answers, a run is live"
    : state === "unreachable" ? "configured, but the advisor does not answer (see below)"
    : "not connected";

  return (
    <div role="presentation" onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 60,
        display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div role="dialog" aria-label="Live game" aria-modal="true" onClick={(e) => e.stopPropagation()}
        style={{ background: "var(--panel, #1c1c22)", color: "var(--fg, #eee)", border: "1px solid var(--border, #444)",
          borderRadius: 8, padding: 18, width: "min(560px, 92vw)", display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <h2 style={{ margin: 0, fontSize: 16 }}>🎮 Live game</h2>
          <span style={{ color: "var(--muted)", fontSize: 12 }}>{status}</span>
        </div>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.45 }}>
          Connect this page to the advisor running next to the game to get the <b>Game tree</b> (rewind the real
          game to a save point), <b>Live search</b> over the live board, and <b>rewind to the viewed floor</b>.
          Three things have to be true, none of them by default:
        </p>
        {servedByControlPlane() ? (
          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.45 }}>
            This page is served by the advisor itself (the workbench bundled with the Spire Mentor mod), so it is
            connected to <code className="mono">{origin}</code> already — nothing to allow, no browser prompt.
            If it stops answering, the advisor has stopped (the game closed, or the AI coach was turned off).
          </p>
        ) : (
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13, lineHeight: 1.45 }}>
          <li>
            The advisor allows this page. In the mod&apos;s settings set <b>Browser workbench origin</b> to
            {" "}<code className="mono">{origin}</code> (or run the advisor with
            {" "}<code className="mono">--control-allow-origin {origin}</code>).
          </li>
          <li>The control plane&apos;s address below points at that machine (the default is right when the game runs on this one).</li>
          <li>Chrome asks once whether this site may reach your local network — allow it. Firefox does not ask.</li>
        </ol>
        )}
        <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 13 }}>
          control plane URL
          <input className="mono" value={text} onChange={(e) => setText(e.target.value)}
            placeholder={DEFAULT_BRIDGE_URL} aria-label="control plane URL"
            style={{ padding: "6px 8px", borderRadius: 4, border: "1px solid var(--border, #444)",
              background: "var(--bg, #111)", color: "inherit" }} />
        </label>
        {text.trim() && !normalized && (
          <div style={{ color: "var(--bad, #d66)", fontSize: 13 }}>that is not an http(s) URL</div>
        )}
        {error && <div style={{ color: "var(--warn, #c90)", fontSize: 13 }}>{error}</div>}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          {url && (
            <button className="mono" onClick={() => onChange(null)}
              style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid var(--border)", cursor: "pointer" }}>
              forget
            </button>
          )}
          <button className="mono" onClick={() => normalized && onChange(normalized)} disabled={!normalized}
            style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid var(--border)",
              cursor: normalized ? "pointer" : "default", opacity: normalized ? 1 : 0.5 }}>
            connect
          </button>
          <button className="mono" onClick={onClose}
            style={{ padding: "6px 14px", borderRadius: 4, border: "1px solid var(--border)", cursor: "pointer" }}>
            close
          </button>
        </div>
      </div>
    </div>
  );
}
