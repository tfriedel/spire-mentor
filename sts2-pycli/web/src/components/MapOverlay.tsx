import { useEffect } from "react";
import type { MapRow, MapGraph } from "../api";
import MapView from "./MapView";

/**
 * A modal overlay that shows the run's map on demand — openable from any
 * screen (combat, card reward, shop, …), not just at a `map_select` decision.
 *
 * It is purely a viewer: nodes are never selectable here (path choices still
 * happen on the dedicated map screen). In replay mode `onJump` is wired so
 * clicking a visited node seeks the timeline there. Closes on backdrop click
 * or Escape.
 */
export default function MapOverlay({
  rows,
  graphs,
  currentIdx,
  loading,
  onJump,
  onClose,
  live = false,
  gameVersion,
}: {
  rows: MapRow[];
  graphs: MapGraph[];
  currentIdx: number;
  loading?: boolean;
  /** Replay only: seek the timeline to a map decision step. */
  onJump?: (idx: number) => void;
  onClose: () => void;
  /** Live run: the AI's planned route follows the hints switch (see MapView). */
  live?: boolean;
  /** Game build of the shown run (boss art overlay, see `bossArt`). */
  gameVersion?: string | null;
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div role="dialog" aria-label="Map" onClick={onClose}
      style={{ position: "fixed", inset: 0, zIndex: 50, background: "oklch(20% 0.02 255 / .55)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      {/* A definite height (not just a max) so the sheet inside can run full
          height and scroll its own body under the torn deckle edges, the way
          the dedicated map screen does. */}
      <div onClick={(e) => e.stopPropagation()}
        style={{ background: "#0d141d", border: "1px solid rgba(180, 150, 80, 0.25)", borderRadius: 10,
          width: "min(1480px, 96vw)", height: "88vh", overflow: "hidden",
          position: "relative", display: "flex", flexDirection: "column",
          boxShadow: "0 12px 40px rgba(0, 0, 0, 0.55)" }}>
        {loading ? (
          <div style={{ padding: 24, color: "var(--muted)", fontFamily: "var(--game-font, Georgia, serif)" }}>loading map…</div>
        ) : (
          <div style={{ flex: "1 1 auto", minHeight: 0 }}>
            <MapView rows={rows} graphs={graphs} currentIdx={currentIdx} onJump={onJump} live={live} gameVersion={gameVersion} />
          </div>
        )}
        {/* The game's red back wedge, bottom-left — here it is the overlay's
            real close affordance, so the furniture carries an interaction
            rather than being painted on. (The dedicated map screen has nothing
            to go back TO, so it gets no wedge.) */}
        <button onClick={onClose} aria-label="Close" className="gsm-back" title="close the map (Esc)">
          <span aria-hidden="true">↩</span>
        </button>
      </div>
    </div>
  );
}
