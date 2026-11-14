import { useLayoutEffect, useRef } from "react";
import { assetUrl, type MapGraph } from "../api";
import MapView from "./MapView";

/**
 * A persistent, always-on-screen minimap docked above the side panel.
 *
 * Unlike {@link MapOverlay} (a modal opened on demand), this stays visible
 * across every screen — combat, card reward, shop, the map decision itself —
 * so the run's position and the path ahead are always in view. It's a pure
 * viewer: nodes aren't selectable here (path picking still happens on the full
 * map screen / overlay). Clicking the header's ⤢ button opens the full overlay.
 *
 * The body auto-scrolls to keep the current node in view, and collapses to just
 * its header to reclaim space.
 */
export default function MiniMap({
  graphs,
  currentIdx,
  collapsed,
  onToggleCollapsed,
  onExpand,
  live = false,
  gameVersion,
}: {
  graphs: MapGraph[];
  currentIdx: number;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  /** Open the full-size map overlay. */
  onExpand: () => void;
  /** Live run: the AI's planned route follows the hints switch (see MapView). */
  live?: boolean;
  /** Game build of the shown run (boss art overlay, see `bossArt`). */
  gameVersion?: string | null;
}) {
  const bodyRef = useRef<HTMLDivElement>(null);

  // Keep the current node in view as the run progresses (the map grows upward,
  // so without this the dock stays pinned to the boss/top while the player is
  // somewhere lower down). Scoped to this container so it never scrolls the page.
  useLayoutEffect(() => {
    if (collapsed) return;
    const box = bodyRef.current;
    const cur = box?.querySelector<SVGGElement>("[data-current]");
    if (!box || !cur) return;
    const br = box.getBoundingClientRect();
    const cr = cur.getBoundingClientRect();
    const target = (cr.top - br.top) + box.scrollTop - br.height / 2 + cr.height / 2;
    box.scrollTop = Math.max(0, target);
  }, [graphs, currentIdx, collapsed]);

  // Nothing to show until a graph has been captured for this run/act.
  if (!graphs || graphs.length === 0) return null;

  // The boss waiting at the top of the current act. The minimap auto-scrolls to
  // keep the *current* node centered, so the boss-name label drawn on its node
  // is usually scrolled out of view (and gone entirely when collapsed) — surface
  // it in the always-visible header so the run's destination is never hidden.
  // Pick the same graph MapView renders: the latest one at/before this step.
  const graph = graphs.filter((g) => g.idx <= currentIdx).slice(-1)[0] ?? graphs[0];
  // A10 DoubleBoss: TWO bosses stand between the run and the end of the act,
  // and they are different encounters. The game's top bar shows both — the act
  // boss with the second pinned beside it, and only the second once the player
  // stands on the first (`NTopBarBossIcon.ShouldOnlyShowSecondBossIcon`). Same
  // here: list the bosses still ahead, in the order they will be fought.
  const fm = graph?.full_map;
  const onFirstBoss = !!fm?.boss && !!fm.current_coord
    && fm.current_coord.col === fm.boss.col && fm.current_coord.row === fm.boss.row;
  // The server resolves each boss's own run_history stamp onto its meta-node
  // (`_enrich_map_nodes`), so the header can name them with the game's art. It
  // used to print "♛" — a chess queen the platform emoji font renders in
  // colour, which is the "emoji instead of a Slay the Spire icon" tell. Same
  // for the panel's own "🗺": the game's map marker ships as art.
  const ahead = [
    ...(fm?.boss && !onFirstBoss ? [fm.boss] : []),
    ...(fm?.second_boss ? [fm.second_boss] : []),
  ].filter((b) => b.name);
  const bossName = ahead.map((b) => b.name).join(" → ");

  return (
    <div style={{ borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
      <div className="mono" style={{ display: "flex", alignItems: "center", gap: 8,
        padding: "6px 12px", fontSize: 11, color: "var(--muted)" }}>
        <button type="button" onClick={onToggleCollapsed}
          title={collapsed ? "show the minimap" : "hide the minimap"}
          style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)",
            cursor: "pointer", padding: 0, width: 14 }}>
          {collapsed ? "▸" : "▾"}
        </button>
        <b style={{ color: "var(--text)", display: "inline-flex", alignItems: "center", gap: 5 }}>
          <img src={assetUrl("map_icons/map_spoils_map_marker.png")} alt="" aria-hidden="true"
            style={{ height: 13, display: "block" }} />
          map
        </b>
        {bossName && (
          <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis",
            whiteSpace: "nowrap" }} title={`next boss: ${bossName}`}>
            <span style={{ color: "var(--muted)" }}>· next boss </span>
            {ahead.map((b, i) => (
              <span key={`${b.col},${b.row}`} style={{ color: "oklch(60% 0.18 20)", fontWeight: 650,
                display: "inline-flex", alignItems: "center", gap: 4, verticalAlign: "middle" }}>
                {i > 0 && <span style={{ color: "var(--muted)", fontWeight: 400 }}>→</span>}
                {b.icon && <img src={b.icon} alt="" aria-hidden="true"
                  style={{ height: 15, display: "block" }} />}
                {b.name}
              </span>
            ))}
          </span>
        )}
        {!bossName && <span style={{ flex: 1 }} />}
        <button type="button" onClick={onExpand} title="open the full map"
          style={{ font: "inherit", background: "var(--bg)", border: "1px solid var(--border)",
            borderRadius: 4, color: "var(--muted)", cursor: "pointer", padding: "1px 7px" }}>
          ⤢ full
        </button>
      </div>
      {!collapsed && (
        <div ref={bodyRef} style={{ maxHeight: "34vh", overflow: "auto", padding: "0 4px 6px" }}>
          <MapView rows={[]} graphs={graphs} currentIdx={currentIdx} scale={0.5} compact live={live} gameVersion={gameVersion} />
        </div>
      )}
    </div>
  );
}
