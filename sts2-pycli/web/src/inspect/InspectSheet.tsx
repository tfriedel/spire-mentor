import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { useInspect, useInspectHost, type InspectItem } from "./useInspect";
import { usePhone } from "../phone";

/** Landscape: a gap narrower than this between the player and the first enemy
 *  is too tight to read in, so the sheet goes top-centre instead. */
const MIN_GAP = 240;
const MARGIN = 8;

function Row({ it }: { it: InspectItem }) {
  return (
    <div className="gs-inspect-row" style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
      {it.icon && <img src={it.icon} alt="" style={{ width: 32, height: 32, flex: "none" }} />}
      <div style={{ minWidth: 0 }}>
        <div className="gs-inspect-name" style={{ fontWeight: 700, whiteSpace: "pre-line" }}>
          {it.name}{it.amount != null && <span className="gs-inspect-amt"> {it.amount}</span>}
        </div>
        {it.description && (
          <div className="gs-inspect-desc" style={{ color: "var(--muted)", whiteSpace: "pre-line" }}>
            {it.description}
          </div>
        )}
      </div>
    </div>
  );
}

/** The tap-to-inspect sheet. Rendered by Board as a child of the stage but
 *  OUTSIDE `.gs-stage-scale`, so its text is real px and it overlays the stage
 *  (position: absolute) without taking layout height — the stage never
 *  resizes. It is confined above the hand: it must never cover what the player
 *  taps next. On a stage too short for its text it scrolls inside. Tapping it
 *  closes it.
 *
 *  `screen`: the host for every screen WITHOUT a board (map, shop, rewards,
 *  events), rendered by App inside the content area under the HUD band, so a
 *  relic tap explains the relic there too. Fixed to the top of that area in
 *  real px, never over the band; a tap anywhere outside it closes it. */
export default function InspectSheet({ screen = false }: { screen?: boolean }) {
  useInspectHost();
  const { item, close } = useInspect();
  const phone = usePhone();
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<CSSProperties>({});

  useLayoutEffect(() => {
    const el = ref.current;
    const stage = el?.parentElement;
    if (!item || !el || !stage) return;
    const measure = () => {
      const st = stage.getBoundingClientRect();
      if (screen) {
        const top = Math.max(st.top, 0) + MARGIN;
        const width = Math.min(360, st.width - 2 * MARGIN);
        setBox({ position: "fixed", top, left: st.left + (st.width - width) / 2, width,
          maxHeight: Math.max(0, Math.min(st.bottom, window.innerHeight) - top - MARGIN) });
        return;
      }
      const hand = stage.querySelector(".gs-hand")?.getBoundingClientRect();
      // Never reach into the hand: cap the height at its resting top edge. On
      // a short stage that can be very little; the sheet scrolls inside.
      const maxHeight = Math.max(0, (hand ? hand.top : st.bottom) - st.top - 2 * MARGIN);
      let left = MARGIN;
      let width = st.width - 2 * MARGIN;
      if (phone === "landscape") {
        // Each side's box, widened by its power column: the column is placed
        // absolutely beside the creature, outside the side's own rect.
        const reach = (side: Element) => [side, ...Array.from(side.querySelectorAll(".gs-powers-col"))]
          .map((n) => n.getBoundingClientRect()).filter((r) => r.width > 0);
        const playerSide = stage.querySelector(".gs-side-player");
        const playerRight = playerSide ? Math.max(...reach(playerSide).map((r) => r.right)) : null;
        const enemyLeft = Math.min(...Array.from(stage.querySelectorAll(".gs-side"))
          .filter((n) => !n.classList.contains("gs-side-player"))
          .flatMap(reach).map((r) => r.left));
        if (playerRight != null && Number.isFinite(enemyLeft)) {
          const gap = enemyLeft - playerRight;
          if (gap >= MIN_GAP) { left = playerRight - st.left; width = gap; }
        }
        if (width === st.width - 2 * MARGIN) {
          width = Math.min(360, width);
          left = (st.width - width) / 2;
        }
      }
      setBox({ left, width, maxHeight });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(stage);
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, [item, phone, screen]);

  // Off the board nothing closes the sheet on a tap elsewhere (the stage does
  // that in combat), so the document does. A relic's own tap stops
  // propagation, so tapping the open relic again still toggles it.
  useEffect(() => {
    if (!item || !screen) return;
    const onDoc = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) close();
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [item, screen, close]);

  useEffect(() => {
    if (!item) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [item, close]);

  if (!item) return null;
  return (
    <div ref={ref} className="gs-inspect" role="dialog" aria-label={item.name} aria-live="polite"
      onClick={(e) => { e.stopPropagation(); close(); }}
      style={{ position: "absolute", top: MARGIN, left: MARGIN, right: box.width ? undefined : MARGIN,
        zIndex: 40, overflowY: "auto", boxSizing: "border-box", padding: "10px 12px", fontSize: 14,
        lineHeight: 1.4, color: "var(--text)", background: "var(--panel)",
        border: "1px solid var(--border)", borderRadius: 8,
        boxShadow: "0 6px 18px oklch(0% 0 0 / .45)", ...box }}>
      <Row it={item} />
      {item.list && item.list.length > 0 && (
        <div className="gs-inspect-list" style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
          {item.list.map((it) => <Row key={it.key} it={it} />)}
        </div>
      )}
    </div>
  );
}
