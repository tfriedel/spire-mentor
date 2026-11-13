import { useLayoutEffect, useRef, useState } from "react";
import type { Relic } from "../api";
import { NO_DESCRIPTION, useChromeInspect } from "../inspect/useInspect";
import { usePhone } from "../phone";

/** Real px of one HUD relic button on a phone (phone.css) plus the tray gap. */
const PHONE_RELIC_PX = 44;
const PHONE_RELIC_GAP = 4;
/** A phone shows at least this many relics before collapsing the rest. */
const MIN_SHOWN = 3;

/**
 * A horizontal tray of owned relics. Each relic shows its icon (or a name
 * chip when art is missing); hovering or focusing one reveals a popover with
 * the relic's name and resolved description. Relics persist for the whole run,
 * so this is rendered above every screen — not just combat.
 *
 * `skin="hud"` renders the game-band variant used by GameHud: bare relic art
 * at game size (no chip borders, no mono font), same tooltip behavior.
 */
export default function RelicTray({ relics, skin, maxShown }: {
  relics?: Relic[];
  skin?: "hud";
  /** Show at most this many relics; the rest collapse into one "+N" button
   *  (class `gs-hud-relic-more`). Unset: no cap, except the hud skin on a
   *  phone, which fits as many 44px buttons as the tray is wide. */
  maxShown?: number;
}) {
  const [active, setActive] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);
  const phone = usePhone();
  const trayRef = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<number | null>(null);
  const hasRelics = (relics?.length ?? 0) > 0;
  const autoFit = skin === "hud" && phone !== null && maxShown === undefined;
  // The tray grows to the room the band leaves it (phone.css), so its width is
  // the room for buttons. Re-measured when that room changes.
  useLayoutEffect(() => {
    const el = trayRef.current;
    if (!autoFit || !el) { setFit(null); return; }
    const measure = () => {
      // Zero width = not laid out yet (or hidden): not measured, do not collapse.
      if (el.clientWidth <= 0) { setFit(null); return; }
      const n = Math.floor((el.clientWidth + PHONE_RELIC_GAP) / (PHONE_RELIC_PX + PHONE_RELIC_GAP));
      setFit(n);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [autoFit, hasRelics]);
  const cap = maxShown ?? fit;
  // Phone: a tap opens the shared inspect sheet instead of this popover (the
  // popover is hover/focus UI; a tap would raise it AND the sheet).
  const inspect = useChromeInspect();
  const hud = skin === "hud";
  // The phone's HUD band never shows the popover: the tray scrolls sideways,
  // which clips it inside the 44px band. A tap goes to the inspect sheet (App
  // and Board host one on every screen), also while a card is armed for a
  // target (`useChromeInspect`), and "+N" never expands in place.
  const sheetOnly = hud && phone !== null;
  const popover = (i: number) => { if (!inspect.enabled && !sheetOnly) setActive(i); };
  if (!relics || relics.length === 0) return null;
  // Collapse: show cap-1 relics (never fewer than MIN_SHOWN; the row scrolls
  // if it must) and a "+N" in the last slot. Off a phone the button expands
  // the tray in place instead.
  const overflow = cap != null && relics.length > Math.max(cap, MIN_SHOWN);
  const collapsed = overflow && !expanded;
  const shown = collapsed ? Math.max(MIN_SHOWN, (cap ?? 0) - 1) : relics.length;
  const hidden = relics.length - shown;
  return (
    <div ref={trayRef} className={hud ? "gs-hud-relictray" : "mono"} data-testid="relic-tray"
      style={hud ? undefined : { display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
      {relics.slice(0, shown).map((r, i) => (
        <div key={i} style={{ position: "relative", display: "inline-flex" }}>
          <button
            type="button"
            className={hud ? "gs-hud-relic" : undefined}
            aria-label={r.name}
            title={r.name}
            onMouseEnter={() => popover(i)}
            onMouseLeave={() => setActive((cur) => (cur === i ? null : cur))}
            onFocus={() => popover(i)}
            onBlur={() => setActive((cur) => (cur === i ? null : cur))}
            onClick={(e) => {
              if (inspect.enabled) {
                e.stopPropagation();
                inspect.open({ key: `relic:${i}:${r.name}`, icon: r.image, name: r.name,
                  description: r.description || NO_DESCRIPTION });
                return;
              }
              if (sheetOnly) return;
              setActive((cur) => (cur === i ? null : i));
            }}
            style={hud ? undefined : { padding: r.image ? 1 : "3px 7px", border: "1px solid var(--border)",
              borderRadius: 4, background: "var(--bg)", cursor: "pointer", lineHeight: 0,
              display: "inline-flex", alignItems: "center" }}>
            {r.image
              ? <img src={r.image} alt={r.name} style={hud ? undefined : { height: 22, display: "block" }} />
              : <span className={hud ? "gs-hud-relicname" : undefined}
                  style={hud ? undefined : { fontSize: 11, lineHeight: 1.2 }}>{r.name}</span>}
          </button>
          {active === i && (
            <div role="tooltip"
              // The tray sits at the right of the screen, so a left-anchored
              // popover overflows the viewport (horizontal scrollbar). Anchor
              // the right edge instead so it grows leftward and stays on-screen.
              style={{ position: "absolute", top: "calc(100% + 6px)", right: 0, zIndex: 30,
                width: 220, padding: "8px 10px", background: "var(--panel)",
                border: "1px solid var(--border)", borderRadius: 6,
                boxShadow: "0 6px 18px oklch(40% 0.03 255 / .25)" }}>
              <div style={{ fontWeight: 650, fontSize: 12, marginBottom: r.description ? 4 : 0 }}>{r.name}</div>
              {r.description && (
                <div style={{ fontSize: 11, lineHeight: 1.4, color: "var(--muted)", whiteSpace: "pre-line" }}>
                  {r.description}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
      {(collapsed || (expanded && overflow)) && (
        <button type="button" className={hud ? "gs-hud-relic gs-hud-relic-more" : undefined}
          aria-label={collapsed ? `${hidden} more relics` : "fewer relics"}
          aria-expanded={!collapsed}
          onClick={(e) => {
            if (collapsed && inspect.enabled) {
              e.stopPropagation();
              inspect.open({
                key: "relics:more", name: `${hidden} more relics`,
                list: relics.slice(shown).map((r, i) => ({
                  key: `relic:${shown + i}:${r.name}`, icon: r.image, name: r.name,
                  description: r.description || NO_DESCRIPTION,
                })),
              });
              return;
            }
            if (sheetOnly) return;
            setExpanded((v) => !v);
          }}
          style={hud ? undefined : { padding: "3px 7px", border: "1px solid var(--border)", borderRadius: 4,
            background: "var(--bg)", cursor: "pointer", fontSize: 11 }}>
          {collapsed ? `+${hidden}` : "−"}
        </button>
      )}
    </div>
  );
}
