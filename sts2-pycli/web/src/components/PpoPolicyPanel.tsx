import { useEffect, useRef, useState } from "react";
import { api, type PpoPolicyPayload } from "../api";
import { isStaticMode } from "../transport";
import { useHints } from "../hints/useHints";
import { CollapseButton, PanelGrip } from "./panelChrome";
import { usePanelFlag, usePanelHeight } from "../panelPrefs";

/** Off-combat PPO run-policy view: the impl toggle (rust ONNX / python torch)
 *  that "play best (G)" / autoplay use, plus per-option probability bars and
 *  V(s) whenever the endpoint returns a distribution. Which run policy is
 *  active (heuristic / PPO) is chosen elsewhere (the top-level run
 *  mode selector) — this panel only shows PPO's own view when it applies.
 *  Renders nothing in combat / at game over — those aren't PPO's decisions.
 *
 *  Gated on the global hints switch: this panel names the net's preferred
 *  option right under the buttons you are about to press, which is the answer
 *  to the decision you are making (and, in autobattle mode, the very verdict the
 *  drill withholds until after you commit). Hints off ⇒ it does not render and
 *  does not even fetch. */

/** Bounds for the resizable body, in px. The floor is about two option rows —
 *  below that the panel says nothing the collapsed header doesn't. */
const MIN_H = 56;
const MAX_H = 800;
const DEFAULT_H = 220;
const HEIGHT_KEY = "wb.ppo.height";
const COLLAPSED_KEY = "wb.ppo.collapsed";

export default function PpoPolicyPanel({
  live,
}: {
  live: { decision: string; meta?: { floor?: number | null } } | null;
}) {
  const hints = useHints();
  const [payload, setPayload] = useState<PpoPolicyPayload | null>(null);
  const [impl, setImpl] = useState<string>("rust");
  // A shop lists every buy AND every card-removal target — thirty-odd rows,
  // which used to push the board itself off the screen because this dock grew
  // with its content. It is now capped (and scrolls inside), foldable to its
  // header, and draggable to whatever height its reader wants. Both prefs
  // persist: someone who tucked it away does not want it back next floor.
  const [collapsed, setCollapsed] = usePanelFlag(COLLAPSED_KEY, false);
  const [height, setHeight] = usePanelHeight(HEIGHT_KEY, DEFAULT_H, MIN_H, MAX_H);
  const bodyRef = useRef<HTMLDivElement>(null);
  const decision = live?.decision ?? "";
  const inactive =
    !hints.on || !live || decision === "combat_play" || decision === "game_over";
  useEffect(() => {
    if (inactive) {
      setPayload(null);
      return;
    }
    let cancelled = false;
    api.live
      .ppoPolicy()
      .then((p) => {
        if (cancelled) return;
        setPayload(p);
        if (p) setImpl(p.impl);
      })
      .catch(() => setPayload(null));
    return () => {
      cancelled = true;
    };
    // `live` is a fresh object every step, so same-floor multi-decision
    // sequences (event pages, reward chains) still refetch.
  }, [live, inactive]);
  async function toggleImpl() {
    const next = impl === "rust" ? "python" : "rust";
    const r = await api.live.runMode({ ppo_impl: next });
    setImpl(r.ppo_impl);
    setPayload(await api.live.ppoPolicy().catch(() => null));
  }
  if (inactive || !payload) return null;
  // The e2e advisor answers this panel too: its classes, no V(s), no backend.
  const e2e = payload.policy === "e2e";
  const who = e2e ? "Instinct" : "PPO";
  // Folded, the header still carries the one line the panel exists to say —
  // the net's pick — so collapsing costs reach, not the answer.
  const best = payload.options.find((o) => o.is_best);
  return (
    // A FLOW dock under the screen column, not an overlay on top of it: App
    // renders this as a sibling of the scrolling screen box, so the box is
    // `flex: 1` against this panel's own height and the screen re-fits into
    // what is left. It was `position: sticky; bottom: 0` INSIDE that box,
    // which kept it on screen the way an overlay does — laid over the bottom
    // of the screen under it. On a card reward that bottom strip is the
    // advisor rail (the per-option EV chips), so the panel hid the very
    // evidence it was sitting beside; folded, it still clipped the chips.
    // Out here it is visible on every non-combat screen without covering
    // anything — so nothing may put it back into the scroll box, and nothing
    // needs `position` or a `z-index` to keep it in view.
    <div
      className="mono"
      style={{
        flex: "0 0 auto",
        borderTop: "1px solid var(--border)",
        background: "var(--panel)",
      }}
    >
      {!collapsed && (
        <PanelGrip
          grow="up"
          label={`drag to resize the ${who} panel`}
          min={MIN_H}
          max={MAX_H}
          measure={() => bodyRef.current?.offsetHeight ?? height}
          onResize={setHeight}
        />
      )}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
        gap: 8, padding: collapsed ? "6px 8px" : "2px 8px 6px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0, flex: 1 }}>
          <CollapseButton collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)}
            label={`the ${who} panel`} />
          <span style={{ color: "var(--muted)", flex: "0 0 auto" }}
            title={e2e ? "Instinct's probabilities over this decision's options — a policy's, not win chances" : undefined}>
            {e2e ? "Instinct" : "PPO backend"}
          </span>
          {collapsed && best && (
            <span title={best.label} style={{ minWidth: 0, overflow: "hidden",
              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              <span style={{ color: "var(--muted)" }}>· </span>
              <b>{best.label}</b>
              {best.prob != null && (
                <span style={{ color: "var(--muted)" }}> {(best.prob * 100).toFixed(1)}%</span>
              )}
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: 6, flex: "0 0 auto" }}>
          {/* The browser build has one backend (tract in the worker); the
              toggle would only relabel it. */}
          {!isStaticMode() && !e2e && (
            <button onClick={toggleImpl} title="PPO inference backend">
              {impl}
            </button>
          )}
        </div>
      </div>
      {!collapsed && (
        // Capped and internally scrolled. The px cap is the dragged
        // preference; the `vh` half of the `min()` is the guarantee that no
        // stored height — from a desktop, from a rotated tablet — can hand the
        // dock more than half the screen the game is being played on.
        <div ref={bodyRef} style={{ maxHeight: `min(${height}px, 45vh)`, overflowY: "auto",
          padding: "0 8px 8px" }}>
          {!payload.ppo_available && (
            <div style={{ color: "var(--muted)" }}>
              {who} {e2e ? "has no pick here" : "net unavailable"}{payload.error ? `: ${payload.error}` : ""}
            </div>
          )}
          {payload.ppo_available && (
            <>
              {payload.value != null && <div>V(s) = {payload.value.toFixed(3)}</div>}
              {payload.options.map((o, i) => (
                <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <div
                    title={o.label}
                    style={{
                      // Flexible label column: a hard 140px truncated even short
                      // names ("Take Stone Armor…") when the panel had room.
                      width: "40%",
                      minWidth: 140,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontWeight: o.is_best ? 700 : 400,
                    }}
                  >
                    {o.label}
                  </div>
                  <div style={{ flex: 1, background: "var(--border)", height: 8 }}>
                    <div
                      style={{
                        width: `${((o.prob ?? 0) * 100).toFixed(1)}%`,
                        height: 8,
                        background: o.is_best ? "var(--accent)" : "var(--muted)",
                      }}
                    />
                  </div>
                  <div style={{ width: 52, textAlign: "right" }}>
                    {o.prob == null ? "—" : `${(o.prob * 100).toFixed(1)}%`}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
