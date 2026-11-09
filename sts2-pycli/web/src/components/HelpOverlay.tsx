import { useEffect } from "react";

const GROUPS: { title: string; keys: [string, string][] }[] = [
  { title: "Mode (top bar)", keys: [["Off", "the AI stays quiet: no hints, no grades (H and P still answer)"], ["Coach", "nothing judges your options before you act; your turns and choices are graded after"], ["Hints", "the default: the AI's picks are ringed and numbered while you decide, ranked in words in the Advice panel"], ["AI plays", "the AI plays the run until you stop it (Esc)"], ["Tools → engine numbers", "win %, Q, shares and W/L/T beside the advice"]] },
  { title: "Global", keys: [["Ctrl+K / ⌘K", "command palette — every Run / Analyze / Tools command, filterable"], ["?", "toggle this help"], ["Esc", "close help / cancel targeting"]] },
  { title: "Replay", keys: [["← / →", "previous / next step"], ["Home / End", "first / last step"], ["h", "recommend (in combat)"]] },
  { title: "Play — combat", keys: [["1–9", "play the Nth card (then Nth enemy to target)"], ["e", "end turn"], ["h", "hint — score this position and show the reasoning"], ["p", "peek — the AI's next action only (Coach style)"], ["g", "play the recommended move"], ["a", "autoplay the turn; press again to end it & advance"], ["f", "play to end — finish the whole run (Esc to stop)"], ["Esc", "cancel targeting / stop autoplay"]] },
  { title: "Play — choices & form", keys: [["1–9", "pick the Nth option"], ["p", "peek — the AI's pick only (Coach style)"], ["g / a", "take the recommended best option"], ["f", "play to end — finish the whole run (Esc to stop)"], ["Esc", "stop autoplay"], ["Enter", "start a run (on the new-run form)"]] },
];

export default function HelpOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div data-testid="help-backdrop" onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "oklch(20% 0.02 255 / .5)", zIndex: 100,
        display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div className="mono" onClick={(e) => e.stopPropagation()}
        style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8,
          padding: 20, minWidth: 360, maxWidth: 480 }}>
        <div style={{ fontWeight: 700, marginBottom: 12 }}>Keyboard shortcuts</div>
        {GROUPS.map((g) => (
          <div key={g.title} style={{ marginBottom: 12 }}>
            <div style={{ color: "var(--muted)", marginBottom: 4 }}>{g.title}</div>
            {g.keys.map(([k, desc]) => (
              <div key={k} style={{ display: "flex", gap: 12, padding: "1px 0" }}>
                <span style={{ minWidth: 90, color: "var(--accent)", fontWeight: 700 }}>{k}</span>
                <span>{desc}</span>
              </div>
            ))}
          </div>
        ))}
        <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 8 }}>
          Unofficial fan project, not affiliated with or endorsed by Mega Crit. Slay the Spire 2 and its art and
          sound are © Mega Crit.
        </div>
        <button onClick={onClose} style={{ marginTop: 4 }}>close</button>
      </div>
    </div>
  );
}
