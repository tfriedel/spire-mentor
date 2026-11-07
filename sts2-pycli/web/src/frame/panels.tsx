// The frame's panels: what the mod used to draw over the game, now drawn
// beside it. Each takes the folded FrameState and renders; buttons call the
// mod's verbs through the host (modAction).

import { useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import EventLog from "../components/EventLog";
import { inHost, modAction, openWorkbenchAt, showNativeMenu, type NativeMenuItem } from "./host";
import { toTimeline } from "./ledger";
import {
  inBandLabels,
  rankedOptions,
  tieBandLines,
  type Advice,
  type AdviceOption,
  type FrameState,
  type ShownChip,
} from "./store";

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

function Btn(props: {
  label: string;
  title: string;
  action?: string;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      className={`fr-btn${props.active ? " fr-btn-on" : ""}`}
      title={props.title}
      disabled={props.disabled}
      onClick={props.onClick ?? (() => props.action && modAction(props.action))}
    >
      {props.label}
    </button>
  );
}

interface MenuItem {
  label: string;
  title: string;
  action?: string;
  onClick?: () => void;
  checked?: boolean;
  separatorBefore?: boolean;
}

function runItem(it: MenuItem) {
  if (it.onClick) it.onClick();
  else if (it.action) modAction(it.action);
}

function Menu(props: { label: string; items: MenuItem[] }) {
  return inHost() ? <NativeMenu {...props} /> : <HtmlMenu {...props} />;
}

/** In the frame host: a native popup, which opens OVER the game window (an
 * HTML dropdown would open behind it -- the game is a native window above
 * the page). */
function NativeMenu(props: { label: string; items: MenuItem[] }) {
  return (
      <div className="fr-menu">
        <button
          type="button"
          className="fr-btn"
          onClick={(e) => {
            const flat: NativeMenuItem[] = [];
            const index: number[] = [];
            props.items.forEach((it, i) => {
              if (it.separatorBefore) flat.push({ label: "", separator: true });
              index[flat.length] = i;
              flat.push({ label: it.label, title: it.title, checked: it.checked });
            });
            showNativeMenu(e.currentTarget.getBoundingClientRect(), flat, (picked) => {
              if (picked != null && index[picked] != null) runItem(props.items[index[picked]]);
            });
          }}
        >
          {props.label} ▾
        </button>
      </div>
  );
}

/** Outside the frame host (a plain browser tab): an ordinary dropdown. */
function HtmlMenu(props: { label: string; items: MenuItem[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div className="fr-menu" ref={ref}>
      <button type="button" className="fr-btn" onClick={() => setOpen(!open)}>
        {props.label} ▾
      </button>
      {open && (
        <div className="fr-menu-list">
          {props.items.map((it) => (
            <button
              key={it.label}
              type="button"
              className="fr-menu-item"
              title={it.title}
              onClick={() => {
                setOpen(false);
                runItem(it);
              }}
            >
              {it.checked ? "✓ " : ""}
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function ToolbarPanel(props: {
  state: FrameState;
  framePanels: { label: string; shown: boolean; toggle: () => void }[];
  onPanelsInside: () => void;
}) {
  const { ui, host } = props.state;
  const connected = host.mod_connected && ui != null;
  const takeover = !!ui?.takeover_enabled;
  return (
    <div className="fr-toolbar">
      <div className="fr-toolbar-row">
        <Btn
          label={ui?.power_label ?? "AI off"}
          title="Turn the AI on / off in the current style (F8)"
          action="toggle_advisor"
          active={ui != null && ui.mode !== "off"}
          disabled={!connected}
        />
        <Btn label={ui?.style_label ?? "Coach"} title="Switch style: Hints / Coach" action="switch_style" disabled={!connected} />
        {ui?.coach_style && <Btn label="Peek" title="Peek at the AI's pick for the next decision (F5)" action="peek" />}
        <span className="fr-sep" />
        <Btn label="AI once" title="Let the AI take just this decision (F6)" action="ai_once" disabled={!takeover} />
        <Btn
          label={ui?.autoplay_armed ? "Stop auto" : "AI auto"}
          title="The AI keeps playing until you stop it (F7)"
          action="ai_auto"
          disabled={!takeover}
          active={ui?.autoplay_armed}
        />
        <Btn
          label={ui?.autobattle_on ? "Stop battle" : "Autobattle"}
          title="The AI plays the fights, you make the run-level decisions (Shift+F7)"
          action="autobattle"
          disabled={!takeover}
          active={ui?.autobattle_on}
        />
        {ui?.brain?.show && (
          <Btn
            label={ui.brain.label}
            title="Advisor brain, click to switch (Shift+F8): Search thinks ahead (a search over the fight, the run net for the screens); Instinct is the end-to-end net, no search (Ironclad on v0.111.0 for now)"
            action="toggle_brain"
            disabled={!connected}
            active={ui.brain.value === "instinct"}
          />
        )}
        {ui?.partner_go.show && (
          <Btn
            label={ui.partner_go.waiting ? `▶ Partner go${ui.partner_go.seconds > 0 ? ` ${ui.partner_go.seconds}s` : ""}` : "Partner go"}
            title="Every waiting AI partner plays its turn now (Shift+F6)"
            action="partner_go"
            disabled={!ui.partner_go.waiting}
          />
        )}
        <span className="fr-sep" />
        <Btn label="−" title="Game speed down (F9)" action="speed_down" disabled={!connected} />
        <span className="fr-speed">{ui?.speed ?? "–"}</span>
        <Btn label="+" title="Game speed up (F10)" action="speed_up" disabled={!connected} />
        <span className="fr-sep" />
        <Menu
          label="Rewind"
          items={[
            { label: "⟲ Turn", title: "Undo to the start of this combat turn (Ctrl+Z)", action: "undo_turn" },
            { label: "⟲ Room", title: "Rewind to the start of this room (Ctrl+Shift+Z)", action: "restart_room" },
            { label: "⟲⟲ Back", title: "Rewind one room further back", action: "back_one_room" },
          ]}
        />
        <Menu
          label="Panels"
          items={[
            ...props.framePanels.map((p) => ({
              label: p.label,
              title: `Show / hide the ${p.label} panel`,
              checked: p.shown,
              onClick: p.toggle,
            })),
            {
              label: "Move the panels into the game",
              title: "Full-size game with the panels drawn over it, as without the frame (the in-game Panels menu brings them back out)",
              onClick: props.onPanelsInside,
              separatorBefore: true,
            },
            {
              label: "Card scores and map markers (F4)",
              title: "Card scores, map markers and enemy tags in the game",
              action: "toggle_hud",
              separatorBefore: true,
            },
            { label: "Help (F1)", title: "What everything does (in-game)", action: "help" },
            { label: "Settings", title: "Mod settings (in-game; works mid-run)", action: "settings" },
            { label: "Report a bug (Ctrl+Shift+B)", title: "Send this position and the logs", action: "report_bug" },
          ]}
        />
        <Menu
          label="Workbench"
          items={[
            { label: "Open in browser", title: "Logs, search tree, run tree", action: "open_workbench" },
            { label: "Copy state (Ctrl+Shift+C)", title: "Copy this position for the workbench", action: "copy_state" },
          ]}
        />
        {/* One line, after the buttons: the row is all the height the toolbar
            gets. The full (multi-line) status is the tooltip. */}
        <span className="fr-status" title={host.mod_connected ? (ui?.status ?? "") : undefined}>
          {!host.mod_connected ? (
            <span className="fr-dim">game: {host.game} · waiting for the mod…</span>
          ) : (
            (ui?.status ?? "")
              .split("\n")
              .filter(Boolean)
              .map((line, i) => (
                <span key={i} className={i === 0 ? "" : "fr-dim"}>
                  {i > 0 ? " · " : ""}
                  {line}
                </span>
              ))
          )}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Advice (SimHud's advice panel)
// ---------------------------------------------------------------------------

function qColor(q: number): string {
  // -1 red .. 0 amber .. +1 green, like the HUD's QColor
  const t = Math.max(-1, Math.min(1, q));
  const hue = 25 + (t + 1) * 60; // 25 (red) .. 145 (green)
  return `oklch(75% 0.15 ${hue})`;
}

function signed(v: number, digits: number): string {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
}

function OptionRow({ o, bestQ }: { o: AdviceOption; bestQ: number | null }) {
  const q = o.q ?? (o.p_win != null ? o.p_win * 2 - 1 : null);
  const delta = !o.is_best && bestQ != null && q != null;
  const shown = q == null ? null : delta ? q - bestQ! : q;
  const share = o.visit_share != null ? Math.max(0, Math.min(1, o.visit_share)) : null;
  return (
    <div className={`fr-opt${o.is_best ? " fr-opt-best" : ""}`}>
      <span className="fr-opt-label" title={o.label}>
        {o.is_best ? "★ " : ""}
        {o.label}
      </span>
      <span className="fr-opt-q" style={{ color: shown == null ? undefined : delta ? (shown < -0.0005 ? "oklch(70% 0.15 30)" : "var(--muted)") : qColor(q!) }}>
        {shown == null ? "" : delta ? signed(shown, 3) : signed(shown, 2)}
      </span>
      <span className="fr-opt-bar">
        {share != null && <span style={{ width: `${share * 100}%`, background: q != null ? qColor(q) : "var(--block)" }} />}
      </span>
      <span className="fr-opt-share">{share != null ? `${Math.round(share * 100)}%` : ""}</span>
      <span className="fr-opt-wlt">
        {o.wins != null || o.losses != null || o.ties != null
          ? `${o.wins ?? 0}/${o.losses ?? 0}/${o.ties ?? 0}`
          : o.visits != null
            ? `${o.visits}v`
            : ""}
        {o.low_confidence ? " ~" : ""}
      </span>
    </div>
  );
}

function EvalBar({ advice }: { advice: Advice }) {
  const q = advice.best_q;
  const fill = q != null ? (q + 1) / 2 : (advice.p_win ?? 0.5);
  return (
    <div className="fr-eval">
      <div className="fr-eval-text">
        eval {q != null ? signed(q, 2) : "?"}
        {advice.p_win != null && <> · est. win {Math.round(advice.p_win * 100)}%</>}
        {advice.low_confidence && " (~)"}
      </div>
      <div className="fr-eval-bar">
        <span style={{ width: `${Math.max(0, Math.min(1, fill)) * 100}%`, background: qColor(fill * 2 - 1) }} />
      </div>
    </div>
  );
}

export function AdvicePanel({ state }: { state: FrameState }) {
  const { advice, peek, ui } = state;
  const ranked = useMemo(() => (advice ? rankedOptions(advice) : []), [advice]);
  if (peek) return <div className="fr-peek">{peek}</div>;
  if (!advice) {
    return (
      <div className="fr-empty">
        {ui?.coach_style ? "Coach style: nothing before you act (Peek / F5 shows the AI's pick)." : "No advice for this screen."}
      </div>
    );
  }
  const tie = tieBandLines(inBandLabels(advice), advice.pick_reason, advice.pick_gap_hp);
  return (
    <div className="fr-advice">
      {(advice.best_q != null || advice.p_win != null) && <EvalBar advice={advice} />}
      <div className="fr-opts">
        {ranked.map((o, i) => (
          <OptionRow key={`${o.label}-${i}`} o={o} bestQ={advice.best_q} />
        ))}
      </div>
      {tie.map((l) => (
        <div key={l} className="fr-dim">
          {l}
        </div>
      ))}
      {advice.pv.length > 0 && <div className="fr-pv">line: {advice.pv.join(" · ")}</div>}
      {advice.low_confidence && <div className="fr-warn">low confidence</div>}
      {advice.source_tag && <div className="fr-note">{advice.source_tag}</div>}
      {advice.note && <div className="fr-note">{advice.note}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Coach: chips + recap / autopsy
// ---------------------------------------------------------------------------

function ChipCard({ chip, dispatch }: { chip: ShownChip; dispatch: Dispatch }) {
  return (
    <div
      className={`fr-chip fr-tone-${chip.tone}${chip.pinned ? " fr-chip-pinned" : ""}`}
      onClick={() => dispatch({ type: "pin_chip", id: chip.id, pinned: !chip.pinned })}
      title={chip.pinned ? "Pinned — click to unpin" : "Click to pin"}
    >
      <div className="fr-chip-head">
        <span className="fr-chip-title">
          {chip.pinned ? "📌 " : ""}
          {chip.title}
        </span>
        {chip.log_name && (
          <button
            type="button"
            className="fr-icon"
            title="Open this moment in the workbench"
            onClick={(e) => {
              e.stopPropagation();
              openWorkbenchAt(chip.log_name, chip.log_idx);
            }}
          >
            ↗
          </button>
        )}
        <button
          type="button"
          className="fr-icon"
          title="Close"
          onClick={(e) => {
            e.stopPropagation();
            dispatch({ type: "close_chip", id: chip.id });
          }}
        >
          ×
        </button>
      </div>
      {chip.lines.map((l, i) => (
        <div key={i} className="fr-chip-line">
          {l}
        </div>
      ))}
    </div>
  );
}

export type Dispatch = (m: import("./store").Incoming) => void;

export function CoachPanel({ state, dispatch }: { state: FrameState; dispatch: Dispatch }) {
  const p = state.coachPanel;
  if (state.chips.length === 0 && !p) {
    return <div className="fr-empty">The coach's verdicts on your turns and picks show here.</div>;
  }
  return (
    <div className="fr-coach">
      {state.chips
        .slice()
        .reverse()
        .map((c) => (
          <ChipCard key={c.id} chip={c} dispatch={dispatch} />
        ))}
      {p && (
        <div className="fr-recap">
          <div className="fr-chip-head">
            <span className="fr-chip-title">{p.title}</span>
            <button type="button" className="fr-icon" title="Close" onClick={() => dispatch({ type: "close_coach_panel" })}>
              ×
            </button>
          </div>
          {p.summary && <div className="fr-dim">{p.summary}</div>}
          {p.sections.map((s, i) => (
            <div key={i} className="fr-recap-section">
              <div className="fr-recap-heading">
                {s.heading}
                {p.log_name && s.log_idx != null && (
                  <button type="button" className="fr-icon" title="Open in the workbench" onClick={() => openWorkbenchAt(p.log_name, s.log_idx)}>
                    ↗
                  </button>
                )}
              </div>
              {s.lines.map((l, j) => (
                <div key={j} className="fr-chip-line">
                  {l}
                </div>
              ))}
            </div>
          ))}
          {p.footer && <div className="fr-dim">{p.footer}</div>}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Action log: the workbench's own EventLog over the mod's log
// ---------------------------------------------------------------------------

export function LogPanel({ state }: { state: FrameState }) {
  const list = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  // Deferred: a fight sends combat rows in bursts, and the log can rebuild
  // once per burst instead of once per row without holding up the panels.
  const log = useDeferredValue(state.log);
  const entries = useMemo(() => toTimeline(log), [log]);
  // Follow the newest entry unless the viewer scrolled up to read.
  useLayoutEffect(() => {
    const el = list.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [entries]);
  return (
    <div
      className="fr-log-list"
      ref={list}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {entries.length === 0 ? (
        // EventLog's own empty text ("load a run…") is the workbench's.
        <div className="fr-empty">Your plays and the AI's show here once a run starts.</div>
      ) : (
        <EventLog entries={entries} current={entries.length - 1} onPick={() => {}} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Workbench (the follow view of the live run)
// ---------------------------------------------------------------------------

export function WorkbenchPanel() {
  return <iframe className="fr-workbench" src="/?follow=1" title="Workbench" />;
}
