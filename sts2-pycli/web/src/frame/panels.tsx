// The frame's panels: what the mod used to draw over the game, now drawn
// beside it. Each takes the folded FrameState and renders; buttons call the
// mod's verbs through the host (modAction).

import { useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import EventLog from "../components/EventLog";
import { inHost, modAction, openWorkbenchWindow, showNativeMenu, type NativeMenuItem } from "./host";
import { toTimeline } from "./ledger";
import { estimateChip, hpRangeText } from "../advice/estimate";
import {
  inBandLabels,
  rankedOptions,
  tieBandLines,
  type Advice,
  adviceColumns,
  type AdviceOption,
  type Cols,
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

/** One choice of several, the current one lit: the Mode and Brain switches. */
function Segmented(props: {
  label: string;
  items: { key: string; label: string; title: string; action: string }[];
  value: string | undefined;
  disabled?: boolean;
}) {
  return (
    <span className="fr-seg" role="radiogroup" aria-label={props.label}>
      {props.items.map((it) => (
        <button
          key={it.key}
          type="button"
          role="radio"
          aria-checked={props.value === it.key}
          className={`fr-seg-btn${props.value === it.key ? " fr-seg-on" : ""} fr-seg-${it.key}`}
          title={it.title}
          disabled={props.disabled}
          onClick={() => props.value !== it.key && modAction(it.action)}
        >
          {it.label}
        </button>
      ))}
    </span>
  );
}

/** The Mode switch's segment; derived for an older mod that sends no `mentor_mode`. */
function mentorMode(ui: FrameState["ui"]): string | undefined {
  if (!ui) return undefined;
  if (ui.mentor_mode) return ui.mentor_mode;
  if (ui.mode === "off") return "off";
  if (ui.mode === "autopilot" || ui.autoplay_armed) return "aiplays";
  return ui.coach_style ? "coach" : "hints";
}

/**
 * The frame's toolbar, the badge's popover laid flat (SimToolbar): one Mode
 * switch instead of an on/off button plus a style button whose labels read
 * as either state or action; the Brain switch; the verbs for this decision;
 * autobattle; speed; the menus; the status in words. While the AI plays, the
 * row turns amber and leads with a Stop button nobody can miss.
 */
export function ToolbarPanel(props: {
  state: FrameState;
  framePanels: { label: string; shown: boolean; toggle: () => void }[];
  onFullscreen: () => void;
}) {
  const { ui, host } = props.state;
  const connected = host.mod_connected && ui != null;
  const takeover = !!ui?.takeover_enabled;
  const mode = mentorMode(ui);
  const aiPlaying = !!ui && (ui.mode === "autopilot" || ui.autoplay_armed || ui.autobattle_on);
  const statusLines = (ui?.status ?? "").split("\n").filter(Boolean);
  return (
    <div className={`fr-toolbar${aiPlaying ? " fr-toolbar-ai" : ""} fr-mode-${mode ?? "none"}`}>
      <div className="fr-toolbar-row">
        {aiPlaying && (
          <button type="button" className="fr-btn fr-stop" title="Stop the AI (F7)" onClick={() => modAction("stop_ai")}>
            ■ Stop AI <kbd>F7</kbd>
          </button>
        )}
        <Segmented
          label="Mode"
          value={mode}
          disabled={!connected}
          items={[
            { key: "off", label: "Off", title: "The AI stays quiet (F8 toggles)", action: "mode_off" },
            {
              key: "coach",
              label: "Coach",
              title: "Nothing before you act; Peek (F5) shows the AI's pick, the coach says afterwards where you differed",
              action: "mode_coach",
            },
            { key: "hints", label: "Hints", title: "The AI's picks are ringed and numbered while you decide", action: "mode_hints" },
            {
              key: "aiplays",
              label: ui?.mode === "autopilot" ? "Autopilot" : "AI plays",
              title: "The AI plays until you stop it (F7)",
              action: "mode_ai_plays",
            },
          ]}
        />
        {ui?.brain?.show && (
          <Segmented
            label="Brain"
            value={ui.brain.value}
            disabled={!connected}
            items={[
              {
                key: "search",
                label: "Search",
                title: "Thinks ahead: a search over the fight, the run net for the screens (Shift+F8)",
                action: "toggle_brain",
              },
              { key: "instinct", label: "Instinct", title: "The end-to-end net, no search: fast, every decision (Shift+F8)", action: "toggle_brain" },
            ]}
          />
        )}
        <span className="fr-sep" />
        {ui?.coach_style && <Btn label="Peek" title="Peek at the AI's pick for this decision (F5)" action="peek" />}
        <Btn label="Let AI pick" title="The AI takes just this decision (F6)" action="ai_once" disabled={!takeover} />
        <Btn
          label={ui?.autobattle_on ? "Stop battle" : "Autobattle"}
          title="The AI plays the fights, you make the run-level decisions (Shift+F7)"
          action="autobattle"
          disabled={!takeover}
          active={ui?.autobattle_on}
        />
        {ui?.partner_go.show && (
          <Btn
            label={ui.partner_go.waiting ? `▶ Partner go${ui.partner_go.seconds > 0 ? ` ${ui.partner_go.seconds}s` : ""}` : "Partner go"}
            title="Every waiting AI partner plays its turn now (Shift+F6)"
            action="partner_go"
            disabled={!ui.partner_go.waiting}
          />
        )}
        <span className="fr-sep" />
        <span className="fr-speed-group" title="Game speed (F9 / F10)">
          <Btn label="−" title="Game speed down (F9)" action="speed_down" disabled={!connected} />
          <span className="fr-speed">{ui?.speed ?? "–"}</span>
          <Btn label="+" title="Game speed up (F10)" action="speed_up" disabled={!connected} />
        </span>
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
              label: "Workbench (its own window)",
              title: "Follows this run: logs, search tree, run tree. Put it on another monitor or alt-tab to it",
              checked: host.workbench_open,
              onClick: () => openWorkbenchWindow(),
            },
            {
              label: "Fullscreen",
              title: FULLSCREEN_TITLE,
              onClick: props.onFullscreen,
              separatorBefore: true,
            },
            {
              label: "Highlights in the game (F4)",
              title: "Rings on the moves to make, the map route, enemy tags",
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
            {
              label: "Open the workbench",
              title: "Its own window, following this run: logs, search tree, run tree",
              onClick: () => openWorkbenchWindow(),
            },
            { label: "Open in browser", title: "The same, in your browser", action: "open_workbench_browser" },
            { label: "Copy state (Ctrl+Shift+C)", title: "Copy this position for the workbench", action: "copy_state" },
          ]}
        />
        {/* One line, after the buttons: the row is all the height the toolbar
            gets. The first status line is the badge's word ("Thinking…",
            "Building the engine 0:42"); the full status is the tooltip. */}
        <span className="fr-status" title={host.mod_connected ? (ui?.status ?? "") : undefined}>
          {!host.mod_connected ? (
            <span className="fr-dim">Waiting for the game…</span>
          ) : (
            <>
              <span className={`fr-status-dot fr-status-${ui?.advisor_status ?? "none"}`} />
              {statusLines.slice(0, 2).map((line, i) => (
                <span key={i} className={i === 0 ? "" : "fr-dim"}>
                  {i > 0 ? " · " : ""}
                  {line}
                </span>
              ))}
            </>
          )}
        </span>
        {/* Last, at the row's right end, where a window's own controls sit:
            players looked for "fullscreen" and did not find it in Panels. */}
        <button type="button" className="fr-btn fr-fullscreen" title={FULLSCREEN_TITLE} disabled={!connected} onClick={props.onFullscreen}>
          ⛶ Fullscreen
        </button>
      </div>
    </div>
  );
}

const FULLSCREEN_TITLE =
  "The game fills the whole screen, with the panels drawn over it as without the frame. " +
  "Back: Exit fullscreen, beside the in-game badge";

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

const KIND_COLOR: Record<string, string> = {
  attack: "oklch(64% 0.17 28)",
  skill: "oklch(64% 0.14 255)",
  power: "oklch(78% 0.14 80)",
  potion: "oklch(70% 0.14 155)",
  end_turn: "oklch(62% 0.02 265)",
  skip: "oklch(62% 0.02 265)",
  pick: "oklch(72% 0.09 210)",
  other: "oklch(66% 0.02 265)",
};

const VERDICT_COLOR: Record<string, string> = {
  Best: "var(--gold)",
  "Also good": "oklch(80% 0.14 150)",
  Then: "oklch(90% 0.07 85)",
  Okay: "oklch(85% 0.05 100)",
  Weak: "oklch(78% 0.14 60)",
  Mistake: "oklch(68% 0.18 28)",
};

/**
 * One ranked row, the same as the in-game panel's (SimHud.BuildOptionRow):
 * a chip in the move's colour carrying its plan step (the card badge's
 * number), the move in words -- wrapped, never cut ("Strike → Sludge ..."
 * hid which enemy) -- a verdict, the expected HP when the search has it, and
 * the AI's share. Engine columns (Q delta, W/L/T) only with the mod's "Show
 * engine numbers". Grid columns that would stay empty are not laid out.
 */
function OptionRow({ o, bestQ, cols }: { o: AdviceOption; bestQ: number | null; cols: Cols }) {
  const q = o.q ?? (o.p_win != null ? o.p_win * 2 - 1 : null);
  const delta = !o.is_best && bestQ != null && q != null;
  const shown = q == null ? null : delta ? q - bestQ! : q;
  const share = o.visit_share != null ? Math.max(0, Math.min(1, o.visit_share)) : null;
  const kind = KIND_COLOR[o.kind ?? "other"] ?? KIND_COLOR.other;
  const verdict = o.verdict ?? (o.is_best ? "Best" : "");
  return (
    <div className={`fr-opt${o.is_best ? " fr-opt-best" : ""}${o.hidden ? " fr-opt-hidden" : ""}`} style={{ gridTemplateColumns: cols.template }}>
      <span className="fr-opt-chip" style={{ borderColor: kind, background: o.step != null ? kind : "transparent" }}>
        {o.step ?? ""}
      </span>
      <span className="fr-opt-label">{o.label}</span>
      <span className="fr-opt-verdict" style={{ color: VERDICT_COLOR[verdict] ?? "var(--muted)" }}>
        {verdict}
      </span>
      {cols.hp && <span className="fr-opt-hp">{o.hp_mean != null ? `♥ ${Math.round(o.hp_mean)}` : ""}</span>}
      {cols.q && (
        <span className="fr-opt-q" style={{ color: shown == null ? undefined : delta ? (shown < -0.0005 ? "oklch(70% 0.15 30)" : "var(--muted)") : qColor(q!) }}>
          {shown == null ? "" : delta ? signed(shown, 3) : signed(shown, 2)}
        </span>
      )}
      {cols.share && (
        <>
          <span className="fr-opt-bar">
            {share != null && <span style={{ width: `${share * 100}%`, background: o.is_best ? "var(--gold)" : kind }} />}
          </span>
          <span className="fr-opt-share">{share == null ? "" : share < 0.005 ? "<1%" : `${Math.round(share * 100)}%`}</span>
        </>
      )}
      {cols.wlt && (
        <span className="fr-opt-wlt">
          {o.wins != null || o.losses != null || o.ties != null
            ? `${o.wins ?? 0}/${o.losses ?? 0}/${o.ties ?? 0}`
            : o.visits != null
              ? `${o.visits}v`
              : ""}
          {o.low_confidence ? " ~" : ""}
        </span>
      )}
    </div>
  );
}

function EvalBar({ advice }: { advice: Advice }) {
  const q = advice.best_q;
  const fill = advice.p_win ?? (q != null ? (q + 1) / 2 : 0.5);
  return (
    <div className="fr-eval">
      <div className="fr-eval-text">
        Win chance {advice.p_win != null ? `${Math.round(advice.p_win * 100)}%` : "?"}
        {advice.show_numbers && q != null && <span className="fr-dim"> · eval {signed(q, 2)}{advice.low_confidence ? " (~)" : ""}</span>}
      </div>
      <div className="fr-eval-bar">
        <span style={{ width: `${Math.max(0, Math.min(1, fill)) * 100}%`, background: qColor(fill * 2 - 1) }} />
      </div>
    </div>
  );
}

/** Instinct's headline in place of the search's win chance (GH #1222): the
 *  run and fight chances side by side, banded, the run as the bar. */
function EstimateBar({ estimate }: { estimate: NonNullable<Advice["estimate"]> }) {
  const parts = estimateChip(estimate);
  const hp = hpRangeText(estimate.hp_loss);
  const run = Math.max(0, Math.min(1, estimate.p_run));
  return (
    <div className="fr-eval" data-testid="frame-estimate">
      <div className="fr-eval-text">
        {parts.map((x, i) => (
          <span key={x.label}>
            {i > 0 && <span className="fr-dim"> · </span>}
            {x.label} <b className={`fr-est-${x.tone}`}>{x.text}</b>
          </span>
        ))}
      </div>
      <div className="fr-eval-bar">
        <span className={`fr-est-bar-${parts[0].tone}`} style={{ width: `${run * 100}%` }} />
      </div>
      {hp && <div className="fr-dim">This fight: {hp}</div>}
      <div className="fr-dim">if Instinct plays every move from here</div>
    </div>
  );
}

export function AdvicePanel({ state }: { state: FrameState }) {
  const { advice, peek, ui } = state;
  const [expanded, setExpanded] = useState(false);
  const ranked = useMemo(() => (advice ? rankedOptions(advice) : []), [advice]);
  if (peek) return <div className="fr-peek">{peek}</div>;
  if (!advice) {
    return (
      <div className="fr-empty">
        {ui?.coach_style ? "Coach style: nothing before you act (Peek / F5 shows the AI's pick)." : "No advice for this screen."}
      </div>
    );
  }
  const showNumbers = !!advice.show_numbers;
  const visible = ranked.filter((o) => !o.hidden);
  const rows = expanded ? ranked : visible.slice(0, 8);
  const more = ranked.length - rows.length;
  const cols = adviceColumns(rows, showNumbers);
  const tie = tieBandLines(inBandLabels(advice), advice.pick_reason, advice.pick_gap_hp);
  const plan = advice.plan ?? [];
  return (
    <div className="fr-advice">
      {advice.p_win != null ? <EvalBar advice={advice} /> : advice.estimate && <EstimateBar estimate={advice.estimate} />}
      {plan.length > 0 && (
        <div className="fr-plan">
          {plan.map((s, i) => (
            <span key={`${i}-${s}`} className="fr-plan-step">
              <span className="fr-plan-n">{i + 1}</span>
              {s}
            </span>
          ))}
        </div>
      )}
      <div className="fr-opts">
        {rows.map((o, i) => (
          <OptionRow key={`${o.label}-${i}`} o={o} bestQ={advice.best_q} cols={cols} />
        ))}
      </div>
      {(more > 0 || expanded) && (
        <button type="button" className="fr-more" onClick={() => setExpanded(!expanded)}>
          {expanded ? "▴ show fewer" : `… ${more} more ▾`}
        </button>
      )}
      {tie.map((l) => (
        <div key={l} className="fr-dim">
          {l}
        </div>
      ))}
      {plan.length === 0 && showNumbers && advice.pv.length > 0 && <div className="fr-pv">line: {advice.pv.join(" · ")}</div>}
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
              openWorkbenchWindow(chip.log_name, chip.log_idx);
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
                  <button type="button" className="fr-icon" title="Open in the workbench" onClick={() => openWorkbenchWindow(p.log_name, s.log_idx)}>
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
