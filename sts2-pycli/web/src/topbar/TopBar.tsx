import { gameVersionBody } from "../gameVersion";
import { useMemo, type ReactNode } from "react";
import Menu, { type MenuItemSpec } from "./Menu";
import CommandPalette from "./CommandPalette";
import type { PaletteEntry } from "./paletteEntries";
import type { Command } from "./commands";

/**
 * The workbench's top bar, rebuilt around three questions a user actually
 * asks — "what am I looking at", "what is the AI doing for me", "what can I
 * do to this run" — instead of one row of every button.
 *
 *   [Live | Replay | MCTS]  [🎓 coach | 💡 hints] [🔎 search | 🧠 e2e net] [⚔️ autobattle]  …context…  [Run ▾] [Analyze ▾] [Tools ▾]  act · floor  ▤
 *
 *  - The VIEW switch and the three PERSISTENT AI CHIPS are always on the
 *    bar: their state matters at a glance, and there are at most four.
 *  - CONTEXT controls are the one or two things the current view is about:
 *    replay's run picker, follow and "Play from here".
 *  - Everything else is a one-shot verb and lives in a menu, grouped by
 *    job. Items that do not apply right now are disabled with the reason,
 *    not removed, so each menu keeps its shape as the run moves.
 *  - Ctrl+K lists all of it.
 *
 * This component owns layout only. What is offered, and why an item is
 * disabled, is decided by `buildCommands` (commands.ts) in App, so the tests
 * there are the tests for the bar's behaviour.
 */

export interface Chip {
  on: boolean;
  toggle: () => void;
  title: string;
}

export interface TopBarProps {
  mode: "play" | "replay" | "mcts";
  setMode: (m: "play" | "replay" | "mcts") => void;
  showViewToggle: boolean;
  replayCount: number;

  commands: Command[];

  /** `undefined` ⇒ not offered in this mode / build. */
  /** `hints` is the advice-style switch: on = Hints, off = Coach.
   *  `advisor`: on = the e2e student advises, off = the search (offered
   *  only on a run the student can read). */
  chips: { hints?: Chip; autobattle?: Chip; advisor?: Chip };

  /** Replay's context controls; only rendered in replay mode. */
  replay?: {
    runs: string[];
    loadRun: (path: string) => void;
    loadedPath: string | null;
    follow: boolean;
    setFollow: (v: boolean) => void;
    followIsLive: boolean;
    /** Follow was switched on by the mod's "Open Workbench" link, which
     *  PROMISES the live run; warn when it is tailing something else. */
    followFromUrl: boolean;
    bridgeError: string | null;
  };

  tools: {
    simlabLink: boolean;
    /** The shown run's game build (GH #877): the synergy catalog opens with
     *  that build's card text. Absent / the default = the default build. */
    gameVersion?: string | null;
    debug: { on: boolean; toggle: () => void };
    onHelp: () => void;
    /** Browser build only: the "Live game…" connection dialog (GH #709). */
    onBridge?: () => void;
    bridgeState?: "off" | "connected" | "unreachable";
    /** Self-contained controls hosted as rows: bug report, bench, offline,
     *  fx, sound. Each renders `null` when it does not apply. */
    controls: ReactNode[];
  };

  /** "Act 2 · floor 21", or null. */
  where: string | null;
  whereTitle?: string;

  palette: { open: boolean; setOpen: (v: boolean) => void };

  /** The side-panel toggle sits OUTSIDE the scroller so it is always on
   *  screen; rendered by the caller and passed through. */
  trailing?: ReactNode;
}

const GROUP_TITLES: Record<string, string> = { run: "Run", analyze: "Analyze" };

function toMenuItem(c: Command): MenuItemSpec {
  return { id: c.id, label: c.label, title: c.title, disabledReason: c.disabledReason,
    danger: c.danger, busy: c.busy, onSelect: c.run };
}

/** The Run menu: everything but the destructive item, a rule, then it. */
function runMenuItems(cmds: Command[]): (MenuItemSpec | null)[] {
  const run = cmds.filter((c) => c.group === "run" && c.id !== "play-from-here");
  const safe = run.filter((c) => !c.danger).map(toMenuItem);
  const danger = run.filter((c) => c.danger).map(toMenuItem);
  return danger.length ? [...safe, null, ...danger] : safe;
}

/** The synergy catalog's URL for game build `v` (`?game_version=` only for a
 *  non-default build, read by `catalog/main.ts`). */
function catalogHref(v: string | null | undefined): string {
  const q = gameVersionBody(v).game_version;
  return q ? `/catalog.html?game_version=${encodeURIComponent(q)}` : "/catalog.html";
}

export default function TopBar(p: TopBarProps) {
  const runItems = useMemo(() => runMenuItems(p.commands), [p.commands]);
  const analyzeItems = useMemo(
    () => p.commands.filter((c) => c.group === "analyze").map(toMenuItem),
    [p.commands],
  );
  const playFromHere = p.commands.find((c) => c.id === "play-from-here");

  const toolItems: (MenuItemSpec | null)[] = [
    ...(p.tools.simlabLink
      ? [{ id: "simlab", label: "🧪 SimLab ↗", href: "/simlab.html",
          title: "SimLab — batch combat statistics (opens in a new tab so your run isn't disturbed)" }]
      : []),
    { id: "synergies", label: "⚡ Synergies ↗", href: catalogHref(p.tools.gameVersion),
      title: "Card Synergy Catalog — pair synergies per character (opens in a new tab so your run isn't disturbed)" },
    ...(p.tools.onBridge
      ? [{ id: "bridge", onSelect: p.tools.onBridge,
          label: p.tools.bridgeState === "connected" ? "🎮 Live game · connected"
            : p.tools.bridgeState === "unreachable" ? "🎮 Live game · unreachable" : "🎮 Live game…",
          title: "connect this page to the running game's advisor (game tree, live search, rewind)" }]
      : []),
    null,
    { id: "debug", label: "🐞 debug log", checked: p.tools.debug.on, onSelect: p.tools.debug.toggle,
      title: "the debug log drawer — records run settings, per-decision MCTS sims, and the play-to-end trace (can be turned on any time)" },
    ...p.tools.controls.filter(Boolean).map((node, i) => ({ id: `control-${i}`, label: "", render: node })),
    null,
    { id: "palette", label: "command palette", title: "every command, filterable", onSelect: () => p.palette.setOpen(true) },
    { id: "help", label: "keyboard shortcuts", title: "what the keys do", onSelect: p.tools.onHelp },
  ];

  // The palette is a view of the same entries the bar renders, plus the
  // chips and the view switch, so nothing is reachable from one and not
  // the other.
  const paletteEntries: PaletteEntry[] = useMemo(() => {
    const out: PaletteEntry[] = [];
    if (p.showViewToggle) {
      out.push({ id: "view-live", group: "View", label: "Live", disabledReason: p.mode === "play" ? "already here" : undefined, run: () => p.setMode("play") });
      out.push({ id: "view-replay", group: "View", label: "Replay",
        disabledReason: p.mode === "replay" ? "already here" : p.replayCount === 0 ? "no run loaded to replay" : undefined,
        run: () => p.setMode("replay") });
    }
    out.push({ id: "view-mcts", group: "View", label: "📊 MCTS Runs", disabledReason: p.mode === "mcts" ? "already here" : undefined, run: () => p.setMode("mcts") });
    for (const [id, chip, label] of [
      ["hints", p.chips.hints, "💡 hints style (off = 🎓 coach)"],
      ["autobattle", p.chips.autobattle, "⚔️ autobattle"],
      ["advisor", p.chips.advisor, "🧠 Instinct advises (off = 🔎 search)"],
    ] as const) {
      if (chip) out.push({ id, group: "AI", label, title: chip.title, checked: chip.on, run: chip.toggle });
    }
    for (const c of p.commands) {
      out.push({ id: c.id, group: GROUP_TITLES[c.group] ?? c.group, label: c.label, title: c.title,
        shortcut: c.id === "recommend" ? "H" : undefined,
        disabledReason: c.busy ? "in progress" : c.disabledReason, run: c.run });
    }
    if (p.tools.simlabLink) out.push({ id: "simlab", group: "Tools", label: "🧪 SimLab ↗", run: () => window.open("/simlab.html", "_blank", "noopener") });
    out.push({ id: "synergies", group: "Tools", label: "⚡ Synergies ↗", run: () => window.open(catalogHref(p.tools.gameVersion), "_blank", "noopener") });
    out.push({ id: "debug", group: "Tools", label: "🐞 debug log", checked: p.tools.debug.on, run: p.tools.debug.toggle });
    out.push({ id: "help", group: "Tools", label: "keyboard shortcuts", shortcut: "?", run: p.tools.onHelp });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.mode, p.showViewToggle, p.replayCount, p.chips.hints, p.chips.autobattle, p.chips.advisor, p.commands, p.tools.simlabLink, p.tools.debug.on, p.tools.gameVersion]);

  const chip = (id: string, c: Chip | undefined, off: string, on: string) => c && (
    <button key={id} onClick={c.toggle} aria-pressed={c.on} title={c.title}
      className={`wb-chip${c.on ? " wb-chip--on" : ""}`}>
      {c.on ? on : off}
    </button>
  );

  return (
    <>
      <div className="wb-topbar-scroll">
        {/* View switch */}
        <span className="wb-segmented" role="group" aria-label="view">
          {p.showViewToggle && (
            <>
              <button onClick={() => p.setMode("play")} disabled={p.mode === "play"} aria-pressed={p.mode === "play"}
                title="the live run / start screen">Live</button>
              <button onClick={() => p.setMode("replay")} disabled={p.mode === "replay" || p.replayCount === 0}
                aria-pressed={p.mode === "replay"}
                title={p.replayCount === 0 ? "no run loaded to replay" : "scrub a loaded run"}>Replay</button>
            </>
          )}
          <button onClick={() => p.setMode("mcts")} disabled={p.mode === "mcts"} aria-pressed={p.mode === "mcts"}
            title="aggregate dashboard + recorded MCTS runs">📊 MCTS Runs</button>
        </span>

        {/* Persistent AI switches */}
        {(p.chips.hints || p.chips.autobattle || p.chips.advisor) && (
          <span className="wb-chips">
            {chip("hints", p.chips.hints, "🎓 coach", "💡 hints")}
            {chip("advisor", p.chips.advisor, "🔎 search", "🧠 Instinct")}
            {chip("autobattle", p.chips.autobattle, "⚔️ autobattle", "⚔️ autobattle ✓")}
          </span>
        )}

        {/* Context controls */}
        {p.mode === "replay" && p.replay && (
          <span className="wb-context">
            <select onChange={(e) => e.target.value && p.replay!.loadRun(e.target.value)} defaultValue=""
              aria-label="load a run">
              <option value="" disabled>load a run…</option>
              {p.replay.runs.map((r) => <option key={r} value={r}>{r.split("/").pop()}</option>)}
            </select>
            {p.replay.loadedPath && (
              <label title={p.replay.followIsLive
                ? "following the file the live advisor says it is writing"
                : "poll the loaded run file and auto-advance to new steps as a live game writes them"}>
                <input type="checkbox" checked={p.replay.follow} onChange={(e) => p.replay!.setFollow(e.target.checked)} />
                {" "}follow{p.replay.follow ? (p.replay.followIsLive ? " · live" : " · not live") : ""}
              </label>
            )}
            {playFromHere && (
              <button onClick={playFromHere.run} disabled={!!playFromHere.disabledReason || playFromHere.busy}
                title={playFromHere.disabledReason ?? playFromHere.title}>
                {playFromHere.busy ? "🎮 rebuilding…" : playFromHere.label}
              </button>
            )}
            {p.replay.loadedPath && p.replay.follow && !p.replay.followIsLive && p.replay.followFromUrl && (
              <span style={{ color: "var(--accent-strong)" }} title={
                "No advisor is answering, so this is the newest file in the logs directory — " +
                "not necessarily this game's run. Turn the AI coach on in the mod and this " +
                "switches to the live run by itself."
              }>
                ⚠ advisor offline — newest file on disk
              </span>
            )}
            {p.replay.bridgeError && <span style={{ color: "var(--hp)" }}>{p.replay.bridgeError}</span>}
          </span>
        )}

        {/* Menus */}
        {p.mode !== "mcts" && (
          <span className="wb-menus">
            {runItems.length > 0 && <Menu label="Run" title="save, load, import, end" items={runItems} />}
            {analyzeItems.length > 0 && <Menu label="Analyze" title="reviews, search tree, sandbox" items={analyzeItems} />}
            <Menu label="Tools" title="links, debug, effects, sound, shortcuts" items={toolItems}
              active={p.tools.debug.on} />
          </span>
        )}
        {p.mode === "mcts" && (
          <span className="wb-menus">
            <Menu label="Tools" title="links, debug, effects, sound, shortcuts" items={toolItems}
              active={p.tools.debug.on} />
          </span>
        )}

        {p.where && (
          <span className="wb-where" title={p.whereTitle}>{p.where}</span>
        )}
      </div>
      {p.trailing}
      <CommandPalette open={p.palette.open} entries={paletteEntries} onClose={() => p.palette.setOpen(false)} />
    </>
  );
}
