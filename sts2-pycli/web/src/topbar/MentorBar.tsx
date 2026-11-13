// The AI's controls on the top bar, laid out as the mod's Spire Mentor badge
// popover is (spire-mentor-mod/SimToolbar.cs, and the frame's toolbar that is
// that popover laid flat, src/frame/panels.tsx):
//
//   [■ Stop AI]  Mode [Off · Coach · Hints · AI plays]  [Search · Instinct]
//   │ Peek · Let AI pick · Autobattle │  ● status in words
//
// One Mode switch instead of a hints/coach chip whose label read as either
// state or action, plus "AI plays" for the autoplay that used to live only
// behind F. The Brain switch shows only where Instinct can read the run. The
// verbs act on THIS decision. While the AI plays, the bar turns amber and
// leads with a Stop button nobody can miss.

import type { Brain, MentorMode, MentorState } from "./mentorStatus";

export type { Brain, MentorMode, MentorState };

export interface MentorBarProps {
  mode: MentorMode;
  setMode: (m: MentorMode) => void;
  /** Present ⇒ "AI plays" cannot start here; the text says why. */
  aiPlaysDisabled?: string;
  brain?: { value: Brain; set: (b: Brain) => void };
  /** Coach style's peek at the AI's next pick (P). */
  onPeek?: () => void;
  /** The AI takes just this decision (G). */
  letAiPick?: { run: () => void; disabledReason?: string };
  autobattle?: { on: boolean; toggle: () => void };
  status: { text: string; state: MentorState; title?: string };
  aiPlaying: boolean;
  onStop: () => void;
  /** A phone held upright splits the bar: `mode` is the bar itself (the
   *  Mode switch at tap size, or Stop alone while the AI plays), `rest`
   *  the Menu panel's share (Brain, this decision's verbs, the status). */
  parts?: "all" | "mode" | "rest";
}

export const MODES: { key: MentorMode; label: string; title: string }[] = [
  { key: "off", label: "Off", title: "The AI stays quiet: no hints, no grades. H, P and the reviews still answer when asked." },
  {
    key: "coach",
    label: "Coach",
    title: "Nothing before you act; Peek (P) shows the AI's pick, and the coach says afterwards where you differed",
  },
  { key: "hints", label: "Hints", title: "The AI's picks are ringed and numbered while you decide" },
  { key: "aiplays", label: "AI plays", title: "The AI plays the run until you stop it (F starts it, Esc stops it)" },
];

const BRAINS: { key: Brain; label: string; title: string }[] = [
  { key: "search", label: "Search", title: "Thinks ahead: a search over the fight, the run net for the screens" },
  {
    key: "instinct",
    label: "Instinct",
    title: "The end-to-end net, no search: fast, every decision. Its percentages are a policy's, not win chances.",
  },
];

export default function MentorBar(p: MentorBarProps) {
  const parts = p.parts ?? "all";
  const rest = parts !== "mode";
  return (
    <span className={`wb-mentor wb-mentor--${p.mode}${p.aiPlaying ? " wb-mentor--ai" : ""}${parts === "mode" ? " wb-mentor--phonebar" : ""}`}>
      {p.aiPlaying && parts !== "rest" && (
        <button type="button" className="wb-mentor-stop" onClick={p.onStop} title="Stop the AI (Esc)">
          ■ Stop AI <kbd>Esc</kbd>
        </button>
      )}
      {(parts === "all" || (parts === "mode" && !p.aiPlaying)) && (
      <span className="wb-seg" role="radiogroup" aria-label="Mode">
        {MODES.map((m) => {
          const disabled = m.key === "aiplays" && !!p.aiPlaysDisabled && p.mode !== "aiplays";
          return (
            <button key={m.key} type="button" role="radio" aria-checked={p.mode === m.key}
              className={`wb-seg-btn wb-seg-${m.key}${p.mode === m.key ? " wb-seg-on" : ""}`}
              title={disabled ? p.aiPlaysDisabled : m.title} disabled={disabled}
              onClick={() => p.mode !== m.key && p.setMode(m.key)}>
              {m.label}
            </button>
          );
        })}
      </span>
      )}
      {rest && p.brain && (
        <span className="wb-seg" role="radiogroup" aria-label="Brain">
          {BRAINS.map((b) => (
            <button key={b.key} type="button" role="radio" aria-checked={p.brain!.value === b.key}
              className={`wb-seg-btn${p.brain!.value === b.key ? " wb-seg-on" : ""}`} title={b.title}
              onClick={() => p.brain!.value !== b.key && p.brain!.set(b.key)}>
              {b.label}
            </button>
          ))}
        </span>
      )}
      {rest && (p.onPeek || p.letAiPick || p.autobattle) && (
        <span className="wb-mentor-verbs" role="group" aria-label="This decision">
          {p.onPeek && (
            <button type="button" onClick={p.onPeek} title="Peek at the AI's pick for this decision (P)">
              Peek <kbd>P</kbd>
            </button>
          )}
          {p.letAiPick && (
            <button type="button" onClick={p.letAiPick.run} disabled={!!p.letAiPick.disabledReason}
              title={p.letAiPick.disabledReason ?? "The AI takes just this decision (G)"}>
              Let AI pick <kbd>G</kbd>
            </button>
          )}
          {p.autobattle && (
            <button type="button" onClick={p.autobattle.toggle} aria-pressed={p.autobattle.on}
              className={p.autobattle.on ? "wb-mentor-on" : undefined}
              title="The AI plays the fights, you make the run-level decisions — with a change-diff at each stop and the AI's pick revealed after you commit">
              {p.autobattle.on ? "Stop autobattle" : "Autobattle"}
            </button>
          )}
        </span>
      )}
      {rest && (
      <span className="wb-mentor-status" title={p.status.title ?? p.status.text} data-testid="mentor-status">
        <span className={`wb-mentor-dot wb-mentor-dot--${p.status.state}`} aria-hidden="true" />
        {p.status.text}
      </span>
      )}
    </span>
  );
}
