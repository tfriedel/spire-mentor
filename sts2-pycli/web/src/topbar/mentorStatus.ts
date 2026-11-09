// The Mode bar's vocabulary and its status in words (MentorBar.tsx draws it).

export type MentorMode = "off" | "coach" | "hints" | "aiplays";
export type Brain = "search" | "instinct";

/** The dot beside the status words. */
export type MentorState = "idle" | "thinking" | "ai" | "off";

/**
 * The words for the status: what the AI is doing for you right now, in a
 * player's words (the mod's MentorBadge). Pure, for the tests.
 */
export function mentorStatus(s: {
  mode: MentorMode;
  aiPlaying: boolean;
  thinking: boolean;
  brain: Brain | null;
  autobattle: boolean;
  /** The step a whole-run autoplay is on, for "AI playing · step 12". */
  aiStep?: number;
}): { text: string; state: MentorState } {
  const who = s.brain === "instinct" ? "Instinct" : "Search";
  if (s.aiPlaying) {
    return { text: `AI playing${s.aiStep ? ` · step ${s.aiStep}` : ""}`, state: "ai" };
  }
  if (s.mode === "off") return { text: "AI off", state: "off" };
  if (s.thinking) return { text: `${who} thinking…`, state: "thinking" };
  if (s.autobattle) return { text: "Autobattle: the AI fights, you decide the run", state: "idle" };
  return s.mode === "coach"
    ? { text: "Coach: your call — P to peek", state: "idle" }
    : { text: `${who} ready`, state: "idle" };
}
