import { useEffect, useMemo, useState, type ReactNode } from "react";
import { HintsCtx, type AdviceStyle, type HintsValue } from "./useHints";

// Supplies the advice style (Off / Coach / Hints) — what it means and the hook
// that reads it live in useHints.ts.

const STORAGE_KEY = "adviceStyle";
/** The old boolean switch: `"1"` (hints on) migrates to Hints, `"0"` to Coach. */
const LEGACY_KEY = "hintsOn";
/** A fresh browser starts in Hints, the mod's default since 0.10 (the AI's
 *  picks ringed and numbered while you decide). The style switched last is
 *  kept from then on. */
const FRESH_STYLE: AdviceStyle = "hints";

function readStored(): AdviceStyle {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === "hints" || s === "coach" || s === "off") return s;
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy === "1") return "hints";
    if (legacy === "0") return "coach";
    return FRESH_STYLE;
  } catch {
    return FRESH_STYLE; // private mode / storage disabled
  }
}

function valueOf(style: AdviceStyle, setStyle: (s: AdviceStyle) => void): HintsValue {
  return {
    style,
    setStyle,
    on: style === "hints",
    off: style === "off",
    setOn: (v) => setStyle(v ? "hints" : "coach"),
    toggle: () => setStyle(style === "hints" ? "coach" : "hints"),
  };
}

export function HintsProvider({ children }: { children: ReactNode }) {
  const [style, setStyle] = useState<AdviceStyle>(readStored);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, style);
    } catch {
      /* storage unavailable — the style still works for this session */
    }
  }, [style]);
  const value = useMemo(() => valueOf(style, setStyle), [style]);
  return <HintsCtx.Provider value={value}>{children}</HintsCtx.Provider>;
}

/** Test/story helper: force a style (or `on` = Hints) without touching storage. */
export function HintsFixture({
  on,
  style,
  children,
}: {
  on?: boolean;
  style?: AdviceStyle;
  children: ReactNode;
}) {
  const s: AdviceStyle = style ?? (on ? "hints" : "coach");
  const value = useMemo(() => valueOf(s, () => {}), [s]);
  return <HintsCtx.Provider value={value}>{children}</HintsCtx.Provider>;
}
