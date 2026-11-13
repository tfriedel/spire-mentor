import type { KeyboardEvent, MouseEvent } from "react";

/** Props that make a power chip, an intent or the block badge an inspect
 *  trigger on a phone: a tap opens the sheet, and for keyboards and screen
 *  readers it is a named button that Enter / Space open. Off a phone (`enabled`
 *  false) nothing at all, so the desktop DOM is exactly what it was.
 *  stopPropagation keeps the stage from closing the sheet again. */
export function inspectTrigger(enabled: boolean, label: string, open: () => void) {
  if (!enabled) return {};
  return {
    role: "button",
    tabIndex: 0,
    "aria-label": label,
    onClick: (e: MouseEvent) => { e.stopPropagation(); open(); },
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      e.stopPropagation();
      open();
    },
  };
}
