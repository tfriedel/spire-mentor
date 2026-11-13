import { useEffect, useState } from "react";

/** Live `window.matchMedia` as React state.
 *
 *  The workbench grew up as a desktop tool, so the few places that must behave
 *  differently on a tablet — the right-hand panel becomes a drawer, a held card
 *  needs a tap to be readable because there is no hover — read the same media
 *  queries the stylesheet does instead of sniffing user agents or measuring
 *  `window.innerWidth` by hand.
 *
 *  Returns `false` where `matchMedia` is missing (jsdom, which is every unit
 *  test): "not a tablet" is the right default there, and it keeps the tests on
 *  the desktop path they were written for.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(query).matches
      : false,
  );
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

/** Viewport too narrow to spend 320px on the side panel — a tablet in
 *  landscape (1024–1194 CSS px) or a half-width desktop window. Kept in sync
 *  with the `--wb-narrow` breakpoint in responsive.css BY HAND: CSS has no way
 *  to hand a media query back to JS. */
export const NARROW_QUERY = "(max-width: 1200px)";

/** No hover: a touchscreen. Anything the UI only reveals on `:hover` is
 *  invisible on such a device, so those affordances need a tap equivalent. */
export const TOUCH_QUERY = "(hover: none)";

/** Short viewport: a tablet in landscape (640–800 CSS px tall) or a laptop
 *  window that is wide but not tall. Kept in sync with the `max-height: 800px`
 *  block in responsive.css BY HAND, for the same reason as NARROW_QUERY. */
export const SHORT_QUERY = "(max-height: 800px)";

/** A phone in landscape (360–412 CSS px tall). The workbench's tool rows are
 *  worth more than the game here, so the top row folds away behind a button
 *  and the HUD and tool rows shrink to a sliver. Kept in sync with the
 *  `max-height: 520px` block in responsive.css BY HAND, like the others. */
export const PHONE_LANDSCAPE_QUERY = "(max-height: 520px)";

/** A phone held upright: touch and a narrow viewport (360–412 CSS px wide).
 *  Together with PHONE_LANDSCAPE_TOUCH_QUERY these define "phone mode"; a touch
 *  tablet (1024x640 and up) matches neither. Kept in sync with the phone media
 *  queries in phone.css BY HAND, like the others. */
export const PHONE_PORTRAIT_QUERY = "(hover: none) and (max-width: 480px)";

/** A phone on its side: touch and a short viewport. The `(hover: none)` guard
 *  is what separates it from PHONE_LANDSCAPE_QUERY above, which also matches a
 *  short desktop window. Kept in sync with phone.css BY HAND. */
export const PHONE_LANDSCAPE_TOUCH_QUERY = "(hover: none) and (max-height: 520px)";
