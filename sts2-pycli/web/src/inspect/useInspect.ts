// The inspect context, types and hooks live apart from the provider so that
// file exports only the component (react-refresh/only-export-components).
import { createContext, useContext, useEffect } from "react";

/** One thing to explain on a phone, where there is no hover: a power, an
 *  intent, the block badge, a relic. `list` is for a group (the powers behind a
 *  "+N" chip, an intent's modifiers): one row per entry. */
export type InspectItem = {
  key: string;
  icon?: string | null;
  name: string;
  amount?: string | number;
  description?: string;
  list?: InspectItem[];
  /** Opened from the chrome OUTSIDE the stage (a HUD relic): survives a card
   *  being armed, which only takes over taps on the creatures. */
  chrome?: boolean;
};

/** What the sheet says for a relic the state carries no text for. */
export const NO_DESCRIPTION = "No description available";

export type InspectApi = {
  item: InspectItem | null;
  /** Opens `item`; opening the key that is already open closes it. */
  open(item: InspectItem): void;
  close(): void;
  /** True only on a phone, when not disabled, and when a sheet is mounted to
   *  show it. Callers fall back to their desktop behaviour when it is false. */
  enabled: boolean;
};

export type Internal = InspectApi & {
  /** `enabled` ignoring the armed-card suppression: for the chrome outside the
   *  stage (HUD relics), whose taps never mean "target this". */
  chromeEnabled: boolean;
  /** A mounted `InspectSheet` registers here; without one there is nowhere to
   *  show an item, so `enabled` stays false (a relic keeps its popover on
   *  screens that have no board). */
  registerHost(): () => void;
  /** Board suppresses inspection while a card is armed for a target. */
  registerSuppress(): () => void;
};

const noop = () => () => {};
const OFF: Internal = {
  item: null, open() {}, close() {}, enabled: false, chromeEnabled: false,
  registerHost: noop, registerSuppress: noop,
};
export const Ctx = createContext<Internal>(OFF);

export function useInspect(): InspectApi {
  const { item, open, close, enabled } = useContext(Ctx);
  return { item, open: enabled ? open : () => {}, close, enabled };
}

/** `useInspect` for the chrome outside the stage (the HUD's relics): still
 *  enabled while a card is armed (an armed card only owns taps on creatures),
 *  and what it opens stays open until closed. Before this a relic tap with a
 *  card armed did nothing at all. */
export function useChromeInspect(): InspectApi {
  const { item, open, close, chromeEnabled } = useContext(Ctx);
  return {
    item, close, enabled: chromeEnabled,
    open: chromeEnabled ? (it: InspectItem) => open({ ...it, chrome: true }) : () => {},
  };
}

/** For the component that renders the sheet: counts it as a host while mounted. */
export function useInspectHost(): void {
  const { registerHost } = useContext(Ctx);
  useEffect(() => registerHost(), [registerHost]);
}

/** Switches inspection off while `on` (Board: a card is armed, so a tap on a
 *  creature means "target this", not "explain this"). */
export function useInspectSuppress(on: boolean): void {
  const { registerSuppress } = useContext(Ctx);
  useEffect(() => (on ? registerSuppress() : undefined), [on, registerSuppress]);
}
