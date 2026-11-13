import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { usePhone } from "../phone";
import { Ctx, type InspectItem, type Internal } from "./useInspect";

/** The inspect state. Outside a provider `useInspect()` is a disabled no-op, so
 *  components used elsewhere (the HUD on a board-less screen, tests) keep
 *  working unchanged. The provider sits above BOTH the HUD band (relics) and
 *  the Board (which renders the sheet), so they share one open item. */
export function InspectProvider({ disabled = false, children }: { disabled?: boolean; children: ReactNode }) {
  const phone = usePhone();
  const [item, setItem] = useState<InspectItem | null>(null);
  const [hosts, setHosts] = useState(0);
  const [suppress, setSuppress] = useState(0);
  const chromeEnabled = phone != null && !disabled && hosts > 0;
  const enabled = chromeEnabled && suppress === 0;

  const close = useCallback(() => setItem(null), []);
  const open = useCallback((next: InspectItem) => {
    setItem((cur) => (cur?.key === next.key ? null : next));
  }, []);
  const registerHost = useCallback(() => {
    setHosts((n) => n + 1);
    return () => setHosts((n) => n - 1);
  }, []);
  const registerSuppress = useCallback(() => {
    setSuppress((n) => n + 1);
    return () => setSuppress((n) => n - 1);
  }, []);
  // Leaving phone mode, or being disabled, must not strand an open sheet; a
  // card being armed closes what was opened on the stage, not a relic.
  useEffect(() => {
    setItem((cur) => (cur && !(enabled || (chromeEnabled && cur.chrome)) ? null : cur));
  }, [enabled, chromeEnabled]);

  const value = useMemo<Internal>(() => ({
    item: item && (enabled || (chromeEnabled && item.chrome)) ? item : null,
    open: chromeEnabled ? open : () => {},
    close, enabled, chromeEnabled, registerHost, registerSuppress,
  }), [enabled, chromeEnabled, item, open, close, registerHost, registerSuppress]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
