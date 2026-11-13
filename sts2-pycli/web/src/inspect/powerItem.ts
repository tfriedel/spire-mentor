import type { Power } from "../api";
import type { InspectItem } from "./useInspect";

/** The inspect-sheet entry for one power. `scope` names the creature that
 *  carries it, so the same power on two creatures is two entries (tapping the
 *  second switches the sheet instead of toggling the first closed). */
export function powerItem(p: Power, scope = ""): InspectItem {
  return { key: `power:${scope}:${p.name}`, icon: p.image, name: p.name,
    amount: p.display_amount ?? p.amount, description: p.description };
}
