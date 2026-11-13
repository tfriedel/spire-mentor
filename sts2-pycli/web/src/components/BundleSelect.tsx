import type { Step, ChooseHandler, Bundle } from "../api";
import CardFrame from "./CardFrame";
import { MarkBadge } from "../advice/AdviceContext";
import { markClass, markFor, useAdviceMarks } from "../advice/useAdviceMarks";

/** Pack / bundle picker: each bundle shows its cards; click to select it. */
export default function BundleSelect({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  const bundles: Bundle[] = step.state.bundles ?? [];
  const { marks } = useAdviceMarks();
  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ margin: 0, fontWeight: 600 }}>Choose a pack</h2>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {bundles.map((b) => (
          <div key={b.index} className={`surface${markClass(markFor(marks, "select_bundle", { bundle_index: b.index }))}`}
            style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <MarkBadge mark={markFor(marks, "select_bundle", { bundle_index: b.index })} />
            <div style={{ fontWeight: 600 }}>{b.name ?? `Pack ${b.index + 1}`}</div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", maxWidth: 360 }}>
              {(b.cards ?? []).map((c, i) => <CardFrame key={i} card={c} width={92} />)}
            </div>
            <button className="gs-bundle-select" disabled={!onChoose} onClick={() => onChoose?.("select_bundle", { bundle_index: b.index })}>
              Select this pack
            </button>
          </div>
        ))}
        {bundles.length === 0 && <span style={{ color: "var(--muted)" }}>no packs</span>}
      </div>
    </div>
  );
}
