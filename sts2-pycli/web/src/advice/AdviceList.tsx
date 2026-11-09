// The ranked list in words (rows.ts builds the rows): the same row as the
// mod's in-game panel and the frame's — kind-coloured chip with the plan step,
// the move wrapped (never cut), the verdict, the AI's share. Engine columns
// only with "engine numbers" on; a column no row fills is not laid out.

import { useState } from "react";
import type { AdviceRow, RowKind } from "./rows";

const KIND_COLOR: Record<RowKind, string> = {
  attack: "oklch(64% 0.17 28)",
  skill: "oklch(64% 0.14 255)",
  power: "oklch(78% 0.14 80)",
  potion: "oklch(70% 0.14 155)",
  end_turn: "oklch(62% 0.02 265)",
  skip: "oklch(62% 0.02 265)",
  pick: "oklch(72% 0.09 210)",
  other: "oklch(66% 0.02 265)",
};

const VERDICT_COLOR: Record<string, string> = {
  Best: "var(--gold)",
  "Also good": "oklch(80% 0.14 150)",
  Then: "oklch(90% 0.07 85)",
  Okay: "oklch(85% 0.05 100)",
  Weak: "oklch(78% 0.14 60)",
  Mistake: "oklch(68% 0.18 28)",
};

// -1 red .. 0 amber .. +1 green.
function qColor(q: number): string {
  const t = Math.max(-1, Math.min(1, q));
  return `oklch(75% 0.15 ${25 + (t + 1) * 60})`;
}

const signed = (v: number, d: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

/** Rows shown before "… N more". */
export const VISIBLE_ROWS = 8;

export default function AdviceList({ rows, showNumbers, plan }: {
  rows: AdviceRow[];
  showNumbers: boolean;
  /** The turn's plan in words, numbered (the mod's Plan line); omitted when
   *  there is no more than the pick. */
  plan?: string[];
}) {
  const [expanded, setExpanded] = useState(false);
  const visible = rows.filter((r) => !r.hidden);
  const shown = expanded ? rows : visible.slice(0, VISIBLE_ROWS);
  const more = rows.length - shown.length;
  const bestQ = rows.find((r) => r.best)?.q ?? null;
  const share = shown.some((r) => r.share != null);
  const q = showNumbers && shown.some((r) => r.q != null);
  const wlt = showNumbers && shown.some((r) => r.wlt != null);
  const template = ["20px", "minmax(0, 1fr)", "auto", ...(q ? ["52px"] : []), ...(share ? ["44px", "34px"] : []),
    ...(wlt ? ["auto"] : [])].join(" ");
  return (
    <div className="wb-advice" data-testid="advice-list">
      {plan && plan.length > 1 && (
        <div className="wb-advice-plan">
          {plan.map((s, i) => (
            <span key={`${i}-${s}`} className="wb-advice-plan-step">
              <span className="wb-advice-plan-n">{i + 1}</span>{s}
            </span>
          ))}
        </div>
      )}
      <div className="wb-advice-rows">
        {shown.map((r) => {
          const kind = KIND_COLOR[r.kind];
          const delta = !r.best && bestQ != null && r.q != null;
          const qShown = r.q == null ? null : delta ? r.q - bestQ! : r.q;
          const pct = r.share == null ? null : Math.max(0, Math.min(1, r.share));
          return (
            <div key={r.id} className={`wb-advice-row${r.best ? " wb-advice-row--best" : ""}${r.hidden ? " wb-advice-row--hidden" : ""}`}
              style={{ gridTemplateColumns: template }} title={r.label}>
              <span className="wb-advice-chip" style={{ borderColor: kind, background: r.step != null ? kind : "transparent" }}>
                {r.step ?? ""}
              </span>
              <span className="wb-advice-label">{r.label}</span>
              <span className="wb-advice-verdict" style={{ color: VERDICT_COLOR[r.verdict] ?? "var(--muted)" }}>
                {r.verdict}
              </span>
              {q && (
                <span className="wb-advice-num" title="the engine's value: +1 a sure win, −1 a sure loss (the pick's; the others as a loss against it)"
                  style={{ color: qShown == null ? undefined : delta ? (qShown < -0.0005 ? "oklch(70% 0.15 30)" : "var(--muted)") : qColor(r.q!) }}>
                  {qShown == null ? "" : delta ? signed(qShown, 3) : signed(qShown, 2)}
                </span>
              )}
              {share && (
                <>
                  <span className="wb-advice-bar">
                    {pct != null && <span style={{ width: `${pct * 100}%`, background: r.best ? "var(--gold)" : kind }} />}
                  </span>
                  <span className="wb-advice-num">{pct == null ? "" : pct < 0.005 ? "<1%" : `${Math.round(pct * 100)}%`}</span>
                </>
              )}
              {wlt && <span className="wb-advice-wlt" title="rollout wins / losses / ties">{r.wlt ?? ""}</span>}
            </div>
          );
        })}
      </div>
      {(more > 0 || expanded) && (
        <button type="button" className="wb-advice-more" onClick={() => setExpanded(!expanded)}>
          {expanded ? "▴ show fewer" : `… ${more} more ▾`}
        </button>
      )}
    </div>
  );
}
