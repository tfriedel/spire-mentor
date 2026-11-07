import { Fragment, type CSSProperties } from "react";
import { UNRESOLVED_TOP2_Q_GAP, type Enemy, type Score, type StepAdvice } from "../api";
import { enemyLabeller } from "../enemyLabels";
import { isE2eScores } from "../advisor";

// q ∈ [-1,1] → red (loss) ⟶ amber ⟶ green (win). Matches the search-tree view.
function qColor(q: number): string {
  const t = Math.max(0, Math.min(1, (q + 1) / 2));
  return `hsl(${Math.round(t * 122)} 62% 44%)`;
}

// Rows are cells of ONE grid, not a flex row each. The numeric columns used to
// carry hand-picked fixed widths (44 + 54 + 30 + 72 px plus four 7px gaps =
// 228 of the 288px a 320px side panel has), which left the label ~60px — nine
// characters — so almost every card read as "Sweeping…" while the number
// columns sat half empty. As grid tracks they size to their widest content
// across ALL rows instead, and the leftover goes to the label.
const ROW_GAP_X = 6;
const SCORE_GRID: CSSProperties = {
  display: "grid",
  // label | Q | bar | share % | W/L/T
  gridTemplateColumns: "minmax(0, 1fr) auto 40px auto auto",
  columnGap: ROW_GAP_X,
  rowGap: 7,
  alignItems: "center",
};
const ADVICE_GRID: CSSProperties = {
  display: "grid",
  // label | bar | share %
  gridTemplateColumns: "minmax(0, 1fr) auto auto",
  columnGap: ROW_GAP_X,
  rowGap: 7,
  alignItems: "center",
};

// The label gets whatever the numbers do not need, and wraps onto a second
// line before it ellipsizes — two lines of this column hold ~34 characters, so
// "drink Potion of Binding" and "Go for the Eyes → Cultist" now read in full.
// The tooltip stays for the rare line that still overflows.
const LABEL_CELL: CSSProperties = {
  minWidth: 0,
  display: "-webkit-box",
  WebkitBoxOrient: "vertical",
  WebkitLineClamp: 2,
  overflow: "hidden",
  overflowWrap: "anywhere",
  lineHeight: 1.3,
};

const NUM_CELL: CSSProperties = { textAlign: "right", color: "var(--muted)", whiteSpace: "nowrap" };

export default function Reasoning({ scores, pv, advice, enemies, heading = true }: {
  scores: Score[];
  pv: string[] | null;
  /** Recorded screen-decision advice (live-run logs): policy ranking shown
   *  when there are no MCTS scores for the step. */
  advice?: StepAdvice | null;
  /** Combatants of the position these scores were computed for, so a target is
   *  rendered by name instead of by slot index. */
  enemies?: Enemy[];
  /** Draw the "Machine reasoning" caption. Off when a surrounding dock already
   *  carries it in its (always-visible) header bar — App.tsx's ReasoningDock —
   *  so the panel is not titled twice. */
  heading?: boolean;
}) {

  // Search % = this action's share of all visits across the root's children,
  // mirroring the search tree's "search %" column. Win_prob hovered near 1 for
  // every move and read flat; visit share + Q separates the lines.
  const totalVisits = scores.reduce((sum, s) => sum + s.visits, 0);
  const labelFor = enemyLabeller(enemies);
  // The run-AI does not search a forced move (one legal action): its record
  // is the move alone with zero visits. There is no Q, share or W/L/T to show,
  // and a "+0.00" would read as an even fight.
  const unsearched = scores.length > 0 && totalVisits === 0;
  // The e2e advisor's rows: one per merged class, its softmax probability —
  // no Q, no visits, no rollouts to show.
  const e2e = isE2eScores(scores);

  // Is the top row actually separated from the runner-up? Sort by Q here
  // rather than trusting row order: rows arrive sorted by win_prob, which is
  // 1.0 for every action in a fight the player always wins — so that ordering
  // carries no information in exactly the positions this check exists for.
  const byQ = scores.map((s) => s.q).sort((a, b) => b - a);
  const top2Gap = byQ.length > 1 ? byQ[0] - byQ[1] : Infinity;
  const unresolved = !e2e && scores.length > 1 && top2Gap < UNRESOLVED_TOP2_Q_GAP;

  // The engine's own tie band (GH #1022): the moves it cannot tell from its
  // pick, and the rule's reason for the ★. Read off the rows, never a
  // threshold of the panel's.
  const rowLabel = (s: Score) =>
    `${s.action}${s.target_index != null ? ` → ${labelFor(s.target_index)}` : ""}`;
  const star = scores.find((s) => s.is_best);
  const band = scores
    .filter((s) => s.in_band)
    .sort((a, b) => (a.pick_rank ?? Infinity) - (b.pick_rank ?? Infinity));
  const tieBand = !unsearched && star?.in_band && band.length > 1 ? band : null;
  return (
    <div>
      {heading && (
        <div className="mono" style={{ fontSize: 10, color: "var(--muted)", borderBottom: "1px solid var(--border)",
          paddingBottom: 5, marginBottom: 9 }}>Machine reasoning</div>
      )}
      {scores.length === 0 && !advice?.options?.length &&
        <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>no scores yet — request a recommendation</div>}
      {unresolved && (
        <div className="mono"
          title={`top-2 Q gap ${top2Gap.toFixed(4)}, under ${UNRESOLVED_TOP2_Q_GAP} — inside the `
            + "search's own cross-seed spread, so a different seed can reorder these. "
            + "Raise sims if you need it settled."}
          style={{ fontSize: 10, color: "var(--muted)", border: "1px solid var(--border)",
            borderRadius: 4, padding: "4px 6px", marginBottom: 8 }}>
          ~ too close to call — top two within {UNRESOLVED_TOP2_Q_GAP} Q
        </div>
      )}
      {tieBand && (
        <div className="mono" data-testid="tie-band"
          title={"The search's tie band: these moves score within its tie_eps of the best, "
            + "so it picks among them by rule rather than by value."}
          style={{ fontSize: 10, color: "var(--muted)", marginBottom: 8 }}>
          ≈ about equal: {tieBand.map(rowLabel).join(" ≈ ")}
          {star?.pick_reason ? ` · ★ ${star.pick_reason}` : ""}
        </div>
      )}
      {/* Policy-ranked advice (no search visits): the recorded advisor ranking
          for map / shop / reward / event decisions in live-game logs. */}
      {scores.length === 0 && (advice?.options?.length ?? 0) > 0 && (
        <div className="mono" style={{ ...ADVICE_GRID, fontSize: 11 }}>
          {advice!.options!.map((o, i) => {
            const share = o.share != null ? Math.round(o.share * 100) : null;
            const label = `${o.is_best ? "★ " : ""}${o.label}`;
            return (
              <Fragment key={i}>
                <span title={label} style={{ ...LABEL_CELL, fontWeight: o.is_best ? 700 : 400 }}>
                  {label}
                </span>
                {share != null ? (
                  <span title={`${share}%`}
                    style={{ width: 40, height: 7, background: "var(--hp-track)",
                      borderRadius: 4, overflow: "hidden" }}>
                    <span style={{ display: "block", height: "100%", width: `${share}%`, background: "var(--accent-strong)" }} />
                  </span>
                ) : <span />}
                {share != null ? <span style={NUM_CELL}>{share}%</span> : <span />}
              </Fragment>
            );
          })}
        </div>
      )}
      {advice?.note && <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 7 }}>{advice.note}</div>}
      {e2e && (
        <div className="mono" data-testid="e2e-rows" style={{ ...ADVICE_GRID, fontSize: 11 }}>
          {scores.map((s) => {
            const pct = (s.prob ?? 0) * 100;
            const label = `${s.is_best ? "★ " : ""}${rowLabel(s)}`;
            const tip = `${rowLabel(s)} — Instinct gives this move ${pct.toFixed(1)}% `
              + "(a policy probability, not a win chance)";
            return (
              <Fragment key={`${s.stable_idx}:${s.target_index}`}>
                <span title={tip} style={{ ...LABEL_CELL, fontWeight: s.is_best ? 700 : 400 }}>{label}</span>
                <span title={tip} style={{ width: 40, height: 7, background: "var(--hp-track)",
                  borderRadius: 4, overflow: "hidden" }}>
                  <span style={{ display: "block", height: "100%", width: `${pct}%`, background: "var(--accent-strong)" }} />
                </span>
                <span title={tip} style={NUM_CELL}>{pct.toFixed(1)}%</span>
              </Fragment>
            );
          })}
        </div>
      )}
      {e2e && (
        <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 7 }}>
          Instinct · one forward, no search · policy probabilities, not win chances
        </div>
      )}
      <div className="mono" style={{ ...SCORE_GRID, fontSize: 11 }}>
        {!e2e && scores.map((s) => {
          const share = totalVisits > 0 ? Math.round((s.visits / totalVisits) * 100) : 0;
          const label = rowLabel(s);
          if (unsearched) {
            return (
              <Fragment key={`${s.stable_idx}:${s.target_index}`}>
                <span title={`${label} — the only legal move, not searched`} style={LABEL_CELL}>{label}</span>
                <span style={{ ...NUM_CELL, gridColumn: "span 4" }}>forced · not searched</span>
              </Fragment>
            );
          }
          // Two lines hold most moves, but a long one ("drink Potion of
          // Binding → Gremlin Nob #2") still clamps, so the label carries the
          // whole line as a tooltip. The numeric cells keep their own, more
          // specific tooltips, which win when you hover them directly.
          const full = `${label} — Q ${s.q >= 0 ? "+" : ""}${s.q.toFixed(2)} · ${share}% of search visits`
            + ` · ${s.visits} visits · ${s.wins ?? 0}W/${s.losses ?? 0}L/${s.ties ?? 0}T`
            + (s.is_best ? " · best" : "");
          return (
            <Fragment key={`${s.stable_idx}:${s.target_index}`}>
              <span title={full} style={{ ...LABEL_CELL, fontWeight: s.is_best && !unresolved ? 700 : 400 }}>
                {label}
              </span>
              <span title="expected value Q: +1 = sure win, 0 = even, −1 = sure loss"
                style={{ ...NUM_CELL, color: qColor(s.q) }}>
                {s.q >= 0 ? "+" : ""}{s.q.toFixed(2)}
              </span>
              <span title={`${share}% of search visits`}
                style={{ width: 40, height: 7, background: "var(--hp-track)",
                  borderRadius: 4, overflow: "hidden" }}>
                <span style={{ display: "block", height: "100%", width: `${share}%`, background: qColor(s.q) }} />
              </span>
              <span title={`${share}% of search visits`} style={NUM_CELL}>{share}%</span>
              <span title="rollout wins / losses / ties" style={{ ...NUM_CELL, fontSize: 10 }}>
                {s.wins ?? 0}/{s.losses ?? 0}/{s.ties ?? 0}
              </span>
            </Fragment>
          );
        })}
      </div>
      {pv && <div className="mono" title={pv.join(" → ")}
        style={{ fontSize: 10, color: "var(--muted)", marginTop: 9 }}>pv: {pv.join(" → ")}</div>}
    </div>
  );
}
