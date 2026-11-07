import { useState } from "react";
import { api, type ChoiceCheck, type ChoiceGrade } from "../api";
import { ordinal } from "../coach/coachText";
import { CHECK_CONFIG, checkSummary } from "./choiceCheck";

type CheckState =
  | { kind: "idle" }
  | { kind: "running"; done: number; total: number }
  | { kind: "done"; check: ChoiceCheck }
  | { kind: "failed"; error: string };

/** Coach chip for the run-level choice just made, graded by agreement with
 *  the AI's policy — never as a win-probability cost. "check it" runs the
 *  slow rollout comparison on demand. */
export default function ChoiceChip({ grade, onDismiss }: {
  grade: ChoiceGrade; onDismiss: () => void;
}) {
  const [check, setCheck] = useState<CheckState>({ kind: "idle" });
  const { verdict, played, top, kind } = grade;
  // Graded against the e2e advisor instead of the PPO run policy.
  const ai = grade.policy === "e2e" ? "Instinct" : "AI";
  const flag = verdict === "flag";
  const tone = flag ? "var(--warn, #c79a3a)" : "var(--accent-strong)";

  async function runCheck() {
    setCheck({ kind: "running", done: 0, total: 0 });
    try {
      const r = await api.live.lastChoiceCheckStream(grade.node_id, CHECK_CONFIG, (ev) => {
        if (ev.type === "start") setCheck({ kind: "running", done: 0, total: ev.total });
        else setCheck({ kind: "running", done: ev.done, total: ev.total });
      });
      setCheck(r.error ? { kind: "failed", error: r.error } : { kind: "done", check: r });
    } catch (e) {
      setCheck({ kind: "failed", error: String(e) });
    }
  }

  const summary = check.kind === "done" ? checkSummary(check.check) : null;
  return (
    <div className="mono" style={{ marginBottom: 8, padding: "6px 10px", borderRadius: 4,
      color: tone, border: `1px solid ${tone}`, background: "var(--field-bg)",
      fontSize: 11, display: "flex", gap: 8, alignItems: "baseline" }}>
      <div style={{ flex: 1 }}>
        {verdict === "agree" && <>✓ {ai} agrees — <b>{played.label}</b></>}
        {verdict === "tossup" && (
          <>
            <div>✓ toss-up — <b>{played.label}</b> · no strong {ai} preference</div>
            <div>{kind} · {ai}'s top picks: {grade.alternatives.map((a) => a.label).join(", ") || top.label}</div>
          </>
        )}
        {flag && (
          <>
            <div>{kind} · you: <b>{played.label}</b></div>
            <div>
              {ai} would take: <b>{top.label}</b> ({grade.strength}) · {played.label} was its {ordinal(played.rank)} choice
            </div>
          </>
        )}
        {grade.checkable && check.kind === "idle" && (
          <button type="button" onClick={runCheck}
            title="Play the rest of the act out from this position both ways — moves wait until it finishes"
            style={{ font: "inherit", marginTop: 4, background: "none", color: "inherit",
              border: "1px solid currentColor", borderRadius: 3, cursor: "pointer", padding: "0 6px" }}>
            check it
          </button>
        )}
        {check.kind === "running" && (
          <div style={{ color: "var(--muted)" }}>checking… {check.done}/{check.total}</div>
        )}
        {summary && <div style={{ color: "var(--text)" }}>{summary}</div>}
        {check.kind === "failed" && <div style={{ color: "var(--muted)" }}>check failed: {check.error}</div>}
      </div>
      <button type="button" onClick={onDismiss} aria-label="Dismiss"
        style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)",
          cursor: "pointer" }}>✕</button>
    </div>
  );
}
