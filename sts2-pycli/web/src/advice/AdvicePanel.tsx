// The side panel's advice for the decision on screen — every decision, as the
// mod's one advice panel shows it: the fight's moves (the search's or
// Instinct's) and every other screen's options (the run policy's), ranked in
// words. With "engine numbers" on, a fight shows the full machine reasoning
// instead (Q, visit shares, W/L/T, the tie band, the principal variation),
// which is the workbench's analysis view and stays exactly as it was.

import type { Card, Enemy, PpoPolicyPayload, Score, StepAdvice } from "../api";
import Reasoning from "../components/Reasoning";
import { enemyLabeller } from "../enemyLabels";
import { isE2eScores } from "../advisor";
import AdviceList from "./AdviceList";
import { humanize, type Mark } from "./marks";
import { policyRows, scoreRows, type AdviceRow } from "./rows";

/** A recorded run's policy ranking (live-game logs): label and share only. */
function recordedRows(advice: StepAdvice): AdviceRow[] {
  return (advice.options ?? []).map((o, i) => ({
    id: `rec:${i}`, label: humanize(o.label), verdict: o.is_best ? "Best" : "", step: o.is_best ? 1 : null,
    kind: "pick", best: !!o.is_best, hidden: false, share: o.share ?? null, q: null, wlt: null,
  }));
}

export function CombatAdvice({ scores, pv, enemies, hand, advice, marks, showNumbers }: {
  scores: Score[];
  pv: string[] | null;
  enemies?: Enemy[];
  hand?: Card[];
  advice?: StepAdvice | null;
  marks: Map<string, Mark>;
  showNumbers: boolean;
}) {
  if (showNumbers) return <Reasoning scores={scores} pv={pv} advice={advice} heading={false} enemies={enemies} />;
  const labelFor = enemyLabeller(enemies);
  const rows = scores.length > 0
    ? scoreRows(scores, marks, {
      hand,
      labelOf: (s) => `${s.action}${s.target_index != null ? ` → ${labelFor(s.target_index)}` : ""}`,
    })
    : advice ? recordedRows(advice) : [];
  if (rows.length === 0) {
    return <div className="wb-advice-note">No advice for this position yet — press H to ask.</div>;
  }
  const forced = scores.length === 1 && scores[0].visits === 0 && scores[0].prob == null;
  return (
    <>
      <AdviceList rows={rows} showNumbers={false} />
      {forced && <div className="wb-advice-note">The only legal move.</div>}
      {isE2eScores(scores) && (
        <div className="wb-advice-note">Instinct · no search · its percentages are a policy's, not win chances</div>
      )}
      {advice?.note && <div className="wb-advice-note">{advice.note}</div>}
    </>
  );
}

export function PolicyAdvice({ payload, marks, showNumbers, implToggle }: {
  payload: PpoPolicyPayload;
  marks: Map<string, Mark>;
  showNumbers: boolean;
  /** The server build's PPO backend switch (rust / python); absent in the browser build. */
  implToggle?: { impl: string; toggle: () => void };
}) {
  const who = payload.policy === "e2e" ? "Instinct" : "The run net";
  if (!payload.ppo_available) {
    return (
      <div className="wb-advice-note">
        {who} {payload.policy === "e2e" ? "has no pick here" : "is unavailable"}{payload.error ? `: ${payload.error}` : ""}
      </div>
    );
  }
  return (
    <>
      <AdviceList rows={policyRows(payload.options, marks)} showNumbers={showNumbers} />
      <div className="wb-advice-note">
        {payload.policy === "e2e" ? "Instinct" : "PPO run policy"}
        {payload.forced ? " · the only option" : ""}
        {showNumbers && payload.value != null ? ` · V(s) = ${payload.value.toFixed(3)}` : ""}
        {showNumbers && implToggle && (
          <>
            {" · "}
            <button type="button" className="wb-advice-more" onClick={implToggle.toggle} title="PPO inference backend">
              {implToggle.impl}
            </button>
          </>
        )}
      </div>
    </>
  );
}
