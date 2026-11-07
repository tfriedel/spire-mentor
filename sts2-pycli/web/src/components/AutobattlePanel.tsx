import type { AutobattleBand, AutobattleDiff, AutobattleVerdict } from "../api";

/**
 * Autobattle mode dock — the run-level drill.
 *
 * An autobattle replays one run with the combats auto-played by the MCTS agent and
 * stops at every RUN-level decision. This panel carries the two things that
 * make that format work:
 *
 * 1. The **change diff** ("since your last stop: +Inflame, −2 Strikes, 52/75
 *    HP"). Drilling run decisions as cold flashcards fails because reading a
 *    stranger's deck costs far more than deciding does. In real play context
 *    accrues one card at a time, so you only ever read what moved — this card
 *    reproduces that, spanning the auto-played combats in between.
 * 2. The **verdict**, revealed only AFTER you commit. Showing the champion's
 *    pick alongside the options would destroy the training signal, so the
 *    recommendation is never in the advance payload; the panel fetches it
 *    once the action has landed.
 *
 * The auto-played fights are watchable (the "watch fights" toggle): the leg is
 * then walked one painted decision at a time instead of resolving invisibly, so
 * you see the combats your HP line was about. Skipping is still there for a long
 * fight you don't want to sit through, and the change-diff spans the leg either
 * way.
 *
 * Honesty rule (shared with WpaReview): the verdict is agreement with the
 * champion and nothing more. The critic's value is a shaped return, not a win
 * probability, so no percentage is shown or implied. Real win-rate grading
 * needs counterfactual finishes, which autobattle mode does not run.
 *
 * Purely presentational: the caller owns the advance/verdict calls. The diff's
 * plain-text `lines` are the server-side rendering (logs, bug reports, tests);
 * this component renders the structured fields so it can colour them.
 */

function Chips({ label, items, tone }: { label: string; items: string[]; tone: string }) {
  if (!items.length) return null;
  return (
    <div className="autobattle-diff-row">
      <span className="autobattle-diff-label">{label}</span>
      {items.map((name, i) => (
        <span key={`${name}-${i}`} className="autobattle-chip" style={{ borderColor: tone, color: tone }}>
          {name}
        </span>
      ))}
    </div>
  );
}

function delta(value: number | null): string {
  if (!value) return "";
  return value < 0 ? ` (−${Math.abs(value)})` : ` (+${value})`;
}

export function AutobattleDiffCard({ diff }: { diff: AutobattleDiff }) {
  const first = diff.hp_delta == null && diff.from_floor == null;
  const nothingMoved =
    !first &&
    !diff.deck.added.length &&
    !diff.deck.removed.length &&
    !diff.deck.upgraded.length &&
    !diff.relics_gained.length &&
    !diff.potions_gained.length &&
    !diff.hp_delta &&
    !diff.gold_delta;
  return (
    <div className="autobattle-diff">
      <div className="autobattle-diff-head">
        {first ? "Run start" : `Since your last stop${diff.to_floor != null ? ` — floor ${diff.to_floor}` : ""}`}
        {diff.autoplayed > 0 && (
          <span className="autobattle-muted">
            {" "}
            · {diff.autoplayed} decision{diff.autoplayed === 1 ? "" : "s"} auto-played
          </span>
        )}
      </div>
      <Chips label="Upgraded" items={diff.deck.upgraded} tone="var(--accent-strong)" />
      <Chips label="Added" items={diff.deck.added} tone="var(--accent-strong)" />
      <Chips label="Removed" items={diff.deck.removed} tone="var(--hp)" />
      <Chips label="Relic" items={diff.relics_gained} tone="var(--accent-strong)" />
      <Chips label="Potion" items={diff.potions_gained} tone="var(--accent-strong)" />
      <Chips label="Used" items={diff.potions_lost} tone="var(--hp)" />
      {nothingMoved && <div className="autobattle-muted">Nothing changed.</div>}
      <div className="autobattle-diff-row autobattle-resources">
        {diff.hp != null && (
          <span>
            HP {diff.hp}
            {diff.max_hp != null ? `/${diff.max_hp}` : ""}
            {delta(diff.hp_delta)}
          </span>
        )}
        {diff.gold != null && (
          <span>
            Gold {diff.gold}
            {delta(diff.gold_delta)}
          </span>
        )}
        <span>Deck {diff.deck_size}</span>
      </div>
    </div>
  );
}

const BAND_STYLE: Record<AutobattleBand, { cls: string; head: string }> = {
  matched: { cls: "autobattle-agreed", head: "✓ Matched the champion" },
  near_tie: { cls: "autobattle-neartie", head: "≈ Near-tie — the policy barely preferred its own pick" },
  differed: { cls: "autobattle-differed", head: "✗ Differed from the champion" },
};

/** Policy probability as a compact percentage. Labelled as POLICY preference at
 *  the call site — it is not a win rate and must never be read as one. */
function pct(p: number | null | undefined): string | null {
  return p == null ? null : `${Math.round(p * 100)}%`;
}

/** The brain that graded the pick, as the verdict names it. */
function brainName(brain: string): string {
  if (brain === "e2e") return "Instinct (one net, no search)";
  if (brain === "ppo") return "PPO champion";
  if (brain === "heuristic") return "heuristic";
  return brain;
}

export function AutobattleVerdictCard({ verdict }: { verdict: AutobattleVerdict }) {
  if (!verdict.available) {
    return <div className="autobattle-verdict autobattle-muted">{verdict.reason ?? "Not graded."}</div>;
  }
  const band: AutobattleBand = verdict.band ?? (verdict.agreed ? "matched" : "differed");
  const style = BAND_STYLE[band];
  const yours = pct(verdict.your_prob);
  const champs = pct(verdict.champion_prob);
  return (
    <div className={`autobattle-verdict ${style.cls}`}>
      <div className="autobattle-verdict-head">
        {style.head}
        {verdict.brain && <span className="autobattle-muted"> · {brainName(verdict.brain)}</span>}
      </div>
      {band !== "matched" && (
        <div className="autobattle-verdict-body">
          <div>
            Champion: <b>{verdict.champion_label}</b>
            {champs && <span className="autobattle-muted"> · {champs}</span>}
          </div>
          <div>
            You: <b>{verdict.your_label}</b>
            {yours && <span className="autobattle-muted"> · {yours}</span>}
          </div>
        </div>
      )}
      <div className="autobattle-muted">
        {band === "near_tie"
          ? // Deliberate wording: a flat policy means the net was nearly
            // indifferent, which is NOT evidence the options are equally good —
            // an uncertain or undertrained surface looks identical here.
            "Percentages are how much the policy preferred each option, not win rates. A close call means it was nearly indifferent — not that the picks are equally good."
          : `Agreement with the ${verdict.brain === "e2e" ? "Instinct" : (verdict.brain ?? "run")} policy — not a win-rate judgement.`}
      </div>
    </div>
  );
}

export default function AutobattlePanel({
  diff,
  verdict,
  busy,
  over,
  won,
  watch,
  onWatchChange,
  onAdvance,
  onStop,
  timing,
}: {
  diff: AutobattleDiff | null;
  verdict: AutobattleVerdict | null;
  busy: boolean;
  over: boolean;
  won: boolean;
  watch: boolean;
  onWatchChange: (v: boolean) => void;
  onAdvance: () => void;
  onStop: () => void;
  /** The e2e student's per-decision timing while it plays the fights. */
  timing?: string | null;
}) {
  return (
    <div className="autobattle-panel">
      <div className="autobattle-head">
        <b>Autobattle</b>
        <span className="autobattle-muted">run-level decisions only</span>
        <label
          className="autobattle-watch"
          title="watch the auto-played fights play out move by move, instead of skipping straight to the next run-level decision">
          <input
            type="checkbox"
            checked={watch}
            onChange={(e) => onWatchChange(e.target.checked)}
          />
          watch fights
        </label>
      </div>
      {over ? (
        <div className="autobattle-diff-head">{won ? "Run won." : "Run over."}</div>
      ) : (
        <>
          {verdict && <AutobattleVerdictCard verdict={verdict} />}
          {diff ? <AutobattleDiffCard diff={diff} /> : <div className="autobattle-muted">Not started.</div>}
        </>
      )}
      {timing && <div className="autobattle-muted mono" data-testid="e2e-timing">{timing}</div>}
      <div className="autobattle-actions">
        <button type="button" onClick={onAdvance} disabled={busy || over}>
          {busy
            ? watch
              ? "Fighting…"
              : "Playing…"
            : diff
              ? watch
                ? "Play to next decision"
                : "Skip to next decision"
              : "Start autobattle"}
        </button>
        {/* Stop leaves the run mid-fight rather than unwinding it: pressing the
            button again resumes the same leg, and the change-diff still spans
            from the last stop. */}
        {busy && watch && (
          <button type="button" onClick={onStop} title="stop watching (Esc)">
            ⏹ stop
          </button>
        )}
      </div>
    </div>
  );
}
