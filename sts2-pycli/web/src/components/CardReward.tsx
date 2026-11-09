import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  api,
  assetUrl,
  type Step,
  type ChooseHandler,
  type RewardPreview,
  type CardAdvice,
  type AdvisorCandidate,
  type AdvisorProgress,
} from "../api";
import CardFrame from "./CardFrame";
import "../skins/cardreward.css";
import { detectCharacter, synergyFor, type PartnerHint } from "../synergy";
import { declaredGameVersion } from "../gameVersion";
import { loadArchetypes, archetypeChipsFor, type ArchetypeChip } from "../archetype";
import { useHints } from "../hints/useHints";
import { MarkBadge } from "../advice/AdviceContext";
import { markClass, markFor, useAdviceMarks } from "../advice/useAdviceMarks";
import { fitRewardCardWidth } from "../rewardFit";

/** Reward-card widths, in px, largest first.
 *
 * Two constraints pick the first number and they agree on it:
 *  - SHARPNESS. The card art is authored at 2x (card_template/frames/*.png is
 *    598x844 for card.tscn's 300x422 box), and the browser has no mipmaps, so
 *    an arbitrary minification ratio is what made the r3 bevels and plaque
 *    lettering read soft. 199 is 598/3 — a clean 1:3 downscale.
 *  - COMPOSITION. The game draws a reward card 233px wide in a 1920px frame
 *    (12.1% of the scene width); 199 in our ~1600px scene is 12.4%, so the
 *    trio occupies the same share of the picture as the game's does.
 *
 * The rest of the ladder exists because the scene is not always 1600px tall's
 * worth of window. On a tablet in landscape the screen box under the chrome is
 * ~517px, the composition wants ~557px, and the bottom of the trio and the
 * whole Skip plate were simply CLIPPED (the scene is `overflow: hidden`). So
 * the trio comes down a rung until it fits — but only onto 598/4 and 598/5,
 * the next two clean minifications, never an arbitrary "whatever fits" width
 * that would trade a clipped card for a soft one. */
const REWARD_CARD_LADDER = [199, 149, 119];

/** Backdrop region by act name, same keying as Board.tsx's arenaRegion()
 *  (the loc titles match the asset dirs); unknown acts fall back to the
 *  act-1 overgrowth ruin. The reward scene grades it down to the game's
 *  dark cave (see skins/cardreward.css). */
const REWARD_REGIONS = new Set(["overgrowth", "hive", "glory", "underdocks"]);
function rewardRegion(actName?: string | null): string {
  const n = (actName ?? "").trim().toLowerCase();
  return REWARD_REGIONS.has(n) ? n : "overgrowth";
}

const QUADRANT_COLOR: Record<PartnerHint["quadrant"], string> = {
  confirmed: "oklch(68% 0.13 145)", // green — co-picked AND causal
  undiscovered: "oklch(62% 0.16 300)", // purple — causal but under-picked
  myth: "var(--muted)",
  noise: "var(--muted)",
};

/** "synergizes with cards already in your deck" chip. Lists the deck partners
 *  this offered card forms a real (confirmed/undiscovered) synergy with, so the
 *  player can compare which of the offered cards fits their deck best. */
function SynergyChip({ hints }: { hints: PartnerHint[] }) {
  if (hints.length === 0) return null;
  const total = hints.reduce((n, h) => n + h.count, 0);
  return (
    <div
      className="mono"
      style={{ fontSize: 11, marginTop: 6, lineHeight: 1.5, maxWidth: 158 }}
      title="cards already in your deck that this card synergizes with (confirmed = green, undiscovered = purple)"
    >
      <div style={{ color: "var(--text)", fontWeight: 600 }}>
        ⚡ synergy ×{total}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 6px", justifyContent: "center" }}>
        {hints.map((h) => (
          <span key={h.partner} style={{ color: QUADRANT_COLOR[h.quadrant] }}>
            {h.label}
            {h.count > 1 ? `×${h.count}` : ""}
          </span>
        ))}
      </div>
    </div>
  );
}

const ARCHETYPE_COLOR: Record<ArchetypeChip["tag"], string> = {
  confirmed: "oklch(68% 0.13 145)", // green — co-picked AND causal
  undiscovered: "oklch(62% 0.16 300)", // purple — causal but under-picked
  folklore: "var(--muted)", // never chipped, here for exhaustiveness
};

/** "advances an archetype you're already building" chip: for each archetype the
 *  deck already commits to that this offered card is a weighty member of, show
 *  which piece it would be and whether that piece crosses into the measured
 *  super-additive part of the dose-response curve. */
function ArchetypeChipView({ chips }: { chips: ArchetypeChip[] }) {
  if (chips.length === 0) return null;
  return (
    <div
      className="mono"
      style={{ fontSize: 11, marginTop: 6, lineHeight: 1.5, maxWidth: 158 }}
      title="archetypes already in your deck that this card advances; ⚡ = the piece that turns super-additive on the measured combat dose-response curve"
    >
      {chips.map((c) => (
        <div key={c.id} style={{ color: ARCHETYPE_COLOR[c.tag] }}>
          🏛 {ordinal(c.piece)} {c.label}
          {c.superAdditive ? " ⚡" : ""}
        </div>
      ))}
    </div>
  );
}

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}

/** Compact "what this pick does to your deck" chip: headline volume delta plus
 *  per-axis (Damage / Block / Velocity) arrows. Green up, red down, grey flat. */
function DeltaChip({ delta }: { delta: RewardPreview["candidates"][number]["delta"] }) {
  const fmt = (v: number) => (v > 0 ? `+${v}` : `${v}`);
  const col = (v: number) =>
    Math.abs(v) < 0.5 ? "var(--muted)" : v > 0 ? "oklch(68% 0.13 145)" : "var(--hp)";
  const axes: [string, number][] = [
    ["D", delta.damage],
    ["B", delta.block],
    ["V", delta.velocity],
  ];
  return (
    <div className="mono" style={{ fontSize: 11, marginTop: 6, lineHeight: 1.5 }}
      title="change to your deck's Damage / Block / Velocity and overall Volume">
      <div style={{ color: col(delta.volume), fontWeight: 600 }}>
        volume {fmt(delta.volume)}
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        {axes.map(([k, v]) => (
          <span key={k} style={{ color: col(v) }}>{k} {fmt(v)}</span>
        ))}
      </div>
    </div>
  );
}

/** Per-candidate boss-fight simulation result attached by the self-play
 *  recorder (run_policy.boss_sim_card_reward): each offered card (and skip)
 *  was played against the act's pinned boss with MCTS; `score` is
 *  `10000 + final HP` on a win, `player HP − enemy HP` otherwise. */
interface RewardSimCandidate {
  idx: number | null;
  label: string;
  score: number;
  win: boolean;
}
interface RewardSim {
  encounter?: string;
  sims?: number;
  candidates?: RewardSimCandidate[];
}

function SimChip({ c }: { c: RewardSimCandidate }) {
  return (
    <div
      className="mono"
      title={`boss-fight simulation score ${c.score.toFixed(0)}`}
      style={{ fontSize: 11, marginTop: 5, color: c.win ? "oklch(68% 0.13 145)" : "var(--muted)" }}
    >
      {c.win ? `▲ boss win · ${(c.score - 10000).toFixed(0)} hp left` : `▼ boss loss · ${c.score.toFixed(0)}`}
    </div>
  );
}

/** Live card-pick advisor result for one option: the act win-rate from the
 *  rest-of-act Monte-Carlo, plus the composite delta vs the best alternative.
 *  Green + ★ on the recommended pick. */
function AdviceChip({ c }: { c: AdvisorCandidate }) {
  const pct = (c.win_rate * 100).toFixed(0);
  const col = c.is_best ? "oklch(68% 0.13 145)" : "var(--muted)";
  const dpct = (c.delta * 100).toFixed(0);
  return (
    <div
      className="mono"
      title={
        `act win-rate over ${c.n} rollouts of the rest of the act ` +
        `(MCTS in combat). Δ is this option's composite value (act-win + ` +
        `depth reached + HP carried) minus the best other option; ` +
        `~${c.avg_floors_advanced.toFixed(1)} floors advanced, ` +
        `${c.avg_hp_at_deepest.toFixed(0)} HP at the deepest room.`
      }
      style={{ fontSize: 11, marginTop: 5, lineHeight: 1.45, color: col, fontWeight: c.is_best ? 600 : 400 }}
    >
      <div>{c.is_best ? "★ " : ""}{pct}% act win</div>
      <div style={{ color: "var(--muted)", fontWeight: 400 }}>
        Δ {c.delta >= 0 ? "+" : ""}{dpct} · {c.avg_hp_at_deepest.toFixed(0)} hp
      </div>
    </div>
  );
}

/** Live progress state for the streaming card-pick advisor. */
type AdvProgress = {
  done: number; // playouts finished
  total: number; // playouts across all options
  rollouts: number; // playouts per option
  perCand: Map<number | "skip", number>; // completed count per option
  cands: { key: number | "skip"; label: string }[]; // option order + labels
};

export default function CardReward({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  // The run policy's marks on the cards and the skip (advice/marks.ts).
  const { marks } = useAdviceMarks();
  const s = step.state;
  const cards = s.cards ?? [];
  const action = step.chosen_action as { action?: string; args?: { card_index?: number } } | null;
  const skipped = action?.action === "skip_card_reward";
  const chosenIdx = action?.action === "select_card_reward" ? action.args?.card_index : undefined;
  // NOTHING on this screen is ever dimmed: the game renders all three reward
  // cards at full brightness until the screen closes, and a veil over the two
  // rejected ones reads as "broken / mid-transition" rather than as a live
  // choice (round-3 P0). The replay record of which card was taken is carried
  // entirely by the gold rim + corner medallion ON the taken card.
  const sim = (s as { reward_sim?: RewardSim }).reward_sim;
  const simByIdx = new Map((sim?.candidates ?? []).map((c) => [c.idx, c]));
  const skipSim = (sim?.candidates ?? []).find((c) => c.idx == null);

  // Every chip on this screen (deck-axis delta, synergy, archetype, the EV
  // advisor) is decision support. While YOU are the one picking — live play,
  // i.e. `onChoose` is set — they wait for the global hints switch. Replay has
  // no decision to spoil: it is review of a pick already made, so it keeps them.
  const hints = useHints();
  const showAdvice = hints.on || onChoose == null;

  // Deck-axis preview: how each offered card would shift Damage/Block/Velocity.
  // The reward-screen deck is the pre-pickup baseline, so the deltas are honest.
  const deck = s.player?.deck ?? [];
  const [preview, setPreview] = useState<RewardPreview | null>(null);
  useEffect(() => {
    setPreview(null);
    if (!showAdvice || cards.length === 0 || deck.length === 0) return;
    let live = true;
    api
      .deckRewardPreview(
        deck.map((c) => ({ id: c.id, upgraded: c.upgraded })),
        cards.map((c) => ({ index: c.index, id: c.id, upgraded: c.upgraded })),
        // The run's own build's card tables (GH #877).
        declaredGameVersion(step),
      )
      .then((r) => { if (live) setPreview(r); })
      .catch(() => {});
    return () => { live = false; };
    // Refetch on a new reward screen or a deck/candidate count change — not on
    // array identity (which churns every render).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step.idx, cards.length, deck.length, showAdvice]);
  const deltaByIdx = new Map((preview?.candidates ?? []).map((c) => [c.index, c.delta]));

  // Synergy hints: for each offered card, which deck cards it synergizes with.
  // Character is detected from the deck's STRIKE_<X>/DEFEND_<X> basic so it is
  // correct in both live play and replay (no run-meta dependency).
  const [synergy, setSynergy] = useState<Map<number, PartnerHint[]>>(new Map());
  useEffect(() => {
    setSynergy(new Map());
    if (!showAdvice || cards.length === 0 || deck.length === 0) return;
    const deckTokens = deck.map((c) => c.id);
    const character = detectCharacter(deckTokens.map((t) => t.includes(".") ? t.slice(t.indexOf(".") + 1) : t));
    if (!character) return;
    let live = true;
    Promise.all(
      cards.map((c) =>
        synergyFor(character, c.id, deckTokens).then((hints) => [c.index, hints] as const),
      ),
    )
      .then((entries) => { if (live) setSynergy(new Map(entries)); })
      .catch(() => {});
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step.idx, cards.length, deck.length, showAdvice]);

  // Archetype chips: for each offered card, which archetypes already in the deck
  // it advances (and whether the next piece is super-additive on the curve).
  const [archetypes, setArchetypes] = useState<Map<number, ArchetypeChip[]>>(new Map());
  useEffect(() => {
    setArchetypes(new Map());
    if (!showAdvice || cards.length === 0 || deck.length === 0) return;
    const deckTokens = deck.map((c) => c.id);
    const character = detectCharacter(deckTokens.map((t) => (t.includes(".") ? t.slice(t.indexOf(".") + 1) : t)));
    if (!character) return;
    let live = true;
    loadArchetypes(character)
      .then((doc) => {
        if (!live) return;
        const entries = cards.map(
          (c) => [c.index, archetypeChipsFor(doc, c.id, deckTokens)] as const,
        );
        setArchetypes(new Map(entries));
      })
      .catch(() => {});
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step.idx, cards.length, deck.length, showAdvice]);

  // Live card-pick advisor (EV-based): the single coolest thing the engine
  // unlocks. Deep + slow (forks the run and Monte-Carlos the rest of the act
  // with MCTS in every fight), so it only runs when the player presses the
  // button — never automatically. Play mode only (needs the live run to fork).
  const [advice, setAdvice] = useState<CardAdvice | null>(null);
  const [advising, setAdvising] = useState(false);
  // Live progress while the analysis streams: total playouts, how many are
  // done, and the per-option completed counts (so each take/skip row fills in).
  const [prog, setProg] = useState<AdvProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(0);
  const adviceByIdx = new Map<number | "skip", AdvisorCandidate>(
    (advice?.candidates ?? []).map((c) => [c.idx == null ? "skip" : c.idx, c]),
  );
  // A fresh reward screen invalidates any prior advice.
  useEffect(() => { setAdvice(null); setAdvising(false); setProg(null); setElapsed(0); }, [step.idx]);
  // Tick the elapsed-seconds readout while the analysis is running.
  useEffect(() => {
    if (!advising) return;
    const id = setInterval(() => setElapsed((Date.now() - startRef.current) / 1000), 100);
    return () => clearInterval(id);
  }, [advising]);
  async function runAdvisor() {
    startRef.current = Date.now();
    setElapsed(0);
    setProg(null);
    setAdvice(null);
    setAdvising(true);
    try {
      const result = await api.live.cardAdvisorStream({}, (ev: AdvisorProgress) => {
        if (ev.type === "start") {
          setProg({
            done: 0,
            total: ev.total,
            rollouts: ev.rollouts,
            perCand: new Map(ev.candidates.map((c) => [c.idx == null ? "skip" : c.idx, 0])),
            cands: ev.candidates.map((c) => ({ key: c.idx == null ? "skip" : c.idx, label: c.label })),
          });
        } else {
          setProg((p) => {
            if (!p) return p;
            const key = ev.candidate_idx == null ? "skip" : ev.candidate_idx;
            const perCand = new Map(p.perCand);
            perCand.set(key, (perCand.get(key) ?? 0) + 1);
            return { ...p, done: ev.done, perCand };
          });
        }
      });
      setAdvice(result);
    } catch (e) {
      setAdvice({ candidates: [], error: e instanceof Error ? e.message : String(e) });
    } finally {
      setAdvising(false);
      setProg(null);
    }
  }

  // The advisor rail only exists when there is (or can be) evidence to show:
  // replay always reviews, live play shows it once the hints switch is on.
  const railVisible = showAdvice || (onChoose != null && hints.on);
  // Skip gets a rail cell only when it carries evidence (sim/EV/the choice).
  const skipCell = s.can_skip && (skipSim != null || adviceByIdx.has("skip") || skipped);

  // Fit the trio to the box the scene actually got. The frame is the box (it is
  // `flex: 1; min-height: 0`, so it is exactly the scene's height and never
  // grows to its own content); the fit layer inside it carries the composition
  // at its natural height, which is what the ladder is chosen against.
  //
  // Both boxes are observed — the frame because the window resizes and the
  // advisor rail below appears, the fit layer because the banner art landing
  // (or a new reward's card text wrapping to another line) moves the natural
  // height under us. That cannot loop: `fitRewardCardWidth` decides from
  // width-INVARIANT measurements, so re-measuring after a rung change returns
  // the same rung.
  const frameRef = useRef<HTMLDivElement>(null);
  const fitRef = useRef<HTMLDivElement>(null);
  const rowRef = useRef<HTMLDivElement>(null);
  const [cardW, setCardW] = useState(REWARD_CARD_LADDER[0]);
  useLayoutEffect(() => {
    const frame = frameRef.current, fit = fitRef.current, row = rowRef.current;
    if (!frame || !fit || !row || typeof ResizeObserver === "undefined") return;
    const measure = () =>
      setCardW((w) => fitRewardCardWidth(
        REWARD_CARD_LADDER, frame.clientHeight, fit.offsetHeight, row.offsetHeight, w));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(frame);
    ro.observe(fit);
    return () => ro.disconnect();
  }, []);

  const region = rewardRegion(step.meta?.act_name);
  // The room's parallax plates, back to front. What separates the reference
  // frame from a graded rectangle is not brightness (r4 matched the backdrop
  // luminance to −3% of the ref and depth still scored 3.0) and not haze — it
  // is SEPARATED PLANES at distinct values: a lit back wall with rock forms on
  // it, a mid band of ruins, near-black columns crowding both edges, and a
  // floor the cards can drop a shadow onto. r4 drew those as CSS polygons,
  // which read as vector sawteeth; these are the game's own room layers — the
  // same plates Board.tsx stacks for the arena — regraded much darker, one
  // exposure step per plane. `03_a` is absent from the underdocks extract; its
  // midground ruins are `03_b`.
  const layer = (rel: string) => `url(${assetUrl(`rooms/${region}/${region}_${rel}.png`)})`;
  const midLayer = region === "underdocks" ? "03_b" : "03_a";

  return (
    <div className="gs-cardreward-wrap">
      {/* ---- game composition: banner → trio → Skip, in the darkened room ---- */}
      <div
        className="gs-scene gs-cardreward"
        style={
          {
            "--gs-scene-bg": layer("00"),
            "--gs-scene-pos": "center 62%",
            "--rw-far": layer("02_b"),
            "--rw-mid": layer(midLayer),
            "--rw-hang": layer("01_a"),
            "--rw-near": `${layer("04_b")}, ${layer("04_a")}`,
          } as CSSProperties
        }
      >
        {/* Five planes, back to front, all under the shared .gs-scene vignette:
            far stone wall → the light pass that lifts it (back-wall dome, warm
            altar pool, floor sheen) → midground ruins → the near-black
            foreground columns and ceiling hang → the atmospheric veil that
            hazes the middle distance, closes the corners and fades the scene's
            bottom edge into black (r4's dead-straight cut at the advisor rail
            was called the single most div-like thing in the frame). */}
        <div className="gs-cardreward-far" aria-hidden="true" />
        <div className="gs-cardreward-cast" aria-hidden="true" />
        <div className="gs-cardreward-mid" aria-hidden="true" />
        <div className="gs-cardreward-near" aria-hidden="true" />
        <div className="gs-cardreward-veil" aria-hidden="true" />
        <div className="gs-cardreward-frame" ref={frameRef}>
          <div className="gs-cardreward-fit" ref={fitRef}>
            <div className="gs-cardreward-banner">
              <img src={assetUrl("ui/reward_screen/reward_banner.png")} alt="" />
              <h2 className="gs-cardreward-title">Choose a Card</h2>
            </div>
            {/* The gold the fight paid, and (in replay) what was actually taken.
                These are FACTS about the run, not advice — they were on the
                screen unconditionally before the advisor rail existed, and the
                rail is gated behind a hints switch that is OFF by default, so
                putting them there hid the fight's gold from live play. */}
            {(s.gold_earned != null || onChoose == null) && (
              <div className="gs-cardreward-ledger mono">
                {s.gold_earned != null && <span>+{s.gold_earned} gold</span>}
                {onChoose == null && (
                  <span>
                    {skipped ? "✓ skipped" : chosenIdx != null ? "✓ card taken" : "no choice recorded"}
                  </span>
                )}
              </div>
            )}
            <div className="gs-cardreward-cards" ref={rowRef}>
              {cards.map((c) => {
                const taken = c.index === chosenIdx;
                // No fan: the game stands all three reward cards upright on one
                // line. The r3 tilt also resampled every glyph on the card
                // (a rotated layer is re-rasterized), which is what read as
                // "drawn small and scaled up".
                const cls =
                  "gs-cardreward-slot" +
                  (onChoose ? " is-live" : "") +
                  (taken && onChoose == null ? " is-chosen" : "");
                const mark = markFor(marks, "select_card_reward", { card_index: c.index });
                return (
                  <div key={c.index} className={cls}>
                    <CardFrame card={c} width={cardW} mark={mark?.tier} step={mark?.order}
                      onClick={onChoose ? () => onChoose("select_card_reward", { card_index: c.index }) : undefined} />
                    {taken && onChoose == null && (
                      <div className="gs-cardreward-chosenbadge" title="taken" aria-hidden="true" />
                    )}
                  </div>
                );
              })}
            </div>
            {s.can_skip && (
              <button
                className={
                  "gs-cardreward-skip" +
                  (skipped && onChoose == null ? " is-chosen" : "") +
                  markClass(markFor(marks, "skip_card_reward"))
                }
                disabled={!onChoose}
                onClick={() => onChoose?.("skip_card_reward", {})}
              >
                <img src={assetUrl("ui/reward_screen/reward_skip_button.png")} alt="" />
                <span>Skip</span>
                <MarkBadge mark={markFor(marks, "skip_card_reward")} />
                {skipped && onChoose == null && (
                  <div className="gs-cardreward-chosenbadge" title="skipped" aria-hidden="true" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ---- advisor rail: the workbench's decision evidence, OFF the game
              surface in its own labelled tooling strip ---- */}
      {railVisible && (
        <div className="gs-cardreward-advisor mono">
          <div className="gs-cardreward-adv-head">
            <span className="gs-cardreward-adv-tag">advisor</span>
            {/* gold + outcome live on the game surface under the banner — see
                the ledger line there; the rail carries only advice. */}
            {sim?.encounter && (
              <span style={{ fontSize: 11 }}>
                picked by simulating {sim.encounter} ({sim.sims ?? "?"} sims/move)
              </span>
            )}
            {onChoose != null && hints.on && (
              <button
                className="mono"
                onClick={runAdvisor}
                disabled={advising}
                title="Deep EV analysis: fork the run and play the rest of the act out many times for take A / B / C / skip (MCTS in every fight). Slow — a few seconds."
                style={{ marginLeft: "auto", fontSize: 11 }}
              >
                {advising ? "analyzing…" : "🎲 analyze picks (EV)"}
              </button>
            )}
          </div>
          {advising && (
            <div style={{ marginBottom: 10, fontSize: 11 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span>
                  analyzing picks…{prog ? ` ${prog.done}/${prog.total} sims` : ""}
                </span>
                <span>{elapsed.toFixed(1)}s</span>
              </div>
              <div style={{ height: 6, background: "var(--field-bg)", borderRadius: 3, overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: prog && prog.total ? `${(prog.done / prog.total) * 100}%` : "0%",
                    background: "var(--accent-strong)",
                    transition: "width 0.2s ease",
                  }}
                />
              </div>
              {prog && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginTop: 6 }}>
                  {prog.cands.map((c) => {
                    const n = prog.perCand.get(c.key) ?? 0;
                    const full = n >= prog.rollouts;
                    return (
                      <span key={String(c.key)} style={{ color: full ? "var(--accent-strong)" : "var(--muted)" }}>
                        {c.key === "skip" ? "skip" : `take ${c.label}`} {n}/{prog.rollouts}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          )}
          {advice?.error && (
            <div style={{ color: "var(--hp)", marginBottom: 10, fontSize: 11 }}>
              advisor unavailable: {advice.error}
            </div>
          )}
          {advice && !advice.error && (
            <div style={{ marginBottom: 10, fontSize: 11 }}>
              act win-rate over {advice.candidates[0]?.n ?? "?"} rollouts of the rest of act {advice.act}
              {" "}· ★ = recommended
            </div>
          )}
          <div className="gs-cardreward-adv-grid">
            {cards.map((c) => {
              const taken = c.index === chosenIdx;
              const cs = simByIdx.get(c.index);
              return (
                <div key={c.index} className="gs-cardreward-adv-cell">
                  <div className="gs-cardreward-adv-cellname">
                    {c.name}
                    {taken && onChoose == null && (
                      <span style={{ color: "var(--accent-strong)" }}> · ✓ taken</span>
                    )}
                  </div>
                  {deltaByIdx.has(c.index) && <DeltaChip delta={deltaByIdx.get(c.index)!} />}
                  {synergy.has(c.index) && <SynergyChip hints={synergy.get(c.index)!} />}
                  {archetypes.has(c.index) && <ArchetypeChipView chips={archetypes.get(c.index)!} />}
                  {cs && showAdvice && <SimChip c={cs} />}
                  {adviceByIdx.has(c.index) && <AdviceChip c={adviceByIdx.get(c.index)!} />}
                </div>
              );
            })}
            {skipCell && (
              <div className="gs-cardreward-adv-cell">
                <div className="gs-cardreward-adv-cellname">
                  skip
                  {skipped && onChoose == null && (
                    <span style={{ color: "var(--accent-strong)" }}> · ✓ skipped</span>
                  )}
                </div>
                {skipSim && <SimChip c={skipSim} />}
                {adviceByIdx.has("skip") && <AdviceChip c={adviceByIdx.get("skip")!} />}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
