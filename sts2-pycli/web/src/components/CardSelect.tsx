import { useEffect, useState, type ReactNode } from "react";
import { api, type ChoiceRankRow, type Step, type ChooseHandler, type Card } from "../api";
import CardFrame from "./CardFrame";
import { cardBoxHeight } from "../cardGeometry";
import { useHints } from "../hints/useHints";
import { useMediaQuery, SHORT_QUERY, TOUCH_QUERY } from "../useMediaQuery";
import { DECK_OP_CONTEXTS } from "../choiceKinds";
import { DecisionAnalyzeButton, DecisionAnalyzeStatus, DecisionWinChip } from "./DecisionAdvisor";
import { useDecisionAdvisor } from "./useDecisionAdvisor";

// Which card_selects are in-combat pending choices (vs. deck operations) lives
// in ../choiceKinds — App shares the same distinction for "play best".

// Contexts whose pick UPGRADES the chosen card (rest-site Smith / Pomander, and
// the in-combat Armaments-style pick). The backend ships each un-upgraded
// option's upgraded twin as ``upgrade_preview``, so the screen can show the
// game's own before → after comparison (`NUpgradePreview`) for the card the
// player is looking at.
const UPGRADE_CONTEXTS = new Set(["upgrade", "upgrade_in_hand"]);

/** Width of one option card in the grid, desktop and short-viewport.
 *
 *  Both sit on the near-clean minifications of the 598px-wide source art
 *  (598/4 = 149.5, 598/5 = 119.6): the browser has no mipmaps, so a ratio far
 *  off an integer one is what makes the frame bevels and the type plaque read
 *  soft. Same reasoning as CardReward's REWARD_CARD_LADDER. */
const OPTION_CARD_W = 150;
const OPTION_CARD_W_SHORT = 120;

// What each card_select is *for*, so the prompt isn't a bare "Select cards".
// Keys are the backend ``context`` = the choice kind lowercased (Rust
// ``ChoiceKind`` via ``_fill_combat_choice``), plus the deck-op flavors. The
// few that already had titles (upgrade/remove) keep their wording.
const CHOICE_TITLES: Record<string, string> = {
  // deck operations
  upgrade: "Upgrade a card",
  remove: "Remove a card",
  transform: "Transform a card",
  enchant: "Enchant a card",
  grant: "Add a card to your deck",
  // in-combat pending choices — keys are the Rust ChoiceKind snake_case names
  // (``choice_kind_python_name``), which the backend passes through as context.
  card_select: "Choose a card",
  put_back: "Put a card on top of your draw pile",
  exhaust_pile: "Choose a card to exhaust",
  exhaust_from_hand: "Choose cards to exhaust",
  orb_slot: "Choose an orb slot",
  enemy_target: "Choose an enemy to target",
  add_generated_to_hand: "Choose a card to add to your hand",
  add_generated_retained_to_hand: "Choose a card to add to your hand (retained)",
  add_generated_free_to_hand: "Choose a card to add to your hand (free this turn)",
  upgrade_in_hand: "Choose a card to upgrade",
  add_from_draw_by_type: "Choose a card from your draw pile",
  add_from_discard_free: "Choose a card from your discard pile (free this turn)",
  discard_from_hand: "Choose cards to discard",
  discard_from_hand_and_draw: "Choose cards to discard, then draw",
  make_free_in_hand: "Make a card free for this combat",
  add_from_draw_to_hand: "Choose cards from your draw pile",
  foregone_conclusion_from_draw: "Choose cards from your draw pile",
  add_from_discard_to_hand: "Choose a card from your discard pile",
  apply_retain_in_hand: "Choose a card to give Retain",
  apply_sly_in_hand: "Choose a Skill to give Sly",
  apply_ethereal_in_hand: "Choose a card to give Ethereal",
  transform_in_hand_to_minion_strike: "Choose a card to transform",
  transform_in_hand_to_minion_sacrifice: "Choose cards to transform",
  transfigure_card_in_hand: "Choose a card to transfigure",
  transform_draw_card_to_soul: "Choose a draw-pile card to turn into a Soul",
  transform_draw_cards_to_minion_dive_bomb: "Choose draw-pile cards to transform",
  clone_chosen_card_next_turn: "Choose a card to clone next turn",
  retain_at_turn_end: "Choose cards to retain this turn",
  auto_play_skill_from_hand: "Choose a Skill to replay",
  monster_curse_select: "Choose a curse",
  paels_tooth_stash: "Choose cards to stash",
};

/**
 * Generic "select N cards" screen (e.g. event card grabs, deck operations).
 * Accumulates a selection up to ``max_select`` then submits ``select_cards``
 * with a comma-joined index list (matching the normalized option contract).
 * Read-only in Replay, where the recorded selection (which card was
 * upgraded / removed / picked) is highlighted and captioned.
 */
export default function CardSelect({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  // Tablet in landscape: ~517px of screen box under the chrome, against a
  // layout authored for a maximised desktop window. The option cards and the
  // upgrade preview both come down a size, and the padding/gaps with them —
  // see OPTION_CARD_W / PV_CARD_W for why these particular numbers.
  const short = useMediaQuery(SHORT_QUERY);
  // No hover means the preview can only ever be filled by a PICK, so its
  // empty state has nothing to hold still for — see UpgradeComparison.
  const touch = useMediaQuery(TOUCH_QUERY);
  const cardW = short ? OPTION_CARD_W_SHORT : OPTION_CARD_W;
  const pad = short ? 12 : 24;
  const gap = short ? 10 : 18;
  const s = step.state;
  const cards: Card[] = s.cards ?? [];
  const min = s.min_select ?? 1;
  const max = s.max_select ?? 1;
  // Two independent ways a select is skippable, and only one is `min === 0`.
  // `can_skip` is the engine's own `canSkip` on a choice that otherwise wants a
  // card: Discovery and the Colorless / Skill / Attack / Power potions are
  // min 1, max 1, can_skip true. Reading `min` alone hid a legal action — the
  // MCTS chips below score a skip the player had no button for (bug
  // wb-31c2d79f995fe1a6).
  const skippable = min === 0 || s.can_skip === true;
  const [picked, setPicked] = useState<number[]>([]);

  // The rust backend stamps the select's purpose on ``context``
  // ("upgrade" = rest-site Smith, "remove" = deck removal, otherwise a
  // combat pending-choice kind).
  const context = (s as { context?: string }).context;
  const title = (context && CHOICE_TITLES[context]) ?? "Choose a card";
  const verb = context === "upgrade" ? "upgraded" : context === "remove" ? "removed" : "chosen";
  const action = step.chosen_action as { action?: string; args?: { indices?: string } } | null;
  const chosen: number[] =
    onChoose == null && action?.action === "select_cards"
      ? String(action.args?.indices ?? "")
          .split(",")
          .filter((t) => t.trim() !== "")
          .map(Number)
      : [];

  // before → after preview: follows the hovered card, else the picked one, else
  // (in replay) the card the run actually upgraded.
  const [hovered, setHovered] = useState<number | null>(null);
  const isUpgrade = UPGRADE_CONTEXTS.has(context ?? "") && cards.some((c) => c.upgrade_preview);
  const focusIdx = hovered ?? picked[picked.length - 1] ?? chosen[0] ?? null;
  const focus = focusIdx == null ? undefined : cards.find((c) => c.index === focusIdx);

  const toggle = (idx: number) => {
    if (!onChoose) return;
    setPicked((cur) => {
      if (cur.includes(idx)) return cur.filter((i) => i !== idx);
      if (max === 1) return [idx];
      return cur.length < max ? [...cur, idx] : cur;
    });
  };
  // The engine's ``select_cards`` verb treats a SINGLE submitted index on a
  // sequential multi-pick choice (max > 1) as an incremental accumulate step
  // (bead sts2-rs-sopgf) — it re-opens the identical choice instead of
  // finalizing, so a Confirm with exactly one card checked would otherwise
  // look stuck (the dialog never closes). Follow up with ``skip_select``
  // (finalize with whatever's accumulated so far) to complete it. 0 or 2+
  // checked cards already resolve in one shot (true skip / atomic multi-select).
  const submit = async () => {
    if (!onChoose) return;
    if (picked.length === 0) {
      await onChoose("skip_select", {});
      return;
    }
    await onChoose("select_cards", { indices: picked.join(",") });
    if (picked.length === 1 && max > 1) {
      await onChoose("skip_select", {});
    }
  };

  // EV advisor: live, single-pick choices only — an in-combat pending choice or
  // a run-level choose screen (Lead Paperweight's 2 Colorless, Hefty Tablet's 3
  // Rares); the advisor forks the run per option and can't fork the deck-op
  // selects. Options key by ``choice:<index>`` (skip = ``choice:skip``) to join
  // the win-rate chips, in both cases.
  const isCombatChoice = !DECK_OP_CONTEXTS.has(context ?? "");
  const advEnabled = onChoose != null && isCombatChoice && max === 1;
  const resetKey = cards.map((c) => `${c.index}:${c.id}`).join(",");
  const adv = useDecisionAdvisor(resetKey, advEnabled);
  // The fast ranking chips below are AI advice fetched with no button press at
  // all, so they follow the global hints switch (the EV advisor gates itself).
  const hints = useHints();

  // Fast combat-MCTS choice ranking: ONE search over the live combat state —
  // the same search the deployed run-AI performs at this decision — fetched
  // automatically when the choice opens.
  // The slow EV analysis behind the button supersedes these chips once run.
  const [ranking, setRanking] = useState<Map<string, ChoiceRankRow>>(new Map());
  useEffect(() => {
    setRanking(new Map());
    if (!advEnabled || !hints.on) return; // live, single-pick, in-combat choices only
    let alive = true;
    api.live
      .choiceRanker({})
      .then((r) => {
        if (!alive || r.error) return;
        setRanking(new Map(r.candidates.map((c) => [c.key, c])));
      })
      .catch(() => {}); // 409 off-choice / search failure → silently omit
    return () => { alive = false; };
  }, [resetKey, advEnabled, hints.on]);

  return (
    <div style={{ padding: pad, display: "flex", flexDirection: "column", gap }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
        <h2 style={{ margin: 0, fontWeight: 600 }}>{title}</h2>
        <span style={{ color: "var(--muted)" }}>
          {onChoose
            ? `${min === max ? `choose ${max}` : `choose ${min}–${max}`} · ${picked.length} selected`
            : chosen.length > 0
              ? `${verb}: ${chosen.map((i) => cards.find((c) => c.index === i)?.name ?? `#${i}`).join(", ")}`
              : action?.action === "skip_select"
                ? "skipped"
                : "no selection recorded"}
        </span>
        <DecisionAnalyzeButton adv={adv} style={{ marginLeft: "auto" }} />
      </div>
      <DecisionAnalyzeStatus adv={adv} />
      {isUpgrade && (
        <UpgradeComparison card={focus} live={onChoose != null}
          width={short ? PV_CARD_W_SHORT : PV_CARD_W} reserveEmpty={!touch} />
      )}
      <div style={{ display: "flex", gap: short ? 10 : 16, flexWrap: "wrap" }}>
        {cards.map((c) => (
          <div key={c.index} style={{ textAlign: "center" }}
            onMouseEnter={isUpgrade ? () => setHovered(c.index) : undefined}
            onMouseLeave={isUpgrade ? () => setHovered(null) : undefined}>
            <CardFrame card={c} width={cardW}
              best={picked.includes(c.index) || chosen.includes(c.index)}
              dim={(onChoose != null && picked.length >= max && !picked.includes(c.index)) ||
                (chosen.length > 0 && !chosen.includes(c.index))}
              onClick={onChoose ? () => toggle(c.index) : undefined} />
            {chosen.includes(c.index) && (
              <div style={{ fontFamily: "var(--game-font)", fontSize: 12, color: "var(--accent-strong)", marginTop: 6 }}>✓ {verb}</div>
            )}
            {adv.byKey.has(`choice:${c.index}`) ? (
              <DecisionWinChip c={adv.byKey.get(`choice:${c.index}`)!} adv={adv} />
            ) : (
              <SearchChip c={ranking.get(`choice:${c.index}`)} />
            )}
          </div>
        ))}
      </div>
      {onChoose && (
        <div style={{ display: "flex", gap: 12 }}>
          <button disabled={picked.length < min} onClick={submit}>Confirm ({picked.length})</button>
          {skippable && (
            <div style={{ textAlign: "center" }}>
              <button onClick={() => onChoose("skip_select", {})}>Skip</button>
              {adv.byKey.has("choice:skip") ? (
                <DecisionWinChip c={adv.byKey.get("choice:skip")!} adv={adv} />
              ) : (
                <SearchChip c={ranking.get("choice:skip")} />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Upgrade-preview geometry. On a device with a pointer the box has to be
 *  EXACTLY as tall empty as it is filled, or the card grid under it steps down
 *  the moment you hover a card — and walking the cursor along the row makes it
 *  flicker as the grid chases the pointer. A `minHeight: 268` guess used to do
 *  this job and was ~20px short.
 *
 *  Every number below is also what the layout is built from (the label and
 *  footer get these as fixed line boxes, and the card height comes from
 *  `cardBoxHeight`, the same helper `CardFrame` sizes itself with), so the
 *  reservation cannot drift away from the thing it reserves for. */
const PV_CARD_W = 160;
const PV_CARD_W_SHORT = 120;
const PV_LABEL_H = 16;
const PV_LABEL_GAP = 6;
const PV_FOOTER_H = 15;
const PV_FOOTER_GAP = 4;
const pvRowH = (w: number) =>
  PV_LABEL_H + PV_LABEL_GAP + cardBoxHeight(w) + PV_FOOTER_GAP + PV_FOOTER_H;

/** The game's upgrade preview: the card as it is now, an arrow, and the card as
 *  it will be once upgraded (`NUpgradePreview` — rest site, Armaments, …).
 *  Follows the hovered/picked card.
 *
 *  `reserveEmpty` is off on a touchscreen, and that is the whole point: with no
 *  hover the box can only fill from a deliberate TAP, so there is no cursor for
 *  the grid to chase and nothing to hold still for. Reserving two full card
 *  heights for one line of placeholder text spent ~290px of a 517px tablet
 *  screen on empty panel and pushed the cards themselves off the bottom. */
function UpgradeComparison(
  { card, live, width, reserveEmpty }:
  { card?: Card; live: boolean; width: number; reserveEmpty: boolean },
) {
  const after = card?.upgrade_preview;
  const filled = card != null && after != null;
  return (
    <div
      style={{
        padding: "12px 16px", borderRadius: 10,
        border: "1px solid var(--border)", background: "var(--panel)",
        // Stays in view while the cursor walks a long deck below it.
        position: "sticky", top: 0, zIndex: 2,
      }}
    >
      {/* Fixed-height row: this is what holds the grid below still. */}
      <div data-testid="upgrade-preview-row" style={{
        height: filled || reserveEmpty ? pvRowH(width) : undefined,
        display: "flex", alignItems: "center", justifyContent: "center", gap: 22,
      }}>
        {filled ? (
          <>
            <Labeled label="now"><CardFrame card={card} width={width} /></Labeled>
            <div aria-hidden="true" style={{ fontSize: 30, color: "var(--accent-strong)" }}>→</div>
            <Labeled label="upgraded" footer={after.keyword_delta}>
              <CardFrame card={after} width={width} />
            </Labeled>
          </>
        ) : (
          <span style={{ color: "var(--muted)" }}>
            {live
              ? reserveEmpty
                ? "Hover or pick a card to preview its upgrade"
                : "Pick a card to preview its upgrade"
              : "No upgrade preview for this pick"}
          </span>
        )}
      </div>
    </div>
  );
}

/** One side of the comparison. The label and footer are fixed line boxes (the
 *  footer is empty on the "now" side) so both columns keep the same height,
 *  stay aligned, and add up to `PV_ROW_H` whatever the text does. */
function Labeled({ label, footer, children }: { label: string; footer?: string; children: ReactNode }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{
        fontFamily: "var(--game-font)", fontSize: 12, color: "var(--muted)",
        height: PV_LABEL_H, lineHeight: `${PV_LABEL_H}px`, marginBottom: PV_LABEL_GAP,
      }}>{label}</div>
      {children}
      <div style={{
        fontFamily: "var(--game-font)", fontSize: 12, color: "oklch(72% 0.13 150)",
        height: PV_FOOTER_H, lineHeight: `${PV_FOOTER_H}px`, marginTop: PV_FOOTER_GAP,
        overflow: "hidden", whiteSpace: "nowrap",
      }}>
        {footer ?? ""}
      </div>
    </div>
  );
}

/** Fast combat-MCTS ranking chip for one choice option, shown automatically
 *  (no button press) on live in-combat card-selects. The slow EV advisor's
 *  DecisionWinChip supersedes it. */
function SearchChip({ c }: { c: ChoiceRankRow | undefined }) {
  if (!c || c.win_prob == null) return null;
  // The e2e advisor ranked this choice: its policy probability, not a search.
  const e2e = c.prob != null;
  return (
    <div
      title={e2e
        ? "Instinct's probability for this option — one forward of the net, " +
          "a policy probability, not a win chance."
        : "Combat-MCTS win probability from one search over this choice — the same " +
          `search the deployed run-AI uses here (${c.visits} visits; q ${c.q?.toFixed(2) ?? "—"}).`
      }
      style={{
        fontFamily: "var(--game-font)",
        fontSize: 12,
        marginTop: 4,
        color: c.is_best ? "oklch(67% 0.13 265)" : "var(--muted)",
        fontWeight: c.is_best ? 600 : 400,
      }}
    >
      {c.is_best ? "★ " : ""}{e2e ? "Instinct" : "search"} {(c.win_prob * 100).toFixed(1)}%
    </div>
  );
}
