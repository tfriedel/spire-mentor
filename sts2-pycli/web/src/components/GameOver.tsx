import type { ReactNode } from "react";
import type { Step } from "../api";
import { assetUrl } from "../api";
import { characterArt, characterKey } from "../characters";
import "../skins/runend.css";

type Tone = "victory" | "defeat" | "ended";

type TallyRow = { icon: string; label: string; value: string; unit?: string };

/** One score-tally row: a genuine game_over_screen icon, a label, and the
 *  TRUE value right-aligned where the game right-aligns its points. Units are
 *  split off the numeral so they can be typeset down (the game keeps its
 *  numeral column clean). The game's point formula (+80 / ×1.4 …) is
 *  deliberately not imitated — the workbench shows only values it knows. */
function tallyRows(step: Step): TallyRow[] {
  const s = step.state;
  const meta = step.meta ?? {};
  const floor = s.floor ?? meta.floor;
  const gold = s.player?.gold;
  const deckSize = s.player?.deck_size ?? s.player?.deck?.length;
  const relicCount = s.player?.relics?.length;
  const rows: TallyRow[] = [];
  if (floor != null)
    rows.push({ icon: "ui/game_over_screen/score_floor.png", label: "Floors Climbed", value: String(floor) });
  if (gold != null)
    rows.push({ icon: "ui/game_over_screen/score_gold.png", label: "Gold", value: String(gold) });
  if (s.ascension != null)
    rows.push({ icon: "ui/game_over_screen/score_ascension.png", label: "Ascension", value: String(s.ascension) });
  if (deckSize != null)
    rows.push({ icon: "ui/game_over_screen/discovery_card.png", label: "Deck", value: String(deckSize), unit: "cards" });
  if (relicCount != null)
    rows.push({ icon: "ui/game_over_screen/discovery_relic.png", label: "Relics", value: String(relicCount) });
  return rows;
}

/** Shared run-end scene: warm near-black maroon room lit by one dim core,
 *  parchment ribbon banner, epitaph, score tally with real icons, the fallen
 *  hero bedded into the floor left of centre, and the primary action on a
 *  large parchment plate anchored bottom-right (the game's "Proceed!" beat).
 *  Used by both GameOver (live/replay) and RunEnded (post-"End run"). */
export function RunEndScene({ step, heading, tone, primaryAction, secondaryActions }: {
  step: Step;
  heading: string;
  tone: Tone;
  /** The one big bottom-right plate (New run). */
  primaryAction?: ReactNode;
  /** Modest plates parked under the tally (Review replay). */
  secondaryActions?: ReactNode;
}) {
  const s = step.state;
  const meta = step.meta ?? {};
  const floor = s.floor ?? meta.floor;
  const name = s.player?.name;
  const spriteKey = characterKey(name);
  const sprite = spriteKey ? assetUrl(characterArt(spriteKey, `character_sprites/${spriteKey}.png`)) : null;
  // Neutral, data-backed epitaph — character + act, no invented lore. The
  // floor number is deliberately NOT repeated here: it already lives in the
  // tally row and the metadata caption.
  const act = meta.act_name ?? (s.act != null ? `Act ${s.act}` : meta.act != null ? `Act ${meta.act}` : null);
  const where = act ? ` in ${act}` : "";
  const epitaph = name
    ? tone === "victory"
      ? `The ${name} triumphed${where}.`
      : tone === "defeat"
        ? `The ${name} fell${where}.`
        : `The ${name} abandoned the climb${where}.`
    : null;
  const rows = tallyRows(step);
  return (
    <div className="gs-scene gs-runend">
      <div className="gs-runend-frame">
        {/* lit floor plane, drifting smoke, film grain and the corner vignette */}
        <div className="gs-runend-field" aria-hidden="true" />
        {sprite && (
          <div className={`gs-runend-body gs-runend-body--${tone}`} aria-hidden="true">
            <img src={sprite} alt="" />
          </div>
        )}
        {/* run identity, parked in the frame's corner like the game's build
            stamp — small, warm, and clear of the body and the tally. */}
        <div className="gs-runend-meta">
          <span>{name}</span>
          {s.ascension != null && <span>· A{s.ascension}</span>}
          {meta.act_name && <span>· {meta.act_name}</span>}
          {floor != null && <span>· floor {floor}</span>}
          {s.player?.hp != null && <span>· {s.player.hp}/{s.player.max_hp} HP</span>}
        </div>
        <div className="gs-runend-banner">
          <img src={assetUrl("ui/reward_screen/reward_banner.png")} alt="" aria-hidden="true" />
          <h1 className={`gs-runend-title gs-runend-title--${tone}`}>{heading}</h1>
        </div>
        {epitaph && <div className="gs-runend-epitaph">{epitaph}</div>}
        {rows.length > 0 && (
          <div className="gs-runend-tally">
            {rows.map((r) => (
              <div key={r.label} className="gs-runend-row">
                <img src={assetUrl(r.icon)} alt="" aria-hidden="true" />
                <span className="gs-runend-rowlabel">{r.label}:</span>
                <span className="gs-runend-rowvalue">
                  {r.value}
                  {r.unit && <span className="gs-runend-rowunit">{r.unit}</span>}
                </span>
              </div>
            ))}
          </div>
        )}
        {secondaryActions && <div className="gs-runend-actions">{secondaryActions}</div>}
        {primaryAction && <div className="gs-runend-cta">{primaryAction}</div>}
      </div>
    </div>
  );
}

/** Victory / defeat screen with a short run summary.
 *
 *  ``onNewRun`` and ``onReviewReplay`` are shown in Play only. Replay is
 *  offered right here because this screen IS where a run ends: before, the only
 *  route to it was the toolbar's red "End run" (which lands on RunEnded), so a
 *  player who climbed to the top and read "Victory!" had no visible way back
 *  into their own run. */
export default function GameOver({ step, onNewRun, onReviewReplay, onCoachReview }: {
  step: Step;
  onNewRun?: () => void;
  onReviewReplay?: () => void;
  /** Open the Coach review (the autopsy, on a death). */
  onCoachReview?: () => void;
}) {
  const victory = step.state.victory;
  return (
    <RunEndScene
      step={step}
      heading={victory ? "Victory!" : "Defeat"}
      tone={victory ? "victory" : "defeat"}
      secondaryActions={(onReviewReplay || onCoachReview) && (
        <>
          {onReviewReplay && (
            <button className="gs-runend-btn gs-runend-btn--secondary" onClick={onReviewReplay}>
              <span className="gs-runend-btnface">Review replay</span>
            </button>
          )}
      {onCoachReview && (
        <button className="gs-runend-btn gs-runend-btn--secondary" onClick={onCoachReview}>
          <span className="gs-runend-btnface">Coach review</span>
        </button>
      )}
        </>
      )}
      primaryAction={onNewRun && (
        <button className="gs-runend-btn gs-runend-btn--primary" onClick={onNewRun}>
          <span className="gs-runend-btnface">New run</span>
        </button>
      )}
    />
  );
}
