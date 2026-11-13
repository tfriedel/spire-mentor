import { useLayoutEffect, useRef, useState } from "react";
import type { Card } from "../api";
import { useAdviceMarks } from "../advice/useAdviceMarks";
import { cardKey } from "../advice/marks";
import { useStageZoom } from "../phone";
import type { ZoomPlacement } from "../zoomPlacement";
import CardFrame from "./CardFrame";

/** The zoomed card's description in real px: the 13px text floor, with a
 *  little to spare for rounding. */
const ZOOM_DESC_PX = 13.5;

/** Card text as it reads aloud: markup tags (`[gold]…[/gold]`) dropped,
 *  line breaks as spaces. */
function plainText(desc: string): string {
  return desc.replace(/\[\/?[a-z_]+(=[^\]]*)?\]/gi, "").replace(/\s+/g, " ").trim();
}

/** PHONE: the selected hand card drawn large enough to read (spec §4 "Zoom").
 *
 *  Lives inside `.gs-stage-scale`, so it is laid out in design px: the
 *  placement (`zoomPlacement`, real px relative to the stage) is divided by the
 *  stage zoom `k`, which makes the card exactly `placement.width` real px wide.
 *  Drawn outside the fan, so the stage's bottom edge never crops it.
 *
 *  Tapping it is tapping the selected card again (`onTap`): it plays a card
 *  that needs no target, or an attack when there is one enemy. The fixed spot
 *  can cover an enemy, so while an attack waits for its target among several
 *  enemies (`awaitsTarget`) a tap on the card goes to the enemy drawn under
 *  that point, if there is one (`enemyUnder`); elsewhere on the card it does
 *  nothing, as before.
 *  In landscape it can hang over the hand strip; that part passes taps through
 *  (`.gs-zoomcard` is `pointer-events: none`, `.gs-zoomcard-hit` covers only
 *  the part above the strip), so a tap there reaches the slot underneath. */
/** The close button, in real px: a 44px tap target straddling the card's
 *  top-right corner by a quarter of its size. */
const CLOSE_PX = 44;
const CLOSE_OUT = 11;

export default function ZoomCard({ card, placement, onTap, onClose, wouldPlay = false, awaitsTarget = false, live = false, winProb, previewDamage }: {
  card: Card;
  placement: ZoomPlacement;
  onTap?: () => void;
  /** The "×": puts the card back (no play, no sound). */
  onClose?: () => void;
  /** A tap on it would play the card (handSelect.tapCard says so); it only
   *  changes the accessible name. */
  wouldPlay?: boolean;
  /** An armed attack with more than one enemy to choose from: a tap on the
   *  card is handed to the enemy under it. */
  awaitsTarget?: boolean;
  /** Live play: shows the unaffordable / blocked state like the hand does. */
  live?: boolean;
  winProb?: number;
  previewDamage?: number | null;
}) {
  const k = useStageZoom();
  // Whether the description overflows its box (and so scrolls): only then
  // does it take touches of its own.
  const frameRef = useRef<HTMLDivElement>(null);
  const [scrolls, setScrolls] = useState(false);
  useLayoutEffect(() => {
    const d = frameRef.current?.querySelector(".gs-card-desc");
    setScrolls(!!d && d.scrollHeight > d.clientHeight + 1);
  }, [card, placement.width, k]);
  const { marks } = useAdviceMarks();
  const mark = marks.get(cardKey(card.index)) ?? null;
  // The accessible name carries the card's text too, so the drawn card (the
  // same name and text again, split over a dozen layers) is hidden from the
  // accessibility tree rather than read twice.
  const text = plainText(card.description_rich ?? card.description ?? "");
  const label = `${wouldPlay ? `Play ${card.name}` : `${card.name}, zoomed`}${text ? `: ${text}` : ""}`;
  // Landscape: only the part above the hand strip takes taps (`placement.hit`);
  // the rest of the card is drawn but lets a tap through to the slot under
  // it, which selects that card and never plays this one.
  const hitH = placement.hit == null ? undefined : placement.hit / k;
  const tapAt = onTap ? (e: { clientX: number; clientY: number; currentTarget: Element }) => {
    const enemy = awaitsTarget ? enemyUnder(e.clientX, e.clientY, e.currentTarget) : null;
    if (enemy) enemy.click(); else onTap();
  } : undefined;
  return (
    <div className="gs-zoomcard" data-zoom-index={card.index} data-zoom-covers={placement.covers ? "1" : undefined}
      data-awaits-target={awaitsTarget ? "1" : undefined}
      style={{ left: placement.left / k, top: placement.top / k }}>
      <div className="gs-zoomcard-hit"
        role="button" tabIndex={0} aria-label={label}
        onClick={tapAt}
        onKeyDown={onTap ? (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onTap(); }
        } : undefined}
        style={{ position: "absolute", left: 0, top: 0, width: "100%", height: hitH ?? "100%" }} />
      <div aria-hidden="true" ref={frameRef}
        // A description long enough to scroll keeps its own touches (phone.css
        // `.gs-zoomcard-scroll`); a tap on it above the strip is still a tap
        // on the card.
        className={scrolls ? "gs-zoomcard-scroll" : undefined}
        onClick={tapAt ? (e) => {
          if (hitH == null || e.clientY < bandTop(e.currentTarget)) tapAt(e);
        } : undefined}>
      <CardFrame
        card={card}
        width={placement.width / k}
        clickable={live && card.can_play !== false}
        unaffordable={live && card.can_play === false && card.unplayable_reason !== "blocked"}
        blocked={live && card.can_play === false && card.unplayable_reason === "blocked"}
        winProb={winProb}
        mark={mark?.tier}
        step={mark?.order}
        previewDamage={previewDamage}
        descFloor={ZOOM_DESC_PX / k}
      />
      </div>
      {onClose && (
        <button type="button" className="gs-zoomcard-close" aria-label="Close"
          onClick={(e) => { e.stopPropagation(); onClose(); }}
          style={{ position: "absolute", width: CLOSE_PX / k, height: CLOSE_PX / k,
            top: -CLOSE_OUT / k, right: -CLOSE_OUT / k, fontSize: 26 / k }}>
          <span aria-hidden="true">×</span>
        </button>
      )}
    </div>
  );
}

/** The enemy drawn at this point of the glass, under the zoom holding `el`. */
function enemyUnder(x: number, y: number, el: Element): HTMLElement | null {
  const zoom = el.closest(".gs-zoomcard");
  for (const n of document.elementsFromPoint(x, y)) {
    if (zoom?.contains(n)) continue;
    const enemy = n.closest<HTMLElement>("[data-enemy-index]");
    if (enemy) return enemy;
  }
  return null;
}

/** The hand strip's top edge on the glass, for the stage holding `el`. */
function bandTop(el: Element): number {
  return el.closest(".gs-arena-stage")?.querySelector(".gs-hand")?.getBoundingClientRect().top ?? Infinity;
}
