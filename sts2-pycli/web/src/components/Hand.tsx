import { useLayoutEffect, useRef, useState } from "react";
import type { Card, Score } from "../api";
import { cardBoxHeight } from "../cardGeometry";
import { CARD_ASPECT, HAND_REST_MIN_F, HAND_REST_VISIBLE, fanLayout } from "../handFan";
import { fitScene } from "../scene";
import CardFrame from "./CardFrame";
import { useMediaQuery, TOUCH_QUERY } from "../useMediaQuery";
import { useAdviceMarks } from "../advice/useAdviceMarks";
import { cardKey } from "../advice/marks";

/** A held card's height as a fraction of the combat stage's height.
 *
 *  Measured off refs/game-combat-1.png: a card is ~215x300 against a ~1000px
 *  combat scene, and roughly its bottom half is cropped away by the screen
 *  edge. Our stage is a good deal shorter than the game's full-screen scene,
 *  so holding the game's raw 30% would shrink the fan back to the strip both
 *  r3 critics ranked the #1 tell ("the hand is tiny and fully contained");
 *  0.39 is what puts the fan at the game's apparent scale in our frame. */
const CARD_H_OF_STAGE = 0.39;

/** Horizontal room the in-scene chrome owns at each end of the hand strip: the
 *  End Turn plate (188px at `right: 12px`) on the right, the draw-pile plaque
 *  (66px at `left: 14px`) on the left. Reserved symmetrically so the fan stays
 *  centred. */
const CHROME_RESERVE = 200;

/** The hand's card width, the room the fan may spread across, and how much of
 *  a resting card the stage's bottom edge leaves visible — all tracked off the
 *  stage it is drawn into (fixed px reads right at one window size only).
 *  Falls back to sane constants until the stage is measured / when
 *  ResizeObserver is unavailable.
 *
 *  Measured in a LAYOUT effect off a plain ref, so the fallback width is never
 *  painted. The hand is the arena's flex sibling, so a card width that lands one
 *  frame late resizes the arena one frame late — and the arena's height is what
 *  Board fits oversized rigs to (`maxBodyH`). At 1600x900 the fallback is 180px
 *  against a measured 210, which shaved 40px off `maxBodyH` just after the board
 *  first painted: every boss too big for the arena visibly re-fitted and its
 *  column shifted under it. (A ref-callback into state, which is how the element
 *  used to arrive here, costs an extra render and lands the measure ~250ms in —
 *  well after the paint, which is why this has to be a ref.) */
function useHandMetrics(ref: React.RefObject<HTMLDivElement | null>, framing: number): { cw: number; availW: number; show: number } {
  const [m, setM] = useState({ cw: 180, availW: 900, show: Math.round(HAND_REST_VISIBLE) });
  useLayoutEffect(() => {
    // The ZOOM layer, not the stage: the fan is laid out inside it, so it has
    // to size itself against the layer's own (pre-transform) box. offsetWidth/
    // offsetHeight for the same reason — getBoundingClientRect would report the
    // box after `--gs-k` and the fan would shrink twice over.
    const stage = ref.current?.closest<HTMLElement>(".gs-stage-scale");
    if (!stage || typeof ResizeObserver === "undefined") return;
    // The hand's own size never feeds back into the stage's (the stage is a
    // flex:1 child of the board column), so observing it cannot loop.
    const measure = () => {
      const cw = Math.round(Math.max(112, Math.min(210, (stage.offsetHeight * CARD_H_OF_STAGE) / CARD_ASPECT)));
      // THE REST LINE IS THE GAME'S, in scene px through the same fit the
      // fighters are placed by (`fitScene`, measured off this same layer). The
      // strip used to crop a flat 36px instead, which is why the resting fan
      // reached far enough up the room to sit over the enemies' power rows —
      // see HAND_REST_VISIBLE.
      // The same framing Board fits the fighters with (stageZoom.ts): on a
      // tablet the scene is drawn larger than its fit, and the rest line —
      // scene px through `k` — has to move with it or the fan sits on the
      // enemies' power rows again.
      const k = fitScene(stage.offsetWidth, stage.offsetHeight, framing).k;
      const next = {
        cw,
        availW: Math.max(240, stage.offsetWidth - CHROME_RESERVE * 2),
        show: Math.round(Math.max(HAND_REST_VISIBLE * k, cardBoxHeight(cw) * HAND_REST_MIN_F)),
      };
      // Object identity, so bail out by value or every observer tick re-renders
      // the whole fan (the old single-number state got that for free).
      setM((prev) => (prev.cw === next.cw && prev.availW === next.availW && prev.show === next.show
        ? prev : next));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(stage);
    return () => ro.disconnect();
  }, [ref, framing]);
  return m;
}

export default function Hand({
  cards,
  playedIndex,
  onCardClick,
  scores,
  previewDamageFor,
  framing = 1,
}: {
  cards: Card[];
  playedIndex?: number;
  onCardClick?: (card: Card) => void;
  scores?: Score[];
  // Live modified per-hit damage to show on each attack card (target-aware).
  previewDamageFor?: (card: Card) => number | null;
  /** Board's compact framing (stageZoom.ts); 1 on a monitor. */
  framing?: number;
}) {
  const n = cards.length;
  const root = useRef<HTMLDivElement>(null);
  const { cw, availW, show } = useHandMetrics(root, framing);
  // Held by CARD index, not fan position: a card can be played out from under
  // the pointer, and a stale position would spread the fan around whichever
  // card slid into that slot.
  const [hoverId, setHoverId] = useState<number | null>(null);
  const hover = hoverId == null ? -1 : cards.findIndex((c) => c.index === hoverId);
  const slots = fanLayout(n, cw, availW, hover);
  // TWO-TAP PLAY ON A TOUCHSCREEN. The fan is deliberately cropped by the
  // stage's bottom edge — the game crops it too — and raising a card is what
  // lets you read it in full. A tablet has no hover, so the hand was a row of
  // half-cards you could only play blind. The raise is JS state rather than
  // CSS `:hover` (see the slot below), so a tap can drive it: first tap
  // raises, second plays. Unplayable cards raise too — reading them is the
  // whole point.
  const touch = useMediaQuery(TOUCH_QUERY);
  // The advice's marks (advice/marks.ts): which cards to play, numbered. The
  // engine's own win % rides along only with "engine numbers" on.
  const { marks, showNumbers } = useAdviceMarks();
  return (
    // `--gs-handmin` RESERVES the strip's height for the card boxes whether or
    // not any are in it. The hand is the arena's flex sibling and the arena is
    // `flex: 1`, so every pixel the strip gives up the arena takes — and the
    // combatants bottom-align in it. Playing the last card used to collapse the
    // strip from a card's height (157-295px) to the CSS floor, which dropped the
    // player and the enemies ~150px down the screen mid-turn (and re-fitted every
    // oversized rig with them, since `maxBodyH` is measured off the arena).
    <div className="gs-hand" ref={root}
      style={{ "--gs-handmin": `${cardBoxHeight(cw)}px`,
               "--gs-handshow": `${show}px` } as React.CSSProperties}>
      {cards.map((c, i) => {
        const sc = showNumbers ? scores?.find((s) => s.stable_idx === c.index) : undefined;
        const mark = marks.get(cardKey(c.index)) ?? null;
        // The plan's cards are pulled part-way out of the fan, as the mod
        // does (the game's own layout target, nudged): the move to make is
        // visible before you go looking for it.
        const pulled = mark != null && mark.order != null;
        const { overlap, dx, rot, drop } = slots[i];
        // Only a live hand has energy agency: there the game rims every
        // affordable card cyan and dims the rest. A replay frame has neither,
        // so its cards stay neutral-bright (see .gs-handslot.gs-playable).
        const clickable = !!onCardClick && c.can_play !== false;
        // Desktop plays on the single click it always had. On a touchscreen
        // the first tap only raises the card; the second commits it.
        const activate = () => {
          if (touch && i !== hover) { setHoverId(c.index); return; }
          if (touch) setHoverId(null);
          if (clickable) onCardClick!(c);
        };
        return (
          <div
            key={c.index}
            className={`gs-handslot${clickable ? " gs-playable" : ""}`
              + `${i === hover ? " gs-hovered" : ""}${pulled ? " gs-pulled" : ""}`}
            // Hover is React state rather than CSS `:hover` because the fan
            // reacts to it as a WHOLE (the neighbours step aside), and because
            // the state has to be read off the hitbox, not off the card that
            // moves. Enter/leave fire for the slot's whole subtree, so both the
            // hitbox and the raised card feed it.
            //
            // Except on a touchscreen: there both events fire as part of the
            // tap itself, so the card would rise and drop again inside one
            // gesture and you would never get to read it. The tap raises it.
            onPointerEnter={touch ? undefined : () => setHoverId(c.index)}
            onPointerLeave={touch ? undefined
              : () => setHoverId((h) => (h === c.index ? null : h))}
            onClick={touch || clickable ? activate : undefined}
            style={{
              marginLeft: -overlap,
              "--fan-x": `${dx.toFixed(1)}px`,
              "--fan-rot": `${rot.toFixed(2)}deg`,
              "--fan-y": `${drop.toFixed(1)}px`,
              "--fan-z": i,
            } as React.CSSProperties}
          >
            {/* The card itself only ever MOVES — it is never what the pointer
                is tested against (`pointer-events` come back to it once it is
                raised, to widen the hover, never to narrow it). The targeting
                arrow springs from the card, so `data-card-index` rides the
                moving box and not the hitbox. */}
            <div className="gs-handslot-lift" data-card-index={c.index}>
              <CardFrame
                card={c}
                width={cw}
                clickable={clickable}
                unaffordable={!!onCardClick && c.can_play === false
                  && c.unplayable_reason !== "blocked"}
                blocked={!!onCardClick && c.can_play === false
                  && c.unplayable_reason === "blocked"}
                played={c.index === playedIndex}
                hotkey={onCardClick && i < 9 ? i + 1 : undefined}
                winProb={sc?.win_prob}
                mark={mark?.tier}
                step={mark?.order}
                previewDamage={previewDamageFor?.(c)}
              />
            </div>
            <div className="gs-handslot-hit" aria-hidden="true" />
          </div>
        );
      })}
    </div>
  );
}
