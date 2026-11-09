import { MarkBadge } from "../advice/AdviceContext";
import { markClass, markFor, useAdviceMarks } from "../advice/useAdviceMarks";
import { useState } from "react";
import type { CSSProperties } from "react";
import type { Step, ChooseHandler, EventOption } from "../api";
import { assetUrl } from "../api";
import SpineSprite from "./SpineSprite";
import RichText from "./RichText";
import { DecisionAnalyzeButton, DecisionAnalyzeStatus, DecisionWinChip } from "./DecisionAdvisor";
import { useDecisionAdvisor } from "./useDecisionAdvisor";
import "../skins/event.css";

// Title words that carry no subject: never the emphasis noun.
const TITLE_STOP = new Set([
  "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "from",
  "with", "your", "his", "her", "their", "full", "more", "less", "some", "that",
  "this", "what", "who", "into", "over", "under",
]);

/** The event's head noun — the last content word of its title.
 *
 *  The game bolds its subject noun everywhere it appears in the body copy
 *  ("This room is full of **cheese**!!!  …made of **cheese**."), but the wire
 *  delivers the description with all ``[b]`` markup already flattened, so the
 *  markup cannot be recovered. This is the honest display-only stand-in: bold
 *  only the noun the event is *named after*, never an invented keyword list.
 *  Returns the word plus its naive singular/plural partner, or ``[]`` when the
 *  title has no usable content word (then nothing is emphasized). */
function emphasisWords(title: string): string[] {
  const words = title.replace(/[^A-Za-z\s]/g, " ").split(/\s+/).filter(Boolean);
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i];
    if (w.length < 4 || TITLE_STOP.has(w.toLowerCase())) continue;
    return w.toLowerCase().endsWith("s") ? [w, w.slice(0, -1)] : [w, `${w}s`];
  }
  return [];
}

/** Chest height in px. The rig renders into a fixed-px box like every other
 *  SpineSprite caller; a constant (not a measurement) because `fallbackHeight`
 *  sits in that component's load-effect deps and a live value rebuilds its
 *  WebGL context on every resize. */
const TREASURE_RIG_H = 420;

/**
 * Event / Neow screen. The event illustration (when extracted) fills the
 * backdrop via the shared `.gs-scene` primitive; `skins/event.css` grades it
 * up to a lit room and anchors title/text/options in the game's right-column
 * composition. Options render on the game's own `event_button` plaque art
 * (nine-sliced at 1:1 so the angular corner cuts survive), with the outline
 * art as the thin cyan stroke. The Neow step has no per-event illustration —
 * it composes the real `scene_bgs/neow_room` layers instead (bg + head cropped
 * from the rig atlas + waterfalls + vignette) and carries the game's engraved
 * bottom-left name lockup instead of a title banner.
 * Interactive in Play (``onChoose`` provided); read-only in Replay.
 */
export default function EventScreen({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  // The run policy's marks on the options (advice/marks.ts).
  const { marks } = useAdviceMarks();
  const s = step.state;
  const illustration = s.art?.illustration ?? null;
  const options: EventOption[] = s.event_options ?? [];
  const chosen = step.chosen_action as { args?: { option_index?: number } } | null;
  const chosenIdx = chosen?.args?.option_index;
  const title = s.event_name ?? "Event";
  const isNeow = (s.event_name ?? "").trim().toLowerCase() === "neow";
  // A treasure room is a place, not a question. The game gives it a whole
  // scene with a chest you walk up to and open; ours arrived as a bare "Open"
  // bar on an empty background, which is why opening one felt like nothing
  // happened at all. The backend names the room (`_fill_generic`) so we can
  // tell it from every other one-button screen.
  const isTreasure = (s.event_name ?? "").trim().toLowerCase() === "treasure";
  // The chest room ships as a Spine rig per act (one clip, `animation`), the
  // same shape the rest-site hero and every monster use.
  const chestAct = Math.min(3, Math.max(1, step.meta?.act ?? 1));
  const openOption = options.find((o) => !o.is_locked) ?? options[0];
  // Bumped on click to fire the chest's one-shot open. A replay frame that
  // already carries the choice starts opened (see `opened` below).
  const [openNonce, setOpenNonce] = useState(0);
  // One continue-style option ("Proceed") gets the game's single wide bar.
  const solo = options.length === 1;
  // Wire descriptions carry real newlines between the game's text beats;
  // render them as paragraphs instead of one merged block.
  const paragraphs = (s.description ?? "").split(/\n+/).filter((p) => p.trim().length > 0);
  const emphasis = emphasisWords(title);

  // EV advisor (live only): Monte-Carlo each option over the rest of the act.
  // Options key by ``option:<index>`` to join the per-option win-rate chips.
  const adv = useDecisionAdvisor(
    `${s.event_name ?? ""}|${options.map((o) => o.index).join(",")}`,
    onChoose != null,
  );

  const sceneVars = {
    "--ev-btn": `url(${assetUrl("ui/common_ui/event_button.png")})`,
    "--ev-btn-glow": `url(${assetUrl("ui/common_ui/event_button_outline.png")})`,
    ...(illustration ? { "--gs-scene-bg": `url(${illustration})` } : {}),
    ...(isNeow
      ? {
          "--neow-bg": `url(${assetUrl("scene_bgs/neow_room/neow_bg.png")})`,
          "--neow-atlas": `url(${assetUrl("scene_bgs/neow_room/neow.png")})`,
          "--neow-fall-l": `url(${assetUrl("scene_bgs/neow_room/l_waterfall.png")})`,
          "--neow-fall-r": `url(${assetUrl("scene_bgs/neow_room/r_waterfalls.png")})`,
          "--neow-vig": `url(${assetUrl("scene_bgs/neow_room/neow_vignette.png")})`,
        }
      : {}),
  } as CSSProperties;

  if (isTreasure) {
    const opened = chosenIdx != null || openNonce > 0;
    return (
      <div className="gs-scene gs-event gs-treasure" role="group" aria-label="Treasure room">
        <div className="gs-treasure-rig" data-act={chestAct}>
          {/* The rig's one clip IS the opening, so it sits on its setup pose
              (closed) and plays once when you open it, holding the last frame.
              Looping it the normal way left the chest open before you touched
              it, on a screen whose whole point is opening it. */}
          <SpineSprite
            rigBase={assetUrl(`room_rigs/treasure_act_${chestAct}/treasure_act_${chestAct}`)}
            holdSetup
            actionAnim="animation"
            actionNonce={openNonce}
            holdLast
            fallbackHeight={TREASURE_RIG_H}
            fallback={assetUrl("run_history/treasure.png")}
            alt=""
          />
        </div>
        {/* The chest IS the button — that is the interaction the game has, and
            a separate bar under a picture of a chest is not it. */}
        <button
          className={`gs-treasure-hit${opened ? " is-opened" : ""}`}
          disabled={!onChoose || openOption == null}
          onClick={() => {
            if (!openOption) return;
            setOpenNonce((n) => n + 1);
            onChoose?.("choose_option", { option_index: openOption.index });
          }}
          aria-label={opened ? "Chest opened" : "Open the chest"}
        >
          <span className="gs-treasure-prompt">
            {opened ? "opened" : onChoose ? "Open" : "not opened in this run"}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className={`gs-scene gs-event${isNeow ? " gs-event--neow" : ""}`} style={sceneVars}>
      {isNeow && (
        <div className="gs-neow-scene" aria-hidden="true">
          <div className="gs-neow-bg" />
          <div className="gs-neow-head" />
          <div className="gs-neow-altar" />
          <div className="gs-neow-fall gs-neow-fall--l" />
          <div className="gs-neow-fall gs-neow-fall--r" />
          <div className="gs-neow-vignette" />
          <div className="gs-neow-glow" />
        </div>
      )}
      {/* Neow's name is an engraved plaque in the bottom-left corner of the
          cave, not a banner over the scene — the epithet is her fixed title in
          the game's own lockup, which the wire never sends. */}
      {isNeow && (
        <div className="gs-neow-lockup">
          <h2 className="gs-neow-name">{title}</h2>
          <div className="gs-neow-epithet">Mother of Resurrection (Exiled)</div>
        </div>
      )}
      <div className={`gs-event-col${solo ? " is-solo" : ""}`}>
        <div className="gs-event-header">
          {!isNeow && <h2 className="gs-event-title">{title}</h2>}
          <DecisionAnalyzeButton adv={adv} style={{ position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)" }} />
        </div>
        <DecisionAnalyzeStatus adv={adv} />
        {paragraphs.length > 0 && (
          <div className="gs-event-body">
            {paragraphs.map((p, i) => (
              <p key={i}><RichText text={p} emphasize={emphasis} /></p>
            ))}
          </div>
        )}
        <div className={`gs-event-options${solo ? " is-solo" : ""}`}>
          {options.map((o) => {
            const locked = o.is_locked;
            const taken = o.index === chosenIdx;
            const cand = adv.byKey.get(`option:${o.index}`);
            return (
              <div key={o.index}>
                <button
                  className={
                    "gs-event-option" + (taken ? " is-taken" : "") + (locked ? " is-locked" : "")
                    + markClass(markFor(marks, "choose_option", { option_index: o.index }))
                  }
                  disabled={locked || !onChoose}
                  onClick={() => onChoose?.("choose_option", { option_index: o.index })}
                >
                  <MarkBadge mark={markFor(marks, "choose_option", { option_index: o.index })} />
                  {o.image && (
                    <span className="gs-event-option-icon">
                      <img src={o.image} alt="" />
                    </span>
                  )}
                  <span className="gs-event-option-text">
                    <span className="gs-event-option-name">
                      <RichText text={o.title} />{locked && " 🔒"}
                    </span>
                    {o.description && (
                      <span className="gs-event-option-desc"><RichText text={o.description} /></span>
                    )}
                  </span>
                  {taken && <span className="gs-event-medallion" role="img" aria-label="chosen" />}
                </button>
                {cand && <DecisionWinChip c={cand} adv={adv} />}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
