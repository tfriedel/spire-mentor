import { MarkBadge } from "../advice/AdviceContext";
import { markClass, markFor, useAdviceMarks } from "../advice/useAdviceMarks";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { Step, ChooseHandler, RestOption } from "../api";
import { assetUrl } from "../api";
import SpineSprite from "./SpineSprite";
import { characterArt, characterKey } from "../characters";
import "../skins/rest.css";

/** Per-region campfire scene layers under `rooms/rest_sites/<region>/`.
 *  File names are NOT uniform across regions (bg vs 00, l_log vs llog vs
 *  log_l), hence the explicit manifest. `flame` is the fire's tint for the
 *  CSS-drawn flame/glow (the game renders the flame as tinted particles; the
 *  extracted `fire_light` / `*_highlight` files are white alpha masks).
 *  Only overgrowth's acid-green is verified against game refs
 *  (docs/experiments/2026-08-22-workbench-visual-gauntlet/refs/game-rest-1.png);
 *  TODO: verify the other regions' flame hues when refs exist. */
const REST_SCENES: Record<
  string,
  { bg: string; fire: string; lLog: string; rLog: string; flame: string }
> = {
  overgrowth: {
    bg: "overgrowth_rest_site_bg.png",
    fire: "overgrowth_rest_site_fire.png",
    lLog: "overgrowth_rest_site_l_log.png",
    rLog: "overgrowth_rest_site_r_log.png",
    flame: "#8efa14",
  },
  hive: {
    bg: "hive_rest_site_00.png",
    fire: "hive_rest_site_fire.png",
    lLog: "hive_rest_site_llog.png",
    rLog: "hive_rest_site_rlog.png",
    flame: "#ffb14d",
  },
  glory: {
    bg: "glory_rest_site_00.png",
    fire: "glory_rest_site_fire.png",
    lLog: "glory_rest_site_l_log.png",
    rLog: "glory_rest_site_r_log.png",
    flame: "#ffd76b",
  },
  underdocks: {
    bg: "underdocks_rest_site_bg.png",
    fire: "underdocks_rest_site_fire.png",
    lLog: "underdocks_rest_site_log_l.png",
    rLog: "underdocks_rest_site_log_r.png",
    flame: "#7fd8e8",
  },
};

/** Wire option_id → game art id. The C# game names the heal option "HEAL"
 *  (icon = option_heal.png, RestSiteOption.IconPath) but the Rust engine's
 *  RestSiteOption enum spells it "Rest"; every other id matches by lowercase
 *  (Smith/Dig/Lift/Toke/Mend/Cook/Hatch/Kindle/Clone). */
const OPTION_ART_ALIAS: Record<string, string> = { rest: "heal" };


/** The clip the seated hero loops. The real game does NOT use the combat rig
 *  here: NRestSiteRoom spawns an NRestSiteCharacter from
 *  CharacterModel.RestSiteAnimPath (`rest_site/characters/<id>_rest_site`), a
 *  separate seated rig. Only that rig's ATLAS PAGE was extracted
 *  (`scene_bgs/rest_site/<char>/restsite_<char>.png` — loose limbs, no
 *  `.atlas`/`.skel`), so it cannot be posed here. `relaxed_loop` is the
 *  closest real clip the shipped combat rig carries (CharacterModel.cs:27,
 *  `relaxedAnim`; every character rig defines it) — an off-duty stance, which
 *  the left log then occludes from the thigh down so it reads as seated. */
const REST_CLIP = "relaxed_loop";

/** Hero body height as a fraction of the scene height — SpineSprite draws the
 *  measured body at exactly this (no `worldScale` here, so it normalizes the
 *  rig to `fallbackHeight`), feet on the ground line. Measured off the game ref
 *  (hero 230px tall in a 1083px frame = 21%) and scaled by this skin's scene
 *  zoom (~1.4x the game's world scale — see rest.css). It used to be 0.36: the
 *  figure was drawn BEHIND the left log and faded out below it, so it was
 *  oversized to leave a ref-sized silhouette showing ABOVE the log. Now that
 *  the whole figure is drawn in front of the log, the ref's own ratio is the
 *  one to use. */
const HERO_H_FRAC = 0.3;

/** Pick the scene by act name, same keying as Board.tsx's arenaRegion()
 *  (loc titles match the asset dirs); unknown acts fall back to the act-1
 *  overgrowth clearing. */
function restScene(actName?: string | null): [string, (typeof REST_SCENES)[string]] {
  const n = (actName ?? "").trim().toLowerCase();
  return n in REST_SCENES ? [n, REST_SCENES[n]] : ["overgrowth", REST_SCENES.overgrowth];
}

/** Campfire / rest-site screen: the game's firelit clearing with the real
 *  illustrated option plaques (`ui/rest_site/option_<id>.png` — the exact
 *  path scheme RestSiteOption.IconPath uses). Interactive in Play; in Replay
 *  the recorded choice keeps the game's ragged gold frame (review keeper). */
export default function RestSite({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  // The run policy's marks on the options (advice/marks.ts).
  const { marks } = useAdviceMarks();
  const s = step.state;
  const options: RestOption[] = s.rest_options ?? [];
  const chosen = step.chosen_action as { args?: { option_index?: number } } | null;
  const chosenIdx = chosen?.args?.option_index;
  // Plaque art that 404'd (unknown/modded option id) → stone-plaque fallback.
  // Keyed by the ART ID, not by the option's index: DecisionScreen renders
  // <RestSite> without a key, so the instance survives a step swap, and index 2
  // at one rest site is a different option from index 2 at the next. A miss
  // recorded for one would have suppressed the other's perfectly good art.
  const [artFailed, setArtFailed] = useState<Record<string, boolean>>({});

  const [region, scene] = restScene(step.meta?.act_name);
  const layer = (file: string) => assetUrl(`rooms/rest_sites/${region}/${file}`);

  // The Spine rig renders into a fixed-px box, so the seated hero has to be
  // sized from the measured scene height (nothing else keeps it in proportion
  // with the % -sized logs when the workbench pane is resized).
  const sceneRef = useRef<HTMLDivElement>(null);
  const [sceneH, setSceneH] = useState(0);
  useEffect(() => {
    const el = sceneRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    setSceneH(el.getBoundingClientRect().height);
    const ro = new ResizeObserver((entries) => setSceneH(entries[0].contentRect.height));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const charKey = characterKey(s.player?.name);
  // Quantised, NOT the raw measurement. `fallbackHeight` sits in SpineSprite's
  // load-effect dependency array, and that effect constructs a fresh
  // `spine.SpineCanvas` (a new WebGL context) and re-decodes the skeleton and
  // atlas. Feeding it a live per-pixel ResizeObserver value tore the rig down
  // and rebuilt it dozens of times a second while the pane splitter was
  // dragged, and browsers cap concurrent WebGL contexts — the hero blanked
  // out. Every other caller passes a constant. A 16px step is far below what
  // the eye reads as a size change on a ~300px figure.
  const heroH = Math.round((sceneH * HERO_H_FRAC) / 16) * 16;

  return (
    <div
      ref={sceneRef}
      className="gs-rest gs-scene"
      role="group"
      aria-label="Rest site"
      style={
        {
          "--gs-scene-bg": `url(${layer(scene.bg)})`,
          "--gs-scene-pos": "center 68%",
          "--rest-flame": scene.flame,
        } as CSSProperties
      }
    >
      {/* flat scene sprites, back to front: logs flanking the fire pit, then the
          hero at the left log (drawn AFTER it, so the figure sits IN FRONT of
          it — see .gs-rest-hero), then the fire on top */}
      <div className="gs-rest-scenery" aria-hidden="true">
        <div className="gs-rest-lightpool" />
        <img className="gs-rest-log gs-rest-log-l" src={layer(scene.lLog)} alt="" />
        <img className="gs-rest-log gs-rest-log-r" src={layer(scene.rLog)} alt="" />
        {charKey && heroH > 60 && (
          <div className="gs-rest-hero">
            <SpineSprite
              rigBase={assetUrl(characterArt(charKey, `character_rigs/${charKey}/${charKey}`))}
              animation={REST_CLIP}
              fallbackHeight={heroH}
              fallback={assetUrl(characterArt(charKey, `character_sprites/${charKey}.png`))}
              alt=""
            />
          </div>
        )}
        <div className="gs-rest-firepit">
          <img className="gs-rest-firepit-img" src={layer(scene.fire)} alt="" />
          <div className="gs-rest-fireglow" />
          <div className="gs-rest-smoke" />
          <div className="gs-rest-flame" />
          <div className="gs-rest-embers">
            <i /><i /><i />
          </div>
        </div>
      </div>

      <div className="gs-rest-content">
        <h2 className="gs-rest-prompt">What shall I do?</h2>
        <div className="gs-rest-options">
          {options.map((o) => {
            const locked = o.is_enabled === false;
            const disabled = locked || !onChoose;
            const taken = o.index === chosenIdx;
            const name = o.name ?? o.option_id ?? `Option ${o.index}`;
            const rawId = (o.option_id ?? "").toLowerCase();
            const id = OPTION_ART_ALIAS[rawId] ?? rawId;
            const art = id && !artFailed[id]
              ? assetUrl(`ui/rest_site/option_${id}.png`)
              : null;
            // locked (in-game unavailable) dims hard; replay-inert options stay
            // near-full color — the chosen one's gold frame is the differentiator
            const mark = markFor(marks, "choose_option", { option_index: o.index });
            const cls =
              "gs-rest-option" +
              (taken ? " is-chosen" : "") +
              (locked ? " is-locked" : disabled && !taken ? " is-dim" : "") +
              markClass(mark);
            return (
              <button
                key={o.index}
                type="button"
                className={cls}
                disabled={disabled}
                aria-label={name}
                title={taken ? `${name} — chosen in this run` : name}
                onClick={() => onChoose?.("choose_option", { option_index: o.index })}
              >
                <MarkBadge mark={mark} />
                <span className="gs-rest-plaque">
                  {/* The game's own selected-plaque silhouette (option_outline.png
                      is a white alpha mask of the plaque, 4px proud of the art),
                      masked over gold and drawn BEHIND the plaque — the ragged
                      welded frame the real screen uses, not a CSS ring. */}
                  {taken && art && (
                    <span
                      className="gs-rest-chosenframe"
                      style={
                        {
                          "--rest-outline": `url(${assetUrl("ui/rest_site/option_outline.png")})`,
                        } as CSSProperties
                      }
                    />
                  )}
                  {art ? (
                    <img
                      src={art}
                      alt=""
                      onError={() => setArtFailed((m) => ({ ...m, [id]: true }))}
                    />
                  ) : (
                    <span className="gs-rest-plaque-fallback">{name}</span>
                  )}
                  {taken && <span className="gs-rest-chosenseal" />}
                </span>
                {art && <span className="gs-rest-caption">{name}</span>}
              </button>
            );
          })}
          {options.length === 0 && <span className="gs-rest-empty">no rest options</span>}
        </div>
      </div>
    </div>
  );
}
