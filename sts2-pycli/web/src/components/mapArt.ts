// Map node glyphs and boss art lookup — kept out of the component file so it
// exports only the component (react-refresh/only-export-components).
import { assetUrl, type BossNode } from "../api";
import { DEFAULT_GAME_VERSION } from "../gameVersion";

/** Fallback glyphs/colors for nodes without server-resolved stamp art —
 *  muted dyed-ink tones that sit on parchment, not neon accents. */
const NODE: Record<string, { glyph: string; color: string }> = {
  Monster: { glyph: "⚔", color: "#6b5a41" },
  Elite: { glyph: "☠", color: "#7f4059" },
  MegaElite: { glyph: "☠", color: "#6e3550" },
  Boss: { glyph: "♛", color: "#3f3020" },
  Shop: { glyph: "$", color: "#48663f" },
  Rest: { glyph: "🔥", color: "#94402f" },
  RestSite: { glyph: "🔥", color: "#94402f" },
  Campfire: { glyph: "🔥", color: "#94402f" },
  Treasure: { glyph: "◆", color: "#45607a" },
  Event: { glyph: "?", color: "#8a6a2f" },
  Ancient: { glyph: "✦", color: "#5b4877" },
};
const fallback = { glyph: "?", color: "#6b5a41" };
export const nodeMeta = (t: string | null) => (t && NODE[t]) || fallback;

/** slug matching the extracted-art naming (mirrors assets.py slugify — but
 *  NOT its leading-"the" strip, which only some of the art trees apply). */
const slugify = (s: string) =>
  s.toLowerCase().replace(/['’]/g, "").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

/** The other half of `assets.py slugify`: the article dropped. `run_history/`
 *  and `boss_icons/` keep it (`the_kin_boss_icon.png`, and `room_icon()`
 *  resolves those straight off `boss_id.lower()`), while `monsters/` is
 *  generated WITH the strip — so `monsters/the_kin.png` does not exist and
 *  `monsters/kin.png` does. `AssetResolver.monster()` tries both forms; the
 *  boss candidate chain has to as well or its last resort is dead for exactly
 *  the bosses whose name starts with an article. */
const slugifyStripped = (s: string) => slugify(s).replace(/^the_/, "");

/** The candidate art for one boss's map silhouette, most faithful first.
 *
 * The game draws a boss node from that encounter's own map art — a Spine rig,
 * or the `images/map/placeholder/<id>_icon.png` pair for the bosses that still
 * ship a placeholder — tinted FLAT with the act's map ink (`NBossMapPoint.
 * _Ready`: `SelfModulate = _act.MapTraveledColor`). A flat ink blot is
 * therefore the right print, but only for art whose ALPHA carries the drawing.
 *
 * ORDER IS THE WHOLE POINT. `run_history/<id>.png` is the run-history BADGE —
 * for most bosses a hexagonal frame that is opaque edge to edge, with the
 * creature knocked into it in COLOUR. Flood its alpha and every one of those
 * bosses prints as the same featureless black lozenge; that is what the map
 * showed for the Queen (live report 2026-08-26). It used to sit second in this
 * chain, ahead of the creature illustration — whose alpha IS the creature, and
 * so inks into exactly the silhouette the game's own boss node draws. The
 * badge belongs last: a stand-in for the bosses we have no other art for (only
 * Aeonglass today, whose badge is line art and does ink legibly).
 *
 * `monsters/<slug>.png` is a lit colour render with a soft drop shadow baked
 * into its alpha — r4's "floating on a blurred ellipse" stag. The boss
 * filter's alpha threshold cuts that shadow, which is why it can lead here.
 */
export function bossArt(
  b: BossNode | null | undefined,
  nodeIcon?: string | null,
  gameVersion?: string | null,
) {
  const name = b?.name ?? b?.id ?? null;
  const slug = name ? slugify(name) : null;
  const stripped = name ? slugifyStripped(name) : null;
  // `boss_icons/` and `run_history/` are both keyed off the encounter id
  // (C# `ImageHelper.GetRoomIconSuffix` = `Id.Entry.ToLowerInvariant()`,
  // `WaterfallGiantBoss.BossNodePath` = `…/<id>_icon`), so the id form is the
  // exact one; the name-derived forms stay as a fallback for hosts that label
  // the node without identifying the encounter. The two spellings coincide for
  // every boss whose display name IS its id minus `_BOSS`, so dedupe — the
  // load-error walk should not retry a URL it just gave up on.
  const id = b?.id ? b.id.toLowerCase() : null;
  // A later build's new bosses (v0.113's Aeonglass) live only in its art
  // overlay (`images/<version>/…`, see `import_game.py` art_overlay), so a
  // non-default build tries its overlay first and the shared tree second —
  // without it the new boss falls through to the server's small map stamp.
  const art = (rel: string) => [
    ...(gameVersion && gameVersion !== DEFAULT_GAME_VERSION
      ? [assetUrl(`${gameVersion}/${rel}`)] : []),
    assetUrl(rel),
  ];
  const candidates = [
    ...(id ? art(`boss_icons/${id}_icon.png`) : []),
    ...(slug ? art(`boss_icons/${slug}_boss_icon.png`) : []),
    ...(slug ? art(`monsters/${slug}.png`) : []),
    ...(stripped && stripped !== slug ? art(`monsters/${stripped}.png`) : []),
    b?.icon ?? nodeIcon,
    id ? assetUrl(`run_history/${id}.png`) : null,
    slug ? assetUrl(`run_history/${slug}_boss.png`) : null,
  ];
  return {
    name: b?.name ?? null,
    candidates: candidates.filter(
      (x, i): x is string => !!x && candidates.indexOf(x) === i,
    ),
  };
}
