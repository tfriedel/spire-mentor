import { Fragment, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { assetUrl, type Card } from "../api";
// card.tscn's card box, in the scene's own units, plus the width→height rule.
// Shared so callers that reserve space for a card get the same number it uses.
import { CARD_H, CARD_W, cardBoxHeight } from "../cardGeometry";

// ---------------------------------------------------------------------------
// This renders a card the way the game does, out of the game's own art.
//
// `scenes/cards/card.tscn` lays a card out as a 300x422 stack of textures, and
// the numbers below are that scene's offsets verbatim (its origin is the card
// *centre*, so x runs -150..150 and y -211..211). Everything is expressed as a
// percentage of the card box, so one `width` prop scales the whole thing.
//
// Three textures carry the colour coding players read at a glance, and they are
// NOT the same axis:
//   * the FRAME is tinted by the card's pool     — red Ironclad, green Silent, …
//   * the BANNER, PORTRAIT BORDER and TYPE PLAQUE are tinted by RARITY
// The game does that with an HSV shader at runtime; `extract_card_template.py`
// bakes one PNG per variant instead, so this file only picks a URL.
// ---------------------------------------------------------------------------


/** Convert a card.tscn rect (centre-origin) into CSS percentages. */
function box(left: number, top: number, right: number, bottom: number) {
  return {
    left: `${((left + CARD_W / 2) / CARD_W) * 100}%`,
    top: `${((top + CARD_H / 2) / CARD_H) * 100}%`,
    width: `${((right - left) / CARD_W) * 100}%`,
    height: `${((bottom - top) / CARD_H) * 100}%`,
  } as const;
}

// Node rects, straight from card.tscn.
const PORTRAIT = box(-125, -168, 125, 22);
const PORTRAIT_BORDER = box(-137.5, -164, 137.5, 46);
const DESCRIPTION = box(-122, 37, 121, 173);
const TITLE_BANNER = box(-163, -207, 164, -124);
const TITLE_LABEL = box(-105, -204, 105, -150);
const ENERGY_ICON = box(-166, -227, -102, -163);
// Regent star cost, tucked under the energy orb on the same overhanging corner.
const STAR_ICON = box(-186, -189, -128, -131);
const TYPE_PLAQUE_TOP = `${((1 + CARD_H / 2) / CARD_H) * 100}%`;
const TYPE_PLAQUE_H = `${(37 / CARD_H) * 100}%`;

/** A card.tscn font size, as a fraction of the card's width. */
const em = (size: number) => size / CARD_W;

// card.tscn's `theme_override_font_sizes`.
const TITLE_SIZE = em(26);
const TITLE_MIN_SIZE = em(15); // no in-game floor; keeps long names legible
const DESC_SIZE = em(21);
const DESC_MIN_SIZE = em(12); // MegaRichTextLabel MinFontSize
const TYPE_SIZE = em(16);
const ENERGY_SIZE = em(32);
const STAR_SIZE = em(22);

// The frame texture is chosen by card TYPE. CardModel.cs folds None/Status/Curse
// into Skill (and the portrait border folds Quest in too), so those cards wear
// the skill frame rather than a missing one.
const FRAME_TYPES = new Set(["attack", "skill", "power", "quest"]);
const BORDER_TYPES = new Set(["attack", "skill", "power"]);

// The banner/plaque/border tint is keyed by rarity. `common` is CardModel.cs's
// fallback arm, so it also covers None / Basic / Token.
const BANNER_RARITIES = new Set([
  "common", "uncommon", "rare", "curse", "status", "event", "quest", "ancient",
]);

// The title's outline is a THIRD rarity-keyed axis (NCard.GetTitleLabelOutlineColor
// -> StsColors.cardTitleOutline*), distinct from the banner tint: it is what
// keeps a cream title legible against each banner colour.
const TITLE_OUTLINE: Record<string, string> = {
  common: "#4D4B40", uncommon: "#005C75", rare: "#6B4B00", curse: "#550B9E",
  quest: "#7E3E15", status: "#4F522F", event: "#1B6131", ancient: "#4D4B40",
};

// Energy orb art is per character; every non-character pool uses the colorless
// orb. Keyed by the frame colour so it needs no extra field.
const ORB_BY_COLOR: Record<string, string> = {
  red: "ironclad", green: "silent", blue: "defect",
  pink: "necrobinder", orange: "regent", quest: "quest",
};

function frameType(type: string | undefined): string {
  const t = (type ?? "").toLowerCase();
  return FRAME_TYPES.has(t) ? t : "skill";
}

function borderType(type: string | undefined): string {
  const t = (type ?? "").toLowerCase();
  return BORDER_TYPES.has(t) ? t : "skill";
}

function bannerRarity(rarity: string | undefined): string {
  const r = (rarity ?? "").toLowerCase();
  return BANNER_RARITIES.has(r) ? r : "common";
}

/** The card's pool colour, which drives the frame tint.
 *
 * The backend sets `color` (it is the only side that knows a card's own
 * character). When it is missing — older payloads, fixtures, the gallery — fall
 * back to the atlas folder in the art URL, which is the same pool by
 * construction (`card_atlas/<pool>/<slug>.png`). */
const POOL_COLOR: Record<string, string> = {
  ironclad: "red", silent: "green", defect: "blue",
  necrobinder: "pink", regent: "orange", curse: "curse", quest: "quest",
};

function frameColor(card: Card): string {
  if (card.color) return card.color;
  const folder = card.image?.match(/card_atlas\/([^/]+)\//)?.[1];
  return (folder && POOL_COLOR[folder]) || "colorless";
}

// Shrink the description (from `max` down to `min`) just enough that it fits its
// fixed box without clipping — the same trick MegaRichTextLabel plays in-game
// (MaxFontSize 21 -> MinFontSize 12). Card size stays constant.
// `fitKey` retriggers the measure when the rendered content changes (it can be
// rich nodes, so it can't be derived from `children` directly).
function FitText({ children, fitKey, max, min, className, style, nowrap }: {
  children: ReactNode; fitKey: string; max: number; min: number;
  className?: string; style?: React.CSSProperties;
  /** Single-line fit (card titles): forbid wrapping and shrink on WIDTH too.
   *  The game's MegaLabel sizes titles with autowrap off against a
   *  width-OR-height overflow test (MegaLabelHelper.IsTooBig); height-only
   *  fitting lets a wide title wrap onto the portrait instead of shrinking. */
  nowrap?: boolean;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [size, setSize] = useState(max);
  useLayoutEffect(() => {
    const el = boxRef.current;
    const text = textRef.current;
    if (!el || !text) return;
    let s = max;
    el.style.fontSize = `${s}px`;
    // Measure the text against the box rather than the box against itself: the
    // box centres its content, so overflow escapes symmetrically and would not
    // show up in the box's own scrollHeight. The +1 width tolerance absorbs
    // sub-pixel rounding on wrapped (non-nowrap) text.
    const tooBig = () =>
      text.scrollHeight > el.clientHeight || text.scrollWidth > el.clientWidth + 1;
    while (s > min && tooBig()) {
      s = Math.round((s - 0.5) * 10) / 10;
      el.style.fontSize = `${s}px`;
    }
    setSize(s);
  }, [fitKey, max, min]);
  return (
    <div ref={boxRef} className={className} style={{ ...style, fontSize: size }}>
      <span ref={textRef} style={nowrap ? { whiteSpace: "nowrap" } : undefined}>{children}</span>
    </div>
  );
}

// The game marks its keywords up in the localization itself — "Apply 2
// [gold]Vulnerable[/gold]" — and draws the tagged span in StsColors.gold. Other
// colour tags exist ([red], [blue], …) and are handled the same way; an unknown
// tag falls through as plain text rather than showing its brackets.
// Colour tags, plus the cost-glyph runs the backend emits for the game's inline
// energy/star icons ("Gain [energy]3[/energy]" -> "Gain ●●●").
const MARKUP =
  /\[(\/?)(gold|red|green|blue|orange|purple|pink|aqua)\]|\[(energy|star)\](\d+)\[\/\3\]/g;
const TAG_COLOR: Record<string, string> = {
  gold: "#EFC851", red: "#FF5555", green: "#7FFF00", blue: "#87CEEB",
  orange: "#FFA518", purple: "#EE82EE", pink: "#FF78A0", aqua: "#2AEBBE",
};

/** Split tagged text into coloured spans and inline cost glyphs.
 *
 * `orb` is the energy-orb sprite for this card's pool, so an inline orb matches
 * the one drawn in its cost corner. Stars are pool-independent. */
function renderMarkup(text: string, orb: string): ReactNode[] {
  const out: ReactNode[] = [];
  let at = 0;
  let color: string | null = null;
  let m: RegExpExecArray | null;

  const push = (chunk: string) => {
    if (!chunk) return;
    out.push(color ? <span key={out.length} style={{ color }}>{chunk}</span> : chunk);
  };

  MARKUP.lastIndex = 0;
  while ((m = MARKUP.exec(text)) !== null) {
    push(text.slice(at, m.index));
    if (m[3]) {
      const src = m[3] === "star"
        ? assetUrl("card_template/energy_star.png")
        : assetUrl(`card_template/energy/${orb}.png`);
      for (let i = 0; i < Number(m[4]); i++) {
        out.push(
          <img key={out.length} className="gs-card-inline-icon" src={src}
            alt={m[3] === "star" ? "star" : "energy"} />
        );
      }
    } else {
      color = m[1] ? null : TAG_COLOR[m[2]];
    }
    at = m.index + m[0].length;
  }
  push(text.slice(at));
  return out;
}

// One printed number to overwrite with its live value: where it sits in the
// description, how many characters it spans, and what to put there.
type Swap = { at: number; len: number; base: number; value: number };

function swapAt(at: number, base: number, value: number): Swap | null {
  return at < 0 ? null : { at, len: String(base).length, base, value };
}

// The attack number, anchored on "<base> damage" so a coincidental Block value
// of the same size is left alone.
function damageSwap(desc: string, base?: number, preview?: number | null): Swap | null {
  if (base == null || preview == null || preview === base) return null;
  return swapAt(desc.indexOf(`${base} damage`), base, preview);
}

// The block number. Its keyword is marked up ("Gain 5 [gold]Block[/gold]"), so
// the anchor has to look PAST any tags between the digits and the word — the
// damage form's plain "<base> <word>" needle never matches here.
function blockSwap(desc: string, base?: number, preview?: number | null): Swap | null {
  if (base == null || preview == null || preview === base) return null;
  const m = new RegExp(String.raw`\b${base}\b(?=(?:\s|\[/?[a-z]+\])*Block)`).exec(desc);
  return swapAt(m ? m.index : -1, base, preview);
}

// Render a card description with each printed number replaced by the value the
// card would really produce right now — green when the live value beats the
// printed one, red when it falls short. Mirrors the game UI, which recolors
// {Damage} / {Block} on the card face. Falls back to the plain text whenever
// there is nothing to highlight.
//
// Both numbers are swapped, so Iron Wave ("Deal 5 damage. Gain 5 Block.")
// updates each half independently under Strength + Dexterity.
function renderDesc(
  description: string, orb: string, card: Card, previewDamage?: number | null
): ReactNode {
  const swaps = [
    damageSwap(description, card.stats?.damage, previewDamage),
    blockSwap(description, card.stats?.block, card.block_preview),
  ]
    .filter((s): s is Swap => s !== null)
    .sort((a, b) => a.at - b.at);
  if (swaps.length === 0) return renderMarkup(description, orb);

  const out: ReactNode[] = [];
  let cursor = 0;
  for (const s of swaps) {
    // Two anchors can only overlap on a malformed template; skip rather than
    // emit scrambled text.
    if (s.at < cursor) continue;
    out.push(
      <Fragment key={`t${s.at}`}>{renderMarkup(description.slice(cursor, s.at), orb)}</Fragment>
    );
    out.push(
      <span key={`v${s.at}`}
        style={{ color: s.value > s.base ? "var(--gs-buff)" : "var(--gs-nerf)", fontWeight: 700 }}>
        {s.value}
      </span>
    );
    cursor = s.at + s.len;
  }
  out.push(<Fragment key="tail">{renderMarkup(description.slice(cursor), orb)}</Fragment>);
  return <>{out}</>;
}

export default function CardFrame({ card, width = 112, height, dim = false, unaffordable = false, blocked = false, played = false, onClick, clickable = false, hotkey, winProb, best, mark, step, previewDamage }: {
  card: Card; width?: number; height?: number; dim?: boolean;
  /** In hand, and you cannot pay for it right now. The game does NOT fade the
   *  card for this: `NCard.UpdateEnergyCostVisuals` paints the cost numeral
   *  `StsColors.red` with the `unplayableEnergyCostOutline` and leaves the
   *  card alone, and the hand's cyan Highlight simply does not light. The red
   *  SLASH over the orb (`unplayable.png`) is a different state — C# shows it
   *  only when `!reason.HasResourceCostReason()`, i.e. a card that cannot be
   *  played at all rather than one you cannot currently afford. */
  unaffordable?: boolean;
  /** In hand, and something other than a resource cost forbids the play — the
   *  Queen's ChainsOfBinding once you have played a Bound card this turn, a
   *  smogged Skill, Ringing. `NCard.UpdateEnergyCostVisuals` shows the red
   *  `unplayable` slash over the orb for exactly this case
   *  (`Visible = !reason.HasResourceCostReason()`, NCard.cs:1019-1026) and
   *  leaves the cost numeral its normal colour: reddening it would claim you
   *  are short on energy when you are not. */
  blocked?: boolean;
  played?: boolean; onClick?: () => void; hotkey?: number; winProb?: number; best?: boolean;
  /** The advice's mark on this card (advice/marks.ts): a gold glow around its
   *  silhouette, breathing on the move to make. */
  mark?: "best" | "good" | "alt" | null;
  /** The mark's step number (1 = play now). */
  step?: number | null;
  /** Reads as clickable (cursor, rim) although the click is handled by an
   *  ancestor. The hand needs this: its cards sit under a hitbox that stays
   *  put while the card itself rises, so the hand's slot — not the card —
   *  carries the handler. */
  clickable?: boolean;
  // Live modified per-hit damage for the target currently being aimed at, when
  // this card is an attack and a target is determined; recolored on the card.
  previewDamage?: number | null;
}) {
  // Fixed portrait aspect ratio (card.tscn's 300x422) so cards never resize
  // with content length.
  const cardHeight = height ?? cardBoxHeight(width);
  const color = frameColor(card);
  const rarity = bannerRarity(card.rarity);
  const orb = ORB_BY_COLOR[color] ?? "colorless";
  // Prefer the keyword-marked text; older payloads and fixtures only have plain.
  // Both can be absent entirely (token cards, fixtures) — the frame still draws.
  const desc = card.description_rich ?? card.description ?? "";

  // Energy cost repainted when something MOVED it. `CardCostHelper` compares
  // the hook/local-modifier cost against the printed one and hands
  // `NCard.UpdateEnergyCostVisuals` `Decreased` -> `StsColors.green` or
  // `Increased` -> `StsColors.energyBlue`; an unchanged cost keeps cream. The
  // wire ships both numbers (`cost` effective, `base_cost` printed), so the
  // comparison is the same one. Unaffordable wins over both — that is C#'s own
  // order, `InsufficientResources` being the first branch it checks.
  const costShift =
    card.base_cost == null || unaffordable || card.cost == null
      ? ""
      : card.cost < card.base_cost
        ? " gs-cost-down"
        : card.cost > card.base_cost
          ? " gs-cost-up"
          : "";

  const cls = ["gs-card"];
  if (onClick || clickable) cls.push("gs-clickable");
  if (dim) cls.push("gs-dim");
  if (unaffordable) cls.push("gs-unaffordable");
  if (blocked) cls.push("gs-blocked");
  if (played) cls.push("gs-played");
  if (best) cls.push("gs-best");
  if (mark) cls.push(`gs-mark-${mark}`);

  return (
    <div
      data-played={played ? "true" : undefined}
      onClick={onClick}
      className={cls.join(" ")}
      // `--cw` is the card's width in px; every font size below is a fraction of
      // it, so the whole card scales from the single `width` prop.
      style={{ width, height: cardHeight, ["--cw" as string]: `${width}px` }}
    >
      {/* Illustration, under the frame — the frame's centre is transparent. */}
      <div className="gs-card-portrait" style={PORTRAIT}>
        {card.image
          ? <img src={card.image} alt={card.name} />
          : <div className="gs-noart">no art</div>}
      </div>

      {/* Frame: tinted by pool. Also paints the description panel behind the text. */}
      <img className="gs-card-layer" alt="" aria-hidden="true"
        src={assetUrl(`card_template/frames/${frameType(card.type)}_${color}.png`)} />

      <FitText fitKey={`${desc}|${card.stats?.damage ?? ""}|${previewDamage ?? ""}`
        + `|${card.stats?.block ?? ""}|${card.block_preview ?? ""}`}
        max={width * DESC_SIZE} min={width * DESC_MIN_SIZE}
        className="gs-card-desc" style={{ ...DESCRIPTION, whiteSpace: "pre-line" }}>
        {renderDesc(desc, orb, card, previewDamage)}
      </FitText>

      {/* Portrait border, title banner and type plaque: all tinted by rarity. */}
      <img className="gs-card-layer" alt="" aria-hidden="true" style={PORTRAIT_BORDER}
        src={assetUrl(`card_template/portrait_borders/${borderType(card.type)}_${rarity}.png`)} />
      {/* card.tscn gives TitleBanner `stretch_mode = 6` = KEEP_ASPECT_COVERED,
          the only layer on the card that is not a plain stretch: the banner
          sprite is 656x193 (a=3.40) against a 327x83 rect (a=3.94), so the game
          scales it to COVER the rect and clips the overflow top and bottom.
          Stretching it to fill instead squashed it ~14% vertically, which both
          thinned the cloth band and dropped it below the title — that is the
          "card name sits too low" tell. */}
      <img className="gs-card-layer gs-card-banner" alt="" aria-hidden="true" style={TITLE_BANNER}
        src={assetUrl(`card_template/banners/${rarity}.png`)} />

      {/* MegaLabel auto-sizes the title down for long names (MaxFontSize 26). */}
      <FitText fitKey={`${card.name}|${card.upgraded ?? false}`}
        max={width * TITLE_SIZE} min={width * TITLE_MIN_SIZE} nowrap
        className={`gs-card-title${card.upgraded ? " gs-upgraded" : ""}`}
        style={{ ...TITLE_LABEL, ["--gs-title-outline" as string]: TITLE_OUTLINE[rarity] }}>
        {card.name}{card.upgraded ? "+" : ""}
      </FitText>

      <div className="gs-card-plaque"
        style={{
          top: TYPE_PLAQUE_TOP, height: TYPE_PLAQUE_H,
          fontSize: `calc(var(--cw) * ${TYPE_SIZE})`,
          borderImageSource: `url(${assetUrl(`card_template/plaques/${rarity}.png`)})`,
        }}>
        {card.type}
      </div>

      <div className={`gs-card-energy${costShift}`} style={ENERGY_ICON}>
        <img src={assetUrl(`card_template/energy/${orb}.png`)} alt="" aria-hidden="true" />
        <span style={{ fontSize: `calc(var(--cw) * ${ENERGY_SIZE})` }}>{card.cost}</span>
      </div>

      {blocked && (
        <img className="gs-card-unplayable" style={ENERGY_ICON} alt="cannot be played"
          src={assetUrl("card_template/unplayable.png")} />
      )}

      {card.star_cost != null && card.star_cost !== 0 && (
        <div className="gs-card-energy gs-card-star" style={STAR_ICON}
          aria-label={`costs ${card.star_cost} stars`}>
          <img src={assetUrl("card_template/energy_star.png")} alt="" aria-hidden="true" />
          <span style={{ fontSize: `calc(var(--cw) * ${STAR_SIZE})` }}>{card.star_cost}</span>
        </div>
      )}

      {/* Workbench affordances — not part of the game's card. */}
      {played && <span className="gs-badge gs-played-tag">played</span>}
      {hotkey != null && <span aria-hidden="true" className="gs-badge gs-hotkey-tag">{hotkey}</span>}
      {step != null && (
        <span className={`gs-badge gs-step-tag gs-step-${mark ?? "best"}`} aria-label={`step ${step}`}>{step}</span>
      )}
      {winProb != null && (
        <span aria-hidden="true" className={`gs-badge gs-win${best ? " gs-best" : ""}`}>
          {Math.round(winProb * 100)}%
        </span>
      )}
    </div>
  );
}
