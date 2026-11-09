import { MarkBadge } from "../advice/AdviceContext";
import { markClass, markFor, useAdviceMarks } from "../advice/useAdviceMarks";
import type { Mark } from "../advice/marks";
import type { CSSProperties } from "react";
import type { Step, ChooseHandler, Card, Relic, Potion } from "../api";
import { assetUrl } from "../api";
import CardFrame from "./CardFrame";
import SpineSprite from "./SpineSprite";
import { cardBoxHeight } from "../cardGeometry";
import { DecisionAnalyzeButton, DecisionAnalyzeStatus, DecisionWinChip } from "./DecisionAdvisor";
import { useDecisionAdvisor } from "./useDecisionAdvisor";
import { useSound } from "../sound/useSound";
import { cueForRelic } from "../sound/runCues";
import "../skins/shop.css";

/** Deterministic 0..1 hash from small integers — the same mixer MapView uses
 *  for its hand-drawn node wobble. Every bit of scatter on this screen is
 *  seeded through it from stable indices (shelf slot + the engine's item
 *  index), so a replay capture renders pixel-identically every time. Never
 *  `Math.random()` at render. */
function hash01(a: number, b: number, salt: number): number {
  let h = (Math.imul(a + 37, 374761393) + Math.imul(b + 101, 668265263) + Math.imul(salt + 13, 69621)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return ((h >>> 0) % 4096) / 4096;
}
/** Symmetric jitter in [-amp, amp]. */
const jit = (a: number, b: number, salt: number, amp: number) =>
  (hash01(a, b, salt) - 0.5) * 2 * amp;

/** Per-ware scatter: items were set down on the rug by hand, so they sit a few
 *  pixels apart in both axes and lean a little. Same hash, different salts. */
function wareScatter(slot: number, index: number): CSSProperties {
  return {
    "--gs-tilt": `${jit(slot, index, 7, 1.4).toFixed(2)}deg`,
    "--gs-drop": `${jit(slot, index, 8, 4).toFixed(1)}px`,
    "--gs-nudge": `${jit(slot, index, 9, 5).toFixed(1)}px`,
  } as CSSProperties;
}

/** Dust drifting in the lamplight, as one `background-image` list of tiny
 *  radial-gradient dots. Two layers are rendered at different dot sizes and
 *  drift speeds (see `.gs-shop-motes` in shop.css): the parallax between them
 *  is the cue that reads as "there is air in this room". Positions come from
 *  the same hash mixer as the scatter, so a replay capture is stable. */
function moteField(layer: number, count: number): string {
  const dots: string[] = [];
  for (let i = 0; i < count; i++) {
    const x = hash01(layer, i, 21) * 100;
    const y = hash01(layer, i, 22) * 100;
    // the near layer's motes are bigger and brighter — that IS the parallax
    const r = (layer === 1 ? 3.4 : 1.9) + hash01(layer, i, 23) * (layer === 1 ? 4.6 : 2.4);
    const a = (layer === 1 ? 0.4 : 0.26) + hash01(layer, i, 24) * 0.36;
    dots.push(
      `radial-gradient(${r.toFixed(2)}px ${r.toFixed(2)}px at ${x.toFixed(2)}% ${y.toFixed(2)}%,` +
        ` rgba(255,222,166,${a.toFixed(2)}) 0 34%, rgba(255,200,132,0) 100%)`,
    );
  }
  return dots.join(",");
}
const MOTES_FAR = moteField(0, 26);
const MOTES_NEAR = moteField(1, 14);

/** Coin-glyph price tag (the game never writes a "g" suffix): the sprite-font
 *  coin + gold Kreon numerals on a small leather tag; numerals go red when the
 *  purse can't cover it, and CHARTREUSE when the card is the shop's discounted
 *  one. `NMerchantCard.UpdateVisual` picks exactly that order — red for
 *  `!EnoughGold` first, then `StsColors.green` (#7FFF00) for `IsOnSale`, then
 *  cream. `as="button"` makes the tag itself the buy control. */
function PriceTag({
  gold,
  affordable,
  onSale = false,
  onClick,
  disabled,
  ariaLabel,
}: {
  gold?: number;
  affordable: boolean;
  /** This slot is the merchant's half-price card. */
  onSale?: boolean;
  onClick?: () => void;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const cls = "gs-shop-pricetag"
    + (affordable ? (onSale ? " is-onsale" : "") : " is-unaffordable");
  const body = (
    <>
      <img src={assetUrl("ui/sprite_fonts/gold_icon.png")} alt="" aria-hidden="true" />
      <span className="gs-shop-pricenum">{gold ?? "?"}</span>
    </>
  );
  if (onClick || disabled != null) {
    return (
      <button className={cls} onClick={onClick} disabled={disabled} aria-label={ariaLabel}>
        {body}
      </button>
    );
  }
  return <span className={cls}>{body}</span>;
}

/** Merchant height in px. A constant, not a measurement: `fallbackHeight`
 *  sits in SpineSprite's load-effect deps, so feeding it a live measured value
 *  rebuilds the rig's WebGL context on every resize. Sized against the game's
 *  own scene, where the merchant rig draws at 0.470095 and the room rig behind
 *  him at 0.505 (`scenes/rooms/merchant_room.tscn`) — he is a little under a
 *  third of the room's height. */
const MERCHANT_H = 255;

/**
 * Shop screen: the merchant's tent as a scene — cards on a display row with
 * hanging price tags, relics and potions as physical items on the shop rug,
 * the card-removal medallion signboard and the hex-plate Leave button.
 * Interactive in Play (``onChoose``); read-only in Replay.
 */
export default function Shop({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  const s = step.state;
  const gold = s.player?.gold ?? 0;
  const cards: Card[] = s.cards ?? [];
  const relics: Relic[] = s.relics ?? [];
  const potions: Potion[] = s.potions ?? [];
  // Shop is rendered inside <SoundProvider> (via DecisionScreen, in Play),
  // so it can call useSound() directly — unlike App.tsx's own handlers,
  // which sit outside the provider it renders (see sound/SoundBridge.tsx).
  // `buy_relic` is the ONLY site in the whole workbench where the frontend
  // both knows it's acquiring a relic AND knows that relic's name: relics
  // gained from combat rewards, chests and events all resolve server-side
  // with no client-visible "which relic" moment, and RelicTray.tsx (named in
  // this task's brief) is a passive display with no activation click of its
  // own — see the Task 10 report for the full judgement call.
  const sound = useSound();

  // The run policy's marks (advice/marks.ts): on the ware, the card, the
  // removal service (the policy names the card to remove, so any removal pick
  // marks the service) and Leave.
  const { marks } = useAdviceMarks();
  const removalMark = [...marks.values()].find((m) => m.key.startsWith("select_cards|")) ?? null;
  const buy = (action: string, args: Record<string, unknown>, cost?: number, stocked?: boolean) => ({
    disabled: !onChoose || !stocked || (cost != null && cost > gold),
    onClick: () => onChoose?.(action, args),
    mark: markFor(marks, action, args),
  });

  const buyRelic = (relic: Relic) => {
    const b = buy("buy_relic", { relic_index: relic.index }, relic.cost, relic.is_stocked);
    return {
      ...b,
      onClick: () => {
        if (sound.map) {
          const cue = cueForRelic(sound.map, relic.name);
          if (cue) sound.emit([cue]);
        }
        b.onClick();
      },
    };
  };

  // EV advisor (live only): Monte-Carlo buying each item (or leaving) over the
  // rest of the act. Items key by ``card:<i>`` / ``relic:<i>`` / ``potion:<i>``
  // and ``leave``, matching the engine's per-shelf index spaces.
  const shopSig =
    `${gold}|${cards.map((c) => c.index).join(",")}|` +
    `${relics.map((r) => r.index).join(",")}|${potions.map((p) => p.index).join(",")}`;
  const adv = useDecisionAdvisor(shopSig, onChoose != null);

  const CARD_W = 165;
  const removalCost = s.card_removal_cost;
  const panelUrl = `url(${assetUrl("ui/common_ui/submenu_panel_short.png")})`;

  /** One relic/potion as a physical item on the rug: sprite (with contact
   *  shadow), hover card, price tag. The whole item is the buy button so its
   *  accessible name carries the item name, what it does, and the price.
   *
   *  The hover card is the name tag PLUS the item's effect text. A shop is the
   *  one screen where you weigh an item against your gold, and a ware on the
   *  rug is a ~50px sprite and a price — the game's own shopfront is no better
   *  (its hover tooltip reads "Energy Potion" and stops). The cards on the rack
   *  have carried their full text all along; the rug had the name and nothing
   *  else, so buying a relic meant already knowing what it did. */
  const wareItem = (
    kind: "relic" | "potion",
    name: string,
    image: string | null | undefined,
    description: string | undefined,
    cost: number | undefined,
    available: boolean | undefined,
    b: { disabled: boolean; onClick: () => void; mark?: Mark | null },
  ) => {
    const affordable = (cost ?? 0) <= gold;
    // "Sold" means the slot is EMPTY — someone bought it. Not being able to
    // afford something is a different fact, and the game says it with a red
    // price and nothing else (refs/game-shop-2.png: gold 30, six red tags, all
    // the wares at full brightness). `available` is absent on backends that do
    // not track bought slots, and absent must never read as sold.
    const soldOut = available === false;
    const cls =
      `gs-shop-item is-${kind}` +
      (soldOut ? " is-soldout" : "") +
      (!affordable ? " is-unaffordable" : "") +
      markClass(b.mark ?? null);
    return (
      <button
        className={cls}
        disabled={b.disabled}
        onClick={b.onClick}
        title={(description ? `${name} — ${description}` : name)
          + (soldOut ? " (sold out)" : !affordable ? " (too expensive)" : "")}
      >
        <MarkBadge mark={b.mark ?? null} />
        <span className="gs-shop-itemart" aria-hidden="true">
          {image ? (
            <img src={image} alt="" />
          ) : (
            <span className="gs-shop-noart">{name.charAt(0)}</span>
          )}
        </span>
        {/* Stays in the DOM (and the a11y tree — the reveal is `opacity`, which
            removes neither), so the button's accessible name is the whole
            pitch: "<name> <effect> <price>". */}
        <span className="gs-shop-itemtip">
          <span className="gs-shop-itemname">{name}</span>
          {description && <span className="gs-shop-itemdesc">{description}</span>}
        </span>
        <PriceTag gold={cost} affordable={affordable} />
      </button>
    );
  };

  return (
    <div
      className="gs-shop gs-scene"
      style={
        {
          "--gs-scene-bg": `url(${assetUrl("scene_bgs/merchant_room/bottom/shop_merchant_bottom.png")})`,
          "--gs-shop-panel": panelUrl,
        } as CSSProperties
      }
    >
      {/* Scene furniture under the vignette: tent-fabric edge dressing (the
          painted backdrop fades to pure black at both margins), lamplight
          pools, and the merchant himself. */}
      <div className="gs-shop-scenery" aria-hidden="true">
        <div className="gs-shop-drape is-left" />
        <div className="gs-shop-drape is-right" />
        {/* aerial perspective: the far half of the room sits behind a cool,
            contrast-killing haze, so the back wall stops sharing a plane with
            the floor the wares are standing on */}
        <div className="gs-shop-haze" />
        <div className="gs-shop-lightpool" />
        <div className="gs-shop-lanternglow" />
        {/* The merchant, posed by his own skeleton — the same rig the game
            puts in this room (`merchant_room.tscn`'s MerchantVisual), and the
            same treatment the treasure chest and every monster already get.
            He was four rotated crops of the rig's ATLAS PAGE before, which is
            not a picture of a merchant: it is a sheet of loose hoods, sleeves
            and arms that only becomes a figure once a skeleton poses them.
            The crops read as exactly that — a floating hood above a
            disembodied arm. */}
        <div className="gs-shop-merchant">
          <SpineSprite
            rigBase={assetUrl("room_rigs/shop_merchant_top/shop_merchant_top")}
            fallbackHeight={MERCHANT_H}
            alt=""
          />
        </div>
      </div>

      <div className="gs-shop-content">
        <header className="gs-shop-header">
          <div className="gs-shop-sign">
            <span className="gs-shop-signtitle">Merchant</span>
            <span className="gs-shop-purse" title="Your gold">
              <img src={assetUrl("ui/sprite_fonts/gold_icon.png")} alt="" aria-hidden="true" />
              {gold}
            </span>
          </div>
          {/* workbench tooling voice, kept OFF the game surfaces */}
          {adv.enabled && (
            <div className="gs-shop-tools">
              <div className="gs-shop-toolscrim">
                <DecisionAnalyzeButton adv={adv} />
              </div>
            </div>
          )}
        </header>
        <div className="gs-shop-toolstatus">
          <DecisionAnalyzeStatus adv={adv} />
        </div>

        {/* the card display row: real-size frames, slight scatter, price tags */}
        <div className="gs-shop-cardrow">
          {cards.map((c, i) => {
            const b = buy("buy_card", { card_index: c.index }, c.cost, c.is_stocked);
            // CardFrame shows `cost` as the energy badge; shop puts energy in card_cost.
            const framed = { ...c, cost: c.card_cost ?? 0 };
            const affordable = (c.cost ?? 0) <= gold;
            // See `wareItem`: sold out ≠ unaffordable.
            const cls =
              "gs-shop-cardstand" +
              (c.is_available === false ? " is-soldout" : "") +
              (!affordable ? " is-unaffordable" : "");
            // The rack arc + hand-laid scatter. r4's ±1.5°/±6px scatter was
            // invisible at this card width (a 1.5° tilt shifts a 165px edge by
            // ~2px), so the row still read as `display:flex`. The dominant
            // term is now the ARC: `off` runs -1..+1 across the rack, the ends
            // fan outward and sit lower, and every card gets its own depth
            // (scale + z), so no two cards share a top or a bottom edge.
            const mid = (cards.length - 1) / 2;
            const off = mid > 0 ? (i - mid) / mid : 0;
            const arc = Math.abs(off) ** 1.6;
            const tilt = off * 3.6 + jit(i, c.index, 1, 1.1);
            const drop = arc * 26 + jit(i, c.index, 2, 7);
            const nudge = jit(i, c.index, 3, 6);
            // cards at the ends of the rack lean away from the lamp, so they
            // are a touch smaller and painted behind their neighbours
            const scale = 1 - arc * 0.05;
            const depth = 12 - Math.round(arc * 9);
            // A card sitting lower casts a tighter, darker pool.
            const shadow = 0.42 + (drop + 6) / 26;
            return (
              <div
                key={c.index}
                className={cls}
                style={
                  {
                    "--gs-tilt": `${tilt.toFixed(2)}deg`,
                    "--gs-drop": `${drop.toFixed(1)}px`,
                    "--gs-nudge": `${nudge.toFixed(1)}px`,
                    "--gs-scale": scale.toFixed(3),
                    "--gs-z": depth,
                    "--gs-shadow": shadow.toFixed(2),
                    "--gs-cardh": `${cardBoxHeight(CARD_W)}px`,
                  } as CSSProperties
                }
              >
                <span className="gs-shop-cardshadow" aria-hidden="true" />
                <CardFrame card={framed} width={CARD_W} onClick={b.disabled ? undefined : b.onClick}
                  mark={b.mark?.tier} step={b.mark?.order} />
                {/* The discounted card's "%" tag, pinned where merchant_card.tscn
                    pins its SaleVisual: a Sprite2D at (146.3, -153.2) on the
                    300x422 card, i.e. centred on 98.8% / 13.7% of the frame. */}
                {c.on_sale && (
                  <img className="gs-shop-cardsale"
                    src={assetUrl("rooms/merchant_room/shop_sales_tag.png")}
                    alt="" aria-hidden="true" />
                )}
                <PriceTag
                  gold={c.cost}
                  affordable={affordable}
                  onSale={!!c.on_sale}
                  disabled={b.disabled}
                  onClick={b.onClick}
                  ariaLabel={`Buy ${c.name} — ${c.cost ?? "?"} gold${c.on_sale ? " (on sale)" : ""}`}
                />
                {adv.byKey.has(`card:${c.index}`) && (
                  <div className="gs-shop-chip">
                    <DecisionWinChip c={adv.byKey.get(`card:${c.index}`)!} adv={adv} />
                  </div>
                )}
              </div>
            );
          })}
          {cards.length === 0 && <span className="gs-shop-empty">Sold out</span>}
        </div>

        {/* the shop floor: wares rug on the left, services on the right */}
        <div className="gs-shop-floor">
          <div className="gs-shop-rugmat">
            <img
              className="gs-shop-rug"
              src={assetUrl("rooms/merchant_room/shop_rug.png")}
              alt=""
              aria-hidden="true"
            />
            <div className="gs-shop-wares">
              {relics.map((r, i) => {
                const b = buyRelic(r);
                const cand = adv.byKey.get(`relic:${r.index}`);
                return (
                  <div key={`r${r.index}`} className="gs-shop-itemwrap" style={wareScatter(i, r.index ?? i)}>
                    {wareItem("relic", r.name, r.image, r.description, r.cost, r.is_available, b)}
                    {cand && (
                      <div className="gs-shop-chip">
                        <DecisionWinChip c={cand} adv={adv} />
                      </div>
                    )}
                  </div>
                );
              })}
              {potions.map((p, i) => {
                const b = buy("buy_potion", { potion_index: p.index }, p.cost, p.is_stocked);
                const cand = adv.byKey.get(`potion:${p.index}`);
                return (
                  <div
                    key={`p${p.index}`}
                    className="gs-shop-itemwrap"
                    style={wareScatter(relics.length + i, p.index)}
                  >
                    {wareItem("potion", p.name, p.image, p.description, p.cost, p.is_available, b)}
                    {cand && (
                      <div className="gs-shop-chip">
                        <DecisionWinChip c={cand} adv={adv} />
                      </div>
                    )}
                  </div>
                );
              })}
              {relics.length === 0 && potions.length === 0 && (
                <span className="gs-shop-empty">Nothing left on the rug</span>
              )}
            </div>
          </div>

          <div className="gs-shop-service">
            {removalCost != null ? (
              <button
                className={
                  "gs-shop-removal" + (removalCost > gold ? " is-unaffordable" : "") + markClass(removalMark)
                }
                disabled={!onChoose || removalCost > gold}
                onClick={() => onChoose?.("remove_card", {})}
                title="Pay the merchant to remove a card from your deck"
              >
                <img
                  className="gs-shop-removal-medal"
                  src={assetUrl("rooms/merchant_room/card_removal_00.png")}
                  alt=""
                  aria-hidden="true"
                />
                <span className="gs-shop-removal-plaque">
                  <span className="gs-shop-removal-label">Remove a card</span>
                  <PriceTag gold={removalCost} affordable={removalCost <= gold} />
                </span>
                <img
                  className="gs-shop-removal-post"
                  src={assetUrl("rooms/merchant_room/card_removal_02.png")}
                  alt=""
                  aria-hidden="true"
                />
              </button>
            ) : (
              <div className="gs-shop-removal is-spent" title="No card removal available">
                <img
                  className="gs-shop-removal-medal"
                  src={assetUrl("rooms/merchant_room/card_removal_04.png")}
                  alt=""
                  aria-hidden="true"
                />
                <span className="gs-shop-removal-plaque">
                  <span className="gs-shop-removal-label">No removals</span>
                </span>
                <img
                  className="gs-shop-removal-post"
                  src={assetUrl("rooms/merchant_room/card_removal_02.png")}
                  alt=""
                  aria-hidden="true"
                />
              </div>
            )}

            <button className={`gs-shop-hexbtn${markClass(markFor(marks, "leave_room"))}`} disabled={!onChoose}
              onClick={() => onChoose?.("leave_room", {})}>
              <span className="gs-shop-hexface">Leave shop</span>
            </button>
            {adv.byKey.has("leave") && (
              <div className="gs-shop-chip">
                <DecisionWinChip c={adv.byKey.get("leave")!} adv={adv} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Air, then grade, then the near plane — everything from here down is
          painted OVER the props, which is what makes them sit IN the tent
          instead of being pasted on it at full brightness. */}
      <div className="gs-shop-motes" aria-hidden="true">
        <i className="is-far" style={{ backgroundImage: MOTES_FAR }} />
        <i className="is-near" style={{ backgroundImage: MOTES_NEAR }} />
      </div>
      <div className="gs-shop-grade" aria-hidden="true" />
      <div className="gs-shop-warm" aria-hidden="true" />
      <div className="gs-shop-grain" aria-hidden="true" />
      {/* the near plane: unlit foreground the room is seen PAST — the tent
          poles at both margins, sacking heaped along the bottom edge and the
          valance hanging into the top of frame. Nothing here is lit, which is
          exactly the point: it is the only thing in the frame that is nearer
          than the wares. */}
      <div className="gs-shop-fg" aria-hidden="true" />
    </div>
  );
}
