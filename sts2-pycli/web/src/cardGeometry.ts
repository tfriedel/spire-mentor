/** `scenes/cards/card.tscn` lays a card out as a fixed 300x422 stack of
 *  textures, so a card's height follows from its width and every position
 *  inside it is a percentage of those two numbers.
 *
 *  This lives outside `CardFrame.tsx` because callers that must RESERVE space
 *  for a card they are not rendering yet — the upgrade preview's empty box, the
 *  shop's card stand — need exactly the number the card will use, and a
 *  hand-copied `* 422 / 300` in each of them drifts. (It also keeps
 *  `CardFrame.tsx` exporting only its component, which is what Fast Refresh
 *  wants.) */
export const CARD_W = 300;
export const CARD_H = 422;

/** The rendered height of a card drawn at `width` px. */
/** How far the cost gem stands proud of the card's left edge, as a fraction
 *  of the card's width: `CardFrame`'s ENERGY_ICON starts at x = -166 on the
 *  300-wide card (half-width 150), 16 px out. */
export const CARD_GEM_PROUD = (166 - CARD_W / 2) / CARD_W;
/** The Regent's star cost stands further out, under the gem: `STAR_ICON`
 *  starts at x = -186, 36 px out. */
export const CARD_STAR_PROUD = (186 - CARD_W / 2) / CARD_W;
/** The furthest any cost icon stands proud of the card's left edge. */
export const CARD_COST_PROUD = Math.max(CARD_GEM_PROUD, CARD_STAR_PROUD);

export const cardBoxHeight = (width: number) => Math.round((width * CARD_H) / CARD_W);
