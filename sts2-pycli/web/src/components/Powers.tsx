import { useLayoutEffect, useRef, useState } from "react";
import type { Power } from "../api";
import { useInspect } from "../inspect/useInspect";
import { powerItem } from "../inspect/powerItem";
import { useStageZoom } from "../phone";
import { DEFAULT_POWER_BOX, POWER_GAP, powerColumns, powerFit } from "../powerRow";
import { splitPowers } from "../powerOverflow";
import { COLUMN_MIN, COLUMN_WANT, bestColumnPlace, columnBottom, columnRoom, type Rect } from "../powerColumn";
import { inspectTrigger } from "../inspect/trigger";

/** Every landscape column re-places itself when the stage's DOM changes. A
 *  single change notifies every column's observer at once, so they queue here
 *  and all run in ONE animation frame, each once, instead of each column
 *  measuring the whole stage on every mutation. */
const queuedFits = new Set<() => void>();
let fitFrame = 0;
function queueFit(fit: () => void): void {
  queuedFits.add(fit);
  if (fitFrame) return;
  fitFrame = requestAnimationFrame(() => {
    fitFrame = 0;
    const fits = Array.from(queuedFits);
    queuedFits.clear();
    fits.forEach((f) => f());
  });
}

/** Row(s) of buff/debuff icons with stack counts; falls back to a labelled chip
 *  when the power has no art.
 *
 *  `boxWidth` is the creature's `Visuals.Bounds` width (plus the game's 25px
 *  margin) in the same px this row is drawn in — `NPowerContainer
 *  .SetCreatureBounds`. It decides only how WIDE a row may run; the two-row cap
 *  above is what keeps the block off the hand strip below.
 *
 *  Phone: `hitSize` (design px) makes each cell a tap box bigger than its icon,
 *  and `maxShown` (2 x the columns that fit `boxWidth`) caps the block at two
 *  rows, so it never grows a third. It runs wider than the creature's box only
 *  when the fit was clamped up to the 2-column minimum (a very narrow
 *  creature). What does not fit hides behind a "+N" chip that calls
 *  `onOverflow` with the hidden powers. */
export default function Powers({
  powers, size = 26, boxWidth = DEFAULT_POWER_BOX, hitSize, maxShown, minShown, cells, onOverflow, inspectScope, column,
}: {
  powers?: Power[]; size?: number; boxWidth?: number;
  hitSize?: number; maxShown?: number; onOverflow?: (hidden: Power[]) => void;
  /** The floor on the `maxShown` cap (`splitPowers`' own default, 4, when
   *  absent): the phone-portrait player band goes down to one cell. */
  minShown?: number;
  /** A fixed count of tap-box cells across, decided by the caller (the
   *  phone-portrait player band, whose width is its own; a split enemy's
   *  `rows` / `compact` chrome, from its solver slot). Skips the portrait
   *  column measuring (`colFit`) entirely. */
  cells?: number;
  /** Phone landscape: one column beside the creature (`.gs-powers-col`)
   *  instead of rows under it. The column measures its own room on the stage
   *  (`powerColumn.columnCells`) and ignores `maxShown`. */
  column?: boolean;
  /** Names the creature, so its inspect keys differ from another's. */
  inspectScope?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const k = useStageZoom();
  const inspect = useInspect();
  const phoneBlock = hitSize != null && !column && cells == null;
  const [colCells, setColCells] = useState(COLUMN_WANT);
  // Phone portrait: each creature owns a COLUMN of the stage, half-way to
  // each neighbour (the stage edge for the outermost). Its block takes as many
  // tap-box cells across as that column holds (`colFit`, at least one: one
  // power and the "+N" chip in two rows) and slides inside it — never the
  // creature — through `--gs-pshift` on its `.gs-powers` wrapper. With four or
  // five enemies a 2-cell-wide block under each used to run into the next
  // creature's block, HP bar and name (the owner: "the icons below them could
  // cause a mess").
  const [colFit, setColFit] = useState<number | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    const wrap = el?.parentElement;
    const me = el?.closest(".gs-combatant");
    const stage = el?.closest(".gs-arena-stage");
    if (!el || !wrap || !me || !stage || !phoneBlock) { setColFit(null); return; }
    const fit = () => {
      const st = stage.getBoundingClientRect();
      const centre = (n: Element) => { const b = n.getBoundingClientRect(); return (b.left + b.right) / 2; };
      const mine = centre(me);
      const all = Array.from(stage.querySelectorAll(".gs-combatant"));
      const xs = all.filter((n) => n !== me).map(centre);
      const rank = xs.filter((x) => x < mine).length;
      const leftN = Math.max(-Infinity, ...xs.filter((x) => x < mine));
      const rightN = Math.min(Infinity, ...xs.filter((x) => x >= mine));
      const AIR = 3; // real px between the block and a name above it
      const EDGE = 1; // real px kept free on each side of the column
      const lo = (Number.isFinite(leftN) ? (leftN + mine) / 2 : st.left) + EDGE;
      const hi = (Number.isFinite(rightN) ? (rightN + mine) / 2 : st.right) - EDGE;
      const cellPx = (hitSize ?? size) * k;
      // 0: the column is narrower than ONE tap box (five enemies on a small
      // phone stand ~42px apart): the block is a single cell, and every second
      // creature's hangs one cell lower, so neighbouring cells never touch.
      const across = Math.max(0, Math.floor((hi - lo + POWER_GAP * k) / (cellPx + POWER_GAP * k) + 1e-6));
      setColFit((prev) => (prev === across ? prev : across));
      wrap.style.setProperty("--gs-pshift", "0px");
      wrap.style.setProperty("--gs-pdrop", "0px");
      const b = el.getBoundingClientRect();
      // Slide into the column (a block wider than it centres on it).
      const over = b.right - b.left > hi - lo ? (lo + hi) / 2 - (b.left + b.right) / 2
        : Math.max(0, lo - b.left) - Math.max(0, b.right - hi);
      wrap.style.setProperty("--gs-pshift", `${over / k}px`);
      // A crowded row staggers (every second creature stands higher), and a
      // neighbour's HP bar or name can reach into this column at the height
      // of this block: drop the block just under it.
      const left = b.left + over, right = b.right + over;
      let drop = 0;
      stage.querySelectorAll(".gs-combatant").forEach((n) => {
        if (n === me) return;
        n.querySelectorAll(":scope > .gs-hpwrap, :scope > .gs-nameplate").forEach((o) => {
          const r = o.getBoundingClientRect();
          if (r.width <= 0 || r.right <= left + 0.5 || r.left >= right - 0.5) return;
          if (r.bottom <= b.top + 0.5 || r.top >= b.bottom - 0.5) return;
          drop = Math.max(drop, r.bottom + AIR - b.top);
        });
      });
      if (across === 0) {
        // One band of single cells, two levels deep, under every name on the
        // row: alternate creatures take alternate levels.
        let names = -Infinity;
        all.forEach((n) => n.querySelectorAll(":scope > .gs-hpwrap, :scope > .gs-nameplate").forEach((o) => {
          const r = o.getBoundingClientRect();
          if (r.width > 0) names = Math.max(names, r.bottom);
        }));
        drop = Math.max(0, names + AIR + (rank % 2) * (cellPx + POWER_GAP * k) - b.top);
      }
      wrap.style.setProperty("--gs-pdrop", `${drop / k}px`);
    };
    fit();
    const ro = new ResizeObserver(fit);
    // The row's x position moves when any fighter's box resizes (rigs load
    // after mount), so watch the stage and every side, not just this block.
    ro.observe(stage);
    stage.querySelectorAll(".gs-side").forEach((n) => ro.observe(n));
    ro.observe(el);
    return () => {
      ro.disconnect(); wrap.style.removeProperty("--gs-pshift"); wrap.style.removeProperty("--gs-pdrop");
    };
  }, [phoneBlock, k, powers?.length, hitSize, size, maxShown]);
  // Landscape column: where it stands and how many cells it has room for.
  // It hugs the creature's BODY (the sprite box), not its HP bar, which on a
  // small creature is twice as wide: measured off the bar, a crowded row put
  // the column on top of the neighbouring creature. Its bottom is the ground
  // line, or the top of the block badge where that hangs over it, so it never
  // covers the HP bar or the badge. Neither the side nor the cell count moves
  // the edge the other is measured from (the bottom), so this cannot loop.
  useLayoutEffect(() => {
    const el = ref.current;
    const wrap = el?.parentElement;
    const me = el?.closest(".gs-combatant");
    const stage = el?.closest(".gs-arena-stage");
    if (!column || !el || !wrap || !me || !stage) return;
    const cellPx = Math.max(size, hitSize ?? size);
    const AIR = 2; // real px between the column and what it stands beside
    const fit = () => {
      const st = stage.getBoundingClientRect();
      const box = me.getBoundingClientRect();
      const body = me.querySelector<HTMLElement>(":scope > .gs-sprite-wrap");
      const b = body?.getBoundingClientRect() ?? box;
      const badge = me.querySelector(".gs-block")?.getBoundingClientRect();
      const ground = Math.min(b.bottom, badge ? badge.top - AIR : Infinity);
      const w = cellPx * k;
      // A raised neighbour's HP bar and name (a staggered row) are floors the
      // column stands on rather than covers (`columnBottom`).
      const floors: Rect[] = [];
      stage.querySelectorAll(".gs-combatant").forEach((n) => {
        if (n === me) return;
        n.querySelectorAll(":scope > .gs-hpwrap, :scope > .gs-nameplate").forEach((o) => floors.push(o.getBoundingClientRect()));
      });
      // Last resort (`lifted`): the neighbours' intents become floors too.
      let lifted: Rect[] = [];
      const bottomAt = (left: number) => columnBottom(ground, left, left + w, AIR, [...floors, ...lifted]);
      // What it must not climb into: every creature's intent (its numeral
      // overhangs the icon's box, so that too), this one's own included, and
      // the energy orb.
      const ownEls = Array.from(me.querySelectorAll(":scope > .gs-sprite-wrap > .gs-intent, :scope > .gs-sprite-wrap > .gs-intent .gs-intent-dmg"));
      const own = ownEls.map((n) => n.getBoundingClientRect()).filter((r) => r.width > 0);
      const others: Rect[] = [];
      stage.querySelectorAll(".gs-intent, .gs-intent-dmg, .gs-energy").forEach((n) => {
        if (!ownEls.includes(n)) others.push(n.getBoundingClientRect());
      });
      const room = (left: number, obstacles: Rect[]) => (left < st.left + 2 || left + w > st.right - 2 ? -1 : columnRoom({
        bottom: bottomAt(left), cell: w, gap: POWER_GAP * k, ceiling: st.top + 4,
        left, right: left + w, obstacles,
      }));
      // Where it may stand, in order of preference: on the creature's left
      // (the outer side for the player, the intent's side for an enemy) clear
      // of its intent and the numeral hanging off it; then hugging the body
      // closer, for a crowded row where that clearance would put it under the
      // neighbour's intent; then its right side, when the left leaves the stage.
      const places = [
        { side: "left", x: Math.min(b.left, ...own.map((r) => r.left)) - AIR - w },
        { side: "left", x: Math.min(b.left, ...own.slice(0, 1).map((r) => r.left)) - AIR - w },
        { side: "left", x: b.left - AIR - w },
        { side: "right", x: Math.max(b.right, ...own.slice(0, 1).map((r) => r.right)) + AIR },
      ];
      let rooms = places.map((p) => room(p.x, [...others, ...own]));
      // A row too crowded for any of them (four enemies on a small phone):
      // give way to the NEIGHBOURS' intents and let the column stand against
      // its own, which is drawn over it (phone.css).
      if (Math.max(...rooms) < COLUMN_MIN) rooms = places.map((p) => room(p.x, others));
      // Five enemies on a small phone leave gaps between the intents narrower
      // than one tap box: the column stands ON the neighbour's intent instead
      // (above it, still at the creature's side) rather than cover it.
      if (Math.max(...rooms) < 1) {
        // (4px more air: an intent bobs.)
        lifted = others.map((r) => ({ left: r.left, right: r.right, top: r.top - 4, bottom: r.bottom }));
        rooms = places.map((p) => room(p.x, others));
      }
      const at = places[bestColumnPlace(rooms)];
      wrap.style.bottom = `${(box.bottom - bottomAt(at.x)) / k}px`;
      wrap.dataset.side = at.side;
      if (at.side === "left") { wrap.style.right = `${(box.right - (at.x + w)) / k}px`; wrap.style.left = "auto"; }
      else { wrap.style.left = `${(at.x - box.left) / k}px`; wrap.style.right = "auto"; }
      // Under one power and the chip only where nothing else fits clear of the
      // neighbours (five enemies on a small phone): one "+N" cell that lists
      // them all. With no room at all it stands on the floor of two.
      const best = rooms[places.indexOf(at)];
      const n = best === 1 ? 1 : Math.max(COLUMN_MIN, best);
      setColCells((prev) => (prev === n ? prev : n));
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(stage);
    // Intents mount once a rig has loaded, which resizes no side: watch them.
    stage.querySelectorAll(".gs-side, .gs-sprite-wrap").forEach((n) => ro.observe(n));
    // Mounts anywhere on the stage (an intent appears once its rig loads), and
    // an intent's TEXT changing (a new damage numeral is wider or narrower):
    // text changes elsewhere (damage numbers, the HP readout) are ignored.
    const mo = new MutationObserver((records) => {
      if (records.some((r) => r.type === "childList"
        || r.target.parentElement?.closest(".gs-intent"))) queueFit(fit);
    });
    mo.observe(stage, { childList: true, subtree: true, characterData: true });
    return () => {
      ro.disconnect(); mo.disconnect();
      queuedFits.delete(fit);
      delete wrap.dataset.side;
      wrap.style.removeProperty("bottom"); wrap.style.removeProperty("left"); wrap.style.removeProperty("right");
    };
  }, [column, k, powers?.length, hitSize, size]);
  if (!powers || powers.length === 0) return null;
  const cell = Math.max(size, hitSize ?? size);
  const { shown, hidden } = column
    ? splitPowers(powers, colCells, Math.min(colCells, COLUMN_MIN))
    : cells != null
      ? splitPowers(powers, maxShown ?? cells, minShown)
      : maxShown == null
      ? { shown: powers, hidden: [] as Power[] }
      : colFit === 0
        ? splitPowers(powers, 1, 1)
        : colFit != null
          ? splitPowers(powers, 2 * Math.min(colFit, maxShown / 2), 2)
          : splitPowers(powers, maxShown, minShown);
  const drawn = shown.length + (hidden.length ? 1 : 0);
  const cols = column ? 1 : cells != null ? Math.max(1, Math.min(drawn, cells)) : maxShown == null
    ? Math.min(powers.length, powerColumns(powers.length, boxWidth, size))
    : Math.min(drawn, colFit != null ? Math.max(1, Math.min(colFit, maxShown / 2)) : powerFit(boxWidth, cell));
  return (
    // A grid, not a wrapping flex row: the column count is the game's answer
    // above, not whatever happens to fit the box CSS gives this element (which
    // is how a narrow creature ended up with a three-row stack — see
    // `.gs-powers`, which now sizes to its content).
    <div ref={ref} style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
      gap: POWER_GAP, justifyContent: "center", justifyItems: "center" }}>
      {shown.map((p) => (
        <div key={p.name} className="gs-power"
          // Phone: a tap explains the power in the inspect sheet (`title` is
          // hover-only); a named button for the keyboard and screen readers.
          {...inspectTrigger(inspect.enabled, `${p.name} ${p.display_amount ?? p.amount ?? ""}`.trim(),
            () => inspect.open(powerItem(p, inspectScope)))}
          title={`${p.name}${p.description ? ` — ${p.description}` : ""}`}
          style={{ position: "relative", display: "flex", alignItems: "center",
            ...(hitSize ? { justifyContent: "center", width: cell, height: cell } : {}) }}>
          {p.image ? (
            <img src={p.image} alt={p.name} style={{ width: size, height: size, display: "block" }} />
          ) : (
            <span className="gs-power-name" style={{ fontFamily: "var(--game-font)", fontSize: 10, padding: "1px 6px", borderRadius: 10,
              border: "1px solid var(--border)", background: "var(--panel)" }}>
              {p.name}
            </span>
          )}
          {p.image && (
            <span className="gs-power-amt" style={{ fontFamily: "var(--game-font)", fontVariantNumeric: "tabular-nums",
              position: "absolute", right: hitSize ? (cell - size) / 2 - 3 : -3,
              bottom: hitSize ? (cell - size) / 2 - 3 : -3, fontSize: 10, fontWeight: 700,
              lineHeight: 1, padding: "1px 3px", borderRadius: 6, color: "var(--text)",
              background: "var(--bg)", border: "1px solid var(--text)",
              textShadow: "0 1px 2px oklch(0% 0 0 / 0.8)" }}>
              {p.display_amount ?? p.amount}
            </span>
          )}
        </div>
      ))}
      {hidden.length > 0 && (
        <button type="button" className="gs-power-more"
          aria-label={`${hidden.length} more powers`}
          // Phone only, and only while inspecting: with a card armed the tap
          // must bubble to the creature (it is the target), not be swallowed.
          onClick={inspect.enabled ? (e) => { e.stopPropagation(); onOverflow?.(hidden); } : undefined}
          style={{ width: cell, height: cell, fontFamily: "var(--game-font)", fontWeight: 700,
            fontSize: Math.max(12, size * 0.5), color: "var(--text)", background: "var(--panel)",
            border: "1px solid var(--border)", borderRadius: 10, padding: 0, cursor: "pointer" }}>
          +{hidden.length}
        </button>
      )}
    </div>
  );
}
