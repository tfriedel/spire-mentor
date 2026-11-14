import { useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode, type SVGProps } from "react";
import type { MapRow, MapGraph, FullMap, FullMapNode, PathPlan, BossNode } from "../api";
import { assetUrl } from "../api";
import { markFor, useAdviceMarks } from "../advice/useAdviceMarks";
import { useHints } from "../hints/useHints";
import "../skins/map.css";
import { nodeMeta, bossArt } from "./mapArt";
import { usePhone } from "../phone";

const coordKey = (c: { col: number; row: number }) => `${c.col},${c.row}`;
/** Undirected edge key so the plan-route highlight matches drawn edges. */
const edgeKey = (a: { col: number; row: number }, b: { col: number; row: number }) =>
  `${coordKey(a)}|${coordKey(b)}`;

/** Deterministic 0..1 hash from small integers. All the hand-drawn wobble on
 *  the map (node placement, path dot jitter) is seeded from node coordinates
 *  through this, so a replay capture renders pixel-identically every time —
 *  never Math.random() at render. */
function hash01(a: number, b: number, salt: number): number {
  let h = (Math.imul(a + 37, 374761393) + Math.imul(b + 101, 668265263) + Math.imul(salt + 13, 69621)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) % 4096 / 4096;
}
/** Symmetric jitter in [-amp, amp]. */
const jit = (a: number, b: number, salt: number, amp: number) =>
  (hash01(a, b, salt) - 0.5) * 2 * amp;

/** Build the sets the optionality overlay needs: the planned-route edge keys
 *  (including current→first), and the per-column ``can_skip_elite`` flag keyed
 *  by the first-move node. The route marks nodes only through its edges — the
 *  nodes themselves carry no plan mark. */
function planOverlay(fm: FullMap, show: boolean) {
  const plan: PathPlan | null | undefined = show ? fm.path_plan : null;
  const edgeSet = new Set<string>();
  const skipCols = new Map<number, boolean>();
  if (!plan) return { plan: null, edgeSet, skipCols };
  const seq = plan.route;
  for (const col of plan.columns) skipCols.set(col.col, col.can_skip_elite);
  let prev = fm.current_coord ?? null;
  for (const c of seq) {
    if (prev) edgeSet.add(edgeKey(prev, c));
    prev = c;
  }
  return { plan, edgeSet, skipCols };
}
const meta = nodeMeta;

/** Client-side stamp fallback: the game's run_history room glyphs, keyed by
 *  room type, for nodes the server did not resolve an icon for (e.g. Unknown
 *  rooms). The game's "?" brush coin covers both Event and Unknown. */
const TYPE_STAMP: Record<string, string> = {
  Monster: "run_history/monster.png",
  Elite: "run_history/elite.png",
  MegaElite: "run_history/elite.png",
  Shop: "run_history/shop.png",
  Rest: "run_history/rest_site.png",
  RestSite: "run_history/rest_site.png",
  Campfire: "run_history/rest_site.png",
  Treasure: "run_history/treasure.png",
  Event: "run_history/event.png",
  Ancient: "run_history/ancient.png",
};
const stampFor = (t: string | null) =>
  t == null ? assetUrl("run_history/event.png")
  : TYPE_STAMP[t] ? assetUrl(TYPE_STAMP[t])
  : t === "Boss" ? null
  : assetUrl("run_history/event.png");

/** ------------------------------------------------------------------ ink lab
 *
 * The room sprites we ship (``run_history/*.png``) are the game's RUN-HISTORY
 * glyphs: two flat, fully saturated UI colours each (hot magenta elite, fire
 * engine red rest, #008C4A shop, #F8C22A "?"). The game's MAP prints the same
 * shapes in a muted paper palette — measured off refs/game-map-2.png:
 *
 *     paper (165,138,99) · enemy (105,85,60) · elite (119,86,82)
 *     black elite (66,52,33) · rest (133,88,68) · merchant (109,97,55)
 *     treasure (94,85,72) → (190,172,148) · unknown coin (86,61,7)
 *     the cream facial features of EVERY stamp: (176,158,132)
 *
 * So each stamp is a two-tone print: one type ink plus one shared cream. That
 * is a duotone, and a duotone is exactly `luminance → linear ramp per channel`,
 * i.e. one feColorMatrix + one feComponentTransfer. `s`/`i` below are the ramp
 * slope/intercept solved from each sprite's own two source luminances so the
 * dark tone lands on the measured map ink and the light tone on the cream.
 *
 * Round 4 instead ran `mix-blend-mode: multiply` with `saturate(1.25)
 * contrast(1.55)`, which pushes the UI hues UP — the neon that both critics
 * read as "emoji". Nothing here boosts saturation; the whole set comes out on
 * one optical weight in the game's own gouache range.
 */
type Duotone = { s: [number, number, number]; i: [number, number, number] };
const DUOTONE: Record<string, Duotone> = {
  monster:   { s: [0.671, 0.690, 0.680], i: [0.244, 0.161, 0.065] },
  elite:     { s: [0.422, 0.533, 0.370], i: [0.395, 0.246, 0.258] },
  megaelite: { s: [0.814, 0.784, 0.733], i: [0.120, 0.070, 0.004] },
  rest:      { s: [0.375, 0.610, 0.558], i: [0.459, 0.243, 0.173] },
  shop:      { s: [0.787, 0.716, 0.904], i: [0.102, 0.084, -0.159] },
  treasure:  { s: [1.340, 1.214, 1.061], i: [-0.012, -0.012, -0.019] },
  coin:      { s: [0.612, 0.627, 0.699], i: [0.231, 0.165, 0.039] },
};
/** Which duotone each room type prints in. */
const INK_FAMILY: Record<string, keyof typeof DUOTONE | "boss"> = {
  Monster: "monster",
  Elite: "elite",
  MegaElite: "megaelite",
  Shop: "shop",
  Rest: "rest",
  RestSite: "rest",
  Campfire: "rest",
  Treasure: "treasure",
  Event: "coin",
  Ancient: "coin",
  Boss: "boss",
};
/** The game's Unknown/Ancient node is a struck COIN — a filled disc with the
 *  glyph knocked out of it — not a bare "?" floating on the paper (which is
 *  all `run_history/event.png` contains). Disc colour sampled from the ref. */
const COIN_DISC: Record<string, string> = { Event: "#553c06", Ancient: "#443a4c" };
const coinDisc = (t: string | null) => (t && COIN_DISC[t]) || COIN_DISC.Event;
const inkFamily = (t: string | null): keyof typeof DUOTONE | "boss" =>
  (t && INK_FAMILY[t]) || "coin";

const COLW = 76;
const ROWH = 58;
const R_BASE = 19;
/** Rich (full-sheet) lattice, in px at scale 1. Column pitch is 2.9x the stamp
 *  diameter, which is the ratio measured off refs/game-map-2.png: seven columns
 *  on a 153 px pitch carrying ~52 px stamps. It used to be 158 (3.4x), and a
 *  lattice that wide simply does not fit the paper — the outermost column
 *  landed ON the torn right deckle, under the legend slip (live report). Rows
 *  run tighter than the game's 152 px because the workbench sheet is ~200 px
 *  shorter than the game's — at the game's row pitch only four floors would be
 *  on screen, and "~12 nodes across a huge sparse sheet" was the r4 tell. */
const RICH_COLW = 134;
const RICH_ROWH = 112;
const RICH_R = 23;
/** Headroom above the topmost node for the boss ink blot. */
const RICH_TOP_PAD = 128;

/** Natural (scale-1) extent of the rich lattice, so {@link MapView} can size
 *  the sheet to it without a two-pass render. */
function richExtent(fm: FullMap) {
  const nodes: FullMapNode[] = (fm.rows || []).flat();
  const cols = nodes.map((n) => n.col).concat(fm.boss ? [fm.boss.col] : []);
  const rows = nodes.map((n) => n.row).concat(fm.boss ? [fm.boss.row] : []);
  return {
    w: (Math.max(0, ...cols) + 1) * RICH_COLW,
    h: (Math.max(0, ...rows) + 1) * RICH_ROWH + RICH_TOP_PAD,
  };
}

const INK = "var(--gsm-ink, #2b2116)";
const INK_FAINT = "var(--gsm-ink-faint, #6f5a3c)";
/** The AI route: a red-pencil line over the print. The game has no equivalent,
 *  so it must read as a second pen someone took to the map — not as a chart
 *  series. r4's #7a641f olive-yellow was the "exists nowhere in the game
 *  palette" tell. */
const PLAN = "var(--gsm-plan, #3f7a33)";

/** The game's hand-inked lasso: an out-of-round brush loop that overshoots its
 *  own start and trails off, drawn around the node you stand on / have walked.
 *  A geometric ring reads as a web state-indicator; this reads as ink.
 *
 *  In the ref the loop clears the stamp by a wide margin (~2.1x the stamp
 *  radius) and is laid with a fat marker (~0.28r); r4 drew it at 1.32x with a
 *  3 px hairline, which at a glance is indistinguishable from a CSS ring —
 *  "a uniform thin CSS circle" in the art-director's words. */
const lasso = (r: number) =>
  `M ${0.10 * r} ${-1.05 * r}`
  + ` C ${-0.70 * r} ${-1.12 * r} ${-1.16 * r} ${-0.44 * r} ${-1.02 * r} ${0.22 * r}`
  + ` C ${-0.88 * r} ${0.88 * r} ${-0.20 * r} ${1.18 * r} ${0.38 * r} ${1.02 * r}`
  + ` C ${1.00 * r} ${0.85 * r} ${1.22 * r} ${0.14 * r} ${1.00 * r} ${-0.42 * r}`
  + ` C ${0.80 * r} ${-0.92 * r} ${0.28 * r} ${-1.16 * r} ${-0.30 * r} ${-1.02 * r}`
  + ` C ${-0.64 * r} ${-0.94 * r} ${-0.88 * r} ${-0.76 * r} ${-1.02 * r} ${-0.52 * r}`;

/** <image> that walks a candidate-URL chain on load error, so the boss can be
 *  drawn from boss_icons/, the monster illustration, or the small run_history
 *  stamp — whichever exists for this act's boss. */
function InkStamp({ candidates, fallback, ...img }: {
  candidates: (string | null | undefined)[];
  /** Drawn when there is nothing left to try — no candidate at all, or every
   *  one of them 404'd. Without it the caller silently renders an empty
   *  ``<g>``; ``assets.py room_icon()`` returns ``None`` for a boss with no
   *  ``boss_id``, so "no candidate at all" is a state the server produces. */
  fallback?: ReactNode;
} & SVGProps<SVGImageElement>) {
  const list = candidates.filter(Boolean) as string[];
  const [i, setI] = useState(0);
  if (i >= list.length) return <>{fallback ?? null}</>;
  return <image href={list[i]} onError={() => setI(i + 1)} {...img} />;
}

/** One edge, drawn the way the game draws it: a DASHED line that bows a little
 *  off true so it reads as ruled by hand. Walked legs are laid over in a fat
 *  near-black dash, which is the only "you came this way" cue the game gives.
 *
 *  r4 drew evenly-stepped round dots instead ("perfect round dots ... instead
 *  of wobbling hand-drawn capsule dashes"); the ref has no dotted edge
 *  anywhere. The bow is seeded from the edge's own coords (see hash01) so a
 *  replay capture is pixel-identical every time. */
function InkDash({ x1, y1, x2, y2, sa, sb, r, ink, width, dash, gap, m1, m2, opacity }: {
  x1: number; y1: number; x2: number; y2: number;
  /** hash seeds (stable per edge) */
  sa: number; sb: number;
  /** node radius — the dash metrics are all expressed against it */
  r: number;
  ink: string; width: number; dash: number; gap: number;
  /** ink-free margins around the endpoints (keeps the line off the stamps) */
  m1: number; m2: number;
  opacity: number;
}) {
  const dx = x2 - x1, dy = y2 - y1;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist, uy = dy / dist;   // along
  const px = -uy, py = ux;                // perpendicular
  const a1 = Math.min(m1, dist * 0.45), a2 = Math.min(m2, dist * 0.45);
  const sx = x1 + ux * a1, sy = y1 + uy * a1;
  const ex = x2 - ux * a2, ey = y2 - uy * a2;
  const bow = jit(sa, sb, 701, 0.11 * r);
  const cx = (sx + ex) / 2 + px * bow * 2, cy = (sy + ey) / 2 + py * bow * 2;
  return (
    <path d={`M ${sx} ${sy} Q ${cx} ${cy} ${ex} ${ey}`} fill="none"
      stroke={ink} strokeWidth={width} strokeLinecap="round"
      strokeDasharray={`${dash} ${gap}`} opacity={opacity} />
  );
}

/** The per-room duotone filters, plus the boss's flat-ink silhouette filter.
 *  Emitted once per <svg>; ids are namespaced by the caller's React id so two
 *  mounted maps (screen + minimap) never fight over one definition. */
function InkDefs({ fid }: { fid: string }) {
  const LUMA = "0.2126 0.7152 0.0722 0 0";
  return (
    <defs>
      {Object.entries(DUOTONE).map(([k, { s, i }]) => (
        <filter key={k} id={`${fid}-${k}`} colorInterpolationFilters="sRGB"
          x="-25%" y="-25%" width="150%" height="150%">
          <feColorMatrix type="matrix" values={`${LUMA} ${LUMA} ${LUMA} 0 0 0 1 0`} />
          <feComponentTransfer>
            <feFuncR type="linear" slope={s[0]} intercept={i[0]} />
            <feFuncG type="linear" slope={s[1]} intercept={i[1]} />
            <feFuncB type="linear" slope={s[2]} intercept={i[2]} />
          </feComponentTransfer>
        </filter>
      ))}
      {/* The act boss is one flat ink blot on the paper. Flooding SourceAlpha
          (rather than darkening the pixels, as r4's brightness(0.1) did) is
          what makes it a PRINT: a colour render's baked-in soft drop shadow
          survives a brightness filter as the blurred grey ellipse both critics
          called out, but cannot survive an alpha threshold. */}
      <filter id={`${fid}-boss`} colorInterpolationFilters="sRGB">
        <feComponentTransfer in="SourceAlpha" result="cut">
          <feFuncA type="linear" slope="3.4" intercept="-1.25" />
        </feComponentTransfer>
        <feFlood floodColor="#241a0e" result="ink" />
        <feComposite in="ink" in2="cut" operator="in" />
      </filter>
    </defs>
  );
}

/** Proper connected map rendered as SVG from the captured full graph.
 *
 * When ``onSelect`` is given (live ``map_select``), nodes whose ``col,row`` key
 * is in ``selectable`` are highlighted and clickable — clicking one fires
 * ``onSelect(col, row)`` so the player picks a path directly on the map.
 *
 * ``scale`` shrinks every layout dimension uniformly so the same SVG can render
 * either full-size (overlay / map screen) or as a compact persistent minimap.
 *
 * ``rich`` (the parchment map screen / overlay) draws the act boss as a large
 * ink silhouette at the top of the scroll instead of a small node stamp. */
function FullMapGraph({ fm, selectable, onSelect, jumpTargets, onJump, scale = 1, colScale, showPlan = true, rich = false, gameVersion }: {
  fm: FullMap;
  selectable?: Set<string>;
  onSelect?: (col: number, row: number) => void;
  /** Replay: ``"col,row" → {idx, floor}`` of the map decision that chose the
   *  node; clicking a visited node seeks the timeline there via ``onJump``. */
  jumpTargets?: Map<string, { idx: number; floor: number | null }>;
  onJump?: (idx: number) => void;
  scale?: number;
  /** The column pitch's own scale, where it differs from `scale` (a narrow
   *  phone: the columns close up to the sheet, the rows keep their scale). */
  colScale?: number;
  /** Draw the run-AI's planned route to the boss. Off during unadvised live
   *  play — see {@link MapView}. */
  showPlan?: boolean;
  rich?: boolean;
  /** Game build: picks the boss art overlay (see `bossArt`). */
  gameVersion?: string | null;
}) {
  // Filter ids are document-global; namespace them so the map screen and the
  // docked minimap (both mounted at once) each own their own definitions.
  const fid = useId().replace(/[^a-zA-Z0-9]/g, "");
  // The run policy's mark on the node to pick next (advice/marks.ts).
  const { marks } = useAdviceMarks();
  const COL = (rich ? RICH_COLW : COLW) * (colScale ?? scale);
  const ROW = (rich ? RICH_ROWH : ROWH) * scale;
  const R = (rich ? RICH_R : R_BASE) * scale;
  const fs = (px: number) => px * scale;
  // On a phone the lattice sits at its narrow fit floor, where the hit disc
  // (R + 9) comes out ~32 px across. The disc is invisible, so it is padded up
  // to a 44 px diameter; the narrowest pitch (44 px, `MIN_COL_FIT`) keeps neighbours apart.
  const phone = usePhone() !== null;
  const nodes: FullMapNode[] = (fm.rows || []).flat();
  if (fm.boss && !nodes.some((n) => n.row === fm.boss!.row && n.col === fm.boss!.col)) {
    nodes.push({ ...fm.boss, children: [] });
  }
  const byKey = new Map(nodes.map((n) => [`${n.col},${n.row}`, n]));
  const maxRow = Math.max(...nodes.map((n) => n.row), 0);
  const maxCol = Math.max(...nodes.map((n) => n.col), 0);
  const W = (maxCol + 1) * COL;
  // Headroom for what sits above the topmost node: the big boss silhouette on
  // the rich map, or just the name label elsewhere (SVG clips overflow).
  const bigBoss = rich && !!fm.boss;
  const TOP_PAD = (rich ? RICH_TOP_PAD : 26) * scale;
  const H = (maxRow + 1) * ROW + TOP_PAD;
  // A10 DoubleBoss: a SECOND boss node sits directly above the act boss, and
  // it is a different encounter (C# `NBossMapPoint` draws
  // `Act.SecondBossEncounter` there). It arrives as an ordinary `rows` entry,
  // so without this every Boss-typed node would print the act boss's face.
  const sameCoord = (a: { col: number; row: number } | null | undefined,
    b: { col: number; row: number }) => !!a && a.col === b.col && a.row === b.row;
  const bossFor = (c: { col: number; row: number }): BossNode | null | undefined =>
    sameCoord(fm.second_boss, c) ? fm.second_boss : fm.boss;
  // Node positions get a small deterministic wobble so the graph reads as
  // hand-placed ink, not a perfect grid. Both boss anchors stay put — the
  // second boss is drawn straight above the first, as the game stacks them.
  const anchored = (c: number, r: number) =>
    sameCoord(fm.boss, { col: c, row: r }) || sameCoord(fm.second_boss, { col: c, row: r });
  const nx = (c: number, r: number) =>
    c * COL + COL / 2 + (anchored(c, r) ? 0 : jit(c, r, 1, 5 * scale));
  const ny = (r: number, c: number) =>
    (maxRow - r) * ROW + ROW / 2 + TOP_PAD + (anchored(c, r) ? 0 : jit(c, r, 2, 4 * scale));
  const pos = (c: { col: number; row: number }) => ({ x: nx(c.col, c.row), y: ny(c.row, c.col) });

  const cur = fm.current_coord;
  const { plan, edgeSet: planEdges, skipCols } = planOverlay(fm, showPlan);
  const edgeMargin = R + 2 * scale;
  const bossMargin = bigBoss ? R + 16 * scale : edgeMargin;
  const isBossCoord = (c: { col: number; row: number }) =>
    sameCoord(fm.boss, c) || sameCoord(fm.second_boss, c);
  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      <InkDefs fid={fid} />
      {/* edges — dashed ink; fat and near-black where the run already walked */}
      {nodes.flatMap((n) =>
        (n.children || []).map((ch) => {
          const c = byKey.get(`${ch.col},${ch.row}`);
          if (!c) return null;
          const onPath = n.visited && c.visited;
          const onPlan = planEdges.has(edgeKey(n, c)) || planEdges.has(edgeKey(c, n));
          const a = pos(n), b = pos(c);
          const trail = (
            <InkDash x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              sa={n.col * 31 + n.row} sb={c.col * 31 + c.row} r={R}
              ink={onPath ? INK : INK_FAINT}
              width={(onPath ? 0.22 : 0.12) * R}
              dash={(onPath ? 0.85 : 0.6) * R} gap={(onPath ? 0.5 : 0.48) * R}
              m1={edgeMargin} m2={isBossCoord(c) ? bossMargin : edgeMargin}
              opacity={onPath ? 0.92 : 0.5} />
          );
          if (onPlan) {
            // The planned route: someone took a red pencil to the map. Bowed
            // and near-continuous, over the printed edge — a second hand on
            // the same paper rather than a chart series drawn on top of art.
            return (
              <g key={`plan-${n.col},${n.row}->${ch.col},${ch.row}`} data-testid="plan-edge">
                {trail}
                <InkDash x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  sa={n.col * 31 + n.row} sb={c.col * 31 + c.row} r={R}
                  ink={PLAN} width={0.13 * R} dash={1.7 * R} gap={0.28 * R}
                  m1={edgeMargin * 0.7} m2={(isBossCoord(c) ? bossMargin : edgeMargin) * 0.7}
                  opacity={0.6} />
              </g>
            );
          }
          return <g key={`${n.col},${n.row}->${ch.col},${ch.row}`}>{trail}</g>;
        }),
      )}
      {/* A10 DoubleBoss: boss → second boss. The engine gives the boss that
          edge (`ActMap::attach_second_boss` adds the child), but the boss is a
          meta-node with no `children` on the wire, so the only leg of the act
          the player still has to walk after the first boss would otherwise be
          drawn floating free of the map. */}
      {fm.boss && fm.second_boss && (() => {
        const a = pos(fm.boss), b = pos(fm.second_boss);
        // The leg counts as walked once the run stands on (or past) the second
        // boss — the boss meta-node itself carries no `visited` flag.
        const sbNode = byKey.get(coordKey(fm.second_boss));
        const onPath = !!sbNode?.visited || sameCoord(cur, fm.second_boss);
        // Drawn with a SHORT ink-free margin, unlike every other edge into a
        // boss: the two silhouettes all but meet, and at `bossMargin` the leg
        // would live entirely under them. The nodes paint over it either way —
        // what shows is the sliver of path between the two, which is what the
        // game's own `DrawPaths(_bossPointNode, …)` leaves visible.
        return (
          <g key="boss->second-boss">
            <InkDash x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              sa={fm.boss.col * 31 + fm.boss.row} sb={fm.second_boss.col * 31 + fm.second_boss.row}
              r={R} ink={onPath ? INK : INK_FAINT}
              width={(onPath ? 0.22 : 0.12) * R}
              dash={(onPath ? 0.85 : 0.6) * R} gap={(onPath ? 0.5 : 0.48) * R}
              m1={0.6 * R} m2={0.6 * R} opacity={onPath ? 0.92 : 0.5} />
          </g>
        );
      })()}
      {/* plan route into the boss (last route node → boss meta-node). Only
          when the plan actually reaches the boss's doorstep: the live
          advisor's run-policy plan has an 8-floor horizon, and drawing this
          leg from a mid-map node read as an impossible shortcut straight to
          the boss (live report 2026-08-21). A truncated plan instead ends
          with a fading stub past its last node. */}
      {plan && plan.route.length > 0 && fm.boss && (() => {
        const last = plan.route[plan.route.length - 1];
        const lp = pos(last);
        if (fm.boss.row - last.row > 1) {
          return (
            <line x1={lp.x} y1={lp.y} x2={lp.x} y2={lp.y - ROW * 0.6}
              stroke={PLAN} strokeWidth={0.13 * R} strokeDasharray={`${0.2 * R} ${0.45 * R}`}
              strokeLinecap="round" opacity={0.45} />
          );
        }
        const bp = pos(fm.boss);
        return (
          <g data-testid="plan-edge">
            <InkDash x1={lp.x} y1={lp.y} x2={bp.x} y2={bp.y + (bigBoss ? 10 * scale : 0)}
              sa={last.col * 31 + last.row} sb={fm.boss.col * 31 + fm.boss.row} r={R}
              ink={PLAN} width={0.15 * R} dash={1.7 * R} gap={0.28 * R}
              m1={edgeMargin * 0.7} m2={bossMargin * 0.7} opacity={0.6} />
          </g>
        );
      })()}
      {/* current → first-hop leg when the wire graph is MISSING the current
          node itself (live logs sometimes carry only the rows ahead): edges
          render between graph nodes only, so without this the route floats,
          disconnected from where you actually stand. */}
      {plan && plan.route.length > 0 && cur != null && !byKey.has(`${cur.col},${cur.row}`)
        && plan.route[0].row - cur.row === 1 && (() => {
        const cp = pos(cur), fp = pos(plan.route[0]);
        return (
          <g data-testid="plan-edge-current">
            <InkDash x1={cp.x} y1={cp.y} x2={fp.x} y2={fp.y}
              sa={cur.col * 31 + cur.row} sb={plan.route[0].col * 31 + plan.route[0].row} r={R}
              ink={PLAN} width={0.15 * R} dash={1.7 * R} gap={0.28 * R}
              m1={0} m2={edgeMargin * 0.7} opacity={0.6} />
            <circle cx={cp.x} cy={cp.y} r={0.2 * R} fill={PLAN} opacity={0.8} />
          </g>
        );
      })()}
      {/* No caption on the route. r4 lettered "route" beside its first step;
          the line is already the only red-pencil mark on a sheet of black
          print, and a word naming it says nothing the mark does not. The
          legend below the sheet still explains what the pencil means. */}
      {/* nodes */}
      {nodes.map((n) => {
        const m = meta(n.type);
        const isCurrent = cur != null && cur.col === n.col && cur.row === n.row;
        const canSelect = !!onSelect && !!selectable?.has(`${n.col},${n.row}`);
        const jump = !canSelect && onJump ? jumpTargets?.get(`${n.col},${n.row}`) : undefined;
        const skipHere = canSelect && skipCols.get(n.col) === true;
        const p = pos(n);
        const activate = canSelect ? () => onSelect!(n.col, n.row) : jump ? () => onJump!(jump.idx) : undefined;
        const faded = !n.visited && !isCurrent && !canSelect;

        // The act boss, drawn big: an ink-silhouette illustration bleeding up
        // from the top of the scroll, its name beneath in the game face —
        // replaces the old small stamp + red mono text label.
        if (rich && n.type === "Boss") {
          // A10 draws BOTH boss nodes at 0.75 — the game's own answer to two
          // full-size silhouettes stacked one row apart (`NMapScreen`:
          // `if (map.SecondBossMapPoint != null) … Scale = new Vector2(0.75f,
          // 0.75f)` on the boss node and on the second). Without it they
          // overlap on a lattice this tight and read as one smear.
          const bs = fm.second_boss ? 0.75 : 1;
          const BW = 6.0 * R * bs, BH = 5.2 * R * bs;
          // WHICH boss stands here: at A10 the node one row above the act boss
          // is the second boss, a different encounter with its own art.
          const b = bossFor(n);
          const art = bossArt(b, n.icon, gameVersion);
          const second = sameCoord(fm.second_boss, n);
          const role = second ? "the act's second boss" : "the act boss";
          return (
            <g key={`${n.col},${n.row}`} transform={`translate(${p.x},${p.y})`}
              data-current={isCurrent || undefined}
              data-can={canSelect || undefined}
              data-jump={jump ? true : undefined}
              tabIndex={canSelect || jump ? 0 : undefined}
              role={canSelect || jump ? "button"
                : art.name ? "img" : undefined}
              aria-label={canSelect ? `Go to ${role}${art.name ? ` (${art.name})` : ""}`
                : jump ? `Jump to floor ${jump.floor ?? "?"}`
                : art.name ? `${second ? "Second boss" : "Act boss"}: ${art.name}` : undefined}
              onClick={activate}
              onKeyDown={activate ? (e) => {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
              } : undefined}
              style={canSelect || jump ? { cursor: "pointer" } : undefined}>
              {/* The game never prints the boss's name on the sheet — the ink
                  blot IS the label. Kept in <title>/aria so the name is one
                  hover (and one screen-reader stop) away. */}
              {art.name && <title>{art.name}</title>}
              {/* The boss is a legal map choice on the pre-boss floor, so this
                  node has to be clickable like any other — the illustration
                  itself is `pointer-events: none`, so without this hit target
                  the last decision of the act could not be made at all. Sized
                  to the drawn silhouette rather than to `R`. */}
              {(canSelect || jump) && (
                <rect x={-BW / 2} y={-BH + 12 * scale} width={BW} height={BH}
                  fill="transparent" />
              )}
              {/* The boss's own art, printed flat — see {@link bossArt} for
                  why the candidate ORDER is the whole fix. */}
              <InkStamp key={`ink-${art.name ?? "boss"}`} className="gsm-boss-ink"
                candidates={art.candidates}
                filter={`url(#${fid}-boss)`}
                x={-BW / 2} y={-BH + 12 * scale} width={BW} height={BH}
                preserveAspectRatio="xMidYMax meet" pointerEvents="none"
                fallback={
                  /* No art at all to print — say the word in the game face
                     rather than leave the last node of the act blank. */
                  <text textAnchor="middle" y={-R * 0.2} fontSize={fs(15)} fontWeight={700}
                    fill={INK} opacity={0.9} pointerEvents="none"
                    style={{ letterSpacing: 1.5 }}>
                    {art.name ?? "BOSS"}
                  </text>
                } />
            </g>
          );
        }
        return (
          <g key={`${n.col},${n.row}`} transform={`translate(${p.x},${p.y})`}
            data-current={isCurrent || undefined}
            data-can={canSelect || undefined}
            data-jump={jump ? true : undefined}
            tabIndex={canSelect || jump ? 0 : undefined}
            role={canSelect || jump ? "button" : undefined}
            aria-label={canSelect ? `Go to ${n.type ?? "node"} (${n.col},${n.row})`
              : jump ? `Jump to floor ${jump.floor ?? "?"}` : undefined}
            onClick={activate}
            onKeyDown={activate ? (e) => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
            } : undefined}
            style={canSelect || jump ? { cursor: "pointer" } : undefined}>
            {canSelect && <title>Go to {n.type ?? "node"} ({n.col},{n.row}){skipHere ? " — keeps an elite-skip option" : ""}</title>}
            {jump && <title>Jump to floor {jump.floor ?? "?"}</title>}
            {/* larger transparent hit target for easier clicking */}
            {(canSelect || jump) && <circle data-hit r={Math.max(R + 9 * scale, phone && rich ? 22 : 0)} fill="transparent" />}
            {/* visited/current nodes are lassoed in ink — a heavy hand-drawn
                brush loop, the game's own marking. The shipped map_ping.png is
                a hairline WHITE ring: on parchment it read as a pale glow halo,
                which is exactly the web-app tell we are removing. */}
            {(n.visited || isCurrent) && (
              /* spun by the node's own hash so no two loops start in the same
                 place — a repeated identical curve reads as a shape tool,
                 which is what a lasso is here to avoid */
              <path className="gsm-ink-lasso" pointerEvents="none"
                d={lasso(R * 1.7)} transform={`rotate(${jit(n.col, n.row, 9, 180)})`}
                stroke={INK} strokeWidth={(isCurrent ? 0.3 : 0.25) * R}
                strokeLinecap="round" fill="none"
                opacity={isCurrent ? 0.94 : 0.72} />
            )}
            {/* reachable next step: the same brush lasso, drawn open/dashed so
                it reads as an invitation rather than a mark already made */}
            {canSelect && <path className="gsm-can-ring" d={lasso(R * 1.62)} fill="none"
              transform={`rotate(${jit(n.col, n.row, 13, 180)})`}
              stroke={INK} strokeWidth={0.26 * R} strokeLinecap="round"
              strokeDasharray={`${0.55 * R} ${0.34 * R}`} opacity={0.8} />}
            {/* The AI's pick: a gold ring, breathing, and its step number —
                the same mark the cards and the screens' options wear. */}
            {canSelect && (() => {
              const mk = markFor(marks, "select_map_node", { col: n.col, row: n.row });
              if (!mk) return null;
              return (
                <g className={`gsm-advice gsm-advice--${mk.tier}`} pointerEvents="none" data-testid="map-advice">
                  {/* The shared mark look (responsive.css --mark-*, .wb-mark-badge). */}
                  <circle r={R * 1.95} fill="none" style={{ stroke: "var(--mark-gold)" }}
                    strokeWidth={(mk.tier === "best" ? 0.28 : 0.16) * R} />
                  {mk.order != null && (
                    <g transform={`translate(${R * 1.5},${-R * 1.5})`}>
                      <circle r={R * 0.72} style={{ fill: "var(--mark-gold)" }} stroke="oklch(32% 0.05 75)" strokeWidth={0.12 * R} />
                      <text textAnchor="middle" dy={R * 0.3} fontSize={R * 0.9} fontWeight={800}
                        style={{ fontFamily: "var(--font)" }} fill="oklch(20% 0.02 70)">{mk.order}</text>
                    </g>
                  )}
                </g>
              );
            })()}
            <g className={`gsm-node-art${faded ? " gsm-node-future" : ""}`}>
              {(() => {
                // The game's own room stamp, printed into the paper by the
                // per-type duotone (see DUOTONE) — one optical weight, one
                // muted palette, no backing disc except where the game itself
                // strikes a coin. Server-resolved icon first, then the
                // client-side run_history stamp for the type.
                const stamp = n.icon ?? stampFor(n.type);
                const fam = inkFamily(n.type);
                if (!stamp) {
                  return (
                    <>
                      <circle r={R * 0.85} fill={n.visited ? m.color : "var(--gsm-parch-hi, #d9c49a)"}
                        stroke={m.color} strokeWidth={1.8 * scale} />
                      <text textAnchor="middle" dy={fs(5)} fontSize={fs(14)}
                        fill={n.visited ? "#f0e6cd" : m.color}>{m.glyph}</text>
                    </>
                  );
                }
                const ink = `url(#${fid}-${fam === "boss" ? "megaelite" : fam})`;
                if (fam === "coin") {
                  // Unknown / Ancient: the game strikes these as a filled coin
                  // with the glyph knocked out. `run_history/event.png` is only
                  // the bare "?" — on its own it read as typography on paper,
                  // which is exactly what the player critic called out.
                  const cr = R * 0.92, gw = R * 1.18;
                  return (
                    <>
                      <circle r={cr} fill={coinDisc(n.type)} />
                      <circle r={cr} fill="none" stroke="#3b2b07" strokeWidth={0.07 * R}
                        opacity={0.5} />
                      <image href={stamp} x={-gw / 2} y={-gw / 2} width={gw} height={gw}
                        filter={ink} pointerEvents="none" />
                    </>
                  );
                }
                return (
                  <image href={stamp} x={-R} y={-R} width={R * 2} height={R * 2}
                    filter={ink} pointerEvents="none" />
                );
              })()}
            </g>
            {/* you-are-here: the game's red map caret — a small solid wedge
                dropped over the lasso. r4's notched double-barbed chevron with
                a dark outline was a UI arrow, not a pen mark. */}
            {isCurrent && (
              <path d={`M${-0.42 * R} ${-(R * 1.92)}
                        L${0.42 * R} ${-(R * 1.92)}
                        L0 ${-(R * 1.18)} Z`}
                fill="#bb3628" pointerEvents="none" />
            )}
            {/* elite-skip badge: this first move keeps an elite-free escape open */}
            {skipHere && <text textAnchor="middle" x={R * 1.25} y={-R * 0.8} fontSize={0.55 * R}
              fill={PLAN} fontWeight={700} opacity={0.8}>⤳</text>}
            {/* boss name on the small stamp (compact/minimap view) */}
            {!rich && n.type === "Boss" && fm.boss?.name && (
              <text textAnchor="middle" y={-R - fs(6)} fontSize={fs(11)} fontWeight={650}
                fill={INK} opacity={0.9}>
                {fm.boss.name}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Fallback: per-floor chosen-path view when no full graph was captured. */
function ReconstructedRows({ rows, currentIdx, onJump }: {
  rows: MapRow[]; currentIdx: number; onJump?: (idx: number) => void;
}) {
  const ordered = [...rows].reverse();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {ordered.map((r) => (
        <div key={r.idx} onClick={onJump ? () => onJump(r.idx) : undefined}
          // Same marker the lattice puts on its you-are-here node, so the
          // scroll effect can centre this row too — without it the effect had
          // nothing to find here and fell through to its lattice-shaped
          // fallback.
          data-current={r.idx === currentIdx || undefined}
          title={onJump ? `Jump to floor ${r.floor ?? "?"}` : undefined}
          style={{ display: "flex", alignItems: "center", gap: 16, padding: "4px 8px", borderRadius: 6,
          cursor: onJump ? "pointer" : undefined,
          background: r.idx === currentIdx ? "rgba(60, 44, 20, 0.18)" : "transparent",
          outline: r.idx === currentIdx ? "1px solid rgba(51, 40, 26, 0.6)" : "none" }}>
          <span className="gsm-muted-parch" style={{ width: 60, fontSize: 12, fontWeight: 600 }}>floor {r.floor}</span>
          <div style={{ display: "flex", gap: 22 }}>
            {r.nodes.map((n) => {
              const m = meta(n.type);
              return (
                <div key={`${n.col},${n.row}`} title={n.type ?? "unknown"}
                  className={n.chosen ? undefined : "gsm-node-future"}
                  style={{ width: 34, height: 34, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 15, color: n.chosen ? "#f0e6cd" : m.color,
                    background: n.icon ? "transparent" : n.chosen ? m.color : "var(--gsm-parch-hi, #d9c49a)",
                    border: n.icon ? "none" : `2px solid ${m.color}`, overflow: "hidden" }}>
                  {n.icon
                    ? <img src={n.icon} alt={n.type ?? "node"} style={{ width: 30, height: 30, objectFit: "contain" }} />
                    : m.glyph}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Legend slip entries — the game's own room stamps with its labels. On the
 *  slip these keep their full saturation (that is how the game's legend prints
 *  them too); only the stamps ON the paper get the duotone. `coin` marks the
 *  rows the game strikes as a filled disc. */
const LEGEND: { label: string; rel: string; coin?: boolean }[] = [
  { label: "Unknown", rel: "run_history/event.png", coin: true },
  { label: "Merchant", rel: "run_history/shop.png" },
  { label: "Treasure", rel: "run_history/treasure.png" },
  { label: "Rest", rel: "run_history/rest_site.png" },
  { label: "Enemy", rel: "run_history/monster.png" },
  { label: "Elite", rel: "run_history/elite.png" },
];

/** Horizontal padding inside .gsm-parch-body — the fallback when the computed
 *  style cannot be read (jsdom); the measurement below reads the real value,
 *  since map.css narrows it on a phone. */
const PARCH_PAD_X = 46;

/** The smallest the lattice is drawn. 0.6 keeps the stamps legible on a sheet
 *  that has the width for it; on a phone-width sheet (< 760px) 0.5, and there
 *  the COLUMNS close up further (`MIN_COL_FIT`) so the act fits the paper
 *  and never scrolls sideways. */
const MIN_FIT = 0.6;
const MIN_FIT_NARROW = 0.5;
/** The tightest column pitch on a phone-width sheet, as a scale of
 *  `RICH_COLW`: 44 px, one tap disc (`FullMapGraph`'s 22 px radius). Only a
 *  sheet under ~7 x 44 px of paper still scrolls sideways. */
const MIN_COL_FIT = 44 / RICH_COLW;

/** Mirrors `--gsm-sheet-inset` in map.css: the parchment layers are 1920 px
 *  wide but the hide itself starts this far in on each side, so the paper the
 *  ink has to land on is narrower than the body that measures it. Fitting the
 *  lattice to the BODY overshot the sheet by exactly this margin. */
const SHEET_INSET = 0.084;

/** What the sheet does when you click it. This used to be printed ON the
 *  parchment in italics; the game's map carries no text at all, so it moved to
 *  the surface's own tooltip (each node also states its own action in a
 *  <title>, which is where a click actually lands). */
function mapHint(hasGraph: boolean, canPick: boolean, canJump: boolean): string {
  if (!hasGraph) return "reconstructed path — re-record the run to capture the full graph";
  if (canPick) return "click a highlighted node to continue";
  if (canJump) return "click a visited node to jump the replay there";
  return "map";
}

const REGIONS = ["overgrowth", "hive", "glory", "underdocks"];
/** Map act name → parchment region set. Only trusts an explicit region name in
 *  the act title; everything else gets the overgrowth parchment (the layers
 *  differ mostly in tint, and guessing an act→region table wrong is louder
 *  than a consistent default). */
function regionFor(actName: string | null): string {
  const s = (actName ?? "").toLowerCase();
  for (const r of REGIONS) if (s.includes(r)) return r;
  return "overgrowth";
}

export default function MapView({ rows, graphs, currentIdx, selectable, onSelect, onJump, scale = 1, compact = false, live = false, gameVersion }: {
  rows: MapRow[]; graphs: MapGraph[]; currentIdx: number;
  selectable?: Set<string>; onSelect?: (col: number, row: number) => void;
  /** Replay: seek the timeline to a map decision step (click a visited node / floor row). */
  onJump?: (idx: number) => void;
  /** Shrink the rendered graph uniformly (1 = full size). Used by the minimap. */
  scale?: number;
  /** Compact chrome for the persistent minimap: tight padding, no header text. */
  compact?: boolean;
  /** This map belongs to the LIVE run, so the AI's planned route is advice
   *  about a decision you have not made yet and follows the hints switch.
   *  Replay is review of a finished run — its route always renders. */
  live?: boolean;
  /** Game build the map belongs to: a later build's new bosses draw from
   *  its art overlay (see `bossArt`). */
  gameVersion?: string | null;
}) {
  // The ⤳ route is the run-AI telling you which way to go: unprompted
  // advice on the map screen, and just as much of a spoiler as the policy
  // panel. It is hidden during unadvised live play, in the minimap and the
  // on-demand overlay as well as the map screen itself — those show the same
  // graph, so gating only one of them would leak the answer anyway.
  const hints = useHints();
  const showPlan = !live || hints.on;
  // Prefer the full graph whose step is at or most recently before the current step.
  const graph = graphs.filter((g) => g.idx <= currentIdx).slice(-1)[0] ?? graphs[0];
  // The act is taller than the viewport: keep the you-are-here marker in view
  // when the full map opens or the step changes (the minimap does the same).
  const sceneRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const curKey = graph?.full_map?.current_coord
    ? coordKey(graph.full_map.current_coord) : null;
  // Which of the two renderers is on screen — they scroll opposite ways.
  const hasGraph = !!graph;

  // Size the lattice to the PAPER's width rather than to a fixed multiplier —
  // and to the paper, not the body: the body runs the full width of the stage
  // while the sheet's hide stops SHEET_INSET short of it on each side, so a
  // fit measured off `clientWidth` puts the outer column on the deckle.
  // r4 hard-coded a 1.55x spread, so on this viewport the act's seven columns
  // over-ran the paper while the rows huddled — "~12 nodes across a huge
  // sparse sheet". Height deliberately stays free to overflow: the act really
  // is taller than the frame in the game too, and it scrolls away under the
  // torn deckle. Quantised to 1/50 so a resize can't oscillate and so a
  // replay capture at a given viewport renders identically every time.
  const [fit, setFit] = useState(1);
  const [colFit, setColFit] = useState<number | undefined>(undefined);
  const natW = graph?.full_map ? richExtent(graph.full_map).w : 0;
  useLayoutEffect(() => {
    if (compact || !natW) return;
    const measure = () => {
      const el = bodyRef.current;
      if (!el) return;
      const cs = typeof getComputedStyle === "function" ? getComputedStyle(el) : null;
      const padX = cs ? parseFloat(cs.paddingLeft) || PARCH_PAD_X : PARCH_PAD_X;
      const avail = el.clientWidth * (1 - 2 * SHEET_INSET) - padX * 2;
      if (avail <= 0) return;
      const floor = el.clientWidth < 760 ? MIN_FIT_NARROW : MIN_FIT;
      const next = Math.max(floor, Math.min(1.12, Math.round((avail / natW) * 50) / 50));
      setFit(next);
      // A narrow sheet at the floor: the columns alone close up to the paper.
      const narrow = el.clientWidth < 760 && avail / natW < next;
      setColFit(narrow ? Math.max(MIN_COL_FIT, Math.floor((avail / natW) * 200) / 200) : undefined);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [compact, natW]);

  useLayoutEffect(() => {
    if (compact) return;
    const cur = sceneRef.current?.querySelector("[data-current]");
    if (cur) { cur.scrollIntoView?.({ block: "center", inline: "nearest" }); return; }
    // Nothing marked current. The SVG lattice is read from the BOTTOM — row 0
    // is the floor the run stands on — so park there rather than at scrollTop
    // 0, which showed the boss over a half-empty sheet.
    //
    // That is wrong for the OTHER renderer: `ReconstructedRows` lays its rows
    // out `[...rows].reverse()`, newest floor first, so its right end is
    // already the top. Scrolling it to `scrollHeight` landed a replay recorded
    // without a captured graph on floor 1 every time.
    if (!hasGraph) return;
    const body = bodyRef.current;
    if (body) body.scrollTop = body.scrollHeight;
  }, [compact, curKey, currentIdx, fit, hasGraph]);
  const actName =
    graph?.act_name ??
    (graph?.act != null ? `act ${graph.act}` : null) ??
    rows[0]?.act_name ??
    (rows[0]?.act != null ? `act ${rows[0].act}` : "this act");

  // Chosen-node → map-decision step, scoped to the displayed act so acts
  // sharing (col,row) coordinates don't cross-jump.
  const jumpTargets = new Map<string, { idx: number; floor: number | null }>();
  if (onJump) {
    for (const r of rows) {
      if (graph?.act != null && r.act !== graph.act) continue;
      for (const n of r.nodes) {
        if (n.chosen && n.col != null && n.row != null) {
          jumpTargets.set(`${n.col},${n.row}`, { idx: r.idx, floor: r.floor });
        }
      }
    }
  }

  const region = regionFor(actName);
  const parchVars = {
    "--gsm-parch-top": `url("${assetUrl(`map_bgs/${region}/map_top_${region}.png`)}")`,
    "--gsm-parch-mid": `url("${assetUrl(`map_bgs/${region}/map_middle_${region}.png`)}")`,
    "--gsm-parch-bot": `url("${assetUrl(`map_bgs/${region}/map_bottom_${region}.png`)}")`,
  } as CSSProperties;

  if (rows.length === 0 && !graph) {
    return <div className="gs-map" style={{ padding: 24, color: "var(--muted)", fontFamily: "var(--game-font, Georgia, serif)" }}>no map data in this run</div>;
  }

  if (compact) {
    // Minimap dock: same ink language on a plain parchment strip; the boss
    // stays a small stamp and the tooling chrome around it is untouched.
    return (
      <div className="gs-map gs-map--compact" style={{ padding: 8, ...parchVars }}>
        {graph
          ? <FullMapGraph fm={graph.full_map} selectable={selectable} onSelect={onSelect}
              jumpTargets={jumpTargets} onJump={onJump} scale={scale} showPlan={showPlan} gameVersion={gameVersion} />
          : <ReconstructedRows rows={rows} currentIdx={currentIdx} onJump={onJump} />}
      </div>
    );
  }

  const plan = graph?.full_map?.path_plan;
  // The game's Proceed banner is the map's confirm affordance. The workbench's
  // equivalent on this screen is "advance the replay to the next map decision"
  // — a real interaction, wired to the same onJump the visited nodes use. In
  // live play the map itself is the picker (you must click a node), so the
  // banner has nothing to confirm and renders disabled, saying so.
  const nextMapIdx = onJump
    ? rows.filter((r) => r.idx > currentIdx).reduce<number | null>(
        (best, r) => (best == null || r.idx < best ? r.idx : best), null)
    : null;
  const proceedTitle = nextMapIdx != null
    ? "jump to the next map decision in this run"
    : onSelect
      ? "pick a node on the map to continue — the map is the picker here"
      : "no later map decision in this run";
  return (
    <div ref={sceneRef} className="gs-map gs-map--scene" style={parchVars}
      title={mapHint(!!graph, !!onSelect && (selectable?.size ?? 0) > 0,
        !onSelect && !!onJump && jumpTargets.size > 0)}>
      {/* The rock the sheet hangs on. Two silhouette walls with a cold rim
          light and a contact shadow where the paper meets them — r4 left flat
          near-black here, which is why the sheet read as "a div with a
          background-color" rather than a map pinned inside a cave. */}
      <div className="gsm-cave gsm-cave--l" aria-hidden="true" />
      <div className="gsm-cave gsm-cave--r" aria-hidden="true" />
      <div className="gsm-stage">
        <div className="gsm-parchment">
          {/* the sheet itself: tileable middle behind the ink … */}
          <div className="gsm-parch-bg" aria-hidden="true">
            <div className="gsm-parch-mid" />
            {/* fibre freckles, foxing blotches and the sheen streak that a
                single tiled texture cannot vary across the sheet */}
            <div className="gsm-parch-grain" />
          </div>
          <div ref={bodyRef} className="gsm-parch-body">
            {graph
              ? <FullMapGraph fm={graph.full_map} selectable={selectable} onSelect={onSelect}
                  jumpTargets={jumpTargets} onJump={onJump} scale={fit} colScale={colFit} showPlan={showPlan} rich gameVersion={gameVersion} />
              : <ReconstructedRows rows={rows} currentIdx={currentIdx} onJump={onJump} />}
          </div>
          {/* … and its two torn deckle edges pinned OVER it, so an act taller
              than the frame scrolls away under real torn paper instead of
              being sliced off by a rectangular clip. */}
          <div className="gsm-parch-edges" aria-hidden="true">
            <div className="gsm-parch-top" />
            {/* the tear, stamped three times at three heights: curl shadow,
                fibrous lip, void — see the .gsm-tear block in map.css */}
            <div className="gsm-tear gsm-tear--shadow" />
            <div className="gsm-tear gsm-tear--lip" />
            <div className="gsm-parch-bot" />
          </div>
        </div>
        {graph && (
          <aside className="gsm-legend" aria-label="Legend">
            <div className="gsm-legend-title">Legend</div>
            {LEGEND.map(({ label, rel, coin }) => (
              <div key={label} className="gsm-legend-row">
                <span className={coin ? "gsm-legend-ico gsm-legend-ico--coin" : "gsm-legend-ico"}>
                  <img src={assetUrl(rel)} alt="" aria-hidden="true" />
                </span>
                <span>{label}</span>
              </div>
            ))}
            {/* The run-AI's route is the one thing on this screen the game does
                not have, so it is named where the game already keeps its map
                key — on the slip. r4 printed this as two lines of italic
                instructional copy across the parchment itself, which is a
                register the game's canvas never uses. */}
            {showPlan && plan && (
              <div className="gsm-legend-route" title={
                // `first_col` set = the run policy that is actually driving
                // chose the first hop and the pathing surface planned the rest
                // (see PathPlan.first_col). Saying so matters: without it the
                // line and the policy panel could name different next rooms
                // and neither said which one the run would take.
                (plan.first_col != null
                  ? `first move ${plan.first_col === plan.best_col ? "agreed by" : "taken from"} the run policy${
                      plan.first_col === plan.best_col ? "" : ` (pathing preferred col ${plan.best_col})`}; the rest is the pathing plan`
                  : plan.diverged
                    ? `re-routed off the hard-max line (col ${plan.baseline_col} → ${plan.best_col}) to keep options open`
                    : "keeps the most options open")
                + (plan.columns.some((c) => c.can_skip_elite)
                  ? " · ⤳ marks first moves that keep an elite-skip option" : "")}>
                <span className="gsm-legend-ico" aria-hidden="true">
                  <svg viewBox="0 0 28 28" width="28" height="28">
                    <path d="M3 24 Q 13 17 15 12 T 25 4" fill="none" stroke={PLAN}
                      strokeWidth="3" strokeLinecap="round" />
                  </svg>
                </span>
                <span>planned route to boss</span>
              </div>
            )}
          </aside>
        )}
        {graph && (
          <button type="button" className="gsm-proceed" title={proceedTitle}
            disabled={nextMapIdx == null}
            onClick={nextMapIdx != null ? () => onJump!(nextMapIdx) : undefined}>
            <span>Proceed</span>
          </button>
        )}
      </div>
    </div>
  );
}
