// The frame's dock layout: which panel sits in which dock around the game,
// in what order, and how big each dock is. Pure; FrameApp renders it and
// persists it per viewer (localStorage).

export type DockId = "top" | "left" | "right" | "bottom";
export type PanelId = "toolbar" | "advice" | "coach" | "log" | "workbench";

export const DOCKS: readonly DockId[] = ["top", "left", "right", "bottom"];
export const PANELS: readonly PanelId[] = ["toolbar", "advice", "coach", "log", "workbench"];

export const PANEL_TITLES: Record<PanelId, string> = {
  toolbar: "Spire Mentor",
  advice: "Advice",
  coach: "Coach",
  log: "Action log",
  workbench: "Workbench",
};

export interface Layout {
  /** Panels per dock, in display order (top/bottom: left to right). */
  docks: Record<DockId, PanelId[]>;
  /** top/bottom: height in CSS px; left/right: width in CSS px. */
  sizes: Record<DockId, number>;
  /** Panels the viewer closed; they keep their dock for when they reopen. */
  hidden: PanelId[];
}

// The game gets everything the panels do not: a one-row toolbar on top and
// one column on the right (advice, coach, log -- the workbench's sidebar
// order). The bottom dock only holds the Workbench panel, hidden by default.
export const DEFAULT_LAYOUT: Layout = {
  docks: { top: ["toolbar"], left: [], right: ["advice", "coach", "log"], bottom: ["workbench"] },
  sizes: { top: 42, left: 320, right: 360, bottom: 260 },
  hidden: ["workbench"],
};

export const MIN_SIZE = 32;

/** The dock a panel lives in (every panel lives in exactly one). */
export function dockOf(layout: Layout, panel: PanelId): DockId {
  for (const d of DOCKS) {
    if (layout.docks[d].includes(panel)) return d;
  }
  return "right";
}

/** Move `panel` into `dock` at `index` (clamped; past the end appends). */
export function movePanel(layout: Layout, panel: PanelId, dock: DockId, index?: number): Layout {
  const docks = { ...layout.docks } as Record<DockId, PanelId[]>;
  const from = dockOf(layout, panel);
  const fromIndex = docks[from].indexOf(panel);
  for (const d of DOCKS) docks[d] = docks[d].filter((p) => p !== panel);
  const target = [...docks[dock]];
  let at = index ?? target.length;
  // Moving later within the same dock: the removal shifted the slot left.
  if (from === dock && fromIndex >= 0 && fromIndex < at) at -= 1;
  at = Math.max(0, Math.min(at, target.length));
  target.splice(at, 0, panel);
  docks[dock] = target;
  return { ...layout, docks, hidden: layout.hidden.filter((p) => p !== panel) };
}

export function setPanelHidden(layout: Layout, panel: PanelId, hidden: boolean): Layout {
  const rest = layout.hidden.filter((p) => p !== panel);
  return { ...layout, hidden: hidden ? [...rest, panel] : rest };
}

export function resizeDock(layout: Layout, dock: DockId, size: number): Layout {
  return { ...layout, sizes: { ...layout.sizes, [dock]: Math.max(MIN_SIZE, Math.round(size)) } };
}

/** The panels a dock shows (hidden ones dropped). */
export function visiblePanels(layout: Layout, dock: DockId): PanelId[] {
  return layout.docks[dock].filter((p) => !layout.hidden.includes(p));
}

/**
 * A stored layout, repaired: unknown panels dropped, duplicates removed,
 * panels it lacks (a newer build added them) put where the default has them.
 * Anything unreadable is the default.
 */
export function normalizeLayout(raw: unknown): Layout {
  if (!raw || typeof raw !== "object") return DEFAULT_LAYOUT;
  const r = raw as Partial<Layout>;
  const docks = { top: [], left: [], right: [], bottom: [] } as Record<DockId, PanelId[]>;
  const seen = new Set<PanelId>();
  for (const d of DOCKS) {
    const list = Array.isArray(r.docks?.[d]) ? (r.docks![d] as unknown[]) : [];
    for (const p of list) {
      if (typeof p === "string" && (PANELS as readonly string[]).includes(p) && !seen.has(p as PanelId)) {
        docks[d].push(p as PanelId);
        seen.add(p as PanelId);
      }
    }
  }
  for (const p of PANELS) {
    if (!seen.has(p)) docks[dockOf(DEFAULT_LAYOUT, p)].push(p);
  }
  const sizes = { ...DEFAULT_LAYOUT.sizes };
  for (const d of DOCKS) {
    const s = r.sizes?.[d];
    if (typeof s === "number" && Number.isFinite(s)) sizes[d] = Math.max(MIN_SIZE, s);
  }
  const hidden = Array.isArray(r.hidden)
    ? (r.hidden.filter((p) => (PANELS as readonly string[]).includes(p as string)) as PanelId[])
    : DEFAULT_LAYOUT.hidden;
  return { docks, sizes, hidden: [...new Set(hidden)] };
}

// v2: the compact default (v1 kept a 220 px bottom dock); a new key so
// existing viewers get it too.
const STORAGE_KEY = "spirementor.frame.layout.v2";

export function loadLayout(): Layout {
  try {
    const text = window.localStorage.getItem(STORAGE_KEY);
    return text ? normalizeLayout(JSON.parse(text)) : DEFAULT_LAYOUT;
  } catch {
    return DEFAULT_LAYOUT;
  }
}

export function saveLayout(layout: Layout): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // private window / blocked storage: the layout just is not remembered
  }
}

