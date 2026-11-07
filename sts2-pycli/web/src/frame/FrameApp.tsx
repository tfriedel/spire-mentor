// The Spire Mentor frame page: four docks (top / left / right / bottom)
// around a 16:9 game slot. The frame host puts the real game window over the
// slot (we report where it is, in device pixels); panels are dragged by their
// header between and within docks, docks are resized by their inner edge,
// and the layout is remembered per viewer.

import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState, type ReactNode } from "react";
import { isTextInput } from "../keys";
import { hostLog, hotkeyAction, inHost, loadUiOutside, modAction, onHost, postHost, saveUiOutside, setUiOutside } from "./host";
import {
  DOCKS,
  loadLayout,
  movePanel,
  PANEL_TITLES,
  PANELS,
  resizeDock,
  saveLayout,
  setPanelHidden,
  visiblePanels,
  type DockId,
  type Layout,
  type PanelId,
} from "./layout";
import { AdvicePanel, CoachPanel, LogPanel, ToolbarPanel, WorkbenchPanel } from "./panels";
import { INITIAL_STATE, reduce, type Incoming } from "./store";

const DRAG_MIME = "application/x-spirementor-panel";
const EMPTY_DROP_ZONE = 28;

export default function FrameApp() {
  const [state, dispatch] = useReducer((s: typeof INITIAL_STATE, m: Incoming) => reduce(s, m), INITIAL_STATE);
  const [layout, setLayoutState] = useState<Layout>(loadLayout);
  const [dragging, setDragging] = useState<PanelId | null>(null);
  const [drop, setDrop] = useState<{ dock: DockId; index: number } | null>(null);
  const slot = useRef<HTMLDivElement>(null);
  const [gameBox, setGameBox] = useState({ x: 0, y: 0, w: 0, h: 0 });

  const setLayout = useCallback((next: Layout) => {
    setLayoutState(next);
    saveLayout(next);
  }, []);

  // Panels around the game, or over it in-game (the game then fills the
  // window). The page remembers the choice; the mod owns it at runtime and
  // can flip it from its in-game toolbar, which the page then adopts.
  const [uiOutside, setUiOutsideState] = useState(loadUiOutside);
  const uiOutsideRef = useRef(uiOutside);
  uiOutsideRef.current = uiOutside;
  /** A value we sent the mod and are waiting to see echoed in `ui`. */
  const pendingUiOutside = useRef<boolean | null>(null);
  const chooseUiOutside = useCallback((outside: boolean) => {
    setUiOutsideState(outside);
    saveUiOutside(outside);
    pendingUiOutside.current = outside;
    setUiOutside(outside);
  }, []);

  // Host channel. Every mod (re)sync starts with `hello`: tell it our mode.
  useEffect(() => {
    const off = onHost((m) => {
      if (m.type === "from_mod" && m.msg.type === "hello") {
        hostLog(`mod hello; asking for panels ${uiOutsideRef.current ? "outside" : "inside"}`);
        pendingUiOutside.current = uiOutsideRef.current;
        setUiOutside(uiOutsideRef.current);
      }
      dispatch(m);
    });
    postHost({ type: "ready" });
    return off;
  }, []);

  // The in-game toolbar switched the mode: follow it (unless it is the echo
  // of a switch we asked for that has not landed yet).
  const modUiOutside = state.ui?.ui_outside;
  useEffect(() => {
    if (modUiOutside == null) return;
    if (pendingUiOutside.current != null) {
      if (modUiOutside === pendingUiOutside.current) pendingUiOutside.current = null;
      return;
    }
    if (modUiOutside !== uiOutsideRef.current) {
      hostLog(`mod switched panels ${modUiOutside ? "outside" : "inside"}`);
      setUiOutsideState(modUiOutside);
      saveUiOutside(modUiOutside);
    }
  }, [modUiOutside]);

  // Chip timers.
  useEffect(() => {
    const id = window.setInterval(() => dispatch({ type: "tick", now: Date.now() }), 1000);
    return () => window.clearInterval(id);
  }, []);

  // The newest toolbar state, for key handlers registered once.
  const uiRef = useRef(state.ui);
  uiRef.current = state.ui;

  const togglePanel = useCallback(
    (p: PanelId) => setLayout(setPanelHidden(layout, p, !layout.hidden.includes(p))),
    [layout, setLayout],
  );

  // In-game Ctrl+L (framed) toggles the frame's log panel.
  const lastToggle = useRef(0);
  useEffect(() => {
    if (state.toggles && state.toggles.seq !== lastToggle.current) {
      lastToggle.current = state.toggles.seq;
      if ((PANELS as readonly string[]).includes(state.toggles.panel)) togglePanel(state.toggles.panel as PanelId);
    }
  }, [state.toggles, togglePanel]);

  // The mod's hotkeys while the page has the keyboard.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTextInput(e.target)) return;
      // A held key auto-repeats; the mod's hotkeys fire once per press (its
      // NHotkeyManager skips echoes), or a held F flips autoplay on, off, on...
      if (e.repeat) return;
      // Esc stops what the AI is running, as in the workbench.
      if (e.key === "Escape") {
        const ui = uiRef.current;
        if (ui?.autoplay_armed) modAction("ai_auto");
        else if (ui?.autobattle_on) modAction("autobattle");
        return;
      }
      if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === "l") {
        e.preventDefault();
        togglePanel("log");
        return;
      }
      const action = hotkeyAction(e);
      if (action) {
        e.preventDefault();
        modAction(action);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePanel]);

  // Measure the slot, tell the host where the game goes. Re-measured on a
  // resize of the slot or the window AND on every layout change: moving
  // panels between docks of equal width shifts the slot without resizing it,
  // which no ResizeObserver reports.
  const measureRef = useRef<() => void>(() => {});
  useLayoutEffect(() => {
    const el = slot.current;
    if (!el) {
      measureRef.current = () => {};
      return;
    }
    const measure = () => {
      const r = el.getBoundingClientRect();
      // The game takes the whole slot and letterboxes itself per its own
      // aspect setting: a 16:9 box of ours would not make it any bigger,
      // and with "Auto" the game can use the extra width / height.
      const fit = { x: 0, y: 0, w: r.width, h: r.height };
      setGameBox(fit);
      const dpr = window.devicePixelRatio || 1;
      postHost({
        type: "game_rect",
        x: Math.round((r.left + fit.x) * dpr),
        y: Math.round((r.top + fit.y) * dpr),
        w: Math.round(fit.w * dpr),
        h: Math.round(fit.h * dpr),
      });
    };
    measureRef.current = measure;
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
      measureRef.current = () => {};
    };
  }, [uiOutside]);
  useLayoutEffect(() => measureRef.current(), [layout]);

  const renderPanel = (p: PanelId): ReactNode => {
    switch (p) {
      case "toolbar":
        return (
          <ToolbarPanel
            state={state}
            framePanels={PANELS.filter((q) => q !== "toolbar").map((q) => ({
              label: PANEL_TITLES[q],
              shown: !layout.hidden.includes(q),
              toggle: () => togglePanel(q),
            }))}
            onPanelsInside={() => chooseUiOutside(false)}
          />
        );
      case "advice":
        return <AdvicePanel state={state} />;
      case "coach":
        return <CoachPanel state={state} dispatch={dispatch} />;
      case "log":
        return <LogPanel state={state} />;
      case "workbench":
        return <WorkbenchPanel />;
    }
  };

  const dockSize = (d: DockId): number => {
    if (!uiOutside) return 0;
    if (visiblePanels(layout, d).length > 0) return layout.sizes[d];
    return dragging ? EMPTY_DROP_ZONE : 0;
  };

  const finishDrop = (dock: DockId, index: number) => {
    if (dragging) setLayout(movePanel(layout, dragging, dock, index));
    setDragging(null);
    setDrop(null);
  };

  const vars = {
    "--fr-top": `${dockSize("top")}px`,
    "--fr-bottom": `${dockSize("bottom")}px`,
    "--fr-left": `${dockSize("left")}px`,
    "--fr-right": `${dockSize("right")}px`,
  } as React.CSSProperties;

  return (
    <div className={`fr-root${dragging ? " fr-dragging" : ""}`} style={vars}>
      {uiOutside && DOCKS.map((d) => (
        <Dock
          key={d}
          dock={d}
          layout={layout}
          drop={drop}
          onDragOverIndex={(index) => setDrop({ dock: d, index })}
          onDrop={(index) => finishDrop(d, index)}
          onResize={(size) => setLayout(resizeDock(layout, d, size))}
        >
          {visiblePanels(layout, d).map((p) => (
            <Panel
              key={p}
              id={p}
              onClose={() => togglePanel(p)}
              onDragStart={() => setDragging(p)}
              onDragEnd={() => {
                setDragging(null);
                setDrop(null);
              }}
            >
              {renderPanel(p)}
            </Panel>
          ))}
        </Dock>
      ))}
      <div className="fr-slot" ref={slot}>
        <div
          className="fr-game"
          style={{ left: gameBox.x, top: gameBox.y, width: gameBox.w, height: gameBox.h }}
        >
          <div className="fr-game-placeholder">
            {inHost()
              ? `game: ${state.host.game}`
              : "Not running inside the Spire Mentor frame — the game would sit here (spire-mentor-frame/)."}
          </div>
        </div>
      </div>
    </div>
  );
}

function Dock(props: {
  dock: DockId;
  layout: Layout;
  drop: { dock: DockId; index: number } | null;
  onDragOverIndex: (index: number) => void;
  onDrop: (index: number) => void;
  onResize: (size: number) => void;
  children: ReactNode[];
}) {
  const { dock } = props;
  const horizontal = dock === "top" || dock === "bottom";
  const ref = useRef<HTMLDivElement>(null);
  const count = props.children.length;

  /** Insertion index for a pointer position: before the first panel whose
   * midpoint lies past it, along the dock's axis. */
  const indexAt = (clientX: number, clientY: number): number => {
    const el = ref.current;
    if (!el) return count;
    const panels = Array.from(el.querySelectorAll<HTMLElement>(":scope > .fr-panel"));
    for (let i = 0; i < panels.length; i++) {
      const r = panels[i].getBoundingClientRect();
      const mid = horizontal ? r.left + r.width / 2 : r.top + r.height / 2;
      if ((horizontal ? clientX : clientY) < mid) return i;
    }
    return panels.length;
  };

  const startResize = (e: React.PointerEvent) => {
    e.preventDefault();
    const start = horizontal ? e.clientY : e.clientX;
    const startSize = props.layout.sizes[dock];
    const sign = dock === "top" || dock === "left" ? 1 : -1;
    const move = (ev: PointerEvent) => {
      const now = horizontal ? ev.clientY : ev.clientX;
      props.onResize(startSize + sign * (now - start));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const dropHere = props.drop?.dock === dock ? props.drop.index : null;
  return (
    <div
      ref={ref}
      className={`fr-dock fr-dock-${dock}${horizontal ? " fr-dock-h" : " fr-dock-v"}${count === 0 ? " fr-dock-empty" : ""}`}
      data-drop-index={dropHere ?? undefined}
      style={dropHere != null ? ({ "--fr-drop": dropHere } as React.CSSProperties) : undefined}
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes(DRAG_MIME)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        props.onDragOverIndex(indexAt(e.clientX, e.clientY));
      }}
      onDrop={(e) => {
        e.preventDefault();
        props.onDrop(indexAt(e.clientX, e.clientY));
      }}
    >
      {props.children.map((child, i) => (
        <DropMarkerWrap key={i} active={dropHere === i}>
          {child}
        </DropMarkerWrap>
      ))}
      {dropHere === count && count > 0 && <div className="fr-drop-marker" />}
      {count > 0 && <div className={`fr-splitter fr-splitter-${dock}`} onPointerDown={startResize} />}
    </div>
  );
}

function DropMarkerWrap(props: { active: boolean; children: ReactNode }) {
  return (
    <>
      {props.active && <div className="fr-drop-marker" />}
      {props.children}
    </>
  );
}

function Panel(props: {
  id: PanelId;
  onClose: () => void;
  onDragStart: () => void;
  onDragEnd: () => void;
  children: ReactNode;
}) {
  const drag = {
    draggable: true,
    title: "Drag to another dock or position",
    onDragStart: (e: React.DragEvent) => {
      e.dataTransfer.setData(DRAG_MIME, props.id);
      e.dataTransfer.effectAllowed = "move";
      props.onDragStart();
    },
    onDragEnd: props.onDragEnd,
  };
  // The toolbar has no title bar (every pixel of height is the game's) and no
  // close button (its Panels menu is how hidden panels come back): a grip at
  // its left edge moves it.
  if (props.id === "toolbar") {
    return (
      <section className="fr-panel fr-panel-toolbar fr-panel-compact">
        <span className="fr-grip fr-grip-compact" {...drag}>
          ⠿
        </span>
        <div className="fr-panel-body">{props.children}</div>
      </section>
    );
  }
  return (
    <section className={`fr-panel fr-panel-${props.id}`}>
      <header className="fr-panel-head" {...drag}>
        <span className="fr-grip">⠿</span>
        <span className="fr-panel-title">{PANEL_TITLES[props.id]}</span>
        <button type="button" className="fr-icon" title="Hide (Panels menu brings it back)" onClick={props.onClose}>
          ×
        </button>
      </header>
      <div className="fr-panel-body">{props.children}</div>
    </section>
  );
}
