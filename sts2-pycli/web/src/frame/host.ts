// The page's line to the frame host (spire-mentor-frame/FrameForm.cs) over
// WebView2's message channel. Outside the host (a plain browser tab, for
// layout work) there is no channel: messages are dropped and the page shows
// its "not in the frame" state.

import type { Incoming } from "./store";

interface WebViewChannel {
  postMessage(message: unknown): void;
  addEventListener(type: "message", listener: (e: { data: unknown }) => void): void;
  removeEventListener(type: "message", listener: (e: { data: unknown }) => void): void;
}

function channel(): WebViewChannel | undefined {
  return (window as unknown as { chrome?: { webview?: WebViewChannel } }).chrome?.webview;
}

export const inHost = (): boolean => channel() !== undefined;

export interface NativeMenuItem {
  label: string;
  title?: string;
  checked?: boolean;
  disabled?: boolean;
  separator?: boolean;
}

export type HostMessage =
  | { type: "ready" }
  | { type: "game_rect"; x: number; y: number; w: number; h: number }
  | { type: "to_mod"; msg: unknown }
  | { type: "focus_game" }
  | { type: "fullscreen"; on: boolean }
  | { type: "workbench"; run: string | null; step: number | null }
  | { type: "menu"; id: string; x: number; y: number; left: number; top: number; right: number; items: NativeMenuItem[] }
  | { type: "log"; text: string };

/** A line in the host's frame.log (the page has no visible console there). */
export function hostLog(text: string): void {
  postHost({ type: "log", text });
}

/** Page errors go to frame.log too. */
export function installErrorLogging(): void {
  window.addEventListener("error", (e) => hostLog(`error: ${e.message} @ ${e.filename}:${e.lineno}`));
  window.addEventListener("unhandledrejection", (e) => hostLog(`unhandled rejection: ${String(e.reason)}`));
}

const pendingMenus = new Map<string, (index: number | null) => void>();
let menuSeq = 0;
let menuListener: (() => void) | null = null;

/**
 * Open a dropdown as a NATIVE popup at the bottom-left of `anchor`. The game
 * window is a native window above the page, so an HTML dropdown would open
 * behind it; the host's popup menu is a top-level window and covers it.
 * `onPick` gets the item index, or null when the menu closed unpicked.
 */
export function showNativeMenu(anchor: DOMRect, items: NativeMenuItem[], onPick: (index: number | null) => void): void {
  const ch = channel();
  if (!ch) return;
  if (!menuListener) {
    const listener = (e: { data: unknown }) => {
      const m = e.data as { type?: string; id?: string; index?: number };
      if ((m.type === "menu_pick" || m.type === "menu_closed") && m.id && pendingMenus.has(m.id)) {
        const cb = pendingMenus.get(m.id)!;
        pendingMenus.delete(m.id);
        cb(m.type === "menu_pick" ? (m.index ?? null) : null);
      }
    };
    ch.addEventListener("message", listener);
    menuListener = () => ch.removeEventListener("message", listener);
  }
  const id = `m${++menuSeq}`;
  pendingMenus.set(id, onPick);
  const dpr = window.devicePixelRatio || 1;
  // x/y place the menu; left/top/right with y give the button, so the host can
  // tell a second click on it (close) from a click elsewhere.
  postHost({ type: "menu", id, x: anchor.left * dpr, y: anchor.bottom * dpr, left: anchor.left * dpr, top: anchor.top * dpr, right: anchor.right * dpr, items });
}

export function postHost(message: HostMessage): void {
  channel()?.postMessage(message);
}

/** Subscribe to host messages; returns the unsubscribe. */
export function onHost(listener: (m: Incoming) => void): () => void {
  const ch = channel();
  if (!ch) return () => {};
  const wrapped = (e: { data: unknown }) => listener(e.data as Incoming);
  ch.addEventListener("message", wrapped);
  return () => ch.removeEventListener("message", wrapped);
}

/** Run one of the mod's verbs (FrameLink's action table), then hand the
 * keyboard back to the game. */
export function modAction(name: string): void {
  postHost({ type: "to_mod", msg: { type: "action", name } });
  postHost({ type: "focus_game" });
}

/** Panels around the game (true) or back over it, in-game (false). */
export function setUiOutside(outside: boolean): void {
  postHost({ type: "to_mod", msg: { type: "set_ui_outside", value: outside } });
  postHost({ type: "focus_game" });
}

/** The frame window borderless over its whole monitor (true), or as it was. */
export function setFullscreen(on: boolean): void {
  postHost({ type: "fullscreen", on });
}

const UI_OUTSIDE_KEY ="spirementor.frame.uiOutside";

/** The viewer's last choice (default: around the game). */
export function loadUiOutside(): boolean {
  try {
    return window.localStorage.getItem(UI_OUTSIDE_KEY) !== "0";
  } catch {
    return true;
  }
}

export function saveUiOutside(outside: boolean): void {
  try {
    window.localStorage.setItem(UI_OUTSIDE_KEY, outside ? "1" : "0");
  } catch {
    // blocked storage: the choice is not remembered
  }
}

/** The workbench's follow view of the live run (`run` / `step`: open that
 * moment of it). */
export function workbenchFollowUrl(run: string | null = null, step: number | null = null): string {
  const q = new URLSearchParams({ follow: "1" });
  if (run) q.set("run", run);
  if (step != null) q.set("step", String(step));
  return `/?${q.toString()}`;
}

/**
 * The workbench in its own window (the host's WorkbenchWindow), opened or
 * brought to the front; with `run` / `step` it shows that moment. Outside
 * the host (a plain browser tab): a new tab.
 */
export function openWorkbenchWindow(run: string | null = null, step: number | null = null): void {
  if (inHost()) {
    postHost({ type: "workbench", run, step });
    return;
  }
  window.open(workbenchFollowUrl(run, step), "_blank", "noopener");
}

/**
 * The mod's hotkeys, for while the page (not the game) has the keyboard:
 * the same chords map to the same verbs. Returns the verb, or null.
 */
export function hotkeyAction(e: Pick<KeyboardEvent, "key" | "ctrlKey" | "shiftKey" | "altKey">): string | null {
  if (e.altKey) return null;
  if (e.ctrlKey) {
    const k = e.key.toLowerCase();
    if (e.shiftKey) {
      if (k === "z") return "restart_room";
      if (k === "c") return "copy_state";
      if (k === "b") return "report_bug";
      return null;
    }
    if (k === "z") return "undo_turn";
    if (k === "h") return "hide_all";
    return null;
  }
  // The workbench's own play shortcuts (its ? help), mapped to the mod's
  // nearest verbs: g play the recommended move, p / h peek or hint (the
  // advice for this decision), a / f the AI plays on (again to stop), ? help.
  // In the game itself the mod binds g / p / h / f (not a: the game's own).
  if (e.key === "?") return "help";
  if (!e.shiftKey) {
    switch (e.key.toLowerCase()) {
      case "g":
        return "ai_once";
      case "p":
      case "h":
        return "peek";
      case "a":
      case "f":
        return "ai_auto";
    }
  }
  switch (e.key) {
    case "F1":
      return "help";
    case "F4":
      return "toggle_hud";
    case "F5":
      return "peek";
    case "F6":
      return e.shiftKey ? "partner_go" : "ai_once";
    case "F7":
      return e.shiftKey ? "autobattle" : "ai_auto";
    case "F8":
      return e.shiftKey ? "toggle_brain" : "toggle_advisor";
    case "F9":
      return "speed_down";
    case "F10":
      return "speed_up";
    default:
      return null;
  }
}
