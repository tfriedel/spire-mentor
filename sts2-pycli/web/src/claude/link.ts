/**
 * The Claude link: this page's half of `python/claude_link.py`.
 *
 * A terminal Claude Code session runs that relay as an MCP server; the relay
 * listens on `ws://127.0.0.1:8091`. With Tools → "🤖 Claude link" on, this
 * page connects to it and answers its requests — so Claude reads the run on
 * screen (which in the browser build lives only in this tab's worker), calls
 * any `/api` route through the same `apiFetch` door the panels use, and puts
 * charts into the Claude dock (`ClaudeDock.tsx`).
 *
 * Pairing: the relay closes any connection whose hello lacks its token (code
 * 4003) — otherwise any local process or localhost page could pose as this
 * page. The token is persistent on the desk, so this origin pairs once: the
 * user pastes it when turning the link on, or opens the workbench with
 * `#claude-link=<token>` (Claude hands either over from `wb_status`).
 *
 * Protocol (JSON text frames):
 *   page → relay  {type: "hello", token, url, mode, static, protocol}   on open, and on mode change
 *   relay → page  {type: "req", id, op: "context" | "call" | "show", …}
 *   page → relay  {type: "res", id, ok, result | error}
 *   relay → page  {type: "cancel", id}   the relay gave up waiting (timeout)
 *
 * The relay refuses a browser origin other than a localhost page (or one in
 * its `CLAUDE_LINK_ORIGINS`), so a hosted build needs that opt-in on the desk.
 * A newer tab takes the link over (close code 4000); the older one then stops
 * retrying, or two tabs would evict each other forever.
 */

export const CLAUDE_LINK_KEY = "sts2.claudeLink";
export const CLAUDE_LINK_TOKEN_KEY = "sts2.claudeLinkToken";
export const DEFAULT_CLAUDE_LINK_URL = "ws://127.0.0.1:8091";
export const PROTOCOL_VERSION = 1;
/** Close code the relay sends to the tab a newer one replaced. */
export const REPLACED_CODE = 4000;
/** Close code for a hello without the relay's pairing token. */
export const BAD_TOKEN_CODE = 4003;

export type LinkStatus = "off" | "waiting" | "connected" | "replaced" | "unauthorized";

export type ShowKind = "html" | "markdown" | "plotly";

export interface ShowItem {
  title: string;
  kind: ShowKind;
  content: unknown;
}

/** What the page lends the link: what is on screen, and how to refresh it. */
export interface LinkHost {
  context(): unknown;
  /** A mutating call succeeded; bring the view in line with it the way the
   *  page's own control for that route would. `result` is the route's reply. */
  afterMutation(route: string, result: CallResult, request: unknown): Promise<void>;
  show(item: ShowItem): void;
}

export type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

/** What a `call` returns to the relay. */
export type CallResult = { status: number; body: unknown } | { status: number; frames: unknown[] };

interface ReqMsg {
  type: "req";
  id: number;
  op: string;
  method?: string;
  route?: string;
  body?: unknown;
  item?: unknown;
}

interface CancelMsg {
  type: "cancel";
  id: number;
}

/** `POST /api/live/*` routes that only READ the run (searches, advisors,
 *  reviews, previews) or touch nothing on screen (a save, the run-mode
 *  config). Refreshing after one would wipe the scores on screen. */
export const READ_ONLY_LIVE: ReadonlySet<string> = new Set([
  "/api/live/hint",
  "/api/live/search_tree",
  "/api/live/choice-ranker",
  "/api/live/card-advisor",
  "/api/live/card-advisor/stream",
  "/api/live/decision-advisor",
  "/api/live/decision-advisor/stream",
  "/api/live/auto-action",
  "/api/live/ppo-policy",
  "/api/live/e2e-advice",
  "/api/live/last-turn-grade",
  "/api/live/last-choice-grade",
  "/api/live/last-choice-check/stream",
  "/api/live/last-move-regret",
  "/api/live/regret-review",
  "/api/live/wpa-review",
  "/api/live/wpa-verify",
  "/api/live/import-state/preview",
  "/api/live/simlab/sweep",
  "/api/live/save",
  "/api/live/run-mode",
]);

/** Non-GET routes that change the live run (or replace it). Every non-GET
 *  `/api/live/*` route must sit in exactly one of these two sets — a test
 *  checks that against `api.ts` and the Rust route table — and an unlisted
 *  one is treated as mutating, which only costs a refresh. */
export const MUTATING: ReadonlySet<string> = new Set([
  "/api/live/start",
  "/api/live/start-faithful",
  "/api/live/from_replay",
  "/api/live/import-state",
  "/api/live/action",
  "/api/live/seek",
  "/api/live/quit",
  "/api/live/run-mode-step",
  "/api/live/faithful-step",
  "/api/live/play-to-end",
  "/api/live/autobattle/advance",
  "/api/live/autobattle/step",
  "/api/saves/load",
]);

export const routePath = (route: string): string => route.split("?")[0];

export function isMutating(method: string, route: string): boolean {
  const path = routePath(route);
  if (method === "GET" || method === "HEAD") return false;
  if (MUTATING.has(path)) return true;
  return path.startsWith("/api/live/") && !READ_ONLY_LIVE.has(path);
}

function parseShowItem(raw: unknown): ShowItem {
  const it = raw as Partial<ShowItem> | null;
  if (!it || typeof it.title !== "string" || !["html", "markdown", "plotly"].includes(it.kind as string)) {
    throw new Error("show: expected {title, kind: html|markdown|plotly, content}");
  }
  return { title: it.title, kind: it.kind as ShowKind, content: it.content };
}

async function callRoute(msg: ReqMsg, host: LinkHost, fetcher: Fetcher, signal?: AbortSignal): Promise<CallResult> {
  const method = (msg.method ?? "GET").toUpperCase();
  const route = msg.route ?? "";
  if (!route.startsWith("/api/") && route !== "/api") throw new Error(`not an /api route: ${route}`);
  const init: RequestInit = { method };
  if (signal) init.signal = signal;
  // fetch() throws on a GET/HEAD with a body; a body there carries nothing.
  if (msg.body != null && method !== "GET" && method !== "HEAD") {
    init.body = JSON.stringify(msg.body);
    init.headers = { "content-type": "application/json" };
  }
  const resp = await fetcher(route, init);
  const text = await resp.text();
  let result: CallResult;
  if ((resp.headers.get("content-type") ?? "").includes("ndjson")) {
    const frames = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as unknown);
    result = { status: resp.status, frames };
  } else {
    let body: unknown = text;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      /* a non-JSON body (an HTML error page) goes back as text */
    }
    result = { status: resp.status, body };
  }
  if (signal?.aborted) throw new Error("cancelled: the relay stopped waiting");
  if (resp.ok && isMutating(method, route)) await host.afterMutation(route, result, msg.body ?? null);
  return result;
}

/** Answer one relay request. Throws → an `ok: false` reply. */
export async function answer(msg: ReqMsg, host: LinkHost, fetcher: Fetcher, signal?: AbortSignal): Promise<unknown> {
  switch (msg.op) {
    case "context":
      return host.context();
    case "call":
      return callRoute(msg, host, fetcher, signal);
    case "show":
      host.show(parseShowItem(msg.item));
      return null;
    default:
      throw new Error(`unknown op: ${msg.op}`);
  }
}

export interface ClientOptions {
  url: string;
  token: string;
  host: LinkHost;
  fetcher: Fetcher;
  onStatus: (s: LinkStatus) => void;
  hello: () => Record<string, unknown>;
  /** Injectable for tests. */
  WebSocketImpl?: typeof WebSocket;
}

const BACKOFF_MS = [1000, 2000, 4000, 8000, 15000];

/** The socket, its reconnect loop, and the request dispatch. */
export class ClaudeLinkClient {
  private ws: WebSocket | null = null;
  private stopped = false;
  private attempt = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  /** Open requests, so a relay `cancel` can abort one. The engine worker
   *  cannot interrupt a search it has started; an abort stops a server-backed
   *  fetch and drops the result (and its refresh) when it does arrive. */
  private inflight = new Map<number, AbortController>();

  constructor(private readonly opts: ClientOptions) {}

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close(1000, "link turned off");
  }

  /** Re-send the hello (the view mode changed). */
  announce(): void {
    this.send({ type: "hello", ...this.opts.hello(), token: this.opts.token });
  }

  private connect(): void {
    if (this.stopped) return;
    const Impl = this.opts.WebSocketImpl ?? WebSocket;
    let ws: WebSocket;
    try {
      ws = new Impl(this.opts.url);
    } catch {
      this.retry();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.opts.onStatus("connected");
      this.announce();
    };
    ws.onmessage = (ev: MessageEvent) => {
      void this.handle(ev.data);
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.stopped) return;
      if (ev.code === REPLACED_CODE || ev.code === BAD_TOKEN_CODE) {
        this.stopped = true;
        this.opts.onStatus(ev.code === REPLACED_CODE ? "replaced" : "unauthorized");
        return;
      }
      this.retry();
    };
  }

  private retry(): void {
    this.opts.onStatus("waiting");
    const delay = BACKOFF_MS[Math.min(this.attempt, BACKOFF_MS.length - 1)];
    this.attempt += 1;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.connect();
    }, delay);
  }

  private send(msg: unknown): void {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  private async handle(data: unknown): Promise<void> {
    let msg: ReqMsg | CancelMsg;
    try {
      msg = JSON.parse(String(data)) as ReqMsg | CancelMsg;
    } catch {
      return;
    }
    if (typeof msg?.id !== "number") return;
    if (msg.type === "cancel") {
      this.inflight.get(msg.id)?.abort();
      return;
    }
    if (msg.type !== "req") return;
    const ctl = new AbortController();
    this.inflight.set(msg.id, ctl);
    try {
      const result = await answer(msg, this.opts.host, this.opts.fetcher, ctl.signal);
      this.send({ type: "res", id: msg.id, ok: true, result: result ?? null });
    } catch (e) {
      this.send({ type: "res", id: msg.id, ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally {
      this.inflight.delete(msg.id);
    }
  }
}

function readKey(key: string): string | null {
  try {
    return globalThis.localStorage?.getItem(key) || null;
  } catch {
    return null;
  }
}

function writeKey(key: string, value: string | null): void {
  try {
    if (value) globalThis.localStorage?.setItem(key, value);
    else globalThis.localStorage?.removeItem(key);
  } catch {
    /* private mode: the setting lives for this page load only */
  }
}

/** The stored relay URL, or null when the link is off. */
export const storedLinkUrl = (): string | null => readKey(CLAUDE_LINK_KEY);
export const storeLinkUrl = (url: string | null): void => writeKey(CLAUDE_LINK_KEY, url);
export const storedLinkToken = (): string | null => readKey(CLAUDE_LINK_TOKEN_KEY);
export const storeLinkToken = (token: string | null): void => writeKey(CLAUDE_LINK_TOKEN_KEY, token?.trim() || null);

/** A `#claude-link=<token>` pairing link: store the token, turn the link on,
 *  and drop the fragment from the address bar (and so from the history).
 *  Returns whether the URL carried one. */
export function consumePairingFragment(loc: Pick<Location, "hash" | "pathname" | "search"> | undefined = globalThis.location,
  replace: (url: string) => void = (url) => globalThis.history?.replaceState(null, "", url)): boolean {
  const m = /(?:^#|&)claude-link=([^&]+)/.exec(loc?.hash ?? "");
  if (!m || !loc) return false;
  // Drop the fragment either way: a malformed one (a truncated paste) must
  // neither crash the first render nor sit in the address bar.
  replace(loc.pathname + loc.search);
  let token: string;
  try {
    token = decodeURIComponent(m[1]);
  } catch {
    return false;
  }
  storeLinkToken(token);
  if (!storedLinkUrl()) storeLinkUrl(DEFAULT_CLAUDE_LINK_URL);
  return true;
}
