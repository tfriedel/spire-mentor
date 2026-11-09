/**
 * The seed checker: paste a game seed, see what it deals, let Instinct play
 * sampled attempts on it in several workers until one wins (or the time cap),
 * copy the verdict, open the winning (or best) run in the replay viewer.
 * "Win rate" measures the bot's sampled win rate on the seed separately.
 *
 * Character, build and ascension are picked on the page. Instinct only plays
 * Ironclad on v0.111.0 for now, so the other characters and builds are shown
 * grayed out; any ascension 0-10 works (it trained at A10).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { seedCheckApi, type Preview, type SeedCheckApi } from "./api";
import { defaultPoolSize, SeedCheckPool, type Job, type JobResult } from "./pool";
import type { PoolWorker } from "./protocol";
import { deathsByFloor, tally, type AttemptResult, type FloorDeaths } from "./stats";
import { checkVerdict, rateVerdict, type Conditions } from "./verdict";

/** What the page needs from a pool (tests pass a fake). */
export interface PoolLike {
  run(job: Job, onProgress: (rs: AttemptResult[]) => void): Promise<JobResult>;
  stop(): void;
  dispose(): void;
  readonly alive: number;
}

export const CHECK_CAP_MS = 300_000;

/** Every character and build the game has; only the enabled ones are playable. */
export const CHARACTERS = [
  { id: "Ironclad", enabled: true },
  { id: "Silent", enabled: false },
  { id: "Defect", enabled: false },
  { id: "Necrobinder", enabled: false },
  { id: "Regent", enabled: false },
] as const;
export const VERSIONS = [
  { id: "v0.111.0", enabled: true },
  { id: "v0.107.1", enabled: false },
] as const;
export const ASCENSIONS = Array.from({ length: 11 }, (_, a) => a);
const TRAINED_ASCENSION = 10;
export const RATE_RUNS = 200;

function spawnWorker(): PoolWorker {
  return new Worker(new URL("./seedcheck.worker.ts", import.meta.url), { type: "module" }) as unknown as PoolWorker;
}

function defaultPool(): PoolLike {
  return new SeedCheckPool({ size: defaultPoolSize(), spawn: spawnWorker });
}

const BUILD: string = (import.meta.env?.VITE_BUILD_ID as string | undefined) ?? "dev";

interface Outcome {
  verdict: string;
  logPath: string | null;
  logLabel: string;
  deaths: FloorDeaths[];
  reason: JobResult["reason"];
  errors: string[];
}

/** One result per attempt index (the first kept): a resumed check must never
 *  count an attempt twice. */
function uniqueByK(rs: AttemptResult[]): AttemptResult[] {
  const seen = new Set<number>();
  return rs.filter((r) => !seen.has(r.k) && (seen.add(r.k), true));
}

function fmtSecs(ms: number): string {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export default function SeedCheckPage(props: { api?: SeedCheckApi; makePool?: () => PoolLike }) {
  const api = props.api ?? seedCheckApi;
  const makePool = props.makePool ?? defaultPool;
  const poolRef = useRef<PoolLike | null>(null);
  const pool = () => (poolRef.current ??= makePool());
  useEffect(() => () => poolRef.current?.dispose(), []);

  const [seedText, setSeedText] = useState("");
  const [character, setCharacter] = useState<string>("Ironclad");
  const [version, setVersion] = useState<string>("v0.111.0");
  const [ascension, setAscension] = useState<number>(TRAINED_ASCENSION);
  const conditions: Conditions = { character, ascension, version };
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [running, setRunning] = useState<"check" | "rate" | null>(null);
  const [live, setLive] = useState<AttemptResult[]>([]);
  const [startedAt, setStartedAt] = useState(0);
  const [now, setNow] = useState(0);
  // The check sequence so far (kept for "keep searching") and where it resumes.
  const checkResults = useRef<AttemptResult[]>([]);
  const [nextK, setNextK] = useState<number | null>(null);
  const pending = useRef<number[]>([]);
  const [check, setCheck] = useState<Outcome | null>(null);
  const [rate, setRate] = useState<Outcome | null>(null);
  const [copied, setCopied] = useState<"check" | "rate" | null>(null);

  // Preview the seed as it is typed (debounced).
  useEffect(() => {
    const s = seedText.trim();
    setPreview(null);
    setPreviewError(null);
    setCheck(null);
    setRate(null);
    setNextK(null);
    checkResults.current = [];
    pending.current = [];
    if (!s) return;
    // A reply for an older text must not land on the current one.
    let current = true;
    const t = setTimeout(() => {
      api.preview(s, ascension).then(
        (p) => current && setPreview(p),
        (e: unknown) => current && setPreviewError(e instanceof Error ? e.message : String(e)),
      );
    }, 300);
    return () => {
      current = false;
      clearTimeout(t);
    };
  }, [seedText, ascension, api]);

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(t);
  }, [running]);

  const seed = preview?.seed ?? null;

  const save = useCallback(
    async (r: AttemptResult | null): Promise<string | null> => {
      if (!r || !seed) return null;
      try {
        return (await api.saveLog(seed, ascension, r.classes, r.seq, r.k)).path;
      } catch {
        return null;
      }
    },
    [api, seed, ascension],
  );

  async function runCheck(startK: number, resume: number[] = []) {
    if (!seed || running) return;
    setRunning("check");
    setStartedAt(performance.now());
    setNow(performance.now());
    const before = startK === 0 ? [] : checkResults.current;
    setLive(before);
    const job: Job = {
      seed,
      seq: "check",
      ascension,
      startK,
      pending: resume,
      stopOnWin: true,
      maxAttempts: Infinity,
      capMs: CHECK_CAP_MS,
    };
    const res = await pool().run(job, (rs) => setLive(uniqueByK([...before, ...rs])));
    const all = uniqueByK([...before, ...res.results]);
    checkResults.current = all;
    pending.current = res.pending;
    setNextK(res.nextK);
    const t = tally(all);
    const wins = all.filter((r) => r.won).sort((a, b) => a.k - b.k);
    const firstWin = wins[0] ?? null;
    const deaths = deathsByFloor(all);
    // The line to follow: the winning attempt's, else the one that got furthest.
    const line = firstWin ?? t.best;
    const summary = line ? await api.summary(seed, ascension, line.classes).catch(() => null) : null;
    const logPath = await save(line);
    setCheck({
      verdict: checkVerdict({ seed, build: BUILD, conditions, tally: t, firstWin, summary, deaths }),
      logPath,
      logLabel: firstWin ? "Open the winning run in the replay viewer" : "Open the best attempt in the replay viewer",
      deaths,
      reason: res.reason,
      errors: res.errors,
    });
    setRunning(null);
  }

  async function runRate() {
    if (!seed || running) return;
    setRunning("rate");
    setStartedAt(performance.now());
    setNow(performance.now());
    setLive([]);
    const job: Job = {
      seed,
      seq: "rate",
      ascension,
      startK: 0,
      stopOnWin: false,
      maxAttempts: RATE_RUNS,
      capMs: Infinity,
    };
    const res = await pool().run(job, (rs) => setLive([...rs]));
    const t = tally(res.results);
    const deaths = deathsByFloor(res.results);
    const n = t.played - t.errored;
    const firstWin = res.results.filter((r) => r.won).sort((a, b) => a.k - b.k)[0] ?? null;
    setRate({
      verdict: rateVerdict({ seed, build: BUILD, conditions, wins: t.wins, n, deaths, errored: t.errored }),
      logPath: await save(firstWin),
      logLabel: "Open a winning sampled run in the replay viewer",
      deaths,
      reason: res.reason,
      errors: res.errors,
    });
    setRunning(null);
  }

  const progress = useMemo(() => tally(live), [live]);

  async function copy(which: "check" | "rate", text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* the textarea stays selectable */
    }
  }

  return (
    <div className="sc-app">
      <header className="sc-header">
        <h1>🎲 Seed check</h1>
        <a href="/">← workbench</a>
      </header>
      <p className="sc-blurb">
        Paste a game seed. Instinct (the end-to-end bot) plays it until it wins, first greedily,
        then sampling its moves, in {makePool === defaultPool ? defaultPoolSize() : "several"} parallel workers for up to{" "}
        {CHECK_CAP_MS / 60_000} minutes. You get a verdict to paste into a thread and the run to replay. Results are from
        the simulator.
      </p>

      <section className="sc-panel">
        <div className="sc-row sc-options">
          <label>
            Character
            <select
              aria-label="character"
              value={character}
              disabled={running !== null}
              onChange={(e) => setCharacter(e.target.value)}
            >
              {CHARACTERS.map((c) => (
                <option key={c.id} value={c.id} disabled={!c.enabled}>
                  {c.id}
                  {c.enabled ? "" : " (not yet)"}
                </option>
              ))}
            </select>
          </label>
          <label>
            Version
            <select
              aria-label="version"
              value={version}
              disabled={running !== null}
              onChange={(e) => setVersion(e.target.value)}
            >
              {VERSIONS.map((v) => (
                <option key={v.id} value={v.id} disabled={!v.enabled}>
                  {v.id}
                  {v.enabled ? "" : " (not yet)"}
                </option>
              ))}
            </select>
          </label>
          <label>
            Ascension
            <select
              aria-label="ascension"
              value={ascension}
              disabled={running !== null}
              onChange={(e) => setAscension(Number(e.target.value))}
            >
              {ASCENSIONS.map((a) => (
                <option key={a} value={a}>
                  A{a}
                </option>
              ))}
            </select>
          </label>
        </div>
        {ascension !== TRAINED_ASCENSION && (
          <div className="sc-note">Instinct trained at A{TRAINED_ASCENSION}; it plays A{ascension} with the same net.</div>
        )}
        <div className="sc-label">Seed</div>
        <div className="sc-row">
          <input
            className="sc-seed"
            aria-label="seed"
            value={seedText}
            disabled={running !== null}
            placeholder="1SNVTCEXKR0Q"
            size={14}
            autoCapitalize="characters"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            onChange={(e) => setSeedText(e.target.value)}
          />
          <button type="button" disabled={!seed || running !== null} onClick={() => void runCheck(0)}>
            Check
          </button>
          <button type="button" disabled={!seed || running !== null} onClick={() => void runRate()}>
            Win rate ({RATE_RUNS} runs)
          </button>
          <button type="button" disabled={running === null} onClick={() => pool().stop()}>
            Stop
          </button>
        </div>
        {previewError && <div className="sc-error">{previewError}</div>}
        {preview && (
          <>
            {!preview.length_ok && (
              <div className="sc-warn">Game seeds are 12 characters; this one is {preview.seed.length}.</div>
            )}
            <dl className="sc-kv">
              <dt>Seed</dt>
              <dd className="mono">{preview.seed}</dd>
              <dt>Neow offers</dt>
              <dd>{preview.neow.join(" · ")}</dd>
              <dt>Act 1</dt>
              <dd>{preview.act1}</dd>
              <dt>Bosses</dt>
              <dd>{preview.bosses.map((b) => b.join(" + ")).join(" → ")}</dd>
            </dl>
          </>
        )}
      </section>

      {(running || live.length > 0) && (
        <section className="sc-panel" aria-label="progress">
          <div className="sc-label">{running === "rate" || (!running && rate && !check) ? "Win rate" : "Check"}</div>
          <div className="sc-progress">
            {progress.played} attempts · {progress.wins} won · {progress.errored} errored · best floor{" "}
            {progress.bestFloor}
            {running && (
              <>
                {" "}
                · {fmtSecs(now - startedAt)}
                {running === "check" ? ` / ${fmtSecs(CHECK_CAP_MS)}` : ` · ${progress.played}/${RATE_RUNS}`}
              </>
            )}
          </div>
        </section>
      )}

      {check && (
        <Result
          title="Verdict"
          outcome={check}
          copied={copied === "check"}
          onCopy={() => void copy("check", check.verdict)}
          extra={
            check.reason !== "won" && check.reason !== "failed" && nextK !== null ? (
              <button type="button" disabled={running !== null} onClick={() => void runCheck(nextK, pending.current)}>
                Keep searching ({CHECK_CAP_MS / 60_000} more minutes)
              </button>
            ) : null
          }
        />
      )}
      {rate && (
        <Result
          title="Win rate"
          outcome={rate}
          copied={copied === "rate"}
          onCopy={() => void copy("rate", rate.verdict)}
          extra={null}
        />
      )}
    </div>
  );
}

function Result(props: {
  title: string;
  outcome: Outcome;
  copied: boolean;
  onCopy: () => void;
  extra: React.ReactNode;
}) {
  const { outcome } = props;
  return (
    <section className="sc-panel" aria-label={props.title}>
      <div className="sc-label">{props.title}</div>
      <textarea className="sc-verdict" readOnly value={outcome.verdict} aria-label={`${props.title} text`} />
      <div className="sc-row">
        <button type="button" onClick={props.onCopy}>
          {props.copied ? "Copied" : "Copy"}
        </button>
        {outcome.logPath && (
          <a href={`/?open=${encodeURIComponent(outcome.logPath)}`} target="_blank" rel="noopener">
            {outcome.logLabel} ↗
          </a>
        )}
        {props.extra}
      </div>
      {outcome.reason === "failed" && <div className="sc-error">The check stopped: {outcome.errors.join("; ")}</div>}
      {outcome.reason !== "failed" && outcome.errors.length > 0 && (
        <div className="sc-warn">Some workers failed: {outcome.errors.join("; ")}</div>
      )}
      {outcome.deaths.length > 0 && (
        <table className="sc-deaths">
          <thead>
            <tr>
              <th>Floor</th>
              <th>Died to</th>
              <th>Runs</th>
            </tr>
          </thead>
          <tbody>
            {outcome.deaths.slice(0, 8).map((d) => (
              <tr key={`${d.floor}|${d.room}`}>
                <td>{d.floor}</td>
                <td>{d.room ?? "unknown"}</td>
                <td>{d.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
