import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Catalog, type SweepListEntry } from "./api";
import { EMPTY_CATALOG } from "./catalog";
import AnalysisPanel from "./AnalysisPanel";
import History from "./History";
import PositionEditor, { type NewPositionRequest } from "./PositionEditor";
import ResultsView from "./ResultsView";
import StudiesForm from "./StudiesForm";
import { decodeHash, emptyPosition, encodeHash, formatPosition, sweepFromSearch, type Position } from "./position";
import { usePanelChoice } from "../panelPrefs";
import { LEGACY_GAME_VERSION, NO_VERSION_PICK, effectiveGameVersion, gameVersionBody, gameVersionLabel } from "../gameVersion";

// The distribution sweeps run on both hosts since GH #707 (the wasm steps
// them between requests); the flag stays so a host could switch them off
// through `capabilities.features.simlab_sweeps` again.
const SWEEPS_ENABLED = true;

type Tab = "sandbox" | "studies";

/** The encounter a fresh tab opens on when the URL carries no position. */
const DEFAULT_ENCOUNTER = "FUZZY_WURM_CRAWLER_WEAK";

/**
 * SimLab: a position sandbox (one exact fight, edited and played out) and the
 * older batch studies (deck variants × encounters × ascensions). A position
 * travels in the URL hash (`#p=…`) so the workbench's "SimLab from here"
 * opens straight on the board, and "Copy link" shares it.
 */
export default function App() {
  const [tab, setTab] = useState<Tab>("sandbox");
  const [catalog, setCatalog] = useState<Catalog>(EMPTY_CATALOG);
  const [position, setPosition] = useState<Position | null>(() => decodeHash(window.location.hash));
  const [flash, setFlash] = useState<string | null>(null);
  const [sweeps, setSweeps] = useState<SweepListEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(() => sweepFromSearch(window.location.search));
  const [busy, setBusy] = useState(false);
  const flashTimer = useRef<number | null>(null);
  // GH #877: the game build a new fight is dealt in -- the workbench run
  // launcher's own remembered pick (same preference key), so SimLab plays the
  // build the selector names, like /api/live/start. Until the server's list
  // lands the pick is taken as-is; a pick the server cannot run then falls
  // back to its default (the pick itself is left alone).
  const [gameVersionPick, setGameVersionPick] = usePanelChoice("wb.gameVersion", NO_VERSION_PICK);
  const [gameVersions, setGameVersions] = useState<string[] | null>(null);
  const [serverDefaultVersion, setServerDefaultVersion] = useState(LEGACY_GAME_VERSION);
  const gameVersion = gameVersions
    ? effectiveGameVersion(gameVersionPick, gameVersions, serverDefaultVersion)
    : gameVersionPick;
  // The board on screen names its own build; the catalog follows it.
  const shownVersion = position ? (position.game_version ?? LEGACY_GAME_VERSION) : gameVersion;
  useEffect(() => {
    let cancelled = false;
    api
      .capabilities()
      .then((c) => {
        if (cancelled) return;
        setGameVersions(c.game_versions?.length ? c.game_versions : [LEGACY_GAME_VERSION]);
        if (c.default_game_version) setServerDefaultVersion(c.default_game_version);
      })
      .catch(() => {
        if (!cancelled) setGameVersions([LEGACY_GAME_VERSION]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const say = useCallback((msg: string) => {
    setFlash(msg);
    if (flashTimer.current) window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setFlash(null), 4000);
  }, []);

  // Catalog once per build; the editor renders with bare ids until it lands.
  useEffect(() => {
    let cancelled = false;
    api
      .catalog(shownVersion)
      .then((c) => {
        if (!cancelled) setCatalog(c);
      })
      .catch(() => {
        // ids still work without names
      });
    return () => {
      cancelled = true;
    };
  }, [shownVersion]);

  // A tab opened with no position gets a real fight rather than a blank form.
  useEffect(() => {
    if (position) return;
    let cancelled = false;
    api
      .newPosition({ character: "Ironclad", encounter: DEFAULT_ENCOUNTER, ascension: 0, seed: 1, ...gameVersionBody(gameVersion) })
      .then((b) => {
        if (!cancelled) setPosition(b.position);
      })
      .catch(() => {
        if (!cancelled) setPosition(emptyPosition());
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the URL in step with the position (replaceState: no history spam),
  // and follow a hash someone else set (a second "SimLab from here").
  useEffect(() => {
    if (!position) return;
    const hash = "#" + encodeHash(position);
    if (window.location.hash !== hash) window.history.replaceState(null, "", window.location.pathname + window.location.search + hash);
  }, [position]);
  useEffect(() => {
    const onHash = () => {
      const p = decodeHash(window.location.hash);
      if (p) setPosition(p);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const { sweeps } = await api.list();
      setSweeps(sweeps);
    } catch {
      // ignore transient list failures
    }
  }, []);
  useEffect(() => {
    if (!SWEEPS_ENABLED) return;
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const onSubmitted = useCallback(
    (id: string) => {
      setSelected(id);
      refresh();
    },
    [refresh],
  );

  const onNew = useCallback(
    async (req: NewPositionRequest, version: string = gameVersion) => {
      setBusy(true);
      try {
        const b = await api.newPosition({
          character: req.character, encounter: req.encounter, ascension: req.ascension, seed: req.seed,
          ...gameVersionBody(version),
        });
        setPosition(b.position);
        if (b.notes.length) say(b.notes.join(" "));
      } catch (e) {
        say(`could not build that fight: ${(e as Error).message ?? e}`);
      } finally {
        setBusy(false);
      }
    },
    [say, gameVersion],
  );

  // Switching the build re-deals the same fight in it (and remembers the
  // pick for the launcher too). A hand-built lineup has no encounter to
  // re-deal, so it just takes the build from the next new fight.
  const onGameVersion = useCallback(
    (v: string) => {
      setGameVersionPick(v);
      if (position?.encounter && !position.encounter.startsWith("monsters:")) {
        void onNew({ character: position.character, encounter: position.encounter, ascension: position.ascension, seed: position.seed }, v);
      }
    },
    [setGameVersionPick, position, onNew],
  );

  const onImportText = useCallback(
    async (text: string) => {
      setBusy(true);
      try {
        const r = await api.importPosition(text);
        setPosition(r.position);
        say(`loaded a ${r.source} paste${r.notes.length ? ": " + r.notes.join(" ") : ""}`);
      } catch (e) {
        say(`could not read that paste: ${(e as Error).message ?? e}`);
      } finally {
        setBusy(false);
      }
    },
    [say],
  );

  const copy = useCallback(
    async (what: "link" | "json") => {
      if (!position) return;
      const text = what === "link" ? `${window.location.origin}${window.location.pathname}#${encodeHash(position)}` : formatPosition(position);
      try {
        await navigator.clipboard.writeText(text);
        say(what === "link" ? "link copied" : "position JSON copied");
      } catch {
        say("clipboard is not available here — copy from the Position JSON box");
      }
    },
    [position, say],
  );

  const selectedName = sweeps.find((s) => s.sweep_id === selected)?.spec.name;

  return (
    <div className="sl-app">
      <div className="sl-header">
        <h1>SimLab</h1>
        <a href="/" title="Back to the STS2 Workbench">
          ← Workbench
        </a>
        <div className="sl-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === "sandbox"} onClick={() => setTab("sandbox")}>
            Sandbox
          </button>
          {SWEEPS_ENABLED && (
            <button type="button" role="tab" aria-selected={tab === "studies"} onClick={() => setTab("studies")}>
              Studies
            </button>
          )}
        </div>
        <div className="sl-header-right">
          {flash && <span className="sl-muted">{flash}</span>}
          {gameVersions && gameVersions.length > 1 && (
            <select aria-label="game version" value={shownVersion} onChange={(e) => onGameVersion(e.target.value)}
              title="the game build this fight plays (the workbench run launcher's pick)">
              {gameVersions.map((v) => (
                <option key={v} value={v}>{gameVersionLabel(v)}</option>
              ))}
            </select>
          )}
          {tab === "sandbox" && position && (
            <>
              <button type="button" onClick={() => copy("link")} title="a link that opens SimLab on this exact position">
                Copy link
              </button>
              <button type="button" onClick={() => copy("json")} title="the position as JSON">
                Copy JSON
              </button>
            </>
          )}
        </div>
      </div>

      {tab === "sandbox" &&
        (position ? (
          <div className="sl-grid">
            <PositionEditor position={position} catalog={catalog} onChange={setPosition} onNew={onNew} onImportText={onImportText} busy={busy} />
            <div>
              <AnalysisPanel position={position} catalog={catalog} onSubmitted={onSubmitted} onNormalized={setPosition} sweepsEnabled={SWEEPS_ENABLED} />
            </div>
          </div>
        ) : (
          <div className="sl-panel sl-muted">Dealing a fight…</div>
        ))}
      {tab === "studies" && SWEEPS_ENABLED && <StudiesForm onSubmitted={onSubmitted} />}

      {selected && SWEEPS_ENABLED && <ResultsView id={selected} name={selectedName} />}
      {SWEEPS_ENABLED && (
        <div className="sl-history-wrap">
          <History sweeps={sweeps} selected={selected} onSelect={setSelected} />
        </div>
      )}
    </div>
  );
}
