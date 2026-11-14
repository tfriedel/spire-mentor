import { useEffect, useId, useState, type CSSProperties } from "react";
import { assetUrl, type Advisor } from "../api";
import { e2eAdvisable } from "../advisor";
import { characterArt, characterInBuild, characterKey } from "../characters";
import { ASCENSION_LEVELS, ascensionLevel, ascensionModifiers } from "../ascension";
import { canonicalizeSeed, generateSeed } from "../seed";
import { DEFAULT_GAME_VERSION, gameVersionLabel } from "../gameVersion";
import CombatBudget, { type CombatPreset } from "./CombatBudget";
import RichText from "./RichText";
import "../skins/runend.css";
import "../skins/launcher.css";

/** The playable characters, in the game's roster order. The Fool is
 *  v0.113.0-only: the launcher offers it only for a build that has it. */
export const CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Fool"] as const;

const ENGINE_LABELS: Record<string, string> = { rust: "Rust", csharp: "C#" };

export type OffCombatBrain = "heuristic" | "ppo" | "e2e";

/** The e2e student plays only the run it was trained on. */
export { E2E_CHARACTER, E2E_GAME_VERSION } from "../advisor";

/**
 * The start screen: pick a character, seed and ascension, then choose HOW the
 * run is played — by you, or by the run AI while you watch.
 *
 * Modelled on the game's own lobby rather than a settings bar: portrait tiles,
 * a game-style seed that is already rolled (with a re-roll, the way the game
 * shows a seed before you commit), the ascension ladder with each level's
 * real name, and the two launch actions each explained in a sentence. The
 * dev-only knobs (act-1 variant, engine) sit in a muted row at the foot.
 *
 * The seed is rolled here, on the client, so the run you start is the run you
 * can read off the screen — an empty field would have meant "whatever the
 * server picks", and the old form's empty default was the fixed seed `test`.
 */
export default function RunLauncher({
  character, onCharacter,
  seed, onSeed,
  ascension, onAscension,
  act1, onAct1,
  backend, onBackend, backends,
  gameVersion = DEFAULT_GAME_VERSION, onGameVersion, gameVersions = [],
  onPlay, onWatch, busy, error,
  offCombatBrain, onOffCombatBrain, e2eAvailable = false,
  advisor = "search", onAdvisor,
  combat,
}: {
  character: string;
  onCharacter: (c: string) => void;
  seed: string;
  onSeed: (s: string) => void;
  ascension: number;
  onAscension: (a: number) => void;
  act1: string;
  onAct1: (a: string) => void;
  backend: string;
  onBackend: (b: string) => void;
  backends: string[];
  /** GH #801: the game build the run plays, from the server's
   *  `game_versions`. The select only shows when there are two or more to
   *  choose between. */
  gameVersion?: string;
  onGameVersion?: (v: string) => void;
  gameVersions?: string[];
  /** "Play yourself": start an interactive run. */
  onPlay: () => void;
  /** "Watch the AI play": start a run and let the run agent play it out.
   *  Omitted when the server does not advertise run mode. */
  onWatch?: () => void;
  /** A start / autoplay is in flight — both launchers are disabled. */
  busy: boolean;
  error: string | null;
  offCombatBrain: OffCombatBrain;
  onOffCombatBrain: (b: OffCombatBrain) => void;
  /** The engine has the e2e student brain (`features.e2e`). */
  e2eAvailable?: boolean;
  /** Who advises the hand-played run ("Play yourself"). The picker shows
   *  only where the e2e student can advise (Ironclad, v0.111.0, Rust). */
  advisor?: Advisor;
  onAdvisor?: (a: Advisor) => void;
  combat: {
    preset: CombatPreset;
    monsterSims: number;
    bossSims: number;
    threads: number;
    onPreset: (p: CombatPreset) => void;
    onMonsterSims: (n: number) => void;
    onBossSims: (n: number) => void;
  };
}) {
  const ids = useId();
  // Roll a seed the moment the screen appears with none, like the game's lobby.
  useEffect(() => {
    if (!seed) onSeed(generateSeed());
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount only
  }, []);

  const canonical = canonicalizeSeed(seed);
  const slug = characterKey(character);
  // The build this run would play (the version select shows the same value):
  // a character it lacks is not offered, and a pick it lacks falls back.
  const runVersion = backend === "rust" ? gameVersion : DEFAULT_GAME_VERSION;
  useEffect(() => {
    if (!characterInBuild(slug, runVersion)) onCharacter(CHARACTERS[0]);
  }, [slug, runVersion, onCharacter]);
  const level = ascensionLevel(ascension);
  const stacked = ascensionModifiers(ascension);
  const [heroOk, setHeroOk] = useState(true);
  useEffect(() => setHeroOk(true), [slug]);

  return (
    <div className="gs-scene gs-scene--plain gs-launch" data-testid="run-launcher">
      <div className="gs-launch-frame">
        <header className="gs-launch-head">
          <h1 className="gs-launch-title">New run</h1>
          <p className="gs-launch-sub">Pick a character, a seed and an ascension, then decide who plays.</p>
        </header>

        <div className="gs-launch-grid">
          <div className="gs-launch-setup">
            {/* ---- character ---- */}
            <section className="gs-launch-field">
              <h2 className="gs-launch-label" id={`${ids}-char`}>Character</h2>
              <div className="gs-launch-chars" role="radiogroup" aria-labelledby={`${ids}-char`}>
                {CHARACTERS.filter((ch) => characterInBuild(characterKey(ch), runVersion)).map((ch) => {
                  const k = characterKey(ch);
                  const on = ch === character;
                  return (
                    <button
                      key={ch} type="button" role="radio" aria-checked={on}
                      className={`gs-launch-char${on ? " is-on" : ""}`}
                      onClick={() => onCharacter(ch)} disabled={busy}
                    >
                      <span className="gs-launch-charface">
                        {k
                          ? <img src={assetUrl(characterArt(k, `ui/top_panel/character_icon_${k}.png`))} alt="" draggable={false} />
                          : <b>{ch[0]}</b>}
                      </span>
                      <span className="gs-launch-charname">{ch}</span>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* ---- seed ---- */}
            <section className="gs-launch-field">
              <label className="gs-launch-label" htmlFor={`${ids}-seed`}>Seed</label>
              <div className="gs-launch-seedrow">
                <input
                  id={`${ids}-seed`} className="gs-launch-seed mono" aria-label="seed"
                  value={seed} onChange={(e) => onSeed(e.target.value)}
                  placeholder="random" spellCheck={false} autoComplete="off"
                  maxLength={64} disabled={busy}
                />
                <button type="button" className="gs-launch-reroll" onClick={() => onSeed(generateSeed())}
                  disabled={busy} title="roll a new seed" aria-label="new seed">
                  <span aria-hidden="true">⟳</span> New seed
                </button>
              </div>
              <p className="gs-launch-hint">
                {seed.trim() === ""
                  ? "Empty rolls a fresh seed when the run starts."
                  : canonical !== seed
                    ? <>Runs as <b className="mono">{canonical}</b> — the game upper-cases seeds and reads O as 0 and I as 1.</>
                    : "The same seed replays the same map, rewards and fights."}
              </p>
            </section>

            {/* ---- ascension ---- */}
            <section className="gs-launch-field">
              <h2 className="gs-launch-label" id={`${ids}-asc`}>Ascension</h2>
              <div className="gs-launch-ladder" role="radiogroup" aria-labelledby={`${ids}-asc`}>
                {ASCENSION_LEVELS.map((l) => {
                  const on = l.level === level.level;
                  const lit = l.level <= level.level;
                  return (
                    <button
                      key={l.level} type="button" role="radio" aria-checked={on}
                      aria-label={`ascension ${l.level}`} title={l.level === 0 ? l.title : `A${l.level} · ${l.title}`}
                      className={`gs-launch-rung${on ? " is-on" : ""}${lit ? " is-lit" : ""}`}
                      onClick={() => onAscension(l.level)} disabled={busy}
                    >
                      {l.level}
                    </button>
                  );
                })}
              </div>
              <div className="gs-launch-asc">
                <span className="gs-launch-ascbadge" aria-hidden="true">
                  <FlameIcon dim={level.level === 0} /><b>{level.level}</b>
                </span>
                <div className="gs-launch-asctext">
                  <div className="gs-launch-asctitle">
                    {level.level === 0 ? level.title : `Ascension ${level.level} · ${level.title}`}
                  </div>
                  <ul className="gs-launch-mods">
                    {level.level === 0
                      ? <li><RichText text={level.description} /></li>
                      : stacked.map((m) => (
                        <li key={m.level} className={m.level === level.level ? "is-top" : undefined}>
                          <RichText text={m.description} />
                        </li>
                      ))}
                  </ul>
                </div>
              </div>
            </section>
          </div>

          {/* ---- hero ---- */}
          <div className="gs-launch-hero" aria-hidden="true">
            {slug && heroOk && (
              <img
                key={slug}
                src={assetUrl(characterArt(slug, `character_sprites/${slug}.png`))}
                alt="" draggable={false}
                onError={() => setHeroOk(false)}
              />
            )}
            <div className="gs-launch-heroname">{character}</div>
          </div>
        </div>

        {/* ---- the two ways to play ---- */}
        <div className={`gs-launch-actions${onWatch ? "" : " gs-launch-actions--solo"}`}>
          <section className="gs-launch-action gs-launch-action--play">
            <h2 className="gs-launch-actiontitle">Play yourself</h2>
            <p className="gs-launch-actiondesc" id={`${ids}-play`}>
              You make every decision — map, cards, fights, shops. AI hints stay off unless you switch them on.
            </p>
            {onAdvisor && e2eAdvisable(e2eAvailable, character, gameVersion, backend) && (
              <div className="gs-launch-knobs">
                <label title={"Who the AI hints, the policy panel, the choice coach and \"play best\" ask. "
                  + "Search: the combat MCTS in fights, the PPO run policy elsewhere. "
                  + "Instinct: one transformer, no search — for every decision "
                  + "(its ~21 MB net downloads the first time you pick it)."}>
                  advisor{" "}
                  <select aria-label="advisor" value={advisor} disabled={busy}
                    onChange={(e) => onAdvisor(e.target.value as Advisor)}>
                    <option value="search">Search (MCTS + PPO)</option>
                    <option value="e2e">Instinct (PPO only)</option>
                  </select>
                </label>
              </div>
            )}
            <button type="button" className="gs-runend-btn gs-runend-btn--primary gs-launch-go"
              onClick={onPlay} disabled={busy} aria-describedby={`${ids}-play`}>
              <span className="gs-runend-btnface">Play yourself</span>
            </button>
          </section>

          {onWatch && (
            <section className="gs-launch-action gs-launch-action--watch">
              <h2 className="gs-launch-actiontitle">Watch the AI play</h2>
              <p className="gs-launch-actiondesc" id={`${ids}-watch`}>
                The run AI plays the whole run out while you watch. Stop it any time and take over from there.
              </p>
              <div className="gs-launch-knobs">
                <label title={e2eAvailable
                  ? "Brain: Heuristic and PPO drive card rewards / shops / events / map / rest while the combat MCTS agent searches every fight. Instinct is one transformer that makes EVERY decision, fights included, with no search (Ironclad, v0.111.0 only; its ~21 MB net downloads the first time you pick it)."
                  : "Off-combat brain: which policy drives card rewards / shops / events / map / rest. Combat itself is always searched by the combat MCTS agent, regardless of this choice."}>
                  brain{" "}
                  <select
                    aria-label="brain"
                    value={offCombatBrain}
                    disabled={busy}
                    onChange={(e) => onOffCombatBrain(e.target.value as OffCombatBrain)}
                  >
                    <option value="heuristic">Heuristic</option>
                    <option value="ppo">PPO</option>
                    {e2eAvailable && <option value="e2e">Instinct (PPO only)</option>}
                  </select>
                </label>
                {offCombatBrain === "e2e" ? (
                  <span className="gs-launch-hint" data-testid="e2e-note">
                    Ironclad · v0.111.0 · act 1 rolled · no combat search
                  </span>
                ) : (
                  <CombatBudget
                    preset={combat.preset} monsterSims={combat.monsterSims} bossSims={combat.bossSims}
                    threads={combat.threads} disabled={busy}
                    onPreset={combat.onPreset} onMonsterSims={combat.onMonsterSims} onBossSims={combat.onBossSims}
                  />
                )}
              </div>
              <button type="button" className="gs-runend-btn gs-runend-btn--secondary gs-launch-go"
                onClick={onWatch} disabled={busy} aria-describedby={`${ids}-watch`}>
                <span className="gs-runend-btnface">{busy ? "Running…" : "Watch the AI play"}</span>
              </button>
              <p className="gs-launch-footnote">
                {offCombatBrain === "e2e"
                  ? "Instinct plays the run it was trained on: Rust engine, Ironclad, v0.111.0, the rolled act 1."
                  : "AI runs always use the Rust engine and the Overgrowth act 1."}
              </p>
            </section>
          )}
        </div>

        {/* The start screen carries its own copy of the error. The shared
            banner lives in the right-hand log panel, which is the last place
            anyone looks while staring at a button that appears to do nothing. */}
        {error && <p className="gs-launch-error" role="alert">⚠ {error}</p>}

        {/* ---- dev knobs ---- */}
        <div className="gs-launch-advanced">
          <span className="gs-launch-advlabel">Advanced</span>
          <label title="Which Act 1 to walk. STS2 ships two: Overgrowth and Underdocks. Roll = the real game's seed-driven pick (50/50), which is what makes a seed reproduce your actual run.">
            act 1{" "}
            <select value={act1} onChange={(e) => onAct1(e.target.value)} disabled={busy}>
              <option value="roll">Roll (as the game does)</option>
              <option value="overgrowth">Overgrowth</option>
              <option value="underdocks">Underdocks</option>
            </select>
          </label>
          <label title="Game engine: C# headless or the Rust port">
            engine{" "}
            <select value={backend} onChange={(e) => onBackend(e.target.value)} disabled={busy}>
              {backends.map((b) => (
                <option key={b} value={b}>{ENGINE_LABELS[b] ?? b}</option>
              ))}
            </select>
          </label>
          {gameVersions.length > 1 && onGameVersion && (
            <label title="Which game build the engine plays. v0.111.0 is the beta: its content and RNG seeding differ, so the same seed is a different run.">
              version{" "}
              {/* Only the Rust engine plays a non-default build: with C#
                  the picker shows (and the run starts on) the default. */}
              <select aria-label="version"
                value={backend === "rust" ? gameVersion : DEFAULT_GAME_VERSION}
                onChange={(e) => onGameVersion(e.target.value)}
                disabled={busy || backend !== "rust"}>
                {gameVersions.map((v) => (
                  <option key={v} value={v}>{gameVersionLabel(v)}</option>
                ))}
              </select>
            </label>
          )}
        </div>
      </div>
    </div>
  );
}

/** The HUD's ascension flame, dimmed to embers at A0. */
function FlameIcon({ dim }: { dim: boolean }) {
  const style: CSSProperties = dim ? { filter: "grayscale(0.7) brightness(0.55)" } : {};
  return (
    <svg className="gs-launch-flame" viewBox="0 0 24 28" aria-hidden="true" focusable="false" style={style}>
      <defs>
        <linearGradient id="gsLaunchFlame" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ffd24a" />
          <stop offset="46%" stopColor="#fb7a1c" />
          <stop offset="100%" stopColor="#c8380d" />
        </linearGradient>
      </defs>
      <path
        d="M12.4.7c.4 5.2-3.5 6.7-6 10.4C4.5 13.9 3.3 16.3 3.3 18.9c0 5.1 3.9 8.4 8.7 8.4s8.7-3.3 8.7-8.4c0-4.1-1.7-7.2-4.4-10.3-1.1 2.2-2.2 2.6-3.1 2-1 -3.4-1.3-6.6-.8-9.9Z"
        fill="url(#gsLaunchFlame)" stroke="#63200a" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}
