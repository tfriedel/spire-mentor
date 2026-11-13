# Spire Mentor

(Called *Spire Coach* in 0.3.0 and *SimBridge* before that, and kept in
`sts2-bridge/` — older logs, experiment records and installs under
`mods/SpireCoach/` or `mods/SimBridge/` use those names.)

A C# mod for **Slay the Spire 2** (game build **v0.107.1**, Windows) that
connects the real, running game to an external AI over a local TCP socket.
Two ways to use it:

- **Advisor mode** — you play; the mod streams each decision to an AI backend
  and renders its advice in-game (card scores, ranked moves, win chance, map
  route). Switches itself on when the game starts (Windows); untick
  "Turn the coach on when the game starts" in the mod settings to stop that.
- **Autopilot mode** — an external agent plays whole runs by itself
  (used for unattended sweeps and evaluation against the headless
  [`sts2-rs`](../sts2-rs) simulator).

Spire Mentor started as a port of zhiyue's **STS2BridgeMod**
([zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent)) onto game
build v0.107.1. In 2026-10 everything derived from it, and everything that
mirrored the game's own AutoSlay framework, was rewritten clean-room from a
behavioural spec, so the mod carries no code from either (decision D5 in
[`../docs/open-source-plan.md`](../docs/open-source-plan.md)). It is MIT.

## How it works

1. Harmony-patches `NGame.IsReleaseGame()` to `false`, unlocking the game's
   built-in `AutoSlay` debug framework (`ICardSelector` override,
   `NonInteractiveMode`, epoch unlocks).
2. Runs a newline-delimited-JSON TCP server on `127.0.0.1:9002`
   (configurable). [`PROTOCOL.md`](PROTOCOL.md) is the complete wire
   reference: every message type, field, and inbound verb.
3. In Autopilot, `SimAutoSlayer` (a mirror of the game's own `AutoSlayer`)
   drives menus, rooms, and screens, with every decision handler replaced by
   one that asks the connected agent — **never a random fallback**: if no
   agent answers, the mod waits.
4. In Advisor mode nothing is patched into the game flow; the mod watches
   public game events, requests advice over the same socket, and draws it via
   the in-game HUD (`SimHud.cs`).

All three behavior patches self-gate on autopilot actually being engaged
(mode Autopilot + slayer running + agent connected), so an installed but idle
mod leaves the game 100% vanilla.

## Requirements

- Windows, with your own Steam install of Slay the Spire 2 at **v0.107.1**
  (the mod patches game internals and clicks concrete UI scene paths — a game
  update can break it silently).
- .NET 9 SDK (`winget install Microsoft.DotNet.SDK.9`, or the repo root's
  `install-windows-deps.ps1`).
- [Alchyr's BaseLib](https://github.com/Alchyr/BaseLib-StS2) (MIT, v3.4.1
  verified) — it provides the mod settings page, and the build **fails with a
  clear error** if it is missing. The repo root's `install-windows-deps.ps1`
  installs it (hash-checked) into `<game>/mods/BaseLib/`, finding the game
  as the build does. By hand: from the
  [v3.4.1 release](https://github.com/Alchyr/BaseLib-StS2/releases) download
  `BaseLib.dll`, `BaseLib.pck` and `BaseLib.json` into
  `<game>/mods/BaseLib/` (create the folder; `<game>` is the folder holding
  `SlayTheSpire2.exe`, e.g.
  `C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2`).
- For the in-game advisor: the repo's Python side, set up once (below).

## Build and install

Players don't need any of this: the mod is on the
[Steam Workshop](https://steamcommunity.com/sharedfiles/filedetails/?id=3821016481).

```powershell
cd spire-mentor-mod
dotnet build
```

Coming from an older build? Delete `<game>/mods/SimBridge/` or
`<game>/mods/SpireCoach/` first — the game would load both, and only one can
listen on port 9002.

`SpireMentor.csproj` finds the game like `just import-game` does: in every
Steam library listed in Steam's `libraryfolders.vdf` (other drives included),
falling back to `C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2`.
`deploy.ps1` uses the same folder. To pick a specific install, set
`STS2_GAME_DIR` (environment variable or MSBuild property):

```powershell
dotnet build -p:STS2_GAME_DIR="D:/SteamLibrary/steamapps/common/Slay the Spire 2"
```

**One DLL for every game build (GH #800).** The same DLL runs on the stable
build (v0.107.1) and the beta (v0.111.0), whichever one it was compiled
against. A few game members it calls changed signature between the two
(`RunRngSet.Seed` uint → ulong, `Rng.Counter` removed,
`CardSelectCmd.UseSelector` gained a parameter); a direct call would compile
either way and then throw `MissingMethodException` on the other build, so
those go through `SimGameApi.cs`, which looks them up at runtime.
`tests/compile/build_mod.sh` fails if the DLL does not bind against every
build in `sts2_binaries/`. At startup the mod logs the running build and warns
on one outside the supported list. Every
outbound message carries the running build as `game_version`
(`PROTOCOL.md` section 1); the advisor builds its engine for it and refuses a
build it does not support.

**Compile-only on a machine without the game (Linux dev box):**
`tests/run_all.sh` is the local gate: for each of v0.107.1 and v0.111.0 it
compiles the mod against that build's own `sts2.dll` (from
`sts2_binaries/<ver>/data_sts2_windows_x86_64/`, gitignored, found in the main
checkout from a worktree), checks that every game type / method / field the DLL
references binds by signature in that `sts2.dll` (`tests/compile/RefCheck`, and
prints how many would not bind on the other build), and runs the BridgeServer
wire test on the result; then the game-free unit tests (`tests/unit`). Nothing
is deployed. BaseLib is not vendored, so by default `tests/compile/build_mod.sh`
compiles against a compile-only API stub (`tests/compile/BaseLibStub`) holding
just the members `SpireMentorModConfig.cs` uses; set `BASELIB_DIR` to a real
BaseLib v3.4.1 install to compile against that instead. This proves the mod
compiles and binds; it does not run it.

It compiles against `sts2.dll` / `0Harmony.dll` / `GodotSharp.dll` straight
from the game's data dir (plain `Microsoft.NET.Sdk`, no Godot SDK or NuGet
feeds). Every successful build auto-deploys `SpireMentor.dll` +
`SpireMentor.json` + `SpireMentor.conf.template` to `<game>/mods/SpireMentor/`
(skip with `-p:SpireMentorSkipDeploy=true`, e.g. while the game is running and
locking the DLL); `./deploy.ps1` redeploys without rebuilding.

To assemble a self-contained mod zip for another machine (bundling BaseLib
and the packaged advisor backend), see `package.ps1` and
`packaging/README-dist.txt` — and note the D5 distribution caveat above.
The package holds no `.json` but `SpireMentor.json`: the game reads every
`.json` under a mod's folder as a manifest (`ModManager.cs`,
`ReadModsInDirRecursive`), so the workbench's and the advisor's data ship
gzipped as `.json.gz`, and `package.ps1` fails on any it finds.

For the Steam Workshop, `workshop.ps1` stages the same files (without
BaseLib, which the item lists as a required Workshop item) into
`workshop-workspace/` for Mega Crit's
[uploader](https://github.com/megacrit/sts2-mod-uploader); `-Upload` runs
it. The item's title, tags and dependencies are in
`packaging/workshop/workshop.json`, its page text in
`packaging/workshop/description.bbcode`, its preview image in
`packaging/workshop/image.png` (Steam needs it under 1 MB), and its id in
`packaging/workshop/mod_id.txt` once the first upload has created it. To
test the subscribed copy, remove `<game>/mods/SpireMentor/` first: the
game disables a Workshop mod whose id is also in the local mods folder
(`ModManager.cs:296`).

## Configuration (`SpireMentor.conf`)

On first run the mod copies `SpireMentor.conf.template` to `SpireMentor.conf`
next to the DLL. Player-install defaults never let the AI play by itself; the
coach (Advisor mode) switches on at launch unless you untick that in the mod
settings. Fields (authoritative docs: `SimConfig.cs`):

| Field | Default | Meaning |
|---|---|---|
| `port` | `9002` | TCP listen port (loopback only) |
| `auto_start` | `false` | Launch Autopilot at boot (the unattended sweep lane). `false` = Off, or Advisor if the "Turn the coach on when the game starts" setting is ticked (default; Windows only) |
| `advisor_autoplay` | `false` | Testing: once a run is live, switch to Advisor mode and toggle AI autoplay (the F8+F7 path) unattended. Forces `pause_on` empty |
| `advisor_brain` | `null` | `"search"` / `"instinct"`: the advisor brain for this session, overriding the in-game setting (not saved over it) until a player switches brain in game; `null` = the in-game setting. How an unattended run plays with Instinct (`ascension_climb.py --advisor-brain instinct`) |
| `pause_on` | `"unsupported"` | Where F7 stops and hands the decision back (F7 resumes). `unsupported` = decisions the AI has no real policy for (the advisor's `hand_back`); add event ids / `advice_for` values to always play those yourself, e.g. `"unsupported, Neow"`; `""` = never pause. A JSON array works too |
| `character` / `ascension` / `seed` | `"Ironclad"` / `0` / `null` | Run setup for Autopilot; `seed: null` = fresh random seed |
| `num_runs` | `1` | Runs to play in sequence before going idle |
| `resume_run` | `false` | Click Continue instead of starting a new run (recover a run after a game crash) |
| `game_mode` | `"standard"` | Which singleplayer mode Autopilot starts: `"standard"`, `"blitz"` or `"dread"` (v0.113.0+, needs that mode unlocked). How the live test lane plays those modes unattended |
| `speed_patches` | `true` | Faster-than-realtime patches while Autopilot is engaged |
| `fast_mode` | `"normal"`* | Game `FastModeType` at run start: `"instant"` / `"fast"` / `"normal"` |
| `time_scale` | `1.0` | Whole-game `Engine.TimeScale` (0.1–4.0), Autopilot only |
| `agent_timeout_seconds` | `0` | `0` = wait indefinitely for the agent; `>0` = watchdog log cadence while waiting (never a fallback action) |
| `client_run_config` | `false` | Let the TCP client pick character/ascension/seed via the `run_config`/`start_run` handshake (PROTOCOL.md) |
| `debug_player_hp` | `0` | Testing cheat: `>0` sets player HP at run start; never use for measurements |
| `overlay` | `true` | Render autopilot `annotations` (status chip + per-card score labels) in the HUD |
| `ai_partners_menu` | `true` | Experimental (`false` hides it): a "With AI partners" entry in the multiplayer host menu, a co-op run with an AI teammate in this one game process, and "Add AI partner" / "Remove AI partner" tiles in the lobby you host (also a Steam lobby: friends join as usual and play next to the AI; they need no mod, and their game shows the partner without a name). The teammate runs its own advisor (one more advisor process while the run lasts); see `docs/plans/2026-11-05-ai-coop-partners.md` for what it does not handle yet |
| `stall_timeout_seconds` | `300` | Mod-side stall limit replacing the game's 30 s AutoSlay watchdog |
| `run_timeout_minutes` | `360` | Absolute per-run wall-clock cap |

\* the template ships `"normal"` for player installs; the in-code default is
`"instant"` (best for unattended experiments).

## In-game controls

- **Settings page** — main menu → Mod Configuration (BaseLib) → Spire Mentor:
  advisor-mode toggle, game-speed slider, HUD toggles, advisor-backend launch
  settings. Main-menu only.
- **The Spire Mentor badge** (`SimToolbar.cs`, words in `MentorBadge.cs`) —
  one compact badge docked under the right end of the game's top bar (under
  the map / deck / settings buttons; the top-left is the game's relic tray and
  co-op player list). Its colour and word say the mode at a glance (grey Off,
  cyan Coach, green Hints, amber while the AI plays) and what the advisor is
  doing (thinking dots — still with *Reduce combat effects* — starting,
  offline, or the engine-build counter). Click it for a popover that mirrors
  every hotkey, grouped: **Mode** `Off · Coach · Hints · AI plays` (one switch
  for F8 + the style; *AI plays* is F7, and reads *Autopilot* while a
  config-armed Autopilot holds the run — `MentorModes` in `CoachStyle.cs`),
  **Brain** Search / Instinct (where Instinct can play), **This decision**
  Peek (F5) / Let AI pick (F6), **Autoplay** Autobattle (Shift+F7) / AI
  plays until stopped (F7), **Speed** (F9/F10), and **More**: three dropdowns
  (`SimUiKit.MakeMenuButton`), **Rewind ▾** (`⟲ Turn` / `⟲ Room` /
  `⟲⟲ Back`), **Panels ▾** (hide all, log, HUD, help, mid-run **Settings** —
  `SimSettingsPanel.cs` — **Report a bug** and **Panels in their own window**) and **Workbench ▾** (open
  the browser view, copy state). The popover closes on the badge, a click
  elsewhere or Esc (read without swallowing them: the click still reaches the
  game), and stays open or closed across launches. While the AI plays, an
  amber frame runs round the screen edge and an **AI is playing — Stop (F7)**
  button sits next to the badge; **Partner go** sits there too in a combat
  with AI partners. Drag the `⠿` grip to move the badge; double-click it to
  dock it again.
- **Panels in their own window** (`SimFrameLauncher.cs`, `FrameLaunchRules.cs`;
  Windows) — the mod starts the Spire Mentor frame (`../spire-mentor-frame/`,
  shipped in `mods/SpireMentor/SpireMentorFrame/`, .NET Framework 4.8) next
  to the running game: the frame adopts the game's window and lays the
  toolbar, advice, coach and action log out around it, from the bundled
  workbench's `frame.html` on the advisor's control plane (:8079;
  `SpireMentorModConfig.FrameUrl` overrides it, e.g. the Vite dev server).
  `FrameLink.Attach` feeds it. The settings pages' tickbox and the Panels ▾
  entry turn it on, and it reopens on the next start; closing the frame window
  gives the game its window back, puts the panels in again and unticks it. A
  fullscreen game goes windowed first. Framed, fullscreen is the frame's
  (`FrameFullscreen.cs`): its toolbar's **⛶ Fullscreen**, the game's Fullscreen
  tickbox or Alt+Enter move the panels into the game and the frame covers the
  monitor, borderless; the **⛶ Exit fullscreen** button beside the badge goes
  back. The workbench keys (g / p / h / f)
  stay bound only for a game the frame launched (`--spire-mentor-frame-port`).
- **Hotkeys** (`SimHotkeys.cs`) — `F1` help panel (the in-game documentation:
  current mode, every key, file locations), `Ctrl+L` action-log panel (it
  sits under the advice column and follows it until you drag it; a dragged
  position is kept across launches, and right-clicking its header puts it back
  under the column — `LogPanelPlacement.cs`), `Ctrl+H` hides every
  panel down to the Spire Mentor badge (again, or click the badge, to show them), `F4` HUD
  on/off, `F6` AI takes this one decision, `Shift+F6` co-op: every AI
  partner waiting for you plays its turn now (below), `F7` AI plays until you stop,
  `Shift+F7` autobattle (below),
  `F8` turns the AI on / off in the current style (from Autopilot: stops the
  AI and hands you the controls; the badge's Mode switch picks the
  style), `F5` peeks at the AI's next pick in the Coach style,
  `F9`/`F10` game speed −/+ in whole steps, 1x–4x (`SpeedSteps.cs`; the
  settings slider still sets tenths), `Ctrl+Z` undo to the start of the
  combat turn, `Ctrl+Shift+Z` restart the room, `Ctrl+Shift+C` copy state,
  `Ctrl+Shift+B` report a bug. `F2` and `F11` belong to the game.
- **AI partners in the lobby and in combat** (`SimAiPartnerLobby.cs`,
  `SimAiPartnerBrain.cs`, `PartnerCharacters.cs`) — click an AI partner's
  slot in a lobby you host to give it the next character (right-click: the
  previous one); each seat keeps its last pick for the next lobby
  (`AiPartnerCharacters` in the BaseLib config). In combat, a partner whose
  best move needs your setup play (Vulnerable on its target, block on it)
  holds it up to **AI partner holds for your setup play** (default 10 s,
  0–60), briefly (2.5 s) when only other AI partners are still to act. The
  **Partner go** button next to the badge (in a combat with AI partners; `▶` and a
  countdown while one waits) or `Shift+F6` ends every wait for this turn:
  the setup hold and the "waits for your first play / end turn" settings.
- **Autobattle** (`Autobattle.cs`, `Shift+F7` or the badge's
  **Autobattle**) — the workbench's autobattle mode in the real game: the AI
  plays the fights (card picks inside a fight included) and every run-level
  decision — card reward, map, shop, rest site, event, card removal /
  upgrade / transform, boss relic — waits for you; your pick resumes it. At
  each stop a chip says what changed since the last one ("Added Inflame ·
  HP 52/75 (−8) · 14 decisions auto-played"), and after you pick, the
  coach's choice chip says whether you matched the AI (a match shows too,
  in autobattle). In the Coach style nothing is shown before you pick, which
  is the drill; Hints shows the AI's opinion as usual. `pause_on` hand-backs
  inside a fight wait for your move too, instead of stopping autoplay. F7
  stops it like any autoplay. Switching it on at 1x raises the speed to 2x;
  any other speed is yours and stays. Switching it off puts 1x back, unless
  you changed the speed in between.
- **Rewind** (`SimSnapshots.cs` / `SimRestore.cs` / `SimCombatUndo.cs`) —
  the mod snapshots every game save (room entry + post-combat) under
  `%LOCALAPPDATA%\SpireMentor\snapshots\` and can reload any of them
  in-process, bit-exact (RNG counters included). Turn undo replays the
  game's own combat tape on top of the room-entry save; completed fights'
  tapes are persisted next to their snapshots, so PAST combats replay too.
  Protocol §7 (`list_snapshots` / `restore_snapshot` / `undo_turn`) exposes
  the same to the workbench, whose game tree rewinds to any room — or any
  recorded combat turn — and tracks forks.
- **Console** (`F11`) — `spirementor` (status), `spirementor mode off|advisor|hints|coach`,
  `spirementor speed <0.1-4.0>`, `spirementor hud on|off`, `spirementor log`,
  `spirementor bugreport [note...]` (no note: the dialog; a note: send now),
  `spirementor waitmult <f>` / `spirementor animmult <f>` (Autopilot-only
  multipliers).

## Bug reports

**Report a bug** (`Ctrl+Shift+B`, Panels ▾, `spirementor bugreport`;
`SimBugReport.cs`) captures the moment the dialog OPENS — the position the
player saw the bug in, not wherever the run went while they typed — and
builds the workbench's schema-1 bug-report dump (`BugReportDump.cs`, unit
tested) from:

- the Copy-state payload (`state.game_state`), which the workbench's
  **Import state** rebuilds, exact fight included;
- the game's save at the start of this room (`repro.run_save`, from
  `SimSnapshots`), which a rewind can restore;
- the advisor's run log (`repro.run_log`, through the control plane's
  `GET /log`) when the advisor is up and the log is under 8 MiB, so the
  workbench can replay every decision with the advice given at the time;
- tails of the action log, `godot.log` and `advisor_stdout.log` (`logs`),
  with the Windows account name taken out of `\Users\<name>\` paths;
- mod / game version, OS, coach mode and style, the mod's settings
  (`meta`, `settings`).

Every report is written to `%LOCALAPPDATA%\SpireMentor\bugreports\` first.
**Send report** then posts it to the hosted workbench's store, through its
intake `<site>/intake/bugreport` (`sts2-pycli/web/functions/intake/`): the one
path the site's sign-in (Cloudflare Access) leaves open, which takes a post
with no `Origin` and only writes. The upload follows no redirect: a redirect
there is the sign-in page, and the dialog says so and keeps the file. The
reply's link goes on the clipboard and its id into the file name; opening it
needs the sign-in. **Save to file only** sends nothing.

The site is private, so no source file names it. A release build carries it
as assembly metadata: `package.ps1 -BugReportSite <origin>` (default:
`$env:STS2_WORKBENCH_SITE`, else the export in `sts2-pycli/deploy/justfile`)
passes `-p:SpireMentorBugReportSite=…` to the build. The advanced setting
**Bug report site** overrides it. A plain dev build has none and offers only
the file. The mod passes the same site to the advisor as
`--bug-report-site`, so the bundled workbench's **🐞 Report** button
(forwarded by the control plane, `realgame_control.BugReportForward`) lands in
the same store, with the game's latest position added. Reading reports back:
`just bug-reports` / `just bug-report <id>` in `sts2-pycli/`
(`docs/browser-workbench.md`, "Bug reports").

## Advisor backend

The AI on the other end of the socket lives in
[`../sts2-pycli`](../sts2-pycli): an MCTS search for combat plus a trained
run policy for map routes, card rewards, shops, events, and rest sites.
`SimAdvisorProcess.cs` auto-launches it when Advisor mode needs one and no
client is connected, preferring (in order) the command configured on the
settings page, the dev launcher `SimAdvisor.cmd` written next to the DLL at
deploy time (runs `sts2-pycli/python/realgame_advisor.py` via `uv`), and the
packaged torch-free `SimAdvisor` exe bundled by `package.ps1`. Any process
that speaks [`PROTOCOL.md`](PROTOCOL.md) can serve instead — connect it to
`127.0.0.1:<port>` and answer the decision messages.

**Set it up once before the first game launch.** The dev launcher runs
`uv run` in `sts2-pycli/`; on a fresh clone that first syncs the whole
environment — downloads torch and friends and compiles the `sts2_rs` Rust
extension — which takes 10+ minutes, while the game waits for its advisor.
Do it ahead:

```powershell
cd sts2-pycli
uv sync                 # BUILDING-WINDOWS.md §4 if the sts2-py build cannot find Python
```

After a `git pull` that touches Rust code, the next launch rebuilds `sts2_rs`
again (a few minutes); `uv sync` beforehand avoids the wait there too.

## Repository tour

| File | Purpose |
|---|---|
| `MainFile.cs` | `[ModInitializer]` entry point, Harmony patches, logging |
| `BridgeServer.cs` | TCP server: newline-JSON, `request_id` correlation, advice lane |
| `SimConfig.cs` | `SpireMentor.conf` loader — the config reference |
| `SimAutoSlayer.cs` | Autopilot run orchestrator (menu nav, room/screen dispatch) |
| `SimAgentGate.cs` | Blocking "ask the agent, never guess" helper |
| `SimStateExtractor.cs` | Shared full-fidelity state serialization (cards, creatures, intents, relics, map graph) |
| `SimProtocolHelpers.cs` | Stale-state (`state_id`) guard, structured errors, choice parsing |
| `SimCombatHandler.cs`, `SimMapHandler.cs`, `SimNonCombatRoomHandlers.cs`, `SimCardSelector.cs`, `Sim*ScreenHandler.cs` | Per-surface decision handlers |
| `SimAdvisor.cs`, `SimScreenAdvice.cs`, `SimAdvisorAutoplay.cs`, `SimAdvisorProcess.cs` | Advisor mode: observe-and-advise loop, screen describers, F7 autoplay, backend auto-launch |
| `SimHud.cs`, `SimActionLog.cs`, `SimHelp.cs`, `SimHotkeys.cs`, `SimConsoleCommands.cs`, `SpireMentorModConfig.cs` | In-game UI and controls |
| `SimMode.cs`, `SimSpeed.cs`, `SimShutdown.cs`, `SimJobObject.cs` | Runtime mode/speed state, clean shutdown, child-process containment |
| `SpireMentor.json` | Mod manifest (`dependencies: [{ id: "BaseLib", min_version: "3.4.7" }]`, `affects_gameplay: true`, `min_game_version: "0.107.1"`) |
| `PROTOCOL.md` | Full wire-protocol reference |
| `tools/` | Unattended drivers and log graders: `ascension_climb.py` (drive the advisor lane up the ascension ladder), `check_f7_log.py` (assert known stall signatures in a game log) |

How the mod fits the rest of the stack: [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## The workbench link

The badge's **More › Workbench ▾ → Open workbench** opens the *Workbench URL* setting.
Blank (the default) tries two pages in order (`SimWorkbenchLink.cs`):

1. **The bundled workbench**, `http://127.0.0.1:8079/?follow=1`. The release zip
   ships the browser-only build in `mods/SpireMentor/workbench/`
   (`package.ps1` builds it with `VITE_BRIDGE_SAME_ORIGIN=1`, with the art
   and sound of the local `just import-game` tree unless `-NoArt`), and the
   advisor's control plane hosts it next to its JSON endpoints
   (`--workbench-dir`; in a dev checkout, `sts2-pycli/web/dist` whenever
   that is a static build). Page and control plane share an origin, so the
   live game's tree, live search and rewind work with no opt-in, and
   `?follow=1` follows the advisor's run log as on the dev workbench (the
   page mirrors it from the control plane's `GET /log`, `web/src/liveLog.ts`).
   It answers `GET /api/health` only while it has a build to serve.
2. **The dev workbench**, `http://127.0.0.1:8077/?follow=1`: the
   Python-backed one, which the advisor starts on its own in dev installs.
   It serves whatever is built in `sts2-pycli/web/dist`:

- **Server build** — `cd sts2-pycli/web; npm run build`. `?follow=1` opens
  the advisor's live run log in the replay view and follows it.
- **Browser-only build** — what the repo root's `just workbench-local`
  builds (and serves on `http://127.0.0.1:4173`). It has no server to proxy
  for it, so it talks to the advisor's control plane (`127.0.0.1:8079`)
  straight from the page: it can show the live game's tree, run a live search
  and rewind the game. That is off unless you allow the page's origin: set
  **Browser workbench origin** in the mod's settings to it (e.g.
  `http://127.0.0.1:4173`, and set **Workbench URL** to the same page), which
  the mod passes to the advisor as `--control-allow-origin`; the control
  plane then answers that origin with the CORS / Private Network Access
  headers and nothing else changes. In the page, Tools → **Live game…** takes
  the control plane's URL. Chrome asks once for local-network access; Firefox
  does not. Details and the browser measurements: `docs/browser-workbench.md`,
  "Live-game bridge".
