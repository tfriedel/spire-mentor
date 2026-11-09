# Live-Game Bridge

The C# mod loaded into the running game, exposing its state and accepting
commands over TCP. Compiles with zero sibling files; couples to the
Integration Hub only when deploying or packaging.

Terms shared with the rest of the monorepo live in
[../CONTEXT-MAP.md](../CONTEXT-MAP.md#shared-language).

## Language

**Spire Mentor**:
The mod itself, running inside the real game process. Identifier `SpireMentor`
(mod id, DLL, namespace, conf file, log prefix). Called **Spire Coach** in
0.3.0 and **SimBridge** before that, in a folder named `sts2-bridge/`;
historical logs and experiment records keep those names.
_Avoid_: the mod, the plugin, the server, SimBridge, Spire Coach

**Protocol**:
The TCP message contract between the mod and its clients, specified in `PROTOCOL.md`. The contract is the specification — the client and the mod ship separately and cannot assume a shared version.
_Avoid_: API, interface, wire format

**Autoplay**:
The mode where the bridge acts on the advisor's recommendations itself, without a human in the loop.
_Avoid_: bot mode, auto, AI mode

**Autobattle**:
Autoplay limited to the fights: the AI plays every combat, and every run-level decision waits for the human, with a chip saying what changed since the last one. The in-game twin of the workbench's autobattle mode.
_Avoid_: auto-battler, combat autoplay

**Badge**:
The mod's one in-game control, docked under the right end of the game's top bar (`SimToolbar`): its colour and word show the mode, a click opens the popover with every control. Its **Mode switch** is Off · Coach · Hints · AI plays, where "AI plays" is F7 autoplay in Advisor mode, not the config-armed **Autopilot** (which only shows there, as "Autopilot").
_Avoid_: toolbar, strip, pill (the 0.9 layout)

**Advisor brain**:
Which AI answers the advice, chosen by the player: **Search** (MCTS for the fights, the PPO run policy for the screens; the default) or **Instinct** (the end-to-end net, no search, every decision). Sent as `brain` on every advice request; the advisor decides per request whether Instinct can play it and says so when it cannot (`PROTOCOL.md` section 5, v5.2).
_Avoid_: mode (that is Off / Advisor / Autopilot), style (Hints / Coach)

**AI partner**:
A co-op teammate played by the AI inside the host's own game process: a virtual peer that joins the lobby like a remote client, reads its player off the host's state, and sends only the messages a remote client sends (`SimAiPartner`, docs/plans/2026-11-05-ai-coop-partners.md). Either alone with the player (a session of its own, in-memory) or added to a Steam lobby next to human friends, who need no mod to play with it. Not the **dummy** of the co-op test lane, which is a second game instance on autoplay.
_Avoid_: bot, AI player, fake client

**Bug report**:
A frozen moment sent from the game (`Ctrl+Shift+B`) or a workbench (🐞 Report):
the position, the start-of-room save, the run log and log tails, in the
workbench's schema-1 dump. One store for all of them, keyed `wb-<16 hex>`.
_Avoid_: crash report, feedback (the game's own F2 feedback screen goes to Mega Crit)

**Agent gate**:
The check governing whether an automated client may act on the live game at all.
_Avoid_: guard, lock, permission
