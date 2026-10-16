Spire Mentor — AI coach for Slay the Spire 2
===========================================

WHAT THIS IS
------------
Spire Mentor connects a Slay the Spire 2 game to an AI backend ("SimAdvisor")
that recommends plays: an MCTS search for combat and a trained run policy
for map routes, card rewards, shops, events and rest sites. It can advise
you while you play, or play whole runs itself.

THE COACH SWITCHES ON WHEN THE GAME STARTS, and never plays for you unless
you ask. Press F8 in game to switch it off (the game is then vanilla: no
HUD, no AI, no modified behaviour). To keep it off at launch, untick "Turn
the coach on when the game starts" on the mod settings page.

IMPORTANT — VERSION PIN
-----------------------
Built against game v0.107.1 and BaseLib v3.4.1. A game update can break
this mod SILENTLY (it patches game internals and clicks concrete UI scene
paths). If the game has updated and things misbehave, disable the mod and
wait for a rebuilt release.

INSTALL
-------
1. If you downloaded this zip from the internet, unblock it first:
   right-click the .zip -> Properties -> check "Unblock" -> OK.
   (Skipping this can make Windows quietly block the mod's DLLs.)
2. Extract the zip into the Slay the Spire 2 install folder, e.g.:
     C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2
   You should end up with:
     <game>\mods\SpireMentor\...
     <game>\mods\BaseLib\...
   (If you already have BaseLib installed, keep your copy — versions
   should match, this zip ships v3.4.1.)
3. Start the game. Mods load automatically.

UPGRADING FROM SPIRE COACH (0.3.x) OR SIMBRIDGE (0.2.x)
-------------------------------------------------------
Spire Mentor was called "Spire Coach" in 0.3.0 and "SimBridge" before
that. Before extracting this zip:
  - Delete the old  <game>\mods\SpireCoach\  or  <game>\mods\SimBridge\
    folder. Otherwise the game loads both copies and they fight over the
    same port.
  - Settings are NOT carried over: Spire Mentor starts from its defaults
    in %LOCALAPPDATA%\SpireMentor\SpireMentor.conf. To keep yours, copy
    the values over from the old .conf by hand. The old
    %LOCALAPPDATA%\SpireCoach (or SimBridge) folder, with its snapshots
    and logs, is not used any more; delete it whenever you like.

TURNING IT ON / OFF
-------------------
The game's own Mods screen (main menu) lists Spire Mentor with a tickbox —
untick it and restart to disable the mod entirely; that screen also has a
master switch for all mods. Steam has nothing to do with it: this is not a
Workshop item.

KEYS  (press F1 in game for this list)
--------------------------------------
  F1   help — mode, keys, file locations
  F8   AI coach on / off. From autopilot: stop the AI and take over
  F6   let the AI take just this one decision
  F7   let the AI keep playing until you stop it (press again to stop)
  F4   show / hide the HUD
  F3   action log — what was played, and what was advised at the time
  F9   game speed down 0.1x
  F10  game speed up 0.1x

The HUD draws a score on each card in hand, a ranked list of the moves the
AI considered, a win-chance bar, and markers on the map for the route it
likes. Each can be switched off on the settings page.

THE WORKBENCH
-------------
Toolbar -> Workbench -> Open workbench opens a browser page that follows
your run as you play: every decision with what the AI advised, a run tree
(click a save point to rewind the real game), live search over the board,
replays, a sandbox and more. It is served by the advisor on
http://127.0.0.1:8079, so the AI coach must be on (F8). It runs entirely on
your machine and works offline.

THE SETTINGS PAGE
-----------------
Main menu -> Options -> Mod Configuration -> SpireMentor: mode switch, game
speed, HUD toggles, advisor launch settings. BaseLib deliberately disables
mod configuration while a run is in progress — mid-run, use the keys.

FIRST LAUNCH / WINDOWS WARNINGS
-------------------------------
- SimAdvisor.exe is not code-signed. SmartScreen may warn on first run;
  the mod launches it in the background, so if the advisor never comes up,
  check Windows Security notifications.
- The advisor only starts when you switch the coach on, and it takes a few
  seconds to load its network the first time.
- The mod opens a local TCP port (127.0.0.1:9002 by default) to talk to
  the advisor on the same machine. Nothing leaves your computer.
- Your machine needs the Microsoft Visual C++ runtime (almost every
  Windows install already has it).

WHERE YOUR FILES LIVE
---------------------
Config:  %LOCALAPPDATA%\SpireMentor\SpireMentor.conf
         (created on first launch from the template; edit it to change
         port, autopilot, character, ascension, timeouts, ...)
         If a SpireMentor.conf also sits in <game>\mods\SpireMentor\ (left by
         an older install), THAT one is used and your edits above do
         nothing — delete it. The mod says which file it loaded in
         %APPDATA%\SlayTheSpire2\logs\godot.log.
Logs:    %LOCALAPPDATA%\SpireMentor\advisor_stdout.log
         %LOCALAPPDATA%\SpireMentor\logs\realgame_advisor\*.jsonl
Settings you change on the in-game mod settings page persist in
%APPDATA%\SlayTheSpire2\mod_configs\SpireMentor.cfg.

LICENSING / DO NOT REDISTRIBUTE
-------------------------------
- BaseLib (mods\BaseLib) is by Alchyr, MIT-licensed — see
  mods\BaseLib\LICENSE-BaseLib.txt. https://github.com/Alchyr/BaseLib-StS2
  (Packager note: the license text is not vendored in the SpireMentor repo.
  If LICENSE-BaseLib.txt is missing from this zip, it was built without a
  local BaseLib-StS2 clone; source the MIT notice from the upstream repo
  and rebuild before passing the zip on.)
- SpireMentor is MIT-licensed (spire-mentor-mod/LICENSE in the source
  repository, github.com/tfriedel/spire-mentor). It started as a port of
  zhiyue's "STS2BridgeMod" (github.com/zhiyue/sts2-rl-agent).
- Spire Mentor is an unofficial fan project, not affiliated with or
  endorsed by Mega Crit. Slay the Spire 2 and the art and sound in
  mods\SpireMentor\workbench are (c) Mega Crit, included for use with your
  own copy of the game under Mega Crit's content policy
  (https://www.megacrit.com/content-policy/). Spire Mentor is free and
  must never be sold or paywalled.
