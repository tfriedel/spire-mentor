// SimHelp.cs -- the mod's only self-explanation: an F1 panel and a boot hint.
//
// Everything SpireMentor does is behind a function key or a settings page that
// BaseLib hides while a run is in progress, and the HUD is invisible in mode
// Off -- so a player who installs the zip sees a game that looks completely
// unmodded and has no way to find out otherwise (reported 2026-08-08 by the
// first outside tester). Two surfaces fix that:
//
//   * ShowBootHint() -- a first-launch tip at the bottom of the main menu for a few
//     seconds, saying the mod is loaded and that F1 explains it. It also
//     carries the LOUD case: if autopilot is armed, the player is told that
//     the AI is about to play and which key takes it back.
//   * Toggle() -- the F1 panel: current mode, every hotkey, what the HUD
//     draws, and where the config and logs live.
//
// Same construction rules as SimHud/SimLogPanel: a plain C# class composing
// built-in Godot nodes (no Node subclass -- this project has no Godot source
// generators), its own CanvasLayer, MouseFilter.Ignore throughout so nothing
// here can eat a click meant for the game, and every entry point guarded so
// a render error can never break gameplay.

using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;

namespace SpireMentor;

internal static class SimHelp
{
    // Above SimHud (99) and the action log (100): help is a modal-ish thing,
    // it should never be drawn under the panels it is explaining.
    private const int LayerIndex = 101;

    private const float PanelWidth = 680f;

    private static CanvasLayer? _layer;
    private static PanelContainer? _panel;
    private static RichTextLabel? _text;
    private static PanelContainer? _hint;
    private static bool _bootHintShown;

    /// <summary>F1: show/hide the help panel.</summary>
    public static void Toggle()
    {
        Guarded(() =>
        {
            if (!EnsureBuilt())
            {
                return;
            }
            bool show = !_panel!.Visible;
            if (show)
            {
                _text!.Text = BuildHelpText();
                // Re-measured on every open, not once at build time: the
                // player can change resolution or go fullscreen between the
                // layer being created (at boot) and the panel being asked for.
                Reposition();
            }
            _panel.Visible = show;
        });
    }

    /// <summary>
    /// Wait for the main menu to actually be on screen, then show the hint.
    ///
    /// Not "as soon as the hotkeys are installed", which is where this used to
    /// hang off: the hotkeys go in the moment NHotkeyManager exists, ~40 s of
    /// logo and load screens before the player can see anything (godot.log
    /// 2026-08-08: hotkeys at line 137, "Main menu visible" at line 241). The
    /// hint had come and gone before there was anyone to read it.
    /// </summary>
    public static async Task ShowBootHintWhenReadyAsync()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(300));
        await WaitHelper.Until(() => NGame.Instance != null, cts.Token,
            TimeSpan.FromSeconds(300), "NGame.Instance not available");
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return;
        }
        await WaitHelper.Until(
            () => tree.Root.GetNodeOrNull<Control>("/root/Game/RootSceneContainer/MainMenu")
                      ?.IsVisibleInTree() ?? false,
            cts.Token, TimeSpan.FromSeconds(300), "Main menu not visible");
        ShowBootHint();
    }

    /// <summary>
    /// First-launch greeting, persisted across sessions. The armed-autopilot
    /// warning still appears each launch; F1 remains available for help.
    /// </summary>
    public static void ShowBootHint()
    {
        bool armed = SimRuntime.Mode == SimMode.Autopilot;
        if (_bootHintShown || (!armed && SpireMentorModConfig.StartupTipShown))
        {
            return;
        }
        Guarded(() =>
        {
            if (!EnsureBuilt())
            {
                return;
            }
            var label = _hint!.GetNode<RichTextLabel>("HintText");
            label.Text = armed
                // Never let the AI take a run without saying so. This is the
                // stale-config safety net: installs made before 0.2.1 have
                // auto_start=true sitting in %LOCALAPPDATA%\SpireMentor.
                ? "[color=#ff9d5c][b]Spire Mentor: AUTOPILOT IS ARMED[/b] — the AI will play by itself as soon as " +
                  "the advisor connects. Press [b]F8[/b] to take the controls back, or set " +
                  "[i]auto_start[/i] to false in SpireMentor.conf. [b]F1[/b] for help.[/color]"
                : SimRuntime.CoachStyleActive
                    ? $"[color=#9aa3b8]Spire Mentor {MainFile.ModVersion} loaded — [b]Coach[/b]: nothing shows " +
                      "before you act, [b]F5[/b] peeks at the AI's next pick, and a note afterwards says where " +
                      "you differed. [b]F8[/b] turns the AI off, the [b]Spire Mentor[/b] badge (top right) switches " +
                      "Hints / Coach, [b]F1[/b] for help.[/color]"
                : SimRuntime.Mode == SimMode.Advisor
                    ? $"[color=#9aa3b8]Spire Mentor {MainFile.ModVersion} loaded — [b]Hints[/b]: the AI rates " +
                      "your options while you decide. [b]F8[/b] turns the AI off, the [b]Spire Mentor[/b] badge " +
                      "(top right) switches Hints / Coach, [b]F1[/b] for help.[/color]"
                    : $"[color=#9aa3b8]Spire Mentor {MainFile.ModVersion} loaded — click the [b]Spire Mentor[/b] " +
                      "badge in the top right and pick Coach or Hints to turn the AI coach on, or press [b]F1[/b] " +
                      "for help.[/color]";
            Reposition();
            _hint.Visible = true;
            Logger.Log($"[SimHelp] Boot hint shown (armed={armed}) at {_hint.Position} " +
                       $"size {_hint.Size} label {label.Size} " +
                       $"viewport {((SceneTree)Engine.GetMainLoop()).Root.GetVisibleRect().Size}");

            // Armed autopilot is a warning, not a greeting: leave it up long
            // enough to read twice.
            var timer = _hint.GetNode<Godot.Timer>("HintTimer");
            timer.WaitTime = armed ? 40.0 : 20.0;
            timer.Start();
            _bootHintShown = true;
            if (!armed)
            {
                SpireMentorModConfig.StartupTipShown = true;
                SpireMentorModConfig.SaveSoon();
            }
        });
    }

    private static string BuildHelpText()
    {
        string mode = SimRuntime.Mode switch
        {
            SimMode.Off => "[color=#9aa3b8]Off[/color] — the mod is doing nothing; the game is vanilla",
            SimMode.Advisor when SimRuntime.Style == AdviceStyle.Coach =>
                "[color=#7fd4e8]Coach[/color] — you play; nothing shows before you act (F5 peeks), " +
                "and afterwards a note says where your turn or choice differed from the AI's",
            SimMode.Advisor => "[color=#7fd4e8]Hints[/color] — you play, the AI rates your options while you decide",
            _ => SimRuntime.AutopilotEngaged
                ? "[color=#ff9d5c]Autopilot[/color] — the AI is playing this run"
                : "[color=#ff9d5c]Autopilot (armed, idle)[/color] — the AI plays as soon as it connects",
        };

        var sb = new StringBuilder(1200);
        sb.Append("[b]Spire Mentor ").Append(MainFile.ModVersion).Append("[/b] — AI coach for Slay the Spire 2\n\n");
        sb.Append("Mode: ").Append(mode).Append("\n\n");

        sb.Append("[b]Keys[/b]\n");
        Row(sb, "F8", "turn the AI on / off (from Autopilot: take the controls back); the badge's Mode switch picks Coach or Hints");
        Row(sb, "F5", "Coach: peek at the AI's pick for just the next decision (also Peek in the badge's popover)");
        Row(sb, "F6", "let the AI take just this one decision");
        Row(sb, "Shift+F6", "co-op: a waiting AI partner plays its turn now");
        Row(sb, "F7", "let the AI keep playing until you stop it (press again to stop)");
        Row(sb, "Shift+F7", "autobattle: the AI plays the fights, you make every run-level decision, with a note on what changed since your last one; 1x becomes 2x (again to stop)");
        Row(sb, "Shift+F8", $"advisor brain: Search (thinks ahead) / Instinct (the end-to-end net, no search; Ironclad on v0.111.0 and the v0.113.0 beta for now)  (now {AdvisorBrainRules.Label(SimAdvisor.CurrentBrain)})");
        Row(sb, "F4", "show / hide the HUD");
        Row(sb, "Ctrl+L", "action log — what was played, and what was advised at the time (drag its header to move it; right-click it to put it back under the advice)");
        Row(sb, "Ctrl+H", "hide everything but the Spire Mentor badge for a quick look; again (or click the badge) to show it");
        Row(sb, "F9 / F10", $"game speed down / up, 1x-8x  (now {SpeedSteps.Label(SimSpeed.Desired)})");
        Row(sb, "Ctrl+Z", "undo to the start of this combat turn");
        Row(sb, "Ctrl+Shift+Z", "rewind to the start of this room");
        Row(sb, "Ctrl+Shift+C", "copy this position — paste it into the workbench to explore it");
        Row(sb, "Ctrl+Shift+B", "report a bug — sends this position and the logs, with your note");
        Row(sb, "F1", "this panel");

        sb.Append("\n[b]The Spire Mentor badge[/b]\n");
        sb.Append("[color=#c8ccd4]The badge in the top right, under the game's map / deck / settings buttons, ")
          .Append("shows the mode by its colour and word — grey Off, cyan Coach, green Hints, amber while the ")
          .Append("AI plays — and what the advisor is doing (thinking, starting, offline, building the ")
          .Append("engine). Click it for everything the keys do, with the mouse; click it again, click ")
          .Append("anywhere else or press Esc to close it. Drag its [b]⠿[/b] grip to move it; double-click ")
          .Append("the grip to put it back.\n")
          .Append("[b]Mode[/b]: Off · Coach · Hints · AI plays (the AI plays this run until you stop it, as ")
          .Append("F7 does). [b]Brain[/b]: Search / Instinct, where Instinct can play. [b]This decision[/b]: ")
          .Append("Peek, Let AI pick. [b]Autoplay[/b]: Autobattle, AI plays until stopped. [b]Speed[/b]. ")
          .Append("[b]More[/b]: [b]Rewind ▾[/b] (turn / room / one room back), [b]Panels ▾[/b] (hide all, ")
          .Append("log, HUD, help, Settings — that one works mid-run — and Report a bug) and ")
          .Append("[b]Workbench ▾[/b], whose [b]Open workbench[/b] opens the browser view (logs, search ")
          .Append("tree, run tree) and [b]Copy state[/b] puts this exact position on the clipboard — paste ")
          .Append("it into the workbench's [b]Import state[/b] to replay it, try other lines, or sweep it ")
          .Append("in SimLab.\n")
          .Append("While the AI plays, an amber frame runs round the screen and a [b]Stop[/b] button sits ")
          .Append("next to the badge; in a co-op fight with AI partners, so does [b]Partner go[/b].[/color]\n");

        sb.Append("\n[b]Rewind[/b]\n");
        sb.Append("[color=#c8ccd4][b]\u27f2 Turn[/b] (More › Rewind ▾) replays this combat to the start of ")
          .Append("the turn; [b]\u27f2 Room[/b] / [b]\u27f2\u27f2 Back[/b] jump to room-entry saves ")
          .Append("(same seed, exact state \u2014 a different line afterwards diverges legitimately). ")
          .Append("The workbench's game tree can rewind to any earlier room \u2014 or any recorded ")
          .Append("combat turn, replaying that fight's tape \u2014 and shows where lines forked.[/color]\n");

        sb.Append("\n[b]What the coach draws[/b]\n");
        sb.Append("[color=#c8ccd4][b]Hints[/b]: a score on each card in hand, a ranked list of the moves it ")
          .Append("considered, a win-chance bar, and markers on the map for the route it likes. ")
          .Append("Each of those can be switched off on the settings page.\n")
          .Append("[b]Coach[/b]: none of that before you act. [b]F5[/b] shows its pick for the next decision ")
          .Append("only. After a turn that cost HP or win chance — or a choice it clearly disagrees with — ")
          .Append("a small note shows what it would have done and why (click it to keep it; good turns stay ")
          .Append("quiet unless you ask for a ✓ in the settings). Both styles get those notes.[/color]\n");

        sb.Append("\n[b]Settings[/b]\n");
        sb.Append("[color=#c8ccd4]Main menu → Options → Mod Configuration → SpireMentor, ")
          .Append("or the badge's [b]More › Panels ▾ › Settings[/b] — that one works mid-run too.[/color]\n");

        sb.Append("\n[b]Files[/b]\n");
        sb.Append("[color=#c8ccd4]Config:  %LOCALAPPDATA%\\SpireMentor\\SpireMentor.conf\n")
          .Append("Logs:    %LOCALAPPDATA%\\SpireMentor\\advisor_stdout.log[/color]\n");

        sb.Append("\n[color=#5b6070]F1 closes this.[/color]");
        return sb.ToString();
    }

    /// <summary>Centre the help panel and park the hint at the bottom-left of
    /// whatever the viewport is right now.</summary>
    private static void Reposition()
    {
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return;
        }
        Vector2 viewport = tree.Root.GetVisibleRect().Size;
        if (_panel != null && GodotObject.IsInstanceValid(_panel))
        {
            _panel.Position = new Vector2(
                Math.Max(16f, viewport.X * 0.5f - PanelWidth * 0.5f), viewport.Y * 0.12f);
        }
        if (_hint != null && GodotObject.IsInstanceValid(_hint))
        {
            _hint.Position = new Vector2(24f, Math.Max(16f, viewport.Y - 96f));
        }
    }

    private static void Row(StringBuilder sb, string key, string what) =>
        sb.Append("  [color=#ffd24d][b]").Append(key).Append("[/b][/color]  [color=#c8ccd4]")
          .Append(what).Append("[/color]\n");

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    private static bool EnsureBuilt()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return false;
        }

        var layer = new CanvasLayer { Name = "SpireMentorHelpLayer", Layer = LayerIndex };
        Vector2 viewport = tree.Root.GetVisibleRect().Size;

        // --- help panel, centred-ish and read-only -----------------------
        var panel = new PanelContainer
        {
            Name = "SpireMentorHelpPanel",
            // Placeholder; Reposition() sets the real spot every time it opens.
            Position = new Vector2(viewport.X * 0.5f - PanelWidth * 0.5f, viewport.Y * 0.12f),
            CustomMinimumSize = new Vector2(PanelWidth, 0f),
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Visible = false,
        };
        panel.AddThemeStyleboxOverride("panel", PanelStyle(0.88f));
        var text = new RichTextLabel
        {
            Name = "HelpText",
            BbcodeEnabled = true,
            FitContent = true,
            ScrollActive = false,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(660f, 0f),
        };
        text.AddThemeFontSizeOverride("normal_font_size", 15);
        text.AddThemeFontSizeOverride("bold_font_size", 15);
        panel.AddChild(text);
        layer.AddChild(panel);

        // --- boot hint, bottom-left and self-dismissing ------------------
        var hint = new PanelContainer
        {
            Name = "SpireMentorHelpHint",
            Position = new Vector2(24f, viewport.Y - 72f),
            CustomMinimumSize = new Vector2(560f, 0f),
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Visible = false,
        };
        hint.AddThemeStyleboxOverride("panel", PanelStyle(0.62f));
        var hintText = new RichTextLabel
        {
            Name = "HintText",
            BbcodeEnabled = true,
            FitContent = true,
            ScrollActive = false,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(540f, 0f),
        };
        hintText.AddThemeFontSizeOverride("normal_font_size", 14);
        hintText.AddThemeFontSizeOverride("bold_font_size", 14);
        hint.AddChild(hintText);
        var hintTimer = new Godot.Timer { Name = "HintTimer", OneShot = true, Autostart = false };
        hintTimer.Connect(Godot.Timer.SignalName.Timeout, Callable.From(() =>
        {
            if (_hint != null && GodotObject.IsInstanceValid(_hint))
            {
                _hint.Visible = false;
            }
        }));
        hint.AddChild(hintTimer);
        layer.AddChild(hint);

        tree.Root.AddChildSafely(layer);
        _layer = layer;
        _panel = panel;
        _text = text;
        _hint = hint;
        Logger.Log("[SimHelp] Help layer created (F1).");
        return true;
    }

    private static StyleBoxFlat PanelStyle(float alpha) => new()
    {
        BgColor = new Color(0.04f, 0.04f, 0.06f, alpha),
        BorderColor = new Color(1f, 1f, 1f, 0.12f),
        BorderWidthLeft = 1,
        BorderWidthRight = 1,
        BorderWidthTop = 1,
        BorderWidthBottom = 1,
        CornerRadiusTopLeft = 6,
        CornerRadiusTopRight = 6,
        CornerRadiusBottomLeft = 6,
        CornerRadiusBottomRight = 6,
        ContentMarginLeft = 14f,
        ContentMarginRight = 14f,
        ContentMarginTop = 10f,
        ContentMarginBottom = 12f,
    };

    private static void Guarded(Action action)
    {
        try
        {
            if (NGame.IsMainThread())
            {
                action();
            }
            else
            {
                Callable.From(() => Guarded(action)).CallDeferred();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHelp] Ignoring render error (never breaks gameplay): {ex.Message}");
        }
    }
}
