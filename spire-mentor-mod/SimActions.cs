// SimActions.cs -- the mod's user-facing verbs, shared by every surface.
//
// Hotkeys (SimHotkeys), the toolbar (SimToolbar), the dev-console command
// (SimConsoleCommands) and the BaseLib settings page all funnel through
// here, so behavior (and the log lines users see) cannot drift between
// surfaces. Bodies moved verbatim from SimHotkeys (2026-08-21).

using System;
using System.Collections.Generic;
using Godot;

namespace SpireMentor;

internal static class SimActions
{
    public static void ToggleHelp() => SimHelp.Toggle();

    public static void ToggleLog() => SimActionLog.TogglePanel();

    /// <summary>Hide every mod panel down to the toolbar pill, or bring them
    /// all back (Ctrl+H / the pill / Panels menu).</summary>
    public static void ToggleHideAll() => SimUiHide.Toggle();

    /// <summary>F9 / F10 and the toolbar -/+: one whole speed down / up
    /// (SpeedSteps).</summary>
    public static void StepSpeed(int direction)
    {
        SimSpeed.Step(direction);
        SpireMentorModConfig.RememberSpeed();
    }

    /// <summary>Copy this position to the clipboard for the workbench's
    /// live-state import (SimStateCopy).</summary>
    public static void CopyState() => SimStateCopy.Copy();

    /// <summary>Open (or close) the Report a bug dialog (SimBugReport).</summary>
    public static void ReportBug() => SimBugReport.Toggle();

    /// <summary>Undo to the start of the current combat turn (replays the
    /// game's own combat tape -- SimCombatUndo).</summary>
    public static void UndoTurn() => SimCombatUndo.RequestUndoTurn();

    /// <summary>Rewind to the start of the current room (same seed, exact
    /// state -- the game's own room-entry save).</summary>
    public static void RestartRoom()
    {
        SimSnapshotMeta? target = SimSnapshots.CurrentRoomEntry();
        if (target == null)
        {
            SimActionLog.Note("Nothing to rewind to yet (the game saves at each room entry).");
            return;
        }
        SimRestore.RequestRestore(target.Id);
    }

    /// <summary>Rewind one room further back.</summary>
    public static void BackOneRoom()
    {
        SimSnapshotMeta? target = SimSnapshots.PreviousRoomEntry();
        if (target == null)
        {
            SimActionLog.Note("No earlier room snapshot in this run yet.");
            return;
        }
        SimRestore.RequestRestore(target.Id);
    }

    public static void ToggleHud() => SetHud(!SpireMentorModConfig.ShowHud);

    public static void SetHud(bool on)
    {
        SpireMentorModConfig.ShowHud = on;
        GD.Print($"[SpireMentor] HUD {(on ? "shown" : "hidden")}");
        SimRuntime.NotifyChanged();
    }

    /// <summary>AI takes this one decision (F6 / toolbar).</summary>
    public static void AiOnce(string surface) =>
        RequireAdvisor(surface, () => SimAdvisor.Instance.ExecuteBestOnce());

    /// <summary>AI plays until stopped (F7 / toolbar), toggled.</summary>
    public static void ToggleAutoplay(string surface) =>
        RequireAdvisor(surface, () => SimAdvisor.Instance.ToggleAutoPlay());

    /// <summary>Autobattle (Shift+F7 / toolbar), toggled: the AI plays the
    /// fights, you make every run-level decision (Autobattle.cs).</summary>
    public static void ToggleAutobattle(string surface) =>
        RequireAdvisor(surface, () => SimAdvisor.Instance.ToggleAutobattle());

    /// <summary>Advisor brain (Shift+F8 / toolbar / frame), toggled: Search
    /// &lt;-&gt; Instinct (SimAdvisor.SetBrain). A preference, not a takeover,
    /// so it needs no Advisor mode: set with the AI off, it applies from the
    /// next advice.</summary>
    public static void ToggleBrain(string surface) => SimAdvisor.Instance.ToggleBrain(surface);

    /// <summary>"Partner: go" (Shift+F6 / toolbar), co-op with AI partners:
    /// every partner that is waiting for you this turn (your first play, your
    /// end turn, your setup play) plays the rest of it now. Needs no coach:
    /// each partner runs on its own advisor.</summary>
    public static void PartnersGo(string surface)
    {
        var released = new List<string>();
        foreach (SimAiPartner partner in SimAiPartner.All)
        {
            if (partner.Brain.Go())
            {
                released.Add(partner.Label);
            }
        }
        if (released.Count > 0)
        {
            SimActionLog.Note($"{surface}: {string.Join(", ", released)} plays now.");
        }
        else
        {
            SimActionLog.Note(SimAiPartner.All.Count == 0
                ? $"{surface}: no AI partner in this run."
                : $"{surface}: no AI partner is waiting for you.");
        }
    }

    /// <summary>
    /// F8 / the on-off button: AI off &lt;-&gt; on in the current style
    /// (CoachStyle.ToggleOnOff). From Autopilot: stop the AI, hand the game
    /// back, and land in the current style. (Autopilot can only be entered at
    /// launch, from the config file.)
    /// </summary>
    public static void ToggleAdvisor()
    {
        (SimModeLite nextMode, AdviceStyle nextStyle) =
            CoachStyle.ToggleOnOff((SimModeLite)(int)SimRuntime.Mode, SimRuntime.Style);
        if (SimRuntime.Mode == SimMode.Autopilot)
        {
            // Taking over from the AI: stop the slayer FIRST so its card
            // selector and non-interactive mode are released before the mode
            // flips (see MainFile.StopAutopilot).
            MainFile.StopAutopilot();
            SimRuntime.SetModeAndStyle((SimMode)(int)nextMode, nextStyle);
            string style = CoachStyle.StyleLabel(nextStyle);
            GD.Print($"[SpireMentor] Autopilot stopped -> {style} (you have the controls).");
            SimActionLog.Note($"Autopilot stopped; you have the controls ({style}).");
            return;
        }
        SimRuntime.SetModeAndStyle((SimMode)(int)nextMode, nextStyle);
        string name = CoachStyle.ModeName(nextMode, nextStyle);
        GD.Print($"[SpireMentor] Mode -> {name}");
        if (nextMode == SimModeLite.Advisor)
        {
            SimActionLog.Note(nextStyle == AdviceStyle.Coach
                ? "Coach: nothing shows before you act -- F5 peeks at the AI's next pick."
                : "Hints: the AI's opinion shows while you decide.");
        }
    }

    /// <summary>
    /// The style button: Hints &lt;-&gt; Coach (CoachStyle.SwitchStyle). Never
    /// changes the mode, so it can't turn the AI (or its process) off.
    /// </summary>
    public static void SwitchStyle()
    {
        (SimModeLite mode, AdviceStyle nextStyle) =
            CoachStyle.SwitchStyle((SimModeLite)(int)SimRuntime.Mode, SimRuntime.Style);
        SimRuntime.SetModeAndStyle((SimMode)(int)mode, nextStyle);
        // The style switched last is the one the coach starts in next time.
        SpireMentorModConfig.StartInCoachStyle = nextStyle == AdviceStyle.Coach;
        SpireMentorModConfig.SaveSoon();
        GD.Print($"[SpireMentor] Style -> {CoachStyle.StyleLabel(nextStyle)}");
        SimActionLog.Note(nextStyle == AdviceStyle.Coach
            ? "Coach: nothing shows before you act -- F5 peeks at the AI's next pick."
            : "Hints: the AI's opinion shows while you decide.");
    }

    /// <summary>
    /// F5: in the Coach style, show the AI's pick for just the next decision
    /// (SimAdvisor.Peek). Hints already shows everything, so F5 does nothing
    /// there; with the coach off or in Autopilot it only says what it is for
    /// (the RequireAdvisor pattern).
    /// </summary>
    public static void Peek()
    {
        GD.Print($"[SpireMentor] F5 pressed (mode {SimRuntime.Mode}, style {SimRuntime.Style}).");
        if (SimRuntime.Mode == SimMode.Advisor && SimRuntime.Style == AdviceStyle.Hints)
        {
            SimToolbar.Instance.SetNote("Hints already shows the AI's pick (F5 peeks in Coach)");
            return;
        }
        if (!SimRuntime.CoachStyleActive)
        {
            GD.Print("[SpireMentor] F5 peeks in Coach mode.");
            SimToolbar.Instance.SetNote("F5 peeks in Coach mode");
            SimActionLog.Note("F5 peeks in Coach mode (F8 turns the AI on; the toolbar's style button picks Hints or Coach).");
            return;
        }
        SimAdvisor.Instance.Peek();
    }

    /// <summary>
    /// The takeover verbs only mean anything in Advisor mode; say so instead
    /// of looking broken (the advisor's own guards would silently no-op).
    /// <paramref name="what"/> names the surface the user pressed, e.g.
    /// "F6 (AI takes this decision)" or "the AI-once button".
    /// </summary>
    public static void RequireAdvisor(string what, Action action)
    {
        if (SimRuntime.Mode != SimMode.Advisor)
        {
            string how = SimRuntime.Mode == SimMode.Autopilot
                ? "the AI is already playing (Autopilot) -- press F8 to take over"
                : "press F8 to turn Advisor mode on";
            GD.Print($"[SpireMentor] {what} needs Advisor mode: {how}.");
            SimActionLog.Note($"{what} needs Advisor mode: {how}.");
            return;
        }
        action();
    }
}
