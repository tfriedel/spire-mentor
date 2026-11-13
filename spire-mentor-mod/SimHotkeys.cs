// SimHotkeys.cs -- global hotkeys for live mod control.
//
// Uses the game's own NHotkeyManager (NHotkeyManager.cs): bindings are
// looked up by InputMap ACTION NAME, invoked via CallDeferred (so handlers
// run on the main thread), skipped while the dev console is open or a
// LineEdit has focus, and SetInputAsHandled() only fires when a binding
// matched. Our custom action names are not in MegaInput.AllInputs, so the
// game's "blocking screen" mechanism (pause menu etc.) never blocks them --
// which is what we want: mode/speed/HUD control should work on any screen.
//
// Defaults (chosen from keys the game leaves unbound):
//   F1  help panel -- mode, every key, where the files are
//   F4  toggle HUD visibility
//   Ctrl+H hide / show every mod panel (SimUiHide) for a quick look
//   Ctrl+L toggle the action-log panel (was F3 until 0.8.1; another
//       popular mod binds F3)
//   F5  Coach style: peek at the AI's pick for the next decision only
//   F6  AI takes this decision (advisor mode, combat)
//   Shift+F6 co-op: every AI partner that is waiting (for your first play,
//       your end turn or your setup play) plays the rest of its turn now.
//       Not F3 (another popular mod binds it, see Ctrl+L) and not a Ctrl
//       chord (game streaming such as Apollo drops those).
//   F7  AI plays until I stop (toggle)
//   Shift+F7 autobattle (toggle): the AI plays the fights, I make every
//       run-level decision (Autobattle.cs)
//   F8  AI on / off in the current style (from Autopilot: stop the AI and take over)
//   Shift+F8 advisor brain (toggle): Search <-> Instinct (AdvisorBrainRules).
//       An F-key chord like Shift+F6 / Shift+F7, not Ctrl: streaming drops those.
//   F9  game speed down to the next speed (1x, 2x, 3x, 4x, 6x, 8x)
//   F10 game speed up to the next speed (1x, 2x, 3x, 4x, 6x, 8x)
//   Ctrl+Shift+B report a bug (SimBugReport): B is unbound in the game's
//       keyboard map (NInputManager: E S D X A M, digits, arrows, space).
//
// Key survey (v0.107.1 decomp), so these don't fight the game:
//   F2  is TAKEN and ungated -- NFeedbackScreenOpener.cs:72 opens the
//       feedback screen on keycode 4194333 with no debug/release check. It
//       was our HUD key until 2026-08-05, when pressing it in a live session
//       opened feedback instead. Do not re-use it.
//   F11 is TAKEN -- NDevConsole.
//   F1/F3-F10 map to DebugHotkey.* actions in NInputManager's _debugInputMap,
//       but ProcessDebugKeyInput returns early unless NGame.IsTrailerMode
//       (NInputManager.cs:463), which only the `trailer` console command
//       sets -- so they are free in a normal session. F1 (keycode 4194332 =
//       DebugHotkey.hideRestSite) was re-checked for the help panel, 0.2.1.

using System;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.CommonUi;

namespace SpireMentor;

public static class SimHotkeys
{
    private const string ActionHelp = "spirementor_help";
    private const string ActionToggleHud = "spirementor_toggle_hud";
    private const string ActionToggleLog = "spirementor_toggle_log";
    private const string ActionHideAll = "spirementor_hide_all";
    private const string ActionPeek = "spirementor_peek";
    private const string ActionAiOnce = "spirementor_ai_once";
    private const string ActionAiAuto = "spirementor_ai_auto";
    private const string ActionAutobattle = "spirementor_autobattle";
    private const string ActionToggleAdvisor = "spirementor_toggle_advisor";
    private const string ActionToggleBrain = "spirementor_toggle_brain";
    private const string ActionSpeedDown = "spirementor_speed_down";
    private const string ActionSpeedUp = "spirementor_speed_up";
    private const string ActionUndoTurn = "spirementor_undo_turn";
    private const string ActionRestartRoom = "spirementor_restart_room";
    private const string ActionCopyState = "spirementor_copy_state";
    private const string ActionReportBug = "spirementor_report_bug";
    private const string ActionPartnerGo = "spirementor_partner_go";

    // The workbench's single-key play shortcuts (its ? help: g play the
    // recommended move, p peek, h hint, f play on), only while the Spire Mentor
    // frame is attached: there the workbench and the game sit side by side
    // and one set of keys should drive both, whichever of them has focus. Free in the game's own key
    // map (NInputManager: E S D X A M, digits, arrows, space) -- which is
    // also why the workbench's "a" is not here.
    private const string ActionWbGo = "spirementor_wb_go";
    private const string ActionWbPeek = "spirementor_wb_peek";
    private const string ActionWbHint = "spirementor_wb_hint";
    private const string ActionWbPlayOn = "spirementor_wb_play_on";

    private static bool _installed;

    /// <summary>Wait for the game to come up, then install on the main
    /// thread. Run via TaskHelper.RunSafely from MainFile.Initialize.</summary>
    public static async Task InstallWhenReadyAsync()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        await WaitHelper.Until(
            () => NGame.Instance != null && NHotkeyManager.Instance != null,
            cts.Token, TimeSpan.FromSeconds(120), "NHotkeyManager not available");
        Callable.From(Install).CallDeferred();
    }

    private static void Install()
    {
        if (_installed)
        {
            return;
        }
        try
        {
            NHotkeyManager? manager = NHotkeyManager.Instance;
            if (manager == null)
            {
                return;
            }
            RegisterAction(ActionHelp, Key.F1);
            RegisterAction(ActionToggleHud, Key.F4);
            RegisterAction(ActionToggleLog, Key.L, ctrl: true);
            RegisterAction(ActionHideAll, Key.H, ctrl: true);
            RegisterAction(ActionPeek, Key.F5);
            RegisterAction(ActionAiOnce, Key.F6);
            RegisterAction(ActionPartnerGo, Key.F6, shift: true);
            RegisterAction(ActionAiAuto, Key.F7);
            RegisterAction(ActionAutobattle, Key.F7, shift: true);
            RegisterAction(ActionToggleAdvisor, Key.F8);
            RegisterAction(ActionToggleBrain, Key.F8, shift: true);
            RegisterAction(ActionSpeedDown, Key.F9);
            RegisterAction(ActionSpeedUp, Key.F10);
            // Ctrl-chorded, not F-keys: Z alone is a card hotkey surface and
            // the remaining F-keys are taken (F2 feedback, F11 console).
            RegisterAction(ActionUndoTurn, Key.Z, ctrl: true);
            RegisterAction(ActionRestartRoom, Key.Z, ctrl: true, shift: true);
            // Also Ctrl-chorded, and SHIFTed: plain Ctrl+C is the OS copy every
            // text field expects, and stealing it from the settings panel's
            // fields would be a worse trade than a two-modifier chord.
            RegisterAction(ActionCopyState, Key.C, ctrl: true, shift: true);
            RegisterAction(ActionReportBug, Key.B, ctrl: true, shift: true);
            // Always registered, acted on only while a frame is attached
            // (WithFrame): the mod opens the frame long after startup.
            RegisterAction(ActionWbGo, Key.G);
            RegisterAction(ActionWbPeek, Key.P);
            RegisterAction(ActionWbHint, Key.H);
            RegisterAction(ActionWbPlayOn, Key.F);

            BindActions(manager);

            _installed = true;
            GD.Print("[SpireMentor] Hotkeys installed: F1=help, F4=HUD, Ctrl+L=log, Ctrl+H=hide all, F5=peek (Coach), F6=AI once, Shift+F6=AI partner go, F7=AI auto, Shift+F7=autobattle, F8=AI on/off, Shift+F8=advisor brain, F9/F10=speed -/+, Ctrl+Z=undo turn, Ctrl+Shift+Z=restart room, Ctrl+Shift+C=copy state, Ctrl+Shift+B=report a bug");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[SpireMentor] Hotkey install failed: {ex.Message}");
        }
    }

    internal static void RestoreAfterClear(NHotkeyManager manager)
    {
        // ClearHotkeys can run before our initial InputMap registration.
        if (!_installed)
        {
            return;
        }
        try
        {
            BindActions(manager);
            GD.Print("[SpireMentor] Hotkeys restored after ClearHotkeys.");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"[SpireMentor] Hotkey restore failed: {ex}");
        }
    }

    private static void BindActions(NHotkeyManager manager)
    {
        SimGameApi.PushHotkeyPressedBinding(manager, ActionHelp, SimActions.ToggleHelp);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionToggleHud, SimActions.ToggleHud);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionToggleLog, SimActions.ToggleLog);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionHideAll, SimActions.ToggleHideAll);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionPeek, SimActions.Peek);
        // Godot matches an action's modifiers non-exactly, so Shift+F6 also
        // presses the plain-F6 action (and Shift+F7 the plain-F7 one); leave
        // those to partner go and autobattle.
        SimGameApi.PushHotkeyPressedBinding(manager, ActionAiOnce, () =>
        {
            if (!Input.IsKeyPressed(Key.Shift))
            {
                SimActions.AiOnce("F6 (AI takes this decision)");
            }
        });
        SimGameApi.PushHotkeyPressedBinding(manager, ActionPartnerGo, () => SimActions.PartnersGo("Shift+F6"));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionAiAuto, () =>
        {
            if (!Input.IsKeyPressed(Key.Shift))
            {
                SimActions.ToggleAutoplay("F7 (AI plays until you stop)");
            }
        });
        SimGameApi.PushHotkeyPressedBinding(manager, ActionAutobattle, () =>
            SimActions.ToggleAutobattle("Shift+F7 (autobattle: the AI fights, you decide the run)"));
        // Shift+F8 also presses plain F8 (non-exact modifiers, as above):
        // switching the brain must not also turn the AI off.
        SimGameApi.PushHotkeyPressedBinding(manager, ActionToggleAdvisor, () =>
        {
            if (!Input.IsKeyPressed(Key.Shift))
            {
                SimActions.ToggleAdvisor();
            }
        });
        SimGameApi.PushHotkeyPressedBinding(manager, ActionToggleBrain, () =>
            SimActions.ToggleBrain("Shift+F8 (advisor brain)"));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionSpeedDown, () => SimActions.StepSpeed(-1));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionSpeedUp, () => SimActions.StepSpeed(1));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionUndoTurn, SimActions.UndoTurn);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionRestartRoom, SimActions.RestartRoom);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionCopyState, SimActions.CopyState);
        SimGameApi.PushHotkeyPressedBinding(manager, ActionReportBug, SimActions.ReportBug);
        // Godot matches an action's modifiers non-exactly: Ctrl+F must stay
        // the game's / the OS's, not "play on".
        SimGameApi.PushHotkeyPressedBinding(manager, ActionWbGo, () => WithFrame(() => SimActions.AiOnce("G (play the recommended move)")));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionWbPeek, () => WithFrame(SimActions.Peek));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionWbHint, () => WithFrame(SimActions.Peek));
        SimGameApi.PushHotkeyPressedBinding(manager, ActionWbPlayOn, () => WithFrame(() => SimActions.ToggleAutoplay("F (the AI plays on until you press F or F7 again)")));
    }

    /// <summary>A workbench key: only while a frame is attached (checked at
    /// the press: the frame comes and goes), and only with no modifier held.</summary>
    private static void WithFrame(Action action)
    {
        if (FrameLink.Framed)
        {
            Plain(action);
        }
    }

    /// <summary>Run a single-key shortcut only when no modifier is held.</summary>
    private static void Plain(Action action)
    {
        if (!Input.IsKeyPressed(Key.Ctrl) && !Input.IsKeyPressed(Key.Alt) && !Input.IsKeyPressed(Key.Shift) && !Input.IsKeyPressed(Key.Meta))
        {
            action();
        }
    }

    private static void RegisterAction(string name, Key key, bool ctrl = false, bool shift = false)
    {
        var actionName = new StringName(name);
        if (InputMap.HasAction(actionName))
        {
            return;
        }
        InputMap.AddAction(actionName);
        var ev = new InputEventKey { Keycode = key, CtrlPressed = ctrl, ShiftPressed = shift };
        InputMap.ActionAddEvent(actionName, ev);
    }

}

// v0.111.0 only: v0.107.1 has no ClearHotkeys and never drops the bindings, so
// the patch looks the method up by name and sits out when it is absent. One DLL
// serves both builds, so a compile-time reference would not build on v0.107.1.
[HarmonyPatch]
internal static class HotkeyResetPatch
{
    private static MethodBase? Target =>
        AccessTools.Method(typeof(NHotkeyManager), "ClearHotkeys");

    private static bool Prepare() => Target != null;

    private static MethodBase? TargetMethod() => Target;

    private static void Postfix(NHotkeyManager __instance) =>
        SimHotkeys.RestoreAfterClear(__instance);
}
