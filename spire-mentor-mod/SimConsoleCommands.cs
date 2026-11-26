// SimConsoleCommands.cs -- `spirementor` dev-console command.
//
// The game's DevConsole auto-registers AbstractConsoleCmd subtypes from mod
// assemblies (DevConsole.cs:33, ReflectionHelper.GetSubtypesInMods). The
// console itself is available whenever mods are loaded
// (ModManager.IsRunningModded()). DebugOnly=false so the command registers
// even outside Autopilot, where IsReleaseGame() reports a release build.
//
// This is the only control surface that can touch the Autopilot wait/anim
// multipliers -- deliberately not on the settings page, so a persisted UI
// value can never silently retune the sweep lane.

using System;
using System.Globalization;
using MegaCrit.Sts2.Core.DevConsole;
using MegaCrit.Sts2.Core.DevConsole.ConsoleCommands;
using MegaCrit.Sts2.Core.Entities.Players;

namespace SpireMentor;

public class SpireMentorConsoleCmd : AbstractConsoleCmd
{
    public override string CmdName => "spirementor";

    public override string Args => "[mode off|advisor|hints|coach] [attach] [speed <0.1-8.0>] [hud on|off] [log] [bugreport [note...]] [waitmult <f>] [animmult <f>]";

    public override string Description => L10n.T("Spire Mentor control: show status, switch mode, hand the run to Autopilot, set game speed / HUD / autopilot multipliers.");

    public override bool IsNetworked => false;

    public override bool DebugOnly => false;

    public override CmdResult Process(Player? issuingPlayer, string[] args)
    {
        if (args.Length == 0)
        {
            return new CmdResult(success: true,
                $"mode={SimRuntime.Mode} style={SimRuntime.Style} speed={SimSpeed.Current:0.0}x hud={(SpireMentorModConfig.ShowHud ? "on" : "off")} " +
                $"waitmult={SimRuntime.WaitMultiplier:0.###} animmult={SimRuntime.AnimMultiplier:0.###}");
        }

        string sub = args[0].ToLowerInvariant();
        string? value = args.Length > 1 ? args[1] : null;
        switch (sub)
        {
            case "mode":
                return SetMode(value);
            case "attach":
                return Attach();
            case "speed":
                if (!TryParseFloat(value, out float speed))
                {
                    return new CmdResult(success: false, L10n.F("usage: {0}", "spirementor speed <0.1-8.0>"));
                }
                SimSpeed.Apply(speed);
                SpireMentorModConfig.RememberSpeed();
                return new CmdResult(success: true, $"speed -> {SimSpeed.Current:0.0}x");
            case "hud":
                if (value is not ("on" or "off"))
                {
                    return new CmdResult(success: false, L10n.F("usage: {0}", "spirementor hud on|off"));
                }
                SimActions.SetHud(value == "on");
                return new CmdResult(success: true, $"hud -> {value}");
            case "log":
                SimActions.ToggleLog();
                return new CmdResult(success: true, L10n.T("toggled action-log panel"));
            case "bugreport":
                if (args.Length == 1)
                {
                    SimActions.ReportBug();
                    return new CmdResult(success: true, L10n.T("opened the bug report dialog"));
                }
                // A note on the command line: send straight away (the action
                // log says where it went).
                SimBugReport.SendNow(string.Join(" ", args, 1, args.Length - 1));
                return new CmdResult(success: true,
                    SimBugReport.Site != null ? L10n.T("sending a bug report (see the action log)") : L10n.T("saving a bug report (see the action log)"));
            case "waitmult":
                if (!TryParseFloat(value, out float wait) || wait <= 0f)
                {
                    return new CmdResult(success: false, L10n.F("usage: {0}", "spirementor waitmult <positive float>"));
                }
                SimRuntime.WaitMultiplier = wait;
                return new CmdResult(success: true, L10n.F("waitmult -> {0:0.###} (applies while Autopilot)", wait));
            case "animmult":
                if (!TryParseFloat(value, out float anim) || anim <= 0f)
                {
                    return new CmdResult(success: false, L10n.F("usage: {0}", "spirementor animmult <positive float>"));
                }
                SimRuntime.AnimMultiplier = anim;
                return new CmdResult(success: true, L10n.F("animmult -> {0:0.###} (applies while Autopilot)", anim));
            default:
                return new CmdResult(success: false, L10n.F("unknown subcommand '{0}'. {1}", sub, Args));
        }
    }

    public override CompletionResult GetArgumentCompletions(Player? player, string[] args)
    {
        if (args.Length <= 1)
        {
            return CompleteArgument(
                ["mode", "attach", "speed", "hud", "log", "waitmult", "animmult"],
                [], args.Length == 1 ? args[0] : "");
        }
        if (args.Length == 2)
        {
            string[] candidates = args[0].ToLowerInvariant() switch
            {
                "mode" => ["off", "advisor", "hints", "coach"],
                "hud" => ["on", "off"],
                _ => [],
            };
            if (candidates.Length > 0)
            {
                return CompleteArgument(candidates, [args[0]], args[1]);
            }
        }
        return base.GetArgumentCompletions(player, args);
    }

    private static CmdResult SetMode(string? value)
    {
        // Leaving Autopilot stops the slayer rather than being refused --
        // `auto_start=true` persists in SpireMentor.conf between a sweep and a
        // human session, and the old refusal left the player stuck with the
        // AI's game modifiers.
        string? target = value?.ToLowerInvariant();
        // Only a valid target stops the slayer -- a typo must not silently
        // end an autopilot session.
        AdviceStyle? style = CoachStyle.Parse(target);
        string note = (target == "off" || target == "advisor" || style != null) && MainFile.StopAutopilot()
            ? L10n.T(" (autopilot stopped)")
            : "";
        if (style is AdviceStyle s)
        {
            SimRuntime.SetModeAndStyle(SimMode.Advisor, s);
            return new CmdResult(success: true, $"mode -> Advisor, style -> {s}{note}");
        }
        switch (target)
        {
            case "off":
                SimRuntime.Mode = SimMode.Off;
                return new CmdResult(success: true, $"mode -> Off{note}");
            case "advisor":
                // Keeps the current style (hints / coach pick one explicitly).
                SimRuntime.Mode = SimMode.Advisor;
                return new CmdResult(success: true, $"mode -> Advisor ({SimRuntime.Style}){note}");
            case "autopilot":
                return new CmdResult(success: false,
                    L10n.T("Autopilot starts new runs only with auto_start=true in SpireMentor.conf at launch; 'spirementor attach' hands it the run in progress."));
            default:
                return new CmdResult(success: false, L10n.F("usage: {0}", "spirementor mode off|advisor|hints|coach"));
        }
    }

    /// <summary>
    /// Hands the run in progress to Autopilot (MainFile.AttachAutopilot): the
    /// connected bridge agent plays it from here. The way back after F8, and
    /// how to start a bridge agent on floor 35 without replaying floors 1-34.
    /// </summary>
    private static CmdResult Attach()
    {
        AttachRefusal? refusal;
        try
        {
            refusal = MainFile.AttachAutopilot();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SpireMentor] Attach failed: {ex}");
            return new CmdResult(success: false, L10n.T("Autopilot could not attach (see the log)."));
        }
        return refusal is AttachRefusal r
            ? new CmdResult(success: false, AutopilotAttach.Message(r))
            : new CmdResult(success: true, L10n.T("Autopilot attached: the bridge agent plays this run from here."));
    }

    private static bool TryParseFloat(string? value, out float result)
    {
        result = 0f;
        return value != null &&
               float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }
}
