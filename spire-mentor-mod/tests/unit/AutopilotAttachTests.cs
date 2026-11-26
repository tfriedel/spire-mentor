// AutopilotAttachTests.cs -- when `spirementor attach` may hand the run in
// progress to Autopilot, and where the attached run's room loop starts.

using SpireMentor;

internal static class AutopilotAttachTests
{
    private static string Show(AttachRefusal? r) => r?.ToString() ?? "go";

    public static void Run(Action<string, string?, string?> check)
    {
        check("attach: all set", Show(AutopilotAttach.Check(true, false, false, true)), "go");
        check("attach: on the menu", Show(AutopilotAttach.Check(false, false, false, true)), "NoRun");
        check("attach: co-op run", Show(AutopilotAttach.Check(true, true, false, true)), "Multiplayer");
        check("attach: a loop already playing", Show(AutopilotAttach.Check(true, false, true, true)), "AlreadyRunning");
        check("attach: no bridge agent", Show(AutopilotAttach.Check(true, false, false, false)), "NoClient");
        // The first reason wins: no run says more than "no client".
        check("attach: no run beats every other reason", Show(AutopilotAttach.Check(false, true, true, false)), "NoRun");
        check("attach: co-op beats a running loop", Show(AutopilotAttach.Check(true, true, true, false)), "Multiplayer");
        check("attach: a running loop beats no client", Show(AutopilotAttach.Check(true, false, true, false)), "AlreadyRunning");

        // A room that is over must not be played again: the combat handler
        // would wait for a fight that already ended ("Combat never started").
        check("attach entry: mid-room", AutopilotAttach.Entry(mapOpen: false, rewardsScreenOpen: false).ToString(), "PlayCurrentRoom");
        check("attach entry: rewards screen up", AutopilotAttach.Entry(mapOpen: false, rewardsScreenOpen: true).ToString(), "FinishCurrentRoom");
        check("attach entry: map open", AutopilotAttach.Entry(mapOpen: true, rewardsScreenOpen: false).ToString(), "FinishCurrentRoom");
        check("attach entry: map over rewards", AutopilotAttach.Entry(mapOpen: true, rewardsScreenOpen: true).ToString(), "FinishCurrentRoom");

        // After the run: back to the player's mode, unless it was Autopilot already.
        check("attach end: from Off/Advisor hands back", AutopilotAttach.HandsBackAfterRun(attachedFromAutopilot: false).ToString(), "True");
        check("attach end: from Autopilot stays", AutopilotAttach.HandsBackAfterRun(attachedFromAutopilot: true).ToString(), "False");

        // Every refusal has its own sentence, not the enum name.
        string? saved = L10n.Override;
        try
        {
            L10n.Override = "en";
            foreach (AttachRefusal r in Enum.GetValues<AttachRefusal>())
            {
                string msg = AutopilotAttach.Message(r);
                check($"attach message: {r} is a sentence", (msg != r.ToString() && msg.EndsWith('.')).ToString(), "True");
            }
        }
        finally
        {
            L10n.Override = saved;
        }
    }
}
