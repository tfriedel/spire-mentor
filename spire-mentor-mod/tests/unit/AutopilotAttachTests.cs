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
        // A potion's "choose a card" opened before the attach: nothing would answer it.
        check("attach: a screen open in a fight", Show(AutopilotAttach.Check(true, false, false, true, overlayInFight: true)), "ScreenOpenInFight");
        check("attach: no client beats an open screen", Show(AutopilotAttach.Check(true, false, false, false, overlayInFight: true)), "NoClient");

        // A room that is over must not be played again: the combat handler
        // would wait for a fight that already ended ("Combat never started").
        check("attach entry: mid-room", AutopilotAttach.Entry(mapOpen: false, overlayUp: false, combatInProgress: false).ToString(), "PlayCurrentRoom");
        check("attach entry: mid-fight", AutopilotAttach.Entry(mapOpen: false, overlayUp: false, combatInProgress: true).ToString(), "PlayCurrentRoom");
        check("attach entry: rewards screen up", AutopilotAttach.Entry(mapOpen: false, overlayUp: true, combatInProgress: false).ToString(), "FinishCurrentRoom");
        check("attach entry: map open", AutopilotAttach.Entry(mapOpen: true, overlayUp: false, combatInProgress: false).ToString(), "FinishCurrentRoom");
        check("attach entry: map over rewards", AutopilotAttach.Entry(mapOpen: true, overlayUp: true, combatInProgress: false).ToString(), "FinishCurrentRoom");
        // #1943: the Crystal Sphere (an overlay over an event room) hides the
        // event options; the overlay drain, then the rest of the event, runs.
        check("attach entry: an event overlay (Crystal Sphere) up", AutopilotAttach.Entry(mapOpen: false, overlayUp: true, combatInProgress: false).ToString(), "FinishCurrentRoom");
        // An overlay inside a fight is refused by Check; the entry stays the fight's.
        check("attach entry: an overlay mid-fight", AutopilotAttach.Entry(mapOpen: false, overlayUp: true, combatInProgress: true).ToString(), "PlayCurrentRoom");

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
