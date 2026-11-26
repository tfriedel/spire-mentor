// AutopilotAttach.cs -- the game-free rules for handing a run in progress
// (back) to Autopilot: `spirementor attach` / MainFile.AttachAutopilot.
//
// Autopilot normally only plays runs it started itself, from the main menu
// (SimAutoSlayer.PlayRunAsync). Attaching skips the menu and the run_config
// handshake and drives the run that is already open, from whatever floor it
// is on. Uses: picking a run up on floor 35 to reproduce a late-run bug
// without replaying from floor 1, and giving a run back to the bridge agent
// after the player took over with F8 to clear a screen. The rest of the run
// goes through the same lifecycle as any Autopilot run (SimAutoSlayer.
// RunOnceAsync): one budget, one teardown, exactly one run_complete.
//
// Kept free of game types so tests/unit can pin the decisions.

namespace SpireMentor;

/// <summary>Why an attach was refused; null from <see cref="AutopilotAttach.Check"/> = go.</summary>
public enum AttachRefusal
{
    /// <summary>No RunState: the game is on the menu (or loading).</summary>
    NoRun,
    /// <summary>A co-op run: Autopilot plays single-player runs only.</summary>
    Multiplayer,
    /// <summary>An Autopilot run loop is already playing.</summary>
    AlreadyRunning,
    /// <summary>
    /// No bridge client connected. Autopilot only engages with an agent
    /// connected (SimRuntime.AutopilotEngaged); an armed slayer waiting for
    /// one would sit inside the run's handlers while the player clicks.
    /// </summary>
    NoClient,
    /// <summary>
    /// A fight with an overlay screen up (a potion's "choose a card"). The
    /// screen opened before Autopilot, so the agent-backed selector never saw
    /// it and nothing would answer it: every play the agent sends queues
    /// behind it and none resolves (PR #1944 live test, Attack/Skill Potion).
    /// The stock overlay handler would pick at random, which the bridge never
    /// does, so the player finishes the screen and attaches after.
    /// </summary>
    ScreenOpenInFight,
}

/// <summary>Where the attached run's room loop starts.</summary>
public enum AttachEntry
{
    /// <summary>The current room is still being played (a fight, an event page, a shop): play it.</summary>
    PlayCurrentRoom,
    /// <summary>
    /// The room's own handler cannot start here: the map is open, or an
    /// overlay screen is up outside a fight (the rewards screen, the Crystal
    /// Sphere minigame, a game-over screen). Playing the room again would wait
    /// for something already past -- a fight that ended ("Combat never
    /// started"), or event options hidden under the overlay ("Event options
    /// never appeared", #1943). So only its tail runs: the overlay drain,
    /// the rest of an event, the post-boss transition, then the map.
    /// </summary>
    FinishCurrentRoom,
}

public static class AutopilotAttach
{
    /// <summary>The first reason an attach cannot go ahead, or null.</summary>
    public static AttachRefusal? Check(bool runInProgress, bool multiplayer, bool loopRunning, bool clientConnected,
        bool overlayInFight = false)
    {
        if (!runInProgress)
            return AttachRefusal.NoRun;
        if (multiplayer)
            return AttachRefusal.Multiplayer;
        if (loopRunning)
            return AttachRefusal.AlreadyRunning;
        if (!clientConnected)
            return AttachRefusal.NoClient;
        if (overlayInFight)
            return AttachRefusal.ScreenOpenInFight;
        return null;
    }

    /// <summary>
    /// Picks the loop entry from what is on screen. A fight in progress is
    /// always played by the combat handler (with an overlay up, Check has
    /// refused already): the room tail would skip to the map with the fight
    /// still running.
    /// </summary>
    public static AttachEntry Entry(bool mapOpen, bool overlayUp, bool combatInProgress)
    {
        if (combatInProgress)
            return AttachEntry.PlayCurrentRoom;
        return mapOpen || overlayUp ? AttachEntry.FinishCurrentRoom : AttachEntry.PlayCurrentRoom;
    }

    /// <summary>
    /// Whether the mode goes back to what it was once the attached run ends.
    /// Yes from Off / Advisor: an armed, idle slayer would turn the player's
    /// next run non-interactive as soon as a bridge client connects. No from
    /// Autopilot: that session was Autopilot before the attach.
    /// </summary>
    public static bool HandsBackAfterRun(bool attachedFromAutopilot) => !attachedFromAutopilot;

    /// <summary>The player-facing line for a refusal.</summary>
    public static string Message(AttachRefusal refusal) => refusal switch
    {
        AttachRefusal.NoRun => L10n.T("No run in progress to attach to: start or continue a run first."),
        AttachRefusal.Multiplayer => L10n.T("Autopilot plays single-player runs only."),
        AttachRefusal.AlreadyRunning => L10n.T("Autopilot is already playing."),
        AttachRefusal.NoClient => L10n.T("No bridge agent connected: connect one, then attach."),
        AttachRefusal.ScreenOpenInFight => L10n.T("Finish the open screen first, then attach."),
        _ => refusal.ToString(),
    };
}
