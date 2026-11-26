// SimShutdown.cs -- one idempotent teardown for every exit path.
//
// There are three ways the mod can learn the game is going away, and none of
// them is reliable on its own:
//
//   1. GameQuitPatch (NGame.Quit)      -- the main menu's Quit button. This
//      is the one that actually fires in practice.
//   2. AppDomain.ProcessExit           -- only if Godot happens to shut the
//      .NET runtime down gracefully. It did not on the 2026-08-08 report.
//   3. SimJobObject                    -- not managed code at all: the OS
//      kills the advisor when the game's process handle closes. Covers
//      Alt+F4, Steam's stop button and crashes.
//
// So this may be called more than once, from more than one thread, possibly
// while the scene tree is already half torn down. It runs once, in order,
// and swallows everything: nothing here may turn a quit into a hang.

using System;
using System.Threading;

namespace SpireMentor;

internal static class SimShutdown
{
    private static int _done;

    public static void Run(string reason)
    {
        if (Interlocked.Exchange(ref _done, 1) != 0)
        {
            return;
        }
        Logger.Log($"Shutting down ({reason}).");

        // Advisor first: it is the process that outlives the game and keeps
        // Steam reporting the game as running.
        Guard(() => SimAdvisorProcess.Stop());
        Guard(() => MainFile.StopAutopilotQuietly());
        Guard(() => BridgeServer.Instance.Stop());
    }

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"Shutdown step failed (ignored): {ex.Message}");
        }
    }
}
