// SimGameVersion.cs -- which game build is running, for the bridge envelope.
//
// The advisor builds its engine for the build the mod reports
// (`game_version`, PROTOCOL.md section 1). The running build is read from the
// game's own release_info.json through ReleaseInfoManager, the same source the
// game's version label and Sentry tags use: `ReleaseInfoManager.Instance
// .ReleaseInfo?.Version` is `"v0.107.1"` / `"v0.111.0"`, identical API in both
// decomps (Core.Debug/ReleaseInfoManager.cs, ReleaseInfo.cs).
//
// One DLL runs on every build in Supported: the sts2 members that changed
// signature between them are reached through SimGameApi, and
// tests/compile/build_mod.sh checks the DLL binds against each build's
// sts2.dll. A build outside the list is logged loudly at startup.

using System;
using MegaCrit.Sts2.Core.Debug;

namespace SpireMentor;

internal static class SimGameVersion
{
    /// <summary>The game builds this DLL is checked against.</summary>
    public static readonly string[] Supported = { "v0.107.1", "v0.111.0", "v0.113.0" };

    /// <summary>What the protocol assumes when the build is unknown (PROTOCOL.md s1).</summary>
    private const string Fallback = "v0.107.1";

    private static string? _wire;

    /// <summary>
    /// The running game's build in wire form (<c>"v0.111.0"</c>), resolved
    /// once. Falls back to <see cref="Fallback"/> when release_info.json is
    /// missing or unreadable (a dev build of the game).
    /// </summary>
    public static string Wire => _wire ??= Resolve();

    /// <summary>Resolve and log the version (MainFile.Initialize, main thread).</summary>
    public static void LogAtStartup()
    {
        Logger.Log($"Game version: {Wire} (mod supports {string.Join(", ", Supported)})");
    }

    private static string Resolve()
    {
        string? raw = null;
        try
        {
            raw = ReleaseInfoManager.Instance.ReleaseInfo?.Version;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimGameVersion] Could not read the game's release info: {ex.Message}");
        }
        string? running = GameVersionFormat.Normalize(raw);
        if (running == null)
        {
            Logger.Log($"[SimGameVersion] release_info.json has no version; reporting {Fallback}.");
            return Fallback;
        }
        if (Array.IndexOf(Supported, running) < 0)
        {
            Logger.Log(
                $"[SimGameVersion] WARNING: game {running} is not one this Spire Mentor supports " +
                $"({string.Join(", ", Supported)}). Parts of the mod may fail on it.");
        }
        return running;
    }
}
