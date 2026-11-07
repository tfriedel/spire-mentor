// FrameLaunchRules.cs -- the pure half of SimFrameLauncher: where the frame
// lives, which page it shows and its command line. No Godot, so the unit
// tests (tests/unit) compile it.

using System.Collections.Generic;
using System.IO;

namespace SpireMentor;

internal static class FrameLaunchRules
{
    /// <summary>The frame's folder under the mod's (package.ps1 and the
    /// deploy put it there).</summary>
    public const string FrameDirName = "SpireMentorFrame";

    public const string FrameExeName = "SpireMentorFrame.exe";

    /// <summary>The packaged frame next to SpireMentor.dll.</summary>
    public static string ExePath(string modDir) => Path.Combine(modDir, FrameDirName, FrameExeName);

    /// <summary>Open the frame when the game starts: the player wants it (the
    /// default), no frame is there yet, on Windows, and not in a test lane.</summary>
    public static bool OpenAtStartup(bool wanted, bool framed, bool windows, bool harness) =>
        wanted && !framed && windows && !harness;

    /// <summary>The page the frame shows: the configured URL when set (dev:
    /// the Vite server), else the bundled workbench's frame page on the
    /// advisor's control plane.</summary>
    public static string PageUrl(string? configured, int controlPort)
    {
        string c = (configured ?? "").Trim();
        return c.Length > 0 ? c : $"http://127.0.0.1:{controlPort}/frame.html";
    }

    /// <summary>The frame's arguments in attach mode (spire-mentor-frame/README.md):
    /// adopt this game's window, relay on <paramref name="port"/>.</summary>
    public static IReadOnlyList<string> Args(int pid, long hwnd, int port, string url, string? gameDir)
    {
        var args = new List<string>
        {
            "--attach", pid.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--port", port.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--url", url,
        };
        if (hwnd != 0)
        {
            args.Add("--attach-hwnd");
            args.Add(hwnd.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        if (!string.IsNullOrEmpty(gameDir))
        {
            args.Add("--game-dir");
            args.Add(gameDir!);
        }
        return args;
    }
}
