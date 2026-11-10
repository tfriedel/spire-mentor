// SimFrameLauncher.cs -- "Panels in their own window": the mod starts the
// Spire Mentor frame (spire-mentor-frame/, shipped in mods/SpireMentor/
// SpireMentorFrame/) next to the running game. The frame adopts the game's
// window, lays the panels out around it and shows the bundled workbench's
// frame page from the advisor's control plane; FrameLink feeds it.
//
// The player's switch is SpireMentorModConfig.PanelsInOwnWindow (settings, the
// toolbar's Panels menu): on at startup opens the frame once the game window is
// up, ticking it opens it now, unticking it asks the frame to hand the game
// its window back. Closing the frame window does the same and unticks it.
//
// A game the frame LAUNCHED (--spire-mentor-frame-port, the dev lane) is
// framed already; nothing here touches it.
//
// Never throws into the game: every entry point swallows and logs.

using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using Godot;

namespace SpireMentor;

internal static class SimFrameLauncher
{
    private static Process? _frame;
    // Set while the mod itself is closing the frame (unticked): its exit
    // then keeps the setting as the player left it.
    private static bool _closing;

    /// <summary>The frame the mod started is running.</summary>
    public static bool Running => _frame is { HasExited: false };

    /// <summary>At startup: open the frame when the player wants it, once
    /// the game window has settled (the game re-applies its display settings
    /// during the first frames).</summary>
    public static void OpenAtStartupIfWanted()
    {
        // On by default, so quiet where it cannot or should not open: off
        // Windows (no note at every start), and in the test lanes -- the
        // advisor_autoplay harness, --spire-mentor-autoplay / -port and
        // --fastmp (two games on one desktop).
        bool harness = SimConfig.Instance.AdvisorAutoplay || SimMp.AutoplayOverride ||
                       SimMp.PortOverride != null || SimMp.FastMpLaunch;
        if (!FrameLaunchRules.OpenAtStartup(SpireMentorModConfig.PanelsInOwnWindow, FrameLink.Framed,
                OperatingSystem.IsWindows(), harness))
        {
            return;
        }
        Guard(() =>
        {
            SceneTree? tree = Engine.GetMainLoop() as SceneTree;
            if (tree == null)
            {
                return;
            }
            tree.CreateTimer(3.0).Timeout += () => Guard(() => Open("startup"));
        });
    }

    /// <summary>The player's switch (settings, toolbar): remember it and
    /// open or close the frame now.</summary>
    public static void SetWanted(bool wanted, string source)
    {
        Guard(() =>
        {
            if (SpireMentorModConfig.PanelsInOwnWindow != wanted)
            {
                SpireMentorModConfig.PanelsInOwnWindow = wanted;
                SpireMentorModConfig.SaveSoon();
            }
            if (wanted)
            {
                Open(source);
            }
            else
            {
                Close(source);
            }
        });
    }

    private static void Open(string source)
    {
        if (FrameLink.Framed)
        {
            // Framed with the panels inside: the frame is there, move them out.
            FrameLink.SetUiOutside(true);
            return;
        }
        if (!OperatingSystem.IsWindows())
        {
            SimActionLog.Note("Panels in their own window: Windows only.");
            return;
        }
        string? modDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        string exe = modDir != null ? FrameLaunchRules.ExePath(modDir) : "";
        if (!File.Exists(exe))
        {
            Logger.Log($"[SimFrameLauncher] no frame at {exe}");
            SimActionLog.Note("Panels in their own window: the frame is missing from the mod folder (reinstall the mod).");
            return;
        }
        int port = FreeLoopbackPort();
        int controlPort = SimAdvisorProcess.ControlPort(BridgeServer.Instance.Port);
        string url = FrameLaunchRules.PageUrl(SpireMentorModConfig.FrameUrl, controlPort);
        // A fullscreen window fights being made a child; the frame is a
        // window, so the game is too while it sits inside.
        DisplayServer.WindowMode mode = DisplayServer.WindowGetMode();
        if (mode is DisplayServer.WindowMode.Fullscreen or DisplayServer.WindowMode.ExclusiveFullscreen)
        {
            DisplayServer.WindowSetMode(DisplayServer.WindowMode.Windowed);
        }
        long hwnd = DisplayServer.WindowGetNativeHandle(DisplayServer.HandleType.WindowHandle);
        string? gameDir = Path.GetDirectoryName(OS.GetExecutablePath());
        var psi = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            WorkingDirectory = Path.GetDirectoryName(exe),
        };
        foreach (string a in FrameLaunchRules.Args(System.Environment.ProcessId, hwnd, port, url, gameDir))
        {
            psi.ArgumentList.Add(a);
        }
        Process frame = Process.Start(psi) ?? throw new InvalidOperationException($"could not start {exe}");
        frame.EnableRaisingEvents = true;
        frame.Exited += (_, _) => Callable.From(() => Guard(() => OnFrameExited(frame))).CallDeferred();
        _frame = frame;
        _closing = false;
        Logger.Log($"[SimFrameLauncher] started the frame ({source}): pid {frame.Id}, relay :{port}, page {url}");
        // Asked for just now (not the startup open): the panels go around
        // the game even when the frame was last left fullscreen.
        FrameLink.Attach(port, playerAsked: source != "startup");
    }

    private static void Close(string source)
    {
        if (!Running)
        {
            if (FrameLink.Framed && !FrameLink.UiOutside)
            {
                return; // a frame-launched game with the panels inside already
            }
            if (FrameLink.Framed)
            {
                // The frame launched the game: it cannot give the window
                // back, so just bring the panels inside.
                FrameLink.SetUiOutside(false);
            }
            return;
        }
        _closing = true;
        Logger.Log($"[SimFrameLauncher] closing the frame ({source})");
        if (!FrameLink.RequestClose())
        {
            // No link to ask on (the frame never connected): it holds no
            // game window yet either, so ending it is safe.
            try
            {
                _frame!.Kill();
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimFrameLauncher] could not end the frame: {ex.Message}");
            }
        }
    }

    private static void OnFrameExited(Process frame)
    {
        if (!ReferenceEquals(frame, _frame))
        {
            return;
        }
        int code = SafeExitCode(frame);
        _frame = null;
        FrameLink.Detach();
        FrameFullscreen.AfterFrameClosed();
        if (_closing)
        {
            _closing = false;
            return;
        }
        // The player closed the frame window (or it failed): panels back in
        // the game, and the setting says so, so the next start does not
        // bring it back unasked.
        Logger.Log($"[SimFrameLauncher] the frame exited (code {code}): panels back in the game");
        if (SpireMentorModConfig.PanelsInOwnWindow)
        {
            SpireMentorModConfig.PanelsInOwnWindow = false;
            SpireMentorModConfig.SaveSoon();
        }
        if (code != 0)
        {
            SimActionLog.Note($"The panels' window closed unexpectedly (code {code}); they are back in the game. %LOCALAPPDATA%\\SpireMentor\\frame\\frame.log says why.");
        }
    }

    private static int SafeExitCode(Process p)
    {
        try
        {
            return p.ExitCode;
        }
        catch (InvalidOperationException)
        {
            return -1;
        }
    }

    /// <summary>A loopback port free right now (the frame binds it a moment
    /// later; a lost race fails its start, logged, and the panels stay in).</summary>
    private static int FreeLoopbackPort()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        try
        {
            return ((IPEndPoint)l.LocalEndpoint).Port;
        }
        finally
        {
            l.Stop();
        }
    }

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimFrameLauncher] ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}
