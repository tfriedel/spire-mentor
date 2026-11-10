// MentorBadge.cs -- the words on the Spire Mentor badge and its popover
// header (SimToolbar), in the player's language.
//
// The badge says, at a glance, which mode is on and whether the advisor is
// busy; the header says the same in a sentence and what to do about it. No
// enum names, no "advisor: crashed": a player reads "Advisor offline", and
// what turns it back on.
//
// Game-free (no Godot, no sts2.dll) so tests/unit pins the wording.

using System;

namespace SpireMentor;

/// <summary>Game-free mirror of <c>SimAdvisorStatus</c> (SimHud.cs): same
/// order, same values.</summary>
public enum AdvisorStatusLite
{
    None,
    Launching,
    Building,
    Connected,
    Thinking,
    Crashed,
}

/// <summary>Everything the badge's words depend on.</summary>
public readonly record struct BadgeState(
    MentorMode Mode,
    bool Autopilot,
    bool AutopilotEngaged,
    bool Autobattle,
    AdvisorStatusLite Advisor,
    string BuildElapsed,
    bool Hidden);

internal static class MentorBadge
{
    public const string Name = "Spire Mentor";

    /// <summary>The thinking indicator's frames, all the same width so the
    /// badge does not jitter while it cycles.</summary>
    private static readonly string[] DotFrames = { "●○○", "○●○", "○○●" };

    /// <summary>Wall-clock as "m:ss"; a clock stepped back renders 0:00.</summary>
    public static string Elapsed(TimeSpan elapsed)
    {
        if (elapsed < TimeSpan.Zero)
        {
            elapsed = TimeSpan.Zero;
        }
        return $"{(int)elapsed.TotalMinutes}:{elapsed.Seconds:00}";
    }

    /// <summary>The mode in one or two words.</summary>
    public static string ModeWord(in BadgeState s) =>
        s.Autopilot
            ? s.AutopilotEngaged ? "Autopilot" : "Autopilot (waiting)"
            : s.Mode switch
            {
                MentorMode.Off => "Off",
                MentorMode.AiPlays => "AI playing",
                _ when s.Autobattle => "Autobattle",
                MentorMode.Coach => "Coach",
                _ => "Hints",
            };

    /// <summary>
    /// What the advisor is doing, in a word or two, or null when there is
    /// nothing to say (ready, or the AI is off). <paramref name="frame"/>
    /// advances the thinking dots; <paramref name="still"/> ("Reduce combat
    /// effects") shows them without motion. Kept apart from the sentence in
    /// <see cref="Status"/>, which must not change every time the advisor
    /// starts and stops thinking (the popover would jump under the mouse).
    /// </summary>
    public static string? Activity(in BadgeState s, int frame, bool still)
    {
        if (s.Advisor == AdvisorStatusLite.Building)
        {
            return $"building the engine {s.BuildElapsed}";
        }
        if (s.Mode == MentorMode.Off || s.Autopilot)
        {
            return null;
        }
        return s.Advisor switch
        {
            AdvisorStatusLite.Thinking => "thinking " + (still ? "…" : DotFrames[Math.Abs(frame) % DotFrames.Length]),
            AdvisorStatusLite.Launching or AdvisorStatusLite.None => "starting",
            AdvisorStatusLite.Crashed => "offline",
            _ => null,
        };
    }

    /// <summary>
    /// The badge's text. An engine build outranks everything (nothing works
    /// until it ends, and a wait that does not count up reads as a hang and
    /// gets killed); then a hidden UI says how to get it back; else the mode
    /// and what the advisor is doing (<see cref="Activity"/>).
    /// </summary>
    public static string Text(in BadgeState s, int frame, bool still)
    {
        if (s.Advisor == AdvisorStatusLite.Building)
        {
            return $"{Name} · Building the engine {s.BuildElapsed}";
        }
        if (s.Hidden)
        {
            return $"{Name} · hidden (click to show)";
        }
        string text = $"{Name} · {ModeWord(s)}";
        string? activity = Activity(s, frame, still);
        return activity == null ? text : $"{text} · {activity}";
    }

    /// <summary>
    /// The popover header's sentence: the mode, or what is in the way and
    /// what to do about it. Steady while the advisor thinks (see
    /// <see cref="Activity"/>).
    /// </summary>
    public static string Status(in BadgeState s)
    {
        if (s.Advisor == AdvisorStatusLite.Building)
        {
            return "The first start after an update builds the engine, which takes minutes. " +
                   "Leave the game running: quitting starts it over.";
        }
        if (s.Autopilot)
        {
            return s.AutopilotEngaged
                ? "Autopilot is playing this run. Pick Coach or Hints to take the controls (F8)."
                : "Autopilot is waiting for its AI client. Pick Coach or Hints to play yourself (F8).";
        }
        if (s.Mode == MentorMode.Off)
        {
            return "The AI is off. Pick Coach or Hints for advice while you play.";
        }
        return s.Advisor switch
        {
            AdvisorStatusLite.Launching or AdvisorStatusLite.None => "Starting the advisor…",
            AdvisorStatusLite.Crashed => "Advisor offline. Switch the AI Off and back on to restart it.",
            _ => s.Mode switch
            {
                MentorMode.AiPlays => "The AI is playing until you stop it (F7).",
                _ when s.Autobattle => "Autobattle: the AI plays the fights, you make the run's decisions.",
                MentorMode.Coach => "Coach: nothing shows before you act. Peek (F5) shows the next pick.",
                _ => "Hints: the AI's opinion shows while you decide.",
            },
        };
    }

    /// <summary>The Stop button shown while the AI plays: its label, or null
    /// when the AI is not playing. Autopilot's stop is a takeover (F8).</summary>
    public static string? StopLabel(in BadgeState s) =>
        s.Autopilot
            ? s.AutopilotEngaged ? "Autopilot is playing — Take over (F8)" : "Autopilot waiting — Take over (F8)"
            : s.Mode == MentorMode.AiPlays
                ? "AI is playing — Stop (F7)"
                : s.Autobattle
                    ? "AI fights — Stop (F7)"
                    : null;
}

/// <summary>
/// Whether the badge says "thinking". Most thinks after a player's action
/// take a few milliseconds; showing each one flashed "· thinking ●○○" on the
/// badge, which widened it and shoved the buttons beside it (Exit
/// fullscreen) left and back after every action (live, 2026-11-09). So a
/// think shows only once it has run <see cref="ShowAfter"/>, and once shown
/// it stays until the advisor has been idle <see cref="HideAfter"/>, so a
/// run of back-to-back thinks (the AI playing) reads as one.
/// </summary>
public sealed class ThinkingIndicator
{
    public static readonly TimeSpan ShowAfter = TimeSpan.FromMilliseconds(600);
    public static readonly TimeSpan HideAfter = TimeSpan.FromMilliseconds(400);

    private DateTime? _thinkingSince;
    private DateTime? _idleSince;
    private bool _shown;

    /// <summary>Feed the advisor's state now; returns whether to show it.</summary>
    public bool Update(bool thinking, DateTime now)
    {
        if (thinking)
        {
            _idleSince = null;
            _thinkingSince ??= now;
            if (now - _thinkingSince.Value >= ShowAfter)
            {
                _shown = true;
            }
        }
        else
        {
            _thinkingSince = null;
            _idleSince ??= now;
            if (now - _idleSince.Value >= HideAfter)
            {
                _shown = false;
            }
        }
        return _shown;
    }

    /// <summary>The status the badge shows: Thinking only while
    /// <see cref="Update"/> says so, Connected otherwise.</summary>
    public AdvisorStatusLite Shown(AdvisorStatusLite status, DateTime now)
    {
        bool show = Update(status == AdvisorStatusLite.Thinking, now);
        return status switch
        {
            AdvisorStatusLite.Thinking => show ? AdvisorStatusLite.Thinking : AdvisorStatusLite.Connected,
            AdvisorStatusLite.Connected => show ? AdvisorStatusLite.Thinking : AdvisorStatusLite.Connected,
            _ => status,
        };
    }
}
