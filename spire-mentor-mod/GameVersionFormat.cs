// GameVersionFormat.cs -- the wire spelling of a game build ("v0.111.0").
//
// Deliberately free of game types so tests/unit can compile it on its own.

namespace SpireMentor;

internal static class GameVersionFormat
{
    /// <summary>
    /// The build the advisor assumes when the envelope carries no
    /// <c>game_version</c> (every mod before the field). PROTOCOL.md section 1.
    /// </summary>
    public const string Default = "v0.107.1";

    /// <summary>
    /// <c>ReleaseInfo.Version</c> (release_info.json <c>"version"</c>, e.g.
    /// <c>"v0.111.0"</c>) in the wire form: a lowercase <c>v</c> and three
    /// dot-separated integers without leading zeros, so <c>"0.111.0"</c>,
    /// <c>" V0.111.0 "</c> and <c>"v0.111.0"</c> all become <c>"v0.111.0"</c>.
    /// Null for null or blank input. Any other shape (a suffix, a fourth
    /// component) is passed through trimmed and <c>v</c>-prefixed rather than
    /// guessed at, so the advisor can name the exact build it refuses.
    /// </summary>
    public static string? Normalize(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }
        string v = raw.Trim();
        if (v[0] == 'v' || v[0] == 'V')
        {
            v = v.Substring(1);
        }
        string[] parts = v.Split('.');
        if (parts.Length == 3)
        {
            var nums = new long[3];
            bool ok = true;
            for (int i = 0; i < 3 && ok; i++)
            {
                ok = parts[i].Length > 0
                    && parts[i].Length <= 9
                    && parts[i].All(char.IsAsciiDigit)
                    && long.TryParse(parts[i], out nums[i]);
            }
            if (ok)
            {
                return $"v{nums[0]}.{nums[1]}.{nums[2]}";
            }
        }
        return "v" + v;
    }

    /// <summary>
    /// The build of <paramref name="builds"/> that plays
    /// <paramref name="version"/>: the build itself, or, for a hotfix (same
    /// major and minor, higher patch: <c>"v0.113.1"</c>), the newest build of
    /// that line whose patch is not above it (<c>"v0.113.0"</c>). The game
    /// ships hotfixes as bug-fix updates, and the advisor runs them on the
    /// base build's engine (<c>sts2_client.hotfix_base</c>). Null when no
    /// build fits: a new minor is a content update, not a hotfix, unless
    /// <see cref="PlaysAs"/> names its line. Both sides are compared in wire
    /// form (<see cref="Normalize"/>).
    /// </summary>
    public static string? EngineBuild(string? version, IEnumerable<string> builds)
    {
        string? wire = Normalize(version);
        if (wire == null || !TryParts(wire, out long major, out long minor, out long patch))
        {
            return null;
        }
        foreach (var (line, engine) in PlaysAs)
        {
            if (line == (major, minor))
            {
                return builds.Select(Normalize).Contains(engine) ? engine : null;
            }
        }
        string? best = null;
        long bestPatch = -1;
        foreach (string b in builds)
        {
            string? bw = Normalize(b);
            if (bw != null && TryParts(bw, out long bMajor, out long bMinor, out long bPatch) &&
                bMajor == major && bMinor == minor && bPatch <= patch && bPatch > bestPatch)
            {
                best = bw;
                bestPatch = bPatch;
            }
        }
        return best;
    }

    /// <summary>
    /// Game lines the advisor has no build of yet, each with the build it
    /// plays on, for every patch of the line. Empty while every released
    /// line has a build (v0.114.x played as v0.113.0 this way until GH
    /// #1924). Twins: <c>sts2_client.PLAYS_AS</c>, Rust
    /// <c>game_version::PLAYS_AS</c>.
    /// </summary>
    public static readonly ((long Major, long Minor) Line, string Engine)[] PlaysAs = { };

    /// <summary>The three numbers of a wire-form build (<c>"v0.113.1"</c>).</summary>
    private static bool TryParts(string wire, out long major, out long minor, out long patch)
    {
        major = minor = patch = 0;
        string[] parts = wire.Length > 1 && wire[0] == 'v' ? wire.Substring(1).Split('.') : Array.Empty<string>();
        return parts.Length == 3
            && parts.All(p => p.Length > 0 && p.Length <= 9 && p.All(char.IsAsciiDigit))
            && long.TryParse(parts[0], out major)
            && long.TryParse(parts[1], out minor)
            && long.TryParse(parts[2], out patch);
    }
}
