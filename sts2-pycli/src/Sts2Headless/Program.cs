using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sts2Headless;

class Program
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    static void Main(string[] args)
    {
        // Prevent unhandled exceptions from crashing the process
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            Console.Error.WriteLine($"[FATAL] Unhandled: {e.ExceptionObject}");
        };
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Console.Error.WriteLine($"[WARN] Unobserved task exception: {e.Exception?.Message}");
            e.SetObserved();
        };

        // Set up assembly resolution to find game DLLs: lib/<GameVersion>/, the
        // build this binary was compiled against (see HeadlessPaths).
        string libDir = HeadlessPaths.LibDir;

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var path = Path.Combine(libDir, name.Name + ".dll");
            if (File.Exists(path))
                return ctx.LoadFromAssemblyPath(Path.GetFullPath(path));

            // Also check game directory (via STS2_GAME_DIR env var)
            var gameDir = Environment.GetEnvironmentVariable("STS2_GAME_DIR") ?? "";
            if (!string.IsNullOrEmpty(gameDir))
            {
                path = Path.Combine(gameDir, name.Name + ".dll");
                if (File.Exists(path))
                    return ctx.LoadFromAssemblyPath(path);
            }

            return null;
        };

        var sim = new RunSimulator();
        WriteLine(new Dictionary<string, object?>
        {
            ["type"] = "ready",
            ["version"] = "0.2.0",
            // The game build this oracle links against (GH #789); the Python
            // client checks it against the version it asked for.
            ["game_version"] = HeadlessPaths.GameVersion,
        });

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line)) continue;

            Dictionary<string, object?>? result;
            try
            {
                var cmd = JsonSerializer.Deserialize<JsonElement>(line);
                result = HandleCommand(sim, cmd);
            }
            catch (JsonException ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Invalid JSON: {ex.Message}" };
            }
            catch (Exception ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"{ex.GetType().Name}: {ex.Message}" };
            }

            if (result != null)
            {
                WriteLine(result);
                if (result.TryGetValue("type", out var resultTypeObj) &&
                    string.Equals(resultTypeObj as string, "quit_result", StringComparison.Ordinal))
                {
                    break;
                }
            }
        }
    }

    static Dictionary<string, object?>? HandleCommand(RunSimulator sim, JsonElement cmd)
    {
        var cmdType = cmd.GetProperty("cmd").GetString() ?? "";
        switch (cmdType)
        {
            case "start_run":
                return sim.StartRun(
                    cmd.TryGetProperty("character", out var ch) ? ch.GetString() ?? "Ironclad" : "Ironclad",
                    cmd.TryGetProperty("ascension", out var asc) ? asc.GetInt32() : 0,
                    cmd.TryGetProperty("seed", out var s) ? s.GetString() : null,
                    cmd.TryGetProperty("lang", out var lang) ? lang.GetString() ?? "en" : "en",
                    cmd.TryGetProperty("played_save", out var ps) && ps.GetBoolean(),
                    // "overgrowth" | "underdocks" | "roll". Omit for the historical
                    // default (GetDefaultList => Overgrowth). See RunSimulator.
                    cmd.TryGetProperty("act1", out var a1) ? a1.GetString() : null,
                    // GH #1069: surface a `potion_swap` decision when a potion
                    // reward finds a full belt, instead of dropping it.
                    cmd.TryGetProperty("interactive_potion_swap", out var ips) && ips.GetBoolean()
                );

            case "roll_acts":
                return sim.RollActs(
                    cmd.TryGetProperty("seed", out var rs) ? rs.GetString() : null,
                    cmd.TryGetProperty("played_save", out var rps) && rps.GetBoolean()
                );

            case "rng_probe":
            {
                var names = new List<string>();
                if (cmd.TryGetProperty("names", out var nm) && nm.ValueKind == JsonValueKind.Array)
                    foreach (var n in nm.EnumerateArray())
                        names.Add(n.GetString() ?? "");
                return sim.RngProbe(
                    cmd.TryGetProperty("seed", out var ps2) ? ps2.GetString() : null,
                    names,
                    cmd.TryGetProperty("draws", out var dr) ? dr.GetInt32() : 4
                );
            }

            case "action":
            {
                var action = cmd.GetProperty("action").GetString() ?? "";
                Dictionary<string, object?>? actionArgs = null;
                if (cmd.TryGetProperty("args", out var argsElem))
                {
                    actionArgs = new Dictionary<string, object?>();
                    foreach (var prop in argsElem.EnumerateObject())
                    {
                        actionArgs[prop.Name] = prop.Value.ValueKind switch
                        {
                            JsonValueKind.Number => prop.Value.GetInt32(),
                            JsonValueKind.String => prop.Value.GetString(),
                            JsonValueKind.True => true,
                            JsonValueKind.False => false,
                            _ => prop.Value.ToString(),
                        };
                    }
                }
                StateTrace.BeginAction(action, actionArgs);
                return sim.ExecuteAction(action, actionArgs);
            }

            case "load_save":
            {
                var savePath = cmd.TryGetProperty("path", out var sp) ? sp.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var sj) ? sj.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Save file not found: {savePath}" };
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return new Dictionary<string, object?> { ["type"] = "error", ["message"] = "Provide 'path' or 'json' for load_save" };
                return sim.LoadSave(saveJson);
            }
            case "get_map":
                return sim.GetFullMap();

            case "dump_rooms":
                return sim.DumpRooms();

            case "dump_maps":
                return sim.DumpMaps();

            case "dump_card_reward":
            {
                var count = cmd.TryGetProperty("count", out var cnt) ? cnt.GetInt32() : 3;
                var rt = cmd.TryGetProperty("room_type", out var rtp) ? (rtp.GetString() ?? "Monster") : "Monster";
                return sim.DumpCardReward(count, rt);
            }

            case "dump_relic_reward":
            {
                var count = cmd.TryGetProperty("count", out var cnt) ? cnt.GetInt32() : 1;
                return sim.DumpRelicReward(count);
            }

            case "dump_shop":
                return sim.DumpShop();

            case "dump_content":
                return sim.DumpContent();

            case "dump_events":
                return sim.DumpEvents();

            case "set_player":
            {
                var args = new Dictionary<string, JsonElement>();
                foreach (var prop in cmd.EnumerateObject())
                    if (prop.Name != "cmd") args[prop.Name] = prop.Value;
                return sim.SetPlayer(args);
            }

            case "enter_room":
            {
                var roomType = cmd.TryGetProperty("type", out var rt) ? rt.GetString() ?? "" : "";
                var encounter = cmd.TryGetProperty("encounter", out var enc) ? enc.GetString() : null;
                var eventId = cmd.TryGetProperty("event", out var ev) ? ev.GetString() : null;
                return sim.EnterRoom(roomType, encounter, eventId);
            }

            case "set_draw_order":
            {
                var cards = new List<string>();
                if (cmd.TryGetProperty("cards", out var cardsArr))
                    foreach (var c in cardsArr.EnumerateArray())
                        cards.Add(c.GetString() ?? "");
                return sim.SetDrawOrder(cards);
            }

            case "legal_actions":
                return sim.LegalActions();

            // GH #1299: hash every game method a host prefix can skip.
            case "override_fingerprints":
                return OverrideFingerprints.Collect(cmd.TryGetProperty("dump", out var dmp) && dmp.GetBoolean());

            case "get_state":
                return sim.GetState();

            // GH #1088 state-mutation trace (see StateTrace).
            case "trace":
            {
                StateTrace.RngStreams = sim.RngStreamMap;
                return StateTrace.SetEnabled(!cmd.TryGetProperty("on", out var on) || on.GetBoolean());
            }

            case "get_trace":
                return StateTrace.GetTrace(cmd.TryGetProperty("since", out var since) ? since.GetInt32() : 0);

            case "write_continue_save":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                return sim.SaveCheckpoint(outputPath);
            }

            case "quit":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                if (!string.IsNullOrEmpty(outputPath))
                {
                    var saveResult = sim.SaveCheckpoint(outputPath);
                    bool saveOk = saveResult.TryGetValue("success", out var sObj) && sObj is bool b && b;
                    if (!saveOk)
                    {
                        // Save failed — do NOT clean up so the caller can retry with a different path.
                        return new Dictionary<string, object?>
                        {
                            ["type"] = "save_error",
                            ["save"] = saveResult,
                        };
                    }
                    sim.CleanUp();
                    return new Dictionary<string, object?>
                    {
                        ["type"] = "quit_result",
                        ["success"] = true,
                        ["save"] = saveResult,
                    };
                }
                sim.CleanUp();
                return new Dictionary<string, object?>
                {
                    ["type"] = "quit_result",
                    ["success"] = true,
                    ["save"] = null,
                };
            }

            default:
                return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Unknown command: {cmdType}" };
        }
    }

    static void WriteLine(Dictionary<string, object?> data)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(data, JsonOpts));
        Console.Out.Flush();
    }
}

/// <summary>
/// Where this oracle build finds its game DLLs and its localization (GH #789).
///
/// The game build is baked in at compile time (<c>-p:GameVersion=…</c> becomes
/// <c>[AssemblyMetadata("GameVersion", …)]</c>), so a binary only ever loads the
/// DLLs it was compiled against: <c>bin/v0.111.0/…</c> resolves
/// <c>lib/v0.111.0/</c>. Neither lookup assumes a fixed depth below the
/// sts2-pycli root — both walk up from the binary, so <c>bin/&lt;ver&gt;/Debug/net9.0/</c>,
/// a flat <c>-o</c> directory and the old <c>bin/Debug/net9.0/</c> all work.
/// </summary>
internal static class HeadlessPaths
{
    public const string DefaultGameVersion = "v0.107.1";

    public static string GameVersion { get; } =
        typeof(HeadlessPaths).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == "GameVersion")?.Value
        ?? DefaultGameVersion;

    private static string? _libDir;

    /// <summary>
    /// The game-DLL directory. In order: <c>&lt;binary dir&gt;/lib</c> (a
    /// self-contained <c>-o</c> layout that carries its DLLs along), the nearest
    /// ancestor's <c>lib/&lt;GameVersion&gt;/</c>, then — v0.107.1 only — a legacy
    /// flat <c>lib/</c> (pre-#789 checkouts; warns on stderr). Otherwise
    /// <c>&lt;binary dir&gt;/lib</c>, leaving the <c>STS2_GAME_DIR</c> fallback to
    /// the assembly resolver.
    /// </summary>
    public static string LibDir => _libDir ??= ResolveLibDir();

    private static string ResolveLibDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "lib");
        if (File.Exists(Path.Combine(local, "sts2.dll")))
            return local;

        string? legacy = null;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            var versioned = Path.Combine(dir.FullName, "lib", GameVersion);
            if (File.Exists(Path.Combine(versioned, "sts2.dll")))
                return versioned;
            var flat = Path.Combine(dir.FullName, "lib");
            if (legacy == null && GameVersion == DefaultGameVersion && File.Exists(Path.Combine(flat, "sts2.dll")))
                legacy = flat;
        }
        if (legacy != null)
        {
            Console.Error.WriteLine(
                $"[WARN] using the legacy flat game-DLL layout {legacy}; move the DLLs to lib/{GameVersion}/ (GH #789)");
            return legacy;
        }
        return local;
    }

    private static string? _locDir;

    /// <summary>
    /// The English localization directory the game's <c>LocManager</c> and the
    /// display-name lookup read. In order: <c>STS2_LOCALIZATION_DIR</c>, a
    /// per-version <c>&lt;LibDir&gt;/localization_eng/</c> (staged by
    /// <c>setup_oracle.py</c> for a build whose text differs from the vendored
    /// tree), then the nearest ancestor's <c>localization_eng/</c> — the vendored
    /// sts2-pycli tree every v0.107.1 build has always read.
    /// </summary>
    public static string LocalizationDir => _locDir ??= ResolveLocalizationDir();

    private static string ResolveLocalizationDir()
    {
        var env = Environment.GetEnvironmentVariable("STS2_LOCALIZATION_DIR");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
            return env;
        var perVersion = Path.Combine(LibDir, "localization_eng");
        if (Directory.Exists(perVersion))
            return perVersion;
        // A non-default build's text is vendored as `localization_eng_<ver>/`
        // beside the default `localization_eng/` (scripts/import_game.py,
        // workbench/localization.py). Without this lookup the v0.111.0 oracle
        // fell through to the v0.107.1 text and showed stale relic descriptions
        // (SERE_TALON's `{Curses}`, DIAMOND_DIADEM's `{CardThreshold}`). A public
        // export ships without the per-version dir and keeps the fallback.
        var versioned = HeadlessPaths.GameVersion == HeadlessPaths.DefaultGameVersion
            ? null
            : $"localization_eng_{HeadlessPaths.GameVersion}";
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            if (versioned != null)
            {
                var own = Path.Combine(dir.FullName, versioned);
                if (Directory.Exists(own))
                    return own;
            }
            var candidate = Path.Combine(dir.FullName, "localization_eng");
            if (Directory.Exists(candidate))
                return candidate;
        }
        // Not found: the historical location (five levels above bin/Debug/net9.0),
        // which the callers then report as missing.
        return Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "localization_eng");
    }
}
