// Program.cs -- host + scenario script for the BridgeServer differential test.
//
// Usage: BridgeHarness --spirementor <SpireMentor.dll> --out <dir>
//                      --godot-stub <GodotSharp.dll(stub)> --sts2 <sts2.dll>
//                      --lib <dir with 0Harmony/MonoMod/Sentry/...> [--baselib <BaseLib.dll>]
//
// Writes <out>/transcript.jsonl (diffed), <out>/server.log and <out>/meta.json
// (not diffed). Exit code 0 when the script ran to completion (regardless of
// what the server did -- behaviour is judged by diffing transcripts).

using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.Loader;
using System.Text;
using System.Text.Json;

namespace BridgeHarness;

static class Program
{
    static Assembly _asm = null!;
    static readonly Dictionary<string, string> Resolved = new();

    static int Main(string[] argv)
    {
        var a = new Dictionary<string, string>();
        var av = argv.Where(x => x != "--no-platform-shim").ToArray();
        for (int i = 0; i + 1 < av.Length; i += 2) a[av[i].TrimStart('-')] = av[i + 1];
        string Need(string k) => a.TryGetValue(k, out var v) ? Path.GetFullPath(v) : throw new ArgumentException($"missing --{k}");
        string dll = Need("spirementor"), outDir = Need("out"), stub = Need("godot-stub"), sts2 = Need("sts2"), lib = Need("lib");
        string? baselib = a.TryGetValue("baselib", out var bl) ? Path.GetFullPath(bl) : null;
        Directory.CreateDirectory(outDir);

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            string? p = name.Name switch
            {
                "GodotSharp" => stub,
                "sts2" => sts2,
                "BaseLib" => baselib,
                _ => null,
            };
            if (p == null)
            {
                foreach (var dir in new[] { lib, Path.GetDirectoryName(dll)! })
                {
                    var c = Path.Combine(dir, name.Name + ".dll");
                    if (File.Exists(c)) { p = c; break; }
                }
            }
            if (p == null || !File.Exists(p)) { lock (Resolved) Resolved[name.Name!] = "(unresolved)"; return null; }
            lock (Resolved) Resolved[name.Name!] = p;
            return ctx.LoadFromAssemblyPath(p);
        };

        _asm = AssemblyLoadContext.Default.LoadFromAssemblyPath(dll);
        LogSink.Open(Path.Combine(outDir, "server.log"));
        T.Open(Path.Combine(outDir, "transcript.jsonl"));
        var notes = new List<string>();
        int rc = 0;
        try
        {
            GameApiSmoke.Run(_asm, notes);
            B.Init(_asm);
            Hooks.Install(_asm, notes);
            bool shim = !argv.Contains("--no-platform-shim");
            if (shim) Hooks.InstallPlatformShim(notes);
            T.Ev("harness_config", new { platform_shim = shim });
            Scenarios.Run(_asm);
        }
        catch (Exception e)
        {
            T.Ev("harness_crash", new { type = e.GetType().Name, message = e.Message });
            Console.Error.WriteLine(e);
            rc = 2;
        }
        finally
        {
            T.Close();
            var godot = AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(x => x.GetName().Name == "GodotSharp");
            File.WriteAllText(Path.Combine(outDir, "meta.json"), JsonSerializer.Serialize(new
            {
                spirementor = dll,
                resolved = Resolved,
                godotsharp_loaded_from = godot?.Location,
                hooks = notes,
                platform_shim_hits = Hooks.ShimHits,
                runtime = Environment.Version.ToString(),
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        Console.Out.Flush();
        // Background accept loops of a stopped/broken server must not keep us alive.
        Environment.Exit(rc);
        return rc;
    }

    public static int FreePort()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        int p = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return p;
    }
}

/// <summary>
/// The members the mod reaches by reflection or emits against, checked at run
/// time against THIS sts2.dll (RefCheck only sees direct IL references): the
/// SimGameApi static initializer (every drifted member it looks up), the
/// v0.113.0+ card selector bridge patches, a watchdog, and both concrete
/// NetHosts SimNetHosts emits. Notes go to meta.json, not the transcript, so
/// the golden is unchanged; a failure crashes the harness (exit 2).
/// </summary>
static class GameApiSmoke
{
    const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;

    public static void Run(Assembly asm, List<string> notes)
    {
        Type api = asm.GetType("SpireMentor.SimGameApi", throwOnError: true)!;
        System.Runtime.CompilerServices.RuntimeHelpers.RunClassConstructor(api.TypeHandle);
        notes.Add("api_smoke: SimGameApi initialised");

        Type harmonyType = AppDomain.CurrentDomain.GetAssemblies()
            .Select(x => x.GetType("HarmonyLib.Harmony"))
            .FirstOrDefault(t => t != null)
            ?? Assembly.Load("0Harmony").GetType("HarmonyLib.Harmony", throwOnError: true)!;
        object harmony = Activator.CreateInstance(harmonyType, "spirementor.wire.apismoke")!;
        Call(api, "InstallSelectorBridge", harmony);
        notes.Add("api_smoke: selector bridge installed");

        Call(api, "NewWatchdog");
        notes.Add("api_smoke: watchdog constructed");

        Type hosts = asm.GetType("SpireMentor.SimNetHosts", throwOnError: true)!;
        foreach (string name in new[] { "SpireMentor.SimLoopbackHost", "SpireMentor.SimMixedHost" })
        {
            var concrete = (Type)Call(hosts, "Build", asm.GetType(name, throwOnError: true)!)!;
            System.Runtime.CompilerServices.RuntimeHelpers.RunClassConstructor(concrete.TypeHandle);
            if (concrete.IsAbstract) throw new InvalidOperationException($"{concrete} is still abstract");
            notes.Add($"api_smoke: {name} -> concrete {concrete.Name}");
        }
    }

    static object? Call(Type t, string method, params object[] args)
    {
        MethodInfo m = t.GetMethod(method, Any) ?? throw new MissingMethodException(t.Name, method);
        try
        {
            return m.Invoke(null, args);
        }
        catch (TargetInvocationException e) when (e.InnerException != null)
        {
            throw new InvalidOperationException($"api_smoke: {t.Name}.{method} threw: {e.InnerException}", e.InnerException);
        }
    }
}

static class Scenarios
{
    static string? Rid(string? line)
    {
        if (line == null) return null;
        try
        {
            using var d = JsonDocument.Parse(line);
            if (d.RootElement.ValueKind == JsonValueKind.Object &&
                d.RootElement.TryGetProperty("request_id", out var r) && r.ValueKind == JsonValueKind.String)
                return r.GetString();
        }
        catch { }
        return null;
    }

    static void Barrier(Client c)
    {
        c.Send("{\"action\":\"ping\"}\n", "barrier");
        c.Expect(1);
        Hooks.Drain();
    }

    static void WaitConn(bool expect, int timeoutMs = 3000)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        bool v = B.IsClientConnectedRaw();
        while (v != expect && sw.ElapsedMilliseconds < timeoutMs) { Thread.Sleep(10); v = B.IsClientConnectedRaw(); }
        T.Ev("is_client_connected", new { expect, observed = v, ok = v == expect });
    }

    static string Choose(string rid, int index = 0) => $"{{\"action\":\"choose\",\"index\":{index},\"request_id\":\"{rid}\"}}\n";
    static readonly TimeSpan Long = TimeSpan.FromSeconds(5);
    static readonly TimeSpan Short = TimeSpan.FromMilliseconds(1000);

    public static void Run(Assembly asm)
    {
        int port = Program.FreePort(), port2 = Program.FreePort();
        B.PortNames[port] = "<PORT>"; B.PortNames[port2] = "<PORT2>";
        LogSink.Normalizers.Add((port.ToString(), "<PORT>"));
        LogSink.Normalizers.Add((port2.ToString(), "<PORT2>"));

        // ------------------------------------------------------------------
        T.Scenario("00_pre_start", "API calls before Start(): no listener, no client");
        B.IsClientConnected();
        B.SendState("{\"type\":\"map_select\"}");
        B.SendStateAndWait("p0", "{\"type\":\"combat_action\"}", Short).Await("no_client");
        B.SendAdviceRequest("{\"type\":\"advice_request\"}", "adv_pre");
        B.CancelAdviceRequest("a999");
        var pw0 = B.WaitForAction("p0w", Short);
        pw0.Check("t+150ms");
        pw0.Await("timeout");
        Hooks.Drain();

        // ------------------------------------------------------------------
        T.Scenario("01_start_connect", "Start, connect, no greeting/handshake, idempotent Start");
        B.Start(port);
        var A = new Client("A");
        A.Connect(port);
        WaitConn(true);
        A.Drain(400); // server sends nothing on connect
        B.IsClientConnected();
        B.Start(port);   // already running
        B.Start(port2);  // already running, different port
        var X = new Client("X");
        if (X.Connect(port2)) X.Forget();
        Barrier(A);

        // ------------------------------------------------------------------
        T.Scenario("02_outbound_types", "SendState passes every PROTOCOL.md outbound message type through as one line");
        string env = "\"seed\":\"ABC123\",\"act\":1,\"floor\":3,\"mod_version\":\"0.1.0\",\"state_id\":41,\"run_state\":{\"hp\":70,\"max_hp\":80,\"gold\":99}";
        var types = new (string t, string extra)[]
        {
            ("combat_action", "\"hand\":[{\"index\":0,\"id\":\"STRIKE_IRONCLAD\",\"cost\":1,\"playable\":true}],\"energy\":3,\"enemies\":[{\"index\":0,\"hp\":40}]"),
            ("map_select", "\"nodes\":[{\"col\":1,\"row\":2,\"type\":\"monster\"}]"),
            ("card_select", "\"cards\":[],\"min_count\":1,\"max_count\":1,\"min_select\":1,\"max_select\":1,\"cancelable\":false"),
            ("card_reward", "\"cards\":[{\"id\":\"BASH\"}],\"can_skip\":true"),
            ("card_bundle", "\"bundles\":[[{\"id\":\"A\"}],[{\"id\":\"B\"}]]"),
            ("crystal_sphere", "\"options\":[]"),
            ("rest_site", "\"options\":[{\"id\":\"HEAL\"},{\"id\":\"SMITH\"}]"),
            ("shop", "\"cards\":[],\"relics\":[],\"potions\":[],\"remove_cost\":75"),
            ("event", "\"event_id\":\"NEOW\",\"options\":[{\"index\":0,\"text\":\"Take it\"}]"),
            ("treasure", "\"options\":[]"),
            ("boss_relic", "\"relics\":[]"),
            ("reward_screen", "\"rewards\":[{\"type\":\"gold\",\"amount\":15}]"),
            ("game_over", "\"victory\":false"),
            ("combat_ended", "\"victory\":true,\"round\":4,\"hp\":61,\"max_hp\":80,\"room_type\":\"Monster\""),
            ("run_complete", "\"victory\":true"),
            ("run_config", "\"characters\":[{\"id\":\"IRONCLAD\",\"name\":\"The Ironclad\"}],\"max_ascension\":10"),
            ("error", "\"error\":\"invalid_index\",\"detail\":\"index 9 out of range\""),
            ("advice_request", "\"advice_for\":\"combat_action\",\"turn\":2"),
            ("human_action", "\"action\":{\"action\":\"play\",\"card_index\":0}"),
            ("human_action", "\"decision_type\":\"combat_action\",\"action\":\"end_turn\",\"detail\":\"end turn\",\"advice_request_id\":\"a7\",\"position_id\":41,\"peeked\":false,\"card\":null,\"potion\":null,\"target_index\":null"),
            ("human_action", "\"decision_type\":\"card_reward\",\"action\":\"choose\",\"detail\":\"Inflame\",\"advice_request_id\":null,\"position_id\":41,\"peeked\":true,\"choice\":{\"index\":1,\"label\":\"Inflame\"}"),
            ("human_action", "\"decision_type\":\"card_reward\",\"action\":\"skip\",\"detail\":\"Skip\",\"advice_request_id\":\"a8\",\"position_id\":41,\"peeked\":false,\"choice\":{\"index\":null,\"label\":\"Skip\"}"),
            ("snapshot_saved", "\"snapshot_id\":\"s1\",\"seq\":1,\"post_combat\":false,\"run_start_time\":1,\"ts\":\"t\""),
            ("snapshots", "\"run_start_time\":1,\"snapshots\":[]"),
            ("snapshot_restored", "\"snapshot_id\":\"s1\""),
            ("combat_undone", "\"turn\":2,\"tape_id\":null,\"replayed_events\":5"),
        };
        foreach (var (t, extra) in types)
        {
            B.SendState($"{{\"type\":\"{t}\",{env},{extra}}}");
            A.Expect(1);
        }
        B.SendState("{\"type\":\"pong\"}");
        A.Expect(1);

        // ------------------------------------------------------------------
        T.Scenario("03_sendstate_framing", "SendState framing: payload is written verbatim + \\n (no re-serialization, no validation)");
        B.SendState("{\n  \"type\": \"event\",\n  \"pretty\": true\n}");
        A.Drain(300);
        B.SendState("");
        A.Drain(300);
        B.SendState("not json at all");
        A.Drain(300);
        B.SendState("{\"type\":\"x\"}\n");
        A.Drain(300);
        B.SendState("{\"type\":\"x\"}\r\n");
        A.Drain(300);
        B.SendState("{\"type\":\"event\",\"label\":\"Bash → Cultist <b>&amp;</b> 'q' \\\"dq\\\" é 日本 😀\",\"esc\":\"\\u00e9\\/\"}");
        A.Drain(300);
        B.SendState("[1,2,3]");
        A.Drain(300);

        // ------------------------------------------------------------------
        T.Scenario("04_request_id_injection", "SendStateAndWaitForActionAsync: request_id injection / re-serialization of the payload");
        var objPayloads = new[]
        {
            "{\"type\":\"combat_action\",\"state_id\":7,\"hand\":[{\"id\":\"STRIKE\",\"cost\":1}],\"energy\":3}",
            "{\"type\":\"map_select\",\"request_id\":\"999\",\"x\":1}",
            "{\"request_id\":5,\"type\":\"event\"}",
            "{ \"type\" : \"shop\",\n  \"gold\": 100 ,\"f\":1.50,\"e\":1e3,\"big\":12345678901234567890,\"neg\":-0.0,\"n\":null,\"t\":true,\"arr\":[ 1 , {\"k\":[]} ]}",
            "{\"type\":\"event\",\"label\":\"Bash → Cultist <b>&amp;</b> 'q' \\\"dq\\\" é 日本 😀\",\"esc\":\"\\u00e9\\/\"}",
            "{\"type\":\"a\",\"type\":\"b\"}",
            "{}",
            "null",
        };
        int k = 0;
        foreach (var pl in objPayloads)
        {
            var p = B.SendStateAndWait($"p4_{k++}", pl, Long);
            var got = A.Expect(1);
            p.Check("after_request_seen");
            string? rid = Rid(got.FirstOrDefault());
            if (rid != null)
            {
                A.Send(Choose(rid));
                p.Await("after_reply");
            }
            else
            {
                T.Ev("note", new { text = "no string request_id on the wire; cannot correlate" });
                p.Await("give_up", 6000);
            }
        }
        foreach (var pl in new[] { "[1,2,3]", "\"str\"", "not json", "" })
        {
            var p = B.SendStateAndWait($"p4_{k++}", pl, Short);
            A.Drain(250);
            p.Check("after_request_seen");
            A.Send("{\"action\":\"choose\",\"index\":0}\n", "uncorrelated reply");
            p.Await("timeout");
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("05_correlation", "single pending slot: stale/missing/typed ids, supersede, cancel, timeout");
        {
            // a) wrong id dropped, then right id resolves
            var p = B.SendStateAndWait("p5a", "{\"type\":\"card_reward\"}", Long);
            string rid = Rid(A.Expect(1).FirstOrDefault())!;
            A.Send("{\"action\":\"choose\",\"index\":1,\"request_id\":\"424242\"}\n", "wrong id");
            Barrier(A);
            p.Check("after_wrong_id");
            // b) missing id
            A.Send("{\"action\":\"choose\",\"index\":1}\n", "missing id");
            Barrier(A);
            p.Check("after_missing_id");
            // c) numeric id equal to the pending one
            A.Send($"{{\"action\":\"choose\",\"index\":1,\"request_id\":{rid}}}\n", "numeric id");
            Barrier(A);
            p.Check("after_numeric_id");
            // d) unparsable line while correlated
            A.Send("{\"action\":\"choose\",\"index\":\n", "malformed json");
            Barrier(A);
            p.Check("after_malformed");
            // i) ping with the pending id: pong, not consumed
            A.Send($"{{\"action\":\"ping\",\"request_id\":\"{rid}\"}}\n");
            A.Expect(1);
            p.Check("after_ping_with_id");
            // j) advice reply carrying the autopilot id: routed to advice lane, not consumed
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{rid}\",\"advice\":{{}}}}\n");
            Barrier(A);
            p.Check("after_advice_with_autopilot_id");
            // k) list_snapshots while pending
            A.Send("{\"action\":\"list_snapshots\"}\n");
            A.Expect(1);
            p.Check("after_list_snapshots");
            // g) padded + CRLF reply resolves (trimmed)
            A.Send($"   {{\"action\":\"choose\",\"index\":2,\"request_id\":\"{rid}\"}}  \t\r\n");
            p.Await("after_correct_reply");
            // e) duplicate reply after resolution
            A.Send(Choose(rid));
            Barrier(A);
        }
        {
            // f) supersede: second request cancels the first
            var p1 = B.SendStateAndWait("p5f1", "{\"type\":\"event\"}", Long);
            string r1 = Rid(A.Expect(1).FirstOrDefault())!;
            var p2 = B.SendStateAndWait("p5f2", "{\"type\":\"event\"}", Long);
            string r2 = Rid(A.Expect(1).FirstOrDefault())!;
            p1.Check("after_second_request");
            A.Send(Choose(r1), "reply to superseded id");
            Barrier(A);
            p2.Check("after_reply_to_superseded");
            A.Send(Choose(r2));
            p2.Await("after_reply");
        }
        {
            // every client verb from PROTOCOL.md section 4 passes through verbatim
            var verbs = new[]
            {
                "{\"action\":\"play\",\"card_index\":0,\"target_index\":1,\"state_id\":41,\"request_id\":\"%\"}",
                "{\"action\":\"end_turn\",\"request_id\":\"%\"}",
                "{\"action\":\"potion\",\"slot\":0,\"target_index\":0,\"discard\":false,\"request_id\":\"%\"}",
                "{\"action\":\"choose\",\"indices\":[0,2],\"request_id\":\"%\"}",
                "{\"action\":\"choose\",\"indexes\":[1],\"request_id\":\"%\"}",
                "{\"action\":\"skip\",\"request_id\":\"%\"}",
                "{\"action\":\"cancel\",\"request_id\":\"%\"}",
                "{\"action\":\"start_run\",\"character\":\"IRONCLAD\",\"ascension\":0,\"seed\":null,\"request_id\":\"%\"}",
                "{\"action\":\"end_turn\",\"annotations\":{\"note\":\"é→\",\"arrows\":[1]},\"request_id\":\"%\"}",
                "{\"request_id\":\"%\",\"action\":\"PLAY\",\"card_index\":3}",
                "{\"request_id\":\"%\"}",
            };
            int v = 0;
            foreach (var verb in verbs)
            {
                var p = B.SendStateAndWait($"p5v_{v++}", "{\"type\":\"combat_action\"}", Long);
                string r = Rid(A.Expect(1).FirstOrDefault())!;
                A.Send(verb.Replace("%", r) + "\n");
                p.Await("after_reply");
            }
        }
        {
            // m) cancellation token
            var cts = new CancellationTokenSource();
            var p = B.SendStateAndWait("p5m", "{\"type\":\"shop\"}", Long, cts.Token, "cancelled_later");
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            p.Check("before_cancel");
            cts.Cancel();
            T.Ev("harness", new { action = "cancel_token" });
            p.Await("after_cancel");
            A.Send(Choose(r), "late reply after cancel");
            Barrier(A);
            // n) timeout
            var pt = B.SendStateAndWait("p5n", "{\"type\":\"shop\"}", Short);
            string rt = Rid(A.Expect(1).FirstOrDefault())!;
            pt.Check("t+150ms");
            pt.Await("timeout");
            A.Send(Choose(rt), "late reply after timeout");
            Barrier(A);
            // o) zero timeout
            var pz = B.SendStateAndWait("p5o", "{\"type\":\"shop\"}", TimeSpan.Zero);
            A.Expect(1);
            pz.Await("zero_timeout");
            Barrier(A);
            // o2) infinite timeout
            var pi = B.SendStateAndWait("p5o2", "{\"type\":\"shop\"}", Timeout.InfiniteTimeSpan);
            string ri = Rid(A.Expect(1).FirstOrDefault())!;
            pi.Check("t+150ms");
            A.Send(Choose(ri));
            pi.Await("after_reply");
            // p) pre-cancelled token
            var pre = new CancellationTokenSource(); pre.Cancel();
            var pp = B.SendStateAndWait("p5p", "{\"type\":\"shop\"}", Long, pre.Token, "pre_cancelled");
            A.Drain(300);
            pp.Await("pre_cancelled");
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("06_wait_for_action", "WaitForActionAsync (uncorrelated): any line is delivered, specials are not");
        foreach (var line in new[]
        {
            "{\"action\":\"end_turn\"}\n",
            "{oops not json\n",
            "{\"action\":\"choose\",\"index\":0,\"request_id\":\"zzz\"}\n",
            "\"just a string\"\n",
            "[1,2]\n",
            "{\"action\":5}\n",
            "{\"action\":null,\"x\":1}\n",
        })
        {
            var p = B.WaitForAction("p6", Long);
            A.Send(line);
            p.Await("after_line");
        }
        {
            var p = B.WaitForAction("p6s", Long);
            A.Send("{\"action\":\"ping\"}\n");
            A.Expect(1);
            A.Send("{\"action\":\"advice\",\"request_id\":\"a12345\"}\n");
            A.Send("{\"action\":\"list_snapshots\"}\n");
            A.Expect(1);
            A.Send("\n\n   \n\r\n");
            Barrier(A);
            p.Check("after_specials_and_blank_lines");
            A.Send("{\"action\":\"skip\"}\n");
            p.Await("after_real_line");
        }
        {
            var p = B.WaitForAction("p6t", Short);
            p.Check("t+150ms");
            p.Await("timeout");
            // WaitForAction supersedes a pending SendStateAndWait and vice versa
            var ps = B.SendStateAndWait("p6x1", "{\"type\":\"event\"}", Long);
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            var pw2 = B.WaitForAction("p6x2", Long);
            ps.Check("after_wait_for_action_started");
            A.Send(Choose(r));
            pw2.Await("after_reply");
            var pw3 = B.WaitForAction("p6x3", Long);
            var ps2 = B.SendStateAndWait("p6x4", "{\"type\":\"event\"}", Long);
            string r2 = Rid(A.Expect(1).FirstOrDefault())!;
            pw3.Check("after_send_and_wait_started");
            A.Send(Choose(r2));
            ps2.Await("after_reply");
        }

        // ------------------------------------------------------------------
        T.Scenario("07_no_handler", "client lines with nothing pending are discarded silently");
        foreach (var line in new[]
        {
            "{\"action\":\"end_turn\"}\n",
            "{\"action\":\"choose\",\"index\":0,\"request_id\":\"7\"}\n",
            "garbage\n",
            "{\"action\":\"start_run\",\"character\":\"IRONCLAD\",\"ascension\":0}\n",
        })
        {
            A.Send(line);
            A.Drain(250);
        }
        Barrier(A);

        // ------------------------------------------------------------------
        T.Scenario("08_ping_variants", "ping / PING spellings and near-misses");
        foreach (var line in new[]
        {
            "{\"action\":\"ping\"}", "{\"action\":\"PING\"}", "{\"action\":\"Ping\",\"x\":1}",
            "{\"type\":\"PING\"}", "{\"type\":\"ping\"}", "{\"type\":\"Ping\"}",
            "{\"type\":\"PING\",\"action\":\"play\"}", "{\"action\":\"ping\",\"type\":\"x\"}",
            "{\"action\":\"pingx\"}", "{\"type\":\"pong\"}", "{\"action\":5,\"type\":\"PING\"}",
            "{\"action\":null,\"type\":\"PING\"}", "{\"type\":5}", "[]", "\"ping\"", "null", "42",
            "{\"action\":\"ping\"}\r{\"action\":\"ping\"}",
        })
        {
            A.Send(line + "\n");
            A.Drain(250);
        }
        Barrier(A);

        // ------------------------------------------------------------------
        T.Scenario("09_rewind_lane", "list_snapshots / restore_snapshot / undo_turn routing");
        A.Send("{\"action\":\"list_snapshots\"}\n");
        A.Expect(1);
        T.Ev("harness", Hooks.SeedSnapshots(asm));
        foreach (var line in new[]
        {
            "{\"action\":\"list_snapshots\"}", "{\"action\":\"LIST_SNAPSHOTS\",\"extra\":1}",
            "{\"action\":\"restore_snapshot\",\"snapshot_id\":\"1760000000_0002\"}",
            "{\"action\":\"Restore_Snapshot\",\"snapshot_id\":\"nope\"}",
            "{\"action\":\"restore_snapshot\"}", "{\"action\":\"restore_snapshot\",\"snapshot_id\":\"\"}",
            "{\"action\":\"restore_snapshot\",\"snapshot_id\":null}", "{\"action\":\"restore_snapshot\",\"snapshot_id\":7}",
            "{\"action\":\"undo_turn\"}", "{\"action\":\"undo_turn\",\"turn\":3}", "{\"action\":\"UNDO_TURN\",\"turn\":2,\"tape_id\":\"1760000000_0002\"}",
            "{\"action\":\"undo_turn\",\"turn\":2.5}", "{\"action\":\"undo_turn\",\"turn\":\"3\"}", "{\"action\":\"undo_turn\",\"tape_id\":5}",
            "{\"action\":\"undo_turn\",\"turn\":-1,\"tape_id\":null}",
        })
        {
            A.Send(line + "\n");
            A.Drain(250);
            Hooks.Drain();
        }
        {
            var p = B.SendStateAndWait("p9", "{\"type\":\"rest_site\"}", Long);
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            A.Send("{\"action\":\"restore_snapshot\",\"snapshot_id\":\"1760000000_0001\"}\n");
            A.Send("{\"action\":\"undo_turn\",\"turn\":1}\n");
            Barrier(A);
            p.Check("after_rewind_verbs");
            A.Send(Choose(r));
            p.Await("after_reply");
        }

        // ------------------------------------------------------------------
        T.Scenario("10_advice_lane", "SendAdviceRequest / CancelAdviceRequest / advice replies");
        {
            string? a1 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"advice_for\":\"combat_action\",\"state_id\":3}", "adv1");
            A.Expect(1);
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a1}\",\"state_id\":3,\"advice\":{{\"note\":\"MCTS\",\"p_win\":0.73,\"best\":{{\"label\":\"Bash → Cultist\"}}}}}}\n");
            Barrier(A);
            // overlapping, answered in reverse order
            string? a2 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":2}", "adv2");
            string? a3 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":3}", "adv3");
            A.Expect(2);
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a3}\",\"advice\":{{}}}}\n{{\"action\":\"ADVICE\",\"request_id\":\"{a2}\",\"advice\":{{}}}}\n");
            Barrier(A);
            // duplicate, unknown, missing, numeric ids
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a2}\"}}\n");
            A.Send("{\"action\":\"advice\",\"request_id\":\"a999\"}\n");
            A.Send("{\"action\":\"advice\"}\n");
            A.Send("{\"action\":\"advice\",\"request_id\":4}\n");
            Barrier(A);
            // cancel then reply
            string? a4 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":4}", "adv4");
            A.Expect(1);
            B.CancelAdviceRequest(a4 ?? "null");
            B.CancelAdviceRequest("a-never-existed");
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a4}\"}}\n");
            Barrier(A);
            // throwing callback
            string? a5 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":5}", "adv5_throws", throwInCallback: true);
            A.Expect(1);
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a5}\"}}\n");
            Barrier(A);
            // advice + autopilot interleaving
            var p = B.SendStateAndWait("p10", "{\"type\":\"combat_action\"}", Long);
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            string? a6 = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":6}", "adv6");
            A.Expect(1);
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{a6}\"}}\n");
            Barrier(A);
            p.Check("after_advice_reply");
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{r}\"}}\n", "advice verb with autopilot id");
            Barrier(A);
            p.Check("after_advice_with_autopilot_id");
            A.Send(Choose(r));
            p.Await("after_reply");
            // non-object payload
            B.SendAdviceRequest("[1,2]", "adv_nonobject");
            A.Drain(250);
            B.SendAdviceRequest("not json", "adv_notjson");
            A.Drain(250);
            // backlog: 33 outstanding requests
            var ids = new List<string?>();
            for (int i = 0; i < 33; i++)
                ids.Add(B.SendAdviceRequest($"{{\"type\":\"advice_request\",\"n\":{100 + i}}}", $"backlog{i}"));
            var bl = A.ExpectQuiet(33);
            T.Ev("burst_summary", new { client = "A", expected = 33, got = bl.Count, rids = bl.Select(Rid).ToList() });
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{ids[0]}\"}}\n", "oldest backlog entry");
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{ids[31]}\"}}\n", "32nd backlog entry");
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{ids[32]}\"}}\n", "newest backlog entry");
            Barrier(A);
            // pending advice survives? -> cleared on disconnect (checked in 14)
        }

        // ------------------------------------------------------------------
        T.Scenario("11_framing", "partial lines, coalesced lines, CRLF, blank lines, UTF-8 split, invalid bytes");
        {
            byte[] U(string s) => Encoding.UTF8.GetBytes(s);
            A.SendChunks(new[] { U("{\"act"), U("ion\":\"pi"), U("ng\"}") }, note: "no newline yet");
            A.Drain(300);
            A.SendChunks(new[] { U("\n") }, note: "newline completes it");
            A.Expect(1);
            A.Send("{\"action\":\"ping\"}\n{\"action\":\"PING\"}\n{\"type\":\"PING\"}\n{\"action\":\"list_snapshots\"}\n", "4 messages in one write");
            A.Expect(4);
            A.SendChunks(new[] { U("{\"action\":\"ping\"}\n{\"act"), U("ion\":\"ping\"}\n") }, note: "boundary inside 2nd message");
            A.Expect(2);
            A.Send("{\"action\":\"ping\"}\r\n\r\n\n   \n\t{\"action\":\"ping\"}   \r\n", "CRLF + blank lines + padding");
            A.Expect(2);
            A.Drain(250);
            var many = new StringBuilder();
            for (int i = 0; i < 50; i++) many.Append("{\"action\":\"ping\"}\n");
            A.Send(many.ToString(), "50 pings in one write");
            var pongs = A.ExpectQuiet(50);
            T.Ev("burst_summary", new { client = "A", expected = 50, got = pongs.Count, distinct = pongs.Distinct().ToList() });
            A.Drain(250);

            // UTF-8 multi-byte char split across segments
            var full = U("{\"action\":\"choose\",\"label\":\"→é\"}\n");
            int cut = Array.IndexOf(full, (byte)0xE2) + 1; // inside the 3-byte arrow
            var p = B.WaitForAction("p11u", Long);
            A.SendChunks(new[] { full[..cut], full[cut..] }, note: "split inside U+2192");
            p.Await("after_line");
            var p2 = B.WaitForAction("p11u2", Long);
            A.Send("{\"action\":\"choose\",\"label\":\"→é\"}\n", "same, one segment (control)");
            p2.Await("after_line");
            // invalid UTF-8 and NUL
            var p3 = B.WaitForAction("p11b", Long);
            A.SendBytes(U("{\"action\":\"x\",\"b\":\"").Concat(new byte[] { 0xFF, 0xFE }).Concat(U("\"}\n")).ToArray(), "invalid utf-8 bytes FF FE inside string");
            p3.Await("after_line");
            var p4 = B.WaitForAction("p11n", Long);
            A.SendBytes(U("{\"action\":\"x\"}").Concat(new byte[] { 0 }).Concat(U("\n")).ToArray(), "NUL before newline");
            p4.Await("after_line");
            // correlated request: reply split across segments
            var p5 = B.SendStateAndWait("p11s", "{\"type\":\"event\"}", Long);
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            var rb = U(Choose(r));
            A.SendChunks(new[] { rb[..5], rb[5..20], rb[20..] }, note: "reply in 3 segments");
            p5.Await("after_reply");
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("12_oversized", "very long lines in both directions");
        {
            string Pad(int n) => new string('x', n);
            var p = B.WaitForAction("p12a", Long);
            A.Send("{\"action\":\"play\",\"pad\":\"" + Pad(64 * 1024) + "\"}\n", "64 KiB line");
            p.Await("after_line");
            var p2 = B.WaitForAction("p12b", TimeSpan.FromSeconds(15));
            A.Send("{\"action\":\"play\",\"pad\":\"" + Pad(1024 * 1024) + "\"}\n", "1 MiB line");
            p2.Await("after_line", 15000);
            A.Send("{\"action\":\"ping\",\"pad\":\"" + Pad(1024 * 1024) + "\"}\n", "1 MiB ping");
            A.Expect(1, 15000);
            B.SendState("{\"type\":\"big\",\"pad\":\"" + Pad(1024 * 1024) + "\"}");
            A.Expect(1, 10000);
            var p3 = B.SendStateAndWait("p12c", "{\"type\":\"big\",\"pad\":\"" + Pad(256 * 1024) + "\"}", Long);
            string? r = Rid(A.Expect(1, 10000).FirstOrDefault());
            A.Send(Choose(r ?? "?"));
            p3.Await("after_reply");
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("13_concurrent_send", "8 threads x 25 SendState: lines never interleave");
        {
            const int threads = 8, per = 25;
            var go = new ManualResetEventSlim();
            var ths = Enumerable.Range(0, threads).Select(t => new Thread(() =>
            {
                go.Wait();
                for (int j = 0; j < per; j++) B.SendStateQuiet($"{{\"type\":\"burst\",\"t\":{t},\"n\":{j}}}");
            })).ToList();
            ths.ForEach(t => t.Start());
            go.Set();
            ths.ForEach(t => t.Join());
            var lines = A.ExpectQuiet(threads * per);
            int intact = 0; bool perThreadOrdered = true; var last = new int[threads]; Array.Fill(last, -1);
            var seen = new HashSet<string>();
            foreach (var l in lines)
            {
                try
                {
                    using var d = JsonDocument.Parse(l);
                    int t = d.RootElement.GetProperty("t").GetInt32(), n = d.RootElement.GetProperty("n").GetInt32();
                    if (l == $"{{\"type\":\"burst\",\"t\":{t},\"n\":{n}}}") intact++;
                    if (n <= last[t]) perThreadOrdered = false;
                    last[t] = n; seen.Add(l);
                }
                catch { }
            }
            T.Ev("burst_summary", new { client = "A", expected = threads * per, got = lines.Count, intact, distinct = seen.Count, perThreadOrdered });
            A.Drain(250);
        }

        // ------------------------------------------------------------------
        T.Scenario("14_disconnect_reconnect", "client disconnects mid-request, no-agent behaviour, reconnect");
        {
            var p = B.SendStateAndWait("p14a", "{\"type\":\"map_select\"}", Long);
            string r = Rid(A.Expect(1).FirstOrDefault())!;
            string? adv = B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":14}", "adv14");
            A.Expect(1);
            p.Check("before_disconnect");
            A.Close();
            p.Await("after_client_close");
            WaitConn(false);
            B.IsClientConnected();
            // while disconnected
            B.SendState("{\"type\":\"lost\"}");
            B.SendStateAndWait("p14b", "{\"type\":\"map_select\"}", Short).Await("no_client");
            B.SendAdviceRequest("{\"type\":\"advice_request\"}", "adv14_noclient");
            // WaitForAction without a client keeps waiting; a client that connects in time can answer it
            var pw = B.WaitForAction("p14c", Long);
            pw.Check("no_client_yet");
            A = new Client("A2");
            A.Connect(port);
            WaitConn(true);
            A.Drain(300);
            A.Send("{\"action\":\"end_turn\"}\n");
            pw.Await("after_line_from_new_client");
            // stale ids from the old connection
            A.Send(Choose(r), "reply to request from previous connection");
            A.Send($"{{\"action\":\"advice\",\"request_id\":\"{adv}\"}}\n", "advice for request from previous connection");
            Barrier(A);
            // RST while pending
            var pr = B.SendStateAndWait("p14d", "{\"type\":\"map_select\"}", Long);
            A.Expect(1);
            A.Reset();
            pr.Await("after_client_reset");
            WaitConn(false);
            // half-close while pending
            A = new Client("A3");
            A.Connect(port);
            WaitConn(true);
            var ph = B.SendStateAndWait("p14e", "{\"type\":\"map_select\"}", Long);
            A.Expect(1);
            A.ShutdownSend();
            ph.Await("after_client_half_close");
            A.ExpectEof();
            WaitConn(false);
            A.Forget();
            // trailing partial line, then close: not delivered
            A = new Client("A4");
            A.Connect(port);
            WaitConn(true);
            var pt = B.WaitForAction("p14f", Long);
            A.Send("{\"action\":\"end_turn\"}", "no trailing newline");
            A.Drain(250);
            pt.Check("before_close");
            A.Close();
            pt.Await("after_close");
            WaitConn(false);
            A = new Client("A5");
            A.Connect(port);
            WaitConn(true);
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("15_second_client", "one client at a time; a second connection queues until the first leaves");
        {
            var Bc = new Client("B");
            Bc.Connect(port);
            Bc.Send("{\"action\":\"ping\"}\n", "queued ping");
            Bc.Drain(400);
            B.SendState("{\"type\":\"probe\",\"n\":1}");
            A.Expect(1);
            Bc.Drain(300);
            Barrier(A);
            A.Close();
            Bc.Expect(1); // its queued ping is answered once accepted
            WaitConn(true);
            B.SendState("{\"type\":\"probe\",\"n\":2}");
            Bc.Expect(1);
            var C = new Client("C");
            C.Connect(port);
            C.Close();
            Bc.Close();
            WaitConn(false);
            Thread.Sleep(300);
            A = new Client("D");
            A.Connect(port);
            WaitConn(true);
            Barrier(A);
        }

        // ------------------------------------------------------------------
        T.Scenario("16_stop_pending", "Stop() while a request and an advice request are pending");
        {
            var p = B.SendStateAndWait("p16", "{\"type\":\"combat_action\"}", Long);
            A.Expect(1);
            B.SendAdviceRequest("{\"type\":\"advice_request\",\"n\":16}", "adv16");
            A.Expect(1);
            var pw = B.WaitForAction("p16w", Long);
            p.Check("after_wait_for_action_superseded_it");
            p = B.SendStateAndWait("p16b", "{\"type\":\"combat_action\"}", Long);
            A.Expect(1);
            pw.Check("after_send_and_wait_superseded_it");
            B.Stop();
            p.Await("after_stop");
            A.ExpectEof();
            Hooks.Drain();
            WaitConn(false);
            B.IsClientConnected();
            var Z = new Client("Z");
            if (Z.Connect(port)) { Z.Drain(300); Z.Forget(); }
            B.SendState("{\"type\":\"after_stop\"}");
            B.SendStateAndWait("p16c", "{\"type\":\"x\"}", Short).Await("stopped");
            B.SendAdviceRequest("{\"type\":\"advice_request\"}", "adv_after_stop");
            var pw2 = B.WaitForAction("p16d", Short);
            pw2.Await("stopped");
            B.Stop();
        }

        // ------------------------------------------------------------------
        int port3 = Program.FreePort();
        B.PortNames[port3] = "<PORT3>"; LogSink.Normalizers.Add((port3.ToString(), "<PORT3>"));
        T.Scenario("17_restart_after_stop", "Start() again on a new port after Stop()");
        {
            B.Start(port3);
            var E = new Client("E");
            if (E.Connect(port3))
            {
                WaitConn(true);
                Barrier(E);
                var p = B.SendStateAndWait("p17", "{\"type\":\"event\"}", Long);
                string? r = Rid(E.Expect(1).FirstOrDefault());
                E.Send(Choose(r ?? "?"));
                p.Await("after_reply");
                E.Close();
                WaitConn(false);
            }
            B.Stop();
        }

        // ------------------------------------------------------------------
        int busy = Program.FreePort(), port5 = Program.FreePort();
        B.PortNames[busy] = "<BUSY>"; B.PortNames[port5] = "<PORT5>";
        LogSink.Normalizers.Add((busy.ToString(), "<BUSY>")); LogSink.Normalizers.Add((port5.ToString(), "<PORT5>"));
        T.Scenario("18_start_busy_port", "Start() on a port already bound, then Start() on a free one");
        {
            var hold = new TcpListener(IPAddress.Loopback, busy);
            hold.Start();
            B.Start(busy);
            B.IsClientConnected();
            B.Start(port5);
            var F = new Client("F");
            if (F.Connect(port5)) { F.Send("{\"action\":\"ping\"}\n"); F.Drain(400); F.Forget(); }
            hold.Stop();
            B.Stop();
        }
        int port6 = Program.FreePort();
        B.PortNames[port6] = "<PORT6>"; LogSink.Normalizers.Add((port6.ToString(), "<PORT6>"));
        T.Scenario("19_hostile_clients", "a browser-style HTTP request and an endless unterminated line are dropped; the server keeps serving");
        {
            B.Start(port6);
            var H = new Client("H");
            if (H.Connect(port6))
            {
                WaitConn(true);
                string body = "{\"action\":\"restore_snapshot\",\"snapshot_id\":\"1760000000_0001\"}\n";
                H.Send("POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\n" +
                       "Content-Type: text/plain\r\nContent-Length: " + body.Length + "\r\n\r\n" + body,
                       "cross-site HTTP POST carrying a rewind verb");
                H.ExpectEof();
                WaitConn(false);
            }
            var J = new Client("J");
            if (J.Connect(port6))
            {
                WaitConn(true);
                J.Send("{\"action\":\"ping\"}\n");
                J.Expect(1);
                string chunk = new string('x', 1024 * 1024);
                for (int i = 0; i < 9; i++)
                    J.SendBytes(Encoding.UTF8.GetBytes(i == 0 ? "{\"action\":\"play\",\"pad\":\"" + chunk : chunk), $"1 MiB chunk {i + 1}/9, no newline");
                J.ExpectEof(15000, resetIsClose: true);
                WaitConn(false);
            }
            var L = new Client("L");
            if (L.Connect(port6))
            {
                WaitConn(true);
                L.Send("{\"action\":\"ping\"}\n");
                L.Expect(1);
                L.Close();
                WaitConn(false);
            }
            Hooks.Drain();
            B.Stop();
        }
        int port7 = Program.FreePort();
        B.PortNames[port7] = "<PORT7>"; LogSink.Normalizers.Add((port7.ToString(), "<PORT7>"));
        T.Scenario("20_coach_lane", "inbound coach messages raise CoachReceived and never touch the pending autopilot slot");
        {
            B.Start(port7);
            var M = new Client("M");
            if (M.Connect(port7))
            {
                WaitConn(true);
                string turnRec = "{\"kind\":\"turn\",\"id\":\"turn:n4\",\"node_id\":\"n4\",\"turn\":1,\"floor\":2,\"verdict\":\"blunder\",\"you\":[\"end turn\"],\"ai\":[\"Bash → Cultist\",\"end turn\"],\"hp_drop\":9.0,\"win_before\":0.97,\"win_after\":0.9,\"peeked\":false,\"ungraded_actions\":0}";
                // an uncorrelated autopilot wait accepts ANY line: coach must not reach it
                var pw = B.WaitForAction("p20w", Long);
                M.Send($"{{\"action\":\"coach\",\"request_id\":null,\"state_id\":null,\"coach\":{turnRec}}}\n", "coach while an uncorrelated wait is pending");
                Barrier(M);
                pw.Check("after_coach");
                M.Send("{\"action\":\"end_turn\"}\n", "the wait's own reply");
                pw.Await("after_reply");
                // correlated autopilot request: coach is not its reply either
                var p = B.SendStateAndWait("p20", "{\"type\":\"combat_action\"}", Long);
                string r = Rid(M.Expect(1).FirstOrDefault())!;
                string recapRec = "{\"kind\":\"recap\",\"id\":\"recap:n2\",\"node_id\":\"n2\",\"log_name\":\"run.jsonl\",\"log_idx\":3,\"floor\":2,\"room_type\":\"Monster\",\"hp_start\":80,\"hp_end\":61,\"hp_lost\":19,\"ai_hp_lost\":8.5,\"turns\":4,\"turns_graded\":3,\"turns_ungraded\":1,\"unfinished_turns\":1,\"costliest\":[" + turnRec + "]}";
                M.Send($"{{\"action\":\"coach\",\"request_id\":null,\"state_id\":null,\"coach\":{recapRec}}}\n", "recap record");
                M.Send("{\"action\":\"coach\",\"request_id\":null,\"state_id\":null,\"coach\":{\"kind\":\"autopsy\",\"id\":\"autopsy:ABC\",\"log_name\":\"run.jsonl\",\"floor\":9,\"act\":1,\"items\":[{\"kind\":\"turn\",\"record\":" + turnRec + "}],\"turns_ungraded\":2,\"decisions_graded\":14}}\n", "autopsy record");
                M.Send("{\"action\":\"COACH\",\"request_id\":null,\"state_id\":null,\"coach\":{\"kind\":\"withdraw\",\"ids\":[\"turn:n4\"]}}\n", "withdraw, verb case-insensitive");
                M.Send("{\"action\":\"coach\",\"request_id\":null,\"coach\":\"not an object\"}\n", "coach field not an object: dropped");
                M.Send("{\"action\":\"coach\"}\n", "no coach field: dropped");
                Barrier(M);
                p.Check("after_coach_lines");
                M.Send(Choose(r));
                p.Await("after_reply");
                M.Close();
                WaitConn(false);
            }
            Hooks.Drain();
            B.Stop();
        }
        T.Scenario("99_end", "done");
    }
}
