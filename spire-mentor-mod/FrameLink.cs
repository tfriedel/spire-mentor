// FrameLink.cs -- the mod's side of the Spire Mentor frame (spire-mentor-frame/).
//
// The frame is a desktop window that embeds the game window and lays the
// mod's free-floating panels (toolbar, advice, coach chips / recap, action
// log) out AROUND it, as web views. It launches the game with
// `--spire-mentor-frame-port N`; when that flag is present the mod is
// "framed": it connects to 127.0.0.1:N and
//   - pushes what those panels would have shown (newline-delimited JSON:
//     `ui`, `advice`, `chip`, `coach_panel`, `log`, ...), and
//   - takes the panels' button presses back (`{"type":"action","name":...}`),
//     dispatched to the same SimActions verbs the hotkeys and toolbar use.
// The panels themselves are then hidden in-game. What is ANCHORED to a game
// element (hand-card scores, map markers / route, enemy #k tags) and the
// modal dialogs (settings, bug report, help) stay in-game.
//
// Its own socket, not BridgeServer's: that one has exactly one client (the
// advisor) and carries the decision protocol; this carries UI state only.
//
// Never throws into the game: every public entry point swallows and logs.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace SpireMentor;

internal static class FrameLink
{
    /// <summary>Protocol version of the frame channel (FRAME-PROTOCOL in
    /// spire-mentor-frame/README.md).</summary>
    public const int ProtocolVersion = 1;

    private const int LogBacklog = 300;

    // The frame's relay port: from the command line (the frame launched the
    // game), or set at runtime when the mod started the frame next to the
    // running game (Attach, SimFrameLauncher). 0 = not framed.
    private static volatile int _port = SimMp.FramePort ?? 0;
    // Cancels the current link's connect / read / write loops (Detach).
    private static CancellationTokenSource _linkCts = new();
    private static readonly ConcurrentQueue<string> _outbox = new();
    private static readonly SemaphoreSlim _outboxSignal = new(0);
    private static readonly object _lock = new();
    private static readonly List<object> _logBacklog = new();
    // Each chip with its kind and when it was shown, so a welcome replays
    // only the chips still up, with their age (the frame's timers continue
    // rather than restart).
    private static readonly Dictionary<string, (object Chip, string Kind, DateTime ShownUtc)> _chips = new();
    private static string? _lastUiJson;
    private static string? _lastAdviceJson;
    private static object? _coachPanel;
    private static volatile bool _connected;
    private static bool _started;

    /// <summary>True while the game sits in the frame (launched by it, or
    /// adopted by the frame the mod started): the mod feeds the frame's
    /// panels (in both UI modes, so a switch shows current state).</summary>
    public static bool Framed => _port != 0;

    /// <summary>The frame link is up (the frame's relay accepted us).</summary>
    public static bool Connected => _connected;

    private static volatile bool _uiOutside = true;

    /// <summary>Framed AND the player wants the panels around the game, not
    /// over it: the in-game toolbar, advice panel, chips, recap and log hide.
    /// The frame's Panels menu and the in-game toolbar switch it
    /// (<see cref="SetUiOutside"/>); the frame's page remembers the choice.</summary>
    public static bool UiOutside => Framed && _uiOutside;

    /// <summary>Move the floating panels out of the game (true) or back in.
    /// Any thread; the surfaces re-render on the main thread.</summary>
    public static void SetUiOutside(bool outside)
    {
        if (!Framed || _uiOutside == outside)
        {
            return;
        }
        _uiOutside = outside;
        Logger.Log($"[FrameLink] panels {(outside ? "outside the game (frame)" : "inside the game")}");
        RefreshSurfaces();
    }

    /// <summary>Re-render the in-game panels for the current UI mode (main
    /// thread, deferred).</summary>
    private static void RefreshSurfaces()
    {
        Callable.From(() =>
        {
            Guard(SimToolbar.Instance.OnFrameUiModeChanged);
            Guard(SimHud.Instance.OnFrameUiModeChanged);
            Guard(SimCoachChip.Instance.OnFrameUiModeChanged);
            Guard(SimCoachPanel.Instance.OnFrameUiModeChanged);
            Guard(SimActionLog.OnFrameUiModeChanged);
        }).CallDeferred();
    }

    public static void ToggleUiOutside() => SetUiOutside(!_uiOutside);

    /// <summary>The frame launched the game (<c>--spire-mentor-frame-port</c>):
    /// connect to it. A no-op otherwise.</summary>
    public static void Start()
    {
        if (!Framed || _started)
        {
            return;
        }
        _started = true;
        Logger.Log($"[FrameLink] framed mode: connecting to the frame on 127.0.0.1:{_port}");
        CancellationToken ct = _linkCts.Token;
        Task.Run(() => RunAsync(_port, ct));
    }

    /// <summary>The mod started the frame next to the running game
    /// (SimFrameLauncher): connect to its relay on <paramref name="port"/>
    /// and move the panels out. False when already framed.</summary>
    public static bool Attach(int port)
    {
        if (Framed || port <= 0)
        {
            return false;
        }
        _linkCts = new CancellationTokenSource();
        _uiOutside = true;
        _port = port;
        _started = true;
        Logger.Log($"[FrameLink] frame started by the mod: connecting on 127.0.0.1:{port}");
        CancellationToken ct = _linkCts.Token;
        Task.Run(() => RunAsync(port, ct));
        RefreshSurfaces();
        return true;
    }

    /// <summary>Ask a frame the mod started to give the game its window
    /// back and close (it ignores this when it launched the game). False
    /// when there is no link to send it on.</summary>
    public static bool RequestClose()
    {
        if (!Framed || !_connected)
        {
            return false;
        }
        Enqueue(Serialize(new { type = "close_frame" }));
        return true;
    }

    /// <summary>The frame is gone (its window closed, the game got its own
    /// window back): stop the link and draw the panels in-game again.</summary>
    public static void Detach()
    {
        if (!Framed)
        {
            return;
        }
        _port = 0;
        _started = false;
        _connected = false;
        _linkCts.Cancel();
        while (_outbox.TryDequeue(out _))
        {
        }
        Logger.Log("[FrameLink] frame closed: panels back in the game");
        RefreshSurfaces();
    }

    // ------------------------------------------------------------------
    // Publishers (any thread)
    // ------------------------------------------------------------------

    /// <summary>Toolbar state; sent only when it changed.</summary>
    public static void PublishUi(object ui)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            string json = Serialize(new { type = "ui", ui });
            lock (_lock)
            {
                if (json == _lastUiJson)
                {
                    return;
                }
                _lastUiJson = json;
            }
            Enqueue(json);
        });
    }

    /// <summary>The advice panel's content, or null to clear it. A peek line
    /// (Coach style, F5) rides along as <paramref name="peek"/>.</summary>
    public static void PublishAdvice(SimAdvice? advice, string? peek = null)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            string json = Serialize(new { type = "advice", advice = advice == null ? null : AdviceDto(advice), peek });
            lock (_lock)
            {
                if (json == _lastAdviceJson)
                {
                    return;
                }
                _lastAdviceJson = json;
            }
            Enqueue(json);
        });
    }

    public static void Chip(ChipText text, bool inCombat)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            var chip = new
            {
                id = text.Id,
                kind = text.Kind,
                title = text.Title,
                lines = text.Lines,
                tone = text.Tone.ToString().ToLowerInvariant(),
                log_name = text.LogName,
                log_idx = text.LogIdx,
                in_combat = inCombat,
            };
            lock (_lock)
            {
                _chips[text.Id] = (chip, text.Kind, DateTime.UtcNow);
            }
            Enqueue(Serialize(new { type = "chip", chip }));
        });
    }

    public static void ChipWithdraw(IReadOnlyCollection<string> ids)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            lock (_lock)
            {
                foreach (string id in ids)
                {
                    _chips.Remove(id);
                }
            }
            Enqueue(Serialize(new { type = "chip_withdraw", ids }));
        });
    }

    /// <summary>The board moved on: the frame drops every chip the player
    /// has not pinned (pinning is the frame's own state).</summary>
    public static void ChipClearUnpinned()
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            lock (_lock)
            {
                _chips.Clear();
            }
            Enqueue(Serialize(new { type = "chip_clear_unpinned" }));
        });
    }

    public static void CoachPanel(PanelText text)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            var panel = new
            {
                id = text.Id,
                kind = text.Kind,
                title = text.Title,
                summary = text.Summary,
                sections = text.Sections.Select(s => new { heading = s.Heading, lines = s.Lines, log_idx = s.LogIdx }).ToList(),
                footer = text.Footer,
                log_name = text.LogName,
            };
            lock (_lock)
            {
                _coachPanel = panel;
            }
            Enqueue(Serialize(new { type = "coach_panel", panel }));
        });
    }

    /// <summary>Hide the recap / autopsy; null ids = whatever is showing.</summary>
    public static void CoachPanelHide(IReadOnlyCollection<string>? ids)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            lock (_lock)
            {
                _coachPanel = null;
            }
            Enqueue(Serialize(new { type = "coach_panel_hide", ids }));
        });
    }

    public static void Log(SimActionLogEntry e)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() =>
        {
            var entry = new
            {
                ts = e.TimestampUtc.ToString("o"),
                actor = e.Actor,
                act = e.Act,
                floor = e.Floor,
                decision = e.DecisionType,
                action = e.Action,
                advice = e.Advice,
                // A combat event as the workbench's combat-log display row (its
                // LogLine renders it). `side` is whose turn it was; the frame
                // fills in the round from where the row sits in the log.
                row = e.Event == null ? null : new
                {
                    kind = e.Event.Kind,
                    verb = e.Event.Verb,
                    actor = e.Event.Actor,
                    actor_side = e.Event.ActorSide,
                    target = e.Event.Target,
                    target_side = e.Event.TargetSide,
                    amount = e.Event.Amount,
                    noun = e.Event.Noun,
                    text = e.Event.Text,
                    killed = e.Event.Killed,
                    side = e.Event.Side,
                },
            };
            lock (_lock)
            {
                _logBacklog.Add(entry);
                if (_logBacklog.Count > LogBacklog + 50)
                {
                    _logBacklog.RemoveRange(0, _logBacklog.Count - LogBacklog);
                }
            }
            Enqueue(Serialize(new { type = "log", entry }));
        });
    }

    /// <summary>Ask the frame to show / focus one of its panels (Ctrl+L
    /// toggles the log there instead of in-game).</summary>
    public static void TogglePanel(string panel)
    {
        if (!Framed)
        {
            return;
        }
        Guard(() => Enqueue(Serialize(new { type = "toggle_panel", panel })));
    }

    // ------------------------------------------------------------------
    // Inbound commands
    // ------------------------------------------------------------------

    private static readonly Dictionary<string, Action> Actions = new()
    {
        ["toggle_advisor"] = SimActions.ToggleAdvisor,
        ["switch_style"] = SimActions.SwitchStyle,
        ["peek"] = SimActions.Peek,
        ["ai_once"] = () => SimActions.AiOnce("The frame's AI-once button"),
        ["ai_auto"] = () => SimActions.ToggleAutoplay("The frame's AI-auto button"),
        ["autobattle"] = () => SimActions.ToggleAutobattle("The frame's Autobattle button"),
        ["toggle_brain"] = () => SimActions.ToggleBrain("The frame's brain button"),
        ["partner_go"] = () => SimActions.PartnersGo("The frame's Partner go button"),
        ["speed_down"] = () => SimActions.StepSpeed(-1),
        ["speed_up"] = () => SimActions.StepSpeed(1),
        ["undo_turn"] = SimActions.UndoTurn,
        ["restart_room"] = SimActions.RestartRoom,
        ["back_one_room"] = SimActions.BackOneRoom,
        ["hide_all"] = SimActions.ToggleHideAll,
        ["toggle_hud"] = SimActions.ToggleHud,
        ["help"] = SimActions.ToggleHelp,
        ["settings"] = () => SimSettingsPanel.Instance.Toggle(),
        ["report_bug"] = SimActions.ReportBug,
        ["open_workbench"] = () => SimWorkbenchLink.Open(),
        ["copy_state"] = SimActions.CopyState,
        ["toggle_ui_outside"] = ToggleUiOutside,
    };

    /// <summary>The action names the frame may send (for tests / docs).</summary>
    public static IReadOnlyCollection<string> ActionNames => Actions.Keys;

    private static void HandleLine(string line)
    {
        using JsonDocument doc = JsonDocument.Parse(line);
        JsonElement root = doc.RootElement;
        string? type = root.TryGetProperty("type", out JsonElement t) ? t.GetString() : null;
        switch (type)
        {
            case "action":
            {
                string? name = root.TryGetProperty("name", out JsonElement n) ? n.GetString() : null;
                if (name != null && Actions.TryGetValue(name, out Action? action))
                {
                    Callable.From(() => Guard(action)).CallDeferred();
                }
                else
                {
                    Logger.Log($"[FrameLink] unknown action '{name}'");
                }
                break;
            }
            case "open_workbench_at":
            {
                string? run = root.TryGetProperty("log_name", out JsonElement r) ? r.GetString() : null;
                int? step = root.TryGetProperty("log_idx", out JsonElement s) && s.ValueKind == JsonValueKind.Number
                    ? s.GetInt32() : null;
                Callable.From(() => Guard(() => SimWorkbenchLink.Open(run, step))).CallDeferred();
                break;
            }
            case "ping":
                Enqueue(Serialize(new { type = "pong" }));
                break;
            case "set_ui_outside":
                if (root.TryGetProperty("value", out JsonElement v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False)
                {
                    SetUiOutside(v.GetBoolean());
                }
                break;
            case "resync":
                // The frame's page (re)loaded: it has nothing; send it all.
                SendWelcome();
                break;
            default:
                Logger.Log($"[FrameLink] ignoring message type '{type}'");
                break;
        }
    }

    // ------------------------------------------------------------------
    // Transport
    // ------------------------------------------------------------------

    private static async Task RunAsync(int port, CancellationToken linkCt)
    {
        while (!linkCt.IsCancellationRequested)
        {
            try
            {
                using var client = new TcpClient { NoDelay = true };
                await client.ConnectAsync("127.0.0.1", port, linkCt).ConfigureAwait(false);
                Logger.Log("[FrameLink] connected to the frame");
                using NetworkStream stream = client.GetStream();
                _connected = true;
                SendWelcome();
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(linkCt);
                Task reader = ReadLoopAsync(stream, cts.Token);
                Task writer = WriteLoopAsync(stream, cts.Token);
                await Task.WhenAny(reader, writer).ConfigureAwait(false);
                cts.Cancel();
                Logger.Log("[FrameLink] frame disconnected; retrying");
            }
            catch (Exception ex) when (ex is SocketException or IOException or ObjectDisposedException or OperationCanceledException)
            {
                // The frame is not listening yet (or went away, or Detach):
                // retry quietly while still framed.
            }
            catch (Exception ex)
            {
                Logger.Log($"[FrameLink] link error: {ex.Message}");
            }
            if (_port == port)
            {
                _connected = false;
            }
            try
            {
                await Task.Delay(1000, linkCt).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return;
            }
        }
    }

    /// <summary>A (re)connected frame gets the current picture, not just
    /// the deltas from now on.</summary>
    private static void SendWelcome()
    {
        // Drop anything queued while disconnected: the welcome is the state.
        while (_outbox.TryDequeue(out _))
        {
        }
        List<string> lines = new()
        {
            Serialize(new { type = "hello", protocol = ProtocolVersion, actions = ActionNames }),
        };
        lock (_lock)
        {
            if (_lastUiJson != null)
            {
                lines.Add(_lastUiJson);
            }
            if (_lastAdviceJson != null)
            {
                lines.Add(_lastAdviceJson);
            }
            lines.Add(Serialize(new { type = "log_backlog", entries = _logBacklog.ToList() }));
            DateTime now = DateTime.UtcNow;
            foreach ((string id, (object chip, string kind, DateTime shown)) in _chips.ToList())
            {
                double ageMs = (now - shown).TotalMilliseconds;
                if (ageMs >= SimCoachChip.ChipSeconds(kind) * 1000)
                {
                    // Expired on its own timer: the frame would only show it
                    // again. A pinned one is the frame's: it kept it.
                    _chips.Remove(id);
                    continue;
                }
                lines.Add(Serialize(new { type = "chip", chip, age_ms = (long)ageMs }));
            }
            if (_coachPanel != null)
            {
                lines.Add(Serialize(new { type = "coach_panel", panel = _coachPanel }));
            }
        }
        foreach (string l in lines)
        {
            _outbox.Enqueue(l);
            _outboxSignal.Release();
        }
    }

    private static async Task ReadLoopAsync(NetworkStream stream, CancellationToken ct)
    {
        using var reader = new StreamReader(stream, new UTF8Encoding(false), false, 8192, leaveOpen: true);
        while (!ct.IsCancellationRequested)
        {
            string? line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
            if (line == null)
            {
                return;
            }
            if (line.Length == 0)
            {
                continue;
            }
            try
            {
                HandleLine(line);
            }
            catch (Exception ex)
            {
                Logger.Log($"[FrameLink] bad message: {ex.Message}");
            }
        }
    }

    private static async Task WriteLoopAsync(NetworkStream stream, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            await _outboxSignal.WaitAsync(ct).ConfigureAwait(false);
            if (!_outbox.TryDequeue(out string? line))
            {
                continue;
            }
            byte[] bytes = Encoding.UTF8.GetBytes(line + "\n");
            await stream.WriteAsync(bytes, ct).ConfigureAwait(false);
        }
    }

    private static void Enqueue(string json)
    {
        if (!_connected)
        {
            return; // the welcome carries the current state on (re)connect
        }
        _outbox.Enqueue(json);
        _outboxSignal.Release();
    }

    // ------------------------------------------------------------------
    // DTOs
    // ------------------------------------------------------------------

    private static object AdviceDto(SimAdvice a) => new
    {
        note = a.Note,
        p_win = a.PWin,
        best_q = a.BestQ,
        low_confidence = a.LowConfidence,
        best = a.Best?.Label,
        pv = a.Pv,
        pick_reason = a.PickReason,
        pick_gap_hp = a.PickGapHp,
        hand_back = a.HandBack,
        // Which brain answered (PROTOCOL.md section 5, v5.2), and the tag the
        // in-game panel shows for it ("Instinct" / "Search (Instinct: ...)").
        brain = a.Brain,
        instinct_unavailable = a.InstinctUnavailable,
        source_tag = AdvisorBrainRules.SourceTag(a.Brain, a.InstinctUnavailable),
        options = a.Options.Select(o => new
        {
            label = o.Label,
            p_win = o.PWin,
            q = o.Q,
            visits = o.Visits,
            visit_share = o.VisitShare,
            wins = o.Wins,
            losses = o.Losses,
            ties = o.Ties,
            is_best = o.IsBest,
            low_confidence = o.LowConfidence,
            in_band = o.InBand,
            pick_rank = o.PickRank,
            card_index = o.CardIndex,
            score = o.Score,
        }).ToList(),
        path = a.Path.Select(p => new { col = p.Col, row = p.Row, type = p.Type, hop = p.Hop, hp = p.Hp }).ToList(),
    };

    private static string Serialize(object value) => JsonSerializer.Serialize(value);

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[FrameLink] ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}
