// SimActionLog.cs -- in-game action log (phase 4).
//
// A thread-safe ring buffer of decision entries plus a scrollable in-game
// panel, modeled on BaseLib's NLogWindow (BaseLibScenes/NLogWindow.cs):
// same ring-buffer discipline (hard cap + chunked front-trim so per-append
// cost stays ~O(1)), same "off-thread writers only flip a dirty bit, the
// main thread renders on _Process at <=30fps" rule, same follow-tail
// behavior (auto-scroll unless the user scrolled up), same filter LineEdit.
// Rebuilt in code as a panel rather than an OS Window, because NLogWindow's
// .tscn lives in BaseLib's pack and an embedded panel also works on the
// Steam Deck (where BaseLib disables its OS window). The panel is a plain
// C# class COMPOSED of built-in Godot nodes (CanvasLayer/Timer/...), NOT a
// Node subclass: this project compiles with plain Microsoft.NET.Sdk and has
// no Godot source generators, so subclassed virtuals like _Process are
// never dispatched (same reason MainFile dropped `: Node`). The refresh
// loop that NLogWindow runs in _Process is driven by a built-in Timer
// node's Timeout signal instead.
//
// Entries record who acted (You / AI), the decision context (floor/act,
// decision type), the action taken, and the advice that was on screen at
// the time -- the raw material for phase 9's move review.
//
// Writers: SimProtocolHelpers.TryParse (autopilot actions, today) and the
// phase-6 advisor loop (human actions + advice). Record() is safe from any
// thread and never throws.

using System;
using System.Collections.Generic;
using System.Text;
using Godot;

namespace SpireMentor;

internal sealed class SimActionLogEntry
{
    public DateTime TimestampUtc = DateTime.UtcNow;
    public string Actor = "AI"; // "AI" | "You"
    public int Floor;
    public int Act;
    public string DecisionType = "";
    public string Action = "";
    public string? Advice;

    /// <summary>A combat-history line's structure (CombatEventText), for the
    /// frame's log; null on every other entry.</summary>
    public CombatEvent? Event;

    // The panel renders with BBCode enabled, so every line carries its own
    // colours and every piece of GAME text has to be escaped -- a card called
    // "[Redacted]" would otherwise be silently swallowed as an unknown tag.
    private const string Dim = "#5b6070";
    private const string AiGold = "#ffd24d";
    private const string YouWhite = "#eceff4";
    private const string ModCyan = "#7fd4e8";
    private const string BodyGrey = "#c8ccd4";

    private static string Esc(string? text) => (text ?? string.Empty).Replace("[", "[lb]");

    private static string Tint(string hex, string text) => $"[color={hex}]{text}[/color]";

    public string Render()
    {
        var sb = new StringBuilder(160);

        // Floor markers are the log's spine, not an event: render them as a
        // heading rather than as one more indistinguishable grey row.
        if (Action.StartsWith("════", StringComparison.Ordinal))
        {
            return Tint(AiGold, $"[b]{Esc(Action)}[/b]");
        }

        sb.Append(Tint(Dim, TimestampUtc.ToString("HH:mm:ss"))).Append(' ');
        sb.Append(Actor switch
        {
            "AI" => Tint(AiGold, "AI "),
            "mod" => Tint(ModCyan, "mod"),
            "game" => Tint(Dim, " · "), // world events: no actor tag
            _ => Tint(YouWhite, "YOU"),
        });
        sb.Append(' ');
        if (Floor > 0 || Act > 0)
        {
            sb.Append(Tint(Dim, $"A{Act}/F{Floor}")).Append(' ');
        }
        if (!string.IsNullOrEmpty(DecisionType))
        {
            sb.Append(Tint(Dim, Esc(DecisionType) + ":")).Append(' ');
        }
        sb.Append(Tint(Actor == "game" ? Dim : BodyGrey, Esc(Action)));
        if (!string.IsNullOrEmpty(Advice))
        {
            sb.Append(Tint(Dim, $"   [i]« {Esc(Advice)} »[/i]"));
        }
        return sb.ToString();
    }
}

internal static class SimActionLog
{
    // Same constants as NLogWindow: cap far above the display size, trim in
    // chunks so removal cost amortizes.
    private const int MaxBufferedLines = 8192;
    private const int TrimChunk = 1024;
    public const int MaxDisplayedLines = 400;

    private static readonly object _lock = new();
    private static readonly List<string> _lines = new();
    private static bool _dirty;

    private static SimLogPanel? _panel;

    /// <summary>Panel size/collapse, remembered across Ctrl+L toggles.</summary>
    internal static Vector2? PanelSize;
    internal static bool PanelCollapsed;

    /// <summary>
    /// Where the player dragged the panel, or null while it follows the
    /// advice column (LogPanelPlacement). Persisted in the mod config, so it
    /// also survives a relaunch.
    /// </summary>
    internal static Vector2? DraggedPosition
    {
        get => LogPanelPlacement.TryParse(SpireMentorModConfig.LogPanelPosition, out float x, out float y)
            ? new Vector2(x, y)
            : null;
        set
        {
            string text = value is { } p ? LogPanelPlacement.Format(p.X, p.Y) : "";
            if (text != SpireMentorModConfig.LogPanelPosition)
            {
                SpireMentorModConfig.LogPanelPosition = text;
                SpireMentorModConfig.SaveSoon();
            }
        }
    }

    /// <summary>Append an entry. Safe from any thread; never throws.</summary>
    public static void Record(SimActionLogEntry entry)
    {
        FrameLink.Log(entry);
        try
        {
            string line = entry.Render();
            lock (_lock)
            {
                _lines.Add(line);
                if (_lines.Count > MaxBufferedLines + TrimChunk)
                {
                    _lines.RemoveRange(0, _lines.Count - MaxBufferedLines);
                }
                _dirty = true;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimActionLog] Record failed: {ex.Message}");
        }
    }

    /// <summary>Append a plain mod-status line (mode switches, advisor
    /// process events) rather than a game decision.</summary>
    public static void Note(string message)
    {
        Record(new SimActionLogEntry { Actor = "mod", DecisionType = "", Action = message });
    }

    /// <summary>Snapshot of lines matching a filter (newest last).</summary>
    public static List<string> Snapshot(string? filter)
    {
        lock (_lock)
        {
            _dirty = false;
            if (string.IsNullOrEmpty(filter))
            {
                return new List<string>(_lines);
            }
            List<string> result = new();
            foreach (string line in _lines)
            {
                if (line.Contains(filter, StringComparison.OrdinalIgnoreCase))
                {
                    result.Add(line);
                }
            }
            return result;
        }
    }

    public static bool ConsumeDirty()
    {
        lock (_lock)
        {
            bool was = _dirty;
            _dirty = false;
            return was;
        }
    }

    /// <summary>The panels moved out of the game (FrameLink): close the
    /// in-game log (the frame has its own). Main thread.</summary>
    public static void OnFrameUiModeChanged()
    {
        if (FrameLink.UiOutside && _panel != null && _panel.IsAlive)
        {
            _panel.Close();
            _panel = null;
        }
    }

    /// <summary>Toggle the in-game log panel (Ctrl+L / console). Main thread.</summary>
    public static void TogglePanel()
    {
        // Panels outside the game: the log is a frame panel; Ctrl+L shows /
        // hides that one.
        if (FrameLink.UiOutside)
        {
            FrameLink.TogglePanel("log");
            return;
        }
        try
        {
            if (_panel != null && _panel.IsAlive)
            {
                _panel.Close();
                _panel = null;
                return;
            }
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return;
            }
            _panel = new SimLogPanel(tree);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimActionLog] TogglePanel failed: {ex.Message}");
        }
    }
}

/// <summary>
/// The scrollable log panel. A plain class composed of built-in nodes (NOT
/// a Node subclass -- no source generators in this project, see file
/// header); the refresh loop runs off a built-in Timer node's Timeout
/// signal, connected via Callable.From.
/// </summary>
internal sealed class SimLogPanel
{
    private const int RefreshMs = 100; // NLogWindow refreshes at <=30fps; 10fps is plenty for a log

    private readonly CanvasLayer _layer;
    private readonly PanelContainer _panel;
    private readonly RichTextLabel _text;
    private readonly ScrollContainer _scroll;
    private readonly LineEdit _filter;
    private readonly Label _title;
    private bool _following = true;
    private bool _filterChanged;

    // Header drag state. A press that ends without moving more than
    // DragThreshold is a click (collapse/expand); anything further is a move.
    private const float DragThreshold = 4f;
    private bool _pressed;
    private bool _dragging;
    private Vector2 _grabOffset;
    private Vector2 _pressPosition;
    private bool _collapsed;
    private Vector2 _expandedSize;

    public bool IsAlive => GodotObject.IsInstanceValid(_layer);

    public SimLogPanel(SceneTree tree)
    {
        _layer = new CanvasLayer
        {
            Name = "SpireMentorLogLayer",
            Layer = 100, // above SimHud's layer 99
        };

        // Free-floating rect (anchors left at 0) rather than anchored to the
        // viewport: the player drags this thing around, so its position has
        // to be a plain offset we can write.
        // Narrower, shorter, and parked on the LEFT. The old default was 44% x
        // 62% of the screen starting at x=55%, i.e. squarely over the enemies
        // (reported 2026-08-06) -- and being over them mattered because the
        // panel also ate the clicks, see the MouseFilter notes below.
        // Undragged, it sits under the advice column and follows it
        // (LogPanelPlacement, FollowColumn); a fixed 30% down the left edge
        // put the coach chips on top of it.
        Vector2 viewport = tree.Root.GetVisibleRect().Size;
        _expandedSize = SimActionLog.PanelSize
            ?? new Vector2(viewport.X * 0.30f, viewport.Y * 0.42f);
        Vector2 position = SimActionLog.DraggedPosition is { } dragged
            // A position saved at another resolution may be off this screen.
            ? new Vector2(Math.Clamp(dragged.X, -_expandedSize.X + 80f, Math.Max(0f, viewport.X - 80f)),
                          Math.Clamp(dragged.Y, 0f, Math.Max(0f, viewport.Y - 30f)))
            : new Vector2(viewport.X * 0.012f, viewport.Y * 0.30f);

        var panel = new PanelContainer
        {
            Name = "SpireMentorLogPanel",
            Position = position,
            Size = _expandedSize,
            // The panel is something you READ while you play. Stopping mouse
            // input over a rectangle this size meant the fight underneath it
            // could not be clicked at all; the header, the filter box and the
            // scrollbar re-arm themselves below, so everything interactive
            // still works and the rest of the rectangle is transparent to the
            // game.
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _panel = panel;
        panel.AddThemeStyleboxOverride("panel", new StyleBoxFlat
        {
            BgColor = new Color(0.04f, 0.04f, 0.06f, 0.72f),
            BorderColor = new Color(1f, 1f, 1f, 0.10f),
            BorderWidthLeft = 1,
            BorderWidthRight = 1,
            BorderWidthTop = 1,
            BorderWidthBottom = 1,
            CornerRadiusTopLeft = 6,
            CornerRadiusTopRight = 6,
            CornerRadiusBottomLeft = 6,
            CornerRadiusBottomRight = 6,
            ContentMarginLeft = 8f,
            ContentMarginRight = 8f,
            ContentMarginTop = 6f,
            ContentMarginBottom = 8f,
        });

        var vbox = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        vbox.AddThemeConstantOverride("separation", 4);

        // The header is the drag handle AND the collapse button, so it must
        // receive mouse input (the rest of the panel stays click-through
        // except the filter box and the scrollbar).
        var header = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Stop };
        header.Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(OnHeaderGuiInput));
        _title = new Label { MouseFilter = Control.MouseFilterEnum.Ignore };
        _title.AddThemeColorOverride("font_color", new Color(0.78f, 0.80f, 0.88f));
        _title.AddThemeFontSizeOverride("font_size", 13);
        _title.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        header.AddChild(_title);
        _filter = new LineEdit
        {
            PlaceholderText = "filter…",
            CustomMinimumSize = new Vector2(130, 0),
        };
        _filter.AddThemeFontSizeOverride("font_size", 12);
        _filter.Connect(LineEdit.SignalName.TextChanged,
            Callable.From<string>(_ => _filterChanged = true));
        header.AddChild(_filter);
        vbox.AddChild(header);

        _scroll = new ScrollContainer
        {
            SizeFlagsVertical = Control.SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
            // Click-through body (see the panel comment). The vertical
            // scrollbar is a separate child control and keeps its own Stop
            // filter, so dragging it still works -- what is given up is
            // wheel-scrolling over the log, which follow-tail makes moot.
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _text = new RichTextLabel
        {
            FitContent = true,
            ScrollActive = false,
            BbcodeEnabled = true,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            SizeFlagsVertical = Control.SizeFlags.ExpandFill,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _text.AddThemeFontSizeOverride("normal_font_size", 12);
        // A log is a column of aligned fields; a proportional font turns it
        // into ragged prose. Falls back silently if none of these are present.
        var mono = new SystemFont
        {
            FontNames = new[] { "Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Courier New" },
        };
        _text.AddThemeFontOverride("normal_font", mono);
        _text.AddThemeFontOverride("bold_font", mono);
        _scroll.AddChild(_text);
        _scroll.GetVScrollBar().Connect(Godot.Range.SignalName.ValueChanged,
            Callable.From<double>(OnScrollChanged));
        vbox.AddChild(_scroll);

        panel.AddChild(vbox);
        _layer.AddChild(panel);

        var timer = new Godot.Timer
        {
            WaitTime = RefreshMs / 1000.0,
            Autostart = true,
            OneShot = false,
        };
        timer.Connect(Godot.Timer.SignalName.Timeout, Callable.From(OnTick));
        _layer.AddChild(timer);

        tree.Root.AddChild(_layer);
        ApplyCollapsed(SimActionLog.PanelCollapsed);
        FollowColumn();
        Rebuild();
    }

    public void Close()
    {
        RememberPlacement();
        if (IsAlive)
        {
            _layer.QueueFree();
        }
    }

    private void RememberPlacement()
    {
        if (!GodotObject.IsInstanceValid(_panel))
        {
            return;
        }
        SimActionLog.PanelSize = _expandedSize;
        SimActionLog.PanelCollapsed = _collapsed;
    }

    // ------------------------------------------------------------------
    // Drag to move / click to collapse
    // ------------------------------------------------------------------

    private void OnHeaderGuiInput(InputEvent ev)
    {
        try
        {
            // Right-click: back under the advice column, following it again.
            if (ev is InputEventMouseButton rb && rb.ButtonIndex == MouseButton.Right && rb.Pressed)
            {
                SimActionLog.DraggedPosition = null;
                FollowColumn();
                return;
            }

            if (ev is InputEventMouseButton mb && mb.ButtonIndex == MouseButton.Left)
            {
                if (mb.Pressed)
                {
                    _pressed = true;
                    _dragging = false;
                    _pressPosition = mb.GlobalPosition;
                    _grabOffset = mb.GlobalPosition - _panel.Position;
                }
                else if (_pressed)
                {
                    _pressed = false;
                    if (_dragging)
                    {
                        // Put here on purpose: it stays, across launches too.
                        SimActionLog.DraggedPosition = _panel.Position;
                    }
                    else
                    {
                        ApplyCollapsed(!_collapsed);
                    }
                    _dragging = false;
                    RememberPlacement();
                }
                return;
            }

            if (ev is InputEventMouseMotion mm && _pressed)
            {
                if (!_dragging &&
                    mm.GlobalPosition.DistanceTo(_pressPosition) < DragThreshold)
                {
                    return;
                }
                _dragging = true;
                _panel.Position = ClampToViewport(mm.GlobalPosition - _grabOffset);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimActionLog] Header input failed: {ex.Message}");
        }
    }

    /// <summary>Keep at least a header's worth of panel on screen so it can
    /// always be grabbed again.</summary>
    private Vector2 ClampToViewport(Vector2 position)
    {
        Rect2 view = _panel.GetViewportRect();
        float maxX = Math.Max(0f, view.Size.X - 80f);
        float maxY = Math.Max(0f, view.Size.Y - 30f);
        return new Vector2(Math.Clamp(position.X, -_panel.Size.X + 80f, maxX),
                           Math.Clamp(position.Y, 0f, maxY));
    }

    private void ApplyCollapsed(bool collapsed)
    {
        _collapsed = collapsed;
        if (!GodotObject.IsInstanceValid(_panel))
        {
            return;
        }
        _scroll.Visible = !collapsed;
        _filter.Visible = !collapsed;
        // Short enough to survive the narrower default panel without being
        // squeezed out by the filter box next to it.
        _title.Text = collapsed ? "▶ action log · Ctrl+L" : "▼ action log · drag · Ctrl+L";
        // Below its minimum the container snaps to the header's height, which
        // is exactly the collapsed look we want.
        _panel.Size = collapsed ? new Vector2(_expandedSize.X, 0f) : _expandedSize;
        SimActionLog.PanelCollapsed = collapsed;
    }

    /// <summary>
    /// Until the player drags it: keep the panel under the advice column
    /// (SimHud.ColumnBottom), shortened when the column leaves too little
    /// room below it (LogPanelPlacement.BelowColumn). Runs every tick, as
    /// the column grows and shrinks with the advice and the chips.
    /// </summary>
    private void FollowColumn()
    {
        try
        {
            if (_pressed || SimActionLog.DraggedPosition != null || !GodotObject.IsInstanceValid(_panel))
            {
                return;
            }
            (float left, float bottom) = SimHud.Instance.ColumnBottom();
            float viewHeight = _panel.GetViewportRect().Size.Y;
            (float top, float height) = LogPanelPlacement.BelowColumn(bottom, _expandedSize.Y, viewHeight);
            var want = new Vector2(left, top);
            if (_panel.Position.DistanceTo(want) > 0.5f)
            {
                _panel.Position = want;
            }
            if (!_collapsed && Math.Abs(_panel.Size.Y - height) > 0.5f)
            {
                _panel.Size = new Vector2(_expandedSize.X, height);
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimActionLog] Follow column failed: {ex.Message}");
        }
    }

    private void OnTick()
    {
        try
        {
            FollowColumn();
            bool dirty = SimActionLog.ConsumeDirty();
            if (!dirty && !_filterChanged)
            {
                return;
            }
            _filterChanged = false;
            Rebuild();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimActionLog] Refresh failed: {ex.Message}");
        }
    }

    private void Rebuild()
    {
        if (!GodotObject.IsInstanceValid(_text))
        {
            return;
        }
        List<string> lines = SimActionLog.Snapshot(_filter.Text);
        int start = Math.Max(0, lines.Count - SimActionLog.MaxDisplayedLines);
        var sb = new StringBuilder();
        for (int i = start; i < lines.Count; i++)
        {
            sb.AppendLine(lines[i]);
        }
        _text.Text = sb.ToString();
        if (_following)
        {
            ScrollToBottomDeferred();
        }
    }

    private async void ScrollToBottomDeferred()
    {
        try
        {
            // Wait one frame so scroll offsets account for the new text.
            SceneTree? tree = Engine.GetMainLoop() as SceneTree;
            if (tree == null)
            {
                return;
            }
            await _layer.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
            if (!GodotObject.IsInstanceValid(_scroll))
            {
                return;
            }
            VScrollBar bar = _scroll.GetVScrollBar();
            _scroll.ScrollVertical = (int)bar.MaxValue;
            _following = true;
        }
        catch
        {
            // panel closing mid-await; ignore
        }
    }

    private void OnScrollChanged(double value)
    {
        if (!GodotObject.IsInstanceValid(_scroll))
        {
            return;
        }
        VScrollBar bar = _scroll.GetVScrollBar();
        _following = bar.MaxValue - bar.Page - value <= 8;
    }
}
