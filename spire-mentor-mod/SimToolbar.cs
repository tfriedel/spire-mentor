// SimToolbar.cs -- the Spire Mentor badge and its popover (the mod's controls).
//
// One compact badge, docked under the RIGHT end of the game's top bar
// (SimTopBarLayout.SafeTopYRight): the top-left belongs to the game -- the
// relic tray, and in co-op the player list under it -- and the old 14-button
// strip there covered both and clipped at narrow windows. The badge says the
// state at a glance: the mode's colour and word (Off / Coach / Hints / AI
// playing / Autobattle / Autopilot), a "thinking" indicator (cycling dots,
// still with "Reduce combat effects"), and the engine-build counter.
//
// Clicking it opens a popover, grouped by job, every entry with its hotkey:
//   Mode           Off · Coach · Hints · AI plays   (MentorModes, CoachStyle.cs)
//   Brain          Search · Instinct                (only where it is offered)
//   This decision  Peek (F5) · Let AI pick (F6)
//   Autoplay       Autobattle (Shift+F7) · AI plays until stopped (F7)
//   Speed          - Nx +  (F9 / F10)
//   More           Rewind ▾ · Panels ▾ · Workbench ▾ (+ extension buttons)
// with a header saying, in the player's words, what the advisor is doing
// (MentorBadge.cs). Contextual controls live next to the badge, not in the
// popover, and only while they apply: "Partner go" (a combat with AI
// partners) and, while the AI plays (F7, autobattle, Autopilot), an amber
// "AI is playing — Stop (F7)" button plus an amber frame round the screen
// edge, so nobody can miss who is playing.
//
// Input-safety contract (see SimUiKit): containers, labels and the AI frame
// are MouseFilter.Ignore; buttons and the grip are Stop with FocusMode.None;
// no native tooltips (hover hints go to an owned label). The popover's own
// panel is Stop while it is open, as SimCoachPanel's body is: it is opaque, so
// a click on its background must not land on a card the player cannot see.
// It opens and closes on CLICKS only (the badge; hover regions over combat eat
// clicks). It also closes on a click outside it or on Esc, read by a 40 ms
// poll of Input while it is open -- never by consuming the event, so that
// click or Esc still reaches the game. The open / closed state is remembered
// within a session; every launch starts closed. CanvasLayer 98, below HUD 99 /
// log 100 / help 101.
//
// Placement: the grip left of the badge drags it anywhere (session-only);
// double-clicking the grip docks it again. The badge's top-RIGHT corner is
// the anchor, so the contextual buttons grow to the left of it. What the
// other panels follow:
//   * BottomY -- docked: the line the LEFT column (advice panel, chips,
//     settings, bug report) parks under, i.e. the game's top-bar clearance on
//     the left; the badge is not over that column (unless the window is so
//     narrow that it is, then it clears the badge too).
//   * UndockedRect -- dragged: the badge, plus the popover while open, which
//     the advice column follows (AdviceColumn).
//   * RightColumnTop -- the line SimCoachPanel (right-aligned) parks under.
//
// Refresh model: SimRuntime.Changed handles mode flips from any surface; a
// 150 ms Godot.Timer handles the state that raises no event (speed, autoplay,
// partners, the build counter, the thinking dots, top-bar drift). No source
// generators -> no _Process (see SpireMentor.csproj header).

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;

namespace SpireMentor;

internal sealed class SimToolbar
{
    public static readonly SimToolbar Instance = new();

    private SimToolbar()
    {
        SimRuntime.Changed += () => RunOnMainThread(Refresh);
    }

    // Mode colours (the help panel's palette): grey off, cyan Coach, green
    // Hints, amber whenever the AI itself is playing.
    private static readonly Color OffColor = new(0.60f, 0.64f, 0.72f);
    private static readonly Color CoachColor = new(0.50f, 0.83f, 0.91f);
    private static readonly Color HintsColor = new(0.55f, 0.85f, 0.45f);
    private static readonly Color AiColor = new(1.00f, 0.62f, 0.36f);
    private static readonly Color BrainColor = new(0.72f, 0.62f, 0.95f);

    /// <summary>Right screen edge to the docked badge.</summary>
    private const float EdgeMargin = 16f;
    /// <summary>Badge row to popover / note.</summary>
    private const float PopGap = 4f;
    private const float PopoverMinWidth = 400f;
    /// <summary>The note / hint under the closed badge wraps at this width.</summary>
    private const float UnderWidth = 320f;
    /// <summary>Right edge of the left column (advice panel 430, settings and
    /// coach panels up to 460, at x 16): a toolbar left of this is over it.</summary>
    private const float LeftColumnRight = 16f + 460f + 8f;

    private CanvasLayer? _layer;
    private HBoxContainer? _row;
    private Button? _badge;
    private Button? _partnerGo;
    private Button? _stop;
    private Panel? _aiFrame;

    private PanelContainer? _popover;
    private VBoxContainer? _popoverBody;
    private Label? _headerTitle;
    private Label? _headerStatus;
    private VBoxContainer? _footer;
    private Label? _headerNote;
    private Label? _footerInfo;
    private Label? _popHint;
    private readonly Dictionary<MentorMode, Button> _modeButtons = new();
    private Label? _brainCaption;
    private HBoxContainer? _brainControls;
    private Button? _brainSearch;
    private Button? _brainInstinct;
    private Button? _peek;
    private Button? _aiOnce;
    private Button? _autobattle;
    private Button? _aiAuto;
    private Label? _speedLabel;
    private HBoxContainer? _moreRow;
    private Button? _exitFullscreen;
    private readonly List<MenuButton> _menus = new();
    private readonly Dictionary<string, Button> _extensions = new();

    private VBoxContainer? _under;
    private Label? _underNote;
    private Label? _underHint;

    private Godot.Timer? _closeWatch;
    private bool _mouseWasDown;
    private bool _escWasDown;

    private bool _expanded;
    private Color? _badgeAccent;
    private readonly Dictionary<Button, Color?> _lit = new();
    private string _hint = "";

    private SimAdvisorStatus _advisorStatus = SimAdvisorStatus.None;
    /// <summary>When the current engine build started, for the badge's counter.</summary>
    private DateTime? _buildStartedUtc;
    private string _lastActionSummary = "";
    private string? _note;
    private int _ticks;
    private int _dotFrame;
    /// <summary>Hides the advisor's millisecond thinks (MentorBadge.cs).</summary>
    private readonly ThinkingIndicator _thinking = new();

    // Grip drag state (same click-vs-drag threshold idiom as SimActionLog's
    // header). _undockedAnchor -- the badge row's top-RIGHT corner -- is null
    // while docked.
    private const float DragThreshold = 4f;
    private bool _gripPressed;
    private bool _dragging;
    private Vector2 _grabOffset;
    private Vector2 _pressPosition;
    private Vector2? _undockedAnchor;

    // ------------------------------------------------------------------
    // Public entry points (any thread)
    // ------------------------------------------------------------------

    /// <summary>Wait for the game to come up, then build on the main thread.
    /// Run via TaskHelper.RunSafely from MainFile.Initialize.</summary>
    public async Task InstallWhenReadyAsync()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        await WaitHelper.Until(
            () => NGame.Instance != null,
            cts.Token, TimeSpan.FromSeconds(120), "NGame not available for SimToolbar");
        RunOnMainThread(() =>
        {
            if (EnsureBuilt())
            {
                Refresh();
            }
        });
    }

    public void SetAdvisorStatus(SimAdvisorStatus status)
    {
        RunOnMainThread(() =>
        {
            // Stamp the start so the badge can count the wait up. A build this
            // long has to prove it is still alive; a static label does not.
            if (status == SimAdvisorStatus.Building && _advisorStatus != SimAdvisorStatus.Building)
            {
                _buildStartedUtc = DateTime.UtcNow;
            }
            else if (status != SimAdvisorStatus.Building)
            {
                _buildStartedUtc = null;
            }
            _advisorStatus = status;
            Refresh();
        });
    }

    public void SetLastAction(string summary)
    {
        RunOnMainThread(() =>
        {
            _lastActionSummary = summary;
            Refresh();
        });
    }

    public void SetNote(string? note)
    {
        RunOnMainThread(() =>
        {
            _note = note;
            Refresh();
        });
    }

    /// <summary>
    /// Extension point for later features: appends a button to the popover's
    /// "More" row, once per id. Safe from any thread.
    /// </summary>
    public void AddExtensionButton(string id, string label, string hint, Action onPressed)
    {
        RunOnMainThread(() =>
        {
            if (!EnsureBuilt() || _moreRow == null || _extensions.ContainsKey(id))
            {
                return;
            }
            Button button = SimUiKit.MakeButton(label, hint, onPressed, SetHint);
            _moreRow.AddChild(button);
            _extensions[id] = button;
        });
    }

    /// <summary>
    /// Docked: the global Y the LEFT column's panels (advice panel, coach
    /// chips, settings, bug report) park below -- they add their own gap.
    /// The badge docks on the right, so that is the game's top-bar clearance
    /// on the left (SafeTopY, less the gap the callers add back), unless the
    /// window is narrow enough that the badge or its popover reaches over the
    /// column, then their bottom too. Null when not built, hidden (panels
    /// outside the game), or dragged off its dock (see <see cref="UndockedRect"/>).
    /// </summary>
    public float? BottomY()
    {
        if (!Valid(_row) || !_row!.IsVisibleInTree() || _undockedAnchor != null)
        {
            return null;
        }
        float bottom = SimTopBarLayout.SafeTopY(_row) - AdviceColumn.Gap;
        Rect2 occupied = OccupiedRect();
        if (occupied.Position.X < LeftColumnRight)
        {
            bottom = Math.Max(bottom, occupied.End.Y);
        }
        return bottom;
    }

    /// <summary>The dragged badge's global rect -- with the popover while it
    /// is open, so nothing parks under that -- and the viewport size, while
    /// the player has dragged it off its dock; else null (docked: BottomY).
    /// SimHud moves the advice panel and the coach chips along with it.</summary>
    public (Rect2 Rect, Vector2 Viewport)? UndockedRect()
    {
        if (_undockedAnchor == null || !Valid(_row) || !_row!.IsVisibleInTree())
        {
            return null;
        }
        return (OccupiedRect(), _row.GetViewportRect().Size);
    }

    /// <summary>The global Y a right-aligned panel (SimCoachPanel) parks
    /// below: under the docked badge, its popover and its note; under the
    /// game's top bar on the right when the badge was dragged away. Null when
    /// not built or hidden.</summary>
    public float? RightColumnTop()
    {
        if (!Valid(_row) || !_row!.IsVisibleInTree())
        {
            return null;
        }
        if (_undockedAnchor != null)
        {
            return SimTopBarLayout.SafeTopYRight(_row);
        }
        float bottom = OccupiedRect().End.Y;
        if (Valid(_under) && _under!.Visible)
        {
            bottom = Math.Max(bottom, _under.GetGlobalRect().End.Y);
        }
        return bottom;
    }

    // ------------------------------------------------------------------
    // Build
    // ------------------------------------------------------------------

    private bool EnsureBuilt()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        try
        {
            if (Engine.GetMainLoop() is not SceneTree tree)
            {
                return false;
            }

            // Closed at every launch: an open popover covered the map legend
            // and event text until the first click elsewhere (live, 2026-11-07).
            _expanded = false;

            var layer = new CanvasLayer
            {
                Name = "SpireMentorToolbarLayer",
                Layer = 98,
                // Panels outside the game: the frame draws the toolbar; this
                // one only ticks (OnFrameUiModeChanged flips it).
                Visible = !FrameLink.UiOutside,
            };

            // The "AI is playing" frame round the screen edge: first, so it
            // draws under everything else here; never takes a click.
            var frameStyle = new StyleBoxFlat
            {
                DrawCenter = false,
                BorderColor = AiColor,
                BorderWidthLeft = 5,
                BorderWidthRight = 5,
                BorderWidthTop = 5,
                BorderWidthBottom = 5,
            };
            var aiFrame = new Panel
            {
                Name = "SpireMentorAiFrame",
                MouseFilter = Control.MouseFilterEnum.Ignore,
                Visible = false,
            };
            aiFrame.AddThemeStyleboxOverride("panel", frameStyle);
            aiFrame.SetAnchorsPreset(Control.LayoutPreset.FullRect);
            layer.AddChild(aiFrame);

            layer.AddChild(BuildRow());
            layer.AddChild(BuildPopover());
            layer.AddChild(BuildUnder());

            var ticker = new Godot.Timer
            {
                Name = "SpireMentorToolbarTicker",
                WaitTime = 0.15,
                Autostart = true,
                OneShot = false,
            };
            ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(OnTick));
            layer.AddChild(ticker);

            // Outside-click / Esc watcher: runs only while the popover is open.
            var closeWatch = new Godot.Timer
            {
                Name = "SpireMentorPopoverWatch",
                WaitTime = 0.04,
                Autostart = false,
                OneShot = false,
            };
            closeWatch.Connect(Godot.Timer.SignalName.Timeout, Callable.From(OnCloseWatch));
            layer.AddChild(closeWatch);

            tree.Root.AddChildSafely(layer);

            _layer = layer;
            _aiFrame = aiFrame;
            _closeWatch = closeWatch;
            Logger.Log("[SimToolbar] Badge created");
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Failed to create the badge: {ex.Message}");
            _layer = null;
            return false;
        }
    }

    /// <summary>[Partner go] [AI is playing — Stop] [Exit fullscreen] [grip] [badge].</summary>
    private HBoxContainer BuildRow()
    {
        var row = new HBoxContainer
        {
            Name = "SpireMentorBadgeRow",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Position = new Vector2(EdgeMargin, SimTopBarLayout.FallbackTop),
        };
        row.AddThemeConstantOverride("separation", 6);

        // Co-op with AI partners, in combat only (RefreshPartnerGo).
        _partnerGo = SimUiKit.MakeButton("Partner go",
            "Your AI partner waits for you (your first play, your end turn, or a setup play " +
            "of yours it wants first): let it play the rest of its turn now (Shift+F6)",
            () => SimActions.PartnersGo("The Partner go button"), SetHint);
        _partnerGo.Visible = false;
        row.AddChild(_partnerGo);

        _stop = SimUiKit.MakeButton("AI is playing — Stop (F7)",
            "The AI is playing: stop it and take the controls back (F7 stops F7 and autobattle, " +
            "Shift+F7 autobattle; F8 takes over from Autopilot)",
            () => SimActions.StopAi("The Stop button"), SetHint);
        SimUiKit.SetLit(_stop, AiColor);
        _stop.Visible = false;
        row.AddChild(_stop);

        // Framed fullscreen (the panels in the game): the way back to the
        // panels around the game, beside the badge like the frame's own
        // Fullscreen button -- in the badge's menu, players went hunting
        // through Panels for it.
        _exitFullscreen = SimUiKit.MakeButton("⛶ Exit fullscreen",
            "Back to the window with the panels around the game: advice, coach and log beside it " +
            "(the game's Fullscreen setting and Alt+Enter switch it too)",
            () => SimFrameLauncher.SetWanted(true, "Exit fullscreen"), SetHint);
        _exitFullscreen.Visible = false;
        row.AddChild(_exitFullscreen);

        // The drag handle: a Stop region of its own, so the badge keeps plain
        // click semantics.
        Label grip = SimUiKit.SmallLabel("⠿", new Color(0.85f, 0.88f, 0.95f), 18);
        grip.MouseFilter = Control.MouseFilterEnum.Stop;
        grip.MouseDefaultCursorShape = Control.CursorShape.Move;
        grip.VerticalAlignment = VerticalAlignment.Center;
        grip.Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(OnGripGuiInput));
        grip.Connect(Control.SignalName.MouseEntered, Callable.From(() =>
            SetHint("Drag to move the badge · double-click to put it back")));
        grip.Connect(Control.SignalName.MouseExited, Callable.From(() => SetHint(null)));
        row.AddChild(grip);

        _badge = SimUiKit.MakeButton(MentorBadge.Name,
            "Spire Mentor: click for the controls (mode, brain, autoplay, speed, rewind, panels)",
            OnBadgePressed, SetHint);
        _badge.AddThemeFontSizeOverride("font_size", 15);
        row.AddChild(_badge);

        _row = row;
        return row;
    }

    private PanelContainer BuildPopover()
    {
        var popover = new PanelContainer
        {
            Name = "SpireMentorPopover",
            // Opaque while open, like SimCoachPanel's body: a click on its
            // background must not reach a card hidden under it.
            MouseFilter = Control.MouseFilterEnum.Stop,
            CustomMinimumSize = new Vector2(PopoverMinWidth, 0f),
            Visible = false,
        };
        StyleBoxFlat style = SimUiKit.PanelStyle();
        style.BgColor = new Color(0.05f, 0.06f, 0.08f, 0.94f);
        style.ContentMarginLeft = 12f;
        style.ContentMarginRight = 12f;
        style.ContentMarginTop = 10f;
        style.ContentMarginBottom = 10f;
        popover.AddThemeStyleboxOverride("panel", style);

        var body = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        body.AddThemeConstantOverride("separation", 8);

        // Header: name + version + what the advisor is doing (one line), then
        // the mode in a sentence, which stays put while the advisor thinks.
        _headerTitle = SimUiKit.SmallLabel(MentorBadge.Name, Colors.White, 16);
        body.AddChild(_headerTitle);
        _headerStatus = WrapLabel(new Color(0.85f, 0.88f, 0.95f), 14);
        body.AddChild(_headerStatus);

        // The lines that come and go -- the note, the last advice, the hover
        // hint -- sit on the far side of the controls from the badge
        // (LayoutFooter), so they never push a button out from under the
        // mouse.
        _footer = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        _footer.AddThemeConstantOverride("separation", 4);
        _headerNote = WrapLabel(new Color(1.0f, 0.86f, 0.55f), 13);
        _headerNote.Visible = false;
        _footer.AddChild(_headerNote);
        _footerInfo = WrapLabel(new Color(0.62f, 0.66f, 0.74f), 13);
        _footerInfo.Visible = false;
        _footer.AddChild(_footerInfo);

        var grid = new GridContainer { Columns = 2, MouseFilter = Control.MouseFilterEnum.Ignore };
        grid.AddThemeConstantOverride("h_separation", 12);
        grid.AddThemeConstantOverride("v_separation", 6);

        // Mode: one switch instead of "AI on/off" + "Coach/Hints".
        HBoxContainer mode = Cell(2);
        foreach (MentorMode m in MentorModes.All)
        {
            MentorMode target = m;
            Button segment = SimUiKit.MakeButton(MentorModes.Label(m, autopilot: false), ModeHint(m),
                () => SimActions.SetMentorMode(target, "The Mode switch"), SetHint);
            _modeButtons[m] = segment;
            mode.AddChild(segment);
        }
        AddSection(grid, "Mode  F8", mode);

        // Brain: only where it is offered today (RefreshBrain).
        _brainControls = Cell(2);
        _brainSearch = SimUiKit.MakeButton("Search",
            "Search thinks ahead: a search over the fight, the run net for the screens (Shift+F8 switches)",
            () => SetBrain(AdvisorBrainRules.Search), SetHint);
        _brainInstinct = SimUiKit.MakeButton("Instinct",
            "Instinct is the end-to-end net, no search: every decision on gut feel (Ironclad on v0.111.0 and the v0.113.0 beta " +
            "for now). Hints, Let AI pick, autoplay and Autobattle all follow it (Shift+F8 switches)",
            () => SetBrain(AdvisorBrainRules.Instinct), SetHint);
        _brainControls.AddChild(_brainSearch);
        _brainControls.AddChild(_brainInstinct);
        _brainCaption = AddSection(grid, "Brain  Shift+F8", _brainControls);
        _brainCaption.Visible = false;
        _brainControls.Visible = false;

        HBoxContainer decision = Cell(4);
        _peek = SimUiKit.MakeButton("Peek  F5",
            "Coach: show the AI's pick for just this decision; it hides again once you act (F5)",
            SimActions.Peek, SetHint);
        _aiOnce = SimUiKit.MakeButton("Let AI pick  F6",
            "Let the AI take just this one decision (F6) -- needs Coach or Hints",
            () => SimActions.AiOnce("The Let AI pick button"), SetHint);
        decision.AddChild(_peek);
        decision.AddChild(_aiOnce);
        AddSection(grid, "This decision", decision);

        HBoxContainer autoplay = Cell(4);
        _autobattle = SimUiKit.MakeButton("Autobattle  Shift+F7",
            "The AI plays the fights, you make every run-level decision -- card rewards, map, shops, " +
            "rest sites, events -- with a note on what changed since your last one (Shift+F7)",
            () => SimActions.ToggleAutobattle("The Autobattle button"), SetHint);
        _aiAuto = SimUiKit.MakeButton("AI plays until stopped  F7",
            "Let the AI keep playing until you stop it (F7) -- needs Coach or Hints",
            () => SimActions.ToggleAutoplay("The AI plays button"), SetHint);
        autoplay.AddChild(_autobattle);
        autoplay.AddChild(_aiAuto);
        AddSection(grid, "Autoplay", autoplay);

        HBoxContainer speed = Cell(4);
        speed.AddChild(SimUiKit.MakeButton("−", "Game speed down: 8x, 6x, 4x, 3x, 2x, 1x (F9)",
            () => SimActions.StepSpeed(-1), SetHint));
        _speedLabel = SimUiKit.SmallLabel("1x", Colors.White, 15);
        _speedLabel.VerticalAlignment = VerticalAlignment.Center;
        _speedLabel.HorizontalAlignment = HorizontalAlignment.Center;
        _speedLabel.CustomMinimumSize = new Vector2(34f, 0f);
        speed.AddChild(_speedLabel);
        speed.AddChild(SimUiKit.MakeButton("+", "Game speed up: 1x, 2x, 3x, 4x, 6x, 8x (F10)",
            () => SimActions.StepSpeed(1), SetHint));
        Label speedKeys = SimUiKit.Caption("F9 / F10");
        speedKeys.VerticalAlignment = VerticalAlignment.Center;
        speed.AddChild(speedKeys);
        AddSection(grid, "Speed", speed);

        // The one-shot verbs, grouped by job into three dropdowns (the
        // workbench's top bar shape). Rewind: the game saves at room entry
        // and post-combat, so those are the room-level targets (SimSnapshots);
        // the turn undo replays the game's own combat tape on top of the
        // room-entry save (SimCombatUndo).
        _moreRow = Cell(4);
        AddMenu(SimUiKit.MakeMenuButton("Rewind",
            "⟲ Turn (Ctrl+Z): undo to the start of this combat turn · ⟲ Room (Ctrl+Shift+Z): restart this room (same seed) · ⟲⟲ Back: one room further back",
            new SimUiKit.MenuEntry[]
            {
                new("⟲ Turn  (Ctrl+Z)", "Undo to the start of the current combat turn (replays the game's combat tape)", SimActions.UndoTurn),
                new("⟲ Room  (Ctrl+Shift+Z)", "Rewind to the start of the current room (same seed, exact state)", SimActions.RestartRoom),
                new("⟲⟲ Back one room", "Rewind one room further back", SimActions.BackOneRoom),
            }, SetHint));
        AddMenu(SimUiKit.MakeMenuButton("Panels",
            "Hide all (Ctrl+H) · Log (Ctrl+L) · HUD (F4) · Help (F1) · Settings (works mid-run) · Report a bug (Ctrl+Shift+B)",
            (SimUiKit.MenuEntry[])
            [
                new("Hide all  (Ctrl+H)", "Hide everything but the badge for a quick look -- click the badge or Ctrl+H to bring it back", SimActions.ToggleHideAll),
                new("Log  (Ctrl+L)", "Show / hide the action log (Ctrl+L)", SimActions.ToggleLog),
                new("HUD  (F4)", "Show / hide the advice HUD (F4)", SimActions.ToggleHud),
                new("Help  (F1)", "What everything does (F1)", SimActions.ToggleHelp),
                new("Settings", "Mod settings -- works mid-run (the main-menu page is run-locked)", SimSettingsPanel.Instance.Toggle),
                new("Report a bug  (Ctrl+Shift+B)", "Send this position and the logs to the developers, with a note on what went wrong", SimActions.ReportBug),
                // The panels in their own window around the game (the
                // Spire Mentor frame; SimFrameLauncher): starts it, or,
                // framed with the panels moved inside, moves them back out
                // (this toolbar hides while they are outside, so the way
                // back in is the frame's own Panels menu or closing it).
                new("Panels in their own window", "Move these panels out of the game into a window around it (closing that window brings them back)",
                    () => SimFrameLauncher.SetWanted(true, "Toolbar")),
            ], SetHint));
        AddMenu(SimUiKit.MakeMenuButton("Workbench",
            "Open the workbench in your browser · Copy state (Ctrl+Shift+C) to paste a position into it",
            new SimUiKit.MenuEntry[]
            {
                new("Open workbench", "Open the workbench in your browser: logs, search tree, run tree", () => SimWorkbenchLink.Open()),
                new("Copy state  (Ctrl+Shift+C)", "Copy this position to the clipboard -- paste it into the workbench to replay it, branch it, or sweep it in SimLab", SimActions.CopyState),
            }, SetHint));
        AddSection(grid, "More", _moreRow);

        body.AddChild(grid);

        // Hover hint: the owned replacement for native tooltips.
        _popHint = WrapLabel(new Color(0.85f, 0.88f, 0.95f), 13);
        _popHint.Visible = false;
        _footer.AddChild(_popHint);
        body.AddChild(_footer);

        popover.AddChild(body);
        _popover = popover;
        _popoverBody = body;
        return popover;
    }

    /// <summary>Under the badge while the popover is closed: the advisor's
    /// note (an autobattle stop, a hand-back) and the hover hint.</summary>
    private VBoxContainer BuildUnder()
    {
        var under = new VBoxContainer
        {
            Name = "SpireMentorBadgeNote",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Visible = false,
        };
        under.AddThemeConstantOverride("separation", 2);
        _underNote = WrapLabel(new Color(1.0f, 0.86f, 0.55f), 13);
        _underNote.HorizontalAlignment = HorizontalAlignment.Right;
        _underNote.CustomMinimumSize = new Vector2(UnderWidth, 0f);
        under.AddChild(_underNote);
        _underHint = WrapLabel(new Color(0.85f, 0.88f, 0.95f), 13);
        _underHint.HorizontalAlignment = HorizontalAlignment.Right;
        _underHint.CustomMinimumSize = new Vector2(UnderWidth, 0f);
        under.AddChild(_underHint);
        _under = under;
        return under;
    }

    /// <summary>A wrapping label. Its minimum width is the popover's inner
    /// width, so it wraps at a sane width from its first frame (an autowrap
    /// label measured at width 0 reports one word per line).</summary>
    private static Label WrapLabel(Color color, int size)
    {
        Label label = SimUiKit.SmallLabel("", color, size);
        label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        label.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        label.CustomMinimumSize = new Vector2(PopoverMinWidth - 24f, 0f);
        return label;
    }

    private static HBoxContainer Cell(int separation)
    {
        var cell = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        cell.AddThemeConstantOverride("separation", separation);
        return cell;
    }

    /// <summary>One grid row: a grey caption and its controls. Returns the
    /// caption (to hide the row with its controls).</summary>
    private static Label AddSection(GridContainer grid, string caption, Control controls)
    {
        Label label = SimUiKit.Caption(caption);
        label.VerticalAlignment = VerticalAlignment.Center;
        grid.AddChild(label);
        grid.AddChild(controls);
        return label;
    }

    private void AddMenu(MenuButton menu)
    {
        _menus.Add(menu);
        _moreRow!.AddChild(menu);
    }

    private static string ModeHint(MentorMode mode) =>
        mode switch
        {
            MentorMode.Off => "Turn the AI off: the game plays exactly as without the mod (F8 toggles)",
            MentorMode.Coach => "Coach: nothing shows before you act; Peek (F5) shows the next pick, and a note " +
                                "afterwards says where you differed from the AI",
            MentorMode.Hints => "Hints: the AI's opinion shows while you decide -- card scores, ranked moves, " +
                                "win chance, the route it likes",
            _ => "AI plays: the AI plays this run until you stop it, handing back what it cannot play " +
                 "(same as F7; a config-armed Autopilot shows here too)",
        };

    // ------------------------------------------------------------------
    // Behavior (main thread only below this line)
    // ------------------------------------------------------------------

    private void OnBadgePressed()
    {
        // While everything is hidden the badge is the "show it again" button.
        if (SimUiHide.Hidden)
        {
            SimUiHide.Toggle();
            return;
        }
        SetExpanded(!_expanded);
    }

    private void SetExpanded(bool open)
    {
        if (_expanded == open)
        {
            return;
        }
        _expanded = open;
        SpireMentorModConfig.ToolbarExpanded = open;
        SpireMentorModConfig.SaveSoon();
        if (!open)
        {
            SetHint(null);
        }
        Refresh();
    }

    private void SetBrain(string brain)
    {
        if (AdvisorBrainRules.Normalize(brain) != SimAdvisor.CurrentBrain)
        {
            SimActions.ToggleBrain("The Brain switch");
        }
    }

    /// <summary>
    /// The popover's outside-click / Esc close, polled every 40 ms while it is
    /// open. Reads Input and never consumes anything: the click or the Esc
    /// still reaches the game (a card played, the pause menu opened). A press
    /// that was already down when the popover opened (the badge click itself)
    /// does not count, nor does anything while one of its dropdowns is open
    /// (those close themselves).
    /// </summary>
    private void OnCloseWatch()
    {
        try
        {
            if (!_expanded || !Valid(_popover) || !_popover!.IsVisibleInTree())
            {
                _closeWatch?.Stop();
                return;
            }
            bool down = Input.IsMouseButtonPressed(MouseButton.Left) || Input.IsMouseButtonPressed(MouseButton.Right);
            bool esc = Input.IsKeyPressed(Key.Escape);
            bool menuOpen = _menus.Any(m => GodotObject.IsInstanceValid(m) && m.GetPopup().Visible);
            if (!menuOpen)
            {
                if (down && !_mouseWasDown)
                {
                    Vector2 mouse = _popover.GetViewport().GetMousePosition();
                    if (!_popover.GetGlobalRect().HasPoint(mouse) &&
                        !(Valid(_row) && _row!.GetGlobalRect().HasPoint(mouse)))
                    {
                        SetExpanded(false);
                    }
                }
                else if (esc && !_escWasDown)
                {
                    SetExpanded(false);
                }
            }
            _mouseWasDown = down;
            _escWasDown = esc;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Popover watch failed: {ex.Message}");
        }
    }

    private void OnGripGuiInput(InputEvent ev)
    {
        try
        {
            if (!Valid(_row))
            {
                return;
            }
            if (ev is InputEventMouseButton mb && mb.ButtonIndex == MouseButton.Left)
            {
                if (mb.Pressed && mb.DoubleClick)
                {
                    _gripPressed = false;
                    _dragging = false;
                    _undockedAnchor = null;
                    Layout();
                }
                else if (mb.Pressed)
                {
                    _gripPressed = true;
                    _dragging = false;
                    _pressPosition = mb.GlobalPosition;
                    _grabOffset = mb.GlobalPosition - CurrentAnchor();
                }
                else
                {
                    _gripPressed = false;
                    _dragging = false;
                }
                return;
            }

            if (ev is InputEventMouseMotion mm && _gripPressed)
            {
                if (!_dragging && mm.GlobalPosition.DistanceTo(_pressPosition) < DragThreshold)
                {
                    return;
                }
                _dragging = true;
                _undockedAnchor = mm.GlobalPosition - _grabOffset;
                Layout();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Grip input failed: {ex.Message}");
        }
    }

    /// <summary>The badge row's top-right corner now (the anchor).</summary>
    private Vector2 CurrentAnchor() =>
        new(_row!.Position.X + _row.Size.X, _row.Position.Y);

    private void SetHint(string? hint)
    {
        _hint = hint ?? "";
        RefreshNotes();
        Layout();
    }

    private void OnTick()
    {
        try
        {
            if (!Valid(_row))
            {
                return;
            }
            // The thinking dots step every ~0.45 s.
            if (++_ticks % 3 == 0)
            {
                _dotFrame++;
            }
            // Speed, autoplay, partners and the build counter raise no
            // Changed event; poll them.
            Refresh();
            // Framed: feed the frame's toolbar in both UI modes (FrameLink
            // dedups unchanged snapshots).
            if (FrameLink.Framed)
            {
                FrameLink.PublishUi(FrameSnapshot());
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Tick failed: {ex.Message}");
        }
    }

    private BadgeState State()
    {
        SimMode mode = SimRuntime.Mode;
        bool autobattle = mode == SimMode.Advisor && SimAdvisor.Instance.AutobattleOn;
        return new BadgeState(
            MentorModes.Current((SimModeLite)(int)mode, SimRuntime.Style,
                mode == SimMode.Advisor && SimRuntime.AdvisorAutoplayArmed, autobattle),
            Autopilot: mode == SimMode.Autopilot,
            AutopilotEngaged: SimRuntime.AutopilotEngaged,
            Autobattle: autobattle,
            Advisor: ShownAdvisorStatus(),
            BuildElapsed: BuildElapsed(),
            Hidden: SimUiHide.Hidden);
    }

    /// <summary>The advisor's status as the badge and the frame show it: a
    /// think shows only once it lasts (ThinkingIndicator).</summary>
    private AdvisorStatusLite ShownAdvisorStatus() =>
        _thinking.Shown((AdvisorStatusLite)(int)_advisorStatus, DateTime.UtcNow);

    /// <summary>The AI itself is playing: F7, autobattle, or an engaged Autopilot.</summary>
    private static bool AiPlaying(in BadgeState s) =>
        s.AutopilotEngaged || (!s.Autopilot && (s.Mode == MentorMode.AiPlays || s.Autobattle));

    private static Color ModeColor(MentorMode mode) =>
        mode switch
        {
            MentorMode.Coach => CoachColor,
            MentorMode.Hints => HintsColor,
            MentorMode.AiPlays => AiColor,
            _ => OffColor,
        };

    private void Refresh()
    {
        if (!EnsureBuilt())
        {
            return;
        }
        BadgeState s = State();
        bool hidden = SimUiHide.Hidden;
        bool aiPlaying = AiPlaying(s);

        if (Valid(_badge))
        {
            string arrow = hidden ? "" : _expanded ? "▾ " : "▸ ";
            SetText(_badge!, arrow + MentorBadge.Text(s, _dotFrame, SpireMentorModConfig.ReduceEffects));
            Color accent = aiPlaying || s.Autopilot ? AiColor : ModeColor(s.Mode);
            if (_badgeAccent != accent)
            {
                _badgeAccent = accent;
                _badge!.AddThemeStyleboxOverride("normal", SimUiKit.BadgeStyle(accent, hover: false));
                _badge.AddThemeStyleboxOverride("hover", SimUiKit.BadgeStyle(accent, hover: true));
                _badge.AddThemeStyleboxOverride("pressed", SimUiKit.BadgeStyle(accent, hover: true));
            }
        }

        string? stop = MentorBadge.StopLabel(s);
        if (Valid(_stop))
        {
            SetVisible(_stop!, stop != null);
            if (stop != null)
            {
                SetText(_stop!, stop);
            }
        }
        if (Valid(_aiFrame))
        {
            SetVisible(_aiFrame!, aiPlaying);
        }
        if (Valid(_exitFullscreen))
        {
            SetVisible(_exitFullscreen!, FrameLink.Framed && !FrameLink.UiOutside);
        }
        if (Valid(_partnerGo))
        {
            RefreshPartnerGo(_partnerGo!);
        }

        bool open = _expanded && !hidden;
        if (Valid(_popover))
        {
            SetVisible(_popover!, open);
            if (open)
            {
                RefreshPopover(s);
            }
        }
        if (open && _closeWatch != null && GodotObject.IsInstanceValid(_closeWatch) && _closeWatch.IsStopped())
        {
            _mouseWasDown = Input.IsMouseButtonPressed(MouseButton.Left) || Input.IsMouseButtonPressed(MouseButton.Right);
            _escWasDown = Input.IsKeyPressed(Key.Escape);
            _closeWatch.Start();
        }
        RefreshNotes();
        Layout();
    }

    private void RefreshPopover(in BadgeState s)
    {
        SimMode mode = SimRuntime.Mode;
        string? activity = MentorBadge.Activity(s, _dotFrame, SpireMentorModConfig.ReduceEffects);
        SetText(_headerTitle, $"{MentorBadge.Name} {MainFile.ModVersion}" + (activity == null ? "" : $" · {activity}"));
        SetText(_headerStatus, MentorBadge.Status(s));

        foreach ((MentorMode m, Button button) in _modeButtons)
        {
            if (!GodotObject.IsInstanceValid(button))
            {
                continue;
            }
            SetText(button, MentorModes.Label(m, s.Autopilot));
            Lit(button, m == s.Mode ? ModeColor(m) : null);
        }

        (bool brainShow, _) = BrainState();
        SetVisible(_brainCaption, brainShow);
        SetVisible(_brainControls, brainShow);
        if (brainShow)
        {
            bool instinct = AdvisorBrainRules.IsInstinct(SimAdvisor.CurrentBrain);
            Lit(_brainSearch, instinct ? null : BrainColor);
            Lit(_brainInstinct, instinct ? BrainColor : null);
        }

        bool advisor = mode == SimMode.Advisor;
        SetDisabled(_peek, !SimRuntime.CoachStyleActive);
        SetDisabled(_aiOnce, !advisor);
        SetDisabled(_autobattle, !advisor);
        SetDisabled(_aiAuto, !advisor);
        bool plainAuto = advisor && SimRuntime.AdvisorAutoplayArmed && !s.Autobattle;
        SetText(_autobattle, s.Autobattle ? "Stop autobattle  Shift+F7" : "Autobattle  Shift+F7");
        Lit(_autobattle, s.Autobattle ? AiColor : null);
        SetText(_aiAuto, plainAuto ? "Stop the AI  F7" : "AI plays until stopped  F7");
        Lit(_aiAuto, plainAuto ? AiColor : null);
        SetText(_speedLabel, SpeedSteps.Label(SimSpeed.Desired));
    }

    /// <summary>What is worth knowing beyond the mode: a hidden HUD, the last
    /// advice. Empty when there is nothing.</summary>
    private string FooterInfo()
    {
        if (SimRuntime.Mode == SimMode.Off)
        {
            return "";
        }
        var lines = new List<string>(2);
        // The HUD toggle persists across launches; a hidden HUD with a
        // connected advisor otherwise reads as "the coach does nothing"
        // (2026-10-17, a v0.5.0 player with ShowHud saved off).
        if (!SpireMentorModConfig.ShowHud)
        {
            lines.Add("The advice display is hidden: More › Panels › HUD (F4) shows it.");
        }
        if (!string.IsNullOrEmpty(_lastActionSummary))
        {
            lines.Add($"Last advice: {_lastActionSummary}");
        }
        return string.Join("\n", lines);
    }

    /// <summary>The note, the footer and the hover hint: in the popover while
    /// it is open, the note and the hint under the badge while it is closed.</summary>
    private void RefreshNotes()
    {
        bool hidden = SimUiHide.Hidden;
        bool open = _expanded && !hidden;
        string note = SimRuntime.Mode != SimMode.Off ? _note ?? "" : "";
        SetText(_headerNote, note);
        SetVisible(_headerNote, open && note.Length > 0);
        string info = open ? FooterInfo() : "";
        SetText(_footerInfo, info);
        SetVisible(_footerInfo, info.Length > 0);
        SetText(_popHint, _hint);
        SetVisible(_popHint, open && _hint.Length > 0);
        SetVisible(_footer, open && (note.Length > 0 || info.Length > 0 || _hint.Length > 0));

        // The advice note is engine detail ("Instinct (e2e net, no search) ·
        // combat turn · 100% · 31 ms"): in the popover only. Under a closed
        // badge it sat on the game's own watermark (live, 2026-11-07).
        bool underNote = false;
        bool underHint = !open && _hint.Length > 0;
        SetText(_underNote, note);
        SetVisible(_underNote, underNote);
        SetText(_underHint, _hint);
        SetVisible(_underHint, underHint);
        SetVisible(_under, underNote || underHint);
    }

    /// <summary>
    /// Place the badge row (docked under the top bar's right end, or where it
    /// was dragged), the popover under it -- above it when the badge is in
    /// the lower half of the screen -- and the note under it, right-aligned to
    /// the badge, all clamped on screen. Called on every refresh and tick;
    /// cheap.
    /// </summary>
    private void Layout()
    {
        if (!Valid(_row))
        {
            return;
        }
        Vector2 view = _row.GetViewportRect().Size;
        Vector2 rowSize = FitSize(_row);
        Vector2 anchor = _undockedAnchor is { } moved
            ? ClampAnchor(moved, rowSize, view)
            : new Vector2(view.X - EdgeMargin, SimTopBarLayout.SafeTopYRight(_row));
        if (_undockedAnchor != null)
        {
            _undockedAnchor = anchor; // came back on screen if the window shrank
        }
        float rowX = Math.Clamp(anchor.X - rowSize.X, 0f, Math.Max(0f, view.X - rowSize.X));
        Place(_row, new Vector2(rowX, anchor.Y));
        float right = rowX + rowSize.X;
        float below = anchor.Y + rowSize.Y + PopGap;

        if (Valid(_popover) && _popover.Visible)
        {
            // Decided by where the badge is, not by the popover's height, so
            // it cannot flip back and forth as the popover's lines change.
            bool up = anchor.Y > view.Y * 0.5f;
            LayoutFooter(up);
            Vector2 size = FitSize(_popover);
            float y = up ? anchor.Y - PopGap - size.Y : below;
            Place(_popover, new Vector2(
                Math.Clamp(right - size.X, 0f, Math.Max(0f, view.X - size.X)),
                Math.Clamp(y, 0f, Math.Max(0f, view.Y - size.Y))));
        }
        if (Valid(_under) && _under.Visible)
        {
            Vector2 size = FitSize(_under);
            Place(_under, new Vector2(Math.Clamp(right - size.X, 0f, Math.Max(0f, view.X - size.X)), below));
        }
    }

    /// <summary>The footer (note, last advice, hover hint) goes on the far
    /// side of the controls from the badge: under them when the popover hangs
    /// down, between the header and them when it opens upwards (its bottom
    /// edge is then the fixed one). Either way a footer that grows or shrinks
    /// moves nothing the mouse is on.</summary>
    private void LayoutFooter(bool up)
    {
        if (!Valid(_popoverBody) || !Valid(_footer) || !Valid(_headerStatus))
        {
            return;
        }
        int want = up ? _headerStatus.GetIndex() + 1 : _popoverBody.GetChildCount() - 1;
        if (_footer.GetIndex() != want)
        {
            _popoverBody.MoveChild(_footer, want);
        }
    }

    /// <summary>Keep the badge grabbable: the whole row on screen.</summary>
    private static Vector2 ClampAnchor(Vector2 anchor, Vector2 rowSize, Vector2 view) =>
        new(Math.Clamp(anchor.X, Math.Min(rowSize.X, view.X), Math.Max(rowSize.X, view.X)),
            Math.Clamp(anchor.Y, 0f, Math.Max(0f, view.Y - rowSize.Y)));

    /// <summary>A free-standing container keeps its largest size; shrink it
    /// back to its content (a button hid, a label got shorter).</summary>
    private static Vector2 FitSize(Control control)
    {
        Vector2 min = control.GetCombinedMinimumSize();
        if (control.Size != min)
        {
            control.Size = min;
        }
        return min;
    }

    private static void Place(Control control, Vector2 position)
    {
        if (control.Position.DistanceTo(position) > 0.5f)
        {
            control.Position = position;
        }
    }

    /// <summary>The badge row, with the popover while it is open (global).</summary>
    private Rect2 OccupiedRect()
    {
        Rect2 rect = _row!.GetGlobalRect();
        if (Valid(_popover) && _popover!.Visible)
        {
            rect = rect.Merge(_popover.GetGlobalRect());
        }
        return rect;
    }

    /// <summary>Wall-clock since the engine build started, as "m:ss".</summary>
    private string BuildElapsed() =>
        _buildStartedUtc is { } started ? MentorBadge.Elapsed(DateTime.UtcNow - started) : "0:00";

    /// <summary>The brain switch is shown (only in a run Instinct plays:
    /// elsewhere the search answers whatever is chosen,
    /// AdvisorBrainRules.Requested), and what it says. Polled: the advisor's
    /// capability and the run's character raise no event.</summary>
    private static (bool Show, string Label) BrainState() =>
        (SimRuntime.Mode == SimMode.Advisor && SimAdvisor.InstinctPlaysThisRun(),
         AdvisorBrainRules.Label(SimAdvisor.CurrentBrain));

    /// <summary>"Partner go": shown in a combat with AI partners, live while
    /// one waits for you ("▶ Partner go", with the setup hold's seconds).</summary>
    private static void RefreshPartnerGo(Button button)
    {
        IReadOnlyList<SimAiPartner> partners = SimAiPartner.All;
        bool show = partners.Count > 0 && CombatManager.Instance.IsInProgress;
        SetVisible(button, show);
        if (!show)
        {
            return;
        }
        SimAiPartner? waiting = partners.FirstOrDefault(p => p.Brain.Waiting != PartnerWait.None);
        int seconds = waiting?.Brain.HoldSecondsLeft ?? 0;
        SetText(button, waiting == null ? "Partner go" : seconds > 0 ? $"▶ Partner go {seconds}s" : "▶ Partner go");
        SetDisabled(button, waiting == null);
    }

    private void Lit(Button? button, Color? accent)
    {
        if (button == null || !GodotObject.IsInstanceValid(button))
        {
            return;
        }
        if (_lit.TryGetValue(button, out Color? was) && was == accent)
        {
            return;
        }
        _lit[button] = accent;
        SimUiKit.SetLit(button, accent);
    }

    private static bool Valid([NotNullWhen(true)] GodotObject? node) => node != null && GodotObject.IsInstanceValid(node);

    private static void SetText(Button? button, string text)
    {
        if (Valid(button) && button!.Text != text)
        {
            button.Text = text;
        }
    }

    private static void SetText(Label? label, string text)
    {
        if (Valid(label) && label!.Text != text)
        {
            label.Text = text;
        }
    }

    private static void SetVisible(CanvasItem? item, bool visible)
    {
        if (Valid(item) && item!.Visible != visible)
        {
            item.Visible = visible;
        }
    }

    private static void SetDisabled(BaseButton? button, bool disabled)
    {
        if (Valid(button) && button!.Disabled != disabled)
        {
            button.Disabled = disabled;
        }
    }

    /// <summary>The panels moved into or out of the game (FrameLink). Main thread.</summary>
    public void OnFrameUiModeChanged()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            _layer.Visible = !FrameLink.UiOutside;
        }
        Refresh();
    }

    /// <summary>Everything the toolbar shows, for the frame's toolbar
    /// (FrameLink `ui`). Main thread (reads Godot-side state).</summary>
    private object FrameSnapshot()
    {
        SimMode mode = SimRuntime.Mode;
        BadgeState s = State();
        IReadOnlyList<SimAiPartner> partners = SimAiPartner.All;
        bool partnerShow = partners.Count > 0 && CombatManager.Instance.IsInProgress;
        SimAiPartner? waiting = partnerShow
            ? partners.FirstOrDefault(p => p.Brain.Waiting != PartnerWait.None)
            : null;
        (bool brainShow, string brainLabel) = BrainState();
        // The badge's words first (dots still: the frame dedups snapshots and
        // moving dots would resend one every half second), then the sentence,
        // the footer and the note.
        var statusLines = new List<string>
        {
            MentorBadge.Text(s, 0, still: true),
            MentorBadge.Status(s),
            FooterInfo(),
        };
        if (SimRuntime.Mode != SimMode.Off && !string.IsNullOrEmpty(_note))
        {
            statusLines.Add(_note);
        }
        string status = string.Join("\n", statusLines.Where(line => line.Length > 0));
        return new
        {
            mode = mode.ToString().ToLowerInvariant(),
            mode_name = mode == SimMode.Off ? "AI off" : CoachStyle.ModeName((SimModeLite)(int)mode, SimRuntime.Style),
            // The badge's Mode switch: off / coach / hints / aiplays.
            mentor_mode = s.Mode.ToString().ToLowerInvariant(),
            power_label = CoachStyle.PowerLabel((SimModeLite)(int)mode),
            style_label = CoachStyle.StyleLabel(SimRuntime.Style),
            coach_style = SimRuntime.CoachStyleActive,
            takeover_enabled = mode == SimMode.Advisor,
            autoplay_armed = SimRuntime.AdvisorAutoplayArmed,
            autobattle_on = SimAdvisor.Instance.AutobattleOn,
            speed = SpeedSteps.Label(SimSpeed.Desired),
            // The brain switch; the frame's button sends `toggle_brain`.
            brain = new
            {
                show = brainShow,
                value = SimAdvisor.CurrentBrain,
                label = brainLabel,
            },
            partner_go = new
            {
                show = partnerShow,
                waiting = waiting != null,
                seconds = waiting?.Brain.HoldSecondsLeft ?? 0,
            },
            advisor_status = ShownAdvisorStatus().ToString().ToLowerInvariant(),
            build_elapsed = _advisorStatus == SimAdvisorStatus.Building ? BuildElapsed() : null,
            hud_shown = SpireMentorModConfig.ShowHud,
            hidden = SimUiHide.Hidden,
            last_action = _lastActionSummary,
            note = _note,
            // In the player's words (MentorBadge.Status), multi-line.
            status,
            ui_outside = FrameLink.UiOutside,
        };
    }

    private static void RunOnMainThread(Action action)
    {
        try
        {
            if (NGame.IsMainThread())
            {
                Guarded(action);
            }
            else
            {
                Callable.From(() => Guarded(action)).CallDeferred();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Dispatch failed: {ex.Message}");
        }
    }

    private static void Guarded(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Ignoring UI error (never breaks gameplay): {ex.Message}");
        }
    }
}
