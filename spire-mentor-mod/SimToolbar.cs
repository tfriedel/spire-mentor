// SimToolbar.cs -- the clickable mod strip (discoverability layer).
//
// Everything the F-keys do, as buttons: mode toggle, AI once/auto, speed,
// then three dropdowns -- Rewind, Panels, Workbench -- so nobody has to
// memorise a key list to use the mod. The live controls (coach state, AI
// once/auto, speed) stay flat because their state is read at a glance;
// the one-shot verbs are grouped by job so the strip fits a 1280-wide
// window. Hotkeys stay as shortcuts; every entry routes through the same
// SimActions verbs.
//
// Input-safety contract (see SimUiKit): only the buttons are
// MouseFilter.Stop; every container, gap and label is Ignore, so clicks
// anywhere but a visible button land in the game. Collapse is CLICK-driven
// via an always-visible pill -- hover-collapse would put a moving Stop
// region over combat. The strip sits at CanvasLayer 98 (below HUD 99 / log
// 100 / help 101) at the top-left, pushed below the game's top bar by
// SimTopBarLayout.
//
// Placement: docked under the game's top bar by default; the grip at the
// row's left end drags it anywhere (the strip can cover something the player
// needs to see), and double-clicking the grip docks it again. The dragged
// position lasts for the session, like the action log's. The advice panel and
// the coach chips follow a dragged strip (UndockedRect, AdviceColumn).
//
// Refresh model: SimRuntime.Changed handles mode flips from any surface;
// a 150 ms Godot.Timer handles the state that raises no event (SimSpeed,
// AdvisorAutoplayArmed, top-bar drift). No source generators -> no _Process
// (see SpireMentor.csproj header).

using System;
using System.Collections.Generic;
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

    private CanvasLayer? _layer;
    private VBoxContainer? _container;
    private Button? _pill;
    private PanelContainer? _strip;
    private HBoxContainer? _buttons;
    private Button? _coach;
    private Button? _style;
    private Button? _peek;
    private Button? _aiOnce;
    private Button? _aiAuto;
    private Button? _autobattle;
    private Button? _partnerGo;
    private Label? _speedLabel;
    private Label? _statusLabel;
    private Label? _hintLabel;
    private readonly Dictionary<string, Button> _extensions = new();

    private bool _expanded = true;
    private SimMode _lastMode = SimMode.Off;
    private bool _modeSeen;

    private SimAdvisorStatus _advisorStatus = SimAdvisorStatus.None;
    /// <summary>When the current engine build started, for the pill's counter.</summary>
    private DateTime? _buildStartedUtc;
    private string _lastActionSummary = "";
    private string? _note;

    // Grip drag state (same click-vs-drag threshold idiom as SimActionLog's
    // header). _undockedPosition is null while docked under the top bar.
    private const float DragThreshold = 4f;
    private bool _gripPressed;
    private bool _dragging;
    private Vector2 _grabOffset;
    private Vector2 _pressPosition;
    private Vector2? _undockedPosition;

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
            // Stamp the start so the pill can count the wait up. A build this
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
    /// Extension point for later features (e.g. the rewind cluster): appends
    /// a button to the strip, once per id. Safe from any thread.
    /// </summary>
    public void AddExtensionButton(string id, string label, string hint, Action onPressed)
    {
        RunOnMainThread(() =>
        {
            if (!EnsureBuilt() || _buttons == null || _extensions.ContainsKey(id))
            {
                return;
            }
            Button button = SimUiKit.MakeButton(label, hint, onPressed, SetHint);
            _buttons.AddChild(button);
            _extensions[id] = button;
        });
    }

    /// <summary>Bottom edge (global Y) of the visible strip, or null when not
    /// built or dragged off its dock -- the docked panels park below this.
    /// A dragged strip is <see cref="UndockedRect"/>.</summary>
    public float? BottomY()
    {
        if (_container == null || !GodotObject.IsInstanceValid(_container) ||
            !_container.IsVisibleInTree() || _undockedPosition != null)
        {
            return null;
        }
        Rect2 rect = _container.GetGlobalRect();
        return rect.Position.Y + rect.Size.Y;
    }

    /// <summary>The visible strip's global rect and the viewport size while
    /// the player has dragged it off its dock, else null (docked: BottomY).
    /// SimHud moves the advice panel and the coach chips along with it.</summary>
    public (Rect2 Rect, Vector2 Viewport)? UndockedRect()
    {
        if (_undockedPosition == null || _container == null ||
            !GodotObject.IsInstanceValid(_container) || !_container.IsVisibleInTree())
        {
            return null;
        }
        return (_container.GetGlobalRect(), _container.GetViewportRect().Size);
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

            _expanded = SpireMentorModConfig.ToolbarExpanded && SimRuntime.Mode != SimMode.Off;

            var layer = new CanvasLayer
            {
                Name = "SpireMentorToolbarLayer",
                Layer = 98,
                // Panels outside the game: the frame draws the toolbar; this
                // one only ticks (OnFrameUiModeChanged flips it).
                Visible = !FrameLink.UiOutside,
            };

            var container = new VBoxContainer
            {
                Name = "SpireMentorToolbar",
                Position = _undockedPosition ?? new Vector2(PanelLeft, SimTopBarLayout.FallbackTop),
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            container.AddThemeConstantOverride("separation", 2);

            // Row 1: the pill + the button strip.
            var row = new HBoxContainer
            {
                Name = "ToolbarRow",
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            row.AddThemeConstantOverride("separation", 6);

            // The drag handle: a Stop region of its own, so the pill and the
            // buttons keep plain click semantics.
            Label grip = SimUiKit.SmallLabel("⠿", new Color(0.85f, 0.88f, 0.95f), 18);
            grip.MouseFilter = Control.MouseFilterEnum.Stop;
            grip.MouseDefaultCursorShape = Control.CursorShape.Move;
            grip.VerticalAlignment = VerticalAlignment.Center;
            grip.Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(OnGripGuiInput));
            grip.Connect(Control.SignalName.MouseEntered, Callable.From(() =>
                SetHint("Drag to move the toolbar · double-click to put it back")));
            grip.Connect(Control.SignalName.MouseExited, Callable.From(() => SetHint(null)));
            row.AddChild(grip);

            _pill = SimUiKit.MakeButton("Spire Mentor", "Show / hide the Spire Mentor toolbar", TogglePill, SetHint);
            row.AddChild(_pill);

            var strip = new PanelContainer
            {
                Name = "ToolbarStrip",
                MouseFilter = Control.MouseFilterEnum.Ignore,
                Visible = false,
            };
            strip.AddThemeStyleboxOverride("panel", SimUiKit.PanelStyle());

            var buttons = new HBoxContainer
            {
                Name = "ToolbarButtons",
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            buttons.AddThemeConstantOverride("separation", 4);

            _coach = SimUiKit.MakeButton("AI off",
                "Turn the AI coach on / off (F8)",
                SimActions.ToggleAdvisor, SetHint);
            buttons.AddChild(_coach);

            _style = SimUiKit.MakeButton("Coach",
                "Switch style: Hints (the AI's opinion while you decide) <-> Coach (nothing " +
                "until you act; F5 peeks, a note afterwards says where you differed)",
                SimActions.SwitchStyle, SetHint);
            buttons.AddChild(_style);

            // Only shown in the Coach style (Hints already shows the pick).
            _peek = SimUiKit.MakeButton("Peek",
                "Show the AI's pick for just this decision; it hides again once you act (F5)",
                SimActions.Peek, SetHint);
            buttons.AddChild(_peek);

            _aiOnce = SimUiKit.MakeButton("AI once",
                "Let the AI take just this one decision (F6) -- needs the coach on",
                () => SimActions.AiOnce("The AI-once button"), SetHint);
            buttons.AddChild(_aiOnce);

            _aiAuto = SimUiKit.MakeButton("AI auto",
                "Let the AI keep playing until you stop it (F7) -- needs the coach on",
                () => SimActions.ToggleAutoplay("The AI-auto button"), SetHint);
            buttons.AddChild(_aiAuto);

            _autobattle = SimUiKit.MakeButton("Autobattle",
                "The AI plays the fights, you make every run-level decision -- card rewards, map, " +
                "shops, rest sites, events -- with a note on what changed since your last one (Shift+F7)",
                () => SimActions.ToggleAutobattle("The Autobattle button"), SetHint);
            buttons.AddChild(_autobattle);

            // Co-op with AI partners, in combat only (RefreshLiveLabels).
            _partnerGo = SimUiKit.MakeButton("Partner go",
                "Your AI partner waits for you (your first play, your end turn, or a setup play " +
                "of yours it wants first): let it play the rest of its turn now (Shift+F6)",
                () => SimActions.PartnersGo("The Partner go button"), SetHint);
            _partnerGo.Visible = false;
            buttons.AddChild(_partnerGo);

            buttons.AddChild(SimUiKit.MakeButton("-", "Game speed down: 4x, 3x, 2x, 1x (F9)",
                () => SimActions.StepSpeed(-1), SetHint));
            _speedLabel = SimUiKit.SmallLabel("1.0x", Colors.White);
            _speedLabel.VerticalAlignment = VerticalAlignment.Center;
            buttons.AddChild(_speedLabel);
            buttons.AddChild(SimUiKit.MakeButton("+", "Game speed up: 1x, 2x, 3x, 4x (F10)",
                () => SimActions.StepSpeed(1), SetHint));

            // Everything past the live controls is a one-shot verb, grouped
            // by job into three dropdowns (same shape as the workbench's top
            // bar): 16 flat buttons used to run off a 1280-wide window.
            //
            // Rewind cluster: the game saves at room entry and post-combat,
            // so those are the room-level rewind targets (see SimSnapshots);
            // the turn undo replays the game's own combat tape on top of the
            // room-entry save (SimCombatUndo).
            buttons.AddChild(SimUiKit.MakeMenuButton("Rewind",
                "⟲ Turn: undo to the start of this combat turn · ⟲ Room: restart this room (same seed) · ⟲⟲ Back: one room further back",
                new SimUiKit.MenuEntry[]
                {
                    new("⟲ Turn", "Undo to the start of the current combat turn (replays the game's combat tape)", SimActions.UndoTurn),
                    new("⟲ Room", "Rewind to the start of the current room (same seed, exact state)", SimActions.RestartRoom),
                    new("⟲⟲ Back", "Rewind one room further back", SimActions.BackOneRoom),
                }, SetHint));
            buttons.AddChild(SimUiKit.MakeMenuButton("Panels",
                "Hide all (Ctrl+H) · Log (Ctrl+L) · HUD (F4) · Help (F1) · Settings (works mid-run) · Report a bug (Ctrl+Shift+B)",
                (SimUiKit.MenuEntry[])
                [
                    new("Hide all  (Ctrl+H)", "Hide everything but the pill for a quick look -- click the pill or Ctrl+H to bring it back", SimActions.ToggleHideAll),
                    new("Log  (Ctrl+L)", "Show / hide the action log (Ctrl+L)", SimActions.ToggleLog),
                    new("HUD  (F4)", "Show / hide the advice HUD (F4)", SimActions.ToggleHud),
                    new("Help  (F1)", "What everything does (F1)", SimActions.ToggleHelp),
                    new("Settings", "Mod settings -- works mid-run (the main-menu page is run-locked)", SimSettingsPanel.Instance.Toggle),
                    new("Report a bug  (Ctrl+Shift+B)", "Send this position and the logs to the developers, with a note on what went wrong", SimActions.ReportBug),
                    // Launched by the Spire Mentor frame with the panels moved
                    // inside: the way back out (the frame's own menu is hidden).
                    .. (FrameLink.Framed
                        ? new SimUiKit.MenuEntry[] { new("Panels outside the game", "Put these panels back around the game, in the Spire Mentor frame", () => FrameLink.SetUiOutside(true)) }
                        : System.Array.Empty<SimUiKit.MenuEntry>()),
                ], SetHint));
            buttons.AddChild(SimUiKit.MakeMenuButton("Workbench",
                "Open the workbench in your browser · Copy state (Ctrl+Shift+C) to paste a position into it",
                new SimUiKit.MenuEntry[]
                {
                    new("Open workbench", "Open the workbench in your browser: logs, search tree, run tree", () => SimWorkbenchLink.Open()),
                    new("Copy state  (Ctrl+Shift+C)", "Copy this position to the clipboard -- paste it into the workbench to replay it, branch it, or sweep it in SimLab", SimActions.CopyState),
                }, SetHint));

            strip.AddChild(buttons);
            row.AddChild(strip);
            container.AddChild(row);

            // Row 2: status line (mode / advisor / last action / note).
            _statusLabel = SimUiKit.SmallLabel("", Colors.White, 15);
            _statusLabel.Visible = false;
            container.AddChild(_statusLabel);

            // Row 3: hover hint -- the owned replacement for native tooltips.
            _hintLabel = SimUiKit.SmallLabel("", new Color(0.85f, 0.88f, 0.95f));
            _hintLabel.Visible = false;
            container.AddChild(_hintLabel);

            var ticker = new Godot.Timer
            {
                Name = "SpireMentorToolbarTicker",
                WaitTime = 0.15,
                Autostart = true,
                OneShot = false,
            };
            ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(OnTick));
            layer.AddChild(ticker);

            layer.AddChild(container);
            tree.Root.AddChildSafely(layer);

            _layer = layer;
            _container = container;
            _strip = strip;
            _buttons = buttons;
            Logger.Log("[SimToolbar] Toolbar created");
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Failed to create toolbar: {ex.Message}");
            _layer = null;
            _container = null;
            return false;
        }
    }

    private const float PanelLeft = 16f;

    // ------------------------------------------------------------------
    // Behavior (main thread only below this line)
    // ------------------------------------------------------------------

    private void TogglePill()
    {
        // While everything is hidden the pill is the "show it again" button.
        if (SimUiHide.Hidden)
        {
            SimUiHide.Toggle();
            return;
        }
        _expanded = !_expanded;
        SpireMentorModConfig.ToolbarExpanded = _expanded;
        SpireMentorModConfig.SaveSoon();
        Refresh();
    }

    private void OnGripGuiInput(InputEvent ev)
    {
        try
        {
            if (_container == null || !GodotObject.IsInstanceValid(_container))
            {
                return;
            }
            if (ev is InputEventMouseButton mb && mb.ButtonIndex == MouseButton.Left)
            {
                if (mb.Pressed && mb.DoubleClick)
                {
                    _gripPressed = false;
                    _dragging = false;
                    _undockedPosition = null; // OnTick re-docks it
                }
                else if (mb.Pressed)
                {
                    _gripPressed = true;
                    _dragging = false;
                    _pressPosition = mb.GlobalPosition;
                    _grabOffset = mb.GlobalPosition - _container.Position;
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
                _undockedPosition = ClampToViewport(mm.GlobalPosition - _grabOffset);
                _container.Position = _undockedPosition.Value;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimToolbar] Grip input failed: {ex.Message}");
        }
    }

    /// <summary>Keep the grip on screen so the strip can always be grabbed
    /// again (the grip is the row's first child, at the container's origin).</summary>
    private Vector2 ClampToViewport(Vector2 position)
    {
        Vector2 view = _container!.GetViewportRect().Size;
        return new Vector2(Math.Clamp(position.X, 0f, Math.Max(0f, view.X - 30f)),
                           Math.Clamp(position.Y, 0f, Math.Max(0f, view.Y - 30f)));
    }

    private void SetHint(string? hint)
    {
        if (_hintLabel == null || !GodotObject.IsInstanceValid(_hintLabel))
        {
            return;
        }
        _hintLabel.Text = hint ?? "";
        _hintLabel.Visible = !string.IsNullOrEmpty(hint);
    }

    private void OnTick()
    {
        try
        {
            if (_container == null || !GodotObject.IsInstanceValid(_container))
            {
                return;
            }
            // Docked: follow the game's top bar. Undocked: stay where the
            // player put it, but come back on screen if the window shrank.
            Vector2 want = _undockedPosition is { } moved
                ? ClampToViewport(moved)
                : new Vector2(PanelLeft, SimTopBarLayout.SafeTopY(_container));
            if (_container.Position.DistanceTo(want) > 0.5f)
            {
                _container.Position = want;
            }
            // Speed and autoplay-armed raise no Changed event; poll them.
            RefreshLiveLabels();
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

    private void Refresh()
    {
        if (!EnsureBuilt())
        {
            return;
        }

        SimMode mode = SimRuntime.Mode;
        if (!_modeSeen || mode != _lastMode)
        {
            // Off auto-collapses to the pill (the mod should take no space
            // while it is doing nothing); turning it on from ANY surface
            // expands the strip so the new state is visible.
            if (_modeSeen)
            {
                _expanded = mode != SimMode.Off;
            }
            _lastMode = mode;
            _modeSeen = true;
        }

        if (_strip != null && GodotObject.IsInstanceValid(_strip))
        {
            _strip.Visible = _expanded && !SimUiHide.Hidden;
        }

        if (_pill != null && GodotObject.IsInstanceValid(_pill))
        {
            _pill.Text = PillText(mode);
        }

        if (_coach != null && GodotObject.IsInstanceValid(_coach))
        {
            _coach.Text = CoachStyle.PowerLabel((SimModeLite)(int)mode);
        }

        if (_style != null && GodotObject.IsInstanceValid(_style))
        {
            _style.Text = CoachStyle.StyleLabel(SimRuntime.Style);
        }

        if (_peek != null && GodotObject.IsInstanceValid(_peek))
        {
            _peek.Visible = SimRuntime.CoachStyleActive;
        }

        bool takeoverEnabled = mode == SimMode.Advisor;
        if (_aiOnce != null && GodotObject.IsInstanceValid(_aiOnce))
        {
            _aiOnce.Disabled = !takeoverEnabled;
        }
        if (_aiAuto != null && GodotObject.IsInstanceValid(_aiAuto))
        {
            _aiAuto.Disabled = !takeoverEnabled;
        }
        if (_autobattle != null && GodotObject.IsInstanceValid(_autobattle))
        {
            _autobattle.Disabled = !takeoverEnabled;
        }

        RefreshLiveLabels();
        RefreshStatusLine();
    }

    /// <summary>
    /// The collapsed pill's label. An engine build outranks the mode here: it
    /// is the one state where nothing works yet and the only correct action is
    /// to wait, and the pill shows collapsed — which the status line does not.
    /// </summary>
    private string PillText(SimMode mode) =>
        _advisorStatus == SimAdvisorStatus.Building
            ? $"▸ Spire Mentor — building engine… {BuildElapsed()} (leave running)"
            : SimUiHide.Hidden
                ? "▸ Spire Mentor — hidden (click or Ctrl+H)"
            : _expanded
                ? "▾ Spire Mentor"
                : mode == SimMode.Off
                    ? "▸ Spire Mentor — AI off (click)"
                    : $"▸ Spire Mentor · {CoachStyle.ModeName((SimModeLite)(int)mode, SimRuntime.Style)}";

    /// <summary>Wall-clock since the engine build started, as "m:ss".</summary>
    private string BuildElapsed()
    {
        if (_buildStartedUtc is not { } started)
        {
            return "0:00";
        }
        TimeSpan d = DateTime.UtcNow - started;
        if (d < TimeSpan.Zero)
        {
            d = TimeSpan.Zero; // clock stepped back mid-build; never render "-1:59"
        }
        return $"{(int)d.TotalMinutes}:{d.Seconds:00}";
    }

    private void RefreshLiveLabels()
    {
        // The build counter has no event to fire on -- it is pure elapsed time,
        // so the 150 ms ticker is what advances it.
        if (_advisorStatus == SimAdvisorStatus.Building
            && _pill != null && GodotObject.IsInstanceValid(_pill))
        {
            string text = PillText(SimRuntime.Mode);
            if (_pill.Text != text)
            {
                _pill.Text = text;
            }
        }
        if (_speedLabel != null && GodotObject.IsInstanceValid(_speedLabel))
        {
            string speed = SpeedSteps.Label(SimSpeed.Desired);
            if (_speedLabel.Text != speed)
            {
                _speedLabel.Text = speed;
            }
        }
        if (_aiAuto != null && GodotObject.IsInstanceValid(_aiAuto))
        {
            string label = SimRuntime.AdvisorAutoplayArmed ? "Stop auto" : "AI auto";
            if (_aiAuto.Text != label)
            {
                _aiAuto.Text = label;
            }
        }
        if (_autobattle != null && GodotObject.IsInstanceValid(_autobattle))
        {
            string label = SimAdvisor.Instance.AutobattleOn ? "Stop battle" : "Autobattle";
            if (_autobattle.Text != label)
            {
                _autobattle.Text = label;
            }
        }
        if (_partnerGo != null && GodotObject.IsInstanceValid(_partnerGo))
        {
            RefreshPartnerGo(_partnerGo);
        }
    }

    /// <summary>"Partner go": shown in a combat with AI partners, live while
    /// one waits for you ("▶ Partner go", with the setup hold's seconds).</summary>
    private static void RefreshPartnerGo(Button button)
    {
        IReadOnlyList<SimAiPartner> partners = SimAiPartner.All;
        bool show = partners.Count > 0 && CombatManager.Instance.IsInProgress;
        if (button.Visible != show)
        {
            button.Visible = show;
        }
        if (!show)
        {
            return;
        }
        SimAiPartner? waiting = partners.FirstOrDefault(p => p.Brain.Waiting != PartnerWait.None);
        int seconds = waiting?.Brain.HoldSecondsLeft ?? 0;
        string text = waiting == null ? "Partner go" : seconds > 0 ? $"▶ Partner go {seconds}s" : "▶ Partner go";
        if (button.Text != text)
        {
            button.Text = text;
        }
        if (button.Disabled != (waiting == null))
        {
            button.Disabled = waiting == null;
        }
    }

    private void RefreshStatusLine()
    {
        if (_statusLabel == null || !GodotObject.IsInstanceValid(_statusLabel))
        {
            return;
        }
        string status = StatusText();
        bool interesting = _expanded ||
            (SimRuntime.Mode == SimMode.Autopilot && !SimRuntime.AutopilotEngaged);
        _statusLabel.Text = status;
        _statusLabel.Visible = interesting && SimRuntime.Mode != SimMode.Off && !SimUiHide.Hidden;
    }

    /// <summary>The status line: mode, advisor state, last action, note.</summary>
    private string StatusText()
    {
        string status = $"Spire Mentor · {SimRuntime.Mode}";
        // Autopilot armed by the config file but with nobody driving it: say
        // so, and say how to get out. Silence here is what let a human play a
        // whole run with the AI's modifiers on (2026-08-05).
        if (SimRuntime.Mode == SimMode.Autopilot && !SimRuntime.AutopilotEngaged)
        {
            status += " (idle — waiting for the AI client; F8 or 'Take over')";
        }
        if (_advisorStatus != SimAdvisorStatus.None)
        {
            status += $" · advisor: {_advisorStatus.ToString().ToLowerInvariant()}";
        }
        // The HUD toggle persists across launches; a hidden HUD with a
        // connected advisor otherwise reads as "the coach does nothing"
        // (2026-10-17, a v0.5.0 player with ShowHud saved off).
        if (SimRuntime.Mode != SimMode.Off && !SpireMentorModConfig.ShowHud)
        {
            status += " · HUD hidden (F4 shows it)";
        }
        if (_advisorStatus == SimAdvisorStatus.Building)
        {
            // Say how long and why, or the wait reads as a hang and gets killed.
            status += $"\nbuilding the Rust engine ({BuildElapsed()}) — first launch after an " +
                      "engine change, usually MINUTES. Quitting kills it and the next launch " +
                      "starts over; pre-build with `just advisor-warm` in sts2-pycli.";
        }
        if (!string.IsNullOrEmpty(_lastActionSummary))
        {
            status += $"\nlast: {_lastActionSummary}";
        }
        if (!string.IsNullOrEmpty(_note))
        {
            status += $"\n{_note}";
        }
        return status;
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
        IReadOnlyList<SimAiPartner> partners = SimAiPartner.All;
        bool partnerShow = partners.Count > 0 && CombatManager.Instance.IsInProgress;
        SimAiPartner? waiting = partnerShow
            ? partners.FirstOrDefault(p => p.Brain.Waiting != PartnerWait.None)
            : null;
        return new
        {
            mode = mode.ToString().ToLowerInvariant(),
            mode_name = mode == SimMode.Off ? "AI off" : CoachStyle.ModeName((SimModeLite)(int)mode, SimRuntime.Style),
            power_label = CoachStyle.PowerLabel((SimModeLite)(int)mode),
            style_label = CoachStyle.StyleLabel(SimRuntime.Style),
            coach_style = SimRuntime.CoachStyleActive,
            takeover_enabled = mode == SimMode.Advisor,
            autoplay_armed = SimRuntime.AdvisorAutoplayArmed,
            autobattle_on = SimAdvisor.Instance.AutobattleOn,
            speed = SpeedSteps.Label(SimSpeed.Desired),
            partner_go = new
            {
                show = partnerShow,
                waiting = waiting != null,
                seconds = waiting?.Brain.HoldSecondsLeft ?? 0,
            },
            advisor_status = _advisorStatus.ToString().ToLowerInvariant(),
            build_elapsed = _advisorStatus == SimAdvisorStatus.Building ? BuildElapsed() : null,
            hud_shown = SpireMentorModConfig.ShowHud,
            hidden = SimUiHide.Hidden,
            last_action = _lastActionSummary,
            note = _note,
            status = StatusText(),
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
