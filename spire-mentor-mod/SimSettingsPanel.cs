// SimSettingsPanel.cs -- mod settings, reachable MID-RUN from the toolbar.
//
// BaseLib deliberately disables its "Mod Configuration" entry while a run is
// in progress (ModConfigPatch: "supporting in-run config will likely need
// work"), and re-enabling it would change BaseLib behavior for every mod.
// So the mod carries its own small panel instead, built from the SAME
// backing store (SpireMentorModConfig) and the same BaseLib NConfig* controls
// (they pull their visuals from the game's own pck, bind the static
// property, and refresh themselves on ConfigReloaded) -- the main-menu page,
// hotkeys, console and this panel can never disagree.
//
// Unlike the rest of the mod's overlays this panel's background is
// MouseFilter.Stop: it is an explicitly opened, explicitly closed dialog,
// so blocking the game underneath it is the expected behavior (same
// rationale as the action-log's header). While its line edits have focus
// the game suppresses our hotkeys (NHotkeyManager checks the focus owner);
// closing the panel releases focus.
//
// Persistence: BaseLib only saves configs on settings-submenu close and on
// quit, so this panel hooks ConfigChanged -> SaveDebounced while it exists
// (a crash mid-run must not lose edits).

using System;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes;

namespace SpireMentor;

internal sealed class SimSettingsPanel
{
    public static readonly SimSettingsPanel Instance = new();

    private SimSettingsPanel()
    {
    }

    /// <summary>Gap kept free under the panel, so it never runs off-screen.</summary>
    private const float BottomMargin = 16f;

    /// <summary>The scroll area never shrinks below this, however short the
    /// window: a few rows are better than none.</summary>
    private const float MinScrollHeight = 120f;

    private CanvasLayer? _layer;
    private PanelContainer? _panel;
    private Control? _title;
    private ScrollContainer? _scroll;
    private VBoxContainer? _rows;
    private VBoxContainer? _advanced;
    private Button? _advancedToggle;

    /// <summary>Toggle the panel (toolbar gear button). Safe from any thread.</summary>
    public void Toggle()
    {
        RunOnMainThread(() =>
        {
            if (!EnsureBuilt() || _panel == null)
            {
                return;
            }
            bool opening = !_panel.Visible;
            if (opening)
            {
                // Park below the toolbar strip, wherever it currently sits.
                float top = (SimToolbar.Instance.BottomY() ?? 60f) + 8f;
                _panel.Position = new Vector2(16f, top);
                FitToScreenSoon();
            }
            else
            {
                // Editing a LineEdit suppresses the game's hotkey dispatch;
                // never leave focus behind on close.
                _panel.GetViewport()?.GuiReleaseFocus();
            }
            _panel.Visible = opening;
        });
    }

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

            var layer = new CanvasLayer
            {
                Name = "SpireMentorSettingsLayer",
                Layer = 102,
            };

            var panel = new PanelContainer
            {
                Name = "SpireMentorSettingsPanel",
                Position = new Vector2(16f, 68f),
                CustomMinimumSize = new Vector2(520f, 0f),
                MouseFilter = Control.MouseFilterEnum.Stop,
                Visible = false,
            };
            panel.AddThemeStyleboxOverride("panel", SimUiKit.PanelStyle());

            // Title stays put; everything below it scrolls. The rows used to
            // sit straight in the panel with no height cap, which on a 1080p
            // screen ended exactly at the bottom edge -- and below that on
            // anything shorter.
            var outer = new VBoxContainer
            {
                Name = "SettingsOuter",
            };
            outer.AddThemeConstantOverride("separation", 4);

            var title = new HBoxContainer();
            title.AddThemeConstantOverride("separation", 8);
            Label titleLabel = SimUiKit.SmallLabel("Spire Mentor settings", Colors.White, 16);
            titleLabel.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
            title.AddChild(titleLabel);
            title.AddChild(SimUiKit.MakeButton("✕", "Close", Toggle));
            outer.AddChild(title);

            var scroll = new ScrollContainer
            {
                Name = "SettingsScroll",
                HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
            };
            outer.AddChild(scroll);

            var rows = new VBoxContainer
            {
                Name = "SettingsRows",
                SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            };
            rows.AddThemeConstantOverride("separation", 4);
            scroll.AddChild(rows);

            // Advisor/workbench plumbing: collapsed, so the everyday rows are
            // what a player sees.
            var advanced = new VBoxContainer
            {
                Name = "SettingsAdvanced",
                Visible = false,
            };
            advanced.AddThemeConstantOverride("separation", 4);
            Button advancedToggle = SimUiKit.MakeButton(AdvancedLabel(false),
                "Advisor command, workbench URL and other backend settings", ToggleAdvanced);
            advancedToggle.Alignment = HorizontalAlignment.Left;

            if (SpireMentorModConfig.BuildLiveRowsInto(rows, advanced))
            {
                rows.AddChild(advancedToggle);
                rows.AddChild(advanced);
            }
            else
            {
                // The NConfig* controls lean on the game's scene paths; if
                // that ever breaks, say where the working page is instead of
                // rendering an empty dialog.
                rows.AddChild(SimUiKit.SmallLabel(
                    "Settings UI unavailable (see godot.log).\n" +
                    "Use Options → Mod Configuration at the main menu instead.",
                    new Color(1f, 0.75f, 0.6f)));
            }

            panel.AddChild(outer);
            layer.AddChild(panel);
            tree.Root.AddChildSafely(layer);

            SpireMentorModConfig.EnsureLivePersistHook();

            _layer = layer;
            _panel = panel;
            _title = title;
            _scroll = scroll;
            _rows = rows;
            _advanced = advanced;
            _advancedToggle = advancedToggle;
            Logger.Log("[SimSettingsPanel] Settings panel created");
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimSettingsPanel] Failed to create panel: {ex.Message}");
            _layer = null;
            _panel = null;
            return false;
        }
    }

    private static string AdvancedLabel(bool open) => open ? "▾ Advanced" : "▸ Advanced";

    private void ToggleAdvanced()
    {
        if (_advanced == null || !GodotObject.IsInstanceValid(_advanced))
        {
            return;
        }
        _advanced.Visible = !_advanced.Visible;
        if (_advancedToggle != null && GodotObject.IsInstanceValid(_advancedToggle))
        {
            _advancedToggle.Text = AdvancedLabel(_advanced.Visible);
        }
        FitToScreenSoon();
    }

    /// <summary>
    /// Containers recompute their minimum size a frame late after a
    /// visibility change or a fresh build, so measure on the next idle.
    /// </summary>
    private void FitToScreenSoon() => Callable.From(() => Guarded(FitToScreen)).CallDeferred();

    /// <summary>
    /// Size the scroll area to its content, capped so the panel ends
    /// <see cref="BottomMargin"/> above the bottom of the screen. Past the
    /// cap the rows scroll.
    /// </summary>
    private void FitToScreen()
    {
        if (_panel == null || !GodotObject.IsInstanceValid(_panel) ||
            _scroll == null || _rows == null || _title == null)
        {
            return;
        }
        float content = _rows.GetCombinedMinimumSize().Y;
        StyleBox? style = _panel.GetThemeStylebox("panel");
        float chrome = (style?.GetMinimumSize().Y ?? 0f) + _title.GetCombinedMinimumSize().Y + 4f;
        float room = _panel.GetViewportRect().Size.Y - _panel.Position.Y - BottomMargin - chrome;
        float height = Mathf.Min(content, Mathf.Max(room, MinScrollHeight));
        _scroll.CustomMinimumSize = new Vector2(0f, height);
        // A PanelContainer only grows on its own; collapsing Advanced must
        // shrink it back to its new minimum.
        _panel.ResetSize();
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
            Logger.Log($"[SimSettingsPanel] Dispatch failed: {ex.Message}");
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
            Logger.Log($"[SimSettingsPanel] Ignoring UI error (never breaks gameplay): {ex.Message}");
        }
    }
}
