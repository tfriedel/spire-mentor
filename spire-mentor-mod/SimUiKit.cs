// SimUiKit.cs -- shared building blocks for the mod's hand-built UI.
//
// SpireMentor ships no .pck (see SpireMentor.csproj), so there are no scenes,
// no theme resources and no loc table: every control is composed in code
// and styled with StyleBoxFlat + AddTheme*Override. This file keeps the
// look consistent between the HUD, the toolbar and the settings panel.
//
// Input-safety contract (the "panel ate combat clicks" regression class):
//   * containers, labels, gaps: MouseFilter.Ignore -- transparent to the game
//   * buttons: MouseFilter.Stop but FocusMode.None. A FOCUSED button would
//     swallow Space/Enter via ui_accept and join arrow-key navigation;
//     hotkeys dispatch from NHotkeyManager._UnhandledInput, so a click
//     consumed by a button never reaches them, which is exactly right.
//   * no native tooltips: TooltipText has zero uses in the v0.107.1 decomp,
//     so its rendering (default-theme popup, embedded-window behavior) is
//     unproven in this game. Buttons wire MouseEntered/MouseExited into an
//     owned hover-hint label instead (see MakeButton's hintSink).

using System;
using System.Collections.Generic;
using Godot;

namespace SpireMentor;

internal static class SimUiKit
{
    /// <summary>The mod's standard translucent panel (HUD chip/panel look).</summary>
    public static StyleBoxFlat PanelStyle() => new()
    {
        BgColor = new Color(0f, 0f, 0f, 0.65f),
        CornerRadiusTopLeft = 6,
        CornerRadiusTopRight = 6,
        CornerRadiusBottomLeft = 6,
        CornerRadiusBottomRight = 6,
        ContentMarginLeft = 10f,
        ContentMarginRight = 10f,
        ContentMarginTop = 6f,
        ContentMarginBottom = 6f,
    };

    private static StyleBoxFlat ButtonStyle(Color bg) => new()
    {
        BgColor = bg,
        CornerRadiusTopLeft = 4,
        CornerRadiusTopRight = 4,
        CornerRadiusBottomLeft = 4,
        CornerRadiusBottomRight = 4,
        ContentMarginLeft = 8f,
        ContentMarginRight = 8f,
        ContentMarginTop = 3f,
        ContentMarginBottom = 3f,
    };

    /// <summary>
    /// A toolbar-sized button following the input-safety contract. The click
    /// handler is exception-guarded: UI failure must never break gameplay.
    /// <paramref name="hintSink"/> receives the hint on hover and null on
    /// leave (the owned-hover-label replacement for native tooltips).
    /// </summary>
    public static Button MakeButton(string text, string hint, Action onPressed,
        Action<string?>? hintSink = null)
    {
        var button = new Button { Text = text };
        StyleToolbarButton(button);
        button.Connect(BaseButton.SignalName.Pressed, Callable.From(() =>
        {
            try
            {
                onPressed();
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimUiKit] Button '{text}' handler threw: {ex.Message}");
            }
        }));
        WireHint(button, hint, hintSink);
        return button;
    }

    /// <summary>One entry of a <see cref="MakeMenuButton"/> dropdown.</summary>
    public readonly record struct MenuEntry(string Label, string Hint, Action OnPressed);

    /// <summary>
    /// A toolbar-sized dropdown following the same input-safety contract as
    /// <see cref="MakeButton"/>: the trigger is the only Stop region; the
    /// popup is a Godot <see cref="PopupMenu"/> (its own window, closes on a
    /// click outside or Escape, so it never sits over combat unasked). Entry
    /// handlers are exception-guarded. The trigger's hover hint lists the
    /// entries so the hint label stays the one tooltip surface.
    /// </summary>
    public static MenuButton MakeMenuButton(string text, string hint,
        IReadOnlyList<MenuEntry> entries, Action<string?>? hintSink = null)
    {
        var button = new MenuButton { Text = text + " ▾", Flat = false };
        StyleToolbarButton(button);
        PopupMenu popup = button.GetPopup();
        for (int i = 0; i < entries.Count; i++)
        {
            popup.AddItem(entries[i].Label, i);
        }
        popup.AddThemeFontSizeOverride("font_size", 14);
        popup.Connect(PopupMenu.SignalName.IdPressed, Callable.From((long id) =>
        {
            if (id < 0 || id >= entries.Count)
            {
                return;
            }
            try
            {
                entries[(int)id].OnPressed();
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimUiKit] Menu '{text}' entry '{entries[(int)id].Label}' threw: {ex.Message}");
            }
        }));
        WireHint(button, hint, hintSink);
        return button;
    }

    private static void WireHint(Control control, string hint, Action<string?>? hintSink)
    {
        if (hintSink != null)
        {
            control.Connect(Control.SignalName.MouseEntered, Callable.From(() => hintSink(hint)));
            control.Connect(Control.SignalName.MouseExited, Callable.From(() => hintSink(null)));
        }
    }

    /// <summary>
    /// Paint a button as lit -- the chosen segment of a switch, a state that
    /// is on -- in <paramref name="accent"/>, or (null) back to the plain look.
    /// Only the colours change: the input-safety flags stay as MakeButton set
    /// them.
    /// </summary>
    public static void SetLit(Button button, Color? accent)
    {
        if (accent is not { } a)
        {
            ApplyButtonColors(button);
            return;
        }
        button.AddThemeStyleboxOverride("normal", ButtonStyle(new Color(a.R * 0.55f, a.G * 0.55f, a.B * 0.55f, 0.95f), a));
        button.AddThemeStyleboxOverride("hover", ButtonStyle(new Color(a.R * 0.70f, a.G * 0.70f, a.B * 0.70f, 0.98f), a));
        button.AddThemeStyleboxOverride("pressed", ButtonStyle(new Color(a.R * 0.45f, a.G * 0.45f, a.B * 0.45f, 0.98f), a));
    }

    /// <summary>The Spire Mentor badge's look: dark, with a thick left edge
    /// and a thin outline in the mode's colour, so the mode reads at a glance
    /// even out of the corner of an eye.</summary>
    public static StyleBoxFlat BadgeStyle(Color accent, bool hover)
    {
        StyleBoxFlat style = ButtonStyle(hover ? new Color(0.20f, 0.23f, 0.30f, 0.96f) : new Color(0.08f, 0.09f, 0.12f, 0.92f), accent);
        style.BorderWidthLeft = 6;
        style.ContentMarginLeft = 10f;
        style.ContentMarginTop = 4f;
        style.ContentMarginBottom = 4f;
        return style;
    }

    private static StyleBoxFlat ButtonStyle(Color bg, Color border)
    {
        StyleBoxFlat style = ButtonStyle(bg);
        style.BorderColor = border;
        style.BorderWidthLeft = 1;
        style.BorderWidthRight = 1;
        style.BorderWidthTop = 1;
        style.BorderWidthBottom = 1;
        return style;
    }

    /// <summary>A small grey section caption (click-through).</summary>
    public static Label Caption(string text) =>
        SmallLabel(text, new Color(0.62f, 0.66f, 0.74f), 13);

    /// <summary>The strip's button look + the input-safety flags, shared by
    /// plain buttons and menu triggers.</summary>
    private static void StyleToolbarButton(Button button)
    {
        button.FocusMode = Control.FocusModeEnum.None;
        button.MouseFilter = Control.MouseFilterEnum.Stop;
        button.MouseDefaultCursorShape = Control.CursorShape.PointingHand;
        ApplyButtonColors(button);
        StyleToolbarText(button);
    }

    private static void ApplyButtonColors(Button button)
    {
        button.AddThemeStyleboxOverride("normal", ButtonStyle(new Color(0.15f, 0.17f, 0.22f, 0.9f)));
        button.AddThemeStyleboxOverride("hover", ButtonStyle(new Color(0.25f, 0.30f, 0.40f, 0.95f)));
        button.AddThemeStyleboxOverride("pressed", ButtonStyle(new Color(0.10f, 0.12f, 0.16f, 0.95f)));
    }

    private static void StyleToolbarText(Button button)
    {
        button.AddThemeStyleboxOverride("disabled", ButtonStyle(new Color(0.12f, 0.12f, 0.13f, 0.6f)));
        button.AddThemeStyleboxOverride("focus", new StyleBoxEmpty());
        button.AddThemeColorOverride("font_color", Colors.White);
        button.AddThemeColorOverride("font_hover_color", Colors.White);
        button.AddThemeColorOverride("font_pressed_color", new Color(0.85f, 0.85f, 0.85f));
        button.AddThemeColorOverride("font_disabled_color", new Color(0.6f, 0.6f, 0.6f));
        button.AddThemeColorOverride("font_outline_color", Colors.Black);
        button.AddThemeConstantOverride("outline_size", 2);
        button.AddThemeFontSizeOverride("font_size", 14);
    }

    /// <summary>Small outlined label, click-through (HUD idiom).</summary>
    public static Label SmallLabel(string text, Color color, int fontSize = 14)
    {
        var label = new Label
        {
            Text = text,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", Colors.Black);
        label.AddThemeConstantOverride("outline_size", 3);
        label.AddThemeFontSizeOverride("font_size", fontSize);
        return label;
    }
}
