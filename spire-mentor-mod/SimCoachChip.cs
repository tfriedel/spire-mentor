// SimCoachChip.cs -- the coach chip: a small, non-modal note on the HUD that
// says, AFTER the fact, what the AI would have done on a turn or a choice
// (PROTOCOL.md section 5, v5 "coach"; the words come from CoachChipText).
//
// Shown in both advice styles: Hints also gets after-the-fact feedback.
//
// Behaviour:
//   * one chip at a time; a new chip replaces an unpinned one, a pinned chip
//     stays and the new one stacks below it (max 2) -- except that
//     autobattle's "since your last stop" chip (Autobattle.cs) only replaces
//     its own kind, so it shows next to the verdict on the last pick, and
//     stays until SimAdvisor withdraws it;
//   * click the chip to pin it (gold border), click again to unpin; the small
//     x closes it;
//   * a chip stays long enough to read -- a turn chip 20 s (it outlives the
//     enemy turn and carries into your next one), a choice chip 25 s -- or
//     until the next chip arrives, unless pinned; it then fades (0.3 s on
//     Modulate:a, instant with "Reduce combat effects"); combat end and a
//     rewind clear every unpinned chip; a `withdraw` hides the chips it
//     names, pinned or not.
//
// Input-safety contract (SimUiKit.cs): the stack container ignores the mouse,
// the chip body stops it (that is the click-to-pin), nothing takes focus --
// it never steals the hand or the map. Main thread only; SimAdvisor marshals.

using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace SpireMentor;

internal sealed class SimCoachChip
{
    public static readonly SimCoachChip Instance = new();

    private const int MaxChips = 2;
    private const double TurnChipSeconds = 20.0;
    private const double ChoiceChipSeconds = 25.0;
    private const double FadeSeconds = 0.3;
    private const float ChipWidth = 430f;

    private sealed class Chip
    {
        public string Id = "";
        public string Kind = "";
        public PanelContainer Panel = null!;
        public StyleBoxFlat Style = null!;
        public bool Pinned;
        public bool Leaving;
        public ulong ShownAtMs;
        /// <summary>Seconds on screen before it fades, unless pinned.</summary>
        public double Seconds;
    }

    private readonly List<Chip> _chips = new();
    private CanvasLayer? _layer;
    private VBoxContainer? _stack;

    /// <summary>How long a chip of this kind stays up unpinned (FrameLink's
    /// welcome replays only chips still inside it).</summary>
    internal static double ChipSeconds(string kind) => kind switch
    {
        "turn" => TurnChipSeconds,
        Autobattle.ChipKind => double.PositiveInfinity,
        _ => ChoiceChipSeconds,
    };

    private SimCoachChip()
    {
        SimRuntime.Changed += () => Callable.From(RefreshVisibility).CallDeferred();
    }

    // ------------------------------------------------------------------
    // Entry points (main thread)
    // ------------------------------------------------------------------

    public void Show(ChipText text, bool inCombat)
    {
        Guard(() =>
        {
            // Framed: the frame gets every chip (it runs its own timers and
            // pins); panels outside the game: it is the only place they show.
            FrameLink.Chip(text, inCombat);
            if (FrameLink.UiOutside)
            {
                return;
            }
            if (!EnsureLayer())
            {
                return;
            }
            // The same record again (a re-send) replaces itself in place.
            foreach (Chip same in _chips.Where(c => c.Id.Length > 0 && c.Id == text.Id).ToList())
            {
                Remove(same, fade: false);
            }
            // A new chip replaces the unpinned one of its own group: the
            // coach's verdict on the last pick and autobattle's "since your
            // last stop" for this one (Autobattle.cs) show side by side.
            bool autobattle = text.Kind == Autobattle.ChipKind;
            foreach (Chip unpinned in _chips.Where(c => !c.Pinned && (c.Kind == Autobattle.ChipKind) == autobattle).ToList())
            {
                Remove(unpinned, fade: false);
            }
            while (_chips.Count >= MaxChips)
            {
                Remove(_chips[0], fade: false);
            }

            var chip = new Chip
            {
                Id = text.Id,
                Kind = text.Kind,
                ShownAtMs = Time.GetTicksMsec(),
                // The enemy turn alone is too short to read a chip in, so a
                // turn chip is not tied to the next turn start.
                // An autobattle chip stays up while the player decides;
                // SimAdvisor withdraws it once the AI plays on.
                Seconds = ChipSeconds(text.Kind),
            };
            Build(chip, text);
            _chips.Add(chip);
            _stack!.AddChildSafely(chip.Panel);
            RefreshVisibility();
        });
    }

    /// <summary>The panels moved out of the game (FrameLink): the in-game
    /// chips go (the frame already has them). Main thread.</summary>
    public void OnFrameUiModeChanged()
    {
        if (!FrameLink.UiOutside)
        {
            return;
        }
        foreach (Chip c in _chips.ToList())
        {
            Remove(c, fade: false);
        }
    }

    /// <summary>Hide every chip showing one of <paramref name="ids"/> --
    /// pinned too: a withdrawn grade is gone.</summary>
    public void Withdraw(IReadOnlyCollection<string> ids)
    {
        Guard(() =>
        {
            FrameLink.ChipWithdraw(ids);
            foreach (Chip c in _chips.Where(c => ids.Contains(c.Id)).ToList())
            {
                Remove(c, fade: true);
            }
        });
    }

    /// <summary>Combat ended or the run rewound: every unpinned chip goes.</summary>
    public void ClearUnpinned() =>
        Guard(() =>
        {
            FrameLink.ChipClearUnpinned();
            foreach (Chip c in _chips.Where(c => !c.Pinned).ToList())
            {
                Remove(c, fade: false);
            }
        });

    // ------------------------------------------------------------------

    private void Build(Chip chip, ChipText text)
    {
        StyleBoxFlat style = SimUiKit.PanelStyle();
        style.BorderColor = new Color(1f, 0.85f, 0.3f);
        chip.Style = style;

        var panel = new PanelContainer
        {
            Name = "SpireMentorCoachChip",
            CustomMinimumSize = new Vector2(ChipWidth, 0f),
            // The body takes the click (pin); everything inside ignores it.
            MouseFilter = Control.MouseFilterEnum.Stop,
            FocusMode = Control.FocusModeEnum.None,
            MouseDefaultCursorShape = Control.CursorShape.PointingHand,
        };
        panel.AddThemeStyleboxOverride("panel", style);
        panel.Connect(Control.SignalName.GuiInput, Callable.From<InputEvent>(ev => OnChipInput(chip, ev)));

        var rows = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        rows.AddThemeConstantOverride("separation", 2);

        var header = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        Label title = SimUiKit.SmallLabel(text.Title, ToneColor(text.Tone), 15);
        title.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        title.ClipText = true;
        header.AddChild(title);
        if (text.LogIdx is int step)
        {
            // A button stops its own click, so it never toggles the pin.
            string? run = text.LogName;
            Button open = SimUiKit.MakeButton("↗", "", () => Guard(() => SimWorkbenchLink.Open(run, step)));
            open.AddThemeFontSizeOverride("font_size", 12);
            header.AddChild(open);
        }
        Button close = SimUiKit.MakeButton("×", "", () => Guard(() => Remove(chip, fade: false)));
        close.AddThemeFontSizeOverride("font_size", 12);
        header.AddChild(close);
        rows.AddChild(header);

        foreach (string line in text.Lines)
        {
            Label l = SimUiKit.SmallLabel(line, new Color(0.88f, 0.9f, 0.95f), 13);
            l.CustomMinimumSize = new Vector2(ChipWidth - 24f, 0f);
            l.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            rows.AddChild(l);
        }
        panel.AddChild(rows);
        chip.Panel = panel;
        ApplyPinned(chip);
    }

    private void OnChipInput(Chip chip, InputEvent ev)
    {
        Guard(() =>
        {
            if (ev is InputEventMouseButton { ButtonIndex: MouseButton.Left, Pressed: false } && !chip.Leaving)
            {
                chip.Pinned = !chip.Pinned;
                ApplyPinned(chip);
            }
        });
    }

    private static void ApplyPinned(Chip chip)
    {
        int w = chip.Pinned ? 2 : 0;
        chip.Style.BorderWidthLeft = w;
        chip.Style.BorderWidthRight = w;
        chip.Style.BorderWidthTop = w;
        chip.Style.BorderWidthBottom = w;
    }

    private void Remove(Chip chip, bool fade)
    {
        if (chip.Leaving && fade)
        {
            return;
        }
        chip.Leaving = true;
        if (fade && !SpireMentorModConfig.ReduceEffects &&
            GodotObject.IsInstanceValid(chip.Panel) && chip.Panel.IsInsideTree())
        {
            Tween tween = chip.Panel.CreateTween();
            tween.TweenProperty(chip.Panel, "modulate:a", 0.0f, FadeSeconds);
            tween.TweenCallback(Callable.From(() => Guard(() => Drop(chip))));
            return;
        }
        Drop(chip);
    }

    private void Drop(Chip chip)
    {
        _chips.Remove(chip);
        if (GodotObject.IsInstanceValid(chip.Panel))
        {
            chip.Panel.QueueFreeSafely();
        }
    }

    private void Tick()
    {
        Guard(() =>
        {
            ulong now = Time.GetTicksMsec();
            foreach (Chip c in _chips.Where(c => !c.Pinned && !c.Leaving).ToList())
            {
                if (!double.IsPositiveInfinity(c.Seconds) && now - c.ShownAtMs > (ulong)(c.Seconds * 1000))
                {
                    Remove(c, fade: true);
                }
            }
            if (_stack != null && GodotObject.IsInstanceValid(_stack))
            {
                Vector2 slot = SimHud.Instance.ChipSlot();
                if (_stack.Position.DistanceTo(slot) > 0.5f)
                {
                    _stack.Position = slot;
                }
            }
        });
    }

    /// <summary>Height of the chips on screen (0 when none or hidden), so
    /// SimHud can keep room for them under the advice panel. Main thread.</summary>
    public float StackHeight()
    {
        if (_stack == null || !GodotObject.IsInstanceValid(_stack) ||
            !_stack.IsVisibleInTree() || _stack.GetChildCount() == 0)
        {
            return 0f;
        }
        return _stack.GetCombinedMinimumSize().Y;
    }

    private void RefreshVisibility()
    {
        Guard(() =>
        {
            if (_layer != null && GodotObject.IsInstanceValid(_layer))
            {
                _layer.Visible = SpireMentorModConfig.ShowHud && SimRuntime.Mode == SimMode.Advisor &&
                                 !SimUiHide.Hidden;
            }
        });
    }

    private bool EnsureLayer()
    {
        if (_layer != null && GodotObject.IsInstanceValid(_layer))
        {
            return true;
        }
        if (Engine.GetMainLoop() is not SceneTree tree)
        {
            return false;
        }
        // The HUD's layer number (99), as its own CanvasLayer so the chip's
        // lifetime never tangles with the advice panel's rebuilds.
        var layer = new CanvasLayer { Name = "SpireMentorCoachChipLayer", Layer = 99 };
        var stack = new VBoxContainer
        {
            Name = "CoachChips",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Position = SimHud.Instance.ChipSlot(),
        };
        stack.AddThemeConstantOverride("separation", 6);
        layer.AddChild(stack);
        var ticker = new Godot.Timer { Name = "CoachChipTicker", WaitTime = 0.25, Autostart = true, OneShot = false };
        ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(Tick));
        layer.AddChild(ticker);
        tree.Root.AddChildSafely(layer);
        _layer = layer;
        _stack = stack;
        Logger.Log("[SimCoachChip] chip layer created");
        return true;
    }

    private static Color ToneColor(ChipTone tone) => tone switch
    {
        ChipTone.Good => new Color(0.55f, 0.9f, 0.55f),
        ChipTone.Slip => new Color(1f, 0.78f, 0.35f),
        ChipTone.Blunder => new Color(1f, 0.45f, 0.4f),
        _ => new Color(0.6f, 0.82f, 1f),
    };

    /// <summary>The mod log, for SimCoachPanel (which keeps game types out of
    /// its own file).</summary>
    internal static void Log(string message) => Logger.Log(message);

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimCoachChip] Ignoring error (never breaks gameplay): {ex.Message}");
        }
    }
}
