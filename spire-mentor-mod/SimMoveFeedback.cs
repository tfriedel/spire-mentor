// SimMoveFeedback.cs -- a word over each card you play, and a fight score.
//
// The moment you play a card, a short verdict floats up from the hand: the
// advice for the position you played it from, graded the same way the ranked
// list words its rows (AdviceMarks.Verdict): "✓ Best", "✓ Good", "Okay",
// "?! Weak", "?? Mistake". When the fight ends, one line sums it up: how many
// of your moves were the AI's pick or as good as it. Chess.com's move badges,
// for a card game: the coach chip (SimCoachChip) says WHY afterwards; this
// says HOW IT WENT, at once, without reading.
//
// The grade book is the advice's options keyed by the hand card they play
// (its CardModel, which outlives hand-index shifts) and target. It is built
// when an advice arrives for a fight position -- in both styles, Coach
// included, where nothing else of that advice is drawn -- and is used ONCE: a
// fast second card plays from a position no advice was computed for, and
// grading it against the first position's advice would be wrong, so it is
// not graded at all.
//
// Visual only, MouseFilter.Ignore throughout; respects "Reduce combat
// effects" (no drift, a plain fade). Main thread only; callers marshal.

using System;
using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;

namespace SpireMentor;

internal sealed class SimMoveFeedback
{
    public static readonly SimMoveFeedback Instance = new();

    private SimMoveFeedback()
    {
    }

    private Dictionary<(CardModel Card, int Target), MarkOption>? _book;
    private double? _bookBestQ;
    private bool _bookUsed;

    private int _graded;
    private int _onLine;
    private int _mistakes;

    private CanvasLayer? _layer;

    /// <summary>A fresh fight advice: the grade book for the next play.</summary>
    public void OnAdvice(SimAdvice advice, Player? player)
    {
        Guard(() =>
        {
            _book = null;
            _bookUsed = false;
            if (player == null || !CombatManager.Instance.IsInProgress)
            {
                return;
            }
            CardPile hand = PileType.Hand.GetPile(player);
            var book = new Dictionary<(CardModel, int), MarkOption>();
            foreach (SimAdviceOption o in advice.Options)
            {
                if (o.CardIndex is not int ci || ci < 0 || ci >= hand.Cards.Count || o.Action is "potion" or "end_turn")
                {
                    continue;
                }
                book[(hand.Cards[ci], o.TargetIndex ?? -1)] = o.ToMark();
            }
            if (book.Count > 0)
            {
                _book = book;
                _bookBestQ = advice.BestQ;
            }
        });
    }

    /// <summary>The human played <paramref name="card"/> at
    /// <paramref name="targetIndex"/> (the combat's Enemies frame).</summary>
    public void OnHumanPlay(CardModel? card, int? targetIndex)
    {
        Guard(() =>
        {
            if (card == null || _book == null || _bookUsed || !SpireMentorModConfig.ShowCardLabels)
            {
                return;
            }
            _bookUsed = true;
            if (!_book.TryGetValue((card, targetIndex ?? -1), out MarkOption? o) &&
                !_book.TryGetValue((card, -1), out o))
            {
                // Same card, a target the search did not rank: grade the card's
                // own best line would flatter it; leave it ungraded.
                return;
            }
            string verdict = AdviceMarks.Verdict(o, _bookBestQ);
            if (verdict.Length == 0)
            {
                return;
            }
            _graded++;
            if (verdict is "Best" or "Also good")
            {
                _onLine++;
            }
            else if (verdict is "Mistake")
            {
                _mistakes++;
            }
            (string text, Color color) = Badge(verdict);
            Float(text, color, 26, 1.3, yShare: 0.60f);
        });
    }

    /// <summary>The fight is over: one line for how it went, then reset.</summary>
    public void OnCombatEnded()
    {
        Guard(() =>
        {
            _book = null;
            if (_graded >= 3 && SpireMentorModConfig.ShowCardLabels)
            {
                int pct = (int)Math.Round(100.0 * _onLine / _graded);
                string text = $"Fight accuracy {pct}%  ·  {_onLine} of {_graded} moves on the AI's line" +
                              (_mistakes > 0 ? $"  ·  {_mistakes} mistake{(_mistakes == 1 ? "" : "s")}" : "");
                Color color = pct >= 80 ? new Color(1f, 0.84f, 0.35f)
                    : pct >= 55 ? new Color(0.85f, 0.88f, 0.95f)
                    : new Color(0.98f, 0.62f, 0.4f);
                Float(text, color, 24, 3.6, yShare: 0.30f);
            }
            _graded = 0;
            _onLine = 0;
            _mistakes = 0;
        });
    }

    private static (string Text, Color Color) Badge(string verdict) => verdict switch
    {
        "Best" => ("✓ Best", new Color(1f, 0.84f, 0.3f)),
        "Also good" => ("✓ Good", new Color(0.55f, 0.92f, 0.6f)),
        "Okay" => ("Okay", new Color(0.88f, 0.88f, 0.8f)),
        "Weak" => ("?! Weak", new Color(0.98f, 0.66f, 0.3f)),
        "Mistake" => ("?? Mistake", new Color(0.98f, 0.4f, 0.34f)),
        _ => ("?! Unusual", new Color(0.8f, 0.8f, 0.85f)),
    };

    private void Float(string text, Color color, int size, double seconds, float yShare)
    {
        if (!EnsureLayer() || SimUiHide.Hidden || !SpireMentorModConfig.ShowHud)
        {
            return;
        }
        Vector2 viewport = _layer!.GetViewport()?.GetVisibleRect().Size ?? new Vector2(1920, 1080);
        var label = new Label
        {
            Text = text,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Size = new Vector2(viewport.X, size * 2f),
            Position = new Vector2(0, viewport.Y * yShare),
        };
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.9f));
        label.AddThemeConstantOverride("outline_size", 8);
        label.AddThemeFontSizeOverride("font_size", size);
        _layer.AddChildSafely(label);

        Tween tween = label.CreateTween();
        if (!SpireMentorModConfig.ReduceEffects)
        {
            label.Scale = new Vector2(1f, 1f);
            tween.Parallel().TweenProperty(label, "position:y", label.Position.Y - 46f, seconds)
                .SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
        }
        tween.Parallel().TweenProperty(label, "modulate:a", 0f, seconds * 0.45).SetDelay(seconds * 0.55);
        tween.TweenCallback(Callable.From(() =>
        {
            if (GodotObject.IsInstanceValid(label))
            {
                label.QueueFreeSafely();
            }
        }));
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
        _layer = new CanvasLayer { Name = "SpireMentorMoveFeedback", Layer = 99 };
        tree.Root.AddChildSafely(_layer);
        return true;
    }

    private static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimMoveFeedback] {ex.Message}");
        }
    }
}
