// SimHighlights.cs -- marks on the things the advice says to click, drawn
// with the game's OWN visuals wherever the game has one.
//
// AdviceMarks decides what is marked (game-free); this draws it:
//
//   * a hand card (or a reward / select card) gets a copy of the game's own
//     card-outline glow (NCard's %Highlight, an outline shader that follows
//     the card's shape) in our colour and wider than the game's cyan
//     "playable" glow, inside the card's own node tree -- so it hugs the
//     card, sits behind its face, and the cards on top of it cover it the way
//     they cover the card. Cards in the plan are also pulled ~20% up out of
//     the fan (the game's own layout target, nudged), so the line stands out
//     before any number is read;
//   * the End Turn button gets a copy of its own glow texture
//     (Visuals/Glow, the one the game shows when nothing is left to play);
//   * the target of the move gets a copy of the game's own targeting
//     brackets (NCreature's %SelectionReticle, which the game sizes to the
//     creature) in red (an ally: green). It stays up: the plan's first
//     targeted step by default, the hovered card's target while you hover
//     or aim one;
//   * anything else (potion, event / rest / shop button, relic) a soft
//     rounded glow drawn over it on our own layer.
//
// Every mark carries a round badge with its step in the turn's plan (1, 2,
// 3...): above a card's title (its top-left corner is the energy cost), on a
// button's corner. Placed every rendered frame from SceneTree.ProcessFrame
// (no source generators, so no _Process); card nodes are re-resolved from
// their MODEL each frame because NCard nodes are pooled (SimHud.CardLabelAnchor).
//
// Reported on the first live look (2026-11-07): flat boxes on a layer above
// every card ran over the neighbouring cards, the End Turn box sat crooked on
// the button, the enemy box did not fit the enemy, and the target vanished
// when the mouse moved. Hence the game's own shapes, inside the game's own
// node trees.
//
// The copies are made without their scripts (plain TextureRect / Control),
// so the game never drives them; ours are freed on Clear and whenever the
// pooled node they sit in moves to another card.
//
// Input-safety contract (SimUiKit.cs): every node here is MouseFilter.Ignore
// and takes no focus. Never-break-gameplay: every entry point and the frame
// hook swallow and log. Main thread only; SimHud marshals.

using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Potions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;

namespace SpireMentor;

internal sealed class SimHighlights
{
    public static readonly SimHighlights Instance = new();

    private SimHighlights()
    {
    }

    // The pick: a warm white-gold, apart from the game's own saturated gold
    // (a card condition, NCardHighlight.gold) and its cyan "playable" glow.
    private static readonly Color PickGold = new(1f, 0.87f, 0.45f, 0.98f);
    private static readonly Color StepGold = new(1f, 0.93f, 0.72f, 0.8f);
    private static readonly Color AltBlue = new(0.72f, 0.82f, 1f, 0.6f);
    private static readonly Color TargetRed = new(1f, 0.35f, 0.28f);
    private static readonly Color AllyGreen = new(0.45f, 0.95f, 0.55f);

    /// <summary>The game's own "playable" outline width (NCardHighlight.AnimShow).</summary>
    private const float GameGlowWidth = 0.075f;

    /// <summary>How far a planned hand card is pulled out of the fan, as a
    /// share of its height (the user's "pulled 20% or so").</summary>
    private const float LiftShare = 0.2f;

    private sealed class Ring
    {
        public Mark Mark = null!;

        /// <summary>Our overlay frame: the drawn ring for a generic node, an
        /// invisible anchor carrying the badge for a card / End Turn.</summary>
        public Panel Frame = null!;
        public Panel? Badge;
        public Tween? Pulse;

        /// <summary>A hand card's model (its node is pooled).</summary>
        public CardModel? Card;

        /// <summary>The node for every other kind, resolved once.</summary>
        public Control? Node;

        public float Pad;

        /// <summary>The copy of the game's glow (card outline or End Turn
        /// glow), and the node it was put into.</summary>
        public CanvasItem? Glow;
        public Node? GlowHost;

        /// <summary>The hand holder we pulled out, and where we put it.</summary>
        public NHandCardHolder? Lifted;
        public Vector2 LiftedTo;
        public Vector2 LiftedFrom;
    }

    private CanvasLayer? _layer;
    private readonly List<Ring> _rings = new();
    private bool _hooked;
    private ulong _frame;
    private readonly List<(NHandCardHolder Holder, Vector2 From, Vector2 To)> _pendingUnlift = new();

    // The target mark: a copy of the game's targeting brackets in the
    // creature's own tree, and our step badge beside it.
    private Control? _reticle;
    private Node? _reticleHost;
    private Label? _reticleBadge;

    /// <summary>The mark whose target stays marked without a hover.</summary>
    private Mark? _pinnedTarget;

    // ------------------------------------------------------------------
    // Entry points (main thread)
    // ------------------------------------------------------------------

    /// <summary>
    /// Draw <paramref name="marks"/> (AdviceMarks.Build) for the position on
    /// screen. Card marks index <paramref name="player"/>'s hand as the advice
    /// does; screen marks index <paramref name="decision"/>'s option nodes.
    /// <paramref name="showTarget"/> is kept for the peek's call; the target
    /// is always marked now.
    /// </summary>
    public void Show(IReadOnlyList<Mark> marks, Player? player, SimScreenDecision? decision, bool showTarget = false)
    {
        Guard(() =>
        {
            Clear();
            if (!SpireMentorModConfig.ShowCardLabels || !EnsureLayer())
            {
                return;
            }
            CardPile? hand = null;
            if (player != null && CombatManager.Instance.IsInProgress)
            {
                try { hand = PileType.Hand.GetPile(player); } catch (Exception) { hand = null; }
            }
            foreach (Mark m in marks)
            {
                Ring? ring = Build(m, hand, decision);
                if (ring != null)
                {
                    _rings.Add(ring);
                }
            }
            // The target that stays up: the plan's first step that aims at
            // something (often step 1; after a Defend, the next Strike's).
            _pinnedTarget = marks
                .Where(m => m.TargetIndex != null || m.AllyIndex != null)
                .OrderBy(m => m.Order ?? int.MaxValue)
                .FirstOrDefault(m => m.Tier != MarkTier.Alt);
            Hook();
            OnFrame();
        });
    }

    /// <summary>Remove every mark, put pulled cards back.</summary>
    public void Clear()
    {
        Guard(() =>
        {
            foreach (Ring r in _rings)
            {
                r.Pulse?.Kill();
                DropGlow(r);
                // Put back next frame, unless the new advice lifts it again
                // (every play re-advises: an instant drop + lift bounced).
                if (r.Lifted != null)
                {
                    _pendingUnlift.Add((r.Lifted, r.LiftedFrom, r.LiftedTo));
                }
                if (GodotObject.IsInstanceValid(r.Frame))
                {
                    r.Frame.QueueFreeSafely();
                }
            }
            _rings.Clear();
            _pinnedTarget = null;
            DropReticle();
        });
    }

    public bool Any => _rings.Count > 0;

    // ------------------------------------------------------------------
    // Building
    // ------------------------------------------------------------------

    private Ring? Build(Mark m, CardPile? hand, SimScreenDecision? decision)
    {
        var ring = new Ring { Mark = m };
        switch (m.Kind)
        {
            case MarkKind.Card:
                if (hand == null || m.Key < 0 || m.Key >= hand.Cards.Count)
                {
                    return null;
                }
                ring.Card = hand.Cards[m.Key];
                break;
            case MarkKind.Potion:
                ring.Node = PotionHolder(m.Key);
                ring.Pad = 3f;
                break;
            case MarkKind.EndTurn:
                ring.Node = EndTurnButton();
                break;
            case MarkKind.ScreenOption:
                ring.Node = decision != null && m.Key >= 0 && m.Key < decision.OptionNodes.Count
                    ? decision.OptionNodes[m.Key]
                    : null;
                ring.Pad = 6f;
                break;
            case MarkKind.Skip:
                ring.Node = decision?.SkipNode;
                ring.Pad = 6f;
                break;
        }
        if (ring.Card == null && (ring.Node == null || !GodotObject.IsInstanceValid(ring.Node)))
        {
            return null;
        }

        Color color = TierColor(m.Tier);
        var style = new StyleBoxFlat
        {
            // A soft rounded glow, not a hard box: thin line, wide halo.
            DrawCenter = false,
            BorderColor = color,
            ShadowColor = new Color(color.R, color.G, color.B, m.Tier == MarkTier.Alt ? 0.25f : 0.5f),
            ShadowSize = m.Tier == MarkTier.Best ? 22 : 12,
            AntiAliasing = true,
        };
        style.SetBorderWidthAll(m.Tier == MarkTier.Best ? 3 : 2);
        style.SetCornerRadiusAll(16);
        var frame = new Panel
        {
            Name = $"SpireMentorMark_{m.Kind}_{m.Key}",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            FocusMode = Control.FocusModeEnum.None,
            Visible = false,
        };
        frame.AddThemeStyleboxOverride("panel", style);
        _layer!.AddChildSafely(frame);
        ring.Frame = frame;

        if (m.Order is int order)
        {
            ring.Badge = MakeBadge(order, m.Tier);
            frame.AddChildSafely(ring.Badge);
        }
        if (m.Tier == MarkTier.Best && !SpireMentorModConfig.ReduceEffects)
        {
            // Breathes: the ring itself (self_modulate, so not the badge), or
            // the card glow's width once it exists (see EnsureCardGlow).
            Tween pulse = frame.CreateTween().SetLoops();
            pulse.TweenProperty(frame, "self_modulate:a", 0.55f, 0.8).SetTrans(Tween.TransitionType.Sine);
            pulse.TweenProperty(frame, "self_modulate:a", 1.0f, 0.8).SetTrans(Tween.TransitionType.Sine);
            ring.Pulse = pulse;
        }
        return ring;
    }

    private static Color TierColor(MarkTier tier) => tier switch
    {
        MarkTier.Best => PickGold,
        MarkTier.Good => StepGold,
        _ => AltBlue,
    };

    private const float BadgeSize = 36f;

    private static Panel MakeBadge(int order, MarkTier tier)
    {
        bool best = tier == MarkTier.Best;
        var style = new StyleBoxFlat
        {
            BgColor = best ? PickGold : new Color(0.08f, 0.07f, 0.05f, 0.92f),
            BorderColor = best ? new Color(0.35f, 0.22f, 0f) : StepGold,
            ShadowColor = new Color(0, 0, 0, 0.5f),
            ShadowSize = 4,
            AntiAliasing = true,
        };
        style.SetBorderWidthAll(2);
        style.SetCornerRadiusAll((int)(BadgeSize / 2));
        var badge = new Panel
        {
            Name = "Badge",
            Size = new Vector2(BadgeSize, BadgeSize),
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        badge.AddThemeStyleboxOverride("panel", style);
        var text = new Label
        {
            Text = order.ToString(System.Globalization.CultureInfo.InvariantCulture),
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Size = new Vector2(BadgeSize, BadgeSize),
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        text.AddThemeColorOverride("font_color", best ? new Color(0.12f, 0.07f, 0f) : StepGold);
        text.AddThemeFontSizeOverride("font_size", 21);
        badge.AddChild(text);
        return badge;
    }

    // ------------------------------------------------------------------
    // Per-frame placement
    // ------------------------------------------------------------------

    private void Hook()
    {
        if (_hooked || Engine.GetMainLoop() is not SceneTree tree)
        {
            return;
        }
        tree.ProcessFrame += () => Guard(OnFrame);
        _hooked = true;
    }

    private void OnFrame()
    {
        if (_layer == null || !GodotObject.IsInstanceValid(_layer))
        {
            return;
        }
        _frame++;
        bool visible = SpireMentorModConfig.ShowHud && SimRuntime.Mode != SimMode.Off && !SimUiHide.Hidden;
        _layer.Visible = visible;

        NPlayerHand? handUi = NPlayerHand.Instance;
        bool handAlive = handUi != null && GodotObject.IsInstanceValid(handUi);
        foreach (Ring r in _rings)
        {
            if (!GodotObject.IsInstanceValid(r.Frame))
            {
                continue;
            }
            try
            {
                PlaceRing(r, handAlive ? handUi : null, visible);
            }
            catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
            {
                // A node of the screen went away mid-frame: hide this one.
                r.Frame.Visible = false;
            }
        }
        FlushUnlifts();

        if (_rings.Count == 0)
        {
            DropReticle();
            return;
        }
        UpdateReticle(handAlive ? handUi : null, visible);
    }

    /// <summary>The cards the last advice pulled out and this one does not:
    /// back to where the game wants them.</summary>
    private void FlushUnlifts()
    {
        if (_pendingUnlift.Count == 0)
        {
            return;
        }
        foreach ((NHandCardHolder holder, Vector2 from, Vector2 to) in _pendingUnlift)
        {
            if (_rings.Any(r => r.Lifted == holder) || !GodotObject.IsInstanceValid(holder) || !holder.IsInsideTree())
            {
                continue;
            }
            if (holder.TargetPosition.DistanceTo(to) < 0.5f)
            {
                holder.SetTargetPosition(from);
            }
        }
        _pendingUnlift.Clear();
    }

    private void PlaceRing(Ring r, NPlayerHand? handUi, bool visible)
    {
        // A card: the hand's (by model) or a screen's (a card holder).
        NCard? card = null;
        if (r.Card != null)
        {
            card = handUi?.GetCard(r.Card);
        }
        else if (r.Node != null && GodotObject.IsInstanceValid(r.Node) && r.Node is NCardHolder holder &&
                 holder.CardNode is { } held && GodotObject.IsInstanceValid(held))
        {
            card = held;
        }
        else if (r.Node is NMerchantSlot slot && GodotObject.IsInstanceValid(slot))
        {
            // A shop card sits in the slot's %CardHolder, away from the
            // slot's own rect (the ring landed beside it, live 2026-11-07).
            card = UiHelper.FindFirst<NCard>(slot);
        }
        if (card != null && (!GodotObject.IsInstanceValid(card) || !card.IsVisibleInTree()))
        {
            card = null;
        }

        if (card != null)
        {
            EnsureCardGlow(r, card, visible);
            if (r.Card != null && handUi != null)
            {
                Lift(r, handUi);
            }
            Control face = card.GetNodeOrNull<Control>("%Frame") is { } f && f.IsVisibleInTree() && f.Size.X > 1f
                ? f
                : card;
            r.Frame.SelfModulate = new Color(1, 1, 1, 0); // the glow is the mark; the frame only carries the badge
            r.Frame.Visible = Place(r, face, cardBadge: true);
            return;
        }
        if (r.Card != null)
        {
            // The card left the hand (played, discarded): nothing to mark.
            DropGlow(r);
            r.Frame.Visible = false;
            return;
        }

        Control? node = r.Node != null && GodotObject.IsInstanceValid(r.Node) && r.Node.IsVisibleInTree() ? r.Node : null;
        if (node == null)
        {
            DropGlow(r);
            r.Frame.Visible = false;
            return;
        }

        if (node is NEndTurnButton button)
        {
            EnsureEndTurnGlow(r, button, visible);
            Control image = button.GetNodeOrNull<Control>("Visuals/Image") is { } img && img.Size.X > 1f ? img : button;
            r.Frame.SelfModulate = new Color(1, 1, 1, 0);
            r.Frame.Visible = Place(r, image, cardBadge: false);
            return;
        }

        r.Frame.Visible = Place(r, Drawn(node), cardBadge: false);
    }

    /// <summary>
    /// The part of <paramref name="node"/> the player actually sees: a bundle
    /// or other container whose own rect is empty -- its hitbox, or its first
    /// visible sized child.
    /// </summary>
    private static Control Drawn(Control node)
    {
        // A shop relic / potion: the slot's rect is the whole column; its
        // %Hitbox is the item.
        if (node is NMerchantSlot && node.GetNodeOrNull<Control>("%Hitbox") is { } item && item.Size.X > 1f)
        {
            return item;
        }
        if (node.Size.X > 1f && node.Size.Y > 1f)
        {
            return node;
        }
        if (node.GetNodeOrNull<Control>("%Hitbox") is { } hit && hit.Size.X > 1f)
        {
            return hit;
        }
        foreach (Node child in node.GetChildren())
        {
            if (child is Control c && c.IsVisibleInTree() && c.Size.X > 1f && c.Size.Y > 1f)
            {
                return c;
            }
        }
        return node;
    }

    /// <summary>Copy the node's canvas transform onto our frame (padded),
    /// and put the badge above a card's title or on a button's corner.</summary>
    private static bool Place(Ring r, Control node, bool cardBadge)
    {
        Vector2 size = node is NCard ? NCard.defaultSize : node.Size;
        if (size.X <= 1f || size.Y <= 1f)
        {
            return false;
        }
        Transform2D xf = node.GetGlobalTransformWithCanvas();
        Vector2 scale = xf.Scale;
        if (Math.Abs(scale.X) < 0.01f || Math.Abs(scale.Y) < 0.01f)
        {
            return false;
        }
        // An NCard's origin is its centre (SimHud.CardFaceRect); everything
        // else here is a top-left-origin Control.
        Vector2 local = node is NCard ? -size / 2f : Vector2.Zero;
        var pad = new Vector2(r.Pad / Math.Abs(scale.X), r.Pad / Math.Abs(scale.Y));
        r.Frame.Position = xf * (local - pad);
        r.Frame.Rotation = xf.Rotation;
        r.Frame.Scale = scale;
        r.Frame.Size = size + pad * 2f;
        if (r.Badge != null && GodotObject.IsInstanceValid(r.Badge))
        {
            // The same size on screen whatever the node's scale.
            var inv = new Vector2(1f / Math.Abs(scale.X), 1f / Math.Abs(scale.Y));
            r.Badge.Scale = inv;
            // A card's top-left corner is its energy cost: a "1" badge on a
            // "1" orb read as noise (live, 2026-11-07).
            r.Badge.Position = cardBadge
                ? new Vector2(r.Frame.Size.X / 2f - BadgeSize * inv.X / 2f, -BadgeSize * inv.Y * 0.85f)
                : new Vector2(-BadgeSize * inv.X * 0.4f, -BadgeSize * inv.Y * 0.4f);
        }
        return true;
    }

    // ------------------------------------------------------------------
    // The game's own glows, copied
    // ------------------------------------------------------------------

    /// <summary>
    /// A copy of the card's own outline glow (%Highlight), in our colour and
    /// wider than the game's cyan, put just behind the original in the card's
    /// tree. Re-made when the pooled card node changed hands.
    /// </summary>
    private static void EnsureCardGlow(Ring r, NCard card, bool visible)
    {
        if (r.Glow == null || !GodotObject.IsInstanceValid(r.Glow) || r.GlowHost != card)
        {
            DropGlow(r);
            NCardHighlight? original = card.CardHighlight;
            if (original == null || !GodotObject.IsInstanceValid(original) || original.GetParent() is not Node parent)
            {
                return;
            }
            if (original.Duplicate(0) is not TextureRect glow)
            {
                return;
            }
            glow.Name = "SpireMentorGlow";
            glow.UniqueNameInOwner = false;
            IgnoreMouse(glow);
            if (original.Material is ShaderMaterial shared && shared.Duplicate() is ShaderMaterial own)
            {
                glow.Material = own;
            }
            parent.AddChild(glow);
            parent.MoveChild(glow, original.GetIndex());
            r.Glow = glow;
            r.GlowHost = card;
            if (r.Pulse != null && r.Mark.Tier == MarkTier.Best && glow.Material is ShaderMaterial pulsing)
            {
                // Breathe the outline's width instead of a frame's alpha.
                r.Pulse.Kill();
                Tween t = glow.CreateTween().SetLoops();
                t.TweenMethod(Callable.From<float>(w => pulsing.SetShaderParameter("width", w)), 0.13f, 0.17f, 0.8)
                    .SetTrans(Tween.TransitionType.Sine);
                t.TweenMethod(Callable.From<float>(w => pulsing.SetShaderParameter("width", w)), 0.17f, 0.13f, 0.8)
                    .SetTrans(Tween.TransitionType.Sine);
                r.Pulse = t;
            }
        }
        if (r.Glow is TextureRect g)
        {
            g.Visible = visible;
            g.Modulate = TierColor(r.Mark.Tier);
            if (r.Pulse == null && g.Material is ShaderMaterial m)
            {
                m.SetShaderParameter("width", r.Mark.Tier switch
                {
                    MarkTier.Best => 0.15f,
                    MarkTier.Good => 0.115f,
                    _ => GameGlowWidth + 0.02f,
                });
            }
        }
    }

    /// <summary>A copy of End Turn's own glow (Visuals/Glow), in our colour.</summary>
    private static void EnsureEndTurnGlow(Ring r, NEndTurnButton button, bool visible)
    {
        if (r.Glow == null || !GodotObject.IsInstanceValid(r.Glow) || r.GlowHost != button)
        {
            DropGlow(r);
            if (button.GetNodeOrNull<Control>("Visuals/Glow") is not { } original || original.GetParent() is not Node parent)
            {
                return;
            }
            if (original.Duplicate(0) is not Control glow)
            {
                return;
            }
            glow.Name = "SpireMentorEndTurnGlow";
            IgnoreMouse(glow);
            parent.AddChild(glow);
            parent.MoveChild(glow, original.GetIndex());
            // The scale the game gives its own glow when it shows it
            // (NEndTurnButton: tween to 0.5).
            glow.Scale = Vector2.One * 0.5f;
            r.Glow = glow;
            r.GlowHost = button;
            if (r.Mark.Tier == MarkTier.Best && !SpireMentorModConfig.ReduceEffects)
            {
                r.Pulse?.Kill();
                Tween t = glow.CreateTween().SetLoops();
                t.TweenProperty(glow, "modulate:a", 0.45f, 0.8).SetTrans(Tween.TransitionType.Sine);
                t.TweenProperty(glow, "modulate:a", 0.9f, 0.8).SetTrans(Tween.TransitionType.Sine);
                r.Pulse = t;
            }
        }
        if (r.Glow is Control g)
        {
            g.Visible = visible;
            Color c = TierColor(r.Mark.Tier);
            g.SelfModulate = new Color(c.R, c.G, c.B, 1f);
            if (r.Pulse == null)
            {
                g.Modulate = new Color(1, 1, 1, r.Mark.Tier == MarkTier.Best ? 0.85f : 0.6f);
            }
        }
    }

    /// <summary>
    /// A copied game node keeps its children's mouse filters; make the whole
    /// copy transparent to input and focus (SimUiKit's input-safety contract).
    /// </summary>
    private static void IgnoreMouse(Node node)
    {
        if (node is Control c)
        {
            c.MouseFilter = Control.MouseFilterEnum.Ignore;
            c.FocusMode = Control.FocusModeEnum.None;
        }
        foreach (Node child in node.GetChildren())
        {
            IgnoreMouse(child);
        }
    }

    private static void DropGlow(Ring r)
    {
        if (r.Glow != null && GodotObject.IsInstanceValid(r.Glow))
        {
            if (r.Pulse != null && r.Pulse.IsValid())
            {
                r.Pulse.Kill();
            }
            r.Glow.QueueFreeSafely();
        }
        r.Glow = null;
        r.GlowHost = null;
    }

    // ------------------------------------------------------------------
    // Pulling the plan's cards out of the fan
    // ------------------------------------------------------------------

    /// <summary>
    /// Nudge the holder's layout target up by <see cref="LiftShare"/> of the
    /// card's height -- only while it sits at the game's own default spot
    /// (HandPosHelper), so a hovered or dragged card is the game's, and the
    /// game's next re-layout simply gets nudged again.
    /// </summary>
    private static void Lift(Ring r, NPlayerHand handUi)
    {
        if (r.Mark.Tier == MarkTier.Alt || handUi.GetCardHolder(r.Card!) is not NHandCardHolder holder ||
            !GodotObject.IsInstanceValid(holder))
        {
            return;
        }
        IReadOnlyList<NHandCardHolder> active = handUi.ActiveHolders;
        int index = -1;
        for (int i = 0; i < active.Count; i++)
        {
            if (active[i] == holder)
            {
                index = i;
                break;
            }
        }
        if (index < 0)
        {
            return;
        }
        Vector2 home = HandPosHelper.GetPosition(active.Count, index);
        float lift = NCard.defaultSize.Y * HandPosHelper.GetScale(active.Count).Y * LiftShare;
        Vector2 up = home + new Vector2(0f, -lift);
        if (holder.TargetPosition.DistanceTo(home) < 0.5f)
        {
            holder.SetTargetPosition(up);
        }
        r.Lifted = holder;
        r.LiftedTo = up;
        r.LiftedFrom = home;
    }

    // ------------------------------------------------------------------
    // Target: the game's targeting brackets, copied
    // ------------------------------------------------------------------

    private void UpdateReticle(NPlayerHand? handUi, bool visible)
    {
        Mark? mark = null;
        CardModel? card = null;
        CardModel? hovered = HoveredCard(handUi);
        if (hovered != null)
        {
            Ring? ring = _rings.FirstOrDefault(r => r.Card == hovered);
            if (ring != null && (ring.Mark.TargetIndex != null || ring.Mark.AllyIndex != null))
            {
                mark = ring.Mark;
                card = hovered;
            }
        }
        if (mark == null && _pinnedTarget != null)
        {
            mark = _pinnedTarget;
            card = _rings.FirstOrDefault(r => r.Mark == _pinnedTarget)?.Card;
        }
        NCreature? creature = mark == null ? null : CreatureNode(TargetOf(mark, card));
        Control? original = creature?.GetNodeOrNull<Control>("%SelectionReticle");
        if (!visible || creature == null || original == null || !GodotObject.IsInstanceValid(original))
        {
            DropReticle();
            return;
        }

        if (_reticle == null || !GodotObject.IsInstanceValid(_reticle) || _reticleHost != original.GetParent())
        {
            DropReticle();
            if (original.Duplicate(0) is not Control copy || original.GetParent() is not Node parent)
            {
                return;
            }
            copy.Name = "SpireMentorTarget";
            copy.UniqueNameInOwner = false;
            // Every bracket piece, not just the root: a copy on top of the
            // creature's %Hitbox that took the mouse ate its MouseEntered, so
            // the creature lost focus and a targeted card's click missed.
            IgnoreMouse(copy);
            parent.AddChild(copy);
            // Where the game keeps its own, below the hitbox in input order.
            parent.MoveChild(copy, original.GetIndex());
            _reticle = copy;
            _reticleHost = parent;
        }

        bool ally = mark!.AllyIndex is int && mark.TargetIndex is not int;
        Color color = ally ? AllyGreen : TargetRed;
        // Where the game keeps its own (it sizes it to the creature's bounds).
        _reticle.Size = original.Size;
        _reticle.GlobalPosition = original.GlobalPosition;
        _reticle.PivotOffset = original.PivotOffset;
        float breathe = SpireMentorModConfig.ReduceEffects
            ? 1f
            : 1f + 0.025f * (float)Math.Sin(_frame * 0.1);
        _reticle.Scale = Vector2.One * breathe;
        _reticle.Modulate = new Color(color.R, color.G, color.B, 0.95f);
        _reticle.Visible = true;

        if (EnsureReticleBadge())
        {
            if (_reticleBadge!.GetThemeStylebox("normal") is StyleBoxFlat badge)
            {
                badge.BgColor = color;
            }
            _reticleBadge.Text = mark.Order is int o
                ? o.ToString(System.Globalization.CultureInfo.InvariantCulture)
                : "◎";
            Vector2 corner = _reticle.GetGlobalTransformWithCanvas() * Vector2.Zero;
            _reticleBadge.Position = corner - new Vector2(BadgeSize * 0.5f, BadgeSize * 0.5f);
            _reticleBadge.Visible = true;
        }
    }

    private void DropReticle()
    {
        if (_reticle != null && GodotObject.IsInstanceValid(_reticle))
        {
            _reticle.QueueFreeSafely();
        }
        _reticle = null;
        _reticleHost = null;
        if (_reticleBadge != null && GodotObject.IsInstanceValid(_reticleBadge))
        {
            _reticleBadge.Visible = false;
        }
    }

    private bool EnsureReticleBadge()
    {
        if (_reticleBadge != null && GodotObject.IsInstanceValid(_reticleBadge))
        {
            return true;
        }
        if (_layer == null || !GodotObject.IsInstanceValid(_layer))
        {
            return false;
        }
        _reticleBadge = new Label
        {
            Name = "SpireMentorTargetBadge",
            Size = new Vector2(BadgeSize, BadgeSize),
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
            Visible = false,
        };
        var style = new StyleBoxFlat
        {
            BgColor = TargetRed,
            BorderColor = new Color(0.2f, 0.04f, 0.02f),
            ShadowColor = new Color(0, 0, 0, 0.5f),
            ShadowSize = 4,
            AntiAliasing = true,
        };
        style.SetBorderWidthAll(2);
        style.SetCornerRadiusAll((int)(BadgeSize / 2));
        _reticleBadge.AddThemeStyleboxOverride("normal", style);
        _reticleBadge.AddThemeFontSizeOverride("font_size", 21);
        _reticleBadge.AddThemeColorOverride("font_color", new Color(0.1f, 0.03f, 0.02f));
        _layer.AddChildSafely(_reticleBadge);
        return true;
    }

    private static CardModel? HoveredCard(NPlayerHand? handUi)
    {
        if (handUi == null)
        {
            return null;
        }
        try
        {
            if (handUi.FocusedHolder is { } focused && GodotObject.IsInstanceValid(focused))
            {
                return focused.CardModel;
            }
            // Picked up to aim: the hand no longer calls it focused.
            foreach (NHandCardHolder h in handUi.ActiveHolders)
            {
                if (GodotObject.IsInstanceValid(h) && handUi.IsAwaitingPlay(h))
                {
                    return h.CardModel;
                }
            }
        }
        catch (Exception)
        {
            // fall through
        }
        return null;
    }

    private static Creature? TargetOf(Mark mark, CardModel? card)
    {
        try
        {
            if (card != null)
            {
                return SimCombatHandler.ResolveTarget(card, mark.TargetIndex ?? -1, mark.AllyIndex ?? -1);
            }
            if (mark.TargetIndex is int t && t >= 0)
            {
                IReadOnlyList<Creature>? enemies = CombatManager.Instance.DebugOnlyGetState()?.Enemies;
                return enemies != null && t < enemies.Count ? enemies[t] : null;
            }
        }
        catch (Exception)
        {
            // no target
        }
        return null;
    }

    private static NCreature? CreatureNode(Creature? creature)
    {
        if (creature == null)
        {
            return null;
        }
        NCombatRoom? room = NCombatRoom.Instance;
        if (room == null || !GodotObject.IsInstanceValid(room))
        {
            return null;
        }
        NCreature? node = room.GetCreatureNode(creature);
        return node != null && GodotObject.IsInstanceValid(node) && node.IsVisibleInTree() ? node : null;
    }

    // ------------------------------------------------------------------
    // Nodes
    // ------------------------------------------------------------------

    private static Control? PotionHolder(int slot)
    {
        try
        {
            NPotionContainer? container = SimTopBarLayout.TopBar()?.PotionContainer;
            if (container == null || !GodotObject.IsInstanceValid(container))
            {
                return null;
            }
            // Holders are created in slot order (NPotionContainer.GrowPotionHolders).
            List<NPotionHolder> holders = UiHelper.FindAll<NPotionHolder>(container);
            return slot >= 0 && slot < holders.Count ? holders[slot] : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static Control? EndTurnButton()
    {
        try
        {
            NCombatRoom? room = NCombatRoom.Instance;
            return room != null && GodotObject.IsInstanceValid(room) ? room.Ui?.EndTurnButton : null;
        }
        catch (Exception)
        {
            return null;
        }
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
        // Under the toolbar (98) and the HUD panels (99). Only badges and the
        // generic rings live here; the glows live in the game's own trees.
        _layer = new CanvasLayer { Name = "SpireMentorHighlightLayer", Layer = 97 };
        tree.Root.AddChildSafely(_layer);
        _reticleBadge = null;
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
            Logger.Log($"[SimHighlights] {ex.Message}");
        }
    }
}
