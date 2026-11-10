// SimTargetHover.cs -- while a card is aimed, the enemy under the cursor is
// the target, even when the game's GUI missed the hover (#1221).
//
// The game takes the target from its creature's %Hitbox: MouseEntered ->
// NCreature.OnFocus -> NTargetManager.OnNodeHovered (arrow red, the game's
// yellow brackets). The arrow itself follows the polled mouse position
// (Viewport.GetMousePosition), but the hover only moves on a mouse-motion
// event that Godot's GUI routes to the hitbox. Reported live (0.10.5,
// 2026-11-09): aiming at the enemy the coach marked in red, the arrow's tip
// sat on the enemy and stayed gray; moving around found a spot that took.
// It did not reproduce on the dev machine (~25 aims, windowed, framed and
// fullscreen), and the game has spots of its own that block the hover inside
// the hitbox (the intent icon, the HP bar's hitbox).
//
// So: while targeting, once the cursor has rested inside an interactable
// creature's hitbox for a moment and the game still has no target, emit the
// hitbox's own MouseEntered -- the exact signal the GUI would have sent. When
// the cursor leaves that hitbox (or targeting ends) and the GUI does not hold
// it, emit MouseExited, so a release elsewhere never plays the card on it and
// the creature's IsFocused does not go stale (OnFocus returns early while it
// is set). Each rescue is logged with what the GUI had under the cursor
// instead, so a bug report shows the real cause.
//
// Off when the coach is off: the game is vanilla then.

using System;
using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;

namespace SpireMentor;

internal static class SimTargetHover
{
    /// <summary>How long the cursor rests in a hitbox with no target before
    /// the rescue: the GUI's own hover lands within a frame or two.</summary>
    private const ulong DwellMs = 120;

    private static readonly PropertyInfo? HoveredNodeProperty =
        typeof(NTargetManager).GetProperty("HoveredNode", BindingFlags.Instance | BindingFlags.NonPublic);

    private static NCreature? _rescued;
    private static NCreature? _waiting;
    private static ulong _waitingSince;
    private static bool _triedThisEntry;

    /// <summary>Once per rendered frame (SimHighlights' ProcessFrame hook).</summary>
    public static void Tick(Viewport? viewport)
    {
        NTargetManager? manager = null;
        try { manager = NTargetManager.Instance; } catch (Exception) { /* no run */ }
        if (viewport == null || manager == null || !GodotObject.IsInstanceValid(manager) ||
            !manager.IsInSelection || SimRuntime.Mode == SimMode.Off)
        {
            Release(viewport);
            _waiting = null;
            return;
        }

        NCreature? under = CreatureUnderMouse();
        if (_rescued != null && under != _rescued)
        {
            Release(viewport);
        }
        if (under == null)
        {
            _waiting = null;
            return;
        }
        if (under != _waiting)
        {
            _waiting = under;
            _waitingSince = Time.GetTicksMsec();
            _triedThisEntry = false;
            return;
        }
        if (_triedThisEntry || Time.GetTicksMsec() - _waitingSince < DwellMs || HasTarget(manager, under))
        {
            return;
        }

        _triedThisEntry = true;
        Control? gui = viewport.GuiGetHoveredControl();
        Logger.Log($"[TargetHover] aiming at {under.Entity?.Name ?? under.Name} with no target; " +
                   $"the GUI had {Describe(gui)} under the cursor (focused={under.IsFocused}); setting the hover");
        if (under.IsFocused)
        {
            // OnFocus returns early while IsFocused is set: clear it first.
            under.Hitbox.EmitSignal(Control.SignalName.MouseExited);
        }
        under.Hitbox.EmitSignal(Control.SignalName.MouseEntered);
        _rescued = under;
    }

    /// <summary>Hand a rescued hover back once the cursor left it, unless the
    /// GUI itself now holds it (then its own MouseExited will follow).</summary>
    private static void Release(Viewport? viewport)
    {
        NCreature? c = _rescued;
        _rescued = null;
        if (c == null || !GodotObject.IsInstanceValid(c) || !GodotObject.IsInstanceValid(c.Hitbox))
        {
            return;
        }
        if (c.IsFocused && viewport?.GuiGetHoveredControl() != c.Hitbox)
        {
            c.Hitbox.EmitSignal(Control.SignalName.MouseExited);
        }
    }

    /// <summary>The game has a target: its HoveredNode (private, read by
    /// name; when a game update renames it, the creature's own IsFocused).</summary>
    private static bool HasTarget(NTargetManager manager, NCreature under)
    {
        if (HoveredNodeProperty != null)
        {
            try
            {
                return HoveredNodeProperty.GetValue(manager) != null;
            }
            catch (Exception)
            {
                // fall through
            }
        }
        return under.IsFocused;
    }

    /// <summary>The interactable creature whose hitbox holds the cursor (the
    /// same polled position the targeting arrow follows).</summary>
    private static NCreature? CreatureUnderMouse()
    {
        NCombatRoom? room = NCombatRoom.Instance;
        if (room == null || !GodotObject.IsInstanceValid(room))
        {
            return null;
        }
        NCreature? found = null;
        foreach (NCreature c in room.CreatureNodes)
        {
            if (!GodotObject.IsInstanceValid(c) || !c.IsVisibleInTree())
            {
                continue;
            }
            Control hit = c.Hitbox;
            if (!GodotObject.IsInstanceValid(hit) || hit.MouseFilter == Control.MouseFilterEnum.Ignore ||
                !hit.IsVisibleInTree())
            {
                continue;
            }
            if (hit.GetGlobalRect().HasPoint(hit.GetGlobalMousePosition()))
            {
                // Later in the tree draws on top: the one the GUI would pick.
                found = c;
            }
        }
        return found;
    }

    private static string Describe(Control? c)
    {
        if (c == null || !GodotObject.IsInstanceValid(c))
        {
            return "nothing";
        }
        string path = c.GetPath().ToString();
        int cut = path.IndexOf("/CombatRoom/", StringComparison.Ordinal);
        return (cut >= 0 ? path[(cut + "/CombatRoom/".Length)..] : path) + $" ({c.GetClass()}, {c.MouseFilter})";
    }
}
