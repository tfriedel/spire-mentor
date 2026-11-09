// SimHud.cs -- in-game HUD v2 (replaces SimOverlay.cs).
//
// Renders, entirely in code (no .pck -- see SpireMentor.csproj header):
//   (a) a persistent status chip: mode, advisor connection state, last
//       action / "thinking" indicator;
//   (b) an advice panel: win-chance bar, ranked action list (label, p_win,
//       visit-share bar, Q-colored), principal-variation line, confidence
//       marker;
//   (c) per-hand-card score labels with a best-move highlight.
//
// Sources: autopilot action annotations (legacy {note, cards[]} and the
// rich shape, both adapted in SimAdvice.Parse) today; advisor-mode advice
// pushes (phase 5/6) reuse ShowAdvice unchanged.
//
// Draws into its OWN CanvasLayer (layer 99) parented under the root
// Viewport, so it sits above all game UI regardless of Control ordering.
// (SimOverlay's old header claimed "no CanvasLayer anywhere in this
// codebase"; that is true of the game's .cs files but false of its scenes
// -- NDevConsole.Create() returns a CanvasLayer -- so the idiom is safe.)
// CanvasLayer coordinates are viewport pixels, the same space as
// Control.GlobalPosition in the game's single Control tree, so card-anchored
// labels can copy NCard.GlobalPosition directly.
//
// THREADING: unlike SimOverlay (which could assume the main thread because
// annotations only ever arrived inside a main-thread action handler),
// advice can arrive on the bridge's socket thread. Every public entry point
// therefore marshals to the main thread via Callable.From(...).CallDeferred
// unless already there. All node access happens on the main thread only.
//
// Never-break-gameplay policy carried over verbatim from SimOverlay: every
// render path is wrapped in try/catch that only logs; nodes we don't own
// (NGame, NPlayerHand, NCard) are guarded with GodotObject.IsInstanceValid;
// our own panels are rebuilt, never incrementally patched.
//
// Card lookup (decomp-verified in SimOverlay, unchanged): wire index ->
// PileType.Hand.GetPile(player).Cards[i] -> NPlayerHand.Instance.GetCard
// (CardModel identity, NOT holder child order -- see NPlayerHand.cs:714-731
// for why holder-index addressing silently mislabels cards) ->
// NCard.GetCurrentSize() with NCard.defaultSize fallback.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace SpireMentor;

// ---------------------------------------------------------------------------
// Typed advice model
// ---------------------------------------------------------------------------

/// <summary>One ranked option in an advice payload.</summary>
internal sealed class SimAdviceOption
{
    public string Label = "";
    public double? PWin;
    public double? Q;
    public long? Visits;
    public double? VisitShare;
    public long? Wins;
    public long? Losses;
    public long? Ties;
    public bool IsBest;
    public bool LowConfidence;

    /// <summary>`in_band`: the engine cannot tell this move from its pick
    /// (GH #1022). False on screen advice and older advisors.</summary>
    public bool InBand;

    /// <summary>`pick_rank`: the engine's preference order, 0 = its pick.</summary>
    public int? PickRank;

    /// <summary>Hand index for card options (drives the per-card label).</summary>
    public int? CardIndex;

    /// <summary>Map-graph coordinate for map options (drives the map marker).</summary>
    public int? Col;
    public int? Row;

    /// <summary>Legacy 0..1 score (old cards[] shape) when no p_win exists.</summary>
    public double? Score;

    /// <summary>`action` ("play" / "potion" / "end_turn" / "choose" / "skip");
    /// empty from an advisor that sends it only on `best`.</summary>
    public string Action = "";

    /// <summary>Enemy (`target_index`) or teammate (`ally_index`) the move
    /// aims at, in the request's own lists; null when it aims at nothing.</summary>
    public int? TargetIndex;
    public int? AllyIndex;

    /// <summary>Potion slot (bridge absolute), for a potion option.</summary>
    public int? Slot;

    /// <summary>Screen option index (`index`), for a screen decision.</summary>
    public int? Index;

    /// <summary>`hp_mean`: the HP the search expects this move to leave you
    /// with (PROTOCOL.md section 5); null when the advisor sends none.</summary>
    public double? HpMean;

    /// <summary>"How much the AI likes it", in [0,1]: the search's visit
    /// share, else the net's probability. Null for a legacy score.</summary>
    public double? Share => VisitShare ?? (Q == null ? PWin : null);

    /// <summary>The game-free view AdviceMarks works on.</summary>
    public MarkOption ToMark() => new(Label, Action, CardIndex, TargetIndex, AllyIndex, Slot, Index,
        IsBest, InBand, Share, Q, PickRank);
}

/// <summary>One step of the turn's plan (`advice.plan[]`, PROTOCOL.md section 5).</summary>
internal sealed class SimPlanStep
{
    public string Action = "";
    public int? CardIndex;
    public int? TargetIndex;
    public int? AllyIndex;
    public int? Slot;
    public string Label = "";

    public MarkPlanStep ToMark() => new(Action, CardIndex, TargetIndex, AllyIndex, Slot, Label);
}

/// <summary>
/// The executable recommendation (phase 7 takeover): what "AI takes this
/// decision" actually enqueues. Parsed from the advice `best` object.
/// </summary>
internal sealed class SimBestAction
{
    public string Action = ""; // "play" | "end_turn" | "potion" | "choose" | "skip"
    public int CardIndex = -1;
    public int TargetIndex = -1;
    /// <summary>Co-op: an AnyAlly card's teammate, an index into the
    /// request's <c>allies[]</c> (GH #1017); -1 when the advice names none.</summary>
    public int AllyIndex = -1;
    public int Slot = -1;

    /// <summary>Option index for screen decisions ("choose").</summary>
    public int Index = -1;

    public string Label = "";
}

/// <summary>One node of the projected route (advice `path[]`).</summary>
internal sealed class SimPathNode
{
    public int Col;
    public int Row;
    public string Type = "";

    /// <summary>0 = the node to click now, 1 = the one after it, ...</summary>
    public int Hop;

    /// <summary>HP the projection expects to ARRIVE here with, before this
    /// node's own room resolves. Null on an advisor too old to send it (the
    /// route then carries no HP story and the label is omitted).</summary>
    public int? Hp;
}

/// <summary>
/// Parsed advice payload. Handles both the legacy autopilot annotations
/// shape ({note, cards:[{index,score,label}]}) and the rich shape
/// ({note, p_win, best, options:[...], pv:[...]}); a payload may carry both.
/// </summary>
internal sealed class SimAdvice
{
    public string? Note;
    public double? PWin;
    public bool LowConfidence;
    public SimBestAction? Best;
    public readonly List<SimAdviceOption> Options = new();
    public readonly List<string> Pv = new();

    /// <summary>The turn's play order (`plan[]`): step 1 is `best`.</summary>
    public readonly List<SimPlanStep> Plan = new();

    /// <summary>
    /// `advice.hand_back`: the advisor's own word that this answer is not
    /// really a decision (unranked, a borrowed head, an error), as a short
    /// player-facing reason. F7 stops on it under `pause_on: "unsupported"`
    /// (GH #874). Null on a real recommendation and on older advisors.
    /// </summary>
    public string? HandBack;

    /// <summary>`advice.brain` (v5.2): "instinct" when the end-to-end net
    /// answered, "search" or null otherwise (AdvisorBrainRules).</summary>
    public string? Brain;

    /// <summary>`advice.instinct_unavailable` (v5.2): Instinct was asked for
    /// and the search answered instead, with the player-facing reason.</summary>
    public string? InstinctUnavailable;

    /// <summary>The source tag the HUD shows ("Instinct" / "Search
    /// (Instinct: ...)"), null for a plain search reply.</summary>
    public string? SourceTag => AdvisorBrainRules.SourceTag(Brain, InstinctUnavailable);

    /// <summary>
    /// `advice.wait_for_ally.max_ms` (co-op): a teammate who has not ended
    /// their turn holds a card that would change this move (Vulnerable before
    /// our attack, Weak or block before our block), so autoplay holds the
    /// move up to this long. 0 on a normal reply and on older advisors.
    /// </summary>
    public int WaitForAllyMs;

    /// <summary>Projected route through the act (map advice only).</summary>
    public readonly List<SimPathNode> Path = new();

    /// <summary>`advice.pick` (combat, GH #1022): why the engine took its
    /// pick over the other in-band options, and the HP gap to the
    /// runner-up. Null reason when the advice carries no engine pick
    /// (screens, a vetoed pick, older advisors).</summary>
    public string? PickReason;
    public double? PickGapHp;

    /// <summary>
    /// Q of the recommended option, in [-1,1]. This -- not p_win -- is what
    /// the eval bar and the per-option numbers show: win_prob saturates at
    /// ~1.0 for every legal action in a fight the search expects to win, so a
    /// p_win list reads "100%" on every row with identical bars (reported
    /// 2026-08-05). The workbench's Reasoning panel made the same call.
    /// </summary>
    public double? BestQ
    {
        get
        {
            SimAdviceOption? best = Options.Find(o => o.IsBest) ?? Options.Find(o => o.Q != null);
            return best?.Q;
        }
    }

    public static SimAdvice Parse(JsonElement el)
    {
        var advice = new SimAdvice();
        if (el.ValueKind != JsonValueKind.Object)
        {
            return advice;
        }

        advice.Note = ReadString(el, "note");
        advice.PWin = ReadDouble(el, "p_win");
        advice.LowConfidence = ReadBool(el, "low_confidence") ?? false;
        advice.HandBack = ReadString(el, "hand_back");
        advice.Brain = ReadString(el, "brain");
        advice.InstinctUnavailable = ReadString(el, "instinct_unavailable");
        if (el.TryGetProperty("wait_for_ally", out JsonElement waitEl) && waitEl.ValueKind == JsonValueKind.Object)
        {
            advice.WaitForAllyMs = Math.Max(0, ReadInt(waitEl, "max_ms") ?? 0);
        }
        if (el.TryGetProperty("pick", out JsonElement pickEl) && pickEl.ValueKind == JsonValueKind.Object)
        {
            advice.PickReason = ReadString(pickEl, "reason");
            advice.PickGapHp = ReadDouble(pickEl, "gap_hp");
        }

        if (el.TryGetProperty("best", out JsonElement bestEl) && bestEl.ValueKind == JsonValueKind.Object)
        {
            advice.Best = new SimBestAction
            {
                Action = ReadString(bestEl, "action") ?? "play",
                CardIndex = ReadInt(bestEl, "card_index") ?? -1,
                TargetIndex = ReadInt(bestEl, "target_index") ?? -1,
                AllyIndex = ReadInt(bestEl, "ally_index") ?? -1,
                Slot = ReadInt(bestEl, "slot") ?? -1,
                Index = ReadInt(bestEl, "index") ?? -1,
                Label = ReadString(bestEl, "label") ?? "",
            };
        }

        if (el.TryGetProperty("options", out JsonElement optionsEl) && optionsEl.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement o in optionsEl.EnumerateArray())
            {
                if (o.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                advice.Options.Add(new SimAdviceOption
                {
                    Label = ReadString(o, "label") ?? "?",
                    PWin = ReadDouble(o, "p_win"),
                    Q = ReadDouble(o, "q"),
                    Visits = ReadLong(o, "visits"),
                    VisitShare = ReadDouble(o, "visit_share"),
                    Wins = ReadLong(o, "wins"),
                    Losses = ReadLong(o, "losses"),
                    Ties = ReadLong(o, "ties"),
                    IsBest = ReadBool(o, "is_best") ?? false,
                    LowConfidence = ReadBool(o, "low_confidence") ?? false,
                    InBand = ReadBool(o, "in_band") ?? false,
                    PickRank = ReadInt(o, "pick_rank"),
                    CardIndex = ReadInt(o, "card_index"),
                    Col = ReadInt(o, "col"),
                    Row = ReadInt(o, "row"),
                    Action = ReadString(o, "action") ?? "",
                    TargetIndex = ReadInt(o, "target_index"),
                    AllyIndex = ReadInt(o, "ally_index"),
                    Slot = ReadInt(o, "slot"),
                    Index = ReadInt(o, "index"),
                    HpMean = ReadDouble(o, "hp_mean"),
                });
            }
        }

        // Legacy per-card scores; also the only card->hand-index mapping the
        // old shape has, so fold them into Options as card entries.
        if (el.TryGetProperty("cards", out JsonElement cardsEl) && cardsEl.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement c in cardsEl.EnumerateArray())
            {
                if (c.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                int? index = ReadInt(c, "index");
                if (index == null)
                {
                    continue;
                }
                advice.Options.Add(new SimAdviceOption
                {
                    Label = ReadString(c, "label") ?? "",
                    Score = ReadDouble(c, "score"),
                    CardIndex = index,
                });
            }
        }

        if (el.TryGetProperty("path", out JsonElement pathEl) && pathEl.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement p in pathEl.EnumerateArray())
            {
                if (p.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                int? col = ReadInt(p, "col");
                int? row = ReadInt(p, "row");
                if (col == null || row == null)
                {
                    continue;
                }
                advice.Path.Add(new SimPathNode
                {
                    Col = col.Value,
                    Row = row.Value,
                    Type = ReadString(p, "type") ?? "",
                    Hop = ReadInt(p, "hop") ?? advice.Path.Count,
                    Hp = ReadInt(p, "hp"),
                });
            }
        }

        if (el.TryGetProperty("plan", out JsonElement planEl) && planEl.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement p in planEl.EnumerateArray())
            {
                if (p.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                advice.Plan.Add(new SimPlanStep
                {
                    Action = ReadString(p, "action") ?? "play",
                    CardIndex = ReadInt(p, "card_index"),
                    TargetIndex = ReadInt(p, "target_index"),
                    AllyIndex = ReadInt(p, "ally_index"),
                    Slot = ReadInt(p, "slot"),
                    Label = ReadString(p, "label") ?? "",
                });
            }
        }

        if (el.TryGetProperty("pv", out JsonElement pvEl) && pvEl.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement p in pvEl.EnumerateArray())
            {
                if (p.ValueKind == JsonValueKind.String)
                {
                    advice.Pv.Add(p.GetString() ?? "");
                }
            }
        }

        return advice;
    }

    private static string? ReadString(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static double? ReadDouble(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

    private static long? ReadLong(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out long r) ? r : null;

    private static int? ReadInt(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int r) ? r : null;

    private static bool? ReadBool(JsonElement el, string name) =>
        el.TryGetProperty(name, out JsonElement v) && (v.ValueKind == JsonValueKind.True || v.ValueKind == JsonValueKind.False)
            ? v.GetBoolean()
            : null;
}

/// <summary>Advisor backend connection state, shown in the status chip.</summary>
internal enum SimAdvisorStatus
{
    None,
    Launching,
    /// <summary>
    /// The dev launcher is compiling the Rust engine before the advisor can
    /// start. `uv run` rebuilds `sts2_rs` whenever a `.rs`/`Cargo.*` changed
    /// (deliberate — its cache-keys are what killed the stale-wheel trap), and
    /// a cold rebuild measured ~14 minutes. Without its own status this showed
    /// as "launching" forever, so on 2026-08-22 three launches in a row were
    /// killed mid-build by quitting the game and the build never once finished.
    /// </summary>
    Building,
    Connected,
    Thinking,
    Crashed,
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

/// <summary>
/// A per-card score label plus the card it belongs to.
///
/// The anchor is the <see cref="CardModel"/>, not the <see cref="NCard"/>:
/// NCard is IPoolable, so the node a played card leaves behind is handed
/// straight to the next card drawn. A label that held on to the NODE
/// therefore stayed put and started describing a different card -- a Strike's
/// score sitting on a Defend (reported 2026-08-05). The node is re-resolved
/// from the model on every tick instead.
/// </summary>
internal sealed class CardLabelAnchor
{
    public Label Label = null!;
    public CardModel Model = null!;
    public NCard? Card;
    public float Width;
    public float Height;
}

/// <summary>A marker pinned to a map node (score, or a step of the route).</summary>
internal sealed class MapMarkerAnchor
{
    public Label Label = null!;
    public Control Point = null!;
    public float Width;
    public float Height;

    /// <summary>Vertical offset from the node centre, in viewport pixels.</summary>
    public float DY;
}

internal sealed class SimHud
{
    public static readonly SimHud Instance = new();

    private SimHud()
    {
        // Status chip tracks mode/HUD-toggle changes from any surface
        // (settings page, hotkey, console).
        SimRuntime.Changed += () => RunOnMainThread(() =>
        {
            RefreshStatusAndVisibility();
            // Switched into the Coach style (or the coach turned on in it):
            // whatever Hints drew for this position must not stay up.
            if (SimRuntime.CoachStyleActive && !_peekShowing)
            {
                ClearPreActionSurfaces();
            }
            else if (!SimRuntime.CoachStyleActive && _peekShowing)
            {
                // Left the Coach style with a peek up: it was Coach-only.
                _peekShowing = false;
                ClearPreActionSurfaces();
            }
        });
    }

    private CanvasLayer? _layer;
    private PanelContainer? _advicePanel;
    private VBoxContainer? _adviceRows;
    private readonly List<CardLabelAnchor> _cardLabels = new();
    private readonly Dictionary<Creature, Label> _enemyTags = new();
    private readonly List<MapMarkerAnchor> _mapMarkers = new();
    private readonly List<Control> _routePoints = new();
    private Line2D? _routeLine;
    private bool _combatEndedSubscribed;
    // The "… N more" toggle on the ranked list: sticky across advice updates,
    // and re-renders from the advice the panel last showed.
    private bool _rankedExpanded;
    private SimAdvice? _panelAdvice;

    /// <summary>The latest advice the advisor delivered, rendered or not
    /// (the Coach style keeps it for the F5 peek), and the player it was
    /// rendered for. Cleared by <see cref="ClearAdvice"/>.</summary>
    private SimAdvice? _latestAdvice;
    private Player? _latestPlayer;
    private SimScreenDecision? _latestDecision;

    /// <summary>The F5 peek (or its "thinking…" line) is on screen.</summary>
    private bool _peekShowing;


    // ------------------------------------------------------------------
    // Public entry points (any thread)
    // ------------------------------------------------------------------

    /// <summary>Render an advice payload. Never throws.
    /// <paramref name="decision"/>: the screen the advice answers, whose
    /// option nodes get the highlight rings (null in combat).</summary>
    public void ShowAdvice(SimAdvice advice, Player? player, SimScreenDecision? decision = null)
    {
        RunOnMainThread(() =>
        {
            _latestDecision = decision;
            // The grade book for the play the human makes from this position.
            SimMoveFeedback.Instance.OnAdvice(advice, player);
            RenderAdvice(advice, player);
        });
    }

    /// <summary>Legacy annotations entry point (autopilot action replies).</summary>
    public void ApplyAnnotations(JsonElement annotations, string actionSummary, Player? player)
    {
        if (!SimConfig.Instance.Overlay)
        {
            return;
        }
        SimAdvice advice;
        try
        {
            advice = SimAdvice.Parse(annotations);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Ignoring malformed annotations: {ex.Message}");
            return;
        }
        SimToolbar.Instance.SetLastAction(actionSummary);
        RunOnMainThread(() => RenderAdvice(advice, player));
    }

    /// <summary>Advisor backend state, shown on the toolbar's status line
    /// (the toolbar absorbed the old status chip, 2026-08-21).</summary>
    public void SetAdvisorStatus(SimAdvisorStatus status) =>
        SimToolbar.Instance.SetAdvisorStatus(status);

    /// <summary>The panels moved into or out of the game (FrameLink): redraw
    /// the standing advice, or the F5 peek, where it now belongs. Main thread.</summary>
    public void OnFrameUiModeChanged()
    {
        if (_peekShowing)
        {
            // RenderAdvice leaves a peek alone (Coach style), so move it here:
            // out of the game it is the frame's alone (ShowPeekLine already
            // sent it); back in, draw it again.
            if (FrameLink.UiOutside)
            {
                HideInGameAdvicePanel();
            }
            else if (!ShowPeekNow())
            {
                ShowPeekThinking();
            }
            return;
        }
        if (_latestAdvice != null)
        {
            RenderAdvice(_latestAdvice, _latestPlayer);
        }
        else if (FrameLink.UiOutside)
        {
            HideInGameAdvicePanel();
        }
    }

    private void HideInGameAdvicePanel()
    {
        ClearAdvicePanelRows();
        if (_advicePanel != null && GodotObject.IsInstanceValid(_advicePanel))
        {
            _advicePanel.Visible = false;
        }
    }

    /// <summary>Drop advice + card labels (stale state / combat end).</summary>
    public void ClearAdvice()
    {
        RunOnMainThread(() =>
        {
            _latestAdvice = null;
            _latestPlayer = null;
            _latestDecision = null;
            _peekShowing = false;
            ClearCardLabels();
            SimHighlights.Instance.Clear();
            ClearMapMarkers();
            ClearAdvicePanel();
        });
    }

    // ------------------------------------------------------------------
    // Main-thread marshalling
    // ------------------------------------------------------------------

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
            Logger.Log($"[SimHud] Dispatch failed: {ex.Message}");
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
            Logger.Log($"[SimHud] Ignoring render error (never breaks gameplay): {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Rendering (main thread only below this line)
    // ------------------------------------------------------------------

    private void RenderAdvice(SimAdvice advice, Player? player)
    {
        _latestAdvice = advice;
        _latestPlayer = player;
        if (!EnsureLayer())
        {
            return;
        }
        EnsureCombatEndedSubscription();
        // THE Coach-style gate: nothing the advisor said reaches the screen
        // before the player acts -- no ranked panel, eval bar, "line:" row,
        // low-confidence note, card score, map marker / route, or toolbar
        // note. The advice is kept (F5 peek); SimAdvisor still holds the
        // standing recommendation for F6/F7 and the unattended drivers.
        if (SimRuntime.CoachStyleActive)
        {
            RefreshStatusAndVisibility();
            if (!_peekShowing)
            {
                ClearPreActionSurfaces();
            }
            return;
        }
        RefreshStatusAndVisibility(AdvisorBrainRules.NoteWithSource(advice.Note, advice.Brain, advice.InstinctUnavailable));
        RenderAdvicePanel(advice);
        RenderCardLabels(advice, player);
        RenderHighlights(advice, player, _latestDecision, peek: false);
        RenderMapMarkers(advice);
    }

    /// <summary>
    /// Ring and number what the advice says to click (AdviceMarks decides,
    /// SimHighlights draws). The map keeps its own markers and route.
    /// <paramref name="peek"/>: the F5 peek -- the pick alone, with its target
    /// marked without a hover.
    /// </summary>
    private static void RenderHighlights(SimAdvice advice, Player? player, SimScreenDecision? decision, bool peek)
    {
        try
        {
            if (advice.Options.Exists(o => o.Col != null && o.Row != null))
            {
                SimHighlights.Instance.Clear();
                return;
            }
            List<Mark> marks = AdviceMarks.Build(
                advice.Options.ConvertAll(o => o.ToMark()),
                peek ? null : advice.Plan.ConvertAll(s => s.ToMark()));
            if (peek)
            {
                marks = marks.FindAll(m => m.Tier == MarkTier.Best);
            }
            SimHighlights.Instance.Show(marks, player, decision, showTarget: peek);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Highlights failed: {ex.Message}");
        }
    }

    /// <summary>A label as the player should read it: no raw game ids.</summary>
    private static string Words(string? label)
    {
        string words = AdviceMarks.Humanize(label);
        // "skip" / "leave" from an advisor reads as a sentence start.
        return words.Length > 0 && char.IsLower(words[0]) ? char.ToUpperInvariant(words[0]) + words[1..] : words;
    }

    // ------------------------------------------------------------------
    // F5 peek (Coach style): the AI's pick for the next decision only
    // ------------------------------------------------------------------

    /// <summary>
    /// Draw the best option of the latest advice with the Hints renderers,
    /// and nothing else: in combat its card's ★ label plus a one-row panel
    /// naming card and target, on the map a single marker (no route, no other
    /// scores), elsewhere the one-row panel. Main thread only. False when
    /// there is no advice to show (the caller then shows "thinking…").
    /// </summary>
    public bool ShowPeekNow()
    {
        if (!NGame.IsMainThread() || !EnsureLayer())
        {
            return false;
        }
        SimAdvice? advice = _latestAdvice;
        if (advice == null)
        {
            return false;
        }
        SimAdviceOption? best = advice.Options.Find(o => o.IsBest) ?? (advice.Options.Count > 0 ? advice.Options[0] : null);
        string label = Words(best?.Label is { Length: > 0 } l ? l : advice.Best?.Label ?? "");
        if (label.Length == 0)
        {
            return false;
        }
        // The peek names its brain too, when it is the net.
        if (AdvisorBrainRules.IsInstinct(advice.Brain))
        {
            label += "  (Instinct)";
        }
        // No "no advice" to the frame first: the peek replaces it.
        ClearPreActionSurfaces(publish: false);
        _peekShowing = true;
        try
        {
            if (best?.Col != null && best.Row != null && PeekMapMarker(best))
            {
                // In game the marker is the peek; the frame gets the line.
                FrameLink.PublishAdvice(null, "AI would: " + label);
                return true;
            }
            // The pick's ring (card, potion, End Turn, reward card, shop
            // slot...) and, in a fight, its target -- no other marks.
            RenderHighlights(advice, _latestPlayer, _latestDecision, peek: true);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Peek anchor failed: {ex.Message}");
        }
        ShowPeekLine("AI would: " + label, new Color(1f, 0.85f, 0.3f));
        return true;
    }

    /// <summary>F5 before the advice arrived: say so; the pick replaces it.</summary>
    public void ShowPeekThinking()
    {
        RunOnMainThread(() =>
        {
            if (!EnsureLayer())
            {
                return;
            }
            ClearPreActionSurfaces(publish: false);
            _peekShowing = true;
            ShowPeekLine("AI is thinking…", new Color(0.7f, 0.7f, 0.75f));
        });
    }

    /// <summary>The decision the peek was for is made (or the board moved):
    /// hide it. No-op unless a peek is showing, so Hints is never touched.</summary>
    public void ClearPeek()
    {
        RunOnMainThread(() =>
        {
            if (!_peekShowing)
            {
                return;
            }
            _peekShowing = false;
            ClearPreActionSurfaces();
        });
    }

    private void ShowPeekLine(string text, Color color)
    {
        // Framed: the frame's advice panel shows the peek too (both UI
        // modes); panels outside the game: only there.
        FrameLink.PublishAdvice(null, text);
        if (FrameLink.UiOutside)
        {
            return;
        }
        if (_adviceRows == null || !GodotObject.IsInstanceValid(_adviceRows) ||
            _advicePanel == null || !GodotObject.IsInstanceValid(_advicePanel))
        {
            return;
        }
        Label row = SmallLabel(text + "   (F5 peek)", color);
        row.CustomMinimumSize = new Vector2(320, 0);
        row.ClipText = true;
        _adviceRows.AddChildSafely(row);
        _advicePanel.Visible = true;
    }

    private bool PeekMapMarker(SimAdviceOption best)
    {
        NMapScreen? map = NMapScreen.Instance;
        if (map == null || !GodotObject.IsInstanceValid(map) || !map.IsOpen || !map.IsVisibleInTree())
        {
            return false;
        }
        foreach (NMapPoint mp in UiHelper.FindAll<NMapPoint>(map))
        {
            if (GodotObject.IsInstanceValid(mp) &&
                mp.Point.coord.col == best.Col!.Value && mp.Point.coord.row == best.Row!.Value)
            {
                AddMapMarker(mp, "★ AI", new Color(1f, 0.85f, 0.3f), 6, dy: -52f);
                return true;
            }
        }
        return false;
    }

    /// <summary>Remove every pre-action advice surface (Coach style).
    /// <paramref name="publish"/>: also clear the frame's advice panel (not
    /// when a peek is about to replace it).</summary>
    private void ClearPreActionSurfaces(bool publish = true)
    {
        ClearCardLabels();
        SimHighlights.Instance.Clear();
        ClearMapMarkers();
        if (publish)
        {
            ClearAdvicePanel();
        }
        else
        {
            ClearAdvicePanelRows();
        }
        if (_advicePanel != null && GodotObject.IsInstanceValid(_advicePanel))
        {
            _advicePanel.Visible = false;
        }
    }

    private bool HudVisible => SpireMentorModConfig.ShowHud && SimRuntime.Mode != SimMode.Off;

    private void RefreshStatusAndVisibility(string? note = null)
    {
        if (!EnsureLayer())
        {
            return;
        }

        _layer!.Visible = HudVisible && !SimUiHide.Hidden;
        // The status text itself lives on the toolbar now; the advice note
        // rides along (and clears on note-less refreshes, as the chip did).
        SimToolbar.Instance.SetNote(HudVisible && !SimRuntime.CoachStyleActive ? note : null);
    }

    private void RenderAdvicePanel(SimAdvice advice)
    {
        // Panels outside the game: the frame draws the advice panel (all of
        // it -- the frame has the room the in-game toggles were rationing).
        if (FrameLink.UiOutside)
        {
            ClearAdvicePanelRows();
            if (_advicePanel != null && GodotObject.IsInstanceValid(_advicePanel))
            {
                _advicePanel.Visible = false;
            }
            _panelAdvice = advice;
            FrameLink.PublishAdvice(advice);
            return;
        }
        ClearAdvicePanelRows();
        _panelAdvice = advice;
        FrameLink.PublishAdvice(advice); // framed, panels inside: keep the frame current

        bool hasRanked = advice.Options.Exists(o => o.PWin != null || o.Q != null || o.VisitShare != null);
        bool hasEval = advice.BestQ != null || advice.PWin != null;
        bool wantPanel = (SpireMentorModConfig.ShowWinBar && hasEval) ||
                         (SpireMentorModConfig.ShowRankedList && hasRanked) ||
                         advice.Pv.Count > 0 || advice.Plan.Count > 0;
        if (!wantPanel || _adviceRows == null || !GodotObject.IsInstanceValid(_adviceRows))
        {
            if (_advicePanel != null && GodotObject.IsInstanceValid(_advicePanel))
            {
                _advicePanel.Visible = false;
            }
            return;
        }

        _advicePanel!.Visible = true;

        // Which brain is talking, when it is not the plain search: the net
        // ("Instinct"), or the search standing in for it with the reason --
        // never a silent fallback (PROTOCOL.md section 5, v5.2). Instinct
        // sends no q / p_win, so hasEval is false and its rows bar by share.
        if (advice.SourceTag is { } source)
        {
            bool instinct = AdvisorBrainRules.IsInstinct(advice.Brain);
            Label tag = SmallLabel(source, instinct ? new Color(0.75f, 0.85f, 1f) : new Color(0.6f, 0.6f, 0.65f));
            tag.CustomMinimumSize = new Vector2(320, 0);
            tag.ClipText = true;
            _adviceRows.AddChildSafely(tag);
        }

        if (SpireMentorModConfig.ShowWinBar && hasEval)
        {
            _adviceRows.AddChildSafely(BuildEvalBar(advice.BestQ, advice.PWin, advice.LowConfidence));
        }

        if (SpireMentorModConfig.ShowRankedList && hasRanked)
        {
            // Best first, then the plan in its order, then by share. A move
            // nobody would make (under 0.5%) waits behind "… N more".
            List<SimAdviceOption> ranked = advice.Options.FindAll(o => o.PWin != null || o.Q != null || o.VisitShare != null);
            Dictionary<SimAdviceOption, int> planOrder = PlanOrder(advice);
            ranked.Sort((a, b) =>
            {
                int best = b.IsBest.CompareTo(a.IsBest);
                if (best != 0)
                {
                    return best;
                }
                int pa = planOrder.TryGetValue(a, out int x) ? x : int.MaxValue;
                int pb = planOrder.TryGetValue(b, out int y) ? y : int.MaxValue;
                if (pa != pb)
                {
                    return pa.CompareTo(pb);
                }
                return (b.Share ?? 0).CompareTo(a.Share ?? 0);
            });
            const int maxRows = 6;
            double? bestQ = advice.BestQ;
            List<SimAdviceOption> visible = ranked.FindAll(o => !AdviceMarks.Hidden(o.ToMark()));
            List<SimAdviceOption> rows = _rankedExpanded ? ranked : visible.GetRange(0, Math.Min(visible.Count, maxRows));
            bool anyHp = rows.Exists(o => o.HpMean != null);
            foreach (SimAdviceOption o in rows)
            {
                _adviceRows.AddChildSafely(BuildOptionRow(o, bestQ, planOrder.TryGetValue(o, out int step) ? step : null, anyHp));
            }
            if (ranked.Count > rows.Count || _rankedExpanded && ranked.Count > maxRows)
            {
                _adviceRows.AddChildSafely(BuildRankedToggle(ranked.Count - rows.Count));
            }

            // Say when the ranking is not really a ranking: the moves the
            // engine's tie band holds, and why it took its pick over them. In
            // a fight the search expects to win the Qs converge, and a 0.003
            // gap printed without comment reads as a real preference. The
            // band and the reason are the engine's own (GH #1022), not a
            // threshold of this HUD's.
            List<SimAdviceOption> band = advice.Options.FindAll(o => o.InBand);
            band.Sort((a, b) => (a.IsBest ? -1 : a.PickRank ?? int.MaxValue)
                .CompareTo(b.IsBest ? -1 : b.PickRank ?? int.MaxValue));
            List<string> inBand = band.ConvertAll(o => Words(o.Label));
            foreach (string line in TieBandText.Lines(inBand, advice.PickReason, advice.PickGapHp))
            {
                _adviceRows.AddChildSafely(SmallLabel(line, new Color(0.7f, 0.7f, 0.75f)));
            }
        }

        // The turn's plan in words, numbered like the cards' badges. Without a
        // plan, the search's line is engine detail (its later plies can only
        // name the action kind): behind "Show engine numbers".
        string? planLine = advice.Plan.Count > 0
            // No-break spaces inside a step: the line wraps BETWEEN steps,
            // never between "3." and its card (live, 2026-11-07).
            ? "Plan: " + string.Join("    ", advice.Plan.Select((s, i) => $"{i + 1}. {Words(s.Label)}".Replace(' ', ' ')))
            : advice.Pv.Count > 0 && SpireMentorModConfig.ShowNumbers
                ? "Line: " + string.Join(" · ", advice.Pv.Select(Words))
                : null;
        if (planLine != null)
        {
            // Wrapped at the panel's width: the panel sizes itself to its
            // widest child, so an unwrapped row would take over the screen.
            Label plan = SmallLabel(planLine, new Color(1f, 0.9f, 0.62f));
            plan.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            plan.CustomMinimumSize = new Vector2(PanelWidth - 24f, 0);
            _adviceRows.AddChildSafely(plan);
        }

        if (advice.LowConfidence)
        {
            _adviceRows.AddChildSafely(SmallLabel("low confidence", new Color(0.95f, 0.75f, 0.3f)));
        }

        ShrinkAdvicePanelToFit();
    }

    /// <summary>
    /// The position's headline: the search's calibrated win chance, as a bar.
    /// The raw eval (Q) and the "~" for a saturated search are engine detail,
    /// behind "Show engine numbers". Q drives the fill only when there is no
    /// win chance -- a fight the net expects to win reports p_win 100% for
    /// every option, which is why the rows never show it.
    /// </summary>
    private static Control BuildEvalBar(double? q, double? pWin, bool lowConfidence)
    {
        double fill = pWin != null
            ? Math.Clamp(pWin.Value, 0.0, 1.0)
            : Math.Clamp(((q ?? 0) + 1.0) / 2.0, 0.0, 1.0);

        var row = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        row.AddThemeConstantOverride("separation", 2);
        string text = pWin != null
            ? $"Win chance {pWin.Value.ToString("0%", CultureInfo.InvariantCulture)}"
            : "Win chance ?";
        if (SpireMentorModConfig.ShowNumbers && q != null)
        {
            text += $"   eval {(q.Value >= 0 ? "+" : "")}{q.Value.ToString("0.00", CultureInfo.InvariantCulture)}";
            if (lowConfidence)
            {
                text += " (~)";
            }
        }
        Label head = SmallLabel(text, Colors.White);
        head.AddThemeFontSizeOverride("font_size", 16);
        row.AddChild(head);

        var barBg = new ColorRect
        {
            Color = new Color(0.22f, 0.1f, 0.1f),
            CustomMinimumSize = new Vector2(0, 8),
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        var barFill = new ColorRect
        {
            Color = QColor(fill * 2 - 1),
            MouseFilter = Control.MouseFilterEnum.Ignore,
            AnchorRight = (float)fill,
            AnchorBottom = 1f,
        };
        barBg.AddChild(barFill);
        row.AddChild(barBg);
        return row;
    }

    /// <summary>What the frame's advice panel shows per option, decided here
    /// once so both panels agree (FrameLink.AdviceDto).</summary>
    internal readonly record struct OptionFacts(string Label, string Kind, int? Step, string Verdict, bool Hidden);

    /// <summary>
    /// <see cref="OptionFacts"/> for every option of <paramref name="advice"/>,
    /// in option order: the label in words, the move's kind ("attack",
    /// "skill", "power", "potion", "end_turn", "pick", "skip", "other"), its
    /// plan step, the verdict word, and whether it is a move nobody would
    /// make. Main thread (reads the hand for the card kinds).
    /// </summary>
    internal List<OptionFacts> FactsFor(SimAdvice advice)
    {
        Dictionary<SimAdviceOption, int> order = PlanOrder(advice);
        double? bestQ = advice.BestQ;
        return advice.Options.ConvertAll(o =>
        {
            MarkOption m = o.ToMark();
            string kind = KindOf(o) switch
            {
                RowKind.EndTurn => "end_turn",
                RowKind k => k.ToString().ToLowerInvariant(),
            };
            return new OptionFacts(Words(o.Label), kind, order.TryGetValue(o, out int s) ? s : null,
                AdviceMarks.RowVerdict(m, bestQ, order.TryGetValue(o, out int st) ? st : null), AdviceMarks.Hidden(m));
        });
    }

    /// <summary>The plan's step labels, in words (FrameLink.AdviceDto).</summary>
    internal static List<string> PlanWords(SimAdvice advice) => advice.Plan.ConvertAll(s => Words(s.Label));

    /// <summary>
    /// The step each option is in the turn's plan (1 = the pick), matched by
    /// the thing it acts on (AdviceMarks.Target) and, for a card, its target.
    /// </summary>
    private static Dictionary<SimAdviceOption, int> PlanOrder(SimAdvice advice)
    {
        var order = new Dictionary<SimAdviceOption, int>();
        for (int i = 0; i < advice.Plan.Count; i++)
        {
            MarkPlanStep s = advice.Plan[i].ToMark();
            var stepThing = AdviceMarks.Target(new MarkOption(s.Label, s.Action, s.CardIndex, s.TargetIndex,
                s.AllyIndex, s.Slot, null, false, false, null, null, null));
            if (stepThing == null)
            {
                continue;
            }
            SimAdviceOption? match = advice.Options.Find(o =>
                !order.ContainsKey(o) && AdviceMarks.Target(o.ToMark()) == stepThing &&
                o.TargetIndex == s.TargetIndex && o.AllyIndex == s.AllyIndex);
            match ??= advice.Options.Find(o => !order.ContainsKey(o) && AdviceMarks.Target(o.ToMark()) == stepThing);
            if (match != null)
            {
                order[match] = i + 1;
            }
        }
        SimAdviceOption? best = advice.Options.Find(o => o.IsBest);
        if (best != null && !order.ContainsKey(best))
        {
            order[best] = 1;
        }
        return order;
    }

    /// <summary>What kind of move a row is, for its colour chip.</summary>
    private enum RowKind
    {
        Attack,
        Skill,
        Power,
        Potion,
        EndTurn,
        Pick,
        Skip,
        Other,
    }

    private RowKind KindOf(SimAdviceOption o)
    {
        switch (o.Action)
        {
            case "end_turn":
                return RowKind.EndTurn;
            case "potion":
                return RowKind.Potion;
            case "skip":
                return RowKind.Skip;
        }
        if (o.CardIndex is int ci && _latestPlayer != null)
        {
            try
            {
                CardPile hand = PileType.Hand.GetPile(_latestPlayer);
                if (ci >= 0 && ci < hand.Cards.Count)
                {
                    return hand.Cards[ci].Type switch
                    {
                        CardType.Attack => RowKind.Attack,
                        CardType.Skill => RowKind.Skill,
                        CardType.Power => RowKind.Power,
                        _ => RowKind.Other,
                    };
                }
            }
            catch (Exception)
            {
                // fall through
            }
        }
        string label = o.Label.Trim().ToLowerInvariant();
        if (label is "end turn" || label.StartsWith("end turn", StringComparison.Ordinal))
        {
            return RowKind.EndTurn;
        }
        if (label is "skip" or "leave" or "leave shop" or "decline" || label.StartsWith("skip", StringComparison.Ordinal))
        {
            return RowKind.Skip;
        }
        return o.Action == "choose" || o.Index != null ? RowKind.Pick : RowKind.Other;
    }

    /// <summary>The game's own palette for card types: red attacks, blue
    /// skills, gold powers; potions green; pass-moves grey.</summary>
    private static Color KindColor(RowKind kind) => kind switch
    {
        RowKind.Attack => new Color(0.86f, 0.32f, 0.26f),
        RowKind.Skill => new Color(0.30f, 0.56f, 0.92f),
        RowKind.Power => new Color(0.92f, 0.70f, 0.25f),
        RowKind.Potion => new Color(0.32f, 0.78f, 0.52f),
        RowKind.EndTurn or RowKind.Skip => new Color(0.52f, 0.53f, 0.58f),
        RowKind.Pick => new Color(0.40f, 0.74f, 0.80f),
        _ => new Color(0.6f, 0.6f, 0.66f),
    };

    private static Color VerdictColor(string verdict) => verdict switch
    {
        "Best" => new Color(1f, 0.82f, 0.3f),
        "Also good" => new Color(0.5f, 0.88f, 0.55f),
        "Then" => new Color(1f, 0.9f, 0.62f),
        "Okay" => new Color(0.85f, 0.85f, 0.7f),
        "Weak" => new Color(0.98f, 0.66f, 0.3f),
        "Mistake" => new Color(0.95f, 0.38f, 0.32f),
        _ => new Color(0.6f, 0.6f, 0.65f),
    };

    /// <summary>
    /// One ranked row: a chip in the move's colour carrying its step in the
    /// plan (the same number as the card's badge), the move in words, wrapped
    /// rather than cut ("Strike → Sludge ..." hid which enemy, reported
    /// 2026-11-07), a verdict word, the expected HP when the search has it,
    /// and how much of the AI's attention it got. The engine's own columns
    /// (Q delta, W/L/T, visits) only with "Show engine numbers".
    /// </summary>
    private Control BuildOptionRow(SimAdviceOption o, double? bestQ, int? step, bool hpColumn)
    {
        var row = new HBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        row.AddThemeConstantOverride("separation", 6);

        // Chip: kind colour, plan step inside.
        RowKind kind = KindOf(o);
        Color kc = KindColor(kind);
        var chipStyle = new StyleBoxFlat
        {
            BgColor = step != null ? kc : new Color(kc.R, kc.G, kc.B, 0.28f),
            BorderColor = kc,
            AntiAliasing = true,
        };
        chipStyle.SetBorderWidthAll(1);
        chipStyle.SetCornerRadiusAll(9);
        var chip = new PanelContainer
        {
            CustomMinimumSize = new Vector2(22, 18),
            SizeFlagsVertical = Control.SizeFlags.ShrinkCenter,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        chip.AddThemeStyleboxOverride("panel", chipStyle);
        Label chipText = SmallLabel(step?.ToString(CultureInfo.InvariantCulture) ?? "", new Color(0.06f, 0.05f, 0.04f));
        chipText.AddThemeConstantOverride("outline_size", 0);
        chipText.AddThemeFontSizeOverride("font_size", 13);
        chipText.HorizontalAlignment = HorizontalAlignment.Center;
        chip.AddChild(chipText);
        row.AddChild(chip);

        // The move, in words, wrapping instead of clipping.
        Label name = SmallLabel(Words(o.Label), o.IsBest ? new Color(1f, 0.85f, 0.3f) : Colors.White);
        name.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        name.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        name.CustomMinimumSize = new Vector2(120, 0);
        row.AddChild(name);

        MarkOption mark = o.ToMark();
        string verdict = AdviceMarks.RowVerdict(mark, bestQ, step);
        Label v = SmallLabel(verdict, VerdictColor(verdict));
        v.CustomMinimumSize = new Vector2(72, 0);
        v.HorizontalAlignment = HorizontalAlignment.Right;
        row.AddChild(v);

        if (hpColumn)
        {
            Label hp = SmallLabel(o.HpMean is double h ? $"♥ {h.ToString("0", CultureInfo.InvariantCulture)}" : "",
                new Color(0.95f, 0.5f, 0.5f));
            hp.CustomMinimumSize = new Vector2(46, 0);
            hp.HorizontalAlignment = HorizontalAlignment.Right;
            row.AddChild(hp);
        }

        double? q = o.Q ?? (o.PWin != null ? o.PWin.Value * 2 - 1 : (double?)null);
        if (SpireMentorModConfig.ShowNumbers && q != null)
        {
            // The best row carries the absolute eval; the rest what they COST
            // against it (two decimals of an absolute Q all read +1.00 once
            // the position is winning -- reported 2026-08-05).
            bool delta = !o.IsBest && bestQ != null;
            double shown = delta ? q.Value - bestQ!.Value : q.Value;
            string text = delta
                ? (shown >= 0 ? "+" : "−") + Math.Abs(shown).ToString("0.000", CultureInfo.InvariantCulture)
                : (shown >= 0 ? "+" : "") + shown.ToString("0.00", CultureInfo.InvariantCulture);
            Label qLabel = SmallLabel(text, delta ? DeltaColor(shown) : QColor(q.Value));
            qLabel.CustomMinimumSize = new Vector2(50, 0);
            qLabel.HorizontalAlignment = HorizontalAlignment.Right;
            row.AddChild(qLabel);
        }

        if (o.Share is double rawShare)
        {
            double share = Math.Clamp(rawShare, 0.0, 1.0);
            var shareBg = new ColorRect
            {
                Color = new Color(0.2f, 0.2f, 0.2f),
                CustomMinimumSize = new Vector2(44, 6),
                SizeFlagsVertical = Control.SizeFlags.ShrinkCenter,
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            var shareFill = new ColorRect
            {
                Color = o.IsBest ? new Color(1f, 0.8f, 0.3f) : new Color(kc.R, kc.G, kc.B, 0.9f),
                AnchorRight = (float)share,
                AnchorBottom = 1f,
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            shareBg.AddChild(shareFill);
            row.AddChild(shareBg);

            Label shareLabel = SmallLabel(
                share < 0.005 ? "<1%" : share.ToString("0%", CultureInfo.InvariantCulture), new Color(0.72f, 0.72f, 0.74f));
            shareLabel.CustomMinimumSize = new Vector2(36, 0);
            shareLabel.HorizontalAlignment = HorizontalAlignment.Right;
            row.AddChild(shareLabel);
        }

        if (SpireMentorModConfig.ShowNumbers)
        {
            if (o.Wins != null || o.Losses != null || o.Ties != null)
            {
                row.AddChild(SmallLabel($"{o.Wins ?? 0}/{o.Losses ?? 0}/{o.Ties ?? 0}",
                    new Color(0.55f, 0.55f, 0.55f)));
            }
            else if (o.Visits != null)
            {
                row.AddChild(SmallLabel($"{o.Visits}v", new Color(0.6f, 0.6f, 0.6f)));
            }
            if (o.LowConfidence)
            {
                row.AddChild(SmallLabel("~", new Color(0.95f, 0.75f, 0.3f)));
            }
        }

        return row;
    }

    /// <summary>
    /// "… N more" / "show fewer": the one clickable control in the otherwise
    /// click-through panel (a Stop-filter button, SimUiKit's input-safety
    /// contract). Re-renders the panel from the advice it last showed.
    /// </summary>
    private Control BuildRankedToggle(int hidden)
    {
        string text = _rankedExpanded ? "▴ show fewer" : $"… {hidden} more ▾";
        Button toggle = SimUiKit.MakeButton(text, "", () =>
        {
            _rankedExpanded = !_rankedExpanded;
            if (_panelAdvice != null)
            {
                RenderAdvicePanel(_panelAdvice);
            }
        });
        toggle.SizeFlagsHorizontal = Control.SizeFlags.ShrinkBegin;
        return toggle;
    }

    /// <summary>Empty the in-game advice panel, and the frame's.</summary>
    private void ClearAdvicePanel()
    {
        FrameLink.PublishAdvice(null);
        ClearAdvicePanelRows();
    }

    /// <summary>Empty the in-game advice panel only: for a caller about to
    /// publish the replacement, so the frame never flashes "No advice" in
    /// between (and the wire carries one message, not two).</summary>
    private void ClearAdvicePanelRows()
    {
        if (_adviceRows == null || !GodotObject.IsInstanceValid(_adviceRows))
        {
            return;
        }
        // Detach before freeing: a queued-free child stays in the tree (and in
        // the container's minimum size) until the end of the frame.
        foreach (Node child in _adviceRows.GetChildren())
        {
            _adviceRows.RemoveChild(child);
            child.QueueFreeSafely();
        }
    }

    /// <summary>
    /// Containers grow to fit their children but never shrink back on their
    /// own, so "show fewer" left the panel at its expanded height. Zeroing the
    /// size clamps it to the new minimum; deferred so it runs after the rows
    /// AddChildSafely may itself have deferred.
    /// </summary>
    private void ShrinkAdvicePanelToFit()
    {
        PanelContainer? panel = _advicePanel;
        Callable.From(() =>
        {
            if (panel != null && GodotObject.IsInstanceValid(panel))
            {
                panel.Size = Vector2.Zero;
            }
        }).CallDeferred();
    }

    // ------------------------------------------------------------------
    // Per-card labels (idioms carried over from SimOverlay verbatim)
    // ------------------------------------------------------------------

    private void RenderCardLabels(SimAdvice advice, Player? player)
    {
        ClearCardLabels();

        // The engine's numbers over the cards are opt-in now ("Show engine
        // numbers"): the rings and step numbers (SimHighlights) say what to
        // play, and a "+0.42 63%" over every card said it in units nobody
        // reads (reported 2026-11-07).
        if (!SpireMentorModConfig.ShowNumbers || !HudVisible)
        {
            return;
        }
        if (player == null || !CombatManager.Instance.IsInProgress)
        {
            return;
        }
        NPlayerHand? handUi = NPlayerHand.Instance;
        if (handUi == null || !GodotObject.IsInstanceValid(handUi))
        {
            return;
        }

        CardPile hand;
        try
        {
            hand = PileType.Hand.GetPile(player);
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Could not resolve hand pile: {ex.Message}");
            return;
        }

        // ONE label per card: a targeted card has an option per target, and
        // a label for each stacked two numbers on one card, unreadable
        // (reported 2026-11-07). The card's label is its best option's. A
        // move nobody would make ("0%") gets none.
        var perCard = new Dictionary<int, SimAdviceOption>();
        foreach (SimAdviceOption o in advice.Options)
        {
            if (o.CardIndex is not int ci || AdviceMarks.Hidden(o.ToMark()))
            {
                continue;
            }
            if (!perCard.TryGetValue(ci, out SimAdviceOption? have) ||
                (o.IsBest && !have.IsBest) ||
                (o.IsBest == have.IsBest && (o.Q ?? o.Share ?? o.Score ?? 0) > (have.Q ?? have.Share ?? have.Score ?? 0)))
            {
                perCard[ci] = o;
            }
        }
        foreach (SimAdviceOption o in perCard.Values)
        {
            try
            {
                RenderOneCardLabel(o, hand, handUi);
            }
            catch (Exception ex)
            {
                Logger.Log($"[SimHud] Error labeling card {o.CardIndex}: {ex.Message}");
            }
        }
    }

    private void RenderOneCardLabel(SimAdviceOption o, CardPile hand, NPlayerHand handUi)
    {
        int index = o.CardIndex!.Value;
        // Same hand indexing as the "play" verb's card_index -- Nth card of
        // PileType.Hand, NOT a UI holder position.
        if (index < 0 || index >= hand.Cards.Count)
        {
            return;
        }
        CardModel cardModel = hand.Cards[index];
        NCard? nCard = handUi.GetCard(cardModel);
        if (nCard == null || !GodotObject.IsInstanceValid(nCard))
        {
            return;
        }

        // Q + search share, not p_win: the same saturation that flattened the
        // ranked list flattens these labels ("100%" over every card).
        double? q = o.Q ?? (o.Score != null ? o.Score.Value * 2 - 1 : (double?)null);
        double score = q != null ? Math.Clamp((q.Value + 1.0) / 2.0, 0.0, 1.0)
                                 : Math.Clamp(o.Score ?? o.PWin ?? 0.5, 0.0, 1.0);

        string text;
        if (q != null)
        {
            text = (q.Value >= 0 ? "+" : "") + q.Value.ToString("0.00", CultureInfo.InvariantCulture);
            if (o.VisitShare != null)
            {
                text += "  " + o.VisitShare.Value.ToString("0%", CultureInfo.InvariantCulture);
            }
        }
        else if (o.VisitShare != null)
        {
            text = o.VisitShare.Value.ToString("0%", CultureInfo.InvariantCulture);
        }
        else
        {
            text = o.Label;
        }
        if (o.IsBest)
        {
            text = "★ " + text;
        }

        const float labelWidth = 90f;
        const float labelHeight = 22f;
        var label = new Label
        {
            Name = $"SpireMentorCardLabel_{index}",
            Text = text,
            Size = new Vector2(labelWidth, labelHeight),
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        label.AddThemeColorOverride("font_color", o.IsBest ? new Color(1f, 0.85f, 0.3f) : ScoreColor(score));
        label.AddThemeColorOverride("font_outline_color", Colors.Black);
        label.AddThemeConstantOverride("outline_size", o.IsBest ? 6 : 4);

        _layer!.AddChildSafely(label);

        var anchor = new CardLabelAnchor
        {
            Label = label,
            Model = cardModel,
            Card = nCard,
            Width = labelWidth,
            Height = labelHeight,
        };
        PositionCardLabel(anchor);
        _cardLabels.Add(anchor);
    }

    /// <summary>
    /// Put the label just above the card's VISIBLE top edge.
    ///
    /// `GlobalPosition + size/2` is not that point: hand cards are scaled
    /// (NCard.GetCurrentSize is `defaultSize * Scale`) and rotated in the fan,
    /// and Control.GlobalPosition ignores both, so the label drifted off the
    /// card the moment the hand animated. GetGlobalTransformWithCanvas maps a
    /// point in the card's own space to final viewport pixels -- scale,
    /// rotation, pivot and canvas transform included -- which is exactly the
    /// space this CanvasLayer draws in.
    ///
    /// What that transform is applied TO is the part this used to get wrong.
    /// Measured live at 1920x1080 with an identity canvas transform
    /// (2026-08-06), one hand card:
    ///
    ///   %Frame.Size=(300,422)   NCard.Size=(0,0)   NCard.Scale=(1,1)
    ///   face rect = (840, 861.2) size (240, 337.6)   -> centre x 960
    ///   label     = (915, 833.2)                     -> 28 px above, centred
    ///
    /// `NCard.Size` is ZERO -- the game does not trust it either, defining
    /// `GetCurrentSize()` as `defaultSize * Scale` and ignoring `Size`
    /// outright (NCard.cs:830) -- and the card's local ORIGIN is its CENTRE,
    /// not its top-left. So the old `xf * (defaultSize.X/2, 0)` resolved to
    /// "the card's centre, shifted 150 local units right", putting the label
    /// at y~1002 (the card's vertical middle, the face being 337.6 tall) and
    /// x~1035 (120 px right of centre). Exactly the reported symptom, and the
    /// reason the old zero-size fallback did not save it: falling back to
    /// `defaultSize` fixes the EXTENT but not the ORIGIN.
    ///
    /// Measure %Frame instead -- it IS the face, and being a TextureRect its
    /// local rect really does start at (0,0) -- and take the transformed
    /// bounding box of its four corners so the fan's rotation cannot tilt the
    /// label off the card.
    /// </summary>
    private static void PositionCardLabel(CardLabelAnchor anchor)
    {
        NCard? card = anchor.Card;
        if (card == null || !GodotObject.IsInstanceValid(card))
        {
            return;
        }
        Rect2? face = CardFaceRect(card);
        if (face == null)
        {
            return;
        }
        Rect2 rect = face.Value;
        anchor.Label.GlobalPosition = new Vector2(
            rect.Position.X + rect.Size.X / 2f - anchor.Width / 2f,
            rect.Position.Y - anchor.Height - 6f);
    }

    /// <summary>
    /// The card's drawn face as an axis-aligned box in viewport pixels.
    /// </summary>
    private static Rect2? CardFaceRect(NCard card)
    {
        try
        {
            // %Frame is the card's border texture: a plain TextureRect, so its
            // local rect genuinely starts at (0,0) and genuinely spans the
            // face. Ancient cards swap it for their own border, hence the
            // visibility test and the design-size fallback.
            Control? frame = card.GetNodeOrNull<Control>("%Frame");
            bool useFrame = frame != null && GodotObject.IsInstanceValid(frame) &&
                            frame.IsVisibleInTree() && frame.Size.X > 0f && frame.Size.Y > 0f;
            Control source = useFrame ? frame! : card;
            Vector2 size = useFrame ? frame!.Size : NCard.defaultSize;

            Transform2D xf = source.GetGlobalTransformWithCanvas();
            Vector2 a = xf * Vector2.Zero;
            Vector2 b = xf * new Vector2(size.X, 0f);
            Vector2 c = xf * new Vector2(0f, size.Y);
            Vector2 d = xf * size;
            float minX = Math.Min(Math.Min(a.X, b.X), Math.Min(c.X, d.X));
            float minY = Math.Min(Math.Min(a.Y, b.Y), Math.Min(c.Y, d.Y));
            float maxX = Math.Max(Math.Max(a.X, b.X), Math.Max(c.X, d.X));
            float maxY = Math.Max(Math.Max(a.Y, b.Y), Math.Max(c.Y, d.Y));
            if (maxX <= minX || maxY <= minY)
            {
                return null;
            }
            return new Rect2(minX, minY, maxX - minX, maxY - minY);
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Re-resolve and re-anchor every card label, each frame-ish. Two things
    /// move under us: the hand re-lays itself out continuously (playing a card
    /// slides the rest across, hovering lifts one), and NCard nodes are POOLED
    /// -- so the node that showed this card a moment ago may now be showing
    /// somebody else's. Asking NPlayerHand for the node of THIS CardModel
    /// every tick is immune to both; a card that has left the hand simply has
    /// no node, and its label hides.
    /// </summary>
    private void RepositionCardLabels()
    {
        if (_cardLabels.Count == 0)
        {
            return;
        }
        try
        {
            NPlayerHand? handUi = NPlayerHand.Instance;
            bool handAlive = handUi != null && GodotObject.IsInstanceValid(handUi);
            foreach (CardLabelAnchor anchor in _cardLabels)
            {
                if (!GodotObject.IsInstanceValid(anchor.Label))
                {
                    continue;
                }
                NCard? card = null;
                if (handAlive && anchor.Model != null)
                {
                    try
                    {
                        card = handUi!.GetCard(anchor.Model);
                    }
                    catch (Exception)
                    {
                        card = null;
                    }
                }
                anchor.Card = card;
                bool alive = card != null && GodotObject.IsInstanceValid(card) &&
                             card.IsInsideTree() && card.IsVisibleInTree();
                anchor.Label.Visible = alive && HudVisible;
                if (alive)
                {
                    PositionCardLabel(anchor);
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Card-label reposition failed: {ex.Message}");
        }
    }

    private void ClearCardLabels()
    {
        try
        {
            foreach (CardLabelAnchor anchor in _cardLabels)
            {
                if (GodotObject.IsInstanceValid(anchor.Label))
                {
                    anchor.Label.QueueFreeSafely();
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Error clearing card labels: {ex.Message}");
        }
        finally
        {
            _cardLabels.Clear();
        }
    }

    // ------------------------------------------------------------------
    // Map markers + projected route
    // ------------------------------------------------------------------

    /// <summary>
    /// Pin the advice to the map itself: a score on every node the player may
    /// click, a star on the recommended one, and the projected route drawn
    /// through the act as a line with numbered stops.
    ///
    /// Without this the map advice was a list of graph coordinates in a
    /// corner panel, next to a map that shows no coordinates -- correct and
    /// unusable (reported 2026-08-05). Nodes are matched by coordinate, not
    /// by list position, because the advice may have been computed a moment
    /// before the screen finished laying itself out.
    /// </summary>
    private void RenderMapMarkers(SimAdvice advice)
    {
        ClearMapMarkers();

        if (!SpireMentorModConfig.ShowMapMarkers || !HudVisible)
        {
            return;
        }
        NMapScreen? map = NMapScreen.Instance;
        if (map == null || !GodotObject.IsInstanceValid(map) || !map.IsOpen || !map.IsVisibleInTree())
        {
            return;
        }

        Dictionary<(int, int), NMapPoint> byCoord = new();
        foreach (NMapPoint mp in UiHelper.FindAll<NMapPoint>(map))
        {
            if (!GodotObject.IsInstanceValid(mp))
            {
                continue;
            }
            byCoord[(mp.Point.coord.col, mp.Point.coord.row)] = mp;
        }
        if (byCoord.Count == 0)
        {
            return;
        }

        // The route first, so the per-node scores draw on top of the line.
        RenderRoute(advice, byCoord);

        foreach (SimAdviceOption o in advice.Options)
        {
            // A path nobody would take gets no "0%" (live, 2026-11-07).
            if (o.Col == null || o.Row == null || AdviceMarks.Hidden(o.ToMark()) ||
                !byCoord.TryGetValue((o.Col.Value, o.Row.Value), out NMapPoint? point))
            {
                continue;
            }
            string text = o.VisitShare != null
                ? o.VisitShare.Value.ToString("0%", CultureInfo.InvariantCulture)
                : o.Label;
            if (o.IsBest)
            {
                text = "★ " + text;
            }
            AddMapMarker(point, text,
                o.IsBest ? new Color(1f, 0.85f, 0.3f) : new Color(0.75f, 0.8f, 0.9f),
                o.IsBest ? 6 : 4, dy: -52f);
        }
    }

    /// <summary>How many hops past the one being decided get a number. The
    /// route spans a whole act now, so the rest stay unlabeled.</summary>
    private const int RouteNumberedHops = 3;

    private void RenderRoute(SimAdvice advice, Dictionary<(int, int), NMapPoint> byCoord)
    {
        _routePoints.Clear();
        if (advice.Path.Count < 2)
        {
            return;
        }
        var routeColor = new Color(1f, 0.85f, 0.3f, 0.85f);
        SimPathNode? lastDrawn = null;
        NMapPoint? lastPoint = null;
        foreach (SimPathNode step in advice.Path)
        {
            if (!byCoord.TryGetValue((step.Col, step.Row), out NMapPoint? point))
            {
                continue;
            }
            _routePoints.Add(point);
            lastDrawn = step;
            lastPoint = point;
            // Hop 0 is the node being decided now and already carries the star
            // + score. Beyond it, number only the next few: the route runs the
            // whole act now (~15 rows), and a number on every node turned the
            // map into a blob of digits.
            if (step.Hop > 0 && step.Hop <= RouteNumberedHops)
            {
                AddMapMarker(point, step.Hop.ToString(CultureInfo.InvariantCulture),
                    routeColor, 4, dy: 34f);
            }
        }
        if (_routePoints.Count < 2)
        {
            _routePoints.Clear();
            return;
        }
        // The far end carries the projected arrival HP -- the number a player
        // reads a route for. Anchored to the last node actually DRAWN, not the
        // last one advised: a node the map screen no longer shows was skipped
        // above and has nowhere to hang a label.
        if (lastDrawn?.Hp != null && lastDrawn.Hop > RouteNumberedHops && lastPoint != null)
        {
            AddMapMarker(lastPoint,
                lastDrawn.Hp.Value.ToString(CultureInfo.InvariantCulture) + " HP",
                routeColor, 4, dy: 34f);
        }

        var line = new Line2D
        {
            Name = "SpireMentorRoute",
            Width = 5f,
            DefaultColor = new Color(1f, 0.85f, 0.3f, 0.6f),
            JointMode = Line2D.LineJointMode.Round,
            BeginCapMode = Line2D.LineCapMode.Round,
            EndCapMode = Line2D.LineCapMode.Round,
            Antialiased = true,
        };
        // Drawn INSIDE the map, behind its nodes: it scrolls with the map and
        // passes under the game's top bar like the map's own paths do. On the
        // HUD layer it ran over the top bar (live, 2026-11-07).
        Node? host = _routePoints[0].GetParent();
        if (host is CanvasItem)
        {
            host.AddChildSafely(line);
            Callable.From(() =>
            {
                if (GodotObject.IsInstanceValid(line) && line.GetParent() == host)
                {
                    host.MoveChild(line, 0);
                }
            }).CallDeferred();
        }
        else
        {
            _layer!.AddChildSafely(line);
        }
        _routeLine = line;
        UpdateRouteLine();
    }

    private void AddMapMarker(Control point, string text, Color color, int outline, float dy)
    {
        const float markerWidth = 76f;
        const float markerHeight = 22f;
        var label = new Label
        {
            Name = "SpireMentorMapMarker",
            Text = text,
            Size = new Vector2(markerWidth, markerHeight),
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", Colors.Black);
        label.AddThemeConstantOverride("outline_size", outline);
        _layer!.AddChildSafely(label);

        var anchor = new MapMarkerAnchor
        {
            Label = label,
            Point = point,
            Width = markerWidth,
            Height = markerHeight,
            DY = dy,
        };
        PositionMapMarker(anchor);
        _mapMarkers.Add(anchor);
    }

    /// <summary>Centre of a map node in viewport pixels, or null if it is gone.</summary>
    private static Vector2? MapPointCentre(Control point)
    {
        // IsVisibleInTree, not just IsInsideTree: leaving the map HIDES its
        // nodes, it does not free them (the same fact that keeps a spent
        // screen on the overlay stack). Without this the coordinates stay
        // valid after the map is gone, so the route line kept drawing itself
        // over the next combat for the few seconds until something else
        // cleared the advice. The per-node labels never had the bug -- they
        // test IsVisibleInTree separately in RepositionMapMarkers -- which is
        // why only the yellow line lingered.
        //
        // Feeding null back means UpdateRouteLine collects fewer than two
        // points and hides the line on the next 50 ms tick.
        if (!GodotObject.IsInstanceValid(point) || !point.IsInsideTree() ||
            !point.IsVisibleInTree())
        {
            return null;
        }
        Vector2 size = point.Size;
        return point.GetGlobalTransformWithCanvas() * (size / 2f);
    }

    private static void PositionMapMarker(MapMarkerAnchor anchor)
    {
        Vector2? centre = MapPointCentre(anchor.Point);
        if (centre == null)
        {
            return;
        }
        anchor.Label.GlobalPosition = centre.Value -
            new Vector2(anchor.Width / 2f, anchor.Height / 2f) + new Vector2(0f, anchor.DY);
    }

    /// <summary>
    /// Re-run every map anchor. The map screen pans and zooms under the
    /// player's mouse, so these move for the same reason the hand labels do.
    /// </summary>
    private void RepositionMapMarkers()
    {
        if (_mapMarkers.Count == 0 && _routeLine == null)
        {
            return;
        }
        try
        {
            foreach (MapMarkerAnchor anchor in _mapMarkers)
            {
                if (!GodotObject.IsInstanceValid(anchor.Label))
                {
                    continue;
                }
                bool alive = GodotObject.IsInstanceValid(anchor.Point) &&
                             anchor.Point.IsInsideTree() &&
                             anchor.Point.IsVisibleInTree();
                anchor.Label.Visible = alive && HudVisible;
                if (alive)
                {
                    PositionMapMarker(anchor);
                    // Scrolled under the game's top bar: hidden, as the map's
                    // own icons are there (live, 2026-11-07).
                    if (anchor.Label.GlobalPosition.Y < SimTopBarLayout.SafeTopY(anchor.Label) - 8f)
                    {
                        anchor.Label.Visible = false;
                    }
                }
            }
            UpdateRouteLine();
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Map-marker reposition failed: {ex.Message}");
        }
    }

    private void UpdateRouteLine()
    {
        if (_routeLine == null || !GodotObject.IsInstanceValid(_routeLine))
        {
            return;
        }
        var points = new List<Vector2>(_routePoints.Count);
        // In the line's own space: the map's (it lives in the map's tree), or
        // the viewport's when it fell back to the HUD layer.
        bool inMap = _routeLine.GetParent() is CanvasItem;
        Transform2D toLocal = inMap ? _routeLine.GetGlobalTransform().AffineInverse() : Transform2D.Identity;
        foreach (Control point in _routePoints)
        {
            if (!GodotObject.IsInstanceValid(point) || !point.IsInsideTree())
            {
                continue;
            }
            Vector2? centre = inMap
                ? toLocal * (point.GetGlobalTransform() * (point.Size / 2f))
                : MapPointCentre(point);
            if (centre != null)
            {
                points.Add(centre.Value);
            }
        }
        _routeLine.Visible = HudVisible && points.Count >= 2;
        _routeLine.Points = points.ToArray();
    }

    private void ClearMapMarkers()
    {
        try
        {
            foreach (MapMarkerAnchor anchor in _mapMarkers)
            {
                if (GodotObject.IsInstanceValid(anchor.Label))
                {
                    anchor.Label.QueueFreeSafely();
                }
            }
            if (_routeLine != null && GodotObject.IsInstanceValid(_routeLine))
            {
                _routeLine.QueueFreeSafely();
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Error clearing map markers: {ex.Message}");
        }
        finally
        {
            _mapMarkers.Clear();
            _routePoints.Clear();
            _routeLine = null;
        }
    }

    // ------------------------------------------------------------------
    // Enemy tags: "#2" beside the health bar of each same-named enemy
    // ------------------------------------------------------------------

    /// <summary>
    /// Keep a "#k" tag beside every enemy whose name it shares with another,
    /// numbered exactly as the advisor numbers its "→ Name #k" targets
    /// (`EnemyTagText`). Independent of advice: the tag is what tells the
    /// player which rat a label means, so it stays up while the HUD is.
    /// Rebuilt every tick because enemies spawn, die and walk mid-fight.
    /// </summary>
    private void RepositionEnemyTags()
    {
        try
        {
            NCombatRoom? room = NCombatRoom.Instance;
            IReadOnlyList<Creature>? enemies =
                CombatManager.Instance?.IsInProgress == true &&
                room != null && GodotObject.IsInstanceValid(room)
                    ? CombatManager.Instance.DebugOnlyGetState()?.Enemies
                    : null;
            if (enemies == null || enemies.Count < 2)
            {
                ClearEnemyTags();
                return;
            }

            var names = new List<string?>(enemies.Count);
            foreach (Creature e in enemies)
            {
                string? name = null;
                try { name = e.IsAlive ? e.Name ?? "?" : null; } catch (Exception) { name = "?"; }
                names.Add(name);
            }
            string?[] tags = EnemyTagText.Tags(names);

            var live = new HashSet<Creature>();
            for (int i = 0; i < enemies.Count; i++)
            {
                Creature enemy = enemies[i];
                Rect2? bar = null;
                if (tags[i] != null)
                {
                    // Per enemy, so one creature whose node lookup throws
                    // loses only its own tag, not every enemy's.
                    try { bar = HealthBarRect(room!.GetCreatureNode(enemy)); }
                    catch (Exception ex) { LogEnemyTagFailure(ex); }
                }
                if (bar == null)
                {
                    continue;
                }
                live.Add(enemy);
                if (!_enemyTags.TryGetValue(enemy, out Label? label) || !GodotObject.IsInstanceValid(label))
                {
                    label = SmallLabel("", new Color(1f, 0.85f, 0.3f));
                    label.Name = "SpireMentorEnemyTag";
                    label.AddThemeColorOverride("font_outline_color", Colors.Black);
                    label.AddThemeConstantOverride("outline_size", 6);
                    label.AddThemeFontSizeOverride("font_size", 22);
                    label.HorizontalAlignment = HorizontalAlignment.Right;
                    label.VerticalAlignment = VerticalAlignment.Center;
                    _layer!.AddChildSafely(label);
                    _enemyTags[enemy] = label;
                }
                label.Text = tags[i];
                label.Size = new Vector2(44f, bar.Value.Size.Y + 8f);
                label.GlobalPosition = new Vector2(
                    bar.Value.Position.X - label.Size.X - 6f,
                    bar.Value.Position.Y + bar.Value.Size.Y / 2f - label.Size.Y / 2f);
                label.Visible = HudVisible;
            }

            foreach (Creature gone in new List<Creature>(_enemyTags.Keys))
            {
                if (!live.Contains(gone))
                {
                    FreeEnemyTag(gone);
                }
            }
        }
        catch (Exception ex)
        {
            LogEnemyTagFailure(ex);
        }
    }

    private string? _lastEnemyTagFailure;

    /// <summary>
    /// <see cref="RepositionEnemyTags"/> runs on the 0.05 s ticker, so a
    /// failure that persists would log 20 lines a second for the whole fight.
    /// Log each distinct message once, until a different one replaces it.
    /// </summary>
    private void LogEnemyTagFailure(Exception ex)
    {
        if (ex.Message == _lastEnemyTagFailure)
        {
            return;
        }
        _lastEnemyTagFailure = ex.Message;
        Logger.Log($"[SimHud] Enemy-tag reposition failed: {ex.Message}");
    }

    /// <summary>
    /// The enemy's drawn health bar in viewport pixels, or null when it has
    /// none on screen (dying, hidden bar, not yet laid out). Same transform
    /// trick as <see cref="CardFaceRect"/>: the room sits under a canvas
    /// transform this layer does not share.
    /// </summary>
    private static Rect2? HealthBarRect(NCreature? node)
    {
        if (node == null || !GodotObject.IsInstanceValid(node) || !node.IsVisibleInTree())
        {
            return null;
        }
        // %HealthBar on the creature is its state display; the bar itself is
        // that display's own %HealthBar.
        Control? display = node.GetNodeOrNull<Control>("%HealthBar");
        Control? bar = display?.GetNodeOrNull<Control>("%HealthBar") ?? display;
        if (bar == null || !GodotObject.IsInstanceValid(bar) || !bar.IsVisibleInTree() ||
            bar.Size.X <= 0f || bar.Size.Y <= 0f)
        {
            return null;
        }
        Transform2D xf = bar.GetGlobalTransformWithCanvas();
        Vector2 a = xf * Vector2.Zero;
        Vector2 d = xf * bar.Size;
        return new Rect2(Math.Min(a.X, d.X), Math.Min(a.Y, d.Y), Math.Abs(d.X - a.X), Math.Abs(d.Y - a.Y));
    }

    private void FreeEnemyTag(Creature enemy)
    {
        if (_enemyTags.Remove(enemy, out Label? label) && GodotObject.IsInstanceValid(label))
        {
            label.QueueFreeSafely();
        }
    }

    private void ClearEnemyTags()
    {
        foreach (Creature enemy in new List<Creature>(_enemyTags.Keys))
        {
            FreeEnemyTag(enemy);
        }
    }

    // ------------------------------------------------------------------
    // Layer / panel construction
    // ------------------------------------------------------------------

    private bool EnsureLayer()
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
                Name = "SpireMentorHudLayer",
                Layer = 99,
            };

            // Advice panel (below the toolbar strip, docked or dragged -- see
            // RepositionPanels).
            var panel = new PanelContainer
            {
                Name = "SpireMentorAdvicePanel",
                Position = new Vector2(PanelLeft, ChipFallbackTop + ChipToPanelGap),
                CustomMinimumSize = new Vector2(PanelWidth, 0f),
                MouseFilter = Control.MouseFilterEnum.Ignore,
                Visible = false,
            };
            panel.AddThemeStyleboxOverride("panel", PanelStyle());
            var rows = new VBoxContainer
            {
                Name = "AdviceRows",
                MouseFilter = Control.MouseFilterEnum.Ignore,
            };
            rows.AddThemeConstantOverride("separation", 4);
            panel.AddChild(rows);
            layer.AddChild(panel);

            // Drives RepositionCardLabels; the hand moves continuously, and
            // this project has no source generators so there is no _Process
            // override available (see the file header).
            var ticker = new Godot.Timer
            {
                Name = "SpireMentorHudTicker",
                WaitTime = 0.05,
                Autostart = true,
                OneShot = false,
            };
            ticker.Connect(Godot.Timer.SignalName.Timeout, Callable.From(() =>
            {
                RepositionPanels();
                RepositionCardLabels();
                RepositionMapMarkers();
                RepositionEnemyTags();
            }));
            layer.AddChild(ticker);

            tree.Root.AddChildSafely(layer);

            _layer = layer;
            _advicePanel = panel;
            _adviceRows = rows;
            // One-time liveness confirmation -- everything else only logs
            // errors, which makes "working" and "never invoked"
            // indistinguishable in godot.log.
            Logger.Log("[SimHud] HUD layer created");
            return true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Failed to create HUD layer: {ex.Message}");
            _layer = null;
            _advicePanel = null;
            _adviceRows = null;
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Panel placement
    // ------------------------------------------------------------------

    private const float PanelLeft = 16f;

    /// <summary>Chip Y when no top bar is on screen (main menu, character select).</summary>
    private const float ChipFallbackTop = SimTopBarLayout.FallbackTop;

    /// <summary>Chip bottom to advice-panel top. The two used to be 16 and 96.</summary>
    private const float ChipToPanelGap = 80f;

    /// <summary>Advice panel width (its minimum; a coach chip is as wide).</summary>
    private const float PanelWidth = 470f;

    /// <summary>
    /// Left edge and bottom of the advice column -- the advice panel and the
    /// coach chips under it -- which the action log sits under until the
    /// player drags it (LogPanelPlacement). Viewport pixels. Main thread.
    /// </summary>
    public (float Left, float Bottom) ColumnBottom()
    {
        Vector2 slot = ChipSlot();
        float chips = SimCoachChip.Instance.StackHeight();
        return (slot.X, slot.Y + chips);
    }

    /// <summary>
    /// Where the coach chip (SimCoachChip) goes: just below the advice panel
    /// when it is showing, else in the panel's own slot (under the toolbar,
    /// wherever the toolbar is). Viewport pixels. Main thread.
    /// </summary>
    public Vector2 ChipSlot()
    {
        try
        {
            if (_advicePanel != null && GodotObject.IsInstanceValid(_advicePanel))
            {
                float top = _advicePanel.Position.Y;
                if (_advicePanel.IsVisibleInTree())
                {
                    top += _advicePanel.Size.Y + AdviceColumn.Gap;
                }
                return new Vector2(_advicePanel.Position.X, top);
            }
            if (SimToolbar.Instance.UndockedRect() is { } toolbar)
            {
                return FollowToolbar(toolbar.Rect, toolbar.Viewport, PanelWidth,
                    SimCoachChip.Instance.StackHeight());
            }
            float? toolbarBottom = SimToolbar.Instance.BottomY();
            return new Vector2(PanelLeft, toolbarBottom.HasValue
                ? toolbarBottom.Value + AdviceColumn.Gap
                : ChipFallbackTop + ChipToPanelGap);
        }
        catch (Exception)
        {
            return new Vector2(PanelLeft, ChipFallbackTop + ChipToPanelGap);
        }
    }

    /// <summary>
    /// Docked toolbar: keep the advice panel (and the chip under it) clear of
    /// the game's own top bar (measured every tick -- see
    /// SimTopBarLayout.SafeTopY for the why). Dragged toolbar: the panel and
    /// the chips travel with it, as one block (AdviceColumn).
    /// </summary>
    private void RepositionPanels()
    {
        try
        {
            if (_advicePanel == null || !GodotObject.IsInstanceValid(_advicePanel))
            {
                return;
            }

            Vector2 want;
            if (SimToolbar.Instance.UndockedRect() is { } toolbar)
            {
                float height = AdviceColumn.BlockHeight(_advicePanel.Size.Y,
                    _advicePanel.IsVisibleInTree(), SimCoachChip.Instance.StackHeight());
                want = FollowToolbar(toolbar.Rect, toolbar.Viewport,
                    Math.Max(PanelWidth, _advicePanel.Size.X), height);
            }
            else
            {
                float top = SimTopBarLayout.SafeTopY(_advicePanel);
                // Below the toolbar strip when there is one; otherwise leave
                // the old chip-sized gap so the panel is where it always was.
                float? toolbarBottom = SimToolbar.Instance.BottomY();
                // Co-op: beside the teammates' list, never on it.
                float left = SimTopBarLayout.CoopRosterRight() is float roster
                    ? Math.Max(PanelLeft, roster + AdviceColumn.Gap)
                    : PanelLeft;
                want = new Vector2(left, toolbarBottom.HasValue
                    ? Math.Max(top, toolbarBottom.Value + AdviceColumn.Gap)
                    : top + ChipToPanelGap);
            }
            if (_advicePanel.Position.DistanceTo(want) > 0.5f)
            {
                _advicePanel.Position = want;
            }
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] RepositionPanels failed: {ex.Message}");
        }
    }

    private static Vector2 FollowToolbar(Rect2 toolbar, Vector2 viewport, float width, float height)
    {
        (float x, float y) = AdviceColumn.BlockTopLeft(toolbar.Position.X, toolbar.Position.Y,
            toolbar.Size.Y, width, height, viewport.X, viewport.Y);
        return new Vector2(x, y);
    }

    private void EnsureCombatEndedSubscription()
    {
        if (_combatEndedSubscribed)
        {
            return;
        }
        try
        {
            CombatManager.Instance.CombatEnded += _ =>
            {
                ClearAdvice();
                RunOnMainThread(SimMoveFeedback.Instance.OnCombatEnded);
            };
            _combatEndedSubscribed = true;
        }
        catch (Exception ex)
        {
            Logger.Log($"[SimHud] Could not subscribe to CombatEnded: {ex.Message}");
        }
    }

    private static StyleBoxFlat PanelStyle() => SimUiKit.PanelStyle();

    private static Label SmallLabel(string text, Color color)
    {
        var label = new Label
        {
            Text = text,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", Colors.Black);
        label.AddThemeConstantOverride("outline_size", 3);
        label.AddThemeFontSizeOverride("font_size", 14);
        return label;
    }

    /// <summary>Red (0) -> green (1) for legacy scores.</summary>
    private static Color ScoreColor(double score)
    {
        Color low = new(0.9f, 0.2f, 0.2f);
        Color high = new(0.2f, 0.9f, 0.2f);
        return low.Lerp(high, (float)score);
    }

    /// <summary>Q in [-1,1]: red -> amber -> green (Reasoning.tsx idiom).</summary>
    private static Color QColor(double q)
    {
        q = Math.Clamp(q, -1.0, 1.0);
        Color red = new(0.9f, 0.25f, 0.2f);
        Color amber = new(0.95f, 0.75f, 0.3f);
        Color green = new(0.25f, 0.85f, 0.35f);
        return q < 0 ? red.Lerp(amber, (float)(q + 1)) : amber.Lerp(green, (float)q);
    }

    /// <summary>
    /// Colour for "how much worse than the best move". Scaled to the range
    /// that matters for a Q delta (a hundredth is already a real mistake),
    /// not to the [-1,1] of an absolute eval, or every row would read grey.
    /// </summary>
    private static Color DeltaColor(double delta)
    {
        double loss = Math.Clamp(-delta / 0.05, 0.0, 1.0);
        Color even = new(0.7f, 0.72f, 0.78f);
        Color bad = new(0.9f, 0.35f, 0.3f);
        return even.Lerp(bad, (float)loss);
    }
}
