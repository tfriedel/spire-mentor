using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sts2Headless;

/// <summary>
/// Referee mode: what a client is allowed to see.
///
/// The host's decision payloads were built for parity work and carry hidden
/// state (the draw pile in draw order, RNG seed and counters, enemy AI
/// internals). This projects every reply onto an allow-list, so a field is
/// sent only because someone decided a player can see it:
///
/// * a field on <see cref="Allow"/> is kept;
/// * a field on <see cref="Deny"/> is dropped (known hidden state);
/// * anything else is dropped too and recorded in <see cref="Unclassified"/>,
///   which the tests require to stay empty. A new host field therefore fails
///   a test instead of leaking.
///
/// Paths use <c>.</c> for object members, <c>[]</c> for list elements and a
/// trailing <c>.*</c> for a map with open keys and scalar values. A leading
/// <c>combat.</c> (the combat state embedded in a mid-combat card selection)
/// follows the same rules as the top level.
/// </summary>
internal static class RefereeView
{
    private static readonly HashSet<string> Allow = BuildAllow();
    private static readonly HashSet<string> Prefixes = BuildPrefixes(Allow);

    private static readonly HashSet<string> Deny = new()
    {
        // RNG seed and per-stream counters.
        "rng",
        // Enemy AI internals: move log, forced next move, and the id of the
        // current move. The player sees intents, not move ids.
        "enemies[].ai", "enemies[].move_state",
        "dead_enemies[].ai", "dead_enemies[].move_state",
        // Position in the draw pile. The pile itself is sent sorted.
        "draw_pile[].index",
        // Duplicates of top-level fields inside the attached map.
        "map.type", "map.context",
    };

    private static readonly object Gate = new();
    private static readonly SortedSet<string> Unclassified = new();

    /// <summary>Fields the host emitted that are on neither list.</summary>
    public static string[] UnclassifiedFields()
    {
        lock (Gate) return Unclassified.ToArray();
    }

    /// <summary>Serialises a reply for a referee client.</summary>
    public static string Render(Dictionary<string, object?> data, JsonSerializerOptions options)
    {
        var node = JsonSerializer.SerializeToNode(data, options);
        return node is JsonObject reply ? Render(reply, options) : "{}";
    }

    /// <summary>The projection itself, on an already parsed reply.</summary>
    /// <param name="record">
    /// Whether this observation's legal actions become the list the next
    /// action is checked against. True for the host's own replies only.
    /// </param>
    public static string Render(JsonObject reply, JsonSerializerOptions options, bool record = true)
    {
        string? type = reply["type"]?.GetValue<string>();
        if (type == "decision")
        {
            var combatLegal = reply[CombatLegalKey] as JsonObject;
            reply.Remove(CombatLegalKey);
            var belt = reply[PotionBeltKey] as JsonObject;
            reply.Remove(PotionBeltKey);
            var projected = (JsonObject?)Project(reply, "") ?? new JsonObject();
            SortDrawPile(projected);
            if (projected["combat"] is JsonObject combat) SortDrawPile(combat);
            HideSecondBoss(projected);
            var legal = LegalActions(projected, combatLegal);
            AddPotionBelt(legal, belt);
            if (record) lock (Gate) _lastLegal = (JsonArray)legal.DeepClone();
            projected["legal_actions"] = legal;
            return projected.ToJsonString(options);
        }
        if (type == "error")
        {
            // A fixed code only. The host's free-text message can quote game
            // internals; it stays on the operator's side (stderr).
            string code = reply["code"]?.GetValue<string>() ?? "rejected";
            Console.Error.WriteLine($"[REFEREE] error {code}: {reply["message"]}");
            return new JsonObject { ["type"] = "error", ["code"] = code }.ToJsonString(options);
        }
        // ready, quit_result, referee_* diagnostics: no game state.
        return reply.ToJsonString(options);
    }

    /// <summary>
    /// Key under which Program hands over the host's combat legality (card
    /// playability and potion usability are game rules, not view fields).
    /// </summary>
    public const string CombatLegalKey = "__combat_legal";

    /// <summary>
    /// Key under which Program hands over the potion belt outside combat
    /// (RunSimulator.PotionBelt: which held potions the popup lets the player
    /// drink or discard).
    /// </summary>
    public const string PotionBeltKey = "__potion_belt";

    /// <summary>
    /// GH #1282: the out-of-combat screens on which the top bar's potion belt
    /// is clickable and the host can apply a drink or discard. Not the card
    /// selection overlays (card_select, card_reward, bundle_select), whose
    /// game task waits on the pick; potion_swap has its own discard.
    /// </summary>
    public static readonly HashSet<string> PotionScreens = new()
    {
        "map_select", "event_choice", "rest_site", "shop", "rewards",
    };

    // Out-of-combat use_potion / discard_potion, appended after the screen's
    // own actions. The host decides usability; the view only lists it.
    private static void AddPotionBelt(JsonArray legal, JsonObject? belt)
    {
        if (belt?["slots"] is not JsonArray slots) return;
        foreach (var flag in new[] { ("usable", "use_potion"), ("discardable", "discard_potion") })
        {
            foreach (var slot in slots.OfType<JsonObject>())
            {
                if (slot[flag.Item1]?.GetValue<bool>() != true) continue;
                legal.Add(new JsonObject
                {
                    ["action"] = flag.Item2,
                    ["args"] = new JsonObject { ["potion_index"] = slot["potion_index"]!.GetValue<int>() },
                });
            }
        }
    }

    private static JsonArray? _lastLegal;

    /// <summary>
    /// Whether an action is one of the legal actions of the last observation.
    /// The host is never asked to apply anything else.
    /// </summary>
    public static bool IsLegal(string action, JsonObject? args)
    {
        JsonArray? legal;
        lock (Gate) legal = _lastLegal;
        if (legal == null) return false;
        var given = Canonical(args);
        foreach (var entry in legal)
        {
            if (entry is not JsonObject candidate) continue;
            if (candidate["action"]?.GetValue<string>() != action) continue;
            if (candidate["choose"] is JsonObject choose)
            {
                if (MatchesChoice(given, choose)) return true;
                continue;
            }
            var wanted = Canonical(candidate["args"] as JsonObject);
            if (wanted.Count == given.Count && wanted.All(kv => given.TryGetValue(kv.Key, out var v) && v == kv.Value))
                return true;
        }
        return false;
    }

    private static Dictionary<string, string> Canonical(JsonObject? args)
    {
        var canonical = new Dictionary<string, string>();
        if (args == null) return canonical;
        foreach (var (name, value) in args)
            if (value != null) canonical[name] = value.ToJsonString();
        return canonical;
    }

    // select_cards: `indices` is a comma-separated list of distinct offered
    // indices, between min and max of them.
    private static bool MatchesChoice(Dictionary<string, string> given, JsonObject choose)
    {
        if (given.Count != 1 || !given.TryGetValue("indices", out var raw)) return false;
        string text = raw.Trim('"');
        var offered = (choose["from"] as JsonArray ?? new JsonArray()).Select(n => n!.GetValue<int>()).ToHashSet();
        var picked = new HashSet<int>();
        foreach (var part in text.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!int.TryParse(part.Trim(), out int index) || !offered.Contains(index) || !picked.Add(index))
                return false;
        }
        int min = choose["min"]!.GetValue<int>(), max = choose["max"]!.GetValue<int>();
        return picked.Count >= min && picked.Count <= max;
    }

    /// <summary>
    /// The second boss is drawn for the player only once it is the next node.
    /// Until then neither its identity nor its node is sent.
    /// </summary>
    private static void HideSecondBoss(JsonObject view)
    {
        if (view["map"] is not JsonObject map || map["second_boss"] is not JsonObject second) return;
        int? col = second["col"]?.GetValue<int>(), row = second["row"]?.GetValue<int>();
        bool reachable = (view["choices"] as JsonArray ?? new JsonArray()).Any(c =>
            c?["col"]?.GetValue<int>() == col && c?["row"]?.GetValue<int>() == row);
        bool there = map["current_coord"] is JsonObject at
            && at["col"]?.GetValue<int>() == col && at["row"]?.GetValue<int>() == row;
        if (reachable || there) return;
        map.Remove("second_boss");
        if (map["rows"] is not JsonArray rows) return;
        foreach (var line in rows.OfType<JsonArray>())
        {
            for (int i = line.Count - 1; i >= 0; i--)
            {
                var node = line[i];
                if (node?["col"]?.GetValue<int>() == col && node?["row"]?.GetValue<int>() == row)
                    line.RemoveAt(i);
            }
        }
        for (int i = rows.Count - 1; i >= 0; i--)
            if (rows[i] is JsonArray { Count: 0 }) rows.RemoveAt(i);
    }

    /// <summary>
    /// Every action the client may send for this observation, in the exact
    /// shape it sends it. Built from the view (plus the host's combat
    /// legality), so it cannot say more than the view does.
    /// </summary>
    private static JsonArray LegalActions(JsonObject view, JsonObject? combatLegal)
    {
        var legal = new JsonArray();
        void Add(string action, params (string Name, JsonNode? Value)[] args)
        {
            var entry = new JsonObject { ["action"] = action };
            if (args.Length > 0)
            {
                var bag = new JsonObject();
                foreach (var (name, value) in args) bag[name] = value;
                entry["args"] = bag;
            }
            legal.Add(entry);
        }
        IEnumerable<JsonObject> Items(string name) =>
            (view[name] as JsonArray ?? new JsonArray()).OfType<JsonObject>();
        static int Int(JsonNode? node, int fallback = 0) =>
            node is JsonValue v && v.TryGetValue<int>(out int i) ? i : fallback;
        static bool Flag(JsonNode? node, bool fallback) =>
            node is JsonValue v && v.TryGetValue<bool>(out bool b) ? b : fallback;
        int gold = Int(view["player"]?["gold"]);

        switch (view["decision"]?.GetValue<string>())
        {
            case "map_select":
                foreach (var choice in Items("choices"))
                    Add("select_map_node", ("col", Int(choice["col"])), ("row", Int(choice["row"])));
                break;

            case "combat_play":
            {
                int alive = Int(combatLegal?["alive_enemy_count"]);
                var potionTargets = ((view["player"]?["potions"]) as JsonArray ?? new JsonArray())
                    .OfType<JsonObject>()
                    .ToDictionary(p => Int(p["index"]), p => p["target_type"]?.GetValue<string>());
                foreach (var entry in (combatLegal?["actions"] as JsonArray ?? new JsonArray()).OfType<JsonObject>())
                {
                    switch (entry["action"]?.GetValue<string>())
                    {
                        case "play_card" when entry["target_index"] is null:
                            Add("play_card", ("card_index", Int(entry["card_index"])));
                            break;
                        case "play_card":
                            Add("play_card", ("card_index", Int(entry["card_index"])),
                                ("target_index", Int(entry["target_index"])));
                            break;
                        case "end_turn":
                            Add("end_turn");
                            break;
                        case "use_potion":
                        {
                            int slot = Int(entry["potion_index"]);
                            if (potionTargets.GetValueOrDefault(slot) == "AnyEnemy")
                                for (int t = 0; t < alive; t++)
                                    Add("use_potion", ("potion_index", slot), ("target_index", t));
                            else
                                Add("use_potion", ("potion_index", slot));
                            break;
                        }
                    }
                }
                foreach (int slot in potionTargets.Keys.OrderBy(k => k))
                    Add("discard_potion", ("potion_index", slot));
                break;
            }

            case "event_choice":
                foreach (var option in Items("options").Where(o => !Flag(o["is_locked"], false)))
                    Add("choose_option", ("option_index", Int(option["index"])));
                if (legal.Count == 0) Add("leave_room");
                break;

            case "rest_site":
                foreach (var option in Items("options").Where(o => Flag(o["is_enabled"], true)))
                    Add("choose_option", ("option_index", Int(option["index"])));
                if (Flag(view["can_leave"], false) || legal.Count == 0) Add("leave_room");
                break;

            case "card_reward":
                foreach (var card in Items("cards"))
                    Add("select_card_reward", ("card_index", Int(card["index"])));
                foreach (var alternative in Items("alternatives"))
                    Add("card_reward_alternative", ("alternative_index", Int(alternative["index"])));
                if (Flag(view["can_skip"], false) || legal.Count == 0) Add("skip_card_reward");
                break;

            case "bundle_select":
                foreach (var bundle in Items("bundles"))
                    Add("select_bundle", ("bundle_index", Int(bundle["index"])));
                break;

            case "card_select":
            {
                var offered = new JsonArray();
                foreach (var card in Items("cards")) offered.Add(Int(card["index"]));
                int min = Int(view["min_select"]), max = Math.Min(Int(view["max_select"]), offered.Count);
                if (offered.Count > 0 && max >= 1)
                {
                    legal.Add(new JsonObject
                    {
                        ["action"] = "select_cards",
                        ["choose"] = new JsonObject { ["from"] = offered, ["min"] = Math.Max(1, Math.Min(min, max)), ["max"] = max },
                    });
                }
                if (Flag(view["cancelable"], false) || min == 0 || legal.Count == 0) Add("skip_select");
                break;
            }

            case "shop":
                foreach (var card in Items("cards").Where(c => Flag(c["is_stocked"], false) && Int(c["cost"], int.MaxValue) <= gold))
                    Add("buy_card", ("card_index", Int(card["index"])));
                foreach (var relic in Items("relics").Where(r => Flag(r["is_stocked"], false) && Int(r["cost"], int.MaxValue) <= gold))
                    Add("buy_relic", ("relic_index", Int(relic["index"])));
                foreach (var potion in Items("potions").Where(p => Flag(p["is_stocked"], false) && Int(p["cost"], int.MaxValue) <= gold))
                    Add("buy_potion", ("potion_index", Int(potion["index"])));
                if (view["card_removal_cost"] is JsonValue && Int(view["card_removal_cost"], int.MaxValue) <= gold)
                    Add("remove_card");
                Add("leave_room");
                break;

            case "rewards":
                // GH #1282: each reward is taken or skipped on its own, in any
                // order; proceed leaves whatever is still on the screen.
                foreach (var reward in Items("rewards"))
                    Add("take_reward", ("reward_index", Int(reward["index"])));
                foreach (var reward in Items("rewards"))
                    Add("skip_reward", ("reward_index", Int(reward["index"])));
                Add("proceed");
                break;

            case "potion_swap":
                Add("potion_swap");
                foreach (var option in Items("options").Where(o => o["discard_slot"] is JsonValue))
                    Add("potion_swap", ("discard_slot", Int(option["discard_slot"])));
                break;

            case "crystal_sphere":
                // Only fogged cells take a click; either tool costs one divination.
                foreach (var cell in Items("cells").Where(c => Flag(c["hidden"], false)))
                {
                    foreach (var tool in new[] { "big", "small" })
                        Add("crystal_sphere_click", ("x", Int(cell["x"])), ("y", Int(cell["y"])), ("tool", tool));
                }
                break;

            case "game_over":
                break;

            default:
                Add("proceed");
                break;
        }
        return legal;
    }

    private static string Normalise(string path) =>
        path.StartsWith("combat.", StringComparison.Ordinal) ? Normalise(path["combat.".Length..]) : path;

    private static JsonNode? Project(JsonNode? node, string path)
    {
        string key = Normalise(path);
        if (Deny.Contains(key)) return null;

        switch (node)
        {
            case JsonObject obj:
            {
                bool open = Allow.Contains(key + ".*");
                if (path.Length > 0 && !open && !Prefixes.Contains(key))
                    return Unknown(path);
                var kept = new JsonObject();
                foreach (var (name, value) in obj)
                {
                    string child = path.Length == 0 ? name : path + "." + name;
                    if (open)
                    {
                        if (value is null or JsonValue) kept[name] = value?.DeepClone();
                        else Unknown(child);
                        continue;
                    }
                    if (value is null)
                    {
                        string childKey = Normalise(child);
                        if (Deny.Contains(childKey)) continue;
                        if (!Allow.Contains(childKey) && !Prefixes.Contains(childKey)) Unknown(child);
                        continue;
                    }
                    var projected = Project(value, child);
                    if (projected != null) kept[name] = projected;
                }
                return kept;
            }
            case JsonArray array:
            {
                string element = path + "[]";
                string elementKey = Normalise(element);
                if (!Allow.Contains(elementKey) && !Prefixes.Contains(elementKey))
                    return Unknown(path);
                var kept = new JsonArray();
                foreach (var item in array)
                {
                    var projected = item is null ? null : Project(item, element);
                    if (projected != null) kept.Add(projected);
                }
                return kept;
            }
            case JsonValue value:
                return Allow.Contains(key) ? value.DeepClone() : Unknown(path);
            default:
                return null;
        }
    }

    private static JsonNode? Unknown(string path)
    {
        lock (Gate) Unclassified.Add(path);
        return null;
    }

    /// <summary>
    /// The draw pile is public as a set and secret as a sequence: send it in
    /// an order that depends only on its contents.
    /// </summary>
    private static void SortDrawPile(JsonObject state)
    {
        if (state["draw_pile"] is not JsonArray pile) return;
        var sorted = pile.Select(card => card?.ToJsonString() ?? "").OrderBy(s => s, StringComparer.Ordinal).ToList();
        var replacement = new JsonArray();
        foreach (var card in sorted) replacement.Add(JsonNode.Parse(card));
        state["draw_pile"] = replacement;
    }

    private static HashSet<string> BuildAllow()
    {
        var allow = new HashSet<string>();
        void Add(string prefix, string fields)
        {
            foreach (var field in fields.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                allow.Add(prefix.Length == 0 ? field : prefix + "." + field);
        }

        // Everything printed on a card, in any pile or offer.
        const string card =
            "id name type rarity cost base_cost card_cost star_cost description keywords keywords[] " +
            "upgraded index can_play target_type affliction affliction_amount will_retain " +
            "enchantment enchantment_amount " +
            "persistent_cost persistent_star_cost played_cost stable_cost sly_this_turn " +
            "tinker_time_rider is_stocked on_sale stats stats.* " +
            "after_upgrade.cost after_upgrade.description after_upgrade.stats after_upgrade.stats.* " +
            "after_upgrade.added_keywords after_upgrade.added_keywords[] " +
            "after_upgrade.removed_keywords after_upgrade.removed_keywords[]";
        foreach (var pile in new[]
                 {
                     "hand[]", "draw_pile[]", "discard_pile[]", "exhaust_pile[]", "play_pile[]",
                     "cards[]", "player.deck[]", "bundles[].cards[]",
                 })
            Add(pile, card);

        const string power = "id name amount display_amount description vars vars.*";
        Add("player_powers[]", power);
        Add("enemies[].powers[]", power);
        Add("dead_enemies[].powers[]", power);

        Add("", "type decision act act_name ascension floor victory abandoned round energy max_energy stars " +
                "orb_slots draw_pile_count discard_pile_count exhaust_pile_count " +
                "can_leave can_skip cancelable from_event gold_earned card_removal_cost " +
                "min_select max_select prompt_key source source_model description event_name potion");
        Add("context", "act act_name floor total_floor room_type boss.id boss.name");
        Add("player", "name hp max_hp block gold deck_size potion_slots cards_played_this_turn[]");
        Add("player.potions[]", "entry name description index target_type vars vars.*");
        Add("player.relics[]", "id name description counter is_wax is_melted vars vars.*");
        Add("enemies[]", "name index hp max_hp block intends_attack");
        Add("enemies[].intents[]", "type damage hits");
        Add("dead_enemies[]", "name hp max_hp block is_dead");
        Add("dead_enemies[].intents[]", "type damage hits");
        Add("orbs[]", "name type index passive evoke");
        Add("osty", "name alive hp max_hp block");
        Add("choices[]", "col row type");
        Add("options[]", "index title name option_id text_key description is_locked is_enabled " +
                         "will_kill_player discard_slot label vars vars.*");
        Add("bundles[]", "index");
        Add("relics[]", "id name description index cost is_stocked vars vars.*");
        Add("potions[]", "id name description index cost is_stocked vars vars.*");
        // The reward screen (GH #1282): what each reward button shows.
        Add("rewards[]", "index type amount id name description");
        // A card reward's extra buttons (GH #1296, #1315): Reroll, Sacrifice.
        Add("alternatives[]", "index id name");

        // The act map as the map screen shows it (attached to map decisions).
        Add("map.rows[][]", "col row type visited current");
        Add("map.rows[][].children[]", "col row");
        Add("map.boss", "col row type id name");
        // Present only once the second boss is the next node (HideSecondBoss).
        Add("map.second_boss", "col row type id name");
        Add("map.current_coord", "col row");

        // The Crystal Sphere grid. The host fills `item` only for cleared cells.
        Add("", "divinations_left");
        Add("grid", "width height");
        Add("cells[]", "x y hidden");
        Add("cells[].item", "kind part_x part_y width height");

        // A mid-combat card selection embeds the combat state.
        Add("combat", "type decision");
        return allow;
    }

    private static HashSet<string> BuildPrefixes(HashSet<string> allow)
    {
        var prefixes = new HashSet<string>();
        foreach (var pattern in allow)
        {
            for (int i = 0; i < pattern.Length; i++)
            {
                if (pattern[i] == '.') prefixes.Add(pattern[..i]);
                else if (pattern[i] == '[' ) { prefixes.Add(pattern[..i]); prefixes.Add(pattern[..(i + 2)]); }
            }
        }
        return prefixes;
    }
}
