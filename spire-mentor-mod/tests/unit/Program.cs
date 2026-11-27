// Mod unit tests (no framework): each Check prints and counts; exit 1 on any failure.

using System.Globalization;
using SpireMentor;

if (args.Length is 2 or 3 && args[0] == "--advisor-child")
{
    return WindowsChildProcessTests.RunChild(args[1], args.Length == 3 ? int.Parse(args[2]) : null);
}

int failures = 0;
int checks = 0;

void Check(string name, string? actual, string? expected)
{
    checks++;
    if (actual == expected)
    {
        return;
    }
    failures++;
    Console.WriteLine($"FAIL {name}: got {actual ?? "<null>"}, want {expected ?? "<null>"}");
}

// GameVersionFormat.Normalize: release_info.json "version" -> wire game_version.
// The advisor binary per platform: the names stage_advisor.py ships (GH #1424/#1425).
Check("advisor: windows", SpireMentor.AdvisorChild.BinaryName(true, false, System.Runtime.InteropServices.Architecture.X64), "SimAdvisor.exe");
Check("advisor: linux x64", SpireMentor.AdvisorChild.BinaryName(false, false, System.Runtime.InteropServices.Architecture.X64), "SimAdvisor-linux-x64");
Check("advisor: linux arm64", SpireMentor.AdvisorChild.BinaryName(false, false, System.Runtime.InteropServices.Architecture.Arm64), "SimAdvisor-linux-arm64");
Check("advisor: macos arm64", SpireMentor.AdvisorChild.BinaryName(false, true, System.Runtime.InteropServices.Architecture.Arm64), "SimAdvisor-macos-arm64");

// The coach switches itself on at launch only where an advisor can start.
Check("coach at launch: windows", SpireMentor.AdvisorChild.CanStartAtLaunch(true, false, false).ToString(), "True");
Check("coach at launch: linux with the packaged binary", SpireMentor.AdvisorChild.CanStartAtLaunch(false, false, true).ToString(), "True");
Check("coach at launch: a configured command", SpireMentor.AdvisorChild.CanStartAtLaunch(false, true, false).ToString(), "True");
Check("coach at launch: no advisor for this platform", SpireMentor.AdvisorChild.CanStartAtLaunch(false, false, false).ToString(), "False");

if (!OperatingSystem.IsWindows())
{
    // The Unix launch keeps psi.ArgumentList (the /bin/sh -c path) and reads UTF-8.
    var psi = new System.Diagnostics.ProcessStartInfo { FileName = "/bin/sh" };
    psi.ArgumentList.Add("-c");
    psi.ArgumentList.Add("printf '%s\\n' \"a \u2192 b\"; echo err >&2");
    using var child = SpireMentor.UnixChildProcess.Start(psi);
    string outLine = child.StandardOutput.ReadToEnd().Trim();
    string errLine = child.StandardError.ReadToEnd().Trim();
    child.Process.WaitForExit();
    Check("unix child: stdout", outLine, "a \u2192 b");
    Check("unix child: stderr", errLine, "err");
}

Check("canonical 0.111", GameVersionFormat.Normalize("v0.111.0"), "v0.111.0");
Check("canonical 0.107", GameVersionFormat.Normalize("v0.107.1"), "v0.107.1");
Check("no v prefix", GameVersionFormat.Normalize("0.111.0"), "v0.111.0");
Check("upper V + padding", GameVersionFormat.Normalize("  V0.107.1\n"), "v0.107.1");
Check("leading zeros dropped", GameVersionFormat.Normalize("v00.0111.00"), "v0.111.0");
Check("null", GameVersionFormat.Normalize(null), null);
Check("blank", GameVersionFormat.Normalize("   "), null);
// Unknown shapes are passed through (trimmed, v-prefixed), never guessed into a
// supported build: the advisor names exactly what it refuses.
Check("suffix passes through", GameVersionFormat.Normalize("v0.112.0-beta"), "v0.112.0-beta");
Check("four parts pass through", GameVersionFormat.Normalize("0.111.0.1"), "v0.111.0.1");
Check("bare v", GameVersionFormat.Normalize("v"), "v");
Check("non-ascii digits pass through", GameVersionFormat.Normalize("v0.१११.0"), "v0.१११.0");
Check("default is stable", GameVersionFormat.Default, "v0.107.1");

// GameVersionFormat.EngineBuild: a hotfix (same major.minor, higher patch)
// plays on its line's newest listed build; a new minor plays on nothing.
string[] supportedBuilds = { "v0.107.1", "v0.111.0", "v0.113.0", "v0.114.0" };
Check("engine: exact build", GameVersionFormat.EngineBuild("v0.113.0", supportedBuilds), "v0.113.0");
Check("engine: hotfix", GameVersionFormat.EngineBuild("v0.113.1", supportedBuilds), "v0.113.0");
Check("engine: raw hotfix spelling", GameVersionFormat.EngineBuild(" 0.113.12 ", supportedBuilds), "v0.113.0");
Check("engine: hotfix of an older line", GameVersionFormat.EngineBuild("v0.107.3", supportedBuilds), "v0.107.1");
Check("engine: v0.111.1 stable", GameVersionFormat.EngineBuild("v0.111.1", supportedBuilds), "v0.111.0");
Check("engine: newest base of the line", GameVersionFormat.EngineBuild("v0.113.5", new[] { "v0.113.0", "v0.113.4", "v0.113.6" }), "v0.113.4");
Check("engine: new minor", GameVersionFormat.EngineBuild("v0.115.0", supportedBuilds), null);
Check("engine: v0.114 is its own build", GameVersionFormat.EngineBuild("v0.114.0", supportedBuilds), "v0.114.0");
Check("engine: v0.114 hotfix", GameVersionFormat.EngineBuild("0.114.2", supportedBuilds), "v0.114.0");
Check("engine: older than the line", GameVersionFormat.EngineBuild("v0.107.0", supportedBuilds), null);
Check("engine: suffix", GameVersionFormat.EngineBuild("v0.113.1-beta", supportedBuilds), null);
Check("engine: null", GameVersionFormat.EngineBuild(null, supportedBuilds), null);

// ChoiceInference: which option the human picked, from a before/after diff.
// null = skip, ChoiceInference.Unknown (-1) = can't tell (send nothing).
string P(int? v) => v == null ? "skip" : v.Value.ToString();
var mapOpts = new List<(int col, int row)> { (1, 2), (3, 2), (5, 2) };
Check("map pick found", P(ChoiceInference.MapPick(mapOpts, (3, 2))), "1");
Check("map pick not offered", P(ChoiceInference.MapPick(mapOpts, (4, 2))), "-1");
Check("map pick nothing visited", P(ChoiceInference.MapPick(mapOpts, null)), "-1");
Check("newly visited: one", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0), (1, 1) }, new List<(int, int)> { (0, 0), (1, 1), (3, 2) })?.ToString(), "(3, 2)");
Check("newly visited: none", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0) }, new List<(int, int)> { (0, 0) })?.ToString(), null);
Check("newly visited: two is ambiguous", ChoiceInference.NewlyVisited(
    new List<(int, int)> { (0, 0) }, new List<(int, int)> { (0, 0), (1, 1), (2, 2) })?.ToString(), null);

var reward = new List<string> { "Pommel Strike", "Inflame", "Shrug It Off" };
var deck = new List<string> { "Strike", "Strike", "Defend", "Bash" };
Check("card pick", P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Inflame").ToList())), "1");
Check("card skip", P(ChoiceInference.AddedCardPick(reward, deck, deck)), "skip");
var deckWithInflame = deck.Append("Inflame").ToList();
Check("duplicate already in deck (count diff)",
    P(ChoiceInference.AddedCardPick(reward, deckWithInflame, deckWithInflame.Append("Inflame").ToList())), "1");
Check("added card not offered -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Clumsy").ToList())), "-1");
Check("offered card + unrelated add -> the offered one",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Clumsy").Append("Inflame").ToList())), "1");
Check("two offered cards added -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Append("Inflame").Append("Shrug It Off").ToList())), "-1");
Check("card removed only -> unknown",
    P(ChoiceInference.AddedCardPick(reward, deck, deck.Skip(1).ToList())), "-1");

var relics = new List<string> { "BURNING_BLOOD" };
var bossOffer = new List<string> { "SOZU", "ECTOPLASM", "PHILOSOPHERS_STONE" };
Check("relic pick", P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics.Append("ECTOPLASM").ToList())), "1");
Check("relic skip", P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics)), "skip");
Check("relic not offered -> unknown",
    P(ChoiceInference.AddedRelicPick(bossOffer, relics, relics.Append("ANCHOR").ToList())), "-1");

// Same rule as SimAdvisor.ReconcileIndex: trimmed, case-insensitive, first match.
var labels = new List<string> { "Strike", " inflame ", "Inflame" };
Check("label exact", ChoiceInference.IndexByLabel(labels, "Strike")?.ToString(), "0");
Check("label case/trim, first match wins", ChoiceInference.IndexByLabel(labels, "INFLAME")?.ToString(), "1");
Check("label missing", ChoiceInference.IndexByLabel(labels, "Bash")?.ToString(), null);
Check("label empty list", ChoiceInference.IndexByLabel(new List<string>(), "Bash")?.ToString(), null);
Check("same label twice in an offer -> first",
    P(ChoiceInference.AddedCardPick(new List<string> { "Inflame", "Inflame" }, deck, deck.Append("Inflame").ToList())), "0");

var bundles = new List<IReadOnlyList<string>>
{
    new List<string> { "A", "B" },
    new List<string> { "C", "D" },
};
Check("bundle pick", P(ChoiceInference.BundlePick(bundles, deck, deck.Append("D").Append("C").ToList())), "1");
Check("bundle partial -> unknown", P(ChoiceInference.BundlePick(bundles, deck, deck.Append("C").ToList())), "-1");
Check("bundle nothing -> unknown (no skip)", P(ChoiceInference.BundlePick(bundles, deck, deck)), "-1");

var shop = new List<(string action, string id)>
{
    ("leave_shop", "leave_shop"), ("buy_card", "Inflame"), ("buy_relic", "ANCHOR"),
    ("buy_potion", "FIRE_POTION"), ("remove_card", "remove_card"),
};
var noPotions = new List<string>();
Check("shop buys card", P(ChoiceInference.ShopPick(shop, deck, deck.Append("Inflame").ToList(),
    relics, relics, noPotions, noPotions)), "1");
Check("shop buys relic", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics.Append("ANCHOR").ToList(), noPotions, noPotions)), "2");
Check("shop buys potion", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics, noPotions, new List<string> { "FIRE_POTION" })), "3");
Check("shop removes a card", P(ChoiceInference.ShopPick(shop, deck, deck.Skip(1).ToList(),
    relics, relics, noPotions, noPotions)), "4");
Check("shop nothing -> unknown", P(ChoiceInference.ShopPick(shop, deck, deck,
    relics, relics, noPotions, noPotions)), "-1");
Check("shop relic + card -> unknown", P(ChoiceInference.ShopPick(shop, deck, deck.Append("Inflame").ToList(),
    relics, relics.Append("ANCHOR").ToList(), noPotions, noPotions)), "-1");

Check("single changed", P(ChoiceInference.SingleChanged(new List<bool> { false, true, false })), "1");
Check("none changed", P(ChoiceInference.SingleChanged(new List<bool> { false, false })), "-1");
Check("two changed", P(ChoiceInference.SingleChanged(new List<bool> { true, true })), "-1");

// CoachStyle: F8 toggles the AI on/off and keeps the style; the style button
// flips Hints <-> Coach and never turns the AI off.
string T(SimModeLite m, AdviceStyle s) => CoachStyle.ToggleOnOff(m, s).ToString();
string W(SimModeLite m, AdviceStyle s) => CoachStyle.SwitchStyle(m, s).ToString();
Check("toggle off -> on keeps coach", T(SimModeLite.Off, AdviceStyle.Coach), "(Advisor, Coach)");
Check("toggle off -> on keeps hints", T(SimModeLite.Off, AdviceStyle.Hints), "(Advisor, Hints)");
Check("toggle on -> off", T(SimModeLite.Advisor, AdviceStyle.Coach), "(Off, Coach)");
Check("toggle autopilot -> advisor", T(SimModeLite.Autopilot, AdviceStyle.Coach), "(Advisor, Coach)");
Check("style coach -> hints stays on", W(SimModeLite.Advisor, AdviceStyle.Coach), "(Advisor, Hints)");
Check("style hints -> coach stays on", W(SimModeLite.Advisor, AdviceStyle.Hints), "(Advisor, Coach)");
Check("style while off stays off", W(SimModeLite.Off, AdviceStyle.Hints), "(Off, Coach)");
Check("power label on", CoachStyle.PowerLabel(SimModeLite.Advisor), "AI on");
Check("power label off", CoachStyle.PowerLabel(SimModeLite.Off), "AI off");
Check("power label autopilot", CoachStyle.PowerLabel(SimModeLite.Autopilot), "Take over");
Check("style label hints", CoachStyle.StyleLabel(AdviceStyle.Hints), "Hints");
Check("style label coach", CoachStyle.StyleLabel(AdviceStyle.Coach), "Coach");
Check("mode name coach", CoachStyle.ModeName(SimModeLite.Advisor, AdviceStyle.Coach), "Coach");
Check("mode name autopilot", CoachStyle.ModeName(SimModeLite.Autopilot, AdviceStyle.Coach), "Autopilot");
Check("fresh config default is Hints",
    CoachStyle.FromDefaultFlag(CoachStyle.CoachByDefault).ToString(), "Hints");
Check("default flag off -> Hints", CoachStyle.FromDefaultFlag(false).ToString(), "Hints");
Check("parse hints", CoachStyle.Parse(" HINTS ")?.ToString(), "Hints");
Check("parse coach", CoachStyle.Parse("coach")?.ToString(), "Coach");
Check("parse junk", CoachStyle.Parse("advisor")?.ToString(), null);

// MentorModes: the badge's Mode switch over SimMode x style x F7 autoplay.
string Cur(SimModeLite m, AdviceStyle s, bool armed = false, bool battle = false) =>
    MentorModes.Current(m, s, armed, battle).ToString();
Check("current off", Cur(SimModeLite.Off, AdviceStyle.Coach), "Off");
Check("current off ignores a stale armed flag", Cur(SimModeLite.Off, AdviceStyle.Hints, armed: true), "Off");
Check("current coach", Cur(SimModeLite.Advisor, AdviceStyle.Coach), "Coach");
Check("current hints", Cur(SimModeLite.Advisor, AdviceStyle.Hints), "Hints");
Check("current F7 is AI plays", Cur(SimModeLite.Advisor, AdviceStyle.Coach, armed: true), "AiPlays");
Check("current autobattle keeps its style lit", Cur(SimModeLite.Advisor, AdviceStyle.Hints, armed: true, battle: true), "Hints");
Check("current autopilot is AI plays", Cur(SimModeLite.Autopilot, AdviceStyle.Coach), "AiPlays");
string Pl(MentorMode t, SimModeLite m, AdviceStyle s, bool armed = false, bool battle = false) =>
    MentorModes.Plan(t, m, s, armed, battle)?.ToString() ?? "no-op";
Check("plan: the lit segment does nothing", Pl(MentorMode.Coach, SimModeLite.Advisor, AdviceStyle.Coach), "no-op");
Check("plan: AI plays under autopilot does nothing", Pl(MentorMode.AiPlays, SimModeLite.Autopilot, AdviceStyle.Coach), "no-op");
Check("plan: coach under autobattle does nothing", Pl(MentorMode.Coach, SimModeLite.Advisor, AdviceStyle.Coach, armed: true, battle: true), "no-op");
Check("plan: off -> coach turns the coach on",
    Pl(MentorMode.Coach, SimModeLite.Off, AdviceStyle.Hints),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("plan: coach -> hints only switches style",
    Pl(MentorMode.Hints, SimModeLite.Advisor, AdviceStyle.Coach),
    "MentorModePlan { Mode = Advisor, Style = Hints, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("plan: hints -> off keeps the style for next time",
    Pl(MentorMode.Off, SimModeLite.Advisor, AdviceStyle.Hints),
    "MentorModePlan { Mode = Off, Style = Hints, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("plan: off -> AI plays turns on in the set style and arms F7",
    Pl(MentorMode.AiPlays, SimModeLite.Off, AdviceStyle.Coach),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = True, DisarmAutoplay = False }");
Check("plan: coach -> AI plays arms F7",
    Pl(MentorMode.AiPlays, SimModeLite.Advisor, AdviceStyle.Coach),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = True, DisarmAutoplay = False }");
Check("plan: AI plays -> hints disarms F7",
    Pl(MentorMode.Hints, SimModeLite.Advisor, AdviceStyle.Coach, armed: true),
    "MentorModePlan { Mode = Advisor, Style = Hints, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = True }");
Check("plan: AI plays -> coach (same style) disarms F7",
    Pl(MentorMode.Coach, SimModeLite.Advisor, AdviceStyle.Coach, armed: true),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = True }");
Check("plan: autobattle -> hints keeps the autobattle",
    Pl(MentorMode.Hints, SimModeLite.Advisor, AdviceStyle.Coach, armed: true, battle: true),
    "MentorModePlan { Mode = Advisor, Style = Hints, StopAutopilot = False, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("plan: autobattle -> AI plays trades it for F7",
    Pl(MentorMode.AiPlays, SimModeLite.Advisor, AdviceStyle.Coach, armed: true, battle: true),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = False, StopAutobattle = True, ArmAutoplay = True, DisarmAutoplay = False }");
Check("plan: autopilot -> coach takes over, like F8",
    Pl(MentorMode.Coach, SimModeLite.Autopilot, AdviceStyle.Hints),
    "MentorModePlan { Mode = Advisor, Style = Coach, StopAutopilot = True, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("plan: autopilot -> off stops it",
    Pl(MentorMode.Off, SimModeLite.Autopilot, AdviceStyle.Coach),
    "MentorModePlan { Mode = Off, Style = Coach, StopAutopilot = True, StopAutobattle = False, ArmAutoplay = False, DisarmAutoplay = False }");
Check("segment label AI plays", MentorModes.Label(MentorMode.AiPlays, autopilot: false), "AI plays");
Check("segment label under autopilot", MentorModes.Label(MentorMode.AiPlays, autopilot: true), "Autopilot");
Check("segment label coach under autopilot", MentorModes.Label(MentorMode.Coach, autopilot: true), "Coach");
Check("segments left to right", string.Join(",", MentorModes.All), "Off,Coach,Hints,AiPlays");

// MentorBadge: the badge's words and the popover header, in player language.
BadgeState Badge(MentorMode mode, AdvisorStatusLite advisor = AdvisorStatusLite.Connected,
    bool autopilot = false, bool engaged = false, bool battle = false, bool hidden = false, string build = "0:00") =>
    new(mode, autopilot, engaged, battle, advisor, build, hidden);
Check("badge off", MentorBadge.Text(Badge(MentorMode.Off, AdvisorStatusLite.None), 0, false), "Spire Mentor · Off");
Check("badge coach ready", MentorBadge.Text(Badge(MentorMode.Coach), 0, false), "Spire Mentor · Coach");
Check("badge hints thinking", MentorBadge.Text(Badge(MentorMode.Hints, AdvisorStatusLite.Thinking), 0, false), "Spire Mentor · Hints · thinking ●○○");
Check("badge thinking dots move", MentorBadge.Text(Badge(MentorMode.Hints, AdvisorStatusLite.Thinking), 4, false), "Spire Mentor · Hints · thinking ○●○");
Check("badge thinking still with reduced effects", MentorBadge.Text(Badge(MentorMode.Hints, AdvisorStatusLite.Thinking), 4, true), "Spire Mentor · Hints · thinking …");
Check("badge building outranks the mode", MentorBadge.Text(Badge(MentorMode.Coach, AdvisorStatusLite.Building, build: "0:42"), 0, false), "Spire Mentor · Building the engine 0:42");
Check("badge hidden", MentorBadge.Text(Badge(MentorMode.Coach, hidden: true), 0, false), "Spire Mentor · hidden (click to show)");
Check("badge crashed", MentorBadge.Text(Badge(MentorMode.Coach, AdvisorStatusLite.Crashed), 0, false), "Spire Mentor · Coach · offline");
Check("badge waiting for the advisor", MentorBadge.Text(Badge(MentorMode.Coach, AdvisorStatusLite.None), 0, false), "Spire Mentor · Coach · starting");
Check("badge AI playing", MentorBadge.Text(Badge(MentorMode.AiPlays), 0, false), "Spire Mentor · AI playing");
{
    // ThinkingIndicator: a millisecond think never reaches the badge; a long
    // one does, and a short gap between thinks keeps it up.
    DateTime t0 = new(2026, 11, 9, 12, 0, 0, DateTimeKind.Utc);
    DateTime At(int ms) => t0.AddMilliseconds(ms);
    var quick = new ThinkingIndicator();
    Check("thinking: a quick think is hidden", quick.Shown(AdvisorStatusLite.Thinking, At(0)).ToString(), "Connected");
    Check("thinking: still hidden at 0.5 s", quick.Shown(AdvisorStatusLite.Thinking, At(500)).ToString(), "Connected");
    Check("thinking: done before it showed", quick.Shown(AdvisorStatusLite.Connected, At(520)).ToString(), "Connected");
    Check("thinking: the next think starts the clock over", quick.Shown(AdvisorStatusLite.Thinking, At(700)).ToString(), "Connected");
    var slow = new ThinkingIndicator();
    slow.Shown(AdvisorStatusLite.Thinking, At(0));
    Check("thinking: shows once it lasts", slow.Shown(AdvisorStatusLite.Thinking, At(600)).ToString(), "Thinking");
    Check("thinking: lingers through a short gap", slow.Shown(AdvisorStatusLite.Connected, At(700)).ToString(), "Thinking");
    Check("thinking: the next think keeps it up", slow.Shown(AdvisorStatusLite.Thinking, At(900)).ToString(), "Thinking");
    slow.Shown(AdvisorStatusLite.Connected, At(1000));
    Check("thinking: goes after a real pause", slow.Shown(AdvisorStatusLite.Connected, At(1400)).ToString(), "Connected");
    Check("thinking: other states pass through", slow.Shown(AdvisorStatusLite.Crashed, At(1500)).ToString(), "Crashed");
}
Check("badge autobattle", MentorBadge.Text(Badge(MentorMode.Hints, battle: true), 0, false), "Spire Mentor · Autobattle");
Check("badge autopilot", MentorBadge.Text(Badge(MentorMode.AiPlays, autopilot: true, engaged: true), 0, false), "Spire Mentor · Autopilot");
Check("badge autopilot idle", MentorBadge.Text(Badge(MentorMode.AiPlays, autopilot: true), 0, false), "Spire Mentor · Autopilot (waiting)");
// The sentence stays put while the advisor thinks (the popover must not jump
// under the mouse); the activity word carries the thinking.
Check("status steady while thinking", MentorBadge.Status(Badge(MentorMode.Coach, AdvisorStatusLite.Thinking)),
    MentorBadge.Status(Badge(MentorMode.Coach)));
Check("status coach", MentorBadge.Status(Badge(MentorMode.Coach)),
    "Coach: nothing shows before you act. Peek (F5) shows the next pick.");
Check("status autobattle steady while thinking", MentorBadge.Status(Badge(MentorMode.Hints, AdvisorStatusLite.Thinking, battle: true)),
    "Autobattle: the AI plays the fights, you make the run's decisions.");
Check("activity thinking", MentorBadge.Activity(Badge(MentorMode.Coach, AdvisorStatusLite.Thinking), 2, false), "thinking ○○●");
Check("activity ready", MentorBadge.Activity(Badge(MentorMode.Coach), 0, false), null);
Check("activity off says nothing", MentorBadge.Activity(Badge(MentorMode.Off, AdvisorStatusLite.Thinking), 0, false), null);
Check("activity building", MentorBadge.Activity(Badge(MentorMode.Off, AdvisorStatusLite.Building, build: "1:00"), 0, false), "building the engine 1:00");
Check("status waiting for the advisor", MentorBadge.Status(Badge(MentorMode.Hints, AdvisorStatusLite.None)), "Starting the advisor…");
Check("status crashed says how to restart", MentorBadge.Status(Badge(MentorMode.Coach, AdvisorStatusLite.Crashed)),
    "Advisor offline. Switch the AI Off and back on to restart it.");
Check("status off", MentorBadge.Status(Badge(MentorMode.Off, AdvisorStatusLite.None)), "The AI is off. Pick Coach or Hints for advice while you play.");
Check("status building", MentorBadge.Status(Badge(MentorMode.Hints, AdvisorStatusLite.Building, build: "3:05")),
    "The first start after an update builds the engine, which takes minutes. Leave the game running: quitting starts it over.");
Check("status autopilot idle", MentorBadge.Status(Badge(MentorMode.AiPlays, AdvisorStatusLite.None, autopilot: true)),
    "Autopilot is waiting for its AI client. Pick Coach or Hints to play yourself (F8).");
Check("status F7", MentorBadge.Status(Badge(MentorMode.AiPlays)), "The AI is playing until you stop it (F7).");
Check("no stop button for the coach", MentorBadge.StopLabel(Badge(MentorMode.Coach)), null);
Check("stop button for F7", MentorBadge.StopLabel(Badge(MentorMode.AiPlays)), "AI is playing — Stop (F7)");
Check("stop button for autobattle", MentorBadge.StopLabel(Badge(MentorMode.Coach, battle: true)), "AI fights — Stop (F7)");
Check("stop button for autopilot is a takeover", MentorBadge.StopLabel(Badge(MentorMode.AiPlays, autopilot: true, engaged: true)),
    "Autopilot is playing — Take over (F8)");
Check("elapsed m:ss", MentorBadge.Elapsed(TimeSpan.FromSeconds(42)), "0:42");
Check("elapsed past an hour", MentorBadge.Elapsed(TimeSpan.FromSeconds(3725)), "62:05");
Check("elapsed clock stepped back", MentorBadge.Elapsed(TimeSpan.FromSeconds(-5)), "0:00");

// CoachChipText.Format: one fixture per row of the plan's copy table.
string Chip(string json, bool showGood = false)
{
    var obj = (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!;
    ChipText? c = CoachChipText.Format(obj, showGood);
    return c == null ? "<none>" : $"[{c.Tone}] {c.Title} | {string.Join(" | ", c.Lines)}";
}
const string turnBase = "\"kind\":\"turn\",\"id\":\"turn:n4\",\"node_id\":\"n4\",\"turn\":3,\"floor\":5," +
    "\"you\":[\"Strike → Cultist\",\"Strike → Cultist\",\"end turn\"],\"ai\":[\"Bash → Cultist\",\"Defend\",\"end turn\"]," +
    "\"ungraded_actions\":0";
Check("turn blunder with HP",
    Chip("{" + turnBase + ",\"verdict\":\"blunder\",\"hp_drop\":9.4,\"win_before\":0.96,\"win_after\":0.941,\"peeked\":false}"),
    "[Blunder] ⚠ blunder | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 9 HP more lost · win 96% → 94%");
Check("turn blunder with a reason line",
    Chip("{" + turnBase + ",\"verdict\":\"blunder\",\"hp_drop\":9.4,\"win_before\":0.96,\"win_after\":0.941,\"peeked\":false," +
         "\"why\":\"Your line wakes Lagavulin Matriarch on turn 3\"}"),
    "[Blunder] ⚠ blunder | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 9 HP more lost · win 96% → 94%" +
    " | Your line wakes Lagavulin Matriarch on turn 3");
Check("turn slip without HP",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":null,\"win_before\":0.80,\"win_after\":0.75,\"peeked\":false}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | win 80% → 75%");
Check("turn slip, HP rounds below 1 -> dropped",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":0.4,\"win_before\":0.80,\"win_after\":0.75,\"peeked\":false}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | win 80% → 75%");
Check("turn peeked suffix",
    Chip("{" + turnBase + ",\"verdict\":\"slip\",\"hp_drop\":4,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":true}"),
    "[Slip] ▽ suboptimal | Turn 3 · you: Strike → Cultist → Strike → Cultist → end turn" +
    " | AI:  Bash → Cultist → Defend → end turn | ≈ 4 HP more lost · win 90% → 90% (peeked)");
Check("turn tie hidden",
    Chip("{" + turnBase + ",\"verdict\":\"tie\",\"hp_drop\":1,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}"),
    "<none>");
Check("turn tie shown",
    Chip("{" + turnBase + ",\"verdict\":\"tie\",\"hp_drop\":1,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}", showGood: true),
    "[Good] ✓ Turn 3 · toss-up | ");
Check("turn best shown",
    Chip("{" + turnBase + ",\"verdict\":\"best\",\"hp_drop\":0,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}", showGood: true),
    "[Good] ✓ Turn 3 | ");
Check("turn best hidden",
    Chip("{" + turnBase + ",\"verdict\":\"best\",\"hp_drop\":0,\"win_before\":0.9,\"win_after\":0.9,\"peeked\":false}"),
    "<none>");

string ChoiceJson(int rank, string extra = "", string verdict = "flag") =>
    "{\"kind\":\"choice\",\"id\":\"choice:n9\",\"node_id\":\"n9\",\"decision\":\"card_reward\"," +
    "\"label_kind\":\"Card reward\",\"verdict\":\"" + verdict + "\",\"strength\":\"strong preference\"," +
    "\"played\":{\"label\":\"Skip\",\"prob\":0.05,\"rank\":" + rank + "},\"top\":{\"label\":\"Inflame\",\"prob\":0.72}," +
    "\"alternatives\":[],\"peeked\":false" + extra + "}";
Check("choice flag rank 3", Chip(ChoiceJson(3)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 3rd choice");
Check("choice flag rank 11", Chip(ChoiceJson(11)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 11th choice");
Check("choice flag rank 2 / 22", Chip(ChoiceJson(22)),
    "[Info] Card reward · you: Skip | AI would take: Inflame (strong preference) · Skip was its 22nd choice");
Check("choice agree hidden", Chip(ChoiceJson(1, verdict: "agree")), "<none>");
Check("choice agree shown", Chip(ChoiceJson(1, verdict: "agree"), showGood: true), "[Good] ✓ AI agrees | ");
Check("choice tossup shown", Chip(ChoiceJson(2, verdict: "tossup"), showGood: true), "[Good] ✓ toss-up | ");
const string route = ",\"path\":{\"you\":[{\"col\":1,\"row\":2,\"hop\":0,\"hp\":60},{\"col\":1,\"row\":3,\"hop\":1,\"hp\":51.6}]," +
    "\"ai\":[{\"col\":3,\"row\":2,\"hop\":0,\"hp\":60},{\"col\":3,\"row\":3,\"hop\":1,\"hp\":58.2}]}";
Check("map route line present", Chip(ChoiceJson(2, route).Replace("Card reward", "Map")),
    "[Info] Map · you: Skip | AI would take: Inflame (strong preference) · Skip was its 2nd choice | route HP: you ≈52 · AI ≈58");
const string halfRoute = ",\"path\":{\"you\":[{\"col\":1,\"row\":2,\"hop\":0}],\"ai\":[{\"col\":3,\"row\":2,\"hop\":0,\"hp\":60}]}";
Check("map route line absent without both HPs", Chip(ChoiceJson(2, halfRoute).Replace("Card reward", "Map")),
    "[Info] Map · you: Skip | AI would take: Inflame (strong preference) · Skip was its 2nd choice");
Check("withdraw is not text", Chip("{\"kind\":\"withdraw\",\"ids\":[\"turn:n4\"]}"), "<none>");
Check("unknown kind", Chip("{\"kind\":\"recap\",\"id\":\"x\"}"), "<none>");
Check("missing kind", Chip("{\"id\":\"x\"}"), "<none>");
Check("ordinal 1/12/13/101/111", string.Join(",", new[] { 1, 12, 13, 101, 111, 112, 23 }.Select(CoachChipText.Ordinal)),
    "1st,12th,13th,101st,111th,112th,23rd");

// ChipText carries the record's run-log position (v5.1).
ChipText? Raw(string json) => CoachChipText.Format(
    (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!, showGood: false);
ChipText? located = Raw(ChoiceJson(3, ",\"log_name\":\"run_a.jsonl\",\"log_idx\":7"));
Check("chip carries log position", $"{located?.LogName}@{located?.LogIdx}", "run_a.jsonl@7");
ChipText? unlocated = Raw(ChoiceJson(3, ",\"log_name\":null,\"log_idx\":null"));
Check("chip without log position", $"{unlocated?.LogName ?? "-"}@{unlocated?.LogIdx?.ToString() ?? "-"}", "-@-");

// WorkbenchUrl.WithFollowParams: the "open in workbench" link.
Check("url: bundled with run+step",
    WorkbenchUrl.WithFollowParams("http://127.0.0.1:8079/?follow=1", "x y.jsonl", 3),
    "http://127.0.0.1:8079/?follow=1&run=x%20y.jsonl&step=3");
Check("url: configured without a query",
    WorkbenchUrl.WithFollowParams("http://localhost:5173/", "a.jsonl", 0),
    "http://localhost:5173/?follow=1&run=a.jsonl&step=0");
Check("url: step only", WorkbenchUrl.WithFollowParams("http://h/?follow=1", null, 12),
    "http://h/?follow=1&step=12");
Check("url: no params unchanged", WorkbenchUrl.WithFollowParams("http://h/?follow=1", null, null),
    "http://h/?follow=1");
Check("url: other query keeps it", WorkbenchUrl.WithFollowParams("http://h/?x=1#top", "r", 2),
    "http://h/?x=1&follow=1&run=r&step=2#top");
// WorkbenchUrl.WithLang: the workbench in the mod's language (?lang=zh).
Check("url: english adds no lang", WorkbenchUrl.WithLang("http://127.0.0.1:8079/?follow=1", "en"),
    "http://127.0.0.1:8079/?follow=1");
Check("url: zh appends lang", WorkbenchUrl.WithLang(
        WorkbenchUrl.WithFollowParams("http://127.0.0.1:8079/?follow=1", "a.jsonl", 3), "zh"),
    "http://127.0.0.1:8079/?follow=1&run=a.jsonl&step=3&lang=zh");
Check("url: zh frame page", WorkbenchUrl.WithLang("http://127.0.0.1:8079/frame.html", "zh"),
    "http://127.0.0.1:8079/frame.html?lang=zh");
Check("url: zh keeps a configured lang and fragment", WorkbenchUrl.WithLang("http://h/?lang=en#top", "zh") + " " +
    WorkbenchUrl.WithLang("http://h/#top", "zh"), "http://h/?lang=en#top http://h/?lang=zh#top");

// CoachPanelText.Format: recap / autopsy panels (plan 4 copy table).
string Panel(string json)
{
    var obj = (System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(json)!;
    PanelText? p = CoachPanelText.Format(obj);
    if (p == null)
    {
        return "<none>";
    }
    var parts = new List<string> { p.Title, p.Summary };
    foreach (PanelSection s in p.Sections)
    {
        parts.Add($"[{s.Heading}{(s.LogIdx is int i ? "@" + i : "")}] " + string.Join(" / ", s.Lines));
    }
    if (p.Footer.Length > 0)
    {
        parts.Add(p.Footer);
    }
    return string.Join(" | ", parts);
}
const string ctx = "\"context\":{\"hand\":[\"Bash\",\"Strike\",\"Defend\"],\"energy\":3,\"max_energy\":3,\"hp\":70,\"max_hp\":80,\"block\":0," +
    "\"enemies\":[{\"name\":\"Jaw Worm\",\"hp\":40,\"intent\":\"attack 11\"}]}";
string CostlyTurn(string extra) =>
    "{\"kind\":\"turn\",\"id\":\"turn:n7\",\"node_id\":\"n7\",\"turn\":2,\"floor\":5,\"verdict\":\"blunder\"," +
    "\"you\":[\"Strike\",\"end turn\"],\"ai\":[\"Bash\",\"Defend\",\"end turn\"],\"win_before\":0.97,\"win_after\":0.95," +
    "\"severity\":9,\"log_name\":\"r.jsonl\",\"log_idx\":12," + extra + "}";
string RecapJson(string extra, string costliest = "") =>
    "{\"kind\":\"recap\",\"id\":\"recap:n5\",\"node_id\":\"n5\",\"log_name\":\"r.jsonl\",\"log_idx\":10,\"floor\":5," +
    "\"room_type\":\"Monster\",\"hp_start\":80,\"hp_end\":76,\"hp_lost\":4,\"turns\":3,\"turns_graded\":3,\"unfinished_turns\":1," +
    "\"costliest\":[" + costliest + "]," + extra + "}";
Check("recap clean", Panel(RecapJson("\"ai_hp_lost\":3.2,\"turns_ungraded\":0")),
    "Fight recap · floor 5 | −4 HP (AI line typically −3)");
Check("recap without HP samples", Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0")),
    "Fight recap · floor 5 | −4 HP");
Check("recap with ungraded turns", Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":2")),
    "Fight recap · floor 5 | −4 HP · 2 turns ungraded");
Check("recap one costliest turn",
    Panel(RecapJson("\"ai_hp_lost\":6,\"turns_ungraded\":1", CostlyTurn(ctx + ",\"hp_drop\":5.4,\"peeked\":false"))),
    "Fight recap · floor 5 | −4 HP (AI line typically −6) · 1 turn ungraded" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / ≈ 5 HP more lost");
Check("recap costliest turn with a reason line",
    Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0",
        CostlyTurn(ctx + ",\"hp_drop\":5.4,\"peeked\":false,\"why\":\"Your line wakes Jaw Worm on turn 2\""))),
    "Fight recap · floor 5 | −4 HP" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / ≈ 5 HP more lost / Your line wakes Jaw Worm on turn 2");
Check("recap costliest without hp_drop -> win line, peeked",
    Panel(RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0", CostlyTurn(ctx + ",\"hp_drop\":null,\"peeked\":true"))),
    "Fight recap · floor 5 | −4 HP" +
    " | [Turn 2 — hand: Bash, Strike, Defend · 3 energy@12] Jaw Worm intends: attack 11 / You: Strike → end turn" +
    " / AI:  Bash → Defend → end turn / win 97% → 95% (peeked)");
Check("autopsy with turn and choice",
    Panel("{\"kind\":\"autopsy\",\"id\":\"autopsy:S\",\"log_name\":\"r.jsonl\",\"floor\":9,\"act\":1,\"items\":[" +
          "{\"kind\":\"turn\",\"record\":" + CostlyTurn(ctx + ",\"hp_drop\":8,\"peeked\":false") + "}," +
          "{\"kind\":\"choice\",\"record\":" + ChoiceJson(3, ",\"log_name\":\"r.jsonl\",\"log_idx\":4") + "}]," +
          "\"turns_ungraded\":1,\"decisions_graded\":14}"),
    "Run autopsy · died on floor 9 | " +
    " | [1. Combat turn@12] Floor 5 · Turn 2 — hand: Bash, Strike, Defend · 3 energy / Jaw Worm intends: attack 11" +
    " / You: Strike → end turn / AI:  Bash → Defend → end turn / ≈ 8 HP more lost" +
    " | [2. Card reward@4] Card reward · you: Skip / AI would take: Inflame (strong preference) · Skip was its 3rd choice" +
    " | 14 decisions graded · 1 turn ungraded");
Check("autopsy empty",
    Panel("{\"kind\":\"autopsy\",\"id\":\"autopsy:S\",\"log_name\":null,\"floor\":3,\"act\":1,\"items\":[],\"turns_ungraded\":0,\"decisions_graded\":2}"),
    "Run autopsy · died on floor 3 | No costly decisions found. | 2 decisions graded · 0 turns ungraded");
Check("panel unknown kind", Panel("{\"kind\":\"turn\",\"id\":\"x\"}"), "<none>");
var recapPanel = CoachPanelText.Format((System.Text.Json.Nodes.JsonObject)System.Text.Json.Nodes.JsonNode.Parse(
    RecapJson("\"ai_hp_lost\":null,\"turns_ungraded\":0", CostlyTurn(ctx + ",\"hp_drop\":5,\"peeked\":false")))!);
Check("panel carries log name and section step", $"{recapPanel?.LogName}@{recapPanel?.Sections[0].LogIdx}", "r.jsonl@12");

// SimCoachReports: payloads and the once-per-run game_over latch.
string Payload(Dictionary<string, object> d) => System.Text.Json.JsonSerializer.Serialize(d);
Check("combat_ended payload", Payload(SimCoachReports.CombatEnded(true, 4, 61, 80, "Monster")),
    "{\"type\":\"combat_ended\",\"victory\":true,\"round\":4,\"hp\":61,\"max_hp\":80,\"room_type\":\"Monster\"}");
Check("game_over payload", Payload(SimCoachReports.GameOver(false)),
    "{\"type\":\"game_over\",\"result\":\"defeat\",\"message\":\"Game over\",\"victory\":false}");
var runA = new object();
var runB = new object();
Check("latch: first claim", SimCoachReports.TryClaimGameOver(runA).ToString(), "True");
Check("latch: second claim same run", SimCoachReports.TryClaimGameOver(runA).ToString(), "False");
Check("latch: a new run claims again", SimCoachReports.TryClaimGameOver(runB).ToString(), "True");
Check("latch: no run never claims", SimCoachReports.TryClaimGameOver(null).ToString(), "False");

// AutoplayPause: `pause_on` parsing and where F7 hands back (GH #874).
string PauseOn(string confJson)
{
    using var doc = System.Text.Json.JsonDocument.Parse(confJson);
    return string.Join(",", AutoplayPause.FromJson(doc.RootElement, "pause_on", AutoplayPause.Default));
}
Check("pause_on absent -> unsupported", PauseOn("{}"), "unsupported");
Check("pause_on string list", PauseOn("{\"pause_on\":\"unsupported, Neow\"}"), "unsupported,neow");
Check("pause_on array", PauseOn("{\"pause_on\":[\"Unsupported\",\"CRYSTAL_SPHERE\"]}"), "unsupported,crystalsphere");
Check("pause_on empty string -> never", PauseOn("{\"pause_on\":\"\"}"), "");
Check("pause_on blanks and dupes", PauseOn("{\"pause_on\":\" , neow,Neow ,\"}"), "neow");
Check("pause_on null -> never", PauseOn("{\"pause_on\":null}"), "");
Check("pause_on wrong type -> default", PauseOn("{\"pause_on\":3}"), "unsupported");
Check("advisor_autoplay forces never",
    AutoplayPause.Effective(AutoplayPause.Parse("unsupported, neow"), advisorAutoplay: true).Count.ToString(), "0");
Check("attended keeps the list",
    string.Join(",", AutoplayPause.Effective(AutoplayPause.Parse("unsupported"), advisorAutoplay: false)), "unsupported");

IReadOnlyList<string> unsupportedOnly = AutoplayPause.Parse("unsupported");
Check("hand_back pauses under unsupported",
    AutoplayPause.Reason(unsupportedOnly, "card bundles have no trained policy", new[] { "card_bundle" }),
    "not covered by the AI (card bundles have no trained policy)");
Check("a real decision plays on",
    AutoplayPause.Reason(unsupportedOnly, null, new[] { "event", "NEventRoom", "NEOW" }), null);
Check("hand_back ignored without unsupported",
    AutoplayPause.Reason(AutoplayPause.Parse("Neow"), "no head", new[] { "shop" }), null);
Check("never pause",
    AutoplayPause.Reason(Array.Empty<string>(), "no head", new[] { "NEOW" }), null);
Check("a listed event id pauses",
    AutoplayPause.Reason(AutoplayPause.Parse("unsupported, Neow"), null, new[] { "event", "NEventRoom", "NEOW" }),
    "you asked to play this yourself (pause_on)");
Check("a listed advice_for pauses",
    AutoplayPause.Reason(AutoplayPause.Parse("shop"), null, new[] { "shop", "NMerchantRoom", null }),
    "you asked to play this yourself (pause_on)");
Check("an unlisted event plays on",
    AutoplayPause.Reason(AutoplayPause.Parse("Neow"), null, new[] { "event", "NEventRoom", "CRYSTAL_SPHERE" }), null);
Check("pretty event id", AutoplayPause.Pretty("CRYSTAL_SPHERE"), "Crystal Sphere");
Check("pretty advice_for", AutoplayPause.Pretty("card_bundle"), "Card Bundle");
Check("pretty nothing", AutoplayPause.Pretty(null), "This decision");
Check("toast", AutoplayPause.Toast("Crystal Sphere", "not covered by the AI (x)"),
    "Crystal Sphere: not covered by the AI (x) — your call. F7 to resume.");
{
    // The same hand-back toast in a game set to Simplified Chinese.
    Func<string?>? savedProbe = L10n.GameLanguage;
    try
    {
        L10n.GameLanguage = () => "zhs";
        Check("toast (zh)", AutoplayPause.Toast(AutoplayPause.Pretty(null), AutoplayPause.Reason(AutoplayPause.Parse("unsupported"), "x", new[] { "shop" })!),
            "本次决策：AI 无法处理（x） — 由你决定。按 F7 继续。");
    }
    finally
    {
        L10n.GameLanguage = savedProbe;
    }
}

// EnemyTagText: the "#k" beside a same-named enemy, numbered as the advisor's
// "→ Name #k" labels are (sts2-pycli/python/enemy_labels.py).
string TagList(params string?[] names) =>
    string.Join(",", EnemyTagText.Tags(names).Select(t => t ?? "-"));
Check("tags: three rats", TagList("Rat", "Rat", "Rat"), "#1,#2,#3");
Check("tags: unique names stay bare", TagList("Cultist", "Jaw Worm"), "-,-");
Check("tags: mixed, ranked within each name", TagList("Louse", "Cultist", "Louse"), "#1,-,#2");
Check("tags: a dead enemy (null) is not counted", TagList("Rat", null, "Rat"), "#1,-,#2");
Check("tags: one live rat beside a dead one stays bare", TagList("Rat", null), "-,-");
Check("tags: empty", TagList(), "");

// RefusalWatch: a move that opens a pick is not a refused move (2026-10-28).
string Judge(int markThen, int markNow, bool screenUp, int screensThen, int screensNow) =>
    RefusalWatch.Judge(markThen, markNow, screenUp, screensThen, screensNow).ToString();
Check("refusal: board moved", Judge(5, 6, false, 0, 0), "Accepted");
Check("refusal: board moved beats an open screen", Judge(5, 6, true, 0, 1), "Accepted");
Check("refusal: potion pick still open", Judge(5, 5, true, 0, 1), "WaitForPick");
Check("refusal: pick answered, card not landed yet", Judge(5, 5, false, 0, 1), "WaitForPick");
Check("refusal: nothing moved, nothing pending", Judge(5, 5, false, 3, 3), "Refused");

// RefusalFallback: a refused pick falls back to the advice's next-best move,
// not to ending the turn (autobattle, 2026-11-23).
MarkOption Opt(string label, string action, int? card = null, int? target = null, int? slot = null,
    bool best = false, double? share = null, int? rank = null) =>
    new(label, action, card, target, null, slot, null, best, false, share, null, rank);
var ranked = new List<MarkOption>
{
    Opt("Pommel Strike -> Shield", "play", card: 3, target: 1, best: true, share: 0.6),
    Opt("End turn", "end_turn", share: 0.1),
    Opt("Fight Me! -> Turret", "play", card: 1, target: 0, share: 0.3),
    Opt("Block Potion", "potion", slot: 0, share: 0.2),
    Opt("Old-advisor row", "", card: 2, share: 0.9),
};
string NextLabel(RefusalFallback f, int mark) => f.Next(mark, ranked)?.Label ?? "none";
var fallback = new RefusalFallback();
Check("fallback: nothing refused yet keeps the pick", NextLabel(fallback, 7), "Pommel Strike -> Shield");
fallback.Refuse(7, RefusalFallback.Key(ranked[0]));
Check("fallback: a refused pick hands over by share", NextLabel(fallback, 7), "Fight Me! -> Turret");
Check("fallback: the best action's key matches its option",
    fallback.IsRefused(7, RefusalFallback.Key("play", 3, 1, -1, -1)).ToString(), "True");
fallback.Refuse(7, RefusalFallback.Key(ranked[2]));
Check("fallback: then a potion", NextLabel(fallback, 7), "Block Potion");
fallback.Refuse(7, RefusalFallback.Key(ranked[3]));
Check("fallback: ending the turn is an option too", NextLabel(fallback, 7), "End turn");
Check("fallback: counts the refusals on this board", fallback.RefusedOn(7).ToString(), "3");
fallback.Refuse(7, RefusalFallback.Key(ranked[1]));
Check("fallback: a row with no action is never played", NextLabel(fallback, 7), "none");
Check("fallback: a board that moved forgets the refusals", NextLabel(fallback, 8), "Pommel Strike -> Shield");
Check("fallback: ... and counts none there", fallback.RefusedOn(8).ToString(), "0");
fallback.Refuse(8, RefusalFallback.Key(ranked[1]));
Check("fallback: refusing on a new board starts over", fallback.RefusedOn(8).ToString(), "1");
Check("fallback: the old board's refusals are gone", fallback.IsRefused(8, RefusalFallback.Key(ranked[0])).ToString(), "False");
var enginePicked = new List<MarkOption>
{
    Opt("Share leader", "play", card: 0, share: 0.9),
    Opt("Engine pick", "play", card: 1, share: 0.1, rank: 0),
    Opt("Engine second", "play", card: 2, share: 0.05, rank: 1),
};
var byRank = new RefusalFallback();
byRank.Refuse(1, RefusalFallback.Key(enginePicked[1]));
Check("fallback: the engine's pick order beats share", byRank.Next(1, enginePicked)?.Label ?? "none", "Engine second");

// TravelGuard: nothing is described mid-travel; a stuck flag gives up once
// per move (GH #987).
var travel = new TravelGuard();
const ulong stuck = TravelGuard.StuckAfterMs;
Check("travel: not traveling", travel.Suppress(false, 3, 0).ToString(), "False");
Check("travel: a move starts", travel.Suppress(true, 3, 1_000).ToString(), "True");
Check("travel: still within the limit", travel.Suppress(true, 3, 1_000 + stuck - 1).ToString(), "True");
Check("travel: past the limit gives up", travel.Suppress(true, 3, 1_000 + stuck).ToString(), "False");
Check("travel: giving up is reported once", travel.JustGaveUp.ToString(), "True");
Check("travel: stays given up for that move", travel.Suppress(true, 3, 1_000 + 2 * stuck).ToString(), "False");
Check("travel: ... without reporting again", travel.JustGaveUp.ToString(), "False");
Check("travel: the next move re-arms with the flag never dropped",
    travel.Suppress(true, 4, 1_000 + 2 * stuck).ToString(), "True");
Check("travel: arrival clears", travel.Suppress(false, 4, 1_000 + 2 * stuck + 5).ToString(), "False");
Check("travel: a later move on the same count starts a fresh clock",
    travel.Suppress(true, 4, 10 * stuck).ToString(), "True");

// TeammateTargetPick: an AnyAlly play's teammate (GH #1017). The players in
// run order; us second of four. allyIndex counts the others, as allies[] does.
var table = new List<(bool IsMe, bool IsAlive)> { (false, true), (true, true), (false, true), (false, true) };
Check("ally 0 is the first other player", TeammateTargetPick.Pick(table, 0).ToString(), "0");
Check("ally 1 skips us", TeammateTargetPick.Pick(table, 1).ToString(), "2");
Check("ally 2", TeammateTargetPick.Pick(table, 2).ToString(), "3");
Check("ally out of range", TeammateTargetPick.Pick(table, 3).ToString(), "-1");
Check("no index: first living teammate", TeammateTargetPick.Pick(table, -1).ToString(), "0");
var firstDead = new List<(bool IsMe, bool IsAlive)> { (false, false), (true, true), (false, true), (false, true) };
Check("a named dead teammate is nobody", TeammateTargetPick.Pick(firstDead, 0).ToString(), "-1");
Check("no index skips the dead", TeammateTargetPick.Pick(firstDead, -1).ToString(), "2");
var oneLeft = new List<(bool IsMe, bool IsAlive)> { (false, false), (true, true), (false, true) };
Check("one living teammate takes any index", TeammateTargetPick.Pick(oneLeft, 0).ToString(), "2");
Check("one living teammate takes an out-of-range index", TeammateTargetPick.Pick(oneLeft, 7).ToString(), "2");
var twoPlayer = new List<(bool IsMe, bool IsAlive)> { (true, true), (false, true) };
Check("two players: any index lands on the teammate", TeammateTargetPick.Pick(twoPlayer, 3).ToString(), "1");
Check("solo: nobody", TeammateTargetPick.Pick(new List<(bool IsMe, bool IsAlive)> { (true, true) }, -1).ToString(), "-1");
// AllyIndex: the reverse, for a human's AnyAlly play. Us second of four.
var seats = new List<bool> { false, true, false, false };
Check("player 0 is ally 0", TeammateTargetPick.AllyIndex(seats, 0).ToString(), "0");
Check("player 2 is ally 1", TeammateTargetPick.AllyIndex(seats, 2).ToString(), "1");
Check("player 3 is ally 2", TeammateTargetPick.AllyIndex(seats, 3).ToString(), "2");
Check("we are no ally", TeammateTargetPick.AllyIndex(seats, 1).ToString(), "-1");
Check("out of range is no ally", TeammateTargetPick.AllyIndex(seats, 4).ToString(), "-1");
var allAlive = seats.Select(me => (IsMe: me, IsAlive: true)).ToList();
foreach (int p in new[] { 0, 2, 3 })
{
    Check($"Pick undoes AllyIndex for player {p}",
        TeammateTargetPick.Pick(allAlive, TeammateTargetPick.AllyIndex(seats, p)).ToString(), p.ToString());
}

// TieBandText: the HUD's "about equal" lines from the engine's band (GH #1022).
string Tie(List<string> band, string? reason, double? gapHp) =>
    string.Join(" | ", TieBandText.Lines(band, reason, gapHp));
Check("band of one says nothing", Tie(new() { "Bash" }, "the highest value", null), "");
Check("band of two",
    Tie(new() { "Inflame+", "Strike" }, "the most searched of the equal moves", 1.2),
    "about equal: Inflame+ ≈ Strike (~1 HP) | → Inflame+: the most searched of the equal moves");
Check("an HP gap under half a point is not shown",
    Tie(new() { "Defend", "end turn" }, "block before ending the turn into damage", -0.3),
    "about equal: Defend ≈ end turn | → Defend: block before ending the turn into damage");
Check("a long band is counted, not listed",
    Tie(new() { "A", "B", "C", "D", "E" }, null, null),
    "about equal: A ≈ B ≈ C (+2)");

// InstinctEstimateText: Instinct's win estimate in the advice panel (GH #1222).
{
    InstinctEstimate? Est(string json) =>
        InstinctEstimateText.Parse(System.Text.Json.JsonDocument.Parse(json).RootElement);
    var fight = Est("{\"estimate\":{\"p_run\":0.712,\"p_fight\":0.55,\"hp_now\":40," +
                    "\"hp_loss\":{\"q10\":2,\"q25\":5,\"q50\":9.4,\"q75\":14,\"q90\":20}}}");
    var screen = Est("{\"estimate\":{\"p_run\":0.25,\"p_fight\":null,\"hp_loss\":null,\"hp_now\":40}}");
    Check("estimate: absent", (Est("{\"note\":\"x\"}") == null).ToString(), "True");
    Check("estimate: no p_run is none", (Est("{\"estimate\":{\"p_fight\":0.5}}") == null).ToString(), "True");
    Check("estimate: headline in a fight", InstinctEstimateText.Headline(fight!), "Run 71% · Fight 55%");
    Check("estimate: headline on a screen", InstinctEstimateText.Headline(screen!), "Run 25%");
    Check("estimate: hp line", InstinctEstimateText.HpLine(fight!), "This fight: −9 HP (−2 HP to −20 HP)");
    Check("estimate: no hp line off a fight", InstinctEstimateText.HpLine(screen!), null);
    Check("estimate: a heal is a plus",
        InstinctEstimateText.HpLine(new InstinctEstimate(1, 1, -6, -6, -6)), "This fight: +6 HP");
    Check("estimate: zero", InstinctEstimateText.HpDelta(0.4), "±0 HP");
    Check("estimate: pct clamps", InstinctEstimateText.Pct(1.2), "100%");
    Check("estimate: run bands",
        $"{InstinctEstimateText.RunTone(0.6)} {InstinctEstimateText.RunTone(0.45)} {InstinctEstimateText.RunTone(0.29)}",
        "Good Mid Bad");
    Check("estimate: fight bands",
        $"{InstinctEstimateText.FightTone(0.9)} {InstinctEstimateText.FightTone(0.6)} {InstinctEstimateText.FightTone(0.59)}",
        "Good Mid Bad");
}

// BugReportSite.Resolve: where the mod's bug reports go.
Check("site: setting wins", BugReportSite.Resolve(" https://a.example/ ", "https://b.example"), "https://a.example");
Check("site: baked when no setting", BugReportSite.Resolve("", "https://b.example"), "https://b.example");
Check("site: none", BugReportSite.Resolve(null, null), null);
Check("site: blank baked", BugReportSite.Resolve("  ", "  "), null);
Check("site: keeps a path and port", BugReportSite.Resolve("http://127.0.0.1:8787/wb/", null), "http://127.0.0.1:8787/wb");
// A bad explicit setting does not fall back to the baked site: the player
// typed it to send reports elsewhere, so nothing goes anywhere they did not say.
Check("site: bad setting is no site", BugReportSite.Resolve("not a url", "https://b.example"), null);
Check("site: ftp refused", BugReportSite.Resolve("ftp://x.example", null), null);
Check("site: a query is refused", BugReportSite.Resolve("https://x.example/?a=1", null), null);
Check("site: endpoint is the intake", BugReportSite.Endpoint("https://a.example"), "https://a.example/intake/bugreport");
Check("redirect: names the sign-in host",
    BugReportDump.RedirectDetail(302, "https://team.cloudflareaccess.com/cdn-cgi/access/login/x?redirect_url=%2F"),
    "HTTP 302: the report store sent us to team.cloudflareaccess.com (a sign-in page?) -- its /intake/bugreport is not exempt from the site's sign-in");
Check("redirect: no location", BugReportDump.RedirectDetail(307, null), "HTTP 307: the report store redirected instead of answering");

// BugReportDump: tails, scrubbing, the upload reply.
Check("tail: fits unchanged", BugReportDump.TailChars("a\nb\n", 10), "a\nb\n");
Check("tail: starts at a whole line", BugReportDump.TailChars("first line\nsecond\nthird\n", 10), "third\n");
Check("tail: one long line is cut raw", BugReportDump.TailChars("abcdefghij", 4), "ghij");
Check("scrub: windows path", BugReportDump.Scrub(@"C:\Users\Alice\AppData\Local", "alice"), @"C:\Users\<user>\AppData\Local");
Check("scrub: forward slashes", BugReportDump.Scrub("C:/Users/alice/x and /home/alice", "alice"), "C:/Users/<user>/x and /home/<user>");
Check("scrub: only path segments", BugReportDump.Scrub(@"steam started; C:\Users\steam\x", "steam"), @"steam started; C:\Users\<user>\x");
Check("scrub: a longer name is not a match", BugReportDump.Scrub(@"C:\Users\alicea\x", "alice"), @"C:\Users\alicea\x");
Check("scrub: no user name", BugReportDump.Scrub(@"C:\Users\alice", null), @"C:\Users\alice");
Check("note: blank is null", BugReportDump.CleanNote("  \n "), null);
Check("note: trimmed", BugReportDump.CleanNote("  card did nothing \n"), "card did nothing");
Check("note: capped", BugReportDump.CleanNote(new string('x', 5000))?.Length.ToString(), BugReportDump.MaxNoteChars.ToString());
Check("reply: id + url", BugReportDump.ParseUploadReply("{\"id\":\"wb-0123456789abcdef\",\"url\":\"https://h/?report=wb-0123456789abcdef\"}")?.Id, "wb-0123456789abcdef");
Check("reply: html is no reply", BugReportDump.ParseUploadReply("<html>oops</html>")?.Id, null);
Check("reply: missing url", BugReportDump.ParseUploadReply("{\"id\":\"wb-1\"}")?.Id, null);
Check("failure: store detail", BugReportDump.FailureDetail(413, "{\"detail\":\"report too large\"}"), "HTTP 413: report too large");
Check("failure: empty body", BugReportDump.FailureDetail(502, ""), "HTTP 502");
Check("local file name", BugReportDump.LocalFileName(new DateTimeOffset(2026, 10, 31, 9, 5, 7, TimeSpan.FromHours(1))), "bugreport-20261031-090507.json");
Check("user agent", BugReportDump.UserAgent("0.8.2"), "SpireMentor/0.8.2 (Slay the Spire 2 mod)");

// BugReportDump.Build: the dump the store validates (schema 1, four objects).
var bugState = new Dictionary<string, object>
{
    ["type"] = "advice_request",
    ["advice_for"] = "combat_action",
    ["seed"] = "ABC123",
    ["act"] = 2,
    ["floor"] = 19,
    ["run_state"] = new Dictionary<string, object?> { ["character"] = "Silent", ["ascension"] = 3 },
};
var bugDump = BugReportDump.Build(new BugReportParts
{
    Note = " Footwork did not add dexterity ",
    ModVersion = "0.8.2",
    GameVersion = "v0.111.0",
    Os = "Windows",
    CoachMode = "Advisor",
    CoachStyle = "Coach",
    AdvisorConnected = true,
    CapturedAt = new DateTimeOffset(2026, 10, 31, 9, 5, 7, TimeSpan.FromHours(1)),
    State = bugState,
    RunSaveMeta = new Dictionary<string, object?> { ["id"] = "123-4", ["floor"] = 19 },
    RunSaveJson = "{\"schema_version\":9}",
    RunLogName = "run.jsonl",
    RunLog = "{\"a\":1}\n",
    ModLog = Enumerable.Range(0, 450).Select(i => $"line {i} C:\\Users\\bob\\x").ToList(),
    GameLog = "boot\nC:/Users/bob/AppData\n",
    UserName = "bob",
});
string bugJson = System.Text.Json.JsonSerializer.Serialize(bugDump);
using (var doc = System.Text.Json.JsonDocument.Parse(bugJson))
{
    var root = doc.RootElement;
    string Str(params string[] path)
    {
        var e = root;
        foreach (string k in path)
        {
            if (e.ValueKind != System.Text.Json.JsonValueKind.Object || !e.TryGetProperty(k, out e))
            {
                return "<missing>";
            }
        }
        return e.ValueKind == System.Text.Json.JsonValueKind.String ? e.GetString()! : e.GetRawText();
    }
    Check("dump: schema", Str("schema_version"), "1");
    foreach (string k in new[] { "meta", "repro", "state", "ui" })
    {
        Check($"dump: {k} is an object", root.GetProperty(k).ValueKind.ToString(), "Object");
    }
    Check("dump: source", Str("meta", "source"), "mod");
    Check("dump: engine build is the game's", Str("meta", "engine_build"), "v0.111.0");
    Check("dump: character from run_state", Str("meta", "character"), "Silent");
    Check("dump: ascension", Str("meta", "ascension"), "3");
    Check("dump: seed", Str("meta", "seed"), "ABC123");
    Check("dump: floor", Str("meta", "floor"), "19");
    Check("dump: advisor connected", Str("meta", "coach", "advisor_connected"), "true");
    Check("dump: captured local", Str("meta", "captured_local"), "2026-10-31T09:05:07+01:00");
    Check("dump: screen from advice_for", Str("ui", "screen"), "combat_action");
    Check("dump: state payload kept", Str("state", "game_state", "advice_for"), "combat_action");
    Check("dump: run save parsed, not a string", Str("repro", "run_save", "save", "schema_version"), "9");
    Check("dump: run save snapshot id", Str("repro", "run_save", "snapshot", "id"), "123-4");
    Check("dump: run log", Str("repro", "run_log"), "{\"a\":1}\n");
    Check("dump: run file", Str("repro", "run_file"), "run.jsonl");
    Check("dump: note trimmed", Str("note"), "Footwork did not add dexterity");
    Check("dump: mod log capped", root.GetProperty("logs").GetProperty("mod").GetArrayLength().ToString(), BugReportDump.MaxModLogLines.ToString());
    Check("dump: mod log keeps the newest", root.GetProperty("logs").GetProperty("mod")[0].GetString(), @"line 50 C:\Users\<user>\x");
    Check("dump: game log scrubbed", Str("logs", "game"), "boot\nC:/Users/<user>/AppData\n");
    Check("dump: no advisor log is null", Str("logs", "advisor"), "null");
}
var bareDump = BugReportDump.Build(new BugReportParts { ModVersion = "0.8.2", RunLogName = "big.jsonl", RunLogOmitted = "9000000 bytes" });
string bareJson = System.Text.Json.JsonSerializer.Serialize(bareDump);
using (var doc = System.Text.Json.JsonDocument.Parse(bareJson))
{
    var root = doc.RootElement;
    Check("bare dump: no state is null", root.GetProperty("state").GetProperty("game_state").ValueKind.ToString(), "Null");
    Check("bare dump: no screen", root.GetProperty("ui").GetProperty("screen").ValueKind.ToString(), "Null");
    Check("bare dump: no note", root.GetProperty("note").ValueKind.ToString(), "Null");
    Check("bare dump: omitted run log says so", root.GetProperty("repro").GetProperty("run_log_omitted").GetString(), "9000000 bytes");
    Check("bare dump: no run log text", root.GetProperty("repro").TryGetProperty("run_log", out _).ToString(), "False");
    Check("bare dump: no run save", root.GetProperty("repro").TryGetProperty("run_save", out _).ToString(), "False");
}
string tailFile = Path.Combine(Path.GetTempPath(), $"sm-tail-{Guid.NewGuid():N}.log");
File.WriteAllText(tailFile, "one\ntwo\nthree\n");
Check("file tail: whole file", BugReportDump.ReadFileTail(tailFile, 1000), "one\ntwo\nthree\n");
Check("file tail: from a line start", BugReportDump.ReadFileTail(tailFile, 8), "three\n");
using (new FileStream(tailFile, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite))
{
    // godot.log is held open for writing by the game while we read it.
    Check("file tail: shared with a writer", BugReportDump.ReadFileTail(tailFile, 6), "three\n");
}
File.Delete(tailFile);
Check("file tail: missing file", BugReportDump.ReadFileTail(tailFile, 10), null);
Check("screen: run snapshot", BugReportDump.ScreenOf(new Dictionary<string, object?> { ["type"] = "run_snapshot" }), "run");

// CrystalSpherePick (GH #872): which hidden cell F7 divines.
{
    // A fresh board: the 11x11 grid minus the four Manhattan-radius-2 corner
    // diamonds the minigame clears up front, in container (x-major) order.
    var fresh = new List<(int X, int Y)>();
    for (int x = 0; x < CrystalSpherePick.GridSize; x++)
    {
        for (int y = 0; y < CrystalSpherePick.GridSize; y++)
        {
            int dx = Math.Min(x, CrystalSpherePick.GridSize - 1 - x);
            int dy = Math.Min(y, CrystalSpherePick.GridSize - 1 - y);
            if (dx + dy > 2)
            {
                fresh.Add((x, y));
            }
        }
    }
    string Cell(List<(int X, int Y)> hidden) => hidden[CrystalSpherePick.Pick(hidden)].ToString();
    Check("sphere: fresh board starts the first column", Cell(fresh), "(3, 2)");
    Check("sphere: not the first hidden cell", fresh[0].ToString(), "(0, 3)");

    // Play the six centres out: each Big divination clears its 3x3.
    var board = new List<(int X, int Y)>(fresh);
    var order = new List<string>();
    for (int t = 0; t < 6; t++)
    {
        (int cx, int cy) = board[CrystalSpherePick.Pick(board)];
        order.Add($"({cx},{cy})");
        board.RemoveAll(c => Math.Abs(c.X - cx) <= 1 && Math.Abs(c.Y - cy) <= 1);
    }
    Check("sphere: column first, then the column beside it",
        string.Join(" ", order), "(3,2) (3,5) (3,8) (6,2) (6,5) (6,8)");
    Check("sphere: the six blocks never overlap", (fresh.Count - board.Count).ToString(), "54");

    // A centre the player already cleared by hand is skipped.
    var handPlayed = new List<(int X, int Y)>(fresh);
    handPlayed.RemoveAll(c => Math.Abs(c.X - 3) <= 1 && Math.Abs(c.Y - 2) <= 1);
    Check("sphere: a cleared centre is skipped", Cell(handPlayed), "(3, 5)");

    // Past the list: most still-hidden cells in the 3x3, nearest the centre.
    var leftovers = new List<(int X, int Y)> { (0, 5), (9, 9), (9, 10), (10, 9), (10, 10), (5, 0) };
    Check("sphere: fallback takes the densest block", Cell(leftovers), "(9, 9)");
    var lone = new List<(int X, int Y)> { (0, 5), (5, 4), (10, 5) };
    Check("sphere: fallback tie goes nearest the centre", Cell(lone), "(5, 4)");
    Check("sphere: nothing hidden", CrystalSpherePick.Pick(new List<(int X, int Y)>()).ToString(), "-1");
}

// CrystalSphereWire: what the crystal_sphere message shows the digger, and
// whether (and how) F7 takes the advisor's pick.
{
    Check("sphere wire: relic", CrystalSphereWire.ItemClass("CrystalSphereRelic"), "relic");
    Check("sphere wire: potion", CrystalSphereWire.ItemClass("CrystalSpherePotion"), "potion");
    Check("sphere wire: card", CrystalSphereWire.ItemClass("CrystalSphereCardReward"), "card");
    Check("sphere wire: curse", CrystalSphereWire.ItemClass("CrystalSphereCurse"), "curse");
    Check("sphere wire: gold", CrystalSphereWire.ItemClass("CrystalSphereGold"), "gold");
    Check("sphere wire: unknown item", CrystalSphereWire.ItemClass("Banana"), null);

    // Cleared relic piece, cleared empty cell, hidden relic, hidden empty:
    // only the cleared relic piece is on the wire.
    var cells = new List<(int X, int Y, bool Hidden, string? ItemTypeName)>
    {
        (3, 4, false, "CrystalSphereRelic"),
        (3, 5, false, null),
        (6, 6, true, "CrystalSphereRelic"),
        (7, 7, true, null),
    };
    Check("sphere wire: revealed shows cleared items only",
        System.Text.Json.JsonSerializer.Serialize(CrystalSphereWire.Revealed(cells)),
        """[{"x":3,"y":4,"item":"relic"}]""");

    int Cap(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        return CrystalSphereWire.Capability(doc.RootElement);
    }
    Check("sphere cap: top level", Cap("""{"capabilities":{"coach":2,"crystal_sphere":1},"advice":{}}""").ToString(), "1");
    Check("sphere cap: inside advice", Cap("""{"advice":{"capabilities":{"crystal_sphere":1}}}""").ToString(), "1");
    Check("sphere cap: absent (older advisor)", Cap("""{"capabilities":{"coach":2,"potion_swap":1},"advice":{}}""").ToString(), "0");

    // The advisor's pick, or -1 (play CrystalSpherePick): F7 never stalls.
    int Advised(string json, int options)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        return CrystalSphereWire.AdvisedIndex(doc.RootElement, options);
    }
    Check("sphere advice: in range",
        Advised("""{"advice":{"best":{"action":"choose","index":2,"label":"reveal cell (6,5)"}}}""", 5).ToString(), "2");
    Check("sphere advice: hand_back falls back",
        Advised("""{"advice":{"note":"n","hand_back":"nothing to choose"}}""", 5).ToString(), "-1");
    Check("sphere advice: out of range falls back",
        Advised("""{"advice":{"best":{"action":"choose","index":5,"label":"proceed"}}}""", 5).ToString(), "-1");
    Check("sphere advice: negative falls back",
        Advised("""{"advice":{"best":{"action":"choose","index":-1}}}""", 5).ToString(), "-1");
    Check("sphere advice: not a choose falls back",
        Advised("""{"advice":{"best":{"action":"skip","index":0}}}""", 5).ToString(), "-1");
    Check("sphere advice: no advice falls back", Advised("""{"action":"advice"}""", 5).ToString(), "-1");

    // The policy that decided, so the action log never credits the digger
    // for a click the advisor's #872 rule made.
    string? Policy(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        return CrystalSphereWire.AdvisedPolicy(doc.RootElement);
    }
    Check("sphere policy: digger",
        Policy("""{"advice":{"best":{"action":"choose","index":2},"policy":"crystal_sphere_digger"}}"""),
        "crystal_sphere_digger");
    Check("sphere policy: rule",
        Policy("""{"advice":{"best":{"action":"choose","index":2},"policy":"crystal_sphere_rule"}}"""),
        "crystal_sphere_rule");
    Check("sphere policy: absent", Policy("""{"advice":{"best":{"action":"choose","index":2}}}"""), null);
    Check("sphere policy: not a string", Policy("""{"advice":{"policy":7}}"""), null);
    Check("sphere policy: no advice", Policy("""{"action":"advice"}"""), null);
    Check("sphere source: digger", CrystalSphereWire.ClickSource(true, "crystal_sphere_digger"), "advisor's digger");
    Check("sphere source: advisor's rule", CrystalSphereWire.ClickSource(true, "crystal_sphere_rule"), "advisor's #872 rule");
    Check("sphere source: unnamed", CrystalSphereWire.ClickSource(true, null), "advisor, policy not named");
    Check("sphere source: other", CrystalSphereWire.ClickSource(true, "x"), "advisor's x");
    Check("sphere source: not advised", CrystalSphereWire.ClickSource(false, "crystal_sphere_digger"), "block order");

    // The delayed click only stands while nobody took over.
    var f7 = new CrystalSphereWire.Takeover(true, true, 4);
    Check("sphere click: F7 still on", CrystalSphereWire.MayStillClick(f7, f7).ToString(), "True");
    Check("sphere click: F7 pressed off",
        CrystalSphereWire.MayStillClick(f7, new(true, false, 5)).ToString(), "False");
    Check("sphere click: F7 off and on again",
        CrystalSphereWire.MayStillClick(f7, new(true, true, 6)).ToString(), "False");
    Check("sphere click: Advisor mode left",
        CrystalSphereWire.MayStillClick(f7, new(false, false, 5)).ToString(), "False");
    Check("sphere click: mode left, flags unchanged",
        CrystalSphereWire.MayStillClick(f7, f7 with { AdvisorMode = false }).ToString(), "False");
    var f6 = new CrystalSphereWire.Takeover(true, false, 2);
    Check("sphere click: F6 one-shot", CrystalSphereWire.MayStillClick(f6, f6).ToString(), "True");
    Check("sphere click: F6 then F7 on",
        CrystalSphereWire.MayStillClick(f6, new(true, true, 3)).ToString(), "False");
    Check("sphere click: never outside Advisor mode",
        CrystalSphereWire.MayStillClick(f6 with { AdvisorMode = false }, f6 with { AdvisorMode = false }).ToString(), "False");
}

// AdviceColumn: the advice panel + coach chips follow a dragged toolbar.
{
    string At((float X, float Y) p) => $"({p.X:0},{p.Y:0})";
    // 1280x720, toolbar 40 px tall; block 430 wide.
    Check("column: under the strip, at its left edge",
        At(AdviceColumn.BlockTopLeft(300, 200, 40, 430, 150, 1280, 720)), "(300,248)");
    Check("column: shifted left to stay on screen",
        At(AdviceColumn.BlockTopLeft(1200, 200, 40, 430, 150, 1280, 720)), "(850,248)");
    Check("column: strip near the bottom -> above it",
        At(AdviceColumn.BlockTopLeft(300, 600, 40, 430, 150, 1280, 720)), "(300,442)");
    Check("column: fits nowhere, more room below -> below",
        At(AdviceColumn.BlockTopLeft(300, 100, 40, 430, 900, 1280, 720)), "(300,148)");
    Check("column: fits nowhere, more room above -> top of the screen",
        At(AdviceColumn.BlockTopLeft(300, 500, 40, 430, 900, 1280, 720)), "(300,0)");
    Check("column: viewport narrower than the block -> left edge",
        At(AdviceColumn.BlockTopLeft(300, 100, 40, 430, 150, 400, 720)), "(0,148)");
    Check("column: panel + chips", AdviceColumn.BlockHeight(150, true, 60).ToString("0"), "218");
    Check("column: panel hidden -> chips only", AdviceColumn.BlockHeight(150, false, 60).ToString("0"), "60");
    Check("column: no chips -> panel only", AdviceColumn.BlockHeight(150, true, 0).ToString("0"), "150");
    Check("column: nothing", AdviceColumn.BlockHeight(150, false, 0).ToString("0"), "0");
}

// RewardSwapLayout (GH #1069): reward_screen option indices with full-belt potion swaps.
{
    string L(List<RewardOptionSlot> slots) => string.Join(" ", slots.Select(s => s.Kind switch
    {
        RewardOptionKind.Pick => $"P{s.Button}",
        RewardOptionKind.SwapDecline => $"D{s.Button}",
        RewardOptionKind.Swap => $"S{s.Button}:{s.DiscardSlot}",
        _ => "X",
    }));
    Check("rewards: no blocked potion is the old layout",
        L(RewardSwapLayout.Build(3, new List<int>(), true)), "P0 P1 P2 X");
    Check("rewards: one blocked potion, 3-slot belt",
        L(RewardSwapLayout.Build(2, new List<int> { 3 }, true)), "P0 P1 D0 S0:0 S0:1 S0:2 X");
    Check("rewards: two blocked potions, no proceed button",
        L(RewardSwapLayout.Build(0, new List<int> { 2, 2 }, false)), "D0 S0:0 S0:1 D1 S1:0 S1:1");
    Check("rewards: zero-slot belt still offers decline",
        L(RewardSwapLayout.Build(1, new List<int> { 0 }, true)), "P0 D0 X");
}

// Autobattle: which decisions stop for the player, and the "since your last
// stop" chip (the workbench's autobattle.py, same expectations).
{
    string Stop(string? adviceFor, bool local, bool inCombat) =>
        Autobattle.IsStop(adviceFor, local, inCombat).ToString();
    Check("autobattle: a combat move plays", Stop("combat_action", false, true), "False");
    Check("autobattle: a card pick inside a fight plays", Stop("card_select", false, true), "False");
    Check("autobattle: an in-fight select confirm plays", Stop("card_select", true, true), "False");
    Check("autobattle: card reward stops", Stop("card_reward", false, false), "True");
    Check("autobattle: map stops", Stop("map_select", false, false), "True");
    Check("autobattle: deck removal stops", Stop("card_select", false, false), "True");
    Check("autobattle: the player's own selection is not confirmed for them", Stop("card_select", true, false), "True");
    Check("autobattle: claiming the loot plays", Stop("reward_screen", true, false), "False");
    Check("autobattle: opening the chest plays", Stop("treasure", true, false), "False");
    Check("autobattle: game over never stops", Stop("game_over", true, false), "False");

    AutobattleSnapshot Snap(int hp = 50, int gold = 99, string[]? deck = null, string[]? relics = null,
        string[]? potions = null, int floor = 1, int maxHp = 75) =>
        new(floor, hp, maxHp, gold,
            (deck ?? Array.Empty<string>())
                .Select(c => c.EndsWith('+') ? (c.TrimEnd('+'), true) : (c, false)).ToList(),
            relics ?? Array.Empty<string>(), potions ?? Array.Empty<string>());
    string Lines(AutobattleSnapshot? prev, AutobattleSnapshot cur, int autoplayed = 0) =>
        string.Join(" | ", Autobattle.Lines(prev, cur, autoplayed));

    Check("autobattle: first stop reports absolutes only",
        Lines(null, Snap(deck: new[] { "Strike", "Defend" })), "HP 50/75 · Gold 99 · Deck 2");
    Check("autobattle: adds and removes",
        Lines(Snap(deck: new[] { "Strike", "Strike", "Defend" }), Snap(deck: new[] { "Strike", "Defend", "Inflame" })),
        "Added Inflame | Removed Strike | HP 50/75 · Gold 99 · Deck 3");
    Check("autobattle: an upgrade is not a removal plus an addition",
        Lines(Snap(deck: new[] { "Strike", "Strike" }), Snap(deck: new[] { "Strike", "Strike+" })),
        "Upgraded Strike | HP 50/75 · Gold 99 · Deck 2");
    Check("autobattle: a surplus upgraded copy is added",
        Lines(Snap(deck: new[] { "Strike", "Strike" }), Snap(deck: new[] { "Strike", "Strike+", "Strike+" })),
        "Upgraded Strike | Added Strike+ | HP 50/75 · Gold 99 · Deck 3");
    Check("autobattle: relics, potions and resources",
        Lines(Snap(hp: 60, gold: 120, relics: new[] { "Burning Blood" }, potions: new[] { "Fire Potion" }),
            Snap(hp: 48, gold: 45, relics: new[] { "Burning Blood", "Bag of Preparation" }), autoplayed: 7),
        "Relic Bag of Preparation | Used potion Fire Potion | HP 48/75 (−12) · Gold 45 (−75) · Deck 0 | 7 decisions auto-played");
    Check("autobattle: one decision auto-played",
        Lines(Snap(hp: 40), Snap(hp: 45), autoplayed: 1), "HP 45/75 (+5) · Gold 99 · Deck 0 | 1 decision auto-played");
    Check("autobattle: nothing changed",
        Lines(Snap(deck: new[] { "Strike" }), Snap(deck: new[] { "Strike" })), "Nothing changed. | HP 50/75 · Gold 99 · Deck 1");
    Check("autobattle: a new floor alone is not a move",
        Autobattle.Moved(Snap(floor: 3, maxHp: 75), Snap(floor: 4, maxHp: 80)).ToString(), "False");
    Check("autobattle: a used potion is a move",
        Autobattle.Moved(Snap(potions: new[] { "Fire Potion" }), Snap()).ToString(), "True");
    ChipText first = Autobattle.Chip("autobattle:1", null, Snap(floor: 3), 0);
    Check("autobattle: first chip title", first.Title, "Autobattle — your call — floor 3");
    Check("autobattle: later chip title", Autobattle.Chip("autobattle:2", Snap(), Snap(floor: 7), 3).Title,
        "Since your last stop — floor 7");
    Check("autobattle: chip kind", first.Kind, Autobattle.ChipKind);
}

// Speed hotkeys step in whole speeds; autobattle raises 1x to 2x and only 1x.
{
    string Up(double s) => SpeedSteps.Label(SpeedSteps.Step(s, 1));
    string Down(double s) => SpeedSteps.Label(SpeedSteps.Step(s, -1));
    Check("speed: up from 1x", Up(1.0), "2x");
    Check("speed: up from 3x", Up(3.0), "4x");
    Check("speed: up from 4x goes to 6x", Up(4.0), "6x");
    Check("speed: up from 6x goes to 8x", Up(6.0), "8x");
    Check("speed: up stops at 8x", Up(8.0), "8x");
    Check("speed: down from 8x", Down(8.0), "6x");
    Check("speed: down from 6x", Down(6.0), "4x");
    Check("speed: 5x from the slider goes up to 6x", Up(5.0), "6x");
    Check("speed: 5x from the slider goes down to 4x", Down(5.0), "4x");
    Check("speed: down from 4x", Down(4.0), "3x");
    Check("speed: down stops at 1x", Down(1.0), "1x");
    Check("speed: between steps goes up to the next whole speed", Up(1.5), "2x");
    Check("speed: between steps goes down to the next whole speed", Down(1.5), "1x");
    Check("speed: a value a hair under a step counts as that step", Up(1.9999999), "3x");
    Check("speed: below 1x, up goes to 1x", Up(0.5), "1x");
    Check("speed: below 1x, down never speeds up", Down(0.5), "0.5x");
    Check("speed: autobattle raises 1x", SpeedSteps.AutobattleOn(1.0)?.ToString() ?? "keep", "2");
    Check("speed: autobattle keeps 1.5x", SpeedSteps.AutobattleOn(1.5)?.ToString() ?? "keep", "keep");
    Check("speed: autobattle keeps 3x", SpeedSteps.AutobattleOn(3.0)?.ToString() ?? "keep", "keep");
    Check("speed: autobattle keeps 0.5x", SpeedSteps.AutobattleOn(0.5)?.ToString() ?? "keep", "keep");
    Check("speed: off restores an untouched 2x", SpeedSteps.RestoreOnOff(2.0).ToString(), "True");
    Check("speed: off keeps a speed the player changed", SpeedSteps.RestoreOnOff(3.0).ToString(), "False");
}

// A hand card's angle eases like NHandCardHolder.AnimAngle: per frame,
// Lerp(angle, target, delta * 10), with delta = speed / fps. At 8x and 30 fps
// the raw weight is 2.67 and the card spins off; the stable weight lands.
{
    string Ease(double speed, double fps, bool stable)
    {
        float angle = 30f, target = 0f;
        for (int frame = 0; frame < 60; frame++)
        {
            float w = (float)(speed / fps) * 10f;
            angle += (target - angle) * (stable ? LerpWeight.Stable(w) : w);
            if (Math.Abs(angle - target) < 0.1f)
            {
                return $"settled at frame {frame}";
            }
        }
        return Math.Abs(angle) > 1000f ? "runaway" : $"unsettled {angle:0.##}";
    }
    Check("lerp: 8x at 30 fps runs away without the cap", Ease(8, 30, stable: false), "runaway");
    Check("lerp: 8x at 30 fps lands with the cap", Ease(8, 30, stable: true), "settled at frame 0");
    Check("lerp: 1x at 60 fps is untouched by the cap", Ease(1, 60, stable: true), Ease(1, 60, stable: false));
    Check("lerp: weight 0.5 is kept", LerpWeight.Stable(0.5f).ToString(CultureInfo.InvariantCulture), "0.5");
    Check("lerp: weight 2.67 becomes 1", LerpWeight.Stable(2.67f).ToString(CultureInfo.InvariantCulture), "1");
}

// The action log sits under the advice column until dragged.
{
    string Below(float bottom, float want, float view)
    {
        (float top, float height) = LogPanelPlacement.BelowColumn(bottom, want, view);
        return $"{top:0} {height:0}";
    }
    Check("log: fits under the column", Below(400f, 450f, 1080f), "408 450");
    Check("log: shortened to the room left", Below(700f, 450f, 1080f), "708 364");
    Check("log: never below its minimum, overlapping instead", Below(1000f, 450f, 1080f), "932 140");
    Check("log: a short log is not stretched to the minimum", Below(1000f, 100f, 1080f), "972 100");
    bool parsed = LogPanelPlacement.TryParse(LogPanelPlacement.Format(23.4f, 324f), out float lx, out float ly);
    Check("log: position round-trips", $"{parsed} {lx} {ly}", "True 23 324");
    Check("log: blank follows the column", LogPanelPlacement.TryParse("", out _, out _).ToString(), "False");
    Check("log: junk follows the column", LogPanelPlacement.TryParse("12;x", out _, out _).ToString(), "False");
}

// AI partner seats: the lobby's click cycles characters, kept per seat;
// setup holds wait long for a human, briefly for another AI partner.
{
    Check("partner: next wraps", PartnerCharacters.Cycle(4, 5, 1).ToString(), "0");
    Check("partner: back wraps", PartnerCharacters.Cycle(0, 5, -1).ToString(), "4");
    Check("partner: next", PartnerCharacters.Cycle(1, 5, 1).ToString(), "2");
    Check("partner: unknown current, next starts at the first", PartnerCharacters.Cycle(-1, 5, 1).ToString(), "0");
    Check("partner: unknown current, back starts at the last", PartnerCharacters.Cycle(-1, 5, -1).ToString(), "4");
    Check("partner: no characters", PartnerCharacters.Cycle(0, 0, 1).ToString(), "-1");
    Check("partner: seat 1 defaults to the second character", PartnerCharacters.DefaultIndex(0, 5).ToString(), "1");
    Check("partner: seat 3 defaults to the fourth", PartnerCharacters.DefaultIndex(2, 5).ToString(), "3");
    Check("partner: saved seat", PartnerCharacters.ForSeat("SILENT,,DEFECT", 2), "DEFECT");
    Check("partner: unset seat", PartnerCharacters.ForSeat("SILENT,,DEFECT", 1) ?? "none", "none");
    Check("partner: seat past the list", PartnerCharacters.ForSeat("SILENT", 2) ?? "none", "none");
    Check("partner: blank prefs", PartnerCharacters.ForSeat("", 0) ?? "none", "none");
    Check("partner: set a later seat", PartnerCharacters.WithSeat("", 2, "DEFECT"), ",,DEFECT");
    Check("partner: replace a seat", PartnerCharacters.WithSeat("SILENT,REGENT", 1, "IRONCLAD"), "SILENT,IRONCLAD");
    Check("partner: clearing the last seat trims", PartnerCharacters.WithSeat("SILENT,REGENT", 1, ""), "SILENT");
    Check("partner: a comma cannot split a seat", PartnerCharacters.WithSeat("", 0, "A,B"), "AB");
    Check("partner: human hold is the setting", PartnerCharacters.HoldSeconds(10, true).ToString(), "10");
    Check("partner: human hold capped at 60 s", PartnerCharacters.HoldSeconds(600, true).ToString(), "60");
    Check("partner: AI-only hold is short", PartnerCharacters.HoldSeconds(10, false).ToString(System.Globalization.CultureInfo.InvariantCulture), "2.5");
    Check("partner: AI-only hold never above the setting", PartnerCharacters.HoldSeconds(1, false).ToString(), "1");
    Check("partner: 0 never holds, even for an AI", PartnerCharacters.HoldSeconds(0, false).ToString(), "0");
    Check("partner: negative setting never holds", PartnerCharacters.HoldSeconds(-3, true).ToString(), "0");
}

// Advisor brain (PROTOCOL.md section 5, v5.2): Instinct is the only brain a
// player gets, the search its fallback; the wire form, the advisor's
// capabilities.instinct, the brain each seat asks for, which reply is stale,
// and the HUD's source tag.
{
    Check("brain: default is search", AdvisorBrainRules.Normalize(null), "search");
    Check("brain: blank is search", AdvisorBrainRules.Normalize("  "), "search");
    Check("brain: case/trim", AdvisorBrainRules.Normalize(" Instinct "), "instinct");
    Check("brain: unknown is search", AdvisorBrainRules.Normalize("e2e"), "search");

    string Caps(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        return string.Join(";", AdvisorBrainRules.ParseCapability(doc.RootElement)
            .Select(t => $"{t.Character}@{t.GameVersion}"));
    }
    Check("caps: top level", Caps("""{"capabilities":{"coach":2,"instinct":[{"character":"Ironclad","game_version":"v0.111.0"}]},"advice":{}}"""), "Ironclad@v0.111.0");
    Check("caps: inside advice", Caps("""{"advice":{"capabilities":{"instinct":[{"character":"Ironclad","game_version":"v0.111.0"}]}}}"""), "Ironclad@v0.111.0");
    Check("caps: top level without instinct falls through to advice", Caps("""{"capabilities":{"coach":2},"advice":{"capabilities":{"instinct":[{"character":"Silent","game_version":"v0.111.0"}]}}}"""), "Silent@v0.111.0");
    Check("caps: absent (older advisor)", Caps("""{"capabilities":{"coach":2},"advice":{"note":"x"}}"""), "");
    Check("caps: empty list (no net)", Caps("""{"capabilities":{"instinct":[]}}"""), "");
    Check("caps: malformed entries skipped", Caps("""{"capabilities":{"instinct":[1,{"character":"Ironclad"},{"character":"","game_version":"v0.111.0"},{"character":"Defect","game_version":"v0.111.0"}]}}"""), "Defect@v0.111.0");
    Check("caps: not an array", Caps("""{"capabilities":{"instinct":{"character":"Ironclad","game_version":"v0.111.0"}}}"""), "");

    var ironclad111 = new List<InstinctTarget> { new("Ironclad", "v0.111.0") };
    string E(string? character, string? version) => AdvisorBrainRules.Eligible(ironclad111, character, version).ToString();
    Check("eligible: run id is upper case", E("IRONCLAD", "v0.111.0"), "True");
    Check("eligible: build without v", E("Ironclad", "0.111.0"), "True");
    Check("eligible: other character", E("SILENT", "v0.111.0"), "False");
    Check("eligible: other build", E("IRONCLAD", "v0.107.1"), "False");
    Check("eligible: hotfix of the build", E("IRONCLAD", "v0.111.2"), "True");
    Check("eligible: next minor", E("IRONCLAD", "v0.112.0"), "False");
    Check("eligible: unknown build is the stable one", E("IRONCLAD", null), "False");
    Check("eligible: no run", E(null, "v0.111.0"), "False");
    Check("eligible: no capability", AdvisorBrainRules.Eligible(new List<InstinctTarget>(), "IRONCLAD", "v0.111.0").ToString(), "False");

    Check("requested: Instinct where it plays", AdvisorBrainRules.Requested("instinct", known: true, eligible: true), "instinct");
    Check("requested: Search, quietly, where it does not", AdvisorBrainRules.Requested("instinct", known: true, eligible: false), "search");
    Check("requested: Instinct before the advisor said where it plays", AdvisorBrainRules.Requested("instinct", known: false, eligible: false), "instinct");
    Check("requested: a tooling Search stays Search where Instinct plays", AdvisorBrainRules.Requested("search", known: true, eligible: true), "search");
    Check("requested: a tooling Search before the capability too", AdvisorBrainRules.Requested("search", known: false, eligible: false), "search");

    // The player's brain: Instinct, whatever an older version's switch saved
    // (BaseLib's AdvisorBrain is read by nothing: Player takes no saved value).
    Check("player: Instinct with no tooling override", AdvisorBrainRules.Player(null), "instinct");
    Check("player: the default is Instinct", AdvisorBrainRules.PlayerDefault, "instinct");
    Check("player: a typo in the override is still Instinct", AdvisorBrainRules.Player("serach"), "instinct");
    Check("player: tooling can force Search", AdvisorBrainRules.Player("search"), "search");
    Check("player: tooling Instinct", AdvisorBrainRules.Player(" Instinct "), "instinct");
    Check("player asks Instinct where capable", AdvisorBrainRules.Requested(AdvisorBrainRules.Player(null), known: true, eligible: true), "instinct");
    Check("player asks Search only where Instinct is not offered", AdvisorBrainRules.Requested(AdvisorBrainRules.Player(null), known: true, eligible: false), "search");
    Check("player asks Instinct before the capability (an older advisor falls back itself)",
        AdvisorBrainRules.Requested(AdvisorBrainRules.Player(null), known: false, eligible: false), "instinct");

    // The AI partners' brain: its own named default, Instinct, sent as it is.
    Check("partner: the default is Instinct", AdvisorBrainRules.PartnerDefault, "instinct");
    Check("partner: Instinct with no tooling override", AdvisorBrainRules.Partner(null), "instinct");
    Check("partner: tooling can force Search", AdvisorBrainRules.Partner("SEARCH"), "search");
    Check("partner: a typo is no override", AdvisorBrainRules.Partner("x"), "instinct");

    Check("stale: asked of search, now instinct", AdvisorBrainRules.IsStale("search", "instinct").ToString(), "True");
    Check("stale: asked of instinct, now search", AdvisorBrainRules.IsStale("instinct", "search").ToString(), "True");
    Check("stale: same brain is never stale (a fallback renders)", AdvisorBrainRules.IsStale("instinct", "instinct").ToString(), "False");
    Check("stale: wire spelling does not matter", AdvisorBrainRules.IsStale("Search", "").ToString(), "False");

    Check("tag: instinct answered", AdvisorBrainRules.SourceTag("instinct", null), "Instinct");
    Check("tag: fallback names the reason", AdvisorBrainRules.SourceTag("search", " Instinct can't read this event yet "),
        "Search (Instinct: Instinct can't read this event yet)");
    Check("tag: fallback with brain absent", AdvisorBrainRules.SourceTag(null, "no net"), "Search (Instinct: no net)");
    Check("tag: plain search says nothing", AdvisorBrainRules.SourceTag("search", null), null);
    Check("tag: older advisor says nothing", AdvisorBrainRules.SourceTag(null, null), null);

    Check("note: instinct prefixed", AdvisorBrainRules.NoteWithSource("e2e net", "instinct", null), "Instinct · e2e net");
    Check("note: instinct without a note", AdvisorBrainRules.NoteWithSource(null, "instinct", null), "Instinct");
    Check("note: reason already in the note is not repeated",
        AdvisorBrainRules.NoteWithSource("MCTS 3000 sims · Instinct can't read this event yet", "search", "Instinct can't read this event yet"),
        "MCTS 3000 sims · Instinct can't read this event yet");
    Check("note: reason missing from the note is added",
        AdvisorBrainRules.NoteWithSource("MCTS 3000 sims", "search", "no net"), "Search (Instinct: no net) · MCTS 3000 sims");
    Check("note: plain search untouched", AdvisorBrainRules.NoteWithSource("MCTS 3000 sims", null, null), "MCTS 3000 sims");

    // The test tooling's override (SpireMentor.conf advisor_brain, the
    // --spire-mentor-brain launch arg): never a guess.
    Check("conf: instinct", AdvisorBrainRules.ParseConf(" INSTINCT "), "instinct");
    Check("conf: search", AdvisorBrainRules.ParseConf("Search"), "search");
    Check("conf: absent", AdvisorBrainRules.ParseConf(null), null);
    Check("conf: blank", AdvisorBrainRules.ParseConf(""), null);
    Check("conf: typo is no override", AdvisorBrainRules.ParseConf("instict"), null);
    Check("tooling: launch arg over the conf", AdvisorBrainRules.ToolingOverride("search", "instinct"), "search");
    Check("tooling: launch arg Instinct over a conf Search", AdvisorBrainRules.ToolingOverride("instinct", "search"), "instinct");
    Check("tooling: no launch arg, the conf", AdvisorBrainRules.ToolingOverride(null, "Search"), "search");
    Check("tooling: a bad launch arg leaves the conf", AdvisorBrainRules.ToolingOverride("x", "search"), "search");
    Check("tooling: nothing set is no override", AdvisorBrainRules.ToolingOverride(null, null), null);
    Check("tooling: nothing set, both seats Instinct",
        AdvisorBrainRules.Player(AdvisorBrainRules.ToolingOverride(null, null)) + "/" +
        AdvisorBrainRules.Partner(AdvisorBrainRules.ToolingOverride(null, null)), "instinct/instinct");
    Check("tooling: a Search override reaches both seats",
        AdvisorBrainRules.Player(AdvisorBrainRules.ToolingOverride("search", null)) + "/" +
        AdvisorBrainRules.Partner(AdvisorBrainRules.ToolingOverride("search", null)), "search/search");
}

// Full-belt potion swap as an advice lane (PROTOCOL.md section 5, v5.2).
{
    int Cap(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        return PotionSwapAdvice.CapabilityOf(doc.RootElement);
    }
    Check("swap cap: top level", Cap("""{"capabilities":{"coach":2,"potion_swap":1},"advice":{}}""").ToString(), "1");
    Check("swap cap: inside advice", Cap("""{"advice":{"capabilities":{"potion_swap":2}}}""").ToString(), "2");
    Check("swap cap: top level without it falls through", Cap("""{"capabilities":{"coach":2},"advice":{"capabilities":{"potion_swap":1}}}""").ToString(), "1");
    Check("swap cap: absent (older advisor)", Cap("""{"capabilities":{"coach":2},"advice":{}}""").ToString(), "0");
    Check("swap cap: not a number", Cap("""{"capabilities":{"potion_swap":true}}""").ToString(), "0");
    Check("swap cap: negative clamps", Cap("""{"capabilities":{"potion_swap":-3}}""").ToString(), "0");

    Check("swap ask: blocked, nothing claimable, capable", PotionSwapAdvice.ShouldAsk(0, 1, 1).ToString(), "True");
    Check("swap ask: claim the rest first", PotionSwapAdvice.ShouldAsk(2, 1, 1).ToString(), "False");
    Check("swap ask: older advisor keeps today's loot screen", PotionSwapAdvice.ShouldAsk(0, 1, 0).ToString(), "False");
    Check("swap ask: nothing blocked", PotionSwapAdvice.ShouldAsk(0, 0, 1).ToString(), "False");

    var belt = new List<string?> { "FIRE_POTION", "BLOCK_POTION", "FIRE_POTION" };
    Check("swap labels: decline, then the autopilot's discard labels",
        string.Join("|", PotionSwapAdvice.Labels(belt)), "decline|discard FIRE_POTION|discard BLOCK_POTION|discard FIRE_POTION");
    string Slot(int i) => PotionSwapAdvice.DiscardSlotOf(i, belt.Count)?.ToString() ?? "decline";
    Check("swap slot: 0 declines", Slot(0), "decline");
    Check("swap slot: 1 is belt slot 0", Slot(1), "0");
    Check("swap slot: last", Slot(3), "2");
    Check("swap slot: a skip (index = option count) declines", Slot(4), "decline");
    Check("swap slot: negative declines", Slot(-1), "decline");
    // Autobattle holds it for the player: a real run-level choice, not a loot click.
    Check("swap: autobattle stops", Autobattle.IsStop(PotionSwapAdvice.AdviceFor, local: false, inCombat: false).ToString(), "True");
}

// AdviceMarks: what the HUD rings and numbers.
{
    MarkOption Play(int card, int? target, double share, bool best = false, double? q = null, string label = "") =>
        new(label, "play", card, target, null, null, null, best, false, share, q, null);
    string Show(List<Mark> ms) => string.Join(" ", ms.Select(m =>
        $"{m.Kind}{m.Key}:{m.Tier}{(m.Order is int o ? "#" + o : "")}{(m.TargetIndex is int t ? ">" + t : "")}"));

    // A targeted card has one option per target: one mark, the best target's.
    var twoTargets = new List<MarkOption>
    {
        Play(0, 1, 0.70, best: true), Play(0, 0, 0.20), Play(1, null, 0.10),
    };
    Check("marks: one mark per card, the pick's target",
        Show(AdviceMarks.Build(twoTargets, null)), "Card0:Best#1>1");

    // 0% moves are not shown; a real alternative gets a thin ring.
    var spread = new List<MarkOption>
    {
        Play(0, null, 0.60, best: true), Play(1, null, 0.30), Play(2, null, 0.004),
        new("End turn", "end_turn", null, null, null, null, null, false, false, 0.06, null, null),
    };
    Check("marks: hidden 0%, alt ring from 15%", Show(AdviceMarks.Build(spread, null)), "Card0:Best#1 Card1:Alt");

    // The plan numbers the turn; its target wins over the option's.
    var plan = new List<MarkPlanStep>
    {
        new("play", 2, 0, null, null, "Bash"), new("play", 0, 1, null, null, "Strike"),
        new("end_turn", null, null, null, null, "end turn"),
    };
    var withPlan = new List<MarkOption> { Play(2, 0, 0.8, best: true), Play(0, 0, 0.1), Play(1, null, 0.1) };
    Check("marks: plan order 1,2 + end turn",
        Show(AdviceMarks.Build(withPlan, plan)), "Card2:Best#1>0 Card0:Good#2>1 EndTurn0:Good#3");

    // Potions and screen picks.
    var potion = new List<MarkOption>
    {
        new("Fire Potion", "potion", null, 0, null, 2, null, true, false, 0.9, null, null),
    };
    Check("marks: potion slot", Show(AdviceMarks.Build(potion, null)), "Potion2:Best#1>0");
    var rewardMarks = new List<MarkOption>
    {
        new("Inflame", "choose", null, null, null, null, 1, true, false, 0.7, null, null),
        new("Skip", "skip", null, null, null, null, -1, false, false, 0.2, null, null),
        new("Clash", "choose", null, null, null, null, 0, false, false, 0.1, null, null),
    };
    Check("marks: card reward pick + skip alt", Show(AdviceMarks.Build(rewardMarks, null)), "ScreenOption1:Best#1 Skip0:Alt");

    // Search: a Q within the band is as good as the pick.
    var qs = new List<MarkOption>
    {
        Play(0, null, 0.5, best: true, q: 0.80), Play(1, null, 0.05, q: 0.797), Play(2, null, 0.05, q: 0.70),
    };
    Check("marks: q-tie is Good", Show(AdviceMarks.Build(qs, null)), "Card0:Best#1 Card1:Good");
    Check("verdict: best", AdviceMarks.Verdict(qs[0], 0.80), "Best");
    Check("verdict: tie", AdviceMarks.Verdict(qs[1], 0.80), "Also good");
    Check("verdict: big loss", AdviceMarks.Verdict(qs[2], 0.80), "Mistake");
    Check("verdict: instinct share", AdviceMarks.Verdict(Play(1, null, 0.1), null), "Okay");
    Check("verdict: instinct unlikely", AdviceMarks.Verdict(Play(1, null, 0.01), null), "Unlikely");
    Check("row verdict: a later plan step reads Then", AdviceMarks.RowVerdict(Play(1, null, 0.08), null, 2), "Then");
    Check("row verdict: the pick stays Best", AdviceMarks.RowVerdict(Play(0, null, 0.9, best: true), null, 1), "Best");

    // No raw ids on screen.
    Check("humanize: shop id", AdviceMarks.Humanize("SHRUG_IT_OFF"), "Shrug It Off");
    Check("humanize: inside text", AdviceMarks.Humanize("discard FIRE_POTION"), "discard Fire Potion");
    Check("humanize: dotted", AdviceMarks.Humanize("buy CARD.PERFECTED_STRIKE"), "buy Perfected Strike");
    Check("humanize: small words", AdviceMarks.Humanize("HAND_OF_GREED"), "Hand of Greed");
    Check("humanize: lookup wins", AdviceMarks.Humanize("STRIKE_IRONCLAD", id => id == "STRIKE_IRONCLAD" ? "Strike" : null), "Strike");
    Check("humanize: plain names untouched", AdviceMarks.Humanize("Strike → Sludge Spinner #2"), "Strike → Sludge Spinner #2");
    Check("humanize: lone caps untouched", AdviceMarks.Humanize("AI would: Bash"), "AI would: Bash");
}

// FrameLaunchRules: the panels' own window (SimFrameLauncher).
{
    Check("frame: opens at startup when wanted", FrameLaunchRules.OpenAtStartup(true, false, true, false).ToString(), "True");
    Check("frame: not where it cannot open (no note every start)", FrameLaunchRules.OpenAtStartup(true, false, false, false).ToString(), "False");
    Check("frame: supported on Windows", FrameLaunchRules.Supported(true, false, "Windows").ToString(), "True");
    Check("frame: supported on Linux X11 / XWayland", FrameLaunchRules.Supported(false, true, "X11").ToString(), "True");
    Check("frame: not on native Wayland", FrameLaunchRules.Supported(false, true, "Wayland").ToString(), "False");
    Check("frame: not on macOS", FrameLaunchRules.Supported(false, false, "macOS").ToString(), "False");
    Check("frame: the Linux host's script",
        FrameLaunchRules.LinuxScriptPath(System.IO.Path.Combine("m", "SpireMentor")),
        System.IO.Path.Combine("m", "SpireMentor", "SpireMentorFrame", "spire_mentor_frame.py"));
    {
        var (file, fargs) = FrameLaunchRules.LinuxCommand(true, "/m/f.py", new[] { "--attach", "7" });
        Check("frame: under Steam's runtime, beside Steam on the host",
            file + "|" + string.Join("|", fargs), "steam-runtime-launch-client|--alongside-steam|--|python3|/m/f.py|--attach|7");
        (file, fargs) = FrameLaunchRules.LinuxCommand(false, "/m/f.py", new[] { "--attach", "7" });
        Check("frame: outside the runtime, python3 directly", file + "|" + string.Join("|", fargs), "python3|/m/f.py|--attach|7");
    }
    Check("frame: in the Steam runtime", FrameLaunchRules.InSteamRuntime("pressure-vessel", "com.steampowered.X").ToString(), "True");
    Check("frame: a container without the launcher service", FrameLaunchRules.InSteamRuntime("pressure-vessel", "").ToString(), "False");
    Check("frame: no container", FrameLaunchRules.InSteamRuntime(null, null).ToString(), "False");
    Check("frame: not in a test lane", FrameLaunchRules.OpenAtStartup(true, false, true, true).ToString(), "False");
    Check("frame: not when the frame launched the game", FrameLaunchRules.OpenAtStartup(true, true, true, false).ToString(), "False");
    Check("frame: not when unticked", FrameLaunchRules.OpenAtStartup(false, false, true, false).ToString(), "False");
    Check("frame: bundled page on the control plane", FrameLaunchRules.PageUrl("", 8079), "http://127.0.0.1:8079/frame.html");
    Check("frame: a second instance's control port", FrameLaunchRules.PageUrl(null, 8089), "http://127.0.0.1:8089/frame.html");
    Check("frame: configured page wins (trimmed)", FrameLaunchRules.PageUrl(" http://127.0.0.1:5199/frame.html ", 8079), "http://127.0.0.1:5199/frame.html");
    Check("frame: exe under the mod folder",
        FrameLaunchRules.ExePath(System.IO.Path.Combine("m", "SpireMentor")),
        System.IO.Path.Combine("m", "SpireMentor", "SpireMentorFrame", "SpireMentorFrame.exe"));
    Check("frame: attach args",
        string.Join("|", FrameLaunchRules.Args(4242, 131072, 50123, "http://127.0.0.1:8079/frame.html", @"C:\Games\Slay the Spire 2")),
        @"--attach|4242|--port|50123|--url|http://127.0.0.1:8079/frame.html|--attach-hwnd|131072|--game-dir|C:\Games\Slay the Spire 2");
    Check("frame: no handle, no game dir",
        string.Join("|", FrameLaunchRules.Args(1, 0, 2, "u", null)), "--attach|1|--port|2|--url|u");
    string Btn(bool framed, bool outside, bool windows = true, bool shipped = true, bool harness = false) =>
        FrameLaunchRules.PanelsButtonLabel(framed, outside, windows, shipped, harness) ?? "none";
    Check("frame button: panels around the game, none", Btn(true, true), "none");
    Check("frame button: framed fullscreen exits it", Btn(true, false), FrameLaunchRules.ExitFullscreenLabel);
    Check("frame button: frame closed, back beside the game", Btn(false, true), FrameLaunchRules.PanelsBesideLabel);
    Check("frame button: frame closed, off Windows", Btn(false, true, windows: false), "none");
    Check("frame button: frame closed, not shipped", Btn(false, true, shipped: false), "none");
    Check("frame button: frame closed, test lane", Btn(false, true, harness: true), "none");
}

// AI partner votes to move on (#47): the last act has two rounds in the same
// act (boss -> The Architect, The Architect -> win), told apart by ActFloor.
{
    Check("partner vote: the Architect round has its own key",
        (PartnerVotes.ActChange(2, 17) != PartnerVotes.ActChange(2, 18)).ToString(), "True");
    Check("partner vote: one round is one key",
        (PartnerVotes.ActChange(2, 17) == PartnerVotes.ActChange(2, 17)).ToString(), "True");
    Check("partner vote: acts differ", (PartnerVotes.ActChange(0, 17) != PartnerVotes.ActChange(1, 17)).ToString(), "True");
}

// Card grids once a card is selected (#1206): an "up to N" combat pile pick
// keeps offering the rest; every other grid only asks to confirm.
{
    const string pile = CardGridRules.CombatPileScreen;
    Check("grid: nothing selected offers the cards", CardGridRules.OffersMorePicks("NDeckEnchantSelectScreen", 0, 1, 5).ToString(), "True");
    Check("grid: unreadable selection offers the cards", CardGridRules.OffersMorePicks(pile, -1, 2, 5).ToString(), "True");
    Check("grid: combat pile 1 of 2 offers the rest", CardGridRules.OffersMorePicks(pile, 1, 2, 3).ToString(), "True");
    Check("grid: combat pile 2 of 2 confirms", CardGridRules.OffersMorePicks(pile, 2, 2, 3).ToString(), "False");
    Check("grid: combat pile with nothing left confirms", CardGridRules.OffersMorePicks(pile, 1, 2, 0).ToString(), "False");
    Check("grid: enchant 1 of 2 confirms", CardGridRules.OffersMorePicks("NDeckEnchantSelectScreen", 1, 2, 3).ToString(), "False");
    Check("grid: deck select 1 of 2 confirms", CardGridRules.OffersMorePicks("NDeckCardSelectScreen", 1, 2, 3).ToString(), "False");
}

// card_select replies that select nothing (#1326): a skip only on a cancelable
// screen. Toolbox / Knowledge Demon come in as (min 0, max 1) with the game's
// canSkip false, so min_select 0 alone must not let them through. Both
// spellings of an empty choice go through this one rule in SimCardSelector:
// single select {"index": <count or more>}, multi select {"indices": []}.
{
    string E(int minSelect, bool cancelable) => CardSelectRules.ForEmptyChoice(minSelect, cancelable).ToString();
    Check("card_select: index past the end, cancelable -> skip", E(0, true), "Skip");
    Check("card_select: index past the end, mandatory (Toolbox) -> not_cancelable", E(0, false), "NotCancelable");
    Check("card_select: empty indices, cancelable -> skip", E(0, true), "Skip");
    Check("card_select: empty indices, mandatory -> not_cancelable", E(0, false), "NotCancelable");
    Check("card_select: empty choice with min_select 1 -> invalid", E(1, false), "Invalid");
    Check("card_select: empty choice with min_select 2, cancelable -> invalid", E(2, true), "Invalid");
}

// In-hand selects that could be confirmed (#1770): an "any number" or "up to
// N" select keeps offering the hand, so the AI takes its whole set (Gambler's
// Brew used to stop at one Defend); at the maximum or an empty hand, confirm.
{
    const int any = 999999999;
    Check("hand: Gambler's Brew 1 taken offers the rest", CardSelectRules.HandOffersMorePicks(1, any, 5).ToString(), "True");
    Check("hand: up to 3, 2 taken offers the rest", CardSelectRules.HandOffersMorePicks(2, 3, 4).ToString(), "True");
    Check("hand: up to 3, 3 taken confirms", CardSelectRules.HandOffersMorePicks(3, 3, 4).ToString(), "False");
    Check("hand: exactly 1 taken confirms", CardSelectRules.HandOffersMorePicks(1, 1, 4).ToString(), "False");
    Check("hand: nothing left confirms", CardSelectRules.HandOffersMorePicks(6, any, 0).ToString(), "False");
    Check("hand: unreadable bounds confirm", CardSelectRules.HandOffersMorePicks(1, 0, 4).ToString(), "False");
}

// LinuxSignalRules: finding the runtime's saved signal actions (GH #1425).
{
    var maps = LinuxSignalRules.ParseMaps(new[]
    {
        "00400000-00500000 r-xp 00000000 00:1f 100 /games/Slay the Spire 2/SlayTheSpire2",
        "10000000-10100000 rw-p 00000000 00:00 0                                  [heap]",
        "20000000-20010000 rw-p 00000000 00:00 0 ",
        "30000000-50000000 rw-p 00000000 00:00 0 ",
        "7f0000000000-7f0000100000 r-xp 00001000 00:1f 200 /games/Slay the Spire 2/data_sts2_linuxbsd_x86_64/libcoreclr.so",
        "7f0000100000-7f0000110000 rw-p 00101000 00:1f 200 /games/Slay the Spire 2/data_sts2_linuxbsd_x86_64/libcoreclr.so",
        "7f1000000000-7f1000200000 r-xp 00028000 00:1f 300 /run/host/usr/lib/libc.so.6",
        "7ffc00000000-7ffc00021000 rw-p 00000000 00:00 0                          [stack]",
        "garbage",
    });
    Check("signals: runtime code found (path with spaces)", maps.Runtime.Count.ToString(), "1");
    Check("signals: libc code found", maps.Libc.Count.ToString(), "1");
    Check("signals: heap found", maps.Heap.Count == 1 ? maps.Heap[0].Lo.ToString("x") : "none", "10000000");
    Check("signals: anonymous rw only, huge one left out, no stack", maps.Anonymous.Count == 1 ? maps.Anonymous[0].Lo.ToString("x") : maps.Anonymous.Count.ToString(), "20000000");

    const ulong clr = 0x7f0000000000, libc = 0x7f1000000000;
    void Put(byte[] block, int slot, ulong handler, uint flags, ulong restorer)
    {
        int o = 16 + slot * LinuxSignalRules.ActionSize;
        BitConverter.GetBytes(handler).CopyTo(block, o);
        BitConverter.GetBytes(flags).CopyTo(block, o + LinuxSignalRules.FlagsOffset);
        BitConverter.GetBytes(restorer).CopyTo(block, o + LinuxSignalRules.RestorerOffset);
    }
    // The reporter's array, indexed by signal number: 3..8 and 11 are the runtime's.
    byte[] Array(ulong segvHandler = clr + 0x700)
    {
        var block = new byte[16 + 12 * LinuxSignalRules.ActionSize + 64];
        for (int sig = 3; sig <= 8; sig++) Put(block, sig, clr + 0x100 * (ulong)sig, 0x14000004, libc + 0x40);
        Put(block, 11, segvHandler, 0x1c000004, libc + 0x40);
        Put(block, 9, 0x400100, 0x14000004, libc + 0x40);          // handler outside the runtime
        Put(block, 10, clr + 0x900, 0x14000004, 0x400200);          // restorer outside libc
        return block;
    }
    var found = new List<LinuxSignalRules.Candidate>();
    LinuxSignalRules.FindCandidates(Array(), 0x10000000, maps, found);
    Check("signals: seven actions recognised, two impostors not", found.Count.ToString(), "7");
    var picked = LinuxSignalRules.Pick(found);
    Check("signals: SIGSEGV is the SA_ONSTACK one", picked == null ? "null" : BitConverter.ToUInt64(picked.Segv, 0).ToString("x"), (clr + 0x700).ToString("x"));
    Check("signals: SIGFPE is three slots before it", picked?.Fpe == null ? "null" : BitConverter.ToUInt64(picked.Fpe, 0).ToString("x"), (clr + 0x800).ToString("x"));

    var two = new List<LinuxSignalRules.Candidate>(found);
    LinuxSignalRules.FindCandidates(Array(), 0x10100000, maps, two);
    Check("signals: two agreeing copies still pick", LinuxSignalRules.Pick(two) != null ? "picked" : "null", "picked");
    var clash = new List<LinuxSignalRules.Candidate>(found);
    LinuxSignalRules.FindCandidates(Array(clr + 0x770), 0x10100000, maps, clash);
    Check("signals: copies that disagree on SIGSEGV pick nothing", LinuxSignalRules.Pick(clash) == null ? "null" : "picked", "null");

    var lone = new List<LinuxSignalRules.Candidate>();
    var only = new byte[16 + 12 * LinuxSignalRules.ActionSize + 64];
    Put(only, 11, clr + 0x700, 0x1c000004, libc + 0x40);
    Put(only, 8, clr + 0x800, 0x14000004, libc + 0x40);            // right slot, but 3..7 missing
    LinuxSignalRules.FindCandidates(only, 0x10000000, maps, lone);
    var lonePick = LinuxSignalRules.Pick(lone);
    Check("signals: unknown layout keeps SIGSEGV, drops SIGFPE", lonePick == null ? "null" : (lonePick.Fpe == null ? "segv only" : "both"), "segv only");
    Check("signals: nothing found picks nothing", LinuxSignalRules.Pick(new List<LinuxSignalRules.Candidate>()) == null ? "null" : "picked", "null");
    var none = new List<LinuxSignalRules.Candidate>();
    LinuxSignalRules.FindCandidates(new byte[100], 0x10000000, maps, none);
    Check("signals: a block shorter than one action is fine", none.Count.ToString(), "0");
}

L10nTests.Run(Check);
CoachL10nTests.Run(Check);
AutopilotAttachTests.Run(Check);
AgentHeartbeatTests.Run(Check);

if (OperatingSystem.IsWindows())
{
    WindowsChildProcessTests.Run(Check);
}
else
{
    Console.WriteLine("SKIP Windows advisor handle-inheritance tests (requires Windows).");
}

Console.WriteLine(failures == 0
    ? $"ModUnitTests: OK ({checks} checks)"
    : $"ModUnitTests: {failures} of {checks} checks FAILED");
return failures == 0 ? 0 : 1;
