import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.build_sfx_map import (
    _REPO_DECOMP_ROOT,
    _event_audio_path,
    audio_fingerprint,
    build_map,
    build_monster_entry,
    card_player_anim,
    card_signature_event,
    load_character_anim_fallbacks,
    load_character_sfx,
    load_intent_tables,
    load_orb_sfx,
    load_treasure_sfx,
    parse_creature_sfx,
    parse_hit_sfx,
    parse_impact_sfx,
    parse_move_sfx,
    parse_move_states,
    tmp_sfx_path,
)

_AUDIO_ROOT = _REPO_DECOMP_ROOT.parents[1] / "site" / "public" / "audio"
# Tests that read the game's decompilation or rendered audio skip without them
# (`just import-game` produces both; the public repository ships neither).
needs_decomp = unittest.skipUnless(
    _REPO_DECOMP_ROOT.is_dir(), f"decomp dir not present ({_REPO_DECOMP_ROOT})"
)
needs_audio = unittest.skipUnless(
    _AUDIO_ROOT.is_dir(), f"rendered audio not present ({_AUDIO_ROOT})"
)

SOURCE = """
        MoveState moveState = new MoveState("MAGISTRATE_SCRUTINY", MagistrateScrutinyMove, new SingleAttackIntent(ScrutinyDamage));
        MoveState moveState2 = new MoveState("PECK_ASSAULT", PeckAssaultMove, new MultiAttackIntent(PeckAssaultDamage, 6));
        MoveState moveState3 = new MoveState("JUDICIAL_FLIGHT", JudicialFlightMove, new BuffIntent());

    private async Task MagistrateScrutinyMove(IReadOnlyList<Creature> targets)
    {
        await DamageCmd.Attack(ScrutinyDamage).FromMonster(this).WithAttackerFx(null, "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck")
            .Execute(null);
    }

    private async Task JudicialFlightMove(IReadOnlyList<Creature> targets)
    {
        SfxCmd.Play("event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_take_off");
        SfxCmd.PlayLoop(base.Creature, "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_fly_loop");
    }
"""  # noqa: E501 — the C# fixture text is verbatim from the task brief; keep it byte-for-byte


class TestParseMoveStates(unittest.TestCase):
    @needs_decomp
    def test_reads_name_method_and_intent(self):
        moves = parse_move_states(SOURCE)
        self.assertEqual(moves[0]["name"], "MAGISTRATE_SCRUTINY")
        self.assertEqual(moves[0]["method"], "MagistrateScrutinyMove")
        self.assertEqual(moves[0]["intent"], "Attack")
        self.assertEqual(moves[0]["hits"], 1)

    @needs_decomp
    def test_multi_attack_records_its_hit_count(self):
        moves = parse_move_states(SOURCE)
        self.assertEqual(moves[1]["intent"], "Attack")
        self.assertEqual(moves[1]["hits"], 6)

    def test_non_attack_intent_is_labelled(self):
        self.assertEqual(parse_move_states(SOURCE)[2]["intent"], "Buff")


class TestParseMoveSfx(unittest.TestCase):
    def test_finds_the_attacker_fx_event(self):
        sfx = parse_move_sfx(SOURCE)
        self.assertEqual(
            sfx["MagistrateScrutinyMove"],
            "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck",
        )

    def test_prefers_a_one_shot_over_a_loop(self):
        # PlayLoop is ambient chatter and must never become a move's cue.
        self.assertEqual(
            parse_move_sfx(SOURCE)["JudicialFlightMove"],
            "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_take_off",
        )


class TestMonsterDeathAndHurtSfx(unittest.TestCase):
    LITERAL = """
\tpublic override string DeathSfx => "event:/sfx/enemy/enemy_attacks/obscura/obscura_die";
\tpublic override string HurtSfx => "event:/sfx/enemy/enemy_attacks/obscura/obscura_hurt";
"""

    CONDITIONAL = """
\tpublic override string DeathSfx
\t{
\t\tget
\t\t{
\t\t\tif (!_hatched)
\t\t\t{
\t\t\t\treturn "event:/sfx/enemy/enemy_attacks/tough_egg/tough_egg_die";
\t\t\t}
\t\t\treturn "event:/sfx/enemy/enemy_attacks/tough_egg/hatchling_die";
\t\t}
\t}
"""

    def test_expression_bodied_overrides_are_read(self):
        self.assertEqual(
            parse_creature_sfx(self.LITERAL),
            {
                "death": "event:/sfx/enemy/enemy_attacks/obscura/obscura_die",
                "hurt": "event:/sfx/enemy/enemy_attacks/obscura/obscura_hurt",
            },
        )

    def test_a_conditional_getter_is_omitted_not_guessed(self):
        # Two possible values, no static way to pick — so neither ships.
        self.assertEqual(parse_creature_sfx(self.CONDITIONAL), {})

    def test_a_monster_with_neither_override_carries_neither_key(self):
        self.assertEqual(parse_creature_sfx("public class Quiet : Monster { }"), {})


class TestMonsterImpactSfx(unittest.TestCase):
    """`SfxCmd.PlayDamage` -> `MonsterModel.TakeDamageSfx` — the sound of an
    attack LANDING, which is not `block_hit` (that one fires only when the
    damage was fully blocked)."""

    IMPACT = "event:/sfx/enemy/enemy_impact_enemy_size/enemy_impact_"

    @staticmethod
    def _declares(kind: str) -> str:
        return f"\tpublic override DamageSfxType TakeDamageSfxType => DamageSfxType.{kind};"

    def test_material_becomes_the_impact_event(self):
        self.assertEqual(parse_impact_sfx(self._declares("Slime")), self.IMPACT + "slime")

    def test_two_word_material_is_snake_cased_like_the_games_slugify(self):
        self.assertEqual(parse_impact_sfx(self._declares("ArmorBig")), self.IMPACT + "armor_big")

    def test_a_monster_that_overrides_nothing_still_has_an_impact(self):
        # MonsterModel's virtual default is Armor, so silence would be wrong.
        self.assertEqual(
            parse_impact_sfx("public class Quiet : MonsterModel { }"), self.IMPACT + "armor"
        )

    def test_a_whole_getter_override_replaces_the_material(self):
        own = "event:/sfx/enemy/enemy_attacks/chomper/chomper_hurt"
        src = self._declares("Fur") + f'\n\tpublic override string TakeDamageSfx => "{own}";\n'
        self.assertEqual(parse_impact_sfx(src), own)

    def test_damage_sfx_type_none_means_silent_not_a_missing_file(self):
        # There is no enemy_impact_none.webm; the enum member means "no sound".
        self.assertIsNone(parse_impact_sfx(self._declares("None")))

    def test_entry_carries_the_impact(self):
        entry = build_monster_entry(self._declares("Stone"))
        self.assertEqual(entry["impact"], self.IMPACT + "stone")


class TestGenericContract(unittest.TestCase):
    # The web looks up its own key names, not the decomp's — FmodSfx.cs calls
    # block-gain `block`. If this drifts, those cues go silently mute.
    REQUIRED = (
        "blockGain",
        "blockBreak",
        "blockHit",
        "buff",
        "debuff",
        "heal",
        "cardPlay",
        "cardPower",
        "uiClick",
        "mapOpen",
        "mapSelect",
        "gold",
        "energy",
        "relic",
        "defeat",
    )

    def test_every_key_the_web_reads_is_present(self):
        from scripts.build_sfx_map import GENERIC_EVENTS

        for key in self.REQUIRED:
            self.assertIn(key, GENERIC_EVENTS)

    def test_block_gain_resolves_to_the_games_event(self):
        from scripts.build_sfx_map import GENERIC_EVENTS

        self.assertEqual(GENERIC_EVENTS["blockGain"], "event:/sfx/block_gain")

    def test_every_value_names_one_of_the_games_two_sound_schemes(self):
        """FMOD event or `res://debug_audio/` stream — and nothing else.

        The second scheme is not decorative: `NDebugAudioManager` is how the
        shipped game voices combat victory, obtaining a relic and every card
        hit sound, none of which has an FMOD event at all. This used to
        require `event:/` alone, which is why those moments were silent.
        """
        from scripts.build_sfx_map import DEBUG_AUDIO_SCHEME, GENERIC_EVENTS

        for key, event in GENERIC_EVENTS.items():
            self.assertTrue(
                event.startswith("event:/") or event.startswith(DEBUG_AUDIO_SCHEME),
                f"{key} -> {event}",
            )


def _write_intents_fixture(decomp_root: Path) -> None:
    """A minimal MegaCrit.Sts2.Core.MonsterMoves.Intents/ fixture: just
    enough of the real class hierarchy (AttackIntent's two attack-family
    subclasses seen in SOURCE, plus BuffIntent) for load_intent_tables to
    resolve build_map's own fixtures the same way it resolves the real
    decomp -- build_map always derives these tables itself rather than
    falling back to the lazily-cached real-decomp default, so any fixture
    that exercises build_map (not just parse_move_states) needs this.
    """
    intents = decomp_root / "MegaCrit.Sts2.Core.MonsterMoves.Intents"
    intents.mkdir(parents=True, exist_ok=True)
    (intents / "AttackIntent.cs").write_text(
        "public abstract class AttackIntent : AbstractIntent\n"
        "{\n"
        "    public override IntentType IntentType => IntentType.Attack;\n"
        "}\n"
    )
    (intents / "SingleAttackIntent.cs").write_text(
        "public class SingleAttackIntent : AttackIntent\n"
        "{\n"
        "    public override int Repeats => 1;\n"
        "}\n"
    )
    (intents / "MultiAttackIntent.cs").write_text(
        "public class MultiAttackIntent : AttackIntent\n{\n}\n"
    )
    (intents / "BuffIntent.cs").write_text(
        "public class BuffIntent : AbstractIntent\n"
        "{\n"
        "    public override IntentType IntentType => IntentType.Buff;\n"
        "}\n"
    )


def _write_titles_fixture(root: Path, filename: str, entries: list[dict]) -> None:
    """Write `data/vtest/<filename>` (a list of `{class_name, title}`
    records), the layout `_class_titles` expects sitting alongside a
    `decompiled/vtest` tree at the same `root`. Every `build_map` fixture
    that wants its monsters/cards/relics to survive the display-title join
    needs one of these — a class with no matching entry here is dropped,
    same as in the real generator.
    """
    data = root / "data" / "vtest"
    data.mkdir(parents=True, exist_ok=True)
    (data / filename).write_text(json.dumps(entries))


class TestBuildMapDropsUnrenderedEvents(unittest.TestCase):
    # The decomp is not internally consistent: some SfxCmd/WithAttackerFx
    # calls name an event nothing was ever rendered for (a typo baked into
    # the C# itself, or a const bound to the wrong literal — both happen for
    # real, see the task-4 report). Shipping that string 404s in the web;
    # the map must drop it instead, the same graceful degradation as a move
    # with no sfx call at all. This is what makes "0 mapped events with no
    # audio file" (asserted by Task 11) true against the REAL audio tree,
    # not just against whatever the regexes happened to find.
    # SOURCE itself already covers both cases: MAGISTRATE_SCRUTINY's peck
    # event has a rendered file, JUDICIAL_FLIGHT's take_off event does not
    # (see _write_fixture) — no extra fixture content needed.
    MONSTER_SOURCE = SOURCE

    # Display title differs from the class name — "Owl Magistrate", not
    # "OwlMagistrate" — deliberately, so these tests exercise the same
    # class-name -> display-title join build_map applies for real (finding 1
    # of the 2026-08-23 final review: every fixture in this file used to key
    # by class name, which let the join bug ship silently).
    TITLE = "Owl Magistrate"

    def _write_fixture(self, root: Path) -> None:
        from scripts.build_sfx_map import GENERIC_EVENTS

        decomp = root / "decompiled" / "vtest"
        monsters = decomp / "MegaCrit.Sts2.Core.Models.Monsters"
        monsters.mkdir(parents=True)
        (monsters / "OwlMagistrate.cs").write_text(self.MONSTER_SOURCE)
        _write_intents_fixture(decomp)
        _write_titles_fixture(
            root, "monsters.json", [{"class_name": "OwlMagistrate", "title": self.TITLE}]
        )

        audio = root / "audio" / "sfx" / "enemy" / "enemy_attacks" / "owl_magistrate"
        audio.mkdir(parents=True)
        # Only the peck event was actually rendered; take_off and the typo'd
        # verdict event were not.
        (audio / "owl_magistrate_attack_peck.webm").write_bytes(b"\x00")

        # Stub every GENERIC_EVENTS file too, so the (unrelated) generic-
        # completeness guard in build_map doesn't fire in this fixture.
        for event in GENERIC_EVENTS.values():
            path = _event_audio_path(event, root / "audio")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00")

    def test_event_with_no_rendered_file_is_dropped_not_shipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            doc = build_map(root / "decompiled" / "vtest", "vtest", audio_root=root / "audio")

            moves = {m["name"]: m for m in doc["monsters"][self.TITLE]["moves"]}
            # JUDICIAL_FLIGHT's take_off event has no rendered file — the move
            # must be ABSENT from the map, not present with "event": null. A
            # null reaching the web is a crash/broken fetch, not a graceful
            # degradation (see task-4 report, fix round 1).
            self.assertNotIn("JUDICIAL_FLIGHT", moves)
            self.assertEqual(
                moves["MAGISTRATE_SCRUTINY"]["event"],
                "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck",
            )

    def test_move_with_no_sfx_call_at_all_is_dropped_regardless_of_audio_root(self):
        # PECK_ASSAULT has a MoveState but no matching method body anywhere in
        # SOURCE, so parse_move_sfx never finds an event for it at all — this
        # must be dropped whether or not audio_root is even supplied.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            decomp = root / "decompiled" / "vtest"
            with_root = build_map(decomp, "vtest", audio_root=root / "audio")
            without_root = build_map(decomp, "vtest")

            for doc in (with_root, without_root):
                names = {m["name"] for m in doc["monsters"][self.TITLE]["moves"]}
                self.assertNotIn("PECK_ASSAULT", names)

    def test_default_attack_is_recomputed_after_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            doc = build_map(root / "decompiled" / "vtest", "vtest", audio_root=root / "audio")

            # The only surviving Attack-intent move with a real file is the
            # peck; the typo'd VERDICT move must not become defaultAttack.
            self.assertEqual(
                doc["monsters"][self.TITLE]["defaultAttack"],
                "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck",
            )

    def test_without_an_audio_root_events_pass_through_unverified(self):
        # build_monster_entry itself (and build_map with no audio_root) never
        # touches disk — used by the unit tests above SOURCE.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            doc = build_map(root / "decompiled" / "vtest", "vtest")  # no audio_root

            moves = {m["name"]: m for m in doc["monsters"][self.TITLE]["moves"]}
            self.assertIsNotNone(moves["JUDICIAL_FLIGHT"]["event"])


class TestBuildMapKeepsDeathOnlyMonster(unittest.TestCase):
    """Fix round 1 (task-8b): a monster whose own moves never resolve to a
    shipped sfx call must still appear in the map if its `DeathSfx`/`HurtSfx`
    DOES resolve to shipped audio -- the real FakeMerchantMonster case (a
    killable "reverse merchant" ambush). Before this fix, `build_map`'s
    `if moves:` gate silently dropped the whole entry, taking a perfectly
    valid, shipped death/hurt sound down with the genuinely-empty moves list.
    """

    # A single move whose sfx event has no rendered file -- moves ends up
    # empty -- plus a literal DeathSfx/HurtSfx that DOES resolve.
    MONSTER_SOURCE = """
        MoveState moveState = new MoveState("AMBUSH", AmbushMove, new SingleAttackIntent(Dmg));

    private async Task AmbushMove(IReadOnlyList<Creature> targets)
    {
        await DamageCmd.Attack(Dmg).FromMonster(this)
            .WithAttackerFx(null, "event:/sfx/npcs/rev_merchant/rev_merchant_unrendered_attack")
            .Execute(null);
    }

    public override string DeathSfx => "event:/sfx/npcs/rev_merchant/rev_merchant_die";
    public override string HurtSfx => "event:/sfx/npcs/rev_merchant/rev_merchant_hurt";
"""

    # Display title differs from the class name (the real FakeMerchantMonster
    # title, v0.107.1) — see the note on TestBuildMapDropsUnrenderedEvents.
    TITLE = "The Merchant???"

    def _write_fixture(self, root: Path) -> None:
        from scripts.build_sfx_map import GENERIC_EVENTS

        decomp = root / "decompiled" / "vtest"
        monsters = decomp / "MegaCrit.Sts2.Core.Models.Monsters"
        monsters.mkdir(parents=True)
        (monsters / "FakeMerchantMonster.cs").write_text(self.MONSTER_SOURCE)
        _write_intents_fixture(decomp)
        _write_titles_fixture(
            root, "monsters.json", [{"class_name": "FakeMerchantMonster", "title": self.TITLE}]
        )

        audio = root / "audio" / "sfx" / "npcs" / "rev_merchant"
        audio.mkdir(parents=True)
        # The attack move's event was never rendered; death and hurt were.
        (audio / "rev_merchant_die.webm").write_bytes(b"\x00")
        (audio / "rev_merchant_hurt.webm").write_bytes(b"\x00")

        for event in GENERIC_EVENTS.values():
            path = _event_audio_path(event, root / "audio")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00")

    def test_monster_with_no_resolvable_moves_but_a_shipped_death_sound_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            doc = build_map(root / "decompiled" / "vtest", "vtest", audio_root=root / "audio")

            self.assertIn(self.TITLE, doc["monsters"])
            entry = doc["monsters"][self.TITLE]
            self.assertEqual(entry["moves"], [])
            self.assertEqual(entry["death"], "event:/sfx/npcs/rev_merchant/rev_merchant_die")
            self.assertEqual(entry["hurt"], "event:/sfx/npcs/rev_merchant/rev_merchant_hurt")
            self.assertNotIn("defaultAttack", entry)

    def test_a_monster_with_nothing_resolvable_at_all_is_still_dropped(self):
        # The gate widening must not become "keep everything" -- a monster
        # with no moves AND no death/hurt/defaultAttack still contributes
        # nothing useful and must still be excluded. Given a resolvable title
        # ("Truly Empty") so this pins the moves/death/hurt gate specifically,
        # not the separate (and separately tested) title-join gate.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from scripts.build_sfx_map import GENERIC_EVENTS

            decomp = root / "decompiled" / "vtest"
            monsters = decomp / "MegaCrit.Sts2.Core.Models.Monsters"
            monsters.mkdir(parents=True)
            (monsters / "TrulyEmpty.cs").write_text("public class TrulyEmpty : Monster { }")
            _write_intents_fixture(decomp)
            _write_titles_fixture(
                root, "monsters.json", [{"class_name": "TrulyEmpty", "title": "Truly Empty"}]
            )
            audio = root / "audio"
            for event in GENERIC_EVENTS.values():
                path = _event_audio_path(event, audio)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x00")

            doc = build_map(decomp, "vtest", audio_root=audio)

            self.assertNotIn("Truly Empty", doc["monsters"])
            self.assertNotIn("TrulyEmpty", doc["monsters"])


class TestBuildMonsterEntry(unittest.TestCase):
    @needs_decomp
    def test_joins_moves_to_their_sfx_and_picks_a_default_attack(self):
        entry = build_monster_entry(SOURCE)
        self.assertEqual(
            entry["defaultAttack"],
            "event:/sfx/enemy/enemy_attacks/owl_magistrate/owl_magistrate_attack_peck",
        )
        self.assertEqual(len(entry["moves"]), 3)
        self.assertEqual(entry["moves"][0]["hits"], 1)


class TestRelicsAreNotUiFiltered(unittest.TestCase):
    # Fix round 1: _first_event's "/ui/" exclusion is right for cards (a
    # generic UI click must not become mistaken for a card's own sound) but
    # was wrong for relics — every relic's activation cue legitimately lives
    # under event:/sfx/ui/relic_activate_draw, so the same exclusion made
    # `relics` come out empty. Pin that a relic mapping to that event
    # survives, so this can't silently regress to {} again.
    RELIC_SOURCE = """
public sealed class GamePiece : AbstractRelicModel
{
    private async Task OnActivate()
    {
        SfxCmd.Play("event:/sfx/ui/relic_activate_draw");
    }
}
"""

    # Display title differs from the class name, same rationale as the
    # monster fixtures above.
    TITLE = "Game Piece"

    def test_relic_ui_event_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomp = root / "decompiled" / "vtest"
            relics = decomp / "MegaCrit.Sts2.Core.Models.Relics"
            relics.mkdir(parents=True)
            (relics / "GamePiece.cs").write_text(self.RELIC_SOURCE)
            _write_titles_fixture(
                root, "relics.json", [{"class_name": "GamePiece", "title": self.TITLE}]
            )

            doc = build_map(decomp, "vtest")  # no audio_root: pure parse

            self.assertEqual(doc["relics"][self.TITLE], "event:/sfx/ui/relic_activate_draw")
            self.assertNotIn("GamePiece", doc["relics"])


class TestGeneratedArtifactHasNoNulls(unittest.TestCase):
    # Property of the ARTIFACT, checked against the real committed file, not
    # against build_map's intermediates: no null event may reach the JSON the
    # web actually reads, because a null there is a crash/broken fetch, not a
    # degrade-to-generic-sound case.
    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )

    def _load(self):
        import json

        return json.loads(self.ARTIFACT.read_text())

    def test_no_null_event_anywhere_in_the_generated_map(self):
        doc = self._load()
        for key, event in doc["generic"].items():
            self.assertIsNotNone(event, f"generic.{key} is null")
        for key, event in doc["cards"].items():
            self.assertIsNotNone(event, f"cards.{key} is null")
        for key, event in doc["relics"].items():
            self.assertIsNotNone(event, f"relics.{key} is null")
        for name, entry in doc["monsters"].items():
            self.assertIsNotNone(
                entry.get("defaultAttack", "not-null-sentinel"),
                f"monsters.{name}.defaultAttack is null",
            )
            for move in entry["moves"]:
                self.assertIsNotNone(
                    move["event"], f"monsters.{name} move {move['name']} has a null event"
                )

    def test_no_null_hits_or_damage_when_present(self):
        # "hits"/"damage" are OMITTED when unresolvable (fix round 2's "absent
        # beats wrong" rule) -- they must never be present-but-null.
        doc = self._load()
        for name, entry in doc["monsters"].items():
            for move in entry["moves"]:
                if "hits" in move:
                    self.assertIsNotNone(move["hits"], f"{name} {move['name']} hits is null")
                if "damage" in move:
                    self.assertIsNotNone(move["damage"], f"{name} {move['name']} damage is null")

    def test_powers_table_values_are_only_buff_or_debuff(self):
        doc = self._load()
        for name, valence in doc["powers"].items():
            self.assertIn(valence, ("Buff", "Debuff"), f"powers.{name} = {valence!r}")

    def test_powers_table_is_keyed_by_display_name_not_class_name(self):
        # A class-name key ("VulnerablePower") can never match a real
        # EffectEvent.power value ("Vulnerable") -- pin that the committed
        # artifact uses the display form.
        doc = self._load()
        self.assertIn("Vulnerable", doc["powers"])
        self.assertNotIn("VulnerablePower", doc["powers"])
        self.assertEqual(doc["powers"]["Vulnerable"], "Debuff")
        self.assertEqual(doc["powers"]["Strength"], "Buff")

    def test_relics_are_non_empty(self):
        # Regression pin for fix round 1: relics must not silently go back to
        # {} because of an over-eager /ui/ exclusion.
        doc = self._load()
        self.assertGreater(len(doc["relics"]), 0)
        self.assertIn("event:/sfx/ui/relic_activate_draw", doc["relics"].values())

    def test_death_blow_is_not_mislabelled_attack(self):
        # Fix round 2: DeathBlowIntent must report "DeathBlow", not "Attack".
        doc = self._load()
        intents = {move["intent"] for entry in doc["monsters"].values() for move in entry["moves"]}
        self.assertIn("DeathBlow", intents)

    def test_status_intent_is_not_left_as_the_raw_prefix(self):
        # Fix round 2: StatusIntent must report "StatusCard" (the real enum
        # name), never fall through to the unmapped raw prefix "Status".
        doc = self._load()
        intents = {move["intent"] for entry in doc["monsters"].values() for move in entry["moves"]}
        self.assertNotIn("Status", intents)


class TestMonstersCardsRelicsAreKeyedByDisplayTitleNotClassName(unittest.TestCase):
    """Finding 1, 2026-08-23 final review: `monsters`/`cards`/`relics` used
    to be keyed by the decomp's file stem (class name), but the wire
    (`RunSimulator.cs`'s `_loc.Monster`/`_loc.Card`/`_loc.Relic`) only ever
    sends the localized DISPLAY name — the exact trap `load_power_table`
    already solved for `powers`, applied to nothing else. Every fixture in
    `src/sound/*.test.ts` supplied class-name keys too, so the two sides
    agreed inside the test suite and disagreed only in production; a class-
    name assertion here would make the SAME mistake at the Python layer, so
    these look up by real display title, several of them multi-word (a
    single-word title can never distinguish "keyed correctly" from "keyed by
    class name" when the two happen to coincide).

    The decisive check is `test_real_recorded_run_enemy_names_resolve`:
    display names actually seen on the wire in a real recorded run
    (`sts2-pycli/logs/20260705_185009_Ironclad_1.jsonl`), not synthetic
    fixtures. Mutation-tested: reverting `build_map` to key by `cs.stem`
    (`monsters[cs.stem] = entry` etc.) fails every test in this class.
    """

    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )

    def _load(self):
        return json.loads(self.ARTIFACT.read_text())

    def test_no_camelcase_class_name_keys_a_monster(self):
        # A class name never contains a space; every real v0.107.1 monster
        # display title with more than one word does (verified against
        # data/v0.107.1/monsters.json). A bare CamelCase key (no space, more
        # than one capital-initial word run together) is exactly the
        # class-name-keying bug's signature.
        doc = self._load()
        offenders = [k for k in doc["monsters"] if " " not in k and re.search(r"[a-z][A-Z]", k)]
        self.assertEqual(offenders, [], f"class-name-shaped monster keys: {offenders}")

    def test_multi_word_monster_titles_are_present(self):
        # These are real v0.107.1 display titles with a shipped death/hurt or
        # attack sound. Under the old cs.stem keying every one of them was
        # reachable only under its (spaceless) class name -- unreachable from
        # the wire, which sends the title.
        doc = self._load()
        for title in ("Owl Magistrate", "Fuzzy Wurm Crawler", "Shrinker Beetle"):
            self.assertIn(title, doc["monsters"], f"{title!r} missing from monsters")

    def test_real_recorded_run_enemy_names_resolve(self):
        # The decisive test: names actually observed on the wire in a real
        # recorded run (sts2-pycli/logs/20260705_185009_Ironclad_1.jsonl), not
        # synthetic fixtures (the final review's finding — a fixture-only test
        # can't distinguish "keyed correctly" from "keyed by class name" when
        # both sides of the test agree on the wrong namespace). All six named
        # by the review were confirmed unresolvable under the old class-name
        # keying ("CubexConstruct" != "Cubex Construct").
        #
        # Two of those six are deliberately NOT asserted here, for reasons
        # unrelated to keying (confirmed by calling `_monsters_by_class`
        # directly, bypassing the title join entirely):
        #   - "Cubex Construct" (CubexConstruct) and "Snapping Jaxfruit"
        #     (SnappingJaxfruit) carry NO one-shot sfx call for any move at
        #     all in v0.107.1 -- both monsters' attacks are pure dynamic-
        #     parameter loops (`SfxCmd.PlayLoop`/`SetParam`, no
        #     `SfxCmd.Play`/`WithAttackerFx`), so `parse_move_sfx` finds
        #     nothing to key regardless of the table's namespace. Re-keying
        #     by display title cannot fix a monster with nothing to key.
        #   - "Eye With Teeth" (capital "With") is the one name from that log
        #     this does NOT check under its logged spelling: v0.107.1's own
        #     localization title is "Eye with Teeth" (lowercase "with",
        #     data/v0.107.1/monsters.json) -- that log predates a text-only
        #     capitalization fix, confirmed by
        #     `sts2_hint.canonical_maps.MONSTER_TITLE_TO_ID`'s own historical
        #     note of the identical case flip. Keying by the CURRENT decomp's
        #     title is correct; asserting the stale log's superseded casing
        #     here would be re-baking a bug, not proving the fix — so it's
        #     checked below under its current, correct casing instead.
        doc = self._load()
        names = (
            "Eye with Teeth",  # current casing; the log's "Eye With Teeth" predates a text fix
            "Fuzzy Wurm Crawler",
            "Shrinker Beetle",
            "Slithering Strangler",
        )
        for name in names:
            self.assertIn(name, doc["monsters"], f"real wire name {name!r} does not resolve")

    def test_multi_word_card_and_relic_titles_are_present(self):
        doc = self._load()
        # BloodWall -> "Blood Wall" (Ironclad power card with its own sfx).
        self.assertIn("Blood Wall", doc["cards"])
        self.assertNotIn("BloodWall", doc["cards"])
        # IronClub -> "Iron Club" relic (all 6 real sound-carrying relics
        # share one generic activation event, so this also exercises the
        # "different titles, equal values" non-conflict path).
        self.assertIn("Iron Club", doc["relics"])
        self.assertNotIn("IronClub", doc["relics"])


class TestStripUnresolvedTemplate(unittest.TestCase):
    """Re-review finding 4 (2026-08-24): a title carrying an unresolved
    loc-template tail (TestSubject's "Test Subject #C{Count}") is stripped
    to its bare species name, not dropped whole -- the workbench's own
    bridge (`sts2_hint.display_maps.monster_title`) already does the same
    strip before the name ever reaches the wire, so the stripped form IS a
    real, matchable wire string.
    """

    def test_strips_a_specimen_number_tail(self):
        from scripts.build_sfx_map import _strip_unresolved_template

        self.assertEqual(_strip_unresolved_template("Test Subject #C{Count}"), "Test Subject")

    def test_is_a_no_op_on_a_title_with_no_template(self):
        from scripts.build_sfx_map import _strip_unresolved_template

        self.assertEqual(_strip_unresolved_template("Owl Magistrate"), "Owl Magistrate")

    def test_class_titles_keys_by_the_stripped_form(self):
        from scripts.build_sfx_map import _class_titles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomp = root / "decompiled" / "vtest"
            decomp.mkdir(parents=True)
            _write_titles_fixture(
                root,
                "monsters.json",
                [{"class_name": "TestSubject", "title": "Test Subject #C{Count}"}],
            )
            titles = _class_titles(decomp, "monsters.json")
            self.assertEqual(titles, {"TestSubject": "Test Subject"})


class TestMergeAliasTitles(unittest.TestCase):
    """Re-review finding 1 (2026-08-24): a monster whose C# `Title` getter
    switches to a different localization key under some in-combat state
    (ToughEgg -> "Tough Egg" pre-hatch, "Hatchling" post-hatch) must be
    reachable under EVERY title it can show, not just the one
    `extract_monsters.py`'s static extraction sees. `_merge_alias_titles` is
    the general mechanism (any future runtime-title-switching monster is a
    new table entry, not a new code path); these test it directly, against
    synthetic data, independent of the real ToughEgg fixture below.
    """

    def test_alias_resolves_to_the_same_value_as_the_primary_title(self):
        from scripts.build_sfx_map import _merge_alias_titles

        class_values = {"ToughEgg": {"moves": ["nibble"]}}
        keyed = {"Tough Egg": {"moves": ["nibble"]}}
        conflicts: set[str] = set()
        _merge_alias_titles(class_values, keyed, conflicts, {"Hatchling": "ToughEgg"})
        self.assertEqual(keyed["Hatchling"], keyed["Tough Egg"])
        self.assertEqual(conflicts, set())

    def test_an_alias_whose_class_has_no_sound_entry_contributes_nothing(self):
        from scripts.build_sfx_map import _merge_alias_titles

        keyed: dict[str, object] = {}
        conflicts: set[str] = set()
        _merge_alias_titles({}, keyed, conflicts, {"Hatchling": "ToughEgg"})
        self.assertNotIn("Hatchling", keyed)

    def test_an_alias_colliding_with_a_different_existing_value_is_dropped_and_reported(self):
        from scripts.build_sfx_map import _merge_alias_titles

        class_values = {"ToughEgg": "A", "Other": "B"}
        # "Hatchling" already legitimately resolves to a DIFFERENT class's
        # value here -- the alias must not silently overwrite it.
        keyed = {"Hatchling": "B"}
        conflicts: set[str] = set()
        _merge_alias_titles(class_values, keyed, conflicts, {"Hatchling": "ToughEgg"})
        self.assertNotIn("Hatchling", keyed)
        self.assertIn("Hatchling", conflicts)


class TestToughEggReachableUnderBothWireTitles(unittest.TestCase):
    """Checked against the real, committed artifact -- the actual ToughEgg
    case, not a synthetic stand-in. Both "Tough Egg" (pre-hatch) and
    "Hatchling" (post-hatch) are real wire names for the SAME class; a
    real-recorded-run test can't distinguish them (a run would show only
    whichever state it happened to record), so this pins the artifact
    property directly instead.
    """

    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )

    def test_tough_egg_and_hatchling_share_the_same_entry(self):
        doc = json.loads(self.ARTIFACT.read_text())
        self.assertIn("Tough Egg", doc["monsters"])
        self.assertIn("Hatchling", doc["monsters"])
        self.assertEqual(doc["monsters"]["Tough Egg"], doc["monsters"]["Hatchling"])

    def test_both_the_hatch_and_the_post_hatch_attack_events_ship_in_it(self):
        doc = json.loads(self.ARTIFACT.read_text())
        events = {mv["event"] for mv in doc["monsters"]["Hatchling"]["moves"]}
        self.assertIn("event:/sfx/enemy/enemy_attacks/tough_egg/tough_egg_hatch", events)
        self.assertIn("event:/sfx/enemy/enemy_attacks/tough_egg/tough_egg_attack", events)


class TestMapPopulationCoversEveryProducibleMonsterTitle(unittest.TestCase):
    """Re-review finding 2 (2026-08-24): the original keying guard checked a
    CamelCase-shape rule plus a SAMPLE of real wire names -- neither can see
    a key that IS a valid localization title but is not a name the wire
    producer ever emits for that class in EVERY state (Tough Egg's
    "Hatchling" post-hatch swap, invisible to a sample of names drawn from
    one recorded run). This checks the FULL population instead of a sample:
    every map key must be a real, producible title (primary or alias), and
    every known alias must actually resolve.
    """

    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )
    MONSTERS_JSON = Path(__file__).resolve().parents[1] / "data" / "v0.107.1" / "monsters.json"

    def _population(self) -> set[str]:
        from scripts.build_sfx_map import MONSTER_TITLE_ALIASES, _strip_unresolved_template

        entries = json.loads(self.MONSTERS_JSON.read_text())
        population = {
            _strip_unresolved_template(x["title"])
            for x in entries
            if x.get("title") and x.get("class_name")
        }
        population.discard("")
        population |= set(MONSTER_TITLE_ALIASES)
        return population

    def test_no_map_key_falls_outside_the_full_title_population(self):
        doc = json.loads(self.ARTIFACT.read_text())
        offenders = set(doc["monsters"]) - self._population()
        self.assertEqual(offenders, set(), f"monster keys with no producible title: {offenders}")

    def test_every_known_alias_resolves_to_its_class_own_entry(self):
        # The decisive half, run both ways and quoted in the fix report:
        # this FAILS before `_merge_alias_titles` is wired into
        # `build_map_with_report` (no "Hatchling" key at all) and PASSES
        # after.
        from scripts.build_sfx_map import MONSTER_TITLE_ALIASES

        doc = json.loads(self.ARTIFACT.read_text())
        titles = {x["class_name"]: x["title"] for x in json.loads(self.MONSTERS_JSON.read_text())}
        for alias_title, class_name in MONSTER_TITLE_ALIASES.items():
            self.assertIn(alias_title, doc["monsters"], f"{alias_title!r} missing from monsters")
            primary_title = titles.get(class_name)
            if primary_title and primary_title in doc["monsters"]:
                self.assertEqual(doc["monsters"][alias_title], doc["monsters"][primary_title])


@needs_decomp
class TestIntentKindDerivedFromDecomp(unittest.TestCase):
    # Fix round 2: the reviewer's own hand-written _INTENT_KIND table shipped
    # DeathBlowIntent as "Attack" and had no entry at all for StatusIntent
    # (silently falling through to the raw, wrong prefix "Status"). Both are
    # now derived from MegaCrit.Sts2.Core.MonsterMoves.Intents/*.cs via
    # load_intent_tables/parse_move_states's lazy default, not hand-typed.
    INTENT_SOURCE = """
        MoveState moveState = new MoveState("FINISH_MOVE", FinishMove, new DeathBlowIntent(FinishingDamageCalc));
        MoveState moveState2 = new MoveState("MARK_MOVE", MarkMove, new StatusIntent(3));
        MoveState moveState3 = new MoveState("GUARD_MOVE", GuardMove, new DefendIntent());
    """  # noqa: E501 — a single over-length fixture line, not worth wrapping

    def test_death_blow_intent_maps_to_death_blow_not_attack(self):
        moves = parse_move_states(self.INTENT_SOURCE)
        self.assertEqual(moves[0]["intent"], "DeathBlow")

    def test_death_blow_still_gets_a_fixed_hit_count(self):
        # DeathBlowIntent inherits SingleAttackIntent.Repeats => 1 (no
        # override) -- it's a single finishing hit, not a multi-attack.
        moves = parse_move_states(self.INTENT_SOURCE)
        self.assertEqual(moves[0]["hits"], 1)

    def test_status_intent_maps_to_status_card(self):
        moves = parse_move_states(self.INTENT_SOURCE)
        self.assertEqual(moves[1]["intent"], "StatusCard")

    def test_non_attack_intents_carry_no_hits_or_damage(self):
        # RunSimulator.cs only ever sets damage/hits `if (intent is
        # AttackIntent atk)` -- StatusCard and Defend moves must carry
        # neither key at all, not a wrong/zero value.
        moves = parse_move_states(self.INTENT_SOURCE)
        self.assertNotIn("hits", moves[1])
        self.assertNotIn("damage", moves[1])
        self.assertNotIn("hits", moves[2])
        self.assertNotIn("damage", moves[2])


class TestIntentClassDeclIsAnchored(unittest.TestCase):
    """`_INTENT_CLASS_DECL` must match the declaration, not doc-comment prose.

    A bare `r"class (\\w+)"` also matches the word "class" inside an XML doc
    comment, and `.search()` returns the FIRST match -- that exact pattern
    silently mis-keyed three power base classes under "represents" before it
    was anchored. No Intents file carries such prose today, so this guards a
    dormant footgun: if one ever does, the class's IntentType stops resolving,
    cues fall back to a wrong kind, and nothing else goes red.
    """

    PROSE_FIRST = """
// <summary>
// This class represents a monster winding up for a big finishing attack.
// </summary>
public class WindupIntent : AttackIntent
{
	public override IntentType IntentType => IntentType.Attack;
}
"""

    def load_from(self, sources: dict[str, str]):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        intents = root / "MegaCrit.Sts2.Core.MonsterMoves.Intents"
        intents.mkdir(parents=True)
        for name, text in sources.items():
            (intents / name).write_text(text, encoding="utf-8")
        return load_intent_tables(root)

    def test_prose_before_the_declaration_does_not_capture_the_wrong_name(self):
        kinds, attacks = self.load_from({"WindupIntent.cs": self.PROSE_FIRST})

        # Keyed by the real class prefix. With the bare pattern this comes out
        # keyed under "represents" instead -- and "Windup" is simply absent.
        self.assertEqual(kinds, {"Windup": "Attack"})
        self.assertEqual(attacks, frozenset({"Windup"}))

    PLAIN = """
public class DefendIntent : Intent
{
	public override IntentType IntentType => IntentType.Defend;
}
"""

    def test_a_normal_declaration_still_parses(self):
        # The control: anchoring must not cost us the ordinary case.
        kinds, _ = self.load_from({"DefendIntent.cs": self.PLAIN})

        self.assertEqual(kinds, {"Defend": "Defend"})


class TestLoadPowerValences(unittest.TestCase):
    """`load_power_valences` derives Buff/Debuff per power class from the
    decomp -- see the module comment above `_POWER_CLASS_DECL`. These fixtures
    mirror the real shapes found in MegaCrit.Sts2.Core.Models.Powers: a direct
    `Type =>` literal, a concrete subclass that inherits a *conditional*
    `Type` getter from an abstract base (TemporaryStrengthPower's real
    pattern), and the doc-comment regression that made those bases silently
    unresolvable (see task-6 report).
    """

    def _write_fixture(self, root: Path) -> Path:
        powers = root / "MegaCrit.Sts2.Core.Models.Powers"
        powers.mkdir(parents=True)
        (powers / "VulnerablePower.cs").write_text(
            "public sealed class VulnerablePower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Debuff;\n"
            "}\n"
        )
        (powers / "StrengthPower.cs").write_text(
            "public sealed class StrengthPower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Buff;\n"
            "}\n"
        )
        # Abstract base with a doc comment that starts "This class
        # represents..." BEFORE the real declaration -- a bare `class (\w+)`
        # regex greedily matches "class represents" here.
        (powers / "TemporaryStrengthPower.cs").write_text(
            "/// <summary>\n"
            "/// This class represents a buff/debuff that gives/takes strength.\n"
            "/// </summary>\n"
            "public abstract class TemporaryStrengthPower : PowerModel, ITemporaryPower\n"
            "{\n"
            "\tpublic override PowerType Type\n"
            "\t{\n"
            "\t\tget\n"
            "\t\t{\n"
            "\t\t\tif (!IsPositive)\n"
            "\t\t\t{\n"
            "\t\t\t\treturn PowerType.Debuff;\n"
            "\t\t\t}\n"
            "\t\t\treturn PowerType.Buff;\n"
            "\t\t}\n"
            "\t}\n"
            "\n"
            "\tprotected virtual bool IsPositive => true;\n"
            "}\n"
        )
        # Inherits the conditional getter with NO override -> the base's
        # default (IsPositive == true) applies -> Buff.
        (powers / "FlexPotionPower.cs").write_text(
            "public class FlexPotionPower : TemporaryStrengthPower\n{\n}\n"
        )
        # Inherits the conditional getter WITH an override -> Debuff.
        (powers / "DarkShacklesPower.cs").write_text(
            "public class DarkShacklesPower : TemporaryStrengthPower\n"
            "{\n"
            "\tprotected override bool IsPositive => false;\n"
            "}\n"
        )
        return powers

    def test_direct_literal_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_valences

            valences = load_power_valences(root)
            self.assertEqual(valences["VulnerablePower"], "Debuff")
            self.assertEqual(valences["StrengthPower"], "Buff")

    def test_base_class_with_doc_comment_prose_still_resolves(self):
        # Regression: a bare `r"class (\w+)"` matches "This class represents"
        # in the XML doc comment above the real declaration -- .search()
        # takes that FIRST match, silently mis-keying the whole class under
        # "represents" and making every concrete subclass unresolvable.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_valences

            valences = load_power_valences(root)
            self.assertIn("TemporaryStrengthPower", valences)

    def test_inherited_conditional_getter_with_no_override_uses_the_base_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_valences

            self.assertEqual(load_power_valences(root)["FlexPotionPower"], "Buff")

    def test_inherited_conditional_getter_with_an_override_uses_the_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_valences

            self.assertEqual(load_power_valences(root)["DarkShacklesPower"], "Debuff")

    def test_a_power_with_no_type_declaration_anywhere_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            powers = root / "MegaCrit.Sts2.Core.Models.Powers"
            powers.mkdir(parents=True)
            (powers / "MysteryPower.cs").write_text(
                "public sealed class MysteryPower : PowerModel\n{\n}\n"
            )
            from scripts.build_sfx_map import load_power_valences

            self.assertNotIn("MysteryPower", load_power_valences(root))


class TestLoadPowerAllowNegative(unittest.TestCase):
    """`load_power_allow_negative` reads `PowerModel.AllowNegative`, whose
    `virtual` default is false (`PowerModel.cs:240`) — so this table records
    only the overriding classes, and absence is a real answer rather than a
    gap.
    """

    def _write_fixture(self, root: Path) -> Path:
        powers = root / "MegaCrit.Sts2.Core.Models.Powers"
        powers.mkdir(parents=True)
        (powers / "StrengthPower.cs").write_text(
            "public sealed class StrengthPower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Buff;\n"
            "\tpublic override bool AllowNegative => true;\n"
            "}\n"
        )
        (powers / "VulnerablePower.cs").write_text(
            "public sealed class VulnerablePower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Debuff;\n"
            "}\n"
        )
        # A base that opts in, and a subclass that says nothing — the
        # inheritance walk is what makes the subclass come out true.
        (powers / "SignedBasePower.cs").write_text(
            "public abstract class SignedBasePower : PowerModel\n"
            "{\n"
            "\tpublic override bool AllowNegative => true;\n"
            "}\n"
        )
        (powers / "InheritsSignedPower.cs").write_text(
            "public sealed class InheritsSignedPower : SignedBasePower\n{\n}\n"
        )
        # ...and one that opts back OUT, which must beat the base.
        (powers / "OptsOutPower.cs").write_text(
            "public sealed class OptsOutPower : SignedBasePower\n"
            "{\n"
            "\tpublic override bool AllowNegative => false;\n"
            "}\n"
        )
        return powers

    def test_only_the_opting_in_classes_appear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_allow_negative

            table = load_power_allow_negative(root)
            self.assertEqual(table["StrengthPower"], True)
            self.assertNotIn("VulnerablePower", table)

    def test_the_flag_is_inherited_and_an_override_beats_the_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            from scripts.build_sfx_map import load_power_allow_negative

            table = load_power_allow_negative(root)
            self.assertIn("InheritsSignedPower", table)
            self.assertNotIn("OptsOutPower", table)

    def test_the_shipped_list_is_display_titles_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomp = root / "decompiled" / "vtest"
            decomp.mkdir(parents=True)
            self._write_fixture(decomp)
            data = root / "data" / "vtest"
            data.mkdir(parents=True)
            (data / "powers.json").write_text(
                json.dumps(
                    [
                        {"class_name": "StrengthPower", "title": "Strength"},
                        {"class_name": "InheritsSignedPower", "title": "Dexterity"},
                        {"class_name": "VulnerablePower", "title": "Vulnerable"},
                    ]
                )
            )
            from scripts.build_sfx_map import load_power_allow_negative_table

            # Display titles, sorted, and only the opting-in classes — the same
            # class-name-to-title join `load_power_table` does, because the wire
            # only ever carries the display name.
            self.assertEqual(load_power_allow_negative_table(decomp), ["Dexterity", "Strength"])


class TestLoadPowerTable(unittest.TestCase):
    """`load_power_table` joins class-keyed valences to their display title
    (the only form `EffectEvent.power` ever carries) -- see its docstring.
    """

    def _write_fixture(self, root: Path, powers_json: list[dict]) -> None:
        decomp = root / "decompiled" / "vtest"
        powers = decomp / "MegaCrit.Sts2.Core.Models.Powers"
        powers.mkdir(parents=True)
        (powers / "VulnerablePower.cs").write_text(
            "public sealed class VulnerablePower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Debuff;\n"
            "}\n"
        )
        (powers / "BackAttackLeftPower.cs").write_text(
            "public sealed class BackAttackLeftPower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Buff;\n"
            "}\n"
        )
        (powers / "BackAttackRightPower.cs").write_text(
            "public sealed class BackAttackRightPower : PowerModel\n"
            "{\n"
            "\tpublic override PowerType Type => PowerType.Buff;\n"
            "}\n"
        )
        data = root / "data" / "vtest"
        data.mkdir(parents=True)
        (data / "powers.json").write_text(json.dumps(powers_json))
        self.decomp_root = decomp

    def test_keys_by_display_title_not_class_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(
                root,
                [{"class_name": "VulnerablePower", "title": "Vulnerable"}],
            )
            from scripts.build_sfx_map import load_power_table

            table = load_power_table(self.decomp_root)
            self.assertEqual(table, {"Vulnerable": "Debuff"})
            self.assertNotIn("VulnerablePower", table)

    def test_two_classes_agreeing_on_one_title_merge_without_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(
                root,
                [
                    {"class_name": "BackAttackLeftPower", "title": "Back Attack"},
                    {"class_name": "BackAttackRightPower", "title": "Back Attack"},
                ],
            )
            from scripts.build_sfx_map import load_power_table

            self.assertEqual(load_power_table(self.decomp_root), {"Back Attack": "Buff"})

    def test_two_classes_disagreeing_on_one_title_are_omitted_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(
                root,
                [
                    # Vulnerable is Debuff in the decomp fixture; lying about
                    # its title so it collides with the Buff-typed class.
                    {"class_name": "VulnerablePower", "title": "Ambiguous"},
                    {"class_name": "BackAttackLeftPower", "title": "Ambiguous"},
                ],
            )
            from scripts.build_sfx_map import load_power_table

            self.assertNotIn("Ambiguous", load_power_table(self.decomp_root))

    def test_a_class_with_no_title_entry_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, [])  # powers.json has no entries at all
            from scripts.build_sfx_map import load_power_table

            self.assertEqual(load_power_table(self.decomp_root), {})

    def test_missing_data_powers_json_yields_an_empty_table_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomp = root / "decompiled" / "vtest"
            powers = decomp / "MegaCrit.Sts2.Core.Models.Powers"
            powers.mkdir(parents=True)
            (powers / "VulnerablePower.cs").write_text(
                "public sealed class VulnerablePower : PowerModel\n"
                "{\n"
                "\tpublic override PowerType Type => PowerType.Debuff;\n"
                "}\n"
            )
            # No data/vtest/powers.json written at all.
            from scripts.build_sfx_map import load_power_table

            self.assertEqual(load_power_table(decomp), {})


class TestBuildMapIncludesPowers(unittest.TestCase):
    def test_powers_key_present_and_derived_from_the_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomp = root / "decompiled" / "vtest"
            monsters = decomp / "MegaCrit.Sts2.Core.Models.Monsters"
            monsters.mkdir(parents=True)
            (monsters / "OwlMagistrate.cs").write_text(SOURCE)
            _write_intents_fixture(decomp)
            powers = decomp / "MegaCrit.Sts2.Core.Models.Powers"
            powers.mkdir(parents=True)
            (powers / "VulnerablePower.cs").write_text(
                "public sealed class VulnerablePower : PowerModel\n"
                "{\n"
                "\tpublic override PowerType Type => PowerType.Debuff;\n"
                "}\n"
            )
            data = root / "data" / "vtest"
            data.mkdir(parents=True)
            (data / "powers.json").write_text(
                json.dumps([{"class_name": "VulnerablePower", "title": "Vulnerable"}])
            )

            doc = build_map(decomp, "vtest")

            self.assertEqual(doc["powers"], {"Vulnerable": "Debuff"})


class TestResolveIntArg(unittest.TestCase):
    # hits/damage resolution: literal, same-file named member (plain or
    # AscensionHelper-guarded, taking the fallback/base value), or
    # unresolvable -> omitted. Mirrors real shapes found in the decomp (e.g.
    # Entomancer.BeesRepeat => AscensionHelper.GetValueIfAscension(
    # AscensionLevel.DeadlyEnemies, 8, 7) -- non-ascension value is 7).
    MEMBER_SOURCE = """
    private int FixedRepeat => 2;

    private int AscensionGuardedDamage => AscensionHelper.GetValueIfAscension(AscensionLevel.DeadlyEnemies, 8, 7);
    """  # noqa: E501 — a single over-length fixture line, not worth wrapping

    def test_literal_digit_resolves_directly(self):
        from scripts.build_sfx_map import _resolve_int_arg

        self.assertEqual(_resolve_int_arg("6", self.MEMBER_SOURCE), 6)

    def test_named_member_with_a_fixed_literal_resolves(self):
        from scripts.build_sfx_map import _resolve_int_arg

        self.assertEqual(_resolve_int_arg("FixedRepeat", self.MEMBER_SOURCE), 2)

    def test_ascension_guarded_member_resolves_to_the_fallback_value(self):
        from scripts.build_sfx_map import _resolve_int_arg

        # non-ascension (base-difficulty) value is the THIRD ctor arg, 7 --
        # verified against MegaCrit.Sts2.Core.Helpers/AscensionHelper.cs.
        self.assertEqual(_resolve_int_arg("AscensionGuardedDamage", self.MEMBER_SOURCE), 7)

    def test_unresolvable_identifier_is_none_not_a_guess(self):
        from scripts.build_sfx_map import _resolve_int_arg

        self.assertIsNone(_resolve_int_arg("SomeMethodGroupNotAnInt", self.MEMBER_SOURCE))


@needs_decomp
class TestDecompReconciliation(unittest.TestCase):
    # Nothing previously checked raw decomp scope against what lands in the
    # map -- the same class of gap that let the null-event and empty-relics
    # bugs ship silently. Pinned so a regex/glob regression that quietly
    # shrinks what's scanned fails a test instead of shrinking the map.
    DECOMP_ROOT = Path(__file__).resolve().parents[1] / "decompiled" / "v0.107.1"

    def test_scans_the_full_monster_cs_count(self):
        cs_files = list((self.DECOMP_ROOT / "MegaCrit.Sts2.Core.Models.Monsters").glob("*.cs"))
        self.assertEqual(len(cs_files), 121)

    def test_scans_the_full_card_cs_count(self):
        cs_files = list(self.DECOMP_ROOT.glob("MegaCrit.Sts2.Core.Models.Cards*/**/*.cs"))
        self.assertEqual(len(cs_files), 585)

    def test_only_nine_cards_carry_an_event_literal(self):
        # Confirmed genuinely exhaustive (not a parsing gap): of 585 card
        # .cs files, only 9 contain any event:/ literal at all.
        from scripts.build_sfx_map import _first_event

        cs_files = self.DECOMP_ROOT.glob("MegaCrit.Sts2.Core.Models.Cards*/**/*.cs")
        with_event = sum(
            1 for cs in cs_files if _first_event(cs.read_text(encoding="utf-8", errors="replace"))
        )
        self.assertEqual(with_event, 9)


class TestUnresolvableFieldsAreOmittedNotDefaulted(unittest.TestCase):
    """Regression test for the ONE rule fix round 2 exists for: an
    unresolvable damage/hits value must be OMITTED from the map, never
    defaulted (e.g. damage=0, hits=1) — a wrong number matches the wrong
    sound, an absent one lets the web degrade cleanly, and a defaulted
    hits=1 in particular is indistinguishable from a genuine single-hit
    move, so it would never look wrong just by inspection.

    These five moves are the decomp's ONLY genuinely-unresolvable cases
    (confirmed by team-lead review reading each site): GasBomb/
    WaterfallGiant's damage reads live combat state (`Current
    PressureGunDamage` accumulates across turns; WaterfallGiant's
    PRESSURE_GUN_MOVE and EXPLODE_MOVE do too), TheForgotten's DREAD damage
    reads a live Dexterity power, and TestSubject's MULTI_CLAW_MOVE hit
    count is `Base + Extra` — none of the five reduce to a static int, so
    none of the five can resolve. If this test starts failing because one
    of them now HAS a value, that's real progress, not a bug in the test —
    but if it's failing because the key exists with a wrong number, that's
    the regression this test exists to catch.

    Checked against the actual COMMITTED artifact, not generator
    intermediates — assertNotIn, never a falsy-value check, since
    move.get("damage") == 0 would pass under exactly the defaulting bug
    this guards against. (Verified by mutating the generator locally to
    default instead of omit and confirming these tests catch it — see
    task-4-report.)

    Looked up by DISPLAY TITLE, not class name (finding 1 of the 2026-08-23
    final review: the artifact keys `monsters` by title, e.g. "Gas Bomb" not
    "GasBomb" — a class-name lookup here would silently always fail once the
    keying fix landed, KeyError-ing instead of testing the intended rule).
    TestSubject's static title is "Test Subject #C{Count}" — genuinely
    templated, resolved only at runtime — but `_strip_unresolved_template`
    (2026-08-24 re-review, finding 4) strips that tail to the bare species
    name "Test Subject", which IS a real, stable wire string: the
    workbench's own bridge (`sts2_hint.display_maps.monster_title`) already
    strips the identical tail before the name ever reaches the wire. So
    MULTI_CLAW_MOVE's "no hits key" claim is checked under "Test Subject",
    same as every other monster here, not under an absent-monster fallback.
    """

    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )

    def _move(self, monster: str, move_name: str):
        doc = json.loads(self.ARTIFACT.read_text())
        for move in doc["monsters"][monster]["moves"]:
            if move["name"] == move_name:
                return move
        self.fail(f"{monster}.{move_name} not found in the generated map")

    def test_gas_bomb_explode_has_no_damage_key(self):
        self.assertNotIn("damage", self._move("Gas Bomb", "EXPLODE_MOVE"))

    def test_waterfall_giant_explode_has_no_damage_key(self):
        self.assertNotIn("damage", self._move("Waterfall Giant", "EXPLODE_MOVE"))

    def test_waterfall_giant_pressure_gun_has_no_damage_key(self):
        self.assertNotIn("damage", self._move("Waterfall Giant", "PRESSURE_GUN_MOVE"))

    def test_the_forgotten_dread_has_no_damage_key(self):
        self.assertNotIn("damage", self._move("The Forgotten", "DREAD"))

    def test_test_subject_multi_claw_has_no_hits_key(self):
        self.assertNotIn("hits", self._move("Test Subject", "MULTI_CLAW_MOVE"))

    def test_test_subject_is_keyed_under_its_stripped_species_name(self):
        # Re-review finding 4 (2026-08-24): "TestSubject" (the class name)
        # must never be a key, and neither must the raw templated title.
        doc = json.loads(self.ARTIFACT.read_text())
        self.assertIn("Test Subject", doc["monsters"])
        self.assertNotIn("TestSubject", doc["monsters"])
        self.assertNotIn("Test Subject #C{Count}", doc["monsters"])


DECOMP_ROOT = Path(__file__).resolve().parents[1] / "decompiled" / "v0.107.1"

# `Ironclad.GetHeavyAnimIfApplicable` / `Necrobinder.GetSummonAnimIfApplicable`
# as the real decomp declares them, so the unit tests below don't have to read
# the whole tree to exercise the character-conditional branch.
FALLBACKS = {
    "Ironclad.GetHeavyAnimIfApplicable(": ("Attack", "Ironclad"),
    "Necrobinder.GetSummonAnimIfApplicable(": ("Cast", "Necrobinder"),
}


class TestCardPlayerAnim(unittest.TestCase):
    """Which sound a card makes the PLAYER produce -- the layer the workbench
    shipped without, which is why an Ironclad Strike used to be nothing but a
    card woosh."""

    def test_plain_attack_card_animates_attack(self):
        # StrikeIronclad, verbatim shape: `FromCard` alone seeds
        # `_attackerAnimName = "Attack"` (AttackCommand.cs:220).
        src = """
        await DamageCmd.Attack(6m).FromCard(this).Targeting(cardPlay.Target)
            .WithHitFx("vfx/vfx_attack_slash")
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertEqual(card_player_anim(src, FALLBACKS), ("attack", None))

    def test_v0111_from_card_passes_the_play_along(self):
        # v0.111.0's shape: `.FromCard(this, cardPlay)`. Matching only the
        # v0.107.1 `.FromCard(this)` lost ~160 attack cards' player anim on
        # a v0.111.0 map (GH #877).
        src = """
        await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this, cardPlay).Targeting(cardPlay.Target)
            .WithHitFx("vfx/vfx_attack_slash")
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertEqual(card_player_anim(src, FALLBACKS), ("attack", None))
        # Not a different `FromCard(thisCard)` local.
        local = src.replace("(this, cardPlay)", "(thisCard)")
        self.assertIsNone(card_player_anim(local, FALLBACKS))

    def test_explicit_cast_trigger(self):
        src = 'await CreatureCmd.TriggerAnim(base.Owner.Creature, "Cast", 0.25f);'
        self.assertEqual(card_player_anim(src, FALLBACKS), ("cast", None))

    def test_power_up_resolves_to_the_cast_sound(self):
        # `PowerUpSfx => CastSfx` (CharacterModel.cs:204) -- a separate
        # animation, but the SAME sound. Mutation caught: giving "PowerUp"
        # its own (nonexistent) sfx slot.
        src = 'await CreatureCmd.TriggerAnim(base.Owner.Creature, "PowerUp", 0.25f);'
        self.assertEqual(card_player_anim(src, FALLBACKS), ("cast", None))

    def test_card_that_animates_nothing_makes_no_sound(self):
        # DefendIronclad: gains block, never triggers an animation. 180 cards
        # are like this, so defaulting to "attack" here would be audible on a
        # third of the deck.
        src = "await CreatureCmd.GainBlock(base.Owner.Creature, base.DynamicVars.Block, cardPlay);"
        self.assertIsNone(card_player_anim(src, FALLBACKS))

    def test_with_no_attacker_anim_silences_the_chain(self):
        # GuidingStar.
        src = """
        await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).WithNoAttackerAnim().FromCard(this)
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertIsNone(card_player_anim(src, FALLBACKS))

    def test_third_argument_retargets_the_animation_off_the_player(self):
        # ByrdSwoop animates the PET, and `TriggerAnim` only plays character
        # sfx for a creature that `IsPlayer`. Mutation caught: reading the
        # anim name and ignoring the visual-attacker argument, which would
        # make the player grunt for the bird's attack.
        src = """
        await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this).Targeting(cardPlay.Target)
            .WithAttackerAnim("Attack", base.Owner.Character.AttackAnimDelay, base.Owner.PlayerCombatState.GetPet<Byrdpip>())
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertIsNone(card_player_anim(src, FALLBACKS))

    def test_character_conditional_anim_records_the_exempt_character(self):
        # Mangle: "heavyAttack" for Ironclad (a name TriggerAnim's switch does
        # not handle -> silence), plain "Attack" for anyone else.
        src = """
        await DamageCmd.Attack(base.DynamicVars.Damage.BaseValue).FromCard(this).Targeting(cardPlay.Target)
            .WithAttackerAnim(Ironclad.GetHeavyAnimIfApplicable(base.Owner.Character), Ironclad.GetHeavyAttackDelayIfApplicable(base.Owner.Character))
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertEqual(card_player_anim(src, FALLBACKS), ("attack", "Ironclad"))

    def test_necrobinder_summon_is_cast_for_everyone_else(self):
        src = """
        await CreatureCmd.TriggerAnim(base.Owner.Creature, Necrobinder.GetSummonAnimIfApplicable(base.Owner.Character), 0.3f);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertEqual(card_player_anim(src, FALLBACKS), ("cast", "Necrobinder"))

    def test_a_runtime_variable_anim_is_unresolved_not_guessed(self):
        # SovereignBlade picks its trigger from a local computed at runtime.
        # Absent beats wrong: returning ("attack", None) here would give every
        # character the wrong sound on a card that already has its own.
        src = """
        AttackCommand attackCommand = DamageCmd.Attack(1m).FromCard(this).WithAttackerAnim(animName, delay)
            .WithAttackerFx(null, "event:/sfx/characters/regent/regent_sovereign_blade");
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertEqual(card_player_anim(src, FALLBACKS), "unresolved")

    def test_a_modifier_on_a_later_line_of_the_chain_is_still_seen(self):
        # A builder chain is ONE statement across several lines. Mutation
        # caught: scanning only the line `.FromCard(this)` sits on, which
        # would miss every `.WithNoAttackerAnim()` below it.
        src = """
        await DamageCmd.Attack(9m)
            .FromCard(this)
            .Targeting(cardPlay.Target)
            .WithNoAttackerAnim()
            .Execute(choiceContext);
        """  # noqa: E501 — verbatim C# from the decomp; keep it byte-for-byte
        self.assertIsNone(card_player_anim(src, FALLBACKS))


@needs_audio
@needs_decomp
class TestLoadCharacterSfx(unittest.TestCase):
    def test_derives_all_five_playable_characters(self):
        chars = load_character_sfx(
            DECOMP_ROOT, DECOMP_ROOT.parents[1] / "site" / "public" / "audio"
        )
        self.assertEqual(sorted(chars), ["Defect", "Ironclad", "Necrobinder", "Regent", "Silent"])
        self.assertEqual(
            chars["Ironclad"],
            {
                "attack": "event:/sfx/characters/ironclad/ironclad_attack",
                "cast": "event:/sfx/characters/ironclad/ironclad_cast",
                "die": "event:/sfx/characters/ironclad/ironclad_die",
            },
        )

    def test_drops_classes_with_no_rendered_audio(self):
        # Deprived / RandomCharacter / DeprecatedCharacter are real .cs files
        # in the same directory. The audio-root check is the ONLY thing that
        # keeps them out, which is also what makes the `Id.Entry`-is-the-
        # class-name assumption self-verifying rather than a guess.
        audio = DECOMP_ROOT.parents[1] / "site" / "public" / "audio"
        self.assertNotIn("Deprived", load_character_sfx(DECOMP_ROOT, audio))
        self.assertIn("Deprived", load_character_sfx(DECOMP_ROOT, None))


V0113_DECOMP = Path(__file__).resolve().parents[1] / "decompiled" / "v0.113.0"


@unittest.skipUnless(V0113_DECOMP.is_dir(), "needs the v0.113.0 decompilation")
class TestCharacterSfxOverride(unittest.TestCase):
    def test_a_declared_death_sfx_wins_over_the_derived_name(self):
        # v0.113.0's Fool dies with the Fugitive's sound (`Fool.cs`:
        # `public override string DeathSfx => "event:/sfx/characters/fugitive/fugitive_die"`).
        # Mutation caught: deriving `fool/fool_die`, an event the banks lack.
        fool = load_character_sfx(V0113_DECOMP, None)["Fool"]
        self.assertEqual(fool["die"], "event:/sfx/characters/fugitive/fugitive_die")
        self.assertEqual(fool["attack"], "event:/sfx/characters/fool/fool_attack")


@needs_decomp
class TestLoadCharacterAnimFallbacks(unittest.TestCase):
    def test_reads_both_halves_from_the_real_decomp(self):
        fb = load_character_anim_fallbacks(DECOMP_ROOT)
        self.assertEqual(fb["Ironclad.GetHeavyAnimIfApplicable("], ("Attack", "Ironclad"))
        self.assertEqual(fb["Necrobinder.GetSummonAnimIfApplicable("], ("Cast", "Necrobinder"))


class TestGeneratedArtifactCarriesPlayerSounds(unittest.TestCase):
    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )

    def setUp(self):
        self.doc = json.loads(self.ARTIFACT.read_text())

    def test_characters_are_keyed_by_class_name_not_display_title(self):
        # The wire sends `state.player.name` == "Ironclad". Keying this table
        # the way `cards`/`monsters`/`relics` are keyed ("The Ironclad") would
        # make every lookup miss and silently restore the woosh-only bug.
        self.assertIn("Ironclad", self.doc["characters"])
        self.assertNotIn("The Ironclad", self.doc["characters"])

    def test_card_anims_are_keyed_by_display_title(self):
        self.assertIn("Strike", self.doc["cardAnims"])
        self.assertNotIn("StrikeIronclad", self.doc["cardAnims"])

    def test_defend_carries_no_anim_entry(self):
        self.assertNotIn("Defend", self.doc["cardAnims"])

    def test_heavy_attack_cards_carry_their_ironclad_exemption(self):
        self.assertEqual(self.doc["cardAnims"]["Mangle"], {"kind": "attack", "except": "Ironclad"})


class TestAudioFingerprint(unittest.TestCase):
    """The cache-buster appended to every audio URL.

    These URLs are served `immutable, max-age=31536000` but are NOT
    content-addressed, so re-encoding puts new bytes at an unchanged URL and
    an already-primed browser will not revalidate for a year. Only a changed
    URL reaches those clients.
    """

    def _tree(self, root: Path, files: dict[str, bytes]) -> Path:
        for rel, data in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return root

    def test_is_stable_when_nothing_changed(self):
        # Load-bearing: `just build-sfx-map` runs often and must NOT invalidate
        # 8.6 MB of audio in every browser each time. Mutation caught: hashing
        # mtime or an iteration order that isn't sorted.
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(Path(d), {"sfx/a.webm": b"aa", "sfx/ui/b.webm": b"bb"})
            self.assertEqual(audio_fingerprint(root), audio_fingerprint(root))

    def test_changes_when_a_file_is_re_encoded(self):
        # Same path, different bytes -- exactly the re-encode case.
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = self._tree(Path(d1), {"sfx/a.webm": b"old"})
            b = self._tree(Path(d2), {"sfx/a.webm": b"new"})
            self.assertNotEqual(audio_fingerprint(a), audio_fingerprint(b))

    def test_changes_when_a_file_is_renamed(self):
        # Mutation caught: hashing contents only. Identical bytes at a new
        # path is a different corpus, and the old URL now 404s.
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = self._tree(Path(d1), {"sfx/a.webm": b"same"})
            b = self._tree(Path(d2), {"sfx/b.webm": b"same"})
            self.assertNotEqual(audio_fingerprint(a), audio_fingerprint(b))

    def test_changes_when_a_file_is_added(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = self._tree(Path(d1), {"sfx/a.webm": b"x"})
            b = self._tree(Path(d2), {"sfx/a.webm": b"x", "sfx/c.webm": b"y"})
            self.assertNotEqual(audio_fingerprint(a), audio_fingerprint(b))


@needs_audio
class TestGeneratedArtifactCarriesAudioRev(unittest.TestCase):
    ARTIFACT = (
        Path(__file__).resolve().parents[2] / "sts2-pycli" / "web" / "public" / "sfx_map.json"
    )
    AUDIO = Path(__file__).resolve().parents[1] / "site" / "public" / "audio"

    def test_matches_the_shipped_audio_tree(self):
        # A map whose rev doesn't match the audio beside it would bust caches
        # to a URL serving different bytes than the map was built against.
        doc = json.loads(self.ARTIFACT.read_text())
        self.assertEqual(doc["audioRev"], audio_fingerprint(self.AUDIO))


class TestParseHitSfx(unittest.TestCase):
    """`WithHitFx(vfx, sfx, tmpSfx)` — the sound an attack makes ON the target.

    Harvesting FMOD literals alone found ONE of the 81 cards that carry a hit
    sound, because the other 80 name a `res://debug_audio/` mp3. That is why
    the workbench's Ironclad thudded on nothing.
    """

    def test_reads_the_tmp_sfx_slot(self):
        self.assertEqual(
            parse_hit_sfx('.WithHitFx("vfx/vfx_attack_blunt", null, "blunt_attack.mp3")'),
            "res://debug_audio/blunt_attack.mp3",
        )

    def test_reads_the_fmod_sfx_slot(self):
        call = '.WithHitFx("vfx/vfx_attack_slash", "event:/sfx/byrdpip/byrdpip_attack")'
        self.assertEqual(parse_hit_sfx(call), "event:/sfx/byrdpip/byrdpip_attack")

    def test_a_vfx_only_call_has_no_sound(self):
        # 77 cards are in this shape (`StrikeIronclad` among them): the hit
        # shows a slash and makes no sound of its own.
        self.assertIsNone(parse_hit_sfx('.WithHitFx("vfx/vfx_attack_slash")'))

    def test_a_missing_vfx_still_yields_the_sound(self):
        self.assertEqual(
            parse_hit_sfx('.WithHitFx(null, null, "blunt_attack.mp3")'),
            "res://debug_audio/blunt_attack.mp3",
        )

    def test_an_mp3_in_the_fmod_slot_is_refused(self):
        # `FallingStar.cs:40` really does pass `"blunt_attack.mp3"` as the FMOD
        # argument — a bug in the game's own source. FMOD answers a lookup on a
        # nonexistent event key with a silent no-op, so that card's hit IS
        # silent in the shipped game. Accepting it here would give the
        # workbench a sound no player has ever heard.
        self.assertIsNone(parse_hit_sfx('.WithHitFx("vfx/vfx_starry_impact", "blunt_attack.mp3")'))

    def test_a_card_with_no_attack_at_all_has_no_hit_sound(self):
        self.assertIsNone(parse_hit_sfx("await BlockCmd.Gain(base.DynamicVars.Block.BaseValue);"))


class TestDebugAudioPaths(unittest.TestCase):
    """The two schemes resolve into ONE encoded tree."""

    def test_an_fmod_event_keeps_its_directories(self):
        self.assertEqual(
            _event_audio_path("event:/sfx/block_gain", Path("/a")),
            Path("/a/sfx/block_gain.webm"),
        )

    def test_a_debug_audio_path_drops_res_and_the_source_extension(self):
        # The encoder transcodes mp3 and wav alike to .webm, so the source
        # extension must not survive into the shipped path.
        self.assertEqual(
            _event_audio_path("res://debug_audio/victory.mp3", Path("/a")),
            Path("/a/debug_audio/victory.webm"),
        )

    def test_tmp_sfx_path_matches_the_games_own_resolver(self):
        # `TmpSfx.GetPath` is what turns the bare filename every call site
        # passes into a res:// path.
        self.assertEqual(tmp_sfx_path("victory.mp3"), "res://debug_audio/victory.mp3")


class TestLoadOrbSfx(unittest.TestCase):
    """The orb table is read from the CALL GRAPH, not from the file tree.

    `OrbModel.Play<Slot>Sfx` prefers the class's FMOD override and otherwise
    falls back to `res://debug_audio/<slug>_<slot>.mp3`, and several orbs never
    call one of the three at all. Both facts are invisible in
    `site/public/audio/`: `frost_orb_channel.mp3` and `glass_orb_evoke.mp3`
    both ship and neither is ever played.
    """

    def _tree(self, root: Path, files: dict[str, str]) -> Path:
        orbs = root / "MegaCrit.Sts2.Core.Models.Orbs"
        orbs.mkdir(parents=True)
        for name, body in files.items():
            (orbs / name).write_text(body, encoding="utf-8")
        return root

    def test_an_fmod_override_wins_over_the_debug_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(
                Path(tmp),
                {
                    "FrostOrb.cs": (
                        "protected override string ChannelSfx => "
                        '"event:/sfx/characters/defect/defect_frost_channel";\n'
                        "PlayEvokeSfx();\n"
                    )
                },
            )
            out = load_orb_sfx(root)
            self.assertEqual(
                out["Frost"]["channel"], "event:/sfx/characters/defect/defect_frost_channel"
            )

    def test_a_slot_with_no_override_falls_back_to_the_debug_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), {"FrostOrb.cs": "PlayEvokeSfx();\n"})
            self.assertEqual(
                load_orb_sfx(root)["Frost"]["evoke"], "res://debug_audio/frost_orb_evoke.mp3"
            )

    def test_a_slot_the_class_never_calls_is_absent(self):
        # `GlassOrb` calls only `PlayPassiveSfx()`, so it has no evoke sound —
        # even though `glass_orb_evoke.mp3` ships. Absent beats wrong.
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), {"GlassOrb.cs": "PlayPassiveSfx();\n"})
            entry = load_orb_sfx(root)["Glass"]
            self.assertIn("passive", entry)
            self.assertNotIn("evoke", entry)

    def test_channel_is_credited_to_every_orb(self):
        # `PlayChannelSfx` is public and called once, for every orb, from
        # `OrbCmd.cs:87` — never from the orb class itself.
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), {"PlasmaOrb.cs": "PlayEvokeSfx();\n"})
            self.assertEqual(
                load_orb_sfx(root)["Plasma"]["channel"], "res://debug_audio/plasma_orb_channel.mp3"
            )

    def test_the_key_is_the_wire_type_not_the_class_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), {"LightningOrb.cs": "PlayEvokeSfx();\n"})
            self.assertEqual(sorted(load_orb_sfx(root)), ["Lightning"])

    @needs_decomp
    def test_the_real_decomp_matches_the_hand_reading(self):
        out = load_orb_sfx(_REPO_DECOMP_ROOT)
        self.assertEqual(sorted(out), ["Dark", "Frost", "Glass", "Lightning", "Plasma"])
        # Lightning overrides all three with FMOD but only ever evokes.
        self.assertEqual(sorted(out["Lightning"]), ["channel", "evoke"])
        self.assertTrue(out["Lightning"]["evoke"].startswith("event:/"))
        # Glass channels and passives, and never evokes.
        self.assertEqual(sorted(out["Glass"]), ["channel", "passive"])


class TestLoadTreasureSfx(unittest.TestCase):
    def test_keys_by_act_number_read_out_of_the_event_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acts = root / "MegaCrit.Sts2.Core.Models.Acts"
            acts.mkdir(parents=True)
            (acts / "Hive.cs").write_text(
                'public override string ChestOpenSfx => "event:/sfx/ui/treasure/treasure_act2";'
            )
            self.assertEqual(load_treasure_sfx(root), {"2": "event:/sfx/ui/treasure/treasure_act2"})

    @needs_decomp
    def test_the_two_act_one_variants_agree_rather_than_conflict(self):
        # Overgrowth and Underdocks are both act 1 and both point at
        # `treasure_act1`, so the duplicate key is expected, not a clash.
        out = load_treasure_sfx(_REPO_DECOMP_ROOT)
        self.assertEqual(sorted(out), ["1", "2", "3"])


class TestCardSignatureEvent(unittest.TestCase):
    """A card's own sound is often in its VFX NODE, not in the card.

    `Hellraiser.cs` holds no event literal at all; `NHellraiserVfx` plays the
    sound. Scanning card sources alone found 9 of them.
    """

    VFX = {"NHellraiserVfx": 'SfxCmd.Play("event:/sfx/characters/ironclad/ironclad_hellraiser");'}

    def test_follows_the_cards_vfx_node(self):
        card = "NHellraiserVfx.Create(base.Owner.Creature)"
        self.assertEqual(
            card_signature_event(card, self.VFX),
            "event:/sfx/characters/ironclad/ironclad_hellraiser",
        )

    def test_the_cards_own_literal_wins(self):
        card = (
            'SfxCmd.Play("event:/sfx/characters/ironclad/ironclad_whirlwind");\n'
            "NHellraiserVfx.Create(x)"
        )
        self.assertEqual(
            card_signature_event(card, self.VFX),
            "event:/sfx/characters/ironclad/ironclad_whirlwind",
        )

    def test_a_silent_vfx_node_does_not_block_a_later_one(self):
        vfx = {"NHitSparkVfx": "no events here", **self.VFX}
        card = "NHitSparkVfx.Create(x); NHellraiserVfx.Create(y)"
        self.assertEqual(
            card_signature_event(card, vfx),
            "event:/sfx/characters/ironclad/ironclad_hellraiser",
        )

    def test_a_card_with_neither_stays_silent(self):
        self.assertIsNone(card_signature_event("await BlockCmd.Gain(5);", self.VFX))
