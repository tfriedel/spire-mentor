"""Seeded scenario generator for the parity-fuzz harness (Task 6).

Generates deterministic `Scenario` instances suitable for driving both the
C# headless and the Rust engine via Task 7/8's scenario builders.

All IDs here are canonical engine names confirmed against the Rust content
crate (which mirrors the C# v0.103.2 ids):

- IC_CARDS: from
  ../sts2-rs/crates/sts2-content/src/cards/ironclad_specs_generated.rs
- ACT1_ENCOUNTERS: single-monster Act-1 encounter ids from
  ../sts2-rs/crates/sts2-content/src/encounters/act1_overgrowth.rs
  (multi-monster groups dropped from v1 to keep the trial driver simple;
  Vantom boss included).
- SAFE_RELICS: just IC's starter `BURNING_BLOOD` for v1 — its only
  in-combat effect is a rest-heal hook, so it can't perturb parity.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

from card_inventory import _version_tuple, has_v0111_cards

# Canonical Ironclad card ids. Cross-checked against the Rust generated
# specs in sts2-content/src/cards/ironclad_specs_generated.rs. STRIKE
# resolves as STRIKE_IRONCLAD (T1 probe confirmed bare "STRIKE" fails).
IC_CARDS: list[str] = [
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "BASH",
    "ANGER",
    "ARMAMENTS",
    "BODY_SLAM",
    "HEADBUTT",
    "IRON_WAVE",
    "PERFECTED_STRIKE",
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
    "SWORD_BOOMERANG",
    "THUNDERCLAP",
    "TRUE_GRIT",
    "TWIN_STRIKE",
    "INFLAME",
    "HEMOKINESIS",
    "UPPERCUT",
]

# Combat-hook relics for parity coverage (v2, 2026-05-29). Curated to
# relics whose effect is an *at-combat-start* or *in-combat reactive*
# hook (not on-obtain — build_scenario_state pushes into state.relics
# without dispatching on-obtain, but DOES run start_combat afterwards,
# and the C# headless fires OnCombatStart on enter_room, so both engines
# exercise the start/reactive hooks symmetrically). Ids are the canonical
# UPPER_SNAKE form shared by Rust (relic_kind_from_python_name) and the
# C# ModelId "RELIC" entry. Deliberately mixes definitely-implemented
# relics with a few whose Rust hook may be missing — surfacing that gap
# is the point of the audit. Relics with init-id mismatches bucket as
# HARNESS_INIT and are skipped, not silently dropped.
SAFE_RELICS: list[str] = [
    "BURNING_BLOOD",  # IC starter (rest-heal only; parity baseline)
    "AKABEKO",  # first attack each combat +8 damage
    "ANCHOR",  # 10 block on turn 1
    "BAG_OF_MARBLES",  # apply Vulnerable 1 to all enemies at combat start
    "BAG_OF_PREPARATION",  # draw 2 extra cards on turn 1
    "BLOOD_VIAL",  # heal 2 at combat start
    "BRONZE_SCALES",  # 3 Thorns at combat start
    "VAJRA",  # +1 Strength at combat start
    "ODDLY_SMOOTH_STONE",  # +1 Dexterity at combat start
    "PEN_NIB",  # every 10th attack deals double damage
    "KUSARIGAMA",  # attack-count trigger
    "NUNCHAKU",  # every 10th attack played: +1 energy
    "ORICHALCUM",  # 6 block if you end your turn with no block
    "CENTENNIAL_PUZZLE",  # draw 3 the first time you lose HP each combat
    "LETTER_OPENER",  # every 3rd Skill: 5 damage to all enemies
    "HORN_CLEAT",  # 14 block on turn 2
    "MERCURY_HOURGLASS",  # 3 damage to all enemies at start of turn
    "MINIATURE_CANNON",  # 6 damage to all enemies at combat start
]

# Deterministic, Ironclad-usable potions for parity coverage (v2). RNG
# potions (ATTACK/SKILL/POWER/COLORLESS, DISTILLED_CHAOS, SNECKO_OIL,
# ENTROPIC_BREW, GAMBLERS_BREW, CLARITY, ...) are excluded — their effect
# branches on RNG and would produce draw/generation noise rather than a
# clean effect-parity signal. Each id is recognized by both Rust
# (potion_kind_from_python_name) and the C# ModelId "POTION" entry.
SAFE_POTIONS: list[str] = [
    "BLOCK_POTION",  # 12 block (self)
    "FIRE_POTION",  # 20 damage (targeted)
    "STRENGTH_POTION",  # +2 Strength (self)
    "DEXTERITY_POTION",  # +2 Dexterity (self)
    "ENERGY_POTION",  # +2 energy (self)
    "WEAK_POTION",  # Weak 3 (targeted)
    "VULNERABLE_POTION",  # Vulnerable 3 (targeted)
    "FLEX_POTION",  # +5 Strength this turn (self)
    "BLOOD_POTION",  # heal 10% max HP (self)
    "SWIFT_POTION",  # draw 3 cards (self)
]

# Single-monster Act-1 encounter ids. From the act1_overgrowth registry.
# Multi-monster groups (SLIMES_*, NIBBITS_*, RUBY_RAIDERS_NORMAL,
# OVERGROWTH_CRAWLERS_NORMAL, INKLETS_NORMAL) intentionally excluded for
# v1 to keep state-diff simpler. VANTOM_BOSS included.
ACT1_ENCOUNTERS: list[str] = [
    "SHRINKER_BEETLE_WEAK",
    "FUZZY_WURM_CRAWLER_WEAK",
    "FLYCONID_NORMAL",
    "FOGMOG_NORMAL",
    "MAWLER_NORMAL",
    "SLITHERING_STRANGLER_NORMAL",
    "SNAPPING_JAXFRUIT_NORMAL",
    "VINE_SHAMBLER_NORMAL",
    "BYGONE_EFFIGY_ELITE",
    "BYRDONIS_ELITE",
    "PHROG_PARASITE_ELITE",
    "VANTOM_BOSS",
]


@dataclass(frozen=True)
class Scenario:
    """A single fuzz trial input.

    `draw_order` is optional — when None, the engine's RNG-driven draw
    order is used. When set, the harness should force the draw pile to
    match this sequence (top-of-pile first).

    `coverage_subject` labels the *one* element a coverage trial varies
    (``"card:WHIRLWIND"`` / ``"relic:AKABEKO"`` / ``"potion:FIRE_POTION"``
    / ``"encounter:MAWLER_NORMAL"``). The audit driver buckets by this
    label in coverage mode, so a divergence is attributed to the varied
    element regardless of which action (init / play_card / use_potion /
    end_turn) actually tripped the diff. None for random `generate()`
    scenarios.
    """

    character: str
    deck: tuple[str, ...]
    hp: int
    max_hp: int
    relics: tuple[str, ...]
    encounter: str
    potions: tuple[str, ...] = field(default=())
    draw_order: tuple[str, ...] | None = field(default=None)
    coverage_subject: str | None = field(default=None)


def generate(seed: int) -> Scenario:
    """Deterministically synthesize a Scenario from `seed`.

    Uses a private `random.Random` so we don't perturb the global RNG.
    """
    rng = random.Random(seed)

    # Deck: 8–15 cards drawn with replacement from IC_CARDS.
    deck_size = rng.randint(8, 15)
    deck = tuple(rng.choice(IC_CARDS) for _ in range(deck_size))

    # Relics: 0–3 unique relics from SAFE_RELICS.
    relic_count = rng.randint(0, min(3, len(SAFE_RELICS)))
    relics = tuple(rng.sample(SAFE_RELICS, relic_count))

    # Potions: 0–2 unique potions from SAFE_POTIONS (belt cap is 5 — well
    # within 2). Held at combat start; the harness picks use_potion from
    # the legal-action set, so they get exercised mid-combat.
    potion_count = rng.randint(0, min(2, len(SAFE_POTIONS)))
    potions = tuple(rng.sample(SAFE_POTIONS, potion_count))

    # HP: max in [60, 80], current in [20, max].
    max_hp = rng.randint(60, 80)
    hp = rng.randint(20, max_hp)

    encounter = rng.choice(ACT1_ENCOUNTERS)

    return Scenario(
        character="Ironclad",
        deck=deck,
        hp=hp,
        max_hp=max_hp,
        relics=relics,
        encounter=encounter,
        potions=potions,
        draw_order=None,
    )


def generate_v2(seed: int, character: str = "Ironclad") -> Scenario:
    """Axis C (sweep v2): full-pool randomized scenario.

    Differences vs v1 ``generate``: the deck draws from the FULL card
    pools (the character's class cards + 64 colorless, not the 18-card
    curated subset) with a 25% per-card upgrade chance; relics come from
    ALL_RELICS (v1 used only the curated SAFE_RELICS); potions from
    ALL_POTIONS; the encounter from all three acts including multi-monster
    groups. Deterministic per (seed, character).

    ``character`` defaults to ``"Ironclad"`` and, for that value, produces
    the exact scenario the Ironclad-only generator did (same rng stream,
    same pool) so the committed Ironclad trace cache stays valid.

    A "+1" on a non-upgradable card (curse/status — none currently in
    the pools) is a silent no-op on both backends.
    """
    rng = random.Random(seed)
    pool = _v2_card_pool(character)
    deck = tuple(
        c + ("+1" if rng.random() < 0.25 else "")
        for c in (rng.choice(pool) for _ in range(rng.randint(8, 20)))
    )
    relics = tuple(
        rng.sample(ALL_RELICS, rng.randint(0, 4))
    )  # 4 << len(ALL_RELICS); no v1-style min() guard needed
    potions = tuple(rng.sample(ALL_POTIONS, rng.randint(0, 3)))
    max_hp = rng.randint(60, 80)
    hp = rng.randint(20, max_hp)
    encounter = rng.choice(ALL_ENCOUNTERS)
    return Scenario(
        character=character,
        deck=deck,
        hp=hp,
        max_hp=max_hp,
        relics=relics,
        encounter=encounter,
        potions=potions,
        draw_order=None,
    )


@dataclass(frozen=True)
class Loadout:
    """A random deck, relic and potion set to graft onto a whole run."""

    deck: tuple[str, ...]
    relics: tuple[str, ...]
    potions: tuple[str, ...]
    gold: int


def generate_loadout(
    seed: str, character: str = "Ironclad", game_version: str | None = None
) -> Loadout:
    """The ``generate_v2`` pools as a whole-run start, not a single fight.

    A run outlives one fight, so it carries more relics than a scenario does
    (rewards, shops and rest sites are what the graft exists to exercise), and
    gold, so shops get bought from. Deterministic per
    ``(seed, character, game_version)``. The relics skip ``EVENT_RELICS``:
    an event can grant those a second time.

    Lives apart from ``generate_v2`` because that function's rng stream keys
    the committed trace cache.
    """
    rng = random.Random(f"loadout:{seed}:{character}")
    pool = _v2_card_pool(character, game_version)
    return Loadout(
        deck=tuple(
            c + ("+1" if rng.random() < 0.25 else "")
            for c in (rng.choice(pool) for _ in range(rng.randint(8, 25)))
        ),
        relics=tuple(rng.sample(LOADOUT_RELICS, rng.randint(3, 12))),
        potions=tuple(rng.sample(ALL_POTIONS, rng.randint(0, 3))),
        gold=rng.randint(0, 999),
    )


# ---------------------------------------------------------------------------
# Exhaustive coverage sweep (2026-05-30)
# ---------------------------------------------------------------------------
#
# The random `generate()` above samples curated subsets to find *interaction*
# bugs. The coverage sweep below does the opposite: it guarantees every
# Ironclad card, colorless card, relic, potion, and Act-1 encounter is
# exercised at least once, with each trial varying exactly ONE element
# against a fixed clean baseline so a divergence is trivially attributable.
# The sweep is *additive* (|cards| + |relics| + |potions| + |encounters|),
# never a cartesian product.
#
# The id inventories are extracted verbatim from the authoritative Rust
# content crate (which mirrors the C# v0.103.2 ids) — regenerate them when
# content lands:
#   ALL_IC_CARDS        cards/ironclad_specs_generated.rs   (name: "...")
#   ALL_COLORLESS_CARDS cards/colorless_specs_generated.rs  (name: "...")
#   ALL_RELICS          sts2-py/src/state_dict.rs RELIC_KINDS → RelicKind::python_name()
#   ALL_POTIONS         potions/all.rs                       (spec(kind, "...", ...))
#   ALL_ENCOUNTERS      encounters/act1_{overgrowth,underdocks}.rs,
#                       encounters/act{2_hive,3_glory}.rs    (id: "...")

ALL_IC_CARDS: list[str] = [
    "AGGRESSION_CARD",
    "ANGER",
    "ARMAMENTS",
    "ASHEN_STRIKE",
    "BARRICADE_CARD",
    "BASH",
    "BATTLE_TRANCE",
    "BLOODLETTING",
    "BLOOD_WALL",
    "BLUDGEON",
    "BODY_SLAM",
    "BRAND",
    "BREAK",
    "BREAKTHROUGH",
    "BULLY",
    "BURNING_PACT",
    "CASCADE",
    "CINDER",
    "COLOSSUS_CARD",
    "CONFLAGRATION",
    "CORRUPTION_CARD",
    "CRIMSON_MANTLE",
    "CRUELTY_CARD",
    "DARK_EMBRACE_CARD",
    "DEFEND_IRONCLAD",
    "DEMON_FORM_CARD",
    "DEMONIC_SHIELD",
    "DISMANTLE",
    "DOMINATE",
    "DRUM_OF_BATTLE_CARD",
    "EVIL_EYE",
    "EXPECT_A_FIGHT",
    "FEED",
    "FEEL_NO_PAIN_CARD",
    "FIEND_FIRE",
    "FIGHT_ME",
    "FLAME_BARRIER_CARD",
    "FORGOTTEN_RITUAL",
    "HAVOC",
    "HEADBUTT",
    "HELLRAISER_CARD",
    "HEMOKINESIS",
    "HOWL_FROM_BEYOND",
    "IMPERVIOUS",
    "INFERNAL_BLADE",
    "INFERNO_CARD",
    "INFLAME",
    "IRON_WAVE",
    "JUGGERNAUT_CARD",
    "JUGGLING_CARD",
    "MANGLE",
    "MOLTEN_FIST",
    "NOT_YET",
    "OFFERING",
    "ONE_TWO_PUNCH_CARD",
    "PACTS_END",
    "PERFECTED_STRIKE",
    "PILLAGE",
    "POMMEL_STRIKE",
    "PRIMAL_FORCE",
    "PYRE",
    "RAGE_CARD",
    "RAMPAGE",
    "RUPTURE_CARD",
    "SECOND_WIND",
    "SETUP_STRIKE_CARD",
    "SHRUG_IT_OFF",
    "SPITE",
    "STAMPEDE_CARD",
    "STOKE",
    "STOMP",
    "STONE_ARMOR",
    "STRIKE_IRONCLAD",
    "SWORD_BOOMERANG",
    "TANK_CARD",
    "TAUNT",
    "TEAR_ASUNDER",
    "THRASH",
    "THUNDERCLAP",
    "TREMBLE",
    "TRUE_GRIT",
    "TWIN_STRIKE",
    "UNMOVABLE",
    "UNRELENTING",
    "UPPERCUT",
    "VICIOUS_CARD",
    "WHIRLWIND",
]

ALL_COLORLESS_CARDS: list[str] = [
    "ALCHEMIZE",
    "ANOINTED",
    "AUTOMATION",
    "BEACON_OF_HOPE",
    "BEAT_DOWN",
    "BELIEVE_IN_YOU",
    "BOLAS",
    "CALAMITY_CARD",
    "CATASTROPHE",
    "COORDINATE_CARD",
    "DARK_SHACKLES",
    "DISCOVERY",
    "DRAMATIC_ENTRANCE",
    "ENTROPY",
    "EQUILIBRIUM",
    "ETERNAL_ARMOR",
    "FASTEN",
    "FINESSE",
    "FISTICUFFS",
    "FLASH_OF_STEEL",
    "GANG_UP",
    "GOLD_AXE",
    "HAND_OF_GREED",
    "HIDDEN_GEM",
    "HUDDLE_UP",
    "IMPATIENCE",
    "INTERCEPT_CARD",
    "JACK_OF_ALL_TRADES",
    "JACKPOT",
    "KNOCKDOWN",
    "LIFT",
    "MASTER_OF_STRATEGY",
    "MAYHEM_CARD",
    "MIMIC",
    "MIND_BLAST",
    "NOSTALGIA_CARD",
    "OMNISLICE",
    "PANACHE_CARD",
    "PANIC_BUTTON",
    "PREP_TIME",
    "PRODUCTION",
    "PROLONG",
    "PROWESS",
    "PURITY",
    "RALLY",
    "REND",
    "RESTLESSNESS",
    "ROLLING_BOULDER",
    "SALVO",
    "SCRAWL",
    "SECRET_TECHNIQUE",
    "SECRET_WEAPON",
    "SEEKER_STRIKE",
    "SHOCKWAVE",
    "SPLASH",
    "STRATAGEM",
    "TAG_TEAM",
    "THE_BOMB_CARD",
    "THE_GAMBIT",
    "THINKING_AHEAD",
    "THRUMMING_HATCHET",
    "ULTIMATE_DEFEND",
    "ULTIMATE_STRIKE",
    "VOLLEY",
]

#: Event- and Ancient-rarity cards — the blind spot the hand-transcribed lists
#: had from the start.
#:
#: These live in `status_specs_generated.rs`, not
#: `colorless_specs_generated.rs`, so the transcription that produced
#: ALL_COLORLESS_CARDS never saw them and NONE of them had ever been run
#: against the C# oracle. That was found the expensive way: one card at a time
#: while porting HelloWorldPower (GH #636) and ReboundPower (GH #637), the
#: second of which had a real engine bug the sweep caught the moment its card
#: was finally included.
#:
#: Character-agnostic (any character can pick these up from an event), so they
#: append to every pack the way ALL_COLORLESS_CARDS does.
#:
#: `tests/test_parity_coverage.py::test_every_playable_engine_card_is_covered`
#: now derives the expected membership from the Rust source via
#: `card_inventory`, so a future card cannot go missing the same way.
ALL_EVENT_CARDS: list[str] = [
    "APOTHEOSIS",
    "APPARITION",
    "BRIGHTEST_FLAME",
    "BYRD_SWOOP",
    "CALTROPS",
    "CLASH",
    "DISTRACTION",
    "DUAL_WIELD",
    "ENLIGHTENMENT",
    "ENTRENCH",
    "EXTERMINATE",
    "FEEDING_FRENZY_CARD",
    "HELLO_WORLD_CARD",
    "MAD_SCIENCE",
    "MAUL",
    "METAMORPHOSIS",
    "NEOWS_FURY",
    "OUTMANEUVER",
    "PECK",
    "REBOUND",
    "RELAX",
    "RIP_AND_TEAR",
    "SQUASH",
    "STACK",
    "TORIC_TOUGHNESS",
    "WHISTLE",
    "WISH",
]


ALL_RELICS: list[str] = [
    "AKABEKO",
    "AMETHYST_AUBERGINE",
    "ANCHOR",
    "BAG_OF_MARBLES",
    "BAG_OF_PREPARATION",
    "BELLOWS",
    "BIG_MUSHROOM",
    "BING_BONG",
    "BLOOD_VIAL",
    "BONE_FLUTE",
    "BONE_TEA",
    "BOOK_OF_FIVE_RINGS",
    "BOOK_REPAIR_KNIFE",
    "BOUND_PHYLACTERY",
    "BOWLER_HAT",
    "BRONZE_SCALES",
    "BURNING_BLOOD",
    "BYRDPIP",
    "CANDELABRA",
    "CENTENNIAL_PUZZLE",
    "CHOSEN_CHEESE",
    "CRACKED_CORE",
    "DARKSTONE_PERIAPT",
    "DATA_DISK",
    "DAUGHTER_OF_THE_WIND",
    "DIVINE_RIGHT",
    "DREAM_CATCHER",
    "EMBER_TEA",
    "ETERNAL_FEATHER",
    "FAKE_ANCHOR",
    "FAKE_BLOOD_VIAL",
    "FAKE_HAPPY_FLOWER",
    "FAKE_LEES_WAFFLE",
    "FAKE_MANGO",
    "FAKE_MERCHANTS_RUG",
    "FAKE_ORICHALCUM",
    "FAKE_SNECKO_EYE",
    "FAKE_STRIKE_DUMMY",
    "FAKE_VENERABLE_TEA_SET",
    "FENCING_MANUAL",
    "FESTIVE_POPPER",
    "FORGOTTEN_SOUL",
    "FRAGRANT_MUSHROOM",
    "FRESNEL_LENS",
    "FUNERARY_MASK",
    "GALACTIC_DUST",
    "GOLD_PLATED_CABLES",
    "GORGET",
    "GREMLIN_HORN",
    "HAND_DRILL",
    "HAPPY_FLOWER",
    "HISTORY_COURSE",
    "HORN_CLEAT",
    "JOSS_PAPER",
    "JUZU_BRACELET",
    "KUSARIGAMA",
    "LANTERN",
    "LETTER_OPENER",
    "LOST_WISP",
    "LUCKY_FYSH",
    "MAW_BANK",
    "MEAL_TICKET",
    "MERCURY_HOURGLASS",
    "MINIATURE_CANNON",
    "MR_STRUGGLES",
    "NUNCHAKU",
    "ODDLY_SMOOTH_STONE",
    "ORICHALCUM",
    "ORNAMENTAL_FAN",
    "PANTOGRAPH",
    "PAPER_PHROG",
    "PARRYING_SHIELD",
    "PEAR",
    "PENDULUM",
    "PEN_NIB",
    "PERMAFROST",
    "PETRIFIED_TOAD",
    "PLANISPHERE",
    "POLLINOUS_CORE",
    "POTION_BELT",
    "RED_MASK",
    "RED_SKULL",
    "REGALITE",
    "REGAL_PILLOW",
    "REPTILE_TRINKET",
    "RING_OF_THE_SNAKE",
    "RIPPLE_BASIN",
    "ROYAL_POISON",
    "SELF_FORMING_CLAY",
    "SNECKO_SKULL",
    "SPARKLING_ROUGE",
    "STONE_CRACKER",
    "STRAWBERRY",
    "STRIKE_DUMMY",
    "SWORD_OF_JADE",
    "SWORD_OF_STONE",
    "SYMBIOTIC_VIRUS",
    "TEA_OF_DISCOURTESY",
    "THE_BOOT",
    "TINGSHA",
    "TINY_MAILBOX",
    "TUNING_FORK",
    "TWISTED_FUNNEL",
    "VAJRA",
    "VAMBRACE",
    "VENERABLE_TEA_SET",
    "WAR_PAINT",
    "WHETSTONE",
    "WONGOS_MYSTERY_TICKET",
    "WONGO_CUSTOMER_APPRECIATION_BADGE",
]

#: The Event-rarity relics in ``ALL_RELICS`` (``rarity`` in
#: ``sts2-wiki/data/<version>/relics.json``; the same 35 in v0.107.1 and
#: v0.111.0). An event hands each of them out by name, not from a grab bag, so
#: a run grafted with one can meet its event and be given a second copy. The
#: real game can't hold two: events don't repeat and every other source pulls
#: from the grab bags, which drop owned relics. C# fires hooks once per copy
#: and Rust once per ``RelicId``, so the copy shows up as a false divergence.
#: ``generate_loadout`` leaves them out. (``ALL_RELICS`` has no Ancient ones.)
EVENT_RELICS: frozenset[str] = frozenset(
    {
        "BIG_MUSHROOM",
        "BING_BONG",
        "BONE_TEA",
        "BYRDPIP",
        "CHOSEN_CHEESE",
        "DARKSTONE_PERIAPT",
        "DAUGHTER_OF_THE_WIND",
        "DREAM_CATCHER",
        "EMBER_TEA",
        "FAKE_ANCHOR",
        "FAKE_BLOOD_VIAL",
        "FAKE_HAPPY_FLOWER",
        "FAKE_LEES_WAFFLE",
        "FAKE_MANGO",
        "FAKE_MERCHANTS_RUG",
        "FAKE_ORICHALCUM",
        "FAKE_SNECKO_EYE",
        "FAKE_STRIKE_DUMMY",
        "FAKE_VENERABLE_TEA_SET",
        "FORGOTTEN_SOUL",
        "FRAGRANT_MUSHROOM",
        "FRESNEL_LENS",
        "HAND_DRILL",
        "HISTORY_COURSE",
        "LOST_WISP",
        "MAW_BANK",
        "MR_STRUGGLES",
        "POLLINOUS_CORE",
        "ROYAL_POISON",
        "SWORD_OF_JADE",
        "SWORD_OF_STONE",
        "TEA_OF_DISCOURTESY",
        "THE_BOOT",
        "WONGOS_MYSTERY_TICKET",
        "WONGO_CUSTOMER_APPRECIATION_BADGE",
    }
)

#: The relics ``generate_loadout`` draws from.
LOADOUT_RELICS: list[str] = [r for r in ALL_RELICS if r not in EVENT_RELICS]

ALL_POTIONS: list[str] = [
    "ATTACK_POTION",
    "BEETLE_JUICE",
    "BLESSING_OF_THE_FORGE",
    "BLOCK_POTION",
    "BOTTLED_POTENTIAL",
    "CLARITY",
    "COLORLESS_POTION",
    "CURE_ALL",
    "DEXTERITY_POTION",
    "DISTILLED_CHAOS",
    "DROPLET_OF_PRECOGNITION",
    "DUPLICATOR",
    "ENERGY_POTION",
    "ENTROPIC_BREW",
    "EXPLOSIVE_AMPOULE",
    "FAIRY_IN_A_BOTTLE",
    "FIRE_POTION",
    "FLEX_POTION",
    "FORTIFIER",
    "FOUL_POTION",
    "FRUIT_JUICE",
    "FYSH_OIL",
    "GAMBLERS_BREW",
    "GIGANTIFICATION_POTION",
    "GLOWWATER_POTION",
    "HEART_OF_IRON",
    "LIQUID_BRONZE",
    "LIQUID_MEMORIES",
    "LUCKY_TONIC",
    "MAZALETHS_GIFT",
    "OROBIC_ACID",
    "POTION_OF_BINDING",
    "POTION_SHAPED_ROCK",
    "POWDERED_DEMISE",
    "POWER_POTION",
    "RADIANT_TINCTURE",
    "REGEN_POTION",
    "SHACKLING_POTION",
    "SHIP_IN_A_BOTTLE",
    "SKILL_POTION",
    "SNECKO_OIL",
    "SPEED_POTION",
    "STABLE_SERUM",
    "STRENGTH_POTION",
    "SWIFT_POTION",
    "TOUCH_OF_INSANITY",
    "VULNERABLE_POTION",
    "WEAK_POTION",
]

#: Canonical Regent (orange) card ids — the `name:` field of every factory in
#: ../sts2-rs/crates/sts2-content/src/cards/regent/*.rs (REGENT_SPECS, 88
#: cards, mirrors C# RegentCardPool.GenerateAllCards). These were ported from
#: the old Python sim (sts2-rl-agent) and have NEVER been A/B'd card-by-card
#: against the C# v0.103.2 oracle — that's exactly what the Regent coverage
#: pack below exercises. Regenerate with:
#:   grep -rhoP 'name:\s*"\K[A-Z0-9_]+' \
#:     ../sts2-rs/crates/sts2-content/src/cards/regent/*.rs | sort -u
ALL_REGENT_CARDS: list[str] = [
    "ALIGNMENT",
    "ARSENAL",
    "ASTRAL_PULSE",
    "BEAT_INTO_SHAPE",
    "BEGONE",
    "BIG_BANG",
    "BLACK_HOLE",
    "BOMBARDMENT",
    "BULWARK",
    "BUNDLE_OF_JOY",
    "CELESTIAL_MIGHT",
    "CHARGE",
    "CHILD_OF_THE_STARS",
    "CLOAK_OF_STARS",
    "COLLISION_COURSE",
    "COMET",
    "CONQUEROR",
    "CONVERGENCE",
    "COSMIC_INDIFFERENCE",
    "CRASH_LANDING",
    "CRESCENT_SPEAR",
    "CRUSH_UNDER",
    "DECISIONS_DECISIONS",
    "DEFEND_REGENT",
    "DEVASTATE",
    "DYING_STAR",
    "FALLING_STAR",
    "FOREGONE_CONCLUSION",
    "FURNACE",
    "GAMMA_BLAST",
    "GATHER_LIGHT",
    "GENESIS",
    "GLIMMER",
    "GLITTERSTREAM",
    "GLOW",
    "GUARDS",
    "GUIDING_STAR",
    "HAMMER_TIME",
    "HEAVENLY_DRILL",
    "HEGEMONY",
    "HEIRLOOM_HAMMER",
    "HIDDEN_CACHE",
    "I_AM_INVINCIBLE",
    "KINGLY_KICK",
    "KINGLY_PUNCH",
    "KNOCKOUT_BLOW",
    "KNOW_THY_PLACE",
    "LARGESSE",
    "LUNAR_BLAST",
    "MAKE_IT_SO",
    "MANIFEST_AUTHORITY",
    "METEOR_SHOWER",
    "MONARCHS_GAZE_CARD",
    "MONOLOGUE_CARD",
    "NEUTRON_AEGIS",
    "ORBIT",
    "PALE_BLUE_DOT",
    "PARRY_CARD",
    "PARTICLE_WALL",
    "PATTER",
    "PHOTON_CUT",
    "PILLAR_OF_CREATION",
    "PROPHESIZE",
    "QUASAR",
    "RADIATE",
    "REFINE_BLADE",
    "REFLECT_CARD",
    "RESONANCE",
    "ROYAL_GAMBLE",
    "ROYALTIES_CARD",
    "SEEKING_EDGE",
    "SEVEN_STARS",
    "SHINING_STRIKE",
    "SOLAR_STRIKE",
    "SPECTRUM_SHIFT",
    "SPOILS_OF_BATTLE",
    "STARDUST",
    "STRIKE_REGENT",
    "SUMMON_FORTH",
    "SUPERMASSIVE",
    "SWORD_SAGE",
    "TERRAFORMING",
    "THE_SEALED_THRONE",
    "THE_SMITH",
    "TYRANNY_CARD",
    "VENERATE",
    "VOID_FORM",
    "WROUGHT_IN_WAR",
]

#: Regent's class relic pool (grab_bag.rs REGENT_{COMMON,UNCOMMON,RARE,SHOP}).
#: python_name strings from sts2-core/src/relics.rs.
ALL_REGENT_RELICS: list[str] = [
    "FENCING_MANUAL",  # common
    "GALACTIC_DUST",  # uncommon
    "REGALITE",  # uncommon
    "LUNAR_PASTRY",  # rare
    "MINI_REGENT",  # rare
    "ORANGE_DOUGH",  # rare
    "VITRUVIAN_MINION",  # shop
]

#: Canonical Silent card ids — the `name:` field of each CardSpec under
#: crates/sts2-content/src/cards/silent/. The Silent coverage pack A/Bs every
#: one against the C# v0.103.2 oracle (clean Slithering Strangler arena).
ALL_SILENT_CARDS: list[str] = [
    "ABRASIVE",
    "ACCELERANT",
    "ACCURACY_CARD",
    "ACROBATICS",
    "ADRENALINE",
    "AFTERIMAGE_CARD",
    "ANTICIPATE",
    "ASSASSINATE",
    "BACKFLIP",
    "BACKSTAB",
    "BLADE_DANCE",
    "BLADE_OF_INK",
    "BLUR_CARD",
    "BOUNCING_FLASK",
    "BUBBLE_BUBBLE",
    "BULLET_TIME",
    "BURST",
    "CALCULATED_GAMBLE",
    "CLOAK_AND_DAGGER",
    "CORROSIVE_WAVE",
    "DAGGER_SPRAY",
    "DAGGER_THROW",
    "DASH",
    "DEADLY_POISON",
    "DEFEND_SILENT",
    "DEFLECT",
    "DODGE_AND_ROLL",
    "ECHOING_SLASH",
    "ENVENOM_CARD",
    "ESCAPE_PLAN",
    "EXPERTISE",
    "EXPOSE",
    "FAN_OF_KNIVES_CARD",
    "FINISHER",
    "FLANKING",
    "FLECHETTES",
    "FLICK_FLACK",
    # v0.107.1: Scare replaces FollowThrough in this exact pool slot (C#
    # SilentCardPool keeps Scare between FlickFlack and Footwork).
    "SCARE",
    "FOOTWORK",
    "GRAND_FINALE",
    "HAND_TRICK",
    "HAZE",
    "HIDDEN_DAGGERS",
    "INFINITE_BLADES_CARD",
    "KNIFE_TRAP",
    "LEADING_STRIKE",
    "LEG_SWEEP",
    "MALAISE",
    "MASTER_PLANNER",
    "MEMENTO_MORI",
    "MIRAGE",
    "MURDER",
    "NEUTRALIZE",
    "NIGHTMARE",
    "NOXIOUS_FUMES_CARD",
    "OUTBREAK_CARD",
    "PHANTOM_BLADES_CARD",
    "PIERCING_WAIL",
    "PINPOINT",
    "POISONED_STAB",
    "POUNCE",
    "PRECISE_CUT",
    "PREDATOR",
    "PREPARED",
    "REFLEX",
    "RICOCHET",
    "SERPENT_FORM_CARD",
    "SHADOWMELD",
    "SHADOW_STEP",
    "SKEWER",
    "SLICE",
    "SNAKEBITE",
    "SNEAKY_CARD",
    "SPEEDSTER_CARD",
    "STORM_OF_STEEL",
    "STRANGLE",
    "STRIKE_SILENT",
    "SUCKER_PUNCH",
    "SUPPRESS",
    "SURVIVOR",
    "TACTICIAN",
    "THE_HUNT",
    "TOOLS_OF_THE_TRADE",
    "TRACKING",
    "UNTOUCHABLE",
    "UP_MY_SLEEVE",
    "WELL_LAID_PLANS",
    "WRAITH_FORM",
]

#: Canonical Defect card ids — the `name:` field of each CardSpec under
#: crates/sts2-content/src/cards/defect/, excluding the inert STRIKE_/DEFEND_
#: basics (used as filler). Never A/B'd against the C# oracle before.
ALL_DEFECT_CARDS: list[str] = [
    "ADAPTIVE_STRIKE",
    "ALL_FOR_ONE",
    "BALL_LIGHTNING",
    "BARRAGE",
    "BEAM_CELL",
    "BIASED_COGNITION_CARD",
    "BOOST_AWAY",
    "BOOT_SEQUENCE",
    "BUFFER_CARD",
    "BULK_UP",
    "CAPACITOR",
    "CHAOS",
    "CHARGE_BATTERY",
    "CHILL",
    "CLAW",
    "COLD_SNAP",
    "COMPACT",
    "COMPILE_DRIVER",
    "CONSUMING_SHADOW",
    "COOLANT",
    "COOLHEADED",
    "CREATIVE_AI_CARD",
    "DARKNESS_CARD",
    "DEFEND_DEFECT",
    "DEFRAGMENT",
    "DOUBLE_ENERGY",
    "DUALCAST",
    "ECHO_FORM_CARD",
    "ENERGY_SURGE",
    "FERAL",
    "FIGHT_THROUGH",
    "FLAK_CANNON",
    "FOCUSED_STRIKE_CARD",
    "FTL",
    "FUSION",
    "GENETIC_ALGORITHM",
    "GLACIER",
    "GLASSWORK",
    "GO_FOR_THE_EYES",
    "GUNK_UP",
    "HAILSTORM",
    "HELIX_DRILL",
    "HOLOGRAM",
    "HOTFIX",
    "HYPERBEAM",
    "ICE_LANCE",
    "IGNITION",
    "ITERATION_CARD",
    "LEAP",
    "LIGHTNING_ROD",
    "LOOP_CARD",
    "MACHINE_LEARNING_CARD",
    "METEOR_STRIKE",
    "MODDED",
    "MOMENTUM_STRIKE",
    "MULTI_CAST",
    "NULL_CARD",
    "OVERCLOCK",
    "QUADCAST",
    "RAINBOW",
    "REBOOT",
    "REFRACT",
    "ROCKET_PUNCH",
    "SCAVENGE",
    "SCRAPE",
    "SHADOW_SHIELD",
    "SHATTER",
    "SIGNAL_BOOST",
    "SKIM",
    "SMOKESTACK",
    "SPINNER_CARD",
    "STORM_CARD",
    "STRIKE_DEFECT",
    "SUBROUTINE",
    "SUNDER",
    "SUPERCRITICAL",
    "SWEEPING_BEAM",
    "SYNCHRONIZE",
    "SYNTHESIS",
    "TEMPEST",
    "TESLA_COIL",
    "THUNDER_CARD",
    "TRASH_TO_TREASURE",
    "TURBO",
    "UPROAR",
    "VOLTAIC",
    "WHITE_NOISE",
    "ZAP",
]

#: Canonical Necrobinder card ids — the `name:` field of each CardSpec under
#: crates/sts2-content/src/cards/necrobinder/, excluding the inert STRIKE_/
#: DEFEND_ basics (used as filler). Never A/B'd against the C# oracle before.
ALL_NECROBINDER_CARDS: list[str] = [
    "AFTERLIFE",
    "BANSHEES_CRY",
    "BLIGHT_STRIKE",
    "BODYGUARD",
    "BONE_SHARDS",
    "BORROWED_TIME",
    "BURY",
    "CALCIFY_CARD",
    "CALL_OF_THE_VOID",
    "CAPTURE_SPIRIT",
    "CLEANSE",
    "COUNTDOWN_CARD",
    "DANSE_MACABRE",
    "DEATHBRINGER",
    "DEATH_MARCH",
    "DEATHS_DOOR",
    "DEBILITATE_CARD",
    "DEFEND_NECROBINDER",
    "DEFILE",
    "DEFY",
    "DELAY",
    "DEMESNE",
    "DEVOUR_LIFE_CARD",
    "DIRGE",
    "DRAIN_POWER",
    "DREDGE",
    "EIDOLON",
    "END_OF_DAYS",
    "ENFEEBLING_TOUCH",
    "ERADICATE",
    "FEAR",
    "FETCH",
    "FLATTEN",
    "FORBIDDEN_GRIMOIRE",
    "FRIENDSHIP",
    "GLIMPSE_BEYOND",
    "GRAVEBLAST",
    "GRAVE_WARDEN",
    "HANG",
    "HAUNT",
    "HIGH_FIVE",
    "INVOKE",
    "LEGION_OF_BONE",
    "LETHALITY_CARD",
    "MELANCHOLY",
    "MISERY",
    "NECRO_MASTERY_CARD",
    "NEGATIVE_PULSE",
    "NEUROSURGE",
    "NO_ESCAPE",
    "OBLIVION",
    "PAGESTORM",
    "PARSE",
    "POKE",
    "PROTECTOR",
    "PULL_AGGRO",
    "PULL_FROM_BELOW",
    "PUTREFY",
    "RATTLE",
    "REANIMATE",
    "REAP",
    "REAPER_FORM",
    "REAVE",
    "RIGHT_HAND_HAND",
    "SACRIFICE",
    "SCOURGE",
    "SCULPTING_STRIKE",
    "SEANCE",
    "SENTRY_MODE",
    "SEVERANCE",
    "SHARED_FATE",
    "SHROUD",
    "SIC_EM",
    "SLEIGHT_OF_FLESH",
    "SNAP",
    "SOUL_STORM",
    "SOW",
    "SPIRIT_OF_ASH",
    "SPUR",
    "SQUEEZE",
    "STRIKE_NECROBINDER",
    "THE_SCYTHE",
    "TIMES_UP",
    "TRANSFIGURE",
    "UNDEATH",
    "UNLEASH",
    "VEILPIERCER",
    "WISP",
]

# Coverage-sweep encounter ids, grouped by act. Sourced from the Rust
# content registry (crates/sts2-content/src/encounters/act{1,2,3}_*.rs).
# The sweep keeps the character fixed to Ironclad but exercises EVERY act's
# encounters — an encounter whose monster roster isn't fully ported yet
# buckets as HARNESS_INIT (a "not-yet-ported" signal), which is informative.
ACT1_COVERAGE_ENCOUNTERS: list[str] = [
    "BYGONE_EFFIGY_ELITE",
    "BYRDONIS_ELITE",
    "CEREMONIAL_BEAST_BOSS",
    "CUBEX_CONSTRUCT_NORMAL",
    "FLYCONID_NORMAL",
    "FOGMOG_NORMAL",
    "FUZZY_WURM_CRAWLER_WEAK",
    "INKLETS_NORMAL",
    "MAWLER_NORMAL",
    "NIBBITS_NORMAL",
    "NIBBITS_WEAK",
    "OVERGROWTH_CRAWLERS",
    "PHROG_PARASITE_ELITE",
    "RUBY_RAIDERS_NORMAL",
    "SHRINKER_BEETLE_WEAK",
    "SLIMES_NORMAL",
    "SLIMES_WEAK",
    "SLITHERING_STRANGLER_NORMAL",
    "SNAPPING_JAXFRUIT_NORMAL",
    "THE_KIN_BOSS",
    "VANTOM_BOSS",
    "VINE_SHAMBLER_NORMAL",
]

ACT2_COVERAGE_ENCOUNTERS: list[str] = [
    "BOWLBUGS_NORMAL",
    "BOWLBUGS_WEAK",
    "CHOMPERS_NORMAL",
    "DECIMILLIPEDE_ELITE",
    "ENTOMANCER_ELITE",
    "EXOSKELETONS_NORMAL",
    "EXOSKELETONS_WEAK",
    "HUNTER_KILLER_NORMAL",
    "INFESTED_PRISMS_ELITE",
    "KAISER_CRAB_BOSS",
    "KNOWLEDGE_DEMON_BOSS",
    "LOUSE_PROGENITOR_NORMAL",
    "MYTES_NORMAL",
    "OVICOPTER_NORMAL",
    "SLUMBERING_BEETLE_NORMAL",
    "SPINY_TOAD_NORMAL",
    "THE_INSATIABLE_BOSS",
    "THE_OBSCURA_NORMAL",
    "THIEVING_HOPPER_WEAK",
    "TUNNELER_NORMAL",
    "TUNNELER_WEAK",
]

ACT3_COVERAGE_ENCOUNTERS: list[str] = [
    "AEONGLASS_BOSS",
    "AXEBOTS_NORMAL",
    "CONSTRUCT_MENAGERIE_NORMAL",
    "DEVOTED_SCULPTOR_WEAK",
    "FABRICATOR_NORMAL",
    "FROG_KNIGHT_NORMAL",
    "GLOBE_HEAD_NORMAL",
    "KNIGHTS_ELITE",
    "MECHA_KNIGHT_ELITE",
    "OWL_MAGISTRATE_NORMAL",
    "QUEEN_BOSS",
    "SCROLLS_OF_BITING_NORMAL",
    "SCROLLS_OF_BITING_WEAK",
    "SLIMED_BERSERKER_NORMAL",
    "SOUL_NEXUS_ELITE",
    "TEST_SUBJECT_BOSS",
    "THE_LOST_AND_FORGOTTEN_NORMAL",
    "TURRET_OPERATOR_WEAK",
]

#: Act 1's ALTERNATE model (`Underdocks.GenerateAllEncounters`, v0.107.1) —
#: the 20 encounters a run walks when the `act_selection` roll picks Underdocks
#: over Overgrowth (GH #544).
#:
#: These went unswept for as long as the engine only modelled one Act 1. The
#: 22 monster AIs behind them were ported and unit-tested but never compared
#: against the C# at the encounter level, because this sweep is the only thing
#: that does that and no id here appeared in it. `monsters/act1_underdocks.rs`
#: was 5200+ lines of never-cross-checked combat.
ACT1_UNDERDOCKS_COVERAGE_ENCOUNTERS: list[str] = [
    "CORPSE_SLUGS_NORMAL",
    "CORPSE_SLUGS_WEAK",
    "CULTISTS_NORMAL",
    "FOSSIL_STALKER_NORMAL",
    "GREMLIN_MERC_NORMAL",
    "HAUNTED_SHIP_NORMAL",
    "LAGAVULIN_MATRIARCH_BOSS",
    "LIVING_FOG_NORMAL",
    "PHANTASMAL_GARDENERS_ELITE",
    "PUNCH_CONSTRUCT_NORMAL",
    "SEAPUNK_NORMAL",
    "SEAPUNK_WEAK",
    "SEWER_CLAM_NORMAL",
    "SKULKING_COLONY_ELITE",
    "SLUDGE_SPINNER_WEAK",
    "SOUL_FYSH_BOSS",
    "TERROR_EEL_ELITE",
    "TOADPOLES_WEAK",
    "TWO_TAILED_RATS_NORMAL",
    "WATERFALL_GIANT_BOSS",
]

ALL_ENCOUNTERS: list[str] = (
    ACT1_COVERAGE_ENCOUNTERS
    + ACT1_UNDERDOCKS_COVERAGE_ENCOUNTERS
    + ACT2_COVERAGE_ENCOUNTERS
    + ACT3_COVERAGE_ENCOUNTERS
)

#: A single-monster Act-1 normal used as the neutral "arena" for card,
#: relic, and potion coverage trials. It must (a) be single-monster and
#: (b) be currently parity-clean against a pure starter deck, so its own
#: behavior contributes no end-turn noise to the varied-element signal.
#: Empirically validated clean (see docs/audits/2026-05-30-coverage-sweep.md).
ARENA_ENCOUNTER: str = "SLITHERING_STRANGLER_NORMAL"

#: Ironclad's real 10-card starter deck — the neutral baseline for relic /
#: potion / encounter trials (combat plays out naturally for a few turns so
#: reactive hooks fire). Only basics/BASH so nothing here perturbs parity.
_STARTER_DECK: tuple[str, ...] = ("STRIKE_IRONCLAD",) * 5 + ("DEFEND_IRONCLAD",) * 4 + ("BASH",)

#: Filler that pads a card trial's deck out to a full 5-card opening hand
#: WITHOUT adding a second card under test. Kept to inert basics.
_CARD_TRIAL_FILLER: tuple[str, ...] = (
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
)

#: Regent's neutral baseline deck. The real starter is 4×Strike, 4×Defend,
#: Falling Star, Venerate — but Falling Star / Venerate are non-inert (they're
#: themselves under test as cards), so the baseline pads with basics only so it
#: contributes no end-turn noise to the varied-element signal.
_REGENT_STARTER_DECK: tuple[str, ...] = ("STRIKE_REGENT",) * 6 + ("DEFEND_REGENT",) * 4

#: Regent card-trial filler (inert Regent basics, mirrors _CARD_TRIAL_FILLER).
_REGENT_CARD_TRIAL_FILLER: tuple[str, ...] = (
    "STRIKE_REGENT",
    "DEFEND_REGENT",
    "STRIKE_REGENT",
    "DEFEND_REGENT",
)

#: Silent's neutral baseline deck (inert Strike/Defend basics — drops the real
#: starter to keep the relic/potion/encounter baselines noise-free).
_SILENT_STARTER_DECK: tuple[str, ...] = ("STRIKE_SILENT",) * 5 + ("DEFEND_SILENT",) * 5

#: Silent card-trial filler (inert Silent basics, mirrors _CARD_TRIAL_FILLER).
_SILENT_CARD_TRIAL_FILLER: tuple[str, ...] = (
    "STRIKE_SILENT",
    "DEFEND_SILENT",
    "STRIKE_SILENT",
    "DEFEND_SILENT",
)

#: Defect's neutral baseline deck (inert Strike/Defend basics — mirrors Silent).
_DEFECT_STARTER_DECK: tuple[str, ...] = ("STRIKE_DEFECT",) * 5 + ("DEFEND_DEFECT",) * 5

#: Defect card-trial filler (inert Defect basics, mirrors _CARD_TRIAL_FILLER).
_DEFECT_CARD_TRIAL_FILLER: tuple[str, ...] = (
    "STRIKE_DEFECT",
    "DEFEND_DEFECT",
    "STRIKE_DEFECT",
    "DEFEND_DEFECT",
)

#: Necrobinder's neutral baseline deck (inert Strike/Defend basics).
_NECROBINDER_STARTER_DECK: tuple[str, ...] = ("STRIKE_NECROBINDER",) * 5 + (
    "DEFEND_NECROBINDER",
) * 5

#: Necrobinder card-trial filler (inert Necrobinder basics).
_NECROBINDER_CARD_TRIAL_FILLER: tuple[str, ...] = (
    "STRIKE_NECROBINDER",
    "DEFEND_NECROBINDER",
    "STRIKE_NECROBINDER",
    "DEFEND_NECROBINDER",
)

# ---------------------------------------------------------------------------
# Per-character v2 (random-seed) sweep pools
# ---------------------------------------------------------------------------
# The v2 sweep was originally Ironclad-only; these maps parameterize the
# fuzz / upgraded / arena-cross axes by character. The Ironclad entries are
# the exact pools the axes hardcoded before, so the Ironclad v2 sweep (and
# its committed trace cache) is byte-for-byte unchanged. The pairwise axis
# stays Ironclad-only — HOOK_GROUPS is an IC-curated hook taxonomy that does
# not generalize across classes.
_V2_CLASS_CARDS: dict[str, list[str]] = {
    "Ironclad": ALL_IC_CARDS,
    "Silent": ALL_SILENT_CARDS,
    "Defect": ALL_DEFECT_CARDS,
    "Necrobinder": ALL_NECROBINDER_CARDS,
    "Regent": ALL_REGENT_CARDS,
}
_V2_FILLER: dict[str, tuple[str, ...]] = {
    "Ironclad": _CARD_TRIAL_FILLER,
    "Silent": _SILENT_CARD_TRIAL_FILLER,
    "Defect": _DEFECT_CARD_TRIAL_FILLER,
    "Necrobinder": _NECROBINDER_CARD_TRIAL_FILLER,
    "Regent": _REGENT_CARD_TRIAL_FILLER,
}


# ---------------------------------------------------------------------------
# Game-build card deltas (GH #793)
# ---------------------------------------------------------------------------
# v0.111.0 removed Scare and added 19 cards (`docs/plans/2026-10-17-v0.111.0-
# port-inventory.md`, "S8a cards"). The v0.107.1 lists above stay exactly as
# they were, so the default sweeps (and their trace caches) do not move. At
# v0.111.0 the removed ids are dropped and the added ones APPENDED at the end
# of each list, the same position-stable shape the autoplay axis uses.
# MP-only cards are swept like the existing ones (TANK_CARD, SNEAKY_CARD, …):
# a scenario deck can hold them even though no single-player pool offers them.
_V0111_REMOVED_CARDS: frozenset[str] = frozenset({"SCARE"})
_V0111_ADDED_CLASS_CARDS: dict[str, list[str]] = {
    "Ironclad": ["BLAZE", "MIDNIGHT", "OUTRAGE"],
    "Silent": ["BLADE_SYMPHONY", "CONCOCT_CARD", "FADE_CARD", "SIDESTEP"],
    "Defect": ["HIBERNATE_CARD", "IMITATION_LEARNING_CARD", "ONE_FOR_ALL_CARD"],
    "Necrobinder": ["CACOPHONY_CARD", "SOULBOUND_CARD", "UNDERWORLD_CARD"],
    "Regent": ["CONSTELLATION", "PLOT", "TUTOR"],
}
_V0111_ADDED_COLORLESS_CARDS: list[str] = ["THE_BALL"]
_V0111_ADDED_EVENT_CARDS: list[str] = ["ABUNDANCE"]


def sweep_game_version() -> str:
    """The game build the sweep lists are for: ``$STS2_GAME_VERSION`` or v0.107.1.

    `sts2_client.apply_game_version_arg` (``--game-version``) sets the env var
    for the whole process tree, so reading it here keeps this module free of
    the client import.
    """
    raw = os.environ.get("STS2_GAME_VERSION", "").strip()
    if not raw:
        return "v0.107.1"
    return raw if raw.startswith("v") else f"v{raw}"


def _for_build(cards: list[str], added: list[str], game_version: str | None) -> list[str]:
    """`cards` as game build `game_version` has them (see the deltas above)."""
    version = game_version or sweep_game_version()
    if not has_v0111_cards(version):
        return cards
    return [c for c in cards if c not in _V0111_REMOVED_CARDS] + added


def class_cards_for_build(character: str, game_version: str | None = None) -> list[str]:
    """A character's class cards at `game_version` (default: the sweep build)."""
    return _for_build(_V2_CLASS_CARDS[character], _V0111_ADDED_CLASS_CARDS[character], game_version)


def colorless_cards_for_build(game_version: str | None = None) -> list[str]:
    """`ALL_COLORLESS_CARDS` at `game_version`.

    v0.113.0 adds `MY_SHINIES` to the colorless pool right after `MIND_BLAST`
    (`ColorlessCardPool.cs`; Rust `specs_in` / `COLORLESS_V0113`).
    """
    version = game_version or sweep_game_version()
    cards = _for_build(ALL_COLORLESS_CARDS, _V0111_ADDED_COLORLESS_CARDS, version)
    if _version_tuple(version) >= (0, 113, 0):
        at = cards.index("MIND_BLAST") + 1
        cards = [*cards[:at], "MY_SHINIES", *cards[at:]]
    return cards


def event_cards_for_build(game_version: str | None = None) -> list[str]:
    """`ALL_EVENT_CARDS` at `game_version`."""
    return _for_build(ALL_EVENT_CARDS, _V0111_ADDED_EVENT_CARDS, game_version)


def _v2_card_pool(character: str, game_version: str | None = None) -> list[str]:
    """Class cards + colorless — the deck pool for a character's v2 sweep."""
    return class_cards_for_build(character, game_version) + colorless_cards_for_build(game_version)


@dataclass(frozen=True)
class CoveragePack:
    """One character's single-variable coverage inventory.

    ``character`` is the baseline played in every trial; ``cards`` are
    one-hot card trials; ``relics`` are one-each relic trials. Potions and
    encounters are character-agnostic, so only the Ironclad pack sweeps them
    (a per-character re-sweep would be redundant) — other packs stay focused
    on the class-specific surface that has never been A/B'd.
    """

    character: str
    cards: list[str]
    relics: list[str]
    starter: tuple[str, ...]
    filler: tuple[str, ...]
    include_potions: bool
    include_encounters: bool


def _coverage_pack(character: str, game_version: str | None = None) -> CoveragePack:
    gv = game_version
    if character == "Ironclad":
        return CoveragePack(
            character="Ironclad",
            cards=class_cards_for_build("Ironclad", gv)
            + colorless_cards_for_build(gv)
            + event_cards_for_build(gv),
            relics=ALL_RELICS,
            starter=_STARTER_DECK,
            filler=_CARD_TRIAL_FILLER,
            include_potions=True,
            include_encounters=True,
        )
    if character == "Silent":
        # Mirrors the Ironclad pack (colorless cards + ALL_RELICS + potions +
        # encounters) but with Silent's class cards / starter / filler. Silent
        # relics aren't a separate pool yet — ALL_RELICS already covers the
        # cross-class grantable surface the relic axis exists to saturate.
        return CoveragePack(
            character="Silent",
            cards=class_cards_for_build("Silent", gv)
            + colorless_cards_for_build(gv)
            + event_cards_for_build(gv),
            relics=ALL_RELICS,
            starter=_SILENT_STARTER_DECK,
            filler=_SILENT_CARD_TRIAL_FILLER,
            include_potions=True,
            include_encounters=True,
        )
    if character == "Defect":
        # Full mirror of the Ironclad pack (colorless + ALL_RELICS + potions +
        # encounters) with Defect's class cards / starter / filler.
        return CoveragePack(
            character="Defect",
            cards=class_cards_for_build("Defect", gv)
            + colorless_cards_for_build(gv)
            + event_cards_for_build(gv),
            relics=ALL_RELICS,
            starter=_DEFECT_STARTER_DECK,
            filler=_DEFECT_CARD_TRIAL_FILLER,
            include_potions=True,
            include_encounters=True,
        )
    if character == "Necrobinder":
        # Full mirror of the Ironclad pack with Necrobinder's class cards.
        return CoveragePack(
            character="Necrobinder",
            cards=class_cards_for_build("Necrobinder", gv)
            + colorless_cards_for_build(gv)
            + event_cards_for_build(gv),
            relics=ALL_RELICS,
            starter=_NECROBINDER_STARTER_DECK,
            filler=_NECROBINDER_CARD_TRIAL_FILLER,
            include_potions=True,
            include_encounters=True,
        )
    if character == "Regent":
        return CoveragePack(
            character="Regent",
            cards=class_cards_for_build("Regent", gv) + event_cards_for_build(gv),
            relics=ALL_REGENT_RELICS,
            starter=_REGENT_STARTER_DECK,
            filler=_REGENT_CARD_TRIAL_FILLER,
            include_potions=False,
            include_encounters=False,
        )
    raise ValueError(f"no coverage pack for character {character!r}")


_COVERAGE_HP = 70
_COVERAGE_MAX_HP = 80


def _coverage_base(**kw: Any) -> Scenario:
    """Fixed clean baseline for single-variable coverage trials."""
    params: dict[str, Any] = dict(
        character="Ironclad",
        hp=_COVERAGE_HP,
        max_hp=_COVERAGE_MAX_HP,
        relics=(),
        potions=(),
        encounter=ARENA_ENCOUNTER,
    )
    params.update(kw)
    return Scenario(**params)


def coverage_scenarios(
    character: str = "Ironclad", game_version: str | None = None
) -> list[Scenario]:
    """Deterministic, single-variable coverage sweep over a character's content.

    One trial per card / relic / potion / encounter — each varying exactly
    that element against an otherwise-fixed clean baseline, so any divergence
    buckets under the element's ``coverage_subject`` label. Additive in size
    (no cartesian explosion).

    ``character`` selects the :class:`CoveragePack`: ``"Ironclad"`` (default)
    is the original full sweep (IC + colorless cards, all relics/potions/
    encounters); ``"Regent"`` is the class-specific surface (88 Regent cards +
    Regent's relic pool) that was ported from the old Python sim and has never
    been A/B'd against the C# oracle.
    """
    pack = _coverage_pack(character, game_version)
    scenarios: list[Scenario] = []

    # 1. Every card: the test card sits in a 5-card deck so the whole deck is
    #    the opening hand → the card is always drawn, and is the only non-basic
    #    in play.
    for card in pack.cards:
        scenarios.append(
            _coverage_base(
                character=pack.character,
                deck=(card, *pack.filler),
                coverage_subject=f"card:{card}",
            )
        )

    # 2. Every relic: exactly one, on the neutral starter deck.
    for relic in pack.relics:
        scenarios.append(
            _coverage_base(
                character=pack.character,
                deck=pack.starter,
                relics=(relic,),
                coverage_subject=f"relic:{relic}",
            )
        )

    # 3. Every potion: exactly one, held at combat start on the starter deck.
    #    Character-agnostic, so only swept once (Ironclad pack).
    if pack.include_potions:
        for potion in ALL_POTIONS:
            scenarios.append(
                _coverage_base(
                    character=pack.character,
                    deck=pack.starter,
                    potions=(potion,),
                    coverage_subject=f"potion:{potion}",
                )
            )

    # 4. Every encounter: neutral starter deck, no relics/potions.
    #    Character-agnostic, so only swept once (Ironclad pack).
    if pack.include_encounters:
        for enc in ALL_ENCOUNTERS:
            scenarios.append(
                _coverage_base(
                    character=pack.character,
                    deck=pack.starter,
                    encounter=enc,
                    coverage_subject=f"encounter:{enc}",
                )
            )

    return scenarios


def coverage_upgraded_scenarios(character: str = "Ironclad") -> list[Scenario]:
    """Axis A (sweep v2): every class + colorless card in its UPGRADED form.

    Same one-hot shape as the v1 card trials — the "+1" card sits first in
    a 5-card deck of unupgraded inert basics, so the upgraded card is the
    only varied element and any divergence buckets under its subject.
    Upgrade-scaling drift is a confirmed rich vein (~10/12 hand-audited
    cards drifted), which is exactly what this axis exists to saturate.

    Both backends accept only "+1"; a "+1" on a non-upgradable id
    (curse/status) is a silent no-op on each side independently — Rust
    via the is_upgradable() gate in build_scenario_state (sts2-py
    lib.rs), C# via the CurrentUpgradeLevel < MaxUpgradeLevel loop
    guard in SetPlayerImpl (RunSimulator.cs). Moot today anyway: the
    pools contain no curse/status ids.
    """
    return [
        _coverage_base(
            character=character,
            deck=(f"{card}+1", *_V2_FILLER[character]),
            coverage_subject=f"card+:{card}",
        )
        for card in _v2_card_pool(character)
    ]


# Subject grammar (load-bearing — parsed in parity_audit):
#   pre-flight: "arena:<act>"         (parser: split(":", 1)[1])
#   cross:      "<kind>:<id>@<act>"   (parser: rsplit("@", 1)[1])
# Both yield a bare act key ("act2", "act3") for blocked-set membership.

#: Axis B (sweep v2): one parity-validated single-monster arena per act.
#: Selection criteria: single-monster, `randomize: None` in the Rust
#: encounter table, encounter-coverage clean at the 369/369 sweep, and
#: minimal passive noise (no summons / illusions / mid-combat spawns).
#: Candidates if a pre-flight ever goes red — act2: SLUMBERING_BEETLE_NORMAL,
#: HUNTER_KILLER_NORMAL; act3: GLOBE_HEAD_NORMAL, OWL_MAGISTRATE_NORMAL.
ARENAS_BY_ACT: dict[str, str] = {
    "act1": ARENA_ENCOUNTER,  # SLITHERING_STRANGLER_NORMAL
    "act2": "SPINY_TOAD_NORMAL",
    "act3": "FROG_KNIGHT_NORMAL",
}


def arena_preflight_scenarios() -> list[Scenario]:
    """Starter-deck probe per non-act1 arena, run BEFORE the cross.

    If an arena diverges on the plain starter deck, every cross trial in
    it would bucket as cascade noise — the driver marks the whole arena
    column BLOCKED instead of running it.
    """
    return [
        _coverage_base(
            deck=_STARTER_DECK,
            encounter=enc,
            coverage_subject=f"arena:{act}",
        )
        for act, enc in ARENAS_BY_ACT.items()
        if act != "act1"
    ]


def coverage_arena_cross_scenarios(character: str = "Ironclad") -> list[Scenario]:
    """Axis B (sweep v2): every card/card+/relic/potion trial × act2+act3.

    The act1 column already exists (v1 coverage + Axis A use the act1
    arena), so the cross emits only the two new arenas. Subjects carry an
    ``@act<N>`` qualifier so a divergence names both the content element
    and the arena that exposed it.

    Per character, only the class+colorless card/card+ trials vary — the
    relic and potion arena trials are character-agnostic and swept once via
    the Ironclad pass (mirrors the ``include_potions``/``include_encounters``
    dedup in :func:`coverage_scenarios`). For ``character="Ironclad"`` this
    reproduces the original (cache-stable) output exactly.
    """
    out: list[Scenario] = []
    filler = _V2_FILLER[character]
    for act in ("act2", "act3"):
        enc = ARENAS_BY_ACT[act]
        for card in _v2_card_pool(character):
            out.append(
                _coverage_base(
                    character=character,
                    deck=(card, *filler),
                    encounter=enc,
                    coverage_subject=f"card:{card}@{act}",
                )
            )
            out.append(
                _coverage_base(
                    character=character,
                    deck=(f"{card}+1", *filler),
                    encounter=enc,
                    coverage_subject=f"card+:{card}@{act}",
                )
            )
        if character == "Ironclad":
            for relic in ALL_RELICS:
                out.append(
                    _coverage_base(
                        deck=_STARTER_DECK,
                        relics=(relic,),
                        encounter=enc,
                        coverage_subject=f"relic:{relic}@{act}",
                    )
                )
            for potion in ALL_POTIONS:
                out.append(
                    _coverage_base(
                        deck=_STARTER_DECK,
                        potions=(potion,),
                        encounter=enc,
                        coverage_subject=f"potion:{potion}@{act}",
                    )
                )
    return out


#: Axis D (sweep v2): hook-surface taxonomy. Each group lists GENERATOR
#: cards (cause the event) and LISTENER content (react to it); the
#: pairwise sweep runs every within-group (generator, listener) pair in
#: one deck so the rider actually fires. The membership unit test rejects
#: ids that drift from the pools.
#:
#: Curated by C# semantics (what the GAME fires), NOT by what the Rust
#: port currently dispatches — a Rust-side hook bypass is precisely the
#: divergence this axis exists to catch.
#:
#: To extend: add a group key with generators/listeners lists spelled per
#: the pool constants above — the membership test and the pair generator
#: pick it up automatically.
#:
#: Past veins these groups encode:
#:   DRAW  — card-effect draws bypassing on-draw riders (lj36b): draw_via_state
#:           bypassed ConfusedPower.AfterCardDrawn / FAKE_SNECKO_EYE cost re-rolls.
#:   EXHAUST — EOT-flush vs active-discard hooks (q6bix TINGSHA), Dark Embrace
#:             / FeelNoPain / ForgottenSoul fire on exhaust from hand.
#:   HP_LOSS — HP-loss self-damage consumers (RUPTURE / SELF_FORMING_CLAY, t9xre):
#:             OFFERING/BLOODLETTING/HEMOKINESIS deal unblocked self-damage.
#:   SHUFFLE — after_shuffle mid-draw choices (Stratagem bead 63mew): a heavy
#:             draw that depletes the draw pile reshuffles and triggers Stratagem.
#:
#: Groups DROPPED after honest curation (no IC/Colorless generators in pool):
#:   DISCARD — TINGSHA/ToughBandages fire on active discards (CardCmd.DiscardAndDraw);
#:             the only active-discard card handlers are Silent (ACROBATICS /
#:             SURVIVOR / DAGGER_THROW) — none present in IC+Colorless pools.
#:
#: Generator verification notes (C# decomp v0.103.2 = source of truth):
#:   SECOND_WIND: C# SecondWind.OnPlay calls CardCmd.Exhaust per non-Attack hand
#:                card → CardCmd.Exhaust:183 fires Hook.AfterCardExhausted. INCLUDED.
#:                (Rust port uses raw move_card and does NOT fire the hook — a
#:                suspected engine bug this pair surfaces.)
#:   TRUE_GRIT: C# TrueGrit.OnPlay calls CardCmd.Exhaust in BOTH branches
#:              (upgraded CardSelectCmd.FromHand + non-upgraded NextItem) →
#:              fires Hook.AfterCardExhausted. INCLUDED. (Rust port uses raw
#:              move_card and does NOT fire the hook — same suspected bug.)
#:   CATASTROPHE: C# Catastrophe.OnPlay shuffles a LOCAL .ToList() copy of the
#:                draw pile via StableShuffle (ListExtensions.cs — a pure
#:                list sort+Fisher-Yates, no Hook.AfterShuffle) and AutoPlays the
#:                first; only CardPileCmd.Shuffle:857 fires Hook.AfterShuffle.
#:                C# does NOT fire AfterShuffle here → genuinely NOT a SHUFFLE
#:                generator on EITHER side → EXCLUDED (no divergence to catch).
#:   DARK_EMBRACE_CARD / FEEL_NO_PAIN_CARD: these apply powers that LISTEN to
#:                exhaust (draw 1 / gain block per exhaust); they are listeners.
#:   RUPTURE_CARD: applies Rupture power that LISTENS to hp_loss (gain Strength);
#:                 it is a listener not a generator.
HOOK_GROUPS: dict[str, dict[str, list[str]]] = {
    "DRAW": {
        # IC generators: draw_via_state call confirmed in each file
        # Colorless generators: draw_via_state / draw_into_hand confirmed
        "generators": [
            "POMMEL_STRIKE",  # ironclad/pommel_strike.rs: draw_via_state(draws)
            "BATTLE_TRANCE",  # ironclad/battle_trance.rs: draw_via_state(cards)
            "OFFERING",  # ironclad/offering.rs: draw_via_state(draws)
            "SHRUG_IT_OFF",  # ironclad/shrug_it_off.rs: draw_via_state(1)
            "MASTER_OF_STRATEGY",  # colorless/master_of_strategy.rs: draw_via_state(count)
            "FINESSE",  # colorless/finesse.rs: draw_into_hand path
        ],
        # FAKE_SNECKO_EYE: before_combat_start applies Confused power;
        # Confused.AfterCardDrawn re-rolls card costs per draw (bead lj36b).
        "listeners": [
            "FAKE_SNECKO_EYE",  # relic — applies Confused; costs re-rolled on each draw
        ],
    },
    "EXHAUST": {
        # Generators that exhaust cards. C# routes all of these through
        # CardCmd.Exhaust → Hook.AfterCardExhausted (CardCmd.cs:183), so the
        # pair must fire FeelNoPain / DarkEmbrace / ForgottenSoul on the C#
        # side. SECOND_WIND / TRUE_GRIT are included on C# semantics even
        # though the Rust port currently uses raw move_card (suspected bug
        # this axis surfaces); the others already call exhaust_hand_card /
        # exhaust_card_uid in Rust too.
        "generators": [
            "FIEND_FIRE",  # C# CardCmd.Exhaust per hand card; Rust exhaust_hand_card
            "THRASH",  # C# CardCmd.Exhaust on a random card; Rust exhaust_hand_card
            "CINDER",  # C# CardCmd.Exhaust; Rust exhaust_hand_card
            "BURNING_PACT",  # C# CardCmd.Exhaust via FromHand; Rust exhaust_card_uid
            "SECOND_WIND",  # C# CardCmd.Exhaust per non-Attack; Rust raw move_card (bug?)
            "TRUE_GRIT",  # C# CardCmd.Exhaust both branches; Rust raw move_card (bug?)
        ],
        # Card listeners: apply powers that trigger on after_card_exhausted.
        # Relic listeners: ForgottenSoul wired on after_card_exhausted bus.
        "listeners": [
            "FEEL_NO_PAIN_CARD",  # applies FeelNoPain power (gain 3/4 block per exhaust)
            "DARK_EMBRACE_CARD",  # applies DarkEmbrace power (draw 1 per exhaust)
            "FORGOTTEN_SOUL",  # relic: deal 1 damage to random enemy per exhaust
        ],
    },
    "HP_LOSS": {
        # IC generators: all call deal_hp_loss / apply self-damage path that fires
        # after_damage_received on the relic-effects bus
        "generators": [
            "OFFERING",  # ironclad/offering.rs: loses 6 hp (DealHpLoss self)
            "BLOODLETTING",  # ironclad/bloodletting.rs: loses 3 hp (DealHpLoss self)
            "HEMOKINESIS",  # ironclad/hemokinesis.rs: loses 2 hp
        ],
        # Relic listeners: fire on after_damage_received
        "listeners": [
            "RUPTURE_CARD",  # applies Rupture power (gain Strength on unblocked HP loss)
            "SELF_FORMING_CLAY",  # relic: queue block-next-turn on unblocked damage
            "CENTENNIAL_PUZZLE",  # relic: draw 3 cards on first HP loss per combat
        ],
    },
    "SHUFFLE": {
        # Cards that draw enough to exhaust the draw pile → mid-draw reshuffle →
        # dispatch_after_shuffle fires → Stratagem power grid-selects.
        # A 5-card starting deck with one draw card means the draw pile (remaining
        # 4 cards after initial 5-card draw) can be depleted by a 5+ draw.
        # SCRAWL draws up to HAND_LIMIT; BATTLE_TRANCE draws 3+; MASTER_OF_STRATEGY
        # draws 3 — all can trigger a reshuffle in a small-deck scenario.
        "generators": [
            "BATTLE_TRANCE",  # ironclad/battle_trance.rs: draw_via_state(3+) — may reshuffle
            "SCRAWL",  # colorless/scrawl.rs: draw to HAND_LIMIT — likely reshuffles
            "MASTER_OF_STRATEGY",  # colorless/master_of_strategy.rs: draw 3 — may reshuffle
        ],
        # Stratagem power listens to after_shuffle (hooks::dispatch_after_shuffle,
        # bead sts2-rs-63mew); STRATAGEM card applies the power.
        "listeners": [
            "STRATAGEM",  # colorless/stratagem.rs: applies Stratagem power (shuffle listener)
        ],
    },
}


def pairwise_hook_scenarios() -> list[Scenario]:
    """Axis D (sweep v2): within-group generator x listener pairs.

    Card listeners join the deck; relic listeners ride the relic slot.
    The deck is padded with inert basics to a 5-card opening hand so both
    halves of the pair are guaranteed drawn turn 1.

    Subjects use a lowercase "x" separator — safe because ids are
    UPPER_SNAKE. If a (gen, listener) pair would appear in two groups it
    is deduped by subject so the uniqueness invariant holds.

    EXHAUST card-listener pairs are probabilistic — if random play exhausts
    the listener card (or plays the generator first), the listener's power
    is never installed and the hook does not fire that trial; the pair
    still catches divergences in aggregate across runs.
    """
    relic_ids = set(ALL_RELICS)
    seen_subjects: set[str] = set()
    out: list[Scenario] = []
    for spec in HOOK_GROUPS.values():
        for gen in spec["generators"]:
            for listener in spec["listeners"]:
                subject = f"pair:{gen}x{listener}"
                if subject in seen_subjects:
                    continue
                seen_subjects.add(subject)
                if listener in relic_ids:
                    deck: tuple[str, ...] = (gen, *_CARD_TRIAL_FILLER)
                    relics: tuple[str, ...] = (listener,)
                else:
                    deck = (gen, listener, *_CARD_TRIAL_FILLER[:3])
                    relics = ()
                assert len(deck) == 5, deck
                out.append(
                    _coverage_base(
                        deck=deck,
                        relics=relics,
                        coverage_subject=subject,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Axis E (sweep v2): the auto-play / replay corner (GH #500)
# ---------------------------------------------------------------------------

#: Deck shape the other axes never assemble. Every card here is either an
#: auto-play DRIVER (MAYHEM's turn-start batch, HAVOC's force-exhaust top,
#: CATASTROPHE's two `StableShuffle` picks — and they nest: a CATASTROPHE pick
#: can be a HAVOC whose own pick is a third card) or a card whose `OnPlay`
#: opens an interactive select (ARMAMENTS upgrade-in-hand, SEEKER_STRIKE
#: add-from-draw, HEADBUTT put-back-from-discard). Paired with HISTORY_COURSE
#: — which replays last round's last Attack/Skill as a dupe at turn start —
#: every combination of "an awaited auto-play surfaces a select" is reachable.
#:
#: The single-variable coverage axes put one card in a deck of inert basics, so
#: they can never produce a nested auto-play; the random `fuzz` axis draws from
#: the full pool and essentially never rolls three drivers plus three
#: select-openers into one 14-card deck. That is why a 6.5%-divergence corner
#: sat unmeasured until the targeted campaign in GH #500 found it.
AUTOPLAY_REPLAY_DECK: tuple[str, ...] = (
    "MAYHEM_CARD",
    "MAYHEM_CARD",
    "HAVOC",
    "HAVOC",
    "CATASTROPHE",
    "ARMAMENTS",
    "ARMAMENTS",
    "SEEKER_STRIKE",
    "SEEKER_STRIKE",
    "HEADBUTT",
    "STRIKE_IRONCLAD",
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "DEFEND_IRONCLAD",
)

#: HP for the auto-play axis. Deliberately far above the Act-1 damage budget:
#: the interesting states are late-combat ones (the draw pile has cycled at
#: least once, so a reshuffle's order-sensitive `StableShuffle` is in play), and
#: a trial that ends in a death never reaches them.
_AUTOPLAY_HP = 200


#: Distinct action trajectories per encounter at the full tier. The sweep
#: derives a trial's action seed from its POSITION, so listing a scenario N
#: times plays it down N different random lines (the same trick ``--paths``
#: applies globally). This axis NEEDS them: which auto-play driver nests inside
#: which, and whether the card it reaches opens a select, is decided by the play
#: ORDER, so the deck alone guarantees nothing. Calibrated against the pre-fix
#: engine of GH #500 — at 4 lines/encounter the axis was entirely clean on a
#: build carrying a known divergence; at 24 it reported 7, including the
#: `DRAW_ORDER :: discard_ids` signature that fix is about.
_AUTOPLAY_LINES = 24


#: GH #1021 (axis F): one deck per character holding every in-combat
#: enchantment on a card C# `CanEnchant` accepts, plus replay sources (Havoc
#: auto-play, Spiral/Glam extra plays, Echo Form) so the enchantment OnPlay /
#: AfterCardPlayed legs run on replays too. No C#-oracle trial built an
#: enchanted deck before this axis.
ENCHANT_DECKS: dict[str, tuple[str, ...]] = {
    "Ironclad": (
        "STRIKE_IRONCLAD@SHARP:2",
        "STRIKE_IRONCLAD@SPIRAL:1",
        "DEFEND_IRONCLAD@GOOPY:1",
        "DEFEND_IRONCLAD@STEADY:1",
        "BASH@MOMENTUM:3",
        "RAMPAGE@MOMENTUM:2",
        "POMMEL_STRIKE@CORRUPTED:1",
        "IRON_WAVE@INSTINCT:1",
        "TWIN_STRIKE@VIGOROUS:3",
        "SHRUG_IT_OFF@ROYALLY_APPROVED:1",
        "TRUE_GRIT@NIMBLE:2",
        "ARMAMENTS@ADROIT:3",
        "BATTLE_TRANCE@SWIFT:1",
        "HEADBUTT@GLAM:1",
        "BLOODLETTING@PERFECT_FIT:1",
        "INFLAME@PERFECT_FIT:1",
        "ENTRENCH@NIMBLE:2",
        "ANGER@TEZCATARAS_EMBER:1",
        "HAVOC",
        "HAVOC",
    ),
    "Defect": (
        "STRIKE_DEFECT@MOMENTUM:2",
        "STRIKE_DEFECT@CORRUPTED:1",
        "DEFEND_DEFECT@ADROIT:3",
        "DEFEND_DEFECT@GOOPY:1",
        "BALL_LIGHTNING@SHARP:2",
        "COLD_SNAP@STEADY:1",
        "CHARGE_BATTERY@NIMBLE:2",
        "LEAP@ROYALLY_APPROVED:1",
        "BEAM_CELL@VIGOROUS:2",
        "COOLHEADED@SWIFT:1",
        "SWEEPING_BEAM@SOWN:1",
        "ECHO_FORM_CARD",
        "ECHO_FORM_CARD",
    ),
}
_ENCHANT_LINES = 4
_ENCHANT_HP = 200


def enchant_scenarios(character: str = "Ironclad", lines: int | None = None) -> list[Scenario]:
    """Axis F (GH #1021): :data:`ENCHANT_DECKS` x every non-boss Act-1
    encounter x ``lines`` action trajectories. ``[]`` for a character without
    a deck. A high HP keeps the fights going long enough for replays."""
    deck = ENCHANT_DECKS.get(character)
    if deck is None:
        return []
    lines = _ENCHANT_LINES if lines is None else lines
    return [
        _coverage_base(
            character=character,
            deck=deck,
            hp=_ENCHANT_HP,
            max_hp=_ENCHANT_HP,
            relics=(),
            encounter=enc,
            coverage_subject=f"enchant:{enc}#{line}",
        )
        for enc in ACT1_ENCOUNTERS
        if "BOSS" not in enc
        for line in range(lines)
    ]


def autoplay_replay_scenarios(
    character: str = "Ironclad", lines: int | None = None
) -> list[Scenario]:
    """Axis E (sweep v2): :data:`AUTOPLAY_REPLAY_DECK` × every Act-1 encounter
    × ``lines`` action trajectories (default :data:`_AUTOPLAY_LINES`).

    Ironclad-only (the deck names IC cards); returns ``[]`` for every other
    character, exactly like the Ironclad-curated ``pairwise`` axis.

    Divergences bucket under the encounter they were found in. Boss encounters
    are excluded: their scripted openers dominate the first turns and crowd out
    the auto-play traffic this axis exists to generate.
    """
    if character != "Ironclad":
        return []
    lines = _AUTOPLAY_LINES if lines is None else lines
    return [
        _coverage_base(
            deck=AUTOPLAY_REPLAY_DECK,
            hp=_AUTOPLAY_HP,
            max_hp=_AUTOPLAY_HP,
            relics=("BURNING_BLOOD", "HISTORY_COURSE"),
            encounter=enc,
            # The line index is part of the subject: the sweep manifest is
            # keyed by subject and silently OVERWRITES duplicates, so trials
            # sharing one label would report as one.
            coverage_subject=f"autoplay_replay:{enc}#{line}",
        )
        for enc in ACT1_ENCOUNTERS
        if "BOSS" not in enc
        for line in range(lines)
    ]
