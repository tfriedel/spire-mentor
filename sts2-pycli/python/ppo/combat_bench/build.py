"""`python -m ppo.combat_bench build`: the bench's fight sets and MANIFEST (GH #830).

The bench replays human A10 v0.107.1 Ironclad fights from their exact entry
state, so every field a fight row carries has to say what the human actually
had. `pilot_human_decks.extract_fights` rebuilds the deck, relics and HP; this
module adds what the bench needs on top, each rule checked against the C#
decompilation (v0.107.1):

- **Pro cohort.** Players are ranked on their A10 runs from builds other than
  v0.107.1 (Wilson lower bound, at least 15 runs, top 25) with ABANDONED RUNS
  COUNTED AS LOSSES. The earlier `pros25.json` dropped abandoned runs, which
  flatters players who abandon the runs they are losing.
- **Abandoned runs.** Their completed fights count. The last fight of an
  abandoned run that ended in combat at 0 HP without lethal damage
  (`damage_taken < hp_before + hp_healed`) is a QUIT, never a death. A
  finished run's death without lethal damage happens too: The Insatiable's
  Sandpit kills through `CreatureCmd.Kill`, which `DamageTaken` does not
  count (`death_hp_unlogged`).
- **Potion belt.** Every potion gained goes through `PotionCmd.TryToProcure`,
  which logs a picked `potion_choices` entry (purchases, rewards, relic and
  in-combat generation alike); drinking logs `potion_used`, discarding
  `potion_discarded`. Replaying those per floor rebuilds the belt at each
  fight; a run whose replay is infeasible, or does not end on the run file's
  final belt and slot count, gets `belt = null`. Slots: 3, minus 1 from A4
  (`AscensionLevel.TightBelt`), plus Potion Belt +2, Phial Holster +1 and
  Alchemical Coffer +4 from the floor they were picked.
- **Deck.** `pilot_human_decks.DeckReplay` replays the deck backward from
  the final deck, the only exact per-copy record, so upgrades and
  enchantments sit on the copies that had them. Goopy (+1 amount per play,
  `Goopy.AfterCardPlayed`) amounts are interpolated over the combats and
  flagged `goopy_amount`. The forward count rule that made `hbank` and the
  `logs/2026-10-18-rollout-net-recipe` sets under-counts upgrades (an upgrade
  of a copy gained later counted against the copies held; removed and
  transformed copies lost theirs); the MANIFEST counts the fights it got
  wrong (`decks`).
- **Pilot potions.** The pilot gets the potions the human drank in the fight
  (those that were on the entry belt first, then those generated mid-fight)
  and every held potion that acts without being drunk. Fairy in a Bottle is
  the only one (`PotionUsage.Automatic`, it triggers on death). The list is
  capped at the belt size. Two kinds of generated potion are left out:
  AnyTime ones (`PotionUsage.AnyTime`: picked on the fight's floor, most
  likely a reward drunk after it), and all of them where an in-combat
  generator is present (Entropic Brew given, Alchemize in the deck, Delicate
  Frond held), because the engine runs that generator for the pilot too
  (`potion_generator`).
- **Petrified Toad.** `PetrifiedToad.BeforeCombatStartLate` procures a
  Potion-Shaped Rock whenever a slot is free, and the engine does the same,
  so one drunk rock is removed from the pilot's list (the toad supplies it).
- **Monster rosters.** `rooms[].monster_ids` lists the encounter's slots in
  order, then every creature added later whose id is new: pets (Osty,
  Byrdpip, Pael's Legion) and summons. Both are filtered. The
  `build_scenario_state(monsters=...)` override drops per-monster spawn
  arguments, so it is only passed where it changes the roster and where the
  encounter rebuilds identically through it.
- **Mods.** Runs with any modifier, or any relic, card or potion id the game
  does not define, are dropped.

Every written fight is built through `sts2_rs.build_scenario_state` and
`RunManager.debug_enter_combat`; failures are excluded and counted by reason.

**Versions.** `--version v1`, the default, writes the frozen v1 names
(`ironclad_pro_v1.jsonl`, `ironclad_devpool_v1.jsonl`, `MANIFEST.json`)
under v1's rules, so it rebuilds the v1 sets byte for byte. A later version
writes `ironclad_<set>_<version>.jsonl` and `MANIFEST_<version>.json` beside
them and applies the rules v1 got wrong (`V2_RULES`):

- One player per `username.strip().casefold()`: v1 split a pro across two
  spellings, so their v0.107.1 runs fell into the dev pool. The pseudonym
  hashes the spelling seen first, so a one-spelling player keeps v1's id.
- Selection runs with modifiers or modded cards (and, for builds up to
  v0.107.1, modded enchantments or ids the game does not define) do not
  count toward the ranking.
- The last fight of an abandoned run at 0 HP is not a quit when its floor
  logs a reward taken (a card or relic choice, or gold that neither Maw
  Bank's room-entry 12 nor Hand of Greed explains): the human won it and
  abandoned on the reward screen (`abandoned_after_win`).
- `DeckReplay` undoes a downgrade on a copy that was gained upgraded first
  and notes a gain undo that lost its record's upgrade (`deck_unmatched`).
- Pro sets flag fights the rollout-net recipe evaluated or screened on
  (`in_rl2_eval`) and leave out fights a frozen dev panel holds
  (`--bench-dir`): those were screened on.

`--extra-top N` also writes the top-N cohort's pro set
(`ironclad_proN_<version>.jsonl`) in the same build and MANIFEST; the dev
pool still drops only the `--top` cohort's runs, and the MANIFEST counts the
players, runs and fights each pro set shares with each dev set. `build`
refuses to overwrite a set or MANIFEST unless `--force` is given.
`ppo.combat_bench.holdout` holds out the runs of every MANIFEST.

    python -m ppo.combat_bench build --hbank hbank.jsonl --private-dir /tmp/x \\
        --out-dir /tmp/x/out
    python -m ppo.combat_bench build --hbank hbank.jsonl --private-dir /tmp/x \\
        --out-dir /tmp/x/out --version v2 --extra-top 50
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from multiprocessing import get_context
from typing import Any

from ppo import human_players
from ppo.pilot_human_decks import (
    COMBAT_ROOMS,
    ENCHANTMENTS,
    _bare,
    _fight_seed,
    extract_fights,
    load_run,
    run_rejection,
)

CHARACTER = "Ironclad"
ASCENSION = 10
BUILD = "v0.107.1"
PRO_SET = "ironclad_pro_v1.jsonl"
DEV_SET = "ironclad_devpool_v1.jsonl"
MANIFEST = "MANIFEST.json"
VERSION_RE = re.compile(r"v[1-9][0-9]*")
# What a version after v1 does differently (module doc); MANIFEST `rules`.
V2_RULES = (
    "casefold_players",
    "clean_ranking",
    "quit_needs_no_reward",
    "deck_downgrade_owed",
    "in_rl2_eval",
    "screened_dropped",
)

_PYCLI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_PYCLI, "data", "combat_bench")
RECIPE_DIR = os.path.join(_PYCLI, "..", "logs", "2026-10-18-rollout-net-recipe")
LOC_DIR = os.path.join(_PYCLI, "localization_eng")
DEV_SOURCES = ("boss800.jsonl", "wg_fights.jsonl", "elite800.jsonl", "hall800.jsonl")

# ── potions (C# v0.107.1) ──
BASE_POTION_SLOTS = 3  # Player.initialMaxPotionSlotCount
TIGHT_BELT_ASCENSION = 4  # AscensionLevel.TightBelt: SubtractFromMaxPotionCount(1)
SLOT_RELICS = {"POTION_BELT": 2, "PHIAL_HOLSTER": 1, "ALCHEMICAL_COFFER": 4}  # AfterObtained
AUTO_POTIONS = frozenset({"FAIRY_IN_A_BOTTLE"})  # the only PotionUsage.Automatic potion
# PotionUsage.AnyTime: drinkable outside combat too. One the human picked on
# the fight's floor (not on the entry belt) and drank was most likely a
# post-combat reward drunk after the fight.
ANYTIME_POTIONS = frozenset({"BLOOD_POTION", "ENTROPIC_BREW", "FOUL_POTION", "FRUIT_JUICE"})
# In-combat potion generators the engine runs itself (`PotionCmd.TryToProcure`
# callers): handing the pilot both the generator and what it made for the
# human would double it.
BREW = "ENTROPIC_BREW"  # fills every empty slot when drunk
FROND = "DELICATE_FROND"  # fills every empty slot at combat start
ALCHEMIZE = "ALCHEMIZE"  # procures a random potion when played
ROCK = "POTION_SHAPED_ROCK"
TOAD = "PETRIFIED_TOAD"
# The pilot's combat belt has the human's slot count: `run` passes each
# fight's `potion_slots` to `build_scenario_state` (sts2-wire
# `build_scenario_state_with_slots`), so the engine's toad adds its rock only
# while fewer potions than slots are given.

# ── rosters (C# v0.107.1) ──
PETS = frozenset({"OSTY", "BYRDPIP", "PAELS_LEGION"})  # PlayerCmd.AddPet callers
# Mid-combat summons (`CreatureCmd.Add` callers) -> their summoner. A summon
# is appended to `monster_ids` only if its id is new, so self-summoners
# (Two-Tailed Rat, Axebot's StockPower) never show up twice and are absent.
SUMMONED_BY = {
    "WRIGGLER": "PHROG_PARASITE",  # InfestedPower
    "TOUGH_EGG": "OVICOPTER",
    "PARAFRIGHT": "THE_OBSCURA",
    "GAS_BOMB": "LIVING_FOG",
    "SNEAKY_GREMLIN": "GREMLIN_MERC",  # SurprisePower
    "FAT_GREMLIN": "GREMLIN_MERC",
    "EYE_WITH_TEETH": "FOGMOG",
    "ZAPBOT": "FABRICATOR",
    "STABBOT": "FABRICATOR",
    "GUARDBOT": "FABRICATOR",
    "NOISEBOT": "FABRICATOR",
}

# Engine monster kinds whose AI factory reads per-spawn arguments
# (`sts2-content` monster_dispatch.rs `MonsterSpawnArgs`); the override path
# hands them `MonsterSpawnArgs::None`.
SPAWN_ARG_KINDS = frozenset(
    {
        "NIBBIT",
        "INKLET",
        "KIN_FOLLOWER",
        "EXOSKELETON",
        "CHOMPER",
        "DECIMILLIPEDE_SEGMENT_FRONT",
        "DECIMILLIPEDE_SEGMENT_MIDDLE",
        "DECIMILLIPEDE_SEGMENT_BACK",
        "SCROLL_OF_BITING",
        "INFESTED_PRISM",
        "MYTE",
        "AXEBOT",
        "WRIGGLER",
        "PUNCH_CONSTRUCT",
        "CORPSE_SLUG",
        "TWO_TAILED_RAT",
    }
)

# ── flags ──
# Relics whose [SavedProperty] counter carries across combats and is read by a
# combat hook; the rebuild starts every one fresh.
COUNTER_RELICS = frozenset(
    {
        "BONE_TEA",
        "EMBER_TEA",
        "TEA_OF_DISCOURTESY",
        "GIRYA",
        "HAPPY_FLOWER",
        "FAKE_HAPPY_FLOWER",
        "LIZARD_TAIL",
        "PEN_NIB",
        "PENDULUM",
        "POLLINOUS_CORE",
        "NUNCHAKU",
        "IRON_CLUB",
        "TUNING_FORK",
        "VENERABLE_TEA_SET",
        "FAKE_VENERABLE_TEA_SET",
        "PUMPKIN_CANDLE",
        "GALACTIC_DUST",
    }
)
TINKER_CARDS = frozenset({"MAD_SCIENCE"})  # per-copy Tinker Time rider, not in the deck id
FLAGS = (
    "enchanted",
    "fresh_relic_counters",
    "fur_coat",
    "tinker_time",
    "potion_overflow",
    "in_rl2_train",
    "in_rl2_gen",
    "in_rl2_eval",
    "event_combat",
    "roster_unmatched",
    "potion_generator",
    "goopy_amount",
    "deck_unmatched",
    "death_hp_unlogged",
    "abandoned_after_win",
    "belt_capped",
)

# ── quits (C# v0.107.1) ──
# Gold a floor can log before or during its fight: Maw Bank pays on entering
# the room (`MawBank.AfterRoomEntered`, GoldVar 12), Hand of Greed on a fatal
# hit. Rewards log `gold_gained` (`GoldReward.OnSelect`), `card_choices`
# (`CardReward`, on a pick, skip or reroll) and `relic_choices` only after a
# win. `potion_choices` is no evidence: in-combat generation logs picks too.
MAW_BANK, MAW_BANK_GOLD = "MAW_BANK", 12
HAND_OF_GREED = "HAND_OF_GREED"

SELECTION_RULE = (
    "Players ranked by the Wilson 95% lower bound of their A10 win rate on spire-codex "
    "Ironclad single-player standard runs from builds other than v0.107.1, abandoned runs "
    "counted as losses, at least 15 such runs; the top 25 form the pro cohort. Their "
    "v0.107.1 A10 runs (abandoned included) are the pro set."
)
SELECTION_RULE_V2 = (
    SELECTION_RULE + " A player is a username compared without case or surrounding space. "
    "Selection runs with modifiers or BaseLib cards, and, from builds up to v0.107.1, with "
    "an enchantment, card, relic or potion id the game does not define, are not counted."
)


def set_file(name: str, version: str) -> str:
    """A fight set's file name: `ironclad_<name>_<version>.jsonl`."""
    return f"ironclad_{name}_{version}.jsonl"


def manifest_file(version: str) -> str:
    """v1's MANIFEST is `MANIFEST.json` (frozen); later versions get `MANIFEST_<version>.json`."""
    return MANIFEST if version == "v1" else f"MANIFEST_{version}.json"


def index_file(version: str) -> str:
    """The `--private-dir` index a build scans into when `--index` is not given."""
    return "players_all.jsonl" if version == "v1" else f"players_all_{version}.jsonl"


def cohort_file(version: str) -> str:
    """The `--private-dir` pro-cohort mapping (usernames) a build writes."""
    return "pro_cohort.json" if version == "v1" else f"pro_cohort_{version}.json"


def _version(text: str) -> str:
    if not VERSION_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a version like v1, v2")
    return text


def player_id(username: str | None, file: str) -> str:
    """Pseudonymous stable player id; usernames never reach a data file."""
    if username:
        return "p_" + hashlib.sha256(username.encode("utf-8")).hexdigest()[:10]
    return "anon_" + hashlib.sha256(file.encode("utf-8")).hexdigest()[:10]


def canonical_username(username: str) -> str:
    """The player a username names from v2 on: case and surrounding space do not count.

    v1 keyed players on the exact string, and a rank-15 pro who uploaded under
    two spellings (one from 2026-06 on) was ranked on one and had the other's
    v0.107.1 runs in the dev pool.
    """
    return username.strip().casefold()


def _build_key(build: str | None) -> tuple[int, ...] | None:
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", build or "")
    return tuple(int(x) for x in m.groups()) if m else None


def selection_rejection(d: dict[str, Any], known: dict[str, frozenset[str]]) -> str | None:
    """Why a selection-build run does not count toward the ranking (v2 on), or None.

    Every build: `modifiers`, `modded_card` (BaseLib `save_dict` cards). Builds
    up to v0.107.1 also: `modded_enchantment` and `mod_rejection`'s unknown ids;
    the vendored tables (v0.107.1's ids plus older builds') do not know what a
    later build added, so those checks would drop its legitimate runs.
    """
    if d.get("modifiers"):
        return "modifiers"
    deck = (d.get("players") or [{}])[0].get("deck") or []
    if any(any(k.startswith("save_dict") for k in c) for c in deck):
        return "modded_card"
    build, cap = _build_key(d.get("build_id")), _build_key(BUILD)
    if build is None or cap is None or build > cap:
        return None
    if any(_bare(c["enchantment"]["id"]) not in ENCHANTMENTS for c in deck if c.get("enchantment")):
        return "modded_enchantment"
    return mod_rejection(d, known)


def _selection_job(path: str) -> tuple[str, str | None]:
    """Worker: a selection run's `selection_rejection`."""
    try:
        d = load_run(path)
    except (OSError, ValueError):
        return path, "unreadable"
    return path, selection_rejection(d, _known())


def _card_id(card: str) -> str:
    """`ID[+1][@ENCH:amt]` -> `ID`."""
    return card.split("@")[0].split("+")[0]


def reward_taken(ps: dict[str, Any], relics: list[str], deck: list[str]) -> bool:
    """The floor logs a post-combat reward, so its fight was won.

    A card or relic choice, or gold that Maw Bank's room-entry payment (when
    held) and Hand of Greed (when in the deck) do not explain. A run abandoned
    on the reward screen logs `current_hp` 0 for that floor.
    """
    if ps.get("card_choices") or ps.get("relic_choices"):
        return True
    gold = (ps.get("gold_gained") or 0) - (MAW_BANK_GOLD if MAW_BANK in relics else 0)
    return gold > 0 and not any(_card_id(c) == HAND_OF_GREED for c in deck)


def _loc_ids(name: str) -> frozenset[str]:
    with open(os.path.join(LOC_DIR, f"{name}.json"), encoding="utf-8") as f:
        return frozenset(k.split(".", 1)[0] for k in json.load(f))


def known_ids() -> dict[str, frozenset[str]]:
    """Card, relic and potion ids the game defines (its English localization).

    The vendored table is v0.107.1's plus keys only older builds had, so it is
    a superset of the build; every modded id seen in the corpus (Hextech
    `*_RUNE`, `*_FORGE`, ...) is outside it.
    """
    return {k: _loc_ids(k) for k in ("cards", "relics", "potions")}


def mod_rejection(d: dict[str, Any], known: dict[str, frozenset[str]]) -> str | None:
    """`modifiers` / `unknown_relic` / `unknown_card` / `unknown_potion`, or None.

    Checks the final inventory and every per-floor event that names an id, so
    a modded relic that was removed again still rejects the run.
    """
    if d.get("modifiers"):
        return "modifiers"
    p = d["players"][0]
    relics = {_bare(r["id"]) for r in p.get("relics") or []}
    cards = {_bare(c["id"]) for c in p.get("deck") or []}
    potions = {_bare(x["id"]) for x in p.get("potions") or []}
    for act in d["map_point_history"]:
        for pt in act:
            ps = pt["player_stats"][0]
            relics |= {
                _bare(c["choice"]) for c in ps.get("relic_choices") or [] if c.get("was_picked")
            }
            relics |= {_bare(r) for r in ps.get("relics_removed") or []}
            cards |= {_bare(c["id"]) for c in ps.get("cards_gained") or []}
            potions |= {
                _bare(c["choice"]) for c in ps.get("potion_choices") or [] if c.get("was_picked")
            }
            potions |= {_bare(x) for x in ps.get("potion_used") or []}
            potions |= {_bare(x) for x in ps.get("potion_discarded") or []}
    if relics - known["relics"]:
        return "unknown_relic"
    if cards - known["cards"]:
        return "unknown_card"
    if potions - known["potions"]:
        return "unknown_potion"
    return None


# ── potions ──────────────────────────────────────────────────────────────


def belt_timeline(
    d: dict[str, Any],
) -> tuple[dict[int, list[str]] | None, dict[int, int], str | None]:
    """The potion belt and its slot count entering every floor.

    Replays each floor as a multiset: belt + picked - drunk - discarded. The
    order inside a floor is not logged (Entropic Brew fills slots mid-fight,
    a reward lands in a slot a drink just freed), so only the floor's net
    effect is checked: every removal must be a held or picked potion, and the
    result must fit the slots. The run file's final belt and
    `max_potion_slot_count` then check the whole replay.

    Returns:
        `(belts, slots, error)`. `belts[floor]` is the belt (oldest first)
        before anything on that floor happened, or `belts` is None with
        `error` naming the first inconsistency. `slots[floor]` is always set.
    """
    p = d["players"][0]
    pts = [pt for act in d["map_point_history"] for pt in act]
    n_slots = BASE_POTION_SLOTS - (1 if (d.get("ascension") or 0) >= TIGHT_BELT_ASCENSION else 0)
    belt: list[str] = []
    belts: dict[int, list[str]] = {}
    slots: dict[int, int] = {}
    error = None
    for i, pt in enumerate(pts):
        floor = i + 1
        belts[floor] = list(belt)
        slots[floor] = n_slots
        ps = pt["player_stats"][0]
        for c in ps.get("relic_choices") or []:
            if c.get("was_picked"):
                n_slots += SLOT_RELICS.get(_bare(c["choice"]), 0)
        belt += [_bare(c["choice"]) for c in ps.get("potion_choices") or [] if c.get("was_picked")]
        for x in [
            _bare(x) for x in (ps.get("potion_used") or []) + (ps.get("potion_discarded") or [])
        ]:
            if x in belt:
                belt.remove(x)
            else:
                error = error or f"floor {floor}: {x} used but never held"
        if len(belt) > n_slots:
            error = error or f"floor {floor}: {len(belt)} potions in {n_slots} slots"
    final = sorted(_bare(x["id"]) for x in p.get("potions") or [])
    if error is None and sorted(belt) != final:
        error = "final belt mismatch"
    if error is None and n_slots != p.get("max_potion_slot_count"):
        error = f"slot count {n_slots} != {p.get('max_potion_slot_count')}"
    return (None if error else belts), slots, error


def pilot_potions(
    used: list[str],
    belt: list[str] | None,
    relics: list[str],
    n_slots: int,
    deck: Iterable[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """What the pilot is given at fight start, and what the rule did.

    Priority: drunk potions that were on the entry belt, then held automatic
    potions (Fairy in a Bottle) the human never had to drink, then drunk
    potions generated mid-fight. Two kinds of generated potion are left out:

    - With an in-combat generator present (Entropic Brew among the given
      potions, Alchemize in the deck, Delicate Frond held), all of them. The
      engine runs the generator for the pilot and rolls its own potions, so
      handing over what it made for the human as well would double it.
    - AnyTime potions (`ANYTIME_POTIONS`). One that was not on the entry belt
      was picked on this floor, most likely as the fight's reward, and drunk
      after it.

    With Petrified Toad held, one drunk rock is left out because the engine's
    toad supplies it, and the rock counts against the belt size.

    Args:
        used: The potions the human drank on the fight's floor.
        belt: The reconstructed entry belt, or None.
        relics: Relics held at fight entry.
        n_slots: The human's belt size.
        deck: The fight's deck (`ID[+1][@ENCH:amt]`), for Alchemize.

    Returns:
        `(potions, info)`; info keys: `overflow` (drank more than the belt
        holds), `toad_dedupe` (a rock was left to the toad), `toad_extra_rock`
        (the pilot starts with more rocks than the human held: the engine's
        3-slot combat belt leaves the toad a slot the human's full belt did
        not), `fairy` (a held Fairy is given though not drunk), `generator`
        (an in-combat generator is present), `generated_dropped` and
        `anytime_dropped` (drunk potions left out by the two rules above).
    """
    held = collections.Counter(belt or [])
    from_belt, generated = [], []
    for x in used:
        if held[x] > 0:
            from_belt.append(x)
            held[x] -= 1
        else:
            generated.append(x)
    auto = [x for x in (belt or []) if x in AUTO_POTIONS]
    auto = list((collections.Counter(auto) - collections.Counter(from_belt)).elements())
    generator = (
        BREW in from_belt
        or FROND in relics
        or any(c.split("@")[0].split("+")[0] == ALCHEMIZE for c in deck)
    )
    anytime = [x for x in generated if x in ANYTIME_POTIONS]
    # A drunk rock stays for the toad rule below (the engine's toad makes it).
    kept = [
        x
        for x in generated
        if (x == ROCK and TOAD in relics) or not (generator or x in ANYTIME_POTIONS)
    ]
    given = from_belt + auto + kept
    info = {
        "overflow": len(used) > n_slots,
        "toad_dedupe": False,
        "toad_extra_rock": False,
        "fairy": bool(auto),
        "generator": generator,
        "generated_dropped": len(generated) - len(kept),
        "anytime_dropped": len(anytime),
    }
    if TOAD in relics:
        if ROCK in given:
            given.remove(ROCK)
            # One slot stays free for the rock the engine's toad procures.
            given = given[: n_slots - 1]
            info["toad_dedupe"] = True
        else:
            given = given[:n_slots]
        if belt is not None:
            human_rocks = belt.count(ROCK) + (len(belt) < n_slots)
            info["toad_extra_rock"] = pilot_belt(given, relics, n_slots).count(ROCK) > human_rocks
    else:
        given = given[:n_slots]
    return given, info


def deck_change(forward: list[str], deck: list[str]) -> str:
    """How the replayed `deck` differs from the forward count rule's `forward`.

    Returns:
        `same`, `more_upgrades` / `fewer_upgrades` (the replay has more / fewer
        upgraded cards), `upgrade_moved` (same cards and upgrade count, the +1
        on a different enchanted copy) or `other`.
    """
    if sorted(forward) == sorted(deck):
        return "same"
    n_fwd, n_new = (sum("+1" in c for c in x) for x in (forward, deck))
    if n_new != n_fwd:
        return "more_upgrades" if n_new > n_fwd else "fewer_upgrades"

    def plain(x: list[str]) -> list[str]:
        return sorted(c.replace("+1", "") for c in x)

    return "upgrade_moved" if plain(forward) == plain(deck) else "other"


def pilot_belt(potions: list[str], relics: list[str], n_slots: int) -> list[str]:
    """The pilot's belt once combat starts: `potions` plus the engine toad's rock."""
    if TOAD in relics and len(potions) < n_slots:
        return [*potions, ROCK]
    return list(potions)


# ── rosters ──────────────────────────────────────────────────────────────


def human_roster(monster_ids: list[str]) -> list[str]:
    """The encounter's initial slots from a room's `monster_ids`.

    Pets are dropped wherever they appear (a pet relic adds its pet at combat
    start, before any summon). A summon is dropped when its summoner is
    already in the roster; it can only have been appended mid-fight.
    """
    out: list[str] = []
    for m in map(_bare, monster_ids):
        if m in PETS or (m in SUMMONED_BY and SUMMONED_BY[m] in out):
            continue
        out.append(m)
    return out


def _snake(engine_id: str) -> str:
    """Engine monster id (`LeafSlimeS`) -> run-file form (`LEAF_SLIME_S`)."""
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", engine_id)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return s.upper()


_PROBE_DECK = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]


def encounter_table(encounters: list[str]) -> dict[str, dict[str, Any]]:
    """Per encounter: the engine's own roster and whether the override is lossless.

    `lossless`: passing the engine's own roster back as `monsters=` rebuilds
    a state with the same `state_hash` on three seeds. Where it does not, the
    encounter's monsters carry spawn arguments (Kin Follower slots, Corpse
    Slug and Chomper openers, ...) that the override path drops.
    """
    import sts2_client
    import sts2_rs

    ver = sts2_client.GAME_VERSION
    out: dict[str, dict[str, Any]] = {}
    for enc in sorted(set(encounters)):
        row: dict[str, Any] = {"default": None, "randomized": False, "lossless": False}
        try:
            rosters, same = [], True
            for seed in (11, 12, 13):
                st = sts2_rs.build_scenario_state(
                    seed,
                    CHARACTER,
                    _PROBE_DECK,
                    50,
                    80,
                    [],
                    enc,
                    ascension=ASCENSION,
                    game_version=ver,
                )
                roster = [_snake(e["id"]) for e in st.get_state()["enemies"]]
                again = sts2_rs.build_scenario_state(
                    seed,
                    CHARACTER,
                    _PROBE_DECK,
                    50,
                    80,
                    [],
                    enc,
                    ascension=ASCENSION,
                    monsters=roster,
                    game_version=ver,
                )
                same &= st.state_hash() == again.state_hash()
                rosters.append(roster)
            row["default"] = rosters[0]
            row["randomized"] = any(r != rosters[0] for r in rosters)
            row["lossless"] = same
        except (ValueError, RuntimeError) as e:
            row["error"] = str(e)
        out[enc] = row
    return out


def resolve_roster(
    roster: list[str] | None, encounter: str, table: dict[str, dict[str, Any]]
) -> tuple[list[str] | None, str]:
    """`(monsters override, status)` for one fight.

    Status `default` (the engine's own roster already is the human's: no
    override), `override` (pass the human's roster), or `unmatched` (no
    roster, an encounter the engine lacks, or an override that would drop
    spawn args). The override is used only in an encounter `encounter_table`
    found lossless, and only with spawn-arg kinds that encounter's own roster
    holds (where that check covered them).
    """
    row = table.get(encounter)
    if not roster or row is None or row.get("default") is None:
        return None, "unmatched"
    if not row["randomized"] and roster == row["default"]:
        return None, "default"
    if row["lossless"] and all(m in row["default"] or m not in SPAWN_ARG_KINDS for m in roster):
        return roster, "override"
    return None, "unmatched"


# ── one run ──────────────────────────────────────────────────────────────


def bench_fights(
    d: dict[str, Any],
    path: str,
    cohort: str,
    known: dict[str, frozenset[str]],
    *,
    pid: str | None = None,
    frozen_v1: bool = False,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    """A run's bench records (before the engine checks), or why it is dropped.

    Args:
        d: The parsed run.
        path: Its path.
        cohort: `pro` / `other`, copied into each record.
        known: `known_ids()`.
        pid: The run's player id (default `player_id` of its own username).
        frozen_v1: v1's rules: no reward check on quits, `DeckReplay`'s
            frozen-v1 replay.

    Returns:
        `(records, rejection, info)`. Records carry private keys the engine
        step consumes: `_roster` (the human's filtered roster or None),
        `_potion` (what `pilot_potions` did), `_hp0_unclassified` and
        `_deck_vs_forward` (`deck_change` against the forward count rule).
    """
    why = run_rejection(d, ASCENSION, BUILD, CHARACTER, include_abandoned=True)
    why = why or mod_rejection(d, known)
    if why:
        return [], why, {}
    file = os.path.basename(path)
    pts = [pt for act in d["map_point_history"] for pt in act]
    belts, slots, belt_error = belt_timeline(d)
    abandoned = bool(d.get("was_abandoned"))
    pid = pid or player_id(d.get("username"), file)
    # The forward count rule's decks, for the MANIFEST's `decks` counts.
    forward = {
        f["floor"]: f["deck"]
        for f in extract_fights(
            d, path, ASCENSION, CHARACTER, count_downgrades=True, event_combat=True
        )
    }
    records = []
    for f in extract_fights(
        d,
        path,
        ASCENSION,
        CHARACTER,
        relic_history=True,
        event_combat=True,
        deck_history=True,
        deck_replay_v1=frozen_v1,
    ):
        floor = f["floor"]
        pt = pts[floor - 1]
        room = next(r for r in pt.get("rooms") or [] if r.get("room_type") == f["room_type"])
        belt = None if belts is None else belts[floor]
        # The engine belt takes any size, as C# `MaxPotionCount` does (GH #1094).
        n_slots = slots[floor]
        potions, pinfo = pilot_potions(f["potions"], belt, f["relics"], n_slots, f["deck"])
        hp_after = f["human_hp_after"]
        dmg = f["human_damage_taken"] or 0
        healed = f["human_hp_healed"] or 0
        last = floor == len(pts)
        lost_all = dmg >= (f["hp_before"] or 0) + healed
        hp0_end = bool(abandoned and last and hp_after == 0 and not lost_all)
        # Won, then abandoned on the reward screen: the floor's HP reads 0.
        after_win = (
            hp0_end
            and not frozen_v1
            and reward_taken(pt["player_stats"][0], f["relics"], f["deck"])
        )
        if after_win:
            hp_after = (f["hp_before"] or 0) - dmg + healed
        quit_ = hp0_end and not after_win
        died = bool(not abandoned and hp_after == 0)
        flags = {
            "enchanted": f["n_enchanted"] > 0,
            "fresh_relic_counters": any(r in COUNTER_RELICS for r in f["relics"]),
            "fur_coat": "FUR_COAT" in f["relics"],
            "tinker_time": any(c.split("@")[0].split("+")[0] in TINKER_CARDS for c in f["deck"]),
            "potion_overflow": pinfo["overflow"],
            "event_combat": f["event_combat"],
            "potion_generator": pinfo["generator"],
            "goopy_amount": f["goopy_estimated"],
            "deck_unmatched": bool(f["deck_notes"]),
            # The Insatiable's Sandpit kills through `CreatureCmd.Kill`, which
            # `DamageTaken` does not count (`report` scores any death as all
            # HP lost, so L_h is unaffected).
            "death_hp_unlogged": died and not lost_all,
            "abandoned_after_win": after_win,
            # Always False since GH #1094 lifted the 5-slot engine cap; kept so
            # the flag columns of earlier builds still line up.
            "belt_capped": False,
        }
        mids = room.get("monster_ids")
        records.append(
            {
                "key": f"{file}:{floor}",
                "file": file,
                "floor": floor,
                "act": f["act"],
                "room_type": f["room_type"],
                "encounter": f["encounter"],
                "character": CHARACTER,
                "ascension": ASCENSION,
                "player": pid,
                "cohort": cohort,
                "run_won": bool(d.get("win")),
                "run_abandoned": abandoned,
                "deck": f["deck"],
                "relics": f["relics"],
                "potions": potions,
                "belt": belt,
                "potion_slots": n_slots,
                "belt_unused": bool(
                    collections.Counter(belt or [])
                    - collections.Counter(pilot_belt(potions, f["relics"], n_slots))
                ),
                "monsters": None,
                "hp_before": f["hp_before"],
                "max_hp": f["max_hp"],
                "n_enchanted": f["n_enchanted"],
                "human": {
                    "hp_after": hp_after,
                    "damage_taken": f["human_damage_taken"],
                    "hp_healed": f["human_hp_healed"],
                    "max_hp_gained": f["human_max_hp_gained"],
                    "max_hp_lost": f["human_max_hp_lost"],
                    "turns": f["human_turns"],
                    "died": died,
                    "quit": quit_,
                },
                "flags": [k for k in FLAGS if flags.get(k)],
                "_roster": human_roster(mids) if mids else None,
                "_potion": pinfo | {"used": len(f["potions"])},
                "_hp0_unclassified": hp_after == 0 and not died and not quit_,
                "_deck_vs_forward": deck_change(forward[floor], f["deck"]),
            }
        )
    return records, None, {"belt_error": belt_error, "player": pid}


def _extract_job(job: tuple[str, str, frozenset[str] | None, str, bool]) -> dict[str, Any]:
    """Worker: one run file -> its bench records (filtered to `keys` if given).

    `job`: `(path, cohort, keys, player id, frozen_v1)`.
    """
    path, cohort, keys, pid, frozen_v1 = job
    try:
        d = load_run(path)
    except (OSError, ValueError) as e:
        return {"path": path, "reject": "unreadable", "error": repr(e), "records": []}
    records, why, info = bench_fights(d, path, cohort, _known(), pid=pid, frozen_v1=frozen_v1)
    if keys is not None:
        records = [r for r in records if r["key"] in keys]
    return {
        "path": path,
        "reject": why,
        "records": records,
        "username": d.get("username"),
        **info,
    }


_KNOWN: dict[str, frozenset[str]] | None = None


def _known() -> dict[str, frozenset[str]]:
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = known_ids()
    return _KNOWN


def _stage_job(path: str) -> str:
    """Worker: the first filter stage that drops a run, or `kept`/`kept_belt_null`."""
    try:
        d = load_run(path)
    except (OSError, ValueError):
        return "unreadable"
    why = run_rejection(d, ASCENSION, BUILD, CHARACTER, include_abandoned=True)
    why = why or mod_rejection(d, _known())
    if why:
        return why
    return "kept" if belt_timeline(d)[2] is None else "kept_belt_null"


def _build_reason(e: Exception) -> str:
    m = re.search(r"unknown (\w+) id", str(e))
    if m:
        return f"unknown_{m.group(1)}"
    if "invalid hp" in str(e):
        return "invalid_hp"
    return type(e).__name__


def _verify_job(job: tuple[dict[str, Any], list[str] | None]) -> dict[str, Any]:
    """Worker: build the fight as the bench runner will (rep-0 seed).

    Tries the roster override first; if the engine rejects it the fight is
    built on the encounter's own roster and flagged `roster_unmatched`.
    Returns `{"key", "ok", "reason", "detail", "monsters", "fallback"}`.
    """
    import sts2_client
    import sts2_rs

    rec, monsters = job
    seed = _fight_seed(rec["key"]) * 100
    ver = sts2_client.GAME_VERSION

    def attempt(mons: list[str] | None) -> None:
        if not rec["hp_before"] or not rec["max_hp"]:
            raise ValueError("invalid hp: missing")
        state = sts2_rs.build_scenario_state(
            seed,
            rec["character"],
            rec["deck"],
            int(rec["hp_before"]),
            int(rec["max_hp"]),
            rec["relics"],
            rec["encounter"],
            ascension=int(rec["ascension"]),
            monsters=mons,
            potions=rec["potions"],
            game_version=ver,
        )
        rm = sts2_rs.RunManager(
            seed=seed,
            character=rec["character"],
            ascension_level=int(rec["ascension"]),
            game_version=ver,
        )
        rm.debug_enter_combat(state, rec["room_type"])
        if rm.phase() != "combat" or rm.is_over():
            raise RuntimeError(f"not in combat after entry (phase {rm.phase()})")

    out = {"key": rec["key"], "ok": True, "reason": None, "monsters": monsters, "fallback": None}
    try:
        attempt(monsters)
        return out
    except Exception as e:  # the engine raises ValueError; anything else is a reason too
        if monsters is None:
            return out | {"ok": False, "reason": _build_reason(e), "detail": str(e)[:200]}
        first = e
    try:
        attempt(None)
        return out | {"monsters": None, "fallback": _build_reason(first)}
    except Exception as e:
        return out | {"ok": False, "reason": _build_reason(e), "detail": str(e)[:200]}


# ── rollout_learn pools (in_rl2_*) ───────────────────────────────────────


def rl2_pools(hbank: list[dict[str, Any]], recipe_dir: str) -> tuple[set[str], set[str], set[str]]:
    """Keys of hbank fights in the gen1 imitation data and the rl1/rl2 start pools.

    Reproduces `rollout_learn gen` / `rl` with the recorded args
    (`logs/2026-10-18-rollout-net-recipe/run_rl{1,2}.sh`, README): holdout =
    boss800's runs plus every run with a Waterfall Giant fight; gen rooms
    monster/elite/boss; rl rooms boss/elite, `--n-states 12000`, seeds 0 and 1.

    Returns:
        `(gen_keys, rl_keys, holdout_files)`, before the buildability check
        (`gen`/`rl` skip fights the engine cannot build).
    """
    holdout = {
        json.loads(line)["file"]
        for line in open(os.path.join(recipe_dir, "boss800.jsonl"), encoding="utf-8")
    }
    holdout |= {f["file"] for f in hbank if f["encounter"] == "WATERFALL_GIANT_BOSS"}

    def key(f: dict[str, Any]) -> str:
        return f"{f['file']}:{f['floor']}"

    gen = {
        key(f)
        for f in hbank
        if f["file"] not in holdout and f["room_type"] in {"monster", "elite", "boss"}
    }
    rl: set[str] = set()
    for seed in (0, 1):
        pool = [
            f for f in hbank if f["file"] not in holdout and f["room_type"] in {"boss", "elite"}
        ]
        random.Random(seed).shuffle(pool)
        rl |= {key(f) for f in pool[:12000]}
    return gen, rl, holdout


def _old_buildable_job(f: dict[str, Any]) -> tuple[str, bool]:
    """Worker: does an old-schema bank row build (as `rollout_learn` built it)?"""
    import sts2_client
    import sts2_rs

    key = f"{f['file']}:{f['floor']}"
    try:
        sts2_rs.build_scenario_state(
            _fight_seed(key) * 100 + 77,
            "Ironclad",
            f["deck"],
            int(f["hp_before"]),
            int(f["max_hp"]),
            f["relics"],
            f["encounter"],
            ascension=int(f["ascension"]),
            potions=f["potions"][:3],
            game_version=sts2_client.GAME_VERSION,
        )
        return key, True
    except ValueError:
        return key, False


# ── driver ───────────────────────────────────────────────────────────────


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kish_n_eff(weights: list[int]) -> float:
    """Kish effective sample size `(sum w)^2 / sum w^2`."""
    s2 = sum(w * w for w in weights)
    return sum(weights) ** 2 / s2 if s2 else 0.0


def set_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts the MANIFEST records for one fight set."""
    per_player = collections.Counter(r["player"] for r in records)
    return {
        "fights": len(records),
        "runs": len({r["file"] for r in records}),
        "players": len(per_player),
        "by_room": dict(collections.Counter(r["room_type"] for r in records)),
        "by_act": {
            str(k): v for k, v in sorted(collections.Counter(r["act"] for r in records).items())
        },
        "by_act_room": {
            f"{a}/{rt}": n
            for (a, rt), n in sorted(
                collections.Counter((r["act"], r["room_type"]) for r in records).items()
            )
        },
        "human_deaths": {
            rt: sum(r["human"]["died"] for r in records if r["room_type"] == rt)
            for rt in ("monster", "elite", "boss")
        },
        "quits": {
            rt: sum(r["human"]["quit"] for r in records if r["room_type"] == rt)
            for rt in ("monster", "elite", "boss")
        },
        "runs_abandoned": len({r["file"] for r in records if r["run_abandoned"]}),
        "runs_won": len({r["file"] for r in records if r["run_won"]}),
        "flags": {k: sum(k in r["flags"] for r in records) for k in FLAGS},
        "belt_null": sum(r["belt"] is None for r in records),
        "belt_unused": sum(r["belt_unused"] for r in records),
        "monsters_override": sum(r["monsters"] is not None for r in records),
        "fights_per_player": dict(sorted(per_player.items(), key=lambda kv: (-kv[1], kv[0]))),
        "kish_n_eff_players": round(kish_n_eff(list(per_player.values())), 2),
    }


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", _PYCLI, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _belt_error_kind(error: str) -> str:
    """`belt_timeline`'s error text without the floor and potion specifics."""
    for needle, kind in (
        ("never held", "used_never_held"),
        ("slots", "over_capacity"),
        ("slot count", "slot_count_mismatch"),
        ("final belt", "final_belt_mismatch"),
    ):
        if needle in error:
            return kind
    return error


def _log(msg: str) -> None:
    print(f"[combat_bench build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.combat_bench build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=human_players.DEFAULT_ROOT, help="spire-codex files root")
    ap.add_argument(
        "--private-dir",
        required=True,
        help="outside the repo: the username index and the pro-cohort mapping go here",
    )
    ap.add_argument(
        "--index",
        default=None,
        help="reuse this `human_players index --include-abandoned` JSONL (default: scan "
        "the corpus and write <private-dir>/players_all.jsonl, players_all_<version>.jsonl "
        "for a version after v1)",
    )
    ap.add_argument("--hbank", required=True, help="the gen1/rl1/rl2 training bank (old schema)")
    ap.add_argument("--recipe-dir", default=RECIPE_DIR)
    ap.add_argument("--out-dir", default=DATA_DIR)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite sets or a MANIFEST already in --out-dir (the v1 files there are frozen)",
    )
    ap.add_argument(
        "--bench-dir",
        default=DATA_DIR,
        help="earlier versions' sets: after v1, pro sets leave out every fight of its dev panels "
        "(ironclad_dev_v*.jsonl, screened on), and the MANIFEST counts what each pro set "
        "shares with its dev sets (ironclad_dev*_v*.jsonl)",
    )
    ap.add_argument("--procs", type=int, default=40)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-runs", type=int, default=15)
    ap.add_argument(
        "--version",
        type=_version,
        default="v1",
        help="output version. v1 writes the frozen names (ironclad_pro_v1.jsonl, "
        "ironclad_devpool_v1.jsonl, MANIFEST.json); vN writes ironclad_<set>_vN.jsonl and "
        "MANIFEST_vN.json beside them, and scans into <private-dir>/players_all_vN.jsonl",
    )
    ap.add_argument(
        "--extra-top",
        type=int,
        nargs="*",
        default=[],
        metavar="N",
        help="also write the pro set of the top-N cohort (N > --top) as "
        "ironclad_proN_<version>.jsonl, recorded in the same MANIFEST; the dev pool still "
        "drops the --top cohort's runs only",
    )
    args = ap.parse_args(argv)
    extra = sorted(args.extra_top)
    if len(set(extra)) != len(extra) or any(n <= args.top for n in extra):
        print(f"--extra-top {args.extra_top}: distinct values above --top", file=sys.stderr)
        return 2
    pro_name, dev_name = set_file("pro", args.version), set_file("devpool", args.version)
    extra_names = {n: set_file(f"pro{n}", args.version) for n in extra}
    frozen_v1 = args.version == "v1"
    own = [pro_name, dev_name, *extra_names.values(), manifest_file(args.version)]
    there = [n for n in own if os.path.exists(os.path.join(args.out_dir, n))]
    if there and not args.force:
        print(
            f"{args.out_dir} already holds {there}; build into a fresh --out-dir, or pass --force",
            file=sys.stderr,
        )
        return 2

    repo = os.path.abspath(os.path.join(_PYCLI, ".."))
    priv = os.path.abspath(args.private_dir)
    if priv == repo or priv.startswith(repo + os.sep):
        print(f"--private-dir must be outside the repo ({repo})", file=sys.stderr)
        return 2
    os.makedirs(priv, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    # Earlier versions' dev sets (never this build's own names): their dev
    # panels' fights were screened on, so a later pro set leaves them out.
    ref_sets = {
        os.path.basename(p): [json.loads(line) for line in open(p, encoding="utf-8")]
        for p in sorted(glob.glob(os.path.join(args.bench_dir, "ironclad_dev*_v*.jsonl")))
        if os.path.basename(p) not in own
    }
    screened = (
        {}
        if frozen_v1
        else {
            n: {r["key"] for r in recs}
            for n, recs in ref_sets.items()
            if re.fullmatch(r"ironclad_dev_v\d+\.jsonl", n)
        }
    )
    screened_keys = set().union(*screened.values()) if screened else set()

    # 1. index (usernames: private). A scan of the whole corpus takes seconds.
    index_path = args.index or os.path.join(priv, index_file(args.version))
    scan: dict[str, int] = {}
    if args.index:
        rows = [json.loads(line) for line in open(index_path, encoding="utf-8")]
        _log(f"index: {len(rows)} rows from {index_path}")
    else:
        rows, reasons = human_players.scan_corpus(args.root, CHARACTER, args.procs, True)
        scan = dict(reasons)
        with open(index_path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        _log(f"index: scanned {reasons['files']} files -> {len(rows)} rows ({index_path})")
    if not rows or "abandoned" not in rows[0]:
        print("the index must be built with --include-abandoned", file=sys.stderr)
        return 2

    # 2. pro cohort: abandoned runs count as losses. A player is a username
    # key (v1: the exact string); the pseudonym hashes its first-seen spelling.
    def canon(username: str) -> str:
        return username if frozen_v1 else canonical_username(username)

    spellings: dict[str, set[str]] = collections.defaultdict(set)
    first_seen: dict[str, tuple[int, str]] = {}
    for r in rows:
        if r["username"]:
            k = canon(r["username"])
            spellings[k].add(r["username"])
            seen = (r.get("start_time") or 0, r["username"])
            first_seen[k] = min(first_seen.get(k, seen), seen)
    shown = {k: u for k, (_, u) in first_seen.items()}

    def pid_of(username: str | None, file: str) -> str:
        return player_id(shown.get(canon(username), username) if username else None, file)

    # A row's player id -> its player key, for overlaps with sets of earlier
    # versions (v1 gave each spelling its own id).
    id_key = {player_id(u, ""): k for k, us in spellings.items() for u in us}
    rank_rows = [r | {"username": canon(r["username"])} for r in rows if r["username"]]
    ranking_excluded: dict[str, int] = {}
    if not frozen_v1:
        sel = [r["file"] for r in rank_rows if r["ascension"] == ASCENSION and r["build"] != BUILD]
        with get_context("spawn").Pool(args.procs) as pool:
            sel_why = dict(pool.imap(_selection_job, sel, chunksize=32))
        ranking_excluded = dict(collections.Counter(w for w in sel_why.values() if w))
        rank_rows = [r for r in rank_rows if not sel_why.get(r["file"])]
        _log(f"selection runs left out of the ranking: {ranking_excluded}")
    ranked, ev = human_players.rank_players(rank_rows, [ASCENSION], BUILD, args.min_runs, True)
    pros = [u for u, *_ in ranked[: args.top]]
    # The widest cohort extracted; each --extra-top set is a prefix of the ranking.
    n_cohort = max([args.top, *extra])
    old_pros = {
        canon(u)
        for u in json.load(open(os.path.join(args.recipe_dir, "pros25.json"), encoding="utf-8"))
    }
    with open(
        os.path.join(priv, cohort_file(args.version)), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(
            [
                {
                    "username": shown[u],
                    **(
                        {"other_spellings": sorted(spellings[u] - {shown[u]})}
                        if len(spellings[u]) > 1
                        else {}
                    ),
                    "player": player_id(shown[u], ""),
                    "sel_wins": k,
                    "sel_runs": n,
                    "wilson_lb": round(lb, 4),
                    "eval_wins_runs": ev.get(u, [0, 0]),
                    "in_pros25": u in old_pros,
                }
                for u, k, n, lb in ranked[:n_cohort]
            ],
            f,
            indent=1,
        )
    _log(f"pro cohort: {len(pros)} players, {len(set(pros) & old_pros)} also in pros25.json")
    for n in extra:
        _log(f"extra cohort top {n}: {len(ranked[:n])} players")

    # 3. extract: pro runs and the dev-pool runs.
    by_base = {os.path.basename(r["file"]): r for r in rows}
    eval_rows = [r for r in rows if r["build"] == BUILD and r["ascension"] == ASCENSION]
    with get_context("spawn").Pool(args.procs) as pool:
        corpus_stages = collections.Counter(
            pool.imap(_stage_job, [r["file"] for r in eval_rows], chunksize=16)
        )
    _log(f"{BUILD} A{ASCENSION} runs by filter stage: {dict(corpus_stages)}")
    pro_set = set(pros)

    def is_pro(username: str | None) -> bool:
        return bool(username) and canon(username) in pro_set

    def cohort_paths(top: int) -> set[str]:
        users = {u for u, *_ in ranked[:top]}
        return {r["file"] for r in eval_rows if r["username"] and canon(r["username"]) in users}

    pro_paths = sorted(cohort_paths(n_cohort))
    user_of = {r["file"]: r["username"] for r in eval_rows}
    dev_keys: set[str] = set()
    source_counts = {}
    for name in DEV_SOURCES:
        ks = {
            f"{x['file']}:{x['floor']}"
            for x in map(json.loads, open(os.path.join(args.recipe_dir, name), encoding="utf-8"))
        }
        source_counts[name] = len(ks)
        dev_keys |= ks
    dev_files = sorted({k.rsplit(":", 1)[0] for k in dev_keys})
    missing_dev = [f for f in dev_files if f not in by_base]
    jobs = [(p, "pro", None, pid_of(user_of[p], os.path.basename(p)), frozen_v1) for p in pro_paths]
    keep = frozenset(dev_keys)
    jobs += [
        (by_base[f]["file"], "other", keep, pid_of(by_base[f]["username"], f), frozen_v1)
        for f in dev_files
        if f in by_base
    ]
    results = []
    with get_context("spawn").Pool(args.procs) as pool:
        for i, res in enumerate(pool.imap(_extract_job, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 500 == 0:
                _log(f"extract {i + 1}/{len(jobs)} runs")
    cohort_res, dev_res = results[: len(pro_paths)], results[len(pro_paths) :]
    top_paths = cohort_paths(args.top)
    pro_res = [r for r in cohort_res if r["path"] in top_paths]
    extra_res = {
        n: [r for r in cohort_res if r["path"] in paths]
        for n, paths in ((n, cohort_paths(n)) for n in extra)
    }
    dev_pro_runs = [r for r in dev_res if is_pro(r.get("username"))]
    dev_res = [r for r in dev_res if not is_pro(r.get("username"))]
    _log(
        f"extract: {len(pro_res)} pro runs, {len(dev_res)} dev runs "
        f"({len(dev_pro_runs)} pro-player runs left out)"
    )

    def stage_counts(res: list[dict[str, Any]]) -> dict[str, Any]:
        rej = collections.Counter(r["reject"] for r in res if r["reject"])
        kept = [r for r in res if not r["reject"]]
        return {
            "runs": len(res),
            "rejected": dict(rej),
            "kept_runs": len(kept),
            "belt_reconstruction_failed_runs": sum(bool(r.get("belt_error")) for r in kept),
            "belt_errors": dict(
                collections.Counter(
                    _belt_error_kind(r["belt_error"]) for r in kept if r.get("belt_error")
                )
            ),
            "extracted_fights": sum(len(r["records"]) for r in kept),
        }

    stages = {"pro": stage_counts(pro_res), "devpool": stage_counts(dev_res)}
    stages |= {f"pro{n}": stage_counts(res) for n, res in extra_res.items()}

    # 4. rosters + engine verification.
    sets = {
        "pro": [rec for r in pro_res for rec in r["records"]],
        "devpool": [rec for r in dev_res for rec in r["records"]],
    }
    # Own copies: finalizing a record pops its private keys.
    sets |= {
        f"pro{n}": [copy.deepcopy(rec) for r in res for rec in r["records"]]
        for n, res in extra_res.items()
    }
    import sts2_rs

    table = encounter_table([rec["encounter"] for recs in sets.values() for rec in recs])
    _log(
        f"encounters: {len(table)}, lossy override: "
        f"{sorted(e for e, row in table.items() if row['default'] and not row['lossless'])}"
    )
    verify_jobs = []
    roster_status: dict[str, str] = {}
    for recs in sets.values():
        for rec in recs:
            if rec["key"] in roster_status:  # a wider cohort's copy of a queued fight
                continue
            mons, status = resolve_roster(rec["_roster"], rec["encounter"], table)
            roster_status[rec["key"]] = status
            verify_jobs.append((rec, mons))
    verdicts = {}
    with get_context("spawn").Pool(args.procs) as pool:
        for i, v in enumerate(pool.imap_unordered(_verify_job, verify_jobs, chunksize=8)):
            verdicts[v["key"]] = v
            if (i + 1) % 2000 == 0:
                _log(f"verify {i + 1}/{len(verify_jobs)} fights")

    # 5. in_rl2_* flags from the training bank.
    hbank = [json.loads(line) for line in open(args.hbank, encoding="utf-8")]
    gen_keys, rl_keys, _ = rl2_pools(hbank, args.recipe_dir)
    ours = {rec["key"] for recs in sets.values() for rec in recs}
    need = ours & (gen_keys | rl_keys)
    to_check = [f for f in hbank if f"{f['file']}:{f['floor']}" in need]
    with get_context("spawn").Pool(args.procs) as pool:
        old_ok = dict(pool.imap_unordered(_old_buildable_job, to_check, chunksize=16))
    gen_keys = {k for k in gen_keys if old_ok.get(k)}
    rl_keys = {k for k in rl_keys if old_ok.get(k)}

    # 6. finalize records.
    manifest_sets: dict[str, Any] = {}
    unbuildable: dict[str, Any] = {}
    rosters: dict[str, Any] = {}
    potion_stats: dict[str, Any] = {}
    deck_stats: dict[str, Any] = {}
    event_stats: dict[str, Any] = {}
    held_out: set[str] = set()
    written: dict[str, list[dict[str, Any]]] = {}
    screened_dropped: dict[str, int] = {}
    outputs = [("pro", pro_name, "pro"), ("devpool", dev_name, "other")]
    outputs += [(f"pro{n}", extra_names[n], "pro") for n in extra]
    for name, fname, cohort in outputs:
        out, bad, bad_detail = [], collections.Counter(), collections.Counter()
        rstat = collections.Counter()
        pstat = collections.Counter()
        dstat = collections.Counter()
        evt = collections.Counter()
        hp0_unclassified = 0
        for rec in sorted(sets[name], key=lambda r: (r["file"], r["floor"])):
            if cohort == "pro" and rec["key"] in screened_keys:
                screened_dropped[fname] = screened_dropped.get(fname, 0) + 1
                continue
            v = verdicts[rec["key"]]
            if "event_combat" in rec["flags"]:
                evt[f"{rec['encounter']}:{'built' if v['ok'] else v['reason']}"] += 1
            if not v["ok"]:
                bad[v["reason"]] += 1
                bad_detail[v.get("detail", "")[:80]] += 1
                continue
            status = roster_status[rec["key"]]
            if v["fallback"]:
                status = "unmatched"
                rstat[f"override_rejected:{v['fallback']}"] += 1
            rstat[status] += 1
            rec["monsters"] = v["monsters"]
            flags = set(rec["flags"])
            if status == "unmatched":
                flags.add("roster_unmatched")
            if rec["key"] in rl_keys:
                flags.add("in_rl2_train")
            if rec["key"] in gen_keys:
                flags.add("in_rl2_gen")
            # The recipe's RL eval and screen sets; every dev-pool fight is one.
            if not frozen_v1 and cohort == "pro" and rec["key"] in dev_keys:
                flags.add("in_rl2_eval")
            rec["flags"] = [k for k in FLAGS if k in flags]
            pinfo = rec.pop("_potion")
            for k in ("overflow", "toad_dedupe", "toad_extra_rock", "fairy", "generator"):
                pstat[k] += bool(pinfo[k])
            for k in ("generated_dropped", "anytime_dropped"):
                pstat[k] += pinfo[k]
                pstat[f"{k}_fights"] += bool(pinfo[k])
            pstat["given_total"] += len(rec["potions"])
            pstat["used_total"] += pinfo["used"]
            hp0_unclassified += rec.pop("_hp0_unclassified")
            dstat[rec.pop("_deck_vs_forward")] += 1
            rec.pop("_roster")
            out.append(rec)
        path = os.path.join(args.out_dir, fname)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for rec in out:
                f.write(json.dumps(rec) + "\n")
        held_out |= {rec["file"] for rec in out}
        written[name] = out
        manifest_sets[fname] = {"sha256": _sha256(path), "cohort": cohort}
        manifest_sets[fname] |= set_summary(out)
        manifest_sets[fname]["hp0_neither_died_nor_quit"] = hp0_unclassified
        unbuildable[fname] = {"by_reason": dict(bad), "top_details": bad_detail.most_common(12)}
        rosters[fname] = dict(rstat)
        potion_stats[fname] = dict(pstat)
        deck_stats[fname] = dict(dstat)
        event_stats[fname] = dict(evt)
        _log(f"{fname}: {len(out)} fights written, {sum(bad.values())} unbuildable {dict(bad)}")

    # 7. MANIFEST.
    pro_summary = manifest_sets[pro_name]

    def player_key(r: dict[str, Any]) -> str:
        return id_key.get(r["player"], r["player"])

    def overlap(recs: list[dict[str, Any]], ref: list[dict[str, Any]]) -> dict[str, Any]:
        """What a pro set shares with one dev set, by player, run and fight."""
        players = {player_key(r) for r in recs} & {player_key(r) for r in ref}
        runs = {r["file"] for r in recs} & {r["file"] for r in ref}
        rest = collections.Counter(r["player"] for r in recs if player_key(r) not in players)
        return {
            "players": len(players),
            "runs": len(runs),
            "fights": len({r["key"] for r in recs} & {r["key"] for r in ref}),
            "fights_of_shared_players": sum(player_key(r) in players for r in recs),
            "fights_in_shared_runs": sum(r["file"] in runs for r in recs),
            "kish_n_eff_players_without_shared": round(kish_n_eff(list(rest.values())), 2),
        }

    def devpool_overlap(name: str) -> dict[str, dict[str, Any]]:
        """Per dev set (this build's dev pool, then `--bench-dir`'s), what `name` shares."""
        refs = {dev_name: written["devpool"], **ref_sets}
        return {ref: overlap(written[name], recs) for ref, recs in refs.items()}

    def eval_tally(users: list[str]) -> dict[str, Any]:
        return {
            "overlap_with_pros25": len(set(users) & old_pros),
            "players_with_eval_runs": sum(1 for u in users if ev.get(u, [0, 0])[1]),
            "eval_runs_wins": [
                sum(ev.get(u, [0, 0])[0] for u in users),
                sum(ev.get(u, [0, 0])[1] for u in users),
            ],
        }

    manifest = {
        "schema": "combat_bench.manifest.v1",
        "version": args.version,
        "issue": "GH #830",
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git_head(),
        "sts2_rs_src_hash": sts2_rs.src_hash(),
        "character": CHARACTER,
        "ascension": ASCENSION,
        "build": BUILD,
        "corpus": {
            "root": os.path.join(os.path.basename(os.path.normpath(args.root)), CHARACTER.lower()),
            "scan": scan or "index reused (see --index); scan counts in the run that built it",
            "index_rows": len(rows),
            "index_rows_abandoned": sum(bool(r.get("abandoned")) for r in rows),
            # Only runs with a username can be ranked or be pro runs; files
            # fetched from spire-codex after v1's scrape carry none.
            "index_rows_named": sum(bool(r["username"]) for r in rows),
            f"{BUILD}_A{ASCENSION}_runs": len(eval_rows),
            f"{BUILD}_A{ASCENSION}_runs_abandoned": sum(
                bool(r.get("abandoned")) for r in eval_rows
            ),
            f"{BUILD}_A{ASCENSION}_runs_named": sum(bool(r["username"]) for r in eval_rows),
            f"{BUILD}_A{ASCENSION}_runs_by_stage": dict(corpus_stages),
        },
        "rules": "v1 (frozen)" if frozen_v1 else list(V2_RULES),
        "selection": {
            "rule": SELECTION_RULE if frozen_v1 else SELECTION_RULE_V2,
            "top": args.top,
            "min_runs": args.min_runs,
            "abandoned_as_losses": True,
            "ranked_players": len(ranked),
            # Usernames that more than one spelling of joins into one player.
            "players_with_several_spellings": sum(len(v) > 1 for v in spellings.values()),
            "cohort_players_with_several_spellings": sum(
                len(spellings[u]) > 1 for u, *_ in ranked[:n_cohort]
            ),
            "ranking_runs_excluded": ranking_excluded,
            **eval_tally(pros),
            "kish_n_eff_players": pro_summary["kish_n_eff_players"],
            "username_mapping": f"kept outside the repo (--private-dir/{cohort_file(args.version)})",
            "devpool_overlap": devpool_overlap("pro"),
            "extra_tops": {
                str(n): {
                    "set": extra_names[n],
                    **eval_tally([u for u, *_ in ranked[:n]]),
                    "kish_n_eff_players": manifest_sets[extra_names[n]]["kish_n_eff_players"],
                    # The dev pool drops only the --top cohort's runs.
                    "devpool_overlap": devpool_overlap(f"pro{n}"),
                }
                for n in extra
            },
        },
        "screened": {
            "rule": "pro sets leave out every fight of an earlier version's dev panel "
            "(--bench-dir ironclad_dev_v*.jsonl): an arm chosen on it was screened on those",
            "panels": {n: len(keys) for n, keys in screened.items()},
            "dropped": screened_dropped,
        },
        "filter_stages": stages,
        "devpool": {
            "rule": "keys of logs/2026-10-18-rollout-net-recipe/{"
            + ",".join(DEV_SOURCES)
            + "}, re-extracted from the raw runs with the build rules, minus every run of a "
            "pro-cohort player",
            "source_keys": source_counts,
            "union_keys": len(dev_keys),
            "runs_missing_from_index": len(missing_dev),
            "pro_player_runs_excluded": len(dev_pro_runs),
        },
        "sets": manifest_sets,
        "unbuildable": unbuildable,
        "rosters": {
            "by_set": rosters,
            "lossy_override_encounters": sorted(
                e for e, row in table.items() if row["default"] and not row["lossless"]
            ),
            "randomized_encounters": sorted(e for e, row in table.items() if row["randomized"]),
            "engine_unknown_encounters": sorted(e for e, row in table.items() if "error" in row),
            "rule": "monster_ids minus pets and summons (summoner already on the roster); "
            "monsters=null when it equals the engine's own roster; the override only where "
            "the encounter rebuilds identically through it and every spawn-arg kind is on its "
            "own roster; else roster_unmatched",
        },
        "decks": {
            "by_set": deck_stats,
            "rule": "per-copy backward replay from the final deck "
            "(pilot_human_decks.DeckReplay"
            + (", frozen_v1" if frozen_v1 else "")
            + "); counts compare it with the forward count rule "
            "that extracted hbank and logs/2026-10-18-rollout-net-recipe/*.jsonl: same, "
            "more_upgrades / fewer_upgrades (the forward rule's upgrade count was off), "
            "upgrade_moved (same count, on a different copy), other (cards or enchantment "
            "amounts differ)",
        },
        "potions": {
            "by_set": potion_stats,
            "rule": "pilot gets: drunk potions on the entry belt, held Fairy in a Bottle, "
            "drunk potions generated mid-fight except AnyTime ones (likely post-combat "
            "rewards drunk after the fight) and except all of them when an in-combat "
            "generator is present (Entropic Brew given, Alchemize in the deck, Delicate "
            "Frond held: the engine rolls its own); capped at the belt's slots; with "
            "Petrified Toad one drunk rock is left to the engine's toad and the rock takes "
            "a slot",
            "engine_belt_slots": "per fight: the human's potion_slots",
        },
        "event_combat": {
            "by_set": event_stats,
            "handling": "extracted (rooms = [event, monster], encounter = the monster room's) "
            "and flagged event_combat; the engine builds none of the *_EVENT_ENCOUNTER "
            "encounters, so all are excluded as unbuildable (unknown_encounter)",
        },
        "rl2_pools": {
            "hbank_sha256": _sha256(args.hbank),
            # in_rl2_eval (pro sets, after v1): the recipe's RL eval and screen sets.
            "eval_sets": list(DEV_SOURCES),
            "gen_keys_in_sets": len(gen_keys & ours),
            "rl_keys_in_sets": len(rl_keys & ours),
        },
        "held_out_runs": sorted(held_out),
    }
    with open(
        os.path.join(args.out_dir, manifest_file(args.version)), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    _log(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")
    pro_keys = ("fights", "runs", "players", "by_room", "human_deaths", "quits")
    pro_keys += ("kish_n_eff_players",)
    print(
        json.dumps(
            {
                "pro": {k: pro_summary[k] for k in pro_keys},
                **{
                    f"pro{n}": {k: manifest_sets[extra_names[n]][k] for k in pro_keys}
                    for n in extra
                },
                "devpool": {
                    k: manifest_sets[dev_name][k]
                    for k in ("fights", "runs", "players", "by_room", "human_deaths", "quits")
                },
                "overlap_with_pros25": manifest["selection"]["overlap_with_pros25"],
                "unbuildable": {k: v["by_reason"] for k, v in unbuildable.items()},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
