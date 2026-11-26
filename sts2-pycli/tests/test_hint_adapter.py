"""Tests for ``sts2_hint.adapter`` (bead sts2-rs-lv9d).

Fixture-driven tests exercising every adapter path enumerated in the
plan's hint-T7 acceptance row:

* Ironclad vs Nibbit mid-turn (canonical happy path)
* Silent vs Nibbits mid-turn
* hand-with-status (Burn)
* hand-with-curse
* multi-enemy intent
* mid-turn with player powers
* mid-turn with potions
* post-relic-trigger
* Defect orbs
* Necrobinder Osty

Each fixture either translates cleanly OR raises
:class:`sts2_hint.errors.UnsupportedStateError` with a specific
``<path>`` annotated. Both cases are tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sts2_hint import adapter
from sts2_hint.errors import UnsupportedStateError
from sts2_hint.schema import WIRE_VERSION

# Make the fixture helpers importable without forcing a package layout.
sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "hint_adapter"))

from _helpers import (  # type: ignore[import-not-found]
    base_state,
    card,
    enemy,
    player_summary,
)

# ---------------------------------------------------------------------------
# Happy-path translations
# ---------------------------------------------------------------------------


def test_ironclad_vs_nibbit_mid_turn():
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=5,
            relics=[{"name": "Burning Blood", "description": "", "vars": None}],
            potions=[],
        ),
        hand=[
            card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
            card(
                index=1,
                cid="IRONCLAD/DEFEND",
                name="Defend",
                cost=1,
                ctype="Skill",
                target_type="Self",
            ),
            card(index=2, cid="IRONCLAD/BASH", name="Bash", cost=2),
        ],
        enemies=[
            enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}]),
        ],
        draw_pile=[
            card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
        ],
    )
    out = adapter.translate(state)
    assert out["wire_version"] == WIRE_VERSION
    assert out["phase"] == "PlayerPlayPhase"
    assert out["player"]["class"] == "ironclad"
    assert out["player"]["hp"] == 70
    assert out["player"]["block"] == 5
    assert out["deck"]["hand"][0]["id"] == "STRIKE_IRONCLAD"
    assert out["deck"]["hand"][0]["stable_idx"] == 0
    assert out["deck"]["hand"][1]["id"] == "DEFEND_IRONCLAD"
    assert out["deck"]["hand"][2]["id"] == "BASH"
    assert out["encounter"][0]["kind"] == "Nibbit"
    assert out["encounter"][0]["intents"][0]["kind"] == "Buff"
    assert out["player"]["relics"][0]["id"] == "BURNING_BLOOD"


def test_potion_slots_passthrough():
    """GH #893: the belt CAPACITY reaches the wire as ``player.potion_slots``.

    ``potions`` is not the slot count on every producer (the C# headless
    ``PlayerSummary`` drops empty slots), so without the explicit key a
    rebuilt A4+ board (Tight Belt: 2 slots) filled potions as if it had 5.
    Absent or malformed -> omitted, so the engine keeps its unset default.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["player"]["potion_slots"] = 2
    out = adapter.translate(state)
    assert out["player"]["potion_slots"] == 2

    for bad in (None, True, "2", 0):
        state["player"]["potion_slots"] = bad
        assert "potion_slots" not in adapter.translate(state)["player"], bad
    del state["player"]["potion_slots"]
    assert "potion_slots" not in adapter.translate(state)["player"]


def test_cards_played_this_turn_passthrough():
    """The optional per-turn play counters reach the wire player dict.

    Feeds the Ringing/Normality/Sloth ShouldPlay gates in the rebuilt
    search state (2026-07-12 rung-3 stall vs Ceremonial Beast). Absent or
    malformed -> omitted, so the engine defaults to all-zeros.
    """
    played = [2, 1, 0, 0, 0]
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["player"]["cards_played_this_turn"] = played
    out = adapter.translate(state)
    assert out["player"]["cards_played_this_turn"] == played

    # Absent -> key omitted.
    del state["player"]["cards_played_this_turn"]
    out = adapter.translate(state)
    assert "cards_played_this_turn" not in out["player"]

    # Wrong arity (pre-field mod build sending junk) -> omitted, not raised.
    state["player"]["cards_played_this_turn"] = [1, 2]
    out = adapter.translate(state)
    assert "cards_played_this_turn" not in out["player"]


def test_transient_discount_does_not_reconstruct_as_permanent():
    """A globally-discounted card keeps its own cost on the wire.

    The C# ships both the resolved cost and the instance's own cost; only
    the latter may become the rebuilt ``Card.cost``, because the global
    hooks (FreeAttack, Tangled, Enthralled) are re-derived from the powers
    that travel on the same wire. Rebuilding from the resolved cost made a
    spent Unrelenting FreeAttack leave every Strike in hand at 0 for the
    rest of the fight -- 1380 of 1662 diverging cards on the realdll_v7
    mirror sweep had the mirror cheaper than the game.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[
            # FreeAttack is up: resolved 0, own cost still 1.
            card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=0, base_cost=1),
        ],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    out = adapter.translate(state)
    strike = out["deck"]["hand"][0]
    assert strike["cost_override"] == 0
    assert strike["base_cost"] == 1

    # An older C# build that predates the field: fall back to the resolved
    # cost rather than inventing one.
    del state["hand"][0]["base_cost"]
    out = adapter.translate(state)
    assert "base_cost" not in out["deck"]["hand"][0]
    assert out["deck"]["hand"][0]["cost_override"] == 0


def test_a_this_turn_discount_carries_its_expiry():
    """A shaved cost travels with the cost it springs back to.

    Stomp costs 1 less per Attack played THIS TURN, and that shave is undone
    at the next turn start. Nothing in the resolved or own cost says so, and a
    reconstruction that cannot tell a this-turn discount from a this-combat
    one keeps it forever: 163 of STOMP's 175 cost divergences on the v8 mirror
    sweep landed on exactly the ``end_turn`` that should have restored it.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[
            # Two Attacks played this turn, so a 3-cost Stomp is showing at 1
            # -- and goes back to 3 next turn.
            card(
                index=0,
                cid="IRONCLAD/STOMP",
                name="Stomp",
                cost=1,
                base_cost=1,
                persistent_cost=3,
                target_type="AllEnemies",
            ),
        ],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    out = adapter.translate(state)
    stomp = out["deck"]["hand"][0]
    assert stomp["cost_override"] == 1
    assert stomp["base_cost"] == 1
    assert stomp["persistent_cost"] == 3

    # The game omits the field when nothing can expire, and so must we --
    # an absent field means "same as base_cost", not "unknown".
    state["hand"][0]["persistent_cost"] = None
    out = adapter.translate(state)
    assert "persistent_cost" not in out["deck"]["hand"][0]


def test_a_discount_carries_both_of_its_expiry_edges():
    """ "This turn" really means "this turn OR until played".

    The game's own helper is ``SetThisTurnOrUntilPlayed`` and it flags the
    pure-end-of-turn variants "BE CAREFUL USING THIS!", so a consumer that
    only expires discounts at the turn boundary keeps one alive on the very
    card that spent it — 169 of the 1569 divergences on the v15 mirror sweep
    were a just-played card still sitting in the discard pile at cost 0.

    ``stable_cost`` is not redundant with the other two: one combined modifier
    and two separate ones produce identical ``(persistent, played)`` pairs and
    have to expire differently.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[
            card(
                index=0,
                cid="IRONCLAD/STRIKE",
                name="Strike",
                cost=0,
                base_cost=0,
                persistent_cost=1,
                played_cost=1,
                stable_cost=1,
            ),
        ],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    strike = adapter.translate(state)["deck"]["hand"][0]
    assert strike["base_cost"] == 0, "it really is free right now"
    assert strike["persistent_cost"] == 1
    assert strike["played_cost"] == 1, "and playing it is the other way to lose that"
    assert strike["stable_cost"] == 1

    # Absent means "cannot differ from base_cost", not "unknown".
    state["hand"][0]["played_cost"] = None
    assert "played_cost" not in adapter.translate(state)["deck"]["hand"][0]


def test_a_relic_counter_is_its_progress_not_its_threshold():
    """``vars`` are the thresholds; the counter is the progress toward them.

    Guessing "the first int-valued var" gave every counter-driven relic its own
    threshold as its progress — a reconstructed Nunchaku started at 10 of 10
    attacks and fired on the 20th instead of the 10th, which is worse than
    starting at zero.
    """
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=0,
            relics=[
                {
                    "id": "NUNCHAKU",
                    "name": "Nunchaku",
                    "description": "",
                    "vars": {"Cards": 10, "Energy": 1},
                    "counter": 4,
                }
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    (relic,) = adapter.translate(state)["player"]["relics"]
    assert relic["id"] == "NUNCHAKU"
    assert relic["counter"] == 4, "the progress, never the threshold"

    # A producer that sends none leaves it unset rather than guessed.
    del state["player"]["relics"][0]["counter"]
    (relic,) = adapter.translate(state)["player"]["relics"]
    assert relic["counter"] is None


def test_a_powers_self_damage_accumulator_reaches_the_engine():
    """``amount`` describes the stack, not the accumulator.

    CrimsonMantle's ``amount`` is the block it grants; the damage it deals you
    every turn is a separate var, bumped once per copy played. Rebuilt without
    it the engine starts that accumulator at zero and skips the damage — 104 of
    the 107 single-field ``$.player.hp`` divergences on the v10 sweep were a
    mirror sitting exactly 1 HP high at ``end_turn`` with CrimsonMantle up.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["player_powers"] = [
        {
            "name": "Crimson Mantle",
            "id": "CRIMSON_MANTLE_POWER",
            "description": "",
            "amount": 8,
            "display_amount": 8,
            "vars": {"SelfDamage": 2},
        }
    ]
    out = adapter.translate(state)
    (pw,) = out["player"]["powers"]
    assert pw["amount"] == 8
    assert pw["counter"] == 2

    # An unmapped var must NOT be forwarded: putting a number the engine reads
    # for something else into a behaviour-changing slot is worse than absence.
    state["player_powers"][0]["vars"] = {"SomeOtherVar": 5}
    (pw,) = adapter.translate(state)["player"]["powers"]
    assert "counter" not in pw


def test_possess_speed_carries_what_it_stole():
    """GH #1836: Possess Speed's stolen Dexterity lives in private data, sent
    as the ``Stolen`` var; the engine keeps it on ``counter`` and gives it
    back when the owner dies."""
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["enemies"][0]["powers"] = [
        {
            "name": "Possess Speed",
            "id": "POSSESS_SPEED_POWER",
            "description": "",
            "amount": 1,
            "display_amount": 1,
            "vars": {"Stolen": 6},
        }
    ]
    (pw,) = adapter.translate(state)["encounter"][0]["powers"]
    assert pw["counter"] == 6


def test_painful_stabs_carries_the_multi_claw_growth():
    """Test Subject's extra MULTI_CLAW hits live in private monster data,
    sent as Painful Stabs' ``ExtraMultiClawCount`` var; the engine keeps them
    on that power's ``counter`` (Spire Bench seeds 180001932, 180000915)."""
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["enemies"][0]["powers"] = [
        {
            "name": "Painful Stabs",
            "id": "PAINFUL_STABS_POWER",
            "description": "",
            "amount": 1,
            "display_amount": 1,
            "vars": {"ExtraMultiClawCount": 2},
        }
    ]
    (pw,) = adapter.translate(state)["encounter"][0]["powers"]
    assert pw["counter"] == 2


def test_toric_toughness_carries_its_stored_block():
    """``amount`` is the turns-left counter; the block lives in a var.

    ToricToughnessPower.SetBlock stashes the play-time GainBlock result (base
    5, 7 upgraded, Dex/Frail folded in) in ``DynamicVars.Block`` and re-grants
    it Unpowered at the start of each of the owner's next turns. The engine
    keeps that value on ``Power.counter``. Rebuilt without it the re-grant
    handed out 0 block — 303 of the 311 positive ``$.player.block`` end_turn
    divergences on the v27wr sweep (2026-08-01).
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["player_powers"] = [
        {
            "name": "Toric Toughness",
            "id": "TORIC_TOUGHNESS_POWER",
            "description": "",
            "amount": 2,
            "display_amount": 2,
            "vars": {"Block": 7},
        }
    ]
    (pw,) = adapter.translate(state)["player"]["powers"]
    assert pw["name"] == "TORIC_TOUGHNESS"
    assert pw["amount"] == 2, "turns left, untouched"
    assert pw["counter"] == 7, "the stored block, from the var"

    # The table is power-scoped: the same var name on a power whose engine arm
    # does not read ``counter`` (CrabRagePower also declares ``Block``) must
    # not be forwarded.
    state["player_powers"][0]["vars"] = {"SelfDamage": 3}
    (pw,) = adapter.translate(state)["player"]["powers"]
    assert "counter" not in pw


def test_the_card_mid_play_travels_in_its_own_pile():
    """A card_select board is always serialized mid-play.

    C# parks the card being played in its own pile so it counts toward neither
    hand nor discard, and moves it to the discard only when the play finishes —
    which, for a card whose OnPlay awaits a selection, is after the pick. Drop
    that pile and the played card appears in NO pile: a state rebuilt from the
    board has nothing to flush when the choice resolves, so the card never
    reaches the discard at all. On the v10 sweep that was 25 discard entries in
    a 3-seed probe holding a card the mirror had nowhere.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/DEFEND", name="Defend", ctype="Skill")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["play_pile"] = [card(index=0, cid="IRONCLAD/HEADBUTT", name="Headbutt")]
    out = adapter.translate(state)
    assert [c["id"] for c in out["deck"]["play_pile"]] == ["HEADBUTT"]

    # A producer with nothing mid-play omits the key rather than sending an
    # empty list — absent is what every older build means too.
    del state["play_pile"]
    assert "play_pile" not in adapter.translate(state)["deck"]


def test_the_runs_dice_are_forwarded_separately_from_the_search_seed():
    """``$.rng`` carries two unrelated things, and conflating them is costly.

    ``seed`` is the caller's *search* seed. ``run_seed``/``counters`` are the
    game's own RNG lineage, forwarded verbatim: without them a reconstruction's
    ``monster_ai`` and ``shuffle`` streams come from a default seed and have
    nothing to do with the game's, so every intent and reshuffle it predicts is
    a guess.
    """
    state = base_state(
        player=player_summary(hp=70, max_hp=75, block=0, relics=[], potions=[]),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    state["rng"] = {"run_seed": 987654, "counters": {"monster_ai": 12, "shuffle": 5}}
    out = adapter.translate(state, seed=42)
    assert out["rng"]["seed"] == 42
    assert out["rng"]["run_seed"] == 987654
    assert out["rng"]["counters"] == {"monster_ai": 12, "shuffle": 5}

    # A producer that carries neither leaves the block as it always was.
    del state["rng"]
    out = adapter.translate(state, seed=42)
    assert out["rng"] == {"seed": 42, "ascension": 0}

    # And HALF a block is refused, because half is worse than none:
    # `combat_mirror.reconcile` reads a bare `run_seed` as "the world owns the
    # dice" and stops carrying its own valid cursors, while `state_from_dict`
    # given a seed and no counters rebuilds every stream at count 0 — so a
    # producer that published the seed and then threw part-way through its
    # counters would destroy lineage that was intact, not degrade to the
    # pre-field behaviour.
    for partial in (
        {"run_seed": 987654},
        {"counters": {"monster_ai": 12}},
        {"run_seed": 987654, "counters": {}},
        {"run_seed": None, "counters": {"monster_ai": 12}},
    ):
        state["rng"] = partial
        assert adapter.translate(state, seed=42)["rng"] == {"seed": 42, "ascension": 0}, (
            f"a partial rng block must be refused whole: {partial}"
        )


def test_silent_vs_nibbits():
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=70,
            block=0,
            relics=[{"name": "Ring of the Snake", "description": "", "vars": None}],
        ),
        hand=[
            card(index=0, cid="SILENT/STRIKE", name="Strike", cost=1),
            card(
                index=1,
                cid="SILENT/SURVIVOR",
                name="Survivor",
                cost=1,
                ctype="Skill",
                target_type="Self",
            ),
        ],
        enemies=[
            enemy(
                index=0,
                name="Nibbit",
                hp=39,
                max_hp=39,
                intents=[{"type": "Attack", "damage": 9, "hits": 1}],
            ),
            enemy(index=1, name="Nibbit", hp=39, max_hp=39, intents=[{"type": "Debuff"}]),
        ],
    )
    # The player object 'name' field is "Ironclad" by default in our
    # helper; we infer class from card-id prefix instead.
    out = adapter.translate(state)
    assert out["player"]["class"] == "silent"
    assert out["deck"]["hand"][0]["id"] == "STRIKE_SILENT"
    assert out["deck"]["hand"][1]["id"] == "SURVIVOR"
    assert len(out["encounter"]) == 2
    assert out["encounter"][0]["uid"] != out["encounter"][1]["uid"]
    assert out["encounter"][0]["intents"][0]["amount"] == 9


def test_hand_with_status_burn():
    state = base_state(
        player=player_summary(),
        hand=[
            card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
            card(
                index=1,
                cid="STATUS/BURN",
                name="Burn",
                cost=-1,
                ctype="Status",
                can_play=False,
                target_type="None",
                affliction="Burn",
            ),
        ],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
    )
    out = adapter.translate(state)
    burn = out["deck"]["hand"][1]
    assert burn["kind"] == "status"
    assert burn["id"] in {"status:Burn", "BURN"}
    assert burn["stable_idx"] == 1


def test_hand_with_curse():
    state = base_state(
        player=player_summary(),
        hand=[
            card(
                index=0,
                cid="CURSE/REGRET",
                name="Regret",
                cost=-1,
                ctype="Curse",
                can_play=False,
                target_type="None",
                affliction="Regret",
            ),
            card(index=1, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
        ],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
    )
    out = adapter.translate(state)
    regret = out["deck"]["hand"][0]
    assert regret["kind"] == "curse"
    assert regret["id"] in {"curse:Regret", "REGRET"}


def test_multi_enemy_multi_intent():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(
                index=0,
                name="Nibbit",
                hp=40,
                max_hp=40,
                intents=[
                    {"type": "Attack", "damage": 11, "hits": 1},
                    {"type": "Buff"},
                ],
            ),
            enemy(index=1, name="Shrinker Beetle", hp=48, max_hp=48, intents=[{"type": "Defend"}]),
        ],
    )
    out = adapter.translate(state)
    assert out["encounter"][0]["kind"] == "Nibbit"
    assert len(out["encounter"][0]["intents"]) == 2
    assert out["encounter"][0]["intents"][0]["amount"] == 11
    assert out["encounter"][1]["kind"] == "ShrinkerBeetle"


def test_mid_turn_with_player_powers():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
        player_powers=[
            {"name": "Strength", "description": "", "amount": 3},
            {"name": "Vulnerable", "description": "", "amount": -1},
        ],
    )
    out = adapter.translate(state)
    pps = out["player"]["powers"]
    assert {p["name"] for p in pps} == {"STRENGTH", "VULNERABLE"}
    assert next(p for p in pps if p["name"] == "STRENGTH")["amount"] == 3


def test_mid_turn_with_potions():
    state = base_state(
        player=player_summary(
            potions=[
                {
                    "index": 0,
                    "name": "Block Potion",
                    "description": "",
                    "vars": None,
                    "target_type": "Self",
                },
                {
                    "index": 1,
                    "name": "Fire Potion",
                    "description": "",
                    "vars": None,
                    "target_type": "AnyEnemy",
                },
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
    )
    out = adapter.translate(state)
    assert out["player"]["potions"] == ["BLOCK_POTION", "FIRE_POTION"]


def test_post_relic_trigger_block_visible():
    # Akabeko grants Vigor at combat start (a relic-triggered side
    # effect). The adapter doesn't care whether the side effect ran —
    # it just snapshots the resulting state. Test we surface relics
    # and the derived state cleanly.
    state = base_state(
        player=player_summary(
            block=6,  # e.g. Anchor granted Block at combat start
            relics=[
                {"name": "Anchor", "description": "", "vars": None},
                {"name": "Akabeko", "description": "", "vars": None},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        player_powers=[{"name": "Vigor", "description": "", "amount": 8}],
    )
    out = adapter.translate(state)
    assert {r["id"] for r in out["player"]["relics"]} == {"ANCHOR", "AKABEKO"}
    assert out["player"]["block"] == 6
    assert any(p["name"] == "VIGOR" for p in out["player"]["powers"])


def test_defect_orbs():
    state = base_state(
        player=player_summary(),
        hand=[
            card(index=0, cid="DEFECT/ZAP", name="Zap", cost=1, ctype="Skill", target_type="None")
        ],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
        orbs=[
            {"index": 0, "name": "Lightning", "type": "Lightning", "passive": 3, "evoke": 8},
            {"index": 1, "name": "Frost", "type": "Frost", "passive": 2, "evoke": 5},
        ],
        orb_slots=3,
    )
    out = adapter.translate(state)
    assert out["player"]["class"] == "defect"
    assert out["player"]["orbs"][0]["type"] == "LIGHTNING"
    assert out["player"]["orbs"][1]["type"] == "FROST"
    assert out["player"]["orb_slots"] == 3


def test_glass_orb_translates():
    # The engine has modeled GLASS since the orb belt landed, but the title
    # map stopped at the four original orbs, so a Glass orb on the wire raised
    # `unknown orb type 'Glass'` and killed the run (rung-3.5 seed 1019,
    # 2026-08-01, floor 33 — 28 failed searches, then the run died).
    state = base_state(
        player=player_summary(),
        hand=[
            card(index=0, cid="DEFECT/ZAP", name="Zap", cost=1, ctype="Skill", target_type="None")
        ],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
        orbs=[{"index": 0, "name": "Glass", "type": "Glass", "passive": 4, "evoke": 8}],
        orb_slots=3,
    )
    out = adapter.translate(state)
    assert out["player"]["orbs"][0] == {"type": "GLASS", "passive": 4, "evoke": 8}


def test_card_enchantment_carries_its_stack():
    # The name alone rebuilt every scaling enchantment at one stack. Nimble is
    # the one the DLL mirror caught: the world blocked 7 off an enchanted
    # Defend (5 base + 2), the mirror 5.
    state = base_state(
        player=player_summary(),
        hand=[
            card(
                index=0,
                cid="IRONCLAD/DEFEND",
                name="Defend",
                cost=1,
                ctype="Skill",
                target_type="None",
                enchantment="Nimble / Wendig",
                enchantment_amount=2,
            ),
            card(
                index=1,
                cid="IRONCLAD/STRIKE",
                name="Strike",
                cost=1,
                enchantment="Steady",
            ),
        ],
        enemies=[
            enemy(
                index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Attack", "damage": 6}]
            )
        ],
    )
    out = adapter.translate(state)
    defend, strike = out["deck"]["hand"]
    assert defend["enchantment"] == "Nimble"  # bilingual title, English half
    assert defend["enchantment_amount"] == 2
    assert strike["enchantment"] == "Steady"
    # C# omits the amount when it is 0; absent must stay absent, because the
    # engine reads absent as one stack and an explicit 0 as no enchantment.
    assert "enchantment_amount" not in strike


def test_necrobinder_osty():
    # Necrobinder support in sts2-rs is in flight (bead bay); the
    # adapter shouldn't crash on the Osty field even when the rest of
    # the character isn't fully ported. We exercise both alive and
    # dead Osty shapes.
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="NECROBINDER/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        osty={"alive": True, "name": "Osty", "hp": 18, "max_hp": 18, "block": 0},
    )
    out = adapter.translate(state)
    assert out["player"]["class"] == "necrobinder"
    assert out["player"]["osty"]["alive"] is True
    assert out["player"]["osty"]["hp"] == 18


def test_necrobinder_osty_dead():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="NECROBINDER/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        osty={"alive": False},
    )
    out = adapter.translate(state)
    assert out["player"]["osty"]["alive"] is False


# ---------------------------------------------------------------------------
# Fail-loud paths
# ---------------------------------------------------------------------------


def test_unknown_relic_soft_drops():
    state = base_state(
        player=player_summary(
            relics=[{"name": "Mystery Trinket", "description": "", "vars": None}],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    out = adapter.translate(state)
    assert out["player"]["relics"] == []


def test_unknown_monster_fails_with_path():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(index=0, name="Glorbo the Hidden", hp=10, max_hp=10, intents=[{"type": "Buff"}])
        ],
    )
    with pytest.raises(UnsupportedStateError) as excinfo:
        adapter.translate(state)
    msg = str(excinfo.value)
    assert "$.enemies[0]" in msg
    assert "Glorbo" in msg


def test_eye_with_teeth_monster_name_resolves():
    """Regression for the live-bridge parity bug: the real game's enemy
    display name is "Eye with Teeth" (lowercase "with"), but
    ``MONSTER_TITLE_TO_ID`` used to carry "Eye With Teeth" (capital "With")
    -- a stale ``localization_eng/monsters.json`` entry -- so every combat
    against this monster raised ``UnsupportedStateError`` and the real-game
    MCTS bridge fell back to random play (12/138 decisions in
    ``logs/realgame/20260711T004200Z.jsonl``). See
    ``tests/fixtures/realgame_adapter/eye_with_teeth_ironclad.json`` for the
    full-pipeline version of this regression captured from that log.
    """
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(index=0, name="Eye with Teeth", hp=6, max_hp=6, intents=[{"type": "StatusCard"}])
        ],
    )
    out = adapter.translate(state)
    assert out["encounter"][0]["kind"] == "EyeWithTeeth"


def test_test_subject_plain_title_resolves():
    """Regression for the real-game Act-3 boss bug: the C# encounter class's
    own title (``TEST_SUBJECT_BOSS.title`` = "Test Subject",
    ``localization_eng/encounters.json``) is a *different* string from the
    in-combat monster title (``TEST_SUBJECT.name`` =
    "Test Subject #C{Count}", ``localization_eng/monsters.json``) -- both
    name the same species/``MonsterKind`` (``TestSubject.cs``'s ``Title``
    getter always appends the ``{Count}`` var; the boss-encounter title never
    does). ``MONSTER_TITLE_TO_ID`` only carries the templated combat form, so
    a state view that surfaces the plain encounter title (repro: scratchpad
    cellC.log / cell_c_errprobe.py, ``$.enemies[0]``) raised
    ``UnsupportedStateError: unknown monster name 'Test Subject'`` on every
    live Act-3 boss fight. The digit->``{Count}`` retry already in the
    adapter doesn't help here: there are no digits to fold back, the title
    is simply missing its templated suffix entirely.
    """
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(index=0, name="Test Subject", hp=111, max_hp=111, intents=[{"type": "Attack"}])
        ],
    )
    out = adapter.translate(state)
    assert out["encounter"][0]["kind"] == "TestSubject"

    # The already-fixed templated in-combat form must keep resolving too —
    # both forms are the SAME species and must map to the SAME MonsterKind.
    state2 = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(
                index=0, name="Test Subject #C8", hp=111, max_hp=111, intents=[{"type": "Attack"}]
            )
        ],
    )
    out2 = adapter.translate(state2)
    assert out2["encounter"][0]["kind"] == "TestSubject"


def test_unknown_potion_soft_drops():
    state = base_state(
        player=player_summary(
            potions=[
                {
                    "index": 0,
                    "name": "Eldritch Brew",
                    "description": "",
                    "vars": None,
                    "target_type": "Self",
                }
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    out = adapter.translate(state)
    assert out["player"]["potions"] == [None]


def test_unknown_power_soft_drops():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        player_powers=[
            {"name": "Made-Up Power", "description": "", "amount": 1},
        ],
    )
    out = adapter.translate(state)
    assert out["player"]["powers"] == []


def test_power_with_engine_id_survives_missing_loc_title():
    """A power whose engine ``id`` is carried must round-trip even when its
    display title isn't in the localization-derived title map.

    Regression for the Entomancer "MCTS suggests end turn" bug: the Rust
    backend serializes ``ManglePower`` (a *TemporaryStrength* power) with
    display title ``"Mangle"`` — which has no ``powers.json`` loc entry, so the
    title->id lookup returns ``None`` and ``_soft_translate_list`` silently
    dropped it. Dropping Mangle stranded the enemy's −10 Strength as permanent,
    defanging it so every reconstructed line scored P(win)=1.0 and End Turn was
    ranked the best move. Carrying the engine ``id`` (the registry python_name
    ``MANGLE_POWER``, which ``state_from_dict`` accepts verbatim) fixes it.
    """
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            enemy(
                index=0,
                name="Entomancer",
                hp=130,
                max_hp=145,
                intents=[{"type": "Attack", "damage": 3, "hits": 7}],
                powers=[
                    # Display title "Mangle" has no loc entry; only the engine
                    # ``id`` resolves it. Strength carries no ``id`` (mirrors a
                    # power that resolves fine via the title map).
                    {"name": "Mangle", "id": "MANGLE_POWER", "amount": 10},
                    {"name": "Strength", "amount": -10},
                ],
            )
        ],
    )
    out = adapter.translate(state)
    enemy_powers = {p["name"] for p in out["encounter"][0]["powers"]}
    assert "MANGLE_POWER" in enemy_powers, (
        "Mangle (TemporaryStrength) was dropped; the enemy's −10 Strength "
        "becomes permanent and the enemy is defanged in reconstruction"
    )
    assert "STRENGTH" in enemy_powers


@pytest.mark.parametrize(
    ("title", "expected_id"),
    [
        ("Mangle", "MANGLE_POWER"),
        ("Temporary Strength", "TEMPORARY_STRENGTH"),
        ("Temporary Dexterity", "TEMPORARY_DEXTERITY"),
        ("Temporary Focus", "TEMPORARY_FOCUS"),
        ("Metallicize", "METALLICIZE"),
        ("Entangled", "ENTANGLED"),
        ("Dark Shackles", "DARK_SHACKLES"),
    ],
)
def test_title_only_power_resolves_via_registry_augmented_map(title, expected_id):
    """Powers with no loc title still resolve from the display title alone.

    The C# backend (and states recorded before the engine ``id`` field existed)
    carry only the display title — no ``id``. The registry-derived augmentation
    in ``canonical_maps`` fills these in so the same combat-relevant powers that
    the ``id`` passthrough fixes for the Rust backend also round-trip here. On a
    base/``…Power`` title collision the creature-applied ``_POWER`` variant wins
    (``"Mangle"`` -> ``MANGLE_POWER``, not ``MANGLE``).

    Skipped when ``sts2_rs`` is unavailable: the augmentation is registry-driven
    and a no-op build-less (and ``state_from_dict`` can't run there anyway).
    """
    pytest.importorskip("sts2_rs")
    out = adapter._translate_power({"name": title, "amount": 1}, "$.x")
    assert out["name"] == expected_id


def test_carried_id_validated_against_registry_falls_back_to_title():
    """A carried ``id`` that isn't a real registry power must not be trusted.

    The C# backend emits its canonical ``Id.Entry`` (e.g. ``VULNERABLE_POWER``),
    whose ``_POWER`` suffix the Rust registry strips (``VULNERABLE``). Trusting
    such an id verbatim would feed ``state_from_dict`` an unknown power and drop
    Vulnerable/Strength/Weak/Frail entirely on the C# backend. The adapter
    validates the id against the registry first and falls back to the (correct)
    localization title map; only a genuinely valid id (``MANGLE_POWER``) is used
    directly. Needs the registry, so skip when ``sts2_rs`` is unavailable.
    """
    pytest.importorskip("sts2_rs")
    out = adapter._translate_power(
        {"name": "Vulnerable", "id": "VULNERABLE_POWER", "amount": 2}, "$.x"
    )
    assert out["name"] == "VULNERABLE", (
        "suffixed C# id was trusted verbatim instead of falling back to the "
        "title map — Vulnerable would be dropped on the C# backend"
    )


# ---------------------------------------------------------------------------
# master_deck wiring (real-game bridge A+ fidelity)
# ---------------------------------------------------------------------------


def test_master_deck_absent_is_backward_compatible():
    """No ``"master_deck"`` key in the input (the historical Sts2Headless
    combat_play shape, and every other fixture/state built with
    ``base_state`` before this field existed) must translate to exactly
    today's behaviour: ``deck.master_deck is None``."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    assert "master_deck" not in state
    out = adapter.translate(state)
    assert out["deck"]["master_deck"] is None


def test_master_deck_translated_when_present():
    """When the caller carries the run-scope deck (the real-game bridge's
    ``run_state.master_deck``, forwarded by
    ``realgame_adapter.to_headless_combat_play`` as a top-level
    ``"master_deck"`` key), it must reach ``wire["deck"]["master_deck"]`` --
    not be discarded the way the adapter's former hardcoded
    ``deck.master_deck = None`` did."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        master_deck=[
            card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
            card(index=1, cid="IRONCLAD/STRIKE", name="Strike", cost=1),
            card(index=2, cid="IRONCLAD/DEFEND", name="Defend", cost=1, ctype="Skill"),
        ],
    )
    out = adapter.translate(state)
    assert out["deck"]["master_deck"] is not None
    assert len(out["deck"]["master_deck"]) == 3
    assert [c["id"] for c in out["deck"]["master_deck"]] == [
        "STRIKE_IRONCLAD",
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
    ]


def test_master_deck_empty_list_is_explicit_not_none():
    """An explicit ``[]`` (a genuinely empty run-scope deck, as opposed to
    "the caller doesn't know") must stay an explicit empty list, not
    collapse back to ``None`` -- ``sts2_rs`` treats the two the same at
    runtime, but the adapter must not silently reinterpret caller intent."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        master_deck=[],
    )
    assert state["master_deck"] == []
    out = adapter.translate(state)
    assert out["deck"]["master_deck"] == []


def test_dead_enemies_append_as_corpses():
    """``dead_enemies`` (revivers the game keeps in combat: Decimillipede
    segments awaiting REATTACH, Eye with Teeth illusions) must append to
    the wire encounter AFTER the alive entries — alive positions are the
    target_index space on both sides — flagged ``is_dead`` with their
    powers and move_state intact, so ``state_from_dict`` rebuilds the
    corpse and the engine's revive machinery runs in rollouts."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    state["dead_enemies"] = [
        {
            "name": "DECIMILLIPEDE_SEGMENT_FRONT.name",
            "hp": 0,
            "max_hp": 52,
            "block": 0,
            "is_dead": True,
            "powers": [{"name": "Reattach", "id": "REATTACH_POWER", "amount": 25}],
            "move_state": "DEAD",
        }
    ]
    out = adapter.translate(state)
    enc = out["encounter"]
    assert len(enc) == 2
    assert enc[0]["kind"] == "Nibbit" and enc[0]["is_dead"] is False
    corpse = enc[1]
    # Loc-key leak resolves to the EXACT per-segment kind, not the generic
    # DecimillipedeSegment (each segment position has its own AI).
    assert corpse["kind"] == "DecimillipedeSegmentFront"
    assert corpse["is_dead"] is True
    assert corpse["hp"] == 0
    assert corpse["move_state"] == "DEAD"
    assert {"name": "REATTACH", "amount": 25} in corpse["powers"]
    assert corpse["uid"] != enc[0]["uid"]
    # DEAD_MOVE carries no intents in C#, so the corpse telegraphs nothing.
    assert corpse["intents"] == []


def test_removed_enemies_come_as_a_roster_in_creation_order():
    """``removed_enemies`` (GH #1716) are corpses the game took off the board.
    The encounter stays the board, so alive positions keep their meaning; the
    wire adds ``removed_roster``, the board with each removed corpse put back
    before the first enemy created after it, for a fight seeded with nothing
    carried."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            {**enemy(index=0, name="Nibbit", hp=48, max_hp=48), "combat_id": 1},
            {**enemy(index=1, name="Nibbit", hp=40, max_hp=48), "combat_id": 4},
        ],
    )
    assert "removed_roster" not in adapter.translate(state)
    corpse = {"name": "Nibbit", "hp": 0, "max_hp": 44, "block": 0, "is_dead": True}
    state["removed_enemies"] = [
        {**corpse, "combat_id": 3, "intents": [{"type": "Attack", "damage": 6, "hits": 1}]},
        {**corpse, "combat_id": 7},
    ]
    out = adapter.translate(state)
    assert [(e["hp"], e["is_dead"]) for e in out["encounter"]] == [(48, False), (40, False)]
    roster = out["removed_roster"]
    assert [(e["hp"], e["is_dead"], e.get("removed_after_death", False)) for e in roster] == [
        (48, False, False),
        (0, True, True),
        (40, False, False),
        (0, True, True),
    ]
    assert roster[1]["intents"] and roster[1]["max_hp"] == 44


def test_a_corpse_stays_behind_a_summon_put_in_front():
    """GH #1837: Two-Tailed Rat #5 is summoned in front of the board after
    #1 and #3 died. The engine keeps each corpse in its place: #1 first, #3
    after #2 (seed 180000492: native slots 1-dead, 5, 2, 3-dead, 4)."""
    rat = {"name": "Two-Tailed Rat", "block": 0}
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            {**enemy(index=0, name="Nibbit", hp=22, max_hp=22), "combat_id": 5},
            {**enemy(index=1, name="Nibbit", hp=21, max_hp=21), "combat_id": 2},
            {**enemy(index=2, name="Nibbit", hp=19, max_hp=19), "combat_id": 4},
        ],
    )
    corpse = {**rat, "name": "Nibbit", "hp": 0, "is_dead": True}
    state["removed_enemies"] = [
        {**corpse, "combat_id": 1, "max_hp": 18},
        {**corpse, "combat_id": 3, "max_hp": 23},
    ]
    roster = adapter.translate(state)["removed_roster"]
    assert [e["max_hp"] for e in roster] == [18, 22, 21, 23, 19]


def test_a_removed_summon_goes_back_to_its_slot():
    """GH #1838: Fabricator's Zapbot (#3) took the front slot and died before
    any decision saw it. The game sorts its board by encounter slot, so the
    corpse goes ahead of Fabricator, not after it by creation order. A
    corpse also stays ahead of the summon that took its slot over."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            {**enemy(index=0, name="Nibbit", hp=16, max_hp=19), "combat_id": 4, "slot": 0},
            {**enemy(index=1, name="Nibbit", hp=94, max_hp=155), "combat_id": 1, "slot": 4},
        ],
    )
    corpse = {"name": "Nibbit", "hp": 0, "block": 0, "is_dead": True}
    state["removed_enemies"] = [
        {**corpse, "combat_id": 3, "max_hp": 20, "slot": 0},
        {**corpse, "combat_id": 5, "max_hp": 21, "slot": 1},
    ]
    roster = adapter.translate(state)["removed_roster"]
    assert [e["max_hp"] for e in roster] == [20, 19, 21, 155]


def test_an_axebot_respawn_stands_ahead_of_the_corpses_it_replaced():
    """Corpus seed 180000714: Stock respawns each Axebot at the dying one's
    place, so the engine's board is the newest Axebot, then the corpses
    newest first; slot order alone put the corpses ahead."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[
            {**enemy(index=0, name="Axebot", hp=59, max_hp=105), "combat_id": 3, "slot": 0},
        ],
    )
    corpse = {"name": "Axebot", "hp": 0, "block": 0, "is_dead": True}
    state["removed_enemies"] = [
        {**corpse, "combat_id": 1, "max_hp": 81, "slot": 0},
        {**corpse, "combat_id": 2, "max_hp": 95, "slot": 0},
    ]
    roster = adapter.translate(state)["removed_roster"]
    assert [e["max_hp"] for e in roster] == [105, 95, 81]


def test_a_reviving_corpse_keeps_its_heal_telegraph():
    """The turn before a Decimillipede segment reattaches, its move machine
    has already advanced DEAD_MOVE -> REATTACH_MOVE (a ``HealIntent``). That
    telegraph is the only warning that a 25-HP enemy is about to re-enter the
    fight, so it has to survive translation like any living enemy's — the
    serializer used to emit intents for the alive list only."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    state["dead_enemies"] = [
        {
            "name": "DECIMILLIPEDE_SEGMENT_FRONT.name",
            "hp": 0,
            "max_hp": 52,
            "block": 0,
            "is_dead": True,
            "powers": [{"name": "Reattach", "id": "REATTACH_POWER", "amount": 25}],
            "move_state": "REATTACH",
            "intents": [{"type": "Heal"}],
        }
    ]
    corpse = adapter.translate(state)["encounter"][1]
    assert corpse["is_dead"] is True
    assert [i["kind"] for i in corpse["intents"]] == ["Heal"]


def test_pending_choice_fails_loud():
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
    )
    state["pending_choice"] = {"prompt": "Pick a card to exhaust"}
    with pytest.raises(UnsupportedStateError) as excinfo:
        adapter.translate(state)
    assert "$.pending_choice" in str(excinfo.value)


def test_wrong_decision_fails_loud():
    state = {"type": "decision", "decision": "map_choose"}
    with pytest.raises(UnsupportedStateError):
        adapter.translate(state)


# ---------------------------------------------------------------------------
# Import-time guarantee from the bead
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    # The bead's validation gate explicitly calls this out. The import
    # being side-effectful (registry pruning) means we want to confirm
    # it doesn't throw under any reachable codepath.
    import importlib

    importlib.reload(adapter)


def test_wax_and_melt_flags_survive_translation():
    """ToyBox melt: a melted relic stays in the inventory but is hook-silent,
    and both producers ship `is_wax`/`is_melted`. The adapter dropped them,
    so every reconstructed mirror played melted relics as live — the dominant
    class of the d1 forensic corpus (54/158, mirror over-blocking on ToyBox
    boards). Only-when-true, matching the engine wire's convention."""
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=0,
            relics=[
                {"name": "Burning Blood", "description": "", "vars": None},
                {
                    "name": "Nunchaku",
                    "description": "",
                    "vars": None,
                    "is_wax": True,
                    "is_melted": True,
                },
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    plain, melted = adapter.translate(state)["player"]["relics"]
    assert "is_wax" not in plain and "is_melted" not in plain
    assert melted["is_wax"] is True and melted["is_melted"] is True


def test_sword_of_stone_reads_the_games_count_into_the_engines():
    """GH #1684: C# counts the elites defeated up, the engine the ones to go
    down; the game's 1 is the engine's 4."""
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=0,
            relics=[
                {"id": "SWORD_OF_STONE", "name": "Sword of Stone", "vars": None, "counter": 1},
                {"name": "Nunchaku", "description": "", "vars": None, "counter": 7},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    sword, nunchaku = adapter.translate(state)["player"]["relics"]
    assert sword["counter"] == 4
    assert nunchaku["counter"] == 7


def test_left_to_go_counters_read_into_the_engines_count():
    """Wongo's ticket (5 - combats) and Silver Crucible (3 - uses) show what is
    left; the engine counts what is done. Read as is, a fresh ticket (5) showed
    no counter (Spire Bench corpus)."""
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=0,
            relics=[
                {
                    "id": "WONGOS_MYSTERY_TICKET",
                    "name": "Wongo's Mystery Ticket",
                    "vars": None,
                    "counter": 5,
                },
                {"id": "SILVER_CRUCIBLE", "name": "Silver Crucible", "vars": None, "counter": 1},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    ticket, crucible = adapter.translate(state)["player"]["relics"]
    assert ticket["counter"] == 0
    assert crucible["counter"] == 2


def test_a_dead_osty_keeps_its_accumulated_max_hp():
    """A dead Osty is not gone -- and its MaxHp is the Necrobinder's bank.

    ``DieForYouPower.ShouldCreatureBeRemovedFromCombatAfterDeath`` returns false
    for its own owner (DieForYouPower.cs:40-47), so
    ``PlayerCombatState.OnPetDied`` never removes the corpse from ``Pets``
    (PlayerCombatState.cs:288-297) and both producers ship
    ``alive: false, hp: 0, max_hp: N``. The dead branch used to return a flat
    ``max_hp: 0``, which made ``restore_osty`` clamp it to 1 and turned the next
    summon into a fresh spawn instead of a re-raise of the grown corpse.
    """
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="NECROBINDER/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        osty={"alive": False, "name": "Osty", "hp": 0, "max_hp": 31, "block": 4},
    )
    osty = adapter.translate(state)["player"]["osty"]
    assert osty == {"alive": False, "hp": 0, "max_hp": 31, "block": 4}


def test_a_dead_osty_without_a_max_hp_still_reads_zero():
    """The headless twin's bare sentinel (``{"alive": false}``,
    RunSimulator.cs:3988-3990) must keep behaving exactly as before."""
    state = base_state(
        player=player_summary(),
        hand=[card(index=0, cid="NECROBINDER/STRIKE", name="Strike", cost=1)],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[{"type": "Buff"}])],
        osty={"alive": False},
    )
    assert adapter.translate(state)["player"]["osty"] == {
        "alive": False,
        "hp": 0,
        "max_hp": 0,
        "block": 0,
    }


def test_a_registry_valid_relic_id_beats_the_rendered_title():
    """Wax relics and loc-snapshot gaps both soft-dropped on the title path.

    ``RelicModel.Title`` prefixes a wax relic with ToyBox's ``"Wax {Title}"``
    (RelicModel.cs:48-61, ToyBox.cs:26, ``relics.json``
    ``"TOY_BOX.waxRelicPrefix"``), and FISHING_ROD / SILKEN_TRESS exist in the
    live build with no ``.title`` entry in this repo's ``localization_eng``
    snapshot at all. Both producers already publish the canonical
    ``Id.Entry`` (RunSimulator.cs:5047, SimStateExtractor.SerializeRelic), so
    the id is preferred whenever the engine registry knows it.
    """
    state = base_state(
        player=player_summary(
            relics=[
                {"id": "BURNING_BLOOD", "name": "Wax Burning Blood", "is_wax": True},
                {"id": "FISHING_ROD", "name": "Fishing Rod"},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    relics = adapter.translate(state)["player"]["relics"]
    assert [r["id"] for r in relics] == ["BURNING_BLOOD", "FISHING_ROD"]
    assert relics[0]["is_wax"] is True


def test_the_title_path_still_wins_when_the_id_is_not_a_registry_name():
    """Non-regression for the headless arm and for the id dialects that are NOT
    registry ids (the bridge's monster/power ids, a producer that sends a
    display string in ``id``): an unusable ``id`` must fall straight back to the
    existing title resolution, including the leaked-loc-key path."""
    state = base_state(
        player=player_summary(
            relics=[
                # Not a registry name -- must resolve via the title.
                {"id": "burning blood (starter)", "name": "Burning Blood"},
                # No id at all -- the pre-2026-08 headless/fixture shape.
                {"name": "Nunchaku", "description": "", "vars": None, "counter": 4},
                # Leaked loc-key, no id -- the `_registry_id_from_leaked` path.
                {"name": "ANCHOR.title"},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    relics = adapter.translate(state)["player"]["relics"]
    assert [r["id"] for r in relics] == ["BURNING_BLOOD", "NUNCHAKU", "ANCHOR"]
    assert relics[1]["counter"] == 4


def test_an_unusable_id_and_an_unknown_title_still_soft_drops():
    state = base_state(
        player=player_summary(
            relics=[{"id": "NOT_A_RELIC", "name": "Mystery Trinket"}],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    assert adapter.translate(state)["player"]["relics"] == []


# --- GH #370: the single-turn halves of Sly and Retain --------------------
#
# C# keeps them in private bools (`_hasSingleTurnSly`/`_hasSingleTurnRetain`,
# cleared by `EndOfTurnCleanup`) that the keyword set cannot show. The
# producers emit the PUBLIC `IsSlyThisTurn`/`ShouldRetainThisTurn`, which OR
# the keyword in, so the adapter subtracts the keyword to recover the grant.


def test_a_single_turn_retain_is_distinguished_from_the_keyword():
    """Well-Laid Plans retains a card for ONE turn. Folded into the keyword it
    would retain for the rest of the fight; dropped, it flushes at end of turn
    while the world keeps it."""
    granted = adapter._flags_for_card({"keywords": [], "will_retain": True})
    assert granted == ["retain_turn"]

    # Same `will_retain`, but the card has the keyword — so there is no grant.
    intrinsic = adapter._flags_for_card({"keywords": ["Retain"], "will_retain": True})
    assert intrinsic == ["retain"]


def test_a_single_turn_sly_is_distinguished_from_the_keyword():
    """Hand Trick's Sly lasts a turn; MasterPlanner's `AddKeyword` Sly lasts
    the combat. The two land in different engine lists."""
    assert adapter._flags_for_card({"keywords": [], "sly_this_turn": True}) == ["sly_turn"]
    assert adapter._flags_for_card({"keywords": ["Sly"], "sly_this_turn": True}) == ["sly"]


def test_the_spirementor_dialect_reaches_the_same_flags():
    """The mod ships one bool per keyword and no `keywords` list at all, so a
    list-only reader saw an empty keyword set and mistook every intrinsic
    Retain for a this-turn grant."""
    assert adapter._flags_for_card({"sly": True, "sly_this_turn": True}) == ["sly"]
    assert adapter._flags_for_card({"retain": True, "will_retain": True}) == []
    assert adapter._flags_for_card({"retain": False, "will_retain": True}) == ["retain_turn"]


def test_the_bridges_keyword_bools_are_read_but_not_emitted():
    """`state_to_dict` emits `sly` but not `retain`/`ethereal`/`innate`, and
    the mirror diffs the flag tuple — so emitting those here would manufacture
    a desync on every affected live-arm card."""
    flags = adapter._flags_for_card({"retain": True, "ethereal": True, "innate": True})
    assert flags == []


def test_localization_falls_back_to_the_gzipped_copy(tmp_path: Path) -> None:
    """The packaged advisor ships `localization_eng/*.json` gzipped (the game
    reads every .json in a mod's folder as a mod manifest)."""
    import gzip

    from sts2_hint import canonical_maps

    (tmp_path / "relics.json.gz").write_bytes(gzip.compress(b'{"ANCHOR.title": "Anchor", "X": 1}'))
    assert canonical_maps._load_json(tmp_path / "relics.json") == {"ANCHOR.title": "Anchor"}
    (tmp_path / "relics.json").write_text('{"ANCHOR.title": "Plain"}', encoding="utf-8")
    assert canonical_maps._load_json(tmp_path / "relics.json") == {"ANCHOR.title": "Plain"}
    assert canonical_maps._load_json(tmp_path / "missing.json") == {}


def test_a_hidden_left_to_go_counter_reads_as_its_end():
    """C# hides Wongo's ticket counter once five combats are done, past its
    payout, and Silver Crucible's once its uses are spent. Read as 0, a
    paid-out ticket showed a fresh 5 for the rest of the run (GH #1829)."""
    state = base_state(
        player=player_summary(
            hp=70,
            max_hp=75,
            block=0,
            relics=[
                {"id": "WONGOS_MYSTERY_TICKET", "name": "Wongo's Mystery Ticket", "vars": None},
                {"id": "SILVER_CRUCIBLE", "name": "Silver Crucible", "vars": None, "counter": None},
            ],
        ),
        hand=[card(index=0, cid="IRONCLAD/STRIKE", name="Strike")],
        enemies=[enemy(index=0, name="Nibbit", hp=48, max_hp=48, intents=[])],
    )
    ticket, crucible = adapter.translate(state)["player"]["relics"]
    assert ticket["counter"] == -1
    assert crucible["counter"] == 3


def test_a_pile_cards_single_turn_retain_rides_pile_flags():
    """C# keeps `HasSingleTurnRetain` until `EndOfTurnCleanup` walks every
    pile, so the host reports `will_retain` on a discard card Expertise drew
    and the turn played. Off the hand it goes under `pile_flags`, the key the
    engine's emitter uses (GH #1197); `flags` stays ethereal-only there."""
    raw = {
        "id": "CARD.SUCKER_PUNCH",
        "index": 3,
        "type": "Attack",
        "cost": 1,
        "upgraded": True,
        "will_retain": True,
        "sly_this_turn": False,
    }
    spec = adapter._translate_card(raw, "$.discard_pile[3]")
    assert spec["pile_flags"] == ["retain_turn"]
    assert "flags" not in spec
    hand = adapter._translate_card(raw, "$.hand[0]", hand=True)
    assert hand["flags"] == ["retain_turn"]
    assert "pile_flags" not in hand


def test_the_combats_draw_count_is_copied():
    """Murder's multiplier: the host's `cards_drawn_this_combat`."""
    out: dict = {}
    adapter._copy_seat_history({"cards_drawn_this_combat": 21}, out)
    assert out["cards_drawn_this_combat"] == 21


def test_the_combats_generated_card_count_is_copied():
    """Supermassive's multiplier: the host's `cards_generated_this_combat`."""
    out: dict = {}
    adapter._copy_seat_history({"cards_generated_this_combat": 2}, out)
    assert out["cards_generated_this_combat"] == 2
