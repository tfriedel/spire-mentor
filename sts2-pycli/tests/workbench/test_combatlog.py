"""Display rendering of the engine combat log (``workbench.combatlog``).

The reference for every expected string here is the game's own log panel:
"The Ironclad dealt 6 damage to Nibbit using Strike", "Nibbit used Butt Move",
"The Ironclad applied 1 Hellraiser to The Ironclad".
"""

from workbench.combatlog import LogNames, format_combat_log

_NAMES = LogNames(
    card=lambda i: {
        "STRIKE_IRONCLAD": "Strike",
        "DEFEND_IRONCLAD": "Defend",
        "SLIMED": "Slimed",
        "BASH": "Bash",
        "HAVOC": "Havoc",
    }.get(i, i),
    monster=lambda i: i,
    power=lambda i: {"HELLRAISER": "Hellraiser"}.get(i, i),
    move=lambda i: {"BUTT": "Butt Move"}.get(i, i),
    character=lambda i: {"IRONCLAD": "The Ironclad", "PLAYER": "The Ironclad"}.get(i, i),
    orb=lambda i: i.capitalize(),
    potion=lambda i: i,
)

_PLAYER = {"kind": "player", "id": "IRONCLAD", "uid": None}
_NIBBIT = {"kind": "monster", "id": "Nibbit", "uid": 1}
_STRIKE_SOURCE = {"kind": "card", "id": "STRIKE_IRONCLAD", "upgraded": False}


def _row(kind, **fields):
    return {"kind": kind, "round": 1, "side": "player", "source": None, **fields}


def _damage(**overrides):
    fields = {
        "dealer": _PLAYER,
        "receiver": _NIBBIT,
        "unblocked": 6,
        "blocked": 0,
        "hp_before": 20,
        "hp_after": 14,
        "source": _STRIKE_SOURCE,
    }
    return _row("damage", **(fields | overrides))


def test_empty_log_renders_nothing():
    assert format_combat_log([]) == []


def test_a_hit_names_dealer_amount_target_and_card():
    (line,) = format_combat_log([_damage()], _NAMES)
    assert line["text"] == "The Ironclad dealt 6 damage to Nibbit using Strike."
    # The UI colours the names, so it needs them apart from the sentence.
    assert (line["actor"], line["actor_side"]) == ("The Ironclad", "player")
    assert (line["target"], line["target_side"]) == ("Nibbit", "enemy")
    assert (line["amount"], line["source"]) == (6, "Strike")


def test_a_hit_with_no_open_card_omits_the_using_clause():
    (line,) = format_combat_log([_damage(source=None)], _NAMES)
    assert line["text"] == "The Ironclad dealt 6 damage to Nibbit."


def test_an_upgraded_source_card_keeps_its_plus():
    upgraded = {"kind": "card", "id": "STRIKE_IRONCLAD", "upgraded": True}
    (line,) = format_combat_log([_damage(source=upgraded)], _NAMES)
    assert line["source"] == "Strike+"
    assert line["text"].endswith("using Strike+.")


def test_source_less_damage_reads_took():
    (line,) = format_combat_log([_damage(dealer=None, receiver=_PLAYER)], _NAMES)
    assert line["text"] == "The Ironclad took 6 damage."


def test_block_gain():
    (line,) = format_combat_log([_row("block", receiver=_PLAYER, amount=5)], _NAMES)
    assert line["text"] == "The Ironclad gained 5 Block."


def test_power_applied_names_applier_and_target():
    row = _row("power", applier=_PLAYER, receiver=_PLAYER, power="HELLRAISER", amount=1)
    (line,) = format_combat_log([row], _NAMES)
    assert line["text"] == "The Ironclad applied 1 Hellraiser to The Ironclad."


def test_power_without_an_applier_reads_gained():
    row = _row("power", applier=None, receiver=_NIBBIT, power="HELLRAISER", amount=2)
    (line,) = format_combat_log([row], _NAMES)
    assert line["text"] == "Nibbit gained 2 Hellraiser."


def test_monster_move():
    row = _row("monster_move", side="enemy", monster=_NIBBIT, move_id="BUTT")
    (line,) = format_combat_log([row], _NAMES)
    assert line["text"] == "Nibbit used Butt Move."
    assert line["actor_side"] == "enemy"


def test_orb_rows():
    rows = [_row("orb_channeled", orb="LIGHTNING"), _row("orb_evoked", orb="LIGHTNING")]
    texts = [r["text"] for r in format_combat_log(rows, _NAMES)]
    assert texts == [
        "The Ironclad channeled Lightning.",
        "The Ironclad evoked Lightning.",
    ]


def test_the_games_log_hides_plays_draws_energy_and_power_flashes():
    # `power_flash` is in this list for a reason the others are not: it is
    # hidden by DEFAULT and un-hidden only by an explicit `keep`, which is the
    # entire premise of that parameter. Nothing else pins it -- every other
    # surviving test asserts the row is PRESENT, so deleting `power_flash`
    # from `_VERBOSE_ONLY` once left the whole workbench suite green (979
    # passed). The tripwire that used to carry this invariant was correctly
    # removed with the hand-mirror it guarded; this row is where it lands now.
    rows = [
        _row("energy_spent", amount=1),
        _row("card_play_started", card="STRIKE_IRONCLAD", upgraded=False),
        _row("card_drawn", card="DEFEND_IRONCLAD", upgraded=False, from_hand_draw=True),
        _damage(),
        _row("power_flash", creature=_NIBBIT, power="THORNS"),
        _row("card_play_finished", card="STRIKE_IRONCLAD", upgraded=False),
    ]
    kinds = [r["kind"] for r in format_combat_log(rows, _NAMES)]
    assert kinds == ["damage"], "the game's panel shows the hit, not the bookkeeping"


def test_verbose_keeps_them_and_says_drew_not_discarded():
    rows = [
        _row("energy_spent", amount=1),
        _row("card_drawn", card="DEFEND_IRONCLAD", upgraded=False, from_hand_draw=True),
    ]
    texts = [r["text"] for r in format_combat_log(rows, _NAMES, verbose=True)]
    # The engine's own `text` says "discarded" here, bug-compatibly with C#.
    # A human-facing log must not.
    assert texts == ["The Ironclad spent 1 Energy.", "The Ironclad drew Defend."]


def test_keep_un_hides_only_the_named_kind():
    """``keep`` is the narrow alternative to ``verbose``.

    The workbench's timeline slices need exactly one hidden kind on the wire —
    ``power_flash``, which the board's VFX layer reads off the node — and none
    of the card tape, which would ride into every saved run's JSONL and force
    the client to re-hide six kinds it never asked for. So this asserts BOTH
    halves: the named kind arrives, and its neighbours in ``_VERBOSE_ONLY``
    still do not.
    """
    rows = [
        _row("energy_spent", amount=1),
        _row("card_drawn", card="DEFEND_IRONCLAD", upgraded=False, from_hand_draw=True),
        _damage(),
        _row("power_flash", creature=_NIBBIT, power="THORNS"),
    ]
    kinds = [r["kind"] for r in format_combat_log(rows, _NAMES, keep={"power_flash"})]
    assert kinds == ["damage", "power_flash"]


def test_keep_is_inert_for_a_kind_that_was_never_hidden():
    """A kind outside ``_VERBOSE_ONLY`` is shown either way — naming it in
    ``keep`` must not change anything, least of all duplicate it.

    The second row is what makes this test able to fail: with only the
    ``damage`` row it passed even with ``keep`` deleted from the filter
    expression entirely, since it then guarded nothing but duplication.
    Naming an unhidden kind must also leave its HIDDEN neighbours hidden —
    ``keep`` un-hides what it names, not everything alongside it.
    """
    rows = [
        _damage(),
        _row("card_drawn", card="DEFEND_IRONCLAD", upgraded=False),
        _row("power_flash", creature=_NIBBIT, power="THORNS"),
    ]
    kinds = [r["kind"] for r in format_combat_log(rows, _NAMES, keep={"damage"})]
    assert kinds == ["damage"]


def test_an_unknown_kind_falls_back_to_the_engine_text():
    row = _row("something_new", text="IRONCLAD did a new thing")
    (line,) = format_combat_log([row], _NAMES)
    assert line["text"] == "IRONCLAD did a new thing"


def test_an_unknown_kind_with_no_text_is_dropped():
    assert format_combat_log([_row("something_new")], _NAMES) == []


def test_ids_are_prettified_when_no_resolvers_are_given():
    (line,) = format_combat_log([_row("monster_move", monster=_NIBBIT, move_id="BUTT_MOVE")])
    assert line["text"] == "Nibbit used Butt Move."


def test_a_generated_card_is_shown_like_the_game_does():
    # A monster's Slimed / Wound lands here: the move row above says what the
    # enemy did, this says what it left behind.
    row = _row("card_generated", card="SLIMED", upgraded=False, pile="discard", by_player=False)
    (line,) = format_combat_log([row], _NAMES)
    assert line["text"] == "The Ironclad generated Slimed."
    assert line["verb"] == "generated"


def test_a_move_source_carries_the_raw_move_and_monster_ids():
    # Two monsters (Crusher, BowlbugNectar) both have a THRASH move, so
    # source_move_id alone is not a unique VFX-binding key — source_monster
    # is the other half.
    move_source = {"kind": "move", "monster": "Crusher", "move_id": "THRASH"}
    (line,) = format_combat_log([_damage(source=move_source)], _NAMES)
    assert line["source_kind"] == "move"
    assert line["source_move_id"] == "THRASH"
    assert line["source_monster"] == "Crusher"
    assert line["source_card"] is None
    assert line["source_upgraded"] is None
    # The display name is unchanged — this is additive.
    assert line["source"] == "THRASH"


def test_a_card_source_carries_id_and_upgrade_but_no_monster():
    (line,) = format_combat_log([_damage()], _NAMES)
    assert line["source_kind"] == "card"
    assert line["source_card"] == "STRIKE_IRONCLAD"
    assert line["source_upgraded"] is False
    assert line["source_move_id"] is None
    assert line["source_monster"] is None
    # The display name is unchanged — this is additive.
    assert line["source"] == "Strike"


def test_an_absent_source_leaves_the_raw_id_fields_none():
    (line,) = format_combat_log([_damage(source=None)], _NAMES)
    assert line["source_kind"] is None
    assert line["source_card"] is None
    assert line["source_upgraded"] is None
    assert line["source_move_id"] is None
    assert line["source_monster"] is None


def test_an_auto_played_card_is_shown_and_its_rows_are_tagged():
    # GH #1155: Stampede auto-plays Bash at end of turn. The step is labelled
    # "End turn", so the auto-play row is the only place the card shows.
    bash = {"kind": "card", "id": "BASH", "upgraded": False}
    rows = [
        _row("card_play_started", card="BASH", upgraded=False, auto_played=True, energy_spent=0),
        _damage(source=bash),
        _row("power", applier=None, receiver=_NIBBIT, power="VULNERABLE", amount=2, source=bash),
        _row("card_play_finished", card="BASH", upgraded=False, source=bash),
        _row("monster_move", side="enemy", monster=_NIBBIT, move_id="BUTT"),
    ]
    out = format_combat_log(rows, _NAMES)
    assert [r["kind"] for r in out] == ["card_auto_played", "damage", "power", "monster_move"]
    assert out[0]["text"] == "The Ironclad auto-played Bash."
    assert (out[0]["verb"], out[0]["noun"]) == ("auto-played", "Bash")
    assert (out[0]["card_id"], out[0]["card_upgraded"]) == ("BASH", False)
    assert all(r["auto_played"] is True for r in out[:3])
    # Absent, not False, outside an auto-play: the enemy's turn is not part of it.
    assert "auto_played" not in out[3]


def test_a_nested_auto_play_names_its_parent_and_only_it_is_tagged():
    # A manual Havoc auto-plays the top card: the parent's rows are the player's
    # own play (untagged), the child's are tagged, and the child names its parent.
    havoc = {"kind": "card", "id": "HAVOC", "upgraded": False}
    rows = [
        _row("card_play_started", card="HAVOC", upgraded=False, auto_played=False),
        _row(
            "card_play_started",
            card="STRIKE_IRONCLAD",
            upgraded=True,
            auto_played=True,
            source=havoc,
        ),
        _damage(),
        _row("card_play_finished", card="STRIKE_IRONCLAD", upgraded=True, source=_STRIKE_SOURCE),
        _row("block", receiver=_PLAYER, amount=3, source=havoc),
        _row("card_play_finished", card="HAVOC", upgraded=False, source=havoc),
    ]
    out = format_combat_log(rows, _NAMES)
    assert [r["kind"] for r in out] == ["card_auto_played", "damage", "block"]
    assert out[0]["text"] == "The Ironclad auto-played Strike+ via Havoc."
    assert out[0]["auto_played"] is True
    assert out[1]["auto_played"] is True
    assert "auto_played" not in out[2]
