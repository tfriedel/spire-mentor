"""Replay recorded co-op fights in the engine, decision by decision, and diff.

Step 6 of ``docs/plans/2026-10-26-coop-multiseat-engine.md``. The advisor's
workbench run log (``realgame_workbench_log``) stores, per decision, the board
the mod sent (headless ``combat_play`` shape, every teammate's whole seat
included) and the game's own combat-log rows since the previous decision
(``combat_log_raw``). Those rows say who played which card at what, in the
order the game resolved them (``card_play_started`` rows carry ``owner``,
``target_index`` and ``play_index``).

For each pair of consecutive combat boards ``(a, b)`` this builds a whole co-op
fight from ``a`` (``$.coop_full``), plays every seat's recorded cards in the
logged order -- ours through ``apply``, a teammate's through
``apply_as_ally`` -- ends the turn when the log shows the enemies acting, and
diffs the result against ``b``. Each pair starts from the recorded board, so a
divergence does not carry into the next one.

Every pick a player made mid-combat is logged too (``player_choice`` rows,
newer mods): each seat's choices -- in a play, at a turn start -- are answered
from them in order, ours from the open choice, a teammate's through the
engine's ``teammate_chooser``. A teammate's pick with no row is left to the
engine (counted as ``unanswered``).

The game sends us our board when OUR turn starts, which can be before a
teammate has made their turn-start pick (Toasty Mittens). That board shows
their hand from before the pick, and the pick is logged with a later board.
So a pair whose turn-start pick is missing replays on to the board the pick
comes with (``span``). A pair that starts from such a board, and a fight won
while the teammate was still choosing, are skipped (GH #989).

Potions are drunk from the logged seat's belt (a teammate's comes with newer
mods; for older boards the potions the log shows them drinking are put on it).
A potion whose choice screen was still open when the game sent board ``a`` is
already off the belt there; it goes back into the slot it was last seen in.

Usage::

    python -m coop_replay <workbench-run-log.jsonl> [...]

Prints one JSON line per replayed pair and a summary. Pairs the replay cannot
drive yet (our choice screen with no logged pick, a potion not on the drinker's
belt, an unknown owner)
are counted as skipped with the reason, never guessed at.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from sts2_hint.adapter import translate


def _norm(name: Any) -> str:
    """A monster / card id, compared loosely: ``Nibbit`` == ``NIBBIT``, and the
    engine wire's ``STAMPEDE_CARD`` (a card named like a power) == the game's
    ``STAMPEDE``, and the adapter's ``STATUS:WOUND`` == ``WOUND`` (and ``CURSE:DOUBT`` == ``DOUBT``)."""
    s = str(name or "").upper()
    s = s.removeprefix("STATUS:").removeprefix("CURSE:").removesuffix("_CARD")
    return s.replace("_", "")


def load_states(path: Path) -> list[dict[str, Any]]:
    """The run log's combat boards, in order (``type == "state"`` records
    whose data is a ``combat_play`` decision).

    The rows of any other decision logged in between (a card-select screen, a
    reward) are carried onto the next board, and that board is marked
    ``interrupted``: the replay cannot answer the screen, so it skips the
    pair rather than miss what the screen did.
    """
    out: list[dict[str, Any]] = []
    carried: list[dict[str, Any]] = []
    interrupted = False
    # The advisor's reconnect mark, which lands on whatever record came next
    # (a reward screen too): it belongs to the next combat board.
    reconnected = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A log still being written ends in a half line.
                continue
            if rec.get("type") != "state":
                continue
            data = rec.get("data") or {}
            rows = data.get("combat_log_raw") or []
            reconnected |= bool(data.get("after_reconnect"))
            if data.get("decision") == "combat_play":
                data["combat_log_raw"] = carried + list(rows)
                data["interrupted"] = interrupted
                if reconnected:
                    data["after_reconnect"] = True
                carried, interrupted, reconnected = [], False, False
                out.append(data)
            else:
                carried.extend(rows)
                interrupted = True
    return out


def full_fight_wire(board: dict[str, Any]) -> dict[str, Any]:
    """``board`` as the engine wire of a whole co-op fight in which the
    teammates' remaining plays come from the log, not the engine's policy."""
    if board.get("coop_rng"):
        # The run's dice at this board (newer mods): shuffles and intents
        # replay exactly instead of from a default seed.
        board = {**board, "rng": board["coop_rng"]}
    wire = translate(board)
    wire["coop_full"] = True
    # Our place in the run's player order: the teammates ahead of us set up,
    # draw and end their turn first, from the shuffle stream the seats share.
    if board.get("seat_index"):
        wire["coop_seat_index"] = board["seat_index"]
    for ally in wire.get("allies") or []:
        ally["ended_turn"] = True
    _seat_histories(wire, board)
    _seat_turn_histories(wire, board)
    _teammate_facing(wire, board)
    return wire


def _seat_histories(wire: dict[str, Any], board: dict[str, Any]) -> None:
    """Each seat's combat-history counters, recounted from the fight's rows
    up to ``board`` (``fight_rows``, set by :func:`replay_log`), which the
    board itself does not carry (GH #989): the hits that took the seat's HP
    this combat (Tear Asunder) and the seat's card hits on each enemy this
    turn (a teammate's Gang Up).

    A row names a player by creature ``combat_id``: the players' are their
    places in the run's player order, ours ``seat_index`` and the teammates'
    the others, in order.

    Newer mods send the counters themselves, read from the game's History,
    which this leaves as they are."""
    rows = board.get("fight_rows")
    if rows is None or (board.get("player") or {}).get("unblocked_hits_this_combat") is not None:
        return
    me = int(board.get("seat_index") or 0)
    seats = [wire["player"], *(wire.get("allies") or [])]
    ids = [me, *(i if i < me else i + 1 for i in range(len(seats) - 1))]
    enemy_index = {e.get("combat_id"): i for i, e in enumerate(board.get("enemies") or [])}
    for seat, cid in zip(seats, ids, strict=True):
        unblocked = 0
        hits: Counter[int] = Counter()
        for row in rows:
            if row.get("kind") != "damage":
                continue
            dealer, receiver = row.get("dealer") or {}, row.get("receiver") or {}
            if (
                receiver.get("kind") == "player"
                and receiver.get("combat_id") == cid
                and (row.get("unblocked") or 0) > 0
            ):
                unblocked += 1
            if (
                row.get("round") == board.get("round")
                and row.get("side") == "player"
                and dealer.get("kind") == "player"
                and dealer.get("combat_id") == cid
                and (row.get("source") or {}).get("kind") == "card"
                and receiver.get("combat_id") in enemy_index
            ):
                hits[enemy_index[receiver["combat_id"]]] += 1
        if unblocked:
            seat["unblocked_hits_this_combat"] = unblocked
        if hits:
            seat["powered_hits_this_turn"] = [
                {"enemy": i, "hits": n} for i, n in sorted(hits.items())
            ]


def _seat_turn_histories(wire: dict[str, Any], board: dict[str, Any]) -> None:
    """Each seat's turn so far, recounted from this turn's rows (``fight_rows``)
    for a board that does not carry it (mods before GH #989 sent it): the
    cards the seat exhausted (Evil Eye's second block) and the block its card
    plays gained (Unmovable doubles only the turn's first).

    Exhausts: C#'s ``CardExhaustedEntry`` actor is the exhausted card's
    owner. Newer mods put it on the row (``owner``); an older row names no
    one, and goes to whoever acted last this turn (a card play, a pick or a
    potion), inside whose action C# records it.

    Block: v0.111.0 Unmovable counts the block gains of the OWNER's card
    plays (``e.CardPlay.Player.Creature == Owner``, and ``CardPlay.Player``
    is the card's owner, ``CardModel.cs:1918``), whoever received them. The
    row names the receiver, not the card's owner, so it goes to the owner
    of the play it is logged inside: the latest ``card_play_started``. Only
    a row with a card for its source counts (C# ``CardPlay != null``).
    v0.114.0 counts the earlier card PLAYS that gained block
    (``DistinctBy(e => e.CardPlay)``), so there a play's gains count once."""
    rows = board.get("fight_rows")
    if rows is None or (board.get("player") or {}).get("cards_exhausted_this_turn") is not None:
        return
    seat_of = {str(board.get("net_id")): wire["player"]}
    for raw, seat in zip(board.get("allies") or [], wire.get("allies") or [], strict=False):
        seat_of[str(raw.get("net_id"))] = seat
    exhausted: Counter[str] = Counter()
    block: Counter[str] = Counter()
    per_play = board.get("game_version") == "v0.114.0"
    plays_with_block: set[int] = set()
    play = 0  # the latest card play's index this turn
    actor: str | None = None  # whoever acted last this turn
    player: str | None = None  # whose card play is resolving
    for row in rows:
        if row.get("round") != board.get("round") or row.get("side") != "player":
            continue
        kind = row.get("kind")
        if kind in ("card_play_started", "player_choice", "potion"):
            actor = str(row.get("owner"))
            if kind == "card_play_started":
                player = actor
                play += 1
        elif kind == "card_exhausted":
            whose = row.get("owner")
            whose = actor if whose is None else str(whose)
            if whose is not None:
                exhausted[whose] += 1
        elif kind == "block" and (row.get("source") or {}).get("kind") == "card":
            if player is not None and not (per_play and play in plays_with_block):
                block[player] += 1
                plays_with_block.add(play)
    for net, seat in seat_of.items():
        if exhausted[net]:
            seat["cards_exhausted_this_turn"] = exhausted[net]
        if block[net]:
            seat["card_block_gains_this_turn"] = block[net]


def _row_target_id(row: dict[str, Any], board: dict[str, Any]) -> Any:
    """The ``combat_id`` of the enemy a play or potion row targets: its
    ``target`` when the row names one, else the enemy at ``target_index`` on
    ``board`` (older potion rows carry only the index)."""
    target = row.get("target") or {}
    if target.get("combat_id") is not None:
        return target["combat_id"]
    index = row.get("target_index")
    enemies = board.get("enemies") or []
    if isinstance(index, int) and 0 <= index < len(enemies):
        return enemies[index].get("combat_id")
    return None


def _teammate_facing(wire: dict[str, Any], board: dict[str, Any]) -> None:
    """Each teammate's Surrounded facing (Kaiser Crab), retraced from the
    fight's rows for a board that does not carry it (mods before GH #989).

    C# ``SurroundedPower`` starts facing Right and turns when its player
    plays a card or drinks a potion at the arm behind them
    (``BeforeCardPlayed`` / ``BeforePotionUsed`` -> ``UpdateDirection``): at a
    Back Attack Left arm from Right, to Left; at a Back Attack Right arm from
    Left, to Right. With one arm left it faces that arm (``AfterDeath``). The
    engine infers our own facing from the arms' intents, which are computed
    against us, so only the teammates' needs this."""
    rows = board.get("fight_rows")
    if rows is None:
        return
    side: dict[Any, str] = {}
    for e in board.get("enemies") or []:
        for p in e.get("powers") or []:
            pid = str(p.get("id") or "").removesuffix("_POWER")
            if pid in ("BACK_ATTACK_LEFT", "BACK_ATTACK_RIGHT"):
                side[e.get("combat_id")] = pid
    alive = [
        side[e.get("combat_id")]
        for e in board.get("enemies") or []
        if e.get("combat_id") in side and not e.get("is_dead") and (e.get("hp") or 0) > 0
    ]
    for raw, seat in zip(board.get("allies") or [], wire.get("allies") or [], strict=False):
        surrounded = next(
            (p for p in seat.get("powers") or [] if p.get("name") == "SURROUNDED"), None
        )
        if surrounded is None or "counter" in surrounded:
            continue
        facing = 0  # Right
        for row in rows:
            if row.get("kind") not in ("card_play_started", "potion") or str(
                row.get("owner")
            ) != str(raw.get("net_id")):
                continue
            arm = side.get(_row_target_id(row, board))
            if facing == 0 and arm == "BACK_ATTACK_LEFT":
                facing = 1
            elif facing == 1 and arm == "BACK_ATTACK_RIGHT":
                facing = 0
        if len(alive) == 1:
            facing = 1 if alive[0] == "BACK_ATTACK_LEFT" else 0
        surrounded["counter"] = facing


def _hand_index(hand: list[dict[str, Any]], card: str, upgraded: bool) -> int | None:
    for i, c in enumerate(hand):
        if _norm(c.get("id")) == _norm(card) and bool(c.get("upgraded")) == upgraded:
            return i
    for i, c in enumerate(hand):
        if _norm(c.get("id")) == _norm(card):
            return i
    return None


def _target(
    wire: dict[str, Any], row: dict[str, Any], board: dict[str, Any], origin: dict[str, Any]
) -> int | None:
    """The engine's target index for a logged play.

    By ``combat_id`` when the mod sent it: the target's position in the enemy
    list of ``origin``, the board the engine was built from, whose order the
    engine keeps, its dead included; for an enemy spawned since, its position
    in ``board``'s. C# drops a killed enemy from its list, so a later board's
    positions (and the row's own ``target_index``) can be stale (GH #989).
    Else the row's index when the monster there matches, else the first
    living enemy of that kind.
    """
    combat_id = (row.get("target") or {}).get("combat_id")
    if combat_id is not None:
        for b in (origin, board):
            for i, e in enumerate(b.get("enemies") or []):
                if e.get("combat_id") == combat_id:
                    return i
    index = row.get("target_index")
    if index is None:
        return None
    want = _norm((row.get("target") or {}).get("id"))
    encounter = wire.get("encounter") or []
    if 0 <= index < len(encounter) and (not want or _norm(encounter[index].get("kind")) == want):
        return index
    for i, e in enumerate(encounter):
        if not e.get("is_dead") and _norm(e.get("kind")) == want:
            return i
    return index


def _ally_target(row: dict[str, Any], actor: str, me: str, board: dict[str, Any]) -> int | None:
    """The engine's target for a logged play or drink aimed at a player (an
    ``AnyAlly`` card, a potion thrown at a teammate): the teammate's index
    into ``allies`` as the ACTOR sees them.

    The row names the player by ``net_id`` (mods since GH #1017) or else by
    creature ``combat_id``, which for a player is their place in the run's
    player order (see :func:`_seat_histories`). Seated, a teammate's
    ``allies`` is ours with us in their own slot (``coop::swap_seat``): from
    teammate ``k``'s seat we are ``allies[k]`` and every other teammate keeps
    their index. ``None`` (the engine's first living teammate) when the row
    names no player, names the actor, or names nobody at the table.
    """
    target = row.get("target") or {}
    if target.get("kind") != "player":
        return None
    allies = board.get("allies") or []
    seats = {str(al.get("net_id")): i for i, al in enumerate(allies)}
    net = target.get("net_id")
    cid = target.get("combat_id")
    if net is None and isinstance(cid, int):
        mine = int(board.get("seat_index") or 0)
        if cid == mine:
            net = me
        elif 0 <= (i := cid if cid < mine else cid - 1) < len(allies):
            net = allies[i].get("net_id")
    if net is None or str(net) == actor:
        return None
    net = str(net)
    if actor == me:
        return seats.get(net)
    return seats.get(actor) if net == me else seats.get(net)


def observables(wire: dict[str, Any]) -> dict[str, Any]:
    """What a replayed pair is judged on: every seat's HP, block and hand, our
    energy, and every LIVING enemy's HP and block, in order (C# drops a killed
    enemy from its list where the engine keeps the corpse in place, so the
    positions of the dead are not comparable)."""
    out: dict[str, Any] = {
        "player.hp": wire["player"]["hp"],
        "player.block": wire["player"]["block"],
        "player.energy": wire["player"]["energy"],
        "player.hand": sorted(_norm(c.get("id")) for c in wire["deck"]["hand"]),
    }
    for i, a in enumerate(wire.get("allies") or []):
        out[f"allies[{i}].hp"] = a["hp"]
        out[f"allies[{i}].block"] = a["block"]
        if "hand" in a:
            out[f"allies[{i}].hand"] = sorted(_norm(c.get("id")) for c in a["hand"])
    living = [e for e in wire.get("encounter") or [] if e["hp"] > 0 and not e.get("is_dead")]
    out["encounter.living"] = len(living)
    for i, e in enumerate(living):
        out[f"encounter[{i}].hp"] = e["hp"]
        out[f"encounter[{i}].block"] = e["block"]
    return out


def _choice_selection(recorded: dict[str, Any], options: list[dict[str, Any]]) -> list[int] | None:
    """The option indexes a logged ``player_choice`` row picked, or None when
    they cannot be told apart. Cards match by id (upgraded first), each option
    at most once; an ``Index`` choice names its options by position (-1, a
    skip, names none)."""
    if recorded.get("choice_type") == "Index":
        picks = [i for i in recorded.get("indexes") or [] if i != -1]
        return picks if all(0 <= i < len(options) for i in picks) else None
    if recorded.get("choice_type") == "Player":
        return None
    used: set[int] = set()
    for card in recorded.get("cards") or []:
        want, upgraded = _norm(card.get("id")), bool(card.get("upgraded"))
        same = [i for i, o in enumerate(options) if i not in used and _norm(o.get("id")) == want]
        exact = [i for i in same if bool(options[i].get("upgraded")) == upgraded]
        if not same:
            return None
        used.add((exact or same)[0])
    return sorted(used)


def _choice_action(
    recorded: dict[str, Any], choice: dict[str, Any], actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The ``choose`` action making a logged pick on the open ``choice``.

    One of ``actions`` when it is there; else, within the choice's
    ``min_select`` / ``max_select``, a ``choose`` with the whole selection:
    the engine lists a multi-pick's single picks only (Gambler's Brew may
    discard any number of cards at once)."""
    selection = _choice_selection(recorded, choice.get("cards") or [])
    if selection is None:
        return None
    for act in actions:
        if sorted(act.get("selection") or []) == selection:
            return act
    lo, hi = choice.get("min_select"), choice.get("max_select")
    if isinstance(lo, int) and isinstance(hi, int) and lo <= len(selection) <= hi:
        return {
            "action": "choose",
            "index": selection[0] if selection else 0,
            "selection": selection,
        }
    return None


def replay_pair(a: dict[str, Any], b: dict[str, Any], *later: dict[str, Any]) -> dict[str, Any]:
    """Replay what happened between boards ``a`` and ``b``; see the module doc.

    With ``later`` boards, replay on through them and diff against the last:
    a teammate's turn-start pick (Toasty Mittens) can come after the game
    sent us ``b``, so ``b`` shows their hand before the pick and the pick's
    row is logged with the next board (GH #989). ``b`` is not compared then.
    """
    import sts2_rs

    me = a.get("net_id")
    seats = {str(al.get("net_id")): i for i, al in enumerate(a.get("allies") or [])}
    if me is None or not seats:
        return {"skipped": "not a co-op board with net ids"}
    if any("draw_pile" not in al for al in a.get("allies") or []):
        return {"skipped": "teammate piles missing (older mod)"}
    # Read from the rows alone: decided before the engine is built.
    if _late_turn_start_auto_play(a, b):
        return {"skipped": "a teammate's turn-start auto-play resolved after board a"}
    boards = [b, *later]
    segments = [board.get("combat_log_raw") or [] for board in boards]
    rows = [row for segment in segments for row in segment]
    # Every player's picks, in the order each made them (a player's choice
    # ids run in sequence). Our own screens show up as decisions of their
    # own; with our picks logged the pair can still be replayed.
    picks: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        if row.get("kind") == "player_choice":
            picks[str(row.get("owner"))].append(row)
    if any(board.get("interrupted") for board in boards) and not picks[str(me)]:
        return {"skipped": "a decision screen in between"}
    if any(board.get("after_reconnect") for board in boards):
        # The advisor lost the mod mid-fight: the requests it never read took
        # their rows with them (GH #989).
        return {"skipped": "the advisor reconnected: rows may be lost"}
    owner_of = {i: net for net, i in seats.items()}
    answered = {"choices": 0, "unanswered": 0}
    # The teammates whose choice had no logged pick: it may be logged with a
    # later board. The game sends us our board when our turn starts, which can
    # be before a teammate's turn-start pick (Toasty Mittens), and whenever the
    # board changes, which can be while a teammate's card waits on its pick
    # (Brand's exhaust): either pick is logged with a later board.
    late: list[str] = []
    # Those raised mid-play, not at a turn end: the engine may also have
    # raised one the game never did, so only a logged pick excuses them.
    late_in_play: list[str] = []
    # Choose-a-card screens whose cards differ from the game's (the mod logs
    # the game's as `offered`): a random offer the engine rolled otherwise.
    offers: list[dict[str, Any]] = []
    ended = False
    # The cards just picked on a multi-pick choice, by owner. The engine takes
    # a multi-pick one card at a time and reopens it (minus the cards taken)
    # until `[]` closes it; the game logs the whole pick at once.
    closing: dict[str, list[int]] = {}

    def answer(owner: str, choice: dict[str, Any], actions: list[Any]) -> tuple[Any, bool]:
        """The action for ``owner``'s open choice, and whether it only closes
        the multi-pick answered before (no logged pick of its own)."""
        taken = closing.pop(owner, None)
        if taken is not None:
            offered = [act.get("selection") or [] for act in actions]
            if [] in offered and not any([i] in offered for i in taken):
                return actions[offered.index([])], True
        queue = picks[owner]
        row = queue.popleft() if queue else None
        offered = [_norm(c.get("id")) for c in (row or {}).get("offered") or []]
        engine = [_norm(c.get("id")) for c in choice.get("cards") or []]
        if offered and offered != engine:
            offers.append({"owner": owner, "game": offered, "engine": engine})
        pick = _choice_action(row, choice, actions) if row is not None else None
        if pick is not None and pick.get("selection") and (choice.get("max_select") or 1) > 1:
            closing[owner] = pick["selection"]
        return pick, False

    def teammate_chooser(
        ally: int, choice: dict[str, Any], actions: list[Any]
    ) -> dict[str, Any] | None:
        # -1: a teammate's card (Tutor) makes us pick.
        owner = str(me) if ally < 0 else owner_of.get(ally, "")
        pick, closes = answer(owner, choice, actions)
        if not closes:
            answered["choices" if pick is not None else "unanswered"] += 1
        if pick is None and ally >= 0:
            late.append(owner)
            if not ended:
                late_in_play.append(owner)
        return pick

    def settle_ours() -> str | None:
        """Answer our open choices from the log; the skip reason if we can't."""
        while (acts := eng.legal_actions()) and acts[0].get("action") == "choose":
            pick, closes = answer(str(me), eng.pending_choice() or {}, acts)
            if pick is None:
                return "a choice screen"
            eng.apply(pick, teammate_chooser=teammate_chooser)
            answered["choices"] += 0 if closes else 1
        return None

    def replay_row(row: dict[str, Any], base: dict[str, Any]) -> str | None:
        """Replay one logged row; the skip reason if it can't be."""
        nonlocal ended
        kind = row.get("kind")
        if row.get("side") == "enemy" and not ended:
            ended = True
            eng.apply({"action": "end_turn"}, teammate_chooser=teammate_chooser)
            return settle_ours()
        # After the turn end, only the next round's plays: a teammate can act
        # in it before the game sends us `b` (GH #989).
        next_round = (row.get("round") or 0) > (a.get("round") or 0)
        if (ended and not next_round) or kind not in ("card_play_started", "potion"):
            return None
        if kind == "card_play_started" and (row.get("auto_played") or row.get("play_index")):
            return None
        owner = _row_owner(row, a)
        now = sts2_rs.state_to_dict(eng)
        if owner == str(me):
            hand, belt = now["deck"]["hand"], now["player"].get("potions") or []

            def apply(action: dict[str, Any]) -> None:
                eng.apply(action, teammate_chooser=teammate_chooser)
        elif owner in seats:
            seat = seats[owner]
            hand = now["allies"][seat].get("hand") or []
            belt = now["allies"][seat].get("potions") or []

            def apply(action: dict[str, Any], seat: int = seat) -> None:
                eng.apply_as_ally(seat, action, teammate_chooser=teammate_chooser)
        else:
            return f"unknown owner {owner}"
        if kind == "potion":
            slot = next(
                (i for i, p in enumerate(belt) if _norm(p) == _norm(row.get("potion"))), None
            )
            if slot is None:
                return f"{row.get('potion')} not in {owner}'s belt"
            drink: dict[str, Any] = {"action": "potion", "slot": slot}
            ally = _ally_target(row, owner, str(me), a)
            if ally is not None:
                # Thrown at a teammate (GH #1017).
                drink["ally_index"] = ally
            else:
                drink["target_index"] = _target(now, row, base, a)
            apply(drink)
            counts["potions"] += 1
        else:
            idx = _hand_index(hand, row.get("card"), bool(row.get("upgraded")))
            if idx is None:
                return f"{row.get('card')} not in {owner}'s hand"
            play: dict[str, Any] = {"action": "play", "card_index": idx}
            ally = None
            if (row.get("target") or {}).get("kind") == "player":
                # An `AnyAlly` card at a teammate (GH #1017).
                ally = _ally_target(row, owner, str(me), a)
            if ally is not None:
                play["ally_index"] = ally
            else:
                play["target_index"] = _target(now, row, base, a)
            apply(play)
            counts["played"] += 1
        return settle_ours()

    counts = {"played": 0, "potions": 0}
    wire = full_fight_wire(_restore_potions_in_use(a, rows))
    _carry_teammate_potions(wire, rows, seats)
    eng = sts2_rs.state_from_dict(wire)
    # Each board's rows, read from the board before it (whose enemy list a
    # row's target is found in); the turn ends at most once per board.
    ended_turn = False
    for base, segment in zip([a, *boards], segments):
        ended = False
        for row in segment:
            why = replay_row(row, base)
            if why is not None:
                return {"skipped": why}
        ended_turn |= ended
    # A teammate's first row is a pick the engine never asked for: the game
    # sent us `a` before their turn-start pick (Toasty Mittens), so `a` is
    # not the board the rest of their turn was played from.
    for owner in seats:
        first = next((r for r in segments[0] if _row_owner(r, a) == owner), None)
        if first is not None and first.get("kind") == "player_choice" and first in picks[owner]:
            return {"skipped": "a teammate's turn-start pick made after board a"}
    got = observables(sts2_rs.state_to_dict(eng))
    want = observables(translate(boards[-1]))
    diffs = {k: [got.get(k), want[k]] for k in want if got.get(k) != want[k]}
    return {
        **counts,
        "ended_turn": ended_turn,
        **answered,
        "late": late,
        **({"late_in_play": late_in_play} if late_in_play else {}),
        **({"offers": offers} if offers else {}),
        "diffs": diffs,
    }


def _belt(board: dict[str, Any], net: str) -> list[Any] | None:
    """``net``'s raw belt on ``board`` (slot dicts), ours or a teammate's."""
    if str(board.get("net_id")) == net:
        return (board.get("player") or {}).get("potions")
    for ally in board.get("allies") or []:
        if str(ally.get("net_id")) == net:
            return ally.get("potions")
    return None


def _restore_potions_in_use(a: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``a`` with every potion still being drunk on it put back on the belt.

    The game takes a potion off the belt when the drink starts
    (``PotionModel.RemoveBeforeUse``) and logs it when the drink ends
    (``PotionUsedEntry``, after ``OnUse``). A potion that opens a card choice
    (Attack / Skill / Power / Colorless Potion, Touch of Insanity) keeps the
    drink open while the drinker picks, so a board sent then has the slot
    empty and its row comes with a later board -- several boards later if
    others act meanwhile. The empty slot an earlier board of the fight last
    held it in (``held_before``) is where it goes back (GH #1017).

    The offer was rolled before the screen opened, from the run's shared
    ``CombatCardGeneration`` stream, so ``a``'s dice are already past it: that
    stream goes back to its count on the last board that still showed the
    potion, and the engine rolls the same cards. The logged ``offered`` cards
    check it (``offers``).
    """
    held = a.get("held_before") or {}
    restored = a
    for row in rows:
        if row.get("kind") != "potion":
            continue
        owner = _row_owner(row, a)
        belt = _belt(restored, owner)
        if belt is None:
            continue
        if any(not p.get("empty") and _norm(p.get("id")) == _norm(row.get("potion")) for p in belt):
            continue
        slot = next(
            (
                i
                for i, p in enumerate(belt)
                if p.get("empty")
                and (h := held.get(owner, {}).get(i)) is not None
                and _norm(h[0].get("id")) == _norm(row.get("potion"))
            ),
            None,
        )
        if slot is None:
            continue
        if restored is a:
            restored = copy.deepcopy(a)
            belt = _belt(restored, owner)
        assert belt is not None
        potion, rolled = held[owner][slot]
        belt[slot] = potion
        counters = (restored.get("coop_rng") or {}).get("counters") or {}
        if isinstance(rolled, int) and rolled <= counters.get("combat_card_generation", rolled):
            counters["combat_card_generation"] = rolled
    return restored


def _carry_teammate_potions(
    wire: dict[str, Any], rows: list[dict[str, Any]], seats: dict[str, int]
) -> None:
    """A board from before the mod sent a teammate's belt: give them the
    potions the log shows them drinking, so the drinks can be replayed."""
    drunk: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        seat = seats.get(str(row.get("owner")))
        if row.get("kind") == "potion" and seat is not None:
            drunk[seat].append(row.get("potion"))
    for seat, ids in drunk.items():
        wire["allies"][seat].setdefault("potions", ids)


def _picked(board: dict[str, Any], owners: list[str], rnd: Any = None) -> bool:
    """Whether ``board``'s rows log a pick by one of ``owners`` (in round
    ``rnd``, when given)."""
    return any(
        row.get("kind") == "player_choice"
        and str(row.get("owner")) in owners
        and (rnd is None or row.get("round") == rnd)
        for row in board.get("combat_log_raw") or []
    )


def _row_owner(row: dict[str, Any], board: dict[str, Any]) -> str:
    """Who played the card in ``row``: its ``owner``, unless the row's own
    text names another seat's character.

    Mods before GH #1017 read ``owner`` from the card's current owner when
    the rows were drained, and The Ball changes hands after its play
    (``CardPileCmd.GiveToAnotherPlayer``), so its row named the teammate
    who got it. The text ("IRONCLAD started playing THE_BALL") comes from
    the history entry's actor, fixed when the play started. Used only when
    exactly one seat has that character.
    """
    owner = str(row.get("owner"))
    if row.get("kind") != "card_play_started":
        return owner
    match = re.match(r"(\w+) started playing ", row.get("text") or "")
    if not match:
        return owner
    seats = [
        (str(board.get("net_id")), board.get("character_id")),
        *((str(al.get("net_id")), al.get("character")) for al in board.get("allies") or []),
    ]
    named = [net for net, char in seats if _norm(char) == _norm(match.group(1))]
    return named[0] if len(named) == 1 else owner


def _late_turn_start_auto_play(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether ``b``'s rows open with a teammate's auto-played card in
    ``a``'s round: a turn-start auto-play (Mayhem) still queued when the game
    sent us ``a`` (GH #1017).

    Each player's turn-start auto-plays run on their own action queue, so
    with teammates the game can send us our board before a teammate's have
    resolved; they are logged with the next board. The engine resolves them
    as part of the turn start, so it is already past them at ``a`` and a
    replay from ``a`` never plays them, while one through to ``b`` does. Any
    ordinary play or potion first, an auto-play of ours (our board comes
    after ours), or a play still open at the end of ``a``'s own rows (its
    auto-played follow-ups are logged with ``b``) means the auto-play came
    from a play, not from the turn start.
    """
    me = str(a.get("net_id"))
    if _open_plays(a.get("combat_log_raw") or []):
        return False
    for row in b.get("combat_log_raw") or []:
        if row.get("kind") == "potion":
            return False
        if row.get("kind") != "card_play_started":
            continue
        return (
            bool(row.get("auto_played"))
            and _row_owner(row, a) != me
            and row.get("round") == a.get("round")
        )
    return False


def _open_plays(rows: list[dict[str, Any]]) -> Counter[Any]:
    """The cards whose play started in ``rows`` and has not finished there
    (``card_play_finished`` rows carry the card id, not the player)."""
    open_: Counter[Any] = Counter()
    for row in rows:
        if row.get("kind") == "card_play_started":
            open_[row.get("card")] += 1
        elif row.get("kind") == "card_play_finished" and open_[row.get("card")] > 0:
            open_[row.get("card")] -= 1
    return +open_


def _auto_play_finished(states: list[dict[str, Any]], k: int, floor: Any) -> int | None:
    """The span from ``states[k - 2]`` to the board by which the turn-start
    auto-plays opening ``states[k]``'s rows have all finished: the game can
    send another board while one is still resolving (its damage logged with
    the board after). Only those auto-plays count, not a teammate's ordinary
    play running alongside. None when they have not finished within
    ``LATE_PICK_BOARDS`` boards of this fight."""
    pending: Counter[Any] = Counter()
    for span, board in enumerate(states[k : k + LATE_PICK_BOARDS], start=2):
        if board.get("floor") != floor:
            return None
        for row in board.get("combat_log_raw") or []:
            if span == 2 and row.get("kind") == "card_play_started" and row.get("auto_played"):
                pending[row.get("card")] += 1
            elif row.get("kind") == "card_play_finished" and pending[row.get("card")] > 0:
                pending[row.get("card")] -= 1
        if not +pending:
            return span
    return None


# How many boards on a replay looks for a teammate's late pick.
LATE_PICK_BOARDS = 4


def replay_log(path: Path, seat_index: int | None = None) -> list[dict[str, Any]]:
    """Replay every pair of ``path``'s boards. ``seat_index`` stands in for the
    boards' own when they carry none (recorded before the mod sent it)."""
    states = load_states(path)
    if seat_index is not None:
        for s in states:
            s.setdefault("seat_index", seat_index)
    # Each board's fight so far, for the counters it does not carry.
    fight: list[dict[str, Any]] = []
    # Per seat and slot, the potion an earlier board of the fight last showed
    # there, with the card-generation count on that board: where a potion
    # still being drunk goes back (`_restore_potions_in_use`).
    held: dict[str, dict[int, tuple[dict[str, Any], Any]]] = {}
    for k, s in enumerate(states):
        if k == 0 or states[k - 1].get("floor") != s.get("floor"):
            fight, held = [], {}
        fight = [*fight, *(s.get("combat_log_raw") or [])]
        s["fight_rows"] = fight
        # A drink logged by this board is over: its slot holds nothing to put
        # back any more.
        for row in s.get("combat_log_raw") or []:
            if row.get("kind") != "potion":
                continue
            owner = _row_owner(row, s)
            slots, belt = held.get(owner, {}), _belt(s, owner) or []
            done = next(
                (
                    i
                    for i, (p, _) in sorted(slots.items())
                    if _norm(p.get("id")) == _norm(row.get("potion"))
                    and (i >= len(belt) or belt[i].get("empty"))
                ),
                None,
            )
            if done is not None:
                del slots[done]
        s["held_before"] = {net: dict(slots) for net, slots in held.items()}
        rolled = ((s.get("coop_rng") or {}).get("counters") or {}).get("combat_card_generation")
        for net in [str(s.get("net_id")), *(str(al.get("net_id")) for al in s.get("allies") or [])]:
            for i, p in enumerate(_belt(s, net) or []):
                if not p.get("empty"):
                    held.setdefault(net, {})[i] = (p, rolled)
    results = []
    i = 0
    while i + 1 < len(states):
        a, b = states[i], states[i + 1]
        if a.get("floor") != b.get("floor"):
            i += 1
            continue
        c = states[i + 2] if i + 2 < len(states) else None
        if c is not None and c.get("floor") == b.get("floor") and _late_turn_start_auto_play(b, c):
            # `b` was sent before a teammate's turn-start auto-play resolved
            # (GH #1017): compare against the board it finished by instead.
            span = _auto_play_finished(states, i + 2, b.get("floor"))
            if span is None:
                res = {"skipped": "a teammate's turn-start auto-play never finished in the log"}
            else:
                res = {**replay_pair(a, *states[i + 1 : i + 1 + span]), "span": span}
        else:
            res = replay_pair(a, b)
        # A teammate picked after the game sent us `b` (GH #989): replay on
        # through the boards to the one their pick is logged with. Without
        # that row the pick may be one only the engine raised, so the pair
        # is reported, not excused -- with one exception kept from before:
        # a turn-start pick still open on the log's last board. The search
        # starts past any boards an auto-play span above already took in,
        # so a later span never shortens it.
        base = res.get("span", 1)
        later = itertools.islice(states, i + 1 + base, i + 1 + base + LATE_PICK_BOARDS)
        for span, board in enumerate(later, start=base + 1):
            if not res.get("late"):
                break
            if board.get("floor") != a.get("floor"):
                # The fight ended first. A pick logged with the next fight's
                # rows, stamped with this fight's last round, was still open
                # on this fight's last board.
                last_round = states[i + span - 1].get("round")
                if _picked(board, res["late"], last_round) or (
                    span == 2 and not res.get("late_in_play")
                ):
                    res = {
                        "skipped": "a teammate's turn-start pick still open when the fight ended"
                    }
                break
            if _picked(board, res["late"]):
                res = {**replay_pair(a, *states[i + 1 : i + 1 + span]), "span": span}
                break
        else:
            if res.get("late") and i + 2 >= len(states) and not res.get("late_in_play"):
                # `b` is the log's last board: the fight ended while the
                # teammate was still on their turn-start screen.
                res = {"skipped": "a teammate's turn-start pick still open when the fight ended"}
        res.update(log=path.name, pair=i, floor=a.get("floor"), round=a.get("round"))
        results.append(res)
        i += res.get("span", 1)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument(
        "--seat-index",
        type=int,
        help="our place in the run's player order, for logs recorded before the mod sent it",
    )
    args = ap.parse_args(argv)
    results = [r for p in args.logs for r in replay_log(p, args.seat_index)]
    for r in results:
        print(json.dumps(r))
    skipped = Counter(r["skipped"] for r in results if "skipped" in r)
    replayed = [r for r in results if "skipped" not in r]
    fields = Counter(k for r in replayed for k in r["diffs"])
    print(
        json.dumps(
            {
                "summary": {
                    "pairs": len(results),
                    "replayed": len(replayed),
                    "in_sync": sum(1 for r in replayed if not r["diffs"]),
                    "choices": sum(r["choices"] for r in replayed),
                    "unanswered": sum(r["unanswered"] for r in replayed),
                    "spanned": sum(1 for r in replayed if r.get("span")),
                    "skipped": dict(skipped),
                    "diverging_fields": dict(fields.most_common()),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
