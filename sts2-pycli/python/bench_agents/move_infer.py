"""Infer each enemy's hidden move from what the referee shows.

The referee hides ``move_state`` (the id of the move an enemy telegraphs) and
its move machine's ``ai`` state; it shows the telegraphed intents. A search
bot needs the move id: without it the engine rebuilds the enemy AI-less.

A player who watched the fight can work it out, and so can this tracker. It
keeps, per enemy, the moves it inferred so far (the move machine's log), and
on each new telegraph asks the engine which moves can follow the last one
(``peek_intents`` on a board where the enemy stands on that move). The
candidate whose engine-rendered intents equal the shown ones is the move. An
enemy's first move is matched among the moves the catalog marks ``initial``,
then among all its moves.

Where two candidates render identically the first (most likely) is taken and
the call is counted as ambiguous; they telegraph the same thing now, and
the next telegraph usually tells them apart.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import sts2_rs
from sts2_hint import adapter


def _shown(intents: list[dict[str, Any]] | None) -> list[tuple[str, int | None, int]]:
    """The referee's intents as comparable tuples."""
    out = []
    for it in intents or []:
        kind = str(it.get("type"))
        # The engine has no CardDebuff intent and reads it as Debuff
        # (`state_dict.rs` `parse_intent`): the Queen's opening Puppet
        # Strings (GH #1933).
        kind = "Debuff" if kind == "CardDebuff" else kind
        out.append((kind, it.get("damage"), int(it.get("hits") or 1)))
    return sorted(out, key=repr)


def _rendered(intents: list[dict[str, Any]] | None) -> list[tuple[str, int | None, int]]:
    """The engine's intents in the same shape (damage only for attacks)."""
    out = []
    for it in intents or []:
        kind = str(it.get("kind"))
        dmg = it.get("effective_amount", it.get("amount")) if kind == "Attack" else None
        out.append((kind, dmg, int(it.get("times") or 1)))
    return sorted(out, key=repr)


def _kinds(t: list[tuple[str, int | None, int]]) -> list[str]:
    return sorted(k for k, _, _ in t)


def _distance(rendered: list, shown: list) -> tuple[int, int]:
    """How far the engine's intents are from the shown ones: hit counts that
    differ first, then the damage gap (modifiers move the shown number)."""
    hits = dmg = 0
    for (k, d, h), (k2, d2, h2) in zip(sorted(rendered), sorted(shown), strict=False):
        if k == k2 == "Attack":
            hits += h != h2
            dmg += abs((d or 0) - (d2 or 0))
    return hits, dmg


@dataclass
class _Enemy:
    kind: str
    log: list[str] = field(default_factory=list)
    shown: list[tuple[str, int | None, int]] | None = None
    round: int | None = None


@dataclass
class MoveTracker:
    """One per fight. ``infer(obs)`` returns ``[{"move_state", "ai"}]`` per enemy."""

    character: str
    ascension: int
    # The build the referee runs (``GET /v0/info``). Observations do not name
    # it, and the engine would otherwise build the legacy one.
    game_version: str
    enemies: dict[int, _Enemy] = field(default_factory=dict)
    calls: int = 0
    ambiguous: int = 0
    unmatched: int = 0

    def _wire(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = {**obs, "game_version": self.game_version}
        return adapter.translate(obs, character=self.character, ascension=self.ascension)

    def _candidates_after(self, wire: dict[str, Any], i: int, prev: str) -> list[tuple[str, list]]:
        """``[(move id, rendered intents)]`` the engine allows after ``prev``."""
        w = copy.deepcopy(wire)
        w["encounter"][i]["move_state"] = prev
        w["encounter"][i]["intents"] = []
        try:
            peek = sts2_rs.state_from_dict(w).peek_intents()[i]
        except Exception:
            return []
        step1 = [p for p in peek if p.get("step") == 1]
        step1.sort(key=lambda p: -float(p.get("certainty", 0)))
        return [(p["move_id"], _rendered(p.get("intents"))) for p in step1]

    def _catalog_fit(self, kind: str, move: str, shown: list) -> tuple[int, int]:
        """``_distance`` of a catalog move's base intent to the shown ones."""
        for mv in sts2_rs.monster_moves(kind, self.ascension, self.game_version):
            if mv["id"] == move and mv.get("intent"):
                return _distance(_rendered([mv["intent"]]), shown)
        return (99, 999)

    def _catalog_match(self, kind: str, shown: list, *, opening: bool) -> str | None:
        """The catalog move closest to the shown intents.

        The catalog lists a move's primary intent with its ascension-scaled
        base amount; the shown damage carries modifiers (Strength, Weak), so
        the closest amount wins rather than an equal one.
        """
        kinds = {k for k, _, _ in shown}
        best: tuple | None = None
        pick = None
        for order, mv in enumerate(sts2_rs.monster_moves(kind, self.ascension, self.game_version)):
            it = mv.get("intent") or {}
            if it.get("kind") not in kinds:
                continue
            same = [x for x in shown if x[0] == it.get("kind")]
            times = int(it.get("times") or 1)
            amount = it.get("amount")
            if it.get("kind") == "Attack":
                gap = min(
                    abs((d or 0) - (amount or 0)) + (0 if h == times else 100) for _, d, h in same
                )
            else:
                gap = 0
            score = (gap, 0 if (mv.get("initial") or not opening) else 1, len(kinds) - 1, order)
            if best is None or score < best:
                best, pick = score, mv["id"]
        return pick

    def infer(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        """Guesses for ``obs["enemies"]``, the living."""
        return self.infer_board(obs)[0]

    def infer_board(self, obs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Guesses for ``obs["enemies"]`` and for ``obs["dead_enemies"]``.

        A dead enemy the host still lists is a corpse that revives (Test
        Subject, Illusion). It telegraphs its revive move, and the engine
        revives it only through that move: a corpse rebuilt with no move
        comes back AI-less, rolling the stand-in Attack 5 (GH #1933).
        """
        wire = self._wire(obs)
        alive = obs.get("enemies") or []
        board = [*alive, *(obs.get("dead_enemies") or [])]
        out = self._infer_all(wire, board, obs.get("round"))
        return out[: len(alive)], out[len(alive) :]

    def _infer_all(
        self, wire: dict[str, Any], board: list[dict[str, Any]], rnd: Any
    ) -> list[dict[str, Any]]:
        # The adapter lays out ``$.encounter`` as the living, then the
        # corpses, both in the order the observation lists them.
        out = []
        for i, e in enumerate(board):
            kind = wire["encounter"][i]["kind"]
            shown = _shown(e.get("intents"))
            # The creature's fight id survives another enemy's death; its
            # board position does not (GH #1651).
            ident = e.get("combat_id", e.get("index", i))
            st = self.enemies.get(ident)
            if st is None or st.kind != kind:
                st = _Enemy(kind)
                self.enemies[ident] = st
            same_round = bool(st.log) and st.round == rnd
            if same_round and st.shown == shown:
                pass  # nothing new
            elif not shown and e.get("is_dead", e.get("hp") == 0):
                pass  # a corpse that shows nothing keeps the move it died with
            else:
                # A new round rolled a new move. Inside a round the move
                # stays, but what it shows can change (Weak, Strength, a
                # stun), so it is inferred again in place.
                if same_round:
                    st.log.pop()
                self.calls += 1
                move = None
                if st.log:
                    cands = self._candidates_after(wire, i, st.log[-1])
                    scored = sorted(
                        (_distance(r, shown), order, m)
                        for order, (m, r) in enumerate(cands)
                        if _kinds(r) == _kinds(shown)
                    )
                    if (
                        len(scored) > 1
                        and scored[0][0] == scored[1][0]
                        and scored[0][2] != scored[1][2]
                    ):
                        self.ambiguous += 1
                    move = scored[0][2] if scored else None
                    # The engine samples one branch of a random move choice;
                    # when that branch fits worse than the catalog's closest
                    # move, the catalog's wins.
                    alt = self._catalog_match(kind, shown, opening=False)
                    if alt is not None and (
                        move is None or self._catalog_fit(kind, alt, shown) < scored[0][0]
                    ):
                        move = alt
                if move is None:
                    move = self._catalog_match(kind, shown, opening=not st.log)
                if move is None:
                    self.unmatched += 1
                else:
                    st.log.append(move)
                st.shown, st.round = shown, rnd
            out.append(
                {
                    "move_state": st.log[-1] if st.log else None,
                    "ai": {"log": list(st.log), "performed_first_move": len(st.log) > 1},
                }
            )
        return out
