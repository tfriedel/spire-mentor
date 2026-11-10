"""Whole-RUN C#-vs-Rust parity driver.

The combat parity harness (``parity_audit.py``) compares a single combat arena
for up to N turns. This driver compares an entire RUN: Neow -> map traversal ->
combats -> events -> shops -> rest sites -> card/relic rewards -> act
transitions -> boss, carrying HP/deck/relics/gold/potions between nodes.

Design (mirrors the combat harness's "C# is the oracle" rule):

  * Both engines are driven through the shared ``GameBackend`` interface
    (``workbench/backends``): the C# ``Game`` subprocess and the Rust
    ``RustGame`` (PyO3 ``PyRunManager``). Both accept the SAME action verbs
    (``select_map_node`` / ``choose_option`` / ``play_card`` / ``end_turn`` /
    ``select_card_reward`` / ``buy_card`` / ...).
  * Every decision is enumerated from the **C# decision view** (the oracle) and
    a single choice is picked with a seeded RNG. The *same* verb is applied to
    BOTH engines. Rust is never consulted for the choice -- only diffed -- so a
    divergence in what Rust thinks is legal surfaces as an ``act`` rejection.
  * After each step both engines' canonical run-snapshots are diffed. The first
    field that differs is the reported divergence (phase + field + values).

Run:  scripts/wheel-venv.sh run -- python python/run_parity.py --seed FOO --steps 400

M6 productionized sweep (cached, parallel, gated, per-character) — the triage
surface for dividing run-parity work across agents (one per character):

  scripts/wheel-venv.sh run -- python python/run_parity.py \\
      --sweep --character Silent --seed RP --seeds 50 -j 8

  scripts/wheel-venv.sh run -- python python/run_parity.py \\
      --sweep --all-characters --seeds 20      # full-roster triage overview
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import hashlib
import itertools
import json
import os
import pathlib
import random
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, "python")

from sts2_client import add_game_version_arg, apply_game_version_arg, rs_engine_call
from workbench.backends.base import make_backend


# --------------------------------------------------------------------------- #
# Canonicalization: the two engines speak different vocabularies (C# localized
# display names + ``CARD.X`` ids; Rust raw ids + ``X.title`` loc-keys). Reduce
# both to an alnum-uppercase key so equal content compares equal.
# --------------------------------------------------------------------------- #
def _canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _canon_card_id(raw: str) -> str:
    """``"CARD.STRIKE_IRONCLAD"`` / ``"STRIKE_IRONCLAD"`` -> ``"STRIKEIRONCLAD"``."""
    s = str(raw)
    if s.upper().startswith("CARD."):
        s = s[5:]
    return _canon(s)


def _canon_reward_card(raw: str) -> str:
    """Reward-card id canon. Strips the ``CARD.`` prefix and a trailing
    ``CARD`` token so the Python pin's suffixed names (``RUPTURE_CARD``,
    ``VICIOUS_CARD``) compare equal to C#'s UI-stripped ``Id.Entry``
    (``RUPTURE``). Symmetric: both sides strip, so cards genuinely ending
    in CARD still match. Mirrors `names::card_id_from_str`'s `_CARD`
    fallback (sts2-rs-u5zc)."""
    c = _canon_card_id(raw)
    return c[:-4] if c.endswith("CARD") else c


# C# decision string -> canonical phase. Rust already emits canonical-ish names.
_CS_PHASE = {
    "map_select": "map",
    "combat_play": "combat",
    "event_choice": "event",
    "rest_site": "rest",
    "shop": "shop",
    "card_reward": "card_reward",
    "card_select": "card_select",
    "bundle_select": "bundle",
    "potion_swap": "potion_swap",
    "game_over": "game_over",
    "unknown": "unknown",
}
_RS_PHASE = {
    "map_choice": "map",
    # The Rust backend serializes decisions in C#'s vocab (`_PHASE_TO_DECISION`
    # → "map_select" / "combat_play"), not its raw phase names, so canon those
    # decision strings to the same buckets as `_CS_PHASE` — otherwise the
    # lockstep false-diverges on the phase label at every map / combat node
    # (`cs='map' rs='map_select'`, `cs='combat' rs='combat_play'`).
    "map_select": "map",
    "combat_play": "combat",
    "combat": "combat",
    "event_choice": "event",
    "rest_site": "rest",
    "shop": "shop",
    "card_reward": "card_reward",
    "relic_reward": "card_reward",
    "card_select": "card_select",
    "card_removal": "card_select",
    "boss_relic": "boss_relic",
    "bundle_select": "bundle",
    "treasure": "treasure",
    "potion_swap": "potion_swap",
    "game_over": "game_over",
}


def _phase_of(view: dict[str, Any], side: str) -> str:
    raw = view.get("decision") or view.get("type") or "?"
    if side == "cs":
        return _CS_PHASE.get(raw, raw)
    return _RS_PHASE.get(raw, raw)


def _player_of(view: dict[str, Any]) -> dict[str, Any]:
    return view.get("player") or {}


def _deck_multiset(player: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in player.get("deck") or []:
        if isinstance(c, dict):
            key = f"{_canon_card_id(c.get('id') or c.get('name') or '?')}|{int(bool(c.get('upgraded')))}"
        else:
            key = f"{_canon_card_id(c)}|0"
        out[key] = out.get(key, 0) + 1
    return out


def _enemy_hp_sig(view: dict[str, Any]) -> tuple[int, ...]:
    """Sorted multiset of ALIVE enemy HPs from a combat view.

    Combat-damage divergences land on the ENEMY, but the lockstep otherwise
    compares only player HP / deck / phase — so a card dealing different
    damage in Rust vs C# stays invisible until the enemy dies in one engine
    but not the other, surfacing far downstream as a phase split
    (`cs=card_reward rs=combat`). Comparing this signature during combat
    surfaces the divergence at its TRUE step (`combat/state/enemies_hp`).

    Sorted + alive-only on purpose: robust to enemy list ORDER (front/back
    serialization) and to a dead enemy being dropped from one engine's list
    but kept at hp 0 in the other — neither is a real damage divergence.
    """
    out: list[int] = []
    for e in view.get("enemies") or []:
        if isinstance(e, dict):
            hp = e.get("hp")
            if hp is not None and hp > 0:
                out.append(int(hp))
    return tuple(sorted(out))


def _relic_set(player: dict[str, Any]) -> list[str]:
    out = []
    for r in player.get("relics") or []:
        if isinstance(r, dict):
            # C# relic has display ``name`` (no id); Rust has ``id``.
            out.append(_canon(r.get("id") or r.get("name") or "?"))
        else:
            out.append(_canon(r))
    return sorted(out)


def _potion_list(player: dict[str, Any]) -> list[str]:
    out = []
    for p in player.get("potions") or []:
        val = (p.get("id") or p.get("name") or "?") if isinstance(p, dict) else p
        out.append(_canon(val))
    return out


@dataclass
class Snap:
    phase: str
    hp: int | None
    max_hp: int | None
    block: int | None
    gold: int | None
    deck: dict[str, int]
    deck_size: int
    relics: list[str]
    potions: list[str]
    enemies: tuple[int, ...]

    @classmethod
    def of(cls, view: dict[str, Any], side: str) -> Snap:
        p = _player_of(view)
        deck = _deck_multiset(p)
        return cls(
            phase=_phase_of(view, side),
            hp=p.get("hp"),
            max_hp=p.get("max_hp"),
            block=p.get("block"),
            gold=p.get("gold"),
            deck=deck,
            deck_size=sum(deck.values()) or p.get("deck_size") or 0,
            relics=_relic_set(p),
            potions=_potion_list(p),
            enemies=_enemy_hp_sig(view),
        )


def diff_snaps(cs: Snap, rs: Snap) -> list[str]:
    """Return a list of human-readable field divergences (empty == match)."""
    out: list[str] = []
    if cs.phase != rs.phase:
        out.append(f"phase: cs={cs.phase!r} rs={rs.phase!r}")
    for fld in ("hp", "max_hp", "gold"):
        cv, rv = getattr(cs, fld), getattr(rs, fld)
        if cv is not None and rv is not None and cv != rv:
            out.append(f"{fld}: cs={cv} rs={rv}")
    if cs.deck_size and rs.deck_size and cs.deck_size != rs.deck_size:
        out.append(f"deck_size: cs={cs.deck_size} rs={rs.deck_size}")
    elif cs.deck != rs.deck:
        only_cs = {k: v for k, v in cs.deck.items() if rs.deck.get(k) != v}
        only_rs = {k: v for k, v in rs.deck.items() if cs.deck.get(k) != v}
        out.append(f"deck: cs_only={only_cs} rs_only={only_rs}")
    if cs.relics != rs.relics:
        out.append(f"relics: cs={cs.relics} rs={rs.relics}")
    if len(cs.potions) != len(rs.potions):
        out.append(f"potions: cs={cs.potions} rs={rs.potions}")
    # Enemy HP — only meaningful (and only compared) while BOTH sides are in
    # combat. Surfaces combat-damage divergences at their true step instead of
    # the downstream `cs=card_reward rs=combat` phase split. Gated on both
    # phases == combat so a phase mismatch (reported above) doesn't also emit a
    # spurious enemy diff from one side's empty non-combat enemy list.
    if cs.phase == "combat" and rs.phase == "combat" and cs.enemies != rs.enemies:
        out.append(f"enemies_hp: cs={list(cs.enemies)} rs={list(rs.enemies)}")
    return out


# --------------------------------------------------------------------------- #
# Map signature: rows of node-types + boss + current coord. A foundational
# run-level RNG surface (map generation). Compared whenever in the map phase.
# --------------------------------------------------------------------------- #
def map_signature(backend: Any) -> dict[Any, Any] | None:
    """Coord-keyed map graph `{(col, row): (type, children)}` for the default
    lockstep's per-step map diff. Uses the SAME normalization as
    `--map-parity` (`_norm_map_graph`, proven 25/25), which keys nodes by
    `(col, row)` and skips the row-0 Ancient/start node that Rust's `rows`
    includes but C# omits — so the lockstep doesn't false-diverge on that
    start-node row offset (the earlier list-of-rows signature did: cs 15 rows
    vs rs 16). Bead `sts2-rs-1hiht` (M5 lockstep)."""
    try:
        m = backend.get_map()
    except Exception:
        return None
    if not isinstance(m, dict) or "rows" not in m:
        return None
    return _norm_map_graph(m)


def diff_maps(cs: dict | None, rs: dict | None) -> list[str]:
    out: list[str] = []
    if cs is None or rs is None or cs == rs:
        return out
    ck, rk = set(cs), set(rs)
    # Keys are heterogeneous: (col, row) coord tuples PLUS a "BOSS" string
    # entry (see _norm_map_graph). Python 3 can't order str against tuple, so
    # sort by str() to keep these sorts total without crashing the lockstep.
    only_cs, only_rs = sorted(ck - rk, key=str)[:5], sorted(rk - ck, key=str)[:5]
    if only_cs or only_rs:
        out.append(f"map.nodes: only_cs={only_cs} only_rs={only_rs}")
    for k in sorted(ck & rk, key=str):
        if cs[k] != rs[k]:
            out.append(f"map.node[{k}]: cs={cs[k]} rs={rs[k]}")
            break
    return out


def rust_finalize_atomic_select(r: Any, rs_view: dict[str, Any]) -> Any:
    """Finalize a Rust ``card_select`` that the C# headless resolved atomically.

    In combat the C# option-0 drain takes the prefix ``[0, max(min_select, 1))``.
    ``skip_select`` finalizes Rust with no pick, which matches only while the
    select is optional. A mandatory one (Tools of the Trade's discard,
    ``min_select`` 1) needs the same prefix, or the hands split (GH #1054,
    Silent:1007). The min-0 path keeps ``skip_select``, and a rejected prefix
    falls back to it.
    """
    cards = rs_view.get("cards") or []
    min_sel = int(rs_view.get("min_select") or 0)
    if min_sel >= 1 and cards:
        idxs = [str(c["index"]) for c in cards[: min(min_sel, len(cards))]]
        nxt = r.act("select_cards", indices=",".join(idxs))
        if not (isinstance(nxt, dict) and (nxt.get("error") or nxt.get("type") == "error")):
            return nxt
    return r.act("skip_select")


# --------------------------------------------------------------------------- #
# Choice enumeration from the C# decision view (the oracle). Each entry is a
# ``(verb, kwargs)`` pair applicable to both backends via ``act``.
# --------------------------------------------------------------------------- #
def enumerate_choices(g: Any, cs_view: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    dec = cs_view.get("decision") or cs_view.get("type")
    out: list[tuple[str, dict[str, Any]]] = []

    if dec == "event_choice" or dec == "rest_site":
        for o in cs_view.get("options") or []:
            # Rest options report `is_enabled` (RunSimulator.RestSiteState); a
            # disabled one (e.g. Smith with nothing upgradable) is not a choice.
            if not o.get("is_locked") and o.get("is_enabled", o.get("enabled", True)):
                out.append(("choose_option", {"option_index": o["index"]}))
        # MiniatureTent: after one option the rest site stays open and the
        # player may leave with the rest unused (C# BeforeLocalRestSiteExited).
        if dec == "rest_site" and cs_view.get("can_leave"):
            out.append(("leave_room", {}))

    elif dec == "map_select":
        for c in cs_view.get("choices") or []:
            out.append(("select_map_node", {"col": c["col"], "row": c["row"]}))

    elif dec == "card_reward":
        for c in cs_view.get("cards") or []:
            out.append(("select_card_reward", {"card_index": c["index"]}))
        if cs_view.get("can_skip", True):
            out.append(("skip_card_reward", {}))

    elif dec == "card_select":
        cards = cs_view.get("cards") or []
        min_sel = int(cs_view.get("min_select") or 0)
        if min_sel <= 1:
            # Single-pick: combat pending choices, single-card removals, deck
            # upgrades. One index per `select_cards`; skip allowed at min 0.
            for c in cards:
                out.append(("select_cards", {"indices": str(c["index"])}))
            if min_sel == 0:
                out.append(("skip_select", {}))
        else:
            # Multi-pick: C# presents a pick-N card_select (e.g. PrecariousShears
            # `FromDeckForRemoval(2)`, min==max==N). A single `select_cards` verb
            # must carry exactly N indices so the whole selection resolves in one
            # step on both engines (the Rust backend drains all N picks at once).
            # Enumerate size-N index combinations (capped — parity needs one
            # valid selection, not exhaustive coverage).
            idxs = [c["index"] for c in cards]
            for combo in itertools.islice(itertools.combinations(idxs, min_sel), 64):
                out.append(("select_cards", {"indices": ",".join(str(i) for i in combo)}))

    elif dec == "potion_swap":
        # GH #1069: decline, or discard held slot k to take the reward potion.
        for o in cs_view.get("options") or []:
            out.append(("potion_swap", {"discard_slot": o.get("discard_slot")}))

    elif dec == "bundle_select":
        for b in cs_view.get("bundles") or []:
            out.append(("select_bundle", {"bundle_index": b["index"]}))

    elif dec == "shop":
        for c in cs_view.get("cards") or []:
            if c.get("affordable", True) and not c.get("sold_out"):
                out.append(("buy_card", {"card_index": c["index"]}))
        for r in cs_view.get("relics") or []:
            if r.get("affordable", True) and not r.get("sold_out"):
                out.append(("buy_relic", {"relic_index": r["index"]}))
        for p in cs_view.get("potions") or []:
            if p.get("affordable", True) and not p.get("sold_out"):
                out.append(("buy_potion", {"potion_index": p["index"]}))
        out.append(("leave_room", {}))  # always allow leaving the shop

    elif dec == "combat_play":
        try:
            la = g.legal_actions()
            for a in la.get("actions") or []:
                verb = a.get("action")
                kw = {k: v for k, v in a.items() if k != "action"}
                out.append((verb, kw))
        except Exception:
            pass

    return out


def potion_swap_enabled() -> bool:
    """GH #1069: whether the lockstep runs both engines with
    ``interactive_potion_swap`` (a full-belt potion reward becomes a
    ``potion_swap`` decision, as on the real reward screen). On by default;
    ``--no-potion-swap`` sets ``RUN_PARITY_POTION_SWAP=0``, which pool
    workers inherit."""
    return os.environ.get("RUN_PARITY_POTION_SWAP", "1") != "0"


@dataclass
class Divergence:
    step: int
    phase: str
    fields: list[str]
    kind: str  # "state" | "map" | "act_reject" | "harness"


@dataclass
class RunResult:
    seed: str
    steps: int
    terminal: str  # "game_over_match" | "max_steps" | "diverged" | "error"
    cs_won: bool | None = None
    rs_won: bool | None = None
    divergence: Divergence | None = None
    trace: list[str] = field(default_factory=list)


def run_one(
    seed: str,
    character: str = "Ironclad",
    ascension: int = 0,
    max_steps: int = 400,
    stop_on_diverge: bool = True,
    verbose: bool = False,
) -> RunResult:
    # Seed the choice picker from the seed STRING directly: `random.Random`
    # hashes str/bytes seeds deterministically (sha512 internally), whereas
    # the built-in `hash(("runparity", seed))` is process-randomized
    # (PYTHONHASHSEED) for tuples — which made the lockstep pick different
    # choices every invocation, so a divergence wasn't reproducible run-to-run.
    rng = random.Random(f"runparity:{seed}")
    CS = make_backend("csharp")
    RS = make_backend("rust", target_convention="csharp")
    g, r = CS(), RS()
    res = RunResult(seed=seed, steps=0, terminal="error")
    try:
        swap = potion_swap_enabled()
        cs_view = g.start(
            character=character, seed=seed, ascension=ascension, interactive_potion_swap=swap
        )
        rs_view = r.start(
            character=character, seed=seed, ascension=ascension, interactive_potion_swap=swap
        )

        for step in range(max_steps):
            res.steps = step
            cs_dec = cs_view.get("decision") or cs_view.get("type")

            # ---- diff state ----
            cs_snap, rs_snap = Snap.of(cs_view, "cs"), Snap.of(rs_view, "rs")

            # Atomic-select-skip guards (same reconciliation as the cached
            # ``_replay_against_rust`` / ``act23_state_parity`` drivers): one
            # engine surfaces a card_select the OTHER resolved ATOMICALLY. Advance
            # the surfacing engine ALONE to its atomic-equivalent resolution, then
            # re-snap; the next diff still catches a genuinely divergent pick.
            #   (a) cs=card_select, rs!=select → drive C# with its first legal pick.
            #   (b) rs=card_select, cs!=select → finalize Rust via skip_select.
            # Both gated so a REAL shared card_select (both paused) is driven
            # normally below. Oracle-artifact "run-combat-cardselect".
            if cs_snap.phase == "card_select" and rs_snap.phase != "card_select":
                cs_choices = enumerate_choices(g, cs_view)
                if cs_choices:
                    verb0, kw0 = cs_choices[0]
                    nxt_cs = g.act(verb0, **kw0)
                    if not (isinstance(nxt_cs, dict) and nxt_cs.get("type") == "error"):
                        cs_view = nxt_cs
                        continue
            sel_guard = 0
            while (
                rs_snap.phase == "card_select" and cs_snap.phase != "card_select" and sel_guard < 8
            ):
                nxt_rs = rust_finalize_atomic_select(r, rs_view)
                if isinstance(nxt_rs, dict) and (
                    nxt_rs.get("error") or nxt_rs.get("type") == "error"
                ):
                    break
                rs_view = nxt_rs
                rs_snap = Snap.of(rs_view, "rs")
                sel_guard += 1

            # Event-granularity skip: C# headless auto-skips Custom-layout events
            # (FakeMerchant → empty options → MapRoom) so it is already on the map
            # while Rust surfaces the event. Drive Rust ALONE through its no-op
            # ``leave`` option (no shared run RNG — private ``FAKE_MERCHANT``
            # stream), then re-snap. Bead sts2-rs-1hiht.
            if rs_snap.phase == "event" and cs_snap.phase == "map":
                leave = next(
                    (
                        o
                        for o in (rs_view.get("options") or [])
                        if o.get("option_id") == "leave" and not o.get("is_locked")
                    ),
                    None,
                )
                if leave is not None:
                    nxt_rs = r.act("choose_option", option_index=leave["index"])
                    if not (
                        isinstance(nxt_rs, dict)
                        and (nxt_rs.get("error") or nxt_rs.get("type") == "error")
                    ):
                        rs_view = nxt_rs
                        rs_snap = Snap.of(rs_view, "rs")

            fields = diff_snaps(cs_snap, rs_snap)
            if cs_snap.phase == "map":
                fields += diff_maps(map_signature(g), map_signature(r))
            if verbose:
                res.trace.append(
                    f"[{step:03d}] {cs_snap.phase:11s} hp={cs_snap.hp} gold={cs_snap.gold} deck={cs_snap.deck_size} relics={len(cs_snap.relics)}"
                )
            if fields:
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    fields,
                    "map" if any(f.startswith("map") for f in fields) else "state",
                )
                res.terminal = "diverged"
                if stop_on_diverge:
                    return res

            # ---- terminal ----
            if cs_dec == "game_over":
                res.cs_won = bool(cs_view.get("victory") or cs_view.get("won"))
                res.rs_won = bool(
                    rs_view.get("victory")
                    or rs_view.get("won")
                    or _phase_of(rs_view, "rs") == "game_over"
                )
                res.terminal = (
                    "game_over_match" if (_phase_of(rs_view, "rs") == "game_over") else "diverged"
                )
                if _phase_of(rs_view, "rs") != "game_over":
                    res.divergence = Divergence(
                        step, "game_over", [f"cs=game_over rs={rs_snap.phase}"], "state"
                    )
                return res

            # ---- choose (off C# oracle) + apply to both ----
            choices = enumerate_choices(g, cs_view)
            if not choices:
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    [f"no choices enumerable for cs decision {cs_dec!r}"],
                    "harness",
                )
                res.terminal = "error"
                return res
            verb, kw = choices[rng.randrange(len(choices))]

            cs_view = g.act(verb, **kw)
            rs_view = r.act(verb, **kw)

            # ---- act rejection ----
            cs_err = isinstance(cs_view, dict) and cs_view.get("type") == "error"
            # Rust signals a rejected action with ``{"type": "error", "message":
            # ...}`` (no ``error`` key) — read both so the reported message is
            # never spuriously ``None``.
            rs_err_msg = None
            if isinstance(rs_view, dict) and (
                rs_view.get("error") or rs_view.get("type") == "error"
            ):
                rs_err_msg = rs_view.get("error") or rs_view.get("message")
            rs_err = rs_err_msg is not None
            # Both reject => agreement: the random policy offered an illegal move
            # (e.g. an unaffordable buy, enumerated regardless of price), C# can't
            # proceed, so end the run cleanly rather than flag a spurious
            # divergence. Only a DISAGREEMENT (exactly one side rejects) is a real
            # act_reject divergence.
            if cs_err and rs_err:
                res.terminal = "cs_act_reject"
                return res
            if cs_err or rs_err:
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    [
                        f"act({verb},{kw}) -> cs_err={cs_view.get('message') if cs_err else None} rs_err={rs_err_msg}"
                    ],
                    "act_reject",
                )
                res.terminal = "diverged"
                if stop_on_diverge:
                    return res
        res.terminal = "max_steps"
        return res
    except Exception as exc:  # harness-level failure
        import traceback

        res.divergence = Divergence(res.steps, "?", [f"{type(exc).__name__}: {exc}"], "harness")
        res.terminal = "error"
        if verbose:
            res.trace.append(traceback.format_exc())
        return res
    finally:
        try:
            g.close()
        except Exception:
            pass
        try:
            r.close()
        except Exception:
            pass


def _canon_boss(s: str) -> str:
    c = _canon(s)
    return c[:-4] if c.endswith("BOSS") else c


def act_parity(base: str, character: str, ascension: int, n: int) -> int:
    """Seed-driven ACT-VARIANT parity: for each seed, which Act 1 does the run
    roll -- Overgrowth or Underdocks?

    This is the gate that no previous check could be: `--rng-parity`,
    `--encounter-parity` and `--map-parity` all compare a C# oracle that never
    rolls an act (`RunState.CreateForTest` falls back to
    `ActModel.GetDefaultList()` => Overgrowth) against a Rust engine that
    hardcodes Overgrowth. Overgrowth vs Overgrowth agrees by construction, so
    those checks could not have failed even though the engine cannot produce
    Underdocks -- Lagavulin Matriarch, the toadpoles, the whole act -- at all.

    The roll lives on its own `act_selection` stream in
    `StartRunLobby.BeginRunLocally`, so this compares the C# `roll_acts`
    command against the Rust `sts2_rs.roll_acts` port. It is a check on the
    SELECTION only; `--underdocks-parity` covers the content.

    Returns the matching-seed count.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    ok = 0
    pool_was_open = False
    g = Game()
    try:
        for i in range(n):
            seed = base if n == 1 else f"{base}{i}"
            cs = g.send({"cmd": "roll_acts", "seed": seed})
            cs_acts = [_canon(a) for a in cs.get("acts") or []]
            unlocked = bool(cs.get("underdocks_unlocked"))
            pool_was_open |= unlocked
            rs_acts = [
                _canon(a)
                for a in rs_engine_call(sts2_rs.roll_acts, seed, underdocks_unlocked=unlocked)
            ]
            match = cs_acts == rs_acts
            ok += match
            if not match:
                print(f"[{seed}] DIFF cs_acts={cs_acts} rs_acts={rs_acts}")
    finally:
        g.close()

    print(f"\nact-selection parity: {ok}/{n} seeds")
    if not pool_was_open:
        # Guard against the exact failure mode this check exists to prevent:
        # a green number produced by a pool with one member in it.
        print(
            "WARNING: Underdocks was LOCKED for every seed, so the act-1 pool was a "
            "singleton and this result is VACUOUS. It would read 100% even if the "
            "selection port were wrong."
        )
    return ok


def underdocks_parity(base: str, character: str, ascension: int, n: int) -> int:
    """Act-1 CONTENT parity for Underdocks: force BOTH engines onto Underdocks
    (`start_run act1=underdocks` / `RunManager(act1="underdocks")`) and compare
    the pre-generated room lists seed-for-seed.

    This is the regression test for GH #544. It failed by construction until
    the port landed: `run_room_gen::ACTS` pinned Act 1 = Overgrowth and
    `sts2-content` had no `encounters/act1_underdocks.rs`, so the engine
    answered Vantom / SLIMES_WEAK where the real game answers Lagavulin
    Matriarch / TOADPOLES_WEAK.

    Note it FORCES the variant on both sides rather than letting the seed roll
    it. `--act-parity` is what checks the roll; this checks the content, and
    forcing is what makes every seed exercise Underdocks instead of the ~50%
    a roll would give.

    Returns the number of seeds whose act-1 lists match.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    ok = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension, act1="underdocks")
            cs = g.send({"cmd": "dump_rooms"}).get("acts", [])
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
            act1="underdocks",
        )
        rs = rm.generated_rooms()
        if not cs or not rs:
            print(f"[{seed}] no rooms dumped (cs={len(cs)} rs={len(rs)})")
            continue
        # Compare EVERY act-1 list the oracle exposes, not just the boss and
        # the normals. The port touches the encounter table, the elite pool and
        # the event pool independently, and each is driven by its own stretch
        # of the `up_front` stream -- a check that only looked at two of them
        # could pass with a wrong event shuffle underneath it.
        cb, rb = _canon(cs[0]["boss"]), _canon(rs[0]["boss"])
        lists = {
            "normal": (
                [_canon(x) for x in cs[0]["normal"]],
                [_canon(x) for x in rs[0]["normal"]],
            ),
            "elite": (
                [_canon(x) for x in cs[0]["elite"]],
                [_canon(x) for x in rs[0]["elite"]],
            ),
            "events": (
                [_canon(x) for x in cs[0]["events"]],
                [_canon(x) for x in rs[0]["events_shuffled"]],
            ),
        }
        match = cb == rb and all(c == r for c, r in lists.values())
        ok += match
        if not match:
            if cb != rb:
                print(f"[{seed}] DIFF act1 boss cs={cb} rs={rb}")
            for name, (c, r) in lists.items():
                if c == r:
                    continue
                if len(c) != len(r):
                    print(f"[{seed}] DIFF act1 {name} length cs={len(c)} rs={len(r)}")
                for j, (a, b) in enumerate(zip(c, r)):
                    if a != b:
                        print(f"[{seed}]     {name}[{j}]: cs={a} rs={b}")
                        break
    print(f"\nUnderdocks act-1 content parity: {ok}/{n} seeds")
    return ok


def rng_parity(base: str, character: str, ascension: int, n: int) -> int:
    """Representation-robust run-level RNG check: does the act-1 boss (rolled
    from the run-setup RNG) match seed-for-seed? This isolates run-level RNG
    alignment from decision-granularity skew. Returns the match count.
    """
    CS, RS = make_backend("csharp"), make_backend("rust", target_convention="csharp")
    match = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g, r = CS(), RS()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            r.start(character=character, seed=seed, ascension=ascension)
            cb = g.get_map().get("boss") or {}
            rb = r.get_map().get("boss") or {}
            cbn = _canon_boss(cb.get("id") or cb.get("name") or "?")
            rbn = _canon_boss(rb.get("id") or rb.get("kind") or rb.get("name") or "?")
            ok = cbn == rbn
            match += ok
            print(f"[{seed}] cs_boss={cbn:20s} rs_boss={rbn:20s} {'OK' if ok else 'DIFF'}")
        finally:
            g.close()
            r.close()
    print(f"\nact-1 boss parity: {match}/{n}")
    return match


def _norm_map_graph(m: dict[str, Any]) -> dict[Any, Any]:
    """Reduce a `full_map`/`get_map` dict to a coord-keyed graph for comparison:
    `{(col, row): (node_type, sorted_children_coords)}` plus a `"BOSS"` entry.
    Skips the row-0 Ancient/start node (present in the Rust `rows` but omitted
    from C#'s), so the two align on the shared `(col, row)` coordinate system
    despite the list row-offset / start-node serialization skew.
    """
    nodes: dict[Any, Any] = {}
    for row in m.get("rows") or []:
        for node in row or []:
            r = node.get("row")
            if r == 0:
                continue
            ch = tuple(sorted((c.get("col"), c.get("row")) for c in (node.get("children") or [])))
            nodes[node.get("col"), r] = (_canon(node.get("type") or node.get("kind") or "?"), ch)
    b = m.get("boss") or {}
    nodes["BOSS"] = (_canon(b.get("type") or "?"), (b.get("col"), b.get("row")))
    return nodes


def map_parity(base: str, character: str, ascension: int, n: int) -> int:
    """M2 map-graph parity: does the act-1 map node-type grid + edges + boss
    position match C# vs Rust? The map graph is generated off the separate
    `act_{N}_map` stream; this confirms end-to-end that it's C#-faithful (the
    `standard_act_map_matches_csharp_snapshot` unit test pins gen given the
    stream; this drives it through the live harness). Returns matching-seed count.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    ok = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            cm = g.get_map()
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )
        cn, rn = _norm_map_graph(cm), _norm_map_graph(rm.full_map())
        if cn == rn:
            ok += 1
        else:
            ck, rk = set(cn), set(rn)
            print(f"[{seed}] DIFF only_cs={sorted(ck - rk)[:5]} only_rs={sorted(rk - ck)[:5]}")
            for k in sorted(ck & rk):
                if cn[k] != rn[k]:
                    print(f"    {k}: cs={cn[k]} rs={rn[k]}")
    print(f"\nmap-graph parity (node-types + edges + boss): {ok}/{n} seeds")
    return ok


def encounter_parity(base: str, character: str, ascension: int, n: int) -> int:
    """M3 encounter-identity parity: do the per-act pre-generated room lists
    (`RoomSet.normalEncounters` / `eliteEncounters` / `Boss`, produced by C#'s
    `GenerateRooms` off the shared `up_front` stream) match the Rust
    `run_room_gen` output entry-for-entry, across all three acts?

    This is stronger than `--rng-parity` (which checks only the act-1 boss): it
    asserts every monster/elite encounter identity in every act. Uses the
    headless `dump_rooms` command and the PyO3 `generated_rooms()` accessor
    directly (the lists are pre-generated at run start, so no map walk needed).
    Returns the number of fully-matching (act) lists.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    acts_ok = acts_total = 0
    norm_ok = norm_total = elite_ok = elite_total = boss_ok = boss_total = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            cs = g.send({"cmd": "dump_rooms"}).get("acts", [])
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )
        rs = rm.generated_rooms()
        for ai in range(min(len(cs), len(rs))):
            cn = [_canon(x) for x in cs[ai]["normal"]]
            rn = [_canon(x) for x in rs[ai]["normal"]]
            ce = [_canon(x) for x in cs[ai]["elite"]]
            re_ = [_canon(x) for x in rs[ai]["elite"]]
            cb, rb = _canon(cs[ai]["boss"]), _canon(rs[ai]["boss"])
            norm_total += len(cn)
            elite_total += len(ce)
            boss_total += 1
            norm_ok += sum(1 for a, b in zip(cn, rn) if a == b)
            elite_ok += sum(1 for a, b in zip(ce, re_) if a == b)
            boss_ok += cb == rb
            act_ok = cn == rn and ce == re_ and cb == rb
            acts_total += 1
            acts_ok += act_ok
            if not act_ok:
                print(f"[{seed} act{ai + 1}] DIFF")
                if cn != rn:
                    print(f"    normal cs={cs[ai]['normal']}\n           rs={rs[ai]['normal']}")
                if ce != re_:
                    print(f"    elite  cs={cs[ai]['elite']}\n           rs={rs[ai]['elite']}")
                if cb != rb:
                    print(f"    boss   cs={cs[ai]['boss']} rs={rs[ai]['boss']}")
    print(
        f"\nencounter parity ({n} seeds x 3 acts): acts {acts_ok}/{acts_total} | "
        f"normal {norm_ok}/{norm_total} | elite {elite_ok}/{elite_total} | "
        f"boss {boss_ok}/{boss_total}"
    )
    return acts_ok


def reward_parity(
    base: str, character: str, ascension: int, n: int, room_type: str = "Monster"
) -> int:
    """M4 card-reward parity: does a single regular-encounter card reward,
    rolled off the player's fresh `PlayerRng.Rewards` stream (counter 0,
    right after run start), match the Rust `roll_card_reward_cards` output
    card-for-card — same rarity slate, same character-pool `NextItem`
    pick, same act-scaled upgrade flag?

    This isolates the reward RNG + character-pool ordering (the one piece
    of the reward flow that static audit can't verify: C#
    `CardPoolModel.GetUnlockedCards` order vs Rust `eligible_ids_for_rarity`)
    with NO combat plumbing — both engines just roll a reward on a fresh
    run. C#: `dump_card_reward` -> `CardFactory.CreateForReward(player, 3,
    CardCreationOptions.ForRoom(player, RoomType.Monster))`. Rust:
    `RunManager.dump_card_reward()`. Both consume reads
    `[rarity, pick, upgrade]` per card on the single Rewards stream.

    Returns the number of fully-matching rewards (out of `n`).
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    rewards_ok = 0
    cards_ok = cards_total = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            cs = g.send({"cmd": "dump_card_reward", "room_type": room_type}).get("cards", [])
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )
        rs = rm.dump_card_reward(3, room_type)

        def _key(c: object) -> tuple[str, bool]:
            d = c if isinstance(c, dict) else {}
            return (_canon_reward_card(d.get("id", "?")), bool(d.get("upgraded", False)))

        cs_keys = [_key(c) for c in cs]
        rs_keys = [_key(c) for c in rs]
        cards_total += max(len(cs_keys), len(rs_keys))
        cards_ok += sum(1 for a, b in zip(cs_keys, rs_keys) if a == b)
        ok = cs_keys == rs_keys
        rewards_ok += ok
        if not ok:
            print(f"[{seed}] REWARD DIFF\n    cs={cs}\n    rs={rs}")
    print(
        f"\nreward parity ({n} seeds, {room_type}): rewards {rewards_ok}/{n} | "
        f"cards {cards_ok}/{cards_total}"
    )
    return rewards_ok


def relic_reward_parity(base: str, character: str, ascension: int, n: int, count: int = 1) -> int:
    """M4 relic-reward parity: do the first `count` relic rewards, rolled off
    the player's fresh `PlayerRng.Rewards` stream (counter 0, right after run
    start), match the Rust output relic-for-relic — same reward-stream rarity
    roll AND same grab-bag ordering (which `up_front` shuffle produced)?

    C#: `RelicFactory.PullNextRelicFromFront(player)` per pull — `RollRarity`
    off `PlayerRng.Rewards`, then a front-of-bucket grab-bag pop. Rust:
    `RunManager.dump_relic_reward(count)` — `roll_reward_relic_rarity` (same
    `rewards` stream) + `pop_relic_from_grab_bag`. This isolates the reward-RNG
    rarity buckets + relic grab-bag order (the `up_front` two-bag shuffle,
    already verified for the SmallCapsule after-obtain roll) with no combat
    plumbing. Returns the number of fully-matching reward slates (out of `n`).
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    slates_ok = 0
    relics_ok = relics_total = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            cs = g.send({"cmd": "dump_relic_reward", "count": count}).get("relics", [])
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )
        rs = rm.dump_relic_reward(count)

        def _rid(r: object) -> str:
            d = r if isinstance(r, dict) else {}
            return _canon(d.get("id", "?"))

        cs_keys = [_rid(r) for r in cs]
        rs_keys = [_rid(r) for r in rs]
        relics_total += max(len(cs_keys), len(rs_keys))
        relics_ok += sum(1 for a, b in zip(cs_keys, rs_keys) if a == b)
        ok = cs_keys == rs_keys
        slates_ok += ok
        if not ok:
            print(f"[{seed}] RELIC REWARD DIFF\n    cs={cs}\n    rs={rs}")
    print(
        f"\nrelic-reward parity ({n} seeds, count={count}): slates {slates_ok}/{n} | "
        f"relics {relics_ok}/{relics_total}"
    )
    return slates_ok


def shop_parity(base: str, character: str, ascension: int, n: int) -> int:
    """M4 shop parity: does a full normal-merchant inventory, generated off
    the player's fresh `Shops` + `Rewards` streams (counter 0, right after run
    start), match the Rust output — same per-stream read counts (25 Shops / 14
    Rewards on a fresh run), same 5 colored + 2 colorless card identities /
    upgraded / on-sale / price, same 3 potion identities / price, same removal
    cost?

    C#: `MerchantInventory.CreateForNormalMerchant`. Rust:
    `RunManager.dump_shop` → `generate_merchant_inventory`. Both walk the exact
    C# two-stream consumption order (card-rarity roll + per-card upgrade no-op +
    relic RollRarity off `Rewards`; sale index + card pick + cost jitter +
    potion roll off `Shops`).

    Relic-slot identity IS gated: each slot rolls its rarity, pulls from the
    BACK of the grab bag filtered by `IsAllowedInShops`
    (`MerchantRelicEntry.FillSlot` → `PullNextRelicFromBack`), and jitters its
    `MerchantCost` (175/225/275/200) — Rust mirrors all three, so relic id +
    price are checked alongside cards/potions.

    Returns the number of fully-matching shops (out of `n`), where "matching"
    covers read counts + card/colorless/relic/potion identities + prices +
    removal.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    shops_ok = 0
    reads_ok = cards_ok = cards_total = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            cs = g.send({"cmd": "dump_shop"})
        finally:
            g.close()
        rm = rs_engine_call(
            sts2_rs.RunManager,
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )
        rs = rm.dump_shop()

        def _card_key(c: object) -> tuple[str, bool, bool, int]:
            d = c if isinstance(c, dict) else {}
            return (
                _canon_reward_card(d.get("id", "?")),
                bool(d.get("upgraded", False)),
                bool(d.get("on_sale", False)),
                int(d.get("price", -1)),
            )

        def _potion_key(p: object) -> tuple[str, int]:
            d = p if isinstance(p, dict) else {}
            return (_canon(d.get("id", "?")), int(d.get("price", -1)))

        def _relic_key(r: object) -> tuple[str, int]:
            d = r if isinstance(r, dict) else {}
            return (_canon(d.get("id", "?")), int(d.get("price", -1)))

        reads_match = cs.get("shops_reads") == rs.get("shops_reads") and cs.get(
            "rewards_reads"
        ) == rs.get("rewards_reads")
        reads_ok += reads_match

        # Gating identity surface: colored + colorless cards, relics, potions,
        # removal.
        cs_cards = [_card_key(c) for c in cs.get("cards", [])] + [
            _card_key(c) for c in cs.get("colorless", [])
        ]
        rs_cards = [_card_key(c) for c in rs.get("cards", [])] + [
            _card_key(c) for c in rs.get("colorless", [])
        ]
        cs_relics = [_relic_key(r) for r in cs.get("relics", [])]
        rs_relics = [_relic_key(r) for r in rs.get("relics", [])]
        cs_pots = [_potion_key(p) for p in cs.get("potions", [])]
        rs_pots = [_potion_key(p) for p in rs.get("potions", [])]
        cards_total += (
            max(len(cs_cards), len(rs_cards))
            + max(len(cs_relics), len(rs_relics))
            + max(len(cs_pots), len(rs_pots))
        )
        cards_ok += sum(1 for a, b in zip(cs_cards, rs_cards) if a == b)
        cards_ok += sum(1 for a, b in zip(cs_relics, rs_relics) if a == b)
        cards_ok += sum(1 for a, b in zip(cs_pots, rs_pots) if a == b)

        ident_match = (
            cs_cards == rs_cards
            and cs_relics == rs_relics
            and cs_pots == rs_pots
            and cs.get("removal_cost") == rs.get("removal_cost")
        )
        ok = reads_match and ident_match
        shops_ok += ok
        if not ok:
            print(f"[{seed}] SHOP DIFF reads={reads_match} ident={ident_match}")
            print(
                f"    reads cs=({cs.get('shops_reads')},{cs.get('rewards_reads')}) "
                f"rs=({rs.get('shops_reads')},{rs.get('rewards_reads')})"
            )
            if cs_cards != rs_cards:
                print(f"    cards cs={cs_cards}\n          rs={rs_cards}")
            if cs_relics != rs_relics:
                print(f"    relics cs={cs_relics}\n           rs={rs_relics}")
            if cs_pots != rs_pots:
                print(f"    potions cs={cs_pots}\n            rs={rs_pots}")
    print(
        f"\nshop parity ({n} seeds): shops {shops_ok}/{n} | reads {reads_ok}/{n} | "
        f"cards+relics+potions {cards_ok}/{cards_total}"
    )
    return shops_ok


def roster_parity(base: str, character: str, ascension: int, n: int) -> int:
    """M3b roster parity: for each `up_front`-generated encounter, does the
    spawned combat ROSTER (monster kinds + per-monster max HP) match C#?

    This is the oracle that gates the deferred M3b RNG-shift sub-part. C#
    rolls every enemy's HP off the run-level `niche` stream
    (`CombatState.CreateCreature` → `Creature.SetUniqueMonsterHpValue(
    RunState.Rng.Niche)`); on a freshly-started run the `niche` cursor is at
    0, so a single `enter_room` + `get_state` is the ground truth for that
    encounter's *first-combat* HP roll. The Rust `RunManager.dump_roster`
    rolls the same encounter off its fresh `niche@0` via
    `dump_roster_for_encounter` (the same path, with no full-run replay).

    Two surfaces are reported separately because they fail independently:
    - **kinds** — roster fidelity (do the EncounterDef monster lists match
      C#?). Independent of the RNG migration.
    - **hp** — the per-monster `niche` HP roll. Until `build_combat_for_room`
      is migrated off the `run_manager` stream onto `niche`, the LIVE run path
      still diverges here; this oracle measures whether `dump_roster`'s
      niche-based roll (the migration target) is already C#-faithful.

    Each encounter spawns in its OWN fresh C# subprocess (the run/combat
    singletons retain state across an in-process re-`start`, so reuse is
    unsound); the Rust side builds a fresh `RunManager` per encounter for the
    same niche@0 isolation. Returns the number of encounters whose kinds match.
    """
    import sts2_rs  # type: ignore[import-not-found]
    from sts2_client import Game

    def _mk(character_seed: str) -> Any:
        return rs_engine_call(
            sts2_rs.RunManager,
            string_seed=character_seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            interactive_card_transform=True,
            direct_event_relic_grant=True,
        )

    def _cs_roster(seed: str, eid: str) -> list[tuple[str, int]]:
        """C# ground truth: spawn `eid` on a fresh run (niche@0) and read the
        spawned roster off `get_state`. Fresh subprocess per call."""
        g = Game()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            g.enter_room("combat", encounter=eid)
            mons = [m for m in g.get_state().get("monsters", []) if m.get("kind")]
        finally:
            g.close()
        return [(_canon(m.get("kind", "?")), int(m.get("max_hp", -1))) for m in mons]

    kinds_ok = kinds_total = 0
    hp_ok = hp_total = mon_ok = mon_total = 0
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        rooms = _mk(seed).generated_rooms()
        # Collect every combat encounter id across all acts (dedup — at niche@0
        # the same id spawns identically regardless of act/slot).
        enc_ids: list[str] = []
        seen: set[str] = set()
        for act in rooms:
            for slot in ("normal", "elite"):
                for eid in act.get(slot, []):
                    if eid and eid not in seen:
                        seen.add(eid)
                        enc_ids.append(eid)
            boss = act.get("boss")
            if boss and boss not in seen:
                seen.add(boss)
                enc_ids.append(boss)

        for eid in enc_ids:
            cs_roster = _cs_roster(seed, eid)
            rs_roster = [
                (_canon(r.get("id", "?")), int(r.get("max_hp", -1)))
                for r in _mk(seed).dump_roster(eid)
            ]

            cs_kinds = [k for k, _ in cs_roster]
            rs_kinds = [k for k, _ in rs_roster]
            kinds_total += 1
            kinds_match = cs_kinds == rs_kinds
            kinds_ok += kinds_match

            # Per-monster HP (only where kinds line up positionally).
            hp_total += 1
            hp_match = cs_roster == rs_roster
            hp_ok += hp_match
            mon_total += max(len(cs_roster), len(rs_roster))
            mon_ok += sum(1 for a, b in zip(cs_roster, rs_roster) if a == b)

            if not kinds_match or not hp_match:
                tag = "KINDS" if not kinds_match else "HP"
                print(f"[{seed} {eid}] {tag} DIFF")
                print(f"    cs={cs_roster}")
                print(f"    rs={rs_roster}")
    print(
        f"\nroster parity ({n} seeds): kinds {kinds_ok}/{kinds_total} | "
        f"hp {hp_ok}/{hp_total} | per-monster {mon_ok}/{mon_total}"
    )
    return kinds_ok


def rng_counters(base: str, character: str, ascension: int, n: int) -> None:
    """M0 diagnostic: dump each engine's run-RNG stream cursors right after
    `start`. Proves the root cause — C# advances `up_front` (boss rolled there);
    Rust advances `run_manager` instead and leaves `up_front` short.
    """
    CS, RS = make_backend("csharp"), make_backend("rust", target_convention="csharp")
    for i in range(n):
        seed = base if n == 1 else f"{base}{i}"
        g, r = CS(), RS()
        try:
            g.start(character=character, seed=seed, ascension=ascension)
            r.start(character=character, seed=seed, ascension=ascension)
            cc = (g.get_state() or {}).get("rng_counters") or {}
            rm = getattr(r, "rm", None)
            rc = rm.rng_counters() if rm is not None else {}
            keys = ["up_front", "shuffle", "run_manager", "monster_ai", "niche"]
            cs_s = " ".join(f"{k}={cc.get(k, '-')}" for k in keys)
            rs_s = " ".join(f"{k}={rc.get(k, '-')}" for k in keys)
            print(f"[{seed}]\n  cs: {cs_s}\n  rs: {rs_s}")
        finally:
            g.close()
            r.close()


# --------------------------------------------------------------------------- #
# M6 — productionized cached sweep (mirrors ``parity_audit.py``).
#
# The lockstep driver above spawns a fresh C# .NET subprocess per seed. For a
# per-character sweep that an agent re-runs many times while closing
# divergences, that subprocess dominates wall-clock. But the C# trajectory for a
# given (seed, character, ascension, max_steps) is **fully deterministic and
# Rust-independent**: every choice is enumerated off the C# oracle and picked
# with the seed-seeded `random.Random(f"runparity:{seed}")`, and C#'s next view
# comes from `g.act(verb)` with no Rust input. So we can RECORD the whole C#
# trajectory once and REPLAY it against Rust — diffing each step against the
# cached C# snapshot — with the .NET process never spawned again (the run-level
# analogue of `parity_audit`'s `_ReplayCsOracle`).
#
# Invalidation rides the cache NAMESPACE: `cs_version_namespace()` (a content
# hash of the C# oracle — lib/*.dll + src/**/*.cs) PLUS a hash of THIS harness
# file (so any change to the choice picker / enumerate_choices / snapshot logic
# auto-invalidates the recorded trajectories, which encode those choices).
# --------------------------------------------------------------------------- #

CHARACTERS: tuple[str, ...] = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
_RUN_TRACE_FORMAT = 1


def _harness_hash() -> str:
    """8-hex content hash of this harness file — folded into the cache namespace
    so a change to the choice picker / enumeration / snapshot logic (which the
    recorded C# trajectories bake in) auto-invalidates the cache."""
    h = hashlib.sha256()
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            h.update(f.read())
    except OSError:
        return "00000000"
    return h.hexdigest()[:8]


class _StepWatchdog:
    """Per-step deadline for the C# recording that works on every platform.

    It used to be SIGALRM, which Windows does not have: every recorded run died
    at step 0 with `module 'signal' has no attribute 'SIGALRM'`, so a Windows
    sweep reported 0/100 clean. A timer thread instead calls ``on_trip`` (the
    backend's ``close``, which terminates the headless process); the main
    thread's blocked read then fails with EOF and the caller, seeing
    ``tripped``, books the run as ``record_timeout``. Unlike a signal it also
    works off the main thread.
    """

    def __init__(self, timeout_s: int, on_trip: Callable[[], None]) -> None:
        self._timeout_s = timeout_s
        self._on_trip = on_trip
        self._timer: threading.Timer | None = None
        self.tripped = False

    def reset(self) -> None:
        """Restart the deadline (a no-op when the watchdog is off)."""
        self.cancel()
        if self._timeout_s > 0:
            self._timer = threading.Timer(self._timeout_s, self._trip)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _trip(self) -> None:
        self.tripped = True
        try:
            self._on_trip()
        except Exception:
            pass


def _record_cs_trajectory(
    seed: str, character: str, ascension: int, max_steps: int
) -> dict[str, Any]:
    """Drive ONLY the C# oracle through the seeded lockstep and capture the full
    trajectory: per step the C# view going in, the raw map (when in the map
    phase), and the chosen ``(verb, kwargs)``. Rust is never consulted, so the
    result is a faithful, replayable record of the C# side of ``run_one``.

    Each entry ``i`` holds the view at the TOP of loop iteration ``i`` (so the
    NEXT step's incoming view == ``entries[i]['view']`` after the act). The final
    entry carries ``verb=None`` (terminal: ``game_over`` / no enumerable choice).
    """
    # Per-step wall-clock watchdog: the C# headless can wedge on a
    # swallowed-NRE stuck turn for specific seeds, which would otherwise hang a
    # whole bundle bake forever. `STS2_RUNPARITY_RECORD_TIMEOUT` (seconds, 0=off,
    # set by pack.sh) abandons a wedged record as `record_timeout` instead.
    timeout_s = int(os.environ.get("STS2_RUNPARITY_RECORD_TIMEOUT", "0") or "0")

    rng = random.Random(f"runparity:{seed}")
    CS = make_backend("csharp")
    g = CS()
    watchdog = _StepWatchdog(timeout_s, g.close)
    entries: list[dict[str, Any]] = []
    terminal = "max_steps"
    try:
        watchdog.reset()
        view = g.start(
            character=character,
            seed=seed,
            ascension=ascension,
            interactive_potion_swap=potion_swap_enabled(),
        )
        for _ in range(max_steps):
            watchdog.reset()  # per step: trips only a wedged step
            dec = view.get("decision") or view.get("type")
            in_map = Snap.of(view, "cs").phase == "map"
            map_raw = g.get_map() if in_map else None

            if dec == "game_over":
                entries.append({"view": view, "map": map_raw, "verb": None, "kwargs": {}})
                terminal = "game_over"
                break

            choices = enumerate_choices(g, view)
            if not choices:
                entries.append({"view": view, "map": map_raw, "verb": None, "kwargs": {}})
                terminal = "no_choices"
                break

            verb, kw = choices[rng.randrange(len(choices))]
            after = g.act(verb, **kw)
            cs_err = isinstance(after, dict) and after.get("type") == "error"
            entry: dict[str, Any] = {"view": view, "map": map_raw, "verb": verb, "kwargs": kw}
            if cs_err:
                entry["cs_err_msg"] = after.get("message")
            entries.append(entry)
            if cs_err:
                terminal = "cs_act_reject"
                break
            view = after
    except Exception:
        if not watchdog.tripped:
            raise
        # Wedged C# turn: the watchdog closed the backend, so the blocked read
        # failed (EOF). Keep the prefix recorded so far; replay diffs Rust
        # against it and reports `incomplete:record_timeout` if nothing diverged.
        terminal = "record_timeout"
    finally:
        watchdog.cancel()
        try:
            g.close()
        except Exception:
            pass
    return {
        "v": _RUN_TRACE_FORMAT,
        "seed": seed,
        "character": character,
        "ascension": ascension,
        "max_steps": max_steps,
        "terminal": terminal,
        "entries": entries,
    }


def _replay_against_rust(rec: dict[str, Any], stop_on_diverge: bool = True) -> RunResult:
    """Replay a recorded C# trajectory against a live Rust backend, producing the
    SAME ``RunResult`` ``run_one`` would (C# served from ``rec`` instead of a
    live subprocess)."""
    seed = rec["seed"]
    character = rec["character"]
    ascension = rec["ascension"]
    entries = rec["entries"]
    RS = make_backend("rust", target_convention="csharp")
    r = RS()
    res = RunResult(seed=seed, steps=0, terminal="error")
    try:
        rs_view = r.start(
            character=character,
            seed=seed,
            ascension=ascension,
            interactive_potion_swap=potion_swap_enabled(),
        )
        for step, e in enumerate(entries):
            res.steps = step
            cs_view = e["view"]
            cs_dec = cs_view.get("decision") or cs_view.get("type")

            cs_snap, rs_snap = Snap.of(cs_view, "cs"), Snap.of(rs_view, "rs")
            # Rust now SURFACES a played card's OnPlay select so an MCTS agent
            # can pick (combat_loop `play_card` defers the tail via
            # `pending_play`), and the standard replay drives it the same way C#
            # does. A few selects Rust still resolves ATOMICALLY: Echo/
            # OneTwoPunch replay siblings (the choice opens mid-`apply()` and
            # cannot surface) and a discard-N whose hand exactly fits (one legal
            # outcome). Those appear in the recorded C# trajectory as a separate
            # in-combat card_select step while Rust is already back in combat —
            # skip the recorded select entry (advance the C# cursor without
            # applying to Rust). The next recorded entry is C#'s post-select
            # combat state, which Rust's atomic first-legal resolution matches;
            # a genuinely divergent pick still surfaces at that state diff.
            # Mirrors the combat `parity_audit` harness, which atomizes the
            # select on both sides. Gated on rs=combat so a real shared
            # card_select (Rust ALSO paused) is replayed normally.
            if cs_snap.phase == "card_select" and rs_snap.phase == "combat":
                continue
            # MIRROR direction: RUST surfaces an OnPlay / turn-start select that
            # C# resolved ATOMICALLY, so the RECORDED C# entry is already the
            # post-select state while Rust is paused at ``card_select``. Finalize
            # Rust's select ALONE (``skip_select`` → ``combat_resolve_choice
            # {option: None}``) so it re-aligns, then diff against the SAME
            # recorded entry — a divergent atomic pick still surfaces in that
            # diff. Gated on ``cs != card_select`` (a real shared select is
            # replayed normally) and bounded against a select that never clears.
            # Oracle-artifact "run-combat-cardselect".
            sel_guard = 0
            while (
                rs_snap.phase == "card_select" and cs_snap.phase != "card_select" and sel_guard < 8
            ):
                nxt = rust_finalize_atomic_select(r, rs_view)
                if isinstance(nxt, dict) and (nxt.get("error") or nxt.get("type") == "error"):
                    break
                rs_view = nxt
                rs_snap = Snap.of(rs_view, "rs")
                sel_guard += 1
            # Event-granularity skip: the C# HEADLESS auto-skips Custom-layout
            # events (FakeMerchant → empty options → MapRoom), so the recorded C#
            # entry is on the ``map`` while Rust surfaces the FakeMerchant
            # ``event``. Drive Rust ALONE through its no-op ``leave`` option
            # (consumes no shared run RNG — private ``FAKE_MERCHANT`` stream) so it
            # re-aligns to the map, then diff against the recorded map entry. A
            # real room-behind desync still surfaces there. Bead sts2-rs-1hiht.
            if rs_snap.phase == "event" and cs_snap.phase == "map":
                leave = next(
                    (
                        o
                        for o in (rs_view.get("options") or [])
                        if o.get("option_id") == "leave" and not o.get("is_locked")
                    ),
                    None,
                )
                if leave is not None:
                    nxt = r.act("choose_option", option_index=leave["index"])
                    if not (
                        isinstance(nxt, dict) and (nxt.get("error") or nxt.get("type") == "error")
                    ):
                        rs_view = nxt
                        rs_snap = Snap.of(rs_view, "rs")
            fields = diff_snaps(cs_snap, rs_snap)
            if cs_snap.phase == "map":
                cs_map = _norm_map_graph(e["map"]) if e.get("map") else None
                fields += diff_maps(cs_map, map_signature(r))
            if fields:
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    fields,
                    "map" if any(f.startswith("map") for f in fields) else "state",
                )
                res.terminal = "diverged"
                if stop_on_diverge:
                    return res

            if cs_dec == "game_over":
                res.cs_won = bool(cs_view.get("victory") or cs_view.get("won"))
                res.rs_won = bool(
                    rs_view.get("victory")
                    or rs_view.get("won")
                    or _phase_of(rs_view, "rs") == "game_over"
                )
                res.terminal = (
                    "game_over_match" if (_phase_of(rs_view, "rs") == "game_over") else "diverged"
                )
                if _phase_of(rs_view, "rs") != "game_over":
                    res.divergence = Divergence(
                        step, "game_over", [f"cs=game_over rs={rs_snap.phase}"], "state"
                    )
                return res

            verb = e["verb"]
            if verb is None:  # recorded terminal: C# had no enumerable choice
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    [f"no choices enumerable for cs decision {cs_dec!r}"],
                    "harness",
                )
                res.terminal = "error"
                return res

            rs_view = r.act(verb, **e["kwargs"])
            cs_err = "cs_err_msg" in e
            # Rust signals a rejected action with ``{"type": "error", "message":
            # ...}`` (no ``error`` key) — read both so the reported message is
            # never spuriously ``None``.
            rs_err_msg = None
            if isinstance(rs_view, dict) and (
                rs_view.get("error") or rs_view.get("type") == "error"
            ):
                rs_err_msg = rs_view.get("error") or rs_view.get("message")
            rs_err = rs_err_msg is not None
            # An act-rejection is a divergence only when the engines DISAGREE
            # (one rejects, the other accepts). When BOTH reject they agree the
            # random-policy move is illegal — e.g. buying an item you can't
            # afford, which the choice enumerator offers regardless of price — so
            # the C# recording ended here (cs_act_reject) and Rust rejecting too
            # is parity, not a divergence.
            if cs_err and rs_err:
                res.terminal = rec.get("terminal", "cs_act_reject")
                return res
            if cs_err or rs_err:
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    [
                        f"act({verb},{e['kwargs']}) -> "
                        f"cs_err={e.get('cs_err_msg') if cs_err else None} "
                        f"rs_err={rs_err_msg}"
                    ],
                    "act_reject",
                )
                res.terminal = "diverged"
                if stop_on_diverge:
                    return res
        res.terminal = rec.get("terminal", "max_steps")
        return res
    except Exception as exc:
        import traceback

        res.divergence = Divergence(res.steps, "?", [f"{type(exc).__name__}: {exc}"], "harness")
        res.terminal = "error"
        res.trace.append(traceback.format_exc())
        return res
    finally:
        try:
            r.close()
        except Exception:
            pass


class _CacheMiss(Exception):
    """A required trajectory is absent from the cache in replay-only mode (so the
    .NET subprocess would otherwise be spawned — refused on a no-game machine)."""


class _RunTraceCache:
    """Filesystem cache of recorded C# run trajectories, namespaced by the oracle
    version + harness hash (mirrors ``parity_audit._TraceCache``)."""

    __slots__ = ("_dir", "namespace")

    def __init__(self, root: str, namespace: str):
        self.namespace = namespace
        self._dir = os.path.join(root, namespace)
        os.makedirs(self._dir, exist_ok=True)

    def _key(self, seed: str, character: str, ascension: int, max_steps: int) -> str:
        key: dict[str, Any] = {
            "seed": seed,
            "character": character,
            "asc": ascension,
            "steps": max_steps,
        }
        if not potion_swap_enabled():
            # The drop-mode trajectory differs from the swap-mode one (GH #1069).
            key["potion_swap"] = False
        payload = json.dumps(
            key,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_or_record(
        self,
        seed: str,
        character: str,
        ascension: int,
        max_steps: int,
        replay_only: bool = False,
    ) -> tuple[dict[str, Any], str]:
        path = os.path.join(self._dir, self._key(seed, character, ascension, max_steps) + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("v") == _RUN_TRACE_FORMAT:
                return rec, "hit"
        except (OSError, ValueError):
            pass
        if replay_only:
            # Fail closed: a no-game machine must never fall through to the .NET
            # recording path. The coordinator bakes the bundle (pack.sh).
            raise _CacheMiss(
                f"{character}/{seed} (asc{ascension}, steps{max_steps}) not in cache "
                f"{self.namespace}; re-bake with data/run-parity-cache/pack.sh"
            )
        rec = _record_cs_trajectory(seed, character, ascension, max_steps)
        # Never cache a watchdog-abandoned partial: persisting a truncated
        # trajectory would serve `incomplete:record_timeout` forever on re-runs.
        # Return it for THIS sweep (it buckets distinctly) but leave the slot
        # empty so a later re-bake (with a healthy oracle) records it in full.
        if rec.get("terminal") != "record_timeout":
            self._write_atomic(path, rec)
        return rec, "miss"

    @staticmethod
    def _write_atomic(path: str, rec: dict[str, Any]) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, separators=(",", ":"))
            pathlib.Path(tmp).replace(path)
        except OSError:
            try:
                pathlib.Path(tmp).unlink()
            except OSError:
                pass


#: Process-global cache, installed in the parent and every pool worker via the
#: ``ProcessPoolExecutor`` initializer (so it survives any start method).
_RUN_CACHE: _RunTraceCache | None = None
#: Replay-only: fail closed on a cache miss instead of spawning C# (so a 4-core /
#: no-game-DLL agent can rely purely on a pre-baked bundle).
_RUN_REPLAY_ONLY: bool = False


def _init_run_worker(
    cache_dir: str | None, namespace: str | None, replay_only: bool = False
) -> None:
    global _RUN_CACHE, _RUN_REPLAY_ONLY
    _RUN_CACHE = _RunTraceCache(cache_dir, namespace) if (cache_dir and namespace) else None
    _RUN_REPLAY_ONLY = replay_only


def run_one_cached(
    seed: str, character: str, ascension: int, max_steps: int, stop_on_diverge: bool = True
) -> RunResult:
    """Lockstep one run, serving the C# half from the trace cache when enabled.

    With no cache (``_RUN_CACHE is None``) this falls back to the live
    :func:`run_one` (fresh C# subprocess), behaviourally identical. With a cache,
    the C# trajectory is loaded (or recorded on miss) and Rust is replayed
    against it; the result carries a transient ``_cache`` status the sweep
    tallies. In replay-only mode a miss raises :class:`_CacheMiss` (never spawns
    C#)."""
    cache = _RUN_CACHE
    if cache is None:
        if _RUN_REPLAY_ONLY:
            raise _CacheMiss("replay-only requires a cache, but none is configured")
        return run_one(seed, character, ascension, max_steps, stop_on_diverge)
    rec, status = cache.get_or_record(
        seed, character, ascension, max_steps, replay_only=_RUN_REPLAY_ONLY
    )
    res = _replay_against_rust(rec, stop_on_diverge)
    res.trace.append(f"_cache={status}")
    return res


def run_signature(res: RunResult) -> str:
    """Stable bucket key for a sweep run: clean runs bucket by terminal, diverged
    runs by ``phase/kind/<normalized-first-field>`` (seed-specific numbers
    stripped) so the SAME root cause groups across seeds into one worklist line.
    """
    if res.divergence is None:
        # An incomplete (wedged-C#) record is NOT clean — surface it distinctly so
        # the bake/worklist shows which seeds the bundle couldn't fully record.
        if res.terminal == "record_timeout":
            return "incomplete:record_timeout"
        return f"clean:{res.terminal}"
    d = res.divergence
    first = d.fields[0] if d.fields else "?"
    if first.startswith("act("):
        m = re.match(r"act\((\w+)", first)
        norm = f"act:{m.group(1) if m else '?'}"
    elif first.startswith("phase:"):
        # A phase-label mismatch IS the signature — keep the cs/rs labels.
        norm = first.replace(" ", "").replace("'", "")
    else:
        name = first.split(":", 1)[0]
        norm = re.sub(r"\[.*?\]", "", name)  # drop map-node coords like [(1,2)]
    return f"{d.phase}/{d.kind}/{norm}"


def _result_row(res: RunResult, character: str) -> dict[str, Any]:
    """Flatten a ``RunResult`` to a picklable dict for the pool + report."""
    status = next((t.split("=", 1)[1] for t in res.trace if t.startswith("_cache=")), None)
    return {
        "seed": res.seed,
        "character": character,
        "steps": res.steps,
        "terminal": res.terminal,
        "cs_won": res.cs_won,
        "rs_won": res.rs_won,
        "signature": run_signature(res),
        "fields": list(res.divergence.fields) if res.divergence else [],
        "kind": res.divergence.kind if res.divergence else None,
        "_cache": status,
    }


def _sweep_one(item: tuple[str, str, int, int]) -> dict[str, Any]:
    seed, character, ascension, max_steps = item
    try:
        res = run_one_cached(seed, character, ascension, max_steps, stop_on_diverge=True)
    except _CacheMiss as exc:
        # Replay-only miss: report which (seed, char) is absent, don't crash the
        # whole sweep — the worklist line says exactly what the bundle is missing.
        return {
            "seed": seed,
            "character": character,
            "ascension": ascension,
            "steps": 0,
            "terminal": "cache_miss",
            "cs_won": None,
            "rs_won": None,
            "signature": "MISS:not_in_cache",
            "fields": [str(exc)],
            "kind": "cache_miss",
            "_cache": "miss",
        }
    except Exception as exc:
        # A single seed's oracle crash/EOF/hang (the C# headless wedging on a
        # swallowed-NRE turn, then dying or being watchdog-killed) must NOT tear
        # down the whole pool: ``pool.map`` re-raises the FIRST worker exception,
        # which would discard every other seed's already-computed result. Isolate
        # it as an error row so the census still reports all seeds that completed;
        # the failing seed surfaces as its own ``ERROR:<type>`` worklist bucket.
        msg = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        return {
            "seed": seed,
            "character": character,
            "ascension": ascension,
            "steps": 0,
            "terminal": "error",
            "cs_won": None,
            "rs_won": None,
            "signature": f"ERROR:{type(exc).__name__}",
            "fields": [msg],
            "kind": "harness_error",
            "_cache": "error",
        }
    row = _result_row(res, character)
    row["ascension"] = ascension
    return row


def sweep(
    base: str,
    characters: list[str],
    ascensions: list[int],
    seeds: int,
    steps: int,
    jobs: int,
    out: str,
    use_cache: bool,
    cache_dir: str,
    replay_only: bool = False,
) -> int:
    """M6 productionized sweep: lockstep ``seeds × characters`` runs in parallel,
    bucketed by divergence signature, with the wheel + oracle freshness gates.

    Returns the number of clean runs. Writes ``buckets.json`` / ``examples.json``
    / ``rows.json`` under ``out/<timestamp>`` and prints a ranked worklist — the
    per-character triage surface an agent reads to pick the next divergence.

    ``replay_only`` (for 4-core / no-game-DLL agents): serve EVERY run from the
    pre-baked cache bundle and fail closed on a miss — the C# subprocess is never
    spawned, and the C#-oracle freshness gate is skipped (there is no built dll
    to check). Requires the cache to be enabled + warmed (``unpack.sh``).
    """
    # Lazy: importing parity_audit pulls in sts2_rs + the combat harness, which
    # we don't want to pay for `run_parity --seed … ` lockstep / `--help`.
    from parity_audit import assert_oracle_fresh, assert_wheel_fresh, cs_version_namespace

    # Bound a wedged C# oracle. A specific seed can hang the headless on a stuck
    # turn forever (observed 60+ min at 0% CPU, then killing it tore down the
    # pool). The per-step record watchdog (`_record_cs_trajectory`,
    # `_StepWatchdog`) closes the backend and abandons such a step as
    # `record_timeout` instead. Default it ON for sweeps so one pathological
    # seed can't stall the whole run; override/disable via the same env var. Set
    # before the pool forks so workers inherit it.
    os.environ.setdefault("STS2_RUNPARITY_RECORD_TIMEOUT", "90")

    namespace: str | None = None
    cdir: str | None = None
    if use_cache and not os.environ.get("STS2_RUNPARITY_NOCACHE"):
        cs_ns = cs_version_namespace()
        if cs_ns:
            namespace = f"{cs_ns}-{_harness_hash()}"
            cdir = cache_dir
            print(f"[cache] enabled — C# {namespace} @ {os.path.join(cache_dir, namespace)}")
        else:
            print("[cache] DISABLED — could not locate C# oracle DLLs to namespace; running live.")
    if replay_only and not (cdir and namespace):
        sys.exit(
            "--replay-only requires a usable cache, but none could be located "
            "(no C# oracle DLLs to namespace, or --no-cache set). Warm the bundle "
            "first: data/run-parity-cache/unpack.sh"
        )
    _init_run_worker(cdir, namespace, replay_only)

    # Never sweep against a stale engine/oracle — a pre-edit wheel makes a real
    # fix look like a no-op, a stale dll serves an old oracle under a fresh
    # namespace. Gates run ONCE in the parent before any worker forks. In
    # replay-only mode the C# subprocess is never spawned, so skip the oracle
    # gate (a no-game machine has no built dll to check) — but still gate the
    # Rust wheel, which IS running.
    assert_wheel_fresh()
    if not replay_only:
        assert_oracle_fresh()

    ascensions = sorted(set(ascensions))
    trials: list[tuple[str, str, int, int]] = []
    for asc in ascensions:
        for character in characters:
            for i in range(seeds):
                seed = base if seeds == 1 else f"{base}{i}"
                trials.append((seed, character, asc, steps))

    jobs = max(1, min(jobs if jobs > 0 else (os.cpu_count() or 1), len(trials)))
    multi_char = len(characters) > 1
    multi_asc = len(ascensions) > 1
    asc_label = ",".join(f"A{a}" for a in ascensions)
    print(
        f"[sweep] {len(trials)} runs ({seeds} seeds × {len(characters)} char × "
        f"{len(ascensions)} asc [{asc_label}]), "
        f"{jobs} worker(s), steps={steps}{' [replay-only]' if replay_only else ''}"
    )

    rows: list[dict[str, Any]] = []
    if jobs == 1:
        _init_run_worker(cdir, namespace, replay_only)
        for t in trials:
            rows.append(_sweep_one(t))
    else:
        with _cf.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_run_worker,
            initargs=(cdir, namespace, replay_only),
        ) as pool:
            rows.extend(pool.map(_sweep_one, trials, chunksize=1))

    # Bucket by (character|)signature; keep one example row per bucket.
    from collections import Counter

    buckets: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    cache_tally: Counter[str] = Counter()
    n_clean = 0
    for r in rows:
        status = r.pop("_cache", None)
        if status:
            cache_tally[status] += 1
        sig = r["signature"]
        prefix = ""
        if multi_asc:
            prefix += f"A{r.get('ascension', 0)}|"
        if multi_char:
            prefix += f"{r['character']}|"
        key = f"{prefix}{sig}"
        buckets[key] += 1
        if sig.startswith("clean:"):
            n_clean += 1
        elif key not in examples:
            examples[key] = r

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out, ts)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "buckets.json"), "w", encoding="utf-8") as f:
        json.dump(dict(buckets.most_common()), f, indent=2)
    with open(os.path.join(out_dir, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, default=str)
    with open(os.path.join(out_dir, "rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\n=== {out_dir} ===")
    print(f"{n_clean}/{len(trials)} clean\n")
    for key, n in buckets.most_common():
        ex = examples.get(key)
        tail = ""
        if ex and ex.get("fields"):
            tail = f"  e.g. [{ex['seed']}] @step{ex['steps']}: {ex['fields'][0]}"
        print(f"{n:5d}  {key}{tail}")
    if namespace:
        hits = cache_tally["hit"]
        misses = cache_tally["miss"]
        if replay_only:
            absent = sum(1 for r in rows if r.get("signature") == "MISS:not_in_cache")
            print(
                f"\n[cache] replay-only — served {hits}/{len(trials)} from bundle "
                f"(C# {namespace}); {absent} absent (re-bake with "
                f"data/run-parity-cache/pack.sh)",
                file=sys.stderr,
            )
        else:
            print(
                f"\n[cache] served {hits}/{hits + misses} from cache (C# {namespace}); "
                f"recorded {misses}",
                file=sys.stderr,
            )
    return n_clean


def main() -> int:
    ap = argparse.ArgumentParser(description="Whole-run C#-vs-Rust parity driver")
    ap.add_argument("--seed", default="PARITY1")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument(
        "--ascensions",
        default=None,
        help="--sweep only: comma-separated ascension levels to cross with "
        "seeds × characters (e.g. '0,2,5,8,10'). Defaults to the single "
        "--ascension value. The per-trajectory cache key includes ascension, "
        "so multiple levels coexist in one bundle namespace.",
    )
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=1, help="run N seeds (seed, seed+1, ...)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--no-potion-swap",
        action="store_true",
        help="run both engines with the historical full-belt potion drop instead "
        "of the real game's potion_swap decision (GH #1069)",
    )
    ap.add_argument("--no-stop", action="store_true", help="keep going past first divergence")
    ap.add_argument(
        "--rng-parity",
        action="store_true",
        help="run the representation-robust act-1 boss RNG-parity check instead of lockstep",
    )
    ap.add_argument(
        "--act-parity",
        action="store_true",
        help="seed-driven ACT-VARIANT parity: which Act 1 (Overgrowth vs Underdocks) "
        "does each seed roll, C# `act_selection` stream vs the Rust port?",
    )
    ap.add_argument(
        "--underdocks-parity",
        action="store_true",
        help="act-1 CONTENT parity with BOTH engines forced onto Underdocks. "
        "The regression test for GH #544 (the Underdocks act-1 port).",
    )
    ap.add_argument(
        "--counters",
        action="store_true",
        help="M0 diagnostic: dump run-RNG stream cursors from both engines after start",
    )
    ap.add_argument(
        "--encounter-parity",
        action="store_true",
        help="M3: compare per-act pre-generated encounter lists (normal/elite/boss) C# vs Rust",
    )
    ap.add_argument(
        "--map-parity",
        action="store_true",
        help="M2: compare the act-1 map node-type grid + edges + boss position C# vs Rust",
    )
    ap.add_argument(
        "--reward-parity",
        action="store_true",
        help="M4: compare a fresh-stream regular card reward (ids + upgraded) C# vs Rust",
    )
    ap.add_argument(
        "--room-type",
        default="Monster",
        choices=["Monster", "Elite", "Boss"],
        help="card-reward context for --reward-parity (default Monster=regular)",
    )
    ap.add_argument(
        "--relic-reward-parity",
        action="store_true",
        help="M4: compare the fresh-stream relic reward(s) (ids) C# vs Rust",
    )
    ap.add_argument(
        "--relic-count",
        type=int,
        default=1,
        help="number of relic rewards to pull for --relic-reward-parity (default 1)",
    )
    ap.add_argument(
        "--shop-parity",
        action="store_true",
        help="M4: compare a fresh-stream normal-merchant inventory C# vs Rust",
    )
    ap.add_argument(
        "--roster-parity",
        action="store_true",
        help="M3b: compare each generated encounter's spawned roster "
        "(monster kinds + niche@0 max HP) C# vs Rust",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="M6: productionized cached lockstep sweep (seeds × characters) in "
        "parallel, with wheel/oracle freshness gates, bucketed by divergence "
        "signature into a ranked per-character worklist",
    )
    ap.add_argument(
        "--all-characters",
        action="store_true",
        help="--sweep over all 5 characters (buckets char-tagged); default is "
        "just --character. Use this for a full-roster triage overview.",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="--sweep parallel worker processes (0 = os.cpu_count()). Each worker "
        "may spawn a C# subprocess on a cache miss, so peak RAM scales with this.",
    )
    ap.add_argument("--out", default="/tmp/run_parity", help="--sweep report output root")
    ap.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--sweep: cache the C# run trajectories (ON by default). The first "
        "sweep records each (seed, char) C# trajectory; later sweeps replay "
        "Rust-only against it, skipping the .NET subprocess. Namespaced by the "
        "C# oracle hash + this harness's hash, so any change auto-invalidates. "
        "--no-cache (or STS2_RUNPARITY_NOCACHE=1) forces a fully-live sweep.",
    )
    ap.add_argument(
        "--cache-dir",
        default=os.environ.get("STS2_RUNPARITY_CACHE_DIR")
        or os.path.join(os.path.expanduser("~"), ".cache", "sts2_run_parity"),
        help="directory for the C# run-trajectory cache (default: "
        "$STS2_RUNPARITY_CACHE_DIR or ~/.cache/sts2_run_parity)",
    )
    ap.add_argument(
        "--replay-only",
        action="store_true",
        default=bool(os.environ.get("STS2_RUNPARITY_REPLAY_ONLY")),
        help="--sweep: serve EVERY run from the pre-baked cache bundle and FAIL "
        "CLOSED on a miss — the C# subprocess is never spawned (for 4-core / "
        "no-game-DLL agents). Warm the bundle first with "
        "data/run-parity-cache/unpack.sh. (env: STS2_RUNPARITY_REPLAY_ONLY)",
    )
    add_game_version_arg(ap)
    args = ap.parse_args()
    if args.no_potion_swap:
        os.environ["RUN_PARITY_POTION_SWAP"] = "0"
    apply_game_version_arg(args)

    if args.sweep:
        chars = list(CHARACTERS) if args.all_characters else [args.character]
        ascensions = (
            [int(a) for a in args.ascensions.split(",") if a.strip() != ""]
            if args.ascensions
            else [args.ascension]
        )
        sweep(
            args.seed,
            chars,
            ascensions,
            max(1, args.seeds),
            args.steps,
            args.jobs,
            args.out,
            args.cache,
            args.cache_dir,
            args.replay_only,
        )
        return 0

    if args.roster_parity:
        roster_parity(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    if args.shop_parity:
        shop_parity(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    if args.reward_parity:
        reward_parity(args.seed, args.character, args.ascension, max(1, args.seeds), args.room_type)
        return 0

    if args.relic_reward_parity:
        relic_reward_parity(
            args.seed, args.character, args.ascension, max(1, args.seeds), max(1, args.relic_count)
        )
        return 0

    if args.encounter_parity:
        encounter_parity(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    if args.map_parity:
        map_parity(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    if args.counters:
        rng_counters(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    if args.act_parity:
        n = max(1, args.seeds)
        return 0 if act_parity(args.seed, args.character, args.ascension, n) == n else 1

    if args.underdocks_parity:
        n = max(1, args.seeds)
        return 0 if underdocks_parity(args.seed, args.character, args.ascension, n) == n else 1

    if args.rng_parity:
        rng_parity(args.seed, args.character, args.ascension, max(1, args.seeds))
        return 0

    base = args.seed
    n_clean = 0
    for i in range(args.seeds):
        seed = base if args.seeds == 1 else f"{base}{i}"
        res = run_one(
            seed, args.character, args.ascension, args.steps, not args.no_stop, args.verbose
        )
        if args.verbose:
            for line in res.trace:
                print(line)
        tag = "clean" if res.divergence is None else "DIVERGE"
        if res.divergence is None:
            n_clean += 1
        print(
            f"[{seed}] steps={res.steps} terminal={res.terminal} cs_won={res.cs_won} rs_won={res.rs_won} :: {tag}"
        )
        if res.divergence:
            d = res.divergence
            print(f"    @step {d.step} phase={d.phase} kind={d.kind}")
            for f in d.fields:
                print(f"      - {f}")
    print(f"\n{n_clean}/{args.seeds} clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
