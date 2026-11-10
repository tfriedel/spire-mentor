#!/usr/bin/env python3
"""Deep act-2/3 whole-run state parity (invincible lockstep).

The default C#-driven whole-run lockstep (``run_parity.py --sweep``) drives a
*random* policy, which dies early in Act 1 — so the carried run state of Acts 2
and 3 (deck/relic/gold/HP evolution, later-act maps, elite/boss combats, shops)
is essentially never sampled. This driver closes that gap (coverage-matrix gap
``(c)``) by pinning the player to ``INVINCIBLE_HP`` on **both** engines whenever
the run is *out of combat*, so the next combat seeds from full invincible HP and
the random walk survives deep into the later acts.

Why this stays a faithful parity check:

* ``set_player`` writes only scalar HP/gold fields — it consumes **no RNG** and
  fires **no hooks** on either engine, so the run's RNG stream is identical to a
  non-invincible run. The trajectory only diverges by *surviving longer*, and it
  does so symmetrically on both engines.
* HP is pinned **only out of combat** (after the per-step diff, before the next
  action). Combat therefore *seeds* from ``INVINCIBLE_HP`` but is never re-pinned
  mid-fight, so in-combat HP decay remains a genuine C#↔Rust parity check (and we
  sidestep the Rust mirror-creature lazy-recreation that an in-combat pin hits).
* ``max_hp`` is pinned too (so heals can't clamp the pinned current HP back down).
  This masks *max-HP-gain* parity in this mode — that surface is covered by the
  normal (non-invincible) sweeps; here the prize is deep act-2/3 *carried state*.

The policy is the same seeded random picker as ``run_parity`` (off the C# oracle's
enumerated choices), with one addition: when an enumerated move is rejected by
**both** engines (the random policy offered an illegal move, e.g. an unaffordable
buy), the choice is dropped and another is picked rather than ending the run — so
one benign reject doesn't truncate a deep run.

Reuses ``run_parity``'s proven ``Snap`` / ``diff_snaps`` / ``diff_maps`` /
``enumerate_choices`` / ``run_signature`` primitives verbatim; ``run_parity.py``
itself is left untouched (this protects its established lockstep baseline).

Usage::

    # single verbose run (triage a divergence)
    python python/act23_state_parity.py --character Ironclad --seed 1000 --verbose

    # sweep
    python python/act23_state_parity.py --all-characters -n 20
    python python/act23_state_parity.py --character Defect -n 50 --max-steps 3000

Checkpoints (start at act 2 or 3 instead of replaying act 1 every trial)::

    # build: play each seed to the first act-2 map, save, stop
    python python/act23_state_parity.py --all-characters -n 20 \
        --checkpoint-dir ckpts --checkpoint-acts 2 --stop-at-checkpoint

    # branch: 8 different walks from every act-2 checkpoint; the branches that
    # reach act 3 save act-3 checkpoints (named after their parent)
    python python/act23_state_parity.py --from-checkpoints ckpts \
        --checkpoint-act 2 --branches 8 --checkpoint-dir ckpts --checkpoint-acts 3

    # deep act 3 only
    python python/act23_state_parity.py --from-checkpoints ckpts \
        --checkpoint-act 3 --branches 8

A checkpoint is the C# oracle's continue save plus the logged Rust calls (see
``write_checkpoint``); ``save_load_identity.py`` checks that a reload is
lossless. A checkpoint goes stale when the Rust engine changes (the replay is
checked against the recorded view and the run ends ``checkpoint_stale``):
rebuild it then.

The walk is the seeded random picker, except that it always plays
``SURVIVAL_CARDS`` and doesn't end a turn early under ``TIMER_KILL_POWERS``,
because The Insatiable's Sandpit kills whatever the HP (``--uniform-policy``
turns that off).
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from full_state_diff import diff_full, path_class, project_csharp, project_rust
from parity_audit import assert_wheel_fresh, cs_version_namespace
from parity_fuzz import _csharp_card_id
from parity_fuzz_scenario import Loadout, generate_loadout
from run_parity import (
    Divergence,
    RunResult,
    Snap,
    diff_maps,
    diff_snaps,
    enumerate_choices,
    map_signature,
    run_signature,
    rust_finalize_atomic_select,
)
from split_trace import explain, suspects
from sts2_client import add_game_version_arg, apply_game_version_arg
from workbench.backends.base import make_backend

# Mirrors ``sts2_core::full_run::INVINCIBLE_HP`` (the engine-internal invincible
# coverage constant). Both backends are pinned to this exact value so the
# lockstep HP diff matches.
INVINCIBLE_HP = 100_000

ALL_CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

# The Rust engine carries a combat choice's option index as ``u8``
# (``Action::ResolveChoice``, ``PendingChoice::picked``), so a pick at option
# 256 or past it is rejected (``do_combat_resolve_choice``'s ``u8::try_from`` →
# ``invalid_for_phase``). C# ``CardSelectCmd`` has no such cap. Only this walk's
# ``INVINCIBLE_HP`` fights reach it: a 150-round fight lets Anger fill the
# discard pile, and Headbutt then offers 445 cards. The walk picks among the
# first 256 options so the run keeps going; that limit is the harness's, not a
# parity signal.
MAX_SELECT_INDEX = 255


def _act_of(view: dict[str, Any] | None) -> int | None:
    """Best-effort current act index from either engine's state dict.

    The C# backend nests ``act`` under a per-decision ``context`` dict (only the
    ``map_select`` / ``game_over`` serializers also lift it to the top level),
    while Rust puts it top-level — so check ``context`` too or C#'s act reads
    ``None`` at every event/combat/card_reward step (which silently pinned
    ``max_act`` to the Rust side alone). Both engines number acts 1-based and
    identically, so the resolved value IS cross-engine comparable.
    """
    if not isinstance(view, dict):
        return None
    ctx = view.get("context") if isinstance(view.get("context"), dict) else None
    p = view.get("player") if isinstance(view.get("player"), dict) else None
    for src in (view, ctx, p):
        if isinstance(src, dict) and src.get("act") is not None:
            return int(src["act"])
    return None


def _floor_of(view: dict[str, Any] | None) -> int | None:
    """Best-effort serialized floor from either engine's state dict.

    NOTE — these are NOT cross-engine comparable in acts 2/3: C# serializes
    ``RunState.ActFloor`` (per-act, resets to 0 each act) while Rust serializes
    ``RunManager.total_floor()`` (cumulative across acts). So ``_floor_of`` is
    surfaced in the trace per-side for human triage ONLY; the divergence diff
    keys on ``act`` + ``current_coord`` (see ``diff_nav``), both of which use the
    same per-act ``(col, row)`` coordinate system the map-parity check already
    proves equal. Within act 1 the two floor schemes happen to coincide.
    """
    if not isinstance(view, dict):
        return None
    ctx = view.get("context") if isinstance(view.get("context"), dict) else None
    p = view.get("player") if isinstance(view.get("player"), dict) else None
    for src in (view, ctx, p):
        if isinstance(src, dict) and src.get("floor") is not None:
            return int(src["floor"])
    return None


def _is_architect_event(view: dict[str, Any] | None) -> bool:
    """Whether a C# ``event_choice`` view is TheArchitect's post-act-3 victory
    event (option text keys ``THE_ARCHITECT.…``, or its display name)."""
    if not isinstance(view, dict):
        return False
    name = str(view.get("event_name") or view.get("event_id") or "")
    if "architect" in name.lower():
        return True
    return any(
        str((o or {}).get("text_key") or "").startswith("THE_ARCHITECT")
        for o in (view.get("options") or [])
    )


def _current_coord(backend: Any) -> tuple[int, int] | None:
    """Current map-node ``(col, row)`` on either engine, or ``None``.

    This is the finer per-step nav signal issue #158 calls for: it pinpoints the
    *room* each engine sits in, so a "C# a room ahead" boundary desync surfaces
    at its TRUE step instead of cascading into downstream gold/deck/phase skews.
    Rust's ``full_map`` exposes a top-level ``current_coord {col,row}``; C# marks
    the current node in-row with ``current=True``. The ``(col, row)`` coordinate
    system is shared (the same one ``_norm_map_graph`` compares 25/25), so the
    two engines' coords ARE directly comparable. Read defensively — a map-less
    phase or a backend without a map returns ``None`` and the diff skips it.
    """
    try:
        m = backend.get_map()
    except Exception:
        return None
    if not isinstance(m, dict):
        return None
    cc = m.get("current_coord")
    if isinstance(cc, dict) and cc.get("col") is not None and cc.get("row") is not None:
        return (int(cc["col"]), int(cc["row"]))
    for row in m.get("rows") or []:
        for node in row or []:
            if isinstance(node, dict) and node.get("current"):
                col, r = node.get("col"), node.get("row")
                if col is not None and r is not None:
                    return (int(col), int(r))
    return None


@dataclass
class Nav:
    """Finer act/floor/current-node nav snapshot for one engine (issue #158)."""

    act: int | None
    floor: int | None
    coord: tuple[int, int] | None

    @classmethod
    def of(cls, view: dict[str, Any], backend: Any) -> Nav:
        return cls(act=_act_of(view), floor=_floor_of(view), coord=_current_coord(backend))


def diff_nav(cs: Nav, rs: Nav) -> list[str]:
    """Cross-engine nav divergences (``act`` + ``current_coord`` only).

    Called as a BLOCKING check only while both engines are on the ``map`` (see
    the gate at the call site) — there the current node is authoritative on
    both, so a mismatch is a genuine "standing on different nodes" desync.

    Raw ``floor`` is deliberately NOT compared (C# ActFloor per-act vs Rust
    cumulative total_floor — see ``_floor_of``). ``act`` and ``coord`` use the
    shared per-act coordinate system. Each side's coord is read only when that
    engine exposes a current node (``None`` otherwise); a one-sided ``None`` is
    skipped so a benign no-map phase doesn't false-flag.
    """
    out: list[str] = []
    if cs.act is not None and rs.act is not None and cs.act != rs.act:
        out.append(f"nav.act: cs={cs.act} rs={rs.act}")
    if cs.coord is not None and rs.coord is not None and cs.coord != rs.coord:
        out.append(f"nav.coord: cs={cs.coord} rs={rs.coord}")
    return out


# Max player-HP delta tolerated as a C# mid-combat-snapshot artifact (issue
# #158 step 3). The snapshot-timing gap is at most the HP swing of one
# unresolved damage/heal instance at the read boundary; a real combat-damage
# divergence compounds turn over turn (so it eventually exceeds this) or
# co-surfaces an ``enemies_hp``/``phase`` field (so it is never peeled here).
# Empirically the observed artifact deltas are single-/low-double-digit; this
# bound leaves wide margin for a self-damage card (Combust/Brutality/Offering)
# resolving across the snapshot while still catching a persistent real skew.
HP_ARTIFACT_TOLERANCE = 60

# Phases where a small player-HP-only delta is the C# mid-combat snapshot
# artifact rather than a real divergence. Combat itself, plus the combat→reward
# exit window (``card_reward`` / the in-combat ``card_select``) where the
# combat-end HP carries one step before the next OUT-OF-COMBAT pin re-syncs both
# engines to ``INVINCIBLE_HP``. Deliberately EXCLUDES ``event`` / ``rest`` — the
# surfaces where a *real* HP-changing effect (event damage/heal, campfire rest)
# lives and must still surface as a genuine divergence.
_HP_ARTIFACT_PHASES = frozenset({"combat", "card_reward", "card_select"})


def _is_combat_hp_artifact(cs: Snap, rs: Snap, fields: list[str]) -> bool:
    """True when ``fields`` is *exactly* a small player-HP delta inside the
    combat / combat-exit window — the C# mid-combat-snapshot oracle-artifact
    class (issue #158 step 3).

    Strict on purpose so it never masks a real bug: it fires ONLY when every
    diverging field is the player ``hp`` line (a phase/deck/enemy/nav mismatch
    would add another field and disqualify it), the shared phase is one of
    ``_HP_ARTIFACT_PHASES``, and ``|Δhp| <= HP_ARTIFACT_TOLERANCE``. Because the
    run is pinned to ``INVINCIBLE_HP`` out of combat, a tolerated delta is wiped
    at the next pin and cannot perturb the trajectory or survival — so peeling it
    only un-masks the deeper structural signal it was hiding.
    """
    if cs.phase not in _HP_ARTIFACT_PHASES:
        return False
    if not fields or any(not f.startswith("hp:") for f in fields):
        return False
    if cs.hp is None or rs.hp is None:
        return False
    return abs(cs.hp - rs.hp) <= HP_ARTIFACT_TOLERANCE


def _is_err(view: Any) -> bool:
    return isinstance(view, dict) and (view.get("error") or view.get("type") == "error")


# Cards the walk plays whenever they are legal, because pinned HP can't save
# the run otherwise. The Insatiable (act-2 boss) applies Sandpit, which
# force-kills its target when it counts down to 0 (``SandpitPower.cs:121``,
# ``CreatureCmd.Kill(..., force: true)``) whatever its HP; only playing
# Frantic Escape pushes it back up (``FranticEscape.cs:43``). A uniform random
# policy rarely does, so before this every invincible walk died to it and
# act 3 was almost never reached. Both engines get the same action, so the
# lockstep stays a fair parity check.
SURVIVAL_CARDS = frozenset({"CARD.FRANTIC_ESCAPE"})

# Enemy powers that kill on a timer. While one is up, the walk doesn't end its
# turn while it can still play a card: random play ended turns with energy
# left, so the boss outlived the Sandpit timer.
TIMER_KILL_POWERS = frozenset({"SANDPIT_POWER"})


def _survival_choice(
    choices: list[tuple[str, dict[str, Any]]], cs_view: dict[str, Any]
) -> int | None:
    """Index into ``choices`` of a legal play of a ``SURVIVAL_CARDS`` card."""
    ids = {c.get("index"): c.get("id") for c in (cs_view.get("hand") or []) if isinstance(c, dict)}
    for i, (verb, kw) in enumerate(choices):
        if verb == "play_card" and ids.get(kw.get("card_index")) in SURVIVAL_CARDS:
            return i
    return None


# Claws (Tanx Ancient relic) opens a min-0/max-6 ``FromDeckForTransformation``
# select (``Claws.cs:22-29``). Rust surfaces no select: ``listener_claws``
# transforms the first 6 candidates in deck order, with C#'s candidate filter
# (not Quest, not Eternal). C# lists exactly those candidates in deck order, so
# Rust's pick is C# indices ``0..min(6, n)``. The generic first-legal pick
# (``select_cards "0"``) transforms ONE card and splits the deck one step
# later. Both picks are legal in the real game; this only makes the two
# engines pick the same cards.
CLAWS_PROMPT = "CLAWS.selectionScreenPrompt"
CLAWS_CARDS = 6


def claws_atomic_choice(cs_view: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The C# ``select_cards`` matching Rust's automatic Claws pick, or ``[]``."""
    if cs_view.get("prompt_key") != CLAWS_PROMPT:
        return []
    cards = cs_view.get("cards") or []
    n = min(CLAWS_CARDS, int(cs_view.get("max_select") or CLAWS_CARDS), len(cards))
    if n == 0:
        return []
    return [("select_cards", {"indices": ",".join(str(c["index"]) for c in cards[:n])})]


def _cs_alone_select(cs_phase: str, rs_phase: str) -> bool:
    """True when C# is at a ``card_select`` Rust has already resolved
    atomically (Rust is no longer selecting), so the driver advances C# alone
    through it.
    """
    return cs_phase == "card_select" and rs_phase != "card_select"


def _cs_select_still_pending(g: Any) -> bool:
    """True when the C# headless holds a pending card select its last view hid.

    The combat fallback in ``RunSimulator.DetectDecisionPoint`` ("combat phase
    settle") returns ``combat_play`` without re-checking
    ``_cardSelector.HasPending``, so a turn-start select (Toasty Mittens'
    ``AfterPlayerTurnStart`` exhaust pick, ``ToastyMittens.cs:31-43``) that
    opens during that wait is reported as ``combat_play``. Probe with an
    argument-less ``select_cards``: ``DoSelectCards`` answers "No pending card
    selection" when nothing is pending and "requires 'indices'" when something
    is, before touching any state.
    """
    probe = g.act("select_cards")
    if not _is_err(probe):
        return False
    msg = str(probe.get("message") or probe.get("error") or "")
    return "requires 'indices'" in msg


def _stale_cs_select_pick(rs_view: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The pick both engines take at a select C#'s view hid.

    The same option prefix ``rust_finalize_atomic_select`` takes: the first
    ``min_select`` options, or ``skip_select`` when the select is optional.
    """
    cards = rs_view.get("cards") or []
    min_sel = int(rs_view.get("min_select") or 0)
    if min_sel >= 1 and cards:
        idxs = [str(c["index"]) for c in cards[: min(min_sel, len(cards))]]
        return "select_cards", {"indices": ",".join(idxs)}
    return "skip_select", {}


def _within_select_index_cap(
    choices: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Drop ``select_cards`` choices that name an option past ``MAX_SELECT_INDEX``.

    A no-op for every select of 256 options or fewer, so it changes neither the
    choice list nor the walk's RNG draws on any other run.
    """

    def fits(verb: str, kw: dict[str, Any]) -> bool:
        if verb != "select_cards":
            return True
        idxs = [i for i in str(kw.get("indices", "")).split(",") if i.strip()]
        return all(int(i) <= MAX_SELECT_INDEX for i in idxs)

    return [(verb, kw) for verb, kw in choices if fits(verb, kw)]


def _timer_kill_active(cs_view: dict[str, Any]) -> bool:
    return any(
        (p or {}).get("id") in TIMER_KILL_POWERS
        for e in (cs_view.get("enemies") or [])
        for p in ((e or {}).get("powers") or [])
    )


def _survival_pool(
    choices: list[tuple[str, dict[str, Any]]], cs_view: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """``choices`` minus ``end_turn`` while a timer kill is up and a play is left."""
    if not _timer_kill_active(cs_view):
        return choices
    plays = [c for c in choices if c[0] != "end_turn"]
    return plays or choices


def _pin(g: Any, r: Any, hp: int) -> None:
    """Pin both engines' current+max HP to ``hp`` (mirror of C# ``set_player``)."""
    g.set_player(hp=hp, max_hp=hp)
    r.set_player(hp=hp, max_hp=hp)


def _relic_ids(view: dict[str, Any]) -> list[str]:
    """The Rust view's relic ids, bare (``BURNING_BLOOD``), in owned order."""
    relics = (view.get("player") or {}).get("relics") or []
    return [str(r["id"]).removeprefix("RELIC.") for r in relics]


def _graft_loadout(g: Any, r: Any, lo: Loadout, rs_view: dict[str, Any]) -> list[str]:
    """Install ``lo`` on both engines; return what stopped them matching it.

    Replaces the deck and potions, and adds to the relics already owned, so the
    starter and Neow relics stay. ``set_player`` fires no on-obtain hook on
    either engine and draws no RNG: anything the graft bypasses (a Potion
    Belt's extra slots, a Whetstone's upgrades) is bypassed on both sides. It
    does take the relics out of the grab bags, so they aren't offered again.

    C# skips an id it doesn't know, so an unusable pool entry would otherwise
    surface later as an engine divergence. It's checked here instead.
    """
    owned = _relic_ids(rs_view)
    relics = owned + [x for x in lo.relics if x not in owned]
    potions = list(lo.potions[: r.rm.max_potion_slots()])
    cs_reply = g.set_player(
        deck=[_csharp_card_id(c) for c in lo.deck],
        relics=relics,
        potions=potions,
        gold=lo.gold,
    )
    if _is_err(cs_reply):
        return [f"C# set_player failed: {cs_reply.get('message')!r}"]
    rs_reply = r.set_player(deck=list(lo.deck), relics=relics, potions=potions, gold=lo.gold)
    if _is_err(rs_reply):
        return [f"Rust set_player failed: {rs_reply.get('message')!r}"]
    cs, rs = Snap.of(cs_reply, "cs"), Snap.of(rs_reply, "rs")
    problems = [f for f in diff_snaps(cs, rs) if not f.startswith("phase")]
    for side, snap in (("cs", cs), ("rs", rs)):
        if snap.deck_size != len(lo.deck):
            problems.append(f"{side} deck_size {snap.deck_size} != loadout {len(lo.deck)}")
        if len(snap.relics) != len(relics):
            problems.append(f"{side} relics {len(snap.relics)} != loadout {len(relics)}")
    return problems


@dataclass
class InvRun:
    """One invincible-lockstep run: a ``run_parity.RunResult`` plus the extra
    act-depth metadata this driver tracks."""

    result: RunResult
    character: str
    max_act: int = 1
    repicks: int = 0
    atomic_select_skips: int = 0
    # Shared selects the C# view hid (``_cs_select_still_pending``).
    stale_cs_selects: int = 0
    # Steps whose ``select_cards`` choices were trimmed to ``MAX_SELECT_INDEX``.
    select_index_caps: int = 0
    event_skips: int = 0
    hp_artifacts: int = 0
    max_hp_artifact: int = 0
    # ``--full-state``: the first combat step where the full-state diff
    # (``full_state_diff``) found any field mismatch, and those fields.
    full_split: FullSplit | None = None
    # Acts whose checkpoint this run wrote (``checkpoint_dir``).
    checkpoints: list[int] = field(default_factory=list)
    # ``random_loadout``: what the run was grafted with.
    loadout: Loadout | None = None
    # A one-sided C# reject's ``stack_trace`` (GH #1147): printed by the sweep
    # too, so a sweep-only harness exception names its throwing frame.
    cs_stack: str | None = None


@dataclass
class FullSplit:
    """The earliest full-state combat divergence of a run."""

    step: int
    act: int
    paths: list[str]  # ``path: cs=.. rs=..`` lines, first few
    # ``--trace-split``: each diverging field's C# change history over the
    # split step (``split_trace.explain``) and the sources behind it.
    trace: list[str] = field(default_factory=list)
    suspects: list[str] = field(default_factory=list)
    # The C# world threw during the split step (``Game.world_errors`` grew):
    # its latest exception line. A swallowed throw aborts the rest of the
    # command's await chain, a card play left on the Play pile for one (loadout
    # Silent 1037: ``Hook.AfterBlockBroken`` on a creature Outbreak had just
    # removed). That is the game's own bug, which Rust does not reproduce.
    cs_threw: str | None = None

    @property
    def classes(self) -> str:
        """The diverging fields with list indices stripped, for bucketing."""
        seen: list[str] = []
        for line in self.paths:
            c = path_class(line.split(":", 1)[0])
            if c not in seen:
                seen.append(c)
        tag = "cs_threw|" if self.cs_threw else ""
        return tag + ",".join(seen[:3])

    def __str__(self) -> str:
        sus = f" src={self.suspects}" if self.suspects else ""
        return f"act{self.act} combat@{self.step}: {self.paths[:3]}{sus}"


# Lines of a full-state split kept per run (the first is what triage needs).
FULL_SPLIT_MAX_PATHS = 8


def full_state_divs(g: Any, r: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The C# ``get_state`` and the full-state divergences for the current
    combat step (no divergences == match)."""
    cs_state = g.get_state()
    if _is_err(cs_state):
        return {}, []
    combat = r.combat_state()
    if combat is None:
        return cs_state, []
    return cs_state, diff_full(project_csharp(cs_state), project_rust(combat))


def _cs_world_errors(g: Any) -> dict[str, Any] | None:
    """``Game.world_errors`` of the C# backend, or ``None`` when it has none."""
    fn = getattr(g, "world_errors", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _split_lines(divs: list[dict[str, Any]]) -> list[str]:
    return [f"{d['path']}: cs={d['csharp']!r} rs={d['rust']!r}" for d in divs]


def full_state_split(g: Any, r: Any) -> list[str]:
    """Full-state diff lines for the current combat step (empty == match)."""
    return _split_lines(full_state_divs(g, r)[1])


# --------------------------------------------------------------------------- #
# Checkpoints: start runs at act 2/3 instead of replaying act 1 every trial
# --------------------------------------------------------------------------- #
# A checkpoint is one directory holding the C# oracle's own continue save
# (``cs.save``) and ``meta.json``: the run's identity, the exact Rust backend
# calls that led there, and the Rust view they produced. Resuming loads the
# save into a fresh C# process and replays the calls into a fresh Rust run.
# The Rust side has no loader for C# saves (and the run-lane state import
# doesn't restore RNG streams), but it runs a step in microseconds, so a
# replay costs well under a second. ``save_load_identity.py`` is the check
# that a C# reload continues exactly like the process that wrote it.
CHECKPOINT_FORMAT = 1


class CheckpointStale(RuntimeError):
    """The Rust replay no longer reproduces the checkpoint's Rust view."""


class _RustRecorder:
    """Pass-through over the Rust backend that logs every state-changing call.

    ``start``, ``act`` and ``set_player`` are the only calls the lockstep makes
    that move the Rust run (``rust_finalize_atomic_select`` goes through
    ``act``); everything else it reads is a query. Replaying the log on a fresh
    backend rebuilds the same run, because the engine is deterministic.
    """

    _MUTATORS = frozenset({"start", "act", "set_player"})

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[list[Any]] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in self._MUTATORS or not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append([name, list(args), kwargs])
            return attr(*args, **kwargs)

        return call


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


@functools.lru_cache(maxsize=1)
def _build_stamp() -> dict[str, str]:
    """Which oracle and engine wrote a checkpoint (recorded, checked on load)."""
    import sts2_rs

    return {"cs_namespace": cs_version_namespace(), "rs_src_hash": sts2_rs.src_hash()}  # ty: ignore[unresolved-attribute]


def checkpoint_name(character: str, seed: str, ascension: int, act: int, parent: str = "") -> str:
    """``Ironclad-1000-a0-act2``; a checkpoint written by a branch run carries
    its parent and branch, e.g. ``Ironclad-1000-a0-act2.b3-act3``."""
    if parent:
        return f"{parent}-act{act}"
    return f"{character}-{seed}-a{ascension}-act{act}"


def write_checkpoint(
    root: Path,
    *,
    character: str,
    seed: str,
    ascension: int,
    act: int,
    step: int,
    g: Any,
    rec: _RustRecorder,
    rs_view: dict[str, Any],
    parent: str = "",
) -> Path:
    """Save the current lockstep position as checkpoint ``root/<name>``."""
    d = root / checkpoint_name(character, seed, ascension, act, parent)
    d.mkdir(parents=True, exist_ok=True)
    saved = g.write_continue_save(str(d / "cs.save"))
    if not saved.get("success"):
        raise RuntimeError(f"C# checkpoint save failed: {saved}")
    meta = {
        "format": CHECKPOINT_FORMAT,
        "character": character,
        "seed": seed,
        "ascension": ascension,
        "act": act,
        "step": step,
        **_build_stamp(),
        "rust_calls": rec.calls,
        "rs_view": json.loads(_canon(rs_view)),
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def read_checkpoint_meta(d: Path) -> dict[str, Any]:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointStale(
            f"{d.name}: checkpoint format {meta.get('format')} != {CHECKPOINT_FORMAT}"
        )
    return meta


def list_checkpoints(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/meta.json"))


def resume_checkpoint(
    d: Path, g: Any, rec: _RustRecorder, meta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load checkpoint ``d`` into fresh backends; return ``(cs_view, rs_view)``.

    Raises :class:`CheckpointStale` when the replayed Rust view differs from
    the recorded one: the engine changed since the checkpoint was written, so
    the act-1 prefix it encodes is no longer what this engine plays.
    """
    cs_view = g.load_save(path=str(d / "cs.save"))
    if _is_err(cs_view):
        raise RuntimeError(f"{d.name}: C# load_save failed: {cs_view}")
    rs_view: Any = None
    for name, args, kwargs in meta["rust_calls"]:
        rs_view = getattr(rec, name)(*args, **kwargs)
    if _canon(rs_view) != _canon(meta["rs_view"]):
        raise CheckpointStale(
            f"{d.name}: the Rust replay no longer reproduces the checkpoint "
            "(engine changed since it was written) — rebuild it"
        )
    return cs_view, rs_view


def run_one_invincible(
    seed: str,
    character: str = "Ironclad",
    ascension: int = 0,
    max_steps: int = 2500,
    invincible: bool = True,
    verbose: bool = False,
    peel_hp_artifacts: bool = True,
    full_state: bool = False,
    full_state_blocking: bool = False,
    trace_split: bool = False,
    checkpoint_dir: Path | None = None,
    checkpoint_acts: frozenset[int] = frozenset(),
    stop_at_checkpoint: bool = False,
    from_checkpoint: Path | None = None,
    branch: int = 0,
    survival_policy: bool = True,
    random_loadout: bool = False,
) -> InvRun:
    """Drive a single invincible lockstep run; return an :class:`InvRun`.

    Mirrors ``run_parity.run_one`` (same Snap/diff/choice logic) with two
    additions: out-of-combat HP pinning and both-reject re-picking.

    ``full_state`` also runs the full-state combat diff (``full_state_diff``)
    at every step where both engines are in combat, and records the first
    mismatch on ``InvRun.full_split`` while the run carries on to its coarse
    verdict. ``full_state_blocking`` ends the run at that first mismatch
    instead (signature ``combat/full/<fields>``). ``trace_split`` (implies
    ``full_state``) turns on the C# oracle's state trace and explains the
    split with each diverging field's C# change history (``split_trace``).

    Checkpoints (see ``write_checkpoint``): with ``checkpoint_dir`` set, the
    run saves a checkpoint at the first ``map_select`` of each act in
    ``checkpoint_acts`` where both engines still agree, and with
    ``stop_at_checkpoint`` it ends (terminal ``checkpointed``) once all are
    saved. ``from_checkpoint`` starts the run at a checkpoint instead of at
    Neow. ``branch`` picks the walk from there: each branch seeds the random
    policy differently, so one checkpoint yields many act-2/3 trials. With
    ``from_checkpoint``, ``max_steps`` counts the steps after the checkpoint.

    ``random_loadout`` replaces the deck, potions and gold, and adds relics, at
    the run's first map (see ``_graft_loadout``). The loadout is a pure
    function of ``(seed, character)``, so rerunning the seed reproduces it.
    """
    full_state = full_state or trace_split
    meta: dict[str, Any] | None = None
    if from_checkpoint is not None:
        meta = read_checkpoint_meta(from_checkpoint)
        character, seed, ascension = meta["character"], meta["seed"], meta["ascension"]
        rng = random.Random(f"act23state:{seed}:act{meta['act']}:branch{branch}")
    else:
        rng = random.Random(f"act23state:{seed}")
    g = make_backend("csharp")()
    r = _RustRecorder(make_backend("rust", target_convention="csharp")())
    res = RunResult(seed=seed, steps=0, terminal="error")
    out = InvRun(result=res, character=character)
    saved_acts: set[int] = set()
    cs_errs_seen = 0
    step0 = 0
    try:
        if meta is not None and from_checkpoint is not None:
            cs_view, rs_view = resume_checkpoint(from_checkpoint, g, r, meta)
            step0 = int(meta["step"])
            out.max_act = int(meta["act"])
            saved_acts = {a for a in checkpoint_acts if a <= out.max_act}
        else:
            cs_view = g.start(character=character, seed=seed, ascension=ascension)
            rs_view = r.start(character=character, seed=seed, ascension=ascension)
        # The trace segment (one per C# action) current at the last clean
        # full-state check: on a split, ``get_trace`` returns what came after.
        trace_seq = 0
        if trace_split:
            g.send({"cmd": "trace", "on": True})

        for step in range(step0, step0 + max_steps):
            res.steps = step
            # Track the deepest act EITHER engine reached. They can briefly
            # disagree at an act boundary (one advances a step before the
            # other), and that disagreement is itself a divergence we want to
            # report as "the driver got this deep" — so take the max.
            acts = [a for a in (_act_of(rs_view), _act_of(cs_view)) if a]
            if acts:
                out.max_act = max(out.max_act, *acts)

            cs_dec = cs_view.get("decision") or cs_view.get("type")
            cs_snap, rs_snap = Snap.of(cs_view, "cs"), Snap.of(rs_view, "rs")

            # Atomic-select skip (ported from ``run_parity.py``'s lockstep guard).
            # Rust SURFACES a played card's OnPlay select so an MCTS agent can
            # pick, but several selects it resolves ATOMICALLY (first-legal /
            # deterministic-order) without surfacing:
            #   - Echo / OneTwoPunch replay siblings (the choice opens mid-apply
            #     and cannot surface) and a discard-N whose hand exactly fits;
            #   - turn-start power selects like ForegoneConclusion
            #     (``hooks.rs`` auto-picks the 2 lowest-(rarity,id) draw cards,
            #     matching C#'s first FromSimpleGrid option, then RACES on — often
            #     all the way through combat-end into ``card_reward``);
            #   - synchronous filtered deck removals (Amalgamator: remove 2 basic
            #     Strikes/Defends), which leave Rust in ``map`` / ``card_reward``.
            # In all of these the *same* action leaves C# paused at a
            # ``card_select`` while Rust has already moved on. Driving the shared
            # lockstep here would (a) false-flag the phase/state gap as a
            # divergence and (b) feed C#'s card_select verb to a Rust that is no
            # longer selecting → a phantom ``act_reject``. Instead advance C# ALONE
            # through the select with its first legal pick — the analog of Rust's
            # atomic first-legal resolution — without touching Rust, then re-snap.
            # The next iteration compares C#'s post-select state against Rust's, so
            # a genuinely divergent pick (Rust's atomic choice ≠ C#'s first option)
            # still surfaces there. Gated on ``rs != card_select`` so a REAL shared
            # card_select (Rust ALSO paused, e.g. a surfaced OnPlay pick) is driven
            # normally through the shared lockstep. Beads `sts2-rs-RP25` (original,
            # rs=combat only) / `sts2-rs-RP28` (generalized to any non-select Rust
            # phase: ForegoneConclusion → card_reward, Amalgamator removal → map).
            if _cs_alone_select(cs_snap.phase, rs_snap.phase):
                cs_choices = claws_atomic_choice(cs_view) or enumerate_choices(g, cs_view)
                if cs_choices:
                    verb, kw = cs_choices[0]
                    nxt_cs = g.act(verb, **kw)
                    if not _is_err(nxt_cs):
                        if verbose:
                            res.trace.append(
                                f"[{step:04d}] act{out.max_act} "
                                f"atomic-select-skip cs:{verb}({kw}) (rust resolved atomically)"
                            )
                        out.atomic_select_skips += 1
                        cs_view = nxt_cs
                        continue
                # Could not advance C# alone (no legal pick, or C# rejected its
                # own choice) — fall through so the state diff reports it rather
                # than silently masking a stuck select.

            # MIRROR of the atomic-select skip: RUST surfaces a played card's
            # OnPlay select (or a turn-start power select) that C# resolved
            # ATOMICALLY, so C# is already back in ``combat`` / has moved on while
            # Rust is paused at ``card_select``. Drive RUST ALONE to FINALIZE its
            # select — ``skip_select`` → ``combat_resolve_choice{option: None}``,
            # the analog of C#'s one-shot ``ResolvePendingByIndices`` — WITHOUT
            # touching C#, then re-snap. The next iteration compares Rust's
            # post-select state against C#'s, so a genuinely divergent atomic pick
            # (Rust's finalize ≠ C#'s resolution) still surfaces there. Gated on
            # ``cs != card_select`` so a REAL shared card_select (both paused) is
            # driven normally. Bounded so a select that never clears (finalize
            # rejected, e.g. unmet ``min_select``) falls through to the diff
            # instead of wedging. Oracle-artifact "run-combat-cardselect".
            #
            # Before that, rule out a C# view that HID its pending select
            # (``_cs_select_still_pending``): then the select is shared, and
            # finalizing Rust alone would leave C# to resolve it later with a
            # different pick. Drive both engines through the same pick.
            if (
                rs_snap.phase == "card_select"
                and cs_snap.phase == "combat"
                and _cs_select_still_pending(g)
            ):
                verb, kw = _stale_cs_select_pick(rs_view)
                nxt_cs = g.act(verb, **kw)
                nxt_rs = r.act(verb, **kw)
                if not _is_err(nxt_cs) and not _is_err(nxt_rs):
                    if verbose:
                        res.trace.append(
                            f"[{step:04d}] act{out.max_act} "
                            f"stale-cs-select both:{verb}({kw}) (c# view hid its select)"
                        )
                    out.stale_cs_selects += 1
                    cs_view, rs_view = nxt_cs, nxt_rs
                    continue
                cs_view = nxt_cs if not _is_err(nxt_cs) else cs_view
                rs_view = nxt_rs if not _is_err(nxt_rs) else rs_view
                cs_snap, rs_snap = Snap.of(cs_view, "cs"), Snap.of(rs_view, "rs")
            guard = 0
            while rs_snap.phase == "card_select" and cs_snap.phase != "card_select" and guard < 8:
                nxt_rs = rust_finalize_atomic_select(r, rs_view)
                if _is_err(nxt_rs):
                    break  # not skippable / min_select unmet — let the diff report it
                if verbose:
                    res.trace.append(
                        f"[{step:04d}] act{out.max_act} "
                        f"atomic-select-skip rs:skip_select (c# resolved atomically)"
                    )
                out.atomic_select_skips += 1
                rs_view = nxt_rs
                rs_snap = Snap.of(rs_view, "rs")
                guard += 1

            # Event-granularity skip: the C# HEADLESS auto-skips Custom-layout
            # events (FakeMerchant: ``GenerateInitialOptions => Array.Empty`` →
            # "no options, auto-skipping" → ``EnterRoom(new MapRoom())``), so C# is
            # already on the ``map`` while Rust correctly surfaces the FakeMerchant
            # ``event``. Drive RUST ALONE through its no-op ``leave`` option so it
            # re-aligns to the map, then re-snap. Sound because leaving draws
            # nothing: FakeMerchant's only shared-RNG reads (6 ``Shops`` price
            # rolls, ``MerchantRelicEntry.CalcCost``) happen in
            # ``BeforeEventStarted``, which both engines run on entry; its
            # inventory shuffle uses a private ``FAKE_MERCHANT`` stream. A
            # genuine "Rust a room behind" desync still surfaces in the
            # subsequent map/nav diff. Bead sts2-rs-1hiht.
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
                    if not _is_err(nxt_rs):
                        if verbose:
                            res.trace.append(
                                f"[{step:04d}] act{out.max_act} "
                                f"event-gran-skip rs:leave (c# headless auto-skipped)"
                            )
                        out.event_skips += 1
                        rs_view = nxt_rs
                        rs_snap = Snap.of(rs_view, "rs")

            # Final-victory event granularity: after the act-3 boss C# enters
            # TheArchitect's VICTORY event room — a single-option dialogue chain
            # (THREATEN → dialogue → PROCEED) whose last PROCEED calls
            # ``RunManager.WinRun`` (see ``RunSimulator.cs``'s TheArchitect
            # note) — while Rust ends the run at the boss kill. Drive C# ALONE
            # through that chain, then re-snap; the ``game_over`` check below
            # still compares the outcome on both sides. Gated on Rust being over
            # and the C# event being TheArchitect, so any other event-vs-over
            # split still surfaces.
            if (
                rs_snap.phase == "game_over"
                and cs_snap.phase == "event"
                and _is_architect_event(cs_view)
            ):
                cs_choices = enumerate_choices(g, cs_view)
                if len(cs_choices) == 1:
                    verb, kw = cs_choices[0]
                    nxt_cs = g.act(verb, **kw)
                    if not _is_err(nxt_cs):
                        if verbose:
                            res.trace.append(
                                f"[{step:04d}] act{out.max_act} "
                                f"victory-event-skip cs:{verb}({kw}) (rust ended at the boss)"
                            )
                        out.event_skips += 1
                        cs_view = nxt_cs
                        continue

            # Finer per-step nav instrumentation (issue #158 step 1): act +
            # current map-node coord for BOTH engines, surfaced every step in the
            # verbose trace so the Act-1→Act-2 "C# a room ahead" desync is
            # *visible at its true step* instead of being reconstructed from a
            # downstream gold/deck/phase cascade.
            cs_nav, rs_nav = Nav.of(cs_view, g), Nav.of(rs_view, r)

            fields = diff_snaps(cs_snap, rs_snap)
            # A WON run ends in C# through ``RunManager.WinRun → OnEnded(isVictory:
            # true) → GuaranteeKillAllPlayers`` (see ``RunSimulator.cs``'s
            # game-over note), so its victory ``game_over`` view reports hp 0
            # while Rust keeps the post-boss HP. Not a parity signal: drop the hp
            # field when both sides are over and C# says it won.
            if (
                cs_snap.phase == "game_over"
                and rs_snap.phase == "game_over"
                and (cs_view.get("victory") or cs_view.get("won"))
            ):
                fields = [f for f in fields if not f.startswith("hp:")]
            if cs_snap.phase == "map" and rs_snap.phase == "map":
                fields += diff_maps(map_signature(g), map_signature(r))
                # Nav (act + current-node) is BLOCKING only while both engines
                # are on the map — there the current coord is authoritative on
                # both, so a mismatch is a genuine "standing on different nodes"
                # desync that ``diff_maps`` (graph-only) misses. Off the map it
                # is trace-only: at the act boundary C# legitimately leads by one
                # act/room for a step (it regenerates the next act's map and
                # enters its entry event before Rust advances), and 92%+ of runs
                # resolve that transient and cross cleanly — so flagging nav at
                # the transitional event/card_reward would false-diverge and
                # REGRESS reach, the opposite of this probe's goal.
                fields += diff_nav(cs_nav, rs_nav)

            # Oracle-artifact classification (issue #158 step 3): the C#
            # mid-combat HP *snapshot* is taken at a slightly different point in
            # the damage/animation resolution than Rust's, so while BOTH engines
            # are in combat the player-HP read can differ by a small,
            # non-directional amount that self-heals a step or two later. Left
            # un-classified it terminates the run and MASKS every deeper real
            # divergence. Peel it off: when the ONLY field is a small player-HP
            # delta and both sides are mid-combat, record it as an artifact and
            # keep walking (the in-combat enemy-HP / deck / phase checks stay
            # un-masked, so a REAL combat-damage divergence still surfaces — it
            # carries an enemies_hp/phase field and is NOT peeled here). A delta
            # that exceeds the threshold, or persists/grows, falls through to a
            # genuine divergence.
            if peel_hp_artifacts and fields and _is_combat_hp_artifact(cs_snap, rs_snap, fields):
                out.hp_artifacts += 1
                out.max_hp_artifact = max(
                    out.max_hp_artifact, abs((cs_snap.hp or 0) - (rs_snap.hp or 0))
                )
                if verbose:
                    res.trace.append(
                        f"[{step:04d}] act{out.max_act} {cs_snap.phase:11s} "
                        f"hp-artifact cs={cs_snap.hp} rs={rs_snap.hp} "
                        f"(Δ{abs((cs_snap.hp or 0) - (rs_snap.hp or 0))}, tolerated)"
                    )
                fields = []

            if (
                (full_state or full_state_blocking)
                and out.full_split is None
                and cs_snap.phase == "combat"
                and rs_snap.phase == "combat"
            ):
                cs_state, divs = full_state_divs(g, r)
                split = _split_lines(divs)
                errs = _cs_world_errors(g)
                # Did the C# world throw since the last compared step?
                threw = errs is not None and errs["count"] > cs_errs_seen
                if errs is not None:
                    cs_errs_seen = errs["count"]
                if not split and trace_split:
                    trace_seq = cs_state.get("trace_seq") or trace_seq
                if split:
                    out.full_split = FullSplit(step, out.max_act, split[:FULL_SPLIT_MAX_PATHS])
                    if threw and errs is not None:
                        sample = errs["sample"]
                        out.full_split.cs_threw = sample[-1] if sample else "exception"
                    if trace_split:
                        segs = (
                            g.send({"cmd": "get_trace", "since": trace_seq}).get("segments") or []
                        )
                        out.full_split.trace = explain(segs, divs[:FULL_SPLIT_MAX_PATHS])
                        out.full_split.suspects = suspects(segs, divs[:FULL_SPLIT_MAX_PATHS])
                    if verbose:
                        res.trace.append(f"[{step:04d}] act{out.max_act} FULL-STATE SPLIT:")
                        res.trace.extend(f"       {line}" for line in split[:FULL_SPLIT_MAX_PATHS])
                    if full_state_blocking and not fields:
                        res.divergence = Divergence(step, "combat", split, "full")
                        res.terminal = "diverged"
                        return out

            if verbose:
                res.trace.append(
                    f"[{step:04d}] act{out.max_act} {cs_snap.phase:11s} "
                    f"cs[hp={cs_snap.hp} gold={cs_snap.gold} deck={cs_snap.deck_size} "
                    f"coord={cs_nav.coord} floor={cs_nav.floor}] "
                    f"rs[hp={rs_snap.hp} coord={rs_nav.coord} floor={rs_nav.floor}] "
                    f"enemies cs={list(cs_snap.enemies)} rs={list(rs_snap.enemies)}"
                )
            if fields:
                # On a phase split, record each engine's raw decision + the C#
                # offered choices, so a ``cs=map rs=event`` style room-entry
                # granularity skew shows WHICH map move C# still has pending vs
                # the room Rust already resolved (issue #158 step 2 triage).
                if verbose and cs_snap.phase != rs_snap.phase:
                    res.trace.append(
                        f"       phase-split: cs_dec={cs_view.get('decision') or cs_view.get('type')!r} "
                        f"cs_choices={enumerate_choices(g, cs_view)[:4]} "
                        f"rs_dec={rs_view.get('decision') or rs_view.get('type')!r}"
                    )
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    fields,
                    "map"
                    if any(f.startswith("map") for f in fields)
                    else ("nav" if all(f.startswith("nav") for f in fields) else "state"),
                )
                res.terminal = "diverged"
                return out

            if cs_dec == "game_over":
                res.cs_won = bool(cs_view.get("victory") or cs_view.get("won"))
                rs_over = Snap.of(rs_view, "rs").phase == "game_over"
                res.rs_won = bool(rs_view.get("victory") or rs_view.get("won") or rs_over)
                res.terminal = "game_over_match" if rs_over else "diverged"
                if not rs_over:
                    res.divergence = Divergence(
                        step, "game_over", [f"cs=game_over rs={rs_snap.phase}"], "state"
                    )
                return out

            # Past Neow, and the diff above just passed: both engines take the
            # loadout from the same state.
            if random_loadout and out.loadout is None and cs_dec == "map_select":
                out.loadout = generate_loadout(seed, character)
                problems = _graft_loadout(g, r, out.loadout, rs_view)
                if verbose:
                    res.trace.append(f"[{step:04d}] graft {out.loadout}")
                if problems:
                    res.divergence = Divergence(step, cs_snap.phase, problems, "harness")
                    res.terminal = "error"
                    return out

            # Checkpoint here: the diff above just passed, and HP isn't pinned
            # for this step yet, so a resume replays the pin exactly as now.
            ck_act = _act_of(cs_view) or 0
            if (
                checkpoint_dir is not None
                and cs_dec == "map_select"
                and ck_act in checkpoint_acts
                and ck_act not in saved_acts
            ):
                write_checkpoint(
                    checkpoint_dir,
                    character=character,
                    seed=seed,
                    ascension=ascension,
                    act=ck_act,
                    step=step,
                    g=g,
                    rec=r,
                    rs_view=rs_view,
                    parent=f"{from_checkpoint.name}.b{branch}" if from_checkpoint else "",
                )
                saved_acts.add(ck_act)
                out.checkpoints.append(ck_act)
                if stop_at_checkpoint and saved_acts >= checkpoint_acts:
                    res.terminal = "checkpointed"
                    return out

            choices = enumerate_choices(g, cs_view)
            if not choices:
                res.divergence = Divergence(
                    step, cs_snap.phase, [f"no choices for cs decision {cs_dec!r}"], "harness"
                )
                res.terminal = "error"
                return out
            capped = _within_select_index_cap(choices)
            if not capped:
                # Every pick names an option past the u8 cap (cannot happen
                # for a single pick, whose options 0..255 always survive).
                # End as a harness cap, not a divergence.
                res.divergence = Divergence(
                    step,
                    cs_snap.phase,
                    [f"every {cs_dec!r} pick is past option {MAX_SELECT_INDEX}"],
                    "harness",
                )
                res.terminal = "select_index_cap"
                return out
            if len(capped) < len(choices):
                out.select_index_caps += 1
            choices = capped

            # Pin HP out of combat (AFTER the genuine diff above): the next
            # combat then seeds from INVINCIBLE_HP on both engines so the random
            # policy survives, while in-combat HP stays an un-masked parity check.
            #
            # Never pin BETWEEN the pages of an event either: an event can bake
            # the player's max HP into a page's option at page-build time, and a
            # pin between that page and the click changes max HP under it in a
            # way the real game cannot. Tablet of Truth's last Decipher page
            # sets ``DecipherMaxHpLoss = MaxHp - 1`` when the page is built
            # (TabletOfTruth.cs:88) and the click pays that stored value
            # (TabletOfTruth.cs:109); Rust computes ``MaxHp - 1`` at the click.
            # Both agree in a real run, but with a pin in between C# ended at
            # 25 max HP and Rust at 1 (Defect:1002, Regent:1007 at v0.107.1).
            # The map step before the event already pinned, so the event and
            # anything it leads to still start from INVINCIBLE_HP.
            #
            # Nor inside a fight that is showing a select (``card_select``
            # phase): a select a kill's ``Hook.AfterDeath`` opened does not
            # block (HookPlayerChoiceContext), so it can surface mid-combat,
            # after the next player turn started, where a pin rewrites HP
            # the fight is still checking (Regent v107 1028).
            if invincible and cs_snap.phase not in ("combat", "event") and r.combat_state() is None:
                _pin(g, r, INVINCIBLE_HP)

            # Pick a choice; on a both-reject (benign illegal move) drop it and
            # re-pick so one bad enumerated move doesn't truncate the run.
            pool = list(choices)
            if invincible and survival_policy:
                pool = _survival_pool(pool, cs_view)
            advanced = False
            while pool:
                # The survival pick draws no RNG, so a walk that never meets a
                # survival card is unchanged.
                sidx = _survival_choice(pool, cs_view) if invincible and survival_policy else None
                idx = sidx if sidx is not None else rng.randrange(len(pool))
                verb, kw = pool.pop(idx)
                nxt_cs = g.act(verb, **kw)
                nxt_rs = r.act(verb, **kw)
                cs_err, rs_err = _is_err(nxt_cs), _is_err(nxt_rs)
                if cs_err and rs_err:
                    out.repicks += 1
                    continue  # both reject => benign, try another choice
                if cs_err or rs_err:
                    rs_msg = (nxt_rs.get("error") or nxt_rs.get("message")) if rs_err else None
                    cs_msg = nxt_cs.get("message") if cs_err else None
                    if cs_err and nxt_cs.get("stack_trace"):
                        out.cs_stack = nxt_cs["stack_trace"]
                        if verbose:
                            res.trace.append(f"C# stack_trace:\n{out.cs_stack}")
                    res.divergence = Divergence(
                        step,
                        cs_snap.phase,
                        [f"act({verb},{kw}) -> cs_err={cs_msg} rs_err={rs_msg}"],
                        "act_reject",
                    )
                    res.terminal = "diverged"
                    return out
                cs_view, rs_view = nxt_cs, nxt_rs
                advanced = True
                break
            if not advanced:
                # Every enumerated choice was rejected by both engines.
                res.terminal = "cs_act_reject"
                return out

        res.terminal = "max_steps"
        return out
    except CheckpointStale as exc:
        res.divergence = Divergence(res.steps, "?", [str(exc)], "harness")
        res.terminal = "checkpoint_stale"
        return out
    except Exception as exc:  # pragma: no cover - harness-level failure
        import traceback

        res.divergence = Divergence(res.steps, "?", [f"{type(exc).__name__}: {exc}"], "harness")
        res.terminal = "error"
        if verbose:
            res.trace.append(traceback.format_exc())
        return out
    finally:
        for backend in (g, r):
            try:
                backend.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
@dataclass
class SweepReport:
    runs: int = 0
    reached_act2: int = 0
    reached_act3: int = 0
    act_hist: Counter = field(default_factory=Counter)
    term_hist: Counter = field(default_factory=Counter)
    buckets: dict[str, list[str]] = field(default_factory=dict)
    hp_artifact_runs: int = 0  # runs that tolerated >=1 mid-combat HP artifact
    hp_artifacts: int = 0  # total artifacts peeled across the sweep
    max_hp_artifact: int = 0  # largest tolerated |Δhp| (sanity-checks the bound)
    # ``--full-state``: earliest full-state split per run, bucketed by field.
    full_buckets: dict[str, list[str]] = field(default_factory=dict)
    full_clean: int = 0
    checkpointed: int = 0  # runs that ended at their last checkpoint


# Frames of a one-sided C# reject's stack trace a sweep prints per run.
SWEEP_STACK_FRAMES = 8


def _stack_head(stack: str | None) -> list[str]:
    """The first ``SWEEP_STACK_FRAMES`` frames of a C# stack trace, compacted."""
    if not stack:
        return []
    frames = [ln.strip() for ln in stack.splitlines() if ln.strip()]
    return frames[:SWEEP_STACK_FRAMES]


def _worker(
    job: dict[str, Any],
) -> tuple[str, str, int, str, str | None, int, int, FullSplit | None, list[str]]:
    """Run one sweep job: ``label`` plus ``run_one_invincible`` keyword args."""
    kwargs = dict(job)
    label = kwargs.pop("label")
    out = run_one_invincible(**kwargs)
    res = out.result
    sig = run_signature(res)
    div = None
    if res.divergence is not None:
        div = f"{res.divergence.phase}@{res.divergence.step}: {res.divergence.fields[:2]}"
    if out.checkpoints:
        sig += f" [ckpt act{','.join(map(str, out.checkpoints))}]"
    return (
        label,
        sig,
        out.max_act,
        res.terminal,
        div,
        out.hp_artifacts,
        out.max_hp_artifact,
        out.full_split,
        _stack_head(out.cs_stack),
    )


def sweep(
    characters: list[str],
    n: int,
    start: int,
    ascension: int,
    max_steps: int,
    workers: int,
    invincible: bool,
    peel_hp_artifacts: bool = True,
    full_state: bool = False,
    full_state_blocking: bool = False,
    trace_split: bool = False,
    checkpoint_dir: Path | None = None,
    checkpoint_acts: frozenset[int] = frozenset(),
    stop_at_checkpoint: bool = False,
    from_checkpoints: list[Path] | None = None,
    branches: int = 1,
    survival_policy: bool = True,
    random_loadout: bool = False,
) -> SweepReport:
    rep = SweepReport()
    common: dict[str, Any] = {
        "max_steps": max_steps,
        "invincible": invincible,
        "peel_hp_artifacts": peel_hp_artifacts,
        "full_state": full_state,
        "full_state_blocking": full_state_blocking,
        "trace_split": trace_split,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_acts": checkpoint_acts,
        "stop_at_checkpoint": stop_at_checkpoint,
        "survival_policy": survival_policy,
        "random_loadout": random_loadout,
    }
    if from_checkpoints is not None:
        # Every branch of every checkpoint is its own trial.
        jobs = [
            {
                **common,
                "label": f"{ck.name}#b{b}",
                "seed": "",
                "from_checkpoint": ck,
                "branch": b,
            }
            for ck in from_checkpoints
            for b in range(branches)
        ]
    else:
        jobs = [
            {
                **common,
                "label": f"{ch}:{start + i}",
                "seed": str(start + i),
                "character": ch,
                "ascension": ascension,
            }
            for ch in characters
            for i in range(n)
        ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, j): j["label"] for j in jobs}
        for fut in as_completed(futs):
            label, sig, max_act, term, div, hp_arts, max_hp_art, split, stack = fut.result()
            rep.runs += 1
            rep.act_hist[max_act] += 1
            rep.term_hist[term] += 1
            if max_act >= 2:
                rep.reached_act2 += 1
            if term == "checkpointed":
                rep.checkpointed += 1
            if max_act >= 3:
                rep.reached_act3 += 1
            if hp_arts:
                rep.hp_artifact_runs += 1
                rep.hp_artifacts += hp_arts
                rep.max_hp_artifact = max(rep.max_hp_artifact, max_hp_art)
            rep.buckets.setdefault(sig, [])
            art = f" [+{hp_arts}hp-art]" if hp_arts else ""
            if div:
                rep.buckets[sig].append(f"{label} act{max_act} {div}{art}")
            else:
                rep.buckets[sig].append(f"{label} act{max_act}{art}")
            full = ""
            if full_state or full_state_blocking or trace_split:
                if split is None:
                    rep.full_clean += 1
                    full = "  full=clean"
                else:
                    rep.full_buckets.setdefault(split.classes, []).append(f"{label} {split}")
                    full = f"  full@{split.step}={split.classes}"
                    if split.suspects:
                        full += f" src={','.join(split.suspects)}"
            print(f"  {label:24s} act{max_act} {term:16s} {sig}{art}{full}", flush=True)
            if div and stack:
                # One-sided C# reject: name the throwing frame (GH #1147).
                print(f"    {label} {div}", flush=True)
                for frame in stack:
                    print(f"      cs_stack| {frame}", flush=True)
    return rep


def print_report(rep: SweepReport) -> None:
    print("\n=== ACT-DEPTH ===")
    print(f"runs:            {rep.runs}")
    pct2 = 100 * rep.reached_act2 / rep.runs if rep.runs else 0
    pct3 = 100 * rep.reached_act3 / rep.runs if rep.runs else 0
    print(f"reached act>=2:  {rep.reached_act2} ({pct2:.0f}%)")
    print(f"reached act>=3:  {rep.reached_act3} ({pct3:.0f}%)")
    print(f"max-act hist:    {dict(sorted(rep.act_hist.items()))}")
    print(f"terminal hist:   {dict(rep.term_hist.most_common())}")
    if rep.checkpointed:
        print(f"checkpointed:    {rep.checkpointed}")
    print(
        f"hp-artifacts:    {rep.hp_artifacts} peeled across {rep.hp_artifact_runs} runs "
        f"(max |Δhp|={rep.max_hp_artifact}, tolerance={HP_ARTIFACT_TOLERANCE})"
    )
    print("\n=== BUCKETS (by run_signature) ===")
    for sig, examples in sorted(rep.buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        clean = sig.startswith("clean:")
        tag = "OK  " if clean else "DIV "
        print(f"{tag}[{len(examples):3d}] {sig}")
        if not clean:
            for ex in examples[:4]:
                print(f"        e.g. {ex}")
    if rep.full_buckets or rep.full_clean:
        print("\n=== FULL-STATE FIRST SPLITS (by diverging fields) ===")
        print(f"full-state clean: {rep.full_clean}/{rep.runs}")
        for cls, examples in sorted(rep.full_buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"[{len(examples):3d}] {cls}")
            for ex in examples[:4]:
                print(f"        e.g. {ex}")


def _utf8_stdio() -> None:
    """Write stdout/stderr as UTF-8.

    Redirected to a file or pipe on Windows, they default to cp1252, which has
    no ``Δ``. The run summary prints ``max |Δhp|``, so a finished run crashed
    there with ``UnicodeEncodeError`` and lost its verdict.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def main() -> int:
    _utf8_stdio()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--character",
        default=None,
        help="default Ironclad; with --from-checkpoints, default every character",
    )
    ap.add_argument("--all-characters", action="store_true")
    ap.add_argument(
        "--ascension",
        type=int,
        default=None,
        help="default 0; with --from-checkpoints, default every ascension",
    )
    ap.add_argument("-n", "--num", type=int, default=10, help="seeds per character")
    ap.add_argument("--start", type=int, default=1000, help="first seed (ints, stringified)")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-invincible", action="store_true", help="baseline (death-limited) compare")
    ap.add_argument(
        "--no-hp-peel",
        action="store_true",
        help="do NOT classify out the C# mid-combat-snapshot HP artifact (raw reach, for A/B)",
    )
    ap.add_argument(
        "--full-state",
        action="store_true",
        help="also diff the full combat state (piles in order, RNG cursors, move "
        "states, ...) every combat step and report each run's first split",
    )
    ap.add_argument(
        "--full-state-blocking",
        action="store_true",
        help="like --full-state, but end the run at its first full-state split",
    )
    ap.add_argument(
        "--trace-split",
        action="store_true",
        help="--full-state plus the C# oracle's state trace: explain each run's "
        "first split with every diverging field's C# change history and source",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="save a checkpoint at the first map of each --checkpoint-acts act "
        "(where both engines still agree) into this directory",
    )
    ap.add_argument(
        "--checkpoint-acts",
        default="2",
        help="comma list of acts to checkpoint (default 2)",
    )
    ap.add_argument(
        "--stop-at-checkpoint",
        action="store_true",
        help="end each run once its last checkpoint is saved (build mode)",
    )
    ap.add_argument(
        "--from-checkpoints",
        type=Path,
        help="start every run from the checkpoints in this directory instead "
        "of at Neow; -n/--start are ignored. Every checkpoint runs unless "
        "--character, --ascension or --checkpoint-act narrows it",
    )
    ap.add_argument(
        "--checkpoint-act",
        type=int,
        help="with --from-checkpoints: only use checkpoints of this act",
    )
    ap.add_argument(
        "--branches",
        type=int,
        default=4,
        help="with --from-checkpoints: random walks per checkpoint (default 4)",
    )
    ap.add_argument(
        "--uniform-policy",
        action="store_true",
        help="pick uniformly at random even when a SURVIVAL_CARDS card is "
        "playable (the pre-2026-11 walk, which dies to The Insatiable)",
    )
    ap.add_argument(
        "--random-loadout",
        action="store_true",
        help="at the first map, replace the deck, potions and gold and add relics "
        "from the full pools, on both engines, without on-obtain hooks; the "
        "seed reproduces the loadout. Not with --from-checkpoints",
    )
    ap.add_argument("--seed", help="single verbose run for this seed (overrides sweep)")
    ap.add_argument("--verbose", action="store_true", help="dump per-step trace (single run)")
    add_game_version_arg(ap)
    args = ap.parse_args()
    apply_game_version_arg(args)

    assert_wheel_fresh()
    invincible = not args.no_invincible
    peel = not args.no_hp_peel
    # With --from-checkpoints, the checkpoints carry their own character and
    # ascension, so an unset --character / --ascension filters nothing.
    # (Defaulting to Ironclad there silently dropped 4 of 5 characters.)
    ckpt_chars = [args.character] if args.character and not args.all_characters else None
    ckpt_asc = args.ascension
    args.character = args.character or "Ironclad"
    args.ascension = 0 if args.ascension is None else args.ascension
    chars = ALL_CHARACTERS if args.all_characters else [args.character]
    if args.random_loadout and args.from_checkpoints:
        # A checkpoint already carries whatever its run was grafted with.
        ap.error("--random-loadout starts at Neow, so it can't take --from-checkpoints")
    ckpt_acts = frozenset(int(a) for a in args.checkpoint_acts.split(",") if a.strip())
    from_ckpts: list[Path] | None = None
    if args.from_checkpoints is not None:
        from_ckpts = []
        for ck in list_checkpoints(args.from_checkpoints):
            m = read_checkpoint_meta(ck)
            if ckpt_chars is not None and m["character"] not in ckpt_chars:
                continue
            if ckpt_asc is not None and m["ascension"] != ckpt_asc:
                continue
            if args.checkpoint_act is not None and m["act"] != args.checkpoint_act:
                continue
            from_ckpts.append(ck)
        if not from_ckpts:
            print(f"no matching checkpoints in {args.from_checkpoints}", file=sys.stderr)
            return 2

    if args.seed is not None:
        out = run_one_invincible(
            args.seed,
            args.character,
            ascension=args.ascension,
            max_steps=args.max_steps,
            invincible=invincible,
            verbose=args.verbose,
            peel_hp_artifacts=peel,
            full_state=args.full_state,
            full_state_blocking=args.full_state_blocking,
            trace_split=args.trace_split,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_acts=ckpt_acts,
            stop_at_checkpoint=args.stop_at_checkpoint,
            survival_policy=not args.uniform_policy,
            random_loadout=args.random_loadout,
        )
        res = out.result
        if args.verbose:
            for line in res.trace[-80:]:
                print(line)
        print(
            f"\nseed={args.seed} char={args.character} max_act={out.max_act} "
            f"steps={res.steps} terminal={res.terminal} repicks={out.repicks} "
            f"atomic_select_skips={out.atomic_select_skips} "
            f"stale_cs_selects={out.stale_cs_selects} "
            f"select_index_caps={out.select_index_caps} "
            f"event_skips={out.event_skips} "
            f"hp_artifacts={out.hp_artifacts} (max |Δhp|={out.max_hp_artifact})"
        )
        if out.full_split is not None:
            fs = out.full_split
            print(f"FULL-STATE SPLIT @step {fs.step} act{fs.act}")
            if fs.cs_threw:
                print(f"  C# THREW during the split step: {fs.cs_threw}")
            for line in fs.paths:
                print(f"  {line}")
            if fs.trace:
                print("C# CHANGES OVER THE SPLIT STEP (state trace):")
                for line in fs.trace:
                    print(f"  {line}")
        if res.divergence:
            d = res.divergence
            print(f"DIVERGENCE @step {d.step} phase={d.phase} kind={d.kind}")
            for fld in d.fields:
                print(f"  {fld}")
            print(f"signature: {run_signature(res)}")
        return 1 if res.divergence else 0

    print(
        f"invincible={invincible} peel_hp_artifacts={peel} chars={chars} n={args.num} "
        f"start={args.start} max_steps={args.max_steps}"
        + (f" from_checkpoints={len(from_ckpts)}x{args.branches}" if from_ckpts else "")
    )
    rep = sweep(
        chars,
        args.num,
        args.start,
        args.ascension,
        args.max_steps,
        args.workers,
        invincible,
        peel_hp_artifacts=peel,
        full_state=args.full_state,
        full_state_blocking=args.full_state_blocking,
        trace_split=args.trace_split,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_acts=ckpt_acts,
        stop_at_checkpoint=args.stop_at_checkpoint,
        from_checkpoints=from_ckpts,
        branches=args.branches,
        survival_policy=not args.uniform_policy,
        random_loadout=args.random_loadout,
    )
    print_report(rep)
    n_div = sum(len(v) for k, v in rep.buckets.items() if not k.startswith("clean:"))
    return 1 if n_div else 0


if __name__ == "__main__":
    sys.exit(main())
