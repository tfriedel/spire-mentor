#!/usr/bin/env python3
"""C#↔Rust EVENT parity — the per-act event POOL and per-event OPTION lists.

Two complementary checks, both combat-free (no map walk, no winning combat —
so they cover ALL THREE ACTS, not just act 1, which the lockstep whole-run
sweep can only reach when random play happens to clear the bosses):

  POOL parity  — does each act's shuffled event pool (the `RoomSet.events`
    list, `(AllEvents ++ AllSharedEvents)` UnstableShuffled off the `up_front`
    stream in C# `ActModel.GenerateRooms`) match Rust's `events_shuffled`
    entry-for-entry, in order, across acts 1/2/3? This catches a wrong event
    set OR a wrong shuffle (RNG misalignment). Uses the headless `dump_rooms`
    command vs the PyO3 `generated_rooms()` accessor — both pre-generated at
    run start, no traversal needed.

  OPTION parity — for every event in the union of those pools, force the
    event on each engine and diff its presented options. C# side:
    `enter_room {type: event, event: <Id.Entry>}` (`RunSimulator.EnterRoomImpl`)
    → the `event_choice` decision view. Rust side: the new
    `RunManager.force_event(<python_name>)` + `event_option_infos()`. We
    compare, by option position:
      * OPTION COUNT                       (hard — a missing/extra option)
      * the ENABLED / locked vector        (hard — C# `is_locked` ⇔ Rust
                                            `!enabled`; a wrongly (un)locked
                                            option is a real divergence)
    and report, advisory-only:
      * option_id STRING drift             (Rust emits a short canonical
                                            `option_id` e.g. `REACH`; C# emits
                                            the loc-key tail `text_key` e.g.
                                            `REACH_INTO_THE_FLESH` — different
                                            id namespaces BY DESIGN, same as the
                                            accepted SelfHelpBook READ_PASSAGE
                                            case, so this never fails the run).

  EFFECT parity (``--effect``) — option count + enabled is necessary but not
    sufficient: a handler can present an identical menu while granting the wrong
    relic, no-opping a deck mutation, finishing a multi-page chain early, or
    skipping a nested combat. With ``--effect`` we additionally RESOLVE every
    option (one fresh forced entry per option) on both engines and diff the
    resulting state — relics, deck multiset, potions, gold, hp, the landed
    phase, and the multi-page follow-up option count. This is the check that
    catches the `sts2-rs-m3n0g` event-side stubs that `OPTION parity` is blind
    to. Combat-free, so it covers all three acts.

Usage:
    scripts/wheel-venv.sh run -- python python/event_parity.py
    scripts/wheel-venv.sh run -- python python/event_parity.py --all-characters
    scripts/wheel-venv.sh run -- python python/event_parity.py --pool-seeds 200
    scripts/wheel-venv.sh run -- python python/event_parity.py --effect
    # effect check only (skip the pool/option sweeps):
    scripts/wheel-venv.sh run -- python python/event_parity.py --no-pool --no-options --effect
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import re
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, "python")

import sts2_client
from run_parity import Snap, diff_snaps
from sts2_client import Game
from workbench.backends.rust_game import RustGame


def _rust_run_manager(character: str, seed: str, ascension: int) -> Any:
    """A fresh Rust `RunManager` for the build the C# oracle runs.

    ``game_version`` is the selected build (``STS2_GAME_VERSION``), like
    `act23_map_parity._rust_act_maps` (GH #893). Without it the pool and option
    checks compared a v0.111.0 C# run against a v0.107.1 Rust run, so every
    v0.111.0 act pool "diverged" in order.
    """
    import sts2_rs  # type: ignore[import-not-found]  # lazy: keep module wheel-free to import

    return sts2_rs.RunManager(
        string_seed=seed,
        character=character,
        ascension_level=ascension,
        neow_start=True,
        game_version=sts2_client.GAME_VERSION,
    )


def _rust_backend() -> RustGame:
    """Rust backend in C# convention so its serialized views share the exact
    decision/player shape the C# `Game` emits (what `Snap.of` consumes)."""
    return RustGame(target_convention="csharp")


# Characters as the C# headless / PyO3 wrapper name them (matches
# `run_parity.CHARACTERS`, the ported roster).
ALL_CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

#: Events whose OPTION STRUCTURE (count and/or locked state) depends on
#: accumulated run state — potions / relics / enchantable cards / gold / shop
#: stock — that a bare fresh-run force-entry CANNOT reproduce. Each is gated by
#: an `IsAllowed` precondition (verified against the v0.103.2 decomp) that
#: force-entry deliberately bypasses, so the event gets presented in a state it
#: would NEVER naturally occur in; its options then differ (locked options, or
#: an empty/fallback trade stock) for reasons that vanish in real play, where
#: the event only spawns once its precondition holds. The option COUNT +
#: enabled-vector check is therefore meaningless for these under force-entry —
#: we run it (so a regression in the *other* fields is still visible in the
#: dump) but classify any diff as ADVISORY, never a hard failure. Their pool
#: membership + shuffle order is still hard-checked by `pool_parity`. Mirrors
#: `spawn_roster_parity.RNG_BRANCHED_ALLOWLIST`.
#:
#: C# precondition (`…Models.Events/<Event>.cs`):
#:   StoneOfAllTime      — IsAllowed: act 2 AND every player has a potion
#:                         (LIFT locks when `DrinkAndLiftPotion == null`).
#:   RanwidTheElder      — IsAllowed: act != 1 AND `GetValidRelics(p).Any()`
#:                         (POTION/RELIC options lock without a potion/relic).
#:   GraveOfTheForgotten — IsAllowed: every player `HasEnchantableCards`
#:                         (CONFRONT enchants — locks with no enchantable card).
#:   ZenWeaver           — IsAllowed: `Gold >= EmotionalAwarenessCost`; the two
#:                         pricier options lock when `gold < cost`.
#:   RelicTrader         — stock = `GetValidRelics(Owner).Take(3)`; PROCEED-only
#:                         fallback with no tradeable relics.
#:   TheFutureOfPotions  — stock built from `Owner.Potions`; empty with none.
#:   FakeMerchant        — builds a card/relic/potion buy stock (needs gold).
CONTEXT_DEPENDENT_EVENTS = frozenset(
    {
        "StoneOfAllTime",
        "RanwidTheElder",
        "GraveOfTheForgotten",
        "ZenWeaver",
        "RelicTrader",
        "TheFutureOfPotions",
        "FakeMerchant",
    }
)

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _canon(s: str | None) -> str:
    """Canonicalize a loc key / display string / option id to a bare id.

    Strips a trailing ``.title``/``.name``/``.description`` suffix, takes the
    rightmost ``.``-segment, uppercases, drops non-alphanumerics. Mirrors
    ``parity_walk._loc_id`` so ``REACH_INTO_THE_FLESH`` → ``REACHINTOTHEFLESH``
    and ``EVT.pages.INITIAL.options.REACH`` → ``REACH``.
    """
    if not s:
        return ""
    for suffix in (".title", ".name", ".description"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return _NON_ALNUM.sub("", s.rsplit(".", 1)[-1].upper())


# ─── pool parity ────────────────────────────────────────────────────────────


def _cs_pools(character: str, seed: str, ascension: int) -> list[list[str]]:
    """Per-act C# `RoomSet.events` lists (Id.Entry strings)."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        acts = g.send({"cmd": "dump_rooms"}).get("acts", [])
    finally:
        g.close()
    return [list(a.get("events", [])) for a in acts]


def _rust_pools(character: str, seed: str, ascension: int) -> list[list[str]]:
    """Per-act Rust `events_shuffled` lists (EventId python names)."""
    rm = _rust_run_manager(character, seed, ascension)
    return [list(a.get("events_shuffled", [])) for a in rm.generated_rooms()]


def pool_parity(characters: list[str], base: str, ascension: int, n: int) -> int:
    """Diff each act's shuffled event pool, order-exact. Returns failure count."""
    acts_ok = acts_total = fails = 0
    for character in characters:
        for i in range(n):
            seed = base if n == 1 else f"{base}{i}"
            cs = _cs_pools(character, seed, ascension)
            rs = _rust_pools(character, seed, ascension)
            for ai in range(min(len(cs), len(rs))):
                ce = [_canon(x) for x in cs[ai]]
                re_ = [_canon(x) for x in rs[ai]]
                acts_total += 1
                if ce == re_:
                    acts_ok += 1
                    continue
                fails += 1
                only_cs = sorted(set(ce) - set(re_))
                only_rs = sorted(set(re_) - set(ce))
                print(f"[{character} {seed} act{ai + 1}] POOL DIFF (cs={len(ce)} rs={len(re_)})")
                if only_cs or only_rs:
                    print(f"    only_cs={only_cs}")
                    print(f"    only_rs={only_rs}")
                else:  # same set, different ORDER → shuffle/RNG misalignment
                    print(f"    order cs={ce}")
                    print(f"          rs={re_}")
    print(
        f"\npool parity: {acts_ok}/{acts_total} act-pools order-exact across {len(characters)} char(s)"
    )
    return fails


# ─── option parity ───────────────────────────────────────────────────────────


def _event_universe(character: str, seed: str, ascension: int) -> list[tuple[str, str]]:
    """Union of all acts' events → sorted ``(rust_python_name, cs_entry)`` pairs.

    Paired by `_canon` (the pool check already proved the canon ids align), so
    each event is forced on both engines with its engine-native id.
    """
    rs = _rust_pools(character, seed, ascension)
    cs = _cs_pools(character, seed, ascension)
    cs_by_canon = {_canon(e): e for act in cs for e in act}
    pairs: dict[str, tuple[str, str]] = {}
    for act in rs:
        for name in act:
            c = _canon(name)
            if c in cs_by_canon:
                pairs[c] = (name, cs_by_canon[c])
    return [pairs[k] for k in sorted(pairs)]


def _rust_options(character: str, seed: str, ascension: int, name: str) -> list[tuple[str, bool]]:
    """``(canon_option_id, enabled)`` for a forced Rust event, in order."""
    rm = _rust_run_manager(character, seed, ascension)
    if rm.force_event(name) is None:
        raise LookupError(f"rust: unknown event {name!r}")
    opts = rm.event_option_infos() or []
    return [(_canon(o.get("option_id")), bool(o.get("enabled", True))) for o in opts]


def _cs_options(character: str, seed: str, ascension: int, entry: str) -> list[tuple[str, bool]]:
    """``(canon_text_key, enabled)`` for a C# enter_room event, in order."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        st = g.enter_room("event", event=entry)
        if st.get("type") == "error":
            raise LookupError(f"csharp: enter_room {entry!r}: {st.get('message')}")
        opts = st.get("options") or []
        # C# `is_locked == True` ⇒ option unavailable ⇒ Rust `enabled == False`.
        return [
            (_canon(o.get("text_key") or o.get("title")), not o.get("is_locked", False))
            for o in opts
        ]
    finally:
        g.close()


def option_parity(characters: list[str], seed: str, ascension: int) -> int:
    """Force every event on both engines; diff option count + enabled vector.

    Returns the count of HARD failures (count or enabled-vector mismatch).
    option_id string drift is reported advisory-only and never fails.
    """
    hard_fails = id_drift = checked = ctx_advisory = 0
    for character in characters:
        universe = _event_universe(character, seed, ascension)
        for rust_name, cs_entry in universe:
            try:
                rs = _rust_options(character, seed, ascension, rust_name)
                cs = _cs_options(character, seed, ascension, cs_entry)
            except LookupError as e:
                print(f"[{character}] SKIP {rust_name}: {e}")
                continue
            count_ok = len(rs) == len(cs)
            enabled_ok = count_ok and [e for _, e in rs] == [e for _, e in cs]
            ids_ok = count_ok and [i for i, _ in rs] == [i for i, _ in cs]
            if rust_name in CONTEXT_DEPENDENT_EVENTS:
                # Precondition-gated: a bare force-entry shows it in an
                # impossible state, so count/enabled diffs are expected and
                # advisory only (pool_parity still hard-checks its presence).
                if not (count_ok and enabled_ok):
                    ctx_advisory += 1
                    print(f"[{character}] ctx-advisory {rust_name}: rs={rs} cs={cs}")
                continue
            checked += 1
            if not (count_ok and enabled_ok):
                hard_fails += 1
                print(f"[{character}] OPTION DIFF {rust_name} (cs={cs_entry})")
                if not count_ok:
                    print(f"    count cs={len(cs)} rs={len(rs)}")
                print(f"    rs={rs}")
                print(f"    cs={cs}")
            elif not ids_ok:
                id_drift += 1  # cosmetic: short option_id vs loc-key tail
    print(
        f"\noption parity: {checked - hard_fails}/{checked} non-context events match on "
        f"count+enabled ({id_drift} with advisory option_id drift, "
        f"{ctx_advisory} context-dependent advisory) across {len(characters)} char(s)"
    )
    return hard_fails


# ─── effect parity ───────────────────────────────────────────────────────────
#
# Option STRUCTURE (count + enabled) matching is necessary but not sufficient:
# a handler that grants the wrong relic, no-ops a deck mutation, finishes a
# multi-page chain early, or skips a nested combat presents an IDENTICAL menu
# while doing the wrong thing. `option_parity` (above) is blind to all of it —
# it never resolves an option. This sweep DOES: for every event, on a fresh
# forced entry, it resolves EACH option (one clean run per option) on both
# engines and diffs the resulting state (relics, deck, potions, gold, hp, the
# landed phase, and multi-page follow-up options). This is the check that
# would have caught the `sts2-rs-m3n0g` event-side stubs.


def _follow_options(view: dict) -> int:
    """Count of follow-up options still presented after a choice (multi-page)."""
    return len(view.get("options") or [])


def _effect_fields(cs_snap: Snap, cs_view: dict, rs_snap: Snap, rs_view: dict) -> list[str]:
    """All effect divergences for one resolved option (empty == faithful).

    `diff_snaps` covers the carried state (relics / deck / potions / gold / hp /
    landed phase). On top of that, when BOTH engines stay in the event phase we
    compare the follow-up option count — a multi-page chain that diverges
    mid-flow (the finish-vs-continue case is already caught by the `phase`
    diff). Pure (no engine) so it's unit-testable.
    """
    fields = diff_snaps(cs_snap, rs_snap)
    if cs_snap.phase == "event" and rs_snap.phase == "event":
        cf, rf = _follow_options(cs_view), _follow_options(rs_view)
        if cf != rf:
            fields.append(f"followup_options: cs={cf} rs={rf}")
    return fields


def _rust_effect(
    character: str, seed: str, ascension: int, name: str, idx: int | tuple[int, ...]
) -> tuple[Snap, dict]:
    """Force `name`, resolve option `idx` (or each option of a page path),
    return (post-choice Snap, view)."""
    path = (idx,) if isinstance(idx, int) else idx
    r = _rust_backend()
    try:
        # act1="roll": the seed's own act 1, as the C# rolls it. RustGame's
        # default is Overgrowth, which put every Underdocks seed's Rust side on
        # the wrong act (GH #1500).
        r.start(character=character, seed=seed, ascension=ascension, act1="roll")
        entered = r.force_event(name)
        if entered.get("error") or entered.get("type") == "error":
            raise LookupError(f"rust: force_event {name!r}: {entered.get('error')}")
        post = entered
        for i in path:
            post = r.act("choose_option", option_index=i)
            if post.get("error") or post.get("type") == "error":
                raise RuntimeError(f"rust: choose_option {i}: {post.get('error')}")
        if post.get("decision") == "event_choice":
            # The page as the ENGINE has it, locked options included. RustGame's
            # view lists only the takeable options (its `available_actions`),
            # so a locked option the C# shows (EndlessConveyor's "Broke") read
            # as a missing one.
            infos = r.rm.event_option_infos() or []
            post = dict(post)
            post["options"] = [
                {
                    "index": i,
                    "option_id": o.get("option_id"),
                    "is_locked": not o.get("enabled", True),
                }
                for i, o in enumerate(infos)
            ]
        return Snap.of(post, "rs"), post
    finally:
        r.close()


def _cs_effect(
    character: str, seed: str, ascension: int, entry: str, idx: int | tuple[int, ...]
) -> tuple[Snap, dict]:
    """Enter `entry`, resolve option `idx` (or each option of a page path),
    return (post-choice Snap, view)."""
    path = (idx,) if isinstance(idx, int) else idx
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        post = g.enter_room("event", event=entry)
        if post.get("type") == "error":
            raise LookupError(f"csharp: enter_room {entry!r}: {post.get('message')}")
        for i in path:
            post = g.choose_option(i)
            if post.get("type") == "error":
                raise RuntimeError(f"csharp: choose_option {i}: {post.get('message')}")
        return Snap.of(post, "cs"), post
    finally:
        g.close()


def effect_parity(characters: list[str], seed: str, ascension: int) -> int:
    """Resolve every option of every event on both engines; diff the effect.

    Returns the count of HARD failures (any state-delta divergence on a
    resolved option). Context-dependent events are skipped (their bare
    force-entry state is impossible — same rationale as `option_parity`).
    """
    hard_fails = checked = 0
    for character in characters:
        universe = _event_universe(character, seed, ascension)
        for rust_name, cs_entry in universe:
            if rust_name in CONTEXT_DEPENDENT_EVENTS:
                continue
            try:
                rs_opts = _rust_options(character, seed, ascension, rust_name)
                cs_opts = _cs_options(character, seed, ascension, cs_entry)
            except LookupError as e:
                print(f"[{character}] SKIP {rust_name}: {e}")
                continue
            # Resolve by index, so the options must align positionally. The
            # contract for that is exactly `option_parity`'s HARD contract:
            # same count + same enabled vector (option_id STRINGS legitimately
            # differ by namespace — `GRAB` vs `GRABPOTIONS` — which is advisory,
            # NOT a reorder). A count/enabled mismatch is already a hard failure
            # there, so skip the effect resolve rather than double-report.
            if len(rs_opts) != len(cs_opts) or [e for _, e in rs_opts] != [e for _, e in cs_opts]:
                print(
                    f"[{character}] SKIP-EFFECT {rust_name}: option count/enabled differ "
                    "(see option parity)"
                )
                continue
            for idx in range(len(rs_opts)):
                # Only resolve an option enabled (unlocked) on BOTH engines.
                if not (rs_opts[idx][1] and cs_opts[idx][1]):
                    continue
                oid = rs_opts[idx][0] or idx
                try:
                    rs_snap, rs_view = _rust_effect(character, seed, ascension, rust_name, idx)
                    cs_snap, cs_view = _cs_effect(character, seed, ascension, cs_entry, idx)
                except (LookupError, RuntimeError) as e:
                    hard_fails += 1
                    print(f"[{character}] EFFECT RESOLVE-ERROR {rust_name}[{idx}={oid}]: {e}")
                    continue
                checked += 1
                fields = _effect_fields(cs_snap, cs_view, rs_snap, rs_view)
                if fields:
                    hard_fails += 1
                    print(f"[{character}] EFFECT DIFF {rust_name}[{idx}={oid}] (cs={cs_entry})")
                    for f in fields:
                        print(f"    {f}")
    print(
        f"\neffect parity: {checked - hard_fails}/{checked} resolved options match "
        f"across {len(characters)} char(s)"
    )
    return hard_fails


# ─── gated effect sweep (GH #1500) ────────────────────────────────────────────
#
# `effect_parity` above resolves the FIRST page of every event in ONE seed's
# pools. One seed has one act-1 variant, so the event gaps of #1491-#1494 (all
# Underdocks events) sat outside it while it printed "CLEAN". The sweep below
# is what `just event-parity` (gate-full) runs: the union of a pinned seed set
# whose act 1s cover both variants, every follow-up page up to a depth cap, and
# no silent skips — a menu mismatch, a resolve error or a skipped event is a
# failure or an allowlisted row with its reason. It also asserts that the
# seeds' pools reach the build's whole event universe, so the blind spot cannot
# come back by a seed change, and writes one row per (event, page path).

#: Pinned seeds: act 1 is Overgrowth on one and Underdocks on the other, on
#: every gated build (the sweep asserts both variants are present).
GATE_SEEDS: tuple[str, ...] = ("EVTPAR0", "EVTPAR")

#: Page-walk caps. Depth 3 reaches every page of the events this repo has seen
#: diverge mid-chain; the per-event path cap bounds the looping chains
#: (SlipperyBridge's Hold On, AbyssalBaths' Linger). Hitting a cap is reported.
GATE_MAX_DEPTH = 3
GATE_MAX_PATHS = 16

#: Events of the universe that are in no act's event pool, so no seed can
#: offer them as an event room. Each is exercised elsewhere or is not content.
NOT_IN_ACT_POOLS: dict[str, str] = {
    "Neow": "the act-1 Ancient (run_parity's Neow walk)",
    "NeowsBlessing": "engine-only run-start flow; no C# class",
    "DeprecatedEvent": "placeholder for retired ids",
    "DeprecatedAncientEvent": "placeholder for retired ids",
    "Darv": "Ancient (rolled per act, not from the event pool)",
    "Orobas": "Ancient",
    "Pael": "Ancient",
    "Tezcatara": "Ancient",
    "Nonupeipe": "Ancient",
    "Tanx": "Ancient",
    "Vakuu": "Ancient",
    "TheArchitect": "the act-3 ending event, entered after the final boss",
}

#: Known effect divergences, ``(event, page path) -> reason``. A row listed
#: here is reported as ``allowlisted``, not as a failure. Keep it empty unless
#: a diff is understood and deliberately left (an oracle artifact, say).
KNOWN_EFFECT_DIFFS: dict[tuple[str, tuple[int, ...]], str] = {}


@dataclass
class EffectRow:
    """One resolved (or deliberately unresolved) option path of one event."""

    character: str
    seed: str
    event: str
    path: tuple[int, ...]
    option: str
    #: match | diff | error | menu (option count/enabled differ) |
    #: allowlisted | skip:context | skip:locked | truncated
    status: str
    detail: str = ""

    @property
    def key(self) -> str:
        return f"{self.event}[{'/'.join(map(str, self.path))}]"

    @property
    def failed(self) -> bool:
        return self.status in ("diff", "error", "menu")


def characters_for(version: str) -> list[str]:
    """The ported roster of ``version``; The Fool exists from v0.113.0."""
    major_minor = tuple(int(x) for x in version.lstrip("v").split(".")[:2])
    return ALL_CHARACTERS + (["Fool"] if major_minor >= (0, 113) else [])


def _view_options(view: dict) -> list[tuple[str, bool]]:
    """``(canon id, enabled)`` of an event page view, either engine."""
    return [
        (
            _canon(o.get("text_key") or o.get("option_id") or o.get("title")),
            not o.get("is_locked", False) and bool(o.get("enabled", True)),
        )
        for o in view.get("options") or []
    ]


def _menus_agree(rs: list[tuple[str, bool]], cs: list[tuple[str, bool]]) -> bool:
    return len(rs) == len(cs) and [e for _, e in rs] == [e for _, e in cs]


def walk_event(
    character: str,
    seed: str,
    ascension: int,
    rust_name: str,
    cs_entry: str,
    *,
    max_depth: int = GATE_MAX_DEPTH,
    max_paths: int = GATE_MAX_PATHS,
) -> list[EffectRow]:
    """Resolve every option of every page of one event on both engines.

    Breadth-first over page paths, each replayed from a fresh run on both
    engines. A page is only descended into when the step that reached it
    matched on both sides (a diverged state has no common next page).
    """

    def row(path: tuple[int, ...], option: str, status: str, detail: str = "") -> EffectRow:
        if status in ("diff", "menu") and (rust_name, path) in KNOWN_EFFECT_DIFFS:
            detail = f"{KNOWN_EFFECT_DIFFS[rust_name, path]} | {detail}"
            status = "allowlisted"
        return EffectRow(character, seed, rust_name, path, option, status, detail)

    rows: list[EffectRow] = []
    try:
        rs_root = _rust_options(character, seed, ascension, rust_name)
        cs_root = _cs_options(character, seed, ascension, cs_entry)
    except LookupError as e:
        return [row((), "<entry>", "error", str(e))]
    frontier: deque[tuple[tuple[int, ...], list, list]] = deque([((), rs_root, cs_root)])
    resolved = 0
    while frontier:
        prefix, rs_opts, cs_opts = frontier.popleft()
        if not _menus_agree(rs_opts, cs_opts):
            rows.append(row(prefix, "<page>", "menu", f"rs={rs_opts} cs={cs_opts}"))
            continue
        for idx, ((oid, rs_on), (_, cs_on)) in enumerate(zip(rs_opts, cs_opts, strict=True)):
            path = (*prefix, idx)
            if not (rs_on and cs_on):
                rows.append(row(path, oid, "skip:locked"))
                continue
            if resolved >= max_paths:
                rows.append(row(path, oid, "truncated", f"{max_paths}-path cap"))
                continue
            resolved += 1
            try:
                rs_snap, rs_view = _rust_effect(character, seed, ascension, rust_name, path)
                cs_snap, cs_view = _cs_effect(character, seed, ascension, cs_entry, path)
            except (LookupError, RuntimeError) as e:
                rows.append(row(path, oid, "error", str(e)))
                continue
            fields = _effect_fields(cs_snap, cs_view, rs_snap, rs_view)
            if fields:
                rows.append(row(path, oid, "diff", "; ".join(fields)))
                continue
            rows.append(row(path, oid, "match"))
            if cs_snap.phase == "event" and rs_snap.phase == "event":
                if len(path) >= max_depth:
                    rows.append(row(path, "<page>", "truncated", f"depth cap {max_depth}"))
                else:
                    frontier.append((path, _view_options(rs_view), _view_options(cs_view)))
    return rows


def effect_sweep(
    characters: list[str],
    seeds: tuple[str, ...] | list[str],
    ascension: int,
    *,
    version: str,
    max_depth: int = GATE_MAX_DEPTH,
    max_paths: int = GATE_MAX_PATHS,
    log: Any = functools.partial(print, flush=True),
) -> tuple[list[EffectRow], list[str]]:
    """The gated sweep: every event of the seeds' pools, every page.

    Each event is resolved once per character, from the first seed whose pools
    hold it. Returns the rows and the sweep-level problems: an act-1 variant
    the seeds miss, and events of ``version``'s universe no seed offers (and
    ``NOT_IN_ACT_POOLS`` does not explain).
    """
    import content_universe  # lazy: keeps this module light for the unit tests

    problems: list[str] = []
    rows: list[EffectRow] = []
    for character in characters:
        events: dict[str, tuple[str, str, str]] = {}
        act1s: set[str] = set()
        for seed in seeds:
            act1s.add(_act1_variant(character, seed, ascension))
            for rust_name, cs_entry in _event_universe(character, seed, ascension):
                events.setdefault(rust_name, (cs_entry, seed, rust_name))
        if act1s != {"Overgrowth", "Underdocks"}:
            problems.append(
                f"{character}: the seeds' act 1s are {sorted(act1s)}, not both variants"
            )
        unreached = sorted(
            set(content_universe.events(version)) - set(events) - set(NOT_IN_ACT_POOLS)
        )
        if unreached:
            problems.append(f"{character}: no seed offers {unreached}")
        for i, name in enumerate(sorted(events), 1):
            cs_entry, seed, _ = events[name]
            if name in CONTEXT_DEPENDENT_EVENTS:
                rows.append(
                    EffectRow(
                        character, seed, name, (), "<entry>", "skip:context", "IsAllowed-gated"
                    )
                )
                continue
            ev_rows = walk_event(
                character,
                seed,
                ascension,
                name,
                cs_entry,
                max_depth=max_depth,
                max_paths=max_paths,
            )
            rows.extend(ev_rows)
            bad = [r for r in ev_rows if r.failed]
            log(
                f"[{version} {character}] {i}/{len(events)} {name} ({seed}): "
                f"{sum(r.status == 'match' for r in ev_rows)} match, {len(bad)} failed"
            )
            for r in bad:
                log(f"    {r.status.upper()} {r.key} {r.option}: {r.detail}")
    return rows, problems


def _act1_variant(character: str, seed: str, ascension: int) -> str:
    """``"Overgrowth"`` or ``"Underdocks"``: which act 1 ``seed`` rolls, read
    off the act-1 boss (the room set names no act)."""
    from parity_fuzz_scenario import ACT1_UNDERDOCKS_COVERAGE_ENCOUNTERS

    boss = _rust_run_manager(character, seed, ascension).generated_rooms()[0].get("boss")
    return "Underdocks" if boss in ACT1_UNDERDOCKS_COVERAGE_ENCOUNTERS else "Overgrowth"


def coverage_table(rows: list[EffectRow]) -> dict[str, int]:
    """Row counts by status (the sweep's coverage summary)."""
    out: dict[str, int] = {}
    for r in rows:
        out[r.status] = out.get(r.status, 0) + 1
    return dict(sorted(out.items()))


def floor_problems(rows: list[EffectRow], floor: list[str], character: str) -> list[str]:
    """Paths the floor says matched before and that no longer match."""
    now = {r.key for r in rows if r.character == character and r.status == "match"}
    lost = sorted(set(floor) - now)
    return [f"{character}: {len(lost)} path(s) no longer covered: {lost}"] if lost else []


#: Per build and character, every page path that matched at the last
#: `--write-floor`. A path that matched then and does not now fails the gate.
FLOOR_PATH = Path(__file__).resolve().parents[1] / "data" / "parity" / "event_effect_floor.json"


def _sweep_one(character: str, ascension: int, version: str) -> tuple[list[EffectRow], list[str]]:
    """`effect_sweep` for one character (a picklable pool task)."""
    return effect_sweep([character], GATE_SEEDS, ascension, version=version)


def event_coverage_problems(
    rows: list[EffectRow], characters: list[str], version: str
) -> list[str]:
    """Per character, the build's events no row exercised (prints the table).

    An event is EXERCISED when one of its paths matched on both engines, and
    SKIPPED when its only row is an allowlisted skip (`CONTEXT_DEPENDENT_EVENTS`).
    Every other event of the universe, outside `NOT_IN_ACT_POOLS`, fails.
    """
    import content_universe

    universe = set(content_universe.events(version)) - set(NOT_IN_ACT_POOLS)
    out: list[str] = []
    for character in characters:
        mine = [r for r in rows if r.character == character]
        exercised = {r.event for r in mine if r.status == "match"}
        skipped = {r.event for r in mine if r.status == "skip:context"} - exercised
        uncovered = sorted(universe - exercised - skipped)
        print(
            f"  {character:<12} events: {len(exercised & universe)}/{len(universe)} exercised, "
            f"{len(skipped)} skipped (context), {len(uncovered)} uncovered"
            + (f" {uncovered}" if uncovered else "")
        )
        if uncovered:
            out.append(f"{character}: no path of {uncovered} matched on both engines")
    return out


def gate(
    characters: list[str],
    ascension: int,
    *,
    json_path: str | None = None,
    write_floor: bool = False,
) -> int:
    """`--gate`: run `effect_sweep` on the selected build; 0 iff clean."""
    version = sts2_client.GAME_VERSION
    rows: list[EffectRow] = []
    problems: list[str] = []
    floor_all = json.loads(FLOOR_PATH.read_text(encoding="utf-8")) if FLOOR_PATH.exists() else {}
    floor = floor_all.get(version, {})
    # One process per character (each drives its own single-threaded C#
    # children); EVENT_PARITY_JOBS caps how many run at once.
    jobs = max(1, int(os.environ.get("EVENT_PARITY_JOBS", "3")))
    with ProcessPoolExecutor(max_workers=min(jobs, len(characters))) as pool:
        futures = {
            pool.submit(_sweep_one, character, ascension, version): character
            for character in characters
        }
        for fut in as_completed(futures):
            character = futures[fut]
            ch_rows, ch_problems = fut.result()
            rows += ch_rows
            problems += ch_problems
            if not write_floor:
                problems += floor_problems(ch_rows, floor.get(character, []), character)
            if json_path:
                Path(json_path).write_text(
                    json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8"
                )
    rows.sort(key=lambda r: (characters.index(r.character), r.event, r.path))
    failed = [r for r in rows if r.failed]
    print(f"\nevent effect sweep @ {version} ({', '.join(characters)}; seeds {GATE_SEEDS}):")
    for character in characters:
        table = coverage_table([r for r in rows if r.character == character])
        print(f"  {character:<12} " + "  ".join(f"{k}={v}" for k, v in table.items()))
    problems += event_coverage_problems(rows, characters, version)
    skipped = sorted({f"{r.key}:{r.status}" for r in rows if r.status.startswith("skip")})
    if skipped:
        print(f"  not exercised (skipped): {len(skipped)} row(s), e.g. {skipped[:8]}")
    for r in failed:
        print(f"  FAIL {r.character} {r.seed} {r.status} {r.key} {r.option}: {r.detail}")
    for p in problems:
        print(f"  FAIL {p}")
    if write_floor:
        floor_all[version] = {
            c: sorted({r.key for r in rows if r.character == c and r.status == "match"})
            for c in characters
        } | {c: v for c, v in floor.items() if c not in characters}
        FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLOOR_PATH.write_text(
            json.dumps(floor_all, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  floor written: {FLOOR_PATH}")
    bad = len(failed) + len(problems)
    print("\n✓ EVENT EFFECT SWEEP CLEAN" if not bad else f"\n✗ {bad} EVENT EFFECT FAILURE(S)")
    return 0 if not bad else 1


def _utf8_stdio() -> None:
    """Write stdout/stderr as UTF-8.

    Redirected to a file on Windows (as `just event-parity` does), they default
    to cp1252, which has no ``✓``: a clean sweep crashed on its verdict line
    and the gate read it as a failure (GH #1781).
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8")


def main() -> int:
    _utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--all-characters", action="store_true", help="run all 5 characters")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--pool-seeds", type=int, default=50, help="seeds for POOL parity sweep")
    ap.add_argument("--base", default="EVTPAR", help="seed prefix for the pool sweep")
    ap.add_argument("--no-options", action="store_true", help="skip the per-event option check")
    ap.add_argument("--no-pool", action="store_true", help="skip the event-pool sweep")
    ap.add_argument(
        "--effect",
        action="store_true",
        help="ALSO resolve every option and diff the state EFFECT (relics/deck/"
        "potions/gold/hp/phase/multi-page) — catches handler bugs the menu check misses",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="the gated sweep (GH #1500): GATE_SEEDS (both act-1 variants), every page, "
        "no silent skips, the build's event universe reached, the coverage floor held",
    )
    ap.add_argument("--json", help="--gate: write every row here (rewritten per character)")
    ap.add_argument(
        "--write-floor",
        action="store_true",
        help="--gate: record this run's matched paths as the floor",
    )
    args = ap.parse_args()

    from parity_audit import assert_wheel_fresh  # lazy: keeps the module wheel-free to import

    assert_wheel_fresh()
    if args.gate:
        chars = (
            [args.character]
            if "--character" in sys.argv
            else characters_for(sts2_client.GAME_VERSION)
        )
        return gate(chars, args.ascension, json_path=args.json, write_floor=args.write_floor)
    characters = ALL_CHARACTERS if args.all_characters else [args.character]

    fails = 0
    if not args.no_pool:
        fails += pool_parity(characters, args.base, args.ascension, args.pool_seeds)
    if not args.no_options:
        # Option lists on a fresh run are deterministic; one seed per character
        # exercises every event's option structure.
        fails += option_parity(characters, f"{args.base}0", args.ascension)
    if args.effect:
        # Effects on a fresh forced entry are deterministic too; one seed per
        # character resolves every option of every event.
        fails += effect_parity(characters, f"{args.base}0", args.ascension)
    print("\n✓ EVENT PARITY CLEAN" if fails == 0 else f"\n✗ {fails} EVENT PARITY FAILURE(S)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
