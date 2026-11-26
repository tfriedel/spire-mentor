"""Bridge ``combat_action`` (v2) -> Sts2Headless ``combat_play`` decision JSON.

Real-game bridge phase 2: the live game (via the SpireMentor mod) speaks the
wire protocol documented in ``spire-mentor-mod/PROTOCOL.md``; the MCTS
recommender (:mod:`sts2_hint.recommender`) only understands the
Sts2Headless ``combat_play`` decision shape consumed by
:func:`sts2_hint.adapter.translate` (which in turn feeds
``sts2_rs.state_from_dict``). This module bridges the two *without*
touching either -- :func:`to_headless_combat_play` produces a dict in
exactly the shape :mod:`sts2_hint.adapter` already knows how to read
(the same shape ``tests/fixtures/hint_adapter/_helpers.py`` hand-crafts
and Sts2Headless's own ``RunSimulator.cs`` emits for a ``combat_play``
decision), so the existing sts2_hint pipeline is reused completely
unmodified.

Field-mapping notes (verified against ``PROTOCOL.md``, the
``sts2_hint.adapter``/``schema`` source, and the ``sts2-rs`` engine's own
draw-pile semantics -- see the real-game-bridge plan's phase-2 task list):

* **Card ids.** The bridge already sends game-native registry ids in bare
  SCREAMING_SNAKE form (e.g. ``"STRIKE_IRONCLAD"``, ``"BASH"``) with no
  ``/`` or ``CARD.`` prefix -- confirmed these match
  ``sts2_rs.list_registry_names("card")`` directly, so
  :mod:`sts2_hint.adapter`'s bare-id resolution path (``_resolve_card_id``)
  accepts them verbatim. No re-mapping needed.
* **Ids, not names (GH #1423).** Every ``name`` the bridge sends is the
  game's localized title, so on a Chinese game it is ``"闪电"``, not
  ``"Lightning"``. Relics, monsters, orbs, potions and powers are therefore
  resolved by their model ``id`` and the title is only the fallback.
  Relic ids (``"BURNING_BLOOD"``) and potion ids match the ``sts2_rs``
  registry directly. Monster ids (``"SHRINKER_BEETLE"``) are matched to the
  registry's CamelCase kinds (``"ShrinkerBeetle"``) ignoring underscores and
  case (``sts2_hint.adapter._monster_kind_from_wire``). Orbs: see the orb
  stem in ``to_headless_combat_play``. Powers: see the next bullet. Both
  ``id`` and ``name`` are forwarded.
* **Powers.** The bridge's power ``id`` is the C# ``Id.Entry``, which
  keeps a ``_POWER`` suffix the Rust registry strips for most powers
  (``VULNERABLE_POWER`` vs the engine's ``VULNERABLE``) -- exactly the
  case ``sts2_hint.adapter._translate_power`` already falls back from id
  to a title lookup for. Both ``id`` and ``name`` are forwarded verbatim
  and the existing fallback handles the rest.
* **Draw-pile order.** The bridge documents "index 0 is the next card
  drawn" (verified against the v0.107.1 decomp); the ``sts2-rs`` engine
  itself pops from the front of ``state.player.draw`` (``draw.remove(0)``
  in ``combat_loop.rs``) and ``state_dict.rs`` preserves wire order
  1:1 into that vec -- so the bridge's draw/discard/exhaust pile order is
  passed straight through with no reversal.
* **Target/enemy indexing.** ``sts2_hint.display.format_target`` resolves
  an MCTS action's ``target_index`` by matching it against each enemy's
  wire ``index`` field -- i.e. ``target_index`` is the enemy's *position*
  in the ``enemies`` array, exactly the same 0-based addressing the
  bridge's own ``play``/``potion`` ``target_index`` uses. So an MCTS
  recommendation's ``target_index`` can be sent back to the bridge
  unchanged.
* **Hand ``stable_idx``.** Sts2Headless's ``card.index`` on a hand card
  becomes ``CardSpec.stable_idx``, which the recommender maps back via
  ``hand[card_index]["stable_idx"]`` where ``card_index`` is the
  *position* in the translated hand list (0-based) -- i.e. positional in
  our own emitted list, matching the bridge's own ``play`` verb
  addressing (``card_index`` always indexes ``hand[card_index]``). We set
  each hand card's ``index`` to its position so ``stable_idx`` round-trips
  as exactly the bridge ``card_index`` to send back.
* **``unplayable`` keyword.** Only set when the bridge's
  ``unplayable_reason`` mentions ``HasUnplayableKeyword`` -- NOT whenever
  ``playable`` is false. The bridge's ``playable`` flag is dynamic (energy
  availability included); naively mapping "not playable right now" to the
  wire's static ``unplayable`` ``CardFlag`` would permanently lock out a
  card that's merely too expensive at this instant, corrupting every
  future-turn rollout the MCTS simulates.
* **``master_deck``.** Bridge sends the full deck at
  ``run_state.master_deck``, and ``sts2_hint.schema.DeckSpec`` documents an
  optional ``master_deck`` slot. We translate it and attach it as a
  top-level ``"master_deck"`` key (this module's own output shape has no
  nested ``deck`` object -- see the module docstring's overview), which
  ``sts2_hint.adapter.translate()`` reads via ``state.get("master_deck")``
  and threads into ``wire["deck"]["master_deck"]`` for
  ``sts2_rs.state_from_dict``.

Known gaps -- fields the headless ``combat_play`` schema can carry that
the bridge does not (yet) send:

* ``context.room_type`` / elite flag: forwarded from a top-level
  ``room_type`` when the bridge sends one (SimCombatHandler emits
  ``RunState.CurrentRoom.RoomType`` since the 2026-08-02 dialect audit;
  deploy-pending); otherwise defaults to ``"Combat"`` (non-elite).
* ``stars`` (Regent mechanic): forwarded from ``player.stars`` when the
  bridge sends it (same deploy-pending mod build); omitted before that.
* Card ``stats`` / ``enchantment`` / ``enchantment_amount``: forwarded
  when present (same deploy-pending mod build -- see ``_translate_card``);
  absent on older mod builds, which leaves the engine's registry/upgrade
  fallbacks in charge exactly as before.

Formerly-listed gaps, now closed:

* ``play_pile``, power ``display_amount`` and power ``vars`` **are now
  forwarded**. All three were added to the headless (``Sts2Headless``) wire by
  the v11/v12 mirror-fidelity fixes and the bridge was left behind on them, so
  the live arm ran the same mirror against a wire that still withheld the card
  mid-play and the CrimsonMantle/WitheringPresence counters. The bridge emits
  them from ``SimCombatHandler.BuildCombatState`` /
  ``SimStateExtractor.SerializePower``; this module passes them through.
* The layered cost fields (``base_cost`` / ``persistent_cost`` /
  ``played_cost`` / ``stable_cost``) **are now forwarded** (see
  ``_translate_card``). The bridge has emitted them since the v15/v20
  mirror-fidelity catch-ups (``SimStateExtractor.SerializeCard``), but this
  module dropped them, so the live reconstruction ran cost_override-only:
  the engine baked transient discounts in permanently and lost every
  turn-boundary/when-played cost reset -- the dominant pile-desync class of
  the first persistent-mirror live sweep.
* The RNG lineage block (``rng.run_seed`` + ``rng.counters``) **is now
  forwarded** (2026-08-02 dialect audit). ``SimStateExtractor.ApplyEnvelope``
  has published it on every ``combat_action`` (and on the ``card_select``
  embedded board) since the mirror's restore path landed, but this module
  dropped it -- so on the live arm ``sts2_hint.adapter._rng_block`` never saw
  a ``run_seed``, ``CombatMirror.reconcile`` never let the world own the
  dice, and every live shuffle/intent prediction was structurally a guess
  (verified: no ``mirror_live2.jsonl`` world_wire carries ``run_seed``).
  Forwarded ALL-OR-NOTHING, mirroring the producer/consumer contract: a
  partial block (seed without counters) is worse than none.
* ``player.gold`` **is now forwarded** (same audit) from
  ``run_state.gold``. The gold-gated relic hooks (SealOfGold's 5-gold
  turn-start energy buy) rebuilt against the engine default of no gold on
  the live arm only -- the exact class the v27wr sweep pinned on the DLL arm
  (90 of 178 ``$.player.energy`` divergences) fixed there but not here.
* ``EnemySpec.move_state`` **is now forwarded** (see ``_translate_enemy``).
  The bridge's telegraphed ``next_move_id`` is the monster's current
  ``MonsterState`` id, and those id strings match sts2-rs's enemy-AI state
  names 1:1, so ``sts2_rs.state_from_dict`` reconstructs the real move-state
  machine (``force_current_state_by_name``) and rollouts replay the true
  moveset -- fixing the prior overconfidence on scaling bosses (e.g. the Kin
  Priest, whose ``RITUAL_MOVE`` stacks Strength every cycle) where the
  heuristic-AI fallback made MCTS score a losing fight as a near-certain win.
"""

from __future__ import annotations

import logging
from typing import Any

from sts2_client import LEGACY_GAME_VERSION, SUPPORTED_GAME_VERSIONS, normalize_game_version
from sts2_hint.canonical_maps import ORB_TITLE_TO_ID, id_key

logger = logging.getLogger(__name__)


class UnsupportedGameVersionError(ValueError):
    """The live game reports a build this advisor has no engine for (GH #800).

    ``str(exc)`` is written for the player: the mod renders it in the HUD as
    the advice note, so it names the build and what to do about it.
    """

    def __init__(self, raw: object) -> None:
        self.raw = raw
        super().__init__(
            f"Spire Mentor's advisor does not support game version {raw!r} "
            f"(supported: {', '.join(SUPPORTED_GAME_VERSIONS)}). Update the advisor, "
            "or switch the game to a supported build in Steam (Properties > Betas)."
        )


def envelope_game_version(msg: dict[str, Any]) -> str:
    """The game build a bridge message was produced by, normalized (``"v0.111.0"``).

    Read from the envelope's ``game_version`` (``SimStateExtractor.ApplyEnvelope``,
    PROTOCOL.md section 1). Absent, ``None`` or blank -- every mod build before
    the field -- means :data:`sts2_client.LEGACY_GAME_VERSION` (v0.107.1), so an
    old mod gets exactly the engine it always got. A hotfix of a supported
    build (``v0.113.1``) is that build (:func:`sts2_client.hotfix_base`).

    Raises:
        UnsupportedGameVersionError: a build outside
            :data:`sts2_client.SUPPORTED_GAME_VERSIONS`, or a non-string value.
    """
    raw = msg.get("game_version")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return LEGACY_GAME_VERSION
    if not isinstance(raw, str):
        raise UnsupportedGameVersionError(raw)
    try:
        version = normalize_game_version(raw)
    except ValueError:
        raise UnsupportedGameVersionError(raw) from None
    if raw.strip().lstrip("vV") != version.lstrip("v") and raw not in _HOTFIXES_LOGGED:
        _HOTFIXES_LOGGED.add(raw)
        logger.info("game build %s plays as %s; advising with the %s engine", raw, version, version)
    return version


#: Hotfix builds already logged by :func:`envelope_game_version` (once each).
_HOTFIXES_LOGGED: set[str] = set()


#: Bridge ``run_state.character`` (SCREAMING_CASE) -> the Title-Case
#: literal :func:`sts2_hint.adapter.translate` accepts as its
#: ``character=`` kwarg. Pass the result as ``RecommendConfig.character``.
CHARACTER_ID_TO_DISPLAY: dict[str, str] = {
    "IRONCLAD": "Ironclad",
    "SILENT": "Silent",
    "DEFECT": "Defect",
    "NECROBINDER": "Necrobinder",
    "REGENT": "Regent",
    "FOOL": "Fool",
}


class UnsupportedCharacterError(ValueError):
    """A board with a character the reported game build has no rules for
    (v0.113.0's The Fool on an older build). ``str(exc)`` is the
    player-facing HUD note; ``raw`` the id."""

    def __init__(self, raw: str, build: str) -> None:
        self.raw = raw
        name, first = _BUILD_GATED_CHARACTERS[raw]
        super().__init__(
            f"{name} needs game version {first} or newer; this game reports {build}, so there "
            "is no advice for this run."
        )


#: Characters that exist only from a game build on: id -> (display name,
#: first build). Rust twin: `sts2_advisor::serve::BUILD_GATED_CHARACTERS`.
_BUILD_GATED_CHARACTERS = {"FOOL": ("The Fool", "v0.113.0")}


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.lstrip("vV").split("."))


def check_supported_characters(msg: dict[str, Any]) -> None:
    """Refuse a board whose own character or any ally's does not exist on the
    game build the board reports (`_BUILD_GATED_CHARACTERS`): the engine has
    no rules for The Fool before v0.113.0.

    Any other id (a modded or renamed character) passes, as do absent / blank
    ids: the adapter then infers the character from card ids. Call it after
    :func:`envelope_game_version`, which refuses an unknown build first.

    Raises:
        UnsupportedCharacterError: the first character the build lacks.
    """
    build = envelope_game_version(msg)
    run_state = msg.get("run_state")
    ids = [run_state.get("character") if isinstance(run_state, dict) else None]
    ids += [a.get("character") for a in (msg.get("allies") or []) if isinstance(a, dict)]
    for cid in ids:
        if not isinstance(cid, str):
            continue
        gated = _BUILD_GATED_CHARACTERS.get(cid.strip().upper())
        if gated is not None and _version_key(build) < _version_key(gated[1]):
            raise UnsupportedCharacterError(cid.strip().upper(), build)


def character_display_name(character_id: str | None) -> str | None:
    """Map a bridge ``run_state.character`` id to the adapter's Title-Case form.

    Args:
        character_id: e.g. ``"IRONCLAD"``, or ``None`` if no run/character
            is known yet.

    Returns:
        e.g. ``"Ironclad"``, or ``None`` if unmapped/absent (the caller
        should then rely on :mod:`sts2_hint.adapter`'s own
        card-id-prefix inference).
    """
    if not character_id:
        return None
    return CHARACTER_ID_TO_DISPLAY.get(character_id.upper())


def run_ascension(run_state: dict[str, Any], default: int = 0) -> int:
    """The ascension a live board must be rebuilt at.

    The mod reports the run's level as ``run_state.ascension``
    (``SimStateExtractor.cs``). A combat lane that rebuilds the board without
    it plans every enemy turn after the current telegraph against A0 tables
    (GH #848).

    Args:
        run_state: The bridge state's ``run_state`` block (may be empty).
        default: Used only when the mod sent no level, e.g. the
            ``RecommendConfig.ascension`` the lane was built with.

    Returns:
        The run's ascension level.
    """
    value = run_state.get("ascension")
    return default if value is None else int(value)


def to_headless_combat_play(state: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2 bridge ``combat_action`` snapshot to a headless ``combat_play`` dict.

    Args:
        state: The ``state`` object of a bridge message with
            ``state["type"] == "combat_action"`` (see ``PROTOCOL.md``
            section 3, "combat_action") -- e.g. the ``state`` field of one
            JSONL line logged by :mod:`realgame_random`/:mod:`realgame_mcts`.

    Returns:
        A dict shaped exactly like Sts2Headless's own ``combat_play``
        decision JSON (``{"type": "decision", "decision": "combat_play",
        ...}``), ready for :func:`sts2_hint.adapter.translate` (called
        downstream by :func:`sts2_hint.recommender.recommend`/``recommend_all``).

    In Rust (``sts2_rs.to_headless_combat_play``, GH #1426). The engine no
    longer reads this dialect (:func:`combat_wire` builds the fight in one
    pass); the advisor still does, to label its regret rows and recognize a
    combat card select.

    Raises:
        ValueError: Not a ``combat_action`` board.
        UnsupportedGameVersionError: A build outside the supported set.
    """
    import sts2_rs

    if state.get("type") == "combat_action":
        # The build's refusal as its own error type, and the hotfix log line.
        envelope_game_version(state)
    return sts2_rs.to_headless_combat_play(state)


def combat_wire(
    state: dict[str, Any],
    *,
    character: str | None,
    ascension: int,
    combat_id: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """A bridge ``combat_action`` board -> the engine wire dict, in Rust.

    ``sts2_rs.combat_wire_from_bridge`` (``sts2-workbench``'s
    ``combat_import``, which the browser workbench runs too) does in one pass
    what :func:`to_headless_combat_play` followed by
    ``sts2_hint.adapter.translate`` does in two. The two are kept identical by
    ``tests/test_board_import_parity.py``. GH #1426 step 1.

    ``character`` is the display name (``"Ironclad"``). ``None`` infers it
    from the card ids, as ``translate`` does (a lane whose run state names no
    character: the mirrored card select).

    Raises:
        UnsupportedGameVersionError: a build the engine has no rules for,
            checked first so the advisor answers it with its refusal.
        ValueError: a board the engine cannot model
            (``sts2_rs.UnsupportedStateError``), or no character to build it for.
    """
    import sts2_rs

    envelope_game_version(state)
    wire, warnings = sts2_rs.combat_wire_from_bridge(
        state, character, ascension, combat_id=combat_id, seed=seed
    )
    for warning in warnings:
        logger.info("adapter: %s", warning)
    return wire


#: Epochs the LIVE save appears not to have revealed yet (GH #367).
#:
#: **FALLBACK ONLY as of GH #374.** SpireMentor now puts the save's real
#: ``UnlockState.ToSerializable().UnlockedEpochs`` on the envelope, and
#: :func:`install_unlock_state_from_wire` supersedes this constant on the first
#: message that carries it — logging a warning when the two disagree, which is
#: the signal that this census has gone stale. This list still covers the gap
#: before the first message arrives, and any mod build predating the field.
#:
#: The engine had no access to the save's ``UnlockState``, so this was inferred
#: from the world wires themselves: ``python -m diag_epoch_census`` counts how
#: often each epoch-gated card is named anywhere in a corpus, and an epoch is
#: all-or-nothing, so a clean 0-of-3 against siblings that appear freely is the
#: signature of an unrevealed epoch.
#:
#: Provenance — the 2026-08-03 A1 corpora (1082 records; Defect 545 / Regent
#: 223 / Necrobinder 186 / Silent 128):
#:
#: * ``REGENT5``  — 0/3, vs Regent2 at 276/84/225 and Regent7 at 155/44/3.
#:   Strongest of the set, and the one with an independent confirmation: it
#:   holds exactly one Attack, which is the +1 ``combat_card_generation`` drift
#:   measured on 37/37 CROSSBOW end-turn records.
#: * ``NECROBINDER5`` / ``NECROBINDER7`` — 0/3 each, vs Necrobinder2 at
#:   330/99/21.
#: * ``SILENT5`` / ``SILENT7`` — 0/3 each, vs Silent2 at 110/3/42. WEAKEST of
#:   the set: only 128 Silent records back it.
#:
#: Defect is the control: all three of its epochs are sighted, so the census
#: can distinguish revealed from unrevealed rather than just reporting silence.
#: No colorless epoch is locked — each has at least one sighting.
#:
#: Absence is weaker evidence than presence, so this is a measurable hypothesis
#: rather than a known fact: if it is wrong for a character, the next live sweep
#: shows a NEW cursor drift for that character instead of the old one.
LIVE_LOCKED_EPOCHS: tuple[str, ...] = (
    "REGENT5_EPOCH",
    "NECROBINDER5_EPOCH",
    "NECROBINDER7_EPOCH",
    "SILENT5_EPOCH",
    "SILENT7_EPOCH",
)


def install_live_unlock_state(override: str | None = None) -> list[str]:
    """Install the live arm's epoch lock into the engine. GH #367.

    C# prunes cards behind unrevealed epochs out of every card pool
    (``CardPoolModel.GetUnlockedCards``), which the engine did not model at all
    — so its generation pools ran one epoch too large and its
    ``combat_card_generation`` cursor drifted by exactly that difference.

    ONLY the live arm should call this. The DLL/headless oracle builds
    ``UnlockState.all``, so locking anything there would manufacture the very
    drift this closes, in the opposite direction.

    ``override`` (or ``$STS2_LIVE_LOCKED_EPOCHS``) replaces
    :data:`LIVE_LOCKED_EPOCHS` with a comma-separated epoch-id list; the empty
    string disables locking entirely, which is the escape hatch if the
    inference turns out wrong for a character. Returns the epoch ids installed.
    """
    import os

    import sts2_rs

    raw = override if override is not None else os.environ.get("STS2_LIVE_LOCKED_EPOCHS")
    if raw is None:
        epochs = list(LIVE_LOCKED_EPOCHS)
    else:
        epochs = [e.strip().upper() for e in raw.split(",") if e.strip()]
    # Raises on an unknown id rather than silently no-op'ing — a typo that
    # quietly disabled the filter would present as an unexplained residual
    # drift, which is the exact bug class this closes.
    locked = sts2_rs.set_locked_epochs(epochs)
    logger.info(
        "unlock state: %d epoch(s) locked, %d card(s) pruned from every pool: %s",
        len(epochs),
        locked,
        ", ".join(epochs) or "(none)",
    )
    return epochs


#: Whether a wire-carried unlock state has already been installed this process.
#:
#: The wire repeats ``unlocked_epochs`` on every run-level message, but the
#: install is a process-global mutation and the log line is worth exactly once.
#: A latch also means the census fallback is only ever *replaced*, never
#: flip-flopped with, if some later message serializes the field as null.
_WIRE_UNLOCKS_INSTALLED: list[str] | None = None


def locked_epochs_from_wire(run_state: dict) -> list[str] | None:
    """Invert the wire's REVEALED epoch list into the locked set. GH #374.

    SpireMentor emits ``run_state.unlocked_epochs`` =
    ``UnlockState.ToSerializable().UnlockedEpochs`` — what the live save has
    revealed. The engine wants the complement, over the epochs *it* models;
    C# knows many more (Relic1-5, Silent1/3/4/6, Underdocks, Darv, …), so the
    coverage list comes from ``sts2_rs.known_epochs()`` rather than a second
    copy of the table here.

    Returns ``None`` when the producer did not carry the field — an OLD MOD, or
    a serialization failure. That is deliberately distinct from ``[]``, which
    is a save that has revealed everything. Conflating them would silently
    unlock all 60 gated cards on a mod that predates this field, reintroducing
    #367's drift with no signal.
    """
    import sts2_rs

    revealed = run_state.get("unlocked_epochs")
    if revealed is None:
        return None
    seen = {str(e).strip().upper() for e in revealed}
    return [e for e in sts2_rs.known_epochs() if e.upper() not in seen]


def install_unlock_state_from_wire(run_state: dict) -> list[str] | None:
    """Install the live save's real unlock state off the wire. GH #374.

    Idempotent and cheap: latches after the first message that carries the
    field, so callers can invoke it per-message without thinking about it.

    Returns the locked epoch ids installed, or ``None`` if the wire did not
    carry an unlock state (in which case whatever
    :func:`install_live_unlock_state` put in at startup stays).

    ``$STS2_LIVE_LOCKED_EPOCHS`` still wins. If someone pinned the lock set
    explicitly they are almost certainly bisecting a drift, and having the wire
    quietly overwrite it mid-session would make that unfalsifiable.
    """
    global _WIRE_UNLOCKS_INSTALLED

    import os

    import sts2_rs

    if _WIRE_UNLOCKS_INSTALLED is not None:
        return _WIRE_UNLOCKS_INSTALLED
    if os.environ.get("STS2_LIVE_LOCKED_EPOCHS") is not None:
        return None

    locked = locked_epochs_from_wire(run_state)
    if locked is None:
        return None

    count = sts2_rs.set_locked_epochs(locked)
    _WIRE_UNLOCKS_INSTALLED = locked

    # The census was a snapshot of one save at one moment, and going stale is
    # its whole failure mode — so say so loudly when the wire disagrees. This
    # line is the signal #374 exists to create: without it, a revealed epoch
    # presents only as an unexplained +1 combat_card_generation drift found by
    # mining a fresh corpus.
    inferred = set(LIVE_LOCKED_EPOCHS)
    if set(locked) != inferred:
        logger.warning(
            "unlock state: wire DISAGREES with the hardcoded census. "
            "wire-locked=[%s] census-locked=[%s]; newly revealed=[%s], "
            "newly locked=[%s]. The census constant LIVE_LOCKED_EPOCHS is "
            "stale — the wire is authoritative and has been installed.",
            ", ".join(sorted(locked)) or "(none)",
            ", ".join(sorted(inferred)) or "(none)",
            ", ".join(sorted(inferred - set(locked))) or "(none)",
            ", ".join(sorted(set(locked) - inferred)) or "(none)",
        )
    logger.info(
        "unlock state (from wire): %d epoch(s) locked, %d card(s) pruned: %s",
        len(locked),
        count,
        ", ".join(sorted(locked)) or "(none)",
    )
    return locked


def _reset_wire_unlock_latch() -> None:
    """Test hook: clear the once-per-process latch."""
    global _WIRE_UNLOCKS_INSTALLED

    _WIRE_UNLOCKS_INSTALLED = None


__all__ = [
    "CHARACTER_ID_TO_DISPLAY",
    "LIVE_LOCKED_EPOCHS",
    "UnsupportedCharacterError",
    "character_display_name",
    "check_supported_characters",
    "install_live_unlock_state",
    "install_unlock_state_from_wire",
    "locked_epochs_from_wire",
    "to_headless_combat_play",
]
