#!/usr/bin/env python3
"""Generate the workbench's sfx_map.json from the C# decompilation.

The decomp names every trigger at its call site — `SfxCmd.Play("event:/…")`,
`.WithAttackerFx(null, "event:/…")` — so the mapping from "what just happened"
to "which sound" is derived rather than hand-written. Regenerate per game
version; a missing entry degrades to the generic hit sound in the web, it never
throws.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

_MOVE_STATE = re.compile(
    r'new MoveState\(\s*"(?P<name>[A-Z0-9_]+)"\s*,\s*(?P<method>\w+)\s*,\s*'
    r"new (?P<intent>\w+)Intent\((?P<args>[^)]*)\)"
)
_METHOD = re.compile(r"private async Task (?P<method>\w+)\(")
_ONESHOT = re.compile(r'(?:SfxCmd\.Play|WithAttackerFx)\([^)]*?"(?P<event>event:/[^"]+)"')
_LOOP = re.compile(r'SfxCmd\.PlayLoop\([^)]*?"(?P<event>event:/[^"]+)"')

# --- Intent kind: derived from the decomp, never hand-written -------------
# `new MoveState(name, method, new XIntent(args))` — _MOVE_STATE captures
# `X` as a prefix (e.g. "MultiAttack" for `MultiAttackIntent`). The engine
# reports `intent.IntentType.ToString()` verbatim (RunSimulator.cs, see
# task-4-report fix round 2), so the prefix -> enum-name mapping must match
# `MegaCrit.Sts2.Core.MonsterMoves.Intents/IntentType.cs`'s real values
# exactly, INCLUDING which classes actually inherit from AttackIntent (only
# those get damage/hits at all -- RunSimulator only serializes them `if
# (intent is AttackIntent atk)`). A hand-written table here silently drifted
# (DeathBlowIntent shipped as "Attack", StatusIntent had no entry at all and
# fell back to the raw "Status" prefix instead of "StatusCard") -- so this is
# parsed from `MegaCrit.Sts2.Core.MonsterMoves.Intents/*.cs` at call time.
#   Anchored to a real declaration line (an access modifier immediately before
#   `class`), for the same reason `_POWER_CLASS_DECL` below is: a bare
#   `r"class (\w+)"` also matches XML-doc prose ("/// This class represents an
#   attack...") when it precedes the declaration, and `.search()` takes the
#   FIRST match. That exact pattern silently mis-keyed three power base classes
#   under "represents"; no Intents file has such prose today, so this is a
#   dormant footgun rather than a live bug -- but it fails the same silent way
#   (intent kinds stop resolving, cues fall back, every test stays green).
_INTENT_CLASS_DECL = re.compile(
    r"(?:^|\n)\s*(?:public|internal)\s+(?:sealed\s+|abstract\s+|static\s+|partial\s+)*"
    r"class\s+(?P<name>\w+)(?:\s*:\s*(?P<base>\w+))?"
)
_INTENT_TYPE_LITERAL = re.compile(r"IntentType\.(?P<value>[A-Za-z]+)")


def load_intent_tables(decomp_root: Path) -> tuple[dict[str, str], frozenset[str]]:
    """Derive {prefix: EnumName} and the attack-family prefix set from
    `MegaCrit.Sts2.Core.MonsterMoves.Intents/*.cs`, by walking each class's
    real inheritance chain to whichever ancestor overrides `IntentType`
    (`SingleAttackIntent`/`MultiAttackIntent` declare no `IntentType` of
    their own -- both inherit `AttackIntent`'s -- while `DeathBlowIntent`
    overrides it directly to `DeathBlow` despite extending
    `SingleAttackIntent`). A class with a runtime-conditional value
    (`DebuffIntent`: `Debuff` or `DebuffStrong` depending on a ctor bool)
    resolves to its FIRST `IntentType.<X>` literal in source order -- that's
    `Debuff` here; there is no static way to know which ctor argument any
    given call site passes, so this is a documented default, not a guess.
    """
    intents_dir = decomp_root / "MegaCrit.Sts2.Core.MonsterMoves.Intents"
    class_base: dict[str, str] = {}
    class_enum: dict[str, str] = {}
    for cs in sorted(intents_dir.glob("*.cs")):
        text = cs.read_text(encoding="utf-8", errors="replace")
        decl = _INTENT_CLASS_DECL.search(text)
        if not decl or not decl["name"].endswith("Intent"):
            continue
        if decl["base"]:
            class_base[decl["name"]] = decl["base"]
        value = _INTENT_TYPE_LITERAL.search(text)
        if value:
            class_enum[decl["name"]] = value["value"]

    def resolve_kind(name: str | None) -> str | None:
        seen: set[str] = set()
        while name and name not in seen:
            seen.add(name)
            if name in class_enum:
                return class_enum[name]
            name = class_base.get(name)
        return None

    def is_attack_family(name: str | None) -> bool:
        seen: set[str] = set()
        while name and name not in seen:
            seen.add(name)
            if name == "AttackIntent":
                return True
            name = class_base.get(name)
        return False

    kind_by_prefix: dict[str, str] = {}
    attack_prefixes: set[str] = set()
    for class_name in class_base.keys() | class_enum.keys():
        prefix = class_name[: -len("Intent")]
        kind = resolve_kind(class_name)
        if kind:
            kind_by_prefix[prefix] = kind
        if is_attack_family(class_name):
            attack_prefixes.add(prefix)
    return kind_by_prefix, frozenset(attack_prefixes)


# Convenience default for callers with no decomp_root handy (unit tests):
# lazily derives the SAME way, from the real committed decomp, rather than
# duplicating a hand-typed copy of the table above. build_map always passes
# its own freshly-loaded tables explicitly instead of relying on this.
_REPO_DECOMP_ROOT = Path(__file__).resolve().parents[1] / "decompiled" / "v0.107.1"


@lru_cache(maxsize=1)
def _default_intent_tables() -> tuple[dict[str, str], frozenset[str]]:
    return load_intent_tables(_REPO_DECOMP_ROOT)


# SingleAttackIntent.Repeats => 1 (a literal, not computed); DeathBlowIntent
# inherits it unchanged (no override) rather than computing repeats from a
# ctor arg the way MultiAttackIntent does. Both verified directly against
# MegaCrit.Sts2.Core.MonsterMoves.Intents/{SingleAttackIntent,
# DeathBlowIntent}.cs -- this is reading a fixed structural fact about two
# specific classes, not guessing a per-monster game-balance number.
_FIXED_HITS = {"SingleAttack": 1, "DeathBlow": 1}

_INT_MEMBER_DECL_TMPL = (
    r"(?:private|protected|public|internal)\s+(?:static\s+)?(?:readonly\s+)?"
    r"(?:const\s+)?int\s+%s\b\s*(?:=>|\{|=)"
)
# AscensionHelper.GetValueIfAscension(level, ascensionValue, fallbackValue) --
# verified against MegaCrit.Sts2.Core.Helpers/AscensionHelper.cs. The THIRD
# (fallback) argument is the non-ascension/base-difficulty value; that's what
# a static, ascension-blind map should show.
_ASCENSION_FALLBACK = re.compile(
    r"AscensionHelper\.GetValueIfAscension\(\s*[^,()]+,\s*-?\d+\s*,\s*(-?\d+)\s*\)"
)
_INT_LITERAL_IMMEDIATE = re.compile(r"^\s*(-?\d+)\s*;")
_INT_LITERAL_RETURN = re.compile(r"return\s+(-?\d+)\s*;")


def _resolve_int_member(name: str, source: str) -> int | None:
    """Resolve a bare identifier used as a damage/hit-count constructor
    argument to its int value, when this file declares one -- a plain
    literal property/const, or an `AscensionHelper.GetValueIfAscension`
    call, from which the fallback value is taken (see module comment
    above `_ASCENSION_FALLBACK`). Returns None when unresolvable (e.g. the
    identifier names a `Func<...>` delegate, not an `int`) -- the caller
    omits the field rather than shipping a wrong number.
    """
    decl = re.search(_INT_MEMBER_DECL_TMPL % re.escape(name), source)
    if not decl:
        return None
    # Stop at this member's own closing ';' — a fixed window would run past
    # a short property (e.g. "=> 2;") straight into the NEXT member's body,
    # and a later AscensionHelper call could then be mistaken for this one's.
    end = source.find(";", decl.end())
    window = source[decl.end() : end + 1] if end != -1 else source[decl.end() : decl.end() + 300]
    m = _ASCENSION_FALLBACK.search(window)
    if m:
        return int(m.group(1))
    m = _INT_LITERAL_IMMEDIATE.match(window) or _INT_LITERAL_RETURN.search(window)
    return int(m.group(1)) if m else None


def _resolve_int_arg(arg: str, source: str) -> int | None:
    if re.fullmatch(r"-?\d+", arg):
        return int(arg)
    return _resolve_int_member(arg, source)


def parse_move_states(
    source: str,
    intent_kind: dict[str, str] | None = None,
    attack_family: frozenset[str] | set[str] | None = None,
) -> list[dict]:
    """Extract each MoveState's name, handler method and intent shape.

    `hits` and `damage` are included ONLY for attack-family intents (mirrors
    the real engine, which only ever serializes them `if (intent is
    AttackIntent atk)` -- RunSimulator.cs) and ONLY when resolvable to a real
    int. An unresolvable damage/hits argument (a `Func<...>` delegate, a
    dynamic method call) is omitted rather than defaulting to a number --
    absent lets the web skip hit-matching cleanly; a wrong number matches the
    wrong sound.
    """
    if intent_kind is None or attack_family is None:
        intent_kind, attack_family = _default_intent_tables()
    moves: list[dict] = []
    for m in _MOVE_STATE.finditer(source):
        prefix = m["intent"]
        intent = intent_kind.get(prefix, prefix)
        args = [a.strip() for a in m["args"].split(",") if a.strip()]
        move: dict = {"name": m["name"], "method": m["method"], "intent": intent}
        if prefix in attack_family:
            damage = _resolve_int_arg(args[0], source) if args else None
            if damage is not None:
                move["damage"] = damage
            hits = _FIXED_HITS.get(prefix)
            if hits is None and prefix == "MultiAttack" and len(args) >= 2:
                hits = _resolve_int_arg(args[1], source)
            if hits is not None:
                move["hits"] = hits
        moves.append(move)
    return moves


# Many moves don't pass an inline "event:/…" literal to SfxCmd.Play/
# WithAttackerFx at all — they pass a bare C# property/const name instead
# (`WithAttackerFx(null, AttackSfx)`, `SfxCmd.Play(FleeSfx)`). Two shapes of
# that, both real in the decomp:
#   - a per-monster property/const defined in the SAME file, holding its own
#     literal (ThievingHopper.FleeSfx returns one of two literals itself);
#   - no override at all, meaning the MonsterModel base class's convention
#     property applies: `AttackSfx`/`CastSfx`/`DeathSfx` string-interpolate
#     `event:/sfx/enemy/enemy_attacks/{slug}/{slug}_{attack|cast|die}` from
#     the monster's own id.
# Resolving the second shape requires guessing the id's snake_case spelling
# from the class name, which is not always right (id and class name diverge
# sometimes) — but every guess still runs through `_resolve()`'s audio-root
# check in build_map, so a wrong guess is dropped exactly like an event with
# no file, never shipped. Skipping this resolution entirely would silently
# leave roughly half of all monster moves without a cue.
_BARE_ATTACKERFX = re.compile(r"WithAttackerFx\(null,\s*(?P<name>[A-Za-z_]\w*)\s*\)")
_BARE_PLAY = re.compile(r"SfxCmd\.Play\(\s*(?P<name>[A-Za-z_]\w*)\s*[,)]")
_PROP_DECL_TMPL = (
    r"(?:private|protected|public|internal)\s+(?:static\s+)?"
    r"(?:override\s+|virtual\s+|new\s+)*(?:readonly\s+)?string\??\s+%s\b\s*(?:=>|\{|=)"
)
_CONVENTION_SUFFIX = {"AttackSfx": "attack", "CastSfx": "cast", "DeathSfx": "die"}
_PASCAL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_PASCAL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def _pascal_to_snake(name: str) -> str:
    """`BruteRubyRaider` -> `brute_ruby_raider` (the id-slugging convention
    used throughout this codebase's `enemy_attacks/` folder names)."""
    s1 = _PASCAL_BOUNDARY_1.sub(r"\1_\2", name)
    return _PASCAL_BOUNDARY_2.sub(r"\1_\2", s1).lower()


def _bare_sfx_name(body: str) -> str | None:
    m = _BARE_ATTACKERFX.search(body) or _BARE_PLAY.search(body)
    return m["name"] if m else None


def _local_literal(source: str, name: str) -> str | None:
    """The first `event:/…` literal inside `name`'s own property/const body,
    if this file declares one (rather than inheriting the base default)."""
    decl = re.search(_PROP_DECL_TMPL % re.escape(name), source)
    if not decl:
        return None
    window = source[decl.end() : decl.end() + 800]
    lit = _EVENT_LITERAL.search(window)
    return lit["event"] if lit else None


def _resolve_bare_name(name: str, source: str, slug: str | None) -> str | None:
    literal = _local_literal(source, name)
    if literal:
        return literal
    if slug and name in _CONVENTION_SUFFIX:
        base = _pascal_to_snake(slug)
        return f"event:/sfx/enemy/enemy_attacks/{base}/{base}_{_CONVENTION_SUFFIX[name]}"
    return None


def parse_move_sfx(source: str, slug: str | None = None) -> dict[str, str]:
    """Map each move-handler method to its one-shot sfx event, if any.

    Looping sounds (`SfxCmd.PlayLoop`) are ambience and are never a move's cue,
    so a method with only a loop yields nothing. `slug` is the monster's own
    class name (its `.cs` stem), used only to resolve a bare `AttackSfx`/
    `CastSfx`/`DeathSfx` reference that this file does not itself override —
    see the block comment above `_bare_sfx_name`.
    """
    out: dict[str, str] = {}
    starts = [(m.start(), m["method"]) for m in _METHOD.finditer(source)]
    for i, (start, method) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(source)
        body = source[start:end]
        loops = {m["event"] for m in _LOOP.finditer(body)}
        resolved = None
        for m in _ONESHOT.finditer(body):
            if m["event"] not in loops:
                resolved = m["event"]
                break
        if resolved is None:
            name = _bare_sfx_name(body)
            if name:
                resolved = _resolve_bare_name(name, source, slug)
        if resolved:
            out[method] = resolved
    return out


# Only the expression-bodied form is matched. A multi-line getter (ToughEgg
# picks tough_egg_die vs hatchling_die on `_hatched`, OwlMagistrate picks by
# `IsFlying`, TestSubject delegates conditionally to `base.DeathSfx`) has two
# possible values and no static way to choose, so it is omitted rather than
# guessed — the same rule as conditional intents and unresolvable
# `hits`/`damage`.
_CREATURE_SFX = re.compile(
    r"public override string (?P<slot>DeathSfx|HurtSfx)\s*=>\s*\"(?P<event>event:/[^\"]+)\"\s*;"
)
_SLOT_KEY = {"DeathSfx": "death", "HurtSfx": "hurt"}

# --- The impact sound: what a monster's BODY sounds like when a hit lands -----
# `CreatureCmd.cs` plays this on every hit that gets through block, via
# `SfxCmd.PlayDamage` -> `MonsterModel.TakeDamageSfx`. It is a per-material
# sound, and it is the sound of an attack landing -- distinct from `block_hit`,
# which the same file plays ONLY when damage was FULLY blocked.
#
# `MonsterModel.cs`:
#   virtual DamageSfxType TakeDamageSfxType => DamageSfxType.Armor;
#   virtual string TakeDamageSfx => "…/enemy_impact_" + Slugify(type).ToLower();
# so a monster that overrides neither still has a real impact sound.
_IMPACT_PREFIX = "event:/sfx/enemy/enemy_impact_enemy_size/enemy_impact_"
# `MonsterModel`'s virtual default, used by every monster that doesn't override.
DEFAULT_DAMAGE_SFX_TYPE = "Armor"

_DAMAGE_SFX_TYPE = re.compile(
    r"public override DamageSfxType TakeDamageSfxType\s*=>\s*DamageSfxType\.(?P<kind>\w+)\s*;"
)
# Expression-bodied only, like `_CREATURE_SFX`. ThievingHopper's multi-line
# getter picks a hover-specific sound OR `base.TakeDamageSfx`; skipping it here
# falls through to its stated `TakeDamageSfxType` (Insect), which IS the base
# branch — the same "don't guess a conditional" rule, landing on a real value.
_TAKE_DAMAGE_SFX = re.compile(
    r"public override string TakeDamageSfx\s*=>\s*\"(?P<event>event:/[^\"]+)\"\s*;"
)


def _slugify_lower(name: str) -> str:
    """`StringHelper.Slugify(name).ToLowerInvariant()` for a PascalCase enum
    member: "ArmorBig" -> "armor_big", "Fur" -> "fur"."""
    return re.sub(r"([A-Za-z0-9])([A-Z])", r"\1_\2", name).lower()


def parse_impact_sfx(source: str) -> str | None:
    """The event `SfxCmd.PlayDamage` plays for this monster, or None.

    A whole-getter `TakeDamageSfx` override wins over the material, because it
    replaces the derivation rather than feeding it. `DamageSfxType.None` means
    the monster is deliberately silent on impact, so it maps to no event
    instead of to a `enemy_impact_none` file that was never rendered.
    """
    override = _TAKE_DAMAGE_SFX.search(source)
    if override:
        return override["event"]
    m = _DAMAGE_SFX_TYPE.search(source)
    kind = m["kind"] if m else DEFAULT_DAMAGE_SFX_TYPE
    if kind == "None":
        return None
    return _IMPACT_PREFIX + _slugify_lower(kind)


def parse_creature_sfx(source: str) -> dict[str, str]:
    """Read a monster's `DeathSfx`/`HurtSfx` overrides, if it declares them.

    Returns only the slots this class states outright; a class that inherits
    a base default, or declares a conditional getter, contributes nothing.
    """
    return {_SLOT_KEY[m["slot"]]: m["event"] for m in _CREATURE_SFX.finditer(source)}


# The game plays sound through TWO systems, and the map carries the decomp's
# literal string for each rather than inventing a scheme of its own:
#   `event:/…`             an FMOD event, rendered out of the banks.
#   `res://debug_audio/…`  a plain Godot stream played by `NDebugAudioManager`
#                          ("temporary sound effects (hopefully they make it
#                          into FMOD someday)"), copied verbatim out of the pck.
# `TmpSfx.GetPath` is what turns the bare `"victory.mp3"` the call sites pass
# into the second form, so this prefix is the game's own, not ours.
DEBUG_AUDIO_SCHEME = "res://debug_audio/"


def tmp_sfx_path(filename: str) -> str:
    """`"victory.mp3"` -> `"res://debug_audio/victory.mp3"` — `TmpSfx.GetPath`."""
    return DEBUG_AUDIO_SCHEME + filename


def _event_audio_path(event: str, audio_root: Path) -> Path:
    """The rendered file a sound path resolves to.

    `event:/sfx/block_gain` -> `<audio_root>/sfx/block_gain.webm`;
    `res://debug_audio/victory.mp3` -> `<audio_root>/debug_audio/victory.webm`.
    Both families are encoded into the same tree by `encode_sfx`, so both end
    in `.webm` — the source extension is dropped, not preserved.
    """
    if event.startswith(DEBUG_AUDIO_SCHEME):
        rel = event.removeprefix("res://")
        return audio_root / (rel.rsplit(".", 1)[0] + ".webm")
    return audio_root / (event[len("event:/") :] + ".webm")


def _resolve(event: str | None, audio_root: Path | None) -> str | None:
    """Drop an event with no rendered audio file behind it.

    The decomp is not internally consistent: some `SfxCmd`/`WithAttackerFx`
    calls name an event that was never rendered (a typo baked into the C#
    itself, e.g. `..._boomerang_slashh` with a doubled `h`, or a const bound
    to the wrong literal, e.g. VineShambler's `_graspingVines` pointing at
    `..._cast` instead of `..._grasping_vines`). In the live game an FMOD
    lookup on a nonexistent event key is a silent no-op, so the faithful
    (and Task-11-required: zero unresolvable mapped events) behaviour here is
    the same graceful degradation as a move with no sfx call at all — drop
    the event rather than ship a path that 404s.
    """
    if event is None or audio_root is None:
        return event
    return event if _event_audio_path(event, audio_root).exists() else None


def _attack_kind_names(
    intent_kind: dict[str, str], attack_family: frozenset[str] | set[str]
) -> set[str]:
    """The resolved enum names (e.g. {"Attack", "DeathBlow"}) that count as
    attack-family, for filtering already-resolved `move["intent"]` values."""
    return {intent_kind.get(prefix, prefix) for prefix in attack_family}


def build_monster_entry(
    source: str,
    slug: str | None = None,
    intent_kind: dict[str, str] | None = None,
    attack_family: frozenset[str] | set[str] | None = None,
) -> dict:
    """Join a monster's moves to their sfx and pick a fallback attack sound.

    Every parsed move is kept, even one with no dedicated one-shot (about half
    of them, in practice — many moves only carry a Buff/Debuff visual with no
    `SfxCmd.Play`/`WithAttackerFx` call). Dropping those silently would starve
    the web of the {intent, damage, hits} shape it matches on for a move that
    never gets its own cue, which is a worse failure than a null "event": the
    move would vanish from the map instead of degrading to a generic sound.

    `slug` (the monster's class name) feeds the `AttackSfx`/`CastSfx`
    convention fallback in `parse_move_sfx` — see its docstring. `intent_kind`
    and `attack_family` come from `load_intent_tables` (falls back to the
    lazily-derived default when omitted — see `_default_intent_tables`).
    `defaultAttack` is chosen from any attack-family move (`Attack` OR
    `DeathBlow`), not just moves literally named "Attack" — a finishing blow
    is exactly as valid a default combat sound as a regular hit.
    """
    if intent_kind is None or attack_family is None:
        intent_kind, attack_family = _default_intent_tables()
    sfx = parse_move_sfx(source, slug)
    attack_kinds = _attack_kind_names(intent_kind, attack_family)
    moves = []
    for mv in parse_move_states(source, intent_kind, attack_family):
        move: dict = {
            "name": mv["name"],
            "intent": mv["intent"],
            "event": sfx.get(mv["method"]),
        }
        if "hits" in mv:
            move["hits"] = mv["hits"]
        if "damage" in mv:
            move["damage"] = mv["damage"]
        moves.append(move)
    attacks = [m["event"] for m in moves if m["intent"] in attack_kinds and m["event"]]
    entry: dict = {"moves": moves}
    if attacks:
        entry["defaultAttack"] = attacks[0]
    for slot, event in parse_creature_sfx(source).items():
        entry[slot] = event
    impact = parse_impact_sfx(source)
    if impact:
        entry["impact"] = impact
    return entry


# --- Power valence (Buff/Debuff): derived from the decomp, never hand-written ---
# `EffectEvent` carries only a power NAME ("Vulnerable", "Strength", …), with no
# indication of whether gaining it is good or bad news — but the decomp knows:
# every power class declares `PowerType.Buff` or `PowerType.Debuff`. Most
# (238/260) do it as a one-line `public override PowerType Type => PowerType.X;`;
# 22 don't (see `_POWER_TYPE_LITERAL` below) — 19 concrete classes that inherit
# a *conditional* `Type` getter from one of three abstract bases
# (TemporaryStrengthPower/TemporaryFocusPower/TemporaryDexterityPower: `Type =>
# IsPositive ? Buff : Debuff`), plus those 3 bases themselves (never
# instantiated directly, so they carry no title and never reach the output
# table anyway). `IsPositive` defaults to `true` (Buff) unless a subclass
# overrides it to a literal `false` — a real, static fact about that subclass,
# not a runtime guess (see `_resolve_is_positive`) — so all 22 resolve; none
# need to be omitted for this game version. A class that somehow resolves to
# neither Buff nor Debuff (e.g. `PowerType.None`, or a chain this can't walk)
# is still omitted — absent beats wrong, same rule as `hits`/`damage` above.
#   Anchored to an actual declaration line (an access modifier immediately
#   before `class`), the same way `_INTENT_CLASS_DECL` above is -- a bare
#   `r"class (\w+)"` also matches XML-doc prose ("/// This class represents a buff/
#   debuff...") when it appears before the real declaration, e.g.
#   TemporaryDexterityPower.cs, and .search() takes the FIRST match: three
#   power base classes were silently mis-keyed under "represents" until this
#   was caught by their concrete subclasses coming out with no resolvable
#   valence at all (see load_power_valences's docstring test coverage).
_POWER_CLASS_DECL = re.compile(
    r"(?:^|\n)\s*(?:public|internal)\s+(?:sealed\s+|abstract\s+|static\s+|partial\s+)*"
    r"class\s+(?P<name>\w+)(?:\s*:\s*(?P<base>\w+))?"
)
_POWER_TYPE_LITERAL = re.compile(
    r"public override PowerType Type\s*=>\s*PowerType\.(?P<value>\w+)\s*;"
)
_IS_POSITIVE_DECL = re.compile(
    r"(?:protected|public)\s+(?:virtual|override)\s+bool IsPositive"
    r"\s*=>\s*(?P<value>true|false)\s*;"
)
# `PowerModel.cs:240` declares `public virtual bool AllowNegative => false`,
# so absence IS the answer for every class that does not override it. Matched
# the same way as `IsPositive` above, and resolved through the inheritance
# chain by `_resolve_allow_negative` for the same reason.
_ALLOW_NEGATIVE_DECL = re.compile(
    r"public\s+(?:virtual|override)\s+bool AllowNegative\s*=>\s*(?P<value>true|false)\s*;"
)


def _load_power_class_tables(
    powers_dir: Path,
) -> tuple[dict[str, str | None], dict[str, str], set[str], dict[str, bool], dict[str, bool]]:
    """One pass over every `*Power.cs` file, gathering the raw facts the
    resolvers below need: each class's base (for walking the inheritance
    chain), its literal `Type` value if it declares one, whether it declares
    the `IsPositive`-conditional `Type` getter instead, its own `IsPositive`
    literal if it declares one (base classes declare the `virtual` default;
    subclasses that care declare an `override`), and its own `AllowNegative`
    literal on the same terms.
    """
    class_base: dict[str, str | None] = {}
    class_type: dict[str, str] = {}
    uses_positivity: set[str] = set()
    class_is_positive: dict[str, bool] = {}
    class_allow_negative: dict[str, bool] = {}
    for cs in sorted(powers_dir.glob("*.cs")):
        text = cs.read_text(encoding="utf-8", errors="replace")
        decl = _POWER_CLASS_DECL.search(text)
        if not decl or not decl["name"].endswith("Power"):
            continue
        name = decl["name"]
        class_base[name] = decl["base"]
        m = _POWER_TYPE_LITERAL.search(text)
        if m:
            class_type[name] = m["value"]
        elif "IsPositive" in text:
            uses_positivity.add(name)
        ispos = _IS_POSITIVE_DECL.search(text)
        if ispos:
            class_is_positive[name] = ispos["value"] == "true"
        allowneg = _ALLOW_NEGATIVE_DECL.search(text)
        if allowneg:
            class_allow_negative[name] = allowneg["value"] == "true"
    return class_base, class_type, uses_positivity, class_is_positive, class_allow_negative


def _resolve_is_positive(
    name: str, class_is_positive: dict[str, bool], class_base: dict[str, str | None]
) -> bool | None:
    """Walk `name`'s own inheritance chain for the nearest declared
    `IsPositive` literal (an override on `name` itself wins over one inherited
    from a base) -- never the chain of whichever base class happened to
    surface the conditional `Type` getter, which would find the base's own
    default instead of a subclass's override.
    """
    cur: str | None = name
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in class_is_positive:
            return class_is_positive[cur]
        cur = class_base.get(cur)
    return None


def _resolve_power_valence(
    name: str,
    class_type: dict[str, str],
    class_base: dict[str, str | None],
    uses_positivity: set[str],
    class_is_positive: dict[str, bool],
) -> str | None:
    """ "Buff" or "Debuff" for one power class, or None if unresolvable."""
    cur: str | None = name
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in class_type:
            value = class_type[cur]
            return value if value in ("Buff", "Debuff") else None
        if cur in uses_positivity:
            is_positive = _resolve_is_positive(name, class_is_positive, class_base)
            return None if is_positive is None else ("Buff" if is_positive else "Debuff")
        cur = class_base.get(cur)
    return None


def load_power_valences(decomp_root: Path) -> dict[str, str]:
    """{class_name: "Buff"|"Debuff"} for every power class resolvable from
    `MegaCrit.Sts2.Core.Models.Powers/*.cs`. Keyed by CLASS name (e.g.
    "VulnerablePower") -- callers join this against the power's localized
    display title to get the name `EffectEvent.power` actually carries; see
    `load_power_table`.
    """
    powers_dir = decomp_root / "MegaCrit.Sts2.Core.Models.Powers"
    if not powers_dir.is_dir():
        return {}
    class_base, class_type, uses_positivity, class_is_positive, _allow_negative = (
        _load_power_class_tables(powers_dir)
    )
    out: dict[str, str] = {}
    for name in class_base:
        value = _resolve_power_valence(
            name, class_type, class_base, uses_positivity, class_is_positive
        )
        if value:
            out[name] = value
    return out


def load_power_allow_negative(decomp_root: Path) -> dict[str, bool]:
    """{class_name: True} for every power class whose `AllowNegative` resolves
    to true. Keyed by CLASS name, like `load_power_valences`; only the true
    rows are returned, since `PowerModel.cs:240`'s `virtual ... => false` makes
    absence the answer for everything else.
    """
    powers_dir = decomp_root / "MegaCrit.Sts2.Core.Models.Powers"
    if not powers_dir.is_dir():
        return {}
    class_base, _type, _pos, _isp, class_allow_negative = _load_power_class_tables(powers_dir)
    out: dict[str, bool] = {}
    for name in class_base:
        if _resolve_allow_negative(name, class_allow_negative, class_base):
            out[name] = True
    return out


def _resolve_allow_negative(
    name: str, class_allow_negative: dict[str, bool], class_base: dict[str, str | None]
) -> bool:
    """`_resolve_is_positive`'s walk for `AllowNegative`: the nearest declared
    literal up `name`'s own chain wins, and no declaration anywhere means the
    `PowerModel` default, false.
    """
    cur: str | None = name
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in class_allow_negative:
            return class_allow_negative[cur]
        cur = class_base.get(cur)
    return False


# Final re-review (2026-08-24, finding 4): TestSubject's static title is
# "Test Subject #C{Count}" -- `{Count}` fills from `SaveManager.Progress.
# TestSubjectKills + 8`, cross-run save progress this generator has no access
# to, so the literal template text can never equal a real wire string.
# Dropping the whole entry (the first-round fix) was wrong: the workbench's
# OWN bridge (`sts2_hint.display_maps.monster_title` /
# `sts2_hint.adapter`'s `_SPECIMEN_TEMPLATE`) already strips this exact tail
# before the name ever reaches the wire, so "Test Subject" (the bare species
# name) IS a stable, real, matchable wire string -- absent beats wrong only
# applies to the unresolved `{Count}` tail, not to the whole title. These two
# regexes are kept as local literals (not imported) to avoid a
# wiki -> pycli dependency, but are otherwise identical to display_maps.py's.
_SPECIMEN_TEMPLATE = re.compile(r"\s*#\S*\{[^}]+\}.*$")
_LOC_TEMPLATE_VAR = re.compile(r"\s*\{[^}]+\}")


def _strip_unresolved_template(title: str) -> str:
    """Strip an unresolved loc-template tail from a title (e.g. "Test
    Subject #C{Count}" -> "Test Subject"), mirroring
    `sts2_hint.display_maps.monster_title`'s own stripping exactly -- see the
    module-level comment above `_SPECIMEN_TEMPLATE`. A no-op on any title
    with no such tail.
    """
    return _LOC_TEMPLATE_VAR.sub("", _SPECIMEN_TEMPLATE.sub("", title)).strip()


def _class_titles(decomp_root: Path, data_filename: str) -> dict[str, str]:
    """{class_name: localized display title} sourced from
    `data/<version>/<data_filename>` (e.g. "powers.json", "monsters.json",
    "cards.json", "relics.json") -- the file each `scripts/extract_*.py`
    already resolves against the real localization tables, so this script has
    no reason to re-derive a title from a class name (it can't, in general:
    "AssassinRubyRaider" -> "Assassin Raider", "BowlbugEgg" -> "Bowlbug
    (Egg)" -- neither is recoverable by transforming the class name).

    That file is committed per game version (unlike `extracted/`, which holds
    the raw localization JSON and is gitignored), so it sits alongside every
    committed `decompiled/<version>/` tree. Returns {} when it's missing (a
    synthetic test fixture's decomp root, or a version with no `data/` entry
    yet) -- callers then come out empty rather than guessing a title.

    Every title is passed through `_strip_unresolved_template` first (a
    no-op unless it carries an unresolved `{Var}` tail); a title that strips
    to nothing is skipped (still absent beats wrong for a title that is
    ENTIRELY a live-state template, though none exist in v0.107.1).
    """
    data_path = decomp_root.parent.parent / "data" / decomp_root.name / data_filename
    if not data_path.exists():
        return {}
    entries = json.loads(data_path.read_text(encoding="utf-8"))
    titles: dict[str, str] = {}
    for entry in entries:
        title = entry.get("title")
        class_name = entry.get("class_name")
        if not title or not class_name:
            continue
        title = _strip_unresolved_template(title)
        if title:
            titles[class_name] = title
    return titles


def _power_class_titles(decomp_root: Path) -> dict[str, str]:
    """{class_name: localized display title}, e.g. "VulnerablePower" ->
    "Vulnerable", "BackAttackLeftPower" -> "Back Attack". See `_class_titles`.
    """
    return _class_titles(decomp_root, "powers.json")


# Final re-review (2026-08-24, finding 1): a monster whose C# `Title` getter
# switches to a DIFFERENT localization key under some in-combat state --
# `extract_monsters.py`'s static extraction only ever sees the DEFAULT key,
# so `_class_titles` above can't produce these. v0.107.1's only case:
# `ToughEgg.cs`'s `Title` returns `L10N(Id.Entry + ".name")` ("Tough Egg")
# before its `HATCH_MOVE` fires and `L10N("HATCHLING.name")` ("Hatchling")
# after -- both are real, reachable wire names for the SAME class's moves
# (post-hatch, the enemy keeps attacking with the class's own NIBBLE_MOVE,
# now under the "Hatchling" name). Mirrors
# `sts2_hint.canonical_maps.MONSTER_TITLE_ALIASES` (kept as a local literal,
# not imported, to avoid a wiki -> pycli dependency) -- {alias_title:
# class_name}, the reverse shape of `_class_titles`' {class_name: title}.
# Merged into `monsters` in `build_map_with_report` as EXTRA keys for the
# same class, through the same `_keyed_by_title` merge-or-drop rule, so a
# monster reachable under two wire names resolves under both, and a genuine
# conflict (an alias colliding with an unrelated class's own title) still
# drops rather than guesses -- add a new class here, not an `if class_name
# == "ToughEgg"` special case in the join logic itself.
MONSTER_TITLE_ALIASES: dict[str, str] = {
    "Hatchling": "ToughEgg",
}

_T = TypeVar("_T")


def _keyed_by_title(
    class_values: dict[str, _T], titles: dict[str, str]
) -> tuple[dict[str, _T], set[str]]:
    """Re-key `class_values` (class_name -> V) by display title. Returns
    `(by_title, conflicting_titles)`.

    A class with no resolvable title contributes nothing (absent beats
    wrong). Two classes sharing one title merge without loss when they agree
    on the value; if they ever disagreed, that title is dropped entirely
    (from the returned dict, but reported via `conflicting_titles`) rather
    than guessing which class a bare display name means. This is the exact
    policy `load_power_table` established for the `powers` table, generalized
    so `build_map` can apply it to `monsters`, `cards` and `relics` too --
    the wire only ever carries a display name for any of the four, never a
    class name.
    """
    by_title: dict[str, _T] = {}
    conflicting: set[str] = set()
    for class_name, value in class_values.items():
        title = titles.get(class_name)
        if title is None:
            continue
        if title in by_title and by_title[title] != value:
            conflicting.add(title)
            continue
        by_title[title] = value
    for title in conflicting:
        by_title.pop(title, None)
    return by_title, conflicting


def _merge_alias_titles(
    class_values: dict[str, _T],
    keyed: dict[str, _T],
    conflicts: set[str],
    aliases: dict[str, str],
) -> None:
    """Merge extra `{alias_title: class_name}` entries into an
    ALREADY-title-keyed table, IN PLACE (mutates `keyed` and `conflicts`).

    Same merge-when-agree / drop-when-disagree policy as `_keyed_by_title`,
    applied against the table it already produced: an alias whose class has
    no sound entry in `class_values` contributes nothing; an alias title that
    collides with an EXISTING key (its primary title, or another class'
    entirely) mapping to a different value is dropped and recorded in
    `conflicts` rather than guessed. See `MONSTER_TITLE_ALIASES`.
    """
    for alias_title, class_name in aliases.items():
        value = class_values.get(class_name)
        if value is None:
            continue
        if alias_title in keyed and keyed[alias_title] != value:
            conflicts.add(alias_title)
            keyed.pop(alias_title, None)
            continue
        keyed[alias_title] = value


def load_power_table(decomp_root: Path) -> dict[str, str]:
    """The `powers` table shipped in sfx_map.json: {display name: "Buff"|
    "Debuff"}, e.g. {"Vulnerable": "Debuff", "Strength": "Buff", …}.

    Joins `load_power_valences` (by class) against `_power_class_titles` (by
    class) to key the result by display name -- the only form
    `EffectEvent.power` (and so `cuesForCombat`) ever sees. A class with no
    resolvable valence, or no title, contributes nothing (absent beats
    wrong). Two classes sharing one title (v0.107.1: only "Back Attack",
    Left/Right, both Buff) is fine when they agree; if they ever disagreed,
    that title is dropped rather than guessing which class a bare name
    means.
    """
    valences = load_power_valences(decomp_root)
    titles = _power_class_titles(decomp_root)
    by_title, _conflicting = _keyed_by_title(valences, titles)
    return by_title


def load_power_allow_negative_table(decomp_root: Path) -> list[str]:
    """The `powersAllowNegative` list shipped in sfx_map.json: the display
    names of every power whose amount is allowed to go below zero, sorted.
    v0.107.1 has five — Strength, Dexterity, Focus, Shriek and Shrink.

    This is the missing half of `GetTypeForAmount` (`PowerModel.cs:462-472`),
    which the `powers` table alone cannot express: a power's rendered valence
    is a function of its amount, not only of its name, and negative Strength
    is a DEBUFF in-game. The consumer's rule is in
    `sts2-pycli/web/src/sound/powerValence.ts`; the note there records why
    `StackType == Counter`, the C#'s third condition, needs no column here.

    Keyed by display title on the same join (and the same drop-on-conflict
    policy) as `load_power_table`, because the display title is the only form
    the wire ever carries.
    """
    allow_negative = load_power_allow_negative(decomp_root)
    titles = _power_class_titles(decomp_root)
    by_title, _conflicting = _keyed_by_title(allow_negative, titles)
    return sorted(by_title)


# The keys the web looks up, and the event each resolves to. Every key that
# `src/sound/triggers.ts` or `src/sound/runCues.ts` reads MUST appear here, or
# that cue is silently mute. `victory` is deliberately absent: the decomp has
# no such event, and inventing one would ship a 404.
GENERIC_EVENTS: dict[str, str] = {
    "blockGain": "event:/sfx/block_gain",
    "blockBreak": "event:/sfx/block_break",
    "blockHit": "event:/sfx/block_hit",
    # Fallback impact for a monster with no map entry of its own: MonsterModel's
    # virtual `DamageSfxType.Armor` default, which is what the game would play.
    "impact": _IMPACT_PREFIX + _slugify_lower(DEFAULT_DAMAGE_SFX_TYPE),
    "buff": "event:/sfx/buff",
    "debuff": "event:/sfx/debuff",
    "heal": "event:/sfx/heal",
    # NOT `card_movement_B_play_into_discard`, despite the name. `SfxCmd.
    # PlayCardSwooshSfx` only reaches that branch when it is given a `prevPile`
    # whose type is `Play`, and the sole two-argument call site
    # (`NCardFlyShuffleVfx.Create`, from `CardPileCmd.cs:612`/`:618`) is fed
    # from `cardsWithoutNodesChangingPiles`, whose membership test
    # (`CardPileCmd.cs:450-458`) admits an `oldPile` of Draw/Discard/Exhaust/
    # Deck and NOTHING else. So `prevPile.Type == Play` is unreachable and
    # `play_into_discard` is dead code in v0.107.1. A card you actually played
    # still has an on-table node, so it takes the `default:` arm of
    # `CardPileCmd.cs:590`, reaching `NCardFlyVfx.PlayAnim` ->
    # `PlayCardSwooshSfx(pile)` — the ONE-argument overload, which for a
    # Discard target plays this.
    "cardPlay": "event:/sfx/ui/cards/card_movement_B_into_discard",
    # A Power card never moves to the discard at all (`CardPileCmd.cs:120`
    # exempts it), so it gets its own flight vfx and sound —
    # `NCardFlyPowerVfx.cs:183`, created from `CardModel.cs:2060`.
    "cardPower": "event:/sfx/ui/cards/card_movement_B_power",
    "uiClick": "event:/sfx/ui/clicks/ui_click",
    "mapOpen": "event:/sfx/ui/map/map_open",
    "mapSelect": "event:/sfx/ui/map/map_select",
    # Three tiers, picked by AMOUNT: `PlayerCmd.cs:152` reads
    # `amount >= 100 ? gold_3 : amount > 30 ? gold_2 : gold_1`. `gold` (the
    # untiered key) is kept so a map read by an older bundle still resolves
    # something; today's web prefers the tiers and only falls back to it.
    "gold": "event:/sfx/ui/gold/gold_1",
    "goldSmall": "event:/sfx/ui/gold/gold_1",
    "goldMedium": "event:/sfx/ui/gold/gold_2",
    "goldLarge": "event:/sfx/ui/gold/gold_3",
    "energy": "event:/sfx/ui/gain_energy",
    "relic": "event:/sfx/ui/relic_activate_general",
    "defeat": "event:/temp/sfx/game_over",
    # --- the `res://debug_audio/` layer (see DEBUG_AUDIO_SCHEME) ---
    # Winning a fight. `NRewardsScreen._Ready` (`:400`) plays this
    # UNCONDITIONALLY as the rewards screen opens, so it is the combat-victory
    # sting, not a run-win fanfare. There is no FMOD event for it — which is
    # why the workbench believed for a while that "a win is silent by design".
    "victory": tmp_sfx_path("victory.mp3"),
    # Obtaining a relic, from ANY source — `RelicCmd.Obtain` (`:49`) plays it
    # for the local player before the relic is even animated in. That covers
    # the Neow boon, a boss relic, a shop purchase and a chest alike.
    "relicGet": tmp_sfx_path("relic_get.mp3"),
    # Neow's greeting as her room opens. Her `AncientDialogueSet`
    # (`Neow.DefineDialogues`) actually picks per character and per visit
    # index, and four of its five agnostic lines are `neow_welcome` — this is
    # the first-visit-ever line (`FirstVisitEverDialogue`) and the modal
    # choice across the set, which is as close as a workbench with no
    # cross-run visit counter can faithfully get.
    "neowWelcome": "event:/sfx/npcs/neow/neow_welcome",
    # --- combat rhythm ---
    # The two turn banners. `NPlayerTurnBanner.cs:112` / `NEnemyTurnBanner.cs:84`
    # play these as their banner animation starts, i.e. once per side per round.
    "playerTurn": tmp_sfx_path("player_turn.mp3"),
    "enemyTurn": tmp_sfx_path("enemy_turn.mp3"),
    # `NCombatStartBanner.cs:99` picks one at random (`Rng.Chaotic.NextItem`)
    # from `TmpSfx.BattleStart`, which is exactly these two.
    "battleStart1": tmp_sfx_path("battle_start_1.mp3"),
    "battleStart2": tmp_sfx_path("battle_start_2.mp3"),
    # Dealing a card into the hand (`CardPileCmd.cs:875`, at 0.25 volume — it
    # fires per card, so the game's own mix keeps a five-card draw from
    # drowning everything).
    "cardDeal": tmp_sfx_path("card_deal.mp3"),
    "cardExhaust": tmp_sfx_path("card_exhaust.mp3"),
    # Picking a card UP (`NMouseCardPlay.StartCardDrag`, 0.5 volume). The
    # workbench's click-to-select is the same gesture as the game's drag-start.
    "cardSelect": tmp_sfx_path("card_select.mp3"),
    # Every monster's death fade, on top of whatever `DeathSfx` its own class
    # names — `NMonsterDeathVfx.cs:227`, which runs for all of them.
    "enemyFade": "event:/sfx/enemy/enemy_fade",
    # --- potions ---
    "potionGain": tmp_sfx_path("gain_potion.mp3"),
    # `NPotionHolder` picks one of the three at random, both on hover (`:286`)
    # and on the start-of-combat shine (`:662`).
    "potionSlosh1": tmp_sfx_path("potion_slosh_1.mp3"),
    "potionSlosh2": tmp_sfx_path("potion_slosh_2.mp3"),
    "potionSlosh3": tmp_sfx_path("potion_slosh_3.mp3"),
    # --- deck operations ---
    # `NCardSmithVfx.cs:179`/`:223` — the upgrade/smith sparkle, also what the
    # Amalgamator event plays (`Amalgamator.cs:45`).
    "cardSmith": tmp_sfx_path("card_smith.mp3"),
    "cardTransform": "event:/sfx/ui/cards/card_transform",
    "enchant": "event:/sfx/ui/enchant_shimmer",
    # --- rest site ---
    # `HealRestSiteOption.cs:98-99` plays `sleep.tres` AND the blanket. That
    # `.tres` is an `AudioStreamRandomizer` over exactly these three jingles
    # (read out of the pck; it is not a decodable stream itself, so the map
    # names the three streams it randomises over rather than the resource).
    "restSleep1": tmp_sfx_path("STS_SleepJingle_1a_NewMix_v1.mp3"),
    "restSleep2": tmp_sfx_path("STS_SleepJingle_1b_NewMix_v1.mp3"),
    "restSleep3": tmp_sfx_path("STS_SleepJingle_1c_NewMix_v1.mp3"),
    "restBlanket": tmp_sfx_path("SOTE_SFX_SleepBlanket_v1.mp3"),
    "restDig": tmp_sfx_path("sts_sfx_shovel_v1.mp3"),
    # --- screens ---
    # Named `map_open` in the game's asset tree but it is NOT the map: the
    # three callers are `NDeckViewScreen.cs:167`, `NCardPileScreen.cs:218` and
    # `NSimpleCardsViewScreen.cs:135`, i.e. opening a card list. (The map
    # overlay has its own FMOD `mapOpen` above.) Keyed by what it does.
    "screenOpen": tmp_sfx_path("map_open.mp3"),
    # --- NPCs ---
    "merchantWelcome": "event:/sfx/npcs/merchant/merchant_welcome",
    "merchantThanks": "event:/sfx/npcs/merchant/merchant_thank_yous",
    # Necrobinder summoning Osty — `OstyCmd.cs:47`.
    "summon": "event:/sfx/characters/necrobinder/necrobinder_summon",
}

# The chest-opening sound, per act. `ActModel.ChestOpenSfx` is an override on
# each act class rather than one table, and acts 1a/1b share a sound — read
# from the decomp by `load_treasure_sfx` rather than hard-coded, so a fourth
# act (or a re-pointed override) travels without an edit here.
_CHEST_OPEN_SFX = re.compile(r"ChestOpenSfx\s*=>\s*\"(?P<event>event:/[^\"]+)\"")


def load_treasure_sfx(decomp_root: Path, audio_root: Path | None = None) -> dict[str, str]:
    """`{"1": event, "2": …}` — the chest sound for each act number.

    Keyed by ACT NUMBER, because that is what the workbench's treasure-reveal
    payload carries (`RoomReveal.act`); the act's class name never reaches the
    wire. Two act-1 variants (Overgrowth and Underdocks) both point at
    `treasure_act1`, so a duplicate value is expected, not a conflict.
    """
    out: dict[str, str] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Acts/*.cs")):
        source = cs.read_text(encoding="utf-8", errors="replace")
        m = _CHEST_OPEN_SFX.search(source)
        if not m:
            continue
        event = _resolve(m["event"], audio_root)
        if not event:
            continue
        # `treasure_act2` -> act "2". The act number is in the event name, not
        # in the class name (`Hive`, `Glory`, `Overgrowth`, `Underdocks`).
        num = event.rsplit("_act", 1)[-1]
        if num.isdigit():
            out[num] = event
    return out


# Which orb methods the class actually calls. `OrbModel.PlayPassiveSfx` /
# `PlayEvokeSfx` are `protected` and each orb calls them from its own
# `Passive`/`Evoke` override — several orbs never call one of them at all, and
# that silence is the game's, so it must be the workbench's too.
_ORB_PLAY_CALL = re.compile(r"Play(?P<slot>Passive|Evoke|Channel)Sfx\(\)")
_ORB_SFX_OVERRIDE = re.compile(
    r"override string (?P<slot>Passive|Evoke|Channel)Sfx\s*=>\s*\"(?P<event>[^\"]*)\""
)


def load_orb_sfx(decomp_root: Path, audio_root: Path | None = None) -> dict[str, dict[str, str]]:
    """`{"Frost": {"passive": …, "evoke": …, "channel": …}, …}`, keyed by orb TYPE.

    Each `Play<Slot>Sfx` in `OrbModel.cs:171-203` prefers the class's FMOD
    override and falls back to a `res://debug_audio/<slug>_<slot>.mp3` derived
    from the model id (`StringHelper.Slugify("FrostOrb")` -> `FROST_ORB` ->
    `frost_orb`). BOTH halves matter and neither is guessable from the file
    tree: `frost_orb_channel.mp3` and `frost_orb_evoke.mp3` both ship, but the
    class overrides `ChannelSfx` with an FMOD event, so the channel mp3 is
    never played — and `glass_orb_evoke.mp3` ships while `GlassOrb` never
    calls `PlayEvokeSfx()` at all.

    `PlayChannelSfx` is the exception: it is `public` and called once, for
    EVERY orb, from `OrbCmd.cs:87`, so it is credited to every orb class
    rather than looked for in the class body.

    The key is the class name minus its `Orb` suffix ("FrostOrb" -> "Frost"),
    which is the string the wire sends as `Orb.type`.
    """
    out: dict[str, dict[str, str]] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Orbs/*.cs")):
        source = cs.read_text(encoding="utf-8", errors="replace")
        overrides = {m["slot"].lower(): m["event"] for m in _ORB_SFX_OVERRIDE.finditer(source)}
        called = {m["slot"].lower() for m in _ORB_PLAY_CALL.finditer(source)} | {"channel"}
        slug = _pascal_to_snake(cs.stem)
        entry: dict[str, str] = {}
        for slot in ("passive", "evoke", "channel"):
            if slot not in called:
                continue
            # An empty override is `OrbModel`'s own `""` default, i.e. "use the
            # debug stream" — not an event named "".
            fmod = overrides.get(slot) or None
            path = fmod or tmp_sfx_path(f"{slug}_{slot}.mp3")
            resolved = _resolve(path, audio_root)
            if resolved:
                entry[slot] = resolved
        if entry:
            out[cs.stem.removesuffix("Orb")] = entry
    return out


_EVENT_LITERAL = re.compile(r'"(?P<event>event:/[^"]+)"')


def _first_event(source: str, exclude_ui: bool = True) -> str | None:
    """The first standalone (non-loop) event literal in a file.

    `exclude_ui` skips anything under `/ui/` — right for cards, where a
    shared UI click must not become mistaken for the card's own signature
    sound, but WRONG for relics: every relic's activation cue legitimately
    lives under `event:/sfx/ui/relic_activate_draw`. Excluding `/ui/` there
    made every relic resolve to nothing (see task-4-report fix round 1).
    """
    loops = {m["event"] for m in _LOOP.finditer(source)}
    for m in _EVENT_LITERAL.finditer(source):
        if m["event"] in loops:
            continue
        if exclude_ui and "/ui/" in m["event"]:
            continue
        return m["event"]
    return None


def _monsters_by_class(
    decomp_root: Path,
    intent_kind: dict[str, str],
    attack_family: frozenset[str] | set[str],
    audio_root: Path | None,
) -> dict[str, dict]:
    """`{class_name: entry}` for every monster with something resolvable —
    the pre-title-join intermediate `build_map` used to ship directly. Split
    out so both `build_map` and its `..._with_report` variant can measure how
    many of these survive the display-title join without building twice.
    """
    attack_kinds = _attack_kind_names(intent_kind, attack_family)
    monsters: dict[str, dict] = {}
    for cs in sorted((decomp_root / "MegaCrit.Sts2.Core.Models.Monsters").glob("*.cs")):
        raw_entry = build_monster_entry(
            cs.read_text(encoding="utf-8", errors="replace"),
            slug=cs.stem,
            intent_kind=intent_kind,
            attack_family=attack_family,
        )
        moves = []
        for mv in raw_entry["moves"]:
            event = _resolve(mv["event"], audio_root)
            if event:
                moves.append({**mv, "event": event})
        entry: dict = {"moves": moves}
        attacks = [m["event"] for m in moves if m["intent"] in attack_kinds]
        if attacks:
            entry["defaultAttack"] = attacks[0]
        for slot in ("death", "hurt", "impact"):
            event = _resolve(raw_entry.get(slot), audio_root)
            if event:
                entry[slot] = event
        # A monster with no resolvable moves can still carry a real, shipped
        # `death`/`hurt`/`defaultAttack` event -- FakeMerchantMonster is the
        # real case (a killable "reverse merchant" ambush whose own moves
        # never resolve to a shipped sfx call, but whose DeathSfx/HurtSfx
        # both do). Dropping the whole entry in that case throws away a
        # perfectly valid, shipped sound along with the genuinely-empty
        # moves list. Only a monster with NOTHING resolvable at all (no
        # moves, no death, no hurt, no defaultAttack) is still excluded --
        # an entry with nothing in it is not useful to the web either way.
        if moves or "death" in entry or "hurt" in entry or "defaultAttack" in entry:
            monsters[cs.stem] = entry
    return monsters


# --- Player-side character sfx -------------------------------------------
# `CreatureCmd.TriggerAnim` (v0.107.1 CreatureCmd.cs:928) is the ONLY place a
# PLAYER's own attack/cast sound comes from: when the animated creature
# `IsPlayer` it switches on the trigger name and plays `character.AttackSfx` /
# `CastSfx` / `PowerUpSfx` BEFORE setting the trigger. That switch has exactly
# three arms — every other trigger name the game uses on a player
# ("heavyAttack", "summonTrigger", "sovereignBladeTrigger", "Shiv",
# "attack_poke", "Hit") falls straight through and plays NOTHING. So this
# table is the switch, not a naming convention: a name missing from it means
# silence, which is why `card_player_anim` returns None for those rather than
# guessing a nearby sound.
_PLAYER_ANIM_SFX: dict[str, str] = {"Attack": "attack", "Cast": "cast", "PowerUp": "cast"}

# `CharacterModel.cs:200-206` builds all three from `Id.Entry` — there is no
# per-character override anywhere in v0.107.1, so the events are derived from
# the class name (an override a later build declares is parsed, below). That
# derivation is SELF-CHECKING: every candidate goes through `_resolve`, so a
# class whose `Id.Entry` is not its own name (or that has no rendered audio at
# all — Deprived, RandomCharacter, DeprecatedCharacter) drops out instead of
# shipping a path that 404s.
_CHARACTER_SFX_SLOTS: tuple[str, ...] = ("attack", "cast", "die")
# v0.113.0 made `DeathSfx` virtual, and the Fool overrides it with a sound
# another character's folder holds (`Fool.cs`: `DeathSfx =>
# "event:/sfx/characters/fugitive/fugitive_die"`). An override, when a
# character declares one, wins over the derived name.
_CHARACTER_SFX_OVERRIDE = re.compile(
    r'public override string (?P<prop>Attack|Cast|Death)Sfx\s*=>\s*"(?P<event>event:[^"]+)"'
)
_CHARACTER_SFX_OVERRIDE_SLOT = {"Attack": "attack", "Cast": "cast", "Death": "die"}

_TRIGGER_ANIM_ON_OWNER = re.compile(
    r"CreatureCmd\.TriggerAnim\(\s*base\.Owner\.Creature\s*,\s*"
    r'(?P<arg>"[A-Za-z_]+"|[\w.]+\()'
)
# v0.111.0 passes the play along (`.FromCard(this, cardPlay)`); v0.107.1 is
# `.FromCard(this)`. Both seed the attacker animation the same way.
_FROM_CARD = re.compile(r"\.FromCard\(this\s*[,)]")
_WITH_ATTACKER_ANIM_ARG = re.compile(r'\.WithAttackerAnim\(\s*(?P<arg>"[A-Za-z_]+"|[\w.]+\()')
# `public static string GetHeavyAnimIfApplicable(CharacterModel character) {
#      if (!(character is Ironclad)) { return "Attack"; } return "heavyAttack"; }`
# — a character-conditional trigger name. Both halves matter: the fallback
# ("Attack") is what EVERY OTHER character gets, and the class in the type
# test ("Ironclad") is the one character for whom the trigger becomes a name
# the TriggerAnim switch does not handle, i.e. silence.
_IF_APPLICABLE_DECL = re.compile(
    r"public static string (?P<method>Get\w+)\(CharacterModel \w+\)\s*\{\s*"
    r'if \(!\(\w+ is (?P<cls>\w+)\)\)\s*\{\s*return "(?P<fallback>\w+)";'
)


def audio_fingerprint(audio_root: Path) -> str:
    """A content hash of the whole rendered audio tree, for cache-busting.

    The audio URLs are NOT content-addressed: `eventAssetUrl` derives
    `audio/sfx/characters/ironclad/ironclad_attack.webm` from the FMOD event
    path, which is stable across every re-encode. The deploy then serves that
    URL with `Cache-Control: public, max-age=31536000, immutable`
    (`sts2-pycli/deploy/sync-assets.sh`), and `immutable` means a browser will
    not even send a revalidation request. So re-rendering the corpus changes
    the bytes at an unchanged URL, and every client that already fetched a
    sound keeps the old one for a YEAR -- a header change cannot reach them,
    because they never ask again. Only the URL can, which is what this
    fingerprint is for (`eventAssetUrl` appends it as `?v=`).

    Hashes CONTENT, never mtime or size: a rebuild that didn't re-encode
    anything must produce the same value, or every `just build-sfx-map` would
    invalidate 8.6 MB of audio in every browser for no reason. The relative
    path goes into the hash too, so a pure rename still moves it.
    """
    digest = hashlib.sha256()
    for path in sorted(audio_root.rglob("*.webm")):
        digest.update(path.relative_to(audio_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def load_character_sfx(
    decomp_root: Path, audio_root: Path | None = None
) -> dict[str, dict[str, str]]:
    """`{class_name: {"attack"|"cast"|"die": event}}` for every playable
    character with rendered audio.

    Keyed by CLASS name, deliberately unlike `monsters`/`cards`/`relics`:
    the wire sends the player's character as `state.player.name`, which is
    the class name ("Ironclad"), not the localized title ("The Ironclad").
    Running this through `_keyed_by_title` would key it "The Ironclad" and
    every lookup would miss.
    """
    out: dict[str, dict[str, str]] = {}
    for cs in sorted((decomp_root / "MegaCrit.Sts2.Core.Models.Characters").glob("*.cs")):
        slug = cs.stem.lower()
        overrides = {
            _CHARACTER_SFX_OVERRIDE_SLOT[m["prop"]]: m["event"]
            for m in _CHARACTER_SFX_OVERRIDE.finditer(cs.read_text(encoding="utf-8"))
        }
        entry: dict[str, str] = {}
        for slot in _CHARACTER_SFX_SLOTS:
            derived = f"event:/sfx/characters/{slug}/{slug}_{slot}"
            event = _resolve(overrides.get(slot, derived), audio_root)
            if event:
                entry[slot] = event
        if entry:
            out[cs.stem] = entry
    return out


def load_character_anim_fallbacks(decomp_root: Path) -> dict[str, tuple[str, str]]:
    """`{"Ironclad.GetHeavyAnimIfApplicable(": ("Attack", "Ironclad"), …}` —
    every `Get…IfApplicable` helper that picks a trigger name by character,
    mapped to (the name every OTHER character gets, the exempt character).
    """
    out: dict[str, tuple[str, str]] = {}
    for cs in sorted((decomp_root / "MegaCrit.Sts2.Core.Models.Characters").glob("*.cs")):
        text = cs.read_text(encoding="utf-8", errors="replace")
        for m in _IF_APPLICABLE_DECL.finditer(text):
            out[f"{cs.stem}.{m['method']}("] = (m["fallback"], m["cls"])
    return out


def _statement_around(source: str, index: int) -> str:
    """The single C# statement containing `index`.

    A builder chain is one statement spread over several lines, and the
    modifiers that change which animation fires (`.WithNoAttackerAnim()`,
    `.WithAttackerAnim(...)`) can sit on any of them — so a fixed-size window
    around `.FromCard(this)` would miss them. Bounded by the nearest
    `;`/`{`/`}` before and the next `;` after, which is enough because no
    attack chain contains a nested statement.
    """
    start = max(
        source.rfind(";", 0, index), source.rfind("{", 0, index), source.rfind("}", 0, index)
    )
    end = source.find(";", index)
    return source[start + 1 : end if end != -1 else len(source)]


def _arg_count(source: str, after_open_paren: int) -> int:
    """How many top-level arguments the call whose `(` was just consumed has."""
    depth, count, i = 1, 1, after_open_paren
    while i < len(source) and depth:
        c = source[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 1:
            count += 1
        i += 1
    return count


def _resolve_anim_arg(
    arg: str,
    fallbacks: dict[str, tuple[str, str]],
    statement: str | None = None,
    arg_end: int | None = None,
) -> tuple[str, str | None] | str | None:
    """One trigger-name argument -> `(sfx_kind, exempt_character)`, None for
    "the game plays nothing here", or the string "unresolved" when the
    argument is a local variable this script cannot evaluate.
    """
    if arg.startswith('"'):
        if statement is not None and arg_end is not None and _arg_count(statement, arg_end) >= 3:
            # `WithAttackerAnim(name, delay, visualAttacker)` retargets the
            # animation to a THIRD creature — Byrd Swoop animates the pet, not
            # the player — and `TriggerAnim` only plays character sfx when the
            # creature it animates `IsPlayer`.
            return None
        kind = _PLAYER_ANIM_SFX.get(arg.strip('"'))
        return (kind, None) if kind else None
    fallback = fallbacks.get(arg)
    if fallback is None:
        return "unresolved"
    kind = _PLAYER_ANIM_SFX.get(fallback[0])
    return (kind, fallback[1]) if kind else None


def card_player_anim(
    source: str, fallbacks: dict[str, tuple[str, str]]
) -> tuple[str, str | None] | str | None:
    """Which character sound one card's `OnPlay` makes the PLAYER produce.

    Returns `(kind, exempt_character)` where `kind` is "attack" or "cast",
    None when the card makes the player no sound of its own (Defend and 179
    others genuinely animate nothing), or "unresolved" when the trigger name
    is a local whose value depends on runtime state — Sovereign Blade is the
    only one, and it already carries its own `cards` entry.

    Two shapes produce a player animation, and the FIRST one in source order
    wins: an explicit `CreatureCmd.TriggerAnim(base.Owner.Creature, …)`, and
    an attack chain, whose `FromCard` seeds `_attackerAnimName = "Attack"`
    (AttackCommand.cs:220) unless the same chain overrides or disables it.
    """
    hits: list[tuple[int, str, str]] = []
    for m in _TRIGGER_ANIM_ON_OWNER.finditer(source):
        hits.append((m.start(), "trigger", m["arg"]))
    for m in _FROM_CARD.finditer(source):
        hits.append((m.start(), "attack", _statement_around(source, m.start())))
    if not hits:
        return None
    hits.sort()
    _pos, kind, payload = hits[0]
    if kind == "trigger":
        return _resolve_anim_arg(payload, fallbacks)
    if ".WithNoAttackerAnim()" in payload:
        return None
    if ".WithAttackerAnim(" in payload:
        override = _WITH_ATTACKER_ANIM_ARG.search(payload)
        if not override:
            return "unresolved"
        return _resolve_anim_arg(override["arg"], fallbacks, payload, override.end())
    return ("attack", None)


def _card_anims_by_class(decomp_root: Path) -> dict[str, dict[str, str]]:
    """`{class_name: {"kind": "attack"|"cast", "except"?: character}}`."""
    fallbacks = load_character_anim_fallbacks(decomp_root)
    out: dict[str, dict[str, str]] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Cards*/**/*.cs")):
        anim = card_player_anim(cs.read_text(encoding="utf-8", errors="replace"), fallbacks)
        if not isinstance(anim, tuple):
            continue
        kind, exempt = anim
        out[cs.stem] = {"kind": kind} if exempt is None else {"kind": kind, "except": exempt}
    return out


# `AttackCommand.WithHitFx(vfx, sfx, tmpSfx)`. All three arguments are string
# literals or `null` at every one of the ~180 call sites, so a positional
# regex is exact here rather than a heuristic — but the two sound slots are
# NOT interchangeable, which is why they are captured separately below.
_WITH_HIT_FX = re.compile(
    r"\.WithHitFx\(\s*(?P<vfx>\"[^\"]*\"|null)\s*"
    r"(?:,\s*(?P<sfx>\"[^\"]*\"|null)\s*"
    r"(?:,\s*(?P<tmp>\"[^\"]*\"|null)\s*)?)?\)"
)


def parse_hit_sfx(source: str) -> str | None:
    """The sound a card's attack makes ON THE TARGET, or None.

    `AttackCommand` (`:578-586`) plays `HitSfx` through FMOD if it is set and
    otherwise `TmpHitSfx` through `NDebugAudioManager` — the second and third
    arguments of `WithHitFx`, in that precedence. 80 cards use the third and
    only one (Byrdpip's) uses the second, which is exactly why the layer was
    missing from the workbench: harvesting FMOD literals alone found one card
    out of eighty-one.

    The slots are read STRICTLY, not interchangeably. `FallingStar.cs:40`
    passes `"blunt_attack.mp3"` in the FMOD slot — a bug in the game's own
    source. FMOD answers a lookup on a nonexistent event key with a silent
    no-op, so that card's hit really is silent in the shipped game, and
    accepting an mp3 name in the `sfx` slot here would give the workbench a
    sound the player has never heard. Same reasoning as `_resolve`.
    """
    for m in _WITH_HIT_FX.finditer(source):
        sfx, tmp = m["sfx"], m["tmp"]
        if sfx and sfx.startswith('"event:/'):
            return sfx.strip('"')
        if tmp and tmp != "null" and not tmp.startswith('"event:/'):
            name = tmp.strip('"')
            if name:
                return tmp_sfx_path(name)
    return None


def _card_hits_by_class(decomp_root: Path, audio_root: Path | None) -> dict[str, str]:
    """`{class_name: sound path}` for every card whose attack has a hit sound."""
    hits: dict[str, str] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Cards*/**/*.cs")):
        event = _resolve(
            parse_hit_sfx(cs.read_text(encoding="utf-8", errors="replace")), audio_root
        )
        if event:
            hits[cs.stem] = event
    return hits


# A card's signature sound often is not in the card at all. `Hellraiser.cs`
# holds no event literal; it spawns `NHellraiserVfx`, and THAT class plays
# `event:/sfx/characters/ironclad/ironclad_hellraiser` (`:133`). Same shape for
# Fan of Knives, Hyperbeam and the Doom kill. Scanning card sources alone found
# 9 of them; following this reference finds the rest.
_VFX_CLASS_REF = re.compile(r"\bN(?P<name>[A-Z]\w*)Vfx\b")


def _vfx_sources(decomp_root: Path) -> dict[str, str]:
    """`{"NHellraiserVfx": <source>}` for every vfx node class in the decomp."""
    out: dict[str, str] = {}
    for cs in decomp_root.glob("MegaCrit.Sts2.Core.Nodes.Vfx*/**/*.cs"):
        out[cs.stem] = cs.read_text(encoding="utf-8", errors="replace")
    return out


def card_signature_event(source: str, vfx: dict[str, str]) -> str | None:
    """A card's own sound: its literal if it has one, else its vfx node's.

    The card's own literal wins, so a card that both names an event and
    spawns a vfx node keeps the one it names. Vfx nodes are tried in the order
    they appear in the card source, and a node with no event of its own (most
    of them — `NDamageNumVfx`, `NHitSparkVfx`, …) is simply skipped rather
    than blocking the ones after it.
    """
    own = _first_event(source)
    if own:
        return own
    for m in _VFX_CLASS_REF.finditer(source):
        node = vfx.get(f"N{m['name']}Vfx")
        if node is None:
            continue
        event = _first_event(node)
        if event:
            return event
    return None


def _cards_by_class(decomp_root: Path, audio_root: Path | None) -> dict[str, str]:
    """`{class_name: event}` for every card with a resolvable sfx call."""
    vfx = _vfx_sources(decomp_root)
    cards: dict[str, str] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Cards*/**/*.cs")):
        source = cs.read_text(encoding="utf-8", errors="replace")
        event = _resolve(card_signature_event(source, vfx), audio_root)
        if event:
            cards[cs.stem] = event
    return cards


def _relics_by_class(decomp_root: Path, audio_root: Path | None) -> dict[str, str]:
    """`{class_name: event}` for every relic with a resolvable sfx call."""
    relics: dict[str, str] = {}
    for cs in sorted(decomp_root.glob("MegaCrit.Sts2.Core.Models.Relics*/**/*.cs")):
        event = _resolve(
            _first_event(cs.read_text(encoding="utf-8", errors="replace"), exclude_ui=False),
            audio_root,
        )
        if event:
            relics[cs.stem] = event
    return relics


def build_map_with_report(
    decomp_root: Path, version: str, audio_root: Path | None = None
) -> tuple[dict, dict]:
    """`build_map`, plus a `report` describing the class-name -> display-title
    join for `monsters`/`cards`/`relics`: `{table: {"before": n, "after": n,
    "dropped": n, "conflicts": [title, ...]}}`. `before`/`after` count entries
    that resolved SOMETHING (a class-name-keyed entry that never had a shipped
    sfx call was never a candidate to begin with) — `dropped` is entries that
    had a real sound but no resolvable display title, e.g. a version whose
    `data/<version>/monsters.json` hasn't been regenerated yet. `main` prints
    this; tests can read it directly instead of re-deriving before/after
    counts by calling the `_..._by_class` helpers themselves.
    """
    intent_kind, attack_family = load_intent_tables(decomp_root)

    monsters_by_class = _monsters_by_class(decomp_root, intent_kind, attack_family, audio_root)
    monster_titles = _class_titles(decomp_root, "monsters.json")
    monsters, monster_conflicts = _keyed_by_title(monsters_by_class, monster_titles)
    monsters_before_aliases = len(monsters)
    _merge_alias_titles(monsters_by_class, monsters, monster_conflicts, MONSTER_TITLE_ALIASES)

    cards_by_class = _cards_by_class(decomp_root, audio_root)
    card_titles = _class_titles(decomp_root, "cards.json")
    cards, card_conflicts = _keyed_by_title(cards_by_class, card_titles)

    relics_by_class = _relics_by_class(decomp_root, audio_root)
    relic_titles = _class_titles(decomp_root, "relics.json")
    relics, relic_conflicts = _keyed_by_title(relics_by_class, relic_titles)

    card_hits_by_class = _card_hits_by_class(decomp_root, audio_root)
    card_hits, card_hit_conflicts = _keyed_by_title(card_hits_by_class, card_titles)

    card_anims_by_class = _card_anims_by_class(decomp_root)
    card_anims, card_anim_conflicts = _keyed_by_title(card_anims_by_class, card_titles)

    generic = dict(GENERIC_EVENTS)
    if audio_root is not None:
        unresolved = [
            key for key, ev in generic.items() if not _event_audio_path(ev, audio_root).exists()
        ]
        if unresolved:
            raise ValueError(f"GENERIC_EVENTS keys with no rendered audio file: {unresolved}")

    doc = {
        "version": version,
        # Absent when there is no audio tree to fingerprint (a synthetic test
        # fixture). The web treats absent as "no cache-buster" and requests
        # the bare URL, i.e. exactly the pre-fingerprint behaviour — never a
        # literal "undefined" in the query string.
        **({"audioRev": audio_fingerprint(audio_root)} if audio_root is not None else {}),
        "generic": generic,
        "powers": load_power_table(decomp_root),
        "powersAllowNegative": load_power_allow_negative_table(decomp_root),
        "characters": load_character_sfx(decomp_root, audio_root),
        "cards": cards,
        "cardHits": card_hits,
        "orbs": load_orb_sfx(decomp_root, audio_root),
        "treasure": load_treasure_sfx(decomp_root, audio_root),
        "cardAnims": card_anims,
        "relics": relics,
        "monsters": monsters,
    }
    report = {
        "monsters": {
            "before": len(monsters_by_class),
            "after": len(monsters),
            "dropped": len(monsters_by_class) - monsters_before_aliases,
            "alias_keys_added": len(monsters) - monsters_before_aliases,
            "conflicts": sorted(monster_conflicts),
        },
        "cards": {
            "before": len(cards_by_class),
            "after": len(cards),
            "dropped": len(cards_by_class) - len(cards),
            "conflicts": sorted(card_conflicts),
        },
        "cardHits": {
            "before": len(card_hits_by_class),
            "after": len(card_hits),
            "dropped": len(card_hits_by_class) - len(card_hits),
            "conflicts": sorted(card_hit_conflicts),
        },
        "cardAnims": {
            "before": len(card_anims_by_class),
            "after": len(card_anims),
            "dropped": len(card_anims_by_class) - len(card_anims),
            "conflicts": sorted(card_anim_conflicts),
        },
        "relics": {
            "before": len(relics_by_class),
            "after": len(relics),
            "dropped": len(relics_by_class) - len(relics),
            "conflicts": sorted(relic_conflicts),
        },
    }
    return doc, report


def build_map(decomp_root: Path, version: str, audio_root: Path | None = None) -> dict:
    """Build the map. When `audio_root` is given, every event is verified
    against the actual rendered files on disk and dropped (never invented or
    substituted) if nothing was rendered for it — see `_resolve`.

    A move with no resolvable event (no sfx call found, or a `_resolve`
    miss) is dropped from the emitted `moves` list entirely, rather than
    kept with `"event": null` — a null event reaching the web is a crash or
    a broken fetch, not a graceful degradation. `build_monster_entry` itself
    still keeps null-event moves (that's its own unit-tested contract, used
    to preserve every move's {intent, damage, hits} shape at that layer);
    this is the point where the null gets resolved down to "absent."

    `monsters`/`cards`/`relics` are built keyed by class name (the decomp's
    own file stems) and then re-keyed by display title via `_keyed_by_title`
    before being returned — the wire (`RunSimulator.cs`'s `_loc.Monster`/
    `_loc.Card`/`_loc.Relic` calls) only ever sends the localized display
    name, never the class name, exactly the trap `load_power_table` already
    solved for `powers`. See `build_map_with_report` for the join's
    before/after/conflict counts.
    """
    doc, _report = build_map_with_report(decomp_root, version, audio_root)
    return doc


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decomp", type=Path, help="decompiled/<version>")
    ap.add_argument("out", type=Path, help="web/public/sfx_map.json")
    ap.add_argument(
        "--audio-root",
        type=Path,
        default=Path("site/public/audio"),
        help="rendered audio tree, used to drop events with no backing file",
    )
    ap.add_argument(
        "--rev-only",
        action="store_true",
        help="only re-derive OUT's audioRev from the audio tree; keep every other entry",
    )
    args = ap.parse_args()
    if args.rev_only:
        doc = json.loads(args.out.read_text(encoding="utf-8"))
        doc["audioRev"] = audio_fingerprint(args.audio_root)
        args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"audioRev={doc['audioRev']}")
        return
    doc, report = build_map_with_report(args.decomp, args.decomp.name, audio_root=args.audio_root)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        f"monsters={len(doc['monsters'])} cards={len(doc['cards'])} "
        f"relics={len(doc['relics'])} generic={len(doc['generic'])} "
        f"powers={len(doc['powers'])} characters={len(doc['characters'])} "
        f"cardAnims={len(doc['cardAnims'])} cardHits={len(doc['cardHits'])} "
        f"orbs={len(doc['orbs'])} treasure={len(doc['treasure'])} "
        f"audioRev={doc.get('audioRev', '-')}"
    )
    for table, r in report.items():
        if r["dropped"] or r["conflicts"]:
            print(
                f"  {table}: {r['before']} had a sound, {r['after']} kept a display-title key "
                f"({r['dropped']} dropped for no resolvable title"
                + (f", conflicts: {r['conflicts']}" if r["conflicts"] else "")
                + ")"
            )


if __name__ == "__main__":
    main()
