#!/usr/bin/env python3
"""Build and write the committed card/move -> VFX effect binding table.

Task 5 (`vfx_bindings.py`) parses one C# file at a time. This module is the
CLI wrapper: it walks every decomp `Cards/*.cs` and `Monsters/*.cs` file,
concatenates their bindings into one table, and serializes it to
`data/v0.107.1/vfx_bindings.json` — the artifact design spec §1.4 requires be
committed and re-extraction-gated (see `test_extract_vfx_bindings.py`).

Design spec: `docs/superpowers/specs/2026-08-28-workbench-vfx-particles-design.md`
§1.4 (the gate) and §1.5 (coverage arithmetic).

Usage:
    uv run python -m scripts.extract_vfx_bindings decompiled/v0.107.1 \
        data/v0.107.1/vfx_bindings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.extract_vfx import FLIPBOOK_EFFECTS, PARTICLE_EFFECTS, SPINE_EFFECTS
from scripts.vfx_bindings import (
    Binding,
    mentions_death_override,
    parse_card_file,
    parse_death_fade,
    parse_monster_file,
)

GAME_VERSION = "v0.107.1"

#: Effects this repo's extractor produces a renderable asset for — the exact
#: inventory `extract_vfx.py` writes to `manifest.json`, imported rather than
#: duplicated so this table can't silently drift from what the renderer can
#: actually draw (the same "single source of truth" reasoning as
#: `card_owner_id()` reading `card_ids_generated.rs` live instead of keeping
#: a second hand list). `vfx_chain` and `vfx_adrenaline` are in `SPINE_EFFECTS`
#: but are not selected by any real binding (see module docstring in
#: `extract_vfx.py` / design spec §4.3) so their presence here is harmless.
_RENDERABLE_EFFECTS = frozenset(
    name for effects in (FLIPBOOK_EFFECTS, SPINE_EFFECTS, PARTICLE_EFFECTS) for name, *_ in effects
)

#: The 13 particle composites this sub-project ships (design spec "Scope: 13
#: composites"). Always present in `effects`, `renderable: true`, even for
#: `vfx_sandy_impact`, which has zero real call sites anywhere in the decomp
#: (`sandyImpactPath` in `VfxCmd.cs:59` is dead) — the effect is real and
#: rendered, it just never gets picked, and the table must say so rather than
#: let it vanish.
_PARTICLE_EFFECT_NAMES = frozenset(name for name, _ in PARTICLE_EFFECTS)

#: Every non-renderable effect's declared substitute (task-6 review round 1:
#: `renderable: false` without a `fallback` forces the renderer to invent
#: one at runtime, which is exactly what the design rules out — design spec
#: §5's fallback chain reads this field first, before it ever falls through
#: to the generic `vfx_attack_blunt -> vfx_attack_slash` tail). This is
#: irreducibly a design choice, not something derivable the way
#: `card_owner_id()` is, so every entry below is hand-picked with its own
#: reason, grounded in the decomp rather than guessed from the name alone:
#: most sites pair their `WithHitFx`/`WithHitVfxNode` call with a sibling
#: `WithHitFx(null, null, "<family>_attack.mp3")` naming the game's own
#: attack-family sound, which is stronger evidence than the effect's name
#: (`vfx_dramatic_stab` and `vfx_stab_vfx` both play `blunt_attack.mp3`,
#: not a slash sound, despite "stab" sounding blade-like). Where no such
#: cue exists and no other decomp signal singles out a family, the default
#: is `vfx_attack_blunt` (design spec: the same substitute already declared
#: for the two deferred screen-space effects).
_DECLARED_FALLBACKS: dict[str, str] = {
    # --- deferred effects (Scope: 13 composites) -- BackBufferCopy screen
    # distortion belongs to sub-project 5, not a design judgment call.
    "vfx_heavy_blunt": "vfx_attack_blunt",
    "vfx_starry_impact": "vfx_attack_blunt",
    # --- bespoke long tail -- decomp-grounded family match ---
    # SovereignBlade.cs pairs `NBigSlashVfx`/`NBigSlashImpactVfx` (this
    # composite's own source) with `vfx_giant_horizontal_slash`, itself
    # tagged "slash_attack.mp3" in the same file.
    "vfx_big_slash": "vfx_attack_slash",
    "vfx_big_slash_impact": "vfx_attack_slash",
    # SovereignBlade.cs: same literal site, "slash_attack.mp3".
    "vfx_giant_horizontal_slash": "vfx_attack_slash",
    # DaggerSpray.cs/DaggerThrow.cs: a thrown dagger is a bladed weapon,
    # same family as vfx_giant_horizontal_slash/vfx_shiv_throw below; no
    # sibling WithHitFx site in either file to confirm by sound cue.
    "vfx_dagger_spray_impact": "vfx_attack_slash",
    # Shiv.cs: a thrown shiv is a bladed weapon; no sibling WithHitFx site
    # to confirm by sound cue.
    "vfx_shiv_throw": "vfx_attack_slash",
    # GoldAxe.cs/Backstab.cs both pair the literal "vfx/vfx_dramatic_stab"
    # site with "blunt_attack.mp3" (PoisonedStab.cs's site carries no
    # third argument, but 2 of 3 sites agree).
    "vfx_dramatic_stab": "vfx_attack_blunt",
    # Finisher.cs pairs `NStabVfx` with a sibling
    # `.WithHitFx(null, null, "blunt_attack.mp3")` on the very next line.
    "vfx_stab_vfx": "vfx_attack_blunt",
    # GrandFinale.cs pairs `NGrandFinaleImpactVfx` with a sibling
    # `.WithHitFx(null, null, "blunt_attack.mp3")` on the very next line.
    "vfx_grand_finale_impact": "vfx_attack_blunt",
    # MoltenFist.cs: literal site tagged "blunt_attack.mp3" (a fist is a
    # blunt strike either way).
    "vfx_molten_fist": "vfx_attack_blunt",
    # GiantRock.cs: literal site tagged "blunt_attack.mp3".
    "vfx_rock_shatter": "vfx_attack_blunt",
    # GlobeHead.cs's GalvanicBurstMove: literal site tagged
    # "blunt_attack.mp3" (its other site, CastMove, carries no sound arg).
    "vfx_attack_lightning": "vfx_attack_blunt",
    # Corrected in task-6 review round 2: every `NSpikeSplashVfx.Create`
    # WithHitVfxNode site (the shape this fallback actually governs) chains
    # with a slash-family effect in the SAME BUILDER, not a separate hit --
    # CeremonialBeast.cs:241-242 (StompMove) and :256-257 (CrushMove) chain
    # `.WithHitFx("vfx/vfx_attack_slash")` with the same `WithHitVfxNode`;
    # MechaKnight.cs:100-102 (ChargeMove) chains it directly with
    # `.WithHitVfxNode(... NBigSlashImpactVfx.Create(...))`. (CrushUnder.cs/
    # Stomp.cs's `vfx_heavy_blunt` co-occurrence, the original basis for
    # "blunt", is the weaker same-file-not-same-chain kind of evidence --
    # there `NSpikeSplashVfx.Create` is a free-standing spawn and the blunt
    # tag belongs to an unrelated hit on the same card, not this call.)
    "vfx_spike_splash_vfx": "vfx_attack_slash",
    # Thrash.cs: no sibling sound cue, but the card is a flailing strike
    # and Crusher's own canonical ThrashMove already maps its literal
    # WithHitFx site to vfx_attack_blunt (see test_vfx_bindings.py's
    # CRUSHER fixture) -- the same family by the game's own precedent.
    "vfx_thrash": "vfx_attack_blunt",
    # KinPriest.cs: a grenade explosion has no slash affinity; default.
    "vfx_kin_priest_grenade_vfx": "vfx_attack_blunt",
    # DevotedSculptor.cs/TerrorEel.cs/TheInsatiable.cs/CeremonialBeast.cs:
    # a vocal/scream cue, not an attack motion at all; default.
    "vfx_scream": "vfx_attack_blunt",
    "vfx_spooky_scream": "vfx_attack_blunt",
    # VineShambler.cs:63 (GraspingVinesMove) — task-7's engine coverage gate
    # (`vfx_binding_coverage.rs`) surfaced this site: the `WithAttackerFx`
    # sibling sound is "vine_shambler_cast" (a named, non-generic cue), so
    # unlike the sites above there is no "*_attack.mp3" family tag to read;
    # default, same as vfx_thrash's "no cue" case.
    "vfx_vine_shambler_vines_vfx": "vfx_attack_blunt",
}


#: C# class whose `GenerateMoveStateMachine()` actually owns an engine
#: monster's moves, when it differs from the engine's own monster id.
#:
#: Published as table DATA rather than kept in one consumer, because the
#: table now has TWO consumers that must agree: `vfx_binding_coverage.rs`
#: (the gate that proves every engine move id resolves to a row) and
#: `sts2-pycli/web/src/effects/bindings.ts` (the workbench renderer that
#: actually looks a row up at runtime). They were out of step: the gate
#: applied both alias tables below, the renderer applied neither, so ~30
#: (monster, move) pairs silently drew the generic slash while the gate
#: stayed green — the exact failure the gate exists to prevent. One
#: published table cannot drift that way.
#:
#: The engine ids are `MonsterKind::python_name()` (what the combat log
#: writes as `source_monster`); the table ids are decomp FILE stems, which
#: is what `parse_monster_file` keys rows on. Every entry was confirmed by
#: reading the C# file, not inferred from the name.
_MONSTER_ALIASES: dict[str, tuple[str, str]] = {
    "MysteriousKnight": (
        "FlailKnight",
        "MysteriousKnight.cs:8 is `class MysteriousKnight : FlailKnight` with no "
        "GenerateMoveStateMachine() override, so it plays FlailKnight's moves; the "
        "extractor walks per-file, so no MysteriousKnight:* row exists or ever will",
    ),
    "DecimillipedeSegmentFront": (
        "DecimillipedeSegment",
        "the three segment subclasses override only SegmentAttack(), never "
        "GenerateMoveStateMachine(), and none declares a MoveState — every binding "
        "is filed under the base class",
    ),
    "DecimillipedeSegmentMiddle": (
        "DecimillipedeSegment",
        "same as DecimillipedeSegmentFront",
    ),
    "DecimillipedeSegmentBack": (
        "DecimillipedeSegment",
        "same as DecimillipedeSegmentFront",
    ),
    "DenseVegetationWriggler": (
        "Wriggler",
        "an engine-only slot variant of Wriggler; there is no DenseVegetationWriggler.cs at all",
    ),
}

#: `<monster>:<engine state id>` -> the real C# state id, for engine ids that
#: are a genuine RENAME rather than a `_MOVE`-suffix difference (suffixes are
#: handled mechanically by both consumers: try `<id>_MOVE`, then the bare
#: `<id>`). The monster half is the key AFTER `_MONSTER_ALIASES` is applied.
#:
#: Every entry was confirmed by reading the C# `MoveState(...)` constructor
#: and its method body, and every one of those bodies really does call
#: `WithHitFx`/`WithHitVfxNode` — an engine rename whose C# move has no
#: hit-vfx call at all is deliberately absent, since aliasing to a row that
#: does not exist would only be confusing (`vfx_binding_coverage.rs`'s
#: `EXEMPT` records those instead, with the real id in its reason).
_STATE_ALIASES: dict[str, tuple[str, str]] = {
    "LeafSlimeS:BUTT": (
        "TACKLE_MOVE",
        "LeafSlimeS.cs declares only TACKLE_MOVE; engine names the animal action",
    ),
    "TwigSlimeS:BUTT": (
        "TACKLE_MOVE",
        "TwigSlimeS.cs declares only TACKLE_MOVE; engine names the animal action",
    ),
    "TwigSlimeM:CLUMP_SHOT": (
        "POKEY_POUNCE_MOVE",
        "TwigSlimeM.cs:36 — the engine id matches the METHOD name (ClumpShotMove), "
        "not the state id",
    ),
    "CubexConstruct:REPEATER": (
        "REPEATER_BLAST_MOVE",
        "CubexConstruct.cs:114 — engine drops _BLAST",
    ),
    "CubexConstruct:REPEATER_2": (
        "REPEATER_BLAST_MOVE_2",
        "CubexConstruct.cs:115 — engine drops _BLAST",
    ),
    "CubexConstruct:EXPEL_BLAST": (
        "EXPEL_MOVE",
        "CubexConstruct.cs:116 — engine moves the word BLAST onto this move instead",
    ),
    "SlitheringStrangler:TWACK": (
        "THWACK",
        "SlitheringStrangler.cs:43 declares THWACK; the engine id is a transcription typo, "
        "flagged here rather than renamed (state ids are internal, and renaming "
        "one is out of scope)",
    ),
    "Tunneler:BELOW_MOVE_1": (
        "BELOW_MOVE",
        "Tunneler.cs:74 — engine appends _1 though there is no BELOW_MOVE_2",
    ),
    "Exoskeleton:MANDIBLE": ("MANDIBLES_MOVE", "Exoskeleton.cs:56 — engine drops the S"),
    "SpinyToad:EXPLOSION": (
        "SPIKE_EXPLOSION_MOVE",
        "SpinyToad.cs:60 — engine keeps only the second word",
    ),
    "SpinyToad:LASH": ("TONGUE_LASH_MOVE", "SpinyToad.cs:61 — engine keeps only the second word"),
    "TheInsatiable:THRASH_1": ("THRASH_MOVE", "TheInsatiable.cs:97 — engine renumbers the pair"),
    "TheInsatiable:THRASH_2": ("THRASH_MOVE_2", "TheInsatiable.cs:98 — engine renumbers the pair"),
    "TurretOperator:UNLOAD_MOVE_1": (
        "UNLOAD_MOVE",
        "TurretOperator.cs:38 — C#'s first Unload carries no number; the engine numbers both",
    ),
    "MagiKnight:FIRST_POWER_SHIELD": (
        "POWER_SHIELD_MOVE",
        "MagiKnight.cs:60 — there is only one, so the engine's FIRST_ is not "
        "disambiguating a second",
    ),
    "MagiKnight:BOMB": ("MAGIC_BOMB", "MagiKnight.cs:63 — engine drops MAGIC_"),
    "TorchHeadAmalgam:TACKLE_1": (
        "TACKLE_MOVE",
        "TorchHeadAmalgam.cs:49 — C#'s first Tackle is bare (the 2nd/3rd/4th ARE "
        "numbered and need no alias)",
    ),
    "TerrorEel:ThrashMove": (
        "THRASH_MOVE",
        "TerrorEel.cs:72 — the engine id is the METHOD's exact casing, not a state-id spelling; "
        "every other TerrorEel state resolves cleanly, so this reads as a "
        "copy-paste slip in the port",
    ),
}


def _alias_rows(table: dict[str, tuple[str, str]], field: str) -> dict[str, dict[str, str]]:
    return {key: {field: value, "reason": reason} for key, (value, reason) in table.items()}


def _binding_row(b: Binding) -> dict[str, str | None]:
    return {"owner": b.owner, "kind": b.kind, "effect": b.effect, "tint": b.tint, "shape": b.shape}


def build_table(decompiled_root: Path) -> dict:
    """The full binding table for `decompiled_root` (a `decompiled/<version>`
    directory), in the exact shape written to `vfx_bindings.json`.

    Deterministic: `Path.glob` results are sorted by file name (a str sort:
    `WindowsPath` ordering is case-insensitive, Linux is not) before scanning, and
    `vfx_bindings.py`'s regex scanning is itself order-preserving, so two
    calls against the same decomp text always produce byte-identical
    `bindings`. This determinism is what makes the re-extraction gate
    (`test_the_committed_table_matches_a_fresh_extraction`) meaningful rather
    than flaky.
    """
    cards_dir = decompiled_root / "MegaCrit.Sts2.Core.Models.Cards"
    monsters_dir = decompiled_root / "MegaCrit.Sts2.Core.Models.Monsters"

    bindings: list[Binding] = []
    death_fade: dict[str, dict[str, object]] = {}
    for cs_file in sorted(cards_dir.glob("*.cs"), key=lambda p: p.name):
        bindings.extend(parse_card_file(cs_file.stem, cs_file.read_text()))
    for cs_file in sorted(monsters_dir.glob("*.cs"), key=lambda p: p.name):
        text = cs_file.read_text()
        bindings.extend(parse_monster_file(cs_file.stem, text))
        row = parse_death_fade(cs_file.stem, text)
        if row is not None:
            death_fade[cs_file.stem] = row
        elif mentions_death_override(text):
            # Completeness, not tidiness. A file that TALKS about either
            # override but parses to nothing means the C# changed shape, and
            # the silent outcome would be a monster dissolving where the game
            # leaves it alone — invisible in every test that checks the
            # extractor rather than its output, which is the failure mode this
            # pipeline has now shipped four times.
            raise ValueError(
                f"{cs_file.stem}: mentions ShouldFadeAfterDeath or "
                "DeathAnimLengthOverride but neither parsed — the decomp's "
                "shape changed and `parse_death_fade` needs updating"
            )

    effects: dict[str, dict[str, object]] = {
        name: {"renderable": True} for name in sorted(_PARTICLE_EFFECT_NAMES)
    }
    for name, fallback in _DECLARED_FALLBACKS.items():
        effects[name] = {"renderable": False, "fallback": fallback}
    for b in bindings:
        if b.effect in effects:
            continue
        effects[b.effect] = {"renderable": b.effect in _RENDERABLE_EFFECTS}

    return {
        "version": GAME_VERSION,
        "effects": dict(sorted(effects.items())),
        "bindings": [_binding_row(b) for b in bindings],
        "aliases": {
            "monsters": _alias_rows(_MONSTER_ALIASES, "monster"),
            "states": _alias_rows(_STATE_ALIASES, "state"),
        },
        "deathFade": dict(sorted(death_fade.items())),
    }


#: Where the workbench frontend actually fetches the table from — same
#: convention as `extract_vfx.py`'s `manifest.json`: unversioned, under
#: `site/public/images/`, so `assetUrl("vfx/vfx_bindings.json")` (`?v=`
#: cache-busted) resolves in local dev (mounted straight off disk) AND in
#: the Cloud Run deploy (synced to the art bucket by `build_asset_manifest.py`
#: walking this exact tree — see `sts2-pycli/web/src/effects/bindings.ts`).
#: `data/{version}/vfx_bindings.json` stays the reviewed, re-extraction-gated
#: source of truth (`test_extract_vfx_bindings.py`); this is a published copy
#: of it, kept from drifting by writing both from the same `table` object in
#: one run rather than requiring a second manual step.
_SITE_PUBLISH_PATH = (
    Path(__file__).resolve().parents[1] / "site" / "public" / "images" / "vfx" / "vfx_bindings.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decompiled_root", type=Path, help="e.g. decompiled/v0.107.1")
    parser.add_argument("out", type=Path, help="e.g. data/v0.107.1/vfx_bindings.json")
    args = parser.parse_args()

    table = build_table(args.decompiled_root)
    text = json.dumps(table, indent=2) + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}: {len(table['bindings'])} bindings, {len(table['effects'])} effects")

    _SITE_PUBLISH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SITE_PUBLISH_PATH.write_text(text)
    print(f"wrote {_SITE_PUBLISH_PATH} (published copy for the workbench frontend)")


if __name__ == "__main__":
    main()
