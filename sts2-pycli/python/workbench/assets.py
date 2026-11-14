"""Resolve STS2 game keys to image paths under the extracted art root.

Mirrors the sts2-wiki naming convention (card_atlas/<character>/<slug>.png,
monsters/<slug>.png, intent_atlas/..., power_atlas/..., sprite_fonts/...).
Returns paths relative to the configured root, or None when absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Apostrophes are dropped (not converted to '_') so "Pael's Legion" → "paels_legion".
_APOSTROPHE = re.compile(r"['’]")

# Event titles whose slug differs from the extracted illustration filename.
_EVENT_ALIASES = {
    "reflections_snoitcelfer": "reflections",
    "welcome_to_wongo_s": "welcome_to_wongos",
}

# Monster display names whose slug differs from the extracted sprite filename.
# Keyed by the slugified *display* name (what the engine sends as ``name``),
# valued by the extracted art slug. Mirrors ``MONSTER_IMAGE_ALIASES`` in the
# web frontend (``sts2-wiki/site/src/utils.ts``); keep the two in sync.
# ``tests/workbench/test_monster_rig_coverage.py`` fails if a modeled monster
# stops resolving to art, catching future drift.
_MONSTER_ALIASES = {
    "flyconid": "flying_mushrooms",
    "cultist": "cultists",
    "assassin_raider": "assassin_ruby_raider",
    "axe_raider": "axe_ruby_raider",
    "brute_raider": "brute_ruby_raider",
    "crossbow_raider": "crossbow_ruby_raider",
    "tracker_raider": "tracker_ruby_raider",
    # Bowlbug variants share the base sprite.
    "bowlbug_egg": "bowlbug",
    "bowlbug_nectar": "bowlbug",
    "bowlbug_rock": "bowlbug",
    "bowlbug_silk": "bowlbug",
    # Crusher (left arm) and Rocket (right arm) are the two targetable parts of
    # the Kaiser Crab boss; both share its sprite/rig.
    "crusher": "kaiser_crab",
    "rocket": "kaiser_crab",
    # Hatchling is Tough Egg's post-hatch display title (ToughEgg.cs overrides
    # Title to HATCHLING.name after hatching) — same creature, same art.
    "hatchling": "tough_egg",
    # Monsters whose sprite is filed under a different name.
    "torch_head_amalgam": "amalgam",
    "globe_head": "orb_head",
    "living_fog": "living_smog",
    "skulking_colony": "living_shield",
    "ovicopter": "egg_layer",
    # Cultist variants share the base "cultists" sprite/rig.
    "calcified_cultist": "cultists",
    "damp_cultist": "cultists",
    # Mysterious Knight is a code subclass of Flail Knight (buffed disguise, no
    # own art) — MysteriousKnight.cs : FlailKnight — so it uses the flail_knight
    # sprite/rig.
    "mysterious_knight": "flail_knight",
    # The Merchant??? (FakeMerchantMonster) — its combat rig is filed under the
    # slug fake_merchant_monster (source skel fake_merchant_top; matches its
    # creature_scales key), while its display name slugs to bare "merchant".
    "merchant": "fake_merchant_monster",
    # Aeonglass has no rig in v0.107.1 — the game itself renders it as a static
    # hourglass placeholder sprite (scenes/creature_visuals/aeonglass.tscn).
    "aeonglass": "hourglass_placeholder",
    # Battle Friend / dummy variants share the battleworn dummy sprite/rig.
    "battle_friend_v1_0": "battleworn_dummy",
    "battle_friend_v2_0": "battleworn_dummy",
    "battle_friend_v3_0": "battleworn_dummy",
    "big_dummy": "battleworn_dummy",
}

# Monsters that are TWO (or more) targetable combatants sharing ONE big Spine rig
# (the game renders them as a single skeleton and drives each part on its own
# animation track — see ``NKaiserCrabBossBackground``). Keyed by the slugified
# display name, valued by ``(group, part)``: ``group`` ties the shared rig
# together (members with the same group are one skeleton); ``part`` selects the
# animation-track family (``left/*`` / ``right/*``) that combatant drives. The
# frontend renders the rig ONCE for the group and pins each member's HP/intent
# overlay over its claw, instead of drawing a whole crab per arm.
_MONSTER_RIG_GROUPS = {
    "crusher": ("kaiser_crab", "left"),  # Kaiser Crab left arm
    "rocket": ("kaiser_crab", "right"),  # Kaiser Crab right arm
}

# Monsters whose rig must be picked by their canonical KIND, not their display
# name — the three Decimillipede segments all localize to the single name
# "Decimillipede" (so a name lookup can't tell them apart), but each has its own
# Spine rig (front/middle/back). Keyed by ``MonsterKind`` (the ``kind`` field on
# the wire enemy, PascalCase, matching ``display_maps.MONSTER_ID_TO_TITLE``).
_MONSTER_KIND_RIGS = {
    "DecimillipedeSegmentFront": "decimillipede_segment_front",
    "DecimillipedeSegmentMiddle": "decimillipede_segment_middle",
    "DecimillipedeSegmentBack": "decimillipede_segment_back",
}

# Multi-part creatures keyed by KIND (when the display name can't distinguish the
# parts). ``(group, part)`` — the frontend renders the group's parts as one
# connected creature (the Decimillipede's three segment rigs tiled into one body)
# with per-part targetable overlays, like the Kaiser Crab.
_MONSTER_KIND_GROUPS = {
    "DecimillipedeSegmentFront": ("decimillipede", "front"),
    "DecimillipedeSegmentMiddle": ("decimillipede", "middle"),
    "DecimillipedeSegmentBack": ("decimillipede", "back"),
}


def normalize_kind(kind: str | None) -> str | None:
    """The wire's monster kind as the ``MonsterKind`` the tables above are keyed by.

    THE TWO PRODUCERS SPELL IT DIFFERENTLY. The engine emits the PascalCase
    ``MonsterKind`` (``state_dict.rs``: ``species.python_name()`` ->
    ``"DecimillipedeSegmentFront"``); SpireMentor emits the game's own
    ``Creature.Monster.Id.Entry``, which is SCREAMING_SNAKE
    (``"DECIMILLIPEDE_SEGMENT_FRONT"``, verified against the raw v2 captures in
    ``logs/realgame/``). Keying the tables on one spelling silently no-ops the
    whole kind-addressed rig path for the other producer, which is how the
    Decimillipede rendered as three sprite fallbacks on every replayed live run
    while rendering correctly in a workbench-played one.

    A kind carrying any lowercase letter is already a ``MonsterKind`` and is
    returned untouched, so this can never rewrite a name it does not recognise.

    Args:
        kind: Wire monster kind in either spelling, or None.

    Returns:
        The PascalCase kind, or None when ``kind`` is empty/None.
    """
    if not kind:
        return None
    if any(c.islower() for c in kind):
        return kind
    return "".join(part.capitalize() for part in kind.split("_") if part)


# Relic / potion display names whose slug differs from the extracted atlas
# filename. Keyed by the slugified DISPLAY name, valued by the art slug.
_RELIC_ALIASES = {
    # FakeMerchantsRug's title is "The Merchant's Rug???"; the sprite keeps the
    # class name's "fake_" prefix like every other Merchant??? counterfeit.
    "merchants_rug": "fake_merchants_rug",
}
_POTION_ALIASES = {
    # Clarity's title is "Clarity Extract"; the sprite is filed under the id.
    "clarity_extract": "clarity",
}


def slugify(name: str) -> str:
    """Lowercase, drop a leading 'the', collapse non-alphanumerics to '_'."""
    s = _APOSTROPHE.sub("", name.strip().lower())
    s = _NON_ALNUM.sub("_", s).strip("_")
    if s.startswith("the_"):
        s = s[len("the_") :]
    return s


def slug_forms(name: str) -> tuple[str, str]:
    """``(stripped, raw)`` slugs for ``name`` — with and without a leading "the".

    :func:`slugify` drops the article, but only SOME of the extracted art trees
    do: ``monsters/`` is generated with the strip (``monsters/kin.png``) while
    ``relic_atlas/`` and ``power_atlas/`` keep it (``the_boot.png``,
    ``the_bomb_power.png``). A resolver that tries one form only is silently
    iconless for every entity whose name starts with "The" — that was The Boot,
    The Abacus, The Courier and four powers.
    """
    raw = _APOSTROPHE.sub("", name.strip().lower())
    raw = _NON_ALNUM.sub("_", raw).strip("_")
    return slugify(name), raw


class AssetResolver:
    """Maps game entities to existing image files under ``root``.

    Two existence-resolution modes:

    * **filesystem** (default) — ``root`` is the extracted art tree on disk and
      existence is a ``Path.is_file()`` check. This is the local-dev path.
    * **manifest** — ``manifest`` is the set of relative paths that exist in the
      art store (e.g. a GCS bucket). Existence is a set-membership check, so the
      art tree need NOT be present on disk. This is the lean / Cloud Run path:
      the 142 MB of art lives in a bucket, the container ships only the manifest
      (a few hundred KB). ``creature_scales.json`` is still read from ``root``
      when present, so ship that one small file alongside the manifest.

    Build a manifest with ``scripts/build_asset_manifest.py``.
    """

    def __init__(
        self, root: str | Path, *, manifest: set[str] | None = None, overlay: str | None = None
    ) -> None:
        self.root = Path(root)
        self._manifest = manifest
        # A game build's art overlay folder (``v0.111.0``), tried before the
        # shared tree on every lookup: the art that build added or redrew
        # (GH #801). None for the default build. Twin of the Rust
        # ``AssetResolver::with_overlay``.
        self._overlay = overlay
        # Per-creature authored scene scale + Bounds box (combat proportions).
        # Keyed by creature_visuals slug ("ironclad", "nibbit", ...). Empty when
        # the manifest is absent — combatants then size by their fallback height.
        self._scales: dict[str, dict[str, float]] = {}
        scales_path = self.root / "creature_scales.json"
        if scales_path.is_file():
            try:
                self._scales = json.loads(scales_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._scales = {}
        # The creatures the overlay's build added (import_game's art overlay
        # step writes their rows there); a shared row is never replaced.
        own_scales = self.root / overlay / "creature_scales.json" if overlay else None
        if own_scales is not None and own_scales.is_file():
            try:
                added = json.loads(own_scales.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                added = {}
            self._scales = {**added, **self._scales}

    def _rel_if_exists(self, rel: str) -> str | None:
        if self._overlay is not None:
            own = self._exists(f"{self._overlay}/{rel}")
            if own is not None:
                return own
        return self._exists(rel)

    def _exists(self, rel: str) -> str | None:
        if self._manifest is not None:
            return rel if rel in self._manifest else None
        return rel if (self.root / rel).is_file() else None

    def character_rig_scale(self, character: str) -> dict[str, float] | None:
        """Return ``{"scale", "bounds_w", "bounds_h"}`` for a character rig, or None.

        Drives the player↔enemy size proportions (every combatant is rendered
        at ``scale * global_factor`` px/unit).
        """
        return self._scales.get(slugify(character))

    def monster_rig_scale(self, name: str, kind: str | None = None) -> dict[str, float] | None:
        """Return ``{"scale", "bounds_w", "bounds_h"}`` for a monster rig, or None.

        Uses the same kind-first slug resolution as :meth:`monster_rig`, with
        one exception: A CREATURE THAT DOES NOT DRAW THE RIG IT IS SENT WITH
        CANNOT SIZE IT.

        ``scale`` is px per RIG unit, so it only means anything paired with the
        rig it was authored against. Normally the two agree — a creature drawing
        a shared rig is authored at its own scale on purpose (the Mysterious
        Knight draws the Flail Knight's rig at 0.17 against its 0.14, and is
        that much bigger). But the members of a SHARED-SKELETON group are not
        drawing that rig at all: the Kaiser Crab's Crusher and Rocket are two
        targetable arms of one boss whose body hangs off the combat-room
        background (``NKaiserCrabBossBackground``), and their own creature
        scenes are authored at 1.5 / -1.5 for visuals that are not it. Handing
        the frontend the crab rig at 1.5 drew the boss at three times its size,
        claws off both edges of the screen.

        ``bounds_w`` / ``bounds_h`` are the creature's own authored ``%Bounds``
        box and stay with the creature either way — ``bounds_w`` is exactly what
        ``NCombatRoom.PositionEnemies`` spaces the row by.
        """
        own = None
        for slug in self._rig_slugs(name, kind):
            if slug in self._scales:
                own = self._scales[slug]
                break
        group = self.monster_rig_group(name, kind)
        rig = self.monster_rig(name, kind)
        if group is not None and rig is not None:
            rig_slug = rig.rsplit("/", 1)[-1]
            # One skeleton for several creatures (the crab's arms), as opposed to
            # a group whose members each have their own rig (the Decimillipede's
            # segments, which resolve by kind and are authored for it).
            shared = rig_slug == group["group"]
            if shared and rig_slug in self._scales:
                sized = dict(own or {})
                sized["scale"] = self._scales[rig_slug]["scale"]
                return sized
        return own

    def _card_folders(self) -> list[str]:
        """All top-level atlas folders that exist under ``card_atlas/``.

        Used for the cross-character fallback scan. Works in both resolution
        modes: filesystem (list subdirectories of ``card_atlas/``) and manifest
        (the ``card_atlas/<folder>/…`` prefixes present in the manifest set).
        Sorted for deterministic resolution.
        """
        folders: set[str] = set()
        if self._manifest is not None:
            for rel in self._manifest:
                parts = rel.split("/")
                if len(parts) >= 3 and parts[0] == "card_atlas":
                    folders.add(parts[1])
        else:
            atlas = self.root / "card_atlas"
            if atlas.is_dir():
                folders = {c.name for c in atlas.iterdir() if c.is_dir()}
        return sorted(folders)

    def card(
        self,
        character: str,
        card_id: str,
        name: str,
        card_character: str | None = None,
    ) -> str | None:
        """Resolve a card to its atlas image path.

        The wiki names card art by the class name in snake_case (e.g.
        ``CARD.STRIKE_IRONCLAD`` → ``strike_ironclad.png``), so the ``card_id``
        is the reliable key; the human name is only a fallback.

        A run frequently holds cards from *another* character (Neow swaps,
        relics, cross-class rewards), whose art lives under that character's
        atlas folder, not the run character's. Resolution therefore searches, in
        order: the card's own character folder (``card_character``), the run
        character's folder, the shared status/curse/colorless/token/event/quest
        folders, and finally — when the card's character is unknown (e.g. a card
        missing from the pinned localization manifest) — every remaining
        character folder, so the image resolves instead of rendering blank.

        Args:
            character: Run character name (e.g. "Ironclad").
            card_id: Card identifier (e.g. "CARD.STRIKE_IRONCLAD").
            name: Human-readable card name (e.g. "Strike").
            card_character: The card's OWN character (e.g. "Defect" for a Claw
                held in an Ironclad run), when known. Searched first.

        Returns:
            Relative path string if the file exists, else None.
        """
        slugs = [s for s in (card_id.split(".")[-1].lower() if card_id else "", slugify(name)) if s]
        if not slugs:
            return None

        # Ordered, deduped primary search: the card's own character first, then
        # the run character, then the shared (non-character) atlas folders.
        folders: list[str] = []
        for c in (card_character, character):
            s = slugify(c) if c else ""
            if s and s not in folders:
                folders.append(s)
        for shared in ("status", "curse", "colorless", "token", "event", "quest"):
            if shared not in folders:
                folders.append(shared)

        def find(search: list[str]) -> str | None:
            # Beta-art cards live in a ``beta/`` subfolder under their atlas.
            for slug in slugs:
                for folder in search:
                    for sub in ("", "beta/"):
                        hit = self._rel_if_exists(f"card_atlas/{folder}/{sub}{slug}.png")
                        if hit:
                            return hit
            return None

        hit = find(folders)
        if hit:
            return hit

        # Last resort: the card's character was unknown or didn't match. Scan
        # every other character atlas folder so a cross-character card still
        # resolves. Precise matches above always win, so the rare bare-slug
        # collision (e.g. "scavenge" under two folders) only matters here.
        seen = set(folders)
        return find([f for f in self._card_folders() if f not in seen])

    def portrait(self, character: str) -> str | None:
        """Resolve a character's portrait icon path.

        Args:
            character: Character name (e.g. "Ironclad" or "The Ironclad").

        Returns:
            Relative path string if the file exists, else None.
        """
        return self._rel_if_exists(f"characters/character_icon_{slugify(character)}.png")

    def room_icon(self, node_type: str | None, boss_id: str | None = None) -> str | None:
        """Resolve a map node's room-type icon (the real in-game map badge).

        Mirrors the game's ``ImageHelper.GetRoomIconPath`` (C# decomp): the map
        draws a small per-room-type icon from ``images/ui/run_history/<slug>.png``
        keyed by the slugified room type, extracted here under ``run_history/``.

        Args:
            node_type: PascalCase map-point type as emitted by the Rust backend
                (``Monster`` / ``Elite`` / ``Shop`` / ``RestSite`` / ``Treasure``
                / ``Event`` / ``Ancient`` / ``Boss`` / ``Unknown``).
            boss_id: For ``Boss`` nodes, the ``UPPER_SNAKE`` encounter id
                (e.g. ``VANTOM`` / ``THE_INSATIABLE_BOSS``) so the specific boss
                icon can be resolved. Ignored for other types.

        Returns:
            Relative path string if the icon exists, else None (the frontend then
            falls back to its glyph).
        """
        if not node_type:
            return None
        if node_type == "Boss":
            if not boss_id:
                return None
            # Boss encounter ids vary on whether they already carry the ``_boss``
            # suffix (``THE_INSATIABLE_BOSS`` vs bare ``VANTOM``); the extracted
            # filenames always do (``vantom_boss.png``). Try the exact slug first,
            # then the ``_boss``-suffixed form.
            slug = boss_id.lower()
            for cand in (slug, f"{slug}_boss"):
                rel = self._rel_if_exists(f"run_history/{cand}.png")
                if rel is not None:
                    return rel
            return None
        # ``MegaElite`` (if ever surfaced) reuses the elite badge; ``Rest`` /
        # ``Campfire`` are UI aliases of ``RestSite``.
        slug = {
            "Monster": "monster",
            "Elite": "elite",
            "MegaElite": "elite",
            "Shop": "shop",
            "RestSite": "rest_site",
            "Rest": "rest_site",
            "Campfire": "rest_site",
            "Treasure": "treasure",
            "Event": "event",
            "Ancient": "ancient",
        }.get(node_type)
        if slug is None:
            return None
        return self._rel_if_exists(f"run_history/{slug}.png")

    def character_sprite(self, character: str) -> str | None:
        """Resolve a character's full-body combat sprite (baked Spine idle cutout).

        These are the real in-game character bodies, baked from the Spine rigs in
        the game's .pck (see ``sts2-wiki/scripts/extract_rig.py`` +
        ``tools/spine-renderer``). Falls back to the class-icon portrait when the
        cutout has not been baked for a character.

        Args:
            character: Character name (e.g. "Ironclad" or "The Ironclad").

        Returns:
            Relative path string if the cutout exists, else None.
        """
        return self._rel_if_exists(f"character_sprites/{slugify(character)}.png")

    def character_rig(self, character: str) -> str | None:
        """Resolve the base path of a character's live Spine rig (no extension).

        Returns ``character_rigs/<slug>/<slug>`` when the ``.skel`` exists; the
        frontend appends ``.skel`` / ``.atlas`` / ``.png`` to load and animate it
        with spine-webgl. Returns None when no rig has been staged (the combatant
        then falls back to the static cutout / portrait).

        Args:
            character: Character name (e.g. "Ironclad" or "The Ironclad").
        """
        slug = slugify(character)
        skel = self._rel_if_exists(f"character_rigs/{slug}/{slug}.skel")
        return skel.removesuffix(".skel") if skel else None

    def monster(self, name: str) -> str | None:
        """Resolve a monster name to its sprite image path.

        Args:
            name: Monster name (e.g. "Scroll of Biting").

        Returns:
            Relative path string if the file exists, else None.
        """
        # ``slugify`` strips a leading ``the_`` (so "The Forgotten" → "forgotten"),
        # but some sprite files keep it (``the_forgotten.png``). Try both forms,
        # then the alias table.
        stripped = slugify(name)
        raw = _APOSTROPHE.sub("", name.strip().lower())
        raw = _NON_ALNUM.sub("_", raw).strip("_")
        for slug in (stripped, raw, _MONSTER_ALIASES.get(stripped), _MONSTER_ALIASES.get(raw)):
            if slug:
                hit = self._rel_if_exists(f"monsters/{slug}.png")
                if hit:
                    return hit
        return None

    def _rig_slugs(self, name: str, kind: str | None) -> list[str]:
        """Candidate rig slugs for a monster, kind-specific ones first.

        ``kind`` (the wire ``MonsterKind``) wins over ``name`` so segmented
        monsters whose segments share one display name (Decimillipede) still
        resolve to their distinct per-segment rigs. Either producer's spelling
        is accepted -- see :func:`normalize_kind`.
        """
        slugs: list[str] = []
        norm = normalize_kind(kind)
        if norm and norm in _MONSTER_KIND_RIGS:
            slugs.append(_MONSTER_KIND_RIGS[norm])
        stripped = slugify(name)
        raw = _APOSTROPHE.sub("", name.strip().lower())
        raw = _NON_ALNUM.sub("_", raw).strip("_")
        slugs += [stripped, raw, _MONSTER_ALIASES.get(stripped), _MONSTER_ALIASES.get(raw)]
        return [s for s in slugs if s]

    def monster_rig(self, name: str, kind: str | None = None) -> str | None:
        """Resolve the base path of a monster's live Spine rig (no extension).

        Returns ``monster_rigs/<slug>/<slug>`` when the ``.skel`` exists; the
        frontend appends ``.skel`` / ``.atlas`` / ``.png`` to animate it. Uses
        the kind-specific rig (Decimillipede segments) when ``kind`` is given,
        then the same slug variants + alias table as :meth:`monster`. Returns
        None when no rig has been staged (the combatant falls back to the sprite).

        Args:
            name: Monster name (e.g. "Cubex Construct").
            kind: Canonical ``MonsterKind`` (e.g. "DecimillipedeSegmentFront"),
                used to disambiguate segments that share one display name.
        """
        for slug in self._rig_slugs(name, kind):
            skel = self._rel_if_exists(f"monster_rigs/{slug}/{slug}.skel")
            if skel:
                return skel.removesuffix(".skel")
        return None

    def monster_rig_group(self, name: str, kind: str | None = None) -> dict[str, str] | None:
        """Return ``{"group", "part"}`` for a multi-part monster, or None.

        ``group`` ties combatants that form one connected creature; ``part`` is
        which piece this combatant is. Two flavours: the Kaiser Crab (Crusher
        left / Rocket right) shares ONE skeleton and ``part`` selects its
        animation track; the Decimillipede shares a ``group`` but each segment
        (front/middle/back) has its OWN rig. The frontend renders a group as one
        connected visual with per-part targetable overlays. ``kind`` (MonsterKind)
        wins over ``name`` for segments that share a display name. None for
        ordinary monsters.

        Args:
            name: Monster name (e.g. "Crusher").
            kind: Canonical ``MonsterKind`` (e.g. "DecimillipedeSegmentFront").
        """
        norm = normalize_kind(kind)
        if norm and norm in _MONSTER_KIND_GROUPS:
            group, part = _MONSTER_KIND_GROUPS[norm]
            return {"group": group, "part": part}
        stripped = slugify(name)
        raw = _APOSTROPHE.sub("", name.strip().lower())
        raw = _NON_ALNUM.sub("_", raw).strip("_")
        for slug in (stripped, raw):
            hit = _MONSTER_RIG_GROUPS.get(slug)
            if hit:
                return {"group": hit[0], "part": hit[1]}
        return None

    def energy(self, character: str) -> str | None:
        """Resolve a character's energy gem icon path.

        Args:
            character: Character name (e.g. "Ironclad").

        Returns:
            Relative path string if the file exists, else None.
        """
        return self._rel_if_exists(f"sprite_fonts/{slugify(character)}_energy_icon.png")

    def intent(self, intent_type: str, total_damage: int | None = None) -> str | None:
        """Resolve an enemy intent type to its icon path.

        Attack intents have no flat icon — the game (``AttackIntent.GetTexture``)
        picks one of five daggers by *total* telegraphed damage:
        ``<5 → 1, <10 → 2, <20 → 3, <40 → 4, ≥40 → 5`` (in
        ``intent_atlas/attack/``). All other intent types use the single
        top-level ``intent_<type>.png``.

        Args:
            intent_type: Intent type string (e.g. "Attack", "Defend").
            total_damage: Total telegraphed damage (damage × hits) for attacks;
                selects the dagger tier. Ignored for non-attack intents.

        Returns:
            Relative path string if the file exists, else None.
        """
        if slugify(intent_type) == "attack":
            dmg = total_damage or 0
            tier = 1 if dmg < 5 else 2 if dmg < 10 else 3 if dmg < 20 else 4 if dmg < 40 else 5
            return self._rel_if_exists(f"intent_atlas/attack/intent_attack_{tier}.png")
        return self._rel_if_exists(f"intent_atlas/intent_{slugify(intent_type)}.png")

    def power(self, name: str) -> str | None:
        """Resolve a power/buff name to its icon path.

        Args:
            name: Power name (e.g. "Strength").

        Returns:
            Relative path string if the file exists, else None.
        """
        for slug in slug_forms(name):
            hit = self._rel_if_exists(f"power_atlas/{slug}_power.png")
            if hit:
                return hit
        return None

    def power_big(self, name: str) -> str | None:
        """Resolve a power name to its 256x256 ``BigIcon``.

        The game keeps two icons per power and uses them in different places.
        ``power()`` above returns ``PackedIconPath`` -- the 64x64 atlas sprite
        the status TRAY draws. This returns ``BigIconPath``
        (``images/powers/<id>.png``), which is what the power VFX assign to
        their own sprite nodes: ``NPowerAppliedVfx`` (``_icon``/``_iconEcho``),
        ``NPowerRemovedVfx`` and ``NPowerFlashVfx``.

        Mirrors ``PowerModel.ResolvedBigIconPath``'s fallback chain by ending
        at ``missing_power.png``, which is not a defensive nicety: 14 powers
        have a tray icon and no big icon in the pck, and the game shows the
        placeholder for exactly those.

        Args:
            name: Power name (e.g. "Strength").

        Returns:
            Relative path string, or None when even the placeholder is absent
            (an art tree extracted before this pass existed).
        """
        for slug in slug_forms(name):
            hit = self._rel_if_exists(f"power_big/{slug}_power.png")
            if hit:
                return hit
        return self._rel_if_exists("power_big/missing_power.png")

    def relic(self, name: str) -> str | None:
        """Resolve a relic name to its atlas image path.

        Args:
            name: Relic name (e.g. "Akabeko").

        Returns:
            Relative path string if the file exists, else None.
        """
        stripped, raw = slug_forms(name)
        for slug in (stripped, raw, _RELIC_ALIASES.get(stripped)):
            if slug:
                hit = self._rel_if_exists(f"relic_atlas/{slug}.png")
                if hit:
                    return hit
        return None

    def potion(self, name: str) -> str | None:
        """Resolve a potion name to its atlas image path.

        Args:
            name: Potion name (e.g. "Fire Potion").

        Returns:
            Relative path string if the file exists, else None.
        """
        stripped, raw = slug_forms(name)
        for slug in (stripped, raw, _POTION_ALIASES.get(stripped)):
            if slug:
                hit = self._rel_if_exists(f"potion_atlas/{slug}.png")
                if hit:
                    return hit
        return None

    def orb(self, orb_type: str) -> str | None:
        """Resolve a Defect orb to its atlas image path.

        The game stores orb icons as ``orbs/<entry>.png`` (e.g.
        ``orbs/lightning.png``); the wiki mirrors that under ``orb_atlas/``
        keyed by the lowercased orb type. The combat state carries the type as
        the C# class name minus ``Orb`` (``"Lightning"``, ``"Frost"``,
        ``"Dark"``, ``"Plasma"``, ``"Glass"``).

        Args:
            orb_type: Orb type string (e.g. "Lightning").

        Returns:
            Relative path string if the file exists, else None.
        """
        return self._rel_if_exists(f"orb_atlas/{slugify(orb_type)}.png")

    def event_illustration(self, event_name: str) -> str | None:
        """Resolve an event's display title to its illustration path.

        The live state carries ``event_name`` as the localized title (e.g.
        "This or That?"). Illustration files are named by the event's internal
        slug; ``slugify`` matches most directly, but some files keep a leading
        ``the_`` while :func:`slugify` strips it, so both forms are tried, plus
        a small alias table for the handful that differ. Events with no
        extracted illustration (special/text-only events) return ``None`` and
        the frontend shows a styled dark panel.

        Args:
            event_name: Localized event title (e.g. "This or That?").

        Returns:
            Relative path string if a matching file exists, else None.
        """
        if not event_name:
            return None
        raw = _NON_ALNUM.sub("_", event_name.strip().lower()).strip("_")
        stripped = raw[len("the_") :] if raw.startswith("the_") else raw
        for slug in (raw, stripped, _EVENT_ALIASES.get(stripped), _EVENT_ALIASES.get(raw)):
            if slug:
                hit = self._rel_if_exists(f"events/{slug}.png")
                if hit:
                    return hit
        return None
