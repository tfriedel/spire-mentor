#!/usr/bin/env python3
"""Extract all card data from STS2 decompiled code + localization."""

import argparse
import re
from collections.abc import Iterator

from scripts.common import (
    class_name_to_loc_key,
    find_loc_key,
    load_localization,
    parse_canonical_vars,
    parse_keywords,
    parse_referenced_powers,
    read_cs_files,
    write_json,
)

# Map pool file class names to character/pool labels
POOL_MAP: dict[str, str] = {
    "IroncladCardPool": "Ironclad",
    "SilentCardPool": "Silent",
    "DefectCardPool": "Defect",
    "NecrobinderCardPool": "Necrobinder",
    "RegentCardPool": "Regent",
    # v0.113.0's new character; an older decomp simply has no such pool.
    "FoolCardPool": "Fool",
    "ColorlessCardPool": "Colorless",
    "CurseCardPool": "Curse",
    "StatusCardPool": "Status",
    "TokenCardPool": "Token",
    "EventCardPool": "Event",
    "QuestCardPool": "Quest",
    "DeprecatedCardPool": "Deprecated",
    "MockCardPool": "Mock",
}


def build_card_pool_map(decompiled_dir: str) -> dict[str, str]:
    """Parse CardPool files to build class_name -> character mapping."""
    import os

    pools_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Models.CardPools")
    card_to_char: dict[str, str] = {}

    for class_name, content in read_cs_files(pools_dir):
        character = POOL_MAP.get(class_name)
        if not character:
            continue

        # Extract all ModelDb.Card<ClassName>() references
        for m in re.finditer(r"ModelDb\.Card<(\w+)>\(\)", content):
            card_class = m.group(1)
            # Don't override character pools with token/status/etc.
            if card_class not in card_to_char or character in (
                "Ironclad",
                "Silent",
                "Defect",
                "Necrobinder",
                "Regent",
                "Fool",
                "Colorless",
            ):
                card_to_char[card_class] = character

    return card_to_char


def parse_card_file(class_name: str, content: str) -> dict:
    """Parse a decompiled card .cs file to extract key properties."""
    card: dict = {"class_name": class_name}

    # Extract constructor: base(cost, CardType.X, CardRarity.Y, TargetType.Z)
    m = re.search(
        r":\s*base\((-?\d+),\s*CardType\.(\w+),\s*CardRarity\.(\w+),\s*TargetType\.(\w+)\)",
        content,
    )
    if m:
        card["energy_cost"] = int(m.group(1))
        card["type"] = m.group(2)
        card["rarity"] = m.group(3)
        card["target"] = m.group(4)

    # Check for X cost
    if "HasEnergyCostX => true" in content:
        card["x_cost"] = True

    # Extract star cost (Regent cards)
    star_match = re.search(r"CanonicalStarCost\s*=>\s*(\d+)", content)
    if star_match:
        card["star_cost"] = int(star_match.group(1))
    if "HasStarCostX => true" in content:
        card["x_star_cost"] = True

    # Extract keywords from CanonicalKeywords property
    keywords = parse_keywords(content)
    if keywords:
        card["keywords"] = keywords

    # Extract dynamic vars
    vars_found = parse_canonical_vars(content)
    if vars_found:
        card["vars"] = vars_found

    # Extract upgrade info
    upgrades: list[dict] = []
    upgrade_section = re.search(r"OnUpgrade\(\).*?\{(.*?)\n\t\}", content, re.DOTALL)
    if upgrade_section:
        body = upgrade_section.group(1)
        for um in re.finditer(r"(\w+)\.UpgradeValueBy\((-?\d+)m\)", body):
            upgrades.append({"var": um.group(1), "amount": int(um.group(2))})
        # Also match DynamicVars["Name"].UpgradeValueBy pattern
        for um in re.finditer(r'DynamicVars\["(\w+)"\]\.UpgradeValueBy\((-?\d+)m\)', body):
            upgrades.append({"var": um.group(1), "amount": int(um.group(2))})
        # Energy-cost upgrades. The engine spells this `base.EnergyCost.
        # UpgradeBy(N)` in every decomp we hold (v0.103.2 and v0.107.1 alike);
        # only the older `UpgradeEnergyCostBy(N)` was matched, so ZERO of the
        # 57 cost-reducing upgrades ever reached the manifest and `upgraded_cost`
        # was empty for every build. Its consumers -- the workbench deck view
        # and the rest-site upgrade preview -- therefore drew Stampede+ at its
        # un-upgraded 2 energy. The old spelling stays matched in case a build
        # predating our decomps is ever extracted.
        for um in re.finditer(r"(?:UpgradeEnergyCostBy|EnergyCost\.UpgradeBy)\((-?\d+)\)", body):
            upgrades.append({"var": "EnergyCost", "amount": int(um.group(1))})
        for km in re.finditer(r"RemoveKeyword\(CardKeyword\.(\w+)\)", body):
            upgrades.append({"action": "remove_keyword", "keyword": km.group(1)})
        for km in re.finditer(r"AddKeyword\(CardKeyword\.(\w+)\)", body):
            upgrades.append({"action": "add_keyword", "keyword": km.group(1)})
    if upgrades:
        card["upgrades"] = upgrades

    # Extract referenced powers
    powers = parse_referenced_powers(content)
    if powers:
        card["referenced_powers"] = powers

    return card


def compute_upgraded_vars(card: dict) -> list[dict]:
    """Compute upgraded values for each var based on upgrade info."""
    vars_list = card.get("vars", [])
    upgrades = card.get("upgrades", [])

    # Build a lookup of upgrade amounts by var name
    upgrade_amounts: dict[str, int] = {}
    for u in upgrades:
        if "var" in u:
            var_name = u["var"]
            # Map DynamicVars property names to var types
            # e.g., "Damage" -> "Damage", "Vulnerable" -> "Vulnerable"
            # The var name from UpgradeValueBy corresponds to the DynamicVars property
            upgrade_amounts[var_name] = u["amount"]

    result = []
    for v in vars_list:
        new_v = dict(v)
        vtype = v["type"]
        base = v["base_value"]

        # Try direct match first
        upgrade_amt = upgrade_amounts.get(vtype)
        # Try with "Power" suffix stripped (PowerVar<VulnerablePower> -> "Vulnerable" type,
        # but upgrade uses DynamicVars.Vulnerable)
        if upgrade_amt is None:
            upgrade_amt = upgrade_amounts.get(vtype + "Power")

        if upgrade_amt is not None:
            new_v["upgraded_value"] = base + upgrade_amt
        result.append(new_v)

    # A Calculated* var reads its value from CalculationBase, so upgrading the
    # base upgrades it too (CalculatedVar.RecalculateForUpgradeOrEnchant) —
    # without this, upgraded Death March still reads "Deal 8 damage".
    base_up = next(
        (v.get("upgraded_value") for v in result if v["type"] == "CalculationBase"), None
    )
    if base_up is not None:
        for v in result:
            if v["type"].startswith("Calculated") and "upgraded_value" not in v:
                v["upgraded_value"] = base_up

    return result


# Map of var type names to the placeholder names used in localization templates
VAR_TYPE_TO_PLACEHOLDER: dict[str, str] = {
    "Damage": "Damage",
    "CalculatedDamage": "Damage",
    "Block": "Block",
    "CalculatedBlock": "Block",
    "Energy": "Energy",
    "Cards": "Cards",
    "HpLoss": "HpLoss",
    "Summon": "Summon",
    "Forge": "Forge",
    "Repeat": "Repeat",
}


def _build_var_lookup(vars_list: list[dict], use_upgraded: bool) -> dict[str, int]:
    """Build a name->value lookup from vars list, handling multiple name forms."""
    lookup: dict[str, int] = {}
    aliases: dict[str, int] = {}
    for v in vars_list:
        vtype = v["type"]
        if use_upgraded and "upgraded_value" in v:
            raw_val = v.get("upgraded_value")
        else:
            raw_val = v["base_value"]
        val: int = int(raw_val) if raw_val is not None else 0
        lookup[vtype] = val
        # Power vars: "Vulnerable" -> also register "VulnerablePower"
        if not vtype.endswith("Power") and vtype not in VAR_TYPE_TO_PLACEHOLDER:
            aliases.setdefault(vtype + "Power", val)
        # Standard mapping: "CalculatedDamage" -> also "Damage"
        if vtype in VAR_TYPE_TO_PLACEHOLDER:
            aliases.setdefault(VAR_TYPE_TO_PLACEHOLDER[vtype], val)
    # Aliases are a fallback for templates that name the plain var when the card
    # only carries the calculated one — they must never shadow a var the card
    # actually declares, or Sovereign Blade's CalculatedBlock would overwrite the
    # Block its "{Block}" placeholder means.
    for key, val in aliases.items():
        lookup.setdefault(key, val)
    return lookup


def _iter_placeholders(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, inner)`` for each balanced top-level ``{...}`` span.

    The game's format strings nest (``{Repeat:plural:| {} times}``), so a
    ``{[^{}]*}`` scan cannot see the outer directive at all — it matches the
    inner ``{}``, fails to resolve it, and the whole directive falls through to
    the "X" fallback below. Balanced scanning is what lets the outer directive
    resolve first and its chosen branch render with that var in scope.
    """
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        yield i, j + 1, text[i + 1 : j]
                        i = j
                        break
            else:  # unbalanced tail — leave it alone
                return
        i += 1


def _split_branches(text: str, sep: str = "|") -> list[str]:
    """Split on ``sep`` at brace depth 0 (branches may contain placeholders)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _branch(rest: str, index: int) -> str:
    """Return branch ``index`` of an ``a|b`` directive body ("" when absent)."""
    parts = _split_branches(rest)
    return parts[index] if index < len(parts) else ""


# Card-level facts the game hands the formatter, from
# ``CardModel.GetDescriptionForPile``. The wiki renders a card as it reads at
# rest — outside combat, in no pile, with nothing targeted — so each of these
# takes its false branch. Resolving them rather than dropping them is the
# difference between "Deal 10 damage." and "Deal 10 damageXX.X".
_STATIC_CARD_FLAGS: dict[str, bool] = {
    "InCombat": False,
    "OnTable": False,
    "IsTargeting": False,
    "IsOstyAlive": False,
}


def _resolve_placeholder(
    inner: str,
    var_lookup: dict[str, int],
    upgraded: bool,
    ctx: dict[str, object],
    self_val: int | None = None,
) -> str:
    """Resolve one ``{...}`` body (braces already stripped).

    Handles the game's SmartFormat dialect: ``diff()`` / ``inverseDiff()`` value
    substitution, ``plural:``, ``cond:``, ``choose(...)``, ``IfUpgraded:show:``,
    the energy/star icon formatters, and boolean branch selection on the
    card-level flags above. Returns ``"{" + inner + "}"`` when nothing matches,
    so the caller's fallback can mark it.
    """
    unresolved = "{" + inner + "}"

    # ``{}`` / ``{:diff()}`` — SmartFormat's self-reference to the enclosing
    # selector's value, used inside plural branches ("{} times").
    if inner == "" or inner.startswith(":"):
        return str(self_val) if self_val is not None else unresolved

    name, _, fmt = inner.partition(":")
    name = name.strip()

    def render(text: str, val: int | None = None) -> str:
        return _render_template(text, var_lookup, upgraded, ctx, val)

    # {IfUpgraded:show:upgraded_text|base_text}
    if name == "IfUpgraded":
        rest = fmt[len("show:") :] if fmt.startswith("show:") else fmt
        return render(_branch(rest, 0 if upgraded else 1))

    # Boolean branch selection on a card-level flag: {Flag:then|else} and the
    # explicit {Flag:cond:then|else}.
    if name in _STATIC_CARD_FLAGS or isinstance(ctx.get(name), bool):
        rest = fmt[len("cond:") :] if fmt.startswith("cond:") else fmt
        flag = bool(ctx.get(name, _STATIC_CARD_FLAGS.get(name, False)))
        return render(_branch(rest, 0 if flag else 1))

    # {Value:choose(a|b|c):A|B|C} — pick the branch parallel to the matching
    # option, else the trailing default branch. ``TargetType`` is the card's own
    # target ("Deal 4 damage{TargetType:choose(AllEnemies): to ALL enemies|}.").
    m = re.match(r"choose\(([^)]*)\):(.*)", fmt, re.DOTALL)
    if m:
        options = [o.strip().lower() for o in m.group(1).split("|")]
        branches = _split_branches(m.group(2))
        value = ctx.get(name, var_lookup.get(name))
        chosen = str(value).strip().lower() if value is not None else None
        idx = options.index(chosen) if chosen in options else len(options)
        return render(branches[idx] if idx < len(branches) else branches[-1])

    val = var_lookup.get(name)

    # {Name:energyIcons()} and {Name:energyIcons(N)}
    if "energyIcons" in fmt:
        if val is not None:
            return f"{val} Energy"
        # energyPrefix is a display-only var (the number is in surrounding text)
        if name == "energyPrefix":
            return " Energy"
        m2 = re.search(r"energyIcons\((\d+)\)", fmt)
        if m2:
            return f"{m2.group(1)} Energy"
        return "Energy"

    # {Name:starIcons()}
    if "starIcons" in fmt:
        return f"{val} Stars" if val is not None else "Stars"

    # {singleStarIcon} — a literal icon reference, no formatter
    if name == "singleStarIcon":
        return "[star]"

    # {Name:inverseDiff()}
    if "inverseDiff" in fmt:
        return str(val) if val is not None else "?"

    # {Name:cond:>N?true_text|false_text}
    m = re.match(r"cond:>(\d+)\?(.*)", fmt, re.DOTALL)
    if m:
        rest = m.group(2)
        hit = val is not None and val > int(m.group(1))
        return render(_branch(rest, 0 if hit else 1), val)

    # {Name:plural:singular|plural}
    if fmt.startswith("plural:"):
        rest = fmt[len("plural:") :]
        # No value — the game always has one; the plural form reads better than
        # the singular for a description shown without a count.
        return render(_branch(rest, 0 if val == 1 else 1), val)

    # {Name:diff()} or a bare {Name}
    if val is not None:
        return str(val)

    return unresolved


def _render_template(
    text: str,
    var_lookup: dict[str, int],
    upgraded: bool,
    ctx: dict[str, object],
    self_val: int | None = None,
) -> str:
    """Substitute every balanced ``{...}`` in ``text``, outermost first."""
    out: list[str] = []
    pos = 0
    for start, end, inner in _iter_placeholders(text):
        out.append(text[pos:start])
        out.append(_resolve_placeholder(inner, var_lookup, upgraded, ctx, self_val))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def render_resolved(template: str, vars_list: list[dict]) -> str:
    """Substitute every DynamicVar in ``template`` with its base value.

    Rich-text tags are kept (the display layer renders them); anything the game
    only knows at runtime stays as its ``{Placeholder}``. This is the
    deterministic twin of the LLM-extracted per-item ``description`` — the
    workbench's relic/potion display text (``load_owned_localization``) — so
    the workbench bundle needs no LLM pass.
    """
    if not template:
        return ""
    rendered = _render_template(template, _build_var_lookup(vars_list, False), False, {})
    # A `{Name:cond:A|B}` whose value only exists at runtime (a picked card, a
    # starter relic) has no value here, which is the false branch.
    return re.sub(r"\{[^{}|]*:cond:[^{}|]*\|([^{}]*)\}", r"\1", rendered)


def render_description(
    template: str,
    vars_list: list[dict],
    use_upgraded: bool = False,
    ctx: dict[str, object] | None = None,
) -> tuple[str, str]:
    """Render a card description template with variable substitution.

    ``ctx`` carries the card-level facts the game's formatter reads that are not
    DynamicVars — ``TargetType`` and ``GainsBlock`` (see
    ``CardModel.GetDescriptionForPile``). Returns a (plain_text, html) tuple.
    """
    if not template:
        return ("", "")

    var_lookup = _build_var_lookup(vars_list, use_upgraded)
    rendered = _render_template(template, var_lookup, use_upgraded, ctx or {})

    # Convert rich text tags to HTML spans
    html = rendered
    from scripts.common import _COLOR_TAGS as color_tags

    for tag, css_class in color_tags.items():
        html = re.sub(
            rf"\[{tag}\](.*?)\[/{tag}\]",
            rf'<span class="{css_class}">\1</span>',
            html,
        )
    html = re.sub(r"\[/?(?:sine|wave|shake|b|i|jitter|center|thinky_dots)\]", "", html)
    # Convert newlines to <br>
    html = html.replace("\n", "<br>")

    # Plain text: convert icon tags to text, then strip remaining tags
    plain = rendered.replace("[star]", "★")
    plain = plain.replace("[energy]", "Energy")
    plain = re.sub(r"\[/?[^\]]*\]", "", plain)

    # Anything still unresolved is a value the game only knows at runtime — show
    # "X" for it rather than "?", which looks broken.
    plain = re.sub(r"\{[^{}]*\}", "X", plain)
    html = re.sub(r"\{[^{}]*\}", "X", html)

    return (plain, html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract STS2 card data")
    parser.add_argument("decompiled_dir", help="Path to decompiled source directory")
    parser.add_argument("loc_dir", help="Path to localization directory (eng/)")
    parser.add_argument("output_dir", help="Path to output data directory")
    args = parser.parse_args()

    import os

    decompiled_dir = os.path.expanduser(args.decompiled_dir)
    loc_dir = os.path.expanduser(args.loc_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # Load localization
    loc_data = load_localization(loc_dir, "cards")
    power_loc = load_localization(loc_dir, "powers")

    # Build power lookup: class_name -> {title, slug}
    power_lookup: dict[str, dict[str, str]] = {}
    for key in power_loc:
        if key.endswith(".title"):
            base_key = key.removesuffix(".title")
            title = power_loc[key]
            # Convert loc key like VULNERABLE_POWER to class name VulnerablePower
            # We store by multiple possible names for lookup
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            info = {"class_name": base_key, "title": title, "slug": slug}
            power_lookup[base_key] = info
            # Also store by common class name patterns
            # e.g., VULNERABLE_POWER -> VulnerablePower
            parts = base_key.split("_")
            camel = "".join(p.capitalize() for p in parts)
            power_lookup[camel] = info

    # Build epoch unlock map: card_class -> epoch info
    card_epoch_map: dict[str, str] = {}
    epochs_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Timeline.Epochs")
    if os.path.exists(epochs_dir):
        for epoch_name, epoch_content in read_cs_files(epochs_dir):
            story_m = re.search(r'StoryId\s*=>\s*"([^"]+)"', epoch_content)
            story = story_m.group(1) if story_m else epoch_name
            for cm in re.finditer(r"ModelDb\.Card<(\w+)>\(\)", epoch_content):
                card_epoch_map[cm.group(1)] = story

    # Build card pool mapping
    card_pool_map = build_card_pool_map(decompiled_dir)

    # Parse all card files
    cards_dir = os.path.join(decompiled_dir, "MegaCrit.Sts2.Core.Models.Cards")
    cards: list[dict] = []

    for class_name, content in read_cs_files(cards_dir):
        card = parse_card_file(class_name, content)

        # Skip if we couldn't parse the constructor (not a real card)
        if "energy_cost" not in card:
            continue

        # Character assignment
        card["character"] = card_pool_map.get(class_name, "Unknown")

        # Localization
        loc_key = find_loc_key(class_name, loc_data)
        if loc_key:
            card["loc_key"] = loc_key
            card["title"] = loc_data.get(f"{loc_key}.title", class_name)
            card["description_template"] = loc_data.get(f"{loc_key}.description", "")
        else:
            card["loc_key"] = class_name_to_loc_key(class_name)
            card["title"] = class_name
            card["description_template"] = ""
            card["_loc_missing"] = True

        # Compute upgraded vars
        card["vars"] = compute_upgraded_vars(card)

        # Compute upgraded cost
        if "upgrades" in card:
            for u in card["upgrades"]:
                if u.get("var") == "EnergyCost":
                    card["upgraded_cost"] = card["energy_cost"] + u["amount"]
                    break

        # Render descriptions. The formatter also reads two card-level
        # facts that are not DynamicVars (CardModel.GetDescriptionForPile):
        # the card's own TargetType, and whether it gains Block.
        ctx: dict[str, object] = {
            "TargetType": card.get("target"),
            "GainsBlock": "GainsBlock => true" in content,
        }
        plain, html = render_description(
            card.get("description_template", ""),
            card.get("vars", []),
            use_upgraded=False,
            ctx=ctx,
        )
        card["description_plain"] = plain
        card["description_html"] = html

        # Render upgraded description
        upgraded_plain, upgraded_html = render_description(
            card.get("description_template", ""),
            card.get("vars", []),
            use_upgraded=True,
            ctx=ctx,
        )
        if upgraded_plain != plain or card.get("upgraded_cost") is not None:
            card["upgraded_description_plain"] = upgraded_plain
            card["upgraded_description_html"] = upgraded_html

        # Enrich referenced_powers with display names and slugs
        if "referenced_powers" in card:
            enriched_powers: list[dict[str, str]] = []
            for power_class in card["referenced_powers"]:
                power_info = power_lookup.get(power_class)
                if power_info:
                    enriched_powers.append(power_info)
                else:
                    # Fallback: strip "Power" suffix and split CamelCase
                    display = power_class.removesuffix("Power")
                    display = re.sub(r"([a-z])([A-Z])", r"\1 \2", display)
                    slug = re.sub(r"[^a-z0-9]+", "-", display.lower()).strip("-")
                    enriched_powers.append(
                        {"class_name": power_class, "title": display, "slug": slug}
                    )
            card["referenced_powers"] = enriched_powers

        # Epoch unlock info
        if class_name in card_epoch_map:
            card["unlocked_by"] = card_epoch_map[class_name]

        # Flag deprecated/mock cards
        if card["character"] == "Deprecated":
            card["deprecated"] = True
        if card["character"] == "Mock":
            card["mock"] = True

        cards.append(card)

    # Write output
    output_path = os.path.join(output_dir, "cards.json")
    write_json(output_path, cards)

    # Stats
    print(f"Extracted {len(cards)} cards to {output_path}")
    by_char: dict[str, int] = {}
    for c in cards:
        ch = c.get("character", "Unknown")
        by_char[ch] = by_char.get(ch, 0) + 1
    for ch, count in sorted(by_char.items()):
        print(f"  {ch}: {count}")

    unmatched = [c for c in cards if c.get("_loc_missing")]
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} cards without localization match:")
        for c in unmatched[:10]:
            print(f"  {c['class_name']} (tried key: {c['loc_key']})")


if __name__ == "__main__":
    main()
