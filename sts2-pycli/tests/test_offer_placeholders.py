"""Regression guard: offered text must never leak template placeholders.

The bug this pins: the Signet Ring relic offer rendered

    "Upon pickup, gain Gold Gold."

instead of "Upon pickup, gain 999 Gold." — the C# offer serializer handed the
renderer the ``{Gold}`` template but no value for ``Gold``, so ``resolve()`` fell
back to the bare variable name. The same class of bug (a numeric DynamicVar that
the offer path forgets to send) had surfaced before, so this test sweeps EVERY
relic and potion the game can offer and asserts none of them resolve to a bare,
unresolved numeric placeholder.

It drives the real headless via the ``dump_content`` command, which serializes
every relic/potion through the exact same name+description+vars helpers the live
offer surfaces (shop, event option, …) use. A future regression that drops the
vars from any of those payloads makes this test fail.

Scope: relics and potions, whose DynamicVars are fully defined at rest (so the
dump faithfully mirrors what an offer would carry). Cards are validated by the
in-combat ``format_card`` unit tests in ``test_sts2_format`` instead — many card
vars are only materialized once the card has a live owner, so an at-rest dump
under-represents them and can't be asserted strictly here.
"""

from __future__ import annotations

import io
import re
import sys

import pytest
import sts2_client
from sts2_format import resolve

# Tokens the game injects at render time that are NOT numeric DynamicVars and so
# are legitimately absent from a relic/potion's vars dict (the value is an icon
# image / energy prefix, not a number). resolve() warns about them but they are
# not the missing-number bug this test guards against.
_ALLOWED_MISSING = {"singlestaricon", "energyprefix"}

# A capitalised "[VarName]" left in the rendered output means a template token
# survived strip-but-not-substitute — i.e. the payload sent no vars dict at all
# (the original shop-relic bug shape). Lowercase bracket tokens are synthetic
# icons and are checked via the stderr warning path instead.
_BRACKET_RESIDUE = re.compile(r"\[[A-Z][A-Za-z0-9]*\]")
_MISSING_WARNING = re.compile(r"missing template var:\s*(\S+)")


def _resolve_capturing(description: str, vars_dict: dict | None) -> tuple[str, set[str]]:
    """resolve() the text, returning the output and any warned-missing var names."""
    err = io.StringIO()
    old = sys.stderr
    sys.stderr = err
    try:
        out = resolve(description, vars_dict)
    finally:
        sys.stderr = old
    missing = {m.group(1).lower() for m in _MISSING_WARNING.finditer(err.getvalue())}
    return out, missing


def _offered_items(game) -> dict[str, list[dict]]:
    game.start("Ironclad", seed="test")
    content = game.dump_content()
    assert content.get("type") == "content"
    return content


def _check(items: list[dict], var_key: str) -> list[str]:
    """Return a list of human-readable failures for the given content list."""
    failures: list[str] = []
    for item in items or []:
        description = item.get("description")
        if not description:
            continue
        ident = item.get("id", "?")
        out, missing = _resolve_capturing(description, item.get(var_key))

        unexpected_missing = missing - _ALLOWED_MISSING
        if unexpected_missing:
            failures.append(
                f"{ident}: unresolved numeric var(s) {sorted(unexpected_missing)} "
                f"in {description!r} -> {out!r}"
            )

        residue = _BRACKET_RESIDUE.findall(out)
        if residue:
            failures.append(
                f"{ident}: leftover placeholder(s) {residue} in {description!r} -> {out!r}"
            )
    return failures


@pytest.fixture(scope="module")
def content():
    from sts2_client import Game

    g = Game()
    try:
        yield _offered_items(g)
    finally:
        g.close()


def test_no_relic_offer_shows_placeholder(content):
    failures = _check(content.get("relics") or [], "vars")
    assert content.get("relics"), "expected the dump to enumerate relics"
    assert not failures, "Relics with unresolved placeholders:\n" + "\n".join(failures)


def test_no_potion_offer_shows_placeholder(content):
    failures = _check(content.get("potions") or [], "vars")
    assert content.get("potions"), "expected the dump to enumerate potions"
    assert not failures, "Potions with unresolved placeholders:\n" + "\n".join(failures)


#: Signet Ring's ``GoldVar`` per game build (``SignetRing.cs``: 999 at v0.107.1,
#: 888 at v0.111.0 and v0.113.0).
_SIGNET_RING_GOLD = {"v0.107.1": "999", "v0.111.0": "888", "v0.113.0": "888"}


def test_signet_ring_offer_shows_gold_number(content):
    """The exact reported case: Signet Ring must render its Gold value."""
    signet = next((r for r in content.get("relics") or [] if r.get("id") == "SIGNET_RING"), None)
    assert signet is not None, "SIGNET_RING not found in relic dump"
    out, missing = _resolve_capturing(signet.get("description"), signet.get("vars"))
    gold = _SIGNET_RING_GOLD[sts2_client.GAME_VERSION]
    assert gold in out, f"expected the Gold amount {gold} in {out!r}"
    assert "Gold Gold" not in out, f"placeholder leaked: {out!r}"
    assert not missing, f"unexpected missing vars: {missing}"


def test_live_shop_offers_show_no_placeholder():
    """Guard the real shop serialization wiring, not just the shared helper.

    ``dump_content`` proves the relic/potion *data* carries resolvable vars, but a
    decision surface can still forget to attach them (the shop-potion path did
    exactly that). Drive an actual MerchantRoom and assert the relics and potions
    it offers resolve cleanly.
    """
    from sts2_client import Game

    g = Game()
    try:
        g.start("Ironclad", seed="test")
        shop = g.enter_room("shop")
    finally:
        g.close()

    assert shop.get("decision") == "shop", f"expected a shop decision, got {shop!r}"
    failures = _check(shop.get("relics") or [], "vars") + _check(shop.get("potions") or [], "vars")
    assert not failures, "Shop items with unresolved placeholders:\n" + "\n".join(failures)
