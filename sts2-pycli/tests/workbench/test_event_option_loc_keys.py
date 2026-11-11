"""Pin every event option's ``loc_key`` to a real row in the game's own loc.

The engine's ``EventOptionSpec.default_label`` / ``default_description`` are
developer shorthand written for run logs — "Gain a random potion." where the
game says "Procure 1 random Potion.". The workbench used to render exactly
those strings, so every event screen was quietly off-message (173 of 177 static
options disagreed with the game). Each option now carries a ``loc_key`` naming
its row in ``localization_eng/events.json``, and the display layer renders that
instead.

Two ways that can rot, both caught here:

1. A ``loc_key`` that names no row — a typo, or a build where the option was
   renamed. The display silently falls back to the shorthand, which is exactly
   the bug this replaced, so it must fail loud instead.
2. A template whose ``{VarName}`` placeholders are not all supplied. That
   renders the variable NAME ("Gain Gold Gold."), so the display layer falls
   back — correct, but it means the option is still off-message. This test
   reports which ones, pinned to a known list so the number can only go down.

Both an option's TITLE and its DESCRIPTION are templates, and five titles take
variables of their own ("Accept the {Relic}"), so both are gated.

The keys are read out of the Rust source rather than the extension because
they are static spec data with no runtime accessor, and the point is to gate
what is WRITTEN, including options no test run happens to reach.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from workbench import localization, render

_ROOT = Path(__file__).resolve().parents[3]
_EVENTS_RS = _ROOT / "sts2-rs" / "crates" / "sts2-content" / "src" / "events"
_LOC = Path(__file__).resolve().parents[2] / "localization_eng" / "events.json"

#: ``(loc_key, field)`` pairs whose localized template takes ``{VarName}``
#: placeholders the engine does not supply, so they still render the engine's
#: shorthand. **Empty** — every option now names what it actually does.
#:
#: It was not always. The last entries here all named a thing the RUN picks,
#: and were blocked on the same thing: the handler was a stub that never made
#: the pick, so naming it would have advertised an outcome the run did not
#: deliver. Adding to this list means shipping that lie; port the handler
#: instead.
_KNOWN_UNSUPPLIED_VARS: set[tuple[str, str]] = set()

_WIKI_DATA = _ROOT / "sts2-wiki" / "data"

#: Mirrors ``server._EVENT_FLAGS``.
_EVENT_FLAGS = {"ismultiplayer": False}

_FIELDS = ("title", "description")


def _covered(template: str, supplied: set[str]) -> bool:
    """True when every placeholder in ``template`` has a value at display time.

    Counts the spec's own vars plus the ``{SomethingDescription}`` splices the
    display layer resolves from the item the name points at (Tea Master).
    """
    # Same branch flags the display layer renders with — the DSL's boolean
    # form needs a flag, not a value (`{IsMultiplayer:Everyone loses|Lose}`).
    needed = render.required_stat_vars(template, extra_flags=_EVENT_FLAGS)
    have = {name.lower() for name in supplied}
    have |= set(
        localization.named_description_vars(
            needed, have, localization.load_owned_localization(_WIKI_DATA)
        )
    )
    return not (needed - have)


# ``cargo fmt`` splits a long entry across lines, so the name and the
# ``EventVarSpec`` can be on separate lines.
_VAR_NAME_RE = re.compile(r'\(\s*"(\w+)",\s*EventVarSpec::', re.S)


#: A `("Name", …)` tuple head — how a var name is written wherever it is built.
_TUPLE_HEAD_RE = re.compile(r'\(\s*"(\w+)",', re.S)


def _closure_var_names(closure: str, source: str) -> set[str]:
    """Var names a ``vars_fn`` supplies, following one level of indirection.

    Three shapes are in use and all three have to be read, or the gate reports
    a wired option as unsupplied and someone "fixes" it by re-pinning it:

    * inline — ``|s| vec![("Gold", …)]``
    * name passed to a shared builder —
      ``|s| relic_trader_vars(s, 0, "TopRelicOwned", "TopRelicNew")``
    * name owned by the builder — ``|s| ranwid_vars(s, true)``, where the
      literals live in ``fn ranwid_vars``.
    """
    names = set(_TUPLE_HEAD_RE.findall(closure)) | set(re.findall(r'"(\w+)"', closure))
    for helper in re.findall(r"\b(\w+)\s*\(\s*s\b", closure):
        body = re.search(rf"\nfn {helper}\((.*?)\n\}}", source, re.S)
        if body:
            names |= set(_TUPLE_HEAD_RE.findall(body.group(1)))
    return names


def _info_loc_keys(body: str, source: str) -> set[str]:
    """The loc key(s) one ``EventOptionInfo`` literal can render as.

    A hand-built option either names its key inline or indexes a per-page
    const array (``TABLET_OF_TRUTH_DECIPHER_KEYS[page]``) — the same key,
    re-pointed per page. Both forms resolve to the set of rows the option can
    ever render, which is what the var check has to cover.
    """
    field = re.search(r"loc_key:\s*(.*?),\n", body, re.S)
    if not field:
        return set()
    expr = field.group(1)
    inline = re.findall(r'"([^"]+)"', expr)
    if inline:
        return set(inline)
    name = re.match(r"\s*([A-Z][A-Z0-9_]*)", expr)
    if not name:
        return set()
    decl = re.search(rf"\nconst {name.group(1)}\s*:.*?=\s*\[(.*?)\];", source, re.S)
    return set(re.findall(r'"([^"]+)"', decl.group(1))) if decl else set()


def _wired() -> dict[str, set[str]]:
    """``loc_key -> names the spec supplies`` for every non-empty key."""
    out: dict[str, set[str]] = {}
    for path in sorted(_EVENTS_RS.glob("*.rs")):
        source = path.read_text(encoding="utf-8")
        for block in re.finditer(r"EventOptionSpec\s*\{(.*?)\n\s*\},", source, re.S):
            body = block.group(1)
            key = re.search(r'loc_key:\s*"([^"]*)"', body)
            if not key or not key.group(1):
                continue
            static = re.search(r"default_vars:\s*(&\[.*?\]),", body, re.S)
            names = set(_VAR_NAME_RE.findall(static.group(1) if static else ""))
            # `vars_fn: Some(|s| vec![("Gold", …)])` supplies the run-computed
            # half. Every string literal inside the closure is a var name —
            # either inline in the tuple, or passed to a shared builder
            # (`relic_trader_vars(s, 0, "TopRelicOwned", "TopRelicNew")`), which
            # is why this takes all literals rather than only tuple heads.
            for field in ("vars_fn", "title_vars_fn"):
                dynamic = re.search(rf"{field}:\s*Some\((.*?)\),\n", body, re.S)
                if dynamic:
                    names |= _closure_var_names(dynamic.group(1), source)
            out.setdefault(key.group(1), set()).update(names)
        # A multi-page presenter builds its option by hand, so its vars live
        # on an `EventOptionInfo` literal rather than an `EventOptionSpec`.
        # Read those too: without this the presenter's keys land in the
        # existence-only sweep below with an EMPTY var set, and an option that
        # DOES supply its placeholder at runtime is reported as unsupplied —
        # which reads as "port the handler", the one instruction that cannot
        # help when the handler is already ported (GH #487's Tablet of Truth
        # escalation pages, whose price is per-page and cannot be static).
        for block in re.finditer(r"EventOptionInfo\s*\{(.*?)\n\s*\};", source, re.S):
            body = block.group(1)
            names = set(_TUPLE_HEAD_RE.findall(body))
            for key in _info_loc_keys(body, source):
                out.setdefault(key, set()).update(names)
        # A presenter that re-points an option at a per-item key from a
        # `const fn loc_key(self)` match (Endless Conveyor's dish) and then
        # assigns that option's `vars` supplies those vars to every key the
        # match can return.
        for assign in re.finditer(
            r"(\w+)\.loc_key\s*=\s*\w+\.loc_key\(\);\s*\n\s*\1\.vars\s*=\s*vec!\[(.*?)\];",
            source,
            re.S,
        ):
            names = set(_TUPLE_HEAD_RE.findall(assign.group(2)))
            for table in re.finditer(r"fn loc_key\(self\)[^{]*\{(.*?)\n    \}", source, re.S):
                for key in re.findall(r'"([^"]+)"', table.group(1)):
                    out.setdefault(key, set()).update(names)
        # Dynamic presenters hard-code a loc_key on an EventOptionInfo /
        # helper; those are gated for existence only (their vars are runtime).
        for m in re.finditer(r'loc_key:\s*"([^"]+)"', source):
            out.setdefault(m.group(1), set())
        for m in re.finditer(r'"(\w+\.pages\.\w+\.options\.\w+)"', source):
            out.setdefault(m.group(1), set())
    return out


def _unsupplied() -> set[tuple[str, str]]:
    """``(loc_key, field)`` pairs whose template still has an unfilled var."""
    loc = json.loads(_LOC.read_text(encoding="utf-8"))
    out: set[tuple[str, str]] = set()
    for key, supplied in _wired().items():
        for field in _FIELDS:
            template = loc.get(f"{key}.{field}")
            if template and not _covered(template, supplied):
                out.add((key, field))
    return out


def test_every_wired_loc_key_names_a_real_row():
    """A key with no row silently degrades to the shorthand this replaced."""
    loc = json.loads(_LOC.read_text(encoding="utf-8"))
    missing = sorted(
        key for key in _wired() if f"{key}.title" not in loc and f"{key}.description" not in loc
    )
    assert missing == [], f"event option loc_key(s) with no row in events.json: {missing}"


def test_supplied_vars_render_the_game_wording_with_no_residue():
    """Where the spec supplies every placeholder, the option must render
    cleanly — no bare ``{Var}`` and no leftover variable NAME."""
    loc = json.loads(_LOC.read_text(encoding="utf-8"))
    broken = []
    for key, supplied in sorted(_wired().items()):
        for field in _FIELDS:
            template = loc.get(f"{key}.{field}")
            if not template or not _covered(template, supplied):
                continue  # not fully supplied — the display layer falls back
            out = render.resolve_card_text(
                template, {name.lower(): 1 for name in supplied}, extra_flags=_EVENT_FLAGS
            )
            if "{" in out or "}" in out:
                broken.append((key, field, out))
    assert broken == [], f"templates that render with residue: {broken}"


def test_unsupplied_var_list_only_shrinks():
    """Options still falling back to the engine's shorthand, pinned exactly.

    Each is blocked on its handler: the engine has to make the pick before the
    text can name it. Wiring one up means deleting its line here.
    """
    unsupplied = _unsupplied()
    assert unsupplied <= _KNOWN_UNSUPPLIED_VARS, (
        "new option(s) rendering the engine's shorthand instead of the game's "
        f"wording: {sorted(unsupplied - _KNOWN_UNSUPPLIED_VARS)}"
    )
    assert unsupplied >= _KNOWN_UNSUPPLIED_VARS, (
        "these options now supply all their vars and should leave "
        f"_KNOWN_UNSUPPLIED_VARS: {sorted(_KNOWN_UNSUPPLIED_VARS - unsupplied)}"
    )
