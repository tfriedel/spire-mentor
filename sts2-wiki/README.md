# sts2-wiki

A wiki *generator* for Slay the Spire 2: it decompiles a copy of the game
**you own**, extracts game data and assets from it, and builds a static
reference site (cards, monsters, encounters, relics, potions, powers, events,
epochs, characters, ascensions) with Astro.

## Never publish the game-derived content

> **WARNING.** Everything this pipeline *produces from the game* is derived
> from MegaCrit's commercial game and is **not redistributable**. That
> includes, concretely:
>
> - `decompiled/` — ILSpy output of `sts2.dll`
> - `data/` — extracted per-version game data (JSON)
> - `extracted/` and `site/public/images/` — localization text and images
>   pulled from the game's `.pck`
> - `overrides/` — hand-written corrections *to* game content
> - `docs/version-diffs/` — diffs of game data between game versions
> - the built site (`site/dist*`)
>
> None of these may ever be published, deployed publicly, or pushed to a
> public remote. Only the generator code itself (scripts, tools, site
> templates) is original work covered by [`LICENSE`](LICENSE) (MIT). See
> [`../PROVENANCE.md`](../PROVENANCE.md) for the full inventory. For the same
> reason the GitHub Pages deploy workflow does **not** live in an active
> `.github/workflows/` — it is parked at
> [`docs/ci-templates/deploy.yml`](docs/ci-templates/deploy.yml), which would
> auto-publish game content if ever restored in a public repository (see the
> README next to it).

## Prerequisites

- Your own installed copy of Slay the Spire 2 (Steam). The justfile defaults
  to the macOS install path; override with `STS2_APP` (path to the app) and
  `STS2_VERSION` (defaults to `v0.107.1`).
- [`just`](https://github.com/casey/just), [`uv`](https://docs.astral.sh/uv/)
  (Python 3.12+), Node.js/npm (for the Astro site), and the .NET tool
  [`ilspycmd`](https://github.com/icsharpcode/ILSpy) for decompilation
  (`dotnet tool install -g ilspycmd --version 9.1.0.7988` — 10+ does not
  install on the .NET 9 SDK; see `CLAUDE.md` for the `DOTNET_ROOT` gotcha on
  macOS).
- The LLM-based extractors (`just llm-extract-*`) additionally use the Claude
  Agent SDK, i.e. a working Claude Code setup.

## Pipeline (bring your own game copy)

```bash
just detect-version   # print the installed game's version
just decompile        # ilspycmd sts2.dll -> decompiled/{version}/
just extract-pck      # localization + image atlases from the .pck -> extracted/{version}/
just extract          # regex extractors: decompiled + localization -> data/{version}/*.json
just llm-extract      # LLM extractors: refresh per-entity JSON in data/{version}/<type>/
just extract-images   # card/monster/relic art -> site/public/images/
just extract-rigs     # Spine creature/room rigs -> site/public/images/*_rigs/
just render-portraits # still creature portraits rendered from the rigs
just generate         # data/{version}/ -> site/src/content/ (markdown)
just build-site       # Astro build -> site/dist/
```

`just update` chains decompile → extract-pck → extract → extract-images →
generate → build-site. `just generate-from-data` regenerates pages from the
committed JSON alone (no `decompiled/`/`extracted/` needed);
`just build-all-versions` builds every archived version into a merged
`site/dist-final/`. `just preview` serves the site locally.

Checks: `just check` (ruff, mypy, content-schema sync, link and image checks,
lookup-tool unit tests) must pass before committing. Scripts run as modules:
`uv run python -m scripts.extract_cards`, not `python scripts/extract_cards.py`.

## Data layout

`data/{version}/` holds one bulk JSON per entity type (`cards.json`,
`monsters.json`, `relics.json`, …) written by the regex extractors, plus
per-entity directories (`cards/`, `monsters/`, `events/`, …) of
`<ClassName>.json` files written by the LLM extractors. The generate scripts
load the bulk file first and then apply the per-entity JSON as overrides, so
an LLM-reviewed entity always wins over the regex extraction.

## Overrides

`overrides/<type>/<slug>.md` is the hand-authored correction layer for
*rendering* problems that aren't data problems: a file whose YAML frontmatter
fields replace the matching fields of the generated page (and whose body, if
present, replaces the page body). Unlike `data/{version}/`, overrides are
version-independent. The merge is applied by the generate step (see
`scripts/generate_events.py` for the mechanism); `scripts/event_enrichments.py`
carries additional hand-written event enrichments.

## Tools

- `tools/lookup/` — `sts2-lookup`, a CLI for quick enemy/card/encounter
  reference straight from `data/` (`uv run sts2-lookup enemy Nibbit`). Honors
  `STS2_WIKI_DATA` when the data tree lives elsewhere.
- `tools/spine-renderer/` — headless Node renderer that turns the extracted
  Spine rigs (`site/public/images/monster_rigs/<slug>/`, written by
  `just extract-rigs`) into idle-pose PNGs: `just render-portraits` renders
  `monsters/*.png` (512px) and `character_sprites/*.png` (640px). Direct use:
  `node render.mjs [--rigs DIR] [--out DIR] [--size N] [--force] [rig ...]`.

## Role in the monorepo

`decompiled/v0.107.1/` is the monorepo's authoritative source of truth for
game mechanics — the Rust engine (`../sts2-rs`) and parity tooling grep it
first for any "is this behavior correct?" question. `decompiled/v0.103.2/` is
retained for content removed after 0.103.2 that the engine still implements.
