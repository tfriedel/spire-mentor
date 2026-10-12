# spire-mentor umbrella tasks. Most work runs inside the subprojects
# (`just --list` in sts2-rs/, sts2-pycli/, sts2-wiki/).
#
# Windows: the recipes in THIS file run under PowerShell (each body is one
# plain command, no `&&`), so everything here — the engine tier, `import-game`
# and `workbench-local` — works from a PowerShell prompt. The subproject
# justfiles are bash: run those from Git Bash. Needs just >= 1.38.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Rust: fast typecheck of the whole workspace
[working-directory: 'sts2-rs']
rs-check:
    cargo check --workspace --all-targets

# Rust: canonical test runner (nextest — never bare `cargo test`)
[working-directory: 'sts2-rs']
rs-test:
    cargo nextest run --workspace

# Rust: doctests — the one `cargo test` invocation (nextest cannot run them)
[working-directory: 'sts2-rs']
rs-doc-test:
    cargo test --workspace --doc

# Python: the test suite (tests that need the C# oracle skip without the game)
[working-directory: 'sts2-pycli']
py-test:
    uv run pytest tests

# Web: vitest (suites that need the extracted art/audio skip without them)
[working-directory: 'sts2-pycli/web']
web-test:
    npm test

# Everything CI runs, in one go
check: rs-check rs-test rs-doc-test py-test web-test

# docs/bring-your-own-game.md; `just import-game --status` shows what exists.
# The game is found through Steam (every library); elsewhere, set STS2_GAME_DIR.
# `uv run` supplies the Python: a bare `python` is the Microsoft Store stub on
# many Windows machines.
# Regenerate every game-derived input from YOUR game install
import-game *args:
    uv run --no-project --python 3.13 scripts/import_game.py {{args}}

# Run `just import-game` first for real art and sound. Please don't host the
# build publicly: the art and audio are the game's, extracted from your copy
# for your own use.
# Build the browser workbench and serve it on http://127.0.0.1:4173 (this machine only)
workbench-local:
    uv run --no-project --python 3.13 sts2-pycli/scripts/workbench_static.py build
    uv run --no-project --python 3.13 sts2-pycli/scripts/workbench_static.py preview

# Serves the SPA `workbench-local` built. The Spire Mentor mod's advisor
# starts this itself when its "Open Workbench" button needs it.
# The Python-backed workbench on http://127.0.0.1:8077
[working-directory: 'sts2-pycli']
workbench-server $PYTHONPATH="python" $STS2_SERVE_SPA="1":
    uv run python -m workbench

# Python envs for the two Python subprojects
setup:
    uv --directory sts2-pycli sync --dev
    uv --directory sts2-wiki sync
    -rustup component add rust-analyzer clippy rustfmt
