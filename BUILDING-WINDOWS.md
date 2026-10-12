# Building on Windows

How to compile and test **spire-mentor** on native Windows (no WSL): the Rust
engine, `just import-game`, the browser workbench and the Spire Mentor mod all
build and run from PowerShell.

## 1. Install the toolchain

Run the idempotent bootstrap script from the repo root (safe to re-run — it
skips anything already present):

```powershell
.\install-windows-deps.ps1            # full stack
.\install-windows-deps.ps1 -RustOnly  # just the Rust engine deps
```

If PowerShell blocks it:
`powershell -ExecutionPolicy Bypass -File .\install-windows-deps.ps1`.

What each tier needs:

| Tier | Tools | Installed via |
|------|-------|---------------|
| **Rust engine** (`sts2-rs`) | `just`, `cargo-nextest`, rust-analyzer | winget / cargo / rustup |
| **Python CLI** (`sts2-pycli`) | Python 3.13 | `uv python install` |
| **C# runner + the mod** | .NET 9 SDK | winget |
| **Web UIs** | Node.js LTS | winget |
| **Workbench with real art** (`-SkipWorkbench` to omit) | `ilspycmd` 9.1, `wasm-bindgen-cli` (the `sts2-rs/Cargo.lock` version), libwebp (`cwebp`) | dotnet tool / cargo / winget |

Prerequisites the script verifies but does **not** install: `rustup`/`cargo`,
`uv`, `git`, and **VS Build Tools 2022 with the "Desktop development with
C++" workload** (provides `link.exe` — Rust cannot link without it).

The repo runs Python through `uv run` everywhere, so the Microsoft Store
`python.exe` alias on PATH does no harm.

## 2. Build & test the Rust engine

```powershell
cd sts2-rs
cargo build --workspace
cargo nextest run --workspace   # canonical test runner — NOT `cargo test`
```

Expected: full workspace builds, **6092 tests pass**.

> Use `cargo nextest`, never `cargo test` — nextest runs one process per test,
> isolating the process-global effect/relic registry. Plain `cargo test` shares
> one process and produces order-dependent false failures.

## 3. The workbench and the mod

From the repo root, in PowerShell:

```powershell
just import-game        # finds the game through Steam (any library); or $env:STS2_GAME_DIR
just workbench-local    # builds the wasm + SPA + art, serves http://127.0.0.1:4173
cd spire-mentor-mod
dotnet build            # builds the mod and installs it into <game>\mods\SpireMentor\
```

`import-game` needs no Developer Mode: it stages the game with a hard link and
a directory junction where it cannot symlink. The mod also needs BaseLib and,
for the in-game advisor, `cd sts2-pycli; uv sync` once (which builds the
`sts2-py` crate below) — see
[`spire-mentor-mod/README.md`](spire-mentor-mod/README.md).

## 4. The `sts2-py` PyO3 crate needs a real Python

The `sts2-py` crate embeds a Python interpreter and needs a **real** one at
**both build and run time**. Two traps on Windows:

1. The Microsoft Store `python.exe` alias in `%LOCALAPPDATA%\Microsoft\WindowsApps`
   is a dead stub that hijacks auto-detection — PyO3 finds it and fails with
   *"no Python 3.x interpreter found"*.
2. The compiled test binary must find `pythonNNN.dll` at load time, or it aborts
   with `0xc0000135` (STATUS_DLL_NOT_FOUND).

**Fix once (machine-local, per-user — never commit this):** point PyO3 straight
at a real interpreter and put its folder on PATH. Using a uv-managed Python:

```powershell
uv python install 3.13
$py = "$env:APPDATA\uv\python\cpython-3.13-windows-x86_64-none"

# Persist for your user account (survives new terminals):
[Environment]::SetEnvironmentVariable('PYO3_PYTHON', "$py\python.exe", 'User')
$userPath = [Environment]::GetEnvironmentVariable('Path','User')
if ($userPath -notlike "*$py*") {
    [Environment]::SetEnvironmentVariable('Path', "$py;$userPath", 'User')
}
```

Open a **new terminal** afterward. Then `cargo nextest run --workspace` works
with no per-session setup.

- `PYO3_PYTHON` (not PATH order) is what defeats the Store stub — it's
  unambiguous.
- These are user-env settings, **not** `sts2-rs/.cargo/config.toml`: that file
  is committed and shared, and Cargo config cannot scope an env var by OS, so a
  Windows path there would break every Linux/macOS dev.
- If uv later drops 3.13 for a newer minor, the folder name changes
  (`cpython-3.14-…`) and you repoint `PYO3_PYTHON` once. Within 3.13.x it's
  stable.
- `cargo build --workspace --exclude sts2-py` needs none of this.

## Line endings

The census/parity golden files under `sts2-rs/tests/parity/` are compared
byte-for-byte against `\n`-built strings. `sts2-rs/.gitattributes` pins them
(and `*.rs`) to `eol=lf` so `core.autocrlf=true` doesn't check them out as CRLF
and fail every golden comparison. Leave that rule in place; if goldens still
show CRLF locally, re-checkout them or strip `\r`.

## Caveats

- The root `justfile` runs under PowerShell on Windows (`rs-check`,
  `rs-test`, `rs-doc-test`, `py-test`, `web-test`, `import-game`,
  `workbench-local`, `workbench-server`). The subproject justfiles are
  written for bash: run them from Git Bash. Most of their recipes work there;
  the worktree tooling (`scripts/wheel-venv.sh`, `fuser`-based recipes such
  as `sim-ui`) is Unix-only.
- `sccache` is not required; set `$env:RUSTC_WRAPPER="sccache"` to use it.
- The C# headless runner also needs the game DLLs — `just import-game`
  (docs/bring-your-own-game.md) stages them from your install.
