<#
.SYNOPSIS
    Bootstrap the spire-mentor build toolchain on Windows.

.DESCRIPTION
    Idempotent installer for the tools needed to compile this monorepo on
    native Windows. Safe to re-run: every step checks for the tool first and
    skips it if already present.

    Tiers (install what you need):
      * Rust engine (sts2-rs)     -> just, cargo-nextest   [always]
      * Python CLI  (sts2-pycli)  -> Python 3.13 via uv     [-SkipPython to omit]
      * C# runner + the mod       -> .NET 9 SDK             [-SkipDotnet to omit]
      * Web UIs                   -> Node.js LTS            [-SkipNode to omit]
      * Workbench with real art   -> ilspycmd 9.1, wasm-bindgen-cli (the version
         and sound                   sts2-rs/Cargo.lock pins), libwebp (cwebp),
                                     ffmpeg (the sound effects' Opus encode)
                                                            [-SkipWorkbench to omit]

    Prerequisites assumed already installed (the script verifies and warns):
      rustup/cargo, uv, git, and VS Build Tools 2022 (VC++ / linker).

.PARAMETER RustOnly
    Only install what the Rust engine needs (just + cargo-nextest +
    rust-analyzer). Implies -SkipPython -SkipDotnet -SkipNode -SkipWorkbench.

.PARAMETER SkipPython
    Skip installing the pinned Python 3.13 via uv.

.PARAMETER SkipDotnet
    Skip installing the .NET 9 SDK (needed only for the C# headless runner).

.PARAMETER SkipNode
    Skip installing Node.js LTS (needed only for the web frontends).

.PARAMETER SkipWorkbench
    Skip the tools `just import-game` and `just workbench-local` need on top
    of the tiers above (ilspycmd, wasm-bindgen-cli, libwebp, ffmpeg).

.EXAMPLE
    ./install-windows-deps.ps1              # full stack
    ./install-windows-deps.ps1 -RustOnly    # just the Rust engine deps
    ./install-windows-deps.ps1 -SkipNode    # everything except the web UIs
#>
[CmdletBinding()]
param(
    [switch]$RustOnly,
    [switch]$SkipPython,
    [switch]$SkipDotnet,
    [switch]$SkipNode,
    [switch]$SkipWorkbench
)

$ErrorActionPreference = 'Stop'

if ($RustOnly) { $SkipPython = $true; $SkipDotnet = $true; $SkipNode = $true; $SkipWorkbench = $true }

# --- pretty output helpers -------------------------------------------------
function Info($m) { Write-Host "  $m" -ForegroundColor Gray }
function Ok($m)   { Write-Host "OK   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "WARN $m" -ForegroundColor Yellow }
function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Have($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

# Pull in tools that winget/cargo dropped onto PATH during THIS session.
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = ($machine, $user, "$env:USERPROFILE\.cargo\bin", "$env:USERPROFILE\.dotnet\tools" -join ';')
}

$installed = @()
$skipped   = @()
$failed    = @()

function Install-Winget($id, $cmd, $label) {
    if (Have $cmd) { Ok "$label already installed ($cmd)"; $script:skipped += $label; return }
    Info "Installing $label via winget ($id) ..."
    winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    Update-SessionPath
    if (Have $cmd) { Ok "$label installed"; $script:installed += $label }
    else { Warn "$label installed but '$cmd' not on PATH yet - reopen your shell"; $script:installed += $label }
}

# ---------------------------------------------------------------------------
Step "Checking prerequisites"

if (-not (Have winget)) {
    Warn "winget not found. Install 'App Installer' from the Microsoft Store, then re-run."
    throw "winget is required."
}
Ok "winget $(winget --version)"

foreach ($p in @(
    @{ cmd = 'cargo';   name = 'Rust (rustup/cargo)'; hint = 'https://rustup.rs' },
    @{ cmd = 'uv';      name = 'uv';                  hint = 'winget install astral-sh.uv' },
    @{ cmd = 'git';     name = 'git';                 hint = 'winget install Git.Git' }
)) {
    if (Have $p.cmd) { Ok "$($p.name) present" }
    else { Warn "$($p.name) MISSING - install it first ($($p.hint))" }
}

# VC++ build tools / linker (Rust needs link.exe on the MSVC target).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasVc = $false
if (Test-Path $vswhere) {
    $vc = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property displayName 2>$null
    if ($vc) { $hasVc = $true; Ok "MSVC build tools present ($vc)" }
}
if (-not $hasVc) {
    Warn "VS Build Tools with 'Desktop development with C++' NOT detected."
    Warn "  Rust linking will fail. Install with:"
    Warn "  winget install --id Microsoft.VisualStudio.2022.BuildTools --override `"--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`""
}

# ---------------------------------------------------------------------------
Step "Rust engine deps (sts2-rs)"

Install-Winget 'Casey.Just' 'just' 'just (command runner)'

# cargo-nextest: no reliable winget package; build/fetch via cargo.
if (Have 'cargo-nextest') {
    Ok "cargo-nextest already installed"; $skipped += 'cargo-nextest'
} elseif (Have 'cargo') {
    Info "Installing cargo-nextest via cargo (compiles; a few minutes) ..."
    try {
        cargo install cargo-nextest --locked
        Update-SessionPath
        if (Have 'cargo-nextest') { Ok "cargo-nextest installed"; $installed += 'cargo-nextest' }
        else { Warn "cargo-nextest build finished but not on PATH"; $failed += 'cargo-nextest' }
    } catch { Warn "cargo-nextest install failed: $_"; $failed += 'cargo-nextest' }
} else {
    Warn "cargo missing - cannot install cargo-nextest"; $failed += 'cargo-nextest'
}

# rust-analyzer component (optional IDE support; cheap, always attempt).
if (Have 'rustup') {
    Info "Ensuring rust-analyzer rustup component ..."
    try { rustup component add rust-analyzer 2>$null; Ok "rust-analyzer component present" }
    catch { Warn "could not add rust-analyzer component (non-fatal)" }
}

# ---------------------------------------------------------------------------
if (-not $SkipPython) {
    Step "Python CLI deps (sts2-pycli)"
    if (Have 'uv') {
        Info "Installing pinned Python 3.13 via uv ..."
        try { uv python install 3.13; Ok "Python 3.13 available to uv"; $installed += 'Python 3.13 (uv)' }
        catch { Warn "uv python install failed: $_"; $failed += 'Python 3.13' }
        Info "Note: run 'uv sync' inside sts2-pycli/ to build the venv + sts2_rs wheel."
    } else {
        Warn "uv missing - skipping Python. Install uv first (winget install astral-sh.uv)."
        $failed += 'Python 3.13'
    }
} else { $skipped += 'Python 3.13' }

# ---------------------------------------------------------------------------
if (-not $SkipDotnet) {
    Step "C# headless runner deps (sts2-pycli/src)"
    # Need the SDK, not just a runtime. `dotnet --list-sdks` is empty with runtimes only.
    $hasSdk = $false
    if (Have 'dotnet') {
        try { if ((dotnet --list-sdks 2>$null | Where-Object { $_ -match '^9\.' })) { $hasSdk = $true } } catch {}
    }
    if ($hasSdk) { Ok ".NET 9 SDK already installed"; $skipped += '.NET 9 SDK' }
    else { Install-Winget 'Microsoft.DotNet.SDK.9' 'dotnet' '.NET 9 SDK' }
    Info "The C# runner also needs the game DLLs: 'just import-game' stages them from your install."
} else { $skipped += '.NET 9 SDK' }

# ---------------------------------------------------------------------------
if (-not $SkipNode) {
    Step "Web UI deps"
    Install-Winget 'OpenJS.NodeJS.LTS' 'node' 'Node.js LTS'
} else { $skipped += 'Node.js LTS' }

# ---------------------------------------------------------------------------
if (-not $SkipWorkbench) {
    Step "Workbench with real art and sound (just import-game, just workbench-local)"

    # ILSpy's CLI decompiles sts2.dll. 10.x+ needs a newer runtime than the
    # .NET 9 SDK; 9.1 is the last release that installs on .NET 9.
    if (Have 'ilspycmd') {
        Ok "ilspycmd already installed"; $skipped += 'ilspycmd'
    } elseif (Have 'dotnet') {
        Info "Installing ilspycmd 9.1 (dotnet global tool) ..."
        dotnet tool install -g ilspycmd --version 9.1.0.7988
        Update-SessionPath
        if (Have 'ilspycmd') { Ok "ilspycmd installed"; $installed += 'ilspycmd' }
        else { Warn "ilspycmd install failed"; $failed += 'ilspycmd' }
    } else {
        Warn "dotnet missing - cannot install ilspycmd"; $failed += 'ilspycmd'
    }

    # wasm-bindgen-cli must match the wasm-bindgen crate in sts2-rs/Cargo.lock.
    $lock = Join-Path $PSScriptRoot 'sts2-rs\Cargo.lock'
    $wbVer = $null
    if (Test-Path $lock) {
        $m = [regex]::Match((Get-Content $lock -Raw), 'name = "wasm-bindgen"\r?\nversion = "([^"]+)"')
        if ($m.Success) { $wbVer = $m.Groups[1].Value }
    }
    $haveWb = $false
    if (Have 'wasm-bindgen') { $haveWb = ((wasm-bindgen --version) -match [regex]::Escape("$wbVer")) }
    if ($haveWb) {
        Ok "wasm-bindgen $wbVer already installed"; $skipped += 'wasm-bindgen-cli'
    } elseif ($wbVer -and (Have 'cargo')) {
        Info "Installing wasm-bindgen-cli $wbVer via cargo (compiles; a few minutes) ..."
        cargo install wasm-bindgen-cli --version $wbVer --locked
        Update-SessionPath
        if (Have 'wasm-bindgen') { Ok "wasm-bindgen-cli $wbVer installed"; $installed += 'wasm-bindgen-cli' }
        else { Warn "wasm-bindgen-cli install failed"; $failed += 'wasm-bindgen-cli' }
    } else {
        Warn "cannot install wasm-bindgen-cli (cargo or sts2-rs/Cargo.lock missing)"; $failed += 'wasm-bindgen-cli'
    }

    # cwebp: the static build ships the extracted art as WebP.
    Install-Winget 'Google.Libwebp' 'cwebp' 'libwebp (cwebp)'

    # ffmpeg: import-game's sound step encodes the rendered effects to Opus.
    # (FMOD itself needs no install: the game ships fmod.dll, and import-game
    # renders the sound banks with it.)
    Install-Winget 'Gyan.FFmpeg' 'ffmpeg' 'ffmpeg'
} else { $skipped += 'workbench tools' }

# ---------------------------------------------------------------------------
Step "Summary"
if ($installed) { Ok  ("Installed: " + ($installed -join ', ')) }
if ($skipped)   { Info ("Already present / skipped: " + ($skipped -join ', ')) }
if ($failed)    { Warn ("Needs attention: " + ($failed -join ', ')) }

Write-Host @"

Next steps (from the repo root, in a NEW terminal so the tools are on PATH):
  * Workbench:     just import-game        # finds the game through Steam; art + sound
                   just workbench-local    # http://127.0.0.1:4173
  * The mod:       cd spire-mentor-mod; dotnet build   (needs BaseLib - spire-mentor-mod\README.md)
  * Rust engine:   just rs-test
  The root justfile runs under PowerShell. The subproject justfiles are bash:
  run those from Git Bash.
"@ -ForegroundColor Cyan
