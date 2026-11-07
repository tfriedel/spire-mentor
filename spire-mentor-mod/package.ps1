# Assemble the distributable SpireMentor mod zip (GH #380).
#
# Output: SpireMentor-v<version>-win64.zip whose root contains mods/, so the
# install instruction is exactly "extract into the Slay the Spire 2 folder".
#
#   mods/SpireMentor/   SpireMentor.dll, SpireMentor.json, SpireMentor.conf.template,
#                     SimAdvisor/            (the packaged torch-free advisor)
#                     SpireMentorFrame/      (the panels' own window, .NET
#                                             Framework 4.8; SimFrameLauncher)
#                     workbench/             (the browser-only workbench, which
#                                             the advisor hosts on :8079)
#   mods/BaseLib/     BaseLib.dll, BaseLib.pck, BaseLib.json,
#                     LICENSE-BaseLib.txt    (MIT notice; only staged when the
#                                             upstream license text is present
#                                             locally -- see below)
#   README.txt
#
# BaseLib is staged from the LOCAL GAME INSTALL (<game>/mods/BaseLib, else the
# Workshop item's folder -- the csproj's BaseLibDir): that is the exact v3.4.1
# binary this SpireMentor.dll was compiled against. BaseLib is MIT-licensed, so redistribution needs its
# license text in the zip. That text is NOT vendored in this repo (the
# authoritative copyright line lives upstream); it is picked up from an
# optional local clone of github.com/Alchyr/BaseLib-StS2 at refs/BaseLib-StS2
# (gitignored). Without the clone the zip builds with a WARNING and no
# LICENSE-BaseLib.txt -- fine for local testing, but source the license text
# from the BaseLib-StS2 upstream before distributing the zip.
#
# The SimAdvisor exe is taken prebuilt from sts2-pycli/dist/SimAdvisor by
# default; pass -BuildAdvisor to rebuild it first (slow-ish).
#
# The workbench is built here (`just workbench-static` with
# VITE_BRIDGE_SAME_ORIGIN=1, so the page talks to the control plane that
# serves it). It carries the game's art and sound when the local tree has
# them (`just import-game` at the workbench / sound tier), placeholders
# otherwise; -NoArt forces placeholders. The PPO nets are left out of it: the
# control plane serves the copies the advisor bundles.
#
# NOTE (licensing): SpireMentor is MIT (spire-mentor-mod/LICENSE); the zip also
# bundles BaseLib (MIT, its notice below), the advisor exe and, unless -NoArt,
# art and sound extracted from the game (© Mega Crit; shipped under Mega Crit's
# content policy for free, unaffiliated mods -- https://www.megacrit.com/content-policy/).
param(
    # Rebuild the advisor exe before packaging instead of reusing the
    # prebuilt sts2-pycli/dist/SimAdvisor.
    [switch]$BuildAdvisor,
    # Stop after staging (staging/mods/...): workshop.ps1 builds its Workshop
    # content from that and needs neither the zip nor BaseLib's license text.
    [switch]$NoZip,
    # Ship the workbench with placeholder art and no sound.
    [switch]$NoArt,
    # Where in-game "Report a bug" sends: the hosted workbench's origin
    # (https://..., its /api/bugreport stores reports). Default:
    # $env:STS2_WORKBENCH_SITE, else the STS2_WORKBENCH_SITE export in
    # sts2-pycli/deploy/justfile (private; absent from the public tree).
    # Empty = the packaged mod saves reports to a file only.
    [string]$BugReportSite = $env:STS2_WORKBENCH_SITE
)
$ErrorActionPreference = "Stop"
$bridge = $PSScriptRoot
$repo = Split-Path -Parent $bridge
$pycli = Join-Path $repo "sts2-pycli"

# --- locate the game (asks SpireMentor.csproj: STS2_GAME_DIR, else every Steam library)
$gameDir = (dotnet msbuild (Join-Path $bridge "SpireMentor.csproj") -getProperty:STS2_GAME_DIR).Trim()
# <game>/mods/BaseLib, else the Workshop item (the csproj's BaseLibDir).
$baseLibDir = (dotnet msbuild (Join-Path $bridge "SpireMentor.csproj") -getProperty:BaseLibDir).Trim()
if (-not (Test-Path (Join-Path $baseLibDir "BaseLib.dll"))) {
    throw "BaseLib not found at $baseLibDir -- subscribe to it on the Workshop or install it into the game " +
          "(https://github.com/Alchyr/BaseLib-StS2/releases, v3.4.1 for game v0.107.1), " +
          "or set STS2_GAME_DIR."
}
# The MIT license text is not vendored here -- it must come from the upstream
# BaseLib-StS2 repo (optional gitignored clone under refs/). Missing text is a
# warning, not an error: the zip still builds, but must not be distributed
# until LICENSE-BaseLib.txt is sourced from upstream.
$baseLibLicense = Join-Path $repo "refs/BaseLib-StS2/LICENSE.txt"
$haveBaseLibLicense = Test-Path $baseLibLicense
if (-not $haveBaseLibLicense -and -not $NoZip) {
    Write-Warning ("BaseLib license text not found at $baseLibLicense -- the zip will NOT " +
        "contain LICENSE-BaseLib.txt. Before distributing, clone " +
        "https://github.com/Alchyr/BaseLib-StS2 into refs/ and re-run (the MIT notice must ship).")
}

# --- the bug-report site (baked into the DLL, see SpireMentor.csproj) ---------
if (-not $BugReportSite) {
    $deployJustfile = Join-Path $pycli "deploy/justfile"
    if (Test-Path $deployJustfile) {
        $m = Select-String -Path $deployJustfile -Pattern '^export STS2_WORKBENCH_SITE\s*:=\s*"([^"]+)"' |
            Select-Object -First 1
        if ($m) { $BugReportSite = $m.Matches[0].Groups[1].Value }
    }
}
if ($BugReportSite) {
    Write-Host "Bug reports from this build go to $BugReportSite/api/bugreport"
} else {
    Write-Warning ("No bug-report site (-BugReportSite / STS2_WORKBENCH_SITE): players of this build " +
        "can only save reports to a file.")
}

# --- build both halves -------------------------------------------------------
dotnet build $bridge -c Release -p:SpireMentorSkipDeploy=true "-p:SpireMentorBugReportSite=$BugReportSite"
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed" }
$dll = Join-Path $bridge "bin/Release/net9.0/SpireMentor.dll"
if (-not (Test-Path $dll)) { throw "missing $dll" }

# The panels' own window (spire-mentor-frame, .NET Framework 4.8: no runtime
# for players to install). The mod starts it next to the game (SimFrameLauncher).
$frameProj = Join-Path $repo "spire-mentor-frame"
dotnet build $frameProj -c Release
if ($LASTEXITCODE -ne 0) { throw "frame build failed" }
$frameOut = Join-Path $frameProj "bin/Release/net48"
if (-not (Test-Path (Join-Path $frameOut "SpireMentorFrame.exe"))) { throw "missing $frameOut\SpireMentorFrame.exe" }

$advisorDist = Join-Path $pycli "dist/SimAdvisor"
if ($BuildAdvisor) {
    pwsh -NoProfile -File (Join-Path $pycli "scripts/build_advisor_exe.ps1")
    if ($LASTEXITCODE -ne 0) { throw "advisor exe build failed" }
}
if (-not (Test-Path (Join-Path $advisorDist "SimAdvisor.exe"))) {
    throw "missing $advisorDist\SimAdvisor.exe -- no prebuilt advisor found; build one with " +
          "-BuildAdvisor (runs sts2-pycli/scripts/build_advisor_exe.ps1)"
}

# The workbench: always rebuilt (a web/dist left by `npm run build` is the
# server build, and a deploy build lacks VITE_BRIDGE_SAME_ORIGIN).
$env:VITE_BRIDGE_SAME_ORIGIN = "1"
$env:STS2_WORKBENCH_NO_ART = $(if ($NoArt) { "1" } else { "" })
try {
    Push-Location $pycli
    uv run --no-project python scripts/workbench_static.py build
    if ($LASTEXITCODE -ne 0) { throw "workbench build failed" }
} finally {
    Pop-Location
    Remove-Item Env:VITE_BRIDGE_SAME_ORIGIN, Env:STS2_WORKBENCH_NO_ART -ErrorAction SilentlyContinue
}
$workbenchDist = Join-Path $pycli "web/dist"
if (-not (Test-Path (Join-Path $workbenchDist "wb-precache.json"))) {
    throw "missing $workbenchDist\wb-precache.json -- not a static workbench build"
}

# --- stage -------------------------------------------------------------------
$version = (Get-Content (Join-Path $bridge "SpireMentor.json") | ConvertFrom-Json).version
$staging = Join-Path $bridge "staging"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
$simDir = New-Item -ItemType Directory -Path (Join-Path $staging "mods/SpireMentor") -Force
$libDir = New-Item -ItemType Directory -Path (Join-Path $staging "mods/BaseLib") -Force

Copy-Item $dll $simDir
Copy-Item (Join-Path $bridge "SpireMentor.json") $simDir
Copy-Item (Join-Path $bridge "SpireMentor.conf.template") $simDir
Copy-Item -Recurse $advisorDist (Join-Path $simDir "SimAdvisor")
# The frame without its debug symbols / XML docs.
$frameDir = New-Item -ItemType Directory -Path (Join-Path $simDir "SpireMentorFrame") -Force
Copy-Item -Recurse (Join-Path $frameOut "*") $frameDir -Exclude *.pdb, *.xml
# The nets stay out (8.5 MB each); /ppo/index.json stays, and the control
# plane serves the files from the advisor's own data/ppo.
Copy-Item -Recurse $workbenchDist (Join-Path $simDir "workbench")
Get-ChildItem (Join-Path $simDir "workbench/ppo") -Filter *.onnx | Remove-Item
# The game reads every .json under a mod's folder as a mod manifest and logs a
# stack trace for each one that is not, so the build's .json files ship
# gzipped: the control plane serves x.json from x.json.gz (StaticSite in
# realgame_control.py). The advisor's own data is gzipped by its spec.
Get-ChildItem (Join-Path $simDir "workbench") -Recurse -File |
    Where-Object { $_.Extension -eq ".json" } | ForEach-Object {
        $src = [IO.File]::OpenRead($_.FullName)
        try {
            $dst = [IO.File]::Create("$($_.FullName).gz")
            try {
                $gz = [IO.Compression.GZipStream]::new($dst, [IO.Compression.CompressionLevel]::Optimal)
                try { $src.CopyTo($gz) } finally { $gz.Dispose() }
            } finally { $dst.Dispose() }
        } finally { $src.Dispose() }
        Remove-Item $_.FullName
    }
$strayJson = Get-ChildItem $simDir -Recurse -File |
    Where-Object { $_.Extension -eq ".json" -and $_.FullName -ne (Join-Path $simDir "SpireMentor.json") }
if ($strayJson) {
    throw "the game would read these as mod manifests -- gzip them or rename them:`n  " +
          (($strayJson | ForEach-Object { $_.FullName.Substring($simDir.FullName.Length + 1) }) -join "`n  ")
}
Copy-Item (Join-Path $baseLibDir "BaseLib.dll") $libDir
Copy-Item (Join-Path $baseLibDir "BaseLib.pck") $libDir
Copy-Item (Join-Path $baseLibDir "BaseLib.json") $libDir
if ($haveBaseLibLicense) {
    Copy-Item $baseLibLicense (Join-Path $libDir "LICENSE-BaseLib.txt")
}
Copy-Item (Join-Path $bridge "packaging/README-dist.txt") (Join-Path $staging "README.txt")

# --- zip ---------------------------------------------------------------------
if ($NoZip) {
    Write-Host "OK: staged $staging (no zip)"
    return
}
$zip = Join-Path $bridge "SpireMentor-v$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zip
$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB)
Write-Host "OK: $zip ($sizeMB MB)"
if (-not $haveBaseLibLicense) {
    Write-Warning "zip lacks LICENSE-BaseLib.txt -- do not distribute until the MIT notice is added."
}
