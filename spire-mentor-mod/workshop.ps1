# Assemble (and optionally upload) the Steam Workshop item for SpireMentor.
#
# Output: workshop-workspace/, a workspace for Mega Crit's uploader
# (https://github.com/megacrit/sts2-mod-uploader):
#
#   content/          what Steam ships: SpireMentor.dll, SpireMentor.json,
#                     SpireMentor.conf.template, SimAdvisor/ (packaged advisor).
#                     The game reads it recursively from the Workshop folder.
#   workshop.json     packaging/workshop/workshop.json + description.bbcode
#                     + -ChangeNote / -Visibility
#   image.png         packaging/workshop/image.png if present, else the web
#                     app icon (Steam caps it at 1 MB)
#   mod_id.txt        the Workshop item id, kept in packaging/workshop/ so every
#                     release updates the same item instead of creating a new one
#
# BaseLib is NOT bundled: workshop.json lists Alchyr's Workshop item
# (3737335127) as a required item, so Steam installs it alongside.
#
# The staging itself is package.ps1 -NoZip (same build, same advisor checks).
#
#   ./workshop.ps1                          # assemble only, inspect the workspace
#   ./workshop.ps1 -Upload -ChangeNote "…"  # assemble + upload
#
# -Upload needs Steam running, logged in as the item's owner, and the uploader:
# -Uploader <path to ModUploader.exe> or $env:STS2_MOD_UPLOADER.
param(
    # Rebuild the advisor exe first (passed through to package.ps1).
    [switch]$BuildAdvisor,
    # Run ModUploader.exe on the assembled workspace.
    [switch]$Upload,
    [string]$Uploader = $env:STS2_MOD_UPLOADER,
    # Shown to subscribers on the item's change notes tab.
    [string]$ChangeNote = "",
    # Omitted: a first upload is private, later uploads keep what the item has
    # (set on the Workshop page or here), so a re-upload never hides a public item.
    [ValidateSet("private", "unlisted", "friends_only", "public")]
    [string]$Visibility
)
$ErrorActionPreference = "Stop"
$bridge = $PSScriptRoot
$repo = Split-Path -Parent $bridge
$source = Join-Path $bridge "packaging/workshop"
$workspace = Join-Path $bridge "workshop-workspace"
$modIdFile = Join-Path $source "mod_id.txt"

if ($Upload -and -not ($Uploader -and (Test-Path $Uploader))) {
    throw "ModUploader.exe not found ('$Uploader') -- download it from " +
          "https://github.com/megacrit/sts2-mod-uploader/releases and pass -Uploader " +
          "or set STS2_MOD_UPLOADER."
}

# --- build + stage (package.ps1 checks the DLL and the packaged advisor) ------
$packageArgs = @{ NoZip = $true }
if ($BuildAdvisor) { $packageArgs.BuildAdvisor = $true }
& (Join-Path $bridge "package.ps1") @packageArgs
$staged = Join-Path $bridge "staging/mods/SpireMentor"
if (-not (Test-Path (Join-Path $staged "SimAdvisor/SimAdvisor.exe"))) {
    throw "package.ps1 did not stage $staged\SimAdvisor\SimAdvisor.exe"
}

# --- workspace ----------------------------------------------------------------
if (Test-Path $workspace) { Remove-Item -Recurse -Force $workspace }
$content = New-Item -ItemType Directory -Path (Join-Path $workspace "content") -Force
Copy-Item -Recurse (Join-Path $staged "*") $content

$image = Join-Path $source "image.png"
if (-not (Test-Path $image)) {
    $image = Join-Path $repo "sts2-pycli/web/public/icons/icon-512.png"
    Write-Warning "no packaging/workshop/image.png -- using the web app icon as the Workshop preview"
}
if ((Get-Item $image).Length -ge 1MB) { throw "$image is 1 MB or more; Steam rejects it" }
Copy-Item $image (Join-Path $workspace "image.png")

$config = Get-Content (Join-Path $source "workshop.json") -Raw | ConvertFrom-Json
$config | Add-Member description (Get-Content (Join-Path $source "description.bbcode") -Raw).TrimEnd()
$version = (Get-Content (Join-Path $bridge "SpireMentor.json") | ConvertFrom-Json).version
$config | Add-Member changeNote $(if ($ChangeNote) { "v${version}: $ChangeNote" } else { "v$version" })
if ($Visibility) {
    $config | Add-Member visibility $Visibility
} elseif (-not (Test-Path $modIdFile)) {
    $config | Add-Member visibility "private"
}
# The uploader parses with System.Text.Json: write UTF-8 without a BOM.
[IO.File]::WriteAllText((Join-Path $workspace "workshop.json"),
    ($config | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))

if (Test-Path $modIdFile) { Copy-Item $modIdFile $workspace }

$sizeMB = [math]::Round((Get-ChildItem $content -Recurse | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "OK: $workspace (content $sizeMB MB, v$version)"
if (-not $Upload) { return }

# --- upload -------------------------------------------------------------------
# From the uploader's own folder, as its README says: Steamworks looks for
# steam_appid.txt in the working directory, and the log lands there too.
$uploaderDir = Split-Path -Parent (Resolve-Path $Uploader)
Push-Location $uploaderDir
try {
    & (Resolve-Path $Uploader) upload -w $workspace
    if ($LASTEXITCODE -ne 0) {
        throw "upload failed -- see $uploaderDir\mod-uploader.log"
    }
} finally {
    Pop-Location
}
# A first upload creates the item and writes its id into the workspace; keep
# it in packaging/ (committed) so the next release updates the same item.
$newId = Join-Path $workspace "mod_id.txt"
if ((Test-Path $newId) -and -not (Test-Path $modIdFile)) {
    Copy-Item $newId $modIdFile
    Write-Host "New Workshop item $((Get-Content $newId).Trim()) -- commit packaging/workshop/mod_id.txt"
}
