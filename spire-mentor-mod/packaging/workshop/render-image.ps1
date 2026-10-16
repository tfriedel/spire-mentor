# Render image.html to image.png (1024x1024), the Steam Workshop preview
# workshop.ps1 uploads. Needs Microsoft Edge (every Windows 11 has it).
$ErrorActionPreference = "Stop"
$edge = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw "Microsoft Edge not found" }

$html = Join-Path $PSScriptRoot "image.html"
$png = Join-Path $PSScriptRoot "image.png"
$uri = ([Uri]$html).AbsoluteUri
# Wait for the process: Edge returns before the file is written otherwise.
Start-Process -Wait -FilePath $edge -ArgumentList @(
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--force-device-scale-factor=1", "--window-size=1024,1024",
    "--screenshot=$png", $uri
)
if (-not (Test-Path $png)) { throw "Edge wrote no $png" }
$kb = [math]::Round((Get-Item $png).Length / 1KB)
if ($kb -ge 1024) { throw "$png is $kb KB; Steam needs it under 1 MB" }
Write-Host "OK: $png ($kb KB)"
