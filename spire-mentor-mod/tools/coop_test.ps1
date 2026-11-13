# coop_test.ps1 -- a local co-op run against dummy partners, on one machine.
#
# Starts 2-4 game instances with the game's own `--fastmp` dev launch
# (NMainMenu.CheckCommandLineArgs): ENet on 127.0.0.1:33771 instead of Steam
# lobbies, so no second Steam account is needed.
#
#   host   --fastmp host_standard        you play here (Spire Mentor as usual)
#   dummy  --fastmp join --clientId 1000 Spire Mentor plays it hands-free
#          --spire-mentor-autoplay       (SimCoopDummy readies the lobby, then
#                                        SimAdvisorAutoplay = F8 + F7)
#   dummy2 --fastmp join --clientId 1001 the same, with -Players 3 or 4
#   dummy3 --fastmp join --clientId 1002 ... and -Players 4
#
# Pick your character in the host window and Embark; the run starts once the
# dummies are ready too. Plan and caveats: docs/plans/2026-10-23-coop-multiplayer.md.
#
# All instances run under the same Steam account and so share one profile;
# the dummies write no saves (--spire-mentor-no-saves, SimNoSaves) so they
# never race on it.
#
# All instances use the mod in <GameDir>/mods. Each dummy's bridge port comes
# from --spire-mentor-port (they share SpireMentor.conf): $DummyPort, then
# +10 per extra dummy. Each instance logs to its own file (Godot's
# --log-file) instead of the shared godot.log.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File spire-mentor-mod/tools/coop_test.ps1
#   ... -Players 3                  2-4 players: the host plus N-1 dummies
#                                   (GH #1017; default 2)
#   ... -Character silent           the dummies' character (default:
#                                   preselected); "silent,defect" gives
#                                   each dummy its own, in join order
#   ... -NoSteamDummy               start the dummies with --force-steam off;
#                                   they then see no Workshop mods (BaseLib!)
#   ... -Speed 1                    game speed of every instance (default 3x)
#   ... -BotHost                    the host plays hands-free too (unattended
#                                   test of every seat; it embarks once all
#                                   -Players are in the lobby)
#   ... -Mode blitz                 the run's game mode: standard (default),
#                                   blitz or dread -- v0.113.0+, whose
#                                   --fastmp knows host_blitz / host_dread

param(
    [string]$GameDir = $(if ($env:STS2_GAME_DIR) { $env:STS2_GAME_DIR } else { "C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2" }),
    [ValidateRange(2, 4)]
    [int]$Players = 2,
    [string]$Character = "",
    [int]$DummyPort = 9012,
    # Game speed for both instances (0.1-4.0; F9/F10 change it live).
    [double]$Speed = 3.0,
    [string]$LogDir = "",
    [switch]$NoSteamDummy,
    [ValidateSet("standard", "blitz", "dread")]
    [string]$Mode = "standard",
    [switch]$BotHost
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $GameDir "SlayTheSpire2.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Game not found: $exe (pass -GameDir or set STS2_GAME_DIR)."
}
if (-not $NoSteamDummy -and -not (Get-Process steam -ErrorAction SilentlyContinue)) {
    # A direct launch without a running Steam client fails Steamworks init
    # (ascension_climb.py launch_game, issue 9).
    Write-Error "Steam is not running; start it first."
}
if (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) {
    Write-Error "A game instance is already running; close it first."
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not $LogDir) {
    $LogDir = Join-Path $repo ("logs/{0}-coop" -f (Get-Date -Format "yyyy-MM-dd-HHmmss"))
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
$hostLog = Join-Path $LogDir "host.godot.log"

# Every instance and advisor logs constantly, and a full disk kills a run
# mid-fight with nothing in its logs; worktree builds fill it fast (2026-10-29).
# Check each drive the run writes to: the game logs, the advisors' run logs
# and stdout (%LOCALAPPDATA%, or the repo for a dev advisor).
$drives = @($LogDir, $repo, $env:LOCALAPPDATA) | Where-Object { $_ } |
    ForEach-Object { (Split-Path -Qualifier (Resolve-Path $_).Path).TrimEnd(':') } | Sort-Object -Unique
foreach ($d in $drives) {
    $freeGb = (Get-PSDrive $d).Free / 1GB
    if ($freeGb -lt 5) {
        Write-Error ("Only {0:N1} GB free on {1}:; free at least 5 GB first." -f $freeGb, $d)
    }
}

# How many instances this run has, for coop_watch.sh (whose logs appear one
# by one as the instances start).
Set-Content -Path (Join-Path $LogDir "players.txt") -Value "$Players" -Encoding ascii

Write-Host "Starting host ($Players players)..."
$speedArg = $Speed.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$hostArgs = @("--fastmp", "host_$Mode", "--log-file", "`"$hostLog`"", "--spire-mentor-speed", $speedArg,
    "--spire-mentor-coop-players", "$Players")
if ($BotHost) { $hostArgs += "--spire-mentor-autoplay" }
Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $hostArgs | Out-Null

# The dummy may only join once the host's ENet socket is up.
$deadline = (Get-Date).AddSeconds(120)
while (-not (Get-NetUDPEndpoint -LocalPort 33771 -ErrorAction SilentlyContinue)) {
    if ((Get-Date) -gt $deadline) {
        Write-Error "The host never opened its co-op lobby (UDP 33771). See $hostLog"
    }
    Start-Sleep -Seconds 1
}

$characters = @($Character -split "," | ForEach-Object { $_.Trim() })
$dummyLogs = @()
for ($k = 0; $k -lt $Players - 1; $k++) {
    # dummy.godot.log for the first, as before; dummy2, dummy3 after it.
    $name = if ($k -eq 0) { "dummy" } else { "dummy$($k + 1)" }
    $dummyLog = Join-Path $LogDir "$name.godot.log"
    $dummyLogs += $dummyLog
    $port = $DummyPort + 10 * $k
    Write-Host "Host lobby is up; starting $name (client $(1000 + $k), port $port)..."
    $dummyArgs = @(
        "--fastmp", "join", "--clientId", "$(1000 + $k)",
        "--spire-mentor-autoplay",
        "--spire-mentor-no-saves",
        "--spire-mentor-port", "$port",
        "--spire-mentor-speed", $speedArg,
        "--log-file", "`"$dummyLog`""
    )
    # One name for every dummy, or one per dummy in join order.
    $char = if ($characters.Count -gt 1) { $characters[[Math]::Min($k, $characters.Count - 1)] } else { $characters[0] }
    if ($char) { $dummyArgs += @("--spire-mentor-character", $char) }
    if ($NoSteamDummy) { $dummyArgs += @("--force-steam", "off") }
    Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $dummyArgs | Out-Null
    # Join one at a time, so the lobby's seats follow the client ids.
    if ($k -lt $Players - 2) { Start-Sleep -Seconds 8 }
}

Write-Host ""
if (-not $BotHost) { Write-Host "Pick a character in the HOST window and Embark." }
Write-Host "Logs:"
Write-Host "  host    $hostLog"
foreach ($l in $dummyLogs) { Write-Host "  $([IO.Path]::GetFileName($l).Replace('.godot.log', '').PadRight(7)) $l" }
Write-Host "Mod lines only:  Select-String '\[SpireMentor\]' '$($dummyLogs[0])' | Select-Object -Last 40"
