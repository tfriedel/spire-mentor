# ai_partner_test.ps1 -- a co-op run with AI partners in ONE game instance.
#
# docs/plans/2026-11-05-ai-coop-partners.md: `--spire-mentor-ai-partners N`
# hosts a co-op lobby from the main menu exactly as the host menu's "With AI
# partners" button does (SimAiPartnerSession, in-memory transport, no port),
# and 1-3 AI partners join it from inside the same process. No second
# SlayTheSpire2.exe (compare tools/coop_test.ps1).
#
#   ... -Partners 2                 AI partners (1-3; default 1)
#   ... -Character silent,defect    each partner's character, in join order
#   ... -BotHost                    the host plays hands-free too (advisor
#                                   autoplay; embarks once everyone is in)
#   ... -Speed 1                    game speed (default 3x)
#   ... -Pick 2                     click partner 1's lobby slot twice (the
#                                   character picker); no -BotHost
#   ... -Pace end-turn -GoTest -BotHost   "Partner go" each turn it waits
#
# Human players next to the AI (phase 4): -Humans N hosts the game's own
# lobby instead (`--fastmp host_standard`, ENet on 127.0.0.1:33771, as
# coop_test.ps1), the partners join it as the lobby's "Add AI partner" does
# (`--spire-mentor-ai-partner-join`), and N more game instances join as
# human clients (`--fastmp join --clientId 1000+k`):
#
#   ... -Humans 1 -BotHost          the clients play hands-free (Spire Mentor
#                                   autoplay), so the run is unattended
#   ... -Continue -BotHost          Load the saved co-op run: its AI partners
#                                   rejoin the load lobby (save / continue)
#   ... -Humans 1 -VanillaHumans    the clients run WITHOUT the mod (`--nomods`):
#                                   what a friend without Spire Mentor sees.
#                                   Click Ready in their window yourself.
#
# Launch from the PowerShell tool as `& '<path>\ai_partner_test.ps1' -BotHost`.

param(
    [string]$GameDir = $(if ($env:STS2_GAME_DIR) { $env:STS2_GAME_DIR } else { "C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2" }),
    [ValidateRange(1, 3)]
    [int]$Partners = 1,
    [string]$Character = "",
    [double]$Speed = 3.0,
    # The partners' pacing, overriding the settings: first-play, end-turn, now.
    [string]$Pace = "",
    # Click AI partner 1's lobby slot N times (the character picker), then
    # screenshot the lobby again (host_menu_lobby_picked.png). Leave out
    # -BotHost: a hands-free host embarks before the clicks land.
    [ValidateRange(0, 10)]
    [int]$Pick = 0,
    # Press "Partner go" once per turn when a partner has waited 1.5 s.
    [switch]$GoTest,
    [string]$LogDir = "",
    [ValidateRange(0, 2)]
    [int]$Humans = 0,
    # The human clients' characters (Spire Mentor's --spire-mentor-character).
    [string]$HumanCharacter = "",
    [switch]$VanillaHumans,
    [int]$HumanPort = 9012,
    # Continue the saved co-op run (the menu's Load): its AI partners rejoin.
    [switch]$Continue,
    [switch]$BotHost
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $GameDir "SlayTheSpire2.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Game not found: $exe (pass -GameDir or set STS2_GAME_DIR)."
}
if (-not (Get-Process steam -ErrorAction SilentlyContinue)) {
    Write-Error "Steam is not running; start it first."
}
if (Get-Process SlayTheSpire2 -ErrorAction SilentlyContinue) {
    Write-Error "A game instance is already running; close it first."
}
if ($Partners + $Humans -gt 3) {
    Write-Error "A lobby holds 4 players: -Partners + -Humans must be 3 or less."
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if (-not $LogDir) {
    $LogDir = Join-Path $repo ("logs/{0}-ai-partner" -f (Get-Date -Format "yyyy-MM-dd-HHmmss"))
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
$hostLog = Join-Path $LogDir "host.godot.log"

$speedArg = $Speed.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$hostArgs = @(
    "--log-file", "`"$hostLog`"",
    "--spire-mentor-speed", $speedArg,
    "--spire-mentor-coop-players", "$($Partners + $Humans + 1)",
    "--spire-mentor-ai-partner-shot", "`"$(Join-Path $LogDir 'host_menu.png')`""
)
if ($Continue) {
    # The menu's Load on the saved co-op run (Steam hosts it): the partners
    # of its saved seats rejoin; -BotHost embarks once all are back.
    $hostArgs += "--spire-mentor-ai-partner-continue"
} elseif ($Humans -gt 0) {
    $hostArgs = @("--fastmp", "host_standard", "--spire-mentor-no-saves", "--spire-mentor-ai-partner-join", "$Partners") + $hostArgs
} else {
    $hostArgs += @("--spire-mentor-ai-partners", "$Partners")
}
if ($Character) { $hostArgs += @("--spire-mentor-ai-partner-character", $Character) }
if ($Pace) { $hostArgs += @("--spire-mentor-ai-partner-pace", $Pace) }
if ($Pick -gt 0) { $hostArgs += @("--spire-mentor-ai-partner-pick", "$Pick") }
if ($GoTest) { $hostArgs += "--spire-mentor-ai-partner-go-test" }
if ($BotHost) { $hostArgs += "--spire-mentor-autoplay" }
$proc = Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $hostArgs -PassThru
$launch = @("host pid {0} args {1}" -f $proc.Id, ($hostArgs -join " "))
Write-Host "Started the host (pid $($proc.Id)) with $Partners AI partner(s)."

if ($Humans -gt 0) {
    # A client may only join once the host's ENet socket is up.
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Get-NetUDPEndpoint -LocalPort 33771 -ErrorAction SilentlyContinue)) {
        if ((Get-Date) -gt $deadline) {
            Write-Error "The host never opened its co-op lobby (UDP 33771). See $hostLog"
        }
        Start-Sleep -Seconds 1
    }
    $humanChars = @($HumanCharacter -split "," | ForEach-Object { $_.Trim() })
    for ($k = 0; $k -lt $Humans; $k++) {
        $name = "human$($k + 1)"
        $clientLog = Join-Path $LogDir "$name.godot.log"
        $clientArgs = @("--fastmp", "join", "--clientId", "$(1000 + $k)", "--log-file", "`"$clientLog`"")
        if ($VanillaHumans) {
            $clientArgs += "--nomods"
        } else {
            $clientArgs += @("--spire-mentor-autoplay", "--spire-mentor-no-saves",
                "--spire-mentor-port", "$($HumanPort + 10 * $k)", "--spire-mentor-speed", $speedArg)
            $char = if ($humanChars.Count -gt 1) { $humanChars[[Math]::Min($k, $humanChars.Count - 1)] } else { $humanChars[0] }
            if ($char) { $clientArgs += @("--spire-mentor-character", $char) }
        }
        $client = Start-Process -FilePath $exe -WorkingDirectory $GameDir -ArgumentList $clientArgs -PassThru
        $launch += "{0} pid {1} args {2}" -f $name, $client.Id, ($clientArgs -join " ")
        Write-Host "Started $name (client $(1000 + $k), pid $($client.Id))$(if ($VanillaHumans) { ', without the mod' })."
        if ($k -lt $Humans - 1) { Start-Sleep -Seconds 8 }
    }
}
Set-Content -Path (Join-Path $LogDir "launch.txt") -Encoding ascii -Value ($launch -join "`n")
if (-not $BotHost) { Write-Host "Pick a character and Embark once the partner(s) show as ready." }
Write-Host "Log: $hostLog"
Write-Host "Partner lines:  Select-String 'SimAiPartner' '$hostLog' | Select-Object -Last 40"
