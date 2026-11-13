# Step 4: register the "HaramCrowdMonitor" scheduled task for the current user (no admin needed).
#
# It runs .venv\Scripts\pythonw.exe scripts\supervise.py (no window), which starts and restarts
# both collectors and keeps the PC awake while on AC power.
#   Triggers: at logon of this user, and every 5 min as a watchdog. If the supervisor is already
#             running, the task does not start a second copy (MultipleInstances IgnoreNew, plus the
#             supervisor's own single-instance lock).
#   Settings: no time limit, runs on battery too, starts as soon as possible after a missed start,
#             restarts up to 999 times (1 min apart) if the task itself fails; priority 4 = normal, like any program
#             (Task Scheduler's default 7 is below normal; user, 2026-10-28: everything at normal priority).
#
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
# Remove:                         powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "Not found: $pythonw (create the venv first)" }
$user = "$env:USERDOMAIN\$env:USERNAME"

$action = New-ScheduledTaskAction -Execute $pythonw -Argument "scripts\supervise.py" -WorkingDirectory $root
$atLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$watchdog = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settingsArgs = @{
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    StartWhenAvailable         = $true
    ExecutionTimeLimit         = [TimeSpan]::Zero
    MultipleInstances          = "IgnoreNew"
    RestartCount               = 999
    RestartInterval            = (New-TimeSpan -Minutes 1)
    Priority                   = 4
}
$settings = New-ScheduledTaskSettingsSet @settingsArgs
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName "HaramCrowdMonitor" -Action $action -Trigger $atLogon, $watchdog `
    -Settings $settings -Principal $principal -Force `
    -Description "Haram Crowd Monitor: crowd + traffic/prayer collectors (scripts\supervise.py)" | Out-Null

$t = Get-ScheduledTask -TaskName "HaramCrowdMonitor"
Write-Output "Registered task '$($t.TaskName)' for $user, state: $($t.State)"
Write-Output ("Triggers: " + (($t.Triggers | ForEach-Object { $_.CimClass.CimClassName + " repeat=" + $_.Repetition.Interval }) -join "; "))
