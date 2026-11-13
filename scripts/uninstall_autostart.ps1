# Remove the "HaramCrowdMonitor" scheduled task and stop the running supervisor (its job object
# then stops both collectors).
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1
$task = Get-ScheduledTask -TaskName "HaramCrowdMonitor" -ErrorAction SilentlyContinue
if ($task) {
    Stop-ScheduledTask -TaskName "HaramCrowdMonitor"
    Unregister-ScheduledTask -TaskName "HaramCrowdMonitor" -Confirm:$false
    Write-Output "Task removed."
} else {
    Write-Output "Task not found."
}
$lock = Join-Path (Split-Path -Parent $PSScriptRoot) "data\.pid_supervisor"
if (Test-Path $lock) {
    $procId = (Get-Content $lock -Raw).Trim()
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $procId -Confirm:$false
        Write-Output "Stopped supervisor (pid $procId)."
    }
}
