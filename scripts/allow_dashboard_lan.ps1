# Let phones on the home Wi-Fi open the dashboard (user request 2026-10-25). Run once, as administrator
# ("Allow Dashboard on Wi-Fi.bat" asks for it). Home network only:
#   - the Wi-Fi connection is set to the Private network profile;
#   - inbound TCP 8000 is allowed on the Private profile only (never Public), from the local subnet only.
# No port forwarding, no tunnels. Undo: Remove-NetFirewallRule -DisplayName 'Haram Monitor dashboard (LAN)'
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$alias = 'Wi-Fi'
$rule = 'Haram Monitor dashboard (LAN)'

Set-NetConnectionProfile -InterfaceAlias $alias -NetworkCategory Private
Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
    -Profile Private -RemoteAddress LocalSubnet | Out-Null

$p = Get-NetConnectionProfile -InterfaceAlias $alias
$ip = (Get-NetIPAddress -InterfaceAlias $alias -AddressFamily IPv4).IPAddress
Write-Host ""
Write-Host "Wi-Fi '$($p.Name)' is now: $($p.NetworkCategory)"
Write-Host "Firewall rule '$rule': TCP 8000 inbound, Private profile only, local subnet only"
Write-Host "Phone (same Wi-Fi): http://${ip}:8000"
Write-Host ""
Read-Host "Press Enter to close"
