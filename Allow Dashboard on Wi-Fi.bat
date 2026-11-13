@echo off
rem Run once: sets the home Wi-Fi to Private and allows the dashboard port (8000) for the Private profile only.
rem Windows asks for administrator rights (click Yes). Script: scripts\allow_dashboard_lan.ps1
cd /d "%~dp0"
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0scripts\allow_dashboard_lan.ps1\"'"
