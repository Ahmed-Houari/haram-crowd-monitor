@echo off
rem Double-click to see if the Haram Crowd Monitor collection is running (read-only).
title Haram Monitor Status
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\status.py
echo.
pause
