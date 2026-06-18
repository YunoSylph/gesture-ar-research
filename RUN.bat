@echo off
cd /d "%~dp0"
title Gesture AR - launch
echo.
echo ==========================================
echo   Gesture AR - quick launch
echo ==========================================
echo Dependencies are NOT reinstalled (fast path).
echo For first-time setup run scripts\start_ar_demo.ps1 without -SkipInstall.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_ar_demo.ps1" -SkipInstall -Restart
echo.
echo If a browser tab opened, the system is running.
echo To stop the services, run STOP.bat
echo.
pause
