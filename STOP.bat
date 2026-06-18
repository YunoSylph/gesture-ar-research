@echo off
title Gesture AR - stop
echo Stopping Gesture AR services (ports 8000, 5173-5179)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach ($p in 8000,5173,5174,5175,5176,5177,5178,5179) { Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }; Write-Host 'Services stopped.'"
echo.
pause
