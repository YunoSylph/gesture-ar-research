@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%demo\ar_interaction_app"
set "VENV_DIR=%ROOT%.venv-gesture-ar"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "BACKEND_URL=http://127.0.0.1:8000/api/health"
set "FRONTEND_URL=http://127.0.0.1:5173/"

title Gesture AR Project Launcher
cd /d "%ROOT%"

echo.
echo ==========================================
echo   Gesture AR Research - one file launcher
echo ==========================================
echo Project: %ROOT%
echo.

call :ensure_python_env || goto :fail
call :ensure_frontend_deps || goto :fail
call :start_backend || goto :fail
call :start_frontend || goto :fail
call :wait_for_url "%BACKEND_URL%" 90 "backend" || goto :fail
call :wait_for_url "%FRONTEND_URL%" 90 "frontend" || goto :fail

echo.
echo Project is running.
echo UI:      %FRONTEND_URL%
echo Backend: %BACKEND_URL%
echo.
echo Use the UI in TARC mode for the intended demo flow.
echo Close the "Gesture AR Backend" and "Gesture AR Frontend" windows to stop the project.
echo.
start "" "%FRONTEND_URL%"
if not "%GESTURE_AR_NO_PAUSE%"=="1" pause
exit /b 0

:ensure_python_env
echo [1/4] Checking Python environment...
if exist "%VENV_PY%" goto :install_python_deps

echo Creating local virtual environment: %VENV_DIR%
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -m venv "%VENV_DIR%" >nul 2>nul
)

if not exist "%VENV_PY%" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.11 and run this file again.
    exit /b 1
  )
  python -m venv "%VENV_DIR%"
)

if not exist "%VENV_PY%" (
  echo ERROR: Could not create .venv-gesture-ar. Install Python 3.11 and run this file again.
  exit /b 1
)

:install_python_deps
"%VENV_PY%" -c "import fastapi, uvicorn, cv2, mediapipe, websockets" >nul 2>nul
if errorlevel 1 (
  echo Installing Python dependencies. This can take several minutes on first launch...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  "%VENV_PY%" -m pip install -e ".[serve,vision,dev]"
  if errorlevel 1 (
    echo.
    echo ERROR: Python dependencies failed to install.
    echo If the error mentions mediapipe, install Python 3.11, delete .venv-gesture-ar, and run START_PROJECT.bat again.
    exit /b 1
  )
)
exit /b 0

:ensure_frontend_deps
echo [2/4] Checking frontend dependencies...
where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: npm was not found. Install Node.js LTS and run this file again.
  exit /b 1
)

call :is_port_open 5173
if not errorlevel 1 (
  echo Frontend already responds on port 5173; skipping npm dependency check.
  exit /b 0
)

if not exist "%APP_DIR%\node_modules\" (
  echo Installing frontend dependencies...
  pushd "%APP_DIR%"
  call npm ci
  set "NPM_STATUS=%ERRORLEVEL%"
  popd
  if not "%NPM_STATUS%"=="0" exit /b 1
)

if not exist "%APP_DIR%\node_modules\.bin\vite.cmd" (
  echo Repairing frontend dependencies...
  pushd "%APP_DIR%"
  call npm install --no-audit --no-fund
  set "NPM_STATUS=%ERRORLEVEL%"
  popd
  if not "%NPM_STATUS%"=="0" exit /b 1
)
exit /b 0

:start_backend
echo [3/4] Starting backend...
call :is_port_open 8000
if not errorlevel 1 (
  echo Backend already responds on port 8000.
  exit /b 0
)
start "Gesture AR Backend" /D "%ROOT%" cmd /k ""%VENV_PY%" -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000 1>CON 2>CON"
exit /b 0

:start_frontend
echo [4/4] Starting frontend...
call :is_port_open 5173
if not errorlevel 1 (
  echo Frontend already responds on port 5173.
  exit /b 0
)
start "Gesture AR Frontend" /D "%APP_DIR%" cmd /k "npm run dev -- --host 127.0.0.1 --port 5173 1>CON 2>CON"
exit /b 0

:is_port_open
set "CHECK_PORT=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port = [int]$env:CHECK_PORT; $client = New-Object Net.Sockets.TcpClient; try { $async = $client.BeginConnect('127.0.0.1', $port, $null, $null); if ($async.AsyncWaitHandle.WaitOne(350) -and $client.Connected) { $client.Close(); exit 0 } exit 1 } catch { exit 1 }"
exit /b %ERRORLEVEL%

:wait_for_url
echo Waiting for %~3...
set "WAIT_URL=%~1"
set "WAIT_SECONDS=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$url = $env:WAIT_URL; $deadline = (Get-Date).AddSeconds([int]$env:WAIT_SECONDS); while ((Get-Date) -lt $deadline) { try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch { } Start-Sleep -Milliseconds 700 }; exit 1"
if errorlevel 1 (
  echo ERROR: %~3 did not become ready in time.
  exit /b 1
)
exit /b 0

:fail
echo.
echo Launch failed. Read the message above, fix the missing dependency, and run START_PROJECT.bat again.
if not "%GESTURE_AR_NO_PAUSE%"=="1" pause
exit /b 1
