param(
  [int]$BackendPort = 8000,
  [int[]]$FrontendPorts = @(5173, 5174, 5175, 5176, 5177, 5178, 5179),
  [switch]$NoBrowser,
  [switch]$Restart
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$AppDir = Join-Path $Root "demo\ar_interaction_app"
$Python = Join-Path $Root ".venv311\Scripts\python.exe"
$LogDir = Join-Path $Root "artifacts\logs"

New-Item -ItemType Directory -Force $LogDir | Out-Null

function Test-ListeningPort {
  param([int]$Port)
  return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Stop-PortProcess {
  param([int]$Port)
  $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($connection in $connections) {
    if ($connection.OwningProcess) {
      Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
    }
  }
}

function Wait-Http {
  param([string]$Url, [int]$Seconds = 30)
  $deadline = (Get-Date).AddSeconds($Seconds)
  do {
    try {
      $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  } while ((Get-Date) -lt $deadline)
  return $false
}

if (-not (Test-Path $Python)) {
  throw "Python environment not found: $Python"
}

if (-not (Test-Path $AppDir)) {
  throw "Frontend app not found: $AppDir"
}

if ($Restart) {
  Stop-PortProcess -Port $BackendPort
  foreach ($port in $FrontendPorts) {
    Stop-PortProcess -Port $port
  }
  Start-Sleep -Seconds 1
}

if (-not (Test-ListeningPort -Port $BackendPort)) {
  $backendLog = Join-Path $LogDir "ar_backend.log"
  $backendCommand = "Set-Location -LiteralPath '$Root'; & '$Python' -m research_pipeline.cli.serve_live --host 127.0.0.1 --port $BackendPort *> '$backendLog'"
  Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden
}

$healthUrl = "http://127.0.0.1:$BackendPort/api/health"
if (-not (Wait-Http -Url $healthUrl -Seconds 45)) {
  throw "Backend did not become ready. Check artifacts\logs\ar_backend.log"
}

if (-not (Test-Path (Join-Path $AppDir "node_modules"))) {
  Push-Location $AppDir
  try {
    npm install
  } finally {
    Pop-Location
  }
}

$frontendPort = $null
$frontendAlreadyRunning = $false
foreach ($port in $FrontendPorts) {
  if (Test-ListeningPort -Port $port) {
    $frontendPort = $port
    $frontendAlreadyRunning = $true
    break
  }
}

if ($null -eq $frontendPort) {
  foreach ($port in $FrontendPorts) {
    if (-not (Test-ListeningPort -Port $port)) {
      $frontendPort = $port
      break
    }
  }
}

if ($null -eq $frontendPort) {
  throw "No frontend port available in: $($FrontendPorts -join ', ')"
}

if (-not $frontendAlreadyRunning) {
  $frontendLog = Join-Path $LogDir "ar_frontend_$frontendPort.log"
  $frontendCommand = "Set-Location -LiteralPath '$AppDir'; & npm.cmd run dev -- --port $frontendPort --strictPort *> '$frontendLog'"

  Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand) `
    -WorkingDirectory $AppDir `
    -WindowStyle Hidden
}

$frontendUrl = "http://127.0.0.1:$frontendPort"
if (-not (Wait-Http -Url $frontendUrl -Seconds 45)) {
  if ($frontendAlreadyRunning) {
    throw "Existing frontend on port $frontendPort did not respond."
  }
  throw "Frontend did not become ready. Check artifacts\logs\ar_frontend_$frontendPort.log"
}

[Console]::Out.WriteLine("")
[Console]::Out.WriteLine("Gesture AR is running.")
[Console]::Out.WriteLine("Backend:  $healthUrl")
[Console]::Out.WriteLine("Frontend: $frontendUrl")
[Console]::Out.WriteLine("")
[Console]::Out.WriteLine("Choose AR Task -> Start Task. Use Advanced Controls only for manual tuning.")
[Console]::Out.WriteLine("If a stale process is on the ports, rerun with: -Restart")

if (-not $NoBrowser) {
  Start-Process $frontendUrl
}
