#!/usr/bin/env bash
# Start the Gesture AR demo on macOS / Linux (bash equivalent of start_ar_demo.ps1).
# Usage:
#   ./scripts/start_ar_demo.sh            # start backend + frontend, open browser
#   NO_BROWSER=1 ./scripts/start_ar_demo.sh
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$ROOT/demo/ar_interaction_app"
PYTHON="$ROOT/.venv-gesture-ar/bin/python"
LOG_DIR="$ROOT/artifacts/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python venv not found at $PYTHON" >&2
  echo "Create it first: python3.11 -m venv .venv-gesture-ar && source .venv-gesture-ar/bin/activate && pip install -r requirements/macos-arm64.txt" >&2
  exit 1
fi

wait_http() {
  local url="$1" deadline=$(( $(date +%s) + 45 ))
  until curl -sf -o /dev/null "$url"; do
    [[ $(date +%s) -ge $deadline ]] && return 1
    sleep 0.5
  done
}

# Backend (FastAPI websocket server).
if ! curl -sf -o /dev/null "http://127.0.0.1:$BACKEND_PORT/api/health" 2>/dev/null; then
  echo "Starting backend on :$BACKEND_PORT ..."
  ( cd "$ROOT" && "$PYTHON" -m research_pipeline.cli.serve_live --host 127.0.0.1 --port "$BACKEND_PORT" \
      > "$LOG_DIR/ar_backend.log" 2>&1 & )
fi
wait_http "http://127.0.0.1:$BACKEND_PORT/api/health" || { echo "Backend did not become ready; see $LOG_DIR/ar_backend.log" >&2; exit 1; }

# Frontend (Vite dev server).
if [[ ! -d "$APP_DIR/node_modules" ]]; then
  ( cd "$APP_DIR" && npm install )
fi
if ! curl -sf -o /dev/null "http://127.0.0.1:$FRONTEND_PORT" 2>/dev/null; then
  echo "Starting frontend on :$FRONTEND_PORT ..."
  ( cd "$APP_DIR" && npm run dev -- --port "$FRONTEND_PORT" --strictPort \
      > "$LOG_DIR/ar_frontend_$FRONTEND_PORT.log" 2>&1 & )
fi
wait_http "http://127.0.0.1:$FRONTEND_PORT" || { echo "Frontend did not become ready; see $LOG_DIR/ar_frontend_$FRONTEND_PORT.log" >&2; exit 1; }

URL="http://127.0.0.1:$FRONTEND_PORT"
echo ""
echo "Gesture AR is running."
echo "Backend:  http://127.0.0.1:$BACKEND_PORT/api/health"
echo "Frontend: $URL"
echo "Stop with: ./scripts/stop_ar_demo.sh"

if [[ "${NO_BROWSER:-0}" != "1" ]]; then
  if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi
fi
