#!/usr/bin/env bash
# Stop the Gesture AR demo on macOS / Linux (bash equivalent of STOP.bat).
set -uo pipefail

for port in 8000 5173 5174 5175 5176 5177 5178 5179; do
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping process on port $port (pid: $pids)"
    kill $pids 2>/dev/null || true
  fi
done
echo "Done."
