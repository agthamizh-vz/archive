#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: Python venv not found at $VENV_PY"
  echo "Create it first: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Kill old ports if in use
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8501/tcp 2>/dev/null || true

echo "Starting FastAPI on :8000 ..."
"$VENV_PY" -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Starting Streamlit on :8501 ..."
"$VENV_PY" -m streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

cleanup() {
  echo "\nStopping services..."
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
  wait "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "\nRunning:"
echo "- API:       http://localhost:8000/docs"
echo "- Streamlit: http://localhost:8501"
echo "Press Ctrl+C to stop both."

wait "$API_PID" "$UI_PID"
