#!/usr/bin/env bash
# One-time setup: virtualenv + Python dependencies + model weights.
# After this finishes, start the UI with ./scripts/run_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "python3 not found - install Python 3.10-3.12 first" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import struct, sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and struct.calcsize("P") == 8 else 1)'; then
  echo "64-bit Python 3.10-3.12 is required (found $("$PYTHON" -V 2>&1))" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "creating virtualenv (.venv)..."
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "downloading model weights (one time, ~1 GB)..."
.venv/bin/python recon/setup_models.py

echo
echo "Setup complete. Start the UI with:"
echo "  ./scripts/run_app.sh"
