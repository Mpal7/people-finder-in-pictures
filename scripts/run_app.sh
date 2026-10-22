#!/usr/bin/env bash
# Launch the local web UI (127.0.0.1 only). Stop with Ctrl-C.
cd "$(dirname "$0")/.."
exec .venv/bin/python webapp.py "$@"
