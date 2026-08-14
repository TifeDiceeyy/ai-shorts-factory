#!/usr/bin/env bash
# Phase 4 review dashboard: ./dashboard.sh  (serves on http://127.0.0.1:8420)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "no .venv found — run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
PYTHONPATH=src uvicorn shorts_factory.dashboard.app:app --host 127.0.0.1 --port 8420 --reload
