#!/usr/bin/env bash
# One command, end to end: ./run.sh soap
# Produces artifacts/<topic>/{<topic>.mp4, <topic>.script.json, captions.*,
# cost-report.json, verification-report.json}. Exit code is non-zero if any
# verification criterion fails or the topic is safety-blocked.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "no .venv found — run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
PYTHONPATH=src python3 -m shorts_factory.pipeline "${1:?usage: ./run.sh <topic>}"
