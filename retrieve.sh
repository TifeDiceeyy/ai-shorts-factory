#!/usr/bin/env bash
# Phase 1 retrieval: ./retrieve.sh soap   or   ./retrieve.sh "roman concrete"
# Requires SEARCH_PROVIDER=tavily + SEARCH_API_KEY + an explicit BUDGET_CAP_USD
# in .env — there is no stub for search, so this refuses to run without them.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "no .venv found — run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
PYTHONPATH=src python3 -m shorts_factory.retrieval "${1:?usage: ./retrieve.sh <topic>}"
