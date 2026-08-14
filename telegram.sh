#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
exec .venv/bin/python -m shorts_factory.telegram_bot
