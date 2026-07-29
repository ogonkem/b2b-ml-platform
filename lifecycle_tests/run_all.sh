#!/usr/bin/env bash
# Runs every lifecycle stage in dependency order, stopping at the first
# failure. Requires `docker compose up -d` already running.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONIOENCODING=utf-8
PYTHON="${PYTHON:-.venv/Scripts/python}"

for stage in lifecycle_tests/0[1-7]_*.py; do
    echo
    echo "############################################################"
    echo "# $stage"
    echo "############################################################"
    "$PYTHON" "$stage"
done

echo
echo "All lifecycle stages passed."
