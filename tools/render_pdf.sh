#!/bin/bash
# The installed builder owns configuration, preprocessing, validation and publication.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Set PYTHON_BIN to the project's installed Python interpreter." >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    exec "$PYTHON_BIN" -m antstack_core.cli.build --project-root "$PWD"
fi
if [ "$1" = "--help" ]; then
    exec "$PYTHON_BIN" -m antstack_core.cli.build --help
fi
for paper in "$@"; do
    "$PYTHON_BIN" -m antstack_core.cli.build --project-root "$PWD" --paper "$paper"
done
