#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY [antstack-ce options]" >&2
    exit 2
fi
output="$1"
shift
exec uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out "$output" "$@"
