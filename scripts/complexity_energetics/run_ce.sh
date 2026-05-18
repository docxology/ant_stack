#!/usr/bin/env bash
set -euo pipefail
uv run antstack-ce papers/complexity_energetics/manifest.example.yaml --out papers/complexity_energetics/out
echo "Results: papers/complexity_energetics/out/results.csv"
echo "JSON: papers/complexity_energetics/out/summary.json"
