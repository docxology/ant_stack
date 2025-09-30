#!/usr/bin/env bash
set -euo pipefail
python3 -m complexity_energetics.src.ce.runner complexity_energetics/manifest.example.yaml --out complexity_energetics/out
echo "Results: complexity_energetics/out/results.csv"
echo "JSON: complexity_energetics/out/summary.json"

