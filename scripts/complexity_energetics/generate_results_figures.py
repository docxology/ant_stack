#!/usr/bin/env python3
"""Results figure generation orchestrator.

Generates a complete PNG figure family from explicitly bounded analytical models.
Select a fresh directory; existing historical assets are never overwritten.

Usage:
    python scripts/complexity_energetics/generate_results_figures.py --output NEW_DIR
"""

import argparse
import sys
from pathlib import Path

# Use the current checkout's package when executing this wrapper directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from antstack_core.analysis.ce_results_figures import run


def main():
    parser = argparse.ArgumentParser(description="Generate Results.md figures")
    parser.add_argument("--output", required=True,
                        help="Fresh absent or empty directory for this complete family")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
