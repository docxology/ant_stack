#!/usr/bin/env python3
"""Manuscript figure generation orchestrator.

Generates a complete enhanced analytical figure family in an explicitly selected
fresh directory. Existing historical assets are never overwritten.

Usage:
    python scripts/complexity_energetics/generate_manuscript_figures.py --output NEW_DIR [--format png|svg|pdf]
"""

import argparse
import sys
from pathlib import Path

# Use the current checkout's package when executing this wrapper directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from antstack_core.analysis.ce_manuscript_figures import run


def main():
    parser = argparse.ArgumentParser(description="Generate all manuscript figures")
    parser.add_argument("--output", required=True,
                        help="Fresh absent or empty directory for this complete family")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png",
                       help="Output format for figures")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
