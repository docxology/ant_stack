#!/usr/bin/env python3
"""Manuscript figure generation orchestrator.

Generates all figures for the complexity_energetics manuscript using enhanced methods.
Creates publication-quality figures with proper captions and statistical analysis.

Usage:
    python scripts/generate_manuscript_figures.py [--output assets_dir] [--format png]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_manuscript_figures import run


def main():
    parser = argparse.ArgumentParser(description="Generate all manuscript figures")
    parser.add_argument("--output", default="complexity_energetics/assets",
                       help="Output directory for figures")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png",
                       help="Output format for figures")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
