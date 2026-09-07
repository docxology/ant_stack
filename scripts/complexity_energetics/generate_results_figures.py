#!/usr/bin/env python3
"""Results figure generation orchestrator.

This thin orchestrator script generates all figures referenced in Results.md
using tested methods from the ce/ module. Each figure is generated through
validated computational models with comprehensive analysis.

Usage:
    python scripts/generate_results_figures.py [--output assets_dir]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_results_figures import run


def main():
    parser = argparse.ArgumentParser(description="Generate Results.md figures")
    parser.add_argument("--output", default="complexity_energetics/assets",
                       help="Output directory for figures")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
