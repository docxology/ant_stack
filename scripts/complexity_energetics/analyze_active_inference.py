#!/usr/bin/env python3
"""Active inference analysis orchestrator.

Thin orchestrator for active inference complexity analysis using ce.workloads methods.
Analyzes policy horizons, branching factors, and bounded rationality effects.

Usage:
    python scripts/complexity_energetics/analyze_active_inference.py --output NEW_DIRECTORY
"""

import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_active_inference import run


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Analyze active inference complexity and scaling")
    parser.add_argument("--output", required=True,
                       help="Explicit absent or empty output family directory")
    parser.add_argument("--max-horizon", type=int, default=15,
                       help="Maximum policy horizon to analyze")
    parser.add_argument("--branching", type=int, default=4,
                       help="Default branching factor")

    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
