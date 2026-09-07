#!/usr/bin/env python3
"""Active inference analysis orchestrator.

Thin orchestrator for active inference complexity analysis using ce.workloads methods.
Analyzes policy horizons, branching factors, and bounded rationality effects.

Usage:
    python scripts/analyze_active_inference.py [--output output_dir] [--max-horizon 20]
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
    parser.add_argument("--output", default="active_inference_output",
                       help="Output directory for results")
    parser.add_argument("--max-horizon", type=int, default=15,
                       help="Maximum policy horizon to analyze")
    parser.add_argument("--branching", type=int, default=4,
                       help="Default branching factor")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
