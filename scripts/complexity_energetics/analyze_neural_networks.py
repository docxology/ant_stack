#!/usr/bin/env python3
"""Neural network analysis orchestrator.

Thin orchestrator for sparse neural network analysis using ce.workloads methods.
Analyzes different connectivity patterns and sparsity levels.

Usage:
    python scripts/complexity_energetics/analyze_neural_networks.py --output NEW_DIRECTORY
"""

import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.ce_neural_networks import run


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Analyze neural network patterns and scaling")
    parser.add_argument("--output", required=True,
                       help="Explicit absent or empty output family directory")
    parser.add_argument("--max-size", type=int, default=100000,
                       help="Maximum network size to analyze")

    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
